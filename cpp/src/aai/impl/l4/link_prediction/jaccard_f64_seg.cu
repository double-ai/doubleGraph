/*
 * Copyright (c) 2025, AA-I Technologies Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cugraph/aai/algorithms.hpp>
#include <cuda_runtime.h>
#include <cstdint>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* vertex_sums = nullptr;
    int32_t vertex_sums_capacity = 0;

    void ensure(int32_t size) {
        if (vertex_sums_capacity < size) {
            if (vertex_sums) cudaFree(vertex_sums);
            cudaMalloc(&vertex_sums, static_cast<size_t>(size) * sizeof(float));
            vertex_sums_capacity = size;
        }
    }

    ~Cache() override {
        if (vertex_sums) cudaFree(vertex_sums);
    }
};





__global__ void vertex_sums_large_kernel(const int32_t* __restrict__ offsets,
                                        const double* __restrict__ w,
                                        float* __restrict__ out,
                                        int32_t v_start,
                                        int32_t v_end)
{
  
  int32_t v = v_start + static_cast<int32_t>(blockIdx.x);
  if (v >= v_end) return;

  int32_t e0 = offsets[v];
  int32_t e1 = offsets[v + 1];
  
  float sum = 0.0f;
  int32_t start = e0;
  int32_t end = e1;
  if (start & 1) {
    if (threadIdx.x == 0) { sum += static_cast<float>(w[start]); }
    start += 1;
  }
  
  const double2* w2 = reinterpret_cast<const double2*>(w + start);
  int32_t vec_end = (end - start) >> 1;
  for (int32_t ve = threadIdx.x; ve < vec_end; ve += blockDim.x) {
    double2 d = w2[ve];
    sum += static_cast<float>(d.x) + static_cast<float>(d.y);
  }
  
  if (((end - start) & 1) && threadIdx.x == 0) {
    sum += static_cast<float>(w[end - 1]);
  }

  
  
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  __shared__ float warp_sums[8];
  if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = sum;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v_sum = (threadIdx.x < (blockDim.x >> 5)) ? warp_sums[threadIdx.x] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      v_sum += __shfl_down_sync(0xffffffff, v_sum, offset);
    }
    if (threadIdx.x == 0) out[v] = v_sum;
  }
}

__global__ void vertex_sums_mid_kernel(const int32_t* __restrict__ offsets,
                                      const double* __restrict__ w,
                                      float* __restrict__ out,
                                      int32_t v_start,
                                      int32_t v_end)
{
  
  int32_t warp_id = static_cast<int32_t>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int32_t lane = threadIdx.x & 31;
  int32_t v = v_start + warp_id;
  if (v >= v_end) return;

  int32_t e0 = offsets[v];
  int32_t e1 = offsets[v + 1];
  
  float sum = 0.0f;
  int32_t start = e0;
  int32_t end = e1;
  int32_t odd = start & 1;
  if (odd && lane == 0) { sum += static_cast<float>(w[start]); }
  start += odd;
  const double2* w2 = reinterpret_cast<const double2*>(w + start);
  int32_t vec_end = (end - start) >> 1;
  for (int32_t ve = lane; ve < vec_end; ve += 32) {
    double2 d = w2[ve];
    sum += static_cast<float>(d.x) + static_cast<float>(d.y);
  }
  
  if (((end - start) & 1) && lane == 0) {
    sum += static_cast<float>(w[end - 1]);
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  if (lane == 0) out[v] = sum;
}

__global__ void vertex_sums_small_kernel(const int32_t* __restrict__ offsets,
                                        const double* __restrict__ w,
                                        float* __restrict__ out,
                                        int32_t v_start,
                                        int32_t v_end)
{
  int32_t v = v_start + static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (v >= v_end) return;
  int32_t e0 = offsets[v];
  int32_t e1 = offsets[v + 1];
  float sum = 0.0f;
  for (int32_t e = e0; e < e1; ++e) sum += static_cast<float>(w[e]);
  out[v] = sum;
}

__global__ void vertex_sums_zero_kernel(float* __restrict__ out, int32_t v_start, int32_t v_end)
{
  int32_t v = v_start + static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (v >= v_end) return;
  out[v] = 0.0;
}





__device__ __forceinline__ int32_t lower_bound_device(const int32_t* __restrict__ arr,
                                                      int32_t lo,
                                                      int32_t hi,
                                                      int32_t target)
{
  while (lo < hi) {
    int32_t mid = lo + ((hi - lo) >> 1);
    int32_t v = arr[mid];
    if (v < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__global__ void jaccard_galloping_kernel(const int32_t* __restrict__ offsets,
                                        const int32_t* __restrict__ indices,
                                        const double* __restrict__ w,
                                        const float* __restrict__ vertex_sums,
                                        const int32_t* __restrict__ first,
                                        const int32_t* __restrict__ second,
                                        double* __restrict__ out,
                                        int64_t num_pairs,
                                        int32_t active_vertices)
{
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

  for (int64_t p = tid; p < num_pairs; p += stride) {
    int32_t u = first[p];
    int32_t v = second[p];

    int32_t u0 = offsets[u];
    int32_t u1 = offsets[u + 1];
    int32_t v0 = offsets[v];
    int32_t v1 = offsets[v + 1];

    int32_t du = u1 - u0;
    int32_t dv = v1 - v0;
    if (du == 0 || dv == 0) {
      out[p] = 0.0;
      continue;
    }

    
    int32_t u_min = indices[u0];
    int32_t u_max = indices[u1 - 1];
    int32_t v_min = indices[v0];
    int32_t v_max = indices[v1 - 1];
    if (u_max < v_min || v_max < u_min) {
      out[p] = 0.0;
      continue;
    }

    float inter = 0.0f;

    
    if (du < dv) {
      const int32_t* s_idx = indices + u0;
      const double* s_w = w + u0;
      int32_t s_n = du;
      const int32_t* l_idx = indices + v0;
      const double* l_w = w + v0;
      int32_t l_n = dv;

      int32_t l_pos = 0;
      if (l_idx[0] < s_idx[0]) {
        l_pos = lower_bound_device(l_idx, 0, l_n, s_idx[0]);
      }

      if (l_n > 32) {
        
        for (int32_t si = 0; si < s_n; ++si) {
          int32_t target = s_idx[si];
          if (l_pos >= l_n) break;
          int32_t cur = l_idx[l_pos];
          if (cur < target) {
            int32_t step = 1;
            int32_t bound = l_pos;
            while (bound + step < l_n && l_idx[bound + step] < target) {
              bound += step;
              step <<= 1;
            }
            int32_t hi = (bound + step + 1 < l_n) ? (bound + step + 1) : l_n;
            l_pos = lower_bound_device(l_idx, bound, hi, target);
          }
          if (l_pos < l_n && l_idx[l_pos] == target) {
            float wu = static_cast<float>(s_w[si]);
            float wv2 = static_cast<float>(l_w[l_pos]);
            inter += (wu < wv2) ? wu : wv2;
            ++l_pos;
          }
        }
      } else {
        
        int32_t i = 0;
        int32_t j = l_pos;
        while (i < s_n && j < l_n) {
          int32_t a = s_idx[i];
          int32_t b = l_idx[j];
          if (a == b) {
            float wu = static_cast<float>(s_w[i]);
            float wv2 = static_cast<float>(l_w[j]);
            inter += (wu < wv2) ? wu : wv2;
            ++i;
            ++j;
          } else if (a < b) {
            ++i;
          } else {
            ++j;
          }
        }
      }
    } else {
      
      const int32_t* s_idx = indices + v0;
      const double* s_w = w + v0;
      int32_t s_n = dv;
      const int32_t* l_idx = indices + u0;
      const double* l_w = w + u0;
      int32_t l_n = du;

      int32_t l_pos = 0;
      if (l_idx[0] < s_idx[0]) {
        l_pos = lower_bound_device(l_idx, 0, l_n, s_idx[0]);
      }

      if (l_n > 32) {
        for (int32_t si = 0; si < s_n; ++si) {
          int32_t target = s_idx[si];
          if (l_pos >= l_n) break;
          int32_t cur = l_idx[l_pos];
          if (cur < target) {
            int32_t step = 1;
            int32_t bound = l_pos;
            while (bound + step < l_n && l_idx[bound + step] < target) {
              bound += step;
              step <<= 1;
            }
            int32_t hi = (bound + step + 1 < l_n) ? (bound + step + 1) : l_n;
            l_pos = lower_bound_device(l_idx, bound, hi, target);
          }
          if (l_pos < l_n && l_idx[l_pos] == target) {
            float wu = static_cast<float>(s_w[si]);
            float wv2 = static_cast<float>(l_w[l_pos]);
            inter += (wu < wv2) ? wu : wv2;
            ++l_pos;
          }
        }
      } else {
        int32_t i = 0;
        int32_t j = l_pos;
        while (i < s_n && j < l_n) {
          int32_t a = s_idx[i];
          int32_t b = l_idx[j];
          if (a == b) {
            float wu = static_cast<float>(s_w[i]);
            float wv2 = static_cast<float>(l_w[j]);
            inter += (wu < wv2) ? wu : wv2;
            ++i;
            ++j;
          } else if (a < b) {
            ++i;
          } else {
            ++j;
          }
        }
      }
    }

    
    
    float su;
    if (u < active_vertices) {
      su = vertex_sums[u];
    } else {
      su = 0.0f;
      for (int32_t e = u0; e < u1; ++e) su += static_cast<float>(w[e]);
    }
    float sv;
    if (v < active_vertices) {
      sv = vertex_sums[v];
    } else {
      sv = 0.0f;
      for (int32_t e = v0; e < v1; ++e) sv += static_cast<float>(w[e]);
    }

    float uni = su + sv - inter;
    out[p] = (uni > 0.0f) ? static_cast<double>(inter / uni) : 0.0;
  }
}





void launch_vertex_weight_sums_binned(const int32_t* offsets,
                                      const double* edge_weights,
                                      float* vertex_sums,
                                      int32_t num_vertices,
                                      const int32_t* segment_offsets_host,
                                      cudaStream_t stream)
{
  
  int32_t active = num_vertices;
  int32_t s1 = segment_offsets_host[1];
  int32_t s2 = segment_offsets_host[2];
  int32_t s3 = active;

  
  if (s1 > active) s1 = active;
  if (s2 > active) s2 = active;
  if (s1 > s2) s1 = s2;

  
  if (s1 > 0) {
    vertex_sums_large_kernel<<<s1, 256, 0, stream>>>(offsets, edge_weights, vertex_sums, 0, s1);
  }

  
  if (s2 > s1) {
    int32_t n = s2 - s1;
    int32_t warps_per_block = 8;
    int32_t grid = (n + warps_per_block - 1) / warps_per_block;
    vertex_sums_mid_kernel<<<grid, 256, 0, stream>>>(offsets, edge_weights, vertex_sums, s1, s2);
  }

  
  if (s3 > s2) {
    int32_t n = s3 - s2;
    int32_t block = 256;
    int32_t grid = (n + block - 1) / block;
    vertex_sums_small_kernel<<<grid, block, 0, stream>>>(offsets, edge_weights, vertex_sums, s2, s3);
  }
}

void launch_jaccard_galloping(const int32_t* offsets,
                              const int32_t* indices,
                              const double* edge_weights,
                              const float* vertex_sums,
                              const int32_t* pairs_first,
                              const int32_t* pairs_second,
                              double* scores,
                              int64_t num_pairs,
                              int32_t active_vertices,
                              cudaStream_t stream)
{
  constexpr int block = 512;
  
  int64_t grid64 = (num_pairs + block - 1) / block;
  int grid = static_cast<int>(grid64 > 2147483647LL ? 2147483647LL : grid64);
  jaccard_galloping_kernel<<<grid, block, 0, stream>>>(
    offsets, indices, edge_weights, vertex_sums, pairs_first, pairs_second, scores, num_pairs, active_vertices);
}

}  

void jaccard_similarity_seg(const graph32_t& graph,
                            const double* edge_weights,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            double* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;

    
    const auto& seg = graph.segment_offsets.value();
    int32_t seg_arr[5] = {seg[0], seg[1], seg[2], seg[3], seg[4]};
    int32_t active_vertices = seg_arr[3];  
    if (active_vertices < 0) active_vertices = 0;
    if (active_vertices > num_vertices) active_vertices = num_vertices;

    
    int32_t sums_size = (active_vertices == 0) ? 1 : active_vertices;
    cache.ensure(sums_size);

    auto stream = cudaStreamPerThread;

    launch_vertex_weight_sums_binned(graph.offsets,
                                     edge_weights,
                                     cache.vertex_sums,
                                     active_vertices,
                                     seg_arr,
                                     stream);

    launch_jaccard_galloping(graph.offsets,
                             graph.indices,
                             edge_weights,
                             cache.vertex_sums,
                             vertex_pairs_first,
                             vertex_pairs_second,
                             similarity_scores,
                             static_cast<int64_t>(num_pairs),
                             active_vertices,
                             stream);
}

}  
