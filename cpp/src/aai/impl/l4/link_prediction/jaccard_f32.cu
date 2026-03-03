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
#include <cfloat>
#include <cub/cub.cuh>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* vertex_sums = nullptr;
    int32_t* sorted_first = nullptr;
    int32_t* pair_indices_in = nullptr;
    int32_t* pair_indices_out = nullptr;
    void* sort_temp = nullptr;

    int64_t vertex_sums_capacity = 0;
    int64_t sorted_first_capacity = 0;
    int64_t pair_indices_in_capacity = 0;
    int64_t pair_indices_out_capacity = 0;
    size_t sort_temp_capacity = 0;

    void ensure(int32_t num_vertices, int64_t num_pairs, size_t sort_temp_bytes) {
        if (vertex_sums_capacity < num_vertices) {
            if (vertex_sums) cudaFree(vertex_sums);
            cudaMalloc(&vertex_sums, (size_t)num_vertices * sizeof(float));
            vertex_sums_capacity = num_vertices;
        }
        if (sorted_first_capacity < num_pairs) {
            if (sorted_first) cudaFree(sorted_first);
            cudaMalloc(&sorted_first, (size_t)num_pairs * sizeof(int32_t));
            sorted_first_capacity = num_pairs;
        }
        if (pair_indices_in_capacity < num_pairs) {
            if (pair_indices_in) cudaFree(pair_indices_in);
            cudaMalloc(&pair_indices_in, (size_t)num_pairs * sizeof(int32_t));
            pair_indices_in_capacity = num_pairs;
        }
        if (pair_indices_out_capacity < num_pairs) {
            if (pair_indices_out) cudaFree(pair_indices_out);
            cudaMalloc(&pair_indices_out, (size_t)num_pairs * sizeof(int32_t));
            pair_indices_out_capacity = num_pairs;
        }
        if (sort_temp_capacity < sort_temp_bytes) {
            if (sort_temp) cudaFree(sort_temp);
            cudaMalloc(&sort_temp, sort_temp_bytes);
            sort_temp_capacity = sort_temp_bytes;
        }
    }

    ~Cache() override {
        if (vertex_sums) cudaFree(vertex_sums);
        if (sorted_first) cudaFree(sorted_first);
        if (pair_indices_in) cudaFree(pair_indices_in);
        if (pair_indices_out) cudaFree(pair_indices_out);
        if (sort_temp) cudaFree(sort_temp);
    }
};

static __device__ __forceinline__ int32_t ld_i32(const int32_t* p)
{
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}

static __device__ __forceinline__ float ld_f32(const float* p)
{
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}


static __device__ __forceinline__ float sum_range_f32(const float* p, const float* end)
{
  float sum = 0.0f;
  int n     = (int)(end - p);

  if (n <= 8) {
    switch (n) {
      case 8: sum += ld_f32(p + 7); [[fallthrough]];
      case 7: sum += ld_f32(p + 6); [[fallthrough]];
      case 6: sum += ld_f32(p + 5); [[fallthrough]];
      case 5: sum += ld_f32(p + 4); [[fallthrough]];
      case 4: sum += ld_f32(p + 3); [[fallthrough]];
      case 3: sum += ld_f32(p + 2); [[fallthrough]];
      case 2: sum += ld_f32(p + 1); [[fallthrough]];
      case 1: sum += ld_f32(p + 0); [[fallthrough]];
      default: break;
    }
    return sum;
  }

  
  uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  while ((p < end) && (addr & 0xF)) {
    sum += ld_f32(p);
    ++p;
    addr += sizeof(float);
  }

  while (p + 4 <= end) {
    float4 v = __ldg(reinterpret_cast<const float4*>(p));
    sum += v.x + v.y + v.z + v.w;
    p += 4;
  }

  while (p < end) {
    sum += ld_f32(p);
    ++p;
  }

  return sum;
}

__global__ __launch_bounds__(256, 2) void vertex_weight_sums_kernel(
  const int32_t* __restrict__ offsets,
  const float* __restrict__ weights,
  float* __restrict__ vertex_sums,
  int32_t num_vertices)
{
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int32_t v = (int32_t)blockIdx.x * 8 + warp;
  if (v >= num_vertices) return;

  int32_t start = ld_i32(offsets + v);
  int32_t end   = ld_i32(offsets + v + 1);

  float sum = 0.0f;
  for (int32_t i = start + lane; i < end; i += 32) {
    sum += ld_f32(weights + i);
  }

  sum += __shfl_down_sync(0xffffffff, sum, 16);
  sum += __shfl_down_sync(0xffffffff, sum, 8);
  sum += __shfl_down_sync(0xffffffff, sum, 4);
  sum += __shfl_down_sync(0xffffffff, sum, 2);
  sum += __shfl_down_sync(0xffffffff, sum, 1);

  if (lane == 0) vertex_sums[v] = sum;
}

__global__ void iota_kernel(int32_t* arr, int64_t n)
{
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) arr[i] = (int32_t)i;
}

__global__ __launch_bounds__(512, 1) void jaccard_sorted_kernel(
  const int32_t* __restrict__ offsets,
  const int32_t* __restrict__ indices,
  const float* __restrict__ weights,
  const float* __restrict__ vertex_sums,
  const int32_t* __restrict__ sorted_first,
  const int32_t* __restrict__ second,
  const int32_t* __restrict__ pair_indices,
  int64_t num_pairs,
  float* __restrict__ out)
{
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_pairs) return;

  int32_t orig = ld_i32(pair_indices + tid);
  int32_t u    = ld_i32(sorted_first + tid);
  int32_t v    = ld_i32(second + orig);

  float sum_u = ld_f32(vertex_sums + u);
  if (u == v) {
    out[orig] = (sum_u <= FLT_MIN) ? 0.0f : 1.0f;
    return;
  }
  float sum_v = ld_f32(vertex_sums + v);
  if ((sum_u <= FLT_MIN) || (sum_v <= FLT_MIN)) {
    out[orig] = 0.0f;
    return;
  }

  int32_t u0 = ld_i32(offsets + u);
  int32_t u1 = ld_i32(offsets + u + 1);
  int32_t v0 = ld_i32(offsets + v);
  int32_t v1 = ld_i32(offsets + v + 1);

  int32_t u_deg = u1 - u0;
  int32_t v_deg = v1 - v0;
  if ((u_deg <= 0) || (v_deg <= 0)) {
    out[orig] = 0.0f;
    return;
  }

  const int32_t* u_idx = indices + u0;
  const int32_t* v_idx = indices + v0;
  const float* u_w = weights + u0;
  const float* v_w = weights + v0;

  const int32_t* a_idx;
  const float* a_w;
  int32_t a_deg;
  const int32_t* b_idx;
  const float* b_w;
  int32_t b_deg;

  if (u_deg <= v_deg) {
    a_idx = u_idx;
    a_w   = u_w;
    a_deg = u_deg;
    b_idx = v_idx;
    b_w   = v_w;
    b_deg = v_deg;
  } else {
    a_idx = v_idx;
    a_w   = v_w;
    a_deg = v_deg;
    b_idx = u_idx;
    b_w   = u_w;
    b_deg = u_deg;
  }

  float inter = 0.0f;

  if (b_deg > (a_deg << 3)) {
    int32_t j = 0;
    for (int32_t i = 0; i < a_deg && j < b_deg; ++i) {
      int32_t target = ld_i32(a_idx + i);
      float aw       = ld_f32(a_w + i);

      int32_t bj = ld_i32(b_idx + j);
      if (bj < target) {
        int32_t step = 1;
        int32_t pos  = j;
        while ((pos + step) < b_deg && ld_i32(b_idx + (pos + step)) < target) {
          pos += step;
          step <<= 1;
        }
        int32_t lo = pos + 1;
        int32_t hi = (pos + step + 1 < b_deg) ? (pos + step + 1) : b_deg;
        while (lo < hi) {
          int32_t mid = (lo + hi) >> 1;
          if (ld_i32(b_idx + mid) < target)
            lo = mid + 1;
          else
            hi = mid;
        }
        j = lo;
        if (j >= b_deg) break;
        bj = ld_i32(b_idx + j);
      }

      if (bj == target) {
        float bw = ld_f32(b_w + j);
        inter += (aw < bw) ? aw : bw;
        ++j;
      }
    }
  } else {
    const int32_t* ia = a_idx;
    const int32_t* ib = b_idx;
    const int32_t* ia_end = a_idx + a_deg;
    const int32_t* ib_end = b_idx + b_deg;
    const float* wa = a_w;
    const float* wb = b_w;

    int32_t ka = ld_i32(ia);
    int32_t kb = ld_i32(ib);

    while (true) {
      if (ka < kb) {
        ++ia;
        ++wa;
        if (ia >= ia_end) break;
        ka = ld_i32(ia);
      } else if (ka > kb) {
        ++ib;
        ++wb;
        if (ib >= ib_end) break;
        kb = ld_i32(ib);
      } else {
        float aw = ld_f32(wa);
        float bw = ld_f32(wb);
        inter += (aw < bw) ? aw : bw;
        ++ia;
        ++ib;
        ++wa;
        ++wb;
        if ((ia >= ia_end) || (ib >= ib_end)) break;
        ka = ld_i32(ia);
        kb = ld_i32(ib);
      }
    }
  }

  float denom = sum_u + sum_v - inter;
  out[orig]   = (denom <= FLT_MIN) ? 0.0f : (inter / denom);
}

}  

void jaccard_similarity(const graph32_t& graph,
                        const float* edge_weights,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;

    auto stream = cudaStreamPerThread;

    int num_bits = 1;
    int tmp = num_vertices;
    while (tmp > 1) {
        ++num_bits;
        tmp >>= 1;
    }

    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr,
                                    sort_temp_bytes,
                                    (const int32_t*)nullptr,
                                    (int32_t*)nullptr,
                                    (const int32_t*)nullptr,
                                    (int32_t*)nullptr,
                                    (int)num_pairs,
                                    0,
                                    num_bits);

    cache.ensure(num_vertices, (int64_t)num_pairs, sort_temp_bytes);

    {
        constexpr int block = 256;
        int grid = (num_vertices + 8 - 1) / 8;
        vertex_weight_sums_kernel<<<grid, block, 0, stream>>>(
            offsets, edge_weights, cache.vertex_sums, num_vertices);
    }

    {
        constexpr int block = 256;
        int grid = (int)(((int64_t)num_pairs + block - 1) / block);
        iota_kernel<<<grid, block, 0, stream>>>(cache.pair_indices_in, (int64_t)num_pairs);
    }

    cub::DeviceRadixSort::SortPairs(cache.sort_temp,
                                    sort_temp_bytes,
                                    vertex_pairs_first,
                                    cache.sorted_first,
                                    cache.pair_indices_in,
                                    cache.pair_indices_out,
                                    (int)num_pairs,
                                    0,
                                    num_bits,
                                    stream);

    {
        int grid_j = (int)(((int64_t)num_pairs + 512 - 1) / 512);
        jaccard_sorted_kernel<<<grid_j, 512, 0, stream>>>(
            offsets,
            indices,
            edge_weights,
            cache.vertex_sums,
            cache.sorted_first,
            vertex_pairs_second,
            cache.pair_indices_out,
            (int64_t)num_pairs,
            similarity_scores);
    }
}

}  
