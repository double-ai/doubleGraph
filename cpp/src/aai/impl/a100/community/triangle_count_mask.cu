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
#include <cub/cub.cuh>
#include <cstdint>
#include <cstddef>

namespace aai {

namespace {





struct Cache : Cacheable {
    int32_t* degree = nullptr;
    int64_t degree_cap = 0;

    int32_t* dodg_offsets = nullptr;
    int64_t dodg_offsets_cap = 0;

    int32_t* dodg_indices = nullptr;
    int64_t dodg_indices_cap = 0;

    int32_t* edge_src = nullptr;
    int64_t edge_src_cap = 0;

    void* cub_temp = nullptr;
    size_t cub_temp_cap = 0;

    int32_t* all_counts = nullptr;
    int64_t all_counts_cap = 0;

    void ensure(int32_t V, int32_t E) {
        if (degree_cap < V) {
            if (degree) cudaFree(degree);
            cudaMalloc(&degree, (size_t)V * sizeof(int32_t));
            degree_cap = V;
        }
        int64_t offsets_needed = (int64_t)V + 1;
        if (dodg_offsets_cap < offsets_needed) {
            if (dodg_offsets) cudaFree(dodg_offsets);
            cudaMalloc(&dodg_offsets, (size_t)offsets_needed * sizeof(int32_t));
            dodg_offsets_cap = offsets_needed;
        }
        if (dodg_indices_cap < E) {
            if (dodg_indices) cudaFree(dodg_indices);
            cudaMalloc(&dodg_indices, (size_t)E * sizeof(int32_t));
            dodg_indices_cap = E;
        }
        if (edge_src_cap < E) {
            if (edge_src) cudaFree(edge_src);
            cudaMalloc(&edge_src, (size_t)E * sizeof(int32_t));
            edge_src_cap = E;
        }
        if (all_counts_cap < V) {
            if (all_counts) cudaFree(all_counts);
            cudaMalloc(&all_counts, (size_t)V * sizeof(int32_t));
            all_counts_cap = V;
        }
    }

    void ensure_cub(size_t bytes) {
        if (cub_temp_cap < bytes) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, bytes);
            cub_temp_cap = bytes;
        }
    }

    ~Cache() override {
        if (degree) cudaFree(degree);
        if (dodg_offsets) cudaFree(dodg_offsets);
        if (dodg_indices) cudaFree(dodg_indices);
        if (edge_src) cudaFree(edge_src);
        if (cub_temp) cudaFree(cub_temp);
        if (all_counts) cudaFree(all_counts);
    }
};





__device__ __forceinline__ int32_t ld_i32(const int32_t* p) {
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}

__device__ __forceinline__ uint32_t ld_u32(const uint32_t* p) {
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}

__device__ __forceinline__ bool is_edge_active(const uint32_t* __restrict__ edge_mask, int e) {
  return (ld_u32(edge_mask + (e >> 5)) >> (e & 31)) & 1u;
}

__device__ __forceinline__ int lower_bound_range_dev(const int32_t* __restrict__ arr, int lo, int hi, int target) {
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    int v = ld_i32(arr + mid);
    if (v < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

__device__ __forceinline__ int warp_reduce_sum(int x) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffff, x, off);
  return x;
}





__global__ void compute_masked_degree_kernel(const int32_t* __restrict__ offsets,
                                             const uint32_t* __restrict__ edge_mask,
                                             int32_t* __restrict__ degree,
                                             int32_t num_vertices) {
  int v = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (v >= num_vertices) return;

  int start = ld_i32(offsets + v);
  int end = ld_i32(offsets + v + 1);
  if (start >= end) {
    degree[v] = 0;
    return;
  }

  int count = 0;
  int first_word = start >> 5;
  int last_word = (end - 1) >> 5;

  if (first_word == last_word) {
    uint32_t mask = ld_u32(edge_mask + first_word);
    int first_bit = start & 31;
    int num_bits = end - start;
    mask >>= first_bit;
    if (num_bits < 32) mask &= (1u << num_bits) - 1u;
    count = __popc(mask);
  } else {
    count += __popc(ld_u32(edge_mask + first_word) >> (start & 31));
    for (int w = first_word + 1; w < last_word; ++w) count += __popc(ld_u32(edge_mask + w));
    int last_bit = end & 31;
    if (last_bit == 0)
      count += __popc(ld_u32(edge_mask + last_word));
    else
      count += __popc(ld_u32(edge_mask + last_word) & ((1u << last_bit) - 1u));
  }

  degree[v] = count;
}





__global__ void compute_dodg_outdeg_warp_kernel(const int32_t* __restrict__ offsets,
                                                const int32_t* __restrict__ indices,
                                                const uint32_t* __restrict__ edge_mask,
                                                const int32_t* __restrict__ degree,
                                                int32_t* __restrict__ outdeg,
                                                int32_t num_vertices) {
  int warp_id = ((int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x) >> 5;
  int lane = (int)threadIdx.x & 31;
  if (warp_id >= num_vertices) return;

  int u = warp_id;
  int start = ld_i32(offsets + u);
  int end = ld_i32(offsets + u + 1);
  int du = ld_i32(degree + u);

  int cnt = 0;
  for (int e = start + lane; e < end; e += 32) {
    if (!is_edge_active(edge_mask, e)) continue;
    int v = ld_i32(indices + e);
    int dv = ld_i32(degree + v);
    if ((du < dv) || (du == dv && u < v)) cnt++;
  }
  cnt = warp_reduce_sum(cnt);
  if (lane == 0) outdeg[u] = cnt;
}





__global__ void fill_dodg_warp_kernel(const int32_t* __restrict__ offsets,
                                      const int32_t* __restrict__ indices,
                                      const uint32_t* __restrict__ edge_mask,
                                      const int32_t* __restrict__ degree,
                                      const int32_t* __restrict__ dodg_offsets,
                                      int32_t* __restrict__ dodg_indices,
                                      int32_t* __restrict__ edge_src,
                                      int32_t num_vertices) {
  int warp_id = ((int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x) >> 5;
  int lane = (int)threadIdx.x & 31;
  if (warp_id >= num_vertices) return;

  int u = warp_id;
  int start = ld_i32(offsets + u);
  int end = ld_i32(offsets + u + 1);
  int du = ld_i32(degree + u);

  int base = ld_i32(dodg_offsets + u);

  for (int e = start + lane; e < end; e += 32) {
    bool valid = false;
    int v = -1;
    if (is_edge_active(edge_mask, e)) {
      v = ld_i32(indices + e);
      int dv = ld_i32(degree + v);
      valid = (du < dv) || (du == dv && u < v);
    }

    unsigned m = __ballot_sync(0xffffffff, valid);
    int before = __popc(m & ((1u << lane) - 1u));
    int total = __popc(m);

    if (valid) {
      int pos = base + before;
      dodg_indices[pos] = v;
      edge_src[pos] = u;
    }

    base += total;
  }
}





__global__ void sort_segments_kernel(const int32_t* __restrict__ dodg_offsets,
                                     int32_t* __restrict__ dodg_indices,
                                     int32_t num_vertices) {
  int u = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (u >= num_vertices) return;

  int start = ld_i32(dodg_offsets + u);
  int end = ld_i32(dodg_offsets + u + 1);
  int len = end - start;
  if (len <= 1) return;

  for (int i = start + 1; i < end; ++i) {
    int key = dodg_indices[i];
    int j = i - 1;
    while (j >= start && dodg_indices[j] > key) {
      dodg_indices[j + 1] = dodg_indices[j];
      --j;
    }
    dodg_indices[j + 1] = key;
  }
}





template <int GROUP_SIZE>
__device__ __forceinline__ int group_reduce_sum(int x, unsigned mask) {
  #pragma unroll
  for (int off = GROUP_SIZE / 2; off > 0; off >>= 1) {
    x += __shfl_down_sync(mask, x, off, GROUP_SIZE);
  }
  return x;
}

template <int WARPS_PER_BLOCK, int GROUP_SIZE>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 2)
count_triangles_subwarp_kernel(const int32_t* __restrict__ dodg_offsets,
                               const int32_t* __restrict__ dodg_indices,
                               const int32_t* __restrict__ edge_src,
                               int32_t* __restrict__ counts,
                               int32_t num_vertices) {
  static_assert(32 % GROUP_SIZE == 0);
  constexpr int GROUPS_PER_WARP = 32 / GROUP_SIZE;

  int tid = (int)threadIdx.x;
  int lane = tid & 31;
  int warp_in_block = tid >> 5;
  int warp_global = ((int)blockIdx.x * WARPS_PER_BLOCK) + warp_in_block;

  int group_id = lane / GROUP_SIZE;
  int lane_in_group = lane & (GROUP_SIZE - 1);

  unsigned group_mask;
  if constexpr (GROUP_SIZE == 32) {
    group_mask = 0xffffffffu;
  } else {
    group_mask = ((1u << GROUP_SIZE) - 1u) << (group_id * GROUP_SIZE);
  }

  int total_edges = ld_i32(dodg_offsets + num_vertices);
  int grid_groups = (int)gridDim.x * WARPS_PER_BLOCK * GROUPS_PER_WARP;
  int group_global = warp_global * GROUPS_PER_WARP + group_id;

  for (int edge_id = group_global; edge_id < total_edges; edge_id += grid_groups) {
    int u = 0, v = 0;
    if (lane_in_group == 0) {
      u = ld_i32(edge_src + edge_id);
      v = ld_i32(dodg_indices + edge_id);
    }
    u = __shfl_sync(group_mask, u, 0, GROUP_SIZE);
    v = __shfl_sync(group_mask, v, 0, GROUP_SIZE);

    int u_start = ld_i32(dodg_offsets + u);
    int u_end = ld_i32(dodg_offsets + u + 1);
    int v_start = ld_i32(dodg_offsets + v);
    int v_end = ld_i32(dodg_offsets + v + 1);

    int u_len = u_end - u_start;
    int v_len = v_end - v_start;
    if (u_len == 0 || v_len == 0) continue;

    const int32_t* A; int a_len;
    const int32_t* B; int b_len;
    if (u_len <= v_len) {
      A = dodg_indices + u_start; a_len = u_len;
      B = dodg_indices + v_start; b_len = v_len;
    } else {
      A = dodg_indices + v_start; a_len = v_len;
      B = dodg_indices + u_start; b_len = u_len;
    }

    int b_min = ld_i32(B + 0);
    int b_max = ld_i32(B + (b_len - 1));

    int local = 0;
    int pos = 0;
    for (int i = lane_in_group; i < a_len; i += GROUP_SIZE) {
      int w = ld_i32(A + i);
      if (w < b_min) continue;
      if (w > b_max) break;
      pos = lower_bound_range_dev(B, pos, b_len, w);
      int key = -1;
      if (pos < b_len && ld_i32(B + pos) == w) {
        local++;
        key = w;
        pos++;
      }

      
      unsigned am = __activemask();
      unsigned eq = __match_any_sync(am, key);
      if (key != -1) {
        int leader = __ffs((int)eq) - 1;
        if (lane == leader) {
          atomicAdd(counts + key, __popc(eq));
        }
      }
    }

    int total = group_reduce_sum<GROUP_SIZE>(local, group_mask);
    if (lane_in_group == 0 && total) {
      atomicAdd(counts + u, total);
      atomicAdd(counts + v, total);
    }
  }
}





__global__ void gather_counts_kernel(const int32_t* __restrict__ all_counts,
                                     const int32_t* __restrict__ vertices,
                                     int32_t* __restrict__ out_counts,
                                     int32_t n_vertices) {
  int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (i >= n_vertices) return;
  out_counts[i] = ld_i32(all_counts + ld_i32(vertices + i));
}





size_t get_cub_temp_bytes(int n) {
    size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, (int32_t*)nullptr, (int32_t*)nullptr, n);
    return bytes;
}

void run_cub_exclusive_sum(void* temp, size_t temp_bytes, int32_t* data, int n,
                           cudaStream_t stream) {
    cub::DeviceScan::ExclusiveSum(temp, temp_bytes, data, data, n, stream);
}

}  

void triangle_count_mask(const graph32_t& graph,
                         int32_t* counts,
                         const int32_t* vertices,
                         std::size_t n_vertices) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t V = graph.number_of_vertices;
    const int32_t E = graph.number_of_edges;
    cudaStream_t stream = 0;

    if (V <= 0) {
        if (vertices && n_vertices > 0)
            cudaMemsetAsync(counts, 0, n_vertices * sizeof(int32_t), stream);
        return;
    }

    if (E <= 0) {
        std::size_t out_size = vertices ? n_vertices : (std::size_t)V;
        if (out_size > 0)
            cudaMemsetAsync(counts, 0, out_size * sizeof(int32_t), stream);
        return;
    }

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_mask = graph.edge_mask;

    cache.ensure(V, E);

    
    {
        int block = 256;
        int grid = (V + block - 1) / block;
        if (grid) compute_masked_degree_kernel<<<grid, block, 0, stream>>>(
            d_offsets, d_mask, cache.degree, V);
    }

    
    {
        int warps_per_block = 8;
        int threads = warps_per_block * 32;
        int grid = (V + warps_per_block - 1) / warps_per_block;
        if (grid) compute_dodg_outdeg_warp_kernel<<<grid, threads, 0, stream>>>(
            d_offsets, d_indices, d_mask, cache.degree, cache.dodg_offsets, V);
    }
    cudaMemsetAsync(cache.dodg_offsets + V, 0, sizeof(int32_t), stream);

    
    int scan_n = V + 1;
    size_t temp_bytes = get_cub_temp_bytes(scan_n);
    cache.ensure_cub(temp_bytes);
    run_cub_exclusive_sum(cache.cub_temp, temp_bytes, cache.dodg_offsets, scan_n, stream);

    
    {
        int warps_per_block = 8;
        int threads = warps_per_block * 32;
        int grid = (V + warps_per_block - 1) / warps_per_block;
        if (grid) fill_dodg_warp_kernel<<<grid, threads, 0, stream>>>(
            d_offsets, d_indices, d_mask, cache.degree, cache.dodg_offsets,
            cache.dodg_indices, cache.edge_src, V);
    }

    
    {
        int block = 256;
        int grid = (V + block - 1) / block;
        if (grid) sort_segments_kernel<<<grid, block, 0, stream>>>(
            cache.dodg_offsets, cache.dodg_indices, V);
    }

    
    int32_t* count_buf;
    if (vertices) {
        count_buf = cache.all_counts;
    } else {
        count_buf = counts;
    }
    cudaMemsetAsync(count_buf, 0, (size_t)V * sizeof(int32_t), stream);

    float avg_deg = static_cast<float>(E) / static_cast<float>(V);
    constexpr int WARPS = 8;
    int blocks = 108 * 8;
    if (avg_deg <= 16.0f) {
        count_triangles_subwarp_kernel<WARPS, 8><<<blocks, WARPS * 32, 0, stream>>>(
            cache.dodg_offsets, cache.dodg_indices, cache.edge_src, count_buf, V);
    } else if (avg_deg <= 64.0f) {
        count_triangles_subwarp_kernel<WARPS, 16><<<blocks, WARPS * 32, 0, stream>>>(
            cache.dodg_offsets, cache.dodg_indices, cache.edge_src, count_buf, V);
    } else {
        count_triangles_subwarp_kernel<WARPS, 32><<<blocks, WARPS * 32, 0, stream>>>(
            cache.dodg_offsets, cache.dodg_indices, cache.edge_src, count_buf, V);
    }

    
    if (vertices) {
        int block = 256;
        int grid = ((int)n_vertices + block - 1) / block;
        if (grid) gather_counts_kernel<<<grid, block, 0, stream>>>(
            count_buf, vertices, counts, (int32_t)n_vertices);
    }
}

}  
