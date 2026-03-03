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

namespace aai {

namespace {



__device__ __forceinline__ int lower_bound_int(const int32_t* __restrict__ a, int lo, int hi, int32_t target) {
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    int32_t v = a[mid];
    if (v < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

__device__ __forceinline__ bool bit_is_set(const uint32_t* __restrict__ mask, int32_t idx) {
  return (mask[idx >> 5] >> (idx & 31)) & 1u;
}



__global__ void compute_src_kernel(const int32_t* __restrict__ offsets, int32_t* __restrict__ src, int32_t n) {
  int u = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (u >= n) return;
  int32_t start = offsets[u];
  int32_t end = offsets[u + 1];
  for (int32_t e = start; e < end; ++e) src[e] = u;
}

__global__ void compute_rev_kernel(const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
                                  const int32_t* __restrict__ src, int32_t* __restrict__ rev, int32_t n_edges) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= n_edges) return;
  int32_t u = src[e];
  int32_t v = indices[e];
  int32_t lo = offsets[v];
  int32_t hi = offsets[v + 1];
  int32_t pos = lower_bound_int(indices, lo, hi, u);
  rev[e] = (pos < hi && indices[pos] == u) ? pos : -1;
}

__global__ void clear_self_loops_kernel(const int32_t* __restrict__ src, const int32_t* __restrict__ dst,
                                       uint32_t* __restrict__ active_mask, int32_t num_edges) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= num_edges) return;
  if (src[e] == dst[e]) {
    atomicAnd(&active_mask[e >> 5], ~(1u << (e & 31)));
  }
}

__global__ void count_support_capped_dirty_owner_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices, const int32_t* __restrict__ src,
    const int32_t* __restrict__ rev,
    const uint32_t* __restrict__ active_mask, const uint8_t* __restrict__ dirty_prev,
    int32_t* __restrict__ support_capped, int32_t threshold, int32_t num_edges) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= num_edges) return;

  int32_t re = rev[e];
  if (re < 0 || e > re) return;  

  if (!bit_is_set(active_mask, e)) return;  

  int32_t u = src[e];
  int32_t v = indices[e];

  if (!dirty_prev[u] && !dirty_prev[v]) return;

  int32_t u0 = offsets[u], u1 = offsets[u + 1];
  int32_t v0 = offsets[v], v1 = offsets[v + 1];
  int32_t du = u1 - u0;
  int32_t dv = v1 - v0;
  if (du == 0 || dv == 0) {
    support_capped[e] = 0;
    return;
  }

  int32_t s0, s1, l0, l1;
  if (du <= dv) {
    s0 = u0; s1 = u1;
    l0 = v0; l1 = v1;
  } else {
    s0 = v0; s1 = v1;
    l0 = u0; l1 = u1;
  }

  
  int32_t l_first = indices[l0];
  int32_t l_last = indices[l1 - 1];
  int32_t s_first = indices[s0];
  int32_t s_last = indices[s1 - 1];
  if (s_last < l_first || s_first > l_last) {
    support_capped[e] = 0;
    return;
  }

  int32_t i = s0;
  if (s_first < l_first) i = lower_bound_int(indices, s0, s1, l_first);
  int32_t s_end = s1;
  if (s_last > l_last && l_last != INT32_MAX) {
    s_end = lower_bound_int(indices, i, s1, l_last + 1);
  }

  int32_t count = 0;

  
  int32_t word_idx = i >> 5;
  uint32_t word = active_mask[word_idx];

  for (int32_t p = i; p < s_end; ++p) {
    int32_t wi = p >> 5;
    if (wi != word_idx) {
      word_idx = wi;
      word = active_mask[word_idx];
    }
    if (!((word >> (p & 31)) & 1u)) continue;

    int32_t w = indices[p];
    int32_t pos = lower_bound_int(indices, l0, l1, w);
    if (pos < l1 && indices[pos] == w) {
      uint32_t w2 = active_mask[pos >> 5];
      if ((w2 >> (pos & 31)) & 1u) {
        ++count;
        if (count >= threshold) {
          support_capped[e] = threshold;
          return;
        }
      }
    }
  }

  support_capped[e] = count;
}

__global__ void remove_by_support_owner_kernel(
    const int32_t* __restrict__ src, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ rev,
    uint32_t* __restrict__ active_mask, const int32_t* __restrict__ support_capped,
    const uint8_t* __restrict__ dirty_prev, uint8_t* __restrict__ dirty_curr,
    int32_t threshold, int32_t num_edges, int32_t* __restrict__ changed_flag) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= num_edges) return;

  int32_t re = rev[e];
  if (re < 0 || e > re) return;

  if (!bit_is_set(active_mask, e)) return;

  int32_t u = src[e];
  int32_t v = indices[e];
  if (!dirty_prev[u] && !dirty_prev[v]) return;

  if (support_capped[e] < threshold) {
    
    atomicAnd(&active_mask[e >> 5], ~(1u << (e & 31)));
    atomicAnd(&active_mask[re >> 5], ~(1u << (re & 31)));
    dirty_curr[u] = 1;
    dirty_curr[v] = 1;
    atomicOr(changed_flag, 1);
  }
}

__global__ void mask_to_flags_kernel(const uint32_t* __restrict__ active_mask, int32_t* __restrict__ flags,
                                    int32_t num_edges) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= num_edges) return;
  flags[e] = bit_is_set(active_mask, e) ? 1 : 0;
}

__global__ void extract_edges_kernel(const int32_t* __restrict__ src, const int32_t* __restrict__ indices,
                                    const uint32_t* __restrict__ active_mask, const int32_t* __restrict__ prefix,
                                    int32_t* __restrict__ out_src, int32_t* __restrict__ out_dst, int32_t num_edges) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= num_edges) return;
  if (!bit_is_set(active_mask, e)) return;
  int32_t pos = prefix[e];
  out_src[pos] = src[e];
  out_dst[pos] = indices[e];
}



struct Cache : Cacheable {
  
  int32_t* src = nullptr;
  int32_t* rev = nullptr;
  int32_t* support = nullptr;
  int32_t* flags = nullptr;
  int32_t* prefix = nullptr;
  int32_t edge_capacity = 0;

  
  uint32_t* active_mask = nullptr;
  int32_t mask_capacity = 0;

  
  uint8_t* dirty_a = nullptr;
  uint8_t* dirty_b = nullptr;
  int32_t vertex_capacity = 0;

  
  int32_t* changed_flag = nullptr;

  
  void* scan_temp = nullptr;
  size_t scan_temp_capacity = 0;

  void ensure(int32_t num_edges, int32_t num_vertices) {
    int32_t mask_words = (num_edges + 31) / 32;

    if (edge_capacity < num_edges) {
      if (src) cudaFree(src);
      if (rev) cudaFree(rev);
      if (support) cudaFree(support);
      if (flags) cudaFree(flags);
      if (prefix) cudaFree(prefix);
      cudaMalloc(&src, (size_t)num_edges * sizeof(int32_t));
      cudaMalloc(&rev, (size_t)num_edges * sizeof(int32_t));
      cudaMalloc(&support, (size_t)num_edges * sizeof(int32_t));
      cudaMalloc(&flags, (size_t)num_edges * sizeof(int32_t));
      cudaMalloc(&prefix, (size_t)num_edges * sizeof(int32_t));
      edge_capacity = num_edges;
    }

    if (mask_capacity < mask_words) {
      if (active_mask) cudaFree(active_mask);
      cudaMalloc(&active_mask, (size_t)mask_words * sizeof(uint32_t));
      mask_capacity = mask_words;
    }

    if (vertex_capacity < num_vertices) {
      if (dirty_a) cudaFree(dirty_a);
      if (dirty_b) cudaFree(dirty_b);
      cudaMalloc(&dirty_a, (size_t)num_vertices * sizeof(uint8_t));
      cudaMalloc(&dirty_b, (size_t)num_vertices * sizeof(uint8_t));
      vertex_capacity = num_vertices;
    }

    if (!changed_flag) {
      cudaMalloc(&changed_flag, sizeof(int32_t));
    }

    size_t needed = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, needed, (const int32_t*)nullptr, (int32_t*)nullptr, num_edges);
    if (scan_temp_capacity < needed) {
      if (scan_temp) cudaFree(scan_temp);
      cudaMalloc(&scan_temp, needed);
      scan_temp_capacity = needed;
    }
  }

  ~Cache() override {
    if (src) cudaFree(src);
    if (rev) cudaFree(rev);
    if (support) cudaFree(support);
    if (flags) cudaFree(flags);
    if (prefix) cudaFree(prefix);
    if (active_mask) cudaFree(active_mask);
    if (dirty_a) cudaFree(dirty_a);
    if (dirty_b) cudaFree(dirty_b);
    if (changed_flag) cudaFree(changed_flag);
    if (scan_temp) cudaFree(scan_temp);
  }
};

}  

k_truss_result_t k_truss_seg_mask(const graph32_t& graph,
                                  int32_t k) {
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);

  const int32_t num_vertices = graph.number_of_vertices;
  const int32_t num_edges = graph.number_of_edges;
  const int32_t* d_offsets = graph.offsets;
  const int32_t* d_indices = graph.indices;
  const uint32_t* d_edge_mask = graph.edge_mask;
  const int32_t threshold = k - 2;

  if (num_edges <= 0) {
    return k_truss_result_t{nullptr, nullptr, 0};
  }

  cache.ensure(num_edges, num_vertices);

  const int32_t mask_words = (num_edges + 31) / 32;
  int block = 256;

  
  int grid_v = (num_vertices + block - 1) / block;
  int grid_e = (num_edges + block - 1) / block;

  if (grid_v) compute_src_kernel<<<grid_v, block>>>(d_offsets, cache.src, num_vertices);
  if (grid_e) compute_rev_kernel<<<grid_e, block>>>(d_offsets, d_indices, cache.src, cache.rev, num_edges);

  
  cudaMemcpyAsync(cache.active_mask, d_edge_mask, (size_t)mask_words * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

  
  if (grid_e) clear_self_loops_kernel<<<grid_e, block>>>(cache.src, d_indices, cache.active_mask, num_edges);

  
  cudaMemsetAsync(cache.dirty_a, 1, (size_t)num_vertices);

  uint8_t* dirty_prev = cache.dirty_a;
  uint8_t* dirty_curr = cache.dirty_b;

  int32_t h_changed = 1;

  while (h_changed) {
    cudaMemsetAsync(cache.changed_flag, 0, sizeof(int32_t));
    cudaMemsetAsync(dirty_curr, 0, (size_t)num_vertices);

    if (grid_e) count_support_capped_dirty_owner_kernel<<<grid_e, block>>>(
        d_offsets, d_indices, cache.src, cache.rev, cache.active_mask, dirty_prev,
        cache.support, threshold, num_edges);

    if (grid_e) remove_by_support_owner_kernel<<<grid_e, block>>>(
        cache.src, d_indices, cache.rev, cache.active_mask, cache.support, dirty_prev, dirty_curr,
        threshold, num_edges, cache.changed_flag);

    cudaMemcpy(&h_changed, cache.changed_flag, sizeof(int32_t), cudaMemcpyDeviceToHost);

    uint8_t* tmp = dirty_prev;
    dirty_prev = dirty_curr;
    dirty_curr = tmp;
  }

  
  if (grid_e) mask_to_flags_kernel<<<grid_e, block>>>(cache.active_mask, cache.flags, num_edges);

  size_t temp_bytes = cache.scan_temp_capacity;
  cub::DeviceScan::ExclusiveSum(cache.scan_temp, temp_bytes, cache.flags, cache.prefix, num_edges);

  int32_t h_last_prefix = 0;
  int32_t h_last_flag = 0;
  cudaMemcpy(&h_last_prefix, cache.prefix + (num_edges - 1), sizeof(int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_last_flag, cache.flags + (num_edges - 1), sizeof(int32_t), cudaMemcpyDeviceToHost);
  const int32_t n_alive = h_last_prefix + h_last_flag;

  int32_t* out_src = nullptr;
  int32_t* out_dst = nullptr;

  if (n_alive > 0) {
    cudaMalloc(&out_src, (size_t)n_alive * sizeof(int32_t));
    cudaMalloc(&out_dst, (size_t)n_alive * sizeof(int32_t));

    extract_edges_kernel<<<grid_e, block>>>(cache.src, d_indices, cache.active_mask, cache.prefix,
                                            out_src, out_dst, num_edges);
  }

  return k_truss_result_t{out_src, out_dst, (std::size_t)n_alive};
}

}  
