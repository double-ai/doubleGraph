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
  int32_t* sources = nullptr;
  int32_t* reverse_idx = nullptr;
  uint8_t* active = nullptr;
  int32_t* changed_flag = nullptr;
  int32_t* support_i32 = nullptr;
  uint8_t* support_u8 = nullptr;
  int32_t* out_srcs = nullptr;
  int32_t* out_dsts = nullptr;
  int32_t* counter = nullptr;

  int64_t sources_capacity = 0;
  int64_t reverse_capacity = 0;
  int64_t active_capacity = 0;
  int64_t changed_capacity = 0;
  int64_t support_i32_capacity = 0;
  int64_t support_u8_capacity = 0;
  int64_t out_srcs_capacity = 0;
  int64_t out_dsts_capacity = 0;
  int64_t counter_capacity = 0;

  void ensure(int32_t num_edges) {
    int64_t ne = (int64_t)num_edges;

    if (sources_capacity < ne) {
      if (sources) cudaFree(sources);
      cudaMalloc(&sources, ne * sizeof(int32_t));
      sources_capacity = ne;
    }
    if (reverse_capacity < ne) {
      if (reverse_idx) cudaFree(reverse_idx);
      cudaMalloc(&reverse_idx, ne * sizeof(int32_t));
      reverse_capacity = ne;
    }
    if (active_capacity < ne) {
      if (active) cudaFree(active);
      cudaMalloc(&active, ne * sizeof(uint8_t));
      active_capacity = ne;
    }
    if (changed_capacity < 1) {
      if (changed_flag) cudaFree(changed_flag);
      cudaMalloc(&changed_flag, sizeof(int32_t));
      changed_capacity = 1;
    }
    if (out_srcs_capacity < ne) {
      if (out_srcs) cudaFree(out_srcs);
      cudaMalloc(&out_srcs, ne * sizeof(int32_t));
      out_srcs_capacity = ne;
    }
    if (out_dsts_capacity < ne) {
      if (out_dsts) cudaFree(out_dsts);
      cudaMalloc(&out_dsts, ne * sizeof(int32_t));
      out_dsts_capacity = ne;
    }
    if (counter_capacity < 1) {
      if (counter) cudaFree(counter);
      cudaMalloc(&counter, sizeof(int32_t));
      counter_capacity = 1;
    }
  }

  void ensure_support_i32(int32_t num_edges) {
    int64_t ne = (int64_t)num_edges;
    if (support_i32_capacity < ne) {
      if (support_i32) cudaFree(support_i32);
      cudaMalloc(&support_i32, ne * sizeof(int32_t));
      support_i32_capacity = ne;
    }
  }

  void ensure_support_u8(int32_t num_edges) {
    int64_t ne = (int64_t)num_edges;
    if (support_u8_capacity < ne) {
      if (support_u8) cudaFree(support_u8);
      cudaMalloc(&support_u8, ne * sizeof(uint8_t));
      support_u8_capacity = ne;
    }
  }

  ~Cache() override {
    if (sources) cudaFree(sources);
    if (reverse_idx) cudaFree(reverse_idx);
    if (active) cudaFree(active);
    if (changed_flag) cudaFree(changed_flag);
    if (support_i32) cudaFree(support_i32);
    if (support_u8) cudaFree(support_u8);
    if (out_srcs) cudaFree(out_srcs);
    if (out_dsts) cudaFree(out_dsts);
    if (counter) cudaFree(counter);
  }
};





__device__ __forceinline__ bool is_bit_active(const uint32_t* __restrict__ mask, int32_t pos) {
  return (mask[pos >> 5] >> (pos & 31)) & 1u;
}

__device__ __forceinline__ int32_t lower_bound_dev(const int32_t* __restrict__ arr, int32_t lo, int32_t hi,
                                                   int32_t target) {
  while (lo < hi) {
    int32_t mid = (lo + hi) >> 1;
    int32_t v = arr[mid];
    if (v < target)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}





__global__ void expand_mask_kernel(const uint32_t* __restrict__ packed_mask, uint8_t* __restrict__ active,
                                   int32_t num_edges) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e < num_edges) active[e] = is_bit_active(packed_mask, e) ? 1 : 0;
}

__global__ void compute_sources_kernel(const int32_t* __restrict__ offsets, int32_t* __restrict__ sources,
                                       int32_t num_vertices) {
  int32_t u = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (u >= num_vertices) return;
  int32_t start = offsets[u];
  int32_t end = offsets[u + 1];
  for (int32_t i = start; i < end; ++i) sources[i] = u;
}

__global__ void compute_reverse_kernel(const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
                                       const int32_t* __restrict__ sources, int32_t* __restrict__ reverse_idx,
                                       int32_t num_edges) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= num_edges) return;
  int32_t u = sources[e];
  int32_t v = indices[e];
  int32_t sv = offsets[v];
  int32_t ev = offsets[v + 1];
  int32_t pos = lower_bound_dev(indices, sv, ev, u);
  if (pos >= ev || indices[pos] != u) pos = 0;
  reverse_idx[e] = pos;
}





__global__ void compute_support_forward_kernel(const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
                                               const int32_t* __restrict__ sources, const uint8_t* __restrict__ active,
                                               int32_t* __restrict__ support, int32_t num_edges, int32_t k_minus_2) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= num_edges) return;

  if (!active[e]) {
    support[e] = 0;
    return;
  }

  int32_t u = sources[e];
  int32_t v = indices[e];
  if (u >= v) {
    support[e] = -1;
    return;
  }

  int32_t su = offsets[u];
  int32_t eu = offsets[u + 1];
  int32_t sv = offsets[v];
  int32_t ev = offsets[v + 1];

  int32_t len_u = eu - su;
  int32_t len_v = ev - sv;
  if (len_u <= 0 || len_v <= 0) {
    support[e] = 0;
    return;
  }

  const int32_t* A = indices + su;
  const int32_t* B = indices + sv;
  const uint8_t* actA = active + su;
  const uint8_t* actB = active + sv;
  int32_t la = len_u;
  int32_t lb = len_v;
  if (la > lb) {
    const int32_t* tptr = A;
    A = B;
    B = tptr;
    const uint8_t* taptr = actA;
    actA = actB;
    actB = taptr;
    int32_t tl = la;
    la = lb;
    lb = tl;
  }

  int32_t first_a = A[0];
  int32_t first_b = B[0];
  int32_t last_a = A[la - 1];
  int32_t last_b = B[lb - 1];
  int32_t max_first = first_a > first_b ? first_a : first_b;
  int32_t min_last = last_a < last_b ? last_a : last_b;
  if (max_first > min_last) {
    support[e] = 0;
    return;
  }

  int32_t ia = (first_a < first_b) ? lower_bound_dev(A, 0, la, first_b) : 0;
  if (ia >= la) {
    support[e] = 0;
    return;
  }
  int32_t a0 = A[ia];
  int32_t ib = (first_b < a0) ? lower_bound_dev(B, 0, lb, a0) : 0;
  if (ib >= lb) {
    support[e] = 0;
    return;
  }

  int32_t count = 0;

  if (lb > (la << 3)) {
    int32_t j = ib;
    for (int32_t i = ia; i < la; ++i) {
      if (!actA[i]) continue;
      int32_t w = A[i];
      if (w > min_last) break;
      if (w == u || w == v) continue;

      int32_t pos = lower_bound_dev(B, j, lb, w);
      if (pos >= lb) break;
      if (B[pos] == w) {
        if (actB[pos]) {
          ++count;
          if (count >= k_minus_2) {
            support[e] = count;
            return;
          }
        }
        j = pos + 1;
      } else {
        j = pos;
      }
    }

    support[e] = count;
    return;
  }

  int32_t i = ia;
  int32_t j = ib;
  while (i < la && j < lb) {
    int32_t ai = A[i];
    int32_t bj = B[j];
    if (ai > min_last || bj > min_last) break;

    if (ai < bj) {
      ++i;
    } else if (ai > bj) {
      ++j;
    } else {
      int32_t w = ai;
      if (w != u && w != v && actA[i] && actB[j]) {
        ++count;
        if (count >= k_minus_2) {
          support[e] = count;
          return;
        }
      }
      ++i;
      ++j;
    }
  }

  support[e] = count;
}

__global__ void copy_and_peel_kernel(const int32_t* __restrict__ sources, const int32_t* __restrict__ indices,
                                     const int32_t* __restrict__ reverse_idx, uint8_t* __restrict__ active,
                                     const int32_t* __restrict__ support, int32_t num_edges, int32_t k_minus_2,
                                     int32_t* __restrict__ changed_flag) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= num_edges) return;
  if (!active[e]) return;

  int32_t u = sources[e];
  int32_t v = indices[e];

  int32_t sup;
  if (u < v) {
    sup = support[e];
  } else if (u > v) {
    sup = support[reverse_idx[e]];
  } else {
    sup = 0;
  }

  if (sup < k_minus_2) {
    active[e] = 0;
    *changed_flag = 1;
  }
}





__global__ void compute_support_forward_u8_kernel(const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
                                                  const int32_t* __restrict__ sources, const uint8_t* __restrict__ active,
                                                  uint8_t* __restrict__ support, int32_t num_edges, int32_t k_minus_2) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= num_edges) return;

  if (!active[e]) {
    support[e] = 0;
    return;
  }

  int32_t u = sources[e];
  int32_t v = indices[e];
  if (u >= v) {
    support[e] = 0;
    return;
  }

  int32_t su = offsets[u];
  int32_t eu = offsets[u + 1];
  int32_t sv = offsets[v];
  int32_t ev = offsets[v + 1];

  int32_t len_u = eu - su;
  int32_t len_v = ev - sv;
  if (len_u <= 0 || len_v <= 0) {
    support[e] = 0;
    return;
  }

  const int32_t* A = indices + su;
  const int32_t* B = indices + sv;
  const uint8_t* actA = active + su;
  const uint8_t* actB = active + sv;
  int32_t la = len_u;
  int32_t lb = len_v;
  if (la > lb) {
    const int32_t* tptr = A;
    A = B;
    B = tptr;
    const uint8_t* taptr = actA;
    actA = actB;
    actB = taptr;
    int32_t tl = la;
    la = lb;
    lb = tl;
  }

  int32_t first_a = A[0];
  int32_t first_b = B[0];
  int32_t last_a = A[la - 1];
  int32_t last_b = B[lb - 1];
  int32_t max_first = first_a > first_b ? first_a : first_b;
  int32_t min_last = last_a < last_b ? last_a : last_b;
  if (max_first > min_last) {
    support[e] = 0;
    return;
  }

  int32_t ia = (first_a < first_b) ? lower_bound_dev(A, 0, la, first_b) : 0;
  if (ia >= la) {
    support[e] = 0;
    return;
  }
  int32_t a0 = A[ia];
  int32_t ib = (first_b < a0) ? lower_bound_dev(B, 0, lb, a0) : 0;
  if (ib >= lb) {
    support[e] = 0;
    return;
  }

  int32_t count = 0;

  if (lb > (la << 3)) {
    int32_t j = ib;
    for (int32_t i = ia; i < la; ++i) {
      if (!actA[i]) continue;
      int32_t w = A[i];
      if (w > min_last) break;
      if (w == u || w == v) continue;

      int32_t pos = lower_bound_dev(B, j, lb, w);
      if (pos >= lb) break;
      if (B[pos] == w) {
        if (actB[pos]) {
          ++count;
          if (count >= k_minus_2) {
            support[e] = (uint8_t)k_minus_2;
            return;
          }
        }
        j = pos + 1;
      } else {
        j = pos;
      }
    }

    support[e] = (uint8_t)count;
    return;
  }

  int32_t i = ia;
  int32_t j = ib;
  while (i < la && j < lb) {
    int32_t ai = A[i];
    int32_t bj = B[j];
    if (ai > min_last || bj > min_last) break;

    if (ai < bj) {
      ++i;
    } else if (ai > bj) {
      ++j;
    } else {
      int32_t w = ai;
      if (w != u && w != v && actA[i] && actB[j]) {
        ++count;
        if (count >= k_minus_2) {
          support[e] = (uint8_t)k_minus_2;
          return;
        }
      }
      ++i;
      ++j;
    }
  }

  support[e] = (uint8_t)count;
}

__global__ void copy_and_peel_u8_kernel(const int32_t* __restrict__ sources, const int32_t* __restrict__ indices,
                                        const int32_t* __restrict__ reverse_idx, uint8_t* __restrict__ active,
                                        const uint8_t* __restrict__ support, int32_t num_edges, int32_t k_minus_2,
                                        int32_t* __restrict__ changed_flag) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= num_edges) return;
  if (!active[e]) return;

  int32_t u = sources[e];
  int32_t v = indices[e];

  int32_t sup;
  if (u < v) {
    sup = (int32_t)support[e];
  } else if (u > v) {
    sup = (int32_t)support[reverse_idx[e]];
  } else {
    sup = 0;
  }

  if (sup < k_minus_2) {
    active[e] = 0;
    *changed_flag = 1;
  }
}





__global__ void extract_edges_kernel(const int32_t* __restrict__ sources, const int32_t* __restrict__ indices,
                                     const uint8_t* __restrict__ active, int32_t* __restrict__ out_srcs,
                                     int32_t* __restrict__ out_dsts, int32_t* __restrict__ counter,
                                     int32_t num_edges) {
  int32_t tid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t lane = threadIdx.x & 31;
  int32_t stride = (int32_t)(gridDim.x * blockDim.x);

  for (int32_t e = tid; e < num_edges; e += stride) {
    bool has = active[e];
    unsigned int mask = __ballot_sync(0xffffffff, has);
    int n = __popc(mask);
    int base;
    if (lane == 0) base = atomicAdd(counter, n);
    base = __shfl_sync(0xffffffff, base, 0);
    if (has) {
      unsigned int lower = mask & ((1u << lane) - 1);
      int off = __popc(lower);
      int pos = base + off;
      out_srcs[pos] = sources[e];
      out_dsts[pos] = indices[e];
    }
  }
}

}  

k_truss_result_t k_truss_mask(const graph32_t& graph,
                              int32_t k) {
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);

  const int32_t* d_offsets = graph.offsets;
  const int32_t* d_indices = graph.indices;
  int32_t num_vertices = graph.number_of_vertices;
  int32_t num_edges = graph.number_of_edges;
  const uint32_t* d_packed_mask = graph.edge_mask;

  if (num_edges == 0) {
    k_truss_result_t result;
    result.edge_srcs = nullptr;
    result.edge_dsts = nullptr;
    result.num_edges = 0;
    return result;
  }

  const int32_t k_minus_2 = k - 2;

  cache.ensure(num_edges);

  int32_t* d_sources = cache.sources;
  int32_t* d_reverse = cache.reverse_idx;
  uint8_t* d_active = cache.active;
  int32_t* d_changed = cache.changed_flag;

  
  int block = 256;
  int grid;

  grid = (num_edges + block - 1) / block;
  expand_mask_kernel<<<grid, block>>>(d_packed_mask, d_active, num_edges);

  grid = (num_vertices + block - 1) / block;
  compute_sources_kernel<<<grid, block>>>(d_offsets, d_sources, num_vertices);

  grid = (num_edges + block - 1) / block;
  compute_reverse_kernel<<<grid, block>>>(d_offsets, d_indices, d_sources, d_reverse, num_edges);

  
  const bool use_u8_support = (k_minus_2 >= 0 && k_minus_2 <= 255);
  int32_t* d_support_i32 = nullptr;
  uint8_t* d_support_u8 = nullptr;

  if (use_u8_support) {
    cache.ensure_support_u8(num_edges);
    d_support_u8 = cache.support_u8;
  } else {
    cache.ensure_support_i32(num_edges);
    d_support_i32 = cache.support_i32;
  }

  
  int32_t h_changed = 1;
  grid = (num_edges + block - 1) / block;
  while (h_changed) {
    cudaMemset(d_changed, 0, sizeof(int32_t));

    if (use_u8_support) {
      compute_support_forward_u8_kernel<<<grid, block>>>(d_offsets, d_indices, d_sources, d_active,
                                                          d_support_u8, num_edges, k_minus_2);
      copy_and_peel_u8_kernel<<<grid, block>>>(d_sources, d_indices, d_reverse, d_active,
                                                d_support_u8, num_edges, k_minus_2, d_changed);
    } else {
      compute_support_forward_kernel<<<grid, block>>>(d_offsets, d_indices, d_sources, d_active,
                                                       d_support_i32, num_edges, k_minus_2);
      copy_and_peel_kernel<<<grid, block>>>(d_sources, d_indices, d_reverse, d_active,
                                             d_support_i32, num_edges, k_minus_2, d_changed);
    }

    cudaMemcpy(&h_changed, d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost);
  }

  
  cudaMemset(cache.counter, 0, sizeof(int32_t));

  grid = (num_edges + block - 1) / block;
  extract_edges_kernel<<<grid, block>>>(d_sources, d_indices, d_active,
                                         cache.out_srcs, cache.out_dsts,
                                         cache.counter, num_edges);

  
  int32_t h_count = 0;
  cudaMemcpy(&h_count, cache.counter, sizeof(int32_t), cudaMemcpyDeviceToHost);

  
  k_truss_result_t result;
  result.num_edges = (std::size_t)h_count;

  if (h_count == 0) {
    result.edge_srcs = nullptr;
    result.edge_dsts = nullptr;
    return result;
  }

  cudaMalloc(&result.edge_srcs, h_count * sizeof(int32_t));
  cudaMalloc(&result.edge_dsts, h_count * sizeof(int32_t));
  cudaMemcpyAsync(result.edge_srcs, cache.out_srcs, h_count * sizeof(int32_t), cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(result.edge_dsts, cache.out_dsts, h_count * sizeof(int32_t), cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();

  return result;
}

}  
