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
#include <utility>

namespace aai {

namespace {





struct Cache : Cacheable {
    
    int32_t* src = nullptr;
    int32_t* rev = nullptr;
    int32_t* support = nullptr;
    uint32_t* work_mask = nullptr;
    int32_t* queue = nullptr;
    int32_t* qsize = nullptr;
    int32_t* changed = nullptr;
    int32_t* out_count = nullptr;
    int32_t ne_cap = 0;

    
    uint8_t* dirty_a = nullptr;
    uint8_t* dirty_b = nullptr;
    int32_t nv_cap = 0;

    void ensure(int32_t ne, int32_t nv) {
        if (ne_cap < ne) {
            if (src) cudaFree(src);
            if (rev) cudaFree(rev);
            if (support) cudaFree(support);
            if (work_mask) cudaFree(work_mask);
            if (queue) cudaFree(queue);
            if (qsize) cudaFree(qsize);
            if (changed) cudaFree(changed);
            if (out_count) cudaFree(out_count);

            int32_t mask_words = (ne + 31) / 32;
            int32_t max_queue = (ne + 1) / 2;

            cudaMalloc(&src, (size_t)ne * sizeof(int32_t));
            cudaMalloc(&rev, (size_t)ne * sizeof(int32_t));
            cudaMalloc(&support, (size_t)ne * sizeof(int32_t));
            cudaMalloc(&work_mask, (size_t)mask_words * sizeof(uint32_t));
            cudaMalloc(&queue, (size_t)max_queue * sizeof(int32_t));
            cudaMalloc(&qsize, sizeof(int32_t));
            cudaMalloc(&changed, sizeof(int32_t));
            cudaMalloc(&out_count, sizeof(int32_t));

            ne_cap = ne;
        }
        if (nv_cap < nv) {
            if (dirty_a) cudaFree(dirty_a);
            if (dirty_b) cudaFree(dirty_b);

            cudaMalloc(&dirty_a, (size_t)nv * sizeof(uint8_t));
            cudaMalloc(&dirty_b, (size_t)nv * sizeof(uint8_t));

            nv_cap = nv;
        }
    }

    ~Cache() override {
        if (src) cudaFree(src);
        if (rev) cudaFree(rev);
        if (support) cudaFree(support);
        if (work_mask) cudaFree(work_mask);
        if (queue) cudaFree(queue);
        if (qsize) cudaFree(qsize);
        if (changed) cudaFree(changed);
        if (out_count) cudaFree(out_count);
        if (dirty_a) cudaFree(dirty_a);
        if (dirty_b) cudaFree(dirty_b);
    }
};





static constexpr uint32_t FULL_MASK = 0xffffffffu;

__device__ __forceinline__ bool edge_active(const uint32_t* __restrict__ mask, int32_t e) {
  return (mask[e >> 5] >> (e & 31)) & 1u;
}

__device__ __forceinline__ void clear_edge(uint32_t* __restrict__ mask, int32_t e) {
  atomicAnd(mask + (e >> 5), ~(1u << (e & 31)));
}

__global__ void compute_src_array_kernel(const int32_t* __restrict__ offsets,
                                        int32_t* __restrict__ src,
                                        int32_t nv) {
  int32_t u = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (u >= nv) return;
  int32_t start = offsets[u];
  int32_t end = offsets[u + 1];
  for (int32_t e = start; e < end; ++e) src[e] = u;
}

__global__ void compute_reverse_kernel(const int32_t* __restrict__ offsets,
                                      const int32_t* __restrict__ indices,
                                      const int32_t* __restrict__ src,
                                      int32_t* __restrict__ rev,
                                      int32_t ne) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= ne) return;
  int32_t u = src[e];
  int32_t v = indices[e];
  int32_t lo = offsets[v];
  int32_t hi = offsets[v + 1];
  while (lo < hi) {
    int32_t mid = lo + ((hi - lo) >> 1);
    int32_t x = indices[mid];
    if (x < u) lo = mid + 1;
    else hi = mid;
  }
  rev[e] = (lo < offsets[v + 1] && indices[lo] == u) ? lo : e;
}

__global__ void remove_self_loops_kernel(uint32_t* __restrict__ mask,
                                        const int32_t* __restrict__ src,
                                        const int32_t* __restrict__ indices,
                                        const int32_t* __restrict__ rev,
                                        int32_t ne) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= ne) return;
  if (!edge_active(mask, e)) return;
  int32_t u = src[e];
  if (u == indices[e]) {
    clear_edge(mask, e);
    clear_edge(mask, rev[e]);
  }
}

__global__ void build_work_queue_dirty_kernel(const int32_t* __restrict__ src,
                                             const int32_t* __restrict__ indices,
                                             const uint32_t* __restrict__ mask,
                                             const uint8_t* __restrict__ dirty_prev,
                                             int32_t* __restrict__ queue,
                                             int32_t* __restrict__ queue_size,
                                             int32_t ne) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int lane = threadIdx.x & 31;

  bool pred = false;
  if (e < ne && edge_active(mask, e)) {
    int32_t u = src[e];
    int32_t v = indices[e];
    if (u < v) pred = (dirty_prev[u] | dirty_prev[v]) != 0;
  }

  uint32_t ballot = __ballot_sync(FULL_MASK, pred);
  int warp_count = __popc(ballot);
  int prefix = __popc(ballot & ((1u << lane) - 1));
  int base = 0;
  if (lane == 0 && warp_count) base = atomicAdd(queue_size, warp_count);
  base = __shfl_sync(FULL_MASK, base, 0);
  if (pred) queue[base + prefix] = e;
}

__device__ __forceinline__ int lower_bound_int(const int32_t* __restrict__ arr, int lo, int hi, int32_t val) {
  while (lo < hi) {
    int mid = lo + ((hi - lo) >> 1);
    if (arr[mid] < val) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

__global__ void count_support_queue_kernel(const int32_t* __restrict__ offsets,
                                          const int32_t* __restrict__ indices,
                                          const int32_t* __restrict__ src,
                                          const uint32_t* __restrict__ mask,
                                          int32_t* __restrict__ support,
                                          const int32_t* __restrict__ queue,
                                          const int32_t* __restrict__ queue_size_ptr) {
  int warp_id = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  int qsz = *queue_size_ptr;
  if (warp_id >= qsz) return;

  int32_t e = queue[warp_id];
  int32_t u = src[e];
  int32_t v = indices[e];

  int32_t u0 = offsets[u];
  int32_t u1 = offsets[u + 1];
  int32_t v0 = offsets[v];
  int32_t v1 = offsets[v + 1];

  int32_t ulen = u1 - u0;
  int32_t vlen = v1 - v0;

  int32_t s_base, s_len, l_base, l_len;
  if (ulen <= vlen) {
    s_base = u0; s_len = ulen; l_base = v0; l_len = vlen;
  } else {
    s_base = v0; s_len = vlen; l_base = u0; l_len = ulen;
  }

  int32_t cnt = 0;

  if (s_len > 0 && l_len > 0) {
    int32_t l_min = indices[l_base];
    int32_t l_max = indices[l_base + l_len - 1];
    int32_t s_first = indices[s_base];
    int32_t s_last = indices[s_base + s_len - 1];

    if (s_first <= l_max && s_last >= l_min) {
      int s_skip = 0;
      if (s_first < l_min) s_skip = lower_bound_int(indices, s_base, s_base + s_len, l_min) - s_base;
      int s_end_eff = s_len;
      if (s_last > l_max) s_end_eff = lower_bound_int(indices, s_base + s_skip, s_base + s_len, l_max + 1) - s_base;
      int eff_s_len = s_end_eff - s_skip;

      int l_skip = 0;
      if (eff_s_len > 0) {
        int32_t first_s = indices[s_base + s_skip];
        if (l_min < first_s) l_skip = lower_bound_int(indices, l_base, l_base + l_len, first_s) - l_base;
      }

      int l_eff_base = l_base + l_skip;
      int l_eff_len = l_len - l_skip;

      int l_lo = 0;
      for (int i = lane; i < eff_s_len; i += 32) {
        int s_idx = s_base + s_skip + i;
        if (!edge_active(mask, s_idx)) continue;
        int32_t w = indices[s_idx];
        if (w == u || w == v) continue;

        int lo = l_lo;
        int hi = l_eff_len;
        while (lo < hi) {
          int mid = lo + ((hi - lo) >> 1);
          if (indices[l_eff_base + mid] < w) lo = mid + 1;
          else hi = mid;
        }
        if (lo < l_eff_len && indices[l_eff_base + lo] == w) {
          if (edge_active(mask, l_eff_base + lo)) cnt++;
          l_lo = lo + 1;
        } else {
          l_lo = lo;
        }
      }
    }
  }

  for (int off = 16; off > 0; off >>= 1) cnt += __shfl_down_sync(FULL_MASK, cnt, off);

  if (lane == 0) support[e] = cnt;
}

__global__ void remove_unsupported_queue_kernel(const int32_t* __restrict__ support,
                                               uint32_t* __restrict__ mask,
                                               const int32_t* __restrict__ rev,
                                               const int32_t* __restrict__ src,
                                               const int32_t* __restrict__ indices,
                                               const int32_t* __restrict__ queue,
                                               const int32_t* __restrict__ queue_size_ptr,
                                               uint8_t* __restrict__ dirty_curr,
                                               int32_t threshold,
                                               int32_t* __restrict__ changed_flag) {
  int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int qsz = *queue_size_ptr;
  if (idx >= qsz) return;

  int32_t e = queue[idx];
  if (!edge_active(mask, e)) return;
  if (support[e] >= threshold) return;

  int32_t u = src[e];
  int32_t v = indices[e];
  clear_edge(mask, e);
  clear_edge(mask, rev[e]);
  dirty_curr[u] = 1;
  dirty_curr[v] = 1;
  atomicOr(changed_flag, 1);
}

__global__ void extract_edges_kernel(const int32_t* __restrict__ src,
                                    const int32_t* __restrict__ indices,
                                    const uint32_t* __restrict__ mask,
                                    int32_t* __restrict__ out_src,
                                    int32_t* __restrict__ out_dst,
                                    int32_t* __restrict__ out_count,
                                    int32_t ne) {
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int lane = threadIdx.x & 31;
  bool pred = (e < ne) && edge_active(mask, e);
  uint32_t ballot = __ballot_sync(FULL_MASK, pred);
  int warp_count = __popc(ballot);
  int prefix = __popc(ballot & ((1u << lane) - 1));
  int base = 0;
  if (lane == 0 && warp_count) base = atomicAdd(out_count, warp_count);
  base = __shfl_sync(FULL_MASK, base, 0);
  if (pred) {
    int pos = base + prefix;
    out_src[pos] = src[e];
    out_dst[pos] = indices[e];
  }
}





void launch_compute_src_array(const int32_t* offsets, int32_t* src, int32_t nv) {
  if (nv <= 0) return;
  int block = 256;
  int grid = (nv + block - 1) / block;
  compute_src_array_kernel<<<grid, block>>>(offsets, src, nv);
}

void launch_compute_reverse(const int32_t* offsets, const int32_t* indices, const int32_t* src,
                           int32_t* rev, int32_t ne) {
  if (ne <= 0) return;
  int block = 256;
  int grid = (ne + block - 1) / block;
  compute_reverse_kernel<<<grid, block>>>(offsets, indices, src, rev, ne);
}

void launch_remove_self_loops(uint32_t* mask, const int32_t* src, const int32_t* indices, const int32_t* rev, int32_t ne) {
  if (ne <= 0) return;
  int block = 256;
  int grid = (ne + block - 1) / block;
  remove_self_loops_kernel<<<grid, block>>>(mask, src, indices, rev, ne);
}

void launch_build_work_queue_dirty(const int32_t* src, const int32_t* indices, const uint32_t* mask,
                                   const uint8_t* dirty_prev, int32_t* queue, int32_t* queue_size, int32_t ne) {
  if (ne <= 0) return;
  int block = 256;
  int grid = (ne + block - 1) / block;
  build_work_queue_dirty_kernel<<<grid, block>>>(src, indices, mask, dirty_prev, queue, queue_size, ne);
}

void launch_count_support_queue(const int32_t* offsets, const int32_t* indices, const int32_t* src,
                                const uint32_t* mask, int32_t* support,
                                const int32_t* queue, const int32_t* queue_size_ptr, int32_t max_queue) {
  if (max_queue <= 0) return;
  int threads = 256;
  int warps_per_block = threads / 32;
  int grid = (max_queue + warps_per_block - 1) / warps_per_block;
  count_support_queue_kernel<<<grid, threads>>>(offsets, indices, src, mask, support, queue, queue_size_ptr);
}

void launch_remove_unsupported_queue(const int32_t* support, uint32_t* mask, const int32_t* rev,
                                     const int32_t* src, const int32_t* indices,
                                     const int32_t* queue, const int32_t* queue_size_ptr,
                                     uint8_t* dirty_curr, int32_t threshold, int32_t* changed_flag,
                                     int32_t max_queue) {
  if (max_queue <= 0) return;
  int block = 256;
  int grid = (max_queue + block - 1) / block;
  remove_unsupported_queue_kernel<<<grid, block>>>(support, mask, rev, src, indices, queue, queue_size_ptr,
                                                   dirty_curr, threshold, changed_flag);
}

void launch_extract_edges(const int32_t* src, const int32_t* indices, const uint32_t* mask,
                          int32_t* out_src, int32_t* out_dst, int32_t* out_count, int32_t ne) {
  if (ne <= 0) return;
  int block = 256;
  int grid = (ne + block - 1) / block;
  extract_edges_kernel<<<grid, block>>>(src, indices, mask, out_src, out_dst, out_count, ne);
}

}  





k_truss_result_t k_truss_seg_mask(const graph32_t& graph, int32_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;

    int32_t threshold = k - 2;
    if (ne == 0 || threshold <= 0) {
        return k_truss_result_t{nullptr, nullptr, 0};
    }

    cache.ensure(ne, nv);

    int32_t mask_words = (ne + 31) / 32;
    int32_t max_queue = (ne + 1) / 2;

    
    cudaMemcpyAsync(cache.work_mask, d_edge_mask,
                    (size_t)mask_words * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    
    launch_compute_src_array(d_offsets, cache.src, nv);
    launch_compute_reverse(d_offsets, d_indices, cache.src, cache.rev, ne);
    launch_remove_self_loops(cache.work_mask, cache.src, d_indices, cache.rev, ne);

    
    cudaMemsetAsync(cache.dirty_a, 1, (size_t)nv);

    uint8_t* d_dirty_prev = cache.dirty_a;
    uint8_t* d_dirty_curr = cache.dirty_b;

    while (true) {
        cudaMemsetAsync(cache.qsize, 0, sizeof(int32_t));
        launch_build_work_queue_dirty(cache.src, d_indices, cache.work_mask,
                                      d_dirty_prev, cache.queue, cache.qsize, ne);

        launch_count_support_queue(d_offsets, d_indices, cache.src, cache.work_mask,
                                   cache.support, cache.queue, cache.qsize, max_queue);

        cudaMemsetAsync(cache.changed, 0, sizeof(int32_t));
        cudaMemsetAsync(d_dirty_curr, 0, (size_t)nv);

        launch_remove_unsupported_queue(cache.support, cache.work_mask, cache.rev,
                                        cache.src, d_indices, cache.queue, cache.qsize,
                                        d_dirty_curr, threshold, cache.changed, max_queue);

        int32_t h_changed = 0;
        cudaMemcpy(&h_changed, cache.changed, sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (h_changed == 0) break;
        std::swap(d_dirty_prev, d_dirty_curr);
    }

    
    int32_t* d_out_src = nullptr;
    int32_t* d_out_dst = nullptr;
    cudaMalloc(&d_out_src, (size_t)ne * sizeof(int32_t));
    cudaMalloc(&d_out_dst, (size_t)ne * sizeof(int32_t));

    cudaMemsetAsync(cache.out_count, 0, sizeof(int32_t));
    launch_extract_edges(cache.src, d_indices, cache.work_mask,
                         d_out_src, d_out_dst, cache.out_count, ne);

    int32_t h_count = 0;
    cudaMemcpy(&h_count, cache.out_count, sizeof(int32_t), cudaMemcpyDeviceToHost);

    return k_truss_result_t{d_out_src, d_out_dst, (std::size_t)h_count};
}

}  
