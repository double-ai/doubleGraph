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
    int32_t* edge_src = nullptr;
    int32_t* canon_map = nullptr;
    int32_t* support = nullptr;
    uint8_t* alive = nullptr;
    uint8_t* remove_mark = nullptr;
    int32_t* canonical_edges = nullptr;
    int32_t* frontier = nullptr;
    int32_t* counts = nullptr;  
    int32_t capacity = 0;

    void ensure(int32_t num_edges) {
        if (capacity >= num_edges) return;
        if (edge_src) cudaFree(edge_src);
        if (canon_map) cudaFree(canon_map);
        if (support) cudaFree(support);
        if (alive) cudaFree(alive);
        if (remove_mark) cudaFree(remove_mark);
        if (canonical_edges) cudaFree(canonical_edges);
        if (frontier) cudaFree(frontier);
        if (counts) cudaFree(counts);
        cudaMalloc(&edge_src, sizeof(int32_t) * (size_t)num_edges);
        cudaMalloc(&canon_map, sizeof(int32_t) * (size_t)num_edges);
        cudaMalloc(&support, sizeof(int32_t) * (size_t)num_edges);
        cudaMalloc(&alive, sizeof(uint8_t) * (size_t)num_edges);
        cudaMalloc(&remove_mark, sizeof(uint8_t) * (size_t)num_edges);
        cudaMalloc(&canonical_edges, sizeof(int32_t) * (size_t)num_edges);
        cudaMalloc(&frontier, sizeof(int32_t) * (size_t)num_edges);
        cudaMalloc(&counts, sizeof(int32_t) * 3);
        capacity = num_edges;
    }

    ~Cache() override {
        if (edge_src) cudaFree(edge_src);
        if (canon_map) cudaFree(canon_map);
        if (support) cudaFree(support);
        if (alive) cudaFree(alive);
        if (remove_mark) cudaFree(remove_mark);
        if (canonical_edges) cudaFree(canonical_edges);
        if (frontier) cudaFree(frontier);
        if (counts) cudaFree(counts);
    }
};

__device__ __forceinline__ int32_t ld_i32(const int32_t* p) { return __ldg(p); }
__device__ __forceinline__ uint8_t ld_u8(const uint8_t* p) { return __ldg(reinterpret_cast<const unsigned char*>(p)); }

__device__ __forceinline__ int32_t lower_bound_dev(const int32_t* __restrict__ arr, int32_t lo, int32_t hi, int32_t val)
{
  while (lo < hi) {
    int32_t mid = lo + ((hi - lo) >> 1);
    int32_t x = ld_i32(arr + mid);
    if (x < val)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

__global__ void fill_edge_src_kernel(const int32_t* __restrict__ offsets, int32_t* __restrict__ edge_src, int32_t n)
{
  int32_t u = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (u >= n) return;
  int32_t start = ld_i32(offsets + u);
  int32_t end = ld_i32(offsets + u + 1);
  for (int32_t p = start; p < end; ++p) {
    edge_src[p] = u;
  }
}

__global__ void build_canon_map_kernel(const int32_t* __restrict__ offsets,
                                      const int32_t* __restrict__ indices,
                                      const int32_t* __restrict__ edge_src,
                                      int32_t* __restrict__ canon_map,
                                      int32_t num_edges)
{
  int32_t p = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (p >= num_edges) return;

  int32_t u = ld_i32(edge_src + p);
  int32_t v = ld_i32(indices + p);
  if (u <= v) {
    canon_map[p] = p;
    return;
  }

  int32_t v_start = ld_i32(offsets + v);
  int32_t v_end = ld_i32(offsets + v + 1);
  int32_t pos = lower_bound_dev(indices, v_start, v_end, u);
  canon_map[p] = pos;
}

__global__ void build_alive_and_canonical_kernel(const int32_t* __restrict__ indices,
                                                const int32_t* __restrict__ edge_src,
                                                uint8_t* __restrict__ alive,
                                                int32_t* __restrict__ canonical_edges,
                                                int32_t* __restrict__ canonical_count,
                                                int32_t num_edges)
{
  int32_t p = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  bool is_canon = false;
  if (p < num_edges) {
    int32_t u = ld_i32(edge_src + p);
    int32_t v = ld_i32(indices + p);
    is_canon = (u < v);
    alive[p] = (uint8_t)(is_canon ? 1 : 0);
  }

  unsigned mask = __ballot_sync(0xFFFFFFFFu, is_canon);
  int lane = threadIdx.x & 31;
  int warp_total = __popc(mask);
  if (warp_total == 0) return;

  int warp_prefix = __popc(mask & ((1u << lane) - 1u));
  int base = 0;
  if (lane == 0) {
    base = atomicAdd(canonical_count, warp_total);
  }
  base = __shfl_sync(0xFFFFFFFFu, base, 0);

  if (is_canon) {
    canonical_edges[base + warp_prefix] = p;
  }
}

__global__ void init_support_kernel(const int32_t* __restrict__ offsets,
                                   const int32_t* __restrict__ indices,
                                   const int32_t* __restrict__ edge_src,
                                   const int32_t* __restrict__ canonical_edges,
                                   int32_t canonical_count,
                                   int32_t* __restrict__ support)
{
  int32_t tid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t warp_id = tid >> 5;
  int lane = tid & 31;
  if (warp_id >= canonical_count) return;

  int32_t p = ld_i32(canonical_edges + warp_id);
  int32_t u = ld_i32(edge_src + p);
  int32_t v = ld_i32(indices + p);

  int32_t u_start = ld_i32(offsets + u);
  int32_t u_end = ld_i32(offsets + u + 1);
  int32_t v_start = ld_i32(offsets + v);
  int32_t v_end = ld_i32(offsets + v + 1);

  int32_t du = u_end - u_start;
  int32_t dv = v_end - v_start;

  int32_t iter_start, iter_end, search_start, search_end;
  bool iter_is_u = (du <= dv);
  if (iter_is_u) {
    iter_start = u_start;
    iter_end = u_end;
    search_start = v_start;
    search_end = v_end;
  } else {
    iter_start = v_start;
    iter_end = v_end;
    search_start = u_start;
    search_end = u_end;
  }

  int32_t local = 0;
  int32_t lo = search_start;
  for (int32_t idx = iter_start + lane; idx < iter_end; idx += 32) {
    int32_t w = ld_i32(indices + idx);
    if (w == u || w == v) continue;

    int32_t pos = lower_bound_dev(indices, lo, search_end, w);
    if (pos >= search_end) break;

    if (ld_i32(indices + pos) == w) {
      local++;
      lo = pos + 1;
    } else {
      lo = pos;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    local += __shfl_down_sync(0xFFFFFFFFu, local, offset);
  }

  if (lane == 0) {
    support[p] = local;
  }
}

__global__ void find_frontier_kernel(const int32_t* __restrict__ canonical_edges,
                                    int32_t canonical_count,
                                    const uint8_t* __restrict__ alive,
                                    const int32_t* __restrict__ support,
                                    uint8_t* __restrict__ remove_mark,
                                    int32_t* __restrict__ frontier,
                                    int32_t* __restrict__ frontier_count,
                                    int32_t threshold)
{
  int32_t idx = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  bool take = false;
  int32_t p = 0;
  if (idx < canonical_count) {
    p = ld_i32(canonical_edges + idx);
    uint8_t a = ld_u8(alive + p);
    int32_t s = ld_i32(support + p);
    take = (a != 0) && (s < threshold);
    if (take) remove_mark[p] = 1;
  }

  unsigned mask = __ballot_sync(0xFFFFFFFFu, take);
  int lane = threadIdx.x & 31;
  int warp_total = __popc(mask);
  if (warp_total == 0) return;
  int warp_prefix = __popc(mask & ((1u << lane) - 1u));
  int base = 0;
  if (lane == 0) base = atomicAdd(frontier_count, warp_total);
  base = __shfl_sync(0xFFFFFFFFu, base, 0);
  if (take) frontier[base + warp_prefix] = p;
}

__global__ void triangle_update_kernel(const int32_t* __restrict__ offsets,
                                      const int32_t* __restrict__ indices,
                                      const int32_t* __restrict__ edge_src,
                                      const int32_t* __restrict__ canon_map,
                                      const uint8_t* __restrict__ alive,
                                      const uint8_t* __restrict__ remove_mark,
                                      const int32_t* __restrict__ frontier,
                                      int32_t frontier_count,
                                      int32_t* __restrict__ support)
{
  int32_t tid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t warp_id = tid >> 5;
  int lane = tid & 31;
  if (warp_id >= frontier_count) return;

  int32_t p = ld_i32(frontier + warp_id);
  int32_t u = ld_i32(edge_src + p);
  int32_t v = ld_i32(indices + p);

  int32_t u_start = ld_i32(offsets + u);
  int32_t u_end = ld_i32(offsets + u + 1);
  int32_t v_start = ld_i32(offsets + v);
  int32_t v_end = ld_i32(offsets + v + 1);
  int32_t du = u_end - u_start;
  int32_t dv = v_end - v_start;

  bool iter_is_u = (du <= dv);
  int32_t s = iter_is_u ? u : v;
  int32_t t = iter_is_u ? v : u;
  int32_t iter_start = iter_is_u ? u_start : v_start;
  int32_t iter_end = iter_is_u ? u_end : v_end;
  int32_t search_start = iter_is_u ? v_start : u_start;
  int32_t search_end = iter_is_u ? v_end : u_end;

  int32_t lo = search_start;
  for (int32_t idx_s = iter_start + lane; idx_s < iter_end; idx_s += 32) {
    int32_t w = ld_i32(indices + idx_s);
    if (w == t || w == s) continue;

    int32_t pos_t = lower_bound_dev(indices, lo, search_end, w);
    if (pos_t >= search_end) break;
    if (ld_i32(indices + pos_t) != w) {
      lo = pos_t;
      continue;
    }
    lo = pos_t + 1;

    int32_t id_sw = ld_i32(canon_map + idx_s);
    int32_t id_tw = ld_i32(canon_map + pos_t);

    if (!ld_u8(alive + id_sw) || !ld_u8(alive + id_tw)) continue;

    if (ld_u8(remove_mark + id_sw) && id_sw < p) continue;
    if (ld_u8(remove_mark + id_tw) && id_tw < p) continue;

    if (!ld_u8(remove_mark + id_sw)) atomicSub(support + id_sw, 1);
    if (!ld_u8(remove_mark + id_tw)) atomicSub(support + id_tw, 1);
  }
}

__global__ void finalize_remove_kernel(uint8_t* __restrict__ alive,
                                      uint8_t* __restrict__ remove_mark,
                                      const int32_t* __restrict__ frontier,
                                      int32_t frontier_count)
{
  int32_t i = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= frontier_count) return;
  int32_t p = ld_i32(frontier + i);
  alive[p] = 0;
  remove_mark[p] = 0;
}

__global__ void compact_directed_edges_kernel(const int32_t* __restrict__ canon_map,
                                             const uint8_t* __restrict__ alive,
                                             int32_t* __restrict__ out_edge_indices,
                                             int32_t* __restrict__ out_count,
                                             int32_t num_edges)
{
  int32_t e = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  bool keep = false;
  if (e < num_edges) {
    int32_t cid = ld_i32(canon_map + e);
    keep = (ld_u8(alive + cid) != 0);
  }

  unsigned mask = __ballot_sync(0xFFFFFFFFu, keep);
  int lane = threadIdx.x & 31;
  int warp_total = __popc(mask);
  if (warp_total == 0) return;
  int warp_prefix = __popc(mask & ((1u << lane) - 1u));
  int base = 0;
  if (lane == 0) base = atomicAdd(out_count, warp_total);
  base = __shfl_sync(0xFFFFFFFFu, base, 0);
  if (keep) out_edge_indices[base + warp_prefix] = e;
}

__global__ void gather_edges_kernel(const int32_t* __restrict__ indices,
                                   const int32_t* __restrict__ edge_src,
                                   const int32_t* __restrict__ out_edge_indices,
                                   const int32_t* __restrict__ out_count,
                                   int32_t* __restrict__ out_src,
                                   int32_t* __restrict__ out_dst)
{
  int32_t tid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t n = ld_i32(out_count);
  int32_t stride = (int32_t)(blockDim.x * gridDim.x);
  for (int32_t i = tid; i < n; i += stride) {
    int32_t e = ld_i32(out_edge_indices + i);
    out_src[i] = ld_i32(edge_src + e);
    out_dst[i] = ld_i32(indices + e);
  }
}

}  

k_truss_result_t k_truss(const graph32_t& graph, int32_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    int32_t threshold = k - 2;

    
    if (num_edges == 0 || threshold <= 0) {
        int32_t* out_src = nullptr;
        int32_t* out_dst = nullptr;
        std::size_t result_count = 0;
        if (num_edges > 0) {
            cudaMalloc(&out_src, sizeof(int32_t) * (size_t)num_edges);
            cudaMalloc(&out_dst, sizeof(int32_t) * (size_t)num_edges);
            cache.ensure(num_edges);
            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            fill_edge_src_kernel<<<grid, block>>>(d_offsets, cache.edge_src, num_vertices);
            cudaMemcpyAsync(out_src, cache.edge_src,
                            sizeof(int32_t) * (size_t)num_edges, cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(out_dst, d_indices,
                            sizeof(int32_t) * (size_t)num_edges, cudaMemcpyDeviceToDevice);
            result_count = (std::size_t)num_edges;
        }
        return k_truss_result_t{out_src, out_dst, result_count};
    }

    cache.ensure(num_edges);

    int32_t* d_edge_src = cache.edge_src;
    int32_t* d_canon_map = cache.canon_map;
    int32_t* d_support = cache.support;
    uint8_t* d_alive = cache.alive;
    uint8_t* d_remove_mark = cache.remove_mark;
    int32_t* d_canonical_edges = cache.canonical_edges;
    int32_t* d_frontier = cache.frontier;
    int32_t* d_counts = cache.counts;
    int32_t* d_canonical_count = d_counts + 0;
    int32_t* d_frontier_count = d_counts + 1;
    int32_t* d_out_count = d_counts + 2;

    
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        fill_edge_src_kernel<<<grid, block>>>(d_offsets, d_edge_src, num_vertices);
    }
    {
        int block = 256;
        int grid = (num_edges + block - 1) / block;
        build_canon_map_kernel<<<grid, block>>>(d_offsets, d_indices, d_edge_src, d_canon_map, num_edges);
    }

    cudaMemsetAsync(d_canonical_count, 0, sizeof(int32_t));
    {
        int block = 256;
        int grid = (num_edges + block - 1) / block;
        build_alive_and_canonical_kernel<<<grid, block>>>(d_indices, d_edge_src, d_alive,
                                                          d_canonical_edges, d_canonical_count, num_edges);
    }

    int32_t h_canonical_count = 0;
    cudaMemcpy(&h_canonical_count, d_canonical_count, sizeof(int32_t), cudaMemcpyDeviceToHost);

    cudaMemsetAsync(d_remove_mark, 0, sizeof(uint8_t) * (size_t)num_edges);

    
    {
        int threads = 256;
        int warps = threads / 32;
        int grid = (h_canonical_count + warps - 1) / warps;
        init_support_kernel<<<grid, threads>>>(d_offsets, d_indices, d_edge_src,
                                               d_canonical_edges, h_canonical_count, d_support);
    }

    
    for (int iter = 0; iter < num_edges; ++iter) {
        cudaMemsetAsync(d_frontier_count, 0, sizeof(int32_t));
        {
            int block = 256;
            int grid = (h_canonical_count + block - 1) / block;
            find_frontier_kernel<<<grid, block>>>(d_canonical_edges, h_canonical_count,
                                                  d_alive, d_support, d_remove_mark,
                                                  d_frontier, d_frontier_count, threshold);
        }

        int32_t h_frontier_count = 0;
        cudaMemcpy(&h_frontier_count, d_frontier_count, sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (h_frontier_count == 0) break;

        {
            int threads = 256;
            int warps = threads / 32;
            int grid = (h_frontier_count + warps - 1) / warps;
            triangle_update_kernel<<<grid, threads>>>(d_offsets, d_indices, d_edge_src, d_canon_map,
                                                      d_alive, d_remove_mark, d_frontier,
                                                      h_frontier_count, d_support);
        }
        {
            int block = 256;
            int grid = (h_frontier_count + block - 1) / block;
            finalize_remove_kernel<<<grid, block>>>(d_alive, d_remove_mark, d_frontier, h_frontier_count);
        }
    }

    
    int32_t* out_src = nullptr;
    int32_t* out_dst = nullptr;
    cudaMalloc(&out_src, sizeof(int32_t) * (size_t)num_edges);
    cudaMalloc(&out_dst, sizeof(int32_t) * (size_t)num_edges);

    
    int32_t* d_out_edge_indices = d_canonical_edges;

    cudaMemsetAsync(d_out_count, 0, sizeof(int32_t));
    {
        int block = 256;
        int grid = (num_edges + block - 1) / block;
        compact_directed_edges_kernel<<<grid, block>>>(d_canon_map, d_alive,
                                                       d_out_edge_indices, d_out_count, num_edges);
    }
    {
        int block = 256;
        int grid = (num_edges + block - 1) / block;
        if (grid > 4096) grid = 4096;
        gather_edges_kernel<<<grid, block>>>(d_indices, d_edge_src,
                                             d_out_edge_indices, d_out_count, out_src, out_dst);
    }

    int32_t h_out_count = 0;
    cudaMemcpy(&h_out_count, d_out_count, sizeof(int32_t), cudaMemcpyDeviceToHost);

    return k_truss_result_t{out_src, out_dst, (std::size_t)h_out_count};
}

}  
