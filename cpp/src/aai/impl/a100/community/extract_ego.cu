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
#include <algorithm>
#include <vector>

namespace aai {

namespace {

struct Cache : Cacheable {
    uint32_t* bitmap = nullptr;
    int32_t* ego = nullptr;
    int32_t* ego_size_ptr = nullptr;
    int32_t* frontier_range = nullptr;
    uint64_t* edge_counter = nullptr;
    int32_t* buf_src = nullptr;
    int32_t* buf_dst = nullptr;

    int64_t cap_bitmap_words = 0;
    int64_t cap_vertices = 0;
    int64_t cap_buf_src = 0;
    int64_t cap_buf_dst = 0;

    ~Cache() override {
        if (bitmap) cudaFree(bitmap);
        if (ego) cudaFree(ego);
        if (ego_size_ptr) cudaFree(ego_size_ptr);
        if (frontier_range) cudaFree(frontier_range);
        if (edge_counter) cudaFree(edge_counter);
        if (buf_src) cudaFree(buf_src);
        if (buf_dst) cudaFree(buf_dst);
    }
};




__device__ __forceinline__ bool bitmap_test(const uint32_t* __restrict__ bitmap, int32_t v)
{
  uint32_t word = bitmap[(uint32_t)v >> 5];
  uint32_t mask = 1u << ((uint32_t)v & 31);
  return (word & mask) != 0;
}




__global__ void init_expand_kernel(const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
                                  uint32_t* __restrict__ bitmap, int32_t source,
                                  int32_t* __restrict__ ego, int32_t* __restrict__ ego_size)
{
  if (threadIdx.x == 0) {
    bitmap[(uint32_t)source >> 5] |= (1u << ((uint32_t)source & 31));
    ego[0] = source;
    *ego_size = 1;
  }
  __syncthreads();

  int32_t s = off[source];
  int32_t e = off[source + 1];

  for (int32_t j = s + (int32_t)threadIdx.x; j < e; j += (int32_t)blockDim.x) {
    int32_t u = idx[j];
    uint32_t mask = 1u << ((uint32_t)u & 31);
    uint32_t* word = bitmap + ((uint32_t)u >> 5);
    uint32_t old = atomicOr(word, mask);
    if ((old & mask) == 0) {
      int32_t pos = atomicAdd(ego_size, 1);
      ego[pos] = u;
    }
  }
}




__global__ void bfs_expand_warp_devrange_kernel(const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
                                               uint32_t* __restrict__ bitmap,
                                               int32_t* __restrict__ ego,
                                               const int32_t* __restrict__ frontier_start_ptr,
                                               const int32_t* __restrict__ frontier_end_ptr,
                                               int32_t* __restrict__ ego_size)
{
  int32_t frontier_start = *frontier_start_ptr;
  int32_t frontier_end = *frontier_end_ptr;
  if (frontier_start >= frontier_end) return;

  int32_t warp_global = (int32_t)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int32_t lane = (int32_t)(threadIdx.x & 31);
  int32_t total_warps = (int32_t)((gridDim.x * blockDim.x) >> 5);

  for (int32_t i = frontier_start + warp_global; i < frontier_end; i += total_warps) {
    int32_t v = ego[i];
    int32_t s = off[v];
    int32_t e = off[v + 1];
    int32_t deg = e - s;
    int32_t chunks = (deg + 31) >> 5;

    for (int32_t c = 0; c < chunks; c++) {
      int32_t j = s + (c << 5) + lane;
      int32_t u = 0;
      bool is_new = false;
      if (j < e) {
        u = idx[j];
        uint32_t mask = 1u << ((uint32_t)u & 31);
        uint32_t* word = bitmap + ((uint32_t)u >> 5);
        uint32_t old = atomicOr(word, mask);
        is_new = ((old & mask) == 0);
      }

      unsigned ballot = __ballot_sync(0xFFFFFFFF, is_new);
      int n_new = __popc(ballot);

      int32_t base = 0;
      if (lane == 0 && n_new) {
        base = atomicAdd(ego_size, n_new);
      }
      base = __shfl_sync(0xFFFFFFFF, base, 0);

      if (is_new) {
        int offset = __popc(ballot & ((1u << lane) - 1));
        ego[base + offset] = u;
      }
    }
  }
}




__global__ void write_edges_atomic_kernel(const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
                                         const uint32_t* __restrict__ bitmap,
                                         const int32_t* __restrict__ ego,
                                         const int32_t* __restrict__ ego_size_ptr,
                                         uint64_t* __restrict__ edge_counter,
                                         int32_t* __restrict__ out_srcs,
                                         int32_t* __restrict__ out_dsts)
{
  int32_t ego_size = *ego_size_ptr;
  if (ego_size <= 0) return;

  int32_t warp_global = (int32_t)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int32_t lane = (int32_t)(threadIdx.x & 31);
  int32_t total_warps = (int32_t)((gridDim.x * blockDim.x) >> 5);

  for (int32_t warp_id = warp_global; warp_id < ego_size; warp_id += total_warps) {
    int32_t v = ego[warp_id];
    int32_t s = off[v];
    int32_t e = off[v + 1];
    int32_t deg = e - s;
    int32_t chunks = (deg + 31) >> 5;

    for (int32_t c = 0; c < chunks; c++) {
      int32_t j = s + (c << 5) + lane;
      int32_t u = 0;
      bool match = false;
      if (j < e) {
        u = idx[j];
        match = bitmap_test(bitmap, u);
      }

      unsigned ballot = __ballot_sync(0xFFFFFFFF, match);
      int n_out = __popc(ballot);

      uint64_t base = 0;
      if (lane == 0 && n_out) {
        base = atomicAdd(reinterpret_cast<unsigned long long*>(edge_counter), (unsigned long long)n_out);
      }
      base = __shfl_sync(0xFFFFFFFF, base, 0);

      if (match) {
        int prefix = __popc(ballot & ((1u << lane) - 1));
        uint64_t out = base + (uint64_t)prefix;
        out_srcs[out] = v;
        out_dsts[out] = u;
      }
    }
  }
}




void launch_init_expand(const int32_t* off, const int32_t* idx, uint32_t* bitmap, int32_t source,
                        int32_t* ego, int32_t* ego_size, cudaStream_t stream)
{
  init_expand_kernel<<<1, 256, 0, stream>>>(off, idx, bitmap, source, ego, ego_size);
}

void launch_bfs_expand_dev(const int32_t* off, const int32_t* idx, uint32_t* bitmap,
                           int32_t* ego,
                           const int32_t* frontier_start_ptr,
                           const int32_t* frontier_end_ptr,
                           int32_t* ego_size,
                           cudaStream_t stream)
{
  int block = 256;
  int grid = 256;
  bfs_expand_warp_devrange_kernel<<<grid, block, 0, stream>>>(off, idx, bitmap, ego, frontier_start_ptr, frontier_end_ptr,
                                                             ego_size);
}

void launch_write_edges_atomic(const int32_t* off, const int32_t* idx, const uint32_t* bitmap,
                              const int32_t* ego, const int32_t* ego_size_ptr,
                              uint64_t* edge_counter,
                              int32_t* out_srcs, int32_t* out_dsts,
                              cudaStream_t stream)
{
  int block = 256;
  int grid = 256;
  write_edges_atomic_kernel<<<grid, block, 0, stream>>>(off, idx, bitmap, ego, ego_size_ptr, edge_counter, out_srcs,
                                                       out_dsts);
}

}  

extract_ego_result_t extract_ego(const graph32_t& graph,
                                 const int32_t* source_vertices,
                                 std::size_t n_sources,
                                 int32_t radius) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    cudaStream_t stream = 0;

    
    int64_t nv = static_cast<int64_t>(num_vertices);
    int64_t bw = (nv + 31) / 32;

    if (bw > cache.cap_bitmap_words) {
        if (cache.bitmap) cudaFree(cache.bitmap);
        cudaMalloc(&cache.bitmap, bw * sizeof(uint32_t));
        cache.cap_bitmap_words = bw;
    }

    if (nv > cache.cap_vertices) {
        if (cache.ego) cudaFree(cache.ego);
        cudaMalloc(&cache.ego, nv * sizeof(int32_t));
        cache.cap_vertices = nv;
    }

    
    if (!cache.ego_size_ptr) {
        cudaMalloc(&cache.ego_size_ptr, sizeof(int32_t));
    }
    if (!cache.frontier_range) {
        cudaMalloc(&cache.frontier_range, 2 * sizeof(int32_t));
    }
    if (!cache.edge_counter) {
        cudaMalloc(&cache.edge_counter, sizeof(uint64_t));
    }

    
    if (cache.cap_buf_src == 0) {
        int64_t initial = std::max<int64_t>(static_cast<int64_t>(num_edges), 1);
        cudaMalloc(&cache.buf_src, initial * sizeof(int32_t));
        cache.cap_buf_src = initial;
    }
    if (cache.cap_buf_dst == 0) {
        int64_t initial = std::max<int64_t>(static_cast<int64_t>(num_edges), 1);
        cudaMalloc(&cache.buf_dst, initial * sizeof(int32_t));
        cache.cap_buf_dst = initial;
    }

    uint32_t* d_bitmap = cache.bitmap;
    int32_t* d_ego = cache.ego;
    int32_t* d_ego_size = cache.ego_size_ptr;
    int32_t* d_frontier_start = cache.frontier_range;
    int32_t* d_frontier_end = cache.frontier_range + 1;
    uint64_t* d_edge_counter = cache.edge_counter;

    
    std::vector<int32_t> h_sources(static_cast<size_t>(n_sources));
    cudaMemcpyAsync(h_sources.data(), source_vertices, n_sources * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    uint64_t cur_off = 0;
    std::vector<std::size_t> h_offsets(static_cast<size_t>(n_sources) + 1);
    h_offsets[0] = 0;

    int64_t bitmap_words = (static_cast<int64_t>(num_vertices) + 31) / 32;

    for (std::size_t s = 0; s < n_sources; s++) {
        int32_t source = h_sources[s];

        cudaMemsetAsync(d_bitmap, 0, bitmap_words * sizeof(uint32_t), stream);
        launch_init_expand(d_off, d_idx, d_bitmap, source, d_ego, d_ego_size, stream);

        
        int32_t h_one = 1;
        cudaMemcpyAsync(d_frontier_start, &h_one, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_frontier_end, d_ego_size, sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

        for (int32_t hop = 1; hop < radius; hop++) {
            launch_bfs_expand_dev(d_off, d_idx, d_bitmap, d_ego, d_frontier_start, d_frontier_end, d_ego_size, stream);
            cudaMemcpyAsync(d_frontier_start, d_frontier_end, sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(d_frontier_end, d_ego_size, sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        }

        
        uint64_t needed_cap = cur_off + static_cast<uint64_t>(num_edges);
        if (needed_cap > static_cast<uint64_t>(cache.cap_buf_src)) {
            int64_t new_cap = static_cast<int64_t>(needed_cap);
            int32_t* new_s = nullptr;
            int32_t* new_d = nullptr;
            cudaMalloc(&new_s, new_cap * sizeof(int32_t));
            cudaMalloc(&new_d, new_cap * sizeof(int32_t));
            if (cur_off > 0) {
                cudaMemcpyAsync(new_s, cache.buf_src, static_cast<size_t>(cur_off) * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(new_d, cache.buf_dst, static_cast<size_t>(cur_off) * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
                cudaStreamSynchronize(stream);
            }
            cudaFree(cache.buf_src);
            cudaFree(cache.buf_dst);
            cache.buf_src = new_s;
            cache.buf_dst = new_d;
            cache.cap_buf_src = new_cap;
            cache.cap_buf_dst = new_cap;
        }

        
        cudaMemcpyAsync(d_edge_counter, &cur_off, sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
        launch_write_edges_atomic(d_off, d_idx, d_bitmap, d_ego, d_ego_size, d_edge_counter,
                                  cache.buf_src, cache.buf_dst, stream);

        
        cudaMemcpyAsync(&cur_off, d_edge_counter, sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        h_offsets[s + 1] = static_cast<std::size_t>(cur_off);
    }

    
    std::size_t total_edges = static_cast<std::size_t>(cur_off);

    int32_t* out_srcs = nullptr;
    int32_t* out_dsts = nullptr;
    std::size_t* out_offsets = nullptr;

    if (total_edges > 0) {
        cudaMalloc(&out_srcs, total_edges * sizeof(int32_t));
        cudaMalloc(&out_dsts, total_edges * sizeof(int32_t));
        cudaMemcpyAsync(out_srcs, cache.buf_src, total_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(out_dsts, cache.buf_dst, total_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    }

    cudaMalloc(&out_offsets, (n_sources + 1) * sizeof(std::size_t));
    cudaMemcpyAsync(out_offsets, h_offsets.data(), (n_sources + 1) * sizeof(std::size_t), cudaMemcpyHostToDevice, stream);

    cudaStreamSynchronize(stream);

    return extract_ego_result_t{
        out_srcs,
        out_dsts,
        out_offsets,
        total_edges,
        n_sources + 1,
    };
}

}  
