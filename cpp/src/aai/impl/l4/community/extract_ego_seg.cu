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
#include <cub/device/device_scan.cuh>
#include <cstdint>

namespace aai {

namespace {

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)


#ifndef NUM_SMS
#define NUM_SMS 58
#endif



__global__ void mark_sources_kernel(uint32_t* __restrict__ visited, uint32_t* __restrict__ frontier,
                                   const int32_t* __restrict__ sources, int32_t n_sources, int32_t bitmap_words) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_sources) {
    int32_t src = sources[tid];
    int64_t base = (int64_t)tid * bitmap_words;
    uint32_t mask = 1u << (src & 31);
    visited[base + (src >> 5)] = mask;
    frontier[base + (src >> 5)] = mask;
  }
}

__global__ void bfs_expand_kernel(const int32_t* __restrict__ csr_offsets, const int32_t* __restrict__ csr_indices,
                                 uint32_t* __restrict__ visited, const uint32_t* __restrict__ frontier,
                                 uint32_t* __restrict__ new_frontier, int32_t num_vertices, int32_t bitmap_words,
                                 int32_t n_sources) {
  for (int src_idx = (int)blockIdx.y; src_idx < n_sources; src_idx += (int)gridDim.y) {
    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    int64_t base = (int64_t)src_idx * bitmap_words;
    const uint32_t* frt = frontier + base;
    uint32_t* vis = visited + base;
    uint32_t* nfrt = new_frontier + base;

    int warp_stride = (int)gridDim.x * WARPS_PER_BLOCK;
    for (int w = (int)blockIdx.x * WARPS_PER_BLOCK + warp_in_block; w < bitmap_words; w += warp_stride) {
      uint32_t word = frt[w];
      if (word == 0) continue;

      while (word) {
        int bit = __ffs(word) - 1;
        word &= word - 1;
        int32_t v = w * 32 + bit;
        if (v >= num_vertices) break;

        int32_t start = __ldg(&csr_offsets[v]);
        int32_t end = __ldg(&csr_offsets[v + 1]);
        for (int32_t e = start + lane; e < end; e += 32) {
          int32_t n = __ldg(&csr_indices[e]);
          uint32_t nmask = 1u << (n & 31);
          uint32_t old = atomicOr(&vis[n >> 5], nmask);
          if (!(old & nmask)) {
            atomicOr(&nfrt[n >> 5], nmask);
          }
        }
      }
    }
  }
}

__global__ void count_edges_per_word_kernel(const int32_t* __restrict__ csr_offsets,
                                           const int32_t* __restrict__ csr_indices,
                                           const uint32_t* __restrict__ visited, int64_t* __restrict__ per_word_counts,
                                           int32_t num_vertices, int32_t bitmap_words,
                                           int32_t n_sources) {
  for (int src_idx = (int)blockIdx.y; src_idx < n_sources; src_idx += (int)gridDim.y) {
    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    const uint32_t* vis = visited + (int64_t)src_idx * bitmap_words;
    int64_t count_base = (int64_t)src_idx * bitmap_words;

    int warp_stride = (int)gridDim.x * WARPS_PER_BLOCK;
    for (int w = (int)blockIdx.x * WARPS_PER_BLOCK + warp_in_block; w < bitmap_words; w += warp_stride) {
      uint32_t word = __ldg(&vis[w]);
      int64_t count = 0;

      while (word) {
        int bit = __ffs(word) - 1;
        word &= word - 1;
        int32_t v = w * 32 + bit;
        if (v >= num_vertices) break;

        int32_t start = __ldg(&csr_offsets[v]);
        int32_t end = __ldg(&csr_offsets[v + 1]);

        for (int32_t e_base = start; e_base < end; e_base += 32) {
          int32_t e = e_base + lane;
          bool match = false;
          if (e < end) {
            int32_t n = __ldg(&csr_indices[e]);
            uint32_t wv = __ldg(&vis[n >> 5]);
            match = (wv >> (n & 31)) & 1;
          }
          uint32_t ballot = __ballot_sync(0xffffffff, match);
          count += (int64_t)__popc(ballot);
        }
      }

      if (lane == 0) per_word_counts[count_base + w] = count;
    }
  }
}

__global__ void extract_source_offsets_kernel(const int64_t* __restrict__ per_word_offsets,
                                             int64_t* __restrict__ source_offsets, int32_t n_sources,
                                             int32_t bitmap_words) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx <= n_sources) {
    source_offsets[idx] = per_word_offsets[(int64_t)idx * bitmap_words];
  }
}

__global__ void write_edges_kernel(const int32_t* __restrict__ csr_offsets, const int32_t* __restrict__ csr_indices,
                                  const uint32_t* __restrict__ visited, const int64_t* __restrict__ per_word_offsets,
                                  int32_t* __restrict__ edge_srcs, int32_t* __restrict__ edge_dsts,
                                  int32_t num_vertices, int32_t bitmap_words,
                                  int32_t n_sources) {
  for (int src_idx = (int)blockIdx.y; src_idx < n_sources; src_idx += (int)gridDim.y) {
    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    unsigned lane_mask_lt = (1u << lane) - 1;

    const uint32_t* vis = visited + (int64_t)src_idx * bitmap_words;
    int64_t offset_base = (int64_t)src_idx * bitmap_words;

    int warp_stride = (int)gridDim.x * WARPS_PER_BLOCK;
    for (int w = (int)blockIdx.x * WARPS_PER_BLOCK + warp_in_block; w < bitmap_words; w += warp_stride) {
      uint32_t word = __ldg(&vis[w]);
      if (word == 0) continue;

      int64_t pos = __ldg(&per_word_offsets[offset_base + w]);

      while (word) {
        int bit = __ffs(word) - 1;
        word &= word - 1;
        int32_t v = w * 32 + bit;
        if (v >= num_vertices) break;

        int32_t start = __ldg(&csr_offsets[v]);
        int32_t end = __ldg(&csr_offsets[v + 1]);

        for (int32_t e_base = start; e_base < end; e_base += 32) {
          int32_t e = e_base + lane;
          int32_t n = -1;
          bool match = false;
          if (e < end) {
            n = __ldg(&csr_indices[e]);
            uint32_t wv = __ldg(&vis[n >> 5]);
            match = (wv >> (n & 31)) & 1;
          }

          uint32_t ballot = __ballot_sync(0xffffffff, match);
          int num_matches = __popc(ballot);
          if (match) {
            int my_offset = __popc(ballot & lane_mask_lt);
            edge_srcs[pos + my_offset] = v;
            edge_dsts[pos + my_offset] = n;
          }
          pos += num_matches;
        }
      }
    }
  }
}



static inline int compute_grid_x(int bitmap_words, int n_sources) {
  int max_bps = (bitmap_words + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  int target_blocks = NUM_SMS * 8;  
  int min_bps = (target_blocks + n_sources - 1) / n_sources;
  if (min_bps < 1) min_bps = 1;
  int bps = max_bps;
  int over = min_bps * 4;
  if (bps > over) bps = over;
  if (bps > max_bps) bps = max_bps;
  if (bps < 1) bps = 1;
  return bps;
}



void launch_mark_sources(uint32_t* visited, uint32_t* frontier, const int32_t* sources, int32_t n_sources,
                         int32_t bitmap_words, cudaStream_t stream) {
  if (n_sources <= 0) return;
  int blocks = (n_sources + BLOCK_SIZE - 1) / BLOCK_SIZE;
  mark_sources_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(visited, frontier, sources, n_sources, bitmap_words);
}

void launch_bfs_expand(const int32_t* csr_offsets, const int32_t* csr_indices, uint32_t* visited,
                       const uint32_t* frontier, uint32_t* new_frontier, int32_t n_sources, int32_t num_vertices,
                       int32_t bitmap_words, cudaStream_t stream) {
  if (n_sources <= 0 || bitmap_words <= 0) return;
  int bx = compute_grid_x(bitmap_words, n_sources);
  unsigned gy = (unsigned)n_sources; if (gy > 65535u) gy = 65535u;
  dim3 grid(bx, gy);
  bfs_expand_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(csr_offsets, csr_indices, visited, frontier, new_frontier,
                                                    num_vertices, bitmap_words, n_sources);
}

void launch_count_edges_per_word(const int32_t* csr_offsets, const int32_t* csr_indices, const uint32_t* visited,
                                 int64_t* per_word_counts, int32_t n_sources, int32_t num_vertices,
                                 int32_t bitmap_words, cudaStream_t stream) {
  if (n_sources <= 0 || bitmap_words <= 0) return;
  int bx = compute_grid_x(bitmap_words, n_sources);
  unsigned gy = (unsigned)n_sources; if (gy > 65535u) gy = 65535u;
  dim3 grid(bx, gy);
  count_edges_per_word_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(csr_offsets, csr_indices, visited, per_word_counts,
                                                              num_vertices, bitmap_words, n_sources);
}

void launch_extract_source_offsets(const int64_t* per_word_offsets, int64_t* source_offsets, int32_t n_sources,
                                   int32_t bitmap_words, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n_sources + 1 + threads - 1) / threads;
  extract_source_offsets_kernel<<<blocks, threads, 0, stream>>>(per_word_offsets, source_offsets, n_sources,
                                                                bitmap_words);
}

void launch_write_edges(const int32_t* csr_offsets, const int32_t* csr_indices, const uint32_t* visited,
                        const int64_t* per_word_offsets, int32_t* edge_srcs, int32_t* edge_dsts, int32_t n_sources,
                        int32_t num_vertices, int32_t bitmap_words, cudaStream_t stream) {
  if (n_sources <= 0 || bitmap_words <= 0) return;
  int bx = compute_grid_x(bitmap_words, n_sources);
  unsigned gy = (unsigned)n_sources; if (gy > 65535u) gy = 65535u;
  dim3 grid(bx, gy);
  write_edges_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(csr_offsets, csr_indices, visited, per_word_offsets, edge_srcs,
                                                     edge_dsts, num_vertices, bitmap_words, n_sources);
}



struct Cache : Cacheable {
    uint32_t* visited = nullptr;
    uint32_t* frontier_a = nullptr;
    uint32_t* frontier_b = nullptr;
    int64_t* per_word_counts = nullptr;
    void* cub_temp = nullptr;

    int64_t visited_cap = 0;
    int64_t frontier_a_cap = 0;
    int64_t frontier_b_cap = 0;
    int64_t per_word_cap = 0;
    size_t cub_temp_cap = 0;

    void ensure(int64_t total_bitmap_words, int64_t per_word_size, size_t temp_bytes) {
        if (visited_cap < total_bitmap_words) {
            if (visited) cudaFree(visited);
            cudaMalloc(&visited, total_bitmap_words * sizeof(uint32_t));
            visited_cap = total_bitmap_words;
        }
        if (frontier_a_cap < total_bitmap_words) {
            if (frontier_a) cudaFree(frontier_a);
            cudaMalloc(&frontier_a, total_bitmap_words * sizeof(uint32_t));
            frontier_a_cap = total_bitmap_words;
        }
        if (frontier_b_cap < total_bitmap_words) {
            if (frontier_b) cudaFree(frontier_b);
            cudaMalloc(&frontier_b, total_bitmap_words * sizeof(uint32_t));
            frontier_b_cap = total_bitmap_words;
        }
        if (per_word_cap < per_word_size) {
            if (per_word_counts) cudaFree(per_word_counts);
            cudaMalloc(&per_word_counts, per_word_size * sizeof(int64_t));
            per_word_cap = per_word_size;
        }
        if (cub_temp_cap < temp_bytes) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, temp_bytes);
            cub_temp_cap = temp_bytes;
        }
    }

    ~Cache() override {
        if (visited) cudaFree(visited);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (per_word_counts) cudaFree(per_word_counts);
        if (cub_temp) cudaFree(cub_temp);
    }
};

}  

extract_ego_result_t extract_ego_seg(const graph32_t& graph,
                                     const int32_t* source_vertices,
                                     std::size_t n_sources,
                                     int32_t radius) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t n_src = static_cast<int32_t>(n_sources);

    cudaStream_t stream = 0;

    int32_t bitmap_words = (num_vertices + 31) / 32;
    int64_t total_bitmap_words = (int64_t)n_src * bitmap_words;
    int64_t per_word_size = total_bitmap_words + 1;

    
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, (int64_t*)nullptr, (int64_t*)nullptr, (int)per_word_size);

    
    cache.ensure(total_bitmap_words, per_word_size, temp_bytes);

    
    cudaMemsetAsync(cache.visited, 0, total_bitmap_words * sizeof(uint32_t), stream);
    cudaMemsetAsync(cache.frontier_a, 0, total_bitmap_words * sizeof(uint32_t), stream);

    
    launch_mark_sources(cache.visited, cache.frontier_a, source_vertices, n_src, bitmap_words, stream);

    
    const uint32_t* frt_read = cache.frontier_a;
    uint32_t* frt_write = cache.frontier_b;
    for (int r = 0; r < radius; r++) {
        cudaMemsetAsync(frt_write, 0, total_bitmap_words * sizeof(uint32_t), stream);
        launch_bfs_expand(d_offsets, d_indices, cache.visited, frt_read, frt_write,
                          n_src, num_vertices, bitmap_words, stream);
        const uint32_t* tmp = frt_read;
        frt_read = frt_write;
        frt_write = const_cast<uint32_t*>(tmp);
    }

    
    cudaMemsetAsync(cache.per_word_counts, 0, per_word_size * sizeof(int64_t), stream);
    launch_count_edges_per_word(d_offsets, d_indices, cache.visited, cache.per_word_counts,
                                n_src, num_vertices, bitmap_words, stream);

    
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, temp_bytes, cache.per_word_counts,
                                  cache.per_word_counts, (int)per_word_size, stream);

    
    int64_t total_edges = 0;
    cudaMemcpyAsync(&total_edges, cache.per_word_counts + total_bitmap_words,
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    
    int32_t* out_srcs = nullptr;
    int32_t* out_dsts = nullptr;
    std::size_t* out_offsets = nullptr;

    if (total_edges > 0) {
        cudaMalloc(&out_srcs, total_edges * sizeof(int32_t));
        cudaMalloc(&out_dsts, total_edges * sizeof(int32_t));
    }
    cudaMalloc(&out_offsets, (n_src + 1) * sizeof(std::size_t));

    
    launch_extract_source_offsets(cache.per_word_counts, reinterpret_cast<int64_t*>(out_offsets),
                                  n_src, bitmap_words, stream);

    
    if (total_edges > 0) {
        launch_write_edges(d_offsets, d_indices, cache.visited, cache.per_word_counts,
                           out_srcs, out_dsts, n_src, num_vertices, bitmap_words, stream);
    }

    return extract_ego_result_t{
        out_srcs,
        out_dsts,
        out_offsets,
        static_cast<std::size_t>(total_edges),
        static_cast<std::size_t>(n_src + 1)
    };
}

}  
