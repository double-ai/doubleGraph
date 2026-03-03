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
#include <limits>

namespace aai {

namespace {

static constexpr int32_t INF_DIST = 0x7fffffff;



struct Cache : Cacheable {
    int32_t* frontier_buf = nullptr;
    uint32_t* visited = nullptr;
    uint32_t* frontier_bmp = nullptr;
    uint32_t* new_frontier_bmp = nullptr;
    int32_t* count = nullptr;

    int64_t frontier_buf_capacity = 0;
    int64_t visited_capacity = 0;
    int64_t frontier_bmp_capacity = 0;
    int64_t new_frontier_bmp_capacity = 0;
    bool count_allocated = false;

    void ensure(int32_t num_vertices, bool use_do) {
        int64_t need_frontier = 2LL * num_vertices;
        if (frontier_buf_capacity < need_frontier) {
            if (frontier_buf) cudaFree(frontier_buf);
            cudaMalloc(&frontier_buf, need_frontier * sizeof(int32_t));
            frontier_buf_capacity = need_frontier;
        }

        int32_t num_words = (num_vertices + 31) >> 5;

        if (visited_capacity < num_words) {
            if (visited) cudaFree(visited);
            cudaMalloc(&visited, num_words * sizeof(uint32_t));
            visited_capacity = num_words;
        }

        if (use_do) {
            if (frontier_bmp_capacity < num_words) {
                if (frontier_bmp) cudaFree(frontier_bmp);
                cudaMalloc(&frontier_bmp, num_words * sizeof(uint32_t));
                frontier_bmp_capacity = num_words;
            }
            if (new_frontier_bmp_capacity < num_words) {
                if (new_frontier_bmp) cudaFree(new_frontier_bmp);
                cudaMalloc(&new_frontier_bmp, num_words * sizeof(uint32_t));
                new_frontier_bmp_capacity = num_words;
            }
        }

        if (!count_allocated) {
            cudaMalloc(&count, sizeof(int32_t));
            count_allocated = true;
        }
    }

    ~Cache() override {
        if (frontier_buf) cudaFree(frontier_buf);
        if (visited) cudaFree(visited);
        if (frontier_bmp) cudaFree(frontier_bmp);
        if (new_frontier_bmp) cudaFree(new_frontier_bmp);
        if (count) cudaFree(count);
    }
};



__global__ void bfs_init_kernel(int32_t* __restrict__ distances,
                               uint32_t* __restrict__ visited,
                               int32_t* __restrict__ predecessors,
                               int32_t num_vertices,
                               int32_t num_words,
                               int32_t compute_pred) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < num_vertices; i += stride) {
    distances[i] = INF_DIST;
    if (compute_pred) predecessors[i] = -1;
  }
  for (int i = tid; i < num_words; i += stride) {
    visited[i] = 0u;
  }
}

__global__ void bfs_set_sources_kernel(int32_t* __restrict__ distances,
                                      uint32_t* __restrict__ visited,
                                      int32_t* __restrict__ predecessors,
                                      int32_t* __restrict__ frontier,
                                      const int32_t* __restrict__ sources,
                                      int32_t n_sources,
                                      int32_t compute_pred) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n_sources) {
    int32_t s = sources[tid];
    distances[s] = 0;
    if (compute_pred) predecessors[s] = -1;
    frontier[tid] = s;
    atomicOr(&visited[s >> 5], 1u << (s & 31));
  }
}

__global__ void queue_to_bitmap_kernel(const int32_t* __restrict__ queue,
                                      int32_t queue_size,
                                      uint32_t* __restrict__ bitmap) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < queue_size; i += stride) {
    int32_t v = queue[i];
    atomicOr(&bitmap[v >> 5], 1u << (v & 31));
  }
}

__global__ void bitmap_to_queue_kernel(const uint32_t* __restrict__ bitmap,
                                      int32_t num_vertices,
                                      int32_t* __restrict__ queue,
                                      int32_t* __restrict__ queue_count) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int lane = threadIdx.x & 31;

  int warp_id = tid >> 5;
  int num_warps = stride >> 5;
  int num_words = (num_vertices + 31) >> 5;

  for (int word_idx = warp_id; word_idx < num_words; word_idx += num_warps) {
    uint32_t word = bitmap[word_idx];
    if (!word) continue;

    int base_v = word_idx << 5;
    int v = base_v + lane;
    bool is_set = (v < num_vertices) && ((word >> lane) & 1u);

    unsigned ballot = __ballot_sync(0xffffffffu, is_set);
    if (ballot) {
      int cnt = __popc(ballot);
      int base = 0;
      if (lane == 0) base = atomicAdd(queue_count, cnt);
      base = __shfl_sync(0xffffffffu, base, 0);
      if (is_set) {
        int offset = __popc(ballot & ((1u << lane) - 1u));
        queue[base + offset] = v;
      }
    }
  }
}

__global__ void bfs_topdown_queue_warp_kernel(const int32_t* __restrict__ offsets,
                                             const int32_t* __restrict__ indices,
                                             int32_t* __restrict__ distances,
                                             uint32_t* __restrict__ visited,
                                             int32_t* __restrict__ predecessors,
                                             const int32_t* __restrict__ cur_frontier,
                                             int32_t frontier_size,
                                             int32_t new_depth,
                                             int32_t* __restrict__ next_frontier,
                                             int32_t* __restrict__ next_count,
                                             int32_t compute_pred) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = tid >> 5;
  int lane = tid & 31;
  int total_warps = (blockDim.x * gridDim.x) >> 5;

  for (int i = warp_id; i < frontier_size; i += total_warps) {
    int32_t src = cur_frontier[i];
    int32_t start = offsets[src];
    int32_t end = offsets[src + 1];

    for (int32_t e = start + lane; e < end; e += 32) {
      int32_t dst = indices[e];
      uint32_t mask = 1u << (dst & 31);
      uint32_t* word_ptr = &visited[dst >> 5];

      uint32_t word = __ldg(word_ptr);
      bool is_new = false;
      if ((word & mask) == 0u) {
        uint32_t old = atomicOr(word_ptr, mask);
        is_new = ((old & mask) == 0u);
        if (is_new) {
          distances[dst] = new_depth;
          if (compute_pred) predecessors[dst] = src;
        }
      }

      unsigned ballot = __ballot_sync(0xffffffffu, is_new);
      int n = __popc(ballot);
      int base = 0;
      if (n && lane == 0) {
        base = atomicAdd(next_count, n);
      }
      base = __shfl_sync(0xffffffffu, base, 0);

      if (is_new) {
        int offset = __popc(ballot & ((1u << lane) - 1u));
        next_frontier[base + offset] = dst;
      }
    }
  }
}

__global__ void bfs_bottomup_kernel(const int32_t* __restrict__ offsets,
                                   const int32_t* __restrict__ indices,
                                   int32_t* __restrict__ distances,
                                   uint32_t* __restrict__ visited,
                                   int32_t* __restrict__ predecessors,
                                   const uint32_t* __restrict__ frontier_bitmap,
                                   uint32_t* __restrict__ new_frontier_bitmap,
                                   int32_t* __restrict__ discovered_count,
                                   int32_t num_vertices,
                                   int32_t new_depth,
                                   int32_t compute_pred) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int lane = threadIdx.x & 31;

  for (int v = tid; v < num_vertices; v += stride) {
    uint32_t mask = 1u << (v & 31);
    uint32_t* vis_word_ptr = &visited[v >> 5];
    uint32_t vis_word = __ldg(vis_word_ptr);
    if (vis_word & mask) continue;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    int32_t parent = -1;
    for (int32_t e = start; e < end; e++) {
      int32_t u = indices[e];
      uint32_t fword = __ldg(&frontier_bitmap[u >> 5]);
      if (fword & (1u << (u & 31))) {
        parent = u;
        break;
      }
    }

    bool discovered = (parent >= 0);
    if (discovered) {
      distances[v] = new_depth;
      if (compute_pred) predecessors[v] = parent;
      atomicOr(vis_word_ptr, mask);
      atomicOr(&new_frontier_bitmap[v >> 5], mask);
    }

    
    unsigned ballot = __ballot_sync(0xffffffffu, discovered);
    int n = __popc(ballot);
    if (n && lane == 0) atomicAdd(discovered_count, n);
  }
}



void launch_bfs_init(int32_t* distances, uint32_t* visited, int32_t* predecessors,
                     int32_t num_vertices, int32_t num_words,
                     int32_t compute_pred, cudaStream_t stream) {
  int threads = 256;
  int n = (num_vertices > num_words) ? num_vertices : num_words;
  int blocks = (n + threads - 1) / threads;
  if (blocks < 1) blocks = 1;
  if (blocks > 8192) blocks = 8192;
  bfs_init_kernel<<<blocks, threads, 0, stream>>>(distances, visited, predecessors, num_vertices, num_words, compute_pred);
}

void launch_bfs_set_sources(int32_t* distances, uint32_t* visited, int32_t* predecessors,
                            int32_t* frontier, const int32_t* sources, int32_t n_sources,
                            int32_t compute_pred, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n_sources + threads - 1) / threads;
  if (blocks < 1) blocks = 1;
  bfs_set_sources_kernel<<<blocks, threads, 0, stream>>>(distances, visited, predecessors, frontier, sources, n_sources, compute_pred);
}

void launch_bfs_topdown_queue(const int32_t* offsets, const int32_t* indices,
                              int32_t* distances, uint32_t* visited, int32_t* predecessors,
                              const int32_t* cur_frontier, int32_t frontier_size,
                              int32_t new_depth,
                              int32_t* next_frontier, int32_t* next_count,
                              int32_t compute_pred, cudaStream_t stream) {
  int threads = 256;
  int warps_per_block = threads / 32;
  int blocks = (frontier_size + warps_per_block - 1) / warps_per_block;
  if (blocks < 1) blocks = 1;
  if (blocks > 4096) blocks = 4096;
  bfs_topdown_queue_warp_kernel<<<blocks, threads, 0, stream>>>(offsets, indices, distances, visited, predecessors,
                                                               cur_frontier, frontier_size, new_depth,
                                                               next_frontier, next_count, compute_pred);
}

void launch_queue_to_bitmap(const int32_t* queue, int32_t queue_size, uint32_t* bitmap, cudaStream_t stream) {
  int threads = 256;
  int blocks = (queue_size + threads - 1) / threads;
  if (blocks < 1) blocks = 1;
  if (blocks > 4096) blocks = 4096;
  queue_to_bitmap_kernel<<<blocks, threads, 0, stream>>>(queue, queue_size, bitmap);
}

void launch_bitmap_to_queue(const uint32_t* bitmap, int32_t num_vertices,
                            int32_t* queue, int32_t* queue_count, cudaStream_t stream) {
  int threads = 256;
  int num_words = (num_vertices + 31) >> 5;
  int blocks = (num_words + (threads - 1)) / threads;
  if (blocks < 1) blocks = 1;
  if (blocks > 4096) blocks = 4096;
  bitmap_to_queue_kernel<<<blocks, threads, 0, stream>>>(bitmap, num_vertices, queue, queue_count);
}

void launch_bfs_bottomup(const int32_t* offsets, const int32_t* indices,
                          int32_t* distances, uint32_t* visited, int32_t* predecessors,
                          const uint32_t* frontier_bitmap, uint32_t* new_frontier_bitmap,
                          int32_t* discovered_count,
                          int32_t num_vertices, int32_t new_depth,
                          int32_t compute_pred, cudaStream_t stream) {
  int threads = 256;
  int blocks = (num_vertices + threads - 1) / threads;
  if (blocks < 1) blocks = 1;
  if (blocks > 65535) blocks = 65535;
  bfs_bottomup_kernel<<<blocks, threads, 0, stream>>>(offsets, indices, distances, visited, predecessors,
                                                     frontier_bitmap, new_frontier_bitmap,
                                                     discovered_count, num_vertices, new_depth, compute_pred);
}

}  



void bfs(const graph32_t& graph,
         int32_t* distances,
         int32_t* predecessors,
         const int32_t* sources,
         std::size_t n_sources,
         int32_t depth_limit) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    bool is_symmetric = graph.is_symmetric;

    bool compute_pred = (predecessors != nullptr);
    int32_t cp = compute_pred ? 1 : 0;
    cudaStream_t stream = 0;

    int32_t num_words = (num_vertices + 31) >> 5;

    bool use_do = is_symmetric && (num_vertices >= 10000);
    cache.ensure(num_vertices, use_do);

    int32_t* d_frontier0 = cache.frontier_buf;
    int32_t* d_frontier1 = cache.frontier_buf + num_vertices;
    int32_t* d_frontier[2] = {d_frontier0, d_frontier1};

    uint32_t* d_visited = cache.visited;

    uint32_t* d_frontier_bmp = cache.frontier_bmp;
    uint32_t* d_new_frontier_bmp = cache.new_frontier_bmp;

    int32_t* d_count = cache.count;

    int32_t* d_dist = distances;
    int32_t* d_pred = compute_pred ? predecessors : nullptr;

    
    launch_bfs_init(d_dist, d_visited, d_pred, num_vertices, num_words, cp, stream);
    launch_bfs_set_sources(d_dist, d_visited, d_pred, d_frontier[0], sources, static_cast<int32_t>(n_sources), cp, stream);

    int32_t fsize = static_cast<int32_t>(n_sources);
    int cur = 0;
    int32_t depth = 0;
    int32_t max_depth = (depth_limit < 0) ? std::numeric_limits<int32_t>::max() : depth_limit;

    bool topdown = true;
    int32_t prev_fsize = 0;

    int32_t td_to_bu = num_vertices / 20;
    if (td_to_bu < 1) td_to_bu = 1;
    int32_t bu_to_td = num_vertices / 200;
    if (bu_to_td < 1) bu_to_td = 1;

    while (fsize > 0 && depth < max_depth) {
      int32_t new_depth = depth + 1;

      if (topdown) {
        cudaMemsetAsync(d_count, 0, sizeof(int32_t), stream);

        launch_bfs_topdown_queue(d_off, d_idx, d_dist, d_visited, d_pred,
                                 d_frontier[cur], fsize, new_depth,
                                 d_frontier[1 - cur], d_count, cp, stream);

        int32_t next_size = 0;
        cudaMemcpyAsync(&next_size, d_count, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cur = 1 - cur;
        prev_fsize = fsize;
        fsize = next_size;
        depth++;

        if (use_do && fsize > td_to_bu && fsize >= prev_fsize) {
          topdown = false;
          cudaMemsetAsync(d_frontier_bmp, 0, static_cast<size_t>(num_words) * sizeof(uint32_t), stream);
          launch_queue_to_bitmap(d_frontier[cur], fsize, d_frontier_bmp, stream);
        }
      } else {
        cudaMemsetAsync(d_count, 0, sizeof(int32_t), stream);
        cudaMemsetAsync(d_new_frontier_bmp, 0, static_cast<size_t>(num_words) * sizeof(uint32_t), stream);

        launch_bfs_bottomup(d_off, d_idx, d_dist, d_visited, d_pred,
                            d_frontier_bmp, d_new_frontier_bmp,
                            d_count, num_vertices, new_depth, cp, stream);

        int32_t next_size = 0;
        cudaMemcpyAsync(&next_size, d_count, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        auto* tmp = d_frontier_bmp;
        d_frontier_bmp = d_new_frontier_bmp;
        d_new_frontier_bmp = tmp;

        prev_fsize = fsize;
        fsize = next_size;
        depth++;

        if (fsize > 0 && fsize < bu_to_td && fsize < prev_fsize) {
          topdown = true;
          cudaMemsetAsync(d_count, 0, sizeof(int32_t), stream);
          launch_bitmap_to_queue(d_frontier_bmp, num_vertices, d_frontier[0], d_count, stream);

          int32_t qsize = 0;
          cudaMemcpyAsync(&qsize, d_count, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
          cudaStreamSynchronize(stream);

          cur = 0;
          fsize = qsize;
        }
      }
    }
}

}  
