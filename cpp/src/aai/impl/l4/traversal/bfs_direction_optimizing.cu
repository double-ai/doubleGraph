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

static constexpr int32_t BFS_INF = 0x7FFFFFFF;





struct Cache : Cacheable {
    int32_t* d_frontier[2] = {nullptr, nullptr};
    int32_t* d_counter_dev = nullptr;
    uint32_t* d_visited = nullptr;
    int32_t* h_counter = nullptr;
    size_t alloc_vertices = 0;

    Cache() {
        alloc_vertices = 8 * 1024 * 1024;
        cudaMalloc(&d_frontier[0], alloc_vertices * sizeof(int32_t));
        cudaMalloc(&d_frontier[1], alloc_vertices * sizeof(int32_t));
        cudaMalloc(&d_counter_dev, sizeof(int32_t));
        cudaMalloc(&d_visited, ((alloc_vertices + 31) / 32) * sizeof(uint32_t));
        cudaMallocHost(&h_counter, sizeof(int32_t));
    }

    void ensure_capacity(size_t num_vertices) {
        if (num_vertices <= alloc_vertices) return;
        cudaFree(d_frontier[0]); cudaFree(d_frontier[1]); cudaFree(d_visited);
        alloc_vertices = num_vertices;
        cudaMalloc(&d_frontier[0], alloc_vertices * sizeof(int32_t));
        cudaMalloc(&d_frontier[1], alloc_vertices * sizeof(int32_t));
        cudaMalloc(&d_visited, ((alloc_vertices + 31) / 32) * sizeof(uint32_t));
    }

    ~Cache() override {
        if (d_frontier[0]) cudaFree(d_frontier[0]);
        if (d_frontier[1]) cudaFree(d_frontier[1]);
        if (d_counter_dev) cudaFree(d_counter_dev);
        if (d_visited) cudaFree(d_visited);
        if (h_counter) cudaFreeHost(h_counter);
    }
};





__global__ void bfs_init_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    int32_t num_vertices,
    int32_t bitmap_words,
    bool has_predecessors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < bitmap_words; i += stride)
        visited[i] = 0;
    for (int i = tid; i < num_vertices; i += stride) {
        distances[i] = BFS_INF;
        if (has_predecessors) predecessors[i] = -1;
    }
}

__global__ void bfs_set_sources_kernel(
    const int32_t* __restrict__ sources,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t* __restrict__ frontier,
    uint32_t* __restrict__ visited,
    int32_t n_sources,
    bool has_predecessors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_sources) return;
    int32_t s = sources[tid];
    distances[s] = 0;
    if (has_predecessors) predecessors[s] = -1;
    frontier[tid] = s;
    atomicOr(&visited[s >> 5], 1u << (s & 31));
}


__global__ __launch_bounds__(256, 6)
void bfs_topdown_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_count,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    int32_t frontier_size,
    int32_t new_dist,
    bool has_predecessors
) {
    const int lane = threadIdx.x & 31;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int total_warps = (gridDim.x * blockDim.x) >> 5;

    for (int vi = warp_id; vi < frontier_size; vi += total_warps) {
        int32_t v = frontier[vi];
        int32_t start = __ldg(&offsets[v]);
        int32_t end = __ldg(&offsets[v + 1]);

        for (int32_t e = start + lane; e < end; e += 32) {
            int32_t u = __ldg(&indices[e]);
            uint32_t word_idx = u >> 5;
            uint32_t bit = 1u << (u & 31);

            if (!(__ldg(&visited[word_idx]) & bit)) {
                uint32_t old = atomicOr(&visited[word_idx], bit);
                if (!(old & bit)) {
                    distances[u] = new_dist;
                    if (has_predecessors) predecessors[u] = v;
                    int32_t pos = atomicAdd(next_frontier_count, 1);
                    next_frontier[pos] = u;
                }
            }
        }
    }
}


__global__ void bfs_topdown_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_count,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    int32_t frontier_size,
    int32_t new_dist,
    bool has_predecessors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int32_t v = frontier[tid];
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    for (int32_t e = start; e < end; e++) {
        int32_t u = indices[e];
        uint32_t word_idx = u >> 5;
        uint32_t bit = 1u << (u & 31);
        if (!(visited[word_idx] & bit)) {
            uint32_t old = atomicOr(&visited[word_idx], bit);
            if (!(old & bit)) {
                distances[u] = new_dist;
                if (has_predecessors) predecessors[u] = v;
                int32_t pos = atomicAdd(next_frontier_count, 1);
                next_frontier[pos] = u;
            }
        }
    }
}


__global__ __launch_bounds__(512, 3)
void bfs_bottomup_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ visited,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_count,
    int32_t num_vertices,
    int32_t new_dist,
    bool has_predecessors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int v = tid; v < num_vertices; v += stride) {
        uint32_t v_word = __ldg(&visited[v >> 5]);
        if ((v_word >> (v & 31)) & 1) continue;

        int32_t start = __ldg(&offsets[v]);
        int32_t end = __ldg(&offsets[v + 1]);
        if (start >= end) continue;

        for (int32_t e = start; e < end; e++) {
            int32_t u = __ldg(&indices[e]);
            uint32_t u_word = __ldg(&visited[u >> 5]);
            if ((u_word >> (u & 31)) & 1) {
                distances[v] = new_dist;
                if (has_predecessors) predecessors[v] = u;
                int32_t pos = atomicAdd(next_frontier_count, 1);
                next_frontier[pos] = v;
                break;
            }
        }
    }
}

__global__ void update_bitmap_kernel(
    const int32_t* __restrict__ frontier,
    uint32_t* __restrict__ visited,
    int32_t frontier_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int32_t v = frontier[tid];
    atomicOr(&visited[v >> 5], 1u << (v & 31));
}





void launch_bfs_init(int32_t* distances, int32_t* predecessors, uint32_t* visited,
                     int32_t num_vertices, bool has_predecessors, cudaStream_t stream) {
    int32_t bitmap_words = (num_vertices + 31) / 32;
    int block = 256;
    int n = (num_vertices > bitmap_words) ? num_vertices : bitmap_words;
    int grid = (n + block - 1) / block;
    if (grid > 2048) grid = 2048;
    bfs_init_kernel<<<grid, block, 0, stream>>>(distances, predecessors, visited,
                                                  num_vertices, bitmap_words, has_predecessors);
}

void launch_bfs_set_sources(const int32_t* sources, int32_t* distances, int32_t* predecessors,
                           int32_t* frontier, uint32_t* visited,
                           int32_t n_sources, bool has_predecessors, cudaStream_t stream) {
    int block = 256;
    int grid = (n_sources + block - 1) / block;
    if (grid < 1) grid = 1;
    bfs_set_sources_kernel<<<grid, block, 0, stream>>>(sources, distances, predecessors,
                                                        frontier, visited, n_sources, has_predecessors);
}

void launch_bfs_topdown(
    const int32_t* offsets, const int32_t* indices,
    const int32_t* frontier, int32_t* next_frontier, int32_t* next_frontier_count,
    int32_t* distances, int32_t* predecessors, uint32_t* visited,
    int32_t frontier_size, int32_t new_dist, bool has_predecessors, cudaStream_t stream
) {
    if (frontier_size == 0) return;
    int block = 256;

    if (frontier_size <= 32) {
        bfs_topdown_thread_kernel<<<1, block, 0, stream>>>(offsets, indices, frontier, next_frontier,
            next_frontier_count, distances, predecessors, visited, frontier_size, new_dist, has_predecessors);
    } else {
        int warps = frontier_size;
        int64_t threads = (int64_t)warps * 32;
        int grid = (int)((threads + block - 1) / block);
        if (grid > 16384) grid = 16384;
        bfs_topdown_warp_kernel<<<grid, block, 0, stream>>>(offsets, indices, frontier, next_frontier,
            next_frontier_count, distances, predecessors, visited, frontier_size, new_dist, has_predecessors);
    }
}

void launch_bfs_bottomup(
    const int32_t* offsets, const int32_t* indices, const uint32_t* visited,
    int32_t* distances, int32_t* predecessors,
    int32_t* next_frontier, int32_t* next_frontier_count,
    int32_t num_vertices, int32_t new_dist, bool has_predecessors, cudaStream_t stream
) {
    int block = 512;
    int grid = (num_vertices + block - 1) / block;
    if (grid > 16384) grid = 16384;
    bfs_bottomup_kernel<<<grid, block, 0, stream>>>(offsets, indices, visited, distances, predecessors,
        next_frontier, next_frontier_count, num_vertices, new_dist, has_predecessors);
}

void launch_update_bitmap(const int32_t* frontier, uint32_t* visited,
                          int32_t frontier_size, cudaStream_t stream) {
    if (frontier_size == 0) return;
    int block = 256;
    int grid = (frontier_size + block - 1) / block;
    update_bitmap_kernel<<<grid, block, 0, stream>>>(frontier, visited, frontier_size);
}

}  





void bfs_direction_optimizing(const graph32_t& graph,
                              int32_t* distances,
                              int32_t* predecessors,
                              const int32_t* sources,
                              std::size_t n_sources,
                              int32_t depth_limit) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    bool compute_predecessors = (predecessors != nullptr);

    if (depth_limit < 0) depth_limit = std::numeric_limits<int32_t>::max();

    cache.ensure_capacity((size_t)num_vertices);
    cudaStream_t stream = 0;

    launch_bfs_init(distances, predecessors, cache.d_visited, num_vertices, compute_predecessors, stream);
    launch_bfs_set_sources(sources, distances, predecessors, cache.d_frontier[0], cache.d_visited,
                          (int32_t)n_sources, compute_predecessors, stream);

    int32_t frontier_size = (int32_t)n_sources;
    int cur = 0;

    
    double avg_degree = (num_vertices > 0) ? (double)num_edges / num_vertices : 0.0;
    double alpha = avg_degree * 0.25;
    if (alpha < 2.0) alpha = 2.0;
    int32_t beta = 24;
    bool topdown = true;
    int64_t total_visited = (int64_t)n_sources;

    for (int32_t depth = 0; frontier_size > 0; depth++) {
        cudaMemsetAsync(cache.d_counter_dev, 0, sizeof(int32_t), stream);

        int32_t new_dist = depth + 1;
        int next = 1 - cur;

        if (topdown) {
            launch_bfs_topdown(
                d_offsets, d_indices,
                cache.d_frontier[cur], cache.d_frontier[next], cache.d_counter_dev,
                distances, predecessors, cache.d_visited,
                frontier_size, new_dist, compute_predecessors, stream);
        } else {
            launch_bfs_bottomup(
                d_offsets, d_indices, cache.d_visited,
                distances, predecessors,
                cache.d_frontier[next], cache.d_counter_dev,
                num_vertices, new_dist, compute_predecessors, stream);
        }

        cudaMemcpyAsync(cache.h_counter, cache.d_counter_dev, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int32_t next_frontier_size = *cache.h_counter;

        if (!topdown && next_frontier_size > 0) {
            launch_update_bitmap(cache.d_frontier[next], cache.d_visited, next_frontier_size, stream);
        }

        total_visited += next_frontier_size;
        int64_t unvisited = (int64_t)num_vertices - total_visited;

        
        if (topdown) {
            if (next_frontier_size >= frontier_size && unvisited > 0) {
                if ((double)next_frontier_size * alpha > (double)unvisited) {
                    topdown = false;
                }
            }
            
            if (topdown && (double)next_frontier_size > (double)num_vertices * 0.10) {
                topdown = false;
            }
        } else {
            if (next_frontier_size < frontier_size) {
                if ((int64_t)next_frontier_size * beta < unvisited) {
                    topdown = true;
                }
            }
        }

        frontier_size = next_frontier_size;
        cur = next;
        if (depth_limit != std::numeric_limits<int32_t>::max() && depth + 1 >= depth_limit) break;
    }
}

}  
