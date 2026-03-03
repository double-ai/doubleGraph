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
#include <limits>

namespace aai {

namespace {

struct Cache : Cacheable {
    uint32_t* visited_bitmap = nullptr;
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* frontier_size_d = nullptr;
    int32_t capacity = 0;

    void ensure(int32_t num_vertices) {
        if (capacity < num_vertices) {
            if (visited_bitmap) cudaFree(visited_bitmap);
            if (frontier_a) cudaFree(frontier_a);
            if (frontier_b) cudaFree(frontier_b);
            if (frontier_size_d) cudaFree(frontier_size_d);

            int32_t bitmap_words = (num_vertices + 31) / 32;
            cudaMalloc(&visited_bitmap, bitmap_words * sizeof(uint32_t));
            cudaMalloc(&frontier_a, num_vertices * sizeof(int32_t));
            cudaMalloc(&frontier_b, num_vertices * sizeof(int32_t));
            cudaMalloc(&frontier_size_d, sizeof(int32_t));

            capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (visited_bitmap) cudaFree(visited_bitmap);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (frontier_size_d) cudaFree(frontier_size_d);
    }
};


__global__ void init_bfs_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    int32_t num_vertices,
    int32_t bitmap_words,
    bool compute_pred
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < num_vertices; i += stride) {
        distances[i] = 0x7FFFFFFF;
        if (compute_pred) predecessors[i] = -1;
    }
    for (int i = idx; i < bitmap_words; i += stride) {
        visited_bitmap[i] = 0;
    }
}


__global__ void set_sources_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    int32_t* __restrict__ frontier,
    const int32_t* __restrict__ sources,
    int32_t n_sources,
    bool compute_pred
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_sources) {
        int32_t src = sources[idx];
        distances[src] = 0;
        if (compute_pred) predecessors[src] = -1;
        frontier[idx] = src;
        atomicOr(&visited_bitmap[src >> 5], 1U << (src & 31));
    }
}



__global__ void bfs_top_down_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t frontier_size,
    int32_t new_distance,
    bool compute_pred
) {
    const int lane_id = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int global_warp_id = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int total_warps = gridDim.x * warps_per_block;

    for (int wi = global_warp_id; wi < frontier_size; wi += total_warps) {
        int32_t v = frontier[wi];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        for (int32_t e = start + lane_id; e < end; e += 32) {
            int32_t neighbor = indices[e];
            int32_t word_idx = neighbor >> 5;
            uint32_t bit_mask = 1U << (neighbor & 31);

            
            if (!(visited_bitmap[word_idx] & bit_mask)) {
                
                int32_t old = atomicCAS(&distances[neighbor], 0x7FFFFFFF, new_distance);
                if (old == 0x7FFFFFFF) {
                    if (compute_pred) predecessors[neighbor] = v;
                    atomicOr(&visited_bitmap[word_idx], bit_mask);
                    int32_t pos = atomicAdd(next_frontier_size, 1);
                    next_frontier[pos] = neighbor;
                }
            }
        }
    }
}


__global__ void bfs_bottom_up_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    const uint32_t* __restrict__ visited_bitmap,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t num_vertices,
    int32_t new_distance,
    bool compute_pred
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int32_t v = tid; v < num_vertices; v += stride) {
        
        if (distances[v] != 0x7FFFFFFF) continue;

        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        if (start == end) continue;

        for (int32_t e = start; e < end; e++) {
            int32_t neighbor = indices[e];
            if (visited_bitmap[neighbor >> 5] & (1U << (neighbor & 31))) {
                distances[v] = new_distance;
                if (compute_pred) predecessors[v] = neighbor;
                int32_t pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = v;
                break;  
            }
        }
    }
}


__global__ void update_bitmap_kernel(
    uint32_t* __restrict__ visited_bitmap,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < frontier_size; i += stride) {
        int32_t v = frontier[i];
        atomicOr(&visited_bitmap[v >> 5], 1U << (v & 31));
    }
}

void launch_init_bfs(int32_t* distances, int32_t* predecessors,
                     uint32_t* visited_bitmap,
                     int32_t num_vertices, int32_t bitmap_words,
                     bool compute_pred, cudaStream_t stream) {
    int block = 256;
    int items = num_vertices > bitmap_words ? num_vertices : bitmap_words;
    int grid = (items + block - 1) / block;
    init_bfs_kernel<<<grid, block, 0, stream>>>(
        distances, predecessors, visited_bitmap,
        num_vertices, bitmap_words, compute_pred);
}

void launch_set_sources(int32_t* distances, int32_t* predecessors,
                        uint32_t* visited_bitmap, int32_t* frontier,
                        const int32_t* sources, int32_t n_sources,
                        bool compute_pred, cudaStream_t stream) {
    int block = 256;
    int grid = (n_sources + block - 1) / block;
    if (grid < 1) grid = 1;
    set_sources_kernel<<<grid, block, 0, stream>>>(
        distances, predecessors, visited_bitmap, frontier,
        sources, n_sources, compute_pred);
}

void launch_bfs_top_down(
    const int32_t* offsets, const int32_t* indices,
    int32_t* distances, int32_t* predecessors,
    uint32_t* visited_bitmap,
    const int32_t* frontier, int32_t* next_frontier,
    int32_t* next_frontier_size,
    int32_t frontier_size, int32_t new_distance,
    bool compute_pred, cudaStream_t stream) {
    if (frontier_size == 0) return;
    int block = 256;
    int warps_per_block = block / 32;
    int needed = (frontier_size + warps_per_block - 1) / warps_per_block;
    
    int grid = needed < 864 ? needed : 864;
    bfs_top_down_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, distances, predecessors,
        visited_bitmap, frontier, next_frontier,
        next_frontier_size, frontier_size, new_distance, compute_pred);
}

void launch_bfs_bottom_up(
    const int32_t* offsets, const int32_t* indices,
    int32_t* distances, int32_t* predecessors,
    const uint32_t* visited_bitmap,
    int32_t* next_frontier, int32_t* next_frontier_size,
    int32_t num_vertices, int32_t new_distance,
    bool compute_pred, cudaStream_t stream) {
    int block = 512;
    int needed = (num_vertices + block - 1) / block;
    int grid = needed < 1024 ? needed : 1024;
    bfs_bottom_up_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, distances, predecessors,
        visited_bitmap, next_frontier, next_frontier_size,
        num_vertices, new_distance, compute_pred);
}

void launch_update_bitmap(
    uint32_t* visited_bitmap,
    const int32_t* frontier, int32_t frontier_size,
    cudaStream_t stream) {
    if (frontier_size == 0) return;
    int block = 256;
    int grid = (frontier_size + block - 1) / block;
    if (grid > 864) grid = 864;
    update_bitmap_kernel<<<grid, block, 0, stream>>>(
        visited_bitmap, frontier, frontier_size);
}

}  

void bfs_direction_optimizing_seg(const graph32_t& graph,
                                  int32_t* distances,
                                  int32_t* predecessors,
                                  const int32_t* sources,
                                  std::size_t n_sources,
                                  int32_t depth_limit) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    bool compute_pred = (predecessors != nullptr);

    if (depth_limit < 0) depth_limit = std::numeric_limits<int32_t>::max();

    cache.ensure(num_vertices);

    int32_t bitmap_words = (num_vertices + 31) / 32;
    uint32_t* d_visited_bitmap = cache.visited_bitmap;
    int32_t* d_frontier_a = cache.frontier_a;
    int32_t* d_frontier_b = cache.frontier_b;
    int32_t* d_frontier_size = cache.frontier_size_d;

    cudaStream_t stream = 0;

    
    launch_init_bfs(distances, predecessors, d_visited_bitmap,
                    num_vertices, bitmap_words, compute_pred, stream);
    launch_set_sources(distances, predecessors, d_visited_bitmap,
                       d_frontier_a, sources, static_cast<int32_t>(n_sources),
                       compute_pred, stream);

    int32_t* current_frontier = d_frontier_a;
    int32_t* next_frontier = d_frontier_b;
    int32_t frontier_size = static_cast<int32_t>(n_sources);
    int64_t visited_count = static_cast<int64_t>(n_sources);

    
    float avg_degree = (num_vertices > 0) ? static_cast<float>(num_edges) / num_vertices : 0.0f;
    float alpha = avg_degree * 0.3f;
    constexpr int32_t beta = 24;

    bool is_top_down = true;
    int32_t current_distance = 0;
    int32_t h_frontier_size = 0;

    while (frontier_size > 0) {
        cudaMemsetAsync(d_frontier_size, 0, sizeof(int32_t), stream);

        int32_t new_distance = current_distance + 1;

        if (is_top_down) {
            launch_bfs_top_down(d_offsets, d_indices, distances, predecessors,
                                d_visited_bitmap, current_frontier, next_frontier,
                                d_frontier_size, frontier_size, new_distance,
                                compute_pred, stream);
        } else {
            launch_bfs_bottom_up(d_offsets, d_indices, distances, predecessors,
                                 d_visited_bitmap, next_frontier, d_frontier_size,
                                 num_vertices, new_distance,
                                 compute_pred, stream);
        }

        
        cudaMemcpyAsync(&h_frontier_size, d_frontier_size, sizeof(int32_t),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        int32_t next_frontier_size_val = h_frontier_size;

        
        if (!is_top_down && next_frontier_size_val > 0) {
            launch_update_bitmap(d_visited_bitmap, next_frontier,
                                 next_frontier_size_val, stream);
        }

        
        visited_count += next_frontier_size_val;
        int64_t unvisited_count = num_vertices - visited_count;

        
        int32_t prev_frontier_size = frontier_size;
        frontier_size = next_frontier_size_val;

        if (is_top_down) {
            if (static_cast<float>(frontier_size) * alpha > static_cast<float>(unvisited_count) &&
                frontier_size >= prev_frontier_size) {
                is_top_down = false;
            }
        } else {
            if (static_cast<int64_t>(frontier_size) * beta < unvisited_count &&
                frontier_size < prev_frontier_size) {
                is_top_down = true;
            }
        }

        std::swap(current_frontier, next_frontier);
        current_distance = new_distance;
        if (depth_limit != std::numeric_limits<int32_t>::max() && current_distance >= depth_limit) break;
    }
}

}  
