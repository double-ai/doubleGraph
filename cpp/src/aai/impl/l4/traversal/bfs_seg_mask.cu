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

struct Cache : Cacheable {
    int32_t* h_frontier_size = nullptr;
    uint32_t* visited = nullptr;
    int32_t* frontier1 = nullptr;
    int32_t* frontier2 = nullptr;
    int32_t* counter = nullptr;

    int64_t visited_capacity = 0;
    int64_t frontier_capacity = 0;
    int64_t counter_capacity = 0;

    Cache() {
        cudaMallocHost(&h_frontier_size, sizeof(int32_t));
    }

    void ensure(int32_t num_vertices) {
        int32_t bitmap_size = (num_vertices + 31) / 32;

        if (visited_capacity < bitmap_size) {
            if (visited) cudaFree(visited);
            cudaMalloc(&visited, (size_t)bitmap_size * sizeof(uint32_t));
            visited_capacity = bitmap_size;
        }
        if (frontier_capacity < num_vertices) {
            if (frontier1) cudaFree(frontier1);
            if (frontier2) cudaFree(frontier2);
            cudaMalloc(&frontier1, (size_t)num_vertices * sizeof(int32_t));
            cudaMalloc(&frontier2, (size_t)num_vertices * sizeof(int32_t));
            frontier_capacity = num_vertices;
        }
        if (counter_capacity < 1) {
            if (counter) cudaFree(counter);
            cudaMalloc(&counter, sizeof(int32_t));
            counter_capacity = 1;
        }
    }

    ~Cache() override {
        if (h_frontier_size) cudaFreeHost(h_frontier_size);
        if (visited) cudaFree(visited);
        if (frontier1) cudaFree(frontier1);
        if (frontier2) cudaFree(frontier2);
        if (counter) cudaFree(counter);
    }
};

__global__ void init_distances_kernel(int32_t* __restrict__ distances, int32_t num_vertices) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < num_vertices; i += stride) {
        distances[i] = 0x7FFFFFFF;
    }
}

__global__ void setup_sources_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    int32_t* __restrict__ frontier,
    const int32_t* __restrict__ sources,
    int32_t n_sources,
    bool compute_predecessors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_sources) return;
    int32_t src = sources[tid];
    distances[src] = 0;
    if (compute_predecessors) predecessors[src] = -1;
    atomicOr(&visited[src >> 5], 1u << (src & 31));
    frontier[tid] = src;
}

template <bool COMPUTE_PRED>
__global__ void bfs_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t frontier_size,
    int32_t new_dist
) {
    const int lane = threadIdx.x & 31;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int num_warps = (gridDim.x * blockDim.x) >> 5;

    for (int fi = warp_id; fi < frontier_size; fi += num_warps) {
        int32_t v = frontier[fi];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        for (int32_t base = start; base < end; base += 32) {
            int32_t e = base + lane;
            bool discovered = false;
            int32_t neighbor = -1;

            if (e < end) {
                if ((edge_mask[e >> 5] >> (e & 31)) & 1u) {
                    neighbor = indices[e];
                    uint32_t bit = 1u << (neighbor & 31);
                    uint32_t word_idx = neighbor >> 5;
                    if (!(visited[word_idx] & bit)) {
                        uint32_t old = atomicOr(&visited[word_idx], bit);
                        if (!(old & bit)) {
                            discovered = true;
                            distances[neighbor] = new_dist;
                            if constexpr (COMPUTE_PRED) {
                                predecessors[neighbor] = v;
                            }
                        }
                    }
                }
            }

            uint32_t disc_mask = __ballot_sync(0xFFFFFFFF, discovered);
            if (disc_mask) {
                int32_t base_pos;
                if (lane == 0) {
                    base_pos = atomicAdd(next_frontier_size, __popc(disc_mask));
                }
                base_pos = __shfl_sync(0xFFFFFFFF, base_pos, 0);
                if (discovered) {
                    int my_offset = __popc(disc_mask & ((1u << lane) - 1));
                    next_frontier[base_pos + my_offset] = neighbor;
                }
            }
        }
    }
}

void do_init(int32_t* distances, int32_t* predecessors, uint32_t* visited,
    int32_t num_vertices, bool compute_predecessors, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_vertices + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    init_distances_kernel<<<blocks, threads, 0, stream>>>(distances, num_vertices);
    if (compute_predecessors) {
        cudaMemsetAsync(predecessors, 0xFF, (size_t)num_vertices * sizeof(int32_t), stream);
    }
    int32_t bitmap_size = (num_vertices + 31) / 32;
    cudaMemsetAsync(visited, 0, (size_t)bitmap_size * sizeof(uint32_t), stream);
}

void do_setup_sources(int32_t* distances, int32_t* predecessors,
    uint32_t* visited, int32_t* frontier, const int32_t* sources,
    int32_t n_sources, bool compute_predecessors, cudaStream_t stream) {
    if (n_sources == 0) return;
    int threads = 256;
    int blocks = (n_sources + threads - 1) / threads;
    setup_sources_kernel<<<blocks, threads, 0, stream>>>(
        distances, predecessors, visited, frontier, sources, n_sources, compute_predecessors);
}

void do_bfs_warp(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    int32_t* distances, int32_t* predecessors, uint32_t* visited,
    const int32_t* frontier, int32_t* next_frontier, int32_t* next_frontier_size,
    int32_t frontier_size, int32_t new_dist, bool compute_predecessors, cudaStream_t stream) {

    if (frontier_size == 0) return;
    int threads = 256;
    int blocks = (frontier_size + 7) / 8;
    if (blocks > 58 * 8) blocks = 58 * 8;

    if (compute_predecessors) {
        bfs_warp_kernel<true><<<blocks, threads, 0, stream>>>(
            offsets, indices, edge_mask, distances, predecessors, visited,
            frontier, next_frontier, next_frontier_size, frontier_size, new_dist);
    } else {
        bfs_warp_kernel<false><<<blocks, threads, 0, stream>>>(
            offsets, indices, edge_mask, distances, predecessors, visited,
            frontier, next_frontier, next_frontier_size, frontier_size, new_dist);
    }
}

}  

void bfs_seg_mask(const graph32_t& graph,
                  int32_t* distances,
                  int32_t* predecessors,
                  const int32_t* sources,
                  std::size_t n_sources,
                  int32_t depth_limit) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    const uint32_t* edge_mask = graph.edge_mask;

    bool compute_predecessors = (predecessors != nullptr);

    if (depth_limit < 0) depth_limit = std::numeric_limits<int32_t>::max();

    cache.ensure(num_vertices);

    cudaStream_t stream = 0;

    do_init(distances, predecessors, cache.visited, num_vertices, compute_predecessors, stream);
    do_setup_sources(distances, predecessors, cache.visited, cache.frontier1,
        sources, static_cast<int32_t>(n_sources), compute_predecessors, stream);

    int32_t frontier_size = static_cast<int32_t>(n_sources);
    int32_t current_depth = 0;
    int32_t* cur_f = cache.frontier1;
    int32_t* nxt_f = cache.frontier2;

    while (frontier_size > 0 && current_depth < depth_limit) {
        cudaMemsetAsync(cache.counter, 0, sizeof(int32_t), stream);

        do_bfs_warp(offsets, indices, edge_mask,
            distances, predecessors, cache.visited,
            cur_f, nxt_f, cache.counter,
            frontier_size, current_depth + 1, compute_predecessors, stream);

        cudaMemcpyAsync(cache.h_frontier_size, cache.counter, sizeof(int32_t),
            cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        frontier_size = *cache.h_frontier_size;

        std::swap(cur_f, nxt_f);
        current_depth++;
    }
}

}  
