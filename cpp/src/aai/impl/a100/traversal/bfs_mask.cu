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
#include <cooperative_groups.h>
#include <cstdint>

namespace aai {

namespace {

namespace cg = cooperative_groups;





struct Cache : Cacheable {
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    uint32_t* visited = nullptr;
    int32_t* frontier_sizes = nullptr;
    int32_t* pinned_size = nullptr;
    size_t alloc_vertices = 0;

    void ensure(size_t num_vertices) {
        if (alloc_vertices < num_vertices) {
            if (frontier_a) cudaFree(frontier_a);
            if (frontier_b) cudaFree(frontier_b);
            if (visited) cudaFree(visited);

            cudaMalloc(&frontier_a, num_vertices * sizeof(int32_t));
            cudaMalloc(&frontier_b, num_vertices * sizeof(int32_t));
            cudaMalloc(&visited, ((num_vertices + 31) / 32) * sizeof(uint32_t));
            alloc_vertices = num_vertices;
        }
        if (!frontier_sizes) {
            cudaMalloc(&frontier_sizes, 2 * sizeof(int32_t));
        }
        if (!pinned_size) {
            cudaHostAlloc(&pinned_size, sizeof(int32_t), cudaHostAllocDefault);
        }
    }

    ~Cache() override {
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (visited) cudaFree(visited);
        if (frontier_sizes) cudaFree(frontier_sizes);
        if (pinned_size) cudaFreeHost(pinned_size);
    }
};





template <bool COMPUTE_PRED>
__global__ void bfs_init_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    int32_t num_vertices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int n4 = num_vertices >> 2;
    for (int i = idx; i < n4; i += stride) {
        reinterpret_cast<int4*>(distances)[i] = make_int4(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF);
    }
    for (int i = (n4 << 2) + idx; i < num_vertices; i += stride) {
        distances[i] = 0x7FFFFFFF;
    }

    if constexpr (COMPUTE_PRED) {
        for (int i = idx; i < n4; i += stride) {
            reinterpret_cast<int4*>(predecessors)[i] = make_int4(-1, -1, -1, -1);
        }
        for (int i = (n4 << 2) + idx; i < num_vertices; i += stride) {
            predecessors[i] = -1;
        }
    }

    int bw = (num_vertices + 31) >> 5;
    int bw4 = bw >> 2;
    for (int i = idx; i < bw4; i += stride) {
        reinterpret_cast<int4*>(visited)[i] = make_int4(0, 0, 0, 0);
    }
    for (int i = (bw4 << 2) + idx; i < bw; i += stride) {
        visited[i] = 0;
    }
}

__global__ void bfs_set_sources_kernel(
    int32_t* __restrict__ distances,
    uint32_t* __restrict__ visited,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_sizes,
    const int32_t* __restrict__ sources,
    int32_t n_sources
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_sources) {
        int32_t v = sources[idx];
        distances[v] = 0;
        uint32_t old = atomicOr(&visited[v >> 5], 1u << (v & 31));
        if (!(old & (1u << (v & 31)))) {
            int slot = atomicAdd(&frontier_sizes[0], 1);
            frontier[slot] = v;
        }
    }
}





template <bool COMPUTE_PRED>
__global__ void bfs_expand_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    const int32_t* __restrict__ frontier_in,
    int32_t frontier_in_size,
    int32_t* __restrict__ frontier_out,
    int32_t* __restrict__ frontier_out_size,
    int32_t depth
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int total_warps = (blockDim.x * gridDim.x) >> 5;

    for (int wi = warp_id; wi < frontier_in_size; wi += total_warps) {
        int32_t v = frontier_in[wi];
        int32_t start = __ldg(&offsets[v]);
        int32_t end = __ldg(&offsets[v + 1]);
        int32_t degree = end - start;
        int32_t num_iters = (degree + 31) >> 5;

        for (int32_t iter = 0; iter < num_iters; iter++) {
            int32_t e = start + (iter << 5) + lane;
            bool discovered = false;
            int32_t neighbor = -1;

            if (e < end) {
                uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
                if ((mask_word >> (e & 31)) & 1u) {
                    neighbor = __ldg(&indices[e]);
                    uint32_t word_idx = neighbor >> 5;
                    uint32_t bit = 1u << (neighbor & 31);

                    if (!(visited[word_idx] & bit)) {
                        uint32_t old = atomicOr(&visited[word_idx], bit);
                        if (!(old & bit)) {
                            distances[neighbor] = depth;
                            if constexpr (COMPUTE_PRED) {
                                predecessors[neighbor] = v;
                            }
                            discovered = true;
                        }
                    }
                }
            }

            unsigned int disc_mask = __ballot_sync(0xFFFFFFFF, discovered);
            if (disc_mask) {
                int count = __popc(disc_mask);
                int prefix = __popc(disc_mask & ((1u << lane) - 1));
                int base;
                if (lane == 0) {
                    base = atomicAdd(frontier_out_size, count);
                }
                base = __shfl_sync(0xFFFFFFFF, base, 0);
                if (discovered) {
                    frontier_out[base + prefix] = neighbor;
                }
            }
        }
    }
}





template <bool COMPUTE_PRED>
__global__ void bfs_loop_persistent(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    int32_t* __restrict__ frontier_a,
    int32_t* __restrict__ frontier_b,
    int32_t num_vertices,
    int32_t depth_limit,
    int32_t* __restrict__ frontier_sizes
) {
    cg::grid_group grid = cg::this_grid();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int total_warps = (blockDim.x * gridDim.x) >> 5;

    int32_t* frontiers[2] = {frontier_a, frontier_b};
    int cur = 0;
    int32_t depth = 1;

    while (frontier_sizes[cur] > 0 && depth <= depth_limit) {
        int32_t frontier_size = frontier_sizes[cur];
        int next = 1 - cur;

        if (tid == 0) frontier_sizes[next] = 0;
        grid.sync();

        int32_t* cur_f = frontiers[cur];
        int32_t* next_f = frontiers[next];

        for (int wi = warp_id; wi < frontier_size; wi += total_warps) {
            int32_t v = cur_f[wi];
            int32_t start = __ldg(&offsets[v]);
            int32_t end = __ldg(&offsets[v + 1]);
            int32_t degree = end - start;
            int32_t num_iters = (degree + 31) >> 5;

            for (int32_t iter = 0; iter < num_iters; iter++) {
                int32_t e = start + (iter << 5) + lane;
                bool discovered = false;
                int32_t neighbor = -1;

                if (e < end) {
                    uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
                    if ((mask_word >> (e & 31)) & 1u) {
                        neighbor = __ldg(&indices[e]);
                        uint32_t word_idx = neighbor >> 5;
                        uint32_t bit = 1u << (neighbor & 31);

                        if (!(visited[word_idx] & bit)) {
                            uint32_t old = atomicOr(&visited[word_idx], bit);
                            if (!(old & bit)) {
                                distances[neighbor] = depth;
                                if constexpr (COMPUTE_PRED) {
                                    predecessors[neighbor] = v;
                                }
                                discovered = true;
                            }
                        }
                    }
                }

                unsigned int disc_mask = __ballot_sync(0xFFFFFFFF, discovered);
                if (disc_mask) {
                    int count = __popc(disc_mask);
                    int prefix = __popc(disc_mask & ((1u << lane) - 1));
                    int base;
                    if (lane == 0) {
                        base = atomicAdd(&frontier_sizes[next], count);
                    }
                    base = __shfl_sync(0xFFFFFFFF, base, 0);
                    if (discovered) {
                        next_f[base + prefix] = neighbor;
                    }
                }
            }
        }

        cur = next;
        depth++;
        grid.sync();
    }
}





static int s_num_sms = 0;
static int s_coop_blocks_pred = 0;
static int s_coop_blocks_nopred = 0;

void launch_bfs_internal(
    const int32_t* offsets,
    const int32_t* indices,
    const uint32_t* edge_mask,
    int32_t* distances,
    int32_t* predecessors,
    int32_t num_vertices,
    int32_t num_edges,
    const int32_t* sources,
    int32_t n_sources,
    int32_t depth_limit,
    bool compute_predecessors,
    int32_t* frontier_a,
    int32_t* frontier_b,
    uint32_t* visited_bitmap,
    int32_t* frontier_sizes,
    int32_t* pinned_size
) {
    constexpr int block_size = 256;

    
    if (s_num_sms == 0) {
        cudaDeviceGetAttribute(&s_num_sms, cudaDevAttrMultiProcessorCount, 0);
        int bps;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &bps, bfs_loop_persistent<true>, block_size, 0);
        s_coop_blocks_pred = bps * s_num_sms;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &bps, bfs_loop_persistent<false>, block_size, 0);
        s_coop_blocks_nopred = bps * s_num_sms;
    }

    int max_blocks = s_num_sms * 8;

    
    if (compute_predecessors) {
        bfs_init_kernel<true><<<max_blocks, block_size>>>(
            distances, predecessors, visited_bitmap, num_vertices);
    } else {
        bfs_init_kernel<false><<<max_blocks, block_size>>>(
            distances, predecessors, visited_bitmap, num_vertices);
    }

    
    cudaMemsetAsync(frontier_sizes, 0, 2 * sizeof(int32_t));
    int grid_src = (n_sources + block_size - 1) / block_size;
    bfs_set_sources_kernel<<<grid_src > 0 ? grid_src : 1, block_size>>>(
        distances, visited_bitmap, frontier_a, frontier_sizes, sources, n_sources);

    
    
    double avg_degree = (num_vertices > 0) ? (double)num_edges / num_vertices : 0.0;
    bool use_persistent = (avg_degree < 8.0);

    if (use_persistent) {
        
        int coop_blocks = compute_predecessors ? s_coop_blocks_pred : s_coop_blocks_nopred;

        if (compute_predecessors) {
            void* args[] = {
                (void*)&offsets, (void*)&indices, (void*)&edge_mask,
                (void*)&distances, (void*)&predecessors, (void*)&visited_bitmap,
                (void*)&frontier_a, (void*)&frontier_b,
                (void*)&num_vertices, (void*)&depth_limit, (void*)&frontier_sizes
            };
            cudaLaunchCooperativeKernel(
                (void*)bfs_loop_persistent<true>, coop_blocks, block_size, args);
        } else {
            void* args[] = {
                (void*)&offsets, (void*)&indices, (void*)&edge_mask,
                (void*)&distances, (void*)&predecessors, (void*)&visited_bitmap,
                (void*)&frontier_a, (void*)&frontier_b,
                (void*)&num_vertices, (void*)&depth_limit, (void*)&frontier_sizes
            };
            cudaLaunchCooperativeKernel(
                (void*)bfs_loop_persistent<false>, coop_blocks, block_size, args);
        }
    } else {
        
        
        cudaMemcpy(pinned_size, frontier_sizes, sizeof(int32_t), cudaMemcpyDeviceToHost);
        int32_t frontier_size_h = *pinned_size;

        int32_t* current_frontier = frontier_a;
        int32_t* next_frontier = frontier_b;
        int32_t depth = 1;

        while (frontier_size_h > 0 && depth <= depth_limit) {
            cudaMemsetAsync(&frontier_sizes[0], 0, sizeof(int32_t));

            int warps_needed = frontier_size_h;
            int blocks_needed = (int)(((int64_t)warps_needed * 32 + block_size - 1) / block_size);
            int grid_bfs = blocks_needed < max_blocks ? blocks_needed : max_blocks;
            if (grid_bfs < 1) grid_bfs = 1;

            if (compute_predecessors) {
                bfs_expand_kernel<true><<<grid_bfs, block_size>>>(
                    offsets, indices, edge_mask, distances, predecessors, visited_bitmap,
                    current_frontier, frontier_size_h,
                    next_frontier, &frontier_sizes[0], depth);
            } else {
                bfs_expand_kernel<false><<<grid_bfs, block_size>>>(
                    offsets, indices, edge_mask, distances, predecessors, visited_bitmap,
                    current_frontier, frontier_size_h,
                    next_frontier, &frontier_sizes[0], depth);
            }

            cudaMemcpy(pinned_size, &frontier_sizes[0], sizeof(int32_t), cudaMemcpyDeviceToHost);
            frontier_size_h = *pinned_size;

            int32_t* temp = current_frontier;
            current_frontier = next_frontier;
            next_frontier = temp;

            depth++;
        }
    }
}

}  

void bfs_mask(const graph32_t& graph,
              int32_t* distances,
              int32_t* predecessors,
              const int32_t* sources,
              std::size_t n_sources,
              int32_t depth_limit) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    cache.ensure(num_vertices);

    if (depth_limit < 0) depth_limit = 0x7FFFFFFF;

    bool compute_predecessors = (predecessors != nullptr);

    launch_bfs_internal(
        graph.offsets,
        graph.indices,
        graph.edge_mask,
        distances,
        predecessors,
        num_vertices,
        num_edges,
        sources,
        static_cast<int32_t>(n_sources),
        depth_limit,
        compute_predecessors,
        cache.frontier_a,
        cache.frontier_b,
        cache.visited,
        cache.frontier_sizes,
        cache.pinned_size
    );
}

}  
