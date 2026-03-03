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
#include <climits>

namespace aai {

namespace {

namespace cg = cooperative_groups;











#define CTRL_CUR_SIZE     0
#define CTRL_NEXT_SIZE    1
#define CTRL_NEXT_DEG     2
#define CTRL_TOPDOWN      4
#define CTRL_PREV_SIZE    5
#define CTRL_VISITED_DEG  6
#define CTRL_VISITED_CNT  8
#define CTRL_EXIT_REASON  9





__global__ void persistent_topdown_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* visited_bitmap,
    int32_t* frontier_a,
    int32_t* frontier_b,
    int32_t num_vertices,
    int32_t num_edges,
    int32_t depth_limit,
    bool compute_predecessors,
    int32_t* ctrl,
    int32_t* depth_out
) {
    cg::grid_group grid = cg::this_grid();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int total_warps = total_threads >> 5;

    int32_t* cur_frontier = frontier_a;
    int32_t* next_frontier = frontier_b;

    int depth = *depth_out;

    double avg_degree = (num_vertices > 0) ? (double)num_edges / (double)num_vertices : 0.0;
    double alpha = avg_degree * 0.3;
    if (alpha < 2.0) alpha = 2.0;

    
    if (depth & 1) {
        int32_t* tmp = cur_frontier;
        cur_frontier = next_frontier;
        next_frontier = tmp;
    }

    
    if (tid == 0) {
        ctrl[CTRL_NEXT_SIZE] = 0;
        *(unsigned long long*)&ctrl[CTRL_NEXT_DEG] = 0ULL;
        ctrl[CTRL_EXIT_REASON] = 0;
    }
    grid.sync();

    while (true) {
        int cur_size = ctrl[CTRL_CUR_SIZE];
        if (cur_size == 0) break;

        depth++;

        
        for (int i = warp_id; i < cur_size; i += total_warps) {
            int32_t src = cur_frontier[i];
            int32_t start = offsets[src];
            int32_t end = offsets[src + 1];

            for (int32_t e = start + lane; e < end; e += 32) {
                int32_t dst = indices[e];
                uint32_t bit = 1u << (dst & 31);
                uint32_t word_idx = dst >> 5;

                if (visited_bitmap[word_idx] & bit) continue;

                uint32_t old = atomicOr(&visited_bitmap[word_idx], bit);
                if (!(old & bit)) {
                    distances[dst] = depth;
                    if (compute_predecessors) predecessors[dst] = src;
                    int32_t pos = atomicAdd(&ctrl[CTRL_NEXT_SIZE], 1);
                    next_frontier[pos] = dst;
                    int32_t deg = offsets[dst + 1] - offsets[dst];
                    atomicAdd((unsigned long long*)&ctrl[CTRL_NEXT_DEG], (unsigned long long)deg);
                }
            }
        }

        grid.sync();  

        
        if (tid == 0) {
            int next_size = ctrl[CTRL_NEXT_SIZE];
            int visited = ctrl[CTRL_VISITED_CNT] + next_size;
            ctrl[CTRL_VISITED_CNT] = visited;

            long long next_deg = *(long long*)&ctrl[CTRL_NEXT_DEG];
            long long visited_deg = *(long long*)&ctrl[CTRL_VISITED_DEG];
            visited_deg += next_deg;
            *(long long*)&ctrl[CTRL_VISITED_DEG] = visited_deg;

            
            long long mu = (long long)num_edges - visited_deg;
            if (mu < 0) mu = 0;
            if ((double)next_deg * alpha > (double)mu && next_size >= cur_size) {
                ctrl[CTRL_TOPDOWN] = 0;
                ctrl[CTRL_EXIT_REASON] = 1;  
            }

            ctrl[CTRL_PREV_SIZE] = next_size;
            ctrl[CTRL_CUR_SIZE] = next_size;

            
            ctrl[CTRL_NEXT_SIZE] = 0;
            *(unsigned long long*)&ctrl[CTRL_NEXT_DEG] = 0ULL;
        }

        
        int32_t* tmp = cur_frontier;
        cur_frontier = next_frontier;
        next_frontier = tmp;

        grid.sync();  

        
        if (ctrl[CTRL_EXIT_REASON] != 0 || depth >= depth_limit) break;
    }

    if (tid == 0) {
        *depth_out = depth;
    }
}

__global__ void bottom_up_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    const uint32_t* __restrict__ visited_bitmap,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    unsigned long long* __restrict__ next_deg_sum,
    int32_t num_vertices,
    int32_t current_depth,
    bool compute_predecessors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int v = tid; v < num_vertices; v += stride) {
        if (distances[v] != INT32_MAX) continue;
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        for (int32_t e = start; e < end; e++) {
            int32_t neighbor = indices[e];
            uint32_t word = __ldg(&visited_bitmap[neighbor >> 5]);
            if (word & (1u << (neighbor & 31))) {
                distances[v] = current_depth;
                if (compute_predecessors) predecessors[v] = neighbor;
                int32_t pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = v;
                atomicAdd(next_deg_sum, (unsigned long long)(end - start));
                break;
            }
        }
    }
}

__global__ void mark_frontier_visited_kernel(
    uint32_t* visited_bitmap,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < frontier_size) {
        int32_t v = frontier[tid];
        atomicOr(&visited_bitmap[v >> 5], 1u << (v & 31));
    }
}

__global__ void init_bfs_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    int32_t num_vertices,
    int32_t bitmap_size,
    bool compute_predecessors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < num_vertices; i += stride) {
        distances[i] = INT32_MAX;
        if (compute_predecessors) predecessors[i] = -1;
    }
    for (int i = idx; i < bitmap_size; i += stride) {
        visited_bitmap[i] = 0u;
    }
}

__global__ void init_sources_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ sources,
    int32_t n_sources,
    int32_t* __restrict__ frontier,
    int32_t* ctrl,
    bool compute_predecessors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_sources) {
        int32_t s = sources[idx];
        distances[s] = 0;
        if (compute_predecessors) predecessors[s] = -1;
        atomicOr(&visited_bitmap[s >> 5], 1u << (s & 31));
        frontier[idx] = s;
        int32_t deg = offsets[s + 1] - offsets[s];
        atomicAdd((unsigned long long*)&ctrl[CTRL_VISITED_DEG], (unsigned long long)deg);
    }
}





int get_persistent_grid_size(int block_size) {
    int num_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, persistent_topdown_kernel, block_size, 0);
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return num_blocks_per_sm * prop.multiProcessorCount;
}

void launch_persistent_topdown(
    const int32_t* offsets, const int32_t* indices,
    int32_t* distances, int32_t* predecessors,
    uint32_t* visited_bitmap,
    int32_t* frontier_a, int32_t* frontier_b,
    int32_t num_vertices, int32_t num_edges,
    int32_t depth_limit, bool compute_predecessors,
    int32_t* ctrl, int32_t* depth_out,
    int block_size, int grid_size
) {
    void* args[] = {
        (void*)&offsets, (void*)&indices,
        (void*)&distances, (void*)&predecessors,
        (void*)&visited_bitmap,
        (void*)&frontier_a, (void*)&frontier_b,
        (void*)&num_vertices, (void*)&num_edges,
        (void*)&depth_limit, (void*)&compute_predecessors,
        (void*)&ctrl, (void*)&depth_out
    };

    cudaLaunchCooperativeKernel(
        (void*)persistent_topdown_kernel,
        dim3(grid_size), dim3(block_size),
        args, 0, 0
    );
}

void launch_init_bfs(int32_t* distances, int32_t* predecessors, uint32_t* visited_bitmap,
                     int32_t num_vertices, int32_t bitmap_size, bool compute_predecessors) {
    int block = 256;
    int n = num_vertices > bitmap_size ? num_vertices : bitmap_size;
    int grid = (n + block - 1) / block;
    if (grid == 0) grid = 1;
    init_bfs_kernel<<<grid, block>>>(distances, predecessors, visited_bitmap,
                                      num_vertices, bitmap_size, compute_predecessors);
}

void launch_init_sources(int32_t* distances, int32_t* predecessors, uint32_t* visited_bitmap,
                          const int32_t* offsets, const int32_t* sources, int32_t n_sources,
                          int32_t* frontier, int32_t* ctrl, bool compute_predecessors) {
    int block = 256;
    int grid = (n_sources + block - 1) / block;
    if (grid == 0) grid = 1;
    init_sources_kernel<<<grid, block>>>(distances, predecessors, visited_bitmap,
                                          offsets, sources, n_sources, frontier,
                                          ctrl, compute_predecessors);
}

void launch_bottom_up(const int32_t* offsets, const int32_t* indices,
                      int32_t* distances, int32_t* predecessors, const uint32_t* visited_bitmap,
                      int32_t* next_frontier, int32_t* next_frontier_size,
                      unsigned long long* next_deg_sum,
                      int32_t num_vertices, int32_t current_depth, bool compute_predecessors) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    bottom_up_kernel<<<grid, block>>>(offsets, indices, distances, predecessors,
                                       visited_bitmap, next_frontier, next_frontier_size,
                                       next_deg_sum, num_vertices, current_depth, compute_predecessors);
}

void launch_mark_frontier_visited(uint32_t* visited_bitmap, const int32_t* frontier,
                                   int32_t frontier_size) {
    if (frontier_size == 0) return;
    int block = 256;
    int grid = (frontier_size + block - 1) / block;
    mark_frontier_visited_kernel<<<grid, block>>>(visited_bitmap, frontier, frontier_size);
}





struct Cache : Cacheable {
    int persistent_block_size = 0;
    int persistent_grid_size = 0;

    uint32_t* visited_bitmap = nullptr;
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* ctrl = nullptr;

    int32_t visited_capacity = 0;
    int32_t frontier_a_capacity = 0;
    int32_t frontier_b_capacity = 0;
    bool ctrl_allocated = false;

    Cache() {
        persistent_block_size = 1024;
        persistent_grid_size = get_persistent_grid_size(persistent_block_size);
        if (persistent_grid_size <= 0) {
            persistent_block_size = 512;
            persistent_grid_size = get_persistent_grid_size(persistent_block_size);
        }
        if (persistent_grid_size <= 0) {
            persistent_block_size = 256;
            persistent_grid_size = get_persistent_grid_size(persistent_block_size);
        }
    }

    void ensure(int32_t num_vertices) {
        int32_t bitmap_size = (num_vertices + 31) / 32;

        if (visited_capacity < bitmap_size) {
            if (visited_bitmap) cudaFree(visited_bitmap);
            cudaMalloc(&visited_bitmap, (size_t)bitmap_size * sizeof(uint32_t));
            visited_capacity = bitmap_size;
        }

        if (frontier_a_capacity < num_vertices) {
            if (frontier_a) cudaFree(frontier_a);
            cudaMalloc(&frontier_a, (size_t)num_vertices * sizeof(int32_t));
            frontier_a_capacity = num_vertices;
        }

        if (frontier_b_capacity < num_vertices) {
            if (frontier_b) cudaFree(frontier_b);
            cudaMalloc(&frontier_b, (size_t)num_vertices * sizeof(int32_t));
            frontier_b_capacity = num_vertices;
        }

        if (!ctrl_allocated) {
            cudaMalloc(&ctrl, 12 * sizeof(int32_t));
            ctrl_allocated = true;
        }
    }

    ~Cache() override {
        if (visited_bitmap) cudaFree(visited_bitmap);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (ctrl) cudaFree(ctrl);
    }
};

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

    bool compute_predecessors = (predecessors != nullptr);
    int32_t n_src = static_cast<int32_t>(n_sources);

    if (depth_limit < 0) depth_limit = INT32_MAX;

    cache.ensure(num_vertices);

    uint32_t* d_visited = cache.visited_bitmap;
    int32_t* d_frontier_a = cache.frontier_a;
    int32_t* d_frontier_b = cache.frontier_b;
    int32_t* d_ctrl = cache.ctrl;
    int32_t* d_depth = &d_ctrl[10];

    
    int32_t bitmap_size = (num_vertices + 31) / 32;
    launch_init_bfs(distances, predecessors, d_visited, num_vertices, bitmap_size,
                    compute_predecessors);

    
    int32_t h_ctrl[12] = {n_src, 0, 0, 0, 1, 0, 0, 0, n_src, 0, 0, 0};
    cudaMemcpy(d_ctrl, h_ctrl, sizeof(h_ctrl), cudaMemcpyHostToDevice);

    launch_init_sources(distances, predecessors, d_visited, d_offsets,
                       sources, n_src, d_frontier_a, d_ctrl, compute_predecessors);

    double avg_degree = (num_vertices > 0) ? (double)num_edges / num_vertices : 0.0;
    double alpha = avg_degree * 0.3;
    if (alpha < 2.0) alpha = 2.0;

    
    launch_persistent_topdown(
        d_offsets, d_indices, distances, predecessors,
        d_visited, d_frontier_a, d_frontier_b,
        num_vertices, num_edges, depth_limit, compute_predecessors,
        d_ctrl, d_depth,
        cache.persistent_block_size, cache.persistent_grid_size
    );

    
    int32_t h_state[12];
    cudaMemcpy(h_state, d_ctrl, sizeof(h_state), cudaMemcpyDeviceToHost);

    int32_t current_depth = h_state[10];  
    int32_t frontier_size = h_state[CTRL_CUR_SIZE];
    int32_t exit_reason = h_state[CTRL_EXIT_REASON];

    if (exit_reason == 1 && frontier_size > 0 && current_depth < depth_limit) {
        
        int64_t visited_deg = *(int64_t*)&h_state[CTRL_VISITED_DEG];
        int32_t visited_count = h_state[CTRL_VISITED_CNT];
        int32_t prev_frontier_size = h_state[CTRL_PREV_SIZE];

        
        int32_t* d_frontier = (current_depth & 1) ? d_frontier_b : d_frontier_a;
        int32_t* d_next_frontier = (current_depth & 1) ? d_frontier_a : d_frontier_b;

        bool topdown = false;

        
        while (frontier_size > 0 && current_depth < depth_limit) {
            current_depth++;

            
            cudaMemset(&d_ctrl[CTRL_NEXT_SIZE], 0, sizeof(int32_t));
            cudaMemset(&d_ctrl[CTRL_NEXT_DEG], 0, sizeof(unsigned long long));

            if (!topdown) {
                launch_bottom_up(d_offsets, d_indices, distances, predecessors,
                                d_visited, d_next_frontier, &d_ctrl[CTRL_NEXT_SIZE],
                                (unsigned long long*)&d_ctrl[CTRL_NEXT_DEG],
                                num_vertices, current_depth, compute_predecessors);
            }

            int32_t h_next_size;
            unsigned long long h_next_deg;
            cudaMemcpy(&h_next_size, &d_ctrl[CTRL_NEXT_SIZE], sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_next_deg, &d_ctrl[CTRL_NEXT_DEG], sizeof(unsigned long long), cudaMemcpyDeviceToHost);

            if (!topdown && h_next_size > 0) {
                launch_mark_frontier_visited(d_visited, d_next_frontier, h_next_size);
            }

            prev_frontier_size = frontier_size;
            frontier_size = h_next_size;
            visited_deg += (int64_t)h_next_deg;
            visited_count += frontier_size;

            int64_t unvisited = (int64_t)num_vertices - visited_count;
            if (unvisited < 0) unvisited = 0;

            
            if (!topdown && (int64_t)frontier_size * 24 < unvisited && frontier_size < prev_frontier_size) {
                
                topdown = true;

                
                int32_t h_update[12];
                h_update[CTRL_CUR_SIZE] = frontier_size;
                h_update[CTRL_NEXT_SIZE] = 0;
                h_update[CTRL_NEXT_DEG] = 0;
                h_update[CTRL_NEXT_DEG + 1] = 0;
                h_update[CTRL_TOPDOWN] = 1;
                h_update[CTRL_PREV_SIZE] = frontier_size;
                *(int64_t*)&h_update[CTRL_VISITED_DEG] = visited_deg;
                h_update[CTRL_VISITED_CNT] = visited_count;
                h_update[CTRL_EXIT_REASON] = 0;
                h_update[10] = current_depth;
                h_update[11] = 0;
                cudaMemcpy(d_ctrl, h_update, sizeof(h_update), cudaMemcpyHostToDevice);

                
                launch_persistent_topdown(
                    d_offsets, d_indices, distances, predecessors,
                    d_visited, d_frontier_a, d_frontier_b,
                    num_vertices, num_edges, depth_limit, compute_predecessors,
                    d_ctrl, d_depth,
                    cache.persistent_block_size, cache.persistent_grid_size
                );

                cudaDeviceSynchronize();
                break;  
            }

            
            int32_t* tmp = d_frontier;
            d_frontier = d_next_frontier;
            d_next_frontier = tmp;
        }
    }
}

}  
