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
#include <cub/cub.cuh>

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define WARP_SIZE 32





struct Cache : Cacheable {
    uint32_t* visited = nullptr;
    uint32_t* frontier_bmp = nullptr;
    int32_t* frontier1 = nullptr;
    int32_t* frontier2 = nullptr;
    int32_t* d_count = nullptr;
    int64_t* d_stats = nullptr;
    int32_t capacity = 0;

    void ensure(int32_t num_vertices) {
        if (capacity >= num_vertices) return;

        if (visited) cudaFree(visited);
        if (frontier_bmp) cudaFree(frontier_bmp);
        if (frontier1) cudaFree(frontier1);
        if (frontier2) cudaFree(frontier2);
        if (d_count) cudaFree(d_count);
        if (d_stats) cudaFree(d_stats);

        int32_t bitmap_words = (num_vertices + 31) / 32;
        cudaMalloc(&visited, bitmap_words * sizeof(uint32_t));
        cudaMalloc(&frontier_bmp, bitmap_words * sizeof(uint32_t));
        cudaMalloc(&frontier1, num_vertices * sizeof(int32_t));
        cudaMalloc(&frontier2, num_vertices * sizeof(int32_t));
        cudaMalloc(&d_count, sizeof(int32_t));
        cudaMalloc(&d_stats, 2 * sizeof(int64_t));

        capacity = num_vertices;
    }

    ~Cache() override {
        if (visited) cudaFree(visited);
        if (frontier_bmp) cudaFree(frontier_bmp);
        if (frontier1) cudaFree(frontier1);
        if (frontier2) cudaFree(frontier2);
        if (d_count) cudaFree(d_count);
        if (d_stats) cudaFree(d_stats);
    }
};





__global__ void init_bfs_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    int32_t num_vertices,
    int32_t bitmap_words,
    bool compute_pred
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;

    
    int32_t vec4_count = num_vertices >> 2;
    for (int32_t i = tid; i < vec4_count; i += stride) {
        reinterpret_cast<int4*>(distances)[i] = make_int4(0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF);
    }
    for (int32_t i = (vec4_count << 2) + tid; i < num_vertices; i += stride) {
        distances[i] = 0x7FFFFFFF;
    }

    if (compute_pred) {
        for (int32_t i = tid; i < vec4_count; i += stride) {
            reinterpret_cast<int4*>(predecessors)[i] = make_int4(-1, -1, -1, -1);
        }
        for (int32_t i = (vec4_count << 2) + tid; i < num_vertices; i += stride) {
            predecessors[i] = -1;
        }
    }

    int32_t bm_vec4 = bitmap_words >> 2;
    for (int32_t i = tid; i < bm_vec4; i += stride) {
        reinterpret_cast<int4*>(visited)[i] = make_int4(0, 0, 0, 0);
    }
    for (int32_t i = (bm_vec4 << 2) + tid; i < bitmap_words; i += stride) {
        visited[i] = 0u;
    }
}

__global__ void set_sources_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    int32_t* __restrict__ frontier,
    const int32_t* __restrict__ sources,
    int32_t n_sources,
    bool compute_pred
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_sources) return;
    int32_t v = sources[tid];
    distances[v] = 0;
    if (compute_pred) predecessors[v] = -1;
    atomicOr(&visited[v >> 5], 1u << (v & 31));
    frontier[tid] = v;
}





__global__ void clear_bitmap_kernel(uint32_t* bitmap, int32_t words) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;
    int32_t vec4_count = words >> 2;
    for (int32_t i = tid; i < vec4_count; i += stride) {
        reinterpret_cast<int4*>(bitmap)[i] = make_int4(0, 0, 0, 0);
    }
    for (int32_t i = (vec4_count << 2) + tid; i < words; i += stride) {
        bitmap[i] = 0u;
    }
}

__global__ void build_frontier_bitmap_kernel(
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    uint32_t* __restrict__ frontier_bmp
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;
    for (int32_t i = tid; i < frontier_size; i += stride) {
        int32_t v = frontier[i];
        atomicOr(&frontier_bmp[v >> 5], 1u << (v & 31));
    }
}





template <bool ComputePred>
__global__ void topdown_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count,
    int32_t new_dist
) {
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;
    int32_t num_warps = (blockDim.x * gridDim.x) >> 5;

    for (int32_t i = warp_id; i < frontier_size; i += num_warps) {
        int32_t src = frontier[i];
        int32_t start = __ldg(&offsets[src]);
        int32_t end = __ldg(&offsets[src + 1]);
        int32_t degree = end - start;

        for (int32_t j = lane; j < degree; j += 32) {
            int32_t dst = __ldg(&indices[start + j]);
            uint32_t bit = 1u << (dst & 31);
            uint32_t word_idx = dst >> 5;

            
            if (__ldg(&visited[word_idx]) & bit) continue;

            uint32_t old = atomicOr(&visited[word_idx], bit);
            if (!(old & bit)) {
                distances[dst] = new_dist;
                if constexpr (ComputePred) {
                    predecessors[dst] = src;
                }
                int32_t pos = atomicAdd(next_count, 1);
                next_frontier[pos] = dst;
            }
        }
    }
}





template <bool ComputePred>
__global__ void bottomup_bitmap_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    const uint32_t* __restrict__ visited,
    uint32_t* __restrict__ new_visited,
    const uint32_t* __restrict__ frontier_bmp,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count,
    int32_t num_vertices,
    int32_t new_dist
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;

    for (int32_t v = tid; v < num_vertices; v += stride) {
        uint32_t my_bit = 1u << (v & 31);
        uint32_t my_word_idx = v >> 5;

        
        if (__ldg(&visited[my_word_idx]) & my_bit) continue;

        int32_t start = __ldg(&offsets[v]);
        int32_t end = __ldg(&offsets[v + 1]);
        if (start == end) continue; 

        
        for (int32_t e = start; e < end; e++) {
            int32_t neighbor = __ldg(&indices[e]);
            uint32_t n_bit = 1u << (neighbor & 31);
            uint32_t n_word_idx = neighbor >> 5;

            if (__ldg(&frontier_bmp[n_word_idx]) & n_bit) {
                
                distances[v] = new_dist;
                if constexpr (ComputePred) {
                    predecessors[v] = neighbor;
                }
                atomicOr(&new_visited[my_word_idx], my_bit);
                int32_t pos = atomicAdd(next_count, 1);
                next_frontier[pos] = v;
                break; 
            }
        }
    }
}





__global__ void sum_frontier_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int64_t* __restrict__ result
) {
    typedef cub::BlockReduce<int64_t, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;

    int64_t local_sum = 0;
    for (int32_t i = tid; i < frontier_size; i += stride) {
        int32_t v = frontier[i];
        local_sum += __ldg(&offsets[v + 1]) - __ldg(&offsets[v]);
    }

    int64_t block_sum = BlockReduce(temp_storage).Sum(local_sum);
    if (threadIdx.x == 0 && block_sum > 0) {
        atomicAdd((unsigned long long*)result, (unsigned long long)block_sum);
    }
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

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    bool is_symmetric = graph.is_symmetric;
    bool compute_pred = (predecessors != nullptr);
    int32_t ns = static_cast<int32_t>(n_sources);

    cache.ensure(num_vertices);

    const int BLOCK = BLOCK_SIZE;
    int32_t bitmap_words = (num_vertices + 31) >> 5;

    
    {
        int32_t total = (num_vertices > bitmap_words) ? num_vertices : bitmap_words;
        int32_t grid = (total + BLOCK - 1) / BLOCK;
        if (grid > 2048) grid = 2048;
        init_bfs_kernel<<<grid, BLOCK>>>(distances, predecessors, cache.visited,
                                          num_vertices, bitmap_words, compute_pred);
    }

    
    {
        int32_t grid = (ns + BLOCK - 1) / BLOCK;
        if (grid < 1) grid = 1;
        set_sources_kernel<<<grid, BLOCK>>>(distances, predecessors, cache.visited,
                                             cache.frontier1, sources, ns, compute_pred);
    }

    int32_t* cur_frontier = cache.frontier1;
    int32_t* nxt_frontier = cache.frontier2;
    int32_t frontier_size = ns;
    int32_t depth = 0;
    int32_t prev_frontier_size = 0;

    if (depth_limit < 0) depth_limit = 0x7FFFFFFF;

    
    double avg_degree = (num_vertices > 0) ? (double)num_edges / num_vertices : 0.0;
    double alpha = avg_degree * 0.25;  
    const int32_t beta = 24;
    bool topdown_mode = true;
    bool can_use_bottomup = is_symmetric && (num_vertices > 10000);

    
    int32_t total_visited = ns;

    while (frontier_size > 0 && depth < depth_limit) {
        cudaMemsetAsync(cache.d_count, 0, sizeof(int32_t));

        bool use_bottomup = false;

        
        if (can_use_bottomup) {
            if (topdown_mode) {
                
                if (frontier_size > 100 && frontier_size >= prev_frontier_size) {
                    
                    cudaMemsetAsync(cache.d_stats, 0, sizeof(int64_t));
                    int32_t grid_fe = (frontier_size + BLOCK - 1) / BLOCK;
                    if (grid_fe > 512) grid_fe = 512;
                    sum_frontier_degrees_kernel<<<grid_fe, BLOCK>>>(
                        offsets, cur_frontier, frontier_size, cache.d_stats);

                    int64_t h_frontier_edges;
                    cudaMemcpy(&h_frontier_edges, cache.d_stats, sizeof(int64_t), cudaMemcpyDeviceToHost);

                    
                    double unvisited_frac = 1.0 - (double)total_visited / num_vertices;
                    double m_u_est = (double)num_edges * unvisited_frac;
                    double m_f = (double)h_frontier_edges;

                    if (m_f * alpha > m_u_est) {
                        topdown_mode = false;
                        use_bottomup = true;
                    }
                }
            } else {
                use_bottomup = true;
            }
        }

        if (use_bottomup) {
            
            {
                int32_t grid_clr = (bitmap_words + BLOCK - 1) / BLOCK;
                if (grid_clr > 512) grid_clr = 512;
                clear_bitmap_kernel<<<grid_clr, BLOCK>>>(cache.frontier_bmp, bitmap_words);
            }
            {
                int32_t grid_bld = (frontier_size + BLOCK - 1) / BLOCK;
                if (grid_bld > 512) grid_bld = 512;
                build_frontier_bitmap_kernel<<<grid_bld, BLOCK>>>(
                    cur_frontier, frontier_size, cache.frontier_bmp);
            }

            
            int32_t grid = (num_vertices + BLOCK - 1) / BLOCK;
            if (grid > 2048) grid = 2048;

            if (compute_pred) {
                bottomup_bitmap_kernel<true><<<grid, BLOCK>>>(
                    offsets, indices, distances, predecessors,
                    cache.visited, cache.visited, cache.frontier_bmp,
                    nxt_frontier, cache.d_count, num_vertices, depth + 1);
            } else {
                bottomup_bitmap_kernel<false><<<grid, BLOCK>>>(
                    offsets, indices, distances, predecessors,
                    cache.visited, cache.visited, cache.frontier_bmp,
                    nxt_frontier, cache.d_count, num_vertices, depth + 1);
            }
        } else {
            
            int32_t warps_per_block = BLOCK / 32;
            int32_t grid = (frontier_size + warps_per_block - 1) / warps_per_block;
            if (grid > 65535) grid = 65535;

            if (compute_pred) {
                topdown_warp_kernel<true><<<grid, BLOCK>>>(
                    offsets, indices, distances, predecessors,
                    cache.visited, cur_frontier, frontier_size,
                    nxt_frontier, cache.d_count, depth + 1);
            } else {
                topdown_warp_kernel<false><<<grid, BLOCK>>>(
                    offsets, indices, distances, predecessors,
                    cache.visited, cur_frontier, frontier_size,
                    nxt_frontier, cache.d_count, depth + 1);
            }
        }

        
        prev_frontier_size = frontier_size;
        cudaMemcpy(&frontier_size, cache.d_count, sizeof(int32_t), cudaMemcpyDeviceToHost);

        total_visited += frontier_size;

        
        if (!topdown_mode && can_use_bottomup) {
            int32_t unvisited_est = num_vertices - total_visited;
            if ((int64_t)frontier_size * beta < (int64_t)unvisited_est &&
                frontier_size < prev_frontier_size) {
                topdown_mode = true;
            }
        }

        
        int32_t* tmp = cur_frontier;
        cur_frontier = nxt_frontier;
        nxt_frontier = tmp;

        depth++;
    }
}

}  
