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

struct Cache : Cacheable {
    uint32_t* visited_bitmap = nullptr;
    uint32_t* new_visited_bitmap = nullptr;
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* frontier_size_d = nullptr;
    unsigned long long* heuristic_buf = nullptr;
    size_t max_vertices = 0;

    void ensure(int32_t num_vertices) {
        if ((size_t)num_vertices <= max_vertices) return;

        free_all();
        max_vertices = num_vertices;

        int bitmap_words = (num_vertices + 31) / 32;
        cudaMalloc(&visited_bitmap, bitmap_words * sizeof(uint32_t));
        cudaMalloc(&new_visited_bitmap, bitmap_words * sizeof(uint32_t));
        cudaMalloc(&frontier_a, num_vertices * sizeof(int32_t));
        cudaMalloc(&frontier_b, num_vertices * sizeof(int32_t));
        cudaMalloc(&frontier_size_d, sizeof(int32_t));
        cudaMalloc(&heuristic_buf, 2 * sizeof(unsigned long long));
    }

    void free_all() {
        if (visited_bitmap) { cudaFree(visited_bitmap); visited_bitmap = nullptr; }
        if (new_visited_bitmap) { cudaFree(new_visited_bitmap); new_visited_bitmap = nullptr; }
        if (frontier_a) { cudaFree(frontier_a); frontier_a = nullptr; }
        if (frontier_b) { cudaFree(frontier_b); frontier_b = nullptr; }
        if (frontier_size_d) { cudaFree(frontier_size_d); frontier_size_d = nullptr; }
        if (heuristic_buf) { cudaFree(heuristic_buf); heuristic_buf = nullptr; }
        max_vertices = 0;
    }

    ~Cache() override {
        free_all();
    }
};


__device__ __forceinline__ int packed_bool_offset(int v) { return v >> 5; }
__device__ __forceinline__ uint32_t packed_bool_mask(int v) { return 1u << (v & 31); }


__device__ __forceinline__ bool is_edge_active(const uint32_t* edge_mask, int edge_idx) {
    return (edge_mask[edge_idx >> 5] >> (edge_idx & 31)) & 1;
}


__device__ __forceinline__ bool is_visited(const uint32_t* bitmap, int v) {
    return (bitmap[v >> 5] >> (v & 31)) & 1;
}




__global__ void bfs_topdown_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t frontier_size,
    int32_t current_depth
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int32_t src = frontier[tid];
    int32_t start = offsets[src];
    int32_t end = offsets[src + 1];
    int32_t next_depth = current_depth + 1;

    for (int32_t e = start; e < end; e++) {
        if (!is_edge_active(edge_mask, e)) continue;

        int32_t dst = indices[e];
        uint32_t mask = packed_bool_mask(dst);
        uint32_t old = atomicOr(&visited_bitmap[packed_bool_offset(dst)], mask);

        if (!(old & mask)) {
            distances[dst] = next_depth;
            if (predecessors) predecessors[dst] = src;
            int pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = dst;
        }
    }
}




__global__ void bfs_topdown_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t frontier_size,
    int32_t current_depth
) {
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread >> 5;
    int lane_id = global_thread & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;

    int32_t next_depth = current_depth + 1;

    for (int w = warp_id; w < frontier_size; w += total_warps) {
        int32_t src = frontier[w];
        int32_t start = offsets[src];
        int32_t end = offsets[src + 1];

        for (int32_t e = start + lane_id; e < end; e += 32) {
            if (!is_edge_active(edge_mask, e)) continue;

            int32_t dst = indices[e];
            uint32_t mask = packed_bool_mask(dst);
            uint32_t old = atomicOr(&visited_bitmap[packed_bool_offset(dst)], mask);

            if (!(old & mask)) {
                distances[dst] = next_depth;
                if (predecessors) predecessors[dst] = src;
                int pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = dst;
            }
        }
    }
}





__global__ void bfs_bottomup_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    const uint32_t* __restrict__ visited_bitmap,
    uint32_t* __restrict__ new_visited_bitmap,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t num_vertices,
    int32_t current_depth
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;

    
    if (is_visited(visited_bitmap, tid)) return;

    
    int32_t start = offsets[tid];
    int32_t end = offsets[tid + 1];
    if (start >= end) return;

    int32_t next_depth = current_depth + 1;

    for (int32_t e = start; e < end; e++) {
        if (!is_edge_active(edge_mask, e)) continue;

        int32_t neighbor = indices[e];

        if (is_visited(visited_bitmap, neighbor)) {
            distances[tid] = next_depth;
            if (predecessors) predecessors[tid] = neighbor;

            atomicOr(&new_visited_bitmap[packed_bool_offset(tid)], packed_bool_mask(tid));
            int pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = tid;
            return; 
        }
    }
}




__global__ void init_bfs_kernel(
    int32_t* distances,
    int32_t* predecessors,
    int32_t num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;
    distances[tid] = INT32_MAX;
    if (predecessors) predecessors[tid] = -1;
}




__global__ void set_sources_kernel(
    const int32_t* sources,
    int32_t n_sources,
    int32_t* distances,
    int32_t* predecessors,
    uint32_t* visited_bitmap,
    int32_t* frontier,
    int32_t* frontier_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_sources) return;

    int32_t src = sources[tid];
    distances[src] = 0;
    if (predecessors) predecessors[src] = -1;
    atomicOr(&visited_bitmap[packed_bool_offset(src)], packed_bool_mask(src));
    int pos = atomicAdd(frontier_size, 1);
    frontier[pos] = src;
}




__global__ void compute_frontier_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    unsigned long long* __restrict__ result
) {
    typedef cub::BlockReduce<unsigned long long, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    unsigned long long thread_sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < frontier_size;
         i += blockDim.x * gridDim.x) {
        int32_t v = frontier[i];
        thread_sum += (unsigned long long)(offsets[v + 1] - offsets[v]);
    }

    unsigned long long block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    if (threadIdx.x == 0 && block_sum > 0) {
        atomicAdd(result, block_sum);
    }
}




__global__ void compute_unvisited_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ visited_bitmap,
    int32_t num_vertices,
    unsigned long long* __restrict__ result
) {
    typedef cub::BlockReduce<unsigned long long, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    unsigned long long thread_sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_vertices;
         i += blockDim.x * gridDim.x) {
        if (!is_visited(visited_bitmap, i)) {
            thread_sum += (unsigned long long)(offsets[i + 1] - offsets[i]);
        }
    }

    unsigned long long block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    if (threadIdx.x == 0 && block_sum > 0) {
        atomicAdd(result, block_sum);
    }
}




__global__ void merge_bitmaps_kernel(
    uint32_t* __restrict__ visited_bitmap,
    const uint32_t* __restrict__ new_visited_bitmap,
    int32_t bitmap_words
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= bitmap_words) return;

    uint32_t new_bits = new_visited_bitmap[tid];
    if (new_bits) {
        visited_bitmap[tid] |= new_bits;
    }
}

}  

void bfs_direction_optimizing_mask(const graph32_t& graph,
                                   int32_t* distances,
                                   int32_t* predecessors,
                                   const int32_t* sources,
                                   std::size_t n_sources,
                                   int32_t depth_limit) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    const uint32_t* edge_mask = graph.edge_mask;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    int32_t n_src = static_cast<int32_t>(n_sources);

    if (depth_limit < 0) depth_limit = INT32_MAX;

    cache.ensure(num_vertices);

    int bitmap_words = (num_vertices + 31) / 32;
    const int THREADS = 256;
    cudaStream_t stream = 0;

    
    {
        int blocks = (num_vertices + THREADS - 1) / THREADS;
        init_bfs_kernel<<<blocks, THREADS, 0, stream>>>(distances, predecessors, num_vertices);
    }

    cudaMemsetAsync(cache.visited_bitmap, 0, bitmap_words * sizeof(uint32_t), stream);
    cudaMemsetAsync(cache.frontier_size_d, 0, sizeof(int32_t), stream);

    {
        int blocks = (n_src + THREADS - 1) / THREADS;
        set_sources_kernel<<<blocks, THREADS, 0, stream>>>(
            sources, n_src, distances, predecessors, cache.visited_bitmap,
            cache.frontier_a, cache.frontier_size_d);
    }

    int32_t frontier_size;
    cudaMemcpyAsync(&frontier_size, cache.frontier_size_d, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int32_t* cur_frontier = cache.frontier_a;
    int32_t* next_frontier = cache.frontier_b;

    bool topdown = true;
    int32_t current_depth = 0;

    double avg_degree = (num_vertices > 0) ? (double)num_edges / num_vertices : 0.0;
    double alpha = avg_degree * 0.3;
    if (alpha < 2.0) alpha = 2.0;
    const int32_t beta = 24;

    
    int32_t total_visited = n_src;
    int32_t prev_frontier_size = 0;

    while (frontier_size > 0) {
        cudaMemsetAsync(cache.frontier_size_d, 0, sizeof(int32_t), stream);

        if (topdown) {
            
            if (frontier_size <= 128) {
                int blocks = (frontier_size + THREADS - 1) / THREADS;
                bfs_topdown_thread_kernel<<<blocks, THREADS, 0, stream>>>(
                    offsets, indices, edge_mask, distances, predecessors,
                    cache.visited_bitmap, cur_frontier, next_frontier,
                    cache.frontier_size_d, frontier_size, current_depth);
            } else {
                
                int64_t total_threads = (int64_t)frontier_size * 32;
                int blocks = (int)((total_threads + THREADS - 1) / THREADS);
                if (blocks > 65535) blocks = 65535;
                bfs_topdown_warp_kernel<<<blocks, THREADS, 0, stream>>>(
                    offsets, indices, edge_mask, distances, predecessors,
                    cache.visited_bitmap, cur_frontier, next_frontier,
                    cache.frontier_size_d, frontier_size, current_depth);
            }
        } else {
            
            cudaMemsetAsync(cache.new_visited_bitmap, 0,
                            bitmap_words * sizeof(uint32_t), stream);

            int blocks = (num_vertices + THREADS - 1) / THREADS;
            bfs_bottomup_kernel<<<blocks, THREADS, 0, stream>>>(
                offsets, indices, edge_mask, distances, predecessors,
                cache.visited_bitmap, cache.new_visited_bitmap,
                next_frontier, cache.frontier_size_d,
                num_vertices, current_depth);

            
            int merge_blocks = (bitmap_words + THREADS - 1) / THREADS;
            merge_bitmaps_kernel<<<merge_blocks, THREADS, 0, stream>>>(
                cache.visited_bitmap, cache.new_visited_bitmap, bitmap_words);
        }

        int32_t next_frontier_size;
        cudaMemcpyAsync(&next_frontier_size, cache.frontier_size_d, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        total_visited += next_frontier_size;

        
        if (topdown && next_frontier_size > 0 && next_frontier_size >= frontier_size) {
            
            
            if (next_frontier_size > 1000) {
                cudaMemsetAsync(cache.heuristic_buf, 0,
                                2 * sizeof(unsigned long long), stream);

                int blocks_f = (next_frontier_size + THREADS - 1) / THREADS;
                if (blocks_f > 1024) blocks_f = 1024;
                compute_frontier_edges_kernel<<<blocks_f, THREADS, 0, stream>>>(
                    offsets, next_frontier, next_frontier_size,
                    &cache.heuristic_buf[0]);

                int blocks_u = (num_vertices + THREADS - 1) / THREADS;
                if (blocks_u > 1024) blocks_u = 1024;
                compute_unvisited_edges_kernel<<<blocks_u, THREADS, 0, stream>>>(
                    offsets, cache.visited_bitmap, num_vertices,
                    &cache.heuristic_buf[1]);

                unsigned long long h_buf[2];
                cudaMemcpyAsync(h_buf, cache.heuristic_buf,
                                2 * sizeof(unsigned long long),
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);

                double m_f = (double)h_buf[0];
                double m_u = (double)h_buf[1];

                if (m_f * alpha > m_u) {
                    topdown = false;
                }
            }
        } else if (!topdown && next_frontier_size > 0) {
            
            int32_t unvisited = num_vertices - total_visited;
            if ((long long)next_frontier_size * beta < (long long)unvisited &&
                next_frontier_size < frontier_size) {
                topdown = true;
            }
        }

        prev_frontier_size = frontier_size;
        frontier_size = next_frontier_size;

        
        int32_t* tmp = cur_frontier;
        cur_frontier = next_frontier;
        next_frontier = tmp;

        current_depth++;
        if (depth_limit != INT32_MAX && current_depth >= depth_limit) break;
    }
}

}  
