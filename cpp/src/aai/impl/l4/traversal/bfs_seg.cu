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

namespace aai {

namespace {




struct Cache : Cacheable {
    int32_t* h_frontier_size = nullptr;  
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    uint32_t* visited_bm = nullptr;
    uint32_t* frontier_bm = nullptr;
    int32_t* d_frontier_size = nullptr;  

    int32_t frontier_a_cap = 0;
    int32_t frontier_b_cap = 0;
    int32_t visited_cap = 0;
    int32_t frontier_bm_cap = 0;

    Cache() {
        cudaHostAlloc(&h_frontier_size, sizeof(int32_t), cudaHostAllocDefault);
        cudaMalloc(&d_frontier_size, sizeof(int32_t));
    }

    void ensure(int32_t num_vertices, int32_t bitmap_words) {
        if (frontier_a_cap < num_vertices) {
            if (frontier_a) cudaFree(frontier_a);
            cudaMalloc(&frontier_a, (size_t)num_vertices * sizeof(int32_t));
            frontier_a_cap = num_vertices;
        }
        if (frontier_b_cap < num_vertices) {
            if (frontier_b) cudaFree(frontier_b);
            cudaMalloc(&frontier_b, (size_t)num_vertices * sizeof(int32_t));
            frontier_b_cap = num_vertices;
        }
        if (visited_cap < bitmap_words) {
            if (visited_bm) cudaFree(visited_bm);
            cudaMalloc(&visited_bm, (size_t)bitmap_words * sizeof(uint32_t));
            visited_cap = bitmap_words;
        }
        if (frontier_bm_cap < bitmap_words) {
            if (frontier_bm) cudaFree(frontier_bm);
            cudaMalloc(&frontier_bm, (size_t)bitmap_words * sizeof(uint32_t));
            frontier_bm_cap = bitmap_words;
        }
    }

    ~Cache() override {
        if (h_frontier_size) cudaFreeHost(h_frontier_size);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (visited_bm) cudaFree(visited_bm);
        if (frontier_bm) cudaFree(frontier_bm);
        if (d_frontier_size) cudaFree(d_frontier_size);
    }
};




__device__ __forceinline__ uint32_t bm_word(int32_t v) { return (uint32_t)v >> 5; }
__device__ __forceinline__ uint32_t bm_bit(int32_t v) { return 1u << (v & 31); }




__global__ void k_bfs_init(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    int32_t num_vertices,
    int32_t bitmap_words,
    int32_t compute_predecessors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < num_vertices; i += stride) {
        distances[i] = 0x7FFFFFFF;
    }
    if (compute_predecessors) {
        for (int i = tid; i < num_vertices; i += stride) {
            predecessors[i] = -1;
        }
    }
    for (int i = tid; i < bitmap_words; i += stride) {
        visited_bitmap[i] = 0;
    }
}




__global__ void k_bfs_set_sources(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    int32_t* __restrict__ frontier,
    const int32_t* __restrict__ sources,
    int32_t n_sources,
    int32_t compute_predecessors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_sources) {
        int32_t src = sources[tid];
        distances[src] = 0;
        if (compute_predecessors) predecessors[src] = -1;
        atomicOr(&visited_bitmap[bm_word(src)], bm_bit(src));
        frontier[tid] = src;
    }
}





__global__ void k_bfs_topdown(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t frontier_size,
    int32_t new_depth,
    int32_t compute_predecessors
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t v = frontier[warp_id];
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    for (int32_t e = start + lane; e < end; e += 32) {
        int32_t neighbor = indices[e];
        uint32_t w = bm_word(neighbor);
        uint32_t b = bm_bit(neighbor);

        if (visited_bitmap[w] & b) continue;

        uint32_t old = atomicOr(&visited_bitmap[w], b);
        if (!(old & b)) {
            distances[neighbor] = new_depth;
            if (compute_predecessors) predecessors[neighbor] = v;
            int32_t pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = neighbor;
        }
    }
}





__global__ void k_bfs_bottomup(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    const uint32_t* __restrict__ frontier_bitmap,
    uint32_t* __restrict__ visited_bitmap,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t num_vertices,
    int32_t new_depth,
    int32_t compute_predecessors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;

    if (visited_bitmap[bm_word(tid)] & bm_bit(tid)) return;

    int32_t start = offsets[tid];
    int32_t end = offsets[tid + 1];
    if (start == end) return;

    for (int32_t e = start; e < end; e++) {
        int32_t neighbor = indices[e];
        if (frontier_bitmap[bm_word(neighbor)] & bm_bit(neighbor)) {
            distances[tid] = new_depth;
            if (compute_predecessors) predecessors[tid] = neighbor;
            atomicOr(&visited_bitmap[bm_word(tid)], bm_bit(tid));
            int32_t pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = tid;
            break;
        }
    }
}




__global__ void k_build_frontier_bitmap(
    const int32_t* __restrict__ frontier,
    uint32_t* __restrict__ frontier_bitmap,
    int32_t frontier_size,
    int32_t bitmap_words
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < bitmap_words; i += stride) {
        frontier_bitmap[i] = 0;
    }
}

__global__ void k_set_frontier_bitmap(
    const int32_t* __restrict__ frontier,
    uint32_t* __restrict__ frontier_bitmap,
    int32_t frontier_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < frontier_size) {
        int32_t v = frontier[tid];
        atomicOr(&frontier_bitmap[bm_word(v)], bm_bit(v));
    }
}

}  

void bfs_seg(const graph32_t& graph,
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
    bool is_symmetric = graph.is_symmetric;

    if (depth_limit < 0) depth_limit = 0x7FFFFFFF;

    int32_t bitmap_words = (num_vertices + 31) / 32;
    int32_t cp = (predecessors != nullptr) ? 1 : 0;

    cudaStream_t stream = 0;

    cache.ensure(num_vertices, bitmap_words);

    int32_t* d_frontier_in = cache.frontier_a;
    int32_t* d_frontier_out = cache.frontier_b;

    
    {
        int block = 512;
        int n = (num_vertices > bitmap_words) ? num_vertices : bitmap_words;
        int grid = (n + block - 1) / block;
        k_bfs_init<<<grid, block, 0, stream>>>(distances, predecessors, cache.visited_bm,
                                                num_vertices, bitmap_words, cp);
    }

    
    {
        int block = 256;
        int grid = ((int)n_sources + block - 1) / block;
        if (grid < 1) grid = 1;
        k_bfs_set_sources<<<grid, block, 0, stream>>>(distances, predecessors, cache.visited_bm,
                                                       d_frontier_in, sources, (int32_t)n_sources, cp);
    }

    int32_t frontier_size = (int32_t)n_sources;
    int32_t current_depth = 0;

    
    double avg_degree = (num_vertices > 0) ? (double)num_edges / (double)num_vertices : 0.0;
    double alpha = avg_degree * 0.3;
    constexpr int32_t beta = 24;
    bool use_topdown = true;
    int32_t prev_frontier_size = 0;

    
    while (frontier_size > 0 && current_depth < depth_limit) {
        cudaMemsetAsync(cache.d_frontier_size, 0, sizeof(int32_t), stream);

        int32_t new_depth = current_depth + 1;

        if (use_topdown) {
            int block = 256;
            int64_t total_threads = (int64_t)frontier_size * 32;
            int grid = (int)((total_threads + block - 1) / block);
            if (grid >= 1) {
                k_bfs_topdown<<<grid, block, 0, stream>>>(d_offsets, d_indices, distances, predecessors,
                                                           cache.visited_bm, d_frontier_in, d_frontier_out,
                                                           cache.d_frontier_size, frontier_size, new_depth, cp);
            }
        } else {
            
            {
                int block = 512;
                int grid = (bitmap_words + block - 1) / block;
                k_build_frontier_bitmap<<<grid, block, 0, stream>>>(d_frontier_in, cache.frontier_bm,
                                                                     frontier_size, bitmap_words);
            }
            {
                int block = 256;
                int grid = (frontier_size + block - 1) / block;
                if (grid < 1) grid = 1;
                k_set_frontier_bitmap<<<grid, block, 0, stream>>>(d_frontier_in, cache.frontier_bm, frontier_size);
            }
            {
                int block = 256;
                int grid = (num_vertices + block - 1) / block;
                k_bfs_bottomup<<<grid, block, 0, stream>>>(d_offsets, d_indices, distances, predecessors,
                                                            cache.frontier_bm, cache.visited_bm,
                                                            d_frontier_out, cache.d_frontier_size,
                                                            num_vertices, new_depth, cp);
            }
        }

        
        cudaMemcpyAsync(cache.h_frontier_size, cache.d_frontier_size, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        prev_frontier_size = frontier_size;
        frontier_size = *cache.h_frontier_size;

        
        int32_t* tmp = d_frontier_in;
        d_frontier_in = d_frontier_out;
        d_frontier_out = tmp;

        current_depth = new_depth;

        
        if (is_symmetric && frontier_size > 0) {
            if (use_topdown) {
                if (frontier_size >= prev_frontier_size &&
                    (int64_t)frontier_size * 10 > (int64_t)num_vertices) {
                    use_topdown = false;
                }
            } else {
                if (frontier_size < prev_frontier_size &&
                    (int64_t)frontier_size * beta < (int64_t)num_vertices) {
                    use_topdown = true;
                }
            }
        }
    }
}

}  
