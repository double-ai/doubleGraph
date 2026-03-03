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
#include <algorithm>
#include <utility>

namespace aai {

namespace {




struct Cache : Cacheable {
    size_t alloc_vertices = 0;

    int32_t* d_frontier_a = nullptr;
    int32_t* d_frontier_b = nullptr;
    int32_t* d_counter = nullptr;
    uint32_t* d_visited = nullptr;
    int32_t* h_pinned = nullptr;

    Cache() {
        cudaMalloc(&d_counter, sizeof(int32_t));
        cudaMallocHost(&h_pinned, 8 * sizeof(int32_t));
    }

    ~Cache() override {
        if (d_frontier_a) cudaFree(d_frontier_a);
        if (d_frontier_b) cudaFree(d_frontier_b);
        if (d_counter) cudaFree(d_counter);
        if (d_visited) cudaFree(d_visited);
        if (h_pinned) cudaFreeHost(h_pinned);
    }

    void ensure_capacity(size_t num_vertices) {
        if (num_vertices <= alloc_vertices) return;
        if (d_frontier_a) cudaFree(d_frontier_a);
        if (d_frontier_b) cudaFree(d_frontier_b);
        if (d_visited) cudaFree(d_visited);
        alloc_vertices = num_vertices;
        cudaMalloc(&d_frontier_a, alloc_vertices * sizeof(int32_t));
        cudaMalloc(&d_frontier_b, alloc_vertices * sizeof(int32_t));
        size_t bitmap_words = (alloc_vertices + 31) / 32;
        cudaMalloc(&d_visited, bitmap_words * sizeof(uint32_t));
    }
};





#define FULL_MASK 0xFFFFFFFF
#define INF_DIST 2147483647

__global__ void init_bfs_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    int32_t num_vertices,
    int32_t bitmap_words,
    bool compute_pred
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    
    int vec4_count = num_vertices / 4;
    int4* dist_vec = reinterpret_cast<int4*>(distances);
    int4 inf4 = make_int4(INF_DIST, INF_DIST, INF_DIST, INF_DIST);
    for (int i = tid; i < vec4_count; i += stride) {
        dist_vec[i] = inf4;
    }
    
    for (int i = vec4_count * 4 + tid; i < num_vertices; i += stride) {
        distances[i] = INF_DIST;
    }

    if (compute_pred) {
        int4* pred_vec = reinterpret_cast<int4*>(predecessors);
        int4 neg1_4 = make_int4(-1, -1, -1, -1);
        for (int i = tid; i < vec4_count; i += stride) {
            pred_vec[i] = neg1_4;
        }
        for (int i = vec4_count * 4 + tid; i < num_vertices; i += stride) {
            predecessors[i] = -1;
        }
    }

    
    int bm_vec4 = bitmap_words / 4;
    int4* bm_vec = reinterpret_cast<int4*>(visited_bitmap);
    int4 zero4 = make_int4(0, 0, 0, 0);
    for (int i = tid; i < bm_vec4; i += stride) {
        bm_vec[i] = zero4;
    }
    for (int i = bm_vec4 * 4 + tid; i < bitmap_words; i += stride) {
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
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_sources) return;

    int32_t s = sources[tid];
    distances[s] = 0;
    if (compute_pred) predecessors[s] = -1;
    atomicOr(&visited_bitmap[s >> 5], 1u << (s & 31));
    frontier[tid] = s;
}

template <bool COMPUTE_PRED>
__global__ void topdown_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t frontier_size,
    int32_t new_dist
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t src = frontier[warp_id];
    int32_t start = __ldg(&offsets[src]);
    int32_t end = __ldg(&offsets[src + 1]);

    for (int32_t e = start + lane; e < end; e += 32) {
        int32_t dst = __ldg(&indices[e]);
        int32_t word_idx = dst >> 5;
        uint32_t bit = 1u << (dst & 31);

        
        if (__ldg(&visited_bitmap[word_idx]) & bit) continue;

        
        uint32_t old = atomicOr(&visited_bitmap[word_idx], bit);
        if (old & bit) continue;

        
        distances[dst] = new_dist;
        if constexpr (COMPUTE_PRED) predecessors[dst] = src;

        int32_t pos = atomicAdd(next_frontier_size, 1);
        next_frontier[pos] = dst;
    }
}

template <bool COMPUTE_PRED>
__global__ void bottomup_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    const uint32_t* __restrict__ visited_bitmap,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t num_vertices,
    int32_t new_dist
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_global = tid >> 5;
    int bitmap_words = (num_vertices + 31) >> 5;

    if (warp_global >= bitmap_words) return;

    
    uint32_t word = __ldg(&visited_bitmap[warp_global]);
    if (word == 0xFFFFFFFF) return;

    int v = warp_global * 32 + lane;
    bool valid = (v < num_vertices) && !(word & (1u << lane));

    bool found = false;
    int32_t parent = -1;

    if (valid) {
        int32_t start = __ldg(&offsets[v]);
        int32_t end = __ldg(&offsets[v + 1]);

        for (int32_t e = start; e < end; e++) {
            int32_t nbr = __ldg(&indices[e]);
            if (__ldg(&visited_bitmap[nbr >> 5]) & (1u << (nbr & 31))) {
                found = true;
                parent = nbr;
                break;
            }
        }
    }

    
    uint32_t found_mask = __ballot_sync(FULL_MASK, found);
    if (found_mask) {
        int count = __popc(found_mask);
        int32_t base;
        if (lane == 0) base = atomicAdd(next_frontier_size, count);
        base = __shfl_sync(FULL_MASK, base, 0);
        if (found) {
            int offset = __popc(found_mask & ((1u << lane) - 1));
            distances[v] = new_dist;
            if constexpr (COMPUTE_PRED) predecessors[v] = parent;
            next_frontier[base + offset] = v;
        }
    }
}

template <bool COMPUTE_PRED>
__global__ void bottomup_list_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    const uint32_t* __restrict__ visited_bitmap,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    const int32_t* __restrict__ unvisited_in,
    int32_t unvisited_count,
    int32_t* __restrict__ unvisited_out,
    int32_t* __restrict__ unvisited_out_size,
    int32_t new_dist
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    bool valid = (tid < unvisited_count);
    int32_t v = valid ? unvisited_in[tid] : 0;

    bool found = false;
    int32_t parent = -1;

    if (valid) {
        int32_t start = __ldg(&offsets[v]);
        int32_t end = __ldg(&offsets[v + 1]);
        for (int32_t e = start; e < end; e++) {
            int32_t nbr = __ldg(&indices[e]);
            if (__ldg(&visited_bitmap[nbr >> 5]) & (1u << (nbr & 31))) {
                found = true;
                parent = nbr;
                break;
            }
        }
    }

    
    uint32_t found_mask = __ballot_sync(FULL_MASK, found);
    if (found_mask) {
        int count = __popc(found_mask);
        int32_t base;
        if (lane == 0) base = atomicAdd(next_frontier_size, count);
        base = __shfl_sync(FULL_MASK, base, 0);
        if (found) {
            int offset = __popc(found_mask & ((1u << lane) - 1));
            distances[v] = new_dist;
            if constexpr (COMPUTE_PRED) predecessors[v] = parent;
            next_frontier[base + offset] = v;
        }
    }

    
    bool still_unvisited = valid && !found;
    uint32_t uv_mask = __ballot_sync(FULL_MASK, still_unvisited);
    if (uv_mask) {
        int count = __popc(uv_mask);
        int32_t base;
        if (lane == 0) base = atomicAdd(unvisited_out_size, count);
        base = __shfl_sync(FULL_MASK, base, 0);
        if (still_unvisited) {
            int offset = __popc(uv_mask & ((1u << lane) - 1));
            unvisited_out[base + offset] = v;
        }
    }
}

__global__ void build_unvisited_kernel(
    const uint32_t* __restrict__ visited_bitmap,
    int32_t* __restrict__ unvisited_out,
    int32_t* __restrict__ unvisited_out_size,
    int32_t num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_global = tid >> 5;
    int bitmap_words = (num_vertices + 31) >> 5;

    if (warp_global >= bitmap_words) return;

    uint32_t word = __ldg(&visited_bitmap[warp_global]);
    if (word == 0xFFFFFFFF) return;

    int v = warp_global * 32 + lane;
    bool valid = (v < num_vertices) && !(word & (1u << lane));

    uint32_t mask = __ballot_sync(FULL_MASK, valid);
    if (mask) {
        int count = __popc(mask);
        int32_t base;
        if (lane == 0) base = atomicAdd(unvisited_out_size, count);
        base = __shfl_sync(FULL_MASK, base, 0);
        if (valid) {
            int offset = __popc(mask & ((1u << lane) - 1));
            unvisited_out[base + offset] = v;
        }
    }
}

__global__ void update_bitmap_kernel(
    uint32_t* __restrict__ visited_bitmap,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int32_t v = frontier[tid];
    atomicOr(&visited_bitmap[v >> 5], 1u << (v & 31));
}





void launch_init_bfs(
    int32_t* distances, int32_t* predecessors, uint32_t* visited_bitmap,
    int32_t num_vertices, int32_t bitmap_words,
    const int32_t* sources, int32_t n_sources,
    int32_t* frontier, bool compute_pred, cudaStream_t stream
) {
    int block = 256;
    int max_elems = num_vertices / 4;
    if (bitmap_words / 4 > max_elems) max_elems = bitmap_words / 4;
    if (max_elems < num_vertices) max_elems = num_vertices;
    int grid = (max_elems + block - 1) / block;
    if (grid < 1) grid = 1;

    init_bfs_kernel<<<grid, block, 0, stream>>>(
        distances, predecessors, visited_bitmap,
        num_vertices, bitmap_words, compute_pred
    );

    if (n_sources > 0) {
        int sgrid = (n_sources + 255) / 256;
        set_sources_kernel<<<sgrid, 256, 0, stream>>>(
            distances, predecessors, visited_bitmap, frontier,
            sources, n_sources, compute_pred
        );
    }
}

void launch_topdown(
    const int32_t* offsets, const int32_t* indices,
    int32_t* distances, int32_t* predecessors,
    uint32_t* visited_bitmap,
    const int32_t* frontier, int32_t* next_frontier,
    int32_t* next_frontier_size,
    int32_t frontier_size, int32_t new_dist,
    bool compute_pred, cudaStream_t stream
) {
    int64_t total_threads = (int64_t)frontier_size * 32;
    int block = 256;
    int grid = (int)((total_threads + block - 1) / block);
    if (grid < 1) grid = 1;

    if (compute_pred) {
        topdown_kernel<true><<<grid, block, 0, stream>>>(
            offsets, indices, distances, predecessors,
            visited_bitmap, frontier, next_frontier, next_frontier_size,
            frontier_size, new_dist
        );
    } else {
        topdown_kernel<false><<<grid, block, 0, stream>>>(
            offsets, indices, distances, predecessors,
            visited_bitmap, frontier, next_frontier, next_frontier_size,
            frontier_size, new_dist
        );
    }
}

void launch_bottomup(
    const int32_t* offsets, const int32_t* indices,
    int32_t* distances, int32_t* predecessors,
    const uint32_t* visited_bitmap,
    int32_t* next_frontier, int32_t* next_frontier_size,
    int32_t num_vertices, int32_t new_dist,
    bool compute_pred, cudaStream_t stream
) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;

    if (compute_pred) {
        bottomup_kernel<true><<<grid, block, 0, stream>>>(
            offsets, indices, distances, predecessors,
            visited_bitmap, next_frontier, next_frontier_size,
            num_vertices, new_dist
        );
    } else {
        bottomup_kernel<false><<<grid, block, 0, stream>>>(
            offsets, indices, distances, predecessors,
            visited_bitmap, next_frontier, next_frontier_size,
            num_vertices, new_dist
        );
    }
}

void launch_bottomup_list(
    const int32_t* offsets, const int32_t* indices,
    int32_t* distances, int32_t* predecessors,
    const uint32_t* visited_bitmap,
    int32_t* next_frontier, int32_t* next_frontier_size,
    const int32_t* unvisited_in, int32_t unvisited_count,
    int32_t* unvisited_out, int32_t* unvisited_out_size,
    int32_t new_dist, bool compute_pred, cudaStream_t stream
) {
    if (unvisited_count <= 0) return;
    int block = 256;
    int grid = (unvisited_count + block - 1) / block;

    if (compute_pred) {
        bottomup_list_kernel<true><<<grid, block, 0, stream>>>(
            offsets, indices, distances, predecessors,
            visited_bitmap, next_frontier, next_frontier_size,
            unvisited_in, unvisited_count, unvisited_out, unvisited_out_size,
            new_dist
        );
    } else {
        bottomup_list_kernel<false><<<grid, block, 0, stream>>>(
            offsets, indices, distances, predecessors,
            visited_bitmap, next_frontier, next_frontier_size,
            unvisited_in, unvisited_count, unvisited_out, unvisited_out_size,
            new_dist
        );
    }
}

void launch_build_unvisited(
    const uint32_t* visited_bitmap,
    int32_t* unvisited_out, int32_t* unvisited_out_size,
    int32_t num_vertices, cudaStream_t stream
) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    build_unvisited_kernel<<<grid, block, 0, stream>>>(
        visited_bitmap, unvisited_out, unvisited_out_size, num_vertices
    );
}

void launch_update_bitmap(
    uint32_t* visited_bitmap, const int32_t* frontier,
    int32_t frontier_size, cudaStream_t stream
) {
    if (frontier_size <= 0) return;
    int block = 256;
    int grid = (frontier_size + block - 1) / block;
    update_bitmap_kernel<<<grid, block, 0, stream>>>(
        visited_bitmap, frontier, frontier_size
    );
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

    cudaStream_t stream = 0;

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    cache.ensure_capacity((size_t)num_vertices);

    bool compute_pred = (predecessors != nullptr);
    if (depth_limit < 0) depth_limit = std::numeric_limits<int32_t>::max();

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    int32_t bitmap_words = (num_vertices + 31) / 32;

    launch_init_bfs(distances, predecessors, cache.d_visited,
                    num_vertices, bitmap_words,
                    sources, (int32_t)n_sources,
                    cache.d_frontier_a, compute_pred, stream);

    int32_t* cur_frontier = cache.d_frontier_a;
    int32_t* nxt_frontier = cache.d_frontier_b;
    int32_t cur_size = (int32_t)n_sources;
    int64_t visited_count = (int64_t)n_sources;
    int32_t depth = 0;
    bool topdown = true;

    double avg_degree = (num_vertices > 0) ? (double)num_edges / num_vertices : 0.0;
    double alpha = avg_degree * 1.0;
    if (alpha < 1.0) alpha = 1.0;
    constexpr int32_t beta = 24;

    while (cur_size > 0) {
        cudaMemsetAsync(cache.d_counter, 0, sizeof(int32_t), stream);
        int32_t new_dist = depth + 1;

        if (topdown) {
            launch_topdown(d_offsets, d_indices, distances, predecessors,
                           cache.d_visited, cur_frontier, nxt_frontier, cache.d_counter,
                           cur_size, new_dist, compute_pred, stream);
        } else {
            launch_bottomup(d_offsets, d_indices, distances, predecessors,
                            cache.d_visited, nxt_frontier, cache.d_counter,
                            num_vertices, new_dist, compute_pred, stream);
        }

        cudaMemcpyAsync(cache.h_pinned, cache.d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int32_t nxt_size = cache.h_pinned[0];

        if (!topdown && nxt_size > 0) {
            launch_update_bitmap(cache.d_visited, nxt_frontier, nxt_size, stream);
        }

        visited_count += nxt_size;
        int64_t unvisited = (int64_t)num_vertices - visited_count;

        if (topdown) {
            if (nxt_size > 0 && (double)nxt_size * alpha > (double)unvisited) {
                topdown = false;
            }
        } else {
            if (nxt_size > 0 && (int64_t)nxt_size * beta < unvisited && nxt_size < cur_size) {
                topdown = true;
            }
        }

        std::swap(cur_frontier, nxt_frontier);
        cur_size = nxt_size;
        depth++;
        if (depth >= depth_limit) break;
    }

    cudaStreamSynchronize(stream);
}

}  
