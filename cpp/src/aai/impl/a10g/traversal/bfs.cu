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
    int32_t* d_frontier_a = nullptr;
    int32_t* d_frontier_b = nullptr;
    uint32_t* d_visited = nullptr;
    uint32_t* d_prev_visited = nullptr;
    uint32_t* d_frontier_bitmap = nullptr;
    int32_t* d_counter_a = nullptr;
    int32_t* d_counter_b = nullptr;
    int32_t* h_pinned = nullptr;
    size_t max_vertices = 0;

    void ensure_scratch(int32_t num_vertices) {
        size_t needed = static_cast<size_t>(num_vertices);
        if (needed <= max_vertices) return;
        free_all();
        max_vertices = needed;
        cudaMalloc(&d_frontier_a, needed * sizeof(int32_t));
        cudaMalloc(&d_frontier_b, needed * sizeof(int32_t));
        size_t bitmap_words = (needed + 31) / 32;
        cudaMalloc(&d_visited, bitmap_words * sizeof(uint32_t));
        cudaMalloc(&d_prev_visited, bitmap_words * sizeof(uint32_t));
        cudaMalloc(&d_frontier_bitmap, bitmap_words * sizeof(uint32_t));
        cudaMalloc(&d_counter_a, sizeof(int32_t));
        cudaMalloc(&d_counter_b, sizeof(int32_t));
        cudaHostAlloc(&h_pinned, 2 * sizeof(int32_t), cudaHostAllocDefault);
    }

    void free_all() {
        if (d_frontier_a) { cudaFree(d_frontier_a); d_frontier_a = nullptr; }
        if (d_frontier_b) { cudaFree(d_frontier_b); d_frontier_b = nullptr; }
        if (d_visited) { cudaFree(d_visited); d_visited = nullptr; }
        if (d_prev_visited) { cudaFree(d_prev_visited); d_prev_visited = nullptr; }
        if (d_frontier_bitmap) { cudaFree(d_frontier_bitmap); d_frontier_bitmap = nullptr; }
        if (d_counter_a) { cudaFree(d_counter_a); d_counter_a = nullptr; }
        if (d_counter_b) { cudaFree(d_counter_b); d_counter_b = nullptr; }
        if (h_pinned) { cudaFreeHost(h_pinned); h_pinned = nullptr; }
        max_vertices = 0;
    }

    ~Cache() override {
        free_all();
    }
};





__device__ __forceinline__ int packed_bool_offset(int v) { return v >> 5; }
__device__ __forceinline__ unsigned int packed_bool_mask(int v) { return 1u << (v & 31); }

template <bool COMPUTE_PRED>
__global__ void bfs_init_bulk_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    int32_t num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int v = tid; v < num_vertices; v += stride) {
        distances[v] = INT32_MAX;
        if (COMPUTE_PRED) predecessors[v] = -1;
    }
    int bitmap_words = (num_vertices + 31) / 32;
    for (int w = tid; w < bitmap_words; w += stride) {
        visited[w] = 0;
    }
}

__global__ void bfs_set_sources_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    const int32_t* __restrict__ sources,
    int32_t* __restrict__ frontier,
    int32_t n_sources,
    bool compute_predecessors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_sources) {
        int src = sources[tid];
        distances[src] = 0;
        if (compute_predecessors) predecessors[src] = -1;
        frontier[tid] = src;
        atomicOr(&visited[packed_bool_offset(src)], packed_bool_mask(src));
    }
}

template <bool COMPUTE_PRED>
__global__ void bfs_topdown_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t frontier_size,
    int32_t new_dist
) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = gtid >> 5;
    int lane = gtid & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;

    for (int fi = warp_id; fi < frontier_size; fi += total_warps) {
        int32_t src = frontier[fi];
        int32_t start = __ldg(&offsets[src]);
        int32_t end = __ldg(&offsets[src + 1]);

        for (int32_t e = start + lane; e < end; e += 32) {
            int32_t dst = __ldg(&indices[e]);
            int bword = packed_bool_offset(dst);
            unsigned int bmask = packed_bool_mask(dst);

            if (__ldg(&visited[bword]) & bmask) continue;
            unsigned int old = atomicOr(&visited[bword], bmask);
            if (old & bmask) continue;

            distances[dst] = new_dist;
            if (COMPUTE_PRED) predecessors[dst] = src;
            int32_t pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = dst;
        }
    }
}

template <bool COMPUTE_PRED>
__global__ void bfs_topdown_devsize_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ frontier_size_in,
    int32_t* __restrict__ next_frontier_size,
    int32_t new_dist
) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = gtid >> 5;
    int lane = gtid & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;

    int32_t fsize = *frontier_size_in;
    if (fsize <= 0) return;

    for (int fi = warp_id; fi < fsize; fi += total_warps) {
        int32_t src = frontier[fi];
        int32_t start = __ldg(&offsets[src]);
        int32_t end = __ldg(&offsets[src + 1]);

        for (int32_t e = start + lane; e < end; e += 32) {
            int32_t dst = __ldg(&indices[e]);
            int bword = packed_bool_offset(dst);
            unsigned int bmask = packed_bool_mask(dst);

            if (__ldg(&visited[bword]) & bmask) continue;
            unsigned int old = atomicOr(&visited[bword], bmask);
            if (old & bmask) continue;

            distances[dst] = new_dist;
            if (COMPUTE_PRED) predecessors[dst] = src;
            int32_t pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = dst;
        }
    }
}

__global__ void build_frontier_bitmap(
    const uint32_t* __restrict__ visited,
    const uint32_t* __restrict__ prev_visited,
    uint32_t* __restrict__ frontier_bitmap,
    int32_t num_words
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int w = tid; w < num_words; w += stride) {
        frontier_bitmap[w] = visited[w] & ~prev_visited[w];
    }
}

__global__ void copy_bitmap(
    const uint32_t* __restrict__ src,
    uint32_t* __restrict__ dst,
    int32_t num_words
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int w = tid; w < num_words; w += stride) {
        dst[w] = src[w];
    }
}

template <bool COMPUTE_PRED>
__global__ void bfs_bottomup_bitmap_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    const uint32_t* __restrict__ frontier_bitmap,
    int32_t* __restrict__ next_frontier_size,
    int32_t num_vertices,
    int32_t new_dist
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int32_t v = tid; v < num_vertices; v += stride) {
        uint32_t vword = visited[packed_bool_offset(v)];
        if (vword & packed_bool_mask(v)) continue;

        int32_t start = __ldg(&offsets[v]);
        int32_t end = __ldg(&offsets[v + 1]);

        for (int32_t e = start; e < end; e++) {
            int32_t neighbor = __ldg(&indices[e]);
            uint32_t fword = __ldg(&frontier_bitmap[packed_bool_offset(neighbor)]);
            if (fword & packed_bool_mask(neighbor)) {
                uint32_t old = atomicOr(&visited[packed_bool_offset(v)], packed_bool_mask(v));
                if (!(old & packed_bool_mask(v))) {
                    distances[v] = new_dist;
                    if (COMPUTE_PRED) predecessors[v] = neighbor;
                    atomicAdd(next_frontier_size, 1);
                }
                break;
            }
        }
    }
}

template <bool COMPUTE_PRED>
__global__ void bfs_bottomup_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ distances,
    int32_t* __restrict__ new_distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    int32_t* __restrict__ next_frontier_size,
    int32_t num_vertices,
    int32_t current_depth
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int32_t v = tid; v < num_vertices; v += stride) {
        uint32_t vword = visited[packed_bool_offset(v)];
        if (vword & packed_bool_mask(v)) continue;
        int32_t start = __ldg(&offsets[v]);
        int32_t end = __ldg(&offsets[v + 1]);
        for (int32_t e = start; e < end; e++) {
            int32_t neighbor = __ldg(&indices[e]);
            if (distances[neighbor] == current_depth) {
                uint32_t old = atomicOr(&visited[packed_bool_offset(v)], packed_bool_mask(v));
                if (!(old & packed_bool_mask(v))) {
                    new_distances[v] = current_depth + 1;
                    if (COMPUTE_PRED) predecessors[v] = neighbor;
                    atomicAdd(next_frontier_size, 1);
                }
                break;
            }
        }
    }
}

__global__ void build_frontier_from_distances(
    const int32_t* __restrict__ distances,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_size,
    int32_t num_vertices,
    int32_t target_depth
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int32_t v = tid; v < num_vertices; v += stride) {
        if (distances[v] == target_depth) {
            int32_t pos = atomicAdd(frontier_size, 1);
            frontier[pos] = v;
        }
    }
}

__global__ void zero_counter(int32_t* counter) { *counter = 0; }





void launch_bfs_init(
    int32_t* distances, int32_t* predecessors, uint32_t* visited,
    const int32_t* sources, int32_t* frontier,
    int32_t num_vertices, int32_t n_sources, bool compute_predecessors,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_vertices + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    if (compute_predecessors) {
        bfs_init_bulk_kernel<true><<<blocks, threads, 0, stream>>>(
            distances, predecessors, visited, num_vertices);
    } else {
        bfs_init_bulk_kernel<false><<<blocks, threads, 0, stream>>>(
            distances, predecessors, visited, num_vertices);
    }
    int src_blocks = (n_sources + threads - 1) / threads;
    if (src_blocks < 1) src_blocks = 1;
    bfs_set_sources_kernel<<<src_blocks, threads, 0, stream>>>(
        distances, predecessors, visited, sources, frontier,
        n_sources, compute_predecessors);
}

void launch_bfs_topdown(
    const int32_t* offsets, const int32_t* indices,
    int32_t* distances, int32_t* predecessors, uint32_t* visited,
    const int32_t* frontier, int32_t* next_frontier, int32_t* next_frontier_size,
    int32_t frontier_size, int32_t new_dist, bool compute_predecessors,
    cudaStream_t stream
) {
    if (frontier_size == 0) return;
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (frontier_size + warps_per_block - 1) / warps_per_block;
    if (blocks > 1280) blocks = 1280;
    if (compute_predecessors) {
        bfs_topdown_kernel<true><<<blocks, threads, 0, stream>>>(
            offsets, indices, distances, predecessors, visited,
            frontier, next_frontier, next_frontier_size, frontier_size, new_dist);
    } else {
        bfs_topdown_kernel<false><<<blocks, threads, 0, stream>>>(
            offsets, indices, distances, predecessors, visited,
            frontier, next_frontier, next_frontier_size, frontier_size, new_dist);
    }
}

void launch_bfs_topdown_2levels(
    const int32_t* offsets, const int32_t* indices,
    int32_t* distances, int32_t* predecessors, uint32_t* visited,
    int32_t* frontier_a, int32_t* frontier_b,
    int32_t* counter_a, int32_t* counter_b,
    int32_t frontier_size_level1, int32_t depth1, int32_t depth2,
    bool compute_predecessors, int32_t num_vertices,
    cudaStream_t stream
) {
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (frontier_size_level1 + warps_per_block - 1) / warps_per_block;
    if (blocks > 1280) blocks = 1280;

    if (compute_predecessors) {
        bfs_topdown_kernel<true><<<blocks, threads, 0, stream>>>(
            offsets, indices, distances, predecessors, visited,
            frontier_a, frontier_b, counter_b, frontier_size_level1, depth1);
    } else {
        bfs_topdown_kernel<false><<<blocks, threads, 0, stream>>>(
            offsets, indices, distances, predecessors, visited,
            frontier_a, frontier_b, counter_b, frontier_size_level1, depth1);
    }

    zero_counter<<<1, 1, 0, stream>>>(counter_a);

    blocks = 1280;
    if (compute_predecessors) {
        bfs_topdown_devsize_kernel<true><<<blocks, threads, 0, stream>>>(
            offsets, indices, distances, predecessors, visited,
            frontier_b, frontier_a, counter_b, counter_a, depth2);
    } else {
        bfs_topdown_devsize_kernel<false><<<blocks, threads, 0, stream>>>(
            offsets, indices, distances, predecessors, visited,
            frontier_b, frontier_a, counter_b, counter_a, depth2);
    }
}

void launch_copy_bitmap(
    const uint32_t* src, uint32_t* dst, int32_t num_words, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_words + threads - 1) / threads;
    if (blocks > 512) blocks = 512;
    copy_bitmap<<<blocks, threads, 0, stream>>>(src, dst, num_words);
}

void launch_bfs_bottomup(
    const int32_t* offsets, const int32_t* indices,
    int32_t* distances, int32_t* predecessors, uint32_t* visited,
    int32_t* next_frontier_size,
    int32_t num_vertices, int32_t current_depth, bool compute_predecessors,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_vertices + threads - 1) / threads;
    if (blocks > 1280) blocks = 1280;
    if (compute_predecessors) {
        bfs_bottomup_kernel<true><<<blocks, threads, 0, stream>>>(
            offsets, indices, distances, distances, predecessors, visited,
            next_frontier_size, num_vertices, current_depth);
    } else {
        bfs_bottomup_kernel<false><<<blocks, threads, 0, stream>>>(
            offsets, indices, distances, distances, predecessors, visited,
            next_frontier_size, num_vertices, current_depth);
    }
}

void launch_build_frontier(
    const int32_t* distances, int32_t* frontier, int32_t* frontier_size,
    int32_t num_vertices, int32_t target_depth, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (num_vertices + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    build_frontier_from_distances<<<blocks, threads, 0, stream>>>(
        distances, frontier, frontier_size, num_vertices, target_depth);
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

    int32_t num_vertices = graph.number_of_vertices;
    bool is_symmetric = graph.is_symmetric;
    bool compute_predecessors = (predecessors != nullptr);
    int32_t ns = static_cast<int32_t>(n_sources);

    if (depth_limit < 0) depth_limit = std::numeric_limits<int32_t>::max();

    cache.ensure_scratch(num_vertices);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    cudaStream_t stream = 0;
    int32_t bitmap_words = (num_vertices + 31) / 32;

    launch_bfs_init(distances, predecessors, cache.d_visited, sources,
                    cache.d_frontier_a, num_vertices, ns,
                    compute_predecessors, stream);

    int32_t* cur_frontier = cache.d_frontier_a;
    int32_t* next_frontier = cache.d_frontier_b;
    int32_t* cur_counter = cache.d_counter_a;
    int32_t* next_counter = cache.d_counter_b;
    int32_t h_frontier_size = ns;
    int32_t current_depth = 0;

    const int32_t ALPHA = 4;
    const int32_t BETA = 24;
    bool topdown = true;
    bool can_do_bottomup = is_symmetric;
    int32_t prev_frontier_size = 0;

    while (h_frontier_size > 0 && current_depth < depth_limit) {
        bool do_batch = topdown &&
                        current_depth + 2 <= depth_limit &&
                        h_frontier_size >= 32 &&
                        (!can_do_bottomup || h_frontier_size <= num_vertices / (ALPHA * 2));

        if (do_batch) {
            cudaMemsetAsync(next_counter, 0, sizeof(int32_t), stream);
            launch_bfs_topdown_2levels(
                d_offsets, d_indices, distances, predecessors, cache.d_visited,
                cur_frontier, next_frontier, cur_counter, next_counter,
                h_frontier_size, current_depth + 1, current_depth + 2,
                compute_predecessors, num_vertices, stream);

            prev_frontier_size = h_frontier_size;
            cudaMemcpyAsync(cache.h_pinned, cur_counter, sizeof(int32_t),
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            h_frontier_size = cache.h_pinned[0];
            current_depth += 2;

            if (can_do_bottomup &&
                h_frontier_size > num_vertices / ALPHA &&
                h_frontier_size >= prev_frontier_size) {
                topdown = false;
            }
        } else {
            cudaMemsetAsync(next_counter, 0, sizeof(int32_t), stream);

            if (topdown) {
                launch_bfs_topdown(d_offsets, d_indices, distances, predecessors,
                                   cache.d_visited, cur_frontier, next_frontier, next_counter,
                                   h_frontier_size, current_depth + 1, compute_predecessors, stream);
            } else {
                launch_copy_bitmap(cache.d_visited, cache.d_prev_visited, bitmap_words, stream);

                launch_bfs_bottomup(d_offsets, d_indices, distances, predecessors,
                                    cache.d_visited, next_counter, num_vertices,
                                    current_depth, compute_predecessors, stream);
            }

            prev_frontier_size = h_frontier_size;
            cudaMemcpyAsync(cache.h_pinned, next_counter, sizeof(int32_t),
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            h_frontier_size = cache.h_pinned[0];

            bool just_rebuilt = false;
            if (can_do_bottomup) {
                if (topdown) {
                    if (h_frontier_size > num_vertices / ALPHA &&
                        h_frontier_size >= prev_frontier_size) {
                        topdown = false;
                    }
                } else {
                    if (h_frontier_size > 0 &&
                        static_cast<int64_t>(h_frontier_size) * BETA < num_vertices &&
                        h_frontier_size < prev_frontier_size) {
                        topdown = true;
                        just_rebuilt = true;
                        cudaMemsetAsync(next_counter, 0, sizeof(int32_t), stream);
                        launch_build_frontier(distances, next_frontier, next_counter,
                                              num_vertices, current_depth + 1, stream);
                    }
                }
            }

            if (topdown) {
                int32_t* tmp;
                tmp = cur_frontier; cur_frontier = next_frontier; next_frontier = tmp;
                tmp = cur_counter; cur_counter = next_counter; next_counter = tmp;
            }
            current_depth++;
        }
    }
}

}  
