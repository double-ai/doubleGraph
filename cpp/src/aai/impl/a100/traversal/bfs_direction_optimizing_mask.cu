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
#include <climits>

namespace aai {

namespace {

struct Cache : Cacheable {
    uint32_t* d_visited_bitmap = nullptr;
    uint32_t* d_next_frontier_bitmap = nullptr;
    int32_t* d_frontier_queue[2] = {nullptr, nullptr};
    int32_t* d_counter = nullptr;
    int32_t* h_counter = nullptr;
    size_t max_bitmap_bytes = 0;
    size_t max_queue_elems = 0;

    void ensure(int32_t num_vertices) {
        size_t bitmap_bytes = ((size_t)(num_vertices + 31) / 32) * sizeof(uint32_t);
        size_t queue_elems = (size_t)num_vertices;

        if (bitmap_bytes > max_bitmap_bytes) {
            if (d_visited_bitmap) cudaFree(d_visited_bitmap);
            if (d_next_frontier_bitmap) cudaFree(d_next_frontier_bitmap);
            cudaMalloc(&d_visited_bitmap, bitmap_bytes);
            cudaMalloc(&d_next_frontier_bitmap, bitmap_bytes);
            max_bitmap_bytes = bitmap_bytes;
        }
        if (queue_elems > max_queue_elems) {
            if (d_frontier_queue[0]) cudaFree(d_frontier_queue[0]);
            if (d_frontier_queue[1]) cudaFree(d_frontier_queue[1]);
            cudaMalloc(&d_frontier_queue[0], queue_elems * sizeof(int32_t));
            cudaMalloc(&d_frontier_queue[1], queue_elems * sizeof(int32_t));
            max_queue_elems = queue_elems;
        }
        if (!d_counter) cudaMalloc(&d_counter, sizeof(int32_t));
        if (!h_counter) cudaMallocHost(&h_counter, sizeof(int32_t));
    }

    ~Cache() override {
        if (d_visited_bitmap) cudaFree(d_visited_bitmap);
        if (d_next_frontier_bitmap) cudaFree(d_next_frontier_bitmap);
        if (d_frontier_queue[0]) cudaFree(d_frontier_queue[0]);
        if (d_frontier_queue[1]) cudaFree(d_frontier_queue[1]);
        if (d_counter) cudaFree(d_counter);
        if (h_counter) cudaFreeHost(h_counter);
    }
};





__global__ void bfs_init_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t num_vertices,
    bool compute_predecessors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < num_vertices; i += stride) {
        distances[i] = INT32_MAX;
        if (compute_predecessors)
            predecessors[i] = -1;
    }
}

__global__ void bfs_set_sources_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    const int32_t* __restrict__ sources,
    int32_t n_sources,
    bool compute_predecessors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_sources) {
        int32_t src = sources[idx];
        distances[src] = 0;
        if (compute_predecessors)
            predecessors[src] = -1;
        atomicOr(&visited_bitmap[src >> 5], 1U << (src & 31));
    }
}

__global__ __launch_bounds__(512) void bfs_top_down_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_count,
    int32_t frontier_size,
    int32_t depth,
    bool compute_predecessors
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t v = frontier[warp_id];
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t new_dist = depth + 1;
    int32_t degree = end - start;

    for (int32_t i = lane_id; i < degree; i += 32) {
        int32_t e = start + i;
        if (!((edge_mask[e >> 5] >> (e & 31)) & 1u)) continue;

        int32_t neighbor = indices[e];
        uint32_t bit = 1U << (neighbor & 31);
        uint32_t old = atomicOr(&visited_bitmap[neighbor >> 5], bit);
        if (!(old & bit)) {
            distances[neighbor] = new_dist;
            if (compute_predecessors)
                predecessors[neighbor] = v;
            int pos = atomicAdd(next_frontier_count, 1);
            next_frontier[pos] = neighbor;
        }
    }
}

__global__ __launch_bounds__(512) void bfs_bottom_up_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    const uint32_t* __restrict__ visited_bitmap,
    uint32_t* __restrict__ next_frontier_bitmap,
    int32_t* __restrict__ next_frontier_count,
    int32_t num_vertices,
    int32_t depth,
    bool compute_predecessors
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int lane_id = threadIdx.x & 31;
    int word_idx = v >> 5;
    int num_words = (num_vertices + 31) >> 5;

    bool need_search = false;
    if (v < num_vertices) {
        need_search = !(visited_bitmap[word_idx] & (1U << lane_id));
    }

    if (!__any_sync(0xFFFFFFFF, need_search)) {
        if (lane_id == 0 && word_idx < num_words)
            next_frontier_bitmap[word_idx] = 0;
        return;
    }

    bool found_parent = false;

    if (need_search) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        if (start < end) {
            int32_t new_dist = depth + 1;

            for (int32_t e = start; e < end; e++) {
                if (!((edge_mask[e >> 5] >> (e & 31)) & 1u)) continue;

                int32_t neighbor = indices[e];
                if (visited_bitmap[neighbor >> 5] & (1U << (neighbor & 31))) {
                    distances[v] = new_dist;
                    if (compute_predecessors)
                        predecessors[v] = neighbor;
                    found_parent = true;
                    break;
                }
            }
        }
    }

    uint32_t ballot = __ballot_sync(0xFFFFFFFF, found_parent);
    if (lane_id == 0) {
        next_frontier_bitmap[word_idx] = ballot;
        if (ballot != 0)
            atomicAdd(next_frontier_count, __popc(ballot));
    }
}

__global__ void merge_bitmaps_kernel(
    uint32_t* __restrict__ dst,
    const uint32_t* __restrict__ src,
    int32_t num_words
) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base + 3 < num_words) {
        uint4 s = *reinterpret_cast<const uint4*>(src + base);
        if (s.x | s.y | s.z | s.w) {
            uint4 d = *reinterpret_cast<const uint4*>(dst + base);
            d.x |= s.x; d.y |= s.y; d.z |= s.z; d.w |= s.w;
            *reinterpret_cast<uint4*>(dst + base) = d;
        }
    } else {
        for (int i = base; i < num_words && i < base + 4; i++) {
            uint32_t s = src[i];
            if (s) dst[i] |= s;
        }
    }
}

__global__ void bitmap_to_queue_kernel(
    const uint32_t* __restrict__ bitmap,
    int32_t* __restrict__ queue,
    int32_t* __restrict__ queue_count,
    int32_t num_words
) {
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (word_idx >= num_words) return;

    uint32_t word = bitmap[word_idx];
    if (word == 0) return;

    int base_v = word_idx << 5;
    int count = __popc(word);
    int base_pos = atomicAdd(queue_count, count);

    while (word) {
        int bit = __ffs(word) - 1;
        word &= word - 1;
        queue[base_pos++] = base_v + bit;
    }
}





void launch_bfs_init(int32_t* distances, int32_t* predecessors,
                     int32_t num_vertices, bool compute_predecessors, cudaStream_t stream) {
    if (num_vertices <= 0) return;
    int block = 512;
    int grid = (num_vertices + block - 1) / block;
    if (grid > 2048) grid = 2048;
    bfs_init_kernel<<<grid, block, 0, stream>>>(distances, predecessors, num_vertices, compute_predecessors);
}

void launch_bfs_set_sources(int32_t* distances, int32_t* predecessors, uint32_t* visited_bitmap,
                            const int32_t* sources, int32_t n_sources, bool compute_predecessors, cudaStream_t stream) {
    if (n_sources <= 0) return;
    int block = 256;
    int grid = (n_sources + block - 1) / block;
    bfs_set_sources_kernel<<<grid, block, 0, stream>>>(distances, predecessors, visited_bitmap, sources, n_sources, compute_predecessors);
}

void launch_bfs_top_down(const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
                         int32_t* distances, int32_t* predecessors, uint32_t* visited_bitmap,
                         const int32_t* frontier, int32_t* next_frontier, int32_t* next_frontier_count,
                         int32_t frontier_size, int32_t depth, bool compute_predecessors, cudaStream_t stream) {
    if (frontier_size <= 0) return;
    int block = 512;
    int64_t total_threads = (int64_t)frontier_size * 32;
    int grid_size = (int)((total_threads + block - 1) / block);
    bfs_top_down_warp_kernel<<<grid_size, block, 0, stream>>>(offsets, indices, edge_mask, distances, predecessors,
        visited_bitmap, frontier, next_frontier, next_frontier_count, frontier_size, depth, compute_predecessors);
}

void launch_bfs_bottom_up(const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
                          int32_t* distances, int32_t* predecessors, const uint32_t* visited_bitmap,
                          uint32_t* next_frontier_bitmap, int32_t* next_frontier_count,
                          int32_t num_vertices, int32_t depth, bool compute_predecessors, cudaStream_t stream) {
    if (num_vertices <= 0) return;
    int block = 512;
    int grid = (num_vertices + block - 1) / block;
    bfs_bottom_up_kernel<<<grid, block, 0, stream>>>(offsets, indices, edge_mask, distances, predecessors,
        visited_bitmap, next_frontier_bitmap, next_frontier_count, num_vertices, depth, compute_predecessors);
}

void launch_merge_bitmaps(uint32_t* dst, const uint32_t* src, int32_t num_words, cudaStream_t stream) {
    if (num_words <= 0) return;
    int block = 256;
    int work = (num_words + 3) / 4;
    int grid = (work + block - 1) / block;
    merge_bitmaps_kernel<<<grid, block, 0, stream>>>(dst, src, num_words);
}

void launch_bitmap_to_queue(const uint32_t* bitmap, int32_t* queue, int32_t* queue_count,
                            int32_t num_words, cudaStream_t stream) {
    if (num_words <= 0) return;
    int block = 256;
    int grid = (num_words + block - 1) / block;
    bitmap_to_queue_kernel<<<grid, block, 0, stream>>>(bitmap, queue, queue_count, num_words);
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

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;
    bool compute_predecessors = (predecessors != nullptr);

    cache.ensure(num_vertices);
    cudaStream_t stream = 0;

    int32_t bitmap_words = (num_vertices + 31) / 32;

    
    launch_bfs_init(distances, predecessors, num_vertices, compute_predecessors, stream);
    cudaMemsetAsync(cache.d_visited_bitmap, 0, bitmap_words * sizeof(uint32_t), stream);

    
    launch_bfs_set_sources(distances, predecessors, cache.d_visited_bitmap, sources, (int32_t)n_sources, compute_predecessors, stream);
    cudaMemcpyAsync(cache.d_frontier_queue[0], sources, n_sources * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

    int32_t frontier_size = (int32_t)n_sources;
    int32_t depth = 0;
    int cur_q = 0;
    bool topdown = true;
    int64_t total_visited = n_sources;

    if (depth_limit < 0) depth_limit = INT32_MAX;

    
    double avg_degree = (num_vertices > 0) ? static_cast<double>(num_edges) / num_vertices : 0.0;
    double alpha = avg_degree * 0.25;
    if (alpha < 2.0) alpha = 2.0;
    constexpr int32_t beta = 24;

    while (frontier_size > 0) {
        int next_q = 1 - cur_q;
        cudaMemsetAsync(cache.d_counter, 0, sizeof(int32_t), stream);

        if (topdown) {
            launch_bfs_top_down(
                d_offsets, d_indices, d_edge_mask,
                distances, predecessors, cache.d_visited_bitmap,
                cache.d_frontier_queue[cur_q], cache.d_frontier_queue[next_q],
                cache.d_counter, frontier_size, depth, compute_predecessors, stream);

            cudaMemcpyAsync(cache.h_counter, cache.d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            int32_t next_size = *cache.h_counter;

            total_visited += next_size;
            int64_t unvisited = num_vertices - total_visited;

            
            if (next_size >= frontier_size && next_size > 0 &&
                static_cast<double>(next_size) * alpha > static_cast<double>(unvisited)) {
                topdown = false;
            }

            frontier_size = next_size;
            cur_q = next_q;

        } else {
            
            launch_bfs_bottom_up(
                d_offsets, d_indices, d_edge_mask,
                distances, predecessors, cache.d_visited_bitmap,
                cache.d_next_frontier_bitmap, cache.d_counter,
                num_vertices, depth, compute_predecessors, stream);

            launch_merge_bitmaps(cache.d_visited_bitmap, cache.d_next_frontier_bitmap, bitmap_words, stream);

            cudaMemcpyAsync(cache.h_counter, cache.d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            int32_t next_size = *cache.h_counter;

            total_visited += next_size;
            int64_t unvisited = num_vertices - total_visited;

            
            if (next_size < frontier_size &&
                static_cast<int64_t>(next_size) * beta < unvisited) {
                topdown = true;
                cudaMemsetAsync(cache.d_counter, 0, sizeof(int32_t), stream);
                launch_bitmap_to_queue(cache.d_next_frontier_bitmap, cache.d_frontier_queue[0], cache.d_counter, bitmap_words, stream);
                cudaStreamSynchronize(stream);
                cur_q = 0;
            }

            frontier_size = next_size;
        }

        depth++;
        if (depth >= depth_limit) break;
    }
}

}  
