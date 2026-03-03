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

static constexpr int32_t PERSIST_THRESHOLD = 4096;

struct Cache : Cacheable {
    int32_t* d_frontier[2] = {nullptr, nullptr};
    int32_t* d_count = nullptr;
    int32_t* d_output_info = nullptr;
    int32_t* h_info = nullptr;
    uint32_t* d_bitmap = nullptr;
    size_t buf_capacity = 0;
    size_t bitmap_capacity = 0;

    Cache() {
        cudaMallocHost(&h_info, 4 * sizeof(int32_t));
    }

    void ensure_buffers(size_t n) {
        if (n > buf_capacity) {
            for (int i = 0; i < 2; i++) {
                if (d_frontier[i]) cudaFree(d_frontier[i]);
                cudaMalloc(&d_frontier[i], n * sizeof(int32_t));
            }
            if (d_count) cudaFree(d_count);
            cudaMalloc(&d_count, sizeof(int32_t));
            if (d_output_info) cudaFree(d_output_info);
            cudaMalloc(&d_output_info, 4 * sizeof(int32_t));
            buf_capacity = n;
        }
        size_t bm_words = (n + 31) / 32;
        if (bm_words > bitmap_capacity) {
            if (d_bitmap) cudaFree(d_bitmap);
            cudaMalloc(&d_bitmap, bm_words * sizeof(uint32_t));
            bitmap_capacity = bm_words;
        }
    }

    ~Cache() override {
        for (int i = 0; i < 2; i++) {
            if (d_frontier[i]) cudaFree(d_frontier[i]);
        }
        if (d_count) cudaFree(d_count);
        if (d_output_info) cudaFree(d_output_info);
        if (h_info) cudaFreeHost(h_info);
        if (d_bitmap) cudaFree(d_bitmap);
    }
};





__global__ void bfs_init_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    int32_t num_vertices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        distances[idx] = INT32_MAX;
        if (predecessors) predecessors[idx] = -1;
    }
    int bw = (num_vertices + 31) >> 5;
    if (idx < bw) visited_bitmap[idx] = 0;
}

__global__ void bfs_set_sources_kernel(
    int32_t* __restrict__ distances,
    uint32_t* __restrict__ visited_bitmap,
    int32_t* __restrict__ frontier,
    const int32_t* __restrict__ sources,
    int32_t n_sources
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sources) return;
    int32_t s = sources[idx];
    distances[s] = 0;
    atomicOr(&visited_bitmap[s >> 5], 1u << (s & 31));
    frontier[idx] = s;
}


__global__ void bfs_expand_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count,
    int32_t frontier_size,
    int32_t next_depth
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t src = frontier[warp_id];
    int32_t row_begin = __ldg(&offsets[src]);
    int32_t row_end = __ldg(&offsets[src + 1]);

    for (int32_t e = row_begin + lane; e < row_end; e += 32) {
        if (!((edge_mask[e >> 5] >> (e & 31)) & 1u)) continue;

        int32_t dst = indices[e];

        uint32_t mask_bit = 1u << (dst & 31);
        if (visited_bitmap[dst >> 5] & mask_bit) continue;

        uint32_t old_word = atomicOr(&visited_bitmap[dst >> 5], mask_bit);
        if (old_word & mask_bit) continue;

        distances[dst] = next_depth;
        if (predecessors) predecessors[dst] = src;
        int pos = atomicAdd(next_count, 1);
        next_frontier[pos] = dst;
    }
}


__global__ void bfs_expand_persistent(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bitmap,
    int32_t* __restrict__ frontier_a,
    int32_t* __restrict__ frontier_b,
    int32_t* __restrict__ output_info,
    int32_t initial_frontier_size,
    int32_t initial_depth,
    int32_t depth_limit,
    int32_t max_frontier_for_persist
) {
    __shared__ int32_t s_count;

    int32_t* cur = frontier_a;
    int32_t* nxt = frontier_b;
    int32_t cur_size = initial_frontier_size;
    int32_t depth = initial_depth;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    while (cur_size > 0 && depth < depth_limit) {
        if (cur_size > max_frontier_for_persist) break;

        if (threadIdx.x == 0) s_count = 0;
        __syncthreads();

        int32_t next_depth = depth + 1;

        
        for (int w = warp_id; w < cur_size; w += num_warps) {
            int32_t src = cur[w];
            int32_t start = offsets[src];
            int32_t end = offsets[src + 1];

            for (int32_t e = start + lane; e < end; e += 32) {
                if (!((edge_mask[e >> 5] >> (e & 31)) & 1u)) continue;

                int32_t dst = indices[e];

                uint32_t mask_bit = 1u << (dst & 31);
                if (visited_bitmap[dst >> 5] & mask_bit) continue;

                uint32_t old_word = atomicOr(&visited_bitmap[dst >> 5], mask_bit);
                if (old_word & mask_bit) continue;

                distances[dst] = next_depth;
                if (predecessors) predecessors[dst] = src;
                int pos = atomicAdd(&s_count, 1);
                nxt[pos] = dst;
            }
        }

        __syncthreads();

        cur_size = s_count;
        int32_t* tmp = cur;
        cur = nxt;
        nxt = tmp;
        depth++;
    }

    if (threadIdx.x == 0) {
        output_info[0] = cur_size;
        output_info[1] = depth;
        output_info[2] = (cur == frontier_a) ? 0 : 1;
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
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_mask = graph.edge_mask;

    if (depth_limit < 0) depth_limit = INT32_MAX;

    cache.ensure_buffers((size_t)num_vertices);

    cudaStream_t stream = 0;

    {
        int b = 512, g = (num_vertices + b - 1) / b;
        if (g > 0) bfs_init_kernel<<<g, b, 0, stream>>>(distances, predecessors, cache.d_bitmap, num_vertices);
    }

    {
        int32_t ns = (int32_t)n_sources;
        int b = 256, g = (ns + b - 1) / b;
        if (g > 0) bfs_set_sources_kernel<<<g, b, 0, stream>>>(distances, cache.d_bitmap, cache.d_frontier[0], sources, ns);
    }

    int cur = 0;
    int32_t frontier_size = (int32_t)n_sources;
    int32_t depth = 0;

    while (frontier_size > 0 && depth < depth_limit) {
        if (frontier_size <= PERSIST_THRESHOLD) {
            int old_cur = cur;
            if (frontier_size > 0) {
                bfs_expand_persistent<<<1, 1024, 0, stream>>>(
                    d_offsets, d_indices, d_mask,
                    distances, predecessors, cache.d_bitmap,
                    cache.d_frontier[cur], cache.d_frontier[1 - cur], cache.d_output_info,
                    frontier_size, depth, depth_limit, PERSIST_THRESHOLD
                );
            }

            cudaMemcpyAsync(cache.h_info, cache.d_output_info, 3 * sizeof(int32_t),
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            frontier_size = cache.h_info[0];
            depth = cache.h_info[1];
            cur = (cache.h_info[2] == 0) ? old_cur : (1 - old_cur);
        } else {
            int nxt = 1 - cur;
            cudaMemsetAsync(cache.d_count, 0, sizeof(int32_t), stream);

            int b = 256, g = (frontier_size + 7) / 8;
            bfs_expand_warp<<<g, b, 0, stream>>>(
                d_offsets, d_indices, d_mask,
                distances, predecessors, cache.d_bitmap,
                cache.d_frontier[cur], cache.d_frontier[nxt], cache.d_count,
                frontier_size, depth + 1
            );

            cudaMemcpyAsync(cache.h_info, cache.d_count, sizeof(int32_t),
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            frontier_size = cache.h_info[0];
            cur = nxt;
            depth++;
        }
    }
}

}  
