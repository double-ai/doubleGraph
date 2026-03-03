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
#include <algorithm>

namespace aai {

namespace {

#define UNREACHABLE_DIST 3.4028235e+38f
#define BLOCK_SIZE 256

struct Cache : Cacheable {
    uint32_t* bitmap = nullptr;
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* d_frontier_size = nullptr;
    int32_t* d_changed = nullptr;

    int64_t bitmap_capacity = 0;
    int64_t frontier_a_capacity = 0;
    int64_t frontier_b_capacity = 0;
    bool fsize_allocated = false;
    bool changed_allocated = false;

    void ensure(int32_t num_vertices) {
        int32_t bitmap_words = (num_vertices + 31) / 32;

        if (bitmap_capacity < bitmap_words) {
            if (bitmap) cudaFree(bitmap);
            cudaMalloc(&bitmap, bitmap_words * sizeof(uint32_t));
            bitmap_capacity = bitmap_words;
        }
        if (frontier_a_capacity < num_vertices) {
            if (frontier_a) cudaFree(frontier_a);
            cudaMalloc(&frontier_a, num_vertices * sizeof(int32_t));
            frontier_a_capacity = num_vertices;
        }
        if (frontier_b_capacity < num_vertices) {
            if (frontier_b) cudaFree(frontier_b);
            cudaMalloc(&frontier_b, num_vertices * sizeof(int32_t));
            frontier_b_capacity = num_vertices;
        }
        if (!fsize_allocated) {
            cudaMalloc(&d_frontier_size, sizeof(int32_t));
            fsize_allocated = true;
        }
        if (!changed_allocated) {
            cudaMalloc(&d_changed, sizeof(int32_t));
            changed_allocated = true;
        }
    }

    ~Cache() override {
        if (bitmap) cudaFree(bitmap);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (d_frontier_size) cudaFree(d_frontier_size);
        if (d_changed) cudaFree(d_changed);
    }
};

__device__ __forceinline__ float atomicMinFloat(float* addr, float val) {
    int* addr_as_i = (int*)addr;
    int old = *addr_as_i;
    int assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= val) return __int_as_float(assumed);
        old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(assumed);
}

__device__ __forceinline__ bool is_edge_active(const uint32_t* __restrict__ mask, int32_t e) {
    return (mask[e >> 5] >> (e & 31)) & 1;
}

__global__ void init_kernel(
    float* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t num_vertices,
    int32_t source
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < num_vertices; i += blockDim.x * gridDim.x) {
        dist[i] = (i == source) ? 0.0f : UNREACHABLE_DIST;
        pred[i] = -1;
    }
}

__global__ void relax_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ dist,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    uint32_t* __restrict__ next_bitmap,
    float cutoff
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (blockDim.x * gridDim.x) >> 5;

    for (int i = warp_id; i < frontier_size; i += total_warps) {
        int32_t u = frontier[i];
        float dist_u = dist[u];

        if (dist_u >= cutoff) continue;

        int32_t start = offsets[u];
        int32_t end = offsets[u + 1];

        for (int32_t e = start + lane; e < end; e += 32) {
            if (!is_edge_active(edge_mask, e)) continue;

            int32_t v = __ldg(&indices[e]);
            float w = __ldg(&weights[e]);
            float new_dist = dist_u + w;

            if (new_dist < cutoff) {
                float cur_dist = dist[v];
                if (new_dist < cur_dist) {
                    float old_dist = atomicMinFloat(&dist[v], new_dist);
                    if (new_dist < old_dist) {
                        atomicOr(&next_bitmap[v >> 5], 1u << (v & 31));
                    }
                }
            }
        }
    }
}

__global__ void compact_bitmap_kernel(
    uint32_t* __restrict__ bitmap,
    int32_t* __restrict__ queue,
    int32_t* __restrict__ queue_size,
    int32_t num_words,
    int32_t num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int w = tid; w < num_words; w += blockDim.x * gridDim.x) {
        uint32_t word = bitmap[w];
        if (word == 0) continue;
        bitmap[w] = 0;

        int count = __popc(word);
        int base_pos = atomicAdd(queue_size, count);

        int base_vertex = w << 5;
        int offset = 0;
        while (word) {
            int bit = __ffs(word) - 1;
            int vertex = base_vertex + bit;
            if (vertex < num_vertices) {
                queue[base_pos + offset] = vertex;
                offset++;
            }
            word &= word - 1;
        }
    }
}

__global__ void compute_pred_positive_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t num_vertices
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    for (; u < num_vertices; u += blockDim.x * gridDim.x) {
        float dist_u = dist[u];
        if (dist_u >= UNREACHABLE_DIST) continue;

        int32_t start = offsets[u];
        int32_t end = offsets[u + 1];

        for (int32_t e = start; e < end; e++) {
            if (!is_edge_active(edge_mask, e)) continue;

            int32_t v = __ldg(&indices[e]);
            float w = __ldg(&weights[e]);
            if (dist_u + w == dist[v] && dist_u < dist[v]) {
                pred[v] = u;
            }
        }
    }
}

__global__ void compute_pred_zero_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t* __restrict__ changed,
    int32_t num_vertices,
    int32_t source
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    for (; u < num_vertices; u += blockDim.x * gridDim.x) {
        if (u != source && pred[u] == -1) continue;

        float dist_u = dist[u];
        if (dist_u >= UNREACHABLE_DIST) continue;

        int32_t start = offsets[u];
        int32_t end = offsets[u + 1];

        for (int32_t e = start; e < end; e++) {
            if (!is_edge_active(edge_mask, e)) continue;

            int32_t v = __ldg(&indices[e]);
            if (v == source) continue;
            float w = __ldg(&weights[e]);
            if (dist_u + w == dist[v] && dist[v] == dist_u && pred[v] == -1) {
                pred[v] = u;
                *changed = 1;
            }
        }
    }
}

}  

void sssp_seg_mask(const graph32_t& graph,
                   const float* edge_weights,
                   int32_t source,
                   float* distances,
                   int32_t* predecessors,
                   float cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    const uint32_t* edge_mask = graph.edge_mask;

    cache.ensure(num_vertices);

    int grid = min((num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);
    init_kernel<<<grid, BLOCK_SIZE>>>(distances, predecessors, num_vertices, source);

    int32_t bitmap_words = (num_vertices + 31) / 32;
    cudaMemsetAsync(cache.bitmap, 0, bitmap_words * sizeof(uint32_t));

    cudaMemcpyAsync(cache.frontier_a, &source, sizeof(int32_t), cudaMemcpyHostToDevice);

    int32_t* cur_frontier = cache.frontier_a;
    int32_t* next_frontier = cache.frontier_b;
    int32_t h_frontier_size = 1;

    for (int iter = 0; h_frontier_size > 0; iter++) {
        cudaMemsetAsync(cache.d_frontier_size, 0, sizeof(int32_t));

        if (h_frontier_size > 0) {
            int warps_needed = h_frontier_size;
            int blocks_needed = (int)(((int64_t)warps_needed * 32 + BLOCK_SIZE - 1) / BLOCK_SIZE);
            int relax_grid = min(blocks_needed, 65535);
            relax_warp_kernel<<<relax_grid, BLOCK_SIZE>>>(offsets, indices, edge_weights, edge_mask, distances,
                                                          cur_frontier, h_frontier_size, cache.bitmap, cutoff);
        }

        int compact_grid = min((bitmap_words + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535);
        compact_bitmap_kernel<<<compact_grid, BLOCK_SIZE>>>(cache.bitmap, next_frontier, cache.d_frontier_size, bitmap_words, num_vertices);

        cudaMemcpy(&h_frontier_size, cache.d_frontier_size, sizeof(int32_t), cudaMemcpyDeviceToHost);

        std::swap(cur_frontier, next_frontier);
    }

    
    {
        int64_t pred_grid64 = ((int64_t)num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int pred_grid = (int)(pred_grid64 < 65535 ? pred_grid64 : 65535);
        compute_pred_positive_kernel<<<pred_grid, BLOCK_SIZE>>>(offsets, indices, edge_weights, edge_mask, distances, predecessors, num_vertices);
    }

    
    {
        int64_t pred_grid64 = ((int64_t)num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int pred_grid = (int)(pred_grid64 < 65535 ? pred_grid64 : 65535);
        int32_t h_changed;
        for (int iter = 0; iter < num_vertices; iter++) {
            cudaMemset(cache.d_changed, 0, sizeof(int32_t));
            compute_pred_zero_kernel<<<pred_grid, BLOCK_SIZE>>>(offsets, indices, edge_weights, edge_mask, distances, predecessors, cache.d_changed, num_vertices, source);
            cudaMemcpy(&h_changed, cache.d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost);
            if (h_changed == 0) break;
        }
    }

    cudaDeviceSynchronize();
}

}  
