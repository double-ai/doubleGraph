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
#include <math_constants.h>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>

namespace aai {

namespace {

#define FLT_MAX_INT 0x7F7FFFFF
__device__ __forceinline__ unsigned long long pack_dp(float dist, int32_t pred) {
    return ((unsigned long long)__float_as_uint(dist) << 32) | (unsigned long long)(unsigned int)pred;
}

#define DP_UNREACHABLE (((unsigned long long)FLT_MAX_INT << 32) | 0xFFFFFFFFULL)

__global__ void init_sssp_kernel(
    unsigned long long* __restrict__ dp,
    int32_t* __restrict__ updated,
    int32_t num_vertices,
    int32_t source
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;
    dp[tid] = (tid == source) ? pack_dp(0.0f, -1) : DP_UNREACHABLE;
    updated[tid] = 0;
}

__global__ void relax_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    unsigned long long* __restrict__ dp,
    const int32_t* __restrict__ frontier,
    const int32_t* __restrict__ frontier_count_ptr,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_count,
    int32_t* __restrict__ updated,
    int32_t cutoff_int
) {
    int32_t frontier_size = *frontier_count_ptr;

    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;
    int32_t total_warps = (gridDim.x * blockDim.x) >> 5;

    for (int32_t w = warp_id; w < frontier_size; w += total_warps) {
        int32_t u = frontier[w];
        unsigned long long u_dp = dp[u];
        int32_t dist_u_int = (int32_t)(unsigned int)(u_dp >> 32);
        if (dist_u_int >= cutoff_int) continue;

        float dist_u = __int_as_float(dist_u_int);
        int32_t start = offsets[u];
        int32_t end = offsets[u + 1];

        for (int32_t e = start + lane; e < end; e += 32) {
            if (!((edge_mask[e >> 5] >> (e & 31)) & 1)) continue;

            int32_t v = indices[e];
            float ew = edge_weights[e];
            float new_dist = dist_u + ew;
            int new_dist_int = __float_as_int(new_dist);

            if (new_dist_int >= cutoff_int) continue;

            unsigned long long new_val = pack_dp(new_dist, u);
            unsigned long long old_val = atomicMin(&dp[v], new_val);

            int32_t old_dist_int = (int32_t)(unsigned int)(old_val >> 32);
            if (new_dist_int < old_dist_int) {
                if (atomicExch(&updated[v], 1) == 0) {
                    int pos = atomicAdd(next_frontier_count, 1);
                    next_frontier[pos] = v;
                }
            }
        }
    }
}

__global__ void clear_updated_kernel(
    int32_t* __restrict__ updated,
    const int32_t* __restrict__ frontier,
    const int32_t* __restrict__ frontier_count_ptr
) {
    int32_t frontier_size = *frontier_count_ptr;
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;
    for (int32_t i = tid; i < frontier_size; i += stride) {
        updated[frontier[i]] = 0;
    }
}

__global__ void unpack_kernel(
    const unsigned long long* __restrict__ dp,
    float* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t num_vertices,
    int32_t source
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;

    unsigned long long packed = dp[tid];
    distances[tid] = __uint_as_float((unsigned int)(packed >> 32));
    predecessors[tid] = (tid == source) ? -1 : (int32_t)(unsigned int)(packed & 0xFFFFFFFFULL);
}

__global__ void fix_predecessors_positive_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t num_vertices,
    int32_t source
) {
    const int32_t lane = threadIdx.x & 31;
    const int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int32_t num_warps = (gridDim.x * blockDim.x) >> 5;

    for (int32_t u = warp_id; u < num_vertices; u += num_warps) {
        float d_u = dist[u];
        if (d_u >= FLT_MAX) continue;

        int32_t start = offsets[u];
        int32_t end = offsets[u + 1];

        for (int32_t e = start + lane; e < end; e += 32) {
            if (!((edge_mask[e >> 5] >> (e & 31)) & 1)) continue;

            int32_t v = indices[e];
            if (v == source) continue;

            float d_v = dist[v];
            if (d_v >= FLT_MAX) continue;

            float w = edge_weights[e];
            if (d_u + w == d_v && d_u < d_v) {
                pred[v] = u;
            }
        }
    }
}

__global__ void fix_predecessors_zero_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t* __restrict__ changed,
    int32_t num_vertices,
    int32_t source
) {
    const int32_t lane = threadIdx.x & 31;
    const int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int32_t num_warps = (gridDim.x * blockDim.x) >> 5;

    for (int32_t u = warp_id; u < num_vertices; u += num_warps) {
        if (u != source && pred[u] == -1) continue;
        float d_u = dist[u];
        if (d_u >= FLT_MAX) continue;

        int32_t start = offsets[u];
        int32_t end = offsets[u + 1];

        for (int32_t e = start + lane; e < end; e += 32) {
            if (!((edge_mask[e >> 5] >> (e & 31)) & 1)) continue;

            int32_t v = indices[e];
            if (v == source) continue;
            float w = edge_weights[e];

            if (d_u + w == dist[v] && dist[v] == d_u && pred[v] == -1) {
                if (atomicCAS(&pred[v], -1, u) == -1) {
                    *changed = 1;
                }
            }
        }
    }
}

struct Cache : Cacheable {
    int32_t* d_frontier_a = nullptr;
    int32_t* d_frontier_b = nullptr;
    int32_t* d_updated = nullptr;
    int32_t* d_count_a = nullptr;
    int32_t* d_count_b = nullptr;
    unsigned long long* d_dp = nullptr;
    int32_t* d_changed = nullptr;
    int32_t* h_count = nullptr;
    int32_t max_vertices = 0;
    int32_t num_sms = 0;

    void ensure_buffers(int32_t num_vertices) {
        if (num_vertices <= max_vertices) return;
        free_buffers();

        cudaMalloc(&d_frontier_a, (size_t)num_vertices * sizeof(int32_t));
        cudaMalloc(&d_frontier_b, (size_t)num_vertices * sizeof(int32_t));
        cudaMalloc(&d_updated, (size_t)num_vertices * sizeof(int32_t));
        cudaMalloc(&d_dp, (size_t)num_vertices * sizeof(unsigned long long));
        cudaMalloc(&d_count_a, sizeof(int32_t));
        cudaMalloc(&d_count_b, sizeof(int32_t));
        cudaMalloc(&d_changed, sizeof(int32_t));
        cudaMallocHost(&h_count, sizeof(int32_t));

        if (num_sms == 0) {
            cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        }

        max_vertices = num_vertices;
    }

    void free_buffers() {
        if (d_frontier_a) { cudaFree(d_frontier_a); d_frontier_a = nullptr; }
        if (d_frontier_b) { cudaFree(d_frontier_b); d_frontier_b = nullptr; }
        if (d_updated) { cudaFree(d_updated); d_updated = nullptr; }
        if (d_dp) { cudaFree(d_dp); d_dp = nullptr; }
        if (d_count_a) { cudaFree(d_count_a); d_count_a = nullptr; }
        if (d_count_b) { cudaFree(d_count_b); d_count_b = nullptr; }
        if (d_changed) { cudaFree(d_changed); d_changed = nullptr; }
        if (h_count) { cudaFreeHost(h_count); h_count = nullptr; }
        max_vertices = 0;
    }

    ~Cache() override {
        free_buffers();
    }
};

}  

void sssp_mask(const graph32_t& graph,
               const float* edge_weights,
               int32_t source,
               float* distances,
               int32_t* predecessors,
               float cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    if (std::isinf(cutoff)) {
        cutoff = std::numeric_limits<float>::max();
    }

    int32_t cutoff_int;
    std::memcpy(&cutoff_int, &cutoff, sizeof(int32_t));

    cache.ensure_buffers(num_vertices);

    cudaStream_t stream = 0;

    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    init_sssp_kernel<<<grid, block, 0, stream>>>(cache.d_dp, cache.d_updated, num_vertices, source);

    int32_t one = 1;
    cudaMemcpyAsync(cache.d_frontier_a, &source, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(cache.d_count_a, &one, sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    int32_t* cur_frontier = cache.d_frontier_a;
    int32_t* next_frontier = cache.d_frontier_b;
    int32_t* cur_count = cache.d_count_a;
    int32_t* next_count = cache.d_count_b;

    const int BATCH_SIZE = 64;
    int32_t max_grid = cache.num_sms * 32;
    int32_t frontier_size = 1;

    for (int outer = 0; outer < num_vertices; outer += BATCH_SIZE) {
        int32_t grid_blocks = std::min(max_grid,
            std::max(1, (frontier_size * 32 + 255) / 256));
        grid_blocks = std::max(grid_blocks, (int32_t)(cache.num_sms * 4));

        int batch_end = std::min(outer + BATCH_SIZE, (int)num_vertices);

        for (int iter = outer; iter < batch_end; iter++) {
            cudaMemsetAsync(next_count, 0, sizeof(int32_t), stream);

            relax_edges_kernel<<<grid_blocks, 256, 0, stream>>>(
                d_offsets, d_indices, edge_weights, d_edge_mask, cache.d_dp,
                cur_frontier, cur_count,
                next_frontier, next_count,
                cache.d_updated, cutoff_int);

            clear_updated_kernel<<<grid_blocks, 256, 0, stream>>>(
                cache.d_updated, next_frontier, next_count);

            std::swap(cur_frontier, next_frontier);
            std::swap(cur_count, next_count);
        }

        cudaMemcpyAsync(cache.h_count, cur_count, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        frontier_size = *cache.h_count;
        if (frontier_size == 0) break;
    }

    unpack_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
        cache.d_dp, distances, predecessors, num_vertices, source);

    cudaMemsetAsync(predecessors, 0xFF, (size_t)num_vertices * sizeof(int32_t), stream);

    
    {
        int warps_per_block = 256 / 32;
        int64_t total_warps = ((int64_t)num_vertices + warps_per_block - 1) / warps_per_block;
        int fix_grid = (int)(total_warps < 65535 ? total_warps : 65535);
        if (fix_grid < 1) fix_grid = 1;
        fix_predecessors_positive_kernel<<<fix_grid, 256, 0, stream>>>(
            d_offsets, d_indices, edge_weights, d_edge_mask, distances, predecessors, num_vertices, source
        );
    }

    
    {
        int warps_per_block = 256 / 32;
        int64_t total_warps = ((int64_t)num_vertices + warps_per_block - 1) / warps_per_block;
        int fix_grid = (int)(total_warps < 65535 ? total_warps : 65535);
        if (fix_grid < 1) fix_grid = 1;

        int32_t h_changed;
        for (int iter = 0; iter < num_vertices; iter++) {
            cudaMemsetAsync(cache.d_changed, 0, sizeof(int32_t), stream);
            fix_predecessors_zero_kernel<<<fix_grid, 256, 0, stream>>>(
                d_offsets, d_indices, edge_weights, d_edge_mask, distances, predecessors,
                cache.d_changed, num_vertices, source);
            cudaMemcpyAsync(cache.h_count, cache.d_changed, sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            h_changed = *cache.h_count;
            if (h_changed == 0) break;
        }
    }

    cudaStreamSynchronize(stream);
}

}  
