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
#include <cfloat>
#include <cstdint>

namespace aai {

namespace {

#define SSSP_INF __FLT_MAX__

struct Cache : Cacheable {
    float* dist = nullptr;
    uint32_t* frontier0 = nullptr;
    uint32_t* frontier1 = nullptr;
    int32_t* updated = nullptr;
    int32_t capacity = 0;

    void ensure(int32_t num_vertices) {
        if (capacity < num_vertices) {
            if (dist) cudaFree(dist);
            if (frontier0) cudaFree(frontier0);
            if (frontier1) cudaFree(frontier1);
            if (updated) cudaFree(updated);

            int32_t frontier_words = (num_vertices + 31) / 32;
            cudaMalloc(&dist, num_vertices * sizeof(float));
            cudaMalloc(&frontier0, frontier_words * sizeof(uint32_t));
            cudaMalloc(&frontier1, frontier_words * sizeof(uint32_t));
            cudaMalloc(&updated, sizeof(int32_t));
            capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (dist) cudaFree(dist);
        if (frontier0) cudaFree(frontier0);
        if (frontier1) cudaFree(frontier1);
        if (updated) cudaFree(updated);
    }
};

__device__ __forceinline__ float atomicMinFloat(float* addr, float val) {
    return __int_as_float(atomicMin((int*)addr, __float_as_int(val)));
}

__global__ void init_kernel(
    float* __restrict__ dist, int32_t num_vertices, int32_t source
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        dist[tid] = (tid == source) ? 0.0f : SSSP_INF;
    }
}


__global__ __launch_bounds__(512, 3)
void relax_high_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ dist,
    const uint32_t* __restrict__ frontier_in,
    uint32_t* __restrict__ frontier_out,
    int32_t* __restrict__ updated,
    int32_t seg_start, int32_t seg_end, float cutoff
) {
    int32_t v = seg_start + blockIdx.x;
    if (v >= seg_end) return;

    uint32_t word = __ldg(&frontier_in[v >> 5]);
    if (!(word & (1u << (v & 31)))) return;

    float d_v = dist[v];
    if (d_v >= cutoff) return;

    int32_t start = __ldg(&offsets[v]);
    int32_t end = __ldg(&offsets[v + 1]);

    bool local_updated = false;
    for (int32_t e = start + threadIdx.x; e < end; e += blockDim.x) {
        int32_t u = __ldg(&indices[e]);
        float w = __ldg(&weights[e]);
        float new_dist = d_v + w;
        if (new_dist < cutoff && new_dist < dist[u]) {
            float old = atomicMinFloat(&dist[u], new_dist);
            if (new_dist < old) {
                atomicOr(&frontier_out[u >> 5], 1u << (u & 31));
                local_updated = true;
            }
        }
    }
    if (__any_sync(0xFFFFFFFF, local_updated)) {
        if ((threadIdx.x & 31) == 0) *updated = 1;
    }
}


__global__ __launch_bounds__(256, 6)
void relax_mid_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ dist,
    const uint32_t* __restrict__ frontier_in,
    uint32_t* __restrict__ frontier_out,
    int32_t* __restrict__ updated,
    int32_t seg_start, int32_t seg_end, float cutoff
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;

    for (int32_t v = seg_start + warp_id; v < seg_end; v += num_warps) {
        uint32_t word = __ldg(&frontier_in[v >> 5]);
        if (!(word & (1u << (v & 31)))) continue;

        float d_v = dist[v];
        if (d_v >= cutoff) continue;

        int32_t start = __ldg(&offsets[v]);
        int32_t end = __ldg(&offsets[v + 1]);

        for (int32_t e = start + lane; e < end; e += 32) {
            int32_t u = __ldg(&indices[e]);
            float w = __ldg(&weights[e]);
            float new_dist = d_v + w;
            if (new_dist < cutoff && new_dist < dist[u]) {
                float old = atomicMinFloat(&dist[u], new_dist);
                if (new_dist < old) {
                    atomicOr(&frontier_out[u >> 5], 1u << (u & 31));
                    *updated = 1;
                }
            }
        }
    }
}


__global__ __launch_bounds__(256, 8)
void relax_low_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ dist,
    const uint32_t* __restrict__ frontier_in,
    uint32_t* __restrict__ frontier_out,
    int32_t* __restrict__ updated,
    int32_t seg_start, int32_t seg_end, float cutoff
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int32_t v = seg_start + tid; v < seg_end; v += stride) {
        uint32_t word = __ldg(&frontier_in[v >> 5]);
        if (!(word & (1u << (v & 31)))) continue;

        float d_v = dist[v];
        if (d_v >= cutoff) continue;

        int32_t start = __ldg(&offsets[v]);
        int32_t end = __ldg(&offsets[v + 1]);

        for (int32_t e = start; e < end; e++) {
            int32_t u = __ldg(&indices[e]);
            float w = __ldg(&weights[e]);
            float new_dist = d_v + w;
            if (new_dist < cutoff && new_dist < dist[u]) {
                float old = atomicMinFloat(&dist[u], new_dist);
                if (new_dist < old) {
                    atomicOr(&frontier_out[u >> 5], 1u << (u & 31));
                    *updated = 1;
                }
            }
        }
    }
}


__global__ void pred_positive_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int32_t u = tid; u < num_vertices; u += stride) {
        float d_u = dist[u];
        if (d_u >= SSSP_INF) continue;
        int32_t start = __ldg(&offsets[u]);
        int32_t end = __ldg(&offsets[u + 1]);
        for (int32_t e = start; e < end; e++) {
            int32_t v = __ldg(&indices[e]);
            float w = __ldg(&weights[e]);
            float dv = dist[v];
            if (d_u + w == dv && d_u < dv) {
                pred[v] = u;
            }
        }
    }
}


__global__ void pred_zero_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t* __restrict__ changed,
    int32_t num_vertices,
    int32_t source
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int32_t u = tid; u < num_vertices; u += stride) {
        if (u != source && pred[u] == -1) continue;
        float d_u = dist[u];
        if (d_u >= SSSP_INF) continue;
        int32_t start = __ldg(&offsets[u]);
        int32_t end = __ldg(&offsets[u + 1]);
        for (int32_t e = start; e < end; e++) {
            int32_t v = __ldg(&indices[e]);
            if (v == source) continue;
            float w = __ldg(&weights[e]);
            if (d_u + w == dist[v] && dist[v] == d_u && pred[v] == -1) {
                pred[v] = u;
                *changed = 1;
            }
        }
    }
}

}  

void sssp_seg(const graph32_t& graph,
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

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0];
    int32_t seg1 = seg[1];
    int32_t seg2 = seg[2];
    int32_t seg3 = seg[3];

    cache.ensure(num_vertices);

    float* dist = cache.dist;
    uint32_t* frontier0 = cache.frontier0;
    uint32_t* frontier1 = cache.frontier1;
    int32_t* updated = cache.updated;

    int32_t num_frontier_words = (num_vertices + 31) / 32;

    
    init_kernel<<<(num_vertices + 255) / 256, 256>>>(dist, num_vertices, source);
    cudaMemsetAsync(frontier0, 0, num_frontier_words * sizeof(uint32_t));
    cudaMemsetAsync(frontier1, 0, num_frontier_words * sizeof(uint32_t));

    
    uint32_t src_bit = 1u << (source & 31);
    cudaMemcpy(frontier0 + (source >> 5), &src_bit,
               sizeof(uint32_t), cudaMemcpyHostToDevice);

    int num_high = seg1 - seg0;
    int num_mid = seg2 - seg1;
    int num_low = seg3 - seg2;

    int grid_high = num_high;

    int grid_mid = (num_mid + 7) / 8;  
    if (grid_mid > 65535) grid_mid = 65535;
    if (grid_mid < 1 && num_mid > 0) grid_mid = 1;

    int grid_low = (num_low + 255) / 256;
    if (grid_low > 65535) grid_low = 65535;

    uint32_t* curr = frontier0;
    uint32_t* next = frontier1;
    const int BATCH = 8;
    int32_t h_updated;

    for (int batch_start = 0; batch_start < num_vertices; batch_start += BATCH) {
        int batch_end = batch_start + BATCH;
        if (batch_end > num_vertices) batch_end = num_vertices;

        cudaMemsetAsync(updated, 0, sizeof(int32_t));

        for (int iter = batch_start; iter < batch_end; iter++) {
            cudaMemsetAsync(next, 0, num_frontier_words * sizeof(uint32_t));

            if (num_high > 0) {
                relax_high_kernel<<<grid_high, 512>>>(
                    offsets, indices, edge_weights, dist,
                    curr, next, updated, seg0, seg1, cutoff);
            }

            if (num_mid > 0) {
                relax_mid_kernel<<<grid_mid, 256>>>(
                    offsets, indices, edge_weights, dist,
                    curr, next, updated, seg1, seg2, cutoff);
            }

            if (num_low > 0) {
                relax_low_kernel<<<grid_low, 256>>>(
                    offsets, indices, edge_weights, dist,
                    curr, next, updated, seg2, seg3, cutoff);
            }

            uint32_t* tmp = curr;
            curr = next;
            next = tmp;
        }

        cudaMemcpy(&h_updated, updated, sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (h_updated == 0) break;
    }

    
    cudaMemcpyAsync(distances, dist,
                    num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);

    
    if (predecessors) {
        cudaMemsetAsync(predecessors, 0xFF, num_vertices * sizeof(int32_t));

        
        int pred_grid = (num_vertices + 255) / 256;
        pred_positive_kernel<<<pred_grid, 256>>>(
            offsets, indices, edge_weights, dist,
            predecessors, num_vertices);

        
        
        int32_t h_changed = 1;
        for (int iter = 0; iter < num_vertices && h_changed; iter++) {
            cudaMemsetAsync(updated, 0, sizeof(int32_t));
            pred_zero_kernel<<<pred_grid, 256>>>(
                offsets, indices, edge_weights, dist,
                predecessors, updated, num_vertices, source);
            cudaMemcpy(&h_changed, updated, sizeof(int32_t), cudaMemcpyDeviceToHost);
        }
    }

    cudaDeviceSynchronize();
}

}  
