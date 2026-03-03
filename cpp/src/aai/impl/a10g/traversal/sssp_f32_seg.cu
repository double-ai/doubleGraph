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




__global__ void relax_high_kernel(
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





__global__ void relax_warp_coop_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ dist,
    const uint32_t* __restrict__ frontier_in,
    uint32_t* __restrict__ frontier_out,
    int32_t* __restrict__ updated,
    int32_t seg_start, int32_t seg_end, float cutoff
) {
    int warp_global_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    int32_t my_v = seg_start + warp_global_id * 32 + lane;
    bool in_range = (my_v < seg_end);

    
    bool is_active = false;
    float d_v = SSSP_INF;
    int32_t start_e = 0, end_e = 0;

    if (in_range) {
        uint32_t word = __ldg(&frontier_in[my_v >> 5]);
        is_active = (word & (1u << (my_v & 31))) != 0;
    }

    if (is_active) {
        d_v = dist[my_v];
        if (d_v >= cutoff) is_active = false;
    }

    if (is_active) {
        start_e = __ldg(&offsets[my_v]);
        end_e = __ldg(&offsets[my_v + 1]);
    }

    uint32_t active_mask = __ballot_sync(0xFFFFFFFF, is_active);
    if (active_mask == 0) return;

    bool local_updated = false;

    while (active_mask) {
        int leader = __ffs(active_mask) - 1;

        int32_t v = __shfl_sync(0xFFFFFFFF, my_v, leader);
        float dv = __shfl_sync(0xFFFFFFFF, d_v, leader);
        int32_t se = __shfl_sync(0xFFFFFFFF, start_e, leader);
        int32_t ee = __shfl_sync(0xFFFFFFFF, end_e, leader);

        for (int32_t e = se + lane; e < ee; e += 32) {
            int32_t u = __ldg(&indices[e]);
            float w = __ldg(&weights[e]);
            float nd = dv + w;
            if (nd < cutoff && nd < dist[u]) {
                float old = atomicMinFloat(&dist[u], nd);
                if (nd < old) {
                    atomicOr(&frontier_out[u >> 5], 1u << (u & 31));
                    local_updated = true;
                }
            }
        }

        active_mask &= ~(1u << leader);
    }

    if (__any_sync(0xFFFFFFFF, local_updated)) {
        if (lane == 0) *updated = 1;
    }
}





__global__ void pred_high_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t seg_start, int32_t seg_end
) {
    int32_t u = seg_start + blockIdx.x;
    if (u >= seg_end) return;
    float d_u = dist[u];
    if (d_u >= SSSP_INF) return;
    int32_t start = __ldg(&offsets[u]);
    int32_t end = __ldg(&offsets[u + 1]);
    for (int32_t e = start + threadIdx.x; e < end; e += blockDim.x) {
        int32_t v = __ldg(&indices[e]);
        float w = __ldg(&weights[e]);
        float d_v = dist[v];
        if (d_v < SSSP_INF && d_v == d_u + w && d_u < d_v) pred[v] = u;
    }
}





__global__ void pred_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t seg_start, int32_t seg_end
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int32_t u = seg_start + warp_id;
    if (u >= seg_end) return;
    float d_u = dist[u];
    if (d_u >= SSSP_INF) return;
    int32_t start = __ldg(&offsets[u]);
    int32_t end = __ldg(&offsets[u + 1]);
    for (int32_t e = start + lane; e < end; e += 32) {
        int32_t v = __ldg(&indices[e]);
        float w = __ldg(&weights[e]);
        float d_v = dist[v];
        if (d_v < SSSP_INF && d_v == d_u + w && d_u < d_v) pred[v] = u;
    }
}




__global__ void zerow_pred_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t* __restrict__ changed,
    int32_t source, int32_t N
) {
    int32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N) return;
    float d_u = dist[u];
    if (d_u >= SSSP_INF) return;
    if (u != source && pred[u] == -1) return;  
    int32_t es = offsets[u], ee = offsets[u + 1];
    for (int32_t e = es; e < ee; e++) {
        int32_t v = indices[e];
        if (v == source) continue;
        float w = weights[e];
        if (d_u + w != dist[v]) continue;   
        if (d_u != dist[v]) continue;        
        if (pred[v] == -1) {
            pred[v] = u;
            *changed = 1;
        }
    }
}

struct Cache : Cacheable {
    float* d_dist = nullptr;
    uint32_t* d_frontier[2] = {nullptr, nullptr};
    int32_t* d_updated = nullptr;
    int32_t* h_updated = nullptr;
    size_t alloc_v_ = 0;

    Cache() {
        cudaMalloc(&d_updated, sizeof(int32_t));
        cudaMallocHost(&h_updated, sizeof(int32_t));
    }

    ~Cache() override {
        if (d_dist) cudaFree(d_dist);
        if (d_frontier[0]) cudaFree(d_frontier[0]);
        if (d_frontier[1]) cudaFree(d_frontier[1]);
        if (d_updated) cudaFree(d_updated);
        if (h_updated) cudaFreeHost(h_updated);
    }

    void ensure_buffers(int32_t num_vertices) {
        size_t needed = static_cast<size_t>(num_vertices);
        if (needed <= alloc_v_) return;
        if (d_dist) { cudaFree(d_dist); d_dist = nullptr; }
        if (d_frontier[0]) { cudaFree(d_frontier[0]); d_frontier[0] = nullptr; }
        if (d_frontier[1]) { cudaFree(d_frontier[1]); d_frontier[1] = nullptr; }
        alloc_v_ = needed;
        size_t frontier_words = (needed + 31) / 32;
        cudaMalloc(&d_dist, needed * sizeof(float));
        cudaMalloc(&d_frontier[0], frontier_words * sizeof(uint32_t));
        cudaMalloc(&d_frontier[1], frontier_words * sizeof(uint32_t));
    }
};

}  

void sssp_seg(const graph32_t& graph,
              const float* edge_weights,
              int32_t source,
              float* distances,
              int32_t* predecessors,
              float cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    cache.ensure_buffers(num_vertices);

    const auto& seg = graph.segment_offsets.value();
    int32_t sh0 = seg[0], sh1 = seg[1];  
    int32_t sm0 = seg[1];                  
    int32_t sl1 = seg[3];                  

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const float* d_weights = edge_weights;

    int32_t num_frontier_words = (num_vertices + 31) / 32;

    
    init_kernel<<<(num_vertices + 255) / 256, 256>>>(cache.d_dist, num_vertices, source);
    cudaMemsetAsync(cache.d_frontier[0], 0, num_frontier_words * sizeof(uint32_t));
    cudaMemsetAsync(cache.d_frontier[1], 0, num_frontier_words * sizeof(uint32_t));
    *cache.h_updated = (int32_t)(1u << (source & 31));
    cudaMemcpyAsync(cache.d_frontier[0] + (source >> 5), cache.h_updated, sizeof(uint32_t), cudaMemcpyHostToDevice);

    int curr = 0;
    const int BATCH = 4;  

    for (int batch_start = 0; batch_start < num_vertices; batch_start += BATCH) {
        int batch_end = batch_start + BATCH;
        if (batch_end > num_vertices) batch_end = num_vertices;

        cudaMemsetAsync(cache.d_updated, 0, sizeof(int32_t));

        for (int iter = batch_start; iter < batch_end; iter++) {
            int next = 1 - curr;
            cudaMemsetAsync(cache.d_frontier[next], 0, num_frontier_words * sizeof(uint32_t));

            
            int n_high = sh1 - sh0;
            if (n_high > 0) {
                relax_high_kernel<<<n_high, 512>>>(d_offsets, d_indices, d_weights, cache.d_dist,
                    cache.d_frontier[curr], cache.d_frontier[next], cache.d_updated, sh0, sh1, cutoff);
            }

            
            int num_verts_ml = sl1 - sm0;
            if (num_verts_ml > 0) {
                int num_warps = (num_verts_ml + 31) / 32;
                int threads = 256;
                int warps_per_block = threads / 32;
                int blocks = (num_warps + warps_per_block - 1) / warps_per_block;
                relax_warp_coop_kernel<<<blocks, threads>>>(d_offsets, d_indices, d_weights, cache.d_dist,
                    cache.d_frontier[curr], cache.d_frontier[next], cache.d_updated, sm0, sl1, cutoff);
            }

            curr = next;
        }

        cudaMemcpyAsync(cache.h_updated, cache.d_updated, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(0);
        if (*cache.h_updated == 0) break;
    }

    
    cudaMemcpyAsync(distances, cache.d_dist,
                    num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);

    
    if (predecessors) {
        cudaMemsetAsync(predecessors, 0xFF, num_vertices * sizeof(int32_t));

        
        int n_high = sh1 - sh0;
        if (n_high > 0) {
            pred_high_kernel<<<n_high, 512>>>(d_offsets, d_indices, d_weights, cache.d_dist, predecessors, sh0, sh1);
        }

        int n_ml = sl1 - sm0;
        if (n_ml > 0) {
            int threads = 256;
            int blocks = (n_ml + threads / 32 - 1) / (threads / 32);
            pred_warp_kernel<<<blocks, threads>>>(d_offsets, d_indices, d_weights, cache.d_dist, predecessors, sm0, sl1);
        }

        
        int32_t h_ch = 1;
        for (int ziter = 0; ziter < num_vertices && h_ch; ziter++) {
            cudaMemsetAsync(cache.d_updated, 0, sizeof(int32_t));
            int grid = (num_vertices + 255) / 256;
            zerow_pred_kernel<<<grid, 256>>>(
                d_offsets, d_indices, d_weights, cache.d_dist, predecessors,
                cache.d_updated, source, num_vertices);
            cudaMemcpy(&h_ch, cache.d_updated, sizeof(int32_t), cudaMemcpyDeviceToHost);
        }
    }

    cudaDeviceSynchronize();
}

}  
