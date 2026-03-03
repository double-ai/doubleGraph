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
#include <cmath>
#include <algorithm>
#include <vector>

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define MAX_BATCH 16


__global__ void init_all_kernel(
    int32_t* __restrict__ distances,
    float* __restrict__ sigmas,
    float* __restrict__ deltas,
    int num_vertices);

struct SourceState {
    int32_t* d_distances = nullptr;
    float* d_sigmas = nullptr;
    float* d_deltas = nullptr;
    int32_t* d_stack = nullptr;
    int32_t* d_next_count = nullptr;
    std::vector<int32_t> level_start;
    std::vector<int32_t> level_end;
    int current_level;
    int stack_top;
    int frontier_size;
    int32_t source_vertex;
    bool active;
};

struct Cache : Cacheable {
    SourceState src_state[MAX_BATCH];
    cudaStream_t streams[MAX_BATCH];
    int32_t* h_next_counts = nullptr;
    bool* d_is_source = nullptr;
    int alloc_V = 0;

    Cache() {
        for (int i = 0; i < MAX_BATCH; i++) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
            src_state[i] = {};
        }
        cudaMallocHost(&h_next_counts, MAX_BATCH * sizeof(int32_t));
    }

    ~Cache() override {
        for (int i = 0; i < MAX_BATCH; i++) {
            auto& s = src_state[i];
            if (s.d_distances) cudaFree(s.d_distances);
            if (s.d_sigmas) cudaFree(s.d_sigmas);
            if (s.d_deltas) cudaFree(s.d_deltas);
            if (s.d_stack) cudaFree(s.d_stack);
            if (s.d_next_count) cudaFree(s.d_next_count);
            if (streams[i]) cudaStreamDestroy(streams[i]);
        }
        if (d_is_source) cudaFree(d_is_source);
        if (h_next_counts) cudaFreeHost(h_next_counts);
    }

    void ensure_buffers(int V) {
        if (V <= alloc_V) return;
        for (int i = 0; i < MAX_BATCH; i++) {
            auto& s = src_state[i];
            if (s.d_distances) cudaFree(s.d_distances);
            if (s.d_sigmas) cudaFree(s.d_sigmas);
            if (s.d_deltas) cudaFree(s.d_deltas);
            if (s.d_stack) cudaFree(s.d_stack);
            if (s.d_next_count) cudaFree(s.d_next_count);

            cudaMalloc(&s.d_distances, (size_t)V * sizeof(int32_t));
            cudaMalloc(&s.d_sigmas, (size_t)V * sizeof(float));
            cudaMalloc(&s.d_deltas, (size_t)V * sizeof(float));
            cudaMalloc(&s.d_stack, (size_t)V * sizeof(int32_t));
            cudaMalloc(&s.d_next_count, sizeof(int32_t));

            
            int blocks = (V + BLOCK_SIZE - 1) / BLOCK_SIZE;
            if (blocks > 0)
                init_all_kernel<<<blocks, BLOCK_SIZE, 0, streams[i]>>>(
                    s.d_distances, s.d_sigmas, s.d_deltas, V);
        }
        if (d_is_source) cudaFree(d_is_source);
        cudaMalloc(&d_is_source, (size_t)V * sizeof(bool));

        
        for (int i = 0; i < MAX_BATCH; i++)
            cudaStreamSynchronize(streams[i]);

        alloc_V = V;
    }
};





__device__ __forceinline__ bool is_edge_active(const uint32_t* __restrict__ mask, int e) {
    return (mask[e >> 5] >> (e & 31)) & 1;
}

__global__ void init_source_kernel(
    int32_t* __restrict__ distances,
    float* __restrict__ sigmas,
    int32_t* __restrict__ stack,
    int32_t* __restrict__ next_count,
    int source
) {
    if (threadIdx.x == 0) {
        distances[source] = 0;
        sigmas[source] = 1.0f;
        stack[0] = source;
        *next_count = 0;
    }
}

__global__ void init_all_kernel(
    int32_t* __restrict__ distances,
    float* __restrict__ sigmas,
    float* __restrict__ deltas,
    int num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;
    distances[tid] = -1;
    sigmas[tid] = 0.0f;
    deltas[tid] = 0.0f;
}

__global__ void bfs_forward_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    float* __restrict__ sigmas,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count,
    int frontier_size,
    int new_dist
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int src = frontier[warp_id];
    int start = offsets[src];
    int end = offsets[src + 1];
    float src_sigma = sigmas[src];

    for (int e = start + lane; e < end; e += 32) {
        if (!is_edge_active(edge_mask, e)) continue;
        int dst = indices[e];

        int cur_dist = distances[dst];
        if (cur_dist != -1 && cur_dist != new_dist) continue;

        if (cur_dist == new_dist) {
            atomicAdd(&sigmas[dst], src_sigma);
            continue;
        }

        int old = atomicCAS(&distances[dst], -1, new_dist);
        if (old != -1 && old != new_dist) continue;

        atomicAdd(&sigmas[dst], src_sigma);
        if (old == -1) {
            int pos = atomicAdd(next_count, 1);
            next_frontier[pos] = dst;
        }
    }
}

__global__ void bc_backward_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ distances,
    const float* __restrict__ sigmas,
    float* __restrict__ deltas,
    const int32_t* __restrict__ level_verts,
    int level_size,
    int next_level
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= level_size) return;

    int v = level_verts[warp_id];
    int start = offsets[v];
    int end = offsets[v + 1];
    float sigma_v = sigmas[v];
    float delta_v = 0.0f;

    for (int e = start + lane; e < end; e += 32) {
        if (!is_edge_active(edge_mask, e)) continue;
        int w = indices[e];
        if (distances[w] == next_level) {
            delta_v += (sigma_v / sigmas[w]) * (1.0f + deltas[w]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        delta_v += __shfl_down_sync(0xFFFFFFFF, delta_v, offset);
    }

    if (lane == 0) {
        deltas[v] = delta_v;
    }
}

__global__ void bc_accumulate_kernel(
    float* __restrict__ bc,
    const float* __restrict__ deltas,
    const int32_t* __restrict__ stack,
    int total_discovered,
    int source,
    bool include_endpoints,
    int reachable_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_discovered) return;
    int v = stack[tid];
    if (v == source) {
        if (include_endpoints && reachable_count > 0)
            atomicAdd(&bc[v], (float)reachable_count);
    } else {
        float val = deltas[v];
        if (include_endpoints) val += 1.0f;
        atomicAdd(&bc[v], val);
    }
}

__global__ void partial_reset_kernel(
    int32_t* __restrict__ distances,
    float* __restrict__ sigmas,
    float* __restrict__ deltas,
    const int32_t* __restrict__ stack,
    int total_discovered
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_discovered) return;
    int v = stack[tid];
    distances[v] = -1;
    sigmas[v] = 0.0f;
    deltas[v] = 0.0f;
}

__global__ void mark_sources_kernel(bool* __restrict__ is_source, const int32_t* __restrict__ sv, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) is_source[sv[tid]] = true;
}

__global__ void normalize_kernel(float* __restrict__ bc, const bool* __restrict__ is_source,
    int nv, float inv_src, float inv_nsrc, bool uniform) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nv) return;
    if (uniform) bc[tid] *= inv_src;
    else bc[tid] *= (is_source[tid] ? inv_src : inv_nsrc);
}

}  

void betweenness_centrality_seg_mask(const graph32_t& graph,
                                      float* centralities,
                                      bool normalized,
                                      bool include_endpoints,
                                      const int32_t* sample_vertices,
                                      std::size_t num_samples) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;
    bool is_symmetric = graph.is_symmetric;

    cache.ensure_buffers(num_vertices);

    
    cudaMemsetAsync(centralities, 0, num_vertices * sizeof(float), cache.streams[0]);
    cudaStreamSynchronize(cache.streams[0]);

    
    std::vector<int32_t> h_samples(num_samples);
    cudaMemcpy(h_samples.data(), sample_vertices, num_samples * sizeof(int32_t), cudaMemcpyDeviceToHost);

    for (int batch_start = 0; batch_start < (int)num_samples; batch_start += MAX_BATCH) {
        int batch_end = std::min(batch_start + MAX_BATCH, (int)num_samples);
        int batch_count = batch_end - batch_start;

        
        for (int i = 0; i < batch_count; i++) {
            auto& s = cache.src_state[i];
            s.source_vertex = h_samples[batch_start + i];
            s.level_start.clear();
            s.level_end.clear();
            s.level_start.push_back(0);
            s.level_end.push_back(1);
            s.current_level = 0;
            s.stack_top = 1;
            s.frontier_size = 1;
            s.active = true;

            init_source_kernel<<<1, 32, 0, cache.streams[i]>>>(
                s.d_distances, s.d_sigmas, s.d_stack,
                s.d_next_count, s.source_vertex);
        }

        
        bool any_active = true;
        while (any_active) {
            for (int i = 0; i < batch_count; i++) {
                auto& s = cache.src_state[i];
                if (!s.active) continue;

                cudaMemsetAsync(s.d_next_count, 0, sizeof(int32_t), cache.streams[i]);
                int fs = s.frontier_size;
                if (fs > 0) {
                    int64_t total = (int64_t)fs * 32;
                    int blocks = (int)((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
                    bfs_forward_warp_kernel<<<blocks, BLOCK_SIZE, 0, cache.streams[i]>>>(
                        d_offsets, d_indices, d_edge_mask,
                        s.d_distances, s.d_sigmas,
                        s.d_stack + s.level_start[s.current_level],
                        s.d_stack + s.stack_top, s.d_next_count,
                        fs, s.current_level + 1);
                }
                cudaMemcpyAsync(&cache.h_next_counts[i], s.d_next_count, sizeof(int32_t),
                                cudaMemcpyDeviceToHost, cache.streams[i]);
            }

            cudaDeviceSynchronize();

            any_active = false;
            for (int i = 0; i < batch_count; i++) {
                auto& s = cache.src_state[i];
                if (!s.active) continue;
                s.frontier_size = cache.h_next_counts[i];
                if (s.frontier_size > 0) {
                    s.current_level++;
                    s.level_start.push_back(s.stack_top);
                    s.stack_top += s.frontier_size;
                    s.level_end.push_back(s.stack_top);
                    any_active = true;
                } else {
                    s.active = false;
                }
            }
            if (!any_active) {
                for (int i = 0; i < batch_count; i++)
                    if (cache.src_state[i].active) { any_active = true; break; }
            }
        }

        
        for (int i = 0; i < batch_count; i++) {
            auto& s = cache.src_state[i];
            for (int L = s.current_level - 1; L >= 0; L--) {
                int lsize = s.level_end[L] - s.level_start[L];
                if (lsize > 0) {
                    int64_t total = (int64_t)lsize * 32;
                    int blocks = (int)((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
                    bc_backward_warp_kernel<<<blocks, BLOCK_SIZE, 0, cache.streams[i]>>>(
                        d_offsets, d_indices, d_edge_mask,
                        s.d_distances, s.d_sigmas, s.d_deltas,
                        s.d_stack + s.level_start[L], lsize, L + 1);
                }
            }
            {
                int blocks = (s.stack_top + BLOCK_SIZE - 1) / BLOCK_SIZE;
                if (blocks > 0)
                    bc_accumulate_kernel<<<blocks, BLOCK_SIZE, 0, cache.streams[i]>>>(
                        centralities, s.d_deltas, s.d_stack,
                        s.stack_top, s.source_vertex, include_endpoints, s.stack_top - 1);
            }
            {
                int blocks = (s.stack_top + BLOCK_SIZE - 1) / BLOCK_SIZE;
                if (blocks > 0)
                    partial_reset_kernel<<<blocks, BLOCK_SIZE, 0, cache.streams[i]>>>(
                        s.d_distances, s.d_sigmas, s.d_deltas,
                        s.d_stack, s.stack_top);
            }
        }

        for (int i = 0; i < batch_count; i++)
            cudaStreamSynchronize(cache.streams[i]);
    }

    
    int n = num_vertices, k = (int)num_samples;
    int adj = include_endpoints ? n : (n - 1);
    bool all_srcs = (k == adj) || include_endpoints;
    double scale_src_d, scale_nsrc_d;
    bool uniform;

    if (all_srcs) {
        double scale;
        if (normalized) scale = (double)k * (double)(adj - 1);
        else if (is_symmetric) scale = (double)k * 2.0 / (double)adj;
        else scale = (double)k / (double)adj;
        scale_src_d = scale; scale_nsrc_d = scale; uniform = true;
    } else if (normalized) {
        scale_nsrc_d = (double)k * (double)(adj - 1);
        scale_src_d = (double)(k - 1) * (double)(adj - 1);
        uniform = false;
    } else {
        double s_ns = (double)k / (double)adj, s_s = (double)(k - 1) / (double)adj;
        if (is_symmetric) { s_ns *= 2.0; s_s *= 2.0; }
        scale_nsrc_d = s_ns; scale_src_d = s_s; uniform = false;
    }

    float inv_src = (float)(1.0 / scale_src_d);
    float inv_nsrc = (float)(1.0 / scale_nsrc_d);

    if (!uniform) {
        cudaMemsetAsync(cache.d_is_source, 0, num_vertices * sizeof(bool), cache.streams[0]);
        int blocks = ((int)num_samples + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (blocks > 0)
            mark_sources_kernel<<<blocks, BLOCK_SIZE, 0, cache.streams[0]>>>(
                cache.d_is_source, sample_vertices, (int)num_samples);
    }
    {
        int blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (blocks > 0)
            normalize_kernel<<<blocks, BLOCK_SIZE, 0, cache.streams[0]>>>(
                centralities, cache.d_is_source, num_vertices, inv_src, inv_nsrc, uniform);
    }
    cudaStreamSynchronize(cache.streams[0]);
}

}  
