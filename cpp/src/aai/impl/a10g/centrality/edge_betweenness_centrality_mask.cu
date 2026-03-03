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
#include <vector>

namespace aai {

namespace {

static constexpr int MAX_STREAMS = 5;





__device__ __forceinline__ bool check_bit(const uint32_t* __restrict__ mask, int32_t idx) {
    return (__ldg(&mask[idx >> 5]) >> (idx & 31)) & 1;
}

__global__ void init_arrays_kernel(
    int32_t* __restrict__ d,
    int32_t* __restrict__ sigma,
    float* __restrict__ delta,
    int32_t num_vertices,
    int32_t source,
    int32_t* __restrict__ all_frontiers,
    int32_t* __restrict__ write_pos)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;
    for (int32_t i = tid; i < num_vertices; i += stride) {
        d[i] = (i == source) ? 0 : -1;
        sigma[i] = (i == source) ? 1 : 0;
        delta[i] = 0.0f;
    }
    if (tid == 0) {
        all_frontiers[0] = source;
        *write_pos = 1;
    }
}

__global__ void __launch_bounds__(256, 7) bfs_expand_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ d,
    int32_t* __restrict__ sigma,
    int32_t* __restrict__ all_frontiers,
    int32_t* __restrict__ write_pos,
    uint32_t* __restrict__ dag_mask,
    int32_t frontier_start,
    int32_t frontier_end,
    int32_t current_level)
{
    const int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int32_t lane = threadIdx.x & 31;
    const int32_t frontier_count = frontier_end - frontier_start;
    if (warp_id >= frontier_count) return;

    const int32_t v = all_frontiers[frontier_start + warp_id];
    const int32_t start = __ldg(&offsets[v]);
    const int32_t end = __ldg(&offsets[v + 1]);
    const int32_t sv = sigma[v];
    const int32_t next_level = current_level + 1;

    for (int32_t e = start + lane; e < end; e += 32) {
        if (!check_bit(edge_mask, e)) continue;
        const int32_t w = __ldg(&indices[e]);
        const int32_t old_d = atomicCAS(&d[w], -1, next_level);
        if (old_d != -1 && old_d != next_level) continue;
        atomicAdd(&sigma[w], sv);
        atomicOr(&dag_mask[e >> 5], 1u << (e & 31));
        if (old_d == -1) {
            const int32_t pos = atomicAdd(write_pos, 1);
            all_frontiers[pos] = w;
        }
    }
}

__global__ void __launch_bounds__(256, 7) backward_pass_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ edge_bc,
    const int32_t* __restrict__ all_frontiers,
    const uint32_t* __restrict__ dag_mask,
    int32_t frontier_start,
    int32_t frontier_end)
{
    const int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int32_t lane = threadIdx.x & 31;
    const int32_t frontier_count = frontier_end - frontier_start;
    if (warp_id >= frontier_count) return;

    const int32_t v = all_frontiers[frontier_start + warp_id];
    const int32_t start = __ldg(&offsets[v]);
    const int32_t end = __ldg(&offsets[v + 1]);
    const float sv = static_cast<float>(__ldg(&sigma[v]));
    float my_delta = 0.0f;

    for (int32_t e = start + lane; e < end; e += 32) {
        if (!check_bit(dag_mask, e)) continue;
        const int32_t w = __ldg(&indices[e]);
        const float sw = static_cast<float>(__ldg(&sigma[w]));
        const float delta_w = delta[w];
        const float coeff = __fdividef(sv, sw) * (1.0f + delta_w);
        atomicAdd(&edge_bc[e], coeff);
        my_delta += coeff;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        my_delta += __shfl_down_sync(0xffffffff, my_delta, offset);
    }
    if (lane == 0) {
        delta[v] = my_delta;
    }
}

__global__ void normalize_kernel(float* __restrict__ edge_bc, int32_t num_edges, float scale_factor) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        edge_bc[tid] /= scale_factor;
    }
}





void launch_init_arrays(int32_t* d, int32_t* sigma, float* delta,
                        int32_t num_vertices, int32_t source,
                        int32_t* all_frontiers, int32_t* write_pos, cudaStream_t stream) {
    int blocks = (num_vertices + 511) / 512;
    blocks = blocks > 160 ? 160 : blocks;
    init_arrays_kernel<<<blocks, 512, 0, stream>>>(d, sigma, delta, num_vertices, source, all_frontiers, write_pos);
}

void launch_bfs_expand(const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
                        int32_t* d, int32_t* sigma,
                        int32_t* all_frontiers, int32_t* write_pos, uint32_t* dag_mask,
                        int32_t frontier_start, int32_t frontier_end, int32_t current_level,
                        cudaStream_t stream) {
    int32_t frontier_count = frontier_end - frontier_start;
    if (frontier_count <= 0) return;
    int warps_per_block = 4;
    int blocks = (frontier_count + warps_per_block - 1) / warps_per_block;
    bfs_expand_warp_kernel<<<blocks, warps_per_block * 32, 0, stream>>>(
        offsets, indices, edge_mask, d, sigma,
        all_frontiers, write_pos, dag_mask,
        frontier_start, frontier_end, current_level);
}

void launch_backward_pass(const int32_t* offsets, const int32_t* indices,
                           const int32_t* sigma, float* delta,
                           float* edge_bc, const int32_t* all_frontiers, const uint32_t* dag_mask,
                           int32_t frontier_start, int32_t frontier_end,
                           cudaStream_t stream) {
    int32_t frontier_count = frontier_end - frontier_start;
    if (frontier_count <= 0) return;
    int warps_per_block = 4;
    int blocks = (frontier_count + warps_per_block - 1) / warps_per_block;
    backward_pass_warp_kernel<<<blocks, warps_per_block * 32, 0, stream>>>(
        offsets, indices, sigma, delta, edge_bc,
        all_frontiers, dag_mask, frontier_start, frontier_end);
}

void launch_normalize(float* edge_bc, int32_t num_edges, float scale_factor) {
    if (num_edges <= 0) return;
    int blocks = (num_edges + 255) / 256;
    normalize_kernel<<<blocks, 256>>>(edge_bc, num_edges, scale_factor);
}





struct Cache : Cacheable {
    cudaStream_t streams[MAX_STREAMS] = {};
    int32_t* h_write_pos = nullptr;
    int32_t* h_samples = nullptr;
    int h_samples_capacity = 0;

    
    int32_t* d_d[MAX_STREAMS] = {};
    int32_t* d_sigma[MAX_STREAMS] = {};
    float* d_delta[MAX_STREAMS] = {};
    int32_t* d_frontier[MAX_STREAMS] = {};
    int32_t* d_write_pos_dev[MAX_STREAMS] = {};
    uint32_t* d_dag_mask[MAX_STREAMS] = {};

    int64_t vertex_capacity = 0;
    int64_t dag_mask_capacity = 0;

    Cache() {
        for (int i = 0; i < MAX_STREAMS; i++)
            cudaStreamCreate(&streams[i]);
        cudaHostAlloc(&h_write_pos, MAX_STREAMS * sizeof(int32_t), cudaHostAllocDefault);
        for (int i = 0; i < MAX_STREAMS; i++)
            cudaMalloc(&d_write_pos_dev[i], sizeof(int32_t));
    }

    ~Cache() override {
        for (int i = 0; i < MAX_STREAMS; i++) {
            if (streams[i]) cudaStreamDestroy(streams[i]);
            if (d_d[i]) cudaFree(d_d[i]);
            if (d_sigma[i]) cudaFree(d_sigma[i]);
            if (d_delta[i]) cudaFree(d_delta[i]);
            if (d_frontier[i]) cudaFree(d_frontier[i]);
            if (d_write_pos_dev[i]) cudaFree(d_write_pos_dev[i]);
            if (d_dag_mask[i]) cudaFree(d_dag_mask[i]);
        }
        if (h_write_pos) cudaFreeHost(h_write_pos);
        if (h_samples) cudaFreeHost(h_samples);
    }

    void ensure_vertex_buffers(int64_t nv) {
        if (vertex_capacity >= nv) return;
        for (int i = 0; i < MAX_STREAMS; i++) {
            if (d_d[i]) cudaFree(d_d[i]);
            cudaMalloc(&d_d[i], nv * sizeof(int32_t));
            if (d_sigma[i]) cudaFree(d_sigma[i]);
            cudaMalloc(&d_sigma[i], nv * sizeof(int32_t));
            if (d_delta[i]) cudaFree(d_delta[i]);
            cudaMalloc(&d_delta[i], nv * sizeof(float));
            if (d_frontier[i]) cudaFree(d_frontier[i]);
            cudaMalloc(&d_frontier[i], nv * sizeof(int32_t));
        }
        vertex_capacity = nv;
    }

    void ensure_dag_mask_buffers(int64_t words) {
        if (dag_mask_capacity >= words) return;
        for (int i = 0; i < MAX_STREAMS; i++) {
            if (d_dag_mask[i]) cudaFree(d_dag_mask[i]);
            cudaMalloc(&d_dag_mask[i], words * sizeof(uint32_t));
        }
        dag_mask_capacity = words;
    }

    void ensure_samples(int ns) {
        if (h_samples_capacity >= ns) return;
        if (h_samples) cudaFreeHost(h_samples);
        cudaHostAlloc(&h_samples, ns * sizeof(int32_t), cudaHostAllocDefault);
        h_samples_capacity = ns;
    }
};

}  

void edge_betweenness_centrality_mask(const graph32_t& graph,
                                       float* edge_centralities,
                                       bool normalized,
                                       const int32_t* sample_vertices,
                                       std::size_t num_samples) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t num_vertices = graph.number_of_vertices;
    const int32_t num_edges = graph.number_of_edges;
    const bool is_symmetric = graph.is_symmetric;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    int ns = static_cast<int>(num_samples);
    int num_streams = std::min(ns, MAX_STREAMS);
    int32_t dag_mask_words = (num_edges + 31) / 32;

    cache.ensure_vertex_buffers(num_vertices);
    cache.ensure_dag_mask_buffers(dag_mask_words);
    cache.ensure_samples(ns);

    
    cudaMemset(edge_centralities, 0, num_edges * sizeof(float));

    
    cudaMemcpy(cache.h_samples, sample_vertices, ns * sizeof(int32_t), cudaMemcpyDeviceToHost);

    for (int batch_start = 0; batch_start < ns; batch_start += num_streams) {
        int batch_size = std::min(num_streams, ns - batch_start);

        
        for (int i = 0; i < batch_size; i++) {
            launch_init_arrays(cache.d_d[i], cache.d_sigma[i], cache.d_delta[i], num_vertices,
                               cache.h_samples[batch_start + i], cache.d_frontier[i],
                               cache.d_write_pos_dev[i], cache.streams[i]);
            cudaMemsetAsync(cache.d_dag_mask[i], 0, dag_mask_words * sizeof(uint32_t), cache.streams[i]);
        }

        
        std::vector<std::vector<int32_t>> level_offsets(batch_size);
        std::vector<bool> active(batch_size, true);

        for (int i = 0; i < batch_size; i++) {
            level_offsets[i].reserve(64);
            level_offsets[i].push_back(0);
            level_offsets[i].push_back(1);
        }

        int32_t level = 0;
        bool any_active = true;

        while (any_active) {
            for (int i = 0; i < batch_size; i++) {
                if (!active[i]) continue;
                int32_t frontier_start = level_offsets[i][level];
                int32_t frontier_end = level_offsets[i][level + 1];
                if (frontier_start >= frontier_end) {
                    active[i] = false;
                    continue;
                }
                launch_bfs_expand(d_offsets, d_indices, d_edge_mask,
                                  cache.d_d[i], cache.d_sigma[i], cache.d_frontier[i],
                                  cache.d_write_pos_dev[i], cache.d_dag_mask[i],
                                  frontier_start, frontier_end, level, cache.streams[i]);
            }

            for (int i = 0; i < batch_size; i++) {
                if (!active[i]) continue;
                cudaMemcpyAsync(&cache.h_write_pos[i], cache.d_write_pos_dev[i], sizeof(int32_t),
                                cudaMemcpyDeviceToHost, cache.streams[i]);
            }

            for (int i = 0; i < batch_size; i++) {
                if (!active[i]) continue;
                cudaStreamSynchronize(cache.streams[i]);
            }

            any_active = false;
            for (int i = 0; i < batch_size; i++) {
                if (!active[i]) continue;
                int32_t new_pos = cache.h_write_pos[i];
                level_offsets[i].push_back(new_pos);
                if (new_pos == level_offsets[i][level + 1]) {
                    active[i] = false;
                } else {
                    any_active = true;
                }
            }
            if (!any_active) break;
            for (int i = 0; i < batch_size; i++)
                if (active[i]) { any_active = true; break; }
            level++;
        }

        
        for (int i = 0; i < batch_size; i++) {
            int32_t num_levels = static_cast<int32_t>(level_offsets[i].size()) - 1;
            for (int32_t l = num_levels - 1; l >= 0; l--) {
                launch_backward_pass(d_offsets, d_indices,
                                     cache.d_sigma[i], cache.d_delta[i],
                                     edge_centralities, cache.d_frontier[i], cache.d_dag_mask[i],
                                     level_offsets[i][l], level_offsets[i][l + 1],
                                     cache.streams[i]);
            }
        }

        cudaDeviceSynchronize();
    }

    
    if (num_vertices > 1) {
        float n = static_cast<float>(num_vertices);
        float scale_factor = 0.0f;
        if (normalized) scale_factor = n * (n - 1.0f);
        else if (is_symmetric) scale_factor = 2.0f;

        if (scale_factor > 0.0f) {
            if (num_samples < static_cast<std::size_t>(num_vertices))
                scale_factor *= static_cast<float>(num_samples) / n;
            launch_normalize(edge_centralities, num_edges, scale_factor);
            cudaDeviceSynchronize();
        }
    }
}

}  
