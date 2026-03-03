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

static constexpr int MAX_STREAMS = 10;





__global__ void init_bfs_kernel(
    int32_t* __restrict__ dist,
    float* __restrict__ sigma,
    float* __restrict__ delta,
    int32_t* __restrict__ frontier,
    int32_t num_vertices,
    int32_t source
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        dist[idx] = (idx == source) ? 0 : -1;
        sigma[idx] = (idx == source) ? 1.0f : 0.0f;
        delta[idx] = 0.0f;
    }
    if (idx == 0) frontier[0] = source;
}


__global__ void bfs_expand_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ dist,
    float* __restrict__ sigma,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count,
    int32_t current_dist
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    const int32_t v = frontier[warp_id];
    const int32_t start = offsets[v];
    const int32_t end = offsets[v + 1];
    const float sigma_v = sigma[v];
    const int32_t next_dist = current_dist + 1;

    for (int32_t e = start + lane; e < end; e += 32) {
        const uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
        if (!((mask_word >> (e & 31)) & 1)) continue;

        const int32_t w = __ldg(&indices[e]);

        
        int32_t w_dist = dist[w];
        if (w_dist >= 0 && w_dist != next_dist) continue;  

        if (w_dist == -1) {
            
            int32_t old = atomicCAS(&dist[w], -1, next_dist);
            if (old == -1) {
                
                int32_t pos = atomicAdd(next_count, 1);
                next_frontier[pos] = w;
            } else if (old != next_dist) {
                continue;  
            }
        }
        
        atomicAdd(&sigma[w], sigma_v);
    }
}


__global__ void backward_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ dist,
    const float* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ edge_bc,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t current_dist
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    const int32_t v = frontier[warp_id];
    const int32_t start = offsets[v];
    const int32_t end = offsets[v + 1];
    const float sigma_v = sigma[v];
    const int32_t child_dist = current_dist + 1;

    float local_delta = 0.0f;

    for (int32_t e = start + lane; e < end; e += 32) {
        const uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
        if (!((mask_word >> (e & 31)) & 1)) continue;

        const int32_t w = __ldg(&indices[e]);
        const int32_t w_dist = dist[w];
        if (w_dist == child_dist) {
            const float sigma_w = sigma[w];
            const float delta_w = delta[w];
            const float coeff = (sigma_v / sigma_w) * (1.0f + delta_w);
            atomicAdd(&edge_bc[e], coeff);
            local_delta += coeff;
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_delta += __shfl_down_sync(0xffffffff, local_delta, off);
    }
    if (lane == 0) {
        delta[v] = local_delta;
    }
}

__global__ void scale_kernel(float* __restrict__ data, int32_t n, float inv_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= inv_scale;
    }
}





void launch_init_bfs(int32_t* dist, float* sigma, float* delta, int32_t* frontier,
                     int32_t num_vertices, int32_t source, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_vertices + threads - 1) / threads;
    init_bfs_kernel<<<blocks, threads, 0, stream>>>(dist, sigma, delta, frontier, num_vertices, source);
}

void launch_bfs_expand(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    int32_t* dist, float* sigma,
    const int32_t* frontier, int32_t frontier_size,
    int32_t* next_frontier, int32_t* next_count,
    int32_t current_dist, cudaStream_t stream)
{
    if (frontier_size == 0) return;
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (frontier_size + warps_per_block - 1) / warps_per_block;
    bfs_expand_kernel<<<blocks, threads, 0, stream>>>(
        offsets, indices, edge_mask, dist, sigma,
        frontier, frontier_size, next_frontier, next_count, current_dist);
}

void launch_backward(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    const int32_t* dist, const float* sigma, float* delta, float* edge_bc,
    const int32_t* frontier, int32_t frontier_size,
    int32_t current_dist, cudaStream_t stream)
{
    if (frontier_size == 0) return;
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (frontier_size + warps_per_block - 1) / warps_per_block;
    backward_kernel<<<blocks, threads, 0, stream>>>(
        offsets, indices, edge_mask, dist, sigma, delta, edge_bc,
        frontier, frontier_size, current_dist);
}

void launch_scale(float* data, int32_t n, float inv_scale, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scale_kernel<<<blocks, threads, 0, stream>>>(data, n, inv_scale);
}





struct Cache : Cacheable {
    cudaStream_t streams[MAX_STREAMS] = {};
    int32_t* h_counters = nullptr;

    int32_t* ints_buf = nullptr;
    float* floats_buf = nullptr;
    int64_t ints_capacity = 0;
    int64_t floats_capacity = 0;

    Cache() {
        for (int i = 0; i < MAX_STREAMS; i++) {
            cudaStreamCreate(&streams[i]);
        }
        cudaHostAlloc(&h_counters, MAX_STREAMS * sizeof(int32_t), cudaHostAllocDefault);
    }

    void ensure(int64_t ints_needed, int64_t floats_needed) {
        if (ints_capacity < ints_needed) {
            if (ints_buf) cudaFree(ints_buf);
            cudaMalloc(&ints_buf, ints_needed * sizeof(int32_t));
            ints_capacity = ints_needed;
        }
        if (floats_capacity < floats_needed) {
            if (floats_buf) cudaFree(floats_buf);
            cudaMalloc(&floats_buf, floats_needed * sizeof(float));
            floats_capacity = floats_needed;
        }
    }

    ~Cache() override {
        for (int i = 0; i < MAX_STREAMS; i++) {
            if (streams[i]) cudaStreamDestroy(streams[i]);
        }
        if (h_counters) cudaFreeHost(h_counters);
        if (ints_buf) cudaFree(ints_buf);
        if (floats_buf) cudaFree(floats_buf);
    }
};

}  

void edge_betweenness_centrality_seg_mask(const graph32_t& graph,
                                           float* edge_centralities,
                                           bool normalized,
                                           const int32_t* sample_vertices,
                                           std::size_t num_samples) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    bool is_symmetric = graph.is_symmetric;
    const uint32_t* d_edge_mask = graph.edge_mask;

    
    cudaMemsetAsync(edge_centralities, 0, num_edges * sizeof(float), 0);

    if (num_vertices <= 1 || num_edges == 0 || num_samples == 0) {
        cudaDeviceSynchronize();
        return;
    }

    int ns = (int)std::min(num_samples, (std::size_t)MAX_STREAMS);

    
    int64_t per_source_ints = (int64_t)num_vertices * 3 + 1;  
    int64_t per_source_floats = (int64_t)num_vertices * 2;    
    cache.ensure(ns * per_source_ints, ns * per_source_floats);

    
    std::vector<int32_t*> d_dist(ns);
    std::vector<float*> d_sigma(ns);
    std::vector<float*> d_delta(ns);
    std::vector<int32_t*> d_frontier(ns);
    std::vector<int32_t*> d_counter(ns);

    int32_t* ints_ptr = cache.ints_buf;
    float* floats_ptr = cache.floats_buf;

    for (int i = 0; i < ns; i++) {
        d_dist[i] = ints_ptr + i * per_source_ints;
        d_frontier[i] = d_dist[i] + num_vertices;
        d_counter[i] = d_frontier[i] + num_vertices;
        d_sigma[i] = floats_ptr + i * per_source_floats;
        d_delta[i] = d_sigma[i] + num_vertices;
    }

    
    std::vector<int32_t> h_samples(num_samples);
    cudaMemcpy(h_samples.data(), sample_vertices, num_samples * sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    cudaDeviceSynchronize();

    
    for (std::size_t batch_start = 0; batch_start < num_samples; batch_start += ns) {
        int batch_size = (int)std::min((std::size_t)ns, num_samples - batch_start);

        
        for (int si = 0; si < batch_size; si++) {
            int32_t source = h_samples[batch_start + si];
            launch_init_bfs(d_dist[si], d_sigma[si], d_delta[si], d_frontier[si],
                           num_vertices, source, cache.streams[si]);
        }

        
        struct SourceState {
            std::vector<int32_t> level_offsets;
            int32_t total;
            int32_t current_dist;
            bool active;
        };
        std::vector<SourceState> states(batch_size);
        for (int si = 0; si < batch_size; si++) {
            states[si].level_offsets = {0, 1};
            states[si].total = 1;
            states[si].current_dist = 0;
            states[si].active = true;
        }

        while (true) {
            bool any_active = false;

            
            for (int si = 0; si < batch_size; si++) {
                if (!states[si].active) continue;
                any_active = true;

                auto& s = states[si];
                int32_t frontier_offset = s.level_offsets[s.current_dist];
                int32_t frontier_size = s.level_offsets[s.current_dist + 1] - frontier_offset;

                cudaMemsetAsync(d_counter[si], 0, sizeof(int32_t), cache.streams[si]);

                launch_bfs_expand(d_offsets, d_indices, d_edge_mask,
                    d_dist[si], d_sigma[si],
                    d_frontier[si] + frontier_offset, frontier_size,
                    d_frontier[si] + s.total, d_counter[si],
                    s.current_dist, cache.streams[si]);

                
                cudaMemcpyAsync(&cache.h_counters[si], d_counter[si], sizeof(int32_t),
                               cudaMemcpyDeviceToHost, cache.streams[si]);
            }

            if (!any_active) break;

            
            for (int si = 0; si < batch_size; si++) {
                if (!states[si].active) continue;
                cudaStreamSynchronize(cache.streams[si]);

                auto& s = states[si];
                int32_t next_size = cache.h_counters[si];
                s.total += next_size;
                s.level_offsets.push_back(s.total);
                s.current_dist++;

                if (next_size == 0) s.active = false;
            }
        }

        
        
        for (int si = 0; si < batch_size; si++) {
            auto& s = states[si];
            int num_levels = (int)s.level_offsets.size() - 1;
            int max_level = num_levels - 1;
            while (max_level >= 0 && s.level_offsets[max_level + 1] == s.level_offsets[max_level])
                max_level--;

            for (int d = max_level - 1; d >= 0; d--) {
                int32_t offset = s.level_offsets[d];
                int32_t size = s.level_offsets[d + 1] - offset;
                if (size > 0) {
                    launch_backward(d_offsets, d_indices, d_edge_mask,
                        d_dist[si], d_sigma[si], d_delta[si], edge_centralities,
                        d_frontier[si] + offset, size,
                        d, cache.streams[si]);
                }
            }
        }

        
        for (int si = 0; si < batch_size; si++) {
            cudaStreamSynchronize(cache.streams[si]);
        }
    }

    
    bool need_scale = false;
    float scale_factor = 1.0f;

    if (normalized) {
        float n = static_cast<float>(num_vertices);
        scale_factor = n * (n - 1.0f);
        need_scale = true;
    } else if (is_symmetric) {
        scale_factor = 2.0f;
        need_scale = true;
    }

    if (need_scale && num_vertices > 1) {
        if (num_samples < static_cast<std::size_t>(num_vertices)) {
            scale_factor *= static_cast<float>(num_samples) / static_cast<float>(num_vertices);
        }
        float inv_scale = 1.0f / scale_factor;
        launch_scale(edge_centralities, num_edges, inv_scale, 0);
    }

    cudaDeviceSynchronize();
}

}  
