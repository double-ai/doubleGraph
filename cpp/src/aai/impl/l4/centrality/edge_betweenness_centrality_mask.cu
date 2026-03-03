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
#include <cstring>
#include <algorithm>
#include <vector>

namespace aai {

namespace {

static constexpr int MAX_SOURCES = 16;

struct SourceBuffers {
    int32_t* d_distance = nullptr;
    float* d_sigma = nullptr;
    float* d_delta = nullptr;
    int32_t* d_frontier[2] = {nullptr, nullptr};
    int* d_frontier_size = nullptr;
    int32_t* d_queue = nullptr;
};

struct Cache : Cacheable {
    SourceBuffers src_bufs[MAX_SOURCES];
    cudaStream_t streams[MAX_SOURCES] = {};
    int* h_frontier_sizes = nullptr;

    int max_vertices = 0;
    int num_streams = 0;

    Cache() {
        for (int i = 0; i < MAX_SOURCES; i++) {
            cudaStreamCreate(&streams[i]);
        }
        cudaHostAlloc(&h_frontier_sizes, MAX_SOURCES * sizeof(int), cudaHostAllocDefault);
    }

    void ensure_scratch(int num_vertices, int num_sources) {
        if (num_vertices <= max_vertices && num_sources <= num_streams) return;

        for (int s = 0; s < num_streams; s++) {
            auto& b = src_bufs[s];
            if (b.d_distance) cudaFree(b.d_distance);
            if (b.d_sigma) cudaFree(b.d_sigma);
            if (b.d_delta) cudaFree(b.d_delta);
            if (b.d_frontier[0]) cudaFree(b.d_frontier[0]);
            if (b.d_frontier[1]) cudaFree(b.d_frontier[1]);
            if (b.d_frontier_size) cudaFree(b.d_frontier_size);
            if (b.d_queue) cudaFree(b.d_queue);
            b = SourceBuffers();
        }

        max_vertices = num_vertices;
        num_streams = num_sources;
        size_t sz = (size_t)num_vertices;

        for (int s = 0; s < num_sources; s++) {
            auto& b = src_bufs[s];
            cudaMalloc(&b.d_distance, sz * sizeof(int32_t));
            cudaMalloc(&b.d_sigma, sz * sizeof(float));
            cudaMalloc(&b.d_delta, sz * sizeof(float));
            cudaMalloc(&b.d_frontier[0], sz * sizeof(int32_t));
            cudaMalloc(&b.d_frontier[1], sz * sizeof(int32_t));
            cudaMalloc(&b.d_frontier_size, sizeof(int));
            cudaMalloc(&b.d_queue, sz * sizeof(int32_t));
        }
    }

    ~Cache() override {
        for (int s = 0; s < num_streams; s++) {
            auto& b = src_bufs[s];
            if (b.d_distance) cudaFree(b.d_distance);
            if (b.d_sigma) cudaFree(b.d_sigma);
            if (b.d_delta) cudaFree(b.d_delta);
            if (b.d_frontier[0]) cudaFree(b.d_frontier[0]);
            if (b.d_frontier[1]) cudaFree(b.d_frontier[1]);
            if (b.d_frontier_size) cudaFree(b.d_frontier_size);
            if (b.d_queue) cudaFree(b.d_queue);
        }
        for (int i = 0; i < MAX_SOURCES; i++) {
            if (streams[i]) cudaStreamDestroy(streams[i]);
        }
        if (h_frontier_sizes) cudaFreeHost(h_frontier_sizes);
    }
};

__device__ __forceinline__ bool is_edge_active(const uint32_t* edge_mask, int edge_idx) {
    return (edge_mask[edge_idx >> 5] >> (edge_idx & 31)) & 1;
}

__global__ void init_arrays(int32_t* distance, float* sigma, float* delta,
                           int num_vertices, int source) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        distance[tid] = (tid == source) ? 0 : -1;
        sigma[tid] = (tid == source) ? 1.0f : 0.0f;
        delta[tid] = 0.0f;
    }
}

__global__ void bfs_expand_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distance,
    float* __restrict__ sigma,
    const int32_t* __restrict__ frontier,
    int frontier_size,
    int32_t* __restrict__ next_frontier,
    int* __restrict__ next_frontier_size,
    int current_depth
) {
    const int WARP_SIZE = 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int v = frontier[warp_id];
    float sigma_v = sigma[v];
    int start = offsets[v];
    int end = offsets[v + 1];
    int next_depth = current_depth + 1;
    int num_edges = end - start;

    for (int i = lane_id; i < num_edges; i += WARP_SIZE) {
        int e = start + i;
        if (!is_edge_active(edge_mask, e)) continue;

        int w = indices[e];

        int old_dist = atomicCAS(&distance[w], -1, next_depth);

        if (old_dist != -1 && old_dist != next_depth) continue;

        atomicAdd(&sigma[w], sigma_v);

        if (old_dist == -1) {
            int pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = w;
        }
    }
}

__global__ void backward_accumulate_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ distance,
    const float* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ edge_bc,
    const int32_t* __restrict__ level_vertices,
    int num_vertices_at_level,
    int current_depth
) {
    const int WARP_SIZE = 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane_id = threadIdx.x & 31;

    if (warp_id >= num_vertices_at_level) return;

    int v = level_vertices[warp_id];
    float sigma_v = sigma[v];
    int start = offsets[v];
    int end = offsets[v + 1];
    int successor_depth = current_depth + 1;
    int num_edges = end - start;

    float local_delta = 0.0f;

    for (int i = lane_id; i < num_edges; i += WARP_SIZE) {
        int e = start + i;
        if (!is_edge_active(edge_mask, e)) continue;

        int w = indices[e];
        if (distance[w] == successor_depth) {
            float c = (sigma_v / sigma[w]) * (1.0f + delta[w]);
            atomicAdd(&edge_bc[e], c);
            local_delta += c;
        }
    }

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_delta += __shfl_down_sync(0xffffffff, local_delta, offset);
    }

    if (lane_id == 0) {
        delta[v] += local_delta;
    }
}

__global__ void normalize_edge_bc(float* edge_bc, int num_edges, float norm_factor) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        edge_bc[tid] *= norm_factor;
    }
}

void launch_init_arrays(int32_t* distance, float* sigma, float* delta,
                       int num_vertices, int source, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    init_arrays<<<grid, block, 0, stream>>>(distance, sigma, delta, num_vertices, source);
}

void launch_bfs_expand(
    const int32_t* offsets,
    const int32_t* indices,
    const uint32_t* edge_mask,
    int32_t* distance,
    float* sigma,
    const int32_t* frontier,
    int frontier_size,
    int32_t* next_frontier,
    int* next_frontier_size,
    int current_depth,
    cudaStream_t stream
) {
    if (frontier_size == 0) return;
    int block = 256;
    int grid = (int)(((int64_t)frontier_size * 32 + block - 1) / block);
    bfs_expand_warp<<<grid, block, 0, stream>>>(offsets, indices, edge_mask, distance, sigma,
                                                  frontier, frontier_size, next_frontier,
                                                  next_frontier_size, current_depth);
}

void launch_backward_accumulate(
    const int32_t* offsets,
    const int32_t* indices,
    const uint32_t* edge_mask,
    const int32_t* distance,
    const float* sigma,
    float* delta,
    float* edge_bc,
    const int32_t* level_vertices,
    int num_vertices_at_level,
    int current_depth,
    cudaStream_t stream
) {
    if (num_vertices_at_level == 0) return;
    int block = 256;
    int grid = (int)(((int64_t)num_vertices_at_level * 32 + block - 1) / block);
    backward_accumulate_warp<<<grid, block, 0, stream>>>(offsets, indices, edge_mask, distance,
                                                          sigma, delta, edge_bc, level_vertices,
                                                          num_vertices_at_level, current_depth);
}

void launch_normalize_edge_bc(float* edge_bc, int num_edges, float norm_factor, cudaStream_t stream) {
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    normalize_edge_bc<<<grid, block, 0, stream>>>(edge_bc, num_edges, norm_factor);
}

}  

void edge_betweenness_centrality_mask(const graph32_t& graph,
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

    
    std::vector<int32_t> sources;
    if (num_samples > 0 && sample_vertices != nullptr) {
        sources.resize(num_samples);
        cudaMemcpy(sources.data(), sample_vertices,
                   num_samples * sizeof(int32_t), cudaMemcpyDeviceToHost);
    } else {
        sources.resize(num_vertices);
        for (int i = 0; i < num_vertices; i++) sources[i] = i;
    }

    int total_sources = sources.size();
    int concurrent = std::min(total_sources, MAX_SOURCES);

    cache.ensure_scratch(num_vertices, concurrent);

    
    float* d_edge_bc = edge_centralities;
    cudaMemsetAsync(d_edge_bc, 0, (size_t)num_edges * sizeof(float), 0);
    cudaDeviceSynchronize();

    
    for (int batch_start = 0; batch_start < total_sources; batch_start += concurrent) {
        int batch_size = std::min(concurrent, total_sources - batch_start);

        
        for (int s = 0; s < batch_size; s++) {
            int source = sources[batch_start + s];
            auto& b = cache.src_bufs[s];
            launch_init_arrays(b.d_distance, b.d_sigma, b.d_delta,
                              num_vertices, source, cache.streams[s]);
            cudaMemcpyAsync(b.d_frontier[0], &sources[batch_start + s],
                           sizeof(int32_t), cudaMemcpyHostToDevice, cache.streams[s]);
            cudaMemcpyAsync(b.d_queue, &sources[batch_start + s],
                           sizeof(int32_t), cudaMemcpyHostToDevice, cache.streams[s]);
        }

        
        std::vector<int> frontier_size(batch_size, 1);
        std::vector<int> current_buf(batch_size, 0);
        std::vector<std::vector<int>> level_offsets(batch_size);
        for (int s = 0; s < batch_size; s++) {
            level_offsets[s].push_back(0);
            level_offsets[s].push_back(1);
        }

        int active_sources = batch_size;
        int depth = 0;

        while (active_sources > 0) {
            
            for (int s = 0; s < batch_size; s++) {
                if (frontier_size[s] <= 0) continue;
                auto& b = cache.src_bufs[s];
                cudaMemsetAsync(b.d_frontier_size, 0, sizeof(int), cache.streams[s]);
                launch_bfs_expand(d_offsets, d_indices, d_edge_mask,
                                 b.d_distance, b.d_sigma,
                                 b.d_frontier[current_buf[s]], frontier_size[s],
                                 b.d_frontier[1 - current_buf[s]], b.d_frontier_size,
                                 depth, cache.streams[s]);
                cudaMemcpyAsync(&cache.h_frontier_sizes[s], b.d_frontier_size,
                               sizeof(int), cudaMemcpyDeviceToHost, cache.streams[s]);
            }

            
            active_sources = 0;
            for (int s = 0; s < batch_size; s++) {
                if (frontier_size[s] <= 0) continue;
                cudaStreamSynchronize(cache.streams[s]);
                int next_size = cache.h_frontier_sizes[s];

                if (next_size > 0) {
                    auto& b = cache.src_bufs[s];
                    cudaMemcpyAsync(b.d_queue + level_offsets[s].back(),
                                   b.d_frontier[1 - current_buf[s]],
                                   (size_t)next_size * sizeof(int32_t),
                                   cudaMemcpyDeviceToDevice, cache.streams[s]);
                    level_offsets[s].push_back(level_offsets[s].back() + next_size);
                }

                frontier_size[s] = next_size;
                current_buf[s] = 1 - current_buf[s];
                if (next_size > 0) active_sources++;
            }
            depth++;
        }

        
        for (int s = 0; s < batch_size; s++) {
            int num_levels = (int)level_offsets[s].size() - 1;
            auto& b = cache.src_bufs[s];
            for (int d = num_levels - 1; d >= 0; d--) {
                int lstart = level_offsets[s][d];
                int lend = level_offsets[s][d + 1];
                int lsize = lend - lstart;
                if (lsize > 0) {
                    launch_backward_accumulate(d_offsets, d_indices, d_edge_mask,
                                             b.d_distance, b.d_sigma, b.d_delta,
                                             d_edge_bc,
                                             b.d_queue + lstart, lsize,
                                             d, cache.streams[s]);
                }
            }
        }
    }

    
    {
        double n = (double)num_vertices;
        int actual_sources = (num_samples > 0) ? (int)num_samples : (int)num_vertices;
        double factor = 1.0;
        if (normalized) {
            factor = 1.0 / ((double)actual_sources * (n - 1.0));
        } else if (is_symmetric) {
            factor = n / (2.0 * (double)actual_sources);
        }
        if (factor != 1.0) {
            launch_normalize_edge_bc(d_edge_bc, num_edges, (float)factor, 0);
        }
    }

    cudaDeviceSynchronize();
}

}  
