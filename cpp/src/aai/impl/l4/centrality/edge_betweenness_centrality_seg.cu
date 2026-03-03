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
#include <cstddef>
#include <vector>

namespace aai {

namespace {

static constexpr int MAX_CONCURRENT = 2;



__global__ void __launch_bounds__(256)
init_arrays_kernel(
    int32_t* __restrict__ distances,
    float* __restrict__ sigma,
    float* __restrict__ delta,
    int32_t num_vertices,
    int32_t source
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices;
         idx += gridDim.x * blockDim.x) {
        distances[idx] = (idx == source) ? 0 : -1;
        sigma[idx] = (idx == source) ? 1.0f : 0.0f;
        delta[idx] = 0.0f;
    }
}

__global__ void __launch_bounds__(256)
bfs_expand_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_count,
    int32_t* __restrict__ distances,
    float* __restrict__ sigma,
    int32_t current_level
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int total_warps = (gridDim.x * blockDim.x) >> 5;
    const int32_t next_level = current_level + 1;

    for (int idx = warp_id; idx < frontier_size; idx += total_warps) {
        int32_t u = frontier[idx];
        float sigma_u = sigma[u];
        int32_t start = __ldg(&offsets[u]);
        int32_t end = __ldg(&offsets[u + 1]);

        for (int32_t e = start + lane; e < end; e += 32) {
            int32_t v = __ldg(&indices[e]);
            int32_t old_d = atomicCAS(&distances[v], -1, next_level);

            if (old_d != -1 && old_d != next_level) continue;

            atomicAdd(&sigma[v], sigma_u);

            if (old_d == -1) {
                int32_t pos = atomicAdd(next_frontier_count, 1);
                next_frontier[pos] = v;
            }
        }
    }
}

__global__ void __launch_bounds__(256)
backprop_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    const int32_t* __restrict__ distances,
    const float* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ edge_bc,
    int32_t current_level
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int total_warps = (gridDim.x * blockDim.x) >> 5;
    const int32_t child_level = current_level + 1;

    for (int idx = warp_id; idx < frontier_size; idx += total_warps) {
        int32_t v = frontier[idx];
        float sigma_v = sigma[v];
        float delta_v = 0.0f;
        int32_t start = __ldg(&offsets[v]);
        int32_t end = __ldg(&offsets[v + 1]);

        for (int32_t e = start + lane; e < end; e += 32) {
            int32_t w = __ldg(&indices[e]);
            if (__ldg(&distances[w]) == child_level) {
                float c = sigma_v / __ldg(&sigma[w]) * (1.0f + __ldg(&delta[w]));
                delta_v += c;
                edge_bc[e] += c;
            }
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            delta_v += __shfl_down_sync(0xffffffff, delta_v, offset);
        }

        if (lane == 0) {
            delta[v] = delta_v;
        }
    }
}

__global__ void normalize_kernel(
    float* __restrict__ edge_bc,
    int32_t num_edges,
    float norm_factor
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges;
         idx += gridDim.x * blockDim.x) {
        edge_bc[idx] *= norm_factor;
    }
}



void launch_init_arrays(int32_t* distances, float* sigma, float* delta,
                        int32_t num_vertices, int32_t source, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    if (grid > 928) grid = 928;
    init_arrays_kernel<<<grid, block, 0, stream>>>(distances, sigma, delta, num_vertices, source);
}

void launch_bfs_expand(const int32_t* offsets, const int32_t* indices,
                       const int32_t* frontier, int32_t frontier_size,
                       int32_t* next_frontier, int32_t* next_frontier_count,
                       int32_t* distances, float* sigma, int32_t current_level,
                       cudaStream_t stream) {
    if (frontier_size <= 0) return;
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks = (frontier_size + warps_per_block - 1) / warps_per_block;
    if (blocks > 928) blocks = 928;

    bfs_expand_kernel<<<blocks, threads_per_block, 0, stream>>>(
        offsets, indices, frontier, frontier_size,
        next_frontier, next_frontier_count, distances, sigma, current_level);
}

void launch_backprop(const int32_t* offsets, const int32_t* indices,
                     const int32_t* frontier, int32_t frontier_size,
                     const int32_t* distances, const float* sigma,
                     float* delta, float* edge_bc, int32_t current_level,
                     cudaStream_t stream) {
    if (frontier_size <= 0) return;
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int blocks = (frontier_size + warps_per_block - 1) / warps_per_block;
    if (blocks > 928) blocks = 928;

    backprop_kernel<<<blocks, threads_per_block, 0, stream>>>(
        offsets, indices, frontier, frontier_size,
        distances, sigma, delta, edge_bc, current_level);
}

void launch_normalize(float* edge_bc, int32_t num_edges, float norm_factor,
                      cudaStream_t stream) {
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    normalize_kernel<<<grid, block, 0, stream>>>(edge_bc, num_edges, norm_factor);
}



struct PerSourceScratch {
    int32_t* d_distances = nullptr;
    float* d_sigma = nullptr;
    float* d_delta = nullptr;
    int32_t* d_frontier_storage = nullptr;
    int32_t* d_frontier_count = nullptr;
    cudaStream_t stream = 0;
    cudaEvent_t event = 0;
};

struct Cache : Cacheable {
    PerSourceScratch scratch[MAX_CONCURRENT];
    int32_t alloc_vertices = 0;

    void free_all() {
        for (int i = 0; i < MAX_CONCURRENT; i++) {
            auto& s = scratch[i];
            if (s.d_distances) { cudaFree(s.d_distances); s.d_distances = nullptr; }
            if (s.d_sigma) { cudaFree(s.d_sigma); s.d_sigma = nullptr; }
            if (s.d_delta) { cudaFree(s.d_delta); s.d_delta = nullptr; }
            if (s.d_frontier_storage) { cudaFree(s.d_frontier_storage); s.d_frontier_storage = nullptr; }
            if (s.d_frontier_count) { cudaFree(s.d_frontier_count); s.d_frontier_count = nullptr; }
            if (s.stream) { cudaStreamDestroy(s.stream); s.stream = 0; }
            if (s.event) { cudaEventDestroy(s.event); s.event = 0; }
        }
        alloc_vertices = 0;
    }

    void ensure_scratch(int32_t num_vertices) {
        if (num_vertices <= alloc_vertices) return;
        free_all();
        for (int i = 0; i < MAX_CONCURRENT; i++) {
            auto& s = scratch[i];
            cudaMalloc(&s.d_distances, (size_t)num_vertices * sizeof(int32_t));
            cudaMalloc(&s.d_sigma, (size_t)num_vertices * sizeof(float));
            cudaMalloc(&s.d_delta, (size_t)num_vertices * sizeof(float));
            cudaMalloc(&s.d_frontier_storage, (size_t)num_vertices * sizeof(int32_t));
            cudaMalloc(&s.d_frontier_count, sizeof(int32_t));
            cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&s.event, cudaEventDisableTiming);
        }
        alloc_vertices = num_vertices;
    }

    ~Cache() override {
        free_all();
    }
};

void process_source(Cache& cache, int scratch_idx, int32_t source, int32_t num_vertices,
                    const int32_t* d_offsets, const int32_t* d_indices,
                    float* d_edge_bc) {
    auto& s = cache.scratch[scratch_idx];
    cudaStream_t stream = s.stream;

    launch_init_arrays(s.d_distances, s.d_sigma, s.d_delta, num_vertices, source, stream);

    std::vector<int32_t> h_level_offsets;
    h_level_offsets.reserve(64);
    h_level_offsets.push_back(0);

    cudaMemcpyAsync(s.d_frontier_storage, &source, sizeof(int32_t),
                   cudaMemcpyHostToDevice, stream);
    h_level_offsets.push_back(1);

    int32_t level = 0;
    int32_t frontier_size = 1;

    while (frontier_size > 0) {
        int32_t frontier_start = h_level_offsets[level];
        int32_t next_start = h_level_offsets[level + 1];

        cudaMemsetAsync(s.d_frontier_count, 0, sizeof(int32_t), stream);

        launch_bfs_expand(d_offsets, d_indices,
                         s.d_frontier_storage + frontier_start, frontier_size,
                         s.d_frontier_storage + next_start, s.d_frontier_count,
                         s.d_distances, s.d_sigma, level, stream);

        int32_t next_size;
        cudaMemcpyAsync(&next_size, s.d_frontier_count, sizeof(int32_t),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        level++;
        h_level_offsets.push_back(next_start + next_size);
        frontier_size = next_size;
    }

    for (int32_t L = level - 2; L >= 0; L--) {
        int32_t frontier_start = h_level_offsets[L];
        int32_t fs = h_level_offsets[L + 1] - frontier_start;

        if (fs > 0) {
            launch_backprop(d_offsets, d_indices,
                          s.d_frontier_storage + frontier_start, fs,
                          s.d_distances, s.d_sigma, s.d_delta, d_edge_bc, L, stream);
        }
    }
}

}  

void edge_betweenness_centrality_seg(const graph32_t& graph,
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

    cudaMemset(edge_centralities, 0, (size_t)num_edges * sizeof(float));

    cache.ensure_scratch(num_vertices);

    std::vector<int32_t> h_sources;
    if (num_samples > 0 && sample_vertices != nullptr) {
        h_sources.resize(num_samples);
        cudaMemcpy(h_sources.data(), sample_vertices,
                   num_samples * sizeof(int32_t), cudaMemcpyDeviceToHost);
    } else {
        h_sources.resize(num_vertices);
        for (int32_t i = 0; i < num_vertices; i++) h_sources[i] = i;
    }

    size_t actual_num_sources = h_sources.size();

    for (size_t s = 0; s < h_sources.size(); s++) {
        process_source(cache, 0, h_sources[s], num_vertices, d_offsets, d_indices, edge_centralities);
    }

    for (int i = 0; i < MAX_CONCURRENT; i++) {
        cudaStreamSynchronize(cache.scratch[i].stream);
    }

    if (num_vertices > 1) {
        float scale_factor = 0.0f;
        bool has_scale = false;

        if (normalized) {
            float n = static_cast<float>(num_vertices);
            scale_factor = n * (n - 1.0f);
            has_scale = true;
        } else if (is_symmetric) {
            scale_factor = 2.0f;
            has_scale = true;
        }

        if (has_scale) {
            if (static_cast<int32_t>(actual_num_sources) < num_vertices) {
                scale_factor *= static_cast<float>(actual_num_sources) /
                                static_cast<float>(num_vertices);
            }
            float norm_factor = 1.0f / scale_factor;
            launch_normalize(edge_centralities, num_edges, norm_factor, cache.scratch[0].stream);
            cudaStreamSynchronize(cache.scratch[0].stream);
        }
    }
}

}  
