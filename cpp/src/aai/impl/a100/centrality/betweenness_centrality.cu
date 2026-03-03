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

struct Cache : Cacheable {
    int32_t* d_distances = nullptr;
    float* d_sigma = nullptr;
    float* d_delta = nullptr;
    int32_t* d_next_frontier_size = nullptr;
    
    
    int32_t* d_all_frontiers = nullptr;
    uint8_t* d_is_source = nullptr;
    
    int32_t* h_next_size = nullptr;
    int alloc_vertices = 0;

    void ensure(int num_vertices) {
        if (num_vertices <= alloc_vertices) return;
        free_buffers();
        alloc_vertices = num_vertices;

        cudaMalloc(&d_distances, (size_t)num_vertices * sizeof(int32_t));
        cudaMalloc(&d_sigma,     (size_t)num_vertices * sizeof(float));
        cudaMalloc(&d_delta,     (size_t)num_vertices * sizeof(float));
        cudaMalloc(&d_next_frontier_size, sizeof(int32_t));
        cudaMalloc(&d_all_frontiers, (size_t)num_vertices * sizeof(int32_t));
        cudaMalloc(&d_is_source, (size_t)num_vertices * sizeof(uint8_t));
        cudaHostAlloc(&h_next_size, sizeof(int32_t), cudaHostAllocDefault);
    }

    void free_buffers() {
        auto free_and_null = [](auto*& ptr) {
            if (ptr) { cudaFree(ptr); ptr = nullptr; }
        };
        free_and_null(d_distances);
        free_and_null(d_sigma);
        free_and_null(d_delta);
        free_and_null(d_next_frontier_size);
        free_and_null(d_all_frontiers);
        free_and_null(d_is_source);
        if (h_next_size) { cudaFreeHost(h_next_size); h_next_size = nullptr; }
        alloc_vertices = 0;
    }

    ~Cache() override {
        free_buffers();
    }
};


__global__ void init_kernel(
    int32_t* __restrict__ distances,
    float* __restrict__ sigma,
    float* __restrict__ delta,
    int32_t* __restrict__ frontier,
    int num_vertices,
    int source
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        bool is_src = (tid == source);
        distances[tid] = is_src ? 0 : -1;
        sigma[tid] = is_src ? 1.0f : 0.0f;
        delta[tid] = 0.0f;
    }
    
    if (tid == 0) {
        frontier[0] = source;
    }
}


__global__ void bfs_forward_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    float* __restrict__ sigma,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int frontier_size,
    int current_level
) {
    const int WARP_SIZE = 32;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / WARP_SIZE;
    int lane_id = global_tid & (WARP_SIZE - 1);

    if (warp_id >= frontier_size) return;

    int src = frontier[warp_id];
    int start = __ldg(&offsets[src]);
    int end = __ldg(&offsets[src + 1]);
    int new_level = current_level + 1;
    float src_sigma = sigma[src];

    
    for (int e = start + lane_id; e < end; e += WARP_SIZE) {
        int dst = indices[e];

        
        
        int d = distances[dst];
        if (d >= 0) {
            
            if (d == new_level) {
                atomicAdd(&sigma[dst], src_sigma);
            }
            continue;
        }

        
        int old_dist = atomicCAS(&distances[dst], -1, new_level);

        if (old_dist == -1) {
            
            atomicAdd(&sigma[dst], src_sigma);
            int pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = dst;
        } else if (old_dist == new_level) {
            
            atomicAdd(&sigma[dst], src_sigma);
        }
        
    }
}


__global__ void dependency_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ distances,
    const float* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ centralities,
    const int32_t* __restrict__ frontier,
    int frontier_size,
    int current_level,
    int source
) {
    const int WARP_SIZE = 32;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / WARP_SIZE;
    int lane_id = global_tid & (WARP_SIZE - 1);

    if (warp_id >= frontier_size) return;

    int v = frontier[warp_id];
    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);
    float sigma_v = sigma[v];
    int next_level = current_level + 1;

    
    float local_accum = 0.0f;
    for (int e = start + lane_id; e < end; e += WARP_SIZE) {
        int w = indices[e];
        if (__ldg(&distances[w]) == next_level) {
            local_accum += (1.0f + delta[w]) / sigma[w];
        }
    }

    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_accum += __shfl_down_sync(0xffffffff, local_accum, offset);
    }

    
    if (lane_id == 0) {
        float result = local_accum * sigma_v;
        delta[v] = result;
        if (v != source) {
            centralities[v] += result;
        }
    }
}


__global__ void endpoints_kernel(
    float* __restrict__ centralities,
    const int32_t* __restrict__ distances,
    int num_vertices,
    int source,
    int reachable_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;

    if (tid == source) {
        centralities[tid] += (float)reachable_count;
    } else if (distances[tid] >= 0) {
        centralities[tid] += 1.0f;
    }
}


__global__ void mark_sources_kernel(
    uint8_t* __restrict__ is_source,
    const int32_t* __restrict__ sample_vertices,
    int64_t num_samples
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_samples) {
        is_source[sample_vertices[tid]] = 1;
    }
}

__global__ void normalize_uniform_kernel(
    float* __restrict__ centralities,
    int num_vertices,
    float inv_scale
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        centralities[tid] *= inv_scale;
    }
}

__global__ void normalize_split_kernel(
    float* __restrict__ centralities,
    const uint8_t* __restrict__ is_source,
    int num_vertices,
    float inv_scale_source,
    float inv_scale_non_source
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        float inv_scale = is_source[tid] ? inv_scale_source : inv_scale_non_source;
        centralities[tid] *= inv_scale;
    }
}



static void launch_init(int32_t* distances, float* sigma, float* delta,
                        int32_t* frontier, int num_vertices, int source) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    init_kernel<<<grid, block>>>(distances, sigma, delta, frontier, num_vertices, source);
}

static void launch_bfs_forward(
    const int32_t* offsets, const int32_t* indices,
    int32_t* distances, float* sigma,
    const int32_t* frontier, int32_t* next_frontier,
    int32_t* next_frontier_size,
    int frontier_size, int current_level
) {
    if (frontier_size <= 0) return;
    
    int block = 256;  
    int warps_per_block = block / 32;
    int grid = (frontier_size + warps_per_block - 1) / warps_per_block;
    bfs_forward_kernel<<<grid, block>>>(
        offsets, indices, distances, sigma,
        frontier, next_frontier, next_frontier_size,
        frontier_size, current_level);
}

static void launch_dependency(
    const int32_t* offsets, const int32_t* indices,
    const int32_t* distances, const float* sigma,
    float* delta, float* centralities,
    const int32_t* frontier, int frontier_size,
    int current_level, int source
) {
    if (frontier_size <= 0) return;
    int block = 256;  
    int warps_per_block = block / 32;
    int grid = (frontier_size + warps_per_block - 1) / warps_per_block;
    dependency_kernel<<<grid, block>>>(
        offsets, indices, distances, sigma,
        delta, centralities, frontier, frontier_size,
        current_level, source);
}

static void launch_endpoints(
    float* centralities, const int32_t* distances,
    int num_vertices, int source, int reachable_count
) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    endpoints_kernel<<<grid, block>>>(
        centralities, distances, num_vertices, source, reachable_count);
}

static void launch_mark_sources(
    uint8_t* is_source, const int32_t* sample_vertices, int64_t num_samples
) {
    int block = 256;
    int grid = ((int)num_samples + block - 1) / block;
    mark_sources_kernel<<<grid, block>>>(is_source, sample_vertices, num_samples);
}

static void launch_normalize_uniform(
    float* centralities, int num_vertices, float inv_scale
) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    normalize_uniform_kernel<<<grid, block>>>(centralities, num_vertices, inv_scale);
}

static void launch_normalize_split(
    float* centralities, const uint8_t* is_source,
    int num_vertices, float inv_scale_source, float inv_scale_non_source
) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    normalize_split_kernel<<<grid, block>>>(
        centralities, is_source, num_vertices,
        inv_scale_source, inv_scale_non_source);
}

}  

void betweenness_centrality(const graph32_t& graph,
                            float* centralities,
                            bool normalized,
                            bool include_endpoints,
                            const int32_t* sample_vertices,
                            std::size_t num_samples) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    bool is_symmetric = graph.is_symmetric;

    cache.ensure(num_vertices);

    cudaMemset(centralities, 0, (size_t)num_vertices * sizeof(float));

    
    std::vector<int32_t> h_samples(num_samples);
    cudaMemcpy(h_samples.data(), sample_vertices,
               num_samples * sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    for (std::size_t s = 0; s < num_samples; s++) {
        int32_t source = h_samples[s];

        
        
        launch_init(cache.d_distances, cache.d_sigma, cache.d_delta,
                    cache.d_all_frontiers, num_vertices, source);

        
        std::vector<int> level_offsets;
        std::vector<int> level_sizes;
        level_offsets.reserve(512);
        level_sizes.reserve(512);
        int num_levels   = 0;
        int total_stored = 0;

        
        level_offsets.push_back(0);
        level_sizes.push_back(1);
        num_levels       = 1;
        total_stored     = 1;

        int frontier_size = 1;
        int current_level = 0;

        
        while (frontier_size > 0) {
            cudaMemset(cache.d_next_frontier_size, 0, sizeof(int32_t));

            
            
            launch_bfs_forward(
                d_offsets, d_indices, cache.d_distances, cache.d_sigma,
                cache.d_all_frontiers + level_offsets[num_levels - 1],
                cache.d_all_frontiers + total_stored,
                cache.d_next_frontier_size,
                frontier_size, current_level);

            
            cudaMemcpy(cache.h_next_size, cache.d_next_frontier_size,
                       sizeof(int32_t), cudaMemcpyDeviceToHost);
            int32_t next_size = *cache.h_next_size;

            if (next_size > 0) {
                level_offsets.push_back(total_stored);
                level_sizes.push_back(next_size);
                num_levels++;
                total_stored += next_size;
            }

            frontier_size = next_size;
            current_level++;
        }

        int max_level = num_levels - 1;

        
        for (int d = max_level - 1; d >= 0; d--) {
            launch_dependency(
                d_offsets, d_indices, cache.d_distances, cache.d_sigma,
                cache.d_delta, centralities,
                cache.d_all_frontiers + level_offsets[d], level_sizes[d],
                d, source);
        }

        
        if (include_endpoints) {
            int reachable_count = total_stored - 1;
            launch_endpoints(centralities, cache.d_distances, num_vertices,
                             source, reachable_count);
        }
    }

    
    int n = num_vertices;
    int k = (int)num_samples;
    float adj = include_endpoints ? (float)n : (float)(n - 1);
    bool all_srcs = (k == (int)adj) || include_endpoints;

    if (all_srcs) {
        float scale;
        if (normalized)        scale = (float)k * (adj - 1.0f);
        else if (is_symmetric) scale = (float)k * 2.0f / adj;
        else                   scale = (float)k / adj;
        if (scale != 0.0f)
            launch_normalize_uniform(centralities, num_vertices, 1.0f / scale);
    } else {
        cudaMemset(cache.d_is_source, 0, (size_t)num_vertices * sizeof(uint8_t));
        launch_mark_sources(cache.d_is_source, sample_vertices, num_samples);
        if (normalized) {
            float scale_ns = (float)k       * (adj - 1.0f);
            float scale_s  = (float)(k - 1) * (adj - 1.0f);
            float inv_s  = 1.0f / scale_s;
            float inv_ns = 1.0f / scale_ns;
            launch_normalize_split(centralities, cache.d_is_source, num_vertices, inv_s, inv_ns);
        } else {
            float s_ns = (float)k       / adj;
            float s_s  = (float)(k - 1) / adj;
            if (is_symmetric) { s_ns *= 2.0f; s_s *= 2.0f; }
            float inv_s  = 1.0f / s_s;
            float inv_ns = 1.0f / s_ns;
            launch_normalize_split(centralities, cache.d_is_source, num_vertices, inv_s, inv_ns);
        }
    }
}

}  
