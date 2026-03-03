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
#include <optional>
#include <vector>

namespace aai {

namespace {





__global__ void bfs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ frontier,
    int frontier_size,
    int32_t* __restrict__ next_frontier,
    int* __restrict__ next_frontier_size,
    int32_t* __restrict__ distances,
    float* __restrict__ sigma,
    int current_level)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane_id = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int v = frontier[warp_id];
    float sigma_v = sigma[v];
    int new_label = current_level + 1;
    int start = offsets[v];
    int end = offsets[v + 1];

    for (int e = start + lane_id; e < end; e += 32) {
        int w = indices[e];

        
        
        int d_w = __ldg(&distances[w]);
        if (d_w >= 0 && d_w != new_label) continue;

        
        int old_label = atomicCAS(&distances[w], -1, new_label);

        if (old_label != -1 && old_label != new_label) continue;

        atomicAdd(&sigma[w], sigma_v);

        if (old_label == -1) {
            int pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = w;
        }
    }
}





__global__ void backprop_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ frontier,
    int frontier_size,
    const int32_t* __restrict__ distances,
    const float* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ edge_bc,
    int current_level)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane_id = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int v = frontier[warp_id];
    float sigma_v = __ldg(&sigma[v]);
    float delta_v = 0.0f;
    int next_level = current_level + 1;

    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);

    for (int e = start + lane_id; e < end; e += 32) {
        int w = indices[e];
        int d_w = __ldg(&distances[w]);
        if (d_w == next_level) {
            float sigma_w = __ldg(&sigma[w]);
            float delta_w = __ldg(&delta[w]);
            float coeff = sigma_v / sigma_w * (1.0f + delta_w);
            delta_v += coeff;
            edge_bc[e] += coeff;
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        delta_v += __shfl_down_sync(0xffffffff, delta_v, offset);
    }

    if (lane_id == 0) {
        delta[v] = delta_v;
    }
}




__global__ void init_arrays_kernel(
    int32_t* __restrict__ distances,
    float* __restrict__ sigma,
    float* __restrict__ delta,
    int num_vertices,
    int32_t source)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;

    distances[tid] = (tid == source) ? 0 : -1;
    sigma[tid] = (tid == source) ? 1.0f : 0.0f;
    delta[tid] = 0.0f;
}




__global__ void normalize_kernel(
    float* __restrict__ edge_bc,
    int num_edges,
    float factor)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int tid4 = tid * 4;
    if (tid4 + 3 < num_edges) {
        float4* ptr = reinterpret_cast<float4*>(&edge_bc[tid4]);
        float4 val = *ptr;
        val.x *= factor;
        val.y *= factor;
        val.z *= factor;
        val.w *= factor;
        *ptr = val;
    } else {
        for (int i = tid4; i < num_edges && i < tid4 + 4; i++) {
            edge_bc[i] *= factor;
        }
    }
}

struct Cache : Cacheable {
    int32_t* d_distances = nullptr;
    float* d_sigma = nullptr;
    float* d_delta = nullptr;
    int32_t* d_all_frontiers = nullptr;
    int* d_next_size = nullptr;
    int* h_next_size = nullptr;

    int32_t distances_capacity = 0;
    int32_t sigma_capacity = 0;
    int32_t delta_capacity = 0;
    int32_t frontiers_capacity = 0;
    bool next_size_allocated = false;
    bool host_next_size_allocated = false;

    void ensure(int32_t num_vertices) {
        if (distances_capacity < num_vertices) {
            if (d_distances) cudaFree(d_distances);
            cudaMalloc(&d_distances, (size_t)num_vertices * sizeof(int32_t));
            distances_capacity = num_vertices;
        }
        if (sigma_capacity < num_vertices) {
            if (d_sigma) cudaFree(d_sigma);
            cudaMalloc(&d_sigma, (size_t)num_vertices * sizeof(float));
            sigma_capacity = num_vertices;
        }
        if (delta_capacity < num_vertices) {
            if (d_delta) cudaFree(d_delta);
            cudaMalloc(&d_delta, (size_t)num_vertices * sizeof(float));
            delta_capacity = num_vertices;
        }
        if (frontiers_capacity < num_vertices) {
            if (d_all_frontiers) cudaFree(d_all_frontiers);
            cudaMalloc(&d_all_frontiers, (size_t)num_vertices * sizeof(int32_t));
            frontiers_capacity = num_vertices;
        }
        if (!next_size_allocated) {
            cudaMalloc(&d_next_size, sizeof(int));
            next_size_allocated = true;
        }
        if (!host_next_size_allocated) {
            cudaMallocHost(&h_next_size, sizeof(int));
            host_next_size_allocated = true;
        }
    }

    ~Cache() override {
        if (d_distances) cudaFree(d_distances);
        if (d_sigma) cudaFree(d_sigma);
        if (d_delta) cudaFree(d_delta);
        if (d_all_frontiers) cudaFree(d_all_frontiers);
        if (d_next_size) cudaFree(d_next_size);
        if (h_next_size) cudaFreeHost(h_next_size);
    }
};

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

    cache.ensure(num_vertices);

    cudaStream_t stream = 0;
    cudaMemsetAsync(edge_centralities, 0, num_edges * sizeof(float), stream);

    
    std::vector<int32_t> h_samples;
    if (sample_vertices != nullptr && num_samples > 0) {
        h_samples.resize(num_samples);
        cudaMemcpy(h_samples.data(), sample_vertices,
                   num_samples * sizeof(int32_t), cudaMemcpyDeviceToHost);
    } else {
        num_samples = num_vertices;
        h_samples.resize(num_vertices);
        for (int32_t i = 0; i < num_vertices; i++) h_samples[i] = i;
    }

    for (size_t s = 0; s < num_samples; s++) {
        int32_t source = h_samples[s];

        
        {
            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            init_arrays_kernel<<<grid, block, 0, stream>>>(
                cache.d_distances, cache.d_sigma, cache.d_delta, num_vertices, source);
        }

        cudaMemcpyAsync(&cache.d_all_frontiers[0], &source, sizeof(int32_t),
                        cudaMemcpyHostToDevice, stream);

        std::vector<int> h_level_offsets;
        h_level_offsets.push_back(0);
        h_level_offsets.push_back(1);

        int current_level = 0;
        int total_frontier = 1;

        while (true) {
            int level_start = h_level_offsets[current_level];
            int level_end = h_level_offsets[current_level + 1];
            int frontier_size = level_end - level_start;

            if (frontier_size == 0) break;

            cudaMemsetAsync(cache.d_next_size, 0, sizeof(int), stream);

            
            {
                int threads_per_block = 256;
                int warps_per_block = threads_per_block / 32;
                int grid = (frontier_size + warps_per_block - 1) / warps_per_block;
                bfs_kernel<<<grid, threads_per_block, 0, stream>>>(
                    d_offsets, d_indices,
                    &cache.d_all_frontiers[level_start], frontier_size,
                    &cache.d_all_frontiers[level_end], cache.d_next_size,
                    cache.d_distances, cache.d_sigma, current_level);
            }

            cudaMemcpyAsync(cache.h_next_size, cache.d_next_size, sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            int next_size = *cache.h_next_size;
            total_frontier += next_size;
            h_level_offsets.push_back(total_frontier);

            current_level++;

            if (next_size == 0) break;
        }

        
        int num_levels = (int)h_level_offsets.size() - 1;
        for (int level = num_levels - 1; level >= 0; level--) {
            int level_start = h_level_offsets[level];
            int level_end = h_level_offsets[level + 1];
            int frontier_size = level_end - level_start;

            if (frontier_size == 0) continue;

            {
                int threads_per_block = 256;
                int warps_per_block = threads_per_block / 32;
                int grid = (frontier_size + warps_per_block - 1) / warps_per_block;
                backprop_kernel<<<grid, threads_per_block, 0, stream>>>(
                    d_offsets, d_indices,
                    &cache.d_all_frontiers[level_start], frontier_size,
                    cache.d_distances, cache.d_sigma, cache.d_delta, edge_centralities,
                    level);
            }
        }
    }

    
    
    
    
    
    std::optional<float> scale_factor;

    if (normalized) {
        float n = (float)num_vertices;
        scale_factor = n * (n - 1.0f);
    } else if (is_symmetric) {
        scale_factor = 2.0f;
    }

    if (scale_factor.has_value() && num_vertices > 1) {
        if (num_samples < (size_t)num_vertices) {
            *scale_factor *= (float)num_samples / (float)num_vertices;
        }
        float factor = 1.0f / *scale_factor;
        int block = 256;
        int items = (num_edges + 3) / 4;
        int grid = (items + block - 1) / block;
        normalize_kernel<<<grid, block, 0, stream>>>(edge_centralities, num_edges, factor);
    }

    cudaStreamSynchronize(stream);
}

}  
