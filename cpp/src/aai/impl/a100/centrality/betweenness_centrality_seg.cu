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
#include <vector>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* h_counter = nullptr;
    int32_t* d_dist = nullptr;
    float* d_sigma = nullptr;
    float* d_delta = nullptr;
    int32_t* d_levels_data = nullptr;
    int32_t* d_next_count = nullptr;
    bool* d_is_source = nullptr;
    int32_t d_dist_capacity = 0;
    int32_t d_sigma_capacity = 0;
    int32_t d_delta_capacity = 0;
    int32_t d_levels_data_capacity = 0;
    int32_t d_is_source_capacity = 0;

    Cache() {
        cudaMallocHost(&h_counter, sizeof(int32_t));
        cudaMalloc(&d_next_count, sizeof(int32_t));
    }

    void ensure(int32_t n) {
        if (d_dist_capacity < n) {
            if (d_dist) cudaFree(d_dist);
            cudaMalloc(&d_dist, (std::size_t)n * sizeof(int32_t));
            d_dist_capacity = n;
        }
        if (d_sigma_capacity < n) {
            if (d_sigma) cudaFree(d_sigma);
            cudaMalloc(&d_sigma, (std::size_t)n * sizeof(float));
            d_sigma_capacity = n;
        }
        if (d_delta_capacity < n) {
            if (d_delta) cudaFree(d_delta);
            cudaMalloc(&d_delta, (std::size_t)n * sizeof(float));
            d_delta_capacity = n;
        }
        if (d_levels_data_capacity < n) {
            if (d_levels_data) cudaFree(d_levels_data);
            cudaMalloc(&d_levels_data, (std::size_t)n * sizeof(int32_t));
            d_levels_data_capacity = n;
        }
        if (d_is_source_capacity < n) {
            if (d_is_source) cudaFree(d_is_source);
            cudaMalloc(&d_is_source, (std::size_t)n * sizeof(bool));
            d_is_source_capacity = n;
        }
    }

    ~Cache() override {
        if (h_counter) cudaFreeHost(h_counter);
        if (d_dist) cudaFree(d_dist);
        if (d_sigma) cudaFree(d_sigma);
        if (d_delta) cudaFree(d_delta);
        if (d_levels_data) cudaFree(d_levels_data);
        if (d_next_count) cudaFree(d_next_count);
        if (d_is_source) cudaFree(d_is_source);
    }
};




__global__ void kern_bfs_expand_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ frontier,
    int frontier_size,
    int32_t* __restrict__ dist,
    float* __restrict__ sigma,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count,
    int next_dist
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t v = frontier[warp_id];
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    float sigma_v = sigma[v];

    for (int32_t e = start + lane; e < end; e += 32) {
        int32_t w = indices[e];
        int32_t old_dist = atomicCAS(&dist[w], -1, next_dist);
        if (old_dist != -1 && old_dist != next_dist) continue;
        atomicAdd(&sigma[w], sigma_v);
        if (old_dist == -1) {
            int pos = atomicAdd(next_count, 1);
            next_frontier[pos] = w;
        }
    }
}




__global__ void kern_backward_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ level_verts,
    int level_size,
    const int32_t* __restrict__ dist,
    const float* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ centrality,
    int32_t source,
    bool include_endpoints
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= level_size) return;

    int32_t v = level_verts[warp_id];
    int32_t d_v = dist[v];
    float sigma_v = sigma[v];
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t target_dist = d_v + 1;

    float sum = 0.0f;
    for (int32_t e = start + lane; e < end; e += 32) {
        int32_t w = indices[e];
        if (dist[w] == target_dist) {
            sum += (1.0f + delta[w]) / sigma[w];
        }
    }

    
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    if (lane == 0) {
        float delta_v = sigma_v * sum;
        delta[v] = delta_v;
        if (v != source) {
            centrality[v] += delta_v;
            if (include_endpoints) {
                centrality[v] += 1.0f;
            }
        }
    }
}



__global__ void kern_add_source_endpoints(float* centrality, int32_t source, int reachable_count) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        centrality[source] += (float)reachable_count;
    }
}

__global__ void kern_mark_sources(bool* is_source, const int32_t* sample_vertices, int num_samples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_samples) {
        is_source[sample_vertices[idx]] = true;
    }
}

__global__ void kern_normalize_uniform(float* centrality, int32_t n, float inv_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        centrality[idx] *= inv_scale;
    }
}

__global__ void kern_normalize_split(float* centrality, const bool* is_source, int32_t n,
                                      float inv_scale_nonsource, float inv_scale_source) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (is_source[idx]) {
            centrality[idx] *= inv_scale_source;
        } else {
            centrality[idx] *= inv_scale_nonsource;
        }
    }
}



static void apply_normalization(float* d_centrality, const int32_t* d_sample_vertices,
                                int32_t n, std::size_t k, bool is_symmetric, bool normalized,
                                bool include_endpoints, Cache& cache, cudaStream_t stream) {
    if (n <= 1) return;

    double adj = include_endpoints ? (double)n : (double)(n - 1);
    bool all_srcs = ((int64_t)k == (int64_t)adj) || include_endpoints;

    if (all_srcs) {
        double scale;
        if (normalized) scale = (double)k * (adj - 1.0);
        else if (is_symmetric) scale = (double)k * 2.0 / adj;
        else scale = (double)k / adj;

        if (scale != 0.0) {
            float inv_scale = (float)(1.0 / scale);
            int block = 256;
            int grid = (n + block - 1) / block;
            kern_normalize_uniform<<<grid, block, 0, stream>>>(d_centrality, n, inv_scale);
        }
    } else {
        double scale_ns, scale_s;
        if (normalized) {
            scale_ns = (double)k * (adj - 1.0);
            scale_s = (double)(k - 1) * (adj - 1.0);
        } else {
            scale_ns = (double)k / adj;
            scale_s = (double)(k - 1) / adj;
            if (is_symmetric) { scale_ns *= 2.0; scale_s *= 2.0; }
        }

        cudaMemsetAsync(cache.d_is_source, 0, (std::size_t)n * sizeof(bool), stream);
        {
            int block = 256;
            int grid = ((int)k + block - 1) / block;
            kern_mark_sources<<<grid, block, 0, stream>>>(cache.d_is_source, d_sample_vertices, (int)k);
        }

        float inv_ns = (float)(1.0 / scale_ns);
        float inv_s = (float)(1.0 / scale_s);
        {
            int block = 256;
            int grid = (n + block - 1) / block;
            kern_normalize_split<<<grid, block, 0, stream>>>(d_centrality, cache.d_is_source, n, inv_ns, inv_s);
        }
    }
}

}  

void betweenness_centrality_seg(const graph32_t& graph,
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
    bool is_symmetric = graph.is_symmetric;

    cache.ensure(num_vertices);

    cudaStream_t stream = 0;
    int32_t h_zero = 0;
    float h_one = 1.0f;

    
    cudaMemsetAsync(centralities, 0, (std::size_t)num_vertices * sizeof(float), stream);

    
    std::vector<int32_t> h_samples(num_samples);
    cudaMemcpyAsync(h_samples.data(), sample_vertices,
                    num_samples * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    
    for (std::size_t s = 0; s < num_samples; s++) {
        int32_t source = h_samples[s];

        
        cudaMemsetAsync(cache.d_dist, 0xFF, (std::size_t)num_vertices * sizeof(int32_t), stream);
        cudaMemsetAsync(cache.d_sigma, 0, (std::size_t)num_vertices * sizeof(float), stream);
        cudaMemsetAsync(cache.d_delta, 0, (std::size_t)num_vertices * sizeof(float), stream);

        
        cudaMemcpyAsync(&cache.d_dist[source], &h_zero, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(&cache.d_sigma[source], &h_one, sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(&cache.d_levels_data[0], &source, sizeof(int32_t), cudaMemcpyHostToDevice, stream);

        
        std::vector<int> level_offsets;
        level_offsets.reserve(64);
        level_offsets.push_back(0);
        level_offsets.push_back(1);

        int current_level = 0;

        while (true) {
            int frontier_start = level_offsets[current_level];
            int frontier_end = level_offsets[current_level + 1];
            int frontier_size = frontier_end - frontier_start;

            if (frontier_size <= 0) break;

            cudaMemsetAsync(cache.d_next_count, 0, sizeof(int32_t), stream);

            {
                int warps = frontier_size;
                int64_t threads = (int64_t)warps * 32;
                int block = 256;
                int grid = (int)((threads + block - 1) / block);
                kern_bfs_expand_warp<<<grid, block, 0, stream>>>(
                    d_offsets, d_indices,
                    cache.d_levels_data + frontier_start, frontier_size,
                    cache.d_dist, cache.d_sigma,
                    cache.d_levels_data + frontier_end, cache.d_next_count,
                    current_level + 1);
            }

            cudaMemcpyAsync(cache.h_counter, cache.d_next_count, sizeof(int32_t),
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            int count = *cache.h_counter;
            if (count == 0) break;

            level_offsets.push_back(frontier_end + count);
            current_level++;
        }

        int max_level = current_level;

        
        for (int d = max_level; d >= 1; d--) {
            int start = level_offsets[d];
            int end = level_offsets[d + 1];
            int size = end - start;

            if (size <= 0) continue;

            int warps = size;
            int64_t threads = (int64_t)warps * 32;
            int block = 256;
            int grid = (int)((threads + block - 1) / block);
            kern_backward_warp<<<grid, block, 0, stream>>>(
                d_offsets, d_indices,
                cache.d_levels_data + start, size,
                cache.d_dist, cache.d_sigma, cache.d_delta,
                centralities, source, include_endpoints);
        }

        if (include_endpoints) {
            int total_reached = level_offsets.back();
            int reachable = total_reached - 1;
            if (reachable > 0) {
                kern_add_source_endpoints<<<1, 1, 0, stream>>>(centralities, source, reachable);
            }
        }
    }

    
    apply_normalization(centralities, sample_vertices, num_vertices, num_samples,
                       is_symmetric, normalized, include_endpoints, cache, stream);

    cudaStreamSynchronize(stream);
}

}  
