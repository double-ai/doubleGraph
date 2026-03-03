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
#include <cub/cub.cuh>
#include <vector>

namespace aai {

namespace {

static constexpr int32_t EDGE_PARALLEL_THRESHOLD = 1024;





struct Cache : Cacheable {
    int32_t* d_distances = nullptr;
    float* d_sigma = nullptr;
    float* d_delta = nullptr;
    int32_t* d_all_frontiers = nullptr;
    int32_t* d_next_frontier_size = nullptr;
    int32_t* d_degrees = nullptr;
    int32_t* d_degree_prefix = nullptr;
    void* d_cub_temp = nullptr;
    size_t cub_temp_bytes = 0;
    int32_t max_vertices = 0;
    int32_t* h_pinned = nullptr;

    Cache() {
        cudaMalloc(&d_next_frontier_size, sizeof(int32_t));
        cudaMallocHost(&h_pinned, 2 * sizeof(int32_t));
    }

    void ensure(int32_t n) {
        if (n <= max_vertices) return;
        if (d_distances) cudaFree(d_distances);
        if (d_sigma) cudaFree(d_sigma);
        if (d_delta) cudaFree(d_delta);
        if (d_all_frontiers) cudaFree(d_all_frontiers);
        if (d_degrees) cudaFree(d_degrees);
        if (d_degree_prefix) cudaFree(d_degree_prefix);
        if (d_cub_temp) cudaFree(d_cub_temp);

        max_vertices = n;
        cudaMalloc(&d_distances, (size_t)n * sizeof(int32_t));
        cudaMalloc(&d_sigma, (size_t)n * sizeof(float));
        cudaMalloc(&d_delta, (size_t)n * sizeof(float));
        cudaMalloc(&d_all_frontiers, (size_t)n * sizeof(int32_t));
        cudaMalloc(&d_degrees, (size_t)n * sizeof(int32_t));
        cudaMalloc(&d_degree_prefix, (size_t)(n + 1) * sizeof(int32_t));

        size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_degrees, d_degree_prefix, n, (cudaStream_t)0);
        cub_temp_bytes = temp_bytes;
        cudaMalloc(&d_cub_temp, cub_temp_bytes);
    }

    ~Cache() override {
        if (d_distances) cudaFree(d_distances);
        if (d_sigma) cudaFree(d_sigma);
        if (d_delta) cudaFree(d_delta);
        if (d_all_frontiers) cudaFree(d_all_frontiers);
        if (d_next_frontier_size) cudaFree(d_next_frontier_size);
        if (d_degrees) cudaFree(d_degrees);
        if (d_degree_prefix) cudaFree(d_degree_prefix);
        if (d_cub_temp) cudaFree(d_cub_temp);
        if (h_pinned) cudaFreeHost(h_pinned);
    }
};





__global__ void init_bfs_kernel(
    int32_t* __restrict__ distances,
    float* __restrict__ sigma,
    float* __restrict__ delta,
    int32_t num_vertices,
    int32_t source
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;
    distances[tid] = (tid == source) ? 0 : -1;
    sigma[tid] = (tid == source) ? 1.0f : 0.0f;
    delta[tid] = 0.0f;
}

__global__ void set_source_kernel(
    int32_t* __restrict__ distances,
    float* __restrict__ sigma,
    int32_t* __restrict__ frontier,
    int32_t source
) {
    distances[source] = 0;
    sigma[source] = 1.0f;
    frontier[0] = source;
}

__global__ void reset_discovered_kernel(
    int32_t* __restrict__ distances,
    float* __restrict__ sigma,
    float* __restrict__ delta,
    const int32_t* __restrict__ discovered,
    int32_t num_discovered
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_discovered) return;
    int32_t v = discovered[tid];
    distances[v] = -1;
    sigma[v] = 0.0f;
    delta[v] = 0.0f;
}






__global__ void compute_frontier_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ degrees,
    int32_t frontier_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int32_t v = frontier[tid];
    degrees[tid] = offsets[v + 1] - offsets[v];
}

__global__ void bfs_edge_parallel_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    float* __restrict__ sigma,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    const int32_t* __restrict__ degree_prefix,
    int32_t frontier_size,
    int32_t total_edges,
    int32_t current_level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_edges) return;

    int lo = 0, hi = frontier_size - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (degree_prefix[mid] <= tid) lo = mid;
        else hi = mid - 1;
    }

    int32_t src = frontier[lo];
    int32_t edge_offset = tid - degree_prefix[lo];
    int32_t e = offsets[src] + edge_offset;
    int32_t dst = indices[e];
    int32_t new_dist = current_level + 1;

    float src_sigma = sigma[src];

    int32_t old_dist = atomicCAS(&distances[dst], -1, new_dist);
    if (old_dist != -1 && old_dist != new_dist) return;

    atomicAdd(&sigma[dst], src_sigma);

    if (old_dist == -1) {
        int32_t pos = atomicAdd(next_frontier_size, 1);
        next_frontier[pos] = dst;
    }
}

__global__ void bfs_vertex_parallel_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    float* __restrict__ sigma,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t frontier_size,
    int32_t current_level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int32_t src = frontier[tid];
    int32_t start = offsets[src];
    int32_t end = offsets[src + 1];
    int32_t new_dist = current_level + 1;
    float src_sigma = sigma[src];

    for (int32_t e = start; e < end; e++) {
        int32_t dst = indices[e];
        int32_t old_dist = atomicCAS(&distances[dst], -1, new_dist);
        if (old_dist != -1 && old_dist != new_dist) continue;
        atomicAdd(&sigma[dst], src_sigma);
        if (old_dist == -1) {
            int32_t pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = dst;
        }
    }
}





__global__ void backward_edge_parallel_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ distances,
    const float* __restrict__ sigma,
    float* __restrict__ delta,
    const int32_t* __restrict__ frontier,
    const int32_t* __restrict__ degree_prefix,
    int32_t frontier_size,
    int32_t total_edges,
    int32_t current_level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_edges) return;

    int lo = 0, hi = frontier_size - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (degree_prefix[mid] <= tid) lo = mid;
        else hi = mid - 1;
    }

    int32_t v = frontier[lo];
    int32_t edge_offset = tid - degree_prefix[lo];
    int32_t e = offsets[v] + edge_offset;
    int32_t w = indices[e];
    int32_t next_level = current_level + 1;

    if (distances[w] == next_level) {
        float contribution = sigma[v] * (1.0f + delta[w]) / sigma[w];
        atomicAdd(&delta[v], contribution);
    }
}

__global__ void backward_vertex_parallel_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ distances,
    const float* __restrict__ sigma,
    float* __restrict__ delta,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t current_level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int32_t v = frontier[tid];
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t next_level = current_level + 1;

    float v_sigma = sigma[v];
    float sum = 0.0f;

    for (int32_t e = start; e < end; e++) {
        int32_t w = indices[e];
        if (distances[w] == next_level) {
            sum += (1.0f + delta[w]) / sigma[w];
        }
    }
    delta[v] = v_sigma * sum;
}





__global__ void accumulate_bc_kernel(
    float* __restrict__ bc,
    const float* __restrict__ delta,
    int32_t num_discovered,
    const int32_t* __restrict__ discovered_vertices,
    int32_t source,
    bool include_endpoints
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_discovered) return;
    int32_t v = discovered_vertices[tid];
    if (v != source) {
        float val = delta[v];
        if (include_endpoints) val += 1.0f;
        bc[v] += val;
    } else {
        if (include_endpoints) bc[v] += static_cast<float>(num_discovered - 1);
    }
}

__global__ void normalize_bc_kernel(float* __restrict__ bc, int32_t nv, float f) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nv) return;
    bc[tid] *= f;
}

__global__ void adjust_sources_kernel(
    float* __restrict__ bc, const int32_t* __restrict__ sources,
    int32_t num_sources, float adjustment
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_sources) return;
    bc[sources[tid]] *= adjustment;
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
    int32_t n = graph.number_of_vertices;
    bool is_symmetric = graph.is_symmetric;

    cache.ensure(n);

    int32_t* d_distances = cache.d_distances;
    float* d_sigma = cache.d_sigma;
    float* d_delta = cache.d_delta;
    int32_t* d_all_frontiers = cache.d_all_frontiers;
    int32_t* d_next_frontier_size = cache.d_next_frontier_size;
    int32_t* d_degrees = cache.d_degrees;
    int32_t* d_degree_prefix = cache.d_degree_prefix;
    void* d_cub_temp = cache.d_cub_temp;
    size_t cub_temp_bytes = cache.cub_temp_bytes;
    int32_t* h_pinned = cache.h_pinned;

    float* d_bc = centralities;
    cudaMemset(d_bc, 0, n * sizeof(float));

    std::vector<int32_t> h_sources;
    bool has_sampling = (num_samples > 0 && sample_vertices != nullptr);
    if (has_sampling) {
        h_sources.resize(num_samples);
        cudaMemcpy(h_sources.data(), sample_vertices,
                   num_samples * sizeof(int32_t), cudaMemcpyDeviceToHost);
    } else {
        h_sources.resize(n);
        for (int32_t i = 0; i < n; i++) h_sources[i] = i;
    }
    int32_t k = static_cast<int32_t>(h_sources.size());

    cudaStream_t stream = 0;
    int32_t prev_total_discovered = 0;
    bool first_source = true;

    for (int32_t source : h_sources) {
        
        if (first_source) {
            {
                int b=256, g=(n+b-1)/b;
                init_bfs_kernel<<<g,b,0,stream>>>(d_distances, d_sigma, d_delta, n, source);
            }
            set_source_kernel<<<1,1,0,stream>>>(d_distances, d_sigma, d_all_frontiers, source);
            first_source = false;
        } else {
            {
                int b=256, g=(prev_total_discovered+b-1)/b;
                if (g > 0)
                    reset_discovered_kernel<<<g,b,0,stream>>>(d_distances, d_sigma, d_delta,
                                                              d_all_frontiers, prev_total_discovered);
            }
            set_source_kernel<<<1,1,0,stream>>>(d_distances, d_sigma, d_all_frontiers, source);
        }

        
        std::vector<int32_t> level_offsets;
        level_offsets.push_back(0);
        int32_t total_discovered = 1;
        int32_t level = 0;

        std::vector<int32_t> level_total_edges;

        while (true) {
            int32_t frontier_start = level_offsets[level];
            int32_t frontier_size = total_discovered - frontier_start;
            if (frontier_size == 0) break;

            const int32_t* frontier_ptr = &d_all_frontiers[frontier_start];

            
            {
                int b=256, g=(frontier_size+b-1)/b;
                if (g > 0)
                    compute_frontier_degrees_kernel<<<g,b,0,stream>>>(d_offsets, frontier_ptr, d_degrees, frontier_size);
            }
            {
                size_t temp = cub_temp_bytes;
                cub::DeviceScan::ExclusiveSum(d_cub_temp, temp, d_degrees, d_degree_prefix, frontier_size, stream);
            }

            
            cudaMemcpyAsync(&h_pinned[0], &d_degree_prefix[frontier_size - 1],
                            sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(&h_pinned[1], &d_degrees[frontier_size - 1],
                            sizeof(int32_t), cudaMemcpyDeviceToHost, stream);

            
            cudaMemsetAsync(d_next_frontier_size, 0, sizeof(int32_t), stream);

            cudaStreamSynchronize(stream);
            int32_t total_edges = h_pinned[0] + h_pinned[1];

            
            if (total_edges > EDGE_PARALLEL_THRESHOLD) {
                int b=256, g=(total_edges+b-1)/b;
                if (g > 0)
                    bfs_edge_parallel_kernel<<<g,b,0,stream>>>(d_offsets, d_indices, d_distances, d_sigma,
                                                               frontier_ptr, &d_all_frontiers[total_discovered],
                                                               d_next_frontier_size, d_degree_prefix,
                                                               frontier_size, total_edges, level);
            } else {
                int b=256, g=(frontier_size+b-1)/b;
                if (g > 0)
                    bfs_vertex_parallel_kernel<<<g,b,0,stream>>>(d_offsets, d_indices, d_distances, d_sigma,
                                                                  frontier_ptr, &d_all_frontiers[total_discovered],
                                                                  d_next_frontier_size,
                                                                  frontier_size, level);
            }

            
            cudaMemcpyAsync(&h_pinned[0], d_next_frontier_size, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            int32_t next_size = h_pinned[0];
            level_offsets.push_back(total_discovered);
            level_total_edges.push_back(total_edges);
            total_discovered += next_size;
            if (next_size == 0) break;
            level++;
        }

        int32_t max_level = static_cast<int32_t>(level_offsets.size()) - 1;

        
        for (int32_t L = max_level - 1; L >= 0; L--) {
            int32_t fs = level_offsets[L];
            int32_t fsize = level_offsets[L + 1] - fs;
            if (fsize == 0) continue;

            const int32_t* frontier_ptr = &d_all_frontiers[fs];

            int32_t total_edges = level_total_edges[L];

            if (total_edges > EDGE_PARALLEL_THRESHOLD) {
                
                {
                    int b=256, g=(fsize+b-1)/b;
                    if (g > 0)
                        compute_frontier_degrees_kernel<<<g,b,0,stream>>>(d_offsets, frontier_ptr, d_degrees, fsize);
                }
                {
                    size_t temp = cub_temp_bytes;
                    cub::DeviceScan::ExclusiveSum(d_cub_temp, temp, d_degrees, d_degree_prefix, fsize, stream);
                }

                {
                    int b=256, g=(total_edges+b-1)/b;
                    if (g > 0)
                        backward_edge_parallel_kernel<<<g,b,0,stream>>>(d_offsets, d_indices, d_distances, d_sigma,
                                                                         d_delta, frontier_ptr, d_degree_prefix,
                                                                         fsize, total_edges, L);
                }
            } else {
                int b=256, g=(fsize+b-1)/b;
                if (g > 0)
                    backward_vertex_parallel_kernel<<<g,b,0,stream>>>(d_offsets, d_indices, d_distances, d_sigma,
                                                                       d_delta, frontier_ptr, fsize, L);
            }
        }

        {
            int b=256, g=(total_discovered+b-1)/b;
            if (g > 0)
                accumulate_bc_kernel<<<g,b,0,stream>>>(d_bc, d_delta, total_discovered, d_all_frontiers,
                                                        source, include_endpoints);
        }
        prev_total_discovered = total_discovered;
    }

    
    int32_t N = include_endpoints ? n : (n - 1);
    if (N >= 2) {
        if (!has_sampling || include_endpoints) {
            int32_t K_source = has_sampling ? k : N;
            float scale;
            if (normalized) {
                scale = 1.0f / (static_cast<float>(K_source) * (N - 1));
            } else {
                int correction = is_symmetric ? 2 : 1;
                scale = static_cast<float>(N) / (static_cast<float>(K_source) * correction);
            }
            if (scale != 1.0f) {
                int b=256, g=(n+b-1)/b;
                if (g > 0)
                    normalize_bc_kernel<<<g,b,0,stream>>>(d_bc, n, scale);
            }
        } else {
            int correction = is_symmetric ? 2 : 1;
            float scale_nonsource, scale_source;
            if (normalized) {
                scale_nonsource = 1.0f / (static_cast<float>(k) * (N - 1));
                scale_source = 1.0f / (static_cast<float>(k - 1) * (N - 1));
            } else {
                scale_nonsource = static_cast<float>(N) / (static_cast<float>(k) * correction);
                scale_source = static_cast<float>(N) / (static_cast<float>(k - 1) * correction);
            }
            {
                int b=256, g=(n+b-1)/b;
                if (g > 0)
                    normalize_bc_kernel<<<g,b,0,stream>>>(d_bc, n, scale_nonsource);
            }
            {
                float adj = scale_source / scale_nonsource;
                int b=256, g=(k+b-1)/b;
                if (g > 0)
                    adjust_sources_kernel<<<g,b,0,stream>>>(d_bc, sample_vertices, k, adj);
            }
        }
    }

    cudaStreamSynchronize(stream);
}

}  
