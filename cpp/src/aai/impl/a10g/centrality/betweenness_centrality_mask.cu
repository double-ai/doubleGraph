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
#include <cub/cub.cuh>
#include <cstdint>
#include <vector>

namespace aai {

namespace {



__global__ void compute_active_flags_kernel(
    const uint32_t* __restrict__ edge_mask, int32_t* __restrict__ active, int32_t num_edges) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e <= num_edges; e += blockDim.x * gridDim.x)
        active[e] = (e < num_edges) ? ((edge_mask[e >> 5] >> (e & 31)) & 1) : 0;
}

__global__ void build_compact_offsets_kernel(
    const int32_t* __restrict__ old_offsets, const int32_t* __restrict__ prefix_sum,
    int32_t* __restrict__ new_offsets, int32_t num_vertices) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v <= num_vertices; v += blockDim.x * gridDim.x)
        new_offsets[v] = prefix_sum[old_offsets[v]];
}

__global__ void compact_edges_kernel(
    const int32_t* __restrict__ old_indices, const int32_t* __restrict__ active,
    const int32_t* __restrict__ prefix_sum, int32_t* __restrict__ new_indices, int32_t num_edges) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_edges; e += blockDim.x * gridDim.x)
        if (active[e]) new_indices[prefix_sum[e]] = old_indices[e];
}



__global__ void set_source_kernel(
    int32_t* __restrict__ dist, float* __restrict__ sigma,
    int32_t* __restrict__ all_frontiers, int32_t source) {
    dist[source] = 0;
    sigma[source] = 1.0f;
    all_frontiers[0] = source;
}


__global__ void bfs_forward_thread_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    int32_t* __restrict__ dist, float* __restrict__ sigma,
    const int32_t* __restrict__ frontier, int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size, int32_t frontier_size, int32_t current_level)
{
    int new_level = current_level + 1;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < frontier_size; tid += blockDim.x * gridDim.x) {
        int src = frontier[tid];
        int start = offsets[src];
        int end = offsets[src + 1];
        float src_sigma = sigma[src];
        for (int e = start; e < end; e++) {
            int dst = indices[e];
            int old_dist = atomicCAS(&dist[dst], -1, new_level);
            if (old_dist != -1 && old_dist != new_level) continue;
            atomicAdd(&sigma[dst], src_sigma);
            if (old_dist == -1) {
                int pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = dst;
            }
        }
    }
}


__global__ void bfs_forward_warp_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    int32_t* __restrict__ dist, float* __restrict__ sigma,
    const int32_t* __restrict__ frontier, int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size, int32_t frontier_size, int32_t current_level)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    int new_level = current_level + 1;

    for (int w = warp_id; w < frontier_size; w += num_warps) {
        int src = frontier[w];
        int start = offsets[src];
        int end = offsets[src + 1];
        float src_sigma = sigma[src];

        for (int e = start + lane; e < end; e += 32) {
            int dst = indices[e];
            int old_dist = atomicCAS(&dist[dst], -1, new_level);
            if (old_dist != -1 && old_dist != new_level) continue;
            atomicAdd(&sigma[dst], src_sigma);
            if (old_dist == -1) {
                int pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = dst;
            }
        }
    }
}



__global__ void backward_warp_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ dist, const float* __restrict__ sigma,
    float* __restrict__ delta, const int32_t* __restrict__ level_vertices,
    int32_t num_level_vertices, int32_t current_level)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;
    int next_level = current_level + 1;

    for (int w = warp_id; w < num_level_vertices; w += num_warps) {
        int v = level_vertices[w];
        int start = offsets[v];
        int end = offsets[v + 1];
        float sigma_v = sigma[v];

        float partial = 0.0f;
        for (int e = start + lane; e < end; e += 32) {
            int nbr = indices[e];
            if (dist[nbr] == next_level)
                partial += (sigma_v / sigma[nbr]) * (1.0f + delta[nbr]);
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);

        if (lane == 0) delta[v] = partial;
    }
}


__global__ void backward_thread_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ dist, const float* __restrict__ sigma,
    float* __restrict__ delta, const int32_t* __restrict__ level_vertices,
    int32_t num_level_vertices, int32_t current_level)
{
    int next_level = current_level + 1;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < num_level_vertices; tid += blockDim.x * gridDim.x) {
        int v = level_vertices[tid];
        int start = offsets[v];
        int end = offsets[v + 1];
        float sigma_v = sigma[v];
        float delta_v = 0.0f;
        for (int e = start; e < end; e++) {
            int nbr = indices[e];
            if (dist[nbr] == next_level)
                delta_v += (sigma_v / sigma[nbr]) * (1.0f + delta[nbr]);
        }
        delta[v] = delta_v;
    }
}


__global__ void accumulate_bc_kernel(
    float* __restrict__ bc, const float* __restrict__ delta,
    const int32_t* __restrict__ vertices, int32_t num_vertices,
    int32_t source, bool include_endpoints, float source_ep)
{
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < num_vertices; tid += blockDim.x * gridDim.x) {
        int v = vertices[tid];
        float add = 0.0f;
        if (v != source) {
            add = delta[v];
            if (include_endpoints) add += 1.0f;
        } else if (include_endpoints) {
            add = source_ep;
        }
        if (add != 0.0f) bc[v] += add;
    }
}

__global__ void normalize_uniform_kernel(float* bc, int32_t nv, float inv_scale) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < nv; v += blockDim.x * gridDim.x)
        bc[v] *= inv_scale;
}

__global__ void normalize_split_kernel(
    float* bc, const uint32_t* bm, int32_t nv, float inv_s, float inv_ns) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < nv; v += blockDim.x * gridDim.x) {
        bool is_src = (bm[v >> 5] >> (v & 31)) & 1;
        bc[v] *= (is_src ? inv_s : inv_ns);
    }
}



struct Cache : Cacheable {
    int32_t* d_dist = nullptr;
    float* d_sigma = nullptr;
    float* d_delta = nullptr;
    int32_t* d_all_frontiers = nullptr;
    int32_t* d_frontier_size = nullptr;
    uint32_t* d_is_source_bitmap = nullptr;
    int32_t* d_active = nullptr;
    int32_t* d_prefix_sum = nullptr;
    int32_t* d_compact_offsets = nullptr;
    int32_t* d_compact_indices = nullptr;
    int32_t* h_frontier_size = nullptr;
    void* d_cub_temp = nullptr;
    size_t cub_temp_bytes = 0;

    int32_t alloc_verts = 0;
    int32_t alloc_edges = 0;

    Cache() {
        cudaMalloc(&d_frontier_size, sizeof(int32_t));
        cudaHostAlloc(&h_frontier_size, sizeof(int32_t), cudaHostAllocDefault);
    }

    void ensure_buffers(int32_t nv, int32_t ne) {
        if (nv > alloc_verts) {
            if (d_dist) cudaFree(d_dist);
            if (d_sigma) cudaFree(d_sigma);
            if (d_delta) cudaFree(d_delta);
            if (d_all_frontiers) cudaFree(d_all_frontiers);
            if (d_is_source_bitmap) cudaFree(d_is_source_bitmap);
            if (d_compact_offsets) cudaFree(d_compact_offsets);
            cudaMalloc(&d_dist, nv * sizeof(int32_t));
            cudaMalloc(&d_sigma, nv * sizeof(float));
            cudaMalloc(&d_delta, nv * sizeof(float));
            cudaMalloc(&d_all_frontiers, nv * sizeof(int32_t));
            cudaMalloc(&d_is_source_bitmap, ((nv + 31) / 32) * sizeof(uint32_t));
            cudaMalloc(&d_compact_offsets, (nv + 1) * sizeof(int32_t));
            alloc_verts = nv;
        }
        if (ne > alloc_edges) {
            if (d_active) cudaFree(d_active);
            if (d_prefix_sum) cudaFree(d_prefix_sum);
            if (d_compact_indices) cudaFree(d_compact_indices);
            cudaMalloc(&d_active, (ne + 1) * sizeof(int32_t));
            cudaMalloc(&d_prefix_sum, (ne + 1) * sizeof(int32_t));
            cudaMalloc(&d_compact_indices, (ne > 0 ? ne : 1) * sizeof(int32_t));
            alloc_edges = ne;
        }
    }

    ~Cache() override {
        if (d_dist) cudaFree(d_dist);
        if (d_sigma) cudaFree(d_sigma);
        if (d_delta) cudaFree(d_delta);
        if (d_all_frontiers) cudaFree(d_all_frontiers);
        if (d_frontier_size) cudaFree(d_frontier_size);
        if (d_is_source_bitmap) cudaFree(d_is_source_bitmap);
        if (d_active) cudaFree(d_active);
        if (d_prefix_sum) cudaFree(d_prefix_sum);
        if (d_compact_offsets) cudaFree(d_compact_offsets);
        if (d_compact_indices) cudaFree(d_compact_indices);
        if (h_frontier_size) cudaFreeHost(h_frontier_size);
        if (d_cub_temp) cudaFree(d_cub_temp);
    }
};

}  

void betweenness_centrality_mask(const graph32_t& graph,
                                  float* centralities,
                                  bool normalized,
                                  bool include_endpoints,
                                  const int32_t* sample_vertices,
                                  std::size_t num_samples) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    bool is_symmetric = graph.is_symmetric;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cache.ensure_buffers(num_vertices, num_edges);

    float* d_bc = centralities;
    cudaMemset(d_bc, 0, num_vertices * sizeof(float));

    std::vector<int32_t> h_samples(num_samples);
    cudaMemcpy(h_samples.data(), sample_vertices, num_samples * sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    {
        int b = 256, g = ((num_edges + 1) + b - 1) / b;
        if (g > 4096) g = 4096;
        compute_active_flags_kernel<<<g, b>>>(d_edge_mask, cache.d_active, num_edges);
    }

    
    {
        size_t needed = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, needed, cache.d_active, cache.d_prefix_sum, num_edges + 1);
        if (needed > cache.cub_temp_bytes) {
            if (cache.d_cub_temp) cudaFree(cache.d_cub_temp);
            cudaMalloc(&cache.d_cub_temp, needed);
            cache.cub_temp_bytes = needed;
        }
        cub::DeviceScan::ExclusiveSum(cache.d_cub_temp, cache.cub_temp_bytes, cache.d_active, cache.d_prefix_sum, num_edges + 1);
    }

    int32_t total_active_edges;
    cudaMemcpy(&total_active_edges, cache.d_prefix_sum + num_edges, sizeof(int32_t), cudaMemcpyDeviceToHost);

    {
        int b = 256, g = ((num_vertices + 1) + b - 1) / b;
        if (g > 4096) g = 4096;
        build_compact_offsets_kernel<<<g, b>>>(d_offsets, cache.d_prefix_sum, cache.d_compact_offsets, num_vertices);
    }

    if (total_active_edges > 0) {
        int b = 256, g = (num_edges + b - 1) / b;
        if (g > 4096) g = 4096;
        compact_edges_kernel<<<g, b>>>(d_indices, cache.d_active, cache.d_prefix_sum, cache.d_compact_indices, num_edges);
    }

    
    int32_t bw = (num_vertices + 31) / 32;
    std::vector<uint32_t> h_bitmap(bw, 0);
    for (std::size_t i = 0; i < num_samples; i++)
        h_bitmap[h_samples[i] >> 5] |= (1u << (h_samples[i] & 31));
    cudaMemcpy(cache.d_is_source_bitmap, h_bitmap.data(), bw * sizeof(uint32_t), cudaMemcpyHostToDevice);

    
    std::vector<int32_t> level_offsets;

    for (std::size_t si = 0; si < num_samples; si++) {
        int32_t source = h_samples[si];

        
        cudaMemsetAsync(cache.d_dist, 0xFF, num_vertices * sizeof(int32_t)); 
        cudaMemsetAsync(cache.d_sigma, 0, num_vertices * sizeof(float));
        cudaMemsetAsync(cache.d_delta, 0, num_vertices * sizeof(float));
        set_source_kernel<<<1, 1>>>(cache.d_dist, cache.d_sigma, cache.d_all_frontiers, source);

        int32_t h_fs = 1, total_reachable = 1, current_level = 0;
        level_offsets.clear();
        level_offsets.push_back(0);
        level_offsets.push_back(1);

        
        while (h_fs > 0) {
            cudaMemsetAsync(cache.d_frontier_size, 0, sizeof(int32_t));
            {
                int b = 256;
                int warps_per_block = b / 32;
                int g = (h_fs + warps_per_block - 1) / warps_per_block;
                if (g > 4096) g = 4096;
                bfs_forward_warp_kernel<<<g, b>>>(
                    cache.d_compact_offsets, cache.d_compact_indices,
                    cache.d_dist, cache.d_sigma,
                    cache.d_all_frontiers + level_offsets[current_level],
                    cache.d_all_frontiers + total_reachable,
                    cache.d_frontier_size, h_fs, current_level);
            }

            cudaMemcpy(cache.h_frontier_size, cache.d_frontier_size, sizeof(int32_t), cudaMemcpyDeviceToHost);
            h_fs = *cache.h_frontier_size;
            if (h_fs > 0) {
                total_reachable += h_fs;
                level_offsets.push_back(total_reachable);
            }
            current_level++;
        }

        
        int32_t num_levels = (int32_t)level_offsets.size() - 1;
        for (int32_t L = num_levels - 2; L >= 0; L--) {
            int32_t nlv = level_offsets[L + 1] - level_offsets[L];
            if (nlv == 0) continue;
            int b = 256;
            int warps_per_block = b / 32;
            int g = (nlv + warps_per_block - 1) / warps_per_block;
            if (g > 4096) g = 4096;
            backward_warp_kernel<<<g, b>>>(
                cache.d_compact_offsets, cache.d_compact_indices,
                cache.d_dist, cache.d_sigma, cache.d_delta,
                cache.d_all_frontiers + level_offsets[L],
                nlv, L);
        }

        
        float source_ep = include_endpoints ? (float)(total_reachable - 1) : 0.0f;
        if (total_reachable > 0) {
            int b = 256, g = (total_reachable + b - 1) / b;
            if (g > 4096) g = 4096;
            accumulate_bc_kernel<<<g, b>>>(d_bc, cache.d_delta, cache.d_all_frontiers, total_reachable, source, include_endpoints, source_ep);
        }
    }

    
    int32_t n = num_vertices;
    int64_t k = (int64_t)num_samples;
    int64_t adj_i = include_endpoints ? (int64_t)n : (int64_t)(n - 1);
    double adj = (double)adj_i;
    bool all_srcs = (k == adj_i) || include_endpoints;

    if (n > 1) {
        if (all_srcs) {
            double scale = normalized ? (double)k * (adj - 1.0) : (is_symmetric ? (double)k * 2.0 / adj : (double)k / adj);
            if (scale != 0.0) {
                int b = 256, g = (n + b - 1) / b;
                normalize_uniform_kernel<<<g, b>>>(d_bc, n, (float)(1.0 / scale));
            }
        } else if (normalized) {
            double sns = (double)k * (adj - 1.0), ss = (double)(k - 1) * (adj - 1.0);
            int b = 256, g = (n + b - 1) / b;
            normalize_split_kernel<<<g, b>>>(d_bc, cache.d_is_source_bitmap, n,
                (float)(1.0 / ss), (float)(1.0 / sns));
        } else {
            double sns = (double)k / adj, ss = (double)(k - 1) / adj;
            if (is_symmetric) { sns *= 2.0; ss *= 2.0; }
            int b = 256, g = (n + b - 1) / b;
            normalize_split_kernel<<<g, b>>>(d_bc, cache.d_is_source_bitmap, n,
                (float)(1.0 / ss), (float)(1.0 / sns));
        }
    }

    cudaDeviceSynchronize();
}

}  
