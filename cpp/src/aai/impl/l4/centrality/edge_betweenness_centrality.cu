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

static constexpr int MAX_CONCURRENT = 20;

struct Cache : Cacheable {
    int32_t* d_dist[MAX_CONCURRENT] = {};
    float* d_sigma[MAX_CONCURRENT] = {};
    float* d_delta[MAX_CONCURRENT] = {};
    int32_t* d_level_vertices[MAX_CONCURRENT] = {};
    int32_t* d_frontier_count[MAX_CONCURRENT] = {};
    int32_t* h_frontier_count[MAX_CONCURRENT] = {};
    cudaStream_t streams[MAX_CONCURRENT] = {};
    size_t alloc_v_ = 0;

    Cache() {
        for (int i = 0; i < MAX_CONCURRENT; i++) {
            cudaMalloc(&d_frontier_count[i], sizeof(int32_t));
            cudaMallocHost(&h_frontier_count[i], sizeof(int32_t));
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
        }
    }

    void ensure_buffers(size_t nv) {
        if (nv <= alloc_v_) return;
        for (int i = 0; i < MAX_CONCURRENT; i++) {
            if (d_dist[i]) { cudaFree(d_dist[i]); d_dist[i] = nullptr; }
            if (d_sigma[i]) { cudaFree(d_sigma[i]); d_sigma[i] = nullptr; }
            if (d_delta[i]) { cudaFree(d_delta[i]); d_delta[i] = nullptr; }
            if (d_level_vertices[i]) { cudaFree(d_level_vertices[i]); d_level_vertices[i] = nullptr; }
        }
        alloc_v_ = nv;
        for (int i = 0; i < MAX_CONCURRENT; i++) {
            cudaMalloc(&d_dist[i], alloc_v_ * sizeof(int32_t));
            cudaMalloc(&d_sigma[i], alloc_v_ * sizeof(float));
            cudaMalloc(&d_delta[i], alloc_v_ * sizeof(float));
            cudaMalloc(&d_level_vertices[i], alloc_v_ * sizeof(int32_t));
        }
    }

    ~Cache() override {
        for (int i = 0; i < MAX_CONCURRENT; i++) {
            if (d_dist[i]) cudaFree(d_dist[i]);
            if (d_sigma[i]) cudaFree(d_sigma[i]);
            if (d_delta[i]) cudaFree(d_delta[i]);
            if (d_level_vertices[i]) cudaFree(d_level_vertices[i]);
            if (d_frontier_count[i]) cudaFree(d_frontier_count[i]);
            if (h_frontier_count[i]) cudaFreeHost(h_frontier_count[i]);
            if (streams[i]) cudaStreamDestroy(streams[i]);
        }
    }
};


__global__ void bfs_expand_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ frontier,
    const int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count,
    int32_t* __restrict__ dist,
    float* __restrict__ sigma,
    const int32_t current_level
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    const int u = frontier[warp_id];
    const int start = __ldg(&offsets[u]);
    const int end = __ldg(&offsets[u + 1]);
    const float sigma_u = sigma[u];
    const int next_level = current_level + 1;

    for (int e = start + lane; e < end; e += 32) {
        const int v = __ldg(&indices[e]);
        const int old_dist = atomicCAS(&dist[v], -1, next_level);
        if (old_dist == -1) {
            const int pos = atomicAdd(next_count, 1);
            next_frontier[pos] = v;
            atomicAdd(&sigma[v], sigma_u);
        } else if (old_dist == next_level) {
            atomicAdd(&sigma[v], sigma_u);
        }
    }
}



__global__ void backward_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ level_verts,
    const int32_t level_size,
    const int32_t* __restrict__ dist,
    const float* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ edge_bc,
    const int32_t current_level
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= level_size) return;

    const int u = level_verts[warp_id];
    const int start = __ldg(&offsets[u]);
    const int end = __ldg(&offsets[u + 1]);
    const float sigma_u = sigma[u];
    float local_delta = 0.0f;
    const int next_level = current_level + 1;

    for (int e = start + lane; e < end; e += 32) {
        const int v = __ldg(&indices[e]);
        if (__ldg(&dist[v]) == next_level) {
            const float coeff = sigma_u / __ldg(&sigma[v]) * (1.0f + __ldg(&delta[v]));
            local_delta += coeff;
            atomicAdd(&edge_bc[e], coeff);
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_delta += __shfl_down_sync(0xffffffff, local_delta, offset);
    }

    if (lane == 0) {
        delta[u] = local_delta;
    }
}


__global__ void set_source_kernel(
    int32_t* __restrict__ dist,
    float* __restrict__ sigma,
    int32_t* __restrict__ level_vertices,
    int32_t source
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dist[source] = 0;
        sigma[source] = 1.0f;
        level_vertices[0] = source;
    }
}


__global__ void scale_kernel(float* __restrict__ data, const int32_t n, const float factor) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] *= factor;
    }
}

struct LevelInfo { int32_t start, size; };

}  

void edge_betweenness_centrality(const graph32_t& graph,
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

    cache.ensure_buffers(num_vertices);

    cudaMemset(edge_centralities, 0, num_edges * sizeof(float));

    std::vector<int32_t> h_samples(num_samples);
    cudaMemcpy(h_samples.data(), sample_vertices,
               num_samples * sizeof(int32_t), cudaMemcpyDeviceToHost);

    int actual_concurrent = std::min((int)num_samples, MAX_CONCURRENT);
    std::vector<std::vector<LevelInfo>> all_levels(actual_concurrent);

    for (std::size_t batch_start = 0; batch_start < num_samples; batch_start += actual_concurrent) {
        int batch_size = std::min((std::size_t)actual_concurrent, num_samples - batch_start);

        
        for (int i = 0; i < batch_size; i++) {
            int32_t source = h_samples[batch_start + i];
            cudaMemsetAsync(cache.d_dist[i], 0xFF, num_vertices * sizeof(int32_t), cache.streams[i]);
            cudaMemsetAsync(cache.d_sigma[i], 0, num_vertices * sizeof(float), cache.streams[i]);
            cudaMemsetAsync(cache.d_delta[i], 0, num_vertices * sizeof(float), cache.streams[i]);
            set_source_kernel<<<1, 1, 0, cache.streams[i]>>>(
                cache.d_dist[i], cache.d_sigma[i], cache.d_level_vertices[i], source);

            all_levels[i].clear();
            all_levels[i].reserve(64);
            all_levels[i].push_back({0, 1});
        }

        
        bool active[MAX_CONCURRENT] = {};
        int32_t f_start[MAX_CONCURRENT] = {};
        int32_t f_size[MAX_CONCURRENT] = {};
        int32_t total_stored[MAX_CONCURRENT] = {};

        for (int i = 0; i < batch_size; i++) {
            active[i] = true;
            f_size[i] = 1;
            total_stored[i] = 1;
        }

        bool any_active = true;
        int level = 0;

        while (any_active) {
            
            for (int i = 0; i < batch_size; i++) {
                if (!active[i]) continue;
                cudaMemsetAsync(cache.d_frontier_count[i], 0, sizeof(int32_t), cache.streams[i]);
                if (f_size[i] > 0) {
                    const int block = 128;
                    const int grid = ((int64_t)f_size[i] * 32 + block - 1) / block;
                    bfs_expand_kernel<<<grid, block, 0, cache.streams[i]>>>(
                        d_offsets, d_indices,
                        cache.d_level_vertices[i] + f_start[i], f_size[i],
                        cache.d_level_vertices[i] + total_stored[i], cache.d_frontier_count[i],
                        cache.d_dist[i], cache.d_sigma[i], level
                    );
                }
                cudaMemcpyAsync(cache.h_frontier_count[i], cache.d_frontier_count[i], sizeof(int32_t),
                               cudaMemcpyDeviceToHost, cache.streams[i]);
            }

            
            any_active = false;
            for (int i = 0; i < batch_size; i++) {
                if (!active[i]) continue;
                cudaStreamSynchronize(cache.streams[i]);
                int32_t new_size = *cache.h_frontier_count[i];
                if (new_size > 0) {
                    f_start[i] = total_stored[i];
                    f_size[i] = new_size;
                    total_stored[i] += new_size;
                    all_levels[i].push_back({f_start[i], new_size});
                    any_active = true;
                } else {
                    active[i] = false;
                }
            }
            level++;
        }

        
        for (int i = 0; i < batch_size; i++) {
            for (int d = (int)all_levels[i].size() - 1; d >= 0; d--) {
                auto& lvl = all_levels[i][d];
                if (lvl.size > 0) {
                    const int block = 128;
                    const int grid = ((int64_t)lvl.size * 32 + block - 1) / block;
                    backward_kernel<<<grid, block, 0, cache.streams[i]>>>(
                        d_offsets, d_indices,
                        cache.d_level_vertices[i] + lvl.start, lvl.size,
                        cache.d_dist[i], cache.d_sigma[i], cache.d_delta[i],
                        edge_centralities, d
                    );
                }
            }
        }

        
        for (int i = 0; i < batch_size; i++) {
            cudaStreamSynchronize(cache.streams[i]);
        }
    }

    
    float scale_factor = 0.0f;
    bool has_scale = false;
    if (normalized) {
        scale_factor = static_cast<float>(num_vertices) * (static_cast<float>(num_vertices) - 1.0f);
        has_scale = true;
    } else if (is_symmetric) {
        scale_factor = 2.0f;
        has_scale = true;
    }
    if (has_scale && num_vertices > 1) {
        if (static_cast<std::size_t>(num_samples) < static_cast<std::size_t>(num_vertices)) {
            scale_factor *= static_cast<float>(num_samples) / static_cast<float>(num_vertices);
        }
        const int block = 256;
        const int grid = (num_edges + block - 1) / block;
        scale_kernel<<<grid, block>>>(edge_centralities, num_edges, 1.0f / scale_factor);
        cudaDeviceSynchronize();
    }
}

}  
