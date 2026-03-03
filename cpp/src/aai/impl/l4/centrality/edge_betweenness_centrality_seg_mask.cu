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
#include <algorithm>

namespace aai {

namespace {

static constexpr int MAX_STREAMS = 10;

struct StreamData {
    int* d_distance = nullptr;
    int* d_sigma = nullptr;
    float* d_delta = nullptr;
    int* d_frontier_a = nullptr;
    int* d_frontier_b = nullptr;
    int* d_counter = nullptr;
    int* d_all_frontiers = nullptr;
    cudaStream_t stream = 0;
};

struct Cache : Cacheable {
    StreamData streams[MAX_STREAMS];
    int* h_counters = nullptr;
    int num_streams = 0;
    int max_alloc = 0;

    void ensure(int n, int nstreams) {
        if (n <= max_alloc && nstreams <= num_streams) return;
        free_all();
        max_alloc = n + 1000;
        num_streams = nstreams;

        cudaMallocHost(&h_counters, nstreams * sizeof(int));

        for (int s = 0; s < nstreams; s++) {
            cudaStreamCreate(&streams[s].stream);
            cudaMalloc(&streams[s].d_distance, max_alloc * sizeof(int));
            cudaMalloc(&streams[s].d_sigma, max_alloc * sizeof(int));
            cudaMalloc(&streams[s].d_delta, max_alloc * sizeof(float));
            cudaMalloc(&streams[s].d_frontier_a, max_alloc * sizeof(int));
            cudaMalloc(&streams[s].d_frontier_b, max_alloc * sizeof(int));
            cudaMalloc(&streams[s].d_counter, sizeof(int));
            cudaMalloc(&streams[s].d_all_frontiers, max_alloc * sizeof(int));
        }
    }

    void free_all() {
        for (int s = 0; s < num_streams; s++) {
            if (streams[s].d_distance) cudaFree(streams[s].d_distance);
            if (streams[s].d_sigma) cudaFree(streams[s].d_sigma);
            if (streams[s].d_delta) cudaFree(streams[s].d_delta);
            if (streams[s].d_frontier_a) cudaFree(streams[s].d_frontier_a);
            if (streams[s].d_frontier_b) cudaFree(streams[s].d_frontier_b);
            if (streams[s].d_counter) cudaFree(streams[s].d_counter);
            if (streams[s].d_all_frontiers) cudaFree(streams[s].d_all_frontiers);
            if (streams[s].stream) cudaStreamDestroy(streams[s].stream);
            streams[s] = StreamData{};
        }
        if (h_counters) { cudaFreeHost(h_counters); h_counters = nullptr; }
        num_streams = 0;
        max_alloc = 0;
    }

    ~Cache() override {
        free_all();
    }
};




__global__ void init_and_set_source(int* distance, int* sigma, float* delta, int n, int source) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        distance[idx] = (idx == source) ? 0 : -1;
        sigma[idx] = (idx == source) ? 1 : 0;
        delta[idx] = 0.0f;
    }
}





__global__ void bfs_warp_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int* __restrict__ distance,
    int* __restrict__ sigma,
    const int* __restrict__ frontier,
    int frontier_size,
    int* __restrict__ next_frontier,
    int* __restrict__ next_count,
    int depth
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int u = frontier[warp_id];
    int sigma_u = sigma[u];
    int start = offsets[u];
    int end = offsets[u + 1];
    int next_depth = depth + 1;

    for (int e = start + lane; e < end; e += 32) {
        if (!((edge_mask[e >> 5] >> (e & 31)) & 1)) continue;
        int v = indices[e];
        int old = atomicCAS(&distance[v], -1, next_depth);
        if (old == -1 || old == next_depth) {
            atomicAdd(&sigma[v], sigma_u);
        }
        if (old == -1) {
            int pos = atomicAdd(next_count, 1);
            next_frontier[pos] = v;
        }
    }
}




__global__ void backward_warp_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int* __restrict__ distance,
    const int* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ edge_bc,
    const int* __restrict__ level_vertices,
    int level_size,
    int depth
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= level_size) return;

    int u = level_vertices[warp_id];
    float sigma_u = (float)sigma[u];
    float delta_u = 0.0f;
    int start = offsets[u];
    int end = offsets[u + 1];
    int child_depth = depth + 1;

    for (int e = start + lane; e < end; e += 32) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1) {
            int v = indices[e];
            if (distance[v] == child_depth) {
                float c = sigma_u / (float)sigma[v] * (1.0f + delta[v]);
                delta_u += c;
                atomicAdd(&edge_bc[e], c);
            }
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        delta_u += __shfl_down_sync(0xffffffff, delta_u, offset);
    }
    if (lane == 0) delta[u] = delta_u;
}




__global__ void scale_kernel(float* data, int n, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= factor;
    }
}




static void launch_init(int* distance, int* sigma, float* delta, int n, int source, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    init_and_set_source<<<grid, block, 0, stream>>>(distance, sigma, delta, n, source);
}

static void launch_bfs_warp(
    const int* offsets, const int* indices, const uint32_t* edge_mask,
    int* distance, int* sigma,
    const int* frontier, int frontier_size,
    int* next_frontier, int* next_count,
    int depth, cudaStream_t stream
) {
    if (frontier_size == 0) return;
    int block = 256;
    int grid = (int)(((int64_t)frontier_size * 32 + block - 1) / block);
    bfs_warp_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, edge_mask, distance, sigma,
        frontier, frontier_size, next_frontier, next_count, depth);
}

static void launch_backward_warp(
    const int* offsets, const int* indices, const uint32_t* edge_mask,
    const int* distance, const int* sigma, float* delta, float* edge_bc,
    const int* level_vertices, int level_size, int depth, cudaStream_t stream
) {
    if (level_size == 0) return;
    int block = 256;
    int grid = (int)(((int64_t)level_size * 32 + block - 1) / block);
    backward_warp_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, edge_mask, distance, sigma, delta, edge_bc,
        level_vertices, level_size, depth);
}

static void launch_scale(float* data, int n, float factor, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    scale_kernel<<<grid, block, 0, stream>>>(data, n, factor);
}

static void process_source(Cache& cache, int src_idx, int source, int num_vertices,
                           const int* d_offsets, const int* d_indices, const uint32_t* d_edge_mask,
                           float* d_edge_bc) {
    auto& sd = cache.streams[src_idx];
    cudaStream_t stream = sd.stream;

    launch_init(sd.d_distance, sd.d_sigma, sd.d_delta, num_vertices, source, stream);

    
    cudaMemcpyAsync(sd.d_frontier_a, &source, sizeof(int), cudaMemcpyHostToDevice, stream);
    int frontier_size = 1;

    std::vector<int> level_offsets;
    level_offsets.push_back(0);
    cudaMemcpyAsync(sd.d_all_frontiers, &source, sizeof(int), cudaMemcpyHostToDevice, stream);
    int total = 1;
    level_offsets.push_back(1);

    int depth = 0;
    int* cur = sd.d_frontier_a;
    int* nxt = sd.d_frontier_b;

    while (frontier_size > 0) {
        cudaMemsetAsync(sd.d_counter, 0, sizeof(int), stream);

        launch_bfs_warp(d_offsets, d_indices, d_edge_mask,
                       sd.d_distance, sd.d_sigma,
                       cur, frontier_size, nxt, sd.d_counter, depth, stream);

        cudaMemcpyAsync(&cache.h_counters[src_idx], sd.d_counter, sizeof(int),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        frontier_size = cache.h_counters[src_idx];

        if (frontier_size > 0) {
            cudaMemcpyAsync(sd.d_all_frontiers + total, nxt,
                           frontier_size * sizeof(int), cudaMemcpyDeviceToDevice, stream);
            total += frontier_size;
            level_offsets.push_back(total);
        }

        std::swap(cur, nxt);
        depth++;
    }

    int num_levels = (int)level_offsets.size() - 1;

    
    for (int d = num_levels - 2; d >= 0; d--) {
        int lstart = level_offsets[d];
        int lsize = level_offsets[d + 1] - level_offsets[d];
        launch_backward_warp(d_offsets, d_indices, d_edge_mask,
                            sd.d_distance, sd.d_sigma, sd.d_delta, d_edge_bc,
                            sd.d_all_frontiers + lstart, lsize, d, stream);
    }
}

}  

void edge_betweenness_centrality_seg_mask(const graph32_t& graph,
                                           float* edge_centralities,
                                           bool normalized,
                                           const int32_t* sample_vertices,
                                           std::size_t num_samples) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int num_vertices = graph.number_of_vertices;
    int num_edges = graph.number_of_edges;
    bool is_symmetric = graph.is_symmetric;
    const int* d_offsets = graph.offsets;
    const int* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    
    std::vector<int> sources;
    if (sample_vertices != nullptr && num_samples > 0) {
        sources.resize(num_samples);
        cudaMemcpy(sources.data(), sample_vertices,
                   num_samples * sizeof(int), cudaMemcpyDeviceToHost);
    } else {
        sources.resize(num_vertices);
        for (int i = 0; i < num_vertices; i++) sources[i] = i;
    }

    int nsrc = (int)sources.size();
    int nstreams = (nsrc <= MAX_STREAMS) ? nsrc : MAX_STREAMS;
    
    if (nsrc > MAX_STREAMS) nstreams = 4;

    cache.ensure(num_vertices, nstreams);

    
    cudaMemset(edge_centralities, 0, num_edges * sizeof(float));

    if (nsrc <= nstreams) {
        
        for (int si = 0; si < nsrc; si++) {
            process_source(cache, si, sources[si], num_vertices,
                          d_offsets, d_indices, d_edge_mask, edge_centralities);
        }
        
        for (int si = 0; si < nsrc; si++) {
            cudaStreamSynchronize(cache.streams[si].stream);
        }
    } else {
        
        for (int batch_start = 0; batch_start < nsrc; batch_start += nstreams) {
            int batch_end = (batch_start + nstreams < nsrc) ? batch_start + nstreams : nsrc;
            int batch_size = batch_end - batch_start;

            for (int si = 0; si < batch_size; si++) {
                process_source(cache, si, sources[batch_start + si], num_vertices,
                              d_offsets, d_indices, d_edge_mask, edge_centralities);
            }
            for (int si = 0; si < batch_size; si++) {
                cudaStreamSynchronize(cache.streams[si].stream);
            }
        }
    }

    
    size_t num_sources_total = sources.size();
    bool has_scale = false;
    float scale_factor = 1.0f;

    if (normalized) {
        float n = (float)num_vertices;
        scale_factor = n * (n - 1.0f);
        has_scale = true;
    } else if (is_symmetric) {
        scale_factor = 2.0f;
        has_scale = true;
    }

    if (has_scale && num_vertices > 1) {
        if ((int)num_sources_total < num_vertices) {
            scale_factor *= (float)num_sources_total / (float)num_vertices;
        }
        float inv_factor = 1.0f / scale_factor;
        launch_scale(edge_centralities, num_edges, inv_factor, 0);
    }

    cudaDeviceSynchronize();
}

}  
