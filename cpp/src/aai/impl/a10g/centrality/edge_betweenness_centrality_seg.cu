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
#include <cstdio>
#include <vector>
#include <algorithm>

namespace aai {

namespace {

static const int MAX_STREAMS = 10;

struct PerStreamData {
    int32_t* dist;
    int32_t* sigma;
    float* delta;
    int32_t* frontier;
    int32_t* d_counter;
};

struct SrcState {
    std::vector<int32_t> level_offsets;
    int32_t num_levels;
    int32_t frontier_start, frontier_end, current_dist;
    bool bfs_done;
    bool backward_done;
};

struct Cache : Cacheable {
    PerStreamData per_stream[MAX_STREAMS];
    cudaStream_t streams[MAX_STREAMS];
    int32_t* h_counters = nullptr;
    int32_t alloc_vertices = 0;

    Cache() {
        for (int i = 0; i < MAX_STREAMS; i++) {
            per_stream[i] = {nullptr, nullptr, nullptr, nullptr, nullptr};
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
            cudaMalloc(&per_stream[i].d_counter, sizeof(int32_t));
        }
        cudaHostAlloc(&h_counters, MAX_STREAMS * sizeof(int32_t), cudaHostAllocDefault);
    }

    void ensure_buffers(int32_t nv) {
        if (nv <= alloc_vertices) return;
        for (int i = 0; i < MAX_STREAMS; i++) {
            if (per_stream[i].dist) { cudaFree(per_stream[i].dist); per_stream[i].dist = nullptr; }
            if (per_stream[i].sigma) { cudaFree(per_stream[i].sigma); per_stream[i].sigma = nullptr; }
            if (per_stream[i].delta) { cudaFree(per_stream[i].delta); per_stream[i].delta = nullptr; }
            if (per_stream[i].frontier) { cudaFree(per_stream[i].frontier); per_stream[i].frontier = nullptr; }
        }
        alloc_vertices = nv;
        for (int i = 0; i < MAX_STREAMS; i++) {
            cudaMalloc(&per_stream[i].dist, alloc_vertices * sizeof(int32_t));
            cudaMalloc(&per_stream[i].sigma, alloc_vertices * sizeof(int32_t));
            cudaMalloc(&per_stream[i].delta, alloc_vertices * sizeof(float));
            cudaMalloc(&per_stream[i].frontier, alloc_vertices * sizeof(int32_t));
        }
    }

    ~Cache() override {
        for (int i = 0; i < MAX_STREAMS; i++) {
            cudaStreamDestroy(streams[i]);
            if (per_stream[i].dist) cudaFree(per_stream[i].dist);
            if (per_stream[i].sigma) cudaFree(per_stream[i].sigma);
            if (per_stream[i].delta) cudaFree(per_stream[i].delta);
            if (per_stream[i].frontier) cudaFree(per_stream[i].frontier);
            if (per_stream[i].d_counter) cudaFree(per_stream[i].d_counter);
        }
        if (h_counters) cudaFreeHost(h_counters);
    }
};



__global__ void reset_visited_kernel(
    const int32_t* __restrict__ frontier, int32_t count,
    int32_t* __restrict__ dist, int32_t* __restrict__ sigma, float* __restrict__ delta
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        int32_t v = frontier[tid];
        dist[v] = -1;
        sigma[v] = 0;
        delta[v] = 0.0f;
    }
}

__global__ void init_arrays_kernel(int32_t* dist, int32_t* sigma, float* delta, int32_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        dist[tid] = -1;
        sigma[tid] = 0;
        delta[tid] = 0.0f;
    }
}

__global__ void set_source_kernel(int32_t* dist, int32_t* sigma, int32_t* frontier, int32_t source) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        dist[source] = 0;
        sigma[source] = 1;
        frontier[0] = source;
    }
}

__global__ void bfs_expand_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ frontier_in,
    int32_t frontier_in_size,
    int32_t* __restrict__ dist,
    int32_t* __restrict__ sigma,
    int32_t current_dist,
    int32_t* __restrict__ frontier_out,
    int32_t* __restrict__ frontier_out_count
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id = threadIdx.x & 31;

    if (warp_id >= frontier_in_size) return;

    int32_t u = frontier_in[warp_id];
    int32_t sigma_u = sigma[u];
    int32_t start = __ldg(&offsets[u]);
    int32_t end = __ldg(&offsets[u + 1]);
    int32_t next_dist = current_dist + 1;

    for (int32_t i = start + lane_id; i < end; i += 32) {
        int32_t v = __ldg(&indices[i]);

        int32_t d_v = dist[v];
        if (d_v >= 0 && d_v < next_dist) continue;

        int32_t old_dist = atomicCAS(&dist[v], -1, next_dist);
        if (old_dist == -1) {
            int32_t pos = atomicAdd(frontier_out_count, 1);
            frontier_out[pos] = v;
        }
        if (old_dist == -1 || old_dist == next_dist) {
            atomicAdd(&sigma[v], sigma_u);
        }
    }
}

__global__ void backward_pass_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    const int32_t* __restrict__ dist,
    const int32_t* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ edge_bc,
    int32_t d
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t w = frontier[warp_id];
    float sigma_w = (float)sigma[w];
    int32_t start = __ldg(&offsets[w]);
    int32_t end = __ldg(&offsets[w + 1]);

    float local_delta = 0.0f;

    for (int32_t i = start + lane_id; i < end; i += 32) {
        int32_t v = __ldg(&indices[i]);
        if (__ldg(&dist[v]) == d) {
            float sigma_v = (float)__ldg(&sigma[v]);
            float delta_v = delta[v];
            float contrib = (sigma_w / sigma_v) * (1.0f + delta_v);
            atomicAdd(&edge_bc[i], contrib);
            local_delta += contrib;
        }
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_delta += __shfl_xor_sync(0xffffffff, local_delta, mask);
    }

    if (lane_id == 0) {
        delta[w] = local_delta;
    }
}

__global__ void scale_edges_kernel(float* edge_bc, int32_t num_edges, float inv_scale) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        edge_bc[tid] *= inv_scale;
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

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    bool is_symmetric = graph.is_symmetric;

    cache.ensure_buffers(num_vertices);

    const int BLOCK = 256;

    cudaMemset(edge_centralities, 0, num_edges * sizeof(float));

    std::vector<int32_t> h_samples(num_samples);
    cudaMemcpy(h_samples.data(), sample_vertices, num_samples * sizeof(int32_t), cudaMemcpyDeviceToHost);

    int init_blocks = (num_vertices + BLOCK - 1) / BLOCK;
    int nb;

    int num_streams = (int)std::min((int64_t)MAX_STREAMS, (int64_t)num_samples);

    std::vector<cudaEvent_t> events(num_streams);
    for (int i = 0; i < num_streams; i++)
        cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);

    for (std::size_t batch_start = 0; batch_start < num_samples; batch_start += num_streams) {
        int batch_size = (int)std::min((int64_t)num_streams, (int64_t)(num_samples - batch_start));

        std::vector<SrcState> states(batch_size);

        for (int i = 0; i < batch_size; i++) {
            int32_t source = h_samples[batch_start + i];
            cudaStream_t s = cache.streams[i];
            PerStreamData& pd = cache.per_stream[i];

            init_arrays_kernel<<<init_blocks, BLOCK, 0, s>>>(pd.dist, pd.sigma, pd.delta, num_vertices);

            set_source_kernel<<<1, 1, 0, s>>>(pd.dist, pd.sigma, pd.frontier, source);

            states[i].level_offsets.clear();
            states[i].level_offsets.reserve(2048);
            states[i].level_offsets.push_back(0);
            states[i].level_offsets.push_back(1);
            states[i].num_levels = 1;
            states[i].frontier_start = 0;
            states[i].frontier_end = 1;
            states[i].current_dist = 0;
            states[i].bfs_done = false;
            states[i].backward_done = false;

            cudaMemsetAsync(pd.d_counter, 0, sizeof(int32_t), s);
            nb = (32 + BLOCK - 1) / BLOCK;
            bfs_expand_kernel<<<nb, BLOCK, 0, s>>>(
                offsets, indices,
                pd.frontier, 1,
                pd.dist, pd.sigma, 0,
                pd.frontier + 1, pd.d_counter
            );
            cudaMemcpyAsync(&cache.h_counters[i], pd.d_counter, sizeof(int32_t),
                           cudaMemcpyDeviceToHost, s);
            cudaEventRecord(events[i], s);
        }

        bool any_bfs_active = true;
        while (any_bfs_active) {
            any_bfs_active = false;
            for (int i = 0; i < batch_size; i++) {
                if (states[i].bfs_done) continue;

                if (cudaEventQuery(events[i]) != cudaSuccess) {
                    any_bfs_active = true;
                    continue;
                }

                int32_t count = cache.h_counters[i];

                if (count == 0) {
                    states[i].bfs_done = true;

                    PerStreamData& pd = cache.per_stream[i];
                    cudaStream_t s = cache.streams[i];

                    for (int d = states[i].current_dist; d >= 1; d--) {
                        int32_t fstart = states[i].level_offsets[d - 1];
                        int32_t fend = states[i].level_offsets[d];
                        int32_t fsize = fend - fstart;

                        if (fsize > 0) {
                            int warps = fsize;
                            nb = (warps * 32 + BLOCK - 1) / BLOCK;
                            backward_pass_kernel<<<nb, BLOCK, 0, s>>>(
                                offsets, indices,
                                pd.frontier + fstart, fsize,
                                pd.dist, pd.sigma, pd.delta, edge_centralities, d
                            );
                        }
                    }
                    continue;
                }

                any_bfs_active = true;
                states[i].frontier_start = states[i].frontier_end;
                states[i].frontier_end += count;
                states[i].current_dist++;
                states[i].num_levels++;
                states[i].level_offsets.push_back(states[i].frontier_end);

                PerStreamData& pd = cache.per_stream[i];
                cudaStream_t s = cache.streams[i];
                int32_t fsize = count;

                cudaMemsetAsync(pd.d_counter, 0, sizeof(int32_t), s);
                int warps = fsize;
                nb = (warps * 32 + BLOCK - 1) / BLOCK;
                bfs_expand_kernel<<<nb, BLOCK, 0, s>>>(
                    offsets, indices,
                    pd.frontier + states[i].frontier_start, fsize,
                    pd.dist, pd.sigma,
                    states[i].current_dist,
                    pd.frontier + states[i].frontier_end,
                    pd.d_counter
                );
                cudaMemcpyAsync(&cache.h_counters[i], pd.d_counter, sizeof(int32_t),
                               cudaMemcpyDeviceToHost, s);
                cudaEventRecord(events[i], s);
            }
        }

        for (int i = 0; i < batch_size; i++)
            cudaStreamSynchronize(cache.streams[i]);
    }

    for (int i = 0; i < num_streams; i++)
        cudaEventDestroy(events[i]);

    float scale_factor = 0.0f;
    bool apply_scale = false;
    float n = (float)num_vertices;

    if (normalized) {
        scale_factor = n * (n - 1.0f);
        apply_scale = true;
    } else if (is_symmetric) {
        scale_factor = 2.0f;
        apply_scale = true;
    }

    if (apply_scale && num_vertices > 1) {
        if ((int64_t)num_samples < (int64_t)num_vertices)
            scale_factor *= (float)num_samples / n;

        float inv_scale = 1.0f / scale_factor;
        nb = (num_edges + BLOCK - 1) / BLOCK;
        scale_edges_kernel<<<nb, BLOCK>>>(edge_centralities, num_edges, inv_scale);
        cudaDeviceSynchronize();
    }
}

}  
