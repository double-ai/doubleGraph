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
#include <algorithm>
#include <vector>

namespace aai {

namespace {

static constexpr int MAX_STREAMS = 64;
static constexpr int BLK = 256;






__global__ void kern_bfs_advance(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ counter,
    int32_t* __restrict__ dist,
    float* __restrict__ sigma,
    int32_t next_level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 0x1f;

    if (warp_id >= frontier_size) return;

    int32_t v = frontier[warp_id];
    int32_t row_start = __ldg(&offsets[v]);
    int32_t row_end = __ldg(&offsets[v + 1]);
    float sigma_v = sigma[v];

    for (int32_t e = row_start + lane_id; e < row_end; e += 32) {
        int32_t w = __ldg(&indices[e]);

        
        int32_t d = dist[w];

        if (d >= 0) {
            
            if (d == next_level) {
                
                atomicAdd(&sigma[w], sigma_v);
            }
            
            continue;
        }

        
        int32_t old = atomicCAS(&dist[w], -1, next_level);
        if (old == -1) {
            
            int32_t pos = atomicAdd(counter, 1);
            next_frontier[pos] = w;
            atomicAdd(&sigma[w], sigma_v);
        } else if (old == next_level) {
            
            atomicAdd(&sigma[w], sigma_v);
        }
    }
}


__global__ void kern_backward(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    const int32_t* __restrict__ dist,
    const float* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ bc,
    int32_t level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 0x1f;

    if (warp_id >= frontier_size) return;

    int32_t v = frontier[warp_id];
    int32_t row_start = __ldg(&offsets[v]);
    int32_t row_end = __ldg(&offsets[v + 1]);
    float sigma_v = sigma[v];
    int32_t child_level = level + 1;

    float dep = 0.0f;
    for (int32_t e = row_start + lane_id; e < row_end; e += 32) {
        int32_t w = __ldg(&indices[e]);
        if (__ldg(&dist[w]) == child_level) {
            dep += (sigma_v / __ldg(&sigma[w])) * (1.0f + delta[w]);
        }
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        dep += __shfl_down_sync(0xffffffff, dep, s);
    }

    if (lane_id == 0) {
        delta[v] = dep;
        atomicAdd(&bc[v], dep);
    }
}

__global__ void kern_endpoints(
    const int32_t* __restrict__ frontier_buf,
    int32_t total_visited,
    float* __restrict__ bc,
    int32_t source
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= 1 && tid < total_visited)
        atomicAdd(&bc[frontier_buf[tid]], 1.0f);
    if (tid == 0 && total_visited > 1)
        atomicAdd(&bc[source], (float)(total_visited - 1));
}

__global__ void kern_zero(float* arr, int32_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) arr[tid] = 0.0f;
}

__global__ void kern_mark_sources(bool* is_source, const int32_t* samples, int32_t num_samples) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_samples) is_source[samples[tid]] = true;
}

__global__ void kern_norm_uniform(float* bc, int32_t n, float inv_scale) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) bc[tid] *= inv_scale;
}

__global__ void kern_norm_split(float* bc, int32_t n, const bool* is_source,
                                 float inv_src, float inv_non) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) bc[tid] *= (is_source[tid] ? inv_src : inv_non);
}

__global__ void kern_set_source(int32_t* dist, float* sigma, int32_t* frontier, int32_t source) {
    dist[source] = 0;
    sigma[source] = 1.0f;
    frontier[0] = source;
}





struct Cache : Cacheable {
    cudaStream_t streams[MAX_STREAMS];
    int32_t* h_counters = nullptr;

    int32_t* dist_buf = nullptr;
    float* sigma_buf = nullptr;
    float* delta_buf = nullptr;
    int32_t* frontier_buf = nullptr;
    int32_t* counter_buf = nullptr;
    uint8_t* is_source_buf = nullptr;

    int64_t dist_capacity = 0;
    int64_t sigma_capacity = 0;
    int64_t delta_capacity = 0;
    int64_t frontier_capacity = 0;
    int64_t counter_capacity = 0;
    int64_t is_source_capacity = 0;

    Cache() {
        for (int i = 0; i < MAX_STREAMS; i++)
            cudaStreamCreate(&streams[i]);
        cudaMallocHost(&h_counters, MAX_STREAMS * sizeof(int32_t));
    }

    void ensure(int64_t total, int ns, int32_t V) {
        if (dist_capacity < total) {
            if (dist_buf) cudaFree(dist_buf);
            cudaMalloc(&dist_buf, total * sizeof(int32_t));
            dist_capacity = total;
        }
        if (sigma_capacity < total) {
            if (sigma_buf) cudaFree(sigma_buf);
            cudaMalloc(&sigma_buf, total * sizeof(float));
            sigma_capacity = total;
        }
        if (delta_capacity < total) {
            if (delta_buf) cudaFree(delta_buf);
            cudaMalloc(&delta_buf, total * sizeof(float));
            delta_capacity = total;
        }
        if (frontier_capacity < total) {
            if (frontier_buf) cudaFree(frontier_buf);
            cudaMalloc(&frontier_buf, total * sizeof(int32_t));
            frontier_capacity = total;
        }
        if (counter_capacity < (int64_t)ns) {
            if (counter_buf) cudaFree(counter_buf);
            cudaMalloc(&counter_buf, ns * sizeof(int32_t));
            counter_capacity = ns;
        }
        if (is_source_capacity < (int64_t)V) {
            if (is_source_buf) cudaFree(is_source_buf);
            cudaMalloc(&is_source_buf, V * sizeof(uint8_t));
            is_source_capacity = V;
        }
    }

    ~Cache() override {
        for (int i = 0; i < MAX_STREAMS; i++)
            cudaStreamDestroy(streams[i]);
        if (h_counters) cudaFreeHost(h_counters);
        if (dist_buf) cudaFree(dist_buf);
        if (sigma_buf) cudaFree(sigma_buf);
        if (delta_buf) cudaFree(delta_buf);
        if (frontier_buf) cudaFree(frontier_buf);
        if (counter_buf) cudaFree(counter_buf);
        if (is_source_buf) cudaFree(is_source_buf);
    }
};

}  

void betweenness_centrality(const graph32_t& graph,
                            float* centralities,
                            bool normalized,
                            bool include_endpoints,
                            const int32_t* sample_vertices,
                            std::size_t num_samples) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t V = graph.number_of_vertices;
    bool is_symmetric = graph.is_symmetric;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int64_t K = (int64_t)num_samples;

    
    kern_zero<<<(V + BLK - 1) / BLK, BLK>>>(centralities, V);

    if (V <= 1 || K == 0) { cudaDeviceSynchronize(); return; }

    int ns = (int)std::min((int64_t)MAX_STREAMS, K);
    int64_t total = (int64_t)ns * V;

    cache.ensure(total, ns, V);

    int32_t* dist_base = cache.dist_buf;
    float* sigma_base = cache.sigma_buf;
    float* delta_base = cache.delta_buf;
    int32_t* frontier_base = cache.frontier_buf;
    int32_t* counter_base = cache.counter_buf;

    std::vector<int32_t> h_samples(K);
    cudaMemcpy(h_samples.data(), sample_vertices, K * sizeof(int32_t), cudaMemcpyDeviceToHost);

    struct LevelInfo { int start; int count; };

    for (int64_t batch_start = 0; batch_start < K; batch_start += ns) {
        int batch_size = (int)std::min((int64_t)ns, K - batch_start);

        std::vector<std::vector<LevelInfo>> all_levels(batch_size);
        std::vector<bool> done(batch_size, false);

        
        for (int s = 0; s < batch_size; s++) {
            int32_t* d = dist_base + (int64_t)s * V;
            float* sig = sigma_base + (int64_t)s * V;
            float* del = delta_base + (int64_t)s * V;
            int32_t* fr = frontier_base + (int64_t)s * V;
            int32_t source = h_samples[batch_start + s];

            
            cudaMemsetAsync(d, 0xFF, V * sizeof(int32_t), cache.streams[s]);
            cudaMemsetAsync(sig, 0, V * sizeof(float), cache.streams[s]);
            cudaMemsetAsync(del, 0, V * sizeof(float), cache.streams[s]);
            kern_set_source<<<1, 1, 0, cache.streams[s]>>>(d, sig, fr, source);

            all_levels[s].push_back({0, 1});
        }

        
        int current_level = 0;
        bool any_active = true;

        while (any_active) {
            for (int s = 0; s < batch_size; s++) {
                if (done[s]) continue;

                int32_t* d = dist_base + (int64_t)s * V;
                float* sig = sigma_base + (int64_t)s * V;
                int32_t* fr = frontier_base + (int64_t)s * V;
                int32_t* cnt = counter_base + s;

                auto& lvls = all_levels[s];
                int lo = lvls[current_level].start;
                int sz = lvls[current_level].count;

                cudaMemsetAsync(cnt, 0, sizeof(int32_t), cache.streams[s]);
                int grid = (int)(((int64_t)sz * 32 + BLK - 1) / BLK);
                if (grid > 0)
                    kern_bfs_advance<<<grid, BLK, 0, cache.streams[s]>>>(
                        offsets, indices,
                        fr + lo, sz,
                        fr + lo + sz, cnt,
                        d, sig,
                        current_level + 1);
                cudaMemcpyAsync(&cache.h_counters[s], cnt, sizeof(int32_t),
                               cudaMemcpyDeviceToHost, cache.streams[s]);
            }

            any_active = false;
            for (int s = 0; s < batch_size; s++) {
                if (done[s]) continue;
                cudaStreamSynchronize(cache.streams[s]);

                int next_count = cache.h_counters[s];
                auto& lvls = all_levels[s];
                int next_start = lvls[current_level].start + lvls[current_level].count;
                lvls.push_back({next_start, next_count});

                if (next_count == 0) done[s] = true;
                else any_active = true;
            }
            current_level++;
        }

        
        for (int s = 0; s < batch_size; s++) {
            int32_t* d = dist_base + (int64_t)s * V;
            float* sig = sigma_base + (int64_t)s * V;
            float* del = delta_base + (int64_t)s * V;
            int32_t* fr = frontier_base + (int64_t)s * V;
            int32_t source = h_samples[batch_start + s];

            auto& lvls = all_levels[s];
            int max_level = (int)lvls.size() - 2;

            for (int L = max_level - 1; L >= 1; L--) {
                if (lvls[L].count > 0) {
                    int grid = (int)(((int64_t)lvls[L].count * 32 + BLK - 1) / BLK);
                    if (grid > 0)
                        kern_backward<<<grid, BLK, 0, cache.streams[s]>>>(
                            offsets, indices,
                            fr + lvls[L].start, lvls[L].count,
                            d, sig, del, centralities, L);
                }
            }

            if (include_endpoints) {
                int total_visited = lvls.back().start;
                if (total_visited > 1) {
                    int grid = (total_visited + BLK - 1) / BLK;
                    if (grid > 0)
                        kern_endpoints<<<grid, BLK, 0, cache.streams[s]>>>(
                            fr, total_visited, centralities, source);
                }
            }
        }
    }

    for (int i = 0; i < ns; i++)
        cudaStreamSynchronize(cache.streams[i]);

    
    double n = (double)V, k = (double)K;
    double adj_d = include_endpoints ? n : (n - 1.0);
    bool all_srcs = ((int64_t)K == (int64_t)(include_endpoints ? V : V - 1)) || include_endpoints;

    if (all_srcs) {
        double scale;
        if (normalized) scale = k * (adj_d - 1.0);
        else if (is_symmetric) scale = k * 2.0 / adj_d;
        else scale = k / adj_d;
        if (scale != 0.0)
            kern_norm_uniform<<<(V + BLK - 1) / BLK, BLK>>>(centralities, V, (float)(1.0 / scale));
    } else {
        cudaMemset(cache.is_source_buf, 0, V);
        kern_mark_sources<<<((int32_t)K + BLK - 1) / BLK, BLK>>>(
            (bool*)cache.is_source_buf, sample_vertices, (int32_t)K);

        double scale_src, scale_non;
        if (normalized) {
            scale_non = k * (adj_d - 1.0);
            scale_src = (k - 1.0) * (adj_d - 1.0);
        } else {
            scale_non = k / adj_d;
            scale_src = (k - 1.0) / adj_d;
            if (is_symmetric) { scale_non *= 2.0; scale_src *= 2.0; }
        }

        float inv_src = (float)(1.0 / scale_src);
        float inv_non = (float)(1.0 / scale_non);
        kern_norm_split<<<(V + BLK - 1) / BLK, BLK>>>(
            centralities, V, (bool*)cache.is_source_buf, inv_src, inv_non);
    }
    cudaDeviceSynchronize();
}

}  
