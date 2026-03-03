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
#include <cmath>
#include <vector>
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* d_dist = nullptr;
    float* d_sigma = nullptr;
    float* d_delta = nullptr;
    int32_t* d_stack = nullptr;
    int32_t* d_next_cnt = nullptr;
    bool* d_is_src = nullptr;
    int32_t* h_next_cnt = nullptr;
    int32_t alloc_n = 0;

    void ensure(int32_t n) {
        if (n <= alloc_n) return;
        free_all();
        alloc_n = n;
        cudaMalloc(&d_dist, (size_t)n * sizeof(int32_t));
        cudaMalloc(&d_sigma, (size_t)n * sizeof(float));
        cudaMalloc(&d_delta, (size_t)n * sizeof(float));
        cudaMalloc(&d_stack, (size_t)n * sizeof(int32_t));
        cudaMalloc(&d_next_cnt, sizeof(int32_t));
        cudaMalloc(&d_is_src, (size_t)n * sizeof(bool));
        cudaHostAlloc(&h_next_cnt, sizeof(int32_t), cudaHostAllocDefault);
    }

    void free_all() {
        if (d_dist) { cudaFree(d_dist); d_dist = nullptr; }
        if (d_sigma) { cudaFree(d_sigma); d_sigma = nullptr; }
        if (d_delta) { cudaFree(d_delta); d_delta = nullptr; }
        if (d_stack) { cudaFree(d_stack); d_stack = nullptr; }
        if (d_next_cnt) { cudaFree(d_next_cnt); d_next_cnt = nullptr; }
        if (d_is_src) { cudaFree(d_is_src); d_is_src = nullptr; }
        if (h_next_cnt) { cudaFreeHost(h_next_cnt); h_next_cnt = nullptr; }
        alloc_n = 0;
    }

    ~Cache() override { free_all(); }
};

__device__ __forceinline__ bool edge_active(const uint32_t* mask, int32_t idx) {
    return (mask[idx >> 5] >> (idx & 31)) & 1;
}

__global__ void reset_kernel(
    int32_t* __restrict__ dist,
    float* __restrict__ sigma,
    float* __restrict__ delta,
    int32_t n,
    int32_t src
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dist[i] = (i == src) ? 0 : -1;
        sigma[i] = (i == src) ? 1.0f : 0.0f;
        delta[i] = 0.0f;
    }
}

__global__ void bfs_forward(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ mask,
    int32_t* __restrict__ dist,
    float* __restrict__ sigma,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_cnt,
    int32_t level,
    int32_t fsize
) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int nwarps = (blockDim.x * gridDim.x) >> 5;
    int next_level = level + 1;

    for (int fi = wid; fi < fsize; fi += nwarps) {
        int32_t v = frontier[fi];
        float sv = sigma[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        for (int32_t e = start + lane; e < end; e += 32) {
            if (!edge_active(mask, e)) continue;

            int32_t w = indices[e];
            int32_t old = atomicCAS(&dist[w], -1, next_level);

            if (old != -1 && old != next_level) continue;

            atomicAdd(&sigma[w], sv);

            if (old == -1) {
                int32_t pos = atomicAdd(next_cnt, 1);
                next_frontier[pos] = w;
            }
        }
    }
}

__global__ void bfs_backward(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ mask,
    const int32_t* __restrict__ dist,
    const float* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ bc,
    const int32_t* __restrict__ lvl_verts,
    int32_t lvl_size,
    int32_t level,
    int32_t src
) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int nwarps = (blockDim.x * gridDim.x) >> 5;

    for (int i = wid; i < lvl_size; i += nwarps) {
        int32_t v = lvl_verts[i];
        float sv = sigma[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float dv = 0.0f;

        for (int32_t e = start + lane; e < end; e += 32) {
            if (!edge_active(mask, e)) continue;
            int32_t w = indices[e];
            if (dist[w] == level + 1) {
                dv += (sv / sigma[w]) * (1.0f + delta[w]);
            }
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dv += __shfl_down_sync(0xFFFFFFFF, dv, off);

        if (lane == 0) {
            delta[v] = dv;
            if (v != src)
                bc[v] += dv;
        }
    }
}

__global__ void endpoint_kernel(
    float* __restrict__ bc,
    const int32_t* __restrict__ dist,
    int32_t n,
    int32_t src
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cnt = 0;

    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (dist[i] >= 0 && i != src) {
            cnt++;
            bc[i] += 1.0f;
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        cnt += __shfl_down_sync(0xFFFFFFFF, cnt, off);

    if ((threadIdx.x & 31) == 0 && cnt > 0)
        atomicAdd(&bc[src], (float)cnt);
}

__global__ void mark_sources(
    bool* __restrict__ is_src,
    const int32_t* __restrict__ samples,
    int32_t k
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < k; i += blockDim.x * gridDim.x)
        is_src[samples[i]] = true;
}

__global__ void normalize(
    float* __restrict__ bc,
    const bool* __restrict__ is_src,
    int32_t n,
    float inv_scale_src,
    float inv_scale_nonsrc
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        bc[i] *= is_src[i] ? inv_scale_src : inv_scale_nonsrc;
    }
}

__global__ void normalize_uniform(
    float* __restrict__ bc,
    int32_t n,
    float inv_scale
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        bc[i] *= inv_scale;
}

}  

void betweenness_centrality_seg_mask(const graph32_t& graph,
                                      float* centralities,
                                      bool normalized,
                                      bool include_endpoints,
                                      const int32_t* sample_vertices,
                                      std::size_t num_samples) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t n = graph.number_of_vertices;
    bool is_sym = graph.is_symmetric;
    const uint32_t* d_mask = graph.edge_mask;

    int32_t k = static_cast<int32_t>(num_samples);
    cudaStream_t stream = 0;

    cache.ensure(n);

    float* d_bc = centralities;
    cudaMemsetAsync(d_bc, 0, (size_t)n * sizeof(float), stream);

    
    std::vector<int32_t> h_samples(k);
    cudaMemcpy(h_samples.data(), sample_vertices, k * sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    for (int si = 0; si < k; si++) {
        int32_t src = h_samples[si];

        
        {
            int B = 256, G = std::min((n + B - 1) / B, 1024);
            reset_kernel<<<G, B, 0, stream>>>(cache.d_dist, cache.d_sigma, cache.d_delta, n, src);
        }

        
        cudaMemcpyAsync(cache.d_stack, &src, sizeof(int32_t), cudaMemcpyHostToDevice, stream);

        
        std::vector<int32_t> lvl_off;
        lvl_off.push_back(0);
        lvl_off.push_back(1);

        int32_t cur_level = 0;
        int32_t stack_top = 1;

        
        while (true) {
            int32_t fsize = lvl_off[cur_level + 1] - lvl_off[cur_level];
            if (fsize == 0) break;

            cudaMemsetAsync(cache.d_next_cnt, 0, sizeof(int32_t), stream);

            {
                int B = 256;
                int warps = fsize;
                int64_t threads = (int64_t)warps * 32;
                int G = std::min((int)((threads + B - 1) / B), 2048);
                bfs_forward<<<G, B, 0, stream>>>(
                    d_off, d_idx, d_mask, cache.d_dist, cache.d_sigma,
                    cache.d_stack + lvl_off[cur_level],
                    cache.d_stack + lvl_off[cur_level + 1],
                    cache.d_next_cnt, cur_level, fsize
                );
            }

            cudaMemcpyAsync(cache.h_next_cnt, cache.d_next_cnt, sizeof(int32_t),
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            int32_t nxt = *cache.h_next_cnt;
            stack_top += nxt;
            lvl_off.push_back(stack_top);
            cur_level++;

            if (nxt == 0) break;
        }

        
        if (include_endpoints) {
            int B = 256, G = std::min((n + B - 1) / B, 1024);
            endpoint_kernel<<<G, B, 0, stream>>>(d_bc, cache.d_dist, n, src);
        }

        
        int32_t max_lvl = cur_level;
        for (int l = max_lvl - 1; l >= 0; l--) {
            int32_t ls = lvl_off[l + 1] - lvl_off[l];
            if (ls > 0) {
                int B = 256;
                int warps = ls;
                int64_t threads = (int64_t)warps * 32;
                int G = std::min((int)((threads + B - 1) / B), 2048);
                bfs_backward<<<G, B, 0, stream>>>(
                    d_off, d_idx, d_mask, cache.d_dist, cache.d_sigma, cache.d_delta,
                    d_bc, cache.d_stack + lvl_off[l], ls, l, src
                );
            }
        }
    }

    
    double adj = include_endpoints ? (double)n : (double)(n - 1);
    bool all_srcs = (k == (int32_t)(include_endpoints ? n : (n - 1))) || include_endpoints;

    if (adj > 1.0) {
        if (all_srcs) {
            double scale;
            if (normalized) {
                scale = (double)k * (adj - 1.0);
            } else if (is_sym) {
                scale = (double)k * 2.0 / adj;
            } else {
                scale = (double)k / adj;
            }
            if (scale != 0.0) {
                int B = 256, G = (n + B - 1) / B;
                normalize_uniform<<<G, B, 0, stream>>>(d_bc, n, (float)(1.0 / scale));
            }
        } else if (normalized) {
            double s_ns = (double)k * (adj - 1.0);
            double s_s = (double)(k - 1) * (adj - 1.0);
            float inv_ns = (float)(1.0 / s_ns);
            float inv_s = (float)(1.0 / s_s);
            cudaMemsetAsync(cache.d_is_src, 0, (size_t)n * sizeof(bool), stream);
            {
                int B = 256, G = (k + B - 1) / B;
                if (G < 1) G = 1;
                mark_sources<<<G, B, 0, stream>>>(cache.d_is_src, sample_vertices, k);
            }
            {
                int B = 256, G = (n + B - 1) / B;
                normalize<<<G, B, 0, stream>>>(d_bc, cache.d_is_src, n, inv_s, inv_ns);
            }
        } else {
            double s_ns = (double)k / adj;
            double s_s = (double)(k - 1) / adj;
            if (is_sym) { s_ns *= 2.0; s_s *= 2.0; }
            float inv_ns = (float)(1.0 / s_ns);
            float inv_s = (float)(1.0 / s_s);
            cudaMemsetAsync(cache.d_is_src, 0, (size_t)n * sizeof(bool), stream);
            {
                int B = 256, G = (k + B - 1) / B;
                if (G < 1) G = 1;
                mark_sources<<<G, B, 0, stream>>>(cache.d_is_src, sample_vertices, k);
            }
            {
                int B = 256, G = (n + B - 1) / B;
                normalize<<<G, B, 0, stream>>>(d_bc, cache.d_is_src, n, inv_s, inv_ns);
            }
        }
    }

    cudaStreamSynchronize(stream);
}

}  
