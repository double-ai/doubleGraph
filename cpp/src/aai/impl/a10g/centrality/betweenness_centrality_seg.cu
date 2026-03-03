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
#include <cmath>
#include <algorithm>
#include <vector>

namespace aai {

namespace {

static const int MAX_CONCURRENT = 10;

struct Cache : Cacheable {
    cudaStream_t streams[MAX_CONCURRENT] = {};
    int32_t* d_dist[MAX_CONCURRENT] = {};
    float* d_sigma[MAX_CONCURRENT] = {};
    float* d_delta[MAX_CONCURRENT] = {};
    int32_t* d_stack[MAX_CONCURRENT] = {};
    int32_t* d_nc_arr[MAX_CONCURRENT] = {};
    int32_t* h_nc[MAX_CONCURRENT] = {};
    int alloc_count = 0;
    int32_t max_v = 0;

    void ensure_buffers(int32_t nv, int nc) {
        if (nc > MAX_CONCURRENT) nc = MAX_CONCURRENT;
        if (nv <= max_v && nc <= alloc_count) return;
        free_buffers();
        max_v = nv;
        alloc_count = nc;
        for (int i = 0; i < nc; i++) {
            cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
            cudaMalloc(&d_dist[i], (size_t)nv * sizeof(int32_t));
            cudaMalloc(&d_sigma[i], (size_t)nv * sizeof(float));
            cudaMalloc(&d_delta[i], (size_t)nv * sizeof(float));
            cudaMalloc(&d_stack[i], (size_t)nv * sizeof(int32_t));
            cudaMalloc(&d_nc_arr[i], (size_t)(nv + 1) * sizeof(int32_t));
            cudaMallocHost(&h_nc[i], sizeof(int32_t));
        }
    }

    void free_buffers() {
        for (int i = 0; i < alloc_count; i++) {
            if (d_dist[i]) cudaFree(d_dist[i]);
            if (d_sigma[i]) cudaFree(d_sigma[i]);
            if (d_delta[i]) cudaFree(d_delta[i]);
            if (d_stack[i]) cudaFree(d_stack[i]);
            if (d_nc_arr[i]) cudaFree(d_nc_arr[i]);
            if (h_nc[i]) cudaFreeHost(h_nc[i]);
            if (streams[i]) cudaStreamDestroy(streams[i]);
            d_dist[i] = nullptr; d_sigma[i] = nullptr;
            d_delta[i] = nullptr; d_stack[i] = nullptr;
            d_nc_arr[i] = nullptr; h_nc[i] = nullptr; streams[i] = nullptr;
        }
        alloc_count = 0; max_v = 0;
    }

    ~Cache() override { free_buffers(); }
};



__global__ void init_bfs_kernel(
    int32_t* __restrict__ distances, float* __restrict__ sigma,
    float* __restrict__ delta, int32_t* __restrict__ stack,
    int32_t num_vertices, int32_t source
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < num_vertices; i += stride) {
        distances[i] = (i == source) ? 0 : -1;
        sigma[i] = (i == source) ? 1.0f : 0.0f;
        delta[i] = 0.0f;
    }
    if (idx == 0) stack[0] = source;
}

__global__ void bfs_forward_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances, float* __restrict__ sigma,
    int32_t* __restrict__ stack, int32_t frontier_start, int32_t frontier_end,
    int32_t next_level, int32_t* __restrict__ next_count
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;

    for (int i = warp_id + frontier_start; i < frontier_end; i += num_warps) {
        int v = stack[i];
        int start = __ldg(&offsets[v]);
        int end = __ldg(&offsets[v + 1]);
        float sv = sigma[v];

        for (int j = start + lane; j < end; j += 32) {
            int u = __ldg(&indices[j]);
            int old_dist = atomicCAS(&distances[u], -1, next_level);
            if (old_dist == -1) {
                int pos = atomicAdd(next_count, 1);
                stack[frontier_end + pos] = u;
            }
            if (old_dist == -1 || old_dist == next_level) {
                atomicAdd(&sigma[u], sv);
            }
        }
    }
}

__global__ void backward_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ distances, const float* __restrict__ sigma,
    float* __restrict__ delta, float* __restrict__ bc,
    const int32_t* __restrict__ stack, int32_t frontier_start, int32_t frontier_end,
    int32_t next_level, int32_t source
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int num_warps = (gridDim.x * blockDim.x) >> 5;

    for (int i = warp_id + frontier_start; i < frontier_end; i += num_warps) {
        int v = stack[i];
        int start = __ldg(&offsets[v]);
        int end = __ldg(&offsets[v + 1]);
        float d = 0.0f;

        for (int j = start + lane; j < end; j += 32) {
            int w = __ldg(&indices[j]);
            if (distances[w] == next_level) {
                d += (1.0f + delta[w]) / sigma[w];
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            d += __shfl_down_sync(0xFFFFFFFF, d, offset);

        if (lane == 0) {
            float result = sigma[v] * d;
            delta[v] = result;
            if (v != source) atomicAdd(&bc[v], result);
        }
    }
}

__global__ void add_endpoints_kernel(
    float* __restrict__ bc, const int32_t* __restrict__ stack,
    int32_t total_discovered, int32_t source, float reachable_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    if (idx == 0) atomicAdd(&bc[source], reachable_count);
    for (int i = idx + 1; i < total_discovered; i += stride)
        atomicAdd(&bc[stack[i]], 1.0f);
}

__global__ void normalize_all_kernel(float* __restrict__ bc, int32_t n, float inv_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = idx; i < n; i += stride) bc[i] *= inv_scale;
}

__global__ void adjust_sources_kernel(float* __restrict__ bc, const int32_t* __restrict__ sv,
                                       int32_t ns, float adj) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ns) bc[sv[idx]] *= adj;
}



void launch_init_bfs(int32_t* d, float* s, float* dl, int32_t* st,
                     int32_t nv, int32_t src, cudaStream_t stream) {
    int b = 256, g = (nv + b - 1) / b;
    init_bfs_kernel<<<g, b, 0, stream>>>(d, s, dl, st, nv, src);
}

void launch_bfs_forward(const int32_t* off, const int32_t* ind,
                        int32_t* dist, float* sigma, int32_t* stack,
                        int32_t fs, int32_t fe, int32_t nl,
                        int32_t* nc, cudaStream_t stream) {
    int frontier_size = fe - fs;
    if (frontier_size == 0) return;
    int tpb = 256;
    int blocks = (int)(((int64_t)frontier_size * 32 + tpb - 1) / tpb);
    bfs_forward_kernel<<<blocks, tpb, 0, stream>>>(off, ind, dist, sigma, stack, fs, fe, nl, nc);
}

void launch_backward(const int32_t* off, const int32_t* ind,
                     const int32_t* dist, const float* sigma,
                     float* delta, float* bc, const int32_t* stack,
                     int32_t fs, int32_t fe, int32_t nl, int32_t src,
                     cudaStream_t stream) {
    int frontier_size = fe - fs;
    if (frontier_size == 0) return;
    int tpb = 256;
    int blocks = (int)(((int64_t)frontier_size * 32 + tpb - 1) / tpb);
    backward_kernel<<<blocks, tpb, 0, stream>>>(off, ind, dist, sigma, delta, bc, stack, fs, fe, nl, src);
}

void launch_add_endpoints(float* bc, const int32_t* stack, int32_t td,
                          int32_t src, float rc, cudaStream_t stream) {
    if (td <= 1) return;
    int b = 256, g = (td + b - 1) / b;
    add_endpoints_kernel<<<g, b, 0, stream>>>(bc, stack, td, src, rc);
}

void launch_normalize_all(float* bc, int32_t nv, float inv, cudaStream_t stream) {
    int b = 256, g = (nv + b - 1) / b;
    normalize_all_kernel<<<g, b, 0, stream>>>(bc, nv, inv);
}

void launch_adjust_sources(float* bc, const int32_t* sv, int32_t ns, float adj, cudaStream_t stream) {
    int b = 256, g = (ns + b - 1) / b;
    adjust_sources_kernel<<<g, b, 0, stream>>>(bc, sv, ns, adj);
}



void process_batch(
    Cache& cache,
    const int32_t* d_off, const int32_t* d_ind, float* d_bc,
    const int32_t* h_samples, int batch_size, int32_t nv,
    bool include_endpoints
) {
    struct SS {
        int32_t source;
        int max_level;
        bool done;
        std::vector<int32_t> lo;
    };
    std::vector<SS> st(batch_size);
    for (int bi = 0; bi < batch_size; bi++) {
        st[bi].source = h_samples[bi];
        st[bi].max_level = 0;
        st[bi].done = false;
        st[bi].lo = {0, 1};
    }

    
    for (int bi = 0; bi < batch_size; bi++) {
        launch_init_bfs(cache.d_dist[bi], cache.d_sigma[bi], cache.d_delta[bi], cache.d_stack[bi],
                       nv, st[bi].source, cache.streams[bi]);
        cudaMemsetAsync(cache.d_nc_arr[bi], 0, (size_t)(nv + 1) * sizeof(int32_t), cache.streams[bi]);
    }

    
    for (int level = 0; level < nv; level++) {
        bool any_active = false;
        for (int bi = 0; bi < batch_size; bi++) {
            if (st[bi].done) continue;
            int32_t fs = st[bi].lo[level];
            int32_t fe = st[bi].lo[level + 1];
            if (fs >= fe) { st[bi].done = true; continue; }

            launch_bfs_forward(d_off, d_ind, cache.d_dist[bi], cache.d_sigma[bi],
                               cache.d_stack[bi], fs, fe, level + 1,
                               &cache.d_nc_arr[bi][level + 1], cache.streams[bi]);
            cudaMemcpyAsync(cache.h_nc[bi], &cache.d_nc_arr[bi][level + 1], sizeof(int32_t),
                            cudaMemcpyDeviceToHost, cache.streams[bi]);
            any_active = true;
        }
        if (!any_active) break;

        for (int i = 0; i < batch_size; i++) {
            if (!st[i].done) cudaStreamSynchronize(cache.streams[i]);
        }

        any_active = false;
        for (int bi = 0; bi < batch_size; bi++) {
            if (st[bi].done) continue;
            int32_t ns = *cache.h_nc[bi];
            if (ns == 0) {
                st[bi].max_level = level;
                st[bi].done = true;
            } else {
                st[bi].lo.push_back(st[bi].lo[level + 1] + ns);
                st[bi].max_level = level + 1;
                any_active = true;
            }
        }
        if (!any_active) break;
    }

    
    for (int bi = 0; bi < batch_size; bi++) {
        int ml = st[bi].max_level;
        int32_t source = st[bi].source;
        for (int level = ml - 1; level >= 1; level--) {
            launch_backward(d_off, d_ind, cache.d_dist[bi], cache.d_sigma[bi],
                           cache.d_delta[bi], d_bc, cache.d_stack[bi],
                           st[bi].lo[level], st[bi].lo[level + 1],
                           level + 1, source, cache.streams[bi]);
        }
        if (include_endpoints) {
            int32_t td = st[bi].lo[ml + 1];
            if (td > 1)
                launch_add_endpoints(d_bc, cache.d_stack[bi], td, source, (float)(td - 1), cache.streams[bi]);
        }
    }

    for (int i = 0; i < batch_size; i++)
        cudaStreamSynchronize(cache.streams[i]);
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

    int32_t nv = graph.number_of_vertices;
    const int32_t* d_off = graph.offsets;
    const int32_t* d_ind = graph.indices;
    bool is_symmetric = graph.is_symmetric;

    int ns = (int)num_samples;
    int concurrent = std::min(ns, MAX_CONCURRENT);
    cache.ensure_buffers(nv, concurrent);

    float* d_bc = centralities;
    cudaMemset(d_bc, 0, (size_t)nv * sizeof(float));

    std::vector<int32_t> h_samples(ns);
    cudaMemcpy(h_samples.data(), sample_vertices, ns * sizeof(int32_t), cudaMemcpyDeviceToHost);

    for (int batch_start = 0; batch_start < ns; batch_start += concurrent) {
        int batch_size = std::min(concurrent, ns - batch_start);
        process_batch(cache, d_off, d_ind, d_bc, &h_samples[batch_start],
                     batch_size, nv, include_endpoints);
    }

    
    cudaStream_t norm_s = cache.streams[0];
    int32_t k = (int32_t)ns;
    int32_t adj_int = include_endpoints ? nv : nv - 1;
    float adj = (float)adj_int;
    bool all_srcs = (k == adj_int) || include_endpoints;

    if (all_srcs) {
        float scale;
        if (normalized) scale = (float)k * (adj - 1.0f);
        else if (is_symmetric) scale = (float)k * 2.0f / adj;
        else scale = (float)k / adj;
        if (scale != 0.0f) launch_normalize_all(d_bc, nv, 1.0f / scale, norm_s);
    } else {
        float sns, ss;
        if (normalized) { sns = (float)k * (adj - 1.0f); ss = (float)(k-1) * (adj - 1.0f); }
        else { sns = (float)k / adj; ss = (float)(k-1) / adj;
               if (is_symmetric) { sns *= 2.0f; ss *= 2.0f; } }
        if (sns != 0.0f) launch_normalize_all(d_bc, nv, 1.0f / sns, norm_s);
        if (sns != 0.0f && ss != sns)
            launch_adjust_sources(d_bc, sample_vertices, k, sns / ss, norm_s);
    }
    cudaStreamSynchronize(norm_s);
}

}  
