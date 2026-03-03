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

namespace aai {

namespace {

typedef unsigned long long ull;

#define SSSP_INF_VAL 3.4028235e+38f

struct Cache : Cacheable {
    ull* dp = nullptr;
    int* changed = nullptr;
    int* f1 = nullptr;
    int* f2 = nullptr;
    int* d_nfsize = nullptr;
    int* h_nfsize = nullptr;

    int64_t dp_capacity = 0;
    int64_t changed_capacity = 0;
    int64_t f1_capacity = 0;
    int64_t f2_capacity = 0;
    bool nfsize_allocated = false;

    void ensure(int64_t num_vertices) {
        if (dp_capacity < num_vertices) {
            if (dp) cudaFree(dp);
            cudaMalloc(&dp, num_vertices * sizeof(ull));
            dp_capacity = num_vertices;
        }
        if (changed_capacity < num_vertices) {
            if (changed) cudaFree(changed);
            cudaMalloc(&changed, num_vertices * sizeof(int));
            changed_capacity = num_vertices;
        }
        if (f1_capacity < num_vertices) {
            if (f1) cudaFree(f1);
            cudaMalloc(&f1, num_vertices * sizeof(int));
            f1_capacity = num_vertices;
        }
        if (f2_capacity < num_vertices) {
            if (f2) cudaFree(f2);
            cudaMalloc(&f2, num_vertices * sizeof(int));
            f2_capacity = num_vertices;
        }
        if (!nfsize_allocated) {
            cudaMalloc(&d_nfsize, sizeof(int));
            cudaMallocHost(&h_nfsize, sizeof(int));
            nfsize_allocated = true;
        }
    }

    ~Cache() override {
        if (dp) cudaFree(dp);
        if (changed) cudaFree(changed);
        if (f1) cudaFree(f1);
        if (f2) cudaFree(f2);
        if (d_nfsize) cudaFree(d_nfsize);
        if (h_nfsize) cudaFreeHost(h_nfsize);
    }
};



__device__ __forceinline__ float dp_dist(ull dp) {
    return __uint_as_float((unsigned)(dp >> 32));
}

__device__ __forceinline__ int dp_pred(ull dp) {
    return (int)(dp & 0xFFFFFFFFULL);
}

__device__ __forceinline__ ull make_dp(float dist, int pred) {
    return ((ull)__float_as_uint(dist) << 32) | (unsigned)pred;
}

__device__ __forceinline__ bool edge_active(const uint32_t* mask, int idx) {
    return (mask[idx >> 5] >> (idx & 31)) & 1;
}

__global__ void kern_init(ull* dp, int num_v, int src) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < num_v) {
        dp[t] = (t == src) ? make_dp(0.0f, -1) : make_dp(SSSP_INF_VAL, -1);
    }
}

__global__ void kern_clear(int* changed, const int* frontier, int fsize) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < fsize) {
        changed[frontier[t]] = 0;
    }
}


__global__ void kern_relax(
    const int* __restrict__ off,
    const int* __restrict__ idx,
    const float* __restrict__ wt,
    const uint32_t* __restrict__ emask,
    ull* __restrict__ dp,
    const int* __restrict__ frontier,
    int fsize,
    int* __restrict__ changed,
    int* __restrict__ nfrontier,
    int* __restrict__ nfsize,
    float cutoff
) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = gtid >> 5;
    int lane = gtid & 31;

    if (warp_id >= fsize) return;

    int v = frontier[warp_id];
    float dv = dp_dist(dp[v]);
    if (dv >= cutoff) return;

    int s = off[v];
    int e = off[v + 1];
    int degree = e - s;
    int iters = (degree + 31) >> 5;

    for (int iter = 0; iter < iters; iter++) {
        int i = s + iter * 32 + lane;
        bool should_add = false;
        int my_u = -1;

        if (i < e && edge_active(emask, i)) {
            int u = idx[i];
            float nd = dv + wt[i];
            if (nd < cutoff) {
                ull ndp = make_dp(nd, v);
                ull old_dp = atomicMin(&dp[u], ndp);

                if (ndp < old_dp) {
                    if (atomicExch(&changed[u], 1) == 0) {
                        should_add = true;
                        my_u = u;
                    }
                }
            }
        }

        
        unsigned int add_mask = __ballot_sync(0xFFFFFFFF, should_add);
        if (add_mask) {
            int count = __popc(add_mask);
            int base;
            if (lane == 0) {
                base = atomicAdd(nfsize, count);
            }
            base = __shfl_sync(0xFFFFFFFF, base, 0);
            if (should_add) {
                unsigned int lower_mask = (1u << lane) - 1;
                int my_offset = __popc(add_mask & lower_mask);
                nfrontier[base + my_offset] = my_u;
            }
        }
    }
}

__global__ void kern_extract(const ull* dp, float* dist, int* pred, int num_v, int source) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < num_v) {
        ull val = dp[t];
        dist[t] = dp_dist(val);
        pred[t] = -1;
    }
}

__global__ void fix_predecessors_positive_kernel(
    const int* __restrict__ off,
    const int* __restrict__ idx,
    const float* __restrict__ wt,
    const uint32_t* __restrict__ emask,
    const float* __restrict__ dist,
    int* __restrict__ pred,
    int num_v,
    int source
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int u = tid; u < num_v; u += stride) {
        float d_u = dist[u];
        if (d_u >= SSSP_INF_VAL) continue;
        int start = off[u];
        int end = off[u + 1];
        for (int e = start; e < end; e++) {
            if (!edge_active(emask, e)) continue;
            int v = idx[e];
            if (v == source) continue;
            float w = wt[e];
            if (d_u + w == dist[v] && d_u < dist[v]) {
                pred[v] = u;
            }
        }
    }
}

__global__ void fix_predecessors_zero_kernel(
    const int* __restrict__ off,
    const int* __restrict__ idx,
    const float* __restrict__ wt,
    const uint32_t* __restrict__ emask,
    const float* __restrict__ dist,
    int* __restrict__ pred,
    int* __restrict__ changed,
    int num_v,
    int source
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int u = tid; u < num_v; u += stride) {
        if (u != source && pred[u] == -1) continue;
        float d_u = dist[u];
        if (d_u >= SSSP_INF_VAL) continue;
        int start = off[u];
        int end = off[u + 1];
        for (int e = start; e < end; e++) {
            if (!edge_active(emask, e)) continue;
            int v = idx[e];
            if (v == source) continue;
            float w = wt[e];
            if (d_u + w == dist[v] && dist[v] == d_u && pred[v] == -1) {
                pred[v] = u;
                *changed = 1;
            }
        }
    }
}

}  

void sssp_seg_mask(const graph32_t& graph,
                   const float* edge_weights,
                   int32_t source,
                   float* distances,
                   int32_t* predecessors,
                   float cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    const uint32_t* edge_mask = graph.edge_mask;

    cache.ensure(num_vertices);

    ull* dp = cache.dp;
    int* changed = cache.changed;
    int* f1 = cache.f1;
    int* f2 = cache.f2;
    int* d_nfsize = cache.d_nfsize;
    int* h_nfsize = cache.h_nfsize;

    
    kern_init<<<(num_vertices + 255) / 256, 256>>>(dp, num_vertices, source);
    cudaMemsetAsync(changed, 0, (size_t)num_vertices * sizeof(int));

    
    int h_src = source;
    cudaMemcpy(f1, &h_src, sizeof(int), cudaMemcpyHostToDevice);
    int h_fsize = 1;

    
    for (int iter = 0; iter < num_vertices && h_fsize > 0; iter++) {
        
        kern_clear<<<(h_fsize + 255) / 256, 256>>>(changed, f1, h_fsize);

        
        cudaMemsetAsync(d_nfsize, 0, sizeof(int));

        
        int threads = 256;
        int warps_per_block = threads / 32;
        int grid = (h_fsize + warps_per_block - 1) / warps_per_block;
        kern_relax<<<grid, threads>>>(offsets, indices, edge_weights, edge_mask, dp,
                                       f1, h_fsize, changed, f2, d_nfsize, cutoff);

        
        cudaMemcpy(h_nfsize, d_nfsize, sizeof(int), cudaMemcpyDeviceToHost);
        h_fsize = *h_nfsize;

        
        int* tmp = f1; f1 = f2; f2 = tmp;
    }

    
    kern_extract<<<(num_vertices + 255) / 256, 256>>>(dp, distances, predecessors, num_vertices, source);

    
    {
        int64_t grid = ((int64_t)num_vertices + 255) / 256;
        if (grid > 65535) grid = 65535;
        fix_predecessors_positive_kernel<<<(int)grid, 256>>>(
            offsets, indices, edge_weights, edge_mask, distances, predecessors, num_vertices, source);
    }

    
    {
        int h_changed = 1;
        for (int iter = 0; iter < num_vertices && h_changed; iter++) {
            cudaMemsetAsync(d_nfsize, 0, sizeof(int));
            int64_t grid = ((int64_t)num_vertices + 255) / 256;
            if (grid > 65535) grid = 65535;
            fix_predecessors_zero_kernel<<<(int)grid, 256>>>(
                offsets, indices, edge_weights, edge_mask, distances, predecessors, d_nfsize, num_vertices, source);
            cudaMemcpy(h_nfsize, d_nfsize, sizeof(int), cudaMemcpyDeviceToHost);
            h_changed = *h_nfsize;
        }
    }
}

}  
