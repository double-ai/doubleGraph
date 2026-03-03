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
#include <cfloat>
#include <cmath>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* d_dist = nullptr;
    int32_t* d_frontier0 = nullptr;
    int32_t* d_frontier1 = nullptr;
    int32_t* d_count = nullptr;
    int32_t* d_in_frontier_gen = nullptr;
    int32_t* h_count = nullptr;
    int32_t alloc_n = 0;

    void ensure(int32_t n) {
        if (n > alloc_n) {
            if (d_dist) cudaFree(d_dist);
            if (d_frontier0) cudaFree(d_frontier0);
            if (d_frontier1) cudaFree(d_frontier1);
            if (d_count) cudaFree(d_count);
            if (d_in_frontier_gen) cudaFree(d_in_frontier_gen);
            if (h_count) cudaFreeHost(h_count);

            cudaMalloc(&d_dist, (size_t)n * sizeof(float));
            cudaMalloc(&d_frontier0, (size_t)n * sizeof(int32_t));
            cudaMalloc(&d_frontier1, (size_t)n * sizeof(int32_t));
            cudaMalloc(&d_count, sizeof(int32_t));
            cudaMalloc(&d_in_frontier_gen, (size_t)n * sizeof(int32_t));
            cudaMallocHost(&h_count, sizeof(int32_t));
            alloc_n = n;
        }
    }

    ~Cache() override {
        if (d_dist) cudaFree(d_dist);
        if (d_frontier0) cudaFree(d_frontier0);
        if (d_frontier1) cudaFree(d_frontier1);
        if (d_count) cudaFree(d_count);
        if (d_in_frontier_gen) cudaFree(d_in_frontier_gen);
        if (h_count) cudaFreeHost(h_count);
    }
};

__device__ __forceinline__ float atomicMinFloat(float* addr, float val) {
    return __int_as_float(atomicMin((int*)addr, __float_as_int(val)));
}

__global__ void init_sssp(float* dist, int32_t n, int32_t src) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dist[i] = (i == src) ? 0.0f : FLT_MAX;
    }
}

__global__ void relax_warp(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ col_indices,
    const float* __restrict__ edge_weights,
    float* __restrict__ dist,
    const int32_t* __restrict__ frontier,
    const int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count,
    int32_t* __restrict__ in_frontier_gen,
    const int32_t curr_gen,
    const float cutoff)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t v;
    float vd;
    int32_t begin, end;

    if (lane == 0) {
        v = frontier[warp_id];
        vd = dist[v];
        begin = row_offsets[v];
        end = row_offsets[v + 1];
    }
    v = __shfl_sync(0xFFFFFFFF, v, 0);
    vd = __shfl_sync(0xFFFFFFFF, vd, 0);
    begin = __shfl_sync(0xFFFFFFFF, begin, 0);
    end = __shfl_sync(0xFFFFFFFF, end, 0);

    if (vd >= cutoff) return;

    for (int32_t e_base = begin; e_base < end; e_base += 32) {
        int32_t e = e_base + lane;
        bool valid = (e < end);

        bool need_add = false;
        int32_t u = -1;

        if (valid) {
            u = col_indices[e];
            float nd = vd + edge_weights[e];

            if (nd < cutoff) {
                float cur_d = dist[u];
                if (nd < cur_d) {
                    float old_d = atomicMinFloat(&dist[u], nd);
                    if (nd < old_d) {
                        int old_gen = atomicMax(&in_frontier_gen[u], curr_gen);
                        if (old_gen < curr_gen) {
                            need_add = true;
                        }
                    }
                }
            }
        }

        unsigned mask = __ballot_sync(0xFFFFFFFF, need_add);
        if (mask) {
            int count = __popc(mask);
            int leader = __ffs(mask) - 1;
            int base;
            if (lane == leader) {
                base = atomicAdd(next_count, count);
            }
            base = __shfl_sync(0xFFFFFFFF, base, leader);
            if (need_add) {
                int offset = __popc(mask & ((1U << lane) - 1));
                next_frontier[base + offset] = u;
            }
        }
    }
}

__global__ void pred_positive_kernel(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ col_indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int32_t u = tid; u < n; u += stride) {
        float d_u = dist[u];
        if (__float_as_int(d_u) >= __float_as_int(FLT_MAX)) continue;
        int32_t start = row_offsets[u];
        int32_t end = row_offsets[u + 1];
        for (int32_t e = start; e < end; e++) {
            int32_t v = col_indices[e];
            float w = edge_weights[e];
            float dv = dist[v];
            if (d_u + w == dv && d_u < dv) {
                pred[v] = u;
            }
        }
    }
}

__global__ void pred_zero_kernel(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ col_indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t* __restrict__ changed,
    int32_t n,
    int32_t source)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int32_t u = tid; u < n; u += stride) {
        if (u != source && pred[u] == -1) continue;
        float d_u = dist[u];
        if (__float_as_int(d_u) >= __float_as_int(FLT_MAX)) continue;
        int32_t start = row_offsets[u];
        int32_t end = row_offsets[u + 1];
        for (int32_t e = start; e < end; e++) {
            int32_t v = col_indices[e];
            if (v == source) continue;
            float w = edge_weights[e];
            if (d_u + w == dist[v] && dist[v] == d_u && pred[v] == -1) {
                pred[v] = u;
                *changed = 1;
            }
        }
    }
}

}  

void sssp_seg(const graph32_t& graph,
              const float* edge_weights,
              int32_t source,
              float* distances,
              int32_t* predecessors,
              float cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t n = graph.number_of_vertices;
    if (std::isinf(cutoff)) cutoff = FLT_MAX;

    cache.ensure(n);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    init_sssp<<<(n + 255) / 256, 256>>>(cache.d_dist, n, source);
    cudaMemset(cache.d_in_frontier_gen, 0, (size_t)n * sizeof(int32_t));
    cudaMemcpy(cache.d_frontier0, &source, sizeof(int32_t), cudaMemcpyHostToDevice);

    int32_t fsize = 1;
    int cur = 0;
    int32_t gen = 1;

    int32_t* frontiers[2] = {cache.d_frontier0, cache.d_frontier1};

    while (fsize > 0) {
        int nxt = 1 - cur;
        cudaMemset(cache.d_count, 0, sizeof(int32_t));

        if (fsize > 0) {
            int block = 256;
            int64_t total_threads = (int64_t)fsize * 32;
            int grid = (int)((total_threads + block - 1) / block);
            relax_warp<<<grid, block>>>(
                d_offsets, d_indices, edge_weights,
                cache.d_dist, frontiers[cur], fsize,
                frontiers[nxt], cache.d_count, cache.d_in_frontier_gen,
                gen, cutoff);
        }

        cudaMemcpy(cache.h_count, cache.d_count, sizeof(int32_t), cudaMemcpyDeviceToHost);
        fsize = *cache.h_count;
        cur = nxt;
        gen++;
    }

    cudaMemcpy(distances, cache.d_dist, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice);

    
    cudaMemsetAsync(predecessors, 0xFF, (size_t)n * sizeof(int32_t));

    
    {
        int block = 256;
        int64_t grid64 = ((int64_t)n + block - 1) / block;
        int grid = (int)grid64;
        pred_positive_kernel<<<grid, block>>>(d_offsets, d_indices, edge_weights,
            cache.d_dist, predecessors, n);
    }

    
    {
        int block = 256;
        int64_t grid64 = ((int64_t)n + block - 1) / block;
        int grid = (int)grid64;
        int32_t h_changed = 1;
        for (int iter = 0; iter < n && h_changed; iter++) {
            cudaMemsetAsync(cache.d_count, 0, sizeof(int32_t));
            pred_zero_kernel<<<grid, block>>>(d_offsets, d_indices, edge_weights,
                cache.d_dist, predecessors, cache.d_count, n, source);
            cudaMemcpy(&h_changed, cache.d_count, sizeof(int32_t), cudaMemcpyDeviceToHost);
        }
    }

    cudaDeviceSynchronize();
}

}  
