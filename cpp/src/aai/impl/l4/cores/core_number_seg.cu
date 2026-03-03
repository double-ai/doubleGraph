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
#include <climits>
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* d_core = nullptr;
    int32_t* d_removed = nullptr;
    int32_t* d_frontier0 = nullptr;
    int32_t* d_frontier1 = nullptr;
    int32_t* d_cnt = nullptr;
    int32_t* h_cnt = nullptr;
    size_t alloc_v = 0;

    Cache() {
        cudaHostAlloc(&h_cnt, 4 * sizeof(int32_t), cudaHostAllocDefault);
    }

    void ensure(size_t nv) {
        if (nv <= alloc_v) return;
        if (d_core) cudaFree(d_core);
        if (d_removed) cudaFree(d_removed);
        if (d_frontier0) cudaFree(d_frontier0);
        if (d_frontier1) cudaFree(d_frontier1);
        if (d_cnt) cudaFree(d_cnt);

        cudaMalloc(&d_core, nv * sizeof(int32_t));
        cudaMalloc(&d_removed, nv * sizeof(int32_t));
        cudaMalloc(&d_frontier0, nv * sizeof(int32_t));
        cudaMalloc(&d_frontier1, nv * sizeof(int32_t));
        cudaMalloc(&d_cnt, 4 * sizeof(int32_t));
        alloc_v = nv;
    }

    ~Cache() override {
        if (d_core) cudaFree(d_core);
        if (d_removed) cudaFree(d_removed);
        if (d_frontier0) cudaFree(d_frontier0);
        if (d_frontier1) cudaFree(d_frontier1);
        if (d_cnt) cudaFree(d_cnt);
        if (h_cnt) cudaFreeHost(h_cnt);
    }
};

static inline __host__ int grid_sz(int n, int block = 256) {
    int g = (n + block - 1) / block;
    return g < 1 ? 1 : (g > 2048 ? 2048 : g);
}


__global__ void init_core_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ core,
    int32_t* __restrict__ removed,
    int32_t num_vertices,
    int32_t delta,
    int32_t k_first
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int v = idx; v < num_vertices; v += stride) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int self_loops = 0;
        for (int e = start; e < end; e++) {
            if (indices[e] == v) self_loops++;
        }
        int degree = (end - start) - self_loops;
        int c = degree * delta;
        if (c > 0 && c < k_first) c = 0;
        core[v] = c;
        removed[v] = (degree == 0) ? 1 : 0;
    }
}



__global__ void build_frontier_min_kernel(
    int32_t* __restrict__ core,
    int32_t* __restrict__ removed,
    int32_t k,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ counters,
    int32_t num_vertices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int local_min = INT_MAX;
    for (int v = idx; v < num_vertices; v += stride) {
        if (!removed[v]) {
            int c = core[v];
            if (c < k) {
                int slot = atomicAdd(&counters[0], 1);
                frontier[slot] = v;
                removed[v] = 1;
            } else {
                if (c < local_min) local_min = c;
            }
        }
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        int other = __shfl_down_sync(0xffffffff, local_min, offset);
        if (other < local_min) local_min = other;
    }
    if ((threadIdx.x & 31) == 0 && local_min < INT_MAX) {
        atomicMin(&counters[1], local_min);
    }
}




__global__ void decrement_threshold_kernel(
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ core,
    int32_t* __restrict__ removed,
    int32_t delta,
    int32_t k,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_size
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int num_warps = (blockDim.x * gridDim.x) / 32;

    for (int i = warp_id; i < frontier_size; i += num_warps) {
        int v = frontier[i];
        int start = offsets[v];
        int end = offsets[v + 1];
        for (int e = start + lane; e < end; e += 32) {
            int u = indices[e];
            if (u != v && !removed[u]) {
                int old = atomicSub(&core[u], delta);
                
                if (old == k) {
                    int slot = atomicAdd(next_size, 1);
                    next_frontier[slot] = u;
                    removed[u] = 1;
                }
            }
        }
    }
}


__global__ void clamp_frontier_kernel(
    int32_t* __restrict__ core,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t k_minus_delta,
    int32_t k_first
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < frontier_size; i += stride) {
        int u = frontier[i];
        int c = core[u];
        if (c < 0) c = 0;
        if (c < k_minus_delta) c = k_minus_delta;
        if (c > 0 && c < k_first) c = 0;
        core[u] = c;
    }
}

}  

void core_number_seg(const graph32_t& graph,
                     int32_t* core_numbers,
                     int degree_type,
                     std::size_t k_first,
                     std::size_t k_last) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    if (nv == 0) return;

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;

    cache.ensure(nv);

    int32_t* d_core = cache.d_core;
    int32_t* d_removed = cache.d_removed;
    int32_t* d_frontier[2] = {cache.d_frontier0, cache.d_frontier1};
    int32_t* d_cnt = cache.d_cnt;
    int32_t* h_cnt = cache.h_cnt;

    int32_t delta = (degree_type == 2) ? 2 : 1;

    init_core_kernel<<<grid_sz(nv), 256>>>(
        d_off, d_idx, d_core, d_removed, nv, delta, (int32_t)k_first);

    size_t k = std::max(k_first, size_t{2});
    if (delta == 2 && (k & 1)) k++;

    int cur = 0, nxt = 1;
    int32_t* d_level_cnt = d_cnt;      
    int32_t* d_next_cnt = d_cnt + 2;   

    while (k <= k_last) {
        
        h_cnt[0] = 0;
        h_cnt[1] = INT_MAX;
        cudaMemcpyAsync(d_level_cnt, h_cnt, 2 * sizeof(int32_t), cudaMemcpyHostToDevice);

        build_frontier_min_kernel<<<grid_sz(nv), 256>>>(
            d_core, d_removed, (int32_t)k,
            d_frontier[cur], d_level_cnt, nv);

        cudaMemcpy(h_cnt, d_level_cnt, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost);
        int32_t fs = h_cnt[0];
        int32_t mc = h_cnt[1];

        if (fs == 0) {
            if (mc == INT_MAX) break;
            size_t nk = static_cast<size_t>(mc) + delta;
            if (delta == 2 && (nk & 1)) nk++;
            k = std::max(k + delta, nk);
            continue;
        }

        
        while (fs > 0) {
            
            cudaMemsetAsync(d_next_cnt, 0, sizeof(int32_t));

            {
                int threads = 256;
                int blocks = (int)(((int64_t)fs * 32 + threads - 1) / threads);
                if (blocks < 1) blocks = 1;
                if (blocks > 2048) blocks = 2048;
                decrement_threshold_kernel<<<blocks, threads>>>(
                    d_frontier[cur], fs, d_off, d_idx,
                    d_core, d_removed, delta, (int32_t)k,
                    d_frontier[nxt], d_next_cnt);
            }

            cudaMemcpy(h_cnt + 2, d_next_cnt, sizeof(int32_t), cudaMemcpyDeviceToHost);
            fs = h_cnt[2];

            if (fs > 0) {
                
                clamp_frontier_kernel<<<grid_sz(fs), 256>>>(
                    d_core, d_frontier[nxt], fs,
                    (int32_t)(k - delta), (int32_t)k_first);
            }

            int t = cur; cur = nxt; nxt = t;
        }

        k += delta;
    }

    cudaMemcpy(core_numbers, d_core, nv * sizeof(int32_t), cudaMemcpyDeviceToDevice);
}

}  
