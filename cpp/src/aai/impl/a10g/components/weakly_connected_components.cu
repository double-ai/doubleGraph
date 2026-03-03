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

struct Cache : Cacheable {
    int* d_flag = nullptr;
    int* h_flag = nullptr;

    Cache() {
        cudaHostAlloc(&h_flag, sizeof(int), cudaHostAllocMapped);
        cudaHostGetDevicePointer(&d_flag, h_flag, 0);
    }

    ~Cache() override {
        if (h_flag) {
            cudaFreeHost(h_flag);
            h_flag = nullptr;
            d_flag = nullptr;
        }
    }
};

__global__ void wcc_init(int* __restrict__ parent, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        parent[i] = i;
    }
}

__device__ __forceinline__ int find_root(int* __restrict__ parent, int v) {
    int p = parent[v];
    while (p != parent[p]) {
        int gp = parent[p];
        parent[v] = gp;
        v = p;
        p = gp;
    }
    return p;
}

__device__ __forceinline__ void link(int* __restrict__ parent, int u, int v) {
    int ru = find_root(parent, u);
    int rv = find_root(parent, v);

    while (ru != rv) {
        int hi = ru > rv ? ru : rv;
        int lo = ru < rv ? ru : rv;

        int old = atomicCAS(&parent[hi], hi, lo);
        if (old == hi) break;

        ru = find_root(parent, u);
        rv = find_root(parent, v);
    }
}

__global__ void wcc_hook_sample(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ parent,
    int n,
    int k
) {
    for (int u = blockIdx.x * blockDim.x + threadIdx.x; u < n; u += blockDim.x * gridDim.x) {
        int start = __ldg(&offsets[u]);
        int end = __ldg(&offsets[u + 1]);
        int degree = end - start;
        int samples = degree < k ? degree : k;

        for (int s = 0; s < samples; s++) {
            int v = __ldg(&indices[start + s]);
            link(parent, u, v);
        }
    }
}

__global__ void wcc_compress(int* __restrict__ parent, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int v = i;
        int p = parent[v];
        while (p != parent[p]) {
            int gp = parent[p];
            parent[v] = gp;
            v = p;
            p = gp;
        }
        parent[i] = p;
    }
}

__global__ void wcc_hook_full(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ parent,
    int n,
    int skip_root
) {
    for (int u = blockIdx.x * blockDim.x + threadIdx.x; u < n; u += blockDim.x * gridDim.x) {
        if (find_root(parent, u) == skip_root) continue;

        int start = __ldg(&offsets[u]);
        int end = __ldg(&offsets[u + 1]);

        for (int e = start; e < end; e++) {
            int v = __ldg(&indices[e]);
            link(parent, u, v);
        }
    }
}

__global__ void wcc_shortcut_check(int* __restrict__ parent, int n, int* __restrict__ not_converged) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int p = parent[i];
        int gp = parent[p];
        if (p != gp) {
            parent[i] = gp;
            *not_converged = 1;
        }
    }
}

__global__ void wcc_shortcut(int* __restrict__ parent, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int p = parent[i];
        int gp = parent[p];
        if (p != gp) parent[i] = gp;
    }
}

}  

void weakly_connected_components(const graph32_t& graph,
                                 int32_t* components) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int num_vertices = graph.number_of_vertices;
    int num_edges = graph.number_of_edges;
    const int* offsets = graph.offsets;
    const int* indices = graph.indices;
    int* parent = components;
    int* d_flag = cache.d_flag;
    int* h_flag = cache.h_flag;
    cudaStream_t stream = 0;

    if (num_vertices == 0) return;

    const int BLOCK = 512;
    int grid = (num_vertices + BLOCK - 1) / BLOCK;

    
    wcc_init<<<grid, BLOCK, 0, stream>>>(parent, num_vertices);

    
    wcc_hook_sample<<<grid, BLOCK, 0, stream>>>(offsets, indices, parent, num_vertices, 2);

    
    wcc_compress<<<grid, BLOCK, 0, stream>>>(parent, num_vertices);

    
    wcc_hook_full<<<grid, BLOCK, 0, stream>>>(offsets, indices, parent, num_vertices, 0);

    
    wcc_compress<<<grid, BLOCK, 0, stream>>>(parent, num_vertices);

    
    *h_flag = 0;
    __sync_synchronize();
    wcc_shortcut_check<<<grid, BLOCK, 0, stream>>>(parent, num_vertices, d_flag);
    cudaStreamSynchronize(stream);

    if (!(*h_flag)) return;

    
    for (int iter = 0; iter < 100; iter++) {
        for (int j = 0; j < 5; j++) {
            wcc_shortcut<<<grid, BLOCK, 0, stream>>>(parent, num_vertices);
        }
        *h_flag = 0;
        __sync_synchronize();
        wcc_shortcut_check<<<grid, BLOCK, 0, stream>>>(parent, num_vertices, d_flag);
        cudaStreamSynchronize(stream);
        if (!(*h_flag)) break;
    }
}

}  
