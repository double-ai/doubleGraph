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

struct Cache : Cacheable {};

__device__ __forceinline__ int32_t find(int32_t* parent, int32_t x) {
    int32_t p = parent[x];
    while (p != x) {
        int32_t gp = parent[p];
        parent[x] = gp;
        x = p;
        p = gp;
    }
    return x;
}

__device__ __forceinline__ void unite(int32_t* parent, int32_t x, int32_t y) {
    while (true) {
        x = find(parent, x);
        y = find(parent, y);
        if (x == y) return;
        if (x > y) { int32_t t = x; x = y; y = t; }
        if (atomicCAS(&parent[y], y, x) == y) return;
    }
}

__global__ void init_kernel(int32_t* __restrict__ parent, int32_t n) {
    for (int32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += blockDim.x * gridDim.x)
        parent[v] = v;
}

__global__ void sample_hook_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ parent,
    int32_t n, int32_t k)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    int32_t pv = parent[v];
    int32_t start = __ldg(&offsets[v]);
    int32_t end = __ldg(&offsets[v + 1]);
    int32_t count = 0;

    for (int32_t e = start; e < end && count < k; e++) {
        if (!((__ldg(&edge_mask[e >> 5]) >> (e & 31)) & 1u)) continue;
        int32_t u = __ldg(&indices[e]);
        int32_t pu = parent[u];
        if (pv < pu) {
            atomicMin(&parent[pu], pv);
        } else if (pu < pv) {
            atomicMin(&parent[pv], pu);
            pv = pu;
        }
        count++;
    }
}

__global__ void compress_kernel(int32_t* __restrict__ parent, int32_t n) {
    for (int32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += blockDim.x * gridDim.x) {
        int32_t p = parent[v];
        while (p != parent[p]) p = parent[p];
        parent[v] = p;
    }
}

__global__ void edge_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ parent,
    int32_t num_vertices,
    int32_t num_edges)
{
    const int32_t EPT = 3;
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t e_start = tid * EPT;
    if (e_start >= num_edges) return;
    int32_t e_end = e_start + EPT;
    if (e_end > num_edges) e_end = num_edges;

    
    int32_t lo = 0, hi = num_vertices;
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (__ldg(&offsets[mid + 1]) <= e_start) lo = mid + 1;
        else hi = mid;
    }
    int32_t v = lo;

    for (int32_t e = e_start; e < e_end; e++) {
        
        while (v < num_vertices - 1 && __ldg(&offsets[v + 1]) <= e) v++;

        
        if (!((__ldg(&edge_mask[e >> 5]) >> (e & 31)) & 1u)) continue;

        int32_t u = __ldg(&indices[e]);
        if (parent[v] == parent[u]) continue;
        unite(parent, v, u);
    }
}

}  

void weakly_connected_components_mask(const graph32_t& graph,
                                      int32_t* components) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    if (num_vertices == 0) return;

    const int block = 256;
    const int grid_v = (num_vertices + block - 1) / block;

    const int EPT = 3;
    int32_t num_threads_e = (num_edges + EPT - 1) / EPT;
    const int grid_e = (num_threads_e + block - 1) / block;

    init_kernel<<<grid_v, block>>>(components, num_vertices);
    sample_hook_kernel<<<grid_v, block>>>(graph.offsets, graph.indices, graph.edge_mask, components, num_vertices, 2);
    compress_kernel<<<grid_v, block>>>(components, num_vertices);
    edge_kernel<<<grid_e, block>>>(graph.offsets, graph.indices, graph.edge_mask, components, num_vertices, num_edges);
    compress_kernel<<<grid_v, block>>>(components, num_vertices);
}

}  
