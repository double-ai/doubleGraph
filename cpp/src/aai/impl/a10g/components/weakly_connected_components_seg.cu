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

__device__ __forceinline__ int find_ro(int* __restrict__ parent, int v) {
    int curr = v;
    while (parent[curr] != curr) curr = parent[curr];
    return curr;
}

__device__ __forceinline__ int find(int* __restrict__ parent, int v) {
    int curr = v;
    while (true) {
        int next = parent[curr];
        if (next == curr) return curr;
        parent[curr] = parent[next];
        curr = next;
    }
}

__global__ void init_kernel(int* __restrict__ parent, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) parent[tid] = tid;
}

__global__ void fused_sample_hook_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ parent,
    int n
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);
    int degree = end - start;

    if (degree >= 1) {
        int u = __ldg(&indices[start]);
        int rv = find_ro(parent, v);
        int ru = find_ro(parent, u);
        if (rv != ru) {
            int high = (rv > ru) ? rv : ru;
            int low  = (rv > ru) ? ru : rv;
            atomicCAS(&parent[high], high, low);
        }
    }

    if (degree >= 2) {
        int u = __ldg(&indices[start + 1]);
        int rv = find_ro(parent, v);
        int ru = find_ro(parent, u);
        if (rv != ru) {
            int high = (rv > ru) ? rv : ru;
            int low  = (rv > ru) ? ru : rv;
            atomicCAS(&parent[high], high, low);
        }
    }
}

__global__ void compress_kernel(int* __restrict__ parent, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int curr = tid;
    while (parent[curr] != curr) curr = parent[curr];
    parent[tid] = curr;
}

__global__ void full_hook_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ parent,
    int n
) {
    int skip_root = parent[0];
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (parent[v] == skip_root) return;

    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);

    for (int i = start; i < end; i++) {
        int u = __ldg(&indices[i]);
        int rv = find(parent, v);
        int ru = find(parent, u);
        while (rv != ru) {
            int high = (rv > ru) ? rv : ru;
            int low  = (rv > ru) ? ru : rv;
            if (atomicCAS(&parent[high], high, low) == high) break;
            rv = find(parent, v);
            ru = find(parent, u);
        }
    }
}

}  

void weakly_connected_components_seg(const graph32_t& graph,
                                     int32_t* components) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    int32_t n = graph.number_of_vertices;
    if (n == 0) return;

    const int* d_offsets = graph.offsets;
    const int* d_indices = graph.indices;
    int* d_parent = components;
    cudaStream_t stream = 0;

    init_kernel<<<(n+1023)/1024, 1024, 0, stream>>>(d_parent, n);
    fused_sample_hook_kernel<<<(n+1023)/1024, 1024, 0, stream>>>(d_offsets, d_indices, d_parent, n);
    compress_kernel<<<(n+1023)/1024, 1024, 0, stream>>>(d_parent, n);
    full_hook_kernel<<<(n+1023)/1024, 1024, 0, stream>>>(d_offsets, d_indices, d_parent, n);
    compress_kernel<<<(n+1023)/1024, 1024, 0, stream>>>(d_parent, n);

    cudaStreamSynchronize(stream);
}

}  
