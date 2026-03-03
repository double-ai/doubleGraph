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

namespace aai {

namespace {

using vertex_t = int32_t;
using edge_t = int32_t;

struct Cache : Cacheable {
    int* d_changed = nullptr;
    int* d_count = nullptr;
    int* h_pinned = nullptr;  
    int* d_active = nullptr;
    int* d_degrees = nullptr;
    int32_t active_capacity = 0;
    int32_t degrees_capacity = 0;

    Cache() {
        cudaMalloc(&d_changed, sizeof(int));
        cudaMalloc(&d_count, sizeof(int));
        cudaMallocHost(&h_pinned, 2 * sizeof(int));
    }

    ~Cache() override {
        if (d_changed) cudaFree(d_changed);
        if (d_count) cudaFree(d_count);
        if (h_pinned) cudaFreeHost(h_pinned);
        if (d_active) cudaFree(d_active);
        if (d_degrees) cudaFree(d_degrees);
    }

    void ensure_active(int32_t n) {
        if (active_capacity < n) {
            if (d_active) cudaFree(d_active);
            cudaMalloc(&d_active, (size_t)n * sizeof(int));
            active_capacity = n;
        }
    }

    void ensure_degrees(int32_t n) {
        if (degrees_capacity < n) {
            if (d_degrees) cudaFree(d_degrees);
            cudaMalloc(&d_degrees, (size_t)n * sizeof(int));
            degrees_capacity = n;
        }
    }
};

__device__ __forceinline__ bool is_edge_active(const uint32_t* __restrict__ edge_mask, int e) {
    return (__ldg(&edge_mask[e >> 5]) >> (e & 31)) & 1;
}

__global__ void compute_degrees_and_init_kernel(
    const edge_t* __restrict__ offsets,
    const vertex_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int* __restrict__ degrees,
    int* __restrict__ active,
    int n,
    int delta_multiplier
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);
    int deg = 0;

    for (int e = start; e < end; e++) {
        if (is_edge_active(edge_mask, e) && __ldg(&indices[e]) != v) {
            deg++;
        }
    }

    degrees[v] = (delta_multiplier == 2 && deg > 0x3FFFFFFF) ? 0x7FFFFFFF : deg * delta_multiplier;
    active[v] = 1;
}

__global__ void peel_kernel(
    const edge_t* __restrict__ offsets,
    const vertex_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int* __restrict__ degrees,
    int* __restrict__ active,
    int n,
    int k,
    int delta,
    int* __restrict__ changed
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (!active[v]) return;
    if (degrees[v] >= k) return;

    active[v] = 0;
    *changed = 1;

    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);
    for (int e = start; e < end; e++) {
        if (is_edge_active(edge_mask, e)) {
            int u = __ldg(&indices[e]);
            if (u != v && active[u]) {
                atomicSub(&degrees[u], delta);
            }
        }
    }
}

__global__ void set_active_from_core_numbers_kernel(
    const int* __restrict__ core_numbers,
    int* __restrict__ active,
    int n,
    int k
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    active[v] = (__ldg(&core_numbers[v]) >= k) ? 1 : 0;
}

__global__ void extract_edges_kernel(
    const edge_t* __restrict__ offsets,
    const vertex_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int* __restrict__ active,
    vertex_t* __restrict__ edge_srcs,
    vertex_t* __restrict__ edge_dsts,
    int n,
    int* __restrict__ count
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    if (!active[v]) return;

    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);
    for (int e = start; e < end; e++) {
        if (is_edge_active(edge_mask, e)) {
            int u = __ldg(&indices[e]);
            if (active[u]) {
                int pos = atomicAdd(count, 1);
                edge_srcs[pos] = v;
                edge_dsts[pos] = u;
            }
        }
    }
}

}  

std::size_t k_core_seg_mask(const graph32_t& graph,
                            std::size_t k,
                            int degree_type,
                            const int32_t* core_numbers,
                            int32_t* edge_srcs,
                            int32_t* edge_dsts,
                            std::size_t max_edges) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;
    cudaStream_t stream = 0;

    
    if (k > INT32_MAX) {
        return 0;
    }
    int k_int = static_cast<int>(k);

    cache.ensure_active(num_vertices);
    int* d_active = cache.d_active;

    if (core_numbers != nullptr) {
        if (num_vertices > 0) {
            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            set_active_from_core_numbers_kernel<<<grid, block, 0, stream>>>(
                core_numbers, d_active, num_vertices, k_int);
        }
    } else {
        int delta = (degree_type == 2) ? 2 : 1;

        cache.ensure_degrees(num_vertices);
        int* d_degrees = cache.d_degrees;

        if (num_vertices > 0) {
            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            compute_degrees_and_init_kernel<<<grid, block, 0, stream>>>(
                d_offsets, d_indices, d_edge_mask,
                d_degrees, d_active, num_vertices, delta);

            
            cache.h_pinned[0] = 1;  
            while (cache.h_pinned[0]) {
                cache.h_pinned[0] = 0;
                cudaMemsetAsync(cache.d_changed, 0, sizeof(int), stream);
                peel_kernel<<<grid, block, 0, stream>>>(
                    d_offsets, d_indices, d_edge_mask,
                    d_degrees, d_active, num_vertices, k_int, delta, cache.d_changed);
                cudaMemcpyAsync(&cache.h_pinned[0], cache.d_changed, sizeof(int),
                              cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
            }
        }
    }

    
    cudaMemsetAsync(cache.d_count, 0, sizeof(int), stream);
    if (num_vertices > 0) {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        extract_edges_kernel<<<grid, block, 0, stream>>>(
            d_offsets, d_indices, d_edge_mask, d_active,
            edge_srcs, edge_dsts, num_vertices, cache.d_count);
    }

    
    cudaMemcpyAsync(&cache.h_pinned[1], cache.d_count, sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return static_cast<std::size_t>(cache.h_pinned[1]);
}

}  
