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

namespace aai {

namespace {

__device__ __forceinline__ bool is_edge_active(const uint32_t* __restrict__ edge_mask, int32_t e) {
    return (edge_mask[e >> 5] >> (e & 31)) & 1;
}

__device__ __forceinline__ bool is_removed(const uint32_t* __restrict__ removed_mask, int32_t v) {
    return (removed_mask[v >> 5] >> (v & 31)) & 1;
}


__global__ void compute_degrees_init_frontier_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ degrees,
    uint32_t* __restrict__ removed_mask,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_size,
    int32_t num_vertices,
    int32_t delta,
    int32_t threshold)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    bool is_rem = false;

    if (v < num_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int deg = 0;

        for (int e = start; e < end; e++) {
            if (is_edge_active(edge_mask, e) && indices[e] != v) {
                deg++;
            }
        }

        int d = (delta == 2 && deg > 0x3FFFFFFF) ? 0x7FFFFFFF : deg * delta;
        degrees[v] = d;

        if (d < threshold) {
            is_rem = true;
            int pos = atomicAdd(frontier_size, 1);
            frontier[pos] = v;
        }
    }

    
    unsigned mask = __ballot_sync(0xffffffff, is_rem);
    if ((threadIdx.x & 31) == 0) {
        int word_idx = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int mask_words = (num_vertices + 31) >> 5;
        if (word_idx < mask_words) {
            removed_mask[word_idx] = mask;
        }
    }
}


__global__ void scatter_in_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ degrees,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int start = offsets[v];
    int end = offsets[v + 1];

    for (int e = start; e < end; e++) {
        if (is_edge_active(edge_mask, e)) {
            int32_t u = indices[e];
            if (u != v) atomicAdd(&degrees[u], 1);
        }
    }
}


__global__ void build_frontier_bitmask_kernel(
    const int32_t* __restrict__ degrees,
    uint32_t* __restrict__ removed_mask,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_size,
    int32_t threshold,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    bool is_rem = false;
    if (v < num_vertices && degrees[v] < threshold) {
        is_rem = true;
        int pos = atomicAdd(frontier_size, 1);
        frontier[pos] = v;
    }

    unsigned mask = __ballot_sync(0xffffffff, is_rem);
    if ((threadIdx.x & 31) == 0) {
        int word_idx = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int mask_words = (num_vertices + 31) >> 5;
        if (word_idx < mask_words) {
            removed_mask[word_idx] = mask;
        }
    }
}


__global__ void peel_frontier_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ degrees,
    uint32_t* __restrict__ removed_mask,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t threshold,
    int32_t delta)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int v = frontier[tid];
    int start = offsets[v];
    int end = offsets[v + 1];

    for (int e = start; e < end; e++) {
        if (is_edge_active(edge_mask, e)) {
            int32_t u = indices[e];
            if (u != v) {
                int old_deg = atomicSub(&degrees[u], delta);
                if (old_deg >= threshold && (old_deg - delta) < threshold) {
                    atomicOr(&removed_mask[u >> 5], 1u << (u & 31));
                    int pos = atomicAdd(next_frontier_size, 1);
                    next_frontier[pos] = u;
                }
            }
        }
    }
}


__global__ void compute_removed_mask_from_cn_kernel(
    const int32_t* __restrict__ core_numbers,
    uint32_t* __restrict__ removed_mask,
    int32_t k, int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    bool is_rem = (v < num_vertices) ? (core_numbers[v] < k) : false;

    unsigned mask = __ballot_sync(0xffffffff, is_rem);
    if ((threadIdx.x & 31) == 0) {
        int word_idx = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int mask_words = (num_vertices + 31) >> 5;
        if (word_idx < mask_words) {
            removed_mask[word_idx] = mask;
        }
    }
}


__global__ void extract_edges_atomic_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const uint32_t* __restrict__ removed_mask,
    int32_t* __restrict__ counter,
    int32_t* __restrict__ edge_srcs,
    int32_t* __restrict__ edge_dsts,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices || is_removed(removed_mask, v)) return;

    int start = offsets[v];
    int end = offsets[v + 1];

    
    int count = 0;
    for (int e = start; e < end; e++) {
        if (is_edge_active(edge_mask, e) && !is_removed(removed_mask, indices[e])) {
            count++;
        }
    }

    if (count == 0) return;

    
    int pos = atomicAdd(counter, count);

    
    for (int e = start; e < end; e++) {
        if (is_edge_active(edge_mask, e) && !is_removed(removed_mask, indices[e])) {
            edge_srcs[pos] = v;
            edge_dsts[pos] = indices[e];
            pos++;
        }
    }
}

struct Cache : Cacheable {
    
    int32_t* d_frontier_size = nullptr;
    int32_t* d_next_frontier_size = nullptr;
    int32_t* d_counter = nullptr;
    
    int32_t* h_pinned = nullptr;

    
    uint32_t* d_removed_mask = nullptr;
    int32_t d_removed_mask_capacity = 0;

    int32_t* d_degrees = nullptr;
    int32_t d_degrees_capacity = 0;

    int32_t* d_frontier = nullptr;
    int32_t d_frontier_capacity = 0;

    int32_t* d_next_frontier = nullptr;
    int32_t d_next_frontier_capacity = 0;

    Cache() {
        cudaMalloc(&d_frontier_size, sizeof(int32_t));
        cudaMalloc(&d_next_frontier_size, sizeof(int32_t));
        cudaMalloc(&d_counter, sizeof(int32_t));
        cudaMallocHost(&h_pinned, 8 * sizeof(int32_t));
    }

    ~Cache() override {
        if (d_frontier_size) cudaFree(d_frontier_size);
        if (d_next_frontier_size) cudaFree(d_next_frontier_size);
        if (d_counter) cudaFree(d_counter);
        if (h_pinned) cudaFreeHost(h_pinned);
        if (d_removed_mask) cudaFree(d_removed_mask);
        if (d_degrees) cudaFree(d_degrees);
        if (d_frontier) cudaFree(d_frontier);
        if (d_next_frontier) cudaFree(d_next_frontier);
    }

    void ensure_removed_mask(int32_t mask_words) {
        if (d_removed_mask_capacity < mask_words) {
            if (d_removed_mask) cudaFree(d_removed_mask);
            cudaMalloc(&d_removed_mask, mask_words * sizeof(uint32_t));
            d_removed_mask_capacity = mask_words;
        }
    }

    void ensure_vertex_buffers(int32_t num_vertices) {
        if (d_degrees_capacity < num_vertices) {
            if (d_degrees) cudaFree(d_degrees);
            cudaMalloc(&d_degrees, num_vertices * sizeof(int32_t));
            d_degrees_capacity = num_vertices;
        }
        if (d_frontier_capacity < num_vertices) {
            if (d_frontier) cudaFree(d_frontier);
            cudaMalloc(&d_frontier, num_vertices * sizeof(int32_t));
            d_frontier_capacity = num_vertices;
        }
        if (d_next_frontier_capacity < num_vertices) {
            if (d_next_frontier) cudaFree(d_next_frontier);
            cudaMalloc(&d_next_frontier, num_vertices * sizeof(int32_t));
            d_next_frontier_capacity = num_vertices;
        }
    }
};

}  

std::size_t k_core_mask(const graph32_t& graph,
                        std::size_t k,
                        int degree_type,
                        const int32_t* core_numbers,
                        int32_t* edge_srcs,
                        int32_t* edge_dsts,
                        std::size_t max_edges) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    
    
    if (k > static_cast<std::size_t>(INT32_MAX)) {
        return 0;
    }
    int32_t k32 = static_cast<int32_t>(k);

    int32_t mask_words = (num_vertices + 31) >> 5;
    cache.ensure_removed_mask(mask_words);
    uint32_t* d_removed_mask = cache.d_removed_mask;

    if (core_numbers != nullptr) {
        
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 0)
            compute_removed_mask_from_cn_kernel<<<grid, block>>>(
                core_numbers, d_removed_mask, k32, num_vertices);
    } else {
        
        int32_t delta = (degree_type == 2) ? 2 : 1;
        int32_t threshold = k32;

        cache.ensure_vertex_buffers(num_vertices);
        int32_t* d_degrees = cache.d_degrees;
        int32_t* d_frontier = cache.d_frontier;
        int32_t* d_next_frontier = cache.d_next_frontier;

        cudaMemset(cache.d_frontier_size, 0, sizeof(int32_t));

        if (degree_type == 0) {
            
            cudaMemset(d_degrees, 0, num_vertices * sizeof(int32_t));

            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            if (grid > 0)
                scatter_in_degrees_kernel<<<grid, block>>>(
                    d_offsets, d_indices, d_edge_mask, d_degrees, num_vertices);
            if (grid > 0)
                build_frontier_bitmask_kernel<<<grid, block>>>(
                    d_degrees, d_removed_mask, d_frontier,
                    cache.d_frontier_size, threshold, num_vertices);
        } else {
            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            if (grid > 0)
                compute_degrees_init_frontier_kernel<<<grid, block>>>(
                    d_offsets, d_indices, d_edge_mask, d_degrees, d_removed_mask,
                    d_frontier, cache.d_frontier_size, num_vertices, delta, threshold);
        }

        int32_t h_frontier_size = 0;
        cudaMemcpy(&h_frontier_size, cache.d_frontier_size, sizeof(int32_t), cudaMemcpyDeviceToHost);

        while (h_frontier_size > 0) {
            cudaMemset(cache.d_next_frontier_size, 0, sizeof(int32_t));

            int block = 256;
            int grid = (h_frontier_size + block - 1) / block;
            if (grid > 0)
                peel_frontier_kernel<<<grid, block>>>(
                    d_offsets, d_indices, d_edge_mask, d_degrees, d_removed_mask,
                    d_frontier, h_frontier_size, d_next_frontier,
                    cache.d_next_frontier_size, threshold, delta);

            cudaMemcpy(&h_frontier_size, cache.d_next_frontier_size, sizeof(int32_t), cudaMemcpyDeviceToHost);

            int32_t* tmp = d_frontier;
            d_frontier = d_next_frontier;
            d_next_frontier = tmp;
        }
    }

    
    cudaMemset(cache.d_counter, 0, sizeof(int32_t));
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 0)
            extract_edges_atomic_kernel<<<grid, block>>>(
                d_offsets, d_indices, d_edge_mask, d_removed_mask,
                cache.d_counter, edge_srcs, edge_dsts, num_vertices);
    }

    int32_t h_total_edges = 0;
    cudaMemcpy(&h_total_edges, cache.d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);

    return static_cast<std::size_t>(h_total_edges);
}

}  
