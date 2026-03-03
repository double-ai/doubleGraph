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





__device__ __forceinline__ bool bitmap_get(const uint32_t* bitmap, int32_t v) {
    return (bitmap[v >> 5] >> (v & 31)) & 1;
}


__global__ void compute_degrees(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ degrees,
    int32_t num_vertices,
    int32_t multiplier
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t deg = end - start;
    for (int32_t i = start; i < end; i++) {
        if (indices[i] == v) deg--;
    }
    degrees[v] = deg * multiplier;
}


__global__ void init_frontier(
    const int32_t* __restrict__ degrees,
    uint8_t* __restrict__ removed,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_size,
    int32_t num_vertices,
    int32_t k
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    if (degrees[v] < k) {
        removed[v] = 1;
        int pos = atomicAdd(frontier_size, 1);
        frontier[pos] = v;
    }
}


__global__ void peel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ degrees,
    uint8_t* __restrict__ removed,
    const int32_t* __restrict__ frontier_in,
    int32_t frontier_in_size,
    int32_t* __restrict__ frontier_out,
    int32_t* __restrict__ frontier_out_size,
    int32_t k,
    int32_t decrement
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_in_size) return;
    int32_t v = frontier_in[tid];
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    for (int32_t i = start; i < end; i++) {
        int32_t u = indices[i];
        if (u == v) continue;
        if (removed[u]) continue;
        int32_t old_deg = atomicSub(&degrees[u], decrement);
        if (old_deg >= k && (old_deg - decrement) < k) {
            removed[u] = 1;
            int pos = atomicAdd(frontier_out_size, 1);
            frontier_out[pos] = u;
        }
    }
}


__global__ void build_bitmap_from_removed(
    const uint8_t* __restrict__ removed,
    uint32_t* __restrict__ bitmap,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = v >> 5;
    int lane = threadIdx.x & 31;
    bool in_kcore = (v < num_vertices) && !removed[v];
    uint32_t mask = __ballot_sync(0xFFFFFFFF, in_kcore);
    if (lane == 0 && warp_id < ((num_vertices + 31) >> 5)) {
        bitmap[warp_id] = mask;
    }
}


__global__ void build_bitmap_from_core_numbers(
    const int32_t* __restrict__ core_numbers,
    uint32_t* __restrict__ bitmap,
    int32_t num_vertices,
    int32_t k
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = v >> 5;
    int lane = threadIdx.x & 31;
    bool in_kcore = (v < num_vertices) && (core_numbers[v] >= k);
    uint32_t mask = __ballot_sync(0xFFFFFFFF, in_kcore);
    if (lane == 0 && warp_id < ((num_vertices + 31) >> 5)) {
        bitmap[warp_id] = mask;
    }
}



__global__ void extract_kcore_edges_per_edge(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ bitmap,
    int32_t* __restrict__ edge_srcs,
    int32_t* __restrict__ edge_dsts,
    int32_t* __restrict__ edge_counter,
    int32_t num_edges,
    int32_t num_vertices
) {
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    bool include = false;
    int32_t src = 0, dst = 0;

    if (edge_idx < num_edges) {
        dst = indices[edge_idx];

        
        int lo = 0, hi = num_vertices - 1;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (__ldg(&offsets[mid + 1]) <= edge_idx)
                lo = mid + 1;
            else
                hi = mid;
        }
        src = lo;

        include = bitmap_get(bitmap, src) && bitmap_get(bitmap, dst);
    }

    
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, include);

    if (ballot > 0) {
        int count = __popc(ballot);

        
        int base;
        if (lane == 0) {
            base = atomicAdd(edge_counter, count);
        }
        
        base = __shfl_sync(0xFFFFFFFF, base, 0);

        if (include) {
            
            unsigned int lower_mask = (1u << lane) - 1;
            int my_offset = __popc(ballot & lower_mask);

            edge_srcs[base + my_offset] = src;
            edge_dsts[base + my_offset] = dst;
        }
    }
}


__global__ void extract_kcore_edges_vertex(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ bitmap,
    int32_t* __restrict__ edge_srcs,
    int32_t* __restrict__ edge_dsts,
    int32_t* __restrict__ edge_counter,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices || !bitmap_get(bitmap, v)) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    int32_t count = 0;
    for (int32_t i = start; i < end; i++) {
        if (bitmap_get(bitmap, indices[i])) count++;
    }
    if (count == 0) return;

    int32_t base = atomicAdd(edge_counter, count);
    for (int32_t i = start; i < end; i++) {
        int32_t u = indices[i];
        if (bitmap_get(bitmap, u)) {
            edge_srcs[base] = v;
            edge_dsts[base] = u;
            base++;
        }
    }
}





struct Cache : Cacheable {
    uint32_t* bitmap = nullptr;
    int32_t* degrees = nullptr;
    uint8_t* removed = nullptr;
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* counter = nullptr;

    int32_t bitmap_capacity = 0;
    int32_t degrees_capacity = 0;
    int32_t removed_capacity = 0;
    int32_t frontier_a_capacity = 0;
    int32_t frontier_b_capacity = 0;
    bool counter_allocated = false;

    void ensure(int32_t num_vertices) {
        int32_t bitmap_words = (num_vertices + 31) / 32;
        if (bitmap_capacity < bitmap_words) {
            if (bitmap) cudaFree(bitmap);
            cudaMalloc(&bitmap, bitmap_words * sizeof(uint32_t));
            bitmap_capacity = bitmap_words;
        }
        if (degrees_capacity < num_vertices) {
            if (degrees) cudaFree(degrees);
            cudaMalloc(&degrees, num_vertices * sizeof(int32_t));
            degrees_capacity = num_vertices;
        }
        if (removed_capacity < num_vertices) {
            if (removed) cudaFree(removed);
            cudaMalloc(&removed, num_vertices * sizeof(uint8_t));
            removed_capacity = num_vertices;
        }
        if (frontier_a_capacity < num_vertices) {
            if (frontier_a) cudaFree(frontier_a);
            cudaMalloc(&frontier_a, num_vertices * sizeof(int32_t));
            frontier_a_capacity = num_vertices;
        }
        if (frontier_b_capacity < num_vertices) {
            if (frontier_b) cudaFree(frontier_b);
            cudaMalloc(&frontier_b, num_vertices * sizeof(int32_t));
            frontier_b_capacity = num_vertices;
        }
        if (!counter_allocated) {
            cudaMalloc(&counter, sizeof(int32_t));
            counter_allocated = true;
        }
    }

    ~Cache() override {
        if (bitmap) cudaFree(bitmap);
        if (degrees) cudaFree(degrees);
        if (removed) cudaFree(removed);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (counter) cudaFree(counter);
    }
};

}  





std::size_t k_core(const graph32_t& graph,
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

    cache.ensure(num_vertices);

    uint32_t* d_bitmap = cache.bitmap;

    if (core_numbers != nullptr) {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 0)
            build_bitmap_from_core_numbers<<<grid, block>>>(
                core_numbers, d_bitmap, num_vertices, (int32_t)k);
    } else {
        int32_t multiplier = (degree_type == 2) ? 2 : 1;
        int32_t decrement = multiplier;

        int32_t* d_degrees = cache.degrees;
        uint8_t* d_removed = cache.removed;
        int32_t* d_frontier_a = cache.frontier_a;
        int32_t* d_frontier_b = cache.frontier_b;
        int32_t* d_counter = cache.counter;

        cudaMemset(d_removed, 0, num_vertices);
        cudaMemset(d_counter, 0, sizeof(int32_t));

        {
            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            if (grid > 0)
                compute_degrees<<<grid, block>>>(
                    d_offsets, d_indices, d_degrees, num_vertices, multiplier);
        }

        {
            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            if (grid > 0)
                init_frontier<<<grid, block>>>(
                    d_degrees, d_removed, d_frontier_a, d_counter, num_vertices, (int32_t)k);
        }

        int32_t h_frontier_size;
        cudaMemcpy(&h_frontier_size, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);

        int32_t* cur_frontier = d_frontier_a;
        int32_t* next_frontier = d_frontier_b;
        int32_t cur_size = h_frontier_size;

        while (cur_size > 0) {
            cudaMemset(d_counter, 0, sizeof(int32_t));
            {
                int block = 256;
                int grid = (cur_size + block - 1) / block;
                peel<<<grid, block>>>(
                    d_offsets, d_indices, d_degrees, d_removed,
                    cur_frontier, cur_size, next_frontier, d_counter,
                    (int32_t)k, decrement);
            }
            cudaMemcpy(&h_frontier_size, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
            int32_t* tmp = cur_frontier;
            cur_frontier = next_frontier;
            next_frontier = tmp;
            cur_size = h_frontier_size;
        }

        {
            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            if (grid > 0)
                build_bitmap_from_removed<<<grid, block>>>(
                    d_removed, d_bitmap, num_vertices);
        }
    }

    
    int32_t* d_edge_counter = cache.counter;
    cudaMemset(d_edge_counter, 0, sizeof(int32_t));

    if (num_edges > 100000) {
        int block = 256;
        int grid = (num_edges + block - 1) / block;
        if (grid > 0)
            extract_kcore_edges_per_edge<<<grid, block>>>(
                d_offsets, d_indices, d_bitmap, edge_srcs, edge_dsts,
                d_edge_counter, num_edges, num_vertices);
    } else {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 0)
            extract_kcore_edges_vertex<<<grid, block>>>(
                d_offsets, d_indices, d_bitmap, edge_srcs, edge_dsts,
                d_edge_counter, num_vertices);
    }

    int32_t h_edge_count;
    cudaMemcpy(&h_edge_count, d_edge_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);

    return (std::size_t)h_edge_count;
}

}  
