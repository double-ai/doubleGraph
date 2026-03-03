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
#include <climits>

namespace aai {

namespace {

#define CHUNK_SIZE 32

struct Cache : Cacheable {
    int32_t* d_degree = nullptr;
    uint8_t* d_alive = nullptr;
    int32_t* d_worklist1 = nullptr;
    int32_t* d_worklist2 = nullptr;
    int32_t* d_lookup = nullptr;
    int32_t* d_counter = nullptr;
    int32_t* h_counter = nullptr;
    size_t alloc_vertices = 0;
    size_t alloc_lookup = 0;

    Cache() {
        cudaMalloc(&d_counter, sizeof(int32_t));
        cudaMallocHost(&h_counter, sizeof(int32_t));
    }

    ~Cache() override {
        if (d_degree) cudaFree(d_degree);
        if (d_alive) cudaFree(d_alive);
        if (d_worklist1) cudaFree(d_worklist1);
        if (d_worklist2) cudaFree(d_worklist2);
        if (d_lookup) cudaFree(d_lookup);
        if (d_counter) cudaFree(d_counter);
        if (h_counter) cudaFreeHost(h_counter);
    }

    void ensure_capacity(size_t need_vertices, size_t need_lookup) {
        if (need_vertices > alloc_vertices) {
            if (d_degree) cudaFree(d_degree);
            if (d_alive) cudaFree(d_alive);
            if (d_worklist1) cudaFree(d_worklist1);
            if (d_worklist2) cudaFree(d_worklist2);
            alloc_vertices = need_vertices;
            cudaMalloc(&d_degree, alloc_vertices * sizeof(int32_t));
            cudaMalloc(&d_alive, alloc_vertices * sizeof(uint8_t));
            cudaMalloc(&d_worklist1, alloc_vertices * sizeof(int32_t));
            cudaMalloc(&d_worklist2, alloc_vertices * sizeof(int32_t));
        }
        if (need_lookup > alloc_lookup) {
            if (d_lookup) cudaFree(d_lookup);
            alloc_lookup = need_lookup;
            cudaMalloc(&d_lookup, alloc_lookup * sizeof(int32_t));
        }
    }
};





__global__ void build_lookup_kernel(
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ lookup,
    int32_t num_vertices,
    int32_t num_chunks
) {
    int chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_id >= num_chunks) return;

    int eid = chunk_id * CHUNK_SIZE;

    
    int lo = 0, hi = num_vertices;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(&offsets[mid]) <= eid) lo = mid + 1;
        else hi = mid;
    }
    lookup[chunk_id] = lo - 1;
}





__global__ void init_degrees_from_offsets_kernel(
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ degree,
    uint8_t* __restrict__ alive,
    int32_t num_vertices,
    int32_t multiplier
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        int32_t raw = __ldg(&offsets[tid + 1]) - __ldg(&offsets[tid]);
        degree[tid] = (multiplier == 2 && raw > 0x3FFFFFFF) ? 0x7FFFFFFF : raw * multiplier;
        alive[tid] = 1;
    }
}


__global__ void fix_self_loops_lookup_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ lookup,
    int32_t* __restrict__ degree,
    int32_t num_vertices,
    int32_t num_edges,
    int32_t multiplier
) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    
    int chunk_id = eid / CHUNK_SIZE;
    int base_v = 0;
    if (lane == 0 && eid < num_edges) {
        base_v = __ldg(&lookup[chunk_id]);
    }
    base_v = __shfl_sync(0xffffffff, base_v, 0);

    if (eid < num_edges) {
        
        int v = base_v;
        while (v < num_vertices - 1 && __ldg(&offsets[v + 1]) <= eid) v++;

        if (v == __ldg(&indices[eid])) {
            atomicSub(&degree[v], multiplier);
        }
    }
}


__global__ void build_worklist_kernel(
    int32_t* __restrict__ degree,
    uint8_t* __restrict__ alive,
    int32_t* __restrict__ worklist,
    int32_t* __restrict__ wl_size,
    int32_t num_vertices,
    int32_t k
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    if (degree[v] < k) {
        alive[v] = 0;
        int pos = atomicAdd(wl_size, 1);
        worklist[pos] = v;
    }
}


__global__ void mark_core_numbers_kernel(
    const int32_t* __restrict__ core_numbers,
    uint8_t* __restrict__ alive,
    int32_t num_vertices,
    int32_t k
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        alive[tid] = (__ldg(&core_numbers[tid]) >= k) ? 1 : 0;
    }
}




__global__ void peel_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ degree,
    uint8_t* __restrict__ alive,
    const int32_t* __restrict__ curr_wl,
    int32_t curr_wl_size,
    int32_t* __restrict__ next_wl,
    int32_t* __restrict__ next_wl_size,
    int32_t k,
    int32_t decrement
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < curr_wl_size; i += stride) {
        int v = curr_wl[i];
        int start = __ldg(&offsets[v]);
        int end = __ldg(&offsets[v + 1]);

        for (int j = start; j < end; j++) {
            int u = __ldg(&indices[j]);
            if (u == v) continue;
            if (!alive[u]) continue;

            int old_deg = atomicSub(&degree[u], decrement);
            if (old_deg >= k && old_deg - decrement < k) {
                alive[u] = 0;
                int pos = atomicAdd(next_wl_size, 1);
                next_wl[pos] = u;
            }
        }
    }
}




__global__ void extract_edges_lookup_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ lookup,
    const uint8_t* __restrict__ alive,
    int32_t* __restrict__ edge_srcs,
    int32_t* __restrict__ edge_dsts,
    int32_t* __restrict__ num_output_edges,
    int32_t num_vertices,
    int32_t num_edges
) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    
    int chunk_id = eid / CHUNK_SIZE;
    int base_v = 0;
    if (lane == 0 && eid < num_edges) {
        base_v = __ldg(&lookup[chunk_id]);
    }
    base_v = __shfl_sync(0xffffffff, base_v, 0);

    bool emit = false;
    int v = 0, u = 0;

    if (eid < num_edges) {
        
        v = base_v;
        while (v < num_vertices - 1 && __ldg(&offsets[v + 1]) <= eid) v++;
        u = __ldg(&indices[eid]);
        emit = alive[v] && alive[u];
    }

    
    unsigned int mask = __ballot_sync(0xffffffff, emit);
    int warp_count = __popc(mask);

    int warp_base = 0;
    if (lane == 0 && warp_count > 0) {
        warp_base = atomicAdd(num_output_edges, warp_count);
    }
    warp_base = __shfl_sync(0xffffffff, warp_base, 0);

    if (emit) {
        int my_pos = warp_base + __popc(mask & ((1u << lane) - 1));
        edge_srcs[my_pos] = v;
        edge_dsts[my_pos] = u;
    }
}

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

    
    if (k > INT32_MAX) return 0;
    int32_t k32 = static_cast<int32_t>(k);

    size_t need_vertices = static_cast<size_t>(num_vertices);
    size_t need_lookup = (static_cast<size_t>(num_edges) + CHUNK_SIZE - 1) / CHUNK_SIZE;
    cache.ensure_capacity(need_vertices, need_lookup);

    int block = 256;

    
    int num_chunks = (num_edges + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int grid_chunks = (num_chunks + block - 1) / block;
    build_lookup_kernel<<<grid_chunks, block>>>(d_offsets, cache.d_lookup, num_vertices, num_chunks);

    if (core_numbers != nullptr) {
        
        int grid_v = (num_vertices + block - 1) / block;
        mark_core_numbers_kernel<<<grid_v, block>>>(core_numbers, cache.d_alive, num_vertices, k32);
    } else {
        int32_t multiplier = (degree_type == 2) ? 2 : 1;
        int32_t decrement = (degree_type == 2) ? 2 : 1;
        int grid_v = (num_vertices + block - 1) / block;
        int grid_e = (num_edges + block - 1) / block;

        
        init_degrees_from_offsets_kernel<<<grid_v, block>>>(
            d_offsets, cache.d_degree, cache.d_alive, num_vertices, multiplier);
        fix_self_loops_lookup_kernel<<<grid_e, block>>>(
            d_offsets, d_indices, cache.d_lookup, cache.d_degree, num_vertices, num_edges, multiplier);

        cudaMemset(cache.d_counter, 0, sizeof(int32_t));
        build_worklist_kernel<<<grid_v, block>>>(
            cache.d_degree, cache.d_alive, cache.d_worklist1, cache.d_counter, num_vertices, k32);

        cudaMemcpy(cache.h_counter, cache.d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
        int32_t wl_size = *cache.h_counter;

        int32_t* curr_wl = cache.d_worklist1;
        int32_t* next_wl = cache.d_worklist2;

        
        while (wl_size > 0) {
            cudaMemset(cache.d_counter, 0, sizeof(int32_t));
            int grid_peel = (wl_size + block - 1) / block;
            if (grid_peel > 4096) grid_peel = 4096;
            peel_kernel<<<grid_peel, block>>>(
                d_offsets, d_indices, cache.d_degree, cache.d_alive,
                curr_wl, wl_size, next_wl, cache.d_counter,
                k32, decrement);
            cudaMemcpy(cache.h_counter, cache.d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
            wl_size = *cache.h_counter;
            int32_t* temp = curr_wl;
            curr_wl = next_wl;
            next_wl = temp;
        }
    }

    
    cudaMemset(cache.d_counter, 0, sizeof(int32_t));
    int grid_extract = (num_edges + block - 1) / block;
    extract_edges_lookup_kernel<<<grid_extract, block>>>(
        d_offsets, d_indices, cache.d_lookup, cache.d_alive,
        edge_srcs, edge_dsts, cache.d_counter, num_vertices, num_edges);

    cudaMemcpy(cache.h_counter, cache.d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
    return static_cast<std::size_t>(*cache.h_counter);
}

}  
