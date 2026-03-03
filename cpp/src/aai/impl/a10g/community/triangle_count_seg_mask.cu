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
#include <cub/cub.cuh>
#include <cstdint>
#include <cstddef>

namespace aai {

namespace {





__device__ __forceinline__ bool is_edge_active(const uint32_t* edge_mask, int edge_idx) {
    return (edge_mask[edge_idx >> 5] >> (edge_idx & 31)) & 1;
}




__global__ void compute_masked_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ masked_deg,
    int32_t num_vertices
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;
    int start = offsets[u];
    int end = offsets[u + 1];
    if (start >= end) { masked_deg[u] = 0; return; }
    int count = 0;
    int first_word = start >> 5;
    int last_word = (end - 1) >> 5;
    for (int w = first_word; w <= last_word; w++) {
        uint32_t word = edge_mask[w];
        if (w == first_word) word &= ~((1u << (start & 31)) - 1);
        if (w == last_word) { int hi = end & 31; if (hi > 0) word &= (1u << hi) - 1; }
        count += __popc(word);
    }
    masked_deg[u] = count;
}




__global__ void count_dodg_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ masked_deg,
    int32_t* __restrict__ dodg_deg,
    int32_t num_vertices
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_vertices) return;
    int u = warp_id;
    int start = offsets[u], end = offsets[u + 1];
    int deg_u = masked_deg[u];
    int count = 0;
    for (int chunk = start; chunk < end; chunk += 32) {
        int e = chunk + lane;
        bool is_dodg = false;
        if (e < end && is_edge_active(edge_mask, e)) {
            int v = indices[e];
            int deg_v = masked_deg[v];
            if (deg_u < deg_v || (deg_u == deg_v && u < v)) is_dodg = true;
        }
        count += __popc(__ballot_sync(0xffffffff, is_dodg));
    }
    if (lane == 0) dodg_deg[u] = count;
}




__global__ void build_dodg_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ masked_deg,
    const int32_t* __restrict__ dodg_offsets,
    int32_t* __restrict__ dodg_indices,
    int32_t num_vertices
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_vertices) return;
    int u = warp_id;
    int start = offsets[u], end = offsets[u + 1];
    int deg_u = masked_deg[u];
    int base = dodg_offsets[u];
    int write_pos = 0;
    for (int chunk = start; chunk < end; chunk += 32) {
        int e = chunk + lane;
        bool is_dodg = false; int v = -1;
        if (e < end && is_edge_active(edge_mask, e)) {
            v = indices[e];
            int deg_v = masked_deg[v];
            if (deg_u < deg_v || (deg_u == deg_v && u < v)) is_dodg = true;
        }
        unsigned int ballot = __ballot_sync(0xffffffff, is_dodg);
        int prefix = __popc(ballot & ((1u << lane) - 1));
        if (is_dodg) dodg_indices[base + write_pos + prefix] = v;
        write_pos += __popc(ballot);
    }
}




__global__ void triangle_count_kernel(
    const int32_t* __restrict__ dodg_offsets,
    const int32_t* __restrict__ dodg_indices,
    int32_t* __restrict__ counts,
    int32_t num_vertices
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_vertices) return;

    int u = warp_id;
    int u_start = dodg_offsets[u];
    int u_end = dodg_offsets[u + 1];
    int u_deg = u_end - u_start;
    if (u_deg == 0) return;

    int total_tri = 0;

    for (int ei = u_start; ei < u_end; ei++) {
        int v = dodg_indices[ei];
        int v_start = dodg_offsets[v];
        int v_end = dodg_offsets[v + 1];
        int v_deg = v_end - v_start;
        if (v_deg == 0) continue;

        
        const int32_t* A = dodg_indices + u_start;
        int na = u_deg;
        const int32_t* B = dodg_indices + v_start;
        int nb = v_deg;
        if (na > nb) {
            const int32_t* t = A; A = B; B = t;
            int ti = na; na = nb; nb = ti;
        }

        int edge_count = 0;
        int lb = 0;  

        for (int chunk = 0; chunk < na; chunk += 32) {
            int i = chunk + lane;
            int32_t w = 0;
            bool found = false;
            if (i < na) {
                w = A[i];
                
                int lo = lb, hi = nb;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (B[mid] < w) lo = mid + 1;
                    else hi = mid;
                }
                found = (lo < nb && B[lo] == w);
                lb = lo;  
            }
            unsigned int ballot = __ballot_sync(0xffffffff, found);
            edge_count += __popc(ballot);
            if (found) atomicAdd(&counts[w], 1);
        }

        if (lane == 0 && edge_count > 0) atomicAdd(&counts[v], edge_count);
        total_tri += edge_count;
    }

    if (lane == 0 && total_tri > 0) atomicAdd(&counts[u], total_tri);
}

__global__ void gather_counts_kernel(
    const int32_t* __restrict__ all_counts,
    const int32_t* __restrict__ vertices,
    int32_t* __restrict__ output, int32_t n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) output[i] = all_counts[vertices[i]];
}




struct Cache : Cacheable {
    int32_t* masked_deg = nullptr;
    int32_t* dodg_deg = nullptr;
    int32_t* dodg_offsets = nullptr;
    int32_t* all_counts = nullptr;
    int32_t* dodg_indices = nullptr;
    void* scan_temp = nullptr;

    int64_t masked_deg_capacity = 0;
    int64_t dodg_deg_capacity = 0;
    int64_t dodg_offsets_capacity = 0;
    int64_t all_counts_capacity = 0;
    int64_t dodg_indices_capacity = 0;
    size_t scan_temp_capacity = 0;

    void ensure(int32_t num_vertices, int32_t num_edges) {
        int64_t nv = num_vertices;
        if (masked_deg_capacity < nv) {
            if (masked_deg) cudaFree(masked_deg);
            cudaMalloc(&masked_deg, nv * sizeof(int32_t));
            masked_deg_capacity = nv;
        }
        if (dodg_deg_capacity < nv) {
            if (dodg_deg) cudaFree(dodg_deg);
            cudaMalloc(&dodg_deg, nv * sizeof(int32_t));
            dodg_deg_capacity = nv;
        }
        if (dodg_offsets_capacity < nv + 1) {
            if (dodg_offsets) cudaFree(dodg_offsets);
            cudaMalloc(&dodg_offsets, (nv + 1) * sizeof(int32_t));
            dodg_offsets_capacity = nv + 1;
        }
        if (all_counts_capacity < nv) {
            if (all_counts) cudaFree(all_counts);
            cudaMalloc(&all_counts, nv * sizeof(int32_t));
            all_counts_capacity = nv;
        }
        int64_t dodg_alloc = (int64_t)(num_edges / 2) + 1;
        if (dodg_indices_capacity < dodg_alloc) {
            if (dodg_indices) cudaFree(dodg_indices);
            cudaMalloc(&dodg_indices, dodg_alloc * sizeof(int32_t));
            dodg_indices_capacity = dodg_alloc;
        }
        size_t scan_bytes = 0;
        cub::DeviceScan::InclusiveSum(nullptr, scan_bytes,
            (int32_t*)nullptr, (int32_t*)nullptr, num_vertices);
        if (scan_temp_capacity < scan_bytes) {
            if (scan_temp) cudaFree(scan_temp);
            cudaMalloc(&scan_temp, scan_bytes);
            scan_temp_capacity = scan_bytes;
        }
    }

    ~Cache() override {
        if (masked_deg) cudaFree(masked_deg);
        if (dodg_deg) cudaFree(dodg_deg);
        if (dodg_offsets) cudaFree(dodg_offsets);
        if (all_counts) cudaFree(all_counts);
        if (dodg_indices) cudaFree(dodg_indices);
        if (scan_temp) cudaFree(scan_temp);
    }
};

}  

void triangle_count_seg_mask(const graph32_t& graph,
                             int32_t* counts,
                             const int32_t* vertices,
                             std::size_t n_vertices) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cudaStream_t stream = 0;

    if (num_vertices == 0) {
        if (vertices != nullptr && n_vertices > 0) {
            cudaMemsetAsync(counts, 0, n_vertices * sizeof(int32_t), stream);
        }
        return;
    }

    cache.ensure(num_vertices, num_edges);

    
    compute_masked_degrees_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
        d_offsets, d_edge_mask, cache.masked_deg, num_vertices);

    
    {
        int w = 8, b = w * 32;
        count_dodg_warp_kernel<<<(num_vertices + w - 1) / w, b, 0, stream>>>(
            d_offsets, d_indices, d_edge_mask, cache.masked_deg,
            cache.dodg_deg, num_vertices);
    }

    
    cudaMemsetAsync(cache.dodg_offsets, 0, sizeof(int32_t), stream);
    {
        size_t scan_bytes = 0;
        cub::DeviceScan::InclusiveSum(nullptr, scan_bytes,
            cache.dodg_deg, cache.dodg_offsets + 1, num_vertices);
        cub::DeviceScan::InclusiveSum(cache.scan_temp, scan_bytes,
            cache.dodg_deg, cache.dodg_offsets + 1, num_vertices, stream);
    }

    
    {
        int w = 8, b = w * 32;
        build_dodg_warp_kernel<<<(num_vertices + w - 1) / w, b, 0, stream>>>(
            d_offsets, d_indices, d_edge_mask, cache.masked_deg,
            cache.dodg_offsets, cache.dodg_indices, num_vertices);
    }

    
    int32_t* d_all_counts;
    if (vertices != nullptr && n_vertices > 0) {
        d_all_counts = cache.all_counts;
    } else {
        d_all_counts = counts;
    }
    cudaMemsetAsync(d_all_counts, 0, num_vertices * sizeof(int32_t), stream);
    {
        int w = 8, b = w * 32;
        triangle_count_kernel<<<(num_vertices + w - 1) / w, b, 0, stream>>>(
            cache.dodg_offsets, cache.dodg_indices, d_all_counts, num_vertices);
    }

    
    if (vertices != nullptr && n_vertices > 0) {
        gather_counts_kernel<<<(n_vertices + 255) / 256, 256, 0, stream>>>(
            d_all_counts, vertices, counts, static_cast<int32_t>(n_vertices));
    }
}

}  
