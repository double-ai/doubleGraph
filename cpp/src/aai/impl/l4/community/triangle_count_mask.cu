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

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* masked_deg = nullptr;
    int32_t* dodg_deg = nullptr;
    int32_t* dodg_offsets = nullptr;
    int32_t* all_counts = nullptr;
    void* cub_temp = nullptr;
    int32_t* dodg_indices = nullptr;
    int32_t* edge_src = nullptr;

    int64_t masked_deg_cap = 0;
    int64_t dodg_deg_cap = 0;
    int64_t dodg_offsets_cap = 0;
    int64_t all_counts_cap = 0;
    size_t cub_temp_cap = 0;
    int64_t dodg_indices_cap = 0;
    int64_t edge_src_cap = 0;

    void ensure_vertex_buffers(int32_t n) {
        if (masked_deg_cap < n) {
            if (masked_deg) cudaFree(masked_deg);
            cudaMalloc(&masked_deg, (int64_t)n * sizeof(int32_t));
            masked_deg_cap = n;
        }
        if (dodg_deg_cap < n) {
            if (dodg_deg) cudaFree(dodg_deg);
            cudaMalloc(&dodg_deg, (int64_t)n * sizeof(int32_t));
            dodg_deg_cap = n;
        }
        if (dodg_offsets_cap < (int64_t)n + 1) {
            if (dodg_offsets) cudaFree(dodg_offsets);
            cudaMalloc(&dodg_offsets, ((int64_t)n + 1) * sizeof(int32_t));
            dodg_offsets_cap = (int64_t)n + 1;
        }
        if (all_counts_cap < n) {
            if (all_counts) cudaFree(all_counts);
            cudaMalloc(&all_counts, (int64_t)n * sizeof(int32_t));
            all_counts_cap = n;
        }
    }

    void ensure_cub_temp(size_t size) {
        if (cub_temp_cap < size) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, size);
            cub_temp_cap = size;
        }
    }

    void ensure_edge_buffers(int32_t ne) {
        if (dodg_indices_cap < ne) {
            if (dodg_indices) cudaFree(dodg_indices);
            cudaMalloc(&dodg_indices, (int64_t)ne * sizeof(int32_t));
            dodg_indices_cap = ne;
        }
        if (edge_src_cap < ne) {
            if (edge_src) cudaFree(edge_src);
            cudaMalloc(&edge_src, (int64_t)ne * sizeof(int32_t));
            edge_src_cap = ne;
        }
    }

    ~Cache() override {
        if (masked_deg) cudaFree(masked_deg);
        if (dodg_deg) cudaFree(dodg_deg);
        if (dodg_offsets) cudaFree(dodg_offsets);
        if (all_counts) cudaFree(all_counts);
        if (cub_temp) cudaFree(cub_temp);
        if (dodg_indices) cudaFree(dodg_indices);
        if (edge_src) cudaFree(edge_src);
    }
};


__global__ void compute_masked_degree_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ masked_deg,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int start = offsets[v];
    int end = offsets[v + 1];
    if (start >= end) { masked_deg[v] = 0; return; }

    int count = 0;
    int first_word = start >> 5;
    int last_word = (end - 1) >> 5;

    if (first_word == last_word) {
        uint32_t mask = edge_mask[first_word];
        int first_bit = start & 31;
        int num_bits = end - start;
        mask >>= first_bit;
        if (num_bits < 32) mask &= (1u << num_bits) - 1;
        count = __popc(mask);
    } else {
        count += __popc(edge_mask[first_word] >> (start & 31));
        for (int w = first_word + 1; w < last_word; w++)
            count += __popc(edge_mask[w]);
        int last_bit = end & 31;
        if (last_bit == 0) count += __popc(edge_mask[last_word]);
        else count += __popc(edge_mask[last_word] & ((1u << last_bit) - 1));
    }
    masked_deg[v] = count;
}


__global__ void compute_dodg_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ masked_deg,
    int32_t* __restrict__ dodg_deg,
    int32_t num_vertices
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;

    int start = offsets[u];
    int end = offsets[u + 1];
    int count = 0;
    int du = masked_deg[u];

    for (int e = start; e < end; e++) {
        if (!((edge_mask[e >> 5] >> (e & 31)) & 1u)) continue;
        int v = indices[e];
        int dv = masked_deg[v];
        if (du < dv || (du == dv && u < v))
            count++;
    }
    dodg_deg[u] = count;
}


__global__ void fill_dodg_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ masked_deg,
    const int32_t* __restrict__ dodg_offsets,
    int32_t* __restrict__ dodg_indices,
    int32_t num_vertices
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;

    int start = offsets[u];
    int end = offsets[u + 1];
    int write_pos = dodg_offsets[u];
    int du = masked_deg[u];

    for (int e = start; e < end; e++) {
        if (!((edge_mask[e >> 5] >> (e & 31)) & 1u)) continue;
        int v = indices[e];
        int dv = masked_deg[v];
        if (du < dv || (du == dv && u < v))
            dodg_indices[write_pos++] = v;
    }
}


__global__ void sort_and_source_kernel(
    const int32_t* __restrict__ dodg_offsets,
    int32_t* __restrict__ dodg_indices,
    int32_t* __restrict__ edge_src,
    int32_t num_vertices
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;

    int start = dodg_offsets[u];
    int end = dodg_offsets[u + 1];
    int n = end - start;

    
    for (int e = start; e < end; e++)
        edge_src[e] = u;

    
    if (n <= 1) return;
    for (int i = start + 1; i < end; i++) {
        int key = dodg_indices[i];
        int j = i - 1;
        while (j >= start && dodg_indices[j] > key) {
            dodg_indices[j + 1] = dodg_indices[j];
            j--;
        }
        dodg_indices[j + 1] = key;
    }
}


__device__ __forceinline__ int lower_bound_dev(const int32_t* arr, int size, int target) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__global__ void count_triangles_kernel(
    const int32_t* __restrict__ dodg_offsets,
    const int32_t* __restrict__ dodg_indices,
    const int32_t* __restrict__ edge_src,
    int32_t* __restrict__ counts,
    int32_t num_dodg_edges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    if (warp_id >= num_dodg_edges) return;

    int edge_id = warp_id;

    int u, v;
    if (lane == 0) {
        u = edge_src[edge_id];
        v = dodg_indices[edge_id];
    }
    u = __shfl_sync(0xFFFFFFFF, u, 0);
    v = __shfl_sync(0xFFFFFFFF, v, 0);

    int start_u = dodg_offsets[u];
    int end_u = dodg_offsets[u + 1];
    int start_v = dodg_offsets[v];
    int end_v = dodg_offsets[v + 1];

    int size_u = end_u - start_u;
    int size_v = end_v - start_v;

    const int32_t* A, *B;
    int size_a, size_b;
    if (size_u <= size_v) {
        A = dodg_indices + start_u; size_a = size_u;
        B = dodg_indices + start_v; size_b = size_v;
    } else {
        A = dodg_indices + start_v; size_a = size_v;
        B = dodg_indices + start_u; size_b = size_u;
    }

    if (size_a == 0) return;

    int local_count = 0;

    
    int b_min = B[0];
    int b_max = B[size_b - 1];

    
    for (int i = lane; i < size_a; i += 32) {
        int target = A[i];
        if (target < b_min || target > b_max) continue;
        int pos = lower_bound_dev(B, size_b, target);
        if (pos < size_b && B[pos] == target) {
            local_count++;
            atomicAdd(&counts[target], 1);
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, offset);
    }

    if (lane == 0 && local_count > 0) {
        atomicAdd(&counts[u], local_count);
        atomicAdd(&counts[v], local_count);
    }
}


__global__ void gather_counts_kernel(
    const int32_t* __restrict__ all_counts,
    const int32_t* __restrict__ vertices,
    int32_t* __restrict__ output_counts,
    int32_t n_vertices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vertices) return;
    output_counts[i] = all_counts[vertices[i]];
}

}  

void triangle_count_mask(const graph32_t& graph,
                         int32_t* counts,
                         const int32_t* vertices,
                         std::size_t n_vertices) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_verts = graph.number_of_vertices;

    if (num_verts == 0) return;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cache.ensure_vertex_buffers(num_verts);

    
    compute_masked_degree_kernel<<<(num_verts + 255) / 256, 256>>>(
        d_offsets, d_edge_mask, cache.masked_deg, num_verts);

    
    compute_dodg_degree_kernel<<<(num_verts + 255) / 256, 256>>>(
        d_offsets, d_indices, d_edge_mask, cache.masked_deg, cache.dodg_deg, num_verts);

    
    cudaMemsetAsync(cache.dodg_offsets, 0, sizeof(int32_t));

    size_t cub_temp_size = 0;
    cub::DeviceScan::InclusiveSum(nullptr, cub_temp_size,
        (int32_t*)nullptr, (int32_t*)nullptr, num_verts);
    if (cub_temp_size < 1) cub_temp_size = 1;
    cache.ensure_cub_temp(cub_temp_size);
    cub::DeviceScan::InclusiveSum(cache.cub_temp, cub_temp_size,
        cache.dodg_deg, cache.dodg_offsets + 1, num_verts);

    
    int32_t total_dodg_edges;
    cudaMemcpy(&total_dodg_edges, cache.dodg_offsets + num_verts,
        sizeof(int32_t), cudaMemcpyDeviceToHost);

    if (total_dodg_edges == 0) {
        std::size_t count_size = (vertices != nullptr) ? n_vertices : (std::size_t)num_verts;
        cudaMemsetAsync(counts, 0, count_size * sizeof(int32_t));
        return;
    }

    
    cache.ensure_edge_buffers(total_dodg_edges);
    fill_dodg_kernel<<<(num_verts + 255) / 256, 256>>>(
        d_offsets, d_indices, d_edge_mask, cache.masked_deg,
        cache.dodg_offsets, cache.dodg_indices, num_verts);

    
    sort_and_source_kernel<<<(num_verts + 255) / 256, 256>>>(
        cache.dodg_offsets, cache.dodg_indices, cache.edge_src, num_verts);

    
    
    int32_t* count_buf;
    if (vertices != nullptr) {
        count_buf = cache.all_counts;
    } else {
        count_buf = counts;
    }
    cudaMemsetAsync(count_buf, 0, (std::size_t)num_verts * sizeof(int32_t));

    int tpb = 256;
    int grid = (int)(((int64_t)total_dodg_edges * 32 + tpb - 1) / tpb);
    count_triangles_kernel<<<grid, tpb>>>(
        cache.dodg_offsets, cache.dodg_indices, cache.edge_src,
        count_buf, total_dodg_edges);

    
    if (vertices != nullptr) {
        int32_t nv = (int32_t)n_vertices;
        gather_counts_kernel<<<(nv + 255) / 256, 256>>>(
            count_buf, vertices, counts, nv);
    }
}

}  
