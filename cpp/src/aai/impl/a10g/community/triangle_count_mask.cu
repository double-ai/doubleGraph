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

struct Cache : Cacheable {
    int32_t* degree = nullptr;
    int64_t degree_cap = 0;

    int32_t* outdeg = nullptr;
    int64_t outdeg_cap = 0;

    int32_t* dodg_off = nullptr;
    int64_t dodg_off_cap = 0;

    void* cub_temp = nullptr;
    size_t cub_temp_cap = 0;

    int32_t* dodg_idx = nullptr;
    int64_t dodg_idx_cap = 0;

    int32_t* edge_src_buf = nullptr;
    int64_t edge_src_cap = 0;

    int32_t* counts_all = nullptr;
    int64_t counts_all_cap = 0;

    ~Cache() override {
        if (degree) cudaFree(degree);
        if (outdeg) cudaFree(outdeg);
        if (dodg_off) cudaFree(dodg_off);
        if (cub_temp) cudaFree(cub_temp);
        if (dodg_idx) cudaFree(dodg_idx);
        if (edge_src_buf) cudaFree(edge_src_buf);
        if (counts_all) cudaFree(counts_all);
    }
};




__global__ void compute_masked_degree_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ degree,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int start = offsets[v];
    int end = offsets[v + 1];
    int count = 0;
    int e = start;
    while (e < end) {
        int word_idx = e >> 5;
        int bit_offset = e & 31;
        uint32_t word = edge_mask[word_idx] >> bit_offset;
        int bits_avail = 32 - bit_offset;
        int bits_needed = end - e;
        int bits = bits_avail < bits_needed ? bits_avail : bits_needed;
        if (bits < 32) word &= (1u << bits) - 1;
        count += __popc(word);
        e += bits;
    }
    degree[v] = count;
}




__global__ void compute_dodg_outdeg_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ degree,
    int32_t* __restrict__ dodg_outdeg,
    int32_t num_vertices)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_vertices) return;

    int u = warp_id;
    int start = offsets[u];
    int end = offsets[u + 1];
    int deg_u = degree[u];
    int count = 0;

    for (int e = start + lane; e < end; e += 32) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1) {
            int v = indices[e];
            int deg_v = degree[v];
            if (deg_u < deg_v || (deg_u == deg_v && u < v)) count++;
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        count += __shfl_down_sync(0xffffffff, count, offset);

    if (lane == 0) dodg_outdeg[u] = count;
}





__global__ void fill_dodg_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ degree,
    const int32_t* __restrict__ dodg_offsets,
    int32_t* __restrict__ dodg_indices,
    int32_t* __restrict__ edge_src,
    int32_t num_vertices)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_vertices) return;

    int u = warp_id;
    int start = offsets[u];
    int end = offsets[u + 1];
    int deg_u = degree[u];
    int base_pos = dodg_offsets[u];

    for (int e = start + lane; e < end; e += 32) {
        bool valid = false;
        int v = -1;
        if ((edge_mask[e >> 5] >> (e & 31)) & 1) {
            v = indices[e];
            int deg_v = degree[v];
            if (deg_u < deg_v || (deg_u == deg_v && u < v)) valid = true;
        }

        
        unsigned mask = __ballot_sync(0xffffffff, valid);
        int count_before = __popc(mask & ((1u << lane) - 1));
        int total_valid = __popc(mask);

        if (valid) {
            int pos = base_pos + count_before;
            dodg_indices[pos] = v;
            edge_src[pos] = u;
        }
        base_pos += total_valid;
    }
}




__global__ void sort_segments_kernel(
    const int32_t* __restrict__ dodg_offsets,
    int32_t* __restrict__ dodg_indices,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int start = dodg_offsets[v];
    int end = dodg_offsets[v + 1];
    int len = end - start;
    if (len <= 1) return;

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




__device__ __forceinline__ int warp_reduce_sum(int val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}


__global__ void __launch_bounds__(128, 8)
count_triangles_warp_kernel(
    const int32_t* __restrict__ dodg_offsets,
    const int32_t* __restrict__ dodg_indices,
    const int32_t* __restrict__ edge_src,
    int32_t* __restrict__ counts,
    int32_t num_dodg_edges)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_dodg_edges) return;

    int u = edge_src[warp_id];
    int v = dodg_indices[warp_id];
    int u_start = dodg_offsets[u];
    int u_end = dodg_offsets[u + 1];
    int v_start = dodg_offsets[v];
    int v_end = dodg_offsets[v + 1];
    int u_len = u_end - u_start;
    int v_len = v_end - v_start;

    if (u_len == 0 || v_len == 0) return;

    const int32_t* A; int a_len;
    const int32_t* B; int b_len;
    if (u_len <= v_len) {
        A = dodg_indices + u_start; a_len = u_len;
        B = dodg_indices + v_start; b_len = v_len;
    } else {
        A = dodg_indices + v_start; a_len = v_len;
        B = dodg_indices + u_start; b_len = u_len;
    }

    int local_count = 0;
    for (int ai = lane; ai < a_len; ai += 32) {
        int target = A[ai];
        int blo = 0, bhi = b_len;
        while (blo < bhi) {
            int bmid = (blo + bhi) >> 1;
            if (B[bmid] < target) blo = bmid + 1;
            else bhi = bmid;
        }
        if (blo < b_len && B[blo] == target) {
            local_count++;
            atomicAdd(&counts[target], 1);
        }
    }

    int total = warp_reduce_sum(local_count);
    if (lane == 0 && total > 0) {
        atomicAdd(&counts[u], total);
        atomicAdd(&counts[v], total);
    }
}


__global__ void __launch_bounds__(256, 6)
count_triangles_thread_kernel(
    const int32_t* __restrict__ dodg_offsets,
    const int32_t* __restrict__ dodg_indices,
    const int32_t* __restrict__ edge_src,
    int32_t* __restrict__ counts,
    int32_t num_dodg_edges)
{
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_idx >= num_dodg_edges) return;

    int u = edge_src[edge_idx];
    int v = dodg_indices[edge_idx];
    int u_start = dodg_offsets[u];
    int u_end = dodg_offsets[u + 1];
    int v_start = dodg_offsets[v];
    int v_end = dodg_offsets[v + 1];
    if (u_start == u_end || v_start == v_end) return;

    int i = u_start, j = v_start;
    int tri_count = 0;
    while (i < u_end && j < v_end) {
        int a = dodg_indices[i];
        int b = dodg_indices[j];
        if (a == b) {
            tri_count++;
            atomicAdd(&counts[a], 1);
            i++; j++;
        } else if (a < b) {
            i++;
        } else {
            j++;
        }
    }
    if (tri_count > 0) {
        atomicAdd(&counts[u], tri_count);
        atomicAdd(&counts[v], tri_count);
    }
}

__global__ void gather_counts_kernel(
    const int32_t* __restrict__ all_counts,
    const int32_t* __restrict__ vertices,
    int32_t* __restrict__ out_counts,
    int32_t n_vertices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vertices) return;
    out_counts[i] = all_counts[vertices[i]];
}

}  

void triangle_count_mask(const graph32_t& graph,
                         int32_t* counts,
                         const int32_t* vertices,
                         std::size_t n_vertices) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t V = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;
    const uint32_t* d_mask = graph.edge_mask;
    cudaStream_t stream = 0;

    int64_t out_size = (vertices != nullptr) ? (int64_t)n_vertices : (int64_t)V;

    if (V == 0 || E == 0) {
        if (out_size > 0)
            cudaMemsetAsync(counts, 0, out_size * sizeof(int32_t), stream);
        cudaStreamSynchronize(stream);
        return;
    }

    
    {
        int64_t v64 = (int64_t)V;
        if (cache.degree_cap < v64) {
            if (cache.degree) cudaFree(cache.degree);
            cudaMalloc(&cache.degree, v64 * sizeof(int32_t));
            cache.degree_cap = v64;
        }
        if (cache.outdeg_cap < v64 + 1) {
            if (cache.outdeg) cudaFree(cache.outdeg);
            cudaMalloc(&cache.outdeg, (v64 + 1) * sizeof(int32_t));
            cache.outdeg_cap = v64 + 1;
        }
        if (cache.dodg_off_cap < v64 + 1) {
            if (cache.dodg_off) cudaFree(cache.dodg_off);
            cudaMalloc(&cache.dodg_off, (v64 + 1) * sizeof(int32_t));
            cache.dodg_off_cap = v64 + 1;
        }
        size_t cub_needed = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, cub_needed,
            (int32_t*)nullptr, (int32_t*)nullptr, V + 1);
        cub_needed += 256;
        if (cache.cub_temp_cap < cub_needed) {
            if (cache.cub_temp) cudaFree(cache.cub_temp);
            cudaMalloc(&cache.cub_temp, cub_needed);
            cache.cub_temp_cap = cub_needed;
        }
    }

    
    {
        int block = 256;
        int grid = (V + block - 1) / block;
        if (grid > 0)
            compute_masked_degree_kernel<<<grid, block, 0, stream>>>(
                d_offsets, d_mask, cache.degree, V);
    }

    
    {
        int warps_per_block = 8;
        int threads = warps_per_block * 32;
        int grid = (V + warps_per_block - 1) / warps_per_block;
        if (grid > 0)
            compute_dodg_outdeg_warp_kernel<<<grid, threads, 0, stream>>>(
                d_offsets, d_indices, d_mask, cache.degree, cache.outdeg, V);
        cudaMemsetAsync(cache.outdeg + V, 0, sizeof(int32_t), stream);
    }

    
    {
        size_t temp_size = cache.cub_temp_cap;
        cub::DeviceScan::ExclusiveSum(cache.cub_temp, temp_size,
            cache.outdeg, cache.dodg_off, V + 1, stream);
    }

    
    int32_t num_dodg;
    cudaMemcpyAsync(&num_dodg, cache.dodg_off + V, sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (num_dodg == 0) {
        if (out_size > 0)
            cudaMemsetAsync(counts, 0, out_size * sizeof(int32_t), stream);
        cudaStreamSynchronize(stream);
        return;
    }

    
    {
        int64_t n = (int64_t)num_dodg;
        if (cache.dodg_idx_cap < n) {
            if (cache.dodg_idx) cudaFree(cache.dodg_idx);
            cudaMalloc(&cache.dodg_idx, n * sizeof(int32_t));
            cache.dodg_idx_cap = n;
        }
        if (cache.edge_src_cap < n) {
            if (cache.edge_src_buf) cudaFree(cache.edge_src_buf);
            cudaMalloc(&cache.edge_src_buf, n * sizeof(int32_t));
            cache.edge_src_cap = n;
        }
    }

    
    {
        int warps_per_block = 8;
        int threads = warps_per_block * 32;
        int grid = (V + warps_per_block - 1) / warps_per_block;
        if (grid > 0)
            fill_dodg_warp_kernel<<<grid, threads, 0, stream>>>(
                d_offsets, d_indices, d_mask, cache.degree, cache.dodg_off,
                cache.dodg_idx, cache.edge_src_buf, V);
    }

    
    {
        int block = 256;
        int grid = (V + block - 1) / block;
        if (grid > 0)
            sort_segments_kernel<<<grid, block, 0, stream>>>(
                cache.dodg_off, cache.dodg_idx, V);
    }

    
    int32_t* counts_buf;
    if (vertices != nullptr) {
        int64_t v64 = (int64_t)V;
        if (cache.counts_all_cap < v64) {
            if (cache.counts_all) cudaFree(cache.counts_all);
            cudaMalloc(&cache.counts_all, v64 * sizeof(int32_t));
            cache.counts_all_cap = v64;
        }
        counts_buf = cache.counts_all;
    } else {
        counts_buf = counts;
    }
    cudaMemsetAsync(counts_buf, 0, (int64_t)V * sizeof(int32_t), stream);

    
    float avg_deg = (float)num_dodg / (float)V;
    if (avg_deg > 8.0f) {
        int warps_per_block = 4;
        int threads_per_block = warps_per_block * 32;
        int grid = (num_dodg + warps_per_block - 1) / warps_per_block;
        if (grid > 0)
            count_triangles_warp_kernel<<<grid, threads_per_block, 0, stream>>>(
                cache.dodg_off, cache.dodg_idx, cache.edge_src_buf,
                counts_buf, num_dodg);
    } else {
        int block = 256;
        int grid = (num_dodg + block - 1) / block;
        if (grid > 0)
            count_triangles_thread_kernel<<<grid, block, 0, stream>>>(
                cache.dodg_off, cache.dodg_idx, cache.edge_src_buf,
                counts_buf, num_dodg);
    }

    
    if (vertices != nullptr) {
        int block = 256;
        int grid = ((int32_t)n_vertices + block - 1) / block;
        if (grid > 0)
            gather_counts_kernel<<<grid, block, 0, stream>>>(
                counts_buf, vertices, counts, (int32_t)n_vertices);
    }

    cudaStreamSynchronize(stream);
}

}  
