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
#include <algorithm>
#include <vector>

namespace aai {

namespace {

struct Cache : Cacheable {
    int grid_single = 0;
    int grid_batched = 0;
    bool initialized = false;
};

__device__ __forceinline__ int upper_bound_search(const int32_t* arr, int n, int val) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] <= val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void init_frontiers_kernel(
    const int32_t* __restrict__ start_vertices,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_sizes,
    int32_t num_start_vertices,
    int64_t stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_start_vertices) {
        frontier[(int64_t)idx * stride] = start_vertices[idx];
        frontier_sizes[idx] = 1;
    }
}

__global__ void prefix_sum_kernel(
    const int32_t* __restrict__ sizes,
    int32_t* __restrict__ cum_sizes,
    int32_t n
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int32_t sum = 0;
        cum_sizes[0] = 0;
        for (int i = 0; i < n; i++) {
            sum += sizes[i];
            cum_sizes[i + 1] = sum;
        }
    }
}

__global__ void expand_all_persistent_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ frontier_cur,
    int32_t* __restrict__ frontier_next,
    int32_t* __restrict__ next_sizes,
    uint32_t* __restrict__ bitmaps,
    const int32_t* __restrict__ cum_sizes,
    int32_t num_start_vertices,
    int64_t stride,
    int32_t bitmap_words
) {
    int total_work = cum_sizes[num_start_vertices];
    if (total_work <= 0) return;

    int total_warps_grid = gridDim.x * (blockDim.x >> 5);
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid >> 5;
    int lane = global_tid & 31;

    for (int w = warp_id; w < total_work; w += total_warps_grid) {
        int s = upper_bound_search(cum_sizes, num_start_vertices + 1, w) - 1;
        int local_idx = w - cum_sizes[s];

        int32_t v = frontier_cur[(int64_t)s * stride + local_idx];
        int32_t start = csr_offsets[v];
        int32_t end = csr_offsets[v + 1];

        uint32_t* bm = bitmaps + (int64_t)s * bitmap_words;
        int32_t* nxt_f = frontier_next + (int64_t)s * stride;
        int32_t* ns_ptr = &next_sizes[s];

        for (int32_t e = start + lane; e < end; e += 32) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1) {
                int32_t u = csr_indices[e];
                uint32_t mask = 1u << (u & 31);
                uint32_t old = atomicOr(&bm[u >> 5], mask);
                if (!(old & mask)) {
                    int32_t pos = atomicAdd(ns_ptr, 1);
                    nxt_f[pos] = u;
                }
            }
        }
    }
}

__global__ void expand_single_persistent_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ frontier_cur,
    const int32_t* __restrict__ frontier_size_ptr,
    int32_t* __restrict__ frontier_next,
    int32_t* __restrict__ next_size,
    uint32_t* __restrict__ bitmap
) {
    int total_work = *frontier_size_ptr;
    if (total_work <= 0) return;

    int total_warps_grid = gridDim.x * (blockDim.x >> 5);
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid >> 5;
    int lane = global_tid & 31;

    for (int w = warp_id; w < total_work; w += total_warps_grid) {
        int32_t v = frontier_cur[w];
        int32_t start = csr_offsets[v];
        int32_t end = csr_offsets[v + 1];

        for (int32_t e = start + lane; e < end; e += 32) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1) {
                int32_t u = csr_indices[e];
                uint32_t mask = 1u << (u & 31);
                uint32_t old = atomicOr(&bitmap[u >> 5], mask);
                if (!(old & mask)) {
                    int32_t pos = atomicAdd(next_size, 1);
                    frontier_next[pos] = u;
                }
            }
        }
    }
}

__global__ void gather_results_kernel(
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ output,
    const int64_t* __restrict__ out_offsets,
    const int32_t* __restrict__ frontier_sizes,
    int32_t num_start_vertices,
    int64_t stride
) {
    for (int s = blockIdx.x; s < num_start_vertices; s += gridDim.x) {
        int64_t out_start = out_offsets[s];
        int32_t count = frontier_sizes[s];
        const int32_t* src = frontier + (int64_t)s * stride;
        int32_t* dst = output + out_start;
        for (int i = threadIdx.x; i < count; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

}  

k_hop_nbrs_result_t k_hop_nbrs_mask(const graph32_t& graph,
                                     const int32_t* start_vertices,
                                     std::size_t num_start_vertices,
                                     std::size_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    if (!cache.initialized) {
        int bpm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpm, expand_single_persistent_kernel, 256, 0);
        int device; cudaGetDevice(&device);
        int num_sms; cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
        cache.grid_single = bpm * num_sms;

        bpm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpm, expand_all_persistent_kernel, 256, 0);
        cache.grid_batched = bpm * num_sms;

        cache.initialized = true;
    }

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    int32_t num_start = (int32_t)num_start_vertices;
    int32_t k_val = (int32_t)k;
    int32_t bw = (num_vertices + 31) / 32;
    cudaStream_t stream = 0;
    int64_t nv = (int64_t)num_vertices;
    int64_t ns = (int64_t)num_start;

    if (num_start == 1) {
        
        uint32_t* d_bitmap;
        int32_t* d_fa;
        int32_t* d_fb;
        int32_t* d_fsize_a;
        int32_t* d_fsize_b;

        cudaMalloc(&d_bitmap, (size_t)bw * sizeof(uint32_t));
        cudaMalloc(&d_fa, (size_t)nv * sizeof(int32_t));
        cudaMalloc(&d_fb, (size_t)nv * sizeof(int32_t));
        cudaMalloc(&d_fsize_a, sizeof(int32_t));
        cudaMalloc(&d_fsize_b, sizeof(int32_t));

        
        cudaMemcpyAsync(d_fa, start_vertices, sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        int32_t one = 1;
        cudaMemcpyAsync(d_fsize_a, &one, sizeof(int32_t), cudaMemcpyHostToDevice, stream);

        int32_t* cur = d_fa, *nxt = d_fb;
        int32_t* cur_sz = d_fsize_a, *nxt_sz = d_fsize_b;

        int grid = cache.grid_single;

        for (int hop = 0; hop < k_val; hop++) {
            cudaMemsetAsync(d_bitmap, 0, (size_t)bw * sizeof(uint32_t), stream);
            cudaMemsetAsync(nxt_sz, 0, sizeof(int32_t), stream);
            expand_single_persistent_kernel<<<grid, 256, 0, stream>>>(
                d_offsets, d_indices, d_edge_mask,
                cur, cur_sz, nxt, nxt_sz, d_bitmap);
            std::swap(cur, nxt);
            std::swap(cur_sz, nxt_sz);
        }

        int32_t final_size;
        cudaMemcpyAsync(&final_size, cur_sz, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        int64_t* out_offsets;
        int32_t* out_nbrs;
        cudaMalloc(&out_offsets, 2 * sizeof(int64_t));
        cudaMalloc(&out_nbrs, std::max((std::size_t)1, (std::size_t)final_size) * sizeof(int32_t));

        int64_t h_offsets[2] = {0, (int64_t)final_size};
        cudaMemcpyAsync(out_offsets, h_offsets, 2 * sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream);
        if (final_size > 0) {
            cudaMemcpyAsync(out_nbrs, cur,
                           (size_t)final_size * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        }
        cudaStreamSynchronize(stream);

        cudaFree(d_bitmap);
        cudaFree(d_fa);
        cudaFree(d_fb);
        cudaFree(d_fsize_a);
        cudaFree(d_fsize_b);

        k_hop_nbrs_result_t result;
        result.offsets = reinterpret_cast<std::size_t*>(out_offsets);
        result.neighbors = out_nbrs;
        result.num_offsets = 2;
        result.num_neighbors = (std::size_t)final_size;
        return result;
    } else {
        
        uint32_t* d_bitmaps;
        int32_t* d_fa;
        int32_t* d_fb;
        int32_t* d_sizes_a;
        int32_t* d_sizes_b;
        int32_t* d_cum;

        cudaMalloc(&d_bitmaps, (size_t)ns * (size_t)bw * sizeof(uint32_t));
        cudaMalloc(&d_fa, (size_t)ns * (size_t)nv * sizeof(int32_t));
        cudaMalloc(&d_fb, (size_t)ns * (size_t)nv * sizeof(int32_t));
        cudaMalloc(&d_sizes_a, (size_t)ns * sizeof(int32_t));
        cudaMalloc(&d_sizes_b, (size_t)ns * sizeof(int32_t));
        cudaMalloc(&d_cum, (size_t)(ns + 1) * sizeof(int32_t));

        int blocks = (num_start + 255) / 256;
        init_frontiers_kernel<<<blocks, 256, 0, stream>>>(
            start_vertices, d_fa, d_sizes_a, num_start, nv);

        int32_t* cur = d_fa, *nxt = d_fb;
        int32_t* cur_s = d_sizes_a, *nxt_s = d_sizes_b;

        int grid = cache.grid_batched;

        for (int hop = 0; hop < k_val; hop++) {
            prefix_sum_kernel<<<1, 1, 0, stream>>>(cur_s, d_cum, num_start);
            cudaMemsetAsync(d_bitmaps, 0, (size_t)ns * (size_t)bw * sizeof(uint32_t), stream);
            cudaMemsetAsync(nxt_s, 0, ns * sizeof(int32_t), stream);
            expand_all_persistent_kernel<<<grid, 256, 0, stream>>>(
                d_offsets, d_indices, d_edge_mask,
                cur, nxt, nxt_s, d_bitmaps, d_cum,
                num_start, nv, bw);
            std::swap(cur, nxt);
            std::swap(cur_s, nxt_s);
        }

        std::vector<int32_t> h_sizes(num_start);
        cudaMemcpyAsync(h_sizes.data(), cur_s, ns * sizeof(int32_t),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::vector<int64_t> h_offsets(num_start + 1);
        h_offsets[0] = 0;
        for (int i = 0; i < num_start; i++)
            h_offsets[i + 1] = h_offsets[i] + h_sizes[i];
        int64_t total = h_offsets[num_start];

        int64_t* out_offsets;
        int32_t* out_nbrs;
        cudaMalloc(&out_offsets, (size_t)(ns + 1) * sizeof(int64_t));
        cudaMalloc(&out_nbrs, std::max((std::size_t)1, (std::size_t)total) * sizeof(int32_t));

        cudaMemcpyAsync(out_offsets, h_offsets.data(),
            (ns + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

        if (total > 0) {
            int gather_blocks = num_start < 2048 ? num_start : 2048;
            gather_results_kernel<<<gather_blocks, 256, 0, stream>>>(
                cur, out_nbrs, out_offsets, cur_s,
                num_start, nv);
        }

        cudaStreamSynchronize(stream);

        cudaFree(d_bitmaps);
        cudaFree(d_fa);
        cudaFree(d_fb);
        cudaFree(d_sizes_a);
        cudaFree(d_sizes_b);
        cudaFree(d_cum);

        k_hop_nbrs_result_t result;
        result.offsets = reinterpret_cast<std::size_t*>(out_offsets);
        result.neighbors = out_nbrs;
        result.num_offsets = (std::size_t)(num_start + 1);
        result.num_neighbors = (std::size_t)total;
        return result;
    }
}

}  
