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
#include <cub/cub.cuh>

namespace aai {

namespace {

constexpr int WARPS_PER_BLOCK = 8;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;


__global__ void iota_kernel(int32_t* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = i;
}


__global__ void gather_i32(const int32_t* __restrict__ src, const int32_t* __restrict__ idx,
                           int32_t* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
}

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
overlap_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ sorted_first,  
    const int32_t* __restrict__ sorted_second, 
    const int32_t* __restrict__ orig_idx,      
    float* __restrict__ scores,
    int num_pairs)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    const int u = sorted_first[warp_id];
    const int v = sorted_second[warp_id];
    const int out_idx = orig_idx[warp_id];

    const int u_start = offsets[u];
    const int u_end = offsets[u + 1];
    const int v_start = offsets[v];
    const int v_end = offsets[v + 1];
    const int u_deg = u_end - u_start;
    const int v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) scores[out_idx] = 0.0f;
        return;
    }

    
    float sum_wu = 0.0f, sum_wv = 0.0f;
    for (int i = lane; i < u_deg; i += 32) sum_wu += edge_weights[u_start + i];
    for (int i = lane; i < v_deg; i += 32) sum_wv += edge_weights[v_start + i];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sum_wu += __shfl_xor_sync(0xffffffff, sum_wu, o);
        sum_wv += __shfl_xor_sync(0xffffffff, sum_wv, o);
    }

    float denom = fminf(sum_wu, sum_wv);
    if (denom == 0.0f) {
        if (lane == 0) scores[out_idx] = 0.0f;
        return;
    }

    
    int s_start, s_deg, l_start, l_deg;
    if (u_deg <= v_deg) {
        s_start = u_start; s_deg = u_deg; l_start = v_start; l_deg = v_deg;
    } else {
        s_start = v_start; s_deg = v_deg; l_start = u_start; l_deg = u_deg;
    }

    float isect_sum = 0.0f;

    
    
    int search_lo = 0;

    for (int i = lane; i < s_deg; i += 32) {
        const int target = indices[s_start + i];

        
        int lo = search_lo, hi = l_deg;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (indices[l_start + mid] < target) lo = mid + 1;
            else hi = mid;
        }

        
        int rank = 0;
        if (i > 0 && indices[s_start + i - 1] == target) {
            rank = 1;
            for (int j = i - 2; j >= 0 && indices[s_start + j] == target; j--)
                rank++;
        }

        int mpos = lo + rank;
        if (mpos < l_deg && indices[l_start + mpos] == target) {
            isect_sum += fminf(edge_weights[s_start + i], edge_weights[l_start + mpos]);
        }

        search_lo = lo;
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        isect_sum += __shfl_xor_sync(0xffffffff, isect_sum, o);

    if (lane == 0) scores[out_idx] = isect_sum / denom;
}

size_t get_sort_temp_bytes(int max_pairs) {
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes,
        (const int32_t*)nullptr, (int32_t*)nullptr,
        (const int32_t*)nullptr, (int32_t*)nullptr,
        max_pairs);
    return temp_bytes;
}

struct Cache : Cacheable {
    int32_t* sort_keys_out = nullptr;
    int32_t* sort_vals_in = nullptr;
    int32_t* sort_vals_out = nullptr;
    int32_t* gathered_second = nullptr;
    void* sort_temp = nullptr;
    size_t sort_temp_bytes = 0;
    int max_pairs = 0;

    void ensure(int num_pairs) {
        if (num_pairs <= max_pairs) return;
        free_buffers();
        max_pairs = num_pairs;
        cudaMalloc(&sort_keys_out, max_pairs * sizeof(int32_t));
        cudaMalloc(&sort_vals_in, max_pairs * sizeof(int32_t));
        cudaMalloc(&sort_vals_out, max_pairs * sizeof(int32_t));
        cudaMalloc(&gathered_second, max_pairs * sizeof(int32_t));
        sort_temp_bytes = get_sort_temp_bytes(max_pairs);
        cudaMalloc(&sort_temp, sort_temp_bytes);
    }

    void free_buffers() {
        if (sort_keys_out) { cudaFree(sort_keys_out); sort_keys_out = nullptr; }
        if (sort_vals_in) { cudaFree(sort_vals_in); sort_vals_in = nullptr; }
        if (sort_vals_out) { cudaFree(sort_vals_out); sort_vals_out = nullptr; }
        if (gathered_second) { cudaFree(gathered_second); gathered_second = nullptr; }
        if (sort_temp) { cudaFree(sort_temp); sort_temp = nullptr; }
        max_pairs = 0;
    }

    ~Cache() override {
        free_buffers();
    }
};

}  

void overlap_similarity(const graph32_t& graph,
                        const float* edge_weights,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int np = static_cast<int>(num_pairs);
    if (np == 0) return;

    cache.ensure(np);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    cudaStream_t stream = 0;

    
    int bs = 256;
    int gs = (np + bs - 1) / bs;
    iota_kernel<<<gs, bs, 0, stream>>>(cache.sort_vals_in, np);

    
    cub::DeviceRadixSort::SortPairs(
        cache.sort_temp, cache.sort_temp_bytes,
        vertex_pairs_first, cache.sort_keys_out,
        cache.sort_vals_in, cache.sort_vals_out,
        np, 0, 32, stream);

    
    gather_i32<<<gs, bs, 0, stream>>>(vertex_pairs_second, cache.sort_vals_out,
                                       cache.gathered_second, np);

    
    int num_blocks = (np + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    overlap_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        offsets, indices, edge_weights,
        cache.sort_keys_out, cache.gathered_second, cache.sort_vals_out,
        similarity_scores, np);
}

}  
