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


__device__ __forceinline__ bool binary_search_exists(
    const int32_t* __restrict__ arr, int size, int32_t target
) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int32_t val = __ldg(&arr[mid]);
        if (val == target) return true;
        if (val < target) lo = mid + 1;
        else hi = mid;
    }
    return false;
}


#define PAIRS_PER_WARP 4

__global__ void cosine_similarity_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int base_pair = warp_id * PAIRS_PER_WARP;

    #pragma unroll
    for (int p = 0; p < PAIRS_PER_WARP; p++) {
        int64_t pair_idx = base_pair + p;
        if (pair_idx >= num_pairs) return;

        int32_t u = first[pair_idx];
        int32_t v = second[pair_idx];

        int32_t u_start = __ldg(&offsets[u]);
        int32_t u_end   = __ldg(&offsets[u + 1]);
        int32_t v_start = __ldg(&offsets[v]);
        int32_t v_end   = __ldg(&offsets[v + 1]);

        int32_t deg_u = u_end - u_start;
        int32_t deg_v = v_end - v_start;

        if (deg_u == 0 || deg_v == 0) {
            if (lane == 0) scores[pair_idx] = 0.0f;
            continue;
        }

        
        const int32_t* small_arr;
        const int32_t* large_arr;
        int small_size, large_size;

        if (deg_u <= deg_v) {
            small_arr = indices + u_start;
            large_arr = indices + v_start;
            small_size = deg_u;
            large_size = deg_v;
        } else {
            small_arr = indices + v_start;
            large_arr = indices + u_start;
            small_size = deg_v;
            large_size = deg_u;
        }

        
        int32_t small_min = __ldg(&small_arr[0]);
        int32_t small_max = __ldg(&small_arr[small_size - 1]);
        int32_t large_min = __ldg(&large_arr[0]);
        int32_t large_max = __ldg(&large_arr[large_size - 1]);

        if (small_max < large_min || large_max < small_min) {
            if (lane == 0) scores[pair_idx] = 0.0f;
            continue;
        }

        
        bool found = false;
        for (int i = lane; i < small_size; i += 32) {
            int32_t target = __ldg(&small_arr[i]);
            if (target >= large_min && target <= large_max) {
                if (binary_search_exists(large_arr, large_size, target)) {
                    found = true;
                    break;
                }
            }
        }

        unsigned mask = __ballot_sync(0xFFFFFFFF, found);
        if (lane == 0) {
            scores[pair_idx] = (mask != 0) ? 1.0f : 0.0f;
        }
    }
}



__global__ void cosine_similarity_large_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int32_t u = first[warp_id];
    int32_t v = second[warp_id];

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end   = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end   = __ldg(&offsets[v + 1]);

    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;

    if (deg_u == 0 || deg_v == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    const int32_t* small_arr;
    const int32_t* large_arr;
    int small_size, large_size;

    if (deg_u <= deg_v) {
        small_arr = indices + u_start;
        large_arr = indices + v_start;
        small_size = deg_u;
        large_size = deg_v;
    } else {
        small_arr = indices + v_start;
        large_arr = indices + u_start;
        small_size = deg_v;
        large_size = deg_u;
    }

    
    bool found = false;
    for (int i = lane; i < small_size && !found; i += 32) {
        int32_t target = __ldg(&small_arr[i]);
        if (binary_search_exists(large_arr, large_size, target)) {
            found = true;
        }
    }

    unsigned mask = __ballot_sync(0xFFFFFFFF, found);
    if (lane == 0) {
        scores[warp_id] = (mask != 0) ? 1.0f : 0.0f;
    }
}

}  

void cosine_similarity(const graph32_t& graph,
                       const int32_t* vertex_pairs_first,
                       const int32_t* vertex_pairs_second,
                       std::size_t num_pairs,
                       float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int pairs_per_block = warps_per_block * PAIRS_PER_WARP;
    int grid = (int)((num_pairs + pairs_per_block - 1) / pairs_per_block);

    cosine_similarity_kernel<<<grid, threads_per_block>>>(
        graph.offsets, graph.indices,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, (int64_t)num_pairs);
}

}  
