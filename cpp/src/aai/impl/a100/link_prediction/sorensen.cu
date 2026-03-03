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


__device__ __forceinline__ int lower_bound_dev(const int32_t* arr, int len, int32_t val) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(&arr[mid]) < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__device__ __forceinline__ int upper_bound_dev(const int32_t* arr, int len, int32_t val) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(&arr[mid]) <= val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__device__ __forceinline__ int binary_search_found(const int32_t* arr, int len, int32_t val) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int32_t m = __ldg(&arr[mid]);
        if (m < val) lo = mid + 1;
        else if (m > val) hi = mid;
        else return 1;
    }
    return 0;
}


__device__ __forceinline__ int warp_reduce_sum(int val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}




__global__ void sorensen_simple_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs)
{
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int32_t u = first[warp_id];
    int32_t v = second[warp_id];

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);

    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;

    if (deg_u + deg_v == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    const int32_t* a_ptr;
    const int32_t* b_ptr;
    int32_t a_len, b_len;
    if (deg_u <= deg_v) {
        a_ptr = indices + u_start; a_len = deg_u;
        b_ptr = indices + v_start; b_len = deg_v;
    } else {
        a_ptr = indices + v_start; a_len = deg_v;
        b_ptr = indices + u_start; b_len = deg_u;
    }

    
    int count = 0;
    for (int i = lane; i < a_len; i += 32) {
        int32_t val = __ldg(&a_ptr[i]);
        count += binary_search_found(b_ptr, b_len, val);
    }

    count = warp_reduce_sum(count);

    if (lane == 0) {
        scores[warp_id] = 2.0f * count / (float)(deg_u + deg_v);
    }
}




__global__ void sorensen_multigraph_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs)
{
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int32_t u = first[warp_id];
    int32_t v = second[warp_id];

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);

    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;

    if (deg_u + deg_v == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    const int32_t* a_ptr;
    const int32_t* b_ptr;
    int32_t a_len, b_len;
    if (deg_u <= deg_v) {
        a_ptr = indices + u_start; a_len = deg_u;
        b_ptr = indices + v_start; b_len = deg_v;
    } else {
        a_ptr = indices + v_start; a_len = deg_v;
        b_ptr = indices + u_start; b_len = deg_u;
    }

    
    int count = 0;
    for (int i = lane; i < a_len; i += 32) {
        int32_t val = __ldg(&a_ptr[i]);

        int lb_b = lower_bound_dev(b_ptr, b_len, val);

        if (lb_b < b_len && __ldg(&b_ptr[lb_b]) == val) {
            int ub_b = upper_bound_dev(b_ptr, b_len, val);
            int count_b = ub_b - lb_b;
            int lb_a = lower_bound_dev(a_ptr, a_len, val);
            int rank_a = i - lb_a;

            if (rank_a < count_b) {
                count++;
            }
        }
    }

    count = warp_reduce_sum(count);

    if (lane == 0) {
        scores[warp_id] = 2.0f * count / (float)(deg_u + deg_v);
    }
}

}  

void sorensen_similarity(const graph32_t& graph,
                         const int32_t* vertex_pairs_first,
                         const int32_t* vertex_pairs_second,
                         std::size_t num_pairs,
                         float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    bool is_multigraph = graph.is_multigraph;

    int64_t np = static_cast<int64_t>(num_pairs);
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int grid = static_cast<int>((np + warps_per_block - 1) / warps_per_block);

    if (is_multigraph) {
        sorensen_multigraph_kernel<<<grid, threads_per_block>>>(
            offsets, indices, vertex_pairs_first, vertex_pairs_second,
            similarity_scores, np);
    } else {
        sorensen_simple_kernel<<<grid, threads_per_block>>>(
            offsets, indices, vertex_pairs_first, vertex_pairs_second,
            similarity_scores, np);
    }
}

}  
