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

__device__ __forceinline__ bool binary_search_exists(
    const int32_t* __restrict__ arr, int32_t size, int32_t target
) {
    int32_t lo = 0, hi = size;
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        int32_t val = __ldg(&arr[mid]);
        if (val < target) lo = mid + 1;
        else if (val > target) hi = mid;
        else return true;
    }
    return false;
}

__global__ void cosine_similarity_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int32_t u = __ldg(&first[idx]);
    int32_t v = __ldg(&second[idx]);

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);

    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;

    if (deg_u == 0 || deg_v == 0) {
        scores[idx] = 0.0f;
        return;
    }

    
    int32_t u_min = __ldg(&indices[u_start]);
    int32_t u_max = __ldg(&indices[u_end - 1]);
    int32_t v_min = __ldg(&indices[v_start]);
    int32_t v_max = __ldg(&indices[v_end - 1]);

    if (u_max < v_min || v_max < u_min) {
        scores[idx] = 0.0f;
        return;
    }

    
    const int32_t* a_ptr;
    const int32_t* b_ptr;
    int32_t a_size, b_size;

    if (deg_u <= deg_v) {
        a_ptr = indices + u_start;
        b_ptr = indices + v_start;
        a_size = deg_u;
        b_size = deg_v;
    } else {
        a_ptr = indices + v_start;
        b_ptr = indices + u_start;
        a_size = deg_v;
        b_size = deg_u;
    }

    
    
    if (b_size > a_size * 4) {
        
        for (int32_t i = 0; i < a_size; i++) {
            int32_t target = __ldg(&a_ptr[i]);
            if (binary_search_exists(b_ptr, b_size, target)) {
                scores[idx] = 1.0f;
                return;
            }
        }
    } else {
        
        int32_t i = 0, j = 0;
        while (i < a_size && j < b_size) {
            int32_t va = __ldg(&a_ptr[i]);
            int32_t vb = __ldg(&b_ptr[j]);
            if (va == vb) {
                scores[idx] = 1.0f;
                return;
            } else if (va < vb) {
                i++;
            } else {
                j++;
            }
        }
    }

    scores[idx] = 0.0f;
}



__global__ void cosine_similarity_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    const int lane = threadIdx.x & 31;
    const int64_t pair_idx = (int64_t)(blockIdx.x * blockDim.x + threadIdx.x) >> 5;

    if (pair_idx >= num_pairs) return;

    int32_t u = __ldg(&first[pair_idx]);
    int32_t v = __ldg(&second[pair_idx]);

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);

    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;

    if (deg_u == 0 || deg_v == 0) {
        if (lane == 0) scores[pair_idx] = 0.0f;
        return;
    }

    
    int32_t u_min = __ldg(&indices[u_start]);
    int32_t u_max = __ldg(&indices[u_end - 1]);
    int32_t v_min = __ldg(&indices[v_start]);
    int32_t v_max = __ldg(&indices[v_end - 1]);

    if (u_max < v_min || v_max < u_min) {
        if (lane == 0) scores[pair_idx] = 0.0f;
        return;
    }

    
    const int32_t* a_ptr;
    const int32_t* b_ptr;
    int32_t a_size, b_size;
    int32_t b_min, b_max;

    if (deg_u <= deg_v) {
        a_ptr = indices + u_start;
        b_ptr = indices + v_start;
        a_size = deg_u;
        b_size = deg_v;
        b_min = v_min;
        b_max = v_max;
    } else {
        a_ptr = indices + v_start;
        b_ptr = indices + u_start;
        a_size = deg_v;
        b_size = deg_u;
        b_min = u_min;
        b_max = u_max;
    }

    
    bool found = false;
    int32_t max_iters = (a_size + 31) >> 5;

    for (int32_t iter = 0; iter < max_iters; iter++) {
        int32_t i = iter * 32 + lane;
        if (i < a_size && !found) {
            int32_t target = __ldg(&a_ptr[i]);
            if (target >= b_min && target <= b_max) {
                if (binary_search_exists(b_ptr, b_size, target)) {
                    found = true;
                }
            }
        }
        if (__any_sync(0xFFFFFFFF, found)) {
            if (lane == 0) scores[pair_idx] = 1.0f;
            return;
        }
    }

    if (lane == 0) scores[pair_idx] = 0.0f;
}

}  

void cosine_similarity(const graph32_t& graph,
                       const int32_t* vertex_pairs_first,
                       const int32_t* vertex_pairs_second,
                       std::size_t num_pairs,
                       float* similarity_scores) {
    if (num_pairs == 0) return;

    
    int block = 256;
    int64_t grid = ((int64_t)num_pairs + block - 1) / block;
    cosine_similarity_kernel<<<(int)grid, block>>>(
        graph.offsets, graph.indices,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, (int64_t)num_pairs);
}

}  
