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
#include <climits>

namespace aai {

namespace {

__device__ __forceinline__ int lower_bound_dev(const int32_t* __restrict__ arr, int len, int32_t target) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__device__ __forceinline__ int gallop_search(const int32_t* __restrict__ arr, int pos, int len, int32_t target) {
    if (pos >= len || arr[pos] >= target) return pos;

    
    int step = 1;
    while (pos + step < len && arr[pos + step] < target) {
        pos += step;
        step <<= 1;
    }

    
    int lo = pos + 1;
    int hi = pos + step;
    if (hi > len) hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}



template <bool IS_MULTIGRAPH>
__device__ __forceinline__ int intersection_merge(
    const int32_t* __restrict__ a_ptr, int a_len,
    const int32_t* __restrict__ b_ptr, int b_len,
    int lane
) {
    int chunk_size = (a_len + 31) >> 5;
    int a_start = lane * chunk_size;
    int a_end = a_start + chunk_size;
    if (a_end > a_len) a_end = a_len;
    if (a_start >= a_len) return 0;

    int32_t first_a = a_ptr[a_start];
    int j = lower_bound_dev(b_ptr, b_len, first_a);

    if constexpr (IS_MULTIGRAPH) {
        int dup_rank = a_start - lower_bound_dev(a_ptr, a_len, first_a);
        if (dup_rank > 0) {
            
            int ub = j;
            while (ub < b_len && b_ptr[ub] == first_a) ub++;
            int count_in_b = ub - j;
            j = (dup_rank < count_in_b) ? (j + dup_rank) : ub;
        }
    }

    int count = 0;
    for (int i = a_start; i < a_end; i++) {
        int32_t target = a_ptr[i];

        
        j = gallop_search(b_ptr, j, b_len, target);
        if (j >= b_len) break;

        if (b_ptr[j] == target) {
            count++;
            j++;
        }
    }

    return count;
}


__device__ __forceinline__ int intersection_bsearch(
    const int32_t* __restrict__ a_ptr, int a_len,
    const int32_t* __restrict__ b_ptr, int b_len,
    int lane
) {
    int count = 0;
    int search_lo = 0;

    for (int i = lane; i < a_len; i += 32) {
        int32_t target = a_ptr[i];

        int lo = search_lo, hi = b_len;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (b_ptr[mid] < target) lo = mid + 1;
            else hi = mid;
        }

        bool found = (lo < b_len && b_ptr[lo] == target);
        if (found) count++;
        search_lo = found ? lo + 1 : lo;
    }

    return count;
}


__device__ __forceinline__ int intersection_bsearch_multi(
    const int32_t* __restrict__ a_ptr, int a_len,
    const int32_t* __restrict__ b_ptr, int b_len,
    int lane
) {
    int count = 0;

    for (int i = lane; i < a_len; i += 32) {
        int32_t target = a_ptr[i];

        int pos_first_a = lower_bound_dev(a_ptr, a_len, target);
        int dup_rank = i - pos_first_a;
        int lo_b = lower_bound_dev(b_ptr, b_len, target);
        int pos = lo_b + dup_rank;
        if (pos < b_len && b_ptr[pos] == target) count++;
    }

    return count;
}

template <bool IS_MULTIGRAPH>
__global__ void sorensen_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int u = pairs_first[warp_id];
    int v = pairs_second[warp_id];

    if (u == v) {
        if (lane == 0) {
            int deg = offsets[u + 1] - offsets[u];
            scores[warp_id] = (deg > 0) ? 1.0f : 0.0f;
        }
        return;
    }

    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int v_start = offsets[v];
    int v_end = offsets[v + 1];
    int deg_u = u_end - u_start;
    int deg_v = v_end - v_start;
    int sum_deg = deg_u + deg_v;

    if (deg_u == 0 || deg_v == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    const int32_t* a_ptr, *b_ptr;
    int a_len, b_len;
    if (deg_u <= deg_v) {
        a_ptr = indices + u_start; a_len = deg_u;
        b_ptr = indices + v_start; b_len = deg_v;
    } else {
        a_ptr = indices + v_start; a_len = deg_v;
        b_ptr = indices + u_start; b_len = deg_u;
    }

    int count = 0;

    
    int32_t a_max = a_ptr[a_len - 1];
    int32_t b_min = b_ptr[0];
    int32_t b_max = b_ptr[b_len - 1];
    int32_t a_min = a_ptr[0];

    if (a_max >= b_min && b_max >= a_min) {
        
        
        
        bool use_merge = (a_len >= 48) && (b_len <= 8 * a_len);

        if (use_merge) {
            count = intersection_merge<IS_MULTIGRAPH>(a_ptr, a_len, b_ptr, b_len, lane);
        } else {
            if constexpr (IS_MULTIGRAPH) {
                count = intersection_bsearch_multi(a_ptr, a_len, b_ptr, b_len, lane);
            } else {
                count = intersection_bsearch(a_ptr, a_len, b_ptr, b_len, lane);
            }
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        count += __shfl_xor_sync(0xFFFFFFFF, count, offset);
    }

    if (lane == 0) {
        scores[warp_id] = 2.0f * (float)count / (float)sum_deg;
    }
}

}  

void sorensen_similarity_seg(const graph32_t& graph,
                             const int32_t* vertex_pairs_first,
                             const int32_t* vertex_pairs_second,
                             std::size_t num_pairs,
                             float* similarity_scores) {
    if (num_pairs == 0) return;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    bool is_multigraph = graph.is_multigraph;

    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    int64_t np = static_cast<int64_t>(num_pairs);
    int64_t blocks64 = (np + warps_per_block - 1) / warps_per_block;
    int blocks = (int)min(blocks64, (int64_t)INT_MAX);

    if (is_multigraph) {
        sorensen_warp_kernel<true><<<blocks, threads_per_block>>>(
            offsets, indices, vertex_pairs_first, vertex_pairs_second,
            similarity_scores, np);
    } else {
        sorensen_warp_kernel<false><<<blocks, threads_per_block>>>(
            offsets, indices, vertex_pairs_first, vertex_pairs_second,
            similarity_scores, np);
    }
}

}  
