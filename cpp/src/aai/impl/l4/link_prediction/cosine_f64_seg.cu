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
#include <math_constants.h>
#include <math.h>
#include <cstdint>
#include <cstddef>

namespace aai {

namespace {

struct Cache : Cacheable {};

__device__ __forceinline__ int32_t dev_lower_bound(
    const int32_t* __restrict__ arr, int32_t lo, int32_t hi, int32_t target) {
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}




__global__ void cosine_sim_warp_bsearch(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ scores,
    int64_t num_pairs
) {
    const int lane = threadIdx.x & 31;
    const int64_t pair_idx = (int64_t)(blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (pair_idx >= num_pairs) return;

    int32_t u = first[pair_idx];
    int32_t v = second[pair_idx];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];

    int32_t u_len = u_end - u_start;
    int32_t v_len = v_end - v_start;

    if (u_len == 0 || v_len == 0) {
        if (lane == 0) scores[pair_idx] = CUDART_NAN;
        return;
    }

    if (u == v) {
        if (lane == 0) scores[pair_idx] = 1.0;
        return;
    }

    
    int32_t s_start, s_len, l_start, l_end, l_len;
    bool swapped;
    if (u_len <= v_len) {
        s_start = u_start; s_len = u_len;
        l_start = v_start; l_end = v_end; l_len = v_len;
        swapped = false;
    } else {
        s_start = v_start; s_len = v_len;
        l_start = u_start; l_end = u_end; l_len = u_len;
        swapped = true;
    }

    
    int32_t s_lo_val, s_hi_val, l_lo_val, l_hi_val;
    if (lane == 0) {
        s_lo_val = indices[s_start];
        s_hi_val = indices[s_start + s_len - 1];
        l_lo_val = indices[l_start];
        l_hi_val = indices[l_start + l_len - 1];
    }
    s_lo_val = __shfl_sync(0xffffffff, s_lo_val, 0);
    s_hi_val = __shfl_sync(0xffffffff, s_hi_val, 0);
    l_lo_val = __shfl_sync(0xffffffff, l_lo_val, 0);
    l_hi_val = __shfl_sync(0xffffffff, l_hi_val, 0);

    if (s_hi_val < l_lo_val || l_hi_val < s_lo_val) {
        if (lane == 0) scores[pair_idx] = CUDART_NAN;
        return;
    }

    
    int32_t search_lo = l_start;
    int32_t search_hi = l_end;
    if (lane == 0) {
        if (s_lo_val > l_lo_val)
            search_lo = dev_lower_bound(indices, l_start, l_end, s_lo_val);
        if (s_hi_val < l_hi_val)
            search_hi = dev_lower_bound(indices, search_lo, l_end, s_hi_val + 1);
    }
    search_lo = __shfl_sync(0xffffffff, search_lo, 0);
    search_hi = __shfl_sync(0xffffffff, search_hi, 0);

    double dot = 0.0, norm_u_sq = 0.0, norm_v_sq = 0.0;

    for (int32_t i = lane; i < s_len; i += 32) {
        int32_t key = indices[s_start + i];
        int32_t pos = dev_lower_bound(indices, search_lo, search_hi, key);

        if (pos < search_hi && indices[pos] == key) {
            double w_short = edge_weights[s_start + i];
            double w_long = edge_weights[pos];
            double wu = swapped ? w_long : w_short;
            double wv = swapped ? w_short : w_long;
            dot += wu * wv;
            norm_u_sq += wu * wu;
            norm_v_sq += wv * wv;
        }
    }

    
    if (!__any_sync(0xffffffff, dot != 0.0)) {
        if (lane == 0) scores[pair_idx] = CUDART_NAN;
        return;
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
        norm_u_sq += __shfl_down_sync(0xffffffff, norm_u_sq, offset);
        norm_v_sq += __shfl_down_sync(0xffffffff, norm_v_sq, offset);
    }

    if (lane == 0) {
        scores[pair_idx] = dot / (sqrt(norm_u_sq) * sqrt(norm_v_sq));
    }
}




__global__ void cosine_sim_thread_merge(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int32_t u = first[idx];
    int32_t v = second[idx];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];

    if (u_start >= u_end || v_start >= v_end) {
        scores[idx] = CUDART_NAN;
        return;
    }

    if (u == v) {
        scores[idx] = 1.0;
        return;
    }

    double dot = 0.0, norm_u_sq = 0.0, norm_v_sq = 0.0;
    int32_t i = u_start, j = v_start;

    while (i < u_end && j < v_end) {
        int32_t a = indices[i];
        int32_t b = indices[j];
        if (a == b) {
            double wu = edge_weights[i];
            double wv = edge_weights[j];
            dot += wu * wv;
            norm_u_sq += wu * wu;
            norm_v_sq += wv * wv;
            ++i; ++j;
        } else if (a < b) {
            ++i;
        } else {
            ++j;
        }
    }

    scores[idx] = dot / (sqrt(norm_u_sq) * sqrt(norm_v_sq));
}

}  

void cosine_similarity_seg(const graph32_t& graph,
                           const double* edge_weights,
                           const int32_t* vertex_pairs_first,
                           const int32_t* vertex_pairs_second,
                           std::size_t num_pairs,
                           double* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    if (num_pairs == 0) return;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    if (graph.is_multigraph) {
        int block = 256;
        int grid = (int)((num_pairs + block - 1) / block);
        cosine_sim_thread_merge<<<grid, block>>>(
            d_offsets, d_indices, edge_weights,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int64_t)num_pairs);
    } else {
        int block = 256;
        int grid = (int)((num_pairs * 32 + block - 1) / block);
        cosine_sim_warp_bsearch<<<grid, block>>>(
            d_offsets, d_indices, edge_weights,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int64_t)num_pairs);
    }
}

}  
