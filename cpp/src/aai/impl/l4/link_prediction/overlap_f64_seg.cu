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

namespace aai {

namespace {

struct Cache : Cacheable {};


__global__ void __launch_bounds__(256)
overlap_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ scores,
    int64_t num_pairs
) {
    const int lane = threadIdx.x & 31;
    const int warp_local = threadIdx.x >> 5;
    const int64_t warp_global = (int64_t)blockIdx.x * (blockDim.x >> 5) + warp_local;

    if (warp_global >= num_pairs) return;

    const int32_t u = __ldg(&first[warp_global]);
    const int32_t v = __ldg(&second[warp_global]);

    const int32_t u_start = __ldg(&offsets[u]);
    const int32_t u_end = __ldg(&offsets[u + 1]);
    const int32_t v_start = __ldg(&offsets[v]);
    const int32_t v_end = __ldg(&offsets[v + 1]);

    const int32_t u_deg = u_end - u_start;
    const int32_t v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) scores[warp_global] = 0.0;
        return;
    }

    
    double wd_u = 0.0;
    for (int32_t i = lane; i < u_deg; i += 32) {
        wd_u += __ldg(&edge_weights[u_start + i]);
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        wd_u += __shfl_xor_sync(0xffffffff, wd_u, mask);

    
    double wd_v = 0.0;
    for (int32_t i = lane; i < v_deg; i += 32) {
        wd_v += __ldg(&edge_weights[v_start + i]);
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        wd_v += __shfl_xor_sync(0xffffffff, wd_v, mask);

    double min_wd = fmin(wd_u, wd_v);
    if (min_wd == 0.0) {
        if (lane == 0) scores[warp_global] = 0.0;
        return;
    }

    
    const int32_t* A_idx;
    const double* A_wt;
    int32_t A_len;
    const int32_t* B_idx;
    const double* B_wt;
    int32_t B_len;

    if (u_deg <= v_deg) {
        A_idx = indices + u_start; A_wt = edge_weights + u_start; A_len = u_deg;
        B_idx = indices + v_start; B_wt = edge_weights + v_start; B_len = v_deg;
    } else {
        A_idx = indices + v_start; A_wt = edge_weights + v_start; A_len = v_deg;
        B_idx = indices + u_start; B_wt = edge_weights + u_start; B_len = u_deg;
    }

    
    int32_t a_last = __ldg(&A_idx[A_len - 1]);
    int32_t b_first = __ldg(&B_idx[0]);
    if (a_last < b_first) {
        if (lane == 0) scores[warp_global] = 0.0;
        return;
    }
    int32_t a_first = __ldg(&A_idx[0]);
    int32_t b_last = __ldg(&B_idx[B_len - 1]);
    if (b_last < a_first) {
        if (lane == 0) scores[warp_global] = 0.0;
        return;
    }

    
    
    
    double local_inter = 0.0;
    int32_t b_lo = 0;  

    for (int32_t i = lane; i < A_len; i += 32) {
        int32_t target = __ldg(&A_idx[i]);

        
        if (target > b_last) break;

        double w_a = __ldg(&A_wt[i]);

        
        int32_t rank = 0;
        if (i > 0 && __ldg(&A_idx[i - 1]) == target) {
            rank = 1;
            int32_t k = i - 2;
            while (k >= 0 && __ldg(&A_idx[k]) == target) { rank++; k--; }
        }

        
        int32_t lo = b_lo, hi = B_len;
        while (lo < hi) {
            int32_t mid = (lo + hi) >> 1;
            if (__ldg(&B_idx[mid]) < target) lo = mid + 1;
            else hi = mid;
        }

        
        b_lo = lo;

        
        int32_t match_pos = lo + rank;
        if (match_pos < B_len && __ldg(&B_idx[match_pos]) == target) {
            local_inter += fmin(w_a, __ldg(&B_wt[match_pos]));
        }
    }

    
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        local_inter += __shfl_xor_sync(0xffffffff, local_inter, mask);

    if (lane == 0) {
        scores[warp_global] = local_inter / min_wd;
    }
}

}  

void overlap_similarity_seg(const graph32_t& graph,
                            const double* edge_weights,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            double* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    int block = 256;
    int warps_per_block = block / 32;
    int grid = ((int)num_pairs + warps_per_block - 1) / warps_per_block;
    overlap_warp_kernel<<<grid, block>>>(
        offsets, indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, (int64_t)num_pairs
    );
}

}  
