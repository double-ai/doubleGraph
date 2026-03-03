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



template <bool IS_MULTIGRAPH, int TPP>  
__global__ void overlap_similarity_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    int64_t num_pairs,
    double* __restrict__ similarity_scores
) {
    int pair_id = (blockIdx.x * blockDim.x + threadIdx.x) / TPP;
    int lane = threadIdx.x % TPP;

    if (pair_id >= num_pairs) return;

    int32_t orig_u = pairs_first[pair_id];
    int32_t orig_v = pairs_second[pair_id];

    int32_t u_start = offsets[orig_u];
    int32_t u_end = offsets[orig_u + 1];
    int32_t v_start = offsets[orig_v];
    int32_t v_end = offsets[orig_v + 1];

    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    
    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) similarity_scores[pair_id] = 0.0;
        return;
    }

    
    const int32_t* a_nbrs, *b_nbrs;
    const double* a_wts, *b_wts;
    int32_t a_deg, b_deg;

    if (u_deg <= v_deg) {
        a_nbrs = indices + u_start; a_wts = edge_weights + u_start; a_deg = u_deg;
        b_nbrs = indices + v_start; b_wts = edge_weights + v_start; b_deg = v_deg;
    } else {
        a_nbrs = indices + v_start; a_wts = edge_weights + v_start; a_deg = v_deg;
        b_nbrs = indices + u_start; b_wts = edge_weights + u_start; b_deg = u_deg;
    }

    
    int32_t a_max = a_nbrs[a_deg - 1];
    int32_t b_min = b_nbrs[0];

    if (a_max < b_min) {
        
        int32_t a_min = a_nbrs[0];
        int32_t b_max = b_nbrs[b_deg - 1];
        if (b_max < a_min) {
            if (lane == 0) similarity_scores[pair_id] = 0.0;
            return;
        }
    }

    
    int32_t a_min = a_nbrs[0];
    int32_t b_max = b_nbrs[b_deg - 1];

    
    if (b_max < a_min) {
        if (lane == 0) similarity_scores[pair_id] = 0.0;
        return;
    }

    
    double wd_large = 0.0;
    for (int i = lane; i < b_deg; i += TPP) {
        wd_large += b_wts[i];
    }

    
    double sum = 0.0;
    double wd_small = 0.0;
    int search_lo = 0;

    for (int i = lane; i < a_deg; i += TPP) {
        double w_a = a_wts[i];
        wd_small += w_a;

        int32_t target = a_nbrs[i];
        if (target < b_min || target > b_max) continue;

        
        int lo = search_lo, hi = b_deg;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (b_nbrs[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        int lb = lo;
        search_lo = lb;

        if constexpr (IS_MULTIGRAPH) {
            int ub_lo = lb, ub_hi = b_deg;
            while (ub_lo < ub_hi) {
                int mid = ub_lo + ((ub_hi - ub_lo) >> 1);
                if (b_nbrs[mid] <= target) ub_lo = mid + 1;
                else ub_hi = mid;
            }
            int ub = ub_lo;

            int rank_lo = 0, rank_hi = i;
            while (rank_lo < rank_hi) {
                int mid = rank_lo + ((rank_hi - rank_lo) >> 1);
                if (a_nbrs[mid] < target) rank_lo = mid + 1;
                else rank_hi = mid;
            }
            int rank = i - rank_lo;

            int match_pos = lb + rank;
            if (match_pos < ub) {
                double w_b = b_wts[match_pos];
                sum += (w_a < w_b) ? w_a : w_b;
            }
        } else {
            if (lb < b_deg && b_nbrs[lb] == target) {
                double w_b = b_wts[lb];
                sum += (w_a < w_b) ? w_a : w_b;
            }
        }
    }

    
    #pragma unroll
    for (int offset = TPP / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset, TPP);
        wd_small += __shfl_down_sync(0xffffffff, wd_small, offset, TPP);
        wd_large += __shfl_down_sync(0xffffffff, wd_large, offset, TPP);
    }

    if (lane == 0) {
        double denom = (wd_small < wd_large) ? wd_small : wd_large;
        similarity_scores[pair_id] = (denom > 0.0) ? (sum / denom) : 0.0;
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

    if (num_pairs == 0) return;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    bool is_multigraph = graph.is_multigraph;

    constexpr int TPP = 32;  
    int threads = 256;
    int pairs_per_block = threads / TPP;
    int blocks = ((int)num_pairs + pairs_per_block - 1) / pairs_per_block;

    if (is_multigraph) {
        overlap_similarity_kernel<true, TPP><<<blocks, threads>>>(
            offsets, indices, edge_weights,
            vertex_pairs_first, vertex_pairs_second, (int64_t)num_pairs, similarity_scores);
    } else {
        overlap_similarity_kernel<false, TPP><<<blocks, threads>>>(
            offsets, indices, edge_weights,
            vertex_pairs_first, vertex_pairs_second, (int64_t)num_pairs, similarity_scores);
    }
}

}  
