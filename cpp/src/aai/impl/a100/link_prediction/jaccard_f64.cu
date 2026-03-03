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

#define WARP_SIZE 32

__global__ void jaccard_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ similarity_scores,
    int64_t num_pairs
) {
    const int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane_id = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int32_t u = first[warp_id];
    int32_t v = second[warp_id];

    
    if (u == v) {
        if (lane_id == 0) {
            int32_t deg = offsets[u + 1] - offsets[u];
            similarity_scores[warp_id] = (deg > 0) ? 1.0 : 0.0;
        }
        return;
    }

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];
    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;

    if (deg_u == 0 || deg_v == 0) {
        if (lane_id == 0) similarity_scores[warp_id] = 0.0;
        return;
    }

    
    int32_t small_start, small_deg, large_start, large_end, large_deg;
    if (deg_u <= deg_v) {
        small_start = u_start; small_deg = deg_u;
        large_start = v_start; large_end = v_end; large_deg = deg_v;
    } else {
        small_start = v_start; small_deg = deg_v;
        large_start = u_start; large_end = u_end; large_deg = deg_u;
    }

    
    int32_t small_first = indices[small_start];
    int32_t small_last = indices[small_start + small_deg - 1];
    int32_t large_first = indices[large_start];
    int32_t large_last = indices[large_end - 1];

    if (small_last < large_first || large_last < small_first) {
        if (lane_id == 0) similarity_scores[warp_id] = 0.0;
        return;
    }

    
    
    double my_w_small = 0.0;
    double my_intersection = 0.0;

    for (int32_t idx = lane_id; idx < small_deg; idx += WARP_SIZE) {
        int32_t target = indices[small_start + idx];
        double small_weight = edge_weights[small_start + idx];
        my_w_small += small_weight;

        
        if (target >= large_first && target <= large_last) {
            
            int32_t dup_count = 0;
            if (idx > 0 && indices[small_start + idx - 1] == target) {
                dup_count = 1;
                for (int32_t prev = idx - 2; prev >= 0; prev--) {
                    if (indices[small_start + prev] == target) dup_count++;
                    else break;
                }
            }

            
            int32_t lo = large_start, hi = large_end;
            while (lo < hi) {
                int32_t mid = lo + ((hi - lo) >> 1);
                if (indices[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            lo += dup_count;

            if (lo < large_end && indices[lo] == target) {
                my_intersection += fmin(small_weight, edge_weights[lo]);
            }
        }
    }

    
    for (int offset = 16; offset > 0; offset >>= 1) {
        my_w_small += __shfl_down_sync(0xffffffff, my_w_small, offset);
        my_intersection += __shfl_down_sync(0xffffffff, my_intersection, offset);
    }

    
    
    double intersection = __shfl_sync(0xffffffff, my_intersection, 0);

    if (intersection > 0.0) {
        double w_small = __shfl_sync(0xffffffff, my_w_small, 0);

        
        double w_large = 0.0;
        for (int32_t i = lane_id; i < large_deg; i += WARP_SIZE)
            w_large += edge_weights[large_start + i];
        for (int offset = 16; offset > 0; offset >>= 1)
            w_large += __shfl_down_sync(0xffffffff, w_large, offset);

        if (lane_id == 0) {
            double union_total = w_small + w_large;
            double denominator = union_total - intersection;
            similarity_scores[warp_id] = (denominator > 0.0) ? intersection / denominator : 0.0;
        }
    } else {
        if (lane_id == 0) similarity_scores[warp_id] = 0.0;
    }
}

}  

void jaccard_similarity(const graph32_t& graph,
                        const double* edge_weights,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        double* similarity_scores) {
    if (num_pairs <= 0) return;
    int threads_per_block = 256;
    int64_t total_threads = (int64_t)num_pairs * (int64_t)WARP_SIZE;
    int64_t grid = (total_threads + threads_per_block - 1) / threads_per_block;
    jaccard_warp_kernel<<<(int)grid, threads_per_block>>>(
        graph.offsets, graph.indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, (int64_t)num_pairs);
}

}  
