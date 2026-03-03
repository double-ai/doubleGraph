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

__device__ __forceinline__ int lower_bound_d(const int32_t* __restrict__ arr, int start, int end, int target) {
    while (start < end) {
        int mid = start + ((end - start) >> 1);
        if (arr[mid] < target) start = mid + 1;
        else end = mid;
    }
    return start;
}


__global__ __launch_bounds__(256, 6)
void compute_sorensen_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ scores,
    int64_t num_pairs
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int u = first[warp_id];
    int v = second[warp_id];

    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int v_start = offsets[v];
    int v_end = offsets[v + 1];
    int u_deg = u_end - u_start;
    int v_deg = v_end - v_start;

    
    double my_sum = 0.0;
    for (int i = lane; i < u_deg; i += 32)
        my_sum += edge_weights[u_start + i];
    for (int offset = 16; offset > 0; offset >>= 1)
        my_sum += __shfl_down_sync(0xffffffff, my_sum, offset);
    double sum_u = __shfl_sync(0xffffffff, my_sum, 0);

    my_sum = 0.0;
    for (int i = lane; i < v_deg; i += 32)
        my_sum += edge_weights[v_start + i];
    for (int offset = 16; offset > 0; offset >>= 1)
        my_sum += __shfl_down_sync(0xffffffff, my_sum, offset);
    double sum_v = __shfl_sync(0xffffffff, my_sum, 0);

    double denom = sum_u + sum_v;
    if (denom == 0.0) { if (lane == 0) scores[warp_id] = 0.0; return; }

    int s_start, s_deg, l_start, l_end;
    if (u_deg <= v_deg) { s_start = u_start; s_deg = u_deg; l_start = v_start; l_end = v_end; }
    else { s_start = v_start; s_deg = v_deg; l_start = u_start; l_end = u_end; }

    double intersection = 0.0;
    int search_lo = l_start;

    for (int i = lane; i < s_deg; i += 32) {
        int pos = s_start + i;
        int target = indices[pos];
        double w_small = edge_weights[pos];

        
        int rank = 0;
        if (i > 0 && indices[pos - 1] == target) {
            rank = 1;
            for (int j = pos - 2; j >= s_start; j--) {
                if (indices[j] == target) rank++; else break;
            }
        }

        int lb = lower_bound_d(indices, search_lo, l_end, target);
        int match_pos = lb + rank;

        if (match_pos < l_end && indices[match_pos] == target) {
            intersection += fmin(w_small, edge_weights[match_pos]);
        }
        search_lo = lb;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        intersection += __shfl_down_sync(0xffffffff, intersection, offset);

    if (lane == 0) scores[warp_id] = 2.0 * intersection / denom;
}

}  

void sorensen_similarity_seg(const graph32_t& graph,
                             const double* edge_weights,
                             const int32_t* vertex_pairs_first,
                             const int32_t* vertex_pairs_second,
                             std::size_t num_pairs,
                             double* similarity_scores) {
    int64_t np = static_cast<int64_t>(num_pairs);
    int block = 256;
    int64_t grid64 = (np * 32 + block - 1) / block;
    int grid = static_cast<int>(grid64 < 2147483647LL ? grid64 : 2147483647LL);
    if (grid > 0)
        compute_sorensen_kernel<<<grid, block>>>(
            graph.offsets, graph.indices, edge_weights,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, np);
}

}  
