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



template <int GROUP_SIZE>
__global__ void cosine_similarity_subwarp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs)
{
    const int groups_per_warp = 32 / GROUP_SIZE;
    const int group_lane = threadIdx.x & (GROUP_SIZE - 1);
    const int group_in_warp = (threadIdx.x & 31) / GROUP_SIZE;
    const int warp_in_block = threadIdx.x >> 5;
    const int64_t pair_idx = (int64_t)blockIdx.x * (blockDim.x / GROUP_SIZE)
                           + warp_in_block * groups_per_warp + group_in_warp;

    
    const unsigned full_mask = 0xffffffff;
    const unsigned group_mask = ((1u << GROUP_SIZE) - 1u) << (group_in_warp * GROUP_SIZE);

    if (pair_idx >= num_pairs) return;

    int32_t u = pairs_first[pair_idx];
    int32_t v = pairs_second[pair_idx];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];

    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;

    if (deg_u == 0 || deg_v == 0) {
        if (group_lane == 0) scores[pair_idx] = 0.0f;
        return;
    }

    
    int32_t u_first = indices[u_start];
    int32_t u_last  = indices[u_end - 1];
    int32_t v_first = indices[v_start];
    int32_t v_last  = indices[v_end - 1];

    if (u_last < v_first || v_last < u_first) {
        if (group_lane == 0) scores[pair_idx] = 0.0f;
        return;
    }

    
    const int32_t* short_ptr;
    const int32_t* long_ptr;
    int32_t short_len, long_len;
    if (deg_u <= deg_v) {
        short_ptr = indices + u_start;
        long_ptr  = indices + v_start;
        short_len = deg_u;
        long_len  = deg_v;
    } else {
        short_ptr = indices + v_start;
        long_ptr  = indices + u_start;
        short_len = deg_v;
        long_len  = deg_u;
    }

    int found = 0;
    int32_t num_iters = (short_len + GROUP_SIZE - 1) / GROUP_SIZE;

    for (int32_t iter = 0; iter < num_iters; iter++) {
        int32_t si = iter * GROUP_SIZE + group_lane;
        if (si < short_len) {
            int32_t target = short_ptr[si];
            
            int32_t lo = 0, hi = long_len;
            while (lo < hi) {
                int32_t mid = (lo + hi) >> 1;
                if (long_ptr[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            if (lo < long_len && long_ptr[lo] == target) {
                found = 1;
            }
        }
        if (__any_sync(group_mask, found)) {
            found = 1;
            break;
        }
    }

    if (group_lane == 0) {
        scores[pair_idx] = found ? 1.0f : 0.0f;
    }
}

}  

void cosine_similarity_seg(const graph32_t& graph,
                           const int32_t* vertex_pairs_first,
                           const int32_t* vertex_pairs_second,
                           std::size_t num_pairs,
                           float* similarity_scores) {
    if (num_pairs == 0) return;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    
    const int GROUP_SIZE = 8;
    const int groups_per_block = 256 / GROUP_SIZE;  
    int grid = (int)((num_pairs + groups_per_block - 1) / groups_per_block);
    cosine_similarity_subwarp_kernel<GROUP_SIZE><<<grid, 256>>>(
        offsets, indices, vertex_pairs_first, vertex_pairs_second,
        similarity_scores, (int64_t)num_pairs);
}

}  
