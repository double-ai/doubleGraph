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

__device__ __forceinline__ int lower_bound_dev(const int32_t* __restrict__ arr, int size, int target) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (__ldg(arr + mid) < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__global__ void cosine_sim_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int num_pairs
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int u = __ldg(pairs_first + warp_id);
    int v = __ldg(pairs_second + warp_id);

    int u_start = __ldg(offsets + u);
    int u_end = __ldg(offsets + u + 1);
    int v_start = __ldg(offsets + v);
    int v_end = __ldg(offsets + v + 1);

    int deg_u = u_end - u_start;
    int deg_v = v_end - v_start;

    
    if (deg_u == 0 || deg_v == 0) {
        if (lane == 0) scores[warp_id] = __int_as_float(0x7fc00000); 
        return;
    }

    
    const int32_t* small_idx;
    const float* small_wt;
    int small_size;
    const int32_t* large_idx;
    const float* large_wt;
    int large_size;

    if (deg_u <= deg_v) {
        small_idx = indices + u_start;
        small_wt = weights + u_start;
        small_size = deg_u;
        large_idx = indices + v_start;
        large_wt = weights + v_start;
        large_size = deg_v;
    } else {
        small_idx = indices + v_start;
        small_wt = weights + v_start;
        small_size = deg_v;
        large_idx = indices + u_start;
        large_wt = weights + u_start;
        large_size = deg_u;
    }

    float dot = 0.0f, ns = 0.0f, nl = 0.0f;

    for (int i = lane; i < small_size; i += 32) {
        int target = __ldg(small_idx + i);

        
        int p_large = lower_bound_dev(large_idx, large_size, target);

        if (p_large < large_size && __ldg(large_idx + p_large) == target) {
            
            int match_pos = p_large;

            
            if (i > 0 && __ldg(small_idx + i - 1) == target) {
                int p_small = lower_bound_dev(small_idx, small_size, target);
                int dup_offset = i - p_small;
                match_pos = p_large + dup_offset;

                if (match_pos >= large_size || __ldg(large_idx + match_pos) != target) {
                    continue;
                }
            }

            float ws = __ldg(small_wt + i);
            float wl = __ldg(large_wt + match_pos);
            dot += ws * wl;
            ns += ws * ws;
            nl += wl * wl;
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
        ns += __shfl_down_sync(0xffffffff, ns, offset);
        nl += __shfl_down_sync(0xffffffff, nl, offset);
    }

    if (lane == 0) {
        float denom = sqrtf(ns) * sqrtf(nl);
        scores[warp_id] = dot / denom;
    }
}

}  

void cosine_similarity_seg(const graph32_t& graph,
                           const float* edge_weights,
                           const int32_t* vertex_pairs_first,
                           const int32_t* vertex_pairs_second,
                           std::size_t num_pairs,
                           float* similarity_scores) {
    if (num_pairs == 0) return;

    int threads = 128;
    int warps_per_block = threads / 32;
    int blocks = (int)((num_pairs + warps_per_block - 1) / warps_per_block);

    cosine_sim_warp_kernel<<<blocks, threads>>>(
        graph.offsets,
        graph.indices,
        edge_weights,
        vertex_pairs_first,
        vertex_pairs_second,
        similarity_scores,
        (int)num_pairs
    );
}

}  
