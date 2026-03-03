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
#include <math_constants.h>

namespace aai {

namespace {

struct Cache : Cacheable {};


__global__ void cosine_sim_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs)
{
    const int warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    const int32_t u = first[warp_id];
    const int32_t v = second[warp_id];

    const int32_t u_start = offsets[u];
    const int32_t u_end   = offsets[u + 1];
    const int32_t v_start = offsets[v];
    const int32_t v_end   = offsets[v + 1];

    const int32_t u_len = u_end - u_start;
    const int32_t v_len = v_end - v_start;

    
    if (u_len == 0 || v_len == 0) {
        if (lane == 0) scores[warp_id] = CUDART_NAN_F;
        return;
    }

    
    int32_t a_start, a_len, b_start, b_end;
    if (u_len <= v_len) {
        a_start = u_start; a_len = u_len;
        b_start = v_start; b_end = v_end;
    } else {
        a_start = v_start; a_len = v_len;
        b_start = u_start; b_end = u_end;
    }

    float dot = 0.0f, na_sq = 0.0f, nb_sq = 0.0f;

    for (int32_t t = lane; t < a_len; t += 32) {
        const int32_t a_pos = a_start + t;
        const int32_t x = __ldg(&indices[a_pos]);

        
        int32_t dup = 0;
        for (int32_t k = a_pos - 1; k >= a_start && __ldg(&indices[k]) == x; k--)
            dup++;

        
        int32_t lo = b_start, hi = b_end;
        while (lo < hi) {
            int32_t mid = lo + ((hi - lo) >> 1);
            if (__ldg(&indices[mid]) < x) lo = mid + 1;
            else hi = mid;
        }

        const int32_t match = lo + dup;
        if (match < b_end && __ldg(&indices[match]) == x) {
            const float wa = __ldg(&weights[a_pos]);
            const float wb = __ldg(&weights[match]);
            dot   += wa * wb;
            na_sq += wa * wa;
            nb_sq += wb * wb;
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot   += __shfl_down_sync(0xffffffff, dot, offset);
        na_sq += __shfl_down_sync(0xffffffff, na_sq, offset);
        nb_sq += __shfl_down_sync(0xffffffff, nb_sq, offset);
    }

    if (lane == 0) {
        
        
        float denom = sqrtf(na_sq) * sqrtf(nb_sq);
        if (denom == 0.0f) {
            scores[warp_id] = CUDART_NAN_F;
        } else {
            scores[warp_id] = dot / denom;
        }
    }
}

}  

void cosine_similarity_seg(const graph32_t& graph,
                           const float* edge_weights,
                           const int32_t* vertex_pairs_first,
                           const int32_t* vertex_pairs_second,
                           std::size_t num_pairs,
                           float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    if (num_pairs == 0) return;

    const int block = 256;
    const int warps_per_block = block / 32;
    const int grid = (int)((num_pairs + warps_per_block - 1) / warps_per_block);
    cosine_sim_warp_kernel<<<grid, block>>>(
        graph.offsets,
        graph.indices,
        edge_weights,
        vertex_pairs_first,
        vertex_pairs_second,
        similarity_scores,
        (int64_t)num_pairs);
}

}  
