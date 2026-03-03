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

struct Cache : Cacheable {};

#define WARPS_PER_BLOCK 8
#define SMEM_PER_WARP 512

__device__ __forceinline__ int32_t lower_bound_dev(
    const int32_t* __restrict__ arr, int32_t lo, int32_t hi, int32_t val
) {
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void sorensen_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    extern __shared__ int32_t smem[];

    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t pair_id = (int64_t)blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (pair_id >= num_pairs) return;

    const int32_t u = pairs_first[pair_id];
    const int32_t v = pairs_second[pair_id];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];
    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    
    int32_t a_start, a_deg, b_start, b_deg;
    if (u_deg <= v_deg) {
        a_start = u_start; a_deg = u_deg; b_start = v_start; b_deg = v_deg;
    } else {
        a_start = v_start; a_deg = v_deg; b_start = u_start; b_deg = u_deg;
    }

    if (a_deg == 0) {
        if (lane == 0) scores[pair_id] = 0.0f;
        return;
    }

    float local_sum = 0.0f;
    float deg_sum;

    if (b_deg <= 32) {
        
        int32_t b_idx = (lane < b_deg) ? indices[b_start + lane] : INT32_MAX;
        float b_w = (lane < b_deg) ? edge_weights[b_start + lane] : 0.0f;
        int32_t a_val = (lane < a_deg) ? indices[a_start + lane] : INT32_MAX;
        float a_w = (lane < a_deg) ? edge_weights[a_start + lane] : 0.0f;

        
        float as = a_w, bs = b_w;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) { as += __shfl_down_sync(0xffffffff, as, o); bs += __shfl_down_sync(0xffffffff, bs, o); }
        deg_sum = __shfl_sync(0xffffffff, as, 0) + __shfl_sync(0xffffffff, bs, 0);

        
        int lo = 0, hi = b_deg;
        #pragma unroll 6
        for (int s = 0; s < 6; s++) {
            int mid = (lo + hi) >> 1;
            int32_t mv = __shfl_sync(0xffffffff, b_idx, mid);
            if (lo < hi) { if (mv < a_val) lo = mid + 1; else hi = mid; }
        }

        
        int32_t mk = (lane < a_deg) ? a_val : (INT32_MIN + lane);
        unsigned int mm = __match_any_sync(0xffffffff, mk);
        lo += __popc(mm & ((1u << lane) - 1));

        int src = lo & 31;
        int32_t found = __shfl_sync(0xffffffff, b_idx, src);
        float fw = __shfl_sync(0xffffffff, b_w, src);
        if (lane < a_deg && lo < b_deg && found == a_val)
            local_sum = fminf(a_w, fw);

    } else if (b_deg <= SMEM_PER_WARP) {
        
        int32_t* bs = smem + warp_in_block * SMEM_PER_WARP;

        float bwd = 0.0f;
        for (int i = lane; i < b_deg; i += 32) {
            bs[i] = indices[b_start + i];
            bwd += edge_weights[b_start + i];
        }
        __syncwarp();
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) bwd += __shfl_down_sync(0xffffffff, bwd, o);
        float b_wdeg = __shfl_sync(0xffffffff, bwd, 0);

        float awd = 0.0f;
        int32_t smem_lo = 0; 
        for (int32_t i = lane; i < a_deg; i += 32) {
            int32_t ai = a_start + i;
            int32_t nb = indices[ai];
            float wa = edge_weights[ai];
            awd += wa;

            int32_t lo = smem_lo, hi = b_deg;
            while (lo < hi) { int32_t mid = lo + ((hi - lo) >> 1); if (bs[mid] < nb) lo = mid + 1; else hi = mid; }
            smem_lo = lo;

            if (i > 0 && indices[ai - 1] == nb) {
                int32_t fa = lower_bound_dev(indices, a_start, ai, nb);
                lo += (ai - fa);
            }
            if (lo < b_deg && bs[lo] == nb)
                local_sum += fminf(wa, edge_weights[b_start + lo]);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) awd += __shfl_down_sync(0xffffffff, awd, o);
        deg_sum = __shfl_sync(0xffffffff, awd, 0) + b_wdeg;

    } else {
        
        float bwd = 0.0f;
        for (int i = lane; i < b_deg; i += 32)
            bwd += edge_weights[b_start + i];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) bwd += __shfl_down_sync(0xffffffff, bwd, o);
        float b_wdeg = __shfl_sync(0xffffffff, bwd, 0);

        float awd = 0.0f;
        int32_t b_lo = b_start; 

        for (int32_t i = lane; i < a_deg; i += 32) {
            int32_t ai = a_start + i;
            int32_t nb = indices[ai];
            float wa = edge_weights[ai];
            awd += wa;

            
            int32_t lo = b_lo, hi = b_start + b_deg;
            while (lo < hi) { int32_t mid = lo + ((hi - lo) >> 1); if (indices[mid] < nb) lo = mid + 1; else hi = mid; }

            b_lo = lo; 
            int32_t pos = lo - b_start;

            if (i > 0 && indices[ai - 1] == nb) {
                int32_t fa = lower_bound_dev(indices, a_start, ai, nb);
                pos += (ai - fa);
            }
            if (pos < b_deg && indices[b_start + pos] == nb)
                local_sum += fminf(wa, edge_weights[b_start + pos]);
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) awd += __shfl_down_sync(0xffffffff, awd, o);
        deg_sum = __shfl_sync(0xffffffff, awd, 0) + b_wdeg;
    }

    
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) local_sum += __shfl_down_sync(0xffffffff, local_sum, o);

    if (lane == 0)
        scores[pair_id] = (deg_sum > 0.0f) ? (2.0f * local_sum / deg_sum) : 0.0f;
}

}  

void sorensen_similarity_seg(const graph32_t& graph,
                             const float* edge_weights,
                             const int32_t* vertex_pairs_first,
                             const int32_t* vertex_pairs_second,
                             std::size_t num_pairs,
                             float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    if (num_pairs == 0) return;

    int block = WARPS_PER_BLOCK * 32;
    int smem_size = WARPS_PER_BLOCK * SMEM_PER_WARP * (int)sizeof(int32_t);
    int64_t grid = ((int64_t)num_pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    sorensen_kernel<<<(int)grid, block, smem_size>>>(
        graph.offsets, graph.indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, (int64_t)num_pairs);
}

}  
