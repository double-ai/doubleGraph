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


#define LDG(ptr) __ldg(ptr)


__device__ __forceinline__ int lower_bound_dev(const int32_t* __restrict__ arr, int len, int32_t val) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        
        int cmp = (LDG(arr + mid) < val) ? 1 : 0;
        lo = cmp ? (mid + 1) : lo;
        hi = cmp ? hi : mid;
    }
    return lo;
}


template <int GROUP_SIZE>
__global__ __launch_bounds__(256, 2)
void overlap_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ similarity_scores,
    int64_t num_pairs
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t pair_id = tid / GROUP_SIZE;
    const int lane = tid & (GROUP_SIZE - 1);

    if (pair_id >= num_pairs) return;

    
    const int32_t u = LDG(first + pair_id);
    const int32_t v = LDG(second + pair_id);

    const int32_t u_start = LDG(offsets + u);
    const int32_t u_end = LDG(offsets + u + 1);
    const int32_t v_start = LDG(offsets + v);
    const int32_t v_end = LDG(offsets + v + 1);
    const int32_t u_deg = u_end - u_start;
    const int32_t v_deg = v_end - v_start;

    
    double u_wsum = 0.0;
    #pragma unroll 4
    for (int32_t i = lane; i < u_deg; i += GROUP_SIZE) {
        u_wsum += LDG(edge_weights + u_start + i);
    }
    #pragma unroll
    for (int s = GROUP_SIZE / 2; s > 0; s >>= 1) {
        u_wsum += __shfl_down_sync(0xffffffff, u_wsum, s, GROUP_SIZE);
    }
    u_wsum = __shfl_sync(0xffffffff, u_wsum, 0, GROUP_SIZE);

    double v_wsum = 0.0;
    #pragma unroll 4
    for (int32_t i = lane; i < v_deg; i += GROUP_SIZE) {
        v_wsum += LDG(edge_weights + v_start + i);
    }
    #pragma unroll
    for (int s = GROUP_SIZE / 2; s > 0; s >>= 1) {
        v_wsum += __shfl_down_sync(0xffffffff, v_wsum, s, GROUP_SIZE);
    }
    v_wsum = __shfl_sync(0xffffffff, v_wsum, 0, GROUP_SIZE);

    double denom = fmin(u_wsum, v_wsum);
    if (denom == 0.0) {
        if (lane == 0) similarity_scores[pair_id] = 0.0;
        return;
    }

    
    int32_t A_off, A_len, B_off, B_len;
    if (u_deg <= v_deg) {
        A_off = u_start; A_len = u_deg; B_off = v_start; B_len = v_deg;
    } else {
        A_off = v_start; A_len = v_deg; B_off = u_start; B_len = u_deg;
    }

    double isect = 0.0;

    if (A_len > 0 && B_len > 0) {
        
        int32_t A_lo = 0, A_hi = A_len;
        if (lane == 0) {
            int32_t b_first = LDG(indices + B_off);
            A_lo = lower_bound_dev(indices + A_off, A_len, b_first);
            if (A_lo < A_len) {
                int32_t b_last = LDG(indices + B_off + B_len - 1);
                int lo2 = A_lo, hi2 = A_len;
                while (lo2 < hi2) {
                    int mid = (lo2 + hi2) >> 1;
                    if (LDG(indices + A_off + mid) <= b_last) lo2 = mid + 1;
                    else hi2 = mid;
                }
                A_hi = lo2;
            } else {
                A_hi = A_lo;
            }
        }
        A_lo = __shfl_sync(0xffffffff, A_lo, 0, GROUP_SIZE);
        A_hi = __shfl_sync(0xffffffff, A_hi, 0, GROUP_SIZE);

        int32_t A_range = A_hi - A_lo;

        
        for (int32_t idx = lane; idx < A_range; idx += GROUP_SIZE) {
            int32_t i = A_lo + idx;
            int32_t val = LDG(indices + A_off + i);

            
            int rank = 0;
            int k = i - 1;
            while (k >= 0 && LDG(indices + A_off + k) == val) {
                rank++;
                k--;
            }

            
            int lo = 0, hi = B_len;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (LDG(indices + B_off + mid) < val) lo = mid + 1;
                else hi = mid;
            }

            int pos = lo + rank;
            if (pos < B_len && LDG(indices + B_off + pos) == val) {
                isect = __fma_rn(1.0, fmin(LDG(edge_weights + A_off + i), LDG(edge_weights + B_off + pos)), isect);
            }
        }
    }

    
    #pragma unroll
    for (int s = GROUP_SIZE / 2; s > 0; s >>= 1) {
        isect += __shfl_down_sync(0xffffffff, isect, s, GROUP_SIZE);
    }

    if (lane == 0) {
        
        similarity_scores[pair_id] = __fma_rn(isect, 1.0 / denom, 0.0);
    }
}

}  

void overlap_similarity(const graph32_t& graph,
                        const double* edge_weights,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        double* similarity_scores) {
    if (num_pairs == 0) return;

    const int GROUP = 16;
    const int BLOCK = 256;
    const int pairs_per_block = BLOCK / GROUP;
    const int nblocks = (int)((num_pairs + pairs_per_block - 1) / pairs_per_block);
    overlap_kernel<GROUP><<<nblocks, BLOCK>>>(
        graph.offsets, graph.indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, (int64_t)num_pairs);
}

}  
