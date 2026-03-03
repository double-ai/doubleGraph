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

__device__ __forceinline__ int lower_bound_dev(const int32_t* __restrict__ arr, int len, int32_t val) {
    int lo = 0;
    while (len > 1) {
        int half = len >> 1;
        lo += (arr[lo + half - 1] < val) ? half : 0;
        len -= half;
    }
    return lo + (len == 1 && arr[lo] < val);
}

__global__ __launch_bounds__(256, 8)
void sorensen_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    int64_t num_pairs,
    double* __restrict__ scores)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    const int u = pairs_first[warp_id];
    const int v = pairs_second[warp_id];

    const int u_start = offsets[u];
    const int u_end = offsets[u + 1];
    const int v_start = offsets[v];
    const int v_end = offsets[v + 1];
    const int deg_u = u_end - u_start;
    const int deg_v = v_end - v_start;

    
    double sum_u = 0.0;
    for (int i = lane; i < deg_u; i += 32)
        sum_u += edge_weights[u_start + i];

    double sum_v = 0.0;
    for (int i = lane; i < deg_v; i += 32)
        sum_v += edge_weights[v_start + i];

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_u += __shfl_down_sync(0xffffffff, sum_u, offset);
        sum_v += __shfl_down_sync(0xffffffff, sum_v, offset);
    }
    double denom = __shfl_sync(0xffffffff, sum_u, 0) + __shfl_sync(0xffffffff, sum_v, 0);

    if (denom == 0.0) {
        if (lane == 0) scores[warp_id] = 0.0;
        return;
    }

    
    
    int A_off, A_len, B_off, B_len;

    if (deg_u <= deg_v) {
        A_off = u_start; A_len = deg_u;
        B_off = v_start; B_len = deg_v;
    } else {
        A_off = v_start; A_len = deg_v;
        B_off = u_start; B_len = deg_u;
    }

    double intersection = 0.0;

    if (A_len > 0 && B_len > 0) {
        
        const int32_t b_first = indices[B_off];
        const int32_t b_last = indices[B_off + B_len - 1];

        for (int k = lane; k < A_len; k += 32) {
            int32_t a_val = indices[A_off + k];

            
            if (a_val > b_last) break;
            if (a_val < b_first) continue;

            
            int rank = 0;
            if (k > 0 && indices[A_off + k - 1] == a_val) {
                int j = k - 1;
                while (j >= 0 && indices[A_off + j] == a_val) {
                    rank++;
                    j--;
                }
            }

            
            int pos = lower_bound_dev(indices + B_off, B_len, a_val) + rank;

            if (pos < B_len && indices[B_off + pos] == a_val) {
                intersection += fmin(edge_weights[A_off + k], edge_weights[B_off + pos]);
            }
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        intersection += __shfl_down_sync(0xffffffff, intersection, offset);
    }

    if (lane == 0) {
        scores[warp_id] = 2.0 * intersection / denom;
    }
}

}  

void sorensen_similarity(const graph32_t& graph,
                         const double* edge_weights,
                         const int32_t* vertex_pairs_first,
                         const int32_t* vertex_pairs_second,
                         std::size_t num_pairs,
                         double* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    if (num_pairs == 0) return;

    const int WARPS_PER_BLOCK = 8;
    const int TPB = WARPS_PER_BLOCK * 32;
    const int blocks = (int)((num_pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    sorensen_kernel<<<blocks, TPB>>>(
        graph.offsets, graph.indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second,
        (int64_t)num_pairs, similarity_scores);
}

}  
