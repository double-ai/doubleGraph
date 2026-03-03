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
#include <math.h>

namespace aai {

namespace {

__device__ __forceinline__ int32_t lower_bound(
    const int32_t* __restrict__ arr,
    int32_t start, int32_t end, int32_t target)
{
    while (start < end) {
        int32_t mid = start + ((end - start) >> 1);
        if (__ldg(&arr[mid]) < target)
            start = mid + 1;
        else
            end = mid;
    }
    return start;
}

__device__ __forceinline__ int32_t upper_bound(
    const int32_t* __restrict__ arr,
    int32_t start, int32_t end, int32_t target)
{
    while (start < end) {
        int32_t mid = start + ((end - start) >> 1);
        if (__ldg(&arr[mid]) <= target)
            start = mid + 1;
        else
            end = mid;
    }
    return start;
}

__global__ void cosine_similarity_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    int64_t num_pairs,
    double* __restrict__ scores)
{
    const int64_t global_thread = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t warp_id = global_thread >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int32_t u = first[warp_id];
    int32_t v = second[warp_id];

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);

    int32_t d_u = u_end - u_start;
    int32_t d_v = v_end - v_start;

    
    if (u == v) {
        if (lane == 0) scores[warp_id] = (d_u > 0) ? 1.0 : (0.0 / 0.0);
        return;
    }

    
    if (d_u > d_v) {
        int32_t tmp;
        tmp = u_start; u_start = v_start; v_start = tmp;
        tmp = u_end; u_end = v_end; v_end = tmp;
        tmp = d_u; d_u = d_v; d_v = tmp;
    }

    if (d_u == 0) {
        if (lane == 0) scores[warp_id] = 0.0 / 0.0;
        return;
    }

    
    int32_t v_first = __ldg(&indices[v_start]);
    int32_t v_last = __ldg(&indices[v_end - 1]);
    int32_t u_first = __ldg(&indices[u_start]);
    int32_t u_last = __ldg(&indices[u_end - 1]);

    
    if (u_last < v_first || v_last < u_first) {
        if (lane == 0) scores[warp_id] = 0.0 / 0.0;
        return;
    }

    
    int32_t vs = v_start;
    int32_t ve = v_end;
    if (u_first > v_first) vs = lower_bound(indices, v_start, v_end, u_first);
    if (u_last < v_last) ve = upper_bound(indices, vs, v_end, u_last);

    double thread_dot = 0.0, thread_nu = 0.0, thread_nv = 0.0;

    
    
    
    int32_t v_lo = vs;
    int32_t prev_u_nbr = -1;  

    for (int32_t i = lane; i < d_u; i += 32) {
        int32_t u_idx = u_start + i;
        int32_t u_nbr = __ldg(&indices[u_idx]);

        
        
        if (u_nbr > v_last) break;

        
        if (u_nbr < v_first) {
            prev_u_nbr = u_nbr;
            continue;
        }

        
        int32_t u_rank = 0;
        int32_t search_lo = v_lo;

        if (u_idx > u_start && __ldg(&indices[u_idx - 1]) == u_nbr) {
            
            int32_t first_u = lower_bound(indices, u_start, u_idx, u_nbr);
            u_rank = u_idx - first_u;
        }

        
        if (u_nbr == prev_u_nbr) {
            search_lo = vs;
        }

        
        int32_t fv = lower_bound(indices, search_lo, ve, u_nbr);
        int32_t v_pos = fv + u_rank;

        if (v_pos < ve && __ldg(&indices[v_pos]) == u_nbr) {
            double wu = __ldg(&weights[u_idx]);
            double wv = __ldg(&weights[v_pos]);
            thread_dot += wu * wv;
            thread_nu += wu * wu;
            thread_nv += wv * wv;
        }

        
        if (u_nbr != prev_u_nbr) {
            v_lo = fv;
        }
        prev_u_nbr = u_nbr;
    }

    
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        thread_dot += __shfl_down_sync(0xffffffff, thread_dot, off);
        thread_nu += __shfl_down_sync(0xffffffff, thread_nu, off);
        thread_nv += __shfl_down_sync(0xffffffff, thread_nv, off);
    }

    if (lane == 0) {
        double denom = sqrt(thread_nu) * sqrt(thread_nv);
        scores[warp_id] = thread_dot / denom;
    }
}

}  

void cosine_similarity_seg(const graph32_t& graph,
                           const double* edge_weights,
                           const int32_t* vertex_pairs_first,
                           const int32_t* vertex_pairs_second,
                           std::size_t num_pairs,
                           double* similarity_scores) {
    if (num_pairs == 0) return;

    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * 32;
    const int num_blocks = (int)((num_pairs + warps_per_block - 1) / warps_per_block);

    cosine_similarity_kernel<<<num_blocks, threads_per_block>>>(
        graph.offsets, graph.indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second,
        (int64_t)num_pairs, similarity_scores);
}

}  
