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


__device__ __forceinline__ int lower_bound(const int32_t* arr, int n, int32_t val) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__global__ void cosine_similarity_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int32_t num_pairs)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    const int32_t u = pairs_first[warp_id];
    const int32_t v = pairs_second[warp_id];

    const int32_t u_start = offsets[u];
    const int32_t u_end = offsets[u + 1];
    const int32_t v_start = offsets[v];
    const int32_t v_end = offsets[v + 1];

    const int32_t du = u_end - u_start;
    const int32_t dv = v_end - v_start;

    
    if (du == 0 || dv == 0) {
        if (lane == 0) scores[warp_id] = CUDART_NAN_F;
        return;
    }
    
    if (u == v) {
        if (lane == 0) scores[warp_id] = 1.0f;
        return;
    }

    
    const int32_t* A_idx;
    const float* A_wt;
    int32_t da;
    const int32_t* B_idx;
    const float* B_wt;
    int32_t db;

    if (du <= dv) {
        A_idx = indices + u_start;
        A_wt = edge_weights + u_start;
        da = du;
        B_idx = indices + v_start;
        B_wt = edge_weights + v_start;
        db = dv;
    } else {
        A_idx = indices + v_start;
        A_wt = edge_weights + v_start;
        da = dv;
        B_idx = indices + u_start;
        B_wt = edge_weights + u_start;
        db = du;
    }

    float local_dot = 0.0f;
    float local_norm_a = 0.0f;
    float local_norm_b = 0.0f;

    
    for (int i = lane; i < da; i += 32) {
        int32_t elem = A_idx[i];
        float wa = A_wt[i];

        
        int rank = 0;
        for (int k = i - 1; k >= 0 && A_idx[k] == elem; k--) {
            rank++;
        }

        
        int lb = lower_bound(B_idx, db, elem);

        
        int match_pos = lb + rank;
        if (match_pos < db && B_idx[match_pos] == elem) {
            float wb = B_wt[match_pos];
            local_dot += wa * wb;
            local_norm_a += wa * wa;
            local_norm_b += wb * wb;
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_dot += __shfl_down_sync(0xffffffff, local_dot, offset);
        local_norm_a += __shfl_down_sync(0xffffffff, local_norm_a, offset);
        local_norm_b += __shfl_down_sync(0xffffffff, local_norm_b, offset);
    }

    if (lane == 0) {
        float denom = sqrtf(local_norm_a) * sqrtf(local_norm_b);
        
        scores[warp_id] = (denom > 0.0f) ? (local_dot / denom) : CUDART_NAN_F;
    }
}

}  

void cosine_similarity(const graph32_t& graph,
                       const float* edge_weights,
                       const int32_t* vertex_pairs_first,
                       const int32_t* vertex_pairs_second,
                       std::size_t num_pairs,
                       float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    if (num_pairs == 0) return;

    const int32_t np = static_cast<int32_t>(num_pairs);
    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    const int num_blocks = (np + warps_per_block - 1) / warps_per_block;

    cosine_similarity_warp_kernel<<<num_blocks, threads_per_block>>>(
        graph.offsets, graph.indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, np);
}

}  
