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

struct Cache : Cacheable {};


__device__ __forceinline__ int lower_bound(const int32_t* __restrict__ arr, int len, int32_t val) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(&arr[mid]) < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__device__ __forceinline__ int upper_bound(const int32_t* __restrict__ arr, int len, int32_t val) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(&arr[mid]) <= val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}




__global__ void cosine_sim_warp_per_pair(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int32_t u = __ldg(&pairs_first[warp_id]);
    int32_t v = __ldg(&pairs_second[warp_id]);

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);

    int32_t m = u_end - u_start;
    int32_t n = v_end - v_start;

    const int32_t* A = indices + u_start;
    const int32_t* B = indices + v_start;
    const float* wA = weights + u_start;
    const float* wB = weights + v_start;

    
    if (m > n) {
        const int32_t* tmp_i = A; A = B; B = tmp_i;
        const float* tmp_w = wA; wA = wB; wB = tmp_w;
        int32_t tmp = m; m = n; n = tmp;
    }

    float dot = 0.0f, norm_a_sq = 0.0f, norm_b_sq = 0.0f;

    
    for (int k = lane; k < m; k += 32) {
        int32_t val = __ldg(&A[k]);

        
        int lb_b = lower_bound(B, n, val);

        
        if (lb_b < n && __ldg(&B[lb_b]) == val) {
            
            int rank = 0;
            if (k > 0 && __ldg(&A[k - 1]) == val) {
                
                int lb_a = lower_bound(A, k, val);
                rank = k - lb_a;
            }

            
            int ub_b = upper_bound(B + lb_b, n - lb_b, val) + lb_b;
            int count_b = ub_b - lb_b;

            if (rank < count_b) {
                float wa = __ldg(&wA[k]);
                float wb = __ldg(&wB[lb_b + rank]);
                dot += wa * wb;
                norm_a_sq += wa * wa;
                norm_b_sq += wb * wb;
            }
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
        norm_a_sq += __shfl_down_sync(0xffffffff, norm_a_sq, offset);
        norm_b_sq += __shfl_down_sync(0xffffffff, norm_b_sq, offset);
    }

    if (lane == 0) {
        float denom = sqrtf(norm_a_sq) * sqrtf(norm_b_sq);
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
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    if (num_pairs == 0) return;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    int threads_per_block = 256;  
    int warps_per_block = threads_per_block / 32;
    int64_t num_blocks = (static_cast<int64_t>(num_pairs) + warps_per_block - 1) / warps_per_block;

    cosine_sim_warp_per_pair<<<static_cast<int>(num_blocks), threads_per_block>>>(
        offsets, indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, static_cast<int64_t>(num_pairs)
    );
}

}  
