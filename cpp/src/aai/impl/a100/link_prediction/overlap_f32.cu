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






__global__ void overlap_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int num_pairs
) {
    const int WARP_SIZE = 32;
    const unsigned FULL_MASK = 0xffffffff;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int u = pairs_first[warp_id];
    int v = pairs_second[warp_id];

    int start_u = __ldg(&offsets[u]);
    int end_u = __ldg(&offsets[u + 1]);
    int start_v = __ldg(&offsets[v]);
    int end_v = __ldg(&offsets[v + 1]);
    int deg_u = end_u - start_u;
    int deg_v = end_v - start_v;

    if (deg_u == 0 || deg_v == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    int min_u = __ldg(&indices[start_u]);
    int max_u = __ldg(&indices[end_u - 1]);
    int min_v = __ldg(&indices[start_v]);
    int max_v = __ldg(&indices[end_v - 1]);

    if (min_u > max_v || min_v > max_u) {
        
        
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    float local_wdeg_u = 0.0f;
    for (int i = lane; i < deg_u; i += WARP_SIZE)
        local_wdeg_u += __ldg(&weights[start_u + i]);

    float local_wdeg_v = 0.0f;
    for (int i = lane; i < deg_v; i += WARP_SIZE)
        local_wdeg_v += __ldg(&weights[start_v + i]);

    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_wdeg_u += __shfl_down_sync(FULL_MASK, local_wdeg_u, offset);
        local_wdeg_v += __shfl_down_sync(FULL_MASK, local_wdeg_v, offset);
    }
    float wdeg_u = __shfl_sync(FULL_MASK, local_wdeg_u, 0);
    float wdeg_v = __shfl_sync(FULL_MASK, local_wdeg_v, 0);

    float min_wdeg = fminf(wdeg_u, wdeg_v);
    if (min_wdeg == 0.0f) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    const int32_t* small_idx;
    const float* small_w;
    int small_deg;
    const int32_t* large_idx;
    const float* large_w;
    int large_deg;

    if (deg_u <= deg_v) {
        small_idx = indices + start_u;
        small_w = weights + start_u;
        small_deg = deg_u;
        large_idx = indices + start_v;
        large_w = weights + start_v;
        large_deg = deg_v;
    } else {
        small_idx = indices + start_v;
        small_w = weights + start_v;
        small_deg = deg_v;
        large_idx = indices + start_u;
        large_w = weights + start_u;
        large_deg = deg_u;
    }

    
    
    float local_intersection = 0.0f;
    int search_lo = 0;  

    for (int i = lane; i < small_deg; i += WARP_SIZE) {
        int target = __ldg(&small_idx[i]);
        float w_small = __ldg(&small_w[i]);

        
        int dup_offset = 0;
        for (int k = i - 1; k >= 0 && __ldg(&small_idx[k]) == target; k--) {
            dup_offset++;
        }

        
        int lo = search_lo, hi = large_deg;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (__ldg(&large_idx[mid]) < target) lo = mid + 1;
            else hi = mid;
        }

        
        search_lo = lo;

        
        int match_pos = lo + dup_offset;
        if (match_pos < large_deg && __ldg(&large_idx[match_pos]) == target) {
            local_intersection += fminf(w_small, __ldg(&large_w[match_pos]));
        }
    }

    
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_intersection += __shfl_down_sync(FULL_MASK, local_intersection, offset);
    }

    if (lane == 0) {
        scores[warp_id] = local_intersection / min_wdeg;
    }
}

}  

void overlap_similarity(const graph32_t& graph,
                        const float* edge_weights,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    if (num_pairs == 0) return;

    const int WARP_SIZE = 32;
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / WARP_SIZE;
    int grid = (static_cast<int>(num_pairs) + warps_per_block - 1) / warps_per_block;
    overlap_warp_kernel<<<grid, threads_per_block>>>(
        graph.offsets, graph.indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, static_cast<int>(num_pairs)
    );
}

}  
