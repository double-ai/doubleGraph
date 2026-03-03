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

__device__ __forceinline__ int32_t lower_bound(
    const int32_t* __restrict__ arr, int32_t lo, int32_t hi, int32_t target
) {
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ float warp_sum(
    const float* __restrict__ data, int32_t start, int32_t end, int lane
) {
    float sum = 0.0f;
    for (int32_t i = start + lane; i < end; i += 32) {
        sum += data[i];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    return __shfl_sync(0xFFFFFFFF, sum, 0);
}

#define BLOCK_SIZE 128

__global__ void __launch_bounds__(BLOCK_SIZE, 12)
sorensen_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    const int lane = threadIdx.x & 31;
    const int64_t warp_id = ((int64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x) >> 5;

    if (warp_id >= num_pairs) return;

    const int32_t u = pairs_first[warp_id];
    const int32_t v = pairs_second[warp_id];

    const int32_t u_start = offsets[u];
    const int32_t u_end = offsets[u + 1];
    const int32_t v_start = offsets[v];
    const int32_t v_end = offsets[v + 1];

    const int32_t u_deg = u_end - u_start;
    const int32_t v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    const int32_t* __restrict__ short_idx;
    const float* __restrict__ short_wgt;
    const int32_t* __restrict__ long_idx;
    const float* __restrict__ long_wgt;
    int32_t short_len, long_len;

    if (u_deg <= v_deg) {
        short_idx = indices + u_start;
        short_wgt = edge_weights + u_start;
        long_idx = indices + v_start;
        long_wgt = edge_weights + v_start;
        short_len = u_deg;
        long_len = v_deg;
    } else {
        short_idx = indices + v_start;
        short_wgt = edge_weights + v_start;
        long_idx = indices + u_start;
        long_wgt = edge_weights + u_start;
        short_len = v_deg;
        long_len = u_deg;
    }

    
    if (short_idx[short_len - 1] < long_idx[0] || long_idx[long_len - 1] < short_idx[0]) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    float partial_sum = 0.0f;

    for (int32_t i = lane; i < short_len; i += 32) {
        int32_t target = short_idx[i];

        
        int32_t count_before = 0;
        if (i > 0 && short_idx[i - 1] == target) {
            int32_t first = lower_bound(short_idx, 0, i, target);
            count_before = i - first;
        }

        int32_t pos = lower_bound(long_idx, 0, long_len, target);
        pos += count_before;

        if (pos < long_len && long_idx[pos] == target) {
            partial_sum += fminf(short_wgt[i], long_wgt[pos]);
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial_sum += __shfl_down_sync(0xFFFFFFFF, partial_sum, offset);
    }

    
    float total_intersection = __shfl_sync(0xFFFFFFFF, partial_sum, 0);

    if (total_intersection == 0.0f) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    float w_deg_u = warp_sum(edge_weights, u_start, u_end, lane);
    float w_deg_v = warp_sum(edge_weights, v_start, v_end, lane);
    float denom = w_deg_u + w_deg_v;

    if (lane == 0) {
        scores[warp_id] = (denom > 0.0f) ? (2.0f * total_intersection / denom) : 0.0f;
    }
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

    int64_t np = static_cast<int64_t>(num_pairs);
    int warps_per_block = BLOCK_SIZE / 32;
    int grid = static_cast<int>((np + warps_per_block - 1) / warps_per_block);
    sorensen_warp_kernel<<<grid, BLOCK_SIZE>>>(
        graph.offsets, graph.indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, np);
}

}  
