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

__device__ __forceinline__ int lower_bound_dev(
    const int32_t* __restrict__ arr, int lo, int hi, int32_t target
) {
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void sorensen_fused_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ first_vertices,
    const int32_t* __restrict__ second_vertices,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    const int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    const int32_t u = first_vertices[warp_id];
    const int32_t v = second_vertices[warp_id];

    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int v_start = offsets[v];
    int v_end = offsets[v + 1];
    int u_deg = u_end - u_start;
    int v_deg = v_end - v_start;

    if (u_deg == 0 && v_deg == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    if (u == v) {
        if (lane == 0) scores[warp_id] = 1.0f;
        return;
    }

    float local_deg_u = 0.0f;
    for (int i = u_start + lane; i < u_end; i += 32) {
        local_deg_u += edge_weights[i];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_deg_u += __shfl_down_sync(0xFFFFFFFF, local_deg_u, offset);
    }
    float deg_u = __shfl_sync(0xFFFFFFFF, local_deg_u, 0);

    float local_deg_v = 0.0f;
    for (int i = v_start + lane; i < v_end; i += 32) {
        local_deg_v += edge_weights[i];
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_deg_v += __shfl_down_sync(0xFFFFFFFF, local_deg_v, offset);
    }
    float deg_v = __shfl_sync(0xFFFFFFFF, local_deg_v, 0);

    float denom = deg_u + deg_v;
    if (denom == 0.0f) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    int small_start, small_end, small_deg;
    int large_start, large_end, large_deg;
    if (u_deg <= v_deg) {
        small_start = u_start; small_end = u_end; small_deg = u_deg;
        large_start = v_start; large_end = v_end; large_deg = v_deg;
    } else {
        small_start = v_start; small_end = v_end; small_deg = v_deg;
        large_start = u_start; large_end = u_end; large_deg = u_deg;
    }

    int32_t large_min = indices[large_start];
    int32_t large_max = indices[large_end - 1];
    int32_t small_min = indices[small_start];
    int32_t small_max = indices[small_end - 1];

    if (small_max < large_min || large_max < small_min) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    float local_sum = 0.0f;
    int search_lo = large_start;

    for (int i = lane; i < small_deg; i += 32) {
        int32_t target = indices[small_start + i];

        if (target > large_max) break;
        if (target < large_min) continue;

        int rank = 0;
        if (i > 0 && indices[small_start + i - 1] == target) {
            int first_small = lower_bound_dev(indices, small_start, small_start + i, target);
            rank = (small_start + i) - first_small;
        }

        int first_large = lower_bound_dev(indices, search_lo, large_end, target);

        search_lo = first_large;

        int match_pos = first_large + rank;
        if (match_pos < large_end && indices[match_pos] == target) {
            local_sum += fminf(edge_weights[small_start + i], edge_weights[match_pos]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    if (lane == 0) {
        scores[warp_id] = 2.0f * local_sum / denom;
    }
}

}  

void sorensen_similarity(const graph32_t& graph,
                         const float* edge_weights,
                         const int32_t* vertex_pairs_first,
                         const int32_t* vertex_pairs_second,
                         std::size_t num_pairs,
                         float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * 32;
    int64_t np = static_cast<int64_t>(num_pairs);
    int64_t grid64 = (np + warps_per_block - 1) / warps_per_block;
    int grid = (int)min(grid64, (int64_t)INT_MAX);

    sorensen_fused_kernel<<<grid, threads_per_block>>>(
        offsets, indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, np);
}

}  
