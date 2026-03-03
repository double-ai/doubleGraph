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

#define SMEM_CACHE 128
#define WARPS_PER_BLOCK 4
#define BLOCK_SIZE (WARPS_PER_BLOCK * 32)

__global__ __launch_bounds__(BLOCK_SIZE, 8)
void overlap_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    const int num_pairs)
{
    __shared__ int32_t smem_idx[WARPS_PER_BLOCK * SMEM_CACHE];

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane_id = threadIdx.x & 31;
    const int warp_in_block = (threadIdx.x >> 5);

    if (warp_id >= num_pairs) return;

    int32_t* cache = smem_idx + warp_in_block * SMEM_CACHE;

    const int u = __ldg(pairs_first + warp_id);
    const int v = __ldg(pairs_second + warp_id);

    const int u_start = __ldg(offsets + u);
    const int u_end = __ldg(offsets + u + 1);
    const int v_start = __ldg(offsets + v);
    const int v_end = __ldg(offsets + v + 1);

    const int u_deg = u_end - u_start;
    const int v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        if (lane_id == 0) scores[warp_id] = 0.0f;
        return;
    }

    int small_start, small_len, large_start, large_len;
    if (u_deg <= v_deg) {
        small_start = u_start; small_len = u_deg;
        large_start = v_start; large_len = v_deg;
    } else {
        small_start = v_start; small_len = v_deg;
        large_start = u_start; large_len = u_deg;
    }

    float my_weight = 0.0f;

    if (large_len <= SMEM_CACHE) {
        
        for (int i = lane_id; i < large_len; i += 32)
            cache[i] = __ldg(indices + large_start + i);
        __syncwarp();

        
        for (int i = lane_id; i < small_len; i += 32) {
            int target = __ldg(indices + small_start + i);

            
            int dup = 0;
            for (int j = i - 1; j >= 0 && __ldg(indices + small_start + j) == target; --j)
                ++dup;

            int lo = 0, hi = large_len;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (cache[mid] < target) lo = mid + 1;
                else hi = mid;
            }

            int pos = lo + dup;
            if (pos < large_len && cache[pos] == target) {
                float ws = __ldg(edge_weights + small_start + i);
                float wl = __ldg(edge_weights + large_start + pos);
                my_weight += fminf(ws, wl);
            }
        }
    } else {
        
        for (int i = lane_id; i < small_len; i += 32) {
            int target = __ldg(indices + small_start + i);

            int dup = 0;
            for (int j = i - 1; j >= 0 && __ldg(indices + small_start + j) == target; --j)
                ++dup;

            int lo = 0, hi = large_len;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (__ldg(indices + large_start + mid) < target) lo = mid + 1;
                else hi = mid;
            }

            int pos = lo + dup;
            if (pos < large_len && __ldg(indices + large_start + pos) == target) {
                float ws = __ldg(edge_weights + small_start + i);
                float wl = __ldg(edge_weights + large_start + pos);
                my_weight += fminf(ws, wl);
            }
        }
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        my_weight += __shfl_down_sync(0xffffffff, my_weight, s);

    float total = __shfl_sync(0xffffffff, my_weight, 0);

    if (total == 0.0f) {
        if (lane_id == 0) scores[warp_id] = 0.0f;
        return;
    }

    float du = 0.0f;
    for (int i = u_start + lane_id; i < u_end; i += 32)
        du += __ldg(edge_weights + i);
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        du += __shfl_down_sync(0xffffffff, du, s);

    float dv = 0.0f;
    for (int i = v_start + lane_id; i < v_end; i += 32)
        dv += __ldg(edge_weights + i);
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        dv += __shfl_down_sync(0xffffffff, dv, s);

    if (lane_id == 0) {
        float md = fminf(du, dv);
        scores[warp_id] = (md > 1.17549435e-38f) ? (total / md) : 0.0f;
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
    const int grid = (static_cast<int>(num_pairs) + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    overlap_warp_kernel<<<grid, BLOCK_SIZE>>>(
        graph.offsets, graph.indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second, similarity_scores,
        static_cast<int>(num_pairs));
}

}  
