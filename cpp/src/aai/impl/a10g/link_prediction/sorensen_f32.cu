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

#define WARPS_PER_BLOCK 8
#define SMEM_CACHE_SIZE 384  // 384 elements * 4 bytes * 8 warps = 12KB

__device__ __forceinline__ int lb_smem(const int32_t* arr, int lo, int hi, int32_t target) {
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int lb_global(const int32_t* __restrict__ arr, int lo, int hi, int32_t target) {
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__global__ void sorensen_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ similarity_scores,
    int num_pairs
) {
    extern __shared__ int32_t smem[];

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane_id = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;

    int32_t* s_cache = smem + warp_in_block * SMEM_CACHE_SIZE;

    if (warp_id >= num_pairs) return;

    const int u = first[warp_id];
    const int v = second[warp_id];

    const int u_start = offsets[u];
    const int u_end = offsets[u + 1];
    const int v_start = offsets[v];
    const int v_end = offsets[v + 1];
    const int u_deg = u_end - u_start;
    const int v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        if (lane_id == 0) similarity_scores[warp_id] = 0.0f;
        return;
    }

    if (u == v) {
        if (lane_id == 0) similarity_scores[warp_id] = 1.0f;
        return;
    }

    
    const int32_t* short_idx;
    const float* short_wt;
    int short_deg;
    const int32_t* long_idx;
    const float* long_wt;
    int long_deg;

    if (u_deg <= v_deg) {
        short_idx = indices + u_start; short_wt = edge_weights + u_start; short_deg = u_deg;
        long_idx = indices + v_start; long_wt = edge_weights + v_start; long_deg = v_deg;
    } else {
        short_idx = indices + v_start; short_wt = edge_weights + v_start; short_deg = v_deg;
        long_idx = indices + u_start; long_wt = edge_weights + u_start; long_deg = u_deg;
    }

    float local_intersect = 0.0f;

    if (long_deg <= SMEM_CACHE_SIZE) {
        
        for (int i = lane_id; i < long_deg; i += 32)
            s_cache[i] = long_idx[i];
        __syncwarp();

        int search_lo = 0;
        for (int i = lane_id; i < short_deg; i += 32) {
            int32_t val = short_idx[i];
            int k = 0;
            for (int p = i - 1; p >= 0 && short_idx[p] == val; p--, k++);

            int lb = lb_smem(s_cache, search_lo, long_deg, val);
            search_lo = lb;
            int pos = lb + k;
            if (pos < long_deg && s_cache[pos] == val) {
                local_intersect += fminf(short_wt[i], long_wt[pos]);
            }
        }
    } else {
        
        int search_lo = 0;
        for (int i = lane_id; i < short_deg; i += 32) {
            int32_t val = short_idx[i];
            int k = 0;
            for (int p = i - 1; p >= 0 && short_idx[p] == val; p--, k++);

            int lb = lb_global(long_idx, search_lo, long_deg, val);
            search_lo = lb;
            int pos = lb + k;
            if (pos < long_deg && long_idx[pos] == val) {
                local_intersect += fminf(short_wt[i], long_wt[pos]);
            }
        }
    }

    
    for (int s = 16; s > 0; s >>= 1)
        local_intersect += __shfl_down_sync(0xffffffff, local_intersect, s);
    float total_intersect = __shfl_sync(0xffffffff, local_intersect, 0);

    if (total_intersect == 0.0f) {
        if (lane_id == 0) similarity_scores[warp_id] = 0.0f;
        return;
    }

    
    float w_deg_u = 0.0f;
    for (int i = lane_id; i < u_deg; i += 32)
        w_deg_u += edge_weights[u_start + i];
    for (int s = 16; s > 0; s >>= 1)
        w_deg_u += __shfl_down_sync(0xffffffff, w_deg_u, s);
    w_deg_u = __shfl_sync(0xffffffff, w_deg_u, 0);

    float w_deg_v = 0.0f;
    for (int i = lane_id; i < v_deg; i += 32)
        w_deg_v += edge_weights[v_start + i];
    for (int s = 16; s > 0; s >>= 1)
        w_deg_v += __shfl_down_sync(0xffffffff, w_deg_v, s);
    w_deg_v = __shfl_sync(0xffffffff, w_deg_v, 0);

    float denom = w_deg_u + w_deg_v;
    if (lane_id == 0) {
        similarity_scores[warp_id] = (denom > 0.0f) ? (2.0f * total_intersect / denom) : 0.0f;
    }
}

}  

void sorensen_similarity(const graph32_t& graph,
                         const float* edge_weights,
                         const int32_t* vertex_pairs_first,
                         const int32_t* vertex_pairs_second,
                         std::size_t num_pairs,
                         float* similarity_scores) {
    if (num_pairs <= 0) return;

    const int threads_per_block = WARPS_PER_BLOCK * 32;
    const int num_blocks = (static_cast<int>(num_pairs) + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const int smem_size = WARPS_PER_BLOCK * SMEM_CACHE_SIZE * sizeof(int32_t);

    sorensen_warp_kernel<<<num_blocks, threads_per_block, smem_size>>>(
        graph.offsets,
        graph.indices,
        edge_weights,
        vertex_pairs_first,
        vertex_pairs_second,
        similarity_scores,
        static_cast<int>(num_pairs)
    );
}

}  
