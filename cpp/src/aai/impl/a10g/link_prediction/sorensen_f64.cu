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

__device__ __forceinline__ int lower_bound_dev(
    const int32_t* __restrict__ arr, int size, int target
) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}




__launch_bounds__(128, 12)
__global__ void sorensen_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ similarity_scores,
    int num_pairs
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    const int u = first[warp_id];
    const int v = second[warp_id];

    if (u == v) {
        if (lane == 0) {
            similarity_scores[warp_id] = (offsets[u + 1] > offsets[u]) ? 1.0 : 0.0;
        }
        return;
    }

    const int u_start = offsets[u];
    const int u_end = offsets[u + 1];
    const int v_start = offsets[v];
    const int v_end = offsets[v + 1];
    const int u_deg = u_end - u_start;
    const int v_deg = v_end - v_start;

    
    if (__all_sync(0xffffffff, u_deg == 0 || v_deg == 0)) {
        if (lane == 0) similarity_scores[warp_id] = 0.0;
        return;
    }

    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) similarity_scores[warp_id] = 0.0;
        return;
    }

    
    const int u_first_nbr = indices[u_start];
    const int u_last_nbr = indices[u_end - 1];
    const int v_first_nbr = indices[v_start];
    const int v_last_nbr = indices[v_end - 1];

    if (u_last_nbr < v_first_nbr || v_last_nbr < u_first_nbr) {
        if (lane == 0) similarity_scores[warp_id] = 0.0;
        return;
    }

    
    const int32_t* small_idx;
    const double* small_wt;
    int small_size;
    const int32_t* large_idx;
    const double* large_wt;
    int large_size;

    if (u_deg <= v_deg) {
        small_idx = indices + u_start; small_wt = edge_weights + u_start; small_size = u_deg;
        large_idx = indices + v_start; large_wt = edge_weights + v_start; large_size = v_deg;
    } else {
        small_idx = indices + v_start; small_wt = edge_weights + v_start; small_size = v_deg;
        large_idx = indices + u_start; large_wt = edge_weights + u_start; large_size = u_deg;
    }

    
    double local_sum = 0.0;
    for (int idx = lane; idx < small_size; idx += 32) {
        int my_val = small_idx[idx];

        
        int dup_rank = 0;
        if (idx > 0 && small_idx[idx - 1] == my_val) {
            dup_rank = idx - lower_bound_dev(small_idx, idx, my_val);
        }

        int pos = lower_bound_dev(large_idx, large_size, my_val) + dup_rank;

        if (pos < large_size && large_idx[pos] == my_val) {
            local_sum += fmin(small_wt[idx], large_wt[pos]);
        }
    }

    
    for (int s = 16; s > 0; s >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, s);

    
    double intersection_sum = __shfl_sync(0xffffffff, local_sum, 0);

    if (intersection_sum == 0.0) {
        if (lane == 0) similarity_scores[warp_id] = 0.0;
        return;
    }

    
    double w_sum_u = 0.0;
    for (int i = u_start + lane; i < u_end; i += 32) w_sum_u += edge_weights[i];
    for (int s = 16; s > 0; s >>= 1) w_sum_u += __shfl_down_sync(0xffffffff, w_sum_u, s);
    w_sum_u = __shfl_sync(0xffffffff, w_sum_u, 0);

    double w_sum_v = 0.0;
    for (int i = v_start + lane; i < v_end; i += 32) w_sum_v += edge_weights[i];
    for (int s = 16; s > 0; s >>= 1) w_sum_v += __shfl_down_sync(0xffffffff, w_sum_v, s);
    w_sum_v = __shfl_sync(0xffffffff, w_sum_v, 0);

    if (lane == 0) {
        double denom = w_sum_u + w_sum_v;
        similarity_scores[warp_id] = (denom > 0.0) ? 2.0 * intersection_sum / denom : 0.0;
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

    int threads = 128;
    int warps_per_block = threads / 32;
    int grid = (static_cast<int>(num_pairs) + warps_per_block - 1) / warps_per_block;
    sorensen_warp_kernel<<<grid, threads, 0, nullptr>>>(
        graph.offsets, graph.indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, static_cast<int>(num_pairs));
}

}  
