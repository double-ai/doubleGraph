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
#include <cfloat>
#include <cstdint>

namespace aai {

namespace {

struct Cache : Cacheable {};



constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;
constexpr int MAX_CACHED = 128;

__global__ __launch_bounds__(THREADS_PER_BLOCK, 12)
void overlap_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ scores,
    const int64_t num_pairs
) {
    __shared__ int s_cache[WARPS_PER_BLOCK * MAX_CACHED];

    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;

    if (warp_id >= num_pairs) return;

    int* __restrict__ my_cache = s_cache + warp_in_block * MAX_CACHED;

    const int u = first[warp_id];
    const int v = second[warp_id];

    const int u_start = offsets[u];
    const int u_end = offsets[u + 1];
    const int v_start = offsets[v];
    const int v_end = offsets[v + 1];
    const int u_deg = u_end - u_start;
    const int v_deg = v_end - v_start;

    
    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) scores[warp_id] = 0.0;
        return;
    }

    
    int small_start, large_start, small_size, large_size;
    if (u_deg <= v_deg) {
        small_start = u_start; small_size = u_deg;
        large_start = v_start; large_size = v_deg;
    } else {
        small_start = v_start; small_size = v_deg;
        large_start = u_start; large_size = u_deg;
    }

    const bool use_cache = (large_size <= MAX_CACHED);

    
    if (use_cache) {
        for (int i = lane; i < large_size; i += 32) {
            my_cache[i] = indices[large_start + i];
        }
        __syncwarp();
    }

    
    double local_sum = 0.0;
    int lo_hint = 0;

    if (use_cache) {
        
        for (int i = lane; i < small_size; i += 32) {
            int target = indices[small_start + i];

            
            int rank = 0;
            if (i > 0 && indices[small_start + i - 1] == target) {
                int lo_r = 0, hi_r = i;
                while (lo_r < hi_r) {
                    int mid = (lo_r + hi_r) >> 1;
                    if (indices[small_start + mid] < target) lo_r = mid + 1;
                    else hi_r = mid;
                }
                rank = i - lo_r;
            }

            
            int lo = lo_hint, hi = large_size;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (my_cache[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            lo_hint = lo;

            int pos = lo + rank;
            if (pos < large_size && my_cache[pos] == target) {
                local_sum += fmin(edge_weights[small_start + i], edge_weights[large_start + pos]);
            }
        }
    } else {
        
        for (int i = lane; i < small_size; i += 32) {
            int target = indices[small_start + i];

            int rank = 0;
            if (i > 0 && indices[small_start + i - 1] == target) {
                int lo_r = 0, hi_r = i;
                while (lo_r < hi_r) {
                    int mid = (lo_r + hi_r) >> 1;
                    if (indices[small_start + mid] < target) lo_r = mid + 1;
                    else hi_r = mid;
                }
                rank = i - lo_r;
            }

            int lo = lo_hint, hi = large_size;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (indices[large_start + mid] < target) lo = mid + 1;
                else hi = mid;
            }
            lo_hint = lo;

            int pos = lo + rank;
            if (pos < large_size && indices[large_start + pos] == target) {
                local_sum += fmin(edge_weights[small_start + i], edge_weights[large_start + pos]);
            }
        }
    }

    
    if (!__any_sync(0xffffffff, local_sum > 0.0)) {
        if (lane == 0) scores[warp_id] = 0.0;
        return;
    }

    
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    double intersection = __shfl_sync(0xffffffff, local_sum, 0);

    
    double local_wd_u = 0.0;
    for (int i = lane; i < u_deg; i += 32) {
        local_wd_u += edge_weights[u_start + i];
    }
    double local_wd_v = 0.0;
    for (int i = lane; i < v_deg; i += 32) {
        local_wd_v += edge_weights[v_start + i];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_wd_u += __shfl_down_sync(0xffffffff, local_wd_u, offset);
        local_wd_v += __shfl_down_sync(0xffffffff, local_wd_v, offset);
    }
    double wd_u = __shfl_sync(0xffffffff, local_wd_u, 0);
    double wd_v = __shfl_sync(0xffffffff, local_wd_v, 0);

    double min_w = fmin(wd_u, wd_v);

    if (lane == 0) {
        scores[warp_id] = (min_w > DBL_MIN) ? intersection / min_w : 0.0;
    }
}

}  

void overlap_similarity_seg(const graph32_t& graph,
                            const double* edge_weights,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            double* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    if (num_pairs == 0) return;

    int num_blocks = (int)((num_pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    overlap_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(
        graph.offsets,
        graph.indices,
        edge_weights,
        vertex_pairs_first,
        vertex_pairs_second,
        similarity_scores,
        static_cast<int64_t>(num_pairs));
}

}  
