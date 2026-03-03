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

#define MAX_CACHED_DEG 384
#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

struct Cache : Cacheable {};

__global__ __launch_bounds__(THREADS_PER_BLOCK)
void jaccard_fused_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    int64_t num_pairs,
    float* __restrict__ scores)
{
    __shared__ int32_t s_cache[WARPS_PER_BLOCK * MAX_CACHED_DEG];

    const int warp_local = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int64_t warp_global = (int64_t)blockIdx.x * WARPS_PER_BLOCK + warp_local;

    if (warp_global >= num_pairs) return;

    int32_t* my_cache = &s_cache[warp_local * MAX_CACHED_DEG];

    const int u = __ldg(&first[warp_global]);
    const int v = __ldg(&second[warp_global]);

    const int u_start = __ldg(&offsets[u]);
    const int u_end = __ldg(&offsets[u + 1]);
    const int v_start = __ldg(&offsets[v]);
    const int v_end = __ldg(&offsets[v + 1]);
    const int u_deg = u_end - u_start;
    const int v_deg = v_end - v_start;

    
    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) scores[warp_global] = 0.0f;
        return;
    }
    if (u == v) {
        if (lane == 0) scores[warp_global] = 1.0f;
        return;
    }

    
    int small_start, small_end, large_start, large_end, large_deg;
    if (u_deg <= v_deg) {
        small_start = u_start; small_end = u_end;
        large_start = v_start; large_end = v_end;
        large_deg = v_deg;
    } else {
        small_start = v_start; small_end = v_end;
        large_start = u_start; large_end = u_end;
        large_deg = u_deg;
    }

    float intersection_sum = 0.0f;
    float sum_small = 0.0f;
    float sum_large = 0.0f;

    if (large_deg <= MAX_CACHED_DEG) {
        
        for (int j = lane; j < large_deg; j += 32) {
            my_cache[j] = __ldg(&indices[large_start + j]);
            sum_large += __ldg(&weights[large_start + j]);
        }
        __syncwarp();

        
        for (int i = small_start + lane; i < small_end; i += 32) {
            int neighbor = __ldg(&indices[i]);
            float w_small = __ldg(&weights[i]);
            sum_small += w_small;

            
            int rank = 0;
            for (int k = i - 1; k >= small_start; k--) {
                if (__ldg(&indices[k]) != neighbor) break;
                rank++;
            }

            
            int lo = 0, hi = large_deg;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (my_cache[mid] < neighbor) lo = mid + 1;
                else hi = mid;
            }

            int match_pos = lo + rank;
            if (match_pos < large_deg && my_cache[match_pos] == neighbor) {
                intersection_sum += fminf(w_small, __ldg(&weights[large_start + match_pos]));
            }
        }
    } else {
        
        for (int j = large_start + lane; j < large_end; j += 32) {
            sum_large += __ldg(&weights[j]);
        }

        
        int thread_search_lo = large_start;

        for (int i = small_start + lane; i < small_end; i += 32) {
            int neighbor = __ldg(&indices[i]);
            float w_small = __ldg(&weights[i]);
            sum_small += w_small;

            int rank = 0;
            for (int k = i - 1; k >= small_start; k--) {
                if (__ldg(&indices[k]) != neighbor) break;
                rank++;
            }

            int lo = thread_search_lo, hi = large_end;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (__ldg(&indices[mid]) < neighbor) lo = mid + 1;
                else hi = mid;
            }
            thread_search_lo = lo;

            int match_pos = lo + rank;
            if (match_pos < large_end && __ldg(&indices[match_pos]) == neighbor) {
                intersection_sum += fminf(w_small, __ldg(&weights[match_pos]));
            }
        }
    }

    
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        intersection_sum += __shfl_xor_sync(0xffffffff, intersection_sum, mask);
        sum_small += __shfl_xor_sync(0xffffffff, sum_small, mask);
        sum_large += __shfl_xor_sync(0xffffffff, sum_large, mask);
    }

    if (lane == 0) {
        float union_sum = sum_small + sum_large - intersection_sum;
        scores[warp_global] = (union_sum > 0.0f) ? (intersection_sum / union_sum) : 0.0f;
    }
}

}  

void jaccard_similarity_seg(const graph32_t& graph,
                            const float* edge_weights,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    int64_t np = static_cast<int64_t>(num_pairs);
    if (np == 0) return;

    int grid = static_cast<int>((np + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    jaccard_fused_kernel<<<grid, THREADS_PER_BLOCK>>>(
        graph.offsets, graph.indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second, np, similarity_scores);
}

}  
