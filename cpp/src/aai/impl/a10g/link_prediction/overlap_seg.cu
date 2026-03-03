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

#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 8
#endif
#ifndef SMEM_BUF_SIZE
#define SMEM_BUF_SIZE 256
#endif
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)

struct Cache : Cacheable {};

__device__ __forceinline__ int warp_reduce_sum(int val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

template<bool IS_MULTIGRAPH>
__global__ __launch_bounds__(THREADS_PER_BLOCK)
void overlap_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    __shared__ int32_t smem_buf[WARPS_PER_BLOCK * SMEM_BUF_SIZE];

    const int lane = threadIdx.x & 31;
    const int warp_local = threadIdx.x >> 5;
    const int64_t warp_id = (int64_t)blockIdx.x * WARPS_PER_BLOCK + warp_local;

    if (warp_id >= num_pairs) return;

    int32_t* my_buf = smem_buf + warp_local * SMEM_BUF_SIZE;

    int u = __ldg(&pairs_first[warp_id]);
    int v = __ldg(&pairs_second[warp_id]);

    int u_start = __ldg(&offsets[u]);
    int u_end = __ldg(&offsets[u + 1]);
    int v_start = __ldg(&offsets[v]);
    int v_end = __ldg(&offsets[v + 1]);

    int deg_u = u_end - u_start;
    int deg_v = v_end - v_start;
    int min_deg = (deg_u < deg_v) ? deg_u : deg_v;

    if (min_deg == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    const int32_t* a, *b;
    int m, n;
    if (deg_u <= deg_v) {
        a = indices + u_start; m = deg_u;
        b = indices + v_start; n = deg_v;
    } else {
        a = indices + v_start; m = deg_v;
        b = indices + u_start; n = deg_u;
    }

    int count = 0;

    if (!IS_MULTIGRAPH && n <= 32) {
        
        int a_val = (lane < m) ? __ldg(&a[lane]) : 0x7FFFFFFF;
        int b_val = (lane < n) ? __ldg(&b[lane]) : 0x7FFFFFFF;

        for (int i = 0; i < m; i++) {
            int target = __shfl_sync(0xFFFFFFFF, a_val, i);
            unsigned match = __ballot_sync(0xFFFFFFFF, (lane < n) & (b_val == target));
            count += (match != 0);
        }
    } else if (n <= SMEM_BUF_SIZE) {
        
        for (int i = lane; i < n; i += 32) {
            my_buf[i] = __ldg(&b[i]);
        }
        __syncwarp();

        int lb = 0;
        int niters = (m + 31) >> 5;

        for (int iter = 0; iter < niters; iter++) {
            int i = iter * 32 + lane;
            bool active = (i < m);
            int lo = lb;

            if (active) {
                int target = __ldg(&a[i]);

                int rank = 0;
                if constexpr (IS_MULTIGRAPH) {
                    int lo_a = 0, hi_a = i;
                    while (lo_a < hi_a) {
                        int mid = (lo_a + hi_a) >> 1;
                        if (__ldg(&a[mid]) < target) lo_a = mid + 1;
                        else hi_a = mid;
                    }
                    rank = i - lo_a;
                }

                int hi = n;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (my_buf[mid] < target) lo = mid + 1;
                    else hi = mid;
                }

                if (lo + rank < n && my_buf[lo + rank] == target) count++;
            }

            unsigned active_mask = __ballot_sync(0xFFFFFFFF, active);
            if (active_mask) {
                int highest = 31 - __clz(active_mask);
                lb = __shfl_sync(0xFFFFFFFF, lo, highest);
            }
        }
        count = warp_reduce_sum(count);
    } else {
        
        int lb = 0;
        int niters = (m + 31) >> 5;

        for (int iter = 0; iter < niters; iter++) {
            int i = iter * 32 + lane;
            bool active = (i < m);
            int lo = lb;

            if (active) {
                int target = __ldg(&a[i]);

                int rank = 0;
                if constexpr (IS_MULTIGRAPH) {
                    int lo_a = 0, hi_a = i;
                    while (lo_a < hi_a) {
                        int mid = (lo_a + hi_a) >> 1;
                        if (__ldg(&a[mid]) < target) lo_a = mid + 1;
                        else hi_a = mid;
                    }
                    rank = i - lo_a;
                }

                int hi = n;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (__ldg(&b[mid]) < target) lo = mid + 1;
                    else hi = mid;
                }

                if (lo + rank < n && __ldg(&b[lo + rank]) == target) count++;
            }

            unsigned active_mask = __ballot_sync(0xFFFFFFFF, active);
            if (active_mask) {
                int highest = 31 - __clz(active_mask);
                lb = __shfl_sync(0xFFFFFFFF, lo, highest);
            }
        }
        count = warp_reduce_sum(count);
    }

    if (lane == 0) {
        scores[warp_id] = (float)count / (float)min_deg;
    }
}

}  

void overlap_similarity_seg(const graph32_t& graph,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    if (num_pairs == 0) return;

    int64_t np = static_cast<int64_t>(num_pairs);
    int num_blocks = static_cast<int>((np + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    if (graph.is_multigraph) {
        overlap_kernel<true><<<num_blocks, THREADS_PER_BLOCK>>>(
            graph.offsets, graph.indices,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, np);
    } else {
        overlap_kernel<false><<<num_blocks, THREADS_PER_BLOCK>>>(
            graph.offsets, graph.indices,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, np);
    }
}

}  
