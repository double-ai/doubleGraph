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

struct Cache : Cacheable {};

__device__ __forceinline__ int lb(const int32_t* arr, int lo, int hi, int32_t val) {
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}



#define PAIRS_PER_WARP 4

__global__ void __launch_bounds__(128)
cosine_sim_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs)
{
    const int warp_base = ((blockIdx.x * 128 + threadIdx.x) >> 5) * PAIRS_PER_WARP;
    const int lane = threadIdx.x & 31;

    #pragma unroll
    for (int p = 0; p < PAIRS_PER_WARP; p++) {
        const int pid = warp_base + p;
        if (pid >= num_pairs) return;

        const int32_t u = first[pid];
        const int32_t v = second[pid];

        const int32_t u_start = offsets[u];
        const int32_t u_end = offsets[u + 1];
        const int32_t v_start = offsets[v];
        const int32_t v_end = offsets[v + 1];

        const int u_deg = u_end - u_start;
        const int v_deg = v_end - v_start;

        if (u_deg == 0 || v_deg == 0) {
            if (lane == 0) scores[pid] = 0.0f;
            continue;
        }

        const int32_t* A = (u_deg <= v_deg) ? indices + u_start : indices + v_start;
        const int32_t* B = (u_deg <= v_deg) ? indices + v_start : indices + u_start;
        const int m = (u_deg <= v_deg) ? u_deg : v_deg;
        const int n = (u_deg <= v_deg) ? v_deg : u_deg;

        int found = 0;
        int b_lo = 0;
        const int num_chunks = (m + 31) >> 5;

        for (int c = 0; c < num_chunks; c++) {
            const int i = (c << 5) | lane;
            if (i < m && !found) {
                const int32_t val = A[i];
                const int pos = lb(B, b_lo, n, val);
                if (pos < n && B[pos] == val) found = 1;
                b_lo = pos;
            }
            if (__any_sync(0xffffffff, found)) break;
        }

        const int any_found = __any_sync(0xffffffff, found);
        if (lane == 0) {
            scores[pid] = any_found ? 1.0f : 0.0f;
        }
    }
}

}  

void cosine_similarity(const graph32_t& graph,
                       const int32_t* vertex_pairs_first,
                       const int32_t* vertex_pairs_second,
                       std::size_t num_pairs,
                       float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    if (num_pairs == 0) return;

    const int64_t np = static_cast<int64_t>(num_pairs);
    const int warps_needed = static_cast<int>((np + PAIRS_PER_WARP - 1) / PAIRS_PER_WARP);
    const int num_blocks = (warps_needed + 3) / 4;  
    cosine_sim_kernel<<<num_blocks, 128>>>(
        graph.offsets, graph.indices,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, np);
}

}  
