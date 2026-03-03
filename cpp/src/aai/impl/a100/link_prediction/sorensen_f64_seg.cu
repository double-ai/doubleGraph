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




__global__ __launch_bounds__(256, 8)
void sorensen_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ similarity_scores,
    int64_t num_pairs
) {
    const int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane_id = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int32_t u = first[warp_id];
    int32_t v = second[warp_id];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];
    int32_t u_len = u_end - u_start;
    int32_t v_len = v_end - v_start;

    
    if (u_len == 0 || v_len == 0) {
        if (lane_id == 0) similarity_scores[warp_id] = 0.0;
        return;
    }

    
    int32_t ss, sl, ls, ll;
    if (u_len <= v_len) {
        ss = u_start; sl = u_len; ls = v_start; ll = v_len;
    } else {
        ss = v_start; sl = v_len; ls = u_start; ll = u_len;
    }

    
    int32_t sf = indices[ss];
    int32_t sla = indices[ss + sl - 1];
    int32_t lf = indices[ls];
    int32_t lla = indices[ls + ll - 1];

    if (sla < lf || lla < sf) {
        if (lane_id == 0) similarity_scores[warp_id] = 0.0;
        return;
    }

    
    double acc = 0.0;
    int32_t search_lo = 0;

    for (int32_t idx = lane_id; idx < sl; idx += 32) {
        int32_t val = indices[ss + idx];

        if (val > lla) break;
        if (val < lf) continue;

        
        int rank = 0;
        for (int32_t k = idx - 1; k >= 0 && indices[ss + k] == val; k--)
            rank++;

        
        int32_t lo = search_lo, hi = ll;
        while (lo < hi) {
            int32_t mid = (lo + hi) >> 1;
            if (indices[ls + mid] < val) lo = mid + 1;
            else hi = mid;
        }
        search_lo = lo;

        int32_t mp = lo + rank;
        if (mp < ll && indices[ls + mp] == val) {
            acc += fmin(edge_weights[ss + idx], edge_weights[ls + mp]);
        }
    }

    
    for (int s = 16; s >= 1; s >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, s);

    
    double intersection = __shfl_sync(0xffffffff, acc, 0);

    
    if (intersection == 0.0) {
        if (lane_id == 0) similarity_scores[warp_id] = 0.0;
        return;
    }

    
    double deg_u = 0.0;
    for (int32_t i = lane_id; i < u_len; i += 32)
        deg_u += edge_weights[u_start + i];
    for (int s = 16; s >= 1; s >>= 1)
        deg_u += __shfl_down_sync(0xffffffff, deg_u, s);
    deg_u = __shfl_sync(0xffffffff, deg_u, 0);

    double deg_v = 0.0;
    for (int32_t i = lane_id; i < v_len; i += 32)
        deg_v += edge_weights[v_start + i];
    for (int s = 16; s >= 1; s >>= 1)
        deg_v += __shfl_down_sync(0xffffffff, deg_v, s);
    deg_v = __shfl_sync(0xffffffff, deg_v, 0);

    if (lane_id == 0) {
        double denom = deg_u + deg_v;
        similarity_scores[warp_id] = (denom > 0.0) ? (2.0 * intersection / denom) : 0.0;
    }
}

}  

void sorensen_similarity_seg(const graph32_t& graph,
                             const double* edge_weights,
                             const int32_t* vertex_pairs_first,
                             const int32_t* vertex_pairs_second,
                             std::size_t num_pairs,
                             double* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    int block = 256;
    int64_t total_threads = (int64_t)num_pairs * 32;
    int64_t grid64 = (total_threads + block - 1) / block;
    int grid = (int)min(grid64, (int64_t)INT32_MAX);
    sorensen_kernel<<<grid, block>>>(
        offsets, indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, (int64_t)num_pairs
    );
}

}  
