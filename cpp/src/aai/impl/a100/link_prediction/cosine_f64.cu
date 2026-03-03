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
#include <math.h>

namespace aai {

namespace {

struct Cache : Cacheable {};

template <int GROUP_SIZE>
__global__ void cosine_similarity_subwarp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    double* __restrict__ scores,
    int64_t num_pairs
) {
    static_assert(GROUP_SIZE > 0 && (GROUP_SIZE & (GROUP_SIZE - 1)) == 0, "GROUP_SIZE must be power of 2");
    static_assert(GROUP_SIZE <= 32, "GROUP_SIZE must be <= 32");

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane_in_group = tid & (GROUP_SIZE - 1);
    const int pair_idx = tid / GROUP_SIZE;

    if (pair_idx >= num_pairs) return;

    const int u = pairs_first[pair_idx];
    const int v = pairs_second[pair_idx];

    const int u_start = offsets[u];
    const int u_end = offsets[u + 1];
    const int v_start = offsets[v];
    const int v_end = offsets[v + 1];

    int u_len = u_end - u_start;
    int v_len = v_end - v_start;

    
    int small_start, small_len, large_start, large_len;
    if (u_len <= v_len) {
        small_start = u_start; small_len = u_len;
        large_start = v_start; large_len = v_len;
    } else {
        small_start = v_start; small_len = v_len;
        large_start = u_start; large_len = u_len;
    }

    double dot = 0.0, norm_s = 0.0, norm_l = 0.0;

    
    int search_lo = 0;

    for (int i = lane_in_group; i < small_len; i += GROUP_SIZE) {
        int target = indices[small_start + i];

        
        int dup_count = 0;
        if (i > 0 && indices[small_start + i - 1] == target) {
            dup_count = 1;
            for (int k = i - 2; k >= 0 && indices[small_start + k] == target; k--) {
                dup_count++;
            }
        }

        
        int lo = search_lo, hi = large_len;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (indices[large_start + mid] < target) lo = mid + 1;
            else hi = mid;
        }

        
        search_lo = lo;

        
        int match_pos = lo + dup_count;
        if (match_pos < large_len && indices[large_start + match_pos] == target) {
            double ws = edge_weights[small_start + i];
            double wl = edge_weights[large_start + match_pos];
            dot += ws * wl;
            norm_s += ws * ws;
            norm_l += wl * wl;
        }
    }

    
    #pragma unroll
    for (int offset = GROUP_SIZE / 2; offset > 0; offset >>= 1) {
        dot += __shfl_xor_sync(0xFFFFFFFF, dot, offset);
        norm_s += __shfl_xor_sync(0xFFFFFFFF, norm_s, offset);
        norm_l += __shfl_xor_sync(0xFFFFFFFF, norm_l, offset);
    }

    if (lane_in_group == 0) {
        double denom = sqrt(norm_s) * sqrt(norm_l);
        scores[pair_idx] = dot / denom;
    }
}

}  

void cosine_similarity(const graph32_t& graph,
                       const double* edge_weights,
                       const int32_t* vertex_pairs_first,
                       const int32_t* vertex_pairs_second,
                       std::size_t num_pairs,
                       double* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    if (num_pairs <= 0) return;

    constexpr int GROUP_SIZE = 16;
    const int threads = 256;
    const int pairs_per_block = threads / GROUP_SIZE;
    const int num_blocks = (int)((num_pairs + pairs_per_block - 1) / pairs_per_block);

    cosine_similarity_subwarp<GROUP_SIZE><<<num_blocks, threads>>>(
        graph.offsets, graph.indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, (int64_t)num_pairs
    );
}

}  
