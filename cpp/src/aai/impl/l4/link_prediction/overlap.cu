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

__device__ __forceinline__ int lb_dev(const int32_t* __restrict__ arr, int size, int32_t target) {
    int lo = 0;
    while (size > 0) {
        int half = size >> 1;
        int mid = lo + half;
        lo = (arr[mid] < target) ? (mid + 1) : lo;
        size = (arr[mid] < target) ? (size - half - 1) : half;
    }
    return lo;
}


__global__ void overlap_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs,
    bool is_multigraph
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int32_t u = first[idx];
    int32_t v = second[idx];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];

    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;
    int32_t min_deg = (deg_u < deg_v) ? deg_u : deg_v;

    if (min_deg == 0) {
        scores[idx] = 0.0f;
        return;
    }

    const int32_t* a = indices + u_start;
    const int32_t* b = indices + v_start;
    int32_t sa = deg_u, sb = deg_v;

    
    int count = 0;
    int i = 0, j = 0;
    while (i < sa && j < sb) {
        int32_t va = a[i];
        int32_t vb = b[j];
        count += (va == vb);
        i += (va <= vb);
        j += (va >= vb);
    }

    scores[idx] = (float)count / (float)min_deg;
}



template<int TPP>
__global__ void overlap_subwarp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs,
    bool is_multigraph
) {
    const int lane = threadIdx.x & (TPP - 1);
    const int64_t pair_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / TPP;
    const unsigned mask = (TPP == 32) ? 0xFFFFFFFF : ((1u << TPP) - 1) << ((threadIdx.x / TPP) * TPP % 32);

    if (pair_id >= num_pairs) return;

    int32_t u = first[pair_id];
    int32_t v = second[pair_id];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];

    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;
    int32_t min_deg = (deg_u < deg_v) ? deg_u : deg_v;

    if (min_deg == 0) {
        if (lane == 0) scores[pair_id] = 0.0f;
        return;
    }

    
    const int32_t* a, *b;
    int32_t sa, sb;
    if (deg_u <= deg_v) {
        a = indices + u_start; sa = deg_u;
        b = indices + v_start; sb = deg_v;
    } else {
        a = indices + v_start; sa = deg_v;
        b = indices + u_start; sb = deg_u;
    }

    int local_count = 0;

    for (int i = lane; i < sa; i += TPP) {
        int32_t target = a[i];

        
        int rank = 0;
        if (is_multigraph && i > 0 && a[i - 1] == target) {
            rank = i - lb_dev(a, i, target);
        }

        int pos = lb_dev(b, sb, target);
        int target_pos = pos + rank;
        if (target_pos < sb && b[target_pos] == target) {
            local_count++;
        }
    }

    
    #pragma unroll
    for (int offset = TPP >> 1; offset > 0; offset >>= 1) {
        local_count += __shfl_down_sync(mask, local_count, offset);
    }

    if (lane == 0) {
        scores[pair_id] = (float)local_count / (float)min_deg;
    }
}

}  

void overlap_similarity(const graph32_t& graph,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        float* similarity_scores) {
    if (num_pairs == 0) return;

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    bool is_multigraph = graph.is_multigraph;

    float avg_degree = (num_vertices > 0) ? ((float)num_edges / num_vertices) : 0;

    int threads_per_block = 256;

    if (avg_degree < 6.0f) {
        
        int grid = (int)((num_pairs + threads_per_block - 1) / threads_per_block);
        overlap_thread_kernel<<<grid, threads_per_block>>>(
            graph.offsets, graph.indices,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, num_pairs, is_multigraph
        );
    } else if (avg_degree < 32.0f) {
        
        constexpr int TPP = 8;
        int64_t threads_needed = num_pairs * TPP;
        int grid = (int)((threads_needed + threads_per_block - 1) / threads_per_block);
        overlap_subwarp_kernel<TPP><<<grid, threads_per_block>>>(
            graph.offsets, graph.indices,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, num_pairs, is_multigraph
        );
    } else {
        
        constexpr int TPP = 32;
        int64_t threads_needed = num_pairs * TPP;
        int grid = (int)((threads_needed + threads_per_block - 1) / threads_per_block);
        overlap_subwarp_kernel<TPP><<<grid, threads_per_block>>>(
            graph.offsets, graph.indices,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, num_pairs, is_multigraph
        );
    }
}

}  
