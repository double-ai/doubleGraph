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

struct Cache : Cacheable {
    float* weighted_degrees = nullptr;
    int64_t wd_capacity = 0;

    void ensure(int32_t num_vertices) {
        int64_t needed = (int64_t)num_vertices;
        if (wd_capacity < needed) {
            if (weighted_degrees) cudaFree(weighted_degrees);
            cudaMalloc(&weighted_degrees, needed * sizeof(float));
            wd_capacity = needed;
        }
    }

    ~Cache() override {
        if (weighted_degrees) cudaFree(weighted_degrees);
    }
};

__global__ void compute_weighted_degree_kernel(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ edge_weights,
    float* __restrict__ weighted_degrees,
    int32_t num_vertices
) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    float sum = 0.0f;
    for (int32_t e = start; e < end; e++) {
        sum += edge_weights[e];
    }
    weighted_degrees[v] = sum;
}

__device__ __forceinline__ int32_t dev_lower_bound(
    const int32_t* __restrict__ arr, int32_t lo, int32_t hi, int32_t val
) {
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        lo = (arr[mid] < val) ? (mid + 1) : lo;
        hi = (arr[mid] < val) ? hi : mid;
    }
    return lo;
}

__device__ __forceinline__ float compute_pair_intersection(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    int32_t u, int32_t v,
    int half_lane, int half_size
) {
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];

    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) return 0.0f;

    int32_t s_start, s_end, l_start, l_end, s_len;
    if (u_deg <= v_deg) {
        s_start = u_start; s_end = u_end; s_len = u_deg;
        l_start = v_start; l_end = v_end;
    } else {
        s_start = v_start; s_end = v_end; s_len = v_deg;
        l_start = u_start; l_end = u_end;
    }

    float local_sum = 0.0f;

    for (int32_t i = half_lane; i < s_len; i += half_size) {
        int32_t neighbor = indices[s_start + i];
        float w_s = edge_weights[s_start + i];

        int32_t pos = dev_lower_bound(indices, l_start, l_end, neighbor);

        if (pos < l_end && indices[pos] == neighbor) {
            local_sum += fminf(w_s, edge_weights[pos]);
        }
    }

    return local_sum;
}

__global__ void jaccard_dual_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ weighted_degrees,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    const int lane = threadIdx.x & 31;
    const int half = lane >> 4;
    const int half_lane = lane & 15;
    const int64_t warp_id = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int64_t pair_id = warp_id * 2 + half;

    if (pair_id >= num_pairs) return;

    int32_t u = pairs_first[pair_id];
    int32_t v = pairs_second[pair_id];

    float wd_u = weighted_degrees[u];
    float wd_v = weighted_degrees[v];

    float local_sum = compute_pair_intersection(
        offsets, indices, edge_weights, u, v, half_lane, 16);

    unsigned half_mask = half ? 0xffff0000u : 0x0000ffffu;
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(half_mask, local_sum, offset);
    }

    if (half_lane == 0) {
        float union_val = wd_u + wd_v - local_sum;
        scores[pair_id] = (union_val > 0.0f) ? (local_sum / union_val) : 0.0f;
    }
}

__global__ void jaccard_warp_kernel_multigraph(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ weighted_degrees,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    const int64_t warp_id = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int32_t u = pairs_first[warp_id];
    int32_t v = pairs_second[warp_id];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];

    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    float wd_u = weighted_degrees[u];
    float wd_v = weighted_degrees[v];

    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    int32_t s_start, s_end, l_start, l_end, s_len;
    if (u_deg <= v_deg) {
        s_start = u_start; s_end = u_end; s_len = u_deg;
        l_start = v_start; l_end = v_end;
    } else {
        s_start = v_start; s_end = v_end; s_len = v_deg;
        l_start = u_start; l_end = u_end;
    }

    float local_sum = 0.0f;

    for (int32_t i = lane; i < s_len; i += 32) {
        int32_t neighbor = indices[s_start + i];
        float w_s = edge_weights[s_start + i];

        int32_t lb_s = dev_lower_bound(indices, s_start, s_end, neighbor);
        int32_t rank = (s_start + i) - lb_s;
        int32_t lb_l = dev_lower_bound(indices, l_start, l_end, neighbor);
        int32_t pos_l = lb_l + rank;

        if (pos_l < l_end && indices[pos_l] == neighbor) {
            local_sum += fminf(w_s, edge_weights[pos_l]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) {
        float union_val = wd_u + wd_v - local_sum;
        scores[warp_id] = (union_val > 0.0f) ? (local_sum / union_val) : 0.0f;
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

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    bool is_multigraph = graph.is_multigraph;

    cache.ensure(num_vertices);

    cudaStream_t stream = 0;

    
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        compute_weighted_degree_kernel<<<grid, block, 0, stream>>>(
            offsets, edge_weights, cache.weighted_degrees, num_vertices);
    }

    
    int64_t np = (int64_t)num_pairs;
    if (is_multigraph) {
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / 32;
        int64_t num_blocks = (np + warps_per_block - 1) / warps_per_block;
        jaccard_warp_kernel_multigraph<<<(int)num_blocks, threads_per_block, 0, stream>>>(
            offsets, indices, edge_weights, cache.weighted_degrees,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, np);
    } else {
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / 32;
        int pairs_per_block = warps_per_block * 2;
        int64_t num_blocks = (np + pairs_per_block - 1) / pairs_per_block;
        jaccard_dual_kernel<<<(int)num_blocks, threads_per_block, 0, stream>>>(
            offsets, indices, edge_weights, cache.weighted_degrees,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, np);
    }
}

}  
