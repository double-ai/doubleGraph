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
    bool* d_is_multigraph = nullptr;

    Cache() {
        cudaMalloc(&d_is_multigraph, sizeof(bool));
    }

    ~Cache() override {
        if (d_is_multigraph) cudaFree(d_is_multigraph);
    }
};



__device__ __forceinline__ int32_t lb_4ary(
    const int32_t* __restrict__ arr, int32_t lo, int32_t hi, int32_t target)
{
    while (hi - lo >= 8) {
        int32_t range = hi - lo;
        int32_t s1 = lo + range / 4;
        int32_t s2 = lo + range / 2;
        int32_t s3 = lo + range * 3 / 4;

        
        int32_t v1 = __ldg(&arr[s1]);
        int32_t v2 = __ldg(&arr[s2]);
        int32_t v3 = __ldg(&arr[s3]);

        
        if (v2 < target) {
            if (v3 < target) lo = s3 + 1;
            else { lo = s2 + 1; hi = s3; }
        } else {
            if (v1 < target) { lo = s1 + 1; hi = s2; }
            else hi = s1;
        }
    }
    
    while (lo < hi) {
        int32_t mid = (lo + hi) >> 1;
        if (__ldg(&arr[mid]) < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__device__ __forceinline__ int32_t lb_bin(
    const int32_t* __restrict__ arr, int32_t lo, int32_t hi, int32_t target)
{
    while (lo < hi) {
        int32_t mid = (lo + hi) >> 1;
        if (__ldg(&arr[mid]) < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void cosine_similarity_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs,
    const bool* __restrict__ is_multigraph_ptr)
{
    const int lane_id = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int64_t warp_id = (int64_t)(blockIdx.x * warps_per_block) + (threadIdx.x >> 5);

    if (warp_id >= num_pairs) return;

    const bool is_multigraph = *is_multigraph_ptr;

    const int32_t u = first[warp_id];
    const int32_t v = second[warp_id];

    const int32_t u_start = offsets[u];
    const int32_t u_end = offsets[u + 1];
    const int32_t v_start = offsets[v];
    const int32_t v_end = offsets[v + 1];

    const int32_t u_len = u_end - u_start;
    const int32_t v_len = v_end - v_start;

    if (u_len == 0 || v_len == 0) {
        if (lane_id == 0) scores[warp_id] = 0.0f / 0.0f;
        return;
    }

    const int32_t* short_idx;
    const float* short_wt;
    int32_t short_len;
    const int32_t* long_idx;
    const float* long_wt;
    int32_t long_len;
    bool u_is_short;

    if (u_len <= v_len) {
        short_idx = indices + u_start;
        short_wt = edge_weights + u_start;
        short_len = u_len;
        long_idx = indices + v_start;
        long_wt = edge_weights + v_start;
        long_len = v_len;
        u_is_short = true;
    } else {
        short_idx = indices + v_start;
        short_wt = edge_weights + v_start;
        short_len = v_len;
        long_idx = indices + u_start;
        long_wt = edge_weights + u_start;
        long_len = u_len;
        u_is_short = false;
    }

    float dot = 0.0f, norm_s_sq = 0.0f, norm_l_sq = 0.0f;

    if (is_multigraph) {
        for (int32_t i = lane_id; i < short_len; i += 32) {
            int32_t elem = __ldg(&short_idx[i]);
            int32_t first_occ = lb_bin(short_idx, 0, i, elem);
            int32_t rank = i - first_occ;
            int32_t lb = lb_4ary(long_idx, 0, long_len, elem) + rank;
            if (lb < long_len && __ldg(&long_idx[lb]) == elem) {
                float ws = __ldg(&short_wt[i]);
                float wl = __ldg(&long_wt[lb]);
                dot += ws * wl;
                norm_s_sq += ws * ws;
                norm_l_sq += wl * wl;
            }
        }
    } else {
        
        int32_t search_lo = 0;
        for (int32_t i = lane_id; i < short_len; i += 32) {
            int32_t elem = __ldg(&short_idx[i]);
            int32_t lb = lb_4ary(long_idx, search_lo, long_len, elem);
            if (lb < long_len && __ldg(&long_idx[lb]) == elem) {
                float ws = __ldg(&short_wt[i]);
                float wl = __ldg(&long_wt[lb]);
                dot += ws * wl;
                norm_s_sq += ws * ws;
                norm_l_sq += wl * wl;
                search_lo = lb + 1;
            } else {
                search_lo = lb;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
        norm_s_sq += __shfl_down_sync(0xffffffff, norm_s_sq, offset);
        norm_l_sq += __shfl_down_sync(0xffffffff, norm_l_sq, offset);
    }

    if (lane_id == 0) {
        float norm_u_sq = u_is_short ? norm_s_sq : norm_l_sq;
        float norm_v_sq = u_is_short ? norm_l_sq : norm_s_sq;
        float denom = sqrtf(norm_u_sq) * sqrtf(norm_v_sq);
        scores[warp_id] = dot / denom;
    }
}

}  

void cosine_similarity(const graph32_t& graph,
                       const float* edge_weights,
                       const int32_t* vertex_pairs_first,
                       const int32_t* vertex_pairs_second,
                       std::size_t num_pairs,
                       float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    
    bool is_mg = graph.is_multigraph;
    cudaMemcpy(cache.d_is_multigraph, &is_mg, sizeof(bool), cudaMemcpyHostToDevice);

    if (num_pairs == 0) return;

    const int block_size = 128;
    int warps_per_block = block_size / 32;
    int blocks = (int)((num_pairs + warps_per_block - 1) / warps_per_block);

    cosine_similarity_kernel<<<blocks, block_size>>>(
        graph.offsets, graph.indices, edge_weights,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores,
        (int64_t)num_pairs,
        cache.d_is_multigraph);
}

}  
