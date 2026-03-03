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


__device__ __forceinline__ int32_t lower_bound_prefetch(
    const int32_t* __restrict__ arr,
    int32_t lo, int32_t hi, int32_t target
) {
    int32_t size = hi - lo;
    while (size > 1) {
        int32_t half = size >> 1;
        int32_t mid = lo + half;
        int32_t val = __ldg(&arr[mid]);

        
        int32_t next_half = (size - half) >> 1;
        if (next_half > 0) {
            
            asm volatile("prefetch.global.L1 [%0];" :: "l"((const void*)&arr[lo + next_half]));
            
            asm volatile("prefetch.global.L1 [%0];" :: "l"((const void*)&arr[mid + 1 + next_half]));
        }

        lo = (val < target) ? mid + 1 : lo;
        size = (val < target) ? (size - half - 1) : half;
    }
    if (size == 1 && __ldg(&arr[lo]) < target) lo++;
    return lo;
}


__device__ __forceinline__ int32_t lower_bound_std(
    const int32_t* __restrict__ arr,
    int32_t lo, int32_t hi, int32_t target
) {
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (__ldg(&arr[mid]) < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

template <int TILE_SIZE>
__device__ __forceinline__ void jaccard_core(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs,
    int lane,
    int64_t pair_idx
) {
    if (pair_idx >= num_pairs) return;

    int32_t u = first[pair_idx];
    int32_t v = second[pair_idx];

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);

    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) scores[pair_idx] = 0.0f;
        return;
    }

    int32_t s_off, s_deg, l_off, l_deg;
    if (u_deg <= v_deg) {
        s_off = u_start; s_deg = u_deg;
        l_off = v_start; l_deg = v_deg;
    } else {
        s_off = v_start; s_deg = v_deg;
        l_off = u_start; l_deg = u_deg;
    }

    int32_t s_min = __ldg(&indices[s_off]);
    int32_t s_max = __ldg(&indices[s_off + s_deg - 1]);
    int32_t l_min = __ldg(&indices[l_off]);
    int32_t l_max = __ldg(&indices[l_off + l_deg - 1]);

    if (s_max < l_min || l_max < s_min) {
        if (lane == 0) scores[pair_idx] = 0.0f;
        return;
    }

    float local_isect = 0.0f;
    int32_t l_end = l_off + l_deg;

    for (int32_t k = lane; k < s_deg; k += TILE_SIZE) {
        int32_t target = __ldg(&indices[s_off + k]);
        if (target > l_max) break;
        if (target < l_min) continue;

        int32_t rank = 0;
        { int32_t c = k - 1; while (c >= 0 && __ldg(&indices[s_off + c]) == target) { rank++; c--; } }

        
        int32_t lb;
        if (l_deg > 16)
            lb = lower_bound_prefetch(indices, l_off, l_end, target);
        else
            lb = lower_bound_std(indices, l_off, l_end, target);

        int32_t mp = lb + rank;
        if (mp < l_end && __ldg(&indices[mp]) == target) {
            local_isect += fminf(__ldg(&edge_weights[s_off + k]), __ldg(&edge_weights[mp]));
        }
    }

    #pragma unroll
    for (int off = TILE_SIZE / 2; off > 0; off >>= 1)
        local_isect += __shfl_down_sync(0xffffffff, local_isect, off, TILE_SIZE);

    int warp_lane = threadIdx.x & 31;
    int tile_base = warp_lane & ~(TILE_SIZE - 1);
    float total_isect = __shfl_sync(0xffffffff, local_isect, tile_base);

    if (total_isect > 0.0f) {
        float su = 0.0f;
        for (int32_t i = lane; i < u_deg; i += TILE_SIZE)
            su += __ldg(&edge_weights[u_start + i]);
        #pragma unroll
        for (int off = TILE_SIZE / 2; off > 0; off >>= 1)
            su += __shfl_down_sync(0xffffffff, su, off, TILE_SIZE);

        float sv = 0.0f;
        for (int32_t i = lane; i < v_deg; i += TILE_SIZE)
            sv += __ldg(&edge_weights[v_start + i]);
        #pragma unroll
        for (int off = TILE_SIZE / 2; off > 0; off >>= 1)
            sv += __shfl_down_sync(0xffffffff, sv, off, TILE_SIZE);

        if (lane == 0) {
            float denom = su + sv - total_isect;
            scores[pair_idx] = (denom > 0.0f) ? (total_isect / denom) : 0.0f;
        }
    } else {
        if (lane == 0) scores[pair_idx] = 0.0f;
    }
}

__global__ __launch_bounds__(128)
void jaccard_w32(const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights, const int32_t* __restrict__ first,
    const int32_t* __restrict__ second, float* __restrict__ scores, int64_t num_pairs) {
    jaccard_core<32>(offsets, indices, edge_weights, first, second, scores, num_pairs,
                     threadIdx.x & 31, ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
}

__global__ __launch_bounds__(128)
void jaccard_w16(const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights, const int32_t* __restrict__ first,
    const int32_t* __restrict__ second, float* __restrict__ scores, int64_t num_pairs) {
    jaccard_core<16>(offsets, indices, edge_weights, first, second, scores, num_pairs,
                     threadIdx.x & 15, ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 4);
}

__global__ __launch_bounds__(128)
void jaccard_w8(const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights, const int32_t* __restrict__ first,
    const int32_t* __restrict__ second, float* __restrict__ scores, int64_t num_pairs) {
    jaccard_core<8>(offsets, indices, edge_weights, first, second, scores, num_pairs,
                    threadIdx.x & 7, ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 3);
}

__global__ __launch_bounds__(128)
void jaccard_w4(const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights, const int32_t* __restrict__ first,
    const int32_t* __restrict__ second, float* __restrict__ scores, int64_t num_pairs) {
    jaccard_core<4>(offsets, indices, edge_weights, first, second, scores, num_pairs,
                    threadIdx.x & 3, ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 2);
}

}  

void jaccard_similarity(const graph32_t& graph,
                        const float* edge_weights,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    float avg_degree = (float)num_edges / (float)num_vertices;
    int tile_size;
    if (avg_degree <= 8) {
        tile_size = 8;
    } else if (avg_degree <= 48) {
        tile_size = 16;
    } else {
        tile_size = 32;
    }

    int block = 128;
    cudaStream_t stream = 0;
    if (tile_size <= 4) {
        int grid = (int)((num_pairs * 4 + block - 1) / block);
        jaccard_w4<<<grid, block, 0, stream>>>(offsets, indices, edge_weights, vertex_pairs_first, vertex_pairs_second, similarity_scores, num_pairs);
    } else if (tile_size <= 8) {
        int grid = (int)((num_pairs * 8 + block - 1) / block);
        jaccard_w8<<<grid, block, 0, stream>>>(offsets, indices, edge_weights, vertex_pairs_first, vertex_pairs_second, similarity_scores, num_pairs);
    } else if (tile_size <= 16) {
        int grid = (int)((num_pairs * 16 + block - 1) / block);
        jaccard_w16<<<grid, block, 0, stream>>>(offsets, indices, edge_weights, vertex_pairs_first, vertex_pairs_second, similarity_scores, num_pairs);
    } else {
        int grid = (int)((num_pairs * 32 + block - 1) / block);
        jaccard_w32<<<grid, block, 0, stream>>>(offsets, indices, edge_weights, vertex_pairs_first, vertex_pairs_second, similarity_scores, num_pairs);
    }
}

}  
