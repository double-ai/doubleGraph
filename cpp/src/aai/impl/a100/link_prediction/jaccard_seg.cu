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



template <int W, int SMEM_PER_PAIR>
__global__ void jaccard_smem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    constexpr int PAIRS_PER_BLOCK = 256 / W;  

    
    extern __shared__ int32_t smem[];
    int32_t* my_b_cache = smem + (threadIdx.x / W) * SMEM_PER_PAIR;

    int64_t pair_id = (int64_t)blockIdx.x * PAIRS_PER_BLOCK + (threadIdx.x / W);
    int sublane = threadIdx.x & (W - 1);

    if (pair_id >= num_pairs) return;

    int32_t u = pairs_first[pair_id];
    int32_t v = pairs_second[pair_id];

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);

    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;

    if (deg_u + deg_v == 0) {
        if (sublane == 0) scores[pair_id] = 0.0f;
        return;
    }

    const int32_t* a_ptr, *b_ptr;
    int32_t a_len, b_len;
    if (deg_u <= deg_v) {
        a_ptr = indices + u_start; a_len = deg_u;
        b_ptr = indices + v_start; b_len = deg_v;
    } else {
        a_ptr = indices + v_start; a_len = deg_v;
        b_ptr = indices + u_start; b_len = deg_u;
    }

    
    bool use_smem = (b_len <= SMEM_PER_PAIR);

    const int32_t* search_ptr = b_ptr;
    if (use_smem) {
        
        for (int j = sublane; j < b_len; j += W) {
            my_b_cache[j] = __ldg(&b_ptr[j]);
        }
        
        
        
        
        search_ptr = my_b_cache;
    }

    int count = 0;
    int b_lo = 0;

    for (int i = sublane; i < a_len; i += W) {
        int32_t val = __ldg(&a_ptr[i]);
        int lo = b_lo, hi = b_len;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            int32_t mid_val = use_smem ? search_ptr[mid] : __ldg(&search_ptr[mid]);
            if (mid_val < val) lo = mid + 1;
            else hi = mid;
        }
        if (lo < b_len) {
            int32_t lo_val = use_smem ? search_ptr[lo] : __ldg(&search_ptr[lo]);
            if (lo_val == val) {
                count++;
                b_lo = lo + 1;
            } else {
                b_lo = lo;
            }
        } else {
            b_lo = lo;
        }
    }

    #pragma unroll
    for (int s = W/2; s > 0; s >>= 1)
        count += __shfl_down_sync(0xffffffff, count, s, W);

    if (sublane == 0) {
        float intersection = (float)count;
        float union_size = (float)(deg_u + deg_v) - intersection;
        scores[pair_id] = (union_size > 0.0f) ? intersection / union_size : 0.0f;
    }
}


template <int W>
__global__ void jaccard_subwarp_simple(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t pair_id = (int64_t)blockIdx.x * (blockDim.x / W) + (threadIdx.x / W);
    int sublane = threadIdx.x & (W - 1);

    if (pair_id >= num_pairs) return;

    int32_t u = pairs_first[pair_id];
    int32_t v = pairs_second[pair_id];

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);

    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;

    if (deg_u + deg_v == 0) {
        if (sublane == 0) scores[pair_id] = 0.0f;
        return;
    }

    const int32_t* a_ptr, *b_ptr;
    int32_t a_len, b_len;
    if (deg_u <= deg_v) {
        a_ptr = indices + u_start; a_len = deg_u;
        b_ptr = indices + v_start; b_len = deg_v;
    } else {
        a_ptr = indices + v_start; a_len = deg_v;
        b_ptr = indices + u_start; b_len = deg_u;
    }

    int count = 0;
    int b_lo = 0;

    for (int i = sublane; i < a_len; i += W) {
        int32_t val = __ldg(&a_ptr[i]);
        int lo = b_lo, hi = b_len;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (__ldg(&b_ptr[mid]) < val) lo = mid + 1;
            else hi = mid;
        }
        if (lo < b_len && __ldg(&b_ptr[lo]) == val) {
            count++;
            b_lo = lo + 1;
        } else {
            b_lo = lo;
        }
    }

    #pragma unroll
    for (int s = W/2; s > 0; s >>= 1)
        count += __shfl_down_sync(0xffffffff, count, s, W);

    if (sublane == 0) {
        float intersection = (float)count;
        float union_size = (float)(deg_u + deg_v) - intersection;
        scores[pair_id] = (union_size > 0.0f) ? intersection / union_size : 0.0f;
    }
}


template <int W>
__global__ void jaccard_multi(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t pair_id = (int64_t)blockIdx.x * (blockDim.x / W) + (threadIdx.x / W);
    int sublane = threadIdx.x & (W - 1);

    if (pair_id >= num_pairs) return;

    int32_t u = pairs_first[pair_id];
    int32_t v = pairs_second[pair_id];

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);

    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;

    if (deg_u + deg_v == 0) {
        if (sublane == 0) scores[pair_id] = 0.0f;
        return;
    }

    const int32_t* a_ptr, *b_ptr;
    int32_t a_len, b_len;
    if (deg_u <= deg_v) {
        a_ptr = indices + u_start; a_len = deg_u;
        b_ptr = indices + v_start; b_len = deg_v;
    } else {
        a_ptr = indices + v_start; a_len = deg_v;
        b_ptr = indices + u_start; b_len = deg_u;
    }

    int count = 0;
    for (int i = sublane; i < a_len; i += W) {
        int32_t val = __ldg(&a_ptr[i]);
        int rank = 0;
        if (i > 0 && __ldg(&a_ptr[i - 1]) == val) {
            int lo2 = 0, hi2 = i;
            while (lo2 < hi2) { int mid = (lo2 + hi2) >> 1; if (__ldg(&a_ptr[mid]) < val) lo2 = mid + 1; else hi2 = mid; }
            rank = i - lo2;
        }
        int lo = 0, hi = b_len;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (__ldg(&b_ptr[mid]) < val) lo = mid + 1;
            else hi = mid;
        }
        int pos = lo + rank;
        if (pos < b_len && __ldg(&b_ptr[pos]) == val) count++;
    }

    #pragma unroll
    for (int s = W/2; s > 0; s >>= 1)
        count += __shfl_down_sync(0xffffffff, count, s, W);

    if (sublane == 0) {
        float intersection = (float)count;
        float union_size = (float)(deg_u + deg_v) - intersection;
        scores[pair_id] = (union_size > 0.0f) ? intersection / union_size : 0.0f;
    }
}

}  

void jaccard_similarity_seg(const graph32_t& graph,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int64_t np = static_cast<int64_t>(num_pairs);
    int threads = 256;

    if (graph.is_multigraph) {
        constexpr int W = 8;
        int pairs_per_block = threads / W;
        int blocks = (int)((np + pairs_per_block - 1) / pairs_per_block);
        jaccard_multi<W><<<blocks, threads>>>(
            offsets, indices, vertex_pairs_first, vertex_pairs_second,
            similarity_scores, np);
    } else {
        constexpr int W = 4;
        int pairs_per_block = threads / W;
        int blocks = (int)((np + pairs_per_block - 1) / pairs_per_block);
        jaccard_subwarp_simple<W><<<blocks, threads>>>(
            offsets, indices, vertex_pairs_first, vertex_pairs_second,
            similarity_scores, np);
    }
}

}  
