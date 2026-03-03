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





__global__ __launch_bounds__(256, 6)
void overlap_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs)
{
    constexpr int SMEM_B_MAX = 128;
    constexpr int WARPS_PER_BLOCK = 8;

    __shared__ int32_t s_b_idx[WARPS_PER_BLOCK][SMEM_B_MAX];
    __shared__ float s_b_wt[WARPS_PER_BLOCK][SMEM_B_MAX];

    int warp_in_block = threadIdx.x >> 5;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int32_t u = __ldg(&pairs_first[warp_id]);
    int32_t v = __ldg(&pairs_second[warp_id]);

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);

    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    const int32_t* a_idx, *b_idx_g;
    const float* a_wt, *b_wt_g;
    int32_t a_len, b_len;

    if (u_deg <= v_deg) {
        a_idx = indices + u_start; a_wt = weights + u_start; a_len = u_deg;
        b_idx_g = indices + v_start; b_wt_g = weights + v_start; b_len = v_deg;
    } else {
        a_idx = indices + v_start; a_wt = weights + v_start; a_len = v_deg;
        b_idx_g = indices + u_start; b_wt_g = weights + u_start; b_len = u_deg;
    }

    bool use_smem = (b_len <= SMEM_B_MAX);

    
    
    bool has_match = false;

    if (use_smem) {
        
        for (int i = lane; i < b_len; i += 32) {
            s_b_idx[warp_in_block][i] = __ldg(&b_idx_g[i]);
        }
        __syncwarp();

        
        for (int i = lane; i < a_len; i += 32) {
            int32_t target = __ldg(&a_idx[i]);
            int lo = 0, hi = b_len;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (s_b_idx[warp_in_block][mid] < target) lo = mid + 1;
                else hi = mid;
            }
            if (lo < b_len && s_b_idx[warp_in_block][lo] == target) {
                has_match = true;
            }
        }
    } else {
        
        int lo_hint = 0;
        for (int i = lane; i < a_len; i += 32) {
            int32_t target = __ldg(&a_idx[i]);
            int lo = lo_hint, hi = b_len;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (__ldg(&b_idx_g[mid]) < target) lo = mid + 1;
                else hi = mid;
            }
            lo_hint = lo;
            if (lo < b_len && __ldg(&b_idx_g[lo]) == target) {
                has_match = true;
            }
        }
    }

    
    uint32_t any_match = __ballot_sync(0xffffffff, has_match);
    if (any_match == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    

    float b_wdeg = 0.0f;
    float a_wdeg = 0.0f;
    float intersection = 0.0f;

    if (use_smem) {
        
        for (int i = lane; i < b_len; i += 32) {
            float bw = __ldg(&b_wt_g[i]);
            s_b_wt[warp_in_block][i] = bw;
            b_wdeg += bw;
        }
        __syncwarp();

        
        for (int i = lane; i < a_len; i += 32) {
            float a_w = __ldg(&a_wt[i]);
            a_wdeg += a_w;
            int32_t target = __ldg(&a_idx[i]);

            int lo = 0, hi = b_len;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (s_b_idx[warp_in_block][mid] < target) lo = mid + 1;
                else hi = mid;
            }
            if (lo < b_len && s_b_idx[warp_in_block][lo] == target) {
                intersection += fminf(a_w, s_b_wt[warp_in_block][lo]);
            }
        }
    } else {
        
        for (int i = lane; i < b_len; i += 32) {
            b_wdeg += __ldg(&b_wt_g[i]);
        }
        
        int lo_hint2 = 0;
        for (int i = lane; i < a_len; i += 32) {
            float a_w = __ldg(&a_wt[i]);
            a_wdeg += a_w;
            int32_t target = __ldg(&a_idx[i]);

            int lo = lo_hint2, hi = b_len;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (__ldg(&b_idx_g[mid]) < target) lo = mid + 1;
                else hi = mid;
            }
            lo_hint2 = lo;
            if (lo < b_len && __ldg(&b_idx_g[lo]) == target) {
                intersection += fminf(a_w, __ldg(&b_wt_g[lo]));
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        a_wdeg += __shfl_down_sync(0xffffffff, a_wdeg, offset);
        b_wdeg += __shfl_down_sync(0xffffffff, b_wdeg, offset);
        intersection += __shfl_down_sync(0xffffffff, intersection, offset);
    }

    if (lane == 0) {
        float min_deg = fminf(a_wdeg, b_wdeg);
        scores[warp_id] = (min_deg > 0.0f) ? (intersection / min_deg) : 0.0f;
    }
}


__global__ __launch_bounds__(256, 6)
void overlap_kernel_multigraph(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs)
{
    constexpr int SMEM_B_MAX = 128;
    constexpr int WARPS_PER_BLOCK = 8;

    __shared__ int32_t s_b_idx[WARPS_PER_BLOCK][SMEM_B_MAX];
    __shared__ float s_b_wt[WARPS_PER_BLOCK][SMEM_B_MAX];

    int warp_in_block = threadIdx.x >> 5;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int32_t u = __ldg(&pairs_first[warp_id]);
    int32_t v = __ldg(&pairs_second[warp_id]);

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);

    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    const int32_t* a_idx, *b_idx_g;
    const float* a_wt, *b_wt_g;
    int32_t a_len, b_len;

    if (u_deg <= v_deg) {
        a_idx = indices + u_start; a_wt = weights + u_start; a_len = u_deg;
        b_idx_g = indices + v_start; b_wt_g = weights + v_start; b_len = v_deg;
    } else {
        a_idx = indices + v_start; a_wt = weights + v_start; a_len = v_deg;
        b_idx_g = indices + u_start; b_wt_g = weights + u_start; b_len = u_deg;
    }

    bool use_smem = (b_len <= SMEM_B_MAX);

    
    bool has_match = false;
    if (use_smem) {
        for (int i = lane; i < b_len; i += 32) {
            s_b_idx[warp_in_block][i] = __ldg(&b_idx_g[i]);
        }
        __syncwarp();
        for (int i = lane; i < a_len; i += 32) {
            int32_t target = __ldg(&a_idx[i]);
            int lo = 0, hi = b_len;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (s_b_idx[warp_in_block][mid] < target) lo = mid + 1;
                else hi = mid;
            }
            if (lo < b_len && s_b_idx[warp_in_block][lo] == target) {
                has_match = true;
            }
        }
    } else {
        int lo_hint = 0;
        for (int i = lane; i < a_len; i += 32) {
            int32_t target = __ldg(&a_idx[i]);
            int lo = lo_hint, hi = b_len;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (__ldg(&b_idx_g[mid]) < target) lo = mid + 1;
                else hi = mid;
            }
            lo_hint = lo;
            if (lo < b_len && __ldg(&b_idx_g[lo]) == target) {
                has_match = true;
            }
        }
    }

    uint32_t any_match = __ballot_sync(0xffffffff, has_match);
    if (any_match == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    float b_wdeg = 0.0f;
    if (use_smem) {
        for (int i = lane; i < b_len; i += 32) {
            float bw = __ldg(&b_wt_g[i]);
            s_b_wt[warp_in_block][i] = bw;
            b_wdeg += bw;
        }
        __syncwarp();
    } else {
        for (int i = lane; i < b_len; i += 32) {
            b_wdeg += __ldg(&b_wt_g[i]);
        }
    }

    float a_wdeg = 0.0f;
    float intersection = 0.0f;

    for (int chunk = 0; chunk < a_len; chunk += 32) {
        int idx = chunk + lane;
        int32_t my_a = (idx < a_len) ? __ldg(&a_idx[idx]) : 0x7fffffff;
        float my_aw = (idx < a_len) ? __ldg(&a_wt[idx]) : 0.0f;
        a_wdeg += my_aw;

        uint32_t match_mask = __match_any_sync(0xffffffff, my_a);
        uint32_t lower_mask = (1u << lane) - 1;
        int rank_in_chunk = __popc(match_mask & lower_mask);

        if (idx < a_len) {
            int rank_before = 0;
            if (chunk > 0) {
                int lo = 0, hi = chunk;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (__ldg(&a_idx[mid]) < my_a) lo = mid + 1;
                    else hi = mid;
                }
                int lower_pos = lo;
                lo = lower_pos; hi = chunk;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (__ldg(&a_idx[mid]) <= my_a) lo = mid + 1;
                    else hi = mid;
                }
                rank_before = lo - lower_pos;
            }

            int total_rank = rank_before + rank_in_chunk;

            if (use_smem) {
                int lo = 0, hi = b_len;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (s_b_idx[warp_in_block][mid] < my_a) lo = mid + 1;
                    else hi = mid;
                }
                int pos = lo + total_rank;
                if (pos < b_len && s_b_idx[warp_in_block][pos] == my_a) {
                    intersection += fminf(my_aw, s_b_wt[warp_in_block][pos]);
                }
            } else {
                int lo = 0, hi = b_len;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (__ldg(&b_idx_g[mid]) < my_a) lo = mid + 1;
                    else hi = mid;
                }
                int pos = lo + total_rank;
                if (pos < b_len && __ldg(&b_idx_g[pos]) == my_a) {
                    intersection += fminf(my_aw, __ldg(&b_wt_g[pos]));
                }
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        a_wdeg += __shfl_down_sync(0xffffffff, a_wdeg, offset);
        b_wdeg += __shfl_down_sync(0xffffffff, b_wdeg, offset);
        intersection += __shfl_down_sync(0xffffffff, intersection, offset);
    }

    if (lane == 0) {
        float min_deg = fminf(a_wdeg, b_wdeg);
        scores[warp_id] = (min_deg > 0.0f) ? (intersection / min_deg) : 0.0f;
    }
}

}  

void overlap_similarity_seg(const graph32_t& graph,
                            const float* edge_weights,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    if (num_pairs == 0) return;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    bool is_multigraph = graph.is_multigraph;

    const int threads = 256;
    const int warps_per_block = threads / 32;
    const int blocks = ((int)num_pairs + warps_per_block - 1) / warps_per_block;

    if (!is_multigraph) {
        overlap_kernel<<<blocks, threads>>>(
            offsets, indices, edge_weights,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int64_t)num_pairs);
    } else {
        overlap_kernel_multigraph<<<blocks, threads>>>(
            offsets, indices, edge_weights,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int64_t)num_pairs);
    }
}

}  
