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
#include <math_constants.h>

namespace aai {

namespace {

struct Cache : Cacheable {};





__global__ void __launch_bounds__(256)
cosine_similarity_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t pair_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= num_pairs) return;

    int u = first[pair_idx];
    int v = second[pair_idx];

    int start_u = offsets[u];
    int end_u = offsets[u + 1];
    int start_v = offsets[v];
    int end_v = offsets[v + 1];
    int len_u = end_u - start_u;
    int len_v = end_v - start_v;

    if (len_u == 0 || len_v == 0) {
        scores[pair_idx] = CUDART_NAN_F;
        return;
    }

    int first_u = indices[start_u];
    int last_u = indices[end_u - 1];
    int first_v = indices[start_v];
    int last_v = indices[end_v - 1];

    if (first_u > last_v || first_v > last_u) {
        scores[pair_idx] = CUDART_NAN_F;
        return;
    }

    int i = start_u, j = start_v;

    
    if (first_u < first_v) {
        int lo = start_u, hi = end_u;
        while (lo < hi) { int mid = (lo + hi) >> 1; if (indices[mid] < first_v) lo = mid + 1; else hi = mid; }
        i = lo;
    } else if (first_v < first_u) {
        int lo = start_v, hi = end_v;
        while (lo < hi) { int mid = (lo + hi) >> 1; if (indices[mid] < first_u) lo = mid + 1; else hi = mid; }
        j = lo;
    }

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    while (i < end_u && j < end_v) {
        int ni = indices[i], nj = indices[j];
        if (ni == nj) {
            float wu = edge_weights[i], wv = edge_weights[j];
            dot += wu * wv;
            norm_a += wu * wu;
            norm_b += wv * wv;
            i++; j++;
        } else if (ni < nj) { i++; } else { j++; }
    }

    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    scores[pair_idx] = dot / denom;
}




template <int SubwarpSize>
__global__ void __launch_bounds__(256)
cosine_similarity_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    int global_tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int pair_id = global_tid / SubwarpSize;
    int lane = global_tid & (SubwarpSize - 1);

    if (pair_id >= num_pairs) return;

    int u = first[pair_id];
    int v = second[pair_id];

    int start_u = offsets[u];
    int end_u = offsets[u + 1];
    int start_v = offsets[v];
    int end_v = offsets[v + 1];
    int len_u = end_u - start_u;
    int len_v = end_v - start_v;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    if (len_u > 0 && len_v > 0) {
        int a_start, a_len, b_start, b_len;
        if (len_u <= len_v) {
            a_start = start_u; a_len = len_u;
            b_start = start_v; b_len = len_v;
        } else {
            a_start = start_v; a_len = len_v;
            b_start = start_u; b_len = len_u;
        }

        
        int first_a = indices[a_start];
        int last_a = indices[a_start + a_len - 1];
        int first_b = indices[b_start];
        int last_b = indices[b_start + b_len - 1];

        if (first_a <= last_b && first_b <= last_a) {
            
            int b_lo = 0;
            if (first_b < first_a) {
                int lo = 0, hi = b_len;
                while (lo < hi) { int mid = (lo + hi) >> 1; if (indices[b_start + mid] < first_a) lo = mid + 1; else hi = mid; }
                b_lo = lo;
            }
            int b_hi = b_len;
            if (last_b > last_a) {
                int lo = b_lo, hi = b_len;
                while (lo < hi) { int mid = (lo + hi) >> 1; if (indices[b_start + mid] <= last_a) lo = mid + 1; else hi = mid; }
                b_hi = lo;
            }

            for (int i = lane; i < a_len; i += SubwarpSize) {
                int target = indices[a_start + i];
                float w_a = edge_weights[a_start + i];

                int lo = b_lo, hi = b_hi;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (indices[b_start + mid] < target) lo = mid + 1;
                    else hi = mid;
                }

                if (lo < b_hi && indices[b_start + lo] == target) {
                    float w_b = edge_weights[b_start + lo];
                    dot += w_a * w_b;
                    norm_a += w_a * w_a;
                    norm_b += w_b * w_b;
                }
            }
        }
    }

    
    #pragma unroll
    for (int offset = SubwarpSize / 2; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, offset, SubwarpSize);
        norm_a += __shfl_down_sync(0xffffffff, norm_a, offset, SubwarpSize);
        norm_b += __shfl_down_sync(0xffffffff, norm_b, offset, SubwarpSize);
    }

    if (lane == 0) {
        float denom = sqrtf(norm_a) * sqrtf(norm_b);
        scores[pair_id] = dot / denom;
    }
}




__global__ void __launch_bounds__(256)
cosine_similarity_multigraph_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    int pair_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (pair_id >= num_pairs) return;

    int u = first[pair_id];
    int v = second[pair_id];

    int start_u = offsets[u];
    int end_u = offsets[u + 1];
    int start_v = offsets[v];
    int end_v = offsets[v + 1];
    int len_u = end_u - start_u;
    int len_v = end_v - start_v;

    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    if (len_u > 0 && len_v > 0) {
        int a_start, a_len, b_start, b_len;
        if (len_u <= len_v) {
            a_start = start_u; a_len = len_u;
            b_start = start_v; b_len = len_v;
        } else {
            a_start = start_v; a_len = len_v;
            b_start = start_u; b_len = len_u;
        }

        for (int i = lane; i < a_len; i += 32) {
            int target = indices[a_start + i];
            float w_a = edge_weights[a_start + i];

            int rank = 0;
            for (int k = i - 1; k >= 0; k--) {
                if (indices[a_start + k] == target) rank++;
                else break;
            }

            int lo = 0, hi = b_len;
            while (lo < hi) { int mid = (lo + hi) >> 1; if (indices[b_start + mid] < target) lo = mid + 1; else hi = mid; }

            int pos = lo + rank;
            if (pos < b_len && indices[b_start + pos] == target) {
                float w_b = edge_weights[b_start + pos];
                dot += w_a * w_b;
                norm_a += w_a * w_a;
                norm_b += w_b * w_b;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
        norm_a += __shfl_down_sync(0xffffffff, norm_a, offset);
        norm_b += __shfl_down_sync(0xffffffff, norm_b, offset);
    }

    if (lane == 0) {
        float denom = sqrtf(norm_a) * sqrtf(norm_b);
        scores[pair_id] = dot / denom;
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
    (void)cache;

    if (num_pairs == 0) return;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    bool is_multigraph = graph.is_multigraph;

    if (is_multigraph) {
        int block = 256;
        int grid = (int)(((int64_t)num_pairs * 32 + block - 1) / block);
        cosine_similarity_multigraph_kernel<<<grid, block>>>(
            offsets, indices, edge_weights,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int64_t)num_pairs);
    } else {
        int avg_degree = (num_vertices > 0) ? (num_edges / num_vertices) : 0;
        if (avg_degree <= 4) {
            int block = 256;
            int grid = (int)((num_pairs + block - 1) / block);
            cosine_similarity_thread_kernel<<<grid, block>>>(
                offsets, indices, edge_weights,
                vertex_pairs_first, vertex_pairs_second,
                similarity_scores, (int64_t)num_pairs);
        } else if (avg_degree <= 24) {
            int block = 256;
            int grid = (int)(((int64_t)num_pairs * 16 + block - 1) / block);
            cosine_similarity_warp_kernel<16><<<grid, block>>>(
                offsets, indices, edge_weights,
                vertex_pairs_first, vertex_pairs_second,
                similarity_scores, (int64_t)num_pairs);
        } else {
            int block = 256;
            int grid = (int)(((int64_t)num_pairs * 32 + block - 1) / block);
            cosine_similarity_warp_kernel<32><<<grid, block>>>(
                offsets, indices, edge_weights,
                vertex_pairs_first, vertex_pairs_second,
                similarity_scores, (int64_t)num_pairs);
        }
    }
}

}  
