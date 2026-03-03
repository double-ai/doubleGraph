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
#include <math_constants.h>

namespace aai {

namespace {


#define MAX_SMEM_PER_WARP 256

struct Cache : Cacheable {};








template<bool IS_MULTIGRAPH>
__global__ __launch_bounds__(256, 6)
void cosine_similarity_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int* __restrict__ first,
    const int* __restrict__ second,
    double* __restrict__ scores,
    int num_pairs
) {
    const int WARP_SIZE = 32;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x / WARP_SIZE;

    
    extern __shared__ int smem[];
    int* my_cache = smem + warp_in_block * MAX_SMEM_PER_WARP;

    if (warp_id >= num_pairs) return;

    int u = __ldg(&first[warp_id]);
    int v = __ldg(&second[warp_id]);

    int u_start = __ldg(&offsets[u]);
    int u_end = __ldg(&offsets[u + 1]);
    int v_start = __ldg(&offsets[v]);
    int v_end = __ldg(&offsets[v + 1]);

    int deg_u = u_end - u_start;
    int deg_v = v_end - v_start;

    
    if (deg_u == 0 || deg_v == 0) {
        if (lane == 0) {
            scores[warp_id] = __longlong_as_double(0x7FF8000000000000ULL);
        }
        return;
    }

    
    int s_start, l_start, s_size, l_size;
    if (deg_u <= deg_v) {
        s_start = u_start; l_start = v_start;
        s_size = deg_u; l_size = deg_v;
    } else {
        s_start = v_start; l_start = u_start;
        s_size = deg_v; l_size = deg_u;
    }

    double dot = 0.0, ns_sq = 0.0, nl_sq = 0.0;

    
    if (l_size <= MAX_SMEM_PER_WARP) {
        
        for (int i = lane; i < l_size; i += WARP_SIZE) {
            my_cache[i] = __ldg(&indices[l_start + i]);
        }
        __syncwarp();

        
        for (int j = lane; j < s_size; j += WARP_SIZE) {
            int target = __ldg(&indices[s_start + j]);

            int lo = 0, hi = l_size;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (my_cache[mid] < target)
                    lo = mid + 1;
                else
                    hi = mid;
            }

            if constexpr (IS_MULTIGRAPH) {
                int first_occ;
                {
                    int lo2 = 0, hi2 = j;
                    while (lo2 < hi2) {
                        int mid2 = (lo2 + hi2) >> 1;
                        if (__ldg(&indices[s_start + mid2]) < target)
                            lo2 = mid2 + 1;
                        else
                            hi2 = mid2;
                    }
                    first_occ = lo2;
                }
                lo += (j - first_occ);
            }

            if (lo < l_size && my_cache[lo] == target) {
                double ws = __ldg(&edge_weights[s_start + j]);
                double wl = __ldg(&edge_weights[l_start + lo]);
                dot += ws * wl;
                ns_sq += ws * ws;
                nl_sq += wl * wl;
            }
        }
    } else {
        
        for (int j = lane; j < s_size; j += WARP_SIZE) {
            int target = __ldg(&indices[s_start + j]);

            int lo = 0, hi = l_size;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (__ldg(&indices[l_start + mid]) < target)
                    lo = mid + 1;
                else
                    hi = mid;
            }

            if constexpr (IS_MULTIGRAPH) {
                int first_occ;
                {
                    int lo2 = 0, hi2 = j;
                    while (lo2 < hi2) {
                        int mid2 = (lo2 + hi2) >> 1;
                        if (__ldg(&indices[s_start + mid2]) < target)
                            lo2 = mid2 + 1;
                        else
                            hi2 = mid2;
                    }
                    first_occ = lo2;
                }
                lo += (j - first_occ);
            }

            if (lo < l_size && __ldg(&indices[l_start + lo]) == target) {
                double ws = __ldg(&edge_weights[s_start + j]);
                double wl = __ldg(&edge_weights[l_start + lo]);
                dot += ws * wl;
                ns_sq += ws * ws;
                nl_sq += wl * wl;
            }
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
        ns_sq += __shfl_down_sync(0xffffffff, ns_sq, offset);
        nl_sq += __shfl_down_sync(0xffffffff, nl_sq, offset);
    }

    if (lane == 0) {
        double denom = sqrt(ns_sq) * sqrt(nl_sq);
        scores[warp_id] = dot / denom;
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

    if (num_pairs == 0) return;

    const int BLOCK_SIZE = 256;
    const int warps_per_block = BLOCK_SIZE / 32;
    int grid = (static_cast<int>(num_pairs) + warps_per_block - 1) / warps_per_block;

    
    size_t smem_size = warps_per_block * MAX_SMEM_PER_WARP * sizeof(int);

    if (graph.is_multigraph) {
        cosine_similarity_kernel<true><<<grid, BLOCK_SIZE, smem_size>>>(
            graph.offsets, graph.indices, edge_weights,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, static_cast<int>(num_pairs));
    } else {
        cosine_similarity_kernel<false><<<grid, BLOCK_SIZE, smem_size>>>(
            graph.offsets, graph.indices, edge_weights,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, static_cast<int>(num_pairs));
    }
}

}  
