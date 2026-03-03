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
    double* weight_sums = nullptr;
    int64_t weight_sums_capacity = 0;

    void ensure(int64_t num_vertices) {
        if (weight_sums_capacity < num_vertices) {
            if (weight_sums) cudaFree(weight_sums);
            cudaMalloc(&weight_sums, num_vertices * sizeof(double));
            weight_sums_capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (weight_sums) cudaFree(weight_sums);
    }
};


__global__ void compute_weight_sums_kernel(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ edge_weights,
    double* __restrict__ weight_sums,
    int32_t num_vertices)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_vertices) return;

    int32_t start = offsets[warp_id];
    int32_t end = offsets[warp_id + 1];

    double sum = 0.0;
    for (int32_t i = start + lane; i < end; i += 32)
        sum += edge_weights[i];

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) weight_sums[warp_id] = sum;
}


__device__ __forceinline__ int32_t lower_bound_dev(
    const int32_t* __restrict__ arr, int32_t size, int32_t target)
{
    int32_t lo = 0, hi = size;
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (__ldg(&arr[mid]) < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}




template<bool IS_MULTIGRAPH>
__global__ void jaccard_subwarp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const double* __restrict__ weight_sums,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ scores,
    int64_t num_pairs)
{
    
    constexpr int SUBWARP = 8;
    constexpr int PAIRS_PER_WARP = 32 / SUBWARP;

    int lane = threadIdx.x & 31;
    int sub_lane = lane & (SUBWARP - 1);  
    int sub_id = lane / SUBWARP;          
    int64_t warp_id = (int64_t)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    int64_t pair_idx = warp_id * PAIRS_PER_WARP + sub_id;

    if (pair_idx >= num_pairs) return;

    int32_t u = __ldg(&first[pair_idx]);
    int32_t v = __ldg(&second[pair_idx]);

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);
    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    double ws_u = __ldg(&weight_sums[u]);
    double ws_v = __ldg(&weight_sums[v]);

    
    const int32_t* u_idx;
    const int32_t* v_idx;
    const double* u_wt;
    const double* v_wt;
    int32_t short_deg, long_deg;

    if (u_deg <= v_deg) {
        u_idx = indices + u_start;
        v_idx = indices + v_start;
        u_wt = edge_weights + u_start;
        v_wt = edge_weights + v_start;
        short_deg = u_deg;
        long_deg = v_deg;
    } else {
        u_idx = indices + v_start;
        v_idx = indices + u_start;
        u_wt = edge_weights + v_start;
        v_wt = edge_weights + u_start;
        short_deg = v_deg;
        long_deg = u_deg;
    }

    double local_intersection = 0.0;

    if (short_deg > 0 && long_deg > 0) {
        
        int32_t short_min = __ldg(&u_idx[0]);
        int32_t short_max = __ldg(&u_idx[short_deg - 1]);
        int32_t long_min = __ldg(&v_idx[0]);
        int32_t long_max = __ldg(&v_idx[long_deg - 1]);

        if (short_max >= long_min && long_max >= short_min) {
            
            int32_t search_lo = lower_bound_dev(v_idx, long_deg, short_min);
            int32_t search_len = long_deg - search_lo;
            const int32_t* v_search = v_idx + search_lo;
            const double* v_search_wt = v_wt + search_lo;

            for (int32_t k = sub_lane; k < short_deg; k += SUBWARP) {
                int32_t target = __ldg(&u_idx[k]);

                if (target >= long_min && target <= long_max) {
                    int32_t rank = 0;
                    if constexpr (IS_MULTIGRAPH) {
                        int32_t lb_u = lower_bound_dev(u_idx, short_deg, target);
                        rank = k - lb_u;
                    }

                    int32_t pos = lower_bound_dev(v_search, search_len, target) + rank;

                    if (pos < search_len && __ldg(&v_search[pos]) == target) {
                        double w_u = __ldg(&u_wt[k]);
                        double w_v = __ldg(&v_search_wt[pos]);
                        local_intersection += fmin(w_u, w_v);
                    }
                }
            }
        }
    }

    
    #pragma unroll
    for (int offset = SUBWARP/2; offset > 0; offset >>= 1)
        local_intersection += __shfl_down_sync(0xffffffff, local_intersection, offset, SUBWARP);

    if (sub_lane == 0) {
        double union_val = ws_u + ws_v - local_intersection;
        scores[pair_idx] = (union_val > 0.0) ? local_intersection / union_val : 0.0;
    }
}

}  

void jaccard_similarity_seg(const graph32_t& graph,
                            const double* edge_weights,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            double* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;

    cache.ensure(num_vertices);

    cudaStream_t stream = 0;

    
    {
        int threads = 256;
        int blocks = (num_vertices + (threads / 32) - 1) / (threads / 32);
        compute_weight_sums_kernel<<<blocks, threads, 0, stream>>>(
            graph.offsets, edge_weights, cache.weight_sums, num_vertices);
    }

    
    {
        constexpr int SUBWARP = 8;
        constexpr int PAIRS_PER_WARP = 32 / SUBWARP;
        int threads = 256;
        int warps_per_block = threads / 32;
        int pairs_per_block = warps_per_block * PAIRS_PER_WARP;
        int64_t blocks = ((int64_t)num_pairs + pairs_per_block - 1) / pairs_per_block;

        if (graph.is_multigraph) {
            jaccard_subwarp_kernel<true><<<(int)blocks, threads, 0, stream>>>(
                graph.offsets, graph.indices, edge_weights, cache.weight_sums,
                vertex_pairs_first, vertex_pairs_second, similarity_scores,
                (int64_t)num_pairs);
        } else {
            jaccard_subwarp_kernel<false><<<(int)blocks, threads, 0, stream>>>(
                graph.offsets, graph.indices, edge_weights, cache.weight_sums,
                vertex_pairs_first, vertex_pairs_second, similarity_scores,
                (int64_t)num_pairs);
        }
    }
}

}  
