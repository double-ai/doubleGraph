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


__global__ void compute_weight_sums_thread(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ edge_weights,
    double* __restrict__ weight_sums,
    int32_t num_vertices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices) return;
    int start = offsets[idx];
    int end = offsets[idx + 1];
    double sum = 0.0;
    for (int i = start; i < end; i++)
        sum += edge_weights[i];
    weight_sums[idx] = sum;
}


__global__ void compute_weight_sums_warp(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ edge_weights,
    double* __restrict__ weight_sums,
    int32_t num_vertices
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_vertices) return;
    int start = offsets[warp_id];
    int end = offsets[warp_id + 1];
    double sum = 0.0;
    for (int i = start + lane; i < end; i += 32)
        sum += edge_weights[i];
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) weight_sums[warp_id] = sum;
}

template<int GROUP_SIZE>
__global__ void jaccard_subwarp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const double* __restrict__ weight_sums,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ similarity_scores,
    int64_t num_pairs
) {
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t pair_idx = global_thread / GROUP_SIZE;
    int lane = global_thread % GROUP_SIZE;

    double local_intersection = 0.0;
    double sum_union = 0.0;
    bool active = (pair_idx < num_pairs);
    bool compute = false;

    if (active) {
        int32_t u = first[pair_idx];
        int32_t v = second[pair_idx];

        int32_t u_start = offsets[u];
        int32_t u_end = offsets[u + 1];
        int32_t v_start = offsets[v];
        int32_t v_end = offsets[v + 1];
        int32_t u_deg = u_end - u_start;
        int32_t v_deg = v_end - v_start;

        sum_union = weight_sums[u] + weight_sums[v];

        if (sum_union > 0.0) {
            int32_t a_start, b_start, a_size, b_end;
            if (u_deg <= v_deg) {
                a_start = u_start; a_size = u_deg;
                b_start = v_start; b_end = v_end;
            } else {
                a_start = v_start; a_size = v_deg;
                b_start = u_start; b_end = u_end;
            }

            if (a_size > 0) {
                compute = true;

                for (int32_t ai = lane; ai < a_size; ai += GROUP_SIZE) {
                    int32_t a_idx = a_start + ai;
                    int32_t target = indices[a_idx];
                    double w_a = edge_weights[a_idx];

                    
                    int32_t rank = 0;
                    if (ai > 0 && indices[a_idx - 1] == target) {
                        int32_t lo2 = a_start, hi2 = a_idx;
                        while (lo2 < hi2) {
                            int32_t mid2 = lo2 + (hi2 - lo2) / 2;
                            if (indices[mid2] < target) lo2 = mid2 + 1;
                            else hi2 = mid2;
                        }
                        rank = ai - (lo2 - a_start);
                    }

                    
                    int32_t lo = b_start, hi = b_end;
                    while (lo < hi) {
                        int32_t mid = lo + (hi - lo) / 2;
                        if (indices[mid] < target) lo = mid + 1;
                        else hi = mid;
                    }

                    int32_t match_pos = lo + rank;
                    if (match_pos < b_end && indices[match_pos] == target) {
                        local_intersection += fmin(w_a, edge_weights[match_pos]);
                    }
                }
            }
        }
    }

    
    #pragma unroll
    for (int offset = GROUP_SIZE / 2; offset > 0; offset /= 2)
        local_intersection += __shfl_down_sync(0xffffffff, local_intersection, offset, GROUP_SIZE);

    if (active && lane == 0) {
        if (compute)
            similarity_scores[pair_idx] = local_intersection / (sum_union - local_intersection);
        else
            similarity_scores[pair_idx] = 0.0;
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
    int32_t num_edges = graph.number_of_edges;
    int32_t avg_degree = (num_vertices > 0) ? (num_edges / num_vertices) : 0;

    cache.ensure(num_vertices);

    
    int threads = 256;
    if (avg_degree <= 32) {
        int blocks = (num_vertices + threads - 1) / threads;
        compute_weight_sums_thread<<<blocks, threads>>>(
            graph.offsets, edge_weights, cache.weight_sums, num_vertices);
    } else {
        int warps_per_block = threads / 32;
        int blocks = (num_vertices + warps_per_block - 1) / warps_per_block;
        compute_weight_sums_warp<<<blocks, threads>>>(
            graph.offsets, edge_weights, cache.weight_sums, num_vertices);
    }

    
    int64_t np = static_cast<int64_t>(num_pairs);

    #define LAUNCH_KERNEL(GS) do { \
        int groups_per_block = threads / (GS); \
        int64_t blocks = (np + groups_per_block - 1) / groups_per_block; \
        jaccard_subwarp_kernel<GS><<<(int)blocks, threads>>>( \
            graph.offsets, graph.indices, edge_weights, cache.weight_sums, \
            vertex_pairs_first, vertex_pairs_second, similarity_scores, np); \
    } while(0)

    if (avg_degree <= 4) {
        LAUNCH_KERNEL(2);
    } else if (avg_degree <= 16) {
        LAUNCH_KERNEL(4);
    } else if (avg_degree <= 64) {
        LAUNCH_KERNEL(8);
    } else if (avg_degree <= 256) {
        LAUNCH_KERNEL(16);
    } else {
        LAUNCH_KERNEL(32);
    }

    #undef LAUNCH_KERNEL
}

}  
