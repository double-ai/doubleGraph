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

#define BLOCK_SIZE 256

__device__ __forceinline__ int lower_bound_dev(const int32_t* __restrict__ arr, int lo, int hi, int32_t target) {
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


template <int SW, int MAX_DEG>
__global__ void __launch_bounds__(BLOCK_SIZE)
jaccard_kernel_smem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ similarity_scores,
    int64_t num_pairs,
    const bool* __restrict__ d_is_multigraph
) {
    constexpr int SUBWARPS_PER_BLOCK = BLOCK_SIZE / SW;
    __shared__ int32_t smem_idx[SUBWARPS_PER_BLOCK * MAX_DEG];

    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int pair_idx = global_tid / SW;
    const int lane = global_tid % SW;
    const int subwarp_id = threadIdx.x / SW;

    const bool valid = pair_idx < num_pairs;
    const int safe_pair = valid ? pair_idx : 0;
    const bool is_multigraph = *d_is_multigraph;

    const int32_t u = first[safe_pair];
    const int32_t v = second[safe_pair];

    const int32_t u_start = offsets[u];
    const int32_t u_end = offsets[u + 1];
    const int32_t v_start = offsets[v];
    const int32_t v_end = offsets[v + 1];
    const int32_t u_deg = u_end - u_start;
    const int32_t v_deg = v_end - v_start;

    
    double w_sum_u = 0.0;
    for (int i = lane; i < u_deg; i += SW) w_sum_u += edge_weights[u_start + i];
    double w_sum_v = 0.0;
    for (int i = lane; i < v_deg; i += SW) w_sum_v += edge_weights[v_start + i];

    #pragma unroll
    for (int offset = SW / 2; offset > 0; offset >>= 1) {
        w_sum_u += __shfl_down_sync(0xffffffff, w_sum_u, offset, SW);
        w_sum_v += __shfl_down_sync(0xffffffff, w_sum_v, offset, SW);
    }

    const int32_t* small_idx; const double* small_wt; int32_t small_deg;
    const int32_t* large_idx; const double* large_wt; int32_t large_deg;

    if (u_deg <= v_deg) {
        small_idx = indices + u_start; small_wt = edge_weights + u_start; small_deg = u_deg;
        large_idx = indices + v_start; large_wt = edge_weights + v_start; large_deg = v_deg;
    } else {
        small_idx = indices + v_start; small_wt = edge_weights + v_start; small_deg = v_deg;
        large_idx = indices + u_start; large_wt = edge_weights + u_start; large_deg = u_deg;
    }

    double intersection_weight = 0.0;
    int32_t* my_smem = smem_idx + subwarp_id * MAX_DEG;
    const bool use_smem = (large_deg <= MAX_DEG);

    if (use_smem) {
        for (int i = lane; i < large_deg; i += SW) my_smem[i] = large_idx[i];
    }
    __syncwarp();

    if (is_multigraph) {
        if (use_smem) {
            for (int i = lane; i < small_deg; i += SW) {
                int32_t target = small_idx[i];
                double w_small = small_wt[i];
                int fs = lower_bound_dev(small_idx, 0, small_deg, target);
                int fl = lower_bound_dev(my_smem, 0, large_deg, target);
                int mp = fl + i - fs;
                if (mp < large_deg && my_smem[mp] == target)
                    intersection_weight += fmin(w_small, large_wt[mp]);
            }
        } else {
            for (int i = lane; i < small_deg; i += SW) {
                int32_t target = small_idx[i];
                double w_small = small_wt[i];
                int fs = lower_bound_dev(small_idx, 0, small_deg, target);
                int fl = lower_bound_dev(large_idx, 0, large_deg, target);
                int mp = fl + i - fs;
                if (mp < large_deg && large_idx[mp] == target)
                    intersection_weight += fmin(w_small, large_wt[mp]);
            }
        }
    } else {
        if (use_smem) {
            int cursor = 0;
            for (int i = lane; i < small_deg; i += SW) {
                int32_t target = small_idx[i];
                double w_small = small_wt[i];
                cursor = lower_bound_dev(my_smem, cursor, large_deg, target);
                if (cursor < large_deg && my_smem[cursor] == target) {
                    intersection_weight += fmin(w_small, large_wt[cursor]);
                    cursor++;
                }
            }
        } else {
            int cursor = 0;
            for (int i = lane; i < small_deg; i += SW) {
                int32_t target = small_idx[i];
                double w_small = small_wt[i];
                cursor = lower_bound_dev(large_idx, cursor, large_deg, target);
                if (cursor < large_deg && large_idx[cursor] == target) {
                    intersection_weight += fmin(w_small, large_wt[cursor]);
                    cursor++;
                }
            }
        }
    }

    #pragma unroll
    for (int offset = SW / 2; offset > 0; offset >>= 1)
        intersection_weight += __shfl_down_sync(0xffffffff, intersection_weight, offset, SW);

    if (valid && lane == 0) {
        double union_weight = w_sum_u + w_sum_v - intersection_weight;
        similarity_scores[pair_idx] = (union_weight > 0.0) ? (intersection_weight / union_weight) : 0.0;
    }
}


void launch_jaccard_sw2(
    const int32_t* offsets, const int32_t* indices, const double* edge_weights,
    const int32_t* first, const int32_t* second, double* similarity_scores,
    int64_t num_pairs, const bool* d_is_multigraph, cudaStream_t stream
) {
    if (num_pairs == 0) return;
    int grid = (int)((num_pairs * 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    jaccard_kernel_smem<2, 32><<<grid, BLOCK_SIZE, 0, stream>>>(
        offsets, indices, edge_weights, first, second, similarity_scores, num_pairs, d_is_multigraph);
}


void launch_jaccard_sw4(
    const int32_t* offsets, const int32_t* indices, const double* edge_weights,
    const int32_t* first, const int32_t* second, double* similarity_scores,
    int64_t num_pairs, const bool* d_is_multigraph, cudaStream_t stream
) {
    if (num_pairs == 0) return;
    int grid = (int)((num_pairs * 4 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    jaccard_kernel_smem<4, 48><<<grid, BLOCK_SIZE, 0, stream>>>(
        offsets, indices, edge_weights, first, second, similarity_scores, num_pairs, d_is_multigraph);
}


void launch_jaccard_sw8(
    const int32_t* offsets, const int32_t* indices, const double* edge_weights,
    const int32_t* first, const int32_t* second, double* similarity_scores,
    int64_t num_pairs, const bool* d_is_multigraph, cudaStream_t stream
) {
    if (num_pairs == 0) return;
    int grid = (int)((num_pairs * 8 + BLOCK_SIZE - 1) / BLOCK_SIZE);
    jaccard_kernel_smem<8, 96><<<grid, BLOCK_SIZE, 0, stream>>>(
        offsets, indices, edge_weights, first, second, similarity_scores, num_pairs, d_is_multigraph);
}

}  

void jaccard_similarity(const graph32_t& graph,
                        const double* edge_weights,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        double* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int64_t nv = graph.number_of_vertices;
    int64_t ne = graph.number_of_edges;

    
    bool is_mg = graph.is_multigraph;
    cudaMemcpy(cache.d_is_multigraph, &is_mg, sizeof(bool), cudaMemcpyHostToDevice);

    
    double avg_degree = (nv > 0) ? (double)ne / nv : 0;

    typedef void (*LaunchFn)(const int32_t*, const int32_t*, const double*,
                             const int32_t*, const int32_t*, double*,
                             int64_t, const bool*, cudaStream_t);

    LaunchFn launch;
    if (avg_degree <= 4) launch = launch_jaccard_sw2;
    else if (avg_degree <= 32) launch = launch_jaccard_sw4;
    else launch = launch_jaccard_sw8;

    launch(offsets, indices, edge_weights,
           vertex_pairs_first, vertex_pairs_second,
           similarity_scores, (int64_t)num_pairs,
           cache.d_is_multigraph, 0);
}

}  
