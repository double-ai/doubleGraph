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
    float* weight_sums = nullptr;
    int64_t weight_sums_capacity = 0;

    void ensure(int64_t num_vertices) {
        if (weight_sums_capacity < num_vertices) {
            if (weight_sums) cudaFree(weight_sums);
            cudaMalloc(&weight_sums, num_vertices * sizeof(float));
            weight_sums_capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (weight_sums) cudaFree(weight_sums);
    }
};

__device__ __forceinline__ int lower_bound_dev(
    const int* __restrict__ arr, int lo, int hi, int target) {
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void compute_weight_sums_kernel(
    const int* __restrict__ offsets,
    const float* __restrict__ weights,
    int num_vertices,
    float* __restrict__ weight_sums) {

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int start = offsets[v];
    int end = offsets[v + 1];
    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        sum += weights[i];
    }
    weight_sums[v] = sum;
}



__global__ void jaccard_4thread_simple(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ weight_sums,
    const int* __restrict__ pairs_first,
    const int* __restrict__ pairs_second,
    int64_t num_pairs,
    float* __restrict__ scores) {

    constexpr int GROUP_SIZE = 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_id = tid >> 2;  
    int group_lane = tid & 3; 

    if (pair_id >= num_pairs) return;

    int u = pairs_first[pair_id];
    int v = pairs_second[pair_id];

    if (u == v) {
        if (group_lane == 0) {
            int deg = offsets[u + 1] - offsets[u];
            scores[pair_id] = (deg > 0) ? 1.0f : 0.0f;
        }
        return;
    }

    int start_u = offsets[u];
    int end_u = offsets[u + 1];
    int start_v = offsets[v];
    int end_v = offsets[v + 1];

    int deg_u = end_u - start_u;
    int deg_v = end_v - start_v;

    if (deg_u == 0 || deg_v == 0) {
        if (group_lane == 0) scores[pair_id] = 0.0f;
        return;
    }

    int start_small, start_large, deg_small, deg_large;
    if (deg_u <= deg_v) {
        start_small = start_u; deg_small = deg_u;
        start_large = start_v; deg_large = deg_v;
    } else {
        start_small = start_v; deg_small = deg_v;
        start_large = start_u; deg_large = deg_u;
    }

    float local_intersection = 0.0f;

    
    
    int j = start_large;
    int end_large = start_large + deg_large;

    for (int i = group_lane; i < deg_small; i += GROUP_SIZE) {
        int target = indices[start_small + i];
        float w_small = weights[start_small + i];

        
        
        if (j < end_large) {
            int pos = lower_bound_dev(indices, j, end_large, target);

            if (pos < end_large && indices[pos] == target) {
                local_intersection += fminf(w_small, weights[pos]);
                j = pos + 1;
            } else {
                j = pos;
            }
        }
    }

    
    local_intersection += __shfl_down_sync(0xffffffff, local_intersection, 2, 4);
    local_intersection += __shfl_down_sync(0xffffffff, local_intersection, 1, 4);

    if (group_lane == 0) {
        float wu = weight_sums[u];
        float wv = weight_sums[v];
        float union_val = wu + wv - local_intersection;
        scores[pair_id] = (union_val > 0.0f) ? (local_intersection / union_val) : 0.0f;
    }
}


__global__ void jaccard_warp_multi(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ weight_sums,
    const int* __restrict__ pairs_first,
    const int* __restrict__ pairs_second,
    int64_t num_pairs,
    float* __restrict__ scores) {

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int u = pairs_first[warp_id];
    int v = pairs_second[warp_id];

    if (u == v) {
        if (lane == 0) {
            int deg = offsets[u + 1] - offsets[u];
            scores[warp_id] = (deg > 0) ? 1.0f : 0.0f;
        }
        return;
    }

    int start_u = offsets[u];
    int end_u = offsets[u + 1];
    int start_v = offsets[v];
    int end_v = offsets[v + 1];

    int deg_u = end_u - start_u;
    int deg_v = end_v - start_v;

    if (deg_u == 0 || deg_v == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    int start_small, start_large, deg_small, deg_large;
    if (deg_u <= deg_v) {
        start_small = start_u; deg_small = deg_u;
        start_large = start_v; deg_large = deg_v;
    } else {
        start_small = start_v; deg_small = deg_v;
        start_large = start_u; deg_large = deg_u;
    }

    float local_intersection = 0.0f;

    for (int i = lane; i < deg_small; i += 32) {
        int target = indices[start_small + i];
        float w_small = weights[start_small + i];

        int rank = i - lower_bound_dev(indices, start_small, start_small + i, target) + start_small;
        int pos = lower_bound_dev(indices, start_large, start_large + deg_large, target);
        pos += rank;

        if (pos < start_large + deg_large && indices[pos] == target) {
            local_intersection += fminf(w_small, weights[pos]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_intersection += __shfl_down_sync(0xffffffff, local_intersection, offset);
    }

    if (lane == 0) {
        float wu = weight_sums[u];
        float wv = weight_sums[v];
        float union_val = wu + wv - local_intersection;
        scores[warp_id] = (union_val > 0.0f) ? (local_intersection / union_val) : 0.0f;
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

    int32_t num_vertices = graph.number_of_vertices;
    cache.ensure(num_vertices);

    
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    compute_weight_sums_kernel<<<grid, block>>>(
        graph.offsets, edge_weights, num_vertices, cache.weight_sums);

    
    int64_t np = static_cast<int64_t>(num_pairs);

    if (graph.is_multigraph) {
        int warps_per_block = 8;
        int threads_per_block = warps_per_block * 32;
        int grid2 = static_cast<int>((np + warps_per_block - 1) / warps_per_block);
        jaccard_warp_multi<<<grid2, threads_per_block>>>(
            graph.offsets, graph.indices, edge_weights,
            cache.weight_sums, vertex_pairs_first, vertex_pairs_second,
            np, similarity_scores);
    } else {
        constexpr int GROUP_SIZE = 4;
        int threads_per_block = 256;
        int64_t total_threads = np * GROUP_SIZE;
        int grid2 = static_cast<int>((total_threads + threads_per_block - 1) / threads_per_block);
        jaccard_4thread_simple<<<grid2, threads_per_block>>>(
            graph.offsets, graph.indices, edge_weights,
            cache.weight_sums, vertex_pairs_first, vertex_pairs_second,
            np, similarity_scores);
    }
}

}  
