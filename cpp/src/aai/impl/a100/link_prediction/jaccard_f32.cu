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
#include <cstddef>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* weight_sums = nullptr;
    int64_t weight_sums_capacity = 0;

    void ensure(int32_t num_vertices) {
        int64_t needed = static_cast<int64_t>(num_vertices);
        if (weight_sums_capacity < needed) {
            if (weight_sums) cudaFree(weight_sums);
            cudaMalloc(&weight_sums, needed * sizeof(float));
            weight_sums_capacity = needed;
        }
    }

    ~Cache() override {
        if (weight_sums) cudaFree(weight_sums);
    }
};


__global__ void compute_weight_sums_kernel(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ weights,
    float* __restrict__ weight_sums,
    int num_vertices
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_vertices) return;

    int start = offsets[warp_id];
    int end = offsets[warp_id + 1];

    float sum = 0.0f;
    for (int e = start + lane; e < end; e += 32) {
        sum += weights[e];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        weight_sums[warp_id] = sum;
    }
}


__global__ void jaccard_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ weight_sums,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int num_pairs
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int u = first[warp_id];
    int v = second[warp_id];

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

    
    const int32_t* small_idx;
    const float* small_wt;
    int small_size;
    const int32_t* large_idx;
    const float* large_wt;
    int large_size;

    if (deg_u <= deg_v) {
        small_idx = indices + start_u;
        small_wt = weights + start_u;
        small_size = deg_u;
        large_idx = indices + start_v;
        large_wt = weights + start_v;
        large_size = deg_v;
    } else {
        small_idx = indices + start_v;
        small_wt = weights + start_v;
        small_size = deg_v;
        large_idx = indices + start_u;
        large_wt = weights + start_u;
        large_size = deg_u;
    }

    float local_intersection = 0.0f;

    for (int k = lane; k < small_size; k += 32) {
        int key = small_idx[k];
        float w_small = small_wt[k];

        
        
        int pos_in_run = 0;
        for (int j = k - 1; j >= 0 && small_idx[j] == key; j--) {
            pos_in_run++;
        }

        
        int lo = 0, hi = large_size;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (large_idx[mid] < key) lo = mid + 1;
            else hi = mid;
        }

        
        int target = lo + pos_in_run;
        if (target < large_size && large_idx[target] == key) {
            local_intersection += fminf(w_small, large_wt[target]);
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_intersection += __shfl_down_sync(0xffffffff, local_intersection, offset);
    }

    if (lane == 0) {
        float sum_u = weight_sums[u];
        float sum_v = weight_sums[v];
        float union_val = sum_u + sum_v - local_intersection;
        scores[warp_id] = (union_val > 0.0f) ? (local_intersection / union_val) : 0.0f;
    }
}


__global__ void jaccard_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ weight_sums,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int num_pairs
) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= num_pairs) return;

    int u = first[pair_idx];
    int v = second[pair_idx];

    int start_u = offsets[u];
    int end_u = offsets[u + 1];
    int start_v = offsets[v];
    int end_v = offsets[v + 1];

    int deg_u = end_u - start_u;
    int deg_v = end_v - start_v;

    if (deg_u == 0 || deg_v == 0) {
        scores[pair_idx] = 0.0f;
        return;
    }

    float sum_u = weight_sums[u];
    float sum_v = weight_sums[v];

    
    int first_u = indices[start_u];
    int last_u = indices[end_u - 1];
    int first_v = indices[start_v];
    int last_v = indices[end_v - 1];

    if (last_u < first_v || last_v < first_u) {
        scores[pair_idx] = 0.0f;
        return;
    }

    
    float intersection = 0.0f;
    int i = start_u, j = start_v;

    while (i < end_u && j < end_v) {
        int ni = indices[i];
        int nj = indices[j];
        if (ni == nj) {
            intersection += fminf(weights[i], weights[j]);
            i++; j++;
        } else if (ni < nj) {
            i++;
        } else {
            j++;
        }
    }

    float union_val = sum_u + sum_v - intersection;
    scores[pair_idx] = (union_val > 0.0f) ? (intersection / union_val) : 0.0f;
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

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    
    cache.ensure(num_vertices);

    cudaStream_t stream = 0;

    
    {
        int threads = 256;
        int warps_per_block = threads / 32;
        int grid = (num_vertices + warps_per_block - 1) / warps_per_block;
        compute_weight_sums_kernel<<<grid, threads, 0, stream>>>(
            offsets, edge_weights, cache.weight_sums, num_vertices);
    }

    
    {
        int threads = 256;
        int warps_per_block = threads / 32;
        int grid = (static_cast<int>(num_pairs) + warps_per_block - 1) / warps_per_block;
        jaccard_warp_kernel<<<grid, threads, 0, stream>>>(
            offsets, indices, edge_weights, cache.weight_sums,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, static_cast<int>(num_pairs));
    }
}

}  
