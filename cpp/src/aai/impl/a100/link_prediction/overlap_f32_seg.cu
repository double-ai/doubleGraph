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
    float* strengths = nullptr;
    int64_t strengths_capacity = 0;

    void ensure(int32_t num_vertices) {
        if (strengths_capacity < num_vertices) {
            if (strengths) cudaFree(strengths);
            cudaMalloc(&strengths, (int64_t)num_vertices * sizeof(float));
            strengths_capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (strengths) cudaFree(strengths);
    }
};


__global__ void compute_strengths_kernel(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ edge_weights,
    float* __restrict__ strengths,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);

    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        sum += __ldg(&edge_weights[i]);
    }
    strengths[v] = sum;
}


__device__ __forceinline__ int lower_bound_range(
    const int32_t* arr, int lo, int hi, int32_t target
) {
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(&arr[mid]) < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__global__ void overlap_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ strengths,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int32_t u = __ldg(&pairs_first[warp_id]);
    int32_t v = __ldg(&pairs_second[warp_id]);

    int u_start = __ldg(&offsets[u]);
    int u_end = __ldg(&offsets[u + 1]);
    int v_start = __ldg(&offsets[v]);
    int v_end = __ldg(&offsets[v + 1]);

    int m = u_end - u_start;
    int n = v_end - v_start;

    float str_u = __ldg(&strengths[u]);
    float str_v = __ldg(&strengths[v]);
    float denom = fminf(str_u, str_v);

    if (denom == 0.0f || m == 0 || n == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    const int32_t* A, *B;
    const float* wA, *wB;
    int len_A, len_B;

    if (m <= n) {
        A = indices + u_start; wA = edge_weights + u_start; len_A = m;
        B = indices + v_start; wB = edge_weights + v_start; len_B = n;
    } else {
        A = indices + v_start; wA = edge_weights + v_start; len_A = n;
        B = indices + u_start; wB = edge_weights + u_start; len_B = m;
    }

    
    int32_t a_max = __ldg(&A[len_A - 1]);
    int32_t b_min = __ldg(&B[0]);
    int32_t a_min = __ldg(&A[0]);
    int32_t b_max = __ldg(&B[len_B - 1]);

    if (a_max < b_min || b_max < a_min) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    float local_sum = 0.0f;

    
    
    
    int b_search_lo = 0;

    for (int k = lane; k < len_A; k += 32) {
        int32_t val = __ldg(&A[k]);

        
        int pos = lower_bound_range(B, b_search_lo, len_B, val);
        b_search_lo = pos;  

        if (pos < len_B && __ldg(&B[pos]) == val) {
            
            int rank = 0;
            if (k > 0 && __ldg(&A[k - 1]) == val) {
                
                int first_occ = lower_bound_range(A, 0, k, val);
                rank = k - first_occ;
            }

            int match_pos = pos + rank;
            if (match_pos < len_B && __ldg(&B[match_pos]) == val) {
                local_sum += fminf(__ldg(&wA[k]), __ldg(&wB[match_pos]));
            }
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) {
        scores[warp_id] = local_sum / denom;
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

    int32_t num_vertices = graph.number_of_vertices;
    cache.ensure(num_vertices);

    cudaStream_t stream = 0;

    
    if (num_vertices > 0) {
        int threads = 256;
        int blocks = (num_vertices + threads - 1) / threads;
        compute_strengths_kernel<<<blocks, threads, 0, stream>>>(
            graph.offsets, edge_weights, cache.strengths, num_vertices);
    }

    
    if (num_pairs > 0) {
        int threads = 256;  
        int warps_per_block = threads / 32;
        int64_t blocks = ((int64_t)num_pairs + warps_per_block - 1) / warps_per_block;
        overlap_warp_kernel<<<(int)blocks, threads, 0, stream>>>(
            graph.offsets, graph.indices, edge_weights, cache.strengths,
            vertex_pairs_first, vertex_pairs_second, similarity_scores,
            (int64_t)num_pairs);
    }
}

}  
