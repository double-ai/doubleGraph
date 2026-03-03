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
#include <cub/cub.cuh>

namespace aai {

namespace {

struct Cache : Cacheable {
    double* weighted_degrees = nullptr;
    int64_t wd_capacity = 0;

    void ensure_wd(int64_t size) {
        if (wd_capacity < size) {
            if (weighted_degrees) cudaFree(weighted_degrees);
            cudaMalloc(&weighted_degrees, size * sizeof(double));
            wd_capacity = size;
        }
    }

    ~Cache() override {
        if (weighted_degrees) cudaFree(weighted_degrees);
    }
};



__global__ void compute_weighted_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ edge_weights,
    double* __restrict__ weighted_degrees,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int start = offsets[v];
    int end = offsets[v + 1];
    double sum = 0.0;
    for (int i = start; i < end; i++) {
        sum += edge_weights[i];
    }
    weighted_degrees[v] = sum;
}




__device__ __forceinline__ int gallop_search(
    const int32_t* __restrict__ arr, int start, int end, int32_t target
) {
    if (start >= end || arr[start] >= target) return start;

    
    int pos = start;
    int step = 1;
    while (pos + step < end && arr[pos + step] < target) {
        pos += step;
        step <<= 1;
    }

    
    int lo = pos;
    int hi = (pos + step < end) ? pos + step + 1 : end;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__device__ double weighted_intersection_adaptive(
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    int u_start, int u_end,
    int v_start, int v_end
) {
    int deg_u = u_end - u_start;
    int deg_v = v_end - v_start;

    if (deg_u == 0 || deg_v == 0) return 0.0;

    
    int32_t u_first = indices[u_start];
    int32_t u_last = indices[u_end - 1];
    int32_t v_first = indices[v_start];
    int32_t v_last = indices[v_end - 1];

    if (u_last < v_first || v_last < u_first) return 0.0;

    
    int s_start, s_end, l_start, l_end;
    if (deg_u <= deg_v) {
        s_start = u_start; s_end = u_end;
        l_start = v_start; l_end = v_end;
    } else {
        s_start = v_start; s_end = v_end;
        l_start = u_start; l_end = u_end;
    }

    int small_size = s_end - s_start;
    int large_size = l_end - l_start;

    double sum = 0.0;

    
    
    if (large_size > 8 * small_size) {
        
        int j = l_start;
        for (int i = s_start; i < s_end && j < l_end; i++) {
            int32_t target = indices[i];
            j = gallop_search(indices, j, l_end, target);
            if (j < l_end && indices[j] == target) {
                sum += fmin(edge_weights[i], edge_weights[j]);
                j++;
            }
        }
    } else {
        
        int i = s_start, j = l_start;

        
        int32_t sf = indices[s_start];
        int32_t lf = indices[l_start];

        if (sf < lf) {
            int lo = s_start, hi = s_end;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (indices[mid] < lf) lo = mid + 1;
                else hi = mid;
            }
            i = lo;
        } else if (lf < sf) {
            int lo = l_start, hi = l_end;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (indices[mid] < sf) lo = mid + 1;
                else hi = mid;
            }
            j = lo;
        }

        while (i < s_end && j < l_end) {
            int32_t a = indices[i];
            int32_t b = indices[j];
            if (a == b) {
                sum += fmin(edge_weights[i], edge_weights[j]);
                i++; j++;
            } else if (a < b) {
                i++;
            } else {
                j++;
            }
        }
    }

    return sum;
}


__global__ void sorensen_precomputed_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const double* __restrict__ weighted_degrees,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    double* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int32_t u = pairs_first[idx];
    int32_t v = pairs_second[idx];

    double wd_u = weighted_degrees[u];
    double wd_v = weighted_degrees[v];
    double denom = wd_u + wd_v;

    if (denom == 0.0) {
        scores[idx] = 0.0;
        return;
    }

    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int v_start = offsets[v];
    int v_end = offsets[v + 1];

    double isect = weighted_intersection_adaptive(indices, edge_weights,
                                                   u_start, u_end, v_start, v_end);
    scores[idx] = 2.0 * isect / denom;
}


__global__ void sorensen_fused_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    double* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int32_t u = pairs_first[idx];
    int32_t v = pairs_second[idx];

    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int v_start = offsets[v];
    int v_end = offsets[v + 1];

    if (u_end == u_start || v_end == v_start) {
        scores[idx] = 0.0;
        return;
    }

    
    double wd_u = 0.0;
    for (int i = u_start; i < u_end; i++) wd_u += edge_weights[i];
    double wd_v = 0.0;
    for (int i = v_start; i < v_end; i++) wd_v += edge_weights[i];

    double denom = wd_u + wd_v;
    if (denom == 0.0) {
        scores[idx] = 0.0;
        return;
    }

    double isect = weighted_intersection_adaptive(indices, edge_weights,
                                                   u_start, u_end, v_start, v_end);
    scores[idx] = 2.0 * isect / denom;
}




__global__ void compute_pair_keys_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    int32_t* __restrict__ keys,
    int32_t* __restrict__ indices_out,
    int64_t num_pairs
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int32_t u = pairs_first[idx];
    int32_t v = pairs_second[idx];
    int deg_u = offsets[u + 1] - offsets[u];
    int deg_v = offsets[v + 1] - offsets[v];

    
    keys[idx] = deg_u + deg_v;
    indices_out[idx] = (int32_t)idx;
}


__global__ void scatter_results_kernel(
    const double* __restrict__ sorted_scores,
    const int32_t* __restrict__ sort_indices,
    double* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    scores[sort_indices[idx]] = sorted_scores[idx];
}

}  

void sorensen_similarity(const graph32_t& graph,
                         const double* edge_weights,
                         const int32_t* vertex_pairs_first,
                         const int32_t* vertex_pairs_second,
                         std::size_t num_pairs,
                         double* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;

    cudaStream_t stream = nullptr;

    bool use_fused = (int64_t)num_vertices > 2 * (int64_t)num_pairs;

    if (use_fused) {
        int block = 256;
        int grid = (int)(((int64_t)num_pairs + block - 1) / block);
        sorensen_fused_kernel<<<grid, block, 0, stream>>>(
            offsets, indices, edge_weights,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int64_t)num_pairs);
    } else {
        cache.ensure_wd((int64_t)num_vertices);

        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        compute_weighted_degrees_kernel<<<grid, block, 0, stream>>>(
            offsets, edge_weights, cache.weighted_degrees, num_vertices);

        int grid2 = (int)(((int64_t)num_pairs + block - 1) / block);
        sorensen_precomputed_kernel<<<grid2, block, 0, stream>>>(
            offsets, indices, edge_weights, cache.weighted_degrees,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int64_t)num_pairs);
    }
}

}  
