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

static constexpr int64_t MAX_PAIRS = 1100000;

struct Cache : Cacheable {
    
    void* d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;

    
    double* weighted_degrees = nullptr;
    int64_t wd_capacity = 0;

    
    int32_t* sort_keys_in = nullptr;
    int32_t* sort_keys_out = nullptr;
    int32_t* sort_vals_in = nullptr;
    int32_t* sort_vals_out = nullptr;
    int64_t sort_keys_in_cap = 0;
    int64_t sort_keys_out_cap = 0;
    int64_t sort_vals_in_cap = 0;
    int64_t sort_vals_out_cap = 0;

    Cache() {
        
        sort_temp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, sort_temp_bytes,
            (int32_t*)nullptr, (int32_t*)nullptr,
            (int32_t*)nullptr, (int32_t*)nullptr,
            (int)MAX_PAIRS, 0, 32, (cudaStream_t)nullptr);
        cudaMalloc(&d_sort_temp, sort_temp_bytes);
    }

    void ensure_wd(int64_t n) {
        if (wd_capacity < n) {
            if (weighted_degrees) cudaFree(weighted_degrees);
            cudaMalloc(&weighted_degrees, n * sizeof(double));
            wd_capacity = n;
        }
    }

    void ensure_sort_buffers(int64_t n) {
        if (sort_keys_in_cap < n) {
            if (sort_keys_in) cudaFree(sort_keys_in);
            cudaMalloc(&sort_keys_in, n * sizeof(int32_t));
            sort_keys_in_cap = n;
        }
        if (sort_keys_out_cap < n) {
            if (sort_keys_out) cudaFree(sort_keys_out);
            cudaMalloc(&sort_keys_out, n * sizeof(int32_t));
            sort_keys_out_cap = n;
        }
        if (sort_vals_in_cap < n) {
            if (sort_vals_in) cudaFree(sort_vals_in);
            cudaMalloc(&sort_vals_in, n * sizeof(int32_t));
            sort_vals_in_cap = n;
        }
        if (sort_vals_out_cap < n) {
            if (sort_vals_out) cudaFree(sort_vals_out);
            cudaMalloc(&sort_vals_out, n * sizeof(int32_t));
            sort_vals_out_cap = n;
        }
    }

    ~Cache() override {
        if (d_sort_temp) cudaFree(d_sort_temp);
        if (weighted_degrees) cudaFree(weighted_degrees);
        if (sort_keys_in) cudaFree(sort_keys_in);
        if (sort_keys_out) cudaFree(sort_keys_out);
        if (sort_vals_in) cudaFree(sort_vals_in);
        if (sort_vals_out) cudaFree(sort_vals_out);
    }
};


__global__ void compute_weighted_degrees(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ edge_weights,
    double* __restrict__ weighted_degrees,
    int32_t num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;
    int start = offsets[tid];
    int end = offsets[tid + 1];
    double sum = 0.0;
    for (int i = start; i < end; i++) {
        sum += edge_weights[i];
    }
    weighted_degrees[tid] = sum;
}


__global__ void iota_kernel(int32_t* arr, int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) arr[tid] = (int32_t)tid;
}


__global__ void create_sort_keys(
    const int32_t* __restrict__ first,
    int32_t* __restrict__ keys,
    int64_t n
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) keys[tid] = first[tid];
}


__device__ __forceinline__ int galloping_lb(
    const int32_t* __restrict__ arr, int start, int end, int32_t target
) {
    if (start >= end || arr[start] >= target) return start;
    int pos = start;
    int step = 1;
    while (pos + step < end && arr[pos + step] < target) {
        pos += step;
        step <<= 1;
    }
    int lo = pos + 1;
    int hi = (pos + step < end) ? pos + step + 1 : end;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__global__ void sorensen_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const double* __restrict__ weighted_degrees,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ similarity_scores,
    const int32_t* __restrict__ perm,  
    int64_t num_pairs
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;

    
    int32_t orig_idx = perm ? perm[tid] : (int32_t)tid;

    int32_t u = first[orig_idx];
    int32_t v = second[orig_idx];

    double deg_u = weighted_degrees[u];
    double deg_v = weighted_degrees[v];
    double denom = deg_u + deg_v;

    if (denom == 0.0) {
        similarity_scores[orig_idx] = 0.0;
        return;
    }

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];
    int32_t u_len = u_end - u_start;
    int32_t v_len = v_end - v_start;

    if (u_len == 0 || v_len == 0) {
        similarity_scores[orig_idx] = 0.0;
        return;
    }

    
    int32_t a_start, a_end, b_start, b_end;
    if (u_len <= v_len) {
        a_start = u_start; a_end = u_end;
        b_start = v_start; b_end = v_end;
    } else {
        a_start = v_start; a_end = v_end;
        b_start = u_start; b_end = u_end;
    }

    
    int32_t a_first_val = indices[a_start];
    int32_t a_last_val = indices[a_end - 1];
    int32_t b_first_val = indices[b_start];
    int32_t b_last_val = indices[b_end - 1];

    if (a_last_val < b_first_val || b_last_val < a_first_val) {
        similarity_scores[orig_idx] = 0.0;
        return;
    }

    
    int a_lo = a_start;
    if (a_first_val < b_first_val) {
        a_lo = galloping_lb(indices, a_start, a_end, b_first_val);
        if (a_lo >= a_end) {
            similarity_scores[orig_idx] = 0.0;
            return;
        }
    }

    int j = b_start;
    if (b_first_val < indices[a_lo]) {
        j = galloping_lb(indices, b_start, b_end, indices[a_lo]);
    }

    
    double intersection_sum = 0.0;
    for (int i = a_lo; i < a_end && j < b_end; i++) {
        int32_t target = indices[i];
        if (target > b_last_val) break;

        j = galloping_lb(indices, j, b_end, target);
        if (j < b_end && indices[j] == target) {
            intersection_sum += fmin(edge_weights[i], edge_weights[j]);
            j++;
        }
    }

    similarity_scores[orig_idx] = 2.0 * intersection_sum / denom;
}

}  

void sorensen_similarity_seg(const graph32_t& graph,
                             const double* edge_weights,
                             const int32_t* vertex_pairs_first,
                             const int32_t* vertex_pairs_second,
                             std::size_t num_pairs,
                             double* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    cudaStream_t stream = nullptr;

    
    cache.ensure_wd(num_vertices);

    
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        compute_weighted_degrees<<<grid, block, 0, stream>>>(
            graph.offsets, edge_weights, cache.weighted_degrees, num_vertices);
    }

    const int32_t* perm_ptr = nullptr;

    
    if (num_pairs >= 10000) {
        int64_t np = static_cast<int64_t>(num_pairs);
        cache.ensure_sort_buffers(np);

        
        {
            int block = 256;
            int grid = static_cast<int>((np + block - 1) / block);
            create_sort_keys<<<grid, block, 0, stream>>>(
                vertex_pairs_first, cache.sort_keys_in, np);
            iota_kernel<<<grid, block, 0, stream>>>(cache.sort_vals_in, np);
        }

        
        void* sort_temp = cache.d_sort_temp;
        size_t temp_bytes = cache.sort_temp_bytes;
        void* dynamic_temp = nullptr;
        if (np > MAX_PAIRS) {
            temp_bytes = 0;
            cub::DeviceRadixSort::SortPairs(
                nullptr, temp_bytes,
                cache.sort_keys_in, cache.sort_keys_out,
                cache.sort_vals_in, cache.sort_vals_out,
                (int)np, 0, 32, stream);
            cudaMalloc(&dynamic_temp, temp_bytes);
            sort_temp = dynamic_temp;
        }

        
        cub::DeviceRadixSort::SortPairs(
            sort_temp, temp_bytes,
            cache.sort_keys_in, cache.sort_keys_out,
            cache.sort_vals_in, cache.sort_vals_out,
            (int)np, 0, 32, stream);

        if (dynamic_temp) cudaFree(dynamic_temp);

        perm_ptr = cache.sort_vals_out;
    }

    
    {
        int64_t np = static_cast<int64_t>(num_pairs);
        int block = 256;
        int grid = static_cast<int>((np + block - 1) / block);
        sorensen_kernel<<<grid, block, 0, stream>>>(
            graph.offsets, graph.indices,
            edge_weights, cache.weighted_degrees,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, perm_ptr, np);
    }
}

}  
