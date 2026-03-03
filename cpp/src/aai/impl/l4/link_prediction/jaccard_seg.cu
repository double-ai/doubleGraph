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
    int32_t* keys_in = nullptr;
    int32_t* idx_in = nullptr;
    int32_t* keys_out = nullptr;
    int32_t* idx_out = nullptr;
    void* sort_temp = nullptr;

    int64_t keys_in_capacity = 0;
    int64_t idx_in_capacity = 0;
    int64_t keys_out_capacity = 0;
    int64_t idx_out_capacity = 0;
    size_t sort_temp_bytes = 0;

    void ensure_pairs(int64_t num_pairs) {
        if (keys_in_capacity < num_pairs) {
            if (keys_in) cudaFree(keys_in);
            cudaMalloc(&keys_in, num_pairs * sizeof(int32_t));
            keys_in_capacity = num_pairs;
        }
        if (idx_in_capacity < num_pairs) {
            if (idx_in) cudaFree(idx_in);
            cudaMalloc(&idx_in, num_pairs * sizeof(int32_t));
            idx_in_capacity = num_pairs;
        }
        if (keys_out_capacity < num_pairs) {
            if (keys_out) cudaFree(keys_out);
            cudaMalloc(&keys_out, num_pairs * sizeof(int32_t));
            keys_out_capacity = num_pairs;
        }
        if (idx_out_capacity < num_pairs) {
            if (idx_out) cudaFree(idx_out);
            cudaMalloc(&idx_out, num_pairs * sizeof(int32_t));
            idx_out_capacity = num_pairs;
        }
    }

    void ensure_sort_temp(size_t needed) {
        if (sort_temp_bytes < needed) {
            if (sort_temp) cudaFree(sort_temp);
            cudaMalloc(&sort_temp, needed);
            sort_temp_bytes = needed;
        }
    }

    ~Cache() override {
        if (keys_in) cudaFree(keys_in);
        if (idx_in) cudaFree(idx_in);
        if (keys_out) cudaFree(keys_out);
        if (idx_out) cudaFree(idx_out);
        if (sort_temp) cudaFree(sort_temp);
    }
};





__device__ __forceinline__ int lower_bound_dev(const int32_t* __restrict__ arr, int size, int32_t target)
{
    int lo = 0;
    int hi = size;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        int32_t v = __ldg(arr + mid);
        if (v < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int lower_bound_from(const int32_t* __restrict__ arr,
                                                int lo,
                                                int hi,
                                                int32_t target)
{
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        int32_t v = __ldg(arr + mid);
        if (v < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int galloping_lb(const int32_t* __restrict__ arr, int start, int end, int32_t target)
{
    if (start >= end) return end;
    if (__ldg(arr + start) >= target) return start;

    int pos = start;
    int step = 1;
    while (pos + step < end && __ldg(arr + pos + step) < target) {
        pos += step;
        step <<= 1;
    }

    int lo = pos + 1;
    int hi = (pos + step + 1 < end) ? (pos + step + 1) : end;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        int32_t v = __ldg(arr + mid);
        if (v < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int intersect_merge_simple(const int32_t* __restrict__ a,
                                                     int m,
                                                     const int32_t* __restrict__ b,
                                                     int n)
{
    int count = 0;
    int i = 0;
    int j = 0;
    while (i < m && j < n) {
        int32_t va = __ldg(a + i);
        int32_t vb = __ldg(b + j);
        count += (va == vb);
        i += (va <= vb);
        j += (va >= vb);
    }
    return count;
}

__device__ __forceinline__ int intersect_merge_skip(const int32_t* __restrict__ a,
                                                   int m,
                                                   const int32_t* __restrict__ b,
                                                   int n)
{
    
    int i = lower_bound_dev(a, m, __ldg(b));
    if (i >= m) return 0;
    int j = lower_bound_dev(b, n, __ldg(a + i));
    if (j >= n) return 0;

    int count = 0;
    while (i < m && j < n) {
        int32_t va = __ldg(a + i);
        int32_t vb = __ldg(b + j);
        count += (va == vb);
        i += (va <= vb);
        j += (va >= vb);
    }
    return count;
}

__device__ __forceinline__ int intersect_binary_search(const int32_t* __restrict__ small,
                                                      int m,
                                                      const int32_t* __restrict__ large,
                                                      int n)
{
    int count = 0;
    int j = 0;
    for (int i = 0; i < m && j < n; ++i) {
        int32_t target = __ldg(small + i);
        j = lower_bound_from(large, j, n, target);
        if (j < n && __ldg(large + j) == target) {
            ++count;
            ++j;
        }
    }
    return count;
}

__device__ __forceinline__ int intersect_gallop(const int32_t* __restrict__ small,
                                               int m,
                                               const int32_t* __restrict__ large,
                                               int n)
{
    int count = 0;
    int j = 0;
    for (int i = 0; i < m && j < n; ++i) {
        int32_t target = __ldg(small + i);
        j = galloping_lb(large, j, n, target);
        if (j < n && __ldg(large + j) == target) {
            ++count;
            ++j;  
        }
    }
    return count;
}

__device__ __forceinline__ int ilog2_ceil_u32(uint32_t x)
{
    
    return 32 - __clz(x - 1);
}


__device__ __forceinline__ int intersect_count(const int32_t* __restrict__ a,
                                              int m,
                                              const int32_t* __restrict__ b,
                                              int n)
{
    if (m == 0 || n == 0) return 0;

    
    if (m > n) {
        const int32_t* t = a;
        a = b;
        b = t;
        int tmp = m;
        m = n;
        n = tmp;
    }

    if (m == 1) {
        int32_t x = __ldg(a);
        int lb = lower_bound_dev(b, n, x);
        return (lb < n && __ldg(b + lb) == x) ? 1 : 0;
    }

    
    int logn = ilog2_ceil_u32((uint32_t)n);
    int cost_bs = m * logn;
    int cost_merge = m + n;

    if (cost_bs < cost_merge) {
        if (n > 8 * m) return intersect_gallop(a, m, b, n);
        return intersect_binary_search(a, m, b, n);
    }

    if (m + n > 48) return intersect_merge_skip(a, m, b, n);
    return intersect_merge_simple(a, m, b, n);
}





__global__ void compute_sort_keys(const int32_t* __restrict__ pairs_first,
                                  int32_t* __restrict__ sort_keys,
                                  int32_t* __restrict__ pair_indices,
                                  int64_t num_pairs)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    sort_keys[idx] = pairs_first[idx];
    pair_indices[idx] = (int32_t)idx;
}

__global__ void jaccard_kernel_direct(const int32_t* __restrict__ offsets,
                                      const int32_t* __restrict__ indices,
                                      const int32_t* __restrict__ pairs_first,
                                      const int32_t* __restrict__ pairs_second,
                                      float* __restrict__ scores,
                                      int64_t num_pairs)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int32_t u = __ldg(pairs_first + idx);
    int32_t v = __ldg(pairs_second + idx);

    int32_t u0 = __ldg(offsets + u);
    int32_t u1 = __ldg(offsets + u + 1);
    int32_t v0 = __ldg(offsets + v);
    int32_t v1 = __ldg(offsets + v + 1);

    int deg_u = (int)(u1 - u0);
    int deg_v = (int)(v1 - v0);

    if (u == v) {
        scores[idx] = (deg_u > 0) ? 1.0f : 0.0f;
        return;
    }
    if (deg_u == 0 || deg_v == 0) {
        scores[idx] = 0.0f;
        return;
    }

    int inter = intersect_count(indices + u0, deg_u, indices + v0, deg_v);
    int uni = deg_u + deg_v - inter;
    scores[idx] = (uni > 0) ? __int2float_rn(inter) / __int2float_rn(uni) : 0.0f;
}

__global__ void jaccard_kernel_indexed(const int32_t* __restrict__ offsets,
                                       const int32_t* __restrict__ indices,
                                       const int32_t* __restrict__ pairs_first,
                                       const int32_t* __restrict__ pairs_second,
                                       const int32_t* __restrict__ sorted_indices,
                                       float* __restrict__ scores,
                                       int64_t num_pairs)
{
    int64_t sorted_pos = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted_pos >= num_pairs) return;

    int64_t idx = (int64_t)__ldg(sorted_indices + sorted_pos);

    int32_t u = __ldg(pairs_first + idx);
    int32_t v = __ldg(pairs_second + idx);

    int32_t u0 = __ldg(offsets + u);
    int32_t u1 = __ldg(offsets + u + 1);
    int32_t v0 = __ldg(offsets + v);
    int32_t v1 = __ldg(offsets + v + 1);

    int deg_u = (int)(u1 - u0);
    int deg_v = (int)(v1 - v0);

    if (u == v) {
        scores[idx] = (deg_u > 0) ? 1.0f : 0.0f;
        return;
    }
    if (deg_u == 0 || deg_v == 0) {
        scores[idx] = 0.0f;
        return;
    }

    int inter = intersect_count(indices + u0, deg_u, indices + v0, deg_v);
    int uni = deg_u + deg_v - inter;
    scores[idx] = (uni > 0) ? __int2float_rn(inter) / __int2float_rn(uni) : 0.0f;
}

}  

void jaccard_similarity_seg(const graph32_t& graph,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    int64_t np = static_cast<int64_t>(num_pairs);

    float avg_degree = (num_vertices > 0)
        ? (static_cast<float>(num_edges) / static_cast<float>(num_vertices))
        : 0.0f;

    
    bool should_sort = (np > 100'000) && (avg_degree > 20.0f);

    if (!should_sort) {
        if (np <= 0) return;
        int threads = 256;
        int blocks = (int)((np + threads - 1) / threads);
        jaccard_kernel_direct<<<blocks, threads>>>(offsets, indices,
                                                   vertex_pairs_first,
                                                   vertex_pairs_second,
                                                   similarity_scores, np);
        return;
    }

    
    cache.ensure_pairs(np);

    
    int end_bit = 1;
    int nv = num_vertices;
    while ((1 << end_bit) < nv + 1) end_bit++;
    if (end_bit > 26) end_bit = 26;

    
    size_t needed_tmp = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, needed_tmp,
                                    (int32_t*)nullptr, (int32_t*)nullptr,
                                    (int32_t*)nullptr, (int32_t*)nullptr,
                                    (int)np, 0, end_bit);
    needed_tmp = ((needed_tmp + 255) / 256) * 256;
    cache.ensure_sort_temp(needed_tmp);

    
    {
        int threads = 256;
        int blocks = (int)((np + threads - 1) / threads);
        compute_sort_keys<<<blocks, threads>>>(vertex_pairs_first,
                                               cache.keys_in,
                                               cache.idx_in, np);
    }

    
    cub::DeviceRadixSort::SortPairs(cache.sort_temp, cache.sort_temp_bytes,
                                    cache.keys_in, cache.keys_out,
                                    cache.idx_in, cache.idx_out,
                                    (int)np, 0, end_bit);

    
    {
        int threads = 256;
        int blocks = (int)((np + threads - 1) / threads);
        jaccard_kernel_indexed<<<blocks, threads>>>(offsets, indices,
                                                    vertex_pairs_first,
                                                    vertex_pairs_second,
                                                    cache.idx_out,
                                                    similarity_scores, np);
    }
}

}  
