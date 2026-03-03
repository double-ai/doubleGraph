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
#include <optional>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* degrees = nullptr;
    int64_t degrees_capacity = 0;

    int32_t* seed_buf = nullptr;
    int64_t seed_capacity = 0;

    int64_t* counts = nullptr;
    int64_t counts_capacity = 0;

    void ensure_degrees(int64_t n) {
        if (degrees_capacity < n) {
            if (degrees) cudaFree(degrees);
            cudaMalloc(&degrees, n * sizeof(float));
            degrees_capacity = n;
        }
    }

    void ensure_seeds(int64_t n) {
        if (seed_capacity < n) {
            if (seed_buf) cudaFree(seed_buf);
            cudaMalloc(&seed_buf, n * sizeof(int32_t));
            seed_capacity = n;
        }
    }

    void ensure_counts(int64_t n) {
        if (counts_capacity < n) {
            if (counts) cudaFree(counts);
            cudaMalloc(&counts, n * sizeof(int64_t));
            counts_capacity = n;
        }
    }

    ~Cache() override {
        if (degrees) cudaFree(degrees);
        if (seed_buf) cudaFree(seed_buf);
        if (counts) cudaFree(counts);
    }
};





__global__ void compute_weighted_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ edge_weights,
    float* __restrict__ degrees,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    float sum = 0.0f;
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    for (int32_t e = start; e < end; e++)
        sum += edge_weights[e];
    degrees[v] = sum;
}

__global__ void count_raw_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ counts)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_seeds) return;
    int32_t u = seeds[sid];
    int64_t c = 0;
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    for (int32_t e = u_start; e < u_end; e++) {
        int32_t w = indices[e];
        int32_t w_deg = offsets[w + 1] - offsets[w];
        c += (int64_t)(w_deg - 1);
    }
    counts[sid] = c;
}

__global__ void generate_pairs_block_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ prefix_sums,
    int64_t* __restrict__ keys,
    int64_t max_v)
{
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int32_t u = seeds[sid];
    int64_t base = prefix_sums[sid];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t u_deg = u_end - u_start;

    __shared__ int64_t s_pos;
    if (threadIdx.x == 0) s_pos = 0;
    __syncthreads();

    for (int32_t w_idx = 0; w_idx < u_deg; w_idx++) {
        int32_t w = indices[u_start + w_idx];
        int32_t w_start = offsets[w];
        int32_t w_end = offsets[w + 1];

        for (int32_t f = w_start + threadIdx.x; f < w_end; f += blockDim.x) {
            int32_t v = indices[f];
            if (v != u) {
                int64_t pos = atomicAdd((unsigned long long*)&s_pos, 1ULL);
                keys[base + pos] = (int64_t)u * max_v + (int64_t)v;
            }
        }
    }
}

__device__ __forceinline__ float weighted_intersection_merge(
    const int32_t* __restrict__ a_idx, const float* __restrict__ a_wt, int32_t a_size,
    const int32_t* __restrict__ b_idx, const float* __restrict__ b_wt, int32_t b_size)
{
    if (a_size == 0 || b_size == 0) return 0.0f;

    int32_t a_first = a_idx[0], a_last = a_idx[a_size - 1];
    int32_t b_first = b_idx[0], b_last = b_idx[b_size - 1];
    if (a_first > b_last || b_first > a_last) return 0.0f;

    float isect = 0.0f;
    int32_t i = 0, j = 0;

    if (a_first < b_first) {
        int32_t lo = 0, hi = a_size;
        while (lo < hi) {
            int32_t mid = lo + (hi - lo) / 2;
            if (a_idx[mid] < b_first) lo = mid + 1; else hi = mid;
        }
        i = lo;
    } else if (b_first < a_first) {
        int32_t lo = 0, hi = b_size;
        while (lo < hi) {
            int32_t mid = lo + (hi - lo) / 2;
            if (b_idx[mid] < a_first) lo = mid + 1; else hi = mid;
        }
        j = lo;
    }

    while (i < a_size && j < b_size) {
        int32_t av = a_idx[i], bv = b_idx[j];
        if (av == bv) {
            isect += fminf(a_wt[i], b_wt[j]);
            i++; j++;
        } else if (av < bv) {
            i++;
        } else {
            j++;
        }
    }
    return isect;
}

__device__ __forceinline__ float weighted_intersection_gallop(
    const int32_t* __restrict__ small_idx, const float* __restrict__ small_wt, int32_t small_size,
    const int32_t* __restrict__ large_idx, const float* __restrict__ large_wt, int32_t large_size)
{
    if (small_size == 0 || large_size == 0) return 0.0f;

    float isect = 0.0f;
    int32_t j = 0;

    for (int32_t i = 0; i < small_size && j < large_size; i++) {
        int32_t target = small_idx[i];

        int32_t step = 1;
        int32_t pos = j;
        while (pos + step < large_size && large_idx[pos + step] < target) {
            pos += step;
            step *= 2;
        }

        int32_t lo = pos, hi = (pos + step + 1 < large_size) ? pos + step + 1 : large_size;
        while (lo < hi) {
            int32_t mid = lo + (hi - lo) / 2;
            if (large_idx[mid] < target) lo = mid + 1; else hi = mid;
        }
        j = lo;

        if (j < large_size && large_idx[j] == target) {
            isect += fminf(small_wt[i], large_wt[j]);
            j++;
        }
    }
    return isect;
}

__global__ void compute_sorensen_kernel(
    const int64_t* __restrict__ keys,
    int64_t num_pairs,
    int64_t max_v,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ degrees,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores)
{
    int64_t pid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_pairs) return;

    int64_t key = keys[pid];
    int32_t u = (int32_t)(key / max_v);
    int32_t v = (int32_t)(key % max_v);

    out_first[pid] = u;
    out_second[pid] = v;

    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int32_t v_start = offsets[v], v_end = offsets[v + 1];
    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    float isect;
    int32_t ratio = (u_deg > v_deg) ? (u_deg / (v_deg > 0 ? v_deg : 1)) : (v_deg / (u_deg > 0 ? u_deg : 1));

    if (ratio > 8) {
        if (u_deg <= v_deg) {
            isect = weighted_intersection_gallop(
                indices + u_start, edge_weights + u_start, u_deg,
                indices + v_start, edge_weights + v_start, v_deg);
        } else {
            isect = weighted_intersection_gallop(
                indices + v_start, edge_weights + v_start, v_deg,
                indices + u_start, edge_weights + u_start, u_deg);
        }
    } else {
        isect = weighted_intersection_merge(
            indices + u_start, edge_weights + u_start, u_deg,
            indices + v_start, edge_weights + v_start, v_deg);
    }

    float denom = degrees[u] + degrees[v];
    out_scores[pid] = (denom > 0.0f) ? (2.0f * isect / denom) : 0.0f;
}

__global__ void iota_kernel(int32_t* arr, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = idx;
}

}  

similarity_result_float_t sorensen_all_pairs_similarity(const graph32_t& graph,
                                                        const float* edge_weights,
                                                        const int32_t* vertices,
                                                        std::size_t num_vertices,
                                                        std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t nv = graph.number_of_vertices;

    
    const int32_t* d_seeds;
    int32_t num_seeds;

    if (vertices != nullptr && num_vertices > 0) {
        d_seeds = vertices;
        num_seeds = (int32_t)num_vertices;
    } else {
        cache.ensure_seeds(nv);
        if (nv > 0) {
            int block = 256;
            int grid = (nv + block - 1) / block;
            iota_kernel<<<grid, block>>>(cache.seed_buf, nv);
        }
        d_seeds = cache.seed_buf;
        num_seeds = nv;
    }

    
    cache.ensure_degrees(nv);
    {
        int block = 256;
        int grid = (nv + block - 1) / block;
        if (grid > 0)
            compute_weighted_degrees_kernel<<<grid, block>>>(d_offsets, edge_weights, cache.degrees, nv);
    }

    
    cache.ensure_counts(num_seeds + 1);
    {
        int block = 256;
        int grid = (num_seeds + block - 1) / block;
        if (grid > 0)
            count_raw_pairs_kernel<<<grid, block>>>(d_offsets, d_indices, d_seeds, num_seeds, cache.counts);
    }

    
    {
        thrust::device_ptr<int64_t> ptr(cache.counts);
        thrust::exclusive_scan(thrust::device, ptr, ptr + (num_seeds + 1), ptr, (int64_t)0);
    }

    
    int64_t total_raw = 0;
    cudaMemcpy(&total_raw, cache.counts + num_seeds, sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (total_raw == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t max_v = (int64_t)nv;
    int64_t* keys = nullptr;
    cudaMalloc(&keys, total_raw * sizeof(int64_t));

    if (num_seeds > 0) {
        int block = 256;
        generate_pairs_block_kernel<<<num_seeds, block>>>(
            d_offsets, d_indices, d_seeds, num_seeds, cache.counts, keys, max_v);
    }

    
    if (total_raw > 1) {
        thrust::device_ptr<int64_t> ptr(keys);
        thrust::sort(thrust::device, ptr, ptr + total_raw);
    }

    
    int64_t unique_count;
    if (total_raw > 0) {
        thrust::device_ptr<int64_t> ptr(keys);
        auto new_end = thrust::unique(thrust::device, ptr, ptr + total_raw);
        unique_count = (int64_t)(new_end - ptr);
    } else {
        unique_count = 0;
    }

    
    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;
    cudaMalloc(&out_first, unique_count * sizeof(int32_t));
    cudaMalloc(&out_second, unique_count * sizeof(int32_t));
    cudaMalloc(&out_scores, unique_count * sizeof(float));

    if (unique_count > 0) {
        int block = 256;
        int grid = (int)((unique_count + block - 1) / block);
        compute_sorensen_kernel<<<grid, block>>>(
            keys, unique_count, max_v, d_offsets, d_indices, edge_weights,
            cache.degrees, out_first, out_second, out_scores);
    }

    cudaFree(keys);

    
    if (topk.has_value() && (int64_t)topk.value() < unique_count) {
        int64_t k = (int64_t)topk.value();

        
        {
            thrust::device_ptr<float> score_ptr(out_scores);
            thrust::device_ptr<int32_t> first_ptr(out_first);
            thrust::device_ptr<int32_t> second_ptr(out_second);

            auto vals = thrust::make_zip_iterator(
                thrust::make_tuple(first_ptr, second_ptr));

            thrust::sort_by_key(thrust::device, score_ptr, score_ptr + unique_count, vals,
                                thrust::greater<float>());
        }

        int32_t* tk_first = nullptr;
        int32_t* tk_second = nullptr;
        float* tk_scores = nullptr;
        cudaMalloc(&tk_first, k * sizeof(int32_t));
        cudaMalloc(&tk_second, k * sizeof(int32_t));
        cudaMalloc(&tk_scores, k * sizeof(float));

        cudaMemcpyAsync(tk_first, out_first, k * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(tk_second, out_second, k * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(tk_scores, out_scores, k * sizeof(float), cudaMemcpyDeviceToDevice);

        cudaFree(out_first);
        cudaFree(out_second);
        cudaFree(out_scores);

        return {tk_first, tk_second, tk_scores, (std::size_t)k};
    }

    return {out_first, out_second, out_scores, (std::size_t)unique_count};
}

}  
