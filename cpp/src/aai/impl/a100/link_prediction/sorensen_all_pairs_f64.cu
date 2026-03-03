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
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <cstdint>
#include <climits>
#include <algorithm>
#include <optional>
#include <vector>

namespace aai {

namespace {

struct Cache : Cacheable {
    double* degrees = nullptr;
    int64_t degrees_capacity = 0;

    int32_t* seeds = nullptr;
    int64_t seeds_capacity = 0;

    int64_t* counts = nullptr;
    int64_t counts_capacity = 0;

    int64_t* pair_offsets_buf = nullptr;
    int64_t pair_offsets_capacity = 0;

    void ensure_degrees(int64_t n) {
        if (degrees_capacity < n) {
            if (degrees) cudaFree(degrees);
            cudaMalloc(&degrees, n * sizeof(double));
            degrees_capacity = n;
        }
    }

    void ensure_seeds(int64_t n) {
        if (seeds_capacity < n) {
            if (seeds) cudaFree(seeds);
            cudaMalloc(&seeds, n * sizeof(int32_t));
            seeds_capacity = n;
        }
    }

    void ensure_counts(int64_t n) {
        if (counts_capacity < n) {
            if (counts) cudaFree(counts);
            cudaMalloc(&counts, n * sizeof(int64_t));
            counts_capacity = n;
        }
    }

    void ensure_pair_offsets(int64_t n) {
        if (pair_offsets_capacity < n) {
            if (pair_offsets_buf) cudaFree(pair_offsets_buf);
            cudaMalloc(&pair_offsets_buf, n * sizeof(int64_t));
            pair_offsets_capacity = n;
        }
    }

    ~Cache() override {
        if (degrees) cudaFree(degrees);
        if (seeds) cudaFree(seeds);
        if (counts) cudaFree(counts);
        if (pair_offsets_buf) cudaFree(pair_offsets_buf);
    }
};



__global__ void compute_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ weights,
    double* __restrict__ degrees,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    double sum = 0.0;
    for (int32_t i = offsets[v]; i < offsets[v + 1]; i++) {
        sum += weights[i];
    }
    degrees[v] = sum;
}

__global__ void count_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ counts
) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_seeds) return;
    int32_t u = seeds[sid];
    int64_t count = 0;
    for (int32_t i = offsets[u]; i < offsets[u + 1]; i++) {
        int32_t w = indices[i];
        count += (int64_t)(offsets[w + 1] - offsets[w] - 1);
    }
    counts[sid] = count;
}

__global__ void fill_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ seed_offsets,
    int64_t* __restrict__ pair_keys,
    int64_t max_v
) {
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int32_t u = seeds[sid];
    int64_t base = seed_offsets[sid];

    __shared__ unsigned long long s_count;
    if (threadIdx.x == 0) s_count = 0ULL;
    __syncthreads();

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];

    for (int32_t i = u_start; i < u_end; i++) {
        int32_t w = indices[i];
        int32_t w_start = offsets[w];
        int32_t w_end = offsets[w + 1];

        for (int32_t j = w_start + (int32_t)threadIdx.x; j < w_end; j += (int32_t)blockDim.x) {
            int32_t v = indices[j];
            if (v != u) {
                unsigned long long pos = atomicAdd(&s_count, 1ULL);
                pair_keys[base + (int64_t)pos] = (int64_t)u * max_v + (int64_t)v;
            }
        }
        __syncthreads();
    }
}

__global__ void compute_sorensen_key_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ degrees,
    const int64_t* __restrict__ pair_keys,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    double* __restrict__ out_scores,
    int64_t num_pairs,
    int64_t max_v
) {
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int64_t key = pair_keys[warp_id];
    int32_t u = (int32_t)(key / max_v);
    int32_t v = (int32_t)(key % max_v);

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t u_deg = u_end - u_start;

    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];
    int32_t v_deg = v_end - v_start;

    const int32_t* iter_base;
    const double* iter_w;
    int32_t iter_deg;
    const int32_t* search_base;
    const double* search_w;
    int32_t search_deg;

    if (u_deg <= v_deg) {
        iter_base = indices + u_start;
        iter_w = weights + u_start;
        iter_deg = u_deg;
        search_base = indices + v_start;
        search_w = weights + v_start;
        search_deg = v_deg;
    } else {
        iter_base = indices + v_start;
        iter_w = weights + v_start;
        iter_deg = v_deg;
        search_base = indices + u_start;
        search_w = weights + u_start;
        search_deg = u_deg;
    }

    double isect_weight = 0.0;

    for (int i = lane; i < iter_deg; i += 32) {
        int32_t target = iter_base[i];
        double w_i = iter_w[i];

        int lo = 0, hi = search_deg;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (search_base[mid] < target) lo = mid + 1;
            else hi = mid;
        }

        if (lo < search_deg && search_base[lo] == target) {
            isect_weight += fmin(w_i, search_w[lo]);
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        isect_weight += __shfl_down_sync(0xffffffff, isect_weight, offset);
    }

    if (lane == 0) {
        out_first[warp_id] = u;
        out_second[warp_id] = v;
        double denom = degrees[u] + degrees[v];
        out_scores[warp_id] = (denom > 0.0) ? (2.0 * isect_weight / denom) : 0.0;
    }
}

__global__ void iota_kernel(int32_t* data, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = i;
}

}  

similarity_result_double_t sorensen_all_pairs_similarity(
    const graph32_t& graph,
    const double* edge_weights,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk) {

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const double* d_weights = edge_weights;

    
    const int32_t* d_seeds;
    int32_t num_seeds;

    if (vertices != nullptr) {
        d_seeds = vertices;
        num_seeds = (int32_t)num_vertices;
    } else {
        cache.ensure_seeds(nv);
        if (nv > 0) {
            iota_kernel<<<(nv + 255) / 256, 256>>>(cache.seeds, nv);
        }
        d_seeds = cache.seeds;
        num_seeds = nv;
    }

    
    cache.ensure_degrees(nv);
    if (nv > 0) {
        int block = 256;
        int grid = (nv + block - 1) / block;
        compute_degrees_kernel<<<grid, block>>>(d_offsets, d_weights, cache.degrees, nv);
    }

    
    cache.ensure_counts(num_seeds);
    if (num_seeds > 0) {
        int block = 256;
        int grid = (num_seeds + block - 1) / block;
        count_pairs_kernel<<<grid, block>>>(d_offsets, d_indices, d_seeds, num_seeds, cache.counts);
    }

    
    std::vector<int64_t> h_counts(num_seeds);
    cudaMemcpy(h_counts.data(), cache.counts, num_seeds * sizeof(int64_t), cudaMemcpyDeviceToHost);

    int64_t total_pairs = 0;
    std::vector<int64_t> h_offsets(num_seeds);
    for (int32_t i = 0; i < num_seeds; i++) {
        h_offsets[i] = total_pairs;
        total_pairs += h_counts[i];
    }

    if (total_pairs == 0) {
        return similarity_result_double_t{nullptr, nullptr, nullptr, 0};
    }

    
    cache.ensure_pair_offsets(num_seeds);
    cudaMemcpy(cache.pair_offsets_buf, h_offsets.data(),
               num_seeds * sizeof(int64_t), cudaMemcpyHostToDevice);

    int64_t max_v = (int64_t)nv;

    
    int64_t* pair_keys = nullptr;
    cudaMalloc(&pair_keys, total_pairs * sizeof(int64_t));
    fill_pairs_kernel<<<num_seeds, 256>>>(d_offsets, d_indices, d_seeds, num_seeds,
                                          cache.pair_offsets_buf, pair_keys, max_v);

    
    int end_bit = 64;
    if (nv > 1) {
        uint64_t max_key = (uint64_t)(nv - 1) * (uint64_t)nv + (uint64_t)(nv - 1);
        end_bit = 0;
        while ((1ULL << end_bit) <= max_key && end_bit < 64) end_bit++;
    }

    
    int64_t* pair_keys_sorted = nullptr;
    cudaMalloc(&pair_keys_sorted, total_pairs * sizeof(int64_t));

    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys((void*)nullptr, sort_temp_bytes,
        (uint64_t*)nullptr, (uint64_t*)nullptr, total_pairs, 0, end_bit);

    void* sort_temp = nullptr;
    cudaMalloc(&sort_temp, sort_temp_bytes + 16);
    cub::DeviceRadixSort::SortKeys(sort_temp, sort_temp_bytes,
        (uint64_t*)pair_keys, (uint64_t*)pair_keys_sorted,
        total_pairs, 0, end_bit);
    cudaFree(sort_temp);
    cudaFree(pair_keys);

    
    int64_t* unique_keys = nullptr;
    cudaMalloc(&unique_keys, total_pairs * sizeof(int64_t));

    int64_t* num_unique_d = nullptr;
    cudaMalloc(&num_unique_d, sizeof(int64_t));

    size_t unique_temp_bytes = 0;
    cub::DeviceSelect::Unique((void*)nullptr, unique_temp_bytes,
        (int64_t*)nullptr, (int64_t*)nullptr, (int64_t*)nullptr, total_pairs);

    void* unique_temp = nullptr;
    cudaMalloc(&unique_temp, unique_temp_bytes + 16);
    cub::DeviceSelect::Unique(unique_temp, unique_temp_bytes,
        pair_keys_sorted, unique_keys, num_unique_d, total_pairs);
    cudaFree(unique_temp);
    cudaFree(pair_keys_sorted);

    int64_t num_unique;
    cudaMemcpy(&num_unique, num_unique_d, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaFree(num_unique_d);

    if (num_unique == 0) {
        cudaFree(unique_keys);
        return similarity_result_double_t{nullptr, nullptr, nullptr, 0};
    }

    
    int32_t* first_out = nullptr;
    int32_t* second_out = nullptr;
    double* scores_out = nullptr;
    cudaMalloc(&first_out, num_unique * sizeof(int32_t));
    cudaMalloc(&second_out, num_unique * sizeof(int32_t));
    cudaMalloc(&scores_out, num_unique * sizeof(double));

    {
        int block = 256;
        int64_t total_threads = num_unique * 32;
        int64_t grid64 = (total_threads + block - 1) / block;
        int grid = (int)std::min(grid64, (int64_t)INT32_MAX);
        compute_sorensen_key_kernel<<<grid, block>>>(
            d_offsets, d_indices, d_weights, cache.degrees,
            unique_keys, first_out, second_out, scores_out,
            num_unique, max_v);
    }

    cudaFree(unique_keys);

    
    if (topk.has_value() && topk.value() < (std::size_t)num_unique) {
        int64_t topk_val = (int64_t)topk.value();

        int32_t* idx = nullptr;
        cudaMalloc(&idx, num_unique * sizeof(int32_t));
        int32_t iota_n = (int32_t)std::min(num_unique, (int64_t)INT32_MAX);
        iota_kernel<<<(iota_n + 255) / 256, 256>>>(idx, iota_n);

        {
            thrust::device_ptr<double> k(scores_out);
            thrust::device_ptr<int32_t> v(idx);
            thrust::sort_by_key(thrust::device, k, k + num_unique, v, thrust::greater<double>());
        }

        int32_t* first_topk = nullptr;
        int32_t* second_topk = nullptr;
        double* scores_topk = nullptr;
        cudaMalloc(&first_topk, topk_val * sizeof(int32_t));
        cudaMalloc(&second_topk, topk_val * sizeof(int32_t));
        cudaMalloc(&scores_topk, topk_val * sizeof(double));

        {
            thrust::device_ptr<const int32_t> m(idx);
            thrust::device_ptr<const int32_t> s1(first_out);
            thrust::device_ptr<int32_t> d1(first_topk);
            thrust::gather(thrust::device, m, m + topk_val, s1, d1);

            thrust::device_ptr<const int32_t> s2(second_out);
            thrust::device_ptr<int32_t> d2(second_topk);
            thrust::gather(thrust::device, m, m + topk_val, s2, d2);
        }

        cudaMemcpy(scores_topk, scores_out, topk_val * sizeof(double), cudaMemcpyDeviceToDevice);

        cudaFree(idx);
        cudaFree(first_out);
        cudaFree(second_out);
        cudaFree(scores_out);

        return similarity_result_double_t{first_topk, second_topk, scores_topk, (std::size_t)topk_val};
    }

    return similarity_result_double_t{first_out, second_out, scores_out, (std::size_t)num_unique};
}

}  
