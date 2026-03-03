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
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <optional>

namespace aai {

namespace {

static const int64_t SENTINEL_KEY = 0x7FFFFFFFFFFFFFFFLL;

static constexpr int64_t MAX_KEYS = 32 * 1024 * 1024;
static constexpr int64_t MAX_RESULTS = 8 * 1024 * 1024;
static constexpr int64_t MAX_VERTICES = 64 * 1024 * 1024;
static constexpr int64_t MAX_SEEDS = 8192;
static constexpr int64_t SORT_TEMP = 512 * 1024 * 1024;

struct Cache : Cacheable {
    float* wsums_buf = nullptr;
    int64_t* counts_buf = nullptr;
    int64_t* offsets_buf = nullptr;
    int32_t* seeds_buf = nullptr;
    int64_t* keys_buf = nullptr;
    int64_t* keys_alt_buf = nullptr;
    int32_t* first_buf = nullptr;
    int32_t* second_buf = nullptr;
    float* scores_buf = nullptr;
    void* sort_temp_buf = nullptr;

    Cache() {
        cudaMalloc(&wsums_buf, MAX_VERTICES * sizeof(float));
        cudaMalloc(&counts_buf, MAX_SEEDS * sizeof(int64_t));
        cudaMalloc(&offsets_buf, (MAX_SEEDS + 1) * sizeof(int64_t));
        cudaMalloc(&seeds_buf, MAX_VERTICES * sizeof(int32_t));
        cudaMalloc(&keys_buf, MAX_KEYS * sizeof(int64_t));
        cudaMalloc(&keys_alt_buf, MAX_KEYS * sizeof(int64_t));
        cudaMalloc(&first_buf, MAX_RESULTS * sizeof(int32_t));
        cudaMalloc(&second_buf, MAX_RESULTS * sizeof(int32_t));
        cudaMalloc(&scores_buf, MAX_RESULTS * sizeof(float));
        cudaMalloc(&sort_temp_buf, SORT_TEMP);
    }

    ~Cache() override {
        if (wsums_buf) cudaFree(wsums_buf);
        if (counts_buf) cudaFree(counts_buf);
        if (offsets_buf) cudaFree(offsets_buf);
        if (seeds_buf) cudaFree(seeds_buf);
        if (keys_buf) cudaFree(keys_buf);
        if (keys_alt_buf) cudaFree(keys_alt_buf);
        if (first_buf) cudaFree(first_buf);
        if (second_buf) cudaFree(second_buf);
        if (scores_buf) cudaFree(scores_buf);
        if (sort_temp_buf) cudaFree(sort_temp_buf);
    }
};

__global__ void weight_sums_kernel(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ edge_weights,
    int32_t num_vertices,
    float* __restrict__ weight_sums
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    float sum = 0.0f;
    for (int32_t i = start; i < end; i++)
        sum += edge_weights[i];
    weight_sums[v] = sum;
}

__global__ void count_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ raw_counts
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_seeds) return;
    int32_t u = seeds[warp_id];
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t u_deg = u_end - u_start;
    int64_t count = 0;
    for (int32_t i = lane; i < u_deg; i += 32) {
        int32_t n = indices[u_start + i];
        count += (int64_t)(offsets[n + 1] - offsets[n]);
    }
    for (int s = 16; s > 0; s >>= 1)
        count += __shfl_down_sync(0xffffffff, count, s);
    if (lane == 0) raw_counts[warp_id] = count;
}

__global__ void generate_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ seed_offsets,
    int64_t* __restrict__ out_keys
) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;
    int32_t u = seeds[seed_idx];
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int64_t base = seed_offsets[seed_idx];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int64_t cum = 0;
    for (int32_t i = u_start; i < u_end; i++) {
        int32_t n = indices[i];
        int32_t n_start = offsets[n];
        int32_t n_end = offsets[n + 1];
        int32_t n_deg = n_end - n_start;
        for (int32_t j = tid; j < n_deg; j += nthreads) {
            int32_t v = indices[n_start + j];
            out_keys[base + cum + j] = (v != u) ?
                (((int64_t)(uint32_t)u << 32) | (int64_t)(uint32_t)v) : SENTINEL_KEY;
        }
        cum += n_deg;
    }
}

__global__ void __launch_bounds__(256, 8)
jaccard_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ weight_sums,
    const int64_t* __restrict__ pair_keys,
    int64_t num_pairs,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores
) {
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_pairs) return;

    int64_t key = pair_keys[warp_id];
    int32_t u = (int32_t)(uint32_t)(key >> 32);
    int32_t v = (int32_t)(uint32_t)(key & 0xFFFFFFFF);

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];
    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    const int32_t* iter_idx;
    const float* iter_wt;
    int32_t iter_len;
    const int32_t* srch_idx;
    const float* srch_wt;
    int32_t srch_len;

    if (u_deg <= v_deg) {
        iter_idx = indices + u_start; iter_wt = edge_weights + u_start; iter_len = u_deg;
        srch_idx = indices + v_start; srch_wt = edge_weights + v_start; srch_len = v_deg;
    } else {
        iter_idx = indices + v_start; iter_wt = edge_weights + v_start; iter_len = v_deg;
        srch_idx = indices + u_start; srch_wt = edge_weights + u_start; srch_len = u_deg;
    }

    float local_iw = 0.0f;
    for (int32_t i = lane; i < iter_len; i += 32) {
        int32_t target = iter_idx[i];
        float w1 = iter_wt[i];
        int32_t lo = 0, hi = srch_len;
        while (lo < hi) {
            int32_t mid = (lo + hi) >> 1;
            if (srch_idx[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        if (lo < srch_len && srch_idx[lo] == target)
            local_iw += fminf(w1, srch_wt[lo]);
    }

    for (int s = 16; s > 0; s >>= 1)
        local_iw += __shfl_down_sync(0xffffffff, local_iw, s);

    if (lane == 0) {
        float denom = weight_sums[u] + weight_sums[v] - local_iw;
        out_scores[warp_id] = (denom > 0.0f) ? (local_iw / denom) : 0.0f;
        out_first[warp_id] = u;
        out_second[warp_id] = v;
    }
}

struct InternalResult {
    int32_t* first;
    int32_t* second;
    float* scores;
    int64_t count;
    bool owns_result;
};

InternalResult launch_jaccard_all_pairs(
    const int32_t* d_offsets,
    const int32_t* d_indices,
    const float* d_edge_weights,
    int32_t num_vertices,
    int32_t num_edges,
    const int32_t* d_seeds_in,
    int32_t num_seeds_in,
    int64_t topk,
    float* d_wsums,
    int64_t* d_counts,
    int64_t* d_seed_offsets,
    int32_t* d_seeds_buf,
    int64_t* d_keys_buf,
    int64_t* d_keys_alt_buf,
    int64_t max_keys_buf_size,
    int32_t* d_first_buf,
    int32_t* d_second_buf,
    float* d_scores_buf,
    int64_t max_result_buf_size,
    void* d_sort_temp,
    size_t sort_temp_buf_size
) {
    InternalResult result = {nullptr, nullptr, nullptr, 0, false};

    const int32_t* d_seeds;
    int32_t num_seeds;
    if (d_seeds_in == nullptr || num_seeds_in == 0) {
        num_seeds = num_vertices;
        d_seeds = d_seeds_buf;
        thrust::device_ptr<int32_t> dp(d_seeds_buf);
        thrust::sequence(thrust::device, dp, dp + num_seeds);
    } else {
        d_seeds = d_seeds_in;
        num_seeds = num_seeds_in;
    }

    
    weight_sums_kernel<<<(num_vertices + 255) / 256, 256>>>(
        d_offsets, d_edge_weights, num_vertices, d_wsums);

    
    {
        int wpb = 8;
        int threads = wpb * 32;
        int grid = (num_seeds + wpb - 1) / wpb;
        count_pairs_kernel<<<grid, threads>>>(d_offsets, d_indices, d_seeds, num_seeds, d_counts);
    }

    
    cudaMemset(d_seed_offsets, 0, sizeof(int64_t));
    thrust::device_ptr<int64_t> dp_c(d_counts);
    thrust::device_ptr<int64_t> dp_o(d_seed_offsets + 1);
    thrust::inclusive_scan(thrust::device, dp_c, dp_c + num_seeds, dp_o);

    int64_t total_raw;
    cudaMemcpy(&total_raw, d_seed_offsets + num_seeds, sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (total_raw == 0) return result;

    
    int64_t* d_keys;
    bool owns_keys = false;
    if (total_raw <= max_keys_buf_size) {
        d_keys = d_keys_buf;
    } else {
        cudaMalloc(&d_keys, total_raw * sizeof(int64_t));
        owns_keys = true;
    }

    generate_pairs_kernel<<<num_seeds, 512>>>(
        d_offsets, d_indices, d_seeds, num_seeds, d_seed_offsets, d_keys);

    
    int64_t* d_sorted_keys = nullptr;
    bool owns_sorted = false;

    if (total_raw <= max_keys_buf_size) {
        int num_bits = 1;
        int32_t max_v = num_vertices - 1;
        while ((1 << num_bits) <= max_v && num_bits < 32) num_bits++;
        int total_bits = num_bits + 32;
        if (total_bits < 63) total_bits = 63;

        size_t temp_needed = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, temp_needed, d_keys, d_keys_alt_buf,
                                        (int)total_raw, 0, total_bits);

        if (temp_needed <= sort_temp_buf_size) {
            cub::DeviceRadixSort::SortKeys(d_sort_temp, temp_needed, d_keys, d_keys_alt_buf,
                                            (int)total_raw, 0, total_bits);
            d_sorted_keys = d_keys_alt_buf;
        } else {
            thrust::device_ptr<int64_t> dp(d_keys);
            thrust::sort(thrust::device, dp, dp + total_raw);
            d_sorted_keys = d_keys;
        }
    } else {
        thrust::device_ptr<int64_t> dp(d_keys);
        thrust::sort(thrust::device, dp, dp + total_raw);
        d_sorted_keys = d_keys;
    }

    
    {
        thrust::device_ptr<int64_t> dp(d_sorted_keys);
        auto end = thrust::unique(thrust::device, dp, dp + total_raw);
        total_raw = end - dp;
    }

    
    if (total_raw > 0) {
        int64_t last;
        cudaMemcpy(&last, d_sorted_keys + total_raw - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
        if (last == SENTINEL_KEY) total_raw--;
    }

    
    if (owns_keys && d_sorted_keys != d_keys) cudaFree(d_keys);

    if (total_raw == 0) {
        if (owns_sorted) cudaFree(d_sorted_keys);
        return result;
    }

    
    int32_t* d_first;
    int32_t* d_second;
    float* d_scores;
    bool owns_result_mem = false;

    if (total_raw <= max_result_buf_size) {
        d_first = d_first_buf;
        d_second = d_second_buf;
        d_scores = d_scores_buf;
    } else {
        cudaMalloc(&d_first, total_raw * sizeof(int32_t));
        cudaMalloc(&d_second, total_raw * sizeof(int32_t));
        cudaMalloc(&d_scores, total_raw * sizeof(float));
        owns_result_mem = true;
    }

    {
        int wpb = 8;
        int threads = wpb * 32;
        int64_t grid = (total_raw + wpb - 1) / wpb;
        jaccard_kernel<<<(int)grid, threads>>>(
            d_offsets, d_indices, d_edge_weights, d_wsums,
            d_sorted_keys, total_raw, d_first, d_second, d_scores);
    }

    if (owns_sorted) cudaFree(d_sorted_keys);
    if (owns_keys && d_sorted_keys == d_keys) cudaFree(d_keys);

    
    int64_t result_count = total_raw;
    if (topk >= 0 && topk < total_raw) {
        thrust::device_ptr<float> dp_sc(d_scores);
        thrust::device_ptr<int32_t> dp_f(d_first);
        thrust::device_ptr<int32_t> dp_s(d_second);
        auto vals = thrust::make_zip_iterator(thrust::make_tuple(dp_f, dp_s));
        thrust::sort_by_key(thrust::device, dp_sc, dp_sc + total_raw, vals, thrust::greater<float>());
        result_count = topk;
    }

    result.first = d_first;
    result.second = d_second;
    result.scores = d_scores;
    result.count = result_count;
    result.owns_result = owns_result_mem;
    return result;
}

}  

similarity_result_float_t jaccard_all_pairs_similarity_seg(const graph32_t& graph,
                                                           const float* edge_weights,
                                                           const int32_t* vertices,
                                                           std::size_t num_vertices,
                                                           std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n_vertices = graph.number_of_vertices;
    int32_t n_edges = graph.number_of_edges;

    const int32_t* d_seeds = vertices;
    int32_t n_seeds = (vertices != nullptr) ? (int32_t)num_vertices : 0;
    int64_t topk_val = topk.has_value() ? (int64_t)topk.value() : -1;

    InternalResult res = launch_jaccard_all_pairs(
        d_offsets, d_indices, edge_weights,
        n_vertices, n_edges,
        d_seeds, n_seeds, topk_val,
        cache.wsums_buf,
        cache.counts_buf,
        cache.offsets_buf,
        cache.seeds_buf,
        cache.keys_buf,
        cache.keys_alt_buf,
        MAX_KEYS,
        cache.first_buf,
        cache.second_buf,
        cache.scores_buf,
        MAX_RESULTS,
        cache.sort_temp_buf,
        SORT_TEMP
    );

    similarity_result_float_t result;
    result.count = (std::size_t)res.count;

    if (res.count == 0) {
        result.first = nullptr;
        result.second = nullptr;
        result.scores = nullptr;
        return result;
    }

    if (res.owns_result) {
        result.first = res.first;
        result.second = res.second;
        result.scores = res.scores;
    } else {
        cudaMalloc(&result.first, res.count * sizeof(int32_t));
        cudaMalloc(&result.second, res.count * sizeof(int32_t));
        cudaMalloc(&result.scores, res.count * sizeof(float));
        cudaMemcpyAsync(result.first, res.first, res.count * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(result.second, res.second, res.count * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(result.scores, res.scores, res.count * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return result;
}

}  
