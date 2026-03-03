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
#include <cstdint>
#include <cstddef>
#include <optional>

namespace aai {

namespace {

struct Cache : Cacheable {};

static __device__ __forceinline__ uint64_t encode_pair(int32_t u, int32_t v) {
    return ((uint64_t)(uint32_t)u << 32) | (uint64_t)(uint32_t)v;
}
static __device__ __forceinline__ int32_t decode_first(uint64_t key) {
    return (int32_t)(key >> 32);
}
static __device__ __forceinline__ int32_t decode_second(uint64_t key) {
    return (int32_t)(key & 0xFFFFFFFF);
}

__global__ void count_pairs_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds, int32_t num_seeds,
    int64_t* __restrict__ counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;
    int32_t u = seeds[idx];
    int64_t count = 0;
    for (int32_t i = offsets[u]; i < offsets[u + 1]; i++)
        count += (int64_t)(offsets[indices[i] + 1] - offsets[indices[i]]);
    counts[idx] = count;
}

__global__ void expand_pairs_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds, int32_t num_seeds,
    const int64_t* __restrict__ pair_offsets, uint64_t* __restrict__ pair_keys) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;
    int32_t u = seeds[seed_idx];
    int64_t out_base = pair_offsets[seed_idx];
    int64_t local_offset = 0;
    for (int32_t i = offsets[u]; i < offsets[u + 1]; i++) {
        int32_t w = indices[i];
        int32_t w_start = offsets[w], w_deg = offsets[w + 1] - w_start;
        for (int32_t j = threadIdx.x; j < w_deg; j += blockDim.x)
            pair_keys[out_base + local_offset + j] = encode_pair(u, indices[w_start + j]);
        local_offset += w_deg;
    }
}

__global__ void intersect_and_score_kernel(
    const uint64_t* __restrict__ unique_keys, int32_t num_unique,
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    int32_t* __restrict__ out_first, int32_t* __restrict__ out_second,
    float* __restrict__ out_scores, int32_t* __restrict__ valid_count) {
    int pair_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (pair_idx >= num_unique) return;

    uint64_t key = unique_keys[pair_idx];
    int32_t u = decode_first(key);
    int32_t v = decode_second(key);
    if (u == v) return;

    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int32_t v_start = offsets[v], v_end = offsets[v + 1];
    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;
    if (deg_u + deg_v == 0) return;

    const int32_t* small_ptr = (deg_u <= deg_v) ? (indices + u_start) : (indices + v_start);
    int32_t small_size = (deg_u <= deg_v) ? deg_u : deg_v;
    const int32_t* large_ptr = (deg_u <= deg_v) ? (indices + v_start) : (indices + u_start);
    int32_t large_size = (deg_u <= deg_v) ? deg_v : deg_u;

    int32_t count = 0;
    for (int32_t i = lane; i < small_size; i += 32) {
        int32_t target = small_ptr[i];
        if (i > 0 && small_ptr[i - 1] == target) continue;
        int32_t cnt_small = 1;
        while (i + cnt_small < small_size && small_ptr[i + cnt_small] == target) cnt_small++;
        int32_t lo = 0, hi = large_size;
        while (lo < hi) {
            int32_t mid = lo + ((hi - lo) >> 1);
            if (large_ptr[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        if (lo < large_size && large_ptr[lo] == target) {
            int32_t cnt_large = 1;
            while (lo + cnt_large < large_size && large_ptr[lo + cnt_large] == target) cnt_large++;
            count += (cnt_small < cnt_large) ? cnt_small : cnt_large;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        count += __shfl_down_sync(0xFFFFFFFF, count, offset);

    if (lane == 0 && count > 0) {
        float score = 2.0f * (float)count / (float)(deg_u + deg_v);
        int32_t out_idx = atomicAdd(valid_count, 1);
        out_first[out_idx] = u;
        out_second[out_idx] = v;
        out_scores[out_idx] = score;
    }
}

__global__ void iota_kernel(int32_t* out, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = idx;
}

__global__ void gather_results_kernel(
    const int32_t* __restrict__ first_in, const int32_t* __restrict__ second_in,
    const float* __restrict__ scores_in, const int32_t* __restrict__ idx_map,
    int64_t count,
    int32_t* __restrict__ first_out, int32_t* __restrict__ second_out,
    float* __restrict__ scores_out) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int32_t src = idx_map[i];
    first_out[i] = first_in[src];
    second_out[i] = second_in[src];
    scores_out[i] = scores_in[src];
}

}  

similarity_result_float_t sorensen_all_pairs_similarity_seg(
    const graph32_t& graph,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk) {

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t graph_n = graph.number_of_vertices;

    
    int32_t* seeds_buf = nullptr;
    const int32_t* d_seeds;
    int32_t num_seeds;
    if (vertices != nullptr && num_vertices > 0) {
        d_seeds = vertices;
        num_seeds = (int32_t)num_vertices;
    } else {
        num_seeds = graph_n;
        cudaMalloc(&seeds_buf, (size_t)graph_n * sizeof(int32_t));
        iota_kernel<<<(graph_n + 255) / 256, 256>>>(seeds_buf, graph_n);
        d_seeds = seeds_buf;
    }

    if (num_seeds == 0) {
        if (seeds_buf) cudaFree(seeds_buf);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t* d_counts;
    cudaMalloc(&d_counts, (size_t)num_seeds * sizeof(int64_t));
    count_pairs_kernel<<<(num_seeds + 255) / 256, 256>>>(
        d_offsets, d_indices, d_seeds, num_seeds, d_counts);

    
    int64_t* d_pair_offsets;
    cudaMalloc(&d_pair_offsets, ((size_t)num_seeds + 1) * sizeof(int64_t));
    cudaMemsetAsync(d_pair_offsets, 0, sizeof(int64_t), 0);

    size_t scan_temp = 0;
    cub::DeviceScan::InclusiveSum(nullptr, scan_temp, d_counts,
        d_pair_offsets + 1, num_seeds);
    void* d_scan_buf;
    cudaMalloc(&d_scan_buf, scan_temp);
    cub::DeviceScan::InclusiveSum(d_scan_buf, scan_temp, d_counts,
        d_pair_offsets + 1, num_seeds);
    cudaFree(d_scan_buf);
    cudaFree(d_counts);

    int64_t total_pairs;
    cudaMemcpy(&total_pairs, d_pair_offsets + num_seeds,
               sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (total_pairs == 0) {
        if (seeds_buf) cudaFree(seeds_buf);
        cudaFree(d_pair_offsets);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    uint64_t* d_pair_keys;
    cudaMalloc(&d_pair_keys, (size_t)total_pairs * sizeof(uint64_t));
    expand_pairs_kernel<<<num_seeds, 256>>>(
        d_offsets, d_indices, d_seeds, num_seeds,
        d_pair_offsets, d_pair_keys);

    if (seeds_buf) cudaFree(seeds_buf);
    cudaFree(d_pair_offsets);

    
    int vbits = 1;
    { int32_t t = graph_n; while (t > 0) { t >>= 1; vbits++; } }
    int end_bit = 32 + vbits;
    if (end_bit > 64) end_bit = 64;

    uint64_t* d_sorted_keys;
    cudaMalloc(&d_sorted_keys, (size_t)total_pairs * sizeof(uint64_t));

    size_t sort_temp = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, sort_temp,
        d_pair_keys, d_sorted_keys, total_pairs, 0, end_bit);
    void* d_sort_buf;
    cudaMalloc(&d_sort_buf, sort_temp);
    cub::DeviceRadixSort::SortKeys(d_sort_buf, sort_temp,
        d_pair_keys, d_sorted_keys, total_pairs, 0, end_bit);
    cudaFree(d_sort_buf);
    cudaFree(d_pair_keys);

    
    uint64_t* d_unique_keys;
    cudaMalloc(&d_unique_keys, (size_t)total_pairs * sizeof(uint64_t));
    int32_t* d_num_unique;
    cudaMalloc(&d_num_unique, sizeof(int32_t));

    size_t unique_temp = 0;
    cub::DeviceSelect::Unique(nullptr, unique_temp,
        d_sorted_keys, d_unique_keys, d_num_unique, total_pairs);
    void* d_unique_buf;
    cudaMalloc(&d_unique_buf, unique_temp);
    cub::DeviceSelect::Unique(d_unique_buf, unique_temp,
        d_sorted_keys, d_unique_keys, d_num_unique, total_pairs);
    cudaFree(d_unique_buf);
    cudaFree(d_sorted_keys);

    int32_t h_num_unique;
    cudaMemcpy(&h_num_unique, d_num_unique, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_num_unique);

    if (h_num_unique == 0) {
        cudaFree(d_unique_keys);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t* d_result_first;
    int32_t* d_result_second;
    float* d_result_scores;
    int32_t* d_vcnt;
    cudaMalloc(&d_result_first, (size_t)h_num_unique * sizeof(int32_t));
    cudaMalloc(&d_result_second, (size_t)h_num_unique * sizeof(int32_t));
    cudaMalloc(&d_result_scores, (size_t)h_num_unique * sizeof(float));
    cudaMalloc(&d_vcnt, sizeof(int32_t));
    cudaMemsetAsync(d_vcnt, 0, sizeof(int32_t), 0);

    {
        int warps = h_num_unique;
        int tpb = 256;
        int blocks = (int)(((int64_t)warps * 32 + tpb - 1) / tpb);
        intersect_and_score_kernel<<<blocks, tpb>>>(
            d_unique_keys, h_num_unique, d_offsets, d_indices,
            d_result_first, d_result_second, d_result_scores, d_vcnt);
    }

    cudaFree(d_unique_keys);

    int32_t valid_count;
    cudaMemcpy(&valid_count, d_vcnt, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_vcnt);

    if (valid_count == 0) {
        cudaFree(d_result_first);
        cudaFree(d_result_second);
        cudaFree(d_result_scores);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    if (topk.has_value() && (std::size_t)valid_count > topk.value()) {
        int64_t topk_val = (int64_t)topk.value();

        int32_t* d_idx;
        cudaMalloc(&d_idx, (size_t)valid_count * sizeof(int32_t));
        iota_kernel<<<(valid_count + 255) / 256, 256>>>(d_idx, valid_count);

        float* d_ss;
        int32_t* d_si;
        cudaMalloc(&d_ss, (size_t)valid_count * sizeof(float));
        cudaMalloc(&d_si, (size_t)valid_count * sizeof(int32_t));

        size_t topk_temp = 0;
        cub::DeviceRadixSort::SortPairsDescending(nullptr, topk_temp,
            d_result_scores, d_ss, d_idx, d_si, valid_count, 0, 32);
        void* d_topk_buf;
        cudaMalloc(&d_topk_buf, topk_temp);
        cub::DeviceRadixSort::SortPairsDescending(d_topk_buf, topk_temp,
            d_result_scores, d_ss, d_idx, d_si, valid_count, 0, 32);
        cudaFree(d_topk_buf);
        cudaFree(d_idx);

        int32_t* d_out_first;
        int32_t* d_out_second;
        float* d_out_scores;
        cudaMalloc(&d_out_first, (size_t)topk_val * sizeof(int32_t));
        cudaMalloc(&d_out_second, (size_t)topk_val * sizeof(int32_t));
        cudaMalloc(&d_out_scores, (size_t)topk_val * sizeof(float));

        gather_results_kernel<<<((int)topk_val + 255) / 256, 256>>>(
            d_result_first, d_result_second, d_result_scores,
            d_si, topk_val, d_out_first, d_out_second, d_out_scores);

        cudaFree(d_result_first);
        cudaFree(d_result_second);
        cudaFree(d_result_scores);
        cudaFree(d_ss);
        cudaFree(d_si);

        return {d_out_first, d_out_second, d_out_scores, (std::size_t)topk_val};
    }

    
    int32_t* d_out_first;
    int32_t* d_out_second;
    float* d_out_scores;
    cudaMalloc(&d_out_first, (size_t)valid_count * sizeof(int32_t));
    cudaMalloc(&d_out_second, (size_t)valid_count * sizeof(int32_t));
    cudaMalloc(&d_out_scores, (size_t)valid_count * sizeof(float));

    cudaMemcpy(d_out_first, d_result_first,
               valid_count * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_out_second, d_result_second,
               valid_count * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_out_scores, d_result_scores,
               valid_count * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaFree(d_result_first);
    cudaFree(d_result_second);
    cudaFree(d_result_scores);

    return {d_out_first, d_out_second, d_out_scores, (std::size_t)valid_count};
}

}  
