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
#include <climits>
#include <optional>
#include <vector>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* seeds = nullptr;
    int64_t seeds_cap = 0;

    int64_t* counts = nullptr;
    int64_t counts_cap = 0;

    int64_t* seed_offs = nullptr;
    int64_t seed_offs_cap = 0;

    int64_t* keys = nullptr;
    int64_t keys_cap = 0;

    int64_t* sort_out = nullptr;
    int64_t sort_out_cap = 0;

    uint8_t* sort_temp = nullptr;
    size_t sort_temp_cap = 0;

    uint8_t* unique_temp = nullptr;
    size_t unique_temp_cap = 0;

    int32_t* num_sel = nullptr;

    float* scores_buf = nullptr;
    int64_t scores_buf_cap = 0;

    uint8_t* topk_temp = nullptr;
    size_t topk_temp_cap = 0;

    float* scores_sorted = nullptr;
    int64_t scores_sorted_cap = 0;

    int64_t* keys_sorted = nullptr;
    int64_t keys_sorted_cap = 0;

    Cache() {
        cudaMalloc(&num_sel, sizeof(int32_t));
    }

    ~Cache() override {
        if (seeds) cudaFree(seeds);
        if (counts) cudaFree(counts);
        if (seed_offs) cudaFree(seed_offs);
        if (keys) cudaFree(keys);
        if (sort_out) cudaFree(sort_out);
        if (sort_temp) cudaFree(sort_temp);
        if (unique_temp) cudaFree(unique_temp);
        if (num_sel) cudaFree(num_sel);
        if (scores_buf) cudaFree(scores_buf);
        if (topk_temp) cudaFree(topk_temp);
        if (scores_sorted) cudaFree(scores_sorted);
        if (keys_sorted) cudaFree(keys_sorted);
    }

    void ensure_seeds(int64_t n) {
        if (seeds_cap < n) { if (seeds) cudaFree(seeds); cudaMalloc(&seeds, n * sizeof(int32_t)); seeds_cap = n; }
    }
    void ensure_counts(int64_t n) {
        if (counts_cap < n) { if (counts) cudaFree(counts); cudaMalloc(&counts, n * sizeof(int64_t)); counts_cap = n; }
    }
    void ensure_seed_offs(int64_t n) {
        if (seed_offs_cap < n) { if (seed_offs) cudaFree(seed_offs); cudaMalloc(&seed_offs, n * sizeof(int64_t)); seed_offs_cap = n; }
    }
    void ensure_keys(int64_t n) {
        if (keys_cap < n) { if (keys) cudaFree(keys); cudaMalloc(&keys, n * sizeof(int64_t)); keys_cap = n; }
    }
    void ensure_sort_out(int64_t n) {
        if (sort_out_cap < n) { if (sort_out) cudaFree(sort_out); cudaMalloc(&sort_out, n * sizeof(int64_t)); sort_out_cap = n; }
    }
    void ensure_sort_temp(size_t n) {
        if (sort_temp_cap < n) { if (sort_temp) cudaFree(sort_temp); cudaMalloc(&sort_temp, n); sort_temp_cap = n; }
    }
    void ensure_unique_temp(size_t n) {
        if (unique_temp_cap < n) { if (unique_temp) cudaFree(unique_temp); cudaMalloc(&unique_temp, n); unique_temp_cap = n; }
    }
    void ensure_scores_buf(int64_t n) {
        if (scores_buf_cap < n) { if (scores_buf) cudaFree(scores_buf); cudaMalloc(&scores_buf, n * sizeof(float)); scores_buf_cap = n; }
    }
    void ensure_topk_temp(size_t n) {
        if (topk_temp_cap < n) { if (topk_temp) cudaFree(topk_temp); cudaMalloc(&topk_temp, n); topk_temp_cap = n; }
    }
    void ensure_scores_sorted(int64_t n) {
        if (scores_sorted_cap < n) { if (scores_sorted) cudaFree(scores_sorted); cudaMalloc(&scores_sorted, n * sizeof(float)); scores_sorted_cap = n; }
    }
    void ensure_keys_sorted(int64_t n) {
        if (keys_sorted_cap < n) { if (keys_sorted) cudaFree(keys_sorted); cudaMalloc(&keys_sorted, n * sizeof(int64_t)); keys_sorted_cap = n; }
    }
};



__global__ void iota_kernel(int32_t* arr, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = idx;
}

__global__ void count_candidates_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ counts
) {
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int32_t u = seeds[sid];
    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int64_t cnt = 0;
    for (int32_t i = u_start + threadIdx.x; i < u_end; i += blockDim.x) {
        int32_t w = indices[i];
        cnt += (int64_t)(offsets[w + 1] - offsets[w]);
    }
    typedef cub::BlockReduce<int64_t, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    int64_t total = BR(tmp).Sum(cnt);
    if (threadIdx.x == 0) counts[sid] = total;
}

__global__ void generate_candidates_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t stride,
    const int64_t* __restrict__ seed_offsets,
    int64_t* __restrict__ keys
) {
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int32_t u = seeds[sid];
    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int64_t base = seed_offsets[sid];
    int64_t write_pos = 0;
    for (int32_t ni = 0; ni < (u_end - u_start); ni++) {
        int32_t w = indices[u_start + ni];
        int32_t w_start = offsets[w], w_end = offsets[w + 1];
        int32_t w_deg = w_end - w_start;
        for (int32_t j = threadIdx.x; j < w_deg; j += blockDim.x) {
            int32_t v = indices[w_start + j];
            keys[base + write_pos + j] = (v == u) ? INT64_MAX : ((int64_t)sid * stride + v);
        }
        write_pos += w_deg;
    }
}

__global__ void compute_overlap_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int64_t* __restrict__ unique_keys,
    int64_t num_pairs, int64_t stride,
    const int32_t* __restrict__ seeds,
    float* __restrict__ scores,
    int32_t* __restrict__ out_u,
    int32_t* __restrict__ out_v
) {
    int warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_pairs) return;

    int64_t key = unique_keys[warp_id];
    int32_t sid = (int32_t)(key / stride);
    int32_t v = (int32_t)(key % stride);
    int32_t u = seeds[sid];

    if (out_u && lane == 0) out_u[warp_id] = u;
    if (out_v && lane == 0) out_v[warp_id] = v;

    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int32_t v_start = offsets[v], v_end = offsets[v + 1];
    int32_t u_deg = u_end - u_start, v_deg = v_end - v_start;

    float w_deg_u = 0.0f;
    for (int i = u_start + lane; i < u_end; i += 32) w_deg_u += weights[i];
    for (int off = 16; off > 0; off >>= 1) w_deg_u += __shfl_down_sync(0xffffffff, w_deg_u, off);
    w_deg_u = __shfl_sync(0xffffffff, w_deg_u, 0);

    float w_deg_v = 0.0f;
    for (int i = v_start + lane; i < v_end; i += 32) w_deg_v += weights[i];
    for (int off = 16; off > 0; off >>= 1) w_deg_v += __shfl_down_sync(0xffffffff, w_deg_v, off);
    w_deg_v = __shfl_sync(0xffffffff, w_deg_v, 0);

    const int32_t* short_idx, *long_idx;
    const float* short_w, *long_w;
    int32_t short_len, long_len;
    if (u_deg <= v_deg) {
        short_idx = indices + u_start; short_w = weights + u_start; short_len = u_deg;
        long_idx = indices + v_start; long_w = weights + v_start; long_len = v_deg;
    } else {
        short_idx = indices + v_start; short_w = weights + v_start; short_len = v_deg;
        long_idx = indices + u_start; long_w = weights + u_start; long_len = u_deg;
    }

    float intersect = 0.0f;
    for (int i = lane; i < short_len; i += 32) {
        int32_t target = __ldg(&short_idx[i]);
        float sw = __ldg(&short_w[i]);
        int lo = 0, hi = long_len;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (__ldg(&long_idx[mid]) < target) lo = mid + 1;
            else hi = mid;
        }
        if (lo < long_len && __ldg(&long_idx[lo]) == target)
            intersect += fminf(sw, __ldg(&long_w[lo]));
    }
    for (int off = 16; off > 0; off >>= 1) intersect += __shfl_down_sync(0xffffffff, intersect, off);

    if (lane == 0) {
        float min_deg = fminf(w_deg_u, w_deg_v);
        scores[warp_id] = (min_deg > 0.0f) ? intersect / min_deg : 0.0f;
    }
}

__global__ void compute_overlap_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int64_t* __restrict__ unique_keys,
    int64_t num_pairs, int64_t stride,
    const int32_t* __restrict__ seeds,
    float* __restrict__ scores,
    int32_t* __restrict__ out_u,
    int32_t* __restrict__ out_v
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int64_t key = unique_keys[idx];
    int32_t sid = (int32_t)(key / stride);
    int32_t v = (int32_t)(key % stride);
    int32_t u = seeds[sid];

    if (out_u) out_u[idx] = u;
    if (out_v) out_v[idx] = v;

    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int32_t v_start = offsets[v], v_end = offsets[v + 1];

    float w_deg_u = 0.0f;
    for (int i = u_start; i < u_end; i++) w_deg_u += weights[i];
    float w_deg_v = 0.0f;
    for (int i = v_start; i < v_end; i++) w_deg_v += weights[i];

    float intersect = 0.0f;
    int i = u_start, j = v_start;
    while (i < u_end && j < v_end) {
        int32_t a = indices[i], b = indices[j];
        if (a == b) { intersect += fminf(weights[i], weights[j]); i++; j++; }
        else if (a < b) i++;
        else j++;
    }

    float min_deg = fminf(w_deg_u, w_deg_v);
    scores[idx] = (min_deg > 0.0f) ? intersect / min_deg : 0.0f;
}

__global__ void decode_keys_kernel(
    const int64_t* __restrict__ keys, int64_t num_pairs,
    int64_t stride, const int32_t* __restrict__ seeds,
    int32_t* __restrict__ pair_u, int32_t* __restrict__ pair_v
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    int64_t key = keys[idx];
    pair_u[idx] = seeds[(int32_t)(key / stride)];
    pair_v[idx] = (int32_t)(key % stride);
}

__global__ void check_sentinel_kernel(int64_t* data, int32_t* count) {
    int32_t n = *count;
    if (n > 0 && data[n - 1] == INT64_MAX) {
        *count = n - 1;
    }
}



static int compute_end_bit(int64_t max_key) {
    if (max_key <= 0) return 1;
    int bits = 0; int64_t v = max_key;
    while (v > 0) { bits++; v >>= 1; }
    return bits + 1;
}

static size_t get_sort_temp_size(int64_t n, int end_bit) {
    if (n == 0) return 256;
    void* d = nullptr; size_t s = 0;
    cub::DeviceRadixSort::SortKeys(d, s, (int64_t*)nullptr, (int64_t*)nullptr, (int)n, 0, end_bit);
    return s;
}

static size_t get_unique_temp_size(int64_t n) {
    if (n == 0) return 256;
    void* d = nullptr; size_t s = 0;
    cub::DeviceSelect::Unique(d, s, (int64_t*)nullptr, (int64_t*)nullptr, (int*)nullptr, (int)n);
    return s;
}

static size_t get_topk_sort_temp_size(int64_t n) {
    if (n == 0) return 256;
    void* d = nullptr; size_t s = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        d, s, (float*)nullptr, (float*)nullptr, (int64_t*)nullptr, (int64_t*)nullptr, (int)n);
    return s;
}

}  

similarity_result_float_t overlap_all_pairs_similarity_seg(const graph32_t& graph,
                                                           const float* edge_weights,
                                                           const int32_t* vertices,
                                                           std::size_t num_vertices,
                                                           std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n_verts = graph.number_of_vertices;
    const float* d_weights = edge_weights;

    int32_t actual_num_seeds;
    const int32_t* d_seeds;
    if (vertices != nullptr) {
        d_seeds = vertices;
        actual_num_seeds = (int32_t)num_vertices;
    } else {
        cache.ensure_seeds(n_verts);
        iota_kernel<<<(n_verts + 255) / 256, 256>>>(cache.seeds, n_verts);
        d_seeds = cache.seeds;
        actual_num_seeds = n_verts;
    }

    if (actual_num_seeds == 0)
        return {nullptr, nullptr, nullptr, 0};

    
    cache.ensure_counts(actual_num_seeds);
    if (actual_num_seeds > 0)
        count_candidates_kernel<<<actual_num_seeds, 256>>>(d_offsets, d_indices, d_seeds, actual_num_seeds, cache.counts);

    
    std::vector<int64_t> h_counts(actual_num_seeds);
    cudaMemcpy(h_counts.data(), cache.counts, actual_num_seeds * sizeof(int64_t), cudaMemcpyDeviceToHost);

    std::vector<int64_t> h_offsets(actual_num_seeds);
    int64_t total_candidates = 0;
    for (int i = 0; i < actual_num_seeds; i++) {
        h_offsets[i] = total_candidates;
        total_candidates += h_counts[i];
    }

    if (total_candidates == 0)
        return {nullptr, nullptr, nullptr, 0};

    cache.ensure_seed_offs(actual_num_seeds);
    cudaMemcpy(cache.seed_offs, h_offsets.data(), actual_num_seeds * sizeof(int64_t), cudaMemcpyHostToDevice);

    
    int64_t stride = (int64_t)n_verts + 1;
    int64_t max_key = (int64_t)(actual_num_seeds - 1) * stride + n_verts;
    int end_bit = compute_end_bit(max_key);
    if (end_bit > 64) end_bit = 64;

    
    cache.ensure_keys(total_candidates);
    cache.ensure_sort_out(total_candidates);
    size_t sort_ts = get_sort_temp_size(total_candidates, end_bit);
    cache.ensure_sort_temp(sort_ts + 256);

    
    if (actual_num_seeds > 0)
        generate_candidates_kernel<<<actual_num_seeds, 256>>>(
            d_offsets, d_indices, d_seeds, actual_num_seeds,
            stride, cache.seed_offs, cache.keys);

    
    if (total_candidates > 0) {
        cub::DeviceRadixSort::SortKeys(cache.sort_temp, sort_ts,
            cache.keys, cache.sort_out, (int)total_candidates, 0, end_bit);
    }

    
    size_t unique_ts = get_unique_temp_size(total_candidates);
    cache.ensure_unique_temp(unique_ts + 256);
    if (total_candidates > 0) {
        cub::DeviceSelect::Unique(cache.unique_temp, unique_ts,
            cache.sort_out, cache.keys, cache.num_sel, (int)total_candidates);
    }

    
    check_sentinel_kernel<<<1, 1>>>(cache.keys, cache.num_sel);

    
    int32_t num_unique_i32;
    cudaMemcpy(&num_unique_i32, cache.num_sel, sizeof(int32_t), cudaMemcpyDeviceToHost);
    int64_t num_unique = num_unique_i32;

    if (num_unique == 0)
        return {nullptr, nullptr, nullptr, 0};

    int64_t* unique_keys = cache.keys;

    bool need_topk = topk.has_value() && topk.value() < (std::size_t)num_unique;

    if (!need_topk) {
        
        int32_t* first_out;
        int32_t* second_out;
        float* scores_out;
        cudaMalloc(&first_out, num_unique * sizeof(int32_t));
        cudaMalloc(&second_out, num_unique * sizeof(int32_t));
        cudaMalloc(&scores_out, num_unique * sizeof(float));

        int threads = 256;
        int warps_per_block = threads / 32;
        int grid = (int)((num_unique + warps_per_block - 1) / warps_per_block);
        compute_overlap_warp_kernel<<<grid, threads>>>(
            d_offsets, d_indices, d_weights,
            unique_keys, num_unique, stride, d_seeds,
            scores_out, first_out, second_out);

        return {first_out, second_out, scores_out, (std::size_t)num_unique};
    } else {
        int64_t output_count = (int64_t)topk.value();

        cache.ensure_scores_buf(num_unique);

        int threads = 256;
        int warps_per_block = threads / 32;
        int grid = (int)((num_unique + warps_per_block - 1) / warps_per_block);
        compute_overlap_warp_kernel<<<grid, threads>>>(
            d_offsets, d_indices, d_weights,
            unique_keys, num_unique, stride, d_seeds,
            cache.scores_buf, nullptr, nullptr);

        size_t topk_ts = get_topk_sort_temp_size(num_unique);
        cache.ensure_topk_temp(topk_ts + 256);
        cache.ensure_scores_sorted(num_unique);
        cache.ensure_keys_sorted(num_unique);

        cub::DeviceRadixSort::SortPairsDescending(
            cache.topk_temp, topk_ts,
            cache.scores_buf, cache.scores_sorted,
            unique_keys, cache.keys_sorted,
            (int)num_unique);

        
        int32_t* first_out;
        int32_t* second_out;
        float* scores_out;
        cudaMalloc(&first_out, output_count * sizeof(int32_t));
        cudaMalloc(&second_out, output_count * sizeof(int32_t));
        cudaMalloc(&scores_out, output_count * sizeof(float));

        decode_keys_kernel<<<(int)((output_count + 255) / 256), 256>>>(
            cache.keys_sorted, output_count, stride, d_seeds,
            first_out, second_out);

        cudaMemcpyAsync(scores_out, cache.scores_sorted,
            output_count * sizeof(float), cudaMemcpyDeviceToDevice);

        return {first_out, second_out, scores_out, (std::size_t)output_count};
    }
}

}  
