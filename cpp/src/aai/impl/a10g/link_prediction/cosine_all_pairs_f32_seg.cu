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
#include <optional>

namespace aai {

namespace {

template <typename T>
void ensure_buf(T*& ptr, int64_t& cap, int64_t needed) {
    if (cap < needed) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, needed * sizeof(T));
        cap = needed;
    }
}

void ensure_buf_bytes(void*& ptr, size_t& cap, size_t needed) {
    if (cap < needed) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, needed);
        cap = needed;
    }
}

struct Cache : Cacheable {
    int32_t* d_count = nullptr;

    int32_t* seeds_buf = nullptr;
    int64_t seeds_cap = 0;

    int32_t* seed_degrees = nullptr;
    int64_t seed_degrees_cap = 0;

    int64_t* seed_deg_prefix = nullptr;
    int64_t seed_deg_prefix_cap = 0;

    void* cub_temp = nullptr;
    size_t cub_temp_cap = 0;

    int64_t* expansions = nullptr;
    int64_t expansions_cap = 0;

    int64_t* write_offsets = nullptr;
    int64_t write_offsets_cap = 0;

    uint64_t* keys_buf = nullptr;
    int64_t keys_cap = 0;

    uint64_t* sorted_keys = nullptr;
    int64_t sorted_keys_cap = 0;

    int32_t* pair_u = nullptr;
    int64_t pair_u_cap = 0;

    int32_t* pair_v = nullptr;
    int64_t pair_v_cap = 0;

    float* scores_buf = nullptr;
    int64_t scores_cap = 0;

    int32_t* idx_buf = nullptr;
    int64_t idx_cap = 0;

    float* sorted_scores_buf = nullptr;
    int64_t sorted_scores_cap = 0;

    int32_t* sorted_idx = nullptr;
    int64_t sorted_idx_cap = 0;

    Cache() {
        cudaMalloc(&d_count, sizeof(int32_t));
    }

    ~Cache() override {
        if (d_count) cudaFree(d_count);
        if (seeds_buf) cudaFree(seeds_buf);
        if (seed_degrees) cudaFree(seed_degrees);
        if (seed_deg_prefix) cudaFree(seed_deg_prefix);
        if (cub_temp) cudaFree(cub_temp);
        if (expansions) cudaFree(expansions);
        if (write_offsets) cudaFree(write_offsets);
        if (keys_buf) cudaFree(keys_buf);
        if (sorted_keys) cudaFree(sorted_keys);
        if (pair_u) cudaFree(pair_u);
        if (pair_v) cudaFree(pair_v);
        if (scores_buf) cudaFree(scores_buf);
        if (idx_buf) cudaFree(idx_buf);
        if (sorted_scores_buf) cudaFree(sorted_scores_buf);
        if (sorted_idx) cudaFree(sorted_idx);
    }
};



__global__ void gen_seq_kernel(int32_t* out, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = idx;
}

__global__ void compute_seed_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int32_t* __restrict__ degrees
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_seeds) return;
    int32_t u = seeds[i];
    degrees[i] = offsets[u + 1] - offsets[u];
}

__global__ void fill_neighbor_expansions_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    const int64_t* __restrict__ seed_deg_prefix,
    int32_t num_seeds,
    int64_t total_level1,
    int64_t* __restrict__ expansions
) {
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= total_level1) return;

    
    int lo = 0, hi = num_seeds;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (seed_deg_prefix[mid + 1] <= j) lo = mid + 1;
        else hi = mid;
    }
    int seed_idx = lo;
    int neighbor_pos = (int)(j - seed_deg_prefix[seed_idx]);

    int32_t u = seeds[seed_idx];
    int32_t k = indices[offsets[u] + neighbor_pos];
    expansions[j] = (int64_t)(offsets[k + 1] - offsets[k]);
}

__global__ void flat_expand_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    const int64_t* __restrict__ seed_deg_prefix,
    const int64_t* __restrict__ write_offsets,
    int32_t num_seeds,
    int64_t total_level1,
    uint64_t* __restrict__ out_keys,
    int32_t num_vertices
) {
    int64_t work_item = blockIdx.x;
    if (work_item >= total_level1) return;

    
    int lo = 0, hi = num_seeds;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (seed_deg_prefix[mid + 1] <= work_item) lo = mid + 1;
        else hi = mid;
    }
    int seed_idx = lo;
    int neighbor_pos = (int)(work_item - seed_deg_prefix[seed_idx]);

    int32_t u = seeds[seed_idx];
    int32_t k = indices[offsets[u] + neighbor_pos];
    int32_t k_start = offsets[k];
    int32_t k_end = offsets[k + 1];
    int32_t k_deg = k_end - k_start;
    int64_t base = write_offsets[work_item];

    
    for (int32_t j = threadIdx.x; j < k_deg; j += blockDim.x) {
        int32_t v = indices[k_start + j];
        
        out_keys[base + j] = (uint64_t)seed_idx * (uint64_t)num_vertices + (uint64_t)v;
    }
}

__global__ void unique_filter_kernel(
    const uint64_t* __restrict__ sorted_keys,
    int64_t n,
    int32_t num_vertices,
    const int32_t* __restrict__ seeds,
    int32_t* __restrict__ out_u,
    int32_t* __restrict__ out_v,
    int32_t* __restrict__ d_count
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t key = sorted_keys[idx];

    
    if (idx > 0 && sorted_keys[idx - 1] == key) return;

    
    uint64_t nv = (uint64_t)num_vertices;
    int32_t seed_idx = (int32_t)(key / nv);
    int32_t v = (int32_t)(key % nv);
    int32_t u = seeds[seed_idx];

    
    if (u == v) return;

    int32_t pos = atomicAdd(d_count, 1);
    out_u[pos] = u;
    out_v[pos] = v;
}



#define MAX_CACHED_DEGREE 8192

__global__ void compute_cosine_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ pair_u,
    const int32_t* __restrict__ pair_v,
    int32_t num_pairs,
    float* __restrict__ scores
) {
    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_idx >= num_pairs) return;

    int32_t u = pair_u[warp_idx];
    int32_t v = pair_v[warp_idx];

    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int32_t v_start = offsets[v], v_end = offsets[v + 1];
    int32_t u_len = u_end - u_start;
    int32_t v_len = v_end - v_start;

    
    const int32_t* small_idx;
    const float* small_wt;
    int32_t small_len;
    const int32_t* large_idx;
    const float* large_wt;
    bool u_is_small;

    if (u_len <= v_len) {
        small_idx = indices + u_start; small_wt = weights + u_start; small_len = u_len;
        large_idx = indices + v_start; large_wt = weights + v_start;
        u_is_small = true;
    } else {
        small_idx = indices + v_start; small_wt = weights + v_start; small_len = v_len;
        large_idx = indices + u_start; large_wt = weights + u_start;
        u_is_small = false;
    }

    int32_t large_len = u_is_small ? v_len : u_len;
    float my_dot = 0.0f, my_nu = 0.0f, my_nv = 0.0f;

    for (int32_t i = lane; i < small_len; i += 32) {
        int32_t k = small_idx[i];
        float w_small = small_wt[i];

        
        int32_t lo = 0, hi = large_len;
        while (lo < hi) {
            int32_t mid = lo + (hi - lo) / 2;
            if (large_idx[mid] < k) lo = mid + 1;
            else hi = mid;
        }

        if (lo < large_len && large_idx[lo] == k) {
            float w_large = large_wt[lo];
            float wu = u_is_small ? w_small : w_large;
            float wv = u_is_small ? w_large : w_small;
            my_dot += wu * wv;
            my_nu += wu * wu;
            my_nv += wv * wv;
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        my_dot += __shfl_down_sync(0xFFFFFFFF, my_dot, offset);
        my_nu += __shfl_down_sync(0xFFFFFFFF, my_nu, offset);
        my_nv += __shfl_down_sync(0xFFFFFFFF, my_nv, offset);
    }

    if (lane == 0) {
        float denom = sqrtf(my_nu) * sqrtf(my_nv);
        scores[warp_idx] = (denom > 0.0f) ? (my_dot / denom) : 0.0f;
    }
}

__global__ void gather_topk_kernel(
    const int32_t* __restrict__ pair_u,
    const int32_t* __restrict__ pair_v,
    const int32_t* __restrict__ sorted_indices,
    int32_t count,
    int32_t* __restrict__ out_u,
    int32_t* __restrict__ out_v
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int32_t src = sorted_indices[idx];
    out_u[idx] = pair_u[src];
    out_v[idx] = pair_v[src];
}

}  

similarity_result_float_t cosine_all_pairs_similarity_seg(const graph32_t& graph,
                                                          const float* edge_weights,
                                                          const int32_t* vertices,
                                                          std::size_t num_vertices,
                                                          std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t nv = graph.number_of_vertices;
    const float* d_weights = edge_weights;

    
    const int32_t* d_seeds;
    int32_t num_seeds;
    if (vertices != nullptr && num_vertices > 0) {
        d_seeds = vertices;
        num_seeds = (int32_t)num_vertices;
    } else {
        ensure_buf(cache.seeds_buf, cache.seeds_cap, (int64_t)nv);
        gen_seq_kernel<<<(nv + 255) / 256, 256>>>(cache.seeds_buf, nv);
        d_seeds = cache.seeds_buf;
        num_seeds = nv;
    }

    if (num_seeds == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    

    
    ensure_buf(cache.seed_degrees, cache.seed_degrees_cap, (int64_t)(num_seeds + 1));
    cudaMemsetAsync(cache.seed_degrees + num_seeds, 0, sizeof(int32_t));
    compute_seed_degrees_kernel<<<(num_seeds + 255) / 256, 256>>>(d_offsets, d_seeds, num_seeds, cache.seed_degrees);

    
    ensure_buf(cache.seed_deg_prefix, cache.seed_deg_prefix_cap, (int64_t)(num_seeds + 1));
    size_t scan1_sz = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan1_sz, (int32_t*)nullptr, (int64_t*)nullptr, (int)(num_seeds + 1));
    ensure_buf_bytes(cache.cub_temp, cache.cub_temp_cap, scan1_sz);
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, scan1_sz, cache.seed_degrees, cache.seed_deg_prefix, (int)(num_seeds + 1));

    
    int64_t total_level1;
    cudaMemcpy(&total_level1, cache.seed_deg_prefix + num_seeds, sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (total_level1 == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    ensure_buf(cache.expansions, cache.expansions_cap, total_level1 + 1);
    cudaMemsetAsync(cache.expansions + total_level1, 0, sizeof(int64_t));
    {
        int block = 256;
        int grid = (int)((total_level1 + block - 1) / block);
        fill_neighbor_expansions_kernel<<<grid, block>>>(d_offsets, d_indices, d_seeds,
            cache.seed_deg_prefix, num_seeds, total_level1, cache.expansions);
    }

    
    ensure_buf(cache.write_offsets, cache.write_offsets_cap, total_level1 + 1);
    size_t scan2_sz = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan2_sz, (int64_t*)nullptr, (int64_t*)nullptr, (int64_t)(total_level1 + 1));
    ensure_buf_bytes(cache.cub_temp, cache.cub_temp_cap, scan2_sz);
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, scan2_sz, cache.expansions, cache.write_offsets, (int64_t)(total_level1 + 1));

    
    int64_t total_expansion;
    cudaMemcpy(&total_expansion, cache.write_offsets + total_level1, sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (total_expansion == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    ensure_buf(cache.keys_buf, cache.keys_cap, total_expansion);
    flat_expand_kernel<<<(int)total_level1, 256>>>(d_offsets, d_indices, d_seeds,
        cache.seed_deg_prefix, cache.write_offsets,
        num_seeds, total_level1, cache.keys_buf, nv);

    

    
    int64_t max_key = (int64_t)(num_seeds - 1) * nv + (nv - 1);
    int end_bit = 1;
    while ((1LL << end_bit) <= max_key && end_bit < 64) end_bit++;

    ensure_buf(cache.sorted_keys, cache.sorted_keys_cap, total_expansion);
    size_t sort_sz = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, sort_sz, (uint64_t*)nullptr, (uint64_t*)nullptr, (int64_t)total_expansion, 0, end_bit);
    ensure_buf_bytes(cache.cub_temp, cache.cub_temp_cap, sort_sz);
    cub::DeviceRadixSort::SortKeys(cache.cub_temp, sort_sz, cache.keys_buf, cache.sorted_keys,
        (int64_t)total_expansion, 0, end_bit);

    
    ensure_buf(cache.pair_u, cache.pair_u_cap, total_expansion);
    ensure_buf(cache.pair_v, cache.pair_v_cap, total_expansion);
    cudaMemsetAsync(cache.d_count, 0, sizeof(int32_t));
    {
        int block = 256;
        int grid = (int)((total_expansion + block - 1) / block);
        unique_filter_kernel<<<grid, block>>>(cache.sorted_keys, total_expansion,
            nv, d_seeds, cache.pair_u, cache.pair_v, cache.d_count);
    }

    int32_t num_pairs;
    cudaMemcpy(&num_pairs, cache.d_count, sizeof(int32_t), cudaMemcpyDeviceToHost);

    if (num_pairs == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    ensure_buf(cache.scores_buf, cache.scores_cap, (int64_t)num_pairs);
    {
        int warps_per_block = 8;
        int block = warps_per_block * 32;
        int grid = (num_pairs + warps_per_block - 1) / warps_per_block;
        compute_cosine_kernel<<<grid, block>>>(d_offsets, d_indices, d_weights,
            cache.pair_u, cache.pair_v, num_pairs, cache.scores_buf);
    }

    
    bool use_topk = topk.has_value() && topk.value() < (std::size_t)num_pairs;
    int32_t output_count = use_topk ? (int32_t)topk.value() : num_pairs;

    
    int32_t* out_first;
    int32_t* out_second;
    float* out_scores;
    cudaMalloc(&out_first, output_count * sizeof(int32_t));
    cudaMalloc(&out_second, output_count * sizeof(int32_t));
    cudaMalloc(&out_scores, output_count * sizeof(float));

    if (use_topk) {
        ensure_buf(cache.idx_buf, cache.idx_cap, (int64_t)num_pairs);
        gen_seq_kernel<<<(num_pairs + 255) / 256, 256>>>(cache.idx_buf, num_pairs);

        ensure_buf(cache.sorted_scores_buf, cache.sorted_scores_cap, (int64_t)num_pairs);
        ensure_buf(cache.sorted_idx, cache.sorted_idx_cap, (int64_t)num_pairs);

        size_t tk_sz = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, tk_sz, (float*)nullptr, (float*)nullptr,
            (int32_t*)nullptr, (int32_t*)nullptr, (int)num_pairs);
        ensure_buf_bytes(cache.cub_temp, cache.cub_temp_cap, tk_sz);
        cub::DeviceRadixSort::SortPairsDescending(
            cache.cub_temp, tk_sz, cache.scores_buf, cache.sorted_scores_buf,
            cache.idx_buf, cache.sorted_idx, (int)num_pairs);

        gather_topk_kernel<<<(output_count + 255) / 256, 256>>>(
            cache.pair_u, cache.pair_v, cache.sorted_idx, output_count,
            out_first, out_second);
        cudaMemcpyAsync(out_scores, cache.sorted_scores_buf,
            output_count * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        cudaMemcpyAsync(out_first, cache.pair_u, num_pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(out_second, cache.pair_v, num_pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(out_scores, cache.scores_buf, num_pairs * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return {out_first, out_second, out_scores, (std::size_t)output_count};
}

}  
