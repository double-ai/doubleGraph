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



__global__ void compute_wd_kernel(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ weights,
    float* __restrict__ wd,
    int32_t n)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    float s = 0.0f;
    int start = offsets[v], end = offsets[v + 1];
    for (int i = start; i < end; i++) s += weights[i];
    wd[v] = s;
}

__global__ void count_triples_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ counts)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_seeds) return;
    int u = seeds[sid];
    int64_t cnt = 0;
    for (int ci = offsets[u]; ci < offsets[u + 1]; ci++) {
        int c = indices[ci];
        cnt += (int64_t)(offsets[c + 1] - offsets[c]);
    }
    counts[sid] = cnt;
}


__global__ void generate_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ pair_offsets,
    int64_t* __restrict__ pair_keys)
{
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int u = seeds[sid];
    int64_t base = pair_offsets[sid];
    int64_t pos = 0;

    for (int ci = offsets[u]; ci < offsets[u + 1]; ci++) {
        int c = indices[ci];
        int c_start = offsets[c];
        int c_end = offsets[c + 1];
        int c_deg = c_end - c_start;

        for (int vi = threadIdx.x; vi < c_deg; vi += blockDim.x) {
            int v = indices[c_start + vi];
            int64_t key;
            if (v == u) {
                key = INT64_MAX; 
            } else {
                key = ((int64_t)(uint32_t)u << 32) | (int64_t)(uint32_t)v;
            }
            pair_keys[base + pos + vi] = key;
        }
        pos += c_deg;
    }
}


__global__ void compute_scores_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ wd,
    const int64_t* __restrict__ pair_keys,
    int32_t num_pairs,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= num_pairs) return;

    int64_t key = pair_keys[warp_id];
    int32_t u = (int32_t)((uint64_t)key >> 32);
    int32_t v = (int32_t)(key & 0xFFFFFFFFU);

    int u_start = offsets[u], u_end = offsets[u + 1];
    int v_start = offsets[v], v_end = offsets[v + 1];
    int u_deg = u_end - u_start;
    int v_deg = v_end - v_start;

    
    const int32_t* short_idx;
    const float* short_w;
    int short_start, short_deg;
    const int32_t* long_idx;
    const float* long_w;
    int long_start, long_end;

    if (u_deg <= v_deg) {
        short_idx = indices; short_w = weights;
        short_start = u_start; short_deg = u_deg;
        long_idx = indices; long_w = weights;
        long_start = v_start; long_end = v_end;
    } else {
        short_idx = indices; short_w = weights;
        short_start = v_start; short_deg = v_deg;
        long_idx = indices; long_w = weights;
        long_start = u_start; long_end = u_end;
    }

    float local_sum = 0.0f;
    for (int i = lane; i < short_deg; i += 32) {
        int c = short_idx[short_start + i];
        float w_short = short_w[short_start + i];

        
        int lo = long_start, hi = long_end;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (long_idx[mid] < c) lo = mid + 1;
            else hi = mid;
        }
        if (lo < long_end && long_idx[lo] == c) {
            local_sum += fminf(w_short, long_w[lo]);
        }
    }

    
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) {
        float du = wd[u], dv = wd[v];
        float denom = du + dv;
        
        float score = (denom <= 1.175494351e-38f) ? 0.0f : (2.0f * local_sum / denom);
        out_first[warp_id] = u;
        out_second[warp_id] = v;
        out_scores[warp_id] = score;
    }
}


__global__ void iota_kernel(int32_t* data, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = idx;
}


__global__ void gather_results_kernel(
    const int32_t* __restrict__ src_first,
    const int32_t* __restrict__ src_second,
    const float* __restrict__ src_scores,
    const int32_t* __restrict__ sorted_indices,
    int32_t n,
    int32_t* __restrict__ dst_first,
    int32_t* __restrict__ dst_second,
    float* __restrict__ dst_scores)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int src_idx = sorted_indices[idx];
    dst_first[idx] = src_first[src_idx];
    dst_second[idx] = src_second[src_idx];
    dst_scores[idx] = src_scores[src_idx];
}


__global__ void find_valid_count_kernel(const int64_t* sorted_keys, int64_t n, int64_t* out_count) {
    
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        int64_t mid = (lo + hi) >> 1;
        if (sorted_keys[mid] < INT64_MAX) lo = mid + 1;
        else hi = mid;
    }
    *out_count = lo;
}


__global__ void iota_seeds_kernel(int32_t* seeds, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) seeds[idx] = idx;
}



void launch_compute_wd(const int32_t* offsets, const float* weights, float* wd, int32_t n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    compute_wd_kernel<<<grid, block, 0, stream>>>(offsets, weights, wd, n);
}

void launch_count_triples(const int32_t* offsets, const int32_t* indices, const int32_t* seeds,
                           int32_t num_seeds, int64_t* counts, cudaStream_t stream) {
    if (num_seeds == 0) return;
    int block = 256;
    int grid = (num_seeds + block - 1) / block;
    count_triples_kernel<<<grid, block, 0, stream>>>(offsets, indices, seeds, num_seeds, counts);
}

void launch_generate_pairs(const int32_t* offsets, const int32_t* indices, const int32_t* seeds,
                             int32_t num_seeds, const int64_t* pair_offsets, int64_t* pair_keys,
                             cudaStream_t stream) {
    if (num_seeds == 0) return;
    int block = 256;
    generate_pairs_kernel<<<num_seeds, block, 0, stream>>>(offsets, indices, seeds, num_seeds, pair_offsets, pair_keys);
}

void launch_compute_scores(const int32_t* offsets, const int32_t* indices, const float* weights,
                             const float* wd, const int64_t* pair_keys, int32_t num_pairs,
                             int32_t* out_first, int32_t* out_second, float* out_scores,
                             cudaStream_t stream) {
    if (num_pairs == 0) return;
    
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int grid = (num_pairs + warps_per_block - 1) / warps_per_block;
    compute_scores_kernel<<<grid, threads_per_block, 0, stream>>>(
        offsets, indices, weights, wd, pair_keys, num_pairs, out_first, out_second, out_scores);
}

void launch_iota_seeds(int32_t* seeds, int32_t n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    iota_seeds_kernel<<<grid, block, 0, stream>>>(seeds, n);
}

void launch_iota(int32_t* data, int32_t n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    iota_kernel<<<grid, block, 0, stream>>>(data, n);
}

void launch_gather(const int32_t* src_first, const int32_t* src_second, const float* src_scores,
                    const int32_t* sorted_indices, int32_t n,
                    int32_t* dst_first, int32_t* dst_second, float* dst_scores, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    gather_results_kernel<<<grid, block, 0, stream>>>(src_first, src_second, src_scores,
                                                       sorted_indices, n, dst_first, dst_second, dst_scores);
}

void launch_find_valid_count(const int64_t* sorted_keys, int64_t n, int64_t* out_count, cudaStream_t stream) {
    find_valid_count_kernel<<<1, 1, 0, stream>>>(sorted_keys, n, out_count);
}



size_t cub_exclusive_sum_temp(int32_t num_items) {
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, (int64_t*)nullptr, (int64_t*)nullptr, num_items);
    return temp_bytes;
}

void cub_exclusive_sum(void* d_temp, size_t temp_bytes, const int64_t* d_in, int64_t* d_out,
                        int32_t num_items, cudaStream_t stream) {
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, num_items, stream);
}

size_t cub_sort_keys_temp(int64_t num_items) {
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, (int64_t*)nullptr, (int64_t*)nullptr,
                                    (int)num_items, 0, 64);
    return temp_bytes;
}

void cub_sort_keys(void* d_temp, size_t temp_bytes, const int64_t* d_in, int64_t* d_out,
                    int64_t num_items, cudaStream_t stream) {
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, (int)num_items, 0, 64, stream);
}

size_t cub_unique_temp(int64_t num_items) {
    size_t temp_bytes = 0;
    cub::DeviceSelect::Unique(nullptr, temp_bytes, (int64_t*)nullptr, (int64_t*)nullptr,
                               (int64_t*)nullptr, (int)num_items);
    return temp_bytes;
}

void cub_unique(void* d_temp, size_t temp_bytes, const int64_t* d_in, int64_t* d_out,
                 int64_t* d_num_out, int64_t num_items, cudaStream_t stream) {
    cub::DeviceSelect::Unique(d_temp, temp_bytes, d_in, d_out, d_num_out, (int)num_items, stream);
}

size_t cub_sort_pairs_desc_temp(int32_t num_items) {
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_bytes, (float*)nullptr, (float*)nullptr,
                                               (int32_t*)nullptr, (int32_t*)nullptr, num_items, 0, 32);
    return temp_bytes;
}

void cub_sort_pairs_desc(void* d_temp, size_t temp_bytes, const float* d_keys_in, float* d_keys_out,
                           const int32_t* d_vals_in, int32_t* d_vals_out, int32_t num_items,
                           cudaStream_t stream) {
    cub::DeviceRadixSort::SortPairsDescending(d_temp, temp_bytes, d_keys_in, d_keys_out,
                                               d_vals_in, d_vals_out, num_items, 0, 32, stream);
}



template <typename T>
void ensure(T*& ptr, int64_t& cap, int64_t needed) {
    if (cap < needed) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, needed * sizeof(T));
        cap = needed;
    }
}

void ensure_bytes(void*& ptr, size_t& cap, size_t needed) {
    if (cap < needed) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, needed);
        cap = needed;
    }
}



struct Cache : Cacheable {
    int32_t* seeds = nullptr;
    int64_t seeds_cap = 0;

    float* wd = nullptr;
    int64_t wd_cap = 0;

    int64_t* counts = nullptr;
    int64_t counts_cap = 0;

    int64_t* offsets_triple = nullptr;
    int64_t offsets_triple_cap = 0;

    int64_t* full_offsets = nullptr;
    int64_t full_offsets_cap = 0;

    int64_t* pair_keys = nullptr;
    int64_t pair_keys_cap = 0;

    int64_t* sorted_keys = nullptr;
    int64_t sorted_keys_cap = 0;

    int64_t* unique_keys = nullptr;
    int64_t unique_keys_cap = 0;

    int64_t* scalar_buf = nullptr;

    void* cub_temp = nullptr;
    size_t cub_temp_cap = 0;

    int32_t* topk_idx = nullptr;
    int64_t topk_idx_cap = 0;

    float* topk_sorted_scores = nullptr;
    int64_t topk_sorted_scores_cap = 0;

    int32_t* topk_sorted_idx = nullptr;
    int64_t topk_sorted_idx_cap = 0;

    Cache() {
        cudaMalloc(&scalar_buf, 2 * sizeof(int64_t));
    }

    ~Cache() override {
        if (seeds) cudaFree(seeds);
        if (wd) cudaFree(wd);
        if (counts) cudaFree(counts);
        if (offsets_triple) cudaFree(offsets_triple);
        if (full_offsets) cudaFree(full_offsets);
        if (pair_keys) cudaFree(pair_keys);
        if (sorted_keys) cudaFree(sorted_keys);
        if (unique_keys) cudaFree(unique_keys);
        if (scalar_buf) cudaFree(scalar_buf);
        if (cub_temp) cudaFree(cub_temp);
        if (topk_idx) cudaFree(topk_idx);
        if (topk_sorted_scores) cudaFree(topk_sorted_scores);
        if (topk_sorted_idx) cudaFree(topk_sorted_idx);
    }
};

}  

similarity_result_float_t sorensen_all_pairs_similarity_seg(
    const graph32_t& graph,
    const float* edge_weights,
    const int32_t* vertices,
    std::size_t num_vertices_param,
    std::optional<std::size_t> topk)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_verts = graph.number_of_vertices;
    const float* d_weights = edge_weights;

    cudaStream_t stream = 0;

    bool use_topk = topk.has_value();
    int64_t topk_val = use_topk ? (int64_t)topk.value() : 0;

    
    int32_t num_seeds;
    const int32_t* d_seeds;

    if (vertices != nullptr && num_vertices_param > 0) {
        num_seeds = (int32_t)num_vertices_param;
        d_seeds = vertices;
    } else {
        num_seeds = num_verts;
        ensure(cache.seeds, cache.seeds_cap, (int64_t)num_verts);
        launch_iota_seeds(cache.seeds, num_verts, stream);
        d_seeds = cache.seeds;
    }

    if (num_seeds == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    ensure(cache.wd, cache.wd_cap, (int64_t)num_verts);
    launch_compute_wd(d_offsets, d_weights, cache.wd, num_verts, stream);

    
    ensure(cache.counts, cache.counts_cap, (int64_t)num_seeds);
    launch_count_triples(d_offsets, d_indices, d_seeds, num_seeds, cache.counts, stream);

    
    ensure(cache.offsets_triple, cache.offsets_triple_cap, (int64_t)num_seeds);
    {
        size_t temp_sz = cub_exclusive_sum_temp(num_seeds);
        ensure_bytes(cache.cub_temp, cache.cub_temp_cap, temp_sz);
        cub_exclusive_sum(cache.cub_temp, temp_sz, cache.counts, cache.offsets_triple, num_seeds, stream);
    }

    
    int64_t h_last_offset, h_last_count;
    cudaMemcpyAsync(&h_last_offset, cache.offsets_triple + num_seeds - 1, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_last_count, cache.counts + num_seeds - 1, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int64_t total_triples = h_last_offset + h_last_count;

    if (total_triples == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    ensure(cache.full_offsets, cache.full_offsets_cap, (int64_t)(num_seeds + 1));
    cudaMemcpyAsync(cache.full_offsets, cache.offsets_triple,
                     num_seeds * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(cache.full_offsets + num_seeds, &total_triples,
                     sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    ensure(cache.pair_keys, cache.pair_keys_cap, total_triples);
    launch_generate_pairs(d_offsets, d_indices, d_seeds, num_seeds,
                           cache.full_offsets, cache.pair_keys, stream);

    
    ensure(cache.sorted_keys, cache.sorted_keys_cap, total_triples);
    {
        size_t temp_sz = cub_sort_keys_temp(total_triples);
        ensure_bytes(cache.cub_temp, cache.cub_temp_cap, temp_sz);
        cub_sort_keys(cache.cub_temp, temp_sz, cache.pair_keys, cache.sorted_keys, total_triples, stream);
    }

    
    launch_find_valid_count(cache.sorted_keys, total_triples, cache.scalar_buf, stream);
    int64_t h_valid_count;
    cudaMemcpyAsync(&h_valid_count, cache.scalar_buf, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (h_valid_count == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    ensure(cache.unique_keys, cache.unique_keys_cap, h_valid_count);
    {
        size_t temp_sz = cub_unique_temp(h_valid_count);
        ensure_bytes(cache.cub_temp, cache.cub_temp_cap, temp_sz);
        cub_unique(cache.cub_temp, temp_sz, cache.sorted_keys, cache.unique_keys,
                    cache.scalar_buf + 1, h_valid_count, stream);
    }

    int64_t h_num_unique;
    cudaMemcpyAsync(&h_num_unique, cache.scalar_buf + 1, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int32_t num_pairs = (int32_t)h_num_unique;
    if (num_pairs == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;
    cudaMalloc(&out_first, (size_t)num_pairs * sizeof(int32_t));
    cudaMalloc(&out_second, (size_t)num_pairs * sizeof(int32_t));
    cudaMalloc(&out_scores, (size_t)num_pairs * sizeof(float));

    launch_compute_scores(d_offsets, d_indices, d_weights, cache.wd,
                           cache.unique_keys, num_pairs,
                           out_first, out_second, out_scores, stream);

    
    if (use_topk && topk_val < (int64_t)num_pairs) {
        int32_t k = (int32_t)topk_val;

        ensure(cache.topk_idx, cache.topk_idx_cap, (int64_t)num_pairs);
        launch_iota(cache.topk_idx, num_pairs, stream);

        ensure(cache.topk_sorted_scores, cache.topk_sorted_scores_cap, (int64_t)num_pairs);
        ensure(cache.topk_sorted_idx, cache.topk_sorted_idx_cap, (int64_t)num_pairs);

        {
            size_t temp_sz = cub_sort_pairs_desc_temp(num_pairs);
            ensure_bytes(cache.cub_temp, cache.cub_temp_cap, temp_sz);
            cub_sort_pairs_desc(cache.cub_temp, temp_sz,
                                 out_scores, cache.topk_sorted_scores,
                                 cache.topk_idx, cache.topk_sorted_idx,
                                 num_pairs, stream);
        }

        int32_t* final_first = nullptr;
        int32_t* final_second = nullptr;
        float* final_scores = nullptr;
        cudaMalloc(&final_first, (size_t)k * sizeof(int32_t));
        cudaMalloc(&final_second, (size_t)k * sizeof(int32_t));
        cudaMalloc(&final_scores, (size_t)k * sizeof(float));

        launch_gather(out_first, out_second, out_scores,
                       cache.topk_sorted_idx, k,
                       final_first, final_second, final_scores, stream);

        cudaFree(out_first);
        cudaFree(out_second);
        cudaFree(out_scores);

        return {final_first, final_second, final_scores, (std::size_t)k};
    }

    return {out_first, out_second, out_scores, (std::size_t)num_pairs};
}

}  
