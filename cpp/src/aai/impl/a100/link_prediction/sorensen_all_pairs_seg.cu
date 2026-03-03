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
#include <algorithm>

namespace aai {

namespace {

template <typename T>
void ensure(T*& ptr, int64_t& cap, int64_t n) {
    if (cap < n) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, n * sizeof(T));
        cap = n;
    }
}

void ensure_bytes(uint8_t*& ptr, size_t& cap, size_t n) {
    if (cap < n) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, n);
        cap = n;
    }
}

struct Cache : Cacheable {
    int32_t* seeds = nullptr;
    int64_t seeds_cap = 0;

    int64_t* exp_sizes = nullptr;
    int64_t exp_sizes_cap = 0;

    int64_t* exp_offsets = nullptr;
    int64_t exp_offsets_cap = 0;

    uint8_t* scan_temp = nullptr;
    size_t scan_temp_cap = 0;

    int64_t* keys_buf = nullptr;
    int64_t keys_buf_cap = 0;

    uint8_t* sort_temp = nullptr;
    size_t sort_temp_cap = 0;

    int64_t* keys_sorted = nullptr;
    int64_t keys_sorted_cap = 0;

    int64_t* unique_keys = nullptr;
    int64_t unique_keys_cap = 0;

    int32_t* counts = nullptr;
    int64_t counts_cap = 0;

    int64_t* num_runs = nullptr;

    uint8_t* rle_temp = nullptr;
    size_t rle_temp_cap = 0;

    int32_t* out_count = nullptr;

    int32_t* sort_idx = nullptr;
    int64_t sort_idx_cap = 0;

    float* scores_sorted = nullptr;
    int64_t scores_sorted_cap = 0;

    int32_t* idx_sorted = nullptr;
    int64_t idx_sorted_cap = 0;

    uint8_t* topk_temp = nullptr;
    size_t topk_temp_cap = 0;

    Cache() {
        cudaMalloc(&num_runs, sizeof(int64_t));
        cudaMalloc(&out_count, sizeof(int32_t));
    }

    ~Cache() override {
        if (seeds) cudaFree(seeds);
        if (exp_sizes) cudaFree(exp_sizes);
        if (exp_offsets) cudaFree(exp_offsets);
        if (scan_temp) cudaFree(scan_temp);
        if (keys_buf) cudaFree(keys_buf);
        if (sort_temp) cudaFree(sort_temp);
        if (keys_sorted) cudaFree(keys_sorted);
        if (unique_keys) cudaFree(unique_keys);
        if (counts) cudaFree(counts);
        if (num_runs) cudaFree(num_runs);
        if (rle_temp) cudaFree(rle_temp);
        if (out_count) cudaFree(out_count);
        if (sort_idx) cudaFree(sort_idx);
        if (scores_sorted) cudaFree(scores_sorted);
        if (idx_sorted) cudaFree(idx_sorted);
        if (topk_temp) cudaFree(topk_temp);
    }
};



__global__ void iota_kernel(int32_t* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}

__global__ void compute_expansion_sizes_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    int64_t* __restrict__ sizes
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;
    if (warp_id >= num_seeds) return;

    int u = seeds[warp_id];
    int s = offsets[u], e = offsets[u + 1];
    int deg = e - s;

    int64_t total = 0;
    for (int i = lane_id; i < deg; i += 32) {
        int w = indices[s + i];
        total += (int64_t)(offsets[w + 1] - offsets[w]);
    }
    for (int off = 16; off > 0; off /= 2)
        total += __shfl_down_sync(0xffffffff, total, off);

    if (lane_id == 0) sizes[warp_id] = total;
}

__global__ void fill_expansion_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    const int64_t* __restrict__ write_offsets,
    int64_t* __restrict__ keys,
    int64_t stride
) {
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int u = seeds[sid];
    int u_s = offsets[u], u_e = offsets[u + 1];
    int64_t base = write_offsets[sid];
    int64_t pos = 0;
    int64_t sid_stride = (int64_t)sid * stride;

    for (int ni = u_s; ni < u_e; ni++) {
        int w = indices[ni];
        int w_s = offsets[w], w_e = offsets[w + 1];
        int deg_w = w_e - w_s;
        for (int j = threadIdx.x; j < deg_w; j += blockDim.x) {
            keys[base + pos + j] = sid_stride + (int64_t)indices[w_s + j];
        }
        pos += deg_w;
    }
}

__global__ void compute_scores_simple_kernel(
    const int64_t* __restrict__ unique_keys,
    const int32_t* __restrict__ counts,
    int64_t num_unique,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ seeds,
    int64_t stride,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int32_t* __restrict__ out_count
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_unique) return;
    int64_t key = unique_keys[idx];
    int sid = (int)(key / stride);
    int v = (int)(key % stride);
    int u = seeds[sid];
    if (v == u) return;
    int deg_u = offsets[u + 1] - offsets[u];
    int deg_v = offsets[v + 1] - offsets[v];
    if (deg_u + deg_v == 0) return;
    float score = 2.0f * (float)counts[idx] / (float)(deg_u + deg_v);
    int pos = atomicAdd(out_count, 1);
    out_first[pos] = u;
    out_second[pos] = v;
    out_scores[pos] = score;
}

__global__ void compute_scores_intersection_kernel(
    const int64_t* __restrict__ unique_keys,
    int64_t num_unique,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int64_t stride,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int32_t* __restrict__ out_count
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_unique) return;
    int64_t key = unique_keys[idx];
    int sid = (int)(key / stride);
    int v = (int)(key % stride);
    int u = seeds[sid];
    if (v == u) return;
    int u_s = offsets[u], u_e = offsets[u + 1];
    int v_s = offsets[v], v_e = offsets[v + 1];
    int deg_u = u_e - u_s;
    int deg_v = v_e - v_s;
    if (deg_u + deg_v == 0) return;

    int i = u_s, j = v_s;
    if (i < u_e && j < v_e) {
        int a0 = indices[i], b0 = indices[j];
        if (a0 < b0) {
            int lo = i, hi = u_e;
            while (lo < hi) { int m = lo + (hi-lo)/2; if (indices[m] < b0) lo = m+1; else hi = m; }
            i = lo;
        } else if (b0 < a0) {
            int lo = j, hi = v_e;
            while (lo < hi) { int m = lo + (hi-lo)/2; if (indices[m] < a0) lo = m+1; else hi = m; }
            j = lo;
        }
    }
    int count = 0;
    while (i < u_e && j < v_e) {
        int a = indices[i], b = indices[j];
        if (a == b) { count++; i++; j++; }
        else if (a < b) i++;
        else j++;
    }

    if (count > 0) {
        float score = 2.0f * (float)count / (float)(deg_u + deg_v);
        int pos = atomicAdd(out_count, 1);
        out_first[pos] = u;
        out_second[pos] = v;
        out_scores[pos] = score;
    }
}

__global__ void gather_int32_kernel(const int32_t* src, const int32_t* idx, int32_t* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
}

}  

similarity_result_float_t sorensen_all_pairs_similarity_seg(
    const graph32_t& graph,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n = graph.number_of_vertices;
    bool is_multigraph = graph.is_multigraph;
    cudaStream_t stream = 0;

    int num_seeds;
    const int32_t* d_seeds;
    if (vertices != nullptr) {
        num_seeds = static_cast<int>(num_vertices);
        d_seeds = vertices;
    } else {
        num_seeds = n;
        ensure(cache.seeds, cache.seeds_cap, (int64_t)n);
        iota_kernel<<<(n + 255) / 256, 256, 0, stream>>>(cache.seeds, n);
        d_seeds = cache.seeds;
    }

    if (num_seeds == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    ensure(cache.exp_sizes, cache.exp_sizes_cap, (int64_t)num_seeds);
    {
        int threads = 256;
        int warps_per_block = threads / 32;
        int blocks = (num_seeds + warps_per_block - 1) / warps_per_block;
        compute_expansion_sizes_kernel<<<blocks, threads, 0, stream>>>(
            d_offsets, d_indices, d_seeds, num_seeds, cache.exp_sizes);
    }

    
    ensure(cache.exp_offsets, cache.exp_offsets_cap, (int64_t)(num_seeds + 1));
    cudaMemsetAsync(cache.exp_offsets, 0, sizeof(int64_t), stream);
    {
        size_t scan_tb = 0;
        cub::DeviceScan::InclusiveSum(nullptr, scan_tb,
            (int64_t*)nullptr, (int64_t*)nullptr, num_seeds);
        ensure_bytes(cache.scan_temp, cache.scan_temp_cap, std::max(scan_tb, (size_t)1));
        cub::DeviceScan::InclusiveSum(cache.scan_temp, scan_tb,
            cache.exp_sizes, cache.exp_offsets + 1, num_seeds, stream);
    }

    
    int64_t total_expansion = 0;
    cudaMemcpyAsync(&total_expansion, cache.exp_offsets + num_seeds,
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (total_expansion == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t stride = (int64_t)n;
    ensure(cache.keys_buf, cache.keys_buf_cap, total_expansion);
    fill_expansion_kernel<<<num_seeds, 512, 0, stream>>>(
        d_offsets, d_indices, d_seeds, num_seeds,
        cache.exp_offsets, cache.keys_buf, stride);

    
    int end_bit = 1;
    {
        int64_t mk = (int64_t)(num_seeds - 1) * stride + (stride - 1);
        while ((1LL << end_bit) <= mk && end_bit < 64) end_bit++;
    }
    {
        size_t sort_tb = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, sort_tb,
            (int64_t*)nullptr, (int64_t*)nullptr, total_expansion, 0, end_bit);
        ensure_bytes(cache.sort_temp, cache.sort_temp_cap, std::max(sort_tb, (size_t)1));
        ensure(cache.keys_sorted, cache.keys_sorted_cap, total_expansion);
        cub::DeviceRadixSort::SortKeys(cache.sort_temp, sort_tb,
            cache.keys_buf, cache.keys_sorted, total_expansion, 0, end_bit, stream);
    }

    
    ensure(cache.unique_keys, cache.unique_keys_cap, total_expansion);
    ensure(cache.counts, cache.counts_cap, total_expansion);
    {
        size_t rle_tb = 0;
        cub::DeviceRunLengthEncode::Encode(nullptr, rle_tb,
            (int64_t*)nullptr, (int64_t*)nullptr, (int32_t*)nullptr, (int64_t*)nullptr,
            total_expansion);
        ensure_bytes(cache.rle_temp, cache.rle_temp_cap, std::max(rle_tb, (size_t)1));
        cub::DeviceRunLengthEncode::Encode(cache.rle_temp, rle_tb,
            cache.keys_sorted, cache.unique_keys, cache.counts, cache.num_runs,
            total_expansion, stream);
    }

    
    int64_t num_unique = 0;
    cudaMemcpyAsync(&num_unique, cache.num_runs,
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (num_unique == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    
    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;
    cudaMalloc(&out_first, num_unique * sizeof(int32_t));
    cudaMalloc(&out_second, num_unique * sizeof(int32_t));
    cudaMalloc(&out_scores, num_unique * sizeof(float));
    cudaMemsetAsync(cache.out_count, 0, sizeof(int32_t), stream);

    if (is_multigraph) {
        int64_t g = (num_unique + 255) / 256;
        int grid = (int)std::min(g, (int64_t)INT_MAX);
        compute_scores_intersection_kernel<<<grid, 256, 0, stream>>>(
            cache.unique_keys, num_unique,
            d_offsets, d_indices, d_seeds, stride,
            out_first, out_second, out_scores, cache.out_count);
    } else {
        int64_t g = (num_unique + 255) / 256;
        int grid = (int)std::min(g, (int64_t)INT_MAX);
        compute_scores_simple_kernel<<<grid, 256, 0, stream>>>(
            cache.unique_keys, cache.counts, num_unique,
            d_offsets, d_seeds, stride,
            out_first, out_second, out_scores, cache.out_count);
    }

    
    int32_t total_pairs = 0;
    cudaMemcpyAsync(&total_pairs, cache.out_count,
                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (total_pairs == 0) {
        cudaFree(out_first);
        cudaFree(out_second);
        cudaFree(out_scores);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    bool need_topk = topk.has_value() && ((std::size_t)total_pairs > topk.value());
    if (need_topk) {
        int topk_val = (int)topk.value();
        ensure(cache.sort_idx, cache.sort_idx_cap, (int64_t)total_pairs);
        iota_kernel<<<(total_pairs + 255) / 256, 256, 0, stream>>>(cache.sort_idx, total_pairs);
        ensure(cache.scores_sorted, cache.scores_sorted_cap, (int64_t)total_pairs);
        ensure(cache.idx_sorted, cache.idx_sorted_cap, (int64_t)total_pairs);
        {
            size_t topk_tb = 0;
            cub::DeviceRadixSort::SortPairsDescending(nullptr, topk_tb,
                (float*)nullptr, (float*)nullptr, (int32_t*)nullptr, (int32_t*)nullptr,
                total_pairs, 0, 32);
            ensure_bytes(cache.topk_temp, cache.topk_temp_cap, std::max(topk_tb, (size_t)1));
            cub::DeviceRadixSort::SortPairsDescending(cache.topk_temp, topk_tb,
                out_scores, cache.scores_sorted,
                cache.sort_idx, cache.idx_sorted,
                total_pairs, 0, 32, stream);
        }
        int32_t* f1 = nullptr;
        int32_t* f2 = nullptr;
        float* fs = nullptr;
        cudaMalloc(&f1, topk_val * sizeof(int32_t));
        cudaMalloc(&f2, topk_val * sizeof(int32_t));
        cudaMalloc(&fs, topk_val * sizeof(float));
        gather_int32_kernel<<<(topk_val + 255) / 256, 256, 0, stream>>>(
            out_first, cache.idx_sorted, f1, topk_val);
        gather_int32_kernel<<<(topk_val + 255) / 256, 256, 0, stream>>>(
            out_second, cache.idx_sorted, f2, topk_val);
        cudaMemcpyAsync(fs, cache.scores_sorted, topk_val * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaFree(out_first);
        cudaFree(out_second);
        cudaFree(out_scores);
        return {f1, f2, fs, (std::size_t)topk_val};
    }

    
    if ((int64_t)total_pairs == num_unique) {
        return {out_first, out_second, out_scores, (std::size_t)total_pairs};
    }
    int32_t* f1 = nullptr;
    int32_t* f2 = nullptr;
    float* fs = nullptr;
    cudaMalloc(&f1, total_pairs * sizeof(int32_t));
    cudaMalloc(&f2, total_pairs * sizeof(int32_t));
    cudaMalloc(&fs, total_pairs * sizeof(float));
    cudaMemcpyAsync(f1, out_first, total_pairs * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(f2, out_second, total_pairs * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(fs, out_scores, total_pairs * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaFree(out_first);
    cudaFree(out_second);
    cudaFree(out_scores);
    return {f1, f2, fs, (std::size_t)total_pairs};
}

}  
