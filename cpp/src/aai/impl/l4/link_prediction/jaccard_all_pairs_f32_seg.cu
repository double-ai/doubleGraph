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

namespace aai {

namespace {

struct Cache : Cacheable {};

static inline int grid_sz(int64_t n, int block = 256) {
    int g = (int)((n + block - 1) / block);
    return (g > 65535) ? 65535 : ((g < 1) ? 1 : g);
}



__global__ void count_2hop_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds, int32_t num_seeds, int64_t* __restrict__ counts)
{
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int32_t u = seeds[sid];
    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int64_t c = 0;
    for (int i = u_start + threadIdx.x; i < u_end; i += blockDim.x) {
        int32_t w = indices[i];
        c += (int64_t)(offsets[w + 1] - offsets[w]);
    }
    for (int off = 16; off > 0; off >>= 1)
        c += __shfl_down_sync(0xffffffff, c, off);
    __shared__ int64_t ws[8];
    int wid = threadIdx.x / 32, lane = threadIdx.x % 32;
    if (lane == 0) ws[wid] = c;
    __syncthreads();
    if (threadIdx.x == 0) {
        int nw = (blockDim.x + 31) / 32;
        int64_t total = 0;
        for (int w = 0; w < nw; w++) total += ws[w];
        counts[sid] = total;
    }
}

__global__ void prefix_sum_small_kernel(
    const int64_t* __restrict__ counts, int64_t* __restrict__ prefix,
    int64_t* __restrict__ total_out, int32_t n)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int64_t sum = 0;
        for (int i = 0; i < n; i++) { prefix[i] = sum; sum += counts[i]; }
        *total_out = sum;
    }
}

__global__ void write_2hop_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds, int32_t num_seeds,
    const int64_t* __restrict__ seed_offsets, uint64_t* __restrict__ out)
{
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int32_t u = seeds[sid];
    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int64_t base = seed_offsets[sid];
    __shared__ unsigned long long shared_pos;
    if (threadIdx.x == 0) shared_pos = 0;
    __syncthreads();
    for (int i = u_start + threadIdx.x; i < u_end; i += blockDim.x) {
        int32_t w = indices[i];
        int32_t w_start = offsets[w], w_end = offsets[w + 1], w_deg = w_end - w_start;
        unsigned long long pos = atomicAdd(&shared_pos, (unsigned long long)w_deg);
        for (int j = 0; j < w_deg; j++) {
            int32_t v = indices[w_start + j];
            out[base + pos + j] = ((uint64_t)(uint32_t)u << 32) | (uint64_t)(uint32_t)v;
        }
    }
}

__global__ void mark_valid_kernel(
    const uint64_t* __restrict__ sorted_pairs, uint8_t* __restrict__ flags, int64_t n)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)gridDim.x * blockDim.x) {
        uint64_t p = sorted_pairs[i];
        uint32_t u = (uint32_t)(p >> 32), v = (uint32_t)(p & 0xFFFFFFFF);
        flags[i] = ((u != v) && (i == 0 || p != sorted_pairs[i - 1])) ? 1 : 0;
    }
}

__global__ void compute_jaccard_warp_kernel(
    const uint64_t* __restrict__ pairs,
    const int32_t* __restrict__ graph_offsets,
    const int32_t* __restrict__ graph_indices,
    const float* __restrict__ edge_weights,
    int32_t* __restrict__ first_out,
    int32_t* __restrict__ second_out,
    float* __restrict__ scores_out,
    int64_t n)
{
    const int lane = threadIdx.x & 31;
    const int64_t warps_per_grid = ((int64_t)gridDim.x * blockDim.x) / 32;
    const int64_t warp_id_base = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;

    for (int64_t warp_id = warp_id_base; warp_id < n; warp_id += warps_per_grid) {
        uint64_t pair = pairs[warp_id];
        int32_t u = (int32_t)(pair >> 32);
        int32_t v = (int32_t)(pair & 0xFFFFFFFF);

        if (lane == 0) { first_out[warp_id] = u; second_out[warp_id] = v; }

        int32_t u_start = graph_offsets[u], u_end = graph_offsets[u + 1];
        int32_t v_start = graph_offsets[v], v_end = graph_offsets[v + 1];
        int32_t u_deg = u_end - u_start, v_deg = v_end - v_start;

        float pw = 0.0f;
        for (int i = u_start + lane; i < u_end; i += 32) pw += edge_weights[i];
        for (int off = 16; off > 0; off >>= 1) pw += __shfl_down_sync(0xffffffff, pw, off);
        float sum_u = __shfl_sync(0xffffffff, pw, 0);

        pw = 0.0f;
        for (int i = v_start + lane; i < v_end; i += 32) pw += edge_weights[i];
        for (int off = 16; off > 0; off >>= 1) pw += __shfl_down_sync(0xffffffff, pw, off);
        float sum_v = __shfl_sync(0xffffffff, pw, 0);

        const int32_t* a_idx, *b_idx;
        const float* a_wt, *b_wt;
        int32_t a_size, b_size;
        if (u_deg <= v_deg) {
            a_idx = graph_indices + u_start; a_wt = edge_weights + u_start; a_size = u_deg;
            b_idx = graph_indices + v_start; b_wt = edge_weights + v_start; b_size = v_deg;
        } else {
            a_idx = graph_indices + v_start; a_wt = edge_weights + v_start; a_size = v_deg;
            b_idx = graph_indices + u_start; b_wt = edge_weights + u_start; b_size = u_deg;
        }

        float my_isect = 0.0f;
        for (int i = lane; i < a_size; i += 32) {
            int32_t target = a_idx[i];
            float aw = a_wt[i];
            int32_t lo = 0, hi = b_size;
            while (lo < hi) {
                int32_t mid = (lo + hi) >> 1;
                if (b_idx[mid] < target) lo = mid + 1; else hi = mid;
            }
            if (lo < b_size && b_idx[lo] == target)
                my_isect += fminf(aw, b_wt[lo]);
        }

        for (int off = 16; off > 0; off >>= 1)
            my_isect += __shfl_down_sync(0xffffffff, my_isect, off);

        if (lane == 0) {
            float denom = sum_u + sum_v - my_isect;
            scores_out[warp_id] = (denom > 0.0f) ? (my_isect / denom) : 0.0f;
        }
    }
}

__global__ void iota_kernel(int32_t* out, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
        out[i] = (int32_t)i;
}

__global__ void negate_scores_kernel(const float* __restrict__ in, float* __restrict__ out, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)gridDim.x * blockDim.x)
        out[i] = -in[i];
}

__global__ void gather_int32_kernel(const int32_t* __restrict__ idx,
    const int32_t* __restrict__ src, int32_t* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)gridDim.x * blockDim.x)
        dst[i] = src[idx[i]];
}

__global__ void gather_float_kernel(const int32_t* __restrict__ idx,
    const float* __restrict__ src, float* __restrict__ dst, int64_t n) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)gridDim.x * blockDim.x)
        dst[i] = src[idx[i]];
}

}  



similarity_result_float_t jaccard_all_pairs_similarity_seg(
    const graph32_t& graph,
    const float* edge_weights,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    similarity_result_float_t result = {nullptr, nullptr, nullptr, 0};

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const float* d_weights = edge_weights;

    
    const int32_t* d_seeds = nullptr;
    int32_t num_seeds = 0;
    int32_t* d_all_seeds = nullptr;

    if (vertices != nullptr) {
        d_seeds = vertices;
        num_seeds = (int32_t)num_vertices;
    } else {
        num_seeds = graph.number_of_vertices;
        cudaMalloc(&d_all_seeds, (size_t)num_seeds * sizeof(int32_t));
        iota_kernel<<<grid_sz(num_seeds), 256>>>(d_all_seeds, (int64_t)num_seeds);
        d_seeds = d_all_seeds;
    }

    if (num_seeds == 0) {
        if (d_all_seeds) cudaFree(d_all_seeds);
        return result;
    }

    
    int bits_per_vertex = 1;
    unsigned int nv = (unsigned int)graph.number_of_vertices;
    while (bits_per_vertex < 32 && (1u << bits_per_vertex) < nv) bits_per_vertex++;
    int sort_bits = 32 + bits_per_vertex;
    if (sort_bits > 64) sort_bits = 64;

    
    int64_t* d_counts = nullptr;
    cudaMalloc(&d_counts, num_seeds * sizeof(int64_t));
    count_2hop_kernel<<<num_seeds, 256>>>(
        d_offsets, d_indices, d_seeds, num_seeds, d_counts);

    int64_t* d_seed_offs = nullptr;
    cudaMalloc(&d_seed_offs, num_seeds * sizeof(int64_t));
    int64_t* d_total = nullptr;
    cudaMalloc(&d_total, sizeof(int64_t));
    prefix_sum_small_kernel<<<1, 1>>>(d_counts, d_seed_offs, d_total, num_seeds);
    cudaFree(d_counts);

    int64_t total_count;
    cudaMemcpy(&total_count, d_total, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_total);

    if (total_count == 0) {
        cudaFree(d_seed_offs);
        if (d_all_seeds) cudaFree(d_all_seeds);
        return result;
    }

    
    uint64_t* d_pairs = nullptr;
    cudaMalloc(&d_pairs, total_count * sizeof(uint64_t));
    write_2hop_kernel<<<num_seeds, 256>>>(
        d_offsets, d_indices, d_seeds, num_seeds, d_seed_offs, d_pairs);
    cudaFree(d_seed_offs);

    
    if (d_all_seeds) { cudaFree(d_all_seeds); d_all_seeds = nullptr; }

    
    uint64_t* d_pairs_sorted = nullptr;
    cudaMalloc(&d_pairs_sorted, total_count * sizeof(uint64_t));
    {
        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, d_pairs, d_pairs_sorted,
            total_count, 0, sort_bits);
        void* d_temp = nullptr;
        cudaMalloc(&d_temp, temp_bytes);
        cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_pairs, d_pairs_sorted,
            total_count, 0, sort_bits);
        cudaFree(d_temp);
    }
    cudaFree(d_pairs);

    
    uint8_t* d_flags = nullptr;
    cudaMalloc(&d_flags, total_count * sizeof(uint8_t));
    mark_valid_kernel<<<grid_sz(total_count), 256>>>(
        d_pairs_sorted, d_flags, total_count);

    uint64_t* d_valid = nullptr;
    cudaMalloc(&d_valid, total_count * sizeof(uint64_t));
    int64_t* d_num_selected = nullptr;
    cudaMalloc(&d_num_selected, sizeof(int64_t));
    {
        size_t temp_bytes = 0;
        cub::DeviceSelect::Flagged(nullptr, temp_bytes, d_pairs_sorted, d_flags,
            d_valid, d_num_selected, total_count);
        void* d_temp = nullptr;
        cudaMalloc(&d_temp, temp_bytes);
        cub::DeviceSelect::Flagged(d_temp, temp_bytes, d_pairs_sorted, d_flags,
            d_valid, d_num_selected, total_count);
        cudaFree(d_temp);
    }
    cudaFree(d_flags);
    cudaFree(d_pairs_sorted);

    int64_t final_count;
    cudaMemcpy(&final_count, d_num_selected, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_num_selected);

    if (final_count == 0) {
        cudaFree(d_valid);
        return result;
    }

    
    int32_t* d_first = nullptr;
    cudaMalloc(&d_first, final_count * sizeof(int32_t));
    int32_t* d_second = nullptr;
    cudaMalloc(&d_second, final_count * sizeof(int32_t));
    float* d_scores = nullptr;
    cudaMalloc(&d_scores, final_count * sizeof(float));

    {
        int threads = 256;
        int warps_per_block = threads / 32;
        int64_t blocks64 = (final_count + warps_per_block - 1) / warps_per_block;
        int blocks = (blocks64 > 65535) ? 65535 : ((blocks64 < 1) ? 1 : (int)blocks64);
        compute_jaccard_warp_kernel<<<blocks, threads>>>(
            d_valid, d_offsets, d_indices, d_weights,
            d_first, d_second, d_scores, final_count);
    }
    cudaFree(d_valid);

    
    if (topk.has_value() && topk.value() < (std::size_t)final_count) {
        int64_t topk_val = (int64_t)topk.value();

        float* d_neg = nullptr;
        cudaMalloc(&d_neg, final_count * sizeof(float));
        negate_scores_kernel<<<grid_sz(final_count), 256>>>(d_scores, d_neg, final_count);

        int32_t* d_idx = nullptr;
        cudaMalloc(&d_idx, final_count * sizeof(int32_t));
        iota_kernel<<<grid_sz(final_count), 256>>>(d_idx, final_count);

        float* d_neg_out = nullptr;
        cudaMalloc(&d_neg_out, final_count * sizeof(float));
        int32_t* d_idx_out = nullptr;
        cudaMalloc(&d_idx_out, final_count * sizeof(int32_t));
        {
            size_t temp_bytes = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
                d_neg, d_neg_out, d_idx, d_idx_out,
                final_count, 0, 32);
            void* d_temp = nullptr;
            cudaMalloc(&d_temp, temp_bytes);
            cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
                d_neg, d_neg_out, d_idx, d_idx_out,
                final_count, 0, 32);
            cudaFree(d_temp);
        }
        cudaFree(d_neg);
        cudaFree(d_neg_out);
        cudaFree(d_idx);

        int32_t* d_first_tk = nullptr;
        cudaMalloc(&d_first_tk, topk_val * sizeof(int32_t));
        int32_t* d_second_tk = nullptr;
        cudaMalloc(&d_second_tk, topk_val * sizeof(int32_t));
        float* d_scores_tk = nullptr;
        cudaMalloc(&d_scores_tk, topk_val * sizeof(float));

        int g = grid_sz(topk_val);
        gather_int32_kernel<<<g, 256>>>(d_idx_out, d_first, d_first_tk, topk_val);
        gather_int32_kernel<<<g, 256>>>(d_idx_out, d_second, d_second_tk, topk_val);
        gather_float_kernel<<<g, 256>>>(d_idx_out, d_scores, d_scores_tk, topk_val);

        cudaFree(d_idx_out);
        cudaFree(d_first);
        cudaFree(d_second);
        cudaFree(d_scores);

        result.first = d_first_tk;
        result.second = d_second_tk;
        result.scores = d_scores_tk;
        result.count = topk.value();
    } else {
        result.first = d_first;
        result.second = d_second;
        result.scores = d_scores;
        result.count = (std::size_t)final_count;
    }

    return result;
}

}  
