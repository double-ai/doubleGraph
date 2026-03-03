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
#include <algorithm>
#include <optional>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* seeds = nullptr;
    int64_t seeds_capacity = 0;

    double* wd = nullptr;
    int64_t wd_capacity = 0;

    uint32_t* bitmaps = nullptr;
    int64_t bitmaps_capacity = 0;

    int32_t* counts = nullptr;
    int64_t counts_capacity = 0;

    int32_t* pair_offsets = nullptr;
    int64_t pair_offsets_capacity = 0;

    void ensure_seeds(int64_t n) {
        if (seeds_capacity < n) {
            if (seeds) cudaFree(seeds);
            cudaMalloc(&seeds, n * sizeof(int32_t));
            seeds_capacity = n;
        }
    }

    void ensure_wd(int64_t n) {
        if (wd_capacity < n) {
            if (wd) cudaFree(wd);
            cudaMalloc(&wd, n * sizeof(double));
            wd_capacity = n;
        }
    }

    void ensure_bitmaps(int64_t n) {
        if (bitmaps_capacity < n) {
            if (bitmaps) cudaFree(bitmaps);
            cudaMalloc(&bitmaps, n * sizeof(uint32_t));
            bitmaps_capacity = n;
        }
    }

    void ensure_counts(int64_t n) {
        if (counts_capacity < n) {
            if (counts) cudaFree(counts);
            cudaMalloc(&counts, n * sizeof(int32_t));
            counts_capacity = n;
        }
    }

    void ensure_pair_offsets(int64_t n) {
        if (pair_offsets_capacity < n) {
            if (pair_offsets) cudaFree(pair_offsets);
            cudaMalloc(&pair_offsets, n * sizeof(int32_t));
            pair_offsets_capacity = n;
        }
    }

    ~Cache() override {
        if (seeds) cudaFree(seeds);
        if (wd) cudaFree(wd);
        if (bitmaps) cudaFree(bitmaps);
        if (counts) cudaFree(counts);
        if (pair_offsets) cudaFree(pair_offsets);
    }
};





__global__ void iota_kernel(int32_t* arr, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = idx;
}





__global__ void compute_weighted_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ weights,
    double* __restrict__ wd,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int start = offsets[v];
    int end = offsets[v + 1];
    double sum = 0.0;
    for (int i = start; i < end; i++) {
        sum += weights[i];
    }
    wd[v] = sum;
}






__global__ void build_2hop_bitmap_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    uint32_t* __restrict__ bitmaps,
    int32_t bitmap_words)
{
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    int u = seeds[seed_idx];
    uint32_t* my_bitmap = bitmaps + (int64_t)seed_idx * bitmap_words;

    int u_start = offsets[u];
    int u_end = offsets[u + 1];

    
    int blocks_y = gridDim.y;
    for (int i = u_start + (int)blockIdx.y; i < u_end; i += blocks_y) {
        int w = indices[i];
        int w_start = offsets[w];
        int w_end = offsets[w + 1];

        for (int j = w_start + threadIdx.x; j < w_end; j += blockDim.x) {
            int v = indices[j];
            if (v != u) {
                atomicOr(&my_bitmap[v >> 5], 1u << (v & 31));
            }
        }
    }
}





__global__ void count_bitmap_bits_kernel(
    const uint32_t* __restrict__ bitmaps,
    int32_t bitmap_words,
    int32_t num_seeds,
    int32_t* __restrict__ counts)
{
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    const uint32_t* my_bitmap = bitmaps + (int64_t)seed_idx * bitmap_words;

    int count = 0;
    for (int i = threadIdx.x; i < bitmap_words; i += blockDim.x) {
        count += __popc(my_bitmap[i]);
    }

    typedef cub::BlockReduce<int, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    int total = BlockReduce(temp).Sum(count);

    if (threadIdx.x == 0) {
        counts[seed_idx] = total;
    }
}





__global__ void enumerate_pairs_kernel(
    const uint32_t* __restrict__ bitmaps,
    int32_t bitmap_words,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int32_t* __restrict__ pair_offsets,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second)
{
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    int u = seeds[seed_idx];
    const uint32_t* my_bitmap = bitmaps + (int64_t)seed_idx * bitmap_words;
    int base = pair_offsets[seed_idx];

    __shared__ int write_pos;
    if (threadIdx.x == 0) write_pos = 0;
    __syncthreads();

    for (int word_idx = threadIdx.x; word_idx < bitmap_words; word_idx += blockDim.x) {
        uint32_t word = my_bitmap[word_idx];
        int cnt = __popc(word);

        if (cnt > 0) {
            int pos = atomicAdd(&write_pos, cnt);
            while (word) {
                int bit = __ffs(word) - 1;
                int v = word_idx * 32 + bit;
                out_first[base + pos] = u;
                out_second[base + pos] = v;
                pos++;
                word &= word - 1;
            }
        }
    }
}






__global__ void compute_overlap_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ weighted_degrees,
    const int32_t* __restrict__ pair_first,
    const int32_t* __restrict__ pair_second,
    double* __restrict__ scores,
    int32_t num_pairs)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int u = pair_first[warp_id];
    int v = pair_second[warp_id];

    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int u_deg = u_end - u_start;

    int v_start = offsets[v];
    int v_end = offsets[v + 1];
    int v_deg = v_end - v_start;

    
    const int32_t* small_idx;
    const double* small_wt;
    int small_deg;
    const int32_t* large_idx;
    const double* large_wt;
    int large_deg;

    if (u_deg <= v_deg) {
        small_idx = indices + u_start;
        small_wt = weights + u_start;
        small_deg = u_deg;
        large_idx = indices + v_start;
        large_wt = weights + v_start;
        large_deg = v_deg;
    } else {
        small_idx = indices + v_start;
        small_wt = weights + v_start;
        small_deg = v_deg;
        large_idx = indices + u_start;
        large_wt = weights + u_start;
        large_deg = u_deg;
    }

    double my_sum = 0.0;

    for (int s = lane; s < small_deg; s += 32) {
        int target = small_idx[s];
        double w_s = small_wt[s];

        
        int lo = 0, hi = large_deg;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (large_idx[mid] < target) lo = mid + 1;
            else hi = mid;
        }

        if (lo < large_deg && large_idx[lo] == target) {
            double w_l = large_wt[lo];
            my_sum += fmin(w_s, w_l);
        }
    }

    
    for (int offset = 16; offset > 0; offset >>= 1) {
        my_sum += __shfl_down_sync(0xffffffff, my_sum, offset);
    }

    if (lane == 0) {
        double wd_u = weighted_degrees[u];
        double wd_v = weighted_degrees[v];
        double min_wd = fmin(wd_u, wd_v);
        scores[warp_id] = (min_wd > 0.0) ? my_sum / min_wd : 0.0;
    }
}





__global__ void gather_int32_kernel(
    const int32_t* __restrict__ src,
    const int32_t* __restrict__ idx,
    int32_t* __restrict__ dst,
    int32_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
}

}  

similarity_result_double_t overlap_all_pairs_similarity_seg(const graph32_t& graph,
                                                            const double* edge_weights,
                                                            const int32_t* vertices,
                                                            std::size_t num_vertices,
                                                            std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_graph_vertices = graph.number_of_vertices;

    
    int32_t num_seeds;
    const int32_t* d_seeds;

    if (vertices != nullptr && num_vertices > 0) {
        num_seeds = static_cast<int32_t>(num_vertices);
        d_seeds = vertices;
    } else {
        
        num_seeds = num_graph_vertices;
        cache.ensure_seeds(num_graph_vertices);
        if (num_graph_vertices > 0) {
            int block = 256;
            int grid = (num_graph_vertices + block - 1) / block;
            iota_kernel<<<grid, block>>>(cache.seeds, num_graph_vertices);
        }
        d_seeds = cache.seeds;
    }

    
    cache.ensure_wd(num_graph_vertices);
    if (num_graph_vertices > 0) {
        int block = 256;
        int grid = (num_graph_vertices + block - 1) / block;
        compute_weighted_degrees_kernel<<<grid, block>>>(d_offsets, edge_weights, cache.wd, num_graph_vertices);
    }

    
    int32_t bitmap_words = (num_graph_vertices + 31) / 32;
    int64_t bitmap_total = (int64_t)num_seeds * bitmap_words;
    cache.ensure_bitmaps(bitmap_total);
    cudaMemsetAsync(cache.bitmaps, 0, bitmap_total * sizeof(uint32_t));

    
    int32_t blocks_per_seed = 1;
    if (num_seeds <= 200) {
        blocks_per_seed = std::max(1, 640 / std::max(1, (int)num_seeds));
        blocks_per_seed = std::min(blocks_per_seed, 32);
    }

    if (num_seeds > 0) {
        dim3 grid(num_seeds, blocks_per_seed);
        build_2hop_bitmap_kernel<<<grid, 256>>>(d_offsets, d_indices, d_seeds, num_seeds,
                                                cache.bitmaps, bitmap_words);
    }

    
    cache.ensure_counts((int64_t)num_seeds + 1);
    cudaMemsetAsync(cache.counts + num_seeds, 0, sizeof(int32_t)); 
    if (num_seeds > 0) {
        count_bitmap_bits_kernel<<<num_seeds, 256>>>(cache.bitmaps, bitmap_words, num_seeds, cache.counts);
    }

    
    cache.ensure_pair_offsets((int64_t)num_seeds + 1);
    {
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, cache.counts, cache.pair_offsets, num_seeds + 1);
        cudaMalloc(&d_temp, temp_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, cache.counts, cache.pair_offsets, num_seeds + 1);
        cudaFree(d_temp);
    }

    
    int32_t total_pairs;
    cudaMemcpy(&total_pairs, cache.pair_offsets + num_seeds, sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    if (total_pairs == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t* d_first = nullptr;
    int32_t* d_second = nullptr;
    cudaMalloc(&d_first, (size_t)total_pairs * sizeof(int32_t));
    cudaMalloc(&d_second, (size_t)total_pairs * sizeof(int32_t));
    if (num_seeds > 0) {
        enumerate_pairs_kernel<<<num_seeds, 256>>>(cache.bitmaps, bitmap_words, d_seeds, num_seeds,
                                                   cache.pair_offsets, d_first, d_second);
    }

    
    double* d_scores = nullptr;
    cudaMalloc(&d_scores, (size_t)total_pairs * sizeof(double));
    {
        int warps_per_block = 8;
        int block = warps_per_block * 32;
        int grid = (total_pairs + warps_per_block - 1) / warps_per_block;
        compute_overlap_kernel<<<grid, block>>>(d_offsets, d_indices, edge_weights, cache.wd,
                                                d_first, d_second, d_scores, total_pairs);
    }

    
    std::size_t output_count = total_pairs;

    if (topk.has_value() && (std::size_t)total_pairs > topk.value()) {
        std::size_t k = topk.value();
        int32_t topk_int = static_cast<int32_t>(k);

        
        int32_t* d_idx = nullptr;
        cudaMalloc(&d_idx, (size_t)total_pairs * sizeof(int32_t));
        iota_kernel<<<(total_pairs + 255) / 256, 256>>>(d_idx, total_pairs);

        
        double* d_sorted_scores = nullptr;
        int32_t* d_sorted_indices = nullptr;
        cudaMalloc(&d_sorted_scores, (size_t)total_pairs * sizeof(double));
        cudaMalloc(&d_sorted_indices, (size_t)total_pairs * sizeof(int32_t));
        {
            void* d_temp = nullptr;
            size_t temp_bytes = 0;
            cub::DeviceRadixSort::SortPairsDescending(d_temp, temp_bytes,
                d_scores, d_sorted_scores, d_idx, d_sorted_indices, total_pairs);
            cudaMalloc(&d_temp, temp_bytes);
            cub::DeviceRadixSort::SortPairsDescending(d_temp, temp_bytes,
                d_scores, d_sorted_scores, d_idx, d_sorted_indices, total_pairs);
            cudaFree(d_temp);
        }

        
        int32_t* d_out_first = nullptr;
        int32_t* d_out_second = nullptr;
        double* d_out_scores = nullptr;
        cudaMalloc(&d_out_first, k * sizeof(int32_t));
        cudaMalloc(&d_out_second, k * sizeof(int32_t));
        cudaMalloc(&d_out_scores, k * sizeof(double));

        
        if (topk_int > 0) {
            gather_int32_kernel<<<(topk_int + 255) / 256, 256>>>(d_first, d_sorted_indices, d_out_first, topk_int);
            gather_int32_kernel<<<(topk_int + 255) / 256, 256>>>(d_second, d_sorted_indices, d_out_second, topk_int);
        }

        
        cudaMemcpyAsync(d_out_scores, d_sorted_scores, k * sizeof(double), cudaMemcpyDeviceToDevice);

        
        cudaFree(d_idx);
        cudaFree(d_sorted_scores);
        cudaFree(d_sorted_indices);
        cudaFree(d_first);
        cudaFree(d_second);
        cudaFree(d_scores);

        return {d_out_first, d_out_second, d_out_scores, k};
    }

    return {d_first, d_second, d_scores, output_count};
}

}  
