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

struct Cache : Cacheable {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    int64_t* h_pinned = nullptr;

    Cache() {
        cudaHostAlloc(&h_pinned, 32, cudaHostAllocDefault);
    }

    ~Cache() override {
        if (d_temp_storage) cudaFree(d_temp_storage);
        if (h_pinned) cudaFreeHost(h_pinned);
    }

    void ensure_temp(size_t bytes) {
        if (bytes > temp_storage_bytes) {
            if (d_temp_storage) cudaFree(d_temp_storage);
            temp_storage_bytes = bytes;
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
        }
    }
};




__global__ void compute_weight_sums(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ weights,
    double* __restrict__ weight_sums,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];
        double sum = 0.0;
        for (int i = start; i < end; i++) {
            sum += weights[i];
        }
        weight_sums[v] = sum;
    }
}






__device__ __forceinline__ int32_t get_seed(const int32_t* seeds, int seed_idx) {
    return seeds ? __ldg(&seeds[seed_idx]) : seed_idx;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ void build_bitmap(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int u, int bitmap_words,
    uint32_t* __restrict__ bitmap)
{
    const int WS = 32;
    int warp_id = threadIdx.x / WS;
    int lane_id = threadIdx.x & (WS - 1);
    int num_warps = BLOCK_SIZE / WS;

    for (int i = threadIdx.x; i < bitmap_words; i += BLOCK_SIZE)
        bitmap[i] = 0;
    __syncthreads();

    int u_start = __ldg(&offsets[u]);
    int u_end = __ldg(&offsets[u + 1]);
    int u_degree = u_end - u_start;

    for (int ni = warp_id; ni < u_degree; ni += num_warps) {
        int n = __ldg(&indices[u_start + ni]);
        int n_start = __ldg(&offsets[n]);
        int n_end = __ldg(&offsets[n + 1]);
        for (int e2 = n_start + lane_id; e2 < n_end; e2 += WS) {
            int v = __ldg(&indices[e2]);
            atomicOr(&bitmap[v >> 5], 1u << (v & 31));
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
        bitmap[u >> 5] &= ~(1u << (u & 31));
    __syncthreads();
}

template <int BLOCK_SIZE>
__device__ __forceinline__ int count_bitmap(uint32_t* bitmap, int bitmap_words) {
    int local_count = 0;
    for (int i = threadIdx.x; i < bitmap_words; i += BLOCK_SIZE)
        local_count += __popc(bitmap[i]);

    typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    int total = BlockReduce(temp).Sum(local_count);
    __shared__ int bcast;
    if (threadIdx.x == 0) bcast = total;
    __syncthreads();
    return bcast;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ void output_pairs_atomic(
    uint32_t* bitmap, int bitmap_words,
    int u, int64_t base, int64_t max_output,
    int32_t* out_first, int32_t* out_second)
{
    __shared__ int write_pos;
    if (threadIdx.x == 0) write_pos = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < bitmap_words; i += BLOCK_SIZE) {
        uint32_t word = bitmap[i];
        while (word) {
            int bit = __ffs(word) - 1;
            word &= word - 1;
            int v = i * 32 + bit;
            int pos = atomicAdd(&write_pos, 1);
            int64_t gpos = base + pos;
            if (gpos < max_output) {
                out_first[gpos] = u;
                out_second[gpos] = v;
            }
        }
    }
}




template <int BLOCK_SIZE>
__global__ void count_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds, int32_t bitmap_words,
    int64_t* __restrict__ counts)
{
    extern __shared__ uint32_t bitmap[];
    int si = blockIdx.x;
    if (si >= num_seeds) return;

    int u = get_seed(seeds, si);
    build_bitmap<BLOCK_SIZE>(offsets, indices, u, bitmap_words, bitmap);
    int total = count_bitmap<BLOCK_SIZE>(bitmap, bitmap_words);
    if (threadIdx.x == 0) counts[si] = total;
}




template <int BLOCK_SIZE>
__global__ void output_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds, int32_t bitmap_words,
    const int64_t* __restrict__ off,
    int32_t* __restrict__ out_first, int32_t* __restrict__ out_second,
    int64_t max_out)
{
    extern __shared__ uint32_t bitmap[];
    int si = blockIdx.x;
    if (si >= num_seeds) return;

    int u = get_seed(seeds, si);
    int64_t base = off[si];
    if (base >= max_out) return;

    build_bitmap<BLOCK_SIZE>(offsets, indices, u, bitmap_words, bitmap);
    output_pairs_atomic<BLOCK_SIZE>(bitmap, bitmap_words, u, base, max_out,
                                     out_first, out_second);
}




template <int BLOCK_SIZE>
__global__ void count_kernel_gmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds, int32_t bitmap_words,
    int64_t* __restrict__ counts,
    uint32_t* __restrict__ global_bitmaps)
{
    int si = blockIdx.x;
    if (si >= num_seeds) return;

    uint32_t* bitmap = global_bitmaps + (int64_t)si * bitmap_words;
    int u = get_seed(seeds, si);
    build_bitmap<BLOCK_SIZE>(offsets, indices, u, bitmap_words, bitmap);
    int total = count_bitmap<BLOCK_SIZE>(bitmap, bitmap_words);
    if (threadIdx.x == 0) counts[si] = total;
}

template <int BLOCK_SIZE>
__global__ void output_kernel_gmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds, int32_t bitmap_words,
    const int64_t* __restrict__ off,
    int32_t* __restrict__ out_first, int32_t* __restrict__ out_second,
    int64_t max_out,
    uint32_t* __restrict__ global_bitmaps)
{
    int si = blockIdx.x;
    if (si >= num_seeds) return;

    uint32_t* bitmap = global_bitmaps + (int64_t)si * bitmap_words;
    int u = get_seed(seeds, si);
    int64_t base = off[si];
    if (base >= max_out) return;

    build_bitmap<BLOCK_SIZE>(offsets, indices, u, bitmap_words, bitmap);
    output_pairs_atomic<BLOCK_SIZE>(bitmap, bitmap_words, u, base, max_out,
                                     out_first, out_second);
}




__global__ void compute_jaccard_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ weight_sums,
    const int32_t* __restrict__ pair_first,
    const int32_t* __restrict__ pair_second,
    double* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int u = pair_first[idx];
    int v = pair_second[idx];

    int off_u = offsets[u];
    int deg_u = offsets[u + 1] - off_u;
    int off_v = offsets[v];
    int deg_v = offsets[v + 1] - off_v;

    const int32_t* short_idx;
    const double* short_wt;
    int short_len;
    const int32_t* long_idx;
    const double* long_wt;
    int long_len;

    if (deg_u <= deg_v) {
        short_idx = indices + off_u;
        short_wt = weights + off_u;
        short_len = deg_u;
        long_idx = indices + off_v;
        long_wt = weights + off_v;
        long_len = deg_v;
    } else {
        short_idx = indices + off_v;
        short_wt = weights + off_v;
        short_len = deg_v;
        long_idx = indices + off_u;
        long_wt = weights + off_u;
        long_len = deg_u;
    }

    double isect = 0.0;
    for (int k = 0; k < short_len; k++) {
        int target = short_idx[k];
        int lo = 0, hi = long_len;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (long_idx[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        if (lo < long_len && long_idx[lo] == target) {
            isect += fmin(short_wt[k], long_wt[lo]);
        }
    }

    double denom = weight_sums[u] + weight_sums[v] - isect;
    scores[idx] = (denom > 0.0) ? isect / denom : 0.0;
}




__global__ void iota_kernel(int32_t* arr, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = (int32_t)idx;
}

__global__ void gather_results(
    const int32_t* __restrict__ src_first,
    const int32_t* __restrict__ src_second,
    const double* __restrict__ src_scores,
    const int32_t* __restrict__ sorted_indices,
    int32_t* __restrict__ dst_first,
    int32_t* __restrict__ dst_second,
    double* __restrict__ dst_scores,
    int64_t count
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int src_idx = sorted_indices[idx];
        dst_first[idx] = src_first[src_idx];
        dst_second[idx] = src_second[src_idx];
        dst_scores[idx] = src_scores[src_idx];
    }
}

int get_smem(int32_t bw) { return bw * 4 + 1024; }

}  

similarity_result_double_t jaccard_all_pairs_similarity_seg(
    const graph32_t& graph,
    const double* edge_weights,
    const int32_t* vertices,
    std::size_t num_vertices_param,
    std::optional<std::size_t> topk)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    const double* d_weights = edge_weights;
    cudaStream_t stream = 0;

    
    const int32_t* d_seeds = vertices;
    int32_t num_seeds;
    if (vertices != nullptr && num_vertices_param > 0) {
        num_seeds = static_cast<int32_t>(num_vertices_param);
    } else {
        num_seeds = num_vertices;
        d_seeds = nullptr;
    }

    if (num_seeds == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    int32_t bitmap_words = (num_vertices + 31) >> 5;

    
    double* d_weight_sums;
    cudaMalloc(&d_weight_sums, (size_t)num_vertices * sizeof(double));
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        compute_weight_sums<<<grid, block, 0, stream>>>(d_offsets, d_weights, d_weight_sums, num_vertices);
    }

    
    bool use_gmem = ((int64_t)bitmap_words * (int)sizeof(uint32_t) > 96 * 1024);
    uint32_t* d_bitmaps = nullptr;
    if (use_gmem) {
        cudaMalloc(&d_bitmaps, (size_t)num_seeds * bitmap_words * sizeof(uint32_t));
    }

    
    int64_t* d_counts;
    cudaMalloc(&d_counts, (size_t)num_seeds * sizeof(int64_t));
    if (use_gmem) {
        if (num_seeds > 0) {
            count_kernel_gmem<512><<<num_seeds, 512, 0, stream>>>(
                d_offsets, d_indices, d_seeds, num_seeds, bitmap_words,
                d_counts, d_bitmaps);
        }
    } else {
        if (num_seeds > 0) {
            int sm = get_smem(bitmap_words);
            cudaFuncSetAttribute(count_kernel<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
            count_kernel<512><<<num_seeds, 512, sm, stream>>>(
                d_offsets, d_indices, d_seeds, num_seeds, bitmap_words, d_counts);
        }
    }

    
    int64_t* d_off;
    cudaMalloc(&d_off, (size_t)num_seeds * sizeof(int64_t));
    {
        size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_counts, d_off, num_seeds, stream);
        cache.ensure_temp(temp_bytes);
        cub::DeviceScan::ExclusiveSum(cache.d_temp_storage, temp_bytes, d_counts, d_off, num_seeds, stream);
    }

    
    cudaMemcpyAsync(&cache.h_pinned[0], d_off + (num_seeds - 1),
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&cache.h_pinned[1], d_counts + (num_seeds - 1),
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int64_t total = cache.h_pinned[0] + cache.h_pinned[1];

    if (total == 0) {
        cudaFree(d_weight_sums);
        if (d_bitmaps) cudaFree(d_bitmaps);
        cudaFree(d_counts);
        cudaFree(d_off);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t* pair_first;
    int32_t* pair_second;
    cudaMalloc(&pair_first, (size_t)total * sizeof(int32_t));
    cudaMalloc(&pair_second, (size_t)total * sizeof(int32_t));

    if (use_gmem) {
        if (num_seeds > 0) {
            output_kernel_gmem<512><<<num_seeds, 512, 0, stream>>>(
                d_offsets, d_indices, d_seeds, num_seeds, bitmap_words,
                d_off, pair_first, pair_second, total, d_bitmaps);
        }
    } else {
        if (num_seeds > 0) {
            int sm = get_smem(bitmap_words);
            cudaFuncSetAttribute(output_kernel<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
            output_kernel<512><<<num_seeds, 512, sm, stream>>>(
                d_offsets, d_indices, d_seeds, num_seeds, bitmap_words,
                d_off, pair_first, pair_second, total);
        }
    }

    
    int64_t num_pairs = total;
    double* scores;
    cudaMalloc(&scores, (size_t)num_pairs * sizeof(double));
    {
        int block = 256;
        int grid = (int)((num_pairs + block - 1) / block);
        compute_jaccard_kernel<<<grid, block, 0, stream>>>(
            d_offsets, d_indices, d_weights, d_weight_sums,
            pair_first, pair_second, scores, num_pairs);
    }

    
    cudaFree(d_weight_sums);
    if (d_bitmaps) cudaFree(d_bitmaps);
    cudaFree(d_counts);
    cudaFree(d_off);

    
    bool use_topk = topk.has_value() && (int64_t)topk.value() < num_pairs;

    if (use_topk) {
        int32_t topk_val = static_cast<int32_t>(topk.value());

        int32_t* idx;
        cudaMalloc(&idx, (size_t)num_pairs * sizeof(int32_t));
        {
            int block = 256;
            int grid = (int)((num_pairs + block - 1) / block);
            iota_kernel<<<grid, block, 0, stream>>>(idx, num_pairs);
        }

        double* scores_sorted;
        int32_t* idx_sorted;
        cudaMalloc(&scores_sorted, (size_t)num_pairs * sizeof(double));
        cudaMalloc(&idx_sorted, (size_t)num_pairs * sizeof(int32_t));

        size_t sort_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, sort_bytes, scores, scores_sorted,
            idx, idx_sorted, num_pairs, 0, 64, stream);

        cache.ensure_temp(sort_bytes);
        cub::DeviceRadixSort::SortPairsDescending(
            cache.d_temp_storage, sort_bytes, scores, scores_sorted,
            idx, idx_sorted, num_pairs, 0, 64, stream);

        int32_t* final_first;
        int32_t* final_second;
        double* final_scores;
        cudaMalloc(&final_first, (size_t)topk_val * sizeof(int32_t));
        cudaMalloc(&final_second, (size_t)topk_val * sizeof(int32_t));
        cudaMalloc(&final_scores, (size_t)topk_val * sizeof(double));

        {
            int block = 256;
            int grid = (int)((topk_val + block - 1) / block);
            gather_results<<<grid, block, 0, stream>>>(
                pair_first, pair_second, scores, idx_sorted,
                final_first, final_second, final_scores, topk_val);
        }

        cudaStreamSynchronize(stream);

        
        cudaFree(pair_first);
        cudaFree(pair_second);
        cudaFree(scores);
        cudaFree(idx);
        cudaFree(scores_sorted);
        cudaFree(idx_sorted);

        return {final_first, final_second, final_scores, (std::size_t)topk_val};
    }

    
    cudaStreamSynchronize(stream);
    return {pair_first, pair_second, scores, (std::size_t)num_pairs};
}

}  
