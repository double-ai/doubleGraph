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

struct Cache : Cacheable {
    int64_t* h_pinned = nullptr;

    void* d_prefix_temp = nullptr;
    size_t prefix_temp_bytes = 0;

    double* d_weight_sums = nullptr;
    int64_t weight_sums_cap = 0;

    int64_t* d_counts = nullptr;
    int64_t counts_cap = 0;

    int64_t* d_off = nullptr;
    int64_t off_cap = 0;

    uint32_t* d_bitmaps = nullptr;
    int64_t bitmaps_cap = 0;

    int32_t* d_idx = nullptr;
    int64_t idx_cap = 0;

    double* d_scores_sorted = nullptr;
    int64_t scores_sorted_cap = 0;

    int32_t* d_idx_sorted = nullptr;
    int64_t idx_sorted_cap = 0;

    void* d_sort_temp = nullptr;
    size_t sort_temp_bytes = 0;

    Cache() {
        cudaHostAlloc(&h_pinned, 32, cudaHostAllocDefault);
    }

    ~Cache() override {
        if (h_pinned) cudaFreeHost(h_pinned);
        if (d_prefix_temp) cudaFree(d_prefix_temp);
        if (d_weight_sums) cudaFree(d_weight_sums);
        if (d_counts) cudaFree(d_counts);
        if (d_off) cudaFree(d_off);
        if (d_bitmaps) cudaFree(d_bitmaps);
        if (d_idx) cudaFree(d_idx);
        if (d_scores_sorted) cudaFree(d_scores_sorted);
        if (d_idx_sorted) cudaFree(d_idx_sorted);
        if (d_sort_temp) cudaFree(d_sort_temp);
    }

    void ensure_weight_sums(int64_t n) {
        if (weight_sums_cap < n) {
            if (d_weight_sums) cudaFree(d_weight_sums);
            cudaMalloc(&d_weight_sums, n * sizeof(double));
            weight_sums_cap = n;
        }
    }

    void ensure_counts(int64_t n) {
        if (counts_cap < n) {
            if (d_counts) cudaFree(d_counts);
            cudaMalloc(&d_counts, n * sizeof(int64_t));
            counts_cap = n;
        }
    }

    void ensure_off(int64_t n) {
        if (off_cap < n) {
            if (d_off) cudaFree(d_off);
            cudaMalloc(&d_off, n * sizeof(int64_t));
            off_cap = n;
        }
    }

    void ensure_bitmaps(int64_t n) {
        if (bitmaps_cap < n) {
            if (d_bitmaps) cudaFree(d_bitmaps);
            cudaMalloc(&d_bitmaps, n * sizeof(uint32_t));
            bitmaps_cap = n;
        }
    }

    void ensure_prefix_temp(size_t bytes) {
        if (prefix_temp_bytes < bytes) {
            if (d_prefix_temp) cudaFree(d_prefix_temp);
            prefix_temp_bytes = bytes;
            cudaMalloc(&d_prefix_temp, prefix_temp_bytes);
        }
    }

    void ensure_idx(int64_t n) {
        if (idx_cap < n) {
            if (d_idx) cudaFree(d_idx);
            cudaMalloc(&d_idx, n * sizeof(int32_t));
            idx_cap = n;
        }
    }

    void ensure_scores_sorted(int64_t n) {
        if (scores_sorted_cap < n) {
            if (d_scores_sorted) cudaFree(d_scores_sorted);
            cudaMalloc(&d_scores_sorted, n * sizeof(double));
            scores_sorted_cap = n;
        }
    }

    void ensure_idx_sorted(int64_t n) {
        if (idx_sorted_cap < n) {
            if (d_idx_sorted) cudaFree(d_idx_sorted);
            cudaMalloc(&d_idx_sorted, n * sizeof(int32_t));
            idx_sorted_cap = n;
        }
    }

    void ensure_sort_temp(size_t bytes) {
        if (sort_temp_bytes < bytes) {
            if (d_sort_temp) cudaFree(d_sort_temp);
            sort_temp_bytes = bytes;
            cudaMalloc(&d_sort_temp, sort_temp_bytes);
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
    std::size_t num_vertices,
    std::optional<std::size_t> topk)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const double* d_weights = edge_weights;
    int32_t nv = graph.number_of_vertices;
    cudaStream_t stream = 0;

    
    const int32_t* d_seeds;
    int32_t num_seeds;
    if (vertices != nullptr && num_vertices > 0) {
        d_seeds = vertices;
        num_seeds = static_cast<int32_t>(num_vertices);
    } else {
        d_seeds = nullptr;
        num_seeds = nv;
    }

    if (num_seeds == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    int32_t bitmap_words = (nv + 31) >> 5;

    
    cache.ensure_weight_sums(nv);
    double* d_weight_sums = cache.d_weight_sums;
    {
        int block = 256;
        int grid = (nv + block - 1) / block;
        compute_weight_sums<<<grid, block, 0, stream>>>(d_offsets, d_weights, d_weight_sums, nv);
    }

    
    bool use_gmem = ((int64_t)bitmap_words * (int)sizeof(uint32_t) > 96 * 1024);
    if (use_gmem) {
        int64_t bitmap_total = (int64_t)num_seeds * bitmap_words;
        cache.ensure_bitmaps(bitmap_total);
    }

    
    cache.ensure_counts(num_seeds);
    if (use_gmem) {
        count_kernel_gmem<512><<<num_seeds, 512, 0, stream>>>(
            d_offsets, d_indices, d_seeds, num_seeds, bitmap_words,
            cache.d_counts, cache.d_bitmaps);
    } else {
        int sm = get_smem(bitmap_words);
        cudaFuncSetAttribute(count_kernel<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
        count_kernel<512><<<num_seeds, 512, sm, stream>>>(
            d_offsets, d_indices, d_seeds, num_seeds, bitmap_words, cache.d_counts);
    }

    
    cache.ensure_off(num_seeds);
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, cache.d_counts, cache.d_off, num_seeds, stream);
    cache.ensure_prefix_temp(temp_bytes);
    cub::DeviceScan::ExclusiveSum(cache.d_prefix_temp, temp_bytes,
                                   cache.d_counts, cache.d_off, num_seeds, stream);

    
    int64_t* h_pinned = cache.h_pinned;
    cudaMemcpyAsync(&h_pinned[0], cache.d_off + (num_seeds - 1),
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_pinned[1], cache.d_counts + (num_seeds - 1),
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int64_t total = h_pinned[0] + h_pinned[1];

    if (total == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t* pair_first = nullptr;
    int32_t* pair_second = nullptr;
    cudaMalloc(&pair_first, total * sizeof(int32_t));
    cudaMalloc(&pair_second, total * sizeof(int32_t));

    if (use_gmem) {
        output_kernel_gmem<512><<<num_seeds, 512, 0, stream>>>(
            d_offsets, d_indices, d_seeds, num_seeds, bitmap_words,
            cache.d_off, pair_first, pair_second, total, cache.d_bitmaps);
    } else {
        int sm = get_smem(bitmap_words);
        cudaFuncSetAttribute(output_kernel<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
        output_kernel<512><<<num_seeds, 512, sm, stream>>>(
            d_offsets, d_indices, d_seeds, num_seeds, bitmap_words,
            cache.d_off, pair_first, pair_second, total);
    }

    
    int64_t num_pairs = total;
    double* scores = nullptr;
    cudaMalloc(&scores, num_pairs * sizeof(double));
    {
        int block_size = 256;
        int grid = (int)((num_pairs + block_size - 1) / block_size);
        compute_jaccard_kernel<<<grid, block_size, 0, stream>>>(
            d_offsets, d_indices, d_weights, d_weight_sums,
            pair_first, pair_second, scores, num_pairs);
    }

    
    bool use_topk = topk.has_value() && static_cast<int64_t>(topk.value()) < num_pairs;

    if (use_topk) {
        int32_t topk_val = static_cast<int32_t>(topk.value());

        
        cache.ensure_idx(num_pairs);
        {
            int block = 256;
            int grid = (int)((num_pairs + block - 1) / block);
            iota_kernel<<<grid, block, 0, stream>>>(cache.d_idx, num_pairs);
        }

        
        cache.ensure_scores_sorted(num_pairs);
        cache.ensure_idx_sorted(num_pairs);

        size_t sort_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, sort_bytes, scores, cache.d_scores_sorted,
            cache.d_idx, cache.d_idx_sorted, num_pairs, 0, 64, stream);

        cache.ensure_sort_temp(sort_bytes);
        cub::DeviceRadixSort::SortPairsDescending(
            cache.d_sort_temp, sort_bytes, scores, cache.d_scores_sorted,
            cache.d_idx, cache.d_idx_sorted, num_pairs, 0, 64, stream);

        
        int32_t* final_first = nullptr;
        int32_t* final_second = nullptr;
        double* final_scores = nullptr;
        cudaMalloc(&final_first, topk_val * sizeof(int32_t));
        cudaMalloc(&final_second, topk_val * sizeof(int32_t));
        cudaMalloc(&final_scores, topk_val * sizeof(double));

        {
            int block = 256;
            int grid = (int)((topk_val + block - 1) / block);
            gather_results<<<grid, block, 0, stream>>>(
                pair_first, pair_second, scores, cache.d_idx_sorted,
                final_first, final_second, final_scores, topk_val);
        }

        cudaStreamSynchronize(stream);

        
        cudaFree(pair_first);
        cudaFree(pair_second);
        cudaFree(scores);

        return {final_first, final_second, final_scores, static_cast<std::size_t>(topk_val)};
    }

    
    cudaStreamSynchronize(stream);
    return {pair_first, pair_second, scores, static_cast<std::size_t>(num_pairs)};
}

}  
