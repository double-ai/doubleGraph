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

namespace aai {

namespace {



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
    int32_t* out_first, int32_t* out_second, float* out_scores)
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
                out_scores[gpos] = 1.0f;
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
    float* __restrict__ out_scores, int64_t max_out)
{
    extern __shared__ uint32_t bitmap[];
    int si = blockIdx.x;
    if (si >= num_seeds) return;

    int u = get_seed(seeds, si);
    int64_t base = off[si];
    if (base >= max_out) return;

    build_bitmap<BLOCK_SIZE>(offsets, indices, u, bitmap_words, bitmap);
    output_pairs_atomic<BLOCK_SIZE>(bitmap, bitmap_words, u, base, max_out,
                                     out_first, out_second, out_scores);
}


template <int BLOCK_SIZE>
__global__ void topk_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds, int32_t bitmap_words,
    int64_t topk, unsigned long long* __restrict__ gctr,
    int32_t* __restrict__ out_first, int32_t* __restrict__ out_second,
    float* __restrict__ out_scores)
{
    extern __shared__ uint32_t bitmap[];
    int si = blockIdx.x;
    if (si >= num_seeds) return;
    if ((long long)(*gctr) >= topk) return;

    int u = get_seed(seeds, si);
    build_bitmap<BLOCK_SIZE>(offsets, indices, u, bitmap_words, bitmap);
    int total = count_bitmap<BLOCK_SIZE>(bitmap, bitmap_words);

    if (total == 0) return;

    __shared__ int64_t block_base;
    if (threadIdx.x == 0)
        block_base = (int64_t)atomicAdd(gctr, (unsigned long long)total);
    __syncthreads();

    if (block_base >= topk) return;

    output_pairs_atomic<BLOCK_SIZE>(bitmap, bitmap_words, u, block_base, topk,
                                     out_first, out_second, out_scores);
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
    float* __restrict__ out_scores, int64_t max_out,
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
                                     out_first, out_second, out_scores);
}

template <int BLOCK_SIZE>
__global__ void topk_kernel_gmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds, int32_t bitmap_words,
    int64_t topk, unsigned long long* __restrict__ gctr,
    int32_t* __restrict__ out_first, int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    uint32_t* __restrict__ global_bitmaps)
{
    int si = blockIdx.x;
    if (si >= num_seeds) return;
    if ((long long)(*gctr) >= topk) return;

    uint32_t* bitmap = global_bitmaps + (int64_t)si * bitmap_words;
    int u = get_seed(seeds, si);
    build_bitmap<BLOCK_SIZE>(offsets, indices, u, bitmap_words, bitmap);
    int total = count_bitmap<BLOCK_SIZE>(bitmap, bitmap_words);

    if (total == 0) return;

    __shared__ int64_t block_base;
    if (threadIdx.x == 0)
        block_base = (int64_t)atomicAdd(gctr, (unsigned long long)total);
    __syncthreads();

    if (block_base >= topk) return;

    output_pairs_atomic<BLOCK_SIZE>(bitmap, bitmap_words, u, block_base, topk,
                                     out_first, out_second, out_scores);
}



int get_smem(int32_t bw) { return bw * 4 + 1024; }

void launch_count_pairs(const int32_t* off, const int32_t* idx,
    const int32_t* seeds, int32_t n, int32_t bw,
    int64_t* counts, cudaStream_t s) {
    if (n <= 0) return;
    int sm = get_smem(bw);
    cudaFuncSetAttribute(count_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    count_kernel<256><<<n, 256, sm, s>>>(off, idx, seeds, n, bw, counts);
}

void launch_output_pairs(const int32_t* off, const int32_t* idx,
    const int32_t* seeds, int32_t n, int32_t bw,
    const int64_t* offsets, int32_t* f, int32_t* s, float* sc,
    int64_t max_out, cudaStream_t st) {
    if (n <= 0) return;
    int sm = get_smem(bw);
    cudaFuncSetAttribute(output_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    output_kernel<256><<<n, 256, sm, st>>>(off, idx, seeds, n, bw, offsets, f, s, sc, max_out);
}

void launch_single_pass_topk(const int32_t* off, const int32_t* idx,
    const int32_t* seeds, int32_t n, int32_t bw,
    int64_t topk, unsigned long long* ctr,
    int32_t* f, int32_t* s, float* sc, cudaStream_t st) {
    if (n <= 0) return;
    int sm = get_smem(bw);
    cudaFuncSetAttribute(topk_kernel<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
    topk_kernel<256><<<n, 256, sm, st>>>(off, idx, seeds, n, bw, topk, ctr, f, s, sc);
}

void launch_prefix_sum(const int64_t* counts, int64_t* out,
    int32_t n, void* temp, size_t* bytes, cudaStream_t st) {
    cub::DeviceScan::ExclusiveSum(temp, *bytes, counts, out, n, st);
}

void launch_count_pairs_gmem(const int32_t* off, const int32_t* idx,
    const int32_t* seeds, int32_t n, int32_t bw,
    int64_t* counts, uint32_t* global_bitmaps, cudaStream_t s) {
    if (n <= 0) return;
    count_kernel_gmem<256><<<n, 256, 0, s>>>(off, idx, seeds, n, bw, counts, global_bitmaps);
}

void launch_output_pairs_gmem(const int32_t* off, const int32_t* idx,
    const int32_t* seeds, int32_t n, int32_t bw,
    const int64_t* offsets, int32_t* f, int32_t* s, float* sc,
    int64_t max_out, uint32_t* global_bitmaps, cudaStream_t st) {
    if (n <= 0) return;
    output_kernel_gmem<256><<<n, 256, 0, st>>>(off, idx, seeds, n, bw, offsets, f, s, sc, max_out, global_bitmaps);
}

void launch_single_pass_topk_gmem(const int32_t* off, const int32_t* idx,
    const int32_t* seeds, int32_t n, int32_t bw,
    int64_t topk, unsigned long long* ctr,
    int32_t* f, int32_t* s, float* sc,
    uint32_t* global_bitmaps, cudaStream_t st) {
    if (n <= 0) return;
    topk_kernel_gmem<256><<<n, 256, 0, st>>>(off, idx, seeds, n, bw, topk, ctr, f, s, sc, global_bitmaps);
}



struct Cache : Cacheable {
    int64_t* h_pinned = nullptr;

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    int64_t* d_counts = nullptr;
    int64_t counts_capacity = 0;

    int64_t* d_off = nullptr;
    int64_t off_capacity = 0;

    unsigned long long* d_counter = nullptr;

    uint32_t* d_bitmaps = nullptr;
    int64_t bitmaps_capacity = 0;

    Cache() {
        cudaHostAlloc(&h_pinned, 32, cudaHostAllocDefault);
        cudaMalloc(&d_counter, sizeof(unsigned long long));
    }

    ~Cache() override {
        if (h_pinned) cudaFreeHost(h_pinned);
        if (d_temp_storage) cudaFree(d_temp_storage);
        if (d_counts) cudaFree(d_counts);
        if (d_off) cudaFree(d_off);
        if (d_counter) cudaFree(d_counter);
        if (d_bitmaps) cudaFree(d_bitmaps);
    }

    void ensure_counts(int64_t n) {
        if (counts_capacity < n) {
            if (d_counts) cudaFree(d_counts);
            cudaMalloc(&d_counts, n * sizeof(int64_t));
            counts_capacity = n;
        }
    }

    void ensure_off(int64_t n) {
        if (off_capacity < n) {
            if (d_off) cudaFree(d_off);
            cudaMalloc(&d_off, n * sizeof(int64_t));
            off_capacity = n;
        }
    }

    void ensure_bitmaps(int64_t n) {
        if (bitmaps_capacity < n) {
            if (d_bitmaps) cudaFree(d_bitmaps);
            cudaMalloc(&d_bitmaps, n * sizeof(uint32_t));
            bitmaps_capacity = n;
        }
    }

    void ensure_temp_storage(size_t bytes) {
        if (temp_storage_bytes < bytes) {
            if (d_temp_storage) cudaFree(d_temp_storage);
            temp_storage_bytes = bytes;
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
        }
    }
};

}  



similarity_result_float_t cosine_all_pairs_similarity_seg(const graph32_t& graph,
                                                          const int32_t* vertices,
                                                          std::size_t num_vertices,
                                                          std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n_verts = graph.number_of_vertices;
    cudaStream_t stream = 0;
    int32_t bitmap_words = (n_verts + 31) >> 5;

    
    const auto& seg = graph.segment_offsets.value();
    int32_t non_zero_vertices = seg[3];

    
    const int32_t* d_seeds = vertices;
    int32_t num_seeds;
    if (vertices != nullptr) {
        num_seeds = static_cast<int32_t>(num_vertices);
    } else {
        num_seeds = non_zero_vertices;
    }

    if (num_seeds == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    bool use_gmem = ((int64_t)bitmap_words * (int)sizeof(uint32_t) > 160 * 1024);
    uint32_t* d_bitmaps = nullptr;
    if (use_gmem) {
        cache.ensure_bitmaps((int64_t)num_seeds * bitmap_words);
        d_bitmaps = cache.d_bitmaps;
    }

    if (topk.has_value()) {
        
        int64_t topk_val = static_cast<int64_t>(topk.value());
        if (topk_val == 0) {
            return {nullptr, nullptr, nullptr, 0};
        }

        int32_t* out_first;
        int32_t* out_second;
        float* out_scores;
        cudaMalloc(&out_first, topk_val * sizeof(int32_t));
        cudaMalloc(&out_second, topk_val * sizeof(int32_t));
        cudaMalloc(&out_scores, topk_val * sizeof(float));

        cudaMemsetAsync(cache.d_counter, 0, sizeof(unsigned long long), stream);

        if (use_gmem) {
            launch_single_pass_topk_gmem(d_offsets, d_indices, d_seeds, num_seeds,
                bitmap_words, topk_val, cache.d_counter,
                out_first, out_second, out_scores, d_bitmaps, stream);
        } else {
            launch_single_pass_topk(d_offsets, d_indices, d_seeds, num_seeds,
                bitmap_words, topk_val, cache.d_counter,
                out_first, out_second, out_scores, stream);
        }

        cudaMemcpyAsync(&cache.h_pinned[0], cache.d_counter,
                       sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int64_t actual = std::min(cache.h_pinned[0], topk_val);

        if (actual < topk_val) {
            if (actual == 0) {
                cudaFree(out_first);
                cudaFree(out_second);
                cudaFree(out_scores);
                return {nullptr, nullptr, nullptr, 0};
            }
            int32_t* f1;
            int32_t* f2;
            float* fs;
            cudaMalloc(&f1, actual * sizeof(int32_t));
            cudaMalloc(&f2, actual * sizeof(int32_t));
            cudaMalloc(&fs, actual * sizeof(float));
            cudaMemcpyAsync(f1, out_first, actual * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(f2, out_second, actual * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(fs, out_scores, actual * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaFree(out_first);
            cudaFree(out_second);
            cudaFree(out_scores);
            return {f1, f2, fs, static_cast<std::size_t>(actual)};
        }
        return {out_first, out_second, out_scores, static_cast<std::size_t>(actual)};
    } else {
        
        cache.ensure_counts(num_seeds);

        if (use_gmem) {
            launch_count_pairs_gmem(d_offsets, d_indices, d_seeds, num_seeds,
                              bitmap_words, cache.d_counts, d_bitmaps, stream);
        } else {
            launch_count_pairs(d_offsets, d_indices, d_seeds, num_seeds,
                              bitmap_words, cache.d_counts, stream);
        }

        cache.ensure_off(num_seeds);
        size_t temp_bytes = 0;
        launch_prefix_sum(cache.d_counts, cache.d_off, num_seeds, nullptr, &temp_bytes, stream);
        cache.ensure_temp_storage(temp_bytes);
        launch_prefix_sum(cache.d_counts, cache.d_off, num_seeds, cache.d_temp_storage, &temp_bytes, stream);

        
        cudaMemcpyAsync(&cache.h_pinned[0], cache.d_off + (num_seeds - 1),
                       sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&cache.h_pinned[1], cache.d_counts + (num_seeds - 1),
                       sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int64_t total = cache.h_pinned[0] + cache.h_pinned[1];

        if (total == 0) {
            return {nullptr, nullptr, nullptr, 0};
        }

        int32_t* out_first;
        int32_t* out_second;
        float* out_scores;
        cudaMalloc(&out_first, total * sizeof(int32_t));
        cudaMalloc(&out_second, total * sizeof(int32_t));
        cudaMalloc(&out_scores, total * sizeof(float));

        if (use_gmem) {
            launch_output_pairs_gmem(d_offsets, d_indices, d_seeds, num_seeds, bitmap_words,
                               cache.d_off, out_first, out_second, out_scores, total, d_bitmaps, stream);
        } else {
            launch_output_pairs(d_offsets, d_indices, d_seeds, num_seeds, bitmap_words,
                               cache.d_off, out_first, out_second, out_scores, total, stream);
        }

        return {out_first, out_second, out_scores, static_cast<std::size_t>(total)};
    }
}

}  
