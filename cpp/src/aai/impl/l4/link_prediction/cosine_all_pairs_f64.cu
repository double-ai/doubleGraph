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
    int32_t* d_all_vertices = nullptr;
    int32_t alloc_size = 0;

    ~Cache() override {
        if (d_all_vertices) { cudaFree(d_all_vertices); d_all_vertices = nullptr; }
    }
};

static bool pool_configured = false;
static void configure_pool() {
    if (!pool_configured) {
        cudaMemPool_t pool;
        cudaDeviceGetDefaultMemPool(&pool, 0);
        uint64_t threshold = UINT64_MAX;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
        pool_configured = true;
    }
}


__global__ void find_unique_twohop_multiblock(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int32_t blocks_per_seed,
    uint32_t* __restrict__ bitmap,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    int32_t* __restrict__ global_count,
    int32_t bitmap_words_per_seed,
    int32_t max_output
) {
    int sid = blockIdx.x / blocks_per_seed;
    int bid = blockIdx.x % blocks_per_seed;
    if (sid >= num_seeds) return;

    int32_t u = seeds[sid];
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t u_deg = u_end - u_start;

    uint32_t* my_bitmap = bitmap + (int64_t)sid * bitmap_words_per_seed;

    int32_t chunk = (u_deg + blocks_per_seed - 1) / blocks_per_seed;
    int32_t start_ni = bid * chunk;
    int32_t end_ni = start_ni + chunk;
    if (end_ni > u_deg) end_ni = u_deg;

    for (int32_t ni = start_ni; ni < end_ni; ni++) {
        int32_t k = indices[u_start + ni];
        int32_t k_start = offsets[k];
        int32_t k_end = offsets[k + 1];

        for (int32_t vi = threadIdx.x; vi < k_end - k_start; vi += blockDim.x) {
            int32_t v = indices[k_start + vi];
            if (v == u) continue;

            uint32_t word_idx = (uint32_t)v >> 5;
            uint32_t bit_mask = 1u << (v & 31);
            uint32_t old = atomicOr(my_bitmap + word_idx, bit_mask);
            if (!(old & bit_mask)) {
                int32_t pos = atomicAdd(global_count, 1);
                if (pos < max_output) {
                    out_first[pos] = u;
                    out_second[pos] = v;
                }
            }
        }
    }
}


__global__ void count_unique_twohop_multiblock(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int32_t blocks_per_seed,
    uint32_t* __restrict__ bitmap,
    int32_t* __restrict__ global_count,
    int32_t bitmap_words_per_seed
) {
    int sid = blockIdx.x / blocks_per_seed;
    int bid = blockIdx.x % blocks_per_seed;
    if (sid >= num_seeds) return;

    int32_t u = seeds[sid];
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t u_deg = u_end - u_start;

    uint32_t* my_bitmap = bitmap + (int64_t)sid * bitmap_words_per_seed;

    int32_t chunk = (u_deg + blocks_per_seed - 1) / blocks_per_seed;
    int32_t start_ni = bid * chunk;
    int32_t end_ni = start_ni + chunk;
    if (end_ni > u_deg) end_ni = u_deg;

    for (int32_t ni = start_ni; ni < end_ni; ni++) {
        int32_t k = indices[u_start + ni];
        int32_t k_start = offsets[k];
        int32_t k_end = offsets[k + 1];

        for (int32_t vi = threadIdx.x; vi < k_end - k_start; vi += blockDim.x) {
            int32_t v = indices[k_start + vi];
            if (v == u) continue;

            uint32_t word_idx = (uint32_t)v >> 5;
            uint32_t bit_mask = 1u << (v & 31);
            uint32_t old = atomicOr(my_bitmap + word_idx, bit_mask);
            if (!(old & bit_mask)) atomicAdd(global_count, 1);
        }
    }
}


__global__ void compute_cosine_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ pair_first,
    const int32_t* __restrict__ pair_second,
    double* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t warp_id = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (warp_id >= num_pairs) return;

    int32_t u = pair_first[warp_id];
    int32_t v = pair_second[warp_id];
    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int32_t v_start = offsets[v], v_end = offsets[v + 1];
    int32_t u_len = u_end - u_start, v_len = v_end - v_start;

    const int32_t* iter_base; const double* iter_wt; int32_t iter_len;
    const int32_t* search_base; const double* search_wt; int32_t search_len;
    bool swapped;

    if (u_len <= v_len) {
        iter_base = indices + u_start; iter_wt = weights + u_start; iter_len = u_len;
        search_base = indices + v_start; search_wt = weights + v_start; search_len = v_len;
        swapped = false;
    } else {
        iter_base = indices + v_start; iter_wt = weights + v_start; iter_len = v_len;
        search_base = indices + u_start; search_wt = weights + u_start; search_len = u_len;
        swapped = true;
    }

    double dot = 0.0, nu = 0.0, nv = 0.0;

    for (int32_t i = lane; i < iter_len; i += 32) {
        int32_t target = iter_base[i];
        double w_iter = iter_wt[i];

        int32_t lo = 0, hi = search_len;
        while (lo < hi) {
            int32_t mid = (lo + hi) >> 1;
            if (search_base[mid] < target) lo = mid + 1;
            else hi = mid;
        }

        if (lo < search_len && search_base[lo] == target) {
            double w_search = search_wt[lo];
            double wu = swapped ? w_search : w_iter;
            double wv = swapped ? w_iter : w_search;
            dot += wu * wv;
            nu += wu * wu;
            nv += wv * wv;
        }
    }

    #pragma unroll
    for (int o = 16; o >= 1; o >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, o);
        nu += __shfl_down_sync(0xffffffff, nu, o);
        nv += __shfl_down_sync(0xffffffff, nv, o);
    }

    if (lane == 0) {
        double d = sqrt(nu) * sqrt(nv);
        scores[warp_id] = (d > 0.0) ? dot / d : 0.0;
    }
}


__global__ void gather_pairs_kernel(
    const int32_t* __restrict__ sorted_idx,
    const int32_t* __restrict__ in_first,
    const int32_t* __restrict__ in_second,
    const double* __restrict__ in_scores,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    double* __restrict__ out_scores,
    int64_t count
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int32_t idx = sorted_idx[i];
    out_first[i] = in_first[idx];
    out_second[i] = in_second[idx];
    out_scores[i] = in_scores[idx];
}

__global__ void fill_sequence_kernel(int32_t* arr, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = idx;
}

}  

similarity_result_double_t cosine_all_pairs_similarity(
    const graph32_t& graph,
    const double* edge_weights,
    const int32_t* vertices,
    std::size_t num_vertices_param,
    std::optional<std::size_t> topk) {

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    configure_pool();
    cudaStream_t stream = 0;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_verts = graph.number_of_vertices;

    
    const int32_t* seeds;
    int32_t num_seeds;
    if (vertices != nullptr) {
        seeds = vertices;
        num_seeds = (int32_t)num_vertices_param;
    } else {
        num_seeds = num_verts;
        if (cache.alloc_size < num_verts) {
            if (cache.d_all_vertices) cudaFree(cache.d_all_vertices);
            cudaMalloc(&cache.d_all_vertices, num_verts * sizeof(int32_t));
            cache.alloc_size = num_verts;
        }
        fill_sequence_kernel<<<(num_verts + 255) / 256, 256>>>(cache.d_all_vertices, num_verts);
        seeds = cache.d_all_vertices;
    }

    int64_t topk_val = topk.has_value() ? (int64_t)topk.value() : -1;

    if (num_seeds == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    int32_t bitmap_words = (num_verts + 31) / 32;

    int32_t batch_size = num_seeds;
    int64_t bm_bytes = (int64_t)num_seeds * bitmap_words * sizeof(uint32_t);
    if (bm_bytes > 2048LL * 1024 * 1024) {
        batch_size = (int32_t)(2048LL * 1024 * 1024 / ((int64_t)bitmap_words * sizeof(uint32_t)));
        if (batch_size < 1) batch_size = 1;
    }
    int64_t actual_bm = (int64_t)batch_size * bitmap_words * sizeof(uint32_t);

    int32_t blocks_per_seed = 8;
    int32_t initial_buf = 16 * 1024 * 1024; 

    uint32_t* d_bitmap;
    int32_t* d_count;
    int32_t* d_first;
    int32_t* d_second;

    cudaMallocAsync(&d_bitmap, actual_bm, stream);
    cudaMallocAsync(&d_count, sizeof(int32_t), stream);
    cudaMallocAsync(&d_first, (int64_t)initial_buf * sizeof(int32_t), stream);
    cudaMallocAsync(&d_second, (int64_t)initial_buf * sizeof(int32_t), stream);
    cudaMemsetAsync(d_count, 0, sizeof(int32_t), stream);

    for (int32_t start = 0; start < num_seeds; start += batch_size) {
        int32_t batch = num_seeds - start;
        if (batch > batch_size) batch = batch_size;
        cudaMemsetAsync(d_bitmap, 0, (int64_t)batch * bitmap_words * sizeof(uint32_t), stream);
        find_unique_twohop_multiblock<<<batch * blocks_per_seed, 256, 0, stream>>>(
            offsets, indices, seeds + start, batch, blocks_per_seed,
            d_bitmap, d_first, d_second, d_count, bitmap_words, initial_buf);
    }

    int32_t num_unique;
    cudaMemcpyAsync(&num_unique, d_count, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (num_unique == 0) {
        cudaFreeAsync(d_bitmap, stream); cudaFreeAsync(d_count, stream);
        cudaFreeAsync(d_first, stream); cudaFreeAsync(d_second, stream);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    if (num_unique > initial_buf) {
        cudaFreeAsync(d_first, stream);
        cudaFreeAsync(d_second, stream);
        cudaMallocAsync(&d_first, (int64_t)num_unique * sizeof(int32_t), stream);
        cudaMallocAsync(&d_second, (int64_t)num_unique * sizeof(int32_t), stream);
        cudaMemsetAsync(d_count, 0, sizeof(int32_t), stream);

        for (int32_t start = 0; start < num_seeds; start += batch_size) {
            int32_t batch = num_seeds - start;
            if (batch > batch_size) batch = batch_size;
            cudaMemsetAsync(d_bitmap, 0, (int64_t)batch * bitmap_words * sizeof(uint32_t), stream);
            find_unique_twohop_multiblock<<<batch * blocks_per_seed, 256, 0, stream>>>(
                offsets, indices, seeds + start, batch, blocks_per_seed,
                d_bitmap, d_first, d_second, d_count, bitmap_words, num_unique);
        }
    }

    cudaFreeAsync(d_bitmap, stream);
    cudaFreeAsync(d_count, stream);

    
    double* d_scores;
    cudaMallocAsync(&d_scores, (int64_t)num_unique * sizeof(double), stream);
    {
        int wpb = 8;
        int grid = (num_unique + wpb - 1) / wpb;
        compute_cosine_kernel<<<grid, wpb * 32, 0, stream>>>(
            offsets, indices, edge_weights, d_first, d_second, d_scores, num_unique);
    }

    
    int64_t final_count;
    if (topk_val >= 0 && topk_val < num_unique) {
        int32_t* d_idx_in;
        int32_t* d_idx_out;
        double* d_scores_out;
        cudaMallocAsync(&d_idx_in, (int64_t)num_unique * sizeof(int32_t), stream);
        cudaMallocAsync(&d_idx_out, (int64_t)num_unique * sizeof(int32_t), stream);
        cudaMallocAsync(&d_scores_out, (int64_t)num_unique * sizeof(double), stream);

        fill_sequence_kernel<<<(num_unique+255)/256, 256, 0, stream>>>(d_idx_in, num_unique);

        
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            d_temp, temp_bytes,
            d_scores, d_scores_out,
            d_idx_in, d_idx_out,
            num_unique, 0, 64, stream);
        cudaMallocAsync(&d_temp, temp_bytes, stream);
        cub::DeviceRadixSort::SortPairsDescending(
            d_temp, temp_bytes,
            d_scores, d_scores_out,
            d_idx_in, d_idx_out,
            num_unique, 0, 64, stream);
        cudaFreeAsync(d_temp, stream);
        cudaFreeAsync(d_idx_in, stream);

        
        int32_t* d_out_first;
        int32_t* d_out_second;
        double* d_out_scores;
        cudaMallocAsync(&d_out_first, topk_val * sizeof(int32_t), stream);
        cudaMallocAsync(&d_out_second, topk_val * sizeof(int32_t), stream);
        cudaMallocAsync(&d_out_scores, topk_val * sizeof(double), stream);

        {
            int grid = (int)((topk_val + 255) / 256);
            gather_pairs_kernel<<<grid, 256, 0, stream>>>(
                d_idx_out, d_first, d_second, d_scores,
                d_out_first, d_out_second, d_out_scores, topk_val);
        }

        
        cudaMemcpyAsync(d_out_scores, d_scores_out, topk_val * sizeof(double), cudaMemcpyDeviceToDevice, stream);

        cudaFreeAsync(d_idx_out, stream);
        cudaFreeAsync(d_scores_out, stream);
        cudaFreeAsync(d_first, stream);
        cudaFreeAsync(d_second, stream);
        cudaFreeAsync(d_scores, stream);

        d_first = d_out_first;
        d_second = d_out_second;
        d_scores = d_out_scores;
        final_count = topk_val;
    } else {
        final_count = num_unique;
    }

    cudaStreamSynchronize(stream);

    return {d_first, d_second, d_scores, (std::size_t)final_count};
}

}  
