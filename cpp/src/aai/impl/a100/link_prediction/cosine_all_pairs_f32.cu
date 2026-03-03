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
    unsigned long long* counter = nullptr;

    uint32_t* global_bitmaps = nullptr;
    int64_t bitmaps_capacity = 0;

    int32_t* temp_first = nullptr;
    int64_t temp_first_capacity = 0;

    int32_t* temp_second = nullptr;
    int64_t temp_second_capacity = 0;

    float* temp_scores = nullptr;
    int64_t temp_scores_capacity = 0;

    int64_t* idx_in = nullptr;
    int64_t idx_in_capacity = 0;

    void* sort_temp = nullptr;
    size_t sort_temp_capacity = 0;

    float* scores_sorted = nullptr;
    int64_t scores_sorted_capacity = 0;

    int64_t* idx_sorted = nullptr;
    int64_t idx_sorted_capacity = 0;

    Cache() {
        cudaMalloc(&counter, sizeof(unsigned long long));
    }

    ~Cache() override {
        if (counter) cudaFree(counter);
        if (global_bitmaps) cudaFree(global_bitmaps);
        if (temp_first) cudaFree(temp_first);
        if (temp_second) cudaFree(temp_second);
        if (temp_scores) cudaFree(temp_scores);
        if (idx_in) cudaFree(idx_in);
        if (sort_temp) cudaFree(sort_temp);
        if (scores_sorted) cudaFree(scores_sorted);
        if (idx_sorted) cudaFree(idx_sorted);
    }

    void ensure_bitmaps(int64_t n) {
        if (bitmaps_capacity < n) {
            if (global_bitmaps) cudaFree(global_bitmaps);
            cudaMalloc(&global_bitmaps, n * sizeof(uint32_t));
            bitmaps_capacity = n;
        }
    }

    void ensure_temp_first(int64_t n) {
        if (temp_first_capacity < n) {
            if (temp_first) cudaFree(temp_first);
            cudaMalloc(&temp_first, n * sizeof(int32_t));
            temp_first_capacity = n;
        }
    }

    void ensure_temp_second(int64_t n) {
        if (temp_second_capacity < n) {
            if (temp_second) cudaFree(temp_second);
            cudaMalloc(&temp_second, n * sizeof(int32_t));
            temp_second_capacity = n;
        }
    }

    void ensure_temp_scores(int64_t n) {
        if (temp_scores_capacity < n) {
            if (temp_scores) cudaFree(temp_scores);
            cudaMalloc(&temp_scores, n * sizeof(float));
            temp_scores_capacity = n;
        }
    }

    void ensure_idx_in(int64_t n) {
        if (idx_in_capacity < n) {
            if (idx_in) cudaFree(idx_in);
            cudaMalloc(&idx_in, n * sizeof(int64_t));
            idx_in_capacity = n;
        }
    }

    void ensure_sort_temp(size_t n) {
        if (sort_temp_capacity < n) {
            if (sort_temp) cudaFree(sort_temp);
            cudaMalloc(&sort_temp, n);
            sort_temp_capacity = n;
        }
    }

    void ensure_scores_sorted(int64_t n) {
        if (scores_sorted_capacity < n) {
            if (scores_sorted) cudaFree(scores_sorted);
            cudaMalloc(&scores_sorted, n * sizeof(float));
            scores_sorted_capacity = n;
        }
    }

    void ensure_idx_sorted(int64_t n) {
        if (idx_sorted_capacity < n) {
            if (idx_sorted) cudaFree(idx_sorted);
            cudaMalloc(&idx_sorted, n * sizeof(int64_t));
            idx_sorted_capacity = n;
        }
    }
};




__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}





__global__ void count_and_generate_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int32_t num_vertices,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    unsigned long long* __restrict__ global_counter,  
    int64_t max_output,
    uint32_t* __restrict__ global_bitmaps,
    int32_t bitmap_words
) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    extern __shared__ char smem_raw[];
    uint32_t* bitmap;
    int smem_off = 0;

    if (global_bitmaps) {
        bitmap = global_bitmaps + (int64_t)seed_idx * bitmap_words;
    } else {
        bitmap = (uint32_t*)smem_raw;
        smem_off = bitmap_words * sizeof(uint32_t);
    }

    
    for (int i = threadIdx.x; i < bitmap_words; i += blockDim.x)
        bitmap[i] = 0;
    __syncthreads();

    int32_t u = seeds ? seeds[seed_idx] : seed_idx;
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t u_deg = u_end - u_start;

    
    for (int i = threadIdx.x; i < u_deg; i += blockDim.x) {
        int32_t k = indices[u_start + i];
        int32_t k_start = offsets[k];
        int32_t k_end = offsets[k + 1];
        for (int j = k_start; j < k_end; j++) {
            int32_t v = indices[j];
            if (v != u)
                atomicOr(&bitmap[v >> 5], 1u << (v & 31));
        }
    }
    __syncthreads();

    
    int local_count = 0;
    for (int i = threadIdx.x; i < bitmap_words; i += blockDim.x)
        local_count += __popc(bitmap[i]);

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);

    
    __shared__ int warp_counts[8];  
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (lane == 0) warp_counts[warp_id] = local_count;
    __syncthreads();

    int block_total = 0;
    if (threadIdx.x == 0) {
        int nwarps = blockDim.x / 32;
        for (int w = 0; w < nwarps; w++)
            block_total += warp_counts[w];
    }

    
    __shared__ int64_t block_out_base;
    __shared__ int block_total_shared;
    if (threadIdx.x == 0) {
        block_total_shared = block_total;
        block_out_base = (int64_t)atomicAdd(global_counter, (unsigned long long)block_total);
    }
    __syncthreads();

    if (block_out_base + block_total_shared > max_output) return;

    
    __shared__ int32_t local_write_idx;
    if (threadIdx.x == 0) local_write_idx = 0;
    __syncthreads();

    for (int word_idx = threadIdx.x; word_idx < bitmap_words; word_idx += blockDim.x) {
        uint32_t word = bitmap[word_idx];
        while (word != 0) {
            int bit = __ffs(word) - 1;
            int32_t v = word_idx * 32 + bit;
            word &= word - 1;
            if (v < num_vertices) {
                int pos = atomicAdd(&local_write_idx, 1);
                int64_t gpos = block_out_base + pos;
                out_first[gpos] = u;
                out_second[gpos] = v;
            }
        }
    }
}




__global__ void compute_cosine_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ pair_first,
    const int32_t* __restrict__ pair_second,
    float* __restrict__ pair_scores,
    int64_t num_pairs
) {
    int64_t pair_idx = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (pair_idx >= num_pairs) return;

    int lane = threadIdx.x & 31;
    int32_t u = pair_first[pair_idx];
    int32_t v = pair_second[pair_idx];

    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int32_t v_start = offsets[v], v_end = offsets[v + 1];
    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    
    bool iter_u = (u_deg <= v_deg);
    int32_t a_off = iter_u ? u_start : v_start;
    int32_t a_len = iter_u ? u_deg : v_deg;
    int32_t b_off = iter_u ? v_start : u_start;
    int32_t b_len = iter_u ? v_deg : u_deg;

    float dot = 0.0f, nu = 0.0f, nv = 0.0f;

    for (int i = lane; i < a_len; i += 32) {
        int32_t elem = __ldg(&indices[a_off + i]);
        float wa = __ldg(&weights[a_off + i]);

        
        int lo = 0, hi = b_len;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (__ldg(&indices[b_off + mid]) < elem) lo = mid + 1;
            else hi = mid;
        }

        if (lo < b_len && __ldg(&indices[b_off + lo]) == elem) {
            float wb = __ldg(&weights[b_off + lo]);
            float wu = iter_u ? wa : wb;
            float wv = iter_u ? wb : wa;
            dot += wu * wv;
            nu += wu * wu;
            nv += wv * wv;
        }
    }

    dot = warp_sum(dot);
    nu = warp_sum(nu);
    nv = warp_sum(nv);

    if (lane == 0) {
        float denom = sqrtf(nu) * sqrtf(nv);
        pair_scores[pair_idx] = (denom > 0.0f) ? (dot / denom) : 0.0f;
    }
}




__global__ void iota_kernel(int64_t* arr, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = idx;
}

__global__ void gather_kernel(
    const int32_t* __restrict__ sf, const int32_t* __restrict__ ss,
    const float* __restrict__ ssc, const int64_t* __restrict__ idx,
    int32_t* __restrict__ df, int32_t* __restrict__ ds,
    float* __restrict__ dsc, int64_t count
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        int64_t si = idx[i];
        df[i] = sf[si]; ds[i] = ss[si]; dsc[i] = ssc[si];
    }
}

}  

similarity_result_float_t cosine_all_pairs_similarity(const graph32_t& graph,
                                                      const float* edge_weights,
                                                      const int32_t* vertices,
                                                      std::size_t num_vertices,
                                                      std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const float* d_weights = edge_weights;
    int32_t graph_num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    const int32_t* d_seeds = nullptr;
    int32_t num_seeds;
    if (vertices != nullptr && num_vertices > 0) {
        d_seeds = vertices;
        num_seeds = (int32_t)num_vertices;
    } else {
        num_seeds = graph_num_vertices;
    }

    bool has_topk = topk.has_value();
    size_t topk_val = has_topk ? topk.value() : 0;
    cudaStream_t stream = 0;

    
    int32_t bitmap_words = (graph_num_vertices + 31) / 32;
    size_t bitmap_bytes = (size_t)bitmap_words * sizeof(uint32_t);
    bool use_smem = (bitmap_bytes <= 160 * 1024);

    uint32_t* d_global_bitmaps = nullptr;
    if (!use_smem) {
        cache.ensure_bitmaps((int64_t)num_seeds * bitmap_words);
        d_global_bitmaps = cache.global_bitmaps;
    }

    
    int64_t estimated_pairs;
    if (vertices != nullptr && num_vertices > 0) {
        int64_t est1 = (int64_t)num_seeds * 50000LL;
        int64_t est2 = (int64_t)num_edges * 2;
        estimated_pairs = (est1 < est2) ? est1 : est2;
    } else {
        int64_t avg_deg = (int64_t)num_edges / ((int64_t)graph_num_vertices > 0 ? (int64_t)graph_num_vertices : 1);
        int64_t est1 = (int64_t)graph_num_vertices * avg_deg * avg_deg * 2;
        int64_t est2 = (int64_t)graph_num_vertices * (int64_t)graph_num_vertices;
        estimated_pairs = (est1 < est2) ? est1 : est2;
        if (estimated_pairs < (int64_t)num_edges) estimated_pairs = (int64_t)num_edges;
    }
    if (estimated_pairs > 100000000LL) estimated_pairs = 100000000LL;

    
    cache.ensure_temp_first(estimated_pairs);
    cache.ensure_temp_second(estimated_pairs);

    
    cudaMemsetAsync(cache.counter, 0, sizeof(unsigned long long), stream);

    
    {
        int bs = 256;
        size_t smem = use_smem ? ((size_t)bitmap_words * sizeof(uint32_t)) : 0;
        smem += sizeof(int64_t) + sizeof(int32_t);

        if (smem > 48 * 1024)
            cudaFuncSetAttribute(count_and_generate_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

        count_and_generate_kernel<<<num_seeds, bs, smem, stream>>>(
            d_offsets, d_indices, d_seeds, num_seeds, graph_num_vertices,
            cache.temp_first, cache.temp_second, cache.counter, estimated_pairs,
            use_smem ? nullptr : d_global_bitmaps, bitmap_words
        );
    }

    
    int64_t total_pairs;
    cudaMemcpyAsync(&total_pairs, cache.counter, sizeof(int64_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (total_pairs == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    if (total_pairs > estimated_pairs) {
        cache.ensure_temp_first(total_pairs);
        cache.ensure_temp_second(total_pairs);
        cudaMemsetAsync(cache.counter, 0, sizeof(unsigned long long), stream);

        int bs = 256;
        size_t smem = use_smem ? ((size_t)bitmap_words * sizeof(uint32_t)) : 0;
        smem += sizeof(int64_t) + sizeof(int32_t);

        if (smem > 48 * 1024)
            cudaFuncSetAttribute(count_and_generate_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

        count_and_generate_kernel<<<num_seeds, bs, smem, stream>>>(
            d_offsets, d_indices, d_seeds, num_seeds, graph_num_vertices,
            cache.temp_first, cache.temp_second, cache.counter, total_pairs,
            use_smem ? nullptr : d_global_bitmaps, bitmap_words
        );
    }

    
    cache.ensure_temp_scores(total_pairs);
    {
        int bs = 256;
        int64_t num_warps = total_pairs;
        int64_t num_threads = num_warps * 32;
        int grid = (int)((num_threads + bs - 1) / bs);
        if (grid > 0) {
            compute_cosine_pairs_kernel<<<grid, bs, 0, stream>>>(
                d_offsets, d_indices, d_weights,
                cache.temp_first, cache.temp_second,
                cache.temp_scores, total_pairs
            );
        }
    }

    
    if (has_topk && total_pairs > (int64_t)topk_val) {
        cache.ensure_idx_in(total_pairs);
        {
            int b = 256, g = ((int)total_pairs + b - 1) / b;
            if (g > 0) iota_kernel<<<g, b, 0, stream>>>(cache.idx_in, total_pairs);
        }

        size_t sort_ts = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, sort_ts, (float*)nullptr, (float*)nullptr,
            (int64_t*)nullptr, (int64_t*)nullptr, (int)total_pairs
        );
        cache.ensure_sort_temp(sort_ts);
        cache.ensure_scores_sorted(total_pairs);
        cache.ensure_idx_sorted(total_pairs);

        size_t sort_temp_size = sort_ts;
        cub::DeviceRadixSort::SortPairsDescending(
            cache.sort_temp, sort_temp_size,
            cache.temp_scores, cache.scores_sorted,
            cache.idx_in, cache.idx_sorted,
            (int)total_pairs, 0, 32, stream
        );

        int64_t out_count = (int64_t)topk_val;

        
        int32_t* out_first = nullptr;
        int32_t* out_second = nullptr;
        float* out_scores = nullptr;
        cudaMalloc(&out_first, out_count * sizeof(int32_t));
        cudaMalloc(&out_second, out_count * sizeof(int32_t));
        cudaMalloc(&out_scores, out_count * sizeof(float));

        {
            int b = 256, g = ((int)out_count + b - 1) / b;
            if (g > 0) gather_kernel<<<g, b, 0, stream>>>(
                cache.temp_first, cache.temp_second, cache.temp_scores,
                cache.idx_sorted, out_first, out_second, out_scores, out_count
            );
        }

        return {out_first, out_second, out_scores, (std::size_t)out_count};
    }

    
    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;
    cudaMalloc(&out_first, total_pairs * sizeof(int32_t));
    cudaMalloc(&out_second, total_pairs * sizeof(int32_t));
    cudaMalloc(&out_scores, total_pairs * sizeof(float));

    cudaMemcpyAsync(out_first, cache.temp_first,
                    total_pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(out_second, cache.temp_second,
                    total_pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(out_scores, cache.temp_scores,
                    total_pairs * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    return {out_first, out_second, out_scores, (std::size_t)total_pairs};
}

}  
