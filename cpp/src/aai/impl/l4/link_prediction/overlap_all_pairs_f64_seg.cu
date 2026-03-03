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
    
    double* degrees = nullptr;
    int64_t degrees_capacity = 0;

    int64_t* counts = nullptr;
    int64_t counts_capacity = 0;

    int64_t* cand_offsets = nullptr;
    int64_t cand_offsets_capacity = 0;

    uint8_t* temp_storage = nullptr;
    int64_t temp_storage_capacity = 0;

    int32_t* pair_first = nullptr;
    int64_t pair_first_capacity = 0;

    int32_t* pair_second = nullptr;
    int64_t pair_second_capacity = 0;

    int32_t* pair_count_dev = nullptr;  

    int32_t* bitmaps = nullptr;
    int64_t bitmaps_capacity = 0;

    double* scores = nullptr;
    int64_t scores_capacity = 0;

    int64_t* score_keys = nullptr;
    int64_t score_keys_capacity = 0;

    int32_t* indices_arr = nullptr;
    int64_t indices_arr_capacity = 0;

    int64_t* sorted_score_keys = nullptr;
    int64_t sorted_score_keys_capacity = 0;

    int32_t* sorted_indices = nullptr;
    int64_t sorted_indices_capacity = 0;

    int32_t* seeds_storage = nullptr;
    int64_t seeds_storage_capacity = 0;

    Cache() {
        cudaMalloc(&pair_count_dev, sizeof(int32_t));
    }

    ~Cache() override {
        if (degrees) cudaFree(degrees);
        if (counts) cudaFree(counts);
        if (cand_offsets) cudaFree(cand_offsets);
        if (temp_storage) cudaFree(temp_storage);
        if (pair_first) cudaFree(pair_first);
        if (pair_second) cudaFree(pair_second);
        if (pair_count_dev) cudaFree(pair_count_dev);
        if (bitmaps) cudaFree(bitmaps);
        if (scores) cudaFree(scores);
        if (score_keys) cudaFree(score_keys);
        if (indices_arr) cudaFree(indices_arr);
        if (sorted_score_keys) cudaFree(sorted_score_keys);
        if (sorted_indices) cudaFree(sorted_indices);
        if (seeds_storage) cudaFree(seeds_storage);
    }

    void ensure_degrees(int64_t n) {
        if (degrees_capacity < n) {
            if (degrees) cudaFree(degrees);
            cudaMalloc(&degrees, n * sizeof(double));
            degrees_capacity = n;
        }
    }

    void ensure_counts(int64_t n) {
        if (counts_capacity < n) {
            if (counts) cudaFree(counts);
            cudaMalloc(&counts, n * sizeof(int64_t));
            counts_capacity = n;
        }
    }

    void ensure_cand_offsets(int64_t n) {
        if (cand_offsets_capacity < n) {
            if (cand_offsets) cudaFree(cand_offsets);
            cudaMalloc(&cand_offsets, n * sizeof(int64_t));
            cand_offsets_capacity = n;
        }
    }

    void ensure_temp_storage(int64_t n) {
        if (temp_storage_capacity < n) {
            if (temp_storage) cudaFree(temp_storage);
            cudaMalloc(&temp_storage, n);
            temp_storage_capacity = n;
        }
    }

    void ensure_pair_first(int64_t n) {
        if (pair_first_capacity < n) {
            if (pair_first) cudaFree(pair_first);
            cudaMalloc(&pair_first, n * sizeof(int32_t));
            pair_first_capacity = n;
        }
    }

    void ensure_pair_second(int64_t n) {
        if (pair_second_capacity < n) {
            if (pair_second) cudaFree(pair_second);
            cudaMalloc(&pair_second, n * sizeof(int32_t));
            pair_second_capacity = n;
        }
    }

    void ensure_bitmaps(int64_t n) {
        if (bitmaps_capacity < n) {
            if (bitmaps) cudaFree(bitmaps);
            cudaMalloc(&bitmaps, n * sizeof(int32_t));
            bitmaps_capacity = n;
        }
    }

    void ensure_scores(int64_t n) {
        if (scores_capacity < n) {
            if (scores) cudaFree(scores);
            cudaMalloc(&scores, n * sizeof(double));
            scores_capacity = n;
        }
    }

    void ensure_score_keys(int64_t n) {
        if (score_keys_capacity < n) {
            if (score_keys) cudaFree(score_keys);
            cudaMalloc(&score_keys, n * sizeof(int64_t));
            score_keys_capacity = n;
        }
    }

    void ensure_indices_arr(int64_t n) {
        if (indices_arr_capacity < n) {
            if (indices_arr) cudaFree(indices_arr);
            cudaMalloc(&indices_arr, n * sizeof(int32_t));
            indices_arr_capacity = n;
        }
    }

    void ensure_sorted_score_keys(int64_t n) {
        if (sorted_score_keys_capacity < n) {
            if (sorted_score_keys) cudaFree(sorted_score_keys);
            cudaMalloc(&sorted_score_keys, n * sizeof(int64_t));
            sorted_score_keys_capacity = n;
        }
    }

    void ensure_sorted_indices(int64_t n) {
        if (sorted_indices_capacity < n) {
            if (sorted_indices) cudaFree(sorted_indices);
            cudaMalloc(&sorted_indices, n * sizeof(int32_t));
            sorted_indices_capacity = n;
        }
    }

    void ensure_seeds_storage(int64_t n) {
        if (seeds_storage_capacity < n) {
            if (seeds_storage) cudaFree(seeds_storage);
            cudaMalloc(&seeds_storage, n * sizeof(int32_t));
            seeds_storage_capacity = n;
        }
    }
};


__global__ void compute_weighted_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ weights,
    double* __restrict__ degrees,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int start = offsets[v];
    int end = offsets[v + 1];
    double sum = 0.0;
    for (int i = start; i < end; i++) sum += weights[i];
    degrees[v] = sum;
}


__global__ void count_candidates_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    int64_t* __restrict__ counts)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_seeds) return;
    int u = seeds[sid];
    int64_t count = 0;
    for (int i = offsets[u]; i < offsets[u + 1]; i++) {
        int n = indices[i];
        count += (int64_t)(offsets[n + 1] - offsets[n]);
    }
    counts[sid] = count;
}


__global__ void flat_enumerate_dedup_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    const int64_t* __restrict__ cand_offsets,
    uint32_t* __restrict__ bitmaps,
    int bitmap_words_per_seed,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    int32_t* __restrict__ pair_count,
    int64_t total_candidates)
{
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_candidates) return;

    
    int lo = 0, hi = num_seeds - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;
        if (cand_offsets[mid] <= gid) lo = mid;
        else hi = mid - 1;
    }
    int sid = lo;

    int u = seeds[sid];
    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int64_t local_idx = gid - cand_offsets[sid];

    
    int64_t running = 0;
    int v = -1;
    for (int i = u_start; i < u_end; i++) {
        int n = indices[i];
        int n_deg = offsets[n + 1] - offsets[n];
        if (running + n_deg > local_idx) {
            int j = (int)(local_idx - running);
            v = indices[offsets[n] + j];
            break;
        }
        running += n_deg;
    }

    if (v < 0 || v == u) return;

    
    uint32_t* my_bitmap = bitmaps + (int64_t)sid * bitmap_words_per_seed;
    uint32_t bit = 1u << (v & 31);
    uint32_t old = atomicOr(&my_bitmap[v >> 5], bit);
    if (!(old & bit)) {
        int pos = atomicAdd(pair_count, 1);
        out_first[pos] = u;
        out_second[pos] = v;
    }
}


__global__ void block_enumerate_dedup_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    int32_t num_vertices,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    int32_t* __restrict__ pair_count)
{
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int u = seeds[sid];
    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int u_deg = u_end - u_start;
    if (u_deg == 0) return;

    extern __shared__ uint32_t bitmap[];
    int bitmap_words = (num_vertices + 31) / 32;

    for (int i = threadIdx.x; i < bitmap_words; i += blockDim.x)
        bitmap[i] = 0;
    __syncthreads();

    
    if (threadIdx.x == 0)
        atomicOr(&bitmap[u >> 5], 1u << (u & 31));
    __syncthreads();

    for (int i = 0; i < u_deg; i++) {
        int n = indices[u_start + i];
        int n_start = offsets[n];
        int n_end = offsets[n + 1];
        int n_deg = n_end - n_start;

        for (int j = threadIdx.x; j < n_deg; j += blockDim.x) {
            int v = indices[n_start + j];
            uint32_t bit = 1u << (v & 31);
            uint32_t old = atomicOr(&bitmap[v >> 5], bit);
            if (!(old & bit)) {
                int pos = atomicAdd(pair_count, 1);
                out_first[pos] = u;
                out_second[pos] = v;
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
    double* __restrict__ pair_scores,
    int64_t num_pairs)
{
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_id >= num_pairs) return;

    int u = pair_first[warp_id];
    int v = pair_second[warp_id];
    int u_start = offsets[u], u_end = offsets[u + 1];
    int v_start = offsets[v], v_end = offsets[v + 1];
    int u_deg = u_end - u_start;
    int v_deg = v_end - v_start;

    const int32_t* short_list, *long_list;
    const double* short_w, *long_w;
    int short_size, long_size;

    if (u_deg <= v_deg) {
        short_list = indices + u_start; long_list = indices + v_start;
        short_w = weights + u_start; long_w = weights + v_start;
        short_size = u_deg; long_size = v_deg;
    } else {
        short_list = indices + v_start; long_list = indices + u_start;
        short_w = weights + v_start; long_w = weights + u_start;
        short_size = v_deg; long_size = u_deg;
    }

    double my_sum = 0.0;
    for (int i = lane; i < short_size; i += 32) {
        int target = short_list[i];
        double w1 = short_w[i];
        int lo = 0, hi = long_size;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (long_list[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        if (lo < long_size && long_list[lo] == target)
            my_sum += fmin(w1, long_w[lo]);
    }

    for (int mask = 16; mask > 0; mask >>= 1)
        my_sum += __shfl_xor_sync(0xffffffff, my_sum, mask);

    if (lane == 0) {
        double w_u = weighted_degrees[u];
        double w_v = weighted_degrees[v];
        double denom = fmin(w_u, w_v);
        pair_scores[warp_id] = (denom > 0.0) ? my_sum / denom : 0.0;
    }
}


__global__ void generate_seeds_kernel(int32_t* seeds, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) seeds[idx] = idx;
}

__global__ void double_to_sortable_kernel(const double* in, uint64_t* out, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint64_t bits;
    memcpy(&bits, &in[idx], sizeof(uint64_t));
    if (bits & 0x8000000000000000ULL) bits = ~bits;
    else bits ^= 0x8000000000000000ULL;
    out[idx] = bits;
}

__global__ void iota_kernel(int32_t* data, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = (int32_t)idx;
}

__global__ void gather_kernel(
    const int32_t* __restrict__ in_first,
    const int32_t* __restrict__ in_second,
    const double* __restrict__ in_scores,
    const int32_t* __restrict__ perm,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    double* __restrict__ out_scores,
    int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int32_t p = perm[idx];
    out_first[idx] = in_first[p];
    out_second[idx] = in_second[p];
    out_scores[idx] = in_scores[p];
}

}  

similarity_result_double_t overlap_all_pairs_similarity_seg(const graph32_t& graph,
                                                            const double* edge_weights,
                                                            const int32_t* vertices,
                                                            std::size_t num_vertices_param,
                                                            std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_verts = graph.number_of_vertices;
    const double* d_weights = edge_weights;

    
    int num_seeds;
    const int32_t* d_seeds;

    if (vertices != nullptr && num_vertices_param > 0) {
        num_seeds = (int)num_vertices_param;
        d_seeds = vertices;
    } else {
        num_seeds = num_verts;
        cache.ensure_seeds_storage(num_verts);
        if (num_verts > 0) {
            generate_seeds_kernel<<<(num_verts + 255) / 256, 256>>>(cache.seeds_storage, num_verts);
        }
        d_seeds = cache.seeds_storage;
    }

    if (num_seeds == 0) {
        return similarity_result_double_t{nullptr, nullptr, nullptr, 0};
    }

    
    cache.ensure_degrees(num_verts);
    if (num_verts > 0) {
        compute_weighted_degrees_kernel<<<(num_verts + 255) / 256, 256>>>(
            d_offsets, d_weights, cache.degrees, num_verts);
    }

    
    cache.ensure_counts(num_seeds + 1);
    cudaMemsetAsync(cache.counts + num_seeds, 0, sizeof(int64_t));
    if (num_seeds > 0) {
        count_candidates_kernel<<<(num_seeds + 255) / 256, 256>>>(
            d_offsets, d_indices, d_seeds, num_seeds, cache.counts);
    }

    cache.ensure_cand_offsets(num_seeds + 1);
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes,
                                  (int64_t*)nullptr, (int64_t*)nullptr, num_seeds + 1);
    cache.ensure_temp_storage((int64_t)(temp_bytes + 16));
    cub::DeviceScan::ExclusiveSum(cache.temp_storage, temp_bytes,
                                  cache.counts, cache.cand_offsets, num_seeds + 1);

    int64_t total_candidates;
    cudaMemcpy(&total_candidates, cache.cand_offsets + num_seeds,
               sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (total_candidates == 0) {
        return similarity_result_double_t{nullptr, nullptr, nullptr, 0};
    }

    
    int64_t buffer_size = total_candidates;
    int64_t max_unique = (int64_t)num_seeds * (int64_t)num_verts;
    if (buffer_size > max_unique) buffer_size = max_unique;

    cache.ensure_pair_first(buffer_size);
    cache.ensure_pair_second(buffer_size);
    cudaMemsetAsync(cache.pair_count_dev, 0, sizeof(int32_t));

    int bitmap_words = (num_verts + 31) / 32;
    int bitmap_bytes = bitmap_words * 4;

    if (num_seeds > 500 && bitmap_bytes <= 99 * 1024) {
        if (bitmap_bytes > 48 * 1024) {
            cudaFuncSetAttribute(block_enumerate_dedup_kernel,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, bitmap_bytes);
        }
        block_enumerate_dedup_kernel<<<num_seeds, 256, bitmap_bytes>>>(
            d_offsets, d_indices, d_seeds, num_seeds, num_verts,
            cache.pair_first, cache.pair_second, cache.pair_count_dev);
    } else {
        int64_t bitmap_total = (int64_t)num_seeds * bitmap_words;
        cache.ensure_bitmaps(bitmap_total);
        cudaMemsetAsync(cache.bitmaps, 0, bitmap_total * sizeof(int32_t));

        if (total_candidates > 0) {
            int block = 256;
            int grid = (int)((total_candidates + block - 1) / block);
            flat_enumerate_dedup_kernel<<<grid, block>>>(
                d_offsets, d_indices, d_seeds, num_seeds,
                cache.cand_offsets,
                (uint32_t*)cache.bitmaps, bitmap_words,
                cache.pair_first, cache.pair_second, cache.pair_count_dev,
                total_candidates);
        }
    }

    int32_t num_pairs_int;
    cudaMemcpy(&num_pairs_int, cache.pair_count_dev, sizeof(int32_t), cudaMemcpyDeviceToHost);
    int64_t num_pairs = num_pairs_int;

    if (num_pairs == 0) {
        return similarity_result_double_t{nullptr, nullptr, nullptr, 0};
    }

    
    cache.ensure_scores(num_pairs);
    {
        int block = 256;
        int grid = (int)((num_pairs * 32 + block - 1) / block);
        compute_overlap_kernel<<<grid, block>>>(
            d_offsets, d_indices, d_weights, cache.degrees,
            cache.pair_first, cache.pair_second, cache.scores, num_pairs);
    }

    
    int64_t output_count = num_pairs;
    int32_t* out_first;
    int32_t* out_second;
    double* out_scores;

    bool do_topk = topk.has_value() && (int64_t)topk.value() < num_pairs;

    if (do_topk) {
        int64_t k = (int64_t)topk.value();
        output_count = k;

        cache.ensure_score_keys(num_pairs);
        double_to_sortable_kernel<<<(int)((num_pairs + 255) / 256), 256>>>(
            cache.scores, (uint64_t*)cache.score_keys, num_pairs);

        cache.ensure_indices_arr(num_pairs);
        iota_kernel<<<(int)((num_pairs + 255) / 256), 256>>>(
            cache.indices_arr, num_pairs);

        cache.ensure_sorted_score_keys(num_pairs);
        cache.ensure_sorted_indices(num_pairs);

        int n = (int)num_pairs;
        temp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_bytes,
            (uint64_t*)nullptr, (uint64_t*)nullptr,
            (int32_t*)nullptr, (int32_t*)nullptr, n);
        cache.ensure_temp_storage((int64_t)(temp_bytes + 16));
        cub::DeviceRadixSort::SortPairsDescending(
            cache.temp_storage, temp_bytes,
            (uint64_t*)cache.score_keys, (uint64_t*)cache.sorted_score_keys,
            cache.indices_arr, cache.sorted_indices, n);

        
        cudaMalloc(&out_first, k * sizeof(int32_t));
        cudaMalloc(&out_second, k * sizeof(int32_t));
        cudaMalloc(&out_scores, k * sizeof(double));

        gather_kernel<<<(int)((k + 255) / 256), 256>>>(
            cache.pair_first, cache.pair_second, cache.scores,
            cache.sorted_indices,
            out_first, out_second, out_scores, k);
    } else {
        
        cudaMalloc(&out_first, num_pairs * sizeof(int32_t));
        cudaMalloc(&out_second, num_pairs * sizeof(int32_t));
        cudaMalloc(&out_scores, num_pairs * sizeof(double));

        cudaMemcpyAsync(out_first, cache.pair_first,
                         num_pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(out_second, cache.pair_second,
                         num_pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(out_scores, cache.scores,
                         num_pairs * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    return similarity_result_double_t{out_first, out_second, out_scores, (std::size_t)output_count};
}

}  
