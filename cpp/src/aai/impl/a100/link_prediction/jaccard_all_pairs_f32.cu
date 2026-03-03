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
#include <vector>

namespace aai {

namespace {





__global__ void count_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ counts)
{
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int32_t u = seeds[sid];
    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int64_t count = 0;
    for (int32_t i = u_start + threadIdx.x; i < u_end; i += blockDim.x) {
        int32_t w = indices[i];
        count += (int64_t)(offsets[w + 1] - offsets[w]);
    }
    typedef cub::BlockReduce<int64_t, 256> BR;
    __shared__ typename BR::TempStorage temp;
    int64_t total = BR(temp).Sum(count);
    if (threadIdx.x == 0) counts[sid] = total;
}

__global__ void write_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ seed_offsets,
    int64_t* __restrict__ out_keys,
    int64_t max_v)
{
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int32_t u = seeds[sid];
    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int64_t base = seed_offsets[sid];

    __shared__ int64_t write_pos;
    if (threadIdx.x == 0) write_pos = 0;
    __syncthreads();

    for (int32_t i = u_start + threadIdx.x; i < u_end; i += blockDim.x) {
        int32_t w = indices[i];
        int32_t w_start = offsets[w], w_end = offsets[w + 1];
        int32_t w_deg = w_end - w_start;
        int64_t my_pos = atomicAdd((unsigned long long*)&write_pos, (unsigned long long)w_deg);
        for (int32_t j = 0; j < w_deg; j++) {
            int32_t v = indices[w_start + j];
            out_keys[base + my_pos + j] = (int64_t)u * max_v + (int64_t)v;
        }
    }
}

__global__ void compute_unique_flags_kernel(
    const int64_t* __restrict__ sorted_keys,
    uint8_t* __restrict__ flags,
    int64_t n,
    int64_t max_v)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t key = sorted_keys[i];
    bool is_unique = (i == 0) || (key != sorted_keys[i - 1]);
    int64_t u_val = key / max_v;
    int64_t v_val = key % max_v;
    bool not_self = (u_val != v_val);
    flags[i] = (is_unique && not_self) ? 1 : 0;
}

__global__ void jaccard_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int64_t* __restrict__ pair_keys,
    int64_t max_v,
    float* __restrict__ scores,
    int64_t num_pairs)
{
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_pairs) return;

    int64_t key = pair_keys[warp_id];
    int32_t u = (int32_t)(key / max_v);
    int32_t v = (int32_t)(key % max_v);

    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int32_t v_start = offsets[v], v_end = offsets[v + 1];
    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    float sum_u = 0.0f;
    for (int32_t i = u_start + lane; i < u_end; i += 32)
        sum_u += weights[i];
    for (int off = 16; off > 0; off >>= 1)
        sum_u += __shfl_down_sync(0xffffffff, sum_u, off);
    sum_u = __shfl_sync(0xffffffff, sum_u, 0);

    float sum_v = 0.0f;
    for (int32_t i = v_start + lane; i < v_end; i += 32)
        sum_v += weights[i];
    for (int off = 16; off > 0; off >>= 1)
        sum_v += __shfl_down_sync(0xffffffff, sum_v, off);
    sum_v = __shfl_sync(0xffffffff, sum_v, 0);

    const int32_t* iter_base;
    const float* iter_wt;
    int32_t iter_len;
    const int32_t* srch_base;
    const float* srch_wt;
    int32_t srch_len;

    if (u_deg <= v_deg) {
        iter_base = indices + u_start; iter_wt = weights + u_start; iter_len = u_deg;
        srch_base = indices + v_start; srch_wt = weights + v_start; srch_len = v_deg;
    } else {
        iter_base = indices + v_start; iter_wt = weights + v_start; iter_len = v_deg;
        srch_base = indices + u_start; srch_wt = weights + u_start; srch_len = u_deg;
    }

    float min_sum = 0.0f;
    for (int32_t i = lane; i < iter_len; i += 32) {
        int32_t target = iter_base[i];
        float w_iter = iter_wt[i];
        int32_t lo = 0, hi = srch_len;
        while (lo < hi) {
            int32_t mid = (lo + hi) >> 1;
            if (srch_base[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        if (lo < srch_len && srch_base[lo] == target) {
            min_sum += fminf(w_iter, srch_wt[lo]);
        }
    }
    for (int off = 16; off > 0; off >>= 1)
        min_sum += __shfl_down_sync(0xffffffff, min_sum, off);

    if (lane == 0) {
        float denom = sum_u + sum_v - min_sum;
        scores[warp_id] = (denom > 0.0f) ? (min_sum / denom) : 0.0f;
    }
}

__global__ void decode_keys_kernel(
    const int64_t* __restrict__ keys, int64_t n, int64_t max_v,
    int32_t* __restrict__ first, int32_t* __restrict__ second)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t key = keys[i];
    first[i] = (int32_t)(key / max_v);
    second[i] = (int32_t)(key % max_v);
}

__global__ void iota_kernel(int32_t* arr, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = (int32_t)i;
}

__global__ void gather_topk_kernel(
    const int64_t* __restrict__ pair_keys,
    const float* __restrict__ sorted_scores,
    const int32_t* __restrict__ perm,
    int64_t max_v,
    int32_t* __restrict__ first_out,
    int32_t* __restrict__ second_out,
    float* __restrict__ scores_out,
    int64_t count)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int32_t p = perm[i];
    int64_t key = pair_keys[p];
    first_out[i] = (int32_t)(key / max_v);
    second_out[i] = (int32_t)(key % max_v);
    scores_out[i] = sorted_scores[i];
}





void launch_count_pairs(const int32_t* offsets, const int32_t* indices,
    const int32_t* seeds, int32_t num_seeds, int64_t* counts, cudaStream_t stream) {
    count_pairs_kernel<<<num_seeds, 256, 0, stream>>>(offsets, indices, seeds, num_seeds, counts);
}

void launch_write_pairs(const int32_t* offsets, const int32_t* indices,
    const int32_t* seeds, int32_t num_seeds, const int64_t* seed_offsets,
    int64_t* out_keys, int64_t max_v, cudaStream_t stream) {
    write_pairs_kernel<<<num_seeds, 256, 0, stream>>>(
        offsets, indices, seeds, num_seeds, seed_offsets, out_keys, max_v);
}

void launch_compute_unique_flags(const int64_t* sorted_keys, uint8_t* flags,
    int64_t n, int64_t max_v, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    compute_unique_flags_kernel<<<grid, block, 0, stream>>>(sorted_keys, flags, n, max_v);
}

void launch_jaccard(const int32_t* offsets, const int32_t* indices,
    const float* weights,
    const int64_t* pair_keys, int64_t max_v,
    float* scores, int64_t num_pairs, cudaStream_t stream) {
    if (num_pairs == 0) return;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int grid = (int)((num_pairs + warps_per_block - 1) / warps_per_block);
    jaccard_kernel<<<grid, threads, 0, stream>>>(
        offsets, indices, weights, pair_keys, max_v, scores, num_pairs);
}

void launch_decode_keys(const int64_t* keys, int64_t n, int64_t max_v,
    int32_t* first, int32_t* second, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    decode_keys_kernel<<<grid, block, 0, stream>>>(keys, n, max_v, first, second);
}

void launch_iota(int32_t* arr, int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    iota_kernel<<<grid, block, 0, stream>>>(arr, n);
}

void launch_gather_topk(const int64_t* pair_keys, const float* sorted_scores,
    const int32_t* perm, int64_t max_v,
    int32_t* first_out, int32_t* second_out, float* scores_out,
    int64_t count, cudaStream_t stream) {
    if (count == 0) return;
    int block = 256;
    int grid = (int)((count + block - 1) / block);
    gather_topk_kernel<<<grid, block, 0, stream>>>(
        pair_keys, sorted_scores, perm, max_v, first_out, second_out, scores_out, count);
}





size_t get_sort_keys_temp_bytes(int64_t num_items, int end_bit) {
    size_t bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, bytes, (int64_t*)nullptr, (int64_t*)nullptr,
        (int)num_items, 0, end_bit);
    return bytes;
}

void launch_sort_keys(int64_t* keys_in, int64_t* keys_out, int64_t num_items,
    void* temp, size_t temp_bytes, int end_bit, cudaStream_t stream) {
    cub::DeviceRadixSort::SortKeys(temp, temp_bytes, keys_in, keys_out,
        (int)num_items, 0, end_bit, stream);
}

size_t get_flagged_temp_bytes(int64_t num_items) {
    size_t bytes = 0;
    cub::DeviceSelect::Flagged(nullptr, bytes, (int64_t*)nullptr, (uint8_t*)nullptr,
        (int64_t*)nullptr, (int64_t*)nullptr, (int)num_items);
    return bytes;
}

void launch_flagged_select(const int64_t* in, const uint8_t* flags, int64_t* out,
    int64_t* num_selected, int64_t num_items,
    void* temp, size_t temp_bytes, cudaStream_t stream) {
    cub::DeviceSelect::Flagged(temp, temp_bytes, in, flags, out, num_selected,
        (int)num_items, stream);
}

size_t get_sort_pairs_desc_temp_bytes(int64_t num_items) {
    size_t bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(nullptr, bytes,
        (float*)nullptr, (float*)nullptr,
        (int32_t*)nullptr, (int32_t*)nullptr, (int)num_items);
    return bytes;
}

void launch_sort_pairs_desc(float* keys_in, float* keys_out,
    int32_t* vals_in, int32_t* vals_out, int64_t num_items,
    void* temp, size_t temp_bytes, cudaStream_t stream) {
    cub::DeviceRadixSort::SortPairsDescending(temp, temp_bytes,
        keys_in, keys_out, vals_in, vals_out, (int)num_items, 0, 32, stream);
}





struct Cache : Cacheable {
    int32_t* seeds_buf = nullptr;
    int64_t seeds_buf_cap = 0;

    int64_t* counts = nullptr;
    int64_t counts_cap = 0;

    int64_t* seed_offsets_d = nullptr;
    int64_t seed_offsets_d_cap = 0;

    int64_t* raw_keys = nullptr;
    int64_t raw_keys_cap = 0;

    int64_t* sorted_keys = nullptr;
    int64_t sorted_keys_cap = 0;

    uint8_t* flags_buf = nullptr;
    int64_t flags_buf_cap = 0;

    int64_t* filtered_keys = nullptr;
    int64_t filtered_keys_cap = 0;

    int64_t* num_filtered_d = nullptr;

    uint8_t* cub_temp = nullptr;
    size_t cub_temp_cap = 0;

    float* scores_tmp = nullptr;
    int64_t scores_tmp_cap = 0;

    int32_t* perm_buf = nullptr;
    int64_t perm_buf_cap = 0;

    float* sorted_scores_buf = nullptr;
    int64_t sorted_scores_buf_cap = 0;

    int32_t* sorted_perm_buf = nullptr;
    int64_t sorted_perm_buf_cap = 0;

    void ensure_seeds(int64_t n) {
        if (seeds_buf_cap < n) {
            if (seeds_buf) cudaFree(seeds_buf);
            cudaMalloc(&seeds_buf, n * sizeof(int32_t));
            seeds_buf_cap = n;
        }
    }

    void ensure_counts(int64_t n) {
        if (counts_cap < n) {
            if (counts) cudaFree(counts);
            cudaMalloc(&counts, n * sizeof(int64_t));
            counts_cap = n;
        }
    }

    void ensure_seed_offsets(int64_t n) {
        if (seed_offsets_d_cap < n) {
            if (seed_offsets_d) cudaFree(seed_offsets_d);
            cudaMalloc(&seed_offsets_d, n * sizeof(int64_t));
            seed_offsets_d_cap = n;
        }
    }

    void ensure_raw(int64_t n) {
        if (raw_keys_cap < n) {
            if (raw_keys) cudaFree(raw_keys);
            cudaMalloc(&raw_keys, n * sizeof(int64_t));
            raw_keys_cap = n;
        }
        if (sorted_keys_cap < n) {
            if (sorted_keys) cudaFree(sorted_keys);
            cudaMalloc(&sorted_keys, n * sizeof(int64_t));
            sorted_keys_cap = n;
        }
        if (flags_buf_cap < n) {
            if (flags_buf) cudaFree(flags_buf);
            cudaMalloc(&flags_buf, n * sizeof(uint8_t));
            flags_buf_cap = n;
        }
        if (filtered_keys_cap < n) {
            if (filtered_keys) cudaFree(filtered_keys);
            cudaMalloc(&filtered_keys, n * sizeof(int64_t));
            filtered_keys_cap = n;
        }
    }

    void ensure_num_filtered() {
        if (!num_filtered_d) {
            cudaMalloc(&num_filtered_d, sizeof(int64_t));
        }
    }

    void ensure_cub(size_t n) {
        if (cub_temp_cap < n) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, n);
            cub_temp_cap = n;
        }
    }

    void ensure_topk_scratch(int64_t n) {
        if (scores_tmp_cap < n) {
            if (scores_tmp) cudaFree(scores_tmp);
            cudaMalloc(&scores_tmp, n * sizeof(float));
            scores_tmp_cap = n;
        }
        if (perm_buf_cap < n) {
            if (perm_buf) cudaFree(perm_buf);
            cudaMalloc(&perm_buf, n * sizeof(int32_t));
            perm_buf_cap = n;
        }
        if (sorted_scores_buf_cap < n) {
            if (sorted_scores_buf) cudaFree(sorted_scores_buf);
            cudaMalloc(&sorted_scores_buf, n * sizeof(float));
            sorted_scores_buf_cap = n;
        }
        if (sorted_perm_buf_cap < n) {
            if (sorted_perm_buf) cudaFree(sorted_perm_buf);
            cudaMalloc(&sorted_perm_buf, n * sizeof(int32_t));
            sorted_perm_buf_cap = n;
        }
    }

    ~Cache() override {
        if (seeds_buf) cudaFree(seeds_buf);
        if (counts) cudaFree(counts);
        if (seed_offsets_d) cudaFree(seed_offsets_d);
        if (raw_keys) cudaFree(raw_keys);
        if (sorted_keys) cudaFree(sorted_keys);
        if (flags_buf) cudaFree(flags_buf);
        if (filtered_keys) cudaFree(filtered_keys);
        if (num_filtered_d) cudaFree(num_filtered_d);
        if (cub_temp) cudaFree(cub_temp);
        if (scores_tmp) cudaFree(scores_tmp);
        if (perm_buf) cudaFree(perm_buf);
        if (sorted_scores_buf) cudaFree(sorted_scores_buf);
        if (sorted_perm_buf) cudaFree(sorted_perm_buf);
    }
};

}  

similarity_result_float_t jaccard_all_pairs_similarity(
    const graph32_t& graph,
    const float* edge_weights,
    const int32_t* vertices,
    std::size_t num_vertices_param,
    std::optional<std::size_t> topk)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    cudaStream_t stream = 0;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t graph_nv = graph.number_of_vertices;
    const float* d_weights = edge_weights;

    
    int32_t num_seeds;
    const int32_t* d_seeds;

    if (vertices != nullptr) {
        num_seeds = (int32_t)num_vertices_param;
        d_seeds = vertices;
    } else {
        num_seeds = graph_nv;
        cache.ensure_seeds(graph_nv);
        launch_iota(cache.seeds_buf, graph_nv, stream);
        d_seeds = cache.seeds_buf;
    }

    int64_t max_v = (int64_t)graph_nv;
    int end_bit = 1;
    {
        int64_t max_key = (max_v - 1) * max_v + (max_v - 1);
        while ((1LL << end_bit) <= max_key && end_bit < 64) end_bit++;
    }

    
    cache.ensure_counts(num_seeds);
    launch_count_pairs(d_offsets, d_indices, d_seeds, num_seeds,
        cache.counts, stream);

    
    std::vector<int64_t> h_counts(num_seeds);
    cudaMemcpyAsync(h_counts.data(), cache.counts,
        num_seeds * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int64_t total_raw = 0;
    std::vector<int64_t> h_offsets(num_seeds);
    for (int i = 0; i < num_seeds; i++) {
        h_offsets[i] = total_raw;
        total_raw += h_counts[i];
    }

    if (total_raw <= 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    cache.ensure_seed_offsets(num_seeds);
    cudaMemcpyAsync(cache.seed_offsets_d, h_offsets.data(),
        num_seeds * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    
    cache.ensure_raw(total_raw);
    launch_write_pairs(d_offsets, d_indices, d_seeds, num_seeds,
        cache.seed_offsets_d, cache.raw_keys, max_v, stream);

    
    size_t sort_temp_bytes = get_sort_keys_temp_bytes(total_raw, end_bit);
    size_t flag_temp_bytes = get_flagged_temp_bytes(total_raw);
    size_t max_temp = sort_temp_bytes;
    if (flag_temp_bytes > max_temp) max_temp = flag_temp_bytes;
    if (max_temp == 0) max_temp = 4;
    cache.ensure_cub(max_temp);

    launch_sort_keys(cache.raw_keys, cache.sorted_keys,
        total_raw, cache.cub_temp, sort_temp_bytes, end_bit, stream);

    
    launch_compute_unique_flags(cache.sorted_keys,
        cache.flags_buf, total_raw, max_v, stream);

    
    cache.ensure_num_filtered();
    launch_flagged_select(cache.sorted_keys, cache.flags_buf,
        cache.filtered_keys, cache.num_filtered_d,
        total_raw, cache.cub_temp, flag_temp_bytes, stream);

    int64_t num_filtered = 0;
    cudaMemcpyAsync(&num_filtered, cache.num_filtered_d,
        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (num_filtered <= 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    if (topk.has_value() && num_filtered > (int64_t)topk.value()) {
        int64_t topk_val = (int64_t)topk.value();

        
        cache.ensure_topk_scratch(num_filtered);
        launch_jaccard(d_offsets, d_indices, d_weights,
            cache.filtered_keys, max_v,
            cache.scores_tmp, num_filtered, stream);

        launch_iota(cache.perm_buf, num_filtered, stream);

        size_t topk_temp_bytes = get_sort_pairs_desc_temp_bytes(num_filtered);
        cache.ensure_cub(topk_temp_bytes);

        launch_sort_pairs_desc(cache.scores_tmp, cache.sorted_scores_buf,
            cache.perm_buf, cache.sorted_perm_buf,
            num_filtered, cache.cub_temp, topk_temp_bytes, stream);

        
        int32_t* out_first = nullptr;
        int32_t* out_second = nullptr;
        float* out_scores = nullptr;
        cudaMalloc(&out_first, topk_val * sizeof(int32_t));
        cudaMalloc(&out_second, topk_val * sizeof(int32_t));
        cudaMalloc(&out_scores, topk_val * sizeof(float));

        launch_gather_topk(cache.filtered_keys,
            cache.sorted_scores_buf, cache.sorted_perm_buf,
            max_v, out_first, out_second, out_scores, topk_val, stream);

        return {out_first, out_second, out_scores, (std::size_t)topk_val};
    }

    
    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;
    cudaMalloc(&out_first, num_filtered * sizeof(int32_t));
    cudaMalloc(&out_second, num_filtered * sizeof(int32_t));
    cudaMalloc(&out_scores, num_filtered * sizeof(float));

    launch_jaccard(d_offsets, d_indices, d_weights,
        cache.filtered_keys, max_v,
        out_scores, num_filtered, stream);

    launch_decode_keys(cache.filtered_keys, num_filtered, max_v,
        out_first, out_second, stream);

    return {out_first, out_second, out_scores, (std::size_t)num_filtered};
}

}  
