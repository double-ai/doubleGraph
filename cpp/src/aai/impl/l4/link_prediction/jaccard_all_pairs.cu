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
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <cstdint>

namespace aai {

namespace {

struct Cache : Cacheable {
    static constexpr size_t MAX_KEYS_PREALLOC = 16 * 1024 * 1024;

    int64_t* d_key_buf = nullptr;
    unsigned long long* d_counter = nullptr;
    void* d_cub_temp = nullptr;
    size_t cub_temp_size = 0;
    float* d_scores_sort = nullptr;
    int32_t* d_idx_buf = nullptr;
    int32_t* d_idx_sorted = nullptr;
    size_t sort_buf_size = 0;

    Cache() {
        cudaMalloc(&d_key_buf, MAX_KEYS_PREALLOC * sizeof(int64_t));
        cudaMalloc(&d_counter, sizeof(unsigned long long));
        cub_temp_size = 1024 * 1024;
        cudaMalloc(&d_cub_temp, cub_temp_size);
        sort_buf_size = 4 * 1024 * 1024;
        cudaMalloc(&d_scores_sort, sort_buf_size * sizeof(float));
        cudaMalloc(&d_idx_buf, sort_buf_size * sizeof(int32_t));
        cudaMalloc(&d_idx_sorted, sort_buf_size * sizeof(int32_t));
    }

    ~Cache() override {
        if (d_key_buf) cudaFree(d_key_buf);
        if (d_counter) cudaFree(d_counter);
        if (d_cub_temp) cudaFree(d_cub_temp);
        if (d_scores_sort) cudaFree(d_scores_sort);
        if (d_idx_buf) cudaFree(d_idx_buf);
        if (d_idx_sorted) cudaFree(d_idx_sorted);
    }

    void ensure_sort_bufs(size_t needed) {
        if (needed > sort_buf_size) {
            cudaFree(d_scores_sort);
            cudaFree(d_idx_buf);
            cudaFree(d_idx_sorted);
            sort_buf_size = needed * 2;
            cudaMalloc(&d_scores_sort, sort_buf_size * sizeof(float));
            cudaMalloc(&d_idx_buf, sort_buf_size * sizeof(int32_t));
            cudaMalloc(&d_idx_sorted, sort_buf_size * sizeof(int32_t));
        }
    }
};




__global__ void count_2hop_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    int64_t* __restrict__ counts)
{
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int32_t u = seeds[sid];
    int32_t us = offsets[u], ue = offsets[u + 1];
    int64_t c = 0;
    for (int i = us + threadIdx.x; i < ue; i += blockDim.x) {
        int32_t w = indices[i];
        c += (int64_t)(offsets[w + 1] - offsets[w]);
    }
    for (int o = 16; o > 0; o >>= 1)
        c += __shfl_down_sync(0xffffffff, c, o);
    __shared__ int64_t smem[8];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) smem[wid] = c;
    __syncthreads();
    if (wid == 0) {
        c = (lane < (blockDim.x >> 5)) ? smem[lane] : 0;
        for (int o = 16; o > 0; o >>= 1)
            c += __shfl_down_sync(0xffffffff, c, o);
        if (lane == 0) counts[sid] = c;
    }
}




__global__ void enumerate_2hop_atomic_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    int64_t* __restrict__ keys,
    unsigned long long max_keys,
    unsigned long long* __restrict__ global_counter)
{
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int32_t u = seeds[sid];
    int32_t us = offsets[u], ue = offsets[u + 1];
    int64_t u64 = (int64_t)u << 32;

    for (int i = us; i < ue; i++) {
        int32_t w = indices[i];
        int32_t ws = offsets[w], we = offsets[w + 1];
        int32_t wd = we - ws;
        if (wd == 0) continue;

        __shared__ unsigned long long s_base;
        if (threadIdx.x == 0)
            s_base = atomicAdd(global_counter, (unsigned long long)wd);
        __syncthreads();

        unsigned long long base = s_base;
        if (base + wd <= max_keys) {
            for (int j = threadIdx.x; j < wd; j += blockDim.x)
                keys[base + j] = u64 | (uint32_t)indices[ws + j];
        }
        __syncthreads();
    }
}




__global__ void enumerate_2hop_keys_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    const int64_t* __restrict__ starts,
    int64_t* __restrict__ keys)
{
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int32_t u = seeds[sid];
    int32_t us = offsets[u], ue = offsets[u + 1];
    int64_t base = starts[sid];
    int64_t u64 = (int64_t)u << 32;
    int64_t off = 0;
    for (int i = us; i < ue; i++) {
        int32_t w = indices[i];
        int32_t ws = offsets[w], we = offsets[w + 1];
        int32_t wd = we - ws;
        for (int j = threadIdx.x; j < wd; j += blockDim.x)
            keys[base + off + j] = u64 | (uint32_t)indices[ws + j];
        off += wd;
    }
}




__device__ __forceinline__ int intersect_sorted(
    const int32_t* __restrict__ a, int na,
    const int32_t* __restrict__ b, int nb)
{
    if (na == 0 || nb == 0) return 0;
    if (na > nb) {
        const int32_t* t = a; a = b; b = t;
        int tn = na; na = nb; nb = tn;
    }
    if (a[na - 1] < b[0] || b[nb - 1] < a[0]) return 0;

    int count = 0, j = 0;
    for (int i = 0; i < na && j < nb; i++) {
        int32_t target = a[i];
        if (b[j] < target) {
            int step = 1, pos = j;
            while (pos + step < nb && b[pos + step] < target) {
                pos += step; step <<= 1;
            }
            int lo = pos, hi = (pos + step < nb) ? (pos + step + 1) : nb;
            while (lo < hi) {
                int m = (lo + hi) >> 1;
                if (b[m] < target) lo = m + 1; else hi = m;
            }
            j = lo;
        }
        if (j < nb && b[j] == target) { count++; j++; }
    }
    return count;
}




__global__ void decode_and_jaccard_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int64_t* __restrict__ keys,
    int64_t num_pairs,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int64_t key = keys[idx];
    int32_t u = (int32_t)(key >> 32);
    int32_t v = (int32_t)(key & 0xFFFFFFFF);
    out_first[idx] = u;
    out_second[idx] = v;

    int32_t us = offsets[u], ue = offsets[u + 1];
    int32_t vs = offsets[v], ve = offsets[v + 1];
    int ud = ue - us, vd = ve - vs;
    int isect = intersect_sorted(indices + us, ud, indices + vs, vd);
    int union_s = ud + vd - isect;
    out_scores[idx] = (union_s > 0) ? (float)isect / (float)union_s : 0.0f;
}




__global__ void gather_topk_kernel(
    const int32_t* __restrict__ first_in,
    const int32_t* __restrict__ second_in,
    const float* __restrict__ scores_in,
    const int32_t* __restrict__ indices,
    int64_t k,
    int32_t* __restrict__ first_out,
    int32_t* __restrict__ second_out,
    float* __restrict__ scores_out)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    int32_t idx = indices[i];
    first_out[i] = first_in[idx];
    second_out[i] = second_in[idx];
    scores_out[i] = scores_in[idx];
}




struct IsSelfLoopKey {
    __host__ __device__ bool operator()(int64_t key) const {
        return ((int32_t)(key >> 32)) == ((int32_t)(key & 0xFFFFFFFF));
    }
};





void do_count_2hop(
    const int32_t* offsets, const int32_t* indices,
    const int32_t* seeds, int num_seeds, int64_t* counts)
{
    if (num_seeds == 0) return;
    count_2hop_kernel<<<num_seeds, 256>>>(offsets, indices, seeds, num_seeds, counts);
}

void do_exclusive_scan(int64_t* data, int64_t* output, int n)
{
    thrust::device_ptr<int64_t> in(data), out(output);
    thrust::exclusive_scan(thrust::device, in, in + n, out);
}

void do_enumerate_2hop(
    const int32_t* offsets, const int32_t* indices,
    const int32_t* seeds, int num_seeds,
    const int64_t* starts, int64_t* keys)
{
    if (num_seeds == 0) return;
    enumerate_2hop_keys_kernel<<<num_seeds, 256>>>(
        offsets, indices, seeds, num_seeds, starts, keys);
}

unsigned long long do_enumerate_2hop_atomic(
    const int32_t* offsets, const int32_t* indices,
    const int32_t* seeds, int num_seeds,
    int64_t* keys, unsigned long long max_keys,
    unsigned long long* d_counter)
{
    if (num_seeds == 0) return 0;
    cudaMemsetAsync(d_counter, 0, sizeof(unsigned long long));
    enumerate_2hop_atomic_kernel<<<num_seeds, 256>>>(
        offsets, indices, seeds, num_seeds, keys, max_keys, d_counter);
    unsigned long long total;
    cudaMemcpy(&total, d_counter, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    return total;
}

int64_t do_sort_unique_dedup(int64_t* keys, int64_t n)
{
    if (n == 0) return 0;
    thrust::device_ptr<int64_t> ptr(keys);
    thrust::sort(thrust::device, ptr, ptr + n);
    auto end = thrust::unique(thrust::device, ptr, ptr + n);
    int64_t n2 = end - ptr;
    auto end2 = thrust::remove_if(thrust::device, ptr, ptr + n2, IsSelfLoopKey());
    return end2 - ptr;
}

void do_decode_and_jaccard(
    const int32_t* offsets, const int32_t* indices,
    const int64_t* keys, int64_t num_pairs,
    int32_t* out_first, int32_t* out_second, float* out_scores)
{
    if (num_pairs == 0) return;
    int grid = (int)((num_pairs + 255) / 256);
    decode_and_jaccard_kernel<<<grid, 256>>>(
        offsets, indices, keys, num_pairs,
        out_first, out_second, out_scores);
}

int64_t do_topk_cub(
    const int32_t* first_in, const int32_t* second_in, const float* scores_in,
    int64_t num_pairs, int64_t k,
    int32_t* first_out, int32_t* second_out, float* scores_out,
    void* d_temp, size_t temp_size,
    float* d_scores_sorted, int32_t* d_indices_buf, int32_t* d_indices_sorted)
{
    if (k >= num_pairs) {
        cudaMemcpy(first_out, first_in, num_pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(second_out, second_in, num_pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(scores_out, scores_in, num_pairs * sizeof(float), cudaMemcpyDeviceToDevice);
        return num_pairs;
    }

    thrust::device_ptr<int32_t> idx_ptr(d_indices_buf);
    thrust::sequence(thrust::device, idx_ptr, idx_ptr + num_pairs);

    size_t needed = 0;
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr, needed,
        scores_in, d_scores_sorted,
        d_indices_buf, d_indices_sorted,
        (int)num_pairs);

    if (needed > temp_size) {
        thrust::device_ptr<float> sc(const_cast<float*>(scores_in));
        auto pair_zip = thrust::make_zip_iterator(
            thrust::make_tuple(thrust::device_ptr<int32_t>(const_cast<int32_t*>(first_in)),
                              thrust::device_ptr<int32_t>(const_cast<int32_t*>(second_in))));
        thrust::sort_by_key(thrust::device, sc, sc + num_pairs,
                           pair_zip, thrust::greater<float>());
        cudaMemcpy(first_out, first_in, k * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(second_out, second_in, k * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(scores_out, scores_in, k * sizeof(float), cudaMemcpyDeviceToDevice);
        return k;
    }

    cub::DeviceRadixSort::SortPairsDescending(
        d_temp, needed,
        scores_in, d_scores_sorted,
        d_indices_buf, d_indices_sorted,
        (int)num_pairs);

    int grid = (int)((k + 255) / 256);
    gather_topk_kernel<<<grid, 256>>>(
        first_in, second_in, scores_in, d_indices_sorted,
        k, first_out, second_out, scores_out);

    return k;
}

}  

similarity_result_float_t jaccard_all_pairs_similarity(const graph32_t& graph,
                                                       const int32_t* vertices,
                                                       std::size_t num_vertices,
                                                       std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t num_verts = graph.number_of_vertices;

    
    int32_t* d_seeds_alloc = nullptr;
    const int32_t* d_seeds;
    int num_seeds;

    if (vertices != nullptr) {
        d_seeds = vertices;
        num_seeds = (int)num_vertices;
    } else {
        cudaMalloc(&d_seeds_alloc, num_verts * sizeof(int32_t));
        thrust::device_ptr<int32_t> ptr(d_seeds_alloc);
        thrust::sequence(thrust::device, ptr, ptr + num_verts);
        d_seeds = d_seeds_alloc;
        num_seeds = num_verts;
    }

    if (num_seeds == 0) {
        if (d_seeds_alloc) cudaFree(d_seeds_alloc);
        return {nullptr, nullptr, nullptr, 0};
    }

    int64_t* d_keys;
    int64_t total;
    int64_t* d_keys_alloc = nullptr;

    if (num_seeds <= 1024) {
        
        unsigned long long actual = do_enumerate_2hop_atomic(
            d_off, d_idx, d_seeds, num_seeds,
            cache.d_key_buf, Cache::MAX_KEYS_PREALLOC, cache.d_counter);

        if (actual <= Cache::MAX_KEYS_PREALLOC) {
            d_keys = cache.d_key_buf;
            total = (int64_t)actual;
        } else {
            
            int64_t* d_counts;
            cudaMalloc(&d_counts, num_seeds * sizeof(int64_t));
            do_count_2hop(d_off, d_idx, d_seeds, num_seeds, d_counts);

            int64_t* d_starts;
            cudaMalloc(&d_starts, num_seeds * sizeof(int64_t));
            do_exclusive_scan(d_counts, d_starts, num_seeds);

            int64_t last_count, last_start;
            cudaMemcpy(&last_count, d_counts + num_seeds - 1,
                       sizeof(int64_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&last_start, d_starts + num_seeds - 1,
                       sizeof(int64_t), cudaMemcpyDeviceToHost);
            total = last_count + last_start;

            cudaFree(d_counts);

            cudaMalloc(&d_keys_alloc, total * sizeof(int64_t));
            d_keys = d_keys_alloc;
            do_enumerate_2hop(d_off, d_idx, d_seeds, num_seeds, d_starts, d_keys);

            cudaFree(d_starts);
        }
    } else {
        
        int64_t* d_counts;
        cudaMalloc(&d_counts, num_seeds * sizeof(int64_t));
        do_count_2hop(d_off, d_idx, d_seeds, num_seeds, d_counts);

        int64_t* d_starts;
        cudaMalloc(&d_starts, num_seeds * sizeof(int64_t));
        do_exclusive_scan(d_counts, d_starts, num_seeds);

        int64_t last_count, last_start;
        cudaMemcpy(&last_count, d_counts + num_seeds - 1,
                   sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_start, d_starts + num_seeds - 1,
                   sizeof(int64_t), cudaMemcpyDeviceToHost);
        total = last_count + last_start;

        cudaFree(d_counts);

        if (total == 0) {
            cudaFree(d_starts);
            if (d_seeds_alloc) cudaFree(d_seeds_alloc);
            return {nullptr, nullptr, nullptr, 0};
        }

        cudaMalloc(&d_keys_alloc, total * sizeof(int64_t));
        d_keys = d_keys_alloc;
        do_enumerate_2hop(d_off, d_idx, d_seeds, num_seeds, d_starts, d_keys);

        cudaFree(d_starts);
    }

    
    if (d_seeds_alloc) { cudaFree(d_seeds_alloc); d_seeds_alloc = nullptr; }

    if (total == 0) {
        if (d_keys_alloc) cudaFree(d_keys_alloc);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t num_unique = do_sort_unique_dedup(d_keys, total);

    if (num_unique == 0) {
        if (d_keys_alloc) cudaFree(d_keys_alloc);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t* d_first;
    int32_t* d_second;
    float* d_scores;
    cudaMalloc(&d_first, num_unique * sizeof(int32_t));
    cudaMalloc(&d_second, num_unique * sizeof(int32_t));
    cudaMalloc(&d_scores, num_unique * sizeof(float));

    do_decode_and_jaccard(d_off, d_idx, d_keys, num_unique,
                          d_first, d_second, d_scores);

    
    if (d_keys_alloc) { cudaFree(d_keys_alloc); d_keys_alloc = nullptr; }

    
    if (topk.has_value() && (int64_t)topk.value() < num_unique) {
        int64_t k = (int64_t)topk.value();
        cache.ensure_sort_bufs((size_t)num_unique);

        int32_t* d_out_first;
        int32_t* d_out_second;
        float* d_out_scores;
        cudaMalloc(&d_out_first, k * sizeof(int32_t));
        cudaMalloc(&d_out_second, k * sizeof(int32_t));
        cudaMalloc(&d_out_scores, k * sizeof(float));

        int64_t result_count = do_topk_cub(
            d_first, d_second, d_scores,
            num_unique, k,
            d_out_first, d_out_second, d_out_scores,
            cache.d_cub_temp, cache.cub_temp_size,
            cache.d_scores_sort, cache.d_idx_buf, cache.d_idx_sorted);

        cudaFree(d_first);
        cudaFree(d_second);
        cudaFree(d_scores);

        return {d_out_first, d_out_second, d_out_scores, (std::size_t)result_count};
    }

    return {d_first, d_second, d_scores, (std::size_t)num_unique};
}

}  
