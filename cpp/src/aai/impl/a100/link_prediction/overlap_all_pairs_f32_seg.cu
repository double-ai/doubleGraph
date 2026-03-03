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
#include <cstdint>
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <cfloat>
#include <algorithm>
#include <vector>

namespace aai {

namespace {

struct Cache : Cacheable {
    int64_t* cnt = nullptr;
    int64_t cnt_cap = 0;

    uint64_t* pk = nullptr;
    int64_t pk_cap = 0;

    uint64_t* pk_sorted = nullptr;
    int64_t pk_sorted_cap = 0;

    void* sort_temp = nullptr;
    size_t sort_temp_cap = 0;

    int32_t* flags = nullptr;
    int64_t flags_cap = 0;

    int64_t* scatter = nullptr;
    int64_t scatter_cap = 0;

    int32_t* seeds = nullptr;
    int64_t seeds_cap = 0;

    int32_t* wf = nullptr;
    int64_t wf_cap = 0;

    int32_t* ws = nullptr;
    int64_t ws_cap = 0;

    float* wsc = nullptr;
    int64_t wsc_cap = 0;

    ~Cache() override {
        if (cnt) cudaFree(cnt);
        if (pk) cudaFree(pk);
        if (pk_sorted) cudaFree(pk_sorted);
        if (sort_temp) cudaFree(sort_temp);
        if (flags) cudaFree(flags);
        if (scatter) cudaFree(scatter);
        if (seeds) cudaFree(seeds);
        if (wf) cudaFree(wf);
        if (ws) cudaFree(ws);
        if (wsc) cudaFree(wsc);
    }

    template <typename T>
    static void ensure(T*& ptr, int64_t& cap, int64_t need) {
        if (cap < need) {
            if (ptr) cudaFree(ptr);
            cudaMalloc(&ptr, need * sizeof(T));
            cap = need;
        }
    }

    static void ensure_bytes(void*& ptr, size_t& cap, size_t need) {
        if (cap < need) {
            if (ptr) cudaFree(ptr);
            cudaMalloc(&ptr, need);
            cap = need;
        }
    }
};



__global__ void count_two_hop_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ counts)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_seeds) return;
    int32_t u = seeds[i];
    int64_t cnt = 0;
    for (int32_t j = offsets[u]; j < offsets[u + 1]; j++) {
        int32_t w = indices[j];
        cnt += (int64_t)(offsets[w + 1] - offsets[w]);
    }
    counts[i] = cnt;
}

__global__ void enumerate_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ pair_offsets,
    uint64_t* __restrict__ pair_keys)
{
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int32_t u = seeds[sid];
    int64_t base = pair_offsets[sid];
    int64_t local_off = 0;
    uint64_t u_shifted = (uint64_t)(uint32_t)u << 32;

    for (int32_t j = offsets[u]; j < offsets[u + 1]; j++) {
        int32_t w = indices[j];
        int32_t w_start = offsets[w];
        int32_t w_deg = offsets[w + 1] - w_start;

        for (int32_t k = threadIdx.x; k < w_deg; k += blockDim.x) {
            pair_keys[base + local_off + k] = u_shifted | (uint64_t)(uint32_t)indices[w_start + k];
        }
        local_off += w_deg;
    }
}

__global__ void mark_unique_valid_kernel(
    const uint64_t* __restrict__ sorted_keys,
    int64_t num_pairs,
    int32_t* __restrict__ flags)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pairs) return;

    uint64_t key = sorted_keys[i];
    uint32_t u = (uint32_t)(key >> 32);
    uint32_t v = (uint32_t)(key & 0xFFFFFFFF);

    
    flags[i] = (u != v && (i == 0 || sorted_keys[i - 1] != key)) ? 1 : 0;
}

__global__ void scatter_pairs_kernel(
    const uint64_t* __restrict__ sorted_keys,
    const int32_t* __restrict__ flags,
    const int64_t* __restrict__ scatter_offsets,
    int64_t num_pairs,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_pairs) return;

    if (flags[i]) {
        int64_t out_idx = scatter_offsets[i];
        uint64_t key = sorted_keys[i];
        out_first[out_idx] = (int32_t)(key >> 32);
        out_second[out_idx] = (int32_t)(key & 0xFFFFFFFF);
    }
}

__global__ void compute_overlap_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    int64_t num_pairs,
    float* __restrict__ scores)
{
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int32_t u = first[warp_id];
    int32_t v = second[warp_id];

    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int32_t v_start = offsets[v], v_end = offsets[v + 1];
    int32_t u_len = u_end - u_start;
    int32_t v_len = v_end - v_start;

    
    float wd_u = 0.0f;
    for (int32_t j = u_start + lane; j < u_end; j += 32)
        wd_u += edge_weights[j];
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) wd_u += __shfl_down_sync(0xffffffff, wd_u, s);
    wd_u = __shfl_sync(0xffffffff, wd_u, 0);

    float wd_v = 0.0f;
    for (int32_t j = v_start + lane; j < v_end; j += 32)
        wd_v += edge_weights[j];
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) wd_v += __shfl_down_sync(0xffffffff, wd_v, s);
    wd_v = __shfl_sync(0xffffffff, wd_v, 0);

    float denom = fminf(wd_u, wd_v);
    if (denom <= FLT_MIN) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    const int32_t* iter_idx; const float* iter_wt; int32_t iter_len;
    const int32_t* srch_idx; const float* srch_wt; int32_t srch_len;

    if (u_len <= v_len) {
        iter_idx = indices + u_start; iter_wt = edge_weights + u_start; iter_len = u_len;
        srch_idx = indices + v_start; srch_wt = edge_weights + v_start; srch_len = v_len;
    } else {
        iter_idx = indices + v_start; iter_wt = edge_weights + v_start; iter_len = v_len;
        srch_idx = indices + u_start; srch_wt = edge_weights + u_start; srch_len = u_len;
    }

    float local_sum = 0.0f;
    for (int32_t i = lane; i < iter_len; i += 32) {
        int32_t target = iter_idx[i];
        float wi = iter_wt[i];

        int32_t lo = 0, hi = srch_len;
        while (lo < hi) {
            int32_t mid = (lo + hi) >> 1;
            if (srch_idx[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        if (lo < srch_len && srch_idx[lo] == target)
            local_sum += fminf(wi, srch_wt[lo]);
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, s);

    if (lane == 0) scores[warp_id] = local_sum / denom;
}

__global__ void pack_pairs_kernel(const int32_t* first, const int32_t* second, int64_t count, uint64_t* keys) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    keys[i] = ((uint64_t)(uint32_t)first[i] << 32) | (uint64_t)(uint32_t)second[i];
}

__global__ void unpack_pairs_kernel(const uint64_t* keys, int64_t count, int32_t* first, int32_t* second) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint64_t k = keys[i];
    first[i] = (int32_t)(k >> 32);
    second[i] = (int32_t)(k & 0xFFFFFFFF);
}



static void do_count_two_hop(const int32_t* offsets, const int32_t* indices, const int32_t* seeds, int32_t ns, int64_t* counts) {
    if (ns == 0) return;
    count_two_hop_kernel<<<(ns + 255) / 256, 256>>>(offsets, indices, seeds, ns, counts);
}

static void do_enumerate_pairs(const int32_t* offsets, const int32_t* indices, const int32_t* seeds, int32_t ns, const int64_t* po, uint64_t* pk) {
    if (ns == 0) return;
    enumerate_pairs_kernel<<<ns, 256>>>(offsets, indices, seeds, ns, po, pk);
}

static size_t get_sort_temp_size(int64_t n, int end_bit) {
    size_t sz = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, sz, (uint64_t*)nullptr, (uint64_t*)nullptr, (int)n, 0, end_bit);
    return sz;
}

static void do_cub_sort(void* temp, size_t temp_sz, const uint64_t* in, uint64_t* out, int64_t n, int end_bit) {
    cub::DeviceRadixSort::SortKeys(temp, temp_sz, in, out, (int)n, 0, end_bit);
}

static void do_mark_unique(const uint64_t* keys, int64_t n, int32_t* flags) {
    if (n == 0) return;
    mark_unique_valid_kernel<<<(int)std::min((n + 255) / 256, (int64_t)2147483647), 256>>>(keys, n, flags);
}

static size_t get_scan_temp_size(int64_t n) {
    size_t sz = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, sz, (int32_t*)nullptr, (int64_t*)nullptr, (int)n);
    return sz;
}

static void do_prefix_sum(void* temp, size_t temp_sz, const int32_t* in, int64_t* out, int64_t n) {
    cub::DeviceScan::ExclusiveSum(temp, temp_sz, in, out, (int)n);
}

static void do_scatter_pairs(const uint64_t* keys, const int32_t* flags, const int64_t* scatter, int64_t n, int32_t* first, int32_t* second) {
    if (n == 0) return;
    scatter_pairs_kernel<<<(int)std::min((n + 255) / 256, (int64_t)2147483647), 256>>>(keys, flags, scatter, n, first, second);
}

static void do_compute_overlap(const int32_t* offsets, const int32_t* indices, const float* ew, const int32_t* first, const int32_t* second, int64_t np, float* scores) {
    if (np == 0) return;
    int G = (int)std::min((np * 32 + 255) / 256, (int64_t)2147483647);
    compute_overlap_kernel<<<G, 256>>>(offsets, indices, ew, first, second, np, scores);
}

static int64_t do_topk_sort(int32_t* first, int32_t* second, float* scores, uint64_t* temp, int64_t count, int64_t topk) {
    if (count == 0) return 0;
    if (topk >= count) return count;

    pack_pairs_kernel<<<(int)std::min((count + 255) / 256, (int64_t)2147483647), 256>>>(first, second, count, temp);
    thrust::device_ptr<float> sp(scores);
    thrust::device_ptr<uint64_t> kp(temp);
    thrust::sort_by_key(thrust::device, sp, sp + count, kp, thrust::greater<float>());

    int64_t keep = topk;
    unpack_pairs_kernel<<<(int)std::min((keep + 255) / 256, (int64_t)2147483647), 256>>>(temp, keep, first, second);
    return keep;
}

}  

similarity_result_float_t overlap_all_pairs_similarity_seg(const graph32_t& graph,
                                                           const float* edge_weights,
                                                           const int32_t* vertices,
                                                           std::size_t num_vertices,
                                                           std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    const float* d_ew = edge_weights;

    
    int32_t ns;
    const int32_t* d_seeds;
    if (vertices != nullptr) {
        ns = (int32_t)num_vertices;
        d_seeds = vertices;
    } else {
        ns = nv;
        Cache::ensure(cache.seeds, cache.seeds_cap, (int64_t)nv);
        thrust::device_ptr<int32_t> ptr(cache.seeds);
        thrust::sequence(thrust::device, ptr, ptr + nv, 0);
        d_seeds = cache.seeds;
    }
    if (ns == 0) return {nullptr, nullptr, nullptr, 0};

    
    Cache::ensure(cache.cnt, cache.cnt_cap, (int64_t)(ns + 1));
    do_count_two_hop(d_off, d_idx, d_seeds, ns, cache.cnt);

    std::vector<int64_t> h_off(ns + 1);
    cudaMemcpy(h_off.data(), cache.cnt, ns * sizeof(int64_t), cudaMemcpyDeviceToHost);

    int64_t total = 0;
    for (int i = 0; i < ns; i++) {
        int64_t c = h_off[i];
        h_off[i] = total;
        total += c;
    }
    h_off[ns] = total;
    if (total == 0) return {nullptr, nullptr, nullptr, 0};

    cudaMemcpy(cache.cnt, h_off.data(), (ns + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);

    
    Cache::ensure(cache.pk, cache.pk_cap, total);
    do_enumerate_pairs(d_off, d_idx, d_seeds, ns, cache.cnt, cache.pk);

    
    int bits_needed = 1;
    { int v = nv - 1; while (v > 0) { v >>= 1; bits_needed++; } }
    int end_bit = 32 + bits_needed;
    if (end_bit > 64) end_bit = 64;

    
    Cache::ensure(cache.pk_sorted, cache.pk_sorted_cap, total);

    size_t sort_temp_sz = get_sort_temp_size(total, end_bit);
    Cache::ensure_bytes(cache.sort_temp, cache.sort_temp_cap, sort_temp_sz);

    do_cub_sort(cache.sort_temp, sort_temp_sz, cache.pk, cache.pk_sorted, total, end_bit);

    
    Cache::ensure(cache.flags, cache.flags_cap, total);
    do_mark_unique(cache.pk_sorted, total, cache.flags);

    
    Cache::ensure(cache.scatter, cache.scatter_cap, total + 1);
    size_t scan_temp_sz = get_scan_temp_size(total);
    Cache::ensure_bytes(cache.sort_temp, cache.sort_temp_cap, scan_temp_sz);

    do_prefix_sum(cache.sort_temp, scan_temp_sz, cache.flags, cache.scatter, total);

    
    int64_t h_last_offset;
    int32_t h_last_flag;
    cudaMemcpy(&h_last_offset, cache.scatter + total - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_last_flag, cache.flags + total - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
    int64_t unique_count = h_last_offset + h_last_flag;

    if (unique_count == 0) return {nullptr, nullptr, nullptr, 0};

    
    Cache::ensure(cache.wf, cache.wf_cap, unique_count);
    Cache::ensure(cache.ws, cache.ws_cap, unique_count);
    do_scatter_pairs(cache.pk_sorted, cache.flags, cache.scatter, total, cache.wf, cache.ws);

    
    Cache::ensure(cache.wsc, cache.wsc_cap, unique_count);
    do_compute_overlap(d_off, d_idx, d_ew, cache.wf, cache.ws, unique_count, cache.wsc);

    
    int64_t final_count = unique_count;
    if (topk.has_value() && unique_count > (int64_t)topk.value()) {
        final_count = do_topk_sort(cache.wf, cache.ws, cache.wsc, cache.pk_sorted, unique_count, (int64_t)topk.value());
    }

    if (final_count == 0) return {nullptr, nullptr, nullptr, 0};

    
    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;

    cudaMalloc(&out_first, final_count * sizeof(int32_t));
    cudaMalloc(&out_second, final_count * sizeof(int32_t));
    cudaMalloc(&out_scores, final_count * sizeof(float));

    cudaMemcpyAsync(out_first, cache.wf, final_count * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(out_second, cache.ws, final_count * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(out_scores, cache.wsc, final_count * sizeof(float), cudaMemcpyDeviceToDevice);

    return {out_first, out_second, out_scores, (std::size_t)final_count};
}

}  
