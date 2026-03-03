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
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <cstdint>
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    int64_t* d_total = nullptr;
    int32_t* d_nu = nullptr;

    Cache() {
        cudaMalloc(&d_total, sizeof(int64_t));
        cudaMalloc(&d_nu, sizeof(int32_t));
    }

    ~Cache() override {
        if (d_total) cudaFree(d_total);
        if (d_nu) cudaFree(d_nu);
    }
};





__global__ void compute_seed_wsum_kernel(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ weights,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    float* __restrict__ seed_wsum)
{
    int32_t wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= num_seeds) return;

    int32_t u = seeds[wid];
    int32_t s = offsets[u];
    int32_t e = offsets[u + 1];
    int32_t deg = e - s;

    float sum = 0.0f;
    for (int32_t i = lane; i < deg; i += 32)
        sum += weights[s + i];
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) seed_wsum[wid] = sum;
}

__global__ void iota_kernel(int32_t* out, int32_t n) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = idx;
}

__global__ void count_expanded_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ counts)
{
    int32_t sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int32_t u = seeds[sid];
    int32_t us = offsets[u];
    int32_t ue = offsets[u + 1];
    int32_t udeg = ue - us;

    int64_t cnt = 0;
    for (int32_t i = threadIdx.x; i < udeg; i += blockDim.x) {
        int32_t c = indices[us + i];
        cnt += (int64_t)(offsets[c + 1] - offsets[c] - 1);
    }

    typedef cub::BlockReduce<int64_t, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    int64_t total = BR(tmp).Sum(cnt);
    if (threadIdx.x == 0) counts[sid] = total;
}

__global__ void prefix_sum_kernel(const int64_t* __restrict__ in, int64_t* __restrict__ out,
                                   int64_t* __restrict__ total_out, int n) {
    if (threadIdx.x == 0) {
        int64_t sum = 0;
        out[0] = 0;
        for (int i = 0; i < n; i++) {
            sum += in[i];
            out[i + 1] = sum;
        }
        *total_out = sum;
    }
}

__global__ void expand_kernel_u32(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ seed_off,
    uint32_t* __restrict__ keys,
    uint32_t vp1)
{
    int32_t sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int32_t u = seeds[sid];
    int32_t us = offsets[u];
    int32_t ue = offsets[u + 1];
    int32_t udeg = ue - us;

    __shared__ int64_t wpos;
    if (threadIdx.x == 0) wpos = seed_off[sid];
    __syncthreads();

    for (int32_t i = threadIdx.x; i < udeg; i += blockDim.x) {
        int32_t c = indices[us + i];
        int32_t cs = offsets[c];
        int32_t ce = offsets[c + 1];
        int32_t cdeg = ce - cs;

        int64_t pos = atomicAdd((unsigned long long*)&wpos, (unsigned long long)(cdeg - 1));

        for (int32_t j = cs; j < ce; j++) {
            int32_t v = indices[j];
            if (v != u) {
                keys[pos++] = (uint32_t)sid * vp1 + (uint32_t)v;
            }
        }
    }
}

__global__ void compute_jaccard_kernel_u32(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ seed_wsum,
    const uint32_t* __restrict__ ukeys,
    int npairs,
    uint32_t vp1,
    const int32_t* __restrict__ seeds,
    int32_t* __restrict__ out_u,
    int32_t* __restrict__ out_v,
    float* __restrict__ out_s)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= npairs) return;

    uint32_t key = ukeys[wid];
    int32_t sid = (int32_t)(key / vp1);
    int32_t v = (int32_t)(key % vp1);
    int32_t u = seeds[sid];

    int32_t us = offsets[u], ue = offsets[u + 1];
    int32_t vs = offsets[v], ve = offsets[v + 1];
    int32_t udeg = ue - us;
    int32_t vdeg = ve - vs;

    
    float wu = seed_wsum[sid];

    
    float wv = 0.0f;
    for (int32_t i = lane; i < vdeg; i += 32) wv += weights[vs + i];
    for (int off = 16; off > 0; off >>= 1)
        wv += __shfl_down_sync(0xffffffff, wv, off);
    wv = __shfl_sync(0xffffffff, wv, 0);

    
    int32_t iter_s, iter_deg, srch_s, srch_e;
    if (udeg <= vdeg) {
        iter_s = us; iter_deg = udeg;
        srch_s = vs; srch_e = ve;
    } else {
        iter_s = vs; iter_deg = vdeg;
        srch_s = us; srch_e = ue;
    }

    float sum_min = 0.0f;
    for (int32_t i = lane; i < iter_deg; i += 32) {
        int32_t target = indices[iter_s + i];
        float w_iter = weights[iter_s + i];

        int32_t lo = srch_s, hi = srch_e;
        while (lo < hi) {
            int32_t mid = lo + ((hi - lo) >> 1);
            if (indices[mid] < target) lo = mid + 1;
            else hi = mid;
        }

        if (lo < srch_e && indices[lo] == target)
            sum_min += fminf(w_iter, weights[lo]);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum_min += __shfl_down_sync(0xffffffff, sum_min, off);

    if (lane == 0) {
        float den = wu + wv - sum_min;
        out_s[wid] = (den > 0.0f) ? (sum_min / den) : 0.0f;
        out_u[wid] = u;
        out_v[wid] = v;
    }
}

__global__ void pack_pairs_kernel(
    const int32_t* __restrict__ first, const int32_t* __restrict__ second,
    int64_t* __restrict__ packed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) packed[i] = ((int64_t)(uint32_t)first[i] << 32) | (uint32_t)second[i];
}

__global__ void unpack_pairs_kernel(
    const int64_t* __restrict__ packed,
    int32_t* __restrict__ first, int32_t* __restrict__ second, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        first[i] = (int32_t)(packed[i] >> 32);
        second[i] = (int32_t)(packed[i] & 0xFFFFFFFF);
    }
}

__global__ void expand_kernel_u64(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ seed_off,
    uint64_t* __restrict__ keys,
    uint64_t vp1)
{
    int32_t sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int32_t u = seeds[sid];
    int32_t us = offsets[u];
    int32_t ue = offsets[u + 1];
    int32_t udeg = ue - us;

    __shared__ int64_t wpos;
    if (threadIdx.x == 0) wpos = seed_off[sid];
    __syncthreads();

    for (int32_t i = threadIdx.x; i < udeg; i += blockDim.x) {
        int32_t c = indices[us + i];
        int32_t cs = offsets[c];
        int32_t ce = offsets[c + 1];
        int32_t cdeg = ce - cs;

        int64_t pos = atomicAdd((unsigned long long*)&wpos, (unsigned long long)(cdeg - 1));

        for (int32_t j = cs; j < ce; j++) {
            int32_t v = indices[j];
            if (v != u) {
                keys[pos++] = (uint64_t)sid * vp1 + (uint64_t)v;
            }
        }
    }
}

__global__ void compute_jaccard_kernel_u64(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ seed_wsum,
    const uint64_t* __restrict__ ukeys,
    int npairs,
    uint64_t vp1,
    const int32_t* __restrict__ seeds,
    int32_t* __restrict__ out_u,
    int32_t* __restrict__ out_v,
    float* __restrict__ out_s)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= npairs) return;

    uint64_t key = ukeys[wid];
    int32_t sid = (int32_t)(key / vp1);
    int32_t v = (int32_t)(key % vp1);
    int32_t u = seeds[sid];

    int32_t us = offsets[u], ue = offsets[u + 1];
    int32_t vs = offsets[v], ve = offsets[v + 1];
    int32_t udeg = ue - us;
    int32_t vdeg = ve - vs;

    float wu = seed_wsum[sid];

    float wv = 0.0f;
    for (int32_t i = lane; i < vdeg; i += 32) wv += weights[vs + i];
    for (int off = 16; off > 0; off >>= 1)
        wv += __shfl_down_sync(0xffffffff, wv, off);
    wv = __shfl_sync(0xffffffff, wv, 0);

    int32_t iter_s, iter_deg, srch_s, srch_e;
    if (udeg <= vdeg) {
        iter_s = us; iter_deg = udeg;
        srch_s = vs; srch_e = ve;
    } else {
        iter_s = vs; iter_deg = vdeg;
        srch_s = us; srch_e = ue;
    }

    float sum_min = 0.0f;
    for (int32_t i = lane; i < iter_deg; i += 32) {
        int32_t target = indices[iter_s + i];
        float w_iter = weights[iter_s + i];

        int32_t lo = srch_s, hi = srch_e;
        while (lo < hi) {
            int32_t mid = lo + ((hi - lo) >> 1);
            if (indices[mid] < target) lo = mid + 1;
            else hi = mid;
        }

        if (lo < srch_e && indices[lo] == target)
            sum_min += fminf(w_iter, weights[lo]);
    }

    for (int off = 16; off > 0; off >>= 1)
        sum_min += __shfl_down_sync(0xffffffff, sum_min, off);

    if (lane == 0) {
        float den = wu + wv - sum_min;
        out_s[wid] = (den > 0.0f) ? (sum_min / den) : 0.0f;
        out_u[wid] = u;
        out_v[wid] = v;
    }
}





static void launch_compute_seed_wsum(
    const int32_t* off, const float* w, const int32_t* seeds,
    int32_t ns, float* out, cudaStream_t s) {
    if (ns == 0) return;
    int wpb = 8; int tpb = wpb * 32;
    compute_seed_wsum_kernel<<<(ns+wpb-1)/wpb, tpb, 0, s>>>(off, w, seeds, ns, out);
}

static void launch_iota(int32_t* out, int32_t n, cudaStream_t s) {
    if (n == 0) return;
    iota_kernel<<<(n+255)/256, 256, 0, s>>>(out, n);
}

static void launch_count_expanded(
    const int32_t* off, const int32_t* idx, const int32_t* seeds,
    int32_t ns, int64_t* counts, cudaStream_t s) {
    if (ns == 0) return;
    count_expanded_kernel<<<ns, 256, 0, s>>>(off, idx, seeds, ns, counts);
}

static void launch_prefix_sum(const int64_t* in, int64_t* out, int64_t* total, int n, cudaStream_t s) {
    prefix_sum_kernel<<<1, 1, 0, s>>>(in, out, total, n);
}

static void launch_expand_u32(
    const int32_t* off, const int32_t* idx, const int32_t* seeds,
    int32_t ns, const int64_t* soff, uint32_t* keys, uint32_t vp1, cudaStream_t s) {
    if (ns == 0) return;
    expand_kernel_u32<<<ns, 256, 0, s>>>(off, idx, seeds, ns, soff, keys, vp1);
}

static size_t get_sort_temp_size(int64_t n, int end_bit) {
    size_t bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, bytes, (uint32_t*)nullptr, (uint32_t*)nullptr, n, 0, end_bit);
    return bytes;
}

static size_t get_unique_temp_size(int64_t n) {
    size_t bytes = 0;
    cub::DeviceSelect::Unique(nullptr, bytes, (uint32_t*)nullptr, (uint32_t*)nullptr, (int*)nullptr, n);
    return bytes;
}

static void launch_cub_sort(uint32_t* d_in, uint32_t* d_out, int64_t n,
                             void* temp, size_t temp_bytes, int end_bit, cudaStream_t s) {
    cub::DeviceRadixSort::SortKeys(temp, temp_bytes, d_in, d_out, n, 0, end_bit, s);
}

static void launch_cub_unique(uint32_t* d_in, uint32_t* d_out, int* d_num, int64_t n,
                               void* temp, size_t temp_bytes, cudaStream_t s) {
    cub::DeviceSelect::Unique(temp, temp_bytes, d_in, d_out, d_num, n, s);
}

static void launch_compute_jaccard_u32(
    const int32_t* off, const int32_t* idx, const float* w,
    const float* sw, const uint32_t* ukeys, int np, uint32_t vp1,
    const int32_t* seeds, int32_t* ou, int32_t* ov, float* os, cudaStream_t s) {
    if (np == 0) return;
    int tpb = 256; int wpb = tpb >> 5;
    int grid = (np + wpb - 1) / wpb;
    compute_jaccard_kernel_u32<<<grid, tpb, 0, s>>>(off, idx, w, sw, ukeys, np, vp1, seeds, ou, ov, os);
}

static size_t get_sort_u64_temp_size(int64_t n, int end_bit) {
    size_t bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, bytes, (uint64_t*)nullptr, (uint64_t*)nullptr, n, 0, end_bit);
    return bytes;
}

static size_t get_unique_u64_temp_size(int64_t n) {
    size_t bytes = 0;
    cub::DeviceSelect::Unique(nullptr, bytes, (uint64_t*)nullptr, (uint64_t*)nullptr, (int*)nullptr, n);
    return bytes;
}

static void launch_cub_sort_u64(uint64_t* d_in, uint64_t* d_out, int64_t n,
                                 void* temp, size_t temp_bytes, int end_bit, cudaStream_t s) {
    cub::DeviceRadixSort::SortKeys(temp, temp_bytes, d_in, d_out, n, 0, end_bit, s);
}

static void launch_cub_unique_u64(uint64_t* d_in, uint64_t* d_out, int* d_num, int64_t n,
                                   void* temp, size_t temp_bytes, cudaStream_t s) {
    cub::DeviceSelect::Unique(temp, temp_bytes, d_in, d_out, d_num, n, s);
}

static void launch_expand_u64(
    const int32_t* off, const int32_t* idx, const int32_t* seeds,
    int32_t ns, const int64_t* soff, uint64_t* keys, uint64_t vp1, cudaStream_t s) {
    if (ns == 0) return;
    expand_kernel_u64<<<ns, 256, 0, s>>>(off, idx, seeds, ns, soff, keys, vp1);
}

static void launch_compute_jaccard_u64(
    const int32_t* off, const int32_t* idx, const float* w,
    const float* sw, const uint64_t* ukeys, int np, uint64_t vp1,
    const int32_t* seeds, int32_t* ou, int32_t* ov, float* os, cudaStream_t s) {
    if (np == 0) return;
    int tpb = 256; int wpb = tpb >> 5;
    int grid = (np + wpb - 1) / wpb;
    compute_jaccard_kernel_u64<<<grid, tpb, 0, s>>>(off, idx, w, sw, ukeys, np, vp1, seeds, ou, ov, os);
}

static void launch_topk_sort(
    float* scores, int32_t* first, int32_t* second,
    int np, int64_t* packed, cudaStream_t s) {
    if (np <= 1) return;
    int blk = 256; int grd = (np + blk - 1) / blk;
    pack_pairs_kernel<<<grd, blk, 0, s>>>(first, second, packed, np);
    thrust::device_ptr<float> ds(scores);
    thrust::device_ptr<int64_t> dp(packed);
    thrust::sort_by_key(thrust::cuda::par.on(s), ds, ds + np, dp, thrust::greater<float>());
    unpack_pairs_kernel<<<grd, blk, 0, s>>>(packed, first, second, np);
}

}  

similarity_result_float_t jaccard_all_pairs_similarity(const graph32_t& graph,
                                                       const float* edge_weights,
                                                       const int32_t* vertices,
                                                       std::size_t num_vertices,
                                                       std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    const float* d_ew = edge_weights;
    cudaStream_t stream = 0;

    
    int32_t* seeds_buf = nullptr;
    const int32_t* d_seeds;
    int32_t ns;
    if (vertices != nullptr && num_vertices > 0) {
        d_seeds = vertices;
        ns = (int32_t)num_vertices;
    } else {
        cudaMalloc(&seeds_buf, (size_t)nv * sizeof(int32_t));
        launch_iota(seeds_buf, nv, stream);
        d_seeds = seeds_buf;
        ns = nv;
    }

    
    float* seed_wsum = nullptr;
    cudaMalloc(&seed_wsum, (size_t)ns * sizeof(float));
    launch_compute_seed_wsum(d_off, d_ew, d_seeds, ns, seed_wsum, stream);

    
    int64_t* counts = nullptr;
    cudaMalloc(&counts, (size_t)ns * sizeof(int64_t));
    launch_count_expanded(d_off, d_idx, d_seeds, ns, counts, stream);

    
    int64_t* d_offsets_arr = nullptr;
    cudaMalloc(&d_offsets_arr, (size_t)(ns + 1) * sizeof(int64_t));
    launch_prefix_sum(counts, d_offsets_arr, cache.d_total, ns, stream);

    int64_t total = 0;
    cudaMemcpy(&total, cache.d_total, sizeof(int64_t), cudaMemcpyDeviceToHost);

    cudaFree(counts);

    if (total == 0) {
        if (seeds_buf) cudaFree(seeds_buf);
        cudaFree(seed_wsum);
        cudaFree(d_offsets_arr);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    uint64_t max_key_64 = (uint64_t)(ns - 1) * (uint64_t)(nv + 1) + (uint64_t)(nv - 1);
    bool use_u32 = (max_key_64 <= 0xFFFFFFFFULL);

    int32_t nu = 0;
    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;
    std::size_t result_count = 0;

    if (use_u32) {
        
        uint32_t vp1 = (uint32_t)(nv + 1);
        uint32_t max_key = (uint32_t)max_key_64;

        int end_bit = 0;
        { uint32_t mk = max_key; while (mk > 0) { end_bit++; mk >>= 1; } }
        if (end_bit == 0) end_bit = 1;

        size_t sort_temp = get_sort_temp_size(total, end_bit);
        size_t unique_temp = get_unique_temp_size(total);
        size_t max_temp = std::max(sort_temp, unique_temp);

        uint32_t* keys = nullptr;
        uint32_t* sorted = nullptr;
        uint32_t* unique_out = nullptr;
        uint8_t* temp_buf = nullptr;
        cudaMalloc(&keys, (size_t)total * sizeof(uint32_t));
        cudaMalloc(&sorted, (size_t)total * sizeof(uint32_t));
        cudaMalloc(&unique_out, (size_t)total * sizeof(uint32_t));
        cudaMalloc(&temp_buf, max_temp);

        launch_expand_u32(d_off, d_idx, d_seeds, ns, d_offsets_arr, keys, vp1, stream);
        launch_cub_sort(keys, sorted, total, temp_buf, sort_temp, end_bit, stream);
        launch_cub_unique(sorted, unique_out, cache.d_nu, total, temp_buf, unique_temp, stream);

        cudaMemcpy(&nu, cache.d_nu, sizeof(int32_t), cudaMemcpyDeviceToHost);

        cudaFree(keys);
        cudaFree(sorted);
        cudaFree(temp_buf);

        if (nu == 0) {
            cudaFree(unique_out);
            if (seeds_buf) cudaFree(seeds_buf);
            cudaFree(seed_wsum);
            cudaFree(d_offsets_arr);
            return {nullptr, nullptr, nullptr, 0};
        }

        
        int32_t* first = nullptr;
        int32_t* second = nullptr;
        float* scores = nullptr;
        cudaMalloc(&first, (size_t)nu * sizeof(int32_t));
        cudaMalloc(&second, (size_t)nu * sizeof(int32_t));
        cudaMalloc(&scores, (size_t)nu * sizeof(float));

        launch_compute_jaccard_u32(d_off, d_idx, d_ew, seed_wsum,
                                   unique_out, nu, vp1, d_seeds,
                                   first, second, scores, stream);

        cudaFree(unique_out);

        
        if (topk.has_value() && (int64_t)topk.value() < (int64_t)nu) {
            int k = (int)topk.value();
            int64_t* packed = nullptr;
            cudaMalloc(&packed, (size_t)nu * sizeof(int64_t));
            launch_topk_sort(scores, first, second, nu, packed, stream);
            cudaFree(packed);

            cudaMalloc(&out_first, (size_t)k * sizeof(int32_t));
            cudaMalloc(&out_second, (size_t)k * sizeof(int32_t));
            cudaMalloc(&out_scores, (size_t)k * sizeof(float));
            cudaMemcpyAsync(out_first, first, (size_t)k * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_second, second, (size_t)k * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_scores, scores, (size_t)k * sizeof(float), cudaMemcpyDeviceToDevice, stream);

            cudaFree(first);
            cudaFree(second);
            cudaFree(scores);
            result_count = (std::size_t)k;
        } else {
            out_first = first;
            out_second = second;
            out_scores = scores;
            result_count = (std::size_t)nu;
        }
    } else {
        
        uint64_t vp1_64 = (uint64_t)(nv + 1);

        int end_bit = 0;
        { uint64_t mk = max_key_64; while (mk > 0) { end_bit++; mk >>= 1; } }
        if (end_bit == 0) end_bit = 1;

        size_t sort_temp = get_sort_u64_temp_size(total, end_bit);
        size_t unique_temp = get_unique_u64_temp_size(total);
        size_t max_temp = std::max(sort_temp, unique_temp);

        uint64_t* keys = nullptr;
        uint64_t* sorted = nullptr;
        uint64_t* unique_out_64 = nullptr;
        uint8_t* temp_buf = nullptr;
        cudaMalloc(&keys, (size_t)total * sizeof(uint64_t));
        cudaMalloc(&sorted, (size_t)total * sizeof(uint64_t));
        cudaMalloc(&unique_out_64, (size_t)total * sizeof(uint64_t));
        cudaMalloc(&temp_buf, max_temp);

        launch_expand_u64(d_off, d_idx, d_seeds, ns, d_offsets_arr, keys, vp1_64, stream);
        launch_cub_sort_u64(keys, sorted, total, temp_buf, sort_temp, end_bit, stream);
        launch_cub_unique_u64(sorted, unique_out_64, cache.d_nu, total, temp_buf, unique_temp, stream);

        cudaMemcpy(&nu, cache.d_nu, sizeof(int32_t), cudaMemcpyDeviceToHost);

        cudaFree(keys);
        cudaFree(sorted);
        cudaFree(temp_buf);

        if (nu == 0) {
            cudaFree(unique_out_64);
            if (seeds_buf) cudaFree(seeds_buf);
            cudaFree(seed_wsum);
            cudaFree(d_offsets_arr);
            return {nullptr, nullptr, nullptr, 0};
        }

        
        int32_t* first = nullptr;
        int32_t* second = nullptr;
        float* scores = nullptr;
        cudaMalloc(&first, (size_t)nu * sizeof(int32_t));
        cudaMalloc(&second, (size_t)nu * sizeof(int32_t));
        cudaMalloc(&scores, (size_t)nu * sizeof(float));

        launch_compute_jaccard_u64(d_off, d_idx, d_ew, seed_wsum,
                                   (const uint64_t*)unique_out_64,
                                   nu, vp1_64, d_seeds,
                                   first, second, scores, stream);

        cudaFree(unique_out_64);

        
        if (topk.has_value() && (int64_t)topk.value() < (int64_t)nu) {
            int k = (int)topk.value();
            int64_t* packed = nullptr;
            cudaMalloc(&packed, (size_t)nu * sizeof(int64_t));
            launch_topk_sort(scores, first, second, nu, packed, stream);
            cudaFree(packed);

            cudaMalloc(&out_first, (size_t)k * sizeof(int32_t));
            cudaMalloc(&out_second, (size_t)k * sizeof(int32_t));
            cudaMalloc(&out_scores, (size_t)k * sizeof(float));
            cudaMemcpyAsync(out_first, first, (size_t)k * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_second, second, (size_t)k * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_scores, scores, (size_t)k * sizeof(float), cudaMemcpyDeviceToDevice, stream);

            cudaFree(first);
            cudaFree(second);
            cudaFree(scores);
            result_count = (std::size_t)k;
        } else {
            out_first = first;
            out_second = second;
            out_scores = scores;
            result_count = (std::size_t)nu;
        }
    }

    
    if (seeds_buf) cudaFree(seeds_buf);
    cudaFree(seed_wsum);
    cudaFree(d_offsets_arr);

    return {out_first, out_second, out_scores, result_count};
}

}  
