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
#include <optional>

namespace aai {

namespace {





struct Cache : Cacheable {
    
    int32_t* seeds_buf = nullptr;
    int64_t seeds_capacity = 0;

    
    float* seed_wsum = nullptr;
    int64_t seed_wsum_capacity = 0;

    
    int64_t* counts = nullptr;
    int64_t counts_capacity = 0;

    
    int64_t* offsets_arr = nullptr;
    int64_t offsets_arr_capacity = 0;

    
    int64_t* d_total = nullptr;
    bool d_total_allocated = false;

    
    int32_t* d_nu = nullptr;
    bool d_nu_allocated = false;

    void ensure_seeds(int64_t n) {
        if (seeds_capacity < n) {
            if (seeds_buf) cudaFree(seeds_buf);
            cudaMalloc(&seeds_buf, n * sizeof(int32_t));
            seeds_capacity = n;
        }
    }

    void ensure_seed_wsum(int64_t n) {
        if (seed_wsum_capacity < n) {
            if (seed_wsum) cudaFree(seed_wsum);
            cudaMalloc(&seed_wsum, n * sizeof(float));
            seed_wsum_capacity = n;
        }
    }

    void ensure_counts(int64_t n) {
        if (counts_capacity < n) {
            if (counts) cudaFree(counts);
            cudaMalloc(&counts, n * sizeof(int64_t));
            counts_capacity = n;
        }
    }

    void ensure_offsets_arr(int64_t n) {
        if (offsets_arr_capacity < n) {
            if (offsets_arr) cudaFree(offsets_arr);
            cudaMalloc(&offsets_arr, n * sizeof(int64_t));
            offsets_arr_capacity = n;
        }
    }

    void ensure_d_total() {
        if (!d_total_allocated) {
            cudaMalloc(&d_total, sizeof(int64_t));
            d_total_allocated = true;
        }
    }

    void ensure_d_nu() {
        if (!d_nu_allocated) {
            cudaMalloc(&d_nu, sizeof(int32_t));
            d_nu_allocated = true;
        }
    }

    ~Cache() override {
        if (seeds_buf) cudaFree(seeds_buf);
        if (seed_wsum) cudaFree(seed_wsum);
        if (counts) cudaFree(counts);
        if (offsets_arr) cudaFree(offsets_arr);
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

static void launch_expand_u64(
    const int32_t* off, const int32_t* idx, const int32_t* seeds,
    int32_t ns, const int64_t* soff, uint64_t* keys, uint64_t vp1, cudaStream_t s) {
    if (ns == 0) return;
    expand_kernel_u64<<<ns, 256, 0, s>>>(off, idx, seeds, ns, soff, keys, vp1);
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

    
    const int32_t* d_seeds;
    int32_t ns;
    if (vertices != nullptr && num_vertices > 0) {
        d_seeds = vertices;
        ns = (int32_t)num_vertices;
    } else {
        cache.ensure_seeds(nv);
        launch_iota(cache.seeds_buf, nv, stream);
        d_seeds = cache.seeds_buf;
        ns = nv;
    }

    
    cache.ensure_seed_wsum(ns);
    launch_compute_seed_wsum(d_off, d_ew, d_seeds, ns, cache.seed_wsum, stream);

    
    cache.ensure_counts(ns);
    launch_count_expanded(d_off, d_idx, d_seeds, ns, cache.counts, stream);

    
    cache.ensure_offsets_arr(ns + 1);
    cache.ensure_d_total();
    launch_prefix_sum(cache.counts, cache.offsets_arr, cache.d_total, ns, stream);

    int64_t total = 0;
    cudaMemcpy(&total, cache.d_total, sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (total == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    uint32_t vp1 = (uint32_t)(nv + 1);
    uint64_t max_key64 = (uint64_t)(ns - 1) * (uint64_t)vp1 + (uint64_t)(nv - 1);
    bool use_u32 = (max_key64 <= UINT32_MAX);

    
    int end_bit;
    if (use_u32) {
        uint32_t max_key = (uint32_t)max_key64;
        end_bit = 1;
        while (max_key >= (1U << end_bit) && end_bit < 32) end_bit++;
    } else {
        end_bit = 1;
        while (max_key64 >= (1ULL << end_bit) && end_bit < 64) end_bit++;
    }

    int64_t total_int = total;

    if (!use_u32) {
        
        uint64_t vp1_64 = (uint64_t)vp1;

        uint64_t* keys = nullptr;
        cudaMalloc(&keys, total * sizeof(uint64_t));
        launch_expand_u64(d_off, d_idx, d_seeds, ns, cache.offsets_arr,
                         keys, vp1_64, stream);

        uint64_t* sorted = nullptr;
        cudaMalloc(&sorted, total * sizeof(uint64_t));

        size_t sort_temp = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, sort_temp, (uint64_t*)nullptr, (uint64_t*)nullptr, total_int, 0, end_bit);
        size_t unique_temp = 0;
        cub::DeviceSelect::Unique(nullptr, unique_temp, (uint64_t*)nullptr, (uint64_t*)nullptr, (int*)nullptr, total_int);
        size_t max_temp = std::max(sort_temp, unique_temp);

        void* temp_buf = nullptr;
        cudaMalloc(&temp_buf, max_temp);

        cub::DeviceRadixSort::SortKeys(temp_buf, sort_temp, keys, sorted, total_int, 0, end_bit, stream);

        uint64_t* unique_out = nullptr;
        cudaMalloc(&unique_out, total * sizeof(uint64_t));
        cache.ensure_d_nu();

        cub::DeviceSelect::Unique(temp_buf, unique_temp, sorted, unique_out, cache.d_nu, total_int, stream);

        int32_t nu = 0;
        cudaMemcpy(&nu, cache.d_nu, sizeof(int32_t), cudaMemcpyDeviceToHost);

        cudaFree(keys);
        cudaFree(sorted);
        cudaFree(temp_buf);

        if (nu == 0) {
            cudaFree(unique_out);
            return {nullptr, nullptr, nullptr, 0};
        }

        int32_t* first = nullptr;
        int32_t* second = nullptr;
        float* scores = nullptr;
        cudaMalloc(&first, (int64_t)nu * sizeof(int32_t));
        cudaMalloc(&second, (int64_t)nu * sizeof(int32_t));
        cudaMalloc(&scores, (int64_t)nu * sizeof(float));

        launch_compute_jaccard_u64(d_off, d_idx, d_ew, cache.seed_wsum,
                                  unique_out, nu, vp1_64, d_seeds,
                                  first, second, scores, stream);

        cudaFree(unique_out);

        
        if (topk.has_value() && (int64_t)topk.value() < (int64_t)nu) {
            int k = (int)topk.value();
            int64_t* packed = nullptr;
            cudaMalloc(&packed, (int64_t)nu * sizeof(int64_t));
            launch_topk_sort(scores, first, second, nu, packed, stream);
            cudaFree(packed);

            int32_t* fo = nullptr;
            int32_t* so = nullptr;
            float* sc = nullptr;
            cudaMalloc(&fo, (int64_t)k * sizeof(int32_t));
            cudaMalloc(&so, (int64_t)k * sizeof(int32_t));
            cudaMalloc(&sc, (int64_t)k * sizeof(float));
            cudaMemcpyAsync(fo, first, k * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(so, second, k * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(sc, scores, k * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaDeviceSynchronize();
            cudaFree(first);
            cudaFree(second);
            cudaFree(scores);
            return {fo, so, sc, (std::size_t)k};
        }

        cudaDeviceSynchronize();
        return {first, second, scores, (std::size_t)nu};

    } else {
        
        size_t sort_temp = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, sort_temp, (uint32_t*)nullptr, (uint32_t*)nullptr, total_int, 0, end_bit);
        size_t unique_temp = 0;
        cub::DeviceSelect::Unique(nullptr, unique_temp, (uint32_t*)nullptr, (uint32_t*)nullptr, (int*)nullptr, total_int);
        size_t max_temp = std::max(sort_temp, unique_temp);

        uint32_t* keys = nullptr;
        uint32_t* sorted = nullptr;
        uint32_t* unique_out = nullptr;
        void* temp_buf = nullptr;
        cudaMalloc(&keys, total * sizeof(uint32_t));
        cudaMalloc(&sorted, total * sizeof(uint32_t));
        cudaMalloc(&unique_out, total * sizeof(uint32_t));
        cudaMalloc(&temp_buf, max_temp);
        cache.ensure_d_nu();

        
        launch_expand_u32(d_off, d_idx, d_seeds, ns, cache.offsets_arr,
                         keys, vp1, stream);

        
        cub::DeviceRadixSort::SortKeys(temp_buf, sort_temp, keys, sorted, total_int, 0, end_bit, stream);

        
        cub::DeviceSelect::Unique(temp_buf, unique_temp, sorted, unique_out, cache.d_nu, total_int, stream);

        
        int32_t nu = 0;
        cudaMemcpy(&nu, cache.d_nu, sizeof(int32_t), cudaMemcpyDeviceToHost);

        cudaFree(keys);
        cudaFree(sorted);
        cudaFree(temp_buf);

        if (nu == 0) {
            cudaFree(unique_out);
            return {nullptr, nullptr, nullptr, 0};
        }

        
        int32_t* first = nullptr;
        int32_t* second = nullptr;
        float* scores = nullptr;
        cudaMalloc(&first, (int64_t)nu * sizeof(int32_t));
        cudaMalloc(&second, (int64_t)nu * sizeof(int32_t));
        cudaMalloc(&scores, (int64_t)nu * sizeof(float));

        launch_compute_jaccard_u32(d_off, d_idx, d_ew, cache.seed_wsum,
                                  unique_out, nu, vp1, d_seeds,
                                  first, second, scores, stream);

        cudaFree(unique_out);

        
        if (topk.has_value() && (int64_t)topk.value() < (int64_t)nu) {
            int k = (int)topk.value();
            int64_t* packed = nullptr;
            cudaMalloc(&packed, (int64_t)nu * sizeof(int64_t));
            launch_topk_sort(scores, first, second, nu, packed, stream);
            cudaFree(packed);

            int32_t* fo = nullptr;
            int32_t* so = nullptr;
            float* sc = nullptr;
            cudaMalloc(&fo, (int64_t)k * sizeof(int32_t));
            cudaMalloc(&so, (int64_t)k * sizeof(int32_t));
            cudaMalloc(&sc, (int64_t)k * sizeof(float));
            cudaMemcpyAsync(fo, first, k * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(so, second, k * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(sc, scores, k * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaDeviceSynchronize();
            cudaFree(first);
            cudaFree(second);
            cudaFree(scores);
            return {fo, so, sc, (std::size_t)k};
        }

        cudaDeviceSynchronize();
        return {first, second, scores, (std::size_t)nu};
    }
}

}  
