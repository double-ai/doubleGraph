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



__global__ void iota_kernel(int32_t* out, int32_t n) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}

__global__ void compute_seed_info_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t ns,
    int64_t* __restrict__ work_sizes
) {
    int sid = blockIdx.x;
    if (sid >= ns) return;
    int32_t u = seeds[sid];
    int32_t us = offsets[u], ue = offsets[u + 1];
    int64_t w = 0;
    for (int i = us + (int)threadIdx.x; i < ue; i += (int)blockDim.x) {
        int32_t nb = indices[i];
        w += (int64_t)(offsets[nb + 1] - offsets[nb]);
    }
    typedef cub::BlockReduce<int64_t, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    int64_t tot = BR(tmp).Sum(w);
    if (threadIdx.x == 0) work_sizes[sid] = tot;
}

__global__ void generate_keys_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t ns,
    const int64_t* __restrict__ work_offsets,
    int64_t* __restrict__ keys,
    int64_t num_vertices_i64
) {
    int sid = blockIdx.x;
    if (sid >= ns) return;
    int32_t u = seeds[sid];
    int32_t us = offsets[u], ue = offsets[u + 1];
    int64_t base = work_offsets[sid];
    int64_t prefix = (int64_t)sid * num_vertices_i64;

    int64_t local_off = 0;
    for (int i = us; i < ue; i++) {
        int32_t w = indices[i];
        int32_t ws = offsets[w];
        int32_t wdeg = offsets[w + 1] - ws;
        for (int j = threadIdx.x; j < wdeg; j += (int)blockDim.x) {
            keys[base + local_off + j] = prefix + (int64_t)indices[ws + j];
        }
        local_off += wdeg;
    }
}

__global__ void compute_scores_kernel(
    const int64_t* __restrict__ unique_keys,
    const int32_t* __restrict__ rle_counts,
    const int32_t* __restrict__ d_num_unique,
    const int32_t* __restrict__ seeds,
    const int32_t* __restrict__ offsets,
    int64_t num_vertices_i64,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int32_t* __restrict__ global_count
) {
    int nu = *d_num_unique;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nu) return;

    int64_t key = unique_keys[idx];
    int32_t sid = (int32_t)(key / num_vertices_i64);
    int32_t v = (int32_t)(key % num_vertices_i64);
    int32_t u = seeds[sid];
    if (v == u) return;

    int32_t cnt = rle_counts[idx];
    int32_t du = offsets[u + 1] - offsets[u];
    int32_t dv = offsets[v + 1] - offsets[v];
    float score = 2.0f * (float)cnt / (float)(du + dv);

    int32_t pos = atomicAdd(global_count, 1);
    out_first[pos] = u;
    out_second[pos] = v;
    out_scores[pos] = score;
}

__device__ int merge_intersection(
    const int32_t* __restrict__ a, int32_t sa,
    const int32_t* __restrict__ b, int32_t sb
) {
    if (sa == 0 || sb == 0) return 0;
    if (a[sa-1] < b[0] || b[sb-1] < a[0]) return 0;
    int i = 0, j = 0;
    if (a[0] < b[0]) {
        int lo = 0, hi = sa;
        while (lo < hi) { int m = (lo+hi)/2; if (a[m] < b[0]) lo=m+1; else hi=m; }
        i = lo;
    } else if (b[0] < a[0]) {
        int lo = 0, hi = sb;
        while (lo < hi) { int m = (lo+hi)/2; if (b[m] < a[0]) lo=m+1; else hi=m; }
        j = lo;
    }
    int c = 0;
    while (i < sa && j < sb) {
        int32_t va = a[i], vb = b[j];
        if (va == vb) { c++; i++; j++; }
        else if (va < vb) i++;
        else j++;
    }
    return c;
}

__global__ void compute_scores_multi_kernel(
    const int64_t* __restrict__ unique_keys,
    const int32_t* __restrict__ d_num_unique,
    const int32_t* __restrict__ seeds,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int64_t num_vertices_i64,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int32_t* __restrict__ global_count
) {
    int nu = *d_num_unique;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nu) return;

    int64_t key = unique_keys[idx];
    int32_t sid = (int32_t)(key / num_vertices_i64);
    int32_t v = (int32_t)(key % num_vertices_i64);
    int32_t u = seeds[sid];
    if (v == u) return;

    int32_t us = offsets[u], ue = offsets[u+1];
    int32_t vs = offsets[v], ve = offsets[v+1];
    int isect = merge_intersection(indices+us, ue-us, indices+vs, ve-vs);
    if (isect == 0) return;

    float score = 2.0f * (float)isect / (float)((ue-us)+(ve-vs));
    int32_t pos = atomicAdd(global_count, 1);
    out_first[pos] = u;
    out_second[pos] = v;
    out_scores[pos] = score;
}

__global__ void gather_topk_kernel(
    const int32_t* __restrict__ f, const int32_t* __restrict__ s,
    const float* __restrict__ sc, const int32_t* __restrict__ si,
    int32_t* __restrict__ of, int32_t* __restrict__ os, float* __restrict__ osc,
    int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int32_t x = si[i];
        of[i] = f[x]; os[i] = s[x]; osc[i] = sc[x];
    }
}

__global__ void compute_total_kernel(
    const int64_t* __restrict__ so, const int64_t* __restrict__ sz,
    int32_t n, int64_t* __restrict__ total
) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *total = (n > 0) ? so[n-1] + sz[n-1] : 0;
}



struct Cache : Cacheable {
    
    int64_t* d_total = nullptr;
    int32_t* d_num_runs = nullptr;
    int32_t* gc = nullptr;
    bool scalars_init = false;

    
    int32_t* seeds_buf = nullptr;
    int64_t seeds_buf_cap = 0;

    
    int64_t* work_sizes = nullptr;
    int64_t work_sizes_cap = 0;
    int64_t* work_offsets = nullptr;
    int64_t work_offsets_cap = 0;

    
    int64_t* keys = nullptr;
    int64_t keys_cap = 0;
    int64_t* sorted_keys = nullptr;
    int64_t sorted_keys_cap = 0;
    int64_t* unique_keys = nullptr;
    int64_t unique_keys_cap = 0;

    
    int32_t* rle_counts = nullptr;
    int64_t rle_counts_cap = 0;
    int32_t* scratch_f = nullptr;
    int64_t scratch_f_cap = 0;
    int32_t* scratch_s = nullptr;
    int64_t scratch_s_cap = 0;

    
    float* scratch_sc = nullptr;
    int64_t scratch_sc_cap = 0;

    
    void* scan_buf = nullptr;
    size_t scan_buf_cap = 0;
    void* sort_buf = nullptr;
    size_t sort_buf_cap = 0;
    void* rle_buf = nullptr;
    size_t rle_buf_cap = 0;

    
    int32_t* topk_ia = nullptr;
    int64_t topk_ia_cap = 0;
    float* topk_sk = nullptr;
    int64_t topk_sk_cap = 0;
    int32_t* topk_si = nullptr;
    int64_t topk_si_cap = 0;
    void* topk_sort_buf = nullptr;
    size_t topk_sort_buf_cap = 0;

    void init_scalars() {
        if (!scalars_init) {
            cudaMalloc(&d_total, sizeof(int64_t));
            cudaMalloc(&d_num_runs, sizeof(int32_t));
            cudaMalloc(&gc, sizeof(int32_t));
            scalars_init = true;
        }
    }

    template<typename T>
    static void ensure(T*& ptr, int64_t& cap, int64_t needed) {
        if (cap < needed) {
            if (ptr) cudaFree(ptr);
            cudaMalloc(&ptr, needed * sizeof(T));
            cap = needed;
        }
    }

    static void ensure_bytes(void*& ptr, size_t& cap, size_t needed) {
        if (cap < needed) {
            if (ptr) cudaFree(ptr);
            cudaMalloc(&ptr, needed);
            cap = needed;
        }
    }

    ~Cache() override {
        if (d_total) cudaFree(d_total);
        if (d_num_runs) cudaFree(d_num_runs);
        if (gc) cudaFree(gc);
        if (seeds_buf) cudaFree(seeds_buf);
        if (work_sizes) cudaFree(work_sizes);
        if (work_offsets) cudaFree(work_offsets);
        if (keys) cudaFree(keys);
        if (sorted_keys) cudaFree(sorted_keys);
        if (unique_keys) cudaFree(unique_keys);
        if (rle_counts) cudaFree(rle_counts);
        if (scratch_f) cudaFree(scratch_f);
        if (scratch_s) cudaFree(scratch_s);
        if (scratch_sc) cudaFree(scratch_sc);
        if (scan_buf) cudaFree(scan_buf);
        if (sort_buf) cudaFree(sort_buf);
        if (rle_buf) cudaFree(rle_buf);
        if (topk_ia) cudaFree(topk_ia);
        if (topk_sk) cudaFree(topk_sk);
        if (topk_si) cudaFree(topk_si);
        if (topk_sort_buf) cudaFree(topk_sort_buf);
    }
};

}  

similarity_result_float_t sorensen_all_pairs_similarity(
    const graph32_t& graph,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk) {

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    cache.init_scalars();

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    bool is_multi = graph.is_multigraph;
    int64_t nv64 = (int64_t)nv;

    
    int32_t ns;
    const int32_t* d_seeds;
    if (vertices == nullptr) {
        ns = nv;
        Cache::ensure(cache.seeds_buf, cache.seeds_buf_cap, (int64_t)nv);
        if (nv > 0) iota_kernel<<<(nv + 255) / 256, 256>>>(cache.seeds_buf, nv);
        d_seeds = cache.seeds_buf;
    } else {
        ns = (int32_t)num_vertices;
        d_seeds = vertices;
    }

    if (ns == 0) return {nullptr, nullptr, nullptr, 0};

    
    int64_t max_key = (int64_t)(ns - 1) * nv64 + (nv64 - 1);
    int end_bit = 1;
    { int64_t mk = max_key; while (mk > 1) { mk >>= 1; end_bit++; } }
    if (max_key == 0) end_bit = 1;

    
    Cache::ensure(cache.work_sizes, cache.work_sizes_cap, (int64_t)ns);
    compute_seed_info_kernel<<<ns, 256>>>(d_off, d_idx, d_seeds, ns, cache.work_sizes);

    
    Cache::ensure(cache.work_offsets, cache.work_offsets_cap, (int64_t)ns);
    size_t st = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, st, (int64_t*)nullptr, (int64_t*)nullptr, ns);
    size_t scan_needed = std::max((size_t)(st + 16), (size_t)128);
    Cache::ensure_bytes(cache.scan_buf, cache.scan_buf_cap, scan_needed);
    cub::DeviceScan::ExclusiveSum(cache.scan_buf, st, cache.work_sizes, cache.work_offsets, ns);

    
    compute_total_kernel<<<1, 1>>>(cache.work_offsets, cache.work_sizes, ns, cache.d_total);
    int64_t total_work = 0;
    cudaMemcpy(&total_work, cache.d_total, sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (total_work == 0) return {nullptr, nullptr, nullptr, 0};

    
    Cache::ensure(cache.keys, cache.keys_cap, total_work);
    generate_keys_kernel<<<ns, 256>>>(d_off, d_idx, d_seeds, ns,
        cache.work_offsets, cache.keys, nv64);

    int tw = (int)total_work;

    
    Cache::ensure(cache.sorted_keys, cache.sorted_keys_cap, total_work);
    size_t sort_t = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, sort_t, (int64_t*)nullptr, (int64_t*)nullptr, tw, 0, end_bit);
    size_t sort_needed = std::max((size_t)(sort_t + 16), (size_t)128);
    Cache::ensure_bytes(cache.sort_buf, cache.sort_buf_cap, sort_needed);
    cub::DeviceRadixSort::SortKeys(cache.sort_buf, sort_t,
        cache.keys, cache.sorted_keys, tw, 0, end_bit);

    
    Cache::ensure(cache.unique_keys, cache.unique_keys_cap, total_work);
    Cache::ensure(cache.rle_counts, cache.rle_counts_cap, total_work);
    size_t rle_t = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, rle_t,
        (int64_t*)nullptr, (int64_t*)nullptr, (int32_t*)nullptr, (int32_t*)nullptr, tw);
    size_t rle_needed = std::max((size_t)(rle_t + 16), (size_t)128);
    Cache::ensure_bytes(cache.rle_buf, cache.rle_buf_cap, rle_needed);
    cub::DeviceRunLengthEncode::Encode(cache.rle_buf, rle_t,
        cache.sorted_keys, cache.unique_keys, cache.rle_counts, cache.d_num_runs, tw);

    
    Cache::ensure(cache.scratch_f, cache.scratch_f_cap, total_work);
    Cache::ensure(cache.scratch_s, cache.scratch_s_cap, total_work);
    Cache::ensure(cache.scratch_sc, cache.scratch_sc_cap, total_work);
    cudaMemset(cache.gc, 0, sizeof(int32_t));

    int grid = (tw + 255) / 256;
    if (!is_multi) {
        compute_scores_kernel<<<grid, 256>>>(cache.unique_keys, cache.rle_counts,
            cache.d_num_runs, d_seeds, d_off, nv64,
            cache.scratch_f, cache.scratch_s, cache.scratch_sc, cache.gc);
    } else {
        compute_scores_multi_kernel<<<grid, 256>>>(cache.unique_keys, cache.d_num_runs,
            d_seeds, d_off, d_idx, nv64,
            cache.scratch_f, cache.scratch_s, cache.scratch_sc, cache.gc);
    }

    
    int32_t total_pairs32 = 0;
    cudaMemcpy(&total_pairs32, cache.gc, sizeof(int32_t), cudaMemcpyDeviceToHost);
    int64_t total_pairs = (int64_t)total_pairs32;

    if (total_pairs == 0) return {nullptr, nullptr, nullptr, 0};

    
    bool has_topk = topk.has_value();
    int64_t topk_val = has_topk ? std::min((int64_t)topk.value(), total_pairs) : total_pairs;

    if (has_topk && topk_val < total_pairs) {
        int n = (int)total_pairs;
        Cache::ensure(cache.topk_ia, cache.topk_ia_cap, total_pairs);
        iota_kernel<<<(n + 255) / 256, 256>>>(cache.topk_ia, n);
        Cache::ensure(cache.topk_sk, cache.topk_sk_cap, total_pairs);
        Cache::ensure(cache.topk_si, cache.topk_si_cap, total_pairs);
        size_t srt = 0;
        cub::DeviceRadixSort::SortPairsDescending(nullptr, srt,
            (float*)nullptr, (float*)nullptr, (int32_t*)nullptr, (int32_t*)nullptr, n);
        Cache::ensure_bytes(cache.topk_sort_buf, cache.topk_sort_buf_cap, srt + 16);
        cub::DeviceRadixSort::SortPairsDescending(cache.topk_sort_buf, srt,
            cache.scratch_sc, cache.topk_sk,
            cache.topk_ia, cache.topk_si, n);

        
        int32_t* out_first = nullptr;
        int32_t* out_second = nullptr;
        float* out_scores = nullptr;
        cudaMalloc(&out_first, topk_val * sizeof(int32_t));
        cudaMalloc(&out_second, topk_val * sizeof(int32_t));
        cudaMalloc(&out_scores, topk_val * sizeof(float));
        gather_topk_kernel<<<(int)((topk_val + 255) / 256), 256>>>(
            cache.scratch_f, cache.scratch_s, cache.scratch_sc,
            cache.topk_si, out_first, out_second, out_scores, topk_val);
        return {out_first, out_second, out_scores, (std::size_t)topk_val};
    }

    
    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;
    cudaMalloc(&out_first, total_pairs * sizeof(int32_t));
    cudaMalloc(&out_second, total_pairs * sizeof(int32_t));
    cudaMalloc(&out_scores, total_pairs * sizeof(float));
    cudaMemcpyAsync(out_first, cache.scratch_f, total_pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(out_second, cache.scratch_s, total_pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(out_scores, cache.scratch_sc, total_pairs * sizeof(float), cudaMemcpyDeviceToDevice);
    return {out_first, out_second, out_scores, (std::size_t)total_pairs};
}

}  
