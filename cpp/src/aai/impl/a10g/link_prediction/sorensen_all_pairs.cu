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

struct Cache : Cacheable {
    
    int64_t* d_total = nullptr;
    int64_t* gc = nullptr;
    int32_t* d_num_runs = nullptr;

    
    int32_t* seeds_buf = nullptr;
    int64_t seeds_cap = 0;

    
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
    int32_t* out_f = nullptr;
    int64_t out_f_cap = 0;
    int32_t* out_s = nullptr;
    int64_t out_s_cap = 0;
    float* out_sc = nullptr;
    int64_t out_sc_cap = 0;

    
    void* cub_temp = nullptr;
    size_t cub_temp_cap = 0;

    
    int32_t* topk_ia = nullptr;
    int64_t topk_ia_cap = 0;
    float* topk_sk = nullptr;
    int64_t topk_sk_cap = 0;
    int32_t* topk_si = nullptr;
    int64_t topk_si_cap = 0;

    Cache() {
        cudaMalloc(&d_total, sizeof(int64_t));
        cudaMalloc(&gc, sizeof(int64_t));
        cudaMalloc(&d_num_runs, sizeof(int32_t));
    }

    void ensure_seeds(int64_t n) {
        if (seeds_cap < n) {
            if (seeds_buf) cudaFree(seeds_buf);
            cudaMalloc(&seeds_buf, n * sizeof(int32_t));
            seeds_cap = n;
        }
    }

    void ensure_work_sizes(int64_t n) {
        if (work_sizes_cap < n) {
            if (work_sizes) cudaFree(work_sizes);
            cudaMalloc(&work_sizes, n * sizeof(int64_t));
            work_sizes_cap = n;
        }
    }

    void ensure_work_offsets(int64_t n) {
        if (work_offsets_cap < n) {
            if (work_offsets) cudaFree(work_offsets);
            cudaMalloc(&work_offsets, n * sizeof(int64_t));
            work_offsets_cap = n;
        }
    }

    void ensure_keys(int64_t n) {
        if (keys_cap < n) {
            if (keys) cudaFree(keys);
            cudaMalloc(&keys, n * sizeof(int64_t));
            keys_cap = n;
        }
    }

    void ensure_sorted_keys(int64_t n) {
        if (sorted_keys_cap < n) {
            if (sorted_keys) cudaFree(sorted_keys);
            cudaMalloc(&sorted_keys, n * sizeof(int64_t));
            sorted_keys_cap = n;
        }
    }

    void ensure_unique_keys(int64_t n) {
        if (unique_keys_cap < n) {
            if (unique_keys) cudaFree(unique_keys);
            cudaMalloc(&unique_keys, n * sizeof(int64_t));
            unique_keys_cap = n;
        }
    }

    void ensure_rle_counts(int64_t n) {
        if (rle_counts_cap < n) {
            if (rle_counts) cudaFree(rle_counts);
            cudaMalloc(&rle_counts, n * sizeof(int32_t));
            rle_counts_cap = n;
        }
    }

    void ensure_out_f(int64_t n) {
        if (out_f_cap < n) {
            if (out_f) cudaFree(out_f);
            cudaMalloc(&out_f, n * sizeof(int32_t));
            out_f_cap = n;
        }
    }

    void ensure_out_s(int64_t n) {
        if (out_s_cap < n) {
            if (out_s) cudaFree(out_s);
            cudaMalloc(&out_s, n * sizeof(int32_t));
            out_s_cap = n;
        }
    }

    void ensure_out_sc(int64_t n) {
        if (out_sc_cap < n) {
            if (out_sc) cudaFree(out_sc);
            cudaMalloc(&out_sc, n * sizeof(float));
            out_sc_cap = n;
        }
    }

    void ensure_cub_temp(size_t n) {
        if (cub_temp_cap < n) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, n);
            cub_temp_cap = n;
        }
    }

    void ensure_topk_ia(int64_t n) {
        if (topk_ia_cap < n) {
            if (topk_ia) cudaFree(topk_ia);
            cudaMalloc(&topk_ia, n * sizeof(int32_t));
            topk_ia_cap = n;
        }
    }

    void ensure_topk_sk(int64_t n) {
        if (topk_sk_cap < n) {
            if (topk_sk) cudaFree(topk_sk);
            cudaMalloc(&topk_sk, n * sizeof(float));
            topk_sk_cap = n;
        }
    }

    void ensure_topk_si(int64_t n) {
        if (topk_si_cap < n) {
            if (topk_si) cudaFree(topk_si);
            cudaMalloc(&topk_si, n * sizeof(int32_t));
            topk_si_cap = n;
        }
    }

    ~Cache() override {
        if (d_total) cudaFree(d_total);
        if (gc) cudaFree(gc);
        if (d_num_runs) cudaFree(d_num_runs);
        if (seeds_buf) cudaFree(seeds_buf);
        if (work_sizes) cudaFree(work_sizes);
        if (work_offsets) cudaFree(work_offsets);
        if (keys) cudaFree(keys);
        if (sorted_keys) cudaFree(sorted_keys);
        if (unique_keys) cudaFree(unique_keys);
        if (rle_counts) cudaFree(rle_counts);
        if (out_f) cudaFree(out_f);
        if (out_s) cudaFree(out_s);
        if (out_sc) cudaFree(out_sc);
        if (cub_temp) cudaFree(cub_temp);
        if (topk_ia) cudaFree(topk_ia);
        if (topk_sk) cudaFree(topk_sk);
        if (topk_si) cudaFree(topk_si);
    }
};



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

    
    int32_t u = __ldg(seeds + sid);
    int32_t us = __ldg(offsets + u);
    int32_t ue = __ldg(offsets + u + 1);

    int64_t w = 0;

    
    int i = us + threadIdx.x;
    int stride = blockDim.x;

    
    for (; i + 3 * stride < ue; i += 4 * stride) {
        int32_t nb0 = __ldg(indices + i);
        int32_t nb1 = __ldg(indices + i + stride);
        int32_t nb2 = __ldg(indices + i + 2 * stride);
        int32_t nb3 = __ldg(indices + i + 3 * stride);

        w += (int64_t)(__ldg(offsets + nb0 + 1) - __ldg(offsets + nb0));
        w += (int64_t)(__ldg(offsets + nb1 + 1) - __ldg(offsets + nb1));
        w += (int64_t)(__ldg(offsets + nb2 + 1) - __ldg(offsets + nb2));
        w += (int64_t)(__ldg(offsets + nb3 + 1) - __ldg(offsets + nb3));
    }

    
    for (; i < ue; i += stride) {
        int32_t nb = __ldg(indices + i);
        w += (int64_t)(__ldg(offsets + nb + 1) - __ldg(offsets + nb));
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

    int32_t u = __ldg(seeds + sid);
    int32_t us = __ldg(offsets + u);
    int32_t ue = __ldg(offsets + u + 1);
    int64_t base = __ldg(work_offsets + sid);
    int64_t prefix = (int64_t)sid * num_vertices_i64;

    int64_t local_off = 0;

    
    for (int i = us; i < ue; i++) {
        int32_t w = __ldg(indices + i);
        int32_t ws = __ldg(offsets + w);
        int32_t wdeg = __ldg(offsets + w + 1) - ws;

        
        int j = threadIdx.x;
        int stride = blockDim.x;

        
        for (; j + 3 * stride < wdeg; j += 4 * stride) {
            int32_t v0 = __ldg(indices + ws + j);
            int32_t v1 = __ldg(indices + ws + j + stride);
            int32_t v2 = __ldg(indices + ws + j + 2 * stride);
            int32_t v3 = __ldg(indices + ws + j + 3 * stride);

            keys[base + local_off + j] = prefix + (int64_t)v0;
            keys[base + local_off + j + stride] = prefix + (int64_t)v1;
            keys[base + local_off + j + 2 * stride] = prefix + (int64_t)v2;
            keys[base + local_off + j + 3 * stride] = prefix + (int64_t)v3;
        }

        
        for (; j < wdeg; j += stride) {
            int32_t v = __ldg(indices + ws + j);
            keys[base + local_off + j] = prefix + (int64_t)v;
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
    int64_t* __restrict__ global_count
) {
    int nu = *d_num_unique;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nu) return;

    int64_t key = __ldg(unique_keys + idx);
    int32_t sid = (int32_t)(key / num_vertices_i64);
    int32_t v = (int32_t)(key % num_vertices_i64);
    int32_t u = __ldg(seeds + sid);
    if (v == u) return;

    int32_t cnt = __ldg(rle_counts + idx);
    int32_t du = __ldg(offsets + u + 1) - __ldg(offsets + u);
    int32_t dv = __ldg(offsets + v + 1) - __ldg(offsets + v);

    
    
    float rcp = __frcp_rn((float)(du + dv));
    float score = __fmaf_rn(2.0f * (float)cnt, rcp, 0.0f);

    int64_t pos = (int64_t)atomicAdd((unsigned long long*)global_count, 1ULL);
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
    int64_t* __restrict__ global_count
) {
    int nu = *d_num_unique;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nu) return;

    int64_t key = __ldg(unique_keys + idx);
    int32_t sid = (int32_t)(key / num_vertices_i64);
    int32_t v = (int32_t)(key % num_vertices_i64);
    int32_t u = __ldg(seeds + sid);
    if (v == u) return;

    int32_t us = __ldg(offsets + u);
    int32_t ue = __ldg(offsets + u + 1);
    int32_t vs = __ldg(offsets + v);
    int32_t ve = __ldg(offsets + v + 1);
    int isect = merge_intersection(indices+us, ue-us, indices+vs, ve-vs);
    if (isect == 0) return;

    float score = 2.0f * (float)isect / (float)((ue-us)+(ve-vs));
    int64_t pos = (int64_t)atomicAdd((unsigned long long*)global_count, 1ULL);
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
        int32_t x = __ldg(si + i);
        of[i] = __ldg(f + x);
        os[i] = __ldg(s + x);
        osc[i] = __ldg(sc + x);
    }
}

__global__ void compute_total_kernel(
    const int64_t* __restrict__ so, const int64_t* __restrict__ sz,
    int32_t n, int64_t* __restrict__ total
) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        *total = (n > 0) ? __ldg(so + n - 1) + __ldg(sz + n - 1) : 0;
}

}  

similarity_result_float_t sorensen_all_pairs_similarity(const graph32_t& graph,
                                                        const int32_t* vertices,
                                                        std::size_t num_vertices,
                                                        std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    bool is_multi = graph.is_multigraph;
    int64_t nv64 = (int64_t)nv;

    
    int32_t ns;
    const int32_t* d_seeds;
    if (vertices == nullptr) {
        ns = nv;
        cache.ensure_seeds(nv);
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

    
    cache.ensure_work_sizes(ns);
    if (ns > 0) compute_seed_info_kernel<<<ns, 256>>>(d_off, d_idx, d_seeds, ns, cache.work_sizes);

    
    cache.ensure_work_offsets(ns);
    size_t st = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, st, (int64_t*)nullptr, (int64_t*)nullptr, ns);
    cache.ensure_cub_temp(st);
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, st,
        cache.work_sizes, cache.work_offsets, ns);

    
    compute_total_kernel<<<1, 1>>>(cache.work_offsets, cache.work_sizes, ns, cache.d_total);
    int64_t total_work = 0;
    cudaMemcpy(&total_work, cache.d_total, sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (total_work == 0) return {nullptr, nullptr, nullptr, 0};

    
    cache.ensure_keys(total_work);
    if (ns > 0) generate_keys_kernel<<<ns, 256>>>(d_off, d_idx, d_seeds, ns,
        cache.work_offsets, cache.keys, nv64);

    
    cache.ensure_sorted_keys(total_work);
    size_t sort_t = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, sort_t, (int64_t*)nullptr, (int64_t*)nullptr,
        total_work, 0, end_bit);
    cache.ensure_cub_temp(sort_t);
    cub::DeviceRadixSort::SortKeys(cache.cub_temp, sort_t,
        cache.keys, cache.sorted_keys, total_work, 0, end_bit);

    
    cache.ensure_unique_keys(total_work);
    cache.ensure_rle_counts(total_work);
    size_t rle_t = 0;
    cub::DeviceRunLengthEncode::Encode(nullptr, rle_t,
        (int64_t*)nullptr, (int64_t*)nullptr, (int32_t*)nullptr, (int32_t*)nullptr, total_work);
    cache.ensure_cub_temp(rle_t);
    cub::DeviceRunLengthEncode::Encode(cache.cub_temp, rle_t,
        cache.sorted_keys, cache.unique_keys,
        cache.rle_counts, cache.d_num_runs, total_work);

    
    cache.ensure_out_f(total_work);
    cache.ensure_out_s(total_work);
    cache.ensure_out_sc(total_work);
    cudaMemset(cache.gc, 0, sizeof(int64_t));

    int grid = (int)std::min(total_work, (int64_t)((total_work + 255) / 256));
    if (grid <= 0) grid = 1;
    if (!is_multi) {
        compute_scores_kernel<<<grid, 256>>>(cache.unique_keys,
            cache.rle_counts, cache.d_num_runs,
            d_seeds, d_off, nv64,
            cache.out_f, cache.out_s,
            cache.out_sc, cache.gc);
    } else {
        compute_scores_multi_kernel<<<grid, 256>>>(cache.unique_keys,
            cache.d_num_runs,
            d_seeds, d_off, d_idx, nv64,
            cache.out_f, cache.out_s,
            cache.out_sc, cache.gc);
    }

    
    int64_t total_pairs = 0;
    cudaMemcpy(&total_pairs, cache.gc, sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (total_pairs == 0) return {nullptr, nullptr, nullptr, 0};

    
    bool has_topk = topk.has_value();
    int64_t topk_val = has_topk ? std::min((int64_t)topk.value(), total_pairs) : total_pairs;

    if (has_topk && topk_val < total_pairs) {
        int n = (int)total_pairs;

        
        cache.ensure_topk_ia(total_pairs);
        iota_kernel<<<(n + 255) / 256, 256>>>(cache.topk_ia, n);

        
        cache.ensure_topk_sk(total_pairs);
        cache.ensure_topk_si(total_pairs);
        size_t srt = 0;
        cub::DeviceRadixSort::SortPairsDescending(nullptr, srt,
            (float*)nullptr, (float*)nullptr, (int32_t*)nullptr, (int32_t*)nullptr, n);
        cache.ensure_cub_temp(srt);
        cub::DeviceRadixSort::SortPairsDescending(cache.cub_temp, srt,
            cache.out_sc, cache.topk_sk,
            cache.topk_ia, cache.topk_si, n);

        
        int32_t* result_first = nullptr;
        int32_t* result_second = nullptr;
        float* result_scores = nullptr;
        cudaMalloc(&result_first, topk_val * sizeof(int32_t));
        cudaMalloc(&result_second, topk_val * sizeof(int32_t));
        cudaMalloc(&result_scores, topk_val * sizeof(float));

        gather_topk_kernel<<<(int)((topk_val + 255) / 256), 256>>>(
            cache.out_f, cache.out_s, cache.out_sc, cache.topk_si,
            result_first, result_second, result_scores, topk_val);

        return {result_first, result_second, result_scores, (std::size_t)topk_val};
    }

    
    int32_t* result_first = nullptr;
    int32_t* result_second = nullptr;
    float* result_scores = nullptr;
    cudaMalloc(&result_first, total_pairs * sizeof(int32_t));
    cudaMalloc(&result_second, total_pairs * sizeof(int32_t));
    cudaMalloc(&result_scores, total_pairs * sizeof(float));
    cudaMemcpyAsync(result_first, cache.out_f, total_pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(result_second, cache.out_s, total_pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(result_scores, cache.out_sc, total_pairs * sizeof(float), cudaMemcpyDeviceToDevice);

    return {result_first, result_second, result_scores, (std::size_t)total_pairs};
}

}  
