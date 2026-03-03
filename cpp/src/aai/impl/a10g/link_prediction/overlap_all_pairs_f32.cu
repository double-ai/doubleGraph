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
    void* scratch = nullptr;
    size_t scratch_size = 0;
    int32_t* num_unique_d = nullptr;

    Cache() {
        cudaMalloc(&num_unique_d, sizeof(int32_t));
    }

    ~Cache() override {
        if (scratch) cudaFree(scratch);
        if (num_unique_d) cudaFree(num_unique_d);
    }

    void ensure_scratch(size_t needed) {
        if (needed <= scratch_size) return;
        if (scratch) cudaFree(scratch);
        scratch_size = needed * 2;
        cudaMalloc(&scratch, scratch_size);
    }
};



__global__ void k_weighted_degrees(const int32_t* __restrict__ off,
                                     const float* __restrict__ w,
                                     int32_t nv, float* __restrict__ wd) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv) return;
    float s = 0.f;
    for (int i = off[v]; i < off[v+1]; i++) s += w[i];
    wd[v] = s;
}

__global__ void k_seed_degrees(const int32_t* __restrict__ off,
                                const int32_t* __restrict__ seeds,
                                int32_t ns, int64_t* __restrict__ deg) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ns) return;
    int32_t u = seeds[i];
    deg[i] = (int64_t)(off[u + 1] - off[u]);
}

__global__ void k_count_pairs_edge(const int32_t* __restrict__ off,
                                     const int32_t* __restrict__ idx,
                                     const int32_t* __restrict__ seeds,
                                     int32_t ns,
                                     const int64_t* __restrict__ seed_edge_off,
                                     int64_t total_edges,
                                     int64_t* __restrict__ cnt) {
    int64_t eid = blockIdx.x;
    if (eid >= total_edges) return;

    int lo = 0, hi = ns;
    while (lo < hi) { int m = (lo+hi)/2; if(seed_edge_off[m+1] <= eid) lo=m+1; else hi=m; }
    int sid = lo;

    int32_t u = seeds[sid];
    int edge_within = (int)(eid - seed_edge_off[sid]);
    int32_t v = idx[off[u] + edge_within];

    int32_t vs = off[v], ve = off[v+1];
    int64_t c = 0;
    for (int j = vs + (int)threadIdx.x; j < ve; j += (int)blockDim.x) {
        if (idx[j] != u) c++;
    }

    typedef cub::BlockReduce<int64_t, 128> BR;
    __shared__ typename BR::TempStorage ts;
    int64_t total = BR(ts).Sum(c);
    if (threadIdx.x == 0) cnt[eid] = total;
}

__global__ void k_write_pairs_edge_32(const int32_t* __restrict__ off,
                                        const int32_t* __restrict__ idx,
                                        const int32_t* __restrict__ seeds,
                                        int32_t ns,
                                        const int64_t* __restrict__ seed_edge_off,
                                        int64_t total_edges,
                                        const int64_t* __restrict__ pair_off,
                                        uint32_t* __restrict__ keys,
                                        int32_t nv) {
    int64_t eid = blockIdx.x;
    if (eid >= total_edges) return;

    int lo = 0, hi = ns;
    while (lo < hi) { int m = (lo+hi)/2; if(seed_edge_off[m+1] <= eid) lo=m+1; else hi=m; }
    int sid = lo;

    int32_t u = seeds[sid];
    int edge_within = (int)(eid - seed_edge_off[sid]);
    int32_t v = idx[off[u] + edge_within];

    int32_t vs = off[v], ve = off[v+1];
    int64_t base = pair_off[eid];
    __shared__ int wpos;
    if (threadIdx.x == 0) wpos = 0;
    __syncthreads();

    for (int j = vs + (int)threadIdx.x; j < ve; j += (int)blockDim.x) {
        int32_t w = idx[j];
        if (w != u) {
            int p = atomicAdd(&wpos, 1);
            keys[base + p] = (uint32_t)u * (uint32_t)nv + (uint32_t)w;
        }
    }
}

__global__ void k_write_pairs_edge_64(const int32_t* __restrict__ off,
                                        const int32_t* __restrict__ idx,
                                        const int32_t* __restrict__ seeds,
                                        int32_t ns,
                                        const int64_t* __restrict__ seed_edge_off,
                                        int64_t total_edges,
                                        const int64_t* __restrict__ pair_off,
                                        int64_t* __restrict__ keys,
                                        int32_t nv) {
    int64_t eid = blockIdx.x;
    if (eid >= total_edges) return;

    int lo = 0, hi = ns;
    while (lo < hi) { int m = (lo+hi)/2; if(seed_edge_off[m+1] <= eid) lo=m+1; else hi=m; }
    int sid = lo;

    int32_t u = seeds[sid];
    int edge_within = (int)(eid - seed_edge_off[sid]);
    int32_t v = idx[off[u] + edge_within];

    int32_t vs = off[v], ve = off[v+1];
    int64_t base = pair_off[eid];
    __shared__ int wpos;
    if (threadIdx.x == 0) wpos = 0;
    __syncthreads();

    for (int j = vs + (int)threadIdx.x; j < ve; j += (int)blockDim.x) {
        int32_t w = idx[j];
        if (w != u) {
            int p = atomicAdd(&wpos, 1);
            keys[base + p] = (int64_t)u * nv + w;
        }
    }
}

__global__ void k_decode_unique_32(const uint32_t* __restrict__ unique_keys,
                                     int64_t n,
                                     int32_t* __restrict__ f,
                                     int32_t* __restrict__ s,
                                     int32_t nv) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t key = unique_keys[i];
    f[i] = (int32_t)(key / (uint32_t)nv);
    s[i] = (int32_t)(key % (uint32_t)nv);
}

__global__ void k_decode_unique_64(const int64_t* __restrict__ unique_keys,
                                     int64_t n,
                                     int32_t* __restrict__ f,
                                     int32_t* __restrict__ s,
                                     int32_t nv) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t key = unique_keys[i];
    f[i] = (int32_t)(key / nv);
    s[i] = (int32_t)(key % nv);
}

__global__ void k_overlap(const int32_t* __restrict__ off,
                           const int32_t* __restrict__ idx,
                           const float* __restrict__ w,
                           const float* __restrict__ wd,
                           const int32_t* __restrict__ f,
                           const int32_t* __restrict__ s,
                           int64_t np, float* __restrict__ sc) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (wid >= (int)np) return;

    int32_t u = f[wid], v = s[wid];
    int32_t us = off[u], ue = off[u+1], ud = ue - us;
    int32_t vs = off[v], ve = off[v+1], vd = ve - vs;

    const int32_t* si; const float* sw_p; int32_t sl;
    const int32_t* li; const float* lw; int32_t ll;
    if (ud <= vd) {
        si = idx + us; sw_p = w + us; sl = ud;
        li = idx + vs; lw = w + vs; ll = vd;
    } else {
        si = idx + vs; sw_p = w + vs; sl = vd;
        li = idx + us; lw = w + us; ll = ud;
    }

    float wi = 0.f;
    for (int i = lane; i < sl; i += 32) {
        int32_t t = si[i]; float ws = sw_p[i];
        int lo = 0, hi = ll;
        while (lo < hi) { int m = (lo+hi)>>1; if(li[m]<t) lo=m+1; else hi=m; }
        if (lo < ll && li[lo] == t) wi += fminf(ws, lw[lo]);
    }

    for (int o = 16; o > 0; o >>= 1)
        wi += __shfl_down_sync(0xffffffff, wi, o);

    if (lane == 0) {
        float d = fminf(wd[u], wd[v]);
        sc[wid] = (d > 0.f) ? wi / d : 0.f;
    }
}

__global__ void k_overlap_lazy(const int32_t* __restrict__ off,
                                const int32_t* __restrict__ idx,
                                const float* __restrict__ w,
                                const int32_t* __restrict__ f,
                                const int32_t* __restrict__ s,
                                int64_t np, float* __restrict__ sc) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (wid >= (int)np) return;

    int32_t u = f[wid], v = s[wid];
    int32_t us = off[u], ue = off[u+1], ud = ue - us;
    int32_t vs = off[v], ve = off[v+1], vd = ve - vs;

    float wd_u = 0.f;
    for (int i = us + lane; i < ue; i += 32) wd_u += w[i];
    for (int o = 16; o > 0; o >>= 1) wd_u += __shfl_down_sync(0xffffffff, wd_u, o);
    wd_u = __shfl_sync(0xffffffff, wd_u, 0);

    float wd_v = 0.f;
    for (int i = vs + lane; i < ve; i += 32) wd_v += w[i];
    for (int o = 16; o > 0; o >>= 1) wd_v += __shfl_down_sync(0xffffffff, wd_v, o);
    wd_v = __shfl_sync(0xffffffff, wd_v, 0);

    const int32_t* si; const float* sw_p; int32_t sl;
    const int32_t* li; const float* lw; int32_t ll;
    if (ud <= vd) {
        si = idx + us; sw_p = w + us; sl = ud;
        li = idx + vs; lw = w + vs; ll = vd;
    } else {
        si = idx + vs; sw_p = w + vs; sl = vd;
        li = idx + us; lw = w + us; ll = ud;
    }

    float wi = 0.f;
    for (int i = lane; i < sl; i += 32) {
        int32_t t = si[i]; float ws = sw_p[i];
        int lo = 0, hi = ll;
        while (lo < hi) { int m = (lo+hi)>>1; if(li[m]<t) lo=m+1; else hi=m; }
        if (lo < ll && li[lo] == t) wi += fminf(ws, lw[lo]);
    }

    for (int o = 16; o > 0; o >>= 1)
        wi += __shfl_down_sync(0xffffffff, wi, o);

    if (lane == 0) {
        float d = fminf(wd_u, wd_v);
        sc[wid] = (d > 0.f) ? wi / d : 0.f;
    }
}

__global__ void k_iota(int32_t* out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (int32_t)i;
}

__global__ void k_gather_i32(const int32_t* __restrict__ src,
                              const int32_t* __restrict__ idx_arr,
                              int32_t* __restrict__ dst, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx_arr[i]];
}

__global__ void k_gather_f32(const float* __restrict__ src,
                              const int32_t* __restrict__ idx_arr,
                              float* __restrict__ dst, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx_arr[i]];
}



static void prefix_sum_i64(Cache& cache, const int64_t* in, int64_t* out, int n) {
    if (n == 0) return;
    size_t ts = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, ts, in, out, n);
    cache.ensure_scratch(ts);
    cub::DeviceScan::ExclusiveSum(cache.scratch, ts, in, out, n);
}

static int64_t sort_dedup_32(Cache& cache, uint32_t* keys_in, int64_t n,
                              uint32_t* keys_sorted, uint32_t* unique_out, int end_bit) {
    if (n == 0) return 0;
    size_t ts = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, ts, keys_in, keys_sorted, (int)n, 0, end_bit);
    cache.ensure_scratch(ts);
    cub::DeviceRadixSort::SortKeys(cache.scratch, ts, keys_in, keys_sorted, (int)n, 0, end_bit);
    ts = 0;
    cub::DeviceSelect::Unique(nullptr, ts, keys_sorted, unique_out, cache.num_unique_d, (int)n);
    cache.ensure_scratch(ts);
    cub::DeviceSelect::Unique(cache.scratch, ts, keys_sorted, unique_out, cache.num_unique_d, (int)n);
    int32_t num_unique = 0;
    cudaMemcpy(&num_unique, cache.num_unique_d, sizeof(int32_t), cudaMemcpyDeviceToHost);
    return (int64_t)num_unique;
}

static int64_t sort_dedup_64(Cache& cache, int64_t* keys_in, int64_t n,
                              int64_t* keys_sorted, int64_t* unique_out, int end_bit) {
    if (n == 0) return 0;
    size_t ts = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, ts, keys_in, keys_sorted, (int)n, 0, end_bit);
    cache.ensure_scratch(ts);
    cub::DeviceRadixSort::SortKeys(cache.scratch, ts, keys_in, keys_sorted, (int)n, 0, end_bit);
    ts = 0;
    cub::DeviceSelect::Unique(nullptr, ts, keys_sorted, unique_out, cache.num_unique_d, (int)n);
    cache.ensure_scratch(ts);
    cub::DeviceSelect::Unique(cache.scratch, ts, keys_sorted, unique_out, cache.num_unique_d, (int)n);
    int32_t num_unique = 0;
    cudaMemcpy(&num_unique, cache.num_unique_d, sizeof(int32_t), cudaMemcpyDeviceToHost);
    return (int64_t)num_unique;
}

static void sort_desc_float_int(Cache& cache, const float* kin, float* kout,
                                 const int32_t* vin, int32_t* vout, int n) {
    if (n == 0) return;
    size_t ts = 0;
    cub::DeviceRadixSort::SortPairsDescending(nullptr, ts, kin, kout, vin, vout, n);
    cache.ensure_scratch(ts);
    cub::DeviceRadixSort::SortPairsDescending(cache.scratch, ts, kin, kout, vin, vout, n);
}

}  

similarity_result_float_t overlap_all_pairs_similarity(const graph32_t& graph,
                                                       const float* edge_weights,
                                                       const int32_t* vertices,
                                                       std::size_t num_vertices,
                                                       std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    const float* d_w = edge_weights;

    bool is_subset = (vertices != nullptr);
    bool use_32bit = ((int64_t)nv * nv <= (int64_t)UINT32_MAX);

    
    int32_t* seeds_buf = nullptr;
    float* wd = nullptr;
    int64_t* seed_deg = nullptr;
    int64_t* seed_edge_off = nullptr;
    int64_t* pair_cnt = nullptr;
    int64_t* pair_off = nullptr;

    auto cleanup = [&]() {
        if (seeds_buf) cudaFree(seeds_buf);
        if (wd) cudaFree(wd);
        if (seed_deg) cudaFree(seed_deg);
        if (seed_edge_off) cudaFree(seed_edge_off);
        if (pair_cnt) cudaFree(pair_cnt);
        if (pair_off) cudaFree(pair_off);
    };

    similarity_result_float_t empty_result = {nullptr, nullptr, nullptr, 0};

    
    int32_t ns;
    const int32_t* d_seeds;
    if (is_subset) {
        ns = (int32_t)num_vertices;
        d_seeds = vertices;
    } else {
        ns = nv;
        cudaMalloc(&seeds_buf, (size_t)nv * sizeof(int32_t));
        if (nv > 0) {
            int B = 256, G = (nv + B - 1) / B;
            k_iota<<<G, B>>>(seeds_buf, (int64_t)nv);
        }
        d_seeds = seeds_buf;
    }

    if (ns == 0) { cleanup(); return empty_result; }

    
    if (!is_subset) {
        cudaMalloc(&wd, (size_t)nv * sizeof(float));
        int B = 256, G = (nv + B - 1) / B;
        k_weighted_degrees<<<G, B>>>(d_off, d_w, nv, wd);
    }

    
    cudaMalloc(&seed_deg, (size_t)ns * sizeof(int64_t));
    {
        int B = 256, G = (ns + B - 1) / B;
        k_seed_degrees<<<G, B>>>(d_off, d_seeds, ns, seed_deg);
    }

    cudaMalloc(&seed_edge_off, ((size_t)ns + 1) * sizeof(int64_t));
    prefix_sum_i64(cache, seed_deg, seed_edge_off, ns);

    
    int64_t last_deg = 0, last_off = 0;
    cudaMemcpy(&last_deg, seed_deg + ns - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_off, seed_edge_off + ns - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
    int64_t total_edges = last_off + last_deg;
    
    cudaMemcpy(seed_edge_off + ns, &total_edges, sizeof(int64_t), cudaMemcpyHostToDevice);

    cudaFree(seed_deg); seed_deg = nullptr;

    if (total_edges == 0) { cleanup(); return empty_result; }

    
    cudaMalloc(&pair_cnt, (size_t)total_edges * sizeof(int64_t));
    k_count_pairs_edge<<<(int)total_edges, 128>>>(d_off, d_idx, d_seeds, ns,
                                                    seed_edge_off, total_edges, pair_cnt);

    
    cudaMalloc(&pair_off, (size_t)total_edges * sizeof(int64_t));
    prefix_sum_i64(cache, pair_cnt, pair_off, (int)total_edges);

    
    int64_t last_pair_cnt = 0, last_pair_off = 0;
    cudaMemcpy(&last_pair_cnt, pair_cnt + total_edges - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_pair_off, pair_off + total_edges - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
    int64_t total_pairs = last_pair_off + last_pair_cnt;

    cudaFree(pair_cnt); pair_cnt = nullptr;

    if (total_pairs <= 0) { cleanup(); return empty_result; }

    
    int end_bit = 0;
    {
        int64_t mx = (int64_t)(nv - 1) * nv + (nv - 1);
        while ((1LL << end_bit) <= mx && end_bit < 64) end_bit++;
    }

    int64_t num_unique = 0;
    int32_t* first = nullptr;
    int32_t* second = nullptr;

    if (use_32bit) {
        uint32_t* keys = nullptr;
        uint32_t* keys_sorted = nullptr;
        uint32_t* unique_keys = nullptr;

        cudaMalloc(&keys, (size_t)total_pairs * sizeof(uint32_t));
        k_write_pairs_edge_32<<<(int)total_edges, 128>>>(d_off, d_idx, d_seeds, ns,
                                                           seed_edge_off, total_edges,
                                                           pair_off, keys, nv);

        cudaMalloc(&keys_sorted, (size_t)total_pairs * sizeof(uint32_t));
        cudaMalloc(&unique_keys, (size_t)total_pairs * sizeof(uint32_t));

        num_unique = sort_dedup_32(cache, keys, total_pairs, keys_sorted, unique_keys, end_bit);

        cudaFree(keys);
        cudaFree(keys_sorted);

        if (num_unique == 0) {
            cudaFree(unique_keys);
            cleanup();
            return empty_result;
        }

        cudaMalloc(&first, (size_t)num_unique * sizeof(int32_t));
        cudaMalloc(&second, (size_t)num_unique * sizeof(int32_t));
        {
            int B = 256, G = (int)((num_unique + B - 1) / B);
            k_decode_unique_32<<<G, B>>>(unique_keys, num_unique, first, second, nv);
        }

        cudaFree(unique_keys);
    } else {
        int64_t* keys = nullptr;
        int64_t* keys_sorted = nullptr;
        int64_t* unique_keys = nullptr;

        cudaMalloc(&keys, (size_t)total_pairs * sizeof(int64_t));
        k_write_pairs_edge_64<<<(int)total_edges, 128>>>(d_off, d_idx, d_seeds, ns,
                                                           seed_edge_off, total_edges,
                                                           pair_off, keys, nv);

        cudaMalloc(&keys_sorted, (size_t)total_pairs * sizeof(int64_t));
        cudaMalloc(&unique_keys, (size_t)total_pairs * sizeof(int64_t));

        num_unique = sort_dedup_64(cache, keys, total_pairs, keys_sorted, unique_keys, end_bit);

        cudaFree(keys);
        cudaFree(keys_sorted);

        if (num_unique == 0) {
            cudaFree(unique_keys);
            cleanup();
            return empty_result;
        }

        cudaMalloc(&first, (size_t)num_unique * sizeof(int32_t));
        cudaMalloc(&second, (size_t)num_unique * sizeof(int32_t));
        {
            int B = 256, G = (int)((num_unique + B - 1) / B);
            k_decode_unique_64<<<G, B>>>(unique_keys, num_unique, first, second, nv);
        }

        cudaFree(unique_keys);
    }

    
    cudaFree(seed_edge_off); seed_edge_off = nullptr;
    cudaFree(pair_off); pair_off = nullptr;

    
    float* scores = nullptr;
    cudaMalloc(&scores, (size_t)num_unique * sizeof(float));
    if (num_unique > 0) {
        int wpb = 8, B = wpb * 32;
        int G = ((int)num_unique + wpb - 1) / wpb;
        if (is_subset) {
            k_overlap_lazy<<<G, B>>>(d_off, d_idx, d_w, first, second, num_unique, scores);
        } else {
            k_overlap<<<G, B>>>(d_off, d_idx, d_w, wd, first, second, num_unique, scores);
        }
    }

    
    if (seeds_buf) { cudaFree(seeds_buf); seeds_buf = nullptr; }
    if (wd) { cudaFree(wd); wd = nullptr; }

    
    int64_t result_count = num_unique;
    if (topk.has_value() && (int64_t)topk.value() < num_unique) {
        result_count = (int64_t)topk.value();

        int32_t* idx_buf = nullptr;
        float* scores_sorted = nullptr;
        int32_t* idx_sorted = nullptr;

        cudaMalloc(&idx_buf, (size_t)num_unique * sizeof(int32_t));
        {
            int B = 256, G = ((int)num_unique + B - 1) / B;
            k_iota<<<G, B>>>(idx_buf, num_unique);
        }

        cudaMalloc(&scores_sorted, (size_t)num_unique * sizeof(float));
        cudaMalloc(&idx_sorted, (size_t)num_unique * sizeof(int32_t));
        sort_desc_float_int(cache, scores, scores_sorted, idx_buf, idx_sorted, (int)num_unique);

        int32_t* tk_first = nullptr;
        int32_t* tk_second = nullptr;
        float* tk_scores = nullptr;

        cudaMalloc(&tk_first, (size_t)result_count * sizeof(int32_t));
        cudaMalloc(&tk_second, (size_t)result_count * sizeof(int32_t));
        cudaMalloc(&tk_scores, (size_t)result_count * sizeof(float));

        if (result_count > 0) {
            int B = 256, G = ((int)result_count + B - 1) / B;
            k_gather_i32<<<G, B>>>(first, idx_sorted, tk_first, result_count);
            k_gather_i32<<<G, B>>>(second, idx_sorted, tk_second, result_count);
            k_gather_f32<<<G, B>>>(scores, idx_sorted, tk_scores, result_count);
        }

        cudaFree(first);
        cudaFree(second);
        cudaFree(scores);
        cudaFree(idx_buf);
        cudaFree(scores_sorted);
        cudaFree(idx_sorted);

        first = tk_first;
        second = tk_second;
        scores = tk_scores;
    }

    return {first, second, scores, (std::size_t)result_count};
}

}  
