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

    void ensure_scratch(size_t needed) {
        if (needed <= scratch_size) return;
        if (scratch) cudaFree(scratch);
        scratch_size = needed * 2;
        cudaMalloc(&scratch, scratch_size);
    }

    ~Cache() override {
        if (scratch) { cudaFree(scratch); scratch = nullptr; scratch_size = 0; }
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



void prefix_sum_i64(Cache& cache, const int64_t* in, int64_t* out, int n) {
    if (n == 0) return;
    size_t ts = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, ts, in, out, n);
    cache.ensure_scratch(ts);
    cub::DeviceScan::ExclusiveSum(cache.scratch, ts, in, out, n);
}

int64_t sort_dedup_32(Cache& cache, uint32_t* keys_in, int64_t n,
                       uint32_t* keys_sorted, uint32_t* unique_out,
                       int32_t* num_unique_d, int end_bit) {
    if (n == 0) return 0;
    size_t ts = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, ts, keys_in, keys_sorted, (int)n, 0, end_bit);
    cache.ensure_scratch(ts);
    cub::DeviceRadixSort::SortKeys(cache.scratch, ts, keys_in, keys_sorted, (int)n, 0, end_bit);
    ts = 0;
    cub::DeviceSelect::Unique(nullptr, ts, keys_sorted, unique_out, num_unique_d, (int)n);
    cache.ensure_scratch(ts);
    cub::DeviceSelect::Unique(cache.scratch, ts, keys_sorted, unique_out, num_unique_d, (int)n);
    int32_t num_unique = 0;
    cudaMemcpy(&num_unique, num_unique_d, sizeof(int32_t), cudaMemcpyDeviceToHost);
    return (int64_t)num_unique;
}

int64_t sort_dedup_64(Cache& cache, int64_t* keys_in, int64_t n,
                       int64_t* keys_sorted, int64_t* unique_out,
                       int32_t* num_unique_d, int end_bit) {
    if (n == 0) return 0;
    size_t ts = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, ts, keys_in, keys_sorted, (int)n, 0, end_bit);
    cache.ensure_scratch(ts);
    cub::DeviceRadixSort::SortKeys(cache.scratch, ts, keys_in, keys_sorted, (int)n, 0, end_bit);
    ts = 0;
    cub::DeviceSelect::Unique(nullptr, ts, keys_sorted, unique_out, num_unique_d, (int)n);
    cache.ensure_scratch(ts);
    cub::DeviceSelect::Unique(cache.scratch, ts, keys_sorted, unique_out, num_unique_d, (int)n);
    int32_t num_unique = 0;
    cudaMemcpy(&num_unique, num_unique_d, sizeof(int32_t), cudaMemcpyDeviceToHost);
    return (int64_t)num_unique;
}

void sort_desc_float_int(Cache& cache, const float* kin, float* kout,
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

    
    int32_t ns;
    const int32_t* d_seeds;
    int32_t* seeds_buf = nullptr;

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

    if (ns == 0) {
        if (seeds_buf) cudaFree(seeds_buf);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    float* wd_buf = nullptr;
    if (!is_subset) {
        cudaMalloc(&wd_buf, (size_t)nv * sizeof(float));
        int B = 256, G = (nv + B - 1) / B;
        k_weighted_degrees<<<G, B>>>(d_off, d_w, nv, wd_buf);
    }

    
    int64_t* seed_deg_buf = nullptr;
    cudaMalloc(&seed_deg_buf, (size_t)ns * sizeof(int64_t));
    {
        int B = 256, G = (ns + B - 1) / B;
        k_seed_degrees<<<G, B>>>(d_off, d_seeds, ns, seed_deg_buf);
    }

    int64_t* seed_edge_off_buf = nullptr;
    cudaMalloc(&seed_edge_off_buf, ((size_t)ns + 1) * sizeof(int64_t));
    prefix_sum_i64(cache, seed_deg_buf, seed_edge_off_buf, ns);

    
    int64_t last_deg = 0, last_off = 0;
    cudaMemcpy(&last_deg, seed_deg_buf + ns - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_off, seed_edge_off_buf + ns - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
    int64_t total_edges = last_off + last_deg;
    
    cudaMemcpy(seed_edge_off_buf + ns, &total_edges, sizeof(int64_t), cudaMemcpyHostToDevice);

    cudaFree(seed_deg_buf);

    if (total_edges == 0) {
        if (seeds_buf) cudaFree(seeds_buf);
        if (wd_buf) cudaFree(wd_buf);
        cudaFree(seed_edge_off_buf);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t* pair_cnt_buf = nullptr;
    cudaMalloc(&pair_cnt_buf, (size_t)total_edges * sizeof(int64_t));
    k_count_pairs_edge<<<(int)total_edges, 128>>>(d_off, d_idx, d_seeds, ns,
                                                    seed_edge_off_buf, total_edges,
                                                    pair_cnt_buf);

    
    int64_t* pair_off_buf = nullptr;
    cudaMalloc(&pair_off_buf, (size_t)total_edges * sizeof(int64_t));
    prefix_sum_i64(cache, pair_cnt_buf, pair_off_buf, (int)total_edges);

    
    int64_t last_pair_cnt = 0, last_pair_off = 0;
    cudaMemcpy(&last_pair_cnt, pair_cnt_buf + total_edges - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_pair_off, pair_off_buf + total_edges - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
    int64_t total_pairs = last_pair_off + last_pair_cnt;

    cudaFree(pair_cnt_buf);

    if (total_pairs <= 0) {
        if (seeds_buf) cudaFree(seeds_buf);
        if (wd_buf) cudaFree(wd_buf);
        cudaFree(seed_edge_off_buf);
        cudaFree(pair_off_buf);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int end_bit = 0;
    {
        int64_t mx = (int64_t)(nv - 1) * nv + (nv - 1);
        while ((1LL << end_bit) <= mx && end_bit < 64) end_bit++;
    }

    int32_t* num_unique_buf = nullptr;
    cudaMalloc(&num_unique_buf, sizeof(int32_t));

    int64_t num_unique = 0;
    int32_t* first_buf = nullptr;
    int32_t* second_buf = nullptr;

    if (use_32bit) {
        uint32_t* keys_buf = nullptr;
        cudaMalloc(&keys_buf, (size_t)total_pairs * sizeof(uint32_t));
        k_write_pairs_edge_32<<<(int)total_edges, 128>>>(d_off, d_idx, d_seeds, ns,
                                                           seed_edge_off_buf, total_edges,
                                                           pair_off_buf, keys_buf, nv);

        cudaFree(seed_edge_off_buf);
        cudaFree(pair_off_buf);

        uint32_t* keys_sorted = nullptr;
        uint32_t* unique_keys = nullptr;
        cudaMalloc(&keys_sorted, (size_t)total_pairs * sizeof(uint32_t));
        cudaMalloc(&unique_keys, (size_t)total_pairs * sizeof(uint32_t));

        num_unique = sort_dedup_32(cache, keys_buf, total_pairs, keys_sorted, unique_keys,
                                    num_unique_buf, end_bit);

        cudaFree(keys_buf);
        cudaFree(keys_sorted);
        cudaFree(num_unique_buf);

        if (num_unique == 0) {
            cudaFree(unique_keys);
            if (seeds_buf) cudaFree(seeds_buf);
            if (wd_buf) cudaFree(wd_buf);
            return {nullptr, nullptr, nullptr, 0};
        }

        cudaMalloc(&first_buf, (size_t)num_unique * sizeof(int32_t));
        cudaMalloc(&second_buf, (size_t)num_unique * sizeof(int32_t));
        {
            int B = 256, G = (int)((num_unique + B - 1) / B);
            k_decode_unique_32<<<G, B>>>(unique_keys, num_unique, first_buf, second_buf, nv);
        }
        cudaFree(unique_keys);
    } else {
        int64_t* keys_buf = nullptr;
        cudaMalloc(&keys_buf, (size_t)total_pairs * sizeof(int64_t));
        k_write_pairs_edge_64<<<(int)total_edges, 128>>>(d_off, d_idx, d_seeds, ns,
                                                           seed_edge_off_buf, total_edges,
                                                           pair_off_buf, keys_buf, nv);

        cudaFree(seed_edge_off_buf);
        cudaFree(pair_off_buf);

        int64_t* keys_sorted = nullptr;
        int64_t* unique_keys = nullptr;
        cudaMalloc(&keys_sorted, (size_t)total_pairs * sizeof(int64_t));
        cudaMalloc(&unique_keys, (size_t)total_pairs * sizeof(int64_t));

        num_unique = sort_dedup_64(cache, keys_buf, total_pairs, keys_sorted, unique_keys,
                                    num_unique_buf, end_bit);

        cudaFree(keys_buf);
        cudaFree(keys_sorted);
        cudaFree(num_unique_buf);

        if (num_unique == 0) {
            cudaFree(unique_keys);
            if (seeds_buf) cudaFree(seeds_buf);
            if (wd_buf) cudaFree(wd_buf);
            return {nullptr, nullptr, nullptr, 0};
        }

        cudaMalloc(&first_buf, (size_t)num_unique * sizeof(int32_t));
        cudaMalloc(&second_buf, (size_t)num_unique * sizeof(int32_t));
        {
            int B = 256, G = (int)((num_unique + B - 1) / B);
            k_decode_unique_64<<<G, B>>>(unique_keys, num_unique, first_buf, second_buf, nv);
        }
        cudaFree(unique_keys);
    }

    
    float* scores_buf = nullptr;
    cudaMalloc(&scores_buf, (size_t)num_unique * sizeof(float));
    if (is_subset) {
        if (num_unique > 0) {
            int wpb = 4;
            int B = wpb * 32;
            int G = ((int)num_unique + wpb - 1) / wpb;
            k_overlap_lazy<<<G, B>>>(d_off, d_idx, d_w, first_buf, second_buf,
                                      num_unique, scores_buf);
        }
    } else {
        if (num_unique > 0) {
            int wpb = 4;
            int B = wpb * 32;
            int G = ((int)num_unique + wpb - 1) / wpb;
            k_overlap<<<G, B>>>(d_off, d_idx, d_w, wd_buf, first_buf, second_buf,
                                 num_unique, scores_buf);
        }
    }

    if (wd_buf) { cudaFree(wd_buf); wd_buf = nullptr; }
    if (seeds_buf) { cudaFree(seeds_buf); seeds_buf = nullptr; }

    
    int64_t result_count = num_unique;
    if (topk.has_value() && (int64_t)topk.value() < num_unique) {
        result_count = (int64_t)topk.value();

        int32_t* idx_buf = nullptr;
        cudaMalloc(&idx_buf, (size_t)num_unique * sizeof(int32_t));
        {
            int B = 256, G = ((int)num_unique + B - 1) / B;
            k_iota<<<G, B>>>(idx_buf, (int64_t)num_unique);
        }

        float* scores_sorted = nullptr;
        int32_t* idx_sorted = nullptr;
        cudaMalloc(&scores_sorted, (size_t)num_unique * sizeof(float));
        cudaMalloc(&idx_sorted, (size_t)num_unique * sizeof(int32_t));
        sort_desc_float_int(cache, scores_buf, scores_sorted, idx_buf, idx_sorted, (int)num_unique);

        cudaFree(idx_buf);

        int32_t* tk_first = nullptr;
        int32_t* tk_second = nullptr;
        float* tk_scores = nullptr;
        cudaMalloc(&tk_first, (size_t)result_count * sizeof(int32_t));
        cudaMalloc(&tk_second, (size_t)result_count * sizeof(int32_t));
        cudaMalloc(&tk_scores, (size_t)result_count * sizeof(float));

        if (result_count > 0) {
            int B = 256, G = ((int)result_count + B - 1) / B;
            k_gather_i32<<<G, B>>>(first_buf, idx_sorted, tk_first, (int64_t)result_count);
            k_gather_i32<<<G, B>>>(second_buf, idx_sorted, tk_second, (int64_t)result_count);
            k_gather_f32<<<G, B>>>(scores_buf, idx_sorted, tk_scores, (int64_t)result_count);
        }

        cudaFree(scores_sorted);
        cudaFree(idx_sorted);
        cudaFree(first_buf);
        cudaFree(second_buf);
        cudaFree(scores_buf);

        first_buf = tk_first;
        second_buf = tk_second;
        scores_buf = tk_scores;
    }

    return {first_buf, second_buf, scores_buf, (std::size_t)result_count};
}

}  
