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
#include <climits>
#include <algorithm>
#include <optional>

namespace aai {

namespace {


struct TmpBuf {
    void* p = nullptr;
    ~TmpBuf() { if (p) cudaFree(p); }
    void* release() { void* tmp = p; p = nullptr; return tmp; }
};

struct Cache : Cacheable {
    void* scratch = nullptr;
    size_t scratch_capacity = 0;

    void ensure_scratch(size_t need) {
        if (need > scratch_capacity) {
            if (scratch) cudaFree(scratch);
            scratch_capacity = std::max(need * 2, (size_t)(1 << 20));
            cudaMalloc(&scratch, scratch_capacity);
        }
    }

    ~Cache() override {
        if (scratch) { cudaFree(scratch); scratch = nullptr; }
    }
};

static int compute_nv_bits(int32_t nv) {
    int b = 0;
    uint32_t v = (uint32_t)(nv - 1);
    while (v > 0) { v >>= 1; b++; }
    return (b == 0) ? 1 : b;
}





__global__ void count_2hop_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_seeds) return;
    int32_t u = seeds[tid];
    int32_t us = offsets[u], ue = offsets[u + 1];
    int64_t c = 0;
    for (int32_t i = us; i < ue; i++)
        c += (int64_t)(offsets[indices[i] + 1] - offsets[indices[i]]);
    counts[tid] = c;
}

__global__ void expand_2hop_block_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int nv_bits,
    const int64_t* __restrict__ pair_offsets,
    uint64_t* __restrict__ keys) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;
    int32_t u = seeds[seed_idx];
    int32_t us = offsets[u], ue = offsets[u + 1];
    int32_t u_deg = ue - us;
    uint64_t prefix = (uint64_t)seed_idx << nv_bits;
    __shared__ int64_t write_pos;
    if (threadIdx.x == 0) write_pos = pair_offsets[seed_idx];
    __syncthreads();
    for (int i = threadIdx.x; i < u_deg; i += blockDim.x) {
        int32_t w = indices[us + i];
        int32_t ws = offsets[w], we = offsets[w + 1];
        int32_t w_deg = we - ws;
        int64_t my_pos = atomicAdd((unsigned long long*)&write_pos, (unsigned long long)w_deg);
        for (int j = 0; j < w_deg; j++)
            keys[my_pos + j] = prefix | (uint64_t)indices[ws + j];
    }
}

__global__ void expand_2hop_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int nv_bits,
    const int64_t* __restrict__ pair_offsets,
    uint64_t* __restrict__ keys) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_seeds) return;
    int32_t u = seeds[tid];
    int32_t us = offsets[u], ue = offsets[u + 1];
    uint64_t prefix = (uint64_t)tid << nv_bits;
    int64_t wp = pair_offsets[tid];
    for (int32_t i = us; i < ue; i++) {
        int32_t w = indices[i];
        int32_t ws = offsets[w], we = offsets[w + 1];
        for (int32_t j = ws; j < we; j++)
            keys[wp++] = prefix | (uint64_t)indices[j];
    }
}

__global__ void expand_2hop_block32_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int nv_bits,
    const int64_t* __restrict__ pair_offsets,
    uint32_t* __restrict__ keys) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;
    int32_t u = seeds[seed_idx];
    int32_t us = offsets[u], ue = offsets[u + 1];
    int32_t u_deg = ue - us;
    uint32_t prefix = (uint32_t)seed_idx << nv_bits;
    __shared__ int64_t write_pos;
    if (threadIdx.x == 0) write_pos = pair_offsets[seed_idx];
    __syncthreads();
    for (int i = threadIdx.x; i < u_deg; i += blockDim.x) {
        int32_t w = indices[us + i];
        int32_t ws = offsets[w], we = offsets[w + 1];
        int32_t w_deg = we - ws;
        int64_t my_pos = atomicAdd((unsigned long long*)&write_pos, (unsigned long long)w_deg);
        for (int j = 0; j < w_deg; j++)
            keys[my_pos + j] = prefix | (uint32_t)indices[ws + j];
    }
}

__global__ void expand_2hop_thread32_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int nv_bits,
    const int64_t* __restrict__ pair_offsets,
    uint32_t* __restrict__ keys) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_seeds) return;
    int32_t u = seeds[tid];
    int32_t us = offsets[u], ue = offsets[u + 1];
    uint32_t prefix = (uint32_t)tid << nv_bits;
    int64_t wp = pair_offsets[tid];
    for (int32_t i = us; i < ue; i++) {
        int32_t w = indices[i];
        int32_t ws = offsets[w], we = offsets[w + 1];
        for (int32_t j = ws; j < we; j++)
            keys[wp++] = prefix | (uint32_t)indices[j];
    }
}

__global__ void compute_from_rle_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ seeds,
    const uint64_t* __restrict__ ukeys,
    const int32_t* __restrict__ counts,
    int64_t num_runs,
    int nv_bits,
    uint64_t nv_mask,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int32_t* __restrict__ out_count) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_runs) return;
    uint64_t key = ukeys[tid];
    int32_t seed_idx = (int32_t)(key >> nv_bits);
    int32_t v = (int32_t)(key & nv_mask);
    int32_t u = seeds[seed_idx];
    if (u == v) return;
    int32_t pos = atomicAdd(out_count, 1);
    out_first[pos] = u;
    out_second[pos] = v;
    int32_t c = counts[tid];
    int32_t ds = (offsets[u+1]-offsets[u]) + (offsets[v+1]-offsets[v]);
    out_scores[pos] = (ds > 0) ? 2.0f * (float)c / (float)ds : 0.0f;
}

__device__ __forceinline__ int32_t intersect_count(
    const int32_t* __restrict__ a, int32_t na,
    const int32_t* __restrict__ b, int32_t nb) {
    if (na == 0 || nb == 0) return 0;
    if (a[na-1] < b[0] || b[nb-1] < a[0]) return 0;
    int32_t i = 0, j = 0;
    if (a[0] < b[0]) {
        int lo=0,hi=na; while(lo<hi){int m=(lo+hi)>>1; if(a[m]<b[0])lo=m+1;else hi=m;} i=lo;
    }
    if (i<na && b[0]<a[i]) {
        int lo=0,hi=nb; while(lo<hi){int m=(lo+hi)>>1; if(b[m]<a[i])lo=m+1;else hi=m;} j=lo;
    }
    int32_t cnt=0;
    while (i<na && j<nb) {
        int32_t va=a[i],vb=b[j]; cnt+=(va==vb); i+=(va<=vb); j+=(va>=vb);
    }
    return cnt;
}

__global__ void compute_intersect_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    const uint64_t* __restrict__ ukeys,
    int64_t num_unique,
    int nv_bits,
    uint64_t nv_mask,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int32_t* __restrict__ out_count) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_unique) return;
    uint64_t key = ukeys[tid];
    int32_t seed_idx = (int32_t)(key >> nv_bits);
    int32_t v = (int32_t)(key & nv_mask);
    int32_t u = seeds[seed_idx];
    if (u == v) return;
    int32_t pos = atomicAdd(out_count, 1);
    out_first[pos] = u;
    out_second[pos] = v;
    int32_t us=offsets[u],ue=offsets[u+1],vs=offsets[v],ve=offsets[v+1];
    int32_t c = intersect_count(indices+us,ue-us,indices+vs,ve-vs);
    int32_t ds = (ue-us)+(ve-vs);
    out_scores[pos] = (ds>0) ? 2.0f*(float)c/(float)ds : 0.0f;
}

__global__ void iota_kernel(int32_t* d, int32_t n) {
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t<n)d[t]=t;
}

__global__ void gather_topk_kernel(
    const int32_t* __restrict__ f, const int32_t* __restrict__ s,
    const float* __restrict__ sc, const int32_t* __restrict__ pm,
    int64_t n, int32_t* of, int32_t* os, float* osc) {
    int64_t t=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=n) return;
    int32_t idx=pm[t]; of[t]=f[idx]; os[t]=s[idx]; osc[t]=sc[t];
}

__global__ void compute_from_rle32_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ seeds,
    const uint32_t* __restrict__ ukeys,
    const int32_t* __restrict__ counts,
    int64_t num_runs,
    int nv_bits,
    uint32_t nv_mask,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int32_t* __restrict__ out_count) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_runs) return;
    uint32_t key = ukeys[tid];
    int32_t seed_idx = (int32_t)(key >> nv_bits);
    int32_t v = (int32_t)(key & nv_mask);
    int32_t u = seeds[seed_idx];
    if (u == v) return;
    int32_t pos = atomicAdd(out_count, 1);
    out_first[pos] = u;
    out_second[pos] = v;
    int32_t c = counts[tid];
    int32_t ds = (offsets[u+1]-offsets[u]) + (offsets[v+1]-offsets[v]);
    out_scores[pos] = (ds > 0) ? 2.0f * (float)c / (float)ds : 0.0f;
}

__global__ void compute_intersect32_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    const uint32_t* __restrict__ ukeys,
    int64_t num_unique,
    int nv_bits,
    uint32_t nv_mask,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int32_t* __restrict__ out_count) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_unique) return;
    uint32_t key = ukeys[tid];
    int32_t seed_idx = (int32_t)(key >> nv_bits);
    int32_t v = (int32_t)(key & nv_mask);
    int32_t u = seeds[seed_idx];
    if (u == v) return;
    int32_t pos = atomicAdd(out_count, 1);
    out_first[pos] = u;
    out_second[pos] = v;
    int32_t us=offsets[u],ue=offsets[u+1],vs=offsets[v],ve=offsets[v+1];
    int32_t c = intersect_count(indices+us,ue-us,indices+vs,ve-vs);
    int32_t ds = (ue-us)+(ve-vs);
    out_scores[pos] = (ds>0) ? 2.0f*(float)c/(float)ds : 0.0f;
}





size_t get_ps_temp(int64_t n) {
    size_t t = 0; int64_t* d = nullptr;
    cub::DeviceScan::ExclusiveSum(nullptr, t, d, d, n); return t;
}

void do_ps(int64_t* i, int64_t* o, int64_t n, void* t, size_t tb, cudaStream_t s) {
    cub::DeviceScan::ExclusiveSum(t, tb, i, o, n, s);
}

size_t get_sort32_temp(int64_t n, int eb) {
    size_t t = 0; cub::DoubleBuffer<uint32_t> db(nullptr, nullptr);
    cub::DeviceRadixSort::SortKeys(nullptr, t, db, n, 0, eb); return t;
}

void do_sort32(uint32_t* k, uint32_t* ka, int64_t n, int eb, void* t, size_t tb, int* sel, cudaStream_t s) {
    cub::DoubleBuffer<uint32_t> db(k, ka);
    cub::DeviceRadixSort::SortKeys(t, tb, db, n, 0, eb, s);
    *sel = (db.Current() == ka) ? 1 : 0;
}

size_t get_sort64_temp(int64_t n, int eb) {
    size_t t = 0; cub::DoubleBuffer<uint64_t> db(nullptr, nullptr);
    cub::DeviceRadixSort::SortKeys(nullptr, t, db, n, 0, eb); return t;
}

void do_sort64(uint64_t* k, uint64_t* ka, int64_t n, int eb, void* t, size_t tb, int* sel, cudaStream_t s) {
    cub::DoubleBuffer<uint64_t> db(k, ka);
    cub::DeviceRadixSort::SortKeys(t, tb, db, n, 0, eb, s);
    *sel = (db.Current() == ka) ? 1 : 0;
}

size_t get_rle64_temp(int64_t n) {
    size_t t = 0; uint64_t* d1 = nullptr; int32_t* d2 = nullptr; int64_t* d3 = nullptr;
    cub::DeviceRunLengthEncode::Encode(nullptr, t, d1, d1, d2, d3, n); return t;
}

void do_rle64(uint64_t* in, uint64_t* uo, int32_t* co, int64_t* nr, int64_t n, void* t, size_t tb, cudaStream_t s) {
    cub::DeviceRunLengthEncode::Encode(t, tb, in, uo, co, nr, n, s);
}

size_t get_rle32_temp(int64_t n) {
    size_t t = 0; uint32_t* d1 = nullptr; int32_t* d2 = nullptr; int64_t* d3 = nullptr;
    cub::DeviceRunLengthEncode::Encode(nullptr, t, d1, d1, d2, d3, n); return t;
}

void do_rle32(uint32_t* in, uint32_t* uo, int32_t* co, int64_t* nr, int64_t n, void* t, size_t tb, cudaStream_t s) {
    cub::DeviceRunLengthEncode::Encode(t, tb, in, uo, co, nr, n, s);
}

size_t get_unique64_temp(int64_t n) {
    size_t t = 0; uint64_t* d = nullptr; int64_t* d2 = nullptr;
    cub::DeviceSelect::Unique(nullptr, t, d, d, d2, n); return t;
}

void do_unique64(uint64_t* in, uint64_t* out, int64_t* ns, int64_t n, void* t, size_t tb, cudaStream_t s) {
    cub::DeviceSelect::Unique(t, tb, in, out, ns, n, s);
}

size_t get_unique32_temp(int64_t n) {
    size_t t = 0; uint32_t* d = nullptr; int64_t* d2 = nullptr;
    cub::DeviceSelect::Unique(nullptr, t, d, d, d2, n); return t;
}

void do_unique32(uint32_t* in, uint32_t* out, int64_t* ns, int64_t n, void* t, size_t tb, cudaStream_t s) {
    cub::DeviceSelect::Unique(t, tb, in, out, ns, n, s);
}

size_t get_sort_desc_temp(int64_t n) {
    size_t t = 0; cub::DoubleBuffer<float> d1(nullptr, nullptr); cub::DoubleBuffer<int32_t> d2(nullptr, nullptr);
    cub::DeviceRadixSort::SortPairsDescending(nullptr, t, d1, d2, n, 0, 32); return t;
}

void do_sort_desc(float* sc, float* sca, int32_t* pm, int32_t* pma,
    int64_t n, void* t, size_t tb, float** osc, int32_t** opm, cudaStream_t s) {
    cub::DoubleBuffer<float> db1(sc, sca); cub::DoubleBuffer<int32_t> db2(pm, pma);
    cub::DeviceRadixSort::SortPairsDescending(t, tb, db1, db2, n, 0, 32, s);
    *osc = db1.Current(); *opm = db2.Current();
}

}  

similarity_result_float_t sorensen_all_pairs_similarity(const graph32_t& graph,
                                                        const int32_t* vertices,
                                                        std::size_t num_vertices,
                                                        std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_ind = graph.indices;
    int32_t nv = graph.number_of_vertices;
    bool is_multi = graph.is_multigraph;
    cudaStream_t stream = 0;

    const int32_t* d_seeds;
    int64_t nseeds;
    TmpBuf seeds_buf;

    if (vertices != nullptr) {
        d_seeds = vertices;
        nseeds = (int64_t)num_vertices;
    } else {
        nseeds = nv;
        cudaMalloc(&seeds_buf.p, nseeds * sizeof(int32_t));
        int32_t* d_seeds_buf = static_cast<int32_t*>(seeds_buf.p);
        if (nseeds > 0) iota_kernel<<<((int)nseeds + 255) / 256, 256, 0, stream>>>(d_seeds_buf, (int32_t)nseeds);
        d_seeds = d_seeds_buf;
    }

    if (nseeds == 0)
        return {nullptr, nullptr, nullptr, 0};

    int nv_bits = compute_nv_bits(nv);
    int seed_bits = compute_nv_bits((int32_t)nseeds);
    int total_bits = nv_bits + seed_bits;
    bool use32 = (total_bits <= 32);

    
    TmpBuf cnt_buf, pfx_buf;
    cudaMalloc(&cnt_buf.p, nseeds * sizeof(int64_t));
    cudaMalloc(&pfx_buf.p, nseeds * sizeof(int64_t));
    int64_t* d_cnt = static_cast<int64_t*>(cnt_buf.p);
    int64_t* d_pfx = static_cast<int64_t*>(pfx_buf.p);

    count_2hop_kernel<<<((int)nseeds + 255) / 256, 256, 0, stream>>>(d_off, d_ind, d_seeds, (int32_t)nseeds, d_cnt);

    
    { size_t tb = get_ps_temp(nseeds); cache.ensure_scratch(tb);
      do_ps(d_cnt, d_pfx, nseeds, cache.scratch, tb, stream); }

    int64_t total_raw = 0;
    { int64_t lo, lc;
      cudaMemcpyAsync(&lo, d_pfx + nseeds - 1, 8, cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(&lc, d_cnt + nseeds - 1, 8, cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      total_raw = lo + lc; }

    if (total_raw == 0)
        return {nullptr, nullptr, nullptr, 0};

    int64_t num_runs = 0;
    int32_t num_pairs = 0;

    
    int32_t* d_of = nullptr;
    int32_t* d_os = nullptr;
    float* d_osc = nullptr;

    if (use32) {
        
        TmpBuf keys32_buf, ka32_buf, ukeys32_buf, nruns_buf, counts_buf, out_cnt_buf;

        cudaMalloc(&keys32_buf.p, total_raw * sizeof(uint32_t));
        uint32_t* d_k32 = static_cast<uint32_t*>(keys32_buf.p);

        if (vertices != nullptr && num_vertices <= 1024) {
            expand_2hop_block32_kernel<<<(int32_t)nseeds, 256, 0, stream>>>(d_off, d_ind, d_seeds, (int32_t)nseeds, nv_bits, d_pfx, d_k32);
        } else {
            if (nseeds > 0) expand_2hop_thread32_kernel<<<((int)nseeds + 255) / 256, 256, 0, stream>>>(d_off, d_ind, d_seeds, (int32_t)nseeds, nv_bits, d_pfx, d_k32);
        }

        cudaMalloc(&ka32_buf.p, total_raw * sizeof(uint32_t));
        uint32_t* d_ka32 = static_cast<uint32_t*>(ka32_buf.p);
        int sel = 0;
        { size_t tb = get_sort32_temp(total_raw, total_bits); cache.ensure_scratch(tb);
          do_sort32(d_k32, d_ka32, total_raw, total_bits, cache.scratch, tb, &sel, stream); }
        uint32_t* d_sorted32 = (sel == 1) ? d_ka32 : d_k32;

        cudaMalloc(&ukeys32_buf.p, total_raw * sizeof(uint32_t));
        uint32_t* d_uk32 = static_cast<uint32_t*>(ukeys32_buf.p);
        cudaMalloc(&nruns_buf.p, sizeof(int64_t));
        int64_t* d_nruns = static_cast<int64_t*>(nruns_buf.p);
        int32_t* d_counts = nullptr;

        if (!is_multi) {
            cudaMalloc(&counts_buf.p, total_raw * sizeof(int32_t));
            d_counts = static_cast<int32_t*>(counts_buf.p);
            { size_t tb = get_rle32_temp(total_raw); cache.ensure_scratch(tb);
              do_rle32(d_sorted32, d_uk32, d_counts, d_nruns, total_raw, cache.scratch, tb, stream); }
        } else {
            { size_t tb = get_unique32_temp(total_raw); cache.ensure_scratch(tb);
              do_unique32(d_sorted32, d_uk32, d_nruns, total_raw, cache.scratch, tb, stream); }
        }
        cudaMemcpyAsync(&num_runs, d_nruns, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (num_runs == 0)
            return {nullptr, nullptr, nullptr, 0};

        cudaMalloc(&d_of, num_runs * sizeof(int32_t));
        cudaMalloc(&d_os, num_runs * sizeof(int32_t));
        cudaMalloc(&d_osc, num_runs * sizeof(float));
        cudaMalloc(&out_cnt_buf.p, sizeof(int32_t));
        int32_t* d_out_cnt = static_cast<int32_t*>(out_cnt_buf.p);
        cudaMemsetAsync(d_out_cnt, 0, 4, stream);

        uint32_t nv_mask32 = (1u << nv_bits) - 1;
        if (!is_multi) {
            int64_t g = (num_runs + 255) / 256; int grid = (int)std::min(g, (int64_t)INT_MAX);
            compute_from_rle32_kernel<<<grid, 256, 0, stream>>>(d_off, d_seeds, d_uk32, d_counts, num_runs, nv_bits, nv_mask32,
                d_of, d_os, d_osc, d_out_cnt);
        } else {
            int64_t g = (num_runs + 255) / 256; int grid = (int)std::min(g, (int64_t)INT_MAX);
            compute_intersect32_kernel<<<grid, 256, 0, stream>>>(d_off, d_ind, d_seeds, d_uk32, num_runs, nv_bits, nv_mask32,
                d_of, d_os, d_osc, d_out_cnt);
        }
        cudaMemcpyAsync(&num_pairs, d_out_cnt, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    } else {
        
        TmpBuf keys64_buf, ka64_buf, ukeys64_buf, nruns_buf, counts_buf, out_cnt_buf;

        cudaMalloc(&keys64_buf.p, total_raw * sizeof(uint64_t));
        uint64_t* d_k64 = static_cast<uint64_t*>(keys64_buf.p);

        if (vertices != nullptr && num_vertices <= 1024) {
            expand_2hop_block_kernel<<<(int32_t)nseeds, 256, 0, stream>>>(d_off, d_ind, d_seeds, (int32_t)nseeds, nv_bits, d_pfx, d_k64);
        } else {
            if (nseeds > 0) expand_2hop_thread_kernel<<<((int)nseeds + 255) / 256, 256, 0, stream>>>(d_off, d_ind, d_seeds, (int32_t)nseeds, nv_bits, d_pfx, d_k64);
        }

        cudaMalloc(&ka64_buf.p, total_raw * sizeof(uint64_t));
        uint64_t* d_ka64 = static_cast<uint64_t*>(ka64_buf.p);
        int sel = 0;
        { size_t tb = get_sort64_temp(total_raw, total_bits); cache.ensure_scratch(tb);
          do_sort64(d_k64, d_ka64, total_raw, total_bits, cache.scratch, tb, &sel, stream); }
        uint64_t* d_sorted64 = (sel == 1) ? d_ka64 : d_k64;

        cudaMalloc(&ukeys64_buf.p, total_raw * sizeof(uint64_t));
        uint64_t* d_uk64 = static_cast<uint64_t*>(ukeys64_buf.p);
        cudaMalloc(&nruns_buf.p, sizeof(int64_t));
        int64_t* d_nruns = static_cast<int64_t*>(nruns_buf.p);
        int32_t* d_counts = nullptr;

        if (!is_multi) {
            cudaMalloc(&counts_buf.p, total_raw * sizeof(int32_t));
            d_counts = static_cast<int32_t*>(counts_buf.p);
            { size_t tb = get_rle64_temp(total_raw); cache.ensure_scratch(tb);
              do_rle64(d_sorted64, d_uk64, d_counts, d_nruns, total_raw, cache.scratch, tb, stream); }
        } else {
            { size_t tb = get_unique64_temp(total_raw); cache.ensure_scratch(tb);
              do_unique64(d_sorted64, d_uk64, d_nruns, total_raw, cache.scratch, tb, stream); }
        }
        cudaMemcpyAsync(&num_runs, d_nruns, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (num_runs == 0)
            return {nullptr, nullptr, nullptr, 0};

        cudaMalloc(&d_of, num_runs * sizeof(int32_t));
        cudaMalloc(&d_os, num_runs * sizeof(int32_t));
        cudaMalloc(&d_osc, num_runs * sizeof(float));
        cudaMalloc(&out_cnt_buf.p, sizeof(int32_t));
        int32_t* d_out_cnt = static_cast<int32_t*>(out_cnt_buf.p);
        cudaMemsetAsync(d_out_cnt, 0, 4, stream);

        uint64_t nv_mask64 = (1ULL << nv_bits) - 1;
        if (!is_multi) {
            int64_t g = (num_runs + 255) / 256; int grid = (int)std::min(g, (int64_t)INT_MAX);
            compute_from_rle_kernel<<<grid, 256, 0, stream>>>(d_off, d_seeds, d_uk64, d_counts, num_runs, nv_bits, nv_mask64,
                d_of, d_os, d_osc, d_out_cnt);
        } else {
            int64_t g = (num_runs + 255) / 256; int grid = (int)std::min(g, (int64_t)INT_MAX);
            compute_intersect_kernel<<<grid, 256, 0, stream>>>(d_off, d_ind, d_seeds, d_uk64, num_runs, nv_bits, nv_mask64,
                d_of, d_os, d_osc, d_out_cnt);
        }
        cudaMemcpyAsync(&num_pairs, d_out_cnt, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    if (num_pairs == 0) {
        cudaFree(d_of); cudaFree(d_os); cudaFree(d_osc);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    if (topk.has_value() && topk.value() < (std::size_t)num_pairs) {
        int64_t topk_val = (int64_t)topk.value();

        TmpBuf pm_buf, sa_buf, pa_buf;
        cudaMalloc(&pm_buf.p, (int64_t)num_pairs * sizeof(int32_t));
        int32_t* d_pm = static_cast<int32_t*>(pm_buf.p);
        iota_kernel<<<(num_pairs + 255) / 256, 256, 0, stream>>>(d_pm, num_pairs);

        cudaMalloc(&sa_buf.p, (int64_t)num_pairs * sizeof(float));
        cudaMalloc(&pa_buf.p, (int64_t)num_pairs * sizeof(int32_t));
        float* d_sa = static_cast<float*>(sa_buf.p);
        int32_t* d_pa = static_cast<int32_t*>(pa_buf.p);

        float* s_sc; int32_t* s_pm;
        { size_t tb = get_sort_desc_temp(num_pairs); cache.ensure_scratch(tb);
          do_sort_desc(d_osc, d_sa, d_pm, d_pa, num_pairs, cache.scratch, tb, &s_sc, &s_pm, stream); }

        int32_t* d_tf = nullptr; int32_t* d_ts = nullptr; float* d_tsc = nullptr;
        cudaMalloc(&d_tf, topk_val * sizeof(int32_t));
        cudaMalloc(&d_ts, topk_val * sizeof(int32_t));
        cudaMalloc(&d_tsc, topk_val * sizeof(float));

        int grid_topk = (int)std::min((topk_val + 255) / 256, (int64_t)INT_MAX);
        gather_topk_kernel<<<grid_topk, 256, 0, stream>>>(d_of, d_os, s_sc, s_pm, topk_val, d_tf, d_ts, d_tsc);

        cudaFree(d_of); cudaFree(d_os); cudaFree(d_osc);
        return {d_tf, d_ts, d_tsc, (std::size_t)topk_val};
    }

    
    if (num_pairs < num_runs) {
        int32_t* d_rf = nullptr; int32_t* d_rs = nullptr; float* d_rsc = nullptr;
        cudaMalloc(&d_rf, (int64_t)num_pairs * sizeof(int32_t));
        cudaMalloc(&d_rs, (int64_t)num_pairs * sizeof(int32_t));
        cudaMalloc(&d_rsc, (int64_t)num_pairs * sizeof(float));
        cudaMemcpyAsync(d_rf, d_of, num_pairs * 4, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_rs, d_os, num_pairs * 4, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_rsc, d_osc, num_pairs * 4, cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_of); cudaFree(d_os); cudaFree(d_osc);
        return {d_rf, d_rs, d_rsc, (std::size_t)num_pairs};
    }

    return {d_of, d_os, d_osc, (std::size_t)num_pairs};
}

}  
