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
#include <numeric>
#include <vector>

namespace aai {

namespace {




static bool g_pool_init = false;
static void init_pool() {
    if (!g_pool_init) {
        cudaMemPool_t pool;
        cudaDeviceGetDefaultMemPool(&pool, 0);
        uint64_t thr = UINT64_MAX;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr);
        g_pool_init = true;
    }
}
#define DALLOC(p, bytes, s) cudaMallocAsync((void**)&(p), (bytes), (s))
#define DFREE(p, s) do { if(p) cudaFreeAsync((p),(s)); } while(0)




struct Cache : Cacheable {
    int32_t* d_all_seeds = nullptr;
    int32_t all_seeds_cap = 0;

    ~Cache() override {
        if (d_all_seeds) cudaFree(d_all_seeds);
    }
};





__global__ void weighted_degrees_kernel(
    const int32_t* __restrict__ off,
    const float*   __restrict__ ew,
    float*         __restrict__ wd,
    int32_t nv)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv) return;
    float s = 0.f;
    for (int i = off[v]; i < off[v+1]; i++) s += ew[i];
    wd[v] = s;
}

__global__ void count_exp_kernel(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ idx,
    const int32_t* __restrict__ seeds,
    int32_t ns,
    int64_t* __restrict__ counts)
{
    int si = blockIdx.x;
    if (si >= ns) return;
    int32_t u = seeds[si], us = off[u], ue = off[u+1];
    int64_t c = 0;
    for (int i = us + threadIdx.x; i < ue; i += blockDim.x) {
        int32_t w = idx[i];
        c += (int64_t)(off[w+1] - off[w]);
    }
    typedef cub::BlockReduce<int64_t, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    int64_t tot = BR(tmp).Sum(c);
    if (threadIdx.x == 0) counts[si] = tot;
}

__global__ void write_exp_kernel(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ idx,
    const int32_t* __restrict__ seeds,
    int32_t ns,
    const int64_t* __restrict__ exp_off,
    int64_t*       __restrict__ out)
{
    int si = blockIdx.x;
    if (si >= ns) return;
    int32_t u = seeds[si], us = off[u], ue = off[u+1];
    int64_t base = exp_off[si];
    __shared__ unsigned long long spos;
    if (threadIdx.x == 0) spos = 0ULL;
    __syncthreads();
    for (int i = us + threadIdx.x; i < ue; i += blockDim.x) {
        int32_t w = idx[i], ws = off[w], we = off[w+1], wd = we - ws;
        int64_t p = (int64_t)atomicAdd(&spos, (unsigned long long)wd);
        for (int j = 0; j < wd; j++) {
            int32_t v = idx[ws + j];
            out[base + p + j] = (v != u)
                ? (((int64_t)(uint32_t)si << 32) | (int64_t)(uint32_t)v)
                : 0x7FFFFFFFFFFFFFFFLL;
        }
    }
}

__global__ __launch_bounds__(256, 8)
void sorensen_kernel(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ idx,
    const float*   __restrict__ ew,
    const float*   __restrict__ wd,
    const int32_t* __restrict__ seeds,
    const int64_t* __restrict__ pkeys,
    int64_t np,
    int32_t* __restrict__ of1,
    int32_t* __restrict__ of2,
    float*   __restrict__ osc)
{
    int64_t pi = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (pi >= np) return;

    int64_t k = pkeys[pi];
    int32_t si = (int32_t)((uint64_t)k >> 32);
    int32_t v  = (int32_t)(k & 0xFFFFFFFFLL);
    int32_t u  = seeds[si];

    int32_t us = off[u], ue = off[u+1], ud = ue - us;
    int32_t vs = off[v], ve = off[v+1], vd = ve - vs;

    int32_t ss, sd, ls, ld;
    if (ud <= vd) { ss = us; sd = ud; ls = vs; ld = vd; }
    else          { ss = vs; sd = vd; ls = us; ld = ud; }

    float sum = 0.f;
    for (int i = lane; i < sd; i += 32) {
        int32_t tgt = __ldg(&idx[ss + i]);
        float   ws  = __ldg(&ew[ss + i]);
        int lo = 0, hi = ld;
        while (lo < hi) { int m = (lo+hi)>>1; if (__ldg(&idx[ls+m]) < tgt) lo=m+1; else hi=m; }
        if (lo < ld && __ldg(&idx[ls+lo]) == tgt)
            sum += fminf(ws, __ldg(&ew[ls+lo]));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, o);

    if (lane == 0) {
        float d = wd[u] + wd[v];
        of1[pi] = u; of2[pi] = v;
        osc[pi] = (d > 0.f) ? (2.f * sum / d) : 0.f;
    }
}

__global__ void init_seq_kernel(int32_t* arr, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = (int32_t)i;
}

__global__ void negate_kernel(const float* __restrict__ in, float* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = -in[i];
}

__global__ void gather_kernel(
    const int32_t* __restrict__ sf, const int32_t* __restrict__ ss,
    const float* __restrict__ sc, const int32_t* __restrict__ perm,
    int32_t* __restrict__ df, int32_t* __restrict__ ds, float* __restrict__ dsc,
    int64_t k)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) {
        int32_t j = perm[i];
        df[i] = sf[j]; ds[i] = ss[j]; dsc[i] = sc[j];
    }
}

}  

similarity_result_float_t sorensen_all_pairs_similarity(const graph32_t& graph,
                                                        const float* edge_weights,
                                                        const int32_t* vertices,
                                                        std::size_t num_vertices,
                                                        std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    init_pool();
    cudaStream_t st = 0;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t nv = graph.number_of_vertices;

    
    const int32_t* d_seeds;
    int32_t ns;
    if (vertices != nullptr) {
        d_seeds = vertices;
        ns = (int32_t)num_vertices;
    } else {
        ns = nv;
        if (ns > cache.all_seeds_cap) {
            if (cache.d_all_seeds) cudaFree(cache.d_all_seeds);
            cudaMalloc(&cache.d_all_seeds, (size_t)ns * sizeof(int32_t));
            cache.all_seeds_cap = ns;
            std::vector<int32_t> h(ns);
            std::iota(h.begin(), h.end(), 0);
            cudaMemcpy(cache.d_all_seeds, h.data(), ns * sizeof(int32_t), cudaMemcpyHostToDevice);
        }
        d_seeds = cache.d_all_seeds;
    }

    if (ns == 0) return {nullptr, nullptr, nullptr, 0};

    
    float* d_wd; DALLOC(d_wd, (size_t)nv*4, st);
    { int b=256,g=(nv+b-1)/b; weighted_degrees_kernel<<<g,b,0,st>>>(offsets,edge_weights,d_wd,nv); }

    
    int64_t* d_cnt; DALLOC(d_cnt, (size_t)ns*8, st);
    count_exp_kernel<<<ns,256,0,st>>>(offsets,indices,d_seeds,ns,d_cnt);

    
    int64_t* d_eoff; DALLOC(d_eoff, (size_t)(ns+1)*8, st);
    { size_t tb=0; cub::DeviceScan::ExclusiveSum(0,tb,d_cnt,d_eoff,ns,st);
      void* dt; DALLOC(dt,tb,st); cub::DeviceScan::ExclusiveSum(dt,tb,d_cnt,d_eoff,ns,st); DFREE(dt,st); }

    
    int64_t total_exp;
    { int64_t h[2];
      cudaMemcpyAsync(h,d_cnt+ns-1,8,cudaMemcpyDeviceToHost,st);
      cudaMemcpyAsync(h+1,d_eoff+ns-1,8,cudaMemcpyDeviceToHost,st);
      cudaStreamSynchronize(st); total_exp=h[0]+h[1]; }
    DFREE(d_cnt,st);

    if (total_exp==0) { DFREE(d_wd,st); DFREE(d_eoff,st); return {nullptr, nullptr, nullptr, 0}; }

    
    int64_t* d_exp; DALLOC(d_exp,(size_t)total_exp*8,st);
    write_exp_kernel<<<ns,256,0,st>>>(offsets,indices,d_seeds,ns,d_eoff,d_exp);
    DFREE(d_eoff,st);

    
    int64_t* d_exp2; DALLOC(d_exp2,(size_t)total_exp*8,st);
    { int sb=1; { int v=ns; while(v>0){sb++;v>>=1;} }
      int eb = 32+sb; if(eb>63) eb=63;
      size_t tb=0;
      cub::DeviceRadixSort::SortKeys(0,tb,d_exp,d_exp2,(int)total_exp,0,eb,st);
      void* dt; DALLOC(dt,tb,st);
      cub::DeviceRadixSort::SortKeys(dt,tb,d_exp,d_exp2,(int)total_exp,0,eb,st);
      DFREE(dt,st); }
    DFREE(d_exp,st);

    
    int64_t* d_uniq; DALLOC(d_uniq,(size_t)total_exp*8,st);
    int32_t* d_nsel; DALLOC(d_nsel,4,st);
    { size_t tb=0;
      cub::DeviceSelect::Unique(0,tb,d_exp2,d_uniq,d_nsel,(int)total_exp,st);
      void* dt; DALLOC(dt,tb,st);
      cub::DeviceSelect::Unique(dt,tb,d_exp2,d_uniq,d_nsel,(int)total_exp,st);
      DFREE(dt,st); }
    DFREE(d_exp2,st);

    int32_t h_nsel;
    cudaMemcpyAsync(&h_nsel,d_nsel,4,cudaMemcpyDeviceToHost,st);
    cudaStreamSynchronize(st);
    DFREE(d_nsel,st);
    int64_t nu = (int64_t)h_nsel;

    
    if (nu>0) {
        int64_t lk;
        cudaMemcpyAsync(&lk,d_uniq+nu-1,8,cudaMemcpyDeviceToHost,st);
        cudaStreamSynchronize(st);
        if (lk==0x7FFFFFFFFFFFFFFFLL) nu--;
    }
    if (nu==0) { DFREE(d_uniq,st); DFREE(d_wd,st); return {nullptr, nullptr, nullptr, 0}; }

    
    int32_t* d_f; DALLOC(d_f,nu*4,st);
    int32_t* d_s; DALLOC(d_s,nu*4,st);
    float*   d_sc; DALLOC(d_sc,nu*4,st);
    { int wpb=8, thr=wpb*32, gr=(int)((nu+wpb-1)/wpb);
      sorensen_kernel<<<gr,thr,0,st>>>(offsets,indices,edge_weights,d_wd,d_seeds,d_uniq,nu,d_f,d_s,d_sc); }
    DFREE(d_uniq,st);
    DFREE(d_wd,st);

    
    int64_t rc = nu;
    if (topk.has_value() && rc>(int64_t)topk.value()) {
        int64_t topk_val = (int64_t)topk.value();

        
        float* d_neg; DALLOC(d_neg,rc*4,st);
        { int b=256,g=(int)((rc+b-1)/b); negate_kernel<<<g,b,0,st>>>(d_sc,d_neg,rc); }

        
        int32_t* d_idx; DALLOC(d_idx,rc*4,st);
        { int b=256,g=(int)((rc+b-1)/b); init_seq_kernel<<<g,b,0,st>>>(d_idx,rc); }

        
        float* d_neg2; DALLOC(d_neg2,rc*4,st);
        int32_t* d_idx2; DALLOC(d_idx2,rc*4,st);
        { size_t tb=0;
          cub::DeviceRadixSort::SortPairs(0,tb,d_neg,d_neg2,d_idx,d_idx2,(int)rc,0,32,st);
          void* dt; DALLOC(dt,tb,st);
          cub::DeviceRadixSort::SortPairs(dt,tb,d_neg,d_neg2,d_idx,d_idx2,(int)rc,0,32,st);
          DFREE(dt,st); }
        DFREE(d_neg,st); DFREE(d_neg2,st); DFREE(d_idx,st);

        
        int32_t* d_f2; DALLOC(d_f2,topk_val*4,st);
        int32_t* d_s2; DALLOC(d_s2,topk_val*4,st);
        float*   d_sc2; DALLOC(d_sc2,topk_val*4,st);
        { int b=256,g=(int)((topk_val+b-1)/b);
          gather_kernel<<<g,b,0,st>>>(d_f,d_s,d_sc,d_idx2,d_f2,d_s2,d_sc2,topk_val); }
        DFREE(d_idx2,st);
        DFREE(d_f,st); DFREE(d_s,st); DFREE(d_sc,st);
        d_f=d_f2; d_s=d_s2; d_sc=d_sc2;
        rc = topk_val;
    }

    cudaDeviceSynchronize();
    return {d_f, d_s, d_sc, (std::size_t)rc};
}

}  
