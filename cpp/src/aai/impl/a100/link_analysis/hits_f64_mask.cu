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
#include <cstddef>

namespace aai {

namespace {

static constexpr int BS = 256;



__global__ void k_count_active(
    const int32_t* __restrict__ off,
    const uint32_t* __restrict__ mask,
    int32_t* __restrict__ cnt,
    int32_t nv)
{
    int v = blockIdx.x * BS + threadIdx.x;
    if (v >= nv) return;
    int s = off[v], e = off[v + 1];
    int c = 0;
    for (int i = s; i < e; i++)
        if ((mask[i >> 5] >> (i & 31)) & 1u) c++;
    cnt[v] = c;
}

__global__ void k_scatter_csc(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ idx,
    const uint32_t* __restrict__ mask,
    const int32_t* __restrict__ coff,
    int32_t* __restrict__ cidx,
    int32_t nv)
{
    int v = blockIdx.x * BS + threadIdx.x;
    if (v >= nv) return;
    int s = off[v], e = off[v + 1];
    int pos = coff[v];
    for (int i = s; i < e; i++)
        if ((mask[i >> 5] >> (i & 31)) & 1u)
            cidx[pos++] = idx[i];
}

__global__ void k_histogram(
    const int32_t* __restrict__ cidx,
    int32_t* __restrict__ cnt,
    const int32_t* __restrict__ coff_ptr, 
    int32_t nv)
{
    int e = blockIdx.x * BS + threadIdx.x;
    int32_t num_active = coff_ptr[nv];
    if (e >= num_active) return;
    atomicAdd(&cnt[cidx[e]], 1);
}

__global__ void k_scatter_csr(
    const int32_t* __restrict__ coff,
    const int32_t* __restrict__ cidx,
    int32_t* __restrict__ wpos,
    int32_t* __restrict__ csr_idx,
    int32_t nv)
{
    int v = blockIdx.x * BS + threadIdx.x;
    if (v >= nv) return;
    int s = coff[v], e = coff[v + 1];
    for (int i = s; i < e; i++) {
        int src = cidx[i];
        int pos = atomicAdd(&wpos[src], 1);
        csr_idx[pos] = v;
    }
}



__global__ void k_init_hubs(double* __restrict__ h, int32_t nv) {
    int i = blockIdx.x * BS + threadIdx.x;
    if (i < nv) h[i] = 1.0 / (double)nv;
}

__global__ void k_spmv(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ idx,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t nv)
{
    int v = blockIdx.x * BS + threadIdx.x;
    if (v >= nv) return;
    int s = off[v], e = off[v + 1];
    double acc = 0.0;
    for (int i = s; i < e; i++)
        acc += x[idx[i]];
    y[v] = acc;
}

__global__ void k_norm(double* __restrict__ a, const double* __restrict__ mx, int32_t n) {
    int i = blockIdx.x * BS + threadIdx.x;
    if (i < n) {
        double m = *mx;
        if (m > 0.0) a[i] /= m;
    }
}

__global__ void k_norm_diff(
    double* __restrict__ nw,
    const double* __restrict__ old_h,
    const double* __restrict__ mx,
    double* __restrict__ df,
    int32_t n)
{
    int i = blockIdx.x * BS + threadIdx.x;
    if (i < n) {
        double m = *mx;
        if (m > 0.0) nw[i] /= m;
        df[i] = fabs(nw[i] - old_h[i]);
    }
}



struct Cache : Cacheable {
    int32_t* compact_off = nullptr;
    int32_t* compact_idx = nullptr;
    int32_t* csr_off = nullptr;
    int32_t* csr_idx = nullptr;
    int32_t* counts = nullptr;
    int32_t* wpos = nullptr;
    double* scratch = nullptr;
    double* dbuf = nullptr;
    double* sc = nullptr;
    void* ctmp = nullptr;

    int32_t nv_cap = 0;
    int32_t ne_cap = 0;
    size_t cub_cap = 0;

    void ensure(int32_t nv, int32_t ne, size_t cub_bytes) {
        if (nv > nv_cap) {
            if (compact_off) cudaFree(compact_off);
            if (csr_off) cudaFree(csr_off);
            if (counts) cudaFree(counts);
            if (wpos) cudaFree(wpos);
            if (scratch) cudaFree(scratch);
            if (dbuf) cudaFree(dbuf);
            if (sc) cudaFree(sc);

            cudaMalloc(&compact_off, (size_t)(nv + 1) * sizeof(int32_t));
            cudaMalloc(&csr_off, (size_t)(nv + 1) * sizeof(int32_t));
            cudaMalloc(&counts, (size_t)(nv + 1) * sizeof(int32_t));
            cudaMalloc(&wpos, (size_t)(nv + 1) * sizeof(int32_t));
            cudaMalloc(&scratch, (size_t)nv * sizeof(double));
            cudaMalloc(&dbuf, (size_t)nv * sizeof(double));
            cudaMalloc(&sc, 2 * sizeof(double));
            nv_cap = nv;
        }
        if (ne > ne_cap) {
            if (compact_idx) cudaFree(compact_idx);
            if (csr_idx) cudaFree(csr_idx);

            cudaMalloc(&compact_idx, (size_t)ne * sizeof(int32_t));
            cudaMalloc(&csr_idx, (size_t)ne * sizeof(int32_t));
            ne_cap = ne;
        }
        if (cub_bytes > cub_cap) {
            if (ctmp) cudaFree(ctmp);
            cudaMalloc(&ctmp, cub_bytes);
            cub_cap = cub_bytes;
        }
    }

    ~Cache() override {
        if (compact_off) cudaFree(compact_off);
        if (compact_idx) cudaFree(compact_idx);
        if (csr_off) cudaFree(csr_off);
        if (csr_idx) cudaFree(csr_idx);
        if (counts) cudaFree(counts);
        if (wpos) cudaFree(wpos);
        if (scratch) cudaFree(scratch);
        if (dbuf) cudaFree(dbuf);
        if (sc) cudaFree(sc);
        if (ctmp) cudaFree(ctmp);
    }
};

}  

HitsResultDouble hits_mask(const graph32_t& graph,
                           double* hubs,
                           double* authorities,
                           double epsilon,
                           std::size_t max_iterations,
                           bool has_initial_hubs_guess,
                           bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;

    if (nv == 0) {
        return {0, false, 0.0};
    }

    const int32_t* off = graph.offsets;
    const int32_t* idx = graph.indices;
    const uint32_t* mask = graph.edge_mask;

    
    size_t s1 = 0, s2 = 0, s3 = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, s1, (int32_t*)nullptr, (int32_t*)nullptr, nv + 1);
    cub::DeviceReduce::Max(nullptr, s2, (double*)nullptr, (double*)nullptr, nv);
    cub::DeviceReduce::Sum(nullptr, s3, (double*)nullptr, (double*)nullptr, nv);
    size_t cub_bytes = s1;
    if (s2 > cub_bytes) cub_bytes = s2;
    if (s3 > cub_bytes) cub_bytes = s3;

    cache.ensure(nv, ne, cub_bytes);

    double tolerance = (double)nv * epsilon;
    int Gv = (nv + BS - 1) / BS;
    int Ge = (ne + BS - 1) / BS;
    double* mx_ptr = cache.sc;
    double* sm_ptr = cache.sc + 1;

    
    k_count_active<<<Gv, BS>>>(off, mask, cache.counts, nv);
    cudaMemset(&cache.counts[nv], 0, sizeof(int32_t));
    {
        size_t t = cub_bytes;
        cub::DeviceScan::ExclusiveSum(cache.ctmp, t, cache.counts, cache.compact_off, nv + 1);
    }
    k_scatter_csc<<<Gv, BS>>>(off, idx, mask, cache.compact_off, cache.compact_idx, nv);

    
    cudaMemset(cache.counts, 0, (size_t)(nv + 1) * sizeof(int32_t));
    k_histogram<<<Ge, BS>>>(cache.compact_idx, cache.counts, cache.compact_off, nv);
    cudaMemset(&cache.counts[nv], 0, sizeof(int32_t));
    {
        size_t t = cub_bytes;
        cub::DeviceScan::ExclusiveSum(cache.ctmp, t, cache.counts, cache.csr_off, nv + 1);
    }
    cudaMemcpy(cache.wpos, cache.csr_off, (size_t)(nv + 1) * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    k_scatter_csr<<<Gv, BS>>>(cache.compact_off, cache.compact_idx, cache.wpos, cache.csr_idx, nv);

    
    if (!has_initial_hubs_guess) {
        k_init_hubs<<<Gv, BS>>>(hubs, nv);
    } else {
        size_t t = cub_bytes;
        cub::DeviceReduce::Sum(cache.ctmp, t, hubs, sm_ptr, nv);
        k_norm<<<Gv, BS>>>(hubs, sm_ptr, nv);
    }

    
    double* cur = hubs;
    double* nxt = cache.scratch;
    double hd = 0.0;
    std::size_t it = 0;
    bool cv = false;

    for (it = 0; it < max_iterations; it++) {
        
        k_spmv<<<Gv, BS>>>(cache.compact_off, cache.compact_idx, cur, authorities, nv);

        
        k_spmv<<<Gv, BS>>>(cache.csr_off, cache.csr_idx, authorities, nxt, nv);

        
        size_t t = cub_bytes;
        cub::DeviceReduce::Max(cache.ctmp, t, nxt, mx_ptr, nv);
        k_norm_diff<<<Gv, BS>>>(nxt, cur, mx_ptr, cache.dbuf, nv);

        
        t = cub_bytes;
        cub::DeviceReduce::Max(cache.ctmp, t, authorities, mx_ptr, nv);
        k_norm<<<Gv, BS>>>(authorities, mx_ptr, nv);

        
        t = cub_bytes;
        cub::DeviceReduce::Sum(cache.ctmp, t, cache.dbuf, sm_ptr, nv);
        cudaMemcpy(&hd, sm_ptr, sizeof(double), cudaMemcpyDeviceToHost);

        
        double* tmp = cur; cur = nxt; nxt = tmp;

        if (hd < tolerance) { cv = true; it++; break; }
    }

    
    if (cur != hubs)
        cudaMemcpy(hubs, cur, (size_t)nv * sizeof(double), cudaMemcpyDeviceToDevice);

    
    if (normalize) {
        size_t t = cub_bytes;
        cub::DeviceReduce::Sum(cache.ctmp, t, hubs, sm_ptr, nv);
        k_norm<<<Gv, BS>>>(hubs, sm_ptr, nv);
        t = cub_bytes;
        cub::DeviceReduce::Sum(cache.ctmp, t, authorities, sm_ptr, nv);
        k_norm<<<Gv, BS>>>(authorities, sm_ptr, nv);
    }

    cudaDeviceSynchronize();

    return {it, cv, hd};
}

}  
