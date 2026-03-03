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
#include <cstddef>
#include <climits>

namespace aai {

namespace {

__device__ __forceinline__ bool is_rem(const uint32_t* bm, int32_t v) {
    return (bm[v >> 5] >> (v & 31)) & 1;
}





__global__ void deg_block(
    const int32_t* __restrict__ off, const int32_t* __restrict__ ind,
    int32_t* __restrict__ deg, int32_t mult, int32_t s0, int32_t s1) {
    int v = s0 + blockIdx.x;
    if (v >= s1) return;
    int32_t rs = off[v], re = off[v+1];
    int c = 0;
    for (int32_t i = rs + threadIdx.x; i < re; i += blockDim.x)
        if (ind[i] != v) c++;
    for (int d = 16; d > 0; d >>= 1) c += __shfl_down_sync(0xFFFFFFFF, c, d);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ int ws[8];
    if (lane == 0) ws[warp] = c;
    __syncthreads();
    if (threadIdx.x == 0) {
        int t = 0; for (int w = 0; w < (int)(blockDim.x>>5); w++) t += ws[w];
        deg[v] = (mult == 2 && t > 0x3FFFFFFF) ? 0x7FFFFFFF : t * mult;
    }
}

__global__ void deg_warp(
    const int32_t* __restrict__ off, const int32_t* __restrict__ ind,
    int32_t* __restrict__ deg, int32_t mult, int32_t s0, int32_t s1) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int v = s0 + wid;
    if (v >= s1) return;
    int32_t s = off[v], e = off[v+1];
    int c = 0;
    for (int32_t i = s + lane; i < e; i += 32)
        if (ind[i] != v) c++;
    for (int d = 16; d > 0; d >>= 1) c += __shfl_down_sync(0xFFFFFFFF, c, d);
    if (lane == 0) deg[v] = (mult == 2 && c > 0x3FFFFFFF) ? 0x7FFFFFFF : c * mult;
}

__global__ void deg_zero(int32_t* __restrict__ deg, int32_t s0, int32_t s1) {
    int v = s0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (v < s1) deg[v] = 0;
}





__global__ void init_frontier_kernel(
    const int32_t* __restrict__ deg, int32_t* __restrict__ rem,
    int32_t* __restrict__ fr, int32_t* __restrict__ fs,
    int32_t nv, int32_t k) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv) return;
    if (deg[v] < k) { rem[v] = 1; fr[atomicAdd(fs, 1)] = v; }
    else rem[v] = 0;
}

__global__ void peel_kernel(
    const int32_t* __restrict__ off, const int32_t* __restrict__ ind,
    int32_t* __restrict__ deg, int32_t* __restrict__ rem,
    const int32_t* __restrict__ fr, int32_t fsz,
    int32_t* __restrict__ nfr, int32_t* __restrict__ nfs,
    int32_t k, int32_t dec) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fsz) return;
    int32_t v = fr[idx], s = off[v], e = off[v+1];
    for (int32_t i = s; i < e; i++) {
        int32_t u = ind[i];
        if (u == v || rem[u]) continue;
        int32_t od = atomicSub(&deg[u], dec);
        if (od >= k && (od - dec) < k)
            if (atomicCAS(&rem[u], 0, 1) == 0)
                nfr[atomicAdd(nfs, 1)] = u;
    }
}





__global__ void build_bm_from_rem(const int32_t* __restrict__ rem, uint32_t* __restrict__ bm, int32_t nv) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    bool r = (v < nv) ? (rem[v] != 0) : false;
    unsigned b = __ballot_sync(0xFFFFFFFF, r);
    if ((threadIdx.x & 31) == 0 && (v >> 5) < ((nv + 31) >> 5))
        bm[v >> 5] = b;
}

__global__ void build_bm_from_cn(const int32_t* __restrict__ cn, uint32_t* __restrict__ bm, int32_t nv, int32_t k) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    bool r = (v < nv) ? (cn[v] < k) : false;
    unsigned b = __ballot_sync(0xFFFFFFFF, r);
    if ((threadIdx.x & 31) == 0 && (v >> 5) < ((nv + 31) >> 5))
        bm[v >> 5] = b;
}





__global__ void extract_block(
    const int32_t* __restrict__ off, const int32_t* __restrict__ ind,
    const uint32_t* __restrict__ bm,
    int32_t* __restrict__ es, int32_t* __restrict__ ed,
    int32_t* __restrict__ counter,
    int32_t s0, int32_t s1) {
    int v = s0 + blockIdx.x;
    if (v >= s1 || is_rem(bm, v)) return;
    int32_t rs = off[v], re = off[v+1];

    int c = 0;
    for (int32_t i = rs + threadIdx.x; i < re; i += blockDim.x)
        if (!is_rem(bm, ind[i])) c++;
    for (int d = 16; d > 0; d >>= 1) c += __shfl_down_sync(0xFFFFFFFF, c, d);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ int ws[8];
    if (lane == 0) ws[warp] = c;
    __syncthreads();

    __shared__ int tc; __shared__ int32_t gb;
    if (threadIdx.x == 0) {
        int t = 0; for (int w = 0; w < (int)(blockDim.x>>5); w++) t += ws[w];
        tc = t;
        if (t > 0) gb = atomicAdd(counter, t);
    }
    __syncthreads();
    if (tc == 0) return;

    __shared__ int wp;
    if (threadIdx.x == 0) wp = 0;
    __syncthreads();

    for (int32_t chunk = rs; chunk < re; chunk += blockDim.x) {
        int32_t i = chunk + threadIdx.x;
        bool keep = false; int32_t u = -1;
        if (i < re) { u = ind[i]; keep = !is_rem(bm, u); }
        unsigned ballot = __ballot_sync(0xFFFFFFFF, keep);
        __shared__ int wbase[8];
        if (lane == 0 && __popc(ballot) > 0)
            wbase[warp] = atomicAdd(&wp, __popc(ballot));
        __syncthreads();
        if (keep) {
            unsigned lm = (1u << lane) - 1;
            int o = wbase[warp] + __popc(ballot & lm);
            es[gb + o] = v; ed[gb + o] = u;
        }
        __syncthreads();
    }
}





__global__ void extract_warp(
    const int32_t* __restrict__ off, const int32_t* __restrict__ ind,
    const uint32_t* __restrict__ bm,
    int32_t* __restrict__ es, int32_t* __restrict__ ed,
    int32_t* __restrict__ counter,
    int32_t s0, int32_t s1) {
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int wib = threadIdx.x >> 5;
    int nw = blockDim.x >> 5;
    int v = s0 + wid;

    int c = 0;
    bool active = (v < s1 && !is_rem(bm, v));
    if (active) {
        int32_t s = off[v], e = off[v+1];
        for (int32_t i = s + lane; i < e; i += 32)
            if (!is_rem(bm, ind[i])) c++;
        for (int d = 16; d > 0; d >>= 1) c += __shfl_down_sync(0xFFFFFFFF, c, d);
    }
    int wc = __shfl_sync(0xFFFFFFFF, c, 0);

    __shared__ int wcnts[8], woffs[8];
    __shared__ int32_t bbase;

    if (lane == 0) wcnts[wib] = wc;
    __syncthreads();

    if (threadIdx.x == 0) {
        int bt = 0;
        for (int w = 0; w < nw; w++) { woffs[w] = bt; bt += wcnts[w]; }
        if (bt > 0) bbase = atomicAdd(counter, bt);
        else bbase = 0;
    }
    __syncthreads();

    if (!active || wc == 0) return;

    int32_t base = bbase + woffs[wib];
    int32_t s = off[v], e = off[v+1];
    int32_t wp = base;
    for (int32_t i = s + lane; i < e; i += 32) {
        int32_t u = ind[i];
        bool keep = !is_rem(bm, u);
        unsigned ballot = __ballot_sync(0xFFFFFFFF, (i < e) ? keep : false);
        if (keep) {
            unsigned lm = (1u << lane) - 1;
            es[wp + __popc(ballot & lm)] = v;
            ed[wp + __popc(ballot & lm)] = u;
        }
        wp += __popc(ballot);
    }
}





struct Cache : Cacheable {
    uint32_t* bm = nullptr;
    int32_t* deg = nullptr;
    int32_t* rem = nullptr;
    int32_t* fr0 = nullptr;
    int32_t* fr1 = nullptr;
    int32_t* fs0 = nullptr;
    int32_t* fs1 = nullptr;
    int32_t* ctr = nullptr;

    int32_t bm_cap = 0;
    int32_t deg_cap = 0;
    int32_t rem_cap = 0;
    int32_t fr0_cap = 0;
    int32_t fr1_cap = 0;

    void ensure(int32_t nv) {
        int32_t bm_words = (nv + 31) / 32;
        if (bm_cap < bm_words) {
            if (bm) cudaFree(bm);
            cudaMalloc(&bm, bm_words * sizeof(uint32_t));
            bm_cap = bm_words;
        }
        if (deg_cap < nv) {
            if (deg) cudaFree(deg);
            cudaMalloc(&deg, nv * sizeof(int32_t));
            deg_cap = nv;
        }
        if (rem_cap < nv) {
            if (rem) cudaFree(rem);
            cudaMalloc(&rem, nv * sizeof(int32_t));
            rem_cap = nv;
        }
        if (fr0_cap < nv) {
            if (fr0) cudaFree(fr0);
            cudaMalloc(&fr0, nv * sizeof(int32_t));
            fr0_cap = nv;
        }
        if (fr1_cap < nv) {
            if (fr1) cudaFree(fr1);
            cudaMalloc(&fr1, nv * sizeof(int32_t));
            fr1_cap = nv;
        }
        if (!fs0) cudaMalloc(&fs0, sizeof(int32_t));
        if (!fs1) cudaMalloc(&fs1, sizeof(int32_t));
        if (!ctr) cudaMalloc(&ctr, sizeof(int32_t));
    }

    ~Cache() override {
        if (bm) cudaFree(bm);
        if (deg) cudaFree(deg);
        if (rem) cudaFree(rem);
        if (fr0) cudaFree(fr0);
        if (fr1) cudaFree(fr1);
        if (fs0) cudaFree(fs0);
        if (fs1) cudaFree(fs1);
        if (ctr) cudaFree(ctr);
    }
};

}  

std::size_t k_core_seg(const graph32_t& graph,
                       std::size_t k,
                       int degree_type,
                       const int32_t* core_numbers,
                       int32_t* edge_srcs,
                       int32_t* edge_dsts,
                       std::size_t max_edges) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    const int32_t* d_off = graph.offsets;
    const int32_t* d_ind = graph.indices;

    if (k > INT32_MAX || nv == 0) {
        return 0;
    }

    const auto& seg_vec = graph.segment_offsets.value();
    int32_t seg[5] = {seg_vec[0], seg_vec[1], seg_vec[2], seg_vec[3], seg_vec[4]};

    cache.ensure(nv);

    uint32_t* d_bm = cache.bm;

    if (core_numbers != nullptr) {
        int t = ((nv + 31) / 32) * 32;
        if (t > 0) build_bm_from_cn<<<(t + 255) / 256, 256>>>(core_numbers, d_bm, nv, (int32_t)k);
    } else {
        int32_t mult = (degree_type == 2) ? 2 : 1;
        int32_t* dd = cache.deg;
        int32_t* dr = cache.rem;
        int32_t* df[2] = {cache.fr0, cache.fr1};
        int32_t* ds[2] = {cache.fs0, cache.fs1};

        
        {
            int n;
            n = seg[1] - seg[0]; if (n > 0) deg_block<<<n, 256>>>(d_off, d_ind, dd, mult, seg[0], seg[1]);
            n = seg[3] - seg[1]; if (n > 0) {
                int wpb = 256 / 32;
                deg_warp<<<(n + wpb - 1) / wpb, 256>>>(d_off, d_ind, dd, mult, seg[1], seg[3]);
            }
            n = seg[4] - seg[3]; if (n > 0) deg_zero<<<(n + 255) / 256, 256>>>(dd, seg[3], seg[4]);
        }

        cudaMemset(ds[0], 0, 4);
        if (nv > 0) init_frontier_kernel<<<(nv + 255) / 256, 256>>>(dd, dr, df[0], ds[0], nv, (int32_t)k);
        int32_t hfs; cudaMemcpy(&hfs, ds[0], 4, cudaMemcpyDeviceToHost);
        int cur = 0;
        while (hfs > 0) {
            int nx = 1 - cur;
            cudaMemset(ds[nx], 0, 4);
            peel_kernel<<<(hfs + 255) / 256, 256>>>(d_off, d_ind, dd, dr, df[cur], hfs, df[nx], ds[nx], (int32_t)k, mult);
            cudaMemcpy(&hfs, ds[nx], 4, cudaMemcpyDeviceToHost);
            cur = nx;
        }

        {
            int t = ((nv + 31) / 32) * 32;
            if (t > 0) build_bm_from_rem<<<(t + 255) / 256, 256>>>(dr, d_bm, nv);
        }
    }

    
    cudaMemset(cache.ctr, 0, 4);
    {
        int n;
        n = seg[1] - seg[0]; if (n > 0) extract_block<<<n, 256>>>(d_off, d_ind, d_bm, edge_srcs, edge_dsts, cache.ctr, seg[0], seg[1]);
        n = seg[3] - seg[1]; if (n > 0) {
            int wpb = 256 / 32;
            extract_warp<<<(n + wpb - 1) / wpb, 256>>>(d_off, d_ind, d_bm, edge_srcs, edge_dsts, cache.ctr, seg[1], seg[3]);
        }
    }

    int32_t total;
    cudaMemcpy(&total, cache.ctr, 4, cudaMemcpyDeviceToHost);

    return (std::size_t)total;
}

}  
