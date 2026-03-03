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
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <math_constants.h>
#include <cstdint>
#include <vector>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* dvw = nullptr; int64_t dvw_cap = 0;
    int32_t* d_ca = nullptr; int64_t d_ca_cap = 0;
    int32_t* d_cb = nullptr; int64_t d_cb_cap = 0;
    float* dcw = nullptr; int64_t dcw_cap = 0;
    int32_t* dt1 = nullptr; int64_t dt1_cap = 0;
    int32_t* dt2 = nullptr; int64_t dt2_cap = 0;
    uint32_t* drk = nullptr; int64_t drk_cap = 0;
    float* dco = nullptr; int64_t dco_cap = 0;
    float* dmw = nullptr; int64_t dmw_cap = 0;
    double* d_mod = nullptr; int64_t d_mod_cap = 0;

    template <typename T>
    static void ensure_buf(T*& ptr, int64_t& cap, int64_t needed) {
        if (cap < needed) {
            if (ptr) cudaFree(ptr);
            cudaMalloc(&ptr, (size_t)needed * sizeof(T));
            cap = needed;
        }
    }

    void ensure(int32_t nv, int32_t ne) {
        ensure_buf(dvw, dvw_cap, (int64_t)nv);
        ensure_buf(d_ca, d_ca_cap, (int64_t)nv);
        ensure_buf(d_cb, d_cb_cap, (int64_t)nv);
        ensure_buf(dcw, dcw_cap, (int64_t)nv);
        ensure_buf(dt1, dt1_cap, (int64_t)nv);
        ensure_buf(dt2, dt2_cap, (int64_t)nv);
        ensure_buf(drk, drk_cap, (int64_t)nv);
        ensure_buf(dco, dco_cap, (int64_t)ne);
        ensure_buf(dmw, dmw_cap, (int64_t)ne);
        ensure_buf(d_mod, d_mod_cap, (int64_t)2);
    }

    ~Cache() override {
        if (dvw) cudaFree(dvw);
        if (d_ca) cudaFree(d_ca);
        if (d_cb) cudaFree(d_cb);
        if (dcw) cudaFree(dcw);
        if (dt1) cudaFree(dt1);
        if (dt2) cudaFree(dt2);
        if (drk) cudaFree(drk);
        if (dco) cudaFree(dco);
        if (dmw) cudaFree(dmw);
        if (d_mod) cudaFree(d_mod);
    }
};

#define HT_SIZE 512
#define BT 128
#define WPB (BT / 32)
#define EMPTY_KEY (-1)

__device__ __forceinline__ uint32_t hash_u32(int32_t k) {
    uint32_t x = (uint32_t)k;
    x = ((x >> 16) ^ x) * 0x45d9f3bU;
    x = ((x >> 16) ^ x) * 0x45d9f3bU;
    x = (x >> 16) ^ x;
    return x;
}



__global__ void k_vw(const int32_t* __restrict__ off, const float* __restrict__ w,
    float* __restrict__ vw, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        float s = 0.0f;
        for (int32_t i = off[v]; i < off[v+1]; i++) s += w[i];
        vw[v] = s;
    }
}

__global__ void k_hash_init(int32_t* c, int32_t n, uint32_t seed) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        uint32_t x = (uint32_t)v ^ seed;
        x *= 0x85ebca6bU; x ^= x >> 13;
        x *= 0xc2b2ae35U; x ^= x >> 16;
        x ^= seed * 2654435761U; x *= 0xcc9e2d51U; x ^= x >> 15;
        c[v] = (int32_t)(x % (uint32_t)n);
    }
}

__global__ void k_scatter_add(const float* __restrict__ vw, const int32_t* __restrict__ c,
    float* __restrict__ cw, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) atomicAdd(&cw[c[v]], vw[v]);
}

__global__ void k_cooc(const int32_t* __restrict__ off, const int32_t* __restrict__ ind,
    const int32_t* __restrict__ c, float* __restrict__ co, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        int32_t cv = c[v];
        for (int32_t e = off[v]; e < off[v+1]; e++)
            if (c[ind[e]] == cv) co[e] += 1.0f;
    }
}

__global__ void k_ecg_wt(const float* __restrict__ ow, const float* __restrict__ co,
    float* __restrict__ mw, float minw, float invens, int32_t m) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < m) mw[e] = minw + (ow[e] - minw) * co[e] * invens;
}

__global__ void k_rng(uint32_t* k, int32_t n, uint32_t s) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t x = (uint32_t)i ^ s;
        x *= 0x85ebca6bU; x ^= x >> 13;
        x *= 0xc2b2ae35U; x ^= x >> 16;
        x ^= s * 2654435761U; x *= 0xcc9e2d51U; x ^= x >> 15;
        k[i] = x;
    }
}

__global__ void k_gather(const int32_t* __restrict__ m, const int32_t* __restrict__ nl,
    int32_t* __restrict__ r, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) r[v] = nl[m[v]];
}


__global__ void k_fused_cw_si(const int32_t* __restrict__ off, const int32_t* __restrict__ ind,
    const float* __restrict__ wt, const int32_t* __restrict__ c, const float* __restrict__ vw,
    float* __restrict__ cw, double* __restrict__ d_mod, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    double ls = 0.0;
    if (v < n) {
        atomicAdd(&cw[c[v]], vw[v]);
        int32_t cv = c[v];
        for (int32_t e = off[v]; e < off[v+1]; e++)
            if (c[ind[e]] == cv) ls += (double)wt[e];
    }
    typedef cub::BlockReduce<double, BT> BR;
    __shared__ typename BR::TempStorage tmp;
    double bs = BR(tmp).Sum(ls);
    if (threadIdx.x == 0 && bs != 0.0) atomicAdd(&d_mod[0], bs);
}


__global__ void k_sum_sq(const float* __restrict__ cw, int32_t n, double* __restrict__ d_mod) {
    double ls = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        double x = (double)cw[i];
        ls += x * x;
    }
    typedef cub::BlockReduce<double, BT> BR;
    __shared__ typename BR::TempStorage tmp;
    double bs = BR(tmp).Sum(ls);
    if (threadIdx.x == 0 && bs != 0.0) atomicAdd(&d_mod[1], bs);
}


__global__ void k_sum_internal(const int32_t* __restrict__ off, const int32_t* __restrict__ ind,
    const float* __restrict__ wt, const int32_t* __restrict__ c,
    double* __restrict__ d_mod, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    double ls = 0.0;
    if (v < n) {
        int32_t cv = c[v];
        for (int32_t e = off[v]; e < off[v+1]; e++)
            if (c[ind[e]] == cv) ls += (double)wt[e];
    }
    typedef cub::BlockReduce<double, BT> BR;
    __shared__ typename BR::TempStorage tmp;
    double bs = BR(tmp).Sum(ls);
    if (threadIdx.x == 0 && bs != 0.0) atomicAdd(&d_mod[0], bs);
}


__global__ void k_move(
    const int32_t* __restrict__ off, const int32_t* __restrict__ ind,
    const float* __restrict__ wt, const int32_t* __restrict__ c,
    const float* __restrict__ cw, const float* __restrict__ vw,
    float tw, float res, int32_t n, int32_t* __restrict__ nc, bool ud)
{
    int wg = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int wl = threadIdx.x >> 5;
    if (wg >= n) return;
    int v = wg;

    extern __shared__ char smem[];
    int32_t* hk = (int32_t*)(smem) + wl * HT_SIZE * 2;
    float* hv = (float*)(hk + HT_SIZE);

    for (int i = lane; i < HT_SIZE; i += 32) { hk[i] = EMPTY_KEY; hv[i] = 0.0f; }
    __syncwarp();

    int32_t s = off[v], e = off[v+1], mc = c[v];
    float kv = vw[v], sl = 0.0f;

    if (s == e) { if (lane == 0) nc[v] = mc; return; }

    for (int32_t i = s + lane; i < e; i += 32) {
        int32_t u = ind[i]; int32_t cc = c[u]; float w = wt[i];
        if (u == v) sl += w;
        uint32_t slot = hash_u32(cc) & (HT_SIZE-1);
        for (int p = 0; p < HT_SIZE; p++) {
            int32_t ok = atomicCAS(&hk[slot], EMPTY_KEY, cc);
            if (ok == EMPTY_KEY || ok == cc) { atomicAdd(&hv[slot], w); break; }
            slot = (slot+1) & (HT_SIZE-1);
        }
    }
    for (int o = 16; o > 0; o >>= 1) sl += __shfl_xor_sync(0xffffffff, sl, o);
    __syncwarp();

    float kvot = 0.0f;
    for (int i = lane; i < HT_SIZE; i += 32) if (hk[i] == mc) kvot = hv[i];
    for (int o = 16; o > 0; o >>= 1) kvot += __shfl_xor_sync(0xffffffff, kvot, o);
    float kvo = kvot - sl;

    float so = cw[mc], it = 1.0f/tw, it2 = it*it;
    float bd = -CUDART_INF_F; int32_t bc = mc;

    for (int i = lane; i < HT_SIZE; i += 32) {
        if (hk[i] != EMPTY_KEY) {
            int32_t cc = hk[i]; float kvc = hv[i], sc = cw[cc];
            float ka, sd;
            if (cc == mc) { ka = kvc - sl; sd = kv; }
            else { ka = kvc; sd = sc - so + kv; }
            float d = 2.0f * ((ka - kvo)*it - res*kv*sd*it2);
            if (d > bd || (d == bd && cc < bc)) { bd = d; bc = cc; }
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        float od = __shfl_xor_sync(0xffffffff, bd, o);
        int32_t oc = __shfl_xor_sync(0xffffffff, bc, o);
        if (od > bd || (od == bd && oc < bc)) { bd = od; bc = oc; }
    }
    if (lane == 0) {
        if (bd > 0.0f && bc != mc && ((bc > mc) == ud)) nc[v] = bc;
        else nc[v] = mc;
    }
}


__global__ void k_edge_keys32(const int32_t* __restrict__ off, const int32_t* __restrict__ ind,
    const int32_t* __restrict__ c, int32_t* __restrict__ ek, int32_t n, int32_t K) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        int32_t cv = c[v];
        for (int32_t e = off[v]; e < off[v+1]; e++)
            ek[e] = cv * K + c[ind[e]];
    }
}

__global__ void k_edge_keys64(const int32_t* __restrict__ off, const int32_t* __restrict__ ind,
    const int32_t* __restrict__ c, int64_t* __restrict__ ek, int32_t n, int32_t K) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        int32_t cv = c[v];
        for (int32_t e = off[v]; e < off[v+1]; e++)
            ek[e] = (int64_t)cv * K + c[ind[e]];
    }
}




static float compute_mod(
    const int32_t* doff, const int32_t* dind, const float* dwt,
    const int32_t* dc, const float* dvw, float* dcw,
    float tw, float res, int32_t n, int32_t mc, double* d_mod)
{
    cudaMemset(dcw, 0, mc * sizeof(float));
    cudaMemset(d_mod, 0, 2 * sizeof(double));

    int blk = (n + BT - 1) / BT;
    k_fused_cw_si<<<blk, BT>>>(doff, dind, dwt, dc, dvw, dcw, d_mod, n);

    int sqblk = (mc + BT - 1) / BT;
    if (sqblk > 256) sqblk = 256;
    if (sqblk < 1) sqblk = 1;
    k_sum_sq<<<sqblk, BT>>>(dcw, mc, d_mod);

    double h_mod[2];
    cudaMemcpy(h_mod, d_mod, 2 * sizeof(double), cudaMemcpyDeviceToHost);

    double t = (double)tw;
    return (float)(h_mod[0] / t - res * h_mod[1] / (t * t));
}


static float compute_mod_with_cw(
    const int32_t* doff, const int32_t* dind, const float* dwt,
    const int32_t* dc, const float* dcw,
    float tw, float res, int32_t n, int32_t mc, double* d_mod)
{
    cudaMemset(d_mod, 0, 2 * sizeof(double));

    int blk = (n + BT - 1) / BT;
    k_sum_internal<<<blk, BT>>>(doff, dind, dwt, dc, d_mod, n);

    int sqblk = (mc + BT - 1) / BT;
    if (sqblk > 256) sqblk = 256;
    if (sqblk < 1) sqblk = 1;
    k_sum_sq<<<sqblk, BT>>>(dcw, mc, d_mod);

    double h_mod[2];
    cudaMemcpy(h_mod, d_mod, 2 * sizeof(double), cudaMemcpyDeviceToHost);

    double t = (double)tw;
    return (float)(h_mod[0] / t - res * h_mod[1] / (t * t));
}

static void compute_cw(const int32_t* c, const float* vw, float* cw, int32_t n, int32_t mc) {
    cudaMemset(cw, 0, mc*sizeof(float));
    k_scatter_add<<<(n+BT-1)/BT, BT>>>(vw, c, cw, n);
}

static int32_t renum(int32_t* c, int32_t n, int32_t* t1, int32_t* t2) {
    thrust::device_ptr<int32_t> cp(c), p1(t1), p2(t2);
    thrust::copy(cp, cp+n, p1);
    thrust::sort(p1, p1+n);
    auto end = thrust::unique_copy(p1, p1+n, p2);
    int32_t K = (int32_t)(end-p2);
    thrust::lower_bound(p2, p2+K, cp, cp+n, p1);
    thrust::copy(p1, p1+n, cp);
    return K;
}

static void gen_perm(int32_t* p, int32_t n, uint32_t s, uint32_t* rk) {
    k_rng<<<(n+BT-1)/BT, BT>>>(rk, n, s);
    thrust::device_ptr<int32_t> pp(p); thrust::sequence(pp, pp+n);
    thrust::sort_by_key(thrust::device_ptr<uint32_t>(rk), thrust::device_ptr<uint32_t>(rk)+n, pp);
}


static void louvain_move(
    const int32_t* doff, const int32_t* dind, const float* dwt,
    int32_t*& d_ca, int32_t*& d_cb,
    float* dcw, const float* dvw,
    float tw, float res, float thr, int32_t n, int32_t mc, double* d_mod,
    int max_iter)
{
    if (n == 0) return;
    int blocks = (n * 32 + BT - 1) / BT;
    size_t smem = WPB * HT_SIZE * 2 * sizeof(int32_t);

    float nQ = compute_mod(doff, dind, dwt, d_ca, dvw, dcw, tw, res, n, mc, d_mod);
    float cQ = nQ - 1.0f;
    bool ud = true;

    for (int it = 0; it < max_iter; it++) {
        if (nQ <= cQ + thr) break;
        cQ = nQ;

        k_move<<<blocks, BT, smem>>>(doff, dind, dwt, d_ca, dcw, dvw, tw, res, n, d_cb, ud);
        std::swap(d_ca, d_cb);

        nQ = compute_mod(doff, dind, dwt, d_ca, dvw, dcw, tw, res, n, mc, d_mod);
        ud = !ud;
    }
}


struct CG { int32_t* off; int32_t* ind; float* wt; int32_t nv, ne; };

static CG coarsen(const int32_t* doff, const int32_t* dind, const float* dwt,
    const int32_t* dc, int32_t n, int32_t m, int32_t K) {

    int blk = (n + BT - 1) / BT;

    
    bool use32 = ((int64_t)K * K < (int64_t)INT32_MAX);

    int32_t nu;
    int32_t* dst_arr;
    float* uv;

    if (use32) {
        int32_t* ek; float* ev;
        cudaMalloc(&ek, (size_t)m*sizeof(int32_t));
        cudaMalloc(&ev, (size_t)m*sizeof(float));
        cudaMemcpy(ev, dwt, (size_t)m*sizeof(float), cudaMemcpyDeviceToDevice);
        k_edge_keys32<<<blk, BT>>>(doff, dind, dc, ek, n, K);
        thrust::device_ptr<int32_t> ekp(ek);
        thrust::device_ptr<float> evp(ev);
        thrust::sort_by_key(ekp, ekp+m, evp);

        int32_t* uk; float* uvt;
        cudaMalloc(&uk, (size_t)m*sizeof(int32_t));
        cudaMalloc(&uvt, (size_t)m*sizeof(float));
        auto ep = thrust::reduce_by_key(ekp, ekp+m, evp,
            thrust::device_ptr<int32_t>(uk), thrust::device_ptr<float>(uvt));
        nu = (int32_t)(ep.first - thrust::device_ptr<int32_t>(uk));
        cudaFree(ek); cudaFree(ev);

        int32_t* src;
        cudaMalloc(&src, (size_t)nu*sizeof(int32_t));
        cudaMalloc(&dst_arr, (size_t)nu*sizeof(int32_t));
        {
            thrust::device_ptr<int32_t> ukp(uk), sp(src), dp(dst_arr);
            int32_t Kv = K;
            thrust::transform(ukp, ukp+nu,
                thrust::make_zip_iterator(sp, dp),
                [Kv] __device__ (int32_t k){
                    return thrust::make_tuple(k/Kv, k%Kv);
                });
        }
        cudaFree(uk);
        uv = uvt;

        int32_t* noff; cudaMalloc(&noff, (K+1)*sizeof(int32_t));
        thrust::lower_bound(thrust::device,
            thrust::device_ptr<int32_t>(src), thrust::device_ptr<int32_t>(src)+nu,
            thrust::counting_iterator<int32_t>(0), thrust::counting_iterator<int32_t>(K+1),
            thrust::device_ptr<int32_t>(noff));
        cudaFree(src);

        CG r; r.off=noff; r.ind=dst_arr; r.wt=uv; r.nv=K; r.ne=nu;
        return r;
    } else {
        int64_t* ek; float* ev;
        cudaMalloc(&ek, (size_t)m*sizeof(int64_t));
        cudaMalloc(&ev, (size_t)m*sizeof(float));
        cudaMemcpy(ev, dwt, (size_t)m*sizeof(float), cudaMemcpyDeviceToDevice);
        k_edge_keys64<<<blk, BT>>>(doff, dind, dc, ek, n, K);
        thrust::device_ptr<int64_t> ekp(ek);
        thrust::device_ptr<float> evp(ev);
        thrust::sort_by_key(ekp, ekp+m, evp);

        int64_t* uk; float* uvt;
        cudaMalloc(&uk, (size_t)m*sizeof(int64_t));
        cudaMalloc(&uvt, (size_t)m*sizeof(float));
        auto ep = thrust::reduce_by_key(ekp, ekp+m, evp,
            thrust::device_ptr<int64_t>(uk), thrust::device_ptr<float>(uvt));
        nu = (int32_t)(ep.first - thrust::device_ptr<int64_t>(uk));
        cudaFree(ek); cudaFree(ev);

        int32_t* src;
        cudaMalloc(&src, (size_t)nu*sizeof(int32_t));
        cudaMalloc(&dst_arr, (size_t)nu*sizeof(int32_t));
        {
            thrust::device_ptr<int64_t> ukp(uk);
            thrust::device_ptr<int32_t> sp(src), dp(dst_arr);
            int32_t Kv = K;
            thrust::transform(ukp, ukp+nu,
                thrust::make_zip_iterator(sp, dp),
                [Kv] __device__ (int64_t k){
                    return thrust::make_tuple((int32_t)(k/Kv),(int32_t)(k%Kv));
                });
        }
        cudaFree(uk);
        uv = uvt;

        int32_t* noff; cudaMalloc(&noff, (K+1)*sizeof(int32_t));
        thrust::lower_bound(thrust::device,
            thrust::device_ptr<int32_t>(src), thrust::device_ptr<int32_t>(src)+nu,
            thrust::counting_iterator<int32_t>(0), thrust::counting_iterator<int32_t>(K+1),
            thrust::device_ptr<int32_t>(noff));
        cudaFree(src);

        CG r; r.off=noff; r.ind=dst_arr; r.wt=uv; r.nv=K; r.ne=nu;
        return r;
    }
}


struct LR { int32_t levels; float mod; };

static LR full_louvain(
    const int32_t* do0, const int32_t* di0, const float* dw0,
    int32_t* dout, int32_t nv, int32_t ne,
    int ml, float thr, float res, float tw, bool rng, uint32_t seed,
    int32_t* d_ca, int32_t* d_cb, float* dvw, float* dcw,
    int32_t* dt1, int32_t* dt2, uint32_t* drk, double* d_mod)
{
    int32_t cn=nv, cm=ne;
    const int32_t* co=do0; const int32_t* ci=di0; const float* cwt=dw0;
    std::vector<int32_t*> ao,ai; std::vector<float*> aw;
    std::vector<int32_t*> lc; std::vector<int32_t> ls;
    float bm=-1.0f; int level=0;
    int32_t* orig_ca = d_ca, *orig_cb = d_cb;

    while (level < ml) {
        d_ca = orig_ca; d_cb = orig_cb;

        if (rng) gen_perm(d_ca, cn, seed+level*13337, drk);
        else { thrust::device_ptr<int32_t> cp(d_ca); thrust::sequence(cp, cp+cn); }

        k_vw<<<(cn+BT-1)/BT, BT>>>(co, cwt, dvw, cn);
        compute_cw(d_ca, dvw, dcw, cn, cn);

        louvain_move(co, ci, cwt, d_ca, d_cb, dcw, dvw, tw, res, thr, cn, cn, d_mod, 100);

        int32_t K = renum(d_ca, cn, dt1, dt2);

        int32_t* lcomm; cudaMalloc(&lcomm, (size_t)cn*sizeof(int32_t));
        cudaMemcpy(lcomm, d_ca, (size_t)cn*sizeof(int32_t), cudaMemcpyDeviceToDevice);
        lc.push_back(lcomm); ls.push_back(cn);

        if (K >= cn) { level++; break; }

        compute_cw(d_ca, dvw, dcw, cn, K);
        float nm = compute_mod_with_cw(co, ci, cwt, d_ca, dcw, tw, res, cn, K, d_mod);
        if (level > 0 && nm - bm < thr) { level++; break; }
        bm = nm;

        CG cg = coarsen(co, ci, cwt, d_ca, cn, cm, K);
        ao.push_back(cg.off); ai.push_back(cg.ind); aw.push_back(cg.wt);
        co=cg.off; ci=cg.ind; cwt=cg.wt; cn=cg.nv; cm=cg.ne;
        level++;
    }

    if (lc.empty()) {
        thrust::device_ptr<int32_t> op(dout); thrust::sequence(op, op+nv);
    } else {
        cudaMemcpy(dout, lc[0], (size_t)ls[0]*sizeof(int32_t), cudaMemcpyDeviceToDevice);
        for (int l=1; l<(int)lc.size(); l++) {
            k_gather<<<(nv+BT-1)/BT, BT>>>(dout, lc[l], orig_cb, nv);
            cudaMemcpy(dout, orig_cb, (size_t)nv*sizeof(int32_t), cudaMemcpyDeviceToDevice);
        }
    }

    
    k_vw<<<(nv+BT-1)/BT, BT>>>(do0, dw0, dvw, nv);
    thrust::device_ptr<int32_t> op(dout);
    int32_t mx = thrust::reduce(op, op+nv, 0, thrust::maximum<int32_t>());
    int32_t nc = mx+1;
    float* fcw = dcw; bool acw = (nc > nv);
    if (acw) cudaMalloc(&fcw, (size_t)nc*sizeof(float));
    float fm = compute_mod(do0, di0, dw0, dout, dvw, fcw, tw, res, nv, nc, d_mod);
    if (acw) cudaFree(fcw);

    for (auto p:lc) cudaFree(p);
    for (auto p:ao) cudaFree(p);
    for (auto p:ai) cudaFree(p);
    for (auto p:aw) cudaFree(p);

    LR r; r.levels=level; r.mod=fm;
    return r;
}

}  



ecg_result_float_t ecg_seg(const graph32_t& graph,
                           const float* edge_weights,
                           int32_t* clusters,
                           float min_weight,
                           std::size_t ensemble_size,
                           std::size_t max_level,
                           float threshold,
                           float resolution) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    const int32_t* doff = graph.offsets;
    const int32_t* dind = graph.indices;
    const float* dew = edge_weights;
    int32_t* dout = clusters;
    float minw = min_weight;
    int esz = (int)ensemble_size;
    int ml = (int)max_level;
    float thr = threshold;
    float res = resolution;

    if (nv == 0) {
        return ecg_result_float_t{0, 0.0f};
    }

    cache.ensure(nv, ne);

    float* dvw = cache.dvw;
    int32_t* d_ca = cache.d_ca;
    int32_t* d_cb = cache.d_cb;
    float* dcw = cache.dcw;
    float* dco = cache.dco;
    float* dmw = cache.dmw;
    int32_t* dt1 = cache.dt1;
    int32_t* dt2 = cache.dt2;
    uint32_t* drk = cache.drk;
    double* d_mod = cache.d_mod;

    int blk = (nv + BT - 1) / BT, eblk = (ne + BT - 1) / BT;

    k_vw<<<blk, BT>>>(doff, dew, dvw, nv);
    thrust::device_ptr<float> vwp(dvw);
    float tw = thrust::reduce(vwp, vwp + nv, 0.0f);

    cudaMemset(dco, 0, (size_t)ne * sizeof(float));

    
    int32_t* eca, *ecb;
    for (int i = 0; i < esz; i++) {
        eca = d_ca; ecb = d_cb;
        k_hash_init<<<blk, BT>>>(eca, nv, (uint32_t)(i * 1000003 + 42));
        compute_cw(eca, dvw, dcw, nv, nv);
        louvain_move(doff, dind, dew, eca, ecb, dcw, dvw, tw, res, thr, nv, nv, d_mod, 100);
        k_cooc<<<blk, BT>>>(doff, dind, eca, dco, nv);
    }

    k_ecg_wt<<<eblk, BT>>>(dew, dco, dmw, minw, 1.0f / (float)esz, ne);

    
    k_vw<<<blk, BT>>>(doff, dmw, dvw, nv);
    thrust::device_ptr<float> vwp2(dvw);
    float tmw = thrust::reduce(vwp2, vwp2 + nv, 0.0f);

    LR lr = full_louvain(doff, dind, dmw, dout, nv, ne, ml, thr, res, tmw,
        true, 9999, d_ca, d_cb, dvw, dcw, dt1, dt2, drk, d_mod);

    
    k_vw<<<blk, BT>>>(doff, dew, dvw, nv);
    thrust::device_ptr<int32_t> op(dout);
    int32_t mx = thrust::reduce(op, op + nv, 0, thrust::maximum<int32_t>());
    int32_t nc = mx + 1;
    float* fcw = dcw; bool acw = (nc > nv);
    if (acw) cudaMalloc(&fcw, (size_t)nc * sizeof(float));
    float fm = compute_mod(doff, dind, dew, dout, dvw, fcw, tw, res, nv, nc, d_mod);
    if (acw) cudaFree(fcw);

    return ecg_result_float_t{(std::size_t)lr.levels, fm};
}

}  
