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
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>
#include <cstdint>
#include <vector>
#include <utility>

namespace aai {

namespace {

#define BS 256
#define HT_SIZE 512
#define LM_BS 128
#define LM_WPB (LM_BS / 32)

struct Cache : Cacheable {
    
    double* d_vw = nullptr;
    double* d_cw = nullptr;
    int32_t* d_cl = nullptr;
    int32_t* d_best = nullptr;
    int32_t* d_cp = nullptr;
    int32_t* d_tmp = nullptr;
    uint32_t* d_rk = nullptr;
    int64_t nv_cap = 0;

    
    double* d_ecg = nullptr;
    double* d_mw = nullptr;
    uint64_t* d_ek = nullptr;
    double* d_ew2 = nullptr;
    uint64_t* d_rkeys = nullptr;
    double* d_rw = nullptr;
    int32_t* d_ns = nullptr;
    int32_t* d_nd = nullptr;
    int64_t ne_cap = 0;

    
    double* d_buf = nullptr;
    int32_t* d_moved = nullptr;
    bool small_alloc = false;

    void ensure(int32_t nv, int32_t ne) {
        if (!small_alloc) {
            cudaMalloc(&d_buf, 8);
            cudaMalloc(&d_moved, 4);
            small_alloc = true;
        }
        if (nv_cap < nv) {
            if (d_vw) cudaFree(d_vw);
            if (d_cw) cudaFree(d_cw);
            if (d_cl) cudaFree(d_cl);
            if (d_best) cudaFree(d_best);
            if (d_cp) cudaFree(d_cp);
            if (d_tmp) cudaFree(d_tmp);
            if (d_rk) cudaFree(d_rk);
            cudaMalloc(&d_vw, (int64_t)nv * 8);
            cudaMalloc(&d_cw, (int64_t)nv * 8);
            cudaMalloc(&d_cl, (int64_t)nv * 4);
            cudaMalloc(&d_best, (int64_t)nv * 4);
            cudaMalloc(&d_cp, (int64_t)nv * 4);
            cudaMalloc(&d_tmp, (int64_t)nv * 4);
            cudaMalloc(&d_rk, (int64_t)nv * 4);
            nv_cap = nv;
        }
        if (ne_cap < ne) {
            if (d_ecg) cudaFree(d_ecg);
            if (d_mw) cudaFree(d_mw);
            if (d_ek) cudaFree(d_ek);
            if (d_ew2) cudaFree(d_ew2);
            if (d_rkeys) cudaFree(d_rkeys);
            if (d_rw) cudaFree(d_rw);
            if (d_ns) cudaFree(d_ns);
            if (d_nd) cudaFree(d_nd);
            cudaMalloc(&d_ecg, (int64_t)ne * 8);
            cudaMalloc(&d_mw, (int64_t)ne * 8);
            cudaMalloc(&d_ek, (int64_t)ne * 8);
            cudaMalloc(&d_ew2, (int64_t)ne * 8);
            cudaMalloc(&d_rkeys, (int64_t)ne * 8);
            cudaMalloc(&d_rw, (int64_t)ne * 8);
            cudaMalloc(&d_ns, (int64_t)ne * 4);
            cudaMalloc(&d_nd, (int64_t)ne * 4);
            ne_cap = ne;
        }
    }

    ~Cache() override {
        if (d_vw) cudaFree(d_vw);
        if (d_cw) cudaFree(d_cw);
        if (d_cl) cudaFree(d_cl);
        if (d_best) cudaFree(d_best);
        if (d_cp) cudaFree(d_cp);
        if (d_tmp) cudaFree(d_tmp);
        if (d_rk) cudaFree(d_rk);
        if (d_ecg) cudaFree(d_ecg);
        if (d_mw) cudaFree(d_mw);
        if (d_ek) cudaFree(d_ek);
        if (d_ew2) cudaFree(d_ew2);
        if (d_rkeys) cudaFree(d_rkeys);
        if (d_rw) cudaFree(d_rw);
        if (d_ns) cudaFree(d_ns);
        if (d_nd) cudaFree(d_nd);
        if (d_buf) cudaFree(d_buf);
        if (d_moved) cudaFree(d_moved);
    }
};





__global__ void k_vertex_weights(const int32_t* __restrict__ off, const double* __restrict__ w,
    double* __restrict__ vw, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    double s = 0; int a = off[v], b = off[v+1];
    for (int i = a; i < b; i++) s += w[i];
    vw[v] = s;
}

__global__ void k_identity(int32_t* d, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = i;
}

__global__ void k_random_keys(uint32_t* k, int32_t n, uint32_t seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t x = (uint32_t)i ^ seed;
    x ^= x >> 16; x *= 0x45d9f3bU;
    x ^= x >> 16; x *= 0x45d9f3bU;
    x ^= x >> 16;
    k[i] = x;
}

__global__ void k_cluster_weights(const int32_t* __restrict__ cl, const double* __restrict__ vw,
    double* __restrict__ cw, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    atomicAdd(&cw[cl[v]], vw[v]);
}

__global__ void k_local_move(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    const double* __restrict__ w, const int32_t* __restrict__ cl_in,
    int32_t* __restrict__ cl_out,
    const double* __restrict__ vw, const double* __restrict__ cw,
    double tw, double res, bool up_down, int32_t n, int32_t* __restrict__ moved)
{
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int v = blockIdx.x * LM_WPB + wid;
    if (v >= n) return;

    extern __shared__ char smem[];
    int32_t* mk = ((int32_t*)smem) + wid * HT_SIZE;
    double* mv = (double*)(((int32_t*)smem) + LM_WPB * HT_SIZE) + wid * HT_SIZE;

    for (int i = lane; i < HT_SIZE; i += 32) { mk[i] = -1; mv[i] = 0.0; }
    __syncwarp();

    int mc = cl_in[v];
    double k = vw[v];
    int s = off[v], e = off[v+1];

    for (int i = s + lane; i < e; i += 32) {
        int nb = idx[i];
        if (nb == v) continue;
        double wt = w[i];
        int c = cl_in[nb];
        uint32_t slot = ((uint32_t)c * 2654435761U) & (HT_SIZE - 1);
        for (int p = 0; p < HT_SIZE; p++) {
            int prev = atomicCAS(&mk[slot], -1, c);
            if (prev == -1 || prev == c) { atomicAdd(&mv[slot], wt); break; }
            slot = (slot + 1) & (HT_SIZE - 1);
        }
    }
    __syncwarp();

    double os = 0.0;
    for (int i = lane; i < HT_SIZE; i += 32) if (mk[i] == mc) os = mv[i];
    for (int o = 16; o > 0; o >>= 1) os += __shfl_xor_sync(0xffffffff, os, o);

    double ao = cw[mc];
    double im = 1.0 / tw, im2 = im * im;

    double bd = -1e30; int bc = 0x7fffffff;
    for (int i = lane; i < HT_SIZE; i += 32) {
        int c = mk[i];
        if (c == -1 || c == mc) continue;
        double ev = mv[i], an = cw[c];
        double d = 2.0 * ((ev - os) * im - res * (an * k - ao * k + k * k) * im2);
        if (d > bd || (d == bd && c < bc)) { bd = d; bc = c; }
    }
    for (int o = 16; o > 0; o >>= 1) {
        double od = __shfl_down_sync(0xffffffff, bd, o);
        int oc = __shfl_down_sync(0xffffffff, bc, o);
        if (od > bd || (od == bd && oc < bc)) { bd = od; bc = oc; }
    }
    bc = __shfl_sync(0xffffffff, bc, 0);
    bd = __shfl_sync(0xffffffff, bd, 0);

    if (lane == 0) {
        if (bd > 0.0 && ((bc > mc) == up_down)) {
            cl_out[v] = bc;
            atomicAdd(moved, 1);
        } else {
            cl_out[v] = mc;
        }
    }
}

__global__ void k_sum_internal(const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    const double* __restrict__ w, const int32_t* __restrict__ cl, double* __restrict__ r, int32_t n) {
    typedef cub::BlockReduce<double, BS> BR;
    __shared__ typename BR::TempStorage ts;
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    double s = 0;
    if (v < n) { int c = cl[v]; int a = off[v], b = off[v+1];
        for (int i = a; i < b; i++) if (cl[idx[i]] == c) s += w[i]; }
    double bs = BR(ts).Sum(s);
    if (threadIdx.x == 0) atomicAdd(r, bs);
}

__global__ void k_ecg_counts(const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    const int32_t* __restrict__ cl, double* __restrict__ cnt, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int c = cl[v]; int s = off[v], e = off[v+1];
    for (int i = s; i < e; i++) if (cl[idx[i]] == c) cnt[i] += 1.0;
}

__global__ void k_ecg_weights(const double* __restrict__ ow, const double* __restrict__ cnt,
    double* __restrict__ mw, double min_w, double inv_ens, int32_t ne) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ne) return;
    mw[e] = min_w + (ow[e] - min_w) * cnt[e] * inv_ens;
}

__global__ void k_flatten(int32_t* r, const int32_t* m, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) r[i] = m[r[i]];
}

__global__ void k_edge_keys(const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    const int32_t* __restrict__ cc, uint64_t* __restrict__ ek, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int32_t cv = cc[v]; int s = off[v], e = off[v+1];
    for (int i = s; i < e; i++)
        ek[i] = ((uint64_t)(uint32_t)cv << 32) | (uint64_t)(uint32_t)cc[idx[i]];
}

__global__ void k_extract(const uint64_t* k, int32_t* s, int32_t* d, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    s[i] = (int32_t)(k[i] >> 32); d[i] = (int32_t)(k[i] & 0xFFFFFFFF);
}

__global__ void k_count_row(const int32_t* src, int32_t* cnt, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicAdd(&cnt[src[i]], 1);
}




#define BLK(n) (((n)+BS-1)/BS)

static void gen_perm(int32_t* p, uint32_t* t, int32_t n, uint32_t seed) {
    k_random_keys<<<BLK(n), BS>>>(t, n, seed);
    k_identity<<<BLK(n), BS>>>(p, n);
    thrust::sort_by_key(thrust::device_ptr<uint32_t>(t), thrust::device_ptr<uint32_t>(t+n),
                         thrust::device_ptr<int32_t>(p));
}

static double get_total_weight(const double* w, int32_t ne) {
    return thrust::reduce(thrust::device_ptr<const double>(w), thrust::device_ptr<const double>(w+ne), 0.0);
}

static double compute_mod(const int32_t* off, const int32_t* idx, const double* w,
    const int32_t* cl, const double* cw, double tw, double res, int32_t n, double* d_buf) {
    cudaMemset(d_buf, 0, 8);
    k_sum_internal<<<BLK(n), BS>>>(off, idx, w, cl, d_buf, n);
    double h; cudaMemcpy(&h, d_buf, 8, cudaMemcpyDeviceToHost);
    auto sq = [] __device__ (double x) -> double { return x * x; };
    thrust::device_ptr<const double> dp(cw);
    double ssq = thrust::reduce(thrust::make_transform_iterator(dp, sq),
                                 thrust::make_transform_iterator(dp+n, sq), 0.0);
    return h / tw - res * ssq / (tw * tw);
}

static void recompute_cw(const int32_t* cl, const double* vw, double* cw, int32_t n) {
    cudaMemset(cw, 0, n * 8);
    k_cluster_weights<<<BLK(n), BS>>>(cl, vw, cw, n);
}

static double run_local_move(
    const int32_t* off, const int32_t* idx, const double* w,
    int32_t* cl, const double* vw, double* cw,
    double tw, double res, double thr, int32_t n,
    int32_t* best_cl, double* d_buf, int32_t* d_moved, int max_iter,
    bool check_modularity, int mod_interval, int32_t* alt_cl)
{
    size_t smem = LM_WPB * HT_SIZE * (4 + 8);
    int blocks = (n + LM_WPB - 1) / LM_WPB;
    recompute_cw(cl, vw, cw, n);

    double cur_Q = -1.0;
    if (check_modularity) {
        cur_Q = compute_mod(off, idx, w, cl, cw, tw, res, n, d_buf);
        cudaMemcpy(best_cl, cl, n*4, cudaMemcpyDeviceToDevice);
    }
    double prev_Q = cur_Q - 1.0;
    bool ud = true;
    int32_t* cur = cl;
    int32_t* nxt = alt_cl;

    for (int iter = 0; iter < max_iter; iter++) {
        cudaMemset(d_moved, 0, 4);
        k_local_move<<<blocks, LM_BS, smem>>>(off, idx, w, cur, nxt, vw, cw, tw, res, ud, n, d_moved);
        std::swap(cur, nxt);
        int32_t h_moved;
        cudaMemcpy(&h_moved, d_moved, 4, cudaMemcpyDeviceToHost);
        if (h_moved == 0) break;
        recompute_cw(cur, vw, cw, n);
        ud = !ud;

        if (check_modularity && ((iter + 1) % mod_interval == 0)) {
            double new_Q = compute_mod(off, idx, w, cur, cw, tw, res, n, d_buf);
            if (new_Q > cur_Q) {
                cur_Q = new_Q;
                cudaMemcpy(best_cl, cur, n*4, cudaMemcpyDeviceToDevice);
            }
            if (new_Q <= prev_Q + thr) break;
            prev_Q = new_Q;
        }
    }
    if (check_modularity) {
        cudaMemcpy(cl, best_cl, n*4, cudaMemcpyDeviceToDevice);
    } else if (cur != cl) {
        cudaMemcpy(cl, cur, n*4, cudaMemcpyDeviceToDevice);
    }
    return cur_Q;
}

struct GCSR { int32_t* off; int32_t* idx; double* w; int32_t nv, ne; };

static int32_t compact_cl(int32_t* cl, int32_t* cp, int32_t* tmp, int32_t n) {
    cudaMemcpy(tmp, cl, n*4, cudaMemcpyDeviceToDevice);
    thrust::device_ptr<int32_t> dt(tmp);
    thrust::sort(dt, dt+n);
    int32_t K = thrust::unique(dt, dt+n) - dt;
    thrust::lower_bound(dt, dt+K, thrust::device_ptr<int32_t>(cl),
                         thrust::device_ptr<int32_t>(cl+n), thrust::device_ptr<int32_t>(cp));
    return K;
}

static GCSR contract(const int32_t* off, const int32_t* idx, const double* w,
    const int32_t* cc, int32_t n, int32_t ne, int32_t K,
    uint64_t* ek, double* ew2, uint64_t* rk, double* rw, int32_t* ns, int32_t* nd)
{
    k_edge_keys<<<BLK(n), BS>>>(off, idx, cc, ek, n);
    cudaMemcpy(ew2, w, ne*8, cudaMemcpyDeviceToDevice);
    thrust::sort_by_key(thrust::device_ptr<uint64_t>(ek), thrust::device_ptr<uint64_t>(ek+ne),
                         thrust::device_ptr<double>(ew2));
    auto re = thrust::reduce_by_key(thrust::device_ptr<uint64_t>(ek), thrust::device_ptr<uint64_t>(ek+ne),
        thrust::device_ptr<double>(ew2), thrust::device_ptr<uint64_t>(rk), thrust::device_ptr<double>(rw));
    int32_t nne = re.first - thrust::device_ptr<uint64_t>(rk);
    k_extract<<<BLK(nne), BS>>>(rk, ns, nd, nne);
    int32_t* no; cudaMalloc(&no, (K+1)*4); cudaMemset(no, 0, (K+1)*4);
    k_count_row<<<BLK(nne), BS>>>(ns, no, nne);
    thrust::exclusive_scan(thrust::device_ptr<int32_t>(no), thrust::device_ptr<int32_t>(no+K+1),
                            thrust::device_ptr<int32_t>(no));
    double* fw; int32_t* fi;
    cudaMalloc(&fw, nne*8); cudaMalloc(&fi, nne*4);
    cudaMemcpy(fw, rw, nne*8, cudaMemcpyDeviceToDevice);
    cudaMemcpy(fi, nd, nne*4, cudaMemcpyDeviceToDevice);
    return {no, fi, fw, K, nne};
}

static void run_louvain(
    const int32_t* off, const int32_t* idx, const double* w,
    int32_t n, int32_t ne, int64_t ml, double thr, double res,
    bool ri, uint32_t seed, int32_t* out,
    int64_t* olv, double* omod,
    int32_t* d_cl, double* d_vw, double* d_cw, int32_t* d_best,
    int32_t* d_cp, int32_t* d_tmp, uint32_t* d_rk,
    double* d_buf, int32_t* d_moved,
    uint64_t* d_ek, double* d_ew2, uint64_t* d_rkeys, double* d_rw,
    int32_t* d_ns, int32_t* d_nd)
{
    double tw = get_total_weight(w, ne);
    int32_t cn = n, ce = ne;
    const int32_t* co = off; const int32_t* ci = idx; const double* cw = w;
    bool owns = false;
    std::vector<int32_t*> dendro;
    double bmod = -1.0; int64_t lv = 0;
    while (lv < ml && cn > 1) {
        if (ri) gen_perm(d_cl, d_rk, cn, seed + (uint32_t)lv * 7919);
        else k_identity<<<BLK(cn), BS>>>(d_cl, cn);
        k_vertex_weights<<<BLK(cn), BS>>>(co, cw, d_vw, cn);
        double mod = run_local_move(co, ci, cw, d_cl, d_vw, d_cw, tw, res, thr, cn,
                                     d_best, d_buf, d_moved, 100, true, 1, d_cp);
        if (mod <= bmod) break;
        bmod = mod;
        int32_t K = compact_cl(d_cl, d_cp, d_tmp, cn);
        int32_t* dl; cudaMalloc(&dl, cn*4);
        cudaMemcpy(dl, d_cp, cn*4, cudaMemcpyDeviceToDevice);
        dendro.push_back(dl);
        if (K >= cn || K <= 1) { lv++; break; }
        GCSR cg = contract(co, ci, cw, d_cp, cn, ce, K, d_ek, d_ew2, d_rkeys, d_rw, d_ns, d_nd);
        if (owns) { cudaFree((void*)co); cudaFree((void*)ci); cudaFree((void*)cw); }
        co = cg.off; ci = cg.idx; cw = cg.w; cn = cg.nv; ce = cg.ne;
        owns = true; lv++;
    }
    if (owns) { cudaFree((void*)co); cudaFree((void*)ci); cudaFree((void*)cw); }
    if (dendro.empty()) k_identity<<<BLK(n), BS>>>(out, n);
    else {
        cudaMemcpy(out, dendro[0], n*4, cudaMemcpyDeviceToDevice);
        for (size_t i = 1; i < dendro.size(); i++) k_flatten<<<BLK(n), BS>>>(out, dendro[i], n);
    }
    for (auto p : dendro) cudaFree(p);
    *olv = (int64_t)dendro.size(); *omod = bmod;
}

}  

ecg_result_double_t ecg_seg(const graph32_t& graph,
                            const double* edge_weights,
                            int32_t* clusters,
                            double min_weight,
                            std::size_t ensemble_size,
                            std::size_t max_level,
                            double threshold,
                            double resolution) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* off = graph.offsets;
    const int32_t* idx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;

    cache.ensure(nv, ne);

    cudaMemset(cache.d_ecg, 0, (int64_t)ne * 8);
    double tw = get_total_weight(edge_weights, ne);
    k_vertex_weights<<<BLK(nv), BS>>>(off, edge_weights, cache.d_vw, nv);

    
    for (std::size_t i = 0; i < ensemble_size; i++) {
        gen_perm(cache.d_cl, cache.d_rk, nv, (uint32_t)(i * 12347 + 42));
        run_local_move(off, idx, edge_weights, cache.d_cl, cache.d_vw, cache.d_cw, tw, resolution, threshold, nv,
                       cache.d_best, cache.d_buf, cache.d_moved, 100, true, 1, cache.d_tmp);
        k_ecg_counts<<<BLK(nv), BS>>>(off, idx, cache.d_cl, cache.d_ecg, nv);
    }

    
    k_ecg_weights<<<BLK(ne), BS>>>(edge_weights, cache.d_ecg, cache.d_mw, min_weight, 1.0 / (double)ensemble_size, ne);

    
    int64_t h_lv; double h_mod;
    run_louvain(off, idx, cache.d_mw, nv, ne, (int64_t)max_level, threshold, resolution,
                true, (uint32_t)(ensemble_size * 54321 + 99), clusters, &h_lv, &h_mod,
                cache.d_cl, cache.d_vw, cache.d_cw, cache.d_best, cache.d_cp, cache.d_tmp, cache.d_rk,
                cache.d_buf, cache.d_moved, cache.d_ek, cache.d_ew2, cache.d_rkeys, cache.d_rw, cache.d_ns, cache.d_nd);

    
    k_vertex_weights<<<BLK(nv), BS>>>(off, edge_weights, cache.d_vw, nv);
    recompute_cw(clusters, cache.d_vw, cache.d_cw, nv);
    double fm = compute_mod(off, idx, edge_weights, clusters, cache.d_cw, tw, resolution, nv, cache.d_buf);

    return ecg_result_double_t{(std::size_t)h_lv, fm};
}

}  
