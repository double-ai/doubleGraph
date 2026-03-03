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
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/extrema.h>
#include <thrust/equal.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

namespace aai {

namespace {

__device__ __host__ inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline int bits_for(int32_t n) {
    if (n <= 1) return 1;
    int b = 0; int32_t v = n - 1;
    while (v > 0) { b++; v >>= 1; }
    return b;
}

__global__ void gen_keys_kernel(uint64_t* k, int32_t n, uint64_t seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) k[i] = splitmix64(seed ^ ((uint64_t)i * 0x517CC1B727220A95ULL));
}

__global__ void compute_vw_kernel(const int32_t* __restrict__ off,
    const double* __restrict__ ew, double* __restrict__ vw, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) { double s=0; for(int e=off[v]; e<off[v+1]; e++) s+=ew[e]; vw[v]=s; }
}

__global__ void accum_cw_kernel(const int32_t* __restrict__ c,
    const double* __restrict__ vw, double* __restrict__ cw, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) atomicAdd(&cw[c[v]], vw[v]);
}

__global__ void expand_encode_kernel(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    const double* __restrict__ ew, const int32_t* __restrict__ comm,
    int64_t* __restrict__ keys, double* __restrict__ vals,
    double* __restrict__ old_sum, double* __restrict__ self_loop,
    int32_t n, int shift) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int32_t mc = comm[v];
    double os = 0, sl = 0;
    int64_t v_shifted = (int64_t)v << shift;
    for (int e = off[v]; e < off[v+1]; e++) {
        int32_t u = idx[e];
        int32_t dc = __ldg(&comm[u]);
        keys[e] = v_shifted | (int64_t)dc;
        vals[e] = ew[e];
        if (u == v) sl += ew[e];
        else if (dc == mc) os += ew[e];
    }
    old_sum[v] = os;
    self_loop[v] = sl;
}

__global__ void decode_delta_kernel(
    const int64_t* __restrict__ akeys, const double* __restrict__ avals,
    const int32_t* __restrict__ comm, const double* __restrict__ vw,
    const double* __restrict__ cw, const double* __restrict__ os,
    const double* __restrict__ sl,
    double tew, double res, int shift, int64_t mask,
    int32_t* __restrict__ out_src, int32_t* __restrict__ out_comm,
    double* __restrict__ out_delta, int32_t na) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= na) return;
    int64_t key = akeys[i];
    int32_t v = (int32_t)(key >> shift);
    int32_t c = (int32_t)(key & mask);
    double w = avals[i];
    int32_t mc = comm[v];
    double kv = vw[v], m = tew;
    double ns = w;
    if (c == mc) ns -= sl[v];
    double an = cw[c], ao = cw[mc];
    out_src[i] = v;
    out_comm[i] = c;
    out_delta[i] = 2.0 * ((ns - os[v]) / m - res * kv * (an - ao + kv) / (m * m));
}

__global__ void find_best_kernel(
    const int32_t* __restrict__ ac, const double* __restrict__ ad,
    const int32_t* __restrict__ seg, const int32_t* __restrict__ comm,
    bool up_down, int32_t* __restrict__ nc, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int s = seg[v], e = seg[v+1];
    int32_t mc = comm[v], bc = -1;
    double bd = 0.0;
    for (int i = s; i < e; i++) {
        double d = ad[i]; int32_t c = ac[i];
        if (d > bd || (d == bd && d > 0.0 && c < bc)) { bd = d; bc = c; }
    }
    if (bd > 0.0 && bc >= 0) {
        nc[v] = ((bc > mc) == up_down) ? bc : mc;
    } else nc[v] = mc;
}

__global__ void modularity_kernel(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    const double* __restrict__ wt, const int32_t* __restrict__ comm,
    double* __restrict__ ps, int32_t n) {
    extern __shared__ double sd[];
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    double s = 0;
    if (v < n) { int32_t mc=comm[v]; for(int e=off[v]; e<off[v+1]; e++) if(comm[idx[e]]==mc) s+=wt[e]; }
    sd[threadIdx.x] = s;
    __syncthreads();
    for (int i = blockDim.x/2; i > 0; i >>= 1) {
        if (threadIdx.x < i) sd[threadIdx.x] += sd[threadIdx.x+i];
        __syncthreads();
    }
    if (threadIdx.x == 0) ps[blockIdx.x] = sd[0];
}

struct SqOp { __device__ double operator()(double x) const { return x*x; } };

double compute_mod(const int32_t* off, const int32_t* idx, const double* wt,
    const int32_t* comm, const double* cw, int32_t n, int32_t ncw,
    double tew, double res, double* ps, int gn) {
    if (n == 0 || tew == 0.0) return 0.0;
    modularity_kernel<<<gn, BLOCK_SIZE, BLOCK_SIZE*sizeof(double)>>>(off, idx, wt, comm, ps, n);
    double si = thrust::reduce(thrust::device_pointer_cast(ps),
                               thrust::device_pointer_cast(ps+gn), 0.0);
    double sq = thrust::transform_reduce(thrust::device_pointer_cast(cw),
        thrust::device_pointer_cast(cw+ncw), SqOp(), 0.0, thrust::plus<double>());
    return si / tew - res * sq / (tew * tew);
}

__global__ void relabel_kernel(int32_t* d, const int32_t* uv, int32_t n, int32_t nu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t c = d[i]; int lo=0, hi=nu;
    while(lo<hi) { int m=(lo+hi)/2; if(uv[m]<c) lo=m+1; else hi=m; }
    d[i] = lo;
}

__global__ void map_edges_encode_kernel(const int32_t* off, const int32_t* idx,
    const double* ew, const int32_t* comm,
    int64_t* keys, double* vals, int32_t n, int shift) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int32_t cv = comm[v];
    int64_t cv_shifted = (int64_t)cv << shift;
    for (int e = off[v]; e < off[v+1]; e++) {
        keys[e] = cv_shifted | (int64_t)comm[idx[e]];
        vals[e] = ew[e];
    }
}

struct Cache : Cacheable {
    int64_t *keys1 = nullptr, *keys2 = nullptr;
    double *vals1 = nullptr, *vals2 = nullptr;
    int64_t *red_keys = nullptr;
    double *red_vals = nullptr;
    int32_t *agg_src = nullptr, *agg_comm = nullptr, *new_comm = nullptr;
    double *agg_delta = nullptr, *old_sum = nullptr, *self_loop = nullptr;
    int32_t *seg_off = nullptr;
    double *partials = nullptr;
    void *sort_temp = nullptr;
    size_t sort_temp_size = 0;
    int64_t *ctr_keys1 = nullptr, *ctr_keys2 = nullptr;
    double *ctr_vals1 = nullptr, *ctr_vals2 = nullptr;
    void *ctr_sort_temp = nullptr;
    size_t ctr_sort_temp_size = 0;
    int64_t *ctr_red_keys = nullptr;
    double *ctr_red_vals = nullptr;

    int32_t alloc_n = 0, alloc_e = 0;

    void ensure(int32_t n, int32_t ne) {
        if (n <= alloc_n && ne <= alloc_e) return;
        cleanup();
        alloc_n = n; alloc_e = ne;
        cudaMalloc(&keys1, ne * 8); cudaMalloc(&keys2, ne * 8);
        cudaMalloc(&vals1, ne * 8); cudaMalloc(&vals2, ne * 8);
        cudaMalloc(&red_keys, ne * 8); cudaMalloc(&red_vals, ne * 8);
        cudaMalloc(&agg_src, ne * 4); cudaMalloc(&agg_comm, ne * 4);
        cudaMalloc(&agg_delta, ne * 8);
        cudaMalloc(&new_comm, n * 4);
        cudaMalloc(&old_sum, n * 8); cudaMalloc(&self_loop, n * 8);
        cudaMalloc(&seg_off, (n+1) * 4);
        cudaMalloc(&partials, ((n+BLOCK_SIZE-1)/BLOCK_SIZE) * 8);
        cudaMalloc(&ctr_keys1, ne * 8); cudaMalloc(&ctr_keys2, ne * 8);
        cudaMalloc(&ctr_vals1, ne * 8); cudaMalloc(&ctr_vals2, ne * 8);
        cudaMalloc(&ctr_red_keys, ne * 8); cudaMalloc(&ctr_red_vals, ne * 8);

        sort_temp_size = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, sort_temp_size, keys1, keys2, vals1, vals2, ne);
        cudaMalloc(&sort_temp, sort_temp_size);
        ctr_sort_temp_size = sort_temp_size;
        cudaMalloc(&ctr_sort_temp, ctr_sort_temp_size);
    }

    template<typename T> static void cfree(T*& p) { if(p){cudaFree(p);p=nullptr;} }
    void cleanup() {
        cfree(keys1); cfree(keys2); cfree(vals1); cfree(vals2);
        cfree(red_keys); cfree(red_vals);
        cfree(agg_src); cfree(agg_comm); cfree(agg_delta);
        cfree(new_comm); cfree(old_sum); cfree(self_loop); cfree(seg_off); cfree(partials);
        cfree(ctr_keys1); cfree(ctr_keys2); cfree(ctr_vals1); cfree(ctr_vals2);
        cfree(ctr_sort_temp); cfree(ctr_red_keys); cfree(ctr_red_vals);
        cfree(sort_temp);
        alloc_n = alloc_e = 0;
    }
    ~Cache() override { cleanup(); }
};

int32_t reduce_by_key_preallocated(int64_t* sorted_keys, double* sorted_vals, int32_t ne,
    int64_t* out_keys, double* out_vals) {
    auto r = thrust::reduce_by_key(
        thrust::device_pointer_cast(sorted_keys),
        thrust::device_pointer_cast(sorted_keys + ne),
        thrust::device_pointer_cast(sorted_vals),
        thrust::device_pointer_cast(out_keys),
        thrust::device_pointer_cast(out_vals));
    return (int32_t)(r.first - thrust::device_pointer_cast(out_keys));
}

bool local_moving_iter(
    const int32_t* off, const int32_t* idx, const double* ew,
    int32_t* comm, double* vw, double* cw,
    int32_t n, int32_t ne, double tew, double res, bool ud, int shift, int end_bit,
    Cache& ws) {
    if (n == 0 || ne == 0) return false;
    const int B = BLOCK_SIZE;
    int gn = (n+B-1)/B;
    int64_t mask = ((int64_t)1 << shift) - 1;

    expand_encode_kernel<<<gn, B>>>(off, idx, ew, comm,
        ws.keys1, ws.vals1, ws.old_sum, ws.self_loop, n, shift);

    cub::DeviceRadixSort::SortPairs(ws.sort_temp, ws.sort_temp_size,
        ws.keys1, ws.keys2, ws.vals1, ws.vals2, ne, 0, end_bit);

    int32_t na = reduce_by_key_preallocated(ws.keys2, ws.vals2, ne,
        ws.red_keys, ws.red_vals);

    int ga = (na+B-1)/B;
    decode_delta_kernel<<<ga, B>>>(ws.red_keys, ws.red_vals, comm, vw, cw,
        ws.old_sum, ws.self_loop, tew, res, shift, mask,
        ws.agg_src, ws.agg_comm, ws.agg_delta, na);

    thrust::lower_bound(
        thrust::device_pointer_cast(ws.agg_src),
        thrust::device_pointer_cast(ws.agg_src + na),
        thrust::counting_iterator<int32_t>(0),
        thrust::counting_iterator<int32_t>(n + 1),
        thrust::device_pointer_cast(ws.seg_off));

    find_best_kernel<<<gn, B>>>(ws.agg_comm, ws.agg_delta, ws.seg_off,
        comm, ud, ws.new_comm, n);

    bool any = !thrust::equal(thrust::device_pointer_cast(comm),
        thrust::device_pointer_cast(comm + n),
        thrust::device_pointer_cast(ws.new_comm));

    cudaMemcpy(comm, ws.new_comm, n * 4, cudaMemcpyDeviceToDevice);
    return any;
}

void louvain_impl(
    const int32_t* off, const int32_t* idx, const double* wt,
    int32_t n, int32_t ne, int32_t* out,
    size_t ml, double thr, double res, bool rng, uint64_t seed,
    int* olv, double* omod, Cache& ws) {
    if (n == 0) { *olv=0; *omod=0; return; }
    double tew = thrust::reduce(thrust::device_pointer_cast(wt),
                                thrust::device_pointer_cast(wt+ne), 0.0);
    if (tew == 0.0) {
        thrust::sequence(thrust::device_pointer_cast(out), thrust::device_pointer_cast(out+n));
        *olv=0; *omod=0; return;
    }

    thrust::device_vector<int32_t> co(off, off+n+1), ci(idx, idx+ne);
    thrust::device_vector<double> cwt(wt, wt+ne);
    int32_t cn=n, ce=ne;
    std::vector<thrust::device_vector<int32_t>> dendro;
    double bmod=-1.0;
    const int B=BLOCK_SIZE;

    thrust::device_vector<int32_t> comm(n), bcomm(n);
    thrust::device_vector<double> vw(n), cw(n);

    for (size_t lv = 0; lv < ml; lv++) {
        comm.resize(cn); bcomm.resize(cn); vw.resize(cn); cw.resize(cn);
        int gn = (cn+B-1)/B;
        int shift = bits_for(cn);
        int end_bit = 2 * shift;

        int32_t* cp = thrust::raw_pointer_cast(comm.data());
        double* vwp = thrust::raw_pointer_cast(vw.data());
        double* cwp = thrust::raw_pointer_cast(cw.data());
        const int32_t* cop = thrust::raw_pointer_cast(co.data());
        const int32_t* cip = thrust::raw_pointer_cast(ci.data());
        const double* cwtp = thrust::raw_pointer_cast(cwt.data());

        if (rng) {
            thrust::device_vector<uint64_t> rk(cn);
            gen_keys_kernel<<<gn, B>>>(thrust::raw_pointer_cast(rk.data()), cn,
                splitmix64(seed ^ ((uint64_t)lv * 0xDEADBEEFULL)));
            thrust::device_vector<int32_t> ti(cn);
            thrust::sequence(ti.begin(), ti.end());
            thrust::sort_by_key(rk.begin(), rk.end(), ti.begin());
            thrust::scatter(thrust::counting_iterator<int32_t>(0),
                           thrust::counting_iterator<int32_t>(cn), ti.begin(), comm.begin());
        } else {
            thrust::sequence(comm.begin(), comm.begin()+cn);
        }

        compute_vw_kernel<<<gn, B>>>(cop, cwtp, vwp, cn);
        thrust::fill(cw.begin(), cw.begin()+cn, 0.0);
        accum_cw_kernel<<<gn, B>>>(cp, vwp, cwp, cn);
        cudaDeviceSynchronize();

        double nQ = compute_mod(cop, cip, cwtp, cp, cwp, cn, cn, tew, res,
            ws.partials, gn);
        double cQ = nQ - 1.0;
        bool ud = true;
        thrust::copy(comm.begin(), comm.begin()+cn, bcomm.begin());
        int mi = 100;

        while (nQ > cQ + thr && mi-- > 0) {
            cQ = nQ;
            bool moved = local_moving_iter(cop, cip, cwtp, cp, vwp, cwp,
                cn, ce, tew, res, ud, shift, end_bit, ws);
            if (!moved) break;
            thrust::fill(cw.begin(), cw.begin()+cn, 0.0);
            accum_cw_kernel<<<gn, B>>>(cp, vwp, cwp, cn);
            cudaDeviceSynchronize();
            ud = !ud;
            nQ = compute_mod(cop, cip, cwtp, cp, cwp, cn, cn, tew, res,
                ws.partials, gn);
            if (nQ > cQ) thrust::copy(comm.begin(), comm.begin()+cn, bcomm.begin());
        }

        if (cQ <= bmod) break;
        bmod = cQ;
        thrust::copy(bcomm.begin(), bcomm.begin()+cn, comm.begin());
        dendro.push_back(thrust::device_vector<int32_t>(comm.begin(), comm.begin()+cn));

        
        thrust::device_vector<int32_t> cs(comm.begin(), comm.begin()+cn);
        thrust::sort(cs.begin(), cs.end());
        auto ue = thrust::unique(cs.begin(), cs.end());
        int32_t nc = (int32_t)(ue-cs.begin());
        cs.resize(nc);
        if (nc >= cn) break;

        relabel_kernel<<<gn, B>>>(cp, thrust::raw_pointer_cast(cs.data()), cn, nc);
        relabel_kernel<<<gn, B>>>(thrust::raw_pointer_cast(dendro.back().data()),
            thrust::raw_pointer_cast(cs.data()), cn, nc);
        cudaDeviceSynchronize();

        
        int ctr_shift = bits_for(nc);
        int ctr_end_bit = 2 * ctr_shift;
        map_edges_encode_kernel<<<gn, B>>>(cop, cip, cwtp, cp,
            ws.ctr_keys1, ws.ctr_vals1, cn, ctr_shift);

        cub::DeviceRadixSort::SortPairs(ws.ctr_sort_temp, ws.ctr_sort_temp_size,
            ws.ctr_keys1, ws.ctr_keys2, ws.ctr_vals1, ws.ctr_vals2, ce, 0, ctr_end_bit);

        int32_t nue = reduce_by_key_preallocated(ws.ctr_keys2, ws.ctr_vals2, ce,
            ws.ctr_red_keys, ws.ctr_red_vals);

        
        co.assign(nc+1, 0);
        ci.resize(nue); cwt.resize(nue);
        auto cop2 = thrust::raw_pointer_cast(co.data());
        auto cip2 = thrust::raw_pointer_cast(ci.data());
        auto cwtp2 = thrust::raw_pointer_cast(cwt.data());
        int64_t ctr_mask = ((int64_t)1 << ctr_shift) - 1;

        thrust::for_each(thrust::device, thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(nue),
            [=, rk=ws.ctr_red_keys, rv=ws.ctr_red_vals] __device__(int i) {
                int64_t k = rk[i];
                int32_t s = (int32_t)(k >> ctr_shift);
                cip2[i] = (int32_t)(k & ctr_mask);
                cwtp2[i] = rv[i];
                atomicAdd(&cop2[s+1], 1);
            });
        thrust::inclusive_scan(co.begin()+1, co.end(), co.begin()+1);

        cn = nc; ce = nue;
    }

    if (dendro.empty()) {
        thrust::sequence(thrust::device_pointer_cast(out), thrust::device_pointer_cast(out+n));
    } else {
        thrust::device_vector<int32_t> r(dendro[0]);
        for (size_t i = 1; i < dendro.size(); i++) {
            thrust::device_vector<int32_t> t(n);
            thrust::gather(r.begin(), r.end(), dendro[i].begin(), t.begin());
            r = t;
        }
        cudaMemcpy(out, thrust::raw_pointer_cast(r.data()), n*4, cudaMemcpyDeviceToDevice);
    }
    *olv=(int)dendro.size(); *omod=bmod;
}

}  

ecg_result_double_t ecg(const graph32_t& graph,
                        const double* edge_weights,
                        int32_t* clusters,
                        double min_weight,
                        std::size_t ensemble_size,
                        std::size_t max_level,
                        double threshold,
                        double resolution) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;

    if (nv == 0) {
        return ecg_result_double_t{0, 0.0};
    }

    cache.ensure(nv, ne);
    thrust::device_vector<double> cc(ne, 0.0);
    thrust::device_vector<int32_t> tc(nv);

    for (std::size_t i = 0; i < ensemble_size; i++) {
        uint64_t seed = splitmix64((uint64_t)(i+1) * 0x9E3779B97F4A7C15ULL);
        int dl; double dm;
        louvain_impl(d_off, d_idx, edge_weights, nv, ne,
            thrust::raw_pointer_cast(tc.data()), 1, threshold, resolution, true, seed, &dl, &dm, cache);
        const int32_t* tp = thrust::raw_pointer_cast(tc.data());
        double* cp = thrust::raw_pointer_cast(cc.data());
        thrust::for_each(thrust::device, thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(nv),
            [=] __device__(int v) {
                for (int e = d_off[v]; e < d_off[v+1]; e++)
                    if (tp[v] == tp[d_idx[e]]) cp[e] += 1.0;
            });
    }

    thrust::device_vector<double> mw(ne);
    {
        const double* ew = edge_weights;
        const double* ccp = thrust::raw_pointer_cast(cc.data());
        double* mp = thrust::raw_pointer_cast(mw.data());
        double esd = (double)ensemble_size, mwv = min_weight;
        thrust::for_each(thrust::device, thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(ne),
            [=] __device__(int e) { mp[e] = mwv + (ew[e] - mwv) * ccp[e] / esd; });
    }

    int lv; double mv;
    louvain_impl(d_off, d_idx, thrust::raw_pointer_cast(mw.data()),
        nv, ne, clusters, max_level, threshold, resolution, true, 0x12345678ULL, &lv, &mv, cache);

    
    double tw = thrust::reduce(thrust::device_pointer_cast(edge_weights),
        thrust::device_pointer_cast(edge_weights+ne), 0.0);
    thrust::device_vector<double> ovw(nv);
    compute_vw_kernel<<<(nv+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(d_off, edge_weights,
        thrust::raw_pointer_cast(ovw.data()), nv);
    int32_t mc = thrust::reduce(thrust::device_pointer_cast(clusters),
        thrust::device_pointer_cast(clusters+nv), (int32_t)0, thrust::maximum<int32_t>());
    thrust::device_vector<double> fcw(mc+1, 0.0);
    accum_cw_kernel<<<(nv+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(clusters,
        thrust::raw_pointer_cast(ovw.data()),
        thrust::raw_pointer_cast(fcw.data()), nv);
    cudaDeviceSynchronize();
    int gn=(nv+BLOCK_SIZE-1)/BLOCK_SIZE;
    mv = compute_mod(d_off, d_idx, edge_weights, clusters,
        thrust::raw_pointer_cast(fcw.data()), nv, mc+1, tw, resolution,
        cache.partials, gn);

    return ecg_result_double_t{(std::size_t)lv, mv};
}

}  
