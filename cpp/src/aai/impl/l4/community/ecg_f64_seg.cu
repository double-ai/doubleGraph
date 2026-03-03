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
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/transform_reduce.h>
#include <cstdint>
#include <vector>

namespace aai {

namespace {

struct Cache : Cacheable {};

static inline int div_up(int a, int b) { return (a + b - 1) / b; }
static const int BS = 256;

__global__ void k_vw(const int* __restrict__ off, const double* __restrict__ ew,
    double* __restrict__ vw, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    double s = 0.0;
    for (int e = off[v]; e < off[v+1]; e++) s += ew[e];
    vw[v] = s;
}

__global__ void k_icw(const int* __restrict__ c, const double* __restrict__ vw,
    double* __restrict__ cw, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    atomicAdd(&cw[c[v]], vw[v]);
}

__device__ __forceinline__ unsigned int hf(unsigned int x) {
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    return (x >> 16) ^ x;
}


#define LO_BLK 128
#define LO_HASH 48

__global__ void __launch_bounds__(LO_BLK, 1)
k_lo(const int* __restrict__ off, const int* __restrict__ idx,
    const double* __restrict__ ew, const int* __restrict__ comm,
    const double* __restrict__ cw, const double* __restrict__ vw,
    int* __restrict__ bc, double* __restrict__ bd,
    int n, double M, double res, int vs) {
    extern __shared__ char sm[];
    int* hk = (int*)sm;
    double* hv = (double*)(sm + LO_HASH*LO_BLK*sizeof(int));
    int tid = threadIdx.x;
    int v = vs + blockIdx.x*LO_BLK + tid;

    for (int p = 0; p < LO_HASH; p++) {
        hk[p*LO_BLK+tid] = -1; hv[p*LO_BLK+tid] = 0.0;
    }
    if (v >= n) return;
    int s = off[v], e = off[v+1], mc = comm[v];
    if (s >= e) { bc[v] = mc; bd[v] = 0.0; return; }

    double k = vw[v], iM = 1.0/M, iM2 = iM*iM;
    for (int i = s; i < e; i++) {
        int u = idx[i]; if (u == v) continue;
        int c = comm[u]; double w = ew[i];
        unsigned h = hf((unsigned)c) % LO_HASH;
        for (int p = 0; p < LO_HASH; p++) {
            int x = ((int)(h+p) % LO_HASH)*LO_BLK+tid;
            if (hk[x]==c) { hv[x]+=w; break; }
            if (hk[x]==-1) { hk[x]=c; hv[x]=w; break; }
        }
    }

    double wo = 0.0;
    { unsigned h = hf((unsigned)mc) % LO_HASH;
      for (int p = 0; p < LO_HASH; p++) {
          int x = ((int)(h+p) % LO_HASH)*LO_BLK+tid;
          if (hk[x]==mc) { wo=hv[x]; break; }
          if (hk[x]==-1) break; }
    }

    double base = wo*iM - res*(cw[mc]-k)*k*iM2;
    double best_d = 0.0; int best_c = mc;
    for (int p = 0; p < LO_HASH; p++) {
        int x = p*LO_BLK+tid;
        int c = hk[x]; if (c==-1||c==mc) continue;
        double d = hv[x]*iM - res*cw[c]*k*iM2 - base;
        if (d > best_d || (d==best_d && c < best_c)) { best_d=d; best_c=c; }
    }
    bc[v]=best_c; bd[v]=best_d;
}


#define HI_BLK 256
#define HI_HASH 512

__global__ void __launch_bounds__(HI_BLK, 1)
k_hi(const int* __restrict__ off, const int* __restrict__ idx,
    const double* __restrict__ ew, const int* __restrict__ comm,
    const double* __restrict__ cw, const double* __restrict__ vw,
    int* __restrict__ bc_out, double* __restrict__ bd_out,
    int n, double M, double res,
    const int* __restrict__ vl, int nl) {
    extern __shared__ char sm[];
    int* hk = (int*)sm;
    double* hv = (double*)(sm + HI_HASH*sizeof(int));

    int vid = blockIdx.x; if (vid >= nl) return;
    int v = vl[vid]; if (v >= n) return;
    int tid = threadIdx.x;

    for (int i = tid; i < HI_HASH; i += HI_BLK) { hk[i]=-1; hv[i]=0.0; }
    __syncthreads();

    int s=off[v], e=off[v+1], mc=comm[v];
    double k=vw[v], iM=1.0/M, iM2=iM*iM;

    for (int i = s+tid; i < e; i += HI_BLK) {
        int u=idx[i]; if(u==v) continue;
        int c=comm[u]; double w=ew[i];
        unsigned h=hf((unsigned)c) % HI_HASH;
        for (int p = 0; p < HI_HASH; p++) {
            int x=(h+p)%HI_HASH;
            int old=atomicCAS(&hk[x],-1,c);
            if(old==-1||old==c) { atomicAdd(&hv[x],w); break; }
        }
    }
    __syncthreads();

    __shared__ double s_wo;
    if (tid==0) {
        s_wo=0.0;
        unsigned h=hf((unsigned)mc)%HI_HASH;
        for (int p=0;p<HI_HASH;p++) {
            int x=(h+p)%HI_HASH;
            if(hk[x]==mc){s_wo=hv[x];break;}
            if(hk[x]==-1) break;
        }
    }
    __syncthreads();

    double base=s_wo*iM-res*(cw[mc]-k)*k*iM2;
    double td=0.0; int tc=mc;
    for (int i=tid; i<HI_HASH; i+=HI_BLK) {
        int c=hk[i]; if(c==-1||c==mc) continue;
        double d=hv[i]*iM-res*cw[c]*k*iM2-base;
        if(d>td||(d==td&&c<tc)){td=d;tc=c;}
    }

    __shared__ double ad[HI_BLK]; __shared__ int ac[HI_BLK];
    ad[tid]=td; ac[tid]=tc;
    __syncthreads();
    for (int s2=HI_BLK/2;s2>0;s2>>=1) {
        if(tid<s2) {
            if(ad[tid+s2]>ad[tid]||(ad[tid+s2]==ad[tid]&&ac[tid+s2]<ac[tid])){
                ad[tid]=ad[tid+s2]; ac[tid]=ac[tid+s2];
            }
        }
        __syncthreads();
    }
    if(tid==0) { bc_out[v]=ac[0]; bd_out[v]=ad[0]; }
}


__global__ void k_apply_ucw(
    const int* __restrict__ comm, const int* __restrict__ bc,
    const double* __restrict__ bd, const double* __restrict__ vw,
    int* __restrict__ nc, double* __restrict__ cw, int* __restrict__ mv,
    int n, bool ud) {
    int v = blockIdx.x*blockDim.x+threadIdx.x;
    if (v>=n) return;
    int mc=comm[v], b=bc[v]; double d=bd[v];
    if (d>0.0&&b!=mc&&((b>mc)==ud)) {
        nc[v]=b;
        double w=vw[v];
        atomicAdd(&cw[mc],-w);
        atomicAdd(&cw[b],w);
        atomicAdd(mv,1);
    } else nc[v]=mc;
}


__global__ void k_iw(const int* __restrict__ off, const int* __restrict__ idx,
    const double* __restrict__ w, const int* __restrict__ comm,
    double* __restrict__ out, int n) {
    int v = blockIdx.x*blockDim.x+threadIdx.x;
    if (v>=n) return;
    double s=0.0; int c=comm[v];
    for (int e=off[v];e<off[v+1];e++) if(comm[idx[e]]==c) s+=w[e];
    out[v]=s;
}

__global__ void k_rk(unsigned int* keys, int n, unsigned int seed) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    unsigned x=(unsigned)i^seed;
    x=((x>>16)^x)*0x45d9f3bu; x=((x>>16)^x)*0x45d9f3bu; x=(x>>16)^x;
    x^=seed*2654435761u; x=((x>>16)^x)*0x45d9f3bu;
    keys[i]=x;
}

__global__ void k_eacc(const int* __restrict__ off, const int* __restrict__ idx,
    const int* __restrict__ comm, double* __restrict__ freq, int n) {
    int v=blockIdx.x*blockDim.x+threadIdx.x;
    if(v>=n) return;
    int c=comm[v];
    for(int e=off[v];e<off[v+1];e++) if(comm[idx[e]]==c) freq[e]+=1.0;
}

__global__ void k_ew(const double* __restrict__ ow, const double* __restrict__ freq,
    double* __restrict__ mw, int ne, double mwt, double ie) {
    int e=blockIdx.x*blockDim.x+threadIdx.x;
    if(e>=ne) return;
    mw[e]=mwt+(ow[e]-mwt)*freq[e]*ie;
}

__global__ void k_ce(const int* __restrict__ off, const int* __restrict__ idx,
    const double* __restrict__ w, const int* __restrict__ comm,
    int64_t* __restrict__ keys, double* __restrict__ wo, int n) {
    int v=blockIdx.x*blockDim.x+threadIdx.x;
    if(v>=n) return;
    int cv=comm[v];
    for(int e=off[v];e<off[v+1];e++){
        keys[e]=((int64_t)cv<<32)|(unsigned)comm[idx[e]];
        wo[e]=w[e];
    }
}

__global__ void k_rn(int* __restrict__ comm, const int* __restrict__ um, int nu, int n) {
    int v=blockIdx.x*blockDim.x+threadIdx.x;
    if(v>=n) return;
    int c=comm[v],lo=0,hi=nu-1;
    while(lo<=hi){int m=(lo+hi)>>1,val=um[m];
        if(val==c){comm[v]=m;return;}
        if(val<c)lo=m+1;else hi=m-1;}
}


struct CSR { thrust::device_vector<int> off,idx; thrust::device_vector<double> w; int nv,ne; };

static double tw(const double* w, int n) {
    thrust::device_ptr<const double> p(w); return thrust::reduce(p,p+n,0.0);
}

static double modQ(const int* d_off, const int* d_idx, const double* d_w,
    const int* d_c, const double* d_cw, int nv, int ncw, double M, double res) {
    thrust::device_vector<double> t(nv);
    k_iw<<<div_up(nv,BS),BS>>>(d_off,d_idx,d_w,d_c,thrust::raw_pointer_cast(t.data()),nv);
    double si=thrust::reduce(t.begin(),t.end(),0.0);
    thrust::device_ptr<const double> cp(d_cw);
    auto sq=thrust::make_transform_iterator(cp,[]__device__(double x) -> double {return x*x;});
    double ssq=thrust::reduce(sq,sq+ncw,0.0);
    return si/M-res*ssq/(M*M);
}

static void gperm(int* d, int n, unsigned seed) {
    thrust::device_vector<unsigned int> keys(n);
    k_rk<<<div_up(n,BS),BS>>>(thrust::raw_pointer_cast(keys.data()),n,seed);
    thrust::device_vector<int> idx(n);
    thrust::sequence(idx.begin(),idx.end());
    thrust::sort_by_key(keys.begin(),keys.end(),idx.begin());
    thrust::scatter(thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(n),
        idx.begin(),thrust::device_pointer_cast(d));
}


static double lvl1(const int* d_off, const int* d_idx, const double* d_w,
    int* d_c, double* d_cw, const double* d_vw,
    int nv, int ne, double M, double res, double thr, int hd) {

    size_t sm_lo = LO_HASH*LO_BLK*(sizeof(int)+sizeof(double));
    cudaFuncSetAttribute(k_lo, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_lo);
    size_t sm_hi = HI_HASH*(sizeof(int)+sizeof(double))+HI_BLK*(sizeof(double)+sizeof(int));
    cudaFuncSetAttribute(k_hi, cudaFuncAttributeMaxDynamicSharedMemorySize, sm_hi);

    thrust::device_vector<int> bc(nv), nc(nv);
    thrust::device_vector<double> bd(nv);
    thrust::device_vector<int> mv(1,0);
    thrust::device_vector<int> hdl(hd);
    if(hd>0) thrust::sequence(hdl.begin(),hdl.end(),0);

    double cur_Q=-2.0;
    double new_Q=modQ(d_off,d_idx,d_w,d_c,d_cw,nv,nv,M,res);
    bool ud=true;

    for(int it=0;it<100;it++) {
        if(!(new_Q > cur_Q + thr)) break;
        cur_Q=new_Q;

        
        int lo_nv=nv-hd;
        if(lo_nv>0) k_lo<<<div_up(lo_nv,LO_BLK),LO_BLK,sm_lo>>>(
            d_off,d_idx,d_w,d_c,d_cw,d_vw,
            thrust::raw_pointer_cast(bc.data()),thrust::raw_pointer_cast(bd.data()),
            nv,M,res,hd);
        if(hd>0) k_hi<<<hd,HI_BLK,sm_hi>>>(
            d_off,d_idx,d_w,d_c,d_cw,d_vw,
            thrust::raw_pointer_cast(bc.data()),thrust::raw_pointer_cast(bd.data()),
            nv,M,res,thrust::raw_pointer_cast(hdl.data()),hd);

        
        cudaMemset(thrust::raw_pointer_cast(mv.data()),0,sizeof(int));
        k_apply_ucw<<<div_up(nv,BS),BS>>>(d_c,
            thrust::raw_pointer_cast(bc.data()),thrust::raw_pointer_cast(bd.data()),
            d_vw, thrust::raw_pointer_cast(nc.data()),
            d_cw, thrust::raw_pointer_cast(mv.data()),nv,ud);

        int moved;
        cudaMemcpy(&moved,thrust::raw_pointer_cast(mv.data()),sizeof(int),cudaMemcpyDeviceToHost);

        if(moved==0) {
            ud=!ud;
            cudaMemset(thrust::raw_pointer_cast(mv.data()),0,sizeof(int));
            k_apply_ucw<<<div_up(nv,BS),BS>>>(d_c,
                thrust::raw_pointer_cast(bc.data()),thrust::raw_pointer_cast(bd.data()),
                d_vw, thrust::raw_pointer_cast(nc.data()),
                d_cw, thrust::raw_pointer_cast(mv.data()),nv,ud);
            cudaMemcpy(&moved,thrust::raw_pointer_cast(mv.data()),sizeof(int),cudaMemcpyDeviceToHost);
        }

        
        cudaMemcpy(d_c,thrust::raw_pointer_cast(nc.data()),nv*sizeof(int),cudaMemcpyDeviceToDevice);

        new_Q=modQ(d_off,d_idx,d_w,d_c,d_cw,nv,nv,M,res);
        ud=!ud;
    }
    return new_Q;
}

static int renum(int* d_c, int nv) {
    thrust::device_vector<int> s(nv);
    thrust::copy(thrust::device_pointer_cast(d_c),thrust::device_pointer_cast(d_c)+nv,s.begin());
    thrust::sort(s.begin(),s.end());
    auto end=thrust::unique(s.begin(),s.end());
    int K=end-s.begin();
    k_rn<<<div_up(nv,BS),BS>>>(d_c,thrust::raw_pointer_cast(s.data()),K,nv);
    return K;
}

static CSR coarsen(const int* d_off, const int* d_idx, const double* d_w,
    const int* d_c, int nv, int ne, int K) {
    thrust::device_vector<int64_t> ek(ne); thrust::device_vector<double> we(ne);
    k_ce<<<div_up(nv,BS),BS>>>(d_off,d_idx,d_w,d_c,
        thrust::raw_pointer_cast(ek.data()),thrust::raw_pointer_cast(we.data()),nv);
    thrust::sort_by_key(ek.begin(),ek.end(),we.begin());
    thrust::device_vector<int64_t> ok(ne); thrust::device_vector<double> ow(ne);
    auto oe=thrust::reduce_by_key(ek.begin(),ek.end(),we.begin(),ok.begin(),ow.begin());
    int nne=oe.first-ok.begin();
    thrust::device_vector<int> ns(nne),nd(nne);
    thrust::transform(ok.begin(),ok.begin()+nne,
        thrust::make_zip_iterator(ns.begin(),nd.begin()),
        []__device__(int64_t k) -> thrust::tuple<int,int> {return thrust::make_tuple((int)(k>>32),(int)(k&0xFFFFFFFF));});
    CSR g; g.nv=K; g.ne=nne;
    g.off.resize(K+1); g.idx.resize(nne); g.w.resize(nne);
    thrust::copy(nd.begin(),nd.end(),g.idx.begin());
    thrust::copy(ow.begin(),ow.begin()+nne,g.w.begin());
    thrust::lower_bound(ns.begin(),ns.end(),
        thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(K+1),g.off.begin());
    return g;
}

static std::pair<double,size_t> louvain_full(
    const int* d_off, const int* d_idx, const double* d_w,
    int* d_cl, int nv, int ne,
    size_t ml, double thr, double res,
    bool rnd, unsigned seed, int hd) {
    double M=tw(d_w,ne);
    thrust::device_vector<int> co(d_off,d_off+nv+1),ci(d_idx,d_idx+ne);
    thrust::device_vector<double> cw(d_w,d_w+ne);
    int cnv=nv,cne=ne,chd=hd;
    std::vector<thrust::device_vector<int>> dend;
    double bestQ=-1.0;

    for(size_t l=0;l<ml;l++){
        thrust::device_vector<int> cm(cnv);
        if(rnd) gperm(thrust::raw_pointer_cast(cm.data()),cnv,seed+(unsigned)l);
        else thrust::sequence(cm.begin(),cm.end());

        thrust::device_vector<double> vw(cnv);
        k_vw<<<div_up(cnv,BS),BS>>>(thrust::raw_pointer_cast(co.data()),
            thrust::raw_pointer_cast(cw.data()),thrust::raw_pointer_cast(vw.data()),cnv);

        thrust::device_vector<double> cwt(cnv,0.0);
        k_icw<<<div_up(cnv,BS),BS>>>(thrust::raw_pointer_cast(cm.data()),
            thrust::raw_pointer_cast(vw.data()),thrust::raw_pointer_cast(cwt.data()),cnv);

        double Q=lvl1(thrust::raw_pointer_cast(co.data()),thrust::raw_pointer_cast(ci.data()),
            thrust::raw_pointer_cast(cw.data()),
            thrust::raw_pointer_cast(cm.data()),thrust::raw_pointer_cast(cwt.data()),
            thrust::raw_pointer_cast(vw.data()),cnv,cne,M,res,thr,chd);

        if(l>0&&Q<=bestQ+thr){
            thrust::sequence(cm.begin(),cm.end());
            dend.push_back(std::move(cm)); break;
        }
        bestQ=Q;
        int K=renum(thrust::raw_pointer_cast(cm.data()),cnv);
        dend.push_back(std::move(cm));
        if(K<=1||K==cnv) break;

        CSR cg=coarsen(thrust::raw_pointer_cast(co.data()),thrust::raw_pointer_cast(ci.data()),
            thrust::raw_pointer_cast(cw.data()),thrust::raw_pointer_cast(dend.back().data()),
            cnv,cne,K);
        co=std::move(cg.off);ci=std::move(cg.idx);cw=std::move(cg.w);
        cnv=cg.nv;cne=cg.ne;chd=0;
    }

    size_t nl=dend.size();
    if(nl==0) thrust::sequence(thrust::device_pointer_cast(d_cl),thrust::device_pointer_cast(d_cl)+nv);
    else if(nl==1) thrust::copy(dend[0].begin(),dend[0].end(),thrust::device_pointer_cast(d_cl));
    else {
        thrust::device_vector<int> cur(dend[0].begin(),dend[0].end());
        for(size_t i=1;i<nl;i++){
            thrust::device_vector<int> nxt(nv);
            thrust::gather(cur.begin(),cur.end(),dend[i].begin(),nxt.begin());
            cur.swap(nxt);
        }
        thrust::copy(cur.begin(),cur.end(),thrust::device_pointer_cast(d_cl));
    }
    return{bestQ,nl};
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
    (void)cache;

    const int* d_off = graph.offsets;
    const int* d_idx = graph.indices;
    int nv = graph.number_of_vertices;
    int ne = graph.number_of_edges;

    const auto& seg = graph.segment_offsets.value();
    int hd = static_cast<int>(seg[2]);

    int es = static_cast<int>(ensemble_size);
    int ml = static_cast<int>(max_level);
    double mw = min_weight;
    double thr = threshold;
    double res = resolution;

    double M = tw(edge_weights, ne);
    thrust::device_vector<double> vw(nv);
    k_vw<<<div_up(nv,BS),BS>>>(d_off, edge_weights, thrust::raw_pointer_cast(vw.data()), nv);

    
    thrust::device_vector<double> freq(ne, 0.0);
    for (int e = 0; e < es; e++) {
        thrust::device_vector<int> cm(nv);
        gperm(thrust::raw_pointer_cast(cm.data()), nv, (unsigned)(e * 1000 + 42));
        thrust::device_vector<double> cwt(nv, 0.0);
        k_icw<<<div_up(nv,BS),BS>>>(thrust::raw_pointer_cast(cm.data()),
            thrust::raw_pointer_cast(vw.data()), thrust::raw_pointer_cast(cwt.data()), nv);
        lvl1(d_off, d_idx, edge_weights, thrust::raw_pointer_cast(cm.data()),
            thrust::raw_pointer_cast(cwt.data()), thrust::raw_pointer_cast(vw.data()),
            nv, ne, M, res, thr, hd);
        k_eacc<<<div_up(nv,BS),BS>>>(d_off, d_idx, thrust::raw_pointer_cast(cm.data()),
            thrust::raw_pointer_cast(freq.data()), nv);
    }

    
    thrust::device_vector<double> mwt(ne);
    k_ew<<<div_up(ne,BS),BS>>>(edge_weights, thrust::raw_pointer_cast(freq.data()),
        thrust::raw_pointer_cast(mwt.data()), ne, mw, 1.0 / (double)es);

    
    auto [Q, nl] = louvain_full(d_off, d_idx, thrust::raw_pointer_cast(mwt.data()),
        clusters, nv, ne, ml, thr, res, true, (unsigned)(es * 1000 + 100), hd);

    
    int mc = *thrust::max_element(thrust::device_pointer_cast(clusters),
        thrust::device_pointer_cast(clusters) + nv);
    thrust::device_vector<double> cwo(mc + 1, 0.0);
    k_icw<<<div_up(nv,BS),BS>>>(clusters, thrust::raw_pointer_cast(vw.data()),
        thrust::raw_pointer_cast(cwo.data()), nv);
    double fQ = modQ(d_off, d_idx, edge_weights, clusters,
        thrust::raw_pointer_cast(cwo.data()), nv, mc + 1, M, res);
    cudaDeviceSynchronize();

    return ecg_result_double_t{nl, fQ};
}

}  
