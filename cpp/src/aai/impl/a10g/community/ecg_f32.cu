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
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <cstdint>

namespace aai {

namespace {

struct BumpAlloc {
    char* base; size_t offset, cap;
    void* overflow[32]; int n_overflow;
    void init(void* b,size_t c){base=(char*)b;offset=0;cap=c;n_overflow=0;}
    void* alloc(size_t n){
        offset=(offset+255)&~(size_t)255;
        if(offset+n>cap){
            void*p=nullptr;cudaMalloc(&p,n);
            if(n_overflow<32)overflow[n_overflow++]=p;
            return p;
        }
        void*p=base+offset;offset+=n;return p;
    }
    template<typename T>T* al(size_t n){return(T*)alloc(n*sizeof(T));}
    void reset(){for(int i=0;i<n_overflow;i++)cudaFree(overflow[i]);n_overflow=0;offset=0;}
};


__global__ void compute_vw(const int*__restrict__ off,const float*__restrict__ wt,float*__restrict__ vw,int nv){
    for(int v=blockIdx.x*blockDim.x+threadIdx.x;v<nv;v+=gridDim.x*blockDim.x){
        float s=0.f;for(int i=off[v];i<off[v+1];i++)s+=wt[i];vw[v]=s;}}
__global__ void cw_kern(const int*__restrict__ c,const float*__restrict__ vw,float*__restrict__ cw,int n){
    for(int v=blockIdx.x*blockDim.x+threadIdx.x;v<n;v+=gridDim.x*blockDim.x)atomicAdd(&cw[c[v]],vw[v]);}
__global__ void init_id(int*a,int n){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)a[i]=i;}
__global__ void gen_rk(unsigned*k,int n,unsigned seed){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){
        unsigned h=seed^(unsigned)i;h^=h>>16;h*=0x85ebca6bU;h^=h>>13;h*=0xc2b2ae35U;h^=h>>16;k[i]=h;}}


__global__ void fused_mod(const int*__restrict__ off,const int*__restrict__ idx,
    const float*__restrict__ wt,const int*__restrict__ cm,const float*__restrict__ cw,
    float*__restrict__ pi,float*__restrict__ pd,float*__restrict__ out,unsigned*__restrict__ ret,int nv){
    typedef cub::BlockReduce<float,256>BR;__shared__ typename BR::TempStorage tmp;
    float li=0.f,ld=0.f;
    for(int v=blockIdx.x*256+threadIdx.x;v<nv;v+=gridDim.x*256){
        int mc=cm[v];float w=cw[v];ld+=w*w;
        for(int e=off[v];e<off[v+1];e++)if(cm[idx[e]]==mc)li+=wt[e];}
    float bi=BR(tmp).Sum(li);if(!threadIdx.x)pi[blockIdx.x]=bi;__syncthreads();
    float bd=BR(tmp).Sum(ld);if(!threadIdx.x)pd[blockIdx.x]=bd;__threadfence();
    __shared__ bool last;if(!threadIdx.x){unsigned t=atomicAdd(ret,1);last=(t==(unsigned)gridDim.x-1);}__syncthreads();
    if(last){float si=0.f,sd=0.f;
        for(int i=threadIdx.x;i<(int)gridDim.x;i+=256){si+=pi[i];sd+=pd[i];}
        si=BR(tmp).Sum(si);if(!threadIdx.x)out[0]=si;__syncthreads();
        sd=BR(tmp).Sum(sd);if(!threadIdx.x){out[1]=sd;*ret=0;}}}


#define HC 512
#define HM (HC-1)
__global__ __launch_bounds__(128)
void lm_kern(const int*__restrict__ off,const int*__restrict__ idx,const float*__restrict__ wt,
    const int*__restrict__ cin,int*__restrict__ cout,const float*__restrict__ cw,
    const float*__restrict__ vw,float tew,float res,bool ud,int nv){
    const int WPB=blockDim.x>>5;int lane=threadIdx.x&31,wib=threadIdx.x>>5,gw=blockIdx.x*WPB+wib;
    extern __shared__ char sm[];int*hk=(int*)sm+wib*HC;float*hv=(float*)((int*)sm+WPB*HC)+wib*HC;
    for(int i=lane;i<HC;i+=32){hk[i]=-1;hv[i]=0.f;}__syncwarp();
    if(gw>=nv)return;int v=gw,mc=cin[v];float kv=vw[v];int s=off[v],e=off[v+1];float sl=0.f;
    for(int i=s+lane;i<e;i+=32){int nb=idx[i];float w=wt[i];int nc=cin[nb];if(nb==v)sl+=w;
        unsigned slot=((unsigned)nc*2654435761U)&HM;for(int p=0;p<HC;p++){int ok=atomicCAS(&hk[slot],-1,nc);
            if(ok==-1||ok==nc){atomicAdd(&hv[slot],w);break;}slot=(slot+1)&HM;}}
    for(int o=16;o>0;o>>=1)sl+=__shfl_xor_sync(0xffffffff,sl,o);__syncwarp();
    float wo=0.f;for(int i=lane;i<HC;i+=32)if(hk[i]==mc)wo=hv[i];
    for(int o=16;o>0;o>>=1)wo+=__shfl_xor_sync(0xffffffff,wo,o);
    float os=wo-sl,ao=cw[mc],im=1.f/tew,im2=im*im;float bd=0.f;int bc=mc;
    for(int i=lane;i<HC;i+=32){if(hk[i]!=-1&&hk[i]!=mc){int c=hk[i];float wc=hv[i],an=cw[c];
        float d=2.f*((wc-os)*im-res*(an*kv-ao*kv+kv*kv)*im2);if(d>bd||(d==bd&&c<bc)){bd=d;bc=c;}}}
    for(int o=16;o>0;o>>=1){float od=__shfl_xor_sync(0xffffffff,bd,o);int oc=__shfl_xor_sync(0xffffffff,bc,o);
        if(od>bd||(od==bd&&oc<bc)){bd=od;bc=oc;}}
    if(!lane){if(bd>0.f){bool mv=ud?(bc>mc):(bc<mc);cout[v]=mv?bc:mc;}else cout[v]=mc;}}


__global__ void ecg_cnt(const int*__restrict__ off,const int*__restrict__ idx,
    const int*__restrict__ cm,float*__restrict__ fr,int nv){
    for(int v=blockIdx.x*blockDim.x+threadIdx.x;v<nv;v+=gridDim.x*blockDim.x){
        int mc=cm[v];for(int i=off[v];i<off[v+1];i++)if(cm[idx[i]]==mc)fr[i]+=1.f;}}
__global__ void ecg_wt(const float*__restrict__ ow,const float*__restrict__ fr,
    float*__restrict__ mw,float minw,float ies,int ne){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<ne;i+=gridDim.x*blockDim.x)
        mw[i]=minw+(ow[i]-minw)*fr[i]*ies;}


__global__ void expand_src(const int*__restrict__ off,int*__restrict__ src,int nv){
    for(int v=blockIdx.x*blockDim.x+threadIdx.x;v<nv;v+=gridDim.x*blockDim.x)
        for(int i=off[v];i<off[v+1];i++)src[i]=v;}
__global__ void pack64(const int*__restrict__ s,const int*__restrict__ d,
    const int*__restrict__ cm,int64_t*__restrict__ pk,int ne,int K){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<ne;i+=gridDim.x*blockDim.x)
        pk[i]=(int64_t)cm[s[i]]*K+cm[d[i]];}
__global__ void pack32(const int*__restrict__ s,const int*__restrict__ d,
    const int*__restrict__ cm,unsigned*__restrict__ pk,int ne,int K){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<ne;i+=gridDim.x*blockDim.x)
        pk[i]=(unsigned)cm[s[i]]*K+cm[d[i]];}
__global__ void unpack64(const int64_t*__restrict__ pk,int*__restrict__ s,int*__restrict__ d,int n,int K){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){s[i]=(int)(pk[i]/K);d[i]=(int)(pk[i]%K);}}
__global__ void unpack32(const unsigned*__restrict__ pk,int*__restrict__ s,int*__restrict__ d,int n,int K){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x){s[i]=(int)(pk[i]/K);d[i]=(int)(pk[i]%K);}}
__global__ void flatten_d(int*__restrict__ r,const int*__restrict__ l,int n){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x)r[i]=l[r[i]];}
__global__ void renum_kern(int*__restrict__ c,const int*__restrict__ u,int n,int nu){
    for(int v=blockIdx.x*blockDim.x+threadIdx.x;v<n;v+=gridDim.x*blockDim.x){
        int cv=c[v];int lo=0,hi=nu;while(lo<hi){int m=(lo+hi)>>1;if(u[m]<cv)lo=m+1;else hi=m;}c[v]=lo;}}


static inline int gSz(int n,int bs=256){int g=(n+bs-1)/bs;return g<1?1:(g>256?256:g);}

static float comp_mod_h(const int*off,const int*idx,const float*wt,
    const int*cm,const float*cw,float tew,float res,int nv,
    float*pi,float*pd,float*out,unsigned*ret,cudaStream_t s){
    int g=gSz(nv);if(g<64)g=64;if(g>512)g=512;
    fused_mod<<<g,256,0,s>>>(off,idx,wt,cm,cw,pi,pd,out,ret,nv);
    float h[2];cudaMemcpyAsync(h,out,8,cudaMemcpyDeviceToHost,s);cudaStreamSynchronize(s);
    return h[0]/tew-res*h[1]/(tew*tew);}

static void recomp_cw(const int*c,const float*vw,float*cw,int n,cudaStream_t s){
    cudaMemsetAsync(cw,0,n*sizeof(float),s);cw_kern<<<gSz(n),256,0,s>>>(c,vw,cw,n);}


static float louvain_lm(const int*off,const int*idx,const float*wt,
    int*cm,int*ct,int*best,float*cw,const float*vw,
    float tew,float res,float thr,int nv,
    float*pi,float*pd,float*out,unsigned*ret,int smem,cudaStream_t s){
    recomp_cw(cm,vw,cw,nv,s);
    float nQ=comp_mod_h(off,idx,wt,cm,cw,tew,res,nv,pi,pd,out,ret,s);
    float cQ=nQ-1.f;bool ud=true;
    cudaMemcpyAsync(best,cm,nv*sizeof(int),cudaMemcpyDeviceToDevice,s);
    int wpb=4,thr_lm=wpb*32,lmg=(nv+wpb-1)/wpb;
    for(int it=0;it<100&&nQ>cQ+thr;it++){
        cQ=nQ;
        lm_kern<<<lmg,thr_lm,smem,s>>>(off,idx,wt,cm,ct,cw,vw,tew,res,ud,nv);
        int*t=cm;cm=ct;ct=t;ud=!ud;
        recomp_cw(cm,vw,cw,nv,s);
        nQ=comp_mod_h(off,idx,wt,cm,cw,tew,res,nv,pi,pd,out,ret,s);
        if(nQ>cQ)cudaMemcpyAsync(best,cm,nv*sizeof(int),cudaMemcpyDeviceToDevice,s);
    }
    
    cudaMemcpyAsync(cm,best,nv*sizeof(int),cudaMemcpyDeviceToDevice,s);
    return cQ>nQ?cQ:nQ;
}

struct Cache : Cacheable {
    void* scratch = nullptr;
    size_t scratch_size = 0;

    void ensure(int32_t nv, int32_t ne) {
        size_t needed = (size_t)128 * nv + (size_t)100 * ne + (1 << 22);
        if (needed < (size_t)1024 * 1024 * 1024) needed = (size_t)1024 * 1024 * 1024;
        if (needed > scratch_size) {
            if (scratch) cudaFree(scratch);
            auto err = cudaMalloc(&scratch, needed);
            if (err != cudaSuccess) {
                scratch = nullptr; scratch_size = 0;
                needed = (size_t)128 * nv + (size_t)100 * ne + (1 << 22);
                cudaMalloc(&scratch, needed);
            }
            scratch_size = needed;
        }
    }

    ~Cache() override {
        if (scratch) { cudaFree(scratch); scratch = nullptr; }
    }
};

}  

ecg_result_float_t ecg(const graph32_t& graph,
                       const float* edge_weights,
                       int32_t* clusters,
                       float min_weight,
                       std::size_t ensemble_size,
                       std::size_t max_level,
                       float threshold,
                       float resolution) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int* d_off = graph.offsets;
    const int* d_idx = graph.indices;
    const float* d_ew = edge_weights;
    int nv = graph.number_of_vertices;
    int ne = graph.number_of_edges;
    float minw = min_weight;
    int ens_sz = static_cast<int>(ensemble_size);
    int max_lv = static_cast<int>(max_level);
    float thr = threshold;
    float res = resolution;
    int32_t* d_out_cl = clusters;

    cache.ensure(nv, ne);

    cudaStream_t s=0;BumpAlloc pool;pool.init(cache.scratch,cache.scratch_size);

    float*d_vw=pool.al<float>(nv);float*d_cw=pool.al<float>(nv);
    int*d_cm=pool.al<int>(nv);int*d_ct=pool.al<int>(nv);int*d_bc=pool.al<int>(nv);
    float*d_pi=pool.al<float>(512);float*d_pd=pool.al<float>(512);
    float*d_out=pool.al<float>(4);unsigned*d_ret=pool.al<unsigned>(1);cudaMemsetAsync(d_ret,0,4,s);
    float*d_ef=pool.al<float>(ne);float*d_mw=pool.al<float>(ne);
    unsigned*d_rk=pool.al<unsigned>(nv);unsigned*d_rk2=pool.al<unsigned>(nv);
    int*d_pi2=pool.al<int>(nv);int*d_pi2b=pool.al<int>(nv);
    int*d_es=pool.al<int>(ne);
    int64_t*d_pk=(int64_t*)pool.alloc(ne*8);int64_t*d_pk2=(int64_t*)pool.alloc(ne*8);
    float*d_ews=pool.al<float>(ne);float*d_ews2=pool.al<float>(ne);
    int*d_uc=pool.al<int>(nv);
    int*d_oa=pool.al<int>(nv+1);int*d_ia=pool.al<int>(ne);float*d_wa=pool.al<float>(ne);
    int*d_ob=pool.al<int>(nv+1);int*d_ib=pool.al<int>(ne);float*d_wb=pool.al<float>(ne);

    size_t cub_sz=0;
    cub::DeviceRadixSort::SortPairs((void*)nullptr,cub_sz,d_pk,d_pk2,d_ews,d_ews2,ne,0,64,s);
    size_t perm_sz=0;
    cub::DeviceRadixSort::SortPairs((void*)nullptr,perm_sz,d_rk,d_rk2,d_pi2,d_pi2b,nv,0,32,s);
    if(perm_sz>cub_sz)cub_sz=perm_sz;
    void*d_cub=pool.alloc(cub_sz);

    int*d_den[128];int nlev=0;for(int i=0;i<128;i++)d_den[i]=nullptr;
    int wpb=4,smem=wpb*HC*(sizeof(int)+sizeof(float));
    cudaFuncSetAttribute(lm_kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);

    compute_vw<<<gSz(nv),256,0,s>>>(d_off,d_ew,d_vw,nv);
    float tew_orig=thrust::reduce(thrust::cuda::par.on(s),d_ew,d_ew+ne,0.f);

    
    cudaMemsetAsync(d_ef,0,ne*sizeof(float),s);
    for(int e=0;e<ens_sz;e++){
        gen_rk<<<gSz(nv),256,0,s>>>(d_rk,nv,(unsigned)(e*1000003+42));
        init_id<<<gSz(nv),256,0,s>>>(d_pi2,nv);
        cub::DeviceRadixSort::SortPairs(d_cub,cub_sz,d_rk,d_rk2,d_pi2,d_pi2b,nv,0,32,s);
        cudaMemcpyAsync(d_cm,d_pi2b,nv*sizeof(int),cudaMemcpyDeviceToDevice,s);
        louvain_lm(d_off,d_idx,d_ew,d_cm,d_ct,d_bc,d_cw,d_vw,
            tew_orig,res,thr,nv,d_pi,d_pd,d_out,d_ret,smem,s);
        
        ecg_cnt<<<gSz(nv),256,0,s>>>(d_off,d_idx,d_bc,d_ef,nv);
    }

    
    ecg_wt<<<gSz(ne),256,0,s>>>(d_ew,d_ef,d_mw,minw,1.f/(float)ens_sz,ne);

    
    const int*co=d_off;const int*ci=d_idx;const float*cwt=d_mw;
    int cv=nv,ce=ne;
    float tew=thrust::reduce(thrust::cuda::par.on(s),cwt,cwt+ce,0.f);
    float bestQ=-1.f;bool usea=true;

    for(int lv=0;lv<max_lv&&lv<128;lv++){
        init_id<<<gSz(cv),256,0,s>>>(d_cm,cv);
        d_den[lv]=pool.al<int>(cv);nlev=lv+1;
        cudaMemcpyAsync(d_den[lv],d_cm,cv*sizeof(int),cudaMemcpyDeviceToDevice,s);
        compute_vw<<<gSz(cv),256,0,s>>>(co,cwt,d_vw,cv);
        float Q=louvain_lm(co,ci,cwt,d_cm,d_ct,d_den[lv],d_cw,d_vw,tew,res,thr,cv,d_pi,d_pd,d_out,d_ret,smem,s);
        if(Q<=bestQ){nlev=lv;break;}
        bestQ=Q;

        
        expand_src<<<gSz(cv),256,0,s>>>(co,d_es,cv);
        cudaMemcpyAsync(d_uc,d_den[lv],cv*sizeof(int),cudaMemcpyDeviceToDevice,s);
        thrust::sort(thrust::cuda::par.on(s),d_uc,d_uc+cv);
        int*ue=thrust::unique(thrust::cuda::par.on(s),d_uc,d_uc+cv);int K=(int)(ue-d_uc);
        renum_kern<<<gSz(cv),256,0,s>>>(d_den[lv],d_uc,cv,K);

        bool use32=((int64_t)K*K<=(int64_t)0xFFFFFFFF);
        if(use32){
            unsigned maxk=(unsigned)K*(unsigned)K;int nb=0;unsigned mk=maxk;while(mk>0){nb++;mk>>=1;}
            pack32<<<gSz(ce),256,0,s>>>(d_es,(const int*)ci,d_den[lv],(unsigned*)d_pk,ce,K);
            cudaMemcpyAsync(d_ews,cwt,ce*sizeof(float),cudaMemcpyDeviceToDevice,s);
            size_t tsz=0;cub::DeviceRadixSort::SortPairs(nullptr,tsz,(unsigned*)d_pk,(unsigned*)d_pk2,d_ews,d_ews2,ce,0,nb,s);
            if(tsz>cub_sz){cub_sz=tsz;d_cub=pool.alloc(cub_sz);}
            cub::DeviceRadixSort::SortPairs(d_cub,cub_sz,(unsigned*)d_pk,(unsigned*)d_pk2,d_ews,d_ews2,ce,0,nb,s);
            int nE=(int)(thrust::reduce_by_key(thrust::cuda::par.on(s),(unsigned*)d_pk2,(unsigned*)d_pk2+ce,d_ews2,(unsigned*)d_pk,d_ews).first-(unsigned*)d_pk);
            int*no=usea?d_oa:d_ob;int*ni=usea?d_ia:d_ib;float*nw=usea?d_wa:d_wb;
            unpack32<<<gSz(nE),256,0,s>>>((unsigned*)d_pk,d_es,ni,nE,K);
            cudaMemsetAsync(no,0,(K+1)*sizeof(int),s);
            thrust::upper_bound(thrust::cuda::par.on(s),d_es,d_es+nE,thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(K),no+1);
            cudaMemcpyAsync(nw,d_ews,nE*sizeof(float),cudaMemcpyDeviceToDevice,s);
            co=no;ci=ni;cwt=nw;cv=K;ce=nE;usea=!usea;
        } else {
            int64_t maxk=(int64_t)K*K;int nb=0;int64_t mk=maxk;while(mk>0){nb++;mk>>=1;}
            pack64<<<gSz(ce),256,0,s>>>(d_es,(const int*)ci,d_den[lv],d_pk,ce,K);
            cudaMemcpyAsync(d_ews,cwt,ce*sizeof(float),cudaMemcpyDeviceToDevice,s);
            size_t tsz=0;cub::DeviceRadixSort::SortPairs(nullptr,tsz,d_pk,d_pk2,d_ews,d_ews2,ce,0,nb,s);
            if(tsz>cub_sz){cub_sz=tsz;d_cub=pool.alloc(cub_sz);}
            cub::DeviceRadixSort::SortPairs(d_cub,cub_sz,d_pk,d_pk2,d_ews,d_ews2,ce,0,nb,s);
            int nE=(int)(thrust::reduce_by_key(thrust::cuda::par.on(s),d_pk2,d_pk2+ce,d_ews2,d_pk,d_ews).first-d_pk);
            int*no=usea?d_oa:d_ob;int*ni=usea?d_ia:d_ib;float*nw=usea?d_wa:d_wb;
            unpack64<<<gSz(nE),256,0,s>>>(d_pk,d_es,ni,nE,K);
            cudaMemsetAsync(no,0,(K+1)*sizeof(int),s);
            thrust::upper_bound(thrust::cuda::par.on(s),d_es,d_es+nE,thrust::counting_iterator<int>(0),thrust::counting_iterator<int>(K),no+1);
            cudaMemcpyAsync(nw,d_ews,nE*sizeof(float),cudaMemcpyDeviceToDevice,s);
            co=no;ci=ni;cwt=nw;cv=K;ce=nE;usea=!usea;
        }
        if(cv<=1)break;
    }

    
    thrust::sequence(thrust::cuda::par.on(s),d_out_cl,d_out_cl+nv);
    for(int l=0;l<nlev;l++)flatten_d<<<gSz(nv),256,0,s>>>(d_out_cl,d_den[l],nv);

    
    compute_vw<<<gSz(nv),256,0,s>>>(d_off,d_ew,d_vw,nv);
    recomp_cw(d_out_cl,d_vw,d_cw,nv,s);
    float fmod=comp_mod_h(d_off,d_idx,d_ew,d_out_cl,d_cw,tew_orig,res,nv,d_pi,d_pd,d_out,d_ret,s);

    cudaStreamSynchronize(s);
    pool.reset();

    return ecg_result_float_t{(std::size_t)nlev, fmod};
}

}  
