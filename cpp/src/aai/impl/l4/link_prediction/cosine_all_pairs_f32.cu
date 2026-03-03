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
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

namespace aai {

namespace {




struct ScratchPool {
    char* base; size_t capacity, offset;
    void reset() { offset = 0; }
    template<typename T> T* alloc(size_t count) {
        size_t bytes = (count * sizeof(T) + 255) & ~(size_t)255;
        if (offset + bytes > capacity) return nullptr;
        T* p = reinterpret_cast<T*>(base + offset); offset += bytes; return p;
    }
};

struct Cache : Cacheable {
    char* base = nullptr;
    size_t capacity = 0;

    Cache() {
        capacity = 512ULL * 1024 * 1024;
        cudaMalloc(&base, capacity);
    }

    ~Cache() override {
        if (base) { cudaFree(base); base = nullptr; }
    }
};





__global__ void count_raw_pairs_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds, int32_t num_seeds, int64_t* __restrict__ counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;
    int32_t u = seeds ? seeds[idx] : idx;
    int32_t s = offsets[u], e = offsets[u + 1];
    int64_t c = 0;
    for (int32_t i = s; i < e; i++) c += offsets[indices[i]+1] - offsets[indices[i]];
    counts[idx] = c;
}

static const int64_t EMPTY_KEY = -1LL;

__global__ void init_hashtable(int64_t* table, int64_t cap) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < cap) table[i] = EMPTY_KEY;
}

__global__ void insert_pairs_hashtable(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds, int32_t ns, int64_t nv,
    int64_t* __restrict__ ht, int64_t mask,
    int32_t* __restrict__ cf, int32_t* __restrict__ cs, int32_t* __restrict__ cc)
{
    int si = blockIdx.x; if (si >= ns) return;
    int32_t u = seeds ? seeds[si] : si;
    int32_t us = offsets[u], ue = offsets[u+1];
    int64_t kb = (int64_t)u * nv;
    for (int32_t i = us; i < ue; i++) {
        int32_t k = indices[i], ks = offsets[k], ke = offsets[k+1];
        for (int32_t j = ks + threadIdx.x; j < ke; j += blockDim.x) {
            int32_t v = indices[j]; if (v == u) continue;
            int64_t key = kb + v;
            uint64_t h = (uint64_t)key * 0x9E3779B97F4A7C15ULL;
            int64_t slot = (int64_t)((h >> 32) & (uint64_t)mask);
            for (int p = 0; p < 256; p++) {
                int64_t idx = (slot + p) & mask;
                int64_t old = atomicCAS((unsigned long long*)&ht[idx],
                    (unsigned long long)EMPTY_KEY, (unsigned long long)key);
                if (old == EMPTY_KEY) { int pos = atomicAdd(cc,1); cf[pos]=u; cs[pos]=v; break; }
                else if (old == key) break;
            }
        }
    }
}

__global__ void write_raw_pairs_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds, int32_t ns, int64_t nv,
    const int64_t* __restrict__ po, int64_t* __restrict__ out)
{
    int si = blockIdx.x; if (si >= ns) return;
    int32_t u = seeds ? seeds[si] : si;
    int32_t us = offsets[u], ue = offsets[u+1]; if (us >= ue) return;
    int64_t wb = po[si], kb = (int64_t)u * nv;
    __shared__ int32_t wp;
    if (threadIdx.x == 0) wp = 0; __syncthreads();
    for (int32_t i = us; i < ue; i++) {
        int32_t k = indices[i], ks = offsets[k], ke = offsets[k+1], kd = ke-ks;
        for (int32_t j = threadIdx.x; j < kd; j += blockDim.x) {
            int32_t p = atomicAdd(&wp, 1);
            out[wb + p] = kb + indices[ks + j];
        }
    }
}

__global__ void extract_unique_kernel(
    const int64_t* __restrict__ keys, int64_t total, int64_t nv,
    int32_t* __restrict__ f, int32_t* __restrict__ s, int32_t* __restrict__ cnt)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t key = keys[idx], u = key/nv, v = key%nv;
    if (u!=v && (idx==0 || key!=keys[idx-1])) {
        int p = atomicAdd(cnt,1); f[p]=(int32_t)u; s[p]=(int32_t)v;
    }
}





__global__ void cosine_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    int64_t num_pairs, float* __restrict__ scores)
{
    int warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_pairs) return;

    int32_t u = first[warp_id], v = second[warp_id];
    int32_t us = offsets[u], ue = offsets[u+1];
    int32_t vs = offsets[v], ve = offsets[v+1];
    int32_t ul = ue-us, vl = ve-vs;
    if (ul==0||vl==0) { if(lane==0) scores[warp_id]=0.0f; return; }

    const int32_t *sn,*ln; const float *sw,*lw; int32_t sl,ll;
    bool u_sm = (ul<=vl);
    if(u_sm){sn=indices+us;sw=weights+us;sl=ul;ln=indices+vs;lw=weights+vs;ll=vl;}
    else{sn=indices+vs;sw=weights+vs;sl=vl;ln=indices+us;lw=weights+us;ll=ul;}

    int per = (sl+31)/32, my_s = lane*per, my_e = my_s+per;
    if(my_e>sl)my_e=sl;
    float dot=0,ns2=0,nl2=0;
    if(my_s<sl) {
        int32_t tgt=sn[my_s];
        int lo=0,hi=ll; while(lo<hi){int m=(lo+hi)>>1;if(ln[m]<tgt)lo=m+1;else hi=m;} int j=lo;
        int32_t last=sn[my_e-1]; lo=j; hi=ll;
        while(lo<hi){int m=(lo+hi)>>1;if(ln[m]<=last)lo=m+1;else hi=m;} int je=lo;
        for(int i=my_s;i<my_e&&j<je;) {
            int32_t si=sn[i],lj=ln[j];
            if(si==lj){float ws=sw[i],wl=lw[j];dot+=ws*wl;ns2+=ws*ws;nl2+=wl*wl;i++;j++;}
            else if(si<lj)i++;else j++;
        }
    }
    for(int m=16;m>0;m>>=1){dot+=__shfl_xor_sync(0xFFFFFFFF,dot,m);
        ns2+=__shfl_xor_sync(0xFFFFFFFF,ns2,m);nl2+=__shfl_xor_sync(0xFFFFFFFF,nl2,m);}
    if(lane==0){float nu,nv2; if(u_sm){nu=ns2;nv2=nl2;}else{nu=nl2;nv2=ns2;}
        float d=sqrtf(nu)*sqrtf(nv2); scores[warp_id]=d>0?dot/d:0.0f;}
}





#define TOPK_BLOCK 256
#define TOPK_IPT 4

__global__ void topk_block_kernel(
    const float* __restrict__ scores,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    int64_t num_pairs, int topk,
    float* __restrict__ out_scores,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second)
{
    typedef cub::BlockRadixSort<float, TOPK_BLOCK, TOPK_IPT, int32_t> BlockSort;
    __shared__ typename BlockSort::TempStorage sort_storage;

    float keys[TOPK_IPT];
    int32_t vals[TOPK_IPT];

    int64_t base = (int64_t)blockIdx.x * TOPK_BLOCK * TOPK_IPT;

    #pragma unroll
    for (int i = 0; i < TOPK_IPT; i++) {
        int64_t idx = base + (int64_t)threadIdx.x * TOPK_IPT + i;
        if (idx < num_pairs) {
            keys[i] = scores[idx];
            vals[i] = (int32_t)idx;
        } else {
            keys[i] = -1e30f;
            vals[i] = -1;
        }
    }

    BlockSort(sort_storage).SortDescending(keys, vals);

    
    int out_base = blockIdx.x * topk;
    int items_needed = (topk + TOPK_IPT - 1) / TOPK_IPT;

    if ((int)threadIdx.x < items_needed) {
        #pragma unroll
        for (int i = 0; i < TOPK_IPT; i++) {
            int pos = (int)threadIdx.x * TOPK_IPT + i;
            if (pos < topk) {
                int32_t orig = vals[i];
                if (orig >= 0) {
                    out_scores[out_base + pos] = keys[i];
                    out_first[out_base + pos] = first[orig];
                    out_second[out_base + pos] = second[orig];
                } else {
                    out_scores[out_base + pos] = -1e30f;
                    out_first[out_base + pos] = -1;
                    out_second[out_base + pos] = -1;
                }
            }
        }
    }
}

static int64_t next_pow2(int64_t n){int64_t p=1;while(p<n)p<<=1;return p;}

}  

similarity_result_float_t cosine_all_pairs_similarity(
    const graph32_t& graph,
    const float* edge_weights,
    const int32_t* vertices,
    std::size_t num_vertices_param,
    std::optional<std::size_t> topk)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    ScratchPool pool = {cache.base, cache.capacity, 0};

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const float* d_weights = edge_weights;
    int32_t nv = graph.number_of_vertices;

    bool has_seeds = (vertices != nullptr && num_vertices_param > 0);
    int32_t num_seeds = has_seeds ? (int32_t)num_vertices_param : nv;
    int64_t topk_param = topk.has_value() ? (int64_t)topk.value() : -1;
    const int32_t* d_seeds = has_seeds ? vertices : nullptr;

    if (num_seeds == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    cudaStream_t stream = 0;
    const int block = 256;

    
    
    

    int64_t* d_counts = pool.alloc<int64_t>(num_seeds);
    { int g=(num_seeds+block-1)/block;
      count_raw_pairs_kernel<<<g,block,0,stream>>>(
          d_offsets,d_indices,d_seeds,num_seeds,d_counts); }

    int64_t* d_scan = pool.alloc<int64_t>(num_seeds+1);
    { void* tmp=nullptr; size_t tb=0;
      cub::DeviceScan::ExclusiveSum(tmp,tb,d_counts,d_scan,num_seeds,stream);
      tmp=pool.alloc<char>(tb); if(!tmp) cudaMalloc(&tmp,tb);
      cub::DeviceScan::ExclusiveSum(tmp,tb,d_counts,d_scan,num_seeds,stream); }

    int64_t total_raw;
    { int64_t lo,lc;
      cudaMemcpyAsync(&lo,d_scan+num_seeds-1,8,cudaMemcpyDeviceToHost,stream);
      cudaMemcpyAsync(&lc,d_counts+num_seeds-1,8,cudaMemcpyDeviceToHost,stream);
      cudaStreamSynchronize(stream); total_raw=lo+lc; }
    if (total_raw == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    int64_t np = 0;
    int32_t *df=nullptr, *ds=nullptr;
    bool pair_first_from_pool=false, pair_second_from_pool=false;

    int64_t ht_cap = next_pow2(total_raw*2);
    size_t ht_mem = ht_cap*8, comp_mem = total_raw*8+256;
    bool use_ht = (ht_mem + comp_mem < 400ULL*1024*1024);

    if (use_ht) {
        int64_t* d_ht = pool.alloc<int64_t>(ht_cap);
        bool htp=(d_ht!=nullptr); if(!d_ht) cudaMalloc(&d_ht,ht_mem);
        { int g=(int)((ht_cap+block-1)/block); init_hashtable<<<g,block,0,stream>>>(d_ht,ht_cap); }
        df=pool.alloc<int32_t>(total_raw); ds=pool.alloc<int32_t>(total_raw);
        pair_first_from_pool=(df!=nullptr); pair_second_from_pool=(ds!=nullptr);
        if(!df) cudaMalloc(&df,total_raw*4); if(!ds) cudaMalloc(&ds,total_raw*4);
        int32_t* dc=pool.alloc<int32_t>(1); if(!dc) cudaMalloc(&dc,4);
        cudaMemsetAsync(dc,0,4,stream);
        insert_pairs_hashtable<<<num_seeds,block,0,stream>>>(
            d_offsets,d_indices,d_seeds,num_seeds,
            (int64_t)nv,d_ht,ht_cap-1,df,ds,dc);
        int32_t hc; cudaMemcpyAsync(&hc,dc,4,cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream); np=hc;
        if(!htp) cudaFree(d_ht);
    } else {
        int64_t* dr=pool.alloc<int64_t>(total_raw);
        bool rp=(dr!=nullptr); if(!dr) cudaMalloc(&dr,total_raw*8);
        write_raw_pairs_kernel<<<num_seeds,block,0,stream>>>(
            d_offsets,d_indices,d_seeds,num_seeds,
            (int64_t)nv,d_scan,dr);
        int nb=1; { int64_t mx=(int64_t)nv*nv;
            while((1LL<<nb)<mx&&nb<64)nb++; }
        int64_t* dso=pool.alloc<int64_t>(total_raw);
        bool sop=(dso!=nullptr); if(!dso) cudaMalloc(&dso,total_raw*8);
        { void* tmp=nullptr; size_t tb=0;
          cub::DeviceRadixSort::SortKeys(tmp,tb,dr,dso,(int)total_raw,0,nb,stream);
          char* t=pool.alloc<char>(tb); bool tp=(t!=nullptr);
          if(!t) cudaMalloc(&t,tb);
          cub::DeviceRadixSort::SortKeys(t,tb,dr,dso,(int)total_raw,0,nb,stream);
          if(!tp&&t) cudaFree(t); }
        if(!rp) cudaFree(dr);
        df=pool.alloc<int32_t>(total_raw); ds=pool.alloc<int32_t>(total_raw);
        pair_first_from_pool=(df!=nullptr); pair_second_from_pool=(ds!=nullptr);
        if(!df) cudaMalloc(&df,total_raw*4); if(!ds) cudaMalloc(&ds,total_raw*4);
        int32_t* dc=pool.alloc<int32_t>(1); if(!dc) cudaMalloc(&dc,4);
        cudaMemsetAsync(dc,0,4,stream);
        { int g=(int)((total_raw+block-1)/block);
          extract_unique_kernel<<<g,block,0,stream>>>(dso,total_raw,(int64_t)nv,df,ds,dc); }
        int32_t hc; cudaMemcpyAsync(&hc,dc,4,cudaMemcpyDeviceToHost,stream);
        cudaStreamSynchronize(stream); np=hc;
        if(!sop) cudaFree(dso);
    }

    if (np == 0) {
        if (!pair_first_from_pool && df) cudaFree(df);
        if (!pair_second_from_pool && ds) cudaFree(ds);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    
    

    int64_t output_size = np;
    if (topk_param >= 0 && topk_param < np) output_size = topk_param;

    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;
    cudaMalloc(&out_first, output_size * sizeof(int32_t));
    cudaMalloc(&out_second, output_size * sizeof(int32_t));
    cudaMalloc(&out_scores, output_size * sizeof(float));

    
    
    

    float* dsc = pool.alloc<float>(np);
    bool scores_from_pool = (dsc!=nullptr); if(!dsc) cudaMalloc(&dsc,np*4);

    
    { int wpb = 8;
      int grid = (int)((np + wpb - 1) / wpb);
      cosine_warp_kernel<<<grid, wpb*32, 0, stream>>>(
          d_offsets, d_indices, d_weights, df, ds, np, dsc); }

    int64_t oc = np;
    if (topk_param >= 0 && topk_param < np) {
        int topk_val = (int)topk_param;
        int elems_per_block = TOPK_BLOCK * TOPK_IPT;
        int num_blocks = (int)((np + elems_per_block - 1) / elems_per_block);
        int64_t stage1_count = (int64_t)num_blocks * topk_val;

        
        float* s1_scores = pool.alloc<float>(stage1_count);
        int32_t* s1_first = pool.alloc<int32_t>(stage1_count);
        int32_t* s1_second = pool.alloc<int32_t>(stage1_count);
        bool s1s_p = (s1_scores!=nullptr), s1f_p = (s1_first!=nullptr), s1sc_p = (s1_second!=nullptr);
        if (!s1_scores) cudaMalloc(&s1_scores, stage1_count*4);
        if (!s1_first) cudaMalloc(&s1_first, stage1_count*4);
        if (!s1_second) cudaMalloc(&s1_second, stage1_count*4);

        topk_block_kernel<<<num_blocks, TOPK_BLOCK, 0, stream>>>(
            dsc, df, ds, np, topk_val, s1_scores, s1_first, s1_second);

        if (stage1_count <= topk_val) {
            cudaMemcpyAsync(out_scores, s1_scores, stage1_count*4, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_first, s1_first, stage1_count*4, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_second, s1_second, stage1_count*4, cudaMemcpyDeviceToDevice, stream);
            oc = stage1_count;
        } else {
            
            thrust::device_ptr<float> sp_ptr(s1_scores);
            thrust::device_ptr<int32_t> fp_ptr(s1_first), sc_ptr(s1_second);
            thrust::sort_by_key(thrust::cuda::par.on(stream), sp_ptr, sp_ptr+stage1_count,
                thrust::make_zip_iterator(thrust::make_tuple(fp_ptr, sc_ptr)),
                thrust::greater<float>());

            cudaMemcpyAsync(out_scores, s1_scores, topk_val*4, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_first, s1_first, topk_val*4, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_second, s1_second, topk_val*4, cudaMemcpyDeviceToDevice, stream);
            oc = topk_val;
        }

        if (!s1s_p) cudaFree(s1_scores);
        if (!s1f_p) cudaFree(s1_first);
        if (!s1sc_p) cudaFree(s1_second);
    } else {
        
        if (oc > 0) {
            cudaMemcpyAsync(out_first, df, oc*4, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_second, ds, oc*4, cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_scores, dsc, oc*4, cudaMemcpyDeviceToDevice, stream);
        }
    }

    cudaStreamSynchronize(stream);

    
    if (!scores_from_pool && dsc) cudaFree(dsc);
    if (!pair_first_from_pool && df) cudaFree(df);
    if (!pair_second_from_pool && ds) cudaFree(ds);

    return {out_first, out_second, out_scores, (std::size_t)oc};
}

}  
