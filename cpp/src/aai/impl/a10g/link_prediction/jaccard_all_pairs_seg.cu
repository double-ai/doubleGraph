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
#include <optional>

namespace aai {

namespace {




struct Cache : Cacheable {
    
};




static int bits_needed(int64_t mv) {
    int b = 0;
    while ((1LL << b) <= mv) b++;
    return b < 1 ? 1 : b;
}




template <typename T>
struct DevBuf {
    T* ptr = nullptr;
    void alloc(int64_t count) {
        if (count > 0) cudaMalloc(&ptr, count * sizeof(T));
    }
    void free() {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
    }
    operator T*() { return ptr; }
    operator const T*() const { return ptr; }
};




__device__ __forceinline__ int32_t galloping_search(
    const int32_t* arr, int32_t start, int32_t end, int32_t target
) {
    if (start >= end || arr[start] >= target) return start;
    int32_t pos = start, step = 1;
    while (pos + step < end && arr[pos + step] < target) { pos += step; step <<= 1; }
    int32_t lo = pos + 1, hi = (pos + step < end) ? pos + step + 1 : end;
    while (lo < hi) { int32_t m = lo + (hi-lo)/2; if(arr[m]<target) lo=m+1; else hi=m; }
    return lo;
}

__device__ __forceinline__ int32_t lower_bound_dev(const int32_t* arr, int32_t lo, int32_t hi, int32_t target) {
    while (lo < hi) { int32_t m = lo+(hi-lo)/2; if(arr[m]<target) lo=m+1; else hi=m; }
    return lo;
}

__device__ __forceinline__ int32_t upper_bound_dev(const int32_t* arr, int32_t lo, int32_t hi, int32_t target) {
    while (lo < hi) { int32_t m = lo+(hi-lo)/2; if(arr[m]<=target) lo=m+1; else hi=m; }
    return lo;
}




__device__ int32_t warp_intersect_multigraph(
    const int32_t* __restrict__ indices,
    int32_t us, int32_t ue, int32_t vs, int32_t ve,
    int lane
) {
    int32_t du = ue - us, dv = ve - vs;
    if (du == 0 || dv == 0) return 0;

    int32_t ss, se, ls, le;
    if (du <= dv) { ss = us; se = ue; ls = vs; le = ve; }
    else { ss = vs; se = ve; ls = us; le = ue; }

    if (indices[se-1] < indices[ls] || indices[le-1] < indices[ss]) return 0;

    int32_t local_count = 0;

    for (int32_t i = ss + lane; i < se; i += 32) {
        int32_t val = indices[i];
        if (i == ss || indices[i-1] != val) {
            int32_t s_ub = upper_bound_dev(indices, i, se, val);
            int32_t count_short = s_ub - i;

            int32_t l_lb = lower_bound_dev(indices, ls, le, val);
            if (l_lb < le && indices[l_lb] == val) {
                int32_t l_ub = upper_bound_dev(indices, l_lb, le, val);
                int32_t count_long = l_ub - l_lb;
                local_count += (count_short < count_long) ? count_short : count_long;
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);

    return local_count;
}


__device__ int32_t thread_intersect_simple(
    const int32_t* __restrict__ indices,
    int32_t us, int32_t ue, int32_t vs, int32_t ve
) {
    int32_t du = ue-us, dv = ve-vs;
    if (du == 0 || dv == 0) return 0;
    if (indices[ue-1] < indices[vs] || indices[ve-1] < indices[us]) return 0;

    int32_t ss, se, ls, le;
    if (du <= dv) { ss=us; se=ue; ls=vs; le=ve; }
    else { ss=vs; se=ve; ls=us; le=ue; }
    int32_t short_len = se-ss, long_len = le-ls;

    if (long_len > short_len * 10) {
        int32_t count = 0, j = ls;
        for (int32_t i = ss; i < se && j < le; i++) {
            j = galloping_search(indices, j, le, indices[i]);
            if (j < le && indices[j] == indices[i]) { count++; j++; }
        }
        return count;
    }

    int32_t i = ss, j = ls;
    if (indices[i] < indices[j]) i = lower_bound_dev(indices, i, se, indices[j]);
    else if (indices[j] < indices[i]) j = lower_bound_dev(indices, j, le, indices[i]);

    int32_t count = 0;
    while (i < se && j < le) {
        int32_t a = indices[i], b = indices[j];
        if (a == b) { count++; i++; j++; }
        else if (a < b) i++;
        else j++;
    }
    return count;
}




__global__ void count_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ counts,
    bool is_multigraph
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;
    int32_t u = seeds ? seeds[idx] : idx;
    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int32_t u_deg = u_end - u_start;
    int64_t count = 0;
    if (!is_multigraph) {
        for (int32_t i = u_start; i < u_end; i++)
            count += (int64_t)(offsets[indices[i] + 1] - offsets[indices[i]]);
        count -= (int64_t)u_deg;
    } else {
        for (int32_t i = u_start; i < u_end; i++) {
            int32_t w = indices[i];
            int32_t ws = offsets[w], wd = offsets[w+1]-ws;
            int32_t lb = lower_bound_dev(indices, ws, ws+wd, u);
            int32_t ub = upper_bound_dev(indices, lb, ws+wd, u);
            count += (int64_t)(wd - (ub - lb));
        }
    }
    counts[idx] = count;
}




__global__ void enumerate_pairs_i32_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int32_t num_vertices,
    const int64_t* __restrict__ pair_offsets,
    int32_t* __restrict__ pair_keys
) {
    int32_t seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;
    int32_t u = seeds ? seeds[seed_idx] : seed_idx;
    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int64_t base = pair_offsets[seed_idx];
    int32_t seed_key = seed_idx * num_vertices;

    __shared__ unsigned long long s_pos;
    if (threadIdx.x == 0) s_pos = 0;
    __syncthreads();

    for (int32_t i = u_start; i < u_end; i++) {
        int32_t w = indices[i];
        int32_t ws = offsets[w], we = offsets[w + 1];
        for (int32_t j = ws + (int32_t)threadIdx.x; j < we; j += (int32_t)blockDim.x) {
            int32_t v = indices[j];
            if (v != u) {
                unsigned long long pos = atomicAdd(&s_pos, 1ull);
                pair_keys[base + pos] = seed_key + v;
            }
        }
        __syncthreads();
    }
}


__global__ void enumerate_pairs_i64_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t num_vertices_i64,
    const int64_t* __restrict__ pair_offsets,
    int64_t* __restrict__ pair_keys
) {
    int32_t seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;
    int32_t u = seeds ? seeds[seed_idx] : seed_idx;
    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int64_t base = pair_offsets[seed_idx];
    int64_t seed_key = (int64_t)seed_idx * num_vertices_i64;

    __shared__ unsigned long long s_pos;
    if (threadIdx.x == 0) s_pos = 0;
    __syncthreads();

    for (int32_t i = u_start; i < u_end; i++) {
        int32_t w = indices[i];
        int32_t ws = offsets[w], we = offsets[w + 1];
        for (int32_t j = ws + (int32_t)threadIdx.x; j < we; j += (int32_t)blockDim.x) {
            int32_t v = indices[j];
            if (v != u) {
                unsigned long long pos = atomicAdd(&s_pos, 1ull);
                pair_keys[base + (int64_t)pos] = seed_key + v;
            }
        }
        __syncthreads();
    }
}




__global__ void jaccard_rle_i32_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ seeds,
    int32_t nv,
    const int32_t* __restrict__ uk,
    const int32_t* __restrict__ ic,
    int64_t n,
    int32_t* __restrict__ fo,
    int32_t* __restrict__ so,
    float* __restrict__ sc
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int32_t key = uk[idx]; int32_t si = key / nv; int32_t v = key % nv;
    int32_t u = seeds ? seeds[si] : si;
    fo[idx] = u; so[idx] = v;
    int32_t isct = ic[idx];
    int32_t un = (offsets[u+1]-offsets[u])+(offsets[v+1]-offsets[v])-isct;
    sc[idx] = (un > 0) ? (float)isct/(float)un : 0.0f;
}

__global__ void jaccard_rle_i64_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ seeds,
    int64_t nv64,
    const int64_t* __restrict__ uk,
    const int32_t* __restrict__ ic,
    int64_t n,
    int32_t* __restrict__ fo,
    int32_t* __restrict__ so,
    float* __restrict__ sc
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int64_t key = uk[idx]; int32_t si = (int32_t)(key/nv64); int32_t v = (int32_t)(key%nv64);
    int32_t u = seeds ? seeds[si] : si;
    fo[idx] = u; so[idx] = v;
    int32_t isct = ic[idx];
    int32_t un = (offsets[u+1]-offsets[u])+(offsets[v+1]-offsets[v])-isct;
    sc[idx] = (un > 0) ? (float)isct/(float)un : 0.0f;
}




__global__ void jaccard_warp_merge_i32_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t nv,
    const int32_t* __restrict__ unique_keys,
    int64_t num_unique,
    int32_t* __restrict__ fo,
    int32_t* __restrict__ so,
    float* __restrict__ sc
) {
    int64_t warp_idx = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_idx >= num_unique) return;

    int32_t key = unique_keys[warp_idx];
    int32_t si = key / nv, v = key % nv;
    int32_t u = seeds ? seeds[si] : si;

    if (lane == 0) { fo[warp_idx] = u; so[warp_idx] = v; }

    int32_t us = offsets[u], ue = offsets[u+1];
    int32_t vs = offsets[v], ve = offsets[v+1];
    int32_t du = ue-us, dv = ve-vs;

    int32_t intersection = warp_intersect_multigraph(indices, us, ue, vs, ve, lane);

    if (lane == 0) {
        int32_t un = du + dv - intersection;
        sc[warp_idx] = (un > 0) ? (float)intersection/(float)un : 0.0f;
    }
}

__global__ void jaccard_warp_merge_i64_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int64_t nv64,
    const int64_t* __restrict__ unique_keys,
    int64_t num_unique,
    int32_t* __restrict__ fo,
    int32_t* __restrict__ so,
    float* __restrict__ sc
) {
    int64_t warp_idx = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_idx >= num_unique) return;

    int64_t key = unique_keys[warp_idx];
    int32_t si = (int32_t)(key/nv64), v = (int32_t)(key%nv64);
    int32_t u = seeds ? seeds[si] : si;

    if (lane == 0) { fo[warp_idx] = u; so[warp_idx] = v; }

    int32_t us = offsets[u], ue = offsets[u+1];
    int32_t vs = offsets[v], ve = offsets[v+1];
    int32_t du = ue-us, dv = ve-vs;

    int32_t intersection = warp_intersect_multigraph(indices, us, ue, vs, ve, lane);

    if (lane == 0) {
        int32_t un = du + dv - intersection;
        sc[warp_idx] = (un > 0) ? (float)intersection/(float)un : 0.0f;
    }
}


__global__ void decode_keys_i32_kernel(const int32_t* __restrict__ keys, const int32_t* __restrict__ seeds,
    int64_t n, int32_t nv, int32_t* __restrict__ f, int32_t* __restrict__ s) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int32_t key = keys[idx]; f[idx] = seeds ? seeds[key/nv] : key/nv; s[idx] = key % nv;
}
__global__ void decode_keys_i64_kernel(const int64_t* __restrict__ keys, const int32_t* __restrict__ seeds,
    int64_t n, int64_t nv64, int32_t* __restrict__ f, int32_t* __restrict__ s) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int64_t key = keys[idx]; int32_t si=(int32_t)(key/nv64);
    f[idx] = seeds ? seeds[si] : si; s[idx] = (int32_t)(key%nv64);
}




static void launch_count_pairs(const int32_t* o, const int32_t* idx, const int32_t* se,
    int32_t ns, int64_t* c, bool mg) {
    if(!ns)return; int bl=256,gr=(ns+bl-1)/bl;
    count_pairs_kernel<<<gr,bl>>>(o,idx,se,ns,c,mg);
}

static void launch_enumerate_i32(const int32_t* o, const int32_t* idx, const int32_t* se,
    int32_t ns, int32_t nv, const int64_t* po, int32_t* pk) {
    if(!ns)return; enumerate_pairs_i32_kernel<<<ns,512>>>(o,idx,se,ns,nv,po,pk);
}
static void launch_enumerate_i64(const int32_t* o, const int32_t* idx, const int32_t* se,
    int32_t ns, int64_t nv64, const int64_t* po, int64_t* pk) {
    if(!ns)return; enumerate_pairs_i64_kernel<<<ns,512>>>(o,idx,se,ns,nv64,po,pk);
}

static void launch_jaccard_rle_i32(const int32_t* o, const int32_t* se, int32_t nv,
    const int32_t* uk, const int32_t* c, int64_t n, int32_t* f, int32_t* s, float* sc) {
    if(!n)return; int bl=256; int gr=(int)((n+bl-1)/bl);
    jaccard_rle_i32_kernel<<<gr,bl>>>(o,se,nv,uk,c,n,f,s,sc);
}
static void launch_jaccard_rle_i64(const int32_t* o, const int32_t* se, int64_t nv64,
    const int64_t* uk, const int32_t* c, int64_t n, int32_t* f, int32_t* s, float* sc) {
    if(!n)return; int bl=256; int gr=(int)((n+bl-1)/bl);
    jaccard_rle_i64_kernel<<<gr,bl>>>(o,se,nv64,uk,c,n,f,s,sc);
}
static void launch_jaccard_merge_i32(const int32_t* o, const int32_t* idx, const int32_t* se,
    int32_t nv, const int32_t* uk, int64_t n, int32_t* f, int32_t* s, float* sc) {
    if(!n)return;
    int threads = 256;
    int warps_per_block = threads / 32;
    int grid = (int)((n + warps_per_block - 1) / warps_per_block);
    jaccard_warp_merge_i32_kernel<<<grid,threads>>>(o,idx,se,nv,uk,n,f,s,sc);
}
static void launch_jaccard_merge_i64(const int32_t* o, const int32_t* idx, const int32_t* se,
    int64_t nv64, const int64_t* uk, int64_t n, int32_t* f, int32_t* s, float* sc) {
    if(!n)return;
    int threads = 256;
    int warps_per_block = threads / 32;
    int grid = (int)((n + warps_per_block - 1) / warps_per_block);
    jaccard_warp_merge_i64_kernel<<<grid,threads>>>(o,idx,se,nv64,uk,n,f,s,sc);
}
static void launch_decode_i32(const int32_t* k, const int32_t* se, int64_t n, int32_t nv, int32_t* f, int32_t* s) {
    if(!n)return; int bl=256; int gr=(int)((n+bl-1)/bl);
    decode_keys_i32_kernel<<<gr,bl>>>(k,se,n,nv,f,s);
}
static void launch_decode_i64(const int64_t* k, const int32_t* se, int64_t n, int64_t nv64, int32_t* f, int32_t* s) {
    if(!n)return; int bl=256; int gr=(int)((n+bl-1)/bl);
    decode_keys_i64_kernel<<<gr,bl>>>(k,se,n,nv64,f,s);
}




static size_t cub_scan_i64_temp(int64_t n){size_t tb=0;cub::DeviceScan::ExclusiveSum(nullptr,tb,(int64_t*)0,(int64_t*)0,n);return tb;}
static void cub_scan_i64(void*t,size_t tb,const int64_t*i,int64_t*o,int64_t n){cub::DeviceScan::ExclusiveSum(t,tb,i,o,n);}
static size_t cub_sort_i32_temp(int64_t n,int b){size_t tb=0;cub::DeviceRadixSort::SortKeys(nullptr,tb,(int32_t*)0,(int32_t*)0,n,0,b);return tb;}
static void cub_sort_i32(void*t,size_t tb,const int32_t*i,int32_t*o,int64_t n,int b){cub::DeviceRadixSort::SortKeys(t,tb,i,o,n,0,b);}
static size_t cub_sort_i64_temp(int64_t n,int b){size_t tb=0;cub::DeviceRadixSort::SortKeys(nullptr,tb,(int64_t*)0,(int64_t*)0,n,0,b);return tb;}
static void cub_sort_i64(void*t,size_t tb,const int64_t*i,int64_t*o,int64_t n,int b){cub::DeviceRadixSort::SortKeys(t,tb,i,o,n,0,b);}
static size_t cub_rle_i32_temp(int64_t n){size_t tb=0;cub::DeviceRunLengthEncode::Encode(nullptr,tb,(int32_t*)0,(int32_t*)0,(int32_t*)0,(int32_t*)0,n);return tb;}
static void cub_rle_i32(void*t,size_t tb,const int32_t*i,int32_t*uk,int32_t*c,int32_t*nr,int64_t n){cub::DeviceRunLengthEncode::Encode(t,tb,i,uk,c,nr,n);}
static size_t cub_rle_i64_temp(int64_t n){size_t tb=0;cub::DeviceRunLengthEncode::Encode(nullptr,tb,(int64_t*)0,(int64_t*)0,(int32_t*)0,(int32_t*)0,n);return tb;}
static void cub_rle_i64(void*t,size_t tb,const int64_t*i,int64_t*uk,int32_t*c,int32_t*nr,int64_t n){cub::DeviceRunLengthEncode::Encode(t,tb,i,uk,c,nr,n);}
static size_t cub_unique_i32_temp(int64_t n){size_t tb=0;cub::DeviceSelect::Unique(nullptr,tb,(int32_t*)0,(int32_t*)0,(int32_t*)0,n);return tb;}
static void cub_unique_i32(void*t,size_t tb,const int32_t*i,int32_t*o,int32_t*ns,int64_t n){cub::DeviceSelect::Unique(t,tb,i,o,ns,n);}
static size_t cub_unique_i64_temp(int64_t n){size_t tb=0;cub::DeviceSelect::Unique(nullptr,tb,(int64_t*)0,(int64_t*)0,(int32_t*)0,n);return tb;}
static void cub_unique_i64(void*t,size_t tb,const int64_t*i,int64_t*o,int32_t*ns,int64_t n){cub::DeviceSelect::Unique(t,tb,i,o,ns,n);}
static size_t cub_sortpd_fi32_temp(int64_t n){size_t tb=0;cub::DeviceRadixSort::SortPairsDescending(nullptr,tb,(float*)0,(float*)0,(int32_t*)0,(int32_t*)0,n);return tb;}
static void cub_sortpd_fi32(void*t,size_t tb,const float*ki,float*ko,const int32_t*vi,int32_t*vo,int64_t n){cub::DeviceRadixSort::SortPairsDescending(t,tb,ki,ko,vi,vo,n);}
static size_t cub_sortpd_fi64_temp(int64_t n){size_t tb=0;cub::DeviceRadixSort::SortPairsDescending(nullptr,tb,(float*)0,(float*)0,(int64_t*)0,(int64_t*)0,n);return tb;}
static void cub_sortpd_fi64(void*t,size_t tb,const float*ki,float*ko,const int64_t*vi,int64_t*vo,int64_t n){cub::DeviceRadixSort::SortPairsDescending(t,tb,ki,ko,vi,vo,n);}




static void* alloc_tmp(size_t bytes) {
    void* ptr = nullptr;
    if (bytes > 0) cudaMalloc(&ptr, bytes + 16);
    return ptr;
}




static similarity_result_float_t run_i32(
    const int32_t* d_off, const int32_t* d_idx, const int32_t* d_seeds,
    int32_t ns, int32_t nv, bool is_mg, std::optional<std::size_t> topk,
    int64_t* d_poff, int64_t total, int64_t nt
) {
    
    DevBuf<int32_t> pk; pk.alloc(total);
    launch_enumerate_i32(d_off, d_idx, d_seeds, ns, nv, d_poff, pk);

    
    int64_t maxk = (int64_t)(ns-1)*nv + (nv-1);
    int nb = bits_needed(maxk);
    size_t stb = cub_sort_i32_temp(nt, nb);
    void* stmp = alloc_tmp(stb);
    DevBuf<int32_t> sk; sk.alloc(total);
    cub_sort_i32(stmp, stb, pk, sk, nt, nb);
    cudaFree(stmp);
    pk.free();

    
    DevBuf<int32_t> uk; uk.alloc(total);
    DevBuf<int32_t> nrt; nrt.alloc(1);

    int64_t nu;
    if (!is_mg) {
        DevBuf<int32_t> rc; rc.alloc(total);
        size_t rtb = cub_rle_i32_temp(nt);
        void* rtmp = alloc_tmp(rtb);
        cub_rle_i32(rtmp, rtb, sk, uk, rc, nrt, nt);
        cudaFree(rtmp);
        sk.free();

        int32_t nuh; cudaMemcpy(&nuh, (int32_t*)nrt, 4, cudaMemcpyDeviceToHost);
        nrt.free();
        nu = nuh;
        if (nu == 0) { uk.free(); rc.free(); return {nullptr, nullptr, nullptr, 0}; }

        int32_t* fo; cudaMalloc(&fo, nu * sizeof(int32_t));
        int32_t* so; cudaMalloc(&so, nu * sizeof(int32_t));
        float* sc; cudaMalloc(&sc, nu * sizeof(float));
        launch_jaccard_rle_i32(d_off, d_seeds, nv, uk, rc, nu, fo, so, sc);

        if (topk.has_value() && nu > (int64_t)topk.value()) {
            int64_t topk_val = (int64_t)topk.value();
            size_t tb = cub_sortpd_fi32_temp(nu);
            void* tmp = alloc_tmp(tb);
            DevBuf<float> ss; ss.alloc(nu);
            DevBuf<int32_t> sv; sv.alloc(nu);
            cub_sortpd_fi32(tmp, tb, sc, ss, uk, sv, nu);
            cudaFree(tmp);
            cudaFree(sc); cudaFree(fo); cudaFree(so);
            rc.free();

            int32_t* tf; cudaMalloc(&tf, topk_val * sizeof(int32_t));
            int32_t* ts; cudaMalloc(&ts, topk_val * sizeof(int32_t));
            float* tsc; cudaMalloc(&tsc, topk_val * sizeof(float));
            cudaMemcpy(tsc, (float*)ss, topk_val * sizeof(float), cudaMemcpyDeviceToDevice);
            launch_decode_i32(sv, d_seeds, topk_val, nv, tf, ts);
            ss.free(); sv.free(); uk.free();
            return {tf, ts, tsc, (std::size_t)topk_val};
        }
        uk.free(); rc.free();
        return {fo, so, sc, (std::size_t)nu};
    } else {
        size_t utb = cub_unique_i32_temp(nt);
        void* utmp = alloc_tmp(utb);
        cub_unique_i32(utmp, utb, sk, uk, nrt, nt);
        cudaFree(utmp);
        sk.free();

        int32_t nuh; cudaMemcpy(&nuh, (int32_t*)nrt, 4, cudaMemcpyDeviceToHost);
        nrt.free();
        nu = nuh;
        if (nu == 0) { uk.free(); return {nullptr, nullptr, nullptr, 0}; }

        int32_t* fo; cudaMalloc(&fo, nu * sizeof(int32_t));
        int32_t* so; cudaMalloc(&so, nu * sizeof(int32_t));
        float* sc; cudaMalloc(&sc, nu * sizeof(float));
        launch_jaccard_merge_i32(d_off, d_idx, d_seeds, nv, uk, nu, fo, so, sc);

        if (topk.has_value() && nu > (int64_t)topk.value()) {
            int64_t topk_val = (int64_t)topk.value();
            size_t tb = cub_sortpd_fi32_temp(nu);
            void* tmp = alloc_tmp(tb);
            DevBuf<float> ss; ss.alloc(nu);
            DevBuf<int32_t> sv; sv.alloc(nu);
            cub_sortpd_fi32(tmp, tb, sc, ss, uk, sv, nu);
            cudaFree(tmp);
            cudaFree(sc); cudaFree(fo); cudaFree(so);

            int32_t* tf; cudaMalloc(&tf, topk_val * sizeof(int32_t));
            int32_t* ts; cudaMalloc(&ts, topk_val * sizeof(int32_t));
            float* tsc; cudaMalloc(&tsc, topk_val * sizeof(float));
            cudaMemcpy(tsc, (float*)ss, topk_val * sizeof(float), cudaMemcpyDeviceToDevice);
            launch_decode_i32(sv, d_seeds, topk_val, nv, tf, ts);
            ss.free(); sv.free(); uk.free();
            return {tf, ts, tsc, (std::size_t)topk_val};
        }
        uk.free();
        return {fo, so, sc, (std::size_t)nu};
    }
}




static similarity_result_float_t run_i64(
    const int32_t* d_off, const int32_t* d_idx, const int32_t* d_seeds,
    int32_t ns, int64_t nv64, bool is_mg, std::optional<std::size_t> topk,
    int64_t* d_poff, int64_t total, int64_t nt
) {
    DevBuf<int64_t> pk; pk.alloc(total);
    launch_enumerate_i64(d_off, d_idx, d_seeds, ns, nv64, d_poff, pk);

    int64_t maxk = (int64_t)(ns-1)*nv64 + (nv64-1);
    int nb = bits_needed(maxk);
    size_t stb = cub_sort_i64_temp(nt, nb);
    void* stmp = alloc_tmp(stb);
    DevBuf<int64_t> sk; sk.alloc(total);
    cub_sort_i64(stmp, stb, pk, sk, nt, nb);
    cudaFree(stmp);
    pk.free();

    DevBuf<int64_t> uk; uk.alloc(total);
    DevBuf<int32_t> nrt; nrt.alloc(1);

    int64_t nu;
    if (!is_mg) {
        DevBuf<int32_t> rc; rc.alloc(total);
        size_t rtb = cub_rle_i64_temp(nt);
        void* rtmp = alloc_tmp(rtb);
        cub_rle_i64(rtmp, rtb, sk, uk, rc, nrt, nt);
        cudaFree(rtmp);
        sk.free();

        int32_t nuh; cudaMemcpy(&nuh, (int32_t*)nrt, 4, cudaMemcpyDeviceToHost);
        nrt.free();
        nu = nuh;
        if (nu == 0) { uk.free(); rc.free(); return {nullptr, nullptr, nullptr, 0}; }

        int32_t* fo; cudaMalloc(&fo, nu * sizeof(int32_t));
        int32_t* so; cudaMalloc(&so, nu * sizeof(int32_t));
        float* sc; cudaMalloc(&sc, nu * sizeof(float));
        launch_jaccard_rle_i64(d_off, d_seeds, nv64, uk, rc, nu, fo, so, sc);

        if (topk.has_value() && nu > (int64_t)topk.value()) {
            int64_t topk_val = (int64_t)topk.value();
            size_t tb = cub_sortpd_fi64_temp(nu);
            void* tmp = alloc_tmp(tb);
            DevBuf<float> ss; ss.alloc(nu);
            DevBuf<int64_t> sv; sv.alloc(nu);
            cub_sortpd_fi64(tmp, tb, sc, ss, uk, sv, nu);
            cudaFree(tmp);
            cudaFree(sc); cudaFree(fo); cudaFree(so);
            rc.free();

            int32_t* tf; cudaMalloc(&tf, topk_val * sizeof(int32_t));
            int32_t* ts; cudaMalloc(&ts, topk_val * sizeof(int32_t));
            float* tsc; cudaMalloc(&tsc, topk_val * sizeof(float));
            cudaMemcpy(tsc, (float*)ss, topk_val * sizeof(float), cudaMemcpyDeviceToDevice);
            launch_decode_i64(sv, d_seeds, topk_val, nv64, tf, ts);
            ss.free(); sv.free(); uk.free();
            return {tf, ts, tsc, (std::size_t)topk_val};
        }
        uk.free(); rc.free();
        return {fo, so, sc, (std::size_t)nu};
    } else {
        size_t utb = cub_unique_i64_temp(nt);
        void* utmp = alloc_tmp(utb);
        cub_unique_i64(utmp, utb, sk, uk, nrt, nt);
        cudaFree(utmp);
        sk.free();

        int32_t nuh; cudaMemcpy(&nuh, (int32_t*)nrt, 4, cudaMemcpyDeviceToHost);
        nrt.free();
        nu = nuh;
        if (nu == 0) { uk.free(); return {nullptr, nullptr, nullptr, 0}; }

        int32_t* fo; cudaMalloc(&fo, nu * sizeof(int32_t));
        int32_t* so; cudaMalloc(&so, nu * sizeof(int32_t));
        float* sc; cudaMalloc(&sc, nu * sizeof(float));
        launch_jaccard_merge_i64(d_off, d_idx, d_seeds, nv64, uk, nu, fo, so, sc);

        if (topk.has_value() && nu > (int64_t)topk.value()) {
            int64_t topk_val = (int64_t)topk.value();
            size_t tb = cub_sortpd_fi64_temp(nu);
            void* tmp = alloc_tmp(tb);
            DevBuf<float> ss; ss.alloc(nu);
            DevBuf<int64_t> sv; sv.alloc(nu);
            cub_sortpd_fi64(tmp, tb, sc, ss, uk, sv, nu);
            cudaFree(tmp);
            cudaFree(sc); cudaFree(fo); cudaFree(so);

            int32_t* tf; cudaMalloc(&tf, topk_val * sizeof(int32_t));
            int32_t* ts; cudaMalloc(&ts, topk_val * sizeof(int32_t));
            float* tsc; cudaMalloc(&tsc, topk_val * sizeof(float));
            cudaMemcpy(tsc, (float*)ss, topk_val * sizeof(float), cudaMemcpyDeviceToDevice);
            launch_decode_i64(sv, d_seeds, topk_val, nv64, tf, ts);
            ss.free(); sv.free(); uk.free();
            return {tf, ts, tsc, (std::size_t)topk_val};
        }
        uk.free();
        return {fo, so, sc, (std::size_t)nu};
    }
}

}  




similarity_result_float_t jaccard_all_pairs_similarity_seg(
    const graph32_t& graph,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk
) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    bool is_mg = graph.is_multigraph;
    int64_t nv64 = (int64_t)nv;

    const int32_t* d_seeds = vertices;
    int32_t ns;
    if (vertices != nullptr && num_vertices > 0) {
        ns = (int32_t)num_vertices;
    } else {
        d_seeds = nullptr;
        ns = nv;
    }

    if (ns == 0) return {nullptr, nullptr, nullptr, 0};

    
    int64_t max_key = (int64_t)(ns - 1) * nv64 + (nv64 - 1);
    bool use_i32 = (max_key <= INT32_MAX);

    
    int64_t* d_counts; cudaMalloc(&d_counts, (int64_t)ns * sizeof(int64_t));
    launch_count_pairs(d_off, d_idx, d_seeds, ns, d_counts, is_mg);

    
    int64_t* d_poff; cudaMalloc(&d_poff, (int64_t)ns * sizeof(int64_t));
    size_t stb = cub_scan_i64_temp(ns);
    void* stmp = alloc_tmp(stb);
    cub_scan_i64(stmp, stb, d_counts, d_poff, ns);
    cudaFree(stmp);

    
    int64_t hl[2];
    cudaMemcpy(hl, d_poff + ns - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(hl+1, d_counts + ns - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
    int64_t total = hl[0] + hl[1];
    cudaFree(d_counts);

    if (total == 0) { cudaFree(d_poff); return {nullptr, nullptr, nullptr, 0}; }

    int64_t nt = total;

    similarity_result_float_t result;
    if (use_i32) {
        result = run_i32(d_off, d_idx, d_seeds, ns, nv, is_mg, topk, d_poff, total, nt);
    } else {
        result = run_i64(d_off, d_idx, d_seeds, ns, nv64, is_mg, topk, d_poff, total, nt);
    }
    cudaFree(d_poff);
    return result;
}

}  
