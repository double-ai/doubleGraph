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

namespace aai {

namespace {




struct BumpAlloc {
    char* base;
    size_t capacity;
    size_t offset;

    __host__ BumpAlloc() : base(nullptr), capacity(0), offset(0) {}
    __host__ BumpAlloc(void* buf, size_t cap) : base((char*)buf), capacity(cap), offset(0) {}

    __host__ void* alloc(size_t bytes, size_t align = 256) {
        offset = (offset + align - 1) & ~(align - 1);
        void* ptr = base + offset;
        offset += bytes;
        return ptr;
    }
    __host__ void reset() { offset = 0; }

    template<typename T>
    __host__ T* alloc_array(size_t count) {
        return (T*)alloc(count * sizeof(T));
    }
};





__global__ void gather_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ seeds,
    int32_t* __restrict__ degrees,
    int32_t num_seeds)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_seeds) {
        int u = seeds[idx];
        degrees[idx] = offsets[u + 1] - offsets[u];
    }
}

__global__ void fill_seed_neighbors_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    const int32_t* __restrict__ seed_nb_offsets,
    int32_t* __restrict__ pair_seed_idx,
    int32_t* __restrict__ pair_neighbors,
    int64_t* __restrict__ expansion_sizes,  
    int32_t num_seeds)
{
    for (int seed_idx = blockIdx.x; seed_idx < num_seeds; seed_idx += gridDim.x) {
        int u = seeds[seed_idx];
        int u_start = offsets[u];
        int u_end = offsets[u + 1];
        int deg = u_end - u_start;
        int out_base = seed_nb_offsets[seed_idx];
        for (int i = threadIdx.x; i < deg; i += blockDim.x) {
            int w = indices[u_start + i];
            pair_seed_idx[out_base + i] = seed_idx;
            pair_neighbors[out_base + i] = w;
            expansion_sizes[out_base + i] = (int64_t)(offsets[w + 1] - offsets[w]);
        }
    }
}

__global__ void compute_expansion_sizes_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ pair_neighbors,
    int64_t* __restrict__ expansion_sizes,
    int32_t num_pairs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pairs) {
        int w = pair_neighbors[idx];
        expansion_sizes[idx] = (int64_t)(offsets[w + 1] - offsets[w]);
    }
}


__global__ void expand_2hop_kernel_i64(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pair_seed_idx,
    const int32_t* __restrict__ pair_neighbors,
    const int64_t* __restrict__ expansion_offsets,
    int64_t* __restrict__ out_keys,
    int64_t total_expansion,
    int32_t num_pairs,
    int64_t multiplier)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_expansion) return;
    int lo = 0, hi = num_pairs - 1;
    while (lo < hi) { int mid = lo + (hi - lo + 1) / 2; if (expansion_offsets[mid] <= idx) lo = mid; else hi = mid - 1; }
    int w = pair_neighbors[lo];
    int w_start = offsets[w];
    int pos = (int)(idx - expansion_offsets[lo]);
    int v = indices[w_start + pos];
    out_keys[idx] = (int64_t)pair_seed_idx[lo] * multiplier + (int64_t)v;
}


__global__ void expand_2hop_kernel_i32(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pair_seed_idx,
    const int32_t* __restrict__ pair_neighbors,
    const int64_t* __restrict__ expansion_offsets,
    int32_t* __restrict__ out_keys,
    int64_t total_expansion,
    int32_t num_pairs,
    int32_t multiplier)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_expansion) return;
    int lo = 0, hi = num_pairs - 1;
    while (lo < hi) { int mid = lo + (hi - lo + 1) / 2; if (expansion_offsets[mid] <= idx) lo = mid; else hi = mid - 1; }
    int w = pair_neighbors[lo];
    int w_start = offsets[w];
    int pos = (int)(idx - expansion_offsets[lo]);
    int v = indices[w_start + pos];
    out_keys[idx] = pair_seed_idx[lo] * multiplier + v;
}


__global__ void mark_self_pairs_i64(int64_t* keys, int64_t n, int64_t multiplier, const int32_t* seeds) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int64_t key = keys[idx];
    int32_t seed_idx = (int32_t)(key / multiplier);
    int32_t v = (int32_t)(key % multiplier);
    if (v == seeds[seed_idx]) keys[idx] = -1LL;
}


__global__ void mark_self_pairs_i32(int32_t* keys, int32_t n, int32_t multiplier, const int32_t* seeds) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int32_t key = keys[idx];
    int32_t seed_idx = key / multiplier;
    int32_t v = key % multiplier;
    if (v == seeds[seed_idx]) keys[idx] = -1;
}




__device__ __forceinline__ int bsearch_exists(const int32_t* __restrict__ arr, int size, int32_t target) {
    int lo = 0, hi = size;
    while (lo < hi) { int mid = lo + ((hi - lo) >> 1); if (arr[mid] < target) lo = mid + 1; else hi = mid; }
    return (lo < size && arr[lo] == target);
}


__global__ void overlap_fused_warp_i64(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int64_t* __restrict__ pair_keys,
    const int32_t* __restrict__ seeds,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int64_t num_pairs,
    int64_t multiplier)
{
    int64_t warp_idx = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_idx >= num_pairs) return;

    int64_t key = pair_keys[warp_idx];
    int32_t seed_idx = (int32_t)(key / multiplier);
    int32_t v = (int32_t)(key % multiplier);
    int32_t u = seeds[seed_idx];

    int u_start = offsets[u], u_end = offsets[u + 1];
    int v_start = offsets[v], v_end = offsets[v + 1];
    int deg_u = u_end - u_start, deg_v = v_end - v_start;

    if (deg_u == 0 || deg_v == 0) {
        if (lane == 0) { out_first[warp_idx] = u; out_second[warp_idx] = v; out_scores[warp_idx] = 0.0f; }
        return;
    }

    const int32_t* small = (deg_u <= deg_v) ? indices + u_start : indices + v_start;
    const int32_t* large = (deg_u <= deg_v) ? indices + v_start : indices + u_start;
    int ss = (deg_u <= deg_v) ? deg_u : deg_v;
    int sl = (deg_u <= deg_v) ? deg_v : deg_u;

    int count = 0;
    for (int i = lane; i < ss; i += 32)
        count += bsearch_exists(large, sl, small[i]);

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        count += __shfl_xor_sync(0xffffffff, count, off);

    if (lane == 0) {
        out_first[warp_idx] = u;
        out_second[warp_idx] = v;
        int md = (deg_u < deg_v) ? deg_u : deg_v;
        out_scores[warp_idx] = (float)count / (float)md;
    }
}


__global__ void overlap_fused_warp_i32(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pair_keys,
    const int32_t* __restrict__ seeds,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int64_t num_pairs,
    int32_t multiplier)
{
    int64_t warp_idx = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_idx >= num_pairs) return;

    int32_t key = pair_keys[warp_idx];
    int32_t seed_idx = key / multiplier;
    int32_t v = key % multiplier;
    int32_t u = seeds[seed_idx];

    int u_start = offsets[u], u_end = offsets[u + 1];
    int v_start = offsets[v], v_end = offsets[v + 1];
    int deg_u = u_end - u_start, deg_v = v_end - v_start;

    if (deg_u == 0 || deg_v == 0) {
        if (lane == 0) { out_first[warp_idx] = u; out_second[warp_idx] = v; out_scores[warp_idx] = 0.0f; }
        return;
    }

    const int32_t* small = (deg_u <= deg_v) ? indices + u_start : indices + v_start;
    const int32_t* large = (deg_u <= deg_v) ? indices + v_start : indices + u_start;
    int ss = (deg_u <= deg_v) ? deg_u : deg_v;
    int sl = (deg_u <= deg_v) ? deg_v : deg_u;

    int count = 0;
    for (int i = lane; i < ss; i += 32)
        count += bsearch_exists(large, sl, small[i]);

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        count += __shfl_xor_sync(0xffffffff, count, off);

    if (lane == 0) {
        out_first[warp_idx] = u;
        out_second[warp_idx] = v;
        int md = (deg_u < deg_v) ? deg_u : deg_v;
        out_scores[warp_idx] = (float)count / (float)md;
    }
}


__global__ void overlap_fused_thread_i64(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int64_t* __restrict__ pair_keys,
    const int32_t* __restrict__ seeds,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int64_t num_pairs,
    int64_t multiplier)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int64_t key = pair_keys[idx];
    int32_t seed_idx = (int32_t)(key / multiplier);
    int32_t v = (int32_t)(key % multiplier);
    int32_t u = seeds[seed_idx];

    out_first[idx] = u;
    out_second[idx] = v;

    int u_start = offsets[u], u_end = offsets[u + 1];
    int v_start = offsets[v], v_end = offsets[v + 1];
    int deg_u = u_end - u_start, deg_v = v_end - v_start;

    if (deg_u == 0 || deg_v == 0) { out_scores[idx] = 0.0f; return; }

    const int32_t* a = indices + u_start;
    const int32_t* b = indices + v_start;
    int count = 0, i = 0, j = 0;
    while (i < deg_u && j < deg_v) {
        int va = a[i], vb = b[j];
        if (va == vb) { count++; i++; j++; }
        else if (va < vb) i++; else j++;
    }
    int md = (deg_u < deg_v) ? deg_u : deg_v;
    out_scores[idx] = (md > 0) ? (float)count / (float)md : 0.0f;
}


__global__ void overlap_scores_warp_i64(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int64_t* __restrict__ pair_keys,
    const int32_t* __restrict__ seeds,
    float* __restrict__ scores,
    int64_t num_pairs,
    int64_t multiplier)
{
    int64_t warp_idx = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_idx >= num_pairs) return;

    int64_t key = pair_keys[warp_idx];
    int32_t seed_idx = (int32_t)(key / multiplier);
    int32_t v = (int32_t)(key % multiplier);
    int32_t u = seeds[seed_idx];

    int u_start = offsets[u], u_end = offsets[u + 1];
    int v_start = offsets[v], v_end = offsets[v + 1];
    int deg_u = u_end - u_start, deg_v = v_end - v_start;

    if (deg_u == 0 || deg_v == 0) { if (lane == 0) scores[warp_idx] = 0.0f; return; }

    const int32_t* small = (deg_u <= deg_v) ? indices + u_start : indices + v_start;
    const int32_t* large = (deg_u <= deg_v) ? indices + v_start : indices + u_start;
    int ss = (deg_u <= deg_v) ? deg_u : deg_v;
    int sl = (deg_u <= deg_v) ? deg_v : deg_u;

    int count = 0;
    for (int i = lane; i < ss; i += 32)
        count += bsearch_exists(large, sl, small[i]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        count += __shfl_xor_sync(0xffffffff, count, off);

    if (lane == 0) {
        int md = (deg_u < deg_v) ? deg_u : deg_v;
        scores[warp_idx] = (float)count / (float)md;
    }
}


__global__ void overlap_scores_warp_i32(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pair_keys,
    const int32_t* __restrict__ seeds,
    float* __restrict__ scores,
    int64_t num_pairs,
    int32_t multiplier)
{
    int64_t warp_idx = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_idx >= num_pairs) return;

    int32_t key = pair_keys[warp_idx];
    int32_t seed_idx = key / multiplier;
    int32_t v = key % multiplier;
    int32_t u = seeds[seed_idx];

    int u_start = offsets[u], u_end = offsets[u + 1];
    int v_start = offsets[v], v_end = offsets[v + 1];
    int deg_u = u_end - u_start, deg_v = v_end - v_start;

    if (deg_u == 0 || deg_v == 0) { if (lane == 0) scores[warp_idx] = 0.0f; return; }

    const int32_t* small = (deg_u <= deg_v) ? indices + u_start : indices + v_start;
    const int32_t* large = (deg_u <= deg_v) ? indices + v_start : indices + u_start;
    int ss = (deg_u <= deg_v) ? deg_u : deg_v;
    int sl = (deg_u <= deg_v) ? deg_v : deg_u;

    int count = 0;
    for (int i = lane; i < ss; i += 32)
        count += bsearch_exists(large, sl, small[i]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        count += __shfl_xor_sync(0xffffffff, count, off);

    if (lane == 0) {
        int md = (deg_u < deg_v) ? deg_u : deg_v;
        scores[warp_idx] = (float)count / (float)md;
    }
}


__global__ void gather_topk_i64(
    const int32_t* __restrict__ sort_idx,
    const int64_t* __restrict__ keys,
    const float* __restrict__ scores,
    const int32_t* __restrict__ seeds,
    int32_t* __restrict__ first, int32_t* __restrict__ second, float* __restrict__ out_scores,
    int64_t count, int64_t multiplier)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int32_t src = sort_idx[idx];
    int64_t key = keys[src];
    int32_t si = (int32_t)(key / multiplier);
    first[idx] = seeds[si];
    second[idx] = (int32_t)(key % multiplier);
    out_scores[idx] = scores[src];
}

__global__ void gather_topk_i32(
    const int32_t* __restrict__ sort_idx,
    const int32_t* __restrict__ keys,
    const float* __restrict__ scores,
    const int32_t* __restrict__ seeds,
    int32_t* __restrict__ first, int32_t* __restrict__ second, float* __restrict__ out_scores,
    int64_t count, int32_t multiplier)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int32_t src = sort_idx[idx];
    int32_t key = keys[src];
    first[idx] = seeds[key / multiplier];
    second[idx] = key % multiplier;
    out_scores[idx] = scores[src];
}


__global__ void overlap_scores_thread_i32(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pair_keys,
    const int32_t* __restrict__ seeds,
    float* __restrict__ scores,
    int64_t num_pairs,
    int32_t multiplier)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    int32_t key = pair_keys[idx];
    int32_t u = seeds[key / multiplier];
    int32_t v = key % multiplier;
    int us = offsets[u], ue = offsets[u+1], vs = offsets[v], ve = offsets[v+1];
    int du = ue-us, dv = ve-vs;
    if (du==0||dv==0) { scores[idx] = 0.0f; return; }
    const int32_t* a = indices+us;
    const int32_t* b = indices+vs;
    int count=0, i=0, j=0;
    while(i<du && j<dv) { int va=a[i],vb=b[j]; if(va==vb){count++;i++;j++;}else if(va<vb)i++;else j++; }
    scores[idx] = (float)count / (float)((du<dv)?du:dv);
}


__global__ void overlap_scores_thread_i64(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int64_t* __restrict__ pair_keys,
    const int32_t* __restrict__ seeds,
    float* __restrict__ scores,
    int64_t num_pairs,
    int64_t multiplier)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    int64_t key = pair_keys[idx];
    int32_t u = seeds[(int32_t)(key / multiplier)];
    int32_t v = (int32_t)(key % multiplier);
    int us = offsets[u], ue = offsets[u+1], vs = offsets[v], ve = offsets[v+1];
    int du = ue-us, dv = ve-vs;
    if (du==0||dv==0) { scores[idx] = 0.0f; return; }
    const int32_t* a = indices+us;
    const int32_t* b = indices+vs;
    int count=0, i=0, j=0;
    while(i<du && j<dv) { int va=a[i],vb=b[j]; if(va==vb){count++;i++;j++;}else if(va<vb)i++;else j++; }
    scores[idx] = (float)count / (float)((du<dv)?du:dv);
}


__global__ void overlap_fused_thread_i32(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pair_keys,
    const int32_t* __restrict__ seeds,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int64_t num_pairs,
    int32_t multiplier)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;
    int32_t key = pair_keys[idx];
    int32_t u = seeds[key / multiplier];
    int32_t v = key % multiplier;
    out_first[idx] = u; out_second[idx] = v;
    int us = offsets[u], ue = offsets[u+1], vs = offsets[v], ve = offsets[v+1];
    int du = ue-us, dv = ve-vs;
    if (du==0||dv==0) { out_scores[idx] = 0.0f; return; }
    const int32_t* a = indices+us;
    const int32_t* b = indices+vs;
    int count=0, i=0, j=0;
    while(i<du && j<dv) { int va=a[i],vb=b[j]; if(va==vb){count++;i++;j++;}else if(va<vb)i++;else j++; }
    out_scores[idx] = (float)count / (float)((du<dv)?du:dv);
}

__global__ void negate_kernel(const float* in, float* out, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = -in[idx];
}

__global__ void seq_kernel(int32_t* arr, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = (int32_t)idx;
}




static int compute_end_bit(int64_t max_val) {
    if (max_val <= 0) return 1;
    int bits = 0;
    while ((1LL << bits) <= max_val) bits++;
    return bits;
}




struct PairData {
    void* keys;        
    float* scores;
    int64_t num_pairs;
    bool use_i32;
    int64_t multiplier_i64;
    int32_t multiplier_i32;
};




struct Cache : Cacheable {
    void* scratch_ = nullptr;
    size_t scratch_size_ = 0;
    void* scratch2_ = nullptr;
    size_t scratch2_size_ = 0;
    int32_t* seeds_buf_ = nullptr;
    int32_t seeds_capacity_ = 0;

    void ensure(int32_t num_vertices) {
        if (!scratch_) {
            scratch_size_ = 768ULL * 1024 * 1024;
            cudaMalloc(&scratch_, scratch_size_);
        }
        if (!scratch2_) {
            scratch2_size_ = 256ULL * 1024 * 1024;
            cudaMalloc(&scratch2_, scratch2_size_);
        }
        if (seeds_capacity_ < num_vertices) {
            if (seeds_buf_) cudaFree(seeds_buf_);
            cudaMalloc(&seeds_buf_, (size_t)num_vertices * sizeof(int32_t));
            seeds_capacity_ = num_vertices;
        }
    }

    ~Cache() override {
        if (scratch_) cudaFree(scratch_);
        if (scratch2_) cudaFree(scratch2_);
        if (seeds_buf_) cudaFree(seeds_buf_);
    }
};




PairData overlap_compute_phase1(
    const int32_t* d_offsets,
    const int32_t* d_indices,
    int32_t num_vertices,
    const int32_t* d_seeds,
    int32_t num_seeds,
    void* scratch_buf,
    size_t scratch_size)
{
    PairData result = {};
    if (num_seeds == 0) return result;

    BumpAlloc scratch(scratch_buf, scratch_size);
    cudaStream_t stream = 0;

    
    int64_t mult64 = (int64_t)num_vertices;
    int64_t max_key = (int64_t)(num_seeds - 1) * mult64 + (int64_t)(num_vertices - 1);
    bool use_i32 = (max_key < (int64_t)INT32_MAX);
    int32_t mult32 = use_i32 ? (int32_t)num_vertices : 0;

    result.use_i32 = use_i32;
    result.multiplier_i64 = mult64;
    result.multiplier_i32 = mult32;

    
    int32_t* d_deg = scratch.alloc_array<int32_t>(num_seeds);
    gather_degrees_kernel<<<(num_seeds + 255) / 256, 256, 0, stream>>>(
        d_offsets, d_seeds, d_deg, num_seeds);

    
    int32_t* d_off = scratch.alloc_array<int32_t>(num_seeds + 1);
    cudaMemsetAsync(d_off, 0, sizeof(int32_t), stream);
    { size_t ts = 0; cub::DeviceScan::InclusiveSum(nullptr, ts, d_deg, d_off + 1, num_seeds, stream);
      void* tmp = scratch.alloc(ts); cub::DeviceScan::InclusiveSum(tmp, ts, d_deg, d_off + 1, num_seeds, stream); }

    int32_t total_nbs;
    cudaMemcpyAsync(&total_nbs, d_off + num_seeds, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (total_nbs == 0) return result;

    
    int32_t* d_pseed = scratch.alloc_array<int32_t>(total_nbs);
    int32_t* d_pnbr = scratch.alloc_array<int32_t>(total_nbs);
    int64_t* d_exp_sizes = scratch.alloc_array<int64_t>(total_nbs);
    { int nb = (num_seeds < 1024) ? num_seeds : 1024;
      fill_seed_neighbors_kernel<<<nb, 256, 0, stream>>>(d_offsets, d_indices, d_seeds, d_off, d_pseed, d_pnbr, d_exp_sizes, num_seeds); }

    
    int64_t* d_exp_off = scratch.alloc_array<int64_t>(total_nbs + 1);
    cudaMemsetAsync(d_exp_off, 0, sizeof(int64_t), stream);
    { size_t ts = 0; cub::DeviceScan::InclusiveSum(nullptr, ts, d_exp_sizes, d_exp_off + 1, total_nbs, stream);
      void* tmp = scratch.alloc(ts); cub::DeviceScan::InclusiveSum(tmp, ts, d_exp_sizes, d_exp_off + 1, total_nbs, stream); }

    int64_t total_exp;
    cudaMemcpyAsync(&total_exp, d_exp_off + total_nbs, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (total_exp == 0) return result;

    
    int end_bit;

    if (use_i32) {
        int32_t* d_keys_in = scratch.alloc_array<int32_t>(total_exp);
        { int T = 256; int64_t B = (total_exp + T - 1) / T; if (B > 2147483647LL) B = 2147483647LL;
          expand_2hop_kernel_i32<<<(int)B, T, 0, stream>>>(d_offsets, d_indices, d_pseed, d_pnbr, d_exp_off, d_keys_in, total_exp, total_nbs, mult32); }

        int32_t* d_keys_out = scratch.alloc_array<int32_t>(total_exp);
        end_bit = compute_end_bit(max_key);
        { cub::DoubleBuffer<int32_t> db(d_keys_in, d_keys_out);
          size_t ts = 0; cub::DeviceRadixSort::SortKeys(nullptr, ts, db, (int)total_exp, 0, end_bit, stream);
          void* tmp = scratch.alloc(ts); cub::DeviceRadixSort::SortKeys(tmp, ts, db, (int)total_exp, 0, end_bit, stream);
          d_keys_in = db.Current(); }

        int32_t* d_unique = scratch.alloc_array<int32_t>(total_exp);
        int32_t* d_nsel = scratch.alloc_array<int32_t>(1);
        { size_t ts = 0; cub::DeviceSelect::Unique(nullptr, ts, d_keys_in, d_unique, d_nsel, (int)total_exp, stream);
          void* tmp = scratch.alloc(ts); cub::DeviceSelect::Unique(tmp, ts, d_keys_in, d_unique, d_nsel, (int)total_exp, stream); }

        int32_t uc; cudaMemcpyAsync(&uc, d_nsel, sizeof(int32_t), cudaMemcpyDeviceToHost, stream); cudaStreamSynchronize(stream);

        mark_self_pairs_i32<<<(uc + 255) / 256, 256, 0, stream>>>(d_unique, uc, mult32, d_seeds);

        int32_t* d_final = scratch.alloc_array<int32_t>(uc);
        int32_t* d_nv = scratch.alloc_array<int32_t>(1);
        { auto pred = [] __device__ (int32_t k) { return k >= 0; };
          size_t ts = 0; cub::DeviceSelect::If(nullptr, ts, d_unique, d_final, d_nv, uc, pred, stream);
          void* tmp = scratch.alloc(ts); cub::DeviceSelect::If(tmp, ts, d_unique, d_final, d_nv, uc, pred, stream); }

        int32_t np; cudaMemcpyAsync(&np, d_nv, sizeof(int32_t), cudaMemcpyDeviceToHost, stream); cudaStreamSynchronize(stream);

        result.keys = d_final;
        result.num_pairs = np;
    } else {
        int64_t* d_keys_in = scratch.alloc_array<int64_t>(total_exp);
        { int T = 256; int64_t B = (total_exp + T - 1) / T; if (B > 2147483647LL) B = 2147483647LL;
          expand_2hop_kernel_i64<<<(int)B, T, 0, stream>>>(d_offsets, d_indices, d_pseed, d_pnbr, d_exp_off, d_keys_in, total_exp, total_nbs, mult64); }

        int64_t* d_keys_out = scratch.alloc_array<int64_t>(total_exp);
        end_bit = compute_end_bit(max_key);
        if (end_bit > 64) end_bit = 64;
        { cub::DoubleBuffer<int64_t> db(d_keys_in, d_keys_out);
          size_t ts = 0; cub::DeviceRadixSort::SortKeys(nullptr, ts, db, (int)total_exp, 0, end_bit, stream);
          void* tmp = scratch.alloc(ts); cub::DeviceRadixSort::SortKeys(tmp, ts, db, (int)total_exp, 0, end_bit, stream);
          d_keys_in = db.Current(); }

        int64_t* d_unique = scratch.alloc_array<int64_t>(total_exp);
        int32_t* d_nsel = scratch.alloc_array<int32_t>(1);
        { size_t ts = 0; cub::DeviceSelect::Unique(nullptr, ts, d_keys_in, d_unique, d_nsel, (int)total_exp, stream);
          void* tmp = scratch.alloc(ts); cub::DeviceSelect::Unique(tmp, ts, d_keys_in, d_unique, d_nsel, (int)total_exp, stream); }

        int32_t uc; cudaMemcpyAsync(&uc, d_nsel, sizeof(int32_t), cudaMemcpyDeviceToHost, stream); cudaStreamSynchronize(stream);

        mark_self_pairs_i64<<<(uc + 255) / 256, 256, 0, stream>>>(d_unique, uc, mult64, d_seeds);

        int64_t* d_final = scratch.alloc_array<int64_t>(uc);
        int32_t* d_nv = scratch.alloc_array<int32_t>(1);
        { auto pred = [] __device__ (int64_t k) { return k >= 0; };
          size_t ts = 0; cub::DeviceSelect::If(nullptr, ts, d_unique, d_final, d_nv, uc, pred, stream);
          void* tmp = scratch.alloc(ts); cub::DeviceSelect::If(tmp, ts, d_unique, d_final, d_nv, uc, pred, stream); }

        int32_t np; cudaMemcpyAsync(&np, d_nv, sizeof(int32_t), cudaMemcpyDeviceToHost, stream); cudaStreamSynchronize(stream);

        result.keys = d_final;
        result.num_pairs = np;
    }

    return result;
}


void overlap_compute_direct(
    const int32_t* d_offsets,
    const int32_t* d_indices,
    const int32_t* d_seeds,
    PairData* pd,
    bool is_multigraph,
    int32_t* out_first, int32_t* out_second, float* out_scores)
{
    if (pd->num_pairs == 0) return;
    int64_t np = pd->num_pairs;

    if (pd->use_i32) {
        if (is_multigraph) {
            int T = 256; int B = (int)((np + T - 1) / T);
            overlap_fused_thread_i32<<<B, T>>>(
                d_offsets, d_indices, (int32_t*)pd->keys, d_seeds,
                out_first, out_second, out_scores, np, pd->multiplier_i32);
        } else {
            int64_t total_threads = np * 32;
            int64_t blocks = (total_threads + 255) / 256;
            if (blocks > 2147483647LL) blocks = 2147483647LL;
            overlap_fused_warp_i32<<<(int)blocks, 256>>>(
                d_offsets, d_indices, (int32_t*)pd->keys, d_seeds,
                out_first, out_second, out_scores, np, pd->multiplier_i32);
        }
    } else {
        if (is_multigraph) {
            int T = 256; int B = (int)((np + T - 1) / T);
            overlap_fused_thread_i64<<<B, T>>>(
                d_offsets, d_indices, (int64_t*)pd->keys, d_seeds,
                out_first, out_second, out_scores, np, pd->multiplier_i64);
        } else {
            int64_t total_threads = np * 32;
            int64_t blocks = (total_threads + 255) / 256;
            if (blocks > 2147483647LL) blocks = 2147483647LL;
            overlap_fused_warp_i64<<<(int)blocks, 256>>>(
                d_offsets, d_indices, (int64_t*)pd->keys, d_seeds,
                out_first, out_second, out_scores, np, pd->multiplier_i64);
        }
    }
}


void overlap_compute_scores(
    const int32_t* d_offsets,
    const int32_t* d_indices,
    const int32_t* d_seeds,
    PairData* pd,
    bool is_multigraph,
    float* out_scores,
    void* scratch_buf,
    size_t scratch_size)
{
    if (pd->num_pairs == 0) return;
    int64_t np = pd->num_pairs;

    pd->scores = out_scores;

    if (pd->use_i32) {
        if (is_multigraph) {
            int T = 256; int B = (int)((np + T - 1) / T);
            overlap_scores_thread_i32<<<B, T>>>(
                d_offsets, d_indices, (int32_t*)pd->keys, d_seeds,
                out_scores, np, pd->multiplier_i32);
        } else {
            int64_t total_threads = np * 32;
            int64_t blocks = (total_threads + 255) / 256;
            if (blocks > 2147483647LL) blocks = 2147483647LL;
            overlap_scores_warp_i32<<<(int)blocks, 256>>>(
                d_offsets, d_indices, (int32_t*)pd->keys, d_seeds,
                out_scores, np, pd->multiplier_i32);
        }
    } else {
        if (is_multigraph) {
            int T = 256; int B = (int)((np + T - 1) / T);
            overlap_scores_thread_i64<<<B, T>>>(
                d_offsets, d_indices, (int64_t*)pd->keys, d_seeds,
                out_scores, np, pd->multiplier_i64);
        } else {
            int64_t total_threads = np * 32;
            int64_t blocks = (total_threads + 255) / 256;
            if (blocks > 2147483647LL) blocks = 2147483647LL;
            overlap_scores_warp_i64<<<(int)blocks, 256>>>(
                d_offsets, d_indices, (int64_t*)pd->keys, d_seeds,
                out_scores, np, pd->multiplier_i64);
        }
    }
}


void overlap_topk_gather(
    PairData* pd,
    const int32_t* d_seeds,
    int64_t topk,
    int32_t* out_first, int32_t* out_second, float* out_scores,
    void* scratch_buf, size_t scratch_size)
{
    if (pd->num_pairs == 0 || topk == 0) return;

    BumpAlloc scratch(scratch_buf, scratch_size);
    int64_t np = pd->num_pairs;
    int64_t k = (topk < np) ? topk : np;

    
    float* d_neg = scratch.alloc_array<float>(np);
    negate_kernel<<<(int)((np + 255) / 256), 256>>>(pd->scores, d_neg, np);

    
    int32_t* d_idx = scratch.alloc_array<int32_t>(np);
    seq_kernel<<<(int)((np + 255) / 256), 256>>>(d_idx, np);

    
    float* d_neg2 = scratch.alloc_array<float>(np);
    int32_t* d_idx2 = scratch.alloc_array<int32_t>(np);
    cub::DoubleBuffer<float> db_k(d_neg, d_neg2);
    cub::DoubleBuffer<int32_t> db_v(d_idx, d_idx2);
    size_t ts = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, ts, db_k, db_v, (int)np, 0, 32);
    void* tmp = scratch.alloc(ts);
    cub::DeviceRadixSort::SortPairs(tmp, ts, db_k, db_v, (int)np, 0, 32);

    int32_t* sorted_idx = db_v.Current();

    
    if (pd->use_i32) {
        gather_topk_i32<<<(int)((k + 255) / 256), 256>>>(
            sorted_idx, (int32_t*)pd->keys, pd->scores, d_seeds,
            out_first, out_second, out_scores, k, pd->multiplier_i32);
    } else {
        gather_topk_i64<<<(int)((k + 255) / 256), 256>>>(
            sorted_idx, (int64_t*)pd->keys, pd->scores, d_seeds,
            out_first, out_second, out_scores, k, pd->multiplier_i64);
    }
}

}  




similarity_result_float_t overlap_all_pairs_similarity(
    const graph32_t& graph,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    cache.ensure(nv);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    bool is_multigraph = graph.is_multigraph;

    
    const int32_t* d_seeds;
    int32_t num_seeds;

    if (vertices != nullptr && num_vertices > 0) {
        d_seeds = vertices;
        num_seeds = (int32_t)num_vertices;
    } else {
        num_seeds = nv;
        std::vector<int32_t> h_seeds(num_seeds);
        for (int32_t i = 0; i < num_seeds; i++) h_seeds[i] = i;
        cudaMemcpy(cache.seeds_buf_, h_seeds.data(),
                   num_seeds * sizeof(int32_t), cudaMemcpyHostToDevice);
        d_seeds = cache.seeds_buf_;
    }

    
    PairData pd = overlap_compute_phase1(
        d_offsets, d_indices, nv, d_seeds, num_seeds,
        cache.scratch_, cache.scratch_size_);

    int64_t num_pairs = pd.num_pairs;

    if (num_pairs == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    bool do_topk = topk.has_value() && (int64_t)topk.value() < num_pairs;
    int64_t output_count = do_topk ? (int64_t)topk.value() : num_pairs;

    
    int32_t* out_first;
    int32_t* out_second;
    float* out_scores;
    cudaMalloc(&out_first, output_count * sizeof(int32_t));
    cudaMalloc(&out_second, output_count * sizeof(int32_t));
    cudaMalloc(&out_scores, output_count * sizeof(float));

    if (do_topk) {
        float* scores_tmp;
        cudaMalloc(&scores_tmp, num_pairs * sizeof(float));

        overlap_compute_scores(d_offsets, d_indices, d_seeds, &pd, is_multigraph,
            scores_tmp, nullptr, 0);

        overlap_topk_gather(&pd, d_seeds, (int64_t)topk.value(),
            out_first, out_second, out_scores,
            cache.scratch2_, cache.scratch2_size_);

        cudaFree(scores_tmp);
    } else {
        overlap_compute_direct(d_offsets, d_indices, d_seeds, &pd, is_multigraph,
            out_first, out_second, out_scores);
    }

    return {out_first, out_second, out_scores, (std::size_t)output_count};
}

}  
