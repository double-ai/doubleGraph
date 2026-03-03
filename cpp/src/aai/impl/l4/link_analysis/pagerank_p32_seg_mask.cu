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
#include <cmath>
#include <algorithm>

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

namespace aai {

namespace {

static constexpr uint32_t FULL_MASK = 0xffffffffu;

struct Cache : Cacheable {
    
    int32_t* counts = nullptr;
    int32_t* new_offsets = nullptr;
    int32_t* out_degree = nullptr;
    float* inv_out = nullptr;
    float* pers_full = nullptr;
    float* pr_a = nullptr;
    float* pr_b = nullptr;
    float* x = nullptr;
    float* scalars = nullptr;
    int64_t vertex_capacity = 0;

    
    int32_t* new_indices = nullptr;
    int64_t edge_capacity = 0;

    
    void* cub_tmp = nullptr;
    size_t cub_capacity = 0;

    void ensure(int32_t n, int32_t m, size_t cub_bytes) {
        if (vertex_capacity < n) {
            if (counts) cudaFree(counts);
            if (new_offsets) cudaFree(new_offsets);
            if (out_degree) cudaFree(out_degree);
            if (inv_out) cudaFree(inv_out);
            if (pers_full) cudaFree(pers_full);
            if (pr_a) cudaFree(pr_a);
            if (pr_b) cudaFree(pr_b);
            if (x) cudaFree(x);
            if (scalars) cudaFree(scalars);

            cudaMalloc(&counts, (size_t)n * sizeof(int32_t));
            cudaMalloc(&new_offsets, (size_t)(n + 1) * sizeof(int32_t));
            cudaMalloc(&out_degree, (size_t)n * sizeof(int32_t));
            cudaMalloc(&inv_out, (size_t)n * sizeof(float));
            cudaMalloc(&pers_full, (size_t)n * sizeof(float));
            cudaMalloc(&pr_a, (size_t)n * sizeof(float));
            cudaMalloc(&pr_b, (size_t)n * sizeof(float));
            cudaMalloc(&x, (size_t)n * sizeof(float));
            cudaMalloc(&scalars, 3 * sizeof(float));
            vertex_capacity = n;
        }
        if (edge_capacity < m) {
            if (new_indices) cudaFree(new_indices);
            cudaMalloc(&new_indices, (size_t)m * sizeof(int32_t));
            edge_capacity = m;
        }
        if (cub_capacity < cub_bytes) {
            if (cub_tmp) cudaFree(cub_tmp);
            cudaMalloc(&cub_tmp, cub_bytes);
            cub_capacity = cub_bytes;
        }
    }

    ~Cache() override {
        if (counts) cudaFree(counts);
        if (new_offsets) cudaFree(new_offsets);
        if (out_degree) cudaFree(out_degree);
        if (inv_out) cudaFree(inv_out);
        if (pers_full) cudaFree(pers_full);
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (x) cudaFree(x);
        if (scalars) cudaFree(scalars);
        if (new_indices) cudaFree(new_indices);
        if (cub_tmp) cudaFree(cub_tmp);
    }
};



__device__ __forceinline__ bool is_edge_active(const uint32_t* __restrict__ mask, int32_t edge_idx)
{
  uint32_t word = __ldg(mask + (edge_idx >> 5));
  return (word >> (edge_idx & 31)) & 1u;
}


__global__ void set_i32_kernel(int32_t* ptr, int32_t val) { *ptr = val; }
__global__ void set_f32_kernel(float* ptr, float val) { *ptr = val; }

__global__ void zero_i32_kernel(int32_t* __restrict__ arr, int32_t n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) arr[i] = 0;
}
__global__ void zero_f32_kernel(float* __restrict__ arr, int32_t n) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) arr[i] = 0.0f;
}




__global__ void count_active_high_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ mask,
    int32_t* __restrict__ counts,
    int32_t start_v, int32_t end_v)
{
  int32_t v = start_v + (int32_t)blockIdx.x;
  if (v >= end_v) return;

  int32_t start = offsets[v];
  int32_t end = offsets[v + 1];

  __shared__ int warp_counts[8];
  __shared__ int warp_offsets[8];

  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;

  int32_t total = 0;
  
  for (int32_t base = start; base < end; base += 256) {
    int32_t e = base + (int32_t)threadIdx.x;
    bool pred = (e < end) && is_edge_active(mask, e);
    unsigned int ballot = __ballot_sync(FULL_MASK, pred);
    int wc = __popc(ballot);
    if (lane == 0) warp_counts[warp] = wc;
    __syncthreads();
    if (threadIdx.x < 8) {
      int sum = 0;
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        warp_offsets[i] = sum;
        sum += warp_counts[i];
      }
      
      warp_counts[0] = sum;
    }
    __syncthreads();
    total += warp_counts[0];
  }

  if (threadIdx.x == 0) counts[v] = total;
}


__global__ void count_active_mid_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ mask,
    int32_t* __restrict__ counts,
    int32_t start_v, int32_t end_v)
{
  int global_tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int warp_id = global_tid >> 5;
  int lane = global_tid & 31;

  int32_t v = start_v + (int32_t)warp_id;
  if (v >= end_v) return;

  int32_t start = offsets[v];
  int32_t end = offsets[v + 1];

  int32_t total = 0;
  
  for (int32_t j = start + lane; j < end; j += 32) {
    bool pred = is_edge_active(mask, j);
    unsigned int ballot = __ballot_sync(FULL_MASK, pred);
    if (lane == 0) total += __popc(ballot);
  }
  if (lane == 0) counts[v] = total;
}


__global__ void count_active_low_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ mask,
    int32_t* __restrict__ counts,
    int32_t start_v, int32_t end_v)
{
  int32_t v = start_v + (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (v >= end_v) return;

  int32_t s = offsets[v];
  int32_t e = offsets[v + 1];
  int32_t c = 0;
  for (int32_t j = s; j < e; ++j) c += (int32_t)is_edge_active(mask, j);
  counts[v] = c;
}




__global__ void compact_high_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t start_v, int32_t end_v)
{
  int32_t v = start_v + (int32_t)blockIdx.x;
  if (v >= end_v) return;

  int32_t start = old_offsets[v];
  int32_t end = old_offsets[v + 1];
  int32_t out_base = new_offsets[v];

  __shared__ int warp_counts[8];
  __shared__ int warp_offsets[8];
  __shared__ int tile_out;

  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;

  if (threadIdx.x == 0) tile_out = 0;
  __syncthreads();

  for (int32_t base = start; base < end; base += 256) {
    int32_t e = base + (int32_t)threadIdx.x;
    bool in_range = (e < end);
    bool pred = in_range && is_edge_active(mask, e);

    unsigned int ballot = __ballot_sync(FULL_MASK, pred);
    int wc = __popc(ballot);
    int rank = __popc(ballot & ((1u << lane) - 1u));

    if (lane == 0) warp_counts[warp] = wc;
    __syncthreads();

    int chunk_total = 0;
    if (threadIdx.x < 8) {
      int sum = 0;
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        warp_offsets[i] = sum;
        sum += warp_counts[i];
      }
      chunk_total = sum;
      warp_counts[0] = sum;
    }
    __syncthreads();
    chunk_total = warp_counts[0];

    int my_tile_out = tile_out;

    if (pred) {
      int out_pos = out_base + my_tile_out + warp_offsets[warp] + rank;
      new_indices[out_pos] = old_indices[e];
    }

    __syncthreads();
    if (threadIdx.x == 0) tile_out += chunk_total;
    __syncthreads();
  }
}


__global__ void compact_mid_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t start_v, int32_t end_v)
{
  int global_tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int warp_id = global_tid >> 5;
  int lane = global_tid & 31;
  int32_t v = start_v + (int32_t)warp_id;
  if (v >= end_v) return;

  int32_t start = old_offsets[v];
  int32_t end = old_offsets[v + 1];
  int32_t out_base = new_offsets[v];

  int32_t out_off = 0;
  for (int32_t j = start + lane; j < end; j += 32) {
    bool pred = is_edge_active(mask, j);
    unsigned int ballot = __ballot_sync(FULL_MASK, pred);
    int count = __popc(ballot);
    int rank = __popc(ballot & ((1u << lane) - 1u));
    int base = __shfl_sync(FULL_MASK, out_off, 0);
    if (pred) {
      new_indices[out_base + base + rank] = old_indices[j];
    }
    if (lane == 0) out_off += count;
  }
}


__global__ void compact_low_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t start_v, int32_t end_v)
{
  int32_t v = start_v + (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (v >= end_v) return;

  int32_t s = old_offsets[v];
  int32_t e = old_offsets[v + 1];
  int32_t out = new_offsets[v];
  for (int32_t j = s; j < e; ++j) {
    if (is_edge_active(mask, j)) new_indices[out++] = old_indices[j];
  }
}



__global__ void out_degree_hist_kernel(
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ total_edges_ptr,
    int32_t num_edges_alloc,
    int32_t* __restrict__ out_degree)
{
  int32_t total = *total_edges_ptr;
  int32_t tid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid < num_edges_alloc && tid < total) {
    int32_t src = indices[tid];
    atomicAdd(&out_degree[src], 1);
  }
}

__global__ void inv_out_from_degree_kernel(const int32_t* __restrict__ out_degree, float* __restrict__ inv_out, int32_t n)
{
  int32_t v = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (v < n) {
    int32_t d = out_degree[v];
    inv_out[v] = (d > 0) ? (1.0f / (float)d) : 0.0f;
  }
}

__global__ void inv_out_from_precomputed_kernel(const float* __restrict__ out_w, float* __restrict__ inv_out, int32_t n)
{
  int32_t v = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (v < n) {
    float d = out_w[v];
    inv_out[v] = (d > 0.0f) ? (1.0f / d) : 0.0f;
  }
}



__global__ void scatter_pers_norm_kernel(
    const int32_t* __restrict__ pers_verts,
    const float* __restrict__ pers_vals,
    const float* __restrict__ pers_sum,
    float* __restrict__ pers_full,
    int32_t pers_size)
{
  int32_t i = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < pers_size) {
    float s = *pers_sum;
    float inv = (s > 0.0f) ? (1.0f / s) : 0.0f;
    pers_full[pers_verts[i]] = pers_vals[i] * inv;
  }
}

__global__ void init_pr_uniform_kernel(float* __restrict__ pr, float val, int32_t n) {
  int32_t i = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) pr[i] = val;
}



__global__ void compute_x_and_dangling_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ inv_out,
    float* __restrict__ x,
    float* __restrict__ dangling_sum,
    int32_t n)
{
  constexpr int WARPS_PER_BLOCK = 8;
  __shared__ float warp_sums[WARPS_PER_BLOCK];

  int32_t v = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  float dang = 0.0f;
  if (v < n) {
    float p = pr[v];
    float inv = inv_out[v];
    x[v] = p * inv;
    if (inv == 0.0f) dang = p;
  }

  int lane = threadIdx.x & 31;
  int warp_in_block = threadIdx.x >> 5;

  
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    dang += __shfl_down_sync(FULL_MASK, dang, off);
  }
  if (lane == 0) warp_sums[warp_in_block] = dang;
  __syncthreads();

  if (threadIdx.x == 0) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < WARPS_PER_BLOCK; ++i) sum += warp_sums[i];
    if (sum != 0.0f) atomicAdd(dangling_sum, sum);
  }
}



__device__ __forceinline__ float warp_reduce_sum(float v) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(FULL_MASK, v, off);
  return v;
}


__global__ void spmv_update_high_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    const float* __restrict__ old_pr,
    float* __restrict__ new_pr,
    const float* __restrict__ pers_full,
    const float* __restrict__ dangling_sum,
    float* __restrict__ diff_sum,
    float alpha,
    float one_minus_alpha,
    int32_t start_v,
    int32_t end_v)
{
  using BR = cub::BlockReduce<float, 256>;
  __shared__ typename BR::TempStorage temp;
  __shared__ float s_dang_factor;

  int32_t v = start_v + (int32_t)blockIdx.x;
  if (v >= end_v) return;

  if (threadIdx.x == 0) {
    s_dang_factor = alpha * (*dangling_sum) + one_minus_alpha;
  }
  __syncthreads();

  int32_t s = offsets[v];
  int32_t e = offsets[v + 1];
  float sum = 0.0f;
  for (int32_t j = s + (int32_t)threadIdx.x; j < e; j += (int32_t)blockDim.x) {
    sum += x[indices[j]];
  }
  float bsum = BR(temp).Sum(sum);
  if (threadIdx.x == 0) {
    float val = alpha * bsum + s_dang_factor * pers_full[v];
    new_pr[v] = val;
    float diff = fabsf(val - old_pr[v]);
    if (diff != 0.0f) atomicAdd(diff_sum, diff);
  }
}


__global__ void spmv_update_mid_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    const float* __restrict__ old_pr,
    float* __restrict__ new_pr,
    const float* __restrict__ pers_full,
    const float* __restrict__ dangling_sum,
    float* __restrict__ diff_sum,
    float alpha,
    float one_minus_alpha,
    int32_t start_v,
    int32_t end_v)
{
  constexpr int WARPS_PER_BLOCK = 8;
  __shared__ float warp_diffs[WARPS_PER_BLOCK];

  int gtid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int warp_id = gtid >> 5;
  int lane = gtid & 31;
  int warp_in_block = threadIdx.x >> 5;
  int32_t v = start_v + (int32_t)warp_id;

  float my_diff = 0.0f;
  float dang_factor = 0.0f;
  if (v < end_v && lane == 0) dang_factor = alpha * (*dangling_sum) + one_minus_alpha;
  dang_factor = __shfl_sync(FULL_MASK, dang_factor, 0);

  if (v < end_v) {
    int32_t s = offsets[v];
    int32_t e = offsets[v + 1];
    float sum = 0.0f;
    for (int32_t j = s + lane; j < e; j += 32) sum += x[indices[j]];
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
      float val = alpha * sum + dang_factor * pers_full[v];
      new_pr[v] = val;
      my_diff = fabsf(val - old_pr[v]);
    }
  }

  if (lane == 0) {
    warp_diffs[warp_in_block] = (v < end_v) ? my_diff : 0.0f;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float sum_diff = 0.0f;
    #pragma unroll
    for (int i = 0; i < WARPS_PER_BLOCK; ++i) sum_diff += warp_diffs[i];
    if (sum_diff != 0.0f) atomicAdd(diff_sum, sum_diff);
  }
}


__global__ void spmv_update_low_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    const float* __restrict__ old_pr,
    float* __restrict__ new_pr,
    const float* __restrict__ pers_full,
    const float* __restrict__ dangling_sum,
    float* __restrict__ diff_sum,
    float alpha,
    float one_minus_alpha,
    int32_t start_v,
    int32_t end_v)
{
  constexpr int WARPS_PER_BLOCK = 8;
  __shared__ float warp_diffs[WARPS_PER_BLOCK];
  __shared__ float s_dang_factor;

  if (threadIdx.x == 0) s_dang_factor = alpha * (*dangling_sum) + one_minus_alpha;
  __syncthreads();

  int32_t v = start_v + (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  float my_diff = 0.0f;
  if (v < end_v) {
    int32_t s = offsets[v];
    int32_t e = offsets[v + 1];
    float sum = 0.0f;
    for (int32_t j = s; j < e; ++j) sum += x[indices[j]];
    float val = alpha * sum + s_dang_factor * pers_full[v];
    new_pr[v] = val;
    my_diff = fabsf(val - old_pr[v]);
  }

  int lane = threadIdx.x & 31;
  int warp_in_block = threadIdx.x >> 5;
  float wsum = warp_reduce_sum(my_diff);
  if (lane == 0) warp_diffs[warp_in_block] = wsum;
  __syncthreads();

  if (threadIdx.x == 0) {
    float sum_diff = 0.0f;
    #pragma unroll
    for (int i = 0; i < WARPS_PER_BLOCK; ++i) sum_diff += warp_diffs[i];
    if (sum_diff != 0.0f) atomicAdd(diff_sum, sum_diff);
  }
}



__global__ void spmv_nodiff_high_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ new_pr,
    const float* __restrict__ pers_full,
    const float* __restrict__ dangling_sum,
    float alpha,
    float one_minus_alpha,
    int32_t start_v,
    int32_t end_v)
{
  using BR = cub::BlockReduce<float, 256>;
  __shared__ typename BR::TempStorage temp;
  __shared__ float s_dang_factor;

  int32_t v = start_v + (int32_t)blockIdx.x;
  if (v >= end_v) return;

  if (threadIdx.x == 0) {
    s_dang_factor = alpha * (*dangling_sum) + one_minus_alpha;
  }
  __syncthreads();

  int32_t s = offsets[v];
  int32_t e = offsets[v + 1];
  float sum = 0.0f;
  for (int32_t j = s + (int32_t)threadIdx.x; j < e; j += (int32_t)blockDim.x) {
    sum += x[indices[j]];
  }
  float bsum = BR(temp).Sum(sum);
  if (threadIdx.x == 0) {
    new_pr[v] = alpha * bsum + s_dang_factor * pers_full[v];
  }
}

__global__ void spmv_nodiff_mid_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ new_pr,
    const float* __restrict__ pers_full,
    const float* __restrict__ dangling_sum,
    float alpha,
    float one_minus_alpha,
    int32_t start_v,
    int32_t end_v)
{
  int gtid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int warp_id = gtid >> 5;
  int lane = gtid & 31;
  int32_t v = start_v + (int32_t)warp_id;

  float dang_factor = 0.0f;
  if (v < end_v && lane == 0) dang_factor = alpha * (*dangling_sum) + one_minus_alpha;
  dang_factor = __shfl_sync(FULL_MASK, dang_factor, 0);

  if (v < end_v) {
    int32_t s = offsets[v];
    int32_t e = offsets[v + 1];
    float sum = 0.0f;
    for (int32_t j = s + lane; j < e; j += 32) sum += x[indices[j]];
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
      new_pr[v] = alpha * sum + dang_factor * pers_full[v];
    }
  }
}

__global__ void spmv_nodiff_low_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ new_pr,
    const float* __restrict__ pers_full,
    const float* __restrict__ dangling_sum,
    float alpha,
    float one_minus_alpha,
    int32_t start_v,
    int32_t end_v)
{
  __shared__ float s_dang_factor;
  if (threadIdx.x == 0) s_dang_factor = alpha * (*dangling_sum) + one_minus_alpha;
  __syncthreads();

  int32_t v = start_v + (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (v < end_v) {
    int32_t s = offsets[v];
    int32_t e = offsets[v + 1];
    float sum = 0.0f;
    for (int32_t j = s; j < e; ++j) sum += x[indices[j]];
    new_pr[v] = alpha * sum + s_dang_factor * pers_full[v];
  }
}




__global__ void compact_high_outdeg_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t* __restrict__ out_degree,
    int32_t start_v, int32_t end_v)
{
  int32_t v = start_v + (int32_t)blockIdx.x;
  if (v >= end_v) return;

  int32_t start = old_offsets[v];
  int32_t end = old_offsets[v + 1];
  int32_t out_base = new_offsets[v];

  __shared__ int warp_counts[8];
  __shared__ int warp_offsets[8];
  __shared__ int tile_out;

  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;

  if (threadIdx.x == 0) tile_out = 0;
  __syncthreads();

  for (int32_t base = start; base < end; base += 256) {
    int32_t e = base + (int32_t)threadIdx.x;
    bool in_range = (e < end);
    bool pred = in_range && is_edge_active(mask, e);

    unsigned int ballot = __ballot_sync(FULL_MASK, pred);
    int wc = __popc(ballot);
    int rank = __popc(ballot & ((1u << lane) - 1u));

    if (lane == 0) warp_counts[warp] = wc;
    __syncthreads();

    int chunk_total = 0;
    if (threadIdx.x < 8) {
      int sum = 0;
      #pragma unroll
      for (int i = 0; i < 8; i++) {
        warp_offsets[i] = sum;
        sum += warp_counts[i];
      }
      chunk_total = sum;
      warp_counts[0] = sum;
    }
    __syncthreads();
    chunk_total = warp_counts[0];

    int my_tile_out = tile_out;

    if (pred) {
      int32_t src = old_indices[e];
      int out_pos = out_base + my_tile_out + warp_offsets[warp] + rank;
      new_indices[out_pos] = src;
      atomicAdd(&out_degree[src], 1);
    }

    __syncthreads();
    if (threadIdx.x == 0) tile_out += chunk_total;
    __syncthreads();
  }
}


__global__ void compact_mid_outdeg_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t* __restrict__ out_degree,
    int32_t start_v, int32_t end_v)
{
  int global_tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int warp_id = global_tid >> 5;
  int lane = global_tid & 31;
  int32_t v = start_v + (int32_t)warp_id;
  if (v >= end_v) return;

  int32_t start = old_offsets[v];
  int32_t end = old_offsets[v + 1];
  int32_t out_base = new_offsets[v];

  int32_t out_off = 0;
  for (int32_t j = start + lane; j < end; j += 32) {
    bool pred = is_edge_active(mask, j);
    unsigned int ballot = __ballot_sync(FULL_MASK, pred);
    int count = __popc(ballot);
    int rank = __popc(ballot & ((1u << lane) - 1u));
    int base = __shfl_sync(FULL_MASK, out_off, 0);
    if (pred) {
      int32_t src = old_indices[j];
      new_indices[out_base + base + rank] = src;
      atomicAdd(&out_degree[src], 1);
    }
    if (lane == 0) out_off += count;
  }
}


__global__ void compact_low_outdeg_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t* __restrict__ out_degree,
    int32_t start_v, int32_t end_v)
{
  int32_t v = start_v + (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (v >= end_v) return;

  int32_t s = old_offsets[v];
  int32_t e = old_offsets[v + 1];
  int32_t out = new_offsets[v];
  for (int32_t j = s; j < e; ++j) {
    if (is_edge_active(mask, j)) {
      int32_t src = old_indices[j];
      new_indices[out++] = src;
      atomicAdd(&out_degree[src], 1);
    }
  }
}



void launch_set_i32(int32_t* ptr, int32_t val, cudaStream_t stream) { set_i32_kernel<<<1, 1, 0, stream>>>(ptr, val); }
void launch_set_f32(float* ptr, float val, cudaStream_t stream) { set_f32_kernel<<<1, 1, 0, stream>>>(ptr, val); }

void launch_zero_i32(int32_t* arr, int32_t n, cudaStream_t stream) {
  int grid = (n + 255) / 256;
  zero_i32_kernel<<<grid, 256, 0, stream>>>(arr, n);
}
void launch_zero_f32(float* arr, int32_t n, cudaStream_t stream) {
  int grid = (n + 255) / 256;
  zero_f32_kernel<<<grid, 256, 0, stream>>>(arr, n);
}

void launch_count_active_edges_high(const int32_t* offsets, const uint32_t* edge_mask, int32_t* counts,
                                    int32_t start_v, int32_t end_v, cudaStream_t stream) {
  int nseg = end_v - start_v;
  if (nseg <= 0) return;
  count_active_high_kernel<<<nseg, 256, 0, stream>>>(offsets, edge_mask, counts, start_v, end_v);
}

void launch_count_active_edges_mid(const int32_t* offsets, const uint32_t* edge_mask, int32_t* counts,
                                   int32_t start_v, int32_t end_v, cudaStream_t stream) {
  int nseg = end_v - start_v;
  if (nseg <= 0) return;
  constexpr int WARPS_PER_BLOCK = 8;
  int grid = (nseg + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  count_active_mid_kernel<<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(offsets, edge_mask, counts, start_v, end_v);
}

void launch_count_active_edges_low(const int32_t* offsets, const uint32_t* edge_mask, int32_t* counts,
                                   int32_t start_v, int32_t end_v, cudaStream_t stream) {
  int nseg = end_v - start_v;
  if (nseg <= 0) return;
  int grid = (nseg + 255) / 256;
  count_active_low_kernel<<<grid, 256, 0, stream>>>(offsets, edge_mask, counts, start_v, end_v);
}

void launch_compact_high(const int32_t* old_offsets, const int32_t* old_indices, const uint32_t* edge_mask,
                          const int32_t* new_offsets, int32_t* new_indices,
                          int32_t start_v, int32_t end_v, cudaStream_t stream) {
  int nseg = end_v - start_v;
  if (nseg <= 0) return;
  compact_high_kernel<<<nseg, 256, 0, stream>>>(old_offsets, old_indices, edge_mask, new_offsets, new_indices, start_v, end_v);
}

void launch_compact_mid(const int32_t* old_offsets, const int32_t* old_indices, const uint32_t* edge_mask,
                         const int32_t* new_offsets, int32_t* new_indices,
                         int32_t start_v, int32_t end_v, cudaStream_t stream) {
  int nseg = end_v - start_v;
  if (nseg <= 0) return;
  constexpr int WARPS_PER_BLOCK = 8;
  int grid = (nseg + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  compact_mid_kernel<<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(old_offsets, old_indices, edge_mask, new_offsets, new_indices, start_v, end_v);
}

void launch_compact_low(const int32_t* old_offsets, const int32_t* old_indices, const uint32_t* edge_mask,
                         const int32_t* new_offsets, int32_t* new_indices,
                         int32_t start_v, int32_t end_v, cudaStream_t stream) {
  int nseg = end_v - start_v;
  if (nseg <= 0) return;
  int grid = (nseg + 255) / 256;
  compact_low_kernel<<<grid, 256, 0, stream>>>(old_offsets, old_indices, edge_mask, new_offsets, new_indices, start_v, end_v);
}

void launch_compact_high_outdeg(const int32_t* old_offsets, const int32_t* old_indices, const uint32_t* edge_mask,
                                 const int32_t* new_offsets, int32_t* new_indices,
                                 int32_t* out_degree,
                                 int32_t start_v, int32_t end_v, cudaStream_t stream) {
  int nseg = end_v - start_v;
  if (nseg <= 0) return;
  compact_high_outdeg_kernel<<<nseg, 256, 0, stream>>>(old_offsets, old_indices, edge_mask, new_offsets, new_indices, out_degree, start_v, end_v);
}

void launch_compact_mid_outdeg(const int32_t* old_offsets, const int32_t* old_indices, const uint32_t* edge_mask,
                                const int32_t* new_offsets, int32_t* new_indices,
                                int32_t* out_degree,
                                int32_t start_v, int32_t end_v, cudaStream_t stream) {
  int nseg = end_v - start_v;
  if (nseg <= 0) return;
  constexpr int WARPS_PER_BLOCK = 8;
  int grid = (nseg + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  compact_mid_outdeg_kernel<<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(old_offsets, old_indices, edge_mask, new_offsets, new_indices, out_degree, start_v, end_v);
}

void launch_compact_low_outdeg(const int32_t* old_offsets, const int32_t* old_indices, const uint32_t* edge_mask,
                                const int32_t* new_offsets, int32_t* new_indices,
                                int32_t* out_degree,
                                int32_t start_v, int32_t end_v, cudaStream_t stream) {
  int nseg = end_v - start_v;
  if (nseg <= 0) return;
  int grid = (nseg + 255) / 256;
  compact_low_outdeg_kernel<<<grid, 256, 0, stream>>>(old_offsets, old_indices, edge_mask, new_offsets, new_indices, out_degree, start_v, end_v);
}

void launch_inv_out_from_degree(const int32_t* out_degree, float* inv_out, int32_t n, cudaStream_t stream) {
  int grid = (n + 255) / 256;
  inv_out_from_degree_kernel<<<grid, 256, 0, stream>>>(out_degree, inv_out, n);
}

void launch_inv_out_from_precomputed(const float* out_w, float* inv_out, int32_t n, cudaStream_t stream) {
  int grid = (n + 255) / 256;
  inv_out_from_precomputed_kernel<<<grid, 256, 0, stream>>>(out_w, inv_out, n);
}

void launch_scatter_pers_norm(const int32_t* pers_verts, const float* pers_vals, const float* pers_sum,
                               float* pers_full, int32_t pers_size, cudaStream_t stream) {
  int grid = (pers_size + 255) / 256;
  scatter_pers_norm_kernel<<<grid, 256, 0, stream>>>(pers_verts, pers_vals, pers_sum, pers_full, pers_size);
}

void launch_init_pr_uniform(float* pr, float val, int32_t n, cudaStream_t stream) {
  int grid = (n + 255) / 256;
  init_pr_uniform_kernel<<<grid, 256, 0, stream>>>(pr, val, n);
}

void launch_compute_x_and_dangling(const float* pr, const float* inv_out, float* x, float* dangling_sum,
                                    int32_t n, cudaStream_t stream) {
  int grid = (n + 255) / 256;
  compute_x_and_dangling_kernel<<<grid, 256, 0, stream>>>(pr, inv_out, x, dangling_sum, n);
}

void launch_spmv_update_high(const int32_t* offsets, const int32_t* indices, const float* x,
                              const float* old_pr, float* new_pr,
                              const float* pers_full, const float* dangling_sum, float* diff_sum,
                              float alpha, float one_minus_alpha,
                              int32_t start_v, int32_t end_v, cudaStream_t stream) {
  int nseg = end_v - start_v;
  if (nseg <= 0) return;
  spmv_update_high_kernel<<<nseg, 256, 0, stream>>>(offsets, indices, x, old_pr, new_pr, pers_full, dangling_sum, diff_sum,
                                                    alpha, one_minus_alpha, start_v, end_v);
}

void launch_spmv_update_mid(const int32_t* offsets, const int32_t* indices, const float* x,
                             const float* old_pr, float* new_pr,
                             const float* pers_full, const float* dangling_sum, float* diff_sum,
                             float alpha, float one_minus_alpha,
                             int32_t start_v, int32_t end_v, cudaStream_t stream) {
  int nseg = end_v - start_v;
  if (nseg <= 0) return;
  constexpr int WARPS_PER_BLOCK = 8;
  int grid = (nseg + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  spmv_update_mid_kernel<<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(offsets, indices, x, old_pr, new_pr, pers_full, dangling_sum, diff_sum,
                                                                     alpha, one_minus_alpha, start_v, end_v);
}

void launch_spmv_update_low(const int32_t* offsets, const int32_t* indices, const float* x,
                             const float* old_pr, float* new_pr,
                             const float* pers_full, const float* dangling_sum, float* diff_sum,
                             float alpha, float one_minus_alpha,
                             int32_t start_v, int32_t end_v, cudaStream_t stream) {
  int nseg = end_v - start_v;
  if (nseg <= 0) return;
  int grid = (nseg + 255) / 256;
  spmv_update_low_kernel<<<grid, 256, 0, stream>>>(offsets, indices, x, old_pr, new_pr, pers_full, dangling_sum, diff_sum,
                                                   alpha, one_minus_alpha, start_v, end_v);
}

void launch_spmv_nodiff_high(const int32_t* offsets, const int32_t* indices, const float* x,
                              float* new_pr,
                              const float* pers_full, const float* dangling_sum,
                              float alpha, float one_minus_alpha,
                              int32_t start_v, int32_t end_v, cudaStream_t stream) {
  int nseg = end_v - start_v;
  if (nseg <= 0) return;
  spmv_nodiff_high_kernel<<<nseg, 256, 0, stream>>>(offsets, indices, x, new_pr, pers_full, dangling_sum,
                                                    alpha, one_minus_alpha, start_v, end_v);
}

void launch_spmv_nodiff_mid(const int32_t* offsets, const int32_t* indices, const float* x,
                             float* new_pr,
                             const float* pers_full, const float* dangling_sum,
                             float alpha, float one_minus_alpha,
                             int32_t start_v, int32_t end_v, cudaStream_t stream) {
  int nseg = end_v - start_v;
  if (nseg <= 0) return;
  constexpr int WARPS_PER_BLOCK = 8;
  int grid = (nseg + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  spmv_nodiff_mid_kernel<<<grid, WARPS_PER_BLOCK * 32, 0, stream>>>(offsets, indices, x, new_pr, pers_full, dangling_sum,
                                                                     alpha, one_minus_alpha, start_v, end_v);
}

void launch_spmv_nodiff_low(const int32_t* offsets, const int32_t* indices, const float* x,
                             float* new_pr,
                             const float* pers_full, const float* dangling_sum,
                             float alpha, float one_minus_alpha,
                             int32_t start_v, int32_t end_v, cudaStream_t stream) {
  int nseg = end_v - start_v;
  if (nseg <= 0) return;
  int grid = (nseg + 255) / 256;
  spmv_nodiff_low_kernel<<<grid, 256, 0, stream>>>(offsets, indices, x, new_pr, pers_full, dangling_sum,
                                                   alpha, one_minus_alpha, start_v, end_v);
}

}  

PageRankResult personalized_pagerank_seg_mask(const graph32_t& graph,
                                              const int32_t* personalization_vertices,
                                              const float* personalization_values,
                                              std::size_t personalization_size,
                                              float* pageranks,
                                              const float* precomputed_vertex_out_weight_sums,
                                              float alpha,
                                              float epsilon,
                                              std::size_t max_iterations,
                                              const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n = graph.number_of_vertices;
    int32_t m = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;

    const auto& seg_vec = graph.segment_offsets.value();
    int32_t seg[5] = {seg_vec[0], seg_vec[1], seg_vec[2], seg_vec[3], seg_vec[4]};

    int32_t pers_size = static_cast<int32_t>(personalization_size);
    float one_minus_alpha = 1.0f - alpha;
    bool has_initial_guess = (initial_pageranks != nullptr);

    cudaStream_t stream = 0;

    
    size_t scan_tmp = 0;
    cub::DeviceScan::InclusiveSum(nullptr, scan_tmp, (const int32_t*)nullptr, (int32_t*)nullptr, n);
    size_t reduce_tmp = 0;
    cub::DeviceReduce::Sum(nullptr, reduce_tmp, (const float*)nullptr, (float*)nullptr, (int)pers_size);
    size_t cub_tmp_bytes = std::max(scan_tmp, reduce_tmp);

    cache.ensure(n, m, cub_tmp_bytes);

    int32_t* d_counts = cache.counts;
    int32_t* d_new_offsets = cache.new_offsets;
    int32_t* d_new_indices = cache.new_indices;
    int32_t* d_out_degree = cache.out_degree;
    float* d_inv_out = cache.inv_out;
    float* d_pers_full = cache.pers_full;
    float* d_pr_a = cache.pr_a;
    float* d_pr_b = cache.pr_b;
    float* d_x = cache.x;
    float* d_scalars = cache.scalars;
    float* d_dangling_sum = d_scalars;
    float* d_diff_sum = d_scalars + 1;
    float* d_pers_sum = d_scalars + 2;
    void* d_cub_tmp = cache.cub_tmp;

    

    
    launch_count_active_edges_high(d_offsets, d_edge_mask, d_counts, seg[0], seg[1], stream);
    launch_count_active_edges_mid(d_offsets, d_edge_mask, d_counts, seg[1], seg[2], stream);
    launch_count_active_edges_low(d_offsets, d_edge_mask, d_counts, seg[2], seg[4], stream);

    
    launch_set_i32(d_new_offsets, 0, stream);
    cub::DeviceScan::InclusiveSum(d_cub_tmp, scan_tmp, d_counts, d_new_offsets + 1, n, stream);

    

    if (precomputed_vertex_out_weight_sums != nullptr) {
      
      launch_compact_high(d_offsets, d_indices, d_edge_mask, d_new_offsets, d_new_indices, seg[0], seg[1], stream);
      launch_compact_mid(d_offsets, d_indices, d_edge_mask, d_new_offsets, d_new_indices, seg[1], seg[2], stream);
      launch_compact_low(d_offsets, d_indices, d_edge_mask, d_new_offsets, d_new_indices, seg[2], seg[4], stream);

      launch_inv_out_from_precomputed(precomputed_vertex_out_weight_sums, d_inv_out, n, stream);
    } else {
      
      launch_zero_i32(d_out_degree, n, stream);

      launch_compact_high_outdeg(d_offsets, d_indices, d_edge_mask, d_new_offsets, d_new_indices, d_out_degree, seg[0], seg[1], stream);
      launch_compact_mid_outdeg(d_offsets, d_indices, d_edge_mask, d_new_offsets, d_new_indices, d_out_degree, seg[1], seg[2], stream);
      launch_compact_low_outdeg(d_offsets, d_indices, d_edge_mask, d_new_offsets, d_new_indices, d_out_degree, seg[2], seg[4], stream);

      launch_inv_out_from_degree(d_out_degree, d_inv_out, n, stream);
    }

    
    launch_zero_f32(d_pers_full, n, stream);
    
    if (pers_size > 0) {
      cub::DeviceReduce::Sum(d_cub_tmp, reduce_tmp, personalization_values, d_pers_sum, (int)pers_size, stream);
      launch_scatter_pers_norm(personalization_vertices, personalization_values, d_pers_sum, d_pers_full, pers_size, stream);
    }

    
    if (has_initial_guess) {
      cudaMemcpyAsync(d_pr_a, initial_pageranks, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
      launch_init_pr_uniform(d_pr_a, 1.0f / (float)n, n, stream);
    }

    
    float* pr_old = d_pr_a;
    float* pr_new = d_pr_b;

    bool converged = false;
    size_t iters_done = 0;

    constexpr int CHECK_INTERVAL = 4;

    for (size_t iter = 0; iter < max_iterations; ++iter) {
      bool do_check = ((iter % CHECK_INTERVAL) == (CHECK_INTERVAL - 1)) || (iter + 1 == max_iterations);

      if (do_check) {
        
        cudaMemsetAsync(d_scalars, 0, 2 * sizeof(float), stream);
      } else {
        
        cudaMemsetAsync(d_dangling_sum, 0, sizeof(float), stream);
      }

      launch_compute_x_and_dangling(pr_old, d_inv_out, d_x, d_dangling_sum, n, stream);

      if (do_check) {
        launch_spmv_update_high(d_new_offsets, d_new_indices, d_x, pr_old, pr_new, d_pers_full, d_dangling_sum, d_diff_sum,
                                alpha, one_minus_alpha, seg[0], seg[1], stream);
        launch_spmv_update_mid(d_new_offsets, d_new_indices, d_x, pr_old, pr_new, d_pers_full, d_dangling_sum, d_diff_sum,
                               alpha, one_minus_alpha, seg[1], seg[2], stream);
        launch_spmv_update_low(d_new_offsets, d_new_indices, d_x, pr_old, pr_new, d_pers_full, d_dangling_sum, d_diff_sum,
                               alpha, one_minus_alpha, seg[2], seg[4], stream);

        iters_done = iter + 1;

        float h_diff = 0.0f;
        cudaMemcpyAsync(&h_diff, d_diff_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (h_diff < epsilon) {
          converged = true;
          break;
        }
      } else {
        launch_spmv_nodiff_high(d_new_offsets, d_new_indices, d_x, pr_new, d_pers_full, d_dangling_sum,
                                alpha, one_minus_alpha, seg[0], seg[1], stream);
        launch_spmv_nodiff_mid(d_new_offsets, d_new_indices, d_x, pr_new, d_pers_full, d_dangling_sum,
                               alpha, one_minus_alpha, seg[1], seg[2], stream);
        launch_spmv_nodiff_low(d_new_offsets, d_new_indices, d_x, pr_new, d_pers_full, d_dangling_sum,
                               alpha, one_minus_alpha, seg[2], seg[4], stream);

        iters_done = iter + 1;
      }

      std::swap(pr_old, pr_new);
    }

    float* final_pr = converged ? pr_new : pr_old;
    cudaMemcpyAsync(pageranks, final_pr, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return PageRankResult{iters_done, converged};
}

}  
