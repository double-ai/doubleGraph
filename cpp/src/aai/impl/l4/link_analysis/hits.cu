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
#include <cusparse.h>
#include <cstdint>
#include <limits>
#include <math_constants.h>

namespace aai {

namespace {





static inline __device__ float warp_reduce_sum(float v)
{
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

static inline __device__ float warp_reduce_max(float v)
{
  for (int offset = 16; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
  }
  return v;
}

template <int BLOCK, int ITEMS_PER_THREAD>
__global__ void reduce_max_stage1(const float* __restrict__ x, int64_t n, float* __restrict__ block_out)
{
  float local = -CUDART_INF_F;
  int64_t tid = (int64_t)blockIdx.x * BLOCK + threadIdx.x;
  int64_t stride = (int64_t)gridDim.x * BLOCK;

  for (int64_t i = tid; i < n; i += stride * ITEMS_PER_THREAD) {
#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
      int64_t idx = i + (int64_t)j * stride;
      if (idx < n) local = fmaxf(local, x[idx]);
    }
  }

  local = warp_reduce_max(local);
  __shared__ float smem[BLOCK / 32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = local;
  __syncthreads();

  float block_max = -CUDART_INF_F;
  if (warp == 0) {
    block_max = (lane < (BLOCK / 32)) ? smem[lane] : -CUDART_INF_F;
    block_max = warp_reduce_max(block_max);
    if (lane == 0) block_out[blockIdx.x] = block_max;
  }
}

template <int BLOCK>
__global__ void reduce_max_stage2(const float* __restrict__ x, int64_t n, float* __restrict__ out)
{
  float local = -CUDART_INF_F;
  for (int64_t i = threadIdx.x; i < n; i += BLOCK) {
    local = fmaxf(local, x[i]);
  }
  local = warp_reduce_max(local);
  __shared__ float smem[BLOCK / 32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = local;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < (BLOCK / 32)) ? smem[lane] : -CUDART_INF_F;
    v = warp_reduce_max(v);
    if (lane == 0) *out = v;
  }
}

template <int BLOCK, int ITEMS_PER_THREAD>
__global__ void reduce_sum_stage1(const float* __restrict__ x, int64_t n, float* __restrict__ block_out)
{
  float local = 0.0f;
  int64_t tid = (int64_t)blockIdx.x * BLOCK + threadIdx.x;
  int64_t stride = (int64_t)gridDim.x * BLOCK;

  for (int64_t i = tid; i < n; i += stride * ITEMS_PER_THREAD) {
#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
      int64_t idx = i + (int64_t)j * stride;
      if (idx < n) local += x[idx];
    }
  }

  local = warp_reduce_sum(local);
  __shared__ float smem[BLOCK / 32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = local;
  __syncthreads();

  if (warp == 0) {
    float v = (lane < (BLOCK / 32)) ? smem[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0) block_out[blockIdx.x] = v;
  }
}

template <int BLOCK>
__global__ void reduce_sum_stage2(const float* __restrict__ x, int64_t n, float* __restrict__ out)
{
  float local = 0.0f;
  for (int64_t i = threadIdx.x; i < n; i += BLOCK) {
    local += x[i];
  }
  local = warp_reduce_sum(local);
  __shared__ float smem[BLOCK / 32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = local;
  __syncthreads();
  if (warp == 0) {
    float v = (lane < (BLOCK / 32)) ? smem[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0) *out = v;
  }
}

__global__ void fill_const(float* __restrict__ x, int64_t n, float v)
{
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    x[i] = v;
  }
}

__global__ void fill_ones(float* __restrict__ x, int64_t n)
{
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    x[i] = 1.0f;
  }
}

__global__ void scale_div(float* __restrict__ x, int64_t n, const float* __restrict__ denom)
{
  float d = denom[0];
  float inv = (d > 0.0f) ? 1.0f / d : 1.0f;
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t i = idx; i < n; i += stride) {
    x[i] *= inv;
  }
}

template <int BLOCK, int ITEMS_PER_THREAD>
__global__ void scale_div_and_diff_stage1(const float* __restrict__ oldx,
                                         float* __restrict__ newx,
                                         int64_t n,
                                         const float* __restrict__ denom,
                                         float* __restrict__ block_out)
{
  float d = denom[0];
  float inv = (d > 0.0f) ? 1.0f / d : 1.0f;

  float local = 0.0f;
  int64_t tid = (int64_t)blockIdx.x * BLOCK + threadIdx.x;
  int64_t stride = (int64_t)gridDim.x * BLOCK;

  for (int64_t i = tid; i < n; i += stride * ITEMS_PER_THREAD) {
#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
      int64_t idx = i + (int64_t)j * stride;
      if (idx < n) {
        float v = newx[idx] * inv;
        newx[idx] = v;
        local += fabsf(v - oldx[idx]);
      }
    }
  }

  local = warp_reduce_sum(local);
  __shared__ float smem[BLOCK / 32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = local;
  __syncthreads();

  if (warp == 0) {
    float v = (lane < (BLOCK / 32)) ? smem[lane] : 0.0f;
    v = warp_reduce_sum(v);
    if (lane == 0) block_out[blockIdx.x] = v;
  }
}





void launch_fill_const(float* x, int64_t n, float v)
{
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  if (grid > 65535) grid = 65535;
  fill_const<<<grid, block>>>(x, n, v);
}

void launch_fill_ones(float* x, int64_t n)
{
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  if (grid > 65535) grid = 65535;
  fill_ones<<<grid, block>>>(x, n);
}

void launch_scale_div(float* x, int64_t n, const float* denom)
{
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  if (grid > 65535) grid = 65535;
  scale_div<<<grid, block>>>(x, n, denom);
}

void launch_reduce_max(const float* x, int64_t n, float* scratch, float* out)
{
  constexpr int BLOCK = 256;
  constexpr int ITEMS = 4;
  int64_t elems_per_block = (int64_t)BLOCK * ITEMS;
  int grid = (int)((n + elems_per_block - 1) / elems_per_block);
  if (grid < 1) grid = 1;
  if (grid > 65535) grid = 65535;
  reduce_max_stage1<BLOCK, ITEMS><<<grid, BLOCK>>>(x, n, scratch);
  reduce_max_stage2<BLOCK><<<1, BLOCK>>>(scratch, grid, out);
}

void launch_reduce_sum(const float* x, int64_t n, float* scratch, float* out)
{
  constexpr int BLOCK = 256;
  constexpr int ITEMS = 4;
  int64_t elems_per_block = (int64_t)BLOCK * ITEMS;
  int grid = (int)((n + elems_per_block - 1) / elems_per_block);
  if (grid < 1) grid = 1;
  if (grid > 65535) grid = 65535;
  reduce_sum_stage1<BLOCK, ITEMS><<<grid, BLOCK>>>(x, n, scratch);
  reduce_sum_stage2<BLOCK><<<1, BLOCK>>>(scratch, grid, out);
}

void launch_scale_div_and_diff(const float* oldx,
                              float* newx,
                              int64_t n,
                              const float* denom,
                              float* scratch,
                              float* out)
{
  constexpr int BLOCK = 256;
  constexpr int ITEMS = 4;
  int64_t elems_per_block = (int64_t)BLOCK * ITEMS;
  int grid = (int)((n + elems_per_block - 1) / elems_per_block);
  if (grid < 1) grid = 1;
  if (grid > 65535) grid = 65535;
  scale_div_and_diff_stage1<BLOCK, ITEMS><<<grid, BLOCK>>>(oldx, newx, n, denom, scratch);
  reduce_sum_stage2<BLOCK><<<1, BLOCK>>>(scratch, grid, out);
}





struct Cache : Cacheable {
  cusparseHandle_t handle_{nullptr};

  
  float* d_tmp0_{nullptr};
  float* d_tmp1_{nullptr};
  float* d_tmp2_{nullptr};

  
  float* d_reduce_scratch_{nullptr};

  
  float* h_diff_{nullptr};

  
  void* d_spmv_buf_{nullptr};
  size_t spmv_buf_capacity_{0};

  
  float* values_{nullptr};
  int64_t values_capacity_{0};

  
  float* hubs_tmp_{nullptr};
  int64_t hubs_tmp_capacity_{0};

  Cache() {
    cusparseCreate(&handle_);
    cusparseSetStream(handle_, 0);
    cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_HOST);

    cudaMalloc(&d_tmp0_, sizeof(float));
    cudaMalloc(&d_tmp1_, sizeof(float));
    cudaMalloc(&d_tmp2_, sizeof(float));
    cudaMalloc(&d_reduce_scratch_, 65535 * sizeof(float));
    cudaHostAlloc(&h_diff_, sizeof(float), cudaHostAllocPortable);
  }

  ~Cache() override {
    if (hubs_tmp_) cudaFree(hubs_tmp_);
    if (values_) cudaFree(values_);
    if (d_spmv_buf_) cudaFree(d_spmv_buf_);
    if (d_reduce_scratch_) cudaFree(d_reduce_scratch_);
    if (d_tmp0_) cudaFree(d_tmp0_);
    if (d_tmp1_) cudaFree(d_tmp1_);
    if (d_tmp2_) cudaFree(d_tmp2_);
    if (h_diff_) cudaFreeHost(h_diff_);
    if (handle_) cusparseDestroy(handle_);
  }

  void ensure_values(int64_t nnz) {
    if (values_capacity_ < nnz) {
      if (values_) cudaFree(values_);
      cudaMalloc(&values_, nnz * sizeof(float));
      values_capacity_ = nnz;
      launch_fill_ones(values_, nnz);
    }
  }

  void ensure_hubs_tmp(int64_t n) {
    if (hubs_tmp_capacity_ < n) {
      if (hubs_tmp_) cudaFree(hubs_tmp_);
      cudaMalloc(&hubs_tmp_, n * sizeof(float));
      hubs_tmp_capacity_ = n;
    }
  }

  void ensure_spmv_buf(size_t bytes) {
    if (spmv_buf_capacity_ < bytes) {
      if (d_spmv_buf_) cudaFree(d_spmv_buf_);
      cudaMalloc(&d_spmv_buf_, bytes);
      spmv_buf_capacity_ = bytes;
    }
  }
};

}  

HitsResult hits(const graph32_t& graph,
                float* hubs,
                float* authorities,
                float epsilon,
                std::size_t max_iterations,
                bool has_initial_hubs_guess,
                bool normalize) {
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);

  int32_t n = graph.number_of_vertices;
  int32_t nnz_i32 = graph.number_of_edges;
  int64_t nnz = static_cast<int64_t>(nnz_i32);

  if (n == 0) {
    return HitsResult{max_iterations, false, std::numeric_limits<float>::max()};
  }

  cache.ensure_hubs_tmp(n);

  float* prev_hubs = hubs;
  float* curr_hubs = cache.hubs_tmp_;

  if (has_initial_hubs_guess) {
    launch_reduce_sum(prev_hubs, n, cache.d_reduce_scratch_, cache.d_tmp0_);
    launch_scale_div(prev_hubs, n, cache.d_tmp0_);
  } else {
    launch_fill_const(prev_hubs, n, 1.0f / (float)n);
  }

  cache.ensure_values(nnz);

  
  
  cusparseSpMatDescr_t matB{nullptr};
  cusparseCreateCsr(&matB,
                    n, n, nnz,
                    const_cast<int32_t*>(graph.offsets),
                    const_cast<int32_t*>(graph.indices),
                    cache.values_,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_32F);

  
  cusparseDnVecDescr_t vecX{nullptr};
  cusparseDnVecDescr_t vecY{nullptr};
  cusparseCreateDnVec(&vecX, n, (void*)prev_hubs, CUDA_R_32F);
  cusparseCreateDnVec(&vecY, n, (void*)authorities, CUDA_R_32F);

  float alpha = 1.0f;
  float beta = 0.0f;

  
  cusparseSpMVAlg_t alg = CUSPARSE_SPMV_CSR_ALG1;

  
  size_t bufN = 0;
  size_t bufT = 0;
  cusparseSpMV_bufferSize(cache.handle_,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          &alpha, matB, vecX, &beta, vecY,
                          CUDA_R_32F, alg, &bufN);

  
  cusparseDnVecSetValues(vecX, (void*)authorities);
  cusparseDnVecSetValues(vecY, (void*)curr_hubs);
  cusparseSpMV_bufferSize(cache.handle_,
                          CUSPARSE_OPERATION_TRANSPOSE,
                          &alpha, matB, vecX, &beta, vecY,
                          CUDA_R_32F, alg, &bufT);

  size_t need = (bufN > bufT) ? bufN : bufT;
  cache.ensure_spmv_buf(need);

  float tolerance = (float)n * epsilon;
  float diff_sum = std::numeric_limits<float>::max();

  std::size_t iter = 0;
  while (iter < max_iterations) {
    
    cusparseDnVecSetValues(vecX, (void*)prev_hubs);
    cusparseDnVecSetValues(vecY, (void*)authorities);
    cusparseSpMV(cache.handle_,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matB, vecX, &beta, vecY,
                 CUDA_R_32F, alg, cache.d_spmv_buf_);

    
    cusparseDnVecSetValues(vecX, (void*)authorities);
    cusparseDnVecSetValues(vecY, (void*)curr_hubs);
    cusparseSpMV(cache.handle_,
                 CUSPARSE_OPERATION_TRANSPOSE,
                 &alpha, matB, vecX, &beta, vecY,
                 CUDA_R_32F, alg, cache.d_spmv_buf_);

    
    launch_reduce_max(curr_hubs, n, cache.d_reduce_scratch_, cache.d_tmp1_);

    
    launch_reduce_max(authorities, n, cache.d_reduce_scratch_, cache.d_tmp0_);
    launch_scale_div(authorities, n, cache.d_tmp0_);

    
    launch_scale_div_and_diff(prev_hubs, curr_hubs, n, cache.d_tmp1_,
                              cache.d_reduce_scratch_, cache.d_tmp2_);

    cudaMemcpyAsync(cache.h_diff_, cache.d_tmp2_, sizeof(float),
                    cudaMemcpyDeviceToHost, 0);
    cudaStreamSynchronize(0);
    diff_sum = *cache.h_diff_;

    std::swap(prev_hubs, curr_hubs);
    iter++;

    if (diff_sum < tolerance) {
      break;
    }
  }

  if (normalize) {
    launch_reduce_sum(prev_hubs, n, cache.d_reduce_scratch_, cache.d_tmp0_);
    launch_scale_div(prev_hubs, n, cache.d_tmp0_);

    launch_reduce_sum(authorities, n, cache.d_reduce_scratch_, cache.d_tmp0_);
    launch_scale_div(authorities, n, cache.d_tmp0_);
  }

  if (prev_hubs != hubs) {
    cudaMemcpy(hubs, prev_hubs, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice);
  }

  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cusparseDestroySpMat(matB);

  return HitsResult{iter, iter < max_iterations, diff_sum};
}

}  
