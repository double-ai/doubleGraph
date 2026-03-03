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
#include <cub/block/block_reduce.cuh>
#include <cub/cub.cuh>
#include <cstdint>
#include <cmath>

namespace aai {

namespace {

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t s_ = (call); \
    if (s_ != CUSPARSE_STATUS_SUCCESS) \
        throw std::runtime_error("cuSPARSE error: " + std::to_string((int)s_)); \
} while(0)





struct Cache : Cacheable {
  cusparseHandle_t handle{nullptr};

  
  void* cub_temp{nullptr};
  size_t cub_temp_capacity{0};

  
  void* spmv_buf{nullptr};
  size_t spmv_buf_capacity{0};

  
  int32_t* tmp_offsets{nullptr};
  int64_t tmp_offsets_capacity{0};

  
  int32_t* tmp_indices{nullptr};
  int64_t tmp_indices_capacity{0};

  
  float* tmp_weights{nullptr};
  int64_t tmp_weights_capacity{0};

  
  float* scalar_out{nullptr};
  bool scalar_out_allocated{false};

  
  float* x0{nullptr};
  int64_t x0_capacity{0};

  float* x1{nullptr};
  int64_t x1_capacity{0};

  Cache() {
    CUSPARSE_CHECK(cusparseCreate(&handle));
  }

  void ensure_cub_temp(size_t needed) {
    if (needed > cub_temp_capacity) {
      if (cub_temp) cudaFree(cub_temp);
      cudaMalloc(&cub_temp, needed);
      cub_temp_capacity = needed;
    }
  }

  void ensure_spmv_buf(size_t needed) {
    if (needed > spmv_buf_capacity) {
      if (spmv_buf) cudaFree(spmv_buf);
      cudaMalloc(&spmv_buf, needed);
      spmv_buf_capacity = needed;
    }
  }

  void ensure_tmp_offsets(int64_t needed) {
    if (needed > tmp_offsets_capacity) {
      if (tmp_offsets) cudaFree(tmp_offsets);
      cudaMalloc(&tmp_offsets, needed * sizeof(int32_t));
      tmp_offsets_capacity = needed;
    }
  }

  void ensure_tmp_indices(int64_t needed) {
    if (needed > tmp_indices_capacity) {
      if (tmp_indices) cudaFree(tmp_indices);
      cudaMalloc(&tmp_indices, needed * sizeof(int32_t));
      tmp_indices_capacity = needed;
    }
  }

  void ensure_tmp_weights(int64_t needed) {
    if (needed > tmp_weights_capacity) {
      if (tmp_weights) cudaFree(tmp_weights);
      cudaMalloc(&tmp_weights, needed * sizeof(float));
      tmp_weights_capacity = needed;
    }
  }

  void ensure_scalar_out() {
    if (!scalar_out_allocated) {
      cudaMalloc(&scalar_out, sizeof(float));
      scalar_out_allocated = true;
    }
  }

  void ensure_x0(int64_t needed) {
    if (needed > x0_capacity) {
      if (x0) cudaFree(x0);
      cudaMalloc(&x0, needed * sizeof(float));
      x0_capacity = needed;
    }
  }

  void ensure_x1(int64_t needed) {
    if (needed > x1_capacity) {
      if (x1) cudaFree(x1);
      cudaMalloc(&x1, needed * sizeof(float));
      x1_capacity = needed;
    }
  }

  ~Cache() override {
    if (handle) cusparseDestroy(handle);
    if (cub_temp) cudaFree(cub_temp);
    if (spmv_buf) cudaFree(spmv_buf);
    if (tmp_offsets) cudaFree(tmp_offsets);
    if (tmp_indices) cudaFree(tmp_indices);
    if (tmp_weights) cudaFree(tmp_weights);
    if (scalar_out) cudaFree(scalar_out);
    if (x0) cudaFree(x0);
    if (x1) cudaFree(x1);
  }
};





__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ degrees,
    int32_t num_vertices) {
  int v = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (v >= num_vertices) return;

  int32_t start = offsets[v];
  int32_t end = offsets[v + 1];
  if (start >= end) {
    degrees[v] = 0;
    return;
  }

  int count = 0;
  int32_t word_start = start >> 5;
  int32_t word_end = (end - 1) >> 5;

  if (word_start == word_end) {
    uint32_t mask = edge_mask[word_start];
    uint32_t lo_bit = (uint32_t)(start & 31);
    uint32_t hi_bit = (uint32_t)((end - 1) & 31);
    uint32_t range_mask = ((2u << hi_bit) - 1u) & ~((1u << lo_bit) - 1u);
    count = __popc(mask & range_mask);
  } else {
    uint32_t first_mask = edge_mask[word_start] & ~((1u << (start & 31)) - 1u);
    count = __popc(first_mask);
    for (int32_t w = word_start + 1; w < word_end; w++) {
      count += __popc(edge_mask[w]);
    }
    uint32_t last_mask = edge_mask[word_end] & ((2u << ((end - 1) & 31)) - 1u);
    count += __popc(last_mask);
  }

  degrees[v] = count;
}

__global__ void compact_edges_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ new_offsets,
    const int32_t* __restrict__ old_indices,
    const float* __restrict__ old_weights,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ new_indices,
    float* __restrict__ new_weights,
    int32_t num_vertices) {

  int v = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (v >= num_vertices) return;

  int32_t old_start = old_offsets[v];
  int32_t old_end = old_offsets[v + 1];
  int32_t new_start = new_offsets[v];
  int idx = 0;

  for (int32_t e = old_start; e < old_end; e++) {
    if ((edge_mask[e >> 5] >> (e & 31)) & 1u) {
      new_indices[new_start + idx] = old_indices[e];
      new_weights[new_start + idx] = old_weights[e];
      idx++;
    }
  }
}





template <int BLOCK>
__global__ void init_beta_betas_atomic_kernel(
    const float* __restrict__ betas,
    float* __restrict__ x_new,
    int32_t n,
    float* __restrict__ out_delta) {

  int v = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  float val = 0.0f;
  if (v < n) {
    val = betas[v];
    x_new[v] = val;
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK>;
  __shared__ typename BlockReduce::TempStorage tmp;
  float block_sum = BlockReduce(tmp).Sum(fabsf(val));
  if (threadIdx.x == 0) {
    atomicAdd(out_delta, block_sum);
  }
}





template <int BLOCK>
__global__ void add_beta_and_delta_atomic_kernel(
    const float* __restrict__ spmv_out,  
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float* __restrict__ betas,
    float beta_scalar,
    int32_t n,
    float* __restrict__ out_delta) {

  int v = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  float my_diff = 0.0f;

  if (v < n) {
    float beta_v = (betas != nullptr) ? betas[v] : beta_scalar;
    float new_val = spmv_out[v] + beta_v;
    x_new[v] = new_val;
    my_diff = fabsf(new_val - x_old[v]);
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK>;
  __shared__ typename BlockReduce::TempStorage tmp;
  float block_sum = BlockReduce(tmp).Sum(my_diff);
  if (threadIdx.x == 0) {
    atomicAdd(out_delta, block_sum);
  }
}





template <int BLOCK>
__global__ void l2_norm_sq_atomic_kernel(const float* __restrict__ x, int32_t n, float* __restrict__ out) {
  int tid = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  float v = 0.0f;
  if (tid < n) {
    float xi = x[tid];
    v = xi * xi;
  }
  using BlockReduce = cub::BlockReduce<float, BLOCK>;
  __shared__ typename BlockReduce::TempStorage tmp;
  float block_sum = BlockReduce(tmp).Sum(v);
  if (threadIdx.x == 0) {
    atomicAdd(out, block_sum);
  }
}

__global__ void fill_kernel(float* __restrict__ x, float val, int32_t n) {
  int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (i < n) x[i] = val;
}

__global__ void scale_kernel(float* __restrict__ x, float scale, int32_t n) {
  int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (i < n) x[i] *= scale;
}





static void launch_count_active_edges(const int32_t* offsets, const uint32_t* edge_mask,
                              int32_t* degrees, int32_t num_vertices) {
  int block = 256;
  int grid = (num_vertices + block - 1) / block;
  count_active_edges_kernel<<<grid, block>>>(offsets, edge_mask, degrees, num_vertices);
}

static size_t get_prefix_sum_temp_size(int32_t num_items) {
  size_t temp_size = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, temp_size, (int32_t*)nullptr, (int32_t*)nullptr, num_items);
  return temp_size;
}

static void launch_prefix_sum(void* temp, size_t temp_size,
                       const int32_t* input, int32_t* output, int32_t num_items) {
  cub::DeviceScan::ExclusiveSum(temp, temp_size, input, output, num_items);
}

static void launch_compact_edges(const int32_t* old_offsets, const int32_t* new_offsets,
                          const int32_t* old_indices, const float* old_weights,
                          const uint32_t* edge_mask,
                          int32_t* new_indices, float* new_weights,
                          int32_t num_vertices) {
  int block = 256;
  int grid = (num_vertices + block - 1) / block;
  compact_edges_kernel<<<grid, block>>>(old_offsets, new_offsets, old_indices, old_weights,
                                        edge_mask, new_indices, new_weights, num_vertices);
}

static void launch_fill_float(float* x, float val, int32_t n) {
  int block = 256;
  int grid = (n + block - 1) / block;
  fill_kernel<<<grid, block>>>(x, val, n);
}

static void launch_init_beta_betas_atomic(const float* betas, float* x_new, int32_t n, float* out_delta) {
  constexpr int BLOCK = 256;
  int grid = (n + BLOCK - 1) / BLOCK;
  init_beta_betas_atomic_kernel<BLOCK><<<grid, BLOCK>>>(betas, x_new, n, out_delta);
}

static void launch_add_beta_and_delta_atomic(const float* spmv_out, const float* x_old, float* x_new,
                                     const float* betas, float beta_scalar,
                                     int32_t n, float* out_delta) {
  constexpr int BLOCK = 256;
  int grid = (n + BLOCK - 1) / BLOCK;
  add_beta_and_delta_atomic_kernel<BLOCK><<<grid, BLOCK>>>(
      spmv_out, x_old, x_new, betas, beta_scalar, n, out_delta);
}

static void launch_l2_norm_sq_atomic(const float* x, int32_t n, float* out) {
  constexpr int BLOCK = 256;
  int grid = (n + BLOCK - 1) / BLOCK;
  l2_norm_sq_atomic_kernel<BLOCK><<<grid, BLOCK>>>(x, n, out);
}

static void launch_scale_inplace(float* x, float scale, int32_t n) {
  int block = 256;
  int grid = (n + block - 1) / block;
  scale_kernel<<<grid, block>>>(x, scale, n);
}

}  





katz_centrality_result_t katz_centrality_mask(const graph32_t& graph,
                           const float* edge_weights,
                           float* centralities,
                           float alpha,
                           float beta,
                           const float* betas,
                           float epsilon,
                           std::size_t max_iterations,
                           bool has_initial_guess,
                           bool normalize) {
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);

  int32_t num_vertices = graph.number_of_vertices;
  int32_t num_edges = graph.number_of_edges;
  const int32_t* offsets = graph.offsets;
  const int32_t* indices = graph.indices;
  const float* weights = edge_weights;
  const uint32_t* edge_mask = graph.edge_mask;

  float beta_scalar = beta;
  const float* d_betas = betas;

  
  float epsilon_eff = fmaxf(epsilon, static_cast<float>(num_vertices) * 1.0e-6f);

  
  
  
  size_t needed_temp = get_prefix_sum_temp_size(num_vertices + 1);
  cache.ensure_cub_temp(needed_temp);

  int64_t need_off = static_cast<int64_t>(num_vertices) + 1;
  cache.ensure_tmp_offsets(need_off);
  int32_t* new_offsets = cache.tmp_offsets;

  launch_count_active_edges(offsets, edge_mask, new_offsets, num_vertices);
  cudaMemsetAsync(new_offsets + num_vertices, 0, sizeof(int32_t));
  
  launch_prefix_sum(cache.cub_temp, cache.cub_temp_capacity, new_offsets, new_offsets, num_vertices + 1);

  
  int32_t nnz = 0;
  cudaMemcpy(&nnz, new_offsets + num_vertices, sizeof(int32_t), cudaMemcpyDeviceToHost);
  if (nnz < 0) nnz = 0;

  int64_t need_nnz = static_cast<int64_t>(nnz);
  if (need_nnz < 1) need_nnz = 1;
  cache.ensure_tmp_indices(need_nnz);
  cache.ensure_tmp_weights(need_nnz);

  if (nnz > 0) {
    launch_compact_edges(offsets, new_offsets, indices, weights, edge_mask,
                         cache.tmp_indices, cache.tmp_weights,
                         num_vertices);
  }

  
  
  
  cache.ensure_x0(num_vertices);
  cache.ensure_x1(num_vertices);
  cache.ensure_scalar_out();

  float* x0 = cache.x0;
  float* x1 = cache.x1;
  float* scalar_out = cache.scalar_out;

  bool converged = false;
  size_t iterations = 0;

  if (max_iterations == 0) {
    if (has_initial_guess) {
      cudaMemcpyAsync(x0, centralities,
                      size_t(num_vertices) * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
      cudaMemsetAsync(x0, 0, size_t(num_vertices) * sizeof(float));
    }
    converged = false;
    iterations = 0;
    
    cudaMemcpyAsync(centralities, x0,
                    size_t(num_vertices) * sizeof(float), cudaMemcpyDeviceToDevice);
  } else {
    int cur = 0;

    if (has_initial_guess) {
      cudaMemcpyAsync(x0, centralities,
                      size_t(num_vertices) * sizeof(float), cudaMemcpyDeviceToDevice);
      cur = 0;
      iterations = 0;
    } else {
      cudaMemsetAsync(x0, 0, size_t(num_vertices) * sizeof(float));

      
      float h_delta0 = 0.0f;
      if (d_betas != nullptr) {
        cudaMemsetAsync(scalar_out, 0, sizeof(float));
        launch_init_beta_betas_atomic(d_betas, x1, num_vertices, scalar_out);
        cudaMemcpy(&h_delta0, scalar_out, sizeof(float), cudaMemcpyDeviceToHost);
      } else {
        launch_fill_float(x1, beta_scalar, num_vertices);
        h_delta0 = static_cast<float>(static_cast<double>(num_vertices) * fabs((double)beta_scalar));
      }

      iterations = 1;
      cur = 1;

      if (h_delta0 < epsilon_eff) {
        converged = true;
      }
    }

    
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr;
    cusparseDnVecDescr_t vecY = nullptr;

    if (nnz > 0) {
      CUSPARSE_CHECK(cusparseCreateCsr(
                        &matA,
                        (int64_t)num_vertices,
                        (int64_t)num_vertices,
                        (int64_t)nnz,
                        (void*)new_offsets,
                        (void*)cache.tmp_indices,
                        (void*)cache.tmp_weights,
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO,
                        CUDA_R_32F));

      CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, (int64_t)num_vertices, (void*)(cur == 0 ? x0 : x1), CUDA_R_32F));
      CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, (int64_t)num_vertices, (void*)(cur == 0 ? x1 : x0), CUDA_R_32F));

      
      size_t bufSize = 0;
      float zero = 0.0f;
      CUSPARSE_CHECK(cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, matA, vecX, &zero, vecY,
                                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize));

      cache.ensure_spmv_buf(bufSize);
    }

    for (size_t iter = iterations; iter < max_iterations && !converged; ++iter) {
      cudaMemsetAsync(scalar_out, 0, sizeof(float));

      if (nnz > 0) {
        
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecX, (void*)(cur == 0 ? x0 : x1)));
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecY, (void*)(cur == 0 ? x1 : x0)));

        float zero = 0.0f;
        CUSPARSE_CHECK(cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &alpha, matA, vecX, &zero, vecY,
                                   CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                   cache.spmv_buf));
      } else {
        
        cudaMemsetAsync((cur == 0 ? x1 : x0), 0, size_t(num_vertices) * sizeof(float));
      }

      
      launch_add_beta_and_delta_atomic((cur == 0 ? x1 : x0),
                                       cur == 0 ? x0 : x1,
                                       cur == 0 ? x1 : x0,
                                       d_betas, beta_scalar,
                                       num_vertices, scalar_out);

      float h_delta;
      cudaMemcpy(&h_delta, scalar_out, sizeof(float), cudaMemcpyDeviceToHost);

      cur = 1 - cur;
      iterations = iter + 1;

      if (h_delta < epsilon_eff) {
        converged = true;
        break;
      }
    }

    if (matA) cusparseDestroySpMat(matA);
    if (vecX) cusparseDestroyDnVec(vecX);
    if (vecY) cusparseDestroyDnVec(vecY);

    float* result_ptr = (cur == 0) ? x0 : x1;

    if (normalize) {
      cudaMemsetAsync(scalar_out, 0, sizeof(float));
      launch_l2_norm_sq_atomic(result_ptr, num_vertices, scalar_out);
      float h_norm_sq;
      cudaMemcpy(&h_norm_sq, scalar_out, sizeof(float), cudaMemcpyDeviceToHost);
      float h_norm = sqrtf(h_norm_sq);
      if (h_norm > 0.0f) {
        float inv = 1.0f / h_norm;
        launch_scale_inplace(result_ptr, inv, num_vertices);
      }
    }

    
    cudaMemcpyAsync(centralities, result_ptr,
                    size_t(num_vertices) * sizeof(float), cudaMemcpyDeviceToDevice);
  }

  return katz_centrality_result_t{iterations, converged};
}

}  
