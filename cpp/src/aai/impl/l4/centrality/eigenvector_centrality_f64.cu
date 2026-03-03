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
#include <cstddef>

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)



__device__ __forceinline__ float warp_reduce_sum(float v)
{
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
  return v;
}

__device__ __forceinline__ float block_reduce_sum(float v)
{
  __shared__ float smem[WARPS_PER_BLOCK];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  v = warp_reduce_sum(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  v = (warp == 0) ? ((lane < WARPS_PER_BLOCK) ? smem[lane] : 0.0f) : 0.0f;
  if (warp == 0) v = warp_reduce_sum(v);
  return v;
}



__global__ void convert_f64_to_f32_kernel(const double* __restrict__ in, float* __restrict__ out, int64_t n)
{
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (float)in[i];
}

__global__ void init_uniform_kernel(float* x, int32_t n)
{
  int32_t tid = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) x[tid] = 1.0f / (float)n;
}



__global__ void build_heavy_rows_kernel(const int32_t* __restrict__ offsets,
                                        int32_t n,
                                        int32_t heavy_th,
                                        int32_t* __restrict__ heavy_rows,
                                        int32_t* __restrict__ heavy_count)
{
  int32_t tid = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
  int32_t stride = (int32_t)gridDim.x * blockDim.x;
  for (int32_t row = tid; row < n; row += stride) {
    int32_t deg = __ldg(&offsets[row + 1]) - __ldg(&offsets[row]);
    if (deg > heavy_th) {
      int32_t pos = atomicAdd(heavy_count, 1);
      heavy_rows[pos] = row;
    }
  }
}



template <int TPR>
__global__ void vector_spmv_identity_l2_light_kernel(const int32_t* __restrict__ offsets,
                                                     const int32_t* __restrict__ indices,
                                                     const float* __restrict__ weights,
                                                     const float* __restrict__ x,
                                                     float* __restrict__ y,
                                                     int32_t n,
                                                     int32_t heavy_th,
                                                     float* __restrict__ d_l2_norm_sq)
{
  constexpr int ROWS_PER_WARP = 32 / TPR;
  __shared__ float smem[WARPS_PER_BLOCK];

  const int global_thread = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const int warp_in_grid = global_thread >> 5;
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;

  const int sub_lane = lane % TPR;
  const int row = warp_in_grid * ROWS_PER_WARP + lane / TPR;

  float sum = 0.0f;
  bool active = (row < n);
  int32_t start = 0, end = 0;
  if (active) {
    start = __ldg(&offsets[row]);
    end = __ldg(&offsets[row + 1]);
    active = ((end - start) <= heavy_th);
  }

  if (active) {
    for (int32_t k = start + sub_lane; k < end; k += TPR) {
      int32_t nbr = __ldg(&indices[k]);
      float w = __ldg(&weights[k]);
      sum = fmaf(w, __ldg(&x[nbr]), sum);
    }
  }

  if constexpr (TPR >= 2) sum += __shfl_down_sync(0xffffffff, sum, 1, TPR);
  if constexpr (TPR >= 4) sum += __shfl_down_sync(0xffffffff, sum, 2, TPR);
  if constexpr (TPR >= 8) sum += __shfl_down_sync(0xffffffff, sum, 4, TPR);
  if constexpr (TPR >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8, TPR);
  if constexpr (TPR >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16, TPR);

  float y_sq = 0.0f;
  if (active && sub_lane == 0) {
    float total = sum + __ldg(&x[row]);
    y[row] = total;
    y_sq = total * total;
  }

  
  y_sq = warp_reduce_sum(y_sq);

  if (lane == 0) smem[warp_in_block] = y_sq;
  __syncthreads();

  if (warp_in_block == 0) {
    float val = (lane < WARPS_PER_BLOCK) ? smem[lane] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane == 0 && val != 0.0f) atomicAdd(d_l2_norm_sq, val);
  }
}



__global__ void heavy_spmv_identity_l2_kernel(const int32_t* __restrict__ offsets,
                                              const int32_t* __restrict__ indices,
                                              const float* __restrict__ weights,
                                              const float* __restrict__ x,
                                              float* __restrict__ y,
                                              const int32_t* __restrict__ heavy_rows,
                                              int32_t heavy_count,
                                              float* __restrict__ d_l2_norm_sq)
{
  int32_t idx = (int32_t)blockIdx.x;
  if (idx >= heavy_count) return;
  int32_t row = __ldg(&heavy_rows[idx]);
  int32_t start = __ldg(&offsets[row]);
  int32_t end = __ldg(&offsets[row + 1]);

  float sum = 0.0f;
  for (int32_t k = start + (int32_t)threadIdx.x; k < end; k += BLOCK_SIZE) {
    int32_t nbr = __ldg(&indices[k]);
    float w = __ldg(&weights[k]);
    sum = fmaf(w, __ldg(&x[nbr]), sum);
  }

  sum = block_reduce_sum(sum);
  if (threadIdx.x == 0) {
    float total = sum + __ldg(&x[row]);
    y[row] = total;
    atomicAdd(d_l2_norm_sq, total * total);
  }
}



__global__ void normalize_diff_kernel(float* __restrict__ x_new,
                                      const float* __restrict__ x_old,
                                      int32_t n,
                                      const float* __restrict__ d_l2_sq,
                                      float* __restrict__ d_l1_diff)
{
  __shared__ float smem[WARPS_PER_BLOCK];
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  float l2_sq = *d_l2_sq;
  float inv_norm = (l2_sq > 0.0f) ? rsqrtf(l2_sq) : 0.0f;

  int32_t tid = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
  float local_diff = 0.0f;

  if (tid < n) {
    float new_val = x_new[tid] * inv_norm;
    local_diff = fabsf(new_val - __ldg(&x_old[tid]));
    x_new[tid] = new_val;
  }

  local_diff = warp_reduce_sum(local_diff);
  if (lane == 0) smem[warp_id] = local_diff;
  __syncthreads();

  if (warp_id == 0) {
    float val = (lane < WARPS_PER_BLOCK) ? smem[lane] : 0.0f;
    val = warp_reduce_sum(val);
    if (lane == 0 && val != 0.0f) atomicAdd(d_l1_diff, val);
  }
}

__global__ void cast_f32_to_f64_kernel(const float* __restrict__ in, double* __restrict__ out, int32_t n)
{
  int32_t i = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = (double)in[i];
}



void launch_convert_f64_to_f32(const double* in, float* out, int64_t n, cudaStream_t stream)
{
  int grid = (int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  convert_f64_to_f32_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(in, out, n);
}

void launch_init_uniform(float* x, int32_t n, cudaStream_t stream)
{
  int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  init_uniform_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(x, n);
}

void launch_build_heavy_rows(const int32_t* offsets,
                             int32_t n,
                             int32_t heavy_th,
                             int32_t* heavy_rows,
                             int32_t* heavy_count,
                             cudaStream_t stream)
{
  int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  if (grid > 4096) grid = 4096;
  build_heavy_rows_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(offsets, n, heavy_th, heavy_rows, heavy_count);
}

void launch_spmv_identity_l2_light(const int32_t* offsets,
                                   const int32_t* indices,
                                   const float* weights,
                                   const float* x,
                                   float* y,
                                   int32_t n,
                                   int32_t avg_degree,
                                   int32_t heavy_th,
                                   float* d_l2,
                                   cudaStream_t stream)
{
  auto launch = [&]<int TPR>() {
    constexpr int ROWS_PER_WARP = 32 / TPR;
    int total_threads = ((n + ROWS_PER_WARP - 1) / ROWS_PER_WARP) * 32;
    int grid = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_spmv_identity_l2_light_kernel<TPR><<<grid, BLOCK_SIZE, 0, stream>>>(
        offsets, indices, weights, x, y, n, heavy_th, d_l2);
  };

  if (avg_degree <= 2)
    launch.template operator()<2>();
  else if (avg_degree <= 5)
    launch.template operator()<4>();
  else if (avg_degree <= 20)
    launch.template operator()<8>();
  else if (avg_degree <= 48)
    launch.template operator()<16>();
  else
    launch.template operator()<32>();
}

void launch_spmv_identity_l2_heavy(const int32_t* offsets,
                                   const int32_t* indices,
                                   const float* weights,
                                   const float* x,
                                   float* y,
                                   const int32_t* heavy_rows,
                                   int32_t heavy_count,
                                   float* d_l2,
                                   cudaStream_t stream)
{
  if (heavy_count <= 0) return;
  heavy_spmv_identity_l2_kernel<<<heavy_count, BLOCK_SIZE, 0, stream>>>(
      offsets, indices, weights, x, y, heavy_rows, heavy_count, d_l2);
}

void launch_normalize_diff(float* x_new,
                           const float* x_old,
                           int32_t n,
                           const float* d_l2,
                           float* d_l1,
                           cudaStream_t stream)
{
  int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  normalize_diff_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(x_new, x_old, n, d_l2, d_l1);
}

void launch_cast_f32_to_f64(const float* in, double* out, int32_t n, cudaStream_t stream)
{
  int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cast_f32_to_f64_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(in, out, n);
}



struct Cache : Cacheable {
    
    float* h_l1 = nullptr;
    int32_t* h_heavy_count = nullptr;

    
    float* x0 = nullptr;
    float* x1 = nullptr;
    int32_t* heavy_rows_buf = nullptr;
    int64_t vertex_cap = 0;

    
    float* w_f = nullptr;
    int64_t edge_cap = 0;

    
    float* scalars = nullptr;       
    int32_t* heavy_count_d = nullptr; 

    Cache() {
        cudaMallocHost(&h_l1, sizeof(float));
        cudaMallocHost(&h_heavy_count, sizeof(int32_t));
        cudaMalloc(&scalars, 2 * sizeof(float));
        cudaMalloc(&heavy_count_d, sizeof(int32_t));
    }

    ~Cache() override {
        if (h_l1) cudaFreeHost(h_l1);
        if (h_heavy_count) cudaFreeHost(h_heavy_count);
        if (x0) cudaFree(x0);
        if (x1) cudaFree(x1);
        if (heavy_rows_buf) cudaFree(heavy_rows_buf);
        if (w_f) cudaFree(w_f);
        if (scalars) cudaFree(scalars);
        if (heavy_count_d) cudaFree(heavy_count_d);
    }

    void ensure(int32_t n, int32_t nnz) {
        if (vertex_cap < n) {
            if (x0) cudaFree(x0);
            if (x1) cudaFree(x1);
            if (heavy_rows_buf) cudaFree(heavy_rows_buf);
            cudaMalloc(&x0, (size_t)n * sizeof(float));
            cudaMalloc(&x1, (size_t)n * sizeof(float));
            cudaMalloc(&heavy_rows_buf, (size_t)n * sizeof(int32_t));
            vertex_cap = n;
        }
        if (edge_cap < nnz) {
            if (w_f) cudaFree(w_f);
            cudaMalloc(&w_f, (size_t)nnz * sizeof(float));
            edge_cap = nnz;
        }
    }
};

}  

eigenvector_centrality_result_t eigenvector_centrality(const graph32_t& graph,
                            const double* edge_weights,
                            double* centralities,
                            double epsilon,
                            std::size_t max_iterations,
                            const double* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t n = graph.number_of_vertices;
    int32_t nnz = graph.number_of_edges;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    cudaStream_t stream = 0;

    int32_t avg_degree = (n > 0) ? (nnz / n) : 0;

    cache.ensure(n, nnz);

    
    launch_convert_f64_to_f32(edge_weights, cache.w_f, (int64_t)nnz, stream);

    float* buf[2] = {cache.x0, cache.x1};

    if (initial_centralities != nullptr) {
        launch_convert_f64_to_f32(initial_centralities, buf[0], (int64_t)n, stream);
    } else {
        launch_init_uniform(buf[0], n, stream);
    }

    float* d_l2 = cache.scalars;
    float* d_l1 = d_l2 + 1;

    
    float threshold = (float)((double)n * epsilon);

    
    constexpr int32_t HEAVY_TH = 512;
    int32_t heavy_count = 0;

    cudaMemsetAsync(cache.heavy_count_d, 0, sizeof(int32_t), stream);
    launch_build_heavy_rows(offsets, n, HEAVY_TH, cache.heavy_rows_buf, cache.heavy_count_d, stream);
    cudaMemcpyAsync(cache.h_heavy_count, cache.heavy_count_d, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    heavy_count = *cache.h_heavy_count;
    if (heavy_count < 0) heavy_count = 0;
    if (heavy_count > n) heavy_count = n;

    bool converged = false;
    int64_t iters = 0;

    constexpr int CHECK_INTERVAL = 10;

    int last_dst = 0;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        int src = (int)(iter & 1);
        int dst = 1 - src;
        last_dst = dst;

        
        cudaMemsetAsync(d_l2, 0, 2 * sizeof(float), stream);

        
        launch_spmv_identity_l2_light(offsets, indices, cache.w_f,
                                      buf[src], buf[dst],
                                      n, avg_degree, HEAVY_TH,
                                      d_l2, stream);

        
        if (heavy_count > 0) {
            launch_spmv_identity_l2_heavy(offsets, indices, cache.w_f,
                                          buf[src], buf[dst],
                                          cache.heavy_rows_buf, heavy_count,
                                          d_l2, stream);
        }

        launch_normalize_diff(buf[dst], buf[src], n, d_l2, d_l1, stream);

        iters = (int64_t)(iter + 1);

        if (((iter + 1) % CHECK_INTERVAL) == 0) {
            cudaMemcpyAsync(cache.h_l1, d_l1, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (*cache.h_l1 < threshold) {
                converged = true;
                break;
            }
        }
    }

    if (!converged) {
        cudaMemcpyAsync(cache.h_l1, d_l1, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (*cache.h_l1 < threshold) converged = true;
    }

    
    launch_cast_f32_to_f64(buf[last_dst], centralities, n, stream);

    return {(std::size_t)iters, converged};
}

}  
