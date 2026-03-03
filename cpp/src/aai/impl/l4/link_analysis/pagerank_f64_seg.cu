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
#include <algorithm>

namespace aai {

namespace {





struct Cache : Cacheable {
    double* h_diff_pinned = nullptr;
    double* pr_a = nullptr;
    double* pr_b = nullptr;
    double* out_w = nullptr;
    double* alpha_scale_buf = nullptr;
    double* scalars = nullptr;  

    int64_t pr_a_capacity = 0;
    int64_t pr_b_capacity = 0;
    int64_t out_w_capacity = 0;
    int64_t alpha_scale_capacity = 0;

    Cache() {
        cudaMallocHost(&h_diff_pinned, sizeof(double));
        cudaMalloc(&scalars, 2 * sizeof(double));
    }

    void ensure(int32_t n, int32_t e) {
        if (pr_a_capacity < n) {
            if (pr_a) cudaFree(pr_a);
            cudaMalloc(&pr_a, (size_t)n * sizeof(double));
            pr_a_capacity = n;
        }
        if (pr_b_capacity < n) {
            if (pr_b) cudaFree(pr_b);
            cudaMalloc(&pr_b, (size_t)n * sizeof(double));
            pr_b_capacity = n;
        }
        if (out_w_capacity < n) {
            if (out_w) cudaFree(out_w);
            cudaMalloc(&out_w, (size_t)n * sizeof(double));
            out_w_capacity = n;
        }
        if (alpha_scale_capacity < e) {
            if (alpha_scale_buf) cudaFree(alpha_scale_buf);
            cudaMalloc(&alpha_scale_buf, (size_t)e * sizeof(double));
            alpha_scale_capacity = e;
        }
    }

    ~Cache() override {
        if (h_diff_pinned) cudaFreeHost(h_diff_pinned);
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (out_w) cudaFree(out_w);
        if (alpha_scale_buf) cudaFree(alpha_scale_buf);
        if (scalars) cudaFree(scalars);
    }
};





static __device__ __forceinline__ double warp_reduce_sum(double v)
{
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

template <int BLOCK_THREADS>
static __device__ __forceinline__ double block_reduce_sum(double v)
{
  static_assert(BLOCK_THREADS % 32 == 0);
  __shared__ double warp_sums[BLOCK_THREADS / 32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;

  v = warp_reduce_sum(v);
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();

  double sum = 0.0;
  if (warp == 0) {
    sum = (lane < (BLOCK_THREADS / 32)) ? warp_sums[lane] : 0.0;
    sum = warp_reduce_sum(sum);
  }
  return sum;
}







__global__ void compute_out_weight_sums_kernel(const int32_t* __restrict__ indices,
                                               const double* __restrict__ edge_weights,
                                               double* __restrict__ out_w,
                                               int64_t num_edges)
{
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t j = tid; j < num_edges; j += stride) {
    atomicAdd(out_w + indices[j], edge_weights[j]);
  }
}

__global__ void compute_alpha_scale_kernel(const int32_t* __restrict__ indices,
                                           const double* __restrict__ edge_weights,
                                           const double* __restrict__ out_w,
                                           double* __restrict__ alpha_scale,
                                           double alpha,
                                           int64_t num_edges)
{
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t j = tid; j < num_edges; j += stride) {
    int32_t src = indices[j];
    alpha_scale[j] = alpha * edge_weights[j] / out_w[src];
  }
}

__global__ void init_pr_kernel(double* __restrict__ pr, double val, int32_t n)
{
  int32_t tid = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
  int32_t stride = (int32_t)blockDim.x * gridDim.x;
  for (int32_t i = tid; i < n; i += stride) pr[i] = val;
}





__global__ void dangling_sum_kernel(const double* __restrict__ pr,
                                    const double* __restrict__ out_w,
                                    double* __restrict__ d_dangling_sum,
                                    int32_t n)
{
  int32_t tid = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
  int32_t stride = (int32_t)blockDim.x * gridDim.x;
  double local = 0.0;
  for (int32_t v = tid; v < n; v += stride) {
    if (out_w[v] == 0.0) local += pr[v];
  }

  double block_sum = block_reduce_sum<256>(local);
  if (threadIdx.x == 0 && block_sum != 0.0) atomicAdd(d_dangling_sum, block_sum);
}


__global__ void spmv_high_kernel(const int32_t* __restrict__ offsets,
                                 const int32_t* __restrict__ indices,
                                 const double* __restrict__ alpha_scale,
                                 const double* __restrict__ pr_old,
                                 double* __restrict__ pr_new,
                                 const double* __restrict__ d_dangling_sum,
                                 double one_minus_alpha,
                                 double alpha,
                                 double inv_n,
                                 int32_t start_vertex,
                                 int32_t end_vertex,
                                 double* __restrict__ d_diff)
{
  int32_t v = start_vertex + (int32_t)blockIdx.x;
  if (v >= end_vertex) return;

  int32_t start = offsets[v];
  int32_t end = offsets[v + 1];

  double sum = 0.0;
  for (int32_t j = start + (int32_t)threadIdx.x; j < end; j += (int32_t)blockDim.x) {
    sum += __ldg(alpha_scale + j) * __ldg(pr_old + indices[j]);
  }

  double block_sum = block_reduce_sum<256>(sum);

  if (threadIdx.x == 0) {
    double base = (one_minus_alpha + alpha * (*d_dangling_sum)) * inv_n;
    double new_val = base + block_sum;
    pr_new[v] = new_val;
    double d = fabs(new_val - pr_old[v]);
    if (d != 0.0) atomicAdd(d_diff, d);
  }
}


__global__ void spmv_mid_kernel(const int32_t* __restrict__ offsets,
                                const int32_t* __restrict__ indices,
                                const double* __restrict__ alpha_scale,
                                const double* __restrict__ pr_old,
                                double* __restrict__ pr_new,
                                const double* __restrict__ d_dangling_sum,
                                double one_minus_alpha,
                                double alpha,
                                double inv_n,
                                int32_t start_vertex,
                                int32_t end_vertex,
                                double* __restrict__ d_diff)
{
  constexpr int WARPS_PER_BLOCK = 8;  
  __shared__ double warp_diffs[WARPS_PER_BLOCK];
  __shared__ double base_s;

  if (threadIdx.x == 0) {
    base_s = (one_minus_alpha + alpha * (*d_dangling_sum)) * inv_n;
  }
  __syncthreads();

  int32_t gtid = (int32_t)blockIdx.x * (int32_t)blockDim.x + (int32_t)threadIdx.x;
  int32_t warp_id = gtid >> 5;
  int32_t lane = gtid & 31;
  int32_t warp_in_block = threadIdx.x >> 5;
  int32_t v = start_vertex + warp_id;

  double sum = 0.0;
  if (v < end_vertex) {
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    for (int32_t j = start + lane; j < end; j += 32) {
      sum += __ldg(alpha_scale + j) * __ldg(pr_old + indices[j]);
    }
  }

  sum = warp_reduce_sum(sum);

  if (lane == 0) {
    if (v < end_vertex) {
      double new_val = base_s + sum;
      pr_new[v] = new_val;
      warp_diffs[warp_in_block] = fabs(new_val - pr_old[v]);
    } else {
      warp_diffs[warp_in_block] = 0.0;
    }
  }

  __syncthreads();

  
  if (warp_in_block == 0) {
    double val = (lane < WARPS_PER_BLOCK) ? warp_diffs[lane] : 0.0;
    val = warp_reduce_sum(val);
    if (lane == 0 && val != 0.0) atomicAdd(d_diff, val);
  }
}


__global__ void spmv_low_kernel(const int32_t* __restrict__ offsets,
                                const int32_t* __restrict__ indices,
                                const double* __restrict__ alpha_scale,
                                const double* __restrict__ pr_old,
                                double* __restrict__ pr_new,
                                const double* __restrict__ d_dangling_sum,
                                double one_minus_alpha,
                                double alpha,
                                double inv_n,
                                int32_t start_vertex,
                                int32_t end_vertex,
                                double* __restrict__ d_diff)
{
  __shared__ double base_s;
  if (threadIdx.x == 0) {
    base_s = (one_minus_alpha + alpha * (*d_dangling_sum)) * inv_n;
  }
  __syncthreads();

  int32_t v = start_vertex + (int32_t)blockIdx.x * (int32_t)blockDim.x + (int32_t)threadIdx.x;

  double diff_val = 0.0;
  if (v < end_vertex) {
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    double sum = 0.0;
    for (int32_t j = start; j < end; ++j) {
      sum += __ldg(alpha_scale + j) * __ldg(pr_old + indices[j]);
    }
    double new_val = base_s + sum;
    pr_new[v] = new_val;
    diff_val = fabs(new_val - pr_old[v]);
  }

  double block_diff = block_reduce_sum<256>(diff_val);
  if (threadIdx.x == 0 && block_diff != 0.0) atomicAdd(d_diff, block_diff);
}





void launch_compute_out_weight_sums(const int32_t* indices, const double* edge_weights, double* out_w, int64_t num_edges,
                                    cudaStream_t stream)
{
  int block = 256;
  int grid = (int)((num_edges + block - 1) / block);
  if (grid > 65535) grid = 65535;
  compute_out_weight_sums_kernel<<<grid, block, 0, stream>>>(indices, edge_weights, out_w, num_edges);
}

void launch_compute_alpha_scale(const int32_t* indices, const double* edge_weights, const double* out_w, double* alpha_scale, double alpha,
                                int64_t num_edges, cudaStream_t stream)
{
  int block = 256;
  int grid = (int)((num_edges + block - 1) / block);
  if (grid > 65535) grid = 65535;
  compute_alpha_scale_kernel<<<grid, block, 0, stream>>>(indices, edge_weights, out_w, alpha_scale, alpha, num_edges);
}

void launch_init_pr(double* pr, double val, int32_t n, cudaStream_t stream)
{
  int block = 256;
  int grid = (n + block - 1) / block;
  if (grid > 65535) grid = 65535;
  init_pr_kernel<<<grid, block, 0, stream>>>(pr, val, n);
}

void launch_dangling_sum(const double* pr, const double* out_w, double* dangling_sum, int32_t n, cudaStream_t stream)
{
  int block = 256;
  int grid = (n + block - 1) / block;
  if (grid > 1024) grid = 1024;
  dangling_sum_kernel<<<grid, block, 0, stream>>>(pr, out_w, dangling_sum, n);
}

void launch_spmv_segments(const int32_t* offsets, const int32_t* indices, const double* alpha_scale, const double* pr_old, double* pr_new,
                          const double* d_dangling_sum, double one_minus_alpha, double alpha, double inv_n,
                          int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3, int32_t seg4,
                          double* d_diff, cudaStream_t stream)
{
  constexpr int BLOCK = 256;
  if (seg1 > seg0) {
    int32_t nverts = seg1 - seg0;
    spmv_high_kernel<<<nverts, BLOCK, 0, stream>>>(offsets, indices, alpha_scale, pr_old, pr_new, d_dangling_sum, one_minus_alpha, alpha, inv_n,
                                                   seg0, seg1, d_diff);
  }
  if (seg2 > seg1) {
    int32_t nverts = seg2 - seg1;
    int warps_per_block = BLOCK / 32;
    int grid = (nverts + warps_per_block - 1) / warps_per_block;
    spmv_mid_kernel<<<grid, BLOCK, 0, stream>>>(offsets, indices, alpha_scale, pr_old, pr_new, d_dangling_sum, one_minus_alpha, alpha, inv_n,
                                                seg1, seg2, d_diff);
  }
  if (seg4 > seg2) {
    int32_t nverts = seg4 - seg2;
    int grid = (nverts + BLOCK - 1) / BLOCK;
    spmv_low_kernel<<<grid, BLOCK, 0, stream>>>(offsets, indices, alpha_scale, pr_old, pr_new, d_dangling_sum, one_minus_alpha, alpha, inv_n,
                                                seg2, seg4, d_diff);
  }
}

}  

PageRankResult pagerank_seg(const graph32_t& graph,
                            const double* edge_weights,
                            double* pageranks,
                            const double* precomputed_vertex_out_weight_sums,
                            double alpha,
                            double epsilon,
                            std::size_t max_iterations,
                            const double* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n = graph.number_of_vertices;
    int32_t e = graph.number_of_edges;

    const auto& seg = graph.segment_offsets.value();

    cache.ensure(n, e);

    cudaStream_t stream = 0;

    double* pr_old = cache.pr_a;
    double* pr_new = cache.pr_b;
    double* out_w = cache.out_w;
    double* alpha_scale = cache.alpha_scale_buf;
    double* d_dangling_sum = cache.scalars;
    double* d_diff = cache.scalars + 1;

    cudaMemsetAsync(out_w, 0, (size_t)n * sizeof(double), stream);
    if (e > 0) {
        launch_compute_out_weight_sums(d_indices, edge_weights, out_w, (int64_t)e, stream);
    }

    if (e > 0) {
        launch_compute_alpha_scale(d_indices, edge_weights, out_w, alpha_scale, alpha, (int64_t)e, stream);
    }

    
    if (initial_pageranks) {
        cudaMemcpyAsync(pr_old, initial_pageranks, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else {
        launch_init_pr(pr_old, 1.0 / (double)n, n, stream);
    }

    const double one_minus_alpha = 1.0 - alpha;
    const double inv_n = 1.0 / (double)n;

    std::size_t iterations = 0;
    bool converged = false;

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        
        double zeros[2] = {0.0, 0.0};
        cudaMemcpyAsync(d_dangling_sum, zeros, 2 * sizeof(double), cudaMemcpyHostToDevice, stream);

        launch_dangling_sum(pr_old, out_w, d_dangling_sum, n, stream);

        launch_spmv_segments(d_offsets, d_indices, alpha_scale, pr_old, pr_new,
                             d_dangling_sum, one_minus_alpha, alpha, inv_n,
                             seg[0], seg[1], seg[2], seg[3], seg[4],
                             d_diff, stream);

        std::swap(pr_old, pr_new);
        iterations = iter + 1;

        cudaMemcpyAsync(cache.h_diff_pinned, d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (*cache.h_diff_pinned < epsilon) {
            converged = true;
            break;
        }
    }

    
    cudaMemcpyAsync(pageranks, pr_old, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return PageRankResult{iterations, converged};
}

}  
