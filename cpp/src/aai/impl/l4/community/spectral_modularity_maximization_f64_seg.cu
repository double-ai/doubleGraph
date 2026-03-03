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
#include <cublas_v2.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cstring>
#include <cub/block/block_reduce.cuh>

namespace aai {

namespace {

struct ull_max {
  __device__ __forceinline__ unsigned long long operator()(unsigned long long a, unsigned long long b) const {
    return a > b ? a : b;
  }
};





__global__ void degrees_all_fused_kernel(const int32_t* __restrict__ offsets,
                                        const double* __restrict__ weights,
                                        double* __restrict__ degrees,
                                        int seg1,
                                        int seg2,
                                        int n) {
  using BlockReduce = cub::BlockReduce<double, 256>;
  __shared__ typename BlockReduce::TempStorage temp;

  if (blockIdx.x < seg1) {
    int row = blockIdx.x;
    int start = offsets[row];
    int end = offsets[row + 1];
    double sum = 0.0;
    for (int j = start + threadIdx.x; j < end; j += blockDim.x) sum += weights[j];
    double block_sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) degrees[row] = block_sum;
    return;
  }

  int tid = threadIdx.x;
  int warp_in_block = tid >> 5;
  int lane = tid & 31;
  constexpr int WARPS_PER_BLOCK = 256 / 32;

  int n_mid = seg2 - seg1;
  int n_tail = n - seg2;
  int tail_units = (n_tail + 3) >> 2;
  int total_units = n_mid + tail_units;

  int midlow_block = blockIdx.x - seg1;
  int unit = midlow_block * WARPS_PER_BLOCK + warp_in_block;
  if (unit >= total_units) return;

  if (unit < n_mid) {
    int row = seg1 + unit;
    int start = offsets[row];
    int end = offsets[row + 1];
    double sum = 0.0;
    for (int j = start + lane; j < end; j += 32) sum += weights[j];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) degrees[row] = sum;
  } else {
    constexpr int SUBW = 8;
    int tunit = unit - n_mid;
    int row_base = seg2 + (tunit << 2);
    int row = row_base + (lane >> 3);
    if (row >= n) return;
    int lane_in_row = lane & (SUBW - 1);

    int start = offsets[row];
    int end = offsets[row + 1];
    double sum = 0.0;
    for (int j = start + lane_in_row; j < end; j += SUBW) sum += weights[j];
    sum += __shfl_down_sync(0xffffffff, sum, 4, SUBW);
    sum += __shfl_down_sync(0xffffffff, sum, 2, SUBW);
    sum += __shfl_down_sync(0xffffffff, sum, 1, SUBW);
    if (lane_in_row == 0) degrees[row] = sum;
  }
}

__global__ void spmv_mod_all_fused_kernel(const int32_t* __restrict__ offsets,
                                         const int32_t* __restrict__ indices,
                                         const double* __restrict__ values,
                                         const double* __restrict__ degrees,
                                         double inv_2m,
                                         const double* __restrict__ x,
                                         double* __restrict__ y,
                                         int seg1,
                                         int seg2,
                                         int n,
                                         const double* __restrict__ d_dot_val) {
  using BlockReduce = cub::BlockReduce<double, 256>;
  __shared__ typename BlockReduce::TempStorage temp;

  if (blockIdx.x < seg1) {
    int row = blockIdx.x;
    int start = offsets[row];
    int end = offsets[row + 1];
    double sum = 0.0;
    for (int j = start + threadIdx.x; j < end; j += blockDim.x) {
      int idx = __ldg(indices + j);
      double val = __ldg(values + j);
      double xv = __ldg(x + idx);
      sum += val * xv;
    }
    double block_sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) y[row] = block_sum - __ldg(degrees + row) * (*d_dot_val) * inv_2m;
    return;
  }

  int tid = threadIdx.x;
  int warp_in_block = tid >> 5;
  int lane = tid & 31;
  constexpr int WARPS_PER_BLOCK = 256 / 32;

  int n_mid = seg2 - seg1;
  int n_tail = n - seg2;
  int tail_units = (n_tail + 3) >> 2;
  int total_units = n_mid + tail_units;

  int midlow_block = blockIdx.x - seg1;
  int unit = midlow_block * WARPS_PER_BLOCK + warp_in_block;
  if (unit >= total_units) return;

  double dot = *d_dot_val;

  if (unit < n_mid) {
    int row = seg1 + unit;
    int start = offsets[row];
    int end = offsets[row + 1];
    double sum = 0.0;
    for (int j = start + lane; j < end; j += 32) {
      int idx = __ldg(indices + j);
      double val = __ldg(values + j);
      double xv = __ldg(x + idx);
      sum += val * xv;
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) y[row] = sum - __ldg(degrees + row) * dot * inv_2m;
  } else {
    constexpr int SUBW = 8;
    int tunit = unit - n_mid;
    int row_base = seg2 + (tunit << 2);
    int row = row_base + (lane >> 3);
    if (row >= n) return;
    int lane_in_row = lane & (SUBW - 1);

    int start = offsets[row];
    int end = offsets[row + 1];
    double sum = 0.0;
    for (int j = start + lane_in_row; j < end; j += SUBW) {
      int idx = __ldg(indices + j);
      double val = __ldg(values + j);
      double xv = __ldg(x + idx);
      sum += val * xv;
    }
    sum += __shfl_down_sync(0xffffffff, sum, 4, SUBW);
    sum += __shfl_down_sync(0xffffffff, sum, 2, SUBW);
    sum += __shfl_down_sync(0xffffffff, sum, 1, SUBW);
    if (lane_in_row == 0) y[row] = sum - degrees[row] * dot * inv_2m;
  }
}





__global__ void neg_axpy_kernel(const double* __restrict__ alpha,
                                const double* __restrict__ x,
                                double* __restrict__ y,
                                int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] -= (*alpha) * x[i];
}

__global__ void neg_axpy2_kernel(const double* __restrict__ alpha,
                                 const double* __restrict__ x1,
                                 const double* __restrict__ beta,
                                 const double* __restrict__ x2,
                                 double* __restrict__ y,
                                 int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] -= (*alpha) * x1[i] + (*beta) * x2[i];
}

__global__ void inv_scale_kernel(const double* __restrict__ beta, double* __restrict__ x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double b = *beta;
    if (b > 1e-30) x[i] *= (1.0 / b);
  }
}

__global__ void init_lanczos_kernel(double* __restrict__ v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    unsigned int x = (unsigned int)(i + 1);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    v[i] = (double)(x & 0x7FFFFF) * (1.0 / (double)(0x7FFFFF)) - 0.5;
  }
}

__global__ void gemv_t_kernel(const double* __restrict__ V, int n, int ncols, int lda,
                              const double* __restrict__ w, double* __restrict__ h) {
  int col = blockIdx.x;
  if (col >= ncols) return;
  const double* v_col = V + (int64_t)col * lda;

  double sum = 0.0;
  int n2 = n >> 1;
  const double2* __restrict__ v2 = reinterpret_cast<const double2*>(v_col);
  const double2* __restrict__ w2 = reinterpret_cast<const double2*>(w);
  for (int i2 = threadIdx.x; i2 < n2; i2 += blockDim.x) {
    double2 a = v2[i2];
    double2 b = w2[i2];
    sum += a.x * b.x + a.y * b.y;
  }
  for (int i = (n2 << 1) + threadIdx.x; i < n; i += blockDim.x) {
    sum += v_col[i] * w[i];
  }

  for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);

  __shared__ double sdata[32];
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  if (lane == 0) sdata[warp] = sum;
  __syncthreads();

  if (threadIdx.x < 32) {
    int nwarps = (blockDim.x + 31) >> 5;
    sum = (threadIdx.x < nwarps) ? sdata[threadIdx.x] : 0.0;
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x == 0) h[col] = sum;
  }
}

__global__ void gemv_n_sub_kernel(const double* __restrict__ V, int n, int ncols, int lda,
                                  const double* __restrict__ h, double* __restrict__ w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double sum = 0.0;
    #pragma unroll 4
    for (int j = 0; j < ncols; j++) sum += V[i + (int64_t)j * lda] * h[j];
    w[i] -= sum;
  }
}

__global__ void row_normalize_kernel(double* __restrict__ Y, int n, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    double norm = 0.0;
    #pragma unroll
    for (int d = 0; d < 12; d++) {
      if (d < dim) {
        double val = Y[i + (size_t)d * n];
        norm += val * val;
      }
    }
    norm = sqrt(norm);
    if (norm > 1e-30) {
      double inv = 1.0 / norm;
      #pragma unroll
      for (int d = 0; d < 12; d++) {
        if (d < dim) Y[i + (size_t)d * n] *= inv;
      }
    }
  }
}





template <int BLOCK>
__global__ void argmax_abs_col0_kernel(const double* __restrict__ Y, int n, unsigned long long* __restrict__ out) {
  using BlockReduce = cub::BlockReduce<unsigned long long, BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;

  unsigned long long local = 0ull;
  for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) {
    float a = (float)fabs(Y[i]);
    unsigned long long p = (unsigned long long)__float_as_uint(a) << 32;
    p |= (unsigned long long)(uint32_t)i;
    if (p > local) local = p;
  }
  unsigned long long block_max = BlockReduce(temp).Reduce(local, ull_max{});
  if (threadIdx.x == 0) atomicMax((unsigned long long*)out, block_max);
}

__global__ void gather_centroid_kernel(const double* __restrict__ Y,
                                       int n,
                                       int dim,
                                       const unsigned long long* __restrict__ packed,
                                       double* __restrict__ centroids,
                                       int centroid_id) {
  int d = threadIdx.x;
  if (d >= dim) return;
  int idx = (int)(*packed & 0xffffffffull);
  centroids[centroid_id * dim + d] = Y[idx + (size_t)d * n];
}

template <int BLOCK>
__global__ void fps_update_argmax_kernel(const double* __restrict__ Y,
                                         int n,
                                         int dim,
                                         const double* __restrict__ centroid,
                                         double* __restrict__ min_dist,
                                         unsigned long long* __restrict__ out_packed) {
  using BlockReduce = cub::BlockReduce<unsigned long long, BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;

  unsigned long long local = 0ull;
  for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) {
    double dist = 0.0;
    #pragma unroll
    for (int d = 0; d < 12; d++) {
      if (d < dim) {
        double diff = Y[i + (size_t)d * n] - centroid[d];
        dist += diff * diff;
      }
    }
    double md = min_dist[i];
    if (dist < md) md = dist;
    min_dist[i] = md;
    float mdf = (float)md;
    unsigned long long p = (unsigned long long)__float_as_uint(mdf) << 32;
    p |= (unsigned long long)(uint32_t)i;
    if (p > local) local = p;
  }
  unsigned long long block_max = BlockReduce(temp).Reduce(local, ull_max{});
  if (threadIdx.x == 0) atomicMax((unsigned long long*)out_packed, block_max);
}

template <int BLOCK>
__global__ void init_min_dist_kernel(const double* __restrict__ Y,
                                     int n,
                                     int dim,
                                     const double* __restrict__ centroid,
                                     double* __restrict__ min_dist) {
  for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK) {
    double dist = 0.0;
    #pragma unroll
    for (int d = 0; d < 12; d++) {
      if (d < dim) {
        double diff = Y[i + (size_t)d * n] - centroid[d];
        dist += diff * diff;
      }
    }
    min_dist[i] = dist;
  }
}





__global__ void kmeans_assign_kernel(const double* __restrict__ data,
                                     const double* __restrict__ centroids,
                                     int32_t* __restrict__ assignments,
                                     int n,
                                     int k,
                                     int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  double x0 = (dim > 0) ? data[i] : 0.0;
  double x1 = (dim > 1) ? data[i + (size_t)1 * n] : 0.0;
  double x2 = (dim > 2) ? data[i + (size_t)2 * n] : 0.0;
  double x3 = (dim > 3) ? data[i + (size_t)3 * n] : 0.0;
  double x4 = (dim > 4) ? data[i + (size_t)4 * n] : 0.0;
  double x5 = (dim > 5) ? data[i + (size_t)5 * n] : 0.0;
  double x6 = (dim > 6) ? data[i + (size_t)6 * n] : 0.0;
  double x7 = (dim > 7) ? data[i + (size_t)7 * n] : 0.0;

  double best = 1e300;
  int best_c = 0;
  for (int c = 0; c < k; c++) {
    const double* cent = centroids + c * dim;
    double dist = 0.0;
    if (dim > 0) { double d0 = x0 - cent[0]; dist += d0 * d0; }
    if (dim > 1) { double d1 = x1 - cent[1]; dist += d1 * d1; }
    if (dim > 2) { double d2 = x2 - cent[2]; dist += d2 * d2; }
    if (dim > 3) { double d3 = x3 - cent[3]; dist += d3 * d3; }
    if (dim > 4) { double d4 = x4 - cent[4]; dist += d4 * d4; }
    if (dim > 5) { double d5 = x5 - cent[5]; dist += d5 * d5; }
    if (dim > 6) { double d6 = x6 - cent[6]; dist += d6 * d6; }
    if (dim > 7) { double d7 = x7 - cent[7]; dist += d7 * d7; }
    if (dist < best) {
      best = dist;
      best_c = c;
    }
  }
  assignments[i] = best_c;
}

__global__ void kmeans_assign_accum_kernel(const double* __restrict__ data,
                                          const double* __restrict__ centroids,
                                          const int32_t* __restrict__ old_assign,
                                          int32_t* __restrict__ new_assign,
                                          double* __restrict__ centroids_sum,
                                          int32_t* __restrict__ counts,
                                          int32_t* __restrict__ changes,
                                          int n,
                                          int k,
                                          int dim) {
  __shared__ double s_sum[96];
  __shared__ int s_cnt[12];

  for (int t = threadIdx.x; t < 96; t += blockDim.x) s_sum[t] = 0.0;
  for (int t = threadIdx.x; t < 12; t += blockDim.x) s_cnt[t] = 0;
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  int32_t best_c = 0;
  bool diff = false;
  double x0 = 0.0, x1 = 0.0, x2 = 0.0, x3 = 0.0, x4 = 0.0, x5 = 0.0, x6 = 0.0, x7 = 0.0;
  if (i < n) {
    x0 = (dim > 0) ? data[i] : 0.0;
    x1 = (dim > 1) ? data[i + (size_t)1 * n] : 0.0;
    x2 = (dim > 2) ? data[i + (size_t)2 * n] : 0.0;
    x3 = (dim > 3) ? data[i + (size_t)3 * n] : 0.0;
    x4 = (dim > 4) ? data[i + (size_t)4 * n] : 0.0;
    x5 = (dim > 5) ? data[i + (size_t)5 * n] : 0.0;
    x6 = (dim > 6) ? data[i + (size_t)6 * n] : 0.0;
    x7 = (dim > 7) ? data[i + (size_t)7 * n] : 0.0;

    double best = 1e300;
    for (int c = 0; c < k; c++) {
      const double* cent = centroids + c * dim;
      double dist = 0.0;
      if (dim > 0) { double d0 = x0 - cent[0]; dist += d0 * d0; }
      if (dim > 1) { double d1 = x1 - cent[1]; dist += d1 * d1; }
      if (dim > 2) { double d2 = x2 - cent[2]; dist += d2 * d2; }
      if (dim > 3) { double d3 = x3 - cent[3]; dist += d3 * d3; }
      if (dim > 4) { double d4 = x4 - cent[4]; dist += d4 * d4; }
      if (dim > 5) { double d5 = x5 - cent[5]; dist += d5 * d5; }
      if (dim > 6) { double d6 = x6 - cent[6]; dist += d6 * d6; }
      if (dim > 7) { double d7 = x7 - cent[7]; dist += d7 * d7; }
      if (dist < best) {
        best = dist;
        best_c = (int32_t)c;
      }
    }

    new_assign[i] = best_c;

    int32_t old = old_assign[i];
    diff = (old != best_c);

    atomicAdd(&s_cnt[best_c], 1);
    if (dim > 0) atomicAdd(&s_sum[best_c * dim + 0], x0);
    if (dim > 1) atomicAdd(&s_sum[best_c * dim + 1], x1);
    if (dim > 2) atomicAdd(&s_sum[best_c * dim + 2], x2);
    if (dim > 3) atomicAdd(&s_sum[best_c * dim + 3], x3);
    if (dim > 4) atomicAdd(&s_sum[best_c * dim + 4], x4);
    if (dim > 5) atomicAdd(&s_sum[best_c * dim + 5], x5);
    if (dim > 6) atomicAdd(&s_sum[best_c * dim + 6], x6);
    if (dim > 7) atomicAdd(&s_sum[best_c * dim + 7], x7);
  }

  unsigned mask = __ballot_sync(0xffffffff, diff);
  if ((threadIdx.x & 31) == 0) atomicAdd(changes, (int32_t)__popc(mask));

  __syncthreads();

  if (threadIdx.x < 96) {
    int denom = (dim > 0) ? dim : 1;
    int c = threadIdx.x / denom;
    int d = threadIdx.x - c * denom;
    if (c < k && d < dim) atomicAdd(&centroids_sum[c * dim + d], s_sum[c * dim + d]);
  }
  if (threadIdx.x < 12) {
    if (threadIdx.x < k) atomicAdd(&counts[threadIdx.x], s_cnt[threadIdx.x]);
  }
}

__global__ void kmeans_accumulate_shared_kernel(const double* __restrict__ data,
                                                const int32_t* __restrict__ assignments,
                                                double* __restrict__ centroids_sum,
                                                int32_t* __restrict__ counts,
                                                int n,
                                                int k,
                                                int dim) {
  __shared__ double s_sum[96];
  __shared__ int s_cnt[12];

  for (int t = threadIdx.x; t < 96; t += blockDim.x) s_sum[t] = 0.0;
  for (int t = threadIdx.x; t < 12; t += blockDim.x) s_cnt[t] = 0;
  __syncthreads();

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int c = assignments[i];
    atomicAdd(&s_cnt[c], 1);
    #pragma unroll
    for (int d = 0; d < 12; d++) {
      if (d < dim) atomicAdd(&s_sum[c * dim + d], data[i + (size_t)d * n]);
    }
  }
  __syncthreads();

  if (threadIdx.x < 96) {
    int c = threadIdx.x / dim;
    int d = threadIdx.x - c * dim;
    if (c < k && d < dim) atomicAdd(&centroids_sum[c * dim + d], s_sum[c * dim + d]);
  }
  if (threadIdx.x < 12) {
    if (threadIdx.x < k) atomicAdd(&counts[threadIdx.x], s_cnt[threadIdx.x]);
  }
}

__global__ void kmeans_divide_kernel(double* __restrict__ centroids,
                                     const double* __restrict__ centroids_sum,
                                     const int32_t* __restrict__ counts,
                                     int k,
                                     int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < k * dim) {
    int c = idx / dim;
    int cnt = counts[c];
    double v = centroids_sum[idx];
    centroids[idx] = (cnt > 0) ? v * (1.0 / (double)cnt) : 0.0;
  }
}

__global__ void count_changes_ballot_kernel(const int32_t* __restrict__ old_assign,
                                            const int32_t* __restrict__ new_assign,
                                            int32_t* __restrict__ count,
                                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x & 31;
  unsigned mask = __ballot_sync(0xffffffff, (i < n) && (old_assign[i] != new_assign[i]));
  if (lane == 0) atomicAdd(count, __popc(mask));
}

__global__ void kmeans_cost_kernel(const double* __restrict__ data,
                                   const double* __restrict__ centroids,
                                   const int32_t* __restrict__ assignments,
                                   double* __restrict__ costs,
                                   int n, int k, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int c = assignments[i];
  const double* cent = centroids + c * dim;
  double dist = 0.0;
  for (int d = 0; d < dim; d++) {
    double diff = data[i + (size_t)d * n] - cent[d];
    dist += diff * diff;
  }
  costs[i] = dist;
}

__global__ void gather_row_kernel(const double* __restrict__ Y,
                                  double* __restrict__ centroid,
                                  int n, int dim, int row_idx) {
  int d = threadIdx.x;
  if (d < dim) {
    centroid[d] = Y[row_idx + (size_t)d * n];
  }
}





static void launch_compute_degrees_segmented(const int32_t* offsets, const double* weights, double* degrees,
                                             int n, const int32_t* h_seg) {
  int seg1 = h_seg[1], seg2 = h_seg[2];
  int n_mid = seg2 - seg1;
  int n_tail = n - seg2;
  int tail_units = (n_tail + 3) >> 2;
  int total_units = n_mid + tail_units;
  int threads = 256;
  int warps_per_block = threads / 32;
  int blocks_midlow = (total_units + warps_per_block - 1) / warps_per_block;
  int grid = seg1 + blocks_midlow;
  degrees_all_fused_kernel<<<grid, threads>>>(offsets, weights, degrees, seg1, seg2, n);
}

static void launch_spmv_mod_segmented(const int32_t* offsets, const int32_t* indices, const double* values,
                                      const double* degrees, double inv_2m, const double* x, double* y,
                                      int n, const double* d_dot_val, const int32_t* h_seg) {
  int seg1 = h_seg[1], seg2 = h_seg[2];
  int n_mid = seg2 - seg1;
  int n_tail = n - seg2;
  int tail_units = (n_tail + 3) >> 2;
  int total_units = n_mid + tail_units;
  int threads = 256;
  int warps_per_block = threads / 32;
  int blocks_midlow = (total_units + warps_per_block - 1) / warps_per_block;
  int grid = seg1 + blocks_midlow;
  spmv_mod_all_fused_kernel<<<grid, threads>>>(offsets, indices, values, degrees, inv_2m, x, y, seg1, seg2, n, d_dot_val);
}

static void launch_neg_axpy(const double* alpha, const double* x, double* y, int n) {
  neg_axpy_kernel<<<(n + 255) / 256, 256>>>(alpha, x, y, n);
}

static void launch_neg_axpy2(const double* alpha, const double* x1, const double* beta, const double* x2, double* y, int n) {
  neg_axpy2_kernel<<<(n + 255) / 256, 256>>>(alpha, x1, beta, x2, y, n);
}

static void launch_inv_scale(const double* beta, double* x, int n) {
  inv_scale_kernel<<<(n + 255) / 256, 256>>>(beta, x, n);
}

static void launch_init_lanczos(double* v, int n) {
  init_lanczos_kernel<<<(n + 255) / 256, 256>>>(v, n);
}

static void launch_gemv_t(const double* V, int n, int ncols, int lda, const double* w, double* h) {
  gemv_t_kernel<<<ncols, 256>>>(V, n, ncols, lda, w, h);
}

static void launch_gemv_n_sub(const double* V, int n, int ncols, int lda, const double* h, double* w) {
  gemv_n_sub_kernel<<<(n + 255) / 256, 256>>>(V, n, ncols, lda, h, w);
}

static void launch_row_normalize(double* Y, int n, int dim) {
  row_normalize_kernel<<<(n + 255) / 256, 256>>>(Y, n, dim);
}

static void launch_argmax_abs_col0(const double* Y, int n, unsigned long long* out) {
  cudaMemset(out, 0, sizeof(unsigned long long));
  int blocks = ((n + 255) / 256 < 256) ? ((n + 255) / 256) : 256;
  argmax_abs_col0_kernel<256><<<blocks, 256>>>(Y, n, out);
}

static void launch_gather_centroid(const double* Y, int n, int dim, const unsigned long long* packed, double* centroids, int cid) {
  gather_centroid_kernel<<<1, 16>>>(Y, n, dim, packed, centroids, cid);
}

static void launch_init_min_dist(const double* Y, int n, int dim, const double* centroid, double* min_dist) {
  int blocks = ((n + 255) / 256 < 256) ? ((n + 255) / 256) : 256;
  init_min_dist_kernel<256><<<blocks, 256>>>(Y, n, dim, centroid, min_dist);
}

static void launch_fps_update_argmax(const double* Y, int n, int dim, const double* centroid, double* min_dist, unsigned long long* out) {
  cudaMemset(out, 0, sizeof(unsigned long long));
  int blocks = ((n + 255) / 256 < 256) ? ((n + 255) / 256) : 256;
  fps_update_argmax_kernel<256><<<blocks, 256>>>(Y, n, dim, centroid, min_dist, out);
}

static void launch_kmeans_assign_accum(const double* data, const double* centroids, const int32_t* old_assign,
                                       int32_t* new_assign, double* centroids_sum, int32_t* counts,
                                       int32_t* changes, int n, int k, int dim) {
  kmeans_assign_accum_kernel<<<(n + 255) / 256, 256>>>(data, centroids, old_assign, new_assign, centroids_sum, counts, changes, n, k, dim);
}

static void launch_kmeans_assign(const double* data, const double* centroids, int32_t* assignments, int n, int k, int dim) {
  kmeans_assign_kernel<<<(n + 255) / 256, 256>>>(data, centroids, assignments, n, k, dim);
}

static void launch_kmeans_accumulate_shared(const double* data, const int32_t* assignments, double* centroids_sum,
                                            int32_t* counts, int n, int k, int dim) {
  kmeans_accumulate_shared_kernel<<<(n + 255) / 256, 256>>>(data, assignments, centroids_sum, counts, n, k, dim);
}

static void launch_kmeans_divide(double* centroids, const double* centroids_sum, const int32_t* counts, int k, int dim) {
  kmeans_divide_kernel<<<(k * dim + 127) / 128, 128>>>(centroids, centroids_sum, counts, k, dim);
}

static void launch_count_changes_ballot(const int32_t* old_assign, const int32_t* new_assign, int32_t* count, int n) {
  count_changes_ballot_kernel<<<(n + 255) / 256, 256>>>(old_assign, new_assign, count, n);
}

static void launch_kmeans_cost(const double* data, const double* centroids, const int32_t* assignments,
                               double* costs, int n, int k, int dim) {
  kmeans_cost_kernel<<<(n + 255) / 256, 256>>>(data, centroids, assignments, costs, n, k, dim);
}

static void launch_gather_row(const double* Y, double* centroid, int n, int dim, int row_idx) {
  gather_row_kernel<<<1, 16>>>(Y, centroid, n, dim, row_idx);
}





static void solve_tridiagonal_eigen(int m,
                                    const double* alpha_arr,
                                    const double* beta_arr,
                                    std::vector<double>& eigenvalues,
                                    std::vector<double>& eigenvectors) {
  eigenvalues.resize(m);
  eigenvectors.resize((size_t)m * m);

  std::vector<double> d(m), e(m, 0.0);
  for (int i = 0; i < m; i++) d[i] = alpha_arr[i];
  for (int i = 0; i < m - 1; i++) e[i] = beta_arr[i];

  std::fill(eigenvectors.begin(), eigenvectors.end(), 0.0);
  for (int i = 0; i < m; i++) eigenvectors[i + (size_t)i * m] = 1.0;

  const double eps = 1e-15;
  for (int l = 0; l < m; l++) {
    int iter = 0;
    int mm;
    do {
      for (mm = l; mm < m - 1; mm++) {
        double dd = std::abs(d[mm]) + std::abs(d[mm + 1]);
        if (std::abs(e[mm]) <= eps * dd) break;
      }
      if (mm != l) {
        if (++iter > 300) break;
        double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
        double r = std::sqrt(g * g + 1.0);
        g = d[mm] - d[l] + e[l] / (g + (g >= 0.0 ? r : -r));
        double s_val = 1.0, c_val = 1.0, p = 0.0;
        int i;
        for (i = mm - 1; i >= l; i--) {
          double f = s_val * e[i], b = c_val * e[i];
          if (std::abs(f) >= std::abs(g)) {
            c_val = g / f;
            r = std::sqrt(c_val * c_val + 1.0);
            e[i + 1] = f * r;
            s_val = 1.0 / r;
            c_val *= s_val;
          } else {
            s_val = f / g;
            r = std::sqrt(s_val * s_val + 1.0);
            e[i + 1] = g * r;
            c_val = 1.0 / r;
            s_val *= c_val;
          }
          g = d[i + 1] - p;
          r = (d[i] - g) * s_val + 2.0 * c_val * b;
          p = s_val * r;
          d[i + 1] = g + p;
          g = c_val * r - b;
          for (int k = 0; k < m; k++) {
            double fk = eigenvectors[k + (size_t)(i + 1) * m];
            eigenvectors[k + (size_t)(i + 1) * m] = s_val * eigenvectors[k + (size_t)i * m] + c_val * fk;
            eigenvectors[k + (size_t)i * m] = c_val * eigenvectors[k + (size_t)i * m] - s_val * fk;
          }
        }
        if (r < eps && i >= l) continue;
        d[l] -= p;
        e[l] = g;
        e[mm] = 0.0;
      }
    } while (mm != l);
  }

  for (int i = 0; i < m; i++) eigenvalues[i] = d[i];

  std::vector<int> perm(m);
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(), [&](int a, int b) { return eigenvalues[a] > eigenvalues[b]; });
  std::vector<double> se(m), sv((size_t)m * m);
  for (int i = 0; i < m; i++) {
    se[i] = eigenvalues[perm[i]];
    for (int k = 0; k < m; k++) sv[k + (size_t)i * m] = eigenvectors[k + (size_t)perm[i] * m];
  }
  eigenvalues = se;
  eigenvectors = sv;
}





struct Cache : Cacheable {
  cublasHandle_t cublas = nullptr;
  double* d_one = nullptr;
  double* d_neg_one = nullptr;
  double* d_zero = nullptr;

  
  double* sum_buf = nullptr;
  double* dot_buf = nullptr;
  double* nrm_buf = nullptr;
  unsigned long long* packed_buf = nullptr;
  int32_t* chg_buf = nullptr;

  
  double* deg = nullptr;         int64_t deg_cap = 0;
  double* min_dist = nullptr;    int64_t min_dist_cap = 0;
  int32_t* asgn0 = nullptr;     int64_t asgn0_cap = 0;
  int32_t* asgn1 = nullptr;     int64_t asgn1_cap = 0;
  double* costs = nullptr;       int64_t costs_cap = 0;
  int32_t* best_asgn = nullptr; int64_t best_asgn_cap = 0;

  
  double* V = nullptr;           int64_t V_cap = 0;

  
  double* alpha_buf = nullptr;   int64_t alpha_cap = 0;
  double* beta_buf = nullptr;    int64_t beta_cap = 0;
  double* h_buf = nullptr;       int64_t h_cap = 0;

  
  double* Y = nullptr;           int64_t Y_cap = 0;

  
  double* S = nullptr;           int64_t S_cap = 0;

  
  double* cent = nullptr;        int64_t cent_cap = 0;
  double* cent_sum = nullptr;    int64_t cent_sum_cap = 0;

  
  int32_t* cnt = nullptr;        int64_t cnt_cap = 0;

  Cache() {
    cublasCreate(&cublas);
    cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);

    cudaMalloc(&d_one, sizeof(double));
    cudaMalloc(&d_neg_one, sizeof(double));
    cudaMalloc(&d_zero, sizeof(double));
    double h = 1.0;
    cudaMemcpy(d_one, &h, sizeof(double), cudaMemcpyHostToDevice);
    h = -1.0;
    cudaMemcpy(d_neg_one, &h, sizeof(double), cudaMemcpyHostToDevice);
    h = 0.0;
    cudaMemcpy(d_zero, &h, sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&sum_buf, sizeof(double));
    cudaMalloc(&dot_buf, sizeof(double));
    cudaMalloc(&nrm_buf, sizeof(double));
    cudaMalloc(&packed_buf, sizeof(unsigned long long));
    cudaMalloc(&chg_buf, sizeof(int32_t));
  }

  ~Cache() override {
    if (cublas) cublasDestroy(cublas);
    if (d_one) cudaFree(d_one);
    if (d_neg_one) cudaFree(d_neg_one);
    if (d_zero) cudaFree(d_zero);
    if (sum_buf) cudaFree(sum_buf);
    if (dot_buf) cudaFree(dot_buf);
    if (nrm_buf) cudaFree(nrm_buf);
    if (packed_buf) cudaFree(packed_buf);
    if (chg_buf) cudaFree(chg_buf);
    if (deg) cudaFree(deg);
    if (min_dist) cudaFree(min_dist);
    if (asgn0) cudaFree(asgn0);
    if (asgn1) cudaFree(asgn1);
    if (costs) cudaFree(costs);
    if (best_asgn) cudaFree(best_asgn);
    if (V) cudaFree(V);
    if (alpha_buf) cudaFree(alpha_buf);
    if (beta_buf) cudaFree(beta_buf);
    if (h_buf) cudaFree(h_buf);
    if (Y) cudaFree(Y);
    if (S) cudaFree(S);
    if (cent) cudaFree(cent);
    if (cent_sum) cudaFree(cent_sum);
    if (cnt) cudaFree(cnt);
  }

  void ensure(int32_t n, int maxL, int num_eigs, int num_clusters) {
    int64_t need;

    need = (int64_t)n;
    if (deg_cap < need) { if (deg) cudaFree(deg); cudaMalloc(&deg, need * sizeof(double)); deg_cap = need; }
    if (min_dist_cap < need) { if (min_dist) cudaFree(min_dist); cudaMalloc(&min_dist, need * sizeof(double)); min_dist_cap = need; }
    if (asgn0_cap < need) { if (asgn0) cudaFree(asgn0); cudaMalloc(&asgn0, need * sizeof(int32_t)); asgn0_cap = need; }
    if (asgn1_cap < need) { if (asgn1) cudaFree(asgn1); cudaMalloc(&asgn1, need * sizeof(int32_t)); asgn1_cap = need; }
    if (costs_cap < need) { if (costs) cudaFree(costs); cudaMalloc(&costs, need * sizeof(double)); costs_cap = need; }
    if (best_asgn_cap < need) { if (best_asgn) cudaFree(best_asgn); cudaMalloc(&best_asgn, need * sizeof(int32_t)); best_asgn_cap = need; }

    need = (int64_t)n * (maxL + 1);
    if (V_cap < need) { if (V) cudaFree(V); cudaMalloc(&V, need * sizeof(double)); V_cap = need; }

    need = (int64_t)maxL;
    if (alpha_cap < need) { if (alpha_buf) cudaFree(alpha_buf); cudaMalloc(&alpha_buf, need * sizeof(double)); alpha_cap = need; }
    if (beta_cap < need) { if (beta_buf) cudaFree(beta_buf); cudaMalloc(&beta_buf, need * sizeof(double)); beta_cap = need; }
    if (h_cap < need) { if (h_buf) cudaFree(h_buf); cudaMalloc(&h_buf, need * sizeof(double)); h_cap = need; }

    need = (int64_t)n * num_eigs;
    if (Y_cap < need) { if (Y) cudaFree(Y); cudaMalloc(&Y, need * sizeof(double)); Y_cap = need; }

    need = (int64_t)maxL * num_eigs;
    if (S_cap < need) { if (S) cudaFree(S); cudaMalloc(&S, need * sizeof(double)); S_cap = need; }

    need = (int64_t)num_clusters * num_eigs;
    if (cent_cap < need) { if (cent) cudaFree(cent); cudaMalloc(&cent, need * sizeof(double)); cent_cap = need; }
    if (cent_sum_cap < need) { if (cent_sum) cudaFree(cent_sum); cudaMalloc(&cent_sum, need * sizeof(double)); cent_sum_cap = need; }

    need = (int64_t)num_clusters;
    if (cnt_cap < need) { if (cnt) cudaFree(cnt); cudaMalloc(&cnt, need * sizeof(int32_t)); cnt_cap = need; }
  }
};

}  

void spectral_modularity_maximization_seg(const graph32_t& graph,
                                          const double* edge_weights,
                                          int32_t num_clusters,
                                          int32_t num_eigenvectors,
                                          double evs_tolerance,
                                          int32_t evs_max_iter,
                                          double kmean_tolerance,
                                          int32_t kmean_max_iter,
                                          int32_t* clustering) {
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);

  const int32_t* d_offsets = graph.offsets;
  const int32_t* d_indices = graph.indices;
  int32_t n = graph.number_of_vertices;
  const double* d_weights = edge_weights;

  const auto& seg = graph.segment_offsets.value();
  int32_t h_seg[5] = {seg[0], seg[1], seg[2], seg[3], seg[4]};

  int32_t num_eigs = num_eigenvectors;
  double evs_tol = evs_tolerance;
  double kmean_tol = kmean_tolerance;

  int maxL = std::max<int>(evs_max_iter, 2 * num_eigs + 1);
  maxL = std::min<int>(maxL, n - 1);

  cache.ensure(n, maxL, num_eigs, num_clusters);

  
  double* d_deg = cache.deg;
  launch_compute_degrees_segmented(d_offsets, d_weights, d_deg, n, h_seg);

  
  cublasDasum(cache.cublas, n, d_deg, 1, cache.sum_buf);
  double total_deg;
  cudaMemcpy(&total_deg, cache.sum_buf, sizeof(double), cudaMemcpyDeviceToHost);
  double inv_2m = (total_deg > 1e-20) ? (1.0 / total_deg) : 0.0;

  
  double* d_V = cache.V;
  double* d_alpha = cache.alpha_buf;
  double* d_beta = cache.beta_buf;
  double* d_dot = cache.dot_buf;
  double* d_nrm = cache.nrm_buf;
  double* d_h = cache.h_buf;

  launch_init_lanczos(d_V, n);
  cublasDnrm2(cache.cublas, n, d_V, 1, d_nrm);
  launch_inv_scale(d_nrm, d_V, n);

  int actual_iter = 0;
  bool converged = false;
  int check_every = (evs_max_iter >= 150) ? 10 : 5;
  int min_steps = std::max(2 * num_eigs + 10, 30);
  std::vector<double> h_alpha(maxL, 0.0), h_beta(maxL, 0.0);
  std::vector<double> prev_ritz(num_eigs, -1e300);

  for (int j = 0; j < maxL; j++) {
    double* vj = d_V + (int64_t)j * n;
    double* w = d_V + (int64_t)(j + 1) * n;

    cublasDdot(cache.cublas, n, d_deg, 1, vj, 1, d_dot);
    launch_spmv_mod_segmented(d_offsets, d_indices, d_weights, d_deg, inv_2m, vj, w, n, d_dot, h_seg);

    cublasDdot(cache.cublas, n, vj, 1, w, 1, d_alpha + j);

    if (j > 0) {
      launch_neg_axpy2(d_alpha + j, vj, d_beta + (j - 1), d_V + (int64_t)(j - 1) * n, w, n);
    } else {
      launch_neg_axpy(d_alpha + j, vj, w, n);
    }

    
    int ncols = j + 1;
    for (int pass = 0; pass < 2; pass++) {
      launch_gemv_t(d_V, n, ncols, n, w, d_h);
      launch_gemv_n_sub(d_V, n, ncols, n, d_h, w);
    }

    cublasDnrm2(cache.cublas, n, w, 1, d_beta + j);
    actual_iter = j + 1;
    launch_inv_scale(d_beta + j, w, n);

    if (actual_iter >= min_steps && (actual_iter % check_every) == 0) {
      cudaMemcpy(h_alpha.data(), d_alpha, (size_t)actual_iter * sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_beta.data(), d_beta, (size_t)actual_iter * sizeof(double), cudaMemcpyDeviceToHost);

      double beta_m = h_beta[j];
      if (beta_m < 1e-14) {
        converged = true;
        break;
      }

      std::vector<double> ev, evec;
      solve_tridiagonal_eigen(actual_iter, h_alpha.data(), h_beta.data(), ev, evec);

      bool conv_by_eigenvalue = true;
      for (int i = 0; i < num_eigs; i++) {
        double cur = ev[i];
        double prev = prev_ritz[i];
        double scale = std::abs(cur) > 1.0 ? std::abs(cur) : 1.0;
        if (std::abs(cur - prev) > evs_tol * scale) {
          conv_by_eigenvalue = false;
        }
        prev_ritz[i] = cur;
      }

      bool conv_by_residual = true;
      for (int i = 0; i < num_eigs; i++) {
        double last_comp = evec[(actual_iter - 1) + (size_t)i * actual_iter];
        double residual = std::abs(beta_m * last_comp);
        double eigenval = std::abs(ev[i]);
        if (eigenval < 1e-10) eigenval = 1e-10;
        if (residual > evs_tol * eigenval) {
          conv_by_residual = false;
          break;
        }
      }

      if (conv_by_eigenvalue || conv_by_residual) {
        converged = true;
        break;
      }
    }
  }

  int m = actual_iter;
  if (!converged) {
    cudaMemcpy(h_alpha.data(), d_alpha, (size_t)m * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_beta.data(), d_beta, (size_t)m * sizeof(double), cudaMemcpyDeviceToHost);
  }

  std::vector<double> evals, evecs;
  solve_tridiagonal_eigen(m, h_alpha.data(), h_beta.data(), evals, evecs);

  std::vector<double> S_host((size_t)m * num_eigs);
  for (int c = 0; c < num_eigs; c++) {
    for (int r = 0; r < m; r++) S_host[r + (size_t)c * m] = evecs[r + (size_t)c * m];
  }

  double* d_S = cache.S;
  cudaMemcpy(d_S, S_host.data(), (size_t)m * num_eigs * sizeof(double), cudaMemcpyHostToDevice);

  double* d_Y = cache.Y;
  cublasDgemm(cache.cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, num_eigs, m, cache.d_one, d_V, n, d_S, m, cache.d_zero,
              d_Y, n);

  launch_row_normalize(d_Y, n, num_eigs);

  
  
  
  double* d_cent = cache.cent;
  double* d_min_dist = cache.min_dist;
  unsigned long long* d_packed = cache.packed_buf;

  int32_t* d_asgn0 = cache.asgn0;
  int32_t* d_asgn1 = cache.asgn1;

  double* d_cent_sum = cache.cent_sum;
  int32_t* d_cnt = cache.cnt;
  int32_t* d_chg = cache.chg_buf;

  double* d_costs = cache.costs;
  int32_t* d_best_asgn = cache.best_asgn;

  int32_t change_threshold = (int32_t)ceil(kmean_tol * (double)n);

  double best_total_cost = 1e300;

  const int NUM_RESTARTS = 3;
  int first_seeds[NUM_RESTARTS];
  first_seeds[0] = -1;
  first_seeds[1] = 0;
  first_seeds[2] = n / 2;

  for (int restart = 0; restart < NUM_RESTARTS; restart++) {
    if (first_seeds[restart] < 0) {
      launch_argmax_abs_col0(d_Y, n, d_packed);
      launch_gather_centroid(d_Y, n, num_eigs, d_packed, d_cent, 0);
    } else {
      launch_gather_row(d_Y, d_cent, n, num_eigs, first_seeds[restart]);
    }

    launch_init_min_dist(d_Y, n, num_eigs, d_cent + 0 * num_eigs, d_min_dist);

    for (int c = 1; c < num_clusters; c++) {
      const double* last_centroid = d_cent + (c - 1) * num_eigs;
      launch_fps_update_argmax(d_Y, n, num_eigs, last_centroid, d_min_dist, d_packed);
      launch_gather_centroid(d_Y, n, num_eigs, d_packed, d_cent, c);
    }

    int32_t* d_asgn_curr = d_asgn0;
    int32_t* d_asgn_next = d_asgn1;

    cudaMemset(d_asgn0, 0, (size_t)n * sizeof(int32_t));
    cudaMemset(d_asgn1, 0, (size_t)n * sizeof(int32_t));

    for (int it = 0; it < kmean_max_iter; it++) {
      cudaMemset(d_cent_sum, 0, (size_t)num_clusters * num_eigs * sizeof(double));
      cudaMemset(d_cnt, 0, (size_t)num_clusters * sizeof(int32_t));
      cudaMemset(d_chg, 0, sizeof(int32_t));

      if (num_clusters <= 8 && num_eigs <= 8) {
        launch_kmeans_assign_accum(d_Y, d_cent, d_asgn_curr, d_asgn_next, d_cent_sum, d_cnt, d_chg, n, num_clusters,
                                   num_eigs);
      } else {
        launch_kmeans_assign(d_Y, d_cent, d_asgn_next, n, num_clusters, num_eigs);
        if (it >= 2 && (it % 2) == 0) {
          launch_count_changes_ballot(d_asgn_curr, d_asgn_next, d_chg, n);
        }
        launch_kmeans_accumulate_shared(d_Y, d_asgn_next, d_cent_sum, d_cnt, n, num_clusters, num_eigs);
      }

      launch_kmeans_divide(d_cent, d_cent_sum, d_cnt, num_clusters, num_eigs);

      if (it >= 2 && (it % 2) == 0) {
        int32_t changes;
        cudaMemcpy(&changes, d_chg, sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (changes <= change_threshold) {
          d_asgn_curr = d_asgn_next;
          break;
        }
      }

      int32_t* tmp = d_asgn_curr;
      d_asgn_curr = d_asgn_next;
      d_asgn_next = tmp;
    }

    
    launch_kmeans_cost(d_Y, d_cent, d_asgn_curr, d_costs, n, num_clusters, num_eigs);
    cublasSetPointerMode(cache.cublas, CUBLAS_POINTER_MODE_HOST);
    double total_cost;
    cublasDasum(cache.cublas, n, d_costs, 1, &total_cost);
    cublasSetPointerMode(cache.cublas, CUBLAS_POINTER_MODE_DEVICE);

    if (total_cost < best_total_cost) {
      best_total_cost = total_cost;
      cudaMemcpy(d_best_asgn, d_asgn_curr, (size_t)n * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    }
  }

  cudaMemcpy(clustering, d_best_asgn, (size_t)n * sizeof(int32_t), cudaMemcpyDeviceToDevice);
}

}  
