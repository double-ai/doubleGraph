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
#include <cusparse.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>
#include <math_constants.h>

namespace aai {

namespace {

struct Cache : Cacheable {
  cublasHandle_t cublas_h = nullptr;
  cusparseHandle_t cusparse_h = nullptr;

  Cache() {
    cublasCreate(&cublas_h);
    cusparseCreate(&cusparse_h);
  }

  ~Cache() override {
    if (cusparse_h) {
      cusparseDestroy(cusparse_h);
      cusparse_h = nullptr;
    }
    if (cublas_h) {
      cublasDestroy(cublas_h);
      cublas_h = nullptr;
    }
  }
};




static void tridiag_ql(int n, float* d, float* e, float* z) {
  for (int i = 0; i < n * n; i++) z[i] = 0.0f;
  for (int i = 0; i < n; i++) z[i + i * n] = 1.0f;
  if (n <= 1) return;

  for (int l = 0; l < n; l++) {
    int iter = 0;
    int m;
    do {
      for (m = l; m < n - 1; m++) {
        float dd = fabsf(d[m]) + fabsf(d[m + 1]);
        if (fabsf(e[m]) + dd == dd) break;
      }
      if (m != l) {
        if (++iter > 128) return;
        float g = (d[l + 1] - d[l]) / (2.0f * e[l]);
        float r = sqrtf(g * g + 1.0f);
        g = d[m] - d[l] + e[l] / (g + copysignf(r, g));
        float s = 1.0f, c = 1.0f, p = 0.0f;
        for (int i = m - 1; i >= l; i--) {
          float f = s * e[i];
          float b = c * e[i];
          if (fabsf(f) >= fabsf(g)) {
            c = g / f;
            r = sqrtf(c * c + 1.0f);
            e[i + 1] = f * r;
            s = 1.0f / r;
            c *= s;
          } else {
            s = f / g;
            r = sqrtf(s * s + 1.0f);
            e[i + 1] = g * r;
            c = 1.0f / r;
            s *= c;
          }
          g = d[i + 1] - p;
          r = (d[i] - g) * s + 2.0f * c * b;
          p = s * r;
          d[i + 1] = g + p;
          g = c * r - b;
          for (int k = 0; k < n; k++) {
            float t = z[k + (i + 1) * n];
            z[k + (i + 1) * n] = s * z[k + i * n] + c * t;
            z[k + i * n] = c * z[k + i * n] - s * t;
          }
        }
        d[l] -= p;
        e[l] = g;
        e[m] = 0.0f;
      }
    } while (m != l);
  }
}





static __device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__global__ void compute_degrees_kernel(const int32_t* __restrict__ offsets,
                                      const float* __restrict__ weights,
                                      float* __restrict__ degrees,
                                      int32_t n) {
  constexpr int WARP = 32;
  int thread = threadIdx.x;
  int lane = thread & 31;
  int warp_id = (blockIdx.x * blockDim.x + thread) >> 5;
  if (warp_id >= n) return;

  int start = __ldg(offsets + warp_id);
  int end = __ldg(offsets + warp_id + 1);
  float sum = 0.0f;
  for (int e = start + lane; e < end; e += WARP) {
    sum += __ldg(weights + e);
  }
  sum = warp_sum(sum);
  if (lane == 0) degrees[warp_id] = sum;
}

__global__ void reduce_sum_stage1_kernel(const float* __restrict__ x,
                                        float* __restrict__ partial,
                                        int32_t n) {
  float sum = 0.0f;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) sum += x[i];

  __shared__ float s[256];
  s[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) s[threadIdx.x] += s[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) partial[blockIdx.x] = s[0];
}

__global__ void reduce_sum_stage2_kernel(const float* __restrict__ partial,
                                        float* __restrict__ out,
                                        int32_t num_blocks) {
  float sum = 0.0f;
  for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) sum += partial[i];

  __shared__ float s[256];
  s[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) s[threadIdx.x] += s[threadIdx.x + offset];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[0] = s[0];
}

__global__ void modularity_correction_kernel(float* __restrict__ w,
                                            const float* __restrict__ degrees,
                                            const float* __restrict__ gamma,
                                            const float* __restrict__ two_m,
                                            int32_t n) {
  float tm = two_m[0];
  float scale = (fabsf(tm) > 1e-20f) ? (gamma[0] / tm) : 0.0f;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) w[tid] -= scale * degrees[tid];
}

__global__ void lanczos_recurrence_kernel(float* __restrict__ w,
                                         const float* __restrict__ v_j,
                                         const float* __restrict__ v_jm1,
                                         const float* __restrict__ alpha_ptr,
                                         const float* __restrict__ beta_ptr,
                                         int32_t n,
                                         bool has_prev) {
  float a = alpha_ptr[0];
  float b = has_prev ? beta_ptr[0] : 0.0f;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    float val = w[tid] - a * v_j[tid];
    if (has_prev) val -= b * v_jm1[tid];
    w[tid] = val;
  }
}

__global__ void normalize_and_copy_kernel(const float* __restrict__ w,
                                         float* __restrict__ v_next,
                                         const float* __restrict__ beta_ptr,
                                         int32_t n) {
  float b = beta_ptr[0];
  float inv = (fabsf(b) > 1e-20f) ? (1.0f / b) : 0.0f;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) v_next[tid] = w[tid] * inv;
}

__global__ void fill_kernel(float* __restrict__ v, int32_t n, float val) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) v[tid] = val;
}

__global__ void col_stats_stage1_small_kernel(const float* __restrict__ Y,
                                             float* __restrict__ partial_sum,
                                             float* __restrict__ partial_sumsq,
                                             int32_t n,
                                             int32_t dim) {
  int tid = threadIdx.x;
  int32_t grid = (int32_t)gridDim.x;

  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f, s4 = 0.f, s5 = 0.f, s6 = 0.f, s7 = 0.f;
  float q0 = 0.f, q1 = 0.f, q2 = 0.f, q3 = 0.f, q4 = 0.f, q5 = 0.f, q6 = 0.f, q7 = 0.f;

  for (int32_t i = (int32_t)blockIdx.x * (int32_t)blockDim.x + tid; i < n; i += (int32_t)blockDim.x * grid) {
    if (dim > 0) {
      float v = Y[(int64_t)0 * n + i];
      s0 += v;
      q0 += v * v;
    }
    if (dim > 1) {
      float v = Y[(int64_t)1 * n + i];
      s1 += v;
      q1 += v * v;
    }
    if (dim > 2) {
      float v = Y[(int64_t)2 * n + i];
      s2 += v;
      q2 += v * v;
    }
    if (dim > 3) {
      float v = Y[(int64_t)3 * n + i];
      s3 += v;
      q3 += v * v;
    }
    if (dim > 4) {
      float v = Y[(int64_t)4 * n + i];
      s4 += v;
      q4 += v * v;
    }
    if (dim > 5) {
      float v = Y[(int64_t)5 * n + i];
      s5 += v;
      q5 += v * v;
    }
    if (dim > 6) {
      float v = Y[(int64_t)6 * n + i];
      s6 += v;
      q6 += v * v;
    }
    if (dim > 7) {
      float v = Y[(int64_t)7 * n + i];
      s7 += v;
      q7 += v * v;
    }
  }

  __shared__ float sh_sum[8][256];
  __shared__ float sh_sumsq[8][256];
  if (dim > 0) {
    sh_sum[0][tid] = s0;
    sh_sumsq[0][tid] = q0;
  }
  if (dim > 1) {
    sh_sum[1][tid] = s1;
    sh_sumsq[1][tid] = q1;
  }
  if (dim > 2) {
    sh_sum[2][tid] = s2;
    sh_sumsq[2][tid] = q2;
  }
  if (dim > 3) {
    sh_sum[3][tid] = s3;
    sh_sumsq[3][tid] = q3;
  }
  if (dim > 4) {
    sh_sum[4][tid] = s4;
    sh_sumsq[4][tid] = q4;
  }
  if (dim > 5) {
    sh_sum[5][tid] = s5;
    sh_sumsq[5][tid] = q5;
  }
  if (dim > 6) {
    sh_sum[6][tid] = s6;
    sh_sumsq[6][tid] = q6;
  }
  if (dim > 7) {
    sh_sum[7][tid] = s7;
    sh_sumsq[7][tid] = q7;
  }
  __syncthreads();

  for (int offset = 128; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (dim > 0) {
        sh_sum[0][tid] += sh_sum[0][tid + offset];
        sh_sumsq[0][tid] += sh_sumsq[0][tid + offset];
      }
      if (dim > 1) {
        sh_sum[1][tid] += sh_sum[1][tid + offset];
        sh_sumsq[1][tid] += sh_sumsq[1][tid + offset];
      }
      if (dim > 2) {
        sh_sum[2][tid] += sh_sum[2][tid + offset];
        sh_sumsq[2][tid] += sh_sumsq[2][tid + offset];
      }
      if (dim > 3) {
        sh_sum[3][tid] += sh_sum[3][tid + offset];
        sh_sumsq[3][tid] += sh_sumsq[3][tid + offset];
      }
      if (dim > 4) {
        sh_sum[4][tid] += sh_sum[4][tid + offset];
        sh_sumsq[4][tid] += sh_sumsq[4][tid + offset];
      }
      if (dim > 5) {
        sh_sum[5][tid] += sh_sum[5][tid + offset];
        sh_sumsq[5][tid] += sh_sumsq[5][tid + offset];
      }
      if (dim > 6) {
        sh_sum[6][tid] += sh_sum[6][tid + offset];
        sh_sumsq[6][tid] += sh_sumsq[6][tid + offset];
      }
      if (dim > 7) {
        sh_sum[7][tid] += sh_sum[7][tid + offset];
        sh_sumsq[7][tid] += sh_sumsq[7][tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    int64_t base = (int64_t)blockIdx.x * dim;
    if (dim > 0) {
      partial_sum[base + 0] = sh_sum[0][0];
      partial_sumsq[base + 0] = sh_sumsq[0][0];
    }
    if (dim > 1) {
      partial_sum[base + 1] = sh_sum[1][0];
      partial_sumsq[base + 1] = sh_sumsq[1][0];
    }
    if (dim > 2) {
      partial_sum[base + 2] = sh_sum[2][0];
      partial_sumsq[base + 2] = sh_sumsq[2][0];
    }
    if (dim > 3) {
      partial_sum[base + 3] = sh_sum[3][0];
      partial_sumsq[base + 3] = sh_sumsq[3][0];
    }
    if (dim > 4) {
      partial_sum[base + 4] = sh_sum[4][0];
      partial_sumsq[base + 4] = sh_sumsq[4][0];
    }
    if (dim > 5) {
      partial_sum[base + 5] = sh_sum[5][0];
      partial_sumsq[base + 5] = sh_sumsq[5][0];
    }
    if (dim > 6) {
      partial_sum[base + 6] = sh_sum[6][0];
      partial_sumsq[base + 6] = sh_sumsq[6][0];
    }
    if (dim > 7) {
      partial_sum[base + 7] = sh_sum[7][0];
      partial_sumsq[base + 7] = sh_sumsq[7][0];
    }
  }
}

__global__ void col_stats_stage2_small_kernel(const float* __restrict__ partial_sum,
                                             const float* __restrict__ partial_sumsq,
                                             float* __restrict__ mean,
                                             float* __restrict__ invstd,
                                             int32_t n,
                                             int32_t dim,
                                             int32_t num_blocks) {
  int j = threadIdx.x;
  if (j >= dim) return;

  float s = 0.0f;
  float q = 0.0f;
  for (int b = 0; b < num_blocks; b++) {
    int64_t idx = (int64_t)b * dim + j;
    s += partial_sum[idx];
    q += partial_sumsq[idx];
  }

  float inv_n = 1.0f / (float)n;
  float mu = s * inv_n;
  float ex2 = q * inv_n;
  float var = fmaxf(ex2 - mu * mu, 1e-20f);
  mean[j] = mu;
  invstd[j] = rsqrtf(var);
}

__global__ void compute_mean_invstd_kernel(float* __restrict__ mean,
                                          float* __restrict__ invstd,
                                          const float* __restrict__ sum,
                                          const float* __restrict__ sumsq,
                                          int32_t n,
                                          int32_t dim) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < dim) {
    float inv_n = 1.0f / (float)n;
    float mu = sum[j] * inv_n;
    float ex2 = sumsq[j] * inv_n;
    float var = fmaxf(ex2 - mu * mu, 1e-20f);
    mean[j] = mu;
    invstd[j] = rsqrtf(var);
  }
}

__global__ void whiten_cols_kernel(float* __restrict__ Y,
                                  const float* __restrict__ mean,
                                  const float* __restrict__ invstd,
                                  int32_t n,
                                  int32_t dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    for (int j = 0; j < dim; j++) {
      float v = Y[(int64_t)j * n + tid];
      v = (v - mean[j]) * invstd[j];
      Y[(int64_t)j * n + tid] = v;
    }
  }
}

__global__ void row_normalize_colmajor_kernel(float* __restrict__ data, int32_t n, int32_t dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    float norm2 = 0.0f;
    for (int d = 0; d < dim; d++) {
      float v = data[(int64_t)d * n + tid];
      norm2 += v * v;
    }
    float inv = rsqrtf(fmaxf(norm2, 1e-20f));
    for (int d = 0; d < dim; d++) {
      data[(int64_t)d * n + tid] *= inv;
    }
  }
}

__global__ void transpose_col_to_row_kernel(const float* __restrict__ col_major,
                                           float* __restrict__ row_major,
                                           int32_t n,
                                           int32_t k) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)n * k;
  if ((int64_t)idx < total) {
    int row = idx / k;
    int col = idx - row * k;
    row_major[idx] = col_major[(int64_t)col * n + row];
  }
}

__global__ void copy_centroid0_kernel(const float* __restrict__ pts, float* __restrict__ centroids, int32_t dim) {
  for (int d = threadIdx.x; d < dim; d += blockDim.x) centroids[d] = pts[d];
}

__global__ void init_min_dists_kernel(const float* __restrict__ pts,
                                     const float* __restrict__ centroid,
                                     float* __restrict__ min_dists,
                                     int32_t n, int32_t dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float* p = pts + (int64_t)idx * dim;
    float dist = 0.0f;
    for (int d = 0; d < dim; d++) {
      float diff = p[d] - centroid[d];
      dist += diff * diff;
    }
    min_dists[idx] = dist;
  }
}

__global__ void update_min_dists_kernel(const float* __restrict__ pts,
                                       const float* __restrict__ centroid,
                                       float* __restrict__ min_dists,
                                       int32_t n, int32_t dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float* p = pts + (int64_t)idx * dim;
    float dist = 0.0f;
    for (int d = 0; d < dim; d++) {
      float diff = p[d] - centroid[d];
      dist += diff * diff;
    }
    float old = min_dists[idx];
    if (dist < old) min_dists[idx] = dist;
  }
}

__global__ void block_argmax_kernel(const float* __restrict__ values,
                                   float* __restrict__ block_best_vals,
                                   int32_t* __restrict__ block_best_idxs,
                                   int32_t n) {
  float best_val = -1.0f;
  int best_idx = 0;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < n; i += stride) {
    float v = values[i];
    if ((v > best_val) || ((v == best_val) && (i < best_idx))) {
      best_val = v;
      best_idx = i;
    }
  }

  __shared__ float s_val[256];
  __shared__ int s_idx[256];
  s_val[threadIdx.x] = best_val;
  s_idx[threadIdx.x] = best_idx;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      float ov = s_val[threadIdx.x + offset];
      int oi = s_idx[threadIdx.x + offset];
      float cv = s_val[threadIdx.x];
      int ci = s_idx[threadIdx.x];
      if ((ov > cv) || ((ov == cv) && (oi < ci))) {
        s_val[threadIdx.x] = ov;
        s_idx[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    block_best_vals[blockIdx.x] = s_val[0];
    block_best_idxs[blockIdx.x] = s_idx[0];
  }
}

__global__ void final_argmax_kernel(const float* __restrict__ block_vals,
                                   const int32_t* __restrict__ block_idxs,
                                   float* __restrict__ out_val,
                                   int32_t* __restrict__ out_idx,
                                   int32_t num_blocks) {
  float best_val = -1.0f;
  int best_idx = 0;
  for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
    float v = block_vals[i];
    int idx = block_idxs[i];
    if ((v > best_val) || ((v == best_val) && (idx < best_idx))) {
      best_val = v;
      best_idx = idx;
    }
  }

  __shared__ float s_val[256];
  __shared__ int s_idx[256];
  s_val[threadIdx.x] = best_val;
  s_idx[threadIdx.x] = best_idx;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      float ov = s_val[threadIdx.x + offset];
      int oi = s_idx[threadIdx.x + offset];
      float cv = s_val[threadIdx.x];
      int ci = s_idx[threadIdx.x];
      if ((ov > cv) || ((ov == cv) && (oi < ci))) {
        s_val[threadIdx.x] = ov;
        s_idx[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out_val[0] = s_val[0];
    out_idx[0] = s_idx[0];
  }
}

__global__ void copy_centroid_from_idx_kernel(const float* __restrict__ pts,
                                             float* __restrict__ centroid_out,
                                             const int32_t* __restrict__ idx,
                                             int32_t dim) {
  int i = idx[0];
  const float* p = pts + (int64_t)i * dim;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) centroid_out[d] = p[d];
}

__global__ void kmeans_assign_kernel(const float* __restrict__ pts,
                                    const float* __restrict__ centroids,
                                    int32_t* __restrict__ assignments,
                                    int32_t n,
                                    int32_t k,
                                    int32_t dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float* p = pts + (int64_t)idx * dim;
    float best_dist = CUDART_INF_F;
    int best = 0;
    for (int c = 0; c < k; c++) {
      const float* cent = centroids + (int64_t)c * dim;
      float dist = 0.0f;
      for (int d = 0; d < dim; d++) {
        float diff = p[d] - cent[d];
        dist += diff * diff;
      }
      if (dist < best_dist) {
        best_dist = dist;
        best = c;
      }
    }
    assignments[idx] = best;
  }
}

__global__ void kmeans_partials_small_kernel(const float* __restrict__ pts,
                                            const int32_t* __restrict__ assignments,
                                            float* __restrict__ partial_sums,
                                            int32_t* __restrict__ partial_counts,
                                            int32_t n,
                                            int32_t k,
                                            int32_t dim) {
  int tid = threadIdx.x;
  float local_sums[64];
  int32_t local_counts[8];
#pragma unroll
  for (int i = 0; i < 64; i++) local_sums[i] = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; i++) local_counts[i] = 0;

  int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
    int c = assignments[i];
    if ((unsigned)c < (unsigned)k) {
      local_counts[c]++;
      const float* p = pts + (int64_t)i * dim;
      int base = c * 8;
      for (int d = 0; d < dim; d++) {
        local_sums[base + d] += p[d];
      }
    }
  }

  extern __shared__ char smem[];
  float* s_sums = reinterpret_cast<float*>(smem);
  int32_t* s_counts = reinterpret_cast<int32_t*>(s_sums + (size_t)blockDim.x * 64);

#pragma unroll
  for (int i = 0; i < 64; i++) s_sums[(size_t)tid * 64 + i] = local_sums[i];
#pragma unroll
  for (int i = 0; i < 8; i++) s_counts[(size_t)tid * 8 + i] = local_counts[i];
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        s_counts[(size_t)tid * 8 + i] += s_counts[(size_t)(tid + offset) * 8 + i];
      }
#pragma unroll
      for (int i = 0; i < 64; i++) {
        s_sums[(size_t)tid * 64 + i] += s_sums[(size_t)(tid + offset) * 64 + i];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    int64_t block_base = (int64_t)blockIdx.x * k;
    for (int c = 0; c < k; c++) {
      partial_counts[block_base + c] = s_counts[c];
      int64_t out_base = (block_base + c) * dim;
      int in_base = c * 8;
      for (int d = 0; d < dim; d++) {
        partial_sums[out_base + d] = s_sums[in_base + d];
      }
    }
  }
}

__global__ void kmeans_assign_partials_small_kernel(const float* __restrict__ pts,
                                                   const float* __restrict__ centroids,
                                                   int32_t* __restrict__ assignments,
                                                   float* __restrict__ partial_sums,
                                                   int32_t* __restrict__ partial_counts,
                                                   int32_t n,
                                                   int32_t k,
                                                   int32_t dim) {
  int tid = threadIdx.x;
  float local_sums[64];
  int32_t local_counts[8];
#pragma unroll
  for (int i = 0; i < 64; i++) local_sums[i] = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; i++) local_counts[i] = 0;

  int stride = blockDim.x * gridDim.x;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) {
    const float* p = pts + (int64_t)i * dim;
    float best_dist = CUDART_INF_F;
    int best = 0;
    for (int c = 0; c < k; c++) {
      const float* cent = centroids + (int64_t)c * dim;
      float dist = 0.0f;
      for (int d = 0; d < dim; d++) {
        float diff = p[d] - cent[d];
        dist += diff * diff;
      }
      if ((dist < best_dist) || ((dist == best_dist) && (c < best))) {
        best_dist = dist;
        best = c;
      }
    }
    assignments[i] = best;
    local_counts[best]++;
    int base = best * 8;
    for (int d = 0; d < dim; d++) {
      local_sums[base + d] += p[d];
    }
  }

  extern __shared__ char smem[];
  float* s_sums = reinterpret_cast<float*>(smem);
  int32_t* s_counts = reinterpret_cast<int32_t*>(s_sums + (size_t)blockDim.x * 64);

#pragma unroll
  for (int i = 0; i < 64; i++) s_sums[(size_t)tid * 64 + i] = local_sums[i];
#pragma unroll
  for (int i = 0; i < 8; i++) s_counts[(size_t)tid * 8 + i] = local_counts[i];
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        s_counts[(size_t)tid * 8 + i] += s_counts[(size_t)(tid + offset) * 8 + i];
      }
#pragma unroll
      for (int i = 0; i < 64; i++) {
        s_sums[(size_t)tid * 64 + i] += s_sums[(size_t)(tid + offset) * 64 + i];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    int64_t block_base = (int64_t)blockIdx.x * k;
    for (int c = 0; c < k; c++) {
      partial_counts[block_base + c] = s_counts[c];
      int64_t out_base = (block_base + c) * dim;
      int in_base = c * 8;
      for (int d = 0; d < dim; d++) {
        partial_sums[out_base + d] = s_sums[in_base + d];
      }
    }
  }
}

__global__ void kmeans_reduce_partials_small_kernel(const float* __restrict__ partial_sums,
                                                   const int32_t* __restrict__ partial_counts,
                                                   float* __restrict__ sums,
                                                   int32_t* __restrict__ counts,
                                                   int32_t num_blocks,
                                                   int32_t k,
                                                   int32_t dim) {
  int c = blockIdx.x;
  if (c >= k) return;
  int tid = threadIdx.x;

  float local[8];
#pragma unroll
  for (int d = 0; d < 8; d++) local[d] = 0.0f;
  int32_t local_cnt = 0;

  for (int b = tid; b < num_blocks; b += blockDim.x) {
    int64_t idx = (int64_t)b * k + c;
    local_cnt += partial_counts[idx];
    int64_t base = idx * dim;
    for (int d = 0; d < dim; d++) {
      local[d] += partial_sums[base + d];
    }
  }

  __shared__ float s_sum[8][256];
  __shared__ int32_t s_cnt[256];
  s_cnt[tid] = local_cnt;
#pragma unroll
  for (int d = 0; d < 8; d++) s_sum[d][tid] = local[d];
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_cnt[tid] += s_cnt[tid + offset];
#pragma unroll
      for (int d = 0; d < 8; d++) {
        s_sum[d][tid] += s_sum[d][tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    counts[c] = s_cnt[0];
    int64_t out_base = (int64_t)c * dim;
    for (int d = 0; d < dim; d++) {
      sums[out_base + d] = s_sum[d][0];
    }
  }
}

__global__ void kmeans_sums_counts_kernel(const float* __restrict__ pts,
                                         const int32_t* __restrict__ assignments,
                                         float* __restrict__ sums,
                                         int32_t* __restrict__ counts,
                                         int32_t n,
                                         int32_t k,
                                         int32_t dim) {
  int c = blockIdx.x;
  if (c >= k) return;

  extern __shared__ char smem[];
  float* s_sums = reinterpret_cast<float*>(smem);
  int32_t* s_counts = reinterpret_cast<int32_t*>(s_sums + (size_t)dim * blockDim.x);

  int tid = threadIdx.x;
  int32_t local_count = 0;

  for (int d = 0; d < dim; d++) s_sums[(size_t)d * blockDim.x + tid] = 0.0f;
  s_counts[tid] = 0;
  __syncthreads();

  for (int i = tid; i < n; i += blockDim.x) {
    if (assignments[i] == c) {
      local_count++;
      const float* p = pts + (int64_t)i * dim;
      for (int d = 0; d < dim; d++) {
        s_sums[(size_t)d * blockDim.x + tid] += p[d];
      }
    }
  }
  s_counts[tid] = local_count;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      s_counts[tid] += s_counts[tid + offset];
      for (int d = 0; d < dim; d++) {
        s_sums[(size_t)d * blockDim.x + tid] += s_sums[(size_t)d * blockDim.x + tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    counts[c] = s_counts[0];
    for (int d = 0; d < dim; d++) {
      sums[(int64_t)c * dim + d] = s_sums[(size_t)d * blockDim.x];
    }
  }
}

__global__ void kmeans_divide_kernel(float* __restrict__ centroids,
                                    const float* __restrict__ sums,
                                    const int32_t* __restrict__ counts,
                                    const float* __restrict__ old_centroids,
                                    int32_t k,
                                    int32_t dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = (int64_t)k * dim;
  if ((int64_t)idx < total) {
    int c = idx / dim;
    int cnt = counts[c];
    if (cnt > 0) centroids[idx] = sums[idx] * (1.0f / (float)cnt);
    else centroids[idx] = old_centroids[idx];
  }
}

__global__ void kmeans_max_shift_kernel(const float* __restrict__ old_centroids,
                                       const float* __restrict__ new_centroids,
                                       float* __restrict__ max_shift,
                                       int32_t k,
                                       int32_t dim) {
  float local_max = 0.0f;
  for (int c = threadIdx.x; c < k; c += blockDim.x) {
    const float* a = old_centroids + (int64_t)c * dim;
    const float* b = new_centroids + (int64_t)c * dim;
    float shift2 = 0.0f;
    for (int d = 0; d < dim; d++) {
      float diff = b[d] - a[d];
      shift2 += diff * diff;
    }
    local_max = fmaxf(local_max, sqrtf(shift2));
  }

  __shared__ float s[256];
  s[threadIdx.x] = local_max;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + offset]);
    __syncthreads();
  }
  if (threadIdx.x == 0) max_shift[0] = s[0];
}





static void launch_compute_degrees(const int32_t* offsets, const float* weights, float* degrees, int32_t n, cudaStream_t stream) {
  int block = 256;
  int grid = (n + 7) / 8;
  compute_degrees_kernel<<<grid, block, 0, stream>>>(offsets, weights, degrees, n);
}

static void launch_reduce_sum(const float* x, float* partial, float* out, int32_t n, int32_t num_blocks, cudaStream_t stream) {
  reduce_sum_stage1_kernel<<<num_blocks, 256, 0, stream>>>(x, partial, n);
  reduce_sum_stage2_kernel<<<1, 256, 0, stream>>>(partial, out, num_blocks);
}

static void launch_modularity_correction(float* w, const float* degrees, const float* gamma, const float* two_m,
                                 int32_t n, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  modularity_correction_kernel<<<grid, block, 0, stream>>>(w, degrees, gamma, two_m, n);
}

static void launch_lanczos_recurrence(float* w, const float* v_j, const float* v_jm1,
                              const float* alpha_ptr, const float* beta_ptr,
                              int32_t n, bool has_prev, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  lanczos_recurrence_kernel<<<grid, block, 0, stream>>>(w, v_j, v_jm1, alpha_ptr, beta_ptr, n, has_prev);
}

static void launch_normalize_and_copy(const float* w, float* v_next, const float* beta_ptr, int32_t n, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  normalize_and_copy_kernel<<<grid, block, 0, stream>>>(w, v_next, beta_ptr, n);
}

static void launch_fill(float* v, int32_t n, float val, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  fill_kernel<<<grid, block, 0, stream>>>(v, n, val);
}

static void launch_compute_mean_invstd(float* mean, float* invstd, const float* sum, const float* sumsq,
                               int32_t n, int32_t dim, cudaStream_t stream) {
  int block = 256;
  int grid = (dim + block - 1) / block;
  compute_mean_invstd_kernel<<<grid, block, 0, stream>>>(mean, invstd, sum, sumsq, n, dim);
}

static void launch_col_stats_small(const float* Y, float* mean, float* invstd, float* partial_sum, float* partial_sumsq,
                            int32_t n, int32_t dim, int32_t num_blocks, cudaStream_t stream) {
  col_stats_stage1_small_kernel<<<num_blocks, 256, 0, stream>>>(Y, partial_sum, partial_sumsq, n, dim);
  col_stats_stage2_small_kernel<<<1, 32, 0, stream>>>(partial_sum, partial_sumsq, mean, invstd, n, dim, num_blocks);
}

static void launch_whiten_cols(float* Y, const float* mean, const float* invstd, int32_t n, int32_t dim, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  whiten_cols_kernel<<<grid, block, 0, stream>>>(Y, mean, invstd, n, dim);
}

static void launch_row_normalize_colmajor(float* data, int32_t n, int32_t dim, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  row_normalize_colmajor_kernel<<<grid, block, 0, stream>>>(data, n, dim);
}

static void launch_transpose_col_to_row(const float* col_major, float* row_major, int32_t n, int32_t k, cudaStream_t stream) {
  int block = 256;
  int64_t total = (int64_t)n * k;
  int grid = (int)((total + block - 1) / block);
  transpose_col_to_row_kernel<<<grid, block, 0, stream>>>(col_major, row_major, n, k);
}

static void launch_kmeanspp_init(const float* pts, float* centroids, float* min_dists, int32_t n, int32_t k, int32_t dim,
                         float* block_best_vals, int32_t* block_best_idxs, float* tmp_best_vals, int32_t* tmp_best_idxs,
                         int32_t* best_idx, cudaStream_t stream) {
  copy_centroid0_kernel<<<1, 256, 0, stream>>>(pts, centroids, dim);
  int block = 256;
  int grid = (n + block - 1) / block;
  init_min_dists_kernel<<<grid, block, 0, stream>>>(pts, centroids, min_dists, n, dim);

  for (int c = 1; c < k; c++) {
    block_argmax_kernel<<<grid, block, 0, stream>>>(min_dists, block_best_vals, block_best_idxs, n);
    final_argmax_kernel<<<1, 256, 0, stream>>>(block_best_vals, block_best_idxs, tmp_best_vals, best_idx, grid);
    copy_centroid_from_idx_kernel<<<1, 256, 0, stream>>>(pts, centroids + (int64_t)c * dim, best_idx, dim);
    update_min_dists_kernel<<<grid, block, 0, stream>>>(pts, centroids + (int64_t)c * dim, min_dists, n, dim);
  }
  (void)tmp_best_idxs;
}

static void launch_kmeans_assign(const float* pts, const float* centroids, int32_t* assignments,
                         int32_t n, int32_t k, int32_t dim, cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  kmeans_assign_kernel<<<grid, block, 0, stream>>>(pts, centroids, assignments, n, k, dim);
}

static void launch_kmeans_sums_counts(const float* pts, const int32_t* assignments, float* sums, int32_t* counts,
                              int32_t n, int32_t k, int32_t dim, cudaStream_t stream) {
  int block = 256;
  size_t smem = (size_t)dim * block * sizeof(float) + (size_t)block * sizeof(int32_t);
  while (smem > 98304 && block > 1) {
    block >>= 1;
    smem = (size_t)dim * block * sizeof(float) + (size_t)block * sizeof(int32_t);
  }
  kmeans_sums_counts_kernel<<<k, block, smem, stream>>>(pts, assignments, sums, counts, n, k, dim);
}

static void launch_kmeans_sums_counts_small(const float* pts, const int32_t* assignments,
                                    float* sums, int32_t* counts,
                                    float* partial_sums, int32_t* partial_counts,
                                    int32_t n, int32_t k, int32_t dim, int32_t num_blocks,
                                    cudaStream_t stream) {
  int block = 256;
  size_t smem = (size_t)block * (64 * sizeof(float) + 8 * sizeof(int32_t));
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute(kmeans_partials_small_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    configured = true;
  }
  kmeans_partials_small_kernel<<<num_blocks, block, smem, stream>>>(pts, assignments, partial_sums, partial_counts, n, k, dim);
  kmeans_reduce_partials_small_kernel<<<k, block, 0, stream>>>(partial_sums, partial_counts, sums, counts, num_blocks, k, dim);
}

static void launch_kmeans_assign_sums_counts_small(const float* pts, const float* centroids, int32_t* assignments,
                                           float* sums, int32_t* counts,
                                           float* partial_sums, int32_t* partial_counts,
                                           int32_t n, int32_t k, int32_t dim, int32_t num_blocks,
                                           cudaStream_t stream) {
  int block = 256;
  size_t smem = (size_t)block * (64 * sizeof(float) + 8 * sizeof(int32_t));
  static bool configured2 = false;
  if (!configured2) {
    cudaFuncSetAttribute(kmeans_assign_partials_small_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    configured2 = true;
  }
  kmeans_assign_partials_small_kernel<<<num_blocks, block, smem, stream>>>(pts, centroids, assignments, partial_sums, partial_counts, n, k, dim);
  kmeans_reduce_partials_small_kernel<<<k, block, 0, stream>>>(partial_sums, partial_counts, sums, counts, num_blocks, k, dim);
}

static void launch_kmeans_divide(float* centroids, const float* sums, const int32_t* counts,
                          const float* old_centroids, int32_t k, int32_t dim, cudaStream_t stream) {
  int block = 256;
  int64_t total = (int64_t)k * dim;
  int grid = (int)((total + block - 1) / block);
  kmeans_divide_kernel<<<grid, block, 0, stream>>>(centroids, sums, counts, old_centroids, k, dim);
}

static void launch_kmeans_max_shift(const float* old_centroids, const float* new_centroids, float* max_shift,
                            int32_t k, int32_t dim, cudaStream_t stream) {
  kmeans_max_shift_kernel<<<1, 256, 0, stream>>>(old_centroids, new_centroids, max_shift, k, dim);
}

}  

void spectral_modularity_maximization(const graph32_t& graph,
                                      const float* edge_weights,
                                      int32_t num_clusters,
                                      int32_t num_eigenvectors,
                                      float evs_tolerance,
                                      int32_t evs_max_iter,
                                      float kmean_tolerance,
                                      int32_t kmean_max_iter,
                                      int32_t* clustering) {
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);

  const int32_t* d_offsets = graph.offsets;
  const int32_t* d_indices = graph.indices;
  int32_t n = graph.number_of_vertices;
  int32_t nnz = graph.number_of_edges;
  const float* d_wts = edge_weights;

  int32_t k = num_clusters;
  int32_t dim = num_eigenvectors;

  cudaStream_t stream = 0;
  cublasSetStream(cache.cublas_h, stream);
  cusparseSetStream(cache.cusparse_h, stream);

  
  float* d_degrees = nullptr;
  float* d_partial = nullptr;
  float* d_two_m = nullptr;
  float* d_V = nullptr;
  float* d_w = nullptr;
  float* d_gamma = nullptr;
  float* d_alpha = nullptr;
  float* d_beta = nullptr;
  void* d_spmv_buffer = nullptr;
  float* d_h = nullptr;
  float* d_Y = nullptr;
  float* d_S = nullptr;
  float* d_mean = nullptr;
  float* d_invstd = nullptr;
  float* d_partial_sum = nullptr;
  float* d_partial_sumsq = nullptr;
  float* d_ones = nullptr;
  float* d_sum = nullptr;
  float* d_sumsq = nullptr;
  float* d_pts = nullptr;
  float* d_centroids = nullptr;
  float* d_centroids_old = nullptr;
  float* d_min_dists = nullptr;
  float* d_block_best_vals = nullptr;
  int32_t* d_block_best_idxs = nullptr;
  float* d_tmp_best_vals = nullptr;
  int32_t* d_tmp_best_idxs = nullptr;
  int32_t* d_best_idx = nullptr;
  float* d_sums = nullptr;
  int32_t* d_counts = nullptr;
  float* d_km_partial_sums = nullptr;
  int32_t* d_km_partial_counts = nullptr;
  float* d_max_shift = nullptr;

  
  cudaMalloc(&d_degrees, (size_t)n * sizeof(float));
  launch_compute_degrees(d_offsets, d_wts, d_degrees, n, stream);

  
  int32_t red_blocks = std::min<int32_t>((n + 255) / 256, 4096);
  cudaMalloc(&d_partial, (size_t)red_blocks * sizeof(float));
  cudaMalloc(&d_two_m, sizeof(float));
  launch_reduce_sum(d_degrees, d_partial, d_two_m, n, red_blocks, stream);

  float h_two_m = 0.0f;
  cudaMemcpyAsync(&h_two_m, d_two_m, sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  if (fabsf(h_two_m) < 1e-20f) {
    cudaMemsetAsync(clustering, 0, (size_t)n * sizeof(int32_t), stream);
    cudaFree(d_degrees);
    cudaFree(d_partial);
    cudaFree(d_two_m);
    return;
  }

  
  int32_t lanczos_dim = std::max<int32_t>(2 * dim + 30, 40);
  lanczos_dim = std::min<int32_t>(lanczos_dim, n);
  lanczos_dim = std::min<int32_t>(lanczos_dim, evs_max_iter);
  lanczos_dim = std::max<int32_t>(lanczos_dim, dim + 2);

  cudaMalloc(&d_V, (size_t)lanczos_dim * n * sizeof(float));
  cudaMalloc(&d_w, (size_t)n * sizeof(float));
  cudaMalloc(&d_gamma, sizeof(float));
  cudaMalloc(&d_alpha, (size_t)lanczos_dim * sizeof(float));
  cudaMalloc(&d_beta, (size_t)(lanczos_dim + 1) * sizeof(float));
  cudaMemsetAsync(d_beta, 0, (size_t)(lanczos_dim + 1) * sizeof(float), stream);

  float init_val = 1.0f / sqrtf((float)n);
  launch_fill(d_V, n, init_val, stream);

  
  cusparseSpMatDescr_t matA;
  cusparseCreateCsr(&matA, n, n, nnz,
                    (void*)d_offsets, (void*)d_indices, (void*)d_wts,
                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

  cusparseDnVecDescr_t vecX, vecY;
  cusparseCreateDnVec(&vecX, n, (void*)d_V, CUDA_R_32F);
  cusparseCreateDnVec(&vecY, n, (void*)d_w, CUDA_R_32F);

  float spmv_alpha = 1.0f, spmv_beta = 0.0f;
  size_t bufferSize = 0;
  cusparseSpMV_bufferSize(
      cache.cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &spmv_alpha, matA, vecX, &spmv_beta, vecY,
      CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &bufferSize);

  size_t spmv_alloc_size = std::max<size_t>(bufferSize, 1);
  cudaMalloc(&d_spmv_buffer, spmv_alloc_size);

  cusparseSpMV_preprocess(
      cache.cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
      &spmv_alpha, matA, vecX, &spmv_beta, vecY,
      CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, d_spmv_buffer);

  cudaMalloc(&d_h, (size_t)lanczos_dim * sizeof(float));

  for (int j = 0; j < lanczos_dim; j++) {
    float* v_j = d_V + (int64_t)j * n;

    cusparseDnVecSetValues(vecX, (void*)v_j);
    cusparseDnVecSetValues(vecY, (void*)d_w);
    cusparseSpMV(
        cache.cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha, matA, vecX, &spmv_beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, d_spmv_buffer);

    cublasSetPointerMode(cache.cublas_h, CUBLAS_POINTER_MODE_DEVICE);
    cublasSdot(cache.cublas_h, n, d_degrees, 1, v_j, 1, d_gamma);
    launch_modularity_correction(d_w, d_degrees, d_gamma, d_two_m, n, stream);

    cublasSdot(cache.cublas_h, n, v_j, 1, d_w, 1, d_alpha + j);

    float* v_jm1 = (j > 0) ? (d_V + (int64_t)(j - 1) * n) : nullptr;
    launch_lanczos_recurrence(d_w, v_j, v_jm1, d_alpha + j, d_beta + j, n, j > 0, stream);

    if (j > 0) {
      cublasSetPointerMode(cache.cublas_h, CUBLAS_POINTER_MODE_HOST);
      float one = 1.0f, zero = 0.0f, neg_one = -1.0f;
      cublasSgemv(cache.cublas_h, CUBLAS_OP_T, n, j + 1, &one, d_V, n, d_w, 1, &zero, d_h, 1);
      cublasSgemv(cache.cublas_h, CUBLAS_OP_N, n, j + 1, &neg_one, d_V, n, d_h, 1, &one, d_w, 1);
    }

    cublasSetPointerMode(cache.cublas_h, CUBLAS_POINTER_MODE_DEVICE);
    cublasSnrm2(cache.cublas_h, n, d_w, 1, d_beta + j + 1);

    if (j + 1 < lanczos_dim) {
      launch_normalize_and_copy(d_w, d_V + (int64_t)(j + 1) * n, d_beta + j + 1, n, stream);
    }
  }

  std::vector<float> h_alpha(lanczos_dim);
  std::vector<float> h_beta_arr(lanczos_dim + 1);
  cudaMemcpyAsync(h_alpha.data(), d_alpha, (size_t)lanczos_dim * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(h_beta_arr.data(), d_beta, (size_t)(lanczos_dim + 1) * sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  int actual_dim = lanczos_dim;
  float beta_eps = std::max(1e-10f, evs_tolerance * fabsf(h_beta_arr[1]));
  for (int j = 1; j <= lanczos_dim; j++) {
    if (fabsf(h_beta_arr[j]) < beta_eps) { actual_dim = j; break; }
  }
  actual_dim = std::max(actual_dim, dim + 2);

  std::vector<float> diag(actual_dim);
  std::vector<float> sub(actual_dim, 0.0f);
  for (int i = 0; i < actual_dim; i++) diag[i] = h_alpha[i];
  for (int i = 0; i < actual_dim - 1; i++) sub[i] = h_beta_arr[i + 1];

  std::vector<float> Z(actual_dim * actual_dim);
  tridiag_ql(actual_dim, diag.data(), sub.data(), Z.data());

  std::vector<int> idx(actual_dim);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) { return diag[a] > diag[b]; });

  std::vector<float> S_top(actual_dim * dim);
  for (int j = 0; j < dim; j++) {
    int col = idx[j];
    for (int i = 0; i < actual_dim; i++) {
      S_top[i + j * actual_dim] = Z[i + col * actual_dim];
    }
  }

  cudaMalloc(&d_Y, (size_t)n * dim * sizeof(float));
  cudaMalloc(&d_S, (size_t)actual_dim * dim * sizeof(float));
  cudaMemcpyAsync(d_S, S_top.data(), (size_t)actual_dim * dim * sizeof(float), cudaMemcpyHostToDevice, stream);

  cublasSetPointerMode(cache.cublas_h, CUBLAS_POINTER_MODE_HOST);
  {
    float one = 1.0f, zero = 0.0f;
    cublasSgemm(cache.cublas_h, CUBLAS_OP_N, CUBLAS_OP_N,
                n, dim, actual_dim,
                &one, d_V, n, d_S, actual_dim,
                &zero, d_Y, n);
  }

  
  cudaMalloc(&d_mean, (size_t)dim * sizeof(float));
  cudaMalloc(&d_invstd, (size_t)dim * sizeof(float));

  if (dim <= 8) {
    int32_t stat_blocks = std::min<int32_t>((n + 255) / 256, 1024);
    cudaMalloc(&d_partial_sum, (size_t)stat_blocks * dim * sizeof(float));
    cudaMalloc(&d_partial_sumsq, (size_t)stat_blocks * dim * sizeof(float));
    launch_col_stats_small(d_Y, d_mean, d_invstd,
                           d_partial_sum, d_partial_sumsq,
                           n, dim, stat_blocks, stream);
  } else {
    cudaMalloc(&d_ones, (size_t)n * sizeof(float));
    launch_fill(d_ones, n, 1.0f, stream);

    cudaMalloc(&d_sum, (size_t)dim * sizeof(float));
    cudaMalloc(&d_sumsq, (size_t)dim * sizeof(float));

    cublasSetPointerMode(cache.cublas_h, CUBLAS_POINTER_MODE_DEVICE);
    for (int j = 0; j < dim; j++) {
      float* col = d_Y + (int64_t)j * n;
      cublasSdot(cache.cublas_h, n, col, 1, d_ones, 1, d_sum + j);
      cublasSdot(cache.cublas_h, n, col, 1, col, 1, d_sumsq + j);
    }
    launch_compute_mean_invstd(d_mean, d_invstd, d_sum, d_sumsq, n, dim, stream);
  }

  launch_whiten_cols(d_Y, d_mean, d_invstd, n, dim, stream);

  launch_row_normalize_colmajor(d_Y, n, dim, stream);

  
  cudaMalloc(&d_pts, (size_t)n * dim * sizeof(float));
  launch_transpose_col_to_row(d_Y, d_pts, n, dim, stream);

  
  cudaMalloc(&d_centroids, (size_t)k * dim * sizeof(float));
  cudaMalloc(&d_centroids_old, (size_t)k * dim * sizeof(float));

  cudaMalloc(&d_min_dists, (size_t)n * sizeof(float));
  int32_t arg_blocks = (n + 255) / 256;
  cudaMalloc(&d_block_best_vals, (size_t)arg_blocks * sizeof(float));
  cudaMalloc(&d_block_best_idxs, (size_t)arg_blocks * sizeof(int32_t));
  cudaMalloc(&d_tmp_best_vals, sizeof(float));
  cudaMalloc(&d_tmp_best_idxs, sizeof(int32_t));
  cudaMalloc(&d_best_idx, sizeof(int32_t));

  launch_kmeanspp_init(d_pts, d_centroids, d_min_dists, n, k, dim,
                       d_block_best_vals, d_block_best_idxs,
                       d_tmp_best_vals, d_tmp_best_idxs,
                       d_best_idx, stream);

  cudaMalloc(&d_sums, (size_t)k * dim * sizeof(float));
  cudaMalloc(&d_counts, (size_t)k * sizeof(int32_t));

  bool small_kmeans = (k <= 8 && dim <= 8);
  int32_t km_blocks = 1;
  if (small_kmeans) {
    km_blocks = std::min<int32_t>((n + 255) / 256, 1024);
    cudaMalloc(&d_km_partial_sums, (size_t)km_blocks * k * dim * sizeof(float));
    cudaMalloc(&d_km_partial_counts, (size_t)km_blocks * k * sizeof(int32_t));
  }

  cudaMalloc(&d_max_shift, sizeof(float));

  for (int iter = 0; iter < kmean_max_iter; iter++) {
    cudaMemcpyAsync(d_centroids_old, d_centroids, (size_t)k * dim * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    if (small_kmeans) {
      launch_kmeans_assign_sums_counts_small(d_pts, d_centroids, clustering,
                                             d_sums, d_counts,
                                             d_km_partial_sums, d_km_partial_counts,
                                             n, k, dim, km_blocks, stream);
    } else {
      launch_kmeans_assign(d_pts, d_centroids, clustering, n, k, dim, stream);
      launch_kmeans_sums_counts(d_pts, clustering, d_sums, d_counts, n, k, dim, stream);
    }
    launch_kmeans_divide(d_centroids, d_sums, d_counts, d_centroids_old, k, dim, stream);

    if (kmean_tolerance > 0.0f) {
      launch_kmeans_max_shift(d_centroids_old, d_centroids, d_max_shift, k, dim, stream);
      float h_shift = 0.0f;
      cudaMemcpyAsync(&h_shift, d_max_shift, sizeof(float), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      if (h_shift <= kmean_tolerance) break;
    }
  }

  cusparseDestroyDnVec(vecX);
  cusparseDestroyDnVec(vecY);
  cusparseDestroySpMat(matA);

  
  cudaFree(d_degrees);
  cudaFree(d_partial);
  cudaFree(d_two_m);
  cudaFree(d_V);
  cudaFree(d_w);
  cudaFree(d_gamma);
  cudaFree(d_alpha);
  cudaFree(d_beta);
  cudaFree(d_spmv_buffer);
  cudaFree(d_h);
  cudaFree(d_Y);
  cudaFree(d_S);
  cudaFree(d_mean);
  cudaFree(d_invstd);
  cudaFree(d_partial_sum);
  cudaFree(d_partial_sumsq);
  cudaFree(d_ones);
  cudaFree(d_sum);
  cudaFree(d_sumsq);
  cudaFree(d_pts);
  cudaFree(d_centroids);
  cudaFree(d_centroids_old);
  cudaFree(d_min_dists);
  cudaFree(d_block_best_vals);
  cudaFree(d_block_best_idxs);
  cudaFree(d_tmp_best_vals);
  cudaFree(d_tmp_best_idxs);
  cudaFree(d_best_idx);
  cudaFree(d_sums);
  cudaFree(d_counts);
  cudaFree(d_km_partial_sums);
  cudaFree(d_km_partial_counts);
  cudaFree(d_max_shift);
}

}  
