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
#include <cmath>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* x_buf0 = nullptr;
    float* x_buf1 = nullptr;
    double* y_buf = nullptr;
    double* scalars = nullptr;
    int32_t x_buf0_cap = 0;
    int32_t x_buf1_cap = 0;
    int32_t y_buf_cap = 0;
    bool scalars_allocated = false;

    void ensure(int32_t n) {
        if (x_buf0_cap < n) {
            if (x_buf0) cudaFree(x_buf0);
            cudaMalloc(&x_buf0, n * sizeof(float));
            x_buf0_cap = n;
        }
        if (x_buf1_cap < n) {
            if (x_buf1) cudaFree(x_buf1);
            cudaMalloc(&x_buf1, n * sizeof(float));
            x_buf1_cap = n;
        }
        if (y_buf_cap < n) {
            if (y_buf) cudaFree(y_buf);
            cudaMalloc(&y_buf, n * sizeof(double));
            y_buf_cap = n;
        }
        if (!scalars_allocated) {
            cudaMalloc(&scalars, 2 * sizeof(double));
            scalars_allocated = true;
        }
    }

    ~Cache() override {
        if (x_buf0) cudaFree(x_buf0);
        if (x_buf1) cudaFree(x_buf1);
        if (y_buf) cudaFree(y_buf);
        if (scalars) cudaFree(scalars);
    }
};



__global__ void __launch_bounds__(256)
spmv_unified(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double*  __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const float*   __restrict__ x,
    double*        __restrict__ y,
    int32_t seg_high, int32_t n_high,
    int32_t seg_mid,  int32_t n_mid,
    int32_t seg_low,  int32_t n_low,
    int32_t seg_zero, int32_t n_zero,
    int32_t high_blocks, int32_t mid_blocks, int32_t low_blocks)
{
    const int bid = blockIdx.x;

    if (bid < high_blocks) {
        if (bid >= n_high) return;
        const int32_t v = seg_high + bid;
        const int32_t rs = offsets[v], re = offsets[v + 1];

        double sum = 0.0;
        for (int32_t e = rs + threadIdx.x; e < re; e += 256) {
            uint32_t mw = __ldg(&edge_mask[e >> 5]);
            if ((mw >> (e & 31)) & 1u)
                sum += __ldg(&weights[e]) * (double)__ldg(&x[__ldg(&indices[e])]);
        }

        #pragma unroll
        for (int s = 16; s > 0; s >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, s);

        __shared__ double ws[8];
        const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        if (lane == 0) ws[warp] = sum;
        __syncthreads();

        if (warp == 0) {
            double val = (lane < 8) ? ws[lane] : 0.0;
            #pragma unroll
            for (int s = 4; s > 0; s >>= 1)
                val += __shfl_down_sync(0xffffffff, val, s);
            if (lane == 0) y[v] = val + (double)x[v];
        }

    } else if (bid < high_blocks + mid_blocks) {
        const int local_bid = bid - high_blocks;
        const int warp_in_blk = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int warp_global = local_bid * 8 + warp_in_blk;

        if (warp_global >= n_mid) return;
        const int32_t v = seg_mid + warp_global;
        const int32_t rs = offsets[v], re = offsets[v + 1];

        double sum = 0.0;
        for (int32_t e = rs + lane; e < re; e += 32) {
            uint32_t mw = __ldg(&edge_mask[e >> 5]);
            if ((mw >> (e & 31)) & 1u)
                sum += __ldg(&weights[e]) * (double)__ldg(&x[__ldg(&indices[e])]);
        }
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, s);
        if (lane == 0) y[v] = sum + (double)x[v];

    } else if (bid < high_blocks + mid_blocks + low_blocks) {
        const int local_bid = bid - high_blocks - mid_blocks;
        const int32_t idx = local_bid * 256 + threadIdx.x;
        if (idx >= n_low) return;
        const int32_t v = seg_low + idx;
        const int32_t rs = offsets[v], re = offsets[v + 1];

        double sum = 0.0;
        for (int32_t e = rs; e < re; e++) {
            uint32_t mw = __ldg(&edge_mask[e >> 5]);
            if ((mw >> (e & 31)) & 1u)
                sum += __ldg(&weights[e]) * (double)__ldg(&x[__ldg(&indices[e])]);
        }
        y[v] = sum + (double)x[v];

    } else {
        const int local_bid = bid - high_blocks - mid_blocks - low_blocks;
        const int32_t idx = local_bid * 256 + threadIdx.x;
        if (idx >= n_zero) return;
        const int32_t v = seg_zero + idx;
        y[v] = (double)x[v];
    }
}



__global__ void __launch_bounds__(256)
norm_sq_kernel(const double* __restrict__ y, double* __restrict__ result, int32_t n)
{
    double loc = 0.0;
    for (int32_t i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        double v = y[i]; loc += v * v;
    }
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        loc += __shfl_down_sync(0xffffffff, loc, s);
    __shared__ double ws[8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) ws[warp] = loc;
    __syncthreads();
    if (warp == 0) {
        double val = (lane < 8) ? ws[lane] : 0.0;
        #pragma unroll
        for (int s = 4; s > 0; s >>= 1)
            val += __shfl_down_sync(0xffffffff, val, s);
        if (lane == 0) atomicAdd(result, val);
    }
}



__global__ void __launch_bounds__(256)
normalize_diff_cast_kernel(
    double*       __restrict__ y,
    const float*  __restrict__ x_old,
    float*        __restrict__ x_new,
    double*       __restrict__ diff_result,
    const double* __restrict__ norm_sq_ptr,
    int32_t n)
{
    const double ns = *norm_sq_ptr;
    const double inv_norm = (ns > 0.0) ? (1.0 / sqrt(ns)) : 1.0;

    double loc = 0.0;
    for (int32_t i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        double val = y[i] * inv_norm;
        y[i] = val;
        x_new[i] = (float)val;
        double d = val - (double)x_old[i];
        loc += (d >= 0.0) ? d : -d;
    }
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        loc += __shfl_down_sync(0xffffffff, loc, s);
    __shared__ double ws[8];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (lane == 0) ws[warp] = loc;
    __syncthreads();
    if (warp == 0) {
        double val = (lane < 8) ? ws[lane] : 0.0;
        #pragma unroll
        for (int s = 4; s > 0; s >>= 1)
            val += __shfl_down_sync(0xffffffff, val, s);
        if (lane == 0) atomicAdd(diff_result, val);
    }
}



__global__ void init_uniform_fp32(float* x, float val, int32_t n) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = val;
}

__global__ void cast_fp64_to_fp32(const double* src, float* dst, int32_t n) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (float)src[i];
}



static void launch_spmv_unified(
    const int32_t* offsets, const int32_t* indices, const double* weights,
    const uint32_t* edge_mask, const float* x, double* y,
    int32_t seg_high, int32_t n_high,
    int32_t seg_mid,  int32_t n_mid,
    int32_t seg_low,  int32_t n_low,
    int32_t seg_zero, int32_t n_zero,
    cudaStream_t stream)
{
    int32_t high_blocks = n_high;
    int32_t mid_blocks  = (n_mid + 7) / 8;
    int32_t low_blocks  = (n_low + 255) / 256;
    int32_t zero_blocks = (n_zero + 255) / 256;
    int32_t total = high_blocks + mid_blocks + low_blocks + zero_blocks;
    if (total <= 0) return;
    spmv_unified<<<total, 256, 0, stream>>>(
        offsets, indices, weights, edge_mask, x, y,
        seg_high, n_high, seg_mid, n_mid,
        seg_low, n_low, seg_zero, n_zero,
        high_blocks, mid_blocks, low_blocks);
}

static void launch_norm_sq(const double* y, double* result, int32_t n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    if (blocks > 256) blocks = 256;
    norm_sq_kernel<<<blocks, 256, 0, stream>>>(y, result, n);
}

static void launch_normalize_diff_cast(
    double* y, const float* x_old, float* x_new,
    double* diff_result, const double* norm_sq, int32_t n, cudaStream_t stream)
{
    int blocks = (n + 255) / 256;
    if (blocks > 256) blocks = 256;
    normalize_diff_cast_kernel<<<blocks, 256, 0, stream>>>(
        y, x_old, x_new, diff_result, norm_sq, n);
}

static void launch_init_uniform_fp32(float* x, float val, int32_t n, cudaStream_t stream) {
    init_uniform_fp32<<<(n+255)/256, 256, 0, stream>>>(x, val, n);
}

static void launch_cast_fp64_to_fp32(const double* src, float* dst, int32_t n, cudaStream_t stream) {
    cast_fp64_to_fp32<<<(n+255)/256, 256, 0, stream>>>(src, dst, n);
}

}  

eigenvector_centrality_result_t eigenvector_centrality_seg_mask(const graph32_t& graph,
                                      const double* edge_weights,
                                      double* centralities,
                                      double epsilon,
                                      std::size_t max_iterations,
                                      const double* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg_high = seg[0], seg_mid = seg[1], seg_low = seg[2];
    int32_t seg_zero = seg[3], seg_end = seg[4];
    int32_t n_high = seg_mid - seg_high;
    int32_t n_mid  = seg_low - seg_mid;
    int32_t n_low  = seg_zero - seg_low;
    int32_t n_zero = seg_end - seg_zero;

    const uint32_t* d_mask = graph.edge_mask;
    const double* d_weights = edge_weights;

    cudaStream_t stream = 0;

    cache.ensure(num_vertices);

    float* d_x[2] = { cache.x_buf0, cache.x_buf1 };
    double* d_y       = cache.y_buf;
    double* d_norm_sq = cache.scalars;
    double* d_diff    = cache.scalars + 1;

    int cur = 0;
    if (initial_centralities != nullptr) {
        launch_cast_fp64_to_fp32(initial_centralities, d_x[0], num_vertices, stream);
    } else {
        launch_init_uniform_fp32(d_x[0], 1.0f / (float)num_vertices,
                                  num_vertices, stream);
    }

    double threshold = (double)num_vertices * epsilon;
    bool converged = false;
    std::size_t iterations = 0;

    const int CHECK_INTERVAL = (num_vertices < 50000) ? 10 :
                               (num_vertices < 500000) ? 5 : 1;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        float* x_old = d_x[cur];
        float* x_new = d_x[1 - cur];

        launch_spmv_unified(d_offsets, d_indices, d_weights, d_mask, x_old, d_y,
                            seg_high, n_high, seg_mid, n_mid,
                            seg_low, n_low, seg_zero, n_zero, stream);

        cudaMemsetAsync(d_norm_sq, 0, 2 * sizeof(double), stream);

        launch_norm_sq(d_y, d_norm_sq, num_vertices, stream);

        launch_normalize_diff_cast(d_y, x_old, x_new,
                                    d_diff, d_norm_sq, num_vertices, stream);

        cur = 1 - cur;
        iterations = iter + 1;

        bool should_check = ((iterations % CHECK_INTERVAL) == 0)
                         || (iter == max_iterations - 1);
        if (should_check) {
            double h_diff;
            cudaMemcpyAsync(&h_diff, d_diff, sizeof(double),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (h_diff < threshold) {
                converged = true;
                break;
            }
        }
    }

    cudaMemcpyAsync(centralities, d_y,
                    num_vertices * sizeof(double),
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return {iterations, converged};
}

}  
