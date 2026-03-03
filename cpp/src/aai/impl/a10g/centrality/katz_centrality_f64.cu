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
#include <cmath>

namespace aai {

namespace {





__device__ __forceinline__ double warpReduceSum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}



__global__ void compute_x2_scalar_kernel(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ weights,
    double* __restrict__ x,
    double alpha_beta,
    double beta,
    int64_t N)
{
    for (int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         v < N; v += (int64_t)gridDim.x * blockDim.x) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        double sum = 0.0;
        for (int32_t j = start; j < end; ++j) {
            sum += weights[j];
        }
        x[v] = alpha_beta * sum + beta;
    }
}

__global__ void fill_kernel(double* __restrict__ x, int64_t n, double val) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)gridDim.x * blockDim.x)
        x[i] = val;
}

template <bool USE_BETAS>
__global__ void add_beta_diff_kernel(
    double* __restrict__ x_new,
    const double* __restrict__ x_old,
    double beta,
    const double* __restrict__ betas,
    int64_t n,
    double* __restrict__ diff_out)
{
    constexpr int BLOCK = 256;
    constexpr int WARPS = BLOCK / 32;
    __shared__ double warp_sums[WARPS];

    double thread_diff = 0.0;
    for (int64_t idx = (int64_t)blockIdx.x * BLOCK + threadIdx.x;
         idx < n; idx += (int64_t)gridDim.x * BLOCK) {
        double beta_v = USE_BETAS ? betas[idx] : beta;
        double val = x_new[idx] + beta_v;
        x_new[idx] = val;
        thread_diff += fabs(val - x_old[idx]);
    }

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    double ws = warpReduceSum(thread_diff);
    if (lane == 0) warp_sums[warp_id] = ws;
    __syncthreads();

    if (threadIdx.x < WARPS) {
        double v = warp_sums[threadIdx.x];
        v = warpReduceSum(v);
        if (threadIdx.x == 0 && v != 0.0) atomicAdd(diff_out, v);
    }
}

__global__ void l2_norm_kernel(const double* __restrict__ x, int64_t n, double* __restrict__ norm_out) {
    constexpr int BLOCK = 256;
    constexpr int WARPS = BLOCK / 32;
    __shared__ double warp_sums[WARPS];

    double local_sum = 0.0;
    for (int64_t idx = (int64_t)blockIdx.x * BLOCK + threadIdx.x;
         idx < n; idx += (int64_t)gridDim.x * BLOCK) {
        double v = x[idx];
        local_sum += v * v;
    }

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    double ws = warpReduceSum(local_sum);
    if (lane == 0) warp_sums[warp_id] = ws;
    __syncthreads();

    if (threadIdx.x < WARPS) {
        double v = warp_sums[threadIdx.x];
        v = warpReduceSum(v);
        if (threadIdx.x == 0 && v != 0.0) atomicAdd(norm_out, v);
    }
}

__global__ void scale_kernel(double* __restrict__ x, int64_t n, double scale) {
    for (int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; idx += (int64_t)gridDim.x * blockDim.x)
        x[idx] *= scale;
}





void launch_compute_x2_scalar(
    const int32_t* offsets, const double* weights, double* x,
    double alpha_beta, double beta, int64_t N, cudaStream_t stream) {
    if (N <= 0) return;
    int block = 256;
    int grid = (int)((N + block - 1) / block);
    if (grid > 4096) grid = 4096;
    compute_x2_scalar_kernel<<<grid, block, 0, stream>>>(offsets, weights, x, alpha_beta, beta, N);
}

void launch_fill(double* x, int64_t n, double val, cudaStream_t stream) {
    if (n <= 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    if (grid > 2048) grid = 2048;
    fill_kernel<<<grid, block, 0, stream>>>(x, n, val);
}

void launch_add_beta_diff(double* x_new, const double* x_old, double beta,
                          const double* betas, bool use_betas,
                          int64_t n, double* diff_out, cudaStream_t stream) {
    if (n <= 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    if (grid > 2048) grid = 2048;
    if (use_betas)
        add_beta_diff_kernel<true><<<grid, block, 0, stream>>>(x_new, x_old, beta, betas, n, diff_out);
    else
        add_beta_diff_kernel<false><<<grid, block, 0, stream>>>(x_new, x_old, beta, betas, n, diff_out);
}

void launch_l2_norm(const double* x, int64_t n, double* norm_out, cudaStream_t stream) {
    if (n <= 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    if (grid > 2048) grid = 2048;
    l2_norm_kernel<<<grid, block, 0, stream>>>(x, n, norm_out);
}

void launch_scale(double* x, int64_t n, double scale, cudaStream_t stream) {
    if (n <= 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    if (grid > 2048) grid = 2048;
    scale_kernel<<<grid, block, 0, stream>>>(x, n, scale);
}





struct Cache : Cacheable {
    cusparseHandle_t cusparse = nullptr;
    double* h_diff = nullptr;
    double* d_diff = nullptr;
    double* buf0 = nullptr;
    double* buf1 = nullptr;
    void* spmvBuf = nullptr;

    int64_t buf0_capacity = 0;
    int64_t buf1_capacity = 0;
    size_t spmvBuf_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse);
        cudaMallocHost(&h_diff, sizeof(double));
        cudaMalloc(&d_diff, sizeof(double));
    }

    void ensure(int64_t N, size_t spmv_size) {
        if (buf0_capacity < N) {
            if (buf0) cudaFree(buf0);
            cudaMalloc(&buf0, N * sizeof(double));
            buf0_capacity = N;
        }
        if (buf1_capacity < N) {
            if (buf1) cudaFree(buf1);
            cudaMalloc(&buf1, N * sizeof(double));
            buf1_capacity = N;
        }
        if (spmvBuf_capacity < spmv_size && spmv_size > 0) {
            if (spmvBuf) cudaFree(spmvBuf);
            cudaMalloc(&spmvBuf, spmv_size);
            spmvBuf_capacity = spmv_size;
        }
    }

    ~Cache() override {
        if (cusparse) cusparseDestroy(cusparse);
        if (h_diff) cudaFreeHost(h_diff);
        if (d_diff) cudaFree(d_diff);
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (spmvBuf) cudaFree(spmvBuf);
    }
};

}  

katz_centrality_result_t katz_centrality(const graph32_t& graph,
                     const double* edge_weights,
                     double* centralities,
                     double alpha,
                     double beta,
                     const double* betas,
                     double epsilon,
                     std::size_t max_iterations,
                     bool has_initial_guess,
                     bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    cudaStream_t stream = nullptr;
    cusparseSetStream(cache.cusparse, stream);

    int64_t N = static_cast<int64_t>(graph.number_of_vertices);
    int64_t E = static_cast<int64_t>(graph.number_of_edges);
    bool use_betas = (betas != nullptr);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const double* d_weights = edge_weights;

    
    cache.ensure(N, 0);

    double* bufs[2] = {cache.buf0, cache.buf1};

    
    cusparseSpMatDescr_t mat;
    cusparseCreateCsr(&mat, N, N, E,
                     (void*)d_offsets, (void*)d_indices, (void*)d_weights,
                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseDnVecDescr_t vecIn, vecOut;
    cusparseCreateDnVec(&vecIn, N, bufs[0], CUDA_R_64F);
    cusparseCreateDnVec(&vecOut, N, bufs[1], CUDA_R_64F);

    double h_alpha = alpha, h_zero = 0.0;

    size_t bufSize = 0;
    cusparseSpMV_bufferSize(cache.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &h_alpha, mat, vecIn, &h_zero, vecOut,
                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);

    
    cache.ensure(N, bufSize);

    
    size_t skipped_iters = 0;
    int cur = 0;

    if (has_initial_guess) {
        
        cudaMemcpyAsync(bufs[0], centralities,
                       N * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else if (!use_betas && max_iterations >= 2) {
        
        
        launch_compute_x2_scalar(d_offsets, d_weights, bufs[0],
                                 alpha * beta, beta, N, stream);
        skipped_iters = 2;
        max_iterations = max_iterations - 2;
    } else if (use_betas) {
        
        cudaMemcpyAsync(bufs[0], betas, N * sizeof(double),
                       cudaMemcpyDeviceToDevice, stream);
        skipped_iters = 1;
        max_iterations = (max_iterations >= 1) ? max_iterations - 1 : 0;
    } else {
        
        launch_fill(bufs[0], N, beta, stream);
        skipped_iters = 1;
        max_iterations = (max_iterations >= 1) ? max_iterations - 1 : 0;
    }

    
    if (alpha > 0.01 || max_iterations > 50) {
        cusparseSpMV_preprocess(cache.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &h_alpha, mat, vecIn, &h_zero, vecOut,
                               CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmvBuf);
    }

    
    size_t iters = 0;
    bool converged = false;

    for (size_t i = 0; i < max_iterations; ++i) {
        int nxt = 1 - cur;

        cusparseDnVecSetValues(vecIn, bufs[cur]);
        cusparseDnVecSetValues(vecOut, bufs[nxt]);

        cusparseSpMV(cache.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &h_alpha, mat, vecIn, &h_zero, vecOut,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmvBuf);

        cudaMemsetAsync(cache.d_diff, 0, sizeof(double), stream);
        launch_add_beta_diff(bufs[nxt], bufs[cur], beta, betas, use_betas,
                            N, cache.d_diff, stream);

        cudaMemcpyAsync(cache.h_diff, cache.d_diff, sizeof(double),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cur = nxt;
        iters = i + 1;

        if (*cache.h_diff < epsilon) {
            converged = true;
            break;
        }
    }

    
    if (normalize) {
        cudaMemsetAsync(cache.d_diff, 0, sizeof(double), stream);
        launch_l2_norm(bufs[cur], N, cache.d_diff, stream);
        cudaMemcpyAsync(cache.h_diff, cache.d_diff, sizeof(double),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        double l2 = sqrt(*cache.h_diff);
        if (l2 > 0.0)
            launch_scale(bufs[cur], N, 1.0 / l2, stream);
    }

    
    cudaMemcpyAsync(centralities, bufs[cur], N * sizeof(double),
                   cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    
    size_t total_iters = iters + skipped_iters;

    cusparseDestroySpMat(mat);
    cusparseDestroyDnVec(vecIn);
    cusparseDestroyDnVec(vecOut);

    return {total_iters, converged};
}

}  
