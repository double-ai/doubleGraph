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
#include <cub/cub.cuh>
#include <cstdint>
#include <cmath>

namespace aai {

namespace {

constexpr int BLOCK_SIZE = 256;

struct Cache : Cacheable {
    cusparseHandle_t cusparse = nullptr;
    double* d_scalar = nullptr;
    double* buf = nullptr;
    void* spmv_buf = nullptr;

    int64_t buf_capacity = 0;
    size_t spmv_buf_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse);
        cudaMalloc(&d_scalar, sizeof(double));
    }

    void ensure_buf(int32_t N) {
        if (buf_capacity < N) {
            if (buf) cudaFree(buf);
            cudaMalloc(&buf, (size_t)N * sizeof(double));
            buf_capacity = N;
        }
    }

    void ensure_spmv_buf(size_t sz) {
        if (spmv_buf_capacity < sz) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, sz);
            spmv_buf_capacity = sz;
        }
    }

    ~Cache() override {
        if (cusparse) { cusparseDestroy(cusparse); cusparse = nullptr; }
        if (d_scalar) { cudaFree(d_scalar); d_scalar = nullptr; }
        if (buf) { cudaFree(buf); buf = nullptr; }
        if (spmv_buf) { cudaFree(spmv_buf); spmv_buf = nullptr; }
    }
};


__global__ void fused_beta_diff_kernel(
    double* __restrict__ y,
    const double* __restrict__ x,
    double beta,
    const double* __restrict__ betas,
    bool use_betas,
    int32_t n,
    double* __restrict__ diff_out
) {
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    double d = 0.0;

    if (tid < n) {
        double b = use_betas ? betas[tid] : beta;
        double yv = y[tid] + b;
        y[tid] = yv;
        d = fabs(yv - x[tid]);
    }

    double block_sum = BlockReduce(temp).Sum(d);
    if (threadIdx.x == 0 && block_sum > 0.0) {
        atomicAdd(diff_out, block_sum);
    }
}


__global__ void sum_squares_kernel(
    const double* __restrict__ x,
    int32_t n,
    double* __restrict__ out
) {
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    double val = 0.0;
    if (tid < n) {
        val = x[tid] * x[tid];
    }

    double block_sum = BlockReduce(temp).Sum(val);
    if (threadIdx.x == 0 && block_sum > 0.0) {
        atomicAdd(out, block_sum);
    }
}


__global__ void normalize_kernel(
    double* __restrict__ x,
    int32_t n,
    const double* __restrict__ norm_sq
) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid < n) {
        x[tid] /= sqrt(*norm_sq);
    }
}

}  

katz_centrality_result_t katz_centrality_seg(const graph32_t& graph,
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

    int32_t N = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    
    cache.ensure_buf(N);

    
    double* bufs[2] = {centralities, cache.buf};
    int cur = 0;

    
    if (!has_initial_guess) {
        cudaMemsetAsync(bufs[cur], 0, N * sizeof(double));
    }

    
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, N, N, E,
                     (void*)d_offsets, (void*)d_indices, (void*)edge_weights,
                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    
    cusparseDnVecDescr_t vecIn, vecOut;
    cusparseCreateDnVec(&vecIn, N, bufs[cur], CUDA_R_64F);
    cusparseCreateDnVec(&vecOut, N, bufs[1 - cur], CUDA_R_64F);

    
    double spmv_alpha = alpha;
    double spmv_beta_val = 0.0;

    
    size_t bufSz = 0;
    cusparseSpMV_bufferSize(cache.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &spmv_alpha, matA, vecIn, &spmv_beta_val, vecOut,
                           CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSz);

    if (bufSz > 0) {
        cache.ensure_spmv_buf(bufSz);
    }

    
    bool use_betas = (betas != nullptr);

    
    bool converged = false;
    size_t iterations = 0;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        int nxt = 1 - cur;

        
        cusparseDnVecSetValues(vecIn, bufs[cur]);
        cusparseDnVecSetValues(vecOut, bufs[nxt]);

        cusparseSpMV(cache.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &spmv_alpha, matA, vecIn, &spmv_beta_val, vecOut,
                    CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                    bufSz > 0 ? cache.spmv_buf : nullptr);

        
        cudaMemsetAsync(cache.d_scalar, 0, sizeof(double));
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fused_beta_diff_kernel<<<grid, BLOCK_SIZE>>>(bufs[nxt], bufs[cur],
                                  beta, betas, use_betas,
                                  N, cache.d_scalar);

        
        double h_diff;
        cudaMemcpy(&h_diff, cache.d_scalar, sizeof(double), cudaMemcpyDeviceToHost);

        iterations = iter + 1;
        cur = nxt;

        if (h_diff < epsilon) {
            converged = true;
            break;
        }
    }

    
    if (normalize) {
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        cudaMemsetAsync(cache.d_scalar, 0, sizeof(double));
        sum_squares_kernel<<<grid, BLOCK_SIZE>>>(bufs[cur], N, cache.d_scalar);
        normalize_kernel<<<grid, BLOCK_SIZE>>>(bufs[cur], N, cache.d_scalar);
    }

    
    if (cur == 1) {
        cudaMemcpyAsync(centralities, cache.buf, (size_t)N * sizeof(double),
                       cudaMemcpyDeviceToDevice);
    }

    
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecIn);
    cusparseDestroyDnVec(vecOut);

    return {iterations, converged};
}

}  
