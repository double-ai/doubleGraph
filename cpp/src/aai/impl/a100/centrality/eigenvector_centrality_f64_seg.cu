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
#include <algorithm>

namespace aai {

namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;

struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    double* buf = nullptr;
    double* l2_sq = nullptr;
    double* diff = nullptr;
    double* alpha = nullptr;
    double* beta = nullptr;
    void* spmv_buf = nullptr;

    int64_t buf_capacity = 0;
    size_t spmv_buf_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cudaMalloc(&l2_sq, sizeof(double));
        cudaMalloc(&diff, sizeof(double));
        cudaMalloc(&alpha, sizeof(double));
        cudaMalloc(&beta, sizeof(double));
        double h_alpha = 1.0, h_beta = 0.0;
        cudaMemcpy(alpha, &h_alpha, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(beta, &h_beta, sizeof(double), cudaMemcpyHostToDevice);
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (buf) cudaFree(buf);
        if (l2_sq) cudaFree(l2_sq);
        if (diff) cudaFree(diff);
        if (alpha) cudaFree(alpha);
        if (beta) cudaFree(beta);
        if (spmv_buf) cudaFree(spmv_buf);
    }

    void ensure_buf(int64_t n) {
        if (buf_capacity < n) {
            if (buf) cudaFree(buf);
            cudaMalloc(&buf, n * sizeof(double));
            buf_capacity = n;
        }
    }

    void ensure_spmv_buf(size_t size) {
        if (size > 0 && spmv_buf_capacity < size) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, size);
            spmv_buf_capacity = size;
        }
    }
};


__global__ void init_uniform_kernel(double* __restrict__ c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = 1.0 / (double)n;
    }
}


__global__ void add_identity_l2_kernel(
    double* __restrict__ y,
    const double* __restrict__ x,
    double* __restrict__ l2_sq,
    int n)
{
    __shared__ double warp_sums[WARPS_PER_BLOCK];

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    double val = 0.0;
    if (idx < n) {
        val = y[idx] + x[idx];
        y[idx] = val;
    }

    double sq = val * val;

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sq += __shfl_down_sync(0xffffffff, sq, offset);

    if (lane == 0) warp_sums[warp] = sq;
    __syncthreads();

    
    if (warp == 0) {
        sq = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1)
            sq += __shfl_down_sync(0xffffffff, sq, offset);
        if (lane == 0)
            atomicAdd(l2_sq, sq);
    }
}


__global__ void normalize_diff_kernel(
    double* __restrict__ y,
    const double* __restrict__ x,
    const double* __restrict__ l2_sq,
    double* __restrict__ diff,
    int n,
    int compute_diff)
{
    __shared__ double warp_sums[WARPS_PER_BLOCK];

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    double norm = sqrt(*l2_sq);
    if (norm < 1e-300) norm = 1e-300;
    double inv_norm = 1.0 / norm;

    double d = 0.0;
    if (idx < n) {
        double val = y[idx] * inv_norm;
        y[idx] = val;
        if (compute_diff) {
            d = fabs(val - x[idx]);
        }
    }

    if (compute_diff) {
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            d += __shfl_down_sync(0xffffffff, d, offset);

        if (lane == 0) warp_sums[warp] = d;
        __syncthreads();

        if (warp == 0) {
            d = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0;
            #pragma unroll
            for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1)
                d += __shfl_down_sync(0xffffffff, d, offset);
            if (lane == 0)
                atomicAdd(diff, d);
        }
    }
}

}  

eigenvector_centrality_result_t eigenvector_centrality_seg(const graph32_t& graph,
                                const double* edge_weights,
                                double* centralities,
                                double epsilon,
                                std::size_t max_iterations,
                                const double* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    
    if (num_vertices == 0) {
        return {0, true};
    }

    
    cache.ensure_buf(num_vertices);

    
    double* x_curr = centralities;
    double* x_next = cache.buf;

    
    if (initial_centralities != nullptr) {
        cudaMemcpy(x_curr, initial_centralities,
                   num_vertices * sizeof(double), cudaMemcpyDeviceToDevice);
    } else {
        int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_uniform_kernel<<<grid, BLOCK_SIZE>>>(x_curr, num_vertices);
    }

    
    cusparseSpMatDescr_t mat_descr;
    cusparseCreateCsr(&mat_descr,
        num_vertices, num_vertices, num_edges,
        (void*)graph.offsets,
        (void*)graph.indices,
        (void*)edge_weights,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    
    cusparseDnVecDescr_t vec_x, vec_y;
    cusparseCreateDnVec(&vec_x, num_vertices, x_curr, CUDA_R_64F);
    cusparseCreateDnVec(&vec_y, num_vertices, x_next, CUDA_R_64F);

    
    double h_alpha = 1.0, h_beta = 0.0;
    size_t buffer_size = 0;
    cusparseSpMV_bufferSize(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha, mat_descr, vec_x, &h_beta, vec_y,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);

    
    cache.ensure_spmv_buf(buffer_size);

    
    cusparseSpMV_preprocess(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha, mat_descr, vec_x, &h_beta, vec_y,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

    
    cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);

    
    double threshold = (double)num_vertices * epsilon;
    const int check_interval = 10;

    bool converged = false;
    size_t actual_iterations = 0;

    
    for (size_t iter = 0; iter < max_iterations; iter++) {
        actual_iterations = iter + 1;

        
        cudaMemsetAsync(cache.l2_sq, 0, sizeof(double));

        
        cusparseDnVecSetValues(vec_x, x_curr);
        cusparseDnVecSetValues(vec_y, x_next);
        cusparseSpMV(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.alpha, mat_descr, vec_x, cache.beta, vec_y,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

        
        int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        add_identity_l2_kernel<<<grid, BLOCK_SIZE>>>(x_next, x_curr, cache.l2_sq, num_vertices);

        
        bool do_check = ((iter + 1) % check_interval == 0) || (iter + 1 == max_iterations);

        if (do_check) {
            cudaMemsetAsync(cache.diff, 0, sizeof(double));
        }

        
        normalize_diff_kernel<<<grid, BLOCK_SIZE>>>(x_next, x_curr, cache.l2_sq, cache.diff,
                                                     num_vertices, do_check ? 1 : 0);

        
        std::swap(x_curr, x_next);

        if (do_check) {
            
            double h_diff;
            cudaMemcpy(&h_diff, cache.diff, sizeof(double), cudaMemcpyDeviceToHost);
            if (h_diff < threshold) {
                converged = true;
                break;
            }
        }
    }

    
    if (x_curr != centralities) {
        cudaMemcpy(centralities, x_curr, num_vertices * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    
    cusparseDestroySpMat(mat_descr);
    cusparseDestroyDnVec(vec_x);
    cusparseDestroyDnVec(vec_y);

    
    cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_HOST);

    return {actual_iterations, converged};
}

}  
