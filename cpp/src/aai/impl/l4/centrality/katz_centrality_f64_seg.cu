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
#include <algorithm>
#include <stdexcept>
#include <string>

namespace aai {

namespace {

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t err = (call); \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuSPARSE error ") + std::to_string(err)); \
    } \
} while(0)



constexpr int BLOCK_SIZE = 256;

__global__ void convergence_kernel(
    const double* __restrict__ x_new,
    const double* __restrict__ x_old,
    const int n,
    double* __restrict__ d_diff)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double thread_diff = 0.0;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += BLOCK_SIZE * gridDim.x) {
        thread_diff += fabs(x_new[i] - x_old[i]);
    }

    double block_diff = BlockReduce(temp_storage).Sum(thread_diff);
    if (threadIdx.x == 0) {
        atomicAdd(d_diff, block_diff);
    }
}

__global__ void fill_scalar_kernel(double* __restrict__ x, int n, double val)
{
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += BLOCK_SIZE * gridDim.x) {
        x[i] = val;
    }
}

__global__ void l2_norm_sq_kernel(const double* __restrict__ x, const int n, double* __restrict__ d_result)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double thread_sum = 0.0;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += BLOCK_SIZE * gridDim.x) {
        thread_sum += x[i] * x[i];
    }

    double block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    if (threadIdx.x == 0) {
        atomicAdd(d_result, block_sum);
    }
}

__global__ void scale_kernel(double* __restrict__ x, const int n, const double factor)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        x[i] *= factor;
    }
}

__global__ void add_beta_convergence_kernel(
    double* __restrict__ x_new,
    const double* __restrict__ x_old,
    const int n,
    const double beta,
    const double* __restrict__ betas,
    const bool use_betas,
    double* __restrict__ d_diff)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double thread_diff = 0.0;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += BLOCK_SIZE * gridDim.x) {
        double b = use_betas ? betas[i] : beta;
        double new_val = x_new[i] + b;
        x_new[i] = new_val;
        thread_diff += fabs(new_val - x_old[i]);
    }

    double block_diff = BlockReduce(temp_storage).Sum(thread_diff);
    if (threadIdx.x == 0) {
        atomicAdd(d_diff, block_diff);
    }
}



void launch_add_beta_convergence(
    double* x_new, const double* x_old, int n,
    double beta, const double* betas, bool use_betas,
    double* d_diff, cudaStream_t stream)
{
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 2048) grid = 2048;
    add_beta_convergence_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        x_new, x_old, n, beta, betas, use_betas, d_diff);
}

void launch_convergence(
    const double* x_new, const double* x_old, int n,
    double* d_diff, cudaStream_t stream)
{
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 2048) grid = 2048;
    convergence_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(x_new, x_old, n, d_diff);
}

void launch_fill_scalar(double* x, int n, double val, cudaStream_t stream)
{
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fill_scalar_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(x, n, val);
}

void launch_l2_norm_sq(const double* x, int n, double* d_result, cudaStream_t stream)
{
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 2048) grid = 2048;
    l2_norm_sq_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(x, n, d_result);
}

void launch_scale(double* x, int n, double factor, cudaStream_t stream)
{
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scale_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(x, n, factor);
}



struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    double* h_pinned = nullptr;
    double* x_buf = nullptr;
    double* y_buf = nullptr;
    double* d_diff = nullptr;
    void* spmv_buffer = nullptr;
    int64_t x_capacity = 0;
    int64_t y_capacity = 0;
    size_t spmv_capacity = 0;

    Cache() {
        CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
        cudaMallocHost(&h_pinned, sizeof(double));
        cudaMalloc(&d_diff, sizeof(double));
    }

    ~Cache() override {
        if (x_buf) cudaFree(x_buf);
        if (y_buf) cudaFree(y_buf);
        if (d_diff) cudaFree(d_diff);
        if (spmv_buffer) cudaFree(spmv_buffer);
        if (h_pinned) cudaFreeHost(h_pinned);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
    }
};

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

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    bool use_betas = (betas != nullptr);

    cudaStream_t stream = 0;
    CUSPARSE_CHECK(cusparseSetStream(cache.cusparse_handle, stream));

    
    if (cache.x_capacity < num_vertices) {
        if (cache.x_buf) cudaFree(cache.x_buf);
        cudaMalloc(&cache.x_buf, (size_t)num_vertices * sizeof(double));
        cache.x_capacity = num_vertices;
    }
    if (cache.y_capacity < num_vertices) {
        if (cache.y_buf) cudaFree(cache.y_buf);
        cudaMalloc(&cache.y_buf, (size_t)num_vertices * sizeof(double));
        cache.y_capacity = num_vertices;
    }

    double* x = cache.x_buf;
    double* y = cache.y_buf;
    double* d_diff = cache.d_diff;

    size_t iterations = 0;
    bool converged = false;

    
    
    if (!has_initial_guess && max_iterations > 0) {
        
        if (use_betas) {
            cudaMemcpyAsync(x, betas, num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
        } else {
            launch_fill_scalar(x, num_vertices, beta, stream);
        }
        iterations = 1;

        if (max_iterations == 1) {
            double expected_diff;
            if (!use_betas) {
                expected_diff = (double)num_vertices * fabs(beta);
            } else {
                cudaMemsetAsync(d_diff, 0, sizeof(double), stream);
                cudaMemsetAsync(y, 0, num_vertices * sizeof(double), stream);
                launch_convergence(x, y, num_vertices, d_diff, stream);
                cudaMemcpyAsync(cache.h_pinned, d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                expected_diff = *cache.h_pinned;
            }
            if (!use_betas) {
                converged = (expected_diff < epsilon);
            } else {
                converged = (expected_diff < epsilon);
            }
        }
    } else if (has_initial_guess) {
        cudaMemcpyAsync(x, centralities, num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else {
        
        cudaMemsetAsync(x, 0, num_vertices * sizeof(double), stream);
    }

    
    if (iterations < max_iterations && !converged) {
        cusparseSpMatDescr_t matA;
        CUSPARSE_CHECK(cusparseCreateCsr(&matA,
            num_vertices, num_vertices, num_edges,
            (void*)graph.offsets,
            (void*)graph.indices,
            (void*)edge_weights,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

        cusparseDnVecDescr_t vecX, vecY;
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, num_vertices, x, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, num_vertices, y, CUDA_R_64F));

        double spmv_alpha = alpha;
        double spmv_beta_val = 0.0;
        size_t bufferSize = 0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(
            cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmv_alpha, matA, vecX, &spmv_beta_val, vecY,
            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, &bufferSize));

        void* d_buffer = nullptr;
        if (bufferSize > 0) {
            if (cache.spmv_capacity < bufferSize) {
                if (cache.spmv_buffer) cudaFree(cache.spmv_buffer);
                cudaMalloc(&cache.spmv_buffer, bufferSize);
                cache.spmv_capacity = bufferSize;
            }
            d_buffer = cache.spmv_buffer;
        }

        
        CUSPARSE_CHECK(cusparseSpMV_preprocess(
            cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmv_alpha, matA, vecX, &spmv_beta_val, vecY,
            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, d_buffer));

        for (size_t iter = iterations; iter < max_iterations; ++iter) {
            CUSPARSE_CHECK(cusparseDnVecSetValues(vecX, x));
            CUSPARSE_CHECK(cusparseDnVecSetValues(vecY, y));

            CUSPARSE_CHECK(cusparseSpMV(
                cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &spmv_alpha, matA, vecX, &spmv_beta_val, vecY,
                CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, d_buffer));

            cudaMemsetAsync(d_diff, 0, sizeof(double), stream);
            launch_add_beta_convergence(y, x, num_vertices,
                beta, betas, use_betas, d_diff, stream);

            cudaMemcpyAsync(cache.h_pinned, d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            iterations = iter + 1;
            std::swap(x, y);

            if (*cache.h_pinned < epsilon) {
                converged = true;
                break;
            }
        }

        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cusparseDestroySpMat(matA);
    }

    
    if (normalize) {
        cudaMemsetAsync(d_diff, 0, sizeof(double), stream);
        launch_l2_norm_sq(x, num_vertices, d_diff, stream);
        cudaMemcpyAsync(cache.h_pinned, d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (*cache.h_pinned > 0.0) {
            double inv_norm = 1.0 / sqrt(*cache.h_pinned);
            launch_scale(x, num_vertices, inv_norm, stream);
        }
    }

    
    cudaMemcpyAsync(centralities, x, (size_t)num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);

    return {iterations, converged};
}

}  
