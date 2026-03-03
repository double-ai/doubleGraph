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

namespace aai {

namespace {

__global__ void fused_add_beta_diff_kernel(
    double* __restrict__ x_new,
    const double* __restrict__ x_old,
    const double* __restrict__ betas,
    double beta_scalar,
    double* __restrict__ diff_out,
    int n)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double thread_diff = 0.0;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        double spmv_val = x_new[idx];
        double beta_v = (betas != nullptr) ? betas[idx] : beta_scalar;
        double new_val = spmv_val + beta_v;
        x_new[idx] = new_val;
        thread_diff += fabs(new_val - x_old[idx]);
    }

    double block_diff = BlockReduce(temp_storage).Sum(thread_diff);
    if (threadIdx.x == 0 && block_diff > 0.0) {
        atomicAdd(diff_out, block_diff);
    }
}

__global__ void l2_norm_sq_kernel(const double* __restrict__ x, double* __restrict__ result, int n) {
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double thread_sum = 0.0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        double val = x[idx];
        thread_sum += val * val;
    }

    double block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    if (threadIdx.x == 0 && block_sum > 0.0) {
        atomicAdd(result, block_sum);
    }
}

__global__ void scale_kernel(double* __restrict__ x, double scale, int n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        x[idx] *= scale;
    }
}

void launch_fused_add_beta_diff(
    double* x_new, const double* x_old,
    const double* betas, double beta_scalar,
    double* diff_out, int n, cudaStream_t stream)
{
    if (n == 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    fused_add_beta_diff_kernel<<<blocks, threads, 0, stream>>>(
        x_new, x_old, betas, beta_scalar, diff_out, n);
}

void launch_l2_norm_sq(const double* x, double* result, int n, cudaStream_t stream) {
    if (n == 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    l2_norm_sq_kernel<<<blocks, threads, 0, stream>>>(x, result, n);
}

void launch_scale(double* x, double scale, int n, cudaStream_t stream) {
    if (n == 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    scale_kernel<<<blocks, threads, 0, stream>>>(x, scale, n);
}

struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;
    double* h_diff_pinned = nullptr;
    double* buf0 = nullptr;
    double* buf1 = nullptr;
    double* d_diff = nullptr;
    void* sp_buffer = nullptr;
    int64_t buf0_capacity = 0;
    int64_t buf1_capacity = 0;
    int64_t sp_buf_capacity = 0;

    Cache() {
        cusparseCreate(&handle);
        cudaHostAlloc(&h_diff_pinned, sizeof(double), cudaHostAllocDefault);
        cudaMalloc(&d_diff, sizeof(double));
    }

    ~Cache() override {
        if (handle) cusparseDestroy(handle);
        if (h_diff_pinned) cudaFreeHost(h_diff_pinned);
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (d_diff) cudaFree(d_diff);
        if (sp_buffer) cudaFree(sp_buffer);
    }

    void ensure(int32_t n_vertices, size_t sp_size) {
        if (buf0_capacity < n_vertices) {
            if (buf0) cudaFree(buf0);
            cudaMalloc(&buf0, (size_t)n_vertices * sizeof(double));
            buf0_capacity = n_vertices;
        }
        if (buf1_capacity < n_vertices) {
            if (buf1) cudaFree(buf1);
            cudaMalloc(&buf1, (size_t)n_vertices * sizeof(double));
            buf1_capacity = n_vertices;
        }
        if (sp_size > 0 && sp_buf_capacity < (int64_t)sp_size) {
            if (sp_buffer) cudaFree(sp_buffer);
            cudaMalloc(&sp_buffer, sp_size);
            sp_buf_capacity = sp_size;
        }
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

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    cudaStream_t stream = 0;
    cusparseSetStream(cache.handle, stream);

    
    cache.ensure(num_vertices, 0);

    double* x_a = cache.buf0;
    double* x_b = cache.buf1;
    double* d_diff = cache.d_diff;

    
    if (has_initial_guess) {
        cudaMemcpyAsync(x_a, centralities,
            (size_t)num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemsetAsync(x_a, 0, (size_t)num_vertices * sizeof(double), stream);
    }

    
    cusparseSpMatDescr_t mat;
    cusparseCreateCsr(&mat, num_vertices, num_vertices, num_edges,
        (void*)offsets,
        (void*)indices,
        (void*)edge_weights,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, num_vertices, x_a, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, num_vertices, x_b, CUDA_R_64F);

    double sp_alpha = alpha;
    double sp_beta = 0.0;

    cusparseSpMVAlg_t alg = CUSPARSE_SPMV_CSR_ALG1;

    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &sp_alpha, mat, vecX, &sp_beta, vecY,
        CUDA_R_64F, alg, &bufferSize);

    
    cache.ensure(num_vertices, bufferSize);

    void* d_sp_buffer = (bufferSize > 0) ? cache.sp_buffer : nullptr;

    cusparseSpMV_preprocess(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &sp_alpha, mat, vecX, &sp_beta, vecY,
        CUDA_R_64F, alg, d_sp_buffer);

    
    bool converged = false;
    std::size_t iterations = 0;

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        cusparseDnVecSetValues(vecX, x_a);
        cusparseDnVecSetValues(vecY, x_b);

        
        cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &sp_alpha, mat, vecX, &sp_beta, vecY,
            CUDA_R_64F, alg, d_sp_buffer);

        
        cudaMemsetAsync(d_diff, 0, sizeof(double), stream);
        launch_fused_add_beta_diff(x_b, x_a, betas, beta,
            d_diff, num_vertices, stream);

        
        cudaMemcpyAsync(cache.h_diff_pinned, d_diff, sizeof(double),
            cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations = iter + 1;

        if (*cache.h_diff_pinned < epsilon) {
            converged = true;
            break;
        }

        std::swap(x_a, x_b);
    }

    double* result_ptr = converged ? x_b : x_a;

    
    if (normalize) {
        cudaMemsetAsync(d_diff, 0, sizeof(double), stream);
        launch_l2_norm_sq(result_ptr, d_diff, num_vertices, stream);
        cudaMemcpyAsync(cache.h_diff_pinned, d_diff, sizeof(double),
            cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        double h_norm_sq = *cache.h_diff_pinned;
        if (h_norm_sq > 0.0) {
            launch_scale(result_ptr, 1.0 / sqrt(h_norm_sq), num_vertices, stream);
        }
    }

    
    cudaMemcpyAsync(centralities, result_ptr,
        (size_t)num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    cusparseDestroySpMat(mat);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    return katz_centrality_result_t{iterations, converged};
}

}  
