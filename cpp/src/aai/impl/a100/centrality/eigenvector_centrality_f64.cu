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

namespace aai {

namespace {

constexpr int BLOCK_SIZE = 256;

__global__ void add_identity_and_l2_kernel(
    double* __restrict__ y,
    const double* __restrict__ x,
    int n,
    double* __restrict__ l2_sq)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double thread_sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        double yi = y[i] + x[i];
        y[i] = yi;
        thread_sum += yi * yi;
    }

    double block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    if (threadIdx.x == 0) {
        atomicAdd(l2_sq, block_sum);
    }
}

__global__ void normalize_and_diff_kernel(
    const double* __restrict__ y,
    double* __restrict__ x,
    int n,
    const double* __restrict__ l2_sq,
    double* __restrict__ diff)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double inv_norm = rsqrt(*l2_sq);

    double thread_diff = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        double x_new = y[i] * inv_norm;
        thread_diff += fabs(x_new - x[i]);
        x[i] = x_new;
    }

    double block_diff = BlockReduce(temp_storage).Sum(thread_diff);
    if (threadIdx.x == 0) {
        atomicAdd(diff, block_diff);
    }
}

__global__ void init_uniform_kernel(double* x, int n) {
    double val = 1.0 / (double)n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        x[i] = val;
    }
}

struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    double* d_l2_sq = nullptr;
    double* d_diff = nullptr;
    double* d_alpha = nullptr;
    double* d_beta = nullptr;
    void* d_spmv_buffer = nullptr;
    size_t spmv_buffer_capacity = 0;
    double* d_y = nullptr;
    int64_t y_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cudaMalloc(&d_l2_sq, sizeof(double));
        cudaMalloc(&d_diff, sizeof(double));
        cudaMalloc(&d_alpha, sizeof(double));
        cudaMalloc(&d_beta, sizeof(double));

        double h_alpha = 1.0, h_beta = 0.0;
        cudaMemcpy(d_alpha, &h_alpha, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, &h_beta, sizeof(double), cudaMemcpyHostToDevice);
    }

    void ensure_y(int64_t n) {
        if (y_capacity < n) {
            if (d_y) cudaFree(d_y);
            cudaMalloc(&d_y, n * sizeof(double));
            y_capacity = n;
        }
    }

    void ensure_spmv_buffer(size_t size) {
        if (spmv_buffer_capacity < size) {
            if (d_spmv_buffer) cudaFree(d_spmv_buffer);
            cudaMalloc(&d_spmv_buffer, size);
            spmv_buffer_capacity = size;
        }
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (d_l2_sq) cudaFree(d_l2_sq);
        if (d_diff) cudaFree(d_diff);
        if (d_alpha) cudaFree(d_alpha);
        if (d_beta) cudaFree(d_beta);
        if (d_spmv_buffer) cudaFree(d_spmv_buffer);
        if (d_y) cudaFree(d_y);
    }
};

}  

eigenvector_centrality_result_t eigenvector_centrality(const graph32_t& graph,
                            const double* edge_weights,
                            double* centralities,
                            double epsilon,
                            std::size_t max_iterations,
                            const double* initial_centralities) {
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    if (num_vertices == 0) {
        return {0, true};
    }

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    cudaStream_t stream = 0;
    cusparseSetStream(cache.cusparse_handle, stream);

    cache.ensure_y(num_vertices);
    double* d_y = cache.d_y;
    double* d_x = centralities;

    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(d_x, initial_centralities,
                        (size_t)num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else {
        int grid = min((num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE, 2048);
        init_uniform_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_x, num_vertices);
    }

    cusparseSpMatDescr_t mat_descr;
    cusparseCreateCsr(&mat_descr,
        (int64_t)num_vertices, (int64_t)num_vertices, (int64_t)num_edges,
        (void*)graph.offsets,
        (void*)graph.indices,
        (void*)edge_weights,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseDnVecDescr_t vec_x_descr, vec_y_descr;
    cusparseCreateDnVec(&vec_x_descr, (int64_t)num_vertices, d_x, CUDA_R_64F);
    cusparseCreateDnVec(&vec_y_descr, (int64_t)num_vertices, d_y, CUDA_R_64F);

    double h_alpha = 1.0, h_beta = 0.0;
    size_t buffer_size = 0;
    cusparseSpMV_bufferSize(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha, mat_descr, vec_x_descr,
        &h_beta, vec_y_descr,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
        &buffer_size);

    cache.ensure_spmv_buffer(buffer_size);

    cusparseSpMV_preprocess(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha, mat_descr, vec_x_descr,
        &h_beta, vec_y_descr,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
        cache.d_spmv_buffer);

    cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);

    double threshold = (double)num_vertices * epsilon;
    bool converged = false;
    size_t actual_iterations = 0;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        cudaMemsetAsync(cache.d_l2_sq, 0, sizeof(double), stream);
        cudaMemsetAsync(cache.d_diff, 0, sizeof(double), stream);

        cusparseSpMV(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha, mat_descr, vec_x_descr,
            cache.d_beta, vec_y_descr,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            cache.d_spmv_buffer);

        int grid = min((num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE, 2048);
        add_identity_and_l2_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_y, d_x, num_vertices, cache.d_l2_sq);

        normalize_and_diff_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_y, d_x, num_vertices, cache.d_l2_sq, cache.d_diff);

        actual_iterations = iter + 1;

        double h_diff;
        cudaMemcpy(&h_diff, cache.d_diff, sizeof(double), cudaMemcpyDeviceToHost);
        if (h_diff < threshold) {
            converged = true;
            break;
        }
    }

    cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_HOST);

    cusparseDestroySpMat(mat_descr);
    cusparseDestroyDnVec(vec_x_descr);
    cusparseDestroyDnVec(vec_y_descr);

    return {actual_iterations, converged};
}

}  
