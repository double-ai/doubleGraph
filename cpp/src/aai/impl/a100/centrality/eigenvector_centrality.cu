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
#include <cub/cub.cuh>

namespace aai {

namespace {

#define BLOCK_SIZE 256

__global__ void fill_kernel(float* arr, int N, float val) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        arr[i] = val;
    }
}


__global__ void add_identity_and_normsq_kernel(float* __restrict__ y,
                                                const float* __restrict__ x,
                                                int N,
                                                float* __restrict__ norm2_out) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage temp;

    float local_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        float yi = y[i] + x[i];
        y[i] = yi;
        local_sum += yi * yi;
    }

    float block_sum = BR(temp).Sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(norm2_out, block_sum);
    }
}


__global__ void normalize_and_diff_kernel(float* __restrict__ y,
                                           const float* __restrict__ x,
                                           int N,
                                           const float* __restrict__ norm2_ptr,
                                           float* __restrict__ diff_out) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage temp;

    float norm2_val = *norm2_ptr;
    float inv_norm = (norm2_val > 0.0f) ? rsqrtf(norm2_val) : 0.0f;

    float local_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        float yi = y[i] * inv_norm;
        y[i] = yi;
        local_sum += fabsf(yi - x[i]);
    }

    float block_sum = BR(temp).Sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(diff_out, block_sum);
    }
}

static void launch_fill(float* arr, int N, float val, cudaStream_t stream) {
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 1024) blocks = 1024;
    fill_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(arr, N, val);
}

static void launch_add_identity_and_normsq(float* y, const float* x, int N,
                                     float* norm2_out,
                                     int num_blocks, cudaStream_t stream) {
    add_identity_and_normsq_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(y, x, N, norm2_out);
}

static void launch_normalize_and_diff(float* y, const float* x, int N,
                                const float* norm2_ptr,
                                float* diff_out,
                                int num_blocks, cudaStream_t stream) {
    normalize_and_diff_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(y, x, N, norm2_ptr, diff_out);
}

struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    float* values = nullptr;
    float* x = nullptr;
    float* y = nullptr;
    float* norm2 = nullptr;
    float* diff = nullptr;
    void* spmv_buffer = nullptr;

    int64_t values_capacity = 0;
    int64_t x_capacity = 0;
    int64_t y_capacity = 0;
    size_t spmv_buffer_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cudaMalloc(&norm2, sizeof(float));
        cudaMalloc(&diff, sizeof(float));
    }

    void ensure(int64_t N, int64_t E) {
        if (values_capacity < E) {
            if (values) cudaFree(values);
            cudaMalloc(&values, E * sizeof(float));
            values_capacity = E;
        }
        if (x_capacity < N) {
            if (x) cudaFree(x);
            cudaMalloc(&x, N * sizeof(float));
            x_capacity = N;
        }
        if (y_capacity < N) {
            if (y) cudaFree(y);
            cudaMalloc(&y, N * sizeof(float));
            y_capacity = N;
        }
    }

    void ensure_spmv_buffer(size_t size) {
        if (spmv_buffer_capacity < size) {
            if (spmv_buffer) cudaFree(spmv_buffer);
            cudaMalloc(&spmv_buffer, size);
            spmv_buffer_capacity = size;
        }
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (values) cudaFree(values);
        if (x) cudaFree(x);
        if (y) cudaFree(y);
        if (norm2) cudaFree(norm2);
        if (diff) cudaFree(diff);
        if (spmv_buffer) cudaFree(spmv_buffer);
    }
};

}  

eigenvector_centrality_result_t eigenvector_centrality(const graph32_t& graph,
                            float* centralities,
                            float epsilon,
                            std::size_t max_iterations,
                            const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int64_t N = graph.number_of_vertices;
    int64_t E = graph.number_of_edges;
    cudaStream_t stream = nullptr;

    cache.ensure(N, E);

    float* d_values = cache.values;
    float* d_x = cache.x;
    float* d_y = cache.y;
    float* d_norm2 = cache.norm2;
    float* d_diff = cache.diff;

    
    launch_fill(d_values, (int)E, 1.0f, stream);

    
    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(d_x, initial_centralities,
                       N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        float init_val = 1.0f / (float)N;
        launch_fill(d_x, (int)N, init_val, stream);
    }

    
    
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, N, N, E,
        (void*)d_offsets, (void*)d_indices, (void*)d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, N, d_y, CUDA_R_32F);

    
    float h_alpha = 1.0f, h_beta = 0.0f;

    
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(
        cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha, matA, vecX, &h_beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

    if (bufferSize > 0) {
        cache.ensure_spmv_buffer(bufferSize);
    }

    
    cusparseSpMV_preprocess(
        cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha, matA, vecX, &h_beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buffer);

    
    int num_blocks = std::min(512, (int)((N + 255) / 256));
    if (num_blocks < 1) num_blocks = 1;

    
    float threshold = (float)N * epsilon;
    bool converged = false;
    size_t iterations = 0;

    float* x_ptr = d_x;
    float* y_ptr = d_y;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        
        cusparseDnVecSetValues(vecX, x_ptr);
        cusparseDnVecSetValues(vecY, y_ptr);

        
        cusparseSpMV(
            cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_alpha, matA, vecX, &h_beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buffer);

        
        cudaMemsetAsync(d_norm2, 0, sizeof(float), stream);
        launch_add_identity_and_normsq(y_ptr, x_ptr, (int)N, d_norm2, num_blocks, stream);

        
        cudaMemsetAsync(d_diff, 0, sizeof(float), stream);
        launch_normalize_and_diff(y_ptr, x_ptr, (int)N, d_norm2, d_diff, num_blocks, stream);

        iterations = iter + 1;

        
        float h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);

        if (h_diff < threshold) {
            converged = true;
            break;
        }

        
        std::swap(x_ptr, y_ptr);
    }

    
    
    
    float* result_ptr = converged ? y_ptr : x_ptr;

    
    cudaMemcpy(centralities, result_ptr, N * sizeof(float), cudaMemcpyDeviceToDevice);

    
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    return {iterations, converged};
}

}  
