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
#include <limits>

namespace aai {

namespace {

constexpr int BLOCK = 256;





__global__ void init_degree_kernel(const int* __restrict__ offsets, float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int deg = offsets[i + 1] - offsets[i];
        x[i] = sqrtf((float)(deg + 1));
    }
}

__global__ void l2_norm_sq_kernel(const float* __restrict__ x, float* __restrict__ norm_sq, int n) {
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage tmp;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? x[i] : 0.0f;
    float block_sum = BR(tmp).Sum(val * val);
    if (threadIdx.x == 0 && block_sum != 0.0f) atomicAdd(norm_sq, block_sum);
}

__global__ void normalize_inplace_kernel(float* __restrict__ x, const float* __restrict__ norm_sq, int n) {
    float ns = *norm_sq;
    float inv = (ns > 0.0f) ? rsqrtf(ns) : 0.0f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= inv;
}

__global__ void fused_spmv_identity_norm_kernel(
    const int* __restrict__ offsets, const int* __restrict__ indices,
    const float* __restrict__ weights, const float* __restrict__ x,
    float* __restrict__ y, float* __restrict__ norm_sq, int n)
{
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage tmp;

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (v < n) {
        int start = offsets[v];
        int end = offsets[v + 1];
        float sum = x[v]; 
        for (int e = start; e < end; e++) {
            sum += weights[e] * x[indices[e]];
        }
        val = sum;
        y[v] = val;
    }

    float block_sum = BR(tmp).Sum(val * val);
    if (threadIdx.x == 0 && block_sum != 0.0f) atomicAdd(norm_sq, block_sum);
}

__global__ void compute_norm_kernel(const float* __restrict__ y, float* __restrict__ norm_sq, int n) {
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage tmp;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < n) ? y[i] : 0.0f;
    float block_sum = BR(tmp).Sum(val * val);
    if (threadIdx.x == 0 && block_sum != 0.0f) atomicAdd(norm_sq, block_sum);
}

__global__ void identity_add_norm_kernel(float* __restrict__ y, const float* __restrict__ x,
                                          float* __restrict__ norm_sq, int n) {
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage tmp;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (i < n) {
        val = y[i] + x[i];
        y[i] = val;
    }
    float block_sum = BR(tmp).Sum(val * val);
    if (threadIdx.x == 0 && block_sum != 0.0f) atomicAdd(norm_sq, block_sum);
}

__global__ void normalize_diff_kernel(const float* __restrict__ y, float* __restrict__ x,
                                       const float* __restrict__ norm_sq, float* __restrict__ diff, int n) {
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage tmp;
    float ns = *norm_sq;
    float inv = (ns > 0.0f) ? rsqrtf(ns) : 0.0f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float d = 0.0f;
    if (i < n) {
        float new_val = y[i] * inv;
        d = fabsf(new_val - x[i]);
        x[i] = new_val;
    }
    float block_sum = BR(tmp).Sum(d);
    if (threadIdx.x == 0 && block_sum != 0.0f) atomicAdd(diff, block_sum);
}

__global__ void normalize_store_kernel(const float* __restrict__ y, float* __restrict__ x,
                                        const float* __restrict__ norm_sq, int n) {
    float ns = *norm_sq;
    float inv = (ns > 0.0f) ? rsqrtf(ns) : 0.0f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = y[i] * inv;
}





struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;
    float* d_norm_sq = nullptr;
    float* d_diff = nullptr;
    float* d_alpha = nullptr;
    float* d_beta_one = nullptr;
    float* h_diff_pinned = nullptr;
    float* d_y = nullptr;
    void* d_spmv_buffer = nullptr;

    int32_t y_capacity = 0;
    size_t spmv_buffer_capacity = 0;

    Cache() {
        cusparseCreate(&handle);
        cudaMalloc(&d_norm_sq, sizeof(float));
        cudaMalloc(&d_diff, sizeof(float));
        cudaMalloc(&d_alpha, sizeof(float));
        cudaMalloc(&d_beta_one, sizeof(float));
        float one = 1.0f;
        cudaMemcpy(d_alpha, &one, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta_one, &one, sizeof(float), cudaMemcpyHostToDevice);
        cudaMallocHost(&h_diff_pinned, sizeof(float));
    }

    void ensure_y(int32_t n) {
        if (y_capacity < n) {
            if (d_y) cudaFree(d_y);
            cudaMalloc(&d_y, (size_t)n * sizeof(float));
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
        if (handle) cusparseDestroy(handle);
        if (d_norm_sq) cudaFree(d_norm_sq);
        if (d_diff) cudaFree(d_diff);
        if (d_alpha) cudaFree(d_alpha);
        if (d_beta_one) cudaFree(d_beta_one);
        if (h_diff_pinned) cudaFreeHost(h_diff_pinned);
        if (d_y) cudaFree(d_y);
        if (d_spmv_buffer) cudaFree(d_spmv_buffer);
    }
};

}  

eigenvector_centrality_result_t eigenvector_centrality(const graph32_t& graph,
                            const float* edge_weights,
                            float* centralities,
                            float epsilon,
                            std::size_t max_iterations,
                            const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n = graph.number_of_vertices;
    int32_t nnz = graph.number_of_edges;

    cache.ensure_y(n);
    float* d_x = centralities;
    float* d_y = cache.d_y;

    
    if (initial_centralities) {
        cudaMemcpy(d_x, initial_centralities, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        int grid = (n + BLOCK - 1) / BLOCK;
        init_degree_kernel<<<grid, BLOCK>>>(d_offsets, d_x, n);
    }

    
    {
        int grid = (n + BLOCK - 1) / BLOCK;
        cudaMemsetAsync(cache.d_norm_sq, 0, sizeof(float));
        l2_norm_sq_kernel<<<grid, BLOCK>>>(d_x, cache.d_norm_sq, n);
        normalize_inplace_kernel<<<grid, BLOCK>>>(d_x, cache.d_norm_sq, n);
    }

    
    float avg_degree = (n > 0) ? (float)nnz / n : 0.0f;
    bool use_custom_spmv = (avg_degree < 4.0f);

    
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;

    if (!use_custom_spmv) {
        cusparseCreateCsr(&matA, (int64_t)n, (int64_t)n, (int64_t)nnz,
            (void*)d_offsets, (void*)d_indices, (void*)edge_weights,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        cusparseCreateDnVec(&vecX, (int64_t)n, d_x, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, (int64_t)n, d_y, CUDA_R_32F);

        cusparseSetPointerMode(cache.handle, CUSPARSE_POINTER_MODE_HOST);
        float h_alpha = 1.0f, h_beta = 1.0f;
        size_t bufSize = 0;
        cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_alpha, matA, vecX, &h_beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, &bufSize);

        cache.ensure_spmv_buffer(bufSize);

        cusparseSpMV_preprocess(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_alpha, matA, vecX, &h_beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, cache.d_spmv_buffer);

        cusparseSetPointerMode(cache.handle, CUSPARSE_POINTER_MODE_DEVICE);
    }

    float threshold = (float)n * epsilon;
    bool converged = false;
    size_t iterations = 0;
    int check_interval = (n > 100000) ? 1 : 4;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        if (use_custom_spmv) {
            int grid = (n + BLOCK - 1) / BLOCK;
            cudaMemsetAsync(cache.d_norm_sq, 0, sizeof(float));
            fused_spmv_identity_norm_kernel<<<grid, BLOCK>>>(d_offsets, d_indices, edge_weights, d_x, d_y, cache.d_norm_sq, n);
        } else {
            cudaMemcpyAsync(d_y, d_x, (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice);
            cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                cache.d_alpha, matA, vecX, cache.d_beta_one, vecY,
                CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, cache.d_spmv_buffer);
            int grid = (n + BLOCK - 1) / BLOCK;
            cudaMemsetAsync(cache.d_norm_sq, 0, sizeof(float));
            compute_norm_kernel<<<grid, BLOCK>>>(d_y, cache.d_norm_sq, n);
        }

        bool check = ((iter + 1) % check_interval == 0) || (iter + 1 >= max_iterations);

        if (check) {
            int grid = (n + BLOCK - 1) / BLOCK;
            cudaMemsetAsync(cache.d_diff, 0, sizeof(float));
            normalize_diff_kernel<<<grid, BLOCK>>>(d_y, d_x, cache.d_norm_sq, cache.d_diff, n);
            cudaMemcpy(cache.h_diff_pinned, cache.d_diff, sizeof(float), cudaMemcpyDeviceToHost);

            iterations = iter + 1;
            if (*cache.h_diff_pinned < threshold) {
                converged = true;
                break;
            }
        } else {
            int grid = (n + BLOCK - 1) / BLOCK;
            normalize_store_kernel<<<grid, BLOCK>>>(d_y, d_x, cache.d_norm_sq, n);
        }
        iterations = iter + 1;
    }

    if (matA) cusparseDestroySpMat(matA);
    if (vecX) cusparseDestroyDnVec(vecX);
    if (vecY) cusparseDestroyDnVec(vecY);

    return {iterations, converged};
}

}  
