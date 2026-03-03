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
#include <cstring>
#include <cmath>
#include <algorithm>

namespace aai {

namespace {





template<int BLOCK_SIZE, bool USE_BETAS>
__global__ void __launch_bounds__(BLOCK_SIZE)
add_beta_diff_kernel(
    float* __restrict__ x_new,
    const float* __restrict__ x_old,
    float beta,
    const float* __restrict__ betas,
    float* __restrict__ d_diff_sum,
    int32_t n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float my_diff = 0.0f;

    if (v < n) {
        float beta_v;
        if constexpr (USE_BETAS) {
            beta_v = betas[v];
        } else {
            beta_v = beta;
        }
        float val = x_new[v] + beta_v;
        x_new[v] = val;
        my_diff = fabsf(val - __ldg(&x_old[v]));
    }

    float block_diff = BlockReduce(temp).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff > 0.0f) {
        atomicAdd(d_diff_sum, block_diff);
    }
}

template<int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
compute_l2_norm_sq(const float* __restrict__ x, float* __restrict__ result, int32_t n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int32_t idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float val = (idx < n) ? x[idx] : 0.0f;

    float block_sum = BlockReduce(temp).Sum(val * val);
    if (threadIdx.x == 0 && block_sum > 0.0f) {
        atomicAdd(result, block_sum);
    }
}

__global__ void scale_kernel(float* __restrict__ x, int32_t n, float inv_norm)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= inv_norm;
    }
}





void launch_add_beta_diff(
    float* x_new, const float* x_old,
    float beta, const float* betas, bool use_betas,
    float* d_diff_sum, int32_t n, cudaStream_t stream)
{
    constexpr int BLOCK = 256;
    int grid = (n + BLOCK - 1) / BLOCK;
    if (use_betas) {
        add_beta_diff_kernel<BLOCK, true><<<grid, BLOCK, 0, stream>>>(
            x_new, x_old, beta, betas, d_diff_sum, n);
    } else {
        add_beta_diff_kernel<BLOCK, false><<<grid, BLOCK, 0, stream>>>(
            x_new, x_old, beta, betas, d_diff_sum, n);
    }
}

void launch_l2_norm_sq(const float* x, int32_t n, float* d_norm_sq, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    int grid = (n + BLOCK - 1) / BLOCK;
    compute_l2_norm_sq<BLOCK><<<grid, BLOCK, 0, stream>>>(x, d_norm_sq, n);
}

void launch_scale(float* x, int32_t n, float inv_norm, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    int grid = (n + BLOCK - 1) / BLOCK;
    scale_kernel<<<grid, BLOCK, 0, stream>>>(x, n, inv_norm);
}





struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;
    size_t l2_persist_max = 0;

    float* buf0 = nullptr;
    float* buf1 = nullptr;
    int64_t buf0_capacity = 0;
    int64_t buf1_capacity = 0;

    float* d_diff_sum = nullptr;
    float* d_alpha = nullptr;
    float* d_zero = nullptr;
    float* d_norm_sq = nullptr;

    void* spmv_buffer = nullptr;
    size_t spmv_buffer_capacity = 0;

    Cache() {
        cusparseCreate(&handle);
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);

        int dev = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        l2_persist_max = prop.persistingL2CacheMaxSize;
        if (l2_persist_max > 0) {
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_persist_max);
        }

        cudaMalloc(&d_diff_sum, sizeof(float));
        cudaMalloc(&d_alpha, sizeof(float));
        cudaMalloc(&d_zero, sizeof(float));
        cudaMalloc(&d_norm_sq, sizeof(float));
    }

    ~Cache() override {
        if (handle) cusparseDestroy(handle);
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (d_diff_sum) cudaFree(d_diff_sum);
        if (d_alpha) cudaFree(d_alpha);
        if (d_zero) cudaFree(d_zero);
        if (d_norm_sq) cudaFree(d_norm_sq);
        if (spmv_buffer) cudaFree(spmv_buffer);
    }

    void ensure_bufs(int64_t n) {
        if (buf0_capacity < n) {
            if (buf0) cudaFree(buf0);
            cudaMalloc(&buf0, n * sizeof(float));
            buf0_capacity = n;
        }
        if (buf1_capacity < n) {
            if (buf1) cudaFree(buf1);
            cudaMalloc(&buf1, n * sizeof(float));
            buf1_capacity = n;
        }
    }

    void ensure_spmv(size_t size) {
        if (spmv_buffer_capacity < size) {
            if (spmv_buffer) cudaFree(spmv_buffer);
            size_t alloc_size = std::max(size, (size_t)1);
            cudaMalloc(&spmv_buffer, alloc_size);
            spmv_buffer_capacity = alloc_size;
        }
    }
};

}  

katz_centrality_result_t katz_centrality_seg(const graph32_t& graph,
                         const float* edge_weights,
                         float* centralities,
                         float alpha,
                         float beta,
                         const float* betas,
                         float epsilon,
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

    
    cache.ensure_bufs(num_vertices);

    
    cudaMemcpyAsync(cache.d_alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice, stream);
    float zero = 0.0f;
    cudaMemcpyAsync(cache.d_zero, &zero, sizeof(float), cudaMemcpyHostToDevice, stream);

    
    if (has_initial_guess) {
        cudaMemcpyAsync(cache.buf0, centralities,
                        num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemsetAsync(cache.buf0, 0, num_vertices * sizeof(float), stream);
    }

    bool use_betas = (betas != nullptr);
    float* x_cur = cache.buf0;
    float* x_next = cache.buf1;

    
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, num_vertices, num_vertices, num_edges,
        (void*)offsets, (void*)indices,
        (void*)edge_weights,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, num_vertices, x_cur, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, num_vertices, x_next, CUDA_R_32F);

    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        cache.d_alpha, matA, vecX, cache.d_zero, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &bufferSize);

    cache.ensure_spmv(bufferSize);

    cusparseSpMV_preprocess(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        cache.d_alpha, matA, vecX, cache.d_zero, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, cache.spmv_buffer);

    
    cudaStreamAttrValue stream_attr;
    memset(&stream_attr, 0, sizeof(stream_attr));
    if (cache.l2_persist_max > 0) {
        size_t x_bytes = (size_t)num_vertices * sizeof(float);
        size_t persist_bytes = std::min(x_bytes, cache.l2_persist_max);
        stream_attr.accessPolicyWindow.base_ptr = (void*)x_cur;
        stream_attr.accessPolicyWindow.num_bytes = persist_bytes;
        stream_attr.accessPolicyWindow.hitRatio = 1.0f;
        stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
    }

    
    size_t iterations = 0;
    bool converged = false;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        cudaMemsetAsync(cache.d_diff_sum, 0, sizeof(float), stream);

        cusparseDnVecSetValues(vecX, x_cur);
        cusparseDnVecSetValues(vecY, x_next);

        cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha, matA, vecX, cache.d_zero, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, cache.spmv_buffer);

        launch_add_beta_diff(x_next, x_cur, beta, betas, use_betas,
                            cache.d_diff_sum, num_vertices, stream);

        float h_diff_sum;
        cudaMemcpy(&h_diff_sum, cache.d_diff_sum, sizeof(float), cudaMemcpyDeviceToHost);

        iterations = iter + 1;

        
        if (cache.l2_persist_max > 0) {
            stream_attr.accessPolicyWindow.base_ptr = (void*)x_next;
            cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
        }

        std::swap(x_cur, x_next);
        if (h_diff_sum < epsilon) { converged = true; break; }
    }

    
    if (cache.l2_persist_max > 0) {
        memset(&stream_attr, 0, sizeof(stream_attr));
        stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
        stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
    }

    if (normalize) {
        cudaMemsetAsync(cache.d_norm_sq, 0, sizeof(float), stream);
        launch_l2_norm_sq(x_cur, num_vertices, cache.d_norm_sq, stream);
        float h_norm_sq;
        cudaMemcpy(&h_norm_sq, cache.d_norm_sq, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_norm_sq > 0.0f) {
            float inv_norm = 1.0f / sqrtf(h_norm_sq);
            launch_scale(x_cur, num_vertices, inv_norm, stream);
        }
    }

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    
    cudaMemcpyAsync(centralities, x_cur, num_vertices * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    return {iterations, converged};
}

}  
