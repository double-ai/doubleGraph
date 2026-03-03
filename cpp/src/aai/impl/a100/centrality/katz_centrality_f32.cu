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
#include <utility>

namespace aai {

namespace {

#define BLOCK_SIZE 256


template<bool USE_BETAS>
__global__ void fused_add_beta_diff_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    const float* __restrict__ betas,
    float beta_scalar,
    float* __restrict__ diff_out,
    int n)
{
    float thread_diff = 0.0f;

    for (int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        float beta_v = USE_BETAS ? betas[idx] : beta_scalar;
        float new_val = dst[idx] + beta_v;
        dst[idx] = new_val;
        thread_diff += fabsf(new_val - src[idx]);
    }

    
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_diff += __shfl_down_sync(0xffffffff, thread_diff, offset);
    }

    if ((threadIdx.x & 31) == 0 && thread_diff > 0.0f) {
        atomicAdd(diff_out, thread_diff);
    }
}


__global__ void l2_norm_sq_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
    float thread_sum = 0.0f;
    for (int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        float v = x[idx];
        thread_sum += v * v;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    if ((threadIdx.x & 31) == 0) {
        atomicAdd(out, thread_sum);
    }
}


__global__ void scale_kernel(float* __restrict__ x, float scale, int n) {
    for (int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        x[idx] *= scale;
    }
}

void launch_fused_add_beta_diff(float* dst, const float* src, const float* betas,
                                 float beta_scalar, float* diff_out, int n,
                                 bool use_betas, cudaStream_t stream) {
    if (n == 0) return;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 2048) grid = 2048;
    if (use_betas) {
        fused_add_beta_diff_kernel<true><<<grid, BLOCK_SIZE, 0, stream>>>(
            dst, src, betas, beta_scalar, diff_out, n);
    } else {
        fused_add_beta_diff_kernel<false><<<grid, BLOCK_SIZE, 0, stream>>>(
            dst, src, betas, beta_scalar, diff_out, n);
    }
}

void launch_l2_norm_sq(const float* x, float* out, int n, cudaStream_t stream) {
    if (n == 0) return;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 2048) grid = 2048;
    l2_norm_sq_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(x, out, n);
}

void launch_scale(float* x, float scale, int n, cudaStream_t stream) {
    if (n == 0) return;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 2048) grid = 2048;
    scale_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(x, scale, n);
}

struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    float* d_scalars = nullptr;  
    float* buf_a = nullptr;
    float* buf_b = nullptr;
    float* diff_dev = nullptr;
    void* cusparse_buffer = nullptr;

    int32_t buf_a_capacity = 0;
    int32_t buf_b_capacity = 0;
    size_t cusparse_buf_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
        cudaMalloc(&d_scalars, 2 * sizeof(float));
        float h_zero = 0.0f;
        cudaMemcpy(&d_scalars[1], &h_zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&diff_dev, sizeof(float));
    }

    void ensure_bufs(int32_t n) {
        if (buf_a_capacity < n) {
            if (buf_a) cudaFree(buf_a);
            cudaMalloc(&buf_a, (size_t)n * sizeof(float));
            buf_a_capacity = n;
        }
        if (buf_b_capacity < n) {
            if (buf_b) cudaFree(buf_b);
            cudaMalloc(&buf_b, (size_t)n * sizeof(float));
            buf_b_capacity = n;
        }
    }

    void ensure_cusparse_buf(size_t size) {
        if (cusparse_buf_capacity < size) {
            if (cusparse_buffer) cudaFree(cusparse_buffer);
            cudaMalloc(&cusparse_buffer, size);
            cusparse_buf_capacity = size;
        }
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (d_scalars) cudaFree(d_scalars);
        if (buf_a) cudaFree(buf_a);
        if (buf_b) cudaFree(buf_b);
        if (diff_dev) cudaFree(diff_dev);
        if (cusparse_buffer) cudaFree(cusparse_buffer);
    }
};

}  

katz_centrality_result_t katz_centrality(const graph32_t& graph,
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

    cache.ensure_bufs(num_vertices);

    
    cudaMemcpyAsync(&cache.d_scalars[0], &alpha, sizeof(float), cudaMemcpyHostToDevice, stream);
    float* d_alpha = &cache.d_scalars[0];
    float* d_zero = &cache.d_scalars[1];

    float* a = cache.buf_a;
    float* b = cache.buf_b;
    float* diff_dev = cache.diff_dev;

    
    if (has_initial_guess) {
        cudaMemcpyAsync(a, centralities, (size_t)num_vertices * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemsetAsync(a, 0, (size_t)num_vertices * sizeof(float), stream);
    }

    
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, num_vertices, num_vertices, num_edges,
                      (void*)offsets, (void*)indices, (void*)edge_weights,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    
    cusparseDnVecDescr_t vecSrc, vecDst;
    cusparseCreateDnVec(&vecSrc, num_vertices, a, CUDA_R_32F);
    cusparseCreateDnVec(&vecDst, num_vertices, b, CUDA_R_32F);

    
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            d_alpha, matA, vecSrc, d_zero, vecDst,
                            CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &bufferSize);

    if (bufferSize > 0) {
        cache.ensure_cusparse_buf(bufferSize);
    }

    
    cusparseSpMV_preprocess(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            d_alpha, matA, vecSrc, d_zero, vecDst,
                            CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, cache.cusparse_buffer);

    bool use_betas = (betas != nullptr);

    float* src = a;
    float* dst = b;

    bool converged = false;
    size_t iterations = 0;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(diff_dev, 0, sizeof(float), stream);

        
        cusparseSpMV(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     d_alpha, matA, vecSrc, d_zero, vecDst,
                     CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, cache.cusparse_buffer);

        
        launch_fused_add_beta_diff(dst, src, betas, beta, diff_dev,
                                   num_vertices, use_betas, stream);

        iterations = iter + 1;

        
        float h_diff;
        cudaMemcpy(&h_diff, diff_dev, sizeof(float), cudaMemcpyDeviceToHost);

        if (h_diff < epsilon) {
            converged = true;
            break;
        }

        
        std::swap(src, dst);
        cusparseDnVecSetValues(vecSrc, src);
        cusparseDnVecSetValues(vecDst, dst);
    }

    
    float* result_ptr = converged ? dst : src;

    
    if (normalize) {
        cudaMemsetAsync(diff_dev, 0, sizeof(float), stream);
        launch_l2_norm_sq(result_ptr, diff_dev, num_vertices, stream);
        float h_l2_sq;
        cudaMemcpy(&h_l2_sq, diff_dev, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_l2_sq > 0.0f) {
            float inv_norm = 1.0f / sqrtf(h_l2_sq);
            launch_scale(result_ptr, inv_norm, num_vertices, stream);
        }
    }

    
    cudaMemcpyAsync(centralities, result_ptr, (size_t)num_vertices * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    
    cusparseDestroyDnVec(vecSrc);
    cusparseDestroyDnVec(vecDst);
    cusparseDestroySpMat(matA);

    return {iterations, converged};
}

}  
