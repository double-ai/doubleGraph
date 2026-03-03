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

struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;
    float* d_alpha = nullptr;
    float* d_beta = nullptr;
    float* x1 = nullptr;
    float* y = nullptr;
    float* accum = nullptr;
    void* spmv_buf = nullptr;

    int64_t x1_capacity = 0;
    int64_t y_capacity = 0;
    size_t spmv_buf_capacity = 0;

    Cache() {
        cusparseCreate(&handle);
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
        cudaMalloc(&d_alpha, sizeof(float));
        cudaMalloc(&d_beta, sizeof(float));
        float h_one = 1.0f, h_zero = 0.0f;
        cudaMemcpy(d_alpha, &h_one, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, &h_zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&accum, 2 * sizeof(float));
    }

    void ensure(int64_t n, size_t spmv_size) {
        if (x1_capacity < n) {
            if (x1) cudaFree(x1);
            cudaMalloc(&x1, n * sizeof(float));
            x1_capacity = n;
        }
        if (y_capacity < n) {
            if (y) cudaFree(y);
            cudaMalloc(&y, n * sizeof(float));
            y_capacity = n;
        }
        if (spmv_size > 0 && spmv_buf_capacity < spmv_size) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, spmv_size);
            spmv_buf_capacity = spmv_size;
        }
    }

    ~Cache() override {
        if (x1) cudaFree(x1);
        if (y) cudaFree(y);
        if (accum) cudaFree(accum);
        if (spmv_buf) cudaFree(spmv_buf);
        if (d_alpha) cudaFree(d_alpha);
        if (d_beta) cudaFree(d_beta);
        if (handle) cusparseDestroy(handle);
    }
};

__global__ void init_uniform_kernel(float* __restrict__ x, int64_t n, float val) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) x[i] = val;
}

__global__ void fused_add_identity_norm_kernel(float* __restrict__ y,
                                                const float* __restrict__ x,
                                                float* __restrict__ d_norm_sq,
                                                int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    extern __shared__ float smem[];

    float local_sum = 0.0f;
    for (int64_t i = idx; i < n; i += stride) {
        float val = y[i] + x[i];
        y[i] = val;
        local_sum += val * val;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);

    if (lane == 0) smem[warp_in_block] = local_sum;
    __syncthreads();

    if (warp_in_block == 0) {
        float v = (lane < warps_per_block) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0) atomicAdd(d_norm_sq, v);
    }
}

__global__ void normalize_diff_kernel(float* __restrict__ x_new,
                                       const float* __restrict__ y,
                                       const float* __restrict__ x_old,
                                       const float* __restrict__ d_norm_sq,
                                       float* __restrict__ d_diff,
                                       int64_t n) {
    float inv_norm = rsqrtf(*d_norm_sq);

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    extern __shared__ float smem[];

    float local_diff = 0.0f;
    for (int64_t i = idx; i < n; i += stride) {
        float new_val = y[i] * inv_norm;
        x_new[i] = new_val;
        local_diff += fabsf(new_val - x_old[i]);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_diff += __shfl_down_sync(0xffffffff, local_diff, off);

    if (lane == 0) smem[warp_in_block] = local_diff;
    __syncthreads();

    if (warp_in_block == 0) {
        float v = (lane < warps_per_block) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        if (lane == 0) atomicAdd(d_diff, v);
    }
}

__global__ void normalize_only_kernel(float* __restrict__ x_new,
                                       const float* __restrict__ y,
                                       const float* __restrict__ d_norm_sq,
                                       int64_t n) {
    float inv_norm = rsqrtf(*d_norm_sq);

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < n; i += stride)
        x_new[i] = y[i] * inv_norm;
}

static void launch_init_uniform(float* x, int64_t n, float val, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    if (grid > 2048) grid = 2048;
    init_uniform_kernel<<<grid, block, 0, stream>>>(x, n, val);
}

static void launch_fused_add_identity_norm(float* y, const float* x, float* d_norm_sq,
                                     int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    if (grid > 2048) grid = 2048;
    int smem_bytes = (block / 32) * sizeof(float);
    fused_add_identity_norm_kernel<<<grid, block, smem_bytes, stream>>>(y, x, d_norm_sq, n);
}

static void launch_normalize_diff(float* x_new, const float* y, const float* x_old,
                            const float* d_norm_sq, float* d_diff,
                            int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    if (grid > 2048) grid = 2048;
    int smem_bytes = (block / 32) * sizeof(float);
    normalize_diff_kernel<<<grid, block, smem_bytes, stream>>>(x_new, y, x_old, d_norm_sq, d_diff, n);
}

static void launch_normalize_only(float* x_new, const float* y, const float* d_norm_sq,
                            int64_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    if (grid > 2048) grid = 2048;
    normalize_only_kernel<<<grid, block, 0, stream>>>(x_new, y, d_norm_sq, n);
}

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
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const float* d_weights = edge_weights;

    int64_t n = num_vertices;
    int64_t nnz = num_edges;
    cudaStream_t stream = 0;
    cusparseSetStream(cache.handle, stream);

    if (n == 0) {
        return {0, true};
    }

    cache.ensure(n, 0);

    float* x_cur = centralities;
    float* x_next = cache.x1;
    float* y = cache.y;
    float* d_norm_sq = cache.accum;
    float* d_diff = cache.accum + 1;

    if (initial_centralities) {
        cudaMemcpyAsync(x_cur, initial_centralities, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        launch_init_uniform(x_cur, n, 1.0f / (float)n, stream);
    }

    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, n, n, nnz,
        (void*)d_offsets, (void*)d_indices, (void*)d_weights,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX0, vecX1, vecY;
    cusparseCreateDnVec(&vecX0, n, centralities, CUDA_R_32F);
    cusparseCreateDnVec(&vecX1, n, cache.x1, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, n, y, CUDA_R_32F);

    float h_alpha = 1.0f, h_beta = 0.0f;
    size_t bufferSize = 0;
    cusparseSetPointerMode(cache.handle, CUSPARSE_POINTER_MODE_HOST);
    cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha, matA, vecX0, &h_beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

    cache.ensure(n, bufferSize);

    cusparseSpMV_preprocess(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha, matA, vecX0, &h_beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);
    cusparseSetPointerMode(cache.handle, CUSPARSE_POINTER_MODE_DEVICE);

    size_t iterations = 0;
    bool converged = false;
    float threshold = (float)n * epsilon;
    const int CONV_CHECK_FREQ = 2;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        cusparseDnVecDescr_t vecX = (x_cur == centralities) ? vecX0 : vecX1;

        cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha, matA, vecX, cache.d_beta, vecY, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

        cudaMemsetAsync(d_norm_sq, 0, sizeof(float), stream);
        launch_fused_add_identity_norm(y, x_cur, d_norm_sq, n, stream);

        bool check_conv = ((iter + 1) % CONV_CHECK_FREQ == 0) || (iter + 1 == max_iterations);

        if (check_conv) {
            cudaMemsetAsync(d_diff, 0, sizeof(float), stream);
            launch_normalize_diff(x_next, y, x_cur, d_norm_sq, d_diff, n, stream);
            iterations = iter + 1;

            float h_diff;
            cudaMemcpyAsync(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            float* tmp = x_cur; x_cur = x_next; x_next = tmp;
            if (h_diff < threshold) { converged = true; break; }
        } else {
            launch_normalize_only(x_next, y, d_norm_sq, n, stream);
            iterations = iter + 1;
            float* tmp = x_cur; x_cur = x_next; x_next = tmp;
        }
    }

    if (x_cur != centralities) {
        cudaMemcpyAsync(centralities, x_cur, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX0);
    cusparseDestroyDnVec(vecX1);
    cusparseDestroyDnVec(vecY);

    return {iterations, converged};
}

}  
