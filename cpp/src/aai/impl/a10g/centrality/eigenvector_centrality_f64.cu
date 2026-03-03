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
#include <math_constants.h>
#include <cstdint>
#include <algorithm>

namespace aai {

namespace {

static constexpr int BLOCK_SIZE = 256;
static constexpr int MAX_BLOCKS = 256;

struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;

    float* d_alpha = nullptr;
    float* d_beta = nullptr;
    float* d_partials = nullptr;
    float* d_norm_sq = nullptr;
    float* d_l1diff = nullptr;
    unsigned int* d_counter = nullptr;

    void* d_spmv_buffer = nullptr;
    size_t spmv_buffer_size = 0;

    float* h_l1diff = nullptr;

    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_w = nullptr;
    int64_t x_capacity = 0;
    int64_t y_capacity = 0;
    int64_t w_capacity = 0;

    Cache() {
        cusparseCreate(&handle);

        cudaMalloc(&d_alpha, sizeof(float));
        cudaMalloc(&d_beta, sizeof(float));
        float h_alpha = 1.0f, h_beta = 0.0f;
        cudaMemcpy(d_alpha, &h_alpha, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, &h_beta, sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_partials, MAX_BLOCKS * sizeof(float));
        cudaMalloc(&d_norm_sq, sizeof(float));
        cudaMalloc(&d_l1diff, sizeof(float));
        cudaMalloc(&d_counter, sizeof(unsigned int));
        cudaMemset(d_counter, 0, sizeof(unsigned int));

        cudaHostAlloc(&h_l1diff, sizeof(float), cudaHostAllocDefault);
    }

    void ensure_buffers(int32_t N, int32_t nnz) {
        if (x_capacity < N) {
            if (d_x) cudaFree(d_x);
            cudaMalloc(&d_x, (int64_t)N * sizeof(float));
            x_capacity = N;
        }
        if (y_capacity < N) {
            if (d_y) cudaFree(d_y);
            cudaMalloc(&d_y, (int64_t)N * sizeof(float));
            y_capacity = N;
        }
        if (w_capacity < nnz) {
            if (d_w) cudaFree(d_w);
            cudaMalloc(&d_w, (int64_t)nnz * sizeof(float));
            w_capacity = nnz;
        }
    }

    void ensure_spmv_buffer(size_t needed) {
        if (needed > spmv_buffer_size) {
            if (d_spmv_buffer) cudaFree(d_spmv_buffer);
            cudaMalloc(&d_spmv_buffer, needed);
            spmv_buffer_size = needed;
        }
    }

    ~Cache() override {
        if (handle) cusparseDestroy(handle);
        if (d_alpha) cudaFree(d_alpha);
        if (d_beta) cudaFree(d_beta);
        if (d_partials) cudaFree(d_partials);
        if (d_norm_sq) cudaFree(d_norm_sq);
        if (d_l1diff) cudaFree(d_l1diff);
        if (d_counter) cudaFree(d_counter);
        if (d_spmv_buffer) cudaFree(d_spmv_buffer);
        if (h_l1diff) cudaFreeHost(h_l1diff);
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        if (d_w) cudaFree(d_w);
    }
};

__global__ void init_uniform_kernel(float* __restrict__ x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        x[idx] = 1.0f / N;
    }
}

__global__ void init_degree_kernel(
    float* __restrict__ x,
    const int32_t* __restrict__ offsets,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float deg = (float)(offsets[idx + 1] - offsets[idx]) + 1.0f;
        x[idx] = deg;
    }
}

__global__ __launch_bounds__(BLOCK_SIZE)
void l2_normalize_kernel(
    float* __restrict__ x,
    float* __restrict__ partials,
    float* __restrict__ d_norm_sq,
    unsigned int* __restrict__ retirement_counter,
    int N,
    int phase
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * BLOCK_SIZE + tid;
    int stride = gridDim.x * BLOCK_SIZE;

    if (phase == 0) {
        float local_l2sq = 0.0f;
        for (int i = gid; i < N; i += stride) {
            float v = x[i];
            local_l2sq += v * v;
        }

        typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp;
        float block_sum = BlockReduce(temp).Sum(local_l2sq);

        if (tid == 0) partials[bid] = block_sum;
        __threadfence();

        __shared__ bool am_last;
        if (tid == 0) {
            unsigned int ticket = atomicAdd(retirement_counter, 1);
            am_last = (ticket == gridDim.x - 1);
        }
        __syncthreads();

        if (am_last) {
            float total = 0.0f;
            for (int i = tid; i < (int)gridDim.x; i += BLOCK_SIZE)
                total += partials[i];
            float reduced = BlockReduce(temp).Sum(total);
            if (tid == 0) {
                *d_norm_sq = reduced;
                *retirement_counter = 0;
            }
        }
    } else {
        float inv_norm = rsqrtf(__ldg(d_norm_sq));
        for (int i = gid; i < N; i += stride) {
            x[i] *= inv_norm;
        }
    }
}

__global__ void convert_d2f_kernel(const double* __restrict__ src, float* __restrict__ dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = __double2float_rn(src[idx]);
    }
}

__global__ void convert_f2d_kernel(const float* __restrict__ src, double* __restrict__ dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        dst[idx] = (double)src[idx];
    }
}

__global__ __launch_bounds__(BLOCK_SIZE)
void fused_identity_l2_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    float* __restrict__ partials,
    float* __restrict__ d_norm_sq,
    unsigned int* __restrict__ retirement_counter,
    int N
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * BLOCK_SIZE + tid;
    int stride = gridDim.x * BLOCK_SIZE;

    float local_l2sq = 0.0f;
    for (int i = gid; i < N; i += stride) {
        float x_val = __ldg(&x[i]);
        float val = y[i] + x_val;
        y[i] = val;
        local_l2sq += val * val;
    }

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    float block_sum = BlockReduce(temp).Sum(local_l2sq);

    if (tid == 0) partials[bid] = block_sum;
    __threadfence();

    __shared__ bool am_last;
    if (tid == 0) {
        unsigned int ticket = atomicAdd(retirement_counter, 1);
        am_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (am_last) {
        float total = 0.0f;
        for (int i = tid; i < (int)gridDim.x; i += BLOCK_SIZE)
            total += partials[i];
        float reduced = BlockReduce(temp).Sum(total);
        if (tid == 0) {
            *d_norm_sq = reduced;
            *retirement_counter = 0;
        }
    }
}

__global__ __launch_bounds__(BLOCK_SIZE)
void fused_normalize_l1diff_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ d_norm_sq,
    float* __restrict__ partials,
    float* __restrict__ d_l1diff,
    unsigned int* __restrict__ retirement_counter,
    int N
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * BLOCK_SIZE + tid;
    int stride = gridDim.x * BLOCK_SIZE;

    float norm_sq = __ldg(d_norm_sq);
    float inv_norm = (norm_sq > 0.0f) ? rsqrtf(norm_sq) : 0.0f;

    float local_l1diff = 0.0f;
    for (int i = gid; i < N; i += stride) {
        float y_val = y[i] * inv_norm;
        y[i] = y_val;
        float x_val = __ldg(&x[i]);
        local_l1diff += fabsf(y_val - x_val);
    }

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    float block_sum = BlockReduce(temp).Sum(local_l1diff);

    if (tid == 0) partials[bid] = block_sum;
    __threadfence();

    __shared__ bool am_last;
    if (tid == 0) {
        unsigned int ticket = atomicAdd(retirement_counter, 1);
        am_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (am_last) {
        float total = 0.0f;
        for (int i = tid; i < (int)gridDim.x; i += BLOCK_SIZE)
            total += partials[i];
        float reduced = BlockReduce(temp).Sum(total);
        if (tid == 0) {
            *d_l1diff = reduced;
            *retirement_counter = 0;
        }
    }
}

}  

eigenvector_centrality_result_t eigenvector_centrality(const graph32_t& graph,
                            const double* edge_weights,
                            double* centralities,
                            double epsilon,
                            std::size_t max_iterations,
                            const double* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t N = graph.number_of_vertices;
    int32_t nnz = graph.number_of_edges;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    cudaStream_t stream = 0;
    cusparseSetStream(cache.handle, stream);

    cache.ensure_buffers(N, nnz);

    float* d_x = cache.d_x;
    float* d_y = cache.d_y;
    float* d_w = cache.d_w;

    int blocks_cvt = (nnz + BLOCK_SIZE - 1) / BLOCK_SIZE;
    convert_d2f_kernel<<<blocks_cvt, BLOCK_SIZE, 0, stream>>>(edge_weights, d_w, nnz);

    int num_blocks = std::min(MAX_BLOCKS, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (initial_centralities != nullptr) {
        int blocks_init = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        convert_d2f_kernel<<<blocks_init, BLOCK_SIZE, 0, stream>>>(initial_centralities, d_x, N);
    } else {
        int blocks_init = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_degree_kernel<<<blocks_init, BLOCK_SIZE, 0, stream>>>(d_x, offsets, N);
        l2_normalize_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            d_x, cache.d_partials, cache.d_norm_sq, cache.d_counter, N, 0);
        l2_normalize_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            d_x, cache.d_partials, cache.d_norm_sq, cache.d_counter, N, 1);
    }

    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(
        &matA, N, N, nnz,
        (void*)offsets,
        (void*)indices,
        (void*)d_w,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F
    );

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, N, d_y, CUDA_R_32F);

    float h_alpha = 1.0f, h_beta = 0.0f;
    size_t needed;
    cusparseSpMV_bufferSize(
        cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha, matA, vecX, &h_beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, &needed
    );

    cache.ensure_spmv_buffer(needed);

    cusparseSpMV_preprocess(
        cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha, matA, vecX, &h_beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, cache.d_spmv_buffer
    );

    cusparseSetPointerMode(cache.handle, CUSPARSE_POINTER_MODE_DEVICE);

    float threshold = (float)(N * epsilon);
    size_t iterations = 0;
    bool converged = false;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        cusparseDnVecSetValues(vecX, d_x);
        cusparseDnVecSetValues(vecY, d_y);

        cusparseSpMV(
            cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha, matA, vecX, cache.d_beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, cache.d_spmv_buffer
        );

        fused_identity_l2_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            d_y, d_x, cache.d_partials, cache.d_norm_sq,
            cache.d_counter, N);

        fused_normalize_l1diff_kernel<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            d_y, d_x, cache.d_norm_sq, cache.d_partials,
            cache.d_l1diff, cache.d_counter, N);

        iterations++;

        cudaMemcpyAsync(cache.h_l1diff, cache.d_l1diff, sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::swap(d_x, d_y);

        if (*cache.h_l1diff < threshold) {
            converged = true;
            break;
        }
    }

    cusparseSetPointerMode(cache.handle, CUSPARSE_POINTER_MODE_HOST);

    int blocks_out = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    convert_f2d_kernel<<<blocks_out, BLOCK_SIZE, 0, stream>>>(d_x, centralities, N);

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    return {iterations, converged};
}

}  
