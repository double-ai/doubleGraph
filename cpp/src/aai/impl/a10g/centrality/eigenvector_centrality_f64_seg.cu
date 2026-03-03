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

namespace aai {

namespace {



__device__ __forceinline__ double warp_reduce_sum_d(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void convert_d2f_kernel(const double* __restrict__ src, float* __restrict__ dst, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        dst[i] = (float)src[i];
    }
}

__global__ void convert_f2d_kernel(const float* __restrict__ src, double* __restrict__ dst, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        dst[i] = (double)src[i];
    }
}

__global__ void init_uniform_kernel(float* __restrict__ x, int n) {
    float val = 1.0f / (float)n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        x[i] = val;
    }
}

__global__ void add_identity_and_norm_kernel(
    float* __restrict__ y,
    const float* __restrict__ x_old,
    int n,
    double* __restrict__ g_norm_sq
) {
    __shared__ double sdata[32]; 

    double local_sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float yi = y[i] + x_old[i];
        y[i] = yi;
        local_sum += (double)yi * (double)yi;
    }

    
    double warp_sum = warp_reduce_sum_d(local_sum);

    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    if (lane == 0) sdata[warp_id] = warp_sum;
    __syncthreads();

    
    int num_warps = blockDim.x >> 5;
    if (warp_id == 0) {
        double val = (lane < num_warps) ? sdata[lane] : 0.0;
        val = warp_reduce_sum_d(val);
        if (lane == 0) {
            atomicAdd(g_norm_sq, val);
        }
    }
}

__global__ void normalize_and_delta_kernel(
    const float* __restrict__ y,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    int n,
    const double* __restrict__ g_norm_sq,
    double* __restrict__ g_delta
) {
    __shared__ double sdata[32];

    double norm_sq = *g_norm_sq;
    double inv_norm = (norm_sq > 0.0) ? rsqrt(norm_sq) : 0.0;

    double local_delta = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        double yi = (double)y[i];
        float xi_new = (float)(yi * inv_norm);
        x_new[i] = xi_new;
        local_delta += fabs((double)xi_new - (double)x_old[i]);
    }

    
    double warp_sum = warp_reduce_sum_d(local_delta);

    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    if (lane == 0) sdata[warp_id] = warp_sum;
    __syncthreads();

    int num_warps = blockDim.x >> 5;
    if (warp_id == 0) {
        double val = (lane < num_warps) ? sdata[lane] : 0.0;
        val = warp_reduce_sum_d(val);
        if (lane == 0) {
            atomicAdd(g_delta, val);
        }
    }
}



void launch_convert_d2f(const double* src, float* dst, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 512) grid = 512;
    convert_d2f_kernel<<<grid, block, 0, stream>>>(src, dst, n);
}

void launch_convert_f2d(const float* src, double* dst, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 512) grid = 512;
    convert_f2d_kernel<<<grid, block, 0, stream>>>(src, dst, n);
}

void launch_init_uniform(float* x, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 512) grid = 512;
    init_uniform_kernel<<<grid, block, 0, stream>>>(x, n);
}

void launch_add_identity_and_norm(float* y, const float* x_old, int n,
                                   double* g_norm_sq, int grid_size, cudaStream_t stream) {
    add_identity_and_norm_kernel<<<grid_size, 256, 0, stream>>>(y, x_old, n, g_norm_sq);
}

void launch_normalize_and_delta(const float* y, const float* x_old, float* x_new, int n,
                                 const double* g_norm_sq, double* g_delta,
                                 int grid_size, cudaStream_t stream) {
    normalize_and_delta_kernel<<<grid_size, 256, 0, stream>>>(y, x_old, x_new, n, g_norm_sq, g_delta);
}



struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    double* h_delta_pinned = nullptr;

    float* weights_f32 = nullptr;
    float* buf0 = nullptr;
    float* buf1 = nullptr;
    float* y = nullptr;
    double* accum = nullptr;
    void* spmv_buffer = nullptr;

    int64_t weights_capacity = 0;
    int64_t buf0_capacity = 0;
    int64_t buf1_capacity = 0;
    int64_t y_capacity = 0;
    int64_t accum_capacity = 0;
    int64_t spmv_buffer_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cudaMallocHost(&h_delta_pinned, sizeof(double));
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (h_delta_pinned) cudaFreeHost(h_delta_pinned);
        if (weights_f32) cudaFree(weights_f32);
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (y) cudaFree(y);
        if (accum) cudaFree(accum);
        if (spmv_buffer) cudaFree(spmv_buffer);
    }

    void ensure_buffers(int32_t num_vertices, int32_t num_edges) {
        if (weights_capacity < num_edges) {
            if (weights_f32) cudaFree(weights_f32);
            cudaMalloc(&weights_f32, (size_t)num_edges * sizeof(float));
            weights_capacity = num_edges;
        }
        if (buf0_capacity < num_vertices) {
            if (buf0) cudaFree(buf0);
            cudaMalloc(&buf0, (size_t)num_vertices * sizeof(float));
            buf0_capacity = num_vertices;
        }
        if (buf1_capacity < num_vertices) {
            if (buf1) cudaFree(buf1);
            cudaMalloc(&buf1, (size_t)num_vertices * sizeof(float));
            buf1_capacity = num_vertices;
        }
        if (y_capacity < num_vertices) {
            if (y) cudaFree(y);
            cudaMalloc(&y, (size_t)num_vertices * sizeof(float));
            y_capacity = num_vertices;
        }
        if (accum_capacity < 2) {
            if (accum) cudaFree(accum);
            cudaMalloc(&accum, 2 * sizeof(double));
            accum_capacity = 2;
        }
    }

    void ensure_spmv_buffer(size_t size) {
        if (spmv_buffer_capacity < (int64_t)size) {
            if (spmv_buffer) cudaFree(spmv_buffer);
            cudaMalloc(&spmv_buffer, size);
            spmv_buffer_capacity = (int64_t)size;
        }
    }
};

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
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    cudaStream_t stream = 0;
    cusparseSetStream(cache.cusparse_handle, stream);

    
    cache.ensure_buffers(num_vertices, num_edges);

    float* d_weights_f32 = cache.weights_f32;
    float* d_buf0 = cache.buf0;
    float* d_buf1 = cache.buf1;
    float* d_y = cache.y;
    double* d_accum = cache.accum;
    double* d_norm_sq = &d_accum[0];
    double* d_delta = &d_accum[1];

    
    launch_convert_d2f(edge_weights, d_weights_f32, num_edges, stream);

    
    if (initial_centralities != nullptr) {
        launch_convert_d2f(initial_centralities, d_buf0, num_vertices, stream);
    } else {
        launch_init_uniform(d_buf0, num_vertices, stream);
    }

    
    cusparseSpMatDescr_t mat_desc = nullptr;
    cusparseCreateCsr(&mat_desc,
        (int64_t)num_vertices, (int64_t)num_vertices, (int64_t)num_edges,
        (void*)offsets, (void*)indices, (void*)d_weights_f32,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    
    cusparseDnVecDescr_t vec_x = nullptr, vec_y = nullptr;
    cusparseCreateDnVec(&vec_x, (int64_t)num_vertices, d_buf0, CUDA_R_32F);
    cusparseCreateDnVec(&vec_y, (int64_t)num_vertices, d_y, CUDA_R_32F);

    
    float alpha = 1.0f, beta = 0.0f;

    
    size_t spmv_buffer_size = 0;
    cusparseSpMV_bufferSize(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_desc, vec_x, &beta, vec_y,
        CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1,
        &spmv_buffer_size);

    if (spmv_buffer_size < 4) spmv_buffer_size = 4;
    cache.ensure_spmv_buffer(spmv_buffer_size);

    
    cusparseSpMV_preprocess(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_desc, vec_x, &beta, vec_y,
        CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1,
        cache.spmv_buffer);

    
    int grid_size = (num_vertices + 255) / 256;
    if (grid_size > 240) grid_size = 240;
    if (grid_size < 1) grid_size = 1;

    
    float* x_curr = d_buf0;
    float* x_next = d_buf1;
    double threshold = (double)num_vertices * epsilon;

    size_t num_iterations = 0;
    bool converged = false;

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        
        cusparseDnVecSetValues(vec_x, x_curr);

        
        cusparseSpMV(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_desc, vec_x, &beta, vec_y,
            CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1,
            cache.spmv_buffer);

        
        cudaMemsetAsync(d_accum, 0, 2 * sizeof(double), stream);

        
        launch_add_identity_and_norm(d_y, x_curr, num_vertices, d_norm_sq, grid_size, stream);

        
        launch_normalize_and_delta(d_y, x_curr, x_next, num_vertices,
                                   d_norm_sq, d_delta, grid_size, stream);

        
        std::swap(x_curr, x_next);
        num_iterations = iter + 1;

        
        cudaMemcpyAsync(cache.h_delta_pinned, d_delta, sizeof(double),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (*cache.h_delta_pinned < threshold) {
            converged = true;
            break;
        }
    }

    
    launch_convert_f2d(x_curr, centralities, num_vertices, stream);

    
    cusparseDestroySpMat(mat_desc);
    cusparseDestroyDnVec(vec_x);
    cusparseDestroyDnVec(vec_y);

    cudaStreamSynchronize(stream);

    return {num_iterations, converged};
}

}  
