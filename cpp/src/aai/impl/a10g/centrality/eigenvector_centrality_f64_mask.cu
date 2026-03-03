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
#include <algorithm>

namespace aai {

namespace {





__global__ void apply_mask_d2f_kernel(
    const double* __restrict__ weights_f64,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ weights_f32,
    int32_t num_edges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    float w = (float)weights_f64[e];
    uint32_t mask_word = edge_mask[e >> 5];
    int bit = (mask_word >> (e & 31)) & 1u;
    weights_f32[e] = bit ? w : 0.0f;
}





__global__ void init_uniform_kernel(float* __restrict__ x, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0f / (float)n;
}

__global__ void d2f_kernel(const double* __restrict__ src, float* __restrict__ dst, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (float)src[i];
}

__global__ void f2d_kernel(const float* __restrict__ src, double* __restrict__ dst, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (double)src[i];
}





__global__ void add_norm_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    float* __restrict__ norm_sq_out,
    int32_t n)
{
    float local_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float val = y[i] + x[i];
        y[i] = val;
        local_sum += val * val;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(norm_sq_out, local_sum);
}

__global__ void normalize_diff_kernel(
    const float* __restrict__ y,
    float* __restrict__ x,
    float* __restrict__ diff_out,
    const float* __restrict__ norm_sq_ptr,
    int32_t n)
{
    float norm = sqrtf(*norm_sq_ptr);
    float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 0.0f;
    float local_diff = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float y_val = y[i];
        float x_old = x[i];
        float x_new = y_val * inv_norm;
        local_diff += fabsf(x_new - x_old);
        x[i] = x_new;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_diff += __shfl_xor_sync(0xffffffff, local_diff, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(diff_out, local_diff);
}





struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    float* h_diff_pinned = nullptr;

    float* weights_f32 = nullptr;
    int64_t weights_f32_capacity = 0;

    float* x = nullptr;
    int64_t x_capacity = 0;

    float* y = nullptr;
    int64_t y_capacity = 0;

    float* alpha = nullptr;
    bool alpha_allocated = false;

    float* beta = nullptr;
    bool beta_allocated = false;

    float* norm_sq = nullptr;
    bool norm_sq_allocated = false;

    float* diff = nullptr;
    bool diff_allocated = false;

    void* spmv_buffer = nullptr;
    size_t spmv_buffer_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
        cudaMallocHost(&h_diff_pinned, sizeof(float));
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (h_diff_pinned) cudaFreeHost(h_diff_pinned);
        if (weights_f32) cudaFree(weights_f32);
        if (x) cudaFree(x);
        if (y) cudaFree(y);
        if (alpha) cudaFree(alpha);
        if (beta) cudaFree(beta);
        if (norm_sq) cudaFree(norm_sq);
        if (diff) cudaFree(diff);
        if (spmv_buffer) cudaFree(spmv_buffer);
    }

    void ensure(int32_t num_vertices, int32_t num_edges, size_t buffer_size) {
        if (weights_f32_capacity < num_edges) {
            if (weights_f32) cudaFree(weights_f32);
            cudaMalloc(&weights_f32, (size_t)num_edges * sizeof(float));
            weights_f32_capacity = num_edges;
        }
        if (x_capacity < num_vertices) {
            if (x) cudaFree(x);
            cudaMalloc(&x, (size_t)num_vertices * sizeof(float));
            x_capacity = num_vertices;
        }
        if (y_capacity < num_vertices) {
            if (y) cudaFree(y);
            cudaMalloc(&y, (size_t)num_vertices * sizeof(float));
            y_capacity = num_vertices;
        }
        if (!alpha_allocated) {
            cudaMalloc(&alpha, sizeof(float));
            alpha_allocated = true;
        }
        if (!beta_allocated) {
            cudaMalloc(&beta, sizeof(float));
            beta_allocated = true;
        }
        if (!norm_sq_allocated) {
            cudaMalloc(&norm_sq, sizeof(float));
            norm_sq_allocated = true;
        }
        if (!diff_allocated) {
            cudaMalloc(&diff, sizeof(float));
            diff_allocated = true;
        }
        if (spmv_buffer_capacity < buffer_size) {
            if (spmv_buffer) cudaFree(spmv_buffer);
            size_t alloc_size = buffer_size > 0 ? buffer_size : 1;
            cudaMalloc(&spmv_buffer, alloc_size);
            spmv_buffer_capacity = alloc_size;
        }
    }
};

}  

eigenvector_centrality_result_t eigenvector_centrality_mask(const graph32_t& graph,
                                  const double* edge_weights,
                                  double* centralities,
                                  double epsilon,
                                  std::size_t max_iterations,
                                  const double* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;

    double tolerance = (double)num_vertices * epsilon;

    
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cusparseSetStream(cache.cusparse_handle, stream);

    
    cache.ensure(num_vertices, num_edges, 0);

    float* d_weights_f32 = cache.weights_f32;
    float* d_x = cache.x;
    float* d_y = cache.y;
    float* d_alpha = cache.alpha;
    float* d_beta = cache.beta;
    float* d_norm_sq = cache.norm_sq;
    float* d_diff = cache.diff;

    
    
    
    {
        int block = 256;
        int grid = (num_edges + block - 1) / block;
        apply_mask_d2f_kernel<<<grid, block>>>(edge_weights, d_edge_mask, d_weights_f32, num_edges);
    }

    
    
    
    cusparseSpMatDescr_t mat_descr;
    cusparseCreateCsr(&mat_descr,
        (int64_t)num_vertices, (int64_t)num_vertices, (int64_t)num_edges,
        (void*)d_offsets, (void*)d_indices, d_weights_f32,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vec_x, vec_y;
    cusparseCreateDnVec(&vec_x, num_vertices, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vec_y, num_vertices, d_y, CUDA_R_32F);

    
    float h_alpha = 1.0f, h_beta = 0.0f;
    cudaMemcpyAsync(d_alpha, &h_alpha, sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_beta, &h_beta, sizeof(float), cudaMemcpyHostToDevice, stream);

    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_alpha, mat_descr, vec_x, d_beta, vec_y,
        CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, &bufferSize);

    
    cache.ensure(num_vertices, num_edges, std::max(bufferSize, (size_t)1));
    void* d_buffer = cache.spmv_buffer;

    
    cusparseSpMV_preprocess(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_alpha, mat_descr, vec_x, d_beta, vec_y,
        CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, d_buffer);

    
    
    
    if (initial_centralities != nullptr) {
        int block = 256;
        int grid = (int)(((int64_t)num_vertices + block - 1) / block);
        d2f_kernel<<<grid, block>>>(initial_centralities, d_x, num_vertices);
    } else {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        init_uniform_kernel<<<grid, block>>>(d_x, num_vertices);
    }
    
    cudaDeviceSynchronize();

    
    
    
    cudaGraph_t cuda_graph;
    cudaGraphExec_t graphExec;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    
    cudaMemsetAsync(d_norm_sq, 0, sizeof(float), stream);
    cusparseSpMV(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_alpha, mat_descr, vec_x, d_beta, vec_y,
        CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, d_buffer);
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 512) grid = 512;
        add_norm_kernel<<<grid, block, 0, stream>>>(d_y, d_x, d_norm_sq, num_vertices);
    }
    cudaMemsetAsync(d_diff, 0, sizeof(float), stream);
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 512) grid = 512;
        normalize_diff_kernel<<<grid, block, 0, stream>>>(d_y, d_x, d_diff, d_norm_sq, num_vertices);
    }

    cudaStreamEndCapture(stream, &cuda_graph);
    cudaGraphInstantiate(&graphExec, cuda_graph, 0);
    cudaGraphDestroy(cuda_graph);

    
    
    
    std::size_t iter = 0;
    bool converged = false;

    for (; iter < max_iterations; iter++) {
        cudaGraphLaunch(graphExec, stream);

        
        cudaMemcpyAsync(cache.h_diff_pinned, d_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if ((double)(*(cache.h_diff_pinned)) < tolerance) {
            converged = true;
            iter++;
            break;
        }
    }

    
    
    
    {
        int block = 256;
        int grid = (int)(((int64_t)num_vertices + block - 1) / block);
        f2d_kernel<<<grid, block>>>(d_x, centralities, num_vertices);
    }
    cudaDeviceSynchronize();

    
    cudaGraphExecDestroy(graphExec);
    cusparseDestroySpMat(mat_descr);
    cusparseDestroyDnVec(vec_x);
    cusparseDestroyDnVec(vec_y);
    cudaStreamDestroy(stream);

    return {iter, converged};
}

}  
