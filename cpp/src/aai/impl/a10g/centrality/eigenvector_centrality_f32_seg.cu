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

constexpr int BLOCK_SIZE = 512;


__global__ void compute_degree_l2sq(
    const int32_t* __restrict__ offsets,
    float* __restrict__ result,
    int32_t n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float deg = float(offsets[i + 1] - offsets[i]);
        sum += deg * deg;
    }

    float block_sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0 && block_sum != 0.0f)
        atomicAdd(result, block_sum);
}


__global__ void init_from_degree(
    const int32_t* __restrict__ offsets,
    float* __restrict__ out,
    int32_t n,
    float inv_norm)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        out[i] = float(offsets[i + 1] - offsets[i]) * inv_norm;
    }
}


__global__ void add_identity_l2sq(
    float* __restrict__ y,
    const float* __restrict__ x,
    float* __restrict__ l2sq,
    int32_t n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float val = y[i] + x[i];
        y[i] = val;
        sum += val * val;
    }

    float block_sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
        atomicAdd(l2sq, block_sum);
}


__global__ void normalize_diff(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ l2sq,
    float* __restrict__ diff_out,
    int32_t n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float inv_norm = rsqrtf(*l2sq);

    float d = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float yn = y[i] * inv_norm;
        y[i] = yn;
        d += fabsf(yn - x[i]);
    }

    float block_d = BlockReduce(temp).Sum(d);
    if (threadIdx.x == 0)
        atomicAdd(diff_out, block_d);
}

struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;
    float* d_alpha = nullptr;
    float* d_beta = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    float* d_scalar = nullptr;
    void* spmv_buf = nullptr;
    size_t vert_cap = 0;
    size_t spmv_buf_cap = 0;

    Cache() {
        cusparseCreate(&handle);
        cudaMalloc(&d_alpha, sizeof(float));
        cudaMalloc(&d_beta, sizeof(float));
        float one = 1.0f, zero = 0.0f;
        cudaMemcpy(d_alpha, &one, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, &zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_scalar, 2 * sizeof(float));
    }

    ~Cache() override {
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
        if (d_scalar) cudaFree(d_scalar);
        if (spmv_buf) cudaFree(spmv_buf);
        if (d_alpha) cudaFree(d_alpha);
        if (d_beta) cudaFree(d_beta);
        if (handle) cusparseDestroy(handle);
    }

    void ensure_vertex_bufs(int32_t n) {
        if ((size_t)n > vert_cap) {
            if (d_x) cudaFree(d_x);
            if (d_y) cudaFree(d_y);
            cudaMalloc(&d_x, (size_t)n * sizeof(float));
            cudaMalloc(&d_y, (size_t)n * sizeof(float));
            vert_cap = n;
        }
    }

    void ensure_spmv_buf(size_t sz) {
        if (sz > spmv_buf_cap) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, sz);
            spmv_buf_cap = sz;
        }
    }
};

}  

eigenvector_centrality_result_t eigenvector_centrality_seg(const graph32_t& graph,
                                const float* edge_weights,
                                float* centralities,
                                float epsilon,
                                std::size_t max_iterations,
                                const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;

    cache.ensure_vertex_bufs(nv);

    float* x = cache.d_x;
    float* y = cache.d_y;
    float* d_l2sq = cache.d_scalar;
    float* d_diff = cache.d_scalar + 1;

    
    cusparseSpMatDescr_t mat;
    cusparseCreateCsr(
        &mat, nv, nv, ne,
        const_cast<int32_t*>(d_offsets),
        const_cast<int32_t*>(d_indices),
        const_cast<float*>(edge_weights),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F
    );

    cusparseDnVecDescr_t vec_x, vec_y;
    cusparseCreateDnVec(&vec_x, nv, x, CUDA_R_32F);
    cusparseCreateDnVec(&vec_y, nv, y, CUDA_R_32F);

    float h_one = 1.0f, h_zero = 0.0f;
    size_t buf_size = 0;
    cusparseSetPointerMode(cache.handle, CUSPARSE_POINTER_MODE_HOST);
    cusparseSpMV_bufferSize(
        cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_one, mat, vec_x, &h_zero, vec_y,
        CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1,
        &buf_size
    );
    cache.ensure_spmv_buf(buf_size);

    cusparseSpMV_preprocess(
        cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_one, mat, vec_x, &h_zero, vec_y,
        CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1,
        cache.spmv_buf
    );

    cusparseSetPointerMode(cache.handle, CUSPARSE_POINTER_MODE_DEVICE);

    
    if (initial_centralities != nullptr) {
        cudaMemcpy(x, initial_centralities,
                   (size_t)nv * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        
        cudaMemset(d_l2sq, 0, sizeof(float));
        int grid = std::min(512, (nv + BLOCK_SIZE - 1) / BLOCK_SIZE);
        if (grid < 1) grid = 1;
        compute_degree_l2sq<<<grid, BLOCK_SIZE>>>(d_offsets, d_l2sq, nv);
        float h_l2sq;
        cudaMemcpy(&h_l2sq, d_l2sq, sizeof(float), cudaMemcpyDeviceToHost);
        float inv_norm = (h_l2sq > 0.0f) ? (1.0f / sqrtf(h_l2sq))
                                          : (1.0f / sqrtf((float)nv));
        int grid2 = (nv + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_from_degree<<<grid2, BLOCK_SIZE>>>(d_offsets, x, nv, inv_norm);
    }

    
    float threshold = (float)nv * epsilon;
    size_t iter = 0;
    bool converged = false;

    for (; iter < max_iterations; ++iter) {
        
        cusparseDnVecSetValues(vec_x, x);
        cusparseDnVecSetValues(vec_y, y);
        cusparseSpMV(
            cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha, mat, vec_x, cache.d_beta, vec_y,
            CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1,
            cache.spmv_buf
        );

        
        cudaMemset(cache.d_scalar, 0, 2 * sizeof(float));

        
        int grid = std::min(512, (nv + BLOCK_SIZE - 1) / BLOCK_SIZE);
        if (grid < 1) grid = 1;
        add_identity_l2sq<<<grid, BLOCK_SIZE>>>(y, x, d_l2sq, nv);

        
        normalize_diff<<<grid, BLOCK_SIZE>>>(y, x, d_l2sq, d_diff, nv);

        
        float h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_diff < threshold) {
            converged = true;
            std::swap(x, y);
            iter++;
            break;
        }

        std::swap(x, y);
    }

    
    cudaMemcpy(centralities, x, (size_t)nv * sizeof(float), cudaMemcpyDeviceToDevice);

    
    cusparseDestroyDnVec(vec_x);
    cusparseDestroyDnVec(vec_y);
    cusparseDestroySpMat(mat);

    return {iter, converged};
}

}  
