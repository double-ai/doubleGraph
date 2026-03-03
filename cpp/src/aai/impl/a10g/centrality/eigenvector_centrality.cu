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
#include <cub/block/block_reduce.cuh>
#include <cstdint>
#include <cmath>

namespace aai {

namespace {

#define BS 256


__global__ void __launch_bounds__(BS, 6)
spmv_identity_norm_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ d_norm_sq,
    int num_vertices
) {
    typedef cub::BlockReduce<float, BS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = blockIdx.x * BS + threadIdx.x;
    float val = 0.0f;

    if (v < num_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];
        val = x[v]; 
        for (int k = start; k < end; k++) {
            val += x[indices[k]];
        }
        y[v] = val;
    }

    float sq = val * val;
    float block_sum = BlockReduce(temp_storage).Sum(sq);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(d_norm_sq, block_sum);
    }
}


__global__ void __launch_bounds__(BS)
add_identity_norm_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    float* __restrict__ d_norm_sq,
    int num_vertices
) {
    typedef cub::BlockReduce<float, BS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = blockIdx.x * BS + threadIdx.x;
    float val = 0.0f;
    if (v < num_vertices) {
        val = y[v] + x[v];
        y[v] = val;
    }
    float sq = val * val;
    float block_sum = BlockReduce(temp_storage).Sum(sq);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(d_norm_sq, block_sum);
    }
}


__global__ void __launch_bounds__(BS)
normalize_diff_kernel(
    const float* __restrict__ y,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float* __restrict__ d_norm_sq,
    float* __restrict__ d_diff,
    int num_vertices
) {
    typedef cub::BlockReduce<float, BS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float norm_sq = *d_norm_sq;
    float inv_norm = (norm_sq > 0.0f) ? rsqrtf(norm_sq) : 0.0f;

    int v = blockIdx.x * BS + threadIdx.x;
    float diff_val = 0.0f;
    if (v < num_vertices) {
        float xn = y[v] * inv_norm;
        x_new[v] = xn;
        diff_val = fabsf(xn - x_old[v]);
    }
    float block_sum = BlockReduce(temp_storage).Sum(diff_val);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(d_diff, block_sum);
    }
}



__global__ void init_degree_kernel(
    const int32_t* __restrict__ offsets,
    float* __restrict__ x,
    float* __restrict__ d_norm_sq,
    int num_vertices
) {
    typedef cub::BlockReduce<float, BS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = blockIdx.x * BS + threadIdx.x;
    float val = 0.0f;
    if (v < num_vertices) {
        val = (float)(offsets[v + 1] - offsets[v] + 1);
        x[v] = val;
    }
    float sq = val * val;
    float block_sum = BlockReduce(temp_storage).Sum(sq);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(d_norm_sq, block_sum);
    }
}


__global__ void normalize_inplace_kernel(
    float* __restrict__ x,
    const float* __restrict__ d_norm_sq,
    int num_vertices
) {
    float norm_sq = *d_norm_sq;
    float inv_norm = (norm_sq > 0.0f) ? rsqrtf(norm_sq) : 0.0f;
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        x[v] *= inv_norm;
    }
}

__global__ void init_uniform_kernel(float* x, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = val;
    }
}

__global__ void fill_ones_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = 1.0f;
    }
}

struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;

    float* buf = nullptr;
    float* y = nullptr;
    float* scalars = nullptr;
    float* values = nullptr;
    void* cusparse_buf = nullptr;

    int64_t buf_cap = 0;
    int64_t y_cap = 0;
    int64_t values_cap = 0;
    size_t cusparse_buf_cap = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cudaMalloc(&scalars, 2 * sizeof(float));
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (buf) cudaFree(buf);
        if (y) cudaFree(y);
        if (scalars) cudaFree(scalars);
        if (values) cudaFree(values);
        if (cusparse_buf) cudaFree(cusparse_buf);
    }

    void ensure(int32_t nv, int32_t ne) {
        if (buf_cap < nv) {
            if (buf) cudaFree(buf);
            cudaMalloc(&buf, (size_t)nv * sizeof(float));
            buf_cap = nv;
        }
        if (y_cap < nv) {
            if (y) cudaFree(y);
            cudaMalloc(&y, (size_t)nv * sizeof(float));
            y_cap = nv;
        }
        if (values_cap < ne) {
            if (values) cudaFree(values);
            cudaMalloc(&values, (size_t)ne * sizeof(float));
            values_cap = ne;
        }
    }

    void ensure_cusparse_buf(size_t size) {
        if (cusparse_buf_cap < size) {
            if (cusparse_buf) cudaFree(cusparse_buf);
            cudaMalloc(&cusparse_buf, size);
            cusparse_buf_cap = size;
        }
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
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    cudaStream_t stream = 0;
    cusparseSetStream(cache.cusparse_handle, stream);

    float avg_degree = (num_vertices > 0) ? (float)num_edges / num_vertices : 0;
    bool use_cusparse = (avg_degree > 4.0f);

    cache.ensure(num_vertices, num_edges);

    float* d_norm_sq = cache.scalars;
    float* d_diff = cache.scalars + 1;

    
    float* bufs[2] = {centralities, cache.buf};
    float* y = cache.y;

    int grid = (num_vertices + BS - 1) / BS;

    
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;

    if (use_cusparse) {
        fill_ones_kernel<<<(num_edges + BS - 1) / BS, BS, 0, stream>>>(cache.values, num_edges);
        cusparseCreateCsr(&matA, num_vertices, num_vertices, num_edges,
                          (void*)d_offsets, (void*)d_indices, (void*)cache.values,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cusparseCreateDnVec(&vecX, num_vertices, bufs[0], CUDA_R_32F);
        cusparseCreateDnVec(&vecY, num_vertices, y, CUDA_R_32F);
        float alpha = 1.0f, beta = 0.0f;
        size_t bufferSize = 0;
        cusparseSpMV_bufferSize(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY,
                                 CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
        if (bufferSize > 0) {
            cache.ensure_cusparse_buf(bufferSize);
        }
        cusparseSpMV_preprocess(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY,
                                 CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.cusparse_buf);
    }

    
    int cur = 0;
    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(bufs[cur], initial_centralities,
                       (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemsetAsync(d_norm_sq, 0, sizeof(float), stream);
        init_degree_kernel<<<grid, BS, 0, stream>>>(d_offsets, bufs[cur], d_norm_sq, num_vertices);
        normalize_inplace_kernel<<<grid, BS, 0, stream>>>(bufs[cur], d_norm_sq, num_vertices);
    }

    bool converged = false;
    size_t iterations = 0;
    float threshold = (float)num_vertices * epsilon;
    float h_diff = 0.0f;
    float alpha = 1.0f, beta = 0.0f;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        int next = 1 - cur;

        cudaMemsetAsync(d_norm_sq, 0, 2 * sizeof(float), stream);

        if (use_cusparse) {
            cusparseDnVecSetValues(vecX, bufs[cur]);
            cusparseSpMV(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &alpha, matA, vecX, &beta, vecY,
                         CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.cusparse_buf);
            add_identity_norm_kernel<<<grid, BS, 0, stream>>>(y, bufs[cur], d_norm_sq, num_vertices);
        } else {
            spmv_identity_norm_kernel<<<grid, BS, 0, stream>>>(d_offsets, d_indices, bufs[cur], y,
                                       d_norm_sq, num_vertices);
        }

        normalize_diff_kernel<<<grid, BS, 0, stream>>>(y, bufs[cur], bufs[next], d_norm_sq, d_diff, num_vertices);
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        iterations = iter + 1;

        if (h_diff < threshold) {
            converged = true;
            cur = next;
            break;
        }
        cur = next;
    }

    if (!converged) iterations = max_iterations;

    if (vecX) cusparseDestroyDnVec(vecX);
    if (vecY) cusparseDestroyDnVec(vecY);
    if (matA) cusparseDestroySpMat(matA);

    
    if (bufs[cur] != centralities) {
        cudaMemcpy(centralities, bufs[cur], (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return {iterations, converged};
}

}  
