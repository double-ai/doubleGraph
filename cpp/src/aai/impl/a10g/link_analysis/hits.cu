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
#include <cstddef>
#include <cfloat>
#include <algorithm>

namespace aai {

namespace {



__global__ void fill_kernel(float* arr, float val, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = val;
}

__global__ void scale_kernel(float* arr, const float* d_scale, int n) {
    float s = *d_scale;
    if (s <= 0.0f) return;
    float inv_s = 1.0f / s;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x) {
        arr[idx] *= inv_s;
    }
}

__global__ void normalize_and_diff_kernel(float* curr, const float* prev,
                                           const float* d_max, float* d_diff, int n) {
    float max_val = *d_max;
    float inv_max = (max_val > 0.0f) ? (1.0f / max_val) : 0.0f;

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float thread_diff = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float val = curr[i] * inv_max;
        curr[i] = val;
        thread_diff += fabsf(val - prev[i]);
    }

    float block_diff = BlockReduce(temp_storage).Sum(thread_diff);
    if (threadIdx.x == 0) atomicAdd(d_diff, block_diff);
}



void launch_fill(float* arr, float val, int64_t n, cudaStream_t stream) {
    if (n <= 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    fill_kernel<<<grid, block, 0, stream>>>(arr, val, n);
}

void launch_scale(float* arr, const float* d_scale, int n, cudaStream_t stream) {
    if (n <= 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 4096) grid = 4096;
    scale_kernel<<<grid, block, 0, stream>>>(arr, d_scale, n);
}

void launch_normalize_and_diff(float* curr, const float* prev, const float* d_max,
                                float* d_diff, int n, cudaStream_t stream) {
    if (n <= 0) return;
    cudaMemsetAsync(d_diff, 0, sizeof(float), stream);
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 4096) grid = 4096;
    normalize_and_diff_kernel<<<grid, block, 0, stream>>>(curr, prev, d_max, d_diff, n);
}

size_t get_cub_max_temp_bytes(int n) {
    size_t temp = 0;
    cub::DeviceReduce::Max((void*)nullptr, temp, (float*)nullptr, (float*)nullptr, n);
    return temp;
}

void launch_cub_max(const float* d_in, float* d_out, int n, void* d_temp,
                     size_t temp_bytes, cudaStream_t stream) {
    cub::DeviceReduce::Max(d_temp, temp_bytes, d_in, d_out, n, stream);
}

size_t get_cub_sum_temp_bytes(int n) {
    size_t temp = 0;
    cub::DeviceReduce::Sum((void*)nullptr, temp, (float*)nullptr, (float*)nullptr, n);
    return temp;
}

void launch_cub_sum(const float* d_in, float* d_out, int n, void* d_temp,
                     size_t temp_bytes, cudaStream_t stream) {
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n, stream);
}



struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;
    float* d_alpha = nullptr;
    float* d_beta = nullptr;
    float* h_pinned = nullptr;

    float* temp_hubs = nullptr;
    int64_t temp_hubs_capacity = 0;

    float* values = nullptr;
    int64_t values_capacity = 0;

    float* d_scalars = nullptr;
    bool scalars_allocated = false;

    uint8_t* cub_temp = nullptr;
    int64_t cub_temp_capacity = 0;

    int32_t* t_offsets = nullptr;
    int64_t t_offsets_capacity = 0;

    int32_t* t_indices = nullptr;
    int64_t t_indices_capacity = 0;

    uint8_t* spmv_buf_A = nullptr;
    int64_t spmv_buf_A_capacity = 0;

    uint8_t* spmv_buf_AT = nullptr;
    int64_t spmv_buf_AT_capacity = 0;

    uint8_t* csr2csc_buf = nullptr;
    int64_t csr2csc_buf_capacity = 0;

    Cache() {
        cusparseCreate(&handle);
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);

        cudaMalloc(&d_alpha, sizeof(float));
        cudaMalloc(&d_beta, sizeof(float));
        float h_one = 1.0f, h_zero = 0.0f;
        cudaMemcpy(d_alpha, &h_one, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, &h_zero, sizeof(float), cudaMemcpyHostToDevice);

        cudaMallocHost(&h_pinned, sizeof(float));
    }

    void ensure_temp_hubs(int64_t n) {
        if (temp_hubs_capacity < n) {
            if (temp_hubs) cudaFree(temp_hubs);
            cudaMalloc(&temp_hubs, n * sizeof(float));
            temp_hubs_capacity = n;
        }
    }

    void ensure_values(int64_t n) {
        if (values_capacity < n) {
            if (values) cudaFree(values);
            cudaMalloc(&values, n * sizeof(float));
            values_capacity = n;
        }
    }

    void ensure_scalars() {
        if (!scalars_allocated) {
            cudaMalloc(&d_scalars, 3 * sizeof(float));
            scalars_allocated = true;
        }
    }

    void ensure_cub_temp(int64_t n) {
        if (cub_temp_capacity < n) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, n);
            cub_temp_capacity = n;
        }
    }

    void ensure_t_offsets(int64_t n) {
        if (t_offsets_capacity < n) {
            if (t_offsets) cudaFree(t_offsets);
            cudaMalloc(&t_offsets, n * sizeof(int32_t));
            t_offsets_capacity = n;
        }
    }

    void ensure_t_indices(int64_t n) {
        if (t_indices_capacity < n) {
            if (t_indices) cudaFree(t_indices);
            cudaMalloc(&t_indices, n * sizeof(int32_t));
            t_indices_capacity = n;
        }
    }

    void ensure_spmv_buf_A(int64_t n) {
        if (spmv_buf_A_capacity < n) {
            if (spmv_buf_A) cudaFree(spmv_buf_A);
            cudaMalloc(&spmv_buf_A, n);
            spmv_buf_A_capacity = n;
        }
    }

    void ensure_spmv_buf_AT(int64_t n) {
        if (spmv_buf_AT_capacity < n) {
            if (spmv_buf_AT) cudaFree(spmv_buf_AT);
            cudaMalloc(&spmv_buf_AT, n);
            spmv_buf_AT_capacity = n;
        }
    }

    void ensure_csr2csc_buf(int64_t n) {
        if (csr2csc_buf_capacity < n) {
            if (csr2csc_buf) cudaFree(csr2csc_buf);
            cudaMalloc(&csr2csc_buf, n);
            csr2csc_buf_capacity = n;
        }
    }

    ~Cache() override {
        if (handle) cusparseDestroy(handle);
        if (d_alpha) cudaFree(d_alpha);
        if (d_beta) cudaFree(d_beta);
        if (h_pinned) cudaFreeHost(h_pinned);
        if (temp_hubs) cudaFree(temp_hubs);
        if (values) cudaFree(values);
        if (d_scalars) cudaFree(d_scalars);
        if (cub_temp) cudaFree(cub_temp);
        if (t_offsets) cudaFree(t_offsets);
        if (t_indices) cudaFree(t_indices);
        if (spmv_buf_A) cudaFree(spmv_buf_A);
        if (spmv_buf_AT) cudaFree(spmv_buf_AT);
        if (csr2csc_buf) cudaFree(csr2csc_buf);
    }
};

}  

HitsResult hits(const graph32_t& graph,
                float* hubs,
                float* authorities,
                float epsilon,
                std::size_t max_iterations,
                bool has_initial_hubs_guess,
                bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t N = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;
    bool is_symmetric = graph.is_symmetric;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    float tolerance = static_cast<float>(N) * epsilon;
    cudaStream_t stream = nullptr;
    cusparseSetStream(cache.handle, stream);

    if (N == 0) {
        return HitsResult{max_iterations, false, FLT_MAX};
    }

    
    cache.ensure_temp_hubs(N);
    cache.ensure_values(std::max(E, (int32_t)1));
    cache.ensure_scalars();

    float* d_temp_hubs = cache.temp_hubs;
    float* d_values = cache.values;
    float* d_max = cache.d_scalars;
    float* d_diff = cache.d_scalars + 1;
    float* d_sum = cache.d_scalars + 2;

    launch_fill(d_values, 1.0f, (int64_t)E, stream);

    size_t cub_max_bytes = get_cub_max_temp_bytes(N);
    size_t cub_sum_bytes = get_cub_sum_temp_bytes(N);
    size_t cub_bytes = std::max(cub_max_bytes, cub_sum_bytes);
    cache.ensure_cub_temp(cub_bytes);
    void* d_cub_temp = cache.cub_temp;

    
    const int32_t* d_t_offsets = d_offsets;
    const int32_t* d_t_indices = d_indices;

    if (!is_symmetric) {
        cache.ensure_t_offsets(N + 1);
        cache.ensure_t_indices(std::max(E, (int32_t)1));

        size_t csr2csc_buf_size = 0;
        cusparseCsr2cscEx2_bufferSize(
            cache.handle, N, N, E,
            (const void*)d_values, (const int*)d_offsets, (const int*)d_indices,
            (void*)d_values, (int*)cache.t_offsets, (int*)cache.t_indices,
            CUDA_R_32F, CUSPARSE_ACTION_SYMBOLIC,
            CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
            &csr2csc_buf_size);

        cache.ensure_csr2csc_buf(std::max(csr2csc_buf_size, (size_t)1));

        cusparseCsr2cscEx2(
            cache.handle, N, N, E,
            (const void*)d_values, (const int*)d_offsets, (const int*)d_indices,
            (void*)d_values, (int*)cache.t_offsets, (int*)cache.t_indices,
            CUDA_R_32F, CUSPARSE_ACTION_SYMBOLIC,
            CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
            cache.csr2csc_buf);

        d_t_offsets = cache.t_offsets;
        d_t_indices = cache.t_indices;
    }

    
    if (has_initial_hubs_guess) {
        launch_cub_sum(hubs, d_sum, N, d_cub_temp, cub_sum_bytes, stream);
        launch_scale(hubs, d_sum, N, stream);
    } else {
        launch_fill(hubs, 1.0f / N, (int64_t)N, stream);
    }

    
    cusparseSpMatDescr_t matA = nullptr;
    cusparseCreateCsr(&matA, N, N, E,
        (void*)d_offsets, (void*)d_indices, (void*)d_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseSpMatDescr_t matAT = nullptr;
    bool same_mat = is_symmetric;
    if (same_mat) {
        matAT = matA;
    } else {
        cusparseCreateCsr(&matAT, N, N, E,
            (void*)d_t_offsets, (void*)d_t_indices, (void*)d_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    }

    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    cusparseCreateDnVec(&vecX, N, hubs, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, N, authorities, CUDA_R_32F);

    
    float h_alpha = 1.0f, h_beta = 0.0f;
    size_t buf_A = 0, buf_AT = 0;
    cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha, matA, vecX, &h_beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_A);

    if (!same_mat) {
        cusparseDnVecSetValues(vecX, authorities);
        cusparseDnVecSetValues(vecY, d_temp_hubs);
        cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_alpha, matAT, vecX, &h_beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_AT);
    } else {
        buf_AT = buf_A;
    }

    cache.ensure_spmv_buf_A(std::max(buf_A, (size_t)1));
    void* d_buf_A = cache.spmv_buf_A;
    void* d_buf_AT = d_buf_A;
    if (!same_mat) {
        cache.ensure_spmv_buf_AT(std::max(buf_AT, (size_t)1));
        d_buf_AT = cache.spmv_buf_AT;
    }

    
    cusparseDnVecSetValues(vecX, hubs);
    cusparseDnVecSetValues(vecY, authorities);
    cusparseSpMV_preprocess(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        cache.d_alpha, matA, vecX, cache.d_beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf_A);

    if (!same_mat) {
        cusparseDnVecSetValues(vecX, authorities);
        cusparseDnVecSetValues(vecY, d_temp_hubs);
        cusparseSpMV_preprocess(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha, matAT, vecX, cache.d_beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf_AT);
    }

    
    float* prev_hubs = hubs;
    float* curr_hubs = d_temp_hubs;
    float diff_sum = FLT_MAX;
    size_t iter = 0;

    if (max_iterations == 0) goto done;

    while (true) {
        
        cusparseDnVecSetValues(vecX, prev_hubs);
        cusparseDnVecSetValues(vecY, authorities);
        cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha, matA, vecX, cache.d_beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf_A);

        
        cusparseDnVecSetValues(vecX, authorities);
        cusparseDnVecSetValues(vecY, curr_hubs);
        cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha, matAT, vecX, cache.d_beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buf_AT);

        
        launch_cub_max(curr_hubs, d_max, N, d_cub_temp, cub_max_bytes, stream);
        launch_normalize_and_diff(curr_hubs, prev_hubs, d_max, d_diff, N, stream);

        
        launch_cub_max(authorities, d_max, N, d_cub_temp, cub_max_bytes, stream);
        launch_scale(authorities, d_max, N, stream);

        
        cudaMemcpyAsync(cache.h_pinned, d_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        diff_sum = *cache.h_pinned;

        std::swap(prev_hubs, curr_hubs);
        iter++;

        if (diff_sum < tolerance) break;
        if (iter >= max_iterations) break;
    }

done:
    bool converged = (diff_sum < tolerance);

    if (normalize) {
        launch_cub_sum(prev_hubs, d_sum, N, d_cub_temp, cub_sum_bytes, stream);
        launch_scale(prev_hubs, d_sum, N, stream);
        launch_cub_sum(authorities, d_sum, N, d_cub_temp, cub_sum_bytes, stream);
        launch_scale(authorities, d_sum, N, stream);
    }

    if (prev_hubs != hubs) {
        cudaMemcpyAsync(hubs, prev_hubs, N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    cusparseDestroySpMat(matA);
    if (!same_mat) cusparseDestroySpMat(matAT);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    return HitsResult{iter, converged, diff_sum};
}

}  
