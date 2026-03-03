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
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace aai {

namespace {

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t s_ = (call); \
    if (s_ != CUSPARSE_STATUS_SUCCESS) \
        throw std::runtime_error("cuSPARSE error: " + std::to_string((int)s_)); \
} while(0)

#define BLOCK_SIZE 256



__global__ __launch_bounds__(BLOCK_SIZE)
void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ counts,
    int32_t num_vertices
) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_vertices;
         j += blockDim.x * gridDim.x) {
        int start = offsets[j];
        int end = offsets[j + 1];
        int count = 0;
        for (int k = start; k < end; k++) {
            if (edge_mask[k >> 5] & (1u << (k & 31))) count++;
        }
        counts[j] = count;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) counts[num_vertices] = 0;
}

__global__ __launch_bounds__(BLOCK_SIZE)
void scatter_active_edges_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    float* __restrict__ new_weights,
    int32_t num_vertices
) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_vertices;
         j += blockDim.x * gridDim.x) {
        int old_start = old_offsets[j];
        int old_end = old_offsets[j + 1];
        int new_pos = new_offsets[j];
        for (int k = old_start; k < old_end; k++) {
            if (edge_mask[k >> 5] & (1u << (k & 31))) {
                new_indices[new_pos] = indices[k];
                new_weights[new_pos] = weights[k];
                new_pos++;
            }
        }
    }
}



__global__ __launch_bounds__(BLOCK_SIZE)
void add_identity_and_norm_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    double* __restrict__ d_norm_sq,
    int32_t num_vertices
) {
    double thread_norm_sq = 0.0;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_vertices;
         j += blockDim.x * gridDim.x) {
        float val = y[j] + x[j];
        y[j] = val;
        thread_norm_sq += (double)val * (double)val;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_norm_sq += __shfl_down_sync(0xffffffff, thread_norm_sq, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(d_norm_sq, thread_norm_sq);
}

__global__ __launch_bounds__(BLOCK_SIZE)
void normalize_and_diff_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const double* __restrict__ d_norm_sq,
    double* __restrict__ d_diff,
    int32_t num_vertices
) {
    double norm = sqrt(*d_norm_sq);
    float inv_norm = (norm > 1e-30) ? (float)(1.0 / norm) : 0.0f;
    double thread_diff = 0.0;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_vertices;
         j += blockDim.x * gridDim.x) {
        float y_j = y[j] * inv_norm;
        y[j] = y_j;
        thread_diff += (double)fabsf(y_j - x[j]);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_diff += __shfl_down_sync(0xffffffff, thread_diff, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(d_diff, thread_diff);
}

__global__ __launch_bounds__(BLOCK_SIZE)
void normalize_only_kernel(
    float* __restrict__ y,
    const double* __restrict__ d_norm_sq,
    int32_t num_vertices
) {
    double norm = sqrt(*d_norm_sq);
    float inv_norm = (norm > 1e-30) ? (float)(1.0 / norm) : 0.0f;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_vertices;
         j += blockDim.x * gridDim.x)
        y[j] *= inv_norm;
}

__global__ void init_uniform_kernel(float* centralities, int32_t n) {
    float val = 1.0f / (float)n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x)
        centralities[i] = val;
}



struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;

    int32_t* counts = nullptr;
    int32_t* new_off = nullptr;
    float* buf0 = nullptr;
    float* buf1 = nullptr;

    int32_t* new_idx = nullptr;
    float* new_wt = nullptr;

    uint8_t* cub_temp = nullptr;
    uint8_t* spmv_buf = nullptr;

    float* d_alpha = nullptr;
    float* d_beta = nullptr;
    double* d_norm = nullptr;
    double* d_diff = nullptr;

    int64_t counts_cap = 0;
    int64_t new_off_cap = 0;
    int64_t buf0_cap = 0;
    int64_t buf1_cap = 0;
    int64_t new_idx_cap = 0;
    int64_t new_wt_cap = 0;
    int64_t cub_temp_cap = 0;
    int64_t spmv_buf_cap = 0;

    Cache() {
        cusparseCreate(&handle);
        cudaMalloc(&d_alpha, sizeof(float));
        cudaMalloc(&d_beta, sizeof(float));
        cudaMalloc(&d_norm, sizeof(double));
        cudaMalloc(&d_diff, sizeof(double));
        float h_a = 1.0f, h_b = 0.0f;
        cudaMemcpy(d_alpha, &h_a, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, &h_b, sizeof(float), cudaMemcpyHostToDevice);
    }

    ~Cache() override {
        if (handle) cusparseDestroy(handle);
        if (counts) cudaFree(counts);
        if (new_off) cudaFree(new_off);
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (new_idx) cudaFree(new_idx);
        if (new_wt) cudaFree(new_wt);
        if (cub_temp) cudaFree(cub_temp);
        if (spmv_buf) cudaFree(spmv_buf);
        if (d_alpha) cudaFree(d_alpha);
        if (d_beta) cudaFree(d_beta);
        if (d_norm) cudaFree(d_norm);
        if (d_diff) cudaFree(d_diff);
    }

    void ensure_vertex(int64_t nv) {
        int64_t nv1 = nv + 1;
        if (counts_cap < nv1) {
            if (counts) cudaFree(counts);
            cudaMalloc(&counts, nv1 * sizeof(int32_t));
            counts_cap = nv1;
        }
        if (new_off_cap < nv1) {
            if (new_off) cudaFree(new_off);
            cudaMalloc(&new_off, nv1 * sizeof(int32_t));
            new_off_cap = nv1;
        }
        if (buf0_cap < nv) {
            if (buf0) cudaFree(buf0);
            cudaMalloc(&buf0, nv * sizeof(float));
            buf0_cap = nv;
        }
        if (buf1_cap < nv) {
            if (buf1) cudaFree(buf1);
            cudaMalloc(&buf1, nv * sizeof(float));
            buf1_cap = nv;
        }
    }

    void ensure_edge(int64_t ne) {
        if (new_idx_cap < ne) {
            if (new_idx) cudaFree(new_idx);
            cudaMalloc(&new_idx, ne * sizeof(int32_t));
            new_idx_cap = ne;
        }
        if (new_wt_cap < ne) {
            if (new_wt) cudaFree(new_wt);
            cudaMalloc(&new_wt, ne * sizeof(float));
            new_wt_cap = ne;
        }
    }

    void ensure_cub(int64_t sz) {
        if (cub_temp_cap < sz) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, sz);
            cub_temp_cap = sz;
        }
    }

    void ensure_spmv(int64_t sz) {
        if (spmv_buf_cap < sz) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, sz);
            spmv_buf_cap = sz;
        }
    }
};

}  

eigenvector_centrality_result_t eigenvector_centrality_mask(const graph32_t& graph,
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
    const uint32_t* d_mask = graph.edge_mask;

    cache.ensure_vertex(num_vertices);
    cache.ensure_edge(num_edges);

    
    size_t cub_sz = 0;
    cub::DeviceScan::ExclusiveSum(
        nullptr, cub_sz,
        (int32_t*)nullptr, (int32_t*)nullptr,
        num_vertices + 1, (cudaStream_t)0);
    cache.ensure_cub((int64_t)cub_sz);

    
    int threads = BLOCK_SIZE;
    int blocks = (num_vertices + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;
    if (blocks < 1) blocks = 1;

    count_active_edges_kernel<<<blocks, threads>>>(
        d_offsets, d_mask, cache.counts, num_vertices);

    cub::DeviceScan::ExclusiveSum(
        cache.cub_temp, cub_sz, cache.counts, cache.new_off,
        num_vertices + 1, (cudaStream_t)0);

    scatter_active_edges_kernel<<<blocks, threads>>>(
        d_offsets, d_indices, edge_weights, d_mask,
        cache.new_off, cache.new_idx, cache.new_wt, num_vertices);

    int32_t n_active;
    cudaMemcpy(&n_active, cache.new_off + num_vertices,
               sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    cusparseSpMatDescr_t mat;
    CUSPARSE_CHECK(cusparseCreateCsr(&mat, num_vertices, num_vertices, n_active,
        (void*)cache.new_off, (void*)cache.new_idx, (void*)cache.new_wt,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    float* b0 = cache.buf0;
    float* b1 = cache.buf1;

    cusparseDnVecDescr_t vx, vy;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vx, num_vertices, b0, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vy, num_vertices, b1, CUDA_R_32F));

    CUSPARSE_CHECK(cusparseSetPointerMode(cache.handle, CUSPARSE_POINTER_MODE_DEVICE));

    size_t spmv_sz = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        cache.d_alpha, mat, vx, cache.d_beta, vy,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_sz));
    if (spmv_sz < 1) spmv_sz = 1;
    cache.ensure_spmv((int64_t)spmv_sz);

    CUSPARSE_CHECK(cusparseSpMV_preprocess(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        cache.d_alpha, mat, vx, cache.d_beta, vy,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf));

    constexpr int thr = 256;
    int blk = std::min((int)((num_vertices + thr - 1) / thr), 4096);
    if (blk < 1) blk = 1;

    
    if (initial_centralities != nullptr)
        cudaMemcpy(b0, initial_centralities,
                   (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    else
        init_uniform_kernel<<<blk, thr>>>(b0, num_vertices);

    double threshold = (double)num_vertices * (double)epsilon;
    size_t iters = 0;
    bool converged = false;
    float* xp = b0, *yp = b1;

    
    auto should_check = [&](size_t iter) -> bool {
        if (iter == max_iterations - 1) return true;
        size_t it1 = iter + 1;
        if (it1 <= 10) return true;
        if (it1 <= 100) return (it1 % 10 == 0);
        return (it1 % 50 == 0);
    };

    for (size_t iter = 0; iter < max_iterations; iter++) {
        
        CUSPARSE_CHECK(cusparseDnVecSetValues(vx, xp));
        CUSPARSE_CHECK(cusparseDnVecSetValues(vy, yp));

        
        CUSPARSE_CHECK(cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha, mat, vx, cache.d_beta, vy,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf));

        
        cudaMemsetAsync(cache.d_norm, 0, sizeof(double), 0);
        add_identity_and_norm_kernel<<<blk, thr>>>(yp, xp, cache.d_norm, num_vertices);

        bool check = should_check(iter);
        if (check) {
            cudaMemsetAsync(cache.d_diff, 0, sizeof(double), 0);
            normalize_and_diff_kernel<<<blk, thr>>>(yp, xp, cache.d_norm, cache.d_diff, num_vertices);
        } else {
            normalize_only_kernel<<<blk, thr>>>(yp, cache.d_norm, num_vertices);
        }

        iters = iter + 1;
        std::swap(xp, yp);

        if (check) {
            double h_diff;
            cudaMemcpy(&h_diff, cache.d_diff, sizeof(double), cudaMemcpyDeviceToHost);
            if (h_diff < threshold) { converged = true; break; }
        }
    }

    cusparseDestroySpMat(mat);
    cusparseDestroyDnVec(vx);
    cusparseDestroyDnVec(vy);

    
    cudaMemcpy(centralities, xp,
               (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);

    return {iters, converged};
}

}  
