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
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    
    cusparseHandle_t cusparse_handle_ = nullptr;
    double* d_alpha_ = nullptr;
    double* d_zero_ = nullptr;

    
    double* d_x0 = nullptr;
    double* d_x1 = nullptr;
    double* d_diff = nullptr;
    int32_t x0_capacity = 0;
    int32_t x1_capacity = 0;

    
    cusparseSpMatDescr_t cached_mat_ = nullptr;
    cusparseDnVecDescr_t cached_x_descr_ = nullptr;
    cusparseDnVecDescr_t cached_y_descr_ = nullptr;
    void* cached_buffer_ = nullptr;
    size_t cached_buffer_size_ = 0;
    int32_t cached_nv_ = 0;
    int32_t cached_ne_ = 0;
    const int32_t* cached_offsets_ = nullptr;
    const int32_t* cached_indices_ = nullptr;

    Cache() {
        cusparseCreate(&cusparse_handle_);
        cusparseSetPointerMode(cusparse_handle_, CUSPARSE_POINTER_MODE_DEVICE);

        cudaMalloc(&d_alpha_, sizeof(double));
        cudaMalloc(&d_zero_, sizeof(double));
        double zero = 0.0;
        cudaMemcpy(d_zero_, &zero, sizeof(double), cudaMemcpyHostToDevice);

        cudaMalloc(&d_diff, sizeof(double));
    }

    void ensure_buffers(int32_t num_vertices) {
        if (x0_capacity < num_vertices) {
            if (d_x0) cudaFree(d_x0);
            cudaMalloc(&d_x0, (size_t)num_vertices * sizeof(double));
            x0_capacity = num_vertices;
        }
        if (x1_capacity < num_vertices) {
            if (d_x1) cudaFree(d_x1);
            cudaMalloc(&d_x1, (size_t)num_vertices * sizeof(double));
            x1_capacity = num_vertices;
        }
    }

    void destroyCachedDescriptors() {
        if (cached_mat_) { cusparseDestroySpMat(cached_mat_); cached_mat_ = nullptr; }
        if (cached_x_descr_) { cusparseDestroyDnVec(cached_x_descr_); cached_x_descr_ = nullptr; }
        if (cached_y_descr_) { cusparseDestroyDnVec(cached_y_descr_); cached_y_descr_ = nullptr; }
        cached_nv_ = 0;
        cached_ne_ = 0;
        cached_offsets_ = nullptr;
        cached_indices_ = nullptr;
    }

    void ensureCuSparseSetup(int32_t nv, int32_t ne,
                              const int32_t* offsets, const int32_t* indices,
                              const double* values,
                              double* x_buf, double* y_buf,
                              double alpha) {
        bool can_reuse = (cached_mat_ != nullptr &&
                          cached_nv_ == nv && cached_ne_ == ne &&
                          cached_offsets_ == offsets && cached_indices_ == indices);

        if (!can_reuse) {
            destroyCachedDescriptors();

            cusparseCreateCsr(&cached_mat_, nv, nv, ne,
                              (void*)offsets, (void*)indices, (void*)values,
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

            cusparseCreateDnVec(&cached_x_descr_, nv, x_buf, CUDA_R_64F);
            cusparseCreateDnVec(&cached_y_descr_, nv, y_buf, CUDA_R_64F);

            cached_nv_ = nv;
            cached_ne_ = ne;
            cached_offsets_ = offsets;
            cached_indices_ = indices;

            cudaMemcpyAsync(d_alpha_, &alpha, sizeof(double), cudaMemcpyHostToDevice);

            cusparseSetPointerMode(cusparse_handle_, CUSPARSE_POINTER_MODE_HOST);
            double h_alpha = alpha;
            double h_zero = 0.0;
            size_t buffer_size = 0;
            cusparseSpMV_bufferSize(cusparse_handle_,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &h_alpha, cached_mat_, cached_x_descr_,
                                    &h_zero, cached_y_descr_,
                                    CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                                    &buffer_size);
            cusparseSetPointerMode(cusparse_handle_, CUSPARSE_POINTER_MODE_DEVICE);

            if (buffer_size > cached_buffer_size_) {
                if (cached_buffer_) cudaFree(cached_buffer_);
                cudaMalloc(&cached_buffer_, buffer_size);
                cached_buffer_size_ = buffer_size;
            }
        } else {
            cusparseCsrSetPointers(cached_mat_, (void*)offsets, (void*)indices, (void*)values);
            cusparseDnVecSetValues(cached_x_descr_, x_buf);
            cusparseDnVecSetValues(cached_y_descr_, y_buf);

            cudaMemcpyAsync(d_alpha_, &alpha, sizeof(double), cudaMemcpyHostToDevice);
        }
    }

    ~Cache() override {
        destroyCachedDescriptors();
        if (d_alpha_) cudaFree(d_alpha_);
        if (d_zero_) cudaFree(d_zero_);
        if (d_diff) cudaFree(d_diff);
        if (d_x0) cudaFree(d_x0);
        if (d_x1) cudaFree(d_x1);
        if (cached_buffer_) cudaFree(cached_buffer_);
        if (cusparse_handle_) cusparseDestroy(cusparse_handle_);
    }
};



template <int BLOCK_SIZE>
__global__ void katz_iteration_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ x_old,
    double* __restrict__ x_new,
    const double* __restrict__ betas,
    double alpha,
    double beta_scalar,
    int use_betas,
    double* __restrict__ global_diff,
    int32_t num_vertices
) {
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double thread_diff = 0.0;

    for (int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
         v < num_vertices;
         v += gridDim.x * BLOCK_SIZE) {

        int32_t row_start = __ldg(&offsets[v]);
        int32_t row_end = __ldg(&offsets[v + 1]);

        double dot = 0.0;
        for (int32_t j = row_start; j < row_end; j++) {
            dot += __ldg(&weights[j]) * __ldg(&x_old[__ldg(&indices[j])]);
        }

        double beta_v = use_betas ? __ldg(&betas[v]) : beta_scalar;
        double val = alpha * dot + beta_v;

        thread_diff += fabs(val - __ldg(&x_old[v]));
        x_new[v] = val;
    }

    double block_sum = BlockReduce(temp_storage).Sum(thread_diff);
    if (threadIdx.x == 0) {
        atomicAdd(global_diff, block_sum);
    }
}

template <int BLOCK_SIZE>
__global__ void katz_first_iter_kernel(
    double* __restrict__ x_new,
    const double* __restrict__ betas,
    double beta_scalar,
    int use_betas,
    double* __restrict__ global_diff,
    int32_t num_vertices
) {
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double thread_diff = 0.0;

    for (int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
         v < num_vertices;
         v += gridDim.x * BLOCK_SIZE) {
        double beta_v = use_betas ? betas[v] : beta_scalar;
        x_new[v] = beta_v;
        thread_diff += fabs(beta_v);
    }

    double block_sum = BlockReduce(temp_storage).Sum(thread_diff);
    if (threadIdx.x == 0) {
        atomicAdd(global_diff, block_sum);
    }
}

template <int BLOCK_SIZE>
__global__ void add_beta_diff_kernel(
    double* __restrict__ x_new,
    const double* __restrict__ x_old,
    const double* __restrict__ betas,
    double beta_scalar,
    int use_betas,
    double* __restrict__ global_diff,
    int32_t n
) {
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double thread_diff = 0.0;

    for (int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
         i < n;
         i += gridDim.x * BLOCK_SIZE) {
        double spmv_val = x_new[i];
        double beta_v = use_betas ? betas[i] : beta_scalar;
        double new_val = spmv_val + beta_v;
        thread_diff += fabs(new_val - x_old[i]);
        x_new[i] = new_val;
    }

    double block_sum = BlockReduce(temp_storage).Sum(thread_diff);
    if (threadIdx.x == 0) {
        atomicAdd(global_diff, block_sum);
    }
}

template <int BLOCK_SIZE>
__global__ void diff_only_kernel(
    const double* __restrict__ x_new,
    const double* __restrict__ x_old,
    double* __restrict__ global_diff,
    int32_t n
) {
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double thread_diff = 0.0;

    for (int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
         i < n;
         i += gridDim.x * BLOCK_SIZE) {
        thread_diff += fabs(x_new[i] - x_old[i]);
    }

    double block_sum = BlockReduce(temp_storage).Sum(thread_diff);
    if (threadIdx.x == 0) {
        atomicAdd(global_diff, block_sum);
    }
}

template <int BLOCK_SIZE>
__global__ void compute_norm_sq_kernel(
    const double* __restrict__ x,
    double* __restrict__ norm_sq,
    int32_t n
) {
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double thread_sum = 0.0;

    for (int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
         i < n;
         i += gridDim.x * BLOCK_SIZE) {
        double val = x[i];
        thread_sum += val * val;
    }

    double block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    if (threadIdx.x == 0) {
        atomicAdd(norm_sq, block_sum);
    }
}

template <int BLOCK_SIZE>
__global__ void normalize_kernel(
    double* __restrict__ x,
    double inv_norm,
    int32_t n
) {
    for (int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
         i < n;
         i += gridDim.x * BLOCK_SIZE) {
        x[i] *= inv_norm;
    }
}

constexpr int BS = 256;

}  

katz_centrality_result_t katz_centrality(const graph32_t& graph,
                     const double* edge_weights,
                     double* centralities,
                     double alpha,
                     double beta,
                     const double* betas,
                     double epsilon,
                     std::size_t max_iterations,
                     bool has_initial_guess,
                     bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    int use_betas = (betas != nullptr) ? 1 : 0;
    const double* d_betas = betas;
    double beta_scalar = beta;

    cache.ensure_buffers(num_vertices);

    double* d_x0 = cache.d_x0;
    double* d_x1 = cache.d_x1;
    double* d_diff = cache.d_diff;

    double* x_old = d_x0;
    double* x_new = d_x1;
    bool converged = false;
    size_t iterations = 0;

    int grid_size = std::min((int64_t)((num_vertices + 255) / 256), (int64_t)(58 * 6));
    if (grid_size < 1) grid_size = 1;

    double h_diff;
    bool use_cusparse = (num_edges > 200000);

    
    if (!has_initial_guess && max_iterations > 0) {
        cudaMemsetAsync(d_diff, 0, sizeof(double));
        katz_first_iter_kernel<BS><<<grid_size, BS>>>(
            x_new, d_betas, beta_scalar, use_betas,
            d_diff, num_vertices);

        cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);
        iterations++;

        if (h_diff < epsilon) {
            converged = true;
        } else {
            std::swap(x_old, x_new);
        }
    } else if (has_initial_guess) {
        cudaMemcpyAsync(d_x0, centralities,
                        (size_t)num_vertices * sizeof(double), cudaMemcpyDeviceToDevice);
    } else {
        cudaMemsetAsync(d_x0, 0, (size_t)num_vertices * sizeof(double));
    }

    if (!converged && use_cusparse && iterations < max_iterations) {
        
        cache.ensureCuSparseSetup(num_vertices, num_edges,
                            offsets, indices, edge_weights,
                            x_old, x_new, alpha);

        for (; iterations < max_iterations; iterations++) {
            cusparseDnVecSetValues(cache.cached_x_descr_, x_old);
            cusparseDnVecSetValues(cache.cached_y_descr_, x_new);

            
            cusparseSpMV(cache.cusparse_handle_,
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         cache.d_alpha_, cache.cached_mat_, cache.cached_x_descr_,
                         cache.d_zero_, cache.cached_y_descr_,
                         CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG2,
                         cache.cached_buffer_);

            
            cudaMemsetAsync(d_diff, 0, sizeof(double));
            add_beta_diff_kernel<BS><<<grid_size, BS>>>(
                x_new, x_old, d_betas, beta_scalar,
                use_betas, d_diff, num_vertices);

            cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);

            if (h_diff < epsilon) {
                converged = true;
                iterations++;
                break;
            }

            std::swap(x_old, x_new);
        }
    } else if (!converged) {
        
        for (; iterations < max_iterations; iterations++) {
            cudaMemsetAsync(d_diff, 0, sizeof(double));

            katz_iteration_kernel<BS><<<grid_size, BS>>>(
                offsets, indices, edge_weights,
                x_old, x_new,
                d_betas, alpha, beta_scalar, use_betas,
                d_diff, num_vertices);

            cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);

            if (h_diff < epsilon) {
                converged = true;
                iterations++;
                break;
            }

            std::swap(x_old, x_new);
        }
    }

    
    double* result_ptr = converged ? x_new : x_old;

    
    if (normalize) {
        cudaMemsetAsync(d_diff, 0, sizeof(double));
        compute_norm_sq_kernel<BS><<<grid_size, BS>>>(result_ptr, d_diff, num_vertices);
        double h_norm_sq;
        cudaMemcpy(&h_norm_sq, d_diff, sizeof(double), cudaMemcpyDeviceToHost);
        if (h_norm_sq > 0.0) {
            normalize_kernel<BS><<<grid_size, BS>>>(result_ptr, 1.0 / std::sqrt(h_norm_sq), num_vertices);
        }
    }

    
    cudaMemcpyAsync(centralities, result_ptr,
                    (size_t)num_vertices * sizeof(double), cudaMemcpyDeviceToDevice);

    return katz_centrality_result_t{iterations, converged};
}

}  
