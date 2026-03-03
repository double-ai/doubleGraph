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
#include <utility>
#include <vector>

namespace aai {

namespace {

#define BLOCK_SIZE 256

static constexpr int CONV_CHECK_INTERVAL = 4;





struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;

    double* out_weights = nullptr;
    double* precomp_w = nullptr;
    int32_t* dangling_idx = nullptr;
    double* pers_norm = nullptr;
    double* spmv = nullptr;
    double* pr_temp = nullptr;
    double* scalars = nullptr;   
    int* int_scalar = nullptr;   
    uint8_t* spmv_buffer = nullptr;

    int64_t out_weights_cap = 0;
    int64_t precomp_w_cap = 0;
    int64_t dangling_idx_cap = 0;
    int64_t pers_norm_cap = 0;
    int64_t spmv_cap = 0;
    int64_t pr_temp_cap = 0;
    bool scalars_allocated = false;
    int64_t spmv_buffer_cap = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (out_weights) cudaFree(out_weights);
        if (precomp_w) cudaFree(precomp_w);
        if (dangling_idx) cudaFree(dangling_idx);
        if (pers_norm) cudaFree(pers_norm);
        if (spmv) cudaFree(spmv);
        if (pr_temp) cudaFree(pr_temp);
        if (scalars) cudaFree(scalars);
        if (int_scalar) cudaFree(int_scalar);
        if (spmv_buffer) cudaFree(spmv_buffer);
    }

    void ensure(int32_t nv, int32_t ne) {
        if (out_weights_cap < nv) {
            if (out_weights) cudaFree(out_weights);
            cudaMalloc(&out_weights, (size_t)nv * sizeof(double));
            out_weights_cap = nv;
        }
        if (precomp_w_cap < ne) {
            if (precomp_w) cudaFree(precomp_w);
            cudaMalloc(&precomp_w, (size_t)ne * sizeof(double));
            precomp_w_cap = ne;
        }
        if (dangling_idx_cap < nv) {
            if (dangling_idx) cudaFree(dangling_idx);
            cudaMalloc(&dangling_idx, (size_t)nv * sizeof(int32_t));
            dangling_idx_cap = nv;
        }
        if (pers_norm_cap < nv) {
            if (pers_norm) cudaFree(pers_norm);
            cudaMalloc(&pers_norm, (size_t)nv * sizeof(double));
            pers_norm_cap = nv;
        }
        if (spmv_cap < nv) {
            if (spmv) cudaFree(spmv);
            cudaMalloc(&spmv, (size_t)nv * sizeof(double));
            spmv_cap = nv;
        }
        if (pr_temp_cap < nv) {
            if (pr_temp) cudaFree(pr_temp);
            cudaMalloc(&pr_temp, (size_t)nv * sizeof(double));
            pr_temp_cap = nv;
        }
        if (!scalars_allocated) {
            cudaMalloc(&scalars, 4 * sizeof(double));
            cudaMalloc(&int_scalar, sizeof(int));
            scalars_allocated = true;
        }
    }

    void ensure_spmv_buf(size_t size) {
        if (spmv_buffer_cap < (int64_t)size) {
            if (spmv_buffer) cudaFree(spmv_buffer);
            cudaMalloc(&spmv_buffer, size);
            spmv_buffer_cap = (int64_t)size;
        }
    }
};





__global__ void compute_out_weights_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    double* __restrict__ out_weights,
    int num_edges)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges;
         idx += gridDim.x * blockDim.x) {
        atomicAdd(&out_weights[__ldg(&indices[idx])], __ldg(&edge_weights[idx]));
    }
}

__global__ void precompute_weights_kernel(
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ indices,
    const double* __restrict__ out_weights,
    double* __restrict__ precomputed_weights,
    int num_edges)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges;
         idx += gridDim.x * blockDim.x) {
        int src = __ldg(&indices[idx]);
        double ow = __ldg(&out_weights[src]);
        precomputed_weights[idx] = (ow != 0.0) ? (__ldg(&edge_weights[idx]) / ow) : 0.0;
    }
}

__global__ void find_dangling_kernel(
    const double* __restrict__ out_weights,
    int32_t* __restrict__ dangling_indices,
    int* __restrict__ d_num_dangling,
    int num_vertices)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices;
         idx += gridDim.x * blockDim.x) {
        if (__ldg(&out_weights[idx]) == 0.0) {
            int pos = atomicAdd(d_num_dangling, 1);
            dangling_indices[pos] = idx;
        }
    }
}

__global__ void scatter_pers_kernel(
    const int32_t* __restrict__ pers_vertices,
    const double* __restrict__ pers_values,
    double* __restrict__ pers_norm,
    int pers_size,
    double inv_pers_sum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pers_size) {
        pers_norm[__ldg(&pers_vertices[idx])] = __ldg(&pers_values[idx]) * inv_pers_sum;
    }
}

__global__ void init_pr_uniform_kernel(
    double* __restrict__ pr, int num_vertices, double val)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices;
         idx += gridDim.x * blockDim.x) {
        pr[idx] = val;
    }
}

__global__ void compute_dangling_sum_kernel(
    const double* __restrict__ pr,
    const int32_t* __restrict__ dangling_indices,
    double* __restrict__ d_dangling_sum,
    int num_dangling)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage temp;

    double val = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_dangling;
         i += gridDim.x * blockDim.x) {
        val += __ldg(&pr[__ldg(&dangling_indices[i])]);
    }

    double bsum = BR(temp).Sum(val);
    if (threadIdx.x == 0 && bsum != 0.0) {
        atomicAdd(d_dangling_sum, bsum);
    }
}

__global__ void postprocess_with_diff_kernel(
    const double* __restrict__ spmv_result,
    const double* __restrict__ pers_norm,
    const double* __restrict__ old_pr,
    double* __restrict__ new_pr,
    const double* __restrict__ d_dangling_sum,
    double* __restrict__ d_diff,
    double alpha,
    double one_minus_alpha,
    int num_vertices)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage temp;

    double base = alpha * __ldg(d_dangling_sum) + one_minus_alpha;

    double my_diff = 0.0;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices;
         idx += gridDim.x * blockDim.x) {
        double pn = alpha * __ldg(&spmv_result[idx]) + base * __ldg(&pers_norm[idx]);
        new_pr[idx] = pn;
        my_diff += fabs(pn - __ldg(&old_pr[idx]));
    }

    double bsum = BR(temp).Sum(my_diff);
    if (threadIdx.x == 0 && bsum != 0.0) {
        atomicAdd(d_diff, bsum);
    }
}

__global__ void postprocess_no_diff_kernel(
    const double* __restrict__ spmv_result,
    const double* __restrict__ pers_norm,
    const double* __restrict__ old_pr,
    double* __restrict__ new_pr,
    const double* __restrict__ d_dangling_sum,
    double alpha,
    double one_minus_alpha,
    int num_vertices)
{
    double base = alpha * __ldg(d_dangling_sum) + one_minus_alpha;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_vertices;
         idx += gridDim.x * blockDim.x) {
        new_pr[idx] = alpha * __ldg(&spmv_result[idx]) + base * __ldg(&pers_norm[idx]);
    }
}





static inline int clamp_grid(int needed, int max_blocks) {
    return (needed < max_blocks) ? ((needed > 0) ? needed : 1) : max_blocks;
}

static void launch_compute_out_weights(const int32_t* indices, const double* edge_weights,
                                       double* out_weights, int num_edges) {
    if (num_edges <= 0) return;
    int grid = clamp_grid((num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, 2048);
    compute_out_weights_kernel<<<grid, BLOCK_SIZE>>>(indices, edge_weights, out_weights, num_edges);
}

static void launch_precompute_weights(const double* edge_weights, const int32_t* indices,
                                      const double* out_weights, double* precomputed_weights,
                                      int num_edges) {
    if (num_edges <= 0) return;
    int grid = clamp_grid((num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, 2048);
    precompute_weights_kernel<<<grid, BLOCK_SIZE>>>(edge_weights, indices, out_weights, precomputed_weights, num_edges);
}

static void launch_find_dangling(const double* out_weights, int32_t* dangling_indices,
                                 int* d_num_dangling, int num_vertices) {
    if (num_vertices <= 0) return;
    int grid = clamp_grid((num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE, 2048);
    find_dangling_kernel<<<grid, BLOCK_SIZE>>>(out_weights, dangling_indices, d_num_dangling, num_vertices);
}

static void launch_scatter_pers(const int32_t* pers_vertices, const double* pers_values,
                                double* pers_norm, int pers_size, double inv_pers_sum) {
    if (pers_size <= 0) return;
    int grid = (pers_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scatter_pers_kernel<<<grid, BLOCK_SIZE>>>(pers_vertices, pers_values, pers_norm, pers_size, inv_pers_sum);
}

static void launch_init_pr(double* pr, int num_vertices, double val) {
    if (num_vertices <= 0) return;
    int grid = clamp_grid((num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE, 2048);
    init_pr_uniform_kernel<<<grid, BLOCK_SIZE>>>(pr, num_vertices, val);
}

static void launch_compute_dangling_sum(const double* pr, const int32_t* dangling_indices,
                                        double* d_dangling_sum, int num_dangling) {
    if (num_dangling <= 0) return;
    int grid = clamp_grid((num_dangling + BLOCK_SIZE - 1) / BLOCK_SIZE, 512);
    compute_dangling_sum_kernel<<<grid, BLOCK_SIZE>>>(pr, dangling_indices, d_dangling_sum, num_dangling);
}

static void launch_postprocess_with_diff(const double* spmv_result, const double* pers_norm,
                                         const double* old_pr, double* new_pr,
                                         const double* d_dangling_sum, double* d_diff,
                                         double alpha, double one_minus_alpha, int num_vertices) {
    if (num_vertices <= 0) return;
    int grid = clamp_grid((num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE, 2048);
    postprocess_with_diff_kernel<<<grid, BLOCK_SIZE>>>(spmv_result, pers_norm, old_pr, new_pr,
                                                       d_dangling_sum, d_diff, alpha, one_minus_alpha, num_vertices);
}

static void launch_postprocess_no_diff(const double* spmv_result, const double* pers_norm,
                                       const double* old_pr, double* new_pr,
                                       const double* d_dangling_sum,
                                       double alpha, double one_minus_alpha, int num_vertices) {
    if (num_vertices <= 0) return;
    int grid = clamp_grid((num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE, 2048);
    postprocess_no_diff_kernel<<<grid, BLOCK_SIZE>>>(spmv_result, pers_norm, old_pr, new_pr,
                                                     d_dangling_sum, alpha, one_minus_alpha, num_vertices);
}

}  





PageRankResult personalized_pagerank_seg(const graph32_t& graph,
                                         const double* edge_weights,
                                         const int32_t* personalization_vertices,
                                         const double* personalization_values,
                                         std::size_t personalization_size,
                                         double* pageranks,
                                         const double* precomputed_vertex_out_weight_sums,
                                         double alpha,
                                         double epsilon,
                                         std::size_t max_iterations,
                                         const double* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    double one_minus_alpha = 1.0 - alpha;

    cache.ensure(num_vertices, num_edges);

    double* d_out_weights = cache.out_weights;
    double* d_precomp_w = cache.precomp_w;
    int32_t* d_dangling_idx = cache.dangling_idx;
    double* d_pers_norm = cache.pers_norm;
    double* d_spmv = cache.spmv;
    double* d_scalars = cache.scalars;
    double* d_dangling_sum = d_scalars;
    double* d_diff = d_scalars + 1;
    double* d_alpha_spmv = d_scalars + 2;
    double* d_beta_spmv = d_scalars + 3;
    int* d_num_dangling = cache.int_scalar;

    
    double* d_pr_a = pageranks;
    double* d_pr_b = cache.pr_temp;

    
    cudaMemset(d_out_weights, 0, num_vertices * sizeof(double));
    cudaMemset(d_pers_norm, 0, num_vertices * sizeof(double));
    cudaMemset(d_num_dangling, 0, sizeof(int));

    launch_compute_out_weights(d_indices, edge_weights, d_out_weights, num_edges);
    launch_precompute_weights(edge_weights, d_indices, d_out_weights, d_precomp_w, num_edges);
    launch_find_dangling(d_out_weights, d_dangling_idx, d_num_dangling, num_vertices);

    int h_num_dangling = 0;
    cudaMemcpy(&h_num_dangling, d_num_dangling, sizeof(int), cudaMemcpyDeviceToHost);

    
    {
        std::vector<double> h_pers(personalization_size);
        cudaMemcpy(h_pers.data(), personalization_values,
                   personalization_size * sizeof(double), cudaMemcpyDeviceToHost);
        double pers_sum = 0.0;
        for (std::size_t i = 0; i < personalization_size; i++) pers_sum += h_pers[i];
        double inv_pers_sum = (pers_sum > 0.0) ? (1.0 / pers_sum) : 0.0;
        launch_scatter_pers(personalization_vertices, personalization_values,
                            d_pers_norm, (int)personalization_size, inv_pers_sum);
    }

    
    double* d_pr_old = d_pr_a;
    double* d_pr_new = d_pr_b;

    if (initial_pageranks != nullptr) {
        cudaMemcpy(d_pr_old, initial_pageranks,
                   num_vertices * sizeof(double), cudaMemcpyDeviceToDevice);
    } else {
        double init_val = 1.0 / (double)num_vertices;
        launch_init_pr(d_pr_old, num_vertices, init_val);
    }

    
    cusparseSpMatDescr_t matDescr = nullptr;
    cusparseCreateCsr(&matDescr,
        (int64_t)num_vertices, (int64_t)num_vertices, (int64_t)num_edges,
        (void*)d_offsets, (void*)d_indices, (void*)d_precomp_w,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    cusparseCreateDnVec(&vecX, (int64_t)num_vertices, d_pr_old, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, (int64_t)num_vertices, d_spmv, CUDA_R_64F);

    cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);

    double h_spmv_params[2] = {1.0, 0.0};
    cudaMemcpy(d_alpha_spmv, &h_spmv_params[0], sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta_spmv, &h_spmv_params[1], sizeof(double), cudaMemcpyHostToDevice);

    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_alpha_spmv, matDescr, vecX, d_beta_spmv, vecY,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

    void* d_spmv_buffer = nullptr;
    if (bufferSize > 0) {
        cache.ensure_spmv_buf(bufferSize);
        d_spmv_buffer = cache.spmv_buffer;
    }

    cusparseSpMV_preprocess(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_alpha_spmv, matDescr, vecX, d_beta_spmv, vecY,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer);

    
    size_t iterations = 0;
    bool converged = false;
    double h_diff = 0.0;
    double* result_ptr = d_pr_old;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        bool check_conv = ((iter + 1) % (size_t)CONV_CHECK_INTERVAL == 0) || (iter == max_iterations - 1);

        if (check_conv) {
            double zeros[2] = {0.0, 0.0};
            cudaMemcpyAsync(d_dangling_sum, zeros, 2 * sizeof(double), cudaMemcpyHostToDevice);
        } else {
            cudaMemsetAsync(d_dangling_sum, 0, sizeof(double));
        }

        if (h_num_dangling > 0) {
            launch_compute_dangling_sum(d_pr_old, d_dangling_idx, d_dangling_sum, h_num_dangling);
        }

        cusparseDnVecSetValues(vecX, d_pr_old);

        cusparseSpMV(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_alpha_spmv, matDescr, vecX, d_beta_spmv, vecY,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer);

        if (check_conv) {
            launch_postprocess_with_diff(d_spmv, d_pers_norm, d_pr_old, d_pr_new,
                                         d_dangling_sum, d_diff, alpha, one_minus_alpha, num_vertices);
        } else {
            launch_postprocess_no_diff(d_spmv, d_pers_norm, d_pr_old, d_pr_new,
                                       d_dangling_sum, alpha, one_minus_alpha, num_vertices);
        }

        result_ptr = d_pr_new;
        iterations = iter + 1;

        if (check_conv) {
            cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);
            if (h_diff < epsilon) {
                converged = true;
                break;
            }
        }

        std::swap(d_pr_old, d_pr_new);
    }

    
    if (result_ptr != pageranks) {
        cudaMemcpy(pageranks, result_ptr,
                   num_vertices * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    
    if (vecX) cusparseDestroyDnVec(vecX);
    if (vecY) cusparseDestroyDnVec(vecY);
    if (matDescr) cusparseDestroySpMat(matDescr);
    cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_HOST);

    return PageRankResult{iterations, converged};
}

}  
