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
#include <cstdlib>
#include <vector>

namespace aai {

namespace {

#define BLOCK_SIZE 256

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t _status = (call); \
    if (_status != CUSPARSE_STATUS_SUCCESS) { \
        std::abort(); \
    } \
} while(0)

struct Cache : Cacheable {
    cusparseHandle_t cusparse_h = nullptr;

    double* pr_buf0 = nullptr;
    int64_t pr_buf0_capacity = 0;

    double* pr_buf1 = nullptr;
    int64_t pr_buf1_capacity = 0;

    double* out_weight = nullptr;
    int64_t out_weight_capacity = 0;

    double* dangling_mask = nullptr;
    int64_t dangling_mask_capacity = 0;

    double* pers_norm = nullptr;
    int64_t pers_norm_capacity = 0;

    double* mod_weights = nullptr;
    int64_t mod_weights_capacity = 0;

    double* scalar_buf = nullptr;  
    int64_t scalar_buf_capacity = 0;

    void* spmv_buf = nullptr;
    size_t spmv_buf_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_h);
    }

    ~Cache() override {
        if (cusparse_h) cusparseDestroy(cusparse_h);
        if (pr_buf0) cudaFree(pr_buf0);
        if (pr_buf1) cudaFree(pr_buf1);
        if (out_weight) cudaFree(out_weight);
        if (dangling_mask) cudaFree(dangling_mask);
        if (pers_norm) cudaFree(pers_norm);
        if (mod_weights) cudaFree(mod_weights);
        if (scalar_buf) cudaFree(scalar_buf);
        if (spmv_buf) cudaFree(spmv_buf);
    }

    void ensure_vertex_buffers(int64_t n) {
        if (pr_buf0_capacity < n) {
            if (pr_buf0) cudaFree(pr_buf0);
            cudaMalloc(&pr_buf0, n * sizeof(double));
            pr_buf0_capacity = n;
        }
        if (pr_buf1_capacity < n) {
            if (pr_buf1) cudaFree(pr_buf1);
            cudaMalloc(&pr_buf1, n * sizeof(double));
            pr_buf1_capacity = n;
        }
        if (out_weight_capacity < n) {
            if (out_weight) cudaFree(out_weight);
            cudaMalloc(&out_weight, n * sizeof(double));
            out_weight_capacity = n;
        }
        if (dangling_mask_capacity < n) {
            if (dangling_mask) cudaFree(dangling_mask);
            cudaMalloc(&dangling_mask, n * sizeof(double));
            dangling_mask_capacity = n;
        }
        if (pers_norm_capacity < n) {
            if (pers_norm) cudaFree(pers_norm);
            cudaMalloc(&pers_norm, n * sizeof(double));
            pers_norm_capacity = n;
        }
        if (scalar_buf_capacity < 2) {
            if (scalar_buf) cudaFree(scalar_buf);
            cudaMalloc(&scalar_buf, 2 * sizeof(double));
            scalar_buf_capacity = 2;
        }
    }

    void ensure_edge_buffer(int64_t ne) {
        int64_t sz = (ne > 0) ? ne : 1;
        if (mod_weights_capacity < sz) {
            if (mod_weights) cudaFree(mod_weights);
            cudaMalloc(&mod_weights, sz * sizeof(double));
            mod_weights_capacity = sz;
        }
    }

    void ensure_spmv_buffer(size_t sz) {
        if (sz == 0) sz = 1;
        if (spmv_buf_capacity < sz) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, sz);
            spmv_buf_capacity = sz;
        }
    }
};


__global__ void compute_out_weights_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    double* __restrict__ out_weight,
    int32_t num_edges)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < num_edges) {
        atomicAdd(&out_weight[indices[idx]], weights[idx]);
    }
}


__global__ void compute_modified_weights_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const double* __restrict__ out_weight,
    double* __restrict__ mod_weights,
    int32_t num_edges)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < num_edges) {
        int32_t src = indices[idx];
        double ow = out_weight[src];
        mod_weights[idx] = (ow > 0.0) ? (edge_weights[idx] / ow) : 0.0;
    }
}


__global__ void build_dangling_mask_kernel(
    const double* __restrict__ out_weight,
    double* __restrict__ dangling_mask,
    int32_t num_vertices)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < num_vertices) {
        dangling_mask[idx] = (out_weight[idx] == 0.0) ? 1.0 : 0.0;
    }
}


__global__ void build_pers_norm_kernel(
    const int32_t* __restrict__ pers_vertices,
    const double* __restrict__ pers_values,
    double* __restrict__ pers_norm,
    double pers_sum_inv,
    int32_t pers_size)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < pers_size) {
        pers_norm[pers_vertices[idx]] = pers_values[idx] * pers_sum_inv;
    }
}


__global__ void init_pr_kernel(double* __restrict__ pr, double val, int32_t n)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < n) {
        pr[idx] = val;
    }
}


__global__ void dangling_sum_kernel(
    const double* __restrict__ pr,
    const double* __restrict__ dangling_mask,
    double* __restrict__ dangling_sum,
    int32_t num_vertices)
{
    __shared__ double sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;

    double val = (idx < num_vertices) ? (pr[idx] * dangling_mask[idx]) : 0.0;
    sdata[tid] = val;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && sdata[0] != 0.0) {
        atomicAdd(dangling_sum, sdata[0]);
    }
}




__global__ void add_base_and_diff_kernel(
    double* __restrict__ new_pr,
    const double* __restrict__ old_pr,
    const double* __restrict__ pers_norm,
    const double* __restrict__ dangling_sum_ptr,
    double* __restrict__ diff_ptr,
    double alpha,
    int32_t num_vertices)
{
    __shared__ double sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;

    double base_factor = alpha * (*dangling_sum_ptr) + (1.0 - alpha);

    double local_diff = 0.0;
    if (idx < num_vertices) {
        double new_val = new_pr[idx] + base_factor * pers_norm[idx];
        new_pr[idx] = new_val;
        local_diff = fabs(new_val - old_pr[idx]);
    }

    sdata[tid] = local_diff;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0 && sdata[0] != 0.0) {
        atomicAdd(diff_ptr, sdata[0]);
    }
}

}  

PageRankResult personalized_pagerank(const graph32_t& graph,
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

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    cudaStream_t stream = 0;
    cusparseSetStream(cache.cusparse_h, stream);

    
    cache.ensure_vertex_buffers(num_vertices);
    cache.ensure_edge_buffer(num_edges);

    double* d_pr0 = cache.pr_buf0;
    double* d_pr1 = cache.pr_buf1;
    double* d_out_w = cache.out_weight;
    double* d_dang = cache.dangling_mask;
    double* d_pers_norm = cache.pers_norm;
    double* d_mod_w = cache.mod_weights;
    double* d_dang_sum = cache.scalar_buf;
    double* d_diff = cache.scalar_buf + 1;

    
    cudaMemsetAsync(d_out_w, 0, num_vertices * sizeof(double), stream);
    cudaMemsetAsync(d_pers_norm, 0, num_vertices * sizeof(double), stream);

    
    {
        int grid = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid > 0)
            compute_out_weights_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                d_indices, edge_weights, d_out_w, num_edges);
    }

    
    {
        int grid = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid > 0)
            compute_modified_weights_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                d_indices, edge_weights, d_out_w, d_mod_w, num_edges);
    }

    
    {
        int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        build_dangling_mask_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_out_w, d_dang, num_vertices);
    }

    
    {
        std::vector<double> h_pv(personalization_size);
        cudaMemcpy(h_pv.data(), personalization_values,
                   personalization_size * sizeof(double), cudaMemcpyDeviceToHost);
        double psum = 0;
        for (std::size_t i = 0; i < personalization_size; i++) psum += h_pv[i];
        double psi = (psum > 0) ? 1.0 / psum : 0.0;
        int grid = (personalization_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid > 0)
            build_pers_norm_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                personalization_vertices, personalization_values, d_pers_norm,
                psi, (int32_t)personalization_size);
    }

    
    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(d_pr0, initial_pageranks,
            num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else {
        int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_pr_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_pr0, 1.0 / num_vertices, num_vertices);
    }

    
    cusparseSpMatDescr_t mat;
    CUSPARSE_CHECK(cusparseCreateCsr(&mat, num_vertices, num_vertices, num_edges,
        (void*)d_offsets, (void*)d_indices, (void*)d_mod_w,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseDnVecDescr_t vec_in, vec_out;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_in, num_vertices, d_pr0, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_out, num_vertices, d_pr1, CUDA_R_64F));

    double h_alpha = alpha, h_beta = 0.0;
    size_t buf_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(cache.cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha, mat, vec_in, &h_beta, vec_out,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size));

    cache.ensure_spmv_buffer(buf_size);

    CUSPARSE_CHECK(cusparseSpMV_preprocess(cache.cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha, mat, vec_in, &h_beta, vec_out,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf));

    
    double* d_old = d_pr0;
    double* d_new = d_pr1;
    size_t iterations = 0;
    bool converged = false;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(d_dang_sum, 0, sizeof(double), stream);
        {
            int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
            dangling_sum_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                d_old, d_dang, d_dang_sum, num_vertices);
        }

        
        CUSPARSE_CHECK(cusparseDnVecSetValues(vec_in, d_old));
        CUSPARSE_CHECK(cusparseDnVecSetValues(vec_out, d_new));
        CUSPARSE_CHECK(cusparseSpMV(cache.cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_alpha, mat, vec_in, &h_beta, vec_out,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf));

        
        cudaMemsetAsync(d_diff, 0, sizeof(double), stream);
        {
            int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
            add_base_and_diff_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                d_new, d_old, d_pers_norm, d_dang_sum,
                d_diff, alpha, num_vertices);
        }

        
        std::swap(d_old, d_new);
        iterations = iter + 1;

        
        double h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);
        if (h_diff < epsilon) {
            converged = true;
            break;
        }
    }

    
    cudaMemcpyAsync(pageranks, d_old,
        num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    
    cusparseDestroySpMat(mat);
    cusparseDestroyDnVec(vec_in);
    cusparseDestroyDnVec(vec_out);

    return PageRankResult{iterations, converged};
}

}  
