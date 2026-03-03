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
#include <cstddef>
#include <algorithm>
#include <vector>

namespace aai {

namespace {

#define BLOCK_SIZE 256



__global__ void compute_out_weights_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ out_weights,
    int32_t num_edges)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        atomicAdd(&out_weights[indices[tid]], weights[tid]);
    }
}

__global__ void compute_modified_weights_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ out_weights,
    float* __restrict__ modified_weights,
    float alpha,
    int32_t num_edges)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        int32_t src = indices[tid];
        float ow = out_weights[src];
        modified_weights[tid] = alpha * weights[tid] / ow;
    }
}

__global__ void build_dangling_list_kernel(
    const float* __restrict__ out_weights,
    int32_t* __restrict__ dangling_list,
    int32_t* __restrict__ dangling_count,
    int32_t num_vertices)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices && out_weights[tid] == 0.0f) {
        int32_t pos = atomicAdd(dangling_count, 1);
        dangling_list[pos] = tid;
    }
}

__global__ void scatter_pers_kernel(
    const int32_t* __restrict__ pers_vertices,
    const float* __restrict__ pers_values,
    float* __restrict__ pers_norm,
    float inv_sum,
    int32_t pers_size)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pers_size) {
        pers_norm[pers_vertices[tid]] = pers_values[tid] * inv_sum;
    }
}

__global__ void init_pr_kernel(float* __restrict__ pr, float val, int32_t n)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        pr[tid] = val;
    }
}

__global__ void dangling_sum_sparse_kernel(
    const float* __restrict__ pr,
    const int32_t* __restrict__ dangling_list,
    float* __restrict__ result,
    int32_t D)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float val = 0.0f;
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;
    for (int32_t i = tid; i < D; i += stride) {
        val += pr[dangling_list[i]];
    }

    float block_sum = BlockReduce(temp).Sum(val);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(result, block_sum);
    }
}

__global__ void sparse_fixup_kernel(
    float* __restrict__ pr_new,
    const int32_t* __restrict__ pers_vertices,
    const float* __restrict__ pers_norm,
    float alpha,
    float one_minus_alpha,
    const float* __restrict__ d_dsum,
    int32_t pers_size)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < pers_size) {
        float base_factor = alpha * d_dsum[0] + one_minus_alpha;
        int32_t v = pers_vertices[tid];
        pr_new[v] += base_factor * pers_norm[v];
    }
}

__global__ void diff_kernel(
    const float* __restrict__ pr_new,
    const float* __restrict__ pr_old,
    float* __restrict__ result,
    int32_t n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float val = 0.0f;
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = gridDim.x * blockDim.x;
    for (int32_t i = tid; i < n; i += stride) {
        val += fabsf(pr_new[i] - pr_old[i]);
    }

    float block_sum = BlockReduce(temp).Sum(val);
    if (threadIdx.x == 0 && block_sum > 0.0f) {
        atomicAdd(result, block_sum);
    }
}

__global__ void spmv_thread_per_row_kernel(
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_ind,
    const float* __restrict__ values,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t num_rows)
{
    int32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int32_t row_start = row_ptr[row];
    int32_t row_end = row_ptr[row + 1];

    float sum = 0.0f;
    for (int32_t i = row_start; i < row_end; i++) {
        sum += values[i] * x[col_ind[i]];
    }
    y[row] = sum;
}



struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    float* h_diff_pinned = nullptr;

    float* out_weights = nullptr;
    float* pers_norm = nullptr;
    int32_t* dangling_list = nullptr;
    float* pr0 = nullptr;
    float* pr1 = nullptr;
    int64_t vert_capacity = 0;

    float* modified_weights = nullptr;
    int64_t edge_capacity = 0;

    int32_t* dangling_count = nullptr;
    float* dsum = nullptr;
    float* diff = nullptr;

    void* spmv_buffer = nullptr;
    size_t spmv_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cudaMallocHost(&h_diff_pinned, sizeof(float));
        cudaMalloc(&dangling_count, sizeof(int32_t));
        cudaMalloc(&dsum, sizeof(float));
        cudaMalloc(&diff, sizeof(float));
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (h_diff_pinned) cudaFreeHost(h_diff_pinned);
        if (out_weights) cudaFree(out_weights);
        if (pers_norm) cudaFree(pers_norm);
        if (dangling_list) cudaFree(dangling_list);
        if (pr0) cudaFree(pr0);
        if (pr1) cudaFree(pr1);
        if (modified_weights) cudaFree(modified_weights);
        if (dangling_count) cudaFree(dangling_count);
        if (dsum) cudaFree(dsum);
        if (diff) cudaFree(diff);
        if (spmv_buffer) cudaFree(spmv_buffer);
    }

    void ensure_vertices(int64_t n) {
        if (vert_capacity < n) {
            if (out_weights) cudaFree(out_weights);
            if (pers_norm) cudaFree(pers_norm);
            if (dangling_list) cudaFree(dangling_list);
            if (pr0) cudaFree(pr0);
            if (pr1) cudaFree(pr1);
            cudaMalloc(&out_weights, n * sizeof(float));
            cudaMalloc(&pers_norm, n * sizeof(float));
            cudaMalloc(&dangling_list, n * sizeof(int32_t));
            cudaMalloc(&pr0, n * sizeof(float));
            cudaMalloc(&pr1, n * sizeof(float));
            vert_capacity = n;
        }
    }

    void ensure_edges(int64_t m) {
        if (edge_capacity < m) {
            if (modified_weights) cudaFree(modified_weights);
            cudaMalloc(&modified_weights, m * sizeof(float));
            edge_capacity = m;
        }
    }

    void ensure_spmv(size_t size) {
        if (spmv_capacity < size) {
            if (spmv_buffer) cudaFree(spmv_buffer);
            cudaMalloc(&spmv_buffer, size);
            spmv_capacity = size;
        }
    }
};

}  

PageRankResult personalized_pagerank(const graph32_t& graph,
                                     const float* edge_weights,
                                     const int32_t* personalization_vertices,
                                     const float* personalization_values,
                                     std::size_t personalization_size,
                                     float* pageranks,
                                     const float* precomputed_vertex_out_weight_sums,
                                     float alpha,
                                     float epsilon,
                                     std::size_t max_iterations,
                                     const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    (void)precomputed_vertex_out_weight_sums;

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    float one_minus_alpha = 1.0f - alpha;
    cudaStream_t stream = nullptr;

    cache.ensure_vertices(num_vertices);
    cache.ensure_edges(num_edges);

    float* d_out_weights = cache.out_weights;
    float* d_modified_weights = cache.modified_weights;
    float* d_pers_norm = cache.pers_norm;
    int32_t* d_dangling_list = cache.dangling_list;
    int32_t* d_dangling_count = cache.dangling_count;
    float* d_pr0 = cache.pr0;
    float* d_pr1 = cache.pr1;
    float* d_dsum = cache.dsum;
    float* d_diff = cache.diff;

    
    cudaMemset(d_out_weights, 0, num_vertices * sizeof(float));
    {
        int grid = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_out_weights_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_indices, edge_weights, d_out_weights, num_edges);
    }
    {
        int grid = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_modified_weights_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_indices, edge_weights, d_out_weights, d_modified_weights, alpha, num_edges);
    }

    cudaMemset(d_dangling_count, 0, sizeof(int32_t));
    {
        int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        build_dangling_list_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_out_weights, d_dangling_list, d_dangling_count, num_vertices);
    }
    int32_t h_dangling_count = 0;
    cudaMemcpy(&h_dangling_count, d_dangling_count, sizeof(int32_t), cudaMemcpyDeviceToHost);
    bool has_dangling = (h_dangling_count > 0);

    cudaMemset(d_pers_norm, 0, num_vertices * sizeof(float));
    std::vector<float> h_pers(personalization_size);
    cudaMemcpy(h_pers.data(), personalization_values,
               personalization_size * sizeof(float), cudaMemcpyDeviceToHost);
    double pers_sum = 0.0;
    for (std::size_t i = 0; i < personalization_size; i++) pers_sum += (double)h_pers[i];
    float inv_pers_sum = (pers_sum > 0.0) ? (float)(1.0 / pers_sum) : 0.0f;
    if (personalization_size > 0) {
        int grid = ((int32_t)personalization_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scatter_pers_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            personalization_vertices, personalization_values, d_pers_norm,
            inv_pers_sum, (int32_t)personalization_size);
    }

    float* d_current = d_pr0;
    float* d_next = d_pr1;
    if (initial_pageranks) {
        cudaMemcpy(d_current, initial_pageranks,
                   num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_pr_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_current, 1.0f / (float)num_vertices, num_vertices);
    }
    cudaMemset(d_dsum, 0, sizeof(float));

    
    float avg_degree = (float)num_edges / (float)num_vertices;
    bool use_cusparse = (avg_degree > 4.0f);

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    float h_one = 1.0f, h_zero = 0.0f;

    if (use_cusparse) {
        cusparseCreateCsr(&matA, num_vertices, num_vertices, num_edges,
                          (void*)d_offsets, (void*)d_indices, (void*)d_modified_weights,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cusparseCreateDnVec(&vecX, num_vertices, d_current, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, num_vertices, d_next, CUDA_R_32F);

        size_t bufferSize = 0;
        cusparseSpMV_bufferSize(cache.cusparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &h_one, matA, vecX, &h_zero, vecY,
                                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                &bufferSize);
        if (bufferSize > 0) {
            cache.ensure_spmv(bufferSize);
        }

        cusparseSpMV_preprocess(cache.cusparse_handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &h_one, matA, vecX, &h_zero, vecY,
                                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                cache.spmv_buffer);
    }

    
    bool converged = false;
    std::size_t iterations = 0;
    const int CHECK_INTERVAL = 4;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        if (has_dangling) {
            cudaMemset(d_dsum, 0, sizeof(float));
            if (h_dangling_count > 0) {
                int grid = (h_dangling_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
                if (grid > 1024) grid = 1024;
                dangling_sum_sparse_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                    d_current, d_dangling_list, d_dsum, h_dangling_count);
            }
        }

        if (use_cusparse) {
            cusparseDnVecSetValues(vecX, d_current);
            cusparseDnVecSetValues(vecY, d_next);
            cusparseSpMV(cache.cusparse_handle,
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &h_one, matA, vecX, &h_zero, vecY,
                         CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                         cache.spmv_buffer);
        } else {
            int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
            spmv_thread_per_row_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                d_offsets, d_indices, d_modified_weights,
                d_current, d_next, num_vertices);
        }

        if (personalization_size > 0) {
            int grid = ((int32_t)personalization_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sparse_fixup_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                d_next, personalization_vertices, d_pers_norm,
                alpha, one_minus_alpha, d_dsum, (int32_t)personalization_size);
        }

        bool check_now;
        if (iter < 8) {
            check_now = true;
        } else {
            check_now = ((iter + 1) % CHECK_INTERVAL == 0) || (iter + 1 >= max_iterations);
        }

        if (check_now) {
            cudaMemset(d_diff, 0, sizeof(float));
            {
                int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
                if (grid > 1024) grid = 1024;
                diff_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                    d_next, d_current, d_diff, num_vertices);
            }
            cudaMemcpy(cache.h_diff_pinned, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
            iterations = iter + 1;
            if (*cache.h_diff_pinned < epsilon) {
                converged = true;
                std::swap(d_current, d_next);
                break;
            }
        }

        std::swap(d_current, d_next);
    }

    if (!converged) iterations = max_iterations;

    cudaMemcpy(pageranks, d_current, num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);

    if (use_cusparse) {
        cusparseDestroySpMat(matA);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
    }

    return PageRankResult{iterations, converged};
}

}  
