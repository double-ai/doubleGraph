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
#include <cub/device/device_scan.cuh>
#include <cstdint>

namespace aai {

namespace {

struct Cache : Cacheable {
    
    double* h_pinned = nullptr;
    cusparseHandle_t cusparse_handle = nullptr;

    
    double* scalar_buf = nullptr;  
    float* d_cusparse_alpha = nullptr;
    float* d_cusparse_beta = nullptr;

    
    int32_t* active_count = nullptr;
    int64_t active_count_cap = 0;

    int32_t* out_degree = nullptr;
    int64_t out_degree_cap = 0;

    int32_t* new_offsets = nullptr;
    int64_t new_offsets_cap = 0;

    float* pr_buf = nullptr;
    int64_t pr_buf_cap = 0;

    float* pr_norm = nullptr;
    int64_t pr_norm_cap = 0;

    
    int32_t* new_indices = nullptr;
    int64_t new_indices_cap = 0;

    float* cusparse_values = nullptr;
    int64_t cusparse_values_cap = 0;

    void* spmv_buffer = nullptr;
    size_t spmv_buffer_cap = 0;

    Cache() {
        cudaMallocHost(&h_pinned, sizeof(double));
        cusparseCreate(&cusparse_handle);
        cudaMalloc(&scalar_buf, 2 * sizeof(double));
        cudaMalloc(&d_cusparse_alpha, sizeof(float));
        cudaMalloc(&d_cusparse_beta, sizeof(float));
    }

    ~Cache() override {
        if (h_pinned) cudaFreeHost(h_pinned);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (scalar_buf) cudaFree(scalar_buf);
        if (d_cusparse_alpha) cudaFree(d_cusparse_alpha);
        if (d_cusparse_beta) cudaFree(d_cusparse_beta);
        if (active_count) cudaFree(active_count);
        if (out_degree) cudaFree(out_degree);
        if (new_offsets) cudaFree(new_offsets);
        if (pr_buf) cudaFree(pr_buf);
        if (pr_norm) cudaFree(pr_norm);
        if (new_indices) cudaFree(new_indices);
        if (cusparse_values) cudaFree(cusparse_values);
        if (spmv_buffer) cudaFree(spmv_buffer);
    }

    void ensure_vertex_buffers(int32_t n) {
        if (active_count_cap < n) {
            if (active_count) cudaFree(active_count);
            cudaMalloc(&active_count, (size_t)n * sizeof(int32_t));
            active_count_cap = n;
        }
        if (out_degree_cap < n) {
            if (out_degree) cudaFree(out_degree);
            cudaMalloc(&out_degree, (size_t)n * sizeof(int32_t));
            out_degree_cap = n;
        }
        int64_t np1 = (int64_t)n + 1;
        if (new_offsets_cap < np1) {
            if (new_offsets) cudaFree(new_offsets);
            cudaMalloc(&new_offsets, (size_t)np1 * sizeof(int32_t));
            new_offsets_cap = np1;
        }
        if (pr_buf_cap < n) {
            if (pr_buf) cudaFree(pr_buf);
            cudaMalloc(&pr_buf, (size_t)n * sizeof(float));
            pr_buf_cap = n;
        }
        if (pr_norm_cap < n) {
            if (pr_norm) cudaFree(pr_norm);
            cudaMalloc(&pr_norm, (size_t)n * sizeof(float));
            pr_norm_cap = n;
        }
    }

    void ensure_edge_buffers(int64_t ne) {
        if (new_indices_cap < ne) {
            if (new_indices) cudaFree(new_indices);
            cudaMalloc(&new_indices, (size_t)ne * sizeof(int32_t));
            new_indices_cap = ne;
        }
    }

    void ensure_cusparse_values(int64_t ne) {
        if (cusparse_values_cap < ne) {
            if (cusparse_values) cudaFree(cusparse_values);
            cudaMalloc(&cusparse_values, (size_t)ne * sizeof(float));
            cusparse_values_cap = ne;
        }
    }

    void ensure_spmv_buffer(size_t sz) {
        if (spmv_buffer_cap < sz) {
            if (spmv_buffer) cudaFree(spmv_buffer);
            cudaMalloc(&spmv_buffer, sz);
            spmv_buffer_cap = sz;
        }
    }
};



__global__ void preprocess_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_count,
    int32_t* __restrict__ out_degree,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int count = 0;
        for (int e = start; e < end; e++) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1u) {
                count++;
                atomicAdd(&out_degree[indices[e]], 1);
            }
        }
        active_count[v] = count;
    }
}

__global__ void compact_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int write_pos = new_offsets[v];
        for (int e = start; e < end; e++) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1u) {
                new_indices[write_pos++] = indices[e];
            }
        }
    }
}

__global__ void fill_ones_kernel(float* __restrict__ arr, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = 1.0f;
}



__global__ void init_pr_kernel(float* __restrict__ pr, int32_t num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        pr[v] = 1.0f / (float)num_vertices;
    }
}

__global__ void normalize_and_dangling_kernel(
    const float* __restrict__ pr,
    const int32_t* __restrict__ out_degree,
    float* __restrict__ pr_norm,
    double* __restrict__ dangling_sum,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    double my_dangling = 0.0;

    if (v < num_vertices) {
        int deg = out_degree[v];
        float p = pr[v];
        if (deg > 0) {
            pr_norm[v] = p / (float)deg;
        } else {
            pr_norm[v] = 0.0f;
            my_dangling = (double)p;
        }
    }

    double block_sum = BlockReduce(temp_storage).Sum(my_dangling);
    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(dangling_sum, block_sum);
    }
}

__global__ void fill_base_val_kernel(
    float* __restrict__ new_pr,
    const double* __restrict__ d_dangling_sum,
    double one_minus_alpha_div_n,
    double alpha_div_n,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        new_pr[v] = (float)(one_minus_alpha_div_n + alpha_div_n * (*d_dangling_sum));
    }
}

__global__ void diff_kernel(
    const float* __restrict__ old_pr,
    const float* __restrict__ new_pr,
    double* __restrict__ diff_sum,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    double my_diff = 0.0;
    if (v < num_vertices) {
        my_diff = (double)fabsf(new_pr[v] - old_pr[v]);
    }

    double block_diff = BlockReduce(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff != 0.0) {
        atomicAdd(diff_sum, block_diff);
    }
}

__global__ void spmv_and_diff_kernel(
    const int32_t* __restrict__ new_offsets,
    const int32_t* __restrict__ new_indices,
    const float* __restrict__ pr_norm,
    const float* __restrict__ old_pr,
    float* __restrict__ new_pr,
    const double* __restrict__ d_dangling_sum,
    float alpha,
    double one_minus_alpha_div_n,
    double alpha_div_n,
    double* __restrict__ diff_sum,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float base_val = (float)(one_minus_alpha_div_n + alpha_div_n * (*d_dangling_sum));

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    double my_diff = 0.0;

    if (v < num_vertices) {
        int start = new_offsets[v];
        int end = new_offsets[v + 1];
        float sum = 0.0f;
        for (int e = start; e < end; e++) {
            sum += pr_norm[new_indices[e]];
        }
        float new_val = base_val + alpha * sum;
        new_pr[v] = new_val;
        my_diff = (double)fabsf(new_val - old_pr[v]);
    }

    double block_diff = BlockReduce(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff != 0.0) {
        atomicAdd(diff_sum, block_diff);
    }
}

}  

PageRankResult pagerank_mask(const graph32_t& graph,
                             float* pageranks,
                             const float* precomputed_vertex_out_weight_sums,
                             float alpha,
                             float epsilon,
                             std::size_t max_iterations,
                             const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cudaStream_t stream = 0;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;

    
    cache.ensure_vertex_buffers(num_vertices);

    int32_t* d_active_count = cache.active_count;
    int32_t* d_out_degree = cache.out_degree;
    int32_t* d_new_offsets = cache.new_offsets;

    
    cudaMemsetAsync(d_out_degree, 0, num_vertices * sizeof(int32_t), stream);

    preprocess_kernel<<<grid, block, 0, stream>>>(d_offsets, d_indices, d_edge_mask,
                                                    d_active_count, d_out_degree, num_vertices);

    cudaMemsetAsync(d_new_offsets, 0, sizeof(int32_t), stream);

    
    {
        size_t temp_bytes = 0;
        cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d_active_count,
                                       d_new_offsets + 1, num_vertices, stream);
        void* d_temp;
        cudaMalloc(&d_temp, temp_bytes);
        cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_active_count,
                                       d_new_offsets + 1, num_vertices, stream);
        cudaFree(d_temp);
    }

    int32_t h_num_active_edges;
    cudaMemcpy(&h_num_active_edges, d_new_offsets + num_vertices,
               sizeof(int32_t), cudaMemcpyDeviceToHost);

    cache.ensure_edge_buffers(h_num_active_edges > 0 ? h_num_active_edges : 1);
    int32_t* d_new_indices = cache.new_indices;

    if (h_num_active_edges > 0) {
        compact_edges_kernel<<<grid, block, 0, stream>>>(d_offsets, d_indices, d_edge_mask,
                                                           d_new_offsets, d_new_indices, num_vertices);
    }

    
    float* d_pr_a = pageranks;  
    float* d_pr_b = cache.pr_buf;
    float* d_pr_norm = cache.pr_norm;
    double* d_dangling = cache.scalar_buf;
    double* d_diff = cache.scalar_buf + 1;

    if (initial_pageranks) {
        cudaMemcpyAsync(d_pr_a, initial_pageranks, num_vertices * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        init_pr_kernel<<<grid, block, 0, stream>>>(d_pr_a, num_vertices);
    }

    double one_minus_alpha_div_n = (1.0 - (double)alpha) / (double)num_vertices;
    double alpha_div_n = (double)alpha / (double)num_vertices;

    bool use_cusparse = (h_num_active_edges > 100000);

    
    cusparseSpMatDescr_t mat_descr = nullptr;
    cusparseDnVecDescr_t x_descr = nullptr, y_descr = nullptr;

    if (use_cusparse) {
        cache.ensure_cusparse_values(h_num_active_edges);
        int grid_e = (h_num_active_edges + block - 1) / block;
        fill_ones_kernel<<<grid_e, block, 0, stream>>>(cache.cusparse_values, h_num_active_edges);

        cusparseCreateCsr(&mat_descr, num_vertices, num_vertices, h_num_active_edges,
                          d_new_offsets, d_new_indices, cache.cusparse_values,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        cusparseCreateDnVec(&x_descr, num_vertices, d_pr_norm, CUDA_R_32F);
        cusparseCreateDnVec(&y_descr, num_vertices, d_pr_b, CUDA_R_32F);

        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
        cusparseSetStream(cache.cusparse_handle, stream);

        float h_alpha_val = alpha;
        float h_beta_val = 1.0f;
        cudaMemcpy(cache.d_cusparse_alpha, &h_alpha_val, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(cache.d_cusparse_beta, &h_beta_val, sizeof(float), cudaMemcpyHostToDevice);

        size_t buffer_size;
        cusparseSpMV_bufferSize(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                cache.d_cusparse_alpha, mat_descr, x_descr,
                                cache.d_cusparse_beta, y_descr,
                                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
        cache.ensure_spmv_buffer(buffer_size);

        cusparseSpMV_preprocess(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                cache.d_cusparse_alpha, mat_descr, x_descr,
                                cache.d_cusparse_beta, y_descr,
                                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buffer);
    }

    
    float* d_cur_pr = d_pr_a;
    float* d_new_pr = d_pr_b;
    std::size_t iterations = 0;
    bool converged = false;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        cudaMemsetAsync(d_dangling, 0, 2 * sizeof(double), stream);

        normalize_and_dangling_kernel<<<grid, block, 0, stream>>>(
            d_cur_pr, d_out_degree, d_pr_norm, d_dangling, num_vertices);

        if (use_cusparse) {
            fill_base_val_kernel<<<grid, block, 0, stream>>>(
                d_new_pr, d_dangling, one_minus_alpha_div_n, alpha_div_n, num_vertices);

            cusparseDnVecSetValues(x_descr, d_pr_norm);
            cusparseDnVecSetValues(y_descr, d_new_pr);

            cusparseSpMV(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                          cache.d_cusparse_alpha, mat_descr, x_descr,
                          cache.d_cusparse_beta, y_descr,
                          CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buffer);

            diff_kernel<<<grid, block, 0, stream>>>(
                d_cur_pr, d_new_pr, d_diff, num_vertices);
        } else {
            spmv_and_diff_kernel<<<grid, block, 0, stream>>>(
                d_new_offsets, d_new_indices, d_pr_norm,
                d_cur_pr, d_new_pr, d_dangling,
                alpha, one_minus_alpha_div_n, alpha_div_n,
                d_diff, num_vertices);
        }

        cudaMemcpyAsync(cache.h_pinned, d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations = iter + 1;
        float* tmp = d_cur_pr; d_cur_pr = d_new_pr; d_new_pr = tmp;

        if (*cache.h_pinned < (double)epsilon) {
            converged = true;
            break;
        }
    }

    
    if (use_cusparse) {
        if (x_descr) cusparseDestroyDnVec(x_descr);
        if (y_descr) cusparseDestroyDnVec(y_descr);
        if (mat_descr) cusparseDestroySpMat(mat_descr);
    }

    
    if (d_cur_pr != pageranks) {
        cudaMemcpyAsync(pageranks, d_cur_pr, num_vertices * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    return PageRankResult{iterations, converged};
}

}  
