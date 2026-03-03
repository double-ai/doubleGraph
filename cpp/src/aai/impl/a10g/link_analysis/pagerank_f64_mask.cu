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
#include <cstddef>
#include <climits>
#include <cmath>

namespace aai {

namespace {

constexpr int BLOCK_SIZE = 256;



__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ counts,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v <= num_vertices) {
        if (v < num_vertices) {
            int32_t start = offsets[v];
            int32_t end = offsets[v + 1];
            int32_t count = 0;
            for (int32_t e = start; e < end; e++) {
                count += (edge_mask[e >> 5] >> (e & 31)) & 1u;
            }
            counts[v] = count;
        } else {
            counts[v] = 0;
        }
    }
}

__global__ void scatter_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    double* __restrict__ new_weights,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        int32_t write_pos = new_offsets[v];
        for (int32_t e = start; e < end; e++) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1u) {
                new_indices[write_pos] = indices[e];
                new_weights[write_pos] = edge_weights[e];
                write_pos++;
            }
        }
    }
}

__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    double* __restrict__ out_weight_sums,
    int32_t num_edges)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        atomicAdd(&out_weight_sums[indices[tid]], weights[tid]);
    }
}

__global__ void compute_coefficients_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ out_weight_sums,
    double* __restrict__ coefficients,
    int32_t num_edges)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        double ow = out_weight_sums[indices[tid]];
        coefficients[tid] = (ow > 0.0) ? (weights[tid] / ow) : 0.0;
    }
}

__global__ void compute_dangling_kernel(
    const double* __restrict__ out_weight_sums,
    uint8_t* __restrict__ is_dangling,
    int32_t num_vertices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        is_dangling[tid] = (out_weight_sums[tid] == 0.0) ? 1 : 0;
    }
}



__global__ void dangling_sum_kernel(
    const double* __restrict__ pr,
    const uint8_t* __restrict__ is_dangling,
    double* __restrict__ d_dangling_sum,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double val = 0.0;
    if (tid < num_vertices && is_dangling[tid]) {
        val = pr[tid];
    }

    double block_sum = BlockReduce(temp_storage).Sum(val);
    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(d_dangling_sum, block_sum);
    }
}

__global__ void spmv_coeff_update_diff_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ coefficients,
    const double* __restrict__ pr,
    double* __restrict__ pr_new,
    const double* __restrict__ d_dangling_sum,
    double* __restrict__ d_diff_sum,
    double alpha,
    double one_minus_alpha_over_n,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ double s_dangling_contrib;

    if (threadIdx.x == 0) {
        s_dangling_contrib = alpha * (*d_dangling_sum) / (double)num_vertices;
    }
    __syncthreads();

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    double my_diff = 0.0;

    if (v < num_vertices) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        double sum = 0.0;
        for (int32_t e = start; e < end; e++) {
            sum += pr[indices[e]] * coefficients[e];
        }
        double new_val = one_minus_alpha_over_n + alpha * sum + s_dangling_contrib;
        pr_new[v] = new_val;
        my_diff = fabs(new_val - pr[v]);
    }

    double block_diff = BlockReduce(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff != 0.0) {
        atomicAdd(d_diff_sum, block_diff);
    }
}

__global__ void update_diff_kernel(
    const double* __restrict__ spmv_result,
    const double* __restrict__ pr_old,
    double* __restrict__ pr_new,
    const double* __restrict__ d_dangling_sum,
    double* __restrict__ d_diff_sum,
    double alpha,
    double one_minus_alpha_over_n,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ double s_dangling_contrib;

    if (threadIdx.x == 0) {
        s_dangling_contrib = alpha * (*d_dangling_sum) / (double)num_vertices;
    }
    __syncthreads();

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    double my_diff = 0.0;

    if (v < num_vertices) {
        double new_val = one_minus_alpha_over_n + alpha * spmv_result[v] + s_dangling_contrib;
        pr_new[v] = new_val;
        my_diff = fabs(new_val - pr_old[v]);
    }

    double block_diff = BlockReduce(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff != 0.0) {
        atomicAdd(d_diff_sum, block_diff);
    }
}

__global__ void init_uniform_kernel(double* pr, int32_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) pr[tid] = 1.0 / (double)n;
}



struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;

    
    int32_t* counts = nullptr;        
    int32_t* new_offsets = nullptr;    
    double* out_ws = nullptr;         
    uint8_t* is_dangling = nullptr;   
    double* pr_tmp = nullptr;         
    double* spmv_result = nullptr;    
    double* iter_scalar = nullptr;    
    double* scalar_alpha_beta = nullptr; 
    int64_t vertex_capacity = 0;

    
    int32_t* new_indices = nullptr;
    double* new_weights = nullptr;
    double* coefficients = nullptr;
    int64_t edge_capacity = 0;

    
    void* ps_temp = nullptr;
    size_t ps_temp_capacity = 0;

    
    void* spmv_buf = nullptr;
    size_t spmv_buf_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (counts) cudaFree(counts);
        if (new_offsets) cudaFree(new_offsets);
        if (out_ws) cudaFree(out_ws);
        if (is_dangling) cudaFree(is_dangling);
        if (pr_tmp) cudaFree(pr_tmp);
        if (spmv_result) cudaFree(spmv_result);
        if (iter_scalar) cudaFree(iter_scalar);
        if (scalar_alpha_beta) cudaFree(scalar_alpha_beta);
        if (new_indices) cudaFree(new_indices);
        if (new_weights) cudaFree(new_weights);
        if (coefficients) cudaFree(coefficients);
        if (ps_temp) cudaFree(ps_temp);
        if (spmv_buf) cudaFree(spmv_buf);
    }

    void ensure_vertex(int64_t nv) {
        if (vertex_capacity < nv) {
            if (counts) cudaFree(counts);
            if (new_offsets) cudaFree(new_offsets);
            if (out_ws) cudaFree(out_ws);
            if (is_dangling) cudaFree(is_dangling);
            if (pr_tmp) cudaFree(pr_tmp);
            if (spmv_result) cudaFree(spmv_result);
            if (iter_scalar) cudaFree(iter_scalar);
            if (scalar_alpha_beta) cudaFree(scalar_alpha_beta);

            cudaMalloc(&counts, (nv + 1) * sizeof(int32_t));
            cudaMalloc(&new_offsets, (nv + 1) * sizeof(int32_t));
            cudaMalloc(&out_ws, nv * sizeof(double));
            cudaMalloc(&is_dangling, nv * sizeof(uint8_t));
            cudaMalloc(&pr_tmp, nv * sizeof(double));
            cudaMalloc(&spmv_result, nv * sizeof(double));
            cudaMalloc(&iter_scalar, 2 * sizeof(double));
            cudaMalloc(&scalar_alpha_beta, 2 * sizeof(double));

            
            double h_vals[2] = {1.0, 0.0};
            cudaMemcpy(scalar_alpha_beta, h_vals, 2 * sizeof(double), cudaMemcpyHostToDevice);

            vertex_capacity = nv;
        }
    }

    void ensure_edges(int64_t ne) {
        int64_t alloc_ne = (ne > 0) ? ne : 1;
        if (edge_capacity < alloc_ne) {
            if (new_indices) cudaFree(new_indices);
            if (new_weights) cudaFree(new_weights);
            if (coefficients) cudaFree(coefficients);

            cudaMalloc(&new_indices, alloc_ne * sizeof(int32_t));
            cudaMalloc(&new_weights, alloc_ne * sizeof(double));
            cudaMalloc(&coefficients, alloc_ne * sizeof(double));

            edge_capacity = alloc_ne;
        }
    }

    void ensure_ps_temp(size_t bytes) {
        if (ps_temp_capacity < bytes) {
            if (ps_temp) cudaFree(ps_temp);
            cudaMalloc(&ps_temp, bytes);
            ps_temp_capacity = bytes;
        }
    }

    void ensure_spmv_buf(size_t bytes) {
        if (spmv_buf_capacity < bytes) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, bytes + 1);
            spmv_buf_capacity = bytes + 1;
        }
    }
};

}  

PageRankResult pagerank_mask(const graph32_t& graph,
                             const double* edge_weights,
                             double* pageranks,
                             const double* precomputed_vertex_out_weight_sums,
                             double alpha,
                             double epsilon,
                             std::size_t max_iterations,
                             const double* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cudaStream_t stream = 0;
    cusparseSetStream(cache.cusparse_handle, stream);

    
    cache.ensure_vertex(num_vertices);

    
    int32_t n_plus_1 = num_vertices + 1;

    count_active_edges_kernel<<<(n_plus_1 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        d_offsets, d_edge_mask, cache.counts, num_vertices);

    size_t ps_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, ps_temp_bytes,
        (int32_t*)nullptr, (int32_t*)nullptr, n_plus_1);
    cache.ensure_ps_temp(ps_temp_bytes);

    cub::DeviceScan::ExclusiveSum(cache.ps_temp, ps_temp_bytes,
        cache.counts, cache.new_offsets, n_plus_1, stream);

    int32_t total_active;
    cudaMemcpy(&total_active, cache.new_offsets + num_vertices,
        sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    cache.ensure_edges(total_active);

    if (total_active > 0) {
        scatter_active_edges_kernel<<<(num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            d_offsets, d_indices, edge_weights, d_edge_mask,
            cache.new_offsets, cache.new_indices, cache.new_weights, num_vertices);
    }

    
    cudaMemsetAsync(cache.out_ws, 0, num_vertices * sizeof(double), stream);
    if (total_active > 0) {
        compute_out_weight_sums_kernel<<<(total_active + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            cache.new_indices, cache.new_weights, cache.out_ws, total_active);
    }

    
    if (total_active > 0) {
        compute_coefficients_kernel<<<(total_active + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            cache.new_indices, cache.new_weights, cache.out_ws, cache.coefficients, total_active);
    }

    
    compute_dangling_kernel<<<(num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
        cache.out_ws, cache.is_dangling, num_vertices);

    
    bool use_cusparse = (total_active > 100000) && (total_active > 4 * (int64_t)num_vertices);

    cusparseSpMatDescr_t mat_descr = nullptr;
    cusparseDnVecDescr_t x_descr = nullptr, y_descr = nullptr;

    if (use_cusparse) {
        cusparseCreateCsr(&mat_descr, num_vertices, num_vertices, total_active,
            cache.new_offsets, cache.new_indices, cache.coefficients,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        cusparseCreateDnVec(&x_descr, num_vertices, cache.pr_tmp, CUDA_R_64F);
        cusparseCreateDnVec(&y_descr, num_vertices, cache.spmv_result, CUDA_R_64F);

        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);

        size_t spmv_buffer_size;
        cusparseSpMV_bufferSize(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.scalar_alpha_beta, mat_descr, x_descr,
            cache.scalar_alpha_beta + 1, y_descr,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buffer_size);

        cache.ensure_spmv_buf(spmv_buffer_size);

        cusparseSpMV_preprocess(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.scalar_alpha_beta, mat_descr, x_descr,
            cache.scalar_alpha_beta + 1, y_descr,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);
    }

    
    
    double* d_pr_a = pageranks;
    double* d_pr_b = cache.pr_tmp;
    double* d_dangling_sum = cache.iter_scalar;
    double* d_diff_sum = cache.iter_scalar + 1;

    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(d_pr_a, initial_pageranks,
            num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else {
        init_uniform_kernel<<<(num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            d_pr_a, num_vertices);
    }

    double one_minus_alpha_over_n = (1.0 - alpha) / num_vertices;
    double* pr_cur = d_pr_a;
    double* pr_next = d_pr_b;
    bool converged = false;
    std::size_t iterations = 0;

    
    if (!use_cusparse) {
        static bool configured = false;
        if (!configured) {
            cudaFuncSetAttribute(spmv_coeff_update_diff_kernel,
                cudaFuncAttributePreferredSharedMemoryCarveout, 0);
            configured = true;
        }
    }

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        cudaMemsetAsync(d_dangling_sum, 0, 2 * sizeof(double), stream);

        dangling_sum_kernel<<<(num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            pr_cur, cache.is_dangling, d_dangling_sum, num_vertices);

        if (use_cusparse) {
            cusparseDnVecSetValues(x_descr, pr_cur);
            cusparseDnVecSetValues(y_descr, cache.spmv_result);

            cusparseSpMV(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                cache.scalar_alpha_beta, mat_descr, x_descr,
                cache.scalar_alpha_beta + 1, y_descr,
                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

            update_diff_kernel<<<(num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                cache.spmv_result, pr_cur, pr_next, d_dangling_sum, d_diff_sum,
                alpha, one_minus_alpha_over_n, num_vertices);
        } else {
            spmv_coeff_update_diff_kernel<<<(num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                cache.new_offsets, cache.new_indices,
                cache.coefficients, pr_cur, pr_next, d_dangling_sum, d_diff_sum,
                alpha, one_minus_alpha_over_n, num_vertices);
        }

        double h_diff;
        cudaMemcpy(&h_diff, d_diff_sum, sizeof(double), cudaMemcpyDeviceToHost);

        iterations++;
        double* temp = pr_cur; pr_cur = pr_next; pr_next = temp;

        if (h_diff < epsilon) { converged = true; break; }
    }

    
    if (mat_descr) cusparseDestroySpMat(mat_descr);
    if (x_descr) cusparseDestroyDnVec(x_descr);
    if (y_descr) cusparseDestroyDnVec(y_descr);

    
    if (pr_cur != pageranks) {
        cudaMemcpyAsync(pageranks, pr_cur,
            num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    return PageRankResult{iterations, converged};
}

}  
