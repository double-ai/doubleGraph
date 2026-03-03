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
#include <algorithm>

namespace aai {

namespace {




__device__ __forceinline__ double warp_reduce_sum_d(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ void block_reduce_atomic(double val, double* result) {
    __shared__ double sdata[8]; 
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    val = warp_reduce_sum_d(val);
    if (lane == 0) sdata[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x >> 5;
        val = (lane < num_warps) ? sdata[lane] : 0.0;
        val = warp_reduce_sum_d(val);
        if (lane == 0) atomicAdd(result, val);
    }
}





__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    double* __restrict__ out_weight_sums,
    int32_t num_edges)
{
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_edges;
         e += blockDim.x * gridDim.x) {
        atomicAdd(&out_weight_sums[indices[e]], edge_weights[e]);
    }
}

__global__ void compute_norm_weights_float_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const double* __restrict__ out_weight_sums,
    float* __restrict__ norm_weights,
    int32_t num_edges)
{
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_edges;
         e += blockDim.x * gridDim.x) {
        int32_t src = indices[e];
        double ow = out_weight_sums[src];
        norm_weights[e] = (ow > 0.0) ? (float)(edge_weights[e] / ow) : 0.0f;
    }
}

__global__ void build_dangling_list_kernel(
    const double* __restrict__ out_weight_sums,
    int32_t* __restrict__ dangling_list,
    int32_t* __restrict__ dangling_count,
    int32_t num_vertices)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices;
         v += blockDim.x * gridDim.x) {
        if (out_weight_sums[v] == 0.0) {
            int idx = atomicAdd(dangling_count, 1);
            dangling_list[idx] = v;
        }
    }
}

__global__ void init_pagerank_kernel(
    double* __restrict__ pr, float* __restrict__ pr_float,
    int32_t num_vertices, double init_val)
{
    float init_f = (float)init_val;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices;
         v += blockDim.x * gridDim.x) {
        pr[v] = init_val;
        pr_float[v] = init_f;
    }
}





__global__ void dangling_sum_kernel(
    const double* __restrict__ pr,
    const int32_t* __restrict__ dangling_list,
    int32_t num_dangling,
    double* __restrict__ result)
{
    double val = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_dangling;
         i += blockDim.x * gridDim.x) {
        val += pr[dangling_list[i]];
    }
    block_reduce_atomic(val, result);
}

__global__ void fused_damping_conv_cast_kernel(
    const float* __restrict__ spmv_float,
    double* __restrict__ new_pr,
    const double* __restrict__ old_pr,
    float* __restrict__ pr_float_out,
    const double* __restrict__ d_dangling_sum,
    double one_minus_alpha_over_n,
    double alpha,
    double alpha_over_n,
    double* __restrict__ d_diff,
    int32_t num_vertices)
{
    double base_score = one_minus_alpha_over_n + alpha_over_n * d_dangling_sum[0];

    double diff = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vertices;
         i += blockDim.x * gridDim.x) {
        double new_val = base_score + alpha * (double)spmv_float[i];
        diff += fabs(new_val - old_pr[i]);
        new_pr[i] = new_val;
        pr_float_out[i] = (float)new_val;
    }
    block_reduce_atomic(diff, d_diff);
}

__global__ void damping_only_kernel(
    const float* __restrict__ spmv_float,
    double* __restrict__ new_pr,
    float* __restrict__ pr_float_out,
    const double* __restrict__ d_dangling_sum,
    double one_minus_alpha_over_n,
    double alpha,
    double alpha_over_n,
    int32_t num_vertices)
{
    double base_score = one_minus_alpha_over_n + alpha_over_n * d_dangling_sum[0];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vertices;
         i += blockDim.x * gridDim.x) {
        double new_val = base_score + alpha * (double)spmv_float[i];
        new_pr[i] = new_val;
        pr_float_out[i] = (float)new_val;
    }
}

__global__ void init_from_guess_kernel(
    const double* __restrict__ initial, double* __restrict__ pr,
    float* __restrict__ pr_float, int32_t num_vertices)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices;
         v += blockDim.x * gridDim.x) {
        double val = initial[v];
        pr[v] = val;
        pr_float[v] = (float)val;
    }
}




struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;

    
    int32_t* dangling_count = nullptr;
    double* dsum = nullptr;
    double* diff = nullptr;

    
    double* out_weight_sums = nullptr;
    int32_t* dangling_list = nullptr;
    double* pr_a = nullptr;
    double* pr_b = nullptr;
    float* pr_float = nullptr;
    float* spmv_float = nullptr;
    int32_t vertex_capacity = 0;

    
    float* norm_weights = nullptr;
    int32_t edge_capacity = 0;

    
    void* cusparse_buf = nullptr;
    size_t cusparse_buf_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cudaMalloc(&dangling_count, sizeof(int32_t));
        cudaMalloc(&dsum, sizeof(double));
        cudaMalloc(&diff, sizeof(double));
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (dangling_count) cudaFree(dangling_count);
        if (dsum) cudaFree(dsum);
        if (diff) cudaFree(diff);
        if (out_weight_sums) cudaFree(out_weight_sums);
        if (dangling_list) cudaFree(dangling_list);
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (pr_float) cudaFree(pr_float);
        if (spmv_float) cudaFree(spmv_float);
        if (norm_weights) cudaFree(norm_weights);
        if (cusparse_buf) cudaFree(cusparse_buf);
    }

    void ensure_vertex_buffers(int32_t n) {
        if (vertex_capacity < n) {
            if (out_weight_sums) cudaFree(out_weight_sums);
            if (dangling_list) cudaFree(dangling_list);
            if (pr_a) cudaFree(pr_a);
            if (pr_b) cudaFree(pr_b);
            if (pr_float) cudaFree(pr_float);
            if (spmv_float) cudaFree(spmv_float);
            cudaMalloc(&out_weight_sums, (size_t)n * sizeof(double));
            cudaMalloc(&dangling_list, (size_t)n * sizeof(int32_t));
            cudaMalloc(&pr_a, (size_t)n * sizeof(double));
            cudaMalloc(&pr_b, (size_t)n * sizeof(double));
            cudaMalloc(&pr_float, (size_t)n * sizeof(float));
            cudaMalloc(&spmv_float, (size_t)n * sizeof(float));
            vertex_capacity = n;
        }
    }

    void ensure_edge_buffers(int32_t m) {
        if (edge_capacity < m) {
            if (norm_weights) cudaFree(norm_weights);
            cudaMalloc(&norm_weights, (size_t)m * sizeof(float));
            edge_capacity = m;
        }
    }

    void ensure_cusparse_buffer(size_t size) {
        size = std::max(size, (size_t)8);
        if (cusparse_buf_capacity < size) {
            if (cusparse_buf) cudaFree(cusparse_buf);
            cudaMalloc(&cusparse_buf, size);
            cusparse_buf_capacity = size;
        }
    }
};

}  

PageRankResult pagerank(const graph32_t& graph,
                        const double* edge_weights,
                        double* pageranks,
                        const double* precomputed_vertex_out_weight_sums,
                        double alpha,
                        double epsilon,
                        std::size_t max_iterations,
                        const double* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    cudaStream_t stream = 0;
    cusparseSetStream(cache.cusparse_handle, stream);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    cache.ensure_vertex_buffers(num_vertices);
    cache.ensure_edge_buffers(num_edges);

    
    cudaMemsetAsync(cache.out_weight_sums, 0,
        (size_t)num_vertices * sizeof(double), stream);
    {
        int block = 256;
        int grid = std::min((num_edges + block - 1) / block, 2048);
        compute_out_weight_sums_kernel<<<grid, block, 0, stream>>>(
            d_indices, edge_weights, cache.out_weight_sums, num_edges);
    }
    const double* out_ws = cache.out_weight_sums;

    {
        int block = 256;
        int grid = std::min((num_edges + block - 1) / block, 2048);
        compute_norm_weights_float_kernel<<<grid, block, 0, stream>>>(
            d_indices, edge_weights, out_ws, cache.norm_weights, num_edges);
    }

    cudaMemsetAsync(cache.dangling_count, 0, sizeof(int32_t), stream);
    {
        int block = 256;
        int grid = std::min((num_vertices + block - 1) / block, 2048);
        build_dangling_list_kernel<<<grid, block, 0, stream>>>(
            out_ws, cache.dangling_list, cache.dangling_count, num_vertices);
    }

    int32_t num_dangling = 0;
    cudaMemcpyAsync(&num_dangling, cache.dangling_count,
        sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    
    cusparseSpMatDescr_t mat_descr;
    cusparseCreateCsr(
        &mat_descr, num_vertices, num_vertices, num_edges,
        (void*)d_offsets, (void*)d_indices, (void*)cache.norm_weights,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    
    if (initial_pageranks) {
        int block = 256;
        int grid = std::min((num_vertices + block - 1) / block, 2048);
        init_from_guess_kernel<<<grid, block, 0, stream>>>(
            initial_pageranks, cache.pr_a, cache.pr_float, num_vertices);
    } else {
        double init_val = 1.0 / num_vertices;
        int block = 256;
        int grid = std::min((num_vertices + block - 1) / block, 2048);
        init_pagerank_kernel<<<grid, block, 0, stream>>>(
            cache.pr_a, cache.pr_float, num_vertices, init_val);
    }

    
    cusparseDnVecDescr_t vec_x, vec_y;
    cusparseCreateDnVec(&vec_x, num_vertices, cache.pr_float, CUDA_R_32F);
    cusparseCreateDnVec(&vec_y, num_vertices, cache.spmv_float, CUDA_R_32F);

    
    float h_one_f = 1.0f, h_zero_f = 0.0f;
    size_t buffer_size = 0;
    cusparseSpMV_bufferSize(
        cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_one_f, mat_descr, vec_x, &h_zero_f, vec_y,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);

    cache.ensure_cusparse_buffer(buffer_size);

    
    cusparseSpMV_preprocess(
        cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_one_f, mat_descr, vec_x, &h_zero_f, vec_y,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.cusparse_buf);

    
    double one_minus_alpha_over_n = (1.0 - alpha) / num_vertices;
    double alpha_over_n = alpha / num_vertices;

    int dangling_blocks = (num_dangling > 0) ?
        std::min((num_dangling + 255) / 256, 512) : 1;
    int vertex_blocks = std::min((num_vertices + 255) / 256, 1024);

    int check_interval = 1;

    double* pr_cur = cache.pr_a;
    double* pr_new = cache.pr_b;

    
    std::size_t iterations = 0;
    bool converged = false;
    double h_diff;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(cache.dsum, 0, sizeof(double), stream);
        if (num_dangling > 0) {
            dangling_sum_kernel<<<dangling_blocks, 256, 0, stream>>>(
                pr_cur, cache.dangling_list, num_dangling, cache.dsum);
        }

        
        cusparseSpMV(
            cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_one_f, mat_descr, vec_x, &h_zero_f, vec_y,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.cusparse_buf);

        
        bool do_check = ((iter + 1) % check_interval == 0) ||
                        (iter == max_iterations - 1);

        if (do_check) {
            
            cudaMemsetAsync(cache.diff, 0, sizeof(double), stream);
            fused_damping_conv_cast_kernel<<<vertex_blocks, 256, 0, stream>>>(
                cache.spmv_float, pr_new, pr_cur, cache.pr_float,
                cache.dsum, one_minus_alpha_over_n, alpha, alpha_over_n,
                cache.diff, num_vertices);

            cudaMemcpyAsync(&h_diff, cache.diff, sizeof(double),
                cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            iterations = iter + 1;

            if (h_diff < epsilon) {
                converged = true;
                pr_cur = pr_new;
                break;
            }
        } else {
            
            damping_only_kernel<<<vertex_blocks, 256, 0, stream>>>(
                cache.spmv_float, pr_new, cache.pr_float,
                cache.dsum, one_minus_alpha_over_n, alpha, alpha_over_n,
                num_vertices);
            iterations = iter + 1;
        }

        
        std::swap(pr_cur, pr_new);
    }

    
    cudaMemcpyAsync(pageranks, pr_cur,
        (size_t)num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    
    cusparseDestroySpMat(mat_descr);
    cusparseDestroyDnVec(vec_x);
    cusparseDestroyDnVec(vec_y);

    return {iterations, converged};
}

}  
