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
#include <cstring>
#include <algorithm>
#include <vector>

namespace aai {

namespace {

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t status = (call); \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        cudaDeviceSynchronize(); \
    } \
} while(0)




__device__ __forceinline__ bool is_edge_active(const uint32_t* __restrict__ mask, int idx) {
    return (mask[idx >> 5] >> (idx & 31)) & 1u;
}





__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_count,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int start = offsets[v], end = offsets[v + 1];
        int count = 0;
        for (int i = start; i < end; i++) {
            if (is_edge_active(edge_mask, i)) count++;
        }
        active_count[v] = count;
    }
}

__global__ void compute_out_weights_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ out_weight,
    int32_t num_edges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges && is_edge_active(edge_mask, idx)) {
        atomicAdd(&out_weight[indices[idx]], weights[idx]);
    }
}

__global__ void compute_inv_out_weight_kernel(
    double* __restrict__ data,
    int32_t num_vertices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        double ow = data[idx];
        data[idx] = (ow > 0.0) ? 1.0 / ow : 0.0;
    }
}

__global__ void scatter_compressed_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ new_offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ inv_out_weight,
    int32_t* __restrict__ new_indices,
    double* __restrict__ scaled_weights,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int old_start = old_offsets[v], old_end = old_offsets[v + 1];
        int new_pos = new_offsets[v];
        for (int i = old_start; i < old_end; i++) {
            if (is_edge_active(edge_mask, i)) {
                int src = indices[i];
                new_indices[new_pos] = src;
                scaled_weights[new_pos] = weights[i] * inv_out_weight[src];
                new_pos++;
            }
        }
    }
}

__global__ void build_pers_norm_kernel(
    const int32_t* __restrict__ pers_vertices,
    const double* __restrict__ pers_values,
    double* __restrict__ pers_norm,
    int32_t pers_size, double pers_sum_inv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pers_size) {
        pers_norm[pers_vertices[idx]] = pers_values[idx] * pers_sum_inv;
    }
}

__global__ void init_pr_kernel(double* __restrict__ pr, double val, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) pr[idx] = val;
}

__global__ void build_dangling_list_kernel(
    const double* __restrict__ inv_out_weight,
    int32_t* __restrict__ dangling_vertices,
    int32_t* __restrict__ dangling_count,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices && inv_out_weight[v] == 0.0) {
        int pos = atomicAdd(dangling_count, 1);
        dangling_vertices[pos] = v;
    }
}





#define BLOCK_D 256
__global__ void dangling_sum_kernel(
    const double* __restrict__ pr,
    const int32_t* __restrict__ dangling_vertices,
    double* __restrict__ d_dangling_sum,
    int32_t num_dangling)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double dv = 0.0;
    if (idx < num_dangling) {
        dv = pr[dangling_vertices[idx]];
    }
    typedef cub::BlockReduce<double, BLOCK_D> BR;
    __shared__ typename BR::TempStorage ts;
    double bs = BR(ts).Sum(dv);
    if (threadIdx.x == 0 && bs != 0.0)
        atomicAdd(d_dangling_sum, bs);
}

#define BLOCK_C 256
__global__ void combine_diff_kernel(
    double* __restrict__ new_pr,
    const double* __restrict__ old_pr,
    const double* __restrict__ pers_norm,
    double alpha,
    const double* __restrict__ d_dangling_sum,
    double one_minus_alpha,
    double* __restrict__ d_diff,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    double dv = 0.0;
    if (v < num_vertices) {
        double base = alpha * (*d_dangling_sum) + one_minus_alpha;
        double nv = new_pr[v] + base * pers_norm[v];
        new_pr[v] = nv;
        dv = fabs(nv - old_pr[v]);
    }
    typedef cub::BlockReduce<double, BLOCK_C> BR;
    __shared__ typename BR::TempStorage ts;
    double bs = BR(ts).Sum(dv);
    if (threadIdx.x == 0 && bs > 0.0)
        atomicAdd(d_diff, bs);
}






__global__ void spmv_block_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ scaled_weights,
    const double* __restrict__ pr,
    double* __restrict__ spmv_out,
    double alpha,
    int32_t seg_start, int32_t seg_end)
{
    int v = blockIdx.x + seg_start;
    if (v >= seg_end) return;

    int start = offsets[v], end = offsets[v + 1];
    double sum = 0.0;
    for (int i = start + (int)threadIdx.x; i < end; i += (int)blockDim.x) {
        sum += scaled_weights[i] * pr[indices[i]];
    }

    typedef cub::BlockReduce<double, 256> BR;
    __shared__ typename BR::TempStorage ts;
    sum = BR(ts).Sum(sum);

    if (threadIdx.x == 0) {
        spmv_out[v] = alpha * sum;
    }
}


#define BLOCK_W 256
#define WARPS_W (BLOCK_W / 32)
__global__ void spmv_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ scaled_weights,
    const double* __restrict__ pr,
    double* __restrict__ spmv_out,
    double alpha,
    int32_t seg_start, int32_t seg_end)
{
    int gwarp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int v = gwarp + seg_start;
    if (v >= seg_end) return;

    int start = offsets[v], end = offsets[v + 1];
    double sum = 0.0;
    for (int i = start + lane; i < end; i += 32) {
        sum += scaled_weights[i] * pr[indices[i]];
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, o);

    if (lane == 0) {
        spmv_out[v] = alpha * sum;
    }
}


__global__ void spmv_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ scaled_weights,
    const double* __restrict__ pr,
    double* __restrict__ spmv_out,
    double alpha,
    int32_t seg_start, int32_t seg_end)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x + seg_start;
    if (v >= seg_end) return;

    int start = offsets[v], end = offsets[v + 1];
    double sum = 0.0;
    for (int i = start; i < end; i++) {
        sum += scaled_weights[i] * pr[indices[i]];
    }

    spmv_out[v] = alpha * sum;
}





struct Cache : Cacheable {
    
    int32_t* active_count = nullptr;
    int32_t active_count_cap = 0;

    int32_t* new_offsets = nullptr;
    int32_t new_offsets_cap = 0;

    double* inv_out_weight = nullptr;
    int32_t inv_out_weight_cap = 0;

    double* pers_norm = nullptr;
    int32_t pers_norm_cap = 0;

    int32_t* dangling_list = nullptr;
    int32_t dangling_list_cap = 0;

    double* pr_a = nullptr;
    int32_t pr_a_cap = 0;

    double* pr_b = nullptr;
    int32_t pr_b_cap = 0;

    
    int32_t* dangling_count = nullptr;
    double* scalars = nullptr;  

    
    int32_t* comp_indices = nullptr;
    int32_t comp_indices_cap = 0;

    double* scaled_weights = nullptr;
    int32_t scaled_weights_cap = 0;

    
    void* cub_temp = nullptr;
    size_t cub_temp_cap = 0;

    
    void* spmv_buf = nullptr;
    size_t spmv_buf_cap = 0;

    ~Cache() override {
        if (active_count) cudaFree(active_count);
        if (new_offsets) cudaFree(new_offsets);
        if (inv_out_weight) cudaFree(inv_out_weight);
        if (pers_norm) cudaFree(pers_norm);
        if (dangling_list) cudaFree(dangling_list);
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (dangling_count) cudaFree(dangling_count);
        if (scalars) cudaFree(scalars);
        if (comp_indices) cudaFree(comp_indices);
        if (scaled_weights) cudaFree(scaled_weights);
        if (cub_temp) cudaFree(cub_temp);
        if (spmv_buf) cudaFree(spmv_buf);
    }
};

}  

PageRankResult personalized_pagerank_seg_mask(const graph32_t& graph,
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
    const uint32_t* d_edge_mask = graph.edge_mask;
    const double* d_weights = edge_weights;
    const int32_t* d_pers_vertices = personalization_vertices;
    const double* d_pers_values = personalization_values;
    int32_t pers_size = static_cast<int32_t>(personalization_size);

    

    if (cache.active_count_cap < num_vertices + 1) {
        if (cache.active_count) cudaFree(cache.active_count);
        cudaMalloc(&cache.active_count, (num_vertices + 1) * sizeof(int32_t));
        cache.active_count_cap = num_vertices + 1;
    }
    if (cache.new_offsets_cap < num_vertices + 1) {
        if (cache.new_offsets) cudaFree(cache.new_offsets);
        cudaMalloc(&cache.new_offsets, (num_vertices + 1) * sizeof(int32_t));
        cache.new_offsets_cap = num_vertices + 1;
    }
    if (cache.inv_out_weight_cap < num_vertices) {
        if (cache.inv_out_weight) cudaFree(cache.inv_out_weight);
        cudaMalloc(&cache.inv_out_weight, num_vertices * sizeof(double));
        cache.inv_out_weight_cap = num_vertices;
    }
    if (cache.pers_norm_cap < num_vertices) {
        if (cache.pers_norm) cudaFree(cache.pers_norm);
        cudaMalloc(&cache.pers_norm, num_vertices * sizeof(double));
        cache.pers_norm_cap = num_vertices;
    }
    if (cache.dangling_list_cap < num_vertices) {
        if (cache.dangling_list) cudaFree(cache.dangling_list);
        cudaMalloc(&cache.dangling_list, num_vertices * sizeof(int32_t));
        cache.dangling_list_cap = num_vertices;
    }
    if (cache.pr_a_cap < num_vertices) {
        if (cache.pr_a) cudaFree(cache.pr_a);
        cudaMalloc(&cache.pr_a, num_vertices * sizeof(double));
        cache.pr_a_cap = num_vertices;
    }
    if (cache.pr_b_cap < num_vertices) {
        if (cache.pr_b) cudaFree(cache.pr_b);
        cudaMalloc(&cache.pr_b, num_vertices * sizeof(double));
        cache.pr_b_cap = num_vertices;
    }

    
    if (!cache.dangling_count) {
        cudaMalloc(&cache.dangling_count, sizeof(int32_t));
    }
    if (!cache.scalars) {
        cudaMalloc(&cache.scalars, 2 * sizeof(double));
    }

    
    size_t cub_temp_size = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_temp_size,
                                  (int32_t*)nullptr, (int32_t*)nullptr,
                                  num_vertices + 1);
    if (cache.cub_temp_cap < cub_temp_size) {
        if (cache.cub_temp) cudaFree(cache.cub_temp);
        cudaMalloc(&cache.cub_temp, cub_temp_size);
        cache.cub_temp_cap = cub_temp_size;
    }

    int32_t* d_active_count = cache.active_count;
    int32_t* d_new_offsets = cache.new_offsets;
    double* d_inv_out_weight = cache.inv_out_weight;
    double* d_pers_norm = cache.pers_norm;
    int32_t* d_dangling_list = cache.dangling_list;
    double* d_pr_a = cache.pr_a;
    double* d_pr_b = cache.pr_b;
    int32_t* d_dangling_count = cache.dangling_count;
    double* d_dangling_sum = cache.scalars;
    double* d_diff = cache.scalars + 1;

    

    
    cudaMemset(d_active_count, 0, (num_vertices + 1) * sizeof(int32_t));
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 0)
            count_active_edges_kernel<<<grid, block>>>(d_offsets, d_edge_mask, d_active_count, num_vertices);
    }

    
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, cub_temp_size,
                                  d_active_count, d_new_offsets,
                                  num_vertices + 1);

    
    int32_t new_num_edges;
    cudaMemcpy(&new_num_edges, &d_new_offsets[num_vertices], sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    cudaMemset(d_inv_out_weight, 0, num_vertices * sizeof(double));
    {
        int block = 256;
        int grid = (num_edges + block - 1) / block;
        if (grid > 0)
            compute_out_weights_kernel<<<grid, block>>>(d_indices, d_weights, d_edge_mask, d_inv_out_weight, num_edges);
    }

    
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 0)
            compute_inv_out_weight_kernel<<<grid, block>>>(d_inv_out_weight, num_vertices);
    }

    
    if (cache.comp_indices_cap < new_num_edges) {
        if (cache.comp_indices) cudaFree(cache.comp_indices);
        cudaMalloc(&cache.comp_indices, (int64_t)new_num_edges * sizeof(int32_t));
        cache.comp_indices_cap = new_num_edges;
    }
    if (cache.scaled_weights_cap < new_num_edges) {
        if (cache.scaled_weights) cudaFree(cache.scaled_weights);
        cudaMalloc(&cache.scaled_weights, (int64_t)new_num_edges * sizeof(double));
        cache.scaled_weights_cap = new_num_edges;
    }

    int32_t* d_new_indices = cache.comp_indices;
    double* d_scaled_weights = cache.scaled_weights;

    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 0)
            scatter_compressed_kernel<<<grid, block>>>(
                d_offsets, d_new_offsets, d_indices, d_weights,
                d_edge_mask, d_inv_out_weight,
                d_new_indices, d_scaled_weights, num_vertices);
    }

    
    cudaMemset(d_pers_norm, 0, num_vertices * sizeof(double));

    double h_pers_sum = 0.0;
    if (pers_size > 0) {
        std::vector<double> h_vals(pers_size);
        cudaMemcpy(h_vals.data(), d_pers_values, pers_size * sizeof(double), cudaMemcpyDeviceToHost);
        for (int32_t i = 0; i < pers_size; i++) h_pers_sum += h_vals[i];
    }
    double pers_sum_inv = (h_pers_sum > 0.0) ? 1.0 / h_pers_sum : 0.0;
    {
        int block = 256;
        int grid = (pers_size + block - 1) / block;
        if (grid > 0)
            build_pers_norm_kernel<<<grid, block>>>(d_pers_vertices, d_pers_values, d_pers_norm, pers_size, pers_sum_inv);
    }

    
    cudaMemset(d_dangling_count, 0, sizeof(int32_t));
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 0)
            build_dangling_list_kernel<<<grid, block>>>(d_inv_out_weight, d_dangling_list, d_dangling_count, num_vertices);
    }

    int32_t num_dangling;
    cudaMemcpy(&num_dangling, d_dangling_count, sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    cusparseHandle_t cusparse_handle;
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle));
    CUSPARSE_CHECK(cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_HOST));

    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &matA, num_vertices, num_vertices, new_num_edges,
        d_new_offsets, d_new_indices, d_scaled_weights,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseDnVecDescr_t vecX, vecY;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, num_vertices, d_pr_a, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, num_vertices, d_pr_b, CUDA_R_64F));

    double h_alpha_val = alpha, h_zero = 0.0;
    size_t spmv_buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha_val, matA, vecX, &h_zero, vecY,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buffer_size));

    if (cache.spmv_buf_cap < spmv_buffer_size + 16) {
        if (cache.spmv_buf) cudaFree(cache.spmv_buf);
        cudaMalloc(&cache.spmv_buf, spmv_buffer_size + 16);
        cache.spmv_buf_cap = spmv_buffer_size + 16;
    }
    void* d_spmv_buffer = cache.spmv_buf;

    CUSPARSE_CHECK(cusparseSpMV_preprocess(
        cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha_val, matA, vecX, &h_zero, vecY,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer));

    
    double* pr_old = d_pr_a;
    double* pr_new = d_pr_b;
    if (initial_pageranks != nullptr) {
        cudaMemcpy(pr_old, initial_pageranks,
                   num_vertices * sizeof(double), cudaMemcpyDeviceToDevice);
    } else {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 0)
            init_pr_kernel<<<grid, block>>>(pr_old, 1.0 / num_vertices, num_vertices);
    }

    
    double one_minus_alpha = 1.0 - alpha;
    std::size_t iterations = 0;
    bool converged = false;
    double h_diff;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemset(d_dangling_sum, 0, 2 * sizeof(double));

        
        if (num_dangling > 0) {
            int block = BLOCK_D;
            int grid = (num_dangling + block - 1) / block;
            dangling_sum_kernel<<<grid, block>>>(pr_old, d_dangling_list, d_dangling_sum, num_dangling);
        }

        
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecX, pr_old));
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecY, pr_new));
        CUSPARSE_CHECK(cusparseSpMV(
            cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_alpha_val, matA, vecX, &h_zero, vecY,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buffer));

        
        {
            int block = BLOCK_C;
            int grid = (num_vertices + block - 1) / block;
            if (grid > 0)
                combine_diff_kernel<<<grid, block>>>(pr_new, pr_old, d_pers_norm,
                                                     alpha, d_dangling_sum, one_minus_alpha, d_diff, num_vertices);
        }

        
        cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);

        iterations = iter + 1;
        if (h_diff < epsilon) {
            converged = true;
            break;
        }

        std::swap(pr_old, pr_new);
    }

    
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(matA);
    cusparseDestroy(cusparse_handle);

    
    double* result = converged ? pr_new : pr_old;
    cudaMemcpy(pageranks, result, num_vertices * sizeof(double), cudaMemcpyDeviceToDevice);

    return PageRankResult{iterations, converged};
}

}  
