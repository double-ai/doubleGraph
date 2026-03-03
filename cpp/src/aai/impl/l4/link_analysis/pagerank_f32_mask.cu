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

namespace aai {

namespace {

#define BLOCK 256





__global__ void count_active_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_counts,
    int32_t num_vertices)
{
    int32_t v = blockIdx.x * BLOCK + threadIdx.x;
    if (v < num_vertices) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        int32_t count = 0;
        for (int32_t e = start; e < end; e++) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1) count++;
        }
        active_counts[v] = count;
    }
}

__global__ void compact_and_outw_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    float* __restrict__ new_weights,
    float* __restrict__ out_w,
    int32_t num_vertices)
{
    int32_t v = blockIdx.x * BLOCK + threadIdx.x;
    if (v < num_vertices) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        int32_t pos = new_offsets[v];
        for (int32_t e = start; e < end; e++) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1) {
                int32_t src = indices[e];
                float w = weights[e];
                new_indices[pos] = src;
                new_weights[pos] = w;
                atomicAdd(&out_w[src], w);
                pos++;
            }
        }
    }
}

__global__ void fill_kernel(float* __restrict__ arr, float val, int32_t n)
{
    int32_t i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) arr[i] = val;
}





__global__ void pr_norm_dangling_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ out_w,
    float* __restrict__ pr_norm,
    float* __restrict__ d_dangling_sum,
    int32_t n)
{
    int32_t i = blockIdx.x * BLOCK + threadIdx.x;
    float dang = 0.0f;
    if (i < n) {
        float w = out_w[i];
        if (w > 0.0f) {
            pr_norm[i] = pr[i] / w;
        } else {
            pr_norm[i] = 0.0f;
            dang = pr[i];
        }
    }
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage temp;
    float s = BR(temp).Sum(dang);
    if (threadIdx.x == 0 && s != 0.0f) atomicAdd(d_dangling_sum, s);
}






__global__ void fused_update_prnorm_kernel(
    const float* __restrict__ spmv_result,  
    const float* __restrict__ pr_old,
    const float* __restrict__ out_w,
    float* __restrict__ pr_new,
    float* __restrict__ pr_norm,            
    float* __restrict__ d_diff,
    const float* __restrict__ d_dangling_cur,
    float* __restrict__ d_dangling_next,
    float one_minus_alpha_over_n,
    float alpha_over_n,
    int32_t num_vertices)
{
    int32_t v = blockIdx.x * BLOCK + threadIdx.x;
    float base_val = one_minus_alpha_over_n + alpha_over_n * (*d_dangling_cur);

    float diff = 0.0f;
    float dang = 0.0f;

    if (v < num_vertices) {
        float new_val = base_val + spmv_result[v];
        pr_new[v] = new_val;
        diff = fabsf(new_val - pr_old[v]);

        float w = out_w[v];
        if (w > 0.0f) {
            pr_norm[v] = new_val / w;
        } else {
            pr_norm[v] = 0.0f;
            dang = new_val;
        }
    }

    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage temp;

    float block_diff = BR(temp).Sum(diff);
    if (threadIdx.x == 0) atomicAdd(d_diff, block_diff);

    __syncthreads();

    float block_dang = BR(temp).Sum(dang);
    if (threadIdx.x == 0 && block_dang != 0.0f) atomicAdd(d_dangling_next, block_dang);
}






__global__ void custom_spmv_fused_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ out_w,
    const float* __restrict__ pr_norm_in,   
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    float* __restrict__ pr_norm_out,        
    float* __restrict__ d_diff,
    const float* __restrict__ d_dangling_cur,
    float* __restrict__ d_dangling_next,
    float alpha,
    float one_minus_alpha_over_n,
    float alpha_over_n,
    int32_t num_vertices)
{
    int32_t v = blockIdx.x * BLOCK + threadIdx.x;
    float base_val = one_minus_alpha_over_n + alpha_over_n * (*d_dangling_cur);

    float diff = 0.0f;
    float dang = 0.0f;

    if (v < num_vertices) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float sum = 0.0f;
        for (int32_t e = start; e < end; e++) {
            sum += pr_norm_in[indices[e]] * weights[e];
        }

        float new_val = base_val + alpha * sum;
        pr_new[v] = new_val;
        diff = fabsf(new_val - pr_old[v]);

        float w = out_w[v];
        if (w > 0.0f) {
            pr_norm_out[v] = new_val / w;
        } else {
            pr_norm_out[v] = 0.0f;
            dang = new_val;
        }
    }

    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage temp;

    float block_diff = BR(temp).Sum(diff);
    if (threadIdx.x == 0) atomicAdd(d_diff, block_diff);

    __syncthreads();

    float block_dang = BR(temp).Sum(dang);
    if (threadIdx.x == 0 && block_dang != 0.0f) atomicAdd(d_dangling_next, block_dang);
}





struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;

    
    float* out_w = nullptr;
    float* pr_norm_a = nullptr;
    float* pr_norm_b = nullptr;
    float* pr_buf = nullptr;
    float* spmv_out = nullptr;
    
    int32_t* active_counts = nullptr;
    int32_t* new_offsets = nullptr;  
    
    float* scratch = nullptr;
    int32_t vertex_capacity = 0;

    
    int32_t* new_indices = nullptr;
    float* new_weights = nullptr;
    int32_t edge_capacity = 0;

    
    void* spmv_buf = nullptr;
    size_t spmv_buf_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cudaMalloc(&scratch, 5 * sizeof(float));
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (out_w) cudaFree(out_w);
        if (pr_norm_a) cudaFree(pr_norm_a);
        if (pr_norm_b) cudaFree(pr_norm_b);
        if (pr_buf) cudaFree(pr_buf);
        if (spmv_out) cudaFree(spmv_out);
        if (active_counts) cudaFree(active_counts);
        if (new_offsets) cudaFree(new_offsets);
        if (scratch) cudaFree(scratch);
        if (new_indices) cudaFree(new_indices);
        if (new_weights) cudaFree(new_weights);
        if (spmv_buf) cudaFree(spmv_buf);
    }

    void ensure_vertex(int32_t n) {
        if (vertex_capacity < n) {
            if (out_w) cudaFree(out_w);
            if (pr_norm_a) cudaFree(pr_norm_a);
            if (pr_norm_b) cudaFree(pr_norm_b);
            if (pr_buf) cudaFree(pr_buf);
            if (spmv_out) cudaFree(spmv_out);
            if (active_counts) cudaFree(active_counts);
            if (new_offsets) cudaFree(new_offsets);
            cudaMalloc(&out_w, n * sizeof(float));
            cudaMalloc(&pr_norm_a, n * sizeof(float));
            cudaMalloc(&pr_norm_b, n * sizeof(float));
            cudaMalloc(&pr_buf, n * sizeof(float));
            cudaMalloc(&spmv_out, n * sizeof(float));
            cudaMalloc(&active_counts, n * sizeof(int32_t));
            cudaMalloc(&new_offsets, ((int64_t)n + 1) * sizeof(int32_t));
            vertex_capacity = n;
        }
    }

    void ensure_edge(int32_t e) {
        if (edge_capacity < e) {
            if (new_indices) cudaFree(new_indices);
            if (new_weights) cudaFree(new_weights);
            cudaMalloc(&new_indices, e * sizeof(int32_t));
            cudaMalloc(&new_weights, e * sizeof(float));
            edge_capacity = e;
        }
    }

    void ensure_spmv(size_t size) {
        if (spmv_buf_capacity < size) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, size);
            spmv_buf_capacity = size;
        }
    }
};

}  

PageRankResult pagerank_mask(const graph32_t& graph,
                             const float* edge_weights,
                             float* pageranks,
                             const float* precomputed_vertex_out_weight_sums,
                             float alpha,
                             float epsilon,
                             std::size_t max_iterations,
                             const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    cache.ensure_vertex(num_vertices);
    cache.ensure_edge(num_edges);

    float* d_dangling_a = cache.scratch;
    float* d_dangling_b = cache.scratch + 1;
    float* d_diff = cache.scratch + 2;
    float* d_alpha_dev = cache.scratch + 3;
    float* d_zero_dev = cache.scratch + 4;

    float h_alpha = alpha, h_zero = 0.0f;
    cudaMemcpy(d_alpha_dev, &h_alpha, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_zero_dev, &h_zero, sizeof(float), cudaMemcpyHostToDevice);

    
    int grid_v = (num_vertices + BLOCK - 1) / BLOCK;
    count_active_kernel<<<grid_v, BLOCK>>>(graph.offsets, graph.edge_mask,
                                           cache.active_counts, num_vertices);

    cudaMemset(cache.new_offsets, 0, sizeof(int32_t));
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, cache.active_counts,
                                  cache.new_offsets + 1, num_vertices);
    void* d_temp;
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, cache.active_counts,
                                  cache.new_offsets + 1, num_vertices);
    cudaFree(d_temp);

    cudaMemset(cache.out_w, 0, num_vertices * sizeof(float));
    compact_and_outw_kernel<<<grid_v, BLOCK>>>(
        graph.offsets, graph.indices, edge_weights, graph.edge_mask,
        cache.new_offsets, cache.new_indices, cache.new_weights,
        cache.out_w, num_vertices);

    int32_t num_active_edges;
    cudaMemcpy(&num_active_edges, cache.new_offsets + num_vertices,
               sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    if (initial_pageranks != nullptr) {
        cudaMemcpy(pageranks, initial_pageranks,
                   num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        fill_kernel<<<grid_v, BLOCK>>>(pageranks, 1.0f / num_vertices, num_vertices);
    }

    float one_minus_alpha_over_n = (1.0f - alpha) / num_vertices;
    float alpha_over_n = alpha / num_vertices;

    
    float avg_degree = (float)num_active_edges / (float)num_vertices;
    bool use_cusparse = (num_active_edges > 500000 && avg_degree > 4.0f);

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;

    if (use_cusparse) {
        cusparseCreateCsr(&matA, num_vertices, num_vertices, num_active_edges,
            cache.new_offsets, cache.new_indices, cache.new_weights,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cusparseCreateDnVec(&vecX, num_vertices, cache.pr_norm_a, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, num_vertices, cache.spmv_out, CUDA_R_32F);
        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);

        size_t bufSize = 0;
        cusparseSpMV_bufferSize(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_alpha_dev, matA, vecX, d_zero_dev, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);
        if (bufSize > 0) {
            cache.ensure_spmv(bufSize);
        }
        cusparseSpMV_preprocess(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_alpha_dev, matA, vecX, d_zero_dev, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
            bufSize > 0 ? cache.spmv_buf : nullptr);
    }

    
    float* d_dangling_cur = d_dangling_a;
    float* d_dangling_next = d_dangling_b;
    float* pr_norm_cur = cache.pr_norm_a;
    float* pr_norm_next = cache.pr_norm_b;

    cudaMemset(d_dangling_cur, 0, sizeof(float));
    pr_norm_dangling_kernel<<<grid_v, BLOCK>>>(
        pageranks, cache.out_w, pr_norm_cur, d_dangling_cur, num_vertices);

    
    float* pr_old = pageranks;
    float* pr_new = cache.pr_buf;
    bool converged = false;
    size_t iterations = 0;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(d_diff, 0, sizeof(float));
        cudaMemsetAsync(d_dangling_next, 0, sizeof(float));

        if (use_cusparse) {
            cusparseDnVecSetValues(vecX, pr_norm_cur);
            cusparseDnVecSetValues(vecY, cache.spmv_out);
            cusparseSpMV(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                d_alpha_dev, matA, vecX, d_zero_dev, vecY,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                cache.spmv_buf);

            fused_update_prnorm_kernel<<<grid_v, BLOCK>>>(
                cache.spmv_out, pr_old, cache.out_w,
                pr_new, pr_norm_cur,  
                d_diff, d_dangling_cur, d_dangling_next,
                one_minus_alpha_over_n, alpha_over_n, num_vertices);
        } else {
            custom_spmv_fused_kernel<<<grid_v, BLOCK>>>(
                cache.new_offsets, cache.new_indices, cache.new_weights,
                cache.out_w,
                pr_norm_cur, pr_old,
                pr_new, pr_norm_next,
                d_diff, d_dangling_cur, d_dangling_next,
                alpha, one_minus_alpha_over_n, alpha_over_n, num_vertices);
            
            float* tmp_pn = pr_norm_cur;
            pr_norm_cur = pr_norm_next;
            pr_norm_next = tmp_pn;
        }

        iterations++;

        
        float h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_diff < epsilon) {
            converged = true;
            float* tmp = pr_old; pr_old = pr_new; pr_new = tmp;
            break;
        }

        
        float* tmp = pr_old; pr_old = pr_new; pr_new = tmp;
        
        float* tmp_d = d_dangling_cur;
        d_dangling_cur = d_dangling_next;
        d_dangling_next = tmp_d;
    }

    if (matA) cusparseDestroySpMat(matA);
    if (vecX) cusparseDestroyDnVec(vecX);
    if (vecY) cusparseDestroyDnVec(vecY);

    if (pr_old != pageranks) {
        cudaMemcpy(pageranks, pr_old,
                   num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return PageRankResult{iterations, converged};
}

}  
