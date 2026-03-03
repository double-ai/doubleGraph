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

constexpr int BLOCK_SIZE = 256;





__global__ void compute_out_weights_kernel(
    const int32_t* __restrict__ csc_indices,
    const float* __restrict__ edge_weights,
    float* __restrict__ out_weights,
    int32_t num_edges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges) {
        atomicAdd(&out_weights[csc_indices[idx]], edge_weights[idx]);
    }
}

__global__ void compute_modified_weights_kernel(
    const int32_t* __restrict__ csc_indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ out_weights,
    float* __restrict__ modified_weights,
    int32_t num_edges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges) {
        modified_weights[idx] = edge_weights[idx] / out_weights[csc_indices[idx]];
    }
}

__global__ void compute_dangling_mask_kernel(
    const float* __restrict__ out_weights,
    uint8_t* __restrict__ dangling_mask,
    int32_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dangling_mask[idx] = (out_weights[idx] == 0.0f) ? 1 : 0;
    }
}

__global__ void init_pr_uniform_kernel(float* __restrict__ pr, int32_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        pr[idx] = 1.0f / (float)n;
    }
}

__global__ void dangling_sum_kernel(
    const float* __restrict__ pr,
    const uint8_t* __restrict__ dangling_mask,
    float* __restrict__ dangling_sum,
    int32_t n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n && dangling_mask[idx]) ? pr[idx] : 0.0f;

    float block_sum = BlockReduce(temp).Sum(val);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(dangling_sum, block_sum);
    }
}

__global__ void add_pers_kernel(
    float* __restrict__ new_pr,
    const int32_t* __restrict__ pers_vertices,
    const float* __restrict__ pers_norm,
    const float* __restrict__ dangling_sum_ptr,
    float alpha,
    float one_minus_alpha,
    int32_t pers_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pers_size) {
        float base_factor = alpha * (*dangling_sum_ptr) + one_minus_alpha;
        new_pr[pers_vertices[idx]] += base_factor * pers_norm[idx];
    }
}

__global__ void compute_delta_kernel(
    const float* __restrict__ new_pr,
    const float* __restrict__ old_pr,
    float* __restrict__ delta,
    int32_t n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? fabsf(new_pr[idx] - old_pr[idx]) : 0.0f;

    float block_sum = BlockReduce(temp).Sum(val);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(delta, block_sum);
    }
}





struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;

    float* pr_a = nullptr;
    int64_t pr_a_cap = 0;

    float* pr_b = nullptr;
    int64_t pr_b_cap = 0;

    float* out_w = nullptr;
    int64_t out_w_cap = 0;

    float* mod_w = nullptr;
    int64_t mod_w_cap = 0;

    uint8_t* dmask = nullptr;
    int64_t dmask_cap = 0;

    float* pnorm = nullptr;
    int64_t pnorm_cap = 0;

    float* scalars = nullptr;  

    void* spmv_buf = nullptr;
    int64_t spmv_buf_cap = 0;

    Cache() {
        cusparseCreate(&handle);
        cudaMalloc(&scalars, 4 * sizeof(float));
    }

    ~Cache() override {
        if (handle) cusparseDestroy(handle);
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (out_w) cudaFree(out_w);
        if (mod_w) cudaFree(mod_w);
        if (dmask) cudaFree(dmask);
        if (pnorm) cudaFree(pnorm);
        if (scalars) cudaFree(scalars);
        if (spmv_buf) cudaFree(spmv_buf);
    }

    void ensure(int64_t n, int64_t e, int64_t ps) {
        if (pr_a_cap < n) {
            if (pr_a) cudaFree(pr_a);
            cudaMalloc(&pr_a, n * sizeof(float));
            pr_a_cap = n;
        }
        if (pr_b_cap < n) {
            if (pr_b) cudaFree(pr_b);
            cudaMalloc(&pr_b, n * sizeof(float));
            pr_b_cap = n;
        }
        if (out_w_cap < n) {
            if (out_w) cudaFree(out_w);
            cudaMalloc(&out_w, n * sizeof(float));
            out_w_cap = n;
        }
        if (dmask_cap < n) {
            if (dmask) cudaFree(dmask);
            cudaMalloc(&dmask, n * sizeof(uint8_t));
            dmask_cap = n;
        }
        if (mod_w_cap < e) {
            if (mod_w) cudaFree(mod_w);
            cudaMalloc(&mod_w, e * sizeof(float));
            mod_w_cap = e;
        }
        if (pnorm_cap < ps) {
            if (pnorm) cudaFree(pnorm);
            cudaMalloc(&pnorm, ps * sizeof(float));
            pnorm_cap = ps;
        }
    }

    void ensure_spmv_buf(int64_t size) {
        if (spmv_buf_cap < size) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, size);
            spmv_buf_cap = size;
        }
    }
};

}  

PageRankResult personalized_pagerank_seg(const graph32_t& graph,
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

    int32_t N = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t pers_size = static_cast<int32_t>(personalization_size);

    cache.ensure(N, E, pers_size);

    float* d_pr_a = cache.pr_a;
    float* d_pr_b = cache.pr_b;
    float* d_out_w = cache.out_w;
    float* d_mod_w = cache.mod_w;
    uint8_t* d_dmask = cache.dmask;
    float* d_pnorm = cache.pnorm;
    float* d_dsum = cache.scalars;          
    float* d_delta = cache.scalars + 1;     
    float* d_alpha_spmv = cache.scalars + 2;
    float* d_beta_spmv = cache.scalars + 3;

    

    
    cudaMemsetAsync(d_out_w, 0, N * sizeof(float));
    if (E > 0) {
        int grid = (E + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_out_weights_kernel<<<grid, BLOCK_SIZE>>>(d_indices, edge_weights, d_out_w, E);
    }

    
    if (E > 0) {
        int grid = (E + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_modified_weights_kernel<<<grid, BLOCK_SIZE>>>(d_indices, edge_weights, d_out_w, d_mod_w, E);
    }

    
    {
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_dangling_mask_kernel<<<grid, BLOCK_SIZE>>>(d_out_w, d_dmask, N);
    }

    
    std::vector<float> h_pnorm(pers_size);
    cudaMemcpy(h_pnorm.data(), personalization_values, pers_size * sizeof(float), cudaMemcpyDeviceToHost);
    double psum = 0.0;
    for (auto v : h_pnorm) psum += v;
    for (auto& v : h_pnorm) v = (float)(v / psum);
    cudaMemcpy(d_pnorm, h_pnorm.data(), pers_size * sizeof(float), cudaMemcpyHostToDevice);

    
    float* d_curr = d_pr_a;
    float* d_new = d_pr_b;
    if (initial_pageranks != nullptr) {
        cudaMemcpy(d_curr, initial_pageranks, N * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_pr_uniform_kernel<<<grid, BLOCK_SIZE>>>(d_curr, N);
    }

    
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, N, N, E,
        (void*)d_offsets, (void*)d_indices, (void*)d_mod_w,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, N, d_curr, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, N, d_new, CUDA_R_32F);

    
    cusparseSetPointerMode(cache.handle, CUSPARSE_POINTER_MODE_HOST);
    float h_alpha_spmv = alpha;
    float h_beta_spmv = 0.0f;
    size_t bufSize = 0;
    cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha_spmv, matA, vecX, &h_beta_spmv, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);

    cache.ensure_spmv_buf(bufSize > 0 ? bufSize : 1);

    
    cusparseSpMV_preprocess(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha_spmv, matA, vecX, &h_beta_spmv, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

    
    cusparseSetPointerMode(cache.handle, CUSPARSE_POINTER_MODE_DEVICE);
    cudaMemcpy(d_alpha_spmv, &h_alpha_spmv, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta_spmv, &h_beta_spmv, sizeof(float), cudaMemcpyHostToDevice);

    
    const size_t CHECK_INTERVAL = 10;
    bool converged = false;
    size_t iterations = 0;
    float one_minus_alpha = 1.0f - alpha;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(d_dsum, 0, sizeof(float));
        {
            int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            dangling_sum_kernel<<<grid, BLOCK_SIZE>>>(d_curr, d_dmask, d_dsum, N);
        }

        
        cusparseDnVecSetValues(vecX, d_curr);
        cusparseDnVecSetValues(vecY, d_new);
        cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_alpha_spmv, matA, vecX, d_beta_spmv, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

        
        if (pers_size > 0) {
            int grid = (pers_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            add_pers_kernel<<<grid, BLOCK_SIZE>>>(d_new, personalization_vertices, d_pnorm, d_dsum,
                                                   alpha, one_minus_alpha, pers_size);
        }

        
        cudaMemsetAsync(d_delta, 0, sizeof(float));
        {
            int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            compute_delta_kernel<<<grid, BLOCK_SIZE>>>(d_new, d_curr, d_delta, N);
        }

        
        std::swap(d_curr, d_new);
        iterations = iter + 1;

        
        if (iterations % CHECK_INTERVAL == 0 || iterations == max_iterations) {
            float h_delta;
            cudaMemcpy(&h_delta, d_delta, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_delta < epsilon) {
                converged = true;
                break;
            }
        }
    }

    
    cudaMemcpy(pageranks, d_curr, N * sizeof(float), cudaMemcpyDeviceToDevice);

    
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    return PageRankResult{iterations, converged};
}

}  
