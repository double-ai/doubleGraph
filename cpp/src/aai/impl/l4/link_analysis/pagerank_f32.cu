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

namespace aai {

namespace {

#define BLOCK_SIZE 256

struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    float* pr2 = nullptr;
    float* out_weight_sums_buf = nullptr;
    float* modified_weights = nullptr;
    float* dangling_mask = nullptr;
    float* scratch = nullptr;
    float* spmv_result = nullptr;
    void* spmv_buf = nullptr;

    int64_t pr2_cap = 0;
    int64_t ows_cap = 0;
    int64_t mw_cap = 0;
    int64_t dm_cap = 0;
    bool scratch_alloc = false;
    int64_t spmv_cap = 0;
    size_t spmv_buf_cap = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (pr2) cudaFree(pr2);
        if (out_weight_sums_buf) cudaFree(out_weight_sums_buf);
        if (modified_weights) cudaFree(modified_weights);
        if (dangling_mask) cudaFree(dangling_mask);
        if (scratch) cudaFree(scratch);
        if (spmv_result) cudaFree(spmv_result);
        if (spmv_buf) cudaFree(spmv_buf);
    }

    void ensure(int32_t nv, int32_t ne, bool need_spmv) {
        if (pr2_cap < nv) {
            if (pr2) cudaFree(pr2);
            cudaMalloc(&pr2, (int64_t)nv * sizeof(float));
            pr2_cap = nv;
        }
        if (ows_cap < nv) {
            if (out_weight_sums_buf) cudaFree(out_weight_sums_buf);
            cudaMalloc(&out_weight_sums_buf, (int64_t)nv * sizeof(float));
            ows_cap = nv;
        }
        if (mw_cap < ne) {
            if (modified_weights) cudaFree(modified_weights);
            cudaMalloc(&modified_weights, (int64_t)ne * sizeof(float));
            mw_cap = ne;
        }
        if (dm_cap < nv) {
            if (dangling_mask) cudaFree(dangling_mask);
            cudaMalloc(&dangling_mask, (int64_t)nv * sizeof(float));
            dm_cap = nv;
        }
        if (!scratch_alloc) {
            cudaMalloc(&scratch, 3 * sizeof(float));
            scratch_alloc = true;
        }
        if (need_spmv && spmv_cap < nv) {
            if (spmv_result) cudaFree(spmv_result);
            cudaMalloc(&spmv_result, (int64_t)nv * sizeof(float));
            spmv_cap = nv;
        }
    }

    void ensure_spmv_buf(size_t size) {
        if (spmv_buf_cap < size) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, size);
            spmv_buf_cap = size;
        }
    }
};





__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    float* __restrict__ out_weight_sums,
    int32_t num_edges
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        atomicAdd(&out_weight_sums[indices[tid]], edge_weights[tid]);
    }
}

__global__ void compute_modified_weights_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ out_weight_sums,
    float* __restrict__ modified_weights,
    int32_t num_edges
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        int32_t src = indices[tid];
        float out_w = out_weight_sums[src];
        modified_weights[tid] = edge_weights[tid] / out_w;
    }
}

__global__ void init_pr_dangling_and_sum_kernel(
    const float* __restrict__ out_weight_sums,
    float* __restrict__ pr,
    float* __restrict__ dangling_mask,
    float* __restrict__ dangling_sum_out,
    int32_t num_vertices,
    float init_val
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    float is_dangling = 0.0f;

    if (tid < num_vertices) {
        pr[tid] = init_val;
        is_dangling = (out_weight_sums[tid] == 0.0f) ? 1.0f : 0.0f;
        dangling_mask[tid] = is_dangling;
    }

    float thread_dang = (tid < num_vertices) ? init_val * is_dangling : 0.0f;
    float block_dang = BlockReduce(temp_storage).Sum(thread_dang);
    if (threadIdx.x == 0 && block_dang != 0.0f) {
        atomicAdd(dangling_sum_out, block_dang);
    }
}

__global__ void fused_pagerank_iter_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ col_indices,
    const float* __restrict__ modified_weights,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    float* __restrict__ diff_out,
    const float* __restrict__ dangling_mask,
    float* __restrict__ dangling_sum_next,
    const float* __restrict__ dangling_sum_cur,
    float alpha,
    float inv_N,
    float base_val,
    int32_t num_vertices
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float dangling_term = alpha * __ldg(dangling_sum_cur) * inv_N;

    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    float t_diff = 0.0f;
    float t_dang = 0.0f;

    if (tid < num_vertices) {
        int row_start = __ldg(&offsets[tid]);
        int row_end = __ldg(&offsets[tid + 1]);

        float spmv_sum = 0.0f;
        for (int e = row_start; e < row_end; ++e) {
            spmv_sum += __ldg(&modified_weights[e]) * __ldg(&pr_old[__ldg(&col_indices[e])]);
        }

        float old_val = pr_old[tid];
        float new_val = base_val + alpha * spmv_sum + dangling_term;
        pr_new[tid] = new_val;
        t_diff = fabsf(new_val - old_val);
        t_dang = new_val * __ldg(&dangling_mask[tid]);
    }

    float b_diff = BlockReduce(temp_storage).Sum(t_diff);
    if (threadIdx.x == 0) atomicAdd(diff_out, b_diff);

    __syncthreads();

    float b_dang = BlockReduce(temp_storage).Sum(t_dang);
    if (threadIdx.x == 0 && b_dang != 0.0f) atomicAdd(dangling_sum_next, b_dang);
}

__global__ void update_diff_dangling_kernel(
    const float* __restrict__ spmv_result,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    float* __restrict__ diff_out,
    const float* __restrict__ dangling_mask,
    float* __restrict__ dangling_sum_next,
    const float* __restrict__ dangling_sum_cur,
    float alpha, float inv_N, float base_val,
    int32_t num_vertices
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float dangling_term = alpha * __ldg(dangling_sum_cur) * inv_N;
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    float t_diff = 0.0f;
    float t_dang = 0.0f;

    if (tid < num_vertices) {
        float new_val = base_val + alpha * spmv_result[tid] + dangling_term;
        pr_new[tid] = new_val;
        t_diff = fabsf(new_val - pr_old[tid]);
        t_dang = new_val * dangling_mask[tid];
    }

    float b_diff = BlockReduce(temp_storage).Sum(t_diff);
    if (threadIdx.x == 0) atomicAdd(diff_out, b_diff);

    __syncthreads();

    float b_dang = BlockReduce(temp_storage).Sum(t_dang);
    if (threadIdx.x == 0 && b_dang != 0.0f) atomicAdd(dangling_sum_next, b_dang);
}

}  

PageRankResult pagerank(const graph32_t& graph,
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

    float avg_degree = (num_vertices > 0) ? (float)num_edges / num_vertices : 0;
    bool use_fused_spmv = (avg_degree <= 6.0f);

    cache.ensure(num_vertices, num_edges, !use_fused_spmv);

    float* pr1 = pageranks;
    float* pr2 = cache.pr2;
    float* d_modified_weights = cache.modified_weights;
    float* d_dangling_mask = cache.dangling_mask;
    float* d_scratch = cache.scratch;

    

    float* ows = cache.out_weight_sums_buf;
    cudaMemset(ows, 0, (int64_t)num_vertices * sizeof(float));
    {
        int grid = ((int64_t)num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_out_weight_sums_kernel<<<grid, BLOCK_SIZE>>>(
            graph.indices, edge_weights, ows, num_edges);
    }
    const float* d_out_weight_sums = ows;

    cudaMemset(d_scratch, 0, 3 * sizeof(float));

    {
        int grid = ((int64_t)num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_modified_weights_kernel<<<grid, BLOCK_SIZE>>>(
            graph.indices, edge_weights, d_out_weight_sums, d_modified_weights, num_edges);
    }

    float init_val = 1.0f / num_vertices;
    {
        int grid = ((int64_t)num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_pr_dangling_and_sum_kernel<<<grid, BLOCK_SIZE>>>(
            d_out_weight_sums, pr1, d_dangling_mask, d_scratch + 1, num_vertices, init_val);
    }

    if (initial_pageranks) {
        cudaMemcpy(pr1, initial_pageranks, (int64_t)num_vertices * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }

    
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    void* dBuffer = nullptr;
    float h_alpha = 1.0f, h_beta = 0.0f;

    if (!use_fused_spmv) {
        cusparseCreateCsr(
            &matA, num_vertices, num_vertices, num_edges,
            (void*)graph.offsets,
            (void*)graph.indices,
            (void*)d_modified_weights,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F
        );

        cusparseCreateDnVec(&vecX, num_vertices, pr1, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, num_vertices, cache.spmv_result, CUDA_R_32F);

        size_t bufferSize = 0;
        cusparseSpMV_bufferSize(
            cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_alpha, matA, vecX, &h_beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize
        );

        if (bufferSize > 0) {
            cache.ensure_spmv_buf(bufferSize);
            dBuffer = cache.spmv_buf;
        }

        cusparseSpMV_preprocess(
            cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_alpha, matA, vecX, &h_beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer
        );
    }

    
    float base_val = (1.0f - alpha) / num_vertices;
    float inv_N = 1.0f / num_vertices;
    float* pr_cur = pr1;
    float* pr_next = pr2;
    float* d_diff = d_scratch;
    float* d_dang_a = d_scratch + 1;
    float* d_dang_b = d_scratch + 2;

    const int CHECK_INTERVAL = 3;
    size_t iterations = 0;
    bool converged = false;
    float h_diff;

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        iterations = iter + 1;

        float* d_dang_cur = (iter % 2 == 0) ? d_dang_a : d_dang_b;
        float* d_dang_next = (iter % 2 == 0) ? d_dang_b : d_dang_a;

        cudaMemsetAsync(d_diff, 0, sizeof(float));
        cudaMemsetAsync(d_dang_next, 0, sizeof(float));

        if (use_fused_spmv) {
            int grid = ((int64_t)num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
            fused_pagerank_iter_kernel<<<grid, BLOCK_SIZE>>>(
                graph.offsets, graph.indices, d_modified_weights,
                pr_cur, pr_next, d_diff,
                d_dangling_mask, d_dang_next, d_dang_cur,
                alpha, inv_N, base_val, num_vertices
            );
        } else {
            cusparseDnVecSetValues(vecX, pr_cur);
            cusparseSpMV(
                cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &h_alpha, matA, vecX, &h_beta, vecY,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dBuffer
            );
            int grid = ((int64_t)num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
            update_diff_dangling_kernel<<<grid, BLOCK_SIZE>>>(
                cache.spmv_result, pr_cur, pr_next, d_diff,
                d_dangling_mask, d_dang_next, d_dang_cur,
                alpha, inv_N, base_val, num_vertices
            );
        }

        float* temp = pr_cur; pr_cur = pr_next; pr_next = temp;

        if (iterations % CHECK_INTERVAL == 0 || iterations == max_iterations) {
            cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_diff < epsilon) {
                converged = true;
                break;
            }
        }
    }

    
    if (matA) cusparseDestroySpMat(matA);
    if (vecX) cusparseDestroyDnVec(vecX);
    if (vecY) cusparseDestroyDnVec(vecY);

    
    if (pr_cur != pr1) {
        cudaMemcpy(pr1, pr_cur, (int64_t)num_vertices * sizeof(float),
                   cudaMemcpyDeviceToDevice);
    }

    return PageRankResult{iterations, converged};
}

}  
