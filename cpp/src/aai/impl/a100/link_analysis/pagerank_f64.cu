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
#include <algorithm>

namespace aai {

namespace {





struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;

    
    float* pr_f32_a = nullptr;
    int64_t pr_f32_a_cap = 0;

    double* out_weight_sums = nullptr;
    int64_t out_weight_sums_cap = 0;

    uint8_t* dangling_mask = nullptr;
    int64_t dangling_mask_cap = 0;

    double* pr_f64_b = nullptr;
    int64_t pr_f64_b_cap = 0;

    float* pr_f32_b = nullptr;
    int64_t pr_f32_b_cap = 0;

    float* spmv_f32 = nullptr;
    int64_t spmv_f32_cap = 0;

    
    float* norm_weights_f32 = nullptr;
    int64_t norm_weights_f32_cap = 0;

    
    double* dangling_sum_a = nullptr;
    double* dangling_sum_b = nullptr;
    double* diff = nullptr;
    float* d_one = nullptr;
    float* d_zero = nullptr;
    bool fixed_alloc = false;

    
    void* spmv_buffer = nullptr;
    size_t spmv_buf_cap = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (pr_f32_a) cudaFree(pr_f32_a);
        if (out_weight_sums) cudaFree(out_weight_sums);
        if (dangling_mask) cudaFree(dangling_mask);
        if (pr_f64_b) cudaFree(pr_f64_b);
        if (pr_f32_b) cudaFree(pr_f32_b);
        if (spmv_f32) cudaFree(spmv_f32);
        if (norm_weights_f32) cudaFree(norm_weights_f32);
        if (dangling_sum_a) cudaFree(dangling_sum_a);
        if (dangling_sum_b) cudaFree(dangling_sum_b);
        if (diff) cudaFree(diff);
        if (d_one) cudaFree(d_one);
        if (d_zero) cudaFree(d_zero);
        if (spmv_buffer) cudaFree(spmv_buffer);
    }

    void ensure_pr_f32_a(int64_t n) {
        if (pr_f32_a_cap < n) {
            if (pr_f32_a) cudaFree(pr_f32_a);
            cudaMalloc(&pr_f32_a, n * sizeof(float));
            pr_f32_a_cap = n;
        }
    }

    void ensure_out_weight_sums(int64_t n) {
        if (out_weight_sums_cap < n) {
            if (out_weight_sums) cudaFree(out_weight_sums);
            cudaMalloc(&out_weight_sums, n * sizeof(double));
            out_weight_sums_cap = n;
        }
    }

    void ensure_dangling_mask(int64_t n) {
        if (dangling_mask_cap < n) {
            if (dangling_mask) cudaFree(dangling_mask);
            cudaMalloc(&dangling_mask, n * sizeof(uint8_t));
            dangling_mask_cap = n;
        }
    }

    void ensure_pr_f64_b(int64_t n) {
        if (pr_f64_b_cap < n) {
            if (pr_f64_b) cudaFree(pr_f64_b);
            cudaMalloc(&pr_f64_b, n * sizeof(double));
            pr_f64_b_cap = n;
        }
    }

    void ensure_pr_f32_b(int64_t n) {
        if (pr_f32_b_cap < n) {
            if (pr_f32_b) cudaFree(pr_f32_b);
            cudaMalloc(&pr_f32_b, n * sizeof(float));
            pr_f32_b_cap = n;
        }
    }

    void ensure_spmv_f32(int64_t n) {
        if (spmv_f32_cap < n) {
            if (spmv_f32) cudaFree(spmv_f32);
            cudaMalloc(&spmv_f32, n * sizeof(float));
            spmv_f32_cap = n;
        }
    }

    void ensure_norm_weights_f32(int64_t m) {
        if (norm_weights_f32_cap < m) {
            if (norm_weights_f32) cudaFree(norm_weights_f32);
            cudaMalloc(&norm_weights_f32, m * sizeof(float));
            norm_weights_f32_cap = m;
        }
    }

    void ensure_fixed() {
        if (!fixed_alloc) {
            cudaMalloc(&dangling_sum_a, sizeof(double));
            cudaMalloc(&dangling_sum_b, sizeof(double));
            cudaMalloc(&diff, sizeof(double));
            cudaMalloc(&d_one, sizeof(float));
            cudaMalloc(&d_zero, sizeof(float));
            float h_one = 1.0f, h_zero = 0.0f;
            cudaMemcpy(d_one, &h_one, sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_zero, &h_zero, sizeof(float), cudaMemcpyHostToDevice);
            fixed_alloc = true;
        }
    }

    void ensure_spmv_buffer(size_t sz) {
        if (spmv_buf_cap < sz) {
            if (spmv_buffer) cudaFree(spmv_buffer);
            cudaMalloc(&spmv_buffer, sz);
            spmv_buf_cap = sz;
        }
    }
};





__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    double* __restrict__ out_weight_sums,
    int32_t num_edges
) {
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         idx < num_edges;
         idx += (int64_t)blockDim.x * gridDim.x) {
        atomicAdd(&out_weight_sums[indices[idx]], edge_weights[idx]);
    }
}

__global__ void compute_normalized_weights_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const double* __restrict__ out_weight_sums,
    float* __restrict__ norm_weights_f32,
    int32_t num_edges
) {
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         idx < num_edges;
         idx += (int64_t)blockDim.x * gridDim.x) {
        double w = edge_weights[idx] / out_weight_sums[indices[idx]];
        norm_weights_f32[idx] = (float)w;
    }
}

__global__ void compute_dangling_mask_kernel(
    const double* __restrict__ out_weight_sums,
    uint8_t* __restrict__ dangling_mask,
    int32_t num_vertices
) {
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         idx < num_vertices;
         idx += (int64_t)blockDim.x * gridDim.x) {
        dangling_mask[idx] = (out_weight_sums[idx] == 0.0) ? 1 : 0;
    }
}

__global__ void init_pageranks_kernel(
    double* __restrict__ pr_f64,
    float* __restrict__ pr_f32,
    int32_t num_vertices
) {
    double val64 = 1.0 / (double)num_vertices;
    float val32 = (float)val64;
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         idx < num_vertices;
         idx += (int64_t)blockDim.x * gridDim.x) {
        pr_f64[idx] = val64;
        pr_f32[idx] = val32;
    }
}





__global__ void compute_initial_dangling_sum_kernel(
    const double* __restrict__ pageranks,
    const uint8_t* __restrict__ dangling_mask,
    double* __restrict__ dangling_sum,
    int32_t num_vertices
) {
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double thread_sum = 0.0;
    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < num_vertices;
         i += (int64_t)blockDim.x * gridDim.x) {
        if (dangling_mask[i]) {
            thread_sum += pageranks[i];
        }
    }

    double block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(dangling_sum, block_sum);
    }
}





__global__ void fused_update_mixed_kernel(
    const float* __restrict__ spmv_result_f32,
    double* __restrict__ pr_f64,
    float* __restrict__ pr_f32,
    const double* __restrict__ d_dangling_sum_cur,
    double* __restrict__ d_dangling_sum_next,
    double* __restrict__ d_diff,
    const uint8_t* __restrict__ dangling_mask,
    double alpha,
    double one_over_n,
    int32_t num_vertices
) {
    typedef cub::BlockReduce<double, 256, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_diff;
    __shared__ typename BlockReduce::TempStorage temp_dangling;

    double dangling_contribution = alpha * (*d_dangling_sum_cur) * one_over_n;
    double base = (1.0 - alpha) * one_over_n + dangling_contribution;

    double thread_diff = 0.0;
    double thread_dangling = 0.0;

    for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         i < num_vertices;
         i += (int64_t)blockDim.x * gridDim.x) {
        double spmv_val = (double)spmv_result_f32[i];
        double new_pr = base + alpha * spmv_val;
        double old_pr = pr_f64[i];
        double d = new_pr - old_pr;
        thread_diff += (d >= 0.0 ? d : -d);
        pr_f64[i] = new_pr;
        pr_f32[i] = (float)new_pr;

        if (dangling_mask[i]) {
            thread_dangling += new_pr;
        }
    }

    double block_diff = BlockReduce(temp_diff).Sum(thread_diff);
    if (threadIdx.x == 0 && block_diff != 0.0) atomicAdd(d_diff, block_diff);

    double block_dangling = BlockReduce(temp_dangling).Sum(thread_dangling);
    if (threadIdx.x == 0 && block_dangling != 0.0) atomicAdd(d_dangling_sum_next, block_dangling);
}






__global__ void pagerank_iteration_sparse_mixed(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ norm_weights_f32,
    const float* __restrict__ pr_old_f32,
    const double* __restrict__ pr_old_f64,
    double* __restrict__ pr_new_f64,
    float* __restrict__ pr_new_f32,
    const double* __restrict__ d_dangling_sum_cur,
    double* __restrict__ d_dangling_sum_next,
    double* __restrict__ d_diff,
    const uint8_t* __restrict__ dangling_mask,
    double alpha,
    double one_over_n,
    int32_t num_vertices
) {
    typedef cub::BlockReduce<double, 256, cub::BLOCK_REDUCE_WARP_REDUCTIONS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_diff;
    __shared__ typename BlockReduce::TempStorage temp_dangling;

    double base = (1.0 - alpha) * one_over_n + alpha * (*d_dangling_sum_cur) * one_over_n;

    double thread_diff = 0.0;
    double thread_dangling = 0.0;

    for (int64_t v = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         v < num_vertices;
         v += (int64_t)blockDim.x * gridDim.x) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float sum_f32 = 0.0f;
        for (int32_t e = start; e < end; e++) {
            sum_f32 += pr_old_f32[indices[e]] * norm_weights_f32[e];
        }

        double new_pr = base + alpha * (double)sum_f32;
        double old_pr = pr_old_f64[v];
        double d = new_pr - old_pr;
        thread_diff += (d >= 0.0 ? d : -d);
        pr_new_f64[v] = new_pr;
        pr_new_f32[v] = (float)new_pr;

        if (dangling_mask[v]) {
            thread_dangling += new_pr;
        }
    }

    double block_diff = BlockReduce(temp_diff).Sum(thread_diff);
    if (threadIdx.x == 0 && block_diff != 0.0) atomicAdd(d_diff, block_diff);

    double block_dangling = BlockReduce(temp_dangling).Sum(thread_dangling);
    if (threadIdx.x == 0 && block_dangling != 0.0) atomicAdd(d_dangling_sum_next, block_dangling);
}





__global__ void copy_f64_to_f32_kernel(
    const double* __restrict__ src_f64,
    float* __restrict__ dst_f32,
    int32_t n
) {
    for (int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
         idx < n;
         idx += (int64_t)blockDim.x * gridDim.x) {
        dst_f32[idx] = (float)src_f64[idx];
    }
}





void launch_setup(
    const int32_t* indices, const double* edge_weights,
    double* out_weight_sums, float* norm_weights_f32,
    uint8_t* dangling_mask, double* pr_f64, float* pr_f32,
    int32_t num_vertices, int32_t num_edges
) {
    int block = 256;
    cudaMemset(out_weight_sums, 0, (int64_t)num_vertices * sizeof(double));

    if (num_edges > 0) {
        int grid_e = (int)min((int64_t)65535, ((int64_t)num_edges + block - 1) / block);
        compute_out_weight_sums_kernel<<<grid_e, block>>>(indices, edge_weights, out_weight_sums, num_edges);
        compute_normalized_weights_kernel<<<grid_e, block>>>(
            indices, edge_weights, out_weight_sums, norm_weights_f32, num_edges);
    }

    int grid_v = (int)min((int64_t)65535, ((int64_t)num_vertices + block - 1) / block);
    compute_dangling_mask_kernel<<<grid_v, block>>>(out_weight_sums, dangling_mask, num_vertices);
    init_pageranks_kernel<<<grid_v, block>>>(pr_f64, pr_f32, num_vertices);
}

void launch_initial_dangling_sum(
    const double* pageranks, const uint8_t* dangling_mask,
    double* dangling_sum, int32_t num_vertices
) {
    cudaMemsetAsync(dangling_sum, 0, sizeof(double), 0);
    int block = 256;
    int grid = (int)min((int64_t)4096, ((int64_t)num_vertices + block - 1) / block);
    compute_initial_dangling_sum_kernel<<<grid, block>>>(pageranks, dangling_mask, dangling_sum, num_vertices);
}

void launch_fused_update_mixed(
    const float* spmv_f32, double* pr_f64, float* pr_f32,
    const double* d_dangling_sum_cur, double* d_dangling_sum_next,
    double* d_diff, const uint8_t* dangling_mask,
    double alpha, double one_over_n, int32_t num_vertices
) {
    cudaMemsetAsync(d_dangling_sum_next, 0, sizeof(double), 0);
    cudaMemsetAsync(d_diff, 0, sizeof(double), 0);
    int block = 256;
    int grid = (int)min((int64_t)4096, ((int64_t)num_vertices + block - 1) / block);
    fused_update_mixed_kernel<<<grid, block>>>(
        spmv_f32, pr_f64, pr_f32,
        d_dangling_sum_cur, d_dangling_sum_next, d_diff, dangling_mask,
        alpha, one_over_n, num_vertices);
}

void launch_sparse_mixed_iteration(
    const int32_t* offsets, const int32_t* indices,
    const float* norm_weights_f32,
    const float* pr_old_f32, const double* pr_old_f64,
    double* pr_new_f64, float* pr_new_f32,
    const double* d_dangling_sum_cur, double* d_dangling_sum_next,
    double* d_diff, const uint8_t* dangling_mask,
    double alpha, double one_over_n, int32_t num_vertices
) {
    cudaMemsetAsync(d_dangling_sum_next, 0, sizeof(double), 0);
    cudaMemsetAsync(d_diff, 0, sizeof(double), 0);
    int block = 256;
    int grid = (int)min((int64_t)4096, ((int64_t)num_vertices + block - 1) / block);
    pagerank_iteration_sparse_mixed<<<grid, block>>>(
        offsets, indices, norm_weights_f32,
        pr_old_f32, pr_old_f64, pr_new_f64, pr_new_f32,
        d_dangling_sum_cur, d_dangling_sum_next, d_diff, dangling_mask,
        alpha, one_over_n, num_vertices);
}

void launch_copy_f64_to_f32(
    const double* src_f64, float* dst_f32, int32_t n
) {
    int block = 256;
    int grid = (int)min((int64_t)65535, ((int64_t)n + block - 1) / block);
    copy_f64_to_f32_kernel<<<grid, block>>>(src_f64, dst_f32, n);
}

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

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    cache.ensure_pr_f32_a(num_vertices);
    cache.ensure_out_weight_sums(num_vertices);
    cache.ensure_dangling_mask(num_vertices);
    cache.ensure_pr_f64_b(num_vertices);
    cache.ensure_pr_f32_b(num_vertices);
    cache.ensure_spmv_f32(num_vertices);
    if (num_edges > 0) cache.ensure_norm_weights_f32(num_edges);
    cache.ensure_fixed();

    double avg_degree = (num_vertices > 0) ? (double)num_edges / num_vertices : 0.0;
    bool use_sparse_fused = (avg_degree <= 4.0) && (num_edges > 0);

    double* d_pr_f64 = pageranks;
    float* d_pr_f32 = cache.pr_f32_a;
    float* d_norm_weights_f32 = cache.norm_weights_f32;
    uint8_t* d_dangling_mask = cache.dangling_mask;
    double* d_dangling_sum_cur = cache.dangling_sum_a;
    double* d_dangling_sum_next = cache.dangling_sum_b;
    double* d_diff = cache.diff;

    
    launch_setup(d_indices, edge_weights,
                 cache.out_weight_sums,
                 d_norm_weights_f32, d_dangling_mask, d_pr_f64, d_pr_f32,
                 num_vertices, num_edges);

    if (initial_pageranks != nullptr) {
        cudaMemcpy(d_pr_f64, initial_pageranks,
                   (int64_t)num_vertices * sizeof(double), cudaMemcpyDeviceToDevice);
        launch_copy_f64_to_f32(d_pr_f64, d_pr_f32, num_vertices);
    }

    launch_initial_dangling_sum(d_pr_f64, d_dangling_mask, d_dangling_sum_cur, num_vertices);

    double one_over_n = 1.0 / (double)num_vertices;
    size_t iterations = 0;
    bool converged = false;

    if (use_sparse_fused) {
        
        double* d_pr_f64_old = d_pr_f64;
        float* d_pr_f32_old = d_pr_f32;
        double* d_pr_f64_new = cache.pr_f64_b;
        float* d_pr_f32_new = cache.pr_f32_b;

        for (size_t iter = 0; iter < max_iterations; iter++) {
            launch_sparse_mixed_iteration(
                d_offsets, d_indices, d_norm_weights_f32,
                d_pr_f32_old, d_pr_f64_old,
                d_pr_f64_new, d_pr_f32_new,
                d_dangling_sum_cur, d_dangling_sum_next,
                d_diff, d_dangling_mask,
                alpha, one_over_n, num_vertices);

            double h_diff;
            cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);
            iterations = iter + 1;
            if (h_diff < epsilon) { converged = true; break; }

            std::swap(d_pr_f64_old, d_pr_f64_new);
            std::swap(d_pr_f32_old, d_pr_f32_new);
            std::swap(d_dangling_sum_cur, d_dangling_sum_next);
        }

        if (!converged) iterations = max_iterations;

        
        double* d_result = converged ? d_pr_f64_new : d_pr_f64_old;
        if (d_result != d_pr_f64) {
            cudaMemcpy(d_pr_f64, d_result,
                       (int64_t)num_vertices * sizeof(double), cudaMemcpyDeviceToDevice);
        }

    } else if (num_edges > 0) {
        
        float* d_spmv_f32 = cache.spmv_f32;

        cusparseSpMatDescr_t mat_descr = nullptr;
        cusparseDnVecDescr_t x_descr = nullptr, y_descr = nullptr;

        cusparseCreateCsr(&mat_descr,
            num_vertices, num_vertices, num_edges,
            (void*)d_offsets, (void*)d_indices, (void*)d_norm_weights_f32,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        cusparseCreateDnVec(&x_descr, num_vertices, d_pr_f32, CUDA_R_32F);
        cusparseCreateDnVec(&y_descr, num_vertices, d_spmv_f32, CUDA_R_32F);

        float h_one = 1.0f, h_zero = 0.0f;
        size_t buffer_size = 0;
        cusparseSpMV_bufferSize(
            cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_one, mat_descr, x_descr, &h_zero, y_descr,
            CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &buffer_size);

        if (buffer_size > 0) {
            cache.ensure_spmv_buffer(buffer_size);
        }

        cusparseSpMV_preprocess(
            cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_one, mat_descr, x_descr, &h_zero, y_descr,
            CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, cache.spmv_buffer);

        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);

        for (size_t iter = 0; iter < max_iterations; iter++) {
            cusparseSpMV(
                cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                cache.d_one, mat_descr, x_descr,
                cache.d_zero, y_descr,
                CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, cache.spmv_buffer);

            launch_fused_update_mixed(d_spmv_f32, d_pr_f64, d_pr_f32,
                                      d_dangling_sum_cur, d_dangling_sum_next,
                                      d_diff, d_dangling_mask,
                                      alpha, one_over_n, num_vertices);

            double h_diff;
            cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);
            iterations = iter + 1;
            if (h_diff < epsilon) { converged = true; break; }

            std::swap(d_dangling_sum_cur, d_dangling_sum_next);
        }

        if (!converged) iterations = max_iterations;

        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
        cusparseDestroySpMat(mat_descr);
        cusparseDestroyDnVec(x_descr);
        cusparseDestroyDnVec(y_descr);

    } else {
        
        cudaMemsetAsync(cache.spmv_f32, 0, (int64_t)num_vertices * sizeof(float), 0);
        for (size_t iter = 0; iter < max_iterations; iter++) {
            launch_fused_update_mixed(cache.spmv_f32, d_pr_f64, d_pr_f32,
                                      d_dangling_sum_cur, d_dangling_sum_next,
                                      d_diff, d_dangling_mask,
                                      alpha, one_over_n, num_vertices);
            double h_diff;
            cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);
            iterations = iter + 1;
            if (h_diff < epsilon) { converged = true; break; }
            std::swap(d_dangling_sum_cur, d_dangling_sum_next);
        }
        if (!converged) iterations = max_iterations;
    }

    return PageRankResult{iterations, converged};
}

}  
