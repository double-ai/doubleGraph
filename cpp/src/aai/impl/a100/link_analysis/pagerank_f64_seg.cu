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
#include <cstddef>
#include <cmath>

namespace aai {

namespace {





__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    double* __restrict__ out_weight_sums,
    int32_t num_edges
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_edges; i += blockDim.x * gridDim.x)
        atomicAdd(&out_weight_sums[indices[i]], edge_weights[i]);
}

__global__ void init_pagerank_kernel(double* pr, int32_t N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        pr[i] = 1.0 / (double)N;
}

__global__ void normalize_and_dangling_kernel(
    const double* __restrict__ pr,
    const double* __restrict__ out_weight_sums,
    double* __restrict__ pr_normalized,
    double* __restrict__ dangling_sum,
    int32_t N
) {
    __shared__ double sdata[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    double my_dangling = 0.0;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) {
        double ow = out_weight_sums[i];
        double p = pr[i];
        if (ow > 0.0) {
            pr_normalized[i] = p / ow;
        } else {
            pr_normalized[i] = 0.0;
            my_dangling += p;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        my_dangling += __shfl_down_sync(0xffffffff, my_dangling, offset);
    if (lane == 0) sdata[warp] = my_dangling;
    __syncthreads();

    int nw = blockDim.x >> 5;
    if (warp == 0) {
        my_dangling = (lane < nw) ? sdata[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1)
            my_dangling += __shfl_down_sync(0xffffffff, my_dangling, offset);
        if (lane == 0) atomicAdd(dangling_sum, my_dangling);
    }
}

__global__ void update_and_diff_inplace_kernel(
    double* __restrict__ pr_new,
    const double* __restrict__ pr_old,
    double alpha,
    const double* __restrict__ dangling_sum,
    double base_score,
    double alpha_over_N,
    double* __restrict__ diff_result,
    int32_t N
) {
    __shared__ double sdata[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    double my_diff = 0.0;
    double dang_contrib = (*dangling_sum) * alpha_over_N;

    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) {
        double spmv_val = pr_new[i];
        double new_val = base_score + dang_contrib + alpha * spmv_val;
        double old_val = pr_old[i];
        pr_new[i] = new_val;
        my_diff += fabs(new_val - old_val);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        my_diff += __shfl_down_sync(0xffffffff, my_diff, offset);
    if (lane == 0) sdata[warp] = my_diff;
    __syncthreads();

    int nw = blockDim.x >> 5;
    if (warp == 0) {
        my_diff = (lane < nw) ? sdata[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1)
            my_diff += __shfl_down_sync(0xffffffff, my_diff, offset);
        if (lane == 0) atomicAdd(diff_result, my_diff);
    }
}





void launch_compute_out_weight_sums(const int32_t* indices, const double* edge_weights,
                                     double* out_weight_sums, int32_t num_edges, cudaStream_t stream) {
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    compute_out_weight_sums_kernel<<<grid, block, 0, stream>>>(indices, edge_weights, out_weight_sums, num_edges);
}

void launch_init_pagerank(double* pr, int32_t N, cudaStream_t stream) {
    int block = 256;
    int grid = (N + block - 1) / block;
    init_pagerank_kernel<<<grid, block, 0, stream>>>(pr, N);
}

void launch_normalize_and_dangling(const double* pr, const double* out_weight_sums,
                                    double* pr_normalized, double* dangling_sum,
                                    int32_t N, cudaStream_t stream) {
    int block = 256;
    int grid = (N + block - 1) / block;
    if (grid > 1024) grid = 1024;
    normalize_and_dangling_kernel<<<grid, block, 0, stream>>>(pr, out_weight_sums, pr_normalized, dangling_sum, N);
}

void launch_update_and_diff_inplace(double* pr_new, const double* pr_old,
                                     double alpha, const double* dangling_sum, double base_score,
                                     double alpha_over_N, double* diff_result, int32_t N,
                                     cudaStream_t stream) {
    int block = 256;
    int grid = (N + block - 1) / block;
    if (grid > 1024) grid = 1024;
    update_and_diff_inplace_kernel<<<grid, block, 0, stream>>>(pr_new, pr_old, alpha,
                                                                dangling_sum, base_score, alpha_over_N,
                                                                diff_result, N);
}





struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    double* out_ws = nullptr;
    double* pr_norm = nullptr;
    double* pr0 = nullptr;
    double* pr1 = nullptr;
    double* scalars = nullptr;
    uint8_t* spmv_buffer = nullptr;

    int32_t out_ws_capacity = 0;
    int32_t pr_norm_capacity = 0;
    int32_t pr0_capacity = 0;
    int32_t pr1_capacity = 0;
    bool scalars_allocated = false;
    std::size_t spmv_buffer_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (out_ws) cudaFree(out_ws);
        if (pr_norm) cudaFree(pr_norm);
        if (pr0) cudaFree(pr0);
        if (pr1) cudaFree(pr1);
        if (scalars) cudaFree(scalars);
        if (spmv_buffer) cudaFree(spmv_buffer);
    }

    void ensure_vertex_buffers(int32_t N) {
        if (out_ws_capacity < N) {
            if (out_ws) cudaFree(out_ws);
            cudaMalloc(&out_ws, (std::size_t)N * sizeof(double));
            out_ws_capacity = N;
        }
        if (pr_norm_capacity < N) {
            if (pr_norm) cudaFree(pr_norm);
            cudaMalloc(&pr_norm, (std::size_t)N * sizeof(double));
            pr_norm_capacity = N;
        }
        if (pr0_capacity < N) {
            if (pr0) cudaFree(pr0);
            cudaMalloc(&pr0, (std::size_t)N * sizeof(double));
            pr0_capacity = N;
        }
        if (pr1_capacity < N) {
            if (pr1) cudaFree(pr1);
            cudaMalloc(&pr1, (std::size_t)N * sizeof(double));
            pr1_capacity = N;
        }
        if (!scalars_allocated) {
            cudaMalloc(&scalars, 2 * sizeof(double));
            scalars_allocated = true;
        }
    }

    void ensure_spmv_buffer(std::size_t size) {
        if (spmv_buffer_capacity < size) {
            if (spmv_buffer) cudaFree(spmv_buffer);
            cudaMalloc(&spmv_buffer, size);
            spmv_buffer_capacity = size;
        }
    }
};

}  

PageRankResult pagerank_seg(const graph32_t& graph,
                            const double* edge_weights,
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
    int32_t N = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;
    cudaStream_t stream = 0;

    cache.ensure_vertex_buffers(N);

    double* d_pr_norm = cache.pr_norm;
    double* d_pr = cache.pr0;
    double* d_pr_new = cache.pr1;
    double* d_dangling_sum = cache.scalars;
    double* d_diff = cache.scalars + 1;

    
    cudaMemsetAsync(cache.out_ws, 0, (std::size_t)N * sizeof(double), stream);
    launch_compute_out_weight_sums(d_indices, edge_weights, cache.out_ws, E, stream);
    const double* d_out_ws = cache.out_ws;

    
    if (initial_pageranks) {
        cudaMemcpyAsync(d_pr, initial_pageranks,
                         (std::size_t)N * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else {
        launch_init_pagerank(d_pr, N, stream);
    }

    
    cusparseSpMatDescr_t mat_descr;
    cusparseCreateCsr(&mat_descr, N, N, E,
                      (void*)d_offsets, (void*)d_indices, (void*)edge_weights,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseDnVecDescr_t x_descr, y_descr;
    cusparseCreateDnVec(&x_descr, N, d_pr_norm, CUDA_R_64F);
    cusparseCreateDnVec(&y_descr, N, d_pr_new, CUDA_R_64F);

    double h_one = 1.0, h_zero = 0.0;

    std::size_t buffer_size = 0;
    cusparseSpMV_bufferSize(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &h_one, mat_descr, x_descr, &h_zero, y_descr,
                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);

    if (buffer_size > 0) {
        cache.ensure_spmv_buffer(buffer_size);
    }
    void* d_buffer = (buffer_size > 0) ? cache.spmv_buffer : nullptr;

    cusparseSpMV_preprocess(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &h_one, mat_descr, x_descr, &h_zero, y_descr,
                            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);

    double base_score = (1.0 - alpha) / (double)N;
    double alpha_over_N = alpha / (double)N;

    
    std::size_t iterations = 0;
    bool converged = false;
    double h_diff;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        cudaMemsetAsync(d_dangling_sum, 0, 2 * sizeof(double), stream);

        launch_normalize_and_dangling(d_pr, d_out_ws, d_pr_norm, d_dangling_sum, N, stream);

        cusparseDnVecSetValues(y_descr, d_pr_new);

        cusparseSpMV(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &h_one, mat_descr, x_descr, &h_zero, y_descr,
                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);

        launch_update_and_diff_inplace(d_pr_new, d_pr, alpha, d_dangling_sum,
                                        base_score, alpha_over_N, d_diff, N, stream);

        cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);

        iterations = iter + 1;

        double* tmp = d_pr;
        d_pr = d_pr_new;
        d_pr_new = tmp;

        if (h_diff < epsilon) {
            converged = true;
            break;
        }
    }

    
    cudaMemcpy(pageranks, d_pr, (std::size_t)N * sizeof(double), cudaMemcpyDeviceToDevice);

    cusparseDestroySpMat(mat_descr);
    cusparseDestroyDnVec(x_descr);
    cusparseDestroyDnVec(y_descr);

    return PageRankResult{iterations, converged};
}

}  
