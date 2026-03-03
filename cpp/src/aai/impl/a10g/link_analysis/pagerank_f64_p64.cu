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

namespace aai {

namespace {

struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    double* h_scalars_pinned = nullptr;
    double* d_alpha_sp = nullptr;
    double* d_beta_sp = nullptr;

    int32_t* d_num_dangling = nullptr;
    double* d_diff = nullptr;
    double* d_dang = nullptr;
    double* d_pers_sum = nullptr;

    double* pr_buf0 = nullptr;
    int64_t pr_buf0_cap = 0;
    double* pr_buf1 = nullptr;
    int64_t pr_buf1_cap = 0;
    double* out_weights_buf = nullptr;
    int64_t out_weights_cap = 0;
    int32_t* dangling_indices = nullptr;
    int64_t dangling_indices_cap = 0;
    double* pers_norm_sparse = nullptr;
    int64_t pers_norm_sparse_cap = 0;
    float* scaled_w_f32 = nullptr;
    int64_t scaled_w_f32_cap = 0;
    void* spmv_buffer = nullptr;
    size_t spmv_buffer_cap = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cudaMallocHost(&h_scalars_pinned, 4 * sizeof(double));
        cudaMalloc(&d_alpha_sp, sizeof(double));
        cudaMalloc(&d_beta_sp, sizeof(double));
        double h_one = 1.0, h_zero = 0.0;
        cudaMemcpy(d_alpha_sp, &h_one, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta_sp, &h_zero, sizeof(double), cudaMemcpyHostToDevice);
        cudaMalloc(&d_num_dangling, sizeof(int32_t));
        cudaMalloc(&d_diff, sizeof(double));
        cudaMalloc(&d_dang, sizeof(double));
        cudaMalloc(&d_pers_sum, sizeof(double));
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (h_scalars_pinned) cudaFreeHost(h_scalars_pinned);
        if (d_alpha_sp) cudaFree(d_alpha_sp);
        if (d_beta_sp) cudaFree(d_beta_sp);
        if (d_num_dangling) cudaFree(d_num_dangling);
        if (d_diff) cudaFree(d_diff);
        if (d_dang) cudaFree(d_dang);
        if (d_pers_sum) cudaFree(d_pers_sum);
        if (pr_buf0) cudaFree(pr_buf0);
        if (pr_buf1) cudaFree(pr_buf1);
        if (out_weights_buf) cudaFree(out_weights_buf);
        if (dangling_indices) cudaFree(dangling_indices);
        if (pers_norm_sparse) cudaFree(pers_norm_sparse);
        if (scaled_w_f32) cudaFree(scaled_w_f32);
        if (spmv_buffer) cudaFree(spmv_buffer);
    }

    void ensure(int32_t n, int32_t m, int64_t ps) {
        if (pr_buf0_cap < n) {
            if (pr_buf0) cudaFree(pr_buf0);
            cudaMalloc(&pr_buf0, (size_t)n * sizeof(double));
            pr_buf0_cap = n;
        }
        if (pr_buf1_cap < n) {
            if (pr_buf1) cudaFree(pr_buf1);
            cudaMalloc(&pr_buf1, (size_t)n * sizeof(double));
            pr_buf1_cap = n;
        }
        if (out_weights_cap < n) {
            if (out_weights_buf) cudaFree(out_weights_buf);
            cudaMalloc(&out_weights_buf, (size_t)n * sizeof(double));
            out_weights_cap = n;
        }
        if (dangling_indices_cap < n) {
            if (dangling_indices) cudaFree(dangling_indices);
            cudaMalloc(&dangling_indices, (size_t)n * sizeof(int32_t));
            dangling_indices_cap = n;
        }
        if (pers_norm_sparse_cap < ps) {
            if (pers_norm_sparse) cudaFree(pers_norm_sparse);
            cudaMalloc(&pers_norm_sparse, (size_t)ps * sizeof(double));
            pers_norm_sparse_cap = ps;
        }
        if (m > 0 && scaled_w_f32_cap < m) {
            if (scaled_w_f32) cudaFree(scaled_w_f32);
            cudaMalloc(&scaled_w_f32, (size_t)m * sizeof(float));
            scaled_w_f32_cap = m;
        }
    }

    void ensure_spmv_buffer(size_t size) {
        if (spmv_buffer_cap < size) {
            if (spmv_buffer) cudaFree(spmv_buffer);
            cudaMalloc(&spmv_buffer, size);
            spmv_buffer_cap = size;
        }
    }
};





__global__ void compute_out_weights_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    double* __restrict__ out_weights,
    int num_edges)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < num_edges) {
        atomicAdd(&out_weights[indices[j]], weights[j]);
    }
}

__global__ void prescale_and_cast_kernel(
    const double* __restrict__ weights,
    const int32_t* __restrict__ indices,
    const double* __restrict__ out_weights,
    float* __restrict__ scaled_w_f32,
    double alpha,
    int num_edges)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < num_edges) {
        scaled_w_f32[j] = (float)(alpha * weights[j] / out_weights[indices[j]]);
    }
}

__global__ void build_dangling_indices_kernel(
    const double* __restrict__ out_weights,
    int32_t* __restrict__ dangling_indices,
    int32_t* __restrict__ d_num_dangling,
    int n)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n && out_weights[v] == 0.0) {
        int idx = atomicAdd(d_num_dangling, 1);
        dangling_indices[idx] = v;
    }
}

__global__ void build_pers_norm_sparse_kernel(
    const double* __restrict__ pers_values,
    double* __restrict__ pers_norm_sparse,
    double inv_pers_sum,
    int pers_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pers_size) {
        pers_norm_sparse[i] = pers_values[i] * inv_pers_sum;
    }
}

__global__ void init_pr_uniform_kernel(double* __restrict__ pr, double inv_n, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        pr[v] = inv_n;
    }
}

__global__ void pers_sum_kernel(
    const double* __restrict__ pers_values,
    double* __restrict__ d_sum,
    int n)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    double thread_sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        thread_sum += pers_values[i];
    double block_sum = BlockReduce(temp).Sum(thread_sum);
    if (threadIdx.x == 0) atomicAdd(d_sum, block_sum);
}





__global__ void gather_dangling_sum_kernel(
    const double* __restrict__ pr,
    const int32_t* __restrict__ dangling_indices,
    double* __restrict__ d_dangling_sum,
    int num_dangling)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    double thread_sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_dangling; i += blockDim.x * gridDim.x)
        thread_sum += pr[dangling_indices[i]];
    double block_sum = BlockReduce(temp).Sum(thread_sum);
    if (threadIdx.x == 0) atomicAdd(d_dangling_sum, block_sum);
}

__global__ void spmv_f32val_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ values_f32,
    const double* __restrict__ x,
    double* __restrict__ y,
    int n)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        int start = offsets[v];
        int end = offsets[v + 1];
        double sum = 0.0;
        for (int j = start; j < end; ++j) {
            sum += (double)values_f32[j] * x[indices[j]];
        }
        y[v] = sum;
    }
}

__global__ void scatter_pers_kernel(
    double* __restrict__ y,
    const int32_t* __restrict__ pers_vertices,
    const double* __restrict__ pers_norm_sparse,
    const double* __restrict__ d_dangling_sum,
    double alpha,
    double one_minus_alpha,
    int pers_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pers_size) {
        double base = alpha * d_dangling_sum[0] + one_minus_alpha;
        y[pers_vertices[i]] += base * pers_norm_sparse[i];
    }
}

__global__ void diff_kernel(
    const double* __restrict__ pr_new,
    const double* __restrict__ pr_old,
    double* __restrict__ d_diff,
    int n)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    double thread_diff = 0.0;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += blockDim.x * gridDim.x)
        thread_diff += fabs(pr_new[v] - pr_old[v]);
    double block_diff = BlockReduce(temp).Sum(thread_diff);
    if (threadIdx.x == 0) atomicAdd(d_diff, block_diff);
}





void launch_compute_out_weights(const int32_t* indices, const double* weights, double* out_weights, int num_edges, cudaStream_t stream) {
    if (num_edges == 0) return;
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    compute_out_weights_kernel<<<grid, block, 0, stream>>>(indices, weights, out_weights, num_edges);
}

void launch_prescale_and_cast(const double* weights, const int32_t* indices, const double* out_weights, float* scaled_w_f32, double alpha, int num_edges, cudaStream_t stream) {
    if (num_edges == 0) return;
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    prescale_and_cast_kernel<<<grid, block, 0, stream>>>(weights, indices, out_weights, scaled_w_f32, alpha, num_edges);
}

void launch_build_dangling_indices(const double* out_weights, int32_t* dangling_indices, int32_t* d_num_dangling, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    build_dangling_indices_kernel<<<grid, block, 0, stream>>>(out_weights, dangling_indices, d_num_dangling, n);
}

void launch_build_pers_norm_sparse(const double* pers_values, double* pers_norm_sparse, double inv_pers_sum, int pers_size, cudaStream_t stream) {
    if (pers_size == 0) return;
    int block = 256;
    int grid = (pers_size + block - 1) / block;
    build_pers_norm_sparse_kernel<<<grid, block, 0, stream>>>(pers_values, pers_norm_sparse, inv_pers_sum, pers_size);
}

void launch_init_pr_uniform(double* pr, double inv_n, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    init_pr_uniform_kernel<<<grid, block, 0, stream>>>(pr, inv_n, n);
}

void launch_pers_sum(const double* pers_values, double* d_sum, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 256) grid = 256;
    pers_sum_kernel<<<grid, block, 0, stream>>>(pers_values, d_sum, n);
}

void launch_gather_dangling_sum(const double* pr, const int32_t* dangling_indices, double* d_dangling_sum, int num_dangling, cudaStream_t stream) {
    if (num_dangling == 0) return;
    int block = 256;
    int grid = (num_dangling + block - 1) / block;
    if (grid > 1024) grid = 1024;
    gather_dangling_sum_kernel<<<grid, block, 0, stream>>>(pr, dangling_indices, d_dangling_sum, num_dangling);
}

void launch_spmv_f32val(const int32_t* offsets, const int32_t* indices, const float* values_f32, const double* x, double* y, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    spmv_f32val_kernel<<<grid, block, 0, stream>>>(offsets, indices, values_f32, x, y, n);
}

void launch_scatter_pers(double* y, const int32_t* pers_vertices, const double* pers_norm_sparse, const double* d_dangling_sum, double alpha, double one_minus_alpha, int pers_size, cudaStream_t stream) {
    if (pers_size == 0) return;
    int block = 256;
    int grid = (pers_size + block - 1) / block;
    scatter_pers_kernel<<<grid, block, 0, stream>>>(y, pers_vertices, pers_norm_sparse, d_dangling_sum, alpha, one_minus_alpha, pers_size);
}

void launch_diff(const double* pr_new, const double* pr_old, double* d_diff, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 2048) grid = 2048;
    diff_kernel<<<grid, block, 0, stream>>>(pr_new, pr_old, d_diff, n);
}

}  

PageRankResult personalized_pagerank(const graph32_t& graph,
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
    int32_t n = graph.number_of_vertices;
    int32_t m = graph.number_of_edges;

    bool has_initial_guess = (initial_pageranks != nullptr);
    bool has_precomputed = (precomputed_vertex_out_weight_sums != nullptr);

    cudaStream_t stream = 0;
    double one_minus_alpha = 1.0 - alpha;
    int pers_size = static_cast<int>(personalization_size);

    cache.ensure(n, m, static_cast<int64_t>(personalization_size));

    double* pr_cur = cache.pr_buf0;
    double* pr_new = cache.pr_buf1;
    double* out_weights = cache.out_weights_buf;

    

    
    if (has_precomputed) {
        cudaMemcpyAsync(out_weights, precomputed_vertex_out_weight_sums,
                        (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemsetAsync(out_weights, 0, (size_t)n * sizeof(double), stream);
        launch_compute_out_weights(d_indices, edge_weights, out_weights, m, stream);
    }

    
    if (m > 0) {
        launch_prescale_and_cast(edge_weights, d_indices, out_weights, cache.scaled_w_f32, alpha, m, stream);
    }

    
    cudaMemsetAsync(cache.d_num_dangling, 0, sizeof(int32_t), stream);
    launch_build_dangling_indices(out_weights, cache.dangling_indices, cache.d_num_dangling, n, stream);

    
    cudaMemsetAsync(cache.d_pers_sum, 0, sizeof(double), stream);
    launch_pers_sum(personalization_values, cache.d_pers_sum, pers_size, stream);

    cudaMemcpyAsync(&cache.h_scalars_pinned[0], cache.d_pers_sum, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&cache.h_scalars_pinned[1], cache.d_num_dangling, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    double h_pers_sum = cache.h_scalars_pinned[0];
    int32_t h_num_dangling;
    memcpy(&h_num_dangling, &cache.h_scalars_pinned[1], sizeof(int32_t));

    double inv_pers_sum = (h_pers_sum > 0.0) ? 1.0 / h_pers_sum : 0.0;
    launch_build_pers_norm_sparse(personalization_values, cache.pers_norm_sparse, inv_pers_sum, pers_size, stream);

    
    if (has_initial_guess) {
        cudaMemcpyAsync(pr_cur, initial_pageranks,
                        (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else {
        double inv_n = 1.0 / (double)n;
        launch_init_pr_uniform(pr_cur, inv_n, n, stream);
    }

    
    cudaMemsetAsync(cache.d_dang, 0, sizeof(double), stream);
    if (h_num_dangling > 0) {
        launch_gather_dangling_sum(pr_cur, cache.dangling_indices, cache.d_dang, h_num_dangling, stream);
    }

    
    double avg_degree = (n > 0) ? (double)m / n : 0.0;
    bool use_cusparse = (avg_degree >= 6.0) && (m > 0);

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;

    if (use_cusparse) {
        cusparseSetStream(cache.cusparse_handle, stream);

        cusparseCreateCsr(&matA, n, n, m,
            (void*)d_offsets, (void*)d_indices, (void*)cache.scaled_w_f32,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        cusparseCreateDnVec(&vecX, n, pr_cur, CUDA_R_64F);
        cusparseCreateDnVec(&vecY, n, pr_new, CUDA_R_64F);

        double h_alpha = 1.0, h_beta = 0.0;
        size_t spmv_buffer_size = 0;
        cusparseSpMV_bufferSize(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_alpha, matA, vecX, &h_beta, vecY,
            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, &spmv_buffer_size);

        if (spmv_buffer_size > 0) {
            cache.ensure_spmv_buffer(spmv_buffer_size);
        }

        cusparseSpMV_preprocess(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_alpha, matA, vecX, &h_beta, vecY,
            CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, cache.spmv_buffer);

        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
    }

    
    bool converged = false;
    size_t iterations = 0;

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        
        if (use_cusparse) {
            cusparseDnVecSetValues(vecX, pr_cur);
            cusparseDnVecSetValues(vecY, pr_new);
            cusparseSpMV(cache.cusparse_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                cache.d_alpha_sp, matA, vecX, cache.d_beta_sp, vecY,
                CUDA_R_64F, CUSPARSE_SPMV_CSR_ALG1, cache.spmv_buffer);
        } else {
            launch_spmv_f32val(d_offsets, d_indices, cache.scaled_w_f32, pr_cur, pr_new, n, stream);
        }

        
        launch_scatter_pers(pr_new, personalization_vertices, cache.pers_norm_sparse, cache.d_dang,
                            alpha, one_minus_alpha, pers_size, stream);

        
        cudaMemsetAsync(cache.d_diff, 0, sizeof(double), stream);
        launch_diff(pr_new, pr_cur, cache.d_diff, n, stream);

        
        cudaMemsetAsync(cache.d_dang, 0, sizeof(double), stream);
        if (h_num_dangling > 0) {
            launch_gather_dangling_sum(pr_new, cache.dangling_indices, cache.d_dang, h_num_dangling, stream);
        }

        iterations = iter + 1;

        
        cudaMemcpyAsync(cache.h_scalars_pinned, cache.d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (cache.h_scalars_pinned[0] < epsilon) {
            converged = true;
            break;
        }

        
        double* tmp = pr_cur;
        pr_cur = pr_new;
        pr_new = tmp;
    }

    
    double* result_ptr = converged ? pr_new : pr_cur;

    
    cudaMemcpyAsync(pageranks, result_ptr,
                    (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    
    if (use_cusparse) {
        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
    }
    if (matA) cusparseDestroySpMat(matA);
    if (vecX) cusparseDestroyDnVec(vecX);
    if (vecY) cusparseDestroyDnVec(vecY);

    return PageRankResult{iterations, converged};
}

}  
