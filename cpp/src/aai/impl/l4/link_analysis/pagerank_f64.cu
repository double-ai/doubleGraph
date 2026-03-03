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
#include <utility>

namespace aai {

namespace {

#define BLOCK_SIZE 256





struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;

    
    double* out_ws = nullptr;
    int64_t out_ws_cap = 0;

    uint8_t* is_dangling = nullptr;
    int64_t is_dangling_cap = 0;

    float* spmv_buf = nullptr;
    int64_t spmv_cap = 0;

    float* pr_a = nullptr;
    int64_t pr_a_cap = 0;

    float* pr_b = nullptr;
    int64_t pr_b_cap = 0;

    
    float* scaled_w = nullptr;
    int64_t scaled_w_cap = 0;

    
    float* scratch = nullptr;

    
    uint8_t* cusparse_buf = nullptr;
    size_t cusparse_buf_cap = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
        cudaMalloc(&scratch, 5 * sizeof(float));
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (out_ws) cudaFree(out_ws);
        if (is_dangling) cudaFree(is_dangling);
        if (spmv_buf) cudaFree(spmv_buf);
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (scaled_w) cudaFree(scaled_w);
        if (scratch) cudaFree(scratch);
        if (cusparse_buf) cudaFree(cusparse_buf);
    }

    void ensure(int64_t n, int64_t m) {
        if (out_ws_cap < n) {
            if (out_ws) cudaFree(out_ws);
            cudaMalloc(&out_ws, n * sizeof(double));
            out_ws_cap = n;
        }
        if (is_dangling_cap < n) {
            if (is_dangling) cudaFree(is_dangling);
            cudaMalloc(&is_dangling, n * sizeof(uint8_t));
            is_dangling_cap = n;
        }
        if (spmv_cap < n) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, n * sizeof(float));
            spmv_cap = n;
        }
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
        if (scaled_w_cap < m) {
            if (scaled_w) cudaFree(scaled_w);
            cudaMalloc(&scaled_w, m * sizeof(float));
            scaled_w_cap = m;
        }
    }

    void ensure_cusparse(size_t size) {
        if (cusparse_buf_cap < size) {
            if (cusparse_buf) cudaFree(cusparse_buf);
            cudaMalloc(&cusparse_buf, size);
            cusparse_buf_cap = size;
        }
    }
};





__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    double* __restrict__ out_weight_sums,
    int32_t num_edges)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x)
        atomicAdd(&out_weight_sums[indices[idx]], edge_weights[idx]);
}

__global__ void compute_scaled_weights_f32_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const double* __restrict__ out_weight_sums,
    float* __restrict__ scaled_weights,
    int32_t num_edges)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x)
        scaled_weights[idx] = (float)(edge_weights[idx] / out_weight_sums[indices[idx]]);
}

__global__ void init_pr_f32_kernel(float* __restrict__ pr, int32_t n) {
    float inv_n = 1.0f / (float)n;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
        pr[idx] = inv_n;
}

__global__ void init_dangling_f32_kernel(
    const float* __restrict__ pr,
    const double* __restrict__ out_weight_sums,
    float* __restrict__ dangling_sum,
    uint8_t* __restrict__ is_dangling,
    int32_t n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float thread_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        bool d = (out_weight_sums[i] == 0.0);
        is_dangling[i] = d ? 1 : 0;
        if (d) thread_sum += pr[i];
    }
    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    if (threadIdx.x == 0 && block_sum != 0.0f) atomicAdd(dangling_sum, block_sum);
}

__global__ void update_diff_dangling_allfp32_kernel(
    const float* __restrict__ pr_old,
    const float* __restrict__ spmv,
    float* __restrict__ pr_new,
    float* __restrict__ d_diff,
    const float* __restrict__ d_dangling_curr,
    float* __restrict__ d_dangling_next,
    const uint8_t* __restrict__ is_dangling,
    int32_t n,
    float one_minus_alpha_over_n,
    float alpha_over_n,
    float alpha_f32)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    __shared__ float s_base;
    if (threadIdx.x == 0) {
        s_base = one_minus_alpha_over_n + alpha_over_n * (*d_dangling_curr);
    }
    __syncthreads();

    float base = s_base;
    float thread_diff = 0.0f;
    float thread_dangling = 0.0f;

    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += blockDim.x * gridDim.x) {
        float new_val = base + alpha_f32 * spmv[v];
        pr_new[v] = new_val;
        thread_diff += fabsf(new_val - pr_old[v]);
        if (is_dangling[v]) thread_dangling += new_val;
    }

    float block_diff = BlockReduce(temp_storage).Sum(thread_diff);
    if (threadIdx.x == 0 && block_diff != 0.0f) atomicAdd(d_diff, block_diff);
    __syncthreads();
    float block_dangling = BlockReduce(temp_storage).Sum(thread_dangling);
    if (threadIdx.x == 0 && block_dangling != 0.0f) atomicAdd(d_dangling_next, block_dangling);
}

__global__ void convert_f64_to_f32_kernel(const double* __restrict__ src, float* __restrict__ dst, int32_t n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
        dst[idx] = (float)src[idx];
}

__global__ void convert_f32_to_f64_kernel(const float* __restrict__ src, double* __restrict__ dst, int32_t n) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
        dst[idx] = (double)src[idx];
}





static void launch_compute_out_weight_sums(const int32_t* indices, const double* edge_weights, double* out_weight_sums, int32_t num_edges) {
    int grid = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE; if (grid > 2048) grid = 2048;
    compute_out_weight_sums_kernel<<<grid, BLOCK_SIZE>>>(indices, edge_weights, out_weight_sums, num_edges);
}

static void launch_compute_scaled_weights_f32(const int32_t* indices, const double* edge_weights, const double* out_weight_sums, float* scaled_weights, int32_t num_edges) {
    int grid = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE; if (grid > 2048) grid = 2048;
    compute_scaled_weights_f32_kernel<<<grid, BLOCK_SIZE>>>(indices, edge_weights, out_weight_sums, scaled_weights, num_edges);
}

static void launch_init_pr_f32(float* pr, int32_t n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; if (grid > 2048) grid = 2048;
    init_pr_f32_kernel<<<grid, BLOCK_SIZE>>>(pr, n);
}

static void launch_init_dangling_f32(const float* pr, const double* out_weight_sums, float* dangling_sum, uint8_t* is_dangling, int32_t n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; if (grid > 1024) grid = 1024;
    init_dangling_f32_kernel<<<grid, BLOCK_SIZE>>>(pr, out_weight_sums, dangling_sum, is_dangling, n);
}

static void launch_update_diff_dangling_f32(const float* pr_old, const float* spmv, float* pr_new, float* d_diff, const float* d_dangling_curr, float* d_dangling_next, const uint8_t* is_dangling, int32_t n, float one_minus_alpha_over_n, float alpha_over_n, float alpha_f32) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; if (grid > 2048) grid = 2048;
    update_diff_dangling_allfp32_kernel<<<grid, BLOCK_SIZE>>>(pr_old, spmv, pr_new, d_diff, d_dangling_curr, d_dangling_next, is_dangling, n, one_minus_alpha_over_n, alpha_over_n, alpha_f32);
}

static void launch_convert_f64_to_f32(const double* src, float* dst, int32_t n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; if (grid > 2048) grid = 2048;
    convert_f64_to_f32_kernel<<<grid, BLOCK_SIZE>>>(src, dst, n);
}

static void launch_convert_f32_to_f64(const float* src, double* dst, int32_t n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE; if (grid > 2048) grid = 2048;
    convert_f32_to_f64_kernel<<<grid, BLOCK_SIZE>>>(src, dst, n);
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
    int64_t n = static_cast<int64_t>(num_vertices);
    int64_t m = static_cast<int64_t>(num_edges);

    cache.ensure(n, m);

    cudaMemsetAsync(cache.out_ws, 0, n * sizeof(double));
    launch_compute_out_weight_sums(graph.indices, edge_weights, cache.out_ws, num_edges);
    const double* d_out_ws = cache.out_ws;

    
    launch_compute_scaled_weights_f32(graph.indices, edge_weights, d_out_ws, cache.scaled_w, num_edges);

    
    if (initial_pageranks) {
        launch_convert_f64_to_f32(initial_pageranks, cache.pr_a, num_vertices);
    } else {
        launch_init_pr_f32(cache.pr_a, num_vertices);
    }

    
    float h_scratch[5] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    cudaMemcpyAsync(cache.scratch, h_scratch, 5 * sizeof(float), cudaMemcpyHostToDevice);

    float* d_dangling_a = cache.scratch;
    float* d_dangling_b = cache.scratch + 1;
    float* d_diff = cache.scratch + 2;
    float* d_alpha_val = cache.scratch + 3;
    float* d_beta_val = cache.scratch + 4;

    
    launch_init_dangling_f32(cache.pr_a, d_out_ws, d_dangling_a, cache.is_dangling, num_vertices);

    
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, n, n, m,
        (void*)graph.offsets, (void*)graph.indices, (void*)cache.scaled_w,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, n, cache.pr_a, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, n, cache.spmv_buf, CUDA_R_32F);

    
    cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
    float h_one = 1.0f, h_zero = 0.0f;
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_one, matA, vecX, &h_zero, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &bufferSize);

    if (bufferSize > 0) {
        cache.ensure_cusparse(bufferSize);
    }

    void* dBuffer = (bufferSize > 0) ? cache.cusparse_buf : nullptr;

    cusparseSpMV_preprocess(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_one, matA, vecX, &h_zero, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, dBuffer);

    cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);

    
    float one_minus_alpha_over_n_f32 = (float)((1.0 - alpha) / (double)num_vertices);
    float alpha_over_n_f32 = (float)(alpha / (double)num_vertices);
    float alpha_f32 = (float)alpha;
    float epsilon_f32 = (float)epsilon;

    float* d_pr_curr = cache.pr_a;
    float* d_pr_next = cache.pr_b;
    float* d_dangling_curr = d_dangling_a;
    float* d_dangling_next = d_dangling_b;
    size_t iterations = 0;
    bool converged = false;

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        
        cudaMemsetAsync(d_dangling_next, 0, sizeof(float));
        cudaMemsetAsync(d_diff, 0, sizeof(float));

        
        cusparseDnVecSetValues(vecX, d_pr_curr);
        cusparseSpMV(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_alpha_val, matA, vecX, d_beta_val, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, dBuffer);

        
        launch_update_diff_dangling_f32(
            d_pr_curr, cache.spmv_buf, d_pr_next, d_diff,
            d_dangling_curr, d_dangling_next,
            cache.is_dangling, num_vertices,
            one_minus_alpha_over_n_f32, alpha_over_n_f32, alpha_f32);

        
        float h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);

        iterations = iter + 1;
        std::swap(d_pr_curr, d_pr_next);
        std::swap(d_dangling_curr, d_dangling_next);

        if (h_diff < epsilon_f32) {
            converged = true;
            break;
        }
    }

    
    launch_convert_f32_to_f64(d_pr_curr, pageranks, num_vertices);

    
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(matA);

    return PageRankResult{iterations, converged};
}

}  
