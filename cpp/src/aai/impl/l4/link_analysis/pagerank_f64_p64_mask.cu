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
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace aai {

namespace {

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t _err = (call); \
    if (_err != CUSPARSE_STATUS_SUCCESS) { \
        throw std::runtime_error("cuSPARSE error"); \
    } \
} while(0)

constexpr int BLOCK_SIZE = 128;



__global__ void compute_out_weights_kernel(
    const int32_t* __restrict__ csc_indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ out_weight,
    int32_t num_edges)
{
    for (int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
         e < num_edges; e += blockDim.x * gridDim.x) {
        uint32_t mask_word = edge_mask[e >> 5];
        if ((mask_word >> (e & 31)) & 1) {
            atomicAdd(&out_weight[csc_indices[e]], weights[e]);
        }
    }
}

__global__ void compute_normalized_values_kernel(
    const int32_t* __restrict__ csc_indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ out_weight,
    double* __restrict__ norm_values,
    int32_t num_edges)
{
    for (int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
         e < num_edges; e += blockDim.x * gridDim.x) {
        uint32_t mask_word = edge_mask[e >> 5];
        if ((mask_word >> (e & 31)) & 1) {
            norm_values[e] = weights[e] / out_weight[csc_indices[e]];
        } else {
            norm_values[e] = 0.0;
        }
    }
}

__global__ void compute_dangling_mask_kernel(
    const double* __restrict__ out_weight,
    uint32_t* __restrict__ dangling_mask,
    int32_t N)
{
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N; i += blockDim.x * gridDim.x) {
        if (out_weight[i] == 0.0) {
            atomicOr(&dangling_mask[i >> 5], 1u << (i & 31));
        }
    }
}

__global__ void init_pr_uniform_kernel(double* __restrict__ pr, int32_t N)
{
    double val = 1.0 / (double)N;
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N; i += blockDim.x * gridDim.x) {
        pr[i] = val;
    }
}



__global__ void dangling_sum_kernel(
    const double* __restrict__ pr,
    const uint32_t* __restrict__ dangling_mask,
    double* __restrict__ d_result,
    int32_t N)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double local_sum = 0.0;
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N; i += blockDim.x * gridDim.x) {
        uint32_t mask_word = dangling_mask[i >> 5];
        if ((mask_word >> (i & 31)) & 1) {
            local_sum += pr[i];
        }
    }

    double block_sum = BlockReduce(temp_storage).Sum(local_sum);
    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(d_result, block_sum);
    }
}

__global__ void apply_pers_kernel(
    double* __restrict__ pr_new,
    const int32_t* __restrict__ pers_vertices,
    const double* __restrict__ pers_norm_vals,
    int32_t pers_size,
    const double* __restrict__ d_dangling_sum,
    double alpha)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pers_size) {
        double base = alpha * (*d_dangling_sum) + (1.0 - alpha);
        pr_new[pers_vertices[i]] += base * pers_norm_vals[i];
    }
}

struct DiffDangState {
    double diff_sum;
    double dang_sum;
};

struct DiffDangReducer {
    __device__ __forceinline__ DiffDangState operator()(
        const DiffDangState& a, const DiffDangState& b) const {
        return {a.diff_sum + b.diff_sum, a.dang_sum + b.dang_sum};
    }
};

__global__ void fused_diff_dangling_kernel(
    const double* __restrict__ pr_new,
    const double* __restrict__ pr_old,
    const uint32_t* __restrict__ dangling_mask,
    double* __restrict__ d_results,
    int32_t N)
{
    typedef cub::BlockReduce<DiffDangState, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    DiffDangState local = {0.0, 0.0};

    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < N; i += blockDim.x * gridDim.x) {
        double pn = pr_new[i];
        double po = pr_old[i];
        local.diff_sum += fabs(pn - po);

        uint32_t mask_word = dangling_mask[i >> 5];
        if ((mask_word >> (i & 31)) & 1) {
            local.dang_sum += pn;
        }
    }

    DiffDangState block_result = BlockReduce(temp_storage).Reduce(local, DiffDangReducer());

    if (threadIdx.x == 0) {
        if (block_result.diff_sum != 0.0) atomicAdd(&d_results[0], block_result.diff_sum);
        if (block_result.dang_sum != 0.0) atomicAdd(&d_results[1], block_result.dang_sum);
    }
}



static void launch_compute_out_weights(
    const int32_t* indices, const double* weights, const uint32_t* edge_mask,
    double* out_weight, int32_t num_edges, cudaStream_t stream)
{
    int grid = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 4096) grid = 4096;
    if (grid < 1) grid = 1;
    compute_out_weights_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        indices, weights, edge_mask, out_weight, num_edges);
}

static void launch_compute_normalized_values(
    const int32_t* indices, const double* weights, const uint32_t* edge_mask,
    const double* out_weight, double* norm_values, int32_t num_edges, cudaStream_t stream)
{
    int grid = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 4096) grid = 4096;
    if (grid < 1) grid = 1;
    compute_normalized_values_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        indices, weights, edge_mask, out_weight, norm_values, num_edges);
}

static void launch_compute_dangling_mask(
    const double* out_weight, uint32_t* dangling_mask, int32_t N, cudaStream_t stream)
{
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 4096) grid = 4096;
    if (grid < 1) grid = 1;
    compute_dangling_mask_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        out_weight, dangling_mask, N);
}

static void launch_init_pr_uniform(double* pr, int32_t N, cudaStream_t stream)
{
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 4096) grid = 4096;
    if (grid < 1) grid = 1;
    init_pr_uniform_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(pr, N);
}

static void launch_dangling_sum(
    const double* pr, const uint32_t* dangling_mask,
    double* d_result, int32_t N, cudaStream_t stream)
{
    int grid = (N + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);
    if (grid < 1) grid = 1;
    if (grid > 1024) grid = 1024;
    dangling_sum_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        pr, dangling_mask, d_result, N);
}

static void launch_apply_pers(
    double* pr_new, const int32_t* pers_vertices, const double* pers_norm_vals,
    int32_t pers_size, const double* d_dangling_sum, double alpha, cudaStream_t stream)
{
    if (pers_size > 0) {
        int grid = (pers_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        apply_pers_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            pr_new, pers_vertices, pers_norm_vals, pers_size, d_dangling_sum, alpha);
    }
}

static void launch_fused_diff_dangling(
    const double* pr_new, const double* pr_old,
    const uint32_t* dangling_mask,
    double* d_results,
    int32_t N, cudaStream_t stream)
{
    int grid = (N + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);
    if (grid < 1) grid = 1;
    if (grid > 1024) grid = 1024;
    fused_diff_dangling_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        pr_new, pr_old, dangling_mask, d_results, N);
}



struct Cache : Cacheable {
    cusparseHandle_t cusparse_h = nullptr;
    cudaStream_t stream_aux = 0;
    cudaEvent_t event_dsum = nullptr;

    
    double* alpha_dev = nullptr;
    double* beta_dev = nullptr;
    double* results = nullptr;    
    double* dsum_cur = nullptr;   

    
    double* out_weight = nullptr;
    int64_t out_weight_cap = 0;

    double* norm_values = nullptr;
    int64_t norm_values_cap = 0;

    uint32_t* dangling_mask = nullptr;
    int64_t dangling_mask_cap = 0;

    double* pers_norm = nullptr;
    int64_t pers_norm_cap = 0;

    double* pr_b = nullptr;
    int64_t pr_b_cap = 0;

    void* spmv_buf = nullptr;
    int64_t spmv_buf_cap = 0;

    Cache() {
        cusparseCreate(&cusparse_h);
        cudaStreamCreate(&stream_aux);
        cudaEventCreateWithFlags(&event_dsum, cudaEventDisableTiming);
        cudaMalloc(&alpha_dev, sizeof(double));
        cudaMalloc(&beta_dev, sizeof(double));
        cudaMalloc(&results, 2 * sizeof(double));
        cudaMalloc(&dsum_cur, sizeof(double));
    }

    ~Cache() override {
        if (cusparse_h) cusparseDestroy(cusparse_h);
        if (stream_aux) cudaStreamDestroy(stream_aux);
        if (event_dsum) cudaEventDestroy(event_dsum);
        if (alpha_dev) cudaFree(alpha_dev);
        if (beta_dev) cudaFree(beta_dev);
        if (results) cudaFree(results);
        if (dsum_cur) cudaFree(dsum_cur);
        if (out_weight) cudaFree(out_weight);
        if (norm_values) cudaFree(norm_values);
        if (dangling_mask) cudaFree(dangling_mask);
        if (pers_norm) cudaFree(pers_norm);
        if (pr_b) cudaFree(pr_b);
        if (spmv_buf) cudaFree(spmv_buf);
    }

    void ensure_out_weight(int64_t n) {
        if (out_weight_cap < n) {
            if (out_weight) cudaFree(out_weight);
            cudaMalloc(&out_weight, n * sizeof(double));
            out_weight_cap = n;
        }
    }

    void ensure_norm_values(int64_t n) {
        if (norm_values_cap < n) {
            if (norm_values) cudaFree(norm_values);
            cudaMalloc(&norm_values, n * sizeof(double));
            norm_values_cap = n;
        }
    }

    void ensure_dangling_mask(int64_t n) {
        if (dangling_mask_cap < n) {
            if (dangling_mask) cudaFree(dangling_mask);
            cudaMalloc(&dangling_mask, n * sizeof(uint32_t));
            dangling_mask_cap = n;
        }
    }

    void ensure_pers_norm(int64_t n) {
        if (pers_norm_cap < n) {
            if (pers_norm) cudaFree(pers_norm);
            cudaMalloc(&pers_norm, n * sizeof(double));
            pers_norm_cap = n;
        }
    }

    void ensure_pr_b(int64_t n) {
        if (pr_b_cap < n) {
            if (pr_b) cudaFree(pr_b);
            cudaMalloc(&pr_b, n * sizeof(double));
            pr_b_cap = n;
        }
    }

    void ensure_spmv_buf(int64_t n) {
        if (spmv_buf_cap < n) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, n);
            spmv_buf_cap = n;
        }
    }
};

}  

PageRankResult personalized_pagerank_mask(const graph32_t& graph,
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

    (void)precomputed_vertex_out_weight_sums;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t N = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cudaStream_t stream_main = 0;
    CUSPARSE_CHECK(cusparseSetStream(cache.cusparse_h, stream_main));

    
    cache.ensure_out_weight((int64_t)N);
    cache.ensure_norm_values(std::max((int64_t)E, (int64_t)1));
    int32_t mask_words = (N + 31) / 32;
    cache.ensure_dangling_mask((int64_t)mask_words);
    cache.ensure_pers_norm(std::max((int64_t)personalization_size, (int64_t)1));
    cache.ensure_pr_b((int64_t)N);

    
    cudaMemsetAsync(cache.out_weight, 0, N * sizeof(double), stream_main);
    launch_compute_out_weights(d_indices, edge_weights, d_edge_mask,
        cache.out_weight, E, stream_main);

    if (E > 0) {
        launch_compute_normalized_values(d_indices, edge_weights, d_edge_mask,
            cache.out_weight, cache.norm_values, E, stream_main);
    }

    cudaMemsetAsync(cache.dangling_mask, 0, mask_words * sizeof(uint32_t), stream_main);
    launch_compute_dangling_mask(cache.out_weight, cache.dangling_mask, N, stream_main);

    
    {
        std::vector<double> h_pv(personalization_size);
        if (personalization_size > 0) {
            cudaMemcpyAsync(h_pv.data(), personalization_values,
                personalization_size * sizeof(double),
                cudaMemcpyDeviceToHost, stream_main);
            cudaStreamSynchronize(stream_main);
        }
        double ps = 0.0;
        for (std::size_t i = 0; i < personalization_size; i++) ps += h_pv[i];
        double inv = (ps > 0.0) ? 1.0 / ps : 0.0;
        std::vector<double> h_pn(personalization_size);
        for (std::size_t i = 0; i < personalization_size; i++) h_pn[i] = h_pv[i] * inv;
        if (personalization_size > 0) {
            cudaMemcpyAsync(cache.pers_norm, h_pn.data(),
                personalization_size * sizeof(double),
                cudaMemcpyHostToDevice, stream_main);
        }
    }

    
    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(pageranks, initial_pageranks,
            N * sizeof(double), cudaMemcpyDeviceToDevice, stream_main);
    } else {
        launch_init_pr_uniform(pageranks, N, stream_main);
    }

    
    cusparseSpMatDescr_t matA;
    CUSPARSE_CHECK(cusparseCreateCsr(&matA, N, N, E,
        const_cast<int32_t*>(d_offsets), const_cast<int32_t*>(d_indices),
        cache.norm_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseDnVecDescr_t vecX, vecY;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, N, pageranks, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, N, cache.pr_b, CUDA_R_64F));

    {
        double ha = alpha, hb = 0.0;
        cudaMemcpyAsync(cache.alpha_dev, &ha, sizeof(double), cudaMemcpyHostToDevice, stream_main);
        cudaMemcpyAsync(cache.beta_dev, &hb, sizeof(double), cudaMemcpyHostToDevice, stream_main);
    }

    CUSPARSE_CHECK(cusparseSetPointerMode(cache.cusparse_h, CUSPARSE_POINTER_MODE_HOST));
    size_t bufferSize = 0;
    {
        double ha = alpha, hb = 0.0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(cache.cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &ha, matA, vecX, &hb, vecY,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
    }

    cache.ensure_spmv_buf(std::max((int64_t)bufferSize, (int64_t)1));

    {
        double ha = alpha, hb = 0.0;
        CUSPARSE_CHECK(cusparseSpMV_preprocess(cache.cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &ha, matA, vecX, &hb, vecY,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf));
    }

    CUSPARSE_CHECK(cusparseSetPointerMode(cache.cusparse_h, CUSPARSE_POINTER_MODE_DEVICE));

    
    double* d_diff = &cache.results[0];
    double* d_dsum = &cache.results[1];

    double* d_pr_cur = pageranks;
    double* d_pr_new = cache.pr_b;

    
    cudaMemsetAsync(cache.dsum_cur, 0, sizeof(double), stream_main);
    launch_dangling_sum(d_pr_cur, cache.dangling_mask, cache.dsum_cur, N, stream_main);

    bool converged = false;
    size_t iterations = 0;
    double* d_result_ptr = d_pr_cur;

    
    cudaEvent_t event_init;
    cudaEventCreateWithFlags(&event_init, cudaEventDisableTiming);
    cudaEventRecord(event_init, stream_main);
    cudaStreamWaitEvent(cache.stream_aux, event_init, 0);
    cudaEventDestroy(event_init);

    for (size_t iter = 0; iter < max_iterations; iter++) {
        
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecX, d_pr_cur));
        CUSPARSE_CHECK(cusparseDnVecSetValues(vecY, d_pr_new));
        CUSPARSE_CHECK(cusparseSpMV(cache.cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.alpha_dev, matA, vecX, cache.beta_dev, vecY,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf));

        
        launch_apply_pers(d_pr_new, personalization_vertices, cache.pers_norm,
            (int32_t)personalization_size, cache.dsum_cur, alpha, stream_main);

        
        cudaMemsetAsync(cache.results, 0, 2 * sizeof(double), stream_main);
        launch_fused_diff_dangling(d_pr_new, d_pr_cur, cache.dangling_mask,
            cache.results, N, stream_main);

        d_result_ptr = d_pr_new;
        iterations = iter + 1;

        
        cudaMemcpyAsync(cache.dsum_cur, d_dsum, sizeof(double),
            cudaMemcpyDeviceToDevice, stream_main);

        
        double h_diff;
        cudaMemcpyAsync(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream_main);
        cudaStreamSynchronize(stream_main);

        if (h_diff < epsilon) {
            converged = true;
            break;
        }

        std::swap(d_pr_cur, d_pr_new);
    }

    
    if (d_result_ptr != pageranks) {
        cudaMemcpyAsync(pageranks, d_result_ptr, N * sizeof(double),
            cudaMemcpyDeviceToDevice, stream_main);
        cudaStreamSynchronize(stream_main);
    }

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    return {iterations, converged};
}

}  
