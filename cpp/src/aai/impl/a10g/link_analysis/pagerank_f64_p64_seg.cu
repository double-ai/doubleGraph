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
#include <cstddef>
#include <algorithm>
#include <vector>

namespace aai {

namespace {

constexpr int BLOCK_SIZE = 256;


struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    cudaStream_t stream_main = 0;
    cudaStream_t stream_aux = 0;
    cudaEvent_t event_dang = nullptr;
    cudaEvent_t event_spmv = nullptr;
    float* h_diff_pinned = nullptr;

    
    double* d_out_weights = nullptr;
    float* d_scaled_w = nullptr;
    uint8_t* d_dang_mask = nullptr;
    float* d_pers_norm_c = nullptr;
    float* d_pr_a = nullptr;
    float* d_pr_b = nullptr;
    float* d_dang_sum = nullptr;
    float* d_diff_sum = nullptr;
    void* d_spmv_buf = nullptr;
    void* d_cub_temp = nullptr;
    double* d_ps_dev = nullptr;

    
    int64_t out_weights_cap = 0;
    int64_t scaled_w_cap = 0;
    int64_t dang_mask_cap = 0;
    int64_t pers_norm_c_cap = 0;
    int64_t pr_a_cap = 0;
    int64_t pr_b_cap = 0;
    bool dang_sum_allocated = false;
    bool diff_sum_allocated = false;
    size_t spmv_buf_cap = 0;
    size_t cub_temp_cap = 0;
    bool ps_dev_allocated = false;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cudaStreamCreate(&stream_main);
        cudaStreamCreate(&stream_aux);
        cudaEventCreateWithFlags(&event_dang, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&event_spmv, cudaEventDisableTiming);
        cusparseSetStream(cusparse_handle, stream_main);
        cudaHostAlloc(&h_diff_pinned, sizeof(float), cudaHostAllocDefault);
    }

    ~Cache() override {
        if (h_diff_pinned) cudaFreeHost(h_diff_pinned);
        if (event_dang) cudaEventDestroy(event_dang);
        if (event_spmv) cudaEventDestroy(event_spmv);
        if (stream_aux) cudaStreamDestroy(stream_aux);
        if (stream_main) cudaStreamDestroy(stream_main);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);

        if (d_out_weights) cudaFree(d_out_weights);
        if (d_scaled_w) cudaFree(d_scaled_w);
        if (d_dang_mask) cudaFree(d_dang_mask);
        if (d_pers_norm_c) cudaFree(d_pers_norm_c);
        if (d_pr_a) cudaFree(d_pr_a);
        if (d_pr_b) cudaFree(d_pr_b);
        if (d_dang_sum) cudaFree(d_dang_sum);
        if (d_diff_sum) cudaFree(d_diff_sum);
        if (d_spmv_buf) cudaFree(d_spmv_buf);
        if (d_cub_temp) cudaFree(d_cub_temp);
        if (d_ps_dev) cudaFree(d_ps_dev);
    }

    void ensure(int32_t num_vertices, int32_t num_edges, int pers_size) {
        if (out_weights_cap < num_vertices) {
            if (d_out_weights) cudaFree(d_out_weights);
            cudaMalloc(&d_out_weights, (size_t)num_vertices * sizeof(double));
            out_weights_cap = num_vertices;
        }
        if (scaled_w_cap < num_edges) {
            if (d_scaled_w) cudaFree(d_scaled_w);
            cudaMalloc(&d_scaled_w, (size_t)num_edges * sizeof(float));
            scaled_w_cap = num_edges;
        }
        if (dang_mask_cap < num_vertices) {
            if (d_dang_mask) cudaFree(d_dang_mask);
            cudaMalloc(&d_dang_mask, (size_t)num_vertices * sizeof(uint8_t));
            dang_mask_cap = num_vertices;
        }
        int ps = pers_size > 0 ? pers_size : 1;
        if (pers_norm_c_cap < ps) {
            if (d_pers_norm_c) cudaFree(d_pers_norm_c);
            cudaMalloc(&d_pers_norm_c, (size_t)ps * sizeof(float));
            pers_norm_c_cap = ps;
        }
        if (pr_a_cap < num_vertices) {
            if (d_pr_a) cudaFree(d_pr_a);
            cudaMalloc(&d_pr_a, (size_t)num_vertices * sizeof(float));
            pr_a_cap = num_vertices;
        }
        if (pr_b_cap < num_vertices) {
            if (d_pr_b) cudaFree(d_pr_b);
            cudaMalloc(&d_pr_b, (size_t)num_vertices * sizeof(float));
            pr_b_cap = num_vertices;
        }
        if (!dang_sum_allocated) {
            cudaMalloc(&d_dang_sum, sizeof(float));
            dang_sum_allocated = true;
        }
        if (!diff_sum_allocated) {
            cudaMalloc(&d_diff_sum, sizeof(float));
            diff_sum_allocated = true;
        }
        if (!ps_dev_allocated) {
            cudaMalloc(&d_ps_dev, sizeof(double));
            ps_dev_allocated = true;
        }
    }

    void ensure_spmv_buf(size_t size) {
        size_t needed = size > 0 ? size : 1;
        if (spmv_buf_cap < needed) {
            if (d_spmv_buf) cudaFree(d_spmv_buf);
            cudaMalloc(&d_spmv_buf, needed);
            spmv_buf_cap = needed;
        }
    }

    void ensure_cub_temp(size_t size) {
        if (cub_temp_cap < size) {
            if (d_cub_temp) cudaFree(d_cub_temp);
            cudaMalloc(&d_cub_temp, size);
            cub_temp_cap = size;
        }
    }
};


__global__ void zero_double_kernel(double* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = 0.0;
}

__global__ void compute_out_weights_kernel(
    const int* __restrict__ indices,
    const double* __restrict__ edge_weights,
    double* __restrict__ out_weights,
    int num_edges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges) {
        atomicAdd(&out_weights[indices[idx]], edge_weights[idx]);
    }
}

__global__ void prescale_weights_kernel(
    const int* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const double* __restrict__ out_weights,
    float* __restrict__ scaled_weights,
    double alpha,
    int num_edges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges) {
        int src = indices[idx];
        double ow = out_weights[src];
        double sw = (ow > 0.0) ? (alpha * edge_weights[idx] / ow) : 0.0;
        scaled_weights[idx] = (float)sw;
    }
}

__global__ void build_dangling_mask_kernel(
    const double* __restrict__ out_weights,
    uint8_t* __restrict__ dangling_mask,
    int num_vertices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        dangling_mask[idx] = (out_weights[idx] == 0.0) ? 1 : 0;
    }
}

__global__ void build_pers_norm_compact_kernel(
    const double* __restrict__ pers_values,
    float* __restrict__ pers_norm_compact,
    double pers_sum_inv,
    int pers_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pers_size) {
        pers_norm_compact[idx] = (float)(pers_values[idx] * pers_sum_inv);
    }
}


__global__ void init_uniform_kernel(float* pr, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) pr[idx] = val;
}

__global__ void copy_f64_to_f32_kernel(const double* __restrict__ src, float* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = (float)src[idx];
}

__global__ void copy_f32_to_f64_kernel(const float* __restrict__ src, double* __restrict__ dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = (double)src[idx];
}



__global__ void dangling_sum_kernel(
    const float* __restrict__ pr,
    const uint8_t* __restrict__ dangling_mask,
    float* __restrict__ result,
    int n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        if (dangling_mask[idx]) sum += pr[idx];
    }

    float block_sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(result, block_sum);
    }
}

__global__ void scatter_add_pers_kernel(
    float* __restrict__ new_pr,
    const int* __restrict__ pers_vertices,
    const float* __restrict__ pers_norm_compact,
    const float* __restrict__ d_dangling_sum,
    float alpha,
    float one_minus_alpha,
    int pers_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pers_size) {
        float base_factor = alpha * d_dangling_sum[0] + one_minus_alpha;
        new_pr[pers_vertices[idx]] += base_factor * pers_norm_compact[idx];
    }
}

__global__ void diff_reduction_kernel(
    const float* __restrict__ new_pr,
    const float* __restrict__ old_pr,
    float* __restrict__ result,
    int n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        sum += fabsf(new_pr[idx] - old_pr[idx]);
    }

    float block_sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(result, block_sum);
    }
}

}  

PageRankResult personalized_pagerank_seg(const graph32_t& graph,
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

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int pers_size = (int)personalization_size;
    bool has_initial_guess = (initial_pageranks != nullptr);

    int num_blocks = (num_vertices + 255) / 256;
    if (num_blocks > 1024) num_blocks = 1024;
    if (num_blocks < 1) num_blocks = 1;

    cache.ensure(num_vertices, num_edges, pers_size);

    double* d_out_weights = cache.d_out_weights;
    float* d_scaled_w = cache.d_scaled_w;
    uint8_t* d_dang_mask = cache.d_dang_mask;
    float* d_pers_norm_c = cache.d_pers_norm_c;
    float* d_pr_a = cache.d_pr_a;
    float* d_pr_b = cache.d_pr_b;
    float* d_dang_sum = cache.d_dang_sum;
    float* d_diff_sum = cache.d_diff_sum;
    cudaStream_t stream_main = cache.stream_main;
    cudaStream_t stream_aux = cache.stream_aux;

    
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        zero_double_kernel<<<grid, block, 0, stream_main>>>(d_out_weights, num_vertices);
        grid = (num_edges + block - 1) / block;
        if (grid > 0)
            compute_out_weights_kernel<<<grid, block, 0, stream_main>>>(d_indices, edge_weights, d_out_weights, num_edges);
    }

    {
        int block = 256;
        int grid = (num_edges + block - 1) / block;
        if (grid > 0)
            prescale_weights_kernel<<<grid, block, 0, stream_main>>>(d_indices, edge_weights, d_out_weights, d_scaled_w, alpha, num_edges);
    }

    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        build_dangling_mask_kernel<<<grid, block, 0, stream_main>>>(d_out_weights, d_dang_mask, num_vertices);
    }

    
    double pers_sum = 0.0;
    if (pers_size > 0) {
        if (pers_size <= 8192) {
            cudaStreamSynchronize(stream_main);
            std::vector<double> h_pv(pers_size);
            cudaMemcpy(h_pv.data(), personalization_values, pers_size * sizeof(double), cudaMemcpyDeviceToHost);
            for (int i = 0; i < pers_size; i++) pers_sum += h_pv[i];
        } else {
            size_t cub_temp_bytes = 0;
            cub::DeviceReduce::Sum(nullptr, cub_temp_bytes, (double*)nullptr, (double*)nullptr, pers_size);
            cache.ensure_cub_temp(cub_temp_bytes);
            cub::DeviceReduce::Sum(cache.d_cub_temp, cub_temp_bytes,
                personalization_values, cache.d_ps_dev, pers_size, stream_main);
            cudaMemcpy(&pers_sum, cache.d_ps_dev, sizeof(double), cudaMemcpyDeviceToHost);
        }
    }
    double pers_sum_inv = (pers_sum > 0.0) ? (1.0 / pers_sum) : 0.0;

    if (pers_size > 0) {
        int block = 256;
        int grid = (pers_size + block - 1) / block;
        build_pers_norm_compact_kernel<<<grid, block, 0, stream_main>>>(personalization_values, d_pers_norm_c, pers_sum_inv, pers_size);
    }

    
    cusparseSpMatDescr_t mat = nullptr;
    cusparseDnVecDescr_t vec_in = nullptr, vec_out = nullptr;

    cusparseCreateCsr(&mat,
        num_vertices, num_vertices, num_edges,
        (void*)d_offsets, (void*)d_indices, (void*)d_scaled_w,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseCreateDnVec(&vec_in, num_vertices, d_pr_a, CUDA_R_32F);
    cusparseCreateDnVec(&vec_out, num_vertices, d_pr_b, CUDA_R_32F);

    float spmv_alpha = 1.0f, spmv_beta = 0.0f;
    cusparseSpMVAlg_t spmv_alg = CUSPARSE_SPMV_CSR_ALG1;

    size_t spmv_buf_size = 0;
    cusparseSpMV_bufferSize(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha, mat, vec_in, &spmv_beta, vec_out,
        CUDA_R_32F, spmv_alg, &spmv_buf_size);

    cache.ensure_spmv_buf(spmv_buf_size);

    cusparseSpMV_preprocess(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &spmv_alpha, mat, vec_in, &spmv_beta, vec_out,
        CUDA_R_32F, spmv_alg, cache.d_spmv_buf);

    
    float* d_old_pr = d_pr_a;
    float* d_new_pr = d_pr_b;

    if (has_initial_guess) {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        copy_f64_to_f32_kernel<<<grid, block, 0, stream_main>>>(initial_pageranks, d_old_pr, num_vertices);
    } else {
        float init_val = 1.0f / (float)num_vertices;
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        init_uniform_kernel<<<grid, block, 0, stream_main>>>(d_old_pr, init_val, num_vertices);
    }

    cudaEventRecord(cache.event_spmv, stream_main);
    cudaStreamWaitEvent(stream_aux, cache.event_spmv, 0);

    
    const int CHECK_INTERVAL = 4;
    float f_alpha = (float)alpha;
    float f_one_minus_alpha = (float)(1.0 - alpha);
    float f_epsilon = (float)epsilon;

    size_t iterations = 0;
    bool converged = false;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(d_dang_sum, 0, sizeof(float), stream_aux);
        dangling_sum_kernel<<<num_blocks, BLOCK_SIZE, 0, stream_aux>>>(d_old_pr, d_dang_mask, d_dang_sum, num_vertices);
        cudaEventRecord(cache.event_dang, stream_aux);

        
        cusparseDnVecSetValues(vec_in, d_old_pr);
        cusparseDnVecSetValues(vec_out, d_new_pr);
        cusparseSpMV(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmv_alpha, mat, vec_in, &spmv_beta, vec_out,
            CUDA_R_32F, spmv_alg, cache.d_spmv_buf);

        
        cudaStreamWaitEvent(stream_main, cache.event_dang, 0);

        
        if (pers_size > 0) {
            int block = 256;
            int grid = (pers_size + block - 1) / block;
            scatter_add_pers_kernel<<<grid, block, 0, stream_main>>>(
                d_new_pr, personalization_vertices, d_pers_norm_c, d_dang_sum,
                f_alpha, f_one_minus_alpha, pers_size);
        }

        iterations = iter + 1;

        
        bool check = ((iter + 1) % CHECK_INTERVAL == 0) || ((iter + 1) >= max_iterations);
        if (check) {
            cudaMemsetAsync(d_diff_sum, 0, sizeof(float), stream_main);
            diff_reduction_kernel<<<num_blocks, BLOCK_SIZE, 0, stream_main>>>(d_new_pr, d_old_pr, d_diff_sum, num_vertices);
            cudaMemcpyAsync(cache.h_diff_pinned, d_diff_sum, sizeof(float), cudaMemcpyDeviceToHost, stream_main);
            cudaStreamSynchronize(stream_main);
            if (*cache.h_diff_pinned < f_epsilon) {
                converged = true;
                break;
            }
        }

        std::swap(d_old_pr, d_new_pr);
        cudaEventRecord(cache.event_spmv, stream_main);
        cudaStreamWaitEvent(stream_aux, cache.event_spmv, 0);
    }

    
    float* result_pr = converged ? d_new_pr : d_old_pr;
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        copy_f32_to_f64_kernel<<<grid, block, 0, stream_main>>>(result_pr, pageranks, num_vertices);
    }
    cudaStreamSynchronize(stream_main);

    cusparseDestroySpMat(mat);
    cusparseDestroyDnVec(vec_in);
    cusparseDestroyDnVec(vec_out);

    return PageRankResult{iterations, converged};
}

}  
