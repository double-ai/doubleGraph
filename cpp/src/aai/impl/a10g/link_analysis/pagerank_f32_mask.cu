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

namespace aai {

namespace {




__global__ void compute_out_weights_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ out_weight_sums,
    int32_t num_edges)
{
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    uint32_t word = __ldg(&edge_mask[e >> 5]);
    if ((word >> (e & 31)) & 1) {
        atomicAdd(&out_weight_sums[__ldg(&indices[e])], __ldg(&edge_weights[e]));
    }
}

__global__ void compute_effective_weights_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ out_weight_sums,
    float* __restrict__ effective_weights,
    int32_t num_edges)
{
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    uint32_t word = __ldg(&edge_mask[e >> 5]);
    if ((word >> (e & 31)) & 1) {
        int32_t src = __ldg(&indices[e]);
        float ow = __ldg(&out_weight_sums[src]);
        effective_weights[e] = (ow > 0.0f) ? (__ldg(&edge_weights[e]) / ow) : 0.0f;
    } else {
        effective_weights[e] = 0.0f;
    }
}


__global__ void find_dangling_kernel(
    const float* __restrict__ out_weight_sums,
    int32_t* __restrict__ dangling_indices,
    int32_t* __restrict__ dangling_count,
    int32_t num_vertices)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices && __ldg(&out_weight_sums[v]) == 0.0f) {
        int32_t pos = atomicAdd(dangling_count, 1);
        dangling_indices[pos] = v;
    }
}




__global__ void fast_dangling_sum_kernel(
    const float* __restrict__ pr,
    const int32_t* __restrict__ dangling_indices,
    float* __restrict__ dangling_sum_ptr,
    int32_t num_dangling)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float my_val = 0.0f;
    if (i < num_dangling) {
        my_val = __ldg(&pr[__ldg(&dangling_indices[i])]);
    }

    float block_sum = BlockReduce(temp_storage).Sum(my_val);
    if (threadIdx.x == 0) atomicAdd(dangling_sum_ptr, block_sum);
}




__global__ void update_and_diff_kernel(
    const float* __restrict__ spmv_result,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    const float* __restrict__ dangling_sum_ptr,
    float* __restrict__ l1_diff_ptr,
    float alpha,
    float one_minus_alpha_over_n,
    float inv_n,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float alpha_dangling_over_n = alpha * (*dangling_sum_ptr) * inv_n;

    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    float my_diff = 0.0f;

    if (v < num_vertices) {
        float old_val = __ldg(&pr_old[v]);
        float new_val = one_minus_alpha_over_n + alpha * __ldg(&spmv_result[v]) + alpha_dangling_over_n;
        pr_new[v] = new_val;
        my_diff = fabsf(new_val - old_val);
    }

    float block_diff = BlockReduce(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0) atomicAdd(l1_diff_ptr, block_diff);
}




__global__ void fused_spmv_update_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ eff_weights,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    const float* __restrict__ dangling_sum_ptr,
    float* __restrict__ l1_diff_ptr,
    float alpha,
    float one_minus_alpha_over_n,
    float inv_n,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float alpha_dangling_over_n = alpha * (*dangling_sum_ptr) * inv_n;

    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    float my_diff = 0.0f;

    if (v < num_vertices) {
        int32_t start = __ldg(&offsets[v]);
        int32_t end = __ldg(&offsets[v + 1]);
        float spmv = 0.0f;

        for (int32_t e = start; e < end; e++) {
            int32_t src = __ldg(&indices[e]);
            spmv += __ldg(&eff_weights[e]) * __ldg(&pr_old[src]);
        }

        float old_val = __ldg(&pr_old[v]);
        float new_val = one_minus_alpha_over_n + alpha * spmv + alpha_dangling_over_n;
        pr_new[v] = new_val;
        my_diff = fabsf(new_val - old_val);
    }

    float block_diff = BlockReduce(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0) atomicAdd(l1_diff_ptr, block_diff);
}

__global__ void fill_kernel(float* arr, int32_t n, float val)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = val;
}




void launch_compute_out_weights(
    const int32_t* indices, const float* edge_weights,
    const uint32_t* edge_mask, float* out_weight_sums,
    int32_t num_vertices, int32_t num_edges, cudaStream_t stream)
{
    cudaMemsetAsync(out_weight_sums, 0, num_vertices * sizeof(float), stream);
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    compute_out_weights_kernel<<<grid, block, 0, stream>>>(
        indices, edge_weights, edge_mask, out_weight_sums, num_edges);
}

void launch_compute_effective_weights(
    const int32_t* indices, const float* edge_weights,
    const uint32_t* edge_mask, const float* out_weight_sums,
    float* effective_weights, int32_t num_edges, cudaStream_t stream)
{
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    compute_effective_weights_kernel<<<grid, block, 0, stream>>>(
        indices, edge_weights, edge_mask, out_weight_sums,
        effective_weights, num_edges);
}

void launch_find_dangling(
    const float* out_weight_sums,
    int32_t* dangling_indices, int32_t* dangling_count,
    int32_t num_vertices, cudaStream_t stream)
{
    cudaMemsetAsync(dangling_count, 0, sizeof(int32_t), stream);
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    find_dangling_kernel<<<grid, block, 0, stream>>>(
        out_weight_sums, dangling_indices, dangling_count, num_vertices);
}

void launch_fast_dangling_sum(
    const float* pr, const int32_t* dangling_indices,
    float* dangling_sum_ptr, int32_t num_dangling, cudaStream_t stream)
{
    cudaMemsetAsync(dangling_sum_ptr, 0, sizeof(float), stream);
    if (num_dangling == 0) return;
    int block = 256;
    int grid = (num_dangling + block - 1) / block;
    fast_dangling_sum_kernel<<<grid, block, 0, stream>>>(
        pr, dangling_indices, dangling_sum_ptr, num_dangling);
}

void launch_update_and_diff(
    const float* spmv_result, const float* pr_old,
    float* pr_new, const float* dangling_sum_ptr,
    float* l1_diff_ptr, float alpha,
    float one_minus_alpha_over_n, float inv_n,
    int32_t num_vertices, cudaStream_t stream)
{
    cudaMemsetAsync(l1_diff_ptr, 0, sizeof(float), stream);
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    update_and_diff_kernel<<<grid, block, 0, stream>>>(
        spmv_result, pr_old, pr_new, dangling_sum_ptr, l1_diff_ptr,
        alpha, one_minus_alpha_over_n, inv_n, num_vertices);
}

void launch_fused_spmv_update(
    const int32_t* offsets, const int32_t* indices,
    const float* eff_weights, const float* pr_old,
    float* pr_new, const float* dangling_sum_ptr,
    float* l1_diff_ptr, float alpha,
    float one_minus_alpha_over_n, float inv_n,
    int32_t num_vertices, cudaStream_t stream)
{
    cudaMemsetAsync(l1_diff_ptr, 0, sizeof(float), stream);
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    fused_spmv_update_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, eff_weights, pr_old, pr_new,
        dangling_sum_ptr, l1_diff_ptr, alpha,
        one_minus_alpha_over_n, inv_n, num_vertices);
}

void launch_fill(float* arr, int32_t n, float val, cudaStream_t stream)
{
    int block = 256;
    int grid = (n + block - 1) / block;
    fill_kernel<<<grid, block, 0, stream>>>(arr, n, val);
}




struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    float* h_pinned = nullptr;

    float* out_weight_sums = nullptr;
    int64_t ows_cap = 0;

    float* eff_weights = nullptr;
    int64_t ew_cap = 0;

    int32_t* dangling_indices = nullptr;
    int64_t di_cap = 0;

    int32_t* dangling_count = nullptr;

    float* pr_new = nullptr;
    int64_t prn_cap = 0;

    float* scalar_buf = nullptr;

    float* spmv_result = nullptr;
    int64_t sr_cap = 0;

    void* spmv_buffer = nullptr;
    size_t sb_cap = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cudaMallocHost(&h_pinned, sizeof(float));
        cudaMalloc(&dangling_count, sizeof(int32_t));
        cudaMalloc(&scalar_buf, 4 * sizeof(float));
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (h_pinned) cudaFreeHost(h_pinned);
        if (out_weight_sums) cudaFree(out_weight_sums);
        if (eff_weights) cudaFree(eff_weights);
        if (dangling_indices) cudaFree(dangling_indices);
        if (dangling_count) cudaFree(dangling_count);
        if (pr_new) cudaFree(pr_new);
        if (scalar_buf) cudaFree(scalar_buf);
        if (spmv_result) cudaFree(spmv_result);
        if (spmv_buffer) cudaFree(spmv_buffer);
    }

    void ensure(int32_t nv, int32_t ne) {
        if (ows_cap < nv) {
            if (out_weight_sums) cudaFree(out_weight_sums);
            cudaMalloc(&out_weight_sums, (size_t)nv * sizeof(float));
            ows_cap = nv;
        }
        if (di_cap < nv) {
            if (dangling_indices) cudaFree(dangling_indices);
            cudaMalloc(&dangling_indices, (size_t)nv * sizeof(int32_t));
            di_cap = nv;
        }
        if (prn_cap < nv) {
            if (pr_new) cudaFree(pr_new);
            cudaMalloc(&pr_new, (size_t)nv * sizeof(float));
            prn_cap = nv;
        }
        if (sr_cap < nv) {
            if (spmv_result) cudaFree(spmv_result);
            cudaMalloc(&spmv_result, (size_t)nv * sizeof(float));
            sr_cap = nv;
        }
        if (ew_cap < ne) {
            if (eff_weights) cudaFree(eff_weights);
            cudaMalloc(&eff_weights, (size_t)ne * sizeof(float));
            ew_cap = ne;
        }
    }

    void ensure_spmv_buf(size_t size) {
        if (sb_cap < size) {
            if (spmv_buffer) cudaFree(spmv_buffer);
            cudaMalloc(&spmv_buffer, size);
            sb_cap = size;
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
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cache.ensure(num_vertices, num_edges);

    cudaStream_t stream = 0;

    
    launch_compute_out_weights(d_indices, edge_weights, d_edge_mask,
                               cache.out_weight_sums, num_vertices, num_edges, stream);
    const float* d_out_weight_sums = cache.out_weight_sums;

    
    launch_compute_effective_weights(d_indices, edge_weights, d_edge_mask,
                                    d_out_weight_sums, cache.eff_weights, num_edges, stream);

    
    launch_find_dangling(d_out_weight_sums, cache.dangling_indices, cache.dangling_count,
                        num_vertices, stream);

    int32_t h_num_dangling;
    cudaMemcpyAsync(&h_num_dangling, cache.dangling_count, sizeof(int32_t),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    
    float* d_pageranks = pageranks;
    float* d_pr_new = cache.pr_new;
    float* d_dangling_sum = cache.scalar_buf;
    float* d_l1_diff = cache.scalar_buf + 1;

    
    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(d_pageranks, initial_pageranks,
                       num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        launch_fill(d_pageranks, num_vertices, 1.0f / num_vertices, stream);
    }

    float one_minus_alpha_over_n = (1.0f - alpha) / num_vertices;
    float inv_n = 1.0f / num_vertices;

    size_t iter = 0;
    bool converged = false;

    
    float avg_degree = (float)num_edges / (float)num_vertices;
    bool use_cusparse = (avg_degree > 4.0f) && (num_edges > 50000);

    if (!use_cusparse) {
        
        for (; iter < max_iterations; iter++) {
            if (h_num_dangling > 0) {
                launch_fast_dangling_sum(d_pageranks, cache.dangling_indices,
                                       d_dangling_sum, h_num_dangling, stream);
            } else {
                cudaMemsetAsync(d_dangling_sum, 0, sizeof(float), stream);
            }
            launch_fused_spmv_update(
                d_offsets, d_indices, cache.eff_weights, d_pageranks,
                d_pr_new, d_dangling_sum, d_l1_diff,
                alpha, one_minus_alpha_over_n, inv_n,
                num_vertices, stream);

            cudaMemcpyAsync(cache.h_pinned, d_l1_diff, sizeof(float),
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            float* temp = d_pageranks; d_pageranks = d_pr_new; d_pr_new = temp;
            if (*cache.h_pinned < epsilon) { converged = true; iter++; break; }
        }
    } else {
        
        cusparseSetStream(cache.cusparse_handle, stream);

        float* d_alpha_dev = cache.scalar_buf + 2;
        float* d_beta_dev = cache.scalar_buf + 3;
        float h_vals[2] = {1.0f, 0.0f};
        cudaMemcpyAsync(d_alpha_dev, &h_vals[0], sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_beta_dev, &h_vals[1], sizeof(float), cudaMemcpyHostToDevice, stream);

        float* d_spmv = cache.spmv_result;

        cusparseSpMatDescr_t mat_descr;
        cusparseCreateCsr(
            &mat_descr, num_vertices, num_vertices, num_edges,
            (void*)d_offsets, (void*)d_indices, (void*)cache.eff_weights,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        cusparseDnVecDescr_t x_descr, y_descr;
        cusparseCreateDnVec(&x_descr, num_vertices, d_pageranks, CUDA_R_32F);
        cusparseCreateDnVec(&y_descr, num_vertices, d_spmv, CUDA_R_32F);

        float h_one = 1.0f, h_zero = 0.0f;
        size_t spmv_buffer_size = 0;
        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
        cusparseSpMV_bufferSize(
            cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_one, mat_descr, x_descr, &h_zero, y_descr,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buffer_size);

        if (spmv_buffer_size > 0) {
            cache.ensure_spmv_buf(spmv_buffer_size);
        }

        if (num_edges > 1000000) {
            cusparseSpMV_preprocess(
                cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &h_one, mat_descr, x_descr, &h_zero, y_descr,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buffer);
        }

        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);

        for (; iter < max_iterations; iter++) {
            if (h_num_dangling > 0) {
                launch_fast_dangling_sum(d_pageranks, cache.dangling_indices,
                                       d_dangling_sum, h_num_dangling, stream);
            } else {
                cudaMemsetAsync(d_dangling_sum, 0, sizeof(float), stream);
            }

            cusparseDnVecSetValues(x_descr, d_pageranks);
            cusparseSpMV(
                cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                d_alpha_dev, mat_descr, x_descr, d_beta_dev, y_descr,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buffer);

            launch_update_and_diff(d_spmv, d_pageranks, d_pr_new,
                                   d_dangling_sum, d_l1_diff,
                                   alpha, one_minus_alpha_over_n, inv_n,
                                   num_vertices, stream);

            cudaMemcpyAsync(cache.h_pinned, d_l1_diff, sizeof(float),
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            float* temp = d_pageranks; d_pageranks = d_pr_new; d_pr_new = temp;
            if (*cache.h_pinned < epsilon) { converged = true; iter++; break; }
        }

        cusparseDestroyDnVec(x_descr);
        cusparseDestroyDnVec(y_descr);
        cusparseDestroySpMat(mat_descr);
    }

    
    if (d_pageranks != pageranks) {
        cudaMemcpyAsync(pageranks, d_pageranks,
                       num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    return PageRankResult{iter, converged};
}

}  
