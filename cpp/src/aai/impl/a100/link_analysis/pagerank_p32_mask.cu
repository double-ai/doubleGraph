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
#include <cub/cub.cuh>
#include <cusparse.h>
#include <cstdint>

namespace aai {

namespace {

#define BLOCK_SIZE 256

template <typename T>
void ensure_buf(T*& ptr, int64_t& cap, int64_t needed) {
    if (cap < needed) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, needed * sizeof(T));
        cap = needed;
    }
}

struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;

    int32_t* new_offsets = nullptr;
    int32_t* new_indices = nullptr;
    int32_t* active_counts = nullptr;
    float* out_degree = nullptr;
    float* pers_full = nullptr;
    float* x_buf = nullptr;
    float* pr_buf = nullptr;
    float* values = nullptr;
    float* d_scalars = nullptr;

    int64_t new_offsets_cap = 0;
    int64_t new_indices_cap = 0;
    int64_t active_counts_cap = 0;
    int64_t out_degree_cap = 0;
    int64_t pers_full_cap = 0;
    int64_t x_buf_cap = 0;
    int64_t pr_buf_cap = 0;
    int64_t values_cap = 0;
    int64_t d_scalars_cap = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (new_offsets) cudaFree(new_offsets);
        if (new_indices) cudaFree(new_indices);
        if (active_counts) cudaFree(active_counts);
        if (out_degree) cudaFree(out_degree);
        if (pers_full) cudaFree(pers_full);
        if (x_buf) cudaFree(x_buf);
        if (pr_buf) cudaFree(pr_buf);
        if (values) cudaFree(values);
        if (d_scalars) cudaFree(d_scalars);
    }

    void ensure(int32_t nv, int32_t ne) {
        int64_t nv1 = static_cast<int64_t>(nv) + 1;
        int64_t ne2 = ne > 0 ? static_cast<int64_t>(ne) : 1;

        ensure_buf(new_offsets, new_offsets_cap, nv1);
        ensure_buf(new_indices, new_indices_cap, ne2);
        ensure_buf(active_counts, active_counts_cap, static_cast<int64_t>(nv));
        ensure_buf(out_degree, out_degree_cap, static_cast<int64_t>(nv));
        ensure_buf(pers_full, pers_full_cap, static_cast<int64_t>(nv));
        ensure_buf(x_buf, x_buf_cap, static_cast<int64_t>(nv));
        ensure_buf(pr_buf, pr_buf_cap, static_cast<int64_t>(nv));
        ensure_buf(values, values_cap, ne2);
        ensure_buf(d_scalars, d_scalars_cap, 4);
    }
};



__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ counts,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int start = offsets[v];
    int end = offsets[v + 1];
    int count = 0;
    for (int e = start; e < end; e++)
        count += (edge_mask[e >> 5] >> (e & 31)) & 1;
    counts[v] = count;
}

__global__ void set_last_offset_kernel(int32_t* new_offsets, const int32_t* counts, int32_t num_vertices)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        new_offsets[num_vertices] = new_offsets[num_vertices - 1] + counts[num_vertices - 1];
}

__global__ void scatter_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int start = offsets[v];
    int end = offsets[v + 1];
    int pos = new_offsets[v];
    for (int e = start; e < end; e++) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1)
            new_indices[pos++] = indices[e];
    }
}

__global__ void compute_out_degrees_kernel(
    const int32_t* __restrict__ indices,
    float* __restrict__ out_degree,
    int32_t num_active_edges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < num_active_edges)
        atomicAdd(&out_degree[indices[e]], 1.0f);
}

__global__ void scatter_pers_kernel(
    const int32_t* __restrict__ pers_vertices,
    const float* __restrict__ pers_values,
    float* __restrict__ pers_full,
    int32_t pers_size,
    float pers_sum_inv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pers_size)
        pers_full[pers_vertices[i]] = pers_values[i] * pers_sum_inv;
}

__global__ void init_pageranks_kernel(
    float* __restrict__ pageranks,
    const float* __restrict__ initial_pr,
    int32_t num_vertices,
    float default_val,
    bool has_initial)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices)
        pageranks[v] = has_initial ? initial_pr[v] : default_val;
}

__global__ void fill_ones_kernel(float* arr, int32_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = 1.0f;
}


__global__ void invert_out_degree_kernel(float* __restrict__ data, int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        float od = data[v];
        data[v] = (od > 0.0f) ? (1.0f / od) : 0.0f;
    }
}




__global__ void scale_and_dangling_kernel(
    const float* __restrict__ pageranks,
    const float* __restrict__ inv_out_degree,
    float* __restrict__ x,
    float* __restrict__ d_dangling_sum,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float dangling_val = 0.0f;

    if (v < num_vertices) {
        float inv_od = inv_out_degree[v];
        float pr = pageranks[v];
        x[v] = pr * inv_od;  
        if (inv_od == 0.0f) dangling_val = pr;
    }

    float block_sum = BlockReduce(temp_storage).Sum(dangling_val);
    if (threadIdx.x == 0 && block_sum != 0.0f)
        atomicAdd(d_dangling_sum, block_sum);
}


__global__ void teleport_diff_kernel(
    float* __restrict__ new_pr,
    const float* __restrict__ old_pr,
    const float* __restrict__ pers_full,
    const float* __restrict__ d_dangling_sum,
    float alpha,
    float one_minus_alpha,
    int32_t num_vertices,
    float* __restrict__ d_diff_sum)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float teleport = alpha * (*d_dangling_sum) + one_minus_alpha;

    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float my_diff = 0.0f;

    if (v < num_vertices) {
        float np = new_pr[v] + teleport * pers_full[v];
        new_pr[v] = np;
        my_diff = fabsf(np - old_pr[v]);
    }

    float block_sum = BlockReduce(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0 && block_sum != 0.0f)
        atomicAdd(d_diff_sum, block_sum);
}


__global__ void spmv_teleport_diff_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ new_pr,
    const float* __restrict__ old_pr,
    const float* __restrict__ pers_full,
    const float* __restrict__ d_dangling_sum,
    float alpha,
    float one_minus_alpha,
    int32_t num_vertices,
    float* __restrict__ d_diff_sum)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float teleport = alpha * (*d_dangling_sum) + one_minus_alpha;

    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float my_diff = 0.0f;

    if (v < num_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];
        float sum = 0.0f;
        for (int e = start; e < end; e++)
            sum += x[indices[e]];

        float np = alpha * sum + teleport * pers_full[v];
        new_pr[v] = np;
        my_diff = fabsf(np - old_pr[v]);
    }

    float block_sum = BlockReduce(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0 && block_sum != 0.0f)
        atomicAdd(d_diff_sum, block_sum);
}

}  

PageRankResult personalized_pagerank_mask(const graph32_t& graph,
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

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const uint32_t* edge_mask = graph.edge_mask;

    int32_t pers_size = static_cast<int32_t>(personalization_size);
    bool has_initial_guess = (initial_pageranks != nullptr);

    cache.ensure(num_vertices, num_edges);

    int32_t* new_offsets = cache.new_offsets;
    int32_t* new_indices = cache.new_indices;
    int32_t* active_counts = cache.active_counts;
    float* out_degree = cache.out_degree;
    float* pers_full = cache.pers_full;
    float* x_buf = cache.x_buf;
    float* pr_buf = cache.pr_buf;
    float* values = cache.values;
    float* d_scalars = cache.d_scalars;

    const int BLOCK = BLOCK_SIZE;
    int grid_v = (num_vertices + BLOCK - 1) / BLOCK;

    
    count_active_edges_kernel<<<grid_v, BLOCK>>>(offsets, edge_mask, active_counts, num_vertices);

    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, active_counts, new_offsets, num_vertices);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, active_counts, new_offsets, num_vertices);
    cudaFree(d_temp);

    set_last_offset_kernel<<<1, 1>>>(new_offsets, active_counts, num_vertices);

    int32_t num_active_edges;
    cudaMemcpy(&num_active_edges, &new_offsets[num_vertices], sizeof(int32_t), cudaMemcpyDeviceToHost);

    scatter_active_edges_kernel<<<grid_v, BLOCK>>>(offsets, indices, edge_mask, new_offsets, new_indices, num_vertices);

    cudaMemset(out_degree, 0, num_vertices * sizeof(float));
    if (num_active_edges > 0) {
        int grid_ae = (num_active_edges + BLOCK - 1) / BLOCK;
        compute_out_degrees_kernel<<<grid_ae, BLOCK>>>(new_indices, out_degree, num_active_edges);
        fill_ones_kernel<<<grid_ae, BLOCK>>>(values, num_active_edges);
    }
    
    invert_out_degree_kernel<<<grid_v, BLOCK>>>(out_degree, num_vertices);

    cudaMemset(pers_full, 0, num_vertices * sizeof(float));
    {
        float h_pers_sum = 0.0f;
        if (pers_size > 0) {
            float* h_pv = new float[pers_size];
            cudaMemcpy(h_pv, personalization_values, pers_size * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < pers_size; i++) h_pers_sum += h_pv[i];
            delete[] h_pv;
        }
        float pers_sum_inv = (h_pers_sum > 0.0f) ? (1.0f / h_pers_sum) : 0.0f;
        int grid_p = (pers_size + BLOCK - 1) / BLOCK;
        if (grid_p == 0) grid_p = 1;
        scatter_pers_kernel<<<grid_p, BLOCK>>>(personalization_vertices, personalization_values, pers_full, pers_size, pers_sum_inv);
    }

    float default_pr = 1.0f / num_vertices;
    init_pageranks_kernel<<<grid_v, BLOCK>>>(pageranks, initial_pageranks, num_vertices, default_pr, has_initial_guess);

    
    bool use_cusparse = (num_active_edges > 50000);

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    void* cusparse_buffer = nullptr;
    float* d_alpha_dev = &d_scalars[2];
    float* d_zero_dev = &d_scalars[3];

    if (use_cusparse) {
        
        cudaMemcpy(d_alpha_dev, &alpha, sizeof(float), cudaMemcpyHostToDevice);
        float zero = 0.0f;
        cudaMemcpy(d_zero_dev, &zero, sizeof(float), cudaMemcpyHostToDevice);

        cusparseCreateCsr(&matA, num_vertices, num_vertices, num_active_edges,
                          (void*)new_offsets, (void*)new_indices, (void*)values,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        cusparseCreateDnVec(&vecX, num_vertices, (void*)x_buf, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, num_vertices, (void*)pr_buf, CUDA_R_32F);  

        
        float h_one = 1.0f, h_zero = 0.0f;
        size_t bufferSize = 0;
        cusparseSpMV_bufferSize(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &h_one, matA, vecX, &h_zero, vecY,
                                 CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &bufferSize);
        if (bufferSize > 0) cudaMalloc(&cusparse_buffer, bufferSize);

        cusparseSpMV_preprocess(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &h_one, matA, vecX, &h_zero, vecY,
                                 CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, cusparse_buffer);

        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
    }

    
    float* pr_cur = pageranks;
    float* pr_new = pr_buf;
    float* d_dangling = &d_scalars[0];
    float* d_diff = &d_scalars[1];
    float one_minus_alpha = 1.0f - alpha;

    size_t iterations = 0;
    bool converged = false;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        cudaMemsetAsync(d_scalars, 0, 2 * sizeof(float));

        scale_and_dangling_kernel<<<grid_v, BLOCK>>>(pr_cur, out_degree, x_buf, d_dangling, num_vertices);

        if (use_cusparse) {
            
            cusparseDnVecSetValues(vecY, pr_new);

            
            cusparseSpMV(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         d_alpha_dev, matA, vecX, d_zero_dev, vecY,
                         CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, cusparse_buffer);

            
            teleport_diff_kernel<<<grid_v, BLOCK>>>(
                pr_new, pr_cur, pers_full,
                d_dangling, alpha, one_minus_alpha,
                num_vertices, d_diff);
        } else {
            spmv_teleport_diff_kernel<<<grid_v, BLOCK>>>(
                new_offsets, new_indices, x_buf,
                pr_new, pr_cur, pers_full,
                d_dangling, alpha, one_minus_alpha,
                num_vertices, d_diff);
        }

        float* tmp = pr_cur; pr_cur = pr_new; pr_new = tmp;

        float h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        iterations = iter + 1;
        if (h_diff < epsilon) { converged = true; break; }
    }

    
    if (use_cusparse) {
        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
        if (matA) cusparseDestroySpMat(matA);
        if (vecX) cusparseDestroyDnVec(vecX);
        if (vecY) cusparseDestroyDnVec(vecY);
        if (cusparse_buffer) cudaFree(cusparse_buffer);
    }

    if (pr_cur != pageranks)
        cudaMemcpy(pageranks, pr_cur, num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);

    return {iterations, converged};
}

}  
