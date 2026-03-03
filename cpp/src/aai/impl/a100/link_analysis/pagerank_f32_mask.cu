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



__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ out_weight_sums,
    int32_t num_edges)
{
    for (int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
         e < num_edges; e += gridDim.x * blockDim.x) {
        uint32_t word = edge_mask[e >> 5];
        if ((word >> (e & 31)) & 1u) {
            atomicAdd(&out_weight_sums[indices[e]], edge_weights[e]);
        }
    }
}

__global__ void compute_norm_weights_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ out_weight_sums,
    float* __restrict__ norm_weights,
    int32_t num_edges)
{
    for (int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
         e < num_edges; e += gridDim.x * blockDim.x) {
        uint32_t word = edge_mask[e >> 5];
        if ((word >> (e & 31)) & 1u) {
            float ows = out_weight_sums[indices[e]];
            norm_weights[e] = (ows > 0.0f) ? __fdividef(edge_weights[e], ows) : 0.0f;
        } else {
            norm_weights[e] = 0.0f;
        }
    }
}

__global__ void init_pr_kernel(float* __restrict__ pr, float val, int32_t n)
{
    for (int32_t v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += gridDim.x * blockDim.x) {
        pr[v] = val;
    }
}



__global__ void __launch_bounds__(BLOCK_SIZE)
dangling_sum_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ out_weight_sums,
    float* __restrict__ d_dangling,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float local = 0.0f;
    for (int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
         v < num_vertices; v += gridDim.x * blockDim.x) {
        if (out_weight_sums[v] == 0.0f) {
            local += pr[v];
        }
    }

    float block_sum = BlockReduce(temp).Sum(local);
    if (threadIdx.x == 0 && block_sum != 0.0f)
        atomicAdd(d_dangling, block_sum);
}

__global__ void __launch_bounds__(BLOCK_SIZE)
post_spmv_inplace_diff_kernel(
    float* __restrict__ pr_new,
    const float* __restrict__ pr_old,
    float alpha,
    float inv_n,
    const float* __restrict__ d_dangling,
    int32_t num_vertices,
    float* __restrict__ d_diff)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float dangling = *d_dangling;
    float base_score = (1.0f - alpha) * inv_n + alpha * dangling * inv_n;

    float local_diff = 0.0f;
    for (int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
         v < num_vertices; v += gridDim.x * blockDim.x) {
        float spmv_val = pr_new[v];
        float new_val = base_score + alpha * spmv_val;
        pr_new[v] = new_val;
        local_diff += fabsf(new_val - pr_old[v]);
    }

    float block_diff = BlockReduce(temp).Sum(local_diff);
    if (threadIdx.x == 0 && block_diff != 0.0f)
        atomicAdd(d_diff, block_diff);
}

__global__ void __launch_bounds__(BLOCK_SIZE)
spmv_diff_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ norm_weights,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    float alpha,
    float inv_n,
    const float* __restrict__ d_dangling,
    int32_t num_vertices,
    float* __restrict__ d_diff)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float dangling = *d_dangling;
    float base_score = (1.0f - alpha) * inv_n + alpha * dangling * inv_n;

    float local_diff = 0.0f;
    for (int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
         v < num_vertices; v += gridDim.x * blockDim.x) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        float sum = 0.0f;
        for (int32_t e = start; e < end; e++) {
            sum += pr_old[indices[e]] * norm_weights[e];
        }
        float new_val = base_score + alpha * sum;
        pr_new[v] = new_val;
        local_diff += fabsf(new_val - pr_old[v]);
    }

    float block_diff = BlockReduce(temp).Sum(local_diff);
    if (threadIdx.x == 0 && block_diff != 0.0f)
        atomicAdd(d_diff, block_diff);
}



struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    float* d_one = nullptr;
    float* d_zero = nullptr;
    float* scratch = nullptr;
    float* pr_b = nullptr;
    float* ows = nullptr;
    float* nw = nullptr;
    void* cusparse_buffer = nullptr;

    int32_t pr_b_capacity = 0;
    int32_t ows_capacity = 0;
    int32_t nw_capacity = 0;
    size_t cusparse_buf_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cudaMalloc(&d_one, sizeof(float));
        cudaMalloc(&d_zero, sizeof(float));
        cudaMalloc(&scratch, 2 * sizeof(float));
        float h_one = 1.0f, h_zero = 0.0f;
        cudaMemcpy(d_one, &h_one, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_zero, &h_zero, sizeof(float), cudaMemcpyHostToDevice);
    }

    ~Cache() override {
        if (d_one) cudaFree(d_one);
        if (d_zero) cudaFree(d_zero);
        if (scratch) cudaFree(scratch);
        if (pr_b) cudaFree(pr_b);
        if (ows) cudaFree(ows);
        if (nw) cudaFree(nw);
        if (cusparse_buffer) cudaFree(cusparse_buffer);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
    }

    void ensure_pr_b(int32_t nv) {
        if (pr_b_capacity < nv) {
            if (pr_b) cudaFree(pr_b);
            cudaMalloc(&pr_b, nv * sizeof(float));
            pr_b_capacity = nv;
        }
    }

    void ensure_ows(int32_t nv) {
        if (ows_capacity < nv) {
            if (ows) cudaFree(ows);
            cudaMalloc(&ows, nv * sizeof(float));
            ows_capacity = nv;
        }
    }

    void ensure_nw(int32_t ne) {
        if (nw_capacity < ne) {
            if (nw) cudaFree(nw);
            cudaMalloc(&nw, ne * sizeof(float));
            nw_capacity = ne;
        }
    }

    void ensure_cusparse_buf(size_t size) {
        if (cusparse_buf_capacity < size) {
            if (cusparse_buffer) cudaFree(cusparse_buffer);
            cudaMalloc(&cusparse_buffer, size);
            cusparse_buf_capacity = size;
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

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    const uint32_t* d_mask = graph.edge_mask;

    cudaStream_t stream = 0;
    cusparseSetStream(cache.cusparse_handle, stream);

    
    cache.ensure_pr_b(nv);
    cache.ensure_nw(ne);

    float* pa = pageranks;
    float* pb = cache.pr_b;
    float* p_nw = cache.nw;
    float* p_dangling = cache.scratch;
    float* p_diff = cache.scratch + 1;

    
    cache.ensure_ows(nv);
    cudaMemsetAsync(cache.ows, 0, nv * sizeof(float), stream);
    {
        int grid = (ne + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid > 4096) grid = 4096;
        compute_out_weight_sums_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_indices, edge_weights, d_mask, cache.ows, ne);
    }
    const float* p_ows = cache.ows;

    
    {
        int grid = (ne + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid > 4096) grid = 4096;
        compute_norm_weights_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_indices, edge_weights, d_mask, p_ows, p_nw, ne);
    }

    
    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(pa, initial_pageranks, nv * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        int grid = (nv + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid > 4096) grid = 4096;
        init_pr_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(pa, 1.0f / nv, nv);
    }

    float inv_n = 1.0f / nv;
    float* pr_old = pa;
    float* pr_new = pb;

    
    float avg_degree = (nv > 0) ? (float)ne / nv : 0;
    bool use_cusparse = (ne > 200000 && avg_degree > 3.5f);

    cusparseSpMatDescr_t mat_descr = nullptr;
    cusparseDnVecDescr_t vec_in_descr = nullptr;
    cusparseDnVecDescr_t vec_out_descr = nullptr;
    cusparseSpMVAlg_t spmv_alg = CUSPARSE_SPMV_CSR_ALG2;

    if (use_cusparse) {
        cusparseCreateCsr(&mat_descr, nv, nv, ne,
            (void*)d_offsets, (void*)d_indices, (void*)p_nw,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        cusparseCreateDnVec(&vec_in_descr, nv, pr_old, CUDA_R_32F);
        cusparseCreateDnVec(&vec_out_descr, nv, pr_new, CUDA_R_32F);

        float h_one = 1.0f, h_zero = 0.0f;
        size_t buf_size = 0;
        cusparseSpMV_bufferSize(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_one, mat_descr, vec_in_descr,
            &h_zero, vec_out_descr,
            CUDA_R_32F, spmv_alg,
            &buf_size);

        if (buf_size > 0) {
            cache.ensure_cusparse_buf(buf_size);
        }

        cusparseSpMV_preprocess(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_one, mat_descr, vec_in_descr,
            &h_zero, vec_out_descr,
            CUDA_R_32F, spmv_alg,
            cache.cusparse_buffer);

        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
    }

    size_t iter = 0;
    bool converged = false;
    float h_diff;

    for (; iter < max_iterations; iter++) {
        cudaMemsetAsync(p_dangling, 0, sizeof(float), stream);
        cudaMemsetAsync(p_diff, 0, sizeof(float), stream);

        {
            int grid = (nv + BLOCK_SIZE - 1) / BLOCK_SIZE;
            if (grid > 1024) grid = 1024;
            dangling_sum_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                pr_old, p_ows, p_dangling, nv);
        }

        if (use_cusparse) {
            cusparseDnVecSetValues(vec_in_descr, pr_old);
            cusparseDnVecSetValues(vec_out_descr, pr_new);

            cusparseSpMV(cache.cusparse_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                cache.d_one, mat_descr, vec_in_descr,
                cache.d_zero, vec_out_descr,
                CUDA_R_32F, spmv_alg,
                cache.cusparse_buffer);

            {
                int grid = (nv + BLOCK_SIZE - 1) / BLOCK_SIZE;
                if (grid > 4096) grid = 4096;
                post_spmv_inplace_diff_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                    pr_new, pr_old, alpha, inv_n, p_dangling, nv, p_diff);
            }
        } else {
            int grid = (nv + BLOCK_SIZE - 1) / BLOCK_SIZE;
            if (grid > 4096) grid = 4096;
            spmv_diff_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                d_offsets, d_indices, p_nw, pr_old, pr_new,
                alpha, inv_n, p_dangling, nv, p_diff);
        }

        cudaMemcpyAsync(&h_diff, p_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        float* tmp = pr_old; pr_old = pr_new; pr_new = tmp;

        if (h_diff < epsilon) {
            converged = true;
            iter++;
            break;
        }
    }

    if (use_cusparse) {
        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
        if (mat_descr) cusparseDestroySpMat(mat_descr);
        if (vec_in_descr) cusparseDestroyDnVec(vec_in_descr);
        if (vec_out_descr) cusparseDestroyDnVec(vec_out_descr);
    }

    
    if (pr_old != pageranks) {
        cudaMemcpyAsync(pageranks, pr_old, nv * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    return PageRankResult{iter, converged};
}

}  
