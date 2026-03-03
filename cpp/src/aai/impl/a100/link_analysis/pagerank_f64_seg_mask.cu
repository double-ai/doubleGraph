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
#include <cmath>

namespace aai {

namespace {





__global__ void create_masked_weights_kernel(
    const double* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ masked_weights,
    int32_t num_edges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    uint32_t word = edge_mask[e >> 5];
    masked_weights[e] = ((word >> (e & 31)) & 1) ? edge_weights[e] : 0.0;
}

__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ masked_weights,
    double* __restrict__ out_weight_sums,
    int32_t num_edges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    double w = masked_weights[e];
    if (w > 0.0) {
        atomicAdd(&out_weight_sums[indices[e]], w);
    }
}

__global__ void init_pageranks_kernel(double* __restrict__ pr, double inv_N, int32_t N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) pr[i] = inv_N;
}

__global__ void compute_pr_norm_kernel(
    const double* __restrict__ pr,
    const double* __restrict__ out_ws,
    double* __restrict__ pr_norm,
    double* __restrict__ d_dangling_sum,
    int32_t N)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double my_dangling = 0.0;

    if (i < N) {
        double ow = out_ws[i];
        if (ow > 0.0) {
            pr_norm[i] = pr[i] / ow;
        } else {
            pr_norm[i] = 0.0;
            my_dangling = pr[i];
        }
    }

    double block_sum = BlockReduce(temp).Sum(my_dangling);
    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(d_dangling_sum, block_sum);
    }
}

__global__ void update_pr_kernel(
    const double* __restrict__ spmv_result,
    const double* __restrict__ pr_old,
    double* __restrict__ pr_new,
    double* __restrict__ d_diff,
    const double* __restrict__ d_dangling_sum,
    double alpha,
    double one_minus_alpha_inv_N,
    double alpha_inv_N,
    int32_t N)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double my_diff = 0.0;

    if (i < N) {
        double base_pr = one_minus_alpha_inv_N + alpha_inv_N * d_dangling_sum[0];
        double new_val = base_pr + alpha * spmv_result[i];
        pr_new[i] = new_val;
        my_diff = fabs(new_val - pr_old[i]);
    }

    double block_diff = BlockReduce(temp).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff > 0.0) {
        atomicAdd(d_diff, block_diff);
    }
}





struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;

    
    double* masked_wt = nullptr;
    int64_t masked_wt_capacity = 0;

    
    double* out_ws = nullptr;
    int64_t out_ws_capacity = 0;

    double* pr_buf = nullptr;
    int64_t pr_buf_capacity = 0;

    double* pr_norm = nullptr;
    int64_t pr_norm_capacity = 0;

    double* spmv_res = nullptr;
    int64_t spmv_res_capacity = 0;

    
    double* d_dangling = nullptr;
    bool d_dangling_allocated = false;

    double* d_diff = nullptr;
    bool d_diff_allocated = false;

    
    void* spmv_buf = nullptr;
    size_t spmv_buf_capacity = 0;

    Cache() {
        cusparseCreate(&handle);
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    }

    void ensure(int64_t nv, int64_t ne) {
        if (masked_wt_capacity < ne) {
            if (masked_wt) cudaFree(masked_wt);
            cudaMalloc(&masked_wt, ne * sizeof(double));
            masked_wt_capacity = ne;
        }
        if (out_ws_capacity < nv) {
            if (out_ws) cudaFree(out_ws);
            cudaMalloc(&out_ws, nv * sizeof(double));
            out_ws_capacity = nv;
        }
        if (pr_buf_capacity < nv) {
            if (pr_buf) cudaFree(pr_buf);
            cudaMalloc(&pr_buf, nv * sizeof(double));
            pr_buf_capacity = nv;
        }
        if (pr_norm_capacity < nv) {
            if (pr_norm) cudaFree(pr_norm);
            cudaMalloc(&pr_norm, nv * sizeof(double));
            pr_norm_capacity = nv;
        }
        if (spmv_res_capacity < nv) {
            if (spmv_res) cudaFree(spmv_res);
            cudaMalloc(&spmv_res, nv * sizeof(double));
            spmv_res_capacity = nv;
        }
        if (!d_dangling_allocated) {
            cudaMalloc(&d_dangling, sizeof(double));
            d_dangling_allocated = true;
        }
        if (!d_diff_allocated) {
            cudaMalloc(&d_diff, sizeof(double));
            d_diff_allocated = true;
        }
    }

    void ensure_spmv_buffer(size_t required) {
        if (spmv_buf_capacity < required) {
            if (spmv_buf) cudaFree(spmv_buf);
            size_t alloc_size = required > 0 ? required : 1;
            cudaMalloc(&spmv_buf, alloc_size);
            spmv_buf_capacity = alloc_size;
        }
    }

    ~Cache() override {
        if (handle) cusparseDestroy(handle);
        if (masked_wt) cudaFree(masked_wt);
        if (out_ws) cudaFree(out_ws);
        if (pr_buf) cudaFree(pr_buf);
        if (pr_norm) cudaFree(pr_norm);
        if (spmv_res) cudaFree(spmv_res);
        if (d_dangling) cudaFree(d_dangling);
        if (d_diff) cudaFree(d_diff);
        if (spmv_buf) cudaFree(spmv_buf);
    }
};

}  

PageRankResult pagerank_seg_mask(const graph32_t& graph,
                                 const double* edge_weights,
                                 double* pageranks,
                                 const double* precomputed_vertex_out_weight_sums,
                                 double alpha,
                                 double epsilon,
                                 std::size_t max_iterations,
                                 const double* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    const uint32_t* edge_mask = graph.edge_mask;

    cache.ensure(nv, ne);

    
    int grid_e = (ne + 255) / 256;
    create_masked_weights_kernel<<<grid_e, 256>>>(edge_weights, edge_mask,
        cache.masked_wt, ne);

    
    cudaMemsetAsync(cache.out_ws, 0, nv * sizeof(double), 0);
    compute_out_weight_sums_kernel<<<grid_e, 256>>>(indices, cache.masked_wt,
        cache.out_ws, ne);

    
    cusparseSpMatDescr_t mat_descr;
    cusparseCreateCsr(&mat_descr, nv, nv, ne,
        (void*)offsets, (void*)indices, (void*)cache.masked_wt,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseDnVecDescr_t x_descr, y_descr;
    cusparseCreateDnVec(&x_descr, nv, cache.pr_norm, CUDA_R_64F);
    cusparseCreateDnVec(&y_descr, nv, cache.spmv_res, CUDA_R_64F);

    double h_alpha = 1.0, h_beta = 0.0;

    size_t buffer_size = 0;
    cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha, mat_descr, x_descr, &h_beta, y_descr,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);

    cache.ensure_spmv_buffer(buffer_size);

    cusparseSpMV_preprocess(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_alpha, mat_descr, x_descr, &h_beta, y_descr,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

    
    if (initial_pageranks) {
        cudaMemcpy(pageranks, initial_pageranks,
                   nv * sizeof(double), cudaMemcpyDeviceToDevice);
    } else {
        double inv_N = 1.0 / (double)nv;
        int grid_v = (nv + 255) / 256;
        init_pageranks_kernel<<<grid_v, 256>>>(pageranks, inv_N, nv);
    }

    
    double inv_N = 1.0 / (double)nv;
    double omaN = (1.0 - alpha) * inv_N;
    double aN = alpha * inv_N;

    double* cur_pr = pageranks;
    double* new_pr = cache.pr_buf;

    bool converged = false;
    std::size_t iter;

    for (iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(cache.d_dangling, 0, sizeof(double), 0);
        int grid_v = (nv + 255) / 256;
        compute_pr_norm_kernel<<<grid_v, 256>>>(cur_pr, cache.out_ws,
            cache.pr_norm, cache.d_dangling, nv);

        
        cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_alpha, mat_descr, x_descr, &h_beta, y_descr,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

        
        cudaMemsetAsync(cache.d_diff, 0, sizeof(double), 0);
        update_pr_kernel<<<grid_v, 256>>>(cache.spmv_res, cur_pr, new_pr,
            cache.d_diff, cache.d_dangling, alpha, omaN, aN, nv);

        
        double* tmp = cur_pr; cur_pr = new_pr; new_pr = tmp;

        
        double h_diff;
        cudaMemcpy(&h_diff, cache.d_diff, sizeof(double), cudaMemcpyDeviceToHost);
        if (h_diff < epsilon) {
            converged = true;
            iter++;
            break;
        }
    }

    
    if (cur_pr != pageranks) {
        cudaMemcpy(pageranks, cur_pr, nv * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    
    cusparseDestroySpMat(mat_descr);
    cusparseDestroyDnVec(x_descr);
    cusparseDestroyDnVec(y_descr);

    return PageRankResult{iter, converged};
}

}  
