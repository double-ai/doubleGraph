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
#include <algorithm>

namespace aai {

namespace {

constexpr int BLOCK_SIZE = 256;





__global__ void compute_flags_and_out_weight_sum_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_flags,
    double* __restrict__ out_weight_sum,
    int32_t num_edges
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < num_edges) {
        uint32_t word = edge_mask[e >> 5];
        int active = (int)((word >> (e & 31)) & 1u);
        active_flags[e] = active;
        if (active) {
            atomicAdd(&out_weight_sum[indices[e]], edge_weights[e]);
        }
    }
}

__global__ void scatter_compact_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ active_flags,
    const int32_t* __restrict__ positions,
    const double* __restrict__ out_weight_sum,
    int32_t* __restrict__ new_indices,
    double* __restrict__ new_norm_weights,
    int32_t num_edges
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < num_edges && active_flags[e]) {
        int32_t pos = positions[e];
        int32_t src = indices[e];
        new_indices[pos] = src;
        double ows = out_weight_sum[src];
        new_norm_weights[pos] = (ows > 0.0) ? edge_weights[e] / ows : 0.0;
    }
}

__global__ void build_new_offsets_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ positions,
    const int32_t* __restrict__ active_flags,
    int32_t* __restrict__ new_offsets,
    int32_t num_vertices,
    int32_t num_edges
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v <= num_vertices) {
        int32_t old_off = old_offsets[v];
        if (old_off < num_edges) {
            new_offsets[v] = positions[old_off];
        } else {
            new_offsets[v] = positions[num_edges - 1] + active_flags[num_edges - 1];
        }
    }
}

__global__ void compute_dangling_mask_kernel(
    const double* __restrict__ out_weight_sum,
    uint8_t* __restrict__ dangling_mask,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        dangling_mask[v] = (out_weight_sum[v] == 0.0) ? 1 : 0;
    }
}





__global__ void fill_kernel(double* __restrict__ arr, double val, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = val;
}

__global__ void dangling_sum_kernel(
    const double* __restrict__ pr,
    const uint8_t* __restrict__ dangling_mask,
    double* __restrict__ d_dangling_sum,
    int32_t num_vertices
) {
    __shared__ double smem[BLOCK_SIZE];
    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tid = threadIdx.x;

    double val = 0.0;
    if (v < num_vertices && dangling_mask[v]) {
        val = pr[v];
    }
    smem[tid] = val;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0 && smem[0] != 0.0) {
        atomicAdd(d_dangling_sum, smem[0]);
    }
}

__global__ void update_and_diff_kernel(
    double* __restrict__ pr_new,
    const double* __restrict__ pr_old,
    const double* __restrict__ spmv,
    const double* __restrict__ d_dangling_sum,
    double one_minus_alpha_over_n,
    double alpha,
    double alpha_over_n,
    int32_t num_vertices,
    double* __restrict__ d_l1_diff
) {
    __shared__ double smem[BLOCK_SIZE];
    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int tid = threadIdx.x;

    double diff = 0.0;
    if (v < num_vertices) {
        double dangling_contrib = (*d_dangling_sum) * alpha_over_n;
        double new_val = one_minus_alpha_over_n + alpha * spmv[v] + dangling_contrib;
        pr_new[v] = new_val;
        diff = fabs(new_val - pr_old[v]);
    }

    smem[tid] = diff;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0 && smem[0] != 0.0) {
        atomicAdd(d_l1_diff, smem[0]);
    }
}





template<typename T>
void ensure_alloc(T*& ptr, int64_t& cap, int64_t needed) {
    if (cap < needed) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, needed * sizeof(T));
        cap = needed;
    }
}

void ensure_alloc_bytes(void*& ptr, size_t& cap, size_t needed) {
    if (cap < needed) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, needed);
        cap = needed;
    }
}





struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;

    double* scalars = nullptr;

    double* out_ws = nullptr;       int64_t out_ws_cap = 0;
    uint8_t* dang_mask = nullptr;   int64_t dang_mask_cap = 0;
    double* spmv = nullptr;         int64_t spmv_cap = 0;
    double* pr_b = nullptr;         int64_t pr_b_cap = 0;
    int32_t* new_offsets = nullptr;  int64_t new_offsets_cap = 0;

    int32_t* flags = nullptr;       int64_t flags_cap = 0;
    int32_t* pos = nullptr;         int64_t pos_cap = 0;

    void* cub_temp = nullptr;       size_t cub_temp_cap = 0;

    int32_t* new_indices = nullptr;  int64_t new_indices_cap = 0;
    double* new_norm_w = nullptr;    int64_t new_norm_w_cap = 0;

    void* spmv_buf = nullptr;       size_t spmv_buf_cap = 0;

    Cache() {
        cusparseCreate(&handle);
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
        cudaMalloc(&scalars, 4 * sizeof(double));
    }

    ~Cache() override {
        if (handle) cusparseDestroy(handle);
        if (scalars) cudaFree(scalars);
        if (out_ws) cudaFree(out_ws);
        if (dang_mask) cudaFree(dang_mask);
        if (spmv) cudaFree(spmv);
        if (pr_b) cudaFree(pr_b);
        if (new_offsets) cudaFree(new_offsets);
        if (flags) cudaFree(flags);
        if (pos) cudaFree(pos);
        if (cub_temp) cudaFree(cub_temp);
        if (new_indices) cudaFree(new_indices);
        if (new_norm_w) cudaFree(new_norm_w);
        if (spmv_buf) cudaFree(spmv_buf);
    }
};

}  

PageRankResult pagerank_mask(const graph32_t& graph,
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
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cudaStream_t stream = 0;
    cusparseSetStream(cache.handle, stream);

    
    ensure_alloc(cache.out_ws, cache.out_ws_cap, (int64_t)num_vertices);
    ensure_alloc(cache.dang_mask, cache.dang_mask_cap, (int64_t)num_vertices);
    ensure_alloc(cache.spmv, cache.spmv_cap, (int64_t)num_vertices);
    ensure_alloc(cache.pr_b, cache.pr_b_cap, (int64_t)num_vertices);
    ensure_alloc(cache.new_offsets, cache.new_offsets_cap, (int64_t)(num_vertices + 1));
    ensure_alloc(cache.flags, cache.flags_cap, (int64_t)num_edges);
    ensure_alloc(cache.pos, cache.pos_cap, (int64_t)num_edges);

    
    size_t cub_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_temp_bytes,
                                  (int32_t*)nullptr, (int32_t*)nullptr, num_edges);
    ensure_alloc_bytes(cache.cub_temp, cache.cub_temp_cap,
                       std::max((size_t)1, cub_temp_bytes));

    double* d_out_ws = cache.out_ws;
    uint8_t* d_dang_mask = cache.dang_mask;
    double* d_spmv = cache.spmv;
    int32_t* d_new_offsets = cache.new_offsets;
    int32_t* d_flags = cache.flags;
    int32_t* d_pos = cache.pos;

    
    
    
    cudaMemsetAsync(d_out_ws, 0, num_vertices * sizeof(double), stream);

    int grid = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_flags_and_out_weight_sum_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_indices, edge_weights, d_edge_mask,
        d_flags, d_out_ws, num_edges);

    cub::DeviceScan::ExclusiveSum(cache.cub_temp, cub_temp_bytes,
                                  d_flags, d_pos, num_edges, stream);

    grid = (num_vertices + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    build_new_offsets_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_offsets, d_pos, d_flags, d_new_offsets,
        num_vertices, num_edges);

    grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_dangling_mask_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_out_ws, d_dang_mask, num_vertices);

    
    int32_t total_active;
    cudaMemcpyAsync(&total_active, d_new_offsets + num_vertices,
                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int64_t ta = std::max((int64_t)1, (int64_t)total_active);
    ensure_alloc(cache.new_indices, cache.new_indices_cap, ta);
    ensure_alloc(cache.new_norm_w, cache.new_norm_w_cap, ta);
    int32_t* d_new_indices = cache.new_indices;
    double* d_new_norm_w = cache.new_norm_w;

    grid = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scatter_compact_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_indices, edge_weights, d_flags, d_pos, d_out_ws,
        d_new_indices, d_new_norm_w, num_edges);

    
    
    
    double* d_scalars = cache.scalars;
    double* d_dangling  = d_scalars + 0;
    double* d_l1_diff   = d_scalars + 1;
    double* d_alpha_one = d_scalars + 2;
    double* d_beta_zero = d_scalars + 3;

    double h_scalars[4] = {0.0, 0.0, 1.0, 0.0};
    cudaMemcpyAsync(d_scalars, h_scalars, 4 * sizeof(double),
                    cudaMemcpyHostToDevice, stream);

    
    double* d_pr_old = pageranks;
    double* d_pr_new = cache.pr_b;
    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(d_pr_old, initial_pageranks,
                       num_vertices * sizeof(double),
                       cudaMemcpyDeviceToDevice, stream);
    } else {
        grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fill_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_pr_old, 1.0 / num_vertices, num_vertices);
    }

    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(
        &matA, num_vertices, num_vertices, (int64_t)total_active,
        d_new_offsets, d_new_indices, d_new_norm_w,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, num_vertices, d_pr_old, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, num_vertices, d_spmv, CUDA_R_64F);

    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(
        cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_alpha_one, matA, vecX, d_beta_zero, vecY,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

    ensure_alloc_bytes(cache.spmv_buf, cache.spmv_buf_cap,
                       std::max((size_t)1, bufferSize));

    cusparseSpMV_preprocess(
        cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_alpha_one, matA, vecX, d_beta_zero, vecY,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

    double one_minus_alpha_over_n = (1.0 - alpha) / num_vertices;
    double alpha_over_n = alpha / num_vertices;

    
    
    
    bool converged = false;
    std::size_t iterations = 0;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(d_dangling, 0, 2 * sizeof(double), stream);

        
        cusparseDnVecSetValues(vecX, d_pr_old);
        cusparseSpMV(
            cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_alpha_one, matA, vecX, d_beta_zero, vecY,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

        
        grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dangling_sum_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_pr_old, d_dang_mask, d_dangling, num_vertices);

        
        update_and_diff_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_pr_new, d_pr_old, d_spmv, d_dangling,
            one_minus_alpha_over_n, alpha, alpha_over_n,
            num_vertices, d_l1_diff);

        
        double h_diff;
        cudaMemcpyAsync(&h_diff, d_l1_diff, sizeof(double),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations = iter + 1;
        if (h_diff < epsilon) {
            converged = true;
            std::swap(d_pr_old, d_pr_new);
            break;
        }

        std::swap(d_pr_old, d_pr_new);
    }

    
    
    
    if (d_pr_old != pageranks) {
        cudaMemcpyAsync(pageranks, d_pr_old, num_vertices * sizeof(double),
                       cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    return PageRankResult{iterations, converged};
}

}  
