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

namespace aai {

namespace {





__global__ void extract_mask_kernel(
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_flags,
    int32_t num_edges
) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_edges;
         e += gridDim.x * blockDim.x) {
        uint32_t word = edge_mask[e >> 5];
        active_flags[e] = (word >> (e & 31)) & 1;
    }
}

__global__ void build_new_offsets_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ prefix_sum,
    const int32_t* __restrict__ active_flags,
    int32_t* __restrict__ new_offsets,
    int32_t num_vertices,
    int32_t num_edges
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v <= num_vertices;
         v += gridDim.x * blockDim.x) {
        int old_off = old_offsets[v];
        if (old_off < num_edges)
            new_offsets[v] = prefix_sum[old_off];
        else
            new_offsets[v] = prefix_sum[num_edges - 1] + active_flags[num_edges - 1];
    }
}

__global__ void scatter_edges_kernel(
    const int32_t* __restrict__ old_indices,
    const double* __restrict__ old_weights,
    const int32_t* __restrict__ active_flags,
    const int32_t* __restrict__ prefix_sum,
    int32_t* __restrict__ new_indices,
    double* __restrict__ new_weights,
    int32_t num_edges
) {
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_edges;
         e += gridDim.x * blockDim.x) {
        if (active_flags[e]) {
            int new_pos = prefix_sum[e];
            new_indices[new_pos] = old_indices[e];
            new_weights[new_pos] = old_weights[e];
        }
    }
}




__global__ __launch_bounds__(256)
void add_identity_and_l2_kernel(
    double* __restrict__ x_new,
    const double* __restrict__ x_old,
    int32_t num_vertices,
    double* __restrict__ d_l2_sq
) {
    double local_l2 = 0.0;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices;
         v += gridDim.x * blockDim.x) {
        double val = x_new[v] + x_old[v];
        x_new[v] = val;
        local_l2 += val * val;
    }
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_l2 = BlockReduce(temp_storage).Sum(local_l2);
    if (threadIdx.x == 0 && block_l2 != 0.0)
        atomicAdd(d_l2_sq, block_l2);
}




__global__ __launch_bounds__(256)
void normalize_diff_kernel(
    double* __restrict__ x_new,
    const double* __restrict__ x_old,
    int32_t num_vertices,
    const double* __restrict__ d_l2_sq,
    double* __restrict__ d_diff
) {
    double norm_sq = *d_l2_sq;
    double inv_norm = (norm_sq > 0.0) ? rsqrt(norm_sq) : 1.0;
    double local_diff = 0.0;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices;
         v += gridDim.x * blockDim.x) {
        double val = x_new[v] * inv_norm;
        x_new[v] = val;
        local_diff += fabs(val - x_old[v]);
    }
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_diff = BlockReduce(temp_storage).Sum(local_diff);
    if (threadIdx.x == 0 && block_diff != 0.0)
        atomicAdd(d_diff, block_diff);
}

__global__ void init_uniform_kernel(double* __restrict__ x, int32_t n) {
    double val = 1.0 / (double)n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        x[i] = val;
}





struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    double* h_diff_pinned = nullptr;

    double* d_l2_sq = nullptr;
    double* d_diff = nullptr;

    int32_t* active_flags = nullptr;
    int64_t active_flags_capacity = 0;

    int32_t* prefix_sum = nullptr;
    int64_t prefix_sum_capacity = 0;

    void* cub_temp = nullptr;
    size_t cub_temp_capacity = 0;

    int32_t* new_offsets = nullptr;
    int64_t new_offsets_capacity = 0;

    double* x_buf = nullptr;
    int64_t x_buf_capacity = 0;

    int32_t* new_indices = nullptr;
    int64_t new_indices_capacity = 0;

    double* new_weights = nullptr;
    int64_t new_weights_capacity = 0;

    void* spmv_buf = nullptr;
    size_t spmv_buf_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
        cudaHostAlloc(&h_diff_pinned, sizeof(double), cudaHostAllocDefault);
        cudaMalloc(&d_l2_sq, sizeof(double));
        cudaMalloc(&d_diff, sizeof(double));
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (h_diff_pinned) cudaFreeHost(h_diff_pinned);
        if (d_l2_sq) cudaFree(d_l2_sq);
        if (d_diff) cudaFree(d_diff);
        if (active_flags) cudaFree(active_flags);
        if (prefix_sum) cudaFree(prefix_sum);
        if (cub_temp) cudaFree(cub_temp);
        if (new_offsets) cudaFree(new_offsets);
        if (x_buf) cudaFree(x_buf);
        if (new_indices) cudaFree(new_indices);
        if (new_weights) cudaFree(new_weights);
        if (spmv_buf) cudaFree(spmv_buf);
    }

    void ensure_edge_buffers(int64_t num_edges) {
        if (active_flags_capacity < num_edges) {
            if (active_flags) cudaFree(active_flags);
            cudaMalloc(&active_flags, num_edges * sizeof(int32_t));
            active_flags_capacity = num_edges;
        }
        if (prefix_sum_capacity < num_edges) {
            if (prefix_sum) cudaFree(prefix_sum);
            cudaMalloc(&prefix_sum, num_edges * sizeof(int32_t));
            prefix_sum_capacity = num_edges;
        }
    }

    void ensure_cub_temp(size_t temp_bytes) {
        if (cub_temp_capacity < temp_bytes) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, temp_bytes);
            cub_temp_capacity = temp_bytes;
        }
    }

    void ensure_vertex_buffers(int64_t num_vertices) {
        if (new_offsets_capacity < num_vertices + 1) {
            if (new_offsets) cudaFree(new_offsets);
            cudaMalloc(&new_offsets, (num_vertices + 1) * sizeof(int32_t));
            new_offsets_capacity = num_vertices + 1;
        }
        if (x_buf_capacity < num_vertices) {
            if (x_buf) cudaFree(x_buf);
            cudaMalloc(&x_buf, num_vertices * sizeof(double));
            x_buf_capacity = num_vertices;
        }
    }

    void ensure_active_edge_buffers(int64_t total_active) {
        int64_t alloc_edges = total_active > 0 ? total_active : 1;
        if (new_indices_capacity < alloc_edges) {
            if (new_indices) cudaFree(new_indices);
            cudaMalloc(&new_indices, alloc_edges * sizeof(int32_t));
            new_indices_capacity = alloc_edges;
        }
        if (new_weights_capacity < alloc_edges) {
            if (new_weights) cudaFree(new_weights);
            cudaMalloc(&new_weights, alloc_edges * sizeof(double));
            new_weights_capacity = alloc_edges;
        }
    }

    void ensure_spmv_buf(size_t buf_size) {
        size_t alloc_size = buf_size > 0 ? buf_size : 1;
        if (spmv_buf_capacity < alloc_size) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, alloc_size);
            spmv_buf_capacity = alloc_size;
        }
    }
};

}  

eigenvector_centrality_result_t eigenvector_centrality_mask(const graph32_t& graph,
                                  const double* edge_weights,
                                  double* centralities,
                                  double epsilon,
                                  std::size_t max_iterations,
                                  const double* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    if (num_vertices == 0) {
        return {0, true};
    }

    cudaStream_t stream = 0;
    cusparseSetStream(cache.cusparse_handle, stream);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    
    cache.ensure_edge_buffers(num_edges);

    int bs = 256;
    int gs = min((num_edges + bs - 1) / bs, 65535);
    extract_mask_kernel<<<gs, bs, 0, stream>>>(d_edge_mask, cache.active_flags, num_edges);

    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, (int32_t*)nullptr,
                                  (int32_t*)nullptr, num_edges);
    cache.ensure_cub_temp(temp_bytes);
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, temp_bytes,
                                  cache.active_flags, cache.prefix_sum, num_edges, stream);

    
    int32_t h_last_prefix, h_last_flag;
    cudaMemcpyAsync(&h_last_prefix, cache.prefix_sum + num_edges - 1, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_last_flag, cache.active_flags + num_edges - 1, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int32_t total_active = h_last_prefix + h_last_flag;

    
    cache.ensure_vertex_buffers(num_vertices);
    cache.ensure_active_edge_buffers(total_active);

    gs = min((num_vertices + 1 + bs - 1) / bs, 65535);
    build_new_offsets_kernel<<<gs, bs, 0, stream>>>(
        d_offsets, cache.prefix_sum, cache.active_flags, cache.new_offsets,
        num_vertices, num_edges);

    if (total_active > 0) {
        gs = min((num_edges + bs - 1) / bs, 65535);
        scatter_edges_kernel<<<gs, bs, 0, stream>>>(
            d_indices, edge_weights, cache.active_flags, cache.prefix_sum,
            cache.new_indices, cache.new_weights, num_edges);
    }

    
    cusparseSpMatDescr_t mat_descr;
    cusparseCreateCsr(&mat_descr,
        (int64_t)num_vertices, (int64_t)num_vertices, (int64_t)total_active,
        cache.new_offsets, cache.new_indices, cache.new_weights,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    
    double* x_a = centralities;
    double* x_b = cache.x_buf;

    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(x_a, initial_centralities,
                       num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else {
        gs = min((num_vertices + bs - 1) / bs, 65535);
        init_uniform_kernel<<<gs, bs, 0, stream>>>(x_a, num_vertices);
    }

    
    cusparseDnVecDescr_t vec_x, vec_y;
    cusparseCreateDnVec(&vec_x, num_vertices, x_a, CUDA_R_64F);
    cusparseCreateDnVec(&vec_y, num_vertices, x_b, CUDA_R_64F);

    double alpha = 1.0, beta_zero = 0.0;
    size_t spmv_buf_size = 0;
    cusparseSpMV_bufferSize(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_descr, vec_x, &beta_zero, vec_y,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buf_size);

    cache.ensure_spmv_buf(spmv_buf_size);

    cusparseSpMV_preprocess(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_descr, vec_x, &beta_zero, vec_y,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

    
    double tol = (double)num_vertices * epsilon;
    size_t iterations = 0;
    bool converged = false;
    const int CHECK_INTERVAL = 10;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        cusparseDnVecSetValues(vec_x, x_a);
        cusparseDnVecSetValues(vec_y, x_b);

        
        cusparseSpMV(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_descr, vec_x, &beta_zero, vec_y,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

        
        cudaMemsetAsync(cache.d_l2_sq, 0, sizeof(double), stream);
        gs = min((num_vertices + bs - 1) / bs, 65535);
        add_identity_and_l2_kernel<<<gs, bs, 0, stream>>>(x_b, x_a, num_vertices, cache.d_l2_sq);

        
        cudaMemsetAsync(cache.d_diff, 0, sizeof(double), stream);
        normalize_diff_kernel<<<gs, bs, 0, stream>>>(x_b, x_a, num_vertices, cache.d_l2_sq, cache.d_diff);

        iterations++;
        double* temp = x_a; x_a = x_b; x_b = temp;

        if ((iter + 1) % CHECK_INTERVAL == 0 || iter == max_iterations - 1) {
            cudaMemcpyAsync(cache.h_diff_pinned, cache.d_diff, sizeof(double),
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (*cache.h_diff_pinned < tol) { converged = true; break; }
        }
    }

    if (x_a != centralities)
        cudaMemcpyAsync(centralities, x_a,
                       num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);

    cusparseDestroySpMat(mat_descr);
    cusparseDestroyDnVec(vec_x);
    cusparseDestroyDnVec(vec_y);

    return {iterations, converged};
}

}  
