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
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <cstdint>
#include <algorithm>

namespace aai {

namespace {

constexpr int BLOCK = 256;




struct MaskToFlag {
    const uint32_t* mask;
    __host__ __device__ __forceinline__ int operator()(int e) const {
        return (mask[e >> 5] >> (e & 31)) & 1;
    }
};





__global__ void compact_edges_kernel(
    const int* __restrict__ old_indices,
    const float* __restrict__ old_weights,
    const uint32_t* __restrict__ edge_mask,
    const int* __restrict__ prefix_sum,
    int* __restrict__ new_indices,
    float* __restrict__ new_weights,
    int num_edges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < num_edges) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1) {
            int ne = prefix_sum[e];
            new_indices[ne] = old_indices[e];
            new_weights[ne] = old_weights[e];
        }
    }
}

__global__ void build_offsets_kernel(
    const int* __restrict__ old_offsets,
    const int* __restrict__ prefix_sum,
    int* __restrict__ new_offsets,
    int num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v <= num_vertices) {
        new_offsets[v] = prefix_sum[old_offsets[v]];
    }
}





__global__ void init_uniform_kernel(float* __restrict__ x, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = 1.0f / (float)n;
    }
}

__global__ void add_identity_l2_kernel(
    float* __restrict__ y,
    const float* __restrict__ x_old,
    float* __restrict__ l2_accum,
    int n)
{
    typedef cub::BlockReduce<float, BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float partial = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float val = y[i] + x_old[i];
        y[i] = val;
        partial += val * val;
    }

    float bsum = BlockReduce(temp).Sum(partial);
    if (threadIdx.x == 0 && bsum != 0.0f) {
        atomicAdd(l2_accum, bsum);
    }
}

__global__ void normalize_diff_kernel(
    const float* __restrict__ y,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float* __restrict__ l2_norm_sq,
    float* __restrict__ diff_accum,
    int n)
{
    typedef cub::BlockReduce<float, BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float l2sq = *l2_norm_sq;
    float inv_norm = (l2sq > 0.0f) ? rsqrtf(l2sq) : 0.0f;

    float partial = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        float nv = y[i] * inv_norm;
        x_new[i] = nv;
        partial += fabsf(nv - x_old[i]);
    }

    float bsum = BlockReduce(temp).Sum(partial);
    if (threadIdx.x == 0 && bsum != 0.0f) {
        atomicAdd(diff_accum, bsum);
    }
}





void launch_compact_edges(const int* oi, const float* ow, const uint32_t* mask,
                          const int* ps, int* ni, float* nw, int ne, cudaStream_t s) {
    if (ne > 0)
        compact_edges_kernel<<<(ne + BLOCK - 1) / BLOCK, BLOCK, 0, s>>>(oi, ow, mask, ps, ni, nw, ne);
}

void launch_build_offsets(const int* oo, const int* ps, int* no, int nv, cudaStream_t s) {
    int n = nv + 1;
    build_offsets_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK, 0, s>>>(oo, ps, no, nv);
}

void launch_init_uniform(float* x, int n, cudaStream_t s) {
    init_uniform_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK, 0, s>>>(x, n);
}

void launch_add_identity_l2(float* y, const float* x, float* l2, int n, cudaStream_t s) {
    int grid = (n + BLOCK - 1) / BLOCK;
    if (grid > 1024) grid = 1024;
    add_identity_l2_kernel<<<grid, BLOCK, 0, s>>>(y, x, l2, n);
}

void launch_normalize_diff(const float* y, const float* xo, float* xn,
                           const float* l2, float* diff, int n, cudaStream_t s) {
    int grid = (n + BLOCK - 1) / BLOCK;
    if (grid > 1024) grid = 1024;
    normalize_diff_kernel<<<grid, BLOCK, 0, s>>>(y, xo, xn, l2, diff, n);
}

size_t get_cub_scan_temp_bytes(int n, const uint32_t* mask_ptr) {
    size_t bytes = 0;
    MaskToFlag op{mask_ptr};
    auto iter = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), op);
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, iter, (int*)nullptr, n);
    return bytes;
}

void launch_cub_scan_mask(void* tmp, size_t bytes, const uint32_t* mask,
                           int* out, int n, cudaStream_t s) {
    MaskToFlag op{mask};
    auto iter = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), op);
    cub::DeviceScan::ExclusiveSum(tmp, bytes, iter, out, n, s);
}





struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;
    cudaStream_t comp_stream = nullptr;
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t event = nullptr;
    float* h_diff_pinned = nullptr;

    int32_t* new_offsets = nullptr;
    int64_t new_offsets_cap = 0;

    float* y = nullptr;
    int64_t y_cap = 0;

    float* x0 = nullptr;
    int64_t x0_cap = 0;

    float* x1 = nullptr;
    int64_t x1_cap = 0;

    float* acc = nullptr;

    int32_t* psum = nullptr;
    int64_t psum_cap = 0;

    int32_t* new_indices = nullptr;
    int64_t new_indices_cap = 0;

    float* new_weights = nullptr;
    int64_t new_weights_cap = 0;

    void* scan_tmp = nullptr;
    size_t scan_tmp_cap = 0;

    void* spmv_buf = nullptr;
    size_t spmv_buf_cap = 0;

    Cache() {
        cusparseCreate(&handle);
        cudaStreamCreate(&comp_stream);
        cudaStreamCreate(&copy_stream);
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        cudaHostAlloc(&h_diff_pinned, sizeof(float), cudaHostAllocDefault);
        cusparseSetStream(handle, comp_stream);
        cudaMalloc(&acc, 2 * sizeof(float));
    }

    ~Cache() override {
        if (acc) cudaFree(acc);
        if (spmv_buf) cudaFree(spmv_buf);
        if (scan_tmp) cudaFree(scan_tmp);
        if (new_weights) cudaFree(new_weights);
        if (new_indices) cudaFree(new_indices);
        if (psum) cudaFree(psum);
        if (x1) cudaFree(x1);
        if (x0) cudaFree(x0);
        if (y) cudaFree(y);
        if (new_offsets) cudaFree(new_offsets);
        if (h_diff_pinned) cudaFreeHost(h_diff_pinned);
        if (event) cudaEventDestroy(event);
        if (copy_stream) cudaStreamDestroy(copy_stream);
        if (comp_stream) cudaStreamDestroy(comp_stream);
        if (handle) cusparseDestroy(handle);
    }
};

}  

eigenvector_centrality_result_t eigenvector_centrality_mask(
    const graph32_t& graph,
    const float* edge_weights,
    float* centralities,
    float epsilon,
    std::size_t max_iterations,
    const float* initial_centralities)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    const float* d_wt = edge_weights;
    const uint32_t* d_mask = graph.edge_mask;

    cudaStream_t st = cache.comp_stream;

    
    int scan_n = ne + 1;

    if (cache.psum_cap < scan_n) {
        if (cache.psum) cudaFree(cache.psum);
        cudaMalloc(&cache.psum, (size_t)scan_n * sizeof(int32_t));
        cache.psum_cap = scan_n;
    }

    size_t scan_bytes = get_cub_scan_temp_bytes(scan_n, d_mask);
    if (cache.scan_tmp_cap < scan_bytes) {
        if (cache.scan_tmp) cudaFree(cache.scan_tmp);
        size_t alloc_bytes = scan_bytes > 0 ? scan_bytes : 1;
        cudaMalloc(&cache.scan_tmp, alloc_bytes);
        cache.scan_tmp_cap = alloc_bytes;
    }

    launch_cub_scan_mask(cache.scan_tmp, scan_bytes, d_mask, cache.psum, scan_n, st);

    int32_t total_active;
    cudaEventRecord(cache.event, cache.comp_stream);
    cudaStreamWaitEvent(cache.copy_stream, cache.event, 0);
    cudaMemcpyAsync(&total_active, cache.psum + ne,
                    sizeof(int32_t), cudaMemcpyDeviceToHost, cache.copy_stream);
    cudaStreamSynchronize(cache.copy_stream);

    int32_t alloc_ne = std::max(total_active, 1);

    int64_t nv1 = (int64_t)nv + 1;
    if (cache.new_offsets_cap < nv1) {
        if (cache.new_offsets) cudaFree(cache.new_offsets);
        cudaMalloc(&cache.new_offsets, (size_t)nv1 * sizeof(int32_t));
        cache.new_offsets_cap = nv1;
    }
    if (cache.y_cap < nv) {
        if (cache.y) cudaFree(cache.y);
        cudaMalloc(&cache.y, (size_t)nv * sizeof(float));
        cache.y_cap = nv;
    }
    if (cache.x0_cap < nv) {
        if (cache.x0) cudaFree(cache.x0);
        cudaMalloc(&cache.x0, (size_t)nv * sizeof(float));
        cache.x0_cap = nv;
    }
    if (cache.x1_cap < nv) {
        if (cache.x1) cudaFree(cache.x1);
        cudaMalloc(&cache.x1, (size_t)nv * sizeof(float));
        cache.x1_cap = nv;
    }

    if (cache.new_indices_cap < alloc_ne) {
        if (cache.new_indices) cudaFree(cache.new_indices);
        cudaMalloc(&cache.new_indices, (size_t)alloc_ne * sizeof(int32_t));
        cache.new_indices_cap = alloc_ne;
    }
    if (cache.new_weights_cap < alloc_ne) {
        if (cache.new_weights) cudaFree(cache.new_weights);
        cudaMalloc(&cache.new_weights, (size_t)alloc_ne * sizeof(float));
        cache.new_weights_cap = alloc_ne;
    }

    launch_build_offsets(d_off, cache.psum, cache.new_offsets, nv, st);
    launch_compact_edges(d_idx, d_wt, d_mask, cache.psum,
                        cache.new_indices, cache.new_weights, ne, st);

    
    cusparseSpMatDescr_t matA = nullptr;
    cusparseCreateCsr(&matA, nv, nv, total_active,
        cache.new_offsets, cache.new_indices, cache.new_weights,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    float* d_y = cache.y;
    float* d_x0 = cache.x0;
    float* d_x1 = cache.x1;
    float* d_l2 = cache.acc;
    float* d_diff = cache.acc + 1;

    float* d_x_old = d_x0;
    float* d_x_new = d_x1;

    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(d_x_old, initial_centralities,
                       (size_t)nv * sizeof(float), cudaMemcpyDeviceToDevice, st);
    } else {
        launch_init_uniform(d_x_old, nv, st);
    }

    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    cusparseCreateDnVec(&vecX, nv, d_x_old, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, nv, d_y, CUDA_R_32F);

    float alpha = 1.0f, beta = 0.0f;
    size_t buf_size = 0;
    cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &buf_size);

    if (cache.spmv_buf_cap < buf_size) {
        if (cache.spmv_buf) cudaFree(cache.spmv_buf);
        size_t alloc_bytes = buf_size > 0 ? buf_size : 1;
        cudaMalloc(&cache.spmv_buf, alloc_bytes);
        cache.spmv_buf_cap = alloc_bytes;
    }

    cusparseSpMV_preprocess(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

    
    float threshold = (float)nv * epsilon;
    std::size_t iter = 0;
    bool converged = false;
    bool first_iter = true;

    for (; iter < max_iterations; ++iter) {
        if (!first_iter) {
            cudaStreamSynchronize(cache.copy_stream);
            if (*cache.h_diff_pinned < threshold) {
                converged = true;
                break;
            }
        }

        cusparseDnVecSetValues(vecX, d_x_old);
        cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

        cudaMemsetAsync(d_l2, 0, 2 * sizeof(float), st);
        launch_add_identity_l2(d_y, d_x_old, d_l2, nv, st);
        launch_normalize_diff(d_y, d_x_old, d_x_new, d_l2, d_diff, nv, st);

        cudaEventRecord(cache.event, cache.comp_stream);
        cudaStreamWaitEvent(cache.copy_stream, cache.event, 0);
        cudaMemcpyAsync(cache.h_diff_pinned, d_diff, sizeof(float),
                       cudaMemcpyDeviceToHost, cache.copy_stream);

        std::swap(d_x_old, d_x_new);
        first_iter = false;
    }

    if (!converged) {
        cudaStreamSynchronize(cache.copy_stream);
        if (*cache.h_diff_pinned < threshold) {
            converged = true;
        }
    }

    cudaStreamSynchronize(cache.comp_stream);

    
    cudaMemcpy(centralities, d_x_old, (size_t)nv * sizeof(float), cudaMemcpyDeviceToDevice);

    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(matA);

    return {iter, converged};
}

}  
