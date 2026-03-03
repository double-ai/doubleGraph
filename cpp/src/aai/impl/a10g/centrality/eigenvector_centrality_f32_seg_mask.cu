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





__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ mask,
    int32_t* __restrict__ counts,
    int32_t nv)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv) return;

    int32_t begin = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t count = 0;
    for (int32_t e = begin; e < end; e++) {
        count += (mask[e >> 5] >> (e & 31)) & 1u;
    }
    counts[v] = count;
}

__global__ void compact_edges_warp_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ new_offsets,
    const int32_t* __restrict__ old_indices,
    int32_t* __restrict__ new_indices,
    const float* __restrict__ old_weights,
    float* __restrict__ new_weights,
    const uint32_t* __restrict__ mask,
    int32_t nv)
{
    int32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t v = gtid >> 5;
    int32_t lane = gtid & 31;
    if (v >= nv) return;

    int32_t old_begin = old_offsets[v];
    int32_t old_end = old_offsets[v + 1];
    int32_t new_begin = new_offsets[v];
    int32_t write_pos = 0;

    for (int32_t base = old_begin; base < old_end; base += 32) {
        int32_t e = base + lane;
        bool active = (e < old_end) && ((mask[e >> 5] >> (e & 31)) & 1u);

        unsigned active_bits = __ballot_sync(0xffffffff, active);
        int pos_in_warp = __popc(active_bits & ((1u << lane) - 1));

        if (active) {
            int32_t wp = new_begin + write_pos + pos_in_warp;
            new_indices[wp] = old_indices[e];
            new_weights[wp] = old_weights[e];
        }
        write_pos += __popc(active_bits);
    }
}






__global__ void l2_norm_sq_identity_kernel(
    const float* __restrict__ x_new,
    const float* __restrict__ x_old,
    float* __restrict__ result,
    int32_t n)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;

    float s = 0.0f;
    for (int32_t i = tid; i < n; i += stride) {
        float v = x_new[i] + x_old[i];
        s += v * v;
    }

    #pragma unroll
    for (int d = 16; d > 0; d >>= 1)
        s += __shfl_down_sync(0xffffffff, s, d);

    if ((threadIdx.x & 31) == 0)
        atomicAdd(result, s);
}

__global__ void normalize_diff_identity_kernel(
    float* __restrict__ x_new,
    const float* __restrict__ x_old,
    const float* __restrict__ l2_sq,
    float* __restrict__ diff_out,
    int32_t n)
{
    float norm_sq = *l2_sq;
    float inv_norm = (norm_sq > 0.0f) ? rsqrtf(norm_sq) : 0.0f;

    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;

    float local_diff = 0.0f;
    for (int32_t i = tid; i < n; i += stride) {
        float val = (x_new[i] + x_old[i]) * inv_norm;
        x_new[i] = val;
        local_diff += fabsf(val - x_old[i]);
    }

    #pragma unroll
    for (int d = 16; d > 0; d >>= 1)
        local_diff += __shfl_down_sync(0xffffffff, local_diff, d);

    if ((threadIdx.x & 31) == 0)
        atomicAdd(diff_out, local_diff);
}

__global__ void init_kernel(float* x, float val, int32_t n) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = val;
}





struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    float* d_l2 = nullptr;
    float* d_diff = nullptr;
    float* h_diff = nullptr;
    float* d_one = nullptr;
    float* d_zero = nullptr;
    void* d_scratch = nullptr;
    size_t scratch_size = 0;

    
    int32_t* d_counts = nullptr;
    int64_t counts_capacity = 0;

    int32_t* d_new_offsets = nullptr;
    int64_t new_offsets_capacity = 0;

    int32_t* d_new_indices = nullptr;
    int64_t new_indices_capacity = 0;

    float* d_new_weights = nullptr;
    int64_t new_weights_capacity = 0;

    float* d_buf0 = nullptr;
    int64_t buf0_capacity = 0;

    float* d_buf1 = nullptr;
    int64_t buf1_capacity = 0;

    void* d_spmv_buf = nullptr;
    size_t spmv_buf_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);

        cudaMalloc(&d_l2, sizeof(float));
        cudaMalloc(&d_diff, sizeof(float));
        cudaMallocHost(&h_diff, sizeof(float));

        cudaMalloc(&d_one, sizeof(float));
        cudaMalloc(&d_zero, sizeof(float));
        float one = 1.0f, zero = 0.0f;
        cudaMemcpy(d_one, &one, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_zero, &zero, sizeof(float), cudaMemcpyHostToDevice);

        scratch_size = 32 * 1024 * 1024;
        cudaMalloc(&d_scratch, scratch_size);
    }

    void ensure(int32_t nv, int32_t ne) {
        int64_t nv1 = (int64_t)nv + 1;
        if (counts_capacity < nv1) {
            if (d_counts) cudaFree(d_counts);
            cudaMalloc(&d_counts, nv1 * sizeof(int32_t));
            counts_capacity = nv1;
        }
        if (new_offsets_capacity < nv1) {
            if (d_new_offsets) cudaFree(d_new_offsets);
            cudaMalloc(&d_new_offsets, nv1 * sizeof(int32_t));
            new_offsets_capacity = nv1;
        }
        if (new_indices_capacity < (int64_t)ne) {
            if (d_new_indices) cudaFree(d_new_indices);
            cudaMalloc(&d_new_indices, (int64_t)ne * sizeof(int32_t));
            new_indices_capacity = ne;
        }
        if (new_weights_capacity < (int64_t)ne) {
            if (d_new_weights) cudaFree(d_new_weights);
            cudaMalloc(&d_new_weights, (int64_t)ne * sizeof(float));
            new_weights_capacity = ne;
        }
        if (buf0_capacity < (int64_t)nv) {
            if (d_buf0) cudaFree(d_buf0);
            cudaMalloc(&d_buf0, (int64_t)nv * sizeof(float));
            buf0_capacity = nv;
        }
        if (buf1_capacity < (int64_t)nv) {
            if (d_buf1) cudaFree(d_buf1);
            cudaMalloc(&d_buf1, (int64_t)nv * sizeof(float));
            buf1_capacity = nv;
        }
    }

    void ensure_spmv_buf(size_t needed) {
        if (spmv_buf_capacity < needed) {
            if (d_spmv_buf) cudaFree(d_spmv_buf);
            cudaMalloc(&d_spmv_buf, needed);
            spmv_buf_capacity = needed;
        }
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (d_l2) cudaFree(d_l2);
        if (d_diff) cudaFree(d_diff);
        if (h_diff) cudaFreeHost(h_diff);
        if (d_one) cudaFree(d_one);
        if (d_zero) cudaFree(d_zero);
        if (d_scratch) cudaFree(d_scratch);
        if (d_counts) cudaFree(d_counts);
        if (d_new_offsets) cudaFree(d_new_offsets);
        if (d_new_indices) cudaFree(d_new_indices);
        if (d_new_weights) cudaFree(d_new_weights);
        if (d_buf0) cudaFree(d_buf0);
        if (d_buf1) cudaFree(d_buf1);
        if (d_spmv_buf) cudaFree(d_spmv_buf);
    }
};

}  

eigenvector_centrality_result_t eigenvector_centrality_seg_mask(const graph32_t& graph,
                                      const float* edge_weights,
                                      float* centralities,
                                      float epsilon,
                                      std::size_t max_iterations,
                                      const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    const uint32_t* edge_mask = graph.edge_mask;

    float threshold = (float)nv * epsilon;
    cudaStream_t stream = 0;
    cusparseSetStream(cache.cusparse_handle, stream);

    cache.ensure(nv, ne);

    
    int blocks = (nv + 255) / 256;
    if (blocks > 0)
        count_active_edges_kernel<<<blocks, 256, 0, stream>>>(offsets, edge_mask, cache.d_counts, nv);
    cudaMemsetAsync(cache.d_counts + nv, 0, sizeof(int32_t), stream);

    
    size_t cub_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_temp_bytes, (int32_t*)nullptr, (int32_t*)nullptr, nv + 1);
    void* cub_temp = cache.d_scratch;
    void* cub_extra = nullptr;
    if (cub_temp_bytes > cache.scratch_size) {
        cudaMalloc(&cub_extra, cub_temp_bytes);
        cub_temp = cub_extra;
    }

    cub::DeviceScan::ExclusiveSum(cub_temp, cub_temp_bytes, cache.d_counts, cache.d_new_offsets, nv + 1, stream);

    if (cub_extra) cudaFree(cub_extra);

    
    {
        int warps_per_block = 256 / 32;
        int cblocks = (nv + warps_per_block - 1) / warps_per_block;
        if (cblocks > 0)
            compact_edges_warp_kernel<<<cblocks, 256, 0, stream>>>(
                offsets, cache.d_new_offsets, indices, cache.d_new_indices,
                edge_weights, cache.d_new_weights, edge_mask, nv);
    }

    
    int32_t total_active;
    cudaMemcpyAsync(&total_active, cache.d_new_offsets + nv, sizeof(int32_t),
                   cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (total_active <= 0) total_active = 1;

    
    cusparseSpMatDescr_t mat;
    cusparseCreateCsr(&mat, nv, nv, total_active,
        (void*)cache.d_new_offsets, (void*)cache.d_new_indices, (void*)cache.d_new_weights,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    float* buf[2] = {cache.d_buf0, cache.d_buf1};

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, nv, buf[0], CUDA_R_32F);
    cusparseCreateDnVec(&vecY, nv, buf[1], CUDA_R_32F);

    
    size_t spmv_buf_size = 0;
    cusparseSpMV_bufferSize(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        cache.d_one, mat, vecX, cache.d_zero, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buf_size);

    void* spmv_buf = cache.d_scratch;
    if (spmv_buf_size > cache.scratch_size) {
        cache.ensure_spmv_buf(spmv_buf_size);
        spmv_buf = cache.d_spmv_buf;
    }

    cusparseSpMV_preprocess(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        cache.d_one, mat, vecX, cache.d_zero, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, spmv_buf);

    
    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(buf[0], initial_centralities,
                       nv * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        float init_val = 1.0f / (float)nv;
        int iblocks = (nv + 255) / 256;
        init_kernel<<<iblocks, 256, 0, stream>>>(buf[0], init_val, nv);
    }

    
    bool converged = false;
    size_t iterations = 0;
    int cur = 0;

    for (size_t i = 0; i < max_iterations; i++) {
        int nxt = 1 - cur;

        cusparseDnVecSetValues(vecX, buf[cur]);
        cusparseDnVecSetValues(vecY, buf[nxt]);

        
        cusparseSpMV(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_one, mat, vecX, cache.d_zero, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, spmv_buf);

        
        cudaMemsetAsync(cache.d_l2, 0, sizeof(float), stream);
        {
            int nblocks = (nv + 255) / 256;
            if (nblocks > 1024) nblocks = 1024;
            l2_norm_sq_identity_kernel<<<nblocks, 256, 0, stream>>>(buf[nxt], buf[cur], cache.d_l2, nv);
        }

        
        cudaMemsetAsync(cache.d_diff, 0, sizeof(float), stream);
        {
            int nblocks = (nv + 255) / 256;
            if (nblocks > 1024) nblocks = 1024;
            normalize_diff_identity_kernel<<<nblocks, 256, 0, stream>>>(buf[nxt], buf[cur], cache.d_l2, cache.d_diff, nv);
        }

        
        cudaMemcpyAsync(cache.h_diff, cache.d_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations = i + 1;
        if (*cache.h_diff < threshold) {
            converged = true;
            cur = nxt;
            break;
        }
        cur = nxt;
    }

    cusparseDestroySpMat(mat);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    
    cudaMemcpyAsync(centralities, buf[cur], nv * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return {iterations, converged};
}

}  
