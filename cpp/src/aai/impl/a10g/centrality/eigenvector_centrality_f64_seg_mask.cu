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





struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;
    float* h_conv = nullptr;

    int32_t* scan = nullptr;
    int64_t scan_cap = 0;

    void* cub_tmp = nullptr;
    size_t cub_tmp_cap = 0;

    int32_t* new_off = nullptr;
    int64_t new_off_cap = 0;

    int32_t* new_idx = nullptr;
    int64_t new_idx_cap = 0;

    float* new_wt = nullptr;
    int64_t new_wt_cap = 0;

    float* ba = nullptr;
    int64_t ba_cap = 0;

    float* bb = nullptr;
    int64_t bb_cap = 0;

    float* ac = nullptr;

    void* spmv_buf = nullptr;
    size_t spmv_cap = 0;

    Cache() {
        cusparseCreate(&handle);
        cudaMallocHost(&h_conv, sizeof(float));
        cudaMalloc(&ac, 2 * sizeof(float));
    }

    ~Cache() override {
        if (handle) cusparseDestroy(handle);
        if (h_conv) cudaFreeHost(h_conv);
        if (scan) cudaFree(scan);
        if (cub_tmp) cudaFree(cub_tmp);
        if (new_off) cudaFree(new_off);
        if (new_idx) cudaFree(new_idx);
        if (new_wt) cudaFree(new_wt);
        if (ba) cudaFree(ba);
        if (bb) cudaFree(bb);
        if (ac) cudaFree(ac);
        if (spmv_buf) cudaFree(spmv_buf);
    }

    void ensure_scan(int64_t n) {
        if (scan_cap < n) {
            if (scan) cudaFree(scan);
            cudaMalloc(&scan, n * sizeof(int32_t));
            scan_cap = n;
        }
    }

    void ensure_cub(size_t sz) {
        if (cub_tmp_cap < sz) {
            if (cub_tmp) cudaFree(cub_tmp);
            cudaMalloc(&cub_tmp, sz);
            cub_tmp_cap = sz;
        }
    }

    void ensure_new_off(int64_t n) {
        if (new_off_cap < n) {
            if (new_off) cudaFree(new_off);
            cudaMalloc(&new_off, n * sizeof(int32_t));
            new_off_cap = n;
        }
    }

    void ensure_new_idx(int64_t n) {
        if (new_idx_cap < n) {
            if (new_idx) cudaFree(new_idx);
            cudaMalloc(&new_idx, n * sizeof(int32_t));
            new_idx_cap = n;
        }
    }

    void ensure_new_wt(int64_t n) {
        if (new_wt_cap < n) {
            if (new_wt) cudaFree(new_wt);
            cudaMalloc(&new_wt, n * sizeof(float));
            new_wt_cap = n;
        }
    }

    void ensure_ba(int64_t n) {
        if (ba_cap < n) {
            if (ba) cudaFree(ba);
            cudaMalloc(&ba, n * sizeof(float));
            ba_cap = n;
        }
    }

    void ensure_bb(int64_t n) {
        if (bb_cap < n) {
            if (bb) cudaFree(bb);
            cudaMalloc(&bb, n * sizeof(float));
            bb_cap = n;
        }
    }

    void ensure_spmv(size_t sz) {
        if (spmv_cap < sz) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, sz);
            spmv_cap = sz;
        }
    }
};





__global__ void compute_flags_kernel(
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ flags, int32_t ne)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx <= ne; idx += stride)
        flags[idx] = (idx < ne) ? ((edge_mask[idx >> 5] >> (idx & 31)) & 1) : 0;
}

__global__ void compact_kernel(
    const int32_t* __restrict__ scan,
    const int32_t* __restrict__ orig_offsets,
    const int32_t* __restrict__ orig_indices,
    const double* __restrict__ orig_weights,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    float* __restrict__ new_weights,
    int32_t ne, int32_t nv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int e = idx; e < ne; e += stride) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1) {
            int32_t pos = scan[e];
            new_indices[pos] = orig_indices[e];
            new_weights[pos] = __double2float_rn(orig_weights[e]);
        }
    }
    for (int v = idx; v <= nv; v += stride)
        new_offsets[v] = scan[orig_offsets[v]];
}





__global__ void init_uniform_kernel(float* __restrict__ x, int32_t n, float val)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        x[i] = val;
}

__global__ void d2f_kernel(const double* __restrict__ src, float* __restrict__ dst, int32_t n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        dst[i] = __double2float_rn(src[i]);
}

__global__ void add_identity_l2norm_kernel(
    float* __restrict__ y,
    const float* __restrict__ x_old,
    float* __restrict__ norm_sq,
    int32_t n)
{
    constexpr int WARPS = 8;
    __shared__ float s_sum[WARPS];

    float local = 0.0f;

    int n4 = n >> 2;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride4 = blockDim.x * gridDim.x;

    float4* y4 = reinterpret_cast<float4*>(y);
    const float4* x4 = reinterpret_cast<const float4*>(x_old);

    for (int i = idx4; i < n4; i += stride4) {
        float4 yv = y4[i];
        float4 xv = x4[i];
        yv.x += xv.x; yv.y += xv.y; yv.z += xv.z; yv.w += xv.w;
        y4[i] = yv;
        local = fmaf(yv.x, yv.x, local);
        local = fmaf(yv.y, yv.y, local);
        local = fmaf(yv.z, yv.z, local);
        local = fmaf(yv.w, yv.w, local);
    }

    for (int i = n4 * 4 + (blockIdx.x * blockDim.x + threadIdx.x); i < n; i += blockDim.x * gridDim.x) {
        float v = y[i] + x_old[i];
        y[i] = v;
        local = fmaf(v, v, local);
    }

    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(0xffffffff, local, off);

    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    if (lane_id == 0) s_sum[warp_id] = local;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < WARPS) ? s_sum[lane_id] : 0.0f;
        for (int off = WARPS/2; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (lane_id == 0)
            atomicAdd(norm_sq, val);
    }
}

__global__ void normalize_l1diff_kernel(
    float* __restrict__ y,
    const float* __restrict__ x_old,
    const float* __restrict__ norm_sq,
    float* __restrict__ l1_diff,
    int32_t n)
{
    constexpr int WARPS = 8;
    __shared__ float s_diff[WARPS];

    float inv_norm = rsqrtf(*norm_sq);
    float local = 0.0f;

    int n4 = n >> 2;
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride4 = blockDim.x * gridDim.x;

    float4* y4 = reinterpret_cast<float4*>(y);
    const float4* x4 = reinterpret_cast<const float4*>(x_old);

    for (int i = idx4; i < n4; i += stride4) {
        float4 yv = y4[i];
        float4 xv = x4[i];
        yv.x *= inv_norm; yv.y *= inv_norm; yv.z *= inv_norm; yv.w *= inv_norm;
        local += fabsf(yv.x - xv.x) + fabsf(yv.y - xv.y) +
                 fabsf(yv.z - xv.z) + fabsf(yv.w - xv.w);
        y4[i] = yv;
    }

    for (int i = n4 * 4 + (blockIdx.x * blockDim.x + threadIdx.x); i < n; i += blockDim.x * gridDim.x) {
        float nv = y[i] * inv_norm;
        local += fabsf(nv - x_old[i]);
        y[i] = nv;
    }

    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(0xffffffff, local, off);

    int warp_id = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    if (lane_id == 0) s_diff[warp_id] = local;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane_id < WARPS) ? s_diff[lane_id] : 0.0f;
        for (int off = WARPS/2; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
        if (lane_id == 0)
            atomicAdd(l1_diff, val);
    }
}

__global__ void f2d_kernel(const float* __restrict__ src, double* __restrict__ dst, int32_t n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        dst[i] = (double)src[i];
}

}  

eigenvector_centrality_result_t eigenvector_centrality_seg_mask(
    const graph32_t& graph,
    const double* edge_weights,
    double* centralities,
    double epsilon,
    std::size_t max_iterations,
    const double* initial_centralities)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    const uint32_t* d_mask = graph.edge_mask;

    cudaStream_t stream = 0;
    cusparseSetStream(cache.handle, stream);

    
    cache.ensure_new_off((int64_t)(nv + 1));
    cache.ensure_ba((int64_t)nv);
    cache.ensure_bb((int64_t)nv);

    
    int32_t compact_ne = 0;

    if (ne > 0) {
        cache.ensure_scan((int64_t)(ne + 1));

        {
            int B = 256, G = min(((ne + 1) + B - 1) / B, 4096);
            compute_flags_kernel<<<G, B, 0, stream>>>(d_mask, cache.scan, ne);
        }

        size_t cub_tmp_sz = 0;
        cub::DeviceScan::ExclusiveSum((void*)nullptr, cub_tmp_sz,
                                      (int32_t*)nullptr, (int32_t*)nullptr, ne + 1);
        cache.ensure_cub(cub_tmp_sz > 0 ? cub_tmp_sz : 1);

        cub::DeviceScan::ExclusiveSum(cache.cub_tmp, cub_tmp_sz,
                                      cache.scan, cache.scan, ne + 1, stream);

        int32_t h_cne;
        cudaMemcpyAsync(&h_cne, cache.scan + ne, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        compact_ne = h_cne;

        int64_t idx_sz = compact_ne > 0 ? (int64_t)compact_ne : (int64_t)1;
        cache.ensure_new_idx(idx_sz);
        cache.ensure_new_wt(idx_sz);

        {
            int B = 256, G = min((max(ne, nv + 1) + B - 1) / B, 4096);
            compact_kernel<<<G, B, 0, stream>>>(
                cache.scan, offsets, indices, edge_weights, d_mask,
                cache.new_off, cache.new_idx, cache.new_wt, ne, nv);
        }
    } else {
        cache.ensure_new_idx(1);
        cache.ensure_new_wt(1);
        cudaMemsetAsync(cache.new_off, 0, (nv + 1) * sizeof(int32_t), stream);
    }

    
    float* d_norm_sq = cache.ac;
    float* d_l1_diff = cache.ac + 1;

    if (initial_centralities != nullptr) {
        int B = 256, G = min((nv + B - 1) / B, 4096);
        if (G > 0) d2f_kernel<<<G, B, 0, stream>>>(initial_centralities, cache.ba, nv);
    } else {
        int B = 256, G = min((nv + B - 1) / B, 4096);
        if (G > 0) init_uniform_kernel<<<G, B, 0, stream>>>(cache.ba, nv, 1.0f / nv);
    }

    
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA,
        (int64_t)nv, (int64_t)nv, (int64_t)compact_ne,
        (void*)cache.new_off,
        (void*)cache.new_idx,
        (void*)cache.new_wt,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, (int64_t)nv, cache.ba, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, (int64_t)nv, cache.bb, CUDA_R_32F);

    float alpha = 1.0f, beta_val = 0.0f;
    size_t bufSz = 0;
    cusparseSpMV_bufferSize(cache.handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta_val, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSz);

    cache.ensure_spmv(bufSz > 0 ? bufSz : 1);

    cusparseSpMV_preprocess(cache.handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta_val, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

    
    float threshold = (float)((double)nv * epsilon);
    size_t iterations = 0;
    bool converged = false;
    float* bufs[2] = {cache.ba, cache.bb};
    int cur = 0;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        int src = cur, dst = 1 - cur;

        cusparseDnVecSetValues(vecX, bufs[src]);
        cusparseDnVecSetValues(vecY, bufs[dst]);

        cusparseSpMV(cache.handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta_val, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

        cudaMemsetAsync(d_norm_sq, 0, 2 * sizeof(float), stream);

        {
            int B = 256, G = min((nv / 4 + B - 1) / B, 256);
            if (G < 1) G = 1;
            add_identity_l2norm_kernel<<<G, B, 0, stream>>>(bufs[dst], bufs[src], d_norm_sq, nv);
        }
        {
            int B = 256, G = min((nv / 4 + B - 1) / B, 256);
            if (G < 1) G = 1;
            normalize_l1diff_kernel<<<G, B, 0, stream>>>(bufs[dst], bufs[src], d_norm_sq, d_l1_diff, nv);
        }

        cudaMemcpyAsync(cache.h_conv, d_l1_diff, sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations++;
        cur = dst;

        if (*cache.h_conv < threshold) {
            converged = true;
            break;
        }
    }

    
    {
        int B = 256, G = min((nv + B - 1) / B, 4096);
        if (G > 0) f2d_kernel<<<G, B, 0, stream>>>(bufs[cur], centralities, nv);
    }

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    return {iterations, converged};
}

}  
