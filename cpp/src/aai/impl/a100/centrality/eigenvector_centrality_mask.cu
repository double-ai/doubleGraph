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
#include <algorithm>

namespace aai {

namespace {

constexpr int BLOCK_SIZE = 256;

struct Cache : Cacheable {
    cusparseHandle_t cusparse = nullptr;
    float* h_diff = nullptr;

    float* d_scratch = nullptr;
    float* d_accum = nullptr;
    int32_t* d_coff = nullptr;
    int32_t* d_counts = nullptr;
    void* d_scan_tmp = nullptr;
    size_t scan_sz = 0;
    size_t an = 0;

    int32_t* d_cidx = nullptr;
    float* d_vals = nullptr;
    size_t ae = 0;

    void* d_spmv_buf = nullptr;
    size_t spmv_sz = 0;

    Cache() {
        cusparseCreate(&cusparse);
        cudaMallocHost(&h_diff, sizeof(float));
    }

    ~Cache() override {
        if (cusparse) { cusparseDestroy(cusparse); cusparse = nullptr; }
        if (d_scratch) { cudaFree(d_scratch); d_scratch = nullptr; }
        if (d_accum) { cudaFree(d_accum); d_accum = nullptr; }
        if (d_coff) { cudaFree(d_coff); d_coff = nullptr; }
        if (d_cidx) { cudaFree(d_cidx); d_cidx = nullptr; }
        if (d_counts) { cudaFree(d_counts); d_counts = nullptr; }
        if (d_vals) { cudaFree(d_vals); d_vals = nullptr; }
        if (d_spmv_buf) { cudaFree(d_spmv_buf); d_spmv_buf = nullptr; }
        if (d_scan_tmp) { cudaFree(d_scan_tmp); d_scan_tmp = nullptr; }
        if (h_diff) { cudaFreeHost(h_diff); h_diff = nullptr; }
    }

    void ensure(int32_t n, int32_t e) {
        if ((size_t)n > an) {
            if (d_scratch) cudaFree(d_scratch);
            if (d_accum) cudaFree(d_accum);
            if (d_coff) cudaFree(d_coff);
            if (d_counts) cudaFree(d_counts);
            cudaMalloc(&d_scratch, (size_t)n * sizeof(float));
            cudaMalloc(&d_accum, 2 * sizeof(float));
            cudaMalloc(&d_coff, ((size_t)n + 1) * sizeof(int32_t));
            cudaMalloc(&d_counts, (size_t)n * sizeof(int32_t));
            size_t need = 0;
            cub::DeviceScan::InclusiveSum(nullptr, need, d_counts, d_coff + 1, (int)n);
            if (need > scan_sz) {
                if (d_scan_tmp) cudaFree(d_scan_tmp);
                cudaMalloc(&d_scan_tmp, need);
                scan_sz = need;
            }
            an = n;
        }
        if ((size_t)e > ae) {
            if (d_cidx) cudaFree(d_cidx);
            if (d_vals) cudaFree(d_vals);
            cudaMalloc(&d_cidx, (size_t)e * sizeof(int32_t));
            cudaMalloc(&d_vals, (size_t)e * sizeof(float));
            ae = e;
        }
    }
};



__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ counts,
    int32_t num_vertices)
{
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_vertices;
         j += gridDim.x * blockDim.x) {
        int32_t start = offsets[j], end = offsets[j + 1], count = 0;
        for (int32_t k = start; k < end;) {
            int32_t wi = k >> 5, bs = k & 31;
            int32_t we = (((wi + 1) << 5) < end) ? ((wi + 1) << 5) : end;
            int32_t bits = we - k;
            uint32_t mw = edge_mask[wi] >> bs;
            if (bits < 32) mw &= (1u << bits) - 1u;
            count += __popc(mw);
            k = we;
        }
        counts[j] = count;
    }
}

__global__ void compact_indices_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t num_vertices)
{
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_vertices;
         j += gridDim.x * blockDim.x) {
        int32_t old_start = old_offsets[j], old_end = old_offsets[j + 1];
        int32_t np = new_offsets[j];
        for (int32_t k = old_start; k < old_end;) {
            int32_t wi = k >> 5, bs = k & 31;
            int32_t we = (((wi + 1) << 5) < old_end) ? ((wi + 1) << 5) : old_end;
            int32_t bits = we - k;
            uint32_t mw = edge_mask[wi] >> bs;
            if (bits < 32) mw &= (1u << bits) - 1u;
            while (mw != 0) {
                int bit = __ffs(mw) - 1;
                new_indices[np++] = indices[k + bit];
                mw &= mw - 1;
            }
            k = we;
        }
    }
}



__global__ void spmv_compact_norm_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ norm_sq_out,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float tnorm = 0.0f;

    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_vertices;
         j += gridDim.x * blockDim.x) {
        float sum = x[j]; 
        int32_t start = offsets[j], end = offsets[j + 1];
        for (int32_t k = start; k < end; k++)
            sum += x[indices[k]];
        y[j] = sum;
        tnorm += sum * sum;
    }

    float bs = BlockReduce(temp_storage).Sum(tnorm);
    if (threadIdx.x == 0) atomicAdd(norm_sq_out, bs);
}




__global__ void add_identity_norm_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    float* __restrict__ norm_sq_out,
    int32_t n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float tnorm = 0.0f;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        float val = y[j] + x[j];
        y[j] = val;
        tnorm += val * val;
    }
    float bs = BlockReduce(temp_storage).Sum(tnorm);
    if (threadIdx.x == 0) atomicAdd(norm_sq_out, bs);
}


__global__ void normalize_diff_kernel(
    float* __restrict__ y,
    const float* __restrict__ x_old,
    const float* __restrict__ norm_sq_in,
    float* __restrict__ diff_out,
    int32_t n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float inv_norm = (*norm_sq_in > 0.0f) ? rsqrtf(*norm_sq_in) : 0.0f;
    float td = 0.0f;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) {
        float val = y[j] * inv_norm;
        td += fabsf(val - x_old[j]);
        y[j] = val;
    }
    float bd = BlockReduce(temp_storage).Sum(td);
    if (threadIdx.x == 0) atomicAdd(diff_out, bd);
}


__global__ void normalize_only_kernel(
    float* __restrict__ y,
    const float* __restrict__ norm_sq_in,
    int32_t n)
{
    float inv_norm = (*norm_sq_in > 0.0f) ? rsqrtf(*norm_sq_in) : 0.0f;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x)
        y[j] *= inv_norm;
}

__global__ void fill_kernel(float* x, float val, int32_t n) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += gridDim.x * blockDim.x) x[j] = val;
}

}  

eigenvector_centrality_result_t eigenvector_centrality_mask(const graph32_t& graph,
                                  float* centralities,
                                  float epsilon,
                                  std::size_t max_iterations,
                                  const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t n = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    const uint32_t* d_mask = graph.edge_mask;

    if (n == 0) {
        return eigenvector_centrality_result_t{0, true};
    }

    cache.ensure(n, ne);
    cudaStream_t stream = 0;
    cusparseSetStream(cache.cusparse, stream);
    int gv = std::min((n + 255) / 256, 4096);

    
    count_active_edges_kernel<<<gv, BLOCK_SIZE, 0, stream>>>(d_off, d_mask, cache.d_counts, n);
    cudaMemsetAsync(cache.d_coff, 0, sizeof(int32_t), stream);
    size_t ssz = cache.scan_sz;
    cub::DeviceScan::InclusiveSum(cache.d_scan_tmp, ssz, cache.d_counts, cache.d_coff + 1, n, stream);
    int32_t h_total = 0;
    cudaMemcpyAsync(&h_total, cache.d_coff + n, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    compact_indices_kernel<<<gv, BLOCK_SIZE, 0, stream>>>(d_off, d_idx, d_mask, cache.d_coff, cache.d_cidx, n);

    
    bool use_cusparse = (h_total > 200000);

    cusparseSpMatDescr_t mat = nullptr;
    cusparseDnVecDescr_t vx = nullptr, vy = nullptr;
    float ha = 1.0f, hb = 0.0f;

    if (use_cusparse) {
        
        int vg = std::min((h_total + 255) / 256, 2048);
        fill_kernel<<<vg, BLOCK_SIZE, 0, stream>>>(cache.d_vals, 1.0f, h_total);

        cusparseCreateCsr(&mat, n, n, h_total,
                          cache.d_coff, cache.d_cidx, cache.d_vals,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        float* buf0 = centralities;
        float* buf1 = cache.d_scratch;

        cusparseCreateDnVec(&vx, n, buf0, CUDA_R_32F);
        cusparseCreateDnVec(&vy, n, buf1, CUDA_R_32F);

        
        size_t spmv_need = 0;
        cusparseSpMV_bufferSize(cache.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &ha, mat, vx, &hb, vy, CUDA_R_32F,
                                CUSPARSE_SPMV_ALG_DEFAULT, &spmv_need);
        if (spmv_need > cache.spmv_sz) {
            if (cache.d_spmv_buf) cudaFree(cache.d_spmv_buf);
            cudaMalloc(&cache.d_spmv_buf, spmv_need);
            cache.spmv_sz = spmv_need;
        }

        
        cusparseSpMV_preprocess(cache.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &ha, mat, vx, &hb, vy, CUDA_R_32F,
                                CUSPARSE_SPMV_ALG_DEFAULT, cache.d_spmv_buf);

        
        if (initial_centralities != nullptr) {
            cudaMemcpyAsync(buf0, initial_centralities,
                           (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        } else {
            fill_kernel<<<gv, BLOCK_SIZE, 0, stream>>>(buf0, 1.0f / (float)n, n);
        }

        
        float* buf[2] = {buf0, buf1};
        int cur = 0;
        size_t iters = 0;
        bool conv = false;
        float thr = (float)n * epsilon;
        int BATCH = 10;
        if (n > 1000000) BATCH = 5;

        for (size_t it = 0; it < max_iterations; it++) {
            int src = cur, dst = 1 - cur;

            cusparseDnVecSetValues(vx, buf[src]);
            cusparseDnVecSetValues(vy, buf[dst]);

            
            cusparseSpMV(cache.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         &ha, mat, vx, &hb, vy, CUDA_R_32F,
                         CUSPARSE_SPMV_ALG_DEFAULT, cache.d_spmv_buf);

            
            cudaMemsetAsync(cache.d_accum, 0, sizeof(float), stream);
            add_identity_norm_kernel<<<gv, BLOCK_SIZE, 0, stream>>>(buf[dst], buf[src], cache.d_accum, n);

            bool check = ((it + 1) % BATCH == 0) || (it + 1 == max_iterations);
            if (check) {
                cudaMemsetAsync(cache.d_accum + 1, 0, sizeof(float), stream);
                normalize_diff_kernel<<<gv, BLOCK_SIZE, 0, stream>>>(buf[dst], buf[src], cache.d_accum, cache.d_accum + 1, n);
                cudaMemcpyAsync(cache.h_diff, cache.d_accum + 1, sizeof(float), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                iters = it + 1; cur = dst;
                if (*cache.h_diff < thr) { conv = true; break; }
            } else {
                normalize_only_kernel<<<gv, BLOCK_SIZE, 0, stream>>>(buf[dst], cache.d_accum, n);
                iters = it + 1; cur = dst;
            }
        }

        cusparseDestroyDnVec(vx);
        cusparseDestroyDnVec(vy);
        cusparseDestroySpMat(mat);

        if (cur != 0) {
            cudaMemcpyAsync(buf[0], buf[1], (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaStreamSynchronize(stream);
        }

        return eigenvector_centrality_result_t{iters, conv};

    } else {
        
        float* buf[2] = {centralities, cache.d_scratch};

        if (initial_centralities != nullptr) {
            cudaMemcpyAsync(buf[0], initial_centralities,
                           (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        } else {
            fill_kernel<<<gv, BLOCK_SIZE, 0, stream>>>(buf[0], 1.0f / (float)n, n);
        }

        int cur = 0;
        size_t iters = 0;
        bool conv = false;
        float thr = (float)n * epsilon;
        int BATCH = 10;

        for (size_t it = 0; it < max_iterations; it++) {
            int src = cur, dst = 1 - cur;

            
            cudaMemsetAsync(cache.d_accum, 0, sizeof(float), stream);
            spmv_compact_norm_kernel<<<gv, BLOCK_SIZE, 0, stream>>>(cache.d_coff, cache.d_cidx, buf[src], buf[dst], cache.d_accum, n);

            bool check = ((it + 1) % BATCH == 0) || (it + 1 == max_iterations);
            if (check) {
                cudaMemsetAsync(cache.d_accum + 1, 0, sizeof(float), stream);
                normalize_diff_kernel<<<gv, BLOCK_SIZE, 0, stream>>>(buf[dst], buf[src], cache.d_accum, cache.d_accum + 1, n);
                cudaMemcpyAsync(cache.h_diff, cache.d_accum + 1, sizeof(float), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                iters = it + 1; cur = dst;
                if (*cache.h_diff < thr) { conv = true; break; }
            } else {
                normalize_only_kernel<<<gv, BLOCK_SIZE, 0, stream>>>(buf[dst], cache.d_accum, n);
                iters = it + 1; cur = dst;
            }
        }

        if (cur != 0) {
            cudaMemcpyAsync(buf[0], buf[1], (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaStreamSynchronize(stream);
        }

        return eigenvector_centrality_result_t{iters, conv};
    }
}

}  
