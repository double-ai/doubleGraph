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

struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;

    
    double* ow_buf = nullptr;
    int32_t* cnt_buf = nullptr;
    int32_t* noff_buf = nullptr;
    double* pc_buf = nullptr;
    double* sr_buf = nullptr;
    double* pn_buf = nullptr;
    double* sc_buf = nullptr;
    int32_t vertex_cap = 0;

    
    int32_t* ni_buf = nullptr;
    double* nw_buf = nullptr;
    int32_t edge_cap = 0;

    
    void* scan_temp = nullptr;
    size_t scan_temp_cap = 0;

    
    void* spmv_ext_buf = nullptr;
    size_t spmv_ext_cap = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
        cudaMalloc(&sc_buf, 2 * sizeof(double));
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (ow_buf) cudaFree(ow_buf);
        if (cnt_buf) cudaFree(cnt_buf);
        if (noff_buf) cudaFree(noff_buf);
        if (pc_buf) cudaFree(pc_buf);
        if (sr_buf) cudaFree(sr_buf);
        if (pn_buf) cudaFree(pn_buf);
        if (sc_buf) cudaFree(sc_buf);
        if (ni_buf) cudaFree(ni_buf);
        if (nw_buf) cudaFree(nw_buf);
        if (scan_temp) cudaFree(scan_temp);
        if (spmv_ext_buf) cudaFree(spmv_ext_buf);
    }

    void ensure_vertices(int32_t nv) {
        if (vertex_cap < nv) {
            if (ow_buf) cudaFree(ow_buf);
            if (cnt_buf) cudaFree(cnt_buf);
            if (noff_buf) cudaFree(noff_buf);
            if (pc_buf) cudaFree(pc_buf);
            if (sr_buf) cudaFree(sr_buf);
            if (pn_buf) cudaFree(pn_buf);
            cudaMalloc(&ow_buf, nv * sizeof(double));
            cudaMalloc(&cnt_buf, nv * sizeof(int32_t));
            cudaMalloc(&noff_buf, (nv + 1) * sizeof(int32_t));
            cudaMalloc(&pc_buf, nv * sizeof(double));
            cudaMalloc(&sr_buf, nv * sizeof(double));
            cudaMalloc(&pn_buf, nv * sizeof(double));
            vertex_cap = nv;
        }
    }

    void ensure_edges(int32_t ne) {
        int32_t needed = std::max(ne, 1);
        if (edge_cap < needed) {
            if (ni_buf) cudaFree(ni_buf);
            if (nw_buf) cudaFree(nw_buf);
            cudaMalloc(&ni_buf, needed * sizeof(int32_t));
            cudaMalloc(&nw_buf, needed * sizeof(double));
            edge_cap = needed;
        }
    }

    void ensure_scan(size_t bytes) {
        if (scan_temp_cap < bytes) {
            if (scan_temp) cudaFree(scan_temp);
            cudaMalloc(&scan_temp, bytes);
            scan_temp_cap = bytes;
        }
    }

    void ensure_spmv(size_t bytes) {
        if (bytes > 0 && spmv_ext_cap < bytes) {
            if (spmv_ext_buf) cudaFree(spmv_ext_buf);
            cudaMalloc(&spmv_ext_buf, bytes);
            spmv_ext_cap = bytes;
        }
    }
};

__global__ void count_and_outweights_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ mask,
    int32_t* __restrict__ active_counts,
    double* __restrict__ out_w,
    int32_t num_vertices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int v = tid; v < num_vertices; v += stride) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        int32_t count = 0;
        for (int32_t e = start; e < end; e++) {
            if ((mask[e >> 5] >> (e & 31)) & 1) {
                atomicAdd(&out_w[indices[e]], weights[e]);
                count++;
            }
        }
        active_counts[v] = count;
    }
}

__global__ void scatter_edges_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    double* __restrict__ new_weights,
    int32_t num_vertices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int v = tid; v < num_vertices; v += stride) {
        int32_t old_start = old_offsets[v];
        int32_t old_end = old_offsets[v + 1];
        int32_t new_pos = new_offsets[v];
        for (int32_t e = old_start; e < old_end; e++) {
            if ((mask[e >> 5] >> (e & 31)) & 1) {
                new_indices[new_pos] = indices[e];
                new_weights[new_pos] = weights[e];
                new_pos++;
            }
        }
    }
}

__global__ void init_pr_kernel(double* __restrict__ pr, int32_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double val = 1.0 / (double)n;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x)
        pr[i] = val;
}

__global__ void prepare_kernel(
    const double* __restrict__ pr, const double* __restrict__ out_w,
    double* __restrict__ pr_contrib, double* __restrict__ dangling_sum, int32_t n)
{
    double local_sum = 0.0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        double w = out_w[i], p = pr[i];
        if (w > 0.0) pr_contrib[i] = p / w;
        else { pr_contrib[i] = 0.0; local_sum += p; }
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(dangling_sum, local_sum);
}

__global__ void spmv_update_diff_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const double* __restrict__ weights, const double* __restrict__ pr_contrib,
    const double* __restrict__ old_pr, double* __restrict__ new_pr,
    const double* __restrict__ dangling_sum_ptr, double* __restrict__ diff_ptr,
    double alpha, int32_t n)
{
    double dangling_sum = *dangling_sum_ptr;
    double base = (1.0 - alpha) / (double)n + alpha * dangling_sum / (double)n;
    double local_diff = 0.0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int v = tid; v < n; v += stride) {
        int32_t start = offsets[v], end = offsets[v + 1];
        double acc = 0.0;
        for (int32_t e = start; e < end; e++)
            acc += pr_contrib[indices[e]] * weights[e];
        double pv = base + alpha * acc;
        new_pr[v] = pv;
        local_diff += fabs(pv - old_pr[v]);
    }
    for (int off = 16; off > 0; off >>= 1)
        local_diff += __shfl_down_sync(0xffffffff, local_diff, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(diff_ptr, local_diff);
}

__global__ void update_diff_kernel(
    const double* __restrict__ spmv_result, const double* __restrict__ old_pr,
    double* __restrict__ new_pr, const double* __restrict__ dangling_sum_ptr,
    double* __restrict__ diff_ptr, double alpha, int32_t n)
{
    double dangling_sum = *dangling_sum_ptr;
    double base = (1.0 - alpha) / (double)n + alpha * dangling_sum / (double)n;
    double local_diff = 0.0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int v = tid; v < n; v += stride) {
        double pv = base + alpha * spmv_result[v];
        new_pr[v] = pv;
        local_diff += fabs(pv - old_pr[v]);
    }
    for (int off = 16; off > 0; off >>= 1)
        local_diff += __shfl_down_sync(0xffffffff, local_diff, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(diff_ptr, local_diff);
}

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

    int32_t nv = graph.number_of_vertices;
    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    const double* d_wt = edge_weights;
    const uint32_t* d_mask = graph.edge_mask;

    const int BS = 256;
    int nbv = std::min((nv + BS - 1) / BS, 2048);

    cache.ensure_vertices(nv);

    
    double* d_ow = cache.ow_buf;
    cudaMemset(d_ow, 0, nv * sizeof(double));
    count_and_outweights_kernel<<<nbv, BS>>>(d_off, d_idx, d_wt, d_mask,
        cache.cnt_buf, d_ow, nv);

    
    int32_t* d_noff = cache.noff_buf;
    size_t stb = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, stb,
        (int32_t*)nullptr, (int32_t*)nullptr, nv + 1);
    cache.ensure_scan(stb);
    cudaMemcpy(d_noff, cache.cnt_buf, nv * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemset(d_noff + nv, 0, sizeof(int32_t));
    cub::DeviceScan::ExclusiveSum(cache.scan_temp, stb, d_noff, d_noff, nv + 1);

    
    int32_t h_nae;
    cudaMemcpy(&h_nae, d_noff + nv, sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    cache.ensure_edges(h_nae);
    scatter_edges_kernel<<<nbv, BS>>>(d_off, d_idx, d_wt, d_mask,
        d_noff, cache.ni_buf, cache.nw_buf, nv);

    
    double avg_deg = (nv > 0) ? (double)h_nae / nv : 0.0;
    bool use_cs = (h_nae > 0 && avg_deg >= 4.0 && nv >= 5000);

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;

    double* d_pc = cache.pc_buf;
    double* d_sr = cache.sr_buf;

    if (use_cs) {
        cusparseCreateCsr(&matA, nv, nv, h_nae, d_noff, cache.ni_buf, cache.nw_buf,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        cusparseCreateDnVec(&vecX, nv, d_pc, CUDA_R_64F);
        cusparseCreateDnVec(&vecY, nv, d_sr, CUDA_R_64F);
        double one = 1.0, zero = 0.0;
        size_t spmv_buf_sz = 0;
        cusparseSpMV_bufferSize(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, matA, vecX, &zero, vecY, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buf_sz);
        cache.ensure_spmv(spmv_buf_sz);
        cusparseSpMV_preprocess(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, matA, vecX, &zero, vecY, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_ext_buf);
    }

    
    double* d_po = pageranks;
    double* d_pn = cache.pn_buf;
    double* d_ds = cache.sc_buf;
    double* d_df = cache.sc_buf + 1;

    if (initial_pageranks != nullptr)
        cudaMemcpy(d_po, initial_pageranks, nv * sizeof(double), cudaMemcpyDeviceToDevice);
    else
        init_pr_kernel<<<nbv, BS>>>(d_po, nv);

    std::size_t iters = 0;
    bool conv = false;
    double h_df;

    for (std::size_t it = 0; it < max_iterations; it++) {
        cudaMemset(d_ds, 0, 2 * sizeof(double));
        prepare_kernel<<<nbv, BS>>>(d_po, d_ow, d_pc, d_ds, nv);

        if (use_cs) {
            double one = 1.0, zero = 0.0;
            cusparseSpMV(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, matA, vecX, &zero, vecY, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_ext_buf);
            update_diff_kernel<<<nbv, BS>>>(d_sr, d_po, d_pn, d_ds, d_df, alpha, nv);
        } else {
            spmv_update_diff_kernel<<<nbv, BS>>>(d_noff, cache.ni_buf, cache.nw_buf,
                d_pc, d_po, d_pn, d_ds, d_df, alpha, nv);
        }

        cudaMemcpy(&h_df, d_df, sizeof(double), cudaMemcpyDeviceToHost);
        iters++;
        double* tmp = d_po; d_po = d_pn; d_pn = tmp;
        if (h_df < epsilon) { conv = true; break; }
    }

    if (d_po != pageranks)
        cudaMemcpy(pageranks, d_po, nv * sizeof(double), cudaMemcpyDeviceToDevice);

    if (matA) cusparseDestroySpMat(matA);
    if (vecX) cusparseDestroyDnVec(vecX);
    if (vecY) cusparseDestroyDnVec(vecY);

    return PageRankResult{iters, conv};
}

}  
