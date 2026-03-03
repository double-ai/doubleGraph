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
#include <climits>

namespace aai {

namespace {

constexpr int BLOCK_PREPROCESS = 256;
constexpr int BLOCK_ITER = 512;



__global__ void compute_out_weight_sums_and_flags_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ out_weight_sums,
    int32_t* __restrict__ flags,
    int32_t num_edges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    uint32_t word = edge_mask[idx >> 5];
    int active = (word >> (idx & 31)) & 1;
    flags[idx] = active;
    if (active) {
        atomicAdd(&out_weight_sums[indices[idx]], edge_weights[idx]);
    }
}

__global__ void scatter_and_normalize_kernel(
    const int32_t* __restrict__ old_indices,
    const float* __restrict__ old_weights,
    const int32_t* __restrict__ prefix_sum,
    const int32_t* __restrict__ flags,
    const float* __restrict__ out_weight_sums,
    int32_t* __restrict__ new_indices,
    float* __restrict__ new_weights,
    int32_t num_edges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    if (flags[idx]) {
        int new_pos = prefix_sum[idx];
        int32_t src = old_indices[idx];
        new_indices[new_pos] = src;
        float ow = out_weight_sums[src];
        new_weights[new_pos] = (ow > 0.0f) ? __fdividef(old_weights[idx], ow) : 0.0f;
    }
}

__global__ void compute_new_offsets_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ prefix_sum,
    const int32_t* __restrict__ flags,
    int32_t* __restrict__ new_offsets,
    int32_t num_vertices,
    int32_t old_num_edges)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > num_vertices) return;
    if (idx == num_vertices) {
        if (old_num_edges > 0)
            new_offsets[idx] = prefix_sum[old_num_edges - 1] + flags[old_num_edges - 1];
        else
            new_offsets[idx] = 0;
    } else {
        int old_off = old_offsets[idx];
        new_offsets[idx] = (old_off < old_num_edges) ? prefix_sum[old_off] :
            (old_num_edges > 0 ? prefix_sum[old_num_edges - 1] + flags[old_num_edges - 1] : 0);
    }
}

__global__ void build_dangling_list_kernel(
    const float* __restrict__ out_weight_sums,
    int32_t* __restrict__ dangling_indices,
    int32_t* __restrict__ dangling_count,
    int32_t num_vertices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vertices) return;
    if (out_weight_sums[idx] == 0.0f) {
        int pos = atomicAdd(dangling_count, 1);
        dangling_indices[pos] = idx;
    }
}



__global__ void dangling_sum_compact_kernel(
    const float* __restrict__ pr,
    const int32_t* __restrict__ dangling_indices,
    float* __restrict__ sum_out,
    int32_t dangling_count)
{
    typedef cub::BlockReduce<float, BLOCK_ITER> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < dangling_count) ? pr[dangling_indices[idx]] : 0.0f;

    float block_sum = BlockReduce(temp_storage).Sum(val);
    if (threadIdx.x == 0) atomicAdd(sum_out, block_sum);
}

__global__ void update_pr_inplace_and_diff_kernel(
    const float* __restrict__ spmv_result,
    float* __restrict__ pr,
    float* __restrict__ diff_out,
    const float* __restrict__ dangling_sum_ptr,
    float alpha,
    float one_minus_alpha_over_n,
    float alpha_over_n,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<float, BLOCK_ITER> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float dangling_sum = *dangling_sum_ptr;
    float bias = one_minus_alpha_over_n + alpha_over_n * dangling_sum;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0.0f;

    if (idx < num_vertices) {
        float old_val = pr[idx];
        float new_val = bias + alpha * spmv_result[idx];
        pr[idx] = new_val;
        diff = fabsf(new_val - old_val);
    }

    float block_diff = BlockReduce(temp_storage).Sum(diff);
    if (threadIdx.x == 0) atomicAdd(diff_out, block_diff);
}

__global__ void update_pr_inplace_and_diff_nodangling_kernel(
    const float* __restrict__ spmv_result,
    float* __restrict__ pr,
    float* __restrict__ diff_out,
    float alpha,
    float bias,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<float, BLOCK_ITER> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0.0f;

    if (idx < num_vertices) {
        float old_val = pr[idx];
        float new_val = bias + alpha * spmv_result[idx];
        pr[idx] = new_val;
        diff = fabsf(new_val - old_val);
    }

    float block_diff = BlockReduce(temp_storage).Sum(diff);
    if (threadIdx.x == 0) atomicAdd(diff_out, block_diff);
}

__global__ void init_pr_kernel(float* pr, float val, int32_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    pr[idx] = val;
}



struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    float* h_pinned = nullptr;    
    int32_t* h_counts = nullptr;  

    
    float* d_ows = nullptr;
    int32_t* d_flags = nullptr;
    int32_t* d_ps = nullptr;
    void* d_scan_tmp = nullptr;
    int32_t* d_ni = nullptr;
    float* d_nw = nullptr;
    int32_t* d_no = nullptr;
    int32_t* d_di = nullptr;
    int32_t* d_dc = nullptr;

    
    float* d_spmv = nullptr;
    float* d_scalars = nullptr;  
    void* d_spmv_buf = nullptr;

    
    int64_t ows_cap = 0;
    int64_t flags_cap = 0;
    int64_t ps_cap = 0;
    size_t scan_tmp_cap = 0;
    int64_t ni_cap = 0;
    int64_t nw_cap = 0;
    int64_t no_cap = 0;
    int64_t di_cap = 0;
    int64_t spmv_cap = 0;
    size_t spmv_buf_cap = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
        cudaMallocHost(&h_pinned, 2 * sizeof(float));
        cudaMallocHost(&h_counts, 2 * sizeof(int32_t));

        
        cudaMalloc(&d_scalars, 4 * sizeof(float));
        
        cudaMalloc(&d_dc, sizeof(int32_t));
    }

    void ensure(int32_t num_vertices, int32_t num_edges, size_t scan_bytes, size_t spmv_buf_size) {
        int64_t nv = num_vertices;
        int64_t ne = num_edges;

        if (ows_cap < nv) {
            if (d_ows) cudaFree(d_ows);
            cudaMalloc(&d_ows, nv * sizeof(float));
            ows_cap = nv;
        }
        if (flags_cap < ne + 1) {
            if (d_flags) cudaFree(d_flags);
            cudaMalloc(&d_flags, (ne + 1) * sizeof(int32_t));
            flags_cap = ne + 1;
        }
        if (ps_cap < ne + 1) {
            if (d_ps) cudaFree(d_ps);
            cudaMalloc(&d_ps, (ne + 1) * sizeof(int32_t));
            ps_cap = ne + 1;
        }
        if (scan_tmp_cap < scan_bytes) {
            if (d_scan_tmp) cudaFree(d_scan_tmp);
            cudaMalloc(&d_scan_tmp, scan_bytes);
            scan_tmp_cap = scan_bytes;
        }
        if (ni_cap < ne + 1) {
            if (d_ni) cudaFree(d_ni);
            cudaMalloc(&d_ni, (ne + 1) * sizeof(int32_t));
            ni_cap = ne + 1;
        }
        if (nw_cap < ne + 1) {
            if (d_nw) cudaFree(d_nw);
            cudaMalloc(&d_nw, (ne + 1) * sizeof(float));
            nw_cap = ne + 1;
        }
        if (no_cap < nv + 1) {
            if (d_no) cudaFree(d_no);
            cudaMalloc(&d_no, (nv + 1) * sizeof(int32_t));
            no_cap = nv + 1;
        }
        if (di_cap < nv + 1) {
            if (d_di) cudaFree(d_di);
            cudaMalloc(&d_di, (nv + 1) * sizeof(int32_t));
            di_cap = nv + 1;
        }
        if (spmv_cap < nv) {
            if (d_spmv) cudaFree(d_spmv);
            cudaMalloc(&d_spmv, nv * sizeof(float));
            spmv_cap = nv;
        }
        if (spmv_buf_cap < spmv_buf_size) {
            if (d_spmv_buf) cudaFree(d_spmv_buf);
            cudaMalloc(&d_spmv_buf, spmv_buf_size);
            spmv_buf_cap = spmv_buf_size;
        }
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (h_pinned) cudaFreeHost(h_pinned);
        if (h_counts) cudaFreeHost(h_counts);
        if (d_ows) cudaFree(d_ows);
        if (d_flags) cudaFree(d_flags);
        if (d_ps) cudaFree(d_ps);
        if (d_scan_tmp) cudaFree(d_scan_tmp);
        if (d_ni) cudaFree(d_ni);
        if (d_nw) cudaFree(d_nw);
        if (d_no) cudaFree(d_no);
        if (d_di) cudaFree(d_di);
        if (d_dc) cudaFree(d_dc);
        if (d_spmv) cudaFree(d_spmv);
        if (d_scalars) cudaFree(d_scalars);
        if (d_spmv_buf) cudaFree(d_spmv_buf);
    }
};

}  

PageRankResult pagerank_seg_mask(const graph32_t& graph,
                                 const float* edge_weights,
                                 float* pageranks,
                                 const float* precomputed_vertex_out_weight_sums,
                                 float alpha,
                                 float epsilon,
                                 std::size_t max_iterations,
                                 const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;
    const float* d_edge_weights = edge_weights;

    cudaStream_t stream = nullptr;
    cusparseSetStream(cache.cusparse_handle, stream);

    
    
    size_t scan_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, (int32_t*)nullptr, (int32_t*)nullptr, num_edges);

    
    
    cache.ensure(num_vertices, num_edges, scan_bytes, 0);

    cudaMemsetAsync(cache.d_ows, 0, num_vertices * sizeof(float), stream);
    if (num_edges > 0) {
        int g = (num_edges + BLOCK_PREPROCESS - 1) / BLOCK_PREPROCESS;
        compute_out_weight_sums_and_flags_kernel<<<g, BLOCK_PREPROCESS, 0, stream>>>(
            d_indices, d_edge_weights, d_edge_mask, cache.d_ows, cache.d_flags, num_edges);
    }

    
    if (num_edges > 0)
        cub::DeviceScan::ExclusiveSum(cache.d_scan_tmp, scan_bytes, cache.d_flags, cache.d_ps, num_edges, stream);

    
    if (num_edges > 0) {
        int g = (num_edges + BLOCK_PREPROCESS - 1) / BLOCK_PREPROCESS;
        scatter_and_normalize_kernel<<<g, BLOCK_PREPROCESS, 0, stream>>>(
            d_indices, d_edge_weights, cache.d_ps, cache.d_flags,
            cache.d_ows, cache.d_ni, cache.d_nw, num_edges);
    }

    
    {
        int g = (num_vertices + 2 + BLOCK_PREPROCESS - 1) / BLOCK_PREPROCESS;
        compute_new_offsets_kernel<<<g, BLOCK_PREPROCESS, 0, stream>>>(
            d_offsets, cache.d_ps, cache.d_flags, cache.d_no, num_vertices, num_edges);
    }

    
    cudaMemsetAsync(cache.d_dc, 0, sizeof(int32_t), stream);
    if (num_vertices > 0) {
        int g = (num_vertices + BLOCK_PREPROCESS - 1) / BLOCK_PREPROCESS;
        build_dangling_list_kernel<<<g, BLOCK_PREPROCESS, 0, stream>>>(
            cache.d_ows, cache.d_di, cache.d_dc, num_vertices);
    }

    
    cudaMemcpyAsync(&cache.h_counts[0], &cache.d_no[num_vertices], sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&cache.h_counts[1], cache.d_dc, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);

    
    float* d_pr = pageranks;  
    float* d_spmv = cache.d_spmv;

    
    float* d_scalars = cache.d_scalars;
    float* d_spmv_alpha = &d_scalars[0];
    float* d_spmv_beta = &d_scalars[1];
    float* d_dangling_sum = &d_scalars[2];
    float* d_diff_sum = &d_scalars[3];

    float h_scalars[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    cudaMemcpyAsync(d_scalars, h_scalars, 4 * sizeof(float), cudaMemcpyHostToDevice, stream);

    bool has_initial_guess = (initial_pageranks != nullptr);
    if (has_initial_guess) {
        cudaMemcpyAsync(d_pr, initial_pageranks,
            num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        if (num_vertices > 0) {
            int g = (num_vertices + BLOCK_PREPROCESS - 1) / BLOCK_PREPROCESS;
            init_pr_kernel<<<g, BLOCK_PREPROCESS, 0, stream>>>(d_pr, 1.0f / num_vertices, num_vertices);
        }
    }

    
    cudaStreamSynchronize(stream);
    int32_t new_num_edges = cache.h_counts[0];
    int32_t dangling_count = cache.h_counts[1];
    bool has_dangling = (dangling_count > 0);

    
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA,
        (int64_t)num_vertices, (int64_t)num_vertices, (int64_t)new_num_edges,
        (void*)cache.d_no, (void*)cache.d_ni, (void*)cache.d_nw,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, (int64_t)num_vertices, d_pr, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, (int64_t)num_vertices, d_spmv, CUDA_R_32F);

    float h_one = 1.0f, h_zero = 0.0f;
    size_t bufferSize = 0;
    cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_HOST);
    cusparseSpMV_bufferSize(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_one, matA, vecX, &h_zero, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

    
    if (bufferSize > 0) {
        cache.ensure(num_vertices, num_edges, scan_bytes, bufferSize);
    }

    cusparseSpMV_preprocess(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_one, matA, vecX, &h_zero, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.d_spmv_buf);

    cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);

    float oman = (1.0f - alpha) / num_vertices;
    float aon = alpha / num_vertices;
    float bias_no_dangling = oman;

    
    std::size_t num_iterations = 0;
    bool converged = false;

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        
        cudaMemsetAsync(d_dangling_sum, 0, 2 * sizeof(float), stream);

        if (has_dangling) {
            int g = (dangling_count + BLOCK_ITER - 1) / BLOCK_ITER;
            dangling_sum_compact_kernel<<<g, BLOCK_ITER, 0, stream>>>(
                d_pr, cache.d_di, d_dangling_sum, dangling_count);
        }

        
        cusparseSpMV(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_spmv_alpha, matA, vecX, d_spmv_beta, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.d_spmv_buf);

        
        if (has_dangling) {
            int g = (num_vertices + BLOCK_ITER - 1) / BLOCK_ITER;
            update_pr_inplace_and_diff_kernel<<<g, BLOCK_ITER, 0, stream>>>(
                d_spmv, d_pr, d_diff_sum, d_dangling_sum,
                alpha, oman, aon, num_vertices);
        } else {
            int g = (num_vertices + BLOCK_ITER - 1) / BLOCK_ITER;
            update_pr_inplace_and_diff_nodangling_kernel<<<g, BLOCK_ITER, 0, stream>>>(
                d_spmv, d_pr, d_diff_sum,
                alpha, bias_no_dangling, num_vertices);
        }

        
        cudaMemcpyAsync(cache.h_pinned, d_diff_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        num_iterations = iter + 1;

        if (*cache.h_pinned < epsilon) {
            converged = true;
            break;
        }
    }

    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(matA);

    return PageRankResult{num_iterations, converged};
}

}  
