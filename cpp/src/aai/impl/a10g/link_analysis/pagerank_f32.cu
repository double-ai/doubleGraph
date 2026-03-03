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
#include <cub/cub.cuh>
#include <cusparse.h>
#include <cstdint>
#include <climits>

namespace aai {

namespace {

constexpr int BLOCK = 256;

struct Cache : Cacheable {
    
    float* out_w = nullptr;
    float* adj_weights = nullptr;
    float* pr_tmp = nullptr;
    float* spmv_buf = nullptr;
    float* d_scalars = nullptr;

    
    int64_t out_w_capacity = 0;
    int64_t adj_weights_capacity = 0;
    int64_t pr_tmp_capacity = 0;
    int64_t spmv_buf_capacity = 0;
    bool d_scalars_allocated = false;

    
    cusparseHandle_t cs_handle = nullptr;
    float* d_alpha_beta = nullptr;
    float* h_diff_pinned = nullptr;

    
    void* cs_buffer = nullptr;
    size_t cs_buffer_size = 0;

    Cache() {
        cusparseCreate(&cs_handle);
        cudaMalloc(&d_alpha_beta, 2 * sizeof(float));
        float h_vals[2] = {1.0f, 0.0f};
        cudaMemcpy(d_alpha_beta, h_vals, 2 * sizeof(float), cudaMemcpyHostToDevice);
        cudaHostAlloc(&h_diff_pinned, sizeof(float), cudaHostAllocDefault);
    }

    void ensure(int32_t nv, int32_t ne) {
        int64_t ne64 = (int64_t)ne > 0 ? (int64_t)ne : 1;

        if (out_w_capacity < nv) {
            if (out_w) cudaFree(out_w);
            cudaMalloc(&out_w, (int64_t)nv * sizeof(float));
            out_w_capacity = nv;
        }
        if (adj_weights_capacity < ne64) {
            if (adj_weights) cudaFree(adj_weights);
            cudaMalloc(&adj_weights, ne64 * sizeof(float));
            adj_weights_capacity = ne64;
        }
        if (pr_tmp_capacity < nv) {
            if (pr_tmp) cudaFree(pr_tmp);
            cudaMalloc(&pr_tmp, (int64_t)nv * sizeof(float));
            pr_tmp_capacity = nv;
        }
        if (spmv_buf_capacity < nv) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, (int64_t)nv * sizeof(float));
            spmv_buf_capacity = nv;
        }
        if (!d_scalars_allocated) {
            cudaMalloc(&d_scalars, 4 * sizeof(float));
            d_scalars_allocated = true;
        }
    }

    void ensure_cs_buffer(size_t needed) {
        if (needed > cs_buffer_size) {
            if (cs_buffer) cudaFree(cs_buffer);
            size_t alloc_size = (needed > 0) ? (needed * 2) : 4096;
            cudaMalloc(&cs_buffer, alloc_size);
            cs_buffer_size = alloc_size;
        }
    }

    ~Cache() override {
        if (out_w) cudaFree(out_w);
        if (adj_weights) cudaFree(adj_weights);
        if (pr_tmp) cudaFree(pr_tmp);
        if (spmv_buf) cudaFree(spmv_buf);
        if (d_scalars) cudaFree(d_scalars);
        if (cs_buffer) cudaFree(cs_buffer);
        if (d_alpha_beta) cudaFree(d_alpha_beta);
        if (h_diff_pinned) cudaFreeHost(h_diff_pinned);
        if (cs_handle) cusparseDestroy(cs_handle);
    }
};



__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ csc_indices,
    const float* __restrict__ edge_weights,
    float* __restrict__ out_weight_sums,
    int32_t num_edges)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x) {
        atomicAdd(&out_weight_sums[csc_indices[idx]], edge_weights[idx]);
    }
}

__global__ void compute_adjusted_weights_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ out_w,
    float* __restrict__ adj_weights,
    int32_t num_edges)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges; idx += blockDim.x * gridDim.x) {
        int src = indices[idx];
        float ow = out_w[src];
        adj_weights[idx] = (ow == 0.f) ? 0.f : __fdividef(edge_weights[idx], ow);
    }
}

__global__ void count_dangling_kernel(
    const float* __restrict__ out_w,
    int32_t* __restrict__ count,
    int32_t n)
{
    typedef cub::BlockReduce<int, BLOCK> BR;
    __shared__ typename BR::TempStorage ts;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = (i < n && out_w[i] == 0.f) ? 1 : 0;
    int bs = BR(ts).Sum(c);
    if (threadIdx.x == 0 && bs > 0) atomicAdd(count, bs);
}



__global__ void init_pr_kernel(float* __restrict__ pr, float val, int32_t n)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
        pr[idx] = val;
}

__global__ void dangling_sum_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ out_w,
    float* __restrict__ d_dangling,
    int32_t n)
{
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage ts;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float dang = (i < n && out_w[i] == 0.f) ? pr[i] : 0.f;
    float bs = BR(ts).Sum(dang);
    if (threadIdx.x == 0 && bs != 0.f)
        atomicAdd(d_dangling, bs);
}


__global__ void update_diff_kernel(
    const float* __restrict__ spmv_result,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    float* __restrict__ d_diff,
    const float* __restrict__ d_dangling,
    float alpha,
    float inv_n,
    int32_t n)
{
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage ts;

    __shared__ float s_base;
    if (threadIdx.x == 0) {
        s_base = (1.f - alpha) * inv_n + alpha * (*d_dangling) * inv_n;
    }
    __syncthreads();

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0.f;
    if (v < n) {
        float np = s_base + alpha * spmv_result[v];
        pr_new[v] = np;
        diff = fabsf(np - pr_old[v]);
    }
    float bs = BR(ts).Sum(diff);
    if (threadIdx.x == 0) atomicAdd(d_diff, bs);
}


__global__ void spmv_update_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ adj_weights,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    float* __restrict__ d_diff,
    const float* __restrict__ d_dangling,
    float alpha,
    float inv_n,
    int32_t n)
{
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage ts;

    __shared__ float s_base;
    if (threadIdx.x == 0) {
        s_base = (1.f - alpha) * inv_n + alpha * (*d_dangling) * inv_n;
    }
    __syncthreads();

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0.f;
    if (v < n) {
        int s = offsets[v];
        int e = offsets[v + 1];
        float sum = 0.f;
        for (int j = s; j < e; j++) {
            sum += __ldg(&pr_old[indices[j]]) * adj_weights[j];
        }
        float np = s_base + alpha * sum;
        pr_new[v] = np;
        diff = fabsf(np - pr_old[v]);
    }
    float bs = BR(ts).Sum(diff);
    if (threadIdx.x == 0) atomicAdd(d_diff, bs);
}

}  

PageRankResult pagerank(const graph32_t& graph,
                        const float* edge_weights,
                        float* pageranks,
                        const float* precomputed_vertex_out_weight_sums,
                        float alpha,
                        float epsilon,
                        std::size_t max_iterations,
                        const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    if (nv == 0) {
        return PageRankResult{0, true};
    }

    cache.ensure(nv, ne);

    float* out_w = cache.out_w;
    float* adj_weights = cache.adj_weights;
    float* pr_tmp = cache.pr_tmp;
    float* spmv_buf = cache.spmv_buf;
    float* d_scalars = cache.d_scalars;

    float* d_dangling = d_scalars;
    float* d_diff = d_scalars + 1;
    int32_t* d_count = (int32_t*)(d_scalars + 2);

    int gv = (nv + BLOCK - 1) / BLOCK;
    int ge = ne > 0 ? (ne + BLOCK - 1) / BLOCK : 1;
    float inv_n = 1.f / (float)nv;

    
    cudaMemsetAsync(out_w, 0, nv * sizeof(float));
    if (ne > 0) {
        compute_out_weight_sums_kernel<<<ge, BLOCK>>>(indices, edge_weights, out_w, ne);
    }
    if (ne > 0) {
        compute_adjusted_weights_kernel<<<ge, BLOCK>>>(indices, edge_weights, out_w, adj_weights, ne);
    }

    
    cudaMemsetAsync(d_count, 0, sizeof(int32_t));
    count_dangling_kernel<<<gv, BLOCK>>>(out_w, d_count, nv);
    int32_t h_count;
    cudaMemcpy(&h_count, d_count, sizeof(int32_t), cudaMemcpyDeviceToHost);
    bool has_dangling = (h_count > 0);

    
    bool has_init = (initial_pageranks != nullptr);
    if (has_init)
        cudaMemcpyAsync(pageranks, initial_pageranks, nv * sizeof(float), cudaMemcpyDeviceToDevice);
    else
        init_pr_kernel<<<gv, BLOCK>>>(pageranks, inv_n, nv);

    
    cudaMemsetAsync(d_dangling, 0, sizeof(float));

    size_t eff_max = max_iterations;
    float* cur = pageranks;
    float* nxt = pr_tmp;
    bool conv = false;
    size_t it;

    float eps_eff = epsilon;

    
    int64_t avg_deg = (nv > 0 && ne > 0) ? (int64_t)ne / (int64_t)nv : 0;
    bool use_cusparse = (ne > 100000) && (avg_deg >= 4) && (cache.cs_handle != nullptr);

    float* d_alpha_beta = cache.d_alpha_beta;
    float* h_diff_pinned = cache.h_diff_pinned;

    if (use_cusparse) {
        cusparseSpMatDescr_t matA;
        cusparseCreateCsr(&matA, nv, nv, ne,
                          (void*)offsets, (void*)indices, (void*)adj_weights,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        cusparseDnVecDescr_t vecX, vecY;
        cusparseCreateDnVec(&vecX, nv, cur, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, nv, spmv_buf, CUDA_R_32F);

        float h_one = 1.0f, h_zero = 0.0f;
        cusparseSetPointerMode(cache.cs_handle, CUSPARSE_POINTER_MODE_HOST);

        size_t bufferSize;
        cusparseSpMV_bufferSize(cache.cs_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &h_one, matA, vecX, &h_zero, vecY,
                                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
        cache.ensure_cs_buffer(bufferSize);

        cusparseSpMV_preprocess(cache.cs_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &h_one, matA, vecX, &h_zero, vecY,
                                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.cs_buffer);

        
        cusparseSetPointerMode(cache.cs_handle, CUSPARSE_POINTER_MODE_DEVICE);
        float* d_one = d_alpha_beta;
        float* d_zero = d_alpha_beta + 1;

        for (it = 0; it < eff_max; it++) {
            cudaMemsetAsync(d_diff, 0, sizeof(float));
            if (has_dangling) {
                cudaMemsetAsync(d_dangling, 0, sizeof(float));
                dangling_sum_kernel<<<gv, BLOCK>>>(cur, out_w, d_dangling, nv);
            }

            cusparseDnVecSetValues(vecX, cur);
            cusparseSpMV(cache.cs_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                         d_one, matA, vecX, d_zero, vecY,
                         CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.cs_buffer);

            update_diff_kernel<<<gv, BLOCK>>>(spmv_buf, cur, nxt, d_diff, d_dangling, alpha, inv_n, nv);

            cudaMemcpyAsync(h_diff_pinned, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            float h_diff = *h_diff_pinned;
            float* t = cur; cur = nxt; nxt = t;
            if (h_diff < eps_eff) { conv = true; it++; break; }
        }

        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cusparseDestroySpMat(matA);
    } else {
        
        for (it = 0; it < eff_max; it++) {
            cudaMemsetAsync(d_diff, 0, sizeof(float));
            if (has_dangling) {
                cudaMemsetAsync(d_dangling, 0, sizeof(float));
                dangling_sum_kernel<<<gv, BLOCK>>>(cur, out_w, d_dangling, nv);
            }

            spmv_update_kernel<<<gv, BLOCK>>>(offsets, indices, adj_weights, cur, nxt, d_diff, d_dangling, alpha, inv_n, nv);

            cudaMemcpyAsync(h_diff_pinned, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            float h_diff = *h_diff_pinned;
            float* t = cur; cur = nxt; nxt = t;
            if (h_diff < eps_eff) { conv = true; it++; break; }
        }
    }

    if (cur != pageranks)
        cudaMemcpy(pageranks, cur, nv * sizeof(float), cudaMemcpyDeviceToDevice);

    return PageRankResult{it, conv};
}

}  
