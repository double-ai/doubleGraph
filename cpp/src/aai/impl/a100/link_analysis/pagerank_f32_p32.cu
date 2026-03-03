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
#include <vector>

namespace aai {

namespace {





struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;
    cudaStream_t dangling_stream = nullptr;
    cudaEvent_t dangling_done = nullptr;

    float* out_weight = nullptr;
    int64_t out_weight_capacity = 0;

    float* pers_norm = nullptr;
    int64_t pers_norm_capacity = 0;

    float* spmv_result = nullptr;
    int64_t spmv_result_capacity = 0;

    float* scalars = nullptr;  

    float* w_mod = nullptr;
    int64_t w_mod_capacity = 0;

    void* spmv_buffer = nullptr;
    size_t spmv_buffer_capacity = 0;

    int32_t* dangling_indices = nullptr;
    int64_t dangling_capacity = 0;

    int* d_count = nullptr;

    Cache() {
        cusparseCreate(&handle);
        cudaStreamCreate(&dangling_stream);
        cudaEventCreateWithFlags(&dangling_done, cudaEventDisableTiming);
        cudaMalloc(&scalars, 2 * sizeof(float));
        cudaMalloc(&d_count, sizeof(int));
    }

    ~Cache() override {
        if (handle) cusparseDestroy(handle);
        if (dangling_stream) cudaStreamDestroy(dangling_stream);
        if (dangling_done) cudaEventDestroy(dangling_done);
        if (out_weight) cudaFree(out_weight);
        if (pers_norm) cudaFree(pers_norm);
        if (spmv_result) cudaFree(spmv_result);
        if (scalars) cudaFree(scalars);
        if (w_mod) cudaFree(w_mod);
        if (spmv_buffer) cudaFree(spmv_buffer);
        if (dangling_indices) cudaFree(dangling_indices);
        if (d_count) cudaFree(d_count);
    }

    void ensure_out_weight(int64_t n) {
        if (out_weight_capacity < n) {
            if (out_weight) cudaFree(out_weight);
            cudaMalloc(&out_weight, n * sizeof(float));
            out_weight_capacity = n;
        }
    }

    void ensure_pers_norm(int64_t n) {
        if (pers_norm_capacity < n) {
            if (pers_norm) cudaFree(pers_norm);
            cudaMalloc(&pers_norm, n * sizeof(float));
            pers_norm_capacity = n;
        }
    }

    void ensure_spmv_result(int64_t n) {
        if (spmv_result_capacity < n) {
            if (spmv_result) cudaFree(spmv_result);
            cudaMalloc(&spmv_result, n * sizeof(float));
            spmv_result_capacity = n;
        }
    }

    void ensure_w_mod(int64_t e) {
        int64_t sz = e > 0 ? e : 1;
        if (w_mod_capacity < sz) {
            if (w_mod) cudaFree(w_mod);
            cudaMalloc(&w_mod, sz * sizeof(float));
            w_mod_capacity = sz;
        }
    }

    void ensure_spmv_buf(size_t sz) {
        if (spmv_buffer_capacity < sz) {
            if (spmv_buffer) cudaFree(spmv_buffer);
            cudaMalloc(&spmv_buffer, sz);
            spmv_buffer_capacity = sz;
        }
    }

    void ensure_dangling(int64_t n) {
        if (dangling_capacity < n) {
            if (dangling_indices) cudaFree(dangling_indices);
            cudaMalloc(&dangling_indices, n * sizeof(int32_t));
            dangling_capacity = n;
        }
    }
};





__global__ void compute_out_weights_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ out_weight,
    int num_edges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < num_edges) {
        atomicAdd(&out_weight[indices[e]], weights[e]);
    }
}

__global__ void compute_wmod_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ out_weight,
    float* __restrict__ w_mod,
    int num_edges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < num_edges) {
        w_mod[e] = weights[e] / out_weight[indices[e]];
    }
}

__global__ void build_pers_norm_kernel(
    const int32_t* __restrict__ pers_vertices,
    const float* __restrict__ pers_values,
    float* __restrict__ pers_norm,
    float inv_pers_sum,
    int pers_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pers_size) {
        pers_norm[pers_vertices[i]] = pers_values[i] * inv_pers_sum;
    }
}

__global__ void init_pr_kernel(float* __restrict__ pr, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pr[i] = val;
    }
}

__global__ void count_dangling_kernel(
    const float* __restrict__ out_weight,
    int* __restrict__ count,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && out_weight[i] == 0.0f) {
        atomicAdd(count, 1);
    }
}

__global__ void collect_dangling_kernel(
    const float* __restrict__ out_weight,
    int32_t* __restrict__ dangling_indices,
    int* __restrict__ offset,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && out_weight[i] == 0.0f) {
        int pos = atomicAdd(offset, 1);
        dangling_indices[pos] = i;
    }
}





template <int BLOCK_SIZE>
__global__ void dangling_sum_sparse_kernel(
    const float* __restrict__ pr,
    const int32_t* __restrict__ dangling_indices,
    float* __restrict__ dangling_sum,
    int num_dangling)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float thread_sum = 0.0f;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < num_dangling; i += gridDim.x * BLOCK_SIZE) {
        thread_sum += pr[dangling_indices[i]];
    }

    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(dangling_sum, block_sum);
    }
}

template <int BLOCK_SIZE>
__global__ void dangling_sum_dense_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ out_weight,
    float* __restrict__ dangling_sum,
    int n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float thread_sum = 0.0f;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        if (out_weight[i] == 0.0f) {
            thread_sum += pr[i];
        }
    }

    float block_sum = BlockReduce(temp_storage).Sum(thread_sum);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(dangling_sum, block_sum);
    }
}

template <int BLOCK_SIZE>
__global__ void post_process_kernel(
    float* __restrict__ pr,
    const float* __restrict__ spmv_result,
    const float* __restrict__ pers_norm,
    float alpha,
    const float* __restrict__ dangling_sum,
    float one_minus_alpha,
    float* __restrict__ diff,
    int n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float base_factor = alpha * (*dangling_sum) + one_minus_alpha;
    float thread_diff = 0.0f;

    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        float new_val = alpha * spmv_result[i] + base_factor * pers_norm[i];
        thread_diff += fabsf(new_val - pr[i]);
        pr[i] = new_val;
    }

    float block_diff = BlockReduce(temp_storage).Sum(thread_diff);
    if (threadIdx.x == 0 && block_diff != 0.0f) {
        atomicAdd(diff, block_diff);
    }
}





void launch_compute_out_weights(const int32_t* indices, const float* weights,
                                 float* out_weight, int num_edges) {
    if (num_edges == 0) return;
    int block = 256, grid = (num_edges + block - 1) / block;
    compute_out_weights_kernel<<<grid, block>>>(indices, weights, out_weight, num_edges);
}

void launch_compute_wmod(const int32_t* indices, const float* weights,
                          const float* out_weight, float* w_mod, int num_edges) {
    if (num_edges == 0) return;
    int block = 256, grid = (num_edges + block - 1) / block;
    compute_wmod_kernel<<<grid, block>>>(indices, weights, out_weight, w_mod, num_edges);
}

void launch_build_pers_norm(const int32_t* pers_vertices, const float* pers_values,
                             float* pers_norm, float inv_pers_sum, int pers_size) {
    if (pers_size == 0) return;
    int block = 256, grid = (pers_size + block - 1) / block;
    build_pers_norm_kernel<<<grid, block>>>(pers_vertices, pers_values, pers_norm, inv_pers_sum, pers_size);
}

void launch_init_pr(float* pr, float val, int n) {
    if (n == 0) return;
    int block = 256, grid = (n + block - 1) / block;
    init_pr_kernel<<<grid, block>>>(pr, val, n);
}

void launch_count_dangling(const float* out_weight, int* count, int n) {
    if (n == 0) return;
    int block = 256, grid = (n + block - 1) / block;
    count_dangling_kernel<<<grid, block>>>(out_weight, count, n);
}

void launch_collect_dangling(const float* out_weight, int32_t* indices, int* offset, int n) {
    if (n == 0) return;
    int block = 256, grid = (n + block - 1) / block;
    collect_dangling_kernel<<<grid, block>>>(out_weight, indices, offset, n);
}

void launch_dangling_sum_sparse(const float* pr, const int32_t* dangling_indices,
                                 float* dangling_sum, int num_dangling, cudaStream_t stream) {
    if (num_dangling == 0) return;
    const int BLOCK = 256;
    int grid = (num_dangling + BLOCK - 1) / BLOCK;
    if (grid > 512) grid = 512;
    dangling_sum_sparse_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(pr, dangling_indices, dangling_sum, num_dangling);
}

void launch_dangling_sum_dense(const float* pr, const float* out_weight,
                                float* dangling_sum, int n, cudaStream_t stream) {
    if (n == 0) return;
    const int BLOCK = 256;
    int grid = (n + BLOCK - 1) / BLOCK;
    if (grid > 1024) grid = 1024;
    dangling_sum_dense_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(pr, out_weight, dangling_sum, n);
}

void launch_post_process(float* pr, const float* spmv_result, const float* pers_norm,
                          float alpha, const float* dangling_sum, float one_minus_alpha,
                          float* diff, int n, cudaStream_t stream) {
    if (n == 0) return;
    const int BLOCK = 256;
    int grid = (n + BLOCK - 1) / BLOCK;
    if (grid > 1024) grid = 1024;
    post_process_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
        pr, spmv_result, pers_norm, alpha, dangling_sum, one_minus_alpha, diff, n);
}

}  





PageRankResult personalized_pagerank(const graph32_t& graph,
                                     const float* edge_weights,
                                     const int32_t* personalization_vertices,
                                     const float* personalization_values,
                                     std::size_t personalization_size,
                                     float* pageranks,
                                     const float* precomputed_vertex_out_weight_sums,
                                     float alpha,
                                     float epsilon,
                                     std::size_t max_iterations,
                                     const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t N = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    float oma = 1.0f - alpha;

    
    cache.ensure_out_weight(N);
    cache.ensure_pers_norm(N);
    cache.ensure_spmv_result(N);
    cache.ensure_w_mod(E);

    float* d_ow = cache.out_weight;
    float* d_pn = cache.pers_norm;
    float* d_sr = cache.spmv_result;
    float* d_ds = cache.scalars;       
    float* d_df = cache.scalars + 1;   
    float* d_wm = cache.w_mod;

    cudaMemset(d_ow, 0, N * sizeof(float));
    cudaMemset(d_pn, 0, N * sizeof(float));

    
    launch_compute_out_weights(indices, edge_weights, d_ow, E);

    
    launch_compute_wmod(indices, edge_weights, d_ow, d_wm, E);

    
    int h_num_dangling = 0;
    cudaMemset(cache.d_count, 0, sizeof(int));
    launch_count_dangling(d_ow, cache.d_count, N);
    cudaMemcpy(&h_num_dangling, cache.d_count, sizeof(int), cudaMemcpyDeviceToHost);

    bool use_sparse_dangling = (h_num_dangling > 0 && h_num_dangling < N / 2);
    if (use_sparse_dangling) {
        cache.ensure_dangling(h_num_dangling);
        cudaMemset(cache.d_count, 0, sizeof(int));
        launch_collect_dangling(d_ow, cache.dangling_indices, cache.d_count, N);
    }

    
    int64_t ps = static_cast<int64_t>(personalization_size);
    if (ps > 0) {
        std::vector<float> hpv(ps);
        cudaMemcpy(hpv.data(), personalization_values, ps * sizeof(float), cudaMemcpyDeviceToHost);
        float psum = 0.0f;
        for (int64_t i = 0; i < ps; i++) psum += hpv[i];
        float inv = (psum > 0.0f) ? (1.0f / psum) : 0.0f;
        launch_build_pers_norm(personalization_vertices, personalization_values, d_pn, inv, (int)ps);
    }

    
    if (initial_pageranks != nullptr) {
        cudaMemcpy(pageranks, initial_pageranks, N * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        launch_init_pr(pageranks, 1.0f / (float)N, N);
    }

    
    cusparseSpMatDescr_t mat = nullptr;
    cusparseDnVecDescr_t vpr = nullptr, vsr = nullptr;

    cusparseCreateCsr(&mat, N, N, E,
        (void*)offsets, (void*)indices, (void*)d_wm,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseCreateDnVec(&vpr, N, pageranks, CUDA_R_32F);
    cusparseCreateDnVec(&vsr, N, d_sr, CUDA_R_32F);

    float h1 = 1.0f, h0 = 0.0f;
    size_t bufsz = 0;
    cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h1, mat, vpr, &h0, vsr, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufsz);

    if (bufsz == 0) bufsz = 4;
    cache.ensure_spmv_buf(bufsz);

    cusparseSpMV_preprocess(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h1, mat, vpr, &h0, vsr, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buffer);

    
    cudaStream_t stream0 = 0;
    cusparseSetStream(cache.handle, stream0);

    size_t iters = 0;
    bool conv = false;
    float h_diff;

    for (size_t it = 0; it < max_iterations; it++) {
        cudaMemsetAsync(d_ds, 0, sizeof(float), cache.dangling_stream);
        cudaMemsetAsync(d_df, 0, sizeof(float), stream0);

        if (h_num_dangling == 0) {
            
        } else if (use_sparse_dangling) {
            launch_dangling_sum_sparse(pageranks, cache.dangling_indices, d_ds, h_num_dangling, cache.dangling_stream);
        } else {
            launch_dangling_sum_dense(pageranks, d_ow, d_ds, N, cache.dangling_stream);
        }
        cudaEventRecord(cache.dangling_done, cache.dangling_stream);

        cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h1, mat, vpr, &h0, vsr, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buffer);

        cudaStreamWaitEvent(stream0, cache.dangling_done);

        launch_post_process(pageranks, d_sr, d_pn, alpha, d_ds, oma, d_df, N, stream0);

        iters = it + 1;

        cudaMemcpy(&h_diff, d_df, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_diff < epsilon) { conv = true; break; }
    }

    
    cusparseDestroySpMat(mat);
    cusparseDestroyDnVec(vpr);
    cusparseDestroyDnVec(vsr);

    return {iters, conv};
}

}  
