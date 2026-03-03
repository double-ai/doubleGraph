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
#include <vector>

namespace aai {

namespace {





struct Cache : Cacheable {
    
    cusparseHandle_t handle = nullptr;
    float* h_diff = nullptr;
    cudaStream_t main_stream = nullptr;
    cudaStream_t aux_stream = nullptr;
    cudaEvent_t ev_main = nullptr;
    cudaEvent_t ev_aux = nullptr;

    
    float* d_ds = nullptr;
    float* d_df = nullptr;
    float* d_one = nullptr;
    float* d_z = nullptr;

    
    float* d_ow = nullptr;
    float* d_pn = nullptr;
    float* d_p0 = nullptr;
    float* d_p1 = nullptr;
    int64_t ow_capacity = 0;
    int64_t pn_capacity = 0;
    int64_t p0_capacity = 0;
    int64_t p1_capacity = 0;

    
    uint32_t* d_dg = nullptr;
    int64_t dg_capacity = 0;

    
    float* d_sw = nullptr;
    int64_t sw_capacity = 0;

    
    void* spmv_buf = nullptr;
    size_t spmv_buf_capacity = 0;

    Cache() {
        cusparseCreate(&handle);
        cudaMallocHost(&h_diff, sizeof(float));
        cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&aux_stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&ev_main, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&ev_aux, cudaEventDisableTiming);

        cudaMalloc(&d_ds, sizeof(float));
        cudaMalloc(&d_df, sizeof(float));
        cudaMalloc(&d_one, sizeof(float));
        cudaMalloc(&d_z, sizeof(float));
    }

    ~Cache() override {
        if (handle) cusparseDestroy(handle);
        if (h_diff) cudaFreeHost(h_diff);
        if (main_stream) cudaStreamDestroy(main_stream);
        if (aux_stream) cudaStreamDestroy(aux_stream);
        if (ev_main) cudaEventDestroy(ev_main);
        if (ev_aux) cudaEventDestroy(ev_aux);
        if (d_ds) cudaFree(d_ds);
        if (d_df) cudaFree(d_df);
        if (d_one) cudaFree(d_one);
        if (d_z) cudaFree(d_z);
        if (d_ow) cudaFree(d_ow);
        if (d_pn) cudaFree(d_pn);
        if (d_p0) cudaFree(d_p0);
        if (d_p1) cudaFree(d_p1);
        if (d_dg) cudaFree(d_dg);
        if (d_sw) cudaFree(d_sw);
        if (spmv_buf) cudaFree(spmv_buf);
    }

    void ensure(int32_t N, int32_t E) {
        if (ow_capacity < N) {
            if (d_ow) cudaFree(d_ow);
            cudaMalloc(&d_ow, N * sizeof(float));
            ow_capacity = N;
        }
        if (pn_capacity < N) {
            if (d_pn) cudaFree(d_pn);
            cudaMalloc(&d_pn, N * sizeof(float));
            pn_capacity = N;
        }
        if (p0_capacity < N) {
            if (d_p0) cudaFree(d_p0);
            cudaMalloc(&d_p0, N * sizeof(float));
            p0_capacity = N;
        }
        if (p1_capacity < N) {
            if (d_p1) cudaFree(d_p1);
            cudaMalloc(&d_p1, N * sizeof(float));
            p1_capacity = N;
        }
        int64_t dg_words = ((int64_t)N + 31) / 32;
        if (dg_capacity < dg_words) {
            if (d_dg) cudaFree(d_dg);
            cudaMalloc(&d_dg, dg_words * sizeof(uint32_t));
            dg_capacity = dg_words;
        }
        if (sw_capacity < E) {
            if (d_sw) cudaFree(d_sw);
            cudaMalloc(&d_sw, E * sizeof(float));
            sw_capacity = E;
        }
    }

    void ensure_spmv_buf(size_t bsz) {
        if (spmv_buf_capacity < bsz) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, bsz);
            spmv_buf_capacity = bsz;
        }
    }
};





#define BLOCK_SIZE 512

__global__ void compute_out_weights_kernel(
    const float* __restrict__ ew, const uint32_t* __restrict__ mask,
    const int32_t* __restrict__ idx, float* __restrict__ ow, int E) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E; i += blockDim.x * gridDim.x) {
        if ((mask[i >> 5] >> (i & 31)) & 1)
            atomicAdd(&ow[idx[i]], ew[i]);
    }
}

__global__ void compute_scaled_weights_kernel(
    const float* __restrict__ ew, const uint32_t* __restrict__ mask,
    const int32_t* __restrict__ idx, const float* __restrict__ ow,
    float* __restrict__ sw, float alpha, int E) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < E; i += blockDim.x * gridDim.x) {
        float w = ((mask[i >> 5] >> (i & 31)) & 1) ? ew[i] : 0.0f;
        float o = ow[idx[i]];
        sw[i] = (o > 0.0f) ? (alpha * w / o) : 0.0f;
    }
}

__global__ void compute_dangling_bitmask_kernel(
    const float* __restrict__ ow, uint32_t* __restrict__ dg_bits, int N) {
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_words = (N + 31) / 32;
    for (int w = word_idx; w < num_words; w += blockDim.x * gridDim.x) {
        uint32_t bits = 0;
        int base = w * 32;
        for (int b = 0; b < 32 && base + b < N; b++) {
            if (ow[base + b] == 0.0f) bits |= (1u << b);
        }
        dg_bits[w] = bits;
    }
}

__global__ void init_pers_norm_kernel(
    const int32_t* __restrict__ pv, const float* __restrict__ pval,
    float* __restrict__ pn, float inv, int ps) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ps; i += blockDim.x * gridDim.x)
        pn[pv[i]] = pval[i] * inv;
}

__global__ void init_uniform_pr_kernel(float* __restrict__ pr, float val, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        pr[i] = val;
}

__global__ void dangling_sum_bitmask_kernel(
    const float* __restrict__ pr, const uint32_t* __restrict__ dg_bits,
    float* __restrict__ res, int N) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage tmp;
    float s = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        if ((dg_bits[i >> 5] >> (i & 31)) & 1)
            s += pr[i];
    }
    float bs = BR(tmp).Sum(s);
    if (threadIdx.x == 0 && bs != 0.0f) atomicAdd(res, bs);
}

__global__ void fused_base_diff_kernel(
    float* __restrict__ np, const float* __restrict__ op,
    const float* __restrict__ pn, const float* __restrict__ ds,
    float alpha, float oma, float* __restrict__ diff, int N) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage tmp;
    __shared__ float s_bf;
    if (threadIdx.x == 0) s_bf = alpha * (*ds) + oma;
    __syncthreads();
    float bf = s_bf;
    float ld = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float v = np[i] + bf * pn[i];
        np[i] = v;
        ld += fabsf(v - op[i]);
    }
    float bd = BR(tmp).Sum(ld);
    if (threadIdx.x == 0 && bd != 0.0f) atomicAdd(diff, bd);
}

__global__ void base_factor_only_kernel(
    float* __restrict__ np, const float* __restrict__ pn,
    const float* __restrict__ ds, float alpha, float oma, int N) {
    __shared__ float s_bf;
    if (threadIdx.x == 0) s_bf = alpha * (*ds) + oma;
    __syncthreads();
    float bf = s_bf;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        np[i] += bf * pn[i];
}





void launch_compute_out_weights(const float* ew, const uint32_t* m, const int32_t* idx, float* ow, int E, cudaStream_t s) {
    if (E > 0) compute_out_weights_kernel<<<min((E+BLOCK_SIZE-1)/BLOCK_SIZE, 65535), BLOCK_SIZE, 0, s>>>(ew, m, idx, ow, E);
}

void launch_compute_scaled_weights(const float* ew, const uint32_t* m, const int32_t* idx, const float* ow, float* sw, float alpha, int E, cudaStream_t s) {
    if (E > 0) compute_scaled_weights_kernel<<<min((E+BLOCK_SIZE-1)/BLOCK_SIZE, 65535), BLOCK_SIZE, 0, s>>>(ew, m, idx, ow, sw, alpha, E);
}

void launch_compute_dangling_bitmask(const float* ow, uint32_t* dg, int N, cudaStream_t s) {
    if (N > 0) {
        int nw = (N + 31) / 32;
        compute_dangling_bitmask_kernel<<<min((nw+BLOCK_SIZE-1)/BLOCK_SIZE, 65535), BLOCK_SIZE, 0, s>>>(ow, dg, N);
    }
}

void launch_init_pers_norm(const int32_t* pv, const float* pval, float* pn, float inv, int ps, cudaStream_t s) {
    if (ps > 0) init_pers_norm_kernel<<<min((ps+BLOCK_SIZE-1)/BLOCK_SIZE, 65535), BLOCK_SIZE, 0, s>>>(pv, pval, pn, inv, ps);
}

void launch_init_uniform_pr(float* pr, float val, int N, cudaStream_t s) {
    if (N > 0) init_uniform_pr_kernel<<<min((N+BLOCK_SIZE-1)/BLOCK_SIZE, 65535), BLOCK_SIZE, 0, s>>>(pr, val, N);
}

void launch_dangling_sum_bitmask(const float* pr, const uint32_t* dg, float* res, int N, cudaStream_t s) {
    if (N > 0) dangling_sum_bitmask_kernel<<<min((N+BLOCK_SIZE-1)/BLOCK_SIZE, 1024), BLOCK_SIZE, 0, s>>>(pr, dg, res, N);
}

void launch_fused_base_diff(float* np, const float* op, const float* pn, const float* ds, float a, float oma, float* diff, int N, cudaStream_t s) {
    if (N > 0) fused_base_diff_kernel<<<min((N+BLOCK_SIZE-1)/BLOCK_SIZE, 1024), BLOCK_SIZE, 0, s>>>(np, op, pn, ds, a, oma, diff, N);
}

void launch_base_factor_only(float* np, const float* pn, const float* ds, float a, float oma, int N, cudaStream_t s) {
    if (N > 0) base_factor_only_kernel<<<min((N+BLOCK_SIZE-1)/BLOCK_SIZE, 1024), BLOCK_SIZE, 0, s>>>(np, pn, ds, a, oma, N);
}

}  





PageRankResult personalized_pagerank_mask(const graph32_t& graph,
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

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t N = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;
    const uint32_t* d_mask = graph.edge_mask;

    cache.ensure(N, E);

    float oma = 1.0f - alpha;
    cusparseSetStream(cache.handle, cache.main_stream);

    
    float h_psum = 0.0f;
    {
        std::vector<float> hv(personalization_size);
        cudaMemcpy(hv.data(), personalization_values, personalization_size * sizeof(float), cudaMemcpyDeviceToHost);
        for (std::size_t i = 0; i < personalization_size; i++) h_psum += hv[i];
    }
    float psi = (h_psum > 0.0f) ? (1.0f / h_psum) : 0.0f;

    float* d_ow = cache.d_ow;
    float* d_sw = cache.d_sw;
    uint32_t* d_dg = cache.d_dg;
    float* d_pn = cache.d_pn;
    float* d_old = cache.d_p0;
    float* d_new = cache.d_p1;
    float* d_ds = cache.d_ds;
    float* d_df = cache.d_df;
    float* d_one = cache.d_one;
    float* d_z = cache.d_z;

    float one = 1.0f, zero = 0.0f;
    cudaMemcpyAsync(d_one, &one, sizeof(float), cudaMemcpyHostToDevice, cache.main_stream);
    cudaMemcpyAsync(d_z, &zero, sizeof(float), cudaMemcpyHostToDevice, cache.main_stream);

    
    cudaMemsetAsync(d_ow, 0, N * sizeof(float), cache.main_stream);
    if (precomputed_vertex_out_weight_sums != nullptr)
        cudaMemcpyAsync(d_ow, precomputed_vertex_out_weight_sums, N * sizeof(float), cudaMemcpyDeviceToDevice, cache.main_stream);
    else
        launch_compute_out_weights(edge_weights, d_mask, d_indices, d_ow, E, cache.main_stream);
    launch_compute_scaled_weights(edge_weights, d_mask, d_indices, d_ow, d_sw, alpha, E, cache.main_stream);
    launch_compute_dangling_bitmask(d_ow, d_dg, N, cache.main_stream);
    cudaMemsetAsync(d_pn, 0, N * sizeof(float), cache.main_stream);
    launch_init_pers_norm(personalization_vertices, personalization_values, d_pn, psi, (int)personalization_size, cache.main_stream);

    if (initial_pageranks != nullptr)
        cudaMemcpyAsync(d_old, initial_pageranks, N * sizeof(float), cudaMemcpyDeviceToDevice, cache.main_stream);
    else
        launch_init_uniform_pr(d_old, 1.0f / (float)N, N, cache.main_stream);

    
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, N, N, E,
        (void*)d_offsets, (void*)d_indices, (void*)d_sw,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, N, d_old, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, N, d_new, CUDA_R_32F);

    cusparseSetPointerMode(cache.handle, CUSPARSE_POINTER_MODE_HOST);
    float h_one = 1.0f, h_zero = 0.0f;
    size_t bsz = 0;
    cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_one, matA, vecX, &h_zero, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bsz);

    if (bsz > 0) {
        cache.ensure_spmv_buf(bsz);
    }
    void* dbuf = (bsz > 0) ? cache.spmv_buf : nullptr;

    cusparseSpMV_preprocess(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_one, matA, vecX, &h_zero, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dbuf);
    cusparseSetPointerMode(cache.handle, CUSPARSE_POINTER_MODE_DEVICE);

    
    cudaEventRecord(cache.ev_main, cache.main_stream);
    cudaStreamWaitEvent(cache.aux_stream, cache.ev_main);

    
    size_t iters = 0;
    bool conv = false;
    int civ = (N > 100000) ? 4 : 1;

    for (size_t it = 0; it < max_iterations; it++) {
        
        if (it > 0) cudaStreamWaitEvent(cache.aux_stream, cache.ev_main);
        cudaMemsetAsync(d_ds, 0, sizeof(float), cache.aux_stream);
        launch_dangling_sum_bitmask(d_old, d_dg, d_ds, N, cache.aux_stream);
        cudaEventRecord(cache.ev_aux, cache.aux_stream);

        
        cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_one, matA, vecX, d_z, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, dbuf);

        
        cudaStreamWaitEvent(cache.main_stream, cache.ev_aux);

        iters = it + 1;
        bool check = ((it + 1) % civ == 0) || (it + 1 >= max_iterations);

        if (check) {
            cudaMemsetAsync(d_df, 0, sizeof(float), cache.main_stream);
            launch_fused_base_diff(d_new, d_old, d_pn, d_ds, alpha, oma, d_df, N, cache.main_stream);
            cudaMemcpyAsync(cache.h_diff, d_df, sizeof(float), cudaMemcpyDeviceToHost, cache.main_stream);
            cudaStreamSynchronize(cache.main_stream);
            if (*cache.h_diff < epsilon) { conv = true; break; }
        } else {
            launch_base_factor_only(d_new, d_pn, d_ds, alpha, oma, N, cache.main_stream);
        }

        cudaEventRecord(cache.ev_main, cache.main_stream);

        std::swap(d_old, d_new);
        cusparseDnVecSetValues(vecX, d_old);
        cusparseDnVecSetValues(vecY, d_new);
    }

    
    float* d_res = conv ? d_new : d_old;
    cudaMemcpyAsync(pageranks, d_res, N * sizeof(float), cudaMemcpyDeviceToDevice, cache.main_stream);

    cusparseSetPointerMode(cache.handle, CUSPARSE_POINTER_MODE_HOST);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cudaStreamSynchronize(cache.main_stream);
    cudaStreamSynchronize(cache.aux_stream);

    return PageRankResult{iters, conv};
}

}  
