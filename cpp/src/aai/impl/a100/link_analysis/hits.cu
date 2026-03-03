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

static constexpr int MAX_GRID = 1024;
static constexpr int BLK = 256;

struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    float* d_alpha = nullptr;
    float* d_beta = nullptr;

    
    float* d_p1 = nullptr;
    float* d_p2 = nullptr;
    float* d_p3 = nullptr;
    float* d_res = nullptr;
    unsigned int* d_c1 = nullptr;
    unsigned int* d_c2 = nullptr;
    float* h_diff = nullptr;

    
    float* d_ones = nullptr;        size_t ones_cap = 0;
    float* d_buf_b = nullptr;       size_t bufb_cap = 0;
    int32_t* d_csr_off = nullptr;   size_t csro_cap = 0;
    int32_t* d_csr_idx = nullptr;   size_t csri_cap = 0;
    void* d_tbuf = nullptr;         size_t tbuf_cap = 0;
    void* d_sbuf1 = nullptr;        size_t sb1_cap = 0;
    void* d_sbuf2 = nullptr;        size_t sb2_cap = 0;

    Cache() {
        cusparseCreate(&handle);
        cudaStreamCreate(&stream);
        cusparseSetStream(handle, stream);
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);

        cudaMalloc(&d_alpha, sizeof(float));
        cudaMalloc(&d_beta, sizeof(float));
        float one = 1.f, zero = 0.f;
        cudaMemcpy(d_alpha, &one, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, &zero, sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_p1, MAX_GRID * sizeof(float));
        cudaMalloc(&d_p2, MAX_GRID * sizeof(float));
        cudaMalloc(&d_p3, MAX_GRID * sizeof(float));
        cudaMalloc(&d_res, 4 * sizeof(float));
        cudaMalloc(&d_c1, sizeof(unsigned int));
        cudaMalloc(&d_c2, sizeof(unsigned int));
        cudaMemset(d_c1, 0, sizeof(unsigned int));
        cudaMemset(d_c2, 0, sizeof(unsigned int));
        cudaMallocHost(&h_diff, sizeof(float));
    }

    ~Cache() override {
        if (handle) cusparseDestroy(handle);
        if (stream) cudaStreamDestroy(stream);
        auto F = [](auto& p) { if (p) { cudaFree(p); p = nullptr; } };
        F(d_alpha); F(d_beta); F(d_ones); F(d_buf_b);
        F(d_csr_off); F(d_csr_idx); F(d_tbuf); F(d_sbuf1); F(d_sbuf2);
        F(d_p1); F(d_p2); F(d_p3); F(d_res); F(d_c1); F(d_c2);
        if (h_diff) { cudaFreeHost(h_diff); h_diff = nullptr; }
    }

    template<typename T> void gr(T*& p, size_t& c, size_t n) {
        if (n > c) { if (p) cudaFree(p); cudaMalloc(&p, n * sizeof(T)); c = n; }
    }
    void grv(void*& p, size_t& c, size_t n) {
        if (n > c) { if (p) cudaFree(p); cudaMalloc(&p, n); c = n; }
    }
};



struct MaxOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return fmaxf(a, b);
    }
};

__global__ void fill_kernel(float* data, float val, int32_t n) {
    for (int i = blockIdx.x * BLK + threadIdx.x; i < n; i += gridDim.x * BLK)
        data[i] = val;
}

__global__ void copy_kernel(float* __restrict__ dst, const float* __restrict__ src, int32_t n) {
    for (int i = blockIdx.x * BLK + threadIdx.x; i < n; i += gridDim.x * BLK)
        dst[i] = src[i];
}

__global__ void dual_max_kernel(
    const float* __restrict__ a1, const float* __restrict__ a2, int32_t n,
    float* __restrict__ p1, float* __restrict__ p2,
    float* __restrict__ res, unsigned int* __restrict__ cnt)
{
    typedef cub::BlockReduce<float, BLK> BR;
    __shared__ typename BR::TempStorage ts;

    float m1 = -1e38f, m2 = -1e38f;
    for (int i = blockIdx.x * BLK + threadIdx.x; i < n; i += gridDim.x * BLK) {
        m1 = fmaxf(m1, a1[i]);
        m2 = fmaxf(m2, a2[i]);
    }
    float b1 = BR(ts).Reduce(m1, MaxOp());
    __syncthreads();
    float b2 = BR(ts).Reduce(m2, MaxOp());

    if (threadIdx.x == 0) { p1[blockIdx.x] = b1; p2[blockIdx.x] = b2; }
    __threadfence();

    __shared__ bool last;
    if (threadIdx.x == 0) {
        unsigned t = atomicAdd(cnt, 1u);
        last = (t == gridDim.x - 1);
    }
    __syncthreads();

    if (last) {
        float r1 = -1e38f, r2 = -1e38f;
        for (int i = threadIdx.x; i < (int)gridDim.x; i += BLK) {
            r1 = fmaxf(r1, p1[i]);
            r2 = fmaxf(r2, p2[i]);
        }
        r1 = BR(ts).Reduce(r1, MaxOp());
        __syncthreads();
        r2 = BR(ts).Reduce(r2, MaxOp());
        if (threadIdx.x == 0) { res[0] = r1; res[1] = r2; *cnt = 0; }
    }
}

__global__ void norm2_diff_kernel(
    float* __restrict__ a1, float* __restrict__ a2,
    const float* __restrict__ prev, const float* __restrict__ maxv,
    int32_t n,
    float* __restrict__ pd, float* __restrict__ diff_out, unsigned int* __restrict__ cnt)
{
    typedef cub::BlockReduce<float, BLK> BR;
    __shared__ typename BR::TempStorage ts;

    float inv1 = (maxv[0] > 0.f) ? (1.f / maxv[0]) : 0.f;
    float inv2 = (maxv[1] > 0.f) ? (1.f / maxv[1]) : 0.f;

    float ld = 0.f;
    for (int i = blockIdx.x * BLK + threadIdx.x; i < n; i += gridDim.x * BLK) {
        float h = a1[i] * inv1;
        a1[i] = h;
        a2[i] *= inv2;
        ld += fabsf(h - prev[i]);
    }
    float bd = BR(ts).Sum(ld);
    if (threadIdx.x == 0) pd[blockIdx.x] = bd;
    __threadfence();

    __shared__ bool last;
    if (threadIdx.x == 0) {
        unsigned t = atomicAdd(cnt, 1u);
        last = (t == gridDim.x - 1);
    }
    __syncthreads();

    if (last) {
        float s = 0.f;
        for (int i = threadIdx.x; i < (int)gridDim.x; i += BLK)
            s += pd[i];
        s = BR(ts).Sum(s);
        if (threadIdx.x == 0) { *diff_out = s; *cnt = 0; }
    }
}

__global__ void sum_kernel(
    const float* __restrict__ data, int32_t n,
    float* __restrict__ ps, float* __restrict__ res, unsigned int* __restrict__ cnt)
{
    typedef cub::BlockReduce<float, BLK> BR;
    __shared__ typename BR::TempStorage ts;

    float ls = 0.f;
    for (int i = blockIdx.x * BLK + threadIdx.x; i < n; i += gridDim.x * BLK)
        ls += data[i];
    float bs = BR(ts).Sum(ls);
    if (threadIdx.x == 0) ps[blockIdx.x] = bs;
    __threadfence();

    __shared__ bool last;
    if (threadIdx.x == 0) {
        unsigned t = atomicAdd(cnt, 1u);
        last = (t == gridDim.x - 1);
    }
    __syncthreads();

    if (last) {
        float s = 0.f;
        for (int i = threadIdx.x; i < (int)gridDim.x; i += BLK)
            s += ps[i];
        s = BR(ts).Sum(s);
        if (threadIdx.x == 0) { *res = s; *cnt = 0; }
    }
}

__global__ void div_kernel(float* data, const float* scalar, int32_t n) {
    float s = *scalar;
    if (s <= 0.f) return;
    float inv = 1.f / s;
    for (int i = blockIdx.x * BLK + threadIdx.x; i < n; i += gridDim.x * BLK)
        data[i] *= inv;
}



void launch_fill(float* d, float v, int32_t n, int g, cudaStream_t s) {
    if (n <= 0) return;
    fill_kernel<<<g, BLK, 0, s>>>(d, v, n);
}

void launch_copy(float* dst, const float* src, int32_t n, int g, cudaStream_t s) {
    if (n <= 0) return;
    copy_kernel<<<g, BLK, 0, s>>>(dst, src, n);
}

void launch_dual_max(const float* a1, const float* a2, int32_t n,
                     float* p1, float* p2, float* res, unsigned int* cnt,
                     int g, cudaStream_t s) {
    dual_max_kernel<<<g, BLK, 0, s>>>(a1, a2, n, p1, p2, res, cnt);
}

void launch_norm2_diff(float* a1, float* a2, const float* prev, const float* maxv,
                       int32_t n, float* pd, float* diff_out, unsigned int* cnt,
                       int g, cudaStream_t s) {
    norm2_diff_kernel<<<g, BLK, 0, s>>>(a1, a2, prev, maxv, n, pd, diff_out, cnt);
}

void launch_sum(const float* d, int32_t n, float* ps, float* res, unsigned int* cnt,
                int g, cudaStream_t s) {
    sum_kernel<<<g, BLK, 0, s>>>(d, n, ps, res, cnt);
}

void launch_div(float* d, const float* sc, int32_t n, int g, cudaStream_t s) {
    div_kernel<<<g, BLK, 0, s>>>(d, sc, n);
}

}  

HitsResult hits(const graph32_t& graph,
                float* hubs,
                float* authorities,
                float epsilon,
                std::size_t max_iterations,
                bool has_initial_hubs_guess,
                bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;

    if (nv == 0) {
        return HitsResult{max_iterations, false, 1e30f};
    }

    float* d_hubs = hubs;
    float* d_auth = authorities;
    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;

    int grid = std::min((nv + 255) / 256, MAX_GRID);
    if (grid < 1) grid = 1;

    
    if (has_initial_hubs_guess) {
        
        launch_sum(d_hubs, nv, cache.d_p3, cache.d_res + 3, cache.d_c2, grid, cache.stream);
        launch_div(d_hubs, cache.d_res + 3, nv, grid, cache.stream);
    } else {
        launch_fill(d_hubs, 1.f / (float)nv, nv, grid, cache.stream);
    }

    
    cache.gr(cache.d_ones, cache.ones_cap, (size_t)ne);
    cache.gr(cache.d_buf_b, cache.bufb_cap, (size_t)nv);
    cache.gr(cache.d_csr_off, cache.csro_cap, (size_t)(nv + 1));
    cache.gr(cache.d_csr_idx, cache.csri_cap, (size_t)ne);

    int egrid = std::min((ne + 255) / 256, MAX_GRID);
    if (egrid < 1) egrid = 1;
    launch_fill(cache.d_ones, 1.f, ne, egrid, cache.stream);

    
    
    {
        size_t tsz = 0;
        cusparseCsr2cscEx2_bufferSize(cache.handle,
            nv, nv, ne, cache.d_ones, (int*)d_off, (int*)d_idx,
            cache.d_ones, (int*)cache.d_csr_off, (int*)cache.d_csr_idx,
            CUDA_R_32F, CUSPARSE_ACTION_SYMBOLIC,
            CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &tsz);
        cache.grv(cache.d_tbuf, cache.tbuf_cap, tsz > 0 ? tsz : 4);

        cusparseCsr2cscEx2(cache.handle,
            nv, nv, ne, cache.d_ones, (int*)d_off, (int*)d_idx,
            cache.d_ones, (int*)cache.d_csr_off, (int*)cache.d_csr_idx,
            CUDA_R_32F, CUSPARSE_ACTION_SYMBOLIC,
            CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, cache.d_tbuf);
    }

    
    
    cusparseSpMatDescr_t csr_at, csr_a;
    cusparseCreateCsr(&csr_at, nv, nv, ne,
        (void*)d_off, (void*)d_idx, (void*)cache.d_ones,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
    cusparseCreateCsr(&csr_a, nv, nv, ne,
        (void*)cache.d_csr_off, (void*)cache.d_csr_idx, (void*)cache.d_ones,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    
    float* buf_a = d_hubs;      
    float* buf_b = cache.d_buf_b;  
    float* prev = buf_a;
    float* curr = buf_b;

    
    cusparseDnVecDescr_t vhub, vauth;
    cusparseCreateDnVec(&vhub, nv, prev, CUDA_R_32F);
    cusparseCreateDnVec(&vauth, nv, d_auth, CUDA_R_32F);

    
    float h1 = 1.f, h0 = 0.f;
    size_t bs1 = 0, bs2 = 0;
    cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h1, csr_at, vhub, &h0, vauth, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bs1);
    cusparseDnVecSetValues(vhub, curr);
    cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h1, csr_a, vauth, &h0, vhub, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bs2);
    cache.grv(cache.d_sbuf1, cache.sb1_cap, bs1 > 0 ? bs1 : 4);
    cache.grv(cache.d_sbuf2, cache.sb2_cap, bs2 > 0 ? bs2 : 4);

    
    cusparseDnVecSetValues(vhub, prev);
    cusparseSpMV_preprocess(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        cache.d_alpha, csr_at, vhub, cache.d_beta, vauth,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.d_sbuf1);
    cusparseDnVecSetValues(vhub, curr);
    cusparseSpMV_preprocess(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        cache.d_alpha, csr_a, vauth, cache.d_beta, vhub,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.d_sbuf2);

    
    float tol = (float)nv * epsilon;
    float diff = 1e30f;
    size_t iter = 0;

    while (iter < max_iterations) {
        
        cusparseDnVecSetValues(vhub, prev);
        cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha, csr_at, vhub, cache.d_beta, vauth,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.d_sbuf1);

        
        cusparseDnVecSetValues(vhub, curr);
        cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha, csr_a, vauth, cache.d_beta, vhub,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.d_sbuf2);

        
        launch_dual_max(curr, d_auth, nv, cache.d_p1, cache.d_p2, cache.d_res, cache.d_c1, grid, cache.stream);
        launch_norm2_diff(curr, d_auth, prev, cache.d_res, nv, cache.d_p3, cache.d_res + 2, cache.d_c2, grid, cache.stream);

        
        cudaMemcpyAsync(cache.h_diff, cache.d_res + 2, sizeof(float), cudaMemcpyDeviceToHost, cache.stream);
        cudaStreamSynchronize(cache.stream);
        diff = *cache.h_diff;

        std::swap(prev, curr);
        iter++;
        if (diff < tol) break;
    }

    
    if (prev != buf_a) {
        launch_copy(buf_a, prev, nv, grid, cache.stream);
    }

    
    if (normalize) {
        launch_sum(d_hubs, nv, cache.d_p3, cache.d_res + 3, cache.d_c2, grid, cache.stream);
        launch_div(d_hubs, cache.d_res + 3, nv, grid, cache.stream);
        launch_sum(d_auth, nv, cache.d_p3, cache.d_res + 3, cache.d_c2, grid, cache.stream);
        launch_div(d_auth, cache.d_res + 3, nv, grid, cache.stream);
    }

    cusparseDestroyDnVec(vhub);
    cusparseDestroyDnVec(vauth);
    cusparseDestroySpMat(csr_at);
    cusparseDestroySpMat(csr_a);
    cudaStreamSynchronize(cache.stream);

    return HitsResult{iter, diff < tol, diff};
}

}  
