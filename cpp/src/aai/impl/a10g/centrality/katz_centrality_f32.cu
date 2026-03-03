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
#include <cstdint>
#include <cmath>

namespace aai {

namespace {




__global__ void fill_beta_kernel(
    float* __restrict__ x,
    const float* __restrict__ betas,
    float beta_scalar,
    int32_t n
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        x[i] = (betas != nullptr) ? betas[i] : beta_scalar;
    }
}



__global__ void add_beta_diff_kernel(
    float* __restrict__ x_new,
    const float* __restrict__ x_old,
    const float* __restrict__ betas,
    float beta_scalar,
    int32_t num_vertices,
    float* __restrict__ diff_global
) {
    const int warps_per_block = blockDim.x >> 5;
    const int local_warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    extern __shared__ float s_diff[];

    float my_diff = 0.0f;

    
    int n4 = num_vertices >> 2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float4* xn4 = (float4*)x_new;
    const float4* xo4 = (const float4*)x_old;
    const float4* b4 = (betas != nullptr) ? (const float4*)betas : nullptr;

    for (int i = tid; i < n4; i += stride) {
        float4 xn = xn4[i];
        float4 xo = xo4[i];
        float4 b;
        if (b4) {
            b = b4[i];
        } else {
            b = make_float4(beta_scalar, beta_scalar, beta_scalar, beta_scalar);
        }
        xn.x += b.x; xn.y += b.y; xn.z += b.z; xn.w += b.w;
        xn4[i] = xn;
        my_diff += fabsf(xn.x - xo.x) + fabsf(xn.y - xo.y) +
                   fabsf(xn.z - xo.z) + fabsf(xn.w - xo.w);
    }

    
    for (int i = n4 * 4 + tid; i < num_vertices; i += stride) {
        float beta_v = (betas != nullptr) ? betas[i] : beta_scalar;
        float nv = x_new[i] + beta_v;
        x_new[i] = nv;
        my_diff += fabsf(nv - x_old[i]);
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        my_diff += __shfl_down_sync(0xffffffff, my_diff, offset);

    if (lane == 0)
        s_diff[local_warp] = my_diff;
    __syncthreads();

    if (local_warp == 0) {
        my_diff = (lane < warps_per_block) ? s_diff[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            my_diff += __shfl_down_sync(0xffffffff, my_diff, offset);
        if (lane == 0 && my_diff > 0.0f)
            atomicAdd(diff_global, my_diff);
    }
}


__global__ void add_beta_kernel(
    float* __restrict__ x_new,
    const float* __restrict__ betas,
    float beta_scalar,
    int32_t num_vertices
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vertices; i += blockDim.x * gridDim.x) {
        float beta_v = (betas != nullptr) ? betas[i] : beta_scalar;
        x_new[i] += beta_v;
    }
}


__global__ void diff_only_kernel(
    const float* __restrict__ x_new,
    const float* __restrict__ x_old,
    int32_t num_vertices,
    float* __restrict__ diff_global
) {
    const int warps_per_block = blockDim.x >> 5;
    const int local_warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    extern __shared__ float s_diff[];
    float my_diff = 0.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vertices; i += blockDim.x * gridDim.x) {
        my_diff += fabsf(x_new[i] - x_old[i]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        my_diff += __shfl_down_sync(0xffffffff, my_diff, offset);

    if (lane == 0) s_diff[local_warp] = my_diff;
    __syncthreads();

    if (local_warp == 0) {
        my_diff = (lane < warps_per_block) ? s_diff[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            my_diff += __shfl_down_sync(0xffffffff, my_diff, offset);
        if (lane == 0 && my_diff > 0.0f)
            atomicAdd(diff_global, my_diff);
    }
}


__global__ void normalize_sq_kernel(const float* __restrict__ x, int32_t n, float* __restrict__ norm_sq) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float val = x[i];
        sum += val * val;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(norm_sq, sum);
}

__global__ void normalize_div_kernel(float* __restrict__ x, int32_t n, const float* __restrict__ norm_sq) {
    float inv_norm = rsqrtf(*norm_sq);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        x[i] *= inv_norm;
}



void launch_fill_beta(float* x, const float* betas, float beta_scalar, int32_t n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    if (blocks > 2048) blocks = 2048;
    fill_beta_kernel<<<blocks, 256, 0, stream>>>(x, betas, beta_scalar, n);
}

void launch_add_beta_diff(
    float* x_new, const float* x_old, const float* betas,
    float beta_scalar, int32_t num_vertices, float* diff_global,
    cudaStream_t stream
) {
    int blocks = (num_vertices + 1023) / 1024;
    if (blocks > 2048) blocks = 2048;
    int smem = (1024 / 32) * sizeof(float);
    add_beta_diff_kernel<<<blocks, 1024, smem, stream>>>(
        x_new, x_old, betas, beta_scalar, num_vertices, diff_global
    );
}

void launch_add_beta(float* x_new, const float* betas, float beta_scalar,
    int32_t num_vertices, cudaStream_t stream) {
    int blocks = (num_vertices + 255) / 256;
    if (blocks > 2048) blocks = 2048;
    add_beta_kernel<<<blocks, 256, 0, stream>>>(x_new, betas, beta_scalar, num_vertices);
}

void launch_normalize(float* x, int32_t n, float* norm_sq, cudaStream_t stream) {
    cudaMemsetAsync(norm_sq, 0, sizeof(float), stream);
    int blocks1 = (n + 255) / 256;
    if (blocks1 > 1024) blocks1 = 1024;
    normalize_sq_kernel<<<blocks1, 256, 0, stream>>>(x, n, norm_sq);
    int blocks2 = (n + 255) / 256;
    normalize_div_kernel<<<blocks2, 256, 0, stream>>>(x, n, norm_sq);
}



struct Cache : Cacheable {
    cusparseHandle_t handle_ = nullptr;
    float* h_diff_pinned_ = nullptr;
    float* d_alpha_ = nullptr;
    float* d_zero_ = nullptr;
    float* d_diff_ = nullptr;
    size_t l2_persist_size_ = 0;

    float* buf0_ = nullptr;
    float* buf1_ = nullptr;
    void* spmv_buffer_ = nullptr;

    int64_t buf0_capacity_ = 0;
    int64_t buf1_capacity_ = 0;
    size_t spmv_buffer_capacity_ = 0;

    Cache() {
        cusparseCreate(&handle_);
        cusparseSetPointerMode(handle_, CUSPARSE_POINTER_MODE_DEVICE);
        cudaMallocHost(&h_diff_pinned_, sizeof(float));
        cudaMalloc(&d_alpha_, sizeof(float));
        cudaMalloc(&d_zero_, sizeof(float));
        cudaMalloc(&d_diff_, sizeof(float));
        float zero = 0.0f;
        cudaMemcpy(d_zero_, &zero, sizeof(float), cudaMemcpyHostToDevice);

        int device;
        cudaGetDevice(&device);
        int l2_size;
        cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, device);
        l2_persist_size_ = (size_t)(l2_size * 0.75);
    }

    void ensure_buffers(int32_t num_vertices) {
        if (buf0_capacity_ < num_vertices) {
            if (buf0_) cudaFree(buf0_);
            cudaMalloc(&buf0_, (size_t)num_vertices * sizeof(float));
            buf0_capacity_ = num_vertices;
        }
        if (buf1_capacity_ < num_vertices) {
            if (buf1_) cudaFree(buf1_);
            cudaMalloc(&buf1_, (size_t)num_vertices * sizeof(float));
            buf1_capacity_ = num_vertices;
        }
    }

    void ensure_spmv_buffer(size_t size) {
        if (spmv_buffer_capacity_ < size) {
            if (spmv_buffer_) cudaFree(spmv_buffer_);
            cudaMalloc(&spmv_buffer_, size);
            spmv_buffer_capacity_ = size;
        }
    }

    ~Cache() override {
        if (handle_) cusparseDestroy(handle_);
        if (h_diff_pinned_) cudaFreeHost(h_diff_pinned_);
        if (d_alpha_) cudaFree(d_alpha_);
        if (d_zero_) cudaFree(d_zero_);
        if (d_diff_) cudaFree(d_diff_);
        if (buf0_) cudaFree(buf0_);
        if (buf1_) cudaFree(buf1_);
        if (spmv_buffer_) cudaFree(spmv_buffer_);
    }
};

}  

katz_centrality_result_t katz_centrality(const graph32_t& graph,
                     const float* edge_weights,
                     float* centralities,
                     float alpha,
                     float beta,
                     const float* betas,
                     float epsilon,
                     std::size_t max_iterations,
                     bool has_initial_guess,
                     bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    cudaStream_t stream = 0;
    cusparseSetStream(cache.handle_, stream);

    cudaMemcpyAsync(cache.d_alpha_, &alpha, sizeof(float), cudaMemcpyHostToDevice, stream);

    cache.ensure_buffers(num_vertices);

    float* x_cur = cache.buf0_;
    float* x_next = cache.buf1_;

    if (has_initial_guess) {
        cudaMemcpyAsync(x_cur, centralities,
                   (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemsetAsync(x_cur, 0, (size_t)num_vertices * sizeof(float), stream);
    }

    
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(
        &matA, num_vertices, num_vertices, num_edges,
        (void*)d_offsets, (void*)d_indices, (void*)edge_weights,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, num_vertices, x_cur, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, num_vertices, x_next, CUDA_R_32F);

    cusparseSpMVAlg_t alg = CUSPARSE_SPMV_CSR_ALG2;

    size_t bufferSize = 0;
    float h_one = 1.0f, h_zero = 0.0f;
    cusparseSpMV_bufferSize(
        cache.handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_one, matA, vecX, &h_zero, vecY,
        CUDA_R_32F, alg, &bufferSize);

    if (bufferSize > 0) {
        cache.ensure_spmv_buffer(bufferSize);
    }

    cusparseSpMV_preprocess(
        cache.handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
        cache.d_alpha_, matA, vecX, cache.d_zero_, vecY,
        CUDA_R_32F, alg, cache.spmv_buffer_);

    
    size_t x_size = (size_t)num_vertices * sizeof(float);
    bool use_l2_persist = (x_size <= cache.l2_persist_size_ && x_size > 0);

    size_t iterations = 0;
    bool converged = false;

    for (size_t iter = 0; iter < max_iterations; ++iter) {
        
        if (iter == 0 && !has_initial_guess) {
            launch_fill_beta(x_next, betas, beta, num_vertices, stream);
            iterations = 1;
            float* tmp = x_cur; x_cur = x_next; x_next = tmp;
            continue;
        }

        
        if (use_l2_persist) {
            cudaStreamAttrValue attr;
            attr.accessPolicyWindow.base_ptr = (void*)x_cur;
            attr.accessPolicyWindow.num_bytes = x_size;
            attr.accessPolicyWindow.hitRatio = 1.0f;
            attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
        }

        cusparseDnVecSetValues(vecX, x_cur);
        cusparseDnVecSetValues(vecY, x_next);

        cusparseSpMV(
            cache.handle_, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha_, matA, vecX, cache.d_zero_, vecY,
            CUDA_R_32F, alg, cache.spmv_buffer_);

        cudaMemsetAsync(cache.d_diff_, 0, sizeof(float), stream);
        launch_add_beta_diff(x_next, x_cur, betas, beta,
                            num_vertices, cache.d_diff_, stream);

        
        if (use_l2_persist) {
            cudaStreamAttrValue attr;
            attr.accessPolicyWindow.num_bytes = 0;
            cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
        }

        cudaMemcpyAsync(cache.h_diff_pinned_, cache.d_diff_, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations = iter + 1;
        float* tmp = x_cur; x_cur = x_next; x_next = tmp;

        if (*cache.h_diff_pinned_ < epsilon) {
            converged = true;
            break;
        }
    }

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    if (normalize) {
        launch_normalize(x_cur, num_vertices, cache.d_diff_, stream);
    }

    
    cudaMemcpyAsync(centralities, x_cur,
                   (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    return {iterations, converged};
}

}  
