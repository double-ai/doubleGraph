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
#include <cublas_v2.h>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

namespace aai {

namespace {


static constexpr int NUM_SMS = 58;
static constexpr int MAX_GRID = 696;  

static inline void check_cusparse(cusparseStatus_t st, const char* msg) {
  if (st != CUSPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuSPARSE error: ") + msg);
  }
}
static inline void check_cublas(cublasStatus_t st, const char* msg) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(std::string("cuBLAS error: ") + msg);
  }
}





static inline __device__ float warp_reduce_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

static inline __device__ float group_reduce_sum(float v, int group_size, unsigned mask) {
    
    for (int offset = group_size >> 1; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset, group_size);
    }
    return v;
}





__global__ void add_beta_l1diff_kernel_f(
    const float* __restrict__ x_old,
    float* __restrict__ x_new_inout, 
    const float* __restrict__ betas,
    float beta_scalar,
    int use_betas,
    int n,
    float* __restrict__ diff_out)
{
    float sum = 0.0f;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (use_betas) {
        for (int i = tid; i < n; i += stride) {
            float v = x_new_inout[i] + betas[i];
            float d = v - x_old[i];
            sum += fabsf(d);
            x_new_inout[i] = v;
        }
    } else {
        for (int i = tid; i < n; i += stride) {
            float v = x_new_inout[i] + beta_scalar;
            float d = v - x_old[i];
            sum += fabsf(d);
            x_new_inout[i] = v;
        }
    }

    sum = warp_reduce_sum(sum);

    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) shared[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) atomicAdd(diff_out, v);
    }
}





template<int GROUP_SIZE>
__global__ void katz_iter_group_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha,
    float beta_scalar,
    const float* __restrict__ betas,
    int use_betas,
    int n,
    float* __restrict__ diff_out)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = tid & 31;
    const int warp_global = tid >> 5;

    const int groups_per_warp = 32 / GROUP_SIZE;
    const int group_in_warp = lane / GROUP_SIZE;
    const int lane_in_group = lane & (GROUP_SIZE - 1);

    const int group_global = warp_global * groups_per_warp + group_in_warp;
    const int total_groups = ((gridDim.x * blockDim.x) >> 5) * groups_per_warp;

    float my_diff = 0.0f;

    
    const int group_lane_start = lane & ~(GROUP_SIZE - 1);
    const unsigned group_mask = (GROUP_SIZE == 32) ? 0xffffffffu
        : (((1u << GROUP_SIZE) - 1u) << group_lane_start);

    for (int v = group_global; v < n; v += total_groups) {
        int start = __ldg(&offsets[v]);
        int end = __ldg(&offsets[v + 1]);

        float sum = 0.0f;
        for (int j = start + lane_in_group; j < end; j += GROUP_SIZE) {
            int u = __ldg(&indices[j]);
            float w = __ldg(&weights[j]);
            sum = __fmaf_rn(w, __ldg(&x_old[u]), sum);
        }

        sum = group_reduce_sum(sum, GROUP_SIZE, group_mask);

        if (lane_in_group == 0) {
            float bv = use_betas ? __ldg(&betas[v]) : beta_scalar;
            float val = __fmaf_rn(alpha, sum, bv);
            x_new[v] = val;
            my_diff += fabsf(val - __ldg(&x_old[v]));
        }
    }

    my_diff = warp_reduce_sum(my_diff);
    __shared__ float shared[32];
    int warp = threadIdx.x >> 5;
    int lane0 = threadIdx.x & 31;
    if (lane0 == 0) shared[warp] = my_diff;
    __syncthreads();

    if (warp == 0) {
        float v = (lane0 < (blockDim.x >> 5)) ? shared[lane0] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane0 == 0) atomicAdd(diff_out, v);
    }
}





__global__ void scale_by_norm_kernel(float* x, const float* __restrict__ norm_ptr, int n) {
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float norm = norm_ptr[0];
        inv = (norm > 0.0f) ? (1.0f / norm) : 1.0f;
    }
    __syncthreads();

    float s = inv;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) x[i] *= s;
}





static inline int choose_grid_groups(int n, int block, int groups_per_block) {
    int grid = (n + groups_per_block - 1) / groups_per_block;
    if (grid < 1) grid = 1;
    if (grid > MAX_GRID) grid = MAX_GRID;
    if (grid > NUM_SMS) grid = (grid / NUM_SMS) * NUM_SMS;
    if (grid < 1) grid = 1;
    return grid;
}

static inline int choose_grid_linear(int n, int block) {
    int grid = (n + block - 1) / block;
    if (grid < 1) grid = 1;
    if (grid > MAX_GRID) grid = MAX_GRID;
    if (grid > NUM_SMS) grid = (grid / NUM_SMS) * NUM_SMS;
    if (grid < 1) grid = 1;
    return grid;
}





void launch_add_beta_l1diff(
    const float* x_old,
    float* x_new_inout,
    const float* betas,
    float beta_scalar,
    int use_betas,
    int n,
    float* diff_out,
    cudaStream_t stream)
{
    int block = 256;
    int grid = choose_grid_linear(n, block);
    add_beta_l1diff_kernel_f<<<grid, block, 0, stream>>>(x_old, x_new_inout, betas, beta_scalar, use_betas, n, diff_out);
}

void launch_katz_iter_group(
    const int32_t* offsets,
    const int32_t* indices,
    const float* weights,
    const float* x_old,
    float* x_new,
    float alpha,
    float beta_scalar,
    const float* betas,
    int use_betas,
    int n,
    int group_size,
    float* diff_out,
    cudaStream_t stream)
{
    int block = 256;
    if (group_size == 8) {
        int groups_per_block = block / 8;
        int grid = choose_grid_groups(n, block, groups_per_block);
        katz_iter_group_kernel<8><<<grid, block, 0, stream>>>(
            offsets, indices, weights, x_old, x_new, alpha, beta_scalar, betas, use_betas, n, diff_out);
    } else if (group_size == 16) {
        int groups_per_block = block / 16;
        int grid = choose_grid_groups(n, block, groups_per_block);
        katz_iter_group_kernel<16><<<grid, block, 0, stream>>>(
            offsets, indices, weights, x_old, x_new, alpha, beta_scalar, betas, use_betas, n, diff_out);
    } else {
        int groups_per_block = block / 32;
        int grid = choose_grid_groups(n, block, groups_per_block);
        katz_iter_group_kernel<32><<<grid, block, 0, stream>>>(
            offsets, indices, weights, x_old, x_new, alpha, beta_scalar, betas, use_betas, n, diff_out);
    }
}

void launch_scale_by_norm(float* x, const float* norm_ptr, int n, cudaStream_t stream) {
    int block = 256;
    int grid = choose_grid_linear(n, block);
    scale_by_norm_kernel<<<grid, block, 0, stream>>>(x, norm_ptr, n);
}





struct Cache : Cacheable {
    cusparseHandle_t cusparse = nullptr;
    cublasHandle_t cublas = nullptr;

    
    float* d_diff = nullptr;
    float* d_norm = nullptr;
    float* h_diff_pinned = nullptr;

    
    float* buf0 = nullptr;
    float* buf1 = nullptr;
    int64_t buf0_capacity = 0;
    int64_t buf1_capacity = 0;

    
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr;
    cusparseDnVecDescr_t vecY = nullptr;
    int64_t cached_n = -1;
    int64_t cached_nnz = -1;
    cusparseSpMVAlg_t cached_alg = CUSPARSE_SPMV_CSR_ALG2;

    
    void* spmv_workspace = nullptr;
    size_t spmv_workspace_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse);
        cublasCreate(&cublas);
        cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);
        cudaMalloc(&d_diff, sizeof(float));
        cudaMalloc(&d_norm, sizeof(float));
        cudaMallocHost((void**)&h_diff_pinned, sizeof(float));
    }

    void ensure_bufs(int64_t n) {
        if (buf0_capacity < n) {
            if (buf0) cudaFree(buf0);
            cudaMalloc(&buf0, n * sizeof(float));
            buf0_capacity = n;
        }
        if (buf1_capacity < n) {
            if (buf1) cudaFree(buf1);
            cudaMalloc(&buf1, n * sizeof(float));
            buf1_capacity = n;
        }
    }

    void destroy_spmv_cache() {
        if (vecX) { cusparseDestroyDnVec(vecX); vecX = nullptr; }
        if (vecY) { cusparseDestroyDnVec(vecY); vecY = nullptr; }
        if (matA) { cusparseDestroySpMat(matA); matA = nullptr; }
        cached_n = -1;
        cached_nnz = -1;
    }

    void ensure_spmv_descriptors(
        int32_t n, int32_t nnz,
        const int32_t* d_offsets, const int32_t* d_indices,
        const float* d_weights, float* x_old, float* x_new) {
        if (!matA || cached_n != n || cached_nnz != nnz) {
            destroy_spmv_cache();

            check_cusparse(
                cusparseCreateCsr(&matA, (int64_t)n, (int64_t)n, (int64_t)nnz,
                                  (void*)d_offsets, (void*)d_indices, (void*)d_weights,
                                  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
                "cusparseCreateCsr");

            check_cusparse(cusparseCreateDnVec(&vecX, (int64_t)n, (void*)x_old, CUDA_R_32F),
                          "cusparseCreateDnVec X");
            check_cusparse(cusparseCreateDnVec(&vecY, (int64_t)n, (void*)x_new, CUDA_R_32F),
                          "cusparseCreateDnVec Y");

            cached_n = n;
            cached_nnz = nnz;
        } else {
            check_cusparse(cusparseCsrSetPointers(matA, (void*)d_offsets, (void*)d_indices, (void*)d_weights),
                          "cusparseCsrSetPointers");
            check_cusparse(cusparseDnVecSetValues(vecX, (void*)x_old), "cusparseDnVecSetValues X");
            check_cusparse(cusparseDnVecSetValues(vecY, (void*)x_new), "cusparseDnVecSetValues Y");
        }
    }

    ~Cache() override {
        destroy_spmv_cache();
        if (cusparse) { cusparseDestroy(cusparse); cusparse = nullptr; }
        if (cublas) { cublasDestroy(cublas); cublas = nullptr; }
        if (d_diff) { cudaFree(d_diff); d_diff = nullptr; }
        if (d_norm) { cudaFree(d_norm); d_norm = nullptr; }
        if (h_diff_pinned) { cudaFreeHost(h_diff_pinned); h_diff_pinned = nullptr; }
        if (buf0) { cudaFree(buf0); buf0 = nullptr; }
        if (buf1) { cudaFree(buf1); buf1 = nullptr; }
        if (spmv_workspace) { cudaFree(spmv_workspace); spmv_workspace = nullptr; }
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

    int32_t n = graph.number_of_vertices;
    int32_t nnz = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const float* d_weights = edge_weights;
    bool use_betas = (betas != nullptr);

    cache.ensure_bufs((int64_t)n);
    float* x_old = cache.buf0;
    float* x_new = cache.buf1;

    cudaStream_t stream = cudaStreamPerThread;
    check_cusparse(cusparseSetStream(cache.cusparse, stream), "cusparseSetStream");
    check_cublas(cublasSetStream(cache.cublas, stream), "cublasSetStream");

    if (has_initial_guess) {
        cudaMemcpyAsync(x_old, centralities, (size_t)n * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemsetAsync(x_old, 0, (size_t)n * sizeof(float), stream);
    }

    int avg_deg = (n > 0) ? (nnz / n) : 0;

    
    bool use_custom = (avg_deg <= 32) && (n <= 200000);

    
    cusparseSpMVAlg_t alg = (avg_deg <= 8) ? CUSPARSE_SPMV_CSR_ALG1 : CUSPARSE_SPMV_CSR_ALG2;
    float spmv_alpha = alpha;
    float spmv_beta_val = 0.0f;
    void* dBuffer = nullptr;

    if (!use_custom) {
        cache.ensure_spmv_descriptors(n, nnz, d_offsets, d_indices, d_weights, x_old, x_new);

        
        size_t bufferSize = 0;
        check_cusparse(
            cusparseSpMV_bufferSize(cache.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &spmv_alpha, cache.matA, cache.vecX, &spmv_beta_val, cache.vecY,
                                    CUDA_R_32F, alg, &bufferSize),
            "cusparseSpMV_bufferSize");

        if (alg != cache.cached_alg || bufferSize > cache.spmv_workspace_capacity || !cache.spmv_workspace) {
            if (cache.spmv_workspace) cudaFree(cache.spmv_workspace);
            cache.spmv_workspace = nullptr;
            cache.spmv_workspace_capacity = 0;
            if (bufferSize > 0) {
                cudaMalloc(&cache.spmv_workspace, bufferSize);
            }
            cache.spmv_workspace_capacity = bufferSize;
            cache.cached_alg = alg;
        }

        dBuffer = cache.spmv_workspace;

        auto pre_st = cusparseSpMV_preprocess(
            cache.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &spmv_alpha, cache.matA, cache.vecX, &spmv_beta_val, cache.vecY,
            CUDA_R_32F, alg, dBuffer);
        if (pre_st != CUSPARSE_STATUS_SUCCESS && pre_st != CUSPARSE_STATUS_NOT_SUPPORTED) {
            check_cusparse(pre_st, "cusparseSpMV_preprocess");
        }
    }

    
    int check_every = 1;

    bool converged = false;
    size_t iters = 0;
    float* result_ptr = x_old;

    int group_size = (avg_deg <= 8) ? 8 : ((avg_deg <= 16) ? 16 : 32);

    for (size_t i = 0; i < max_iterations; ++i) {
        cudaMemsetAsync(cache.d_diff, 0, sizeof(float), stream);

        if (use_custom) {
            launch_katz_iter_group(d_offsets, d_indices, d_weights,
                                   x_old, x_new,
                                   alpha, beta, betas, use_betas ? 1 : 0,
                                   n, group_size, cache.d_diff, stream);
        } else {
            check_cusparse(cusparseDnVecSetValues(cache.vecX, (void*)x_old), "cusparseDnVecSetValues X");
            check_cusparse(cusparseDnVecSetValues(cache.vecY, (void*)x_new), "cusparseDnVecSetValues Y");

            check_cusparse(cusparseSpMV(cache.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &spmv_alpha, cache.matA, cache.vecX, &spmv_beta_val, cache.vecY,
                                        CUDA_R_32F, alg, dBuffer),
                          "cusparseSpMV");

            launch_add_beta_l1diff(x_old, x_new, betas, beta, use_betas ? 1 : 0, n, cache.d_diff, stream);
        }

        iters++;
        result_ptr = x_new;

        bool do_check = (epsilon > 0.0f) && ((iters % (size_t)check_every) == 0 || (iters == max_iterations));
        if (do_check) {
            cudaMemcpyAsync(cache.h_diff_pinned, cache.d_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (*cache.h_diff_pinned < epsilon) {
                converged = true;
                break;
            }
        }

        std::swap(x_old, x_new);
        result_ptr = x_old;
    }

    
    cudaMemcpyAsync(centralities, result_ptr, (size_t)n * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    if (normalize && n > 0) {
        check_cublas(cublasSnrm2(cache.cublas, n, centralities, 1, cache.d_norm), "cublasSnrm2");
        launch_scale_by_norm(centralities, cache.d_norm, n, stream);
    }

    return katz_centrality_result_t{iters, converged};
}

}  
