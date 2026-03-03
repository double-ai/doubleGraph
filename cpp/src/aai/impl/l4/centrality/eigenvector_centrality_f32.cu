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
#include <cooperative_groups.h>
#include <cusparse.h>
#include <cstdint>
#include <cstddef>

namespace aai {

namespace {

namespace cg = cooperative_groups;





__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__device__ __forceinline__ void block_reduce_atomicadd(float val, float* global_acc) {
    val = warp_reduce_sum(val);
    __shared__ float s_warp[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;
    if (lane == 0) s_warp[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lane < nwarps) ? s_warp[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) atomicAdd(global_acc, val);
    }
}





__global__ void __launch_bounds__(256)
fused_postprocess_kernel(
    float* __restrict__ x_new,
    const float* __restrict__ x_old,
    int32_t n,
    float* __restrict__ norm_sq,
    float* __restrict__ diff,
    bool compute_diff)
{
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (tid == 0) *norm_sq = 0.0f;
    if (compute_diff && tid == 0) *diff = 0.0f;
    grid.sync();

    for (int v = tid; v < n; v += stride) {
        float val = x_new[v] + x_old[v];
        x_new[v] = val;
        atomicAdd(norm_sq, val * val);
    }

    grid.sync();

    float inv_norm = (*norm_sq > 0.0f) ? rsqrtf(*norm_sq) : 0.0f;

    if (compute_diff) {
        for (int v = tid; v < n; v += stride) {
            float nv = x_new[v] * inv_norm;
            x_new[v] = nv;
            atomicAdd(diff, fabsf(nv - x_old[v]));
        }
    } else {
        for (int v = tid; v < n; v += stride) {
            x_new[v] *= inv_norm;
        }
    }
}

__global__ void __launch_bounds__(256)
fused_postprocess_v2_kernel(
    float* __restrict__ x_new,
    const float* __restrict__ x_old,
    int32_t n,
    float* __restrict__ norm_sq,
    float* __restrict__ diff,
    bool compute_diff)
{
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (tid == 0) *norm_sq = 0.0f;
    if (compute_diff && tid == 0) *diff = 0.0f;
    grid.sync();

    float local_norm_sq = 0.0f;
    for (int v = tid; v < n; v += stride) {
        float val = x_new[v] + x_old[v];
        x_new[v] = val;
        local_norm_sq += val * val;
    }
    block_reduce_atomicadd(local_norm_sq, norm_sq);

    grid.sync();

    float inv_norm = (*norm_sq > 0.0f) ? rsqrtf(*norm_sq) : 0.0f;

    if (compute_diff) {
        float local_diff = 0.0f;
        for (int v = tid; v < n; v += stride) {
            float nv = x_new[v] * inv_norm;
            x_new[v] = nv;
            local_diff += fabsf(nv - x_old[v]);
        }
        block_reduce_atomicadd(local_diff, diff);
    } else {
        for (int v = tid; v < n; v += stride) {
            x_new[v] *= inv_norm;
        }
    }
}

__global__ void init_kernel(float* x, int32_t n, float val) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) x[v] = val;
}

__global__ void add_identity_l2_kernel(
    float* __restrict__ y, const float* __restrict__ x,
    int32_t n, float* __restrict__ norm_sq) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (v < n) { val = y[v] + x[v]; y[v] = val; }
    block_reduce_atomicadd(val * val, norm_sq);
}

__global__ void normalize_kernel(float* __restrict__ x_new, int32_t n, const float* __restrict__ norm_sq) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        float nsq = *norm_sq;
        float inv = (nsq > 0.0f) ? rsqrtf(nsq) : 0.0f;
        x_new[v] *= inv;
    }
}

__global__ void normalize_diff_kernel(
    float* __restrict__ x_new, const float* __restrict__ x_old,
    int32_t n, const float* __restrict__ norm_sq, float* __restrict__ diff) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float d = 0.0f;
    if (v < n) {
        float nsq = *norm_sq;
        float inv = (nsq > 0.0f) ? rsqrtf(nsq) : 0.0f;
        float nv = x_new[v] * inv;
        x_new[v] = nv;
        d = fabsf(nv - x_old[v]);
    }
    block_reduce_atomicadd(d, diff);
}





int get_coop_max_blocks(int block_size) {
    int max_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, fused_postprocess_v2_kernel, block_size, 0);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    return max_blocks_per_sm * num_sms;
}





struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    float* d_scratch = nullptr;
    int coop_max_blocks = 0;

    float* buf = nullptr;
    int64_t buf_capacity = 0;

    void* spmv_buffer = nullptr;
    std::size_t spmv_buffer_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
        cudaMalloc(&d_scratch, 4 * sizeof(float));
        float h_vals[4] = {0.0f, 0.0f, 1.0f, 0.0f};
        cudaMemcpy(d_scratch, h_vals, 4 * sizeof(float), cudaMemcpyHostToDevice);
        coop_max_blocks = get_coop_max_blocks(256);
    }

    void ensure_buf(int64_t n) {
        if (buf_capacity < n) {
            if (buf) cudaFree(buf);
            cudaMalloc(&buf, n * sizeof(float));
            buf_capacity = n;
        }
    }

    void ensure_spmv_buffer(std::size_t size) {
        if (spmv_buffer_capacity < size) {
            if (spmv_buffer) cudaFree(spmv_buffer);
            cudaMalloc(&spmv_buffer, size);
            spmv_buffer_capacity = size;
        }
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (d_scratch) cudaFree(d_scratch);
        if (buf) cudaFree(buf);
        if (spmv_buffer) cudaFree(spmv_buffer);
    }
};

}  

eigenvector_centrality_result_t eigenvector_centrality(const graph32_t& graph,
                            const float* edge_weights,
                            float* centralities,
                            float epsilon,
                            std::size_t max_iterations,
                            const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    cudaStream_t stream = 0;
    cusparseSetStream(cache.cusparse_handle, stream);
    float threshold = (float)num_vertices * epsilon;

    cache.ensure_buf(num_vertices);

    float* x_a = centralities;
    float* x_b = cache.buf;
    float* d_norm_sq = cache.d_scratch;
    float* d_diff = cache.d_scratch + 1;
    float* d_alpha = cache.d_scratch + 2;
    float* d_beta = cache.d_scratch + 3;

    
    cusparseSpMatDescr_t mat_descr;
    cusparseCreateCsr(&mat_descr, num_vertices, num_vertices, num_edges,
                      (void*)d_offsets, (void*)d_indices, (void*)edge_weights,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t x_descr, y_descr;
    cusparseCreateDnVec(&x_descr, num_vertices, x_a, CUDA_R_32F);
    cusparseCreateDnVec(&y_descr, num_vertices, x_b, CUDA_R_32F);

    
    std::size_t buffer_size = 0;
    float h_alpha = 1.0f, h_beta = 0.0f;
    cusparseSpMV_bufferSize(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &h_alpha, mat_descr, x_descr, &h_beta, y_descr,
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);

    cache.ensure_spmv_buffer(buffer_size);

    cusparseSpMV_preprocess(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &h_alpha, mat_descr, x_descr, &h_beta, y_descr,
                            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buffer);

    
    float* x_old = x_a;
    float* x_new = x_b;

    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(x_old, initial_centralities,
                       num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        int block = 256, grid = (num_vertices + block - 1) / block;
        init_kernel<<<grid, block, 0, stream>>>(x_old, num_vertices, 1.0f / num_vertices);
    }

    bool converged = false;
    std::size_t iterations = 0;
    bool use_coop = (cache.coop_max_blocks > 0);

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        cusparseDnVecSetValues(x_descr, x_old);
        cusparseDnVecSetValues(y_descr, x_new);

        
        cusparseSpMV(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     d_alpha, mat_descr, x_descr, d_beta, y_descr,
                     CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buffer);

        iterations = iter + 1;
        bool check_now = (iterations <= 50 || iterations % 10 == 0 || iterations == max_iterations);

        if (use_coop) {
            int block_size = 256;
            int needed_blocks = (num_vertices + block_size - 1) / block_size;
            int grid_size = (needed_blocks < cache.coop_max_blocks) ? needed_blocks : cache.coop_max_blocks;
            void* args[] = {&x_new, &x_old, &num_vertices, &d_norm_sq, &d_diff, &check_now};
            cudaLaunchCooperativeKernel(
                (void*)fused_postprocess_v2_kernel,
                dim3(grid_size), dim3(block_size), args, 0, stream);
        } else {
            int block = 256, grid = (num_vertices + block - 1) / block;
            cudaMemsetAsync(d_norm_sq, 0, sizeof(float), stream);
            add_identity_l2_kernel<<<grid, block, 0, stream>>>(x_new, x_old, num_vertices, d_norm_sq);
            if (check_now) {
                cudaMemsetAsync(d_diff, 0, sizeof(float), stream);
                normalize_diff_kernel<<<grid, block, 0, stream>>>(x_new, x_old, num_vertices, d_norm_sq, d_diff);
            } else {
                normalize_kernel<<<grid, block, 0, stream>>>(x_new, num_vertices, d_norm_sq);
            }
        }

        if (check_now) {
            float h_diff;
            cudaMemcpyAsync(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (h_diff < threshold) {
                converged = true;
                break;
            }
        }

        
        float* tmp = x_old;
        x_old = x_new;
        x_new = tmp;
    }

    
    float* result = converged ? x_new : x_old;
    if (result != x_a) {
        cudaMemcpyAsync(x_a, result, num_vertices * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    cusparseDestroySpMat(mat_descr);
    cusparseDestroyDnVec(x_descr);
    cusparseDestroyDnVec(y_descr);

    return {iterations, converged};
}

}  
