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
#include <cub/block/block_reduce.cuh>
#include <cstdint>
#include <cmath>

namespace aai {

namespace {

struct Cache : Cacheable {
    double* d_diff = nullptr;
    double* h_diff = nullptr;

    float* aw = nullptr;
    int64_t aw_capacity = 0;

    float* buf0 = nullptr;
    int64_t buf0_capacity = 0;

    float* buf1 = nullptr;
    int64_t buf1_capacity = 0;

    float* betas_f = nullptr;
    int64_t betas_f_capacity = 0;

    Cache() {
        cudaMalloc(&d_diff, sizeof(double));
        cudaMallocHost(&h_diff, sizeof(double));
    }

    ~Cache() override {
        if (d_diff) cudaFree(d_diff);
        if (h_diff) cudaFreeHost(h_diff);
        if (aw) cudaFree(aw);
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (betas_f) cudaFree(betas_f);
    }

    void ensure_aw(int64_t n) {
        if (aw_capacity < n) {
            if (aw) cudaFree(aw);
            cudaMalloc(&aw, n * sizeof(float));
            aw_capacity = n;
        }
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

    void ensure_betas_f(int64_t n) {
        if (betas_f_capacity < n) {
            if (betas_f) cudaFree(betas_f);
            cudaMalloc(&betas_f, n * sizeof(float));
            betas_f_capacity = n;
        }
    }
};


__global__ void precompute_aw_kernel(const double* __restrict__ weights, float* __restrict__ aw,
                                      double alpha, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) aw[i] = (float)(alpha * weights[i]);
}


__global__ void fill_f_kernel(float* __restrict__ out, float val, int32_t n) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) out[i] = val;
}


__global__ void d2f_kernel(const double* __restrict__ in, float* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (float)in[i];
}


__global__ void f2d_kernel(const float* __restrict__ in, double* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (double)in[i];
}


__global__ void spmv_high(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ aw,        
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float beta_scalar_f,
    const float* __restrict__ betas_f,
    int use_betas,
    int32_t seg_start, int32_t seg_end,
    double* __restrict__ global_diff)
{
    typedef cub::BlockReduce<double, 256> BlockReduceD;
    __shared__ typename BlockReduceD::TempStorage temp_d;
    typedef cub::BlockReduce<float, 256> BlockReduceF;
    __shared__ typename BlockReduceF::TempStorage temp_f;

    int v = seg_start + blockIdx.x;
    if (v >= seg_end) return;

    int start = offsets[v];
    int end = offsets[v + 1];

    float sum = 0.0f;
    for (int e = start + (int)threadIdx.x; e < end; e += 256) {
        uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
        if (mask_word & (1u << (e & 31))) {
            sum += __ldg(&aw[e]) * __ldg(&x_old[__ldg(&indices[e])]);
        }
    }

    float block_sum = BlockReduceF(temp_f).Sum(sum);

    if (threadIdx.x == 0) {
        float beta_v = use_betas ? betas_f[v] : beta_scalar_f;
        float new_val = block_sum + beta_v;
        x_new[v] = new_val;
        atomicAdd(global_diff, (double)fabsf(new_val - x_old[v]));
    }
}


__global__ void spmv_mid(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ aw,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float beta_scalar_f,
    const float* __restrict__ betas_f,
    int use_betas,
    int32_t seg_start, int32_t seg_end,
    double* __restrict__ global_diff)
{
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int v = seg_start + warp_global;
    if (v >= seg_end) return;

    int start = offsets[v];
    int end = offsets[v + 1];

    float sum = 0.0f;
    for (int e = start + lane; e < end; e += 32) {
        uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
        if (mask_word & (1u << (e & 31))) {
            sum += __ldg(&aw[e]) * __ldg(&x_old[__ldg(&indices[e])]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) {
        float beta_v = use_betas ? betas_f[v] : beta_scalar_f;
        float new_val = sum + beta_v;
        x_new[v] = new_val;
        atomicAdd(global_diff, (double)fabsf(new_val - x_old[v]));
    }
}


__global__ void spmv_low(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ aw,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float beta_scalar_f,
    const float* __restrict__ betas_f,
    int use_betas,
    int32_t seg_start, int32_t seg_end,
    double* __restrict__ global_diff)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = seg_start + blockIdx.x * 256 + threadIdx.x;
    double local_diff = 0.0;

    if (v < seg_end) {
        int start = offsets[v];
        int end = offsets[v + 1];

        float sum = 0.0f;
        for (int e = start; e < end; e++) {
            uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
            if (mask_word & (1u << (e & 31))) {
                sum += __ldg(&aw[e]) * __ldg(&x_old[__ldg(&indices[e])]);
            }
        }

        float beta_v = use_betas ? betas_f[v] : beta_scalar_f;
        float new_val = sum + beta_v;
        x_new[v] = new_val;
        local_diff = (double)fabsf(new_val - x_old[v]);
    }

    double block_diff = BlockReduce(temp_storage).Sum(local_diff);
    if (threadIdx.x == 0 && block_diff > 0.0) {
        atomicAdd(global_diff, block_diff);
    }
}


__global__ void l2_sq_kernel(const double* __restrict__ x, double* __restrict__ result, int32_t n) {
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int i = blockIdx.x * 256 + threadIdx.x;
    double val = 0.0;
    if (i < n) val = x[i] * x[i];
    double block_sum = BlockReduce(temp_storage).Sum(val);
    if (threadIdx.x == 0 && block_sum > 0.0) atomicAdd(result, block_sum);
}


__global__ void normalize_kernel(double* x, int32_t n, double inv_norm) {
    int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) x[i] *= inv_norm;
}

}  

katz_centrality_result_t katz_centrality_seg_mask(const graph32_t& graph,
                               const double* edge_weights,
                               double* centralities,
                               double alpha,
                               double beta,
                               const double* betas,
                               double epsilon,
                               std::size_t max_iterations,
                               bool has_initial_guess,
                               bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg_arr[5] = {seg[0], seg[1], seg[2], seg[3], seg[4]};

    cudaStream_t stream = 0;

    
    cache.ensure_aw(num_edges);
    {
        int grid = (int)((num_edges + 255) / 256);
        if (grid > 0) precompute_aw_kernel<<<grid, 256, 0, stream>>>(edge_weights, cache.aw, alpha, num_edges);
    }
    const float* d_aw = cache.aw;

    
    cache.ensure_bufs(num_vertices);
    float* buffers[2] = {cache.buf0, cache.buf1};

    
    bool use_betas = (betas != nullptr);
    const float* d_betas_f = nullptr;
    if (use_betas) {
        cache.ensure_betas_f(num_vertices);
        int grid = (int)((num_vertices + 255) / 256);
        if (grid > 0) d2f_kernel<<<grid, 256, 0, stream>>>(betas, cache.betas_f, num_vertices);
        d_betas_f = cache.betas_f;
    }
    float beta_f = (float)beta;

    int cur = 0;
    std::size_t iterations = 0;
    bool converged = false;
    int ub = use_betas ? 1 : 0;

    if (has_initial_guess) {
        
        int grid = (int)((num_vertices + 255) / 256);
        if (grid > 0) d2f_kernel<<<grid, 256, 0, stream>>>(centralities, buffers[0], num_vertices);
    } else {
        
        
        
        if (max_iterations > 0) {
            if (use_betas) {
                
                cudaMemcpyAsync(buffers[0], d_betas_f, num_vertices * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream);
            } else {
                int grid = (num_vertices + 255) / 256;
                if (grid > 0) fill_f_kernel<<<grid, 256, 0, stream>>>(buffers[0], beta_f, num_vertices);
            }
            iterations = 1;

            
            
            double first_diff = (double)num_vertices * fabs((double)beta_f);
            if (use_betas) {
                
                
                first_diff = epsilon + 1.0;  
            }
            if (first_diff < epsilon) {
                converged = true;
            }
        }
    }

    if (!converged) {
        for (std::size_t iter = iterations; iter < max_iterations; iter++) {
            float* x_old = buffers[cur];
            float* x_new = buffers[1 - cur];

            cudaMemsetAsync(cache.d_diff, 0, sizeof(double), stream);

            
            {
                int n = seg_arr[1] - seg_arr[0];
                if (n > 0) {
                    spmv_high<<<n, 256, 0, stream>>>(d_offsets, d_indices, d_aw, d_mask,
                        x_old, x_new, beta_f, d_betas_f, ub,
                        seg_arr[0], seg_arr[1], cache.d_diff);
                }
            }

            
            {
                int n = seg_arr[2] - seg_arr[1];
                if (n > 0) {
                    int warps_per_block = 8;
                    int grid = (n + warps_per_block - 1) / warps_per_block;
                    spmv_mid<<<grid, 256, 0, stream>>>(d_offsets, d_indices, d_aw, d_mask,
                        x_old, x_new, beta_f, d_betas_f, ub,
                        seg_arr[1], seg_arr[2], cache.d_diff);
                }
            }

            
            {
                int n = seg_arr[4] - seg_arr[2];
                if (n > 0) {
                    int grid = (n + 255) / 256;
                    spmv_low<<<grid, 256, 0, stream>>>(d_offsets, d_indices, d_aw, d_mask,
                        x_old, x_new, beta_f, d_betas_f, ub,
                        seg_arr[2], seg_arr[4], cache.d_diff);
                }
            }

            cudaMemcpyAsync(cache.h_diff, cache.d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            cur = 1 - cur;
            iterations++;

            if (*cache.h_diff < epsilon) {
                converged = true;
                break;
            }
        }
    }

    
    float* result_f = buffers[cur];
    {
        int grid = (int)((num_vertices + 255) / 256);
        if (grid > 0) f2d_kernel<<<grid, 256, 0, stream>>>(result_f, centralities, num_vertices);
    }

    if (normalize) {
        cudaMemsetAsync(cache.d_diff, 0, sizeof(double), stream);
        {
            int grid = (num_vertices + 255) / 256;
            l2_sq_kernel<<<grid, 256, 0, stream>>>(centralities, cache.d_diff, num_vertices);
        }
        cudaMemcpyAsync(cache.h_diff, cache.d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        double l2_norm = sqrt(*cache.h_diff);
        if (l2_norm > 0.0) {
            int grid = (num_vertices + 255) / 256;
            normalize_kernel<<<grid, 256, 0, stream>>>(centralities, num_vertices, 1.0 / l2_norm);
        }
    }

    return {iterations, converged};
}

}  
