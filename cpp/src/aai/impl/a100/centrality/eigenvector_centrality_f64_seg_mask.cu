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
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <math.h>

namespace aai {

namespace {

namespace cg = cooperative_groups;


__device__ __forceinline__ double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}



__device__ __forceinline__ double block_reduce_sum(double val, double* warp_sums) {
    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int num_warps = blockDim.x >> 5;

    val = warp_reduce_sum(val);

    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();

    double result = 0.0;
    if (warp_id == 0) {
        double v = (lane < num_warps) ? warp_sums[lane] : 0.0;
        result = warp_reduce_sum(v);
    }
    __syncthreads();
    return result;
}


__global__ __launch_bounds__(256)
void eigenvector_centrality_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ x_buf0,
    double* __restrict__ x_buf1,
    const int32_t num_vertices,
    const int32_t seg0, const int32_t seg1, const int32_t seg2, const int32_t seg4,
    const double threshold,
    const size_t max_iterations,
    double* __restrict__ d_scalars,   
    int64_t* __restrict__ d_iterations,
    bool* __restrict__ d_converged)
{
    auto grid = cg::this_grid();
    __shared__ double warp_sums[8];

    const int32_t total_high = seg1 - seg0;
    const int32_t total_mid = seg2 - seg1;
    const int32_t total_low = seg4 - seg2;
    const int32_t warps_per_block = blockDim.x >> 5;
    const int32_t warp_id = threadIdx.x >> 5;
    const int32_t lane = threadIdx.x & 31;
    const int32_t tid = threadIdx.x;
    const int32_t bid = blockIdx.x;
    const int32_t threads_per_grid = gridDim.x * (int32_t)blockDim.x;
    const int32_t warps_per_grid = gridDim.x * warps_per_block;

    
    if (tid == 0 && bid == 0) {
        *d_converged = false;
        *d_iterations = 0;
        d_scalars[0] = 0.0; 
        d_scalars[1] = 0.0; 
    }
    grid.sync();

    for (size_t iter = 0; iter < max_iterations; iter++) {
        double* __restrict__ x_old = (iter & 1) ? x_buf1 : x_buf0;
        double* __restrict__ x_new = (iter & 1) ? x_buf0 : x_buf1;

        
        
        
        double thread_l2 = 0.0;

        
        for (int32_t b = bid; b < total_high; b += (int32_t)gridDim.x) {
            int32_t v = seg0 + b;
            int32_t row_start = offsets[v];
            int32_t row_end = offsets[v + 1];

            double sum = 0.0;
            for (int32_t k = row_start + tid; k < row_end; k += (int32_t)blockDim.x) {
                uint32_t mask_word = __ldg(&edge_mask[k >> 5]);
                if (mask_word & (1u << (k & 31))) {
                    sum += __ldg(&weights[k]) * __ldg(&x_old[__ldg(&indices[k])]);
                }
            }

            double blk_sum = block_reduce_sum(sum, warp_sums);
            if (tid == 0) {
                double xv = blk_sum + x_old[v];
                x_new[v] = xv;
                thread_l2 += xv * xv;
            }
        }

        
        for (int32_t w_global = bid * warps_per_block + warp_id;
             w_global < total_mid; w_global += warps_per_grid) {
            int32_t v = seg1 + w_global;
            int32_t row_start = offsets[v];
            int32_t row_end = offsets[v + 1];

            double sum = 0.0;
            for (int32_t k = row_start + lane; k < row_end; k += 32) {
                uint32_t mask_word = __ldg(&edge_mask[k >> 5]);
                if (mask_word & (1u << (k & 31))) {
                    sum += __ldg(&weights[k]) * __ldg(&x_old[__ldg(&indices[k])]);
                }
            }

            double warp_sum = warp_reduce_sum(sum);
            if (lane == 0) {
                double xv = warp_sum + x_old[v];
                x_new[v] = xv;
                thread_l2 += xv * xv;
            }
        }

        
        for (int32_t t_global = bid * (int32_t)blockDim.x + tid;
             t_global < total_low; t_global += threads_per_grid) {
            int32_t v = seg2 + t_global;
            int32_t row_start = offsets[v];
            int32_t row_end = offsets[v + 1];

            double sum = x_old[v];
            for (int32_t k = row_start; k < row_end; k++) {
                uint32_t mask_word = __ldg(&edge_mask[k >> 5]);
                if (mask_word & (1u << (k & 31))) {
                    sum += __ldg(&weights[k]) * __ldg(&x_old[__ldg(&indices[k])]);
                }
            }
            x_new[v] = sum;
            thread_l2 += sum * sum;
        }

        
        double blk_l2 = block_reduce_sum(thread_l2, warp_sums);
        if (tid == 0 && blk_l2 != 0.0) {
            atomicAdd(&d_scalars[0], blk_l2);
        }

        grid.sync(); 

        
        
        
        
        double inv_l2 = (d_scalars[0] > 0.0) ? 1.0 / sqrt(d_scalars[0]) : 1.0;

        double thread_diff = 0.0;
        for (int32_t i = bid * (int32_t)blockDim.x + tid;
             i < num_vertices; i += threads_per_grid) {
            double val = x_new[i] * inv_l2;
            x_new[i] = val;
            thread_diff += fabs(val - x_old[i]);
        }

        double blk_diff = block_reduce_sum(thread_diff, warp_sums);
        if (tid == 0 && blk_diff != 0.0) {
            atomicAdd(&d_scalars[1], blk_diff);
        }

        grid.sync(); 

        
        
        
        if (tid == 0 && bid == 0) {
            *d_iterations = (int64_t)(iter + 1u);
            if (d_scalars[1] < threshold) {
                *d_converged = true;
            }
            
            d_scalars[0] = 0.0;
            d_scalars[1] = 0.0;
        }

        grid.sync(); 

        if (*d_converged) return;
    }
}

__global__ void init_uniform_kernel(double* __restrict__ x, int32_t n) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0 / (double)n;
}

struct Cache : Cacheable {
    double* x_buf0 = nullptr;
    double* x_buf1 = nullptr;
    double* d_scalars = nullptr;
    int64_t* d_iterations = nullptr;
    bool* d_converged = nullptr;

    int64_t x_buf0_capacity = 0;
    int64_t x_buf1_capacity = 0;

    int max_blocks = 0;
    bool max_blocks_queried = false;

    void ensure(int32_t num_vertices) {
        if (x_buf0_capacity < num_vertices) {
            if (x_buf0) cudaFree(x_buf0);
            cudaMalloc(&x_buf0, (size_t)num_vertices * sizeof(double));
            x_buf0_capacity = num_vertices;
        }
        if (x_buf1_capacity < num_vertices) {
            if (x_buf1) cudaFree(x_buf1);
            cudaMalloc(&x_buf1, (size_t)num_vertices * sizeof(double));
            x_buf1_capacity = num_vertices;
        }
        if (!d_scalars) {
            cudaMalloc(&d_scalars, 3 * sizeof(double));
        }
        if (!d_iterations) {
            cudaMalloc(&d_iterations, sizeof(int64_t));
        }
        if (!d_converged) {
            cudaMalloc(&d_converged, sizeof(bool));
        }
    }

    int get_max_blocks() {
        if (!max_blocks_queried) {
            int num_blocks_per_sm;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &num_blocks_per_sm, eigenvector_centrality_kernel, 256, 0);
            int num_sms;
            cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
            max_blocks = num_blocks_per_sm * num_sms;
            max_blocks_queried = true;
        }
        return max_blocks;
    }

    ~Cache() override {
        if (x_buf0) cudaFree(x_buf0);
        if (x_buf1) cudaFree(x_buf1);
        if (d_scalars) cudaFree(d_scalars);
        if (d_iterations) cudaFree(d_iterations);
        if (d_converged) cudaFree(d_converged);
    }
};

}  

eigenvector_centrality_result_t eigenvector_centrality_seg_mask(const graph32_t& graph,
                                      const double* edge_weights,
                                      double* centralities,
                                      double epsilon,
                                      std::size_t max_iterations,
                                      const double* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;

    cache.ensure(num_vertices);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2];
    int32_t seg4 = seg[4];

    cudaStream_t stream = 0;

    double* d_x0 = cache.x_buf0;
    double* d_x1 = cache.x_buf1;
    double* d_scalars = cache.d_scalars;
    int64_t* d_iterations = cache.d_iterations;
    bool* d_converged = cache.d_converged;

    
    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(d_x0, initial_centralities,
                       num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else {
        int threads = 256;
        int blocks = (num_vertices + threads - 1) / threads;
        init_uniform_kernel<<<blocks, threads, 0, stream>>>(d_x0, num_vertices);
    }

    double threshold = (double)num_vertices * epsilon;

    
    int num_blocks = cache.get_max_blocks();
    int blocks_needed = (num_vertices + 255) / 256;
    if (blocks_needed < num_blocks) num_blocks = blocks_needed;
    if (num_blocks < 1) num_blocks = 1;

    
    bool use_l2_persist = (num_vertices * (int64_t)sizeof(double) <= 20 * 1024 * 1024);
    if (use_l2_persist) {
        cudaStreamAttrValue attr;
        memset(&attr, 0, sizeof(attr));
        attr.accessPolicyWindow.base_ptr = (void*)d_x0;
        attr.accessPolicyWindow.num_bytes = num_vertices * sizeof(double);
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    }

    
    cudaFuncSetAttribute(eigenvector_centrality_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout, 0);

    void* args[] = {
        (void*)&d_offsets, (void*)&d_indices, (void*)&edge_weights, (void*)&d_edge_mask,
        (void*)&d_x0, (void*)&d_x1,
        (void*)&num_vertices,
        (void*)&seg0, (void*)&seg1, (void*)&seg2, (void*)&seg4,
        (void*)&threshold, (void*)&max_iterations,
        (void*)&d_scalars,
        (void*)&d_iterations, (void*)&d_converged
    };
    cudaLaunchCooperativeKernel(
        (void*)eigenvector_centrality_kernel,
        dim3(num_blocks), dim3(256),
        args, 0, stream);

    cudaStreamSynchronize(stream);

    
    if (use_l2_persist) {
        cudaStreamAttrValue attr;
        memset(&attr, 0, sizeof(attr));
        attr.accessPolicyWindow.base_ptr = nullptr;
        attr.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    }

    
    int64_t h_iterations;
    bool h_converged;
    cudaMemcpy(&h_iterations, d_iterations, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_converged, d_converged, sizeof(bool), cudaMemcpyDeviceToHost);

    
    double* result_buf = (h_iterations % 2 == 1) ? d_x1 : d_x0;
    cudaMemcpyAsync(centralities, result_buf,
                   num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    eigenvector_centrality_result_t result;
    result.iterations = (std::size_t)h_iterations;
    result.converged = h_converged;
    return result;
}

}  
