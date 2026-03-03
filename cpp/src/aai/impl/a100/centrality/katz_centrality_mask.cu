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
#include <cstdint>
#include <cstddef>
#include <cmath>

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define MAX_BLOCKS (108 * 8)

static __host__ __device__ inline int grid_size(int n) {
    int b = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    return b > MAX_BLOCKS ? MAX_BLOCKS : b;
}



__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ degrees,
    int32_t num_vertices
) {
    for (int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x; v < num_vertices;
         v += gridDim.x * BLOCK_SIZE) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        int32_t count = 0;
        for (int32_t e = start; e < end; ++e) {
            count += (edge_mask[e >> 5] >> (e & 31)) & 1u;
        }
        degrees[v] = count;
    }
}

__global__ void compact_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t num_vertices
) {
    for (int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x; v < num_vertices;
         v += gridDim.x * BLOCK_SIZE) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        int32_t write_pos = new_offsets[v];
        for (int32_t e = start; e < end; ++e) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1u) {
                new_indices[write_pos++] = indices[e];
            }
        }
    }
}



__global__ void katz_analytical_iter1_kernel(
    const int32_t* __restrict__ new_offsets,
    float* __restrict__ x,
    float alpha, float beta, const float* __restrict__ betas,
    int32_t num_vertices
) {
    for (int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x; v < num_vertices;
         v += gridDim.x * BLOCK_SIZE) {
        int32_t deg = new_offsets[v + 1] - new_offsets[v];
        float bv = betas ? betas[v] : beta;
        x[v] = alpha * (float)deg + bv;
    }
}



__global__ void katz_spmv_tpr_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha, float beta, const float* __restrict__ betas,
    int32_t num_vertices,
    float* __restrict__ diff_sum
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float thread_diff = 0.0f;

    for (int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x; v < num_vertices;
         v += gridDim.x * BLOCK_SIZE) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        float sum = 0.0f;
        for (int32_t e = start; e < end; ++e) {
            sum += x_old[indices[e]];
        }
        float bv = betas ? betas[v] : beta;
        float val = alpha * sum + bv;
        x_new[v] = val;
        thread_diff += fabsf(val - x_old[v]);
    }

    float block_diff = BlockReduce(temp_storage).Sum(thread_diff);
    if (threadIdx.x == 0) {
        atomicAdd(diff_sum, block_diff);
    }
}



__global__ void katz_spmv_wpr_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha, float beta, const float* __restrict__ betas,
    int32_t num_vertices,
    float* __restrict__ diff_sum
) {
    const int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int warp_id = (blockIdx.x * WARPS_PER_BLOCK) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    int total_warps = gridDim.x * WARPS_PER_BLOCK;

    float thread_diff = 0.0f;

    for (int32_t v = warp_id; v < num_vertices; v += total_warps) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float sum = 0.0f;
        for (int32_t e = start + lane; e < end; e += 32) {
            sum += x_old[indices[e]];
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            float bv = betas ? betas[v] : beta;
            float val = alpha * sum + bv;
            x_new[v] = val;
            thread_diff += fabsf(val - x_old[v]);
        }
    }

    
    float block_diff = BlockReduce(temp_storage).Sum(thread_diff);
    if (threadIdx.x == 0) {
        atomicAdd(diff_sum, block_diff);
    }
}



__global__ void fill_float_kernel(float* __restrict__ x, int32_t n, float val) {
    for (int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n;
         i += gridDim.x * BLOCK_SIZE) {
        x[i] = val;
    }
}

__global__ void sum_abs_kernel(const float* __restrict__ x, int32_t n, float* __restrict__ result) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float sum = 0.0f;
    for (int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n;
         i += gridDim.x * BLOCK_SIZE) {
        sum += fabsf(x[i]);
    }
    float block_sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) atomicAdd(result, block_sum);
}

__global__ void l2_norm_sq_kernel(const float* __restrict__ x, int32_t n, float* __restrict__ result) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float sum = 0.0f;
    for (int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n;
         i += gridDim.x * BLOCK_SIZE) {
        float v = x[i]; sum += v * v;
    }
    float block_sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) atomicAdd(result, block_sum);
}

__global__ void scale_kernel(float* __restrict__ x, int32_t n, float s) {
    for (int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n;
         i += gridDim.x * BLOCK_SIZE) {
        x[i] *= s;
    }
}



struct Cache : Cacheable {
    float* d_scratch = nullptr;
    float* d_tmp = nullptr;
    int32_t* d_degrees = nullptr;
    int32_t* d_new_offsets = nullptr;
    int32_t* d_new_indices = nullptr;
    void* d_cub_temp = nullptr;

    int64_t scratch_capacity = 0;
    int64_t tmp_capacity = 0;
    int64_t degrees_capacity = 0;
    int64_t new_offsets_capacity = 0;
    int64_t new_indices_capacity = 0;
    size_t cub_temp_capacity = 0;

    void ensure_scratch(int64_t size) {
        if (scratch_capacity < size) {
            if (d_scratch) cudaFree(d_scratch);
            cudaMalloc(&d_scratch, size * sizeof(float));
            scratch_capacity = size;
        }
    }

    void ensure_tmp(int64_t size) {
        if (tmp_capacity < size) {
            if (d_tmp) cudaFree(d_tmp);
            cudaMalloc(&d_tmp, size * sizeof(float));
            tmp_capacity = size;
        }
    }

    void ensure_degrees(int64_t size) {
        if (degrees_capacity < size) {
            if (d_degrees) cudaFree(d_degrees);
            cudaMalloc(&d_degrees, size * sizeof(int32_t));
            degrees_capacity = size;
        }
    }

    void ensure_new_offsets(int64_t size) {
        if (new_offsets_capacity < size) {
            if (d_new_offsets) cudaFree(d_new_offsets);
            cudaMalloc(&d_new_offsets, size * sizeof(int32_t));
            new_offsets_capacity = size;
        }
    }

    void ensure_new_indices(int64_t size) {
        if (new_indices_capacity < size) {
            if (d_new_indices) cudaFree(d_new_indices);
            cudaMalloc(&d_new_indices, size * sizeof(int32_t));
            new_indices_capacity = size;
        }
    }

    void ensure_cub_temp(size_t size) {
        if (cub_temp_capacity < size) {
            if (d_cub_temp) cudaFree(d_cub_temp);
            cudaMalloc(&d_cub_temp, size);
            cub_temp_capacity = size;
        }
    }

    ~Cache() override {
        if (d_scratch) cudaFree(d_scratch);
        if (d_tmp) cudaFree(d_tmp);
        if (d_degrees) cudaFree(d_degrees);
        if (d_new_offsets) cudaFree(d_new_offsets);
        if (d_new_indices) cudaFree(d_new_indices);
        if (d_cub_temp) cudaFree(d_cub_temp);
    }
};

}  

katz_centrality_result_t katz_centrality_mask(const graph32_t& graph,
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
    int32_t N = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;
    const uint32_t* d_mask = graph.edge_mask;
    bool use_betas = (betas != nullptr);

    cudaStream_t s = 0;

    
    cache.ensure_scratch(2);
    cache.ensure_tmp(N);
    cache.ensure_degrees(N + 1);
    cache.ensure_new_offsets(N + 1);

    float* d_x0 = centralities;
    float* d_x1 = cache.d_tmp;
    float* d_diff = cache.d_scratch;
    int32_t* d_degrees = cache.d_degrees;
    int32_t* d_new_offsets = cache.d_new_offsets;

    
    count_active_edges_kernel<<<grid_size(N), BLOCK_SIZE, 0, s>>>(d_offsets, d_mask, d_degrees, N);
    cudaMemsetAsync(d_degrees + N, 0, sizeof(int32_t), s);

    
    size_t temp_size = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_size, d_degrees, d_new_offsets, N + 1, s);
    cache.ensure_cub_temp(temp_size);
    cub::DeviceScan::ExclusiveSum(cache.d_cub_temp, temp_size, d_degrees, d_new_offsets, N + 1, s);

    int32_t total_active;
    cudaMemcpyAsync(&total_active, d_new_offsets + N, sizeof(int32_t), cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    cache.ensure_new_indices(total_active > 0 ? total_active : 1);
    int32_t* d_new_indices = cache.d_new_indices;
    if (total_active > 0) {
        compact_edges_kernel<<<grid_size(N), BLOCK_SIZE, 0, s>>>(d_offsets, d_indices, d_mask, d_new_offsets, d_new_indices, N);
    }

    
    float* d_result = d_x0;
    size_t iters = 0;
    bool converged = false;
    float h_diff;

    if (!has_initial_guess && max_iterations > 0) {
        
        if (use_betas) {
            cudaMemcpyAsync(d_x0, betas, (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice, s);
            cudaMemsetAsync(d_diff, 0, sizeof(float), s);
            int b = grid_size(N); if (b > 1024) b = 1024;
            sum_abs_kernel<<<b, BLOCK_SIZE, 0, s>>>(d_x0, N, d_diff);
            cudaMemcpyAsync(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
        } else {
            fill_float_kernel<<<grid_size(N), BLOCK_SIZE, 0, s>>>(d_x0, N, beta);
            h_diff = (float)N * fabsf(beta);
        }
        iters = 1;
        if (h_diff < epsilon) {
            converged = true;
        }

        
        if (!converged && max_iterations > 1 && !use_betas) {
            katz_analytical_iter1_kernel<<<grid_size(N), BLOCK_SIZE, 0, s>>>(d_new_offsets, d_x0, alpha * beta, beta, betas, N);
            h_diff = alpha * beta * (float)total_active;
            iters = 2;
            if (h_diff < epsilon) {
                converged = true;
            }
        }
    } else if (has_initial_guess) {
        
    } else {
        
        cudaMemsetAsync(d_x0, 0, (size_t)N * sizeof(float), s);
    }

    
    if (!converged) {
        for (size_t it = iters; it < max_iterations; ++it) {
            float* x_old = d_result;
            float* x_new = (d_result == d_x0) ? d_x1 : d_x0;

            cudaMemsetAsync(d_diff, 0, sizeof(float), s);
            
            float avg_deg = (N > 0) ? (float)total_active / N : 0;
            if (avg_deg > 16.0f) {
                int warps_per_block = BLOCK_SIZE / 32;
                int blocks = (N + warps_per_block - 1) / warps_per_block;
                if (blocks > MAX_BLOCKS) blocks = MAX_BLOCKS;
                katz_spmv_wpr_kernel<<<blocks, BLOCK_SIZE, 0, s>>>(d_new_offsets, d_new_indices,
                                x_old, x_new, alpha, beta, betas,
                                N, d_diff);
            } else {
                katz_spmv_tpr_kernel<<<grid_size(N), BLOCK_SIZE, 0, s>>>(d_new_offsets, d_new_indices,
                                x_old, x_new, alpha, beta, betas,
                                N, d_diff);
            }

            cudaMemcpyAsync(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);

            d_result = x_new;
            iters++;

            if (h_diff < epsilon) {
                converged = true;
                break;
            }
        }
    }

    
    if (d_result != d_x0) {
        cudaMemcpyAsync(d_x0, d_result, (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice, s);
    }

    
    if (normalize) {
        float* d_nsq = cache.d_scratch + 1;
        cudaMemsetAsync(d_nsq, 0, sizeof(float), s);
        int b = grid_size(N); if (b > 1024) b = 1024;
        l2_norm_sq_kernel<<<b, BLOCK_SIZE, 0, s>>>(d_x0, N, d_nsq);
        float h_nsq;
        cudaMemcpyAsync(&h_nsq, d_nsq, sizeof(float), cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        if (h_nsq > 0.0f) {
            scale_kernel<<<grid_size(N), BLOCK_SIZE, 0, s>>>(d_x0, N, 1.0f / sqrtf(h_nsq));
        }
    }

    return katz_centrality_result_t{iters, converged};
}

}  
