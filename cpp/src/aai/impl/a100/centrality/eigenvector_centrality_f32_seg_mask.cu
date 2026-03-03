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
#include <cmath>
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* d_control = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;
    int64_t x_capacity = 0;
    int64_t y_capacity = 0;

    Cache() {
        cudaMalloc(&d_control, 4 * sizeof(float));
        cudaMemset(d_control, 0, 4 * sizeof(float));
    }

    void ensure(int32_t n) {
        if (x_capacity < n) {
            if (d_x) cudaFree(d_x);
            cudaMalloc(&d_x, n * sizeof(float));
            x_capacity = n;
        }
        if (y_capacity < n) {
            if (d_y) cudaFree(d_y);
            cudaMalloc(&d_y, n * sizeof(float));
            y_capacity = n;
        }
    }

    ~Cache() override {
        if (d_control) cudaFree(d_control);
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
    }
};




template<int BLOCK_SIZE>
__global__ void spmv_high_fused(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ mask,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ d_norm_sq,
    int start_v, int end_v)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage temp;

    int v = start_v + blockIdx.x;
    if (v >= end_v) return;

    int es = offsets[v], ee = offsets[v + 1];
    float sum = 0.0f;

    for (int e = es + threadIdx.x; e < ee; e += BLOCK_SIZE) {
        uint32_t mw = __ldg(&mask[e >> 5]);
        if ((mw >> (e & 31)) & 1) {
            sum += __ldg(&weights[e]) * x[__ldg(&indices[e])];
        }
    }

    sum = BR(temp).Sum(sum);
    if (threadIdx.x == 0) {
        float yv = sum + x[v];
        y[v] = yv;
        atomicAdd(d_norm_sq, yv * yv);
    }
}


__global__ void spmv_mid_fused(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ mask,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ d_norm_sq,
    int start_v, int end_v)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid >> 5;
    int lane = global_tid & 31;
    int local_warp = threadIdx.x >> 5;

    int v = start_v + warp_id;
    float yv = 0.0f;

    if (v < end_v) {
        int es = offsets[v], ee = offsets[v + 1];
        float sum = 0.0f;
        for (int e = es + lane; e < ee; e += 32) {
            uint32_t mw = __ldg(&mask[e >> 5]);
            if ((mw >> (e & 31)) & 1) {
                sum += __ldg(&weights[e]) * x[__ldg(&indices[e])];
            }
        }
        #pragma unroll
        for (int s = 16; s > 0; s >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, s);
        if (lane == 0) {
            yv = sum + x[v];
            y[v] = yv;
        }
    }

    
    constexpr int WARPS = 8;
    __shared__ float warp_sq[WARPS];
    if (lane == 0) {
        warp_sq[local_warp] = (v < end_v) ? yv * yv : 0.0f;
    }
    __syncthreads();

    if (threadIdx.x < WARPS) {
        float val = warp_sq[threadIdx.x];
        
        #pragma unroll
        for (int s = WARPS/2; s > 0; s >>= 1)
            val += __shfl_down_sync(0xFF, val, s);
        if (threadIdx.x == 0 && val != 0.0f) {
            atomicAdd(d_norm_sq, val);
        }
    }
}


template<int BLOCK_SIZE>
__global__ void spmv_low_fused(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ mask,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ d_norm_sq,
    int start_v, int end_v, int zero_start)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage temp;

    int v = start_v + blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float yv = 0.0f;

    if (v < end_v) {
        if (v < zero_start) {
            
            int es = offsets[v], ee = offsets[v + 1];
            float sum = 0.0f;
            for (int e = es; e < ee; e++) {
                uint32_t mw = __ldg(&mask[e >> 5]);
                if ((mw >> (e & 31)) & 1) {
                    sum += __ldg(&weights[e]) * x[__ldg(&indices[e])];
                }
            }
            yv = sum + x[v];
        } else {
            
            yv = x[v];
        }
        y[v] = yv;
    }

    float my_sq = (v < end_v) ? yv * yv : 0.0f;
    float block_sq = BR(temp).Sum(my_sq);
    if (threadIdx.x == 0 && block_sq != 0.0f) {
        atomicAdd(d_norm_sq, block_sq);
    }
}



__global__ void init_uniform_kernel(float* x, int n, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = val;
}


__global__ void inv_norm_and_clear_kernel(float* d_control) {
    float ns = d_control[0];
    d_control[1] = (ns > 0.0f) ? rsqrtf(ns) : 0.0f;
    d_control[0] = 0.0f; 
    d_control[2] = 0.0f; 
}


template<int BLOCK_SIZE>
__global__ void normalize_and_diff_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    float* __restrict__ d_control,
    int n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage temp;

    float inv_norm = d_control[1];
    float local_diff = 0.0f;

    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        float yn = y[i] * inv_norm;
        y[i] = yn;
        local_diff += fabsf(yn - x[i]);
    }

    float bsum = BR(temp).Sum(local_diff);
    if (threadIdx.x == 0 && bsum != 0.0f) {
        atomicAdd(&d_control[2], bsum);
    }
}


template<int BLOCK_SIZE>
__global__ void normalize_only_kernel(
    float* __restrict__ y,
    const float* __restrict__ d_control,
    int n)
{
    float inv_norm = d_control[1];
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        y[i] *= inv_norm;
    }
}



static inline int div_up(int a, int b) { return (a + b - 1) / b; }
static inline int min_val(int a, int b) { return a < b ? a : b; }

}  

eigenvector_centrality_result_t eigenvector_centrality_seg_mask(const graph32_t& graph,
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
    const uint32_t* d_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg3 = seg[3], seg4 = seg[4];

    const float* d_weights = edge_weights;

    cache.ensure(num_vertices);
    float* d_x = cache.d_x;
    float* d_y = cache.d_y;
    float* d_control = cache.d_control;

    
    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(d_x, initial_centralities,
                       num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        float init_val = 1.0f / num_vertices;
        init_uniform_kernel<<<div_up(num_vertices, 256), 256>>>(d_x, num_vertices, init_val);
    }

    
    cudaMemsetAsync(d_control, 0, 4 * sizeof(float));

    float threshold = static_cast<float>(num_vertices) * epsilon;
    bool converged = false;
    size_t iter = 0;

    
    int check_interval;
    if (num_vertices < 10000) check_interval = 10;
    else if (num_vertices < 100000) check_interval = 5;
    else check_interval = 3;

    for (iter = 0; iter < max_iterations; iter++) {
        
        
        int n_high = seg1 - seg0;
        if (n_high > 0) {
            spmv_high_fused<256><<<n_high, 256>>>(d_offsets, d_indices, d_weights, d_mask,
                                                   d_x, d_y, d_control, seg0, seg1);
        }

        
        int n_mid = seg2 - seg1;
        if (n_mid > 0) {
            int warps_per_block = 8;
            int blocks = div_up(n_mid, warps_per_block);
            spmv_mid_fused<<<blocks, warps_per_block * 32>>>(d_offsets, d_indices, d_weights, d_mask,
                                                              d_x, d_y, d_control, seg1, seg2);
        }

        
        int n_low_zero = seg4 - seg2;
        if (n_low_zero > 0) {
            spmv_low_fused<256><<<div_up(n_low_zero, 256), 256>>>(
                d_offsets, d_indices, d_weights, d_mask, d_x, d_y, d_control, seg2, seg4, seg3);
        }

        
        inv_norm_and_clear_kernel<<<1, 1>>>(d_control);

        
        bool do_check = ((iter + 1) % check_interval == 0) || (iter == max_iterations - 1);

        if (do_check) {
            
            int blocks = min_val(div_up(num_vertices, 256), 1024);
            normalize_and_diff_kernel<256><<<blocks, 256>>>(d_y, d_x, d_control, num_vertices);

            
            float h_diff;
            cudaMemcpy(&h_diff, &d_control[2], sizeof(float), cudaMemcpyDeviceToHost);

            std::swap(d_x, d_y);

            if (h_diff < threshold) {
                converged = true;
                iter++;
                break;
            }
        } else {
            
            int blocks = min_val(div_up(num_vertices, 256), 1024);
            normalize_only_kernel<256><<<blocks, 256>>>(d_y, d_control, num_vertices);
            std::swap(d_x, d_y);
        }
    }

    
    cudaMemcpyAsync(centralities, d_x, num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);

    return {iter, converged};
}

}  
