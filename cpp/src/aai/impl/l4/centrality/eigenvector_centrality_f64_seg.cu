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
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* d_scalars = nullptr;
    float* weights_f32 = nullptr;
    float* buf0 = nullptr;
    float* buf1 = nullptr;
    int64_t weights_capacity = 0;
    int64_t buf0_capacity = 0;
    int64_t buf1_capacity = 0;

    Cache() {
        cudaMalloc(&d_scalars, 2 * sizeof(float));
    }

    void ensure(int64_t n, int64_t m) {
        if (weights_capacity < m) {
            if (weights_f32) cudaFree(weights_f32);
            cudaMalloc(&weights_f32, m * sizeof(float));
            weights_capacity = m;
        }
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

    ~Cache() override {
        if (d_scalars) cudaFree(d_scalars);
        if (weights_f32) cudaFree(weights_f32);
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
    }
};




__global__ void convert_f64_to_f32_kernel(const double* __restrict__ src, float* __restrict__ dst, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (idx < n) dst[idx] = (float)src[idx];
}

__global__ void convert_f32_to_f64_kernel(const float* __restrict__ src, double* __restrict__ dst, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (idx < n) dst[idx] = (double)src[idx];
}

__global__ void init_uniform_f32_kernel(float* __restrict__ x, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (idx < n) x[idx] = 1.0f / (float)n;
}





#define SPMV_BLOCK 256
#define SPMV_WARPS (SPMV_BLOCK / 32)

__global__ __launch_bounds__(SPMV_BLOCK)
void spmv_combined_f32_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float* __restrict__ d_norm_sq,
    int32_t seg_high_end,
    int32_t seg_mid_end,
    int32_t n,
    int32_t blocks_high,
    int32_t blocks_mid)
{
    typedef cub::BlockReduce<float, SPMV_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    if (blockIdx.x < blocks_high) {
        
        int32_t v = blockIdx.x;
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float sum = 0.0f;
        for (int32_t e = start + threadIdx.x; e < end; e += SPMV_BLOCK) {
            sum += weights[e] * x_old[indices[e]];
        }

        sum = BlockReduce(temp).Sum(sum);

        if (threadIdx.x == 0) {
            sum += x_old[v];
            x_new[v] = sum;
            atomicAdd(d_norm_sq, sum * sum);
        }
    }
    else if (blockIdx.x < blocks_high + blocks_mid) {
        
        int32_t block_in_seg = blockIdx.x - blocks_high;
        int warp_in_block = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        int32_t v = seg_high_end + block_in_seg * SPMV_WARPS + warp_in_block;

        float val_sq = 0.0f;
        if (v < seg_mid_end) {
            int32_t start = offsets[v];
            int32_t end = offsets[v + 1];

            float sum = 0.0f;
            for (int32_t e = start + lane; e < end; e += 32) {
                sum += weights[e] * x_old[indices[e]];
            }

            
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            }

            if (lane == 0) {
                sum += x_old[v];
                x_new[v] = sum;
                val_sq = sum * sum;
            }
        }

        float block_sum = BlockReduce(temp).Sum(val_sq);
        if (threadIdx.x == 0 && block_sum != 0.0f) {
            atomicAdd(d_norm_sq, block_sum);
        }
    }
    else {
        
        int32_t block_in_seg = blockIdx.x - blocks_high - blocks_mid;
        int32_t v = seg_mid_end + block_in_seg * SPMV_BLOCK + threadIdx.x;

        float val_sq = 0.0f;
        if (v < n) {
            int32_t start = offsets[v];
            int32_t end = offsets[v + 1];

            float sum = x_old[v];
            for (int32_t e = start; e < end; ++e) {
                sum += weights[e] * x_old[indices[e]];
            }
            x_new[v] = sum;
            val_sq = sum * sum;
        }

        float block_sum = BlockReduce(temp).Sum(val_sq);
        if (threadIdx.x == 0 && block_sum != 0.0f) {
            atomicAdd(d_norm_sq, block_sum);
        }
    }
}




#define NORM_BLOCK 256
#define NORM_VEC 4
#define NORM_ELEMS_PER_BLOCK (NORM_BLOCK * NORM_VEC)

__global__ __launch_bounds__(NORM_BLOCK)
void normalize_l1diff_f32_kernel(
    float* __restrict__ y,
    const float* __restrict__ x_old,
    const float* __restrict__ d_norm_sq,
    float* __restrict__ d_diff,
    int64_t n)
{
    typedef cub::BlockReduce<float, NORM_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float s_inv_norm;

    if (threadIdx.x == 0) {
        float norm_sq = *d_norm_sq;
        s_inv_norm = (norm_sq > 0.0f) ? rsqrtf(norm_sq) : 0.0f;
    }
    __syncthreads();

    float inv_norm = s_inv_norm;
    int64_t base = (int64_t)blockIdx.x * NORM_ELEMS_PER_BLOCK + (int64_t)threadIdx.x * NORM_VEC;
    float local_diff = 0.0f;

    if (base + 3 < n) {
        
        float4 yv = *reinterpret_cast<const float4*>(&y[base]);
        float4 xv = *reinterpret_cast<const float4*>(&x_old[base]);

        yv.x *= inv_norm; yv.y *= inv_norm; yv.z *= inv_norm; yv.w *= inv_norm;
        *reinterpret_cast<float4*>(&y[base]) = yv;

        local_diff = fabsf(yv.x - xv.x) + fabsf(yv.y - xv.y) +
                     fabsf(yv.z - xv.z) + fabsf(yv.w - xv.w);
    } else {
        
        for (int i = 0; i < NORM_VEC && base + i < n; i++) {
            float val = y[base + i] * inv_norm;
            y[base + i] = val;
            local_diff += fabsf(val - x_old[base + i]);
        }
    }

    float block_diff = BlockReduce(temp).Sum(local_diff);
    if (threadIdx.x == 0 && block_diff != 0.0f) {
        atomicAdd(d_diff, block_diff);
    }
}

}  

eigenvector_centrality_result_t eigenvector_centrality_seg(const graph32_t& graph,
                                const double* edge_weights,
                                double* centralities,
                                double epsilon,
                                std::size_t max_iterations,
                                const double* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    int64_t n = (int64_t)num_vertices;
    int64_t m = (int64_t)num_edges;

    cache.ensure(n, m);

    
    const auto& seg = graph.segment_offsets.value();

    int32_t blocks_high = seg[1] - seg[0];
    int32_t blocks_mid = (seg[2] - seg[1] + SPMV_WARPS - 1) / SPMV_WARPS;
    int32_t blocks_low = (seg[4] - seg[2] + SPMV_BLOCK - 1) / SPMV_BLOCK;
    int32_t total_blocks = blocks_high + blocks_mid + blocks_low;

    
    {
        int grid = (int)((m + 255) / 256);
        if (grid > 0)
            convert_f64_to_f32_kernel<<<grid, 256>>>(edge_weights, cache.weights_f32, m);
    }

    float* buf0 = cache.buf0;
    float* buf1 = cache.buf1;

    
    if (initial_centralities != nullptr) {
        int grid = (int)((n + 255) / 256);
        if (grid > 0)
            convert_f64_to_f32_kernel<<<grid, 256>>>(initial_centralities, buf0, n);
    } else {
        int grid = (int)((n + 255) / 256);
        if (grid > 0)
            init_uniform_f32_kernel<<<grid, 256>>>(buf0, n);
    }

    float* d_norm_sq = cache.d_scalars;
    float* d_diff = cache.d_scalars + 1;

    
    float threshold = (float)((double)n * epsilon);
    bool converged = false;
    std::size_t total_iters = 0;

    float* x_old = buf0;
    float* x_new = buf1;

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        
        cudaMemsetAsync(cache.d_scalars, 0, 2 * sizeof(float), 0);

        
        if (total_blocks > 0) {
            spmv_combined_f32_kernel<<<total_blocks, SPMV_BLOCK>>>(
                graph.offsets, graph.indices, cache.weights_f32,
                x_old, x_new, d_norm_sq,
                seg[1], seg[2], num_vertices,
                blocks_high, blocks_mid);
        }

        
        {
            int grid = (int)((n + NORM_ELEMS_PER_BLOCK - 1) / NORM_ELEMS_PER_BLOCK);
            if (grid > 0)
                normalize_l1diff_f32_kernel<<<grid, NORM_BLOCK>>>(
                    x_new, x_old, d_norm_sq, d_diff, n);
        }

        
        std::swap(x_old, x_new);
        total_iters = iter + 1;

        
        float h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_diff < threshold) {
            converged = true;
            break;
        }
    }

    
    {
        int grid = (int)((n + 255) / 256);
        if (grid > 0)
            convert_f32_to_f64_kernel<<<grid, 256>>>(x_old, centralities, n);
    }

    return {total_iters, converged};
}

}  
