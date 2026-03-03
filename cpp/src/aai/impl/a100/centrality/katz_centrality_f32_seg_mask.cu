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
#include <cstdint>
#include <cstddef>
#include <utility>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* x0 = nullptr;
    float* x1 = nullptr;
    float* d_scalar = nullptr;
    int32_t x0_capacity = 0;
    int32_t x1_capacity = 0;
    bool scalar_allocated = false;

    void ensure(int32_t n) {
        if (x0_capacity < n) {
            if (x0) cudaFree(x0);
            cudaMalloc(&x0, (size_t)n * sizeof(float));
            x0_capacity = n;
        }
        if (x1_capacity < n) {
            if (x1) cudaFree(x1);
            cudaMalloc(&x1, (size_t)n * sizeof(float));
            x1_capacity = n;
        }
        if (!scalar_allocated) {
            cudaMalloc(&d_scalar, sizeof(float));
            scalar_allocated = true;
        }
    }

    ~Cache() override {
        if (x0) cudaFree(x0);
        if (x1) cudaFree(x1);
        if (d_scalar) cudaFree(d_scalar);
    }
};


__global__ void katz_high_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha,
    float beta,
    const float* __restrict__ betas,
    int32_t seg_start,
    int32_t seg_end)
{
    extern __shared__ float shmem[];

    int32_t v = seg_start + blockIdx.x;
    if (v >= seg_end) return;

    int32_t start = offsets[v];
    int32_t end_edge = offsets[v + 1];
    int32_t tid = threadIdx.x;
    int32_t bs = blockDim.x;

    float sum = 0.0f;
    for (int32_t e = start + tid; e < end_edge; e += bs) {
        uint32_t mask_word = edge_mask[e >> 5];
        if ((mask_word >> (e & 31)) & 1) {
            sum += edge_weights[e] * x_old[indices[e]];
        }
    }

    
    int lane = tid & 31;
    int warp_in_block = tid >> 5;
    int warps = bs >> 5;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) shmem[warp_in_block] = sum;
    __syncthreads();

    if (warp_in_block == 0) {
        sum = (lane < warps) ? shmem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) {
            x_new[v] = alpha * sum + (betas != nullptr ? betas[v] : beta);
        }
    }
}


__global__ void katz_mid_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha,
    float beta,
    const float* __restrict__ betas,
    int32_t seg_start,
    int32_t seg_end)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t warp_id = tid >> 5;
    int32_t lane = tid & 31;

    int32_t v = seg_start + warp_id;
    if (v >= seg_end) return;

    int32_t start = offsets[v];
    int32_t end_edge = offsets[v + 1];

    float sum = 0.0f;
    for (int32_t e = start + lane; e < end_edge; e += 32) {
        uint32_t mask_word = edge_mask[e >> 5];
        if ((mask_word >> (e & 31)) & 1) {
            sum += edge_weights[e] * x_old[indices[e]];
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) {
        x_new[v] = alpha * sum + (betas != nullptr ? betas[v] : beta);
    }
}


__global__ void katz_low_zero(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha,
    float beta,
    const float* __restrict__ betas,
    int32_t seg_start,
    int32_t seg_end)
{
    int32_t v = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= seg_end) return;

    int32_t start = offsets[v];
    int32_t end_edge = offsets[v + 1];

    float sum = 0.0f;
    for (int32_t e = start; e < end_edge; ++e) {
        uint32_t mask_word = edge_mask[e >> 5];
        if ((mask_word >> (e & 31)) & 1) {
            sum += edge_weights[e] * x_old[indices[e]];
        }
    }

    x_new[v] = alpha * sum + (betas != nullptr ? betas[v] : beta);
}


__global__ void l1_diff_reduce(
    const float* __restrict__ a,
    const float* __restrict__ b,
    int32_t n,
    float* __restrict__ result)
{
    extern __shared__ float shmem[];
    int32_t tid = threadIdx.x;
    int32_t idx = blockIdx.x * blockDim.x + tid;
    int32_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int32_t i = idx; i < n; i += stride)
        sum += fabsf(a[i] - b[i]);

    int lane = tid & 31;
    int warp = tid >> 5;
    int warps = blockDim.x >> 5;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) shmem[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < warps) ? shmem[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, off);
        if (lane == 0) atomicAdd(result, sum);
    }
}


__global__ void l2_norm_sq(
    const float* __restrict__ x,
    int32_t n,
    float* __restrict__ result)
{
    extern __shared__ float shmem[];
    int32_t tid = threadIdx.x;
    int32_t idx = blockIdx.x * blockDim.x + tid;
    int32_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int32_t i = idx; i < n; i += stride) {
        float v = x[i];
        sum += v * v;
    }

    int lane = tid & 31;
    int warp = tid >> 5;
    int warps = blockDim.x >> 5;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) shmem[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        sum = (lane < warps) ? shmem[lane] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, off);
        if (lane == 0) atomicAdd(result, sum);
    }
}


__global__ void normalize_vec(float* __restrict__ x, int32_t n, const float* __restrict__ norm_sq) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= rsqrtf(*norm_sq);
    }
}

}  

katz_centrality_result_t katz_centrality_seg_mask(const graph32_t& graph,
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

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    const auto& seg_opt = graph.segment_offsets.value();
    int32_t seg[5] = {seg_opt[0], seg_opt[1], seg_opt[2], seg_opt[3], seg_opt[4]};

    cache.ensure(num_vertices);
    float* x0 = cache.x0;
    float* x1 = cache.x1;
    float* d_scalar = cache.d_scalar;

    
    if (has_initial_guess) {
        cudaMemcpyAsync(x0, centralities,
                   num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        cudaMemsetAsync(x0, 0, num_vertices * sizeof(float));
    }

    float* x_old = x0;
    float* x_new = x1;

    size_t iterations = 0;
    bool converged = false;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(d_scalar, 0, sizeof(float));

        
        
        {
            int32_t count = seg[1] - seg[0];
            if (count > 0) {
                int bs = 256;
                katz_high_degree<<<count, bs, (bs / 32) * sizeof(float)>>>(
                    d_offsets, d_indices, edge_weights, d_edge_mask,
                    x_old, x_new, alpha, beta, betas, seg[0], seg[1]);
            }
        }
        
        {
            int32_t count = seg[2] - seg[1];
            if (count > 0) {
                int bs = 256;
                int wpb = bs / 32;
                int grid = (count + wpb - 1) / wpb;
                katz_mid_degree<<<grid, bs>>>(
                    d_offsets, d_indices, edge_weights, d_edge_mask,
                    x_old, x_new, alpha, beta, betas, seg[1], seg[2]);
            }
        }
        
        {
            int32_t count = seg[4] - seg[2];
            if (count > 0) {
                int bs = 256;
                int grid = (count + bs - 1) / bs;
                katz_low_zero<<<grid, bs>>>(
                    d_offsets, d_indices, edge_weights, d_edge_mask,
                    x_old, x_new, alpha, beta, betas, seg[2], seg[4]);
            }
        }

        
        {
            int bs = 256;
            int grid = (num_vertices + bs - 1) / bs;
            if (grid > 1024) grid = 1024;
            l1_diff_reduce<<<grid, bs, (bs / 32) * sizeof(float)>>>(x_new, x_old, num_vertices, d_scalar);
        }

        iterations++;

        
        float h_diff;
        cudaMemcpy(&h_diff, d_scalar, sizeof(float), cudaMemcpyDeviceToHost);

        if (h_diff < epsilon) {
            converged = true;
            break;
        }

        
        std::swap(x_old, x_new);
    }

    
    float* result_ptr = converged ? x_new : x_old;

    
    if (normalize) {
        cudaMemsetAsync(d_scalar, 0, sizeof(float));
        {
            int bs = 256;
            int grid = (num_vertices + bs - 1) / bs;
            if (grid > 1024) grid = 1024;
            l2_norm_sq<<<grid, bs, (bs / 32) * sizeof(float)>>>(result_ptr, num_vertices, d_scalar);
        }
        {
            int bs = 256;
            int grid = (num_vertices + bs - 1) / bs;
            normalize_vec<<<grid, bs>>>(result_ptr, num_vertices, d_scalar);
        }
    }

    
    cudaMemcpyAsync(centralities, result_ptr,
                    num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);

    return {iterations, converged};
}

}  
