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
#include <cmath>

namespace aai {

namespace {





struct Cache : Cacheable {
    float* buf_a = nullptr;
    float* buf_b = nullptr;
    float* diff_buf = nullptr;
    int64_t buf_capacity = 0;
    int64_t diff_capacity = 0;

    void ensure(int32_t num_vertices) {
        int64_t n = static_cast<int64_t>(num_vertices);
        if (buf_capacity < n) {
            if (buf_a) cudaFree(buf_a);
            if (buf_b) cudaFree(buf_b);
            cudaMalloc(&buf_a, n * sizeof(float));
            cudaMalloc(&buf_b, n * sizeof(float));
            buf_capacity = n;
        }
        if (diff_capacity < 1) {
            if (diff_buf) cudaFree(diff_buf);
            cudaMalloc(&diff_buf, sizeof(float));
            diff_capacity = 1;
        }
    }

    ~Cache() override {
        if (buf_a) cudaFree(buf_a);
        if (buf_b) cudaFree(buf_b);
        if (diff_buf) cudaFree(diff_buf);
    }
};





__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, int num_warps) {
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        val = (lane < num_warps) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}




__global__ void __launch_bounds__(256)
katz_high_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha,
    float beta,
    const float* __restrict__ betas,
    int32_t start_v,
    int32_t end_v,
    float* __restrict__ global_diff)
{
    int v = start_v + blockIdx.x;
    if (v >= end_v) return;

    int edge_start = offsets[v];
    int edge_end   = offsets[v + 1];

    float sum = 0.0f;
    for (int j = edge_start + threadIdx.x; j < edge_end; j += blockDim.x) {
        sum += x_old[indices[j]];
    }

    sum = block_reduce_sum(sum, blockDim.x >> 5);

    if (threadIdx.x == 0) {
        float beta_v = betas ? betas[v] : beta;
        float new_val = alpha * sum + beta_v;
        x_new[v] = new_val;
        atomicAdd(global_diff, fabsf(new_val - x_old[v]));
    }
}




__global__ void __launch_bounds__(256)
katz_mid_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha,
    float beta,
    const float* __restrict__ betas,
    int32_t start_v,
    int32_t end_v,
    float* __restrict__ global_diff)
{
    constexpr int WARPS_PER_BLOCK = 8;
    int warp_id = threadIdx.x >> 5;
    int lane    = threadIdx.x & 31;
    int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    int v = start_v + global_warp;

    float local_diff = 0.0f;

    if (v < end_v) {
        int edge_start = offsets[v];
        int edge_end   = offsets[v + 1];

        float sum = 0.0f;
        for (int j = edge_start + lane; j < edge_end; j += 32) {
            sum += x_old[indices[j]];
        }
        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            float beta_v = betas ? betas[v] : beta;
            float new_val = alpha * sum + beta_v;
            x_new[v] = new_val;
            local_diff = fabsf(new_val - x_old[v]);
        }
    }

    __shared__ float warp_diffs[WARPS_PER_BLOCK];
    if (lane == 0) warp_diffs[warp_id] = local_diff;
    __syncthreads();

    if (threadIdx.x == 0) {
        float bd = 0.0f;
        #pragma unroll
        for (int i = 0; i < WARPS_PER_BLOCK; i++) bd += warp_diffs[i];
        if (bd > 0.0f) atomicAdd(global_diff, bd);
    }
}




__global__ void __launch_bounds__(256)
katz_low_zero(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha,
    float beta,
    const float* __restrict__ betas,
    int32_t start_v,
    int32_t end_v,
    float* __restrict__ global_diff)
{
    int num_v = end_v - start_v;
    float local_diff = 0.0f;

    for (int tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < num_v;
         tid += gridDim.x * blockDim.x)
    {
        int v = start_v + tid;
        int edge_start = offsets[v];
        int edge_end   = offsets[v + 1];

        float sum = 0.0f;
        for (int j = edge_start; j < edge_end; j++) {
            sum += x_old[indices[j]];
        }

        float beta_v = betas ? betas[v] : beta;
        float new_val = alpha * sum + beta_v;
        x_new[v] = new_val;
        local_diff += fabsf(new_val - x_old[v]);
    }

    local_diff = block_reduce_sum(local_diff, blockDim.x >> 5);
    if (threadIdx.x == 0 && local_diff > 0.0f) {
        atomicAdd(global_diff, local_diff);
    }
}




__global__ void __launch_bounds__(256)
l2_norm_sq_kernel(const float* __restrict__ x, int32_t n, float* __restrict__ out) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float val = x[i];
        sum += val * val;
    }
    sum = block_reduce_sum(sum, blockDim.x >> 5);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

__global__ void inv_sqrt_kernel(float* val) {
    *val = rsqrtf(*val);
}

__global__ void __launch_bounds__(256)
scale_kernel(float* __restrict__ x, int32_t n, const float* __restrict__ scale_ptr) {
    float s = *scale_ptr;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        x[i] *= s;
    }
}





void launch_katz_high_degree(
    const int32_t* offsets, const int32_t* indices,
    const float* x_old, float* x_new,
    float alpha, float beta, const float* betas,
    int32_t start_v, int32_t end_v,
    float* global_diff, cudaStream_t stream)
{
    int n = end_v - start_v;
    if (n <= 0) return;
    katz_high_degree<<<n, 256, 0, stream>>>(
        offsets, indices, x_old, x_new, alpha, beta, betas,
        start_v, end_v, global_diff);
}

void launch_katz_mid_degree(
    const int32_t* offsets, const int32_t* indices,
    const float* x_old, float* x_new,
    float alpha, float beta, const float* betas,
    int32_t start_v, int32_t end_v,
    float* global_diff, cudaStream_t stream)
{
    int n = end_v - start_v;
    if (n <= 0) return;
    int blocks = (n + 7) / 8;
    katz_mid_degree<<<blocks, 256, 0, stream>>>(
        offsets, indices, x_old, x_new, alpha, beta, betas,
        start_v, end_v, global_diff);
}

void launch_katz_low_zero(
    const int32_t* offsets, const int32_t* indices,
    const float* x_old, float* x_new,
    float alpha, float beta, const float* betas,
    int32_t start_v, int32_t end_v,
    float* global_diff, cudaStream_t stream)
{
    int n = end_v - start_v;
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;
    katz_low_zero<<<blocks, threads, 0, stream>>>(
        offsets, indices, x_old, x_new, alpha, beta, betas,
        start_v, end_v, global_diff);
}

void launch_normalize(float* x, int32_t n, float* scratch, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    l2_norm_sq_kernel<<<blocks, threads, 0, stream>>>(x, n, scratch);
    inv_sqrt_kernel<<<1, 1, 0, stream>>>(scratch);
    blocks = (n + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;
    scale_kernel<<<blocks, threads, 0, stream>>>(x, n, scratch);
}

}  

katz_centrality_result_t katz_centrality_seg(const graph32_t& graph,
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

    
    const auto& seg = graph.segment_offsets.value();
    int32_t s0 = seg[0], s1 = seg[1], s2 = seg[2], s3 = seg[3], s4 = seg[4];

    
    cache.ensure(num_vertices);

    float* bufs[2] = {cache.buf_a, cache.buf_b};
    float* d_diff = cache.diff_buf;

    
    if (has_initial_guess) {
        cudaMemcpy(bufs[0], centralities,
                   num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        cudaMemset(bufs[0], 0, num_vertices * sizeof(float));
    }

    cudaStream_t stream = 0;
    bool converged = false;
    std::size_t iterations = 0;
    int cur = 0;

    for (std::size_t it = 0; it < max_iterations; it++) {
        float* x_old = bufs[cur];
        float* x_new = bufs[1 - cur];

        
        cudaMemsetAsync(d_diff, 0, sizeof(float), stream);

        
        launch_katz_high_degree(d_offsets, d_indices, x_old, x_new,
                                alpha, beta, betas,
                                s0, s1, d_diff, stream);
        launch_katz_mid_degree(d_offsets, d_indices, x_old, x_new,
                               alpha, beta, betas,
                               s1, s2, d_diff, stream);
        launch_katz_low_zero(d_offsets, d_indices, x_old, x_new,
                             alpha, beta, betas,
                             s2, s4, d_diff, stream);

        iterations = it + 1;
        cur = 1 - cur;

        
        float h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);

        if (h_diff < epsilon) {
            converged = true;
            break;
        }
    }

    
    float* result = bufs[cur];

    
    if (normalize) {
        cudaMemsetAsync(d_diff, 0, sizeof(float), stream);
        launch_normalize(result, num_vertices, d_diff, stream);
    }

    
    cudaMemcpy(centralities, result,
               num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);

    return {iterations, converged};
}

}  
