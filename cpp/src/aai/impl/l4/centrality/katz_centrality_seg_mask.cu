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
#include <cmath>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* x_buf1 = nullptr;
    float* x_buf2 = nullptr;
    float* diff = nullptr;
    int64_t x_buf1_capacity = 0;
    int64_t x_buf2_capacity = 0;
    bool diff_allocated = false;

    void ensure(int32_t num_vertices) {
        if (x_buf1_capacity < num_vertices) {
            if (x_buf1) cudaFree(x_buf1);
            cudaMalloc(&x_buf1, (size_t)num_vertices * sizeof(float));
            x_buf1_capacity = num_vertices;
        }
        if (x_buf2_capacity < num_vertices) {
            if (x_buf2) cudaFree(x_buf2);
            cudaMalloc(&x_buf2, (size_t)num_vertices * sizeof(float));
            x_buf2_capacity = num_vertices;
        }
        if (!diff_allocated) {
            cudaMalloc(&diff, sizeof(float));
            diff_allocated = true;
        }
    }

    ~Cache() override {
        if (x_buf1) cudaFree(x_buf1);
        if (x_buf2) cudaFree(x_buf2);
        if (diff) cudaFree(diff);
    }
};




__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    if (lane == 0) smem[warp] = val;
    __syncthreads();

    if (threadIdx.x < 32) {
        val = (threadIdx.x < nwarps) ? smem[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffff, val, off);
    }
    return val;
}




__global__ void __launch_bounds__(256)
katz_skip2_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ x,
    float alpha_beta,
    float beta,
    int32_t num_vertices)
{
    int v = blockIdx.x * 256 + threadIdx.x;
    if (v >= num_vertices) return;

    int s = offsets[v], e = offsets[v + 1];
    int count = 0;

    if (s < e) {
        int s_word = s >> 5;
        int e_word = (e - 1) >> 5;
        int s_bit = s & 31;
        int e_bit = (e - 1) & 31;

        if (s_word == e_word) {
            uint32_t mask = edge_mask[s_word] >> s_bit;
            mask &= (~0u >> (31 - e_bit + s_bit));
            count = __popc(mask);
        } else {
            count = __popc(edge_mask[s_word] >> s_bit);
            for (int w = s_word + 1; w < e_word; w++)
                count += __popc(edge_mask[w]);
            count += __popc(edge_mask[e_word] & (~0u >> (31 - e_bit)));
        }
    }

    x[v] = alpha_beta * (float)count + beta;
}




__global__ void __launch_bounds__(256)
katz_high(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float* __restrict__ betas,
    float alpha, float beta_scalar,
    float* __restrict__ g_diff,
    int32_t v_start, int32_t v_end)
{
    __shared__ float smem[8];

    int32_t v = v_start + blockIdx.x;
    if (v >= v_end) return;

    int32_t s = offsets[v], e = offsets[v + 1];
    float sum = 0.0f;
    for (int32_t i = s + threadIdx.x; i < e; i += 256) {
        uint32_t mw = edge_mask[i >> 5];
        if (mw & (1u << (i & 31)))
            sum += x_old[indices[i]];
    }
    sum = block_reduce_sum(sum, smem);

    float diff = 0.0f;
    if (threadIdx.x == 0) {
        float r = alpha * sum + (betas ? betas[v] : beta_scalar);
        x_new[v] = r;
        diff = fabsf(r - x_old[v]);
    }

    __syncthreads();
    diff = block_reduce_sum(diff, smem);
    if (threadIdx.x == 0 && diff > 0.0f)
        atomicAdd(g_diff, diff);
}




__global__ void __launch_bounds__(256)
katz_mid(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float* __restrict__ betas,
    float alpha, float beta_scalar,
    float* __restrict__ g_diff,
    int32_t v_start, int32_t v_end)
{
    constexpr int WARPS = 8;
    int warp_global = (blockIdx.x * 256 + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int local_warp = threadIdx.x >> 5;
    int v = v_start + warp_global;

    __shared__ float warp_diffs[WARPS];

    float my_diff = 0.0f;
    if (v < v_end) {
        int32_t s = offsets[v], e = offsets[v + 1];
        float sum = 0.0f;
        for (int32_t i = s + lane; i < e; i += 32) {
            uint32_t mw = edge_mask[i >> 5];
            if (mw & (1u << (i & 31)))
                sum += x_old[indices[i]];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, off);

        if (lane == 0) {
            float r = alpha * sum + (betas ? betas[v] : beta_scalar);
            x_new[v] = r;
            my_diff = fabsf(r - x_old[v]);
        }
    }

    if (lane == 0) warp_diffs[local_warp] = my_diff;
    __syncthreads();

    if (threadIdx.x < 32) {
        float d = (threadIdx.x < WARPS) ? warp_diffs[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            d += __shfl_down_sync(0xffffffff, d, off);
        if (threadIdx.x == 0 && d > 0.0f)
            atomicAdd(g_diff, d);
    }
}




__global__ void __launch_bounds__(256)
katz_low(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float* __restrict__ betas,
    float alpha, float beta_scalar,
    float* __restrict__ g_diff,
    int32_t v_start, int32_t v_end)
{
    __shared__ float smem[8];

    int32_t v = v_start + blockIdx.x * 256 + threadIdx.x;
    float d = 0.0f;
    if (v < v_end) {
        int32_t s = offsets[v], e = offsets[v + 1];
        float sum = 0.0f;
        for (int32_t i = s; i < e; i++) {
            uint32_t mw = edge_mask[i >> 5];
            if (mw & (1u << (i & 31)))
                sum += x_old[indices[i]];
        }
        float r = alpha * sum + (betas ? betas[v] : beta_scalar);
        x_new[v] = r;
        d = fabsf(r - x_old[v]);
    }
    d = block_reduce_sum(d, smem);
    if (threadIdx.x == 0 && d > 0.0f)
        atomicAdd(g_diff, d);
}




__global__ void fill_scalar_kernel(float* x, float val, int32_t n) {
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx < n) x[idx] = val;
}




__global__ void l2_norm_sq(const float* __restrict__ x, float* out, int32_t n) {
    __shared__ float smem[8];
    float s = 0.0f;
    for (int32_t i = blockIdx.x * 256 + threadIdx.x; i < n;
         i += gridDim.x * 256) {
        float v = x[i]; s += v * v;
    }
    s = block_reduce_sum(s, smem);
    if (threadIdx.x == 0) atomicAdd(out, s);
}

__global__ void l2_scale(float* x, const float* norm_sq, int32_t n) {
    float sc = rsqrtf(*norm_sq);
    for (int32_t i = blockIdx.x * 256 + threadIdx.x; i < n;
         i += gridDim.x * 256)
        x[i] *= sc;
}





void launch_katz_skip2(
    const int32_t* offsets, const uint32_t* edge_mask,
    float* x, float alpha, float beta, int32_t num_vertices)
{
    if (num_vertices > 0) {
        int blocks = (num_vertices + 255) / 256;
        katz_skip2_kernel<<<blocks, 256>>>(offsets, edge_mask, x,
            alpha * beta, beta, num_vertices);
    }
}

void launch_katz_high(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    const float* x_old, float* x_new, const float* betas,
    float alpha, float beta_scalar, float* g_diff,
    int32_t v_start, int32_t v_end)
{
    int n = v_end - v_start;
    if (n > 0)
        katz_high<<<n, 256>>>(offsets, indices, edge_mask,
            x_old, x_new, betas, alpha, beta_scalar, g_diff, v_start, v_end);
}

void launch_katz_mid(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    const float* x_old, float* x_new, const float* betas,
    float alpha, float beta_scalar, float* g_diff,
    int32_t v_start, int32_t v_end)
{
    int n = v_end - v_start;
    if (n > 0) {
        int blocks = (n + 7) / 8;
        katz_mid<<<blocks, 256>>>(offsets, indices, edge_mask,
            x_old, x_new, betas, alpha, beta_scalar, g_diff, v_start, v_end);
    }
}

void launch_katz_low(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    const float* x_old, float* x_new, const float* betas,
    float alpha, float beta_scalar, float* g_diff,
    int32_t v_start, int32_t v_end)
{
    int n = v_end - v_start;
    if (n > 0) {
        int blocks = (n + 255) / 256;
        katz_low<<<blocks, 256>>>(offsets, indices, edge_mask,
            x_old, x_new, betas, alpha, beta_scalar, g_diff, v_start, v_end);
    }
}

void launch_fill_scalar(float* x, float val, int32_t n) {
    if (n > 0) {
        int blocks = (n + 255) / 256;
        fill_scalar_kernel<<<blocks, 256>>>(x, val, n);
    }
}

void launch_l2_normalize(float* x, int32_t n, float* temp_buf) {
    cudaMemsetAsync(temp_buf, 0, sizeof(float));
    int blocks = (n + 255) / 256;
    if (blocks > 1024) blocks = 1024;
    l2_norm_sq<<<blocks, 256>>>(x, temp_buf, n);
    l2_scale<<<blocks, 256>>>(x, temp_buf, n);
}

}  

katz_centrality_result_t katz_centrality_seg_mask(const graph32_t& graph,
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
    cache.ensure(num_vertices);

    const int32_t*  d_offsets   = graph.offsets;
    const int32_t*  d_indices   = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int32_t sv[5] = {seg[0], seg[1], seg[2], seg[3], seg[4]};

    float* x_ptrs[2] = {cache.x_buf1, cache.x_buf2};
    float* d_diff = cache.diff;

    int cur = 0;
    std::size_t num_iters = 0;
    bool converged = false;
    bool use_betas = (betas != nullptr);

    if (has_initial_guess) {
        cudaMemcpyAsync(x_ptrs[0], centralities,
                   num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    } else if (!use_betas && max_iterations >= 2) {
        launch_katz_skip2(d_offsets, d_edge_mask, x_ptrs[0],
                          alpha, beta, num_vertices);
        num_iters = 2;
    } else if (!use_betas) {
        launch_fill_scalar(x_ptrs[0], beta, num_vertices);
        num_iters = 1;
    } else {
        cudaMemcpyAsync(x_ptrs[0], betas,
                   num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
        num_iters = 1;
    }

    while (!converged && num_iters < max_iterations) {
        float* x_old = x_ptrs[cur];
        float* x_new = x_ptrs[cur ^ 1];

        cudaMemsetAsync(d_diff, 0, sizeof(float));

        launch_katz_high(d_offsets, d_indices, d_edge_mask,
                         x_old, x_new, betas,
                         alpha, beta, d_diff,
                         sv[0], sv[1]);
        launch_katz_mid(d_offsets, d_indices, d_edge_mask,
                        x_old, x_new, betas,
                        alpha, beta, d_diff,
                        sv[1], sv[2]);
        launch_katz_low(d_offsets, d_indices, d_edge_mask,
                        x_old, x_new, betas,
                        alpha, beta, d_diff,
                        sv[2], sv[4]);

        float h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);

        cur ^= 1;
        num_iters++;

        if (h_diff < epsilon) {
            converged = true;
        }
    }

    if (normalize) {
        launch_l2_normalize(x_ptrs[cur], num_vertices, d_diff);
        cudaDeviceSynchronize();
    }

    cudaMemcpyAsync(centralities, x_ptrs[cur],
               (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);

    return {num_iters, converged};
}

}  
