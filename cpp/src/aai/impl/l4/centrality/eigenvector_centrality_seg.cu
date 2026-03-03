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
#include <climits>

namespace aai {

namespace {

namespace cg = cooperative_groups;

static constexpr int MAX_BLOCKS = 2048;

struct Cache : Cacheable {
    float* d_partial = nullptr;
    float* d_inv_norm = nullptr;
    float* d_diff = nullptr;
    int* d_iterations = nullptr;
    int* d_converged = nullptr;
    float* scratch = nullptr;
    int64_t scratch_capacity = 0;

    Cache() {
        cudaMalloc(&d_partial, MAX_BLOCKS * sizeof(float));
        cudaMalloc(&d_inv_norm, sizeof(float));
        cudaMalloc(&d_diff, sizeof(float));
        cudaMalloc(&d_iterations, sizeof(int));
        cudaMalloc(&d_converged, sizeof(int));
    }

    ~Cache() override {
        if (d_partial) cudaFree(d_partial);
        if (d_inv_norm) cudaFree(d_inv_norm);
        if (d_diff) cudaFree(d_diff);
        if (d_iterations) cudaFree(d_iterations);
        if (d_converged) cudaFree(d_converged);
        if (scratch) cudaFree(scratch);
    }

    void ensure(int64_t n) {
        if (scratch_capacity < n) {
            if (scratch) cudaFree(scratch);
            cudaMalloc(&scratch, n * sizeof(float));
            scratch_capacity = n;
        }
    }
};

__device__ __forceinline__ float block_sum_256(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);

    __shared__ float s_warp[8];
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    if (lane == 0) s_warp[warp] = val;
    __syncthreads();

    if (threadIdx.x < 8) {
        val = s_warp[threadIdx.x];
        val += __shfl_xor_sync(0xff, val, 4);
        val += __shfl_xor_sync(0xff, val, 2);
        val += __shfl_xor_sync(0xff, val, 1);
    }
    return val;
}

__global__ __launch_bounds__(256, 3)
void power_iteration_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    float* __restrict__ buf_a,
    float* __restrict__ buf_b,
    const int n,
    const int seg0, const int seg1, const int seg2, const int seg4,
    const float threshold,
    const int max_iterations,
    float* __restrict__ partial_buf,
    float* __restrict__ d_inv_norm,
    float* __restrict__ d_diff,
    int* __restrict__ d_iterations,
    int* __restrict__ d_converged)
{
    cg::grid_group grid = cg::this_grid();

    const int tid = blockIdx.x * 256 + threadIdx.x;
    const int total_threads = gridDim.x * 256;
    const int total_warps = total_threads >> 5;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_blocks = gridDim.x;

    float* x = buf_a;
    float* y = buf_b;

    for (int iter = 0; iter < max_iterations; iter++) {
        

        
        for (int v = blockIdx.x + seg0; v < seg1; v += num_blocks) {
            int rb = offsets[v], re = offsets[v + 1];
            float s = 0.0f;
            for (int j = rb + threadIdx.x; j < re; j += 256)
                s += __ldg(&x[indices[j]]);
            s = block_sum_256(s);
            if (threadIdx.x == 0) y[v] = s + x[v];
            __syncthreads();
        }

        
        for (int v = warp_id + seg1; v < seg2; v += total_warps) {
            int rb = offsets[v], re = offsets[v + 1];
            float s = 0.0f;
            for (int j = rb + lane; j < re; j += 32)
                s += __ldg(&x[indices[j]]);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                s += __shfl_down_sync(0xffffffff, s, o);
            if (lane == 0) y[v] = s + x[v];
        }

        
        for (int v = tid + seg2; v < seg4; v += total_threads) {
            int rb = offsets[v], re = offsets[v + 1];
            float s = x[v];
            for (int j = rb; j < re; j++)
                s += __ldg(&x[indices[j]]);
            y[v] = s;
        }

        grid.sync();

        
        {
            float ns = 0.0f;
            for (int i = tid; i < n; i += total_threads) {
                float v = y[i]; ns += v * v;
            }
            float bn = block_sum_256(ns);
            if (threadIdx.x == 0) partial_buf[blockIdx.x] = bn;
        }

        grid.sync();

        if (blockIdx.x == 0) {
            float s = 0.0f;
            for (int i = threadIdx.x; i < num_blocks; i += 256)
                s += partial_buf[i];
            float t = block_sum_256(s);
            if (threadIdx.x == 0)
                *d_inv_norm = (t > 0.0f) ? rsqrtf(t) : 0.0f;
        }

        grid.sync();

        
        {
            __shared__ float s_inv;
            if (threadIdx.x == 0) s_inv = *d_inv_norm;
            __syncthreads();
            float inv = s_inv;

            float ds = 0.0f;
            for (int i = tid; i < n; i += total_threads) {
                float yv = y[i] * inv;
                ds += fabsf(yv - x[i]);
                y[i] = yv;
            }
            float bd = block_sum_256(ds);
            if (threadIdx.x == 0) partial_buf[blockIdx.x] = bd;
        }

        grid.sync();

        if (blockIdx.x == 0) {
            float s = 0.0f;
            for (int i = threadIdx.x; i < num_blocks; i += 256)
                s += partial_buf[i];
            float t = block_sum_256(s);
            if (threadIdx.x == 0) {
                *d_diff = t;
                if (t < threshold) {
                    *d_converged = 1;
                    *d_iterations = iter + 1;
                }
            }
        }

        grid.sync();

        if (*d_converged) break;

        float* tmp = x; x = y; y = tmp;
    }

    if (blockIdx.x == 0 && threadIdx.x == 0 && !(*d_converged))
        *d_iterations = max_iterations;
}

__global__ void init_kernel(float* x, int n, float val) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        x[i] = val;
}

void launch_init(float* x, int n, float val) {
    int grid = (n + 255) / 256;
    if (grid > 1024) grid = 1024;
    init_kernel<<<grid, 256>>>(x, n, val);
}

void launch_power_iteration(
    const int* offsets, const int* indices,
    float* buf_a, float* buf_b,
    int n, int seg0, int seg1, int seg2, int seg4,
    float threshold, int max_iterations,
    float* partial_buf, float* d_inv_norm, float* d_diff,
    int* d_iterations, int* d_converged)
{
    int num_blocks_per_sm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm, power_iteration_kernel, 256, 0);
    int device;
    cudaGetDevice(&device);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);
    int max_blocks = num_blocks_per_sm * num_sms;

    int desired = (n + 255) / 256;
    int num_blocks = desired < max_blocks ? desired : max_blocks;
    if (num_blocks < 1) num_blocks = 1;

    int num_high = seg1 - seg0;
    if (num_blocks < num_high && num_high <= max_blocks)
        num_blocks = num_high;

    void* args[] = {
        &offsets, &indices, &buf_a, &buf_b,
        &n, &seg0, &seg1, &seg2, &seg4,
        &threshold, &max_iterations,
        &partial_buf, &d_inv_norm, &d_diff,
        &d_iterations, &d_converged
    };

    cudaLaunchCooperativeKernel(
        (void*)power_iteration_kernel,
        dim3(num_blocks), dim3(256), args, 0, 0);
}

}  

eigenvector_centrality_result_t eigenvector_centrality_seg(
    const graph32_t& graph,
    float* centralities,
    float epsilon,
    std::size_t max_iterations,
    const float* initial_centralities)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t n = graph.number_of_vertices;
    cache.ensure(n);

    float* d_buf_a = centralities;
    float* d_buf_b = cache.scratch;

    if (initial_centralities != nullptr) {
        cudaMemcpy(d_buf_a, initial_centralities,
                   (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        launch_init(d_buf_a, n, 1.0f / n);
    }

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg4 = seg[4];

    float threshold = (float)n * epsilon;

    int effective_max_iterations =
        (max_iterations > (std::size_t)INT_MAX) ? INT_MAX : (int)max_iterations;

    int zero = 0;
    cudaMemcpy(cache.d_converged, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cache.d_iterations, &zero, sizeof(int), cudaMemcpyHostToDevice);

    launch_power_iteration(
        graph.offsets, graph.indices,
        d_buf_a, d_buf_b,
        n, seg0, seg1, seg2, seg4,
        threshold, effective_max_iterations,
        cache.d_partial, cache.d_inv_norm, cache.d_diff,
        cache.d_iterations, cache.d_converged);

    cudaDeviceSynchronize();

    int h_iterations, h_converged;
    cudaMemcpy(&h_iterations, cache.d_iterations, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_converged, cache.d_converged, sizeof(int), cudaMemcpyDeviceToHost);

    float* result_ptr;
    if (h_converged) {
        result_ptr = ((h_iterations - 1) % 2 == 0) ? d_buf_b : d_buf_a;
    } else {
        result_ptr = (h_iterations % 2 == 0) ? d_buf_a : d_buf_b;
    }

    if (result_ptr != centralities) {
        cudaMemcpy(centralities, result_ptr,
                   (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return eigenvector_centrality_result_t{
        .iterations = (std::size_t)h_iterations,
        .converged = (h_converged != 0)
    };
}

}  
