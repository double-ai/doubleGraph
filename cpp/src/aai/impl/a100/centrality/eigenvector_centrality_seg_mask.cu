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

struct Cache : Cacheable {
    float* buf0 = nullptr;
    float* buf1 = nullptr;
    float* l2_sq = nullptr;
    float* diff = nullptr;

    int32_t buf0_capacity = 0;
    int32_t buf1_capacity = 0;
    bool scalars_allocated = false;

    void ensure(int32_t num_vertices) {
        if (buf0_capacity < num_vertices) {
            if (buf0) cudaFree(buf0);
            cudaMalloc(&buf0, (size_t)num_vertices * sizeof(float));
            buf0_capacity = num_vertices;
        }
        if (buf1_capacity < num_vertices) {
            if (buf1) cudaFree(buf1);
            cudaMalloc(&buf1, (size_t)num_vertices * sizeof(float));
            buf1_capacity = num_vertices;
        }
        if (!scalars_allocated) {
            cudaMalloc(&l2_sq, sizeof(float));
            cudaMalloc(&diff, sizeof(float));
            scalars_allocated = true;
        }
    }

    ~Cache() override {
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (l2_sq) cudaFree(l2_sq);
        if (diff) cudaFree(diff);
    }
};







__global__ void __launch_bounds__(256)
spmv_high_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    int32_t seg_start, int32_t seg_end)
{
    int v = seg_start + blockIdx.x;
    if (v >= seg_end) return;

    int row_start = offsets[v];
    int row_end = offsets[v + 1];

    float sum = 0.0f;
    for (int e = row_start + threadIdx.x; e < row_end; e += 256) {
        uint32_t mask_word = edge_mask[e >> 5];
        if (mask_word & (1u << (e & 31))) {
            sum += x_old[indices[e]];
        }
    }

    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage temp;
    float block_sum = BR(temp).Sum(sum);

    if (threadIdx.x == 0) {
        x_new[v] = block_sum + x_old[v];
    }
}


__global__ void __launch_bounds__(256)
spmv_mid_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    int32_t seg_start, int32_t seg_end)
{
    int warp_id_local = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int global_warp = blockIdx.x * 8 + warp_id_local;
    int v = seg_start + global_warp;
    if (v >= seg_end) return;

    int row_start = offsets[v];
    int row_end = offsets[v + 1];

    float sum = 0.0f;
    for (int e = row_start + lane; e < row_end; e += 32) {
        uint32_t mask_word = edge_mask[e >> 5];
        if (mask_word & (1u << (e & 31))) {
            sum += x_old[indices[e]];
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        x_new[v] = sum + x_old[v];
    }
}


__global__ void __launch_bounds__(256)
spmv_low_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    int32_t seg_start, int32_t seg_end)
{
    int v = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= seg_end) return;

    int row_start = offsets[v];
    int row_end = offsets[v + 1];

    float sum = 0.0f;
    for (int e = row_start; e < row_end; ++e) {
        uint32_t mask_word = edge_mask[e >> 5];
        if (mask_word & (1u << (e & 31))) {
            sum += x_old[indices[e]];
        }
    }

    x_new[v] = sum + x_old[v];
}


__global__ void spmv_zero_degree(
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    int32_t seg_start, int32_t seg_end)
{
    int v = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= seg_end) return;
    x_new[v] = x_old[v];
}






__global__ void __launch_bounds__(256)
reduce_sum_sq_kernel(
    const float* __restrict__ x,
    float* __restrict__ result,
    int32_t n)
{
    float sum = 0.0f;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        float val = x[i];
        sum += val * val;
    }

    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage temp;
    float block_sum = BR(temp).Sum(sum);

    if (threadIdx.x == 0) {
        atomicAdd(result, block_sum);
    }
}


__global__ void __launch_bounds__(256)
normalize_diff_kernel(
    float* __restrict__ x_new,
    const float* __restrict__ x_old,
    const float* __restrict__ l2_sq_ptr,
    float* __restrict__ diff_result,
    int32_t n)
{
    float l2_sq = *l2_sq_ptr;
    float inv_norm = (l2_sq > 0.0f) ? rsqrtf(l2_sq) : 1.0f;

    float sum = 0.0f;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        float val = x_new[i] * inv_norm;
        x_new[i] = val;
        sum += fabsf(val - x_old[i]);
    }

    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage temp;
    float block_sum = BR(temp).Sum(sum);

    if (threadIdx.x == 0) {
        atomicAdd(diff_result, block_sum);
    }
}


__global__ void init_uniform_kernel(float* x, float val, int32_t n) {
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        x[i] = val;
    }
}





static inline int clamp_blocks(int b, int max_b) {
    return b > max_b ? max_b : (b < 1 ? 1 : b);
}

static void launch_spmv(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    const float* x_old, float* x_new,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3, int32_t seg4,
    cudaStream_t stream)
{
    
    if (seg1 > seg0) {
        int n = seg1 - seg0;
        spmv_high_degree<<<n, 256, 0, stream>>>(
            offsets, indices, edge_mask, x_old, x_new, seg0, seg1);
    }

    
    if (seg2 > seg1) {
        int n = seg2 - seg1;
        int blocks = (n + 7) / 8;
        spmv_mid_degree<<<blocks, 256, 0, stream>>>(
            offsets, indices, edge_mask, x_old, x_new, seg1, seg2);
    }

    
    if (seg3 > seg2) {
        int n = seg3 - seg2;
        int blocks = (n + 255) / 256;
        spmv_low_degree<<<blocks, 256, 0, stream>>>(
            offsets, indices, edge_mask, x_old, x_new, seg2, seg3);
    }

    
    if (seg4 > seg3) {
        int n = seg4 - seg3;
        int blocks = (n + 255) / 256;
        spmv_zero_degree<<<blocks, 256, 0, stream>>>(
            x_old, x_new, seg3, seg4);
    }
}

static void launch_reduce_sum_sq(const float* x, float* result, int32_t n, cudaStream_t stream) {
    cudaMemsetAsync(result, 0, sizeof(float), stream);
    int blocks = clamp_blocks((n + 255) / 256, 1024);
    reduce_sum_sq_kernel<<<blocks, 256, 0, stream>>>(x, result, n);
}

static void launch_normalize_diff(float* x_new, const float* x_old,
                                  const float* l2_sq, float* diff_result,
                                  int32_t n, cudaStream_t stream) {
    cudaMemsetAsync(diff_result, 0, sizeof(float), stream);
    int blocks = clamp_blocks((n + 255) / 256, 1024);
    normalize_diff_kernel<<<blocks, 256, 0, stream>>>(x_new, x_old, l2_sq, diff_result, n);
}

static void launch_init_uniform(float* x, int32_t n, cudaStream_t stream) {
    float val = 1.0f / (float)n;
    int blocks = clamp_blocks((n + 255) / 256, 1024);
    init_uniform_kernel<<<blocks, 256, 0, stream>>>(x, val, n);
}

}  

eigenvector_centrality_result_t eigenvector_centrality_seg_mask(const graph32_t& graph,
                                      float* centralities,
                                      float epsilon,
                                      std::size_t max_iterations,
                                      const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    
    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2];
    int32_t seg3 = seg[3], seg4 = seg[4];

    cudaStream_t stream = 0;

    
    cache.ensure(num_vertices);

    float* bufs[2] = {cache.buf0, cache.buf1};
    float* d_l2_sq = cache.l2_sq;
    float* d_diff = cache.diff;

    int cur = 0; 

    
    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(bufs[cur], initial_centralities,
                        (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        launch_init_uniform(bufs[cur], num_vertices, stream);
    }

    float threshold = (float)num_vertices * epsilon;
    std::size_t iterations = 0;
    bool converged = false;
    const int CHECK_INTERVAL = 10;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        float* x_old = bufs[cur];
        float* x_new = bufs[1 - cur];

        
        launch_spmv(d_offsets, d_indices, d_edge_mask, x_old, x_new,
                    seg0, seg1, seg2, seg3, seg4, stream);

        
        launch_reduce_sum_sq(x_new, d_l2_sq, num_vertices, stream);

        
        launch_normalize_diff(x_new, x_old, d_l2_sq, d_diff, num_vertices, stream);

        iterations = iter + 1;
        cur = 1 - cur; 

        
        if (iterations % CHECK_INTERVAL == 0 || iter == max_iterations - 1) {
            float h_diff;
            cudaMemcpyAsync(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (h_diff < threshold) {
                converged = true;
                break;
            }
        }
    }

    
    cudaMemcpyAsync(centralities, bufs[cur],
                    (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return {iterations, converged};
}

}  
