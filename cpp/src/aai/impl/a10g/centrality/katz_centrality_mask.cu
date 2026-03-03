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

namespace aai {

namespace {

#define BLOCK_V 256
#define BLOCK_SPMV 256
#define WARPS_PER_BLOCK (BLOCK_SPMV / 32)

struct Cache : Cacheable {
    float* buf = nullptr;
    float* scratch = nullptr;
    int64_t buf_capacity = 0;
    bool scratch_allocated = false;

    void ensure(int32_t num_vertices) {
        int64_t needed = num_vertices;
        if (buf_capacity < needed) {
            if (buf) cudaFree(buf);
            cudaMalloc(&buf, needed * sizeof(float));
            buf_capacity = needed;
        }
        if (!scratch_allocated) {
            cudaMalloc(&scratch, sizeof(float));
            scratch_allocated = true;
        }
    }

    ~Cache() override {
        if (buf) cudaFree(buf);
        if (scratch) cudaFree(scratch);
    }
};




__global__ void fill_constant(float* __restrict__ out, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = val;
}




__global__ void analytical_iter2(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ x,
    float alpha_beta,
    float beta,
    int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int start = offsets[v];
    int end = offsets[v + 1];

    int count = 0;
    for (int j = start; j < end; j++) {
        count += (edge_mask[j >> 5] >> (j & 31)) & 1;
    }
    x[v] = alpha_beta * (float)count + beta;
}





__global__ void spmv_thread_conv(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha,
    float beta,
    const float* __restrict__ betas,
    int num_vertices,
    float* __restrict__ d_diff_sum
) {
    typedef cub::BlockReduce<float, BLOCK_V> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float my_diff = 0.0f;

    if (v < num_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];

        float sum = 0.0f;
        for (int j = start; j < end; j++) {
            if ((edge_mask[j >> 5] >> (j & 31)) & 1) {
                sum += __ldg(&x_old[__ldg(&indices[j])]);
            }
        }

        float beta_v = betas ? betas[v] : beta;
        float new_val = alpha * sum + beta_v;
        x_new[v] = new_val;
        my_diff = fabsf(new_val - x_old[v]);
    }

    float block_diff = BlockReduce(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff > 0.0f) {
        atomicAdd(d_diff_sum, block_diff);
    }
}





__global__ void spmv_warp_conv(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha,
    float beta,
    const float* __restrict__ betas,
    int num_vertices,
    float* __restrict__ d_diff_sum
) {
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;

    __shared__ float s_diff[WARPS_PER_BLOCK];
    float warp_diff = 0.0f;

    if (warp_global < num_vertices) {
        int v = warp_global;
        int start = offsets[v];
        int end = offsets[v + 1];

        float sum = 0.0f;
        for (int j = start + lane; j < end; j += 32) {
            if ((edge_mask[j >> 5] >> (j & 31)) & 1) {
                sum += __ldg(&x_old[__ldg(&indices[j])]);
            }
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            float beta_v = betas ? betas[v] : beta;
            float new_val = alpha * sum + beta_v;
            x_new[v] = new_val;
            warp_diff = fabsf(new_val - x_old[v]);
        }
    }

    if (lane == 0) s_diff[warp_in_block] = warp_diff;
    __syncthreads();

    if (warp_in_block == 0 && lane < WARPS_PER_BLOCK) {
        float val = s_diff[lane];
        #pragma unroll
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0 && val > 0.0f) {
            atomicAdd(d_diff_sum, val);
        }
    }
}




__global__ void l2_norm_squared_kernel(const float* __restrict__ x, float* __restrict__ d_result, int n) {
    typedef cub::BlockReduce<float, BLOCK_V> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? x[idx] : 0.0f;
    float block_sum = BlockReduce(temp_storage).Sum(val * val);
    if (threadIdx.x == 0 && block_sum > 0.0f) atomicAdd(d_result, block_sum);
}


__global__ void normalize_kernel(float* __restrict__ x, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= scale;
}

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

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cache.ensure(num_vertices);

    bool use_betas = (betas != nullptr);

    
    int avg_degree = (num_vertices > 0) ? (num_edges / num_vertices) : 0;
    bool use_warp = (avg_degree >= 6);

    
    float* bufs[2] = {centralities, cache.buf};
    float* d_scratch = cache.scratch;
    int cur = 0;

    std::size_t iter = 0;
    bool converged = false;

    
    int skip_conv = 0;

    if (!has_initial_guess) {
        if (max_iterations >= 1) {
            if (use_betas) {
                cudaMemcpyAsync(bufs[cur], betas,
                                (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
            } else {
                if (num_vertices > 0)
                    fill_constant<<<(num_vertices + BLOCK_V - 1) / BLOCK_V, BLOCK_V>>>(
                        bufs[cur], beta, num_vertices);
            }
            iter = 1;

            if (!use_betas && iter < max_iterations) {
                if (num_vertices > 0)
                    analytical_iter2<<<(num_vertices + BLOCK_V - 1) / BLOCK_V, BLOCK_V>>>(
                        d_offsets, d_edge_mask, bufs[1 - cur], alpha * beta, beta, num_vertices);
                cur = 1 - cur;
                iter = 2;
                if (alpha < 0.01f) skip_conv = 1;
            }
        }
    }
    

    
    auto do_spmv_conv = [&]() {
        cudaMemsetAsync(d_scratch, 0, sizeof(float));
        if (use_warp) {
            if (num_vertices > 0) {
                int64_t total_threads = (int64_t)num_vertices * 32;
                int grid = (int)((total_threads + BLOCK_SPMV - 1) / BLOCK_SPMV);
                spmv_warp_conv<<<grid, BLOCK_SPMV>>>(
                    d_offsets, d_indices, d_edge_mask,
                    bufs[cur], bufs[1-cur], alpha, beta, betas,
                    num_vertices, d_scratch);
            }
        } else {
            if (num_vertices > 0)
                spmv_thread_conv<<<(num_vertices + BLOCK_V - 1) / BLOCK_V, BLOCK_V>>>(
                    d_offsets, d_indices, d_edge_mask,
                    bufs[cur], bufs[1-cur], alpha, beta, betas,
                    num_vertices, d_scratch);
        }
        cur = 1 - cur;
    };

    
    for (; iter < max_iterations && skip_conv > 0; iter++, skip_conv--) {
        do_spmv_conv();
    }

    
    for (; iter < max_iterations; iter++) {
        do_spmv_conv();

        float h_diff;
        cudaMemcpy(&h_diff, d_scratch, sizeof(float), cudaMemcpyDeviceToHost);

        if (h_diff < epsilon) {
            converged = true;
            iter++;
            break;
        }
    }

    
    if (normalize) {
        if (num_vertices > 0) {
            cudaMemsetAsync(d_scratch, 0, sizeof(float));
            l2_norm_squared_kernel<<<(num_vertices + BLOCK_V - 1) / BLOCK_V, BLOCK_V>>>(
                bufs[cur], d_scratch, num_vertices);
            float h_l2_sq;
            cudaMemcpy(&h_l2_sq, d_scratch, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_l2_sq > 0.0f) {
                normalize_kernel<<<(num_vertices + BLOCK_V - 1) / BLOCK_V, BLOCK_V>>>(
                    bufs[cur], 1.0f / sqrtf(h_l2_sq), num_vertices);
            }
        }
    }

    
    if (cur != 0) {
        cudaMemcpyAsync(centralities, bufs[cur],
                        (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return {iter, converged};
}

}  
