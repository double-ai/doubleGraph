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
    float* d_norm_sq = nullptr;
    float* d_l1_diff = nullptr;
    float* x_buf0 = nullptr;
    float* x_buf1 = nullptr;
    int64_t norm_capacity = 0;
    int64_t diff_capacity = 0;
    int64_t buf0_capacity = 0;
    int64_t buf1_capacity = 0;

    void ensure(int64_t num_vertices) {
        if (norm_capacity < 1) {
            if (d_norm_sq) cudaFree(d_norm_sq);
            cudaMalloc(&d_norm_sq, sizeof(float));
            norm_capacity = 1;
        }
        if (diff_capacity < 1) {
            if (d_l1_diff) cudaFree(d_l1_diff);
            cudaMalloc(&d_l1_diff, sizeof(float));
            diff_capacity = 1;
        }
        if (buf0_capacity < num_vertices) {
            if (x_buf0) cudaFree(x_buf0);
            cudaMalloc(&x_buf0, num_vertices * sizeof(float));
            buf0_capacity = num_vertices;
        }
        if (buf1_capacity < num_vertices) {
            if (x_buf1) cudaFree(x_buf1);
            cudaMalloc(&x_buf1, num_vertices * sizeof(float));
            buf1_capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (d_norm_sq) cudaFree(d_norm_sq);
        if (d_l1_diff) cudaFree(d_l1_diff);
        if (x_buf0) cudaFree(x_buf0);
        if (x_buf1) cudaFree(x_buf1);
    }
};

__device__ __forceinline__ bool is_edge_active(const uint32_t* __restrict__ edge_mask, int e) {
    return (edge_mask[e >> 5] >> (e & 31)) & 1;
}






__global__ void __launch_bounds__(256)
unified_spmv_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float* __restrict__ norm_sq_global,
    int high_start, int high_count,
    int mid_start, int mid_count,
    int low_start, int low_zero_count,
    int high_blocks, int mid_blocks)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ union {
        typename cub::BlockReduce<float, 256>::TempStorage cub_temp;
        float warp_vals[8];
    } smem;

    if (bid < high_blocks) {
        
        if (bid >= high_count) return;
        int v = high_start + bid;
        int row_start = offsets[v];
        int row_end = offsets[v + 1];

        
        double dsum = 0.0;
        for (int e = row_start + tid; e < row_end; e += 256) {
            if (is_edge_active(edge_mask, e)) {
                dsum += (double)weights[e] * (double)__ldg(&x_old[indices[e]]);
            }
        }
        float sum = (float)dsum;

        float block_sum = cub::BlockReduce<float, 256>(smem.cub_temp).Sum(sum);

        if (tid == 0) {
            float val = block_sum + x_old[v];
            x_new[v] = val;
            atomicAdd(norm_sq_global, val * val);
        }

    } else if (bid < high_blocks + mid_blocks) {
        
        int local_bid = bid - high_blocks;
        int warp_in_block = tid / 32;
        int lane = tid & 31;
        int warp_global = local_bid * 8 + warp_in_block;

        if (warp_global < mid_count) {
            int v = mid_start + warp_global;
            int row_start = offsets[v];
            int row_end = offsets[v + 1];

            double dsum = 0.0;
            for (int e = row_start + lane; e < row_end; e += 32) {
                if (is_edge_active(edge_mask, e)) {
                    dsum += (double)weights[e] * (double)__ldg(&x_old[indices[e]]);
                }
            }
            float sum = (float)dsum;

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            }

            if (lane == 0) {
                float val = sum + x_old[v];
                x_new[v] = val;
                smem.warp_vals[warp_in_block] = val * val;
            }
        } else {
            if (lane == 0) smem.warp_vals[warp_in_block] = 0.0f;
        }
        __syncthreads();

        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < 8; i++) acc += smem.warp_vals[i];
            if (acc > 0.0f) atomicAdd(norm_sq_global, acc);
        }

    } else {
        
        int local_bid = bid - high_blocks - mid_blocks;
        int vtx_idx = local_bid * 256 + tid;

        float val = 0.0f;
        bool valid = (vtx_idx < low_zero_count);
        if (valid) {
            int v = low_start + vtx_idx;
            int row_start = offsets[v];
            int row_end = offsets[v + 1];

            double dsum = 0.0;
            for (int e = row_start; e < row_end; e++) {
                if (is_edge_active(edge_mask, e)) {
                    dsum += (double)weights[e] * (double)__ldg(&x_old[indices[e]]);
                }
            }
            val = (float)(dsum + (double)x_old[v]);
            x_new[v] = val;
        }

        float val_sq = valid ? val * val : 0.0f;
        float block_norm = cub::BlockReduce<float, 256>(smem.cub_temp).Sum(val_sq);
        if (tid == 0 && block_norm > 0.0f) {
            atomicAdd(norm_sq_global, block_norm);
        }
    }
}




__global__ void __launch_bounds__(256)
normalize_diff_kernel(
    float* __restrict__ x_new,
    const float* __restrict__ x_old,
    const float* __restrict__ norm_sq_global,
    float* __restrict__ l1_diff_global,
    int num_vertices)
{
    int idx = blockIdx.x * 256 + threadIdx.x;

    float norm = sqrtf(*norm_sq_global);
    float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 0.0f;

    float local_diff = 0.0f;
    if (idx < num_vertices) {
        float val = x_new[idx] * inv_norm;
        x_new[idx] = val;
        local_diff = fabsf(val - x_old[idx]);
    }

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    float block_diff = BlockReduce(temp).Sum(local_diff);
    if (threadIdx.x == 0 && block_diff > 0.0f) {
        atomicAdd(l1_diff_global, block_diff);
    }
}

__global__ void init_uniform_kernel(float* __restrict__ x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] = 1.0f / (float)n;
    }
}

}  

eigenvector_centrality_result_t eigenvector_centrality_seg_mask(const graph32_t& graph,
                                      const float* edge_weights,
                                      float* centralities,
                                      float epsilon,
                                      std::size_t max_iterations,
                                      const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;

    cache.ensure(num_vertices);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int high_start = seg[0];
    int mid_start = seg[1];
    int low_start = seg[2];
    int seg_end = seg[4];
    int high_count = mid_start - high_start;
    int mid_count = low_start - mid_start;
    int low_zero_count = seg_end - low_start;

    float* d_x_curr = cache.x_buf0;
    float* d_x_next = cache.x_buf1;

    cudaStream_t stream = 0;

    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(d_x_curr, initial_centralities,
            num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        init_uniform_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
            d_x_curr, num_vertices);
    }

    float conv_threshold = (float)num_vertices * epsilon;
    std::size_t iterations = 0;
    bool converged = false;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        cudaMemsetAsync(cache.d_norm_sq, 0, sizeof(float), stream);
        cudaMemsetAsync(cache.d_l1_diff, 0, sizeof(float), stream);

        int high_blocks = high_count;
        int mid_blocks = (mid_count + 7) / 8;
        int low_blocks = (low_zero_count + 255) / 256;
        int total = high_blocks + mid_blocks + low_blocks;
        if (total > 0) {
            unified_spmv_kernel<<<total, 256, 0, stream>>>(
                d_offsets, d_indices, edge_weights, d_mask,
                d_x_curr, d_x_next, cache.d_norm_sq,
                high_start, high_count, mid_start, mid_count,
                low_start, low_zero_count, high_blocks, mid_blocks);
        }

        normalize_diff_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
            d_x_next, d_x_curr, cache.d_norm_sq, cache.d_l1_diff,
            num_vertices);

        float h_diff;
        cudaMemcpy(&h_diff, cache.d_l1_diff, sizeof(float), cudaMemcpyDeviceToHost);

        float* tmp = d_x_curr;
        d_x_curr = d_x_next;
        d_x_next = tmp;
        iterations = iter + 1;

        if (h_diff < conv_threshold) {
            converged = true;
            break;
        }
    }

    
    if (d_x_curr != centralities) {
        cudaMemcpyAsync(centralities, d_x_curr,
            num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaDeviceSynchronize();
    }

    return eigenvector_centrality_result_t{iterations, converged};
}

}  
