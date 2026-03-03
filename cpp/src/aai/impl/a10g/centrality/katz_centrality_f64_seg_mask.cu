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
#include <cub/block/block_reduce.cuh>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define N_REDUCE_BLOCKS 256





struct Cache : Cacheable {
    double* x0 = nullptr;
    double* x1 = nullptr;
    double* diff_sums = nullptr;
    double* result_buf = nullptr;

    int64_t x0_capacity = 0;
    int64_t x1_capacity = 0;
    int64_t diff_sums_capacity = 0;
    int64_t result_buf_capacity = 0;

    void ensure(int32_t num_vertices, int32_t max_blocks_needed) {
        if (x0_capacity < num_vertices) {
            if (x0) cudaFree(x0);
            cudaMalloc(&x0, (int64_t)num_vertices * sizeof(double));
            x0_capacity = num_vertices;
        }
        if (x1_capacity < num_vertices) {
            if (x1) cudaFree(x1);
            cudaMalloc(&x1, (int64_t)num_vertices * sizeof(double));
            x1_capacity = num_vertices;
        }
        if (diff_sums_capacity < max_blocks_needed) {
            if (diff_sums) cudaFree(diff_sums);
            cudaMalloc(&diff_sums, (int64_t)max_blocks_needed * sizeof(double));
            diff_sums_capacity = max_blocks_needed;
        }
        if (result_buf_capacity < 1) {
            if (result_buf) cudaFree(result_buf);
            cudaMalloc(&result_buf, sizeof(double));
            result_buf_capacity = 1;
        }
    }

    ~Cache() override {
        if (x0) cudaFree(x0);
        if (x1) cudaFree(x1);
        if (diff_sums) cudaFree(diff_sums);
        if (result_buf) cudaFree(result_buf);
    }
};





__global__ void fill_d_kernel(double* __restrict__ dst, double val, int32_t n) {
    int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) dst[i] = val;
}







__global__ void spmv_high_fused(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ x_old,
    double* __restrict__ x_new,
    double alpha, double beta_scalar,
    const double* __restrict__ betas, bool use_betas,
    double* __restrict__ diff_sums, int32_t diff_offset,
    int32_t v_start, int32_t v_end)
{
    int32_t v = blockIdx.x + v_start;
    if (v >= v_end) return;

    int32_t row_start = offsets[v];
    int32_t row_end = offsets[v + 1];

    double sum = 0.0;
    for (int32_t e = row_start + threadIdx.x; e < row_end; e += BLOCK_SIZE) {
        uint32_t mask_word = edge_mask[e >> 5];
        if (mask_word & (1u << (e & 31))) {
            sum += weights[e] * x_old[indices[e]];
        }
    }

    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        double beta_v = use_betas ? betas[v] : beta_scalar;
        double new_val = alpha * sum + beta_v;
        x_new[v] = new_val;
        diff_sums[blockIdx.x + diff_offset] = fabs(new_val - x_old[v]);
    }
}


__global__ void spmv_mid_fused(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ x_old,
    double* __restrict__ x_new,
    double alpha, double beta_scalar,
    const double* __restrict__ betas, bool use_betas,
    double* __restrict__ diff_sums, int32_t diff_offset,
    int32_t v_start, int32_t v_end)
{
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int32_t lane = threadIdx.x & (WARP_SIZE - 1);
    int32_t warp_in_block = threadIdx.x / WARP_SIZE;
    int32_t v = warp_id + v_start;

    double my_diff = 0.0;

    if (v < v_end) {
        int32_t row_start = offsets[v];
        int32_t row_end = offsets[v + 1];

        double sum = 0.0;
        for (int32_t e = row_start + lane; e < row_end; e += WARP_SIZE) {
            uint32_t mask_word = edge_mask[e >> 5];
            if (mask_word & (1u << (e & 31))) {
                sum += weights[e] * x_old[indices[e]];
            }
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            double beta_v = use_betas ? betas[v] : beta_scalar;
            double new_val = alpha * sum + beta_v;
            x_new[v] = new_val;
            my_diff = fabs(new_val - x_old[v]);
        }
    }

    
    __shared__ double warp_diffs[WARPS_PER_BLOCK];
    if (lane == 0) warp_diffs[warp_in_block] = my_diff;
    __syncthreads();

    if (threadIdx.x == 0) {
        double block_diff = 0.0;
        for (int i = 0; i < WARPS_PER_BLOCK; i++) block_diff += warp_diffs[i];
        diff_sums[blockIdx.x + diff_offset] = block_diff;
    }
}


__global__ void spmv_low_zero_fused(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ x_old,
    double* __restrict__ x_new,
    double alpha, double beta_scalar,
    const double* __restrict__ betas, bool use_betas,
    double* __restrict__ diff_sums, int32_t diff_offset,
    int32_t v_start, int32_t v_end)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduceD;
    __shared__ typename BlockReduceD::TempStorage temp_storage;

    int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x + v_start;

    double my_diff = 0.0;

    if (v < v_end) {
        int32_t row_start = offsets[v];
        int32_t row_end = offsets[v + 1];

        double sum = 0.0;
        for (int32_t e = row_start; e < row_end; ++e) {
            uint32_t mask_word = edge_mask[e >> 5];
            if (mask_word & (1u << (e & 31))) {
                sum += weights[e] * x_old[indices[e]];
            }
        }

        double beta_v = use_betas ? betas[v] : beta_scalar;
        double new_val = alpha * sum + beta_v;
        x_new[v] = new_val;
        my_diff = fabs(new_val - x_old[v]);
    }

    double block_diff = BlockReduceD(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0) {
        diff_sums[blockIdx.x + diff_offset] = block_diff;
    }
}






__global__ void reduce_sums_kernel(
    const double* __restrict__ block_sums,
    double* __restrict__ result,
    int32_t n_blocks)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double sum = 0.0;
    for (int32_t i = threadIdx.x; i < n_blocks; i += BLOCK_SIZE) {
        sum += block_sums[i];
    }

    sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) {
        result[0] = sum;
    }
}


__global__ void l2_norm_sq_kernel(
    const double* __restrict__ x,
    double* __restrict__ block_sums,
    int32_t n)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    double sum = 0.0;
    for (int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n;
         i += N_REDUCE_BLOCKS * BLOCK_SIZE) {
        double val = x[i];
        sum += val * val;
    }

    sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = sum;
    }
}


__global__ void normalize_kernel(double* __restrict__ x, double scale, int32_t n) {
    int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) x[i] *= scale;
}

}  

katz_centrality_result_t katz_centrality_seg_mask(const graph32_t& graph,
                               const double* edge_weights,
                               double* centralities,
                               double alpha,
                               double beta,
                               const double* betas,
                               double epsilon,
                               std::size_t max_iterations,
                               bool has_initial_guess,
                               bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    const uint32_t* d_edge_mask = graph.edge_mask;
    const double* d_weights = edge_weights;

    
    const auto& seg_vec = graph.segment_offsets.value();
    int32_t seg[5] = {seg_vec[0], seg_vec[1], seg_vec[2], seg_vec[3], seg_vec[4]};

    
    int32_t high_blocks = seg[1] - seg[0];
    int32_t mid_verts = seg[2] - seg[1];
    int32_t mid_blocks = (mid_verts + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int32_t low_zero_verts = seg[4] - seg[2];
    int32_t low_zero_blocks = (low_zero_verts + 255) / 256;
    int32_t total_diff_blocks = high_blocks + mid_blocks + low_zero_blocks;

    
    int32_t max_blocks_needed = std::max(total_diff_blocks, N_REDUCE_BLOCKS);

    cache.ensure(num_vertices, max_blocks_needed);

    double* x_old = cache.x0;
    double* x_new = cache.x1;
    double* d_diff_sums = cache.diff_sums;
    double* d_result = cache.result_buf;

    cudaStream_t stream = 0;

    bool use_betas = (betas != nullptr);
    const double* d_betas = betas;

    
    if (has_initial_guess) {
        cudaMemcpyAsync(x_old, centralities, num_vertices * sizeof(double),
                       cudaMemcpyDeviceToDevice, stream);
    } else {
        if (use_betas) {
            cudaMemcpyAsync(x_old, d_betas, num_vertices * sizeof(double),
                           cudaMemcpyDeviceToDevice, stream);
        } else {
            int32_t blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
            fill_d_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(x_old, beta, num_vertices);
        }
    }

    
    int32_t high_offset = 0;
    int32_t mid_offset = high_blocks;
    int32_t low_zero_offset = high_blocks + mid_blocks;

    bool converged = false;
    std::size_t iterations = 0;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        {
            int32_t n = seg[1] - seg[0];
            if (n > 0) {
                spmv_high_fused<<<n, BLOCK_SIZE, 0, stream>>>(
                    d_offsets, d_indices, d_weights, d_edge_mask,
                    x_old, x_new, alpha, beta, d_betas, use_betas,
                    d_diff_sums, high_offset, seg[0], seg[1]);
            }
        }

        
        {
            int32_t n = seg[2] - seg[1];
            if (n > 0) {
                int32_t blocks = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
                spmv_mid_fused<<<blocks, BLOCK_SIZE, 0, stream>>>(
                    d_offsets, d_indices, d_weights, d_edge_mask,
                    x_old, x_new, alpha, beta, d_betas, use_betas,
                    d_diff_sums, mid_offset, seg[1], seg[2]);
            }
        }

        
        {
            int32_t n = seg[4] - seg[2];
            if (n > 0) {
                int32_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                spmv_low_zero_fused<<<blocks, BLOCK_SIZE, 0, stream>>>(
                    d_offsets, d_indices, d_weights, d_edge_mask,
                    x_old, x_new, alpha, beta, d_betas, use_betas,
                    d_diff_sums, low_zero_offset, seg[2], seg[4]);
            }
        }

        
        reduce_sums_kernel<<<1, BLOCK_SIZE, 0, stream>>>(d_diff_sums, d_result, total_diff_blocks);

        double h_diff;
        cudaMemcpy(&h_diff, d_result, sizeof(double), cudaMemcpyDeviceToHost);

        iterations = iter + 1;

        double* temp = x_old;
        x_old = x_new;
        x_new = temp;

        if (h_diff < epsilon) {
            converged = true;
            break;
        }
    }

    
    if (normalize) {
        l2_norm_sq_kernel<<<N_REDUCE_BLOCKS, BLOCK_SIZE, 0, stream>>>(x_old, d_diff_sums, num_vertices);
        reduce_sums_kernel<<<1, BLOCK_SIZE, 0, stream>>>(d_diff_sums, d_result, N_REDUCE_BLOCKS);

        double h_norm_sq;
        cudaMemcpy(&h_norm_sq, d_result, sizeof(double), cudaMemcpyDeviceToHost);

        if (h_norm_sq > 0.0) {
            double scale = 1.0 / sqrt(h_norm_sq);
            int32_t blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
            normalize_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(x_old, scale, num_vertices);
        }
    }

    
    cudaMemcpyAsync(centralities, x_old, num_vertices * sizeof(double),
                    cudaMemcpyDeviceToDevice, stream);

    return {iterations, converged};
}

}  
