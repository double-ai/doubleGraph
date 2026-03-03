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
#include <utility>

namespace aai {

namespace {




__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int WARPS = BLOCK_SIZE / 32;
    __shared__ float warp_sums[WARPS];
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    val = warp_reduce_sum(val);
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = (lane < WARPS) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}




__global__ void katz_fill_beta(
    float* __restrict__ x_new,
    float beta,
    const float* __restrict__ betas,
    int32_t num_vertices,
    float* __restrict__ diff_sum
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0.0f;
    if (tid < num_vertices) {
        float beta_v = betas ? betas[tid] : beta;
        x_new[tid] = beta_v;
        diff = fabsf(beta_v);
    }
    float block_diff = block_reduce_sum<256>(diff);
    if (threadIdx.x == 0 && block_diff > 0.0f) atomicAdd(diff_sum, block_diff);
}




__global__ void katz_uniform_spmv(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha,
    float x_old_uniform,
    float beta,
    const float* __restrict__ betas,
    int32_t num_vertices,
    float* __restrict__ diff_sum
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0.0f;
    if (tid < num_vertices) {
        int32_t v = tid;
        int32_t row_start = offsets[v];
        int32_t row_end = offsets[v + 1];
        int32_t active_count = 0;
        int32_t j = row_start;

        
        int32_t first_word = j >> 5;
        int32_t last_word = (row_end - 1) >> 5;
        int32_t first_bit = j & 31;

        if (row_end > row_start) {
            if (first_word == last_word) {
                
                uint32_t word = edge_mask[first_word];
                int32_t last_bit = (row_end - 1) & 31;
                uint32_t mask = word >> first_bit;
                int32_t nbits = last_bit - first_bit + 1;
                if (nbits < 32) mask &= (1u << nbits) - 1;
                active_count = __popc(mask);
            } else {
                
                uint32_t word = edge_mask[first_word];
                active_count += __popc(word >> first_bit);
                
                for (int32_t w = first_word + 1; w < last_word; w++) {
                    active_count += __popc(edge_mask[w]);
                }
                
                int32_t last_bit = (row_end - 1) & 31;
                uint32_t last_w = edge_mask[last_word];
                if (last_bit < 31) {
                    last_w &= (2u << last_bit) - 1;
                }
                active_count += __popc(last_w);
            }
        }

        float beta_v = betas ? betas[v] : beta;
        float new_val = alpha * (float)active_count * x_old_uniform + beta_v;
        x_new[v] = new_val;
        diff = fabsf(new_val - x_old[v]);
    }
    float block_diff = block_reduce_sum<256>(diff);
    if (threadIdx.x == 0 && block_diff > 0.0f) atomicAdd(diff_sum, block_diff);
}





__global__ void __launch_bounds__(256, 8)
katz_spmv_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha,
    float beta,
    const float* __restrict__ betas,
    int32_t start_vertex,
    int32_t num_vertices,
    float* __restrict__ diff_sum
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0.0f;
    if (tid < num_vertices) {
        int32_t v = start_vertex + tid;
        int32_t row_start = offsets[v];
        int32_t row_end = offsets[v + 1];

        float sum = 0.0f;
        for (int32_t j = row_start; j < row_end; j++) {
            if ((edge_mask[j >> 5] >> (j & 31)) & 1u) {
                sum += x_old[indices[j]];
            }
        }

        float beta_v = betas ? betas[v] : beta;
        float new_val = alpha * sum + beta_v;
        x_new[v] = new_val;
        diff = fabsf(new_val - x_old[v]);
    }
    float block_diff = block_reduce_sum<256>(diff);
    if (threadIdx.x == 0 && block_diff > 0.0f) atomicAdd(diff_sum, block_diff);
}




__global__ void __launch_bounds__(256, 8)
katz_spmv_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha,
    float beta,
    const float* __restrict__ betas,
    int32_t start_vertex,
    int32_t num_vertices,
    float* __restrict__ diff_sum
) {
    int32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t warp_id = global_tid >> 5;
    int32_t lane = global_tid & 31;
    float diff = 0.0f;

    if (warp_id < num_vertices) {
        int32_t v = start_vertex + warp_id;
        int32_t row_start = offsets[v];
        int32_t row_end = offsets[v + 1];

        float local_sum = 0.0f;
        for (int32_t j = row_start + lane; j < row_end; j += 32) {
            
            int32_t idx = __ldcs(&indices[j]);
            uint32_t mask_word = edge_mask[j >> 5];
            if ((mask_word >> (j & 31)) & 1u) {
                local_sum += x_old[idx];
            }
        }

        local_sum = warp_reduce_sum(local_sum);

        if (lane == 0) {
            float beta_v = betas ? betas[v] : beta;
            float new_val = alpha * local_sum + beta_v;
            x_new[v] = new_val;
            diff = fabsf(new_val - x_old[v]);
        }
    }

    float block_diff = block_reduce_sum<256>(diff);
    if (threadIdx.x == 0 && block_diff > 0.0f) atomicAdd(diff_sum, block_diff);
}




__global__ void __launch_bounds__(256)
katz_spmv_block(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha,
    float beta,
    const float* __restrict__ betas,
    int32_t start_vertex,
    int32_t num_vertices,
    float* __restrict__ diff_sum
) {
    if (blockIdx.x >= (unsigned)num_vertices) return;
    int32_t v = start_vertex + blockIdx.x;
    int32_t row_start = offsets[v];
    int32_t row_end = offsets[v + 1];
    int32_t tid = threadIdx.x;

    float local_sum = 0.0f;
    for (int32_t j = row_start + tid; j < row_end; j += 256) {
        int32_t idx = __ldcs(&indices[j]);
        uint32_t mask_word = edge_mask[j >> 5];
        if ((mask_word >> (j & 31)) & 1u) {
            local_sum += x_old[idx];
        }
    }

    float block_sum = block_reduce_sum<256>(local_sum);
    if (tid == 0) {
        float beta_v = betas ? betas[v] : beta;
        float new_val = alpha * block_sum + beta_v;
        x_new[v] = new_val;
        atomicAdd(diff_sum, fabsf(new_val - x_old[v]));
    }
}




__global__ void compute_norm_sq(const float* __restrict__ x, int32_t n, float* __restrict__ norm_sq) {
    float local_sum = 0.0f;
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;
    for (int32_t i = tid; i < n; i += stride) {
        float v = x[i];
        local_sum += v * v;
    }
    float block_sum = block_reduce_sum<256>(local_sum);
    if (threadIdx.x == 0 && block_sum > 0.0f) atomicAdd(norm_sq, block_sum);
}

__global__ void scale_vector(float* __restrict__ x, int32_t n, float scale) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) x[tid] *= scale;
}




struct Cache : Cacheable {
    float* buf0 = nullptr;
    float* buf1 = nullptr;
    float* diff_sum = nullptr;
    float* norm_sq = nullptr;
    float* h_pinned = nullptr;
    int64_t buf0_capacity = 0;
    int64_t buf1_capacity = 0;

    void ensure(int32_t num_vertices) {
        int64_t n = static_cast<int64_t>(num_vertices);
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
        if (!diff_sum) cudaMalloc(&diff_sum, sizeof(float));
        if (!norm_sq) cudaMalloc(&norm_sq, sizeof(float));
        if (!h_pinned) cudaMallocHost(&h_pinned, sizeof(float) * 2);
    }

    ~Cache() override {
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (diff_sum) cudaFree(diff_sum);
        if (norm_sq) cudaFree(norm_sq);
        if (h_pinned) cudaFreeHost(h_pinned);
    }
};




void do_full_spmv(
    const int32_t* d_offsets, const int32_t* d_indices, const uint32_t* d_edge_mask,
    const float* d_x_old, float* d_x_new,
    float alpha, float beta, const float* d_betas,
    int32_t seg0, int32_t seg1, int32_t seg2,
    int32_t n_large, int32_t n_medium, int32_t n_small_zero,
    float* d_diff_sum, cudaStream_t stream
) {
    if (n_large > 0)
        katz_spmv_block<<<n_large, 256, 0, stream>>>(
            d_offsets, d_indices, d_edge_mask, d_x_old, d_x_new,
            alpha, beta, d_betas, seg0, n_large, d_diff_sum);
    if (n_medium > 0)
        katz_spmv_warp<<<(n_medium + 7) / 8, 256, 0, stream>>>(
            d_offsets, d_indices, d_edge_mask, d_x_old, d_x_new,
            alpha, beta, d_betas, seg1, n_medium, d_diff_sum);
    if (n_small_zero > 0)
        katz_spmv_thread<<<(n_small_zero + 255) / 256, 256, 0, stream>>>(
            d_offsets, d_indices, d_edge_mask, d_x_old, d_x_new,
            alpha, beta, d_betas, seg2, n_small_zero, d_diff_sum);
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
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg4 = seg[4];
    int32_t n_large = seg1 - seg0, n_medium = seg2 - seg1, n_small_zero = seg4 - seg2;

    cache.ensure(num_vertices);

    float* buf0 = cache.buf0;
    float* buf1 = cache.buf1;
    float* d_diff_sum = cache.diff_sum;
    float* h_pinned = cache.h_pinned;
    cudaStream_t stream = 0;

    float* d_x_old = buf0;
    float* d_x_new = buf1;

    std::size_t iterations = 0;
    bool converged = false;

    if (!has_initial_guess && max_iterations > 0) {
        
        cudaMemsetAsync(buf0, 0, num_vertices * sizeof(float), stream);
        cudaMemsetAsync(d_diff_sum, 0, sizeof(float), stream);
        katz_fill_beta<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
            d_x_new, beta, betas, num_vertices, d_diff_sum);

        cudaMemcpyAsync(h_pinned, d_diff_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::swap(d_x_old, d_x_new);
        iterations = 1;

        if (h_pinned[0] < epsilon) { converged = true; }

        
        if (!converged && betas == nullptr && iterations < max_iterations) {
            cudaMemsetAsync(d_diff_sum, 0, sizeof(float), stream);
            katz_uniform_spmv<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
                d_offsets, d_edge_mask, d_x_old, d_x_new,
                alpha, beta, beta, betas, num_vertices, d_diff_sum);

            cudaMemcpyAsync(h_pinned, d_diff_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            std::swap(d_x_old, d_x_new);
            iterations = 2;
            if (h_pinned[0] < epsilon) { converged = true; }
        }
    } else if (has_initial_guess) {
        cudaMemcpyAsync(buf0, centralities,
                   num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemsetAsync(buf0, 0, num_vertices * sizeof(float), stream);
    }

    
    for (std::size_t iter = iterations; iter < max_iterations && !converged; iter++) {
        cudaMemsetAsync(d_diff_sum, 0, sizeof(float), stream);

        do_full_spmv(d_offsets, d_indices, d_edge_mask, d_x_old, d_x_new,
                     alpha, beta, betas, seg0, seg1, seg2,
                     n_large, n_medium, n_small_zero, d_diff_sum, stream);

        cudaMemcpyAsync(h_pinned, d_diff_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::swap(d_x_old, d_x_new);
        iterations = iter + 1;
        if (h_pinned[0] < epsilon) { converged = true; }
    }

    
    if (normalize) {
        cudaMemcpyAsync(centralities, d_x_old,
                   num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        float* d_norm_sq = cache.norm_sq;
        cudaMemsetAsync(d_norm_sq, 0, sizeof(float), stream);
        int grid = (num_vertices + 255) / 256;
        if (grid > 1024) grid = 1024;
        compute_norm_sq<<<grid, 256, 0, stream>>>(centralities, num_vertices, d_norm_sq);

        cudaMemcpyAsync(h_pinned + 1, d_norm_sq, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        float norm = sqrtf(h_pinned[1]);
        if (norm > 0.0f) {
            scale_vector<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
                centralities, num_vertices, 1.0f / norm);
            cudaStreamSynchronize(stream);
        }
    } else {
        cudaMemcpyAsync(centralities, d_x_old,
                   num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    return {iterations, converged};
}

}  
