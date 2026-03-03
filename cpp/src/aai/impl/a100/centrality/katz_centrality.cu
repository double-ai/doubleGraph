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

constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;

struct Cache : Cacheable {
    float* scratch = nullptr;
    float* diff = nullptr;
    char* h_pinned = nullptr;
    int64_t scratch_capacity = 0;
    int64_t diff_capacity = 0;

    void ensure(int32_t N) {
        if (scratch_capacity < N) {
            if (scratch) cudaFree(scratch);
            cudaMalloc(&scratch, (size_t)N * sizeof(float));
            scratch_capacity = N;
        }
        if (diff_capacity < 1) {
            if (diff) cudaFree(diff);
            cudaMalloc(&diff, sizeof(float));
            diff_capacity = 1;
        }
        if (!h_pinned) {
            cudaMallocHost(&h_pinned, 128);
        }
    }

    ~Cache() override {
        if (scratch) cudaFree(scratch);
        if (diff) cudaFree(diff);
        if (h_pinned) cudaFreeHost(h_pinned);
    }
};


template <bool HAS_BETAS>
__global__ void __launch_bounds__(BLOCK_SIZE)
katz_spmv_diff_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float alpha,
    const float beta,
    const float* __restrict__ betas,
    const int32_t N,
    float* __restrict__ diff_out
) {
    const int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int total_warps = gridDim.x * WARPS_PER_BLOCK;
    const int warp_in_block = threadIdx.x >> 5;

    __shared__ float warp_diffs[WARPS_PER_BLOCK];
    float my_diff = 0.0f;

    for (int32_t v = warp_id; v < N; v += total_warps) {
        const int32_t start = offsets[v];
        const int32_t end = offsets[v + 1];

        float sum = 0.0f;
        for (int32_t j = start + lane; j < end; j += 32) {
            sum += x_old[indices[j]];
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            float bv;
            if constexpr (HAS_BETAS) { bv = betas[v]; }
            else { bv = beta; }
            float val = alpha * sum + bv;
            x_new[v] = val;
            my_diff += fabsf(val - x_old[v]);
        }
    }

    
    if (lane == 0) warp_diffs[warp_in_block] = my_diff;
    __syncthreads();

    if (warp_in_block == 0) {
        float val = (lane < WARPS_PER_BLOCK) ? warp_diffs[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0 && val > 0.0f) {
            atomicAdd(diff_out, val);
        }
    }
}


template <bool HAS_BETAS>
__global__ void __launch_bounds__(BLOCK_SIZE)
katz_spmv_diff_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float alpha,
    const float beta,
    const float* __restrict__ betas,
    const int32_t N,
    float* __restrict__ diff_out
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float my_diff = 0.0f;

    for (int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x; v < N;
         v += gridDim.x * BLOCK_SIZE) {
        float old_val = x_old[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        float sum = 0.0f;
        for (int32_t j = start; j < end; j++) {
            sum += x_old[indices[j]];
        }
        float bv;
        if constexpr (HAS_BETAS) { bv = betas[v]; }
        else { bv = beta; }
        float val = alpha * sum + bv;
        x_new[v] = val;
        my_diff += fabsf(val - old_val);
    }

    float block_diff = BlockReduce(temp).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff > 0.0f) {
        atomicAdd(diff_out, block_diff);
    }
}



__global__ void __launch_bounds__(BLOCK_SIZE)
init_skip2_kernel(
    const int32_t* __restrict__ offsets,
    float* __restrict__ x,
    const float alpha,
    const float beta,
    const int32_t N
) {
    for (int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x; v < N;
         v += gridDim.x * BLOCK_SIZE) {
        int32_t deg = offsets[v + 1] - offsets[v];
        x[v] = alpha * beta * (float)deg + beta;
    }
}


__global__ void __launch_bounds__(BLOCK_SIZE)
init_copy_betas_kernel(
    const float* __restrict__ betas,
    float* __restrict__ x,
    const int32_t N
) {
    for (int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x; v < N;
         v += gridDim.x * BLOCK_SIZE) {
        x[v] = betas[v];
    }
}


__global__ void __launch_bounds__(BLOCK_SIZE)
l2_norm_sq_kernel(const float* __restrict__ x, const int32_t N, float* __restrict__ out) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    float s = 0.0f;
    for (int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < N;
         i += gridDim.x * BLOCK_SIZE) {
        float v = x[i]; s += v * v;
    }
    float bs = BlockReduce(temp).Sum(s);
    if (threadIdx.x == 0) atomicAdd(out, bs);
}

__global__ void __launch_bounds__(BLOCK_SIZE)
scale_vec_kernel(float* __restrict__ x, const int32_t N, const float scale) {
    for (int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < N;
         i += gridDim.x * BLOCK_SIZE) {
        x[i] *= scale;
    }
}

void launch_katz_spmv_diff(
    const int32_t* offsets, const int32_t* indices,
    const float* x_old, float* x_new,
    float alpha, float beta, const float* betas, bool has_betas,
    int32_t N, int32_t num_edges, float* diff_out, cudaStream_t stream)
{
    float avg_degree = (N > 0) ? (float)num_edges / (float)N : 0.0f;

    if (avg_degree >= 8.0f) {
        int total_warps_needed = N;
        int nb = (total_warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        if (nb > 2048) nb = 2048;
        if (nb < 1) nb = 1;
        if (has_betas) {
            katz_spmv_diff_warp_kernel<true><<<nb, BLOCK_SIZE, 0, stream>>>(
                offsets, indices, x_old, x_new, alpha, beta, betas, N, diff_out);
        } else {
            katz_spmv_diff_warp_kernel<false><<<nb, BLOCK_SIZE, 0, stream>>>(
                offsets, indices, x_old, x_new, alpha, beta, betas, N, diff_out);
        }
    } else {
        int nb = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (nb > 2048) nb = 2048;
        if (nb < 1) nb = 1;
        if (has_betas) {
            katz_spmv_diff_thread_kernel<true><<<nb, BLOCK_SIZE, 0, stream>>>(
                offsets, indices, x_old, x_new, alpha, beta, betas, N, diff_out);
        } else {
            katz_spmv_diff_thread_kernel<false><<<nb, BLOCK_SIZE, 0, stream>>>(
                offsets, indices, x_old, x_new, alpha, beta, betas, N, diff_out);
        }
    }
}

void launch_init_skip2(
    const int32_t* offsets, float* x,
    float alpha, float beta, int32_t N, cudaStream_t stream)
{
    int nb = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (nb > 2048) nb = 2048;
    init_skip2_kernel<<<nb, BLOCK_SIZE, 0, stream>>>(offsets, x, alpha, beta, N);
}

void launch_init_copy_betas(
    const float* betas, float* x, int32_t N, cudaStream_t stream)
{
    int nb = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (nb > 2048) nb = 2048;
    init_copy_betas_kernel<<<nb, BLOCK_SIZE, 0, stream>>>(betas, x, N);
}

void launch_l2_norm_sq(const float* x, int32_t N, float* out, cudaStream_t stream) {
    int nb = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (nb > 2048) nb = 2048;
    if (nb < 1) nb = 1;
    l2_norm_sq_kernel<<<nb, BLOCK_SIZE, 0, stream>>>(x, N, out);
}

void launch_scale_vec(float* x, int32_t N, float scale, cudaStream_t stream) {
    int nb = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (nb > 2048) nb = 2048;
    if (nb < 1) nb = 1;
    scale_vec_kernel<<<nb, BLOCK_SIZE, 0, stream>>>(x, N, scale);
}

}  

katz_centrality_result_t katz_centrality(const graph32_t& graph,
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

    int32_t N = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    bool use_betas = (betas != nullptr);

    cache.ensure(N);

    float* d_out = centralities;
    float* d_scr = cache.scratch;
    float* d_diff = cache.diff;
    float* h_diff = (float*)(cache.h_pinned + 64);

    float* x_old = d_out;
    float* x_new = d_scr;
    bool converged = false;
    size_t iters = 0;

    if (has_initial_guess) {
        
    } else if (!use_betas && max_iterations >= 2) {
        
        float diff2 = alpha * fabsf(beta) * (float)num_edges;
        if (diff2 < epsilon) {
            launch_init_skip2(d_offsets, d_out, alpha, beta, N, 0);
            converged = true;
            iters = 2;
        } else {
            launch_init_skip2(d_offsets, d_out, alpha, beta, N, 0);
            iters = 2;
        }
    } else if (use_betas && !has_initial_guess && max_iterations >= 1) {
        
        launch_init_copy_betas(betas, d_out, N, 0);
        iters = 1;
    } else {
        cudaMemsetAsync(d_out, 0, (size_t)N * sizeof(float), 0);
    }

    
    if (!converged) {
        for (size_t i = iters; i < max_iterations; i++) {
            cudaMemsetAsync(d_diff, 0, sizeof(float), 0);

            launch_katz_spmv_diff(d_offsets, d_indices, x_old, x_new,
                                  alpha, beta, betas, use_betas,
                                  N, num_edges, d_diff, 0);

            cudaMemcpyAsync(h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost, 0);
            cudaStreamSynchronize(0);

            iters = i + 1;
            float* tmp = x_old; x_old = x_new; x_new = tmp;

            if (*h_diff < epsilon) {
                converged = true;
                break;
            }
        }
    }

    
    if (x_old != d_out) {
        cudaMemcpyAsync(d_out, x_old, (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice, 0);
    }

    
    if (normalize) {
        cudaMemsetAsync(d_diff, 0, sizeof(float), 0);
        launch_l2_norm_sq(d_out, N, d_diff, 0);
        cudaMemcpyAsync(h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost, 0);
        cudaStreamSynchronize(0);

        if (*h_diff > 0.0f) {
            float inv_norm = 1.0f / sqrtf(*h_diff);
            launch_scale_vec(d_out, N, inv_norm, 0);
        }
    }

    cudaStreamSynchronize(0);

    return {iters, converged};
}

}  
