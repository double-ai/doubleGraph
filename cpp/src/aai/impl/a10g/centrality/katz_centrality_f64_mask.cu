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
#include <algorithm>
#include <cub/cub.cuh>

namespace aai {

namespace {

struct Cache : Cacheable {
    double* d_l1_diff = nullptr;
    double* d_l2_result = nullptr;
    float* h_pinned = nullptr;
    double* d_x0 = nullptr;
    double* d_x1 = nullptr;
    int64_t x_capacity = 0;

    Cache() {
        cudaMalloc(&d_l1_diff, sizeof(double));
        cudaMalloc(&d_l2_result, sizeof(double));
        cudaMallocHost(&h_pinned, 256);
    }

    ~Cache() override {
        if (d_l1_diff) cudaFree(d_l1_diff);
        if (d_l2_result) cudaFree(d_l2_result);
        if (h_pinned) cudaFreeHost(h_pinned);
        if (d_x0) cudaFree(d_x0);
        if (d_x1) cudaFree(d_x1);
    }

    void ensure(int64_t num_vertices) {
        if (x_capacity < num_vertices) {
            if (d_x0) cudaFree(d_x0);
            if (d_x1) cudaFree(d_x1);
            cudaMalloc(&d_x0, num_vertices * sizeof(double));
            cudaMalloc(&d_x1, num_vertices * sizeof(double));
            x_capacity = num_vertices;
        }
    }
};





__global__ void fill_d(double* __restrict__ dst, double val, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dst[i] = val;
    }
}

template <bool USE_BETAS>
__global__ __launch_bounds__(256)
void katz_spmv_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ x_old,
    double* __restrict__ x_new,
    const double* __restrict__ betas,
    double alpha,
    double beta_scalar,
    int num_vertices,
    double* __restrict__ d_l1_diff
) {
    double thread_diff = 0.0;

    for (int v = blockIdx.x * blockDim.x + threadIdx.x;
         v < num_vertices;
         v += gridDim.x * blockDim.x) {

        int start = offsets[v];
        int end = offsets[v + 1];

        double sum = 0.0;

        for (int e = start; e < end; e++) {
            uint32_t mask_word = edge_mask[e >> 5];
            bool active = (mask_word >> (e & 31)) & 1;

            if (active) {
                sum += weights[e] * __ldg(&x_old[indices[e]]);
            }
        }

        double new_val = alpha * sum;
        if constexpr (USE_BETAS) {
            new_val += betas[v];
        } else {
            new_val += beta_scalar;
        }
        x_new[v] = new_val;
        thread_diff += fabs(new_val - x_old[v]);
    }

    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    double block_sum = BlockReduce(temp).Sum(thread_diff);
    if (threadIdx.x == 0 && block_sum > 0.0) {
        atomicAdd(d_l1_diff, block_sum);
    }
}

template <bool USE_BETAS>
__global__ __launch_bounds__(256)
void katz_spmv_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ x_old,
    double* __restrict__ x_new,
    const double* __restrict__ betas,
    double alpha,
    double beta_scalar,
    int num_vertices,
    double* __restrict__ d_l1_diff
) {
    constexpr int WARP_SIZE = 32;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int warp_id_in_block = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    double thread_diff = 0.0;

    for (int v = blockIdx.x * warps_per_block + warp_id_in_block;
         v < num_vertices;
         v += gridDim.x * warps_per_block) {

        int start = offsets[v];
        int end = offsets[v + 1];

        double sum = 0.0;

        for (int e = start + lane; e < end; e += WARP_SIZE) {
            uint32_t mask_word = edge_mask[e >> 5];
            bool active = (mask_word >> (e & 31)) & 1;

            if (active) {
                sum += weights[e] * __ldg(&x_old[indices[e]]);
            }
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            double new_val = alpha * sum;
            if constexpr (USE_BETAS) {
                new_val += betas[v];
            } else {
                new_val += beta_scalar;
            }
            x_new[v] = new_val;
            thread_diff += fabs(new_val - x_old[v]);
        }
    }

    __shared__ double s_diff[32];
    if (lane == 0) {
        s_diff[warp_id_in_block] = thread_diff;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        double val = (threadIdx.x < warps_per_block) ? s_diff[threadIdx.x] : 0.0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (threadIdx.x == 0 && val > 0.0) {
            atomicAdd(d_l1_diff, val);
        }
    }
}

__global__ void l2_norm_sq_kernel(const double* __restrict__ x, double* __restrict__ result, int n) {
    double sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        double val = x[i];
        sum += val * val;
    }
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    double block_sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) {
        atomicAdd(result, block_sum);
    }
}

__global__ void normalize_kernel(double* __restrict__ x, double inv_norm, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        x[i] *= inv_norm;
    }
}





void launch_fill_d(double* dst, double val, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 65536) grid = 65536;
    fill_d<<<grid, block, 0, stream>>>(dst, val, n);
}

void launch_katz_spmv_thread(
    const int32_t* offsets, const int32_t* indices,
    const double* weights, const uint32_t* edge_mask,
    const double* x_old, double* x_new,
    const double* betas, double alpha, double beta_scalar,
    bool use_betas, int num_vertices, double* d_l1_diff,
    int grid_size, cudaStream_t stream
) {
    if (use_betas) {
        katz_spmv_thread_kernel<true><<<grid_size, 256, 0, stream>>>(
            offsets, indices, weights, edge_mask,
            x_old, x_new, betas, alpha, beta_scalar,
            num_vertices, d_l1_diff);
    } else {
        katz_spmv_thread_kernel<false><<<grid_size, 256, 0, stream>>>(
            offsets, indices, weights, edge_mask,
            x_old, x_new, betas, alpha, beta_scalar,
            num_vertices, d_l1_diff);
    }
}

void launch_katz_spmv_warp(
    const int32_t* offsets, const int32_t* indices,
    const double* weights, const uint32_t* edge_mask,
    const double* x_old, double* x_new,
    const double* betas, double alpha, double beta_scalar,
    bool use_betas, int num_vertices, double* d_l1_diff,
    int grid_size, cudaStream_t stream
) {
    if (use_betas) {
        katz_spmv_warp_kernel<true><<<grid_size, 256, 0, stream>>>(
            offsets, indices, weights, edge_mask,
            x_old, x_new, betas, alpha, beta_scalar,
            num_vertices, d_l1_diff);
    } else {
        katz_spmv_warp_kernel<false><<<grid_size, 256, 0, stream>>>(
            offsets, indices, weights, edge_mask,
            x_old, x_new, betas, alpha, beta_scalar,
            num_vertices, d_l1_diff);
    }
}

void launch_l2_norm_sq(const double* x, double* result, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 4096) grid = 4096;
    l2_norm_sq_kernel<<<grid, block, 0, stream>>>(x, result, n);
}

void launch_normalize(double* x, double inv_norm, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 65536) grid = 65536;
    normalize_kernel<<<grid, block, 0, stream>>>(x, inv_norm, n);
}

}  

katz_centrality_result_t katz_centrality_mask(const graph32_t& graph,
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

    cudaStream_t stream = 0;

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    cache.ensure(num_vertices);

    double* d_x_old = cache.d_x0;
    double* d_x_new = cache.d_x1;

    bool use_betas = (betas != nullptr);

    
    std::size_t iter_start = 0;
    if (!has_initial_guess && max_iterations > 0) {
        if (use_betas) {
            cudaMemcpyAsync(d_x_old, betas, num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
        } else {
            launch_fill_d(d_x_old, beta, num_vertices, stream);
        }
        iter_start = 1;
    } else if (has_initial_guess) {
        cudaMemcpyAsync(d_x_old, centralities, num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemsetAsync(d_x_old, 0, num_vertices * sizeof(double), stream);
    }

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const double* d_weights = edge_weights;
    const uint32_t* d_edge_mask = graph.edge_mask;

    float avg_degree = (num_vertices > 0) ? (float)num_edges / num_vertices : 0.0f;
    bool use_warp = (avg_degree >= 4.0f);

    int grid_size;
    if (use_warp) {
        int warps_per_block = 8;
        int max_blocks = (num_vertices + warps_per_block - 1) / warps_per_block;
        grid_size = std::min(max_blocks, 40960);
    } else {
        int max_blocks = (num_vertices + 255) / 256;
        grid_size = std::min(max_blocks, 65536);
    }

    std::size_t iterations = 0;
    bool converged = false;
    double* h_l1 = (double*)cache.h_pinned;

    for (std::size_t iter = iter_start; iter < max_iterations; iter++) {
        cudaMemsetAsync(cache.d_l1_diff, 0, sizeof(double), stream);

        if (use_warp) {
            launch_katz_spmv_warp(
                d_offsets, d_indices, d_weights, d_edge_mask,
                d_x_old, d_x_new, betas,
                alpha, beta, use_betas,
                num_vertices, cache.d_l1_diff,
                grid_size, stream);
        } else {
            launch_katz_spmv_thread(
                d_offsets, d_indices, d_weights, d_edge_mask,
                d_x_old, d_x_new, betas,
                alpha, beta, use_betas,
                num_vertices, cache.d_l1_diff,
                grid_size, stream);
        }

        cudaMemcpyAsync(h_l1, cache.d_l1_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations = iter + 1;

        double* temp = d_x_old;
        d_x_old = d_x_new;
        d_x_new = temp;

        if (*h_l1 < epsilon) {
            converged = true;
            break;
        }
    }

    if (iterations == 0 && iter_start > 0) {
        iterations = iter_start;
    }

    
    if (normalize) {
        cudaMemsetAsync(cache.d_l2_result, 0, sizeof(double), stream);
        launch_l2_norm_sq(d_x_old, cache.d_l2_result, num_vertices, stream);
        double* h_l2 = (double*)cache.h_pinned;
        cudaMemcpyAsync(h_l2, cache.d_l2_result, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        double l2_norm = sqrt(*h_l2);
        if (l2_norm > 0.0) {
            launch_normalize(d_x_old, 1.0 / l2_norm, num_vertices, stream);
        }
    }

    
    cudaMemcpyAsync(centralities, d_x_old, num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);

    return katz_centrality_result_t{iterations, converged};
}

}  
