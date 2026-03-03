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

namespace aai {

namespace {

constexpr int BLOCK_SIZE = 256;





struct Cache : Cacheable {
    double* buf_a = nullptr;
    double* buf_b = nullptr;
    double* scratch = nullptr;
    int64_t buf_a_capacity = 0;
    int64_t buf_b_capacity = 0;
    bool scratch_allocated = false;

    void ensure(int32_t num_vertices) {
        int64_t n = num_vertices;
        if (buf_a_capacity < n) {
            if (buf_a) cudaFree(buf_a);
            cudaMalloc(&buf_a, n * sizeof(double));
            buf_a_capacity = n;
        }
        if (buf_b_capacity < n) {
            if (buf_b) cudaFree(buf_b);
            cudaMalloc(&buf_b, n * sizeof(double));
            buf_b_capacity = n;
        }
        if (!scratch_allocated) {
            cudaMalloc(&scratch, 2 * sizeof(double));
            scratch_allocated = true;
        }
    }

    ~Cache() override {
        if (buf_a) cudaFree(buf_a);
        if (buf_b) cudaFree(buf_b);
        if (scratch) cudaFree(scratch);
    }
};





__global__ void fill_double(double* __restrict__ x, double val, int32_t n) {
    for (int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        x[i] = val;
    }
}

__global__ void add_beta_kernel(
    double* __restrict__ x_new,
    const double* __restrict__ betas,
    double beta_scalar,
    bool use_betas,
    int32_t n
) {
    for (int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        x_new[i] += use_betas ? betas[i] : beta_scalar;
    }
}


template <bool COMPUTE_DIFF>
__global__ void katz_spmv_block(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ x_old,
    double* __restrict__ x_new,
    const double* __restrict__ betas,
    double alpha,
    double beta_scalar,
    bool use_betas,
    int32_t v_start,
    int32_t v_end,
    double* __restrict__ d_diff
) {
    int32_t v = v_start + blockIdx.x;
    if (v >= v_end) return;

    int32_t start = offsets[v];
    int32_t end_idx = offsets[v + 1];
    double sum = 0.0;

    for (int32_t j = start + threadIdx.x; j < end_idx; j += BLOCK_SIZE) {
        if ((edge_mask[j >> 5] >> (j & 31)) & 1u) {
            sum += weights[j] * x_old[indices[j]];
        }
    }

    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    double total = BlockReduce(temp).Sum(sum);

    if (threadIdx.x == 0) {
        double beta_v = use_betas ? betas[v] : beta_scalar;
        double new_val = alpha * total + beta_v;
        x_new[v] = new_val;
        if constexpr (COMPUTE_DIFF) {
            atomicAdd(d_diff, fabs(new_val - x_old[v]));
        }
    }
}


template <bool COMPUTE_DIFF>
__global__ void katz_spmv_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ x_old,
    double* __restrict__ x_new,
    const double* __restrict__ betas,
    double alpha,
    double beta_scalar,
    bool use_betas,
    int32_t v_start,
    int32_t v_end,
    double* __restrict__ d_diff
) {
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;
    int32_t v = v_start + warp_id;

    double thread_diff = 0.0;

    if (v < v_end) {
        int32_t start = offsets[v];
        int32_t end_idx = offsets[v + 1];
        double sum = 0.0;

        for (int32_t j = start + lane; j < end_idx; j += 32) {
            if ((edge_mask[j >> 5] >> (j & 31)) & 1u) {
                sum += weights[j] * x_old[indices[j]];
            }
        }

        
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, off);
        }

        if (lane == 0) {
            double beta_v = use_betas ? betas[v] : beta_scalar;
            double new_val = alpha * sum + beta_v;
            x_new[v] = new_val;
            if constexpr (COMPUTE_DIFF) {
                thread_diff = fabs(new_val - x_old[v]);
            }
        }
    }

    if constexpr (COMPUTE_DIFF) {
        typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp;
        double block_diff = BlockReduce(temp).Sum(thread_diff);
        if (threadIdx.x == 0 && block_diff > 0.0) {
            atomicAdd(d_diff, block_diff);
        }
    }
}


template <bool COMPUTE_DIFF>
__global__ void katz_spmv_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ x_old,
    double* __restrict__ x_new,
    const double* __restrict__ betas,
    double alpha,
    double beta_scalar,
    bool use_betas,
    int32_t v_start,
    int32_t v_end,
    double* __restrict__ d_diff
) {
    int32_t v = v_start + blockIdx.x * blockDim.x + threadIdx.x;
    double thread_diff = 0.0;

    if (v < v_end) {
        int32_t start = offsets[v];
        int32_t end_idx = offsets[v + 1];
        double sum = 0.0;

        for (int32_t j = start; j < end_idx; j++) {
            if ((edge_mask[j >> 5] >> (j & 31)) & 1u) {
                sum += weights[j] * x_old[indices[j]];
            }
        }

        double beta_v = use_betas ? betas[v] : beta_scalar;
        double new_val = alpha * sum + beta_v;
        x_new[v] = new_val;
        if constexpr (COMPUTE_DIFF) {
            thread_diff = fabs(new_val - x_old[v]);
        }
    }

    if constexpr (COMPUTE_DIFF) {
        typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp;
        double block_diff = BlockReduce(temp).Sum(thread_diff);
        if (threadIdx.x == 0 && block_diff > 0.0) {
            atomicAdd(d_diff, block_diff);
        }
    }
}





__global__ void l2_norm_sq_kernel(
    const double* __restrict__ x, int32_t n, double* __restrict__ out
) {
    double local_sum = 0.0;
    for (int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        double v = x[i];
        local_sum += v * v;
    }
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    double block_sum = BlockReduce(temp).Sum(local_sum);
    if (threadIdx.x == 0 && block_sum > 0.0) {
        atomicAdd(out, block_sum);
    }
}

__global__ void scale_kernel(
    double* __restrict__ x, int32_t n, const double* __restrict__ norm_sq
) {
    double inv = rsqrt(*norm_sq);
    for (int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        x[i] *= inv;
    }
}





template <bool COMPUTE_DIFF>
void launch_katz_iteration_impl(
    const int32_t* offsets,
    const int32_t* indices,
    const double* weights,
    const uint32_t* edge_mask,
    const double* x_old,
    double* x_new,
    const double* betas,
    double alpha,
    double beta_scalar,
    bool use_betas,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3, int32_t seg4,
    double* d_diff
) {
    if constexpr (COMPUTE_DIFF) {
        cudaMemsetAsync(d_diff, 0, sizeof(double));
    }

    
    int32_t n_high = seg1 - seg0;
    if (n_high > 0) {
        katz_spmv_block<COMPUTE_DIFF><<<n_high, BLOCK_SIZE>>>(
            offsets, indices, weights, edge_mask,
            x_old, x_new, betas, alpha, beta_scalar, use_betas,
            seg0, seg1, d_diff);
    }

    
    int32_t n_mid = seg2 - seg1;
    if (n_mid > 0) {
        int warps_per_block = BLOCK_SIZE / 32;
        int blocks = (n_mid + warps_per_block - 1) / warps_per_block;
        katz_spmv_warp<COMPUTE_DIFF><<<blocks, BLOCK_SIZE>>>(
            offsets, indices, weights, edge_mask,
            x_old, x_new, betas, alpha, beta_scalar, use_betas,
            seg1, seg2, d_diff);
    }

    
    int32_t n_low = seg4 - seg2;
    if (n_low > 0) {
        int blocks = (n_low + BLOCK_SIZE - 1) / BLOCK_SIZE;
        katz_spmv_thread<COMPUTE_DIFF><<<blocks, BLOCK_SIZE>>>(
            offsets, indices, weights, edge_mask,
            x_old, x_new, betas, alpha, beta_scalar, use_betas,
            seg2, seg4, d_diff);
    }
}

void launch_katz_iteration(
    const int32_t* offsets, const int32_t* indices,
    const double* weights, const uint32_t* edge_mask,
    const double* x_old, double* x_new,
    const double* betas, double alpha, double beta_scalar, bool use_betas,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3, int32_t seg4,
    double* d_diff, bool compute_diff
) {
    if (compute_diff) {
        launch_katz_iteration_impl<true>(offsets, indices, weights, edge_mask,
            x_old, x_new, betas, alpha, beta_scalar, use_betas,
            seg0, seg1, seg2, seg3, seg4, d_diff);
    } else {
        launch_katz_iteration_impl<false>(offsets, indices, weights, edge_mask,
            x_old, x_new, betas, alpha, beta_scalar, use_betas,
            seg0, seg1, seg2, seg3, seg4, d_diff);
    }
}

void launch_normalize(double* x, int32_t n, double* d_scratch) {
    cudaMemsetAsync(d_scratch, 0, sizeof(double));
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 1024) blocks = 1024;
    l2_norm_sq_kernel<<<blocks, BLOCK_SIZE>>>(x, n, d_scratch);
    scale_kernel<<<blocks, BLOCK_SIZE>>>(x, n, d_scratch);
}

void launch_fill(double* x, double val, int32_t n) {
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (blocks > 1024) blocks = 1024;
    fill_double<<<blocks, BLOCK_SIZE>>>(x, val, n);
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

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg_arr[5] = {seg[0], seg[1], seg[2], seg[3], seg[4]};

    bool use_betas = (betas != nullptr);

    cache.ensure(num_vertices);

    double* d_a = cache.buf_a;
    double* d_b = cache.buf_b;
    double* d_diff = cache.scratch;
    double* d_norm = cache.scratch + 1;

    double* x_old = d_a;
    double* x_new = d_b;

    std::size_t effective_max = max_iterations;

    if (has_initial_guess) {
        cudaMemcpyAsync(d_a, centralities,
                   (std::size_t)num_vertices * sizeof(double), cudaMemcpyDeviceToDevice);
    } else if (max_iterations > 0) {
        if (use_betas) {
            cudaMemcpyAsync(d_a, betas,
                       (std::size_t)num_vertices * sizeof(double), cudaMemcpyDeviceToDevice);
        } else {
            launch_fill(d_a, beta, num_vertices);
        }
        effective_max = max_iterations - 1;
    } else {
        cudaMemsetAsync(d_a, 0, (std::size_t)num_vertices * sizeof(double));
    }

    bool converged = false;
    std::size_t iter = 0;

    const std::size_t BATCH_SIZE = 4;

    while (iter < effective_max) {
        std::size_t batch_end = iter + BATCH_SIZE;
        if (batch_end > effective_max) batch_end = effective_max;

        
        for (std::size_t i = iter; i < batch_end - 1; i++) {
            launch_katz_iteration(
                d_offsets, d_indices, edge_weights, d_edge_mask,
                x_old, x_new, betas,
                alpha, beta, use_betas,
                seg_arr[0], seg_arr[1], seg_arr[2], seg_arr[3], seg_arr[4],
                d_diff, false);
            double* tmp = x_old; x_old = x_new; x_new = tmp;
        }

        
        launch_katz_iteration(
            d_offsets, d_indices, edge_weights, d_edge_mask,
            x_old, x_new, betas,
            alpha, beta, use_betas,
            seg_arr[0], seg_arr[1], seg_arr[2], seg_arr[3], seg_arr[4],
            d_diff, true);
        double* tmp = x_old; x_old = x_new; x_new = tmp;

        
        double h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);

        iter = batch_end;

        if (h_diff < epsilon) {
            converged = true;
            break;
        }
    }

    
    if (normalize) {
        launch_normalize(x_old, num_vertices, d_norm);
    }

    
    cudaMemcpyAsync(centralities, x_old,
               (std::size_t)num_vertices * sizeof(double), cudaMemcpyDeviceToDevice);

    return katz_centrality_result_t{iter, converged};
}

}  
