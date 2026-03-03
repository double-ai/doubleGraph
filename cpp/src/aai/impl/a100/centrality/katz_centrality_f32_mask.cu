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
#include <cstddef>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* d_l1_diff = nullptr;
    float* d_norm_sq = nullptr;
    float* h_l1_pinned = nullptr;
    float* buf_a = nullptr;
    float* buf_b = nullptr;
    int32_t buf_a_capacity = 0;
    int32_t buf_b_capacity = 0;

    Cache() {
        cudaMalloc(&d_l1_diff, sizeof(float));
        cudaMalloc(&d_norm_sq, sizeof(float));
        cudaMallocHost(&h_l1_pinned, sizeof(float));
    }

    ~Cache() override {
        if (d_l1_diff) cudaFree(d_l1_diff);
        if (d_norm_sq) cudaFree(d_norm_sq);
        if (h_l1_pinned) cudaFreeHost(h_l1_pinned);
        if (buf_a) cudaFree(buf_a);
        if (buf_b) cudaFree(buf_b);
    }

    void ensure(int32_t num_vertices) {
        if (buf_a_capacity < num_vertices) {
            if (buf_a) cudaFree(buf_a);
            cudaMalloc(&buf_a, num_vertices * sizeof(float));
            buf_a_capacity = num_vertices;
        }
        if (buf_b_capacity < num_vertices) {
            if (buf_b) cudaFree(buf_b);
            cudaMalloc(&buf_b, num_vertices * sizeof(float));
            buf_b_capacity = num_vertices;
        }
    }
};


__global__ void katz_spmv_persistent(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha,
    float beta_scalar,
    const float* __restrict__ betas,
    int32_t num_vertices,
    float* __restrict__ d_l1_diff
) {
    extern __shared__ float smem[];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id_global = tid >> 5;
    const int lane = tid & 31;
    const int local_warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int total_warps = gridDim.x * warps_per_block;

    float warp_diff = 0.0f;

    for (int v = warp_id_global; v < num_vertices; v += total_warps) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float sum = 0.0f;

        for (int32_t e = start + lane; e < end; e += 32) {
            uint32_t mask_word = edge_mask[e >> 5];
            if ((mask_word >> (e & 31)) & 1u) {
                sum += edge_weights[e] * x_old[indices[e]];
            }
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            float beta_v = betas ? betas[v] : beta_scalar;
            float new_val = alpha * sum + beta_v;
            x_new[v] = new_val;
            warp_diff += fabsf(new_val - x_old[v]);
        }
    }

    
    if (lane == 0) {
        smem[local_warp] = warp_diff;
    }
    __syncthreads();

    if (local_warp == 0) {
        float val = (lane < warps_per_block) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            atomicAdd(d_l1_diff, val);
        }
    }
}

__global__ void fill_kernel(float* __restrict__ x, float val, int32_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        x[i] = val;
    }
}

__global__ void copy_kernel(float* __restrict__ dst, const float* __restrict__ src, int32_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        dst[i] = src[i];
    }
}

__global__ void l2_norm_sq_kernel(
    const float* __restrict__ x,
    int32_t n,
    float* __restrict__ d_norm_sq
) {
    extern __shared__ float smem[];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        float val = x[i];
        sum += val * val;
    }

    
    smem[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(d_norm_sq, smem[0]);
    }
}

__global__ void scale_kernel(
    float* __restrict__ x,
    int32_t n,
    const float* __restrict__ d_norm_sq
) {
    float inv_norm = rsqrtf(*d_norm_sq);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        x[i] *= inv_norm;
    }
}

static int g_spmv_grid_size = 0;

void launch_katz_spmv(
    const int32_t* offsets,
    const int32_t* indices,
    const float* edge_weights,
    const uint32_t* edge_mask,
    const float* x_old,
    float* x_new,
    float alpha,
    float beta_scalar,
    const float* betas,
    int32_t num_vertices,
    float* d_l1_diff,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int warps_per_block = block_size / 32;
    int smem_size = warps_per_block * sizeof(float);

    if (g_spmv_grid_size == 0) {
        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        int blocks_per_sm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, katz_spmv_persistent, block_size, smem_size);
        g_spmv_grid_size = num_sms * blocks_per_sm;
    }

    int needed_blocks = (num_vertices + warps_per_block - 1) / warps_per_block;
    int grid = (needed_blocks < g_spmv_grid_size) ? needed_blocks : g_spmv_grid_size;

    katz_spmv_persistent<<<grid, block_size, smem_size, stream>>>(
        offsets, indices, edge_weights, edge_mask,
        x_old, x_new, alpha, beta_scalar, betas,
        num_vertices, d_l1_diff
    );
}

void launch_fill(float* x, float val, int32_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 1024) grid = 1024;
    fill_kernel<<<grid, block, 0, stream>>>(x, val, n);
}

void launch_copy(float* dst, const float* src, int32_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 1024) grid = 1024;
    copy_kernel<<<grid, block, 0, stream>>>(dst, src, n);
}

void launch_l2_normalize(
    float* x,
    int32_t n,
    float* d_norm_sq,
    cudaStream_t stream
) {
    const int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    if (grid_size > 1024) grid_size = 1024;

    cudaMemsetAsync(d_norm_sq, 0, sizeof(float), stream);
    l2_norm_sq_kernel<<<grid_size, block_size, block_size * sizeof(float), stream>>>(
        x, n, d_norm_sq
    );
    scale_kernel<<<grid_size, block_size, 0, stream>>>(
        x, n, d_norm_sq
    );
}

}  

katz_centrality_result_t katz_centrality_mask(const graph32_t& graph,
                           const float* edge_weights,
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
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    const uint32_t* edge_mask = graph.edge_mask;

    cache.ensure(num_vertices);

    cudaStream_t stream = 0;

    float* ptr_a = cache.buf_a;
    float* ptr_b = cache.buf_b;
    float* x_old = ptr_a;
    float* x_new = ptr_b;

    std::size_t iterations = 0;
    bool converged = false;

    if (has_initial_guess) {
        cudaMemcpyAsync(ptr_a, centralities, num_vertices * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemsetAsync(ptr_a, 0, num_vertices * sizeof(float), stream);

        if (max_iterations > 0) {
            if (betas) {
                launch_copy(ptr_b, betas, num_vertices, stream);
            } else {
                launch_fill(ptr_b, beta, num_vertices, stream);
            }
            iterations = 1;
            x_old = ptr_b;
            x_new = ptr_a;
        }
    }

    for (std::size_t iter = iterations; iter < max_iterations; iter++) {
        cudaMemsetAsync(cache.d_l1_diff, 0, sizeof(float), stream);

        launch_katz_spmv(offsets, indices, edge_weights, edge_mask,
                         x_old, x_new, alpha, beta, betas,
                         num_vertices, cache.d_l1_diff, stream);

        cudaMemcpyAsync(cache.h_l1_pinned, cache.d_l1_diff, sizeof(float),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations = iter + 1;

        float* temp = x_old;
        x_old = x_new;
        x_new = temp;

        if (*cache.h_l1_pinned < epsilon) {
            converged = true;
            break;
        }
    }

    if (normalize) {
        launch_l2_normalize(x_old, num_vertices, cache.d_norm_sq, stream);
    }

    cudaMemcpyAsync(centralities, x_old, num_vertices * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    return {iterations, converged};
}

}  
