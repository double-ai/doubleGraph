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

#define BLOCK_SIZE 256

struct Cache : Cacheable {
    float* buf0 = nullptr;
    float* buf1 = nullptr;
    float* pr_norm = nullptr;
    int32_t* out_deg = nullptr;
    float* inv_out_deg = nullptr;
    float* scalars = nullptr;  

    int64_t buf_capacity = 0;
    int64_t out_deg_capacity = 0;
    int64_t inv_out_deg_capacity = 0;
    int64_t pr_norm_capacity = 0;
    int64_t scalars_capacity = 0;

    void ensure(int32_t N) {
        int64_t n = static_cast<int64_t>(N);
        if (buf_capacity < n) {
            if (buf0) cudaFree(buf0);
            if (buf1) cudaFree(buf1);
            cudaMalloc(&buf0, n * sizeof(float));
            cudaMalloc(&buf1, n * sizeof(float));
            buf_capacity = n;
        }
        if (pr_norm_capacity < n) {
            if (pr_norm) cudaFree(pr_norm);
            cudaMalloc(&pr_norm, n * sizeof(float));
            pr_norm_capacity = n;
        }
        if (out_deg_capacity < n) {
            if (out_deg) cudaFree(out_deg);
            cudaMalloc(&out_deg, n * sizeof(int32_t));
            out_deg_capacity = n;
        }
        if (inv_out_deg_capacity < n) {
            if (inv_out_deg) cudaFree(inv_out_deg);
            cudaMalloc(&inv_out_deg, n * sizeof(float));
            inv_out_deg_capacity = n;
        }
        if (scalars_capacity < 2) {
            if (scalars) cudaFree(scalars);
            cudaMalloc(&scalars, 2 * sizeof(float));
            scalars_capacity = 2;
        }
    }

    ~Cache() override {
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (pr_norm) cudaFree(pr_norm);
        if (out_deg) cudaFree(out_deg);
        if (inv_out_deg) cudaFree(inv_out_deg);
        if (scalars) cudaFree(scalars);
    }
};


__global__ void compute_out_degree_kernel(
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ out_degree,
    int32_t num_edges)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_edges;
         idx += blockDim.x * gridDim.x) {
        atomicAdd(&out_degree[indices[idx]], 1);
    }
}


__global__ void init_pr_kernel(float* __restrict__ pr, int32_t n)
{
    float val = 1.0f / (float)n;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n;
         idx += blockDim.x * gridDim.x) {
        pr[idx] = val;
    }
}


__global__ void compute_inv_out_degree_kernel(
    const int32_t* __restrict__ out_degree,
    float* __restrict__ inv_out_deg,
    int32_t num_vertices)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_vertices;
         idx += blockDim.x * gridDim.x) {
        int deg = out_degree[idx];
        inv_out_deg[idx] = (deg > 0) ? (1.0f / (float)deg) : 0.0f;
    }
}


__global__ __launch_bounds__(BLOCK_SIZE)
void normalize_dangling_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ inv_out_deg,
    float* __restrict__ pr_norm,
    float* __restrict__ d_dangling_sum,
    int32_t num_vertices)
{
    __shared__ float shared_sum[BLOCK_SIZE / 32];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    float dangling = 0.0f;

    if (idx < num_vertices) {
        float p = pr[idx];
        float inv_d = inv_out_deg[idx];
        pr_norm[idx] = p * inv_d;
        if (inv_d == 0.0f) {
            dangling = p;
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dangling += __shfl_down_sync(0xffffffff, dangling, offset);
    }

    if (lane == 0) shared_sum[warp_id] = dangling;
    __syncthreads();

    
    if (warp_id == 0) {
        float val = (lane < (BLOCK_SIZE / 32)) ? shared_sum[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0 && val != 0.0f) {
            atomicAdd(d_dangling_sum, val);
        }
    }
}


__global__ __launch_bounds__(BLOCK_SIZE)
void spmv_update_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_norm,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    const float* __restrict__ d_dangling_sum,
    float* __restrict__ d_l1_diff,
    float alpha,
    int32_t num_vertices)
{
    __shared__ float shared_diff[BLOCK_SIZE / 32];

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    float one_over_n = 1.0f / (float)num_vertices;
    float dangling_sum = *d_dangling_sum;
    float base = (1.0f - alpha) * one_over_n + alpha * dangling_sum * one_over_n;

    float diff = 0.0f;

    if (v < num_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];
        float sum = 0.0f;
        for (int j = start; j < end; j++) {
            sum += pr_norm[indices[j]];
        }
        float new_val = base + alpha * sum;
        pr_new[v] = new_val;
        diff = fabsf(new_val - pr_old[v]);
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        diff += __shfl_down_sync(0xffffffff, diff, offset);
    }

    if (lane == 0) shared_diff[warp_id] = diff;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < (BLOCK_SIZE / 32)) ? shared_diff[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0 && val != 0.0f) {
            atomicAdd(d_l1_diff, val);
        }
    }
}

}  

PageRankResult pagerank(const graph32_t& graph,
                        float* pageranks,
                        const float* precomputed_vertex_out_weight_sums,
                        float alpha,
                        float epsilon,
                        std::size_t max_iterations,
                        const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t N = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    cache.ensure(N);

    float* d_pr[2] = {cache.buf0, cache.buf1};
    float* d_pr_norm = cache.pr_norm;
    int32_t* d_out_deg = cache.out_deg;
    float* d_inv_out_deg = cache.inv_out_deg;
    float* d_dangling = cache.scalars;
    float* d_l1_diff = cache.scalars + 1;

    
    cudaMemsetAsync(d_out_deg, 0, N * sizeof(int32_t));
    int grid_e = (E + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid_e > 0)
        compute_out_degree_kernel<<<grid_e, BLOCK_SIZE>>>(d_indices, d_out_deg, E);
    int grid_v = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid_v > 0)
        compute_inv_out_degree_kernel<<<grid_v, BLOCK_SIZE>>>(d_out_deg, d_inv_out_deg, N);

    
    int cur = 0;
    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(d_pr[0], initial_pageranks,
                       N * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid > 0)
            init_pr_kernel<<<grid, BLOCK_SIZE>>>(d_pr[0], N);
    }

    
    bool converged = false;
    std::size_t iterations = 0;
    float h_l1;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        int next = 1 - cur;

        
        cudaMemsetAsync(d_dangling, 0, 2 * sizeof(float));

        
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid > 0)
            normalize_dangling_kernel<<<grid, BLOCK_SIZE>>>(
                d_pr[cur], d_inv_out_deg, d_pr_norm, d_dangling, N);

        
        if (grid > 0)
            spmv_update_kernel<<<grid, BLOCK_SIZE>>>(
                d_offsets, d_indices, d_pr_norm, d_pr[cur], d_pr[next],
                d_dangling, d_l1_diff, alpha, N);

        
        cudaMemcpy(&h_l1, d_l1_diff, sizeof(float), cudaMemcpyDeviceToHost);

        cur = next;
        iterations = iter + 1;

        if (h_l1 < epsilon) {
            converged = true;
            break;
        }
    }

    
    cudaMemcpyAsync(pageranks, d_pr[cur], N * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();

    return PageRankResult{iterations, converged};
}

}  
