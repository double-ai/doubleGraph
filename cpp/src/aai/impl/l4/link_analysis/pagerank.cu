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
#include <algorithm>

namespace aai {

namespace {


constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;


constexpr int WARP_DEGREE_THRESHOLD = 6;


struct Cache : Cacheable {
    float* pr_a = nullptr;
    float* pr_b = nullptr;
    float* pr_norm = nullptr;
    float* inv_deg = nullptr;
    float* scratch = nullptr;
    int32_t* out_deg = nullptr;
    int64_t capacity = 0;
    bool scratch_allocated = false;
    int max_warp_blocks = 0;
    bool warp_blocks_init = false;

    void ensure(int32_t N) {
        if (capacity < N) {
            if (pr_a) cudaFree(pr_a);
            if (pr_b) cudaFree(pr_b);
            if (pr_norm) cudaFree(pr_norm);
            if (inv_deg) cudaFree(inv_deg);
            if (out_deg) cudaFree(out_deg);
            cudaMalloc(&pr_a, (size_t)N * sizeof(float));
            cudaMalloc(&pr_b, (size_t)N * sizeof(float));
            cudaMalloc(&pr_norm, (size_t)N * sizeof(float));
            cudaMalloc(&inv_deg, (size_t)N * sizeof(float));
            cudaMalloc(&out_deg, (size_t)N * sizeof(int32_t));
            capacity = N;
        }
        if (!scratch_allocated) {
            cudaMalloc(&scratch, 2 * sizeof(float));
            scratch_allocated = true;
        }
    }

    ~Cache() override {
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (pr_norm) cudaFree(pr_norm);
        if (inv_deg) cudaFree(inv_deg);
        if (scratch) cudaFree(scratch);
        if (out_deg) cudaFree(out_deg);
    }
};



__global__ void compute_out_degrees_kernel(
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ out_degrees,
    int32_t num_edges)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < num_edges) {
        atomicAdd(&out_degrees[indices[idx]], 1);
    }
}

__global__ void compute_inv_out_degrees_kernel(
    const int32_t* __restrict__ out_degrees,
    float* __restrict__ inv_out_degrees,
    int32_t N)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < N) {
        int deg = out_degrees[idx];
        inv_out_degrees[idx] = (deg > 0) ? (1.0f / (float)deg) : 0.0f;
    }
}

__global__ void init_pr_kernel(float* __restrict__ pr, float val, int32_t N)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < N) {
        pr[idx] = val;
    }
}


__global__ void normalize_dangling_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ inv_out_degrees,
    float* __restrict__ pr_norm,
    float* __restrict__ dangling_sum,
    int32_t N)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float dang = 0.0f;

    if (idx < N) {
        float p = pr[idx];
        float inv_deg = inv_out_degrees[idx];
        pr_norm[idx] = p * inv_deg;  
        if (inv_deg == 0.0f) {
            dang = p;
        }
    }

    float block_sum = BlockReduce(temp).Sum(dang);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(dangling_sum, block_sum);
    }
}


__global__ void spmv_update_diff_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_norm,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    const float* __restrict__ dangling_sum_ptr,
    float* __restrict__ l1_diff_ptr,
    float alpha,
    float one_minus_alpha_over_N,
    float alpha_over_N,
    int32_t N)
{
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane = tid & (WARP_SIZE - 1);
    int total_warps = gridDim.x * WARPS_PER_BLOCK;
    int warp_in_block = threadIdx.x / WARP_SIZE;

    
    __shared__ float s_dangling_contrib;
    __shared__ float s_warp_diff[WARPS_PER_BLOCK];

    if (threadIdx.x == 0) {
        s_dangling_contrib = (*dangling_sum_ptr) * alpha_over_N + one_minus_alpha_over_N;
    }
    __syncthreads();

    float base_score = s_dangling_contrib;
    float warp_diff = 0.0f;

    for (int v = warp_id; v < N; v += total_warps) {
        int start = offsets[v];
        int end = offsets[v + 1];

        float sum = 0.0f;
        
        for (int k = start + lane; k < end; k += WARP_SIZE) {
            sum += pr_norm[indices[k]];
        }

        
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            float new_val = alpha * sum + base_score;
            pr_new[v] = new_val;
            warp_diff += fabsf(new_val - pr_old[v]);
        }
    }

    
    if (lane == 0) {
        s_warp_diff[warp_in_block] = warp_diff;
    }
    __syncthreads();

    if (warp_in_block == 0) {
        float val = (lane < WARPS_PER_BLOCK) ? s_warp_diff[lane] : 0.0f;
        #pragma unroll
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0 && val != 0.0f) {
            atomicAdd(l1_diff_ptr, val);
        }
    }
}


__global__ void spmv_update_diff_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_norm,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    const float* __restrict__ dangling_sum_ptr,
    float* __restrict__ l1_diff_ptr,
    float alpha,
    float one_minus_alpha_over_N,
    float alpha_over_N,
    int32_t N)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float s_base_score;

    if (threadIdx.x == 0) {
        s_base_score = (*dangling_sum_ptr) * alpha_over_N + one_minus_alpha_over_N;
    }
    __syncthreads();

    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float my_diff = 0.0f;

    if (v < N) {
        int start = offsets[v];
        int end = offsets[v + 1];

        float sum = 0.0f;
        for (int k = start; k < end; k++) {
            sum += pr_norm[indices[k]];
        }

        float new_val = alpha * sum + s_base_score;
        pr_new[v] = new_val;
        my_diff = fabsf(new_val - pr_old[v]);
    }

    float block_diff = BlockReduce(temp).Sum(my_diff);
    if (threadIdx.x == 0) {
        atomicAdd(l1_diff_ptr, block_diff);
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
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    cudaStream_t stream = 0;

    cache.ensure(N);

    
    if (!cache.warp_blocks_init) {
        int max_blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks_per_sm, spmv_update_diff_warp_kernel, BLOCK_SIZE, 0);
        int num_sms = 0;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        cache.max_warp_blocks = max_blocks_per_sm * num_sms;
        cache.warp_blocks_init = true;
    }

    
    int avg_degree = (N > 0) ? (num_edges / N) : 0;
    bool use_warp = (avg_degree >= WARP_DEGREE_THRESHOLD);

    
    int warp_grid = std::min(cache.max_warp_blocks,
                            (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    float* d_pr = cache.pr_a;
    float* d_pr_new = cache.pr_b;
    float* d_pr_norm = cache.pr_norm;
    float* d_inv_deg = cache.inv_deg;
    float* d_dangling_sum = cache.scratch;
    float* d_l1_diff = cache.scratch + 1;

    float inv_N = 1.0f / (float)N;
    float one_minus_alpha_over_N = (1.0f - alpha) * inv_N;
    float alpha_over_N = alpha * inv_N;

    
    {
        int32_t* d_out_deg = cache.out_deg;
        cudaMemsetAsync(d_out_deg, 0, (size_t)N * sizeof(int32_t), stream);
        int grid = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid > 0)
            compute_out_degrees_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_indices, d_out_deg, num_edges);
        int grid2 = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_inv_out_degrees_kernel<<<grid2, BLOCK_SIZE, 0, stream>>>(d_out_deg, d_inv_deg, N);
    }

    
    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(d_pr, initial_pageranks,
                       (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        init_pr_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(d_pr, inv_N, N);
    }

    
    bool converged = false;
    std::size_t iterations = 0;
    float h_l1_diff;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(d_dangling_sum, 0, 2 * sizeof(float), stream);

        
        {
            int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            normalize_dangling_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                d_pr, d_inv_deg, d_pr_norm, d_dangling_sum, N);
        }

        
        if (use_warp) {
            spmv_update_diff_warp_kernel<<<warp_grid, BLOCK_SIZE, 0, stream>>>(
                d_offsets, d_indices, d_pr_norm, d_pr,
                d_pr_new, d_dangling_sum, d_l1_diff,
                alpha, one_minus_alpha_over_N, alpha_over_N, N);
        } else {
            int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            spmv_update_diff_thread_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
                d_offsets, d_indices, d_pr_norm, d_pr,
                d_pr_new, d_dangling_sum, d_l1_diff,
                alpha, one_minus_alpha_over_N, alpha_over_N, N);
        }

        
        cudaMemcpyAsync(&h_l1_diff, d_l1_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations++;

        
        float* tmp = d_pr;
        d_pr = d_pr_new;
        d_pr_new = tmp;

        if (h_l1_diff < epsilon) {
            converged = true;
            break;
        }
    }

    
    cudaMemcpyAsync(pageranks, d_pr, (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    return PageRankResult{iterations, converged};
}

}  
