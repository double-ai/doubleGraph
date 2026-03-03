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
#include <cstring>

namespace aai {

namespace {




struct Cache : Cacheable {
    float* pr_a = nullptr;
    float* pr_b = nullptr;
    float* pr_norm = nullptr;
    int32_t* out_degree = nullptr;
    float* inv_out_deg = nullptr;
    double* scalars = nullptr;  

    int32_t pr_a_cap = 0;
    int32_t pr_b_cap = 0;
    int32_t pr_norm_cap = 0;
    int32_t out_degree_cap = 0;
    int32_t inv_out_deg_cap = 0;
    int32_t scalars_cap = 0;

    void ensure(int32_t N) {
        if (pr_a_cap < N) {
            if (pr_a) cudaFree(pr_a);
            cudaMalloc(&pr_a, (size_t)N * sizeof(float));
            pr_a_cap = N;
        }
        if (pr_b_cap < N) {
            if (pr_b) cudaFree(pr_b);
            cudaMalloc(&pr_b, (size_t)N * sizeof(float));
            pr_b_cap = N;
        }
        if (pr_norm_cap < N) {
            if (pr_norm) cudaFree(pr_norm);
            cudaMalloc(&pr_norm, (size_t)N * sizeof(float));
            pr_norm_cap = N;
        }
        if (out_degree_cap < N) {
            if (out_degree) cudaFree(out_degree);
            cudaMalloc(&out_degree, (size_t)N * sizeof(int32_t));
            out_degree_cap = N;
        }
        if (inv_out_deg_cap < N) {
            if (inv_out_deg) cudaFree(inv_out_deg);
            cudaMalloc(&inv_out_deg, (size_t)N * sizeof(float));
            inv_out_deg_cap = N;
        }
        if (scalars_cap < 2) {
            if (scalars) cudaFree(scalars);
            cudaMalloc(&scalars, 2 * sizeof(double));
            scalars_cap = 2;
        }
    }

    ~Cache() override {
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (pr_norm) cudaFree(pr_norm);
        if (out_degree) cudaFree(out_degree);
        if (inv_out_deg) cudaFree(inv_out_deg);
        if (scalars) cudaFree(scalars);
    }
};




template<int NUM_WARPS>
__device__ __forceinline__ void block_reduce_add_double(double val, double* __restrict__ d_out) {
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }

    
    __shared__ double warp_sums[NUM_WARPS];
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();

    
    if (warp_id == 0) {
        double d = (lane < NUM_WARPS) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
            d += __shfl_down_sync(0xFFFFFFFF, d, offset);
        }
        if (lane == 0 && d != 0.0) {
            atomicAdd(d_out, d);
        }
    }
}




__global__ void compute_out_degree_kernel(
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ out_degree,
    const int32_t num_edges)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_edges;
         idx += blockDim.x * gridDim.x) {
        atomicAdd(&out_degree[__ldg(&indices[idx])], 1);
    }
}




__global__ void compute_inv_degree_kernel(
    const int32_t* __restrict__ out_degree,
    float* __restrict__ inv_out_deg,
    const int32_t N)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N;
         idx += blockDim.x * gridDim.x) {
        int deg = out_degree[idx];
        inv_out_deg[idx] = (deg > 0) ? (1.0f / (float)deg) : 0.0f;
    }
}




__global__ void init_pr_kernel(float* __restrict__ pr, const float val, const int32_t N) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < N;
         idx += blockDim.x * gridDim.x) {
        pr[idx] = val;
    }
}




template<int BLOCK_SIZE>
__global__ void pr_norm_and_dangling_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ inv_out_deg,
    float* __restrict__ pr_norm,
    double* __restrict__ d_dangling_sum,
    const int32_t N)
{
    double dangling_local = 0.0;

    for (int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x; idx < N;
         idx += BLOCK_SIZE * gridDim.x) {
        float inv_deg = inv_out_deg[idx];
        float pr_val = pr[idx];
        pr_norm[idx] = pr_val * inv_deg;
        if (inv_deg == 0.0f) {
            dangling_local += (double)pr_val;
        }
    }

    block_reduce_add_double<BLOCK_SIZE / 32>(dangling_local, d_dangling_sum);
}




template <int BLOCK_SIZE>
__global__ void spmv_high_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_norm,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    double* __restrict__ d_diff,
    const double* __restrict__ d_dangling_sum,
    const int32_t seg_start, const int32_t seg_end,
    const float alpha, const float base_val, const float alpha_over_N)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int v = seg_start + blockIdx.x;
    if (v >= seg_end) return;

    const int start = __ldg(&offsets[v]);
    const int end = __ldg(&offsets[v + 1]);

    float sum = 0.0f;
    for (int k = start + threadIdx.x; k < end; k += BLOCK_SIZE) {
        const int u = __ldg(&indices[k]);
        sum += __ldg(&pr_norm[u]);
    }

    float block_sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        const float dangling_s = (float)(*d_dangling_sum);
        const float new_val = base_val + alpha * block_sum + alpha_over_N * dangling_s;
        pr_new[v] = new_val;
        const double d = (double)fabsf(new_val - __ldg(&pr_old[v]));
        atomicAdd(d_diff, d);
    }
}




template <int WARPS_PER_BLOCK>
__global__ void spmv_mid_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_norm,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    double* __restrict__ d_diff,
    const double* __restrict__ d_dangling_sum,
    const int32_t seg_start, const int32_t seg_end,
    const float alpha, const float base_val, const float alpha_over_N)
{
    const int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;

    const int v = seg_start + global_warp_id;

    double diff = 0.0;

    if (v < seg_end) {
        const int start = __ldg(&offsets[v]);
        const int end = __ldg(&offsets[v + 1]);

        float sum = 0.0f;
        for (int k = start + lane; k < end; k += 32) {
            const int u = __ldg(&indices[k]);
            sum += __ldg(&pr_norm[u]);
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        if (lane == 0) {
            const float dangling_s = (float)(*d_dangling_sum);
            const float new_val = base_val + alpha * sum + alpha_over_N * dangling_s;
            pr_new[v] = new_val;
            diff = (double)fabsf(new_val - __ldg(&pr_old[v]));
        }
    }

    
    __shared__ double warp_diffs[WARPS_PER_BLOCK];
    if (lane == 0) {
        warp_diffs[warp_in_block] = diff;
    }
    __syncthreads();

    if (warp_in_block == 0 && lane < WARPS_PER_BLOCK) {
        double d = warp_diffs[lane];
        #pragma unroll
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1) {
            d += __shfl_down_sync(0xFFFFFFFF, d, offset);
        }
        if (lane == 0 && d != 0.0) {
            atomicAdd(d_diff, d);
        }
    }
}




template <int BLOCK_SIZE>
__global__ void spmv_low_zero_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_norm,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    double* __restrict__ d_diff,
    const double* __restrict__ d_dangling_sum,
    const int32_t seg_start, const int32_t seg_end,
    const float alpha, const float base_val, const float alpha_over_N)
{
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;

    
    const float constant = base_val + alpha_over_N * (float)(*d_dangling_sum);

    const int v = seg_start + blockIdx.x * BLOCK_SIZE + threadIdx.x;

    double diff = 0.0;

    if (v < seg_end) {
        const int start = __ldg(&offsets[v]);
        const int end   = __ldg(&offsets[v + 1]);
        const int degree = end - start;

        float sum = 0.0f;
        
        if (degree <= 4) {
            if (degree >= 1) sum += __ldg(&pr_norm[__ldg(&indices[start])]);
            if (degree >= 2) sum += __ldg(&pr_norm[__ldg(&indices[start + 1])]);
            if (degree >= 3) sum += __ldg(&pr_norm[__ldg(&indices[start + 2])]);
            if (degree >= 4) sum += __ldg(&pr_norm[__ldg(&indices[start + 3])]);
        } else {
            for (int k = start; k < end; k++) {
                sum += __ldg(&pr_norm[__ldg(&indices[k])]);
            }
        }

        const float new_val = constant + alpha * sum;
        pr_new[v] = new_val;
        diff = (double)fabsf(new_val - __ldg(&pr_old[v]));
    }

    block_reduce_add_double<NUM_WARPS>(diff, d_diff);
}





void launch_compute_out_degree(const int32_t* indices, int32_t* out_degree, int32_t num_edges) {
    if (num_edges <= 0) return;
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    if (grid > 4096) grid = 4096;
    compute_out_degree_kernel<<<grid, block>>>(indices, out_degree, num_edges);
}

void launch_compute_inv_degree(const int32_t* out_degree, float* inv_out_deg, int32_t N) {
    if (N <= 0) return;
    int block = 256;
    int grid = (N + block - 1) / block;
    if (grid > 4096) grid = 4096;
    compute_inv_degree_kernel<<<grid, block>>>(out_degree, inv_out_deg, N);
}

void launch_init_pr(float* pr, float val, int32_t N) {
    if (N <= 0) return;
    int block = 256;
    int grid = (N + block - 1) / block;
    if (grid > 4096) grid = 4096;
    init_pr_kernel<<<grid, block>>>(pr, val, N);
}

void launch_pr_norm_and_dangling(const float* pr, const float* inv_out_deg,
                                  float* pr_norm, double* d_dangling_sum, int32_t N) {
    if (N <= 0) return;
    constexpr int block = 256;
    int grid = (N + block - 1) / block;
    if (grid > 4096) grid = 4096;
    pr_norm_and_dangling_kernel<block><<<grid, block>>>(pr, inv_out_deg, pr_norm, d_dangling_sum, N);
}

void launch_spmv_high(const int32_t* offsets, const int32_t* indices,
                      const float* pr_norm, const float* pr_old, float* pr_new,
                      double* d_diff, const double* d_dangling_sum,
                      int32_t seg_start, int32_t seg_end,
                      float alpha, float base_val, float alpha_over_N) {
    int num_vertices = seg_end - seg_start;
    if (num_vertices <= 0) return;
    spmv_high_degree_kernel<256><<<num_vertices, 256>>>(
        offsets, indices, pr_norm, pr_old, pr_new,
        d_diff, d_dangling_sum,
        seg_start, seg_end, alpha, base_val, alpha_over_N);
}

void launch_spmv_mid(const int32_t* offsets, const int32_t* indices,
                     const float* pr_norm, const float* pr_old, float* pr_new,
                     double* d_diff, const double* d_dangling_sum,
                     int32_t seg_start, int32_t seg_end,
                     float alpha, float base_val, float alpha_over_N) {
    int num_vertices = seg_end - seg_start;
    if (num_vertices <= 0) return;
    constexpr int WARPS_PER_BLOCK = 8;
    int threads_per_block = WARPS_PER_BLOCK * 32;
    int num_blocks = (num_vertices + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    spmv_mid_degree_kernel<WARPS_PER_BLOCK><<<num_blocks, threads_per_block>>>(
        offsets, indices, pr_norm, pr_old, pr_new,
        d_diff, d_dangling_sum,
        seg_start, seg_end, alpha, base_val, alpha_over_N);
}

void launch_spmv_low_zero(const int32_t* offsets, const int32_t* indices,
                          const float* pr_norm, const float* pr_old, float* pr_new,
                          double* d_diff, const double* d_dangling_sum,
                          int32_t seg_start, int32_t seg_end,
                          float alpha, float base_val, float alpha_over_N) {
    int num_vertices = seg_end - seg_start;
    if (num_vertices <= 0) return;
    constexpr int block = 256;
    int grid = (num_vertices + block - 1) / block;
    spmv_low_zero_degree_kernel<block><<<grid, block>>>(
        offsets, indices, pr_norm, pr_old, pr_new,
        d_diff, d_dangling_sum,
        seg_start, seg_end, alpha, base_val, alpha_over_N);
}

}  




PageRankResult pagerank_seg(const graph32_t& graph,
                            float* pageranks,
                            const float* precomputed_vertex_out_weight_sums,
                            float alpha,
                            float epsilon,
                            std::size_t max_iterations,
                            const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t N = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg_high_start = seg[0];
    int32_t seg_mid_start  = seg[1];
    int32_t seg_low_start  = seg[2];
    int32_t seg_end_val    = seg[4];  

    cache.ensure(N);

    float* d_pr_a = cache.pr_a;
    float* d_pr_b = cache.pr_b;
    float* d_pr_norm = cache.pr_norm;
    int32_t* d_out_degree = cache.out_degree;
    float* d_inv_out_deg = cache.inv_out_deg;
    double* d_dangling_sum = cache.scalars;
    double* d_diff = d_dangling_sum + 1;

    
    cudaMemset(d_out_degree, 0, N * sizeof(int32_t));
    launch_compute_out_degree(d_indices, d_out_degree, num_edges);
    launch_compute_inv_degree(d_out_degree, d_inv_out_deg, N);

    
    if (initial_pageranks != nullptr) {
        cudaMemcpy(d_pr_a, initial_pageranks,
                   N * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        launch_init_pr(d_pr_a, 1.0f / (float)N, N);
    }

    float base_val = (1.0f - alpha) / (float)N;
    float alpha_over_N = alpha / (float)N;

    bool result_in_a = true;
    std::size_t iterations = 0;
    bool converged = false;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        float* pr_old = result_in_a ? d_pr_a : d_pr_b;
        float* pr_new = result_in_a ? d_pr_b : d_pr_a;

        
        cudaMemset(d_dangling_sum, 0, 2 * sizeof(double));

        
        launch_pr_norm_and_dangling(pr_old, d_inv_out_deg, d_pr_norm,
                                     d_dangling_sum, N);

        
        launch_spmv_high(d_offsets, d_indices, d_pr_norm, pr_old, pr_new,
                        d_diff, d_dangling_sum,
                        seg_high_start, seg_mid_start,
                        alpha, base_val, alpha_over_N);

        launch_spmv_mid(d_offsets, d_indices, d_pr_norm, pr_old, pr_new,
                       d_diff, d_dangling_sum,
                       seg_mid_start, seg_low_start,
                       alpha, base_val, alpha_over_N);

        launch_spmv_low_zero(d_offsets, d_indices, d_pr_norm, pr_old, pr_new,
                            d_diff, d_dangling_sum,
                            seg_low_start, seg_end_val,
                            alpha, base_val, alpha_over_N);

        result_in_a = !result_in_a;
        iterations = iter + 1;

        
        double h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);

        if (h_diff < (double)epsilon) {
            converged = true;
            break;
        }
    }

    
    float* result_ptr = result_in_a ? d_pr_a : d_pr_b;
    cudaMemcpy(pageranks, result_ptr,
               N * sizeof(float), cudaMemcpyDeviceToDevice);

    return PageRankResult{iterations, converged};
}

}  
