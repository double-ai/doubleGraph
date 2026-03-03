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
    int32_t* d_out_degrees = nullptr;
    float* d_pr_a = nullptr;
    float* d_pr_b = nullptr;
    float* d_pr_norm = nullptr;
    double* d_dangling_a = nullptr;
    double* d_dangling_b = nullptr;
    double* d_l1_diff = nullptr;

    int64_t out_degrees_cap = 0;
    int64_t pr_a_cap = 0;
    int64_t pr_b_cap = 0;
    int64_t pr_norm_cap = 0;
    int64_t dangling_a_cap = 0;
    int64_t dangling_b_cap = 0;
    int64_t l1_diff_cap = 0;

    void ensure(int32_t num_vertices) {
        int64_t n = static_cast<int64_t>(num_vertices);
        if (out_degrees_cap < n) {
            if (d_out_degrees) cudaFree(d_out_degrees);
            cudaMalloc(&d_out_degrees, n * sizeof(int32_t));
            out_degrees_cap = n;
        }
        if (pr_a_cap < n) {
            if (d_pr_a) cudaFree(d_pr_a);
            cudaMalloc(&d_pr_a, n * sizeof(float));
            pr_a_cap = n;
        }
        if (pr_b_cap < n) {
            if (d_pr_b) cudaFree(d_pr_b);
            cudaMalloc(&d_pr_b, n * sizeof(float));
            pr_b_cap = n;
        }
        if (pr_norm_cap < n) {
            if (d_pr_norm) cudaFree(d_pr_norm);
            cudaMalloc(&d_pr_norm, n * sizeof(float));
            pr_norm_cap = n;
        }
        if (dangling_a_cap < 1) {
            cudaMalloc(&d_dangling_a, sizeof(double));
            dangling_a_cap = 1;
        }
        if (dangling_b_cap < 1) {
            cudaMalloc(&d_dangling_b, sizeof(double));
            dangling_b_cap = 1;
        }
        if (l1_diff_cap < 1) {
            cudaMalloc(&d_l1_diff, sizeof(double));
            l1_diff_cap = 1;
        }
    }

    ~Cache() override {
        if (d_out_degrees) cudaFree(d_out_degrees);
        if (d_pr_a) cudaFree(d_pr_a);
        if (d_pr_b) cudaFree(d_pr_b);
        if (d_pr_norm) cudaFree(d_pr_norm);
        if (d_dangling_a) cudaFree(d_dangling_a);
        if (d_dangling_b) cudaFree(d_dangling_b);
        if (d_l1_diff) cudaFree(d_l1_diff);
    }
};

__global__ void compute_out_degrees_kernel(
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ out_degrees,
    int32_t num_edges)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        atomicAdd(&out_degrees[indices[tid]], 1);
    }
}

__global__ void init_pr_kernel(float* __restrict__ pr, int32_t num_vertices, float init_val)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        pr[tid] = init_val;
    }
}

__global__ void initial_normalize_kernel(
    const float* __restrict__ pr,
    float* __restrict__ pr_norm,
    const int32_t* __restrict__ out_degrees,
    double* __restrict__ dangling_sum,
    int32_t num_vertices)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    double my_dangling = 0.0;
    if (tid < num_vertices) {
        int32_t deg = out_degrees[tid];
        if (deg > 0) {
            pr_norm[tid] = pr[tid] / (float)deg;
        } else {
            pr_norm[tid] = 0.0f;
            my_dangling = (double)pr[tid];
        }
    }

    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_sum = BlockReduce(temp_storage).Sum(my_dangling);

    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(dangling_sum, block_sum);
    }
}

template <int BLOCK_SIZE>
__global__ void spmv_high_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_norm,
    float* __restrict__ spmv_result,
    int32_t seg_start, int32_t seg_end)
{
    int32_t v = seg_start + blockIdx.x;
    if (v >= seg_end) return;
    int32_t start = offsets[v], end = offsets[v + 1];
    float sum = 0.0f;
    for (int32_t e = start + threadIdx.x; e < end; e += BLOCK_SIZE)
        sum += pr_norm[indices[e]];
    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage ts;
    float bs = BR(ts).Sum(sum);
    if (threadIdx.x == 0) spmv_result[v] = bs;
}

__global__ void spmv_mid_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_norm,
    float* __restrict__ spmv_result,
    int32_t seg_start, int32_t seg_end)
{
    int32_t wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int32_t lane = threadIdx.x & 31;
    int32_t v = seg_start + wid;
    if (v >= seg_end) return;
    int32_t start = offsets[v], end = offsets[v + 1];
    float sum = 0.0f;
    for (int32_t e = start + lane; e < end; e += 32)
        sum += pr_norm[indices[e]];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, o);
    if (lane == 0) spmv_result[v] = sum;
}

__global__ void spmv_low_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_norm,
    float* __restrict__ spmv_result,
    int32_t seg_start, int32_t seg_end)
{
    int32_t v = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= seg_end) return;
    int32_t start = offsets[v], end = offsets[v + 1];
    float sum = 0.0f;
    for (int32_t e = start; e < end; e++)
        sum += pr_norm[indices[e]];
    spmv_result[v] = sum;
}

__global__ void fused_update_normalize_kernel(
    float* __restrict__ pr_new,
    const float* __restrict__ pr,
    float* __restrict__ pr_norm_next,
    const int32_t* __restrict__ out_degrees,
    const double* __restrict__ dangling_sum_ptr,
    double* __restrict__ dangling_sum_next,
    double* __restrict__ l1_diff,
    int32_t num_vertices,
    int32_t zero_start,
    float alpha,
    float inv_n,
    float base_pr)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    float dangling_contrib = (float)(alpha * (*dangling_sum_ptr) * (double)inv_n);

    double my_diff = 0.0;
    double my_dangling = 0.0;

    if (tid < num_vertices) {
        float spmv_val = (tid < zero_start) ? pr_new[tid] : 0.0f;
        float new_val = base_pr + alpha * spmv_val + dangling_contrib;
        pr_new[tid] = new_val;

        my_diff = fabs((double)new_val - (double)pr[tid]);

        int32_t deg = out_degrees[tid];
        if (deg > 0) {
            pr_norm_next[tid] = new_val / (float)deg;
        } else {
            pr_norm_next[tid] = 0.0f;
            my_dangling = (double)new_val;
        }
    }

    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp1;
    double block_diff = BlockReduce(temp1).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff > 0.0) {
        atomicAdd(l1_diff, block_diff);
    }

    __syncthreads();
    __shared__ typename BlockReduce::TempStorage temp2;
    double block_dang = BlockReduce(temp2).Sum(my_dangling);
    if (threadIdx.x == 0 && block_dang != 0.0) {
        atomicAdd(dangling_sum_next, block_dang);
    }
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
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    if (num_vertices == 0) {
        return {0, true};
    }

    cache.ensure(num_vertices);

    int32_t* d_out_degrees = cache.d_out_degrees;
    float* d_pr_a = cache.d_pr_a;
    float* d_pr_b = cache.d_pr_b;
    float* d_pr_norm = cache.d_pr_norm;
    double* d_dangling_a = cache.d_dangling_a;
    double* d_dangling_b = cache.d_dangling_b;
    double* d_l1_diff = cache.d_l1_diff;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0];
    int32_t seg1 = seg[1];
    int32_t seg2 = seg[2];
    int32_t seg3 = seg[3];

    float inv_n = 1.0f / (float)num_vertices;
    float base_pr = (1.0f - alpha) * inv_n;

    const int BLOCK = 256;
    int grid_verts = (num_vertices + BLOCK - 1) / BLOCK;

    int n_high = seg1 - seg0;
    int n_mid = seg2 - seg1;
    int n_low = seg3 - seg2;

    int warps_per_block = BLOCK / 32;
    int blocks_mid = (n_mid > 0) ? (n_mid + warps_per_block - 1) / warps_per_block : 0;
    int blocks_low = (n_low > 0) ? (n_low + BLOCK - 1) / BLOCK : 0;

    
    cudaMemset(d_out_degrees, 0, num_vertices * sizeof(int32_t));
    int grid_edges = (num_edges + BLOCK - 1) / BLOCK;
    if (grid_edges > 0)
        compute_out_degrees_kernel<<<grid_edges, BLOCK>>>(d_indices, d_out_degrees, num_edges);

    
    float* d_pr = d_pr_a;
    float* d_pr_new = d_pr_b;
    double* d_dangling_curr = d_dangling_a;
    double* d_dangling_next = d_dangling_b;

    if (initial_pageranks != nullptr) {
        cudaMemcpy(d_pr, initial_pageranks, num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        init_pr_kernel<<<grid_verts, BLOCK>>>(d_pr, num_vertices, inv_n);
    }

    
    cudaMemset(d_dangling_curr, 0, sizeof(double));
    initial_normalize_kernel<<<grid_verts, BLOCK>>>(
        d_pr, d_pr_norm, d_out_degrees, d_dangling_curr, num_vertices);

    bool converged = false;
    std::size_t iter;
    int check_interval = 4;

    for (iter = 0; iter < max_iterations; iter++) {
        
        if (n_high > 0) {
            spmv_high_degree_kernel<256><<<n_high, 256>>>(
                d_offsets, d_indices, d_pr_norm, d_pr_new, seg0, seg1);
        }
        if (blocks_mid > 0) {
            spmv_mid_degree_kernel<<<blocks_mid, BLOCK>>>(
                d_offsets, d_indices, d_pr_norm, d_pr_new, seg1, seg2);
        }
        if (blocks_low > 0) {
            spmv_low_degree_kernel<<<blocks_low, BLOCK>>>(
                d_offsets, d_indices, d_pr_norm, d_pr_new, seg2, seg3);
        }

        
        cudaMemsetAsync(d_l1_diff, 0, sizeof(double));
        cudaMemsetAsync(d_dangling_next, 0, sizeof(double));

        
        fused_update_normalize_kernel<<<grid_verts, BLOCK>>>(
            d_pr_new, d_pr, d_pr_norm, d_out_degrees,
            d_dangling_curr, d_dangling_next, d_l1_diff,
            num_vertices, seg3, alpha, inv_n, base_pr);

        
        { float* t = d_pr; d_pr = d_pr_new; d_pr_new = t; }
        { double* t = d_dangling_curr; d_dangling_curr = d_dangling_next; d_dangling_next = t; }

        
        bool should_check = ((iter + 1) % (std::size_t)check_interval == 0) || (iter + 1 >= max_iterations);
        if (should_check) {
            double h_l1;
            cudaMemcpy(&h_l1, d_l1_diff, sizeof(double), cudaMemcpyDeviceToHost);
            if (h_l1 < (double)epsilon) {
                converged = true;
                iter++;
                break;
            }
        }
    }

    
    if (d_pr != pageranks) {
        cudaMemcpy(pageranks, d_pr, num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return {iter, converged};
}

}  
