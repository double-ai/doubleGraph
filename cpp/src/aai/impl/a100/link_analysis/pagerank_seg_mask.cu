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
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* pr_b = nullptr;
    float* out_degree = nullptr;
    float* inv_out_degree = nullptr;
    float* pr_scaled = nullptr;
    double* scratch = nullptr;  
    int64_t capacity = 0;

    void ensure(int32_t num_vertices) {
        int64_t n = static_cast<int64_t>(num_vertices);
        if (capacity < n) {
            if (pr_b) cudaFree(pr_b);
            if (out_degree) cudaFree(out_degree);
            if (inv_out_degree) cudaFree(inv_out_degree);
            if (pr_scaled) cudaFree(pr_scaled);
            if (scratch) cudaFree(scratch);
            cudaMalloc(&pr_b, n * sizeof(float));
            cudaMalloc(&out_degree, n * sizeof(float));
            cudaMalloc(&inv_out_degree, n * sizeof(float));
            cudaMalloc(&pr_scaled, n * sizeof(float));
            cudaMalloc(&scratch, 2 * sizeof(double));
            capacity = n;
        }
    }

    ~Cache() override {
        if (pr_b) cudaFree(pr_b);
        if (out_degree) cudaFree(out_degree);
        if (inv_out_degree) cudaFree(inv_out_degree);
        if (pr_scaled) cudaFree(pr_scaled);
        if (scratch) cudaFree(scratch);
    }
};




__global__ void compute_out_degree_masked_kernel(
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ out_degree,
    int32_t num_edges)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        uint32_t word = edge_mask[tid >> 5];
        if ((word >> (tid & 31)) & 1u) {
            atomicAdd(&out_degree[indices[tid]], 1.0f);
        }
    }
}




__global__ void compute_inv_out_degree_kernel(
    const float* __restrict__ out_degree,
    float* __restrict__ inv_out_degree,
    int32_t num_vertices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        float d = out_degree[tid];
        inv_out_degree[tid] = (d > 0.0f) ? (1.0f / d) : 0.0f;
    }
}




__global__ void init_pageranks_kernel(
    float* __restrict__ pr,
    float init_val,
    int32_t num_vertices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        pr[tid] = init_val;
    }
}





__global__ void prepare_iteration_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ inv_out_degree,
    float* __restrict__ pr_scaled,
    double* __restrict__ dangling_sum,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    double dval = 0.0;
    if (tid < num_vertices) {
        float p = pr[tid];
        float iow = inv_out_degree[tid];
        pr_scaled[tid] = p * iow;
        if (iow == 0.0f) {
            dval = (double)p;
        }
    }

    double block_sum = BlockReduce(temp_storage).Sum(dval);
    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(dangling_sum, block_sum);
    }
}




__global__ void spmv_high_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ pr_scaled,
    float* __restrict__ pr_new,
    const double* __restrict__ d_dangling_sum,
    float alpha,
    float base_no_dang,
    float alpha_over_N,
    int32_t seg_start,
    int32_t seg_end)
{
    int v = seg_start + blockIdx.x;
    if (v >= seg_end) return;

    __shared__ float s_base_val;
    if (threadIdx.x == 0) {
        s_base_val = base_no_dang + alpha_over_N * (float)(*d_dangling_sum);
    }
    __syncthreads();
    float base_val = s_base_val;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    float sum = 0.0f;
    for (int32_t j = start + threadIdx.x; j < end; j += blockDim.x) {
        uint32_t word = edge_mask[j >> 5];
        if ((word >> (j & 31)) & 1u) {
            sum += pr_scaled[indices[j]];
        }
    }

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        pr_new[v] = base_val + alpha * block_sum;
    }
}




__global__ void spmv_mid_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ pr_scaled,
    float* __restrict__ pr_new,
    const double* __restrict__ d_dangling_sum,
    float alpha,
    float base_no_dang,
    float alpha_over_N,
    int32_t seg_start,
    int32_t seg_end)
{
    __shared__ float s_base_val;
    if (threadIdx.x == 0) {
        s_base_val = base_no_dang + alpha_over_N * (float)(*d_dangling_sum);
    }
    __syncthreads();
    float base_val = s_base_val;

    int warps_per_block = blockDim.x >> 5;
    int warp_in_block = threadIdx.x >> 5;
    int lane_id = threadIdx.x & 31;
    int global_warp_id = blockIdx.x * warps_per_block + warp_in_block;
    int v = seg_start + global_warp_id;
    if (v >= seg_end) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    float sum = 0.0f;
    for (int32_t j = start + lane_id; j < end; j += 32) {
        uint32_t word = edge_mask[j >> 5];
        if ((word >> (j & 31)) & 1u) {
            sum += pr_scaled[indices[j]];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane_id == 0) {
        pr_new[v] = base_val + alpha * sum;
    }
}




__global__ void spmv_low_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ pr_scaled,
    float* __restrict__ pr_new,
    const double* __restrict__ d_dangling_sum,
    float alpha,
    float base_no_dang,
    float alpha_over_N,
    int32_t seg_start,
    int32_t seg_end)
{
    __shared__ float s_base_val;
    if (threadIdx.x == 0) {
        s_base_val = base_no_dang + alpha_over_N * (float)(*d_dangling_sum);
    }
    __syncthreads();
    float base_val = s_base_val;

    int v = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= seg_end) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    float sum = 0.0f;
    for (int32_t j = start; j < end; j++) {
        uint32_t word = edge_mask[j >> 5];
        if ((word >> (j & 31)) & 1u) {
            sum += pr_scaled[indices[j]];
        }
    }

    pr_new[v] = base_val + alpha * sum;
}




__global__ void spmv_zero_degree_kernel(
    float* __restrict__ pr_new,
    const double* __restrict__ d_dangling_sum,
    float base_no_dang,
    float alpha_over_N,
    int32_t seg_start,
    int32_t seg_end)
{
    __shared__ float s_base_val;
    if (threadIdx.x == 0) {
        s_base_val = base_no_dang + alpha_over_N * (float)(*d_dangling_sum);
    }
    __syncthreads();

    int v = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (v < seg_end) {
        pr_new[v] = s_base_val;
    }
}




__global__ void compute_l1_diff_kernel(
    const float* __restrict__ pr_new,
    const float* __restrict__ pr_old,
    double* __restrict__ diff_output,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double val = 0.0;
    if (tid < num_vertices) {
        val = fabs((double)pr_new[tid] - (double)pr_old[tid]);
    }

    double block_sum = BlockReduce(temp_storage).Sum(val);
    if (threadIdx.x == 0 && block_sum > 0.0) {
        atomicAdd(diff_output, block_sum);
    }
}





static void launch_compute_out_degree_masked(
    const int32_t* indices, const uint32_t* edge_mask,
    float* out_degree, int32_t num_edges, cudaStream_t stream)
{
    if (num_edges <= 0) return;
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    compute_out_degree_masked_kernel<<<grid, block, 0, stream>>>(
        indices, edge_mask, out_degree, num_edges);
}

static void launch_compute_inv_out_degree(
    const float* out_degree, float* inv_out_degree,
    int32_t num_vertices, cudaStream_t stream)
{
    if (num_vertices <= 0) return;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    compute_inv_out_degree_kernel<<<grid, block, 0, stream>>>(
        out_degree, inv_out_degree, num_vertices);
}

static void launch_init_pageranks(float* pr, float init_val, int32_t num_vertices, cudaStream_t stream)
{
    if (num_vertices <= 0) return;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    init_pageranks_kernel<<<grid, block, 0, stream>>>(pr, init_val, num_vertices);
}

static void launch_prepare_iteration(
    const float* pr, const float* inv_out_degree,
    float* pr_scaled, double* dangling_sum,
    int32_t num_vertices, cudaStream_t stream)
{
    if (num_vertices <= 0) return;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    prepare_iteration_kernel<<<grid, block, 0, stream>>>(
        pr, inv_out_degree, pr_scaled, dangling_sum, num_vertices);
}

static void launch_spmv_high_degree(
    const int32_t* offsets, const int32_t* indices,
    const uint32_t* edge_mask, const float* pr_scaled,
    float* pr_new, const double* d_dangling_sum,
    float alpha, float base_no_dang, float alpha_over_N,
    int32_t seg_start, int32_t seg_end, cudaStream_t stream)
{
    int num_verts = seg_end - seg_start;
    if (num_verts <= 0) return;
    spmv_high_degree_kernel<<<num_verts, 256, 0, stream>>>(
        offsets, indices, edge_mask, pr_scaled, pr_new, d_dangling_sum,
        alpha, base_no_dang, alpha_over_N, seg_start, seg_end);
}

static void launch_spmv_mid_degree(
    const int32_t* offsets, const int32_t* indices,
    const uint32_t* edge_mask, const float* pr_scaled,
    float* pr_new, const double* d_dangling_sum,
    float alpha, float base_no_dang, float alpha_over_N,
    int32_t seg_start, int32_t seg_end, cudaStream_t stream)
{
    int num_verts = seg_end - seg_start;
    if (num_verts <= 0) return;
    int warps_per_block = 4;
    int block = warps_per_block * 32;
    int grid = (num_verts + warps_per_block - 1) / warps_per_block;
    spmv_mid_degree_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, edge_mask, pr_scaled, pr_new, d_dangling_sum,
        alpha, base_no_dang, alpha_over_N, seg_start, seg_end);
}

static void launch_spmv_low_degree(
    const int32_t* offsets, const int32_t* indices,
    const uint32_t* edge_mask, const float* pr_scaled,
    float* pr_new, const double* d_dangling_sum,
    float alpha, float base_no_dang, float alpha_over_N,
    int32_t seg_start, int32_t seg_end, cudaStream_t stream)
{
    int num_verts = seg_end - seg_start;
    if (num_verts <= 0) return;
    int block = 256;
    int grid = (num_verts + block - 1) / block;
    spmv_low_degree_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, edge_mask, pr_scaled, pr_new, d_dangling_sum,
        alpha, base_no_dang, alpha_over_N, seg_start, seg_end);
}

static void launch_spmv_zero_degree(
    float* pr_new, const double* d_dangling_sum,
    float base_no_dang, float alpha_over_N,
    int32_t seg_start, int32_t seg_end, cudaStream_t stream)
{
    int num_verts = seg_end - seg_start;
    if (num_verts <= 0) return;
    int block = 256;
    int grid = (num_verts + block - 1) / block;
    spmv_zero_degree_kernel<<<grid, block, 0, stream>>>(
        pr_new, d_dangling_sum, base_no_dang, alpha_over_N, seg_start, seg_end);
}

static void launch_compute_l1_diff(
    const float* pr_new, const float* pr_old,
    double* diff_output, int32_t num_vertices, cudaStream_t stream)
{
    if (num_vertices <= 0) return;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    compute_l1_diff_kernel<<<grid, block, 0, stream>>>(
        pr_new, pr_old, diff_output, num_vertices);
}

}  

PageRankResult pagerank_seg_mask(const graph32_t& graph,
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
    const uint32_t* d_edge_mask = graph.edge_mask;

    const auto& seg_vec = graph.segment_offsets.value();
    int32_t seg[5] = {seg_vec[0], seg_vec[1], seg_vec[2], seg_vec[3], seg_vec[4]};

    cache.ensure(num_vertices);

    cudaStream_t stream = 0;

    float* d_pr = pageranks;
    float* d_pr_new = cache.pr_b;
    float* d_out_degree = cache.out_degree;
    float* d_inv_out_degree = cache.inv_out_degree;
    float* d_pr_scaled = cache.pr_scaled;
    double* d_dangling_sum = cache.scratch;
    double* d_diff = cache.scratch + 1;

    
    cudaMemsetAsync(d_out_degree, 0, num_vertices * sizeof(float), stream);
    launch_compute_out_degree_masked(d_indices, d_edge_mask, d_out_degree, num_edges, stream);

    
    launch_compute_inv_out_degree(d_out_degree, d_inv_out_degree, num_vertices, stream);

    
    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(d_pr, initial_pageranks,
                       num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        float init_val = 1.0f / static_cast<float>(num_vertices);
        launch_init_pageranks(d_pr, init_val, num_vertices, stream);
    }

    
    float base_no_dang = (1.0f - alpha) / static_cast<float>(num_vertices);
    float alpha_over_N = alpha / static_cast<float>(num_vertices);

    bool converged = false;
    std::size_t iterations = 0;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(d_dangling_sum, 0, 2 * sizeof(double), stream);

        
        launch_prepare_iteration(d_pr, d_inv_out_degree, d_pr_scaled, d_dangling_sum,
                                num_vertices, stream);

        
        launch_spmv_high_degree(d_offsets, d_indices, d_edge_mask, d_pr_scaled,
                               d_pr_new, d_dangling_sum,
                               alpha, base_no_dang, alpha_over_N,
                               seg[0], seg[1], stream);

        launch_spmv_mid_degree(d_offsets, d_indices, d_edge_mask, d_pr_scaled,
                              d_pr_new, d_dangling_sum,
                              alpha, base_no_dang, alpha_over_N,
                              seg[1], seg[2], stream);

        launch_spmv_low_degree(d_offsets, d_indices, d_edge_mask, d_pr_scaled,
                              d_pr_new, d_dangling_sum,
                              alpha, base_no_dang, alpha_over_N,
                              seg[2], seg[3], stream);

        launch_spmv_zero_degree(d_pr_new, d_dangling_sum,
                               base_no_dang, alpha_over_N,
                               seg[3], seg[4], stream);

        
        launch_compute_l1_diff(d_pr_new, d_pr, d_diff, num_vertices, stream);

        
        double h_diff;
        cudaMemcpyAsync(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations = iter + 1;

        
        std::swap(d_pr, d_pr_new);

        if (h_diff < (double)epsilon) {
            converged = true;
            break;
        }
    }

    
    if (d_pr != pageranks) {
        cudaMemcpyAsync(pageranks, d_pr, num_vertices * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    return PageRankResult{iterations, converged};
}

}  
