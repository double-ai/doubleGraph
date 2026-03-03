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

namespace aai {

namespace {




struct Cache : Cacheable {
    float* pr_b = nullptr;
    int32_t pr_b_cap = 0;

    float* x = nullptr;
    int32_t x_cap = 0;

    float* out_weights = nullptr;
    int32_t out_weights_cap = 0;

    float* pers = nullptr;
    int32_t pers_cap = 0;

    float* scalars = nullptr;
    bool scalars_allocated = false;

    float* h_diff_pinned = nullptr;

    void ensure(int32_t n) {
        if (pr_b_cap < n) {
            cudaFree(pr_b);
            cudaMalloc(&pr_b, (int64_t)n * sizeof(float));
            pr_b_cap = n;
        }
        if (x_cap < n) {
            cudaFree(x);
            cudaMalloc(&x, (int64_t)n * sizeof(float));
            x_cap = n;
        }
        if (out_weights_cap < n) {
            cudaFree(out_weights);
            cudaMalloc(&out_weights, (int64_t)n * sizeof(float));
            out_weights_cap = n;
        }
        if (pers_cap < n) {
            cudaFree(pers);
            cudaMalloc(&pers, (int64_t)n * sizeof(float));
            pers_cap = n;
        }
        if (!scalars_allocated) {
            cudaMalloc(&scalars, 2 * sizeof(float));
            scalars_allocated = true;
        }
        if (!h_diff_pinned) {
            cudaMallocHost(&h_diff_pinned, sizeof(float));
        }
    }

    ~Cache() override {
        cudaFree(pr_b);
        cudaFree(x);
        cudaFree(out_weights);
        cudaFree(pers);
        cudaFree(scalars);
        if (h_diff_pinned) cudaFreeHost(h_diff_pinned);
    }
};





__device__ __forceinline__ bool is_edge_active(const uint32_t* mask, int j) {
    return (mask[j >> 5] >> (j & 31)) & 1u;
}

__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ out_weight_sums,
    int num_edges)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < num_edges && is_edge_active(edge_mask, j)) {
        atomicAdd(&out_weight_sums[indices[j]], edge_weights[j]);
    }
}

__global__ void init_pagerank_kernel(float* __restrict__ pr, int num_vertices, float init_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_vertices) {
        pr[i] = init_val;
    }
}

__global__ void build_pers_normalized_kernel(
    const int32_t* __restrict__ pers_vertices,
    const float* __restrict__ pers_values,
    float* __restrict__ pers_normalized,
    int pers_size)
{
    __shared__ float s_sum;

    float my_sum = 0.0f;
    for (int i = threadIdx.x; i < pers_size; i += blockDim.x) {
        my_sum += pers_values[i];
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        my_sum += __shfl_down_sync(0xffffffff, my_sum, offset);
    }

    __shared__ float warp_sums[8];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = my_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int num_warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < num_warps; i++) total += warp_sums[i];
        s_sum = (total > 0.0f) ? (1.0f / total) : 0.0f;
    }
    __syncthreads();

    float sum_inv = s_sum;
    for (int i = threadIdx.x; i < pers_size; i += blockDim.x) {
        pers_normalized[pers_vertices[i]] = pers_values[i] * sum_inv;
    }
}

__global__ void prepare_x_and_dangling_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ out_weights,
    float* __restrict__ x,
    float* __restrict__ dangling_sum_global,
    int num_vertices)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float my_dangling = 0.0f;

    if (tid < num_vertices) {
        float ow = out_weights[tid];
        float p = pr[tid];
        if (ow > 0.0f) {
            x[tid] = p / ow;
        } else {
            x[tid] = 0.0f;
            my_dangling = p;
        }
    }

    float block_sum = BlockReduce(temp_storage).Sum(my_dangling);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(dangling_sum_global, block_sum);
    }
}

__global__ void spmv_high_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x,
    const float* __restrict__ pers,
    const float* __restrict__ old_pr,
    float* __restrict__ new_pr,
    float* __restrict__ diff_sum,
    float alpha,
    const float* __restrict__ dangling_sum_ptr,
    float one_minus_alpha,
    int seg_start, int seg_end)
{
    typedef cub::BlockReduce<float, 512> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = seg_start + blockIdx.x;
    if (v >= seg_end) return;

    float base_factor = alpha * (*dangling_sum_ptr) + one_minus_alpha;

    int start = offsets[v];
    int end = offsets[v + 1];

    float sum = 0.0f;
    for (int j = start + threadIdx.x; j < end; j += blockDim.x) {
        if (is_edge_active(edge_mask, j)) {
            sum += edge_weights[j] * x[indices[j]];
        }
    }

    float total = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        float new_val = alpha * total + base_factor * pers[v];
        new_pr[v] = new_val;
        atomicAdd(diff_sum, fabsf(new_val - old_pr[v]));
    }
}

__global__ void spmv_mid_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x,
    const float* __restrict__ pers,
    const float* __restrict__ old_pr,
    float* __restrict__ new_pr,
    float* __restrict__ diff_sum,
    float alpha,
    const float* __restrict__ dangling_sum_ptr,
    float one_minus_alpha,
    int seg_start, int seg_end)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int v = seg_start + warp_id;
    if (v >= seg_end) return;

    float base_factor = alpha * (*dangling_sum_ptr) + one_minus_alpha;

    int start = offsets[v];
    int end = offsets[v + 1];

    float sum = 0.0f;
    for (int j = start + lane; j < end; j += 32) {
        if (is_edge_active(edge_mask, j)) {
            sum += edge_weights[j] * x[indices[j]];
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        float new_val = alpha * sum + base_factor * pers[v];
        new_pr[v] = new_val;
        atomicAdd(diff_sum, fabsf(new_val - old_pr[v]));
    }
}

__global__ void spmv_low_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x,
    const float* __restrict__ pers,
    const float* __restrict__ old_pr,
    float* __restrict__ new_pr,
    float* __restrict__ diff_sum,
    float alpha,
    const float* __restrict__ dangling_sum_ptr,
    float one_minus_alpha,
    int seg_start, int seg_end)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = seg_start + blockIdx.x * blockDim.x + threadIdx.x;

    float base_factor = alpha * (*dangling_sum_ptr) + one_minus_alpha;

    float d = 0.0f;
    if (v < seg_end) {
        int start = offsets[v];
        int end = offsets[v + 1];

        float sum = 0.0f;
        for (int j = start; j < end; j++) {
            if (is_edge_active(edge_mask, j)) {
                sum += edge_weights[j] * x[indices[j]];
            }
        }

        float new_val = alpha * sum + base_factor * pers[v];
        new_pr[v] = new_val;
        d = fabsf(new_val - old_pr[v]);
    }

    float block_sum = BlockReduce(temp_storage).Sum(d);
    if (threadIdx.x == 0 && block_sum > 0.0f) {
        atomicAdd(diff_sum, block_sum);
    }
}

__global__ void handle_zero_degree_kernel(
    const float* __restrict__ pers,
    const float* __restrict__ old_pr,
    float* __restrict__ new_pr,
    float* __restrict__ diff_sum,
    const float* __restrict__ dangling_sum_ptr,
    float alpha, float one_minus_alpha,
    int seg_start, int seg_end)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = seg_start + blockIdx.x * blockDim.x + threadIdx.x;

    float base_factor = alpha * (*dangling_sum_ptr) + one_minus_alpha;

    float d = 0.0f;
    if (v < seg_end) {
        float new_val = base_factor * pers[v];
        new_pr[v] = new_val;
        d = fabsf(new_val - old_pr[v]);
    }

    float block_sum = BlockReduce(temp_storage).Sum(d);
    if (threadIdx.x == 0 && block_sum > 0.0f) {
        atomicAdd(diff_sum, block_sum);
    }
}





void launch_compute_out_weight_sums(
    const int32_t* indices, const float* edge_weights, const uint32_t* edge_mask,
    float* out_weight_sums, int num_edges, cudaStream_t stream)
{
    if (num_edges <= 0) return;
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    compute_out_weight_sums_kernel<<<grid, block, 0, stream>>>(
        indices, edge_weights, edge_mask, out_weight_sums, num_edges);
}

void launch_init_pagerank(float* pr, int num_vertices, float init_val, cudaStream_t stream) {
    if (num_vertices <= 0) return;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    init_pagerank_kernel<<<grid, block, 0, stream>>>(pr, num_vertices, init_val);
}

void launch_build_pers_normalized(
    const int32_t* pers_vertices, const float* pers_values,
    float* pers_normalized, int pers_size, cudaStream_t stream)
{
    if (pers_size <= 0) return;
    build_pers_normalized_kernel<<<1, 64, 0, stream>>>(
        pers_vertices, pers_values, pers_normalized, pers_size);
}

void launch_prepare_x_and_dangling(
    const float* pr, const float* out_weights, float* x,
    float* dangling_sum_global, int num_vertices, cudaStream_t stream)
{
    if (num_vertices <= 0) return;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    prepare_x_and_dangling_kernel<<<grid, block, 0, stream>>>(
        pr, out_weights, x, dangling_sum_global, num_vertices);
}

void launch_spmv_high_degree(
    const int32_t* offsets, const int32_t* indices,
    const float* edge_weights, const uint32_t* edge_mask,
    const float* x, const float* pers, const float* old_pr,
    float* new_pr, float* diff_sum,
    float alpha, const float* dangling_sum_ptr, float one_minus_alpha,
    int seg_start, int seg_end, cudaStream_t stream)
{
    int num_verts = seg_end - seg_start;
    if (num_verts <= 0) return;
    spmv_high_degree_kernel<<<num_verts, 512, 0, stream>>>(
        offsets, indices, edge_weights, edge_mask,
        x, pers, old_pr, new_pr, diff_sum,
        alpha, dangling_sum_ptr, one_minus_alpha, seg_start, seg_end);
}

void launch_spmv_mid_degree(
    const int32_t* offsets, const int32_t* indices,
    const float* edge_weights, const uint32_t* edge_mask,
    const float* x, const float* pers, const float* old_pr,
    float* new_pr, float* diff_sum,
    float alpha, const float* dangling_sum_ptr, float one_minus_alpha,
    int seg_start, int seg_end, cudaStream_t stream)
{
    int num_verts = seg_end - seg_start;
    if (num_verts <= 0) return;
    int warps_needed = num_verts;
    int block = 128;
    int grid = (int)(((int64_t)warps_needed * 32 + block - 1) / block);
    spmv_mid_degree_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, edge_weights, edge_mask,
        x, pers, old_pr, new_pr, diff_sum,
        alpha, dangling_sum_ptr, one_minus_alpha, seg_start, seg_end);
}

void launch_spmv_low_degree(
    const int32_t* offsets, const int32_t* indices,
    const float* edge_weights, const uint32_t* edge_mask,
    const float* x, const float* pers, const float* old_pr,
    float* new_pr, float* diff_sum,
    float alpha, const float* dangling_sum_ptr, float one_minus_alpha,
    int seg_start, int seg_end, cudaStream_t stream)
{
    int num_verts = seg_end - seg_start;
    if (num_verts <= 0) return;
    int block = 256;
    int grid = (num_verts + block - 1) / block;
    spmv_low_degree_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, edge_weights, edge_mask,
        x, pers, old_pr, new_pr, diff_sum,
        alpha, dangling_sum_ptr, one_minus_alpha, seg_start, seg_end);
}

void launch_handle_zero_degree(
    const float* pers, const float* old_pr,
    float* new_pr, float* diff_sum,
    const float* dangling_sum_ptr,
    float alpha, float one_minus_alpha,
    int seg_start, int seg_end, cudaStream_t stream)
{
    int num_verts = seg_end - seg_start;
    if (num_verts <= 0) return;
    int block = 256;
    int grid = (num_verts + block - 1) / block;
    handle_zero_degree_kernel<<<grid, block, 0, stream>>>(
        pers, old_pr, new_pr, diff_sum,
        dangling_sum_ptr, alpha, one_minus_alpha, seg_start, seg_end);
}

}  

PageRankResult personalized_pagerank_seg_mask(const graph32_t& graph,
                                              const float* edge_weights,
                                              const int32_t* personalization_vertices,
                                              const float* personalization_values,
                                              std::size_t personalization_size,
                                              float* pageranks,
                                              const float* precomputed_vertex_out_weight_sums,
                                              float alpha,
                                              float epsilon,
                                              std::size_t max_iterations,
                                              const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    int pers_size = static_cast<int>(personalization_size);
    float one_minus_alpha = 1.0f - alpha;

    const auto& seg = graph.segment_offsets.value();
    int seg_arr[5] = {seg[0], seg[1], seg[2], seg[3], seg[4]};

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cudaStream_t stream = 0;

    cache.ensure(num_vertices);

    float* d_pr_a = pageranks;
    float* d_pr_b = cache.pr_b;
    float* d_x = cache.x;
    float* d_out_weights = cache.out_weights;
    float* d_pers = cache.pers;
    float* d_dangling_sum = cache.scalars;
    float* d_diff_sum = cache.scalars + 1;

    
    cudaMemsetAsync(d_out_weights, 0, num_vertices * sizeof(float), stream);
    launch_compute_out_weight_sums(d_indices, edge_weights, d_edge_mask, d_out_weights, num_edges, stream);

    
    cudaMemsetAsync(d_pers, 0, num_vertices * sizeof(float), stream);
    launch_build_pers_normalized(personalization_vertices, personalization_values, d_pers, pers_size, stream);

    
    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(d_pr_a, initial_pageranks,
                        num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        float init_val = 1.0f / num_vertices;
        launch_init_pagerank(d_pr_a, num_vertices, init_val, stream);
    }

    
    float* d_old_pr = d_pr_a;
    float* d_new_pr = d_pr_b;
    bool converged = false;
    std::size_t iterations = 0;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        cudaMemsetAsync(d_dangling_sum, 0, sizeof(float), stream);
        cudaMemsetAsync(d_diff_sum, 0, sizeof(float), stream);

        launch_prepare_x_and_dangling(d_old_pr, d_out_weights, d_x, d_dangling_sum, num_vertices, stream);

        launch_spmv_high_degree(d_offsets, d_indices, edge_weights, d_edge_mask,
            d_x, d_pers, d_old_pr, d_new_pr, d_diff_sum,
            alpha, d_dangling_sum, one_minus_alpha, seg_arr[0], seg_arr[1], stream);

        launch_spmv_mid_degree(d_offsets, d_indices, edge_weights, d_edge_mask,
            d_x, d_pers, d_old_pr, d_new_pr, d_diff_sum,
            alpha, d_dangling_sum, one_minus_alpha, seg_arr[1], seg_arr[2], stream);

        launch_spmv_low_degree(d_offsets, d_indices, edge_weights, d_edge_mask,
            d_x, d_pers, d_old_pr, d_new_pr, d_diff_sum,
            alpha, d_dangling_sum, one_minus_alpha, seg_arr[2], seg_arr[3], stream);

        launch_handle_zero_degree(d_pers, d_old_pr, d_new_pr, d_diff_sum,
            d_dangling_sum, alpha, one_minus_alpha, seg_arr[3], seg_arr[4], stream);

        cudaMemcpyAsync(cache.h_diff_pinned, d_diff_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations = iter + 1;

        float* tmp = d_old_pr;
        d_old_pr = d_new_pr;
        d_new_pr = tmp;

        if (*cache.h_diff_pinned < epsilon) {
            converged = true;
            break;
        }
    }

    
    if (d_old_pr != pageranks) {
        cudaMemcpyAsync(pageranks, d_old_pr,
                        (int64_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    return PageRankResult{iterations, converged};
}

}  
