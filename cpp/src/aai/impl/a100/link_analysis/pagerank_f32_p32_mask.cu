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

struct Cache : Cacheable {
    
    int32_t* active_counts = nullptr;
    int32_t* new_offsets = nullptr;
    float* out_weight = nullptr;
    float* pers_norm = nullptr;
    float* pr_a = nullptr;
    float* pr_b = nullptr;
    float* x = nullptr;
    float* scalar_buf = nullptr;
    int64_t vertex_capacity = 0;

    
    int32_t* new_indices = nullptr;
    float* new_weights = nullptr;
    int64_t edge_capacity = 0;

    
    void* cub_temp = nullptr;
    size_t cub_temp_size = 0;

    void ensure_vertex_buffers(int32_t n) {
        if (vertex_capacity < n) {
            if (active_counts) cudaFree(active_counts);
            if (new_offsets) cudaFree(new_offsets);
            if (out_weight) cudaFree(out_weight);
            if (pers_norm) cudaFree(pers_norm);
            if (pr_a) cudaFree(pr_a);
            if (pr_b) cudaFree(pr_b);
            if (x) cudaFree(x);
            if (scalar_buf) cudaFree(scalar_buf);

            cudaMalloc(&active_counts, (size_t)n * sizeof(int32_t));
            cudaMalloc(&new_offsets, (size_t)(n + 1) * sizeof(int32_t));
            cudaMalloc(&out_weight, (size_t)n * sizeof(float));
            cudaMalloc(&pers_norm, (size_t)n * sizeof(float));
            cudaMalloc(&pr_a, (size_t)n * sizeof(float));
            cudaMalloc(&pr_b, (size_t)n * sizeof(float));
            cudaMalloc(&x, (size_t)n * sizeof(float));
            cudaMalloc(&scalar_buf, 2 * sizeof(float));

            vertex_capacity = n;
        }
    }

    void ensure_edge_buffers(int32_t ne) {
        if (edge_capacity < ne) {
            if (new_indices) cudaFree(new_indices);
            if (new_weights) cudaFree(new_weights);

            cudaMalloc(&new_indices, (size_t)ne * sizeof(int32_t));
            cudaMalloc(&new_weights, (size_t)ne * sizeof(float));

            edge_capacity = ne;
        }
    }

    void ensure_cub_temp(size_t bytes) {
        if (cub_temp_size < bytes) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, bytes);
            cub_temp_size = bytes;
        }
    }

    ~Cache() override {
        if (active_counts) cudaFree(active_counts);
        if (new_offsets) cudaFree(new_offsets);
        if (out_weight) cudaFree(out_weight);
        if (pers_norm) cudaFree(pers_norm);
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (x) cudaFree(x);
        if (scalar_buf) cudaFree(scalar_buf);
        if (new_indices) cudaFree(new_indices);
        if (new_weights) cudaFree(new_weights);
        if (cub_temp) cudaFree(cub_temp);
    }
};





__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_counts,
    int32_t num_vertices)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t count = 0;

    for (int32_t j = start; j < end; j++) {
        uint32_t word = edge_mask[j >> 5];
        if ((word >> (j & 31)) & 1) count++;
    }

    active_counts[v] = count;
}




__global__ void compact_edges_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const float* __restrict__ old_weights,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    float* __restrict__ new_weights,
    int32_t num_vertices)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int32_t old_start = old_offsets[v];
    int32_t old_end = old_offsets[v + 1];
    int32_t new_pos = new_offsets[v];

    for (int32_t j = old_start; j < old_end; j++) {
        uint32_t word = edge_mask[j >> 5];
        if ((word >> (j & 31)) & 1) {
            new_indices[new_pos] = old_indices[j];
            new_weights[new_pos] = old_weights[j];
            new_pos++;
        }
    }
}




__global__ void compute_out_weights_compact_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    float* __restrict__ out_weight,
    int32_t num_edges)
{
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_edges) return;
    atomicAdd(&out_weight[indices[j]], edge_weights[j]);
}




__global__ void init_pers_norm_kernel(
    const int32_t* __restrict__ pers_vertices,
    const float* __restrict__ pers_values,
    float* __restrict__ pers_norm,
    int32_t pers_size,
    float pers_sum_inv)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pers_size) return;
    pers_norm[pers_vertices[i]] = pers_values[i] * pers_sum_inv;
}




__global__ void compute_x_dangling_kernel(
    const float* __restrict__ pageranks,
    const float* __restrict__ out_weight,
    float* __restrict__ x,
    float* __restrict__ dangling_sum_out,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float dangling_val = 0.0f;

    if (idx < num_vertices) {
        float pr_val = pageranks[idx];
        float ow = out_weight[idx];
        if (ow == 0.0f) {
            x[idx] = 0.0f;
            dangling_val = pr_val;
        } else {
            x[idx] = pr_val / ow;
        }
    }

    float block_sum = BlockReduce(temp_storage).Sum(dangling_val);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(dangling_sum_out, block_sum);
    }
}





__global__ void spmv_l1_compact_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ x,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    float alpha,
    float one_minus_alpha,
    const float* __restrict__ dangling_sum_ptr,
    const float* __restrict__ pers_norm,
    float* __restrict__ l1_out,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float base_factor = alpha * (*dangling_sum_ptr) + one_minus_alpha;

    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0.0f;

    if (v < num_vertices) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float sum = 0.0f;
        for (int32_t j = start; j < end; j++) {
            sum += edge_weights[j] * x[indices[j]];
        }

        float new_val = alpha * sum + base_factor * pers_norm[v];
        pr_new[v] = new_val;
        diff = fabsf(new_val - pr_old[v]);
    }

    float block_sum = BlockReduce(temp_storage).Sum(diff);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(l1_out, block_sum);
    }
}




__global__ void init_uniform_kernel(float* __restrict__ pr, int32_t n, float val) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) pr[idx] = val;
}

__global__ void sum_values_kernel(
    const float* __restrict__ values, float* __restrict__ sum_out, int32_t n)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? values[idx] : 0.0f;
    float block_sum = BlockReduce(temp_storage).Sum(val);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(sum_out, block_sum);
    }
}

__global__ void zero_scalar_kernel(float* a) { *a = 0.0f; }
__global__ void zero_two_scalars_kernel(float* a, float* b) { *a = 0.0f; *b = 0.0f; }

}  

PageRankResult personalized_pagerank_mask(const graph32_t& graph,
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
    int32_t pers_size = static_cast<int32_t>(personalization_size);
    float one_minus_alpha = 1.0f - alpha;
    cudaStream_t stream = 0;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const float* d_edge_weights = edge_weights;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cache.ensure_vertex_buffers(num_vertices);

    int block = 256;
    int grid = (num_vertices + block - 1) / block;

    
    
    
    count_active_edges_kernel<<<grid, block, 0, stream>>>(
        d_offsets, d_edge_mask, cache.active_counts, num_vertices);

    
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes,
        cache.active_counts, cache.new_offsets, num_vertices, stream);

    cache.ensure_cub_temp(temp_bytes);

    
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, temp_bytes,
        cache.active_counts, cache.new_offsets, num_vertices, stream);

    
    int32_t h_last_count = 0, h_last_offset = 0;
    cudaMemcpyAsync(&h_last_count, cache.active_counts + num_vertices - 1,
                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_last_offset, cache.new_offsets + num_vertices - 1,
                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int32_t num_active_edges = h_last_offset + h_last_count;

    
    cudaMemcpyAsync(cache.new_offsets + num_vertices, &num_active_edges,
                    sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    
    cache.ensure_edge_buffers(num_active_edges > 0 ? num_active_edges : 1);

    
    compact_edges_kernel<<<grid, block, 0, stream>>>(
        d_offsets, d_indices, d_edge_weights, d_edge_mask,
        cache.new_offsets, cache.new_indices, cache.new_weights,
        num_vertices);

    
    
    
    cudaMemsetAsync(cache.out_weight, 0, num_vertices * sizeof(float), stream);
    if (num_active_edges > 0) {
        int egrid = (num_active_edges + block - 1) / block;
        compute_out_weights_compact_kernel<<<egrid, block, 0, stream>>>(
            cache.new_indices, cache.new_weights, cache.out_weight, num_active_edges);
    }

    
    
    
    cudaMemsetAsync(cache.pers_norm, 0, num_vertices * sizeof(float), stream);

    float* d_dangling_sum = cache.scalar_buf;
    float* d_l1 = cache.scalar_buf + 1;

    zero_scalar_kernel<<<1, 1, 0, stream>>>(d_dangling_sum);
    if (pers_size > 0) {
        int pgrid = (pers_size + block - 1) / block;
        sum_values_kernel<<<pgrid, block, 0, stream>>>(
            personalization_values, d_dangling_sum, pers_size);
    }

    float h_pers_sum = 0.0f;
    cudaMemcpyAsync(&h_pers_sum, d_dangling_sum, sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float pers_sum_inv = (h_pers_sum > 0.0f) ? (1.0f / h_pers_sum) : 0.0f;
    if (pers_size > 0) {
        int pgrid = (pers_size + block - 1) / block;
        init_pers_norm_kernel<<<pgrid, block, 0, stream>>>(
            personalization_vertices, personalization_values, cache.pers_norm,
            pers_size, pers_sum_inv);
    }

    
    
    
    float* d_pr_cur = cache.pr_a;
    float* d_pr_next = cache.pr_b;

    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(d_pr_cur, initial_pageranks,
                        num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        float init_val = 1.0f / num_vertices;
        init_uniform_kernel<<<grid, block, 0, stream>>>(d_pr_cur, num_vertices, init_val);
    }

    
    
    
    bool converged = false;
    size_t iterations = 0;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        zero_two_scalars_kernel<<<1, 1, 0, stream>>>(d_dangling_sum, d_l1);

        compute_x_dangling_kernel<<<grid, block, 0, stream>>>(
            d_pr_cur, cache.out_weight, cache.x, d_dangling_sum, num_vertices);

        spmv_l1_compact_kernel<<<grid, block, 0, stream>>>(
            cache.new_offsets, cache.new_indices, cache.new_weights, cache.x,
            d_pr_cur, d_pr_next,
            alpha, one_minus_alpha, d_dangling_sum,
            cache.pers_norm, d_l1, num_vertices);

        float h_l1 = 0.0f;
        cudaMemcpyAsync(&h_l1, d_l1, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations = iter + 1;

        float* tmp = d_pr_cur;
        d_pr_cur = d_pr_next;
        d_pr_next = tmp;

        if (h_l1 < epsilon) {
            converged = true;
            break;
        }
    }

    
    cudaMemcpyAsync(pageranks, d_pr_cur,
                    num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return PageRankResult{iterations, converged};
}

}  
