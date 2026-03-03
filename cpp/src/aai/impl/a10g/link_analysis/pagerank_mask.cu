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





__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ counts,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int start = offsets[v];
    int end = offsets[v + 1];
    int count = 0;
    for (int e = start; e < end; e++) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1u)
            count++;
    }
    counts[v] = count;
}

__global__ void compact_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ compact_offsets,
    int32_t* __restrict__ compact_indices,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int start = offsets[v];
    int end = offsets[v + 1];
    int out = compact_offsets[v];
    for (int e = start; e < end; e++) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1u)
            compact_indices[out++] = indices[e];
    }
}

__global__ void compute_out_degree_kernel(
    const int32_t* __restrict__ compact_indices,
    float* __restrict__ out_degree,
    int32_t num_compact_edges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_compact_edges) return;
    atomicAdd(&out_degree[compact_indices[e]], 1.0f);
}





__global__ void init_pagerank_kernel(float* __restrict__ pr, int32_t N)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    pr[v] = 1.0f / (float)N;
}

__global__ void compute_pr_norm_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ out_degree,
    float* __restrict__ pr_norm,
    float* __restrict__ dangling_contrib,
    int32_t N)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;

    float od = out_degree[v];
    float p = pr[v];
    if (od > 0.0f) {
        pr_norm[v] = p / od;
        dangling_contrib[v] = 0.0f;
    } else {
        pr_norm[v] = 0.0f;
        dangling_contrib[v] = p;
    }
}

__global__ void spmv_warp_per_row_kernel(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ col_indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t num_rows)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_rows) return;

    int start = row_offsets[warp_id];
    int end = row_offsets[warp_id + 1];

    float sum = 0.0f;
    for (int e = start + lane; e < end; e += 32) {
        sum += x[col_indices[e]];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[warp_id] = sum;
    }
}

__global__ void spmv_thread_per_row_kernel(
    const int32_t* __restrict__ row_offsets,
    const int32_t* __restrict__ col_indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t num_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    int start = row_offsets[row];
    int end = row_offsets[row + 1];

    float sum = 0.0f;
    for (int e = start; e < end; e++) {
        sum += x[col_indices[e]];
    }
    y[row] = sum;
}

__global__ void apply_teleport_fused_diff_kernel(
    const float* __restrict__ spmv_result,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    const float* __restrict__ d_dangling_sum,
    float* __restrict__ d_l1_diff,
    float alpha,
    float one_minus_alpha_over_n,
    int32_t N)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    float diff_val = 0.0f;
    if (v < N) {
        float dangling_contribution = alpha * __ldg(d_dangling_sum) / (float)N;
        float new_val = one_minus_alpha_over_n + alpha * spmv_result[v] + dangling_contribution;
        pr_new[v] = new_val;
        diff_val = fabsf(new_val - pr_old[v]);
    }

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(diff_val);

    if (threadIdx.x == 0) {
        atomicAdd(d_l1_diff, block_sum);
    }
}

__global__ void set_last_offset_kernel(int32_t* compact_offsets, int32_t N, int32_t total)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        compact_offsets[N] = total;
    }
}





struct Cache : Cacheable {
    
    int32_t* compact_counts = nullptr;
    int32_t* compact_offsets = nullptr;
    int32_t* compact_indices = nullptr;
    int64_t compact_counts_capacity = 0;
    int64_t compact_offsets_capacity = 0;
    int64_t compact_indices_capacity = 0;

    
    void* cub_temp = nullptr;
    int64_t cub_temp_capacity = 0;

    
    float* out_degree = nullptr;
    int64_t out_degree_capacity = 0;

    
    float* pr_norm = nullptr;
    float* spmv_result = nullptr;
    float* pr_temp = nullptr;
    float* dangling_contrib = nullptr;
    float* scalar_buf = nullptr;
    int64_t work_capacity = 0;
    int64_t scalar_buf_capacity = 0;

    void ensure_compaction(int64_t N, int64_t cub_bytes) {
        if (compact_counts_capacity < N) {
            if (compact_counts) cudaFree(compact_counts);
            cudaMalloc(&compact_counts, N * sizeof(int32_t));
            compact_counts_capacity = N;
        }
        if (compact_offsets_capacity < N + 1) {
            if (compact_offsets) cudaFree(compact_offsets);
            cudaMalloc(&compact_offsets, (N + 1) * sizeof(int32_t));
            compact_offsets_capacity = N + 1;
        }
        if (cub_temp_capacity < cub_bytes) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, cub_bytes);
            cub_temp_capacity = cub_bytes;
        }
    }

    void ensure_compact_indices(int64_t num_edges) {
        if (compact_indices_capacity < num_edges) {
            if (compact_indices) cudaFree(compact_indices);
            cudaMalloc(&compact_indices, num_edges * sizeof(int32_t));
            compact_indices_capacity = num_edges;
        }
    }

    void ensure_work_buffers(int64_t N) {
        if (out_degree_capacity < N) {
            if (out_degree) cudaFree(out_degree);
            cudaMalloc(&out_degree, N * sizeof(float));
            out_degree_capacity = N;
        }
        if (work_capacity < N) {
            if (pr_norm) cudaFree(pr_norm);
            if (spmv_result) cudaFree(spmv_result);
            if (pr_temp) cudaFree(pr_temp);
            if (dangling_contrib) cudaFree(dangling_contrib);
            cudaMalloc(&pr_norm, N * sizeof(float));
            cudaMalloc(&spmv_result, N * sizeof(float));
            cudaMalloc(&pr_temp, N * sizeof(float));
            cudaMalloc(&dangling_contrib, N * sizeof(float));
            work_capacity = N;
        }
        if (scalar_buf_capacity < 4) {
            if (scalar_buf) cudaFree(scalar_buf);
            cudaMalloc(&scalar_buf, 4 * sizeof(float));
            scalar_buf_capacity = 4;
        }
    }

    ~Cache() override {
        if (compact_counts) cudaFree(compact_counts);
        if (compact_offsets) cudaFree(compact_offsets);
        if (compact_indices) cudaFree(compact_indices);
        if (cub_temp) cudaFree(cub_temp);
        if (out_degree) cudaFree(out_degree);
        if (pr_norm) cudaFree(pr_norm);
        if (spmv_result) cudaFree(spmv_result);
        if (pr_temp) cudaFree(pr_temp);
        if (dangling_contrib) cudaFree(dangling_contrib);
        if (scalar_buf) cudaFree(scalar_buf);
    }
};

}  

PageRankResult pagerank_mask(const graph32_t& graph,
                             float* pageranks,
                             const float* precomputed_vertex_out_weight_sums,
                             float alpha,
                             float epsilon,
                             std::size_t max_iterations,
                             const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t N = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;
    cudaStream_t stream = 0;

    
    size_t scan_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes,
        (const int32_t*)nullptr, (int32_t*)nullptr, (int)N);
    size_t reduce_temp_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, reduce_temp_bytes,
        (const float*)nullptr, (float*)nullptr, (int)N);
    size_t cub_temp_bytes = std::max(scan_temp_bytes, reduce_temp_bytes);
    if (cub_temp_bytes < 256) cub_temp_bytes = 256;

    
    cache.ensure_compaction(N, cub_temp_bytes);

    int32_t* d_compact_counts = cache.compact_counts;
    int32_t* d_compact_offsets = cache.compact_offsets;

    
    {
        int block = 256;
        int grid = (N + block - 1) / block;
        if (grid > 0)
            count_active_edges_kernel<<<grid, block, 0, stream>>>(
                d_offsets, d_edge_mask, d_compact_counts, N);
    }

    
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, cub_temp_bytes,
        d_compact_counts, d_compact_offsets, N, stream);

    
    int32_t h_last_offset = 0, h_last_count = 0;
    if (N > 0) {
        cudaMemcpyAsync(&h_last_offset, d_compact_offsets + N - 1,
            sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&h_last_count, d_compact_counts + N - 1,
            sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    int32_t total_compact_edges = h_last_offset + h_last_count;
    set_last_offset_kernel<<<1, 1, 0, stream>>>(d_compact_offsets, N, total_compact_edges);

    
    int32_t compact_alloc = (total_compact_edges > 0) ? total_compact_edges : 1;
    cache.ensure_compact_indices(compact_alloc);
    int32_t* d_compact_indices = cache.compact_indices;
    {
        int block = 256;
        int grid = (N + block - 1) / block;
        if (grid > 0)
            compact_edges_kernel<<<grid, block, 0, stream>>>(
                d_offsets, d_indices, d_edge_mask, d_compact_offsets,
                d_compact_indices, N);
    }

    
    cache.ensure_work_buffers(N);

    float* d_out_degree = cache.out_degree;
    cudaMemsetAsync(d_out_degree, 0, N * sizeof(float), stream);
    if (total_compact_edges > 0) {
        int block = 256;
        int grid = (total_compact_edges + block - 1) / block;
        if (grid > 0)
            compute_out_degree_kernel<<<grid, block, 0, stream>>>(
                d_compact_indices, d_out_degree, total_compact_edges);
    }

    
    float avg_degree = (N > 0) ? (float)total_compact_edges / (float)N : 0.0f;
    bool use_warp_spmv = (avg_degree >= 8.0f);

    float* d_pr_norm = cache.pr_norm;
    float* d_spmv_result = cache.spmv_result;
    float* d_pr_temp = cache.pr_temp;
    float* d_dangling_contrib = cache.dangling_contrib;
    float* d_dangling_sum = cache.scalar_buf;
    float* d_l1_diff = cache.scalar_buf + 1;

    
    float* d_pr_a = pageranks;
    float* d_pr_b = d_pr_temp;

    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(d_pr_a, initial_pageranks,
            N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        int block = 256;
        int grid = (N + block - 1) / block;
        if (grid > 0)
            init_pagerank_kernel<<<grid, block, 0, stream>>>(d_pr_a, N);
    }

    
    float one_minus_alpha_over_n = (1.0f - alpha) / (float)N;
    float h_l1_diff;
    std::size_t iteration = 0;
    bool converged = false;

    for (; iteration < max_iterations; iteration++) {
        cudaMemsetAsync(d_l1_diff, 0, sizeof(float), stream);

        
        {
            int block = 256;
            int grid = (N + block - 1) / block;
            if (grid > 0)
                compute_pr_norm_kernel<<<grid, block, 0, stream>>>(
                    d_pr_a, d_out_degree, d_pr_norm, d_dangling_contrib, N);
        }

        
        cub::DeviceReduce::Sum(cache.cub_temp, cub_temp_bytes,
            d_dangling_contrib, d_dangling_sum, N, stream);

        
        if (total_compact_edges > 0) {
            if (use_warp_spmv) {
                int threads = 256;
                int warps_per_block = threads / 32;
                int grid = (N + warps_per_block - 1) / warps_per_block;
                if (grid > 0)
                    spmv_warp_per_row_kernel<<<grid, threads, 0, stream>>>(
                        d_compact_offsets, d_compact_indices, d_pr_norm,
                        d_spmv_result, N);
            } else {
                int block = 256;
                int grid = (N + block - 1) / block;
                if (grid > 0)
                    spmv_thread_per_row_kernel<<<grid, block, 0, stream>>>(
                        d_compact_offsets, d_compact_indices, d_pr_norm,
                        d_spmv_result, N);
            }
        } else {
            cudaMemsetAsync(d_spmv_result, 0, N * sizeof(float), stream);
        }

        
        {
            int block = 256;
            int grid = (N + block - 1) / block;
            if (grid > 0)
                apply_teleport_fused_diff_kernel<<<grid, block, 0, stream>>>(
                    d_spmv_result, d_pr_a, d_pr_b,
                    d_dangling_sum, d_l1_diff, alpha,
                    one_minus_alpha_over_n, N);
        }

        
        cudaMemcpyAsync(&h_l1_diff, d_l1_diff, sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        float* temp = d_pr_a; d_pr_a = d_pr_b; d_pr_b = temp;

        if (h_l1_diff < epsilon) {
            converged = true;
            iteration++;
            break;
        }
    }

    
    if (d_pr_a != pageranks) {
        cudaMemcpyAsync(pageranks, d_pr_a,
            N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    return PageRankResult{iteration, converged};
}

}  
