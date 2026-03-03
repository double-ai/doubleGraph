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
#include <cmath>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* pr_a = nullptr;
    float* pr_b = nullptr;
    float* pr_scaled = nullptr;
    float* out_weight_sums = nullptr;
    float* dangling_sum = nullptr;
    float* diff = nullptr;

    int64_t pr_capacity = 0;
    int64_t out_w_capacity = 0;
    int64_t dangling_capacity = 0;
    int64_t diff_capacity = 0;

    void ensure(int32_t num_vertices) {
        int64_t nv = static_cast<int64_t>(num_vertices);
        if (pr_capacity < nv) {
            if (pr_a) cudaFree(pr_a);
            if (pr_b) cudaFree(pr_b);
            if (pr_scaled) cudaFree(pr_scaled);
            cudaMalloc(&pr_a, nv * sizeof(float));
            cudaMalloc(&pr_b, nv * sizeof(float));
            cudaMalloc(&pr_scaled, nv * sizeof(float));
            pr_capacity = nv;
        }
        if (out_w_capacity < nv) {
            if (out_weight_sums) cudaFree(out_weight_sums);
            cudaMalloc(&out_weight_sums, nv * sizeof(float));
            out_w_capacity = nv;
        }
        if (dangling_capacity < 1) {
            if (dangling_sum) cudaFree(dangling_sum);
            cudaMalloc(&dangling_sum, sizeof(float));
            dangling_capacity = 1;
        }
        if (diff_capacity < 1) {
            if (diff) cudaFree(diff);
            cudaMalloc(&diff, sizeof(float));
            diff_capacity = 1;
        }
    }

    ~Cache() override {
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (pr_scaled) cudaFree(pr_scaled);
        if (out_weight_sums) cudaFree(out_weight_sums);
        if (dangling_sum) cudaFree(dangling_sum);
        if (diff) cudaFree(diff);
    }
};




__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ out_weight_sums,
    int32_t num_edges)
{
    int32_t idx = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= num_edges) return;

    uint32_t word = edge_mask[idx >> 5];
    if (!((word >> (idx & 31)) & 1)) return;

    atomicAdd(&out_weight_sums[indices[idx]], edge_weights[idx]);
}




__global__ void init_pr_kernel(float* __restrict__ pr, float val, int32_t n)
{
    int32_t i = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) pr[i] = val;
}




__global__ void prepare_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ out_weight_sums,
    float* __restrict__ pr_scaled,
    float* __restrict__ d_dangling_sum,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int32_t v = (int32_t)(blockIdx.x * 256 + threadIdx.x);
    float dangling = 0.0f;

    if (v < num_vertices) {
        float ow = out_weight_sums[v];
        float p = pr[v];
        if (ow > 0.0f) {
            pr_scaled[v] = p / ow;
        } else {
            pr_scaled[v] = 0.0f;
            dangling = p;
        }
    }

    float block_sum = BlockReduce(temp).Sum(dangling);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(d_dangling_sum, block_sum);
    }
}




__global__ void spmv_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ pr_scaled,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    const float* __restrict__ d_dangling_sum,
    float alpha,
    float one_minus_alpha_over_n,
    float alpha_over_n,
    int32_t vertex_start,
    int32_t vertex_end,
    float* __restrict__ d_diff)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float s_base;

    if (threadIdx.x == 0) {
        s_base = one_minus_alpha_over_n + alpha_over_n * (*d_dangling_sum);
    }
    __syncthreads();

    int32_t v = vertex_start + (int32_t)(blockIdx.x * 256 + threadIdx.x);
    float my_diff = 0.0f;

    if (v < vertex_end) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float sum = 0.0f;
        for (int32_t e = start; e < end; e++) {
            uint32_t word = edge_mask[e >> 5];
            if ((word >> (e & 31)) & 1) {
                sum += pr_scaled[indices[e]] * edge_weights[e];
            }
        }

        float new_val = s_base + alpha * sum;
        pr_new[v] = new_val;
        my_diff = fabsf(new_val - pr_old[v]);
    }

    float block_diff = BlockReduce(temp).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff != 0.0f) {
        atomicAdd(d_diff, block_diff);
    }
}




__global__ void spmv_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ pr_scaled,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    const float* __restrict__ d_dangling_sum,
    float alpha,
    float one_minus_alpha_over_n,
    float alpha_over_n,
    int32_t vertex_start,
    int32_t vertex_end,
    float* __restrict__ d_diff)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float s_base;

    if (threadIdx.x == 0) {
        s_base = one_minus_alpha_over_n + alpha_over_n * (*d_dangling_sum);
    }
    __syncthreads();

    const int32_t global_thread = (int32_t)(blockIdx.x * 256 + threadIdx.x);
    const int32_t warp_id = global_thread >> 5;
    const int32_t lane = threadIdx.x & 31;
    const int32_t v = vertex_start + warp_id;

    float my_diff = 0.0f;

    if (v < vertex_end) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float sum = 0.0f;
        for (int32_t e = start + lane; e < end; e += 32) {
            uint32_t word = edge_mask[e >> 5];
            if ((word >> (e & 31)) & 1) {
                sum += pr_scaled[indices[e]] * edge_weights[e];
            }
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            float new_val = s_base + alpha * sum;
            pr_new[v] = new_val;
            my_diff = fabsf(new_val - pr_old[v]);
        }
    }

    float block_diff = BlockReduce(temp).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff != 0.0f) {
        atomicAdd(d_diff, block_diff);
    }
}




__global__ void spmv_block_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ pr_scaled,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    const float* __restrict__ d_dangling_sum,
    float alpha,
    float one_minus_alpha_over_n,
    float alpha_over_n,
    int32_t vertex_start,
    int32_t vertex_end,
    float* __restrict__ d_diff)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    const int32_t v = vertex_start + (int32_t)blockIdx.x;
    if (v >= vertex_end) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    float sum = 0.0f;
    for (int32_t e = start + (int32_t)threadIdx.x; e < end; e += 256) {
        uint32_t word = edge_mask[e >> 5];
        if ((word >> (e & 31)) & 1) {
            sum += pr_scaled[indices[e]] * edge_weights[e];
        }
    }

    float block_sum = BlockReduce(temp).Sum(sum);

    if (threadIdx.x == 0) {
        float base = one_minus_alpha_over_n + alpha_over_n * (*d_dangling_sum);
        float new_val = base + alpha * block_sum;
        pr_new[v] = new_val;
        atomicAdd(d_diff, fabsf(new_val - pr_old[v]));
    }
}

}  

PageRankResult pagerank_seg_mask(const graph32_t& graph,
                                 const float* edge_weights,
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

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg4 = seg[4];

    cudaStream_t stream = 0;

    float n_f = (float)num_vertices;
    float one_minus_alpha_over_n = (1.0f - alpha) / n_f;
    float alpha_over_n = alpha / n_f;

    cache.ensure(num_vertices);

    float* d_pr[2] = {cache.pr_a, cache.pr_b};
    float* d_pr_scaled = cache.pr_scaled;
    float* d_out_w = cache.out_weight_sums;
    float* d_dangling = cache.dangling_sum;
    float* d_diff = cache.diff;

    
    cudaMemsetAsync(d_out_w, 0, num_vertices * sizeof(float), stream);
    if (num_edges > 0) {
        int grid = (num_edges + 255) / 256;
        compute_out_weight_sums_kernel<<<grid, 256, 0, stream>>>(
            d_indices, edge_weights, d_edge_mask, d_out_w, num_edges);
    }

    
    float init_val = 1.0f / n_f;
    if (initial_pageranks) {
        cudaMemcpyAsync(d_pr[0], initial_pageranks,
                        num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        if (num_vertices > 0) {
            int grid = (num_vertices + 255) / 256;
            init_pr_kernel<<<grid, 256, 0, stream>>>(d_pr[0], init_val, num_vertices);
        }
    }

    
    int cur = 0;
    bool converged = false;
    std::size_t iterations = 0;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(d_dangling, 0, sizeof(float), stream);
        cudaMemsetAsync(d_diff, 0, sizeof(float), stream);

        
        if (num_vertices > 0) {
            int grid = (num_vertices + 255) / 256;
            prepare_kernel<<<grid, 256, 0, stream>>>(
                d_pr[cur], d_out_w, d_pr_scaled, d_dangling, num_vertices);
        }

        
        
        if (seg1 > seg0) {
            int32_t count = seg1 - seg0;
            spmv_block_kernel<<<count, 256, 0, stream>>>(
                d_offsets, d_indices, edge_weights, d_edge_mask,
                d_pr_scaled, d_pr[cur], d_pr[1-cur], d_dangling,
                alpha, one_minus_alpha_over_n, alpha_over_n,
                seg0, seg1, d_diff);
        }

        
        if (seg2 > seg1) {
            int32_t count = seg2 - seg1;
            int64_t num_threads = (int64_t)count * 32;
            int grid = (int)((num_threads + 255) / 256);
            spmv_warp_kernel<<<grid, 256, 0, stream>>>(
                d_offsets, d_indices, edge_weights, d_edge_mask,
                d_pr_scaled, d_pr[cur], d_pr[1-cur], d_dangling,
                alpha, one_minus_alpha_over_n, alpha_over_n,
                seg1, seg2, d_diff);
        }

        
        if (seg4 > seg2) {
            int32_t count = seg4 - seg2;
            int grid = (count + 255) / 256;
            spmv_thread_kernel<<<grid, 256, 0, stream>>>(
                d_offsets, d_indices, edge_weights, d_edge_mask,
                d_pr_scaled, d_pr[cur], d_pr[1-cur], d_dangling,
                alpha, one_minus_alpha_over_n, alpha_over_n,
                seg2, seg4, d_diff);
        }

        
        float h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);

        iterations = iter + 1;
        cur = 1 - cur;

        if (h_diff < epsilon) {
            converged = true;
            break;
        }
    }

    
    cudaMemcpyAsync(pageranks, d_pr[cur], num_vertices * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return PageRankResult{iterations, converged};
}

}  
