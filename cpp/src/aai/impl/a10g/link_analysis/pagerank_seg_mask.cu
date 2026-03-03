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
    float* pr_temp = nullptr;
    float* out_degree = nullptr;
    float* inv_out_degree = nullptr;
    float* pr_scaled = nullptr;
    float* dangling_sum = nullptr;
    float* l1_norm = nullptr;
    float* partials = nullptr;
    unsigned int* retire_cnt = nullptr;

    int64_t pr_temp_capacity = 0;
    int64_t out_degree_capacity = 0;
    int64_t inv_out_degree_capacity = 0;
    int64_t pr_scaled_capacity = 0;
    int64_t dangling_sum_capacity = 0;
    int64_t l1_norm_capacity = 0;
    int64_t partials_capacity = 0;
    int64_t retire_cnt_capacity = 0;

    void ensure(int32_t num_vertices) {
        if (pr_temp_capacity < num_vertices) {
            if (pr_temp) cudaFree(pr_temp);
            cudaMalloc(&pr_temp, num_vertices * sizeof(float));
            pr_temp_capacity = num_vertices;
        }
        if (out_degree_capacity < num_vertices) {
            if (out_degree) cudaFree(out_degree);
            cudaMalloc(&out_degree, num_vertices * sizeof(float));
            out_degree_capacity = num_vertices;
        }
        if (inv_out_degree_capacity < num_vertices) {
            if (inv_out_degree) cudaFree(inv_out_degree);
            cudaMalloc(&inv_out_degree, num_vertices * sizeof(float));
            inv_out_degree_capacity = num_vertices;
        }
        if (pr_scaled_capacity < num_vertices) {
            if (pr_scaled) cudaFree(pr_scaled);
            cudaMalloc(&pr_scaled, num_vertices * sizeof(float));
            pr_scaled_capacity = num_vertices;
        }
        if (dangling_sum_capacity < 1) {
            if (dangling_sum) cudaFree(dangling_sum);
            cudaMalloc(&dangling_sum, sizeof(float));
            dangling_sum_capacity = 1;
        }
        if (l1_norm_capacity < 1) {
            if (l1_norm) cudaFree(l1_norm);
            cudaMalloc(&l1_norm, sizeof(float));
            l1_norm_capacity = 1;
        }
        if (partials_capacity < 1024) {
            if (partials) cudaFree(partials);
            cudaMalloc(&partials, 1024 * sizeof(float));
            partials_capacity = 1024;
        }
        if (retire_cnt_capacity < 1) {
            if (retire_cnt) cudaFree(retire_cnt);
            cudaMalloc(&retire_cnt, sizeof(unsigned int));
            retire_cnt_capacity = 1;
        }
    }

    ~Cache() override {
        if (pr_temp) cudaFree(pr_temp);
        if (out_degree) cudaFree(out_degree);
        if (inv_out_degree) cudaFree(inv_out_degree);
        if (pr_scaled) cudaFree(pr_scaled);
        if (dangling_sum) cudaFree(dangling_sum);
        if (l1_norm) cudaFree(l1_norm);
        if (partials) cudaFree(partials);
        if (retire_cnt) cudaFree(retire_cnt);
    }
};

__device__ __forceinline__ bool is_edge_active(const uint32_t* __restrict__ edge_mask, int32_t edge_idx) {
    return (edge_mask[edge_idx >> 5] >> (edge_idx & 31)) & 1;
}

__global__ void fill_kernel(float* __restrict__ arr, float val, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = val;
}

__global__ void compute_out_degrees_kernel(
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ out_degree,
    int32_t num_edges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges && is_edge_active(edge_mask, tid)) {
        atomicAdd(&out_degree[indices[tid]], 1.0f);
    }
}

__global__ void compute_inv_out_degree_kernel(
    const float* __restrict__ out_degree,
    float* __restrict__ inv_out_degree,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        float deg = out_degree[v];
        inv_out_degree[v] = (deg > 0.0f) ? (1.0f / deg) : 0.0f;
    }
}


__global__ void __launch_bounds__(256)
prepare_and_dangling_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ inv_out_degree,
    float* __restrict__ pr_scaled,
    float* __restrict__ d_dangling_sum,
    float* __restrict__ d_l1_norm,        
    float* __restrict__ partials,
    unsigned int* __restrict__ retirement_count,
    int32_t num_vertices
) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float dangling_local = 0.0f;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += gridDim.x * blockDim.x) {
        float p = pr[v];
        float inv_deg = inv_out_degree[v];
        pr_scaled[v] = p * inv_deg;
        if (inv_deg == 0.0f) dangling_local += p;
    }

    float block_sum = BlockReduce(temp_storage).Sum(dangling_local);
    if (threadIdx.x == 0) partials[blockIdx.x] = block_sum;
    __threadfence();

    __shared__ bool am_last;
    __shared__ int s_num_blocks;
    if (threadIdx.x == 0) {
        s_num_blocks = gridDim.x;
        unsigned int ticket = atomicAdd(retirement_count, 1);
        am_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (am_last) {
        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < s_num_blocks; i += blockDim.x) {
            local_sum += partials[i];
        }
        float final_sum = BlockReduce(temp_storage).Sum(local_sum);
        if (threadIdx.x == 0) {
            d_dangling_sum[0] = final_sum;
            d_l1_norm[0] = 0.0f;  
            *retirement_count = 0;
        }
    }
}


__global__ void __launch_bounds__(256)
spmv_block_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ pr_scaled,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    float base_pr, float alpha, float alpha_over_n,
    const float* __restrict__ d_dangling_sum,
    float* __restrict__ d_l1_norm,
    int32_t start_vertex, int32_t end_vertex
) {
    int v = blockIdx.x + start_vertex;
    if (v >= end_vertex) return;

    __shared__ float s_base_value;
    if (threadIdx.x == 0) s_base_value = base_pr + alpha_over_n * d_dangling_sum[0];
    __syncthreads();

    int row_start = offsets[v];
    int row_end = offsets[v + 1];
    float sum = 0.0f;
    for (int e = row_start + (int)threadIdx.x; e < row_end; e += 256) {
        if (is_edge_active(edge_mask, e)) sum += pr_scaled[indices[e]];
    }

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        float new_val = s_base_value + alpha * block_sum;
        pr_new[v] = new_val;
        atomicAdd(d_l1_norm, fabsf(new_val - pr_old[v]));
    }
}


__global__ void __launch_bounds__(256)
spmv_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ pr_scaled,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    float base_pr, float alpha, float alpha_over_n,
    const float* __restrict__ d_dangling_sum,
    float* __restrict__ d_l1_norm,
    int32_t start_vertex, int32_t end_vertex
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid >> 5;
    int lane = global_tid & 31;
    int v = start_vertex + warp_id;

    float base_value = 0.0f;
    float diff = 0.0f;

    if (v < end_vertex) {
        if (lane == 0) base_value = base_pr + alpha_over_n * d_dangling_sum[0];
        base_value = __shfl_sync(0xffffffff, base_value, 0);

        int row_start = offsets[v];
        int row_end = offsets[v + 1];
        float sum = 0.0f;
        for (int e = row_start + lane; e < row_end; e += 32) {
            if (is_edge_active(edge_mask, e)) sum += pr_scaled[indices[e]];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        if (lane == 0) {
            float new_val = base_value + alpha * sum;
            pr_new[v] = new_val;
            diff = fabsf(new_val - pr_old[v]);
        }
    }

    
    
    diff = __shfl_sync(0xffffffff, diff, 0); 
    
    if (lane != 0) diff = 0.0f;

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_diff = BlockReduce(temp_storage).Sum(diff);
    if (threadIdx.x == 0) atomicAdd(d_l1_norm, block_diff);
}


__global__ void __launch_bounds__(256)
spmv_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ pr_scaled,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    float base_pr, float alpha, float alpha_over_n,
    const float* __restrict__ d_dangling_sum,
    float* __restrict__ d_l1_norm,
    int32_t start_vertex, int32_t end_vertex
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x + start_vertex;
    float base_value = base_pr + alpha_over_n * d_dangling_sum[0];

    float diff = 0.0f;
    if (v < end_vertex) {
        int row_start = offsets[v];
        int row_end = offsets[v + 1];
        float sum = 0.0f;
        for (int e = row_start; e < row_end; e++) {
            if (is_edge_active(edge_mask, e)) sum += pr_scaled[indices[e]];
        }
        float new_val = base_value + alpha * sum;
        pr_new[v] = new_val;
        diff = fabsf(new_val - pr_old[v]);
    }

    
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_diff = BlockReduce(temp_storage).Sum(diff);
    if (threadIdx.x == 0) atomicAdd(d_l1_norm, block_diff);
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

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2];
    int32_t seg3 = seg[3], seg4 = seg[4];

    cache.ensure(num_vertices);

    float* d_pr_temp = cache.pr_temp;
    float* d_out_degree = cache.out_degree;
    float* d_inv_out_degree = cache.inv_out_degree;
    float* d_pr_scaled = cache.pr_scaled;
    float* d_dangling_sum = cache.dangling_sum;
    float* d_l1_norm = cache.l1_norm;
    float* d_partials = cache.partials;
    unsigned int* d_retire_cnt = cache.retire_cnt;

    cudaStream_t stream = 0;
    const int BLOCK = 256;
    int grid_v = (num_vertices + BLOCK - 1) / BLOCK;

    
    cudaMemsetAsync(d_out_degree, 0, num_vertices * sizeof(float), stream);
    if (num_edges > 0) {
        int grid = (num_edges + BLOCK - 1) / BLOCK;
        compute_out_degrees_kernel<<<grid, BLOCK, 0, stream>>>(
            d_indices, d_edge_mask, d_out_degree, num_edges);
    }

    
    compute_inv_out_degree_kernel<<<grid_v, BLOCK, 0, stream>>>(
        d_out_degree, d_inv_out_degree, num_vertices);

    
    bool has_initial_guess = (initial_pageranks != nullptr);
    if (has_initial_guess) {
        cudaMemcpyAsync(pageranks, initial_pageranks,
                        num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        float init_val = 1.0f / (float)num_vertices;
        fill_kernel<<<grid_v, BLOCK, 0, stream>>>(pageranks, init_val, num_vertices);
    }

    cudaMemsetAsync(d_retire_cnt, 0, sizeof(unsigned int), stream);
    
    cudaMemsetAsync(d_l1_norm, 0, sizeof(float), stream);

    float base_pr = (1.0f - alpha) / (float)num_vertices;
    float alpha_over_n = alpha / (float)num_vertices;

    int prepare_blocks = grid_v;
    if (prepare_blocks > 1024) prepare_blocks = 1024;

    float* d_src = pageranks;
    float* d_dst = d_pr_temp;
    bool converged = false;
    std::size_t iterations = 0;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        prepare_and_dangling_kernel<<<prepare_blocks, BLOCK, 0, stream>>>(
            d_src, d_inv_out_degree, d_pr_scaled, d_dangling_sum, d_l1_norm,
            d_partials, d_retire_cnt, num_vertices);

        
        if (seg1 > seg0) {
            spmv_block_kernel<<<seg1 - seg0, BLOCK, 0, stream>>>(
                d_offsets, d_indices, d_edge_mask, d_pr_scaled, d_src, d_dst,
                base_pr, alpha, alpha_over_n, d_dangling_sum, d_l1_norm, seg0, seg1);
        }
        if (seg2 > seg1) {
            int num_mid = seg2 - seg1;
            int grid = (int)(((int64_t)num_mid * 32 + BLOCK - 1) / BLOCK);
            spmv_warp_kernel<<<grid, BLOCK, 0, stream>>>(
                d_offsets, d_indices, d_edge_mask, d_pr_scaled, d_src, d_dst,
                base_pr, alpha, alpha_over_n, d_dangling_sum, d_l1_norm, seg1, seg2);
        }
        if (seg4 > seg2) {
            int num_verts = seg4 - seg2;
            int grid = (num_verts + BLOCK - 1) / BLOCK;
            spmv_thread_kernel<<<grid, BLOCK, 0, stream>>>(
                d_offsets, d_indices, d_edge_mask, d_pr_scaled, d_src, d_dst,
                base_pr, alpha, alpha_over_n, d_dangling_sum, d_l1_norm, seg2, seg4);
        }

        float* t = d_src; d_src = d_dst; d_dst = t;
        iterations = iter + 1;

        
        float h_l1_norm;
        cudaMemcpyAsync(&h_l1_norm, d_l1_norm, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (h_l1_norm < epsilon) { converged = true; break; }
    }

    if (d_src != pageranks) {
        cudaMemcpyAsync(pageranks, d_src, num_vertices * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    return PageRankResult{iterations, converged};
}

}  
