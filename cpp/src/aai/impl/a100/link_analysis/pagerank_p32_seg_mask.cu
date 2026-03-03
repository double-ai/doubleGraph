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
#include <vector>
#include <cmath>

namespace aai {

namespace {


__device__ __forceinline__ bool is_edge_active(const uint32_t* edge_mask, int32_t edge_idx) {
    return (edge_mask[edge_idx >> 5] >> (edge_idx & 31)) & 1;
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}






__global__ void compute_out_degrees(
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ out_degree,
    int32_t num_edges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges && is_edge_active(edge_mask, tid)) {
        atomicAdd(&out_degree[indices[tid]], 1);
    }
}


__global__ void compute_inv_out_deg(
    const int32_t* __restrict__ out_degree,
    float* __restrict__ inv_out_deg,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int32_t deg = out_degree[v];
        inv_out_deg[v] = (deg > 0) ? (1.0f / (float)deg) : 0.0f;
    }
}


__global__ void build_pers(
    const int32_t* __restrict__ pers_verts,
    const float* __restrict__ pers_vals,
    float* __restrict__ pers_full,
    int32_t pers_size,
    float inv_sum
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pers_size) {
        pers_full[pers_verts[i]] = pers_vals[i] * inv_sum;
    }
}


__global__ void init_pr_uniform(float* __restrict__ pr, int32_t num_vertices, float init_val) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        pr[v] = init_val;
    }
}






__global__ void compute_x_and_dangling(
    float* __restrict__ x,
    const float* __restrict__ pr,
    const float* __restrict__ inv_out_deg,
    float* __restrict__ dangling_sum,
    int32_t num_vertices
) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float my_dang = 0.0f;

    if (v < num_vertices) {
        float p = pr[v];
        float inv_d = inv_out_deg[v];
        x[v] = p * inv_d;
        if (inv_d == 0.0f) {
            my_dang = p;
        }
    }

    float block_sum = BlockReduce(temp_storage).Sum(my_dang);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(dangling_sum, block_sum);
    }
}


__global__ void spmv_high_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x,
    float* __restrict__ result,
    int32_t seg_start,
    int32_t seg_end
) {
    int v = seg_start + blockIdx.x;
    if (v >= seg_end) return;

    int start = offsets[v];
    int end = offsets[v + 1];

    float sum = 0.0f;
    for (int j = start + threadIdx.x; j < end; j += blockDim.x) {
        if (is_edge_active(edge_mask, j)) {
            sum += x[indices[j]];
        }
    }

    typedef cub::BlockReduce<float, 512> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        result[v] = block_sum;
    }
}


__global__ void spmv_mid_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x,
    float* __restrict__ result,
    int32_t seg_start,
    int32_t seg_end
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int v = seg_start + warp_id;
    if (v >= seg_end) return;

    int start = offsets[v];
    int end = offsets[v + 1];

    float sum = 0.0f;
    for (int j = start + lane; j < end; j += 32) {
        if (is_edge_active(edge_mask, j)) {
            sum += x[indices[j]];
        }
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        result[v] = sum;
    }
}


__global__ void spmv_low_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x,
    float* __restrict__ result,
    int32_t seg_start,
    int32_t seg_end
) {
    int v = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= seg_end) return;

    int start = offsets[v];
    int end = offsets[v + 1];

    float sum = 0.0f;
    for (int j = start; j < end; j++) {
        if (is_edge_active(edge_mask, j)) {
            sum += x[indices[j]];
        }
    }

    result[v] = sum;
}



__global__ void update_pr_and_diff(
    float* __restrict__ spmv_and_new_pr,  
    const float* __restrict__ old_pr,
    const float* __restrict__ pers_full,
    const float* __restrict__ dangling_sum_ptr,
    float alpha,
    float one_minus_alpha,
    float* __restrict__ diff_sum,
    int32_t num_vertices
) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    
    float base_factor = alpha * (*dangling_sum_ptr) + one_minus_alpha;

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float my_diff = 0.0f;

    if (v < num_vertices) {
        float spmv_val = spmv_and_new_pr[v];
        float old_val = old_pr[v];
        float new_val = alpha * spmv_val + base_factor * pers_full[v];
        spmv_and_new_pr[v] = new_val;
        my_diff = fabsf(new_val - old_val);
    }

    float block_sum = BlockReduce(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0) {
        atomicAdd(diff_sum, block_sum);
    }
}





struct Cache : Cacheable {
    float* pr_a = nullptr;
    float* pr_b = nullptr;
    float* inv_out_deg = nullptr;
    float* pers_full = nullptr;
    float* x = nullptr;
    int32_t* out_degree = nullptr;
    float* scalars = nullptr;  
    int32_t capacity = 0;

    void ensure(int32_t num_vertices) {
        if (capacity < num_vertices) {
            if (pr_a) cudaFree(pr_a);
            if (pr_b) cudaFree(pr_b);
            if (inv_out_deg) cudaFree(inv_out_deg);
            if (pers_full) cudaFree(pers_full);
            if (x) cudaFree(x);
            if (out_degree) cudaFree(out_degree);
            if (scalars) cudaFree(scalars);

            cudaMalloc(&pr_a, num_vertices * sizeof(float));
            cudaMalloc(&pr_b, num_vertices * sizeof(float));
            cudaMalloc(&inv_out_deg, num_vertices * sizeof(float));
            cudaMalloc(&pers_full, num_vertices * sizeof(float));
            cudaMalloc(&x, num_vertices * sizeof(float));
            cudaMalloc(&out_degree, num_vertices * sizeof(int32_t));
            cudaMalloc(&scalars, 2 * sizeof(float));

            capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (inv_out_deg) cudaFree(inv_out_deg);
        if (pers_full) cudaFree(pers_full);
        if (x) cudaFree(x);
        if (out_degree) cudaFree(out_degree);
        if (scalars) cudaFree(scalars);
    }
};

}  

PageRankResult personalized_pagerank_seg_mask(const graph32_t& graph,
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
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;
    int32_t pers_size = static_cast<int32_t>(personalization_size);

    
    const auto& seg_vec = graph.segment_offsets.value();
    int32_t seg[5] = {seg_vec[0], seg_vec[1], seg_vec[2], seg_vec[3], seg_vec[4]};

    cudaStream_t stream = 0;

    
    cache.ensure(num_vertices);

    float* d_pr_a = cache.pr_a;
    float* d_pr_b = cache.pr_b;
    float* d_inv_out_deg = cache.inv_out_deg;
    float* d_pers_full = cache.pers_full;
    float* d_x = cache.x;
    int32_t* d_out_degree = cache.out_degree;
    float* d_dangling_sum = cache.scalars;
    float* d_diff_sum = cache.scalars + 1;

    

    
    cudaMemsetAsync(d_out_degree, 0, num_vertices * sizeof(int32_t), stream);
    {
        int block = 256;
        int grid = (num_edges + block - 1) / block;
        compute_out_degrees<<<grid, block, 0, stream>>>(d_indices, d_edge_mask, d_out_degree, num_edges);
    }

    
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        compute_inv_out_deg<<<grid, block, 0, stream>>>(d_out_degree, d_inv_out_deg, num_vertices);
    }

    
    cudaMemsetAsync(d_pers_full, 0, num_vertices * sizeof(float), stream);

    
    float h_pers_sum = 0.0f;
    {
        std::vector<float> h_pers_vals(pers_size);
        cudaMemcpy(h_pers_vals.data(), personalization_values, pers_size * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < pers_size; i++) h_pers_sum += h_pers_vals[i];
    }
    float inv_pers_sum = (h_pers_sum > 0.0f) ? (1.0f / h_pers_sum) : 0.0f;
    {
        int block = 256;
        int grid = (pers_size + block - 1) / block;
        build_pers<<<grid, block, 0, stream>>>(personalization_vertices, personalization_values, d_pers_full, pers_size, inv_pers_sum);
    }

    
    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(d_pr_a, initial_pageranks,
                       num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        init_pr_uniform<<<grid, block, 0, stream>>>(d_pr_a, num_vertices, 1.0f / num_vertices);
    }

    
    float one_minus_alpha = 1.0f - alpha;
    bool converged = false;
    std::size_t iterations = 0;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(cache.scalars, 0, 2 * sizeof(float), stream);

        
        {
            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            compute_x_and_dangling<<<grid, block, 0, stream>>>(d_x, d_pr_a, d_inv_out_deg, d_dangling_sum, num_vertices);
        }

        
        
        {
            int n = seg[1] - seg[0];
            if (n > 0)
                spmv_high_degree<<<n, 512, 0, stream>>>(d_offsets, d_indices, d_edge_mask, d_x, d_pr_b, seg[0], seg[1]);
        }
        
        {
            int n = seg[2] - seg[1];
            if (n > 0) {
                int warps_per_block = 8;
                int block = warps_per_block * 32;
                int grid = (n + warps_per_block - 1) / warps_per_block;
                spmv_mid_degree<<<grid, block, 0, stream>>>(d_offsets, d_indices, d_edge_mask, d_x, d_pr_b, seg[1], seg[2]);
            }
        }
        
        {
            int n = seg[4] - seg[2];
            if (n > 0) {
                int block = 256;
                int grid = (n + block - 1) / block;
                spmv_low_degree<<<grid, block, 0, stream>>>(d_offsets, d_indices, d_edge_mask, d_x, d_pr_b, seg[2], seg[4]);
            }
        }

        
        {
            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            update_pr_and_diff<<<grid, block, 0, stream>>>(d_pr_b, d_pr_a, d_pers_full, d_dangling_sum,
                                                           alpha, one_minus_alpha, d_diff_sum,
                                                           num_vertices);
        }

        
        float h_diff;
        cudaMemcpyAsync(&h_diff, d_diff_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        iterations = iter + 1;

        if (h_diff < epsilon) {
            converged = true;
            break;
        }

        
        std::swap(d_pr_a, d_pr_b);
    }

    
    
    
    float* d_final = converged ? d_pr_b : d_pr_a;

    
    cudaMemcpyAsync(pageranks, d_final,
                   num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return PageRankResult{iterations, converged};
}

}  
