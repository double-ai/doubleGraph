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
#include <algorithm>
#include <vector>

namespace aai {

namespace {

struct Cache : Cacheable {
    void* workspace = nullptr;
    size_t workspace_capacity = 0;

    void ensure(size_t needed) {
        if (needed > workspace_capacity) {
            if (workspace) cudaFree(workspace);
            cudaMalloc(&workspace, needed);
            workspace_capacity = needed;
        }
    }

    ~Cache() override {
        if (workspace) {
            cudaFree(workspace);
            workspace = nullptr;
        }
    }
};


__device__ __forceinline__ bool is_edge_active(const uint32_t* mask, int32_t idx) {
    return (mask[idx >> 5] >> (idx & 31)) & 1u;
}



__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ counts,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int start = offsets[v];
    int end = offsets[v + 1];
    int cnt = 0;
    for (int j = start; j < end; j++) {
        if (is_edge_active(edge_mask, j)) cnt++;
    }
    counts[v] = cnt;
}

__global__ void set_int32_kernel(int32_t* ptr, int32_t val) {
    *ptr = val;
}

__global__ void compact_csc_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int old_start = old_offsets[v];
    int old_end = old_offsets[v + 1];
    int new_pos = new_offsets[v];
    for (int j = old_start; j < old_end; j++) {
        if (is_edge_active(edge_mask, j)) {
            new_indices[new_pos++] = old_indices[j];
        }
    }
}

__global__ void compute_out_degrees_kernel(
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ out_degrees,
    int32_t num_edges
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= num_edges) return;
    atomicAdd(&out_degrees[indices[j]], 1);
}

__global__ void compute_inv_out_kernel(
    const int32_t* __restrict__ out_degrees,
    float* __restrict__ inv_out,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int d = out_degrees[v];
    inv_out[v] = (d > 0) ? (1.0f / (float)d) : 0.0f;
}

__global__ void zero_float_kernel(float* __restrict__ arr, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = 0.0f;
}

__global__ void scatter_pers_kernel(
    const int32_t* __restrict__ pers_verts,
    const float* __restrict__ pers_vals,
    float* __restrict__ pers_full,
    int32_t pers_size,
    float inv_sum
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pers_size) return;
    pers_full[pers_verts[i]] = pers_vals[i] * inv_sum;
}

__global__ void init_pr_uniform_kernel(float* __restrict__ pr, float val, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pr[i] = val;
}



__global__ void compute_x_and_dangling_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ inv_out,
    float* __restrict__ x,
    float* __restrict__ dangling_sum,
    int32_t num_vertices
) {
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage temp;

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float dang = 0.0f;

    if (v < num_vertices) {
        float inv = inv_out[v];
        float p = pr[v];
        x[v] = p * inv;
        if (inv == 0.0f) dang = p;
    }

    float bsum = BR(temp).Sum(dang);
    if (threadIdx.x == 0 && bsum != 0.0f) atomicAdd(dangling_sum, bsum);
}

__global__ void spmv_update_block_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    const float* __restrict__ old_pr,
    float* __restrict__ new_pr,
    const float* __restrict__ pers_full,
    const float* __restrict__ d_dangling_sum,
    float* __restrict__ d_diff,
    float alpha,
    float one_minus_alpha,
    int32_t start_v,
    int32_t end_v
) {
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage temp;
    __shared__ float s_dangling_factor;

    int v = start_v + blockIdx.x;
    if (v >= end_v) return;

    if (threadIdx.x == 0) {
        s_dangling_factor = alpha * (*d_dangling_sum) + one_minus_alpha;
    }
    __syncthreads();

    int s = offsets[v];
    int e = offsets[v + 1];
    float sum = 0.0f;
    for (int j = s + threadIdx.x; j < e; j += blockDim.x) {
        sum += x[indices[j]];
    }

    float bsum = BR(temp).Sum(sum);
    if (threadIdx.x == 0) {
        float val = alpha * bsum + s_dangling_factor * pers_full[v];
        float diff = fabsf(val - old_pr[v]);
        new_pr[v] = val;
        if (diff != 0.0f) atomicAdd(d_diff, diff);
    }
}

__global__ void spmv_update_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    const float* __restrict__ old_pr,
    float* __restrict__ new_pr,
    const float* __restrict__ pers_full,
    const float* __restrict__ d_dangling_sum,
    float* __restrict__ d_diff,
    float alpha,
    float one_minus_alpha,
    int32_t start_v,
    int32_t end_v
) {
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage temp;

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / 32;
    int lane = global_tid & 31;
    int v = start_v + warp_id;

    float dangling_factor;
    if (lane == 0 && v < end_v) {
        dangling_factor = alpha * (*d_dangling_sum) + one_minus_alpha;
    }
    dangling_factor = __shfl_sync(0xffffffff, dangling_factor, 0);

    float my_diff = 0.0f;

    if (v < end_v) {
        int s = offsets[v];
        int e = offsets[v + 1];
        float sum = 0.0f;
        for (int j = s + lane; j < e; j += 32) {
            sum += x[indices[j]];
        }
        
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, off);
        }
        if (lane == 0) {
            float val = alpha * sum + dangling_factor * pers_full[v];
            my_diff = fabsf(val - old_pr[v]);
            new_pr[v] = val;
        }
    }

    
    float block_diff = BR(temp).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff != 0.0f) atomicAdd(d_diff, block_diff);
}

__global__ void spmv_update_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    const float* __restrict__ old_pr,
    float* __restrict__ new_pr,
    const float* __restrict__ pers_full,
    const float* __restrict__ d_dangling_sum,
    float* __restrict__ d_diff,
    float alpha,
    float one_minus_alpha,
    int32_t start_v,
    int32_t end_v
) {
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage temp;
    __shared__ float s_dangling_factor;

    if (threadIdx.x == 0) {
        s_dangling_factor = alpha * (*d_dangling_sum) + one_minus_alpha;
    }
    __syncthreads();

    int v = start_v + blockIdx.x * blockDim.x + threadIdx.x;
    float my_diff = 0.0f;

    if (v < end_v) {
        int s = offsets[v];
        int e = offsets[v + 1];
        float sum = 0.0f;
        for (int j = s; j < e; j++) {
            sum += x[indices[j]];
        }
        float val = alpha * sum + s_dangling_factor * pers_full[v];
        my_diff = fabsf(val - old_pr[v]);
        new_pr[v] = val;
    }

    float block_diff = BR(temp).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff != 0.0f) atomicAdd(d_diff, block_diff);
}

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
    float one_minus_alpha = 1.0f - alpha;

    
    const auto& seg_vec = graph.segment_offsets.value();
    int32_t seg[5] = {seg_vec[0], seg_vec[1], seg_vec[2], seg_vec[3], seg_vec[4]};

    
    size_t cub_temp_size = 0;
    cub::DeviceScan::InclusiveSum(nullptr, cub_temp_size,
        (int32_t*)nullptr, (int32_t*)nullptr, num_vertices);

    
    
    
    
    
    
    
    
    
    
    
    
    size_t off_new_offsets = 0;
    size_t off_new_indices = off_new_offsets + (size_t)(num_vertices + 1) * sizeof(int32_t);
    size_t off_out_degrees = off_new_indices + (size_t)num_edges * sizeof(int32_t);
    size_t off_inv_out = off_out_degrees + (size_t)num_vertices * sizeof(int32_t);
    size_t off_pers_full = off_inv_out + (size_t)num_vertices * sizeof(float);
    size_t off_x_buf = off_pers_full + (size_t)num_vertices * sizeof(float);
    size_t off_pr_a = off_x_buf + (size_t)num_vertices * sizeof(float);
    size_t off_pr_b = off_pr_a + (size_t)num_vertices * sizeof(float);
    size_t off_dang = off_pr_b + (size_t)num_vertices * sizeof(float);
    size_t off_diff = off_dang + sizeof(float);
    
    size_t off_cub = (off_diff + sizeof(float) + 255) & ~255ULL;
    size_t total_workspace = off_cub + cub_temp_size;

    cache.ensure(total_workspace);
    char* ws = (char*)cache.workspace;

    int32_t* d_new_offsets = (int32_t*)(ws + off_new_offsets);
    int32_t* d_new_indices = (int32_t*)(ws + off_new_indices);
    int32_t* d_out_degrees = (int32_t*)(ws + off_out_degrees);
    float* d_inv_out = (float*)(ws + off_inv_out);
    float* d_pers_full = (float*)(ws + off_pers_full);
    float* d_x = (float*)(ws + off_x_buf);
    float* d_pr_a = (float*)(ws + off_pr_a);
    float* d_pr_b = (float*)(ws + off_pr_b);
    float* d_dangling_sum = (float*)(ws + off_dang);
    float* d_diff = (float*)(ws + off_diff);
    void* d_cub_temp = (void*)(ws + off_cub);

    

    
    {
        int grid = (num_vertices + 255) / 256;
        count_active_edges_kernel<<<grid, 256>>>(graph.offsets, graph.edge_mask,
            d_out_degrees, num_vertices);
    }

    
    set_int32_kernel<<<1, 1>>>(d_new_offsets, 0);
    cub::DeviceScan::InclusiveSum(d_cub_temp, cub_temp_size,
        d_out_degrees, d_new_offsets + 1, num_vertices);

    
    int32_t num_active_edges = 0;
    cudaMemcpy(&num_active_edges, d_new_offsets + num_vertices,
        sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    {
        int grid = (num_vertices + 255) / 256;
        compact_csc_kernel<<<grid, 256>>>(graph.offsets, graph.indices,
            graph.edge_mask, d_new_offsets, d_new_indices, num_vertices);
    }

    
    cudaMemset(d_out_degrees, 0, (size_t)num_vertices * sizeof(int32_t));
    if (num_active_edges > 0) {
        int grid = (num_active_edges + 255) / 256;
        compute_out_degrees_kernel<<<grid, 256>>>(d_new_indices, d_out_degrees, num_active_edges);
    }

    
    {
        int grid = (num_vertices + 255) / 256;
        compute_inv_out_kernel<<<grid, 256>>>(d_out_degrees, d_inv_out, num_vertices);
    }

    
    {
        int grid = (num_vertices + 255) / 256;
        zero_float_kernel<<<grid, 256>>>(d_pers_full, num_vertices);
    }

    float pers_sum = 0.0f;
    int32_t pers_size_i32 = static_cast<int32_t>(personalization_size);
    if (pers_size_i32 > 0) {
        std::vector<float> h_pers_vals(pers_size_i32);
        cudaMemcpy(h_pers_vals.data(), personalization_values,
            pers_size_i32 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < pers_size_i32; i++) pers_sum += h_pers_vals[i];
    }
    float inv_pers_sum = (pers_sum > 0.0f) ? (1.0f / pers_sum) : 0.0f;
    if (pers_size_i32 > 0) {
        int grid = (pers_size_i32 + 255) / 256;
        scatter_pers_kernel<<<grid, 256>>>(personalization_vertices, personalization_values,
            d_pers_full, pers_size_i32, inv_pers_sum);
    }

    
    if (initial_pageranks) {
        cudaMemcpy(d_pr_a, initial_pageranks,
            (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        float init_val = 1.0f / (float)num_vertices;
        int grid = (num_vertices + 255) / 256;
        init_pr_uniform_kernel<<<grid, 256>>>(d_pr_a, init_val, num_vertices);
    }

    
    float* pr_old = d_pr_a;
    float* pr_new = d_pr_b;
    bool converged = false;
    std::size_t num_iters = 0;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(d_dangling_sum, 0, sizeof(float));
        cudaMemsetAsync(d_diff, 0, sizeof(float));

        
        {
            int grid = (num_vertices + 255) / 256;
            compute_x_and_dangling_kernel<<<grid, 256>>>(pr_old, d_inv_out, d_x,
                d_dangling_sum, num_vertices);
        }

        
        
        {
            int num_v = seg[1] - seg[0];
            if (num_v > 0) {
                spmv_update_block_kernel<<<num_v, 256>>>(d_new_offsets, d_new_indices, d_x,
                    pr_old, pr_new, d_pers_full, d_dangling_sum, d_diff,
                    alpha, one_minus_alpha, seg[0], seg[1]);
            }
        }

        
        {
            int num_v = seg[2] - seg[1];
            if (num_v > 0) {
                int warps_per_block = 256 / 32;  
                int num_blocks = (num_v + warps_per_block - 1) / warps_per_block;
                spmv_update_warp_kernel<<<num_blocks, 256>>>(d_new_offsets, d_new_indices, d_x,
                    pr_old, pr_new, d_pers_full, d_dangling_sum, d_diff,
                    alpha, one_minus_alpha, seg[1], seg[2]);
            }
        }

        
        {
            int num_v = seg[4] - seg[2];
            if (num_v > 0) {
                int num_blocks = (num_v + 255) / 256;
                spmv_update_thread_kernel<<<num_blocks, 256>>>(d_new_offsets, d_new_indices, d_x,
                    pr_old, pr_new, d_pers_full, d_dangling_sum, d_diff,
                    alpha, one_minus_alpha, seg[2], seg[4]);
            }
        }

        num_iters = iter + 1;

        
        float h_diff = 0.0f;
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_diff < epsilon) {
            converged = true;
            break;
        }

        
        std::swap(pr_old, pr_new);
    }

    
    float* final_pr = converged ? pr_new : pr_old;
    cudaMemcpy(pageranks, final_pr,
        (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);

    return PageRankResult{num_iters, converged};
}

}  
