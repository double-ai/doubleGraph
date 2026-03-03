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

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define SPMV_HIGH_BLOCK 512
#define SPMV_MID_WARPS_PER_BLOCK 8




struct Cache : Cacheable {
    float* scratch_pr = nullptr;
    float* x_buf = nullptr;
    float* pers_norm = nullptr;
    float* inv_od = nullptr;
    float* od_temp = nullptr;
    float* scalars = nullptr;
    void* cub_temp = nullptr;
    float* h_diff = nullptr;

    int64_t vertex_cap = 0;
    int64_t scalars_cap = 0;
    std::size_t cub_temp_cap = 0;
    bool has_h_diff = false;

    void ensure(int32_t num_vertices, std::size_t cub_bytes) {
        if (vertex_cap < num_vertices) {
            if (scratch_pr) cudaFree(scratch_pr);
            if (x_buf) cudaFree(x_buf);
            if (pers_norm) cudaFree(pers_norm);
            if (inv_od) cudaFree(inv_od);
            if (od_temp) cudaFree(od_temp);
            cudaMalloc(&scratch_pr, (std::size_t)num_vertices * sizeof(float));
            cudaMalloc(&x_buf, (std::size_t)num_vertices * sizeof(float));
            cudaMalloc(&pers_norm, (std::size_t)num_vertices * sizeof(float));
            cudaMalloc(&inv_od, (std::size_t)num_vertices * sizeof(float));
            cudaMalloc(&od_temp, (std::size_t)num_vertices * sizeof(float));
            vertex_cap = num_vertices;
        }
        if (scalars_cap < 4) {
            if (scalars) cudaFree(scalars);
            cudaMalloc(&scalars, 4 * sizeof(float));
            scalars_cap = 4;
        }
        if (cub_temp_cap < cub_bytes) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, cub_bytes);
            cub_temp_cap = cub_bytes;
        }
        if (!has_h_diff) {
            cudaMallocHost(&h_diff, sizeof(float));
            has_h_diff = true;
        }
    }

    ~Cache() override {
        if (scratch_pr) cudaFree(scratch_pr);
        if (x_buf) cudaFree(x_buf);
        if (pers_norm) cudaFree(pers_norm);
        if (inv_od) cudaFree(inv_od);
        if (od_temp) cudaFree(od_temp);
        if (scalars) cudaFree(scalars);
        if (cub_temp) cudaFree(cub_temp);
        if (h_diff) cudaFreeHost(h_diff);
    }
};




__global__ void compute_out_degrees_kernel(const int32_t* __restrict__ indices,
                                           int32_t num_edges,
                                           float* __restrict__ out_degree) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_edges) {
        atomicAdd(&out_degree[indices[i]], 1.0f);
    }
}




__global__ void compute_inv_out_degree_kernel(const float* __restrict__ out_degree,
                                               float* __restrict__ inv_out_degree,
                                               int32_t num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        float od = out_degree[v];
        inv_out_degree[v] = (od > 0.0f) ? (1.0f / od) : 0.0f;
    }
}




__global__ void fill_kernel(float* __restrict__ arr, int32_t n, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = val;
}




__global__ void scatter_pers_kernel(const int32_t* __restrict__ pers_verts,
                                    const float* __restrict__ pers_vals,
                                    int32_t pers_size,
                                    const float* __restrict__ d_pers_sum,
                                    float* __restrict__ pers_norm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pers_size) {
        float inv_sum = 1.0f / (*d_pers_sum);
        pers_norm[pers_verts[i]] = pers_vals[i] * inv_sum;
    }
}




__global__ void compute_x_and_dangling_kernel(const float* __restrict__ pr,
                                               const float* __restrict__ inv_out_degree,
                                               int32_t num_vertices,
                                               float* __restrict__ x,
                                               float* __restrict__ d_dangling_sum) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float my_dang = 0.0f;
    if (v < num_vertices) {
        float inv_od = inv_out_degree[v];
        float p = pr[v];
        x[v] = p * inv_od;
        if (inv_od == 0.0f) my_dang = p;
    }
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(my_dang);
    if (threadIdx.x == 0 && block_sum != 0.0f)
        atomicAdd(d_dangling_sum, block_sum);
}




__global__ void spmv_fused_high_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ x, const float* __restrict__ old_pr,
    const float* __restrict__ pers_norm, const float* __restrict__ d_dangling_sum,
    float alpha, float* __restrict__ new_pr, float* __restrict__ d_diff,
    int32_t start_v, int32_t end_v)
{
    int v = start_v + blockIdx.x;
    if (v >= end_v) return;
    int begin = offsets[v];
    int end_e = offsets[v + 1];
    float sum = 0.0f;
    for (int i = begin + threadIdx.x; i < end_e; i += SPMV_HIGH_BLOCK)
        sum += x[__ldg(&indices[i])];

    typedef cub::BlockReduce<float, SPMV_HIGH_BLOCK> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    float block_sum = BlockReduce(temp).Sum(sum);

    if (threadIdx.x == 0) {
        float base_factor = alpha * (*d_dangling_sum) + (1.0f - alpha);
        float np = alpha * block_sum + base_factor * pers_norm[v];
        float diff = fabsf(np - old_pr[v]);
        new_pr[v] = np;
        if (diff != 0.0f) atomicAdd(d_diff, diff);
    }
}




__global__ void spmv_fused_mid_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ x, const float* __restrict__ old_pr,
    const float* __restrict__ pers_norm, const float* __restrict__ d_dangling_sum,
    float alpha, float* __restrict__ new_pr, float* __restrict__ d_diff,
    int32_t start_v, int32_t end_v)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / 32;
    int lane = global_tid & 31;
    int v = start_v + warp_id;
    if (v >= end_v) return;
    int begin = offsets[v];
    int end_e = offsets[v + 1];
    float sum = 0.0f;
    for (int i = begin + lane; i < end_e; i += 32)
        sum += x[__ldg(&indices[i])];

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) {
        float base_factor = alpha * (*d_dangling_sum) + (1.0f - alpha);
        float np = alpha * sum + base_factor * pers_norm[v];
        float diff = fabsf(np - old_pr[v]);
        new_pr[v] = np;
        if (diff != 0.0f) atomicAdd(d_diff, diff);
    }
}




__global__ void spmv_fused_low_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ x, const float* __restrict__ old_pr,
    const float* __restrict__ pers_norm, const float* __restrict__ d_dangling_sum,
    float alpha, float* __restrict__ new_pr, float* __restrict__ d_diff,
    int32_t start_v, int32_t end_v)
{
    float base_factor = alpha * (*d_dangling_sum) + (1.0f - alpha);
    int v = start_v + blockIdx.x * blockDim.x + threadIdx.x;
    float my_diff = 0.0f;
    if (v < end_v) {
        int begin = offsets[v];
        int end_e = offsets[v + 1];
        float sum = 0.0f;
        for (int i = begin; i < end_e; i++)
            sum += x[__ldg(&indices[i])];
        float np = alpha * sum + base_factor * pers_norm[v];
        my_diff = fabsf(np - old_pr[v]);
        new_pr[v] = np;
    }
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    float block_sum = BlockReduce(temp).Sum(my_diff);
    if (threadIdx.x == 0 && block_sum != 0.0f)
        atomicAdd(d_diff, block_sum);
}





void launch_compute_out_degrees(const int32_t* indices, int32_t num_edges,
                                float* out_degree, int32_t num_vertices) {
    cudaMemset(out_degree, 0, num_vertices * sizeof(float));
    if (num_edges > 0) {
        int grid = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_out_degrees_kernel<<<grid, BLOCK_SIZE>>>(indices, num_edges, out_degree);
    }
}

void launch_compute_inv_out_degree(const float* out_degree, float* inv_out_degree, int32_t num_vertices) {
    int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_inv_out_degree_kernel<<<grid, BLOCK_SIZE>>>(out_degree, inv_out_degree, num_vertices);
}

void launch_fill(float* arr, int32_t n, float val) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fill_kernel<<<grid, BLOCK_SIZE>>>(arr, n, val);
}

void launch_scatter_pers(const int32_t* pers_verts, const float* pers_vals,
                         int32_t pers_size, const float* d_pers_sum,
                         float* pers_norm, int32_t num_vertices) {
    cudaMemset(pers_norm, 0, num_vertices * sizeof(float));
    if (pers_size > 0) {
        int grid = (pers_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scatter_pers_kernel<<<grid, BLOCK_SIZE>>>(pers_verts, pers_vals, pers_size, d_pers_sum, pers_norm);
    }
}

std::size_t cub_reduce_sum_temp_bytes(int32_t n) {
    std::size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, temp_bytes, (const float*)nullptr, (float*)nullptr, n);
    return temp_bytes;
}

void launch_cub_reduce_sum(const float* d_in, float* d_out, int32_t n,
                           void* d_temp, std::size_t temp_bytes) {
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);
}

void launch_pagerank_iteration(const float* pr_cur, float* pr_next,
                               const float* inv_out_degree, const float* pers_norm,
                               const int32_t* offsets, const int32_t* indices,
                               float* x, float alpha, int32_t num_vertices,
                               float* d_scalars,
                               int32_t seg0, int32_t seg1, int32_t seg2,
                               int32_t seg3, int32_t seg4) {
    float* d_dangling_sum = d_scalars;
    float* d_diff = d_scalars + 1;

    
    cudaMemset(d_scalars, 0, 2 * sizeof(float));

    
    {
        int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        compute_x_and_dangling_kernel<<<grid, BLOCK_SIZE>>>(pr_cur, inv_out_degree, num_vertices, x, d_dangling_sum);
    }

    
    int32_t n_high = seg1 - seg0;
    if (n_high > 0) {
        spmv_fused_high_kernel<<<n_high, SPMV_HIGH_BLOCK>>>(
            offsets, indices, x, pr_cur, pers_norm, d_dangling_sum,
            alpha, pr_next, d_diff, seg0, seg1);
    }

    int32_t n_mid = seg2 - seg1;
    if (n_mid > 0) {
        int threads_per_block = SPMV_MID_WARPS_PER_BLOCK * 32;
        int grid = (n_mid + SPMV_MID_WARPS_PER_BLOCK - 1) / SPMV_MID_WARPS_PER_BLOCK;
        spmv_fused_mid_kernel<<<grid, threads_per_block>>>(
            offsets, indices, x, pr_cur, pers_norm, d_dangling_sum,
            alpha, pr_next, d_diff, seg1, seg2);
    }

    int32_t n_low_zero = seg4 - seg2;
    if (n_low_zero > 0) {
        int grid = (n_low_zero + BLOCK_SIZE - 1) / BLOCK_SIZE;
        spmv_fused_low_kernel<<<grid, BLOCK_SIZE>>>(
            offsets, indices, x, pr_cur, pers_norm, d_dangling_sum,
            alpha, pr_next, d_diff, seg2, seg4);
    }
}

}  

PageRankResult personalized_pagerank_seg(const graph32_t& graph,
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

    
    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2];
    int32_t seg3 = seg[3], seg4 = seg[4];

    int32_t pers_size = static_cast<int32_t>(personalization_size);

    
    std::size_t cub_bytes = cub_reduce_sum_temp_bytes(pers_size);
    if (cub_bytes == 0) cub_bytes = 1;

    
    cache.ensure(num_vertices, cub_bytes);

    float* d_scratch = cache.scratch_pr;
    float* d_x = cache.x_buf;
    float* d_pers_norm = cache.pers_norm;
    float* d_inv_od = cache.inv_od;
    float* d_scalars = cache.scalars;

    
    if (precomputed_vertex_out_weight_sums != nullptr) {
        launch_compute_inv_out_degree(precomputed_vertex_out_weight_sums, d_inv_od, num_vertices);
    } else {
        launch_compute_out_degrees(d_indices, num_edges, cache.od_temp, num_vertices);
        launch_compute_inv_out_degree(cache.od_temp, d_inv_od, num_vertices);
    }

    
    float* d_pers_sum = d_scalars + 2;
    launch_cub_reduce_sum(personalization_values, d_pers_sum, pers_size,
                          cache.cub_temp, cub_bytes);
    launch_scatter_pers(personalization_vertices, personalization_values,
                       pers_size, d_pers_sum, d_pers_norm, num_vertices);

    
    float* pr_cur = pageranks;
    float* pr_next = d_scratch;
    if (initial_pageranks != nullptr) {
        cudaMemcpy(pr_cur, initial_pageranks,
                   (std::size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        launch_fill(pr_cur, num_vertices, 1.0f / (float)num_vertices);
    }

    
    std::size_t iterations = 0;
    bool converged = false;

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        launch_pagerank_iteration(pr_cur, pr_next, d_inv_od, d_pers_norm,
                                  d_offsets, d_indices, d_x, alpha, num_vertices,
                                  d_scalars, seg0, seg1, seg2, seg3, seg4);

        cudaMemcpy(cache.h_diff, d_scalars + 1, sizeof(float), cudaMemcpyDeviceToHost);
        std::swap(pr_cur, pr_next);
        iterations = iter + 1;
        if (*cache.h_diff < epsilon) { converged = true; break; }
    }

    if (pr_cur != pageranks)
        cudaMemcpy(pageranks, pr_cur, (std::size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);

    return PageRankResult{iterations, converged};
}

}  
