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
    int32_t* new_indices = nullptr;
    float* out_weight = nullptr;
    float* inv_out_weight = nullptr;
    float* pr_buf = nullptr;
    float* pr_scaled = nullptr;
    double* d_results = nullptr;

    int32_t vertex_capacity = 0;
    int32_t edge_capacity = 0;

    void ensure(int32_t num_vertices, int32_t num_edges) {
        if (vertex_capacity < num_vertices) {
            if (active_counts) cudaFree(active_counts);
            if (new_offsets) cudaFree(new_offsets);
            if (out_weight) cudaFree(out_weight);
            if (inv_out_weight) cudaFree(inv_out_weight);
            if (pr_buf) cudaFree(pr_buf);
            if (pr_scaled) cudaFree(pr_scaled);
            if (d_results) cudaFree(d_results);

            cudaMalloc(&active_counts, (size_t)(num_vertices + 1) * sizeof(int32_t));
            cudaMalloc(&new_offsets, (size_t)(num_vertices + 1) * sizeof(int32_t));
            cudaMalloc(&out_weight, (size_t)num_vertices * sizeof(float));
            cudaMalloc(&inv_out_weight, (size_t)num_vertices * sizeof(float));
            cudaMalloc(&pr_buf, (size_t)num_vertices * sizeof(float));
            cudaMalloc(&pr_scaled, (size_t)num_vertices * sizeof(float));
            cudaMalloc(&d_results, 2 * sizeof(double));

            vertex_capacity = num_vertices;
        }
        if (edge_capacity < num_edges) {
            if (new_indices) cudaFree(new_indices);
            cudaMalloc(&new_indices, (size_t)num_edges * sizeof(int32_t));
            edge_capacity = num_edges;
        }
    }

    ~Cache() override {
        if (active_counts) cudaFree(active_counts);
        if (new_offsets) cudaFree(new_offsets);
        if (new_indices) cudaFree(new_indices);
        if (out_weight) cudaFree(out_weight);
        if (inv_out_weight) cudaFree(inv_out_weight);
        if (pr_buf) cudaFree(pr_buf);
        if (pr_scaled) cudaFree(pr_scaled);
        if (d_results) cudaFree(d_results);
    }
};



__global__ void count_active_edges(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_counts,
    int32_t num_vertices)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int count = 0;
        if (start < end) {
            int w_start = start >> 5;
            int w_end = (end - 1) >> 5;
            for (int w = w_start; w <= w_end; w++) {
                uint32_t word = edge_mask[w];
                int lo = (w == w_start) ? (start & 31) : 0;
                int hi = (w == w_end) ? ((end - 1) & 31) + 1 : 32;
                word >>= lo;
                int nbits = hi - lo;
                if (nbits < 32) word &= (1u << nbits) - 1;
                count += __popc(word);
            }
        }
        active_counts[v] = count;
    }
}

__global__ void compact_indices(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t num_vertices)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        int old_start = old_offsets[v];
        int old_end = old_offsets[v + 1];
        int write_pos = new_offsets[v];
        for (int e = old_start; e < old_end; e++) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1u) {
                new_indices[write_pos++] = old_indices[e];
            }
        }
    }
}



__global__ void compute_out_weights(
    const int32_t* __restrict__ indices,
    float* __restrict__ out_weight,
    int32_t num_active_edges)
{
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_active_edges; e += blockDim.x * gridDim.x) {
        atomicAdd(&out_weight[indices[e]], 1.0f);
    }
}

__global__ void compute_inv_out_weights(
    const float* __restrict__ out_weight,
    float* __restrict__ inv_out_weight,
    int32_t num_vertices)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        float ow = out_weight[v];
        inv_out_weight[v] = (ow > 0.0f) ? (1.0f / ow) : 0.0f;
    }
}



__global__ void init_pr(float* __restrict__ pr, float val, int32_t num_vertices)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        pr[v] = val;
    }
}



__global__ void prepare_iteration(
    const float* __restrict__ pr,
    const float* __restrict__ inv_out_weight,
    float* __restrict__ pr_scaled,
    double* __restrict__ d_dangling_sum,
    int32_t num_vertices)
{
    __shared__ double warp_sums[8];
    double thread_sum = 0.0;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        float iow = inv_out_weight[v];
        float p = pr[v];
        pr_scaled[v] = p * iow;
        if (iow == 0.0f) thread_sum += (double)p;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = thread_sum;
    __syncthreads();
    if (warp == 0) {
        double val = (lane < 8) ? warp_sums[lane] : 0.0;
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) atomicAdd(d_dangling_sum, val);
    }
}


__global__ void spmv_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_scaled,
    float* __restrict__ pr_new,
    float alpha,
    double alpha_over_n,
    const double* __restrict__ d_dangling_sum,
    float base_rank,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int start = offsets[v];
    int end = offsets[v + 1];
    float sum = 0.0f;
    for (int e = start; e < end; e++) {
        sum += pr_scaled[indices[e]];
    }
    float dc = (float)(alpha_over_n * d_dangling_sum[0]);
    pr_new[v] = base_rank + alpha * sum + dc;
}


__global__ void spmv_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_scaled,
    float* __restrict__ pr_new,
    float alpha,
    double alpha_over_n,
    const double* __restrict__ d_dangling_sum,
    float base_rank,
    int32_t num_vertices)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;

    float dc = (float)(alpha_over_n * d_dangling_sum[0]);

    for (int v = warp_id; v < num_vertices; v += total_warps) {
        int start = offsets[v];
        int end = offsets[v + 1];

        float sum = 0.0f;
        for (int e = start + lane; e < end; e += 32) {
            sum += pr_scaled[indices[e]];
        }

        
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        if (lane == 0) {
            pr_new[v] = base_rank + alpha * sum + dc;
        }
    }
}


__global__ void fused_convergence_prepare(
    const float* __restrict__ pr_new,
    const float* __restrict__ pr_old,
    const float* __restrict__ inv_out_weight,
    float* __restrict__ pr_scaled,
    double* __restrict__ d_results,
    int32_t num_vertices)
{
    __shared__ double warp_diff[8];
    __shared__ double warp_dang[8];
    double thread_diff = 0.0;
    double thread_dang = 0.0;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        float pn = pr_new[v];
        float po = pr_old[v];
        float iow = inv_out_weight[v];
        thread_diff += fabs((double)pn - (double)po);
        pr_scaled[v] = pn * iow;
        if (iow == 0.0f) thread_dang += (double)pn;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_diff += __shfl_down_sync(0xffffffff, thread_diff, offset);
        thread_dang += __shfl_down_sync(0xffffffff, thread_dang, offset);
    }
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) { warp_diff[warp] = thread_diff; warp_dang[warp] = thread_dang; }
    __syncthreads();
    if (warp == 0) {
        double vd = (lane < 8) ? warp_diff[lane] : 0.0;
        double vg = (lane < 8) ? warp_dang[lane] : 0.0;
        for (int offset = 4; offset > 0; offset >>= 1) {
            vd += __shfl_down_sync(0xffffffff, vd, offset);
            vg += __shfl_down_sync(0xffffffff, vg, offset);
        }
        if (lane == 0) {
            atomicAdd(&d_results[0], vg);
            atomicAdd(&d_results[1], vd);
        }
    }
}

}  

PageRankResult pagerank_mask(const graph32_t& graph,
                             float* pageranks,
                             const float* precomputed_vertex_out_weight_sums,
                             float alpha,
                             float epsilon,
                             std::size_t max_iterations,
                             const float* initial_pageranks) {
    (void)precomputed_vertex_out_weight_sums;

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;

    bool has_initial_guess = (initial_pageranks != nullptr);

    if (num_vertices == 0) {
        return PageRankResult{0, true};
    }

    cache.ensure(num_vertices, num_edges);

    int32_t* active_counts = cache.active_counts;
    int32_t* new_offsets = cache.new_offsets;
    int32_t* new_indices = cache.new_indices;
    float* out_weight = cache.out_weight;
    float* inv_out_weight = cache.inv_out_weight;
    float* pr_buf = cache.pr_buf;
    float* pr_scaled = cache.pr_scaled;
    double* d_results = cache.d_results;

    const int BLOCK = 256;
    int grid_v = (num_vertices + BLOCK - 1) / BLOCK;
    int grid_reduce = (grid_v < 1024) ? grid_v : 1024;

    
    count_active_edges<<<grid_v, BLOCK>>>(d_offsets, d_edge_mask, active_counts, num_vertices);
    cudaMemset(active_counts + num_vertices, 0, sizeof(int32_t));

    size_t cub_temp_size = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_temp_size, active_counts, new_offsets, num_vertices + 1);
    void* d_cub_temp;
    cudaMalloc(&d_cub_temp, cub_temp_size);
    cub::DeviceScan::ExclusiveSum(d_cub_temp, cub_temp_size, active_counts, new_offsets, num_vertices + 1);
    cudaFree(d_cub_temp);

    int32_t total_active;
    cudaMemcpy(&total_active, new_offsets + num_vertices, sizeof(int32_t), cudaMemcpyDeviceToHost);

    compact_indices<<<grid_v, BLOCK>>>(d_offsets, d_indices, d_edge_mask, new_offsets, new_indices, num_vertices);

    
    cudaMemset(out_weight, 0, num_vertices * sizeof(float));
    if (total_active > 0) {
        int grid_ae = (total_active + BLOCK - 1) / BLOCK;
        compute_out_weights<<<grid_ae, BLOCK>>>(new_indices, out_weight, total_active);
    }
    compute_inv_out_weights<<<grid_v, BLOCK>>>(out_weight, inv_out_weight, num_vertices);

    
    float init_val = 1.0f / (float)num_vertices;
    if (has_initial_guess) {
        cudaMemcpy(pageranks, initial_pageranks, num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        init_pr<<<grid_v, BLOCK>>>(pageranks, init_val, num_vertices);
    }

    
    float base_rank = (1.0f - alpha) / (float)num_vertices;
    double alpha_over_n = (double)alpha / (double)num_vertices;

    
    float avg_degree = (num_vertices > 0) ? (float)total_active / (float)num_vertices : 0.0f;
    bool use_warp = (avg_degree >= 12.0f);

    
    int grid_warp = 0;
    if (use_warp) {
        int warps_per_block = BLOCK / 32;
        grid_warp = (num_vertices + warps_per_block - 1) / warps_per_block;
        if (grid_warp > 8192) grid_warp = 8192;
    }

    float* pr_cur = pageranks;
    float* pr_next = pr_buf;

    cudaMemset(d_results, 0, 2 * sizeof(double));
    prepare_iteration<<<grid_reduce, BLOCK>>>(pr_cur, inv_out_weight, pr_scaled, d_results, num_vertices);

    bool converged = false;
    size_t iter = 0;

    for (iter = 0; iter < max_iterations; iter++) {
        if (use_warp) {
            spmv_warp<<<grid_warp, BLOCK>>>(new_offsets, new_indices, pr_scaled, pr_next,
                                             alpha, alpha_over_n, d_results, base_rank, num_vertices);
        } else {
            spmv_thread<<<grid_v, BLOCK>>>(new_offsets, new_indices, pr_scaled, pr_next,
                                            alpha, alpha_over_n, d_results, base_rank, num_vertices);
        }

        cudaMemset(d_results, 0, 2 * sizeof(double));
        fused_convergence_prepare<<<grid_reduce, BLOCK>>>(pr_next, pr_cur, inv_out_weight, pr_scaled, d_results, num_vertices);

        float* tmp = pr_cur; pr_cur = pr_next; pr_next = tmp;

        double h_diff;
        cudaMemcpy(&h_diff, &d_results[1], sizeof(double), cudaMemcpyDeviceToHost);

        if (h_diff < (double)epsilon) {
            converged = true;
            iter++;
            break;
        }
    }

    if (pr_cur != pageranks) {
        cudaMemcpy(pageranks, pr_cur, num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return PageRankResult{iter, converged};
}

}  
