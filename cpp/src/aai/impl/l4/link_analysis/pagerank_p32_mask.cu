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
#include <vector>

namespace aai {

namespace {

static constexpr int CONV_BATCH = 4;

struct Cache : Cacheable {
    int32_t* flags = nullptr;
    int32_t* prefix = nullptr;
    void* workspace = nullptr;
    int32_t* new_offsets = nullptr;
    int32_t* new_indices = nullptr;
    int32_t* out_degree = nullptr;
    float* buf_a = nullptr;
    float* buf_b = nullptr;
    float* x = nullptr;
    float* dangling = nullptr;
    float* diff = nullptr;
    float* pers_norm = nullptr;

    int64_t flags_cap = 0;
    int64_t prefix_cap = 0;
    size_t workspace_cap = 0;
    int64_t new_offsets_cap = 0;
    int64_t new_indices_cap = 0;
    int64_t out_degree_cap = 0;
    int64_t buf_a_cap = 0;
    int64_t buf_b_cap = 0;
    int64_t x_cap = 0;
    bool dangling_alloc = false;
    bool diff_alloc = false;
    int64_t pers_norm_cap = 0;

    void ensure(int32_t N, int32_t E, size_t ws_size, int64_t pers_sz) {
        int64_t ep1 = (int64_t)(E + 1);
        int64_t np1 = (int64_t)(N + 1);
        int64_t n = std::max((int64_t)N, (int64_t)1);
        int64_t e = std::max((int64_t)E, (int64_t)1);

        if (flags_cap < ep1) {
            if (flags) cudaFree(flags);
            cudaMalloc(&flags, ep1 * sizeof(int32_t));
            flags_cap = ep1;
        }
        if (prefix_cap < ep1) {
            if (prefix) cudaFree(prefix);
            cudaMalloc(&prefix, ep1 * sizeof(int32_t));
            prefix_cap = ep1;
        }
        if (workspace_cap < ws_size) {
            if (workspace) cudaFree(workspace);
            cudaMalloc(&workspace, std::max(ws_size, (size_t)1));
            workspace_cap = ws_size;
        }
        if (new_offsets_cap < np1) {
            if (new_offsets) cudaFree(new_offsets);
            cudaMalloc(&new_offsets, np1 * sizeof(int32_t));
            new_offsets_cap = np1;
        }
        if (new_indices_cap < e) {
            if (new_indices) cudaFree(new_indices);
            cudaMalloc(&new_indices, e * sizeof(int32_t));
            new_indices_cap = e;
        }
        if (out_degree_cap < n) {
            if (out_degree) cudaFree(out_degree);
            cudaMalloc(&out_degree, n * sizeof(int32_t));
            out_degree_cap = n;
        }
        if (buf_a_cap < n) {
            if (buf_a) cudaFree(buf_a);
            cudaMalloc(&buf_a, n * sizeof(float));
            buf_a_cap = n;
        }
        if (buf_b_cap < n) {
            if (buf_b) cudaFree(buf_b);
            cudaMalloc(&buf_b, n * sizeof(float));
            buf_b_cap = n;
        }
        if (x_cap < n) {
            if (x) cudaFree(x);
            cudaMalloc(&x, n * sizeof(float));
            x_cap = n;
        }
        if (!dangling_alloc) {
            cudaMalloc(&dangling, sizeof(float));
            dangling_alloc = true;
        }
        if (!diff_alloc) {
            cudaMalloc(&diff, sizeof(float));
            diff_alloc = true;
        }
        if (pers_norm_cap < pers_sz) {
            if (pers_norm) cudaFree(pers_norm);
            cudaMalloc(&pers_norm, pers_sz * sizeof(float));
            pers_norm_cap = pers_sz;
        }
    }

    ~Cache() override {
        if (flags) cudaFree(flags);
        if (prefix) cudaFree(prefix);
        if (workspace) cudaFree(workspace);
        if (new_offsets) cudaFree(new_offsets);
        if (new_indices) cudaFree(new_indices);
        if (out_degree) cudaFree(out_degree);
        if (buf_a) cudaFree(buf_a);
        if (buf_b) cudaFree(buf_b);
        if (x) cudaFree(x);
        if (dangling) cudaFree(dangling);
        if (diff) cudaFree(diff);
        if (pers_norm) cudaFree(pers_norm);
    }
};





__global__ void expand_mask_kernel(
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ flags,
    int num_edges
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    flags[e] = (edge_mask[e >> 5] >> (e & 31)) & 1;
}

__global__ void scatter_edges_kernel(
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ prefix,
    const int32_t* __restrict__ flags,
    int32_t* __restrict__ new_indices,
    int num_edges
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    if (flags[e]) new_indices[prefix[e]] = indices[e];
}

__global__ void build_new_offsets_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ prefix,
    int32_t* __restrict__ new_offsets,
    int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v > num_vertices) return;
    new_offsets[v] = prefix[old_offsets[v]];
}

__global__ void compute_out_degrees_kernel(
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ out_degree,
    int num_active_edges
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_active_edges) return;
    atomicAdd(&out_degree[indices[e]], 1);
}





__global__ void normalize_and_dangling_kernel(
    const float* __restrict__ pr,
    const int32_t* __restrict__ out_deg,
    float* __restrict__ x,
    float* __restrict__ dangling_sum,
    int N
) {
    __shared__ float warp_sums[8]; 
    int tid = threadIdx.x;
    int v = blockIdx.x * blockDim.x + tid;

    float local_dangling = 0.0f;
    if (v < N) {
        int od = out_deg[v];
        float pv = pr[v];
        x[v] = (od > 0) ? __fdividef(pv, (float)od) : pv;
        if (od == 0) local_dangling = pv;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_dangling += __shfl_down_sync(0xffffffff, local_dangling, offset);

    if ((tid & 31) == 0) warp_sums[tid >> 5] = local_dangling;
    __syncthreads();

    if (tid < (blockDim.x >> 5)) {
        local_dangling = warp_sums[tid];
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            local_dangling += __shfl_down_sync(0xffffffff, local_dangling, offset);
        if (tid == 0) atomicAdd(dangling_sum, local_dangling);
    }
}

__global__ void spmv_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    float alpha,
    int N
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;

    int start = offsets[v], end = offsets[v + 1];
    float sum = 0.0f;
    for (int e = start; e < end; e++) {
        sum += x[indices[e]];
    }
    y[v] = alpha * sum;
}

__global__ void spmv_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    float alpha,
    int N
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= N) return;

    int v = warp_id;
    int start = offsets[v], end = offsets[v + 1];

    float sum = 0.0f;
    for (int e = start + lane; e < end; e += 32) {
        sum += x[indices[e]];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) y[v] = alpha * sum;
}

__global__ void teleport_kernel(
    float* __restrict__ y,
    const float* __restrict__ dangling_sum,
    const int32_t* __restrict__ pers_vertices,
    const float* __restrict__ pers_norm,
    int pers_size,
    float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pers_size) return;
    float base_factor = __fmaf_rn(alpha, *dangling_sum, 1.0f - alpha);
    y[pers_vertices[i]] += base_factor * pers_norm[i];
}

__global__ void diff_reduce_kernel(
    const float* __restrict__ new_pr,
    const float* __restrict__ old_pr,
    float* __restrict__ diff_sum,
    int N
) {
    __shared__ float warp_sums[8];
    int tid = threadIdx.x;
    int v = blockIdx.x * blockDim.x + tid;

    float local_diff = (v < N) ? fabsf(new_pr[v] - old_pr[v]) : 0.0f;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_diff += __shfl_down_sync(0xffffffff, local_diff, offset);

    if ((tid & 31) == 0) warp_sums[tid >> 5] = local_diff;
    __syncthreads();

    if (tid < (blockDim.x >> 5)) {
        local_diff = warp_sums[tid];
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            local_diff += __shfl_down_sync(0xffffffff, local_diff, offset);
        if (tid == 0) atomicAdd(diff_sum, local_diff);
    }
}

__global__ void init_pr_kernel(float* pr, int N) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    pr[v] = 1.0f / (float)N;
}





void launch_expand_mask(const uint32_t* edge_mask, int32_t* flags, int num_edges) {
    if (num_edges <= 0) return;
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    expand_mask_kernel<<<grid, block>>>(edge_mask, flags, num_edges);
}

size_t get_prefix_sum_workspace(int num_items) {
    size_t ws = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, ws, (int32_t*)nullptr, (int32_t*)nullptr, num_items);
    return ws;
}

void launch_prefix_sum(int32_t* in, int32_t* out, void* workspace, size_t ws_size, int num_items) {
    cub::DeviceScan::ExclusiveSum(workspace, ws_size, in, out, num_items);
}

void launch_scatter_edges(const int32_t* indices, const int32_t* prefix,
                           const int32_t* flags, int32_t* new_indices, int num_edges) {
    if (num_edges <= 0) return;
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    scatter_edges_kernel<<<grid, block>>>(indices, prefix, flags, new_indices, num_edges);
}

void launch_build_new_offsets(const int32_t* old_offsets, const int32_t* prefix,
                               int32_t* new_offsets, int num_vertices) {
    int block = 256;
    int grid = (num_vertices + 1 + block - 1) / block;
    build_new_offsets_kernel<<<grid, block>>>(old_offsets, prefix, new_offsets, num_vertices);
}

void launch_compute_out_degrees(const int32_t* indices, int32_t* out_degree, int n) {
    if (n <= 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    compute_out_degrees_kernel<<<grid, block>>>(indices, out_degree, n);
}

void launch_normalize_and_dangling(const float* pr, const int32_t* out_deg,
                                    float* x, float* dangling_sum, int N) {
    if (N <= 0) return;
    int block = 256;
    int grid = (N + block - 1) / block;
    normalize_and_dangling_kernel<<<grid, block>>>(pr, out_deg, x, dangling_sum, N);
}

void launch_spmv_thread(const int32_t* offsets, const int32_t* indices,
                         const float* x, float* y, float alpha, int N) {
    if (N <= 0) return;
    int block = 256;
    int grid = (N + block - 1) / block;
    spmv_thread_kernel<<<grid, block>>>(offsets, indices, x, y, alpha, N);
}

void launch_spmv_warp(const int32_t* offsets, const int32_t* indices,
                       const float* x, float* y, float alpha, int N) {
    if (N <= 0) return;
    int block = 256;
    int warps_per_block = block / 32;
    int grid = (N + warps_per_block - 1) / warps_per_block;
    spmv_warp_kernel<<<grid, block>>>(offsets, indices, x, y, alpha, N);
}

void launch_teleport(float* y, const float* dangling_sum,
                      const int32_t* pers_vertices, const float* pers_norm,
                      int pers_size, float alpha) {
    if (pers_size <= 0) return;
    int block = 256;
    int grid = (pers_size + block - 1) / block;
    teleport_kernel<<<grid, block>>>(y, dangling_sum, pers_vertices, pers_norm, pers_size, alpha);
}

void launch_diff_reduce(const float* new_pr, const float* old_pr, float* diff_sum, int N) {
    if (N <= 0) return;
    int block = 256;
    int grid = (N + block - 1) / block;
    diff_reduce_kernel<<<grid, block>>>(new_pr, old_pr, diff_sum, N);
}

void launch_init_pr(float* pr, int N) {
    if (N <= 0) return;
    int block = 256;
    int grid = (N + block - 1) / block;
    init_pr_kernel<<<grid, block>>>(pr, N);
}

}  

PageRankResult personalized_pagerank_mask(const graph32_t& graph,
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

    (void)precomputed_vertex_out_weight_sums;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t N = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;

    size_t ws_size = get_prefix_sum_workspace(E + 1);
    cache.ensure(N, E, ws_size, (int64_t)personalization_size);

    
    
    int32_t zero = 0;
    cudaMemcpy(cache.flags + E, &zero, sizeof(int32_t), cudaMemcpyHostToDevice);
    if (E > 0) launch_expand_mask(d_edge_mask, cache.flags, E);

    
    launch_prefix_sum(cache.flags, cache.prefix, cache.workspace, ws_size, E + 1);

    int32_t new_E = 0;
    cudaMemcpy(&new_E, cache.prefix + E, sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    launch_build_new_offsets(d_offsets, cache.prefix, cache.new_offsets, N);
    if (E > 0) launch_scatter_edges(d_indices, cache.prefix, cache.flags, cache.new_indices, E);

    
    cudaMemset(cache.out_degree, 0, N * sizeof(int32_t));
    if (new_E > 0) launch_compute_out_degrees(cache.new_indices, cache.out_degree, new_E);

    
    std::vector<float> h_pers(personalization_size);
    cudaMemcpy(h_pers.data(), personalization_values, personalization_size * sizeof(float), cudaMemcpyDeviceToHost);
    double pers_sum = 0;
    for (std::size_t i = 0; i < personalization_size; i++) pers_sum += (double)h_pers[i];
    for (std::size_t i = 0; i < personalization_size; i++) h_pers[i] = (float)((double)h_pers[i] / pers_sum);
    cudaMemcpy(cache.pers_norm, h_pers.data(), personalization_size * sizeof(float), cudaMemcpyHostToDevice);

    
    float avg_degree = (N > 0) ? (float)new_E / (float)N : 0.0f;
    bool use_warp_spmv = (avg_degree >= 12.0f);

    
    float* d_curr = cache.buf_a;
    float* d_next = cache.buf_b;

    
    if (initial_pageranks != nullptr) {
        cudaMemcpy(d_curr, initial_pageranks, N * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        launch_init_pr(d_curr, N);
    }

    
    bool converged = false;
    std::size_t iterations = 0;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(cache.dangling, 0, sizeof(float));

        
        launch_normalize_and_dangling(d_curr, cache.out_degree, cache.x, cache.dangling, N);

        
        if (use_warp_spmv) {
            launch_spmv_warp(cache.new_offsets, cache.new_indices, cache.x, d_next, alpha, N);
        } else {
            launch_spmv_thread(cache.new_offsets, cache.new_indices, cache.x, d_next, alpha, N);
        }

        
        launch_teleport(d_next, cache.dangling, personalization_vertices, cache.pers_norm, (int)personalization_size, alpha);

        
        float* tmp = d_curr;
        d_curr = d_next;
        d_next = tmp;

        iterations = iter + 1;

        
        
        bool should_check = ((iter + 1) % CONV_BATCH == 0) || (iter == max_iterations - 1);
        if (should_check) {
            cudaMemsetAsync(cache.diff, 0, sizeof(float));
            
            launch_diff_reduce(d_curr, d_next, cache.diff, N);

            float h_diff;
            cudaMemcpy(&h_diff, cache.diff, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_diff < epsilon) {
                converged = true;
                break;
            }
        }
    }

    
    cudaMemcpy(pageranks, d_curr, N * sizeof(float), cudaMemcpyDeviceToDevice);

    return PageRankResult{iterations, converged};
}

}  
