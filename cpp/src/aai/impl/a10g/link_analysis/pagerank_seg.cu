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

#define BLOCK_SIZE 256
#define WARP_SIZE 32

struct Cache : Cacheable {
    
    int32_t* out_deg = nullptr;
    float* inv_od = nullptr;
    float* contrib = nullptr;
    float* pr_a = nullptr;
    float* pr_b = nullptr;

    
    int32_t* sort_a = nullptr;
    int32_t* sort_b = nullptr;

    
    void* sort_temp = nullptr;

    
    double* dangling = nullptr;
    double* diff = nullptr;

    
    size_t l2_persist_max = 0;

    
    int64_t vertex_capacity = 0;
    int64_t edge_capacity = 0;
    size_t sort_temp_capacity = 0;
    bool fixed_allocated = false;

    Cache() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        l2_persist_max = prop.persistingL2CacheMaxSize;
        if (l2_persist_max > 0) {
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_persist_max);
        }
    }

    void ensure_vertex(int64_t N) {
        if (vertex_capacity < N) {
            if (out_deg) cudaFree(out_deg);
            if (inv_od) cudaFree(inv_od);
            if (contrib) cudaFree(contrib);
            if (pr_a) cudaFree(pr_a);
            if (pr_b) cudaFree(pr_b);
            cudaMalloc(&out_deg, N * sizeof(int32_t));
            cudaMalloc(&inv_od, N * sizeof(float));
            cudaMalloc(&contrib, N * sizeof(float));
            cudaMalloc(&pr_a, N * sizeof(float));
            cudaMalloc(&pr_b, N * sizeof(float));
            vertex_capacity = N;
        }
    }

    void ensure_edge(int64_t num_edges) {
        if (edge_capacity < num_edges) {
            if (sort_a) cudaFree(sort_a);
            if (sort_b) cudaFree(sort_b);
            cudaMalloc(&sort_a, num_edges * sizeof(int32_t));
            cudaMalloc(&sort_b, num_edges * sizeof(int32_t));
            edge_capacity = num_edges;
        }
    }

    void ensure_sort_temp(size_t bytes) {
        if (sort_temp_capacity < bytes) {
            if (sort_temp) cudaFree(sort_temp);
            cudaMalloc(&sort_temp, bytes);
            sort_temp_capacity = bytes;
        }
    }

    void ensure_fixed() {
        if (!fixed_allocated) {
            cudaMalloc(&dangling, sizeof(double));
            cudaMalloc(&diff, sizeof(double));
            fixed_allocated = true;
        }
    }

    ~Cache() override {
        if (out_deg) cudaFree(out_deg);
        if (inv_od) cudaFree(inv_od);
        if (contrib) cudaFree(contrib);
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (sort_a) cudaFree(sort_a);
        if (sort_b) cudaFree(sort_b);
        if (sort_temp) cudaFree(sort_temp);
        if (dangling) cudaFree(dangling);
        if (diff) cudaFree(diff);
    }
};



__global__ void sorted_histogram_kernel(
    const int* __restrict__ sorted_keys,
    int* __restrict__ histogram,
    int n
) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (tid >= n) return;

    int key = sorted_keys[tid];

    unsigned mask = __match_any_sync(0xffffffff, key);
    int leader = __ffs(mask) - 1;
    int count = __popc(mask);

    if ((threadIdx.x % 32) == leader) {
        atomicAdd(&histogram[key], count);
    }
}

__global__ void compute_inv_out_degree_kernel(
    const int* __restrict__ out_degree,
    float* __restrict__ inv_out_degree,
    int N
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int deg = out_degree[tid];
        inv_out_degree[tid] = (deg > 0) ? (1.0f / (float)deg) : 0.0f;
    }
}

__global__ void init_pr_kernel(float* __restrict__ pr, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        pr[tid] = 1.0f / (float)N;
    }
}



__global__ void compute_contrib_dangling_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ inv_out_degree,
    float* __restrict__ contrib,
    double* __restrict__ dangling_sum,
    int N
) {
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    double my_dangling = 0.0;
    if (tid < N) {
        float pr_val = pr[tid];
        float inv_deg = inv_out_degree[tid];
        contrib[tid] = pr_val * inv_deg;
        if (inv_deg == 0.0f) {
            my_dangling = (double)pr_val;
        }
    }

    double block_sum = BlockReduce(temp_storage).Sum(my_dangling);
    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(dangling_sum, block_sum);
    }
}

__global__ void spmv_high_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ contrib,
    float* __restrict__ new_pr,
    const double* __restrict__ dangling_sum,
    float alpha, float alpha_over_N, float one_minus_alpha_over_N,
    int start_vertex, int end_vertex
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int vertex = start_vertex + blockIdx.x;
    if (vertex >= end_vertex) return;

    int row_start = offsets[vertex];
    int row_end = offsets[vertex + 1];

    float sum = 0.0f;
    for (int i = row_start + threadIdx.x; i < row_end; i += BLOCK_SIZE) {
        sum += contrib[indices[i]];
    }

    float block_sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        float base = (float)(alpha_over_N * (*dangling_sum)) + one_minus_alpha_over_N;
        new_pr[vertex] = alpha * block_sum + base;
    }
}

__global__ void spmv_mid_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ contrib,
    float* __restrict__ new_pr,
    const double* __restrict__ dangling_sum,
    float alpha, float alpha_over_N, float one_minus_alpha_over_N,
    int start_vertex, int end_vertex
) {
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    int local_warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int global_warp = blockIdx.x * warps_per_block + local_warp;
    int vertex = start_vertex + global_warp;

    if (vertex >= end_vertex) return;

    int row_start = offsets[vertex];
    int row_end = offsets[vertex + 1];

    float sum = 0.0f;
    for (int i = row_start + lane; i < row_end; i += WARP_SIZE) {
        sum += contrib[indices[i]];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        float base = (float)(alpha_over_N * (*dangling_sum)) + one_minus_alpha_over_N;
        new_pr[vertex] = alpha * sum + base;
    }
}

__global__ void spmv_low_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ contrib,
    float* __restrict__ new_pr,
    const double* __restrict__ dangling_sum,
    float alpha, float alpha_over_N, float one_minus_alpha_over_N,
    int start_vertex, int end_vertex
) {
    int vertex = start_vertex + blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (vertex < end_vertex) {
        int row_start = offsets[vertex];
        int row_end = offsets[vertex + 1];

        float sum = 0.0f;
        for (int i = row_start; i < row_end; i++) {
            sum += contrib[indices[i]];
        }

        float base = (float)(alpha_over_N * (*dangling_sum)) + one_minus_alpha_over_N;
        new_pr[vertex] = alpha * sum + base;
    }
}

__global__ void update_zero_kernel(
    float* __restrict__ new_pr,
    const double* __restrict__ dangling_sum,
    float alpha_over_N, float one_minus_alpha_over_N,
    int start_vertex, int end_vertex
) {
    int vertex = start_vertex + blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (vertex < end_vertex) {
        float base = (float)(alpha_over_N * (*dangling_sum)) + one_minus_alpha_over_N;
        new_pr[vertex] = base;
    }
}

__global__ void compute_diff_kernel(
    const float* __restrict__ new_pr,
    const float* __restrict__ old_pr,
    double* __restrict__ diff_sum,
    int N
) {
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    double my_diff = 0.0;
    if (tid < N) {
        my_diff = (double)fabsf(new_pr[tid] - old_pr[tid]);
    }

    double block_sum = BlockReduce(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(diff_sum, block_sum);
    }
}



static size_t query_sort_temp_size(int num_items, int end_bit) {
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, (int*)nullptr, (int*)nullptr, num_items, 0, end_bit);
    return temp_bytes;
}

static void launch_sort_histogram(
    const int* indices, int* buf_a, int* buf_b, int* out_degree,
    int num_edges, int end_bit, void* temp_storage, size_t temp_bytes
) {
    cudaMemcpyAsync(buf_a, indices, (size_t)num_edges * sizeof(int), cudaMemcpyDeviceToDevice);
    cub::DeviceRadixSort::SortKeys(temp_storage, temp_bytes, buf_a, buf_b, num_edges, 0, end_bit);
    int grid = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sorted_histogram_kernel<<<grid, BLOCK_SIZE>>>(buf_b, out_degree, num_edges);
}

static void launch_compute_inv_out_degree(const int* out_degree, float* inv_out_degree, int N) {
    if (N <= 0) return;
    compute_inv_out_degree_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(out_degree, inv_out_degree, N);
}

static void launch_init_pr(float* pr, int N) {
    if (N <= 0) return;
    init_pr_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(pr, N);
}

static void launch_compute_contrib_dangling(const float* pr, const float* inv_out_degree,
                                             float* contrib, double* dangling_sum, int N) {
    if (N <= 0) return;
    compute_contrib_dangling_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(pr, inv_out_degree, contrib, dangling_sum, N);
}

static void launch_spmv_high(const int* offsets, const int* indices, const float* contrib,
                              float* new_pr, const double* dangling_sum,
                              float alpha, float alpha_over_N, float one_minus_alpha_over_N,
                              int start, int end) {
    if (start >= end) return;
    spmv_high_kernel<<<end - start, BLOCK_SIZE>>>(offsets, indices, contrib, new_pr,
                                                   dangling_sum, alpha, alpha_over_N,
                                                   one_minus_alpha_over_N, start, end);
}

static void launch_spmv_mid(const int* offsets, const int* indices, const float* contrib,
                              float* new_pr, const double* dangling_sum,
                              float alpha, float alpha_over_N, float one_minus_alpha_over_N,
                              int start, int end) {
    if (start >= end) return;
    int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    int grid = ((end - start) + warps_per_block - 1) / warps_per_block;
    spmv_mid_kernel<<<grid, BLOCK_SIZE>>>(offsets, indices, contrib, new_pr,
                                           dangling_sum, alpha, alpha_over_N,
                                           one_minus_alpha_over_N, start, end);
}

static void launch_spmv_low(const int* offsets, const int* indices, const float* contrib,
                              float* new_pr, const double* dangling_sum,
                              float alpha, float alpha_over_N, float one_minus_alpha_over_N,
                              int start, int end) {
    if (start >= end) return;
    int grid = ((end - start) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    spmv_low_kernel<<<grid, BLOCK_SIZE>>>(offsets, indices, contrib, new_pr,
                                           dangling_sum, alpha, alpha_over_N,
                                           one_minus_alpha_over_N, start, end);
}

static void launch_update_zero(float* new_pr, const double* dangling_sum,
                                float alpha_over_N, float one_minus_alpha_over_N,
                                int start, int end) {
    if (start >= end) return;
    int grid = ((end - start) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    update_zero_kernel<<<grid, BLOCK_SIZE>>>(new_pr, dangling_sum,
                                              alpha_over_N, one_minus_alpha_over_N, start, end);
}

static void launch_compute_diff(const float* new_pr, const float* old_pr,
                                 double* diff_sum, int N) {
    if (N <= 0) return;
    compute_diff_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(new_pr, old_pr, diff_sum, N);
}

}  

PageRankResult pagerank_seg(const graph32_t& graph,
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
    int N = graph.number_of_vertices;
    int num_edges = graph.number_of_edges;

    const auto& seg = graph.segment_offsets.value();
    int s0 = seg[0], s1 = seg[1], s2 = seg[2], s3 = seg[3], s4 = seg[4];

    float alpha_over_N = alpha / (float)N;
    float one_minus_alpha_over_N = (1.0f - alpha) / (float)N;

    
    cache.ensure_vertex((int64_t)N);
    cache.ensure_edge((int64_t)num_edges);
    cache.ensure_fixed();

    int32_t* d_out_deg = cache.out_deg;
    float* d_inv_od = cache.inv_od;
    float* d_contrib = cache.contrib;
    float* d_pr_a = cache.pr_a;
    float* d_pr_b = cache.pr_b;
    double* d_dangling = cache.dangling;
    double* d_diff = cache.diff;

    
    int end_bit = 1;
    int temp_n = N;
    while ((1 << end_bit) < temp_n) end_bit++;
    end_bit = std::min(end_bit, 32);

    size_t sort_temp_bytes = query_sort_temp_size(num_edges, end_bit);
    cache.ensure_sort_temp(sort_temp_bytes);

    cudaMemsetAsync(d_out_deg, 0, (size_t)N * sizeof(int));
    launch_sort_histogram(d_indices, cache.sort_a, cache.sort_b,
                          d_out_deg, num_edges, end_bit, cache.sort_temp, sort_temp_bytes);
    launch_compute_inv_out_degree(d_out_deg, d_inv_od, N);

    
    if (cache.l2_persist_max > 0) {
        size_t contrib_bytes = (size_t)N * sizeof(float);
        size_t persist_bytes = std::min(contrib_bytes, cache.l2_persist_max);

        cudaStreamAttrValue attr;
        attr.accessPolicyWindow.base_ptr = (void*)d_contrib;
        attr.accessPolicyWindow.num_bytes = persist_bytes;
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
    }

    
    float* d_pr_cur = d_pr_a;
    float* d_pr_new = d_pr_b;

    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(d_pr_cur, initial_pageranks,
                        (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        launch_init_pr(d_pr_cur, N);
    }

    bool converged = false;
    size_t iterations = 0;
    double h_diff;
    const int check_interval = 4;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        cudaMemsetAsync(d_dangling, 0, sizeof(double));

        launch_compute_contrib_dangling(d_pr_cur, d_inv_od, d_contrib, d_dangling, N);

        launch_spmv_high(d_offsets, d_indices, d_contrib, d_pr_new,
                         d_dangling, alpha, alpha_over_N, one_minus_alpha_over_N, s0, s1);
        launch_spmv_mid(d_offsets, d_indices, d_contrib, d_pr_new,
                        d_dangling, alpha, alpha_over_N, one_minus_alpha_over_N, s1, s2);
        launch_spmv_low(d_offsets, d_indices, d_contrib, d_pr_new,
                        d_dangling, alpha, alpha_over_N, one_minus_alpha_over_N, s2, s3);
        launch_update_zero(d_pr_new, d_dangling,
                           alpha_over_N, one_minus_alpha_over_N, s3, s4);

        float* tmp = d_pr_cur;
        d_pr_cur = d_pr_new;
        d_pr_new = tmp;

        iterations = iter + 1;

        if ((iterations % check_interval == 0) || (iterations == max_iterations)) {
            cudaMemsetAsync(d_diff, 0, sizeof(double));
            launch_compute_diff(d_pr_cur, d_pr_new, d_diff, N);
            cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);

            if (h_diff < (double)epsilon) {
                converged = true;
                break;
            }
        }
    }

    
    if (cache.l2_persist_max > 0) {
        cudaStreamAttrValue attr;
        attr.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
        cudaCtxResetPersistingL2Cache();
    }

    
    cudaMemcpyAsync(pageranks, d_pr_cur,
                    (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice);

    return PageRankResult{iterations, converged};
}

}  
