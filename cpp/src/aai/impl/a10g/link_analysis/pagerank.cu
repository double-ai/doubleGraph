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

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;





__global__ void compute_out_degrees_kernel(
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ out_degree,
    int32_t num_edges)
{
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < num_edges) {
        atomicAdd(&out_degree[indices[idx]], 1);
    }
}

__global__ void init_kernel(
    const int32_t* __restrict__ out_degree,
    float* __restrict__ inv_out_degree,
    float* __restrict__ pr,
    int32_t num_vertices,
    bool has_initial_guess,
    const float* __restrict__ initial_pageranks)
{
    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (v < num_vertices) {
        int d = out_degree[v];
        inv_out_degree[v] = (d > 0) ? (1.0f / (float)d) : 0.0f;
        pr[v] = has_initial_guess ? initial_pageranks[v] : (1.0f / (float)num_vertices);
    }
}





__global__ __launch_bounds__(BLOCK_SIZE)
void prepare_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ inv_out_degree,
    float* __restrict__ pr_scaled,
    float* __restrict__ d_dangling_sum,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float dangling = 0.0f;

    if (v < num_vertices) {
        float p = pr[v];
        float inv_d = inv_out_degree[v];
        pr_scaled[v] = p * inv_d;
        if (inv_d == 0.0f) dangling = p;
    }

    float block_sum = BlockReduce(temp).Sum(dangling);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(d_dangling_sum, block_sum);
    }
}





__global__ __launch_bounds__(BLOCK_SIZE)
void spmv_update_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_scaled,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    float* __restrict__ d_l1_norm,
    const float* __restrict__ d_dangling_sum,
    float one_minus_alpha_over_n,
    float alpha,
    float alpha_over_n,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float diff = 0.0f;

    if (v < num_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];

        float spmv_val = 0.0f;
        for (int j = start; j < end; j++) {
            spmv_val += pr_scaled[indices[j]];
        }

        float base = one_minus_alpha_over_n + alpha_over_n * (*d_dangling_sum);
        float new_val = base + alpha * spmv_val;
        pr_new[v] = new_val;
        diff = fabsf(new_val - pr_old[v]);
    }

    float block_diff = BlockReduce(temp).Sum(diff);
    if (threadIdx.x == 0) {
        atomicAdd(d_l1_norm, block_diff);
    }
}





__global__ __launch_bounds__(BLOCK_SIZE, 6)
void spmv_update_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_scaled,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    float* __restrict__ d_l1_norm,
    const float* __restrict__ d_dangling_sum,
    float one_minus_alpha_over_n,
    float alpha,
    float alpha_over_n,
    int32_t num_vertices)
{
    int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int local_warp = threadIdx.x / WARP_SIZE;

    __shared__ float warp_diffs[WARPS_PER_BLOCK];

    float diff = 0.0f;

    if (global_warp < num_vertices) {
        int v = global_warp;
        int start = offsets[v];
        int end = offsets[v + 1];

        float spmv_val = 0.0f;
        for (int j = start + lane; j < end; j += WARP_SIZE) {
            spmv_val += pr_scaled[indices[j]];
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            spmv_val += __shfl_down_sync(0xffffffff, spmv_val, offset);
        }

        if (lane == 0) {
            float base = one_minus_alpha_over_n + alpha_over_n * (*d_dangling_sum);
            float new_val = base + alpha * spmv_val;
            pr_new[v] = new_val;
            diff = fabsf(new_val - pr_old[v]);
        }
    }

    
    if (lane == 0) {
        warp_diffs[local_warp] = diff;
    }
    __syncthreads();

    
    if (threadIdx.x < WARPS_PER_BLOCK) {
        diff = warp_diffs[threadIdx.x];
    } else {
        diff = 0.0f;
    }

    if (threadIdx.x < WARP_SIZE) {
        #pragma unroll
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1) {
            diff += __shfl_down_sync(0xffffffff, diff, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(d_l1_norm, diff);
        }
    }
}





void launch_compute_out_degrees(const int32_t* indices, int32_t* out_degree,
    int32_t num_vertices, int32_t num_edges, cudaStream_t stream) {
    cudaMemsetAsync(out_degree, 0, num_vertices * sizeof(int32_t), stream);
    int grid = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_out_degrees_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(indices, out_degree, num_edges);
}

void launch_init(const int32_t* out_degree, float* inv_out_degree, float* pr,
    int32_t num_vertices, bool has_initial_guess, const float* initial_pageranks,
    cudaStream_t stream) {
    int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(out_degree, inv_out_degree, pr,
        num_vertices, has_initial_guess, initial_pageranks);
}

void launch_prepare(const float* pr, const float* inv_out_degree, float* pr_scaled,
    float* d_dangling_sum, int32_t num_vertices, cudaStream_t stream) {
    cudaMemsetAsync(d_dangling_sum, 0, sizeof(float), stream);
    int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    prepare_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(pr, inv_out_degree, pr_scaled,
        d_dangling_sum, num_vertices);
}

void launch_spmv_update_thread(const int32_t* offsets, const int32_t* indices,
    const float* pr_scaled, const float* pr_old, float* pr_new,
    float* d_l1_norm, const float* d_dangling_sum,
    float one_minus_alpha_over_n, float alpha, float alpha_over_n,
    int32_t num_vertices, cudaStream_t stream) {
    cudaMemsetAsync(d_l1_norm, 0, sizeof(float), stream);
    int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    spmv_update_thread_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(offsets, indices, pr_scaled,
        pr_old, pr_new, d_l1_norm, d_dangling_sum,
        one_minus_alpha_over_n, alpha, alpha_over_n, num_vertices);
}

void launch_spmv_update_warp(const int32_t* offsets, const int32_t* indices,
    const float* pr_scaled, const float* pr_old, float* pr_new,
    float* d_l1_norm, const float* d_dangling_sum,
    float one_minus_alpha_over_n, float alpha, float alpha_over_n,
    int32_t num_vertices, cudaStream_t stream) {
    cudaMemsetAsync(d_l1_norm, 0, sizeof(float), stream);
    int grid = (num_vertices + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    spmv_update_warp_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(offsets, indices, pr_scaled,
        pr_old, pr_new, d_l1_norm, d_dangling_sum,
        one_minus_alpha_over_n, alpha, alpha_over_n, num_vertices);
}

void set_l2_persist(const float* ptr, size_t num_bytes, cudaStream_t stream) {
    int max_persist;
    cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, 0);
    if (max_persist > 0 && num_bytes > 0) {
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t)max_persist);
        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.base_ptr = (void*)ptr;
        size_t win = num_bytes;
        if (win > (size_t)max_persist) win = (size_t)max_persist;
        attr.accessPolicyWindow.num_bytes = win;
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    }
}

void reset_l2_persist(cudaStream_t stream) {
    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr = nullptr;
    attr.accessPolicyWindow.num_bytes = 0;
    attr.accessPolicyWindow.hitRatio = 0.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
    cudaCtxResetPersistingL2Cache();
}





struct Cache : Cacheable {
    float* h_l1_norm_pinned = nullptr;
    int32_t* out_degree = nullptr;
    float* inv_out_degree = nullptr;
    float* pr_a = nullptr;
    float* pr_b = nullptr;
    float* pr_scaled = nullptr;
    float* scalars = nullptr;

    int32_t out_degree_cap = 0;
    int32_t inv_out_degree_cap = 0;
    int32_t pr_a_cap = 0;
    int32_t pr_b_cap = 0;
    int32_t pr_scaled_cap = 0;
    bool scalars_allocated = false;
    bool pinned_allocated = false;

    void ensure(int32_t num_vertices) {
        if (!pinned_allocated) {
            cudaHostAlloc(&h_l1_norm_pinned, sizeof(float), cudaHostAllocDefault);
            pinned_allocated = true;
        }
        if (!scalars_allocated) {
            cudaMalloc(&scalars, 2 * sizeof(float));
            scalars_allocated = true;
        }
        if (out_degree_cap < num_vertices) {
            if (out_degree) cudaFree(out_degree);
            cudaMalloc(&out_degree, (size_t)num_vertices * sizeof(int32_t));
            out_degree_cap = num_vertices;
        }
        if (inv_out_degree_cap < num_vertices) {
            if (inv_out_degree) cudaFree(inv_out_degree);
            cudaMalloc(&inv_out_degree, (size_t)num_vertices * sizeof(float));
            inv_out_degree_cap = num_vertices;
        }
        if (pr_a_cap < num_vertices) {
            if (pr_a) cudaFree(pr_a);
            cudaMalloc(&pr_a, (size_t)num_vertices * sizeof(float));
            pr_a_cap = num_vertices;
        }
        if (pr_b_cap < num_vertices) {
            if (pr_b) cudaFree(pr_b);
            cudaMalloc(&pr_b, (size_t)num_vertices * sizeof(float));
            pr_b_cap = num_vertices;
        }
        if (pr_scaled_cap < num_vertices) {
            if (pr_scaled) cudaFree(pr_scaled);
            cudaMalloc(&pr_scaled, (size_t)num_vertices * sizeof(float));
            pr_scaled_cap = num_vertices;
        }
    }

    ~Cache() override {
        if (h_l1_norm_pinned) cudaFreeHost(h_l1_norm_pinned);
        if (out_degree) cudaFree(out_degree);
        if (inv_out_degree) cudaFree(inv_out_degree);
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (pr_scaled) cudaFree(pr_scaled);
        if (scalars) cudaFree(scalars);
    }
};

}  

PageRankResult pagerank(const graph32_t& graph,
                        float* pageranks,
                        const float* precomputed_vertex_out_weight_sums,
                        float alpha,
                        float epsilon,
                        std::size_t max_iterations,
                        const float* initial_pageranks) {
    (void)precomputed_vertex_out_weight_sums;

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    cudaStream_t stream = 0;

    cache.ensure(num_vertices);

    float* d_dangling_sum = cache.scalars;
    float* d_l1_norm = cache.scalars + 1;

    
    launch_compute_out_degrees(indices, cache.out_degree, num_vertices, num_edges, stream);

    
    bool has_initial_guess = (initial_pageranks != nullptr);
    launch_init(cache.out_degree, cache.inv_out_degree, cache.pr_a,
                num_vertices, has_initial_guess, initial_pageranks, stream);

    float one_minus_alpha_over_n = (1.0f - alpha) / (float)num_vertices;
    float alpha_over_n = alpha / (float)num_vertices;

    float* d_pr = cache.pr_a;
    float* d_pr_new = cache.pr_b;

    
    float avg_degree = (num_vertices > 0) ? (float)num_edges / (float)num_vertices : 0.0f;
    bool use_warp = (avg_degree > 8.0f);

    
    size_t pr_scaled_bytes = (size_t)num_vertices * sizeof(float);
    set_l2_persist(cache.pr_scaled, pr_scaled_bytes, stream);

    bool converged = false;
    size_t iterations = 0;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        launch_prepare(d_pr, cache.inv_out_degree, cache.pr_scaled,
                       d_dangling_sum, num_vertices, stream);

        if (use_warp) {
            launch_spmv_update_warp(offsets, indices, cache.pr_scaled,
                d_pr, d_pr_new, d_l1_norm, d_dangling_sum,
                one_minus_alpha_over_n, alpha, alpha_over_n,
                num_vertices, stream);
        } else {
            launch_spmv_update_thread(offsets, indices, cache.pr_scaled,
                d_pr, d_pr_new, d_l1_norm, d_dangling_sum,
                one_minus_alpha_over_n, alpha, alpha_over_n,
                num_vertices, stream);
        }

        cudaMemcpyAsync(cache.h_l1_norm_pinned, d_l1_norm, sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations++;

        float* temp = d_pr;
        d_pr = d_pr_new;
        d_pr_new = temp;

        if (*cache.h_l1_norm_pinned < epsilon) {
            converged = true;
            break;
        }
    }

    
    reset_l2_persist(stream);

    
    cudaMemcpyAsync(pageranks, d_pr, (size_t)num_vertices * sizeof(float),
        cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return PageRankResult{iterations, converged};
}

}  
