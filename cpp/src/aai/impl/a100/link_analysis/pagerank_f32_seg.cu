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
#include <cstdint>

namespace aai {

namespace {




struct Cache : Cacheable {
    float* new_pr = nullptr;
    float* out_w = nullptr;
    float* eff_w = nullptr;
    int* dangling_indices = nullptr;
    float* scalar_buf = nullptr;  

    int64_t new_pr_capacity = 0;
    int64_t out_w_capacity = 0;
    int64_t eff_w_capacity = 0;
    int64_t dangling_capacity = 0;
    bool scalar_buf_allocated = false;

    void ensure(int32_t num_vertices, int32_t num_edges, bool need_out_w) {
        if (new_pr_capacity < num_vertices) {
            if (new_pr) cudaFree(new_pr);
            cudaMalloc(&new_pr, (int64_t)num_vertices * sizeof(float));
            new_pr_capacity = num_vertices;
        }
        if (need_out_w && out_w_capacity < num_vertices) {
            if (out_w) cudaFree(out_w);
            cudaMalloc(&out_w, (int64_t)num_vertices * sizeof(float));
            out_w_capacity = num_vertices;
        }
        if (eff_w_capacity < num_edges) {
            if (eff_w) cudaFree(eff_w);
            cudaMalloc(&eff_w, (int64_t)num_edges * sizeof(float));
            eff_w_capacity = num_edges;
        }
        if (dangling_capacity < num_vertices) {
            if (dangling_indices) cudaFree(dangling_indices);
            cudaMalloc(&dangling_indices, (int64_t)num_vertices * sizeof(int));
            dangling_capacity = num_vertices;
        }
        if (!scalar_buf_allocated) {
            cudaMalloc(&scalar_buf, 3 * sizeof(float));
            scalar_buf_allocated = true;
        }
    }

    ~Cache() override {
        if (new_pr) cudaFree(new_pr);
        if (out_w) cudaFree(out_w);
        if (eff_w) cudaFree(eff_w);
        if (dangling_indices) cudaFree(dangling_indices);
        if (scalar_buf) cudaFree(scalar_buf);
    }
};




__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ float shared[NUM_WARPS];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}




__global__ void compute_out_weights_kernel(
    const int* __restrict__ indices, const float* __restrict__ weights,
    float* __restrict__ out_w, int num_edges)
{
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_edges;
         e += gridDim.x * blockDim.x) {
        atomicAdd(&out_w[indices[e]], weights[e]);
    }
}




__global__ void precompute_and_dangling_kernel(
    const int* __restrict__ indices, const float* __restrict__ weights,
    const float* __restrict__ out_w, float* __restrict__ eff_w,
    float* __restrict__ pr,
    int* __restrict__ dangling_indices, int* __restrict__ d_dangling_count,
    float alpha, int num_edges, int num_vertices, float init_val)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    
    for (int e = tid; e < num_edges; e += stride) {
        float ow = out_w[indices[e]];
        eff_w[e] = (ow > 0.0f) ? (alpha * weights[e] / ow) : 0.0f;
    }

    
    for (int v = tid; v < num_vertices; v += stride) {
        pr[v] = init_val;
        if (out_w[v] == 0.0f) {
            int idx = atomicAdd(d_dangling_count, 1);
            dangling_indices[idx] = v;
        }
    }
}




__global__ void dangling_sum_compact_kernel(
    const float* __restrict__ pr, const int* __restrict__ dangling_indices,
    float* __restrict__ d_dangling_sum, int num_dangling)
{
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_dangling;
         i += gridDim.x * blockDim.x) {
        sum += pr[dangling_indices[i]];
    }
    sum = block_reduce_sum<256>(sum);
    if (threadIdx.x == 0 && sum != 0.0f) atomicAdd(d_dangling_sum, sum);
}






__global__ void spmv_high_degree_kernel(
    const int* __restrict__ offsets, const int* __restrict__ indices,
    const float* __restrict__ eff_w,
    const float* __restrict__ pr, float* __restrict__ new_pr,
    int seg_start, int seg_end,
    float base,
    float* __restrict__ d_diff)
{
    int v = blockIdx.x + seg_start;
    if (v >= seg_end) return;

    int start = offsets[v];
    int end = offsets[v + 1];
    float sum = 0.0f;
    for (int e = start + threadIdx.x; e < end; e += blockDim.x) {
        sum += pr[indices[e]] * eff_w[e];
    }

    sum = block_reduce_sum<256>(sum);

    if (threadIdx.x == 0) {
        float val = sum + base;
        float diff = fabsf(val - pr[v]);
        new_pr[v] = val;
        atomicAdd(d_diff, diff);
    }
}


__global__ void spmv_mid_degree_kernel(
    const int* __restrict__ offsets, const int* __restrict__ indices,
    const float* __restrict__ eff_w,
    const float* __restrict__ pr, float* __restrict__ new_pr,
    int seg_start, int seg_end,
    float base,
    float* __restrict__ d_diff)
{
    int warps_per_block = blockDim.x >> 5;
    int global_warp = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    int v = global_warp + seg_start;

    float diff_val = 0.0f;

    if (v < seg_end) {
        int start = offsets[v];
        int end = offsets[v + 1];
        float sum = 0.0f;
        for (int e = start + lane; e < end; e += 32) {
            sum += pr[indices[e]] * eff_w[e];
        }

        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            float val = sum + base;
            diff_val = fabsf(val - pr[v]);
            new_pr[v] = val;
        }
    }

    diff_val = block_reduce_sum<256>(diff_val);
    if (threadIdx.x == 0 && diff_val > 0.0f) atomicAdd(d_diff, diff_val);
}


__global__ void spmv_low_degree_kernel(
    const int* __restrict__ offsets, const int* __restrict__ indices,
    const float* __restrict__ eff_w,
    const float* __restrict__ pr, float* __restrict__ new_pr,
    int seg_start, int seg_end,
    float base,
    float* __restrict__ d_diff)
{
    float diff_val = 0.0f;
    int v = blockIdx.x * blockDim.x + threadIdx.x + seg_start;

    if (v < seg_end) {
        int start = offsets[v];
        int end = offsets[v + 1];
        float sum = 0.0f;
        for (int e = start; e < end; e++) {
            sum += pr[indices[e]] * eff_w[e];
        }

        float val = sum + base;
        diff_val = fabsf(val - pr[v]);
        new_pr[v] = val;
    }

    diff_val = block_reduce_sum<256>(diff_val);
    if (threadIdx.x == 0) atomicAdd(d_diff, diff_val);
}


__global__ void spmv_zero_degree_kernel(
    float* __restrict__ new_pr, const float* __restrict__ pr,
    int seg_start, int seg_end,
    float base,
    float* __restrict__ d_diff)
{
    float diff_val = 0.0f;
    int v = blockIdx.x * blockDim.x + threadIdx.x + seg_start;

    if (v < seg_end) {
        diff_val = fabsf(base - pr[v]);
        new_pr[v] = base;
    }

    diff_val = block_reduce_sum<256>(diff_val);
    if (threadIdx.x == 0) atomicAdd(d_diff, diff_val);
}





void launch_precompute(const int* indices, const float* weights,
                       float* out_w, float* eff_w, float* pr,
                       int* dangling_indices, int* d_dangling_count,
                       float alpha, int num_vertices, int num_edges)
{
    
    cudaMemsetAsync(out_w, 0, num_vertices * sizeof(float));
    cudaMemsetAsync(d_dangling_count, 0, sizeof(int));

    int block = 256;
    int grid = (num_edges + block - 1) / block;
    if (grid > 65535) grid = 65535;
    compute_out_weights_kernel<<<grid, block>>>(indices, weights, out_w, num_edges);

    
    float init_val = 1.0f / num_vertices;
    int n = num_edges > num_vertices ? num_edges : num_vertices;
    int grid2 = (n + block - 1) / block;
    if (grid2 > 65535) grid2 = 65535;
    precompute_and_dangling_kernel<<<grid2, block>>>(
        indices, weights, out_w, eff_w, pr,
        dangling_indices, d_dangling_count,
        alpha, num_edges, num_vertices, init_val);
}

void launch_dangling_sum_compact(const float* pr, const int* dangling_indices,
                                  float* d_dangling_sum, int num_dangling)
{
    cudaMemsetAsync(d_dangling_sum, 0, sizeof(float));
    if (num_dangling == 0) return;

    int block = 256;
    int grid = (num_dangling + block - 1) / block;
    if (grid > 1024) grid = 1024;
    dangling_sum_compact_kernel<<<grid, block>>>(pr, dangling_indices, d_dangling_sum, num_dangling);
}

void launch_spmv(
    const int* offsets, const int* indices, const float* eff_w,
    const float* pr, float* new_pr,
    float base, float* d_diff,
    const int* h_seg)
{
    cudaMemsetAsync(d_diff, 0, sizeof(float));

    int seg0 = h_seg[0], seg1 = h_seg[1], seg2 = h_seg[2];
    int seg3 = h_seg[3], seg4 = h_seg[4];

    
    if (seg1 > seg0) {
        spmv_high_degree_kernel<<<seg1 - seg0, 256>>>(
            offsets, indices, eff_w, pr, new_pr,
            seg0, seg1, base, d_diff);
    }

    
    if (seg2 > seg1) {
        int num_warps = seg2 - seg1;
        int warps_per_block = 8;
        int grid = (num_warps + warps_per_block - 1) / warps_per_block;
        spmv_mid_degree_kernel<<<grid, 256>>>(
            offsets, indices, eff_w, pr, new_pr,
            seg1, seg2, base, d_diff);
    }

    
    if (seg3 > seg2) {
        int block = 256;
        int grid = (seg3 - seg2 + block - 1) / block;
        spmv_low_degree_kernel<<<grid, block>>>(
            offsets, indices, eff_w, pr, new_pr,
            seg2, seg3, base, d_diff);
    }

    
    if (seg4 > seg3) {
        int block = 256;
        int grid = (seg4 - seg3 + block - 1) / block;
        spmv_zero_degree_kernel<<<grid, block>>>(
            new_pr, pr,
            seg3, seg4, base, d_diff);
    }
}

}  

PageRankResult pagerank_seg(const graph32_t& graph,
                            const float* edge_weights,
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
    const int* d_offsets = graph.offsets;
    const int* d_indices = graph.indices;

    const auto& seg = graph.segment_offsets.value();
    int32_t h_seg[5] = {seg[0], seg[1], seg[2], seg[3], seg[4]};

    cache.ensure(num_vertices, num_edges, true);

    float* d_eff_w = cache.eff_w;
    int* d_dangling_indices = cache.dangling_indices;
    float* d_dangling_sum = cache.scalar_buf;
    float* d_diff = cache.scalar_buf + 1;
    int* d_dangling_count = reinterpret_cast<int*>(cache.scalar_buf + 2);

    
    launch_precompute(d_indices, edge_weights, cache.out_w, d_eff_w, pageranks,
                     d_dangling_indices, d_dangling_count,
                     alpha, num_vertices, num_edges);

    
    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(pageranks, initial_pageranks,
                       num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    
    int h_dangling_count;
    cudaMemcpy(&h_dangling_count, d_dangling_count, sizeof(int), cudaMemcpyDeviceToHost);

    float one_minus_alpha_over_n = (1.0f - alpha) / num_vertices;
    float alpha_over_n = alpha / num_vertices;

    float* d_pr = pageranks;
    float* d_new_pr = cache.new_pr;

    
    std::size_t iterations = 0;
    bool converged = false;
    float h_diff;
    float h_dangling_sum;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        if (h_dangling_count > 0) {
            launch_dangling_sum_compact(d_pr, d_dangling_indices,
                                       d_dangling_sum, h_dangling_count);
            
            cudaMemcpy(&h_dangling_sum, d_dangling_sum, sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            h_dangling_sum = 0.0f;
        }

        float base = one_minus_alpha_over_n + alpha_over_n * h_dangling_sum;

        
        launch_spmv(d_offsets, d_indices, d_eff_w,
                   d_pr, d_new_pr,
                   base, d_diff, h_seg);

        
        float* tmp = d_pr;
        d_pr = d_new_pr;
        d_new_pr = tmp;

        iterations = iter + 1;

        
        if ((iter + 1) % 4 == 0 || iter == max_iterations - 1) {
            cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_diff < epsilon) {
                converged = true;
                break;
            }
        }
    }

    
    if (d_pr != pageranks) {
        cudaMemcpyAsync(pageranks, d_pr, num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return {iterations, converged};
}

}  
