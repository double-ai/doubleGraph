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
#include <cub/block/block_reduce.cuh>
#include <cstdint>
#include <utility>

namespace aai {

namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int DIFF_SLOTS = 256;





struct Cache : Cacheable {
    float* h_diff = nullptr;    
    float* x_buf = nullptr;     
    float* diff_buf = nullptr;  
    int32_t x_buf_capacity = 0;

    Cache() {
        cudaHostAlloc(&h_diff, sizeof(float), cudaHostAllocDefault);
        cudaMalloc(&diff_buf, 257 * sizeof(float));
    }

    void ensure(int32_t num_vertices) {
        if (x_buf_capacity < num_vertices) {
            if (x_buf) cudaFree(x_buf);
            cudaMalloc(&x_buf, (size_t)num_vertices * sizeof(float));
            x_buf_capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (h_diff) cudaFreeHost(h_diff);
        if (x_buf) cudaFree(x_buf);
        if (diff_buf) cudaFree(diff_buf);
    }
};





__global__ void __launch_bounds__(BLOCK_SIZE)
spmv_unified(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha, float beta, const float* __restrict__ betas, int use_betas,
    int seg0, int seg1, int seg2, int seg4,
    int high_blocks, int mid_blocks, int total_low_blocks,
    float* __restrict__ diff_slots
) {
    int bid = blockIdx.x;

    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage temp;

    if (bid < high_blocks) {
        
        int v = bid + seg0;
        int start = offsets[v];
        int end = offsets[v + 1];

        float sum = 0.0f;
        for (int e = start + threadIdx.x; e < end; e += BLOCK_SIZE)
            sum += weights[e] * x_old[indices[e]];

        sum = BR(temp).Sum(sum);

        float diff = 0.0f;
        if (threadIdx.x == 0) {
            float bv = use_betas ? betas[v] : beta;
            float nv = alpha * sum + bv;
            diff = fabsf(nv - x_old[v]);
            x_new[v] = nv;
            atomicAdd(&diff_slots[bid % DIFF_SLOTS], diff);
        }
    } else if (bid < high_blocks + mid_blocks) {
        
        constexpr int WPB = BLOCK_SIZE / 32;
        int local_bid = bid - high_blocks;
        int warp = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        int v = local_bid * WPB + warp + seg1;

        float diff = 0.0f;
        if (v < seg2) {
            int start = offsets[v];
            int end = offsets[v + 1];

            float sum = 0.0f;
            for (int e = start + lane; e < end; e += 32)
                sum += weights[e] * x_old[indices[e]];

            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                sum += __shfl_down_sync(0xFFFFFFFF, sum, o);

            if (lane == 0) {
                float bv = use_betas ? betas[v] : beta;
                float nv = alpha * sum + bv;
                diff = fabsf(nv - x_old[v]);
                x_new[v] = nv;
            }
        }

        float bd = BR(temp).Sum(diff);
        if (threadIdx.x == 0) atomicAdd(&diff_slots[bid % DIFF_SLOTS], bd);
    } else {
        
        int local_bid = bid - high_blocks - mid_blocks;
        float thread_diff = 0.0f;

        for (int v = local_bid * BLOCK_SIZE + threadIdx.x + seg2;
             v < seg4;
             v += total_low_blocks * BLOCK_SIZE) {
            int start = offsets[v];
            int end = offsets[v + 1];

            float sum = 0.0f;
            for (int e = start; e < end; ++e)
                sum += weights[e] * x_old[indices[e]];

            float bv = use_betas ? betas[v] : beta;
            float nv = alpha * sum + bv;
            thread_diff += fabsf(nv - x_old[v]);
            x_new[v] = nv;
        }

        float bd = BR(temp).Sum(thread_diff);
        if (threadIdx.x == 0) atomicAdd(&diff_slots[bid % DIFF_SLOTS], bd);
    }
}





__global__ void reduce_diff_slots(const float* __restrict__ slots, float* __restrict__ result) {
    float sum = 0.0f;
    for (int i = threadIdx.x; i < DIFF_SLOTS; i += 32)
        sum += slots[i];

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, o);

    if (threadIdx.x == 0) *result = sum;
}





__global__ void fill_scalar_kernel(float* __restrict__ x, float val, int n) {
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE)
        x[i] = val;
}

__global__ void __launch_bounds__(BLOCK_SIZE)
l2_norm_sq_kernel(const float* __restrict__ x, int n, float* __restrict__ out) {
    float sum = 0.0f;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        float v = x[i];
        sum += v * v;
    }
    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage temp;
    sum = BR(temp).Sum(sum);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

__global__ void __launch_bounds__(BLOCK_SIZE)
normalize_kernel(float* __restrict__ x, int n, const float* __restrict__ norm_sq) {
    float inv = rsqrtf(*norm_sq);
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE)
        x[i] *= inv;
}





static void do_spmv_iteration(
    const int32_t* offsets, const int32_t* indices, const float* weights,
    const float* x_old, float* x_new,
    float alpha, float beta, const float* betas, int use_betas,
    int seg0, int seg1, int seg2, int seg3, int seg4,
    float* diff_slots, float* diff_result
) {
    cudaMemsetAsync(diff_slots, 0, DIFF_SLOTS * sizeof(float), 0);

    int high_blocks = seg1 - seg0;
    int mid_verts = seg2 - seg1;
    int mid_blocks = (mid_verts + (BLOCK_SIZE/32) - 1) / (BLOCK_SIZE/32);
    int low_verts = seg4 - seg2;
    int low_blocks = (low_verts + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (low_blocks > 4096) low_blocks = 4096;

    int total_blocks = high_blocks + mid_blocks + low_blocks;
    if (total_blocks == 0) return;

    spmv_unified<<<total_blocks, BLOCK_SIZE>>>(
        offsets, indices, weights, x_old, x_new,
        alpha, beta, betas, use_betas,
        seg0, seg1, seg2, seg4,
        high_blocks, mid_blocks, low_blocks,
        diff_slots);

    reduce_diff_slots<<<1, 32>>>(diff_slots, diff_result);
}

static void do_fill(float* x, float val, int n) {
    int nb = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (nb > 1024) nb = 1024;
    fill_scalar_kernel<<<nb, BLOCK_SIZE>>>(x, val, n);
}

static void do_normalize(float* x, int n, float* buf) {
    cudaMemsetAsync(buf, 0, sizeof(float), 0);
    int nb = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (nb > 1024) nb = 1024;
    l2_norm_sq_kernel<<<nb, BLOCK_SIZE>>>(x, n, buf);
    normalize_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(x, n, buf);
}

}  

katz_centrality_result_t katz_centrality_seg(const graph32_t& graph,
                         const float* edge_weights,
                         float* centralities,
                         float alpha,
                         float beta,
                         const float* betas,
                         float epsilon,
                         std::size_t max_iterations,
                         bool has_initial_guess,
                         bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const float* d_weights = edge_weights;

    const auto& seg_vec = graph.segment_offsets.value();
    int seg[5] = {seg_vec[0], seg_vec[1], seg_vec[2], seg_vec[3], seg_vec[4]};

    int use_betas = (betas != nullptr) ? 1 : 0;

    cache.ensure(num_vertices);

    float* d_x0 = centralities;
    float* d_x1 = cache.x_buf;
    float* d_diff_slots = cache.diff_buf;
    float* d_diff_result = d_diff_slots + 256;

    std::size_t iterations = 0;
    bool converged = false;

    if (!has_initial_guess && max_iterations > 0) {
        if (betas != nullptr) {
            cudaMemcpyAsync(d_x0, betas, num_vertices * sizeof(float),
                           cudaMemcpyDeviceToDevice, 0);
        } else {
            do_fill(d_x0, beta, num_vertices);
        }
        iterations = 1;
    } else if (!has_initial_guess) {
        cudaMemsetAsync(d_x0, 0, num_vertices * sizeof(float), 0);
    }
    

    float* d_cur = d_x0;
    float* d_next = d_x1;

    while (!converged && iterations < max_iterations) {
        do_spmv_iteration(
            d_offsets, d_indices, d_weights,
            d_cur, d_next,
            alpha, beta, betas, use_betas,
            seg[0], seg[1], seg[2], seg[3], seg[4],
            d_diff_slots, d_diff_result
        );

        cudaMemcpy(cache.h_diff, d_diff_result, sizeof(float), cudaMemcpyDeviceToHost);

        std::swap(d_cur, d_next);
        iterations++;

        if (*cache.h_diff < epsilon) {
            converged = true;
        }
    }

    if (d_cur != d_x0) {
        cudaMemcpy(d_x0, d_cur, num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    if (normalize) {
        do_normalize(d_x0, num_vertices, d_diff_result);
        cudaDeviceSynchronize();
    }

    return {iterations, converged};
}

}  
