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

struct Cache : Cacheable {
    float* buf0 = nullptr;
    float* buf1 = nullptr;
    float* partial_diffs = nullptr;
    float* diff_result = nullptr;
    float* norm_buf = nullptr;

    int64_t buf_capacity = 0;
    int64_t pd_capacity = 0;

    void ensure(int64_t num_vertices, int64_t max_blocks) {
        if (buf_capacity < num_vertices) {
            if (buf0) cudaFree(buf0);
            if (buf1) cudaFree(buf1);
            if (diff_result) cudaFree(diff_result);
            if (norm_buf) cudaFree(norm_buf);
            cudaMalloc(&buf0, num_vertices * sizeof(float));
            cudaMalloc(&buf1, num_vertices * sizeof(float));
            cudaMalloc(&diff_result, sizeof(float));
            cudaMalloc(&norm_buf, sizeof(float));
            buf_capacity = num_vertices;
        }
        if (pd_capacity < max_blocks) {
            if (partial_diffs) cudaFree(partial_diffs);
            cudaMalloc(&partial_diffs, max_blocks * sizeof(float));
            pd_capacity = max_blocks;
        }
    }

    ~Cache() override {
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (partial_diffs) cudaFree(partial_diffs);
        if (diff_result) cudaFree(diff_result);
        if (norm_buf) cudaFree(norm_buf);
    }
};





__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <int BLOCK_SIZE>
__global__ void spmv_unified_fused(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha, float beta, const float* __restrict__ betas,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg4,
    int32_t high_blocks, int32_t mid_blocks,
    float* __restrict__ partial_diffs
) {
    int32_t bid = blockIdx.x;
    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage ts;

    if (bid < high_blocks) {
        int32_t v = seg0 + bid;
        int32_t rb = offsets[v];
        int32_t re = offsets[v + 1];
        float sum = 0.0f;
        for (int32_t e = rb + threadIdx.x; e < re; e += BLOCK_SIZE) {
            uint32_t mw = edge_mask[e >> 5];
            if (mw & (1u << (e & 31)))
                sum += weights[e] * x_old[indices[e]];
        }
        float bs = BR(ts).Sum(sum);
        if (threadIdx.x == 0) {
            float b = betas ? betas[v] : beta;
            float nv = alpha * bs + b;
            x_new[v] = nv;
            partial_diffs[bid] = fabsf(nv - x_old[v]);
        }
    } else if (bid < high_blocks + mid_blocks) {
        constexpr int WPB = BLOCK_SIZE / 32;
        int32_t local_bid = bid - high_blocks;
        int32_t wib = threadIdx.x >> 5;
        int32_t lane = threadIdx.x & 31;
        int32_t v = seg1 + local_bid * WPB + wib;
        float diff = 0.0f;
        if (v < seg2) {
            int32_t rb = offsets[v];
            int32_t re = offsets[v + 1];
            float sum = 0.0f;
            for (int32_t e = rb + lane; e < re; e += 32) {
                uint32_t mw = edge_mask[e >> 5];
                if (mw & (1u << (e & 31)))
                    sum += weights[e] * x_old[indices[e]];
            }
            sum = warp_reduce_sum(sum);
            if (lane == 0) {
                float b = betas ? betas[v] : beta;
                float nv = alpha * sum + b;
                x_new[v] = nv;
                diff = fabsf(nv - x_old[v]);
            }
        }
        float bd = BR(ts).Sum(diff);
        if (threadIdx.x == 0) partial_diffs[bid] = bd;
    } else {
        int32_t local_bid = bid - high_blocks - mid_blocks;
        int32_t v = seg2 + local_bid * BLOCK_SIZE + threadIdx.x;
        float diff = 0.0f;
        if (v < seg4) {
            int32_t rb = offsets[v];
            int32_t re = offsets[v + 1];
            float sum = 0.0f;
            for (int32_t e = rb; e < re; e++) {
                uint32_t mw = edge_mask[e >> 5];
                if (mw & (1u << (e & 31)))
                    sum += weights[e] * x_old[indices[e]];
            }
            float b = betas ? betas[v] : beta;
            float nv = alpha * sum + b;
            x_new[v] = nv;
            diff = fabsf(nv - x_old[v]);
        }
        float bd = BR(ts).Sum(diff);
        if (threadIdx.x == 0) partial_diffs[bid] = bd;
    }
}

template <int BLOCK_SIZE>
__global__ void spmv_uniform_unified(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha_times_C, float beta, const float* __restrict__ betas,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg4,
    int32_t high_blocks, int32_t mid_blocks,
    float* __restrict__ partial_diffs
) {
    int32_t bid = blockIdx.x;
    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage ts;

    if (bid < high_blocks) {
        int32_t v = seg0 + bid;
        int32_t rb = offsets[v], re = offsets[v + 1];
        float wsum = 0.0f;
        for (int32_t e = rb + threadIdx.x; e < re; e += BLOCK_SIZE) {
            uint32_t mw = edge_mask[e >> 5];
            if (mw & (1u << (e & 31))) wsum += weights[e];
        }
        float bs = BR(ts).Sum(wsum);
        if (threadIdx.x == 0) {
            float b = betas ? betas[v] : beta;
            float nv = alpha_times_C * bs + b;
            x_new[v] = nv;
            partial_diffs[bid] = fabsf(nv - x_old[v]);
        }
    } else if (bid < high_blocks + mid_blocks) {
        constexpr int WPB = BLOCK_SIZE / 32;
        int32_t local_bid = bid - high_blocks;
        int32_t wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
        int32_t v = seg1 + local_bid * WPB + wib;
        float diff = 0.0f;
        if (v < seg2) {
            int32_t rb = offsets[v], re = offsets[v + 1];
            float wsum = 0.0f;
            for (int32_t e = rb + lane; e < re; e += 32) {
                uint32_t mw = edge_mask[e >> 5];
                if (mw & (1u << (e & 31))) wsum += weights[e];
            }
            wsum = warp_reduce_sum(wsum);
            if (lane == 0) {
                float b = betas ? betas[v] : beta;
                float nv = alpha_times_C * wsum + b;
                x_new[v] = nv;
                diff = fabsf(nv - x_old[v]);
            }
        }
        float bd = BR(ts).Sum(diff);
        if (threadIdx.x == 0) partial_diffs[bid] = bd;
    } else {
        int32_t local_bid = bid - high_blocks - mid_blocks;
        int32_t v = seg2 + local_bid * BLOCK_SIZE + threadIdx.x;
        float diff = 0.0f;
        if (v < seg4) {
            int32_t rb = offsets[v], re = offsets[v + 1];
            float wsum = 0.0f;
            for (int32_t e = rb; e < re; e++) {
                uint32_t mw = edge_mask[e >> 5];
                if (mw & (1u << (e & 31))) wsum += weights[e];
            }
            float b = betas ? betas[v] : beta;
            float nv = alpha_times_C * wsum + b;
            x_new[v] = nv;
            diff = fabsf(nv - x_old[v]);
        }
        float bd = BR(ts).Sum(diff);
        if (threadIdx.x == 0) partial_diffs[bid] = bd;
    }
}

template <int BLOCK_SIZE>
__global__ void set_beta_fused(
    float* __restrict__ x_new, float beta, const float* __restrict__ betas,
    int32_t nv, float* __restrict__ pd, int32_t doff
) {
    int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float d = 0.0f;
    if (v < nv) {
        float b = betas ? betas[v] : beta;
        x_new[v] = b;
        d = fabsf(b);
    }
    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage ts;
    float bd = BR(ts).Sum(d);
    if (threadIdx.x == 0) pd[doff + blockIdx.x] = bd;
}

__global__ void reduce_partials_kernel(
    const float* __restrict__ p, int32_t n, float* __restrict__ r
) {
    typedef cub::BlockReduce<float, 1024> BR;
    __shared__ typename BR::TempStorage ts;
    float s = 0.0f;
    for (int32_t i = threadIdx.x; i < n; i += 1024) s += p[i];
    float bs = BR(ts).Sum(s);
    if (threadIdx.x == 0) *r = bs;
}

template <int BS>
__global__ void compute_l2_norm_sq(const float* __restrict__ x, int32_t n, float* __restrict__ r) {
    typedef cub::BlockReduce<float, BS> BR;
    __shared__ typename BR::TempStorage ts;
    float s = 0.0f;
    for (int32_t i = blockIdx.x * BS + threadIdx.x; i < n; i += gridDim.x * BS) {
        float v = x[i]; s += v * v;
    }
    float bs = BR(ts).Sum(s);
    if (threadIdx.x == 0) atomicAdd(r, bs);
}

__global__ void scale_by_norm(float* __restrict__ x, int32_t n, const float* __restrict__ norm_sq) {
    float inv_norm = rsqrtf(*norm_sq);
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= inv_norm;
}





int do_spmv_unified(
    const int32_t* offsets, const int32_t* indices, const float* weights,
    const uint32_t* edge_mask, const float* x_old, float* x_new,
    float alpha, float beta, const float* betas,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg4,
    float* pd, cudaStream_t s
) {
    constexpr int BS = 256;
    int hb = seg1 - seg0;
    int mb = (seg2 - seg1 > 0) ? ((seg2 - seg1) + (BS/32) - 1) / (BS/32) : 0;
    int lb = (seg4 - seg2 > 0) ? ((seg4 - seg2) + BS - 1) / BS : 0;
    int total = hb + mb + lb;
    if (total == 0) return 0;
    spmv_unified_fused<BS><<<total, BS, 0, s>>>(
        offsets, indices, weights, edge_mask, x_old, x_new,
        alpha, beta, betas, seg0, seg1, seg2, seg4, hb, mb, pd);
    return total;
}

int do_spmv_uniform_unified(
    const int32_t* offsets, const float* weights,
    const uint32_t* edge_mask, const float* x_old, float* x_new,
    float alpha_C, float beta, const float* betas,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg4,
    float* pd, cudaStream_t s
) {
    constexpr int BS = 256;
    int hb = seg1 - seg0;
    int mb = (seg2 - seg1 > 0) ? ((seg2 - seg1) + (BS/32) - 1) / (BS/32) : 0;
    int lb = (seg4 - seg2 > 0) ? ((seg4 - seg2) + BS - 1) / BS : 0;
    int total = hb + mb + lb;
    if (total == 0) return 0;
    spmv_uniform_unified<BS><<<total, BS, 0, s>>>(
        offsets, weights, edge_mask, x_old, x_new,
        alpha_C, beta, betas, seg0, seg1, seg2, seg4, hb, mb, pd);
    return total;
}

int do_set_beta_fused(
    float* xn, float beta, const float* betas, int32_t nv,
    float* pd, int32_t doff, cudaStream_t s
) {
    constexpr int BS = 256;
    int blocks = (nv + BS - 1) / BS;
    set_beta_fused<BS><<<blocks, BS, 0, s>>>(xn, beta, betas, nv, pd, doff);
    return blocks;
}

void do_reduce_partials(const float* p, int32_t n, float* r, cudaStream_t s) {
    reduce_partials_kernel<<<1, 1024, 0, s>>>(p, n, r);
}

void do_l2_norm_and_scale(float* x, int32_t n, float* norm_buf, cudaStream_t s) {
    constexpr int B = 256;
    int bl = (n + B - 1) / B;
    if (bl > 1024) bl = 1024;
    compute_l2_norm_sq<B><<<bl, B, 0, s>>>(x, n, norm_buf);
    int bl2 = (n + B - 1) / B;
    scale_by_norm<<<bl2, B, 0, s>>>(x, n, norm_buf);
}

}  

katz_centrality_result_t katz_centrality_seg_mask(const graph32_t& graph,
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

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    const float* d_weights = edge_weights;
    const uint32_t* d_edge_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2];
    int32_t seg4 = seg[4];

    constexpr int BS = 256;
    int hb = seg1 - seg0;
    int mb = (seg2 - seg1 > 0) ? ((seg2 - seg1) + (BS/32) - 1) / (BS/32) : 0;
    int lb = (seg4 - seg2 > 0) ? ((seg4 - seg2) + BS - 1) / BS : 0;
    int spmv_blocks = hb + mb + lb;
    int beta_blocks = (num_vertices + BS - 1) / BS;
    int max_blocks = spmv_blocks > beta_blocks ? spmv_blocks : beta_blocks;
    if (max_blocks < 1) max_blocks = 1;

    cache.ensure(num_vertices, max_blocks);

    float* x_buf0 = cache.buf0;
    float* x_buf1 = cache.buf1;
    float* d_pd = cache.partial_diffs;
    float* d_dr = cache.diff_result;

    cudaStream_t stream = 0;

    if (has_initial_guess) {
        cudaMemcpyAsync(x_buf0, centralities,
                        num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemsetAsync(x_buf0, 0, num_vertices * sizeof(float), stream);
    }

    float* x_old = x_buf0;
    float* x_new = x_buf1;
    bool result_in_buf0 = true;
    bool converged = false;
    std::size_t iterations = 0;
    bool x_old_is_uniform = false;
    float x_uniform_val = 0.0f;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        int total_blocks = 0;

        if (iter == 0 && !has_initial_guess) {
            total_blocks = do_set_beta_fused(
                x_new, beta, betas, num_vertices, d_pd, 0, stream);
            x_old_is_uniform = (betas == nullptr);
            x_uniform_val = beta;
        } else if (x_old_is_uniform) {
            float alpha_C = alpha * x_uniform_val;
            total_blocks = do_spmv_uniform_unified(
                d_offsets, d_weights, d_edge_mask, x_old, x_new,
                alpha_C, beta, betas,
                seg0, seg1, seg2, seg4, d_pd, stream);
            x_old_is_uniform = false;
        } else {
            total_blocks = do_spmv_unified(
                d_offsets, d_indices, d_weights, d_edge_mask, x_old, x_new,
                alpha, beta, betas,
                seg0, seg1, seg2, seg4, d_pd, stream);
        }

        do_reduce_partials(d_pd, total_blocks, d_dr, stream);

        float h_diff;
        cudaMemcpy(&h_diff, d_dr, sizeof(float), cudaMemcpyDeviceToHost);

        std::swap(x_old, x_new);
        result_in_buf0 = !result_in_buf0;
        iterations = iter + 1;

        if (h_diff < epsilon) {
            converged = true;
            break;
        }
    }

    if (normalize) {
        float* d_norm = cache.norm_buf;
        cudaMemsetAsync(d_norm, 0, sizeof(float), stream);
        do_l2_norm_and_scale(x_old, num_vertices, d_norm, stream);
    }

    
    float* result_ptr = result_in_buf0 ? x_buf0 : x_buf1;
    cudaMemcpyAsync(centralities, result_ptr,
                    num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return katz_centrality_result_t{iterations, converged};
}

}  
