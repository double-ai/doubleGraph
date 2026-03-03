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
#include <cuda_fp16.h>
#include <cub/block/block_reduce.cuh>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <climits>

namespace aai {

namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;


static constexpr int32_t FP16_THRESHOLD = 500000;

struct Cache : Cacheable {
    float* h_pinned = nullptr;
    float* d_buf0 = nullptr;
    float* d_buf1 = nullptr;
    float* d_diff = nullptr;
    __half* d_fp16 = nullptr;

    int64_t buf_capacity = 0;
    int64_t fp16_capacity = 0;

    Cache() {
        cudaMallocHost(&h_pinned, 8 * sizeof(float));
    }

    ~Cache() override {
        if (h_pinned) cudaFreeHost(h_pinned);
        if (d_buf0) cudaFree(d_buf0);
        if (d_buf1) cudaFree(d_buf1);
        if (d_diff) cudaFree(d_diff);
        if (d_fp16) cudaFree(d_fp16);
    }

    void ensure_bufs(int64_t n) {
        if (buf_capacity < n) {
            if (d_buf0) cudaFree(d_buf0);
            if (d_buf1) cudaFree(d_buf1);
            if (d_diff) cudaFree(d_diff);
            cudaMalloc(&d_buf0, n * sizeof(float));
            cudaMalloc(&d_buf1, n * sizeof(float));
            cudaMalloc(&d_diff, sizeof(float));
            buf_capacity = n;
        }
    }

    void ensure_fp16(int64_t n) {
        if (fp16_capacity < n) {
            if (d_fp16) cudaFree(d_fp16);
            cudaMalloc(&d_fp16, n * sizeof(__half));
            fp16_capacity = n;
        }
    }
};


__global__ __launch_bounds__(BLOCK_SIZE)
void katz_first_iter_scalar(float* __restrict__ x_new, float beta, int32_t n) {
    int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) x_new[i] = beta;
}

__global__ __launch_bounds__(BLOCK_SIZE)
void katz_first_iter_betas(float* __restrict__ x_new, const float* __restrict__ betas,
                           int32_t n, float* __restrict__ diff_sum) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float d = 0.0f;
    if (v < n) { float bv = __ldg(&betas[v]); x_new[v] = bv; d = fabsf(bv); }
    d = BlockReduce(temp).Sum(d);
    if (threadIdx.x == 0) atomicAdd(diff_sum, d);
}


__global__ __launch_bounds__(BLOCK_SIZE)
void katz_second_iter_scalar(
    const int32_t* __restrict__ offsets,
    float* __restrict__ x_new,
    const float* __restrict__ x_old,
    float alpha, float beta,
    int32_t num_vertices,
    float* __restrict__ diff_sum
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float d = 0.0f;
    if (v < num_vertices) {
        int32_t deg = __ldg(&offsets[v + 1]) - __ldg(&offsets[v]);
        float val = beta * (alpha * (float)deg + 1.0f);
        x_new[v] = val;
        d = fabsf(val - __ldg(&x_old[v]));
    }
    d = BlockReduce(temp).Sum(d);
    if (threadIdx.x == 0) atomicAdd(diff_sum, d);
}


__global__ void f32_to_f16(const float* __restrict__ src, __half* __restrict__ dst, int32_t n) {
    int32_t i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}


__global__ __launch_bounds__(BLOCK_SIZE)
void katz_spmv_fp16(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const __half* __restrict__ x_old_h,   
    const float* __restrict__ x_old_f,    
    float* __restrict__ x_new,
    const float alpha,
    const float beta,
    const float* __restrict__ betas,
    const int32_t n_high_blocks,
    const int32_t n_mid_blocks,
    const int32_t n_low_blocks,
    const int32_t seg0, const int32_t seg1,
    const int32_t seg2, const int32_t seg3, const int32_t seg4,
    float* __restrict__ diff_sum
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_hm = n_high_blocks + n_mid_blocks;
    const int n_hml = n_hm + n_low_blocks;

    if (bid < n_high_blocks) {
        const int32_t v = seg0 + bid;
        const int32_t row_start = __ldg(&offsets[v]);
        const int32_t row_end = __ldg(&offsets[v + 1]);
        float sum = 0.0f;
        for (int32_t j = row_start + tid; j < row_end; j += BLOCK_SIZE)
            sum += __half2float(__ldg(&x_old_h[__ldg(&indices[j])]));
        sum = BlockReduce(temp_storage).Sum(sum);
        if (tid == 0) {
            float bv = betas ? __ldg(&betas[v]) : beta;
            float val = alpha * sum + bv;
            x_new[v] = val;
            atomicAdd(diff_sum, fabsf(val - __ldg(&x_old_f[v])));
        }
    }
    else if (bid < n_hm) {
        const int local_bid = bid - n_high_blocks;
        const int warp = tid >> 5, lane = tid & 31;
        const int32_t v = seg1 + local_bid * WARPS_PER_BLOCK + warp;
        float my_diff = 0.0f;
        if (v < seg2) {
            const int32_t rs = __ldg(&offsets[v]), re = __ldg(&offsets[v + 1]);
            float sum = 0.0f;
            for (int32_t j = rs + lane; j < re; j += 32)
                sum += __half2float(__ldg(&x_old_h[__ldg(&indices[j])]));
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                sum += __shfl_down_sync(0xffffffff, sum, o);
            if (lane == 0) {
                float bv = betas ? __ldg(&betas[v]) : beta;
                float val = alpha * sum + bv;
                x_new[v] = val;
                my_diff = fabsf(val - __ldg(&x_old_f[v]));
            }
        }
        my_diff = BlockReduce(temp_storage).Sum(my_diff);
        if (tid == 0) atomicAdd(diff_sum, my_diff);
    }
    else if (bid < n_hml) {
        const int32_t v = seg2 + (bid - n_hm) * BLOCK_SIZE + tid;
        float my_diff = 0.0f;
        if (v < seg3) {
            const int32_t rs = __ldg(&offsets[v]), re = __ldg(&offsets[v + 1]);
            float sum = 0.0f;
            for (int32_t j = rs; j < re; j++)
                sum += __half2float(__ldg(&x_old_h[__ldg(&indices[j])]));
            float bv = betas ? __ldg(&betas[v]) : beta;
            float val = alpha * sum + bv;
            x_new[v] = val;
            my_diff = fabsf(val - __ldg(&x_old_f[v]));
        }
        my_diff = BlockReduce(temp_storage).Sum(my_diff);
        if (tid == 0) atomicAdd(diff_sum, my_diff);
    }
    else {
        const int32_t v = seg3 + (bid - n_hml) * BLOCK_SIZE + tid;
        float my_diff = 0.0f;
        if (v < seg4) {
            float bv = betas ? __ldg(&betas[v]) : beta;
            x_new[v] = bv;
            my_diff = fabsf(bv - __ldg(&x_old_f[v]));
        }
        my_diff = BlockReduce(temp_storage).Sum(my_diff);
        if (tid == 0) atomicAdd(diff_sum, my_diff);
    }
}


__global__ __launch_bounds__(BLOCK_SIZE)
void katz_spmv_fp32(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float alpha,
    const float beta,
    const float* __restrict__ betas,
    const int32_t n_high_blocks,
    const int32_t n_mid_blocks,
    const int32_t n_low_blocks,
    const int32_t seg0, const int32_t seg1,
    const int32_t seg2, const int32_t seg3, const int32_t seg4,
    float* __restrict__ diff_sum
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int n_hm = n_high_blocks + n_mid_blocks;
    const int n_hml = n_hm + n_low_blocks;

    if (bid < n_high_blocks) {
        const int32_t v = seg0 + bid;
        const int32_t row_start = __ldg(&offsets[v]);
        const int32_t row_end = __ldg(&offsets[v + 1]);
        float sum = 0.0f;
        for (int32_t j = row_start + tid; j < row_end; j += BLOCK_SIZE)
            sum += __ldg(&x_old[__ldg(&indices[j])]);
        sum = BlockReduce(temp_storage).Sum(sum);
        if (tid == 0) {
            float bv = betas ? __ldg(&betas[v]) : beta;
            float val = alpha * sum + bv;
            x_new[v] = val;
            atomicAdd(diff_sum, fabsf(val - __ldg(&x_old[v])));
        }
    }
    else if (bid < n_hm) {
        const int local_bid = bid - n_high_blocks;
        const int warp = tid >> 5, lane = tid & 31;
        const int32_t v = seg1 + local_bid * WARPS_PER_BLOCK + warp;
        float my_diff = 0.0f;
        if (v < seg2) {
            const int32_t rs = __ldg(&offsets[v]), re = __ldg(&offsets[v + 1]);
            float sum = 0.0f;
            for (int32_t j = rs + lane; j < re; j += 32)
                sum += __ldg(&x_old[__ldg(&indices[j])]);
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                sum += __shfl_down_sync(0xffffffff, sum, o);
            if (lane == 0) {
                float bv = betas ? __ldg(&betas[v]) : beta;
                float val = alpha * sum + bv;
                x_new[v] = val;
                my_diff = fabsf(val - __ldg(&x_old[v]));
            }
        }
        my_diff = BlockReduce(temp_storage).Sum(my_diff);
        if (tid == 0) atomicAdd(diff_sum, my_diff);
    }
    else if (bid < n_hml) {
        const int32_t v = seg2 + (bid - n_hm) * BLOCK_SIZE + tid;
        float my_diff = 0.0f;
        if (v < seg3) {
            const int32_t rs = __ldg(&offsets[v]), re = __ldg(&offsets[v + 1]);
            float sum = 0.0f;
            for (int32_t j = rs; j < re; j++)
                sum += __ldg(&x_old[__ldg(&indices[j])]);
            float bv = betas ? __ldg(&betas[v]) : beta;
            float val = alpha * sum + bv;
            x_new[v] = val;
            my_diff = fabsf(val - __ldg(&x_old[v]));
        }
        my_diff = BlockReduce(temp_storage).Sum(my_diff);
        if (tid == 0) atomicAdd(diff_sum, my_diff);
    }
    else {
        const int32_t v = seg3 + (bid - n_hml) * BLOCK_SIZE + tid;
        float my_diff = 0.0f;
        if (v < seg4) {
            float bv = betas ? __ldg(&betas[v]) : beta;
            x_new[v] = bv;
            my_diff = fabsf(bv - __ldg(&x_old[v]));
        }
        my_diff = BlockReduce(temp_storage).Sum(my_diff);
        if (tid == 0) atomicAdd(diff_sum, my_diff);
    }
}

__global__ __launch_bounds__(256)
void l2_norm_sq_kernel(const float* __restrict__ x, float* __restrict__ result, int32_t n) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    float s = 0.0f;
    for (int32_t i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        float v = __ldg(&x[i]); s += v * v;
    }
    s = BlockReduce(temp).Sum(s);
    if (threadIdx.x == 0) atomicAdd(result, s);
}

__global__ void scale_kernel(float* __restrict__ x, float scale, int32_t n) {
    int32_t i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) x[i] *= scale;
}



static void launch_katz_first_iter_scalar(float* x_new, float beta, int32_t n, cudaStream_t s) {
    int bl = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (bl > 0) katz_first_iter_scalar<<<bl, BLOCK_SIZE, 0, s>>>(x_new, beta, n);
}

static void launch_katz_first_iter_betas(float* x_new, const float* betas, int32_t n,
                                  float* diff_sum, cudaStream_t s) {
    cudaMemsetAsync(diff_sum, 0, sizeof(float), s);
    int bl = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (bl > 0) katz_first_iter_betas<<<bl, BLOCK_SIZE, 0, s>>>(x_new, betas, n, diff_sum);
}

static void launch_katz_second_iter_scalar(const int32_t* offsets, float* x_new, const float* x_old,
                                    float alpha, float beta, int32_t n,
                                    float* diff_sum, cudaStream_t s) {
    cudaMemsetAsync(diff_sum, 0, sizeof(float), s);
    int bl = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (bl > 0) katz_second_iter_scalar<<<bl, BLOCK_SIZE, 0, s>>>(
        offsets, x_new, x_old, alpha, beta, n, diff_sum);
}

static void launch_f32_to_f16(const float* src, __half* dst, int32_t n, cudaStream_t s) {
    int bl = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (bl > 0) f32_to_f16<<<bl, BLOCK_SIZE, 0, s>>>(src, dst, n);
}

static void launch_katz_spmv_fp16(
    const int32_t* offsets, const int32_t* indices,
    const __half* x_old_h, const float* x_old_f,
    float* x_new, float alpha, float beta, const float* betas,
    int32_t s0, int32_t s1, int32_t s2, int32_t s3, int32_t s4,
    float* diff_sum, cudaStream_t s
) {
    int32_t nh = s1 - s0;
    int32_t nm = (s2 - s1 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int32_t nl = (s3 - s2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int32_t nz = (s4 - s3 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int32_t tot = nh + nm + nl + nz;
    if (tot == 0) return;
    cudaMemsetAsync(diff_sum, 0, sizeof(float), s);
    katz_spmv_fp16<<<tot, BLOCK_SIZE, 0, s>>>(
        offsets, indices, x_old_h, x_old_f, x_new, alpha, beta, betas,
        nh, nm, nl, s0, s1, s2, s3, s4, diff_sum);
}

static void launch_katz_spmv_fp32(
    const int32_t* offsets, const int32_t* indices,
    const float* x_old, float* x_new,
    float alpha, float beta, const float* betas,
    int32_t s0, int32_t s1, int32_t s2, int32_t s3, int32_t s4,
    float* diff_sum, cudaStream_t s
) {
    int32_t nh = s1 - s0;
    int32_t nm = (s2 - s1 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int32_t nl = (s3 - s2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int32_t nz = (s4 - s3 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int32_t tot = nh + nm + nl + nz;
    if (tot == 0) return;
    cudaMemsetAsync(diff_sum, 0, sizeof(float), s);
    katz_spmv_fp32<<<tot, BLOCK_SIZE, 0, s>>>(
        offsets, indices, x_old, x_new, alpha, beta, betas,
        nh, nm, nl, s0, s1, s2, s3, s4, diff_sum);
}

static void launch_l2_norm(const float* x, float* result, int32_t n, cudaStream_t s) {
    cudaMemsetAsync(result, 0, sizeof(float), s);
    int bl = (n + 255) / 256; if (bl > 1024) bl = 1024;
    l2_norm_sq_kernel<<<bl, 256, 0, s>>>(x, result, n);
}

static void launch_scale(float* x, float scale, int32_t n, cudaStream_t s) {
    int bl = (n + 255) / 256;
    scale_kernel<<<bl, 256, 0, s>>>(x, scale, n);
}

}  

katz_centrality_result_t katz_centrality_seg(const graph32_t& graph,
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

    const cudaStream_t stream = 0;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const int32_t num_vertices = graph.number_of_vertices;

    const auto& seg = graph.segment_offsets.value();
    const int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2];
    const int32_t seg3 = seg[3], seg4 = seg[4];

    const bool use_betas = (betas != nullptr);

    
    const int32_t max_iter = (max_iterations > (std::size_t)INT32_MAX) ? INT32_MAX : (int32_t)max_iterations;

    
    cache.ensure_bufs(num_vertices > 0 ? num_vertices : 1);

    float* d_buf0 = cache.d_buf0;
    float* d_buf1 = cache.d_buf1;
    float* d_diff = cache.d_diff;
    float* h_pinned = cache.h_pinned;

    
    const bool use_fp16 = (num_vertices >= FP16_THRESHOLD);
    __half* d_x_old_h = nullptr;
    if (use_fp16) {
        cache.ensure_fp16(num_vertices);
        d_x_old_h = cache.d_fp16;
    }

    float* x_old = d_buf0;
    float* x_new = d_buf1;
    bool converged = false;
    int64_t iterations = 0;

    auto check_conv = [&]() -> bool {
        cudaMemcpyAsync(h_pinned, d_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return *h_pinned < epsilon;
    };
    auto swap = [&]() { float* t = x_old; x_old = x_new; x_new = t; };

    
    auto do_spmv = [&]() {
        if (use_fp16) {
            launch_f32_to_f16(x_old, d_x_old_h, num_vertices, stream);
            launch_katz_spmv_fp16(d_offsets, d_indices, d_x_old_h, x_old,
                x_new, alpha, beta, betas,
                seg0, seg1, seg2, seg3, seg4, d_diff, stream);
        } else {
            launch_katz_spmv_fp32(d_offsets, d_indices, x_old, x_new,
                alpha, beta, betas,
                seg0, seg1, seg2, seg3, seg4, d_diff, stream);
        }
    };

    if (num_vertices == 0 || max_iter == 0) {
        if (!has_initial_guess && num_vertices > 0)
            cudaMemsetAsync(d_buf0, 0, (size_t)num_vertices * sizeof(float), stream);
    } else if (has_initial_guess) {
        cudaMemcpyAsync(d_buf0, centralities,
            (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        for (int64_t iter = 0; iter < max_iter; ++iter) {
            do_spmv();
            iterations = iter + 1;
            swap();
            if (check_conv()) { converged = true; break; }
        }
    } else if (!use_betas) {
        
        launch_katz_first_iter_scalar(x_new, beta, num_vertices, stream);
        iterations = 1; swap();

        if (max_iter >= 2) {
            launch_katz_second_iter_scalar(d_offsets, x_new, x_old,
                alpha, beta, num_vertices, d_diff, stream);
            iterations = 2; swap();
            if (check_conv()) {
                converged = true;
            } else {
                for (int64_t iter = 2; iter < max_iter; ++iter) {
                    do_spmv();
                    iterations = iter + 1;
                    swap();
                    if (check_conv()) { converged = true; break; }
                }
            }
        }
    } else {
        
        launch_katz_first_iter_betas(x_new, betas, num_vertices, d_diff, stream);
        iterations = 1; swap();
        if (check_conv()) {
            converged = true;
        } else {
            for (int64_t iter = 1; iter < max_iter; ++iter) {
                do_spmv();
                iterations = iter + 1;
                swap();
                if (check_conv()) { converged = true; break; }
            }
        }
    }

    
    if (normalize && num_vertices > 0) {
        launch_l2_norm(x_old, d_diff, num_vertices, stream);
        cudaMemcpyAsync(h_pinned, d_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        float norm = sqrtf(*h_pinned);
        if (norm > 0.0f) launch_scale(x_old, 1.0f / norm, num_vertices, stream);
    }

    
    if (num_vertices > 0) {
        cudaMemcpyAsync(centralities, x_old,
            (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    return katz_centrality_result_t{(std::size_t)iterations, converged};
}

}  
