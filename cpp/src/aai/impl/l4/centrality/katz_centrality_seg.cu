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
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <utility>

namespace aai {

namespace {





struct Cache : Cacheable {
    float* buf = nullptr;
    int16_t* h0 = nullptr;
    int16_t* h1 = nullptr;
    float* d_diff = nullptr;
    float* d_norm = nullptr;

    int64_t buf_capacity = 0;
    int64_t h0_capacity = 0;
    int64_t h1_capacity = 0;

    int max_persist_bytes = 0;

    Cache() {
        cudaDeviceGetAttribute(&max_persist_bytes, cudaDevAttrMaxPersistingL2CacheSize, 0);
        if (max_persist_bytes > 0) {
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, max_persist_bytes);
        }
        cudaMalloc(&d_diff, sizeof(float));
        cudaMalloc(&d_norm, sizeof(float));
    }

    void ensure(int64_t n) {
        if (buf_capacity < n) {
            if (buf) cudaFree(buf);
            cudaMalloc(&buf, n * sizeof(float));
            buf_capacity = n;
        }
        if (h0_capacity < n) {
            if (h0) cudaFree(h0);
            cudaMalloc(&h0, n * sizeof(int16_t));
            h0_capacity = n;
        }
        if (h1_capacity < n) {
            if (h1) cudaFree(h1);
            cudaMalloc(&h1, n * sizeof(int16_t));
            h1_capacity = n;
        }
    }

    ~Cache() override {
        if (buf) cudaFree(buf);
        if (h0) cudaFree(h0);
        if (h1) cudaFree(h1);
        if (d_diff) cudaFree(d_diff);
        if (d_norm) cudaFree(d_norm);
        if (max_persist_bytes > 0) {
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
        }
    }
};





__device__ __forceinline__ float warp_reduce_sum(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v)
{
    __shared__ float smem[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (lane < (blockDim.x >> 5)) ? smem[lane] : 0.0f;
        v = warp_reduce_sum(v);
    }
    return v;
}





__global__ void init_scalar_beta2(const int32_t* __restrict__ offsets,
                                 float* __restrict__ x_f,
                                 int16_t* __restrict__ x_h,
                                 float alpha, float beta, int32_t n)
{
    int32_t v = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        int32_t deg = __ldg(&offsets[v + 1]) - __ldg(&offsets[v]);
        float val = beta + (alpha * beta) * (float)deg;
        x_f[v] = val;
        reinterpret_cast<__half*>(x_h)[v] = __float2half_rn(val);
    }
}

__global__ void init_from_float(const float* __restrict__ src,
                               float* __restrict__ dst_f,
                               int16_t* __restrict__ dst_h,
                               int32_t n)
{
    int32_t v = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        float val = __ldg(&src[v]);
        dst_f[v] = val;
        reinterpret_cast<__half*>(dst_h)[v] = __float2half_rn(val);
    }
}

__global__ void fill_zero(float* __restrict__ x_f, int16_t* __restrict__ x_h, int32_t n)
{
    int32_t v = (int32_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        x_f[v] = 0.0f;
        reinterpret_cast<__half*>(x_h)[v] = __float2half_rn(0.0f);
    }
}





template <bool USE_BETAS, bool COMPUTE_NORM>
__global__ __launch_bounds__(256)
void katz_iter_fp16_fused(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x_old_f,
    const __half* __restrict__ x_old_h,
    float* __restrict__ x_new_f,
    __half* __restrict__ x_new_h,
    float alpha,
    float beta,
    const float* __restrict__ betas,
    float* __restrict__ diff_out,
    float* __restrict__ norm_out,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3, int32_t seg4,
    int32_t mid_blocks, int32_t low_blocks, int32_t zero_blocks)
{
    constexpr int BLOCK = 256;
    constexpr int WARPS_PER_BLOCK = 8;

    int32_t bid = (int32_t)blockIdx.x;

    int32_t n_high = seg1 - seg0;
    int32_t base_mid = n_high;
    int32_t base_low = base_mid + mid_blocks;
    int32_t base_zero = base_low + low_blocks;

    if (bid < n_high) {
        
        int32_t v = seg0 + bid;
        int32_t es = __ldg(&offsets[v]);
        int32_t ee = __ldg(&offsets[v + 1]);
        float sum = 0.0f;
        for (int32_t j = es + (int32_t)threadIdx.x; j < ee; j += BLOCK) {
            int32_t idx = __ldg(&indices[j]);
            sum += __half2float(__ldg(&x_old_h[idx]));
        }
        sum = block_reduce_sum(sum);
        if (threadIdx.x == 0) {
            float bv = USE_BETAS ? __ldg(&betas[v]) : beta;
            float val = alpha * sum + bv;
            x_new_f[v] = val;
            x_new_h[v] = __float2half_rn(val);
            float d = fabsf(val - __ldg(&x_old_f[v]));
            atomicAdd(diff_out, d);
            if constexpr (COMPUTE_NORM) atomicAdd(norm_out, val * val);
        }
        return;
    }

    if (bid < base_low) {
        
        int32_t local_bid = bid - base_mid;
        int32_t warp = threadIdx.x >> 5;
        int32_t lane = threadIdx.x & 31;
        int32_t global_warp0 = local_bid * WARPS_PER_BLOCK + warp;
        int32_t warp_stride = mid_blocks * WARPS_PER_BLOCK;

        float warp_diff = 0.0f;
        float warp_norm = 0.0f;

        for (int32_t v = seg1 + global_warp0; v < seg2; v += warp_stride) {
            int32_t es = __ldg(&offsets[v]);
            int32_t ee = __ldg(&offsets[v + 1]);
            float sum = 0.0f;
            for (int32_t j = es + lane; j < ee; j += 32) {
                int32_t idx = __ldg(&indices[j]);
                sum += __half2float(__ldg(&x_old_h[idx]));
            }
            sum = warp_reduce_sum(sum);
            if (lane == 0) {
                float bv = USE_BETAS ? __ldg(&betas[v]) : beta;
                float val = alpha * sum + bv;
                x_new_f[v] = val;
                x_new_h[v] = __float2half_rn(val);
                warp_diff += fabsf(val - __ldg(&x_old_f[v]));
                if constexpr (COMPUTE_NORM) warp_norm += val * val;
            }
        }

        
        __shared__ float sh_diff[WARPS_PER_BLOCK];
        __shared__ float sh_norm[WARPS_PER_BLOCK];
        if (lane == 0) {
            sh_diff[warp] = warp_diff;
            if constexpr (COMPUTE_NORM) sh_norm[warp] = warp_norm;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float bd = 0.0f;
            float bn = 0.0f;
            #pragma unroll
            for (int i = 0; i < WARPS_PER_BLOCK; i++) {
                bd += sh_diff[i];
                if constexpr (COMPUTE_NORM) bn += sh_norm[i];
            }
            if (bd != 0.0f) atomicAdd(diff_out, bd);
            if constexpr (COMPUTE_NORM) if (bn != 0.0f) atomicAdd(norm_out, bn);
        }
        return;
    }

    if (bid < base_zero) {
        
        int32_t local_bid = bid - base_low;
        int32_t stride = low_blocks * BLOCK;
        float local_diff = 0.0f;
        float local_norm = 0.0f;

        for (int32_t v = seg2 + local_bid * BLOCK + (int32_t)threadIdx.x; v < seg3; v += stride) {
            int32_t es = __ldg(&offsets[v]);
            int32_t ee = __ldg(&offsets[v + 1]);
            float sum = 0.0f;
            
            #pragma unroll 4
            for (int32_t j = es; j < ee; j++) {
                int32_t idx = __ldg(&indices[j]);
                sum += __half2float(__ldg(&x_old_h[idx]));
            }
            float bv = USE_BETAS ? __ldg(&betas[v]) : beta;
            float val = alpha * sum + bv;
            x_new_f[v] = val;
            x_new_h[v] = __float2half_rn(val);
            local_diff += fabsf(val - __ldg(&x_old_f[v]));
            if constexpr (COMPUTE_NORM) local_norm += val * val;
        }

        float bd = block_reduce_sum(local_diff);
        if (threadIdx.x == 0 && bd != 0.0f) atomicAdd(diff_out, bd);
        if constexpr (COMPUTE_NORM) {
            float bn = block_reduce_sum(local_norm);
            if (threadIdx.x == 0 && bn != 0.0f) atomicAdd(norm_out, bn);
        }
        return;
    }

    
    int32_t local_bid = bid - base_zero;
    int32_t stride = zero_blocks * BLOCK;
    float local_diff = 0.0f;
    float local_norm = 0.0f;

    for (int32_t v = seg3 + local_bid * BLOCK + (int32_t)threadIdx.x; v < seg4; v += stride) {
        float bv = USE_BETAS ? __ldg(&betas[v]) : beta;
        x_new_f[v] = bv;
        x_new_h[v] = __float2half_rn(bv);
        local_diff += fabsf(bv - __ldg(&x_old_f[v]));
        if constexpr (COMPUTE_NORM) local_norm += bv * bv;
    }

    float bd = block_reduce_sum(local_diff);
    if (threadIdx.x == 0 && bd != 0.0f) atomicAdd(diff_out, bd);
    if constexpr (COMPUTE_NORM) {
        float bn = block_reduce_sum(local_norm);
        if (threadIdx.x == 0 && bn != 0.0f) atomicAdd(norm_out, bn);
    }
}

template <bool USE_BETAS, bool COMPUTE_NORM>
__global__ __launch_bounds__(256)
void katz_iter_fp32_fused(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha,
    float beta,
    const float* __restrict__ betas,
    float* __restrict__ diff_out,
    float* __restrict__ norm_out,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3, int32_t seg4,
    int32_t mid_blocks, int32_t low_blocks, int32_t zero_blocks)
{
    constexpr int BLOCK = 256;
    constexpr int WARPS_PER_BLOCK = 8;

    int32_t bid = (int32_t)blockIdx.x;

    int32_t n_high = seg1 - seg0;
    int32_t base_mid = n_high;
    int32_t base_low = base_mid + mid_blocks;
    int32_t base_zero = base_low + low_blocks;

    if (bid < n_high) {
        int32_t v = seg0 + bid;
        int32_t es = __ldg(&offsets[v]);
        int32_t ee = __ldg(&offsets[v + 1]);
        float sum = 0.0f;
        for (int32_t j = es + (int32_t)threadIdx.x; j < ee; j += BLOCK) {
            int32_t idx = __ldg(&indices[j]);
            sum += __ldg(&x_old[idx]);
        }
        sum = block_reduce_sum(sum);
        if (threadIdx.x == 0) {
            float bv = USE_BETAS ? __ldg(&betas[v]) : beta;
            float val = alpha * sum + bv;
            x_new[v] = val;
            atomicAdd(diff_out, fabsf(val - __ldg(&x_old[v])));
            if constexpr (COMPUTE_NORM) atomicAdd(norm_out, val * val);
        }
        return;
    }

    if (bid < base_low) {
        int32_t local_bid = bid - base_mid;
        int32_t warp = threadIdx.x >> 5;
        int32_t lane = threadIdx.x & 31;
        int32_t global_warp0 = local_bid * WARPS_PER_BLOCK + warp;
        int32_t warp_stride = mid_blocks * WARPS_PER_BLOCK;

        float warp_diff = 0.0f;
        float warp_norm = 0.0f;

        for (int32_t v = seg1 + global_warp0; v < seg2; v += warp_stride) {
            int32_t es = __ldg(&offsets[v]);
            int32_t ee = __ldg(&offsets[v + 1]);
            float sum = 0.0f;
            for (int32_t j = es + lane; j < ee; j += 32) {
                int32_t idx = __ldg(&indices[j]);
                sum += __ldg(&x_old[idx]);
            }
            sum = warp_reduce_sum(sum);
            if (lane == 0) {
                float bv = USE_BETAS ? __ldg(&betas[v]) : beta;
                float val = alpha * sum + bv;
                x_new[v] = val;
                warp_diff += fabsf(val - __ldg(&x_old[v]));
                if constexpr (COMPUTE_NORM) warp_norm += val * val;
            }
        }

        __shared__ float sh_diff[WARPS_PER_BLOCK];
        __shared__ float sh_norm[WARPS_PER_BLOCK];
        if (lane == 0) {
            sh_diff[warp] = warp_diff;
            if constexpr (COMPUTE_NORM) sh_norm[warp] = warp_norm;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float bd = 0.0f;
            float bn = 0.0f;
            #pragma unroll
            for (int i = 0; i < WARPS_PER_BLOCK; i++) {
                bd += sh_diff[i];
                if constexpr (COMPUTE_NORM) bn += sh_norm[i];
            }
            if (bd != 0.0f) atomicAdd(diff_out, bd);
            if constexpr (COMPUTE_NORM) if (bn != 0.0f) atomicAdd(norm_out, bn);
        }
        return;
    }

    if (bid < base_zero) {
        int32_t local_bid = bid - base_low;
        int32_t stride = low_blocks * BLOCK;
        float local_diff = 0.0f;
        float local_norm = 0.0f;

        for (int32_t v = seg2 + local_bid * BLOCK + (int32_t)threadIdx.x; v < seg3; v += stride) {
            int32_t es = __ldg(&offsets[v]);
            int32_t ee = __ldg(&offsets[v + 1]);
            float sum = 0.0f;
            #pragma unroll 4
            for (int32_t j = es; j < ee; j++) {
                int32_t idx = __ldg(&indices[j]);
                sum += __ldg(&x_old[idx]);
            }
            float bv = USE_BETAS ? __ldg(&betas[v]) : beta;
            float val = alpha * sum + bv;
            x_new[v] = val;
            local_diff += fabsf(val - __ldg(&x_old[v]));
            if constexpr (COMPUTE_NORM) local_norm += val * val;
        }

        float bd = block_reduce_sum(local_diff);
        if (threadIdx.x == 0 && bd != 0.0f) atomicAdd(diff_out, bd);
        if constexpr (COMPUTE_NORM) {
            float bn = block_reduce_sum(local_norm);
            if (threadIdx.x == 0 && bn != 0.0f) atomicAdd(norm_out, bn);
        }
        return;
    }

    int32_t local_bid = bid - base_zero;
    int32_t stride = zero_blocks * BLOCK;
    float local_diff = 0.0f;
    float local_norm = 0.0f;

    for (int32_t v = seg3 + local_bid * BLOCK + (int32_t)threadIdx.x; v < seg4; v += stride) {
        float bv = USE_BETAS ? __ldg(&betas[v]) : beta;
        x_new[v] = bv;
        local_diff += fabsf(bv - __ldg(&x_old[v]));
        if constexpr (COMPUTE_NORM) local_norm += bv * bv;
    }

    float bd = block_reduce_sum(local_diff);
    if (threadIdx.x == 0 && bd != 0.0f) atomicAdd(diff_out, bd);
    if constexpr (COMPUTE_NORM) {
        float bn = block_reduce_sum(local_norm);
        if (threadIdx.x == 0 && bn != 0.0f) atomicAdd(norm_out, bn);
    }
}





__global__ __launch_bounds__(256)
void l2_norm_sq_kernel(const float* __restrict__ x, int32_t n, float* __restrict__ out)
{
    float sum = 0.0f;
    for (int32_t i = (int32_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int32_t)gridDim.x * blockDim.x) {
        float v = __ldg(&x[i]);
        sum += v * v;
    }
    sum = block_reduce_sum(sum);
    if (threadIdx.x == 0) atomicAdd(out, sum);
}

__global__ void inv_sqrt_kernel(float* x) {
    float v = *x;
    *x = (v > 0.0f) ? rsqrtf(v) : 0.0f;
}

__global__ __launch_bounds__(256)
void scale_kernel(float* __restrict__ x, int32_t n, const float* __restrict__ s_ptr)
{
    float s = *s_ptr;
    
    int32_t idx4 = ((int32_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int32_t stride4 = (int32_t)gridDim.x * blockDim.x * 4;
    float4* x4 = reinterpret_cast<float4*>(x);
    int32_t n4 = n / 4;
    for (int32_t i = idx4 / 4; i < n4; i += stride4 / 4) {
        float4 v = x4[i];
        v.x *= s; v.y *= s; v.z *= s; v.w *= s;
        x4[i] = v;
    }
    
    for (int32_t i = n4 * 4 + (int32_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int32_t)gridDim.x * blockDim.x) {
        x[i] *= s;
    }
}





void launch_katz_init_scalar_beta2(const int32_t* offsets, float* x_f, int16_t* x_h,
                                  float alpha, float beta, int32_t n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    init_scalar_beta2<<<blocks, threads, 0, stream>>>(offsets, x_f, x_h, alpha, beta, n);
}

void launch_katz_init_from_float(const float* src, float* dst_f, int16_t* dst_h, int32_t n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    init_from_float<<<blocks, threads, 0, stream>>>(src, dst_f, dst_h, n);
}

void launch_katz_init_from_betas(const float* betas_arr, float* x_f, int16_t* x_h, int32_t n, cudaStream_t stream)
{
    launch_katz_init_from_float(betas_arr, x_f, x_h, n, stream);
}

void launch_katz_fill_zero(float* x_f, int16_t* x_h, int32_t n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fill_zero<<<blocks, threads, 0, stream>>>(x_f, x_h, n);
}

void launch_katz_iter_fp16(const int32_t* offsets, const int32_t* indices,
                           const float* x_old_f, const int16_t* x_old_h,
                           float* x_new_f, int16_t* x_new_h,
                           float alpha, float beta, const float* betas_arr,
                           float* diff_out, float* norm_out,
                           int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3, int32_t seg4,
                           int32_t mid_blocks, int32_t low_blocks, int32_t zero_blocks,
                           bool compute_norm, cudaStream_t stream)
{
    int32_t n_high = seg1 - seg0;
    int32_t total_blocks = n_high + mid_blocks + low_blocks + zero_blocks;
    if (total_blocks <= 0) return;

    const __half* xh_old = reinterpret_cast<const __half*>(x_old_h);
    __half* xh_new = reinterpret_cast<__half*>(x_new_h);

    if (betas_arr) {
        if (compute_norm) {
            katz_iter_fp16_fused<true, true><<<total_blocks, 256, 0, stream>>>(
                offsets, indices, x_old_f, xh_old, x_new_f, xh_new,
                alpha, beta, betas_arr, diff_out, norm_out,
                seg0, seg1, seg2, seg3, seg4,
                mid_blocks, low_blocks, zero_blocks);
        } else {
            katz_iter_fp16_fused<true, false><<<total_blocks, 256, 0, stream>>>(
                offsets, indices, x_old_f, xh_old, x_new_f, xh_new,
                alpha, beta, betas_arr, diff_out, norm_out,
                seg0, seg1, seg2, seg3, seg4,
                mid_blocks, low_blocks, zero_blocks);
        }
    } else {
        if (compute_norm) {
            katz_iter_fp16_fused<false, true><<<total_blocks, 256, 0, stream>>>(
                offsets, indices, x_old_f, xh_old, x_new_f, xh_new,
                alpha, beta, betas_arr, diff_out, norm_out,
                seg0, seg1, seg2, seg3, seg4,
                mid_blocks, low_blocks, zero_blocks);
        } else {
            katz_iter_fp16_fused<false, false><<<total_blocks, 256, 0, stream>>>(
                offsets, indices, x_old_f, xh_old, x_new_f, xh_new,
                alpha, beta, betas_arr, diff_out, norm_out,
                seg0, seg1, seg2, seg3, seg4,
                mid_blocks, low_blocks, zero_blocks);
        }
    }
}

void launch_katz_iter_fp32(const int32_t* offsets, const int32_t* indices,
                           const float* x_old_f, const float*,
                           float* x_new_f, float*,
                           float alpha, float beta, const float* betas_arr,
                           float* diff_out, float* norm_out,
                           int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3, int32_t seg4,
                           int32_t mid_blocks, int32_t low_blocks, int32_t zero_blocks,
                           bool compute_norm, cudaStream_t stream)
{
    int32_t n_high = seg1 - seg0;
    int32_t total_blocks = n_high + mid_blocks + low_blocks + zero_blocks;
    if (total_blocks <= 0) return;

    if (betas_arr) {
        if (compute_norm) {
            katz_iter_fp32_fused<true, true><<<total_blocks, 256, 0, stream>>>(
                offsets, indices, x_old_f, x_new_f,
                alpha, beta, betas_arr, diff_out, norm_out,
                seg0, seg1, seg2, seg3, seg4,
                mid_blocks, low_blocks, zero_blocks);
        } else {
            katz_iter_fp32_fused<true, false><<<total_blocks, 256, 0, stream>>>(
                offsets, indices, x_old_f, x_new_f,
                alpha, beta, betas_arr, diff_out, norm_out,
                seg0, seg1, seg2, seg3, seg4,
                mid_blocks, low_blocks, zero_blocks);
        }
    } else {
        if (compute_norm) {
            katz_iter_fp32_fused<false, true><<<total_blocks, 256, 0, stream>>>(
                offsets, indices, x_old_f, x_new_f,
                alpha, beta, betas_arr, diff_out, norm_out,
                seg0, seg1, seg2, seg3, seg4,
                mid_blocks, low_blocks, zero_blocks);
        } else {
            katz_iter_fp32_fused<false, false><<<total_blocks, 256, 0, stream>>>(
                offsets, indices, x_old_f, x_new_f,
                alpha, beta, betas_arr, diff_out, norm_out,
                seg0, seg1, seg2, seg3, seg4,
                mid_blocks, low_blocks, zero_blocks);
        }
    }
}

void launch_l2_norm_sq(const float* x, int32_t n, float* out, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;
    l2_norm_sq_kernel<<<blocks, threads, 0, stream>>>(x, n, out);
}

void launch_inv_sqrt(float* val, cudaStream_t stream) { inv_sqrt_kernel<<<1,1,0,stream>>>(val); }

void launch_scale(float* x, int32_t n, const float* scale, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;
    scale_kernel<<<blocks, threads, 0, stream>>>(x, n, scale);
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

    cudaStream_t stream = 0;

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    
    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg3 = seg[3], seg4 = seg[4];

    bool use_betas = (betas != nullptr);
    const float* d_betas = betas;

    
    cache.ensure(num_vertices);

    
    float* x0 = centralities;
    float* x1 = cache.buf;
    int16_t* h0 = cache.h0;
    int16_t* h1 = cache.h1;
    float* d_diff = cache.d_diff;
    float* d_norm = cache.d_norm;

    float* x_old_f = x0;
    float* x_new_f = x1;
    int16_t* x_old_h = h0;
    int16_t* x_new_h = h1;

    std::size_t iterations = 0;
    bool converged = false;

    
    if (max_iterations == 0) {
        if (has_initial_guess) {
            launch_katz_init_from_float(centralities, x_old_f, x_old_h, num_vertices, stream);
        } else {
            launch_katz_fill_zero(x_old_f, x_old_h, num_vertices, stream);
        }
    } else if (has_initial_guess) {
        launch_katz_init_from_float(centralities, x_old_f, x_old_h, num_vertices, stream);
    } else if (!use_betas) {
        if (max_iterations >= 2) {
            launch_katz_init_scalar_beta2(d_offsets, x_old_f, x_old_h, alpha, beta, num_vertices, stream);
            iterations = 2;
            float diff2 = fabsf(alpha * beta) * (float)num_edges;
            if (diff2 < epsilon) converged = true;
        } else {
            float diff1 = (float)num_vertices * fabsf(beta);
            if (diff1 < epsilon) converged = true;
            launch_katz_init_scalar_beta2(d_offsets, x_old_f, x_old_h, 0.0f, beta, num_vertices, stream);
            iterations = 1;
        }
    } else {
        launch_katz_init_from_betas(d_betas, x_old_f, x_old_h, num_vertices, stream);
        iterations = 1;
    }

    
    int check_every = 1;
    if (num_edges < 200000) check_every = 20;
    if (max_iterations < 64) check_every = 1;

    
    if (!converged && iterations < max_iterations) {
        
        std::size_t x_bytes = (std::size_t)num_vertices * sizeof(int16_t);
        bool use_l2_persist = (cache.max_persist_bytes > 0 && x_bytes > 0);
        cudaStreamAttrValue attr = {};
        if (use_l2_persist) {
            std::size_t persist_size = std::min(x_bytes, (std::size_t)cache.max_persist_bytes);
            attr.accessPolicyWindow.base_ptr = (void*)x_old_h;
            attr.accessPolicyWindow.num_bytes = persist_size;
            attr.accessPolicyWindow.hitRatio = 1.0f;
            attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
        }

        while (iterations < max_iterations) {
            if (use_l2_persist) {
                attr.accessPolicyWindow.base_ptr = (void*)x_old_h;
                cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
            }

            cudaMemsetAsync(d_diff, 0, sizeof(float), stream);
            if (normalize) cudaMemsetAsync(d_norm, 0, sizeof(float), stream);

            int32_t mid_blocks = (seg2 > seg1) ? std::min<int32_t>(4096, (seg2 - seg1 + 7) / 8) : 0;
            int32_t low_blocks = (seg3 > seg2) ? std::min<int32_t>(4096, (seg3 - seg2 + 255) / 256) : 0;
            int32_t zero_blocks = (seg4 > seg3) ? std::min<int32_t>(4096, (seg4 - seg3 + 255) / 256) : 0;

            bool use_fp16 = (num_edges >= 1000000);

            if (use_fp16) {
                launch_katz_iter_fp16(d_offsets, d_indices, x_old_f, x_old_h,
                                      x_new_f, x_new_h,
                                      alpha, beta, d_betas,
                                      d_diff, d_norm,
                                      seg0, seg1, seg2, seg3, seg4,
                                      mid_blocks, low_blocks, zero_blocks,
                                      normalize, stream);
            } else {
                launch_katz_iter_fp32(d_offsets, d_indices, x_old_f, nullptr,
                                      x_new_f, nullptr,
                                      alpha, beta, d_betas,
                                      d_diff, d_norm,
                                      seg0, seg1, seg2, seg3, seg4,
                                      mid_blocks, low_blocks, zero_blocks,
                                      normalize, stream);
            }

            iterations++;

            bool do_check = ((iterations % (std::size_t)check_every) == 0) || (iterations == max_iterations);
            if (do_check) {
                float h_diff;
                cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
                if (h_diff < epsilon) {
                    converged = true;
                    std::swap(x_old_f, x_new_f);
                    std::swap(x_old_h, x_new_h);
                    break;
                }
            }

            std::swap(x_old_f, x_new_f);
            std::swap(x_old_h, x_new_h);
        }

        if (use_l2_persist) {
            attr.accessPolicyWindow.num_bytes = 0;
            cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
        }
    }

    
    if (normalize) {
        if (iterations <= 2 && !has_initial_guess) {
            cudaMemsetAsync(d_norm, 0, sizeof(float), stream);
            launch_l2_norm_sq(x_old_f, num_vertices, d_norm, stream);
        }
        launch_inv_sqrt(d_norm, stream);
        launch_scale(x_old_f, num_vertices, d_norm, stream);
    }

    
    if (x_old_f != centralities) {
        cudaMemcpyAsync(centralities, x_old_f, (std::size_t)num_vertices * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    return katz_centrality_result_t{iterations, converged};
}

}  
