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
#include <cstring>

namespace aai {

namespace {

constexpr int BLOCK = 256;
constexpr int WPB = BLOCK / 32;




__global__ void __launch_bounds__(BLOCK)
katz_high(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float* __restrict__ betas,
    float alpha, float beta, int use_betas,
    int v_start, int v_end,
    float* __restrict__ diff_acc)
{
    int v = v_start + blockIdx.x;
    if (v >= v_end) return;

    int rs = __ldg(&offsets[v]);
    int re = __ldg(&offsets[v + 1]);

    float sum = 0.0f;
    for (int e = rs + (int)threadIdx.x; e < re; e += BLOCK) {
        uint32_t mw = __ldg(&edge_mask[e >> 5]);
        if (mw & (1u << (e & 31))) {
            sum += __ldg(&x_old[__ldg(&indices[e])]);
        }
    }

    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage tmp;
    float total = BR(tmp).Sum(sum);

    if (threadIdx.x == 0) {
        float bv = use_betas ? __ldg(&betas[v]) : beta;
        float val = alpha * total + bv;
        float d = fabsf(val - __ldg(&x_old[v]));
        x_new[v] = val;
        atomicAdd(diff_acc, d);
    }
}




__global__ void __launch_bounds__(BLOCK)
katz_mid(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float* __restrict__ betas,
    float alpha, float beta, int use_betas,
    int v_start, int v_end,
    float* __restrict__ diff_acc)
{
    int gwarp = (blockIdx.x * BLOCK + (int)threadIdx.x) >> 5;
    int lane  = threadIdx.x & 31;
    int wib   = threadIdx.x >> 5;
    int v = v_start + gwarp;

    float wdiff = 0.0f;
    if (v < v_end) {
        int rs = __ldg(&offsets[v]);
        int re = __ldg(&offsets[v + 1]);

        float sum = 0.0f;
        for (int e = rs + lane; e < re; e += 32) {
            uint32_t mw = __ldg(&edge_mask[e >> 5]);
            if (mw & (1u << (e & 31))) {
                sum += __ldg(&x_old[__ldg(&indices[e])]);
            }
        }

        #pragma unroll
        for (int s = 16; s > 0; s >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, s);

        if (lane == 0) {
            float bv = use_betas ? __ldg(&betas[v]) : beta;
            float val = alpha * sum + bv;
            wdiff = fabsf(val - __ldg(&x_old[v]));
            x_new[v] = val;
        }
    }

    __shared__ float sdiff[WPB];
    if (lane == 0) sdiff[wib] = wdiff;
    __syncthreads();

    if (threadIdx.x == 0) {
        float bd = 0.0f;
        #pragma unroll
        for (int w = 0; w < WPB; w++) bd += sdiff[w];
        atomicAdd(diff_acc, bd);
    }
}




__global__ void __launch_bounds__(BLOCK)
katz_low(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float* __restrict__ betas,
    float alpha, float beta, int use_betas,
    int v_start, int v_end,
    float* __restrict__ diff_acc)
{
    int v = v_start + blockIdx.x * BLOCK + (int)threadIdx.x;

    float d = 0.0f;
    if (v < v_end) {
        int rs = __ldg(&offsets[v]);
        int re = __ldg(&offsets[v + 1]);

        float sum = 0.0f;
        for (int e = rs; e < re; e++) {
            uint32_t mw = __ldg(&edge_mask[e >> 5]);
            if (mw & (1u << (e & 31))) {
                sum += __ldg(&x_old[__ldg(&indices[e])]);
            }
        }

        float bv = use_betas ? __ldg(&betas[v]) : beta;
        float val = alpha * sum + bv;
        d = fabsf(val - __ldg(&x_old[v]));
        x_new[v] = val;
    }

    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage tmp;
    float bd = BR(tmp).Sum(d);
    if (threadIdx.x == 0) atomicAdd(diff_acc, bd);
}




__global__ void __launch_bounds__(BLOCK)
katz_zero(
    float* __restrict__ x_new,
    const float* __restrict__ x_old,
    const float* __restrict__ betas,
    float beta, int use_betas,
    int v_start, int v_end,
    float* __restrict__ diff_acc)
{
    int v = v_start + blockIdx.x * BLOCK + (int)threadIdx.x;
    float d = 0.0f;
    if (v < v_end) {
        float bv = use_betas ? __ldg(&betas[v]) : beta;
        d = fabsf(bv - __ldg(&x_old[v]));
        x_new[v] = bv;
    }
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage tmp;
    float bd = BR(tmp).Sum(d);
    if (threadIdx.x == 0) atomicAdd(diff_acc, bd);
}




__global__ void __launch_bounds__(BLOCK)
katz_fill_beta(
    float* __restrict__ x_new,
    const float* __restrict__ betas,
    float beta, int use_betas, int n)
{
    int v = blockIdx.x * BLOCK + (int)threadIdx.x;
    if (v < n) {
        x_new[v] = use_betas ? __ldg(&betas[v]) : beta;
    }
}





__global__ void __launch_bounds__(BLOCK)
katz_iter2_uniform(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ x_new,
    float alpha_beta,  
    float beta,        
    int num_vertices)
{
    int v = blockIdx.x * BLOCK + (int)threadIdx.x;
    if (v >= num_vertices) return;

    int rs = __ldg(&offsets[v]);
    int re = __ldg(&offsets[v + 1]);

    
    int cnt = 0;
    int e = rs;

    
    if ((e & 31) != 0 && e < re) {
        uint32_t mw = __ldg(&edge_mask[e >> 5]);
        
        mw &= ~((1u << (e & 31)) - 1);
        int word_end = ((e >> 5) + 1) << 5; 
        if (word_end > re) {
            
            mw &= (1u << (re & 31)) - 1;
            cnt = __popc(mw);
            x_new[v] = alpha_beta * cnt + beta;
            return;
        }
        cnt = __popc(mw);
        e = word_end;
    }

    
    while (e + 32 <= re) {
        cnt += __popc(__ldg(&edge_mask[e >> 5]));
        e += 32;
    }

    
    if (e < re) {
        uint32_t mw = __ldg(&edge_mask[e >> 5]);
        mw &= (1u << (re - e)) - 1;
        cnt += __popc(mw);
    }

    x_new[v] = alpha_beta * cnt + beta;
}




__global__ void __launch_bounds__(BLOCK)
l2_sq_kernel(const float* __restrict__ x, int n, float* __restrict__ out)
{
    int i = blockIdx.x * BLOCK + (int)threadIdx.x;
    float v = 0.0f;
    if (i < n) { float xi = x[i]; v = xi * xi; }
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage tmp;
    float bs = BR(tmp).Sum(v);
    if (threadIdx.x == 0) atomicAdd(out, bs);
}

__global__ void __launch_bounds__(BLOCK)
norm_kernel(float* __restrict__ x, int n, const float* __restrict__ norm_sq)
{
    int i = blockIdx.x * BLOCK + (int)threadIdx.x;
    if (i < n) {
        float ns = *norm_sq;
        float inv = (ns > 0.0f) ? rsqrtf(ns) : 0.0f;
        x[i] *= inv;
    }
}




__global__ void __launch_bounds__(BLOCK)
compute_diff_kernel(const float* __restrict__ x_new, const float* __restrict__ x_old,
                     int n, float* __restrict__ diff_acc)
{
    int v = blockIdx.x * BLOCK + (int)threadIdx.x;
    float d = 0.0f;
    if (v < n) d = fabsf(x_new[v] - x_old[v]);
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage tmp;
    float bd = BR(tmp).Sum(d);
    if (threadIdx.x == 0 && bd != 0.0f) atomicAdd(diff_acc, bd);
}




static void set_l2_persist(const float* ptr, size_t bytes) {
    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));
    if (bytes > 0 && ptr != nullptr) {
        attr.accessPolicyWindow.base_ptr = (void*)ptr;
        attr.accessPolicyWindow.num_bytes = bytes;
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    }
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
}




struct Cache : Cacheable {
    float* buf0 = nullptr;
    float* buf1 = nullptr;
    float* d_scratch = nullptr;
    int32_t buf0_capacity = 0;
    int32_t buf1_capacity = 0;
    size_t l2_persist_max = 0;

    Cache() {
        cudaMalloc(&d_scratch, sizeof(float));
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        l2_persist_max = prop.persistingL2CacheMaxSize;
        if (l2_persist_max > 0) {
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_persist_max);
        }
    }

    void ensure(int32_t n) {
        if (buf0_capacity < n) {
            if (buf0) cudaFree(buf0);
            cudaMalloc(&buf0, (size_t)n * sizeof(float));
            buf0_capacity = n;
        }
        if (buf1_capacity < n) {
            if (buf1) cudaFree(buf1);
            cudaMalloc(&buf1, (size_t)n * sizeof(float));
            buf1_capacity = n;
        }
    }

    ~Cache() override {
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (d_scratch) cudaFree(d_scratch);
    }
};

}  

katz_centrality_result_t katz_centrality_seg_mask(const graph32_t& graph,
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

    int32_t N = graph.number_of_vertices;
    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    const uint32_t* d_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int s0 = seg[0], s1 = seg[1], s2 = seg[2], s3 = seg[3], s4 = seg[4];

    int ub = (betas != nullptr) ? 1 : 0;

    cache.ensure(N);
    float* cur = cache.buf0;
    float* nxt = cache.buf1;
    float* d_scratch = cache.d_scratch;

    if (has_initial_guess) {
        cudaMemcpyAsync(cur, centralities,
                        (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        cudaMemsetAsync(cur, 0, (size_t)N * sizeof(float));
    }

    std::size_t iterations = 0;
    bool converged = false;
    float* result = cur;

    
    bool x_is_uniform_beta = false;

    for (std::size_t it = 0; it < max_iterations; it++) {
        if (it == 0 && !has_initial_guess) {
            
            int blk = (N + BLOCK - 1) / BLOCK;
            katz_fill_beta<<<blk, BLOCK>>>(nxt, betas, beta, ub, N);
            x_is_uniform_beta = (betas == nullptr);

            cudaMemsetAsync(d_scratch, 0, sizeof(float));
            blk = (N + BLOCK - 1) / BLOCK;
            compute_diff_kernel<<<blk, BLOCK>>>(nxt, cur, N, d_scratch);
        } else if (x_is_uniform_beta) {
            
            
            
            int blk = (N + BLOCK - 1) / BLOCK;
            katz_iter2_uniform<<<blk, BLOCK>>>(d_off, d_mask, nxt,
                alpha * beta, beta, N);
            x_is_uniform_beta = false;

            cudaMemsetAsync(d_scratch, 0, sizeof(float));
            blk = (N + BLOCK - 1) / BLOCK;
            compute_diff_kernel<<<blk, BLOCK>>>(nxt, cur, N, d_scratch);
        } else {
            
            if (cache.l2_persist_max > 0) {
                size_t x_bytes = (size_t)N * sizeof(float);
                size_t persist_bytes = (x_bytes < cache.l2_persist_max) ? x_bytes : cache.l2_persist_max;
                set_l2_persist(cur, persist_bytes);
            }

            cudaMemsetAsync(d_scratch, 0, sizeof(float));

            int nh = s1 - s0;
            if (nh > 0)
                katz_high<<<nh, BLOCK>>>(d_off, d_idx, d_mask,
                    cur, nxt, betas, alpha, beta, ub, s0, s1, d_scratch);

            int nm = s2 - s1;
            if (nm > 0) {
                int blk = (int)(((int64_t)nm * 32 + BLOCK - 1) / BLOCK);
                katz_mid<<<blk, BLOCK>>>(d_off, d_idx, d_mask,
                    cur, nxt, betas, alpha, beta, ub, s1, s2, d_scratch);
            }

            int nl = s3 - s2;
            if (nl > 0) {
                int blk = (nl + BLOCK - 1) / BLOCK;
                katz_low<<<blk, BLOCK>>>(d_off, d_idx, d_mask,
                    cur, nxt, betas, alpha, beta, ub, s2, s3, d_scratch);
            }

            int nz = s4 - s3;
            if (nz > 0) {
                int blk = (nz + BLOCK - 1) / BLOCK;
                katz_zero<<<blk, BLOCK>>>(nxt, cur, betas, beta,
                                           ub, s3, s4, d_scratch);
            }
        }

        result = nxt;
        iterations++;

        float h_diff;
        cudaMemcpy(&h_diff, d_scratch, sizeof(float),
                   cudaMemcpyDeviceToHost);

        if (h_diff < epsilon) {
            converged = true;
            break;
        }

        float* tmp = cur; cur = nxt; nxt = tmp;
    }

    
    if (cache.l2_persist_max > 0) {
        set_l2_persist(nullptr, 0);
    }

    if (normalize) {
        cudaMemsetAsync(d_scratch, 0, sizeof(float));
        int blk = (N + BLOCK - 1) / BLOCK;
        l2_sq_kernel<<<blk, BLOCK>>>(result, N, d_scratch);
        norm_kernel<<<blk, BLOCK>>>(result, N, d_scratch);
    }

    
    cudaMemcpyAsync(centralities, result,
                    (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice);

    return {iterations, converged};
}

}  
