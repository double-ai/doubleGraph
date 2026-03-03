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
#include <cmath>
#include <cstring>
#include <utility>

namespace aai {

namespace {





struct Cache : Cacheable {
    float* buf_a = nullptr;
    float* buf_b = nullptr;
    float* accum = nullptr;
    int32_t buf_a_capacity = 0;
    int32_t buf_b_capacity = 0;
    bool accum_allocated = false;
    size_t l2_persist_max = 0;
    bool l2_persist_initialized = false;

    void ensure(int32_t num_vertices) {
        if (buf_a_capacity < num_vertices) {
            if (buf_a) cudaFree(buf_a);
            cudaMalloc(&buf_a, (size_t)num_vertices * sizeof(float));
            buf_a_capacity = num_vertices;
        }
        if (buf_b_capacity < num_vertices) {
            if (buf_b) cudaFree(buf_b);
            cudaMalloc(&buf_b, (size_t)num_vertices * sizeof(float));
            buf_b_capacity = num_vertices;
        }
        if (!accum_allocated) {
            cudaMalloc(&accum, 2 * sizeof(float));
            accum_allocated = true;
        }
    }

    void ensure_l2(void) {
        if (!l2_persist_initialized) {
            int l2_size = 0;
            cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, 0);
            l2_persist_max = (size_t)(l2_size * 2 / 3);
            if (l2_persist_max > 0) {
                cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_persist_max);
            }
            l2_persist_initialized = true;
        }
    }

    ~Cache() override {
        if (l2_persist_max > 0) {
            cudaStreamAttrValue attr;
            memset(&attr, 0, sizeof(attr));
            attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
            attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
            cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
        }
        if (buf_a) cudaFree(buf_a);
        if (buf_b) cudaFree(buf_b);
        if (accum) cudaFree(accum);
    }
};





__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val) {
    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    __shared__ float warp_sums[NUM_WARPS];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();
    val = (threadIdx.x < NUM_WARPS) ? warp_sums[threadIdx.x] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}





__global__ void __launch_bounds__(256, 4) spmv_high_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float*   __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float*   __restrict__ x_old,
    float*         __restrict__ x_new,
    float alpha, float beta_scalar, const float* __restrict__ betas,
    int32_t v_start, int32_t v_end,
    float* __restrict__ d_l1_diff
) {
    int v = v_start + blockIdx.x;
    if (v >= v_end) return;
    int start = __ldg(&offsets[v]);
    int end   = __ldg(&offsets[v + 1]);
    float sum = 0.0f;
    for (int e = start + threadIdx.x; e < end; e += blockDim.x) {
        uint32_t mw = __ldg(&edge_mask[e >> 5]);
        if ((mw >> (e & 31)) & 1) {
            int u = __ldg(&indices[e]);
            sum += __ldg(&edge_weights[e]) * __ldg(&x_old[u]);
        }
    }
    sum = block_reduce_sum<256>(sum);
    if (threadIdx.x == 0) {
        float beta_v = betas ? __ldg(&betas[v]) : beta_scalar;
        float new_val = alpha * sum + beta_v;
        x_new[v] = new_val;
        atomicAdd(d_l1_diff, fabsf(new_val - x_old[v]));
    }
}





__global__ void __launch_bounds__(256, 4) spmv_mid_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float*   __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float*   __restrict__ x_old,
    float*         __restrict__ x_new,
    float alpha, float beta_scalar, const float* __restrict__ betas,
    int32_t v_start, int32_t v_end,
    float* __restrict__ d_l1_diff
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid >> 5;
    int lane    = global_tid & 31;
    int v = v_start + warp_id;
    if (v >= v_end) return;
    int start = __ldg(&offsets[v]);
    int end   = __ldg(&offsets[v + 1]);
    float sum = 0.0f;
    for (int e = start + lane; e < end; e += 32) {
        uint32_t mw = __ldg(&edge_mask[e >> 5]);
        if ((mw >> (e & 31)) & 1) {
            int u = __ldg(&indices[e]);
            sum += __ldg(&edge_weights[e]) * __ldg(&x_old[u]);
        }
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        float beta_v = betas ? __ldg(&betas[v]) : beta_scalar;
        float new_val = alpha * sum + beta_v;
        x_new[v] = new_val;
        atomicAdd(d_l1_diff, fabsf(new_val - x_old[v]));
    }
}





__global__ void __launch_bounds__(256, 4) spmv_low_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float*   __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const float*   __restrict__ x_old,
    float*         __restrict__ x_new,
    float alpha, float beta_scalar, const float* __restrict__ betas,
    int32_t v_start, int32_t v_end,
    float* __restrict__ d_l1_diff
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int v = v_start + tid;
    float local_diff = 0.0f;
    if (v < v_end) {
        int start = __ldg(&offsets[v]);
        int end   = __ldg(&offsets[v + 1]);
        float old_val = x_old[v];
        float sum = 0.0f;
        for (int e = start; e < end; e++) {
            uint32_t mw = __ldg(&edge_mask[e >> 5]);
            if ((mw >> (e & 31)) & 1) {
                int u = __ldg(&indices[e]);
                sum += __ldg(&edge_weights[e]) * __ldg(&x_old[u]);
            }
        }
        float beta_v = betas ? __ldg(&betas[v]) : beta_scalar;
        float new_val = alpha * sum + beta_v;
        x_new[v] = new_val;
        local_diff = fabsf(new_val - old_val);
    }
    float bd = block_reduce_sum<256>(local_diff);
    if (threadIdx.x == 0) atomicAdd(d_l1_diff, bd);
}





__global__ void __launch_bounds__(256, 4) spmv_const_x_high(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ x_new,
    float alpha_times_x_const, float beta_scalar, const float* __restrict__ betas,
    float x_const, int32_t v_start, int32_t v_end,
    float* __restrict__ d_l1_diff
) {
    int v = v_start + blockIdx.x;
    if (v >= v_end) return;
    int start = __ldg(&offsets[v]);
    int end   = __ldg(&offsets[v + 1]);
    float wsum = 0.0f;
    for (int e = start + threadIdx.x; e < end; e += blockDim.x) {
        uint32_t mw = __ldg(&edge_mask[e >> 5]);
        if ((mw >> (e & 31)) & 1) wsum += __ldg(&edge_weights[e]);
    }
    wsum = block_reduce_sum<256>(wsum);
    if (threadIdx.x == 0) {
        float beta_v = betas ? __ldg(&betas[v]) : beta_scalar;
        float new_val = alpha_times_x_const * wsum + beta_v;
        x_new[v] = new_val;
        atomicAdd(d_l1_diff, fabsf(new_val - x_const));
    }
}

__global__ void __launch_bounds__(256, 4) spmv_const_x_mid(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ x_new,
    float alpha_times_x_const, float beta_scalar, const float* __restrict__ betas,
    float x_const, int32_t v_start, int32_t v_end,
    float* __restrict__ d_l1_diff
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid >> 5;
    int lane    = global_tid & 31;
    int v = v_start + warp_id;
    if (v >= v_end) return;
    int start = __ldg(&offsets[v]);
    int end   = __ldg(&offsets[v + 1]);
    float wsum = 0.0f;
    for (int e = start + lane; e < end; e += 32) {
        uint32_t mw = __ldg(&edge_mask[e >> 5]);
        if ((mw >> (e & 31)) & 1) wsum += __ldg(&edge_weights[e]);
    }
    wsum = warp_reduce_sum(wsum);
    if (lane == 0) {
        float beta_v = betas ? __ldg(&betas[v]) : beta_scalar;
        float new_val = alpha_times_x_const * wsum + beta_v;
        x_new[v] = new_val;
        atomicAdd(d_l1_diff, fabsf(new_val - x_const));
    }
}

__global__ void __launch_bounds__(256, 4) spmv_const_x_low(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ x_new,
    float alpha_times_x_const, float beta_scalar, const float* __restrict__ betas,
    float x_const, int32_t v_start, int32_t v_end,
    float* __restrict__ d_l1_diff
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int v = v_start + tid;
    float local_diff = 0.0f;
    if (v < v_end) {
        int start = __ldg(&offsets[v]);
        int end   = __ldg(&offsets[v + 1]);
        float wsum = 0.0f;
        for (int e = start; e < end; e++) {
            uint32_t mw = __ldg(&edge_mask[e >> 5]);
            if ((mw >> (e & 31)) & 1) wsum += __ldg(&edge_weights[e]);
        }
        float beta_v = betas ? __ldg(&betas[v]) : beta_scalar;
        float new_val = alpha_times_x_const * wsum + beta_v;
        x_new[v] = new_val;
        local_diff = fabsf(new_val - x_const);
    }
    float bd = block_reduce_sum<256>(local_diff);
    if (threadIdx.x == 0) atomicAdd(d_l1_diff, bd);
}





__global__ void fill_beta_fused(
    float* __restrict__ x, float beta_scalar, const float* __restrict__ betas,
    int32_t n, float* __restrict__ d_sum
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (i < n) {
        val = betas ? __ldg(&betas[i]) : beta_scalar;
        x[i] = val;
        val = fabsf(val);
    }
    float bsum = block_reduce_sum<256>(val);
    if (threadIdx.x == 0) atomicAdd(d_sum, bsum);
}

__global__ void sum_squares_kernel(const float* __restrict__ x, int32_t n, float* __restrict__ d_sum) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (i < n) { float xi = x[i]; val = xi * xi; }
    float bsum = block_reduce_sum<256>(val);
    if (threadIdx.x == 0) atomicAdd(d_sum, bsum);
}

__global__ void scale_kernel(float* __restrict__ x, int32_t n, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}





void launch_spmv_segments(
    const int32_t* offsets, const int32_t* indices,
    const float* edge_weights, const uint32_t* edge_mask,
    const float* x_old, float* x_new,
    float alpha, float beta_scalar, const float* betas,
    int32_t num_vertices, const int32_t* seg, float* d_l1_diff
) {
    int n_high = seg[1] - seg[0];
    if (n_high > 0) {
        spmv_high_degree<<<n_high, 256>>>(offsets, indices, edge_weights, edge_mask,
            x_old, x_new, alpha, beta_scalar, betas, seg[0], seg[1], d_l1_diff);
    }
    int n_mid = seg[2] - seg[1];
    if (n_mid > 0) {
        int warps_per_block = 8;
        int blocks = (n_mid + warps_per_block - 1) / warps_per_block;
        spmv_mid_degree<<<blocks, warps_per_block * 32>>>(offsets, indices, edge_weights, edge_mask,
            x_old, x_new, alpha, beta_scalar, betas, seg[1], seg[2], d_l1_diff);
    }
    int n_low_zero = seg[4] - seg[2];
    if (n_low_zero > 0) {
        int blocks = (n_low_zero + 255) / 256;
        spmv_low_degree<<<blocks, 256>>>(offsets, indices, edge_weights, edge_mask,
            x_old, x_new, alpha, beta_scalar, betas, seg[2], seg[4], d_l1_diff);
    }
}

void launch_spmv_const_x(
    const int32_t* offsets,
    const float* edge_weights, const uint32_t* edge_mask,
    float* x_new,
    float alpha_times_x_const, float beta_scalar, const float* betas,
    float x_const, int32_t num_vertices, const int32_t* seg, float* d_l1_diff
) {
    int n_high = seg[1] - seg[0];
    if (n_high > 0) {
        spmv_const_x_high<<<n_high, 256>>>(offsets, edge_weights, edge_mask, x_new,
            alpha_times_x_const, beta_scalar, betas, x_const, seg[0], seg[1], d_l1_diff);
    }
    int n_mid = seg[2] - seg[1];
    if (n_mid > 0) {
        int warps_per_block = 8;
        int blocks = (n_mid + warps_per_block - 1) / warps_per_block;
        spmv_const_x_mid<<<blocks, warps_per_block * 32>>>(offsets, edge_weights, edge_mask, x_new,
            alpha_times_x_const, beta_scalar, betas, x_const, seg[1], seg[2], d_l1_diff);
    }
    int n_low_zero = seg[4] - seg[2];
    if (n_low_zero > 0) {
        int blocks = (n_low_zero + 255) / 256;
        spmv_const_x_low<<<blocks, 256>>>(offsets, edge_weights, edge_mask, x_new,
            alpha_times_x_const, beta_scalar, betas, x_const, seg[2], seg[4], d_l1_diff);
    }
}





void set_l2_persist(const void* ptr, size_t bytes) {
    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));
    attr.accessPolicyWindow.base_ptr = const_cast<void*>(ptr);
    attr.accessPolicyWindow.num_bytes = bytes;
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
}

void clear_l2_persist() {
    cudaStreamAttrValue attr;
    memset(&attr, 0, sizeof(attr));
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
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

    int32_t num_vertices = graph.number_of_vertices;

    cache.ensure(num_vertices);
    cache.ensure_l2();

    const int32_t*  d_offsets      = graph.offsets;
    const int32_t*  d_indices      = graph.indices;
    const uint32_t* d_edge_mask    = graph.edge_mask;

    const auto& seg_vec = graph.segment_offsets.value();
    int32_t seg[5] = {seg_vec[0], seg_vec[1], seg_vec[2], seg_vec[3], seg_vec[4]};

    float* x_old     = cache.buf_a;
    float* x_new     = cache.buf_b;
    float* d_l1_diff = cache.accum;
    float* d_l2_norm = cache.accum + 1;

    
    if (has_initial_guess) {
        cudaMemcpyAsync(x_old, centralities,
                        (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        cudaMemsetAsync(x_old, 0, (size_t)num_vertices * sizeof(float));
    }

    size_t iterations = 0;
    bool converged = false;
    bool can_use_const_x = !has_initial_guess && (betas == nullptr);

    
    size_t x_bytes = (size_t)num_vertices * sizeof(float);
    size_t persist_bytes = (x_bytes < cache.l2_persist_max) ? x_bytes : cache.l2_persist_max;

    
    if (!has_initial_guess && max_iterations > 0) {
        cudaMemsetAsync(d_l1_diff, 0, sizeof(float));
        fill_beta_fused<<<(num_vertices + 255)/256, 256>>>(x_new, beta, betas, num_vertices, d_l1_diff);
        std::swap(x_old, x_new);
        iterations = 1;

        float h_diff;
        cudaMemcpy(&h_diff, d_l1_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_diff < epsilon) converged = true;
    }

    
    if (!converged && iterations < max_iterations && can_use_const_x) {
        cudaMemsetAsync(d_l1_diff, 0, sizeof(float));
        launch_spmv_const_x(
            d_offsets, edge_weights, d_edge_mask, x_new,
            alpha * beta, beta, betas, beta,
            num_vertices, seg, d_l1_diff);
        std::swap(x_old, x_new);
        iterations++;

        float h_diff;
        cudaMemcpy(&h_diff, d_l1_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_diff < epsilon) converged = true;
    }

    
    while (!converged && iterations < max_iterations) {
        if (cache.l2_persist_max > 0 && persist_bytes > 0) {
            set_l2_persist(x_old, persist_bytes);
        }

        cudaMemsetAsync(d_l1_diff, 0, sizeof(float));
        launch_spmv_segments(
            d_offsets, d_indices, edge_weights, d_edge_mask,
            x_old, x_new, alpha, beta, betas,
            num_vertices, seg, d_l1_diff);

        std::swap(x_old, x_new);
        iterations++;

        float h_diff;
        cudaMemcpy(&h_diff, d_l1_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_diff < epsilon) converged = true;
    }

    if (cache.l2_persist_max > 0) {
        clear_l2_persist();
    }

    
    if (normalize) {
        cudaMemsetAsync(d_l2_norm, 0, sizeof(float));
        sum_squares_kernel<<<(num_vertices + 255)/256, 256>>>(x_old, num_vertices, d_l2_norm);
        float h_sum_sq;
        cudaMemcpy(&h_sum_sq, d_l2_norm, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_sum_sq > 0.0f) {
            scale_kernel<<<(num_vertices + 255)/256, 256>>>(x_old, num_vertices, 1.0f / sqrtf(h_sum_sq));
        }
    }

    
    cudaMemcpyAsync(centralities, x_old, (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);

    return {iterations, converged};
}

}  
