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
#include <cmath>

namespace aai {

namespace {


struct alignas(8) ScalarBuf {
    float diff;
};

struct Cache : Cacheable {
    float* d_diff = nullptr;
    ScalarBuf* h_buf = nullptr;
    cudaStream_t stream = nullptr;
    bool l2_persist_active = false;

    float* x0 = nullptr;
    float* x1 = nullptr;
    int64_t x0_capacity = 0;
    int64_t x1_capacity = 0;

    Cache() {
        cudaMalloc(&d_diff, sizeof(float));
        cudaHostAlloc(&h_buf, sizeof(ScalarBuf), cudaHostAllocDefault);
        cudaStreamCreate(&stream);
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 4 * 1024 * 1024);
    }

    ~Cache() override {
        if (d_diff) cudaFree(d_diff);
        if (h_buf) cudaFreeHost(h_buf);
        if (stream) cudaStreamDestroy(stream);
        if (x0) cudaFree(x0);
        if (x1) cudaFree(x1);
    }

    void ensure(int64_t n) {
        if (x0_capacity < n) {
            if (x0) cudaFree(x0);
            cudaMalloc(&x0, n * sizeof(float));
            x0_capacity = n;
        }
        if (x1_capacity < n) {
            if (x1) cudaFree(x1);
            cudaMalloc(&x1, n * sizeof(float));
            x1_capacity = n;
        }
    }

    void set_l2_persist(void* ptr, size_t bytes) {
        if (bytes == 0 || bytes > 32ULL * 1024 * 1024) {
            if (l2_persist_active) {
                cudaStreamAttrValue attr = {};
                attr.accessPolicyWindow.num_bytes = 0;
                cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
                l2_persist_active = false;
            }
            return;
        }
        const size_t L2_PERSIST = 4ULL * 1024 * 1024;
        float ratio = (float)L2_PERSIST / (float)bytes;
        if (ratio > 1.0f) ratio = 1.0f;
        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.base_ptr = ptr;
        attr.accessPolicyWindow.num_bytes = bytes;
        attr.accessPolicyWindow.hitRatio = ratio;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
        l2_persist_active = true;
    }
};





template <int WARPS_PER_BLK>
__global__ void katz_spmv_warp_kernel(
    const int n,
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float alpha,
    const float beta,
    const float* __restrict__ betas,
    const bool use_betas,
    float* __restrict__ diff_out)
{
    constexpr int WARP_SIZE = 32;
    const int warp_id = (blockIdx.x * WARPS_PER_BLK) + (threadIdx.x / WARP_SIZE);
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x / WARP_SIZE;

    __shared__ float smem[WARPS_PER_BLK];

    float my_diff = 0.0f;

    if (warp_id < n) {
        const int v = warp_id;
        const int start = offsets[v];
        const int end = offsets[v + 1];

        float sum = 0.0f;
        for (int e = start + lane; e < end; e += WARP_SIZE) {
            uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
            if (mask_word & (1u << (e & 31))) {
                sum += __ldg(&weights[e]) * __ldg(&x_old[__ldg(&indices[e])]);
            }
        }

        
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, d);
        }

        if (lane == 0) {
            float beta_v = use_betas ? __ldg(&betas[v]) : beta;
            float new_val = alpha * sum + beta_v;
            float old_val = __ldg(&x_old[v]);
            x_new[v] = new_val;
            my_diff = fabsf(new_val - old_val);
        }
    }

    
    if (lane == 0) {
        smem[warp_in_block] = my_diff;
    }
    __syncthreads();

    if (warp_in_block == 0) {
        float val = (lane < WARPS_PER_BLK) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, d);
        }
        if (lane == 0 && val > 0.0f) {
            atomicAdd(diff_out, val);
        }
    }
}





__global__ void katz_spmv_thread_kernel(
    const int n,
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float alpha,
    const float beta,
    const float* __restrict__ betas,
    const bool use_betas,
    float* __restrict__ diff_out)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0.0f;

    if (v < n) {
        int start = offsets[v];
        int end = offsets[v + 1];
        float old_val = x_old[v];

        float sum = 0.0f;
        for (int e = start; e < end; e++) {
            uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
            if (mask_word & (1u << (e & 31))) {
                sum += __ldg(&weights[e]) * __ldg(&x_old[__ldg(&indices[e])]);
            }
        }

        float beta_v = use_betas ? betas[v] : beta;
        float new_val = alpha * sum + beta_v;
        x_new[v] = new_val;
        diff = fabsf(new_val - old_val);
    }

    float block_sum = BlockReduce(temp_storage).Sum(diff);
    if (threadIdx.x == 0 && block_sum > 0.0f) {
        atomicAdd(diff_out, block_sum);
    }
}




__global__ void fill_beta_diff_kernel(
    float* __restrict__ x_new,
    const float* __restrict__ x_old,
    const float beta,
    const float* __restrict__ betas,
    const bool use_betas,
    const int n,
    float* __restrict__ diff_out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (diff_out != nullptr) {
        typedef cub::BlockReduce<float, 256> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        float diff = 0.0f;

        if (i < n) {
            float val = use_betas ? betas[i] : beta;
            x_new[i] = val;
            diff = fabsf(val - x_old[i]);
        }

        float block_sum = BlockReduce(temp_storage).Sum(diff);
        if (threadIdx.x == 0 && block_sum > 0.0f) {
            atomicAdd(diff_out, block_sum);
        }
    } else {
        if (i < n) {
            x_new[i] = use_betas ? betas[i] : beta;
        }
    }
}




__global__ void l2_norm_sq_kernel(const float* __restrict__ x, float* __restrict__ norm_sq, int n)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float v = x[i];
        sum += v * v;
    }

    float block_sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0 && block_sum > 0.0f) {
        atomicAdd(norm_sq, block_sum);
    }
}




__global__ void normalize_kernel(float* __restrict__ x, float inv_norm, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= inv_norm;
    }
}





void launch_katz_spmv(
    int n, const int* offsets, const int* indices, const float* weights,
    const uint32_t* edge_mask, const float* x_old, float* x_new,
    float alpha, float beta, const float* betas, bool use_betas,
    float* diff_out, int avg_degree, cudaStream_t stream)
{
    if (n <= 0) return;

    if (avg_degree <= 8) {
        
        int block = 256;
        int grid = (n + block - 1) / block;
        katz_spmv_thread_kernel<<<grid, block, 0, stream>>>(
            n, offsets, indices, weights, edge_mask, x_old, x_new,
            alpha, beta, betas, use_betas, diff_out);
    } else {
        
        constexpr int WPB = 8;
        int block = WPB * 32; 
        int grid = (n + WPB - 1) / WPB;
        katz_spmv_warp_kernel<WPB><<<grid, block, 0, stream>>>(
            n, offsets, indices, weights, edge_mask, x_old, x_new,
            alpha, beta, betas, use_betas, diff_out);
    }
}

void launch_fill_beta_diff(
    float* x_new, const float* x_old, float beta,
    const float* betas, bool use_betas, int n,
    float* diff_out, cudaStream_t stream)
{
    if (n <= 0) return;
    int grid = (n + 255) / 256;
    fill_beta_diff_kernel<<<grid, 256, 0, stream>>>(
        x_new, x_old, beta, betas, use_betas, n, diff_out);
}

void launch_l2_norm_sq(const float* x, float* norm_sq, int n, cudaStream_t stream)
{
    if (n <= 0) return;
    int grid = (n + 255) / 256;
    if (grid > 1024) grid = 1024;
    l2_norm_sq_kernel<<<grid, 256, 0, stream>>>(x, norm_sq, n);
}

void launch_normalize(float* x, float inv_norm, int n, cudaStream_t stream)
{
    if (n <= 0) return;
    int grid = (n + 255) / 256;
    normalize_kernel<<<grid, 256, 0, stream>>>(x, inv_norm, n);
}

}  

katz_centrality_result_t katz_centrality_mask(const graph32_t& graph,
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

    const int* offsets = graph.offsets;
    const int* indices = graph.indices;
    int n = graph.number_of_vertices;
    int num_edges = graph.number_of_edges;
    const float* weights = edge_weights;
    const uint32_t* edge_mask = graph.edge_mask;
    bool use_betas = (betas != nullptr);
    bool do_normalize = normalize;

    cache.ensure((int64_t)n);
    float* x0 = cache.x0;
    float* x1 = cache.x1;
    float* x_old = x0;
    float* x_new = x1;

    if (has_initial_guess) {
        cudaMemcpyAsync(x_old, centralities,
            (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice, cache.stream);
    } else {
        cudaMemsetAsync(x_old, 0, (size_t)n * sizeof(float), cache.stream);
    }

    int avg_degree = (n > 0) ? (num_edges / n) : 0;
    size_t x_size = (size_t)n * sizeof(float);

    
    cache.set_l2_persist((void*)x_old, x_size);

    size_t iterations = 0;
    bool converged = false;

    
    if (!has_initial_guess && iterations < max_iterations) {
        bool skip_first_check = (!use_betas && (float)n * fabsf(beta) > epsilon * 10.0f);

        if (skip_first_check) {
            launch_fill_beta_diff(x_new, x_old, beta, betas, use_betas, n, nullptr, cache.stream);
            iterations++;
            float* temp = x_old; x_old = x_new; x_new = temp;
            cache.set_l2_persist((void*)x_old, x_size);
        } else {
            cudaMemsetAsync(cache.d_diff, 0, sizeof(float), cache.stream);
            launch_fill_beta_diff(x_new, x_old, beta, betas, use_betas, n, cache.d_diff, cache.stream);
            cudaMemcpyAsync(&cache.h_buf->diff, cache.d_diff, sizeof(float), cudaMemcpyDeviceToHost, cache.stream);
            cudaStreamSynchronize(cache.stream);
            iterations++;
            if (cache.h_buf->diff < epsilon) {
                converged = true;
            }
            float* temp = x_old; x_old = x_new; x_new = temp;
            cache.set_l2_persist((void*)x_old, x_size);
        }
    }

    
    while (iterations < max_iterations && !converged) {
        cudaMemsetAsync(cache.d_diff, 0, sizeof(float), cache.stream);

        launch_katz_spmv(n, offsets, indices, weights, edge_mask,
            x_old, x_new, alpha, beta, betas, use_betas, cache.d_diff, avg_degree, cache.stream);

        cudaMemcpyAsync(&cache.h_buf->diff, cache.d_diff, sizeof(float), cudaMemcpyDeviceToHost, cache.stream);
        cudaStreamSynchronize(cache.stream);

        iterations++;
        if (cache.h_buf->diff < epsilon) {
            converged = true;
        }

        float* temp = x_old; x_old = x_new; x_new = temp;
        cache.set_l2_persist((void*)x_old, x_size);
    }

    
    cache.set_l2_persist(nullptr, 0);

    float* result = x_old;

    if (do_normalize && n > 0) {
        cudaMemsetAsync(cache.d_diff, 0, sizeof(float), cache.stream);
        launch_l2_norm_sq(result, cache.d_diff, n, cache.stream);
        cudaMemcpyAsync(&cache.h_buf->diff, cache.d_diff, sizeof(float), cudaMemcpyDeviceToHost, cache.stream);
        cudaStreamSynchronize(cache.stream);
        float l2_norm = sqrtf(cache.h_buf->diff);
        if (l2_norm > 0.0f) {
            launch_normalize(result, 1.0f / l2_norm, n, cache.stream);
        }
    }

    
    cudaMemcpyAsync(centralities, result,
        (size_t)n * sizeof(float), cudaMemcpyDeviceToDevice, cache.stream);
    cudaStreamSynchronize(cache.stream);

    return {iterations, converged};
}

}  
