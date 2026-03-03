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





struct Cache : Cacheable {
    float* f_weights = nullptr;
    float* f_betas = nullptr;
    uint8_t* work_buf = nullptr;

    int64_t f_weights_capacity = 0;
    int64_t f_betas_capacity = 0;
    int64_t work_buf_capacity = 0;

    void ensure_weights(int64_t num_edges) {
        if (f_weights_capacity < num_edges) {
            if (f_weights) cudaFree(f_weights);
            cudaMalloc(&f_weights, num_edges * sizeof(float));
            f_weights_capacity = num_edges;
        }
    }

    void ensure_betas(int64_t num_vertices) {
        if (f_betas_capacity < num_vertices) {
            if (f_betas) cudaFree(f_betas);
            cudaMalloc(&f_betas, num_vertices * sizeof(float));
            f_betas_capacity = num_vertices;
        }
    }

    void ensure_work(int64_t size) {
        if (work_buf_capacity < size) {
            if (work_buf) cudaFree(work_buf);
            cudaMalloc(&work_buf, size);
            work_buf_capacity = size;
        }
    }

    ~Cache() override {
        if (f_weights) cudaFree(f_weights);
        if (f_betas) cudaFree(f_betas);
        if (work_buf) cudaFree(work_buf);
    }
};






__global__ void __launch_bounds__(256)
spmv_high_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha, float beta, const float* __restrict__ betas,
    int32_t start_v, int32_t end_v,
    float* __restrict__ g_diff)
{
    int v = blockIdx.x + start_v;
    if (v >= end_v) return;

    int e_start = offsets[v];
    int e_end = offsets[v + 1];

    float sum = 0.0f;
    for (int e = e_start + threadIdx.x; e < e_end; e += 256) {
        sum += weights[e] * x_old[indices[e]];
    }

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        float new_val = alpha * sum + (betas != nullptr ? betas[v] : beta);
        x_new[v] = new_val;
        float d = new_val - x_old[v];
        atomicAdd(g_diff, d < 0.0f ? -d : d);
    }
}


__global__ void __launch_bounds__(256)
spmv_mid_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha, float beta, const float* __restrict__ betas,
    int32_t start_v, int32_t end_v,
    float* __restrict__ g_diff)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid / 32;
    int lane = global_tid & 31;
    int v = start_v + warp_id;

    float my_diff = 0.0f;
    if (v < end_v) {
        int e_start = offsets[v];
        int e_end = offsets[v + 1];

        float sum = 0.0f;
        for (int e = e_start + lane; e < e_end; e += 32) {
            sum += weights[e] * x_old[indices[e]];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            float new_val = alpha * sum + (betas != nullptr ? betas[v] : beta);
            x_new[v] = new_val;
            float d = new_val - x_old[v];
            my_diff = d < 0.0f ? -d : d;
        }
    }

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_diff = BlockReduce(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff > 0.0f) {
        atomicAdd(g_diff, block_diff);
    }
}


__global__ void __launch_bounds__(256)
spmv_low_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float alpha, float beta, const float* __restrict__ betas,
    int32_t start_v, int32_t end_v,
    float* __restrict__ g_diff)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int v = start_v + tid;

    float my_diff = 0.0f;
    if (v < end_v) {
        int e_start = offsets[v];
        int e_end = offsets[v + 1];

        float sum = 0.0f;
        for (int e = e_start; e < e_end; ++e) {
            sum += weights[e] * x_old[indices[e]];
        }

        float new_val = alpha * sum + (betas != nullptr ? betas[v] : beta);
        x_new[v] = new_val;
        float d = new_val - x_old[v];
        my_diff = d < 0.0f ? -d : d;
    }

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_diff = BlockReduce(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff > 0.0f) {
        atomicAdd(g_diff, block_diff);
    }
}


__global__ void __launch_bounds__(256)
spmv_zero_degree(
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float beta, const float* __restrict__ betas,
    int32_t start_v, int32_t end_v,
    float* __restrict__ g_diff)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int v = start_v + tid;

    float my_diff = 0.0f;
    if (v < end_v) {
        float new_val = (betas != nullptr ? betas[v] : beta);
        x_new[v] = new_val;
        float d = new_val - x_old[v];
        my_diff = d < 0.0f ? -d : d;
    }

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_diff = BlockReduce(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff > 0.0f) {
        atomicAdd(g_diff, block_diff);
    }
}


__global__ void double_to_float_kernel(
    const double* __restrict__ src,
    float* __restrict__ dst,
    int32_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = (float)src[i];
    }
}


__global__ void float_to_double_kernel(
    const float* __restrict__ src,
    double* __restrict__ dst,
    int32_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = (double)src[i];
    }
}


__global__ void __launch_bounds__(256)
l2_norm_sq_kernel(
    const float* __restrict__ x,
    float* __restrict__ result,
    float* __restrict__ partials,
    unsigned int* __restrict__ retire_count,
    int32_t n)
{
    float sum = 0.0f;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        float v = x[i];
        sum += v * v;
    }

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float block_sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        partials[blockIdx.x] = block_sum;
    }

    __threadfence();

    __shared__ bool am_last;
    if (threadIdx.x == 0) {
        unsigned int ticket = atomicAdd(retire_count, 1);
        am_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (am_last) {
        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < (int)gridDim.x; i += 256) {
            local_sum += partials[i];
        }
        float final_sum = BlockReduce(temp_storage).Sum(local_sum);
        if (threadIdx.x == 0) {
            result[0] = final_sum;
            *retire_count = 0;
        }
    }
}


__global__ void normalize_kernel(
    float* __restrict__ x,
    float inv_norm,
    int32_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= inv_norm;
    }
}


__global__ void fill_scalar_kernel(float* __restrict__ dst, float val, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = val;
}







__global__ void __launch_bounds__(256)
spmv_const_high(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ weights,
    float* __restrict__ x_new,
    float alpha_times_const, float beta,
    int32_t start_v, int32_t end_v)
{
    int v = blockIdx.x + start_v;
    if (v >= end_v) return;
    int e_start = offsets[v];
    int e_end = offsets[v + 1];
    float wsum = 0.0f;
    for (int e = e_start + threadIdx.x; e < e_end; e += 256)
        wsum += weights[e];
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage ts;
    wsum = BR(ts).Sum(wsum);
    if (threadIdx.x == 0)
        x_new[v] = alpha_times_const * wsum + beta;
}

__global__ void __launch_bounds__(256)
spmv_const_mid(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ weights,
    float* __restrict__ x_new,
    float alpha_times_const, float beta,
    int32_t start_v, int32_t end_v)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = gid / 32;
    int lane = gid & 31;
    int v = start_v + warp_id;
    if (v >= end_v) return;
    int e_start = offsets[v];
    int e_end = offsets[v + 1];
    float wsum = 0.0f;
    for (int e = e_start + lane; e < e_end; e += 32)
        wsum += weights[e];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        wsum += __shfl_down_sync(0xffffffff, wsum, off);
    if (lane == 0)
        x_new[v] = alpha_times_const * wsum + beta;
}

__global__ void __launch_bounds__(256)
spmv_const_low(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ weights,
    float* __restrict__ x_new,
    float alpha_times_const, float beta,
    int32_t start_v, int32_t end_v)
{
    int v = start_v + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= end_v) return;
    int e_start = offsets[v];
    int e_end = offsets[v + 1];
    float wsum = 0.0f;
    for (int e = e_start; e < e_end; ++e)
        wsum += weights[e];
    x_new[v] = alpha_times_const * wsum + beta;
}





static void launch_spmv_fused(
    const int32_t* offsets, const int32_t* indices, const float* weights,
    const float* x_old, float* x_new,
    float alpha, float beta, const float* betas,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3, int32_t seg4,
    float* g_diff, cudaStream_t stream)
{
    cudaMemsetAsync(g_diff, 0, sizeof(float), stream);

    int n_high = seg1 - seg0;
    if (n_high > 0) {
        spmv_high_degree<<<n_high, 256, 0, stream>>>(
            offsets, indices, weights, x_old, x_new,
            alpha, beta, betas, seg0, seg1, g_diff);
    }

    int n_mid = seg2 - seg1;
    if (n_mid > 0) {
        int blocks = (n_mid + 7) / 8;
        spmv_mid_degree<<<blocks, 256, 0, stream>>>(
            offsets, indices, weights, x_old, x_new,
            alpha, beta, betas, seg1, seg2, g_diff);
    }

    int n_low = seg3 - seg2;
    if (n_low > 0) {
        int blocks = (n_low + 255) / 256;
        spmv_low_degree<<<blocks, 256, 0, stream>>>(
            offsets, indices, weights, x_old, x_new,
            alpha, beta, betas, seg2, seg3, g_diff);
    }

    int n_zero = seg4 - seg3;
    if (n_zero > 0) {
        int blocks = (n_zero + 255) / 256;
        spmv_zero_degree<<<blocks, 256, 0, stream>>>(
            x_old, x_new, beta, betas, seg3, seg4, g_diff);
    }
}

static void launch_spmv_const(
    const int32_t* offsets, const float* weights, float* x_new,
    float alpha_times_const, float beta,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3, int32_t seg4,
    cudaStream_t stream)
{
    int n_high = seg1 - seg0;
    if (n_high > 0) {
        spmv_const_high<<<n_high, 256, 0, stream>>>(
            offsets, weights, x_new, alpha_times_const, beta, seg0, seg1);
    }
    int n_mid = seg2 - seg1;
    if (n_mid > 0) {
        spmv_const_mid<<<(n_mid + 7) / 8, 256, 0, stream>>>(
            offsets, weights, x_new, alpha_times_const, beta, seg1, seg2);
    }
    int n_low = seg3 - seg2;
    if (n_low > 0) {
        spmv_const_low<<<(n_low + 255) / 256, 256, 0, stream>>>(
            offsets, weights, x_new, alpha_times_const, beta, seg2, seg3);
    }
    int n_zero = seg4 - seg3;
    if (n_zero > 0) {
        int blocks = (n_zero + 255) / 256;
        fill_scalar_kernel<<<blocks, 256, 0, stream>>>(
            x_new + seg3, beta, n_zero);
    }
}

static void launch_d2f(const double* src, float* dst, int32_t n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    double_to_float_kernel<<<blocks, 256, 0, stream>>>(src, dst, n);
}

static void launch_f2d(const float* src, double* dst, int32_t n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    float_to_double_kernel<<<blocks, 256, 0, stream>>>(src, dst, n);
}

static void launch_l2_norm(
    const float* x, float* result,
    float* partials, unsigned int* retire_count,
    int32_t n, int num_blocks, cudaStream_t stream)
{
    l2_norm_sq_kernel<<<num_blocks, 256, 0, stream>>>(
        x, result, partials, retire_count, n);
}

static void launch_normalize(float* x, float inv_norm, int32_t n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    normalize_kernel<<<blocks, 256, 0, stream>>>(x, inv_norm, n);
}

static void launch_fill(float* dst, float val, int32_t n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    fill_scalar_kernel<<<blocks, 256, 0, stream>>>(dst, val, n);
}

}  





katz_centrality_result_t katz_centrality_seg(const graph32_t& graph,
                         const double* edge_weights,
                         double* centralities,
                         double alpha,
                         double beta,
                         const double* betas,
                         double epsilon,
                         std::size_t max_iterations,
                         bool has_initial_guess,
                         bool normalize)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2];
    int32_t seg3 = seg[3], seg4 = seg[4];

    cudaStream_t stream = 0;

    float f_alpha = (float)alpha;
    float f_beta = (float)beta;
    float f_epsilon = (float)epsilon;
    bool use_betas = (betas != nullptr);

    
    cache.ensure_weights(num_edges);
    launch_d2f(edge_weights, cache.f_weights, num_edges, stream);

    
    const float* d_betas = nullptr;
    if (use_betas) {
        cache.ensure_betas(num_vertices);
        launch_d2f(betas, cache.f_betas, num_vertices, stream);
        d_betas = cache.f_betas;
    }

    
    
    int l2_blocks = 256;
    int64_t x_size = (int64_t)num_vertices * sizeof(float);
    int64_t aligned_x = (x_size + 255) & ~(int64_t)255;
    int64_t scratch_offset = 2 * aligned_x;
    int64_t l2_offset = scratch_offset + 256;
    int64_t retire_offset = l2_offset + l2_blocks * sizeof(float);
    int64_t total_size = retire_offset + sizeof(unsigned int) + 256;

    cache.ensure_work(total_size);
    uint8_t* wp = cache.work_buf;
    float* d_x0 = reinterpret_cast<float*>(wp);
    float* d_x1 = reinterpret_cast<float*>(wp + aligned_x);
    float* g_diff = reinterpret_cast<float*>(wp + scratch_offset);
    float* l2_partials = reinterpret_cast<float*>(wp + l2_offset);
    unsigned int* retire_count = reinterpret_cast<unsigned int*>(wp + retire_offset);

    float* d_x[2] = {d_x0, d_x1};
    int cur = 0;
    cudaMemsetAsync(retire_count, 0, sizeof(unsigned int), stream);

    const float* d_fw = cache.f_weights;

    bool converged = false;
    size_t iterations = 0;

    
    
    if (!has_initial_guess && !use_betas && max_iterations > 0) {
        
        launch_fill(d_x[1], f_beta, num_vertices, stream);
        cur = 1;
        iterations = 1;

        if (f_beta == 0.0f) {
            converged = true;
        }

        
        if (!converged && max_iterations > 1) {
            float atc = f_alpha * f_beta;
            launch_spmv_const(d_offsets, d_fw, d_x[0],
                             atc, f_beta,
                             seg0, seg1, seg2, seg3, seg4, stream);
            cur = 0;
            iterations = 2;
        }
    } else if (!has_initial_guess && use_betas && max_iterations > 0) {
        
        cudaMemcpyAsync(d_x[1], d_betas, num_vertices * sizeof(float),
            cudaMemcpyDeviceToDevice, stream);
        cur = 1;
        iterations = 1;
    } else if (has_initial_guess) {
        
        launch_d2f(centralities, d_x[cur], num_vertices, stream);
    } else {
        cudaMemsetAsync(d_x[cur], 0, x_size, stream);
    }

    
    if (!converged) {
        for (size_t iter = iterations; iter < max_iterations; ++iter) {
            int next = 1 - cur;

            launch_spmv_fused(d_offsets, d_indices, d_fw,
                            d_x[cur], d_x[next],
                            f_alpha, f_beta, d_betas,
                            seg0, seg1, seg2, seg3, seg4,
                            g_diff, stream);

            float h_diff;
            cudaMemcpy(&h_diff, g_diff, sizeof(float), cudaMemcpyDeviceToHost);

            cur = next;
            ++iterations;

            if (h_diff < f_epsilon) {
                converged = true;
                break;
            }
        }
    }

    if (normalize) {
        launch_l2_norm(d_x[cur], g_diff, l2_partials, retire_count,
                      num_vertices, l2_blocks, stream);
        float h_norm_sq;
        cudaMemcpy(&h_norm_sq, g_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_norm_sq > 0.0f) {
            launch_normalize(d_x[cur], 1.0f / sqrtf(h_norm_sq), num_vertices, stream);
        }
    }

    
    launch_f2d(d_x[cur], centralities, num_vertices, stream);

    return katz_centrality_result_t{iterations, converged};
}

}  
