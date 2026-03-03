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

namespace aai {

namespace {




struct Cache : Cacheable {
    float* buf = nullptr;
    float* scratch = nullptr;
    int64_t buf_capacity = 0;

    void ensure(int32_t n) {
        if (buf_capacity < n) {
            if (buf) cudaFree(buf);
            cudaMalloc(&buf, (int64_t)n * sizeof(float));
            buf_capacity = n;
        }
        if (!scratch) {
            cudaMalloc(&scratch, 2 * sizeof(float));
        }
    }

    ~Cache() override {
        if (buf) cudaFree(buf);
        if (scratch) cudaFree(scratch);
    }
};




__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ void block_reduce_add(float val, float* __restrict__ d_out) {
    constexpr int NWARPS = BLOCK_SIZE / 32;
    __shared__ float warp_sums[NWARPS];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) warp_sums[warp_id] = val;
    __syncthreads();
    if (warp_id == 0) {
        val = (lane < NWARPS) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) atomicAdd(d_out, val);
    }
}





__global__ void spmv_high(
    const int* __restrict__ offsets, const int* __restrict__ indices,
    const float* __restrict__ weights, const float* __restrict__ x_old,
    float* __restrict__ x_new, const float* __restrict__ betas,
    float alpha, float beta_scalar, int use_betas,
    int start_vertex, int num_verts, float* __restrict__ d_diff)
{
    int v_idx = blockIdx.x;
    if (v_idx >= num_verts) return;
    int v = v_idx + start_vertex;
    int row_start = offsets[v], row_end = offsets[v + 1];
    float sum = 0.0f;
    for (int e = row_start + (int)threadIdx.x; e < row_end; e += 256)
        sum += weights[e] * x_old[indices[e]];

    __shared__ float ws[8];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    sum = warp_reduce_sum(sum);
    if (lane == 0) ws[wid] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < 8; i++) total += ws[i];
        float new_val = alpha * total + (use_betas ? betas[v] : beta_scalar);
        x_new[v] = new_val;
        atomicAdd(d_diff, fabsf(new_val - x_old[v]));
    }
}

__global__ void spmv_mid(
    const int* __restrict__ offsets, const int* __restrict__ indices,
    const float* __restrict__ weights, const float* __restrict__ x_old,
    float* __restrict__ x_new, const float* __restrict__ betas,
    float alpha, float beta_scalar, int use_betas,
    int start_vertex, int num_verts, float* __restrict__ d_diff)
{
    int warp_id = (blockIdx.x * 8) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    float my_diff = 0.0f;
    if (warp_id < num_verts) {
        int v = warp_id + start_vertex;
        int row_start = offsets[v], row_end = offsets[v + 1];
        float sum = 0.0f;
        for (int e = row_start + lane; e < row_end; e += 32)
            sum += weights[e] * x_old[indices[e]];
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            float new_val = alpha * sum + (use_betas ? betas[v] : beta_scalar);
            x_new[v] = new_val;
            my_diff = fabsf(new_val - x_old[v]);
        }
    }
    block_reduce_add<256>(my_diff, d_diff);
}

__global__ void spmv_low(
    const int* __restrict__ offsets, const int* __restrict__ indices,
    const float* __restrict__ weights, const float* __restrict__ x_old,
    float* __restrict__ x_new, const float* __restrict__ betas,
    float alpha, float beta_scalar, int use_betas,
    int start_vertex, int num_verts, float* __restrict__ d_diff)
{
    int tid = blockIdx.x * 256 + threadIdx.x;
    float my_diff = 0.0f;
    if (tid < num_verts) {
        int v = tid + start_vertex;
        int row_start = offsets[v], row_end = offsets[v + 1];
        float sum = 0.0f;
        for (int e = row_start; e < row_end; e++)
            sum += weights[e] * x_old[indices[e]];
        float new_val = alpha * sum + (use_betas ? betas[v] : beta_scalar);
        x_new[v] = new_val;
        my_diff = fabsf(new_val - x_old[v]);
    }
    block_reduce_add<256>(my_diff, d_diff);
}





__global__ void fill_beta(float* __restrict__ x_new, float beta, int n) {
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        x_new[i] = beta;
}

__global__ void fill_betas(float* __restrict__ x_new, const float* __restrict__ betas,
                            int n, float* __restrict__ d_diff) {
    float my_diff = 0.0f;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float val = betas[i];
        x_new[i] = val;
        my_diff += fabsf(val); 
    }
    block_reduce_add<256>(my_diff, d_diff);
}






__global__ void iter2_high(
    const int* __restrict__ offsets, const float* __restrict__ weights,
    float* __restrict__ x_new, float alpha_beta, float beta,
    int start_vertex, int num_verts, float* __restrict__ d_diff)
{
    int v_idx = blockIdx.x;
    if (v_idx >= num_verts) return;
    int v = v_idx + start_vertex;
    int row_start = offsets[v], row_end = offsets[v + 1];
    float ws = 0.0f;
    for (int e = row_start + (int)threadIdx.x; e < row_end; e += 256)
        ws += weights[e];

    __shared__ float warp_s[8];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    ws = warp_reduce_sum(ws);
    if (lane == 0) warp_s[wid] = ws;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        for (int i = 0; i < 8; i++) total += warp_s[i];
        float new_val = alpha_beta * total + beta;
        x_new[v] = new_val;
        atomicAdd(d_diff, fabsf(new_val - beta));
    }
}

__global__ void iter2_mid(
    const int* __restrict__ offsets, const float* __restrict__ weights,
    float* __restrict__ x_new, float alpha_beta, float beta,
    int start_vertex, int num_verts, float* __restrict__ d_diff)
{
    int warp_id = (blockIdx.x * 8) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    float my_diff = 0.0f;
    if (warp_id < num_verts) {
        int v = warp_id + start_vertex;
        int row_start = offsets[v], row_end = offsets[v + 1];
        float ws = 0.0f;
        for (int e = row_start + lane; e < row_end; e += 32)
            ws += weights[e];
        ws = warp_reduce_sum(ws);
        if (lane == 0) {
            float new_val = alpha_beta * ws + beta;
            x_new[v] = new_val;
            my_diff = fabsf(new_val - beta);
        }
    }
    block_reduce_add<256>(my_diff, d_diff);
}

__global__ void iter2_low(
    const int* __restrict__ offsets, const float* __restrict__ weights,
    float* __restrict__ x_new, float alpha_beta, float beta,
    int start_vertex, int num_verts, float* __restrict__ d_diff)
{
    int tid = blockIdx.x * 256 + threadIdx.x;
    float my_diff = 0.0f;
    if (tid < num_verts) {
        int v = tid + start_vertex;
        int row_start = offsets[v], row_end = offsets[v + 1];
        float ws = 0.0f;
        for (int e = row_start; e < row_end; e++)
            ws += weights[e];
        float new_val = alpha_beta * ws + beta;
        x_new[v] = new_val;
        my_diff = fabsf(new_val - beta);
    }
    block_reduce_add<256>(my_diff, d_diff);
}




__global__ void l2_norm_sq_kernel(const float* __restrict__ x, int n, float* __restrict__ norm_sq) {
    float sum = 0.0f;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float v = x[i]; sum += v * v;
    }
    block_reduce_add<256>(sum, norm_sq);
}

__global__ void normalize_kernel(float* __restrict__ x, int n, const float* __restrict__ norm_sq) {
    float inv = rsqrtf(*norm_sq);
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        x[i] *= inv;
}





void launch_katz_iteration(
    const int* offsets, const int* indices, const float* weights,
    const float* x_old, float* x_new, const float* betas,
    float alpha, float beta_scalar, int use_betas,
    int seg0, int seg1, int seg2, int seg3, int seg4,
    float* d_diff)
{
    cudaMemsetAsync(d_diff, 0, sizeof(float));
    int n_high = seg1 - seg0;
    if (n_high > 0)
        spmv_high<<<n_high, 256>>>(offsets, indices, weights, x_old, x_new,
            betas, alpha, beta_scalar, use_betas, seg0, n_high, d_diff);
    int n_mid = seg2 - seg1;
    if (n_mid > 0)
        spmv_mid<<<(n_mid+7)/8, 256>>>(offsets, indices, weights, x_old, x_new,
            betas, alpha, beta_scalar, use_betas, seg1, n_mid, d_diff);
    int n_low = seg4 - seg2;
    if (n_low > 0)
        spmv_low<<<(n_low+255)/256, 256>>>(offsets, indices, weights, x_old, x_new,
            betas, alpha, beta_scalar, use_betas, seg2, n_low, d_diff);
}

void launch_iter1_scalar(float* x_new, float beta, int n) {
    int grid = (n + 255) / 256;
    if (grid > 1024) grid = 1024;
    fill_beta<<<grid, 256>>>(x_new, beta, n);
}

void launch_iter1_betas(float* x_new, const float* betas, int n, float* d_diff) {
    cudaMemsetAsync(d_diff, 0, sizeof(float));
    int grid = (n + 255) / 256;
    if (grid > 1024) grid = 1024;
    fill_betas<<<grid, 256>>>(x_new, betas, n, d_diff);
}

void launch_iter2_scalar(
    const int* offsets, const float* weights, float* x_new,
    float alpha_beta, float beta,
    int seg0, int seg1, int seg2, int seg3, int seg4,
    float* d_diff)
{
    cudaMemsetAsync(d_diff, 0, sizeof(float));
    int n_high = seg1 - seg0;
    if (n_high > 0)
        iter2_high<<<n_high, 256>>>(offsets, weights, x_new,
            alpha_beta, beta, seg0, n_high, d_diff);
    int n_mid = seg2 - seg1;
    if (n_mid > 0)
        iter2_mid<<<(n_mid+7)/8, 256>>>(offsets, weights, x_new,
            alpha_beta, beta, seg1, n_mid, d_diff);
    int n_low = seg4 - seg2;
    if (n_low > 0)
        iter2_low<<<(n_low+255)/256, 256>>>(offsets, weights, x_new,
            alpha_beta, beta, seg2, n_low, d_diff);
}

void launch_l2_normalize(float* x, int n, float* norm_sq) {
    cudaMemsetAsync(norm_sq, 0, sizeof(float));
    int grid = (n + 255) / 256;
    if (grid > 1024) grid = 1024;
    l2_norm_sq_kernel<<<grid, 256>>>(x, n, norm_sq);
    normalize_kernel<<<grid, 256>>>(x, n, norm_sq);
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

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;

    const auto& seg = graph.segment_offsets.value();
    int seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg3 = seg[3], seg4 = seg[4];

    int use_betas = (betas != nullptr) ? 1 : 0;
    float beta_scalar = beta;
    const float* betas_ptr = betas;

    cache.ensure(num_vertices);

    float* d_diff = cache.scratch;
    float* d_norm_sq = cache.scratch + 1;

    
    
    float* x[2] = {centralities, cache.buf};

    bool converged = false;
    size_t iterations = 0;
    int cur = 0;

    
    
    
    if (!has_initial_guess && max_iterations > 0) {
        

        
        if (!use_betas) {
            
            launch_iter1_scalar(x[1], beta_scalar, num_vertices);
            
            float diff1 = (float)num_vertices * fabsf(beta_scalar);
            cur = 1;
            iterations = 1;

            if (diff1 < epsilon) {
                converged = true;
            } else if (max_iterations > 1) {
                
                float alpha_beta = alpha * beta_scalar;
                launch_iter2_scalar(offsets, edge_weights, x[0],
                                    alpha_beta, beta_scalar,
                                    seg0, seg1, seg2, seg3, seg4, d_diff);
                cur = 0;
                iterations = 2;

                float h_diff;
                cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
                if (h_diff < epsilon) {
                    converged = true;
                }
            }
        } else {
            
            launch_iter1_betas(x[1], betas_ptr, num_vertices, d_diff);
            cur = 1;
            iterations = 1;

            float h_diff;
            cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_diff < epsilon) {
                converged = true;
            } else if (max_iterations > 1) {
                
                cudaMemsetAsync(d_diff, 0, sizeof(float));
                launch_katz_iteration(offsets, indices, edge_weights,
                                      x[1], x[0], betas_ptr,
                                      alpha, 0.0f, use_betas,
                                      seg0, seg1, seg2, seg3, seg4, d_diff);
                cur = 0;
                iterations = 2;

                cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
                if (h_diff < epsilon) {
                    converged = true;
                }
            }
        }
    } else {
        if (!has_initial_guess) {
            cudaMemsetAsync(centralities, 0, num_vertices * sizeof(float));
        }
        
    }

    
    
    
    if (!converged) {
        for (size_t iter = iterations; iter < max_iterations; iter++) {
            launch_katz_iteration(offsets, indices, edge_weights,
                                  x[cur], x[1 - cur], betas_ptr,
                                  alpha, beta_scalar, use_betas,
                                  seg0, seg1, seg2, seg3, seg4, d_diff);
            cur = 1 - cur;
            iterations++;

            float h_diff;
            cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_diff < epsilon) {
                converged = true;
                break;
            }
        }
    }

    
    if (normalize) {
        launch_l2_normalize(x[cur], num_vertices, d_norm_sq);
    }

    
    if (cur != 0) {
        cudaMemcpy(centralities, x[cur], (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return {iterations, converged};
}

}  
