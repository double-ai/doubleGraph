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
    float* d_scratch = nullptr;    
    float* h_diff = nullptr;       
    float* h_norm = nullptr;       
    float* temp_buf = nullptr;     
    int64_t temp_capacity = 0;

    Cache() {
        cudaMalloc(&d_scratch, 2 * sizeof(float));
        cudaMallocHost(&h_diff, sizeof(float));
        cudaMallocHost(&h_norm, sizeof(float));
    }

    void ensure(int64_t n) {
        if (temp_capacity < n) {
            if (temp_buf) cudaFree(temp_buf);
            cudaMalloc(&temp_buf, n * sizeof(float));
            temp_capacity = n;
        }
    }

    ~Cache() override {
        if (d_scratch) cudaFree(d_scratch);
        if (h_diff) cudaFreeHost(h_diff);
        if (h_norm) cudaFreeHost(h_norm);
        if (temp_buf) cudaFree(temp_buf);
    }
};




__device__ __forceinline__ void block_reduce_diff(float local_diff, float* __restrict__ diff_sum) {
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_diff += __shfl_down_sync(mask, local_diff, offset);
    }

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    __shared__ float warp_sums[8];

    if (lane == 0) warp_sums[warp_id] = local_diff;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (lane == 0) atomicAdd(diff_sum, val);
    }
}




__global__ __launch_bounds__(256)
void katz_spmv_thread_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float alpha,
    const float beta,
    const float* __restrict__ betas,
    float* __restrict__ diff_sum,
    const int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;

    float local_diff = 0.0f;

    if (v < num_vertices) {
        int start = __ldg(&offsets[v]);
        int end = __ldg(&offsets[v + 1]);

        float sum = 0.0f;
        for (int j = start; j < end; j++) {
            sum += x_old[__ldg(&indices[j])];
        }

        float beta_v = betas ? __ldg(&betas[v]) : beta;
        float new_val = alpha * sum + beta_v;
        x_new[v] = new_val;
        local_diff = fabsf(new_val - x_old[v]);
    }

    block_reduce_diff(local_diff, diff_sum);
}




__global__ __launch_bounds__(256)
void katz_spmv_warp_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float alpha,
    const float beta,
    const float* __restrict__ betas,
    float* __restrict__ diff_sum,
    const int num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & 31;
    int v = tid >> 5;

    float local_diff = 0.0f;

    if (v < num_vertices) {
        int start, end;
        if (lane == 0) {
            start = offsets[v];
            end = offsets[v + 1];
        }
        start = __shfl_sync(0xffffffff, start, 0);
        end = __shfl_sync(0xffffffff, end, 0);

        float sum = 0.0f;
        for (int j = start + lane; j < end; j += 32) {
            sum += x_old[indices[j]];
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            float beta_v = betas ? betas[v] : beta;
            float new_val = alpha * sum + beta_v;
            x_new[v] = new_val;
            local_diff = fabsf(new_val - x_old[v]);
        }
    }

    block_reduce_diff(local_diff, diff_sum);
}




__global__ __launch_bounds__(256)
void katz_init_skip2_kernel(
    const int* __restrict__ offsets,
    float* __restrict__ x,
    const float alpha,
    const float beta,
    const int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int degree = __ldg(&offsets[v + 1]) - __ldg(&offsets[v]);
        x[v] = alpha * beta * (float)degree + beta;
    }
}




__global__ __launch_bounds__(256)
void katz_fill_kernel(float* __restrict__ x, const float val, const int N) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N) x[v] = val;
}




__global__ __launch_bounds__(256)
void katz_l2_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ norm_sum,
    const int N
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;
    if (v < N) {
        float val = x[v];
        local_sum = val * val;
    }

    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(mask, local_sum, offset);

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    __shared__ float ws[8];
    if (lane == 0) ws[warp_id] = local_sum;
    __syncthreads();
    if (warp_id == 0) {
        float val = (lane < 8) ? ws[lane] : 0.0f;
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(mask, val, offset);
        if (lane == 0) atomicAdd(norm_sum, val);
    }
}




__global__ __launch_bounds__(256)
void katz_scale_kernel(float* __restrict__ x, const float scale, const int N) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N) x[v] *= scale;
}




__global__ __launch_bounds__(256)
void katz_spmv_thread_nodiff_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float alpha,
    const float beta,
    const float* __restrict__ betas,
    const int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];
        float sum = 0.0f;
        for (int j = start; j < end; j++)
            sum += x_old[__ldg(&indices[j])];
        float beta_v = betas ? __ldg(&betas[v]) : beta;
        x_new[v] = alpha * sum + beta_v;
    }
}

__global__ __launch_bounds__(256)
void katz_spmv_warp_nodiff_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float alpha,
    const float beta,
    const float* __restrict__ betas,
    const int num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & 31;
    int v = tid >> 5;
    if (v < num_vertices) {
        int start, end;
        if (lane == 0) { start = offsets[v]; end = offsets[v + 1]; }
        start = __shfl_sync(0xffffffff, start, 0);
        end = __shfl_sync(0xffffffff, end, 0);
        float sum = 0.0f;
        for (int j = start + lane; j < end; j += 32)
            sum += x_old[indices[j]];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) {
            float beta_v = betas ? betas[v] : beta;
            x_new[v] = alpha * sum + beta_v;
        }
    }
}





void launch_katz_spmv_thread(
    const int* offsets, const int* indices,
    const float* x_old, float* x_new,
    float alpha, float beta, const float* betas,
    float* diff_sum, int N, cudaStream_t stream
) {
    if (N == 0) return;
    int grid = (N + 255) / 256;
    katz_spmv_thread_kernel<<<grid, 256, 0, stream>>>(
        offsets, indices, x_old, x_new, alpha, beta, betas, diff_sum, N);
}

void launch_katz_spmv_warp(
    const int* offsets, const int* indices,
    const float* x_old, float* x_new,
    float alpha, float beta, const float* betas,
    float* diff_sum, int N, cudaStream_t stream
) {
    if (N == 0) return;
    int64_t total_threads = (int64_t)N * 32;
    int grid = (int)((total_threads + 255) / 256);
    katz_spmv_warp_kernel<<<grid, 256, 0, stream>>>(
        offsets, indices, x_old, x_new, alpha, beta, betas, diff_sum, N);
}

void launch_katz_init_skip2(
    const int* offsets, float* x,
    float alpha, float beta, int N, cudaStream_t stream
) {
    if (N == 0) return;
    katz_init_skip2_kernel<<<(N+255)/256, 256, 0, stream>>>(offsets, x, alpha, beta, N);
}

void launch_katz_fill(float* x, float val, int N, cudaStream_t stream) {
    if (N == 0) return;
    katz_fill_kernel<<<(N+255)/256, 256, 0, stream>>>(x, val, N);
}

void launch_katz_l2_norm(const float* x, float* norm_sum, int N, cudaStream_t stream) {
    if (N == 0) return;
    katz_l2_norm_kernel<<<(N+255)/256, 256, 0, stream>>>(x, norm_sum, N);
}

void launch_katz_scale(float* x, float scale, int N, cudaStream_t stream) {
    if (N == 0) return;
    katz_scale_kernel<<<(N+255)/256, 256, 0, stream>>>(x, scale, N);
}

void launch_katz_spmv_thread_nodiff(
    const int* offsets, const int* indices,
    const float* x_old, float* x_new,
    float alpha, float beta, const float* betas,
    int N, cudaStream_t stream
) {
    if (N == 0) return;
    katz_spmv_thread_nodiff_kernel<<<(N+255)/256, 256, 0, stream>>>(
        offsets, indices, x_old, x_new, alpha, beta, betas, N);
}

void launch_katz_spmv_warp_nodiff(
    const int* offsets, const int* indices,
    const float* x_old, float* x_new,
    float alpha, float beta, const float* betas,
    int N, cudaStream_t stream
) {
    if (N == 0) return;
    int64_t total_threads = (int64_t)N * 32;
    katz_spmv_warp_nodiff_kernel<<<(int)((total_threads+255)/256), 256, 0, stream>>>(
        offsets, indices, x_old, x_new, alpha, beta, betas, N);
}




typedef void (*spmv_fn_t)(const int*, const int*, const float*, float*,
    float, float, const float*, float*, int, cudaStream_t);
typedef void (*spmv_nodiff_fn_t)(const int*, const int*, const float*, float*,
    float, float, const float*, int, cudaStream_t);

}  




katz_centrality_result_t katz_centrality(const graph32_t& graph,
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
    int32_t E = graph.number_of_edges;
    const int* d_offsets = graph.offsets;
    const int* d_indices = graph.indices;
    cudaStream_t stream = nullptr;

    bool use_betas = (betas != nullptr);
    float beta_scalar = beta;
    const float* d_betas = betas;

    float* d_diff_sum = cache.d_scratch;
    float* d_norm_sum = cache.d_scratch + 1;

    std::size_t num_iterations = 0;
    bool converged = false;

    if (N == 0) {
        return {num_iterations, converged};
    }

    cache.ensure(N);

    float* buf_a = centralities;
    float* buf_b = cache.temp_buf;
    float* x_cur = buf_a;
    float* x_next = buf_b;

    float avg_degree = (float)E / (float)N;
    spmv_fn_t spmv_fn = (avg_degree >= 8.0f) ? launch_katz_spmv_warp : launch_katz_spmv_thread;
    spmv_nodiff_fn_t spmv_nodiff_fn = (avg_degree >= 8.0f) ? launch_katz_spmv_warp_nodiff : launch_katz_spmv_thread_nodiff;

    
    
    
    if (has_initial_guess) {
        
    } else if (!use_betas) {
        
        if (max_iterations >= 2) {
            float diff1 = fabsf(beta_scalar) * (float)N;
            if (diff1 < epsilon) {
                launch_katz_fill(x_cur, beta_scalar, N, stream);
                num_iterations = 1;
                converged = true;
            } else {
                launch_katz_init_skip2(d_offsets, x_cur, alpha, beta_scalar, N, stream);
                num_iterations = 2;
                float diff2 = alpha * fabsf(beta_scalar) * (float)E;
                if (diff2 < epsilon) {
                    converged = true;
                } else {
                    
                    float cf = alpha * avg_degree;
                    if (cf > 0.0f && cf < 1.0f) {
                        float log_ratio = logf(epsilon / diff2);
                        float log_cf = logf(cf);
                        int n_needed = (int)ceilf(log_ratio / log_cf);
                        int n_skip = (n_needed >= 2) ? n_needed - 2 : 0;
                        if (n_skip > 20) n_skip = 20;
                        for (int i = 0; i < n_skip && num_iterations < max_iterations; i++) {
                            spmv_nodiff_fn(d_offsets, d_indices, x_cur, x_next,
                                alpha, beta_scalar, d_betas, N, stream);
                            ++num_iterations;
                            float* tmp = x_cur; x_cur = x_next; x_next = tmp;
                        }
                    }
                }
            }
        } else if (max_iterations == 1) {
            launch_katz_fill(x_cur, beta_scalar, N, stream);
            num_iterations = 1;
            converged = (fabsf(beta_scalar) * (float)N < epsilon);
        } else {
            cudaMemsetAsync(x_cur, 0, (size_t)N * sizeof(float), stream);
        }
    } else {
        
        if (max_iterations >= 2) {
            cudaMemsetAsync(d_diff_sum, 0, sizeof(float), stream);
            spmv_fn(d_offsets, d_indices, d_betas, x_cur,
                alpha, beta_scalar, d_betas, d_diff_sum, N, stream);
            cudaMemcpyAsync(cache.h_diff, d_diff_sum, sizeof(float),
                cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            num_iterations = 2;
            if (*cache.h_diff < epsilon) {
                converged = true;
            } else {
                
                float cf = alpha * avg_degree;
                if (cf > 0.0f && cf < 1.0f) {
                    float log_ratio = logf(epsilon / *cache.h_diff);
                    float log_cf = logf(cf);
                    int n_needed = (int)ceilf(log_ratio / log_cf);
                    int n_skip = (n_needed >= 2) ? n_needed - 2 : 0;
                    if (n_skip > 20) n_skip = 20;
                    for (int i = 0; i < n_skip && num_iterations < max_iterations; i++) {
                        spmv_nodiff_fn(d_offsets, d_indices, x_cur, x_next,
                            alpha, beta_scalar, d_betas, N, stream);
                        ++num_iterations;
                        float* tmp = x_cur; x_cur = x_next; x_next = tmp;
                    }
                }
            }
        } else if (max_iterations == 1) {
            cudaMemcpyAsync(x_cur, d_betas, (size_t)N * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
            num_iterations = 1;
        } else {
            cudaMemsetAsync(x_cur, 0, (size_t)N * sizeof(float), stream);
        }
    }

    
    
    
    while (!converged && num_iterations < max_iterations) {
        cudaMemsetAsync(d_diff_sum, 0, sizeof(float), stream);
        spmv_fn(d_offsets, d_indices, x_cur, x_next,
            alpha, beta_scalar, d_betas, d_diff_sum, N, stream);
        cudaMemcpyAsync(cache.h_diff, d_diff_sum, sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        ++num_iterations;
        if (*cache.h_diff < epsilon) converged = true;
        float* tmp = x_cur; x_cur = x_next; x_next = tmp;
    }

    
    
    
    if (normalize && N > 0) {
        cudaMemsetAsync(d_norm_sum, 0, sizeof(float), stream);
        launch_katz_l2_norm(x_cur, d_norm_sum, N, stream);
        cudaMemcpyAsync(cache.h_norm, d_norm_sum, sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        float l2 = sqrtf(*cache.h_norm);
        if (l2 > 0.0f) launch_katz_scale(x_cur, 1.0f / l2, N, stream);
    }

    if (x_cur != buf_a) {
        cudaMemcpyAsync(buf_a, x_cur, (size_t)N * sizeof(float),
            cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    return {num_iterations, converged};
}

}  
