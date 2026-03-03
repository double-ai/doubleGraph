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
#include <cusparse.h>
#include <cstdint>
#include <vector>

namespace aai {

namespace {

struct Cache : Cacheable {
    
    double* h_scalar = nullptr;
    double* d_scalars = nullptr;
    float* d_cusparse_consts = nullptr;
    cusparseHandle_t cusparse_handle = nullptr;

    
    float* d_pr_temp = nullptr;
    int32_t pr_temp_cap = 0;

    float* d_pers_norm = nullptr;
    int32_t pers_norm_cap = 0;

    int32_t* d_out_degrees = nullptr;
    int32_t out_degrees_cap = 0;

    float* d_inv_out_deg = nullptr;
    int32_t inv_out_deg_cap = 0;

    float* d_edge_values = nullptr;
    int32_t edge_values_cap = 0;

    float* d_temp = nullptr;
    int32_t temp_cap = 0;

    uint8_t* d_spmv_buffer = nullptr;
    size_t spmv_buffer_cap = 0;

    Cache() {
        cudaHostAlloc(&h_scalar, sizeof(double), cudaHostAllocDefault);
        cudaMalloc(&d_scalars, 2 * sizeof(double));
        cudaMalloc(&d_cusparse_consts, 2 * sizeof(float));
        cusparseCreate(&cusparse_handle);
        cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
    }

    ~Cache() override {
        if (h_scalar) cudaFreeHost(h_scalar);
        if (d_scalars) cudaFree(d_scalars);
        if (d_cusparse_consts) cudaFree(d_cusparse_consts);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (d_pr_temp) cudaFree(d_pr_temp);
        if (d_pers_norm) cudaFree(d_pers_norm);
        if (d_out_degrees) cudaFree(d_out_degrees);
        if (d_inv_out_deg) cudaFree(d_inv_out_deg);
        if (d_edge_values) cudaFree(d_edge_values);
        if (d_temp) cudaFree(d_temp);
        if (d_spmv_buffer) cudaFree(d_spmv_buffer);
    }

    void ensure_pr_temp(int32_t n) {
        if (pr_temp_cap < n) {
            if (d_pr_temp) cudaFree(d_pr_temp);
            cudaMalloc(&d_pr_temp, (size_t)n * sizeof(float));
            pr_temp_cap = n;
        }
    }

    void ensure_pers_norm(int32_t n) {
        if (pers_norm_cap < n) {
            if (d_pers_norm) cudaFree(d_pers_norm);
            cudaMalloc(&d_pers_norm, (size_t)n * sizeof(float));
            pers_norm_cap = n;
        }
    }

    void ensure_out_degrees(int32_t n) {
        if (out_degrees_cap < n) {
            if (d_out_degrees) cudaFree(d_out_degrees);
            cudaMalloc(&d_out_degrees, (size_t)n * sizeof(int32_t));
            out_degrees_cap = n;
        }
    }

    void ensure_inv_out_deg(int32_t n) {
        if (inv_out_deg_cap < n) {
            if (d_inv_out_deg) cudaFree(d_inv_out_deg);
            cudaMalloc(&d_inv_out_deg, (size_t)n * sizeof(float));
            inv_out_deg_cap = n;
        }
    }

    void ensure_edge_values(int32_t ne) {
        if (edge_values_cap < ne) {
            if (d_edge_values) cudaFree(d_edge_values);
            cudaMalloc(&d_edge_values, (size_t)ne * sizeof(float));
            edge_values_cap = ne;
        }
    }

    void ensure_temp(int32_t n) {
        if (temp_cap < n) {
            if (d_temp) cudaFree(d_temp);
            cudaMalloc(&d_temp, (size_t)n * sizeof(float));
            temp_cap = n;
        }
    }

    void ensure_spmv_buffer(size_t size) {
        if (spmv_buffer_cap < size) {
            if (d_spmv_buffer) cudaFree(d_spmv_buffer);
            cudaMalloc(&d_spmv_buffer, size);
            spmv_buffer_cap = size;
        }
    }
};



__global__ void compute_out_degrees_kernel(
    const int32_t* __restrict__ indices, int32_t* __restrict__ out_degrees, int32_t num_edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges) atomicAdd(&out_degrees[indices[idx]], 1);
}

__global__ void compute_inv_out_degree_kernel(
    const int32_t* __restrict__ out_degrees, float* __restrict__ inv_out_degree, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int od = out_degrees[idx];
        inv_out_degree[idx] = (od > 0) ? (1.0f / (float)od) : 0.0f;
    }
}

__global__ void build_edge_values_kernel(
    const int32_t* __restrict__ indices, const float* __restrict__ inv_out_degree,
    float* __restrict__ edge_values, int32_t num_edges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges) edge_values[idx] = inv_out_degree[indices[idx]];
}

__global__ void build_pers_norm_kernel(
    const int32_t* __restrict__ pers_vertices, const float* __restrict__ pers_values,
    float* __restrict__ pers_norm, float pers_sum_inv, int32_t pers_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pers_size) pers_norm[pers_vertices[idx]] = pers_values[idx] * pers_sum_inv;
}

__global__ void init_pr_kernel(float* __restrict__ pr, float val, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) pr[idx] = val;
}

__global__ void dangling_teleport_diff_kernel(
    float* __restrict__ pr_new,
    const float* __restrict__ pr_old,
    const float* __restrict__ inv_out_degree,
    const float* __restrict__ pers_norm,
    float alpha, float one_minus_alpha,
    double* __restrict__ d_dangling_sum,
    double* __restrict__ d_l1_diff,
    int32_t num_vertices,
    int phase
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (phase == 0) {
        double dangling = 0.0;
        if (idx < num_vertices && inv_out_degree[idx] == 0.0f) {
            dangling = (double)pr_old[idx];
        }
        unsigned mask = 0xffffffff;
        for (int offset = 16; offset > 0; offset >>= 1)
            dangling += __shfl_down_sync(mask, dangling, offset);
        int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        __shared__ double s[32];
        if (lane == 0) s[warp] = dangling;
        __syncthreads();
        int wpb = blockDim.x >> 5;
        if (warp == 0) {
            double val = (lane < wpb) ? s[lane] : 0.0;
            for (int offset = 16; offset > 0; offset >>= 1)
                val += __shfl_down_sync(mask, val, offset);
            if (lane == 0) atomicAdd(d_dangling_sum, val);
        }
    } else {
        double diff = 0.0;
        float teleport = alpha * (float)(*d_dangling_sum) + one_minus_alpha;
        if (idx < num_vertices) {
            float new_val = pr_new[idx] + teleport * pers_norm[idx];
            pr_new[idx] = new_val;
            diff = (double)fabsf(new_val - pr_old[idx]);
        }
        unsigned mask = 0xffffffff;
        for (int offset = 16; offset > 0; offset >>= 1)
            diff += __shfl_down_sync(mask, diff, offset);
        int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
        __shared__ double s[32];
        if (lane == 0) s[warp] = diff;
        __syncthreads();
        int wpb = blockDim.x >> 5;
        if (warp == 0) {
            double val = (lane < wpb) ? s[lane] : 0.0;
            for (int offset = 16; offset > 0; offset >>= 1)
                val += __shfl_down_sync(mask, val, offset);
            if (lane == 0) atomicAdd(d_l1_diff, val);
        }
    }
}

__global__ void prepare_iteration_kernel(
    const float* __restrict__ pr, const float* __restrict__ inv_out_degree,
    float* __restrict__ temp, double* __restrict__ d_dangling_sum, int32_t num_vertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double dangling = 0.0;
    if (idx < num_vertices) {
        float inv_od = inv_out_degree[idx];
        float pr_val = pr[idx];
        temp[idx] = pr_val * inv_od;
        if (inv_od == 0.0f) dangling = (double)pr_val;
    }
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1)
        dangling += __shfl_down_sync(mask, dangling, offset);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ double s[32];
    if (lane == 0) s[warp] = dangling;
    __syncthreads();
    int wpb = blockDim.x >> 5;
    if (warp == 0) {
        double val = (lane < wpb) ? s[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(mask, val, offset);
        if (lane == 0) atomicAdd(d_dangling_sum, val);
    }
}

__global__ void spmv_teleport_check_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ temp, const float* __restrict__ pr_old,
    float* __restrict__ pr_new, const float* __restrict__ pers_norm,
    float alpha, const double* __restrict__ d_dangling_sum, float one_minus_alpha,
    double* __restrict__ d_l1_diff, int32_t num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    double diff = 0.0;
    float teleport = alpha * (float)(*d_dangling_sum) + one_minus_alpha;
    if (v < num_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];
        float sum = 0.0f;
        for (int e = start; e < end; e++) sum += temp[indices[e]];
        float new_val = alpha * sum + teleport * pers_norm[v];
        pr_new[v] = new_val;
        diff = (double)fabsf(new_val - pr_old[v]);
    }
    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1)
        diff += __shfl_down_sync(mask, diff, offset);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ double s[32];
    if (lane == 0) s[warp] = diff;
    __syncthreads();
    int wpb = blockDim.x >> 5;
    if (warp == 0) {
        double val = (lane < wpb) ? s[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(mask, val, offset);
        if (lane == 0) atomicAdd(d_l1_diff, val);
    }
}

}  

PageRankResult personalized_pagerank(const graph32_t& graph,
                                     const int32_t* personalization_vertices,
                                     const float* personalization_values,
                                     std::size_t personalization_size,
                                     float* pageranks,
                                     const float* precomputed_vertex_out_weight_sums,
                                     float alpha,
                                     float epsilon,
                                     std::size_t max_iterations,
                                     const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    cudaStream_t stream = 0;

    cusparseSetStream(cache.cusparse_handle, stream);

    
    int32_t avg_degree = (num_vertices > 0) ? (num_edges / num_vertices) : 0;
    bool use_cusparse = (avg_degree >= 4) && ((int64_t)num_edges * 4 < 2000000000LL);

    
    cache.ensure_pr_temp(num_vertices);
    cache.ensure_pers_norm(num_vertices);
    cache.ensure_out_degrees(num_vertices);
    cache.ensure_inv_out_deg(num_vertices);

    float* d_pr_a = pageranks;
    float* d_pr_b = cache.d_pr_temp;
    float* d_pr_old = d_pr_a;
    float* d_pr_new = d_pr_b;
    float* d_pers_norm = cache.d_pers_norm;
    int32_t* d_out_degrees = cache.d_out_degrees;
    float* d_inv_out_deg = cache.d_inv_out_deg;
    double* d_dangling_sum = cache.d_scalars;
    double* d_l1_diff = cache.d_scalars + 1;

    
    float h_consts[2] = {alpha, 0.0f};
    cudaMemcpyAsync(cache.d_cusparse_consts, h_consts, 2 * sizeof(float), cudaMemcpyHostToDevice, stream);
    float* d_alpha = cache.d_cusparse_consts;
    float* d_beta = cache.d_cusparse_consts + 1;

    
    bool has_initial_guess = (initial_pageranks != nullptr);
    if (has_initial_guess) {
        cudaMemcpyAsync(d_pr_old, initial_pageranks,
                   (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        if (num_vertices > 0) {
            init_pr_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
                d_pr_old, 1.0f / (float)num_vertices, num_vertices);
        }
    }

    
    cudaMemsetAsync(d_out_degrees, 0, (size_t)num_vertices * sizeof(int32_t), stream);
    if (num_edges > 0) {
        compute_out_degrees_kernel<<<(num_edges + 255) / 256, 256, 0, stream>>>(
            d_indices, d_out_degrees, num_edges);
    }
    if (num_vertices > 0) {
        compute_inv_out_degree_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
            d_out_degrees, d_inv_out_deg, num_vertices);
    }

    
    cudaMemsetAsync(d_pers_norm, 0, (size_t)num_vertices * sizeof(float), stream);
    int32_t pers_size = static_cast<int32_t>(personalization_size);
    float pers_sum = 0.0f;
    {
        std::vector<float> h_pers(pers_size);
        cudaMemcpy(h_pers.data(), personalization_values,
                   pers_size * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < pers_size; i++) pers_sum += h_pers[i];
    }
    if (pers_size > 0) {
        build_pers_norm_kernel<<<(pers_size + 255) / 256, 256, 0, stream>>>(
            personalization_vertices, personalization_values, d_pers_norm,
            (pers_sum > 0.0f) ? (1.0f / pers_sum) : 0.0f, pers_size);
    }

    float one_minus_alpha = 1.0f - alpha;
    bool converged = false;
    size_t iterations = 0;

    if (use_cusparse) {
        
        cache.ensure_edge_values(num_edges);
        float* d_edge_values = cache.d_edge_values;
        if (num_edges > 0) {
            build_edge_values_kernel<<<(num_edges + 255) / 256, 256, 0, stream>>>(
                d_indices, d_inv_out_deg, d_edge_values, num_edges);
        }

        
        cusparseSpMatDescr_t matDescr = nullptr;
        cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
        cusparseCreateCsr(&matDescr, num_vertices, num_vertices, num_edges,
                          (void*)d_offsets, (void*)d_indices, (void*)d_edge_values,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cusparseCreateDnVec(&vecX, num_vertices, d_pr_old, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, num_vertices, d_pr_new, CUDA_R_32F);

        
        size_t bufferSize = 0;
        float h_one = 1.0f, h_zero = 0.0f;
        cusparseSpMV_bufferSize(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_one, matDescr, vecX, &h_zero, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

        void* d_buffer = nullptr;
        if (bufferSize > 0) {
            cache.ensure_spmv_buffer(bufferSize);
            d_buffer = cache.d_spmv_buffer;
        }

        cusparseSpMV_preprocess(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_one, matDescr, vecX, &h_zero, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);

        for (size_t iter = 0; iter < max_iterations; iter++) {
            cudaMemsetAsync(cache.d_scalars, 0, 2 * sizeof(double), stream);

            
            if (num_vertices > 0) {
                dangling_teleport_diff_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
                    d_pr_new, d_pr_old, d_inv_out_deg, d_pers_norm,
                    alpha, one_minus_alpha, d_dangling_sum, d_l1_diff, num_vertices, 0);
            }

            
            cusparseDnVecSetValues(vecX, d_pr_old);
            cusparseDnVecSetValues(vecY, d_pr_new);
            cusparseSpMV(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                d_alpha, matDescr, vecX, d_beta, vecY,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_buffer);

            
            if (num_vertices > 0) {
                dangling_teleport_diff_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
                    d_pr_new, d_pr_old, d_inv_out_deg, d_pers_norm,
                    alpha, one_minus_alpha, d_dangling_sum, d_l1_diff, num_vertices, 1);
            }

            cudaMemcpyAsync(cache.h_scalar, d_l1_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            iterations = iter + 1;
            if ((float)*cache.h_scalar < epsilon) { converged = true; break; }
            float* tmp = d_pr_old; d_pr_old = d_pr_new; d_pr_new = tmp;
        }

        cusparseDestroySpMat(matDescr);
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
    } else {
        
        cache.ensure_temp(num_vertices);
        float* d_temp = cache.d_temp;

        for (size_t iter = 0; iter < max_iterations; iter++) {
            cudaMemsetAsync(cache.d_scalars, 0, 2 * sizeof(double), stream);
            if (num_vertices > 0) {
                prepare_iteration_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
                    d_pr_old, d_inv_out_deg, d_temp, d_dangling_sum, num_vertices);
                spmv_teleport_check_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
                    d_offsets, d_indices, d_temp, d_pr_old, d_pr_new,
                    d_pers_norm, alpha, d_dangling_sum, one_minus_alpha, d_l1_diff, num_vertices);
            }

            cudaMemcpyAsync(cache.h_scalar, d_l1_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            iterations = iter + 1;
            if ((float)*cache.h_scalar < epsilon) { converged = true; break; }
            float* tmp = d_pr_old; d_pr_old = d_pr_new; d_pr_new = tmp;
        }
    }

    
    float* result_ptr = converged ? d_pr_new : d_pr_old;
    if (result_ptr != pageranks) {
        cudaMemcpy(pageranks, result_ptr, (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return PageRankResult{iterations, converged};
}

}  
