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
#include <algorithm>

namespace aai {

namespace {




__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}




__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float warp_sums[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();

    val = (lane < num_warps) ? warp_sums[lane] : 0.0f;
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}




__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    float* __restrict__ out_weight_sums,
    int32_t num_edges
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_edges;
         idx += blockDim.x * gridDim.x) {
        atomicAdd(&out_weight_sums[indices[idx]], edge_weights[idx]);
    }
}




__global__ void compute_norm_weights_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ out_weight_sums,
    float* __restrict__ norm_weights,
    int32_t num_edges
) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < num_edges;
         idx += blockDim.x * gridDim.x) {
        int32_t src = indices[idx];
        float ow = out_weight_sums[src];
        norm_weights[idx] = (ow > 0.0f) ? __fdividef(edge_weights[idx], ow) : 0.0f;
    }
}




__global__ void init_pr_kernel(float* __restrict__ pr, float init_val, int32_t N) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x) {
        pr[idx] = init_val;
    }
}




__global__ void dangling_sum_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ out_weight_sums,
    float* __restrict__ result,
    int32_t N
) {
    float thread_sum = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x) {
        if (out_weight_sums[idx] == 0.0f) {
            thread_sum += pr[idx];
        }
    }

    float block_sum = block_reduce_sum(thread_sum);
    if (threadIdx.x == 0) {
        atomicAdd(result, block_sum);
    }
}







__global__ void update_and_diff_kernel(
    const float* __restrict__ spmv_result,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    float* __restrict__ diff_out,
    const float* __restrict__ dangling_sum_ptr,
    float base_val,      
    float alpha_over_n,  
    int32_t N
) {
    
    float alpha_dang = alpha_over_n * dangling_sum_ptr[0];

    float thread_diff = 0.0f;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < N;
         idx += blockDim.x * gridDim.x) {
        float new_val = base_val + spmv_result[idx] + alpha_dang;
        thread_diff += fabsf(new_val - pr_old[idx]);
        pr_new[idx] = new_val;
    }

    float block_diff = block_reduce_sum(thread_diff);
    if (threadIdx.x == 0) {
        atomicAdd(diff_out, block_diff);
    }
}





void launch_compute_out_weight_sums(
    const int32_t* indices, const float* edge_weights, float* out_weight_sums,
    int32_t num_edges, cudaStream_t stream
) {
    if (num_edges == 0) return;
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    if (grid > 4096) grid = 4096;
    compute_out_weight_sums_kernel<<<grid, block, 0, stream>>>(
        indices, edge_weights, out_weight_sums, num_edges);
}

void launch_compute_norm_weights(
    const int32_t* indices, const float* edge_weights, const float* out_weight_sums,
    float* norm_weights, int32_t num_edges, cudaStream_t stream
) {
    if (num_edges == 0) return;
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    if (grid > 4096) grid = 4096;
    compute_norm_weights_kernel<<<grid, block, 0, stream>>>(
        indices, edge_weights, out_weight_sums, norm_weights, num_edges);
}

void launch_init_pr(float* pr, float init_val, int32_t N, cudaStream_t stream) {
    if (N == 0) return;
    int block = 256;
    int grid = (N + block - 1) / block;
    if (grid > 4096) grid = 4096;
    init_pr_kernel<<<grid, block, 0, stream>>>(pr, init_val, N);
}

void launch_dangling_sum(
    const float* pr, const float* out_weight_sums, float* result,
    int32_t N, cudaStream_t stream
) {
    cudaMemsetAsync(result, 0, sizeof(float), stream);
    if (N == 0) return;
    int block = 256;
    int grid = (N + block - 1) / block;
    if (grid > 2048) grid = 2048;
    dangling_sum_kernel<<<grid, block, 0, stream>>>(pr, out_weight_sums, result, N);
}

void launch_update_and_diff(
    const float* spmv_result, const float* pr_old, float* pr_new,
    float* diff_out, const float* dangling_sum_ptr,
    float base_val, float alpha_over_n, int32_t N, cudaStream_t stream
) {
    cudaMemsetAsync(diff_out, 0, sizeof(float), stream);
    if (N == 0) return;
    int block = 256;
    int grid = (N + block - 1) / block;
    if (grid > 2048) grid = 2048;
    update_and_diff_kernel<<<grid, block, 0, stream>>>(
        spmv_result, pr_old, pr_new, diff_out, dangling_sum_ptr,
        base_val, alpha_over_n, N);
}




struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;
    float* out_weight_sums = nullptr;
    float* norm_weights = nullptr;
    float* pr_a = nullptr;
    float* pr_b = nullptr;
    float* spmv_out = nullptr;
    float* dangling = nullptr;
    float* diff = nullptr;
    void* spmv_buffer = nullptr;

    int64_t ows_capacity = 0;
    int64_t nw_capacity = 0;
    int64_t pra_capacity = 0;
    int64_t prb_capacity = 0;
    int64_t spmv_out_capacity = 0;
    size_t spmv_buf_capacity = 0;

    Cache() {
        cusparseCreate(&handle);
        cudaMalloc(&dangling, sizeof(float));
        cudaMalloc(&diff, sizeof(float));
    }

    void ensure(int32_t num_vertices, int32_t num_edges) {
        int64_t nv = num_vertices;
        int64_t ne = num_edges > 0 ? (int64_t)num_edges : 1;

        if (ows_capacity < nv) {
            if (out_weight_sums) cudaFree(out_weight_sums);
            cudaMalloc(&out_weight_sums, nv * sizeof(float));
            ows_capacity = nv;
        }
        if (nw_capacity < ne) {
            if (norm_weights) cudaFree(norm_weights);
            cudaMalloc(&norm_weights, ne * sizeof(float));
            nw_capacity = ne;
        }
        if (pra_capacity < nv) {
            if (pr_a) cudaFree(pr_a);
            cudaMalloc(&pr_a, nv * sizeof(float));
            pra_capacity = nv;
        }
        if (prb_capacity < nv) {
            if (pr_b) cudaFree(pr_b);
            cudaMalloc(&pr_b, nv * sizeof(float));
            prb_capacity = nv;
        }
        if (spmv_out_capacity < nv) {
            if (spmv_out) cudaFree(spmv_out);
            cudaMalloc(&spmv_out, nv * sizeof(float));
            spmv_out_capacity = nv;
        }
    }

    void ensure_spmv_buffer(size_t size) {
        if (spmv_buf_capacity < size) {
            if (spmv_buffer) cudaFree(spmv_buffer);
            cudaMalloc(&spmv_buffer, size);
            spmv_buf_capacity = size;
        }
    }

    ~Cache() override {
        if (handle) cusparseDestroy(handle);
        if (out_weight_sums) cudaFree(out_weight_sums);
        if (norm_weights) cudaFree(norm_weights);
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (spmv_out) cudaFree(spmv_out);
        if (dangling) cudaFree(dangling);
        if (diff) cudaFree(diff);
        if (spmv_buffer) cudaFree(spmv_buffer);
    }
};

}  

PageRankResult pagerank(const graph32_t& graph,
                        const float* edge_weights,
                        float* pageranks,
                        const float* precomputed_vertex_out_weight_sums,
                        float alpha,
                        float epsilon,
                        std::size_t max_iterations,
                        const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    cudaStream_t stream = 0;

    cache.ensure(num_vertices, num_edges);

    
    cudaMemsetAsync(cache.out_weight_sums, 0, num_vertices * sizeof(float), stream);
    launch_compute_out_weight_sums(d_indices, edge_weights, cache.out_weight_sums, num_edges, stream);
    const float* d_out_ws = cache.out_weight_sums;

    launch_compute_norm_weights(d_indices, edge_weights, d_out_ws, cache.norm_weights, num_edges, stream);

    
    float inv_n = 1.0f / (float)num_vertices;
    float* d_pr = cache.pr_a;
    float* d_pr_new = cache.pr_b;

    if (initial_pageranks) {
        cudaMemcpyAsync(d_pr, initial_pageranks,
                        num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        launch_init_pr(d_pr, inv_n, num_vertices, stream);
    }

    
    
    
    cusparseSpMatDescr_t mat_descr = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr, vec_y = nullptr;

    float h_alpha = alpha, h_zero = 0.0f;

    if (num_edges > 0) {
        cusparseCreateCsr(
            &mat_descr,
            (int64_t)num_vertices, (int64_t)num_vertices, (int64_t)num_edges,
            (void*)d_offsets, (void*)d_indices, (void*)cache.norm_weights,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        cusparseCreateDnVec(&vec_x, (int64_t)num_vertices, d_pr, CUDA_R_32F);
        cusparseCreateDnVec(&vec_y, (int64_t)num_vertices, cache.spmv_out, CUDA_R_32F);

        
        size_t buffer_size = 0;
        cusparseSpMV_bufferSize(
            cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_alpha, mat_descr, vec_x, &h_zero, vec_y,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);

        if (buffer_size > 0) {
            cache.ensure_spmv_buffer(buffer_size);
        }

        
        cusparseSpMV_preprocess(
            cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_alpha, mat_descr, vec_x, &h_zero, vec_y,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buffer);
    }

    
    float base_val = (1.0f - alpha) * inv_n;
    float alpha_over_n = alpha * inv_n;
    float h_diff;
    std::size_t iterations = 0;
    bool converged = false;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        launch_dangling_sum(d_pr, d_out_ws, cache.dangling, num_vertices, stream);

        
        if (num_edges > 0) {
            cusparseDnVecSetValues(vec_x, d_pr);
            cusparseSpMV(
                cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &h_alpha, mat_descr, vec_x, &h_zero, vec_y,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buffer);
        } else {
            cudaMemsetAsync(cache.spmv_out, 0, num_vertices * sizeof(float), stream);
        }

        
        launch_update_and_diff(cache.spmv_out, d_pr, d_pr_new, cache.diff, cache.dangling,
                               base_val, alpha_over_n, num_vertices, stream);

        
        cudaMemcpyAsync(&h_diff, cache.diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations++;

        
        std::swap(d_pr, d_pr_new);

        if (h_diff < epsilon) {
            converged = true;
            break;
        }
    }

    
    if (vec_x) cusparseDestroyDnVec(vec_x);
    if (vec_y) cusparseDestroyDnVec(vec_y);
    if (mat_descr) cusparseDestroySpMat(mat_descr);

    
    
    cudaMemcpyAsync(pageranks, d_pr,
                    num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return PageRankResult{iterations, converged};
}

}  
