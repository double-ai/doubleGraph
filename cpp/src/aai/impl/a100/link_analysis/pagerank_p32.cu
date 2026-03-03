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
#include <cstddef>
#include <vector>
#include <utility>

namespace aai {

namespace {





struct Cache : Cacheable {
    
    cusparseHandle_t cusparse_handle = nullptr;
    float* d_spmv_alpha = nullptr;
    float* d_spmv_beta = nullptr;

    
    int32_t* out_degree = nullptr;
    float* inv_out_weight = nullptr;
    float* pers_norm = nullptr;
    float* pr_buf = nullptr;
    int32_t* dangling_list = nullptr;
    int64_t vertex_capacity = 0;

    
    float* edge_values = nullptr;
    int64_t edge_capacity = 0;

    
    float* scalars = nullptr;
    int32_t* num_dangling_buf = nullptr;

    
    void* spmv_buffer = nullptr;
    size_t spmv_buffer_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);

        cudaMalloc(&d_spmv_alpha, sizeof(float));
        cudaMalloc(&d_spmv_beta, sizeof(float));
        float h_one = 1.0f, h_zero = 0.0f;
        cudaMemcpy(d_spmv_alpha, &h_one, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_spmv_beta, &h_zero, sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&scalars, 4 * sizeof(float));
        cudaMalloc(&num_dangling_buf, sizeof(int32_t));
    }

    ~Cache() override {
        if (d_spmv_alpha) cudaFree(d_spmv_alpha);
        if (d_spmv_beta) cudaFree(d_spmv_beta);
        if (out_degree) cudaFree(out_degree);
        if (inv_out_weight) cudaFree(inv_out_weight);
        if (pers_norm) cudaFree(pers_norm);
        if (pr_buf) cudaFree(pr_buf);
        if (dangling_list) cudaFree(dangling_list);
        if (edge_values) cudaFree(edge_values);
        if (scalars) cudaFree(scalars);
        if (num_dangling_buf) cudaFree(num_dangling_buf);
        if (spmv_buffer) cudaFree(spmv_buffer);
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
    }

    void ensure_vertex(int64_t N) {
        if (vertex_capacity < N) {
            if (out_degree) cudaFree(out_degree);
            if (inv_out_weight) cudaFree(inv_out_weight);
            if (pers_norm) cudaFree(pers_norm);
            if (pr_buf) cudaFree(pr_buf);
            if (dangling_list) cudaFree(dangling_list);

            cudaMalloc(&out_degree, N * sizeof(int32_t));
            cudaMalloc(&inv_out_weight, N * sizeof(float));
            cudaMalloc(&pers_norm, N * sizeof(float));
            cudaMalloc(&pr_buf, N * sizeof(float));
            cudaMalloc(&dangling_list, N * sizeof(int32_t));

            vertex_capacity = N;
        }
    }

    void ensure_edge(int64_t E) {
        if (edge_capacity < E) {
            if (edge_values) cudaFree(edge_values);
            cudaMalloc(&edge_values, E * sizeof(float));
            edge_capacity = E;
        }
    }

    void ensure_spmv_buffer(size_t size) {
        size_t needed = size > 0 ? size : 1;
        if (spmv_buffer_capacity < needed) {
            if (spmv_buffer) cudaFree(spmv_buffer);
            cudaMalloc(&spmv_buffer, needed);
            spmv_buffer_capacity = needed;
        }
    }
};





__global__ void compute_out_degrees_kernel(
    const int* __restrict__ indices,
    int* __restrict__ out_degree,
    int num_edges
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_edges) {
        atomicAdd(&out_degree[indices[i]], 1);
    }
}

__global__ void compute_inv_out_weight_kernel(
    const int* __restrict__ out_degree,
    float* __restrict__ inv_out_weight,
    int num_vertices
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < num_vertices) {
        int deg = out_degree[u];
        inv_out_weight[u] = (deg > 0) ? (1.0f / (float)deg) : 0.0f;
    }
}

__global__ void compute_edge_values_kernel(
    const int* __restrict__ indices,
    const float* __restrict__ inv_out_weight,
    float* __restrict__ edge_values,
    int num_edges
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_edges) {
        edge_values[i] = inv_out_weight[indices[i]];
    }
}

__global__ void init_pr_kernel(float* __restrict__ pr, int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        pr[v] = 1.0f / (float)num_vertices;
    }
}

__global__ void build_pers_norm_kernel(
    const int* __restrict__ pers_vertices,
    const float* __restrict__ pers_values,
    float* __restrict__ pers_norm,
    int pers_size, float inv_pers_sum
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pers_size) {
        pers_norm[pers_vertices[i]] = pers_values[i] * inv_pers_sum;
    }
}

__global__ void build_dangling_list_kernel(
    const int* __restrict__ out_degree,
    int* __restrict__ dangling_vertices,
    int* __restrict__ d_num_dangling,
    int num_vertices
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < num_vertices && out_degree[u] == 0) {
        int idx = atomicAdd(d_num_dangling, 1);
        dangling_vertices[idx] = u;
    }
}





__global__ void dangling_sum_compact_kernel(
    const float* __restrict__ pr,
    const int* __restrict__ dangling_vertices,
    float* __restrict__ d_dangling_sum,
    int num_dangling
) {
    extern __shared__ float sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float val = 0.0f;
    if (i < num_dangling) {
        val = pr[dangling_vertices[i]];
    }

    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);

    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) sdata[warp_id] = val;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x >> 5;
        val = (lane < num_warps) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(mask, val, offset);
        if (lane == 0 && val != 0.0f)
            atomicAdd(d_dangling_sum, val);
    }
}

__global__ void update_kernel(
    float* __restrict__ pr_new,
    const float* __restrict__ pers_norm,
    float alpha,
    const float* __restrict__ d_dangling_sum,
    int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        float teleport = alpha * (*d_dangling_sum) + (1.0f - alpha);
        pr_new[v] = alpha * pr_new[v] + teleport * pers_norm[v];
    }
}

__global__ void update_diff_kernel(
    float* __restrict__ pr_new,
    const float* __restrict__ pr_old,
    const float* __restrict__ pers_norm,
    float alpha,
    const float* __restrict__ d_dangling_sum,
    float* __restrict__ d_diff_sum,
    int num_vertices
) {
    extern __shared__ float sdata[];
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float teleport = alpha * (*d_dangling_sum) + (1.0f - alpha);

    float local_diff = 0.0f;
    if (v < num_vertices) {
        float new_val = alpha * pr_new[v] + teleport * pers_norm[v];
        local_diff = fabsf(new_val - pr_old[v]);
        pr_new[v] = new_val;
    }

    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1)
        local_diff += __shfl_down_sync(mask, local_diff, offset);

    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) sdata[warp_id] = local_diff;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = blockDim.x >> 5;
        local_diff = (lane < num_warps) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            local_diff += __shfl_down_sync(mask, local_diff, offset);
        if (lane == 0)
            atomicAdd(d_diff_sum, local_diff);
    }
}





void launch_compute_out_degrees(const int* indices, int* out_degree, int num_edges, cudaStream_t stream) {
    if (num_edges <= 0) return;
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    compute_out_degrees_kernel<<<grid, block, 0, stream>>>(indices, out_degree, num_edges);
}

void launch_compute_inv_out_weight(const int* out_degree, float* inv_out_weight, int num_vertices, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    compute_inv_out_weight_kernel<<<grid, block, 0, stream>>>(out_degree, inv_out_weight, num_vertices);
}

void launch_compute_edge_values(const int* indices, const float* inv_out_weight, float* edge_values, int num_edges, cudaStream_t stream) {
    if (num_edges <= 0) return;
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    compute_edge_values_kernel<<<grid, block, 0, stream>>>(indices, inv_out_weight, edge_values, num_edges);
}

void launch_init_pr(float* pr, int num_vertices, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    init_pr_kernel<<<grid, block, 0, stream>>>(pr, num_vertices);
}

void launch_build_pers_norm(const int* pers_vertices, const float* pers_values, float* pers_norm,
                            int pers_size, float inv_pers_sum, cudaStream_t stream) {
    if (pers_size <= 0) return;
    int block = 256;
    int grid = (pers_size + block - 1) / block;
    build_pers_norm_kernel<<<grid, block, 0, stream>>>(pers_vertices, pers_values, pers_norm, pers_size, inv_pers_sum);
}

void launch_build_dangling_list(const int* out_degree, int* dangling_vertices, int* d_num_dangling,
                                 int num_vertices, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    build_dangling_list_kernel<<<grid, block, 0, stream>>>(out_degree, dangling_vertices, d_num_dangling, num_vertices);
}

void launch_dangling_sum_compact(const float* pr, const int* dangling_vertices, float* d_dangling_sum,
                                  int num_dangling, cudaStream_t stream) {
    if (num_dangling <= 0) return;
    int block = 256;
    int grid = (num_dangling + block - 1) / block;
    int smem = (block / 32) * sizeof(float);
    dangling_sum_compact_kernel<<<grid, block, smem, stream>>>(pr, dangling_vertices, d_dangling_sum, num_dangling);
}

void launch_update(float* pr_new, const float* pers_norm, float alpha,
                    const float* d_dangling_sum, int num_vertices, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    update_kernel<<<grid, block, 0, stream>>>(pr_new, pers_norm, alpha, d_dangling_sum, num_vertices);
}

void launch_update_diff(float* pr_new, const float* pr_old, const float* pers_norm, float alpha,
                         const float* d_dangling_sum, float* d_diff_sum,
                         int num_vertices, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    int smem = (block / 32) * sizeof(float);
    update_diff_kernel<<<grid, block, smem, stream>>>(pr_new, pr_old, pers_norm, alpha, d_dangling_sum, d_diff_sum, num_vertices);
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

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    cudaStream_t stream = 0;
    cusparseSetStream(cache.cusparse_handle, stream);
    int64_t N = (int64_t)num_vertices;
    int64_t E = (int64_t)num_edges;

    cache.ensure_vertex(N);
    cache.ensure_edge(E);

    float* d_inv_out_weight = cache.inv_out_weight;
    float* d_edge_values = cache.edge_values;
    float* d_pers_norm = cache.pers_norm;
    float* d_dangling_sum = cache.scalars;
    float* d_diff_sum = cache.scalars + 1;
    float* d_pr_a = pageranks;
    float* d_pr_b = cache.pr_buf;
    int* d_dangling_vertices = cache.dangling_list;
    int* d_num_dangling = cache.num_dangling_buf;

    
    
    
    int32_t* d_out_degree = cache.out_degree;
    cudaMemsetAsync(d_out_degree, 0, N * sizeof(int32_t), stream);
    launch_compute_out_degrees(d_indices, d_out_degree, num_edges, stream);
    launch_compute_inv_out_weight(d_out_degree, d_inv_out_weight, num_vertices, stream);
    launch_compute_edge_values(d_indices, d_inv_out_weight, d_edge_values, num_edges, stream);

    
    cudaMemsetAsync(d_num_dangling, 0, sizeof(int), stream);
    launch_build_dangling_list(d_out_degree, d_dangling_vertices, d_num_dangling, num_vertices, stream);

    
    int h_num_dangling = 0;
    cudaMemcpy(&h_num_dangling, d_num_dangling, sizeof(int), cudaMemcpyDeviceToHost);

    
    cudaMemsetAsync(d_pers_norm, 0, N * sizeof(float), stream);
    std::vector<float> h_pers(personalization_size);
    cudaMemcpy(h_pers.data(), personalization_values, personalization_size * sizeof(float), cudaMemcpyDeviceToHost);
    float pers_sum = 0.0f;
    for (std::size_t i = 0; i < personalization_size; i++) pers_sum += h_pers[i];
    float inv_pers_sum = (pers_sum > 0.0f) ? (1.0f / pers_sum) : 0.0f;
    launch_build_pers_norm(personalization_vertices, personalization_values,
                           d_pers_norm, (int)personalization_size, inv_pers_sum, stream);

    
    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(d_pr_a, initial_pageranks, N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        launch_init_pr(d_pr_a, num_vertices, stream);
    }

    
    
    
    cusparseSpMatDescr_t mat_desc = nullptr;
    cusparseDnVecDescr_t x_desc = nullptr, y_desc = nullptr;

    cusparseCreateCsr(&mat_desc, N, N, E,
        (void*)d_offsets, (void*)d_indices, (void*)d_edge_values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseCreateDnVec(&x_desc, N, d_pr_a, CUDA_R_32F);
    cusparseCreateDnVec(&y_desc, N, d_pr_b, CUDA_R_32F);

    size_t buffer_size = 0;
    float h_one = 1.0f, h_zero = 0.0f;
    cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_HOST);

    cusparseSpMV_bufferSize(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_one, mat_desc, x_desc, &h_zero, y_desc,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size);

    cache.ensure_spmv_buffer(buffer_size);

    cusparseSpMV_preprocess(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &h_one, mat_desc, x_desc, &h_zero, y_desc,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buffer);

    cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);

    
    
    
    bool converged = false;
    size_t actual_iterations = max_iterations;
    float* pr_current = d_pr_a;
    float* pr_next = d_pr_b;

    size_t check_interval = 5;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        bool check_this_iter = ((iter + 1) % check_interval == 0) ||
                                (iter == max_iterations - 1);

        
        if (h_num_dangling > 0) {
            cudaMemsetAsync(d_dangling_sum, 0, sizeof(float), stream);
            launch_dangling_sum_compact(pr_current, d_dangling_vertices, d_dangling_sum,
                                        h_num_dangling, stream);
        } else {
            cudaMemsetAsync(d_dangling_sum, 0, sizeof(float), stream);
        }

        
        cusparseDnVecSetValues(x_desc, pr_current);
        cusparseDnVecSetValues(y_desc, pr_next);

        cusparseSpMV(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_spmv_alpha, mat_desc, x_desc, cache.d_spmv_beta, y_desc,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buffer);

        
        if (check_this_iter) {
            cudaMemsetAsync(d_diff_sum, 0, sizeof(float), stream);
            launch_update_diff(pr_next, pr_current, d_pers_norm, alpha,
                                d_dangling_sum, d_diff_sum, num_vertices, stream);
        } else {
            launch_update(pr_next, d_pers_norm, alpha, d_dangling_sum, num_vertices, stream);
        }

        std::swap(pr_current, pr_next);

        if (check_this_iter) {
            float h_diff;
            cudaMemcpy(&h_diff, d_diff_sum, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_diff < epsilon) {
                converged = true;
                actual_iterations = iter + 1;
                break;
            }
        }
    }

    
    cusparseDestroyDnVec(x_desc);
    cusparseDestroyDnVec(y_desc);
    cusparseDestroySpMat(mat_desc);

    
    if (pr_current != d_pr_a) {
        cudaMemcpyAsync(d_pr_a, pr_current, N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    return PageRankResult{actual_iterations, converged};
}

}  
