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
#include <cub/cub.cuh>
#include <cstdint>

namespace aai {

namespace {

struct Cache : Cacheable {
    
    float* out_weight = nullptr;
    float* pers_norm = nullptr;
    float* buf0 = nullptr;
    float* buf1 = nullptr;
    float* spmv_result = nullptr;
    float* x = nullptr;
    int32_t vertex_cap = 0;

    
    float* edge_vals = nullptr;
    int32_t edge_cap = 0;

    
    float* dangling = nullptr;
    float* diff = nullptr;
    float* pers_sum = nullptr;
    float* d_alpha_dev = nullptr;
    float* d_beta_dev = nullptr;
    bool scalars_allocated = false;

    
    uint8_t* cub_temp = nullptr;
    size_t cub_temp_cap = 0;

    
    uint8_t* spmv_buf = nullptr;
    size_t spmv_buf_cap = 0;

    void ensure_vertex_bufs(int32_t N) {
        if (vertex_cap < N) {
            if (out_weight) cudaFree(out_weight);
            if (pers_norm) cudaFree(pers_norm);
            if (buf0) cudaFree(buf0);
            if (buf1) cudaFree(buf1);
            if (spmv_result) cudaFree(spmv_result);
            if (x) cudaFree(x);
            cudaMalloc(&out_weight, (size_t)N * sizeof(float));
            cudaMalloc(&pers_norm, (size_t)N * sizeof(float));
            cudaMalloc(&buf0, (size_t)N * sizeof(float));
            cudaMalloc(&buf1, (size_t)N * sizeof(float));
            cudaMalloc(&spmv_result, (size_t)N * sizeof(float));
            cudaMalloc(&x, (size_t)N * sizeof(float));
            vertex_cap = N;
        }
    }

    void ensure_edge_bufs(int32_t E) {
        if (edge_cap < E) {
            if (edge_vals) cudaFree(edge_vals);
            cudaMalloc(&edge_vals, (size_t)E * sizeof(float));
            edge_cap = E;
        }
    }

    void ensure_scalars() {
        if (!scalars_allocated) {
            cudaMalloc(&dangling, sizeof(float));
            cudaMalloc(&diff, sizeof(float));
            cudaMalloc(&pers_sum, sizeof(float));
            cudaMalloc(&d_alpha_dev, sizeof(float));
            cudaMalloc(&d_beta_dev, sizeof(float));
            scalars_allocated = true;
        }
    }

    void ensure_cub_temp(size_t size) {
        if (cub_temp_cap < size) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, size);
            cub_temp_cap = size;
        }
    }

    void ensure_spmv_buf(size_t size) {
        if (spmv_buf_cap < size) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, size);
            spmv_buf_cap = size;
        }
    }

    ~Cache() override {
        if (out_weight) cudaFree(out_weight);
        if (pers_norm) cudaFree(pers_norm);
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (spmv_result) cudaFree(spmv_result);
        if (x) cudaFree(x);
        if (edge_vals) cudaFree(edge_vals);
        if (dangling) cudaFree(dangling);
        if (diff) cudaFree(diff);
        if (pers_sum) cudaFree(pers_sum);
        if (d_alpha_dev) cudaFree(d_alpha_dev);
        if (d_beta_dev) cudaFree(d_beta_dev);
        if (cub_temp) cudaFree(cub_temp);
        if (spmv_buf) cudaFree(spmv_buf);
    }
};



__global__ void compute_out_degrees_kernel(
    const int32_t* __restrict__ indices,
    float* __restrict__ out_weight,
    int32_t num_edges)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_edges;
         i += blockDim.x * gridDim.x) {
        atomicAdd(&out_weight[indices[i]], 1.0f);
    }
}

__global__ void init_pageranks_kernel(float* __restrict__ pr, int32_t N, float val) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        pr[i] = val;
    }
}

__global__ void build_pers_norm_kernel(
    const int32_t* __restrict__ pers_vertices,
    const float* __restrict__ pers_values,
    float* __restrict__ pers_norm,
    int32_t pers_size,
    float inv_pers_sum)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < pers_size;
         i += blockDim.x * gridDim.x) {
        pers_norm[pers_vertices[i]] = pers_values[i] * inv_pers_sum;
    }
}


__global__ void precompute_edge_values_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ out_weight,
    float* __restrict__ edge_values,
    int32_t num_edges)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_edges;
         i += blockDim.x * gridDim.x) {
        float ow = out_weight[indices[i]];
        edge_values[i] = (ow > 0.0f) ? (1.0f / ow) : 0.0f;
    }
}




__global__ void compute_dangling_sum_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ out_weight,
    float* __restrict__ dangling_sum,
    int32_t N)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float local_dangling = 0.0f;

    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
         tid += blockDim.x * gridDim.x) {
        if (out_weight[tid] == 0.0f) {
            local_dangling += pr[tid];
        }
    }

    float block_sum = BlockReduce(temp_storage).Sum(local_dangling);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(dangling_sum, block_sum);
    }
}


__global__ void teleport_and_diff_kernel(
    const float* __restrict__ spmv_result,
    const float* __restrict__ pers_norm,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    const float* __restrict__ dangling_sum_ptr,
    float* __restrict__ diff_out,
    float alpha,
    float one_minus_alpha,
    int32_t N)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float teleport_base = alpha * (*dangling_sum_ptr) + one_minus_alpha;
    float local_diff = 0.0f;

    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < N;
         v += blockDim.x * gridDim.x) {
        float new_val = spmv_result[v] + teleport_base * pers_norm[v];
        pr_new[v] = new_val;
        local_diff += fabsf(new_val - pr_old[v]);
    }

    float block_diff = BlockReduce(temp_storage).Sum(local_diff);
    if (threadIdx.x == 0 && block_diff != 0.0f) {
        atomicAdd(diff_out, block_diff);
    }
}


__global__ void prepare_x_and_dangling_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ out_weight,
    float* __restrict__ x,
    float* __restrict__ dangling_sum,
    int32_t N)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float local_dangling = 0.0f;

    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N;
         tid += blockDim.x * gridDim.x) {
        float ow = out_weight[tid];
        float p = pr[tid];
        if (ow > 0.0f) {
            x[tid] = p / ow;
        } else {
            x[tid] = 0.0f;
            local_dangling += p;
        }
    }

    float block_sum = BlockReduce(temp_storage).Sum(local_dangling);
    if (threadIdx.x == 0 && block_sum != 0.0f) {
        atomicAdd(dangling_sum, block_sum);
    }
}


__global__ void spmv_thread_per_vertex_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    const float* __restrict__ pers_norm,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    const float* __restrict__ dangling_sum_ptr,
    float* __restrict__ diff_out,
    float alpha,
    float one_minus_alpha,
    int32_t N)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float teleport_base = alpha * (*dangling_sum_ptr) + one_minus_alpha;
    float local_diff = 0.0f;

    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < N;
         v += blockDim.x * gridDim.x) {
        int start = offsets[v];
        int end = offsets[v + 1];

        float sum = 0.0f;
        for (int j = start; j < end; j++) {
            sum += x[indices[j]];
        }

        float new_val = alpha * sum + teleport_base * pers_norm[v];
        pr_new[v] = new_val;
        local_diff += fabsf(new_val - pr_old[v]);
    }

    float block_diff = BlockReduce(temp_storage).Sum(local_diff);
    if (threadIdx.x == 0 && block_diff != 0.0f) {
        atomicAdd(diff_out, block_diff);
    }
}


__global__ void spmv_warp_per_vertex_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    const float* __restrict__ pers_norm,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    const float* __restrict__ dangling_sum_ptr,
    float* __restrict__ diff_out,
    float alpha,
    float one_minus_alpha,
    int32_t N)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float teleport_base = alpha * (*dangling_sum_ptr) + one_minus_alpha;

    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int global_warp_id = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int total_warps = gridDim.x * warps_per_block;

    float local_diff = 0.0f;

    for (int v = global_warp_id; v < N; v += total_warps) {
        int start = offsets[v];
        int end = offsets[v + 1];

        float sum = 0.0f;
        for (int j = start + lane; j < end; j += 32) {
            sum += x[indices[j]];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            float new_val = alpha * sum + teleport_base * pers_norm[v];
            pr_new[v] = new_val;
            local_diff += fabsf(new_val - pr_old[v]);
        }
    }

    float block_diff = BlockReduce(temp_storage).Sum(local_diff);
    if (threadIdx.x == 0 && block_diff != 0.0f) {
        atomicAdd(diff_out, block_diff);
    }
}



void launch_compute_out_degrees(const int32_t* indices, float* out_weight,
                                int32_t num_edges, cudaStream_t stream) {
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    if (grid > 4096) grid = 4096;
    compute_out_degrees_kernel<<<grid, block, 0, stream>>>(indices, out_weight, num_edges);
}

void launch_init_pageranks(float* pr, int32_t N, float val, cudaStream_t stream) {
    int block = 256;
    int grid = (N + block - 1) / block;
    if (grid > 4096) grid = 4096;
    init_pageranks_kernel<<<grid, block, 0, stream>>>(pr, N, val);
}

void launch_build_pers_norm(const int32_t* pers_vertices, const float* pers_values,
                            float* pers_norm, int32_t pers_size, float inv_pers_sum,
                            cudaStream_t stream) {
    int block = 256;
    int grid = (pers_size + block - 1) / block;
    if (grid > 256) grid = 256;
    build_pers_norm_kernel<<<grid, block, 0, stream>>>(
        pers_vertices, pers_values, pers_norm, pers_size, inv_pers_sum);
}

void launch_precompute_edge_values(const int32_t* indices, const float* out_weight,
                                    float* edge_values, int32_t num_edges, cudaStream_t stream) {
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    if (grid > 4096) grid = 4096;
    precompute_edge_values_kernel<<<grid, block, 0, stream>>>(indices, out_weight, edge_values, num_edges);
}

void launch_compute_dangling_sum(const float* pr, const float* out_weight,
                                  float* dangling_sum, int32_t N, cudaStream_t stream) {
    int block = 256;
    int grid = (N + block - 1) / block;
    if (grid > 4096) grid = 4096;
    compute_dangling_sum_kernel<<<grid, block, 0, stream>>>(pr, out_weight, dangling_sum, N);
}

void launch_teleport_and_diff(const float* spmv_result, const float* pers_norm,
                               const float* pr_old, float* pr_new,
                               const float* dangling_sum_ptr, float* diff_out,
                               float alpha, float one_minus_alpha,
                               int32_t N, cudaStream_t stream) {
    int block = 256;
    int grid = (N + block - 1) / block;
    if (grid > 4096) grid = 4096;
    teleport_and_diff_kernel<<<grid, block, 0, stream>>>(
        spmv_result, pers_norm, pr_old, pr_new, dangling_sum_ptr, diff_out,
        alpha, one_minus_alpha, N);
}

void launch_prepare_x_and_dangling(const float* pr, const float* out_weight,
                                   float* x, float* dangling_sum,
                                   int32_t N, cudaStream_t stream) {
    int block = 256;
    int grid = (N + block - 1) / block;
    if (grid > 4096) grid = 4096;
    prepare_x_and_dangling_kernel<<<grid, block, 0, stream>>>(
        pr, out_weight, x, dangling_sum, N);
}

void launch_spmv_teleport_diff(const int32_t* offsets, const int32_t* indices,
                                const float* x, const float* pers_norm,
                                const float* pr_old, float* pr_new,
                                const float* dangling_sum_ptr, float* diff_out,
                                float alpha, float one_minus_alpha,
                                int32_t N, int mode, cudaStream_t stream) {
    int block = 256;
    if (mode == 0) {
        int grid = (N + block - 1) / block;
        if (grid > 65535) grid = 65535;
        spmv_thread_per_vertex_kernel<<<grid, block, 0, stream>>>(
            offsets, indices, x, pers_norm, pr_old, pr_new,
            dangling_sum_ptr, diff_out, alpha, one_minus_alpha, N);
    } else {
        int warps_per_block = block / 32;
        int grid = (N + warps_per_block - 1) / warps_per_block;
        if (grid > 65535) grid = 65535;
        spmv_warp_per_vertex_kernel<<<grid, block, 0, stream>>>(
            offsets, indices, x, pers_norm, pr_old, pr_new,
            dangling_sum_ptr, diff_out, alpha, one_minus_alpha, N);
    }
}

}  

PageRankResult personalized_pagerank(
    const graph32_t& graph,
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

    int32_t N = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    cudaStream_t stream = 0;

    
    cache.ensure_vertex_bufs(N);
    cache.ensure_scalars();

    
    const float* d_out_weight;
    if (precomputed_vertex_out_weight_sums) {
        d_out_weight = precomputed_vertex_out_weight_sums;
    } else {
        cudaMemsetAsync(cache.out_weight, 0, (size_t)N * sizeof(float), stream);
        launch_compute_out_degrees(d_indices, cache.out_weight, E, stream);
        d_out_weight = cache.out_weight;
    }

    
    size_t cub_temp_size = 0;
    {
        float* d_in_dummy = nullptr;
        float* d_out_dummy = nullptr;
        cub::DeviceReduce::Sum(nullptr, cub_temp_size, d_in_dummy, d_out_dummy,
                               static_cast<int>(personalization_size));
    }
    if (cub_temp_size == 0) cub_temp_size = 1;
    cache.ensure_cub_temp(cub_temp_size);

    cub::DeviceReduce::Sum(cache.cub_temp, cub_temp_size,
                           personalization_values, cache.pers_sum,
                           static_cast<int>(personalization_size), stream);

    float h_pers_sum;
    cudaMemcpy(&h_pers_sum, cache.pers_sum, sizeof(float), cudaMemcpyDeviceToHost);
    float inv_pers_sum = (h_pers_sum > 0.0f) ? (1.0f / h_pers_sum) : 0.0f;

    
    cudaMemsetAsync(cache.pers_norm, 0, (size_t)N * sizeof(float), stream);
    launch_build_pers_norm(personalization_vertices, personalization_values,
                           cache.pers_norm, static_cast<int32_t>(personalization_size),
                           inv_pers_sum, stream);
    const float* d_pers_norm = cache.pers_norm;

    
    float* pr_ptrs[2] = {cache.buf0, cache.buf1};
    int cur = 0;

    if (initial_pageranks) {
        cudaMemcpyAsync(pr_ptrs[cur], initial_pageranks,
                       (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        float init_val = 1.0f / static_cast<float>(N);
        launch_init_pageranks(pr_ptrs[cur], N, init_val, stream);
    }

    
    float* d_dangling = cache.dangling;
    float* d_diff = cache.diff;
    float one_minus_alpha = 1.0f - alpha;

    int avg_degree = (N > 0) ? (E / N) : 0;

    
    bool use_cusparse = (E > 500000) && (avg_degree >= 4);

    size_t iterations = 0;
    bool converged = false;

    if (use_cusparse) {
        
        cache.ensure_edge_bufs(E);

        
        launch_precompute_edge_values(d_indices, d_out_weight, cache.edge_vals, E, stream);

        
        float* d_spmv_result = cache.spmv_result;

        
        cusparseHandle_t cusparse_handle;
        cusparseCreate(&cusparse_handle);
        cusparseSetStream(cusparse_handle, stream);

        
        cudaMemcpyAsync(cache.d_alpha_dev, &alpha, sizeof(float), cudaMemcpyHostToDevice, stream);
        float zero = 0.0f;
        cudaMemcpyAsync(cache.d_beta_dev, &zero, sizeof(float), cudaMemcpyHostToDevice, stream);

        cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);

        
        cusparseSpMatDescr_t mat_descr;
        cusparseCreateCsr(
            &mat_descr, N, N, E,
            (void*)d_offsets, (void*)d_indices, (void*)cache.edge_vals,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        
        cusparseDnVecDescr_t vec_x, vec_y;
        cusparseCreateDnVec(&vec_x, N, pr_ptrs[cur], CUDA_R_32F);
        cusparseCreateDnVec(&vec_y, N, d_spmv_result, CUDA_R_32F);

        
        size_t spmv_buffer_size = 0;
        cusparseSpMV_bufferSize(
            cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha_dev, mat_descr, vec_x,
            cache.d_beta_dev, vec_y,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buffer_size);

        if (spmv_buffer_size == 0) spmv_buffer_size = 1;
        cache.ensure_spmv_buf(spmv_buffer_size);

        
        cusparseSpMV_preprocess(
            cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha_dev, mat_descr, vec_x,
            cache.d_beta_dev, vec_y,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

        
        for (size_t iter = 0; iter < max_iterations; iter++) {
            cudaMemsetAsync(d_dangling, 0, sizeof(float), stream);
            cudaMemsetAsync(d_diff, 0, sizeof(float), stream);

            
            launch_compute_dangling_sum(pr_ptrs[cur], d_out_weight, d_dangling, N, stream);

            
            cusparseDnVecSetValues(vec_x, pr_ptrs[cur]);
            cusparseSpMV(
                cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                cache.d_alpha_dev, mat_descr, vec_x,
                cache.d_beta_dev, vec_y,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);

            
            launch_teleport_and_diff(d_spmv_result, d_pers_norm,
                                      pr_ptrs[cur], pr_ptrs[1 - cur],
                                      d_dangling, d_diff,
                                      alpha, one_minus_alpha, N, stream);

            cur = 1 - cur;
            iterations++;

            float h_diff;
            cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_diff < epsilon) { converged = true; break; }
        }

        
        cusparseDestroyDnVec(vec_x);
        cusparseDestroyDnVec(vec_y);
        cusparseDestroySpMat(mat_descr);
        cusparseDestroy(cusparse_handle);

    } else {
        
        float* d_x = cache.x;
        int spmv_mode = (avg_degree >= 12) ? 1 : 0;

        for (size_t iter = 0; iter < max_iterations; iter++) {
            cudaMemsetAsync(d_dangling, 0, sizeof(float), stream);
            cudaMemsetAsync(d_diff, 0, sizeof(float), stream);
            launch_prepare_x_and_dangling(pr_ptrs[cur], d_out_weight, d_x, d_dangling, N, stream);
            launch_spmv_teleport_diff(d_offsets, d_indices, d_x, d_pers_norm,
                                       pr_ptrs[cur], pr_ptrs[1 - cur],
                                       d_dangling, d_diff, alpha, one_minus_alpha, N, spmv_mode, stream);
            cur = 1 - cur;
            iterations++;
            float h_diff;
            cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_diff < epsilon) { converged = true; break; }
        }
    }

    
    cudaMemcpy(pageranks, pr_ptrs[cur], (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice);

    return PageRankResult{iterations, converged};
}

}  
