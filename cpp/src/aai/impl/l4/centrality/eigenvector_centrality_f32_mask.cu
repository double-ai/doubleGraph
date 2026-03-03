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
#include <cstddef>

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)



__global__ void init_uniform_kernel(float* __restrict__ x, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        x[v] = 1.0f / (float)n;
    }
}

__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_counts,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int count = 0;
        for (int e = start; e < end; e++) {
            count += (edge_mask[e >> 5] >> (e & 31)) & 1;
        }
        active_counts[v] = count;
    }
}

__global__ void compact_edges_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const float* __restrict__ old_weights,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    float* __restrict__ new_weights,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int old_start = old_offsets[v];
        int old_end = old_offsets[v + 1];
        int new_start = new_offsets[v];
        int pos = new_start;

        for (int e = old_start; e < old_end; e++) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1) {
                new_indices[pos] = old_indices[e];
                new_weights[pos] = old_weights[e];
                pos++;
            }
        }
    }
}

__global__ __launch_bounds__(BLOCK_SIZE)
void add_identity_norm_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    float* __restrict__ d_norm_sq,
    int32_t num_vertices
) {
    int tid = threadIdx.x;
    int v = blockIdx.x * blockDim.x + tid;

    float sq = 0.0f;
    if (v < num_vertices) {
        float val = y[v] + x[v];
        y[v] = val;
        sq = val * val;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sq += __shfl_down_sync(0xffffffff, sq, offset);
    }

    __shared__ float s_warp[WARPS_PER_BLOCK];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) s_warp[warp_id] = sq;
    __syncthreads();

    if (warp_id == 0) {
        sq = (lane < WARPS_PER_BLOCK) ? s_warp[lane] : 0.0f;
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1) {
            sq += __shfl_down_sync(0xffffffff, sq, offset);
        }
        if (lane == 0) atomicAdd(d_norm_sq, sq);
    }
}

__global__ void normalize_kernel(
    const float* __restrict__ y,
    float* __restrict__ x_new,
    const float* __restrict__ d_norm_sq,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        float inv_norm = rsqrtf(*d_norm_sq);
        x_new[v] = y[v] * inv_norm;
    }
}

__global__ __launch_bounds__(BLOCK_SIZE)
void normalize_diff_kernel(
    const float* __restrict__ y,
    float* __restrict__ x_new,
    const float* __restrict__ x_old,
    float* __restrict__ d_diff,
    const float* __restrict__ d_norm_sq,
    int32_t num_vertices
) {
    int tid = threadIdx.x;
    int v = blockIdx.x * blockDim.x + tid;

    float inv_norm = rsqrtf(*d_norm_sq);
    float diff_val = 0.0f;

    if (v < num_vertices) {
        float val = y[v] * inv_norm;
        x_new[v] = val;
        diff_val = fabsf(val - x_old[v]);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        diff_val += __shfl_down_sync(0xffffffff, diff_val, offset);
    }

    __shared__ float s_warp[WARPS_PER_BLOCK];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) s_warp[warp_id] = diff_val;
    __syncthreads();

    if (warp_id == 0) {
        diff_val = (lane < WARPS_PER_BLOCK) ? s_warp[lane] : 0.0f;
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1) {
            diff_val += __shfl_down_sync(0xffffffff, diff_val, offset);
        }
        if (lane == 0) atomicAdd(d_diff, diff_val);
    }
}



struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;

    int32_t* new_offsets = nullptr;
    int64_t new_offsets_cap = 0;

    float* x_a = nullptr;
    int64_t x_a_cap = 0;

    float* x_b = nullptr;
    int64_t x_b_cap = 0;

    float* y_buf = nullptr;
    int64_t y_buf_cap = 0;

    float* scratch = nullptr;
    int64_t scratch_cap = 0;

    void* temp_buf = nullptr;
    size_t temp_buf_cap = 0;

    int32_t* compact_indices = nullptr;
    int64_t compact_indices_cap = 0;

    float* compact_weights = nullptr;
    int64_t compact_weights_cap = 0;

    void* spmv_buf = nullptr;
    size_t spmv_buf_cap = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cusparseSetPointerMode(cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (new_offsets) cudaFree(new_offsets);
        if (x_a) cudaFree(x_a);
        if (x_b) cudaFree(x_b);
        if (y_buf) cudaFree(y_buf);
        if (scratch) cudaFree(scratch);
        if (temp_buf) cudaFree(temp_buf);
        if (compact_indices) cudaFree(compact_indices);
        if (compact_weights) cudaFree(compact_weights);
        if (spmv_buf) cudaFree(spmv_buf);
    }
};

template <typename T>
static void ensure_buf(T*& ptr, int64_t& cap, int64_t needed) {
    if (cap < needed) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, needed * sizeof(T));
        cap = needed;
    }
}

static void ensure_buf_bytes(void*& ptr, size_t& cap, size_t needed) {
    if (cap < needed) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, needed);
        cap = needed;
    }
}

}  

eigenvector_centrality_result_t eigenvector_centrality_mask(const graph32_t& graph,
                                  const float* edge_weights,
                                  float* centralities,
                                  float epsilon,
                                  std::size_t max_iterations,
                                  const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const float* d_weights = edge_weights;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cudaStream_t stream = 0;
    cusparseSetStream(cache.cusparse_handle, stream);

    

    ensure_buf(cache.new_offsets, cache.new_offsets_cap, (int64_t)num_vertices + 1);

    int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_active_edges_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_offsets, d_edge_mask, cache.new_offsets, num_vertices);

    
    size_t prefix_sum_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, prefix_sum_temp_bytes,
                                  (int32_t*)nullptr, (int32_t*)nullptr,
                                  num_vertices + 1);
    if (prefix_sum_temp_bytes > 0) {
        ensure_buf_bytes(cache.temp_buf, cache.temp_buf_cap, prefix_sum_temp_bytes);
    }

    size_t temp_bytes = prefix_sum_temp_bytes;
    cub::DeviceScan::ExclusiveSum(cache.temp_buf, temp_bytes,
                                  cache.new_offsets, cache.new_offsets,
                                  num_vertices + 1, stream);

    
    int32_t h_total_active;
    cudaMemcpy(&h_total_active, cache.new_offsets + num_vertices,
               sizeof(int32_t), cudaMemcpyDeviceToHost);

    int64_t active_edges = (int64_t)h_total_active;
    if (active_edges == 0) active_edges = 1;  

    ensure_buf(cache.compact_indices, cache.compact_indices_cap, active_edges);
    ensure_buf(cache.compact_weights, cache.compact_weights_cap, active_edges);

    
    compact_edges_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        d_offsets, d_indices, d_weights, d_edge_mask,
        cache.new_offsets, cache.compact_indices, cache.compact_weights,
        num_vertices);

    

    ensure_buf(cache.x_a, cache.x_a_cap, (int64_t)num_vertices);
    ensure_buf(cache.x_b, cache.x_b_cap, (int64_t)num_vertices);
    ensure_buf(cache.y_buf, cache.y_buf_cap, (int64_t)num_vertices);
    ensure_buf(cache.scratch, cache.scratch_cap, (int64_t)4);

    float* d_norm_sq = cache.scratch;
    float* d_diff = cache.scratch + 1;
    float* d_alpha = cache.scratch + 2;
    float* d_beta = cache.scratch + 3;

    
    float h_init[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    cudaMemcpyAsync(cache.scratch, h_init, 4 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    float* x_ptrs[2] = { cache.x_a, cache.x_b };
    float* y = cache.y_buf;
    int cur = 0;

    
    cusparseSpMatDescr_t mat_descr;
    cusparseCreateCsr(&mat_descr,
        num_vertices, num_vertices, h_total_active,
        (void*)cache.new_offsets, (void*)cache.compact_indices, (void*)cache.compact_weights,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vec_x_descr, vec_y_descr;
    cusparseCreateDnVec(&vec_x_descr, num_vertices, x_ptrs[0], CUDA_R_32F);
    cusparseCreateDnVec(&vec_y_descr, num_vertices, y, CUDA_R_32F);

    
    size_t spmv_buffer_size = 0;
    cusparseSpMV_bufferSize(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_alpha, mat_descr, vec_x_descr,
        d_beta, vec_y_descr,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
        &spmv_buffer_size);

    if (spmv_buffer_size > 0) {
        ensure_buf_bytes(cache.spmv_buf, cache.spmv_buf_cap, spmv_buffer_size);
    }

    
    cusparseSpMV_preprocess(cache.cusparse_handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_alpha, mat_descr, vec_x_descr,
        d_beta, vec_y_descr,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
        cache.spmv_buf);

    
    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(x_ptrs[0], initial_centralities,
                        (size_t)num_vertices * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    } else {
        init_uniform_kernel<<<(num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE,
                              BLOCK_SIZE, 0, stream>>>(x_ptrs[0], num_vertices);
    }

    
    float threshold = (float)num_vertices * epsilon;
    std::size_t iterations = 0;
    bool converged = false;
    const int CHECK_INTERVAL = 2;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        cusparseDnVecSetValues(vec_x_descr, x_ptrs[cur]);

        
        cusparseSpMV(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_alpha, mat_descr, vec_x_descr,
            d_beta, vec_y_descr,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
            cache.spmv_buf);

        
        cudaMemsetAsync(d_norm_sq, 0, sizeof(float), stream);

        
        int iter_grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        add_identity_norm_kernel<<<iter_grid, BLOCK_SIZE, 0, stream>>>(
            y, x_ptrs[cur], d_norm_sq, num_vertices);

        iterations = iter + 1;
        bool is_check = (iterations % CHECK_INTERVAL == 0 || iter == max_iterations - 1);

        if (is_check) {
            cudaMemsetAsync(d_diff, 0, sizeof(float), stream);
            normalize_diff_kernel<<<iter_grid, BLOCK_SIZE, 0, stream>>>(
                y, x_ptrs[1 - cur], x_ptrs[cur], d_diff, d_norm_sq, num_vertices);
        } else {
            normalize_kernel<<<iter_grid, BLOCK_SIZE, 0, stream>>>(
                y, x_ptrs[1 - cur], d_norm_sq, num_vertices);
        }

        cur = 1 - cur;

        if (is_check) {
            float h_diff;
            cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_diff < threshold) {
                converged = true;
                break;
            }
        }
    }

    
    cusparseDestroyDnVec(vec_x_descr);
    cusparseDestroyDnVec(vec_y_descr);
    cusparseDestroySpMat(mat_descr);

    
    cudaMemcpyAsync(centralities, x_ptrs[cur],
                    (size_t)num_vertices * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    return {iterations, converged};
}

}  
