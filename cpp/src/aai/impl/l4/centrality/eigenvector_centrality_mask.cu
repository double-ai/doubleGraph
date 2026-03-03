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
#include <cub/cub.cuh>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* d_reduction = nullptr;
    int32_t* d_counts = nullptr;
    int32_t* d_new_offsets = nullptr;
    int32_t* d_new_indices = nullptr;
    void* d_temp = nullptr;
    float* d_x = nullptr;
    float* d_y = nullptr;

    int32_t counts_cap = 0;
    int32_t new_offsets_cap = 0;
    int32_t new_indices_cap = 0;
    size_t temp_cap = 0;
    int32_t x_cap = 0;
    int32_t y_cap = 0;

    Cache() {
        cudaMalloc(&d_reduction, 2 * sizeof(float));
    }

    void ensure(int32_t n_vertices, int32_t n_edges, size_t temp_size) {
        if (counts_cap < n_vertices) {
            if (d_counts) cudaFree(d_counts);
            cudaMalloc(&d_counts, n_vertices * sizeof(int32_t));
            counts_cap = n_vertices;
        }
        if (new_offsets_cap < n_vertices + 1) {
            if (d_new_offsets) cudaFree(d_new_offsets);
            cudaMalloc(&d_new_offsets, (n_vertices + 1) * sizeof(int32_t));
            new_offsets_cap = n_vertices + 1;
        }
        if (new_indices_cap < n_edges) {
            if (d_new_indices) cudaFree(d_new_indices);
            cudaMalloc(&d_new_indices, n_edges * sizeof(int32_t));
            new_indices_cap = n_edges;
        }
        if (temp_cap < temp_size) {
            if (d_temp) cudaFree(d_temp);
            cudaMalloc(&d_temp, temp_size);
            temp_cap = temp_size;
        }
        if (x_cap < n_vertices) {
            if (d_x) cudaFree(d_x);
            cudaMalloc(&d_x, n_vertices * sizeof(float));
            x_cap = n_vertices;
        }
        if (y_cap < n_vertices) {
            if (d_y) cudaFree(d_y);
            cudaMalloc(&d_y, n_vertices * sizeof(float));
            y_cap = n_vertices;
        }
    }

    ~Cache() override {
        if (d_reduction) cudaFree(d_reduction);
        if (d_counts) cudaFree(d_counts);
        if (d_new_offsets) cudaFree(d_new_offsets);
        if (d_new_indices) cudaFree(d_new_indices);
        if (d_temp) cudaFree(d_temp);
        if (d_x) cudaFree(d_x);
        if (d_y) cudaFree(d_y);
    }
};





__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_counts,
    int32_t n_vertices
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n_vertices; v += blockDim.x * gridDim.x) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int count = 0;
        for (int e = start; e < end; e++) {
            count += (edge_mask[e >> 5] >> (e & 31)) & 1u;
        }
        active_counts[v] = count;
    }
}

__global__ void compact_and_set_last_offset(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    const int32_t* __restrict__ active_counts,
    int32_t n_vertices,
    int32_t* __restrict__ d_total_active
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n_vertices; v += blockDim.x * gridDim.x) {
        int old_start = old_offsets[v];
        int old_end = old_offsets[v + 1];
        int new_start = new_offsets[v];

        int write_pos = new_start;
        for (int e = old_start; e < old_end; e++) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1u) {
                new_indices[write_pos++] = old_indices[e];
            }
        }
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *d_total_active = new_offsets[n_vertices - 1] + active_counts[n_vertices - 1];
    }
}





__global__ __launch_bounds__(256)
void spmv_l2_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ d_l2_sq,
    int32_t n_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float sq = 0.0f;

    if (v < n_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];
        float sum = x[v]; 

        for (int e = start; e < end; e++) {
            sum += x[indices[e]];
        }

        y[v] = sum;
        sq = sum * sum;
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sq += __shfl_down_sync(0xffffffff, sq, offset);

    __shared__ float smem[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    if (lane == 0) smem[warp] = sq;
    __syncthreads();

    if (warp == 0) {
        float val = (lane < 8) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) atomicAdd(d_l2_sq, val);
    }
}

__global__ __launch_bounds__(256)
void normalize_diff_kernel(
    const float* __restrict__ y,
    float* __restrict__ x,
    const float* __restrict__ d_l2_sq,
    float* __restrict__ d_l1_diff,
    int32_t n_vertices
) {
    float inv_norm = rsqrtf(*d_l2_sq);
    if (!isfinite(inv_norm)) inv_norm = 1.0f;

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0.0f;

    if (v < n_vertices) {
        float nv = y[v] * inv_norm;
        float old_val = x[v];
        diff = fabsf(nv - old_val);
        x[v] = nv;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        diff += __shfl_down_sync(0xffffffff, diff, offset);

    __shared__ float smem[8];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    if (lane == 0) smem[warp] = diff;
    __syncthreads();

    if (warp == 0) {
        float val = (lane < 8) ? smem[lane] : 0.0f;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) atomicAdd(d_l1_diff, val);
    }
}

__global__ void fill_kernel(float* arr, float val, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = val;
}

}  

eigenvector_centrality_result_t eigenvector_centrality_mask(const graph32_t& graph,
                                  float* centralities,
                                  float epsilon,
                                  std::size_t max_iterations,
                                  const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n_vertices = graph.number_of_vertices;
    int32_t n_edges = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cudaStream_t stream = 0;

    
    size_t temp_size = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_size, (int32_t*)nullptr, (int32_t*)nullptr, n_vertices);

    cache.ensure(n_vertices, n_edges, temp_size);

    float* d_l2_sq = cache.d_reduction;
    float* d_l1_diff = cache.d_reduction + 1;

    
    
    
    int block = 256;
    int grid = (n_vertices + block - 1) / block;

    count_active_edges_kernel<<<grid, block, 0, stream>>>(d_offsets, d_edge_mask, cache.d_counts, n_vertices);

    cub::DeviceScan::ExclusiveSum(cache.d_temp, temp_size, cache.d_counts, cache.d_new_offsets, n_vertices, stream);

    int32_t* d_total = cache.d_new_offsets + n_vertices;

    compact_and_set_last_offset<<<grid, block, 0, stream>>>(
        d_offsets, d_indices, d_edge_mask, cache.d_new_offsets,
        cache.d_new_indices, cache.d_counts, n_vertices, d_total);

    
    
    
    float* d_x = cache.d_x;
    float* d_y = cache.d_y;

    
    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(d_x, initial_centralities,
                       n_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        float init_val = 1.0f / (float)n_vertices;
        fill_kernel<<<(n_vertices + 255) / 256, 256, 0, stream>>>(d_x, init_val, n_vertices);
    }

    float threshold = (float)n_vertices * epsilon;
    bool converged = false;
    size_t iterations = 0;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        cudaMemsetAsync(cache.d_reduction, 0, 2 * sizeof(float), stream);

        
        spmv_l2_kernel<<<(n_vertices + 255) / 256, 256, 0, stream>>>(
            cache.d_new_offsets, cache.d_new_indices, d_x, d_y, d_l2_sq, n_vertices);

        
        normalize_diff_kernel<<<(n_vertices + 255) / 256, 256, 0, stream>>>(
            d_y, d_x, d_l2_sq, d_l1_diff, n_vertices);

        iterations = iter + 1;

        float h_l1_diff;
        cudaMemcpy(&h_l1_diff, d_l1_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_l1_diff < threshold) {
            converged = true;
            break;
        }
    }

    
    cudaMemcpy(centralities, d_x, n_vertices * sizeof(float), cudaMemcpyDeviceToDevice);

    return {iterations, converged};
}

}  
