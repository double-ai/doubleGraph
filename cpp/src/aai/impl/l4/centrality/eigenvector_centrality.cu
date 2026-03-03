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

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define MAX_GRID 4096

struct Cache : Cacheable {
    float* temp_buf = nullptr;
    float* partials = nullptr;
    float* l2_out = nullptr;
    float* diff_out = nullptr;
    unsigned int* retire = nullptr;

    int64_t temp_buf_capacity = 0;
    int64_t partials_capacity = 0;
    bool scalars_allocated = false;

    void ensure(int64_t n, int64_t grid_size) {
        if (temp_buf_capacity < n) {
            if (temp_buf) cudaFree(temp_buf);
            cudaMalloc(&temp_buf, n * sizeof(float));
            temp_buf_capacity = n;
        }
        if (partials_capacity < grid_size) {
            if (partials) cudaFree(partials);
            cudaMalloc(&partials, grid_size * sizeof(float));
            partials_capacity = grid_size;
        }
        if (!scalars_allocated) {
            cudaMalloc(&l2_out, sizeof(float));
            cudaMalloc(&diff_out, sizeof(float));
            cudaMalloc(&retire, 2 * sizeof(unsigned int));
            scalars_allocated = true;
        }
    }

    ~Cache() override {
        if (temp_buf) cudaFree(temp_buf);
        if (partials) cudaFree(partials);
        if (l2_out) cudaFree(l2_out);
        if (diff_out) cudaFree(diff_out);
        if (retire) cudaFree(retire);
    }
};


__global__ void init_centralities_kernel(float* __restrict__ x, float val, int n) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += gridDim.x * blockDim.x) {
        x[v] = val;
    }
}


__global__ __launch_bounds__(BLOCK_SIZE)
void spmv_l2norm_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ partials,
    unsigned int* __restrict__ retire_count,
    float* __restrict__ l2_out,
    int n)
{
    float thread_l2 = 0.0f;
    const int stride = gridDim.x * blockDim.x;

    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += stride) {
        int start = __ldg(&offsets[v]);
        int end = __ldg(&offsets[v + 1]);
        float sum = x[v]; 

        
        int k = start;
        for (; k + 3 < end; k += 4) {
            int i0 = __ldg(&indices[k]);
            int i1 = __ldg(&indices[k+1]);
            int i2 = __ldg(&indices[k+2]);
            int i3 = __ldg(&indices[k+3]);
            sum += x[i0] + x[i1] + x[i2] + x[i3];
        }
        for (; k < end; k++) {
            sum += x[__ldg(&indices[k])];
        }

        y[v] = sum;
        thread_l2 += sum * sum;
    }

    
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    float block_l2 = BlockReduce(temp).Sum(thread_l2);

    if (threadIdx.x == 0) {
        partials[blockIdx.x] = block_l2;
    }
    __threadfence();

    __shared__ bool am_last;
    if (threadIdx.x == 0) {
        unsigned int ticket = atomicAdd(retire_count, 1);
        am_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (am_last) {
        float sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            sum += partials[i];
        }
        sum = BlockReduce(temp).Sum(sum);
        if (threadIdx.x == 0) {
            l2_out[0] = sum;
            *retire_count = 0;
        }
    }
}


__global__ __launch_bounds__(BLOCK_SIZE)
void normalize_l1diff_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ l2_norm_sq,
    float* __restrict__ partials,
    unsigned int* __restrict__ retire_count,
    float* __restrict__ diff_out,
    int n)
{
    float l2sq = l2_norm_sq[0];
    float inv_norm = (l2sq > 0.0f) ? rsqrtf(l2sq) : 0.0f;

    float thread_diff = 0.0f;
    const int stride = gridDim.x * blockDim.x;

    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += stride) {
        float new_val = y[v] * inv_norm;
        y[v] = new_val;
        thread_diff += fabsf(new_val - x[v]);
    }

    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    float block_diff = BlockReduce(temp).Sum(thread_diff);

    if (threadIdx.x == 0) {
        partials[blockIdx.x] = block_diff;
    }
    __threadfence();

    __shared__ bool am_last;
    if (threadIdx.x == 0) {
        unsigned int ticket = atomicAdd(retire_count, 1);
        am_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (am_last) {
        float sum = 0.0f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x) {
            sum += partials[i];
        }
        sum = BlockReduce(temp).Sum(sum);
        if (threadIdx.x == 0) {
            diff_out[0] = sum;
            *retire_count = 0;
        }
    }
}

}  

eigenvector_centrality_result_t eigenvector_centrality(const graph32_t& graph,
                            float* centralities,
                            float epsilon,
                            std::size_t max_iterations,
                            const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t n = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid_size > MAX_GRID) grid_size = MAX_GRID;
    if (grid_size < 1) grid_size = 1;

    cache.ensure(n, grid_size);

    float* partials = cache.partials;
    float* l2_out = cache.l2_out;
    float* diff_out = cache.diff_out;
    unsigned int* retire1 = cache.retire;
    unsigned int* retire2 = cache.retire + 1;

    cudaMemset(cache.retire, 0, 2 * sizeof(unsigned int));

    
    if (initial_centralities != nullptr) {
        cudaMemcpy(centralities, initial_centralities,
                  n * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        int init_grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (init_grid > MAX_GRID) init_grid = MAX_GRID;
        init_centralities_kernel<<<init_grid, BLOCK_SIZE>>>(
            centralities, 1.0f / static_cast<float>(n), n);
    }

    float threshold = static_cast<float>(n) * epsilon;
    std::size_t iterations = 0;
    bool converged = false;

    float* x = centralities;
    float* y = cache.temp_buf;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        spmv_l2norm_kernel<<<grid_size, BLOCK_SIZE>>>(
            d_offsets, d_indices, x, y,
            partials, retire1, l2_out, n);

        
        normalize_l1diff_kernel<<<grid_size, BLOCK_SIZE>>>(
            y, x, l2_out,
            partials, retire2, diff_out, n);

        
        float h_diff;
        cudaMemcpy(&h_diff, diff_out, sizeof(float), cudaMemcpyDeviceToHost);

        iterations = iter + 1;

        if (h_diff < threshold) {
            converged = true;
            break;
        }

        std::swap(x, y);
    }

    
    float* result_ptr = converged ? y : x;
    if (result_ptr != centralities) {
        cudaMemcpy(centralities, result_ptr, n * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return {iterations, converged};
}

}  
