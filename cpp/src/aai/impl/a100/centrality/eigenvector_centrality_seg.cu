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
#include <cub/block/block_reduce.cuh>

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)

struct Cache : Cacheable {
    float* scratch = nullptr;
    float* accum = nullptr;
    int64_t scratch_capacity = 0;
    int64_t accum_capacity = 0;

    void ensure(int64_t num_vertices) {
        if (scratch_capacity < num_vertices) {
            if (scratch) cudaFree(scratch);
            cudaMalloc(&scratch, num_vertices * sizeof(float));
            scratch_capacity = num_vertices;
        }
        if (accum_capacity < 2) {
            if (accum) cudaFree(accum);
            cudaMalloc(&accum, 2 * sizeof(float));
            accum_capacity = 2;
        }
    }

    ~Cache() override {
        if (scratch) cudaFree(scratch);
        if (accum) cudaFree(accum);
    }
};

__global__ void init_uniform_kernel(float* __restrict__ x, int n, float val) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) x[i] = val;
}


__global__ void spmv_unified_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ norm_sq,
    int seg1, int seg2, int n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    float my_norm_sq = 0.0f;

    
    for (int v = blockIdx.x; v < seg1; v += gridDim.x) {
        int rs = offsets[v];
        int re = offsets[v + 1];

        float sum = 0.0f;
        for (int j = rs + threadIdx.x; j < re; j += BLOCK_SIZE) {
            sum += x[indices[j]];
        }

        float block_sum = BlockReduce(temp_storage).Sum(sum);
        

        if (threadIdx.x == 0) {
            float yv = block_sum + x[v];
            y[v] = yv;
            my_norm_sq += yv * yv;
        }
    }

    
    for (int base = seg1 + blockIdx.x * WARPS_PER_BLOCK;
         base < seg2;
         base += gridDim.x * WARPS_PER_BLOCK)
    {
        int v = base + warp_in_block;
        if (v < seg2) {
            int rs = offsets[v];
            int re = offsets[v + 1];

            float sum = 0.0f;
            for (int j = rs + lane; j < re; j += 32) {
                sum += x[indices[j]];
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }

            if (lane == 0) {
                float yv = sum + x[v];
                y[v] = yv;
                my_norm_sq += yv * yv;
            }
        }
    }

    
    for (int v = seg2 + threadIdx.x + blockIdx.x * BLOCK_SIZE;
         v < n;
         v += gridDim.x * BLOCK_SIZE)
    {
        int rs = offsets[v];
        int re = offsets[v + 1];

        float sum = x[v];
        for (int j = rs; j < re; j++) {
            sum += x[indices[j]];
        }
        y[v] = sum;
        my_norm_sq += sum * sum;
    }

    
    float block_norm = BlockReduce(temp_storage).Sum(my_norm_sq);
    if (threadIdx.x == 0 && block_norm != 0.0f) {
        atomicAdd(norm_sq, block_norm);
    }
}


__global__ void normalize_diff_kernel(
    float* __restrict__ y,
    const float* __restrict__ x_old,
    float* __restrict__ diff_out,
    const float* __restrict__ norm_sq_ptr,
    int n)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    float norm_sq_val = *norm_sq_ptr;
    float inv_norm = (norm_sq_val > 0.0f) ? rsqrtf(norm_sq_val) : 0.0f;

    float local_diff = 0.0f;

    
    int base = (threadIdx.x + blockIdx.x * BLOCK_SIZE) * 4;
    int stride = gridDim.x * BLOCK_SIZE * 4;

    for (int i = base; i + 3 < n; i += stride) {
        float4 yv = *reinterpret_cast<const float4*>(y + i);
        float4 xv = *reinterpret_cast<const float4*>(x_old + i);

        float4 nv;
        nv.x = yv.x * inv_norm;
        nv.y = yv.y * inv_norm;
        nv.z = yv.z * inv_norm;
        nv.w = yv.w * inv_norm;

        local_diff += fabsf(nv.x - xv.x) + fabsf(nv.y - xv.y) +
                      fabsf(nv.z - xv.z) + fabsf(nv.w - xv.w);

        *reinterpret_cast<float4*>(y + i) = nv;
    }

    
    int tail_start = (n / 4) * 4;
    for (int i = tail_start + threadIdx.x + blockIdx.x * BLOCK_SIZE; i < n; i += gridDim.x * BLOCK_SIZE) {
        float new_val = y[i] * inv_norm;
        local_diff += fabsf(new_val - x_old[i]);
        y[i] = new_val;
    }

    float block_diff = BlockReduce(temp_storage).Sum(local_diff);
    if (threadIdx.x == 0) {
        atomicAdd(diff_out, block_diff);
    }
}

void launch_init_uniform(float* x, int n, float val, cudaStream_t stream) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_uniform_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(x, n, val);
}

void launch_spmv_unified(
    const int* offsets, const int* indices,
    const float* x, float* y, float* norm_sq,
    int seg1, int seg2, int n, cudaStream_t stream)
{
    if (n <= 0) return;

    
    int grid = 0;

    
    int large_needs = seg1;
    int med_needs = (seg2 - seg1 + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int small_needs = (n - seg2 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    grid = large_needs;
    if (med_needs > grid) grid = med_needs;
    if (small_needs > grid) grid = small_needs;

    if (grid > 4096) grid = 4096;
    if (grid < 1) grid = 1;

    spmv_unified_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        offsets, indices, x, y, norm_sq, seg1, seg2, n);
}

void launch_normalize_diff(
    float* y, const float* x_old,
    float* diff_out, const float* norm_sq_ptr,
    int n, cudaStream_t stream)
{
    
    int elements_per_block = BLOCK_SIZE * 4;
    int grid = (n + elements_per_block - 1) / elements_per_block;
    if (grid > 1024) grid = 1024;
    if (grid < 1) grid = 1;
    normalize_diff_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        y, x_old, diff_out, norm_sq_ptr, n);
}

}  

eigenvector_centrality_result_t eigenvector_centrality_seg(const graph32_t& graph,
                                float* centralities,
                                float epsilon,
                                std::size_t max_iterations,
                                const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    if (num_vertices == 0) {
        return {0, true};
    }

    cache.ensure(num_vertices);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    
    const auto& seg = graph.segment_offsets.value();
    int seg1 = seg[1];  
    int seg2 = seg[2];  

    float* d_buf1 = centralities;
    float* d_buf2 = cache.scratch;
    float* d_norm_sq = cache.accum;
    float* d_diff = d_norm_sq + 1;

    float* d_x = d_buf1;
    float* d_y = d_buf2;

    cudaStream_t stream = 0;

    
    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(d_x, initial_centralities,
                      num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        float init_val = 1.0f / num_vertices;
        launch_init_uniform(d_x, num_vertices, init_val, stream);
    }

    float threshold = (float)num_vertices * epsilon;
    std::size_t iterations = 0;
    bool converged = false;

    const int CHECK_INTERVAL = 10;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(d_norm_sq, 0, 2 * sizeof(float), stream);

        
        launch_spmv_unified(d_offsets, d_indices, d_x, d_y, d_norm_sq,
                          seg1, seg2, num_vertices, stream);

        
        launch_normalize_diff(d_y, d_x, d_diff, d_norm_sq, num_vertices, stream);

        
        float* tmp = d_x;
        d_x = d_y;
        d_y = tmp;

        iterations = iter + 1;

        
        if ((iter + 1) % CHECK_INTERVAL == 0 || iter + 1 == max_iterations) {
            float h_diff;
            cudaMemcpyAsync(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            if (h_diff < threshold) {
                converged = true;
                break;
            }
        }
    }

    
    if (d_x != centralities) {
        cudaMemcpyAsync(centralities, d_x,
                      num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    return {iterations, converged};
}

}  
