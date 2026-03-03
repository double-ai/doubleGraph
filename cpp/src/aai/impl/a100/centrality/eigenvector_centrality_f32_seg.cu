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
#include <cooperative_groups.h>
#include <cstdint>
#include <algorithm>
#include <climits>

namespace aai {

namespace {

namespace cg = cooperative_groups;





__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
    val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
    return val;
}


__global__ __launch_bounds__(256, 5)
void eigenvector_centrality_coop_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ buf_x,
    float* __restrict__ buf_y,
    const int n,
    const int seg0, const int seg1, const int seg2, const int seg3, const int seg4,
    const unsigned int max_iterations,
    const float threshold,
    float* __restrict__ block_scratch,
    int* __restrict__ d_iterations,
    int* __restrict__ d_converged,
    int* __restrict__ d_result_buf
) {
    cg::grid_group grid = cg::this_grid();
    __shared__ float smem[8];

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int global_warp = blockIdx.x * 8 + warp_in_block;
    const int total_warps = gridDim.x * 8;
    const int nblocks = gridDim.x;

    const int num_high = seg1 - seg0;
    const int num_mid  = seg2 - seg1;
    const int num_low  = seg3 - seg2;
    const int num_zero = seg4 - seg3;

    float* cur = buf_x;
    float* nxt = buf_y;

    for (unsigned int iter = 0; iter < max_iterations; iter++) {
        float local_norm = 0.0f;

        
        for (int vid = blockIdx.x; vid < num_high; vid += nblocks) {
            int v = seg0 + vid;
            int start = offsets[v];
            int end = offsets[v + 1];
            float ts = 0.0f;
            for (int j = start + threadIdx.x; j < end; j += 256)
                ts += weights[j] * cur[indices[j]];
            
            ts = warp_reduce_sum(ts);
            if (lane == 0) smem[warp_in_block] = ts;
            __syncthreads();
            if (warp_in_block == 0) {
                float v2 = (threadIdx.x < 8) ? smem[threadIdx.x] : 0.0f;
                v2 = warp_reduce_sum(v2);
                if (threadIdx.x == 0) {
                    float val = v2 + cur[v];
                    nxt[v] = val;
                    local_norm += val * val;
                }
            }
            __syncthreads();
        }

        
        for (int wid = global_warp; wid < num_mid; wid += total_warps) {
            int v = seg1 + wid;
            int start = offsets[v];
            int end = offsets[v + 1];
            float ts = 0.0f;
            for (int j = start + lane; j < end; j += 32)
                ts += weights[j] * cur[indices[j]];
            ts = warp_reduce_sum(ts);
            if (lane == 0) {
                float val = ts + cur[v];
                nxt[v] = val;
                local_norm += val * val;
            }
        }

        
        for (int vid = tid; vid < num_low; vid += stride) {
            int v = seg2 + vid;
            int start = offsets[v];
            int end = offsets[v + 1];
            float s = 0.0f;
            for (int j = start; j < end; j++)
                s += weights[j] * cur[indices[j]];
            float val = s + cur[v];
            nxt[v] = val;
            local_norm += val * val;
        }

        
        for (int vid = tid; vid < num_zero; vid += stride) {
            int v = seg3 + vid;
            float val = cur[v];
            nxt[v] = val;
            local_norm += val * val;
        }

        
        local_norm = warp_reduce_sum(local_norm);
        if (lane == 0) smem[warp_in_block] = local_norm;
        __syncthreads();
        if (warp_in_block == 0) {
            float v = (threadIdx.x < 8) ? smem[threadIdx.x] : 0.0f;
            v = warp_reduce_sum(v);
            if (threadIdx.x == 0) block_scratch[blockIdx.x] = v;
        }
        __syncthreads();

        grid.sync();

        
        if (blockIdx.x == 0) {
            float total = 0.0f;
            for (int i = threadIdx.x; i < nblocks; i += 256)
                total += block_scratch[i];
            total = warp_reduce_sum(total);
            if (lane == 0) smem[warp_in_block] = total;
            __syncthreads();
            if (warp_in_block == 0) {
                float v = (threadIdx.x < 8) ? smem[threadIdx.x] : 0.0f;
                v = warp_reduce_sum(v);
                if (threadIdx.x == 0)
                    block_scratch[0] = (v > 0.0f) ? rsqrtf(v) : 0.0f;
            }
            __syncthreads();
        }

        grid.sync();

        
        float inv_norm = block_scratch[0];
        float local_diff = 0.0f;
        for (int v = tid; v < n; v += stride) {
            float old_v = cur[v];
            float new_v = nxt[v] * inv_norm;
            nxt[v] = new_v;
            local_diff += fabsf(new_v - old_v);
        }

        
        local_diff = warp_reduce_sum(local_diff);
        if (lane == 0) smem[warp_in_block] = local_diff;
        __syncthreads();
        if (warp_in_block == 0) {
            float v = (threadIdx.x < 8) ? smem[threadIdx.x] : 0.0f;
            v = warp_reduce_sum(v);
            if (threadIdx.x == 0) block_scratch[blockIdx.x] = v;
        }
        __syncthreads();

        grid.sync();

        
        if (blockIdx.x == 0) {
            float total = 0.0f;
            for (int i = threadIdx.x; i < nblocks; i += 256)
                total += block_scratch[i];
            total = warp_reduce_sum(total);
            if (lane == 0) smem[warp_in_block] = total;
            __syncthreads();
            if (warp_in_block == 0) {
                float v = (threadIdx.x < 8) ? smem[threadIdx.x] : 0.0f;
                v = warp_reduce_sum(v);
                if (threadIdx.x == 0) {
                    block_scratch[0] = (v < threshold) ? 1.0f : 0.0f;
                    if (v < threshold || iter + 1 == max_iterations) {
                        *d_iterations = (int)(iter + 1);
                        *d_converged = (v < threshold) ? 1 : 0;
                        *d_result_buf = (nxt == buf_y) ? 1 : 0;
                    }
                }
            }
            __syncthreads();
        }

        grid.sync();

        if (block_scratch[0] > 0.5f) return;

        float* tmp = cur; cur = nxt; nxt = tmp;
    }
}

__global__ __launch_bounds__(256)
void init_uniform_kernel(float* __restrict__ c, int n) {
    float v = 1.0f / (float)n;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256)
        c[i] = v;
}





static int get_max_coop_blocks() {
    int nb = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &nb, eigenvector_centrality_coop_kernel, 256, 0);
    return nb * prop.multiProcessorCount;
}





struct Cache : Cacheable {
    float* d_block_scratch = nullptr;
    int* d_result_info = nullptr;    
    float* d_buf_y = nullptr;        
    int max_coop_blocks = 0;

    size_t scratch_capacity = 0;
    int32_t buf_y_capacity = 0;

    Cache() {
        max_coop_blocks = get_max_coop_blocks();
        scratch_capacity = (size_t)max_coop_blocks * sizeof(float);
        cudaMalloc(&d_block_scratch, scratch_capacity);
        cudaMalloc(&d_result_info, 3 * sizeof(int));
    }

    void ensure(int32_t num_vertices, int num_blocks) {
        
        if (buf_y_capacity < num_vertices) {
            if (d_buf_y) cudaFree(d_buf_y);
            cudaMalloc(&d_buf_y, (size_t)num_vertices * sizeof(float));
            buf_y_capacity = num_vertices;
        }
        
        size_t needed = (size_t)num_blocks * sizeof(float);
        if (scratch_capacity < needed) {
            if (d_block_scratch) cudaFree(d_block_scratch);
            cudaMalloc(&d_block_scratch, needed);
            scratch_capacity = needed;
        }
    }

    ~Cache() override {
        if (d_block_scratch) cudaFree(d_block_scratch);
        if (d_result_info) cudaFree(d_result_info);
        if (d_buf_y) cudaFree(d_buf_y);
    }
};

}  





eigenvector_centrality_result_t eigenvector_centrality_seg(const graph32_t& graph,
                                const float* edge_weights,
                                float* centralities,
                                float epsilon,
                                std::size_t max_iterations,
                                const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;

    if (num_vertices == 0) {
        return {0, true};
    }

    float threshold = (float)num_vertices * epsilon;
    unsigned int max_iter = (max_iterations > UINT_MAX)
        ? UINT_MAX : static_cast<unsigned int>(max_iterations);
    cudaStream_t stream = 0;

    
    const auto& seg = graph.segment_offsets.value();
    int seg0 = seg[0], seg1 = seg[1], seg2 = seg[2];
    int seg3 = seg[3], seg4 = seg[4];

    
    int num_blocks = cache.max_coop_blocks;
    int min_blocks_needed = (num_vertices + 255) / 256;
    if (min_blocks_needed < num_blocks) {
        num_blocks = std::max(min_blocks_needed, 1);
    }

    
    cache.ensure(num_vertices, num_blocks);

    
    float* d_x = centralities;
    float* d_y = cache.d_buf_y;

    
    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(d_x, initial_centralities,
                       (size_t)num_vertices * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);
    } else {
        int g = (num_vertices + 255) / 256;
        if (g > 1024) g = 1024;
        init_uniform_kernel<<<g, 256, 0, stream>>>(d_x, num_vertices);
    }

    
    int h_init[3] = {0, 0, 0};
    cudaMemcpyAsync(cache.d_result_info, h_init, 3 * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    
    int* d_iterations = cache.d_result_info;
    int* d_converged = cache.d_result_info + 1;
    int* d_result_buf = cache.d_result_info + 2;
    int n = num_vertices;

    void* kernel_args[] = {
        (void*)&graph.offsets, (void*)&graph.indices, (void*)&edge_weights,
        (void*)&d_x, (void*)&d_y,
        (void*)&n,
        (void*)&seg0, (void*)&seg1, (void*)&seg2, (void*)&seg3, (void*)&seg4,
        (void*)&max_iter, (void*)&threshold,
        (void*)&cache.d_block_scratch,
        (void*)&d_iterations, (void*)&d_converged, (void*)&d_result_buf
    };

    cudaLaunchCooperativeKernel(
        (void*)eigenvector_centrality_coop_kernel,
        dim3(num_blocks), dim3(256), kernel_args, 0, stream);

    
    int h_result[3];
    cudaMemcpy(h_result, cache.d_result_info, 3 * sizeof(int),
               cudaMemcpyDeviceToHost);

    int iterations = h_result[0];
    bool converged = h_result[1] != 0;
    int result_buf = h_result[2];  

    
    if (result_buf == 1) {
        cudaMemcpyAsync(centralities, d_y,
                       (size_t)num_vertices * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    return {static_cast<std::size_t>(iterations), converged};
}

}  
