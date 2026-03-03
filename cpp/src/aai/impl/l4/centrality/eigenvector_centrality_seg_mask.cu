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
#include <climits>

namespace aai {

namespace {

namespace cg = cooperative_groups;





__global__ void __launch_bounds__(256, 6)
eigenvector_centrality_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ buf0,
    float* __restrict__ buf1,
    float* __restrict__ partials,
    float* __restrict__ results,  
    int32_t n,
    int32_t seg1,        
    int32_t seg2,        
    float threshold,
    int32_t max_iterations)
{
    cg::grid_group grid = cg::this_grid();
    extern __shared__ float smem[];

    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = gridDim.x * blockDim.x;
    const int warp_id = gtid >> 5;
    const int lane = gtid & 31;
    const int total_warps = grid_size >> 5;
    const int num_blocks = gridDim.x;

    float* buf[2] = {buf0, buf1};
    int cur = 0;

    for (int32_t iter = 0; iter < max_iterations; iter++) {
        float* __restrict__ src = buf[cur];
        float* __restrict__ dst = buf[1 - cur];

        float local_l2 = 0.0f;

        
        for (int32_t v = blockIdx.x; v < seg1; v += num_blocks) {
            int32_t row_start = offsets[v];
            int32_t row_end = offsets[v + 1];

            float sum = 0.0f;
            for (int32_t e = row_start + threadIdx.x; e < row_end; e += blockDim.x) {
                uint32_t word = edge_mask[e >> 5];
                if (word & (1u << (e & 31)))
                    sum += src[indices[e]];
            }

            smem[threadIdx.x] = sum;
            __syncthreads();
            #pragma unroll
            for (int s = 128; s > 0; s >>= 1) {
                if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
                __syncthreads();
            }

            if (threadIdx.x == 0) {
                float y_val = smem[0] + src[v];
                dst[v] = y_val;
                local_l2 += y_val * y_val;
            }
        }

        
        {
            int32_t num_mid = seg2 - seg1;
            for (int32_t w = warp_id; w < num_mid; w += total_warps) {
                int32_t v = seg1 + w;
                int32_t row_start = offsets[v];
                int32_t row_end = offsets[v + 1];

                float sum = 0.0f;
                for (int32_t e = row_start + lane; e < row_end; e += 32) {
                    uint32_t word = edge_mask[e >> 5];
                    if (word & (1u << (e & 31)))
                        sum += src[indices[e]];
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

                if (lane == 0) {
                    float y_val = sum + src[v];
                    dst[v] = y_val;
                    local_l2 += y_val * y_val;
                }
            }
        }

        
        for (int32_t v = seg2 + gtid; v < n; v += grid_size) {
            int32_t row_start = offsets[v];
            int32_t row_end = offsets[v + 1];

            float sum = src[v];
            for (int32_t e = row_start; e < row_end; e++) {
                uint32_t word = edge_mask[e >> 5];
                if (word & (1u << (e & 31)))
                    sum += src[indices[e]];
            }
            dst[v] = sum;
            local_l2 += sum * sum;
        }

        
        smem[threadIdx.x] = local_l2;
        __syncthreads();
        #pragma unroll
        for (int s = 128; s > 0; s >>= 1) {
            if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) partials[blockIdx.x] = smem[0];

        
        grid.sync();

        
        if (blockIdx.x == 0) {
            float s = 0.0f;
            for (int i = threadIdx.x; i < num_blocks; i += blockDim.x)
                s += partials[i];
            smem[threadIdx.x] = s;
            __syncthreads();
            #pragma unroll
            for (int sz = 128; sz > 0; sz >>= 1) {
                if (threadIdx.x < sz) smem[threadIdx.x] += smem[threadIdx.x + sz];
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                float norm = sqrtf(smem[0]);
                results[0] = (norm > 0.0f) ? (1.0f / norm) : 1.0f;
            }
        }

        
        grid.sync();

        
        float inv_norm = results[0];
        float local_diff = 0.0f;
        for (int32_t i = gtid; i < n; i += grid_size) {
            float yn = dst[i] * inv_norm;
            dst[i] = yn;
            local_diff += fabsf(yn - src[i]);
        }

        smem[threadIdx.x] = local_diff;
        __syncthreads();
        #pragma unroll
        for (int s = 128; s > 0; s >>= 1) {
            if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
            __syncthreads();
        }
        if (threadIdx.x == 0) partials[blockIdx.x] = smem[0];

        
        grid.sync();

        
        if (blockIdx.x == 0) {
            float s = 0.0f;
            for (int i = threadIdx.x; i < num_blocks; i += blockDim.x)
                s += partials[i];
            smem[threadIdx.x] = s;
            __syncthreads();
            #pragma unroll
            for (int sz = 128; sz > 0; sz >>= 1) {
                if (threadIdx.x < sz) smem[threadIdx.x] += smem[threadIdx.x + sz];
                __syncthreads();
            }
            if (threadIdx.x == 0) {
                if (smem[0] < threshold) {
                    results[3] = 1.0f;  
                    results[2] = (float)(iter + 1);
                }
            }
        }

        
        cur = 1 - cur;

        
        grid.sync();

        
        if (results[3] > 0.5f) break;
    }

    
    if (gtid == 0) {
        results[4] = (float)cur;
        if (results[3] < 0.5f) {
            results[2] = (float)max_iterations;
        }
    }
}

__global__ void init_uniform_kernel(float* x, int32_t n) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = 1.0f / (float)n;
}





struct Cache : Cacheable {
    float* d_partials = nullptr;
    float* d_results = nullptr;
    float* buf0 = nullptr;
    float* buf1 = nullptr;
    int max_blocks = 0;
    int64_t buf0_capacity = 0;
    int64_t buf1_capacity = 0;
    static constexpr int BLOCK_SIZE = 256;

    Cache() {
        int smem_size = BLOCK_SIZE * sizeof(float);
        int numSMs;
        cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
        int numBlocksPerSm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
            eigenvector_centrality_kernel, BLOCK_SIZE, smem_size);
        max_blocks = numBlocksPerSm * numSMs;
        if (max_blocks < 1) max_blocks = 1;

        cudaMalloc(&d_partials, max_blocks * sizeof(float));
        cudaMalloc(&d_results, 8 * sizeof(float));
    }

    void ensure_bufs(int64_t n) {
        if (buf0_capacity < n) {
            if (buf0) cudaFree(buf0);
            cudaMalloc(&buf0, n * sizeof(float));
            buf0_capacity = n;
        }
        if (buf1_capacity < n) {
            if (buf1) cudaFree(buf1);
            cudaMalloc(&buf1, n * sizeof(float));
            buf1_capacity = n;
        }
    }

    ~Cache() override {
        if (d_partials) cudaFree(d_partials);
        if (d_results) cudaFree(d_results);
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
    }
};

}  

eigenvector_centrality_result_t eigenvector_centrality_seg_mask(const graph32_t& graph,
                                      float* centralities,
                                      float epsilon,
                                      std::size_t max_iterations,
                                      const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t n = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg1 = seg[1];
    int32_t seg2 = seg[2];

    cache.ensure_bufs(n);

    
    if (initial_centralities != nullptr) {
        cudaMemcpy(cache.buf0, initial_centralities,
                   n * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        int bs = 256;
        init_uniform_kernel<<<(n + bs - 1) / bs, bs>>>(cache.buf0, n);
    }

    float threshold = (float)n * epsilon;
    int smem_size = Cache::BLOCK_SIZE * sizeof(float);

    cudaMemset(cache.d_results, 0, 8 * sizeof(float));

    int32_t max_iter_i32 = (max_iterations <= 0) ? INT32_MAX : (int32_t)max_iterations;

    void* args[] = {
        (void*)&d_offsets, (void*)&d_indices, (void*)&d_edge_mask,
        (void*)&cache.buf0, (void*)&cache.buf1,
        (void*)&cache.d_partials, (void*)&cache.d_results,
        (void*)&n, (void*)&seg1, (void*)&seg2, (void*)&threshold, (void*)&max_iter_i32
    };
    cudaLaunchCooperativeKernel(
        (void*)eigenvector_centrality_kernel,
        dim3(cache.max_blocks), dim3(Cache::BLOCK_SIZE),
        args, smem_size, 0);

    float h_results[8];
    cudaMemcpy(h_results, cache.d_results, 8 * sizeof(float), cudaMemcpyDeviceToHost);

    int32_t iterations = (int32_t)h_results[2];
    bool converged = (h_results[3] > 0.5f);
    int result_buf = (int)h_results[4];

    
    float* result_ptr = (result_buf == 0) ? cache.buf0 : cache.buf1;
    cudaMemcpy(centralities, result_ptr, n * sizeof(float), cudaMemcpyDeviceToDevice);

    return {static_cast<std::size_t>(iterations), converged};
}

}  
