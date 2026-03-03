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

struct Cache : Cacheable {
    double* global_sum = nullptr;

    Cache() {
        cudaMalloc(&global_sum, sizeof(double));
    }

    ~Cache() override {
        if (global_sum) cudaFree(global_sum);
    }
};




template <int BLOCK_SIZE>
__global__ void edge_cut_unified(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ cluster,
    int32_t seg0,          
    int32_t seg1,          
    int32_t seg2,          
    int32_t seg3,          
    int32_t large_blocks,  
    int32_t medium_blocks, 
    double* __restrict__ global_sum)
{
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;

    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ union {
        typename BlockReduce::TempStorage cub_storage;
        double warp_sums[WARPS_PER_BLOCK];
    } smem;

    double block_sum = 0.0;

    if (blockIdx.x < large_blocks) {
        
        int32_t vertex = seg0 + blockIdx.x;
        int32_t c_u = cluster[vertex];
        int32_t start = offsets[vertex];
        int32_t end = offsets[vertex + 1];

        double local_sum = 0.0;
        for (int32_t e = start + threadIdx.x; e < end; e += BLOCK_SIZE) {
            int32_t v = __ldg(&indices[e]);
            float w = __ldg(&weights[e]);
            if (__ldg(&cluster[v]) != c_u) {
                local_sum += (double)w;
            }
        }

        block_sum = BlockReduce(smem.cub_storage).Sum(local_sum);

    } else if (blockIdx.x < large_blocks + medium_blocks) {
        
        int warp_in_block = threadIdx.x / 32;
        int lane = threadIdx.x % 32;

        int block_offset = blockIdx.x - large_blocks;
        int vertex_idx = block_offset * WARPS_PER_BLOCK + warp_in_block;
        int32_t num_medium = seg2 - seg1;

        double local_sum = 0.0;
        if (vertex_idx < num_medium) {
            int32_t vertex = seg1 + vertex_idx;
            int32_t c_u = cluster[vertex];
            int32_t start = offsets[vertex];
            int32_t end = offsets[vertex + 1];

            for (int32_t e = start + lane; e < end; e += 32) {
                int32_t v = __ldg(&indices[e]);
                float w = __ldg(&weights[e]);
                if (__ldg(&cluster[v]) != c_u) {
                    local_sum += (double)w;
                }
            }
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }

        
        if (lane == 0) {
            smem.warp_sums[warp_in_block] = local_sum;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            block_sum = 0.0;
            for (int i = 0; i < WARPS_PER_BLOCK; i++) {
                block_sum += smem.warp_sums[i];
            }
        } else {
            return; 
        }

    } else {
        
        int block_offset = blockIdx.x - large_blocks - medium_blocks;
        int vertex_idx = block_offset * BLOCK_SIZE + threadIdx.x;
        int32_t num_small = seg3 - seg2;

        double local_sum = 0.0;
        if (vertex_idx < num_small) {
            int32_t vertex = seg2 + vertex_idx;
            int32_t c_u = cluster[vertex];
            int32_t start = offsets[vertex];
            int32_t end = offsets[vertex + 1];

            for (int32_t e = start; e < end; e++) {
                int32_t v = __ldg(&indices[e]);
                float w = __ldg(&weights[e]);
                if (__ldg(&cluster[v]) != c_u) {
                    local_sum += (double)w;
                }
            }
        }

        block_sum = BlockReduce(smem.cub_storage).Sum(local_sum);
    }

    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(global_sum, block_sum * 0.5);
    }
}

}  

double analyze_clustering_edge_cut_seg(const graph32_t& graph,
                                       const float* edge_weights,
                                       std::size_t num_clusters,
                                       const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0];
    int32_t seg1 = seg[1];
    int32_t seg2 = seg[2];
    int32_t seg3 = seg[3];

    constexpr int BLOCK_SIZE = 256;
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;

    int num_large = seg1 - seg0;
    int num_medium = seg2 - seg1;
    int num_small = seg3 - seg2;

    int large_blocks = num_large;  
    int medium_blocks = (num_medium + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int small_blocks = (num_small + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int total_blocks = large_blocks + medium_blocks + small_blocks;

    cudaMemset(cache.global_sum, 0, sizeof(double));

    if (total_blocks > 0) {
        edge_cut_unified<BLOCK_SIZE><<<total_blocks, BLOCK_SIZE>>>(
            offsets, indices, edge_weights, cluster_assignments,
            seg0, seg1, seg2, seg3,
            large_blocks, medium_blocks,
            cache.global_sum);
    }

    double result;
    cudaMemcpy(&result, cache.global_sum, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

}  
