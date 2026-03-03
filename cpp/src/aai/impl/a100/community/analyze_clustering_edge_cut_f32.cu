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

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)

struct Cache : Cacheable {
    double* d_result = nullptr;
    uint8_t* clusters_narrow = nullptr;
    int64_t clusters_narrow_capacity = 0;

    Cache() {
        cudaMalloc(&d_result, sizeof(double));
    }

    ~Cache() override {
        if (d_result) cudaFree(d_result);
        if (clusters_narrow) cudaFree(clusters_narrow);
    }

    void ensure_clusters(int64_t bytes) {
        if (clusters_narrow_capacity < bytes) {
            if (clusters_narrow) cudaFree(clusters_narrow);
            cudaMalloc(&clusters_narrow, bytes);
            clusters_narrow_capacity = bytes;
        }
    }
};


__global__ void convert_clusters_u8_kernel(
    const int32_t* __restrict__ in,
    uint8_t* __restrict__ out,
    int32_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < (n >> 2); i += stride) {
        int base = i << 2;
        int32_t a = in[base], b = in[base+1], c = in[base+2], d = in[base+3];
        uint32_t packed = ((uint32_t)(uint8_t)a) |
                         ((uint32_t)(uint8_t)b << 8) |
                         ((uint32_t)(uint8_t)c << 16) |
                         ((uint32_t)(uint8_t)d << 24);
        *(uint32_t*)(out + base) = packed;
    }
    
    int rem_start = (n >> 2) << 2;
    for (int i = rem_start + (tid - (tid / stride) * stride); i < n; i += stride) {
        if (i >= rem_start && i < n) out[i] = (uint8_t)in[i];
    }
}


__global__ void convert_clusters_u16_kernel(
    const int32_t* __restrict__ in,
    uint16_t* __restrict__ out,
    int32_t n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = tid; i < (n >> 1); i += stride) {
        int base = i << 1;
        int32_t a = in[base], b = in[base+1];
        uint32_t packed = ((uint32_t)(uint16_t)a) |
                         ((uint32_t)(uint16_t)b << 16);
        *(uint32_t*)(out + base) = packed;
    }
    
    if ((n & 1) && tid == 0) {
        out[n - 1] = (uint16_t)in[n - 1];
    }
}


template<typename ClusterT>
__global__ void edge_cut_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const ClusterT* __restrict__ clusters,
    int32_t num_vertices,
    double* __restrict__ result)
{
    double local_sum = 0.0;

    const int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int num_warps = (blockDim.x * gridDim.x) >> 5;

    for (int v = warp_id_global; v < num_vertices; v += num_warps) {
        const int32_t cv = (int32_t)clusters[v];
        const int32_t start = offsets[v];
        const int32_t end = offsets[v + 1];
        const int32_t degree = end - start;

        for (int e = lane; e < degree; e += 32) {
            const int32_t idx = start + e;
            const int32_t u = indices[idx];
            if ((int32_t)__ldg(&clusters[u]) != cv) {
                local_sum += (double)weights[idx];
            }
        }
    }

    
    const unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    const int wid = threadIdx.x >> 5;
    __shared__ double warp_sums[WARPS_PER_BLOCK];
    if (lane == 0) warp_sums[wid] = local_sum;
    __syncthreads();

    if (wid == 0) {
        double val = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (lane == 0) {
            atomicAdd(result, val * 0.5);
        }
    }
}


template<typename ClusterT>
__global__ void edge_cut_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const ClusterT* __restrict__ clusters,
    int32_t num_vertices,
    double* __restrict__ result)
{
    double local_sum = 0.0;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int v = tid; v < num_vertices; v += stride) {
        const int32_t cv = (int32_t)clusters[v];
        const int32_t start = offsets[v];
        const int32_t end = offsets[v + 1];

        for (int e = start; e < end; e++) {
            int32_t u = indices[e];
            if ((int32_t)__ldg(&clusters[u]) != cv) {
                local_sum += (double)weights[e];
            }
        }
    }

    
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    __shared__ double warp_sums[WARPS_PER_BLOCK];
    if (lane == 0) warp_sums[wid] = local_sum;
    __syncthreads();

    if (wid == 0) {
        double val = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (lane == 0) {
            atomicAdd(result, val * 0.5);
        }
    }
}

}  

double analyze_clustering_edge_cut(const graph32_t& graph,
                                   const float* edge_weights,
                                   std::size_t num_clusters,
                                   const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    float avg_degree = (float)num_edges / (float)num_vertices;
    bool use_warp = (avg_degree >= 6.0f);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const float* d_weights = edge_weights;
    double* d_result = cache.d_result;

    bool large_graph = (num_vertices > 5000000);

    if (large_graph && num_clusters <= 255) {
        cache.ensure_clusters((int64_t)num_vertices * sizeof(uint8_t));
        uint8_t* clusters_u8 = reinterpret_cast<uint8_t*>(cache.clusters_narrow);

        int num_blocks_conv = (num_vertices + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);
        if (num_blocks_conv > 108 * 4) num_blocks_conv = 108 * 4;
        convert_clusters_u8_kernel<<<num_blocks_conv, BLOCK_SIZE>>>(
            cluster_assignments, clusters_u8, num_vertices);

        cudaMemset(d_result, 0, sizeof(double));
        if (use_warp) {
            int num_blocks = 108 * 8;
            int max_blocks = (num_vertices + (BLOCK_SIZE/32) - 1) / (BLOCK_SIZE/32);
            if (num_blocks > max_blocks) num_blocks = max_blocks;
            edge_cut_warp_kernel<uint8_t><<<num_blocks, BLOCK_SIZE>>>(
                d_offsets, d_indices, d_weights, clusters_u8, num_vertices, d_result);
        } else {
            int num_blocks = 108 * 8;
            int max_blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
            if (num_blocks > max_blocks) num_blocks = max_blocks;
            edge_cut_thread_kernel<uint8_t><<<num_blocks, BLOCK_SIZE>>>(
                d_offsets, d_indices, d_weights, clusters_u8, num_vertices, d_result);
        }
    } else if (large_graph && num_clusters <= 65535) {
        cache.ensure_clusters((int64_t)num_vertices * sizeof(uint16_t));
        uint16_t* clusters_u16 = reinterpret_cast<uint16_t*>(cache.clusters_narrow);

        int num_blocks_conv = (num_vertices + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);
        if (num_blocks_conv > 108 * 4) num_blocks_conv = 108 * 4;
        convert_clusters_u16_kernel<<<num_blocks_conv, BLOCK_SIZE>>>(
            cluster_assignments, clusters_u16, num_vertices);

        cudaMemset(d_result, 0, sizeof(double));
        if (use_warp) {
            int num_blocks = 108 * 8;
            int max_blocks = (num_vertices + (BLOCK_SIZE/32) - 1) / (BLOCK_SIZE/32);
            if (num_blocks > max_blocks) num_blocks = max_blocks;
            edge_cut_warp_kernel<uint16_t><<<num_blocks, BLOCK_SIZE>>>(
                d_offsets, d_indices, d_weights, clusters_u16, num_vertices, d_result);
        } else {
            int num_blocks = 108 * 8;
            int max_blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
            if (num_blocks > max_blocks) num_blocks = max_blocks;
            edge_cut_thread_kernel<uint16_t><<<num_blocks, BLOCK_SIZE>>>(
                d_offsets, d_indices, d_weights, clusters_u16, num_vertices, d_result);
        }
    } else {
        cudaMemset(d_result, 0, sizeof(double));
        if (use_warp) {
            int num_blocks = 108 * 8;
            int max_blocks = (num_vertices + (BLOCK_SIZE/32) - 1) / (BLOCK_SIZE/32);
            if (num_blocks > max_blocks) num_blocks = max_blocks;
            edge_cut_warp_kernel<int32_t><<<num_blocks, BLOCK_SIZE>>>(
                d_offsets, d_indices, d_weights, cluster_assignments, num_vertices, d_result);
        } else {
            int num_blocks = 108 * 8;
            int max_blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
            if (num_blocks > max_blocks) num_blocks = max_blocks;
            edge_cut_thread_kernel<int32_t><<<num_blocks, BLOCK_SIZE>>>(
                d_offsets, d_indices, d_weights, cluster_assignments, num_vertices, d_result);
        }
    }

    double h_result;
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return h_result;
}

}  
