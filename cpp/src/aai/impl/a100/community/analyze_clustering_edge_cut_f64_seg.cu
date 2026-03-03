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
    double* d_result = nullptr;

    Cache() {
        cudaMalloc(&d_result, sizeof(double));
    }

    ~Cache() override {
        if (d_result) cudaFree(d_result);
    }
};



template <int BLOCK_SIZE>
__global__ void edge_cut_high_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ result,
    int32_t start_vertex,
    int32_t num_vertices_in_segment)
{
    int vertex_idx = blockIdx.x;
    if (vertex_idx >= num_vertices_in_segment) return;

    int32_t v = start_vertex + vertex_idx;
    int32_t cluster_v = cluster_assignments[v];
    int32_t edge_start = offsets[v];
    int32_t edge_end = offsets[v + 1];

    double sum = 0.0;
    for (int32_t e = edge_start + threadIdx.x; e < edge_end; e += BLOCK_SIZE) {
        int32_t neighbor = __ldg(&indices[e]);
        if (cluster_v != __ldg(&cluster_assignments[neighbor])) {
            sum += edge_weights[e];
        }
    }

    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(result, block_sum);
    }
}



template <int BLOCK_SIZE>
__global__ void edge_cut_medium_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ result,
    int32_t start_vertex,
    int32_t num_vertices_in_segment)
{
    constexpr int WARP_SIZE = 32;
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_in_block = threadIdx.x / WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    double sum = 0.0;

    if (warp_id < num_vertices_in_segment) {
        int32_t v = start_vertex + warp_id;
        int32_t cluster_v = __ldg(&cluster_assignments[v]);
        int32_t edge_start = __ldg(&offsets[v]);
        int32_t edge_end = __ldg(&offsets[v + 1]);

        for (int32_t e = edge_start + lane; e < edge_end; e += WARP_SIZE) {
            int32_t neighbor = __ldg(&indices[e]);
            if (cluster_v != __ldg(&cluster_assignments[neighbor])) {
                sum += edge_weights[e];
            }
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
    }

    
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    
    double val = (lane == 0) ? sum : 0.0;
    double block_sum = BlockReduce(temp_storage).Sum(val);

    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(result, block_sum);
    }
}



template <int BLOCK_SIZE>
__global__ void edge_cut_low_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ result,
    int32_t start_vertex,
    int32_t num_vertices_in_segment)
{
    double total_sum = 0.0;

    
    for (int tid = threadIdx.x + blockIdx.x * blockDim.x;
         tid < num_vertices_in_segment;
         tid += blockDim.x * gridDim.x) {

        int32_t v = start_vertex + tid;
        int32_t cluster_v = __ldg(&cluster_assignments[v]);
        int32_t edge_start = __ldg(&offsets[v]);
        int32_t edge_end = __ldg(&offsets[v + 1]);

        for (int32_t e = edge_start; e < edge_end; e++) {
            int32_t neighbor = __ldg(&indices[e]);
            if (cluster_v != __ldg(&cluster_assignments[neighbor])) {
                total_sum += edge_weights[e];
            }
        }
    }

    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_sum = BlockReduce(temp_storage).Sum(total_sum);

    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(result, block_sum);
    }
}


__global__ void halve_result(double* result) {
    result[0] *= 0.5;
}

}  

double analyze_clustering_edge_cut_seg(const graph32_t& graph,
                                       const double* edge_weights,
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

    cudaMemsetAsync(cache.d_result, 0, sizeof(double));

    
    int32_t num_high = seg1 - seg0;
    if (num_high > 0) {
        edge_cut_high_degree<512><<<num_high, 512>>>(
            offsets, indices, edge_weights, cluster_assignments,
            cache.d_result, seg0, num_high);
    }

    
    int32_t num_medium = seg2 - seg1;
    if (num_medium > 0) {
        constexpr int BLOCK_SIZE = 256;
        int warps_per_block = BLOCK_SIZE / 32;
        int grid = (num_medium + warps_per_block - 1) / warps_per_block;
        edge_cut_medium_degree<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(
            offsets, indices, edge_weights, cluster_assignments,
            cache.d_result, seg1, num_medium);
    }

    
    int32_t num_low = seg3 - seg2;
    if (num_low > 0) {
        constexpr int BLOCK_SIZE = 256;
        int grid = (num_low + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        if (grid > 2048) grid = 2048;
        edge_cut_low_degree<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(
            offsets, indices, edge_weights, cluster_assignments,
            cache.d_result, seg2, num_low);
    }

    halve_result<<<1, 1>>>(cache.d_result);

    double h_result;
    cudaMemcpy(&h_result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);

    return h_result;
}

}  
