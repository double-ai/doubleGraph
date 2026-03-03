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
#include <cstddef>

namespace aai {

namespace {

struct Cache : Cacheable {
    double* d_intra_sum = nullptr;
    double* d_cluster_degree_sum = nullptr;
    double* d_result = nullptr;
    double* h_result_pinned = nullptr;

    std::size_t cluster_buf_cap = 0;

    Cache() {
        cudaMalloc(&d_intra_sum, sizeof(double));
        cudaMalloc(&d_result, sizeof(double));
        cluster_buf_cap = 8192;
        cudaMalloc(&d_cluster_degree_sum, cluster_buf_cap * sizeof(double));
        cudaMallocHost(&h_result_pinned, sizeof(double));
    }

    void ensure_cluster_buf(std::size_t needed) {
        if (needed > cluster_buf_cap) {
            if (d_cluster_degree_sum) cudaFree(d_cluster_degree_sum);
            cluster_buf_cap = needed * 2;
            cudaMalloc(&d_cluster_degree_sum, cluster_buf_cap * sizeof(double));
        }
    }

    ~Cache() override {
        if (d_intra_sum) cudaFree(d_intra_sum);
        if (d_cluster_degree_sum) cudaFree(d_cluster_degree_sum);
        if (d_result) cudaFree(d_result);
        if (h_result_pinned) cudaFreeHost(h_result_pinned);
    }
};




template<int BLOCK_SIZE>
__global__ void modularity_high_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ d_intra_sum,
    double* __restrict__ d_cluster_degree_sum,
    int32_t vertex_start,
    int32_t vertex_end)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = vertex_start + blockIdx.x;
    if (v >= vertex_end) return;

    int32_t edge_start = offsets[v];
    int32_t edge_end = offsets[v + 1];
    int32_t my_cluster = cluster_assignments[v];

    double intra_w = 0.0;
    double degree_w = 0.0;

    for (int32_t e = edge_start + threadIdx.x; e < edge_end; e += BLOCK_SIZE) {
        double w = (double)edge_weights[e];
        degree_w += w;
        if (cluster_assignments[indices[e]] == my_cluster) {
            intra_w += w;
        }
    }

    double block_intra = BlockReduce(temp_storage).Sum(intra_w);
    if (threadIdx.x == 0 && block_intra != 0.0)
        atomicAdd(d_intra_sum, block_intra);
    __syncthreads();
    double block_degree = BlockReduce(temp_storage).Sum(degree_w);
    if (threadIdx.x == 0 && block_degree != 0.0)
        atomicAdd(&d_cluster_degree_sum[my_cluster], block_degree);
}





template<int WARPS_PER_BLOCK>
__global__ void modularity_mid_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ d_intra_sum,
    double* __restrict__ d_cluster_degree_sum,
    int32_t vertex_start,
    int32_t vertex_end,
    int32_t num_clusters)
{
    constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * 32;
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    extern __shared__ float s_cluster_sum[];

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE)
        s_cluster_sum[i] = 0.0f;
    __syncthreads();

    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int v = vertex_start + blockIdx.x * WARPS_PER_BLOCK + warp_in_block;

    double intra_w = 0.0;
    double degree_w = 0.0;
    int32_t my_cluster = 0;

    if (v < vertex_end) {
        int32_t edge_start = offsets[v];
        int32_t edge_end = offsets[v + 1];
        my_cluster = cluster_assignments[v];

        for (int32_t e = edge_start + lane; e < edge_end; e += 32) {
            double w = (double)edge_weights[e];
            degree_w += w;
            if (cluster_assignments[indices[e]] == my_cluster)
                intra_w += w;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            degree_w += __shfl_down_sync(0xFFFFFFFF, degree_w, offset);

        if (lane == 0 && degree_w != 0.0)
            atomicAdd(&s_cluster_sum[my_cluster], (float)degree_w);
    }

    __syncthreads();
    double block_intra = BlockReduce(temp_storage).Sum(intra_w);
    if (threadIdx.x == 0 && block_intra != 0.0)
        atomicAdd(d_intra_sum, block_intra);

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        float val = s_cluster_sum[i];
        if (val != 0.0f)
            atomicAdd(&d_cluster_degree_sum[i], (double)val);
    }
}




template<int WARPS_PER_BLOCK>
__global__ void modularity_mid_degree_noshmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ d_intra_sum,
    double* __restrict__ d_cluster_degree_sum,
    int32_t vertex_start,
    int32_t vertex_end)
{
    constexpr int BLOCK_SIZE = WARPS_PER_BLOCK * 32;
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int v = vertex_start + blockIdx.x * WARPS_PER_BLOCK + warp_in_block;

    double intra_w = 0.0;
    double degree_w = 0.0;
    int32_t my_cluster = 0;

    if (v < vertex_end) {
        int32_t edge_start = offsets[v];
        int32_t edge_end = offsets[v + 1];
        my_cluster = cluster_assignments[v];

        for (int32_t e = edge_start + lane; e < edge_end; e += 32) {
            double w = (double)edge_weights[e];
            degree_w += w;
            if (cluster_assignments[indices[e]] == my_cluster)
                intra_w += w;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            degree_w += __shfl_down_sync(0xFFFFFFFF, degree_w, offset);

        if (lane == 0 && degree_w != 0.0)
            atomicAdd(&d_cluster_degree_sum[my_cluster], degree_w);
    }

    double block_intra = BlockReduce(temp_storage).Sum(intra_w);
    if (threadIdx.x == 0 && block_intra != 0.0)
        atomicAdd(d_intra_sum, block_intra);
}





template<int BLOCK_SIZE>
__global__ void modularity_low_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ d_intra_sum,
    double* __restrict__ d_cluster_degree_sum,
    int32_t vertex_start,
    int32_t vertex_end,
    int32_t num_clusters)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    extern __shared__ float s_cluster_sum[];

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE)
        s_cluster_sum[i] = 0.0f;
    __syncthreads();

    int v = vertex_start + blockIdx.x * BLOCK_SIZE + threadIdx.x;

    double intra_w = 0.0;
    float degree_w = 0.0f;
    int32_t my_cluster = 0;

    if (v < vertex_end) {
        int32_t edge_start = offsets[v];
        int32_t edge_end = offsets[v + 1];
        my_cluster = cluster_assignments[v];

        for (int32_t e = edge_start; e < edge_end; e++) {
            float w = edge_weights[e];
            degree_w += w;
            if (cluster_assignments[indices[e]] == my_cluster)
                intra_w += (double)w;
        }
    }

    if (v < vertex_end && degree_w != 0.0f)
        atomicAdd(&s_cluster_sum[my_cluster], degree_w);
    __syncthreads();

    double block_intra = BlockReduce(temp_storage).Sum(intra_w);
    if (threadIdx.x == 0 && block_intra != 0.0)
        atomicAdd(d_intra_sum, block_intra);

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        float val = s_cluster_sum[i];
        if (val != 0.0f)
            atomicAdd(&d_cluster_degree_sum[i], (double)val);
    }
}




template<int BLOCK_SIZE>
__global__ void modularity_low_degree_noshmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ d_intra_sum,
    double* __restrict__ d_cluster_degree_sum,
    int32_t vertex_start,
    int32_t vertex_end)
{
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = vertex_start + blockIdx.x * BLOCK_SIZE + threadIdx.x;

    double intra_w = 0.0;
    double degree_w = 0.0;
    int32_t my_cluster = 0;

    if (v < vertex_end) {
        int32_t edge_start = offsets[v];
        int32_t edge_end = offsets[v + 1];
        my_cluster = cluster_assignments[v];

        for (int32_t e = edge_start; e < edge_end; e++) {
            double w = (double)edge_weights[e];
            degree_w += w;
            if (cluster_assignments[indices[e]] == my_cluster)
                intra_w += w;
        }
    }

    double block_intra = BlockReduce(temp_storage).Sum(intra_w);
    if (threadIdx.x == 0 && block_intra != 0.0)
        atomicAdd(d_intra_sum, block_intra);

    if (v < vertex_end && degree_w != 0.0)
        atomicAdd(&d_cluster_degree_sum[my_cluster], degree_w);
}




__global__ void compute_modularity_kernel(
    const double* __restrict__ d_intra_sum,
    const double* __restrict__ d_cluster_degree_sum,
    double* __restrict__ d_result,
    int32_t num_clusters)
{
    double total_weight = 0.0;
    for (int c = 0; c < num_clusters; c++)
        total_weight += d_cluster_degree_sum[c];

    double modularity = 0.0;
    if (total_weight > 0.0) {
        modularity = *d_intra_sum / total_weight;
        double inv_total = 1.0 / total_weight;
        for (int c = 0; c < num_clusters; c++) {
            double ac = d_cluster_degree_sum[c] * inv_total;
            modularity -= ac * ac;
        }
    }
    *d_result = modularity;
}

}  

double analyze_clustering_modularity_seg(const graph32_t& graph,
                                         const float* edge_weights,
                                         std::size_t num_clusters,
                                         const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0];
    int32_t seg1 = seg[1];
    int32_t seg2 = seg[2];
    int32_t seg3 = seg[3];

    int32_t nc = (int32_t)num_clusters;

    
    bool reallocated = (num_clusters > cache.cluster_buf_cap);
    cache.ensure_cluster_buf(num_clusters);

    
    cudaMemsetAsync(cache.d_cluster_degree_sum, 0, 4096 * sizeof(double));

    if (reallocated) {
        
        cudaMemsetAsync(cache.d_cluster_degree_sum, 0, num_clusters * sizeof(double));
    } else if (nc > 4096) {
        
        cudaMemsetAsync(cache.d_cluster_degree_sum + 4096, 0,
                        (nc - 4096) * sizeof(double), 0);
    }

    
    cudaMemsetAsync(cache.d_intra_sum, 0, sizeof(double));

    
    {
        int n = seg1 - seg0;
        if (n > 0) {
            modularity_high_degree<256><<<n, 256>>>(
                d_offsets, d_indices, edge_weights, cluster_assignments,
                cache.d_intra_sum, cache.d_cluster_degree_sum, seg0, seg1);
        }
    }

    
    {
        int n = seg2 - seg1;
        if (n > 0) {
            constexpr int WPB = 8;
            int blocks = (n + WPB - 1) / WPB;
            if (nc <= 4096) {
                size_t smem = nc * sizeof(float);
                modularity_mid_degree<WPB><<<blocks, WPB * 32, smem>>>(
                    d_offsets, d_indices, edge_weights, cluster_assignments,
                    cache.d_intra_sum, cache.d_cluster_degree_sum, seg1, seg2, nc);
            } else {
                modularity_mid_degree_noshmem<WPB><<<blocks, WPB * 32>>>(
                    d_offsets, d_indices, edge_weights, cluster_assignments,
                    cache.d_intra_sum, cache.d_cluster_degree_sum, seg1, seg2);
            }
        }
    }

    
    {
        int n = seg3 - seg2;
        if (n > 0) {
            constexpr int BS = 256;
            int blocks = (n + BS - 1) / BS;
            if (nc <= 4096) {
                size_t smem = nc * sizeof(float);
                modularity_low_degree<BS><<<blocks, BS, smem>>>(
                    d_offsets, d_indices, edge_weights, cluster_assignments,
                    cache.d_intra_sum, cache.d_cluster_degree_sum, seg2, seg3, nc);
            } else {
                modularity_low_degree_noshmem<BS><<<blocks, BS>>>(
                    d_offsets, d_indices, edge_weights, cluster_assignments,
                    cache.d_intra_sum, cache.d_cluster_degree_sum, seg2, seg3);
            }
        }
    }

    
    compute_modularity_kernel<<<1, 1>>>(
        cache.d_intra_sum, cache.d_cluster_degree_sum,
        cache.d_result, nc);

    
    cudaMemcpy(cache.h_result_pinned, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);

    return *cache.h_result_pinned;
}

}  
