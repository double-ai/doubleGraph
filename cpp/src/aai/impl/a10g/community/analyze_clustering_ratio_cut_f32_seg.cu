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
#include <algorithm>

namespace aai {

namespace {

#define BLOCK_SIZE 256

struct Cache : Cacheable {
    int32_t* d_cluster_sizes = nullptr;
    uint8_t* d_cluster_u8 = nullptr;
    double* d_result = nullptr;
    size_t sizes_capacity = 0;
    size_t u8_capacity = 0;
    size_t l2_persist_size = 0;

    Cache() {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        l2_persist_size = prop.persistingL2CacheMaxSize;
        if (l2_persist_size > 0)
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_persist_size);

        sizes_capacity = 1024;
        u8_capacity = 64 * 1024 * 1024;
        cudaMalloc(&d_cluster_sizes, sizes_capacity * sizeof(int32_t));
        cudaMalloc(&d_cluster_u8, u8_capacity);
        cudaMalloc(&d_result, sizeof(double));
    }

    void ensure(size_t num_clusters, size_t num_vertices) {
        if (num_clusters > sizes_capacity) {
            if (d_cluster_sizes) cudaFree(d_cluster_sizes);
            sizes_capacity = num_clusters * 2;
            cudaMalloc(&d_cluster_sizes, sizes_capacity * sizeof(int32_t));
        }
        if (num_vertices > u8_capacity) {
            if (d_cluster_u8) cudaFree(d_cluster_u8);
            u8_capacity = num_vertices * 2;
            cudaMalloc(&d_cluster_u8, u8_capacity);
        }
    }

    ~Cache() override {
        if (d_cluster_sizes) { cudaFree(d_cluster_sizes); d_cluster_sizes = nullptr; }
        if (d_cluster_u8) { cudaFree(d_cluster_u8); d_cluster_u8 = nullptr; }
        if (d_result) { cudaFree(d_result); d_result = nullptr; }
        if (l2_persist_size > 0)
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
    }
};




__global__ void convert_and_histogram_kernel(
    const int32_t* __restrict__ cluster_assignments_i32,
    uint8_t* __restrict__ cluster_assignments_u8,
    int32_t* __restrict__ cluster_sizes,
    int32_t num_vertices,
    int num_clusters)
{
    extern __shared__ int32_t s_hist[];
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x)
        s_hist[i] = 0;
    __syncthreads();

    
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = base + i;
        if (idx < num_vertices) {
            int32_t c = cluster_assignments_i32[idx];
            cluster_assignments_u8[idx] = (uint8_t)c;
            atomicAdd(&s_hist[c], 1);
        }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        if (s_hist[i] > 0)
            atomicAdd(&cluster_sizes[i], s_hist[i]);
    }
}




__global__ void histogram_kernel(
    const int32_t* __restrict__ cluster_assignments,
    int32_t* __restrict__ cluster_sizes,
    int32_t num_vertices,
    int num_clusters)
{
    extern __shared__ int32_t s_hist[];
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x)
        s_hist[i] = 0;
    __syncthreads();

    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = base + i;
        if (idx < num_vertices)
            atomicAdd(&s_hist[cluster_assignments[idx]], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        if (s_hist[i] > 0)
            atomicAdd(&cluster_sizes[i], s_hist[i]);
    }
}




__device__ __forceinline__ float process_edges_u8(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint8_t* __restrict__ cluster_u8,
    int start, int end, int lane_offset, int stride, int c_v)
{
    float local_cut = 0.0f;
    for (int e = start + lane_offset; e < end; e += stride) {
        int nb = __ldg(&indices[e]);
        float w = __ldg(&edge_weights[e]);
        local_cut += ((int)__ldg(&cluster_u8[nb]) != c_v) * w;
    }
    return local_cut;
}

__device__ __forceinline__ float process_edges_i32(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_i32,
    int start, int end, int lane_offset, int stride, int c_v)
{
    float local_cut = 0.0f;
    for (int e = start + lane_offset; e < end; e += stride) {
        int nb = __ldg(&indices[e]);
        float w = __ldg(&edge_weights[e]);
        local_cut += (__ldg(&cluster_i32[nb]) != c_v) * w;
    }
    return local_cut;
}




__global__ void ratio_cut_fused_u8_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint8_t* __restrict__ cluster_u8,
    const int32_t* __restrict__ cluster_sizes,
    double* __restrict__ global_result,
    int32_t high_start, int32_t high_end,
    int32_t mid_start, int32_t mid_end,
    int32_t low_start, int32_t low_end,
    int32_t high_blocks, int32_t high_plus_mid_blocks)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float contribution = 0.0f;

    if (blockIdx.x < high_blocks) {
        int v = high_start + blockIdx.x;
        int c_v = (int)cluster_u8[v];
        int sz = cluster_sizes[c_v];
        float local_cut = 0.0f;
        if (sz > 0) {
            int start = offsets[v], end = offsets[v + 1];
            local_cut = process_edges_u8(indices, edge_weights, cluster_u8, start, end, threadIdx.x, BLOCK_SIZE, c_v);
        }
        float block_sum = BlockReduce(temp).Sum(local_cut);
        if (threadIdx.x == 0 && block_sum != 0.0f && sz > 0)
            atomicAdd(global_result, (double)block_sum / (double)sz);
        return;
    } else if (blockIdx.x < high_plus_mid_blocks) {
        int block_in_mid = blockIdx.x - high_blocks;
        int warp_in_block = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        int v = mid_start + block_in_mid * 8 + warp_in_block;
        if (v < mid_end) {
            int c_v = (int)cluster_u8[v];
            int sz = cluster_sizes[c_v];
            if (sz > 0) {
                int start = offsets[v], end = offsets[v + 1];
                float local_cut = process_edges_u8(indices, edge_weights, cluster_u8, start, end, lane, 32, c_v);
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    local_cut += __shfl_down_sync(0xffffffff, local_cut, offset);
                if (lane == 0)
                    contribution = local_cut / (float)sz;
            }
        }
    } else {
        int block_in_low = blockIdx.x - high_plus_mid_blocks;
        int v = low_start + block_in_low * BLOCK_SIZE + threadIdx.x;
        if (v < low_end) {
            int c_v = (int)cluster_u8[v];
            int sz = cluster_sizes[c_v];
            if (sz > 0) {
                int start = offsets[v], end = offsets[v + 1];
                float local_cut = process_edges_u8(indices, edge_weights, cluster_u8, start, end, 0, 1, c_v);
                contribution = local_cut / (float)sz;
            }
        }
    }

    float block_sum = BlockReduce(temp).Sum(contribution);
    if (threadIdx.x == 0 && block_sum != 0.0f)
        atomicAdd(global_result, (double)block_sum);
}




__global__ void ratio_cut_fused_i32_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_i32,
    const int32_t* __restrict__ cluster_sizes,
    double* __restrict__ global_result,
    int32_t high_start, int32_t high_end,
    int32_t mid_start, int32_t mid_end,
    int32_t low_start, int32_t low_end,
    int32_t high_blocks, int32_t high_plus_mid_blocks)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float contribution = 0.0f;

    if (blockIdx.x < high_blocks) {
        int v = high_start + blockIdx.x;
        int c_v = cluster_i32[v];
        int sz = cluster_sizes[c_v];
        float local_cut = 0.0f;
        if (sz > 0) {
            int start = offsets[v], end = offsets[v + 1];
            local_cut = process_edges_i32(indices, edge_weights, cluster_i32, start, end, threadIdx.x, BLOCK_SIZE, c_v);
        }
        float block_sum = BlockReduce(temp).Sum(local_cut);
        if (threadIdx.x == 0 && block_sum != 0.0f && sz > 0)
            atomicAdd(global_result, (double)block_sum / (double)sz);
        return;
    } else if (blockIdx.x < high_plus_mid_blocks) {
        int block_in_mid = blockIdx.x - high_blocks;
        int warp_in_block = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;
        int v = mid_start + block_in_mid * 8 + warp_in_block;
        if (v < mid_end) {
            int c_v = cluster_i32[v];
            int sz = cluster_sizes[c_v];
            if (sz > 0) {
                int start = offsets[v], end = offsets[v + 1];
                float local_cut = process_edges_i32(indices, edge_weights, cluster_i32, start, end, lane, 32, c_v);
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    local_cut += __shfl_down_sync(0xffffffff, local_cut, offset);
                if (lane == 0)
                    contribution = local_cut / (float)sz;
            }
        }
    } else {
        int block_in_low = blockIdx.x - high_plus_mid_blocks;
        int v = low_start + block_in_low * BLOCK_SIZE + threadIdx.x;
        if (v < low_end) {
            int c_v = cluster_i32[v];
            int sz = cluster_sizes[c_v];
            if (sz > 0) {
                int start = offsets[v], end = offsets[v + 1];
                float local_cut = process_edges_i32(indices, edge_weights, cluster_i32, start, end, 0, 1, c_v);
                contribution = local_cut / (float)sz;
            }
        }
    }

    float block_sum = BlockReduce(temp).Sum(contribution);
    if (threadIdx.x == 0 && block_sum != 0.0f)
        atomicAdd(global_result, (double)block_sum);
}




void launch_convert_and_histogram(
    const int32_t* i32, uint8_t* u8, int32_t* sizes,
    int32_t num_vertices, int num_clusters)
{
    if (num_vertices <= 0) return;
    int block = 256;
    int grid = ((num_vertices + 3) / 4 + block - 1) / block;
    convert_and_histogram_kernel<<<grid, block, num_clusters * sizeof(int32_t)>>>(
        i32, u8, sizes, num_vertices, num_clusters);
}

void launch_histogram(
    const int32_t* assignments, int32_t* sizes,
    int32_t num_vertices, int num_clusters)
{
    if (num_vertices <= 0) return;
    int block = 256;
    int grid = ((num_vertices + 3) / 4 + block - 1) / block;
    histogram_kernel<<<grid, block, num_clusters * sizeof(int32_t)>>>(
        assignments, sizes, num_vertices, num_clusters);
}

void launch_ratio_cut_fused_u8(
    const int32_t* offsets, const int32_t* indices, const float* edge_weights,
    const uint8_t* cluster_u8, const int32_t* cluster_sizes,
    double* global_result,
    int32_t hs, int32_t he, int32_t ms, int32_t me, int32_t ls, int32_t le)
{
    int32_t hb = he - hs, mb = (me - ms + 7) / 8, lb = (le - ls + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int32_t total = hb + mb + lb;
    if (total <= 0) return;
    ratio_cut_fused_u8_kernel<<<total, BLOCK_SIZE>>>(
        offsets, indices, edge_weights, cluster_u8, cluster_sizes,
        global_result, hs, he, ms, me, ls, le, hb, hb + mb);
}

void launch_ratio_cut_fused_i32(
    const int32_t* offsets, const int32_t* indices, const float* edge_weights,
    const int32_t* cluster_i32, const int32_t* cluster_sizes,
    double* global_result,
    int32_t hs, int32_t he, int32_t ms, int32_t me, int32_t ls, int32_t le)
{
    int32_t hb = he - hs, mb = (me - ms + 7) / 8, lb = (le - ls + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int32_t total = hb + mb + lb;
    if (total <= 0) return;
    ratio_cut_fused_i32_kernel<<<total, BLOCK_SIZE>>>(
        offsets, indices, edge_weights, cluster_i32, cluster_sizes,
        global_result, hs, he, ms, me, ls, le, hb, hb + mb);
}

}  

double analyze_clustering_ratio_cut_seg(const graph32_t& graph,
                                        const float* edge_weights,
                                        std::size_t num_clusters,
                                        const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int num_clusters_int = static_cast<int>(num_clusters);

    cache.ensure(num_clusters, (size_t)num_vertices);

    cudaMemsetAsync(cache.d_cluster_sizes, 0, num_clusters_int * sizeof(int32_t));
    cudaMemsetAsync(cache.d_result, 0, sizeof(double));

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    const auto& seg = graph.segment_offsets.value();

    bool use_u8 = (num_clusters_int <= 256);

    if (use_u8 && num_vertices > 0) {
        launch_convert_and_histogram(cluster_assignments, cache.d_cluster_u8,
            cache.d_cluster_sizes, num_vertices, num_clusters_int);

        
        size_t u8_bytes = (size_t)num_vertices;
        if (cache.l2_persist_size > 0 && u8_bytes > 0) {
            cudaStreamAttrValue a;
            a.accessPolicyWindow.base_ptr = (void*)cache.d_cluster_u8;
            a.accessPolicyWindow.num_bytes = std::min(u8_bytes, cache.l2_persist_size);
            a.accessPolicyWindow.hitRatio = (u8_bytes <= cache.l2_persist_size) ? 1.0f :
                (float)cache.l2_persist_size / (float)u8_bytes;
            a.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            a.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &a);
        }

        if (seg[3] > seg[0])
            launch_ratio_cut_fused_u8(d_offsets, d_indices, edge_weights, cache.d_cluster_u8,
                cache.d_cluster_sizes, cache.d_result, seg[0], seg[1], seg[1], seg[2], seg[2], seg[3]);

        if (cache.l2_persist_size > 0 && u8_bytes > 0) {
            cudaStreamAttrValue a = {};
            a.accessPolicyWindow.base_ptr = nullptr;
            a.accessPolicyWindow.num_bytes = 0;
            a.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
            a.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
            cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &a);
        }
    } else if (num_vertices > 0) {
        launch_histogram(cluster_assignments, cache.d_cluster_sizes,
            num_vertices, num_clusters_int);
        if (seg[3] > seg[0])
            launch_ratio_cut_fused_i32(d_offsets, d_indices, edge_weights, cluster_assignments,
                cache.d_cluster_sizes, cache.d_result, seg[0], seg[1], seg[1], seg[2], seg[2], seg[3]);
    }

    double result;
    cudaMemcpy(&result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);

    return result;
}

}  
