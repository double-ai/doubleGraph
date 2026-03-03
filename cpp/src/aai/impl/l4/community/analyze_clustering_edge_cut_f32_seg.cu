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
    void* compress_buf = nullptr;
    double* result_dev = nullptr;
    int64_t compress_capacity = 0;
    bool result_allocated = false;

    void ensure(int32_t num_vertices) {
        
        int64_t needed = (int64_t)num_vertices * sizeof(int16_t);
        if (compress_capacity < needed) {
            if (compress_buf) cudaFree(compress_buf);
            cudaMalloc(&compress_buf, needed);
            compress_capacity = needed;
        }
        if (!result_allocated) {
            cudaMalloc(&result_dev, sizeof(double));
            result_allocated = true;
        }
    }

    ~Cache() override {
        if (compress_buf) cudaFree(compress_buf);
        if (result_dev) cudaFree(result_dev);
    }
};




__global__ void compress_clusters_to_u8(
    const int32_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int32_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx4 = idx * 4;
    if (idx4 + 3 < n) {
        int4 vals = reinterpret_cast<const int4*>(src)[idx];
        uint32_t packed = ((uint32_t)(uint8_t)vals.x) |
                         (((uint32_t)(uint8_t)vals.y) << 8) |
                         (((uint32_t)(uint8_t)vals.z) << 16) |
                         (((uint32_t)(uint8_t)vals.w) << 24);
        reinterpret_cast<uint32_t*>(dst)[idx] = packed;
    } else {
        for (int i = idx4; i < n && i < idx4 + 4; i++)
            dst[i] = (uint8_t)src[i];
    }
}

__global__ void compress_clusters_to_i16(
    const int32_t* __restrict__ src,
    int16_t* __restrict__ dst,
    int32_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;
    if (idx2 + 1 < n) {
        int2 vals = reinterpret_cast<const int2*>(src)[idx];
        int16_t lo = (int16_t)vals.x;
        int16_t hi = (int16_t)vals.y;
        int32_t packed = ((int32_t)(unsigned short)hi << 16) | (int32_t)(unsigned short)lo;
        reinterpret_cast<int32_t*>(dst)[idx] = packed;
    } else if (idx2 < n) {
        dst[idx2] = (int16_t)src[idx2];
    }
}






template<typename ClusterT>
__global__ void edge_cut_high_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const ClusterT* __restrict__ clusters,
    int32_t start_vertex,
    int32_t end_vertex,
    double* __restrict__ result)
{
    for (int v = start_vertex + blockIdx.x; v < end_vertex; v += gridDim.x) {
        ClusterT cluster_v = clusters[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float thread_sum = 0.0f;
        for (int e = start + threadIdx.x; e < end; e += blockDim.x) {
            int32_t u = indices[e];
            if (clusters[u] != cluster_v) {
                thread_sum += weights[e];
            }
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }

        __shared__ float warp_sums[8];
        int warp_id = threadIdx.x >> 5;
        int lane = threadIdx.x & 31;

        if (lane == 0) warp_sums[warp_id] = thread_sum;
        __syncthreads();

        if (threadIdx.x == 0) {
            float block_sum = 0.0f;
            for (int i = 0; i < (blockDim.x >> 5); i++)
                block_sum += warp_sums[i];
            if (block_sum != 0.0f)
                atomicAdd(result, (double)block_sum * 0.5);
        }
        __syncthreads();
    }
}


template<typename ClusterT>
__global__ void edge_cut_mid_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const ClusterT* __restrict__ clusters,
    int32_t start_vertex,
    int32_t end_vertex,
    double* __restrict__ result)
{
    int warp_id_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (blockDim.x * gridDim.x) >> 5;

    float local_sum = 0.0f;

    for (int v = start_vertex + warp_id_global; v < end_vertex; v += total_warps) {
        ClusterT cluster_v = clusters[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        for (int e = start + lane; e < end; e += 32) {
            int32_t u = indices[e];
            if (clusters[u] != cluster_v) {
                local_sum += weights[e];
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0 && local_sum != 0.0f) {
        atomicAdd(result, (double)local_sum * 0.5);
    }
}




template<typename ClusterT>
__global__ void edge_cut_block_cooperative(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const ClusterT* __restrict__ clusters,
    int32_t start_vertex,
    int32_t end_vertex,
    double* __restrict__ result)
{
    const int VPB = 256;  

    __shared__ int32_t s_offsets[VPB + 1];
    __shared__ ClusterT s_clusters[VPB];

    float local_sum = 0.0f;

    for (int vb = start_vertex + blockIdx.x * VPB;
         vb < end_vertex;
         vb += gridDim.x * VPB) {

        int num_v = end_vertex - vb;
        if (num_v > VPB) num_v = VPB;

        
        for (int i = threadIdx.x; i <= num_v; i += blockDim.x) {
            s_offsets[i] = offsets[vb + i];
        }
        if (threadIdx.x < num_v) {
            s_clusters[threadIdx.x] = clusters[vb + threadIdx.x];
        }
        __syncthreads();

        int32_t edge_start = s_offsets[0];
        int32_t edge_end = s_offsets[num_v];

        
        for (int e = edge_start + threadIdx.x; e < edge_end; e += blockDim.x) {
            
            int lo = 0, hi = num_v;
            while (lo < hi) {
                int mid = (lo + hi + 1) >> 1;
                if (s_offsets[mid] <= e) lo = mid;
                else hi = mid - 1;
            }

            int32_t u = indices[e];
            if (s_clusters[lo] != clusters[u]) {
                local_sum += weights[e];
            }
        }
        __syncthreads();
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if ((threadIdx.x & 31) == 0 && local_sum != 0.0f) {
        atomicAdd(result, (double)local_sum * 0.5);
    }
}




template<typename ClusterT>
static void launch_kernels(
    const int32_t* offsets,
    const int32_t* indices,
    const float* weights,
    const ClusterT* clusters,
    int32_t num_vertices,
    int32_t s0, int32_t s1, int32_t s2, int32_t s3,
    double* result,
    cudaStream_t stream)
{
    
    if (s1 > s0) {
        int num_high = s1 - s0;
        int grid = num_high;
        if (grid < 58) grid = 58;  
        edge_cut_high_degree<ClusterT><<<grid, 256, 0, stream>>>(
            offsets, indices, weights, clusters, s0, s1, result);
    }

    
    if (s2 > s1) {
        int num_mid = s2 - s1;
        int block_size = 256;
        int warps_per_block = block_size / 32;
        int grid = (num_mid + warps_per_block - 1) / warps_per_block;
        if (grid > 4096) grid = 4096;
        edge_cut_mid_degree<ClusterT><<<grid, block_size, 0, stream>>>(
            offsets, indices, weights, clusters, s1, s2, result);
    }

    
    
    int64_t l2_threshold;
    if (sizeof(ClusterT) == 1)
        l2_threshold = (int64_t)(48 * 1024 * 1024);    
    else if (sizeof(ClusterT) == 2)
        l2_threshold = (int64_t)(24 * 1024 * 1024);    
    else
        l2_threshold = (int64_t)(12 * 1024 * 1024);    

    
    if (s3 > s2) {
        int num_low = s3 - s2;

        if ((int64_t)num_vertices <= l2_threshold) {
            
            int block_size = 256;
            int warps_per_block = block_size / 32;
            int grid = (num_low + warps_per_block - 1) / warps_per_block;
            if (grid > 4096) grid = 4096;
            edge_cut_mid_degree<ClusterT><<<grid, block_size, 0, stream>>>(
                offsets, indices, weights, clusters, s2, s3, result);
        } else {
            
            int VPB = 256;
            int grid = (num_low + VPB - 1) / VPB;
            if (grid > 4096) grid = 4096;
            edge_cut_block_cooperative<ClusterT><<<grid, 256, 0, stream>>>(
                offsets, indices, weights, clusters, s2, s3, result);
        }
    }
}

}  

double analyze_clustering_edge_cut_seg(const graph32_t& graph,
                                       const float* edge_weights,
                                       std::size_t num_clusters,
                                       const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;

    cache.ensure(num_vertices);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    
    const auto& seg = graph.segment_offsets.value();
    int32_t s0 = seg[0], s1 = seg[1], s2 = seg[2], s3 = seg[3];

    double* result_dev = cache.result_dev;

    
    void* compress_buf = nullptr;
    if (num_clusters <= 32767) {
        compress_buf = cache.compress_buf;
    }

    cudaStream_t stream = 0;

    cudaMemsetAsync(result_dev, 0, sizeof(double), stream);

    if (num_clusters <= 255 && compress_buf != nullptr) {
        
        uint8_t* buf = (uint8_t*)compress_buf;
        int threads_needed = (num_vertices + 3) / 4;
        int compress_blocks = (threads_needed + 255) / 256;
        compress_clusters_to_u8<<<compress_blocks, 256, 0, stream>>>(
            cluster_assignments, buf, num_vertices);
        launch_kernels<uint8_t>(
            offsets, indices, edge_weights, buf,
            num_vertices, s0, s1, s2, s3, result_dev, stream);
    } else if (num_clusters <= 32767 && compress_buf != nullptr) {
        
        int16_t* buf = (int16_t*)compress_buf;
        int threads_needed = (num_vertices + 1) / 2;
        int compress_blocks = (threads_needed + 255) / 256;
        compress_clusters_to_i16<<<compress_blocks, 256, 0, stream>>>(
            cluster_assignments, buf, num_vertices);
        launch_kernels<int16_t>(
            offsets, indices, edge_weights, buf,
            num_vertices, s0, s1, s2, s3, result_dev, stream);
    } else {
        
        launch_kernels<int32_t>(
            offsets, indices, edge_weights, cluster_assignments,
            num_vertices, s0, s1, s2, s3, result_dev, stream);
    }

    
    double result_host;
    cudaMemcpy(&result_host, result_dev, sizeof(double), cudaMemcpyDeviceToHost);
    return result_host;
}

}  
