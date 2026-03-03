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
    
    int16_t* compress_buf = nullptr;
    int64_t compress_capacity = 0;

    
    double* d_result = nullptr;
    bool result_allocated = false;

    
    double* h_result = nullptr;

    Cache() {
        cudaMallocHost(&h_result, sizeof(double));
        cudaMalloc(&d_result, sizeof(double));
        result_allocated = true;
    }

    void ensure_compress(int64_t n) {
        if (compress_capacity < n) {
            if (compress_buf) cudaFree(compress_buf);
            cudaMalloc(&compress_buf, n * sizeof(int16_t));
            compress_capacity = n;
        }
    }

    ~Cache() override {
        if (compress_buf) cudaFree(compress_buf);
        if (d_result) cudaFree(d_result);
        if (h_result) cudaFreeHost(h_result);
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
        int4 vals = __ldg(reinterpret_cast<const int4*>(src) + idx);
        uint8_t a = (uint8_t)vals.x;
        uint8_t b = (uint8_t)vals.y;
        uint8_t c = (uint8_t)vals.z;
        uint8_t d = (uint8_t)vals.w;
        uint32_t packed = (uint32_t)a | ((uint32_t)b << 8) | ((uint32_t)c << 16) | ((uint32_t)d << 24);
        reinterpret_cast<uint32_t*>(dst)[idx] = packed;
    } else {
        for (int i = idx4; i < n && i < idx4 + 4; i++)
            dst[i] = (uint8_t)__ldg(src + i);
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
        int2 vals = __ldg(reinterpret_cast<const int2*>(src) + idx);
        int16_t lo = (int16_t)vals.x;
        int16_t hi = (int16_t)vals.y;
        int32_t packed = ((int32_t)(unsigned short)hi << 16) | (int32_t)(unsigned short)lo;
        reinterpret_cast<int32_t*>(dst)[idx] = packed;
    } else if (idx2 < n) {
        dst[idx2] = (int16_t)__ldg(src + idx2);
    }
}






template<typename ClusterT, int BLOCK_SIZE>
__global__ void edge_cut_high_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const ClusterT* __restrict__ clusters,
    int32_t seg_start, int32_t seg_end,
    double* __restrict__ result)
{
    int vid = seg_start + blockIdx.x;
    if (vid >= seg_end) return;

    ClusterT cu = __ldg(clusters + vid);
    int32_t rs = __ldg(offsets + vid);
    int32_t re = __ldg(offsets + vid + 1);

    float s = 0.0f;
    for (int32_t e = rs + threadIdx.x; e < re; e += BLOCK_SIZE) {
        int32_t v = __ldg(indices + e);
        s = __fmaf_rn((cu != __ldg(clusters + v)) ? 1.0f : 0.0f, __ldg(weights + e), s);
    }

    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage ts;
    float bs = BR(ts).Sum(s);

    if (threadIdx.x == 0)
        atomicAdd(result, (double)bs * 0.5);
}


template<typename ClusterT, int BLOCK_SIZE>
__global__ void edge_cut_warp_per_vertex(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const ClusterT* __restrict__ clusters,
    int32_t seg_start, int32_t num_vertices_seg,
    double* __restrict__ result)
{
    constexpr int WPB = BLOCK_SIZE / 32;
    int wid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int global_warps = gridDim.x * WPB;
    int gw = blockIdx.x * WPB + wid;

    float total_s = 0.0f;

    for (int w = gw; w < num_vertices_seg; w += global_warps) {
        int vid = seg_start + w;
        ClusterT cu = __ldg(clusters + vid);
        int32_t rs = __ldg(offsets + vid);
        int32_t re = __ldg(offsets + vid + 1);

        float s = 0.0f;
        for (int32_t e = rs + lane; e < re; e += 32) {
            int32_t v = __ldg(indices + e);
            s = __fmaf_rn((cu != __ldg(clusters + v)) ? 1.0f : 0.0f, __ldg(weights + e), s);
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            s += __shfl_down_sync(0xffffffff, s, o);

        if (lane == 0) total_s += s;
    }

    __shared__ float ws[WPB];
    if (lane == 0) ws[wid] = total_s;
    __syncthreads();

    if (wid == 0) {
        float v = (lane < WPB) ? ws[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffff, v, o);
        if (lane == 0)
            atomicAdd(result, (double)v * 0.5);
    }
}


template<typename ClusterT, int BLOCK_SIZE>
__global__ void edge_cut_low_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const ClusterT* __restrict__ clusters,
    int32_t seg_start, int32_t seg_end,
    double* __restrict__ result)
{
    int vid = seg_start + blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float s = 0.0f;
    if (vid < seg_end) {
        ClusterT cu = __ldg(clusters + vid);
        int32_t rs = __ldg(offsets + vid);
        int32_t re = __ldg(offsets + vid + 1);

        for (int32_t e = rs; e < re; e++) {
            int32_t v = __ldg(indices + e);
            s = __fmaf_rn((cu != __ldg(clusters + v)) ? 1.0f : 0.0f, __ldg(weights + e), s);
        }
    }

    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage ts;
    float bs = BR(ts).Sum(s);

    if (threadIdx.x == 0)
        atomicAdd(result, (double)bs * 0.5);
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
        edge_cut_high_degree<ClusterT, 512><<<s1 - s0, 512, 0, stream>>>(
            offsets, indices, weights, clusters, s0, s1, result);
    }

    if (s2 > s1) {
        int nv = s2 - s1;
        int warps_per_block = 8;
        int blocks = (nv + warps_per_block - 1) / warps_per_block;
        if (blocks > 80 * 6) blocks = 80 * 6;
        edge_cut_warp_per_vertex<ClusterT, 256><<<blocks, 256, 0, stream>>>(
            offsets, indices, weights, clusters, s1, nv, result);
    }

    
    
    int64_t l2_threshold;
    if (sizeof(ClusterT) == 1)
        l2_threshold = (int64_t)(6 * 1024 * 1024);    
    else if (sizeof(ClusterT) == 2)
        l2_threshold = (int64_t)(3 * 1024 * 1024);     
    else
        l2_threshold = (int64_t)(1536 * 1024);          

    if (s3 > s2) {
        int nv = s3 - s2;

        if ((int64_t)num_vertices <= l2_threshold) {
            int warps_per_block = 8;
            int blocks = (nv + warps_per_block - 1) / warps_per_block;
            if (blocks > 80 * 6) blocks = 80 * 6;
            edge_cut_warp_per_vertex<ClusterT, 256><<<blocks, 256, 0, stream>>>(
                offsets, indices, weights, clusters, s2, nv, result);
        } else {
            int blocks = (nv + 255) / 256;
            edge_cut_low_degree<ClusterT, 256><<<blocks, 256, 0, stream>>>(
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

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;

    const auto& seg = graph.segment_offsets.value();
    int32_t s0 = seg[0], s1 = seg[1], s2 = seg[2], s3 = seg[3];

    cudaMemset(cache.d_result, 0, sizeof(double));

    void* compress_buf = nullptr;
    if (num_clusters <= 32767) {
        cache.ensure_compress(num_vertices);
        compress_buf = cache.compress_buf;
    }

    if (num_clusters <= 255 && compress_buf != nullptr) {
        uint8_t* buf = (uint8_t*)compress_buf;
        int threads_needed = (num_vertices + 3) / 4;
        int compress_blocks = (threads_needed + 255) / 256;
        compress_clusters_to_u8<<<compress_blocks, 256>>>(
            cluster_assignments, buf, num_vertices);
        launch_kernels<uint8_t>(
            offsets, indices, edge_weights, buf,
            num_vertices, s0, s1, s2, s3, cache.d_result, 0);
    } else if (num_clusters <= 32767 && compress_buf != nullptr) {
        int16_t* buf = (int16_t*)compress_buf;
        int threads_needed = (num_vertices + 1) / 2;
        int compress_blocks = (threads_needed + 255) / 256;
        compress_clusters_to_i16<<<compress_blocks, 256>>>(
            cluster_assignments, buf, num_vertices);
        launch_kernels<int16_t>(
            offsets, indices, edge_weights, buf,
            num_vertices, s0, s1, s2, s3, cache.d_result, 0);
    } else {
        launch_kernels<int32_t>(
            offsets, indices, edge_weights, cluster_assignments,
            num_vertices, s0, s1, s2, s3, cache.d_result, 0);
    }

    cudaMemcpy(cache.h_result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return *cache.h_result;
}

}  
