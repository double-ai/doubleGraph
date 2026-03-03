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

constexpr int BLOCK_SIZE = 256;

struct Cache : Cacheable {
    uint8_t* d_cluster_u8 = nullptr;
    int32_t cluster_u8_capacity = 0;
    int32_t* d_u8_overflow_flag = nullptr;
    int32_t* d_chunk_counter = nullptr;
    double* d_result = nullptr;

    Cache() {
        cluster_u8_capacity = 70 * 1024 * 1024;  
        cudaMalloc(&d_cluster_u8, (size_t)cluster_u8_capacity);
        cudaMalloc(&d_u8_overflow_flag, sizeof(int32_t));
        cudaMalloc(&d_chunk_counter, sizeof(int32_t));
        cudaMalloc(&d_result, sizeof(double));
    }

    ~Cache() override {
        if (d_cluster_u8) cudaFree(d_cluster_u8);
        if (d_u8_overflow_flag) cudaFree(d_u8_overflow_flag);
        if (d_chunk_counter) cudaFree(d_chunk_counter);
        if (d_result) cudaFree(d_result);
    }
};

static __device__ __forceinline__ float warp_reduce_sum_f32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}


__global__ void convert_i32_to_u8_check(
    const int32_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int32_t n,
    int32_t* __restrict__ overflow_flag)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    int n16 = n / 16;
    for (int i = tid; i < n16; i += stride) {
        const int4* p = reinterpret_cast<const int4*>(src) + (i * 4);
        int4 v0 = p[0];
        int4 v1 = p[1];
        int4 v2 = p[2];
        int4 v3 = p[3];

        int or_all = (v0.x | v0.y | v0.z | v0.w | v1.x | v1.y | v1.z | v1.w | v2.x | v2.y | v2.z | v2.w |
                      v3.x | v3.y | v3.z | v3.w);
        if (or_all & 0xFFFFFF00) atomicOr(overflow_flag, 1);

        uint32_t packed0 = ((uint8_t)v0.x) | ((uint8_t)v0.y << 8) | ((uint8_t)v0.z << 16) | ((uint8_t)v0.w << 24);
        uint32_t packed1 = ((uint8_t)v1.x) | ((uint8_t)v1.y << 8) | ((uint8_t)v1.z << 16) | ((uint8_t)v1.w << 24);
        uint32_t packed2 = ((uint8_t)v2.x) | ((uint8_t)v2.y << 8) | ((uint8_t)v2.z << 16) | ((uint8_t)v2.w << 24);
        uint32_t packed3 = ((uint8_t)v3.x) | ((uint8_t)v3.y << 8) | ((uint8_t)v3.z << 16) | ((uint8_t)v3.w << 24);

        reinterpret_cast<uint4*>(dst)[i] = make_uint4(packed0, packed1, packed2, packed3);
    }

    for (int i = n16 * 16 + tid; i < n; i += stride) {
        int32_t x = src[i];
        if (x & 0xFFFFFF00) atomicOr(overflow_flag, 1);
        dst[i] = (uint8_t)x;
    }
}

__global__ void edge_cut_block_coop_persistent_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_i32,
    const uint8_t* __restrict__ cluster_u8,
    const int32_t* __restrict__ overflow_flag,
    int32_t num_vertices,
    int32_t* __restrict__ chunk_counter,
    double* __restrict__ result)
{
    using BlockScan = cub::BlockScan<int32_t, BLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage scan_temp;

    __shared__ int32_t s_seg_start[BLOCK_SIZE];
    __shared__ uint32_t s_src_cluster[BLOCK_SIZE];
    __shared__ int32_t s_prefix[BLOCK_SIZE + 1];
    __shared__ float warp_sums[BLOCK_SIZE / 32];

    const bool use_u8 = (cluster_u8 != nullptr) && (__ldg(overflow_flag) == 0);

    float local_sum = 0.0f;

    while (true) {
        int32_t chunk = 0;
        if (threadIdx.x == 0) {
            chunk = atomicAdd(chunk_counter, 1);
            s_prefix[0] = chunk;  
        }
        __syncthreads();
        chunk = s_prefix[0];

        int32_t chunk_start = chunk * BLOCK_SIZE;
        if (chunk_start >= num_vertices) break;

        int32_t local_idx = threadIdx.x;
        int32_t u = chunk_start + local_idx;

        int32_t seg_len = 0;
        int32_t seg_start = 0;
        uint32_t cu = 0;

        if (u < num_vertices) {
            seg_start = __ldg(&offsets[u]);
            int32_t end = __ldg(&offsets[u + 1]);
            seg_len = end - seg_start;
            cu = use_u8 ? (uint32_t)__ldg(&cluster_u8[u]) : (uint32_t)__ldg(&cluster_i32[u]);
        }
        s_seg_start[local_idx] = seg_start;
        s_src_cluster[local_idx] = cu;

        int32_t prefix, aggregate;
        BlockScan(scan_temp).ExclusiveSum(seg_len, prefix, aggregate);
        __syncthreads();

        s_prefix[local_idx] = prefix;
        if (local_idx == 0) s_prefix[BLOCK_SIZE] = aggregate;
        __syncthreads();

        if (aggregate == 0) {
            __syncthreads();
            continue;
        }

        for (int32_t elem = local_idx; elem < aggregate; elem += BLOCK_SIZE) {
            int owner = 0;
            if (s_prefix[owner + 128] <= elem) owner += 128;
            if (s_prefix[owner + 64] <= elem) owner += 64;
            if (s_prefix[owner + 32] <= elem) owner += 32;
            if (s_prefix[owner + 16] <= elem) owner += 16;
            if (s_prefix[owner + 8] <= elem) owner += 8;
            if (s_prefix[owner + 4] <= elem) owner += 4;
            if (s_prefix[owner + 2] <= elem) owner += 2;
            if (s_prefix[owner + 1] <= elem) owner += 1;

            int32_t e = s_seg_start[owner] + (elem - s_prefix[owner]);

            int32_t v = __ldg(&indices[e]);
            uint32_t cv = use_u8 ? (uint32_t)__ldg(&cluster_u8[v]) : (uint32_t)__ldg(&cluster_i32[v]);

            if (cv != s_src_cluster[owner]) {
                local_sum += __ldg(&edge_weights[e]);
            }
        }
        __syncthreads();
    }

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    float w = warp_reduce_sum_f32(local_sum);
    if (lane == 0) warp_sums[warp] = w;
    __syncthreads();

    if (warp == 0) {
        float x = (lane < (BLOCK_SIZE / 32)) ? warp_sums[lane] : 0.0f;
        x = warp_reduce_sum_f32(x);
        if (lane == 0) atomicAdd(result, (double)x * 0.5);
    }
}

}  

double analyze_clustering_edge_cut(const graph32_t& graph,
                                   const float* edge_weights,
                                   std::size_t num_clusters,
                                   const int32_t* cluster_assignments) {
    (void)num_clusters;

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;

    cudaMemsetAsync(cache.d_result, 0, sizeof(double));
    if (num_vertices <= 0) {
        double h_result = 0.0;
        cudaDeviceSynchronize();
        return h_result;
    }

    cudaMemsetAsync(cache.d_u8_overflow_flag, 0, sizeof(int32_t));
    cudaMemsetAsync(cache.d_chunk_counter, 0, sizeof(int32_t));

    uint8_t* u8_buf = (cache.d_cluster_u8 && num_vertices <= cache.cluster_u8_capacity)
                          ? cache.d_cluster_u8 : nullptr;

    
    uint8_t* u8_ptr = nullptr;
    if (u8_buf != nullptr && num_vertices >= (1 << 20)) {
        int conv_blocks = (num_vertices / 16 + 255) / 256;
        int max_conv = 58 * 4;
        if (conv_blocks > max_conv) conv_blocks = max_conv;
        if (conv_blocks < 1) conv_blocks = 1;
        convert_i32_to_u8_check<<<conv_blocks, 256>>>(
            cluster_assignments, u8_buf, num_vertices, cache.d_u8_overflow_flag);
        u8_ptr = u8_buf;
    }

    
    int num_blocks = 58 * 4;

    edge_cut_block_coop_persistent_kernel<<<num_blocks, BLOCK_SIZE>>>(
        graph.offsets, graph.indices, edge_weights, cluster_assignments, u8_ptr,
        cache.d_u8_overflow_flag, num_vertices, cache.d_chunk_counter, cache.d_result);

    double h_result;
    cudaMemcpy(&h_result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return h_result;
}

}  
