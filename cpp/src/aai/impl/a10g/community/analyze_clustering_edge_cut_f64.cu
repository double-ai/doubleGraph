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

struct Cache : Cacheable {
    double* d_output = nullptr;
    void* d_compressed = nullptr;
    int64_t compressed_capacity = 0;

    Cache() {
        cudaMalloc(&d_output, sizeof(double));
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 4 * 1024 * 1024);
    }

    void ensure_compressed(int64_t bytes) {
        if (compressed_capacity < bytes) {
            if (d_compressed) cudaFree(d_compressed);
            cudaMalloc(&d_compressed, bytes);
            compressed_capacity = bytes;
        }
    }

    ~Cache() override {
        if (d_output) cudaFree(d_output);
        if (d_compressed) cudaFree(d_compressed);
    }
};


__device__ __forceinline__ int32_t load_cs_i32(const int32_t* ptr) {
    int32_t val;
    asm volatile("ld.global.cs.s32 %0, [%1];" : "=r"(val) : "l"(ptr));
    return val;
}
__device__ __forceinline__ double load_cs_f64(const double* ptr) {
    double val;
    asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(val) : "l"(ptr));
    return val;
}


__global__ void compress_clusters_u8(
    const int32_t* __restrict__ cluster_in,
    uint8_t* __restrict__ cluster_out,
    int32_t num_vertices,
    double* output_to_zero
) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && output_to_zero) {
        *output_to_zero = 0.0;
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int num_vec = num_vertices / 4;
    for (int i = tid; i < num_vec; i += stride) {
        int4 vals = reinterpret_cast<const int4*>(cluster_in)[i];
        uint8_t out[4] = {(uint8_t)vals.x, (uint8_t)vals.y, (uint8_t)vals.z, (uint8_t)vals.w};
        reinterpret_cast<uint32_t*>(cluster_out)[i] = *reinterpret_cast<uint32_t*>(out);
    }
    int base = num_vec * 4;
    for (int i = base + tid; i < num_vertices; i += stride)
        cluster_out[i] = (uint8_t)cluster_in[i];
}


__global__ void compress_clusters_i16(
    const int32_t* __restrict__ cluster_in,
    int16_t* __restrict__ cluster_out,
    int32_t num_vertices,
    double* output_to_zero
) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && output_to_zero) {
        *output_to_zero = 0.0;
    }
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < num_vertices; i += stride)
        cluster_out[i] = (int16_t)cluster_in[i];
}

template <typename ClusterT>
__global__ void edge_cut_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const ClusterT* __restrict__ cluster_assignments,
    double* __restrict__ output,
    int32_t num_vertices,
    int32_t num_edges
) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
    int warp_id_global = (blockIdx.x * BLOCK_SIZE + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int num_warps = gridDim.x * WARPS_PER_BLOCK;

    double my_sum = 0.0;

    int warp_chunk = (num_edges + num_warps - 1) / num_warps;
    int warp_start = warp_id_global * warp_chunk;
    int warp_end = warp_start + warp_chunk;
    if (warp_end > num_edges) warp_end = num_edges;

    if (warp_start < num_edges) {
        int src;
        if (lane == 0) {
            int lo = 0, hi = num_vertices;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (__ldg(&offsets[mid + 1]) <= warp_start) lo = mid + 1;
                else hi = mid;
            }
            src = lo;
        }
        src = __shfl_sync(0xffffffff, src, 0);

        int my_src = src;
        int my_next_boundary = __ldg(&offsets[my_src + 1]);
        int my_cluster_src = (int)__ldg(&cluster_assignments[my_src]);

        for (int base = warp_start; base < warp_end; base += 32) {
            int e = base + lane;
            if (e < warp_end) {
                
                int32_t dst = load_cs_i32(&indices[e]);

                
                while (e >= my_next_boundary) {
                    my_src++;
                    my_next_boundary = __ldg(&offsets[my_src + 1]);
                    my_cluster_src = (int)__ldg(&cluster_assignments[my_src]);
                }

                
                int cluster_dst = (int)__ldg(&cluster_assignments[dst]);
                if (my_cluster_src != cluster_dst) {
                    my_sum += load_cs_f64(&edge_weights[e]);
                }
            }
        }
    }

    unsigned mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset >>= 1)
        my_sum += __shfl_down_sync(mask, my_sum, offset);

    int warp_id = threadIdx.x >> 5;
    __shared__ double warp_sums[WARPS_PER_BLOCK];
    if (lane == 0) warp_sums[warp_id] = my_sum;
    __syncthreads();

    if (warp_id == 0) {
        double val = (lane < WARPS_PER_BLOCK) ? warp_sums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(mask, val, offset);
        if (lane == 0) atomicAdd(output, val * 0.5);
    }
}

void launch_edge_cut(
    const int32_t* offsets,
    const int32_t* indices,
    const double* edge_weights,
    const int32_t* cluster_assignments,
    double* output,
    int32_t num_vertices,
    int32_t num_edges,
    int64_t num_clusters,
    void* cluster_compressed
) {
    constexpr int BS = 256;
    constexpr int WPB = BS / 32;

    int num_warps_needed = (num_edges + 255) / 256;
    int num_blocks = (num_warps_needed + WPB - 1) / WPB;
    if (num_blocks > 65536) num_blocks = 65536;
    if (num_blocks < 80) num_blocks = 80;

    int compress_blocks = (num_vertices + BS * 4 - 1) / (BS * 4);
    if (compress_blocks > 256) compress_blocks = 256;
    if (compress_blocks < 1) compress_blocks = 1;

    if (num_clusters <= 255 && cluster_compressed != nullptr) {
        compress_clusters_u8<<<compress_blocks, BS>>>(
            cluster_assignments, (uint8_t*)cluster_compressed, num_vertices, output
        );

        
        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.base_ptr = cluster_compressed;
        attr.accessPolicyWindow.num_bytes = (size_t)num_vertices;
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);

        edge_cut_kernel<uint8_t><<<num_blocks, BS>>>(
            offsets, indices, edge_weights, (uint8_t*)cluster_compressed, output,
            num_vertices, num_edges
        );
    } else if (num_clusters <= 32767 && cluster_compressed != nullptr) {
        compress_clusters_i16<<<compress_blocks, BS>>>(
            cluster_assignments, (int16_t*)cluster_compressed, num_vertices, output
        );

        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.base_ptr = cluster_compressed;
        attr.accessPolicyWindow.num_bytes = (size_t)num_vertices * 2;
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);

        edge_cut_kernel<int16_t><<<num_blocks, BS>>>(
            offsets, indices, edge_weights, (int16_t*)cluster_compressed, output,
            num_vertices, num_edges
        );
    } else {
        cudaMemsetAsync(output, 0, sizeof(double));
        edge_cut_kernel<int32_t><<<num_blocks, BS>>>(
            offsets, indices, edge_weights, cluster_assignments, output,
            num_vertices, num_edges
        );
    }
}

}  

double analyze_clustering_edge_cut(const graph32_t& graph,
                                   const double* edge_weights,
                                   std::size_t num_clusters,
                                   const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    void* compressed_ptr = nullptr;
    if (num_clusters <= 255) {
        cache.ensure_compressed((int64_t)num_vertices);
        compressed_ptr = cache.d_compressed;
    } else if (num_clusters <= 32767) {
        cache.ensure_compressed((int64_t)num_vertices * 2);
        compressed_ptr = cache.d_compressed;
    }

    launch_edge_cut(graph.offsets, graph.indices, edge_weights,
                    cluster_assignments, cache.d_output,
                    num_vertices, num_edges, (int64_t)num_clusters, compressed_ptr);

    double result;
    cudaMemcpy(&result, cache.d_output, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

}  
