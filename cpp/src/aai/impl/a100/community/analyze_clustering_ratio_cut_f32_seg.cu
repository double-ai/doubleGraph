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

    void ensure_sizes(size_t num_clusters) {
        if (num_clusters > sizes_capacity) {
            if (d_cluster_sizes) cudaFree(d_cluster_sizes);
            sizes_capacity = num_clusters * 2;
            cudaMalloc(&d_cluster_sizes, sizes_capacity * sizeof(int32_t));
        }
    }

    void ensure_u8(size_t num_vertices) {
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

    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int n4 = num_vertices / 4;
    for (int i = tid; i < n4; i += stride) {
        int4 vals = reinterpret_cast<const int4*>(cluster_assignments_i32)[i];
        uint8_t out[4];
        out[0] = (uint8_t)vals.x;
        out[1] = (uint8_t)vals.y;
        out[2] = (uint8_t)vals.z;
        out[3] = (uint8_t)vals.w;
        *reinterpret_cast<uint32_t*>(&cluster_assignments_u8[i * 4]) = *reinterpret_cast<uint32_t*>(out);
        atomicAdd(&s_hist[vals.x], 1);
        atomicAdd(&s_hist[vals.y], 1);
        atomicAdd(&s_hist[vals.z], 1);
        atomicAdd(&s_hist[vals.w], 1);
    }

    
    for (int i = n4 * 4 + tid; i < num_vertices; i += stride) {
        int32_t c = cluster_assignments_i32[i];
        cluster_assignments_u8[i] = (uint8_t)c;
        atomicAdd(&s_hist[c], 1);
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

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int v = tid; v < num_vertices; v += stride)
        atomicAdd(&s_hist[cluster_assignments[v]], 1);
    __syncthreads();

    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        if (s_hist[i] > 0)
            atomicAdd(&cluster_sizes[i], s_hist[i]);
    }
}


__global__ void histogram_kernel_global(
    const int32_t* __restrict__ cluster_assignments,
    int32_t* __restrict__ cluster_sizes,
    int32_t num_vertices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int v = tid; v < num_vertices; v += stride)
        atomicAdd(&cluster_sizes[cluster_assignments[v]], 1);
}







template <int BLOCK_SIZE>
__global__ void ratio_cut_high_degree_u8(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint8_t* __restrict__ cluster_assignments,
    const int32_t* __restrict__ cluster_sizes,
    int32_t seg_start,
    int32_t seg_end,
    double* __restrict__ result
) {
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = seg_start + blockIdx.x;
    if (v >= seg_end) return;

    int cluster_v = (int)cluster_assignments[v];
    int size_v = cluster_sizes[cluster_v];
    int start = offsets[v];
    int end = offsets[v + 1];

    double local_cut = 0.0;
    for (int e = start + threadIdx.x; e < end; e += BLOCK_SIZE) {
        int u = indices[e];
        if (cluster_v != (int)cluster_assignments[u]) {
            local_cut += (double)edge_weights[e];
        }
    }

    double block_cut = BlockReduce(temp_storage).Sum(local_cut);

    if (threadIdx.x == 0 && block_cut > 0.0) {
        atomicAdd(result, block_cut / (double)size_v);
    }
}


template <int BLOCK_SIZE>
__global__ void ratio_cut_high_degree_i32(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    const int32_t* __restrict__ cluster_sizes,
    int32_t seg_start,
    int32_t seg_end,
    double* __restrict__ result
) {
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = seg_start + blockIdx.x;
    if (v >= seg_end) return;

    int cluster_v = cluster_assignments[v];
    int size_v = cluster_sizes[cluster_v];
    int start = offsets[v];
    int end = offsets[v + 1];

    double local_cut = 0.0;
    for (int e = start + threadIdx.x; e < end; e += BLOCK_SIZE) {
        int u = indices[e];
        if (cluster_v != cluster_assignments[u]) {
            local_cut += (double)edge_weights[e];
        }
    }

    double block_cut = BlockReduce(temp_storage).Sum(local_cut);

    if (threadIdx.x == 0 && block_cut > 0.0) {
        atomicAdd(result, block_cut / (double)size_v);
    }
}







__global__ void ratio_cut_mid_degree_u8(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint8_t* __restrict__ cluster_assignments,
    const int32_t* __restrict__ cluster_sizes,
    int32_t seg_start,
    int32_t seg_end,
    double* __restrict__ result
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_local = threadIdx.x >> 5;
    int lane = tid & 31;
    int warp_global = tid >> 5;
    int total_warps = (gridDim.x * blockDim.x) >> 5;

    double warp_contribution = 0.0;

    for (int v = seg_start + warp_global; v < seg_end; v += total_warps) {
        int cluster_v = (int)cluster_assignments[v];
        int size_v = cluster_sizes[cluster_v];
        int start = offsets[v];
        int end = offsets[v + 1];

        float local_cut = 0.0f;
        for (int e = start + lane; e < end; e += 32) {
            int u = indices[e];
            if (cluster_v != (int)cluster_assignments[u]) {
                local_cut += edge_weights[e];
            }
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_cut += __shfl_down_sync(0xffffffff, local_cut, offset);
        }

        if (lane == 0 && local_cut > 0.0f) {
            warp_contribution += (double)local_cut / (double)size_v;
        }
    }

    
    __shared__ double warp_sums[8]; 
    if (lane == 0) {
        warp_sums[warp_local] = warp_contribution;
    }
    __syncthreads();

    if (warp_local == 0) {
        double val = (lane < 8) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0 && val != 0.0) {
            atomicAdd(result, val);
        }
    }
}


__global__ void ratio_cut_mid_degree_i32(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    const int32_t* __restrict__ cluster_sizes,
    int32_t seg_start,
    int32_t seg_end,
    double* __restrict__ result
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_local = threadIdx.x >> 5;
    int lane = tid & 31;
    int warp_global = tid >> 5;
    int total_warps = (gridDim.x * blockDim.x) >> 5;

    double warp_contribution = 0.0;

    for (int v = seg_start + warp_global; v < seg_end; v += total_warps) {
        int cluster_v = cluster_assignments[v];
        int size_v = cluster_sizes[cluster_v];
        int start = offsets[v];
        int end = offsets[v + 1];

        float local_cut = 0.0f;
        for (int e = start + lane; e < end; e += 32) {
            int u = indices[e];
            if (cluster_v != cluster_assignments[u]) {
                local_cut += edge_weights[e];
            }
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_cut += __shfl_down_sync(0xffffffff, local_cut, offset);
        }

        if (lane == 0 && local_cut > 0.0f) {
            warp_contribution += (double)local_cut / (double)size_v;
        }
    }

    
    __shared__ double warp_sums[8]; 
    if (lane == 0) {
        warp_sums[warp_local] = warp_contribution;
    }
    __syncthreads();

    if (warp_local == 0) {
        double val = (lane < 8) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0 && val != 0.0) {
            atomicAdd(result, val);
        }
    }
}







template <int BLOCK_SIZE>
__global__ void ratio_cut_low_degree_u8(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const uint8_t* __restrict__ cluster_assignments,
    const int32_t* __restrict__ cluster_sizes,
    int32_t seg_start,
    int32_t seg_end,
    int32_t num_clusters,
    double* __restrict__ result
) {
    
    extern __shared__ int32_t s_cluster_sizes[];
    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        s_cluster_sizes[i] = cluster_sizes[i];
    }
    __syncthreads();

    double thread_sum = 0.0;

    for (int v = seg_start + blockIdx.x * BLOCK_SIZE + threadIdx.x; v < seg_end; v += gridDim.x * BLOCK_SIZE) {
        int cluster_v = (int)cluster_assignments[v];
        int size_v = s_cluster_sizes[cluster_v];
        int start = offsets[v];
        int end = offsets[v + 1];

        float local_cut = 0.0f;
        for (int e = start; e < end; e++) {
            int u = indices[e];
            if (cluster_v != (int)cluster_assignments[u]) {
                local_cut += edge_weights[e];
            }
        }

        if (local_cut > 0.0f) {
            thread_sum += (double)local_cut / (double)size_v;
        }
    }

    
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_sum = BlockReduce(temp_storage).Sum(thread_sum);

    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(result, block_sum);
    }
}


template <int BLOCK_SIZE>
__global__ void ratio_cut_low_degree_i32(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    const int32_t* __restrict__ cluster_sizes,
    int32_t seg_start,
    int32_t seg_end,
    int32_t num_clusters,
    double* __restrict__ result
) {
    
    extern __shared__ int32_t s_cluster_sizes[];
    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        s_cluster_sizes[i] = cluster_sizes[i];
    }
    __syncthreads();

    double thread_sum = 0.0;

    for (int v = seg_start + blockIdx.x * BLOCK_SIZE + threadIdx.x; v < seg_end; v += gridDim.x * BLOCK_SIZE) {
        int cluster_v = cluster_assignments[v];
        int size_v = s_cluster_sizes[cluster_v];
        int start = offsets[v];
        int end = offsets[v + 1];

        float local_cut = 0.0f;
        for (int e = start; e < end; e++) {
            int u = indices[e];
            if (cluster_v != cluster_assignments[u]) {
                local_cut += edge_weights[e];
            }
        }

        if (local_cut > 0.0f) {
            thread_sum += (double)local_cut / (double)size_v;
        }
    }

    
    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_sum = BlockReduce(temp_storage).Sum(thread_sum);

    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(result, block_sum);
    }
}


template <int BLOCK_SIZE>
__global__ void ratio_cut_low_degree_i32_global(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    const int32_t* __restrict__ cluster_sizes,
    int32_t seg_start,
    int32_t seg_end,
    double* __restrict__ result
) {
    double thread_sum = 0.0;

    for (int v = seg_start + blockIdx.x * BLOCK_SIZE + threadIdx.x; v < seg_end; v += gridDim.x * BLOCK_SIZE) {
        int cluster_v = cluster_assignments[v];
        int size_v = cluster_sizes[cluster_v];
        int start = offsets[v];
        int end = offsets[v + 1];

        float local_cut = 0.0f;
        for (int e = start; e < end; e++) {
            int u = indices[e];
            if (cluster_v != cluster_assignments[u]) {
                local_cut += edge_weights[e];
            }
        }

        if (local_cut > 0.0f) {
            thread_sum += (double)local_cut / (double)size_v;
        }
    }

    typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_sum = BlockReduce(temp_storage).Sum(thread_sum);

    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(result, block_sum);
    }
}






static constexpr int SMEM_CLUSTER_THRESH = 8192;

void launch_convert_and_histogram(
    const int32_t* src, uint8_t* dst, int32_t* sizes,
    int32_t num_vertices, int num_clusters
) {
    if (num_vertices <= 0) return;
    int block = 256;
    int grid = (num_vertices / 4 + block - 1) / block;
    if (grid < 1) grid = 1;
    if (grid > 1728) grid = 1728;
    convert_and_histogram_kernel<<<grid, block, num_clusters * sizeof(int32_t)>>>(
        src, dst, sizes, num_vertices, num_clusters);
}

void launch_histogram(
    const int32_t* cluster_assignments, int32_t* cluster_sizes,
    int32_t num_vertices, int num_clusters
) {
    if (num_vertices <= 0) return;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    if (grid > 1728) grid = 1728;
    if (num_clusters <= SMEM_CLUSTER_THRESH) {
        histogram_kernel<<<grid, block, num_clusters * sizeof(int32_t)>>>(
            cluster_assignments, cluster_sizes, num_vertices, num_clusters);
    } else {
        histogram_kernel_global<<<grid, block>>>(
            cluster_assignments, cluster_sizes, num_vertices);
    }
}

void launch_ratio_cut_high_u8(
    const int32_t* offsets, const int32_t* indices,
    const float* edge_weights, const uint8_t* cluster_assignments,
    const int32_t* cluster_sizes,
    int32_t seg_start, int32_t seg_end, double* result
) {
    int num_v = seg_end - seg_start;
    if (num_v <= 0) return;
    ratio_cut_high_degree_u8<256><<<num_v, 256>>>(
        offsets, indices, edge_weights, cluster_assignments,
        cluster_sizes, seg_start, seg_end, result
    );
}

void launch_ratio_cut_high_i32(
    const int32_t* offsets, const int32_t* indices,
    const float* edge_weights, const int32_t* cluster_assignments,
    const int32_t* cluster_sizes,
    int32_t seg_start, int32_t seg_end, double* result
) {
    int num_v = seg_end - seg_start;
    if (num_v <= 0) return;
    ratio_cut_high_degree_i32<256><<<num_v, 256>>>(
        offsets, indices, edge_weights, cluster_assignments,
        cluster_sizes, seg_start, seg_end, result
    );
}

void launch_ratio_cut_mid_u8(
    const int32_t* offsets, const int32_t* indices,
    const float* edge_weights, const uint8_t* cluster_assignments,
    const int32_t* cluster_sizes,
    int32_t seg_start, int32_t seg_end, double* result
) {
    int num_v = seg_end - seg_start;
    if (num_v <= 0) return;
    int warps_per_block = 8;
    int grid = (num_v + warps_per_block - 1) / warps_per_block;
    if (grid > 1728) grid = 1728;
    ratio_cut_mid_degree_u8<<<grid, 256>>>(
        offsets, indices, edge_weights, cluster_assignments,
        cluster_sizes, seg_start, seg_end, result
    );
}

void launch_ratio_cut_mid_i32(
    const int32_t* offsets, const int32_t* indices,
    const float* edge_weights, const int32_t* cluster_assignments,
    const int32_t* cluster_sizes,
    int32_t seg_start, int32_t seg_end, double* result
) {
    int num_v = seg_end - seg_start;
    if (num_v <= 0) return;
    int warps_per_block = 8;
    int grid = (num_v + warps_per_block - 1) / warps_per_block;
    if (grid > 1728) grid = 1728;
    ratio_cut_mid_degree_i32<<<grid, 256>>>(
        offsets, indices, edge_weights, cluster_assignments,
        cluster_sizes, seg_start, seg_end, result
    );
}

void launch_ratio_cut_low_u8(
    const int32_t* offsets, const int32_t* indices,
    const float* edge_weights, const uint8_t* cluster_assignments,
    const int32_t* cluster_sizes,
    int32_t seg_start, int32_t seg_end, int32_t num_clusters, double* result
) {
    int num_v = seg_end - seg_start;
    if (num_v <= 0) return;
    int block = 256;
    int grid = (num_v + block - 1) / block;
    if (grid > 1728) grid = 1728;
    size_t smem = num_clusters * sizeof(int32_t);
    ratio_cut_low_degree_u8<256><<<grid, block, smem>>>(
        offsets, indices, edge_weights, cluster_assignments,
        cluster_sizes, seg_start, seg_end, num_clusters, result
    );
}

void launch_ratio_cut_low_i32(
    const int32_t* offsets, const int32_t* indices,
    const float* edge_weights, const int32_t* cluster_assignments,
    const int32_t* cluster_sizes,
    int32_t seg_start, int32_t seg_end, int32_t num_clusters, double* result
) {
    int num_v = seg_end - seg_start;
    if (num_v <= 0) return;
    int block = 256;
    int grid = (num_v + block - 1) / block;
    if (grid > 1728) grid = 1728;
    if (num_clusters <= SMEM_CLUSTER_THRESH) {
        size_t smem = num_clusters * sizeof(int32_t);
        ratio_cut_low_degree_i32<256><<<grid, block, smem>>>(
            offsets, indices, edge_weights, cluster_assignments,
            cluster_sizes, seg_start, seg_end, num_clusters, result
        );
    } else {
        ratio_cut_low_degree_i32_global<256><<<grid, block>>>(
            offsets, indices, edge_weights, cluster_assignments,
            cluster_sizes, seg_start, seg_end, result
        );
    }
}

}  

double analyze_clustering_ratio_cut_seg(const graph32_t& graph,
                                        const float* edge_weights,
                                        std::size_t num_clusters,
                                        const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;

    
    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0];
    int32_t seg1 = seg[1];
    int32_t seg2 = seg[2];
    int32_t seg3 = seg[3];

    int nc = static_cast<int>(num_clusters);

    
    cache.ensure_sizes(num_clusters);
    cache.ensure_u8(num_vertices);

    
    cudaMemsetAsync(cache.d_cluster_sizes, 0, nc * sizeof(int32_t));
    cudaMemsetAsync(cache.d_result, 0, sizeof(double));

    bool use_u8 = (nc <= 256);

    if (use_u8 && num_vertices > 0) {
        
        launch_convert_and_histogram(cluster_assignments, cache.d_cluster_u8, cache.d_cluster_sizes, num_vertices, nc);

        
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

        launch_ratio_cut_high_u8(d_offsets, d_indices, edge_weights,
                                 cache.d_cluster_u8, cache.d_cluster_sizes,
                                 seg0, seg1, cache.d_result);
        launch_ratio_cut_mid_u8(d_offsets, d_indices, edge_weights,
                                cache.d_cluster_u8, cache.d_cluster_sizes,
                                seg1, seg2, cache.d_result);
        launch_ratio_cut_low_u8(d_offsets, d_indices, edge_weights,
                                cache.d_cluster_u8, cache.d_cluster_sizes,
                                seg2, seg3, nc, cache.d_result);

        if (cache.l2_persist_size > 0 && u8_bytes > 0) {
            cudaStreamAttrValue a = {};
            a.accessPolicyWindow.base_ptr = nullptr;
            a.accessPolicyWindow.num_bytes = 0;
            a.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
            a.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
            cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &a);
        }
    } else if (num_vertices > 0) {
        
        launch_histogram(cluster_assignments, cache.d_cluster_sizes, num_vertices, nc);

        launch_ratio_cut_high_i32(d_offsets, d_indices, edge_weights,
                                  cluster_assignments, cache.d_cluster_sizes,
                                  seg0, seg1, cache.d_result);
        launch_ratio_cut_mid_i32(d_offsets, d_indices, edge_weights,
                                 cluster_assignments, cache.d_cluster_sizes,
                                 seg1, seg2, cache.d_result);
        launch_ratio_cut_low_i32(d_offsets, d_indices, edge_weights,
                                 cluster_assignments, cache.d_cluster_sizes,
                                 seg2, seg3, nc, cache.d_result);
    }

    double result;
    cudaMemcpy(&result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

}  
