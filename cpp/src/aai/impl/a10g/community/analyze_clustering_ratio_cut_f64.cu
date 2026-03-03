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
#include <cstring>

namespace aai {

namespace {

#define SMEM_CLUSTER_THRESHOLD 4096




struct Cache : Cacheable {
    int64_t* counts = nullptr;
    double* result_d = nullptr;
    double* inv_buf = nullptr;
    int16_t* i16_buf = nullptr;

    int64_t counts_capacity = 0;
    int64_t inv_capacity = 0;
    int64_t i16_capacity = 0;
    bool result_allocated = false;

    Cache() {
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 4 * 1024 * 1024);
    }

    void ensure(int num_vertices, int num_clusters) {
        if (!result_allocated) {
            cudaMalloc(&result_d, sizeof(double));
            result_allocated = true;
        }
        if (counts_capacity < num_clusters) {
            if (counts) cudaFree(counts);
            cudaMalloc(&counts, (size_t)num_clusters * sizeof(int64_t));
            counts_capacity = num_clusters;
        }
        if (num_clusters > SMEM_CLUSTER_THRESHOLD && inv_capacity < num_clusters) {
            if (inv_buf) cudaFree(inv_buf);
            cudaMalloc(&inv_buf, (size_t)num_clusters * sizeof(double));
            inv_capacity = num_clusters;
        }
        if (num_clusters <= 32767 && i16_capacity < num_vertices) {
            if (i16_buf) cudaFree(i16_buf);
            cudaMalloc(&i16_buf, (size_t)num_vertices * sizeof(int16_t));
            i16_capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (counts) cudaFree(counts);
        if (result_d) cudaFree(result_d);
        if (inv_buf) cudaFree(inv_buf);
        if (i16_buf) cudaFree(i16_buf);
    }
};




__global__ void histogram_convert_kernel_smem(
    const int32_t* __restrict__ assignments_i32,
    int16_t* __restrict__ assignments_i16,
    int64_t* __restrict__ global_counts,
    int num_vertices,
    int num_clusters
) {
    extern __shared__ int s_counts[];
    for (int k = threadIdx.x; k < num_clusters; k += blockDim.x)
        s_counts[k] = 0;
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vertices; i += gridDim.x * blockDim.x) {
        int c = __ldg(&assignments_i32[i]);
        if ((unsigned)c >= (unsigned)num_clusters) c = (c < 0) ? 0 : (num_clusters - 1);
        assignments_i16[i] = (int16_t)c;
        atomicAdd(&s_counts[c], 1);
    }
    __syncthreads();

    for (int k = threadIdx.x; k < num_clusters; k += blockDim.x) {
        if (s_counts[k] > 0)
            atomicAdd((unsigned long long*)&global_counts[k], (unsigned long long)s_counts[k]);
    }
}




__global__ void histogram_convert_kernel_gmem(
    const int32_t* __restrict__ assignments_i32,
    int16_t* __restrict__ assignments_i16,
    int64_t* __restrict__ global_counts,
    int num_vertices,
    int num_clusters
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vertices; i += gridDim.x * blockDim.x) {
        int c = __ldg(&assignments_i32[i]);
        if ((unsigned)c >= (unsigned)num_clusters) c = (c < 0) ? 0 : (num_clusters - 1);
        assignments_i16[i] = (int16_t)c;
        atomicAdd((unsigned long long*)&global_counts[c], 1ULL);
    }
}




__global__ void histogram_kernel_smem(
    const int32_t* __restrict__ assignments,
    int64_t* __restrict__ global_counts,
    int num_vertices,
    int num_clusters
) {
    extern __shared__ int s_counts2[];
    for (int k = threadIdx.x; k < num_clusters; k += blockDim.x)
        s_counts2[k] = 0;
    __syncthreads();

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vertices; i += gridDim.x * blockDim.x) {
        int c = __ldg(&assignments[i]);
        if ((unsigned)c >= (unsigned)num_clusters) c = (c < 0) ? 0 : (num_clusters - 1);
        atomicAdd(&s_counts2[c], 1);
    }
    __syncthreads();

    for (int k = threadIdx.x; k < num_clusters; k += blockDim.x) {
        if (s_counts2[k] > 0)
            atomicAdd((unsigned long long*)&global_counts[k], (unsigned long long)s_counts2[k]);
    }
}

__global__ void histogram_kernel_gmem(
    const int32_t* __restrict__ assignments,
    int64_t* __restrict__ global_counts,
    int num_vertices,
    int num_clusters
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vertices; i += gridDim.x * blockDim.x) {
        int c = __ldg(&assignments[i]);
        if ((unsigned)c >= (unsigned)num_clusters) c = (c < 0) ? 0 : (num_clusters - 1);
        atomicAdd((unsigned long long*)&global_counts[c], 1ULL);
    }
}




__global__ void compute_inv_kernel(
    const int64_t* __restrict__ cluster_counts,
    double* __restrict__ inv_array,
    int num_clusters
) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < num_clusters; k += gridDim.x * blockDim.x) {
        int64_t sz = cluster_counts[k];
        inv_array[k] = (sz > 0) ? 1.0 / (double)sz : 0.0;
    }
}




template<int GROUP_SIZE, typename ClusterT>
__global__ void ratio_cut_kernel_smem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const ClusterT* __restrict__ cluster_assignments,
    const int64_t* __restrict__ cluster_counts,
    double* __restrict__ result,
    int num_vertices,
    int num_clusters
) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int GROUPS_PER_BLOCK = BLOCK_SIZE / GROUP_SIZE;

    extern __shared__ char smem_raw[];
    double* s_inv = (double*)smem_raw;
    double* group_partials = (double*)(smem_raw + num_clusters * sizeof(double));

    const int lane_in_group = threadIdx.x % GROUP_SIZE;
    const int my_group_in_block = threadIdx.x / GROUP_SIZE;
    const int global_group = blockIdx.x * GROUPS_PER_BLOCK + my_group_in_block;
    const int total_groups = gridDim.x * GROUPS_PER_BLOCK;

    for (int k = threadIdx.x; k < num_clusters; k += BLOCK_SIZE) {
        int64_t sz = cluster_counts[k];
        s_inv[k] = (sz > 0) ? 1.0 / (double)sz : 0.0;
    }
    __syncthreads();

    double partial = 0.0;

    for (int u = global_group; u < num_vertices; u += total_groups) {
        int cluster_u = (int)__ldg(&cluster_assignments[u]);
        double inv_u = s_inv[cluster_u];

        int start = __ldg(&offsets[u]);
        int end = __ldg(&offsets[u + 1]);

        for (int e = start + lane_in_group; e < end; e += GROUP_SIZE) {
            int v = __ldg(&indices[e]);
            double w = __ldg(&edge_weights[e]);
            int cluster_v = (int)__ldg(&cluster_assignments[v]);
            if (cluster_u != cluster_v) {
                partial += w * inv_u;
            }
        }
    }

    
    if constexpr (GROUP_SIZE > 1) {
        #pragma unroll
        for (int offset = GROUP_SIZE / 2; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset, GROUP_SIZE);
    }

    if (lane_in_group == 0)
        group_partials[my_group_in_block] = partial;
    __syncthreads();

    if (threadIdx.x < 32) {
        double val = 0.0;
        for (int i = threadIdx.x; i < GROUPS_PER_BLOCK; i += 32)
            val += group_partials[i];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x == 0) atomicAdd(result, val);
    }
}




template<int GROUP_SIZE, typename ClusterT>
__global__ void ratio_cut_kernel_gmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const ClusterT* __restrict__ cluster_assignments,
    const double* __restrict__ inv_array,
    double* __restrict__ result,
    int num_vertices
) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int GROUPS_PER_BLOCK = BLOCK_SIZE / GROUP_SIZE;

    __shared__ double group_partials[GROUPS_PER_BLOCK];

    const int lane_in_group = threadIdx.x % GROUP_SIZE;
    const int my_group_in_block = threadIdx.x / GROUP_SIZE;
    const int global_group = blockIdx.x * GROUPS_PER_BLOCK + my_group_in_block;
    const int total_groups = gridDim.x * GROUPS_PER_BLOCK;

    double partial = 0.0;

    for (int u = global_group; u < num_vertices; u += total_groups) {
        int cluster_u = (int)__ldg(&cluster_assignments[u]);
        double inv_u = __ldg(&inv_array[cluster_u]);

        int start = __ldg(&offsets[u]);
        int end = __ldg(&offsets[u + 1]);

        for (int e = start + lane_in_group; e < end; e += GROUP_SIZE) {
            int v = __ldg(&indices[e]);
            double w = __ldg(&edge_weights[e]);
            int cluster_v = (int)__ldg(&cluster_assignments[v]);
            if (cluster_u != cluster_v) {
                partial += w * inv_u;
            }
        }
    }

    
    if constexpr (GROUP_SIZE > 1) {
        #pragma unroll
        for (int offset = GROUP_SIZE / 2; offset > 0; offset >>= 1)
            partial += __shfl_down_sync(0xffffffff, partial, offset, GROUP_SIZE);
    }

    if (lane_in_group == 0)
        group_partials[my_group_in_block] = partial;
    __syncthreads();

    if (threadIdx.x < 32) {
        double val = 0.0;
        for (int i = threadIdx.x; i < GROUPS_PER_BLOCK; i += 32)
            val += group_partials[i];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x == 0) atomicAdd(result, val);
    }
}




template<typename ClusterT>
static void dispatch_ratio_cut_kernels(
    const int32_t* offsets,
    const int32_t* indices,
    const double* edge_weights,
    const ClusterT* cluster_assignments,
    int64_t* counts,
    double* inv_or_null,
    double* result,
    int num_vertices,
    int num_edges,
    int num_clusters
) {
    bool use_smem = (num_clusters <= SMEM_CLUSTER_THRESHOLD);

    
    {
        size_t assign_bytes = (size_t)num_vertices * sizeof(ClusterT);
        size_t persist_bytes = (assign_bytes < 4u * 1024 * 1024) ? assign_bytes : 4u * 1024 * 1024;

        cudaStreamAttrValue attr;
        memset(&attr, 0, sizeof(attr));
        attr.accessPolicyWindow.base_ptr = const_cast<void*>((const void*)cluster_assignments);
        attr.accessPolicyWindow.num_bytes = persist_bytes;
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
    }

    
    if (!use_smem && inv_or_null != nullptr) {
        int inv_block = 256;
        int inv_grid = (num_clusters + inv_block - 1) / inv_block;
        if (inv_grid > 128) inv_grid = 128;
        compute_inv_kernel<<<inv_grid, inv_block>>>(counts, inv_or_null, num_clusters);
    }

    int avg_degree = (num_vertices > 0) ? (num_edges / num_vertices) : 0;
    int block = 256;

    auto compute_grid = [](int nv, int groups_per_block) -> int {
        int grid = (nv + groups_per_block - 1) / groups_per_block;
        if (grid > 1440) grid = 1440;
        return grid > 0 ? grid : 1;
    };

    if (use_smem) {
        if (avg_degree <= 4) {
            size_t smem_bytes = (size_t)num_clusters * sizeof(double) + (256 / 2) * sizeof(double);
            int grid = compute_grid(num_vertices, 256 / 2);
            ratio_cut_kernel_smem<2, ClusterT><<<grid, block, smem_bytes>>>(
                offsets, indices, edge_weights, cluster_assignments,
                counts, result, num_vertices, num_clusters);
        } else if (avg_degree <= 8) {
            size_t smem_bytes = (size_t)num_clusters * sizeof(double) + (256 / 4) * sizeof(double);
            int grid = compute_grid(num_vertices, 256 / 4);
            ratio_cut_kernel_smem<4, ClusterT><<<grid, block, smem_bytes>>>(
                offsets, indices, edge_weights, cluster_assignments,
                counts, result, num_vertices, num_clusters);
        } else {
            size_t smem_bytes = (size_t)num_clusters * sizeof(double) + (256 / 32) * sizeof(double);
            int grid = compute_grid(num_vertices, 256 / 32);
            ratio_cut_kernel_smem<32, ClusterT><<<grid, block, smem_bytes>>>(
                offsets, indices, edge_weights, cluster_assignments,
                counts, result, num_vertices, num_clusters);
        }
    } else {
        if (avg_degree <= 4) {
            int grid = compute_grid(num_vertices, 256 / 2);
            ratio_cut_kernel_gmem<2, ClusterT><<<grid, block>>>(
                offsets, indices, edge_weights, cluster_assignments,
                inv_or_null, result, num_vertices);
        } else if (avg_degree <= 8) {
            int grid = compute_grid(num_vertices, 256 / 4);
            ratio_cut_kernel_gmem<4, ClusterT><<<grid, block>>>(
                offsets, indices, edge_weights, cluster_assignments,
                inv_or_null, result, num_vertices);
        } else {
            int grid = compute_grid(num_vertices, 256 / 32);
            ratio_cut_kernel_gmem<32, ClusterT><<<grid, block>>>(
                offsets, indices, edge_weights, cluster_assignments,
                inv_or_null, result, num_vertices);
        }
    }

    
    {
        cudaStreamAttrValue attr;
        memset(&attr, 0, sizeof(attr));
        attr.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
    }
}

}  

double analyze_clustering_ratio_cut(const graph32_t& graph,
                                    const double* edge_weights,
                                    std::size_t num_clusters,
                                    const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int nv = graph.number_of_vertices;
    int ne = graph.number_of_edges;
    int nc = static_cast<int>(num_clusters);

    cache.ensure(nv, nc);

    double* inv_ptr = (nc > SMEM_CLUSTER_THRESHOLD) ? cache.inv_buf : nullptr;

    cudaMemsetAsync(cache.counts, 0, (size_t)nc * sizeof(int64_t));
    cudaMemsetAsync(cache.result_d, 0, sizeof(double));

    if (nc <= 32767) {
        
        bool use_smem = (nc <= SMEM_CLUSTER_THRESHOLD);
        int block = 256;
        int grid = (nv + block - 1) / block;
        if (grid > 640) grid = 640;
        if (use_smem) {
            size_t smem_bytes = (size_t)nc * sizeof(int);
            histogram_convert_kernel_smem<<<grid, block, smem_bytes>>>(
                cluster_assignments, cache.i16_buf, cache.counts, nv, nc);
        } else {
            histogram_convert_kernel_gmem<<<grid, block>>>(
                cluster_assignments, cache.i16_buf, cache.counts, nv, nc);
        }

        dispatch_ratio_cut_kernels<int16_t>(
            graph.offsets, graph.indices, edge_weights, cache.i16_buf,
            cache.counts, inv_ptr, cache.result_d, nv, ne, nc);
    } else {
        
        bool use_smem = (nc <= SMEM_CLUSTER_THRESHOLD);
        int block = 256;
        int grid = (nv + block - 1) / block;
        if (grid > 640) grid = 640;
        if (use_smem) {
            size_t smem_bytes = (size_t)nc * sizeof(int);
            histogram_kernel_smem<<<grid, block, smem_bytes>>>(
                cluster_assignments, cache.counts, nv, nc);
        } else {
            histogram_kernel_gmem<<<grid, block>>>(
                cluster_assignments, cache.counts, nv, nc);
        }

        dispatch_ratio_cut_kernels<int32_t>(
            graph.offsets, graph.indices, edge_weights, cluster_assignments,
            cache.counts, inv_ptr, cache.result_d, nv, ne, nc);
    }

    double result_host;
    cudaMemcpy(&result_host, cache.result_d, sizeof(double), cudaMemcpyDeviceToHost);
    return result_host;
}

}  
