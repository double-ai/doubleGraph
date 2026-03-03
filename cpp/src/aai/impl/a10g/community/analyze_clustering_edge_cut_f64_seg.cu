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
    int16_t* cluster_i16 = nullptr;
    double* result_d = nullptr;
    int64_t cluster_i16_capacity = 0;
    bool result_allocated = false;
    bool l2_configured = false;

    void ensure(int32_t num_vertices, bool need_i16) {
        if (!result_allocated) {
            cudaMalloc(&result_d, sizeof(double));
            result_allocated = true;
        }
        if (need_i16 && cluster_i16_capacity < num_vertices) {
            if (cluster_i16) cudaFree(cluster_i16);
            cudaMalloc(&cluster_i16, (size_t)num_vertices * sizeof(int16_t));
            cluster_i16_capacity = num_vertices;
        }
        if (!l2_configured) {
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 4 * 1024 * 1024);
            l2_configured = true;
        }
    }

    ~Cache() override {
        if (cluster_i16) cudaFree(cluster_i16);
        if (result_d) cudaFree(result_d);
    }
};


__global__ void convert_clusters_kernel(
    const int32_t* __restrict__ src,
    int16_t* __restrict__ dst,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (int16_t)src[i];
}


__device__ __forceinline__ int find_vertex_global(
    int edge_idx, const int32_t* offsets, int num_vertices)
{
    int lo = 0, hi = num_vertices;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (offsets[mid] <= edge_idx) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}


template <typename ClusterT, int BLOCK_SIZE = 256, int ITEMS_PER_THREAD = 4>
__global__ __launch_bounds__(BLOCK_SIZE)
void edge_cut_flat(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const ClusterT* __restrict__ cluster,
    double* __restrict__ global_sum,
    int num_edges,
    int num_vertices)
{
    constexpr int EDGES_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;

    int block_edge_start = blockIdx.x * EDGES_PER_BLOCK;
    if (block_edge_start >= num_edges) return;
    int block_edge_end = block_edge_start + EDGES_PER_BLOCK;
    if (block_edge_end > num_edges) block_edge_end = num_edges;

    __shared__ int s_v_start;
    __shared__ int s_num_verts;

    if (threadIdx.x == 0) {
        s_v_start = find_vertex_global(block_edge_start, offsets, num_vertices);
        int v_end = find_vertex_global(block_edge_end - 1, offsets, num_vertices) + 1;
        s_num_verts = v_end - s_v_start;
    }
    __syncthreads();

    int v_start = s_v_start;
    int num_block_verts = s_num_verts;

    
    union SharedStorage {
        int offsets_buf[EDGES_PER_BLOCK + 2];
        typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage reduce;
    };
    __shared__ SharedStorage shared;

    for (int i = threadIdx.x; i <= num_block_verts; i += BLOCK_SIZE) {
        shared.offsets_buf[i] = offsets[v_start + i];
    }
    __syncthreads();

    double local_sum = 0.0;
    int hint = 0;

    #pragma unroll
    for (int t = 0; t < ITEMS_PER_THREAD; t++) {
        int e = block_edge_start + t * BLOCK_SIZE + threadIdx.x;
        if (e >= block_edge_end) break;

        
        int lo = hint, hi = num_block_verts;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (shared.offsets_buf[mid] <= e) lo = mid;
            else hi = mid - 1;
        }
        hint = lo;

        int src = v_start + lo;
        int dst_v = indices[e];

        if (cluster[src] != cluster[dst_v]) {
            local_sum += edge_weights[e];
        }
    }

    __syncthreads();

    double block_sum = cub::BlockReduce<double, BLOCK_SIZE>(shared.reduce).Sum(local_sum);

    
    if (threadIdx.x == 0) {
        atomicAdd(global_sum, block_sum * 0.5);
    }
}

}  

double analyze_clustering_edge_cut_seg(const graph32_t& graph,
                                       const double* edge_weights,
                                       std::size_t num_clusters,
                                       const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    bool use_i16 = (num_vertices > 1000000 && num_clusters <= 32767);

    cache.ensure(num_vertices, use_i16);

    
    cudaMemsetAsync(cache.result_d, 0, sizeof(double));

    if (num_edges <= 0) {
        double host_result = 0.0;
        return host_result;
    }

    constexpr int BLOCK_SIZE = 256;
    constexpr int IPT = 4;
    constexpr int EPB = BLOCK_SIZE * IPT;
    int grid = (num_edges + EPB - 1) / EPB;

    if (use_i16) {
        
        {
            int cgrid = (num_vertices + 255) / 256;
            convert_clusters_kernel<<<cgrid, 256>>>(cluster_assignments, cache.cluster_i16, num_vertices);
        }

        
        size_t cluster_bytes = (size_t)num_vertices * sizeof(int16_t);
        if (cluster_bytes <= 4ULL * 1024 * 1024) {
            cudaStreamAttrValue attr = {};
            attr.accessPolicyWindow.base_ptr = (void*)cache.cluster_i16;
            attr.accessPolicyWindow.num_bytes = cluster_bytes;
            attr.accessPolicyWindow.hitRatio = 1.0f;
            attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
        }

        edge_cut_flat<int16_t, BLOCK_SIZE, IPT><<<grid, BLOCK_SIZE>>>(
            graph.offsets, graph.indices, edge_weights, cache.cluster_i16,
            cache.result_d, num_edges, num_vertices);

        
        {
            cudaStreamAttrValue attr = {};
            attr.accessPolicyWindow.num_bytes = 0;
            cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
        }
    } else {
        
        size_t cluster_bytes = (size_t)num_vertices * sizeof(int32_t);
        if (cluster_bytes <= 4ULL * 1024 * 1024) {
            cudaStreamAttrValue attr = {};
            attr.accessPolicyWindow.base_ptr = (void*)cluster_assignments;
            attr.accessPolicyWindow.num_bytes = cluster_bytes;
            attr.accessPolicyWindow.hitRatio = 1.0f;
            attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
            attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
            cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
        }

        edge_cut_flat<int32_t, BLOCK_SIZE, IPT><<<grid, BLOCK_SIZE>>>(
            graph.offsets, graph.indices, edge_weights, cluster_assignments,
            cache.result_d, num_edges, num_vertices);

        
        {
            cudaStreamAttrValue attr = {};
            attr.accessPolicyWindow.num_bytes = 0;
            cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
        }
    }

    double host_result;
    cudaMemcpy(&host_result, cache.result_d, sizeof(double), cudaMemcpyDeviceToHost);
    return host_result;
}

}  
