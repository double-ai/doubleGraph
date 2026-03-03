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
    double* cluster_weights = nullptr;
    double* global_intra = nullptr;
    double* result_d = nullptr;
    int64_t cluster_weights_capacity = 0;
    bool fixed_allocated = false;

    void ensure(int num_clusters) {
        if (cluster_weights_capacity < num_clusters) {
            if (cluster_weights) cudaFree(cluster_weights);
            cudaMalloc(&cluster_weights, num_clusters * sizeof(double));
            cluster_weights_capacity = num_clusters;
        }
        if (!fixed_allocated) {
            cudaMalloc(&global_intra, sizeof(double));
            cudaMalloc(&result_d, sizeof(double));
            fixed_allocated = true;
        }
    }

    ~Cache() override {
        if (cluster_weights) cudaFree(cluster_weights);
        if (global_intra) cudaFree(global_intra);
        if (result_d) cudaFree(result_d);
    }
};

constexpr int BLOCK_SIZE = 256;
constexpr int MAX_SHARED_OFFSETS = 512;

__device__ __forceinline__ int32_t find_source_global(
    const int32_t* __restrict__ offsets,
    int32_t edge_idx,
    int32_t lo, int32_t hi
) {
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo + 1) >> 1);
        if (offsets[mid] <= edge_idx) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

__global__ void edge_modularity_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ cluster_weights,
    double* __restrict__ global_intra,
    int32_t num_vertices,
    int32_t num_edges,
    int num_clusters
) {
    int block_start = blockIdx.x * BLOCK_SIZE;
    int block_end = block_start + BLOCK_SIZE;
    if (block_end > num_edges) block_end = num_edges;
    if (block_start >= num_edges) return;

    
    __shared__ int32_t s_vtx_lo;
    __shared__ int32_t s_vtx_range;
    __shared__ int32_t s_offsets[MAX_SHARED_OFFSETS + 1];

    if (threadIdx.x == 0) {
        int32_t vlo = find_source_global(offsets, block_start, 0, num_vertices);
        int32_t vhi = find_source_global(offsets, block_end - 1, vlo, num_vertices);
        s_vtx_lo = vlo;
        s_vtx_range = vhi - vlo + 1;
    }
    __syncthreads();

    int32_t vtx_lo = s_vtx_lo;
    int32_t vtx_range = s_vtx_range;

    
    bool use_shared = (vtx_range <= MAX_SHARED_OFFSETS);

    if (use_shared) {
        
        for (int i = threadIdx.x; i <= vtx_range; i += BLOCK_SIZE) {
            s_offsets[i] = offsets[vtx_lo + i];
        }
        __syncthreads();
    }

    int edge_idx = block_start + threadIdx.x;

    double intra = 0.0;
    double w_val = 0.0;
    int32_t src_cluster = 0;
    bool valid = (edge_idx < num_edges);

    if (valid) {
        w_val = edge_weights[edge_idx];
        int32_t dst = indices[edge_idx];

        int32_t src;
        if (use_shared) {
            
            int lo = 0, hi = vtx_range;
            while (lo < hi) {
                int mid = lo + ((hi - lo + 1) >> 1);
                if (s_offsets[mid] <= edge_idx) lo = mid;
                else hi = mid - 1;
            }
            src = vtx_lo + lo;
        } else {
            
            src = find_source_global(offsets, edge_idx, vtx_lo, vtx_lo + vtx_range);
        }

        src_cluster = cluster_assignments[src];
        int32_t dst_cluster = cluster_assignments[dst];

        if (src_cluster == dst_cluster) {
            intra = w_val;
        }
    }

    
    unsigned active_mask = __ballot_sync(0xffffffff, valid);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        intra += __shfl_down_sync(0xffffffff, intra, offset);
    }

    
    int lane = threadIdx.x & 31;
    if (lane == 0 && intra != 0.0) {
        atomicAdd(global_intra, intra);
    }

    
    if (valid) {
        unsigned match = __match_any_sync(0xffffffff, src_cluster);

        
        double sum = w_val;
        unsigned scan_mask = match;

        
        int leader = __ffs(match) - 1;

        
        unsigned peers = match;
        double total = 0.0;
        while (peers) {
            int j = __ffs(peers) - 1;
            total += __shfl_sync(match, w_val, j);
            peers &= peers - 1;
        }

        if (lane == leader) {
            atomicAdd(&cluster_weights[src_cluster], total);
        }
    }
}

__global__ void finalize_kernel(
    const double* __restrict__ cluster_weights,
    const double* __restrict__ global_intra,
    double* __restrict__ result,
    int num_clusters
) {
    double M = 0.0;
    for (int c = 0; c < num_clusters; c++) {
        M += cluster_weights[c];
    }

    if (M == 0.0) {
        *result = 0.0;
        return;
    }

    double inv_M = 1.0 / M;
    double Q = *global_intra * inv_M;
    for (int c = 0; c < num_clusters; c++) {
        double Kc = cluster_weights[c] * inv_M;
        Q -= Kc * Kc;
    }
    *result = Q;
}

}  

double analyze_clustering_modularity(const graph32_t& graph,
                                     const double* edge_weights,
                                     std::size_t num_clusters,
                                     const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    int nc = static_cast<int>(num_clusters);

    cache.ensure(nc);

    cudaMemsetAsync(cache.cluster_weights, 0, nc * sizeof(double));
    cudaMemsetAsync(cache.global_intra, 0, sizeof(double));

    if (num_edges > 0) {
        int grid_size = (num_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
        edge_modularity_kernel<<<grid_size, BLOCK_SIZE>>>(
            graph.offsets, graph.indices, edge_weights, cluster_assignments,
            cache.cluster_weights, cache.global_intra, num_vertices, num_edges, nc
        );
    }

    finalize_kernel<<<1, 1>>>(cache.cluster_weights, cache.global_intra, cache.result_d, nc);

    double result_h;
    cudaMemcpy(&result_h, cache.result_d, sizeof(double), cudaMemcpyDeviceToHost);

    return result_h;
}

}  
