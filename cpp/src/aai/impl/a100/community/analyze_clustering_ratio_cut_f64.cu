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
#include <cub/block/block_reduce.cuh>

namespace aai {

namespace {





#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static constexpr int BLOCK = 256;
static constexpr int WARPS_PER_BLOCK = BLOCK / 32;
static constexpr int SMEM_HIST_MAX = 8192;  
static constexpr int MAX_BLOCKS = 1024;





struct Cache : Cacheable {
    int32_t* d_cluster_sizes = nullptr;
    double* d_inv_sizes = nullptr;
    double* d_partials = nullptr;
    unsigned int* d_retire_counter = nullptr;
    int64_t* d_num_clusters = nullptr;
    double* d_result = nullptr;

    std::size_t cluster_sizes_cap = 0;
    std::size_t inv_sizes_cap = 0;

    Cache() {
        cudaMalloc(&d_partials, MAX_BLOCKS * sizeof(double));
        cudaMalloc(&d_retire_counter, sizeof(unsigned int));
        cudaMemset(d_retire_counter, 0, sizeof(unsigned int));
        cudaMalloc(&d_num_clusters, sizeof(int64_t));
        cudaMalloc(&d_result, sizeof(double));
    }

    ~Cache() override {
        if (d_cluster_sizes) cudaFree(d_cluster_sizes);
        if (d_inv_sizes) cudaFree(d_inv_sizes);
        if (d_partials) cudaFree(d_partials);
        if (d_retire_counter) cudaFree(d_retire_counter);
        if (d_num_clusters) cudaFree(d_num_clusters);
        if (d_result) cudaFree(d_result);
    }

    void ensure(std::size_t num_clusters) {
        if (cluster_sizes_cap < num_clusters) {
            if (d_cluster_sizes) cudaFree(d_cluster_sizes);
            cluster_sizes_cap = num_clusters * 2;
            if (cluster_sizes_cap < 65536) cluster_sizes_cap = 65536;
            cudaMalloc(&d_cluster_sizes, cluster_sizes_cap * sizeof(int32_t));
        }
        if (inv_sizes_cap < num_clusters) {
            if (d_inv_sizes) cudaFree(d_inv_sizes);
            inv_sizes_cap = num_clusters * 2;
            if (inv_sizes_cap < 65536) inv_sizes_cap = 65536;
            cudaMalloc(&d_inv_sizes, inv_sizes_cap * sizeof(double));
        }
    }
};





__device__ __forceinline__ int lane_id() { return threadIdx.x & 31; }


__device__ __forceinline__ int32_t find_src_vertex(const int32_t* __restrict__ offsets,
                                                   int32_t e,
                                                   int32_t num_vertices) {
    int32_t lo = 0, hi = num_vertices;
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo + 1) >> 1);
        int32_t off = __ldg(offsets + mid);
        if (off <= e)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}




__global__ __launch_bounds__(BLOCK, 4)
void compute_cluster_sizes_kernel(const int32_t* __restrict__ cluster_assignments,
                                  const int64_t* __restrict__ d_num_clusters,
                                  int32_t num_vertices,
                                  int32_t* __restrict__ cluster_sizes) {
    __shared__ int32_t s_hist[SMEM_HIST_MAX];
    const int num_clusters = (int)(*d_num_clusters);

    if (num_clusters <= SMEM_HIST_MAX) {
        for (int i = threadIdx.x; i < num_clusters; i += BLOCK) s_hist[i] = 0;
        __syncthreads();

        int idx = (int)(blockIdx.x * BLOCK + threadIdx.x);
        int stride = (int)(gridDim.x * BLOCK);
        for (int v = idx; v < num_vertices; v += stride) {
            int c = __ldg(cluster_assignments + v);
            atomicAdd(&s_hist[c], 1);
        }
        __syncthreads();

        for (int i = threadIdx.x; i < num_clusters; i += BLOCK) {
            int32_t val = s_hist[i];
            if (val) atomicAdd(cluster_sizes + i, val);
        }
    } else {
#if __CUDA_ARCH__ >= 700
        int idx = (int)(blockIdx.x * BLOCK + threadIdx.x);
        int stride = (int)(gridDim.x * BLOCK);
        for (int v = idx; v < num_vertices; v += stride) {
            int c = __ldg(cluster_assignments + v);
            unsigned am = __activemask();
            unsigned m = __match_any_sync(am, c);
            int leader = __ffs(m) - 1;
            if (lane_id() == leader) {
                atomicAdd(cluster_sizes + c, __popc(m));
            }
        }
#else
        int idx = (int)(blockIdx.x * BLOCK + threadIdx.x);
        int stride = (int)(gridDim.x * BLOCK);
        for (int v = idx; v < num_vertices; v += stride) {
            int c = __ldg(cluster_assignments + v);
            atomicAdd(cluster_sizes + c, 1);
        }
#endif
    }
}

__global__ void compute_inv_sizes_kernel(const int32_t* __restrict__ cluster_sizes,
                                        const int64_t* __restrict__ d_num_clusters,
                                        double* __restrict__ inv_sizes) {
    const int num_clusters = (int)(*d_num_clusters);
    for (int k = (int)(blockIdx.x * blockDim.x + threadIdx.x); k < num_clusters;
         k += (int)(gridDim.x * blockDim.x)) {
        int32_t sz = __ldg(cluster_sizes + k);
        inv_sizes[k] = (sz > 0) ? (1.0 / (double)sz) : 0.0;
    }
}




__device__ __forceinline__ void block_retire_sum(double block_sum,
                                                 double* __restrict__ out,
                                                 double* __restrict__ partials,
                                                 unsigned int* __restrict__ retire_counter) {
    if (threadIdx.x == 0) partials[blockIdx.x] = block_sum;
    __threadfence();

    __shared__ int am_last;
    __shared__ int nblocks;
    if (threadIdx.x == 0) {
        nblocks = gridDim.x;
        unsigned int ticket = atomicAdd(retire_counter, 1);
        am_last = (ticket == (unsigned int)(gridDim.x - 1));
    }
    __syncthreads();

    if (am_last) {
        
    }
}




template <int UNROLL>
__global__ __launch_bounds__(BLOCK, 3)
void ratio_cut_edge_warp_retire(const int32_t* __restrict__ offsets,
                                const int32_t* __restrict__ indices,
                                const double* __restrict__ weights,
                                const int32_t* __restrict__ cluster_assignments,
                                const double* __restrict__ inv_sizes,
                                int32_t num_vertices,
                                int32_t num_edges,
                                double* __restrict__ out,
                                double* __restrict__ partials,
                                unsigned int* __restrict__ retire_counter) {
    double sum = 0.0;

    const int lane = lane_id();
    const int global_warp = (int)((blockIdx.x * BLOCK + threadIdx.x) >> 5);
    const int num_warps = (int)((gridDim.x * BLOCK) >> 5);

    const int32_t edges_per_warp = (num_edges + num_warps - 1) / num_warps;
    const int32_t warp_e0 = global_warp * edges_per_warp;
    int32_t warp_e1 = warp_e0 + edges_per_warp;
    if (warp_e1 > num_edges) warp_e1 = num_edges;

    if (warp_e0 < num_edges) {
        int32_t u0 = 0;
        if (lane == 0) u0 = find_src_vertex(offsets, warp_e0, num_vertices);
        u0 = __shfl_sync(0xffffffff, u0, 0);

        int32_t e = warp_e0 + lane;
        int32_t u = u0;
        if (e < warp_e1) {
            int32_t u_end = __ldg(offsets + u + 1);
            while (u_end <= e && u < num_vertices - 1) {
                ++u;
                u_end = __ldg(offsets + u + 1);
            }
        }

        int32_t u_end_edge = (e < warp_e1) ? __ldg(offsets + u + 1) : num_edges;
        int32_t c_u = (e < warp_e1) ? __ldg(cluster_assignments + u) : 0;
        double inv_u = (e < warp_e1) ? __ldg(inv_sizes + c_u) : 0.0;

        for (int32_t ee = e; ee < warp_e1; ee += 32 * UNROLL) {
#pragma unroll
            for (int it = 0; it < UNROLL; ++it) {
                int32_t eid = ee + (int32_t)(it * 32);
                if (eid >= warp_e1) break;

                while (eid >= u_end_edge && u < num_vertices - 1) {
                    ++u;
                    u_end_edge = __ldg(offsets + u + 1);
                    c_u = __ldg(cluster_assignments + u);
                    inv_u = __ldg(inv_sizes + c_u);
                }

                int32_t v = __ldg(indices + eid);
                int32_t c_v = __ldg(cluster_assignments + v);
                if (c_u != c_v) sum += __ldg(weights + eid) * inv_u;
            }
        }
    }

    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    double block_sum = BlockReduce(temp).Sum(sum);

    
    if (threadIdx.x == 0) partials[blockIdx.x] = block_sum;
    __threadfence();

    __shared__ int am_last;
    __shared__ int nblocks;
    if (threadIdx.x == 0) {
        nblocks = gridDim.x;
        unsigned int ticket = atomicAdd(retire_counter, 1);
        am_last = (ticket == (unsigned int)(gridDim.x - 1));
    }
    __syncthreads();

    if (am_last) {
        double local = 0.0;
        for (int i = threadIdx.x; i < nblocks; i += BLOCK) local += partials[i];
        double final_sum = BlockReduce(temp).Sum(local);
        if (threadIdx.x == 0) {
            out[0] = final_sum;
            *retire_counter = 0;
        }
    }
}




__global__ __launch_bounds__(BLOCK, 3)
void ratio_cut_vertex_retire(const int32_t* __restrict__ offsets,
                             const int32_t* __restrict__ indices,
                             const double* __restrict__ weights,
                             const int32_t* __restrict__ cluster_assignments,
                             const double* __restrict__ inv_sizes,
                             int32_t num_vertices,
                             double* __restrict__ out,
                             double* __restrict__ partials,
                             unsigned int* __restrict__ retire_counter) {
    double sum = 0.0;

    int idx = (int)(blockIdx.x * BLOCK + threadIdx.x);
    int stride = (int)(gridDim.x * BLOCK);

    for (int32_t u = idx; u < num_vertices; u += stride) {
        int32_t c_u = __ldg(cluster_assignments + u);
        double inv_u = __ldg(inv_sizes + c_u);
        int32_t start = __ldg(offsets + u);
        int32_t end = __ldg(offsets + u + 1);

        double cut = 0.0;
        for (int32_t e = start; e < end; ++e) {
            int32_t v = __ldg(indices + e);
            int32_t c_v = __ldg(cluster_assignments + v);
            if (c_u != c_v) cut += __ldg(weights + e);
        }
        sum += cut * inv_u;
    }

    using BlockReduce = cub::BlockReduce<double, BLOCK>;
    __shared__ typename BlockReduce::TempStorage temp;
    double block_sum = BlockReduce(temp).Sum(sum);

    if (threadIdx.x == 0) partials[blockIdx.x] = block_sum;
    __threadfence();

    __shared__ int am_last;
    __shared__ int nblocks;
    if (threadIdx.x == 0) {
        nblocks = gridDim.x;
        unsigned int ticket = atomicAdd(retire_counter, 1);
        am_last = (ticket == (unsigned int)(gridDim.x - 1));
    }
    __syncthreads();

    if (am_last) {
        double local = 0.0;
        for (int i = threadIdx.x; i < nblocks; i += BLOCK) local += partials[i];
        double final_sum = BlockReduce(temp).Sum(local);
        if (threadIdx.x == 0) {
            out[0] = final_sum;
            *retire_counter = 0;
        }
    }
}

}  





double analyze_clustering_ratio_cut(const graph32_t& graph,
                                    const double* edge_weights,
                                    std::size_t num_clusters,
                                    const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    cache.ensure(num_clusters);

    
    int64_t nc = static_cast<int64_t>(num_clusters);
    cudaMemcpyAsync(cache.d_num_clusters, &nc, sizeof(int64_t), cudaMemcpyHostToDevice);

    cudaMemsetAsync(cache.d_cluster_sizes, 0,
                    static_cast<std::size_t>(num_clusters) * sizeof(int32_t));
    cudaMemsetAsync(cache.d_retire_counter, 0, sizeof(unsigned int));

    
    {
        int grid = (num_vertices + BLOCK - 1) / BLOCK;
        if (grid > 1024) grid = 1024;
        compute_cluster_sizes_kernel<<<grid, BLOCK>>>(
            cluster_assignments, cache.d_num_clusters, num_vertices, cache.d_cluster_sizes);
    }

    
    {
        int block = 256;
        int grid = 1024;
        compute_inv_sizes_kernel<<<grid, block>>>(
            cache.d_cluster_sizes, cache.d_num_clusters, cache.d_inv_sizes);
    }

    
    
    double avg_deg = (num_vertices > 0) ? ((double)num_edges / (double)num_vertices) : 0.0;

    if (avg_deg < 6.0) {
        int grid = (num_vertices + BLOCK - 1) / BLOCK;
        if (grid < 80) grid = 80;
        if (grid > 1024) grid = 1024;
        ratio_cut_vertex_retire<<<grid, BLOCK>>>(
            offsets, indices, edge_weights, cluster_assignments,
            cache.d_inv_sizes, num_vertices, cache.d_result,
            cache.d_partials, cache.d_retire_counter);
    } else {
        
        constexpr int edges_per_thread_target = 256;
        int64_t denom = (int64_t)BLOCK * (int64_t)edges_per_thread_target;
        int grid = (int)((num_edges + denom - 1) / denom);
        if (grid < 80) grid = 80;
        if (grid > 1024) grid = 1024;
        ratio_cut_edge_warp_retire<2><<<grid, BLOCK>>>(
            offsets, indices, edge_weights, cluster_assignments,
            cache.d_inv_sizes, num_vertices, num_edges, cache.d_result,
            cache.d_partials, cache.d_retire_counter);
    }

    
    double result;
    cudaMemcpy(&result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

}  
