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
#include <algorithm>

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)

static constexpr int MAX_BLOCKS = 8192;

struct Cache : Cacheable {
    int32_t* d_sizes = nullptr;
    double* d_inv_sizes = nullptr;
    double* d_block_results = nullptr;
    uint8_t* d_clusters_u8 = nullptr;
    double* d_result = nullptr;

    int32_t sizes_capacity = 0;
    int32_t inv_sizes_capacity = 0;
    int32_t clusters_u8_capacity = 0;

    Cache() {
        cudaMalloc(&d_block_results, MAX_BLOCKS * sizeof(double));
        cudaMalloc(&d_result, sizeof(double));
    }

    void ensure_cluster_alloc(int32_t num_clusters) {
        if (num_clusters > sizes_capacity) {
            if (d_sizes) cudaFree(d_sizes);
            cudaMalloc(&d_sizes, num_clusters * sizeof(int32_t));
            sizes_capacity = num_clusters;
        }
        if (num_clusters > inv_sizes_capacity) {
            if (d_inv_sizes) cudaFree(d_inv_sizes);
            cudaMalloc(&d_inv_sizes, num_clusters * sizeof(double));
            inv_sizes_capacity = num_clusters;
        }
    }

    void ensure_u8_alloc(int32_t num_vertices) {
        if (num_vertices > clusters_u8_capacity) {
            if (d_clusters_u8) cudaFree(d_clusters_u8);
            cudaMalloc(&d_clusters_u8, num_vertices * sizeof(uint8_t));
            clusters_u8_capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (d_sizes) cudaFree(d_sizes);
        if (d_inv_sizes) cudaFree(d_inv_sizes);
        if (d_block_results) cudaFree(d_block_results);
        if (d_clusters_u8) cudaFree(d_clusters_u8);
        if (d_result) cudaFree(d_result);
    }
};





__device__ __forceinline__ double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}


__global__ void preprocess_kernel(
    const int32_t* __restrict__ clusters_in,
    uint8_t* __restrict__ clusters_out,
    int32_t* __restrict__ sizes,
    int32_t num_vertices,
    int32_t num_clusters)
{
    extern __shared__ int32_t s_hist[];
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x)
        s_hist[i] = 0;
    __syncthreads();

    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices;
         v += gridDim.x * blockDim.x) {
        int32_t c = clusters_in[v];
        clusters_out[v] = (uint8_t)c;
        atomicAdd(&s_hist[c], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        if (s_hist[i] > 0)
            atomicAdd(&sizes[i], s_hist[i]);
    }
}

__global__ void compute_inv_sizes_kernel(
    const int32_t* __restrict__ sizes,
    double* __restrict__ inv_sizes,
    int32_t num_clusters)
{
    for (int k = threadIdx.x; k < num_clusters; k += blockDim.x) {
        int sz = sizes[k];
        inv_sizes[k] = (sz > 0) ? (1.0 / (double)sz) : 0.0;
    }
}


template<typename ClusterT>
__global__ void edge_contiguous_ratio_cut_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const ClusterT* __restrict__ clusters,
    const double* __restrict__ inv_sizes,
    double* __restrict__ block_results,
    int32_t num_vertices,
    int32_t num_edges)
{
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int global_warp = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    int total_warps = gridDim.x * WARPS_PER_BLOCK;

    
    int64_t total_chunks = ((int64_t)num_edges + 31) >> 5;
    int64_t chunks_per_warp = (total_chunks + total_warps - 1) / total_warps;
    int64_t my_start_chunk = (int64_t)global_warp * chunks_per_warp;
    int64_t my_end_chunk = my_start_chunk + chunks_per_warp;
    if (my_end_chunk > total_chunks) my_end_chunk = total_chunks;

    int64_t my_start_edge = my_start_chunk << 5;
    int64_t my_end_edge = my_end_chunk << 5;
    if (my_end_edge > num_edges) my_end_edge = num_edges;

    double local_sum = 0.0;

    if (my_start_edge >= num_edges) goto reduce;

    {
        
        int v;
        if (lane == 0) {
            int lo = 0, hi = num_vertices;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (offsets[mid + 1] <= (int)my_start_edge)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            v = lo;
        }
        v = __shfl_sync(0xffffffff, v, 0);

        int v_end_edge = offsets[v + 1];

        
        for (int64_t base_e = my_start_edge; base_e < my_end_edge; base_e += 32) {
            
            if (lane == 0) {
                while (v_end_edge <= (int)base_e) {
                    v++;
                    v_end_edge = offsets[v + 1];
                }
            }
            v = __shfl_sync(0xffffffff, v, 0);
            v_end_edge = __shfl_sync(0xffffffff, v_end_edge, 0);

            int64_t e = base_e + lane;

            if (e < my_end_edge && e < num_edges) {
                
                int my_v = v;
                int my_v_end = v_end_edge;
                while (my_v_end <= (int)e) {
                    my_v++;
                    my_v_end = __ldg(&offsets[my_v + 1]);
                }

                int32_t c_src = (int32_t)clusters[my_v];
                int32_t dst = __ldcs(&indices[e]);  
                int32_t c_dst = (int32_t)clusters[dst];

                if (c_src != c_dst) {
                    local_sum += __ldcs(&weights[e]) * inv_sizes[c_src];
                }
            }
        }
    }

reduce:
    
    local_sum = warp_reduce_sum(local_sum);

    __shared__ double warp_sums[WARPS_PER_BLOCK];
    if (lane == 0) warp_sums[warp_in_block] = local_sum;
    __syncthreads();

    if (threadIdx.x < 32) {
        double val = (threadIdx.x < WARPS_PER_BLOCK) ? warp_sums[threadIdx.x] : 0.0;
        val = warp_reduce_sum(val);
        if (threadIdx.x == 0) {
            block_results[blockIdx.x] = val;
        }
    }
}


__global__ void sum_blocks_kernel(
    const double* __restrict__ block_results,
    double* __restrict__ result,
    int32_t num_blocks)
{
    double sum = 0.0;
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x)
        sum += block_results[i];

    sum = warp_reduce_sum(sum);

    __shared__ double warp_sums[WARPS_PER_BLOCK];
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (threadIdx.x < 32) {
        double val = (threadIdx.x < WARPS_PER_BLOCK) ? warp_sums[threadIdx.x] : 0.0;
        val = warp_reduce_sum(val);
        if (threadIdx.x == 0)
            *result = val;
    }
}


__global__ void cluster_sizes_kernel(
    const int32_t* __restrict__ clusters,
    int32_t* __restrict__ sizes,
    int32_t num_vertices,
    int32_t num_clusters)
{
    extern __shared__ int32_t s_hist2[];
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x)
        s_hist2[i] = 0;
    __syncthreads();
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices;
         v += gridDim.x * blockDim.x)
        atomicAdd(&s_hist2[clusters[v]], 1);
    __syncthreads();
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x)
        if (s_hist2[i] > 0)
            atomicAdd(&sizes[i], s_hist2[i]);
}


__global__ void cluster_sizes_kernel_global(
    const int32_t* __restrict__ clusters,
    int32_t* __restrict__ sizes,
    int32_t num_vertices)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices;
         v += gridDim.x * blockDim.x)
        atomicAdd(&sizes[clusters[v]], 1);
}

static constexpr int SMEM_CLUSTER_THRESH = 12000; 

}  

double analyze_clustering_ratio_cut_seg(const graph32_t& graph,
                                        const double* edge_weights,
                                        std::size_t num_clusters,
                                        const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    int32_t nc = static_cast<int32_t>(num_clusters);

    cache.ensure_cluster_alloc(nc);

    cudaStream_t stream = 0;

    cudaMemsetAsync(cache.d_sizes, 0, nc * sizeof(int32_t), stream);

    
    int num_blocks = std::max(480, std::min((int)MAX_BLOCKS, (num_edges + 255) / 256));

    bool use_u8 = (nc <= 256);

    if (use_u8) {
        cache.ensure_u8_alloc(num_vertices);

        int g = std::min(512, (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE);
        preprocess_kernel<<<g, BLOCK_SIZE, nc * sizeof(int32_t), stream>>>(
            cluster_assignments, cache.d_clusters_u8, cache.d_sizes, num_vertices, nc);
        compute_inv_sizes_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
            cache.d_sizes, cache.d_inv_sizes, nc);
        edge_contiguous_ratio_cut_kernel<uint8_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            d_offsets, d_indices, edge_weights, cache.d_clusters_u8,
            cache.d_inv_sizes, cache.d_block_results,
            num_vertices, num_edges);
    } else {
        int g = std::min(512, (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE);
        if (nc <= SMEM_CLUSTER_THRESH) {
            cluster_sizes_kernel<<<g, BLOCK_SIZE, nc * sizeof(int32_t), stream>>>(
                cluster_assignments, cache.d_sizes, num_vertices, nc);
        } else {
            cluster_sizes_kernel_global<<<g, BLOCK_SIZE, 0, stream>>>(
                cluster_assignments, cache.d_sizes, num_vertices);
        }
        compute_inv_sizes_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
            cache.d_sizes, cache.d_inv_sizes, nc);
        edge_contiguous_ratio_cut_kernel<int32_t><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            d_offsets, d_indices, edge_weights, cluster_assignments,
            cache.d_inv_sizes, cache.d_block_results,
            num_vertices, num_edges);
    }

    sum_blocks_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        cache.d_block_results, cache.d_result, num_blocks);

    double result;
    cudaMemcpy(&result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

}  
