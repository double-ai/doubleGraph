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
#include <cstddef>

namespace aai {

namespace {

struct Cache : Cacheable {
    double* d_sum_deg = nullptr;
    double* d_intra_sum = nullptr;
    double* d_result = nullptr;
    std::size_t sum_deg_capacity = 0;

    Cache() {
        cudaMalloc(&d_intra_sum, sizeof(double));
        cudaMalloc(&d_result, sizeof(double));
    }

    void ensure_sum_deg(std::size_t num_clusters) {
        if (sum_deg_capacity < num_clusters) {
            if (d_sum_deg) cudaFree(d_sum_deg);
            sum_deg_capacity = num_clusters * 2;
            cudaMalloc(&d_sum_deg, sum_deg_capacity * sizeof(double));
        }
    }

    ~Cache() override {
        if (d_sum_deg) cudaFree(d_sum_deg);
        if (d_intra_sum) cudaFree(d_intra_sum);
        if (d_result) cudaFree(d_result);
    }
};




template <int BLOCK_SIZE>
__global__ void modularity_high_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster,
    double* __restrict__ sum_deg,
    double* __restrict__ intra_sum,
    int32_t start_vertex,
    int32_t end_vertex)
{
    int32_t v = blockIdx.x + start_vertex;
    if (v >= end_vertex) return;

    int32_t row_start = offsets[v];
    int32_t row_end = offsets[v + 1];
    int32_t cv = cluster[v];

    double local_deg = 0.0;
    double local_intra = 0.0;

    for (int32_t e = row_start + threadIdx.x; e < row_end; e += BLOCK_SIZE) {
        double w = (double)edge_weights[e];
        local_deg += w;
        if (cluster[indices[e]] == cv) {
            local_intra += w;
        }
    }

    
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_deg += __shfl_down_sync(mask, local_deg, offset);
        local_intra += __shfl_down_sync(mask, local_intra, offset);
    }

    constexpr int NWARPS = BLOCK_SIZE / 32;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    __shared__ double s_deg[NWARPS];
    __shared__ double s_intra[NWARPS];

    if (lane == 0) {
        s_deg[warp_id] = local_deg;
        s_intra[warp_id] = local_intra;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double total_deg = 0.0;
        double total_intra = 0.0;
        for (int i = 0; i < NWARPS; i++) {
            total_deg += s_deg[i];
            total_intra += s_intra[i];
        }
        atomicAdd(&sum_deg[cv], total_deg);
        atomicAdd(intra_sum, total_intra);
    }
}




template <int BLOCK_SIZE>
__global__ void modularity_mid_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster,
    double* __restrict__ sum_deg,
    double* __restrict__ intra_sum,
    int32_t start_vertex,
    int32_t end_vertex)
{
    constexpr int NWARPS = BLOCK_SIZE / 32;
    int32_t lane = threadIdx.x & 31;
    int32_t warp_in_block = threadIdx.x >> 5;
    int32_t global_warp = blockIdx.x * NWARPS + warp_in_block;
    int32_t v = global_warp + start_vertex;
    if (v >= end_vertex) return;

    int32_t row_start = offsets[v];
    int32_t row_end = offsets[v + 1];
    int32_t cv = cluster[v];

    double local_deg = 0.0;
    double local_intra = 0.0;

    for (int32_t e = row_start + lane; e < row_end; e += 32) {
        double w = (double)edge_weights[e];
        local_deg += w;
        if (cluster[indices[e]] == cv) {
            local_intra += w;
        }
    }

    
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_deg += __shfl_down_sync(mask, local_deg, offset);
        local_intra += __shfl_down_sync(mask, local_intra, offset);
    }

    if (lane == 0) {
        atomicAdd(&sum_deg[cv], local_deg);
        atomicAdd(intra_sum, local_intra);
    }
}





template <int BLOCK_SIZE>
__global__ void modularity_low_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster,
    double* __restrict__ sum_deg,
    double* __restrict__ intra_sum,
    int32_t start_vertex,
    int32_t end_vertex,
    int num_clusters)
{
    constexpr int NWARPS = BLOCK_SIZE / 32;
    extern __shared__ double smem[];
    double* s_sum_deg = smem;              
    double* s_warp_intra = smem + num_clusters; 

    
    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        s_sum_deg[i] = 0.0;
    }
    if (threadIdx.x < NWARPS) {
        s_warp_intra[threadIdx.x] = 0.0;
    }
    __syncthreads();

    int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x + start_vertex;

    double local_intra = 0.0;

    if (v < end_vertex) {
        int32_t row_start = offsets[v];
        int32_t row_end = offsets[v + 1];
        int32_t cv = cluster[v];
        double local_deg = 0.0;

        for (int32_t e = row_start; e < row_end; e++) {
            double w = (double)edge_weights[e];
            local_deg += w;
            if (cluster[indices[e]] == cv) {
                local_intra += w;
            }
        }

        
        atomicAdd(&s_sum_deg[cv], local_deg);
    }

    
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_intra += __shfl_down_sync(mask, local_intra, offset);
    }

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        s_warp_intra[warp_id] = local_intra;
    }
    __syncthreads();

    
    if (threadIdx.x == 0) {
        double block_intra = 0.0;
        for (int w = 0; w < NWARPS; w++) {
            block_intra += s_warp_intra[w];
        }
        if (block_intra != 0.0) {
            atomicAdd(intra_sum, block_intra);
        }
    }

    
    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        double val = s_sum_deg[i];
        if (val != 0.0) {
            atomicAdd(&sum_deg[i], val);
        }
    }
}




template <int BLOCK_SIZE>
__global__ void modularity_low_degree_noshmem_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster,
    double* __restrict__ sum_deg,
    double* __restrict__ intra_sum,
    int32_t start_vertex,
    int32_t end_vertex)
{
    constexpr int NWARPS = BLOCK_SIZE / 32;

    int32_t v = blockIdx.x * BLOCK_SIZE + threadIdx.x + start_vertex;

    double local_intra = 0.0;
    double local_deg = 0.0;
    int32_t cv = -1;

    if (v < end_vertex) {
        int32_t row_start = offsets[v];
        int32_t row_end = offsets[v + 1];
        cv = cluster[v];

        for (int32_t e = row_start; e < row_end; e++) {
            double w = (double)edge_weights[e];
            local_deg += w;
            if (cluster[indices[e]] == cv) {
                local_intra += w;
            }
        }
    }

    
    unsigned mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_intra += __shfl_down_sync(mask, local_intra, offset);
    }

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    __shared__ double s_warp_intra[NWARPS];
    if (lane == 0) {
        s_warp_intra[warp_id] = local_intra;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double block_intra = 0.0;
        for (int w = 0; w < NWARPS; w++) {
            block_intra += s_warp_intra[w];
        }
        if (block_intra != 0.0) {
            atomicAdd(intra_sum, block_intra);
        }
    }

    
    if (cv >= 0 && local_deg != 0.0) {
        atomicAdd(&sum_deg[cv], local_deg);
    }
}




__global__ void compute_modularity_kernel(
    const double* __restrict__ sum_deg,
    const double* __restrict__ intra_sum,
    double* __restrict__ result,
    int num_clusters)
{
    double total_weight = 0.0;
    for (int c = 0; c < num_clusters; c++) {
        total_weight += sum_deg[c];
    }

    if (total_weight == 0.0) {
        *result = 0.0;
        return;
    }

    double inv_total = 1.0 / total_weight;
    double Q = (*intra_sum) * inv_total;

    for (int c = 0; c < num_clusters; c++) {
        double ac = sum_deg[c] * inv_total;
        Q -= ac * ac;
    }

    *result = Q;
}

}  

double analyze_clustering_modularity_seg(const graph32_t& graph,
                                         const float* edge_weights,
                                         std::size_t num_clusters,
                                         const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    cache.ensure_sum_deg(num_clusters);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0];
    int32_t seg1 = seg[1];
    int32_t seg2 = seg[2];
    int32_t seg3 = seg[3];

    int nc = (int)num_clusters;

    
    cudaMemsetAsync(cache.d_sum_deg, 0, num_clusters * sizeof(double));
    cudaMemsetAsync(cache.d_intra_sum, 0, sizeof(double));

    constexpr int BS_HIGH = 256;
    constexpr int BS_MID = 256;
    constexpr int BS_LOW = 256;

    
    if (seg1 > seg0) {
        modularity_high_degree_kernel<BS_HIGH><<<seg1 - seg0, BS_HIGH>>>(
            offsets, indices, edge_weights, cluster_assignments,
            cache.d_sum_deg, cache.d_intra_sum, seg0, seg1);
    }

    
    if (seg2 > seg1) {
        int warps_per_block = BS_MID / 32;
        int n_mid = seg2 - seg1;
        int blocks = (n_mid + warps_per_block - 1) / warps_per_block;
        modularity_mid_degree_kernel<BS_MID><<<blocks, BS_MID>>>(
            offsets, indices, edge_weights, cluster_assignments,
            cache.d_sum_deg, cache.d_intra_sum, seg1, seg2);
    }

    
    if (seg3 > seg2) {
        int n_low = seg3 - seg2;
        int blocks = (n_low + BS_LOW - 1) / BS_LOW;
        if (nc <= 4096) {
            constexpr int NWARPS = BS_LOW / 32;
            size_t smem_size = (nc + NWARPS) * sizeof(double);
            modularity_low_degree_kernel<BS_LOW><<<blocks, BS_LOW, smem_size>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cache.d_sum_deg, cache.d_intra_sum, seg2, seg3, nc);
        } else {
            modularity_low_degree_noshmem_kernel<BS_LOW><<<blocks, BS_LOW>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cache.d_sum_deg, cache.d_intra_sum, seg2, seg3);
        }
    }

    
    compute_modularity_kernel<<<1, 1>>>(cache.d_sum_deg, cache.d_intra_sum,
                                        cache.d_result, nc);

    
    double h_result;
    cudaMemcpy(&h_result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);

    return h_result;
}

}  
