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
    int32_t* d_cluster_sizes = nullptr;
    double* d_result = nullptr;
    int cluster_sizes_capacity = 0;

    void ensure(int num_clusters) {
        if (cluster_sizes_capacity < num_clusters) {
            if (d_cluster_sizes) cudaFree(d_cluster_sizes);
            cluster_sizes_capacity = num_clusters * 2;
            cudaMalloc(&d_cluster_sizes, cluster_sizes_capacity * sizeof(int32_t));
        }
        if (!d_result) {
            cudaMalloc(&d_result, sizeof(double));
        }
    }

    ~Cache() override {
        if (d_cluster_sizes) cudaFree(d_cluster_sizes);
        if (d_result) cudaFree(d_result);
    }
};




__global__ void compute_cluster_sizes_kernel(
    const int32_t* __restrict__ cluster_assignments,
    int32_t* __restrict__ g_cluster_sizes,
    int32_t num_vertices,
    int32_t num_clusters)
{
    extern __shared__ int32_t s_sizes[];
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x)
        s_sizes[i] = 0;
    __syncthreads();

    for (int v = blockIdx.x * blockDim.x + threadIdx.x;
         v < num_vertices;
         v += blockDim.x * gridDim.x)
        atomicAdd(&s_sizes[cluster_assignments[v]], 1);

    __syncthreads();
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x)
        if (s_sizes[i] > 0) atomicAdd(&g_cluster_sizes[i], s_sizes[i]);
}


__global__ void compute_cluster_sizes_kernel_global(
    const int32_t* __restrict__ cluster_assignments,
    int32_t* __restrict__ g_cluster_sizes,
    int32_t num_vertices)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x;
         v < num_vertices;
         v += blockDim.x * gridDim.x)
        atomicAdd(&g_cluster_sizes[cluster_assignments[v]], 1);
}




__global__ void process_high_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    const int32_t* __restrict__ cluster_sizes,
    double* __restrict__ g_partial_sums,
    int32_t vertex_start,
    int32_t vertex_end)
{
    int v = vertex_start + blockIdx.x;
    if (v >= vertex_end) return;

    int cv = cluster_assignments[v];
    int sz = cluster_sizes[cv];
    if (sz <= 0) return;
    double inv_s = 1.0 / (double)sz;

    int start = offsets[v];
    int end = offsets[v + 1];

    double thread_sum = 0.0;
    for (int e = start + threadIdx.x; e < end; e += blockDim.x) {
        int u = indices[e];
        if (cluster_assignments[u] != cv)
            thread_sum += (double)edge_weights[e];
    }

    
    for (int offset = 16; offset > 0; offset /= 2)
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);

    __shared__ double warp_sums[32];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (lane == 0) warp_sums[warp_id] = thread_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        double block_sum = 0.0;
        int num_warps = blockDim.x / 32;
        for (int w = 0; w < num_warps; w++)
            block_sum += warp_sums[w];
        if (block_sum != 0.0)
            atomicAdd(g_partial_sums, block_sum * inv_s);
    }
}




__global__ void process_mid_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    const int32_t* __restrict__ cluster_sizes,
    double* __restrict__ g_partial_sums,
    int32_t vertex_start,
    int32_t vertex_end,
    int32_t num_clusters)
{
    
    extern __shared__ double s_inv_sizes[];
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        int sz = cluster_sizes[i];
        s_inv_sizes[i] = (sz > 0) ? (1.0 / (double)sz) : 0.0;
    }
    __syncthreads();

    int warps_per_block = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int total_warps = gridDim.x * warps_per_block;
    int num_verts = vertex_end - vertex_start;

    
    
    double* s_warp_sums = s_inv_sizes + num_clusters;

    double warp_contrib = 0.0;

    for (int wi = blockIdx.x * warps_per_block + warp_id;
         wi < num_verts;
         wi += total_warps) {
        int v = vertex_start + wi;
        int cv = cluster_assignments[v];
        int start = offsets[v];
        int end = offsets[v + 1];

        double lane_sum = 0.0;
        for (int e = start + lane; e < end; e += 32) {
            int u = indices[e];
            if (cluster_assignments[u] != cv)
                lane_sum += (double)edge_weights[e];
        }

        
        for (int offset = 16; offset > 0; offset /= 2)
            lane_sum += __shfl_down_sync(0xFFFFFFFF, lane_sum, offset);

        if (lane == 0)
            warp_contrib += lane_sum * s_inv_sizes[cv];
    }

    
    if (lane == 0)
        s_warp_sums[warp_id] = warp_contrib;
    __syncthreads();

    if (threadIdx.x == 0) {
        double block_sum = 0.0;
        for (int w = 0; w < warps_per_block; w++)
            block_sum += s_warp_sums[w];
        if (block_sum != 0.0)
            atomicAdd(g_partial_sums, block_sum);
    }
}


__global__ void process_mid_degree_kernel_global(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    const int32_t* __restrict__ cluster_sizes,
    double* __restrict__ g_partial_sums,
    int32_t vertex_start,
    int32_t vertex_end)
{
    int warps_per_block = blockDim.x / 32;
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int total_warps = gridDim.x * warps_per_block;
    int num_verts = vertex_end - vertex_start;

    __shared__ double s_warp_sums[8];

    double warp_contrib = 0.0;

    for (int wi = blockIdx.x * warps_per_block + warp_id;
         wi < num_verts;
         wi += total_warps) {
        int v = vertex_start + wi;
        int cv = cluster_assignments[v];
        int sz = cluster_sizes[cv];
        double inv_s = (sz > 0) ? (1.0 / (double)sz) : 0.0;
        int start = offsets[v];
        int end = offsets[v + 1];

        double lane_sum = 0.0;
        for (int e = start + lane; e < end; e += 32) {
            int u = indices[e];
            if (cluster_assignments[u] != cv)
                lane_sum += (double)edge_weights[e];
        }

        for (int offset = 16; offset > 0; offset /= 2)
            lane_sum += __shfl_down_sync(0xFFFFFFFF, lane_sum, offset);

        if (lane == 0)
            warp_contrib += lane_sum * inv_s;
    }

    if (lane == 0)
        s_warp_sums[warp_id] = warp_contrib;
    __syncthreads();

    if (threadIdx.x == 0) {
        double block_sum = 0.0;
        for (int w = 0; w < warps_per_block; w++)
            block_sum += s_warp_sums[w];
        if (block_sum != 0.0)
            atomicAdd(g_partial_sums, block_sum);
    }
}




__global__ void process_low_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    const int32_t* __restrict__ cluster_sizes,
    double* __restrict__ g_partial_sums,
    int32_t vertex_start,
    int32_t vertex_end,
    int32_t num_clusters)
{
    extern __shared__ double s_inv_sizes[];
    for (int i = threadIdx.x; i < num_clusters; i += blockDim.x) {
        int sz = cluster_sizes[i];
        s_inv_sizes[i] = (sz > 0) ? (1.0 / (double)sz) : 0.0;
    }
    __syncthreads();

    double thread_sum = 0.0;

    for (int v = vertex_start + blockIdx.x * blockDim.x + threadIdx.x;
         v < vertex_end;
         v += blockDim.x * gridDim.x) {
        int cv = cluster_assignments[v];
        int start = offsets[v];
        int end = offsets[v + 1];

        double local_cut = 0.0;
        for (int e = start; e < end; e++) {
            int u = indices[e];
            if (cluster_assignments[u] != cv)
                local_cut += (double)edge_weights[e];
        }
        thread_sum += local_cut * s_inv_sizes[cv];
    }

    
    for (int offset = 16; offset > 0; offset /= 2)
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x / 32;
    
    double* s_warp_sums = s_inv_sizes + num_clusters;
    if (lane == 0) s_warp_sums[warp_id] = thread_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        double block_sum = 0.0;
        int nwarps = blockDim.x / 32;
        for (int w = 0; w < nwarps; w++)
            block_sum += s_warp_sums[w];
        if (block_sum != 0.0)
            atomicAdd(g_partial_sums, block_sum);
    }
}


__global__ void process_low_degree_kernel_global(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    const int32_t* __restrict__ cluster_sizes,
    double* __restrict__ g_partial_sums,
    int32_t vertex_start,
    int32_t vertex_end)
{
    double thread_sum = 0.0;

    for (int v = vertex_start + blockIdx.x * blockDim.x + threadIdx.x;
         v < vertex_end;
         v += blockDim.x * gridDim.x) {
        int cv = cluster_assignments[v];
        int sz = cluster_sizes[cv];
        if (sz <= 0) continue;
        double inv_s = 1.0 / (double)sz;
        int start = offsets[v];
        int end = offsets[v + 1];

        double local_cut = 0.0;
        for (int e = start; e < end; e++) {
            int u = indices[e];
            if (cluster_assignments[u] != cv)
                local_cut += (double)edge_weights[e];
        }
        thread_sum += local_cut * inv_s;
    }

    for (int offset = 16; offset > 0; offset /= 2)
        thread_sum += __shfl_down_sync(0xFFFFFFFF, thread_sum, offset);

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x / 32;
    __shared__ double s_warp_sums[8];
    if (lane == 0) s_warp_sums[warp_id] = thread_sum;
    __syncthreads();

    if (threadIdx.x == 0) {
        double block_sum = 0.0;
        int nwarps = blockDim.x / 32;
        for (int w = 0; w < nwarps; w++)
            block_sum += s_warp_sums[w];
        if (block_sum != 0.0)
            atomicAdd(g_partial_sums, block_sum);
    }
}

}  

double analyze_clustering_ratio_cut_seg(const graph32_t& graph,
                                        const float* edge_weights,
                                        std::size_t num_clusters,
                                        const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t nc = static_cast<int32_t>(num_clusters);

    if (num_vertices == 0) {
        return 0.0;
    }

    cache.ensure(nc);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0];
    int32_t seg1 = seg[1];
    int32_t seg2 = seg[2];
    int32_t seg3 = seg[3];

    
    constexpr int kMaxClustersSmemInv = 6000;
    constexpr int kMaxClustersSmemHist = 12000;

    
    cudaMemsetAsync(cache.d_cluster_sizes, 0, nc * sizeof(int32_t));
    cudaMemsetAsync(cache.d_result, 0, sizeof(double));

    
    {
        int bs = 256;
        int gs = (num_vertices + bs - 1) / bs;
        if (gs > 256) gs = 256;
        if (nc <= kMaxClustersSmemHist) {
            int smem = nc * sizeof(int32_t);
            compute_cluster_sizes_kernel<<<gs, bs, smem>>>(
                cluster_assignments, cache.d_cluster_sizes, num_vertices, nc);
        } else {
            if (gs > 2048) gs = 2048;
            compute_cluster_sizes_kernel_global<<<gs, bs>>>(
                cluster_assignments, cache.d_cluster_sizes, num_vertices);
        }
    }

    
    int num_high = seg1 - seg0;
    if (num_high > 0) {
        int bs = 256;
        process_high_degree_kernel<<<num_high, bs>>>(
            offsets, indices, edge_weights, cluster_assignments,
            cache.d_cluster_sizes, cache.d_result, seg0, seg1);
    }

    bool use_smem_inv = (nc <= kMaxClustersSmemInv);

    
    int num_mid = seg2 - seg1;
    if (num_mid > 0) {
        int bs = 256;
        int warps_per_block = bs / 32;
        int gs = (num_mid + warps_per_block - 1) / warps_per_block;
        if (gs > 1024) gs = 1024;
        if (use_smem_inv) {
            int smem = (nc + warps_per_block) * sizeof(double);
            process_mid_degree_kernel<<<gs, bs, smem>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cache.d_cluster_sizes, cache.d_result, seg1, seg2, nc);
        } else {
            process_mid_degree_kernel_global<<<gs, bs>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cache.d_cluster_sizes, cache.d_result, seg1, seg2);
        }
    }

    
    int num_low = seg3 - seg2;
    if (num_low > 0) {
        int bs = 256;
        int warps_per_block = bs / 32;
        int gs = (num_low + bs - 1) / bs;
        if (gs > 1024) gs = 1024;
        if (use_smem_inv) {
            int smem = (nc + warps_per_block) * sizeof(double);
            process_low_degree_kernel<<<gs, bs, smem>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cache.d_cluster_sizes, cache.d_result, seg2, seg3, nc);
        } else {
            process_low_degree_kernel_global<<<gs, bs>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cache.d_cluster_sizes, cache.d_result, seg2, seg3);
        }
    }

    
    double h_result;
    cudaMemcpy(&h_result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return h_result;
}

}  
