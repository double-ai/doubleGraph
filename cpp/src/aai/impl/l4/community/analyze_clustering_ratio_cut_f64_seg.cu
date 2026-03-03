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
    int32_t* cluster_sizes = nullptr;
    double* inv_sizes = nullptr;
    double* d_out = nullptr;
    int32_t cluster_sizes_capacity = 0;
    int32_t inv_sizes_capacity = 0;
    bool out_allocated = false;

    void ensure(int32_t num_clusters) {
        if (cluster_sizes_capacity < num_clusters) {
            if (cluster_sizes) cudaFree(cluster_sizes);
            cluster_sizes_capacity = (num_clusters < 8192) ? 8192 : num_clusters * 2;
            cudaMalloc(&cluster_sizes, (size_t)cluster_sizes_capacity * sizeof(int32_t));
        }
        if (inv_sizes_capacity < num_clusters) {
            if (inv_sizes) cudaFree(inv_sizes);
            inv_sizes_capacity = (num_clusters < 8192) ? 8192 : num_clusters * 2;
            cudaMalloc(&inv_sizes, (size_t)inv_sizes_capacity * sizeof(double));
        }
        if (!out_allocated) {
            cudaMalloc(&d_out, sizeof(double));
            out_allocated = true;
        }
    }

    ~Cache() override {
        if (cluster_sizes) cudaFree(cluster_sizes);
        if (inv_sizes) cudaFree(inv_sizes);
        if (d_out) cudaFree(d_out);
    }
};





__device__ __forceinline__ int lane_id() { return (int)(threadIdx.x & 31); }

__device__ __forceinline__ double warp_reduce_sum(double v, unsigned mask = 0xffffffffu) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

template<int BLOCK>
__device__ __forceinline__ double block_reduce_sum(double v) {
    constexpr int WARPS = BLOCK / 32;
    __shared__ double warp_sums[WARPS];
    const int lane = lane_id();
    const int warp = (int)(threadIdx.x >> 5);

    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();

    double out = 0.0;
    if (warp == 0) {
        out = (lane < WARPS) ? warp_sums[lane] : 0.0;
        out = warp_reduce_sum(out);
    }
    return out;
}





template<int BLOCK>
__global__ void compute_cluster_sizes_smem(
    const int32_t* __restrict__ clusters,
    int32_t num_vertices,
    int32_t num_clusters,
    int32_t* __restrict__ sizes)
{
    extern __shared__ int32_t s_hist[];

    for (int i = (int)threadIdx.x; i < num_clusters; i += BLOCK) {
        s_hist[i] = 0;
    }
    __syncthreads();

    int32_t tid = (int32_t)blockIdx.x * BLOCK + (int32_t)threadIdx.x;
    int32_t stride = (int32_t)gridDim.x * BLOCK;
    for (int32_t v = tid; v < num_vertices; v += stride) {
        atomicAdd(&s_hist[clusters[v]], 1);
    }
    __syncthreads();

    for (int i = (int)threadIdx.x; i < num_clusters; i += BLOCK) {
        int32_t val = s_hist[i];
        if (val) atomicAdd(&sizes[i], val);
    }
}

__global__ void compute_cluster_sizes_global(
    const int32_t* __restrict__ clusters,
    int32_t num_vertices,
    int32_t* __restrict__ sizes)
{
    int32_t tid = (int32_t)blockIdx.x * (int32_t)blockDim.x + (int32_t)threadIdx.x;
    int32_t stride = (int32_t)gridDim.x * (int32_t)blockDim.x;
    for (int32_t v = tid; v < num_vertices; v += stride) {
        atomicAdd(&sizes[clusters[v]], 1);
    }
}

__global__ void compute_inv_sizes(
    const int32_t* __restrict__ sizes,
    int32_t num_clusters,
    double* __restrict__ inv_sizes)
{
    for (int32_t c = (int32_t)blockIdx.x * (int32_t)blockDim.x + (int32_t)threadIdx.x;
         c < num_clusters;
         c += (int32_t)gridDim.x * (int32_t)blockDim.x) {
        int32_t sz = sizes[c];
        inv_sizes[c] = (sz > 0) ? (1.0 / (double)sz) : 0.0;
    }
}






template<int BLOCK>
__global__ void ratio_cut_high_degree_direct(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ w,
    const int32_t* __restrict__ clusters,
    const double* __restrict__ inv_sizes,
    int32_t seg_start,
    int32_t seg_end,
    double* __restrict__ out)
{
    int32_t v = seg_start + (int32_t)blockIdx.x;
    if (v >= seg_end) return;

    const int32_t c_v = clusters[v];
    const double inv = inv_sizes[c_v];
    const int32_t start = offsets[v];
    const int32_t end = offsets[v + 1];

    double local = 0.0;
    for (int32_t e = start + (int32_t)threadIdx.x; e < end; e += BLOCK) {
        const int32_t n = indices[e];
        const int32_t c_n = clusters[n];
        if (c_n != c_v) local += w[e] * inv;
    }

    double sum = block_reduce_sum<BLOCK>(local);
    if (threadIdx.x == 0 && sum != 0.0) {
        atomicAdd(out, sum);
    }
}


template<int BLOCK>
__global__ void ratio_cut_mid_degree_direct(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ w,
    const int32_t* __restrict__ clusters,
    const double* __restrict__ inv_sizes,
    int32_t seg_start,
    int32_t seg_end,
    double* __restrict__ out)
{
    constexpr int WARPS = BLOCK / 32;
    __shared__ double warp_sums[WARPS];

    const int warp_in_block = (int)(threadIdx.x >> 5);
    const int lane = lane_id();
    int32_t global_warp = (int32_t)blockIdx.x * WARPS + warp_in_block;
    int32_t total_warps = (int32_t)gridDim.x * WARPS;

    double warp_accum = 0.0;

    for (int32_t v = seg_start + global_warp; v < seg_end; v += total_warps) {
        const int32_t c_v = clusters[v];
        const double inv = inv_sizes[c_v];
        const int32_t start = offsets[v];
        const int32_t end = offsets[v + 1];

        double local = 0.0;
        for (int32_t e = start + lane; e < end; e += 32) {
            const int32_t n = indices[e];
            const int32_t c_n = clusters[n];
            if (c_n != c_v) local += w[e] * inv;
        }
        local = warp_reduce_sum(local);
        if (lane == 0) warp_accum += local;
    }

    if (lane == 0) warp_sums[warp_in_block] = warp_accum;
    __syncthreads();

    if (warp_in_block == 0) {
        double v = (lane < WARPS) ? warp_sums[lane] : 0.0;
        v = warp_reduce_sum(v);
        if (lane == 0 && v != 0.0) atomicAdd(out, v);
    }
}


template<int BLOCK>
__global__ void ratio_cut_low_degree_direct(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ w,
    const int32_t* __restrict__ clusters,
    const double* __restrict__ inv_sizes,
    int32_t seg_start,
    int32_t seg_end,
    double* __restrict__ out)
{
    int32_t tid = (int32_t)blockIdx.x * BLOCK + (int32_t)threadIdx.x;
    int32_t stride = (int32_t)gridDim.x * BLOCK;

    double acc = 0.0;
    for (int32_t v = seg_start + tid; v < seg_end; v += stride) {
        const int32_t c_v = clusters[v];
        const double inv = inv_sizes[c_v];
        const int32_t start = offsets[v];
        const int32_t end = offsets[v + 1];

        double local = 0.0;
        #pragma unroll
        for (int32_t e = start; e < end; e++) {
            const int32_t n = indices[e];
            const int32_t c_n = clusters[n];
            if (c_n != c_v) local += w[e] * inv;
        }
        acc += local;
    }

    double sum = block_reduce_sum<BLOCK>(acc);
    if (threadIdx.x == 0 && sum != 0.0) {
        atomicAdd(out, sum);
    }
}

}  

double analyze_clustering_ratio_cut_seg(const graph32_t& graph,
                                        const double* edge_weights,
                                        std::size_t num_clusters,
                                        const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0];
    int32_t seg1 = seg[1];
    int32_t seg2 = seg[2];
    int32_t seg3 = seg[3];
    int32_t seg4 = seg[4];

    int32_t nc = (int32_t)num_clusters;
    cache.ensure(nc);

    cudaStream_t stream = 0;
    cudaMemsetAsync(cache.cluster_sizes, 0, (size_t)nc * sizeof(int32_t), stream);
    cudaMemsetAsync(cache.d_out, 0, sizeof(double), stream);

    constexpr int BLOCK = 256;
    constexpr int SMEM_CLUSTER_THRESHOLD = 2048;

    
    {
        int grid = (num_vertices + BLOCK - 1) / BLOCK;
        if (grid > 1024) grid = 1024;
        if (grid < 1) grid = 1;

        if (nc <= SMEM_CLUSTER_THRESHOLD) {
            size_t smem = (size_t)nc * sizeof(int32_t);
            compute_cluster_sizes_smem<BLOCK><<<grid, BLOCK, smem, stream>>>(
                cluster_assignments, num_vertices, nc, cache.cluster_sizes);
        } else {
            compute_cluster_sizes_global<<<grid, BLOCK, 0, stream>>>(
                cluster_assignments, num_vertices, cache.cluster_sizes);
        }
    }

    
    {
        int grid = (nc + BLOCK - 1) / BLOCK;
        if (grid < 1) grid = 1;
        if (grid > 64) grid = 64;
        compute_inv_sizes<<<grid, BLOCK, 0, stream>>>(cache.cluster_sizes, nc, cache.inv_sizes);
    }

    
    const int32_t n_high = seg1 - seg0;
    const int32_t n_mid = seg2 - seg1;
    const int32_t n_low = seg3 - seg2;
    (void)seg4;
    (void)num_edges;

    if (n_high > 0) {
        ratio_cut_high_degree_direct<BLOCK><<<n_high, BLOCK, 0, stream>>>(
            offsets, indices, edge_weights, cluster_assignments, cache.inv_sizes, seg0, seg1, cache.d_out);
    }

    if (n_mid > 0) {
        const int warps_per_block = BLOCK / 32;
        int grid = (n_mid + warps_per_block - 1) / warps_per_block;
        if (grid < 1) grid = 1;
        if (grid > 8192) grid = 8192;
        ratio_cut_mid_degree_direct<BLOCK><<<grid, BLOCK, 0, stream>>>(
            offsets, indices, edge_weights, cluster_assignments, cache.inv_sizes, seg1, seg2, cache.d_out);
    }

    if (n_low > 0) {
        int grid = (n_low + BLOCK - 1) / BLOCK;
        if (grid < 1) grid = 1;
        if (grid > 16384) grid = 16384;
        ratio_cut_low_degree_direct<BLOCK><<<grid, BLOCK, 0, stream>>>(
            offsets, indices, edge_weights, cluster_assignments, cache.inv_sizes, seg2, seg3, cache.d_out);
    }

    double result = 0.0;
    cudaMemcpy(&result, cache.d_out, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

}  
