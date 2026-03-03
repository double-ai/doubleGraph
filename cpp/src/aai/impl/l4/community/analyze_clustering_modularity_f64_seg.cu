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

struct Cache : Cacheable {
    double* temp = nullptr;
    int64_t temp_capacity = 0;

    void ensure(int64_t size) {
        if (temp_capacity < size) {
            if (temp) cudaFree(temp);
            cudaMalloc(&temp, size * sizeof(double));
            temp_capacity = size;
        }
    }

    ~Cache() override {
        if (temp) cudaFree(temp);
    }
};

__device__ __forceinline__ float warp_reduce_sum_f(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

struct PairFI {
    float deg;
    float intra;
};

struct PairSum {
    __device__ __forceinline__ PairFI operator()(const PairFI& a, const PairFI& b) const {
        return {a.deg + b.deg, a.intra + b.intra};
    }
};





template <int BLOCK_SIZE>
__global__ void modularity_cta_per_vertex(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ clusters,
    double* __restrict__ cluster_deg,
    double* __restrict__ total_intra,
    double* __restrict__ total_weight,
    int32_t v_start,
    int32_t v_end)
{
    int32_t v = v_start + (int32_t)blockIdx.x;
    if (v >= v_end) return;

    int32_t c = clusters[v];
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    float deg = 0.0f;
    float intra = 0.0f;

    for (int32_t e = start + (int32_t)threadIdx.x; e < end; e += BLOCK_SIZE) {
        float w = (float)weights[e];
        deg += w;
        if (clusters[indices[e]] == c) intra += w;
    }

    
    deg = warp_reduce_sum_f(deg);
    intra = warp_reduce_sum_f(intra);

    constexpr int NUM_WARPS = BLOCK_SIZE / 32;
    int lane = (int)threadIdx.x & 31;
    int warp_id = (int)threadIdx.x >> 5;

    __shared__ float s_deg[NUM_WARPS];
    __shared__ float s_intra[NUM_WARPS];

    if (lane == 0) {
        s_deg[warp_id] = deg;
        s_intra[warp_id] = intra;
    }
    __syncthreads();

    if (warp_id == 0) {
        float vd = (lane < NUM_WARPS) ? s_deg[lane] : 0.0f;
        float vi = (lane < NUM_WARPS) ? s_intra[lane] : 0.0f;
        vd = warp_reduce_sum_f(vd);
        vi = warp_reduce_sum_f(vi);
        if (lane == 0) {
            atomicAdd(&cluster_deg[c], vd);
            atomicAdd(total_intra, vi);
            atomicAdd(total_weight, vd);
        }
    }
}





template <int BLOCK_SIZE>
__global__ void modularity_warp_per_vertex(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ clusters,
    double* __restrict__ cluster_deg,
    double* __restrict__ total_intra,
    double* __restrict__ total_weight,
    int32_t v_start,
    int32_t v_end)
{
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
    int tid = (int)blockIdx.x * BLOCK_SIZE + (int)threadIdx.x;
    int lane = tid & 31;
    int warp_global = tid >> 5;
    int warp_in_block = (int)threadIdx.x >> 5;

    int32_t v = v_start + warp_global;
    float deg = 0.0f;
    float intra = 0.0f;
    int32_t c = 0;

    if (v < v_end) {
        c = clusters[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        for (int32_t e = start + lane; e < end; e += 32) {
            float w = (float)weights[e];
            deg += w;
            if (clusters[indices[e]] == c) intra += w;
        }
    }

    deg = warp_reduce_sum_f(deg);
    intra = warp_reduce_sum_f(intra);

    if (lane == 0 && v < v_end) {
        atomicAdd(&cluster_deg[c], deg);
    }

    __shared__ float s_deg[WARPS_PER_BLOCK];
    __shared__ float s_intra[WARPS_PER_BLOCK];
    if (lane == 0) {
        s_deg[warp_in_block] = deg;
        s_intra[warp_in_block] = intra;
    }
    __syncthreads();

    if (warp_in_block == 0) {
        float vd = (lane < WARPS_PER_BLOCK) ? s_deg[lane] : 0.0f;
        float vi = (lane < WARPS_PER_BLOCK) ? s_intra[lane] : 0.0f;
        vd = warp_reduce_sum_f(vd);
        vi = warp_reduce_sum_f(vi);
        if (lane == 0) {
            atomicAdd(total_intra, vi);
            atomicAdd(total_weight, vd);
        }
    }
}





template <int BLOCK_SIZE>
__global__ void modularity_thread_per_vertex_global(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ clusters,
    double* __restrict__ cluster_deg,
    double* __restrict__ total_intra,
    double* __restrict__ total_weight,
    int32_t v_start,
    int32_t v_end)
{
    int32_t v = v_start + (int32_t)blockIdx.x * BLOCK_SIZE + (int32_t)threadIdx.x;
    float deg = 0.0f;
    float intra = 0.0f;
    if (v < v_end) {
        int32_t c = clusters[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        for (int32_t e = start; e < end; ++e) {
            float w = (float)weights[e];
            deg += w;
            if (clusters[indices[e]] == c) intra += w;
        }
        atomicAdd(&cluster_deg[c], deg);
    }

    using BlockReduce = cub::BlockReduce<PairFI, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage tmp;
    PairFI sum = BlockReduce(tmp).Reduce({deg, intra}, PairSum{});
    if (threadIdx.x == 0) {
        atomicAdd(total_weight, sum.deg);
        atomicAdd(total_intra, sum.intra);
    }
}





template <int BLOCK_SIZE>
__global__ void modularity_thread_per_vertex_shared(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ clusters,
    double* __restrict__ cluster_deg,
    double* __restrict__ total_intra,
    double* __restrict__ total_weight,
    int32_t v_start,
    int32_t v_end,
    int32_t num_clusters)
{
    extern __shared__ float s_deg[]; 

    for (int i = (int)threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        s_deg[i] = 0.0f;
    }
    __syncthreads();

    int32_t v = v_start + (int32_t)blockIdx.x * BLOCK_SIZE + (int32_t)threadIdx.x;
    float deg = 0.0f;
    float intra = 0.0f;
    if (v < v_end) {
        int32_t c = clusters[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        for (int32_t e = start; e < end; ++e) {
            float w = (float)weights[e];
            deg += w;
            if (clusters[indices[e]] == c) intra += w;
        }
        atomicAdd(&s_deg[c], deg);
    }

    using BlockReduce = cub::BlockReduce<PairFI, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage tmp;
    PairFI sum = BlockReduce(tmp).Reduce({deg, intra}, PairSum{});
    if (threadIdx.x == 0) {
        atomicAdd(total_weight, sum.deg);
        atomicAdd(total_intra, sum.intra);
    }

    __syncthreads();
    for (int i = (int)threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        float vdeg = s_deg[i];
        if (vdeg != 0.0f) atomicAdd(&cluster_deg[i], vdeg);
    }
}





__global__ void compute_score_kernel(
    const double* __restrict__ cluster_deg,
    const double* __restrict__ total_intra,
    const double* __restrict__ total_weight,
    int32_t num_clusters,
    double* __restrict__ out)
{
    double tw = (double)(*total_weight);
    double ti = (double)(*total_intra);
    if (tw == 0.0) {
        if (threadIdx.x == 0) out[0] = 0.0;
        return;
    }

    double sum_sq = 0.0;
    for (int i = (int)threadIdx.x; i < num_clusters; i += (int)blockDim.x) {
        double a = (double)cluster_deg[i] / tw;
        sum_sq += a * a;
    }

    using BlockReduce = cub::BlockReduce<double, 256>;
    __shared__ typename BlockReduce::TempStorage tmp;
    double sum_sq_block = BlockReduce(tmp).Sum(sum_sq);

    if (threadIdx.x == 0) {
        out[0] = ti / tw - sum_sq_block;
    }
}

}  

double analyze_clustering_modularity_seg(const graph32_t& graph,
                                         const double* edge_weights,
                                         std::size_t num_clusters,
                                         const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nc = static_cast<int32_t>(num_clusters);

    
    int64_t temp_size = static_cast<int64_t>(nc) + 3;
    cache.ensure(temp_size);

    double* d_cluster_deg = cache.temp;
    double* d_total_intra = d_cluster_deg + nc;
    double* d_total_weight = d_total_intra + 1;
    double* d_result = d_total_weight + 1;

    cudaMemsetAsync(cache.temp, 0, static_cast<size_t>(nc + 2) * sizeof(double));

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg3 = seg[3];

    if (seg1 > seg0) {
        int32_t n = seg1 - seg0;
        constexpr int BS = 256;
        modularity_cta_per_vertex<BS><<<n, BS>>>(
            d_offsets, d_indices, edge_weights, cluster_assignments,
            d_cluster_deg, d_total_intra, d_total_weight, seg0, seg1);
    }
    if (seg2 > seg1) {
        int32_t n = seg2 - seg1;
        constexpr int BS = 256;
        constexpr int WPB = BS / 32;
        int grid = (n + WPB - 1) / WPB;
        modularity_warp_per_vertex<BS><<<grid, BS>>>(
            d_offsets, d_indices, edge_weights, cluster_assignments,
            d_cluster_deg, d_total_intra, d_total_weight, seg1, seg2);
    }
    if (seg3 > seg2) {
        constexpr int BS = 256;
        int32_t n = seg3 - seg2;
        int grid = (n + BS - 1) / BS;
        if (nc <= 256) {
            int smem = nc * (int)sizeof(float);
            modularity_thread_per_vertex_shared<BS><<<grid, BS, smem>>>(
                d_offsets, d_indices, edge_weights, cluster_assignments,
                d_cluster_deg, d_total_intra, d_total_weight,
                seg2, seg3, nc);
        } else {
            modularity_thread_per_vertex_global<BS><<<grid, BS>>>(
                d_offsets, d_indices, edge_weights, cluster_assignments,
                d_cluster_deg, d_total_intra, d_total_weight,
                seg2, seg3);
        }
    }

    compute_score_kernel<<<1, 256>>>(d_cluster_deg, d_total_intra, d_total_weight, nc, d_result);

    double result;
    cudaMemcpy(&result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

}  
