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

__device__ __forceinline__ int lane_id() { return (int)(threadIdx.x & 31); }






template <typename ClusterT, int BLOCK>
__global__ __launch_bounds__(BLOCK, 6) void tpv_smem_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const ClusterT* __restrict__ clusters,
    int32_t* __restrict__ cluster_sizes,
    double* __restrict__ cluster_cuts,
    int32_t num_vertices,
    int32_t num_clusters)
{
    extern __shared__ char smem[];
    float* s_cuts = (float*)smem;
    int32_t* s_sizes = (int32_t*)(s_cuts + num_clusters);

    for (int i = (int)threadIdx.x; i < num_clusters; i += BLOCK) {
        s_cuts[i] = 0.0f;
        s_sizes[i] = 0;
    }
    __syncthreads();

    for (int u = (int)(blockIdx.x * BLOCK + threadIdx.x); u < num_vertices; u += (int)(gridDim.x * BLOCK)) {
        int cu = (int)__ldg(&clusters[u]);
        atomicAdd(&s_sizes[cu], 1);

        int start = offsets[u];
        int end = offsets[u + 1];

        float local_cut = 0.0f;
        for (int e = start; e < end; ++e) {
            int v = __ldg(&indices[e]);
            float w = __ldg(&edge_weights[e]);
            int cv = (int)__ldg(&clusters[v]);
            local_cut += (cu != cv) ? w : 0.0f;
        }
        if (local_cut != 0.0f) atomicAdd(&s_cuts[cu], local_cut);
    }

    __syncthreads();
    for (int i = (int)threadIdx.x; i < num_clusters; i += BLOCK) {
        float c = s_cuts[i];
        int32_t sz = s_sizes[i];
        if (c != 0.0f) atomicAdd(&cluster_cuts[i], (double)c);
        if (sz != 0) atomicAdd(&cluster_sizes[i], sz);
    }
}


template <typename ClusterT, int BLOCK>
__global__ __launch_bounds__(BLOCK, 6) void wpv_smem_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const ClusterT* __restrict__ clusters,
    int32_t* __restrict__ cluster_sizes,
    double* __restrict__ cluster_cuts,
    int32_t num_vertices,
    int32_t num_clusters)
{
    extern __shared__ char smem[];
    float* s_cuts = (float*)smem;
    int32_t* s_sizes = (int32_t*)(s_cuts + num_clusters);

    for (int i = (int)threadIdx.x; i < num_clusters; i += BLOCK) {
        s_cuts[i] = 0.0f;
        s_sizes[i] = 0;
    }
    __syncthreads();

    const int lane = lane_id();
    constexpr int warps_per_block = BLOCK >> 5;
    const int warp_in_block = (int)(threadIdx.x >> 5);
    const int global_warp = (int)(blockIdx.x * warps_per_block + warp_in_block);
    const int total_warps = (int)(gridDim.x * warps_per_block);

    for (int u = global_warp; u < num_vertices; u += total_warps) {
        int start = offsets[u];
        int end = offsets[u + 1];

        int cu;
        if (lane == 0) {
            cu = (int)__ldg(&clusters[u]);
            atomicAdd(&s_sizes[cu], 1);
        }
        cu = __shfl_sync(0xFFFFFFFFu, cu, 0);

        float local_cut = 0.0f;
        
        for (int e = start + lane; e < end; e += 64) {
            int v0 = __ldg(&indices[e]);
            float w0 = __ldg(&edge_weights[e]);
            int cv0 = (int)__ldg(&clusters[v0]);
            local_cut += (cu != cv0) ? w0 : 0.0f;

            int e1 = e + 32;
            if (e1 < end) {
                int v1 = __ldg(&indices[e1]);
                float w1 = __ldg(&edge_weights[e1]);
                int cv1 = (int)__ldg(&clusters[v1]);
                local_cut += (cu != cv1) ? w1 : 0.0f;
            }
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) local_cut += __shfl_down_sync(0xFFFFFFFFu, local_cut, off);

        if (lane == 0 && local_cut != 0.0f) atomicAdd(&s_cuts[cu], local_cut);
    }

    __syncthreads();
    for (int i = (int)threadIdx.x; i < num_clusters; i += BLOCK) {
        float c = s_cuts[i];
        int32_t sz = s_sizes[i];
        if (c != 0.0f) atomicAdd(&cluster_cuts[i], (double)c);
        if (sz != 0) atomicAdd(&cluster_sizes[i], sz);
    }
}


__global__ void ratio_cut_final_kernel(
    const double* __restrict__ cluster_cuts,
    const int32_t* __restrict__ cluster_sizes,
    double* __restrict__ out,
    int32_t num_clusters)
{
    double sum = 0.0;
    for (int c = (int)threadIdx.x; c < num_clusters; c += (int)blockDim.x) {
        int32_t sz = cluster_sizes[c];
        if (sz > 0) sum += cluster_cuts[c] / (double)sz;
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xFFFFFFFFu, sum, off);

    extern __shared__ double warp_sums[];
    int lane = lane_id();
    int warp = (int)(threadIdx.x >> 5);
    int nwarps = (int)(blockDim.x >> 5);

    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        double v = (lane < nwarps) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xFFFFFFFFu, v, off);
        if (lane == 0) *out = v;
    }
}







template <typename ClusterT>
__global__ void cluster_sizes_global_kernel(const ClusterT* __restrict__ clusters,
                                            int32_t* __restrict__ cluster_sizes,
                                            int32_t num_vertices)
{
    for (int u = (int)(blockIdx.x * blockDim.x + threadIdx.x); u < num_vertices;
         u += (int)(blockDim.x * gridDim.x)) {
        int cu = (int)__ldg(&clusters[u]);
        atomicAdd(&cluster_sizes[cu], 1);
    }
}


template <typename ClusterT, int BLOCK>
__global__ __launch_bounds__(BLOCK, 6) void tpv_direct_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const ClusterT* __restrict__ clusters,
    const int32_t* __restrict__ cluster_sizes,
    int32_t num_vertices,
    double* __restrict__ out)
{
    __shared__ double s_sum[BLOCK];

    double local = 0.0;
    for (int u = (int)(blockIdx.x * BLOCK + threadIdx.x); u < num_vertices; u += (int)(gridDim.x * BLOCK)) {
        int cu = (int)__ldg(&clusters[u]);
        int32_t sz = __ldg(&cluster_sizes[cu]);
        if (sz == 0) continue;

        int start = offsets[u];
        int end = offsets[u + 1];

        float cut = 0.0f;
        for (int e = start; e < end; ++e) {
            int v = __ldg(&indices[e]);
            float w = __ldg(&edge_weights[e]);
            int cv = (int)__ldg(&clusters[v]);
            cut += (cu != cv) ? w : 0.0f;
        }
        local += (double)cut / (double)sz;
    }

    s_sum[threadIdx.x] = local;
    __syncthreads();

    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        __syncthreads();
    }

    if (threadIdx.x == 0 && s_sum[0] != 0.0) atomicAdd(out, s_sum[0]);
}


template <typename ClusterT, int BLOCK>
__global__ __launch_bounds__(BLOCK, 6) void wpv_direct_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const ClusterT* __restrict__ clusters,
    const int32_t* __restrict__ cluster_sizes,
    int32_t num_vertices,
    double* __restrict__ out)
{
    const int lane = lane_id();
    constexpr int warps_per_block = BLOCK >> 5;
    const int warp_in_block = (int)(threadIdx.x >> 5);
    const int global_warp = (int)(blockIdx.x * warps_per_block + warp_in_block);
    const int total_warps = (int)(gridDim.x * warps_per_block);

    double warp_accum = 0.0;

    for (int u = global_warp; u < num_vertices; u += total_warps) {
        int start = offsets[u];
        int end = offsets[u + 1];

        int cu;
        int32_t sz;
        if (lane == 0) {
            cu = (int)__ldg(&clusters[u]);
            sz = __ldg(&cluster_sizes[cu]);
        }
        cu = __shfl_sync(0xFFFFFFFFu, cu, 0);
        sz = __shfl_sync(0xFFFFFFFFu, sz, 0);
        if (sz == 0) continue;

        float local_cut = 0.0f;
        for (int e = start + lane; e < end; e += 64) {
            int v0 = __ldg(&indices[e]);
            float w0 = __ldg(&edge_weights[e]);
            int cv0 = (int)__ldg(&clusters[v0]);
            local_cut += (cu != cv0) ? w0 : 0.0f;

            int e1 = e + 32;
            if (e1 < end) {
                int v1 = __ldg(&indices[e1]);
                float w1 = __ldg(&edge_weights[e1]);
                int cv1 = (int)__ldg(&clusters[v1]);
                local_cut += (cu != cv1) ? w1 : 0.0f;
            }
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) local_cut += __shfl_down_sync(0xFFFFFFFFu, local_cut, off);

        if (lane == 0 && local_cut != 0.0f) warp_accum += (double)local_cut / (double)sz;
    }

    extern __shared__ double s_warp[];
    if (lane == 0) s_warp[warp_in_block] = warp_accum;
    __syncthreads();

    if (threadIdx.x == 0) {
        double block_sum = 0.0;
        #pragma unroll
        for (int w = 0; w < warps_per_block; ++w) block_sum += s_warp[w];
        if (block_sum != 0.0) atomicAdd(out, block_sum);
    }
}





struct Cache : Cacheable {
    int sm_count = 0;
    size_t persist_l2_bytes = 4 * 1024 * 1024;

    int32_t* sizes = nullptr;
    int64_t sizes_capacity = 0;

    double* cuts = nullptr;
    int64_t cuts_capacity = 0;

    double* result = nullptr;

    Cache() {
        (void)cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

        int max_persist = 0;
        if (cudaDeviceGetAttribute(&max_persist, cudaDevAttrMaxPersistingL2CacheSize, 0) == cudaSuccess && max_persist > 0) {
            persist_l2_bytes = (size_t)max_persist;
        }
        (void)cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist_l2_bytes);

        cudaMalloc(&result, sizeof(double));
    }

    void ensure_sizes(int64_t nc) {
        if (sizes_capacity < nc) {
            if (sizes) cudaFree(sizes);
            cudaMalloc(&sizes, nc * sizeof(int32_t));
            sizes_capacity = nc;
        }
    }

    void ensure_cuts(int64_t nc) {
        if (cuts_capacity < nc) {
            if (cuts) cudaFree(cuts);
            cudaMalloc(&cuts, nc * sizeof(double));
            cuts_capacity = nc;
        }
    }

    ~Cache() override {
        if (sizes) cudaFree(sizes);
        if (cuts) cudaFree(cuts);
        if (result) cudaFree(result);
        (void)cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
    }
};

}  

double analyze_clustering_ratio_cut(const graph32_t& graph,
                                    const float* edge_weights,
                                    std::size_t num_clusters,
                                    const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* off_ptr = graph.offsets;
    const int32_t* ind_ptr = graph.indices;
    const float* ew_ptr = edge_weights;
    const int32_t* cl_ptr = cluster_assignments;
    int32_t nc = static_cast<int32_t>(num_clusters);

    constexpr int32_t kMaxClustersSmem = 6000;
    bool use_smem_bins = (nc <= kMaxClustersSmem);

    float avg_degree = (num_vertices > 0) ? (float)num_edges / (float)num_vertices : 0.0f;
    bool use_warp = (avg_degree >= 16.0f);

    int max_blocks = std::max(1, cache.sm_count) * 6;

    constexpr int kWarpsPerBlock = 8;
    int grid = 1;
    if (use_warp) {
        grid = std::min((num_vertices + kWarpsPerBlock - 1) / kWarpsPerBlock, max_blocks);
    } else {
        grid = std::min((num_vertices + 255) / 256, max_blocks);
    }
    if (grid < 1) grid = 1;

    
    {
        cudaStreamAttrValue attr{};
        attr.accessPolicyWindow.base_ptr = const_cast<void*>((const void*)cl_ptr);
        attr.accessPolicyWindow.num_bytes = std::min((size_t)num_vertices * sizeof(int32_t), cache.persist_l2_bytes);
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        (void)cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
    }

    auto reset_l2_persist = []() {
        cudaStreamAttrValue reset{};
        reset.accessPolicyWindow.num_bytes = 0;
        (void)cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &reset);
    };

    if (use_smem_bins) {
        cache.ensure_sizes(nc);
        cache.ensure_cuts(nc);

        (void)cudaMemsetAsync(cache.sizes, 0, (size_t)nc * sizeof(int32_t));
        (void)cudaMemsetAsync(cache.cuts, 0, (size_t)nc * sizeof(double));

        int smem = (int)((size_t)nc * (sizeof(float) + sizeof(int32_t)));

        if (use_warp)
            wpv_smem_kernel<int32_t, 256><<<grid, 256, smem>>>(off_ptr, ind_ptr, ew_ptr, cl_ptr, cache.sizes, cache.cuts, num_vertices, nc);
        else
            tpv_smem_kernel<int32_t, 256><<<grid, 256, smem>>>(off_ptr, ind_ptr, ew_ptr, cl_ptr, cache.sizes, cache.cuts, num_vertices, nc);

        reset_l2_persist();

        int final_smem = (256 / 32) * (int)sizeof(double);
        ratio_cut_final_kernel<<<1, 256, final_smem>>>(cache.cuts, cache.sizes, cache.result, nc);
    } else {
        cache.ensure_sizes(nc);

        (void)cudaMemsetAsync(cache.sizes, 0, (size_t)nc * sizeof(int32_t));
        (void)cudaMemsetAsync(cache.result, 0, sizeof(double));

        int grid_hist = std::min(std::max(1, cache.sm_count) * 12, 4096);
        cluster_sizes_global_kernel<int32_t><<<grid_hist, 256>>>(cl_ptr, cache.sizes, num_vertices);

        if (use_warp) {
            int smem = (256 / 32) * (int)sizeof(double);
            wpv_direct_kernel<int32_t, 256><<<grid, 256, smem>>>(off_ptr, ind_ptr, ew_ptr, cl_ptr, cache.sizes, num_vertices, cache.result);
        } else {
            tpv_direct_kernel<int32_t, 256><<<grid, 256>>>(off_ptr, ind_ptr, ew_ptr, cl_ptr, cache.sizes, num_vertices, cache.result);
        }

        reset_l2_persist();
    }

    double host_result;
    cudaMemcpy(&host_result, cache.result, sizeof(double), cudaMemcpyDeviceToHost);
    return host_result;
}

}  
