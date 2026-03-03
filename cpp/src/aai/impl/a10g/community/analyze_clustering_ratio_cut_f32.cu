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

struct Cache : Cacheable {
    int32_t* sizes = nullptr;
    int64_t sizes_capacity = 0;

    double* cuts = nullptr;
    int64_t cuts_capacity = 0;

    uint8_t* packed_u8 = nullptr;
    int64_t packed_u8_capacity = 0;

    int16_t* packed_i16 = nullptr;
    int64_t packed_i16_capacity = 0;

    double* result_d = nullptr;

    Cache() {
        cudaMalloc(&result_d, sizeof(double));
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 4 * 1024 * 1024);
    }

    ~Cache() override {
        if (sizes) cudaFree(sizes);
        if (cuts) cudaFree(cuts);
        if (packed_u8) cudaFree(packed_u8);
        if (packed_i16) cudaFree(packed_i16);
        if (result_d) cudaFree(result_d);
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
    }

    void ensure_sizes(int64_t n) {
        if (sizes_capacity < n) {
            if (sizes) cudaFree(sizes);
            cudaMalloc(&sizes, n * sizeof(int32_t));
            sizes_capacity = n;
        }
    }

    void ensure_cuts(int64_t n) {
        if (cuts_capacity < n) {
            if (cuts) cudaFree(cuts);
            cudaMalloc(&cuts, n * sizeof(double));
            cuts_capacity = n;
        }
    }

    void ensure_packed_u8(int64_t n) {
        if (packed_u8_capacity < n) {
            if (packed_u8) cudaFree(packed_u8);
            cudaMalloc(&packed_u8, n * sizeof(uint8_t));
            packed_u8_capacity = n;
        }
    }

    void ensure_packed_i16(int64_t n) {
        if (packed_i16_capacity < n) {
            if (packed_i16) cudaFree(packed_i16);
            cudaMalloc(&packed_i16, n * sizeof(int16_t));
            packed_i16_capacity = n;
        }
    }
};

__global__ void pack_clusters_u8_kernel(
    const int32_t* __restrict__ in, uint8_t* __restrict__ out, int32_t n, int32_t nc) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int32_t c = in[i];
        if ((uint32_t)c >= (uint32_t)nc) c = (c < 0) ? 0 : (nc - 1);
        out[i] = (uint8_t)c;
    }
}
__global__ void pack_clusters_i16_kernel(
    const int32_t* __restrict__ in, int16_t* __restrict__ out, int32_t n, int32_t nc) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int32_t c = in[i];
        if ((uint32_t)c >= (uint32_t)nc) c = (c < 0) ? 0 : (nc - 1);
        out[i] = (int16_t)c;
    }
}



template <typename ClusterT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 6)
void thread_per_vertex_kernel(
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

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE)
        { s_cuts[i] = 0.0f; s_sizes[i] = 0; }
    __syncthreads();

    for (int u = blockIdx.x * BLOCK_SIZE + threadIdx.x; u < num_vertices; u += gridDim.x * BLOCK_SIZE) {
        int cu = (int)__ldg(&clusters[u]);
        atomicAdd(&s_sizes[cu], 1);

        int start = offsets[u];
        int end = offsets[u + 1];

        float local_cut = 0.0f;
        for (int e = start; e < end; e++) {
            int v = __ldg(&indices[e]);
            float w = __ldg(&edge_weights[e]);
            int cv = (int)__ldg(&clusters[v]);
            local_cut += (cu != cv) ? w : 0.0f;
        }
        if (local_cut > 0.0f) atomicAdd(&s_cuts[cu], local_cut);
    }

    __syncthreads();
    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        if (s_cuts[i] > 0.0f) atomicAdd(&cluster_cuts[i], (double)s_cuts[i]);
        if (s_sizes[i] > 0) atomicAdd(&cluster_sizes[i], s_sizes[i]);
    }
}


template <typename ClusterT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 6)
void warp_per_vertex_kernel(
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

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE)
        { s_cuts[i] = 0.0f; s_sizes[i] = 0; }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    constexpr int warps_per_block = BLOCK_SIZE >> 5;
    const int global_warp = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int total_warps = gridDim.x * warps_per_block;

    for (int u = global_warp; u < num_vertices; u += total_warps) {
        int start = offsets[u];
        int end = offsets[u + 1];

        int cu;
        if (lane == 0) {
            cu = (int)__ldg(&clusters[u]);
            atomicAdd(&s_sizes[cu], 1);
        }
        cu = __shfl_sync(0xFFFFFFFF, cu, 0);

        float local_cut = 0.0f;
        for (int e = start + lane; e < end; e += 32) {
            int v = indices[e];
            float w = edge_weights[e];
            int cv = (int)__ldg(&clusters[v]);
            local_cut += (cu != cv) ? w : 0.0f;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            local_cut += __shfl_down_sync(0xFFFFFFFF, local_cut, offset);

        if (lane == 0 && local_cut > 0.0f)
            atomicAdd(&s_cuts[cu], local_cut);
    }

    __syncthreads();
    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        if (s_cuts[i] > 0.0f) atomicAdd(&cluster_cuts[i], (double)s_cuts[i]);
        if (s_sizes[i] > 0) atomicAdd(&cluster_sizes[i], s_sizes[i]);
    }
}

__global__ void ratio_cut_final_kernel(
    const double* __restrict__ cluster_cuts,
    const int32_t* __restrict__ cluster_sizes,
    double* __restrict__ result,
    int32_t num_clusters)
{
    double sum = 0.0;
    for (int c = threadIdx.x; c < num_clusters; c += blockDim.x) {
        int size = cluster_sizes[c];
        if (size > 0) sum += cluster_cuts[c] / (double)size;
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    if (threadIdx.x == 0) *result = sum;
}






template <typename ClusterT>
__global__ void cluster_sizes_global_kernel(
    const ClusterT* __restrict__ clusters,
    int32_t* __restrict__ cluster_sizes,
    int32_t num_vertices)
{
    for (int u = blockIdx.x * blockDim.x + threadIdx.x; u < num_vertices;
         u += blockDim.x * gridDim.x)
        atomicAdd(&cluster_sizes[(int)__ldg(&clusters[u])], 1);
}


template <typename ClusterT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 6)
void tpv_direct_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const ClusterT* __restrict__ clusters,
    const int32_t* __restrict__ cluster_sizes,
    int32_t num_vertices,
    double* __restrict__ out)
{
    double local = 0.0;
    for (int u = blockIdx.x * BLOCK_SIZE + threadIdx.x; u < num_vertices;
         u += gridDim.x * BLOCK_SIZE) {
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

    
    __shared__ double s_sum[BLOCK_SIZE];
    s_sum[threadIdx.x] = local;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0 && s_sum[0] != 0.0) atomicAdd(out, s_sum[0]);
}


template <typename ClusterT, int BLOCK_SIZE>
__global__ __launch_bounds__(BLOCK_SIZE, 6)
void wpv_direct_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const ClusterT* __restrict__ clusters,
    const int32_t* __restrict__ cluster_sizes,
    int32_t num_vertices,
    double* __restrict__ out)
{
    const int lane = threadIdx.x & 31;
    constexpr int warps_per_block = BLOCK_SIZE >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int global_warp = blockIdx.x * warps_per_block + warp_in_block;
    const int total_warps = gridDim.x * warps_per_block;

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
        cu = __shfl_sync(0xFFFFFFFF, cu, 0);
        sz = __shfl_sync(0xFFFFFFFF, sz, 0);
        if (sz == 0) continue;

        float local_cut = 0.0f;
        for (int e = start + lane; e < end; e += 32) {
            int v = indices[e];
            float w = edge_weights[e];
            int cv = (int)__ldg(&clusters[v]);
            local_cut += (cu != cv) ? w : 0.0f;
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2)
            local_cut += __shfl_down_sync(0xFFFFFFFF, local_cut, offset);

        if (lane == 0 && local_cut != 0.0f)
            warp_accum += (double)local_cut / (double)sz;
    }

    __shared__ double s_warp[warps_per_block];
    if (lane == 0) s_warp[warp_in_block] = warp_accum;
    __syncthreads();

    if (threadIdx.x == 0) {
        double block_sum = 0.0;
        for (int w = 0; w < warps_per_block; ++w) block_sum += s_warp[w];
        if (block_sum != 0.0) atomicAdd(out, block_sum);
    }
}


static constexpr int32_t kMaxClustersSmem = 6000;

}  

double analyze_clustering_ratio_cut(const graph32_t& graph,
                                    const float* edge_weights,
                                    std::size_t num_clusters,
                                    const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    int32_t nc = static_cast<int32_t>(num_clusters);

    cache.ensure_sizes(nc);

    bool use_smem = (nc <= kMaxClustersSmem);

    cudaMemsetAsync(cache.sizes, 0, nc * sizeof(int32_t));

    
    float avg_degree = (num_vertices > 0) ? (float)num_edges / (float)num_vertices : 0.0f;
    bool use_warp = (avg_degree >= 16.0f);

    int grid_size;
    if (use_warp) {
        int warps_per_block = 8; 
        grid_size = std::min((num_vertices + warps_per_block - 1) / warps_per_block, 80 * 6);
    } else {
        grid_size = std::min((num_vertices + 255) / 256, 80 * 6);
    }

    const int32_t* off_ptr = graph.offsets;
    const int32_t* ind_ptr = graph.indices;
    int32_t* sz_ptr = cache.sizes;

    auto set_l2_persist = [](const void* ptr, size_t bytes) {
        cudaStreamAttrValue attr = {};
        attr.accessPolicyWindow.base_ptr = const_cast<void*>(ptr);
        attr.accessPolicyWindow.num_bytes = std::min(bytes, (size_t)(4*1024*1024));
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
    };

    auto reset_l2_persist = []() {
        cudaStreamAttrValue reset = {};
        reset.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &reset);
    };

    if (use_smem) {
        
        cache.ensure_cuts(nc);
        cudaMemsetAsync(cache.cuts, 0, nc * sizeof(double));
        double* cut_ptr = cache.cuts;

        if (nc <= 255) {
            cache.ensure_packed_u8(num_vertices);
            pack_clusters_u8_kernel<<<(num_vertices+255)/256, 256>>>(cluster_assignments, cache.packed_u8, num_vertices, nc);
            set_l2_persist(cache.packed_u8, num_vertices);
            int smem = nc * (sizeof(float) + sizeof(int32_t));
            if (use_warp)
                warp_per_vertex_kernel<uint8_t, 256><<<grid_size, 256, smem>>>(off_ptr, ind_ptr, edge_weights, cache.packed_u8, sz_ptr, cut_ptr, num_vertices, nc);
            else
                thread_per_vertex_kernel<uint8_t, 256><<<grid_size, 256, smem>>>(off_ptr, ind_ptr, edge_weights, cache.packed_u8, sz_ptr, cut_ptr, num_vertices, nc);
        } else {
            cache.ensure_packed_i16(num_vertices);
            pack_clusters_i16_kernel<<<(num_vertices+255)/256, 256>>>(cluster_assignments, cache.packed_i16, num_vertices, nc);
            set_l2_persist(cache.packed_i16, (size_t)num_vertices * 2);
            int smem = nc * (sizeof(float) + sizeof(int32_t));
            if (use_warp)
                warp_per_vertex_kernel<int16_t, 256><<<grid_size, 256, smem>>>(off_ptr, ind_ptr, edge_weights, cache.packed_i16, sz_ptr, cut_ptr, num_vertices, nc);
            else
                thread_per_vertex_kernel<int16_t, 256><<<grid_size, 256, smem>>>(off_ptr, ind_ptr, edge_weights, cache.packed_i16, sz_ptr, cut_ptr, num_vertices, nc);
        }

        reset_l2_persist();
        ratio_cut_final_kernel<<<1, 32>>>(cut_ptr, sz_ptr, cache.result_d, nc);
    } else {
        
        cudaMemsetAsync(cache.result_d, 0, sizeof(double));
        int grid_hist = std::min((num_vertices + 255) / 256, 4096);

        if (nc <= 32767) {
            cache.ensure_packed_i16(num_vertices);
            pack_clusters_i16_kernel<<<(num_vertices+255)/256, 256>>>(cluster_assignments, cache.packed_i16, num_vertices, nc);
            set_l2_persist(cache.packed_i16, (size_t)num_vertices * 2);
            cluster_sizes_global_kernel<int16_t><<<grid_hist, 256>>>(cache.packed_i16, sz_ptr, num_vertices);
            if (use_warp)
                wpv_direct_kernel<int16_t, 256><<<grid_size, 256, (256/32) * (int)sizeof(double)>>>(off_ptr, ind_ptr, edge_weights, cache.packed_i16, sz_ptr, num_vertices, cache.result_d);
            else
                tpv_direct_kernel<int16_t, 256><<<grid_size, 256>>>(off_ptr, ind_ptr, edge_weights, cache.packed_i16, sz_ptr, num_vertices, cache.result_d);
        } else {
            cluster_sizes_global_kernel<int32_t><<<grid_hist, 256>>>(cluster_assignments, sz_ptr, num_vertices);
            if (use_warp)
                wpv_direct_kernel<int32_t, 256><<<grid_size, 256, (256/32) * (int)sizeof(double)>>>(off_ptr, ind_ptr, edge_weights, cluster_assignments, sz_ptr, num_vertices, cache.result_d);
            else
                tpv_direct_kernel<int32_t, 256><<<grid_size, 256>>>(off_ptr, ind_ptr, edge_weights, cluster_assignments, sz_ptr, num_vertices, cache.result_d);
        }

        reset_l2_persist();
    }

    double host_result;
    cudaMemcpy(&host_result, cache.result_d, sizeof(double), cudaMemcpyDeviceToHost);
    return host_result;
}

}  
