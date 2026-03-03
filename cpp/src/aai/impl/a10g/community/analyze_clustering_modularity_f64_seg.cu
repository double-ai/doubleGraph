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
    double* d_intra = nullptr;
    double* d_cs = nullptr;
    int16_t* d_ca16 = nullptr;
    double* d_result = nullptr;
    size_t cs_cap = 0;
    size_t ca16_cap = 0;

    Cache() {
        cudaMalloc(&d_intra, sizeof(double));
        cudaMalloc(&d_result, sizeof(double));
        cs_cap = 4096;
        cudaMalloc(&d_cs, cs_cap * sizeof(double));
        ca16_cap = 64 * 1024 * 1024;
        cudaMalloc(&d_ca16, ca16_cap * sizeof(int16_t));
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 4 * 1024 * 1024);
    }

    ~Cache() override {
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
        if (d_intra) cudaFree(d_intra);
        if (d_cs) cudaFree(d_cs);
        if (d_ca16) cudaFree(d_ca16);
        if (d_result) cudaFree(d_result);
    }

    void ensure(int num_clusters, int32_t num_vertices) {
        if ((size_t)num_clusters > cs_cap) {
            cudaFree(d_cs);
            cs_cap = num_clusters;
            cudaMalloc(&d_cs, cs_cap * sizeof(double));
        }
        if ((size_t)num_vertices > ca16_cap) {
            cudaFree(d_ca16);
            ca16_cap = num_vertices;
            cudaMalloc(&d_ca16, ca16_cap * sizeof(int16_t));
        }
    }
};




__global__ void modularity_finalize(
    const double* __restrict__ d_intra_sum,
    const double* __restrict__ d_cluster_strength,
    int num_clusters, double* __restrict__ d_result)
{
    double intra = d_intra_sum[0];
    double two_m = 0.0, penalty = 0.0;
    for (int c = 0; c < num_clusters; c++) {
        double cs = d_cluster_strength[c];
        two_m += cs;
        penalty += cs * cs;
    }
    d_result[0] = (two_m > 0.0) ? (intra / two_m - penalty / (two_m * two_m)) : 0.0;
}






template <typename CA_T, int BLOCK_SIZE>
__global__ void modularity_high_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const CA_T* __restrict__ ca,
    double* __restrict__ d_intra_sum,
    double* __restrict__ d_cluster_strength,
    int seg_start, int seg_end)
{
    int v = seg_start + blockIdx.x;
    if (v >= seg_end) return;
    int32_t es = offsets[v], ee = offsets[v + 1];
    CA_T cv = ca[v];
    double lintra = 0.0, ldeg = 0.0;
    for (int32_t e = es + threadIdx.x; e < ee; e += BLOCK_SIZE) {
        double w = weights[e];
        ldeg += w;
        if (__ldg(&ca[indices[e]]) == cv) lintra += w;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        lintra += __shfl_down_sync(0xFFFFFFFF, lintra, off);
        ldeg += __shfl_down_sync(0xFFFFFFFF, ldeg, off);
    }
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int NW = BLOCK_SIZE / 32;
    __shared__ double si[NW], sd[NW];
    if (lane == 0) { si[warp] = lintra; sd[warp] = ldeg; }
    __syncthreads();
    if (threadIdx.x == 0) {
        double ti = 0.0, td = 0.0;
        for (int i = 0; i < NW; i++) { ti += si[i]; td += sd[i]; }
        atomicAdd(d_intra_sum, ti);
        atomicAdd(&d_cluster_strength[(int32_t)cv], td);
    }
}


template <typename CA_T, int WPB>
__global__ void modularity_mid_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const CA_T* __restrict__ ca,
    double* __restrict__ d_intra_sum,
    double* __restrict__ d_cluster_strength,
    int seg_start, int seg_end, int num_clusters)
{
    extern __shared__ double s_dyn[];
    double* s_cluster = s_dyn;
    double* s_intra = s_dyn + num_clusters;
    int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int v = seg_start + blockIdx.x * WPB + wib;
    for (int i = threadIdx.x; i < num_clusters; i += WPB * 32)
        s_cluster[i] = 0.0;
    __syncthreads();
    double wi = 0.0, wd = 0.0;
    CA_T cv = 0;
    if (v < seg_end) {
        int32_t es = offsets[v], ee = offsets[v + 1];
        cv = ca[v];
        for (int32_t e = es + lane; e < ee; e += 32) {
            double w = weights[e];
            wd += w;
            if (__ldg(&ca[indices[e]]) == cv) wi += w;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            wi += __shfl_down_sync(0xFFFFFFFF, wi, off);
            wd += __shfl_down_sync(0xFFFFFFFF, wd, off);
        }
    }
    if (lane == 0) s_intra[wib] = wi;
    __syncthreads();
    if (threadIdx.x == 0) {
        double s = 0.0;
        for (int i = 0; i < WPB; i++) s += s_intra[i];
        if (s != 0.0) atomicAdd(d_intra_sum, s);
    }
    if (lane == 0 && v < seg_end && wd != 0.0)
        atomicAdd(&s_cluster[(int32_t)cv], wd);
    __syncthreads();
    for (int i = threadIdx.x; i < num_clusters; i += WPB * 32) {
        double val = s_cluster[i];
        if (val != 0.0) atomicAdd(&d_cluster_strength[i], val);
    }
}


template <typename CA_T, int WPB>
__global__ void modularity_mid_degree_noshmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const CA_T* __restrict__ ca,
    double* __restrict__ d_intra_sum,
    double* __restrict__ d_cluster_strength,
    int seg_start, int seg_end)
{
    int wib = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int v = seg_start + blockIdx.x * WPB + wib;
    double wi = 0.0, wd = 0.0;
    CA_T cv = 0;
    if (v < seg_end) {
        int32_t es = offsets[v], ee = offsets[v + 1];
        cv = ca[v];
        for (int32_t e = es + lane; e < ee; e += 32) {
            double w = weights[e];
            wd += w;
            if (__ldg(&ca[indices[e]]) == cv) wi += w;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            wi += __shfl_down_sync(0xFFFFFFFF, wi, off);
            wd += __shfl_down_sync(0xFFFFFFFF, wd, off);
        }
    }
    __shared__ double s_intra[WPB];
    if (lane == 0) s_intra[wib] = wi;
    __syncthreads();
    if (threadIdx.x == 0) {
        double s = 0.0;
        for (int i = 0; i < WPB; i++) s += s_intra[i];
        if (s != 0.0) atomicAdd(d_intra_sum, s);
    }
    if (lane == 0 && v < seg_end && wd != 0.0)
        atomicAdd(&d_cluster_strength[(int32_t)cv], wd);
}


template <typename CA_T, int BS>
__global__ void modularity_low_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const CA_T* __restrict__ ca,
    double* __restrict__ d_intra_sum,
    double* __restrict__ d_cluster_strength,
    int seg_start, int seg_end, int num_clusters)
{
    extern __shared__ double s_dyn[];
    double* s_cluster = s_dyn;
    int v = seg_start + blockIdx.x * BS + threadIdx.x;
    for (int i = threadIdx.x; i < num_clusters; i += BS)
        s_cluster[i] = 0.0;
    __syncthreads();
    double lintra = 0.0, ldeg = 0.0;
    CA_T cv = 0;
    bool active = (v < seg_end);
    if (active) {
        int32_t es = offsets[v], ee = offsets[v + 1];
        cv = ca[v];
        for (int32_t e = es; e < ee; e++) {
            double w = weights[e];
            ldeg += w;
            if (__ldg(&ca[indices[e]]) == cv) lintra += w;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        lintra += __shfl_down_sync(0xFFFFFFFF, lintra, off);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int NW = BS / 32;
    __shared__ double s_w[32];
    if (lane == 0) s_w[warp] = lintra;
    __syncthreads();
    if (threadIdx.x == 0) {
        double s = 0.0;
        for (int i = 0; i < NW; i++) s += s_w[i];
        if (s != 0.0) atomicAdd(d_intra_sum, s);
    }
    if (active && ldeg != 0.0)
        atomicAdd(&s_cluster[(int32_t)cv], ldeg);
    __syncthreads();
    for (int i = threadIdx.x; i < num_clusters; i += BS) {
        double val = s_cluster[i];
        if (val != 0.0) atomicAdd(&d_cluster_strength[i], val);
    }
}


template <typename CA_T, int BS>
__global__ void modularity_low_degree_noshmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const CA_T* __restrict__ ca,
    double* __restrict__ d_intra_sum,
    double* __restrict__ d_cluster_strength,
    int seg_start, int seg_end)
{
    int v = seg_start + blockIdx.x * BS + threadIdx.x;
    double lintra = 0.0, ldeg = 0.0;
    CA_T cv = 0;
    bool active = (v < seg_end);
    if (active) {
        int32_t es = offsets[v], ee = offsets[v + 1];
        cv = ca[v];
        for (int32_t e = es; e < ee; e++) {
            double w = weights[e];
            ldeg += w;
            if (__ldg(&ca[indices[e]]) == cv) lintra += w;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        lintra += __shfl_down_sync(0xFFFFFFFF, lintra, off);
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int NW = BS / 32;
    __shared__ double s_w[32];
    if (lane == 0) s_w[warp] = lintra;
    __syncthreads();
    if (threadIdx.x == 0) {
        double s = 0.0;
        for (int i = 0; i < NW; i++) s += s_w[i];
        if (s != 0.0) atomicAdd(d_intra_sum, s);
    }
    if (active && ldeg != 0.0)
        atomicAdd(&d_cluster_strength[(int32_t)cv], ldeg);
}




__global__ void compress_i32_to_i16(const int32_t* __restrict__ in, int16_t* __restrict__ out, int n) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (base + 1 < n) {
        int2 v = reinterpret_cast<const int2*>(in + base)[0];
        uint32_t packed = (uint32_t)(uint16_t)(int16_t)v.x | ((uint32_t)(uint16_t)(int16_t)v.y << 16);
        reinterpret_cast<uint32_t*>(out)[base / 2] = packed;
    } else if (base < n) {
        out[base] = (int16_t)in[base];
    }
}

}  

double analyze_clustering_modularity_seg(const graph32_t& graph,
                                         const double* edge_weights,
                                         std::size_t num_clusters,
                                         const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int nc = static_cast<int>(num_clusters);

    cache.ensure(nc, num_vertices);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const double* d_weights = edge_weights;
    const int32_t* d_clustering = cluster_assignments;

    const auto& seg = graph.segment_offsets.value();

    bool use_i16 = (nc <= 32767);

    
    void* persist_ptr;
    size_t data_bytes;
    if (use_i16) {
        persist_ptr = (void*)cache.d_ca16;
        data_bytes = (size_t)num_vertices * sizeof(int16_t);
    } else {
        persist_ptr = (void*)d_clustering;
        data_bytes = (size_t)num_vertices * sizeof(int32_t);
    }
    size_t persist_bytes = std::min(data_bytes, (size_t)(4 * 1024 * 1024));

    cudaStreamAttrValue attr = {};
    attr.accessPolicyWindow.base_ptr = persist_ptr;
    attr.accessPolicyWindow.num_bytes = persist_bytes;
    attr.accessPolicyWindow.hitRatio = 1.0f;
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);

    
    cudaMemsetAsync(cache.d_intra, 0, sizeof(double));
    cudaMemsetAsync(cache.d_cs, 0, nc * sizeof(double));

    if (use_i16) {
        int non_isolated = seg[3];
        if (non_isolated > 0) {
            int threads = 256;
            int blocks = ((non_isolated + 1) / 2 + threads - 1) / threads;
            if (blocks <= 0) blocks = 1;
            compress_i32_to_i16<<<blocks, threads>>>(d_clustering, cache.d_ca16, non_isolated);
        }

        
        {
            int n = seg[1] - seg[0];
            if (n > 0)
                modularity_high_degree<int16_t, 256><<<n, 256>>>(
                    d_offsets, d_indices, d_weights, cache.d_ca16,
                    cache.d_intra, cache.d_cs, seg[0], seg[1]);
        }
        
        {
            int n = seg[2] - seg[1];
            if (n > 0) {
                constexpr int WPB = 8;
                if (nc <= 4096) {
                    modularity_mid_degree<int16_t, WPB><<<(n+WPB-1)/WPB, WPB*32, (nc+WPB)*sizeof(double)>>>(
                        d_offsets, d_indices, d_weights, cache.d_ca16,
                        cache.d_intra, cache.d_cs, seg[1], seg[2], nc);
                } else {
                    modularity_mid_degree_noshmem<int16_t, WPB><<<(n+WPB-1)/WPB, WPB*32>>>(
                        d_offsets, d_indices, d_weights, cache.d_ca16,
                        cache.d_intra, cache.d_cs, seg[1], seg[2]);
                }
            }
        }
        
        {
            int n = seg[3] - seg[2];
            if (n > 0) {
                constexpr int BS = 256;
                if (nc <= 4096) {
                    modularity_low_degree<int16_t, BS><<<(n+BS-1)/BS, BS, nc*sizeof(double)>>>(
                        d_offsets, d_indices, d_weights, cache.d_ca16,
                        cache.d_intra, cache.d_cs, seg[2], seg[3], nc);
                } else {
                    modularity_low_degree_noshmem<int16_t, BS><<<(n+BS-1)/BS, BS>>>(
                        d_offsets, d_indices, d_weights, cache.d_ca16,
                        cache.d_intra, cache.d_cs, seg[2], seg[3]);
                }
            }
        }
    } else {
        
        
        {
            int n = seg[1] - seg[0];
            if (n > 0)
                modularity_high_degree<int32_t, 256><<<n, 256>>>(
                    d_offsets, d_indices, d_weights, d_clustering,
                    cache.d_intra, cache.d_cs, seg[0], seg[1]);
        }
        
        {
            int n = seg[2] - seg[1];
            if (n > 0) {
                constexpr int WPB = 8;
                if (nc <= 4096) {
                    modularity_mid_degree<int32_t, WPB><<<(n+WPB-1)/WPB, WPB*32, (nc+WPB)*sizeof(double)>>>(
                        d_offsets, d_indices, d_weights, d_clustering,
                        cache.d_intra, cache.d_cs, seg[1], seg[2], nc);
                } else {
                    modularity_mid_degree_noshmem<int32_t, WPB><<<(n+WPB-1)/WPB, WPB*32>>>(
                        d_offsets, d_indices, d_weights, d_clustering,
                        cache.d_intra, cache.d_cs, seg[1], seg[2]);
                }
            }
        }
        
        {
            int n = seg[3] - seg[2];
            if (n > 0) {
                constexpr int BS = 256;
                if (nc <= 4096) {
                    modularity_low_degree<int32_t, BS><<<(n+BS-1)/BS, BS, nc*sizeof(double)>>>(
                        d_offsets, d_indices, d_weights, d_clustering,
                        cache.d_intra, cache.d_cs, seg[2], seg[3], nc);
                } else {
                    modularity_low_degree_noshmem<int32_t, BS><<<(n+BS-1)/BS, BS>>>(
                        d_offsets, d_indices, d_weights, d_clustering,
                        cache.d_intra, cache.d_cs, seg[2], seg[3]);
                }
            }
        }
    }

    modularity_finalize<<<1, 1>>>(cache.d_intra, cache.d_cs, nc, cache.d_result);

    
    attr.accessPolicyWindow.num_bytes = 0;
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);

    
    double result;
    cudaMemcpy(&result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

}  
