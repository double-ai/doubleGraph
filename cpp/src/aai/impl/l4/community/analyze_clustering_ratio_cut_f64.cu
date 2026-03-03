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
#include <cstring>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* cluster_sizes = nullptr;
    int32_t cluster_sizes_capacity = 0;

    double* result_d = nullptr;
    bool result_allocated = false;

    int32_t* assign_bits2 = nullptr;
    int32_t assign_bits2_capacity = 0;

    uint8_t* assign_u8 = nullptr;
    int32_t assign_u8_capacity = 0;

    int16_t* assign_i16 = nullptr;
    int32_t assign_i16_capacity = 0;

    void ensure(int32_t num_clusters, int32_t num_vertices) {
        if (cluster_sizes_capacity < num_clusters) {
            if (cluster_sizes) cudaFree(cluster_sizes);
            cudaMalloc(&cluster_sizes, (size_t)num_clusters * sizeof(int32_t));
            cluster_sizes_capacity = num_clusters;
        }

        if (!result_allocated) {
            cudaMalloc(&result_d, sizeof(double));
            result_allocated = true;
        }

        if (num_clusters == 2) {
            int32_t nwords = (num_vertices + 31) / 32;
            if (assign_bits2_capacity < nwords) {
                if (assign_bits2) cudaFree(assign_bits2);
                cudaMalloc(&assign_bits2, (size_t)nwords * sizeof(int32_t));
                assign_bits2_capacity = nwords;
            }
        } else if (num_clusters <= 255) {
            if (assign_u8_capacity < num_vertices) {
                if (assign_u8) cudaFree(assign_u8);
                cudaMalloc(&assign_u8, (size_t)num_vertices * sizeof(uint8_t));
                assign_u8_capacity = num_vertices;
            }
        } else if (num_clusters <= 32767) {
            if (assign_i16_capacity < num_vertices) {
                if (assign_i16) cudaFree(assign_i16);
                cudaMalloc(&assign_i16, (size_t)num_vertices * sizeof(int16_t));
                assign_i16_capacity = num_vertices;
            }
        }
    }

    ~Cache() override {
        if (cluster_sizes) cudaFree(cluster_sizes);
        if (result_d) cudaFree(result_d);
        if (assign_bits2) cudaFree(assign_bits2);
        if (assign_u8) cudaFree(assign_u8);
        if (assign_i16) cudaFree(assign_i16);
    }
};







__global__ __launch_bounds__(256, 6)
void histogram_pack2(
    const int32_t* __restrict__ assign_i32,
    int32_t* __restrict__ assign_bits,
    int32_t* __restrict__ cluster_sizes, 
    int32_t n)
{
    __shared__ int32_t warp_c0[8];
    __shared__ int32_t warp_c1[8];

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;

    int warps_per_block = (int)(blockDim.x >> 5);
    int global_warp = (int)(blockIdx.x * warps_per_block + warp);
    int total_warps = (int)(gridDim.x * warps_per_block);

    int32_t local0 = 0;
    int32_t local1 = 0;

    
    for (int32_t base = global_warp * 32; base < n; base += total_warps * 32) {
        int32_t idx = base + lane;
        unsigned active = __ballot_sync(0xffffffff, idx < n);
        int c = (idx < n) ? assign_i32[idx] : 0;
        unsigned m1 = __ballot_sync(active, c != 0);
        if (lane == 0) {
            assign_bits[(uint32_t)(base >> 5)] = (int32_t)m1; 
            int n1 = __popc(m1);
            int na = __popc(active);
            local1 += n1;
            local0 += (na - n1);
        }
    }

    if (lane == 0) {
        warp_c0[warp] = local0;
        warp_c1[warp] = local1;
    }
    __syncthreads();

    if (warp == 0) {
        int v0 = (lane < warps_per_block) ? warp_c0[lane] : 0;
        int v1 = (lane < warps_per_block) ? warp_c1[lane] : 0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            v0 += __shfl_down_sync(0xffffffff, v0, offset);
            v1 += __shfl_down_sync(0xffffffff, v1, offset);
        }
        if (lane == 0) {
            atomicAdd(&cluster_sizes[0], v0);
            atomicAdd(&cluster_sizes[1], v1);
        }
    }
}

__global__ __launch_bounds__(256, 6)
void histogram_smem_convert_i16(
    const int32_t* __restrict__ assign_i32,
    int16_t* __restrict__ assign_i16,
    int32_t* __restrict__ cluster_sizes,
    int32_t n,
    int32_t num_clusters)
{
    extern __shared__ int32_t s_counts[];

    for (int k = threadIdx.x; k < num_clusters; k += blockDim.x) s_counts[k] = 0;
    __syncthreads();

    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(blockDim.x * gridDim.x);

    for (int i = idx; i < n; i += stride) {
        int c = __ldg(assign_i32 + i);
        if ((uint32_t)c >= (uint32_t)num_clusters) c = (c < 0) ? 0 : (num_clusters - 1);
        assign_i16[i] = (int16_t)c;
        atomicAdd(&s_counts[c], 1);
    }

    __syncthreads();

    for (int k = threadIdx.x; k < num_clusters; k += blockDim.x) {
        int v = s_counts[k];
        if (v) atomicAdd(&cluster_sizes[k], v);
    }
}

__global__ __launch_bounds__(256, 6)
void histogram_smem_convert_u8(
    const int32_t* __restrict__ assign_i32,
    uint8_t* __restrict__ assign_u8,
    int32_t* __restrict__ cluster_sizes,
    int32_t n,
    int32_t num_clusters)
{
    extern __shared__ int32_t s_counts[];

    for (int k = threadIdx.x; k < num_clusters; k += blockDim.x) s_counts[k] = 0;
    __syncthreads();

    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(blockDim.x * gridDim.x);

    for (int i = idx; i < n; i += stride) {
        int c = __ldg(assign_i32 + i);
        if ((uint32_t)c >= (uint32_t)num_clusters) c = (c < 0) ? 0 : (num_clusters - 1);
        assign_u8[i] = (uint8_t)c;
        atomicAdd(&s_counts[c], 1);
    }

    __syncthreads();

    for (int k = threadIdx.x; k < num_clusters; k += blockDim.x) {
        int v = s_counts[k];
        if (v) atomicAdd(&cluster_sizes[k], v);
    }
}

__global__ __launch_bounds__(256, 6)
void histogram_smem(
    const int32_t* __restrict__ assign,
    int32_t* __restrict__ cluster_sizes,
    int32_t n,
    int32_t num_clusters)
{
    extern __shared__ int32_t s_counts[];

    for (int k = threadIdx.x; k < num_clusters; k += blockDim.x) s_counts[k] = 0;
    __syncthreads();

    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(blockDim.x * gridDim.x);

    for (int i = idx; i < n; i += stride) {
        int c = __ldg(assign + i);
        if ((uint32_t)c >= (uint32_t)num_clusters) c = (c < 0) ? 0 : (num_clusters - 1);
        atomicAdd(&s_counts[c], 1);
    }

    __syncthreads();

    for (int k = threadIdx.x; k < num_clusters; k += blockDim.x) {
        int v = s_counts[k];
        if (v) atomicAdd(&cluster_sizes[k], v);
    }
}

__global__ __launch_bounds__(256, 6)
void histogram_gmem(
    const int32_t* __restrict__ assign,
    int32_t* __restrict__ cluster_sizes,
    int32_t n,
    int32_t num_clusters)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(blockDim.x * gridDim.x);
    for (int i = idx; i < n; i += stride) {
        int c = __ldg(assign + i);
        if ((uint32_t)c >= (uint32_t)num_clusters) c = (c < 0) ? 0 : (num_clusters - 1);
        atomicAdd(&cluster_sizes[c], 1);
    }
}

__global__ __launch_bounds__(256, 6)
void histogram_gmem_convert_i16(
    const int32_t* __restrict__ assign_i32,
    int16_t* __restrict__ assign_i16,
    int32_t* __restrict__ cluster_sizes,
    int32_t n,
    int32_t num_clusters)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(blockDim.x * gridDim.x);
    for (int i = idx; i < n; i += stride) {
        int c = __ldg(assign_i32 + i);
        if ((uint32_t)c >= (uint32_t)num_clusters) c = (c < 0) ? 0 : (num_clusters - 1);
        assign_i16[i] = (int16_t)c;
        atomicAdd(&cluster_sizes[c], 1);
    }
}

__global__ __launch_bounds__(256, 6)
void histogram_gmem_convert_u8(
    const int32_t* __restrict__ assign_i32,
    uint8_t* __restrict__ assign_u8,
    int32_t* __restrict__ cluster_sizes,
    int32_t n,
    int32_t num_clusters)
{
    int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(blockDim.x * gridDim.x);
    for (int i = idx; i < n; i += stride) {
        int c = __ldg(assign_i32 + i);
        if ((uint32_t)c >= (uint32_t)num_clusters) c = (c < 0) ? 0 : (num_clusters - 1);
        assign_u8[i] = (uint8_t)c;
        atomicAdd(&cluster_sizes[c], 1);
    }
}





__device__ __forceinline__ int32_t find_source_vertex_binsearch(
    const int32_t* __restrict__ offsets,
    int32_t e,
    int32_t num_vertices)
{
    int32_t lo = 0, hi = num_vertices;
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo + 1) >> 1);
        if (__ldg(offsets + mid) <= e) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

template <typename AssignT>
__device__ __forceinline__ int32_t load_cluster(const AssignT* p, int32_t idx) {
    return (int32_t)__ldg(p + idx);
}

__device__ __forceinline__ int32_t load_cluster_bits2(const int32_t* bits, int32_t idx) {
    uint32_t w = (uint32_t)bits[(uint32_t)idx >> 5];
    return (int32_t)((w >> (idx & 31)) & 1u);
}


__global__ __launch_bounds__(256, 6)
void ratio_cut_edge_warp_kernel_2c(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ assign_bits,
    const int32_t* __restrict__ cluster_sizes,
    int32_t num_vertices,
    int32_t num_edges,
    double* __restrict__ result)
{
    __shared__ double inv0;
    __shared__ double inv1;
    __shared__ double warp_sums[8];

    if (threadIdx.x == 0) {
        int32_t s0 = cluster_sizes[0];
        int32_t s1 = cluster_sizes[1];
        inv0 = (s0 > 0) ? (1.0 / (double)s0) : 0.0;
        inv1 = (s1 > 0) ? (1.0 / (double)s1) : 0.0;
    }
    __syncthreads();

    const int lane = (int)(threadIdx.x & 31);
    const int warp_in_block = (int)(threadIdx.x >> 5);

    const int global_warp_id = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int num_warps = (int)((blockDim.x * gridDim.x) >> 5);

    int32_t edges_per_warp = (num_edges + num_warps - 1) / num_warps;
    int32_t e_start = global_warp_id * edges_per_warp;
    int32_t e_end = e_start + edges_per_warp;
    if (e_end > num_edges) e_end = num_edges;

    double sum = 0.0;

    if (e_start < num_edges) {
        int32_t u0;
        if (lane == 0) u0 = find_source_vertex_binsearch(offsets, e_start, num_vertices);
        u0 = __shfl_sync(0xffffffff, u0, 0);

        int32_t e0 = e_start + lane;
        int32_t u = u0;
        if (e0 < e_end) {
            int32_t u_next_edge = __ldg(offsets + u + 1);
            while (u_next_edge <= e0 && u < num_vertices - 1) {
                ++u;
                u_next_edge = __ldg(offsets + u + 1);
            }
        }

        int32_t u_end_edge = (u < num_vertices) ? __ldg(offsets + u + 1) : num_edges;
        int32_t c_u = (e0 < e_end) ? load_cluster_bits2(assign_bits, u) : 0;
        double inv_u = (e0 < e_end) ? (c_u ? inv1 : inv0) : 0.0;

        for (int32_t e = e0; e < e_end; e += 32) {
            while (e >= u_end_edge && u < num_vertices - 1) {
                ++u;
                u_end_edge = __ldg(offsets + u + 1);
                c_u = load_cluster_bits2(assign_bits, u);
                inv_u = c_u ? inv1 : inv0;
            }

            int32_t v = __ldg(indices + e);
            
            uint32_t word_idx = (uint32_t)v >> 5;
            unsigned am = __activemask();
            unsigned mm = __match_any_sync(am, word_idx);
            int leader = __ffs((int)mm) - 1;
            uint32_t word;
            if (lane == leader) word = (uint32_t)assign_bits[word_idx];
            word = __shfl_sync(mm, word, leader);
            int32_t c_v = (int32_t)((word >> (v & 31)) & 1u);
            if (c_u != c_v) sum += __ldg(weights + e) * inv_u;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) warp_sums[warp_in_block] = sum;
    __syncthreads();

    if (warp_in_block == 0) {
        double v = (lane < 8) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
        if (lane == 0 && v != 0.0) atomicAdd(result, v);
    }
}

__global__ __launch_bounds__(256, 6)
void ratio_cut_vertex_kernel_2c(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ assign_bits,
    const int32_t* __restrict__ cluster_sizes,
    int32_t num_vertices,
    double* __restrict__ result)
{
    __shared__ double inv0;
    __shared__ double inv1;
    __shared__ double warp_sums[8];

    if (threadIdx.x == 0) {
        int32_t s0 = cluster_sizes[0];
        int32_t s1 = cluster_sizes[1];
        inv0 = (s0 > 0) ? (1.0 / (double)s0) : 0.0;
        inv1 = (s1 > 0) ? (1.0 / (double)s1) : 0.0;
    }
    __syncthreads();

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp_in_block = tid >> 5;

    int idx = (int)(blockIdx.x * blockDim.x + tid);
    int stride = (int)(blockDim.x * gridDim.x);

    double sum = 0.0;

    for (int32_t u = idx; u < num_vertices; u += stride) {
        int32_t c_u = load_cluster_bits2(assign_bits, u);
        double inv_u = c_u ? inv1 : inv0;
        int32_t start = __ldg(offsets + u);
        int32_t end = __ldg(offsets + u + 1);
        for (int32_t e = start; e < end; ++e) {
            int32_t v = __ldg(indices + e);
            int32_t c_v = load_cluster_bits2(assign_bits, v);
            if (c_u != c_v) sum += __ldg(weights + e) * inv_u;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) warp_sums[warp_in_block] = sum;
    __syncthreads();

    if (warp_in_block == 0) {
        double v = (lane < 8) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
        if (lane == 0 && v != 0.0) atomicAdd(result, v);
    }
}



template <typename AssignT>
__global__ __launch_bounds__(256, 6)
void ratio_cut_edge_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const AssignT* __restrict__ assign,
    const int32_t* __restrict__ cluster_sizes,
    int32_t num_vertices,
    int32_t num_edges,
    int32_t num_clusters,
    double* __restrict__ result)
{
    __shared__ double warp_sums[8];
    extern __shared__ double s_inv[];

    for (int k = threadIdx.x; k < num_clusters; k += 256) {
        int32_t sz = __ldg(cluster_sizes + k);
        s_inv[k] = (sz > 0) ? (1.0 / (double)sz) : 0.0;
    }
    __syncthreads();

    const int lane = (int)(threadIdx.x & 31);
    const int warp_in_block = (int)(threadIdx.x >> 5);

    const int global_warp_id = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int num_warps = (int)((blockDim.x * gridDim.x) >> 5);

    int32_t edges_per_warp = (num_edges + num_warps - 1) / num_warps;
    int32_t e_start = global_warp_id * edges_per_warp;
    int32_t e_end = e_start + edges_per_warp;
    if (e_end > num_edges) e_end = num_edges;

    double sum = 0.0;

    if (e_start < num_edges) {
        int32_t u0;
        if (lane == 0) u0 = find_source_vertex_binsearch(offsets, e_start, num_vertices);
        u0 = __shfl_sync(0xffffffff, u0, 0);

        int32_t e0 = e_start + lane;
        int32_t u = u0;
        if (e0 < e_end) {
            int32_t u_next_edge = __ldg(offsets + u + 1);
            while (u_next_edge <= e0 && u < num_vertices - 1) {
                ++u;
                u_next_edge = __ldg(offsets + u + 1);
            }
        }

        int32_t u_end_edge = (u < num_vertices) ? __ldg(offsets + u + 1) : num_edges;
        int32_t c_u = (e0 < e_end) ? load_cluster(assign, u) : 0;
        double inv_u = (e0 < e_end) ? s_inv[c_u] : 0.0;

        for (int32_t e = e0; e < e_end; e += 32) {
            while (e >= u_end_edge && u < num_vertices - 1) {
                ++u;
                u_end_edge = __ldg(offsets + u + 1);
                c_u = load_cluster(assign, u);
                inv_u = s_inv[c_u];
            }

            int32_t v = __ldg(indices + e);
            int32_t c_v = load_cluster(assign, v);
            if (c_u != c_v) sum += __ldg(weights + e) * inv_u;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) warp_sums[warp_in_block] = sum;
    __syncthreads();

    if (warp_in_block == 0) {
        double v = (lane < 8) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
        if (lane == 0 && v != 0.0) atomicAdd(result, v);
    }
}

template <typename AssignT>
__global__ __launch_bounds__(256, 6)
void ratio_cut_vertex_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const AssignT* __restrict__ assign,
    const int32_t* __restrict__ cluster_sizes,
    int32_t num_vertices,
    int32_t num_clusters,
    double* __restrict__ result)
{
    __shared__ double warp_sums[8];
    extern __shared__ double s_inv[];

    for (int k = threadIdx.x; k < num_clusters; k += 256) {
        int32_t sz = __ldg(cluster_sizes + k);
        s_inv[k] = (sz > 0) ? (1.0 / (double)sz) : 0.0;
    }
    __syncthreads();

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp_in_block = tid >> 5;

    int idx = (int)(blockIdx.x * blockDim.x + tid);
    int stride = (int)(blockDim.x * gridDim.x);

    double sum = 0.0;

    for (int32_t u = idx; u < num_vertices; u += stride) {
        int32_t c_u = load_cluster(assign, u);
        double inv_u = s_inv[c_u];
        int32_t start = __ldg(offsets + u);
        int32_t end = __ldg(offsets + u + 1);
        for (int32_t e = start; e < end; ++e) {
            int32_t v = __ldg(indices + e);
            int32_t c_v = load_cluster(assign, v);
            if (c_u != c_v) sum += __ldg(weights + e) * inv_u;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) warp_sums[warp_in_block] = sum;
    __syncthreads();

    if (warp_in_block == 0) {
        double v = (lane < 8) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
        if (lane == 0 && v != 0.0) atomicAdd(result, v);
    }
}






template <typename AssignT>
__global__ __launch_bounds__(256, 6)
void ratio_cut_edge_warp_kernel_bigK(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const AssignT* __restrict__ assign,
    const int32_t* __restrict__ cluster_sizes,
    int32_t num_vertices,
    int32_t num_edges,
    double* __restrict__ result)
{
    __shared__ double warp_sums[8];

    const int lane = (int)(threadIdx.x & 31);
    const int warp_in_block = (int)(threadIdx.x >> 5);

    const int global_warp_id = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int num_warps = (int)((blockDim.x * gridDim.x) >> 5);

    int32_t edges_per_warp = (num_edges + num_warps - 1) / num_warps;
    int32_t e_start = global_warp_id * edges_per_warp;
    int32_t e_end = e_start + edges_per_warp;
    if (e_end > num_edges) e_end = num_edges;

    double sum = 0.0;

    if (e_start < num_edges) {
        int32_t u0;
        if (lane == 0) u0 = find_source_vertex_binsearch(offsets, e_start, num_vertices);
        u0 = __shfl_sync(0xffffffff, u0, 0);

        int32_t e0 = e_start + lane;
        int32_t u = u0;
        if (e0 < e_end) {
            int32_t u_next_edge = __ldg(offsets + u + 1);
            while (u_next_edge <= e0 && u < num_vertices - 1) {
                ++u;
                u_next_edge = __ldg(offsets + u + 1);
            }
        }

        int32_t u_end_edge = (u < num_vertices) ? __ldg(offsets + u + 1) : num_edges;
        int32_t c_u = (e0 < e_end) ? load_cluster(assign, u) : 0;
        int32_t sz = (e0 < e_end) ? cluster_sizes[c_u] : 0;
        double inv_u = (sz > 0) ? (1.0 / (double)sz) : 0.0;

        for (int32_t e = e0; e < e_end; e += 32) {
            while (e >= u_end_edge && u < num_vertices - 1) {
                ++u;
                u_end_edge = __ldg(offsets + u + 1);
                c_u = load_cluster(assign, u);
                sz = cluster_sizes[c_u];
                inv_u = (sz > 0) ? (1.0 / (double)sz) : 0.0;
            }

            int32_t v = __ldg(indices + e);
            int32_t c_v = load_cluster(assign, v);
            if (c_u != c_v) sum += __ldg(weights + e) * inv_u;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) warp_sums[warp_in_block] = sum;
    __syncthreads();

    if (warp_in_block == 0) {
        double v = (lane < 8) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
        if (lane == 0 && v != 0.0) atomicAdd(result, v);
    }
}

template <typename AssignT>
__global__ __launch_bounds__(256, 6)
void ratio_cut_vertex_kernel_bigK(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const AssignT* __restrict__ assign,
    const int32_t* __restrict__ cluster_sizes,
    int32_t num_vertices,
    double* __restrict__ result)
{
    __shared__ double warp_sums[8];

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp_in_block = tid >> 5;

    int idx = (int)(blockIdx.x * blockDim.x + tid);
    int stride = (int)(blockDim.x * gridDim.x);

    double sum = 0.0;

    for (int32_t u = idx; u < num_vertices; u += stride) {
        int32_t c_u = load_cluster(assign, u);
        int32_t sz = cluster_sizes[c_u];
        double inv_u = (sz > 0) ? (1.0 / (double)sz) : 0.0;
        int32_t start = __ldg(offsets + u);
        int32_t end = __ldg(offsets + u + 1);
        for (int32_t e = start; e < end; ++e) {
            int32_t v = __ldg(indices + e);
            int32_t c_v = load_cluster(assign, v);
            if (c_u != c_v) sum += __ldg(weights + e) * inv_u;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) warp_sums[warp_in_block] = sum;
    __syncthreads();

    if (warp_in_block == 0) {
        double v = (lane < 8) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
        if (lane == 0 && v != 0.0) atomicAdd(result, v);
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
    int32_t nc = static_cast<int32_t>(num_clusters);

    cache.ensure(nc, num_vertices);

    const int16_t* d_assign_i16 = nullptr;
    const uint8_t* d_assign_u8 = nullptr;
    const int32_t* d_assign_bits2 = nullptr;

    if (nc == 2) {
        d_assign_bits2 = cache.assign_bits2;
    } else if (nc <= 255) {
        d_assign_u8 = cache.assign_u8;
    } else if (nc <= 32767) {
        d_assign_i16 = cache.assign_i16;
    }

    cudaStream_t stream = 0;
    cudaMemsetAsync(cache.cluster_sizes, 0, (size_t)nc * sizeof(int32_t), stream);
    cudaMemsetAsync(cache.result_d, 0, sizeof(double), stream);

    
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 512) grid = 512;

        constexpr int GMEM_CLUSTER_THRESHOLD = 4096;
        if (nc == 2 && d_assign_bits2) {
            if (grid > 2048) grid = 2048;
            histogram_pack2<<<grid, block, 0, stream>>>(
                cluster_assignments, (int32_t*)d_assign_bits2, cache.cluster_sizes, num_vertices);
        } else if (nc > GMEM_CLUSTER_THRESHOLD) {
            if (grid > 2048) grid = 2048;
            if (d_assign_u8) {
                histogram_gmem_convert_u8<<<grid, block, 0, stream>>>(
                    cluster_assignments, (uint8_t*)d_assign_u8, cache.cluster_sizes, num_vertices, nc);
            } else if (d_assign_i16) {
                histogram_gmem_convert_i16<<<grid, block, 0, stream>>>(
                    cluster_assignments, (int16_t*)d_assign_i16, cache.cluster_sizes, num_vertices, nc);
            } else {
                histogram_gmem<<<grid, block, 0, stream>>>(cluster_assignments, cache.cluster_sizes, num_vertices, nc);
            }
        } else {
            size_t smem = (size_t)nc * sizeof(int32_t);
            if (d_assign_u8) {
                histogram_smem_convert_u8<<<grid, block, smem, stream>>>(
                    cluster_assignments, (uint8_t*)d_assign_u8, cache.cluster_sizes, num_vertices, nc);
            } else if (d_assign_i16) {
                histogram_smem_convert_i16<<<grid, block, smem, stream>>>(
                    cluster_assignments, (int16_t*)d_assign_i16, cache.cluster_sizes, num_vertices, nc);
            } else {
                histogram_smem<<<grid, block, smem, stream>>>(cluster_assignments, cache.cluster_sizes, num_vertices, nc);
            }
        }
    }

    
    double avg_degree = (num_vertices > 0) ? ((double)num_edges / (double)num_vertices) : 0.0;

    constexpr int INV_SMEM_THRESHOLD = 5800;
    const bool use_smem_inv = (nc <= INV_SMEM_THRESHOLD);

    if (avg_degree >= 4.0) {
        int block = 256;
        int warps_per_block = 8;
        int desired_warps = (nc == 2) ? ((num_edges + 127) / 128) : ((num_edges + 63) / 64);
        int grid = (desired_warps + warps_per_block - 1) / warps_per_block;
        if (grid > 2048) grid = 2048;
        if (grid < 1) grid = 1;

        if (nc == 2 && d_assign_bits2) {
            ratio_cut_edge_warp_kernel_2c<<<grid, block, 0, stream>>>(
                offsets, indices, edge_weights, d_assign_bits2, cache.cluster_sizes,
                num_vertices, num_edges, cache.result_d);
        } else if (d_assign_u8) {
            if (use_smem_inv) {
                size_t smem_inv = (size_t)nc * sizeof(double);
                ratio_cut_edge_warp_kernel<uint8_t><<<grid, block, smem_inv, stream>>>(
                    offsets, indices, edge_weights, d_assign_u8, cache.cluster_sizes,
                    num_vertices, num_edges, nc, cache.result_d);
            } else {
                ratio_cut_edge_warp_kernel_bigK<uint8_t><<<grid, block, 0, stream>>>(
                    offsets, indices, edge_weights, d_assign_u8, cache.cluster_sizes,
                    num_vertices, num_edges, cache.result_d);
            }
        } else if (d_assign_i16) {
            if (use_smem_inv) {
                size_t smem_inv = (size_t)nc * sizeof(double);
                ratio_cut_edge_warp_kernel<int16_t><<<grid, block, smem_inv, stream>>>(
                    offsets, indices, edge_weights, d_assign_i16, cache.cluster_sizes,
                    num_vertices, num_edges, nc, cache.result_d);
            } else {
                ratio_cut_edge_warp_kernel_bigK<int16_t><<<grid, block, 0, stream>>>(
                    offsets, indices, edge_weights, d_assign_i16, cache.cluster_sizes,
                    num_vertices, num_edges, cache.result_d);
            }
        } else {
            if (use_smem_inv) {
                size_t smem_inv = (size_t)nc * sizeof(double);
                ratio_cut_edge_warp_kernel<int32_t><<<grid, block, smem_inv, stream>>>(
                    offsets, indices, edge_weights, cluster_assignments, cache.cluster_sizes,
                    num_vertices, num_edges, nc, cache.result_d);
            } else {
                ratio_cut_edge_warp_kernel_bigK<int32_t><<<grid, block, 0, stream>>>(
                    offsets, indices, edge_weights, cluster_assignments, cache.cluster_sizes,
                    num_vertices, num_edges, cache.result_d);
            }
        }
    } else {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 4096) grid = 4096;
        if (grid < 1) grid = 1;

        if (nc == 2 && d_assign_bits2) {
            ratio_cut_vertex_kernel_2c<<<grid, block, 0, stream>>>(
                offsets, indices, edge_weights, d_assign_bits2, cache.cluster_sizes,
                num_vertices, cache.result_d);
        } else if (d_assign_u8) {
            if (use_smem_inv) {
                size_t smem_inv = (size_t)nc * sizeof(double);
                ratio_cut_vertex_kernel<uint8_t><<<grid, block, smem_inv, stream>>>(
                    offsets, indices, edge_weights, d_assign_u8, cache.cluster_sizes,
                    num_vertices, nc, cache.result_d);
            } else {
                ratio_cut_vertex_kernel_bigK<uint8_t><<<grid, block, 0, stream>>>(
                    offsets, indices, edge_weights, d_assign_u8, cache.cluster_sizes,
                    num_vertices, cache.result_d);
            }
        } else if (d_assign_i16) {
            if (use_smem_inv) {
                size_t smem_inv = (size_t)nc * sizeof(double);
                ratio_cut_vertex_kernel<int16_t><<<grid, block, smem_inv, stream>>>(
                    offsets, indices, edge_weights, d_assign_i16, cache.cluster_sizes,
                    num_vertices, nc, cache.result_d);
            } else {
                ratio_cut_vertex_kernel_bigK<int16_t><<<grid, block, 0, stream>>>(
                    offsets, indices, edge_weights, d_assign_i16, cache.cluster_sizes,
                    num_vertices, cache.result_d);
            }
        } else {
            if (use_smem_inv) {
                size_t smem_inv = (size_t)nc * sizeof(double);
                ratio_cut_vertex_kernel<int32_t><<<grid, block, smem_inv, stream>>>(
                    offsets, indices, edge_weights, cluster_assignments, cache.cluster_sizes,
                    num_vertices, nc, cache.result_d);
            } else {
                ratio_cut_vertex_kernel_bigK<int32_t><<<grid, block, 0, stream>>>(
                    offsets, indices, edge_weights, cluster_assignments, cache.cluster_sizes,
                    num_vertices, cache.result_d);
            }
        }
    }

    double host_result;
    cudaMemcpy(&host_result, cache.result_d, sizeof(double), cudaMemcpyDeviceToHost);
    return host_result;
}

}  
