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
    void* d_compressed = nullptr;
    double* d_result = nullptr;
    size_t compressed_capacity = 0;

    Cache() {
        cudaMalloc(&d_result, sizeof(double));
    }

    void ensure_compressed(int32_t num_vertices) {
        size_t needed = (size_t)num_vertices * sizeof(int32_t);
        if (compressed_capacity < needed) {
            if (d_compressed) cudaFree(d_compressed);
            cudaMalloc(&d_compressed, needed);
            compressed_capacity = needed;
        }
    }

    ~Cache() override {
        if (d_compressed) cudaFree(d_compressed);
        if (d_result) cudaFree(d_result);
    }
};





__device__ __forceinline__ double warpReduceSum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ double blockReduceSum_impl(double val, double* warp_sums) {
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    val = warpReduceSum(val);
    if (lane == 0) warp_sums[wid] = val;
    __syncthreads();
    int nw = blockDim.x >> 5;
    val = (threadIdx.x < nw) ? warp_sums[threadIdx.x] : 0.0;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}




constexpr int BLOCK_VP = 256;

template <typename ClusterT>
__global__ void edge_cut_vertex_parallel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const ClusterT* __restrict__ clusters,
    int32_t num_vertices,
    double* __restrict__ result
) {
    __shared__ double ws[BLOCK_VP / 32];
    float tsum = 0.0f;

    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices;
         v += blockDim.x * gridDim.x) {
        int start = __ldg(&offsets[v]);
        int end = __ldg(&offsets[v + 1]);
        ClusterT mc = __ldg(&clusters[v]);

        for (int e = start; e < end; e++) {
            int nb = __ldg(&indices[e]);
            if (__ldg(&clusters[nb]) != mc) {
                tsum += __ldg(&edge_weights[e]);
            }
        }
    }

    double bsum = blockReduceSum_impl((double)tsum, ws);
    if (threadIdx.x == 0)
        atomicAdd(result, bsum * 0.5);
}




constexpr int BLOCK_OP = 256;
constexpr int EPB = 2048;
constexpr int MAX_SEGS = EPB + 1;

__device__ __forceinline__ int find_src(const int32_t* offsets, int nv, int target) {
    int lo = 0, hi = nv;
    while (lo < hi) {
        int mid = lo + ((hi - lo + 1) >> 1);
        if (__ldg(&offsets[mid]) <= target) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

template <typename ClusterT>
__global__ void edge_cut_output_partitioned(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const ClusterT* __restrict__ clusters,
    int32_t num_vertices,
    int32_t num_edges,
    double* __restrict__ result
) {
    __shared__ double ws[BLOCK_OP / 32];
    __shared__ int meta[2];
    extern __shared__ char smem_raw[];
    int* s_off = (int*)smem_raw;

    int bs = blockIdx.x * EPB;
    int be = bs + EPB;
    if (be > num_edges) be = num_edges;
    if (bs >= num_edges) return;

    if (threadIdx.x == 0) {
        int ss = find_src(offsets, num_vertices, bs);
        int se = find_src(offsets, num_vertices, be - 1);
        meta[0] = ss;
        meta[1] = se - ss + 1;
    }
    __syncthreads();

    int seg_start = meta[0];
    int nsegs = meta[1];

    float tsum = 0.0f;

    if (nsegs <= MAX_SEGS) {
        ClusterT* s_cl = (ClusterT*)(s_off + nsegs + 1);

        for (int i = threadIdx.x; i <= nsegs; i += BLOCK_OP)
            s_off[i] = __ldg(&offsets[seg_start + i]);
        for (int i = threadIdx.x; i < nsegs; i += BLOCK_OP)
            s_cl[i] = __ldg(&clusters[seg_start + i]);
        __syncthreads();

        int plo = 0;
        for (int e = bs + threadIdx.x; e < be; e += BLOCK_OP) {
            int lo = plo, hi = nsegs - 1;
            while (lo < hi) {
                int mid = (lo + hi + 1) >> 1;
                if (s_off[mid] <= e) lo = mid;
                else hi = mid - 1;
            }
            plo = lo;

            ClusterT sc = s_cl[lo];
            int dst = __ldg(&indices[e]);
            if (__ldg(&clusters[dst]) != sc)
                tsum += __ldg(&edge_weights[e]);
        }
    } else {
        for (int e = bs + threadIdx.x; e < be; e += BLOCK_OP) {
            int src = find_src(offsets, num_vertices, e);
            int dst = __ldg(&indices[e]);
            if (__ldg(&clusters[src]) != __ldg(&clusters[dst]))
                tsum += __ldg(&edge_weights[e]);
        }
    }

    double bsum = blockReduceSum_impl((double)tsum, ws);
    if (threadIdx.x == 0)
        atomicAdd(result, bsum * 0.5);
}





__global__ void compress_clusters_u8(const int32_t* __restrict__ src, uint8_t* __restrict__ dst, int n) {
    const int4* src4 = (const int4*)src;
    int n4 = n >> 2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
        int4 v = __ldg(&src4[i]);
        int base = i << 2;
        dst[base]     = (uint8_t)v.x;
        dst[base + 1] = (uint8_t)v.y;
        dst[base + 2] = (uint8_t)v.z;
        dst[base + 3] = (uint8_t)v.w;
    }
    for (int i = (n4 << 2) + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        dst[i] = (uint8_t)__ldg(&src[i]);
}

__global__ void compress_clusters_i16(const int32_t* __restrict__ src, int16_t* __restrict__ dst, int n) {
    const int4* src4 = (const int4*)src;
    int n4 = n >> 2;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += blockDim.x * gridDim.x) {
        int4 v = __ldg(&src4[i]);
        int16_t r0 = (int16_t)v.x;
        int16_t r1 = (int16_t)v.y;
        int16_t r2 = (int16_t)v.z;
        int16_t r3 = (int16_t)v.w;
        int base = i << 2;
        dst[base]     = r0;
        dst[base + 1] = r1;
        dst[base + 2] = r2;
        dst[base + 3] = r3;
    }
    for (int i = (n4 << 2) + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        dst[i] = (int16_t)__ldg(&src[i]);
}





void launch_compress_u8(
    const int32_t* clusters, uint8_t* cl8, int32_t num_vertices
) {
    int n4 = num_vertices >> 2;
    int cnb = (n4 + 255) / 256;
    if (cnb > 512) cnb = 512;
    if (cnb < 1) cnb = 1;
    compress_clusters_u8<<<cnb, 256>>>(clusters, cl8, num_vertices);
}

void launch_compress_i16(
    const int32_t* clusters, int16_t* cl16, int32_t num_vertices
) {
    int n4 = num_vertices >> 2;
    int cnb = (n4 + 255) / 256;
    if (cnb > 512) cnb = 512;
    if (cnb < 1) cnb = 1;
    compress_clusters_i16<<<cnb, 256>>>(clusters, cl16, num_vertices);
}

void launch_edge_cut_vp_u8(
    const int32_t* offsets, const int32_t* indices, const float* edge_weights,
    const uint8_t* clusters, int32_t num_vertices, double* result
) {
    cudaMemset(result, 0, sizeof(double));
    int nb = (num_vertices + BLOCK_VP - 1) / BLOCK_VP;
    if (nb > 2048) nb = 2048;
    edge_cut_vertex_parallel<uint8_t><<<nb, BLOCK_VP>>>(
        offsets, indices, edge_weights, clusters, num_vertices, result);
}

void launch_edge_cut_vp_i16(
    const int32_t* offsets, const int32_t* indices, const float* edge_weights,
    const int16_t* clusters, int32_t num_vertices, double* result
) {
    cudaMemset(result, 0, sizeof(double));
    int nb = (num_vertices + BLOCK_VP - 1) / BLOCK_VP;
    if (nb > 2048) nb = 2048;
    edge_cut_vertex_parallel<int16_t><<<nb, BLOCK_VP>>>(
        offsets, indices, edge_weights, clusters, num_vertices, result);
}

void launch_edge_cut_vp_i32(
    const int32_t* offsets, const int32_t* indices, const float* edge_weights,
    const int32_t* clusters, int32_t num_vertices, double* result
) {
    cudaMemset(result, 0, sizeof(double));
    int nb = (num_vertices + BLOCK_VP - 1) / BLOCK_VP;
    if (nb > 2048) nb = 2048;
    edge_cut_vertex_parallel<int32_t><<<nb, BLOCK_VP>>>(
        offsets, indices, edge_weights, clusters, num_vertices, result);
}

void launch_edge_cut_op_u8(
    const int32_t* offsets, const int32_t* indices, const float* edge_weights,
    const uint8_t* clusters, int32_t num_vertices, int32_t num_edges, double* result
) {
    cudaMemset(result, 0, sizeof(double));
    int nb = (num_edges + EPB - 1) / EPB;
    size_t smem = ((size_t)MAX_SEGS + 1) * sizeof(int) + (size_t)MAX_SEGS * sizeof(uint8_t);
    edge_cut_output_partitioned<uint8_t><<<nb, BLOCK_OP, smem>>>(
        offsets, indices, edge_weights, clusters,
        num_vertices, num_edges, result);
}

void launch_edge_cut_op_i16(
    const int32_t* offsets, const int32_t* indices, const float* edge_weights,
    const int16_t* clusters, int32_t num_vertices, int32_t num_edges, double* result
) {
    cudaMemset(result, 0, sizeof(double));
    int nb = (num_edges + EPB - 1) / EPB;
    size_t smem = ((size_t)MAX_SEGS + 1) * sizeof(int) + (size_t)MAX_SEGS * sizeof(int16_t);
    edge_cut_output_partitioned<int16_t><<<nb, BLOCK_OP, smem>>>(
        offsets, indices, edge_weights, clusters,
        num_vertices, num_edges, result);
}

void launch_edge_cut_op_i32(
    const int32_t* offsets, const int32_t* indices, const float* edge_weights,
    const int32_t* clusters, int32_t num_vertices, int32_t num_edges, double* result
) {
    cudaMemset(result, 0, sizeof(double));
    int nb = (num_edges + EPB - 1) / EPB;
    size_t smem = ((size_t)MAX_SEGS + 1) * sizeof(int) + (size_t)MAX_SEGS * sizeof(int32_t);
    edge_cut_output_partitioned<int32_t><<<nb, BLOCK_OP, smem>>>(
        offsets, indices, edge_weights, clusters,
        num_vertices, num_edges, result);
}

}  





double analyze_clustering_edge_cut(const graph32_t& graph,
                                   const float* edge_weights,
                                   std::size_t num_clusters,
                                   const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    cache.ensure_compressed(num_vertices);
    double* d_result = cache.d_result;

    int64_t avg_degree = (num_vertices > 0) ? ((int64_t)num_edges / num_vertices) : 0;

    if (num_clusters <= 255) {
        uint8_t* cl8 = (uint8_t*)cache.d_compressed;
        launch_compress_u8(cluster_assignments, cl8, num_vertices);
        if (avg_degree > 6) {
            launch_edge_cut_op_u8(offsets, indices, edge_weights, cl8,
                                  num_vertices, num_edges, d_result);
        } else {
            launch_edge_cut_vp_u8(offsets, indices, edge_weights, cl8,
                                  num_vertices, d_result);
        }
    } else if (num_clusters <= 32767) {
        int16_t* cl16 = (int16_t*)cache.d_compressed;
        launch_compress_i16(cluster_assignments, cl16, num_vertices);
        if (avg_degree > 6) {
            launch_edge_cut_op_i16(offsets, indices, edge_weights, cl16,
                                   num_vertices, num_edges, d_result);
        } else {
            launch_edge_cut_vp_i16(offsets, indices, edge_weights, cl16,
                                   num_vertices, d_result);
        }
    } else {
        if (avg_degree > 6) {
            launch_edge_cut_op_i32(offsets, indices, edge_weights, cluster_assignments,
                                   num_vertices, num_edges, d_result);
        } else {
            launch_edge_cut_vp_i32(offsets, indices, edge_weights, cluster_assignments,
                                   num_vertices, d_result);
        }
    }

    double h_result;
    cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return h_result;
}

}  
