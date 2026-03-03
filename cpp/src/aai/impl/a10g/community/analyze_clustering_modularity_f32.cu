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

#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)
#define MAX_SMEM_CLUSTERS 8192

struct Cache : Cacheable {
    float* cluster_degrees = nullptr;
    std::size_t cluster_degrees_capacity = 0;

    float* d_intra_sum = nullptr;

    void* d_compressed = nullptr;
    std::size_t compressed_capacity = 0;

    double* d_result = nullptr;

    Cache() {
        cudaMalloc(&d_intra_sum, sizeof(float));
        cudaMalloc(&d_result, sizeof(double));
        cluster_degrees_capacity = 65536;
        cudaMalloc(&cluster_degrees, cluster_degrees_capacity * sizeof(float));
    }

    ~Cache() override {
        if (cluster_degrees) cudaFree(cluster_degrees);
        if (d_intra_sum) cudaFree(d_intra_sum);
        if (d_compressed) cudaFree(d_compressed);
        if (d_result) cudaFree(d_result);
    }

    void ensure(int32_t num_clusters, int32_t num_vertices) {
        if ((std::size_t)num_clusters > cluster_degrees_capacity) {
            cudaFree(cluster_degrees);
            cluster_degrees_capacity = (std::size_t)num_clusters * 2;
            cudaMalloc(&cluster_degrees, cluster_degrees_capacity * sizeof(float));
        }
        std::size_t needed = (std::size_t)num_vertices * sizeof(int32_t);
        if (needed > compressed_capacity) {
            if (d_compressed) cudaFree(d_compressed);
            compressed_capacity = needed;
            cudaMalloc(&d_compressed, compressed_capacity);
        }
    }
};





__global__ void compress_to_u8(const int32_t* __restrict__ src, uint8_t* __restrict__ dst, int n, int32_t nc) {
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        int32_t c = src[i];
        if ((uint32_t)c >= (uint32_t)nc) c = (c < 0) ? 0 : (nc - 1);
        dst[i] = (uint8_t)c;
    }
}

__global__ void compress_to_i16(const int32_t* __restrict__ src, int16_t* __restrict__ dst, int n, int32_t nc) {
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        int32_t c = src[i];
        if ((uint32_t)c >= (uint32_t)nc) c = (c < 0) ? 0 : (nc - 1);
        dst[i] = (int16_t)c;
    }
}




template<typename CType>
__global__ void modularity_kernel_warp_smem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const CType* __restrict__ cluster_assignments,
    float* __restrict__ cluster_degrees,
    float* __restrict__ d_intra_sum,
    int32_t num_clusters,
    int32_t num_vertices
) {
    extern __shared__ float s_cluster_deg[];

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE)
        s_cluster_deg[i] = 0.0f;
    __syncthreads();

    const int warp_id_global = (blockIdx.x * BLOCK_SIZE + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    float warp_intra = 0.0f;

    if (warp_id_global < num_vertices) {
        const int v = warp_id_global;
        int32_t c_v_tmp = 0, start_tmp = 0, end_tmp = 0;
        if (lane == 0) {
            c_v_tmp = (int32_t)__ldg(&cluster_assignments[v]);
            start_tmp = __ldg(&offsets[v]);
            end_tmp = __ldg(&offsets[v + 1]);
        }
        const int32_t c_v = __shfl_sync(0xffffffff, c_v_tmp, 0);
        const int start = __shfl_sync(0xffffffff, start_tmp, 0);
        const int end = __shfl_sync(0xffffffff, end_tmp, 0);

        float deg = 0.0f;
        float intra = 0.0f;

        for (int e = start + lane; e < end; e += 32) {
            float w = edge_weights[e];
            deg += w;
            int32_t neighbor = __ldg(&indices[e]);
            if ((int32_t)__ldg(&cluster_assignments[neighbor]) == c_v)
                intra += w;
        }

        for (int offset = 16; offset >= 1; offset >>= 1) {
            deg += __shfl_down_sync(0xffffffff, deg, offset);
            intra += __shfl_down_sync(0xffffffff, intra, offset);
        }

        if (lane == 0) {
            atomicAdd(&s_cluster_deg[c_v], deg);
            warp_intra = intra;
        }
    }

    __shared__ float s_warp_intra[WARPS_PER_BLOCK];
    const int warp_id_local = threadIdx.x >> 5;

    if (lane == 0) s_warp_intra[warp_id_local] = warp_intra;
    __syncthreads();

    if (warp_id_local == 0) {
        float val = (lane < WARPS_PER_BLOCK) ? s_warp_intra[lane] : 0.0f;
        for (int offset = (WARPS_PER_BLOCK + 1) / 2; offset >= 1; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) atomicAdd(d_intra_sum, val);
    }

    __syncthreads();

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        float val = s_cluster_deg[i];
        if (val != 0.0f) atomicAdd(&cluster_degrees[i], val);
    }
}




template<typename CType>
__global__ void modularity_kernel_warp_gmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const CType* __restrict__ cluster_assignments,
    float* __restrict__ cluster_degrees,
    float* __restrict__ d_intra_sum,
    int32_t num_vertices
) {
    const int warp_id_global = (blockIdx.x * BLOCK_SIZE + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    float warp_intra = 0.0f;

    if (warp_id_global < num_vertices) {
        const int v = warp_id_global;
        int32_t c_v_tmp = 0, start_tmp = 0, end_tmp = 0;
        if (lane == 0) {
            c_v_tmp = (int32_t)__ldg(&cluster_assignments[v]);
            start_tmp = __ldg(&offsets[v]);
            end_tmp = __ldg(&offsets[v + 1]);
        }
        const int32_t c_v = __shfl_sync(0xffffffff, c_v_tmp, 0);
        const int start = __shfl_sync(0xffffffff, start_tmp, 0);
        const int end = __shfl_sync(0xffffffff, end_tmp, 0);

        float deg = 0.0f;
        float intra = 0.0f;

        for (int e = start + lane; e < end; e += 32) {
            float w = edge_weights[e];
            deg += w;
            int32_t neighbor = __ldg(&indices[e]);
            if ((int32_t)__ldg(&cluster_assignments[neighbor]) == c_v)
                intra += w;
        }

        for (int offset = 16; offset >= 1; offset >>= 1) {
            deg += __shfl_down_sync(0xffffffff, deg, offset);
            intra += __shfl_down_sync(0xffffffff, intra, offset);
        }

        if (lane == 0) {
            atomicAdd(&cluster_degrees[c_v], deg);
            warp_intra = intra;
        }
    }

    __shared__ float s_warp_intra[WARPS_PER_BLOCK];
    const int warp_id_local = threadIdx.x >> 5;

    if (lane == 0) s_warp_intra[warp_id_local] = warp_intra;
    __syncthreads();

    if (warp_id_local == 0) {
        float val = (lane < WARPS_PER_BLOCK) ? s_warp_intra[lane] : 0.0f;
        for (int offset = (WARPS_PER_BLOCK + 1) / 2; offset >= 1; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (lane == 0) atomicAdd(d_intra_sum, val);
    }
}




template<typename CType>
__global__ void modularity_kernel_thread_smem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const CType* __restrict__ cluster_assignments,
    float* __restrict__ cluster_degrees,
    float* __restrict__ d_intra_sum,
    int32_t num_clusters,
    int32_t num_vertices
) {
    extern __shared__ float s_cluster_deg[];

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE)
        s_cluster_deg[i] = 0.0f;
    __syncthreads();

    const int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float local_intra = 0.0f;

    if (v < num_vertices) {
        const int32_t c_v = (int32_t)__ldg(&cluster_assignments[v]);
        const int start = __ldg(&offsets[v]);
        const int end = __ldg(&offsets[v + 1]);

        float deg = 0.0f;
        float intra = 0.0f;

        for (int e = start; e < end; e++) {
            float w = edge_weights[e];
            deg += w;
            int32_t neighbor = __ldg(&indices[e]);
            if ((int32_t)__ldg(&cluster_assignments[neighbor]) == c_v)
                intra += w;
        }

        atomicAdd(&s_cluster_deg[c_v], deg);
        local_intra = intra;
    }

    for (int offset = 16; offset >= 1; offset >>= 1)
        local_intra += __shfl_down_sync(0xffffffff, local_intra, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(d_intra_sum, local_intra);

    __syncthreads();

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        float val = s_cluster_deg[i];
        if (val != 0.0f) atomicAdd(&cluster_degrees[i], val);
    }
}




template<typename CType>
__global__ void modularity_kernel_thread_gmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const CType* __restrict__ cluster_assignments,
    float* __restrict__ cluster_degrees,
    float* __restrict__ d_intra_sum,
    int32_t num_vertices
) {
    const int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float local_intra = 0.0f;

    if (v < num_vertices) {
        const int32_t c_v = (int32_t)__ldg(&cluster_assignments[v]);
        const int start = __ldg(&offsets[v]);
        const int end = __ldg(&offsets[v + 1]);

        float deg = 0.0f;
        float intra = 0.0f;

        for (int e = start; e < end; e++) {
            float w = edge_weights[e];
            deg += w;
            int32_t neighbor = __ldg(&indices[e]);
            if ((int32_t)__ldg(&cluster_assignments[neighbor]) == c_v)
                intra += w;
        }

        atomicAdd(&cluster_degrees[c_v], deg);
        local_intra = intra;
    }

    for (int offset = 16; offset >= 1; offset >>= 1)
        local_intra += __shfl_down_sync(0xffffffff, local_intra, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(d_intra_sum, local_intra);
}




__global__ void finalize_kernel_parallel(
    const float* __restrict__ cluster_degrees,
    const float* __restrict__ d_intra_sum,
    double* __restrict__ d_result,
    int32_t num_clusters
) {
    __shared__ double s_total[WARPS_PER_BLOCK];
    __shared__ double s_sumsq[WARPS_PER_BLOCK];

    double local_total = 0.0;
    double local_sumsq = 0.0;

    for (int c = threadIdx.x; c < num_clusters; c += BLOCK_SIZE) {
        double d = (double)cluster_degrees[c];
        local_total += d;
        local_sumsq += d * d;
    }

    for (int offset = 16; offset >= 1; offset >>= 1) {
        local_total += __shfl_down_sync(0xffffffff, local_total, offset);
        local_sumsq += __shfl_down_sync(0xffffffff, local_sumsq, offset);
    }

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        s_total[warp_id] = local_total;
        s_sumsq[warp_id] = local_sumsq;
    }
    __syncthreads();

    if (warp_id == 0) {
        double val_total = (lane < WARPS_PER_BLOCK) ? s_total[lane] : 0.0;
        double val_sumsq = (lane < WARPS_PER_BLOCK) ? s_sumsq[lane] : 0.0;
        for (int offset = (WARPS_PER_BLOCK + 1) / 2; offset >= 1; offset >>= 1) {
            val_total += __shfl_down_sync(0xffffffff, val_total, offset);
            val_sumsq += __shfl_down_sync(0xffffffff, val_sumsq, offset);
        }
        if (lane == 0) {
            if (val_total > 0.0) {
                double intra = (double)(*d_intra_sum);
                *d_result = intra / val_total - val_sumsq / (val_total * val_total);
            } else {
                *d_result = 0.0;
            }
        }
    }
}




template<typename CType>
static void launch_main_kernels(
    const int32_t* offsets, const int32_t* indices, const float* edge_weights,
    const CType* cluster_assignments,
    float* cluster_degrees, float* d_intra_sum,
    int32_t num_vertices, int32_t num_edges, int32_t num_clusters,
    cudaStream_t stream
) {
    float avg_degree = (num_vertices > 0) ? (float)num_edges / (float)num_vertices : 0.0f;
    bool use_smem = (num_clusters <= MAX_SMEM_CLUSTERS);

    if (avg_degree >= 6.0f) {
        int warps_needed = num_vertices;
        int grid = (warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        if (use_smem) {
            int smem_size = num_clusters * sizeof(float);
            modularity_kernel_warp_smem<CType><<<grid, BLOCK_SIZE, smem_size, stream>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cluster_degrees, d_intra_sum, num_clusters, num_vertices);
        } else {
            modularity_kernel_warp_gmem<CType><<<grid, BLOCK_SIZE, 0, stream>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cluster_degrees, d_intra_sum, num_vertices);
        }
    } else {
        int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (use_smem) {
            int smem_size = num_clusters * sizeof(float);
            modularity_kernel_thread_smem<CType><<<grid, BLOCK_SIZE, smem_size, stream>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cluster_degrees, d_intra_sum, num_clusters, num_vertices);
        } else {
            modularity_kernel_thread_gmem<CType><<<grid, BLOCK_SIZE, 0, stream>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cluster_degrees, d_intra_sum, num_vertices);
        }
    }
}

}  

double analyze_clustering_modularity(const graph32_t& graph,
                                     const float* edge_weights,
                                     std::size_t num_clusters,
                                     const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t num_vertices = graph.number_of_vertices;
    const int32_t num_edges = graph.number_of_edges;
    const int32_t nc = static_cast<int32_t>(num_clusters);

    cache.ensure(nc, num_vertices);

    cudaStream_t stream = 0;

    cudaMemsetAsync(cache.cluster_degrees, 0, (std::size_t)nc * sizeof(float), stream);
    cudaMemsetAsync(cache.d_intra_sum, 0, sizeof(float), stream);

    if (nc <= 255 && cache.d_compressed) {
        int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid > 1024) grid = 1024;
        compress_to_u8<<<grid, BLOCK_SIZE, 0, stream>>>(
            cluster_assignments, (uint8_t*)cache.d_compressed, num_vertices, nc);
        launch_main_kernels<uint8_t>(graph.offsets, graph.indices, edge_weights,
            (const uint8_t*)cache.d_compressed,
            cache.cluster_degrees, cache.d_intra_sum, num_vertices, num_edges, nc, stream);
    } else if (nc <= 32767 && cache.d_compressed) {
        int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid > 1024) grid = 1024;
        compress_to_i16<<<grid, BLOCK_SIZE, 0, stream>>>(
            cluster_assignments, (int16_t*)cache.d_compressed, num_vertices, nc);
        launch_main_kernels<int16_t>(graph.offsets, graph.indices, edge_weights,
            (const int16_t*)cache.d_compressed,
            cache.cluster_degrees, cache.d_intra_sum, num_vertices, num_edges, nc, stream);
    } else {
        launch_main_kernels<int32_t>(graph.offsets, graph.indices, edge_weights,
            cluster_assignments,
            cache.cluster_degrees, cache.d_intra_sum, num_vertices, num_edges, nc, stream);
    }

    finalize_kernel_parallel<<<1, BLOCK_SIZE, 0, stream>>>(
        cache.cluster_degrees, cache.d_intra_sum, cache.d_result, nc);

    double result;
    cudaMemcpy(&result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

}  
