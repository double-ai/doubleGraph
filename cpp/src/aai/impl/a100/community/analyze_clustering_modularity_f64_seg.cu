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
    double* d_cluster_sums = nullptr;
    double* d_intra = nullptr;
    double* d_result = nullptr;
    size_t cluster_sums_cap = 0;

    Cache() {
        cluster_sums_cap = 8192;
        cudaMalloc(&d_cluster_sums, cluster_sums_cap * sizeof(double));
        cudaMalloc(&d_intra, sizeof(double));
        cudaMalloc(&d_result, sizeof(double));
    }

    ~Cache() override {
        if (d_cluster_sums) cudaFree(d_cluster_sums);
        if (d_intra) cudaFree(d_intra);
        if (d_result) cudaFree(d_result);
    }

    void ensure_cluster_buf(size_t needed) {
        if (needed > cluster_sums_cap) {
            if (d_cluster_sums) cudaFree(d_cluster_sums);
            cluster_sums_cap = needed * 2;
            cudaMalloc(&d_cluster_sums, cluster_sums_cap * sizeof(double));
        }
    }
};


__device__ __forceinline__ double warp_sum(double val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}



__global__ void mod_warp_per_vertex(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ clusters,
    double* __restrict__ cluster_sums,
    double* __restrict__ g_intra,
    int32_t v_start, int32_t v_end,
    int32_t num_clusters
) {
    extern __shared__ double shmem[];

    const int BLOCK_SIZE = blockDim.x;
    const int NUM_WARPS = BLOCK_SIZE >> 5;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_in_block = tid >> 5;

    
    for (int i = tid; i <= num_clusters; i += BLOCK_SIZE) {
        shmem[i] = 0.0;
    }
    __syncthreads();

    int warp_global = (blockIdx.x * NUM_WARPS) + warp_in_block;
    int32_t v = v_start + warp_global;

    if (v < v_end) {
        int c_v = clusters[v];
        int start = offsets[v];
        int end = offsets[v + 1];

        double deg = 0.0, intra = 0.0;
        for (int e = start + lane; e < end; e += 32) {
            double w = weights[e];
            deg += w;
            if (clusters[indices[e]] == c_v) intra += w;
        }

        
        deg = warp_sum(deg);
        intra = warp_sum(intra);

        if (lane == 0) {
            atomicAdd(&shmem[c_v], deg);
            atomicAdd(&shmem[num_clusters], intra);
        }
    }
    __syncthreads();

    
    for (int c = tid; c < num_clusters; c += BLOCK_SIZE) {
        double val = shmem[c];
        if (val != 0.0) atomicAdd(&cluster_sums[c], val);
    }
    if (tid == 0) {
        double val = shmem[num_clusters];
        if (val != 0.0) atomicAdd(g_intra, val);
    }
}


__global__ void mod_high_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ clusters,
    double* __restrict__ cluster_sums,
    double* __restrict__ g_intra,
    int32_t v_start, int32_t v_end
) {
    __shared__ double smem[32]; 

    int32_t v = v_start + blockIdx.x;
    if (v >= v_end) return;

    int c_v = clusters[v];
    int start = offsets[v];
    int end = offsets[v + 1];

    double deg = 0.0, intra = 0.0;
    for (int e = start + threadIdx.x; e < end; e += blockDim.x) {
        double w = weights[e];
        deg += w;
        if (clusters[indices[e]] == c_v) intra += w;
    }

    
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    double v1 = warp_sum(deg);
    if (lane == 0) smem[wid] = v1;
    __syncthreads();
    v1 = (threadIdx.x < num_warps) ? smem[threadIdx.x] : 0.0;
    if (wid == 0) v1 = warp_sum(v1);
    __syncthreads();

    
    double v2 = warp_sum(intra);
    if (lane == 0) smem[wid] = v2;
    __syncthreads();
    v2 = (threadIdx.x < num_warps) ? smem[threadIdx.x] : 0.0;
    if (wid == 0) v2 = warp_sum(v2);

    if (threadIdx.x == 0) {
        atomicAdd(&cluster_sums[c_v], v1);
        atomicAdd(g_intra, v2);
    }
}



__global__ void mod_thread_per_vertex(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ clusters,
    double* __restrict__ cluster_sums,
    double* __restrict__ g_intra,
    int32_t v_start, int32_t v_end,
    int32_t num_clusters
) {
    extern __shared__ double shmem[];

    const int BLOCK_SIZE = blockDim.x;
    int tid = threadIdx.x;

    
    for (int i = tid; i <= num_clusters; i += BLOCK_SIZE) {
        shmem[i] = 0.0;
    }
    __syncthreads();

    int32_t v = v_start + blockIdx.x * BLOCK_SIZE + tid;

    if (v < v_end) {
        int c_v = clusters[v];
        int start = offsets[v];
        int end = offsets[v + 1];

        double deg = 0.0, intra = 0.0;
        for (int e = start; e < end; e++) {
            double w = weights[e];
            deg += w;
            if (clusters[indices[e]] == c_v) intra += w;
        }

        atomicAdd(&shmem[c_v], deg);
        atomicAdd(&shmem[num_clusters], intra);
    }
    __syncthreads();

    
    for (int c = tid; c < num_clusters; c += BLOCK_SIZE) {
        double val = shmem[c];
        if (val != 0.0) atomicAdd(&cluster_sums[c], val);
    }
    if (tid == 0) {
        double val = shmem[num_clusters];
        if (val != 0.0) atomicAdd(g_intra, val);
    }
}


__global__ void mod_thread_per_vertex_noshmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ clusters,
    double* __restrict__ cluster_sums,
    double* __restrict__ g_intra,
    int32_t v_start, int32_t v_end
) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    int32_t v = v_start + blockIdx.x * blockDim.x + tid;
    double my_intra = 0.0, my_deg = 0.0;
    int cv = -1;

    if (v < v_end) {
        cv = clusters[v];
        int start = offsets[v];
        int end = offsets[v + 1];

        for (int e = start; e < end; e++) {
            double w = weights[e];
            my_deg += w;
            if (clusters[indices[e]] == cv) my_intra += w;
        }
    }

    
    double warp_intra = warp_sum(my_intra);
    __shared__ double s_intra[8];
    if (lane == 0) s_intra[wid] = warp_intra;
    __syncthreads();
    if (tid == 0) {
        double bi = 0.0;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) bi += s_intra[i];
        if (bi != 0.0) atomicAdd(g_intra, bi);
    }
    if (cv >= 0 && my_deg != 0.0) atomicAdd(&cluster_sums[cv], my_deg);
}


__global__ void mod_warp_per_vertex_noshmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ clusters,
    double* __restrict__ cluster_sums,
    double* __restrict__ g_intra,
    int32_t v_start, int32_t v_end
) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    int NUM_WARPS = blockDim.x >> 5;

    int warp_global = (blockIdx.x * NUM_WARPS) + wid;
    int32_t v = v_start + warp_global;

    double my_deg = 0.0, my_intra = 0.0;
    int cv = -1;

    if (v < v_end) {
        cv = clusters[v];
        int start = offsets[v];
        int end = offsets[v + 1];

        for (int e = start + lane; e < end; e += 32) {
            double w = weights[e];
            my_deg += w;
            if (clusters[indices[e]] == cv) my_intra += w;
        }

        my_deg = warp_sum(my_deg);
        my_intra = warp_sum(my_intra);

        if (lane == 0) {
            if (my_deg != 0.0) atomicAdd(&cluster_sums[cv], my_deg);
            if (my_intra != 0.0) atomicAdd(g_intra, my_intra);
        }
    }
}


__global__ void mod_final(
    const double* __restrict__ cluster_sums,
    const double* __restrict__ g_intra,
    double* __restrict__ result,
    int32_t num_clusters
) {
    double intra = *g_intra;
    double two_m = 0.0;
    double sq_sum = 0.0;

    for (int c = 0; c < num_clusters; c++) {
        double s = cluster_sums[c];
        two_m += s;
        sq_sum += s * s;
    }

    if (two_m > 0.0) {
        *result = (intra / two_m) - (sq_sum / (two_m * two_m));
    } else {
        *result = 0.0;
    }
}



void launch_mod_high_degree(
    const int32_t* offsets, const int32_t* indices,
    const double* weights, const int32_t* clusters,
    double* cluster_sums, double* g_intra,
    int32_t v_start, int32_t v_end, cudaStream_t stream
) {
    int n = v_end - v_start;
    if (n <= 0) return;
    mod_high_degree<<<n, 256, 0, stream>>>(
        offsets, indices, weights, clusters, cluster_sums, g_intra, v_start, v_end
    );
}

void launch_mod_warp_per_vertex(
    const int32_t* offsets, const int32_t* indices,
    const double* weights, const int32_t* clusters,
    double* cluster_sums, double* g_intra,
    int32_t v_start, int32_t v_end, int32_t num_clusters,
    cudaStream_t stream
) {
    int n = v_end - v_start;
    if (n <= 0) return;
    int threads_per_block = 256; 
    int warps_per_block = threads_per_block / 32;
    int blocks = (n + warps_per_block - 1) / warps_per_block;

    
    if (num_clusters <= 4096) {
        int smem_bytes = (num_clusters + 1) * sizeof(double);
        mod_warp_per_vertex<<<blocks, threads_per_block, smem_bytes, stream>>>(
            offsets, indices, weights, clusters, cluster_sums, g_intra,
            v_start, v_end, num_clusters
        );
    } else {
        mod_warp_per_vertex_noshmem<<<blocks, threads_per_block, 0, stream>>>(
            offsets, indices, weights, clusters, cluster_sums, g_intra,
            v_start, v_end
        );
    }
}

void launch_mod_thread_per_vertex(
    const int32_t* offsets, const int32_t* indices,
    const double* weights, const int32_t* clusters,
    double* cluster_sums, double* g_intra,
    int32_t v_start, int32_t v_end, int32_t num_clusters,
    cudaStream_t stream
) {
    int n = v_end - v_start;
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    
    if (num_clusters <= 4096) {
        int smem_bytes = (num_clusters + 1) * sizeof(double);
        mod_thread_per_vertex<<<blocks, threads, smem_bytes, stream>>>(
            offsets, indices, weights, clusters, cluster_sums, g_intra,
            v_start, v_end, num_clusters
        );
    } else {
        mod_thread_per_vertex_noshmem<<<blocks, threads, 0, stream>>>(
            offsets, indices, weights, clusters, cluster_sums, g_intra,
            v_start, v_end
        );
    }
}

void launch_mod_final(
    const double* cluster_sums, const double* g_intra,
    double* result, int32_t num_clusters, cudaStream_t stream
) {
    mod_final<<<1, 1, 0, stream>>>(cluster_sums, g_intra, result, num_clusters);
}

}  

double analyze_clustering_modularity_seg(const graph32_t& graph,
                                         const double* edge_weights,
                                         std::size_t num_clusters,
                                         const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nc = static_cast<int32_t>(num_clusters);

    
    cache.ensure_cluster_buf(num_clusters);

    cudaStream_t stream = 0;

    
    cudaMemsetAsync(cache.d_cluster_sums, 0, num_clusters * sizeof(double), stream);
    cudaMemsetAsync(cache.d_intra, 0, sizeof(double), stream);

    
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    
    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0];
    int32_t seg1 = seg[1];
    int32_t seg2 = seg[2];
    int32_t seg3 = seg[3];

    
    
    launch_mod_high_degree(d_offsets, d_indices, edge_weights, cluster_assignments,
                           cache.d_cluster_sums, cache.d_intra, seg0, seg1, stream);

    
    launch_mod_warp_per_vertex(d_offsets, d_indices, edge_weights, cluster_assignments,
                                cache.d_cluster_sums, cache.d_intra, seg1, seg2, nc, stream);

    
    launch_mod_thread_per_vertex(d_offsets, d_indices, edge_weights, cluster_assignments,
                                  cache.d_cluster_sums, cache.d_intra, seg2, seg3, nc, stream);

    
    launch_mod_final(cache.d_cluster_sums, cache.d_intra, cache.d_result, nc, stream);

    
    double result;
    cudaMemcpy(&result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);

    return result;
}

}  
