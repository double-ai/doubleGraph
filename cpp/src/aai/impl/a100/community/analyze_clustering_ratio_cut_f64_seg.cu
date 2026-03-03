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
#include <cub/cub.cuh>

namespace aai {

namespace {

struct Cache : Cacheable {
    double* cluster_cut = nullptr;
    int32_t* cluster_sizes = nullptr;
    double* d_result = nullptr;
    size_t cluster_cut_capacity = 0;
    size_t cluster_sizes_capacity = 0;

    Cache() {
        cudaMalloc(&d_result, sizeof(double));
        cluster_cut_capacity = 1024;
        cluster_sizes_capacity = 1024;
        cudaMalloc(&cluster_cut, cluster_cut_capacity * sizeof(double));
        cudaMalloc(&cluster_sizes, cluster_sizes_capacity * sizeof(int32_t));
    }

    void ensure(size_t num_clusters) {
        if (cluster_cut_capacity < num_clusters) {
            if (cluster_cut) cudaFree(cluster_cut);
            cluster_cut_capacity = num_clusters * 2;
            cudaMalloc(&cluster_cut, cluster_cut_capacity * sizeof(double));
        }
        if (cluster_sizes_capacity < num_clusters) {
            if (cluster_sizes) cudaFree(cluster_sizes);
            cluster_sizes_capacity = num_clusters * 2;
            cudaMalloc(&cluster_sizes, cluster_sizes_capacity * sizeof(int32_t));
        }
    }

    ~Cache() override {
        if (cluster_cut) cudaFree(cluster_cut);
        if (cluster_sizes) cudaFree(cluster_sizes);
        if (d_result) cudaFree(d_result);
    }
};



template <int BLOCK_SIZE>
__global__ void compute_cut_high_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ cluster_cut,
    int32_t* __restrict__ cluster_sizes,
    int32_t seg_start,
    int32_t seg_end
) {
    for (int v = seg_start + blockIdx.x; v < seg_end; v += gridDim.x) {
        int32_t cluster_v = cluster_assignments[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        if (threadIdx.x == 0) {
            atomicAdd(&cluster_sizes[cluster_v], 1);
        }

        double local_cut = 0.0;
        for (int e = start + threadIdx.x; e < end; e += BLOCK_SIZE) {
            int32_t neighbor = indices[e];
            if (cluster_v != cluster_assignments[neighbor]) {
                local_cut += edge_weights[e];
            }
        }

        
        typedef cub::BlockReduce<double, BLOCK_SIZE> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp;
        double block_sum = BlockReduce(temp).Sum(local_cut);

        if (threadIdx.x == 0 && block_sum != 0.0) {
            atomicAdd(&cluster_cut[cluster_v], block_sum);
        }
    }
}




template <int BLOCK_SIZE>
__global__ void compute_cut_mid_low(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ cluster_cut,
    int32_t* __restrict__ cluster_sizes,
    int32_t seg_start,
    int32_t seg_end,
    int32_t num_clusters
) {
    constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;

    
    extern __shared__ char smem[];
    double* s_cut = (double*)smem;
    int32_t* s_sizes = (int32_t*)(s_cut + num_clusters);

    
    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        s_cut[i] = 0.0;
        s_sizes[i] = 0;
    }
    __syncthreads();

    int warp_id_global = (blockIdx.x * BLOCK_SIZE + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int total_warps = (gridDim.x * BLOCK_SIZE) / 32;

    for (int v = seg_start + warp_id_global; v < seg_end; v += total_warps) {
        int32_t cluster_v = cluster_assignments[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        if (lane == 0) {
            atomicAdd(&s_sizes[cluster_v], 1);
        }

        double local_cut = 0.0;
        for (int e = start + lane; e < end; e += 32) {
            int32_t neighbor = indices[e];
            if (cluster_v != cluster_assignments[neighbor]) {
                local_cut += edge_weights[e];
            }
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_cut += __shfl_down_sync(0xffffffff, local_cut, offset);
        }

        if (lane == 0 && local_cut != 0.0) {
            atomicAdd(&s_cut[cluster_v], local_cut);
        }
    }

    __syncthreads();

    
    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        if (s_cut[i] != 0.0) atomicAdd(&cluster_cut[i], s_cut[i]);
        if (s_sizes[i] != 0) atomicAdd(&cluster_sizes[i], s_sizes[i]);
    }
}



template <int BLOCK_SIZE>
__global__ void compute_cut_low_degree(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ cluster_cut,
    int32_t* __restrict__ cluster_sizes,
    int32_t seg_start,
    int32_t seg_end,
    int32_t num_clusters
) {
    extern __shared__ char smem[];
    double* s_cut = (double*)smem;
    int32_t* s_sizes = (int32_t*)(s_cut + num_clusters);

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        s_cut[i] = 0.0;
        s_sizes[i] = 0;
    }
    __syncthreads();

    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int stride = gridDim.x * BLOCK_SIZE;

    for (int v = seg_start + tid; v < seg_end; v += stride) {
        int32_t cluster_v = cluster_assignments[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        atomicAdd(&s_sizes[cluster_v], 1);

        double local_cut = 0.0;
        #pragma unroll 4
        for (int e = start; e < end; e++) {
            int32_t neighbor = indices[e];
            if (cluster_v != cluster_assignments[neighbor]) {
                local_cut += edge_weights[e];
            }
        }

        if (local_cut != 0.0) {
            atomicAdd(&s_cut[cluster_v], local_cut);
        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        if (s_cut[i] != 0.0) atomicAdd(&cluster_cut[i], s_cut[i]);
        if (s_sizes[i] != 0) atomicAdd(&cluster_sizes[i], s_sizes[i]);
    }
}


template <int BLOCK_SIZE>
__global__ void count_isolated_kernel(
    const int32_t* __restrict__ cluster_assignments,
    int32_t* __restrict__ cluster_sizes,
    int32_t seg_start,
    int32_t seg_end,
    int32_t num_clusters
) {
    extern __shared__ int32_t s_sizes[];

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        s_sizes[i] = 0;
    }
    __syncthreads();

    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int stride = gridDim.x * BLOCK_SIZE;

    for (int v = seg_start + tid; v < seg_end; v += stride) {
        atomicAdd(&s_sizes[cluster_assignments[v]], 1);
    }

    __syncthreads();

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        if (s_sizes[i] != 0) atomicAdd(&cluster_sizes[i], s_sizes[i]);
    }
}


template <int BLOCK_SIZE>
__global__ void compute_cut_mid_low_noshmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ cluster_cut,
    int32_t* __restrict__ cluster_sizes,
    int32_t seg_start,
    int32_t seg_end
) {
    int warp_id_global = (blockIdx.x * BLOCK_SIZE + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int total_warps = (gridDim.x * BLOCK_SIZE) / 32;

    for (int v = seg_start + warp_id_global; v < seg_end; v += total_warps) {
        int32_t cluster_v = cluster_assignments[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        if (lane == 0) {
            atomicAdd(&cluster_sizes[cluster_v], 1);
        }

        double local_cut = 0.0;
        for (int e = start + lane; e < end; e += 32) {
            int32_t neighbor = indices[e];
            if (cluster_v != cluster_assignments[neighbor]) {
                local_cut += edge_weights[e];
            }
        }

        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_cut += __shfl_down_sync(0xffffffff, local_cut, offset);
        }

        if (lane == 0 && local_cut != 0.0) {
            atomicAdd(&cluster_cut[cluster_v], local_cut);
        }
    }
}


template <int BLOCK_SIZE>
__global__ void compute_cut_low_degree_noshmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ cluster_cut,
    int32_t* __restrict__ cluster_sizes,
    int32_t seg_start,
    int32_t seg_end
) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int stride = gridDim.x * BLOCK_SIZE;

    for (int v = seg_start + tid; v < seg_end; v += stride) {
        int32_t cluster_v = cluster_assignments[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        atomicAdd(&cluster_sizes[cluster_v], 1);

        double local_cut = 0.0;
        #pragma unroll 4
        for (int e = start; e < end; e++) {
            int32_t neighbor = indices[e];
            if (cluster_v != cluster_assignments[neighbor]) {
                local_cut += edge_weights[e];
            }
        }

        if (local_cut != 0.0) {
            atomicAdd(&cluster_cut[cluster_v], local_cut);
        }
    }
}


template <int BLOCK_SIZE>
__global__ void count_isolated_kernel_noshmem(
    const int32_t* __restrict__ cluster_assignments,
    int32_t* __restrict__ cluster_sizes,
    int32_t seg_start,
    int32_t seg_end
) {
    int tid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int stride = gridDim.x * BLOCK_SIZE;

    for (int v = seg_start + tid; v < seg_end; v += stride) {
        atomicAdd(&cluster_sizes[cluster_assignments[v]], 1);
    }
}


__global__ void compute_final_score(
    const double* __restrict__ cluster_cut,
    const int32_t* __restrict__ cluster_sizes,
    double* __restrict__ result,
    int32_t num_clusters
) {
    double score = 0.0;
    for (int c = threadIdx.x; c < num_clusters; c += 32) {
        int32_t sz = cluster_sizes[c];
        if (sz > 0) {
            score += cluster_cut[c] / (double)sz;
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        score += __shfl_down_sync(0xffffffff, score, offset);
    }

    if (threadIdx.x == 0) {
        *result = score;
    }
}

}  

double analyze_clustering_ratio_cut_seg(const graph32_t& graph,
                                        const double* edge_weights,
                                        std::size_t num_clusters,
                                        const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nc = static_cast<int32_t>(num_clusters);
    cache.ensure(num_clusters);

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

    
    cudaMemsetAsync(cache.cluster_cut, 0, nc * sizeof(double));
    cudaMemsetAsync(cache.cluster_sizes, 0, nc * sizeof(int32_t));

    constexpr int BLOCK = 256;
    size_t smem_bytes = nc * sizeof(double) + nc * sizeof(int32_t);

    
    constexpr int32_t SMEM_CLUSTER_THRESHOLD = 4096;
    bool use_smem = (nc <= SMEM_CLUSTER_THRESHOLD);

    
    int high_count = seg1 - seg0;
    if (high_count > 0) {
        compute_cut_high_degree<BLOCK><<<high_count, BLOCK>>>(
            offsets, indices, edge_weights, cluster_assignments,
            cache.cluster_cut, cache.cluster_sizes, seg0, seg1
        );
    }

    
    int mid_count = seg2 - seg1;
    if (mid_count > 0) {
        int warps_needed = mid_count;
        int warps_per_block = BLOCK / 32;
        int blocks = (warps_needed + warps_per_block - 1) / warps_per_block;
        if (blocks > 65535) blocks = 65535;
        if (use_smem) {
            compute_cut_mid_low<BLOCK><<<blocks, BLOCK, smem_bytes>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cache.cluster_cut, cache.cluster_sizes, seg1, seg2, nc
            );
        } else {
            compute_cut_mid_low_noshmem<BLOCK><<<blocks, BLOCK>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cache.cluster_cut, cache.cluster_sizes, seg1, seg2
            );
        }
    }

    
    int low_count = seg3 - seg2;
    if (low_count > 0) {
        int blocks = (low_count + BLOCK - 1) / BLOCK;
        if (blocks > 65535) blocks = 65535;
        if (use_smem) {
            compute_cut_low_degree<BLOCK><<<blocks, BLOCK, smem_bytes>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cache.cluster_cut, cache.cluster_sizes, seg2, seg3, nc
            );
        } else {
            compute_cut_low_degree_noshmem<BLOCK><<<blocks, BLOCK>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cache.cluster_cut, cache.cluster_sizes, seg2, seg3
            );
        }
    }

    
    int iso_count = seg4 - seg3;
    if (iso_count > 0) {
        int blocks = (iso_count + BLOCK - 1) / BLOCK;
        if (blocks > 65535) blocks = 65535;
        if (use_smem) {
            size_t smem_iso = nc * sizeof(int32_t);
            count_isolated_kernel<BLOCK><<<blocks, BLOCK, smem_iso>>>(
                cluster_assignments, cache.cluster_sizes, seg3, seg4, nc
            );
        } else {
            count_isolated_kernel_noshmem<BLOCK><<<blocks, BLOCK>>>(
                cluster_assignments, cache.cluster_sizes, seg3, seg4
            );
        }
    }

    
    compute_final_score<<<1, 32>>>(cache.cluster_cut, cache.cluster_sizes, cache.d_result, nc);

    double result;
    cudaMemcpy(&result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

}  
