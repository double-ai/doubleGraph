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

constexpr int BLOCK_SIZE = 256;
constexpr int ITEMS_PER_THREAD = 16;
constexpr int EDGES_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;  
constexpr int MAX_SHARED_VERTS = 4096;

struct Cache : Cacheable {
    double* d_result = nullptr;

    Cache() {
        cudaMalloc(&d_result, sizeof(double));
    }

    ~Cache() override {
        if (d_result) cudaFree(d_result);
    }
};

__device__ __forceinline__ int32_t global_find_vertex(
    const int32_t* offsets, int32_t num_vertices, int32_t edge_idx
) {
    int32_t lo = 0, hi = num_vertices;
    while (lo < hi) {
        int32_t mid = (lo + hi) >> 1;
        if (offsets[mid + 1] <= edge_idx) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void edge_cut_kernel(
    const int32_t* __restrict__ g_offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    int32_t num_vertices,
    int32_t num_edges,
    double* __restrict__ result
) {
    extern __shared__ int32_t s_data[];
    int32_t* s_offsets = s_data;

    int64_t block_edge_start_i64 = (int64_t)blockIdx.x * EDGES_PER_BLOCK;
    if (block_edge_start_i64 >= num_edges) return;
    int32_t block_edge_start = (int32_t)block_edge_start_i64;
    int32_t remaining = num_edges - block_edge_start;
    int32_t block_edge_end = (remaining >= EDGES_PER_BLOCK) ? block_edge_start + EDGES_PER_BLOCK : num_edges;

    int32_t seg_start, num_segs;

    
    if (threadIdx.x == 0) {
        seg_start = global_find_vertex(g_offsets, num_vertices, block_edge_start);
        int32_t seg_end = global_find_vertex(g_offsets, num_vertices, block_edge_end - 1) + 1;
        if (seg_end > num_vertices) seg_end = num_vertices;
        num_segs = seg_end - seg_start;
        s_data[MAX_SHARED_VERTS + 1] = seg_start;
        s_data[MAX_SHARED_VERTS + 2] = num_segs;
    }
    __syncthreads();

    seg_start = s_data[MAX_SHARED_VERTS + 1];
    num_segs = s_data[MAX_SHARED_VERTS + 2];

    bool use_shared = (num_segs <= MAX_SHARED_VERTS);

    double thread_sum = 0.0;

    if (use_shared) {
        
        for (int i = threadIdx.x; i <= num_segs; i += BLOCK_SIZE) {
            s_offsets[i] = g_offsets[seg_start + i];
        }
        __syncthreads();

        
        for (int item = 0; item < ITEMS_PER_THREAD; item++) {
            int32_t e = block_edge_start + item * BLOCK_SIZE + threadIdx.x;
            if (e >= block_edge_end) break;

            int32_t lo = 0, hi = num_segs;
            while (lo < hi) {
                int32_t mid = (lo + hi) >> 1;
                if (s_offsets[mid + 1] <= e) lo = mid + 1;
                else hi = mid;
            }
            int32_t src = seg_start + lo;
            int32_t dst = indices[e];

            if (cluster_assignments[src] != cluster_assignments[dst]) {
                thread_sum += edge_weights[e];
            }
        }
    } else {
        
        for (int item = 0; item < ITEMS_PER_THREAD; item++) {
            int32_t e = block_edge_start + item * BLOCK_SIZE + threadIdx.x;
            if (e >= block_edge_end) break;

            int32_t src = global_find_vertex(g_offsets, num_vertices, e);
            int32_t dst = indices[e];

            if (cluster_assignments[src] != cluster_assignments[dst]) {
                thread_sum += edge_weights[e];
            }
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        atomicAdd(result, thread_sum);
    }
}

__global__ void halve_kernel(double* result) {
    *result *= 0.5;
}

}  

double analyze_clustering_edge_cut(const graph32_t& graph,
                                   const double* edge_weights,
                                   std::size_t num_clusters,
                                   const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    cudaMemsetAsync(cache.d_result, 0, sizeof(double));

    int grid_size = (num_edges + EDGES_PER_BLOCK - 1) / EDGES_PER_BLOCK;
    if (grid_size < 1) grid_size = 1;

    size_t smem_size = (MAX_SHARED_VERTS + 3) * sizeof(int32_t);

    edge_cut_kernel<<<grid_size, BLOCK_SIZE, smem_size>>>(
        offsets, indices, edge_weights, cluster_assignments,
        num_vertices, num_edges, cache.d_result
    );

    halve_kernel<<<1, 1>>>(cache.d_result);

    double h_result;
    cudaMemcpy(&h_result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);

    return h_result;
}

}  
