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
#include <cub/block/block_reduce.cuh>
#include <cstdint>

namespace aai {

namespace {

struct Cache : Cacheable {
    double* partials = nullptr;
    unsigned int* counter = nullptr;
    double* result_dev = nullptr;
    int partials_capacity = 0;

    Cache() {
        partials_capacity = 1 << 16; 
        cudaMalloc(&partials, (size_t)partials_capacity * sizeof(double));
        cudaMalloc(&counter, sizeof(unsigned int));
        cudaMemset(counter, 0, sizeof(unsigned int));
        cudaMalloc(&result_dev, sizeof(double));
    }

    ~Cache() override {
        if (partials) cudaFree(partials);
        if (counter) cudaFree(counter);
        if (result_dev) cudaFree(result_dev);
    }

    void ensure_partials(int required_blocks) {
        if (required_blocks > partials_capacity) {
            int new_cap = partials_capacity;
            if (new_cap <= 0) new_cap = 1;
            while (new_cap < required_blocks) new_cap <<= 1;
            if (partials) cudaFree(partials);
            cudaMalloc(&partials, (size_t)new_cap * sizeof(double));
            partials_capacity = new_cap;
        }
    }
};

__device__ __forceinline__ int find_vertex_global(int edge_idx_abs, const int32_t* offsets, int num_vertices) {
    int lo = 0, hi = num_vertices;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (__ldg(&offsets[mid]) <= edge_idx_abs) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ __launch_bounds__(BLOCK_SIZE)
void edge_cut_flat_retire_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster,
    double* __restrict__ out,
    double* __restrict__ partials,
    unsigned int* __restrict__ counter,
    int num_edges,
    int num_vertices)
{
    using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;

    constexpr int EDGES_PER_BLOCK = BLOCK_SIZE * ITEMS_PER_THREAD;

    const int base = __ldg(&offsets[0]);

    int64_t block_edge_start_i64 = (int64_t)blockIdx.x * EDGES_PER_BLOCK;
    if (block_edge_start_i64 >= num_edges) return;
    int block_edge_start = (int)block_edge_start_i64;
    int remaining = num_edges - block_edge_start;
    int block_edge_end = (remaining >= EDGES_PER_BLOCK) ? block_edge_start + EDGES_PER_BLOCK : num_edges;

    int block_edge_start_abs = base + block_edge_start;
    int block_edge_end_abs = base + block_edge_end;

    __shared__ int s_v_start;
    __shared__ int s_num_verts;

    if (threadIdx.x == 0) {
        int v0 = find_vertex_global(block_edge_start_abs, offsets, num_vertices);
        int v1 = find_vertex_global(block_edge_end_abs - 1, offsets, num_vertices) + 1;
        s_v_start = v0;
        s_num_verts = v1 - v0;
    }
    __syncthreads();

    int v_start = s_v_start;
    int num_block_verts = s_num_verts;
    if (num_block_verts > EDGES_PER_BLOCK + 1) num_block_verts = EDGES_PER_BLOCK + 1;

    __shared__ int s_offsets_buf[EDGES_PER_BLOCK + 2];
    __shared__ typename BlockReduce::TempStorage reduce_temp;

    for (int i = (int)threadIdx.x; i <= num_block_verts; i += BLOCK_SIZE) {
        s_offsets_buf[i] = __ldg(&offsets[v_start + i]);
    }
    __syncthreads();

    double local_sum = 0.0;
    int hint = 0;

    #pragma unroll
    for (int t = 0; t < ITEMS_PER_THREAD; t++) {
        int e = block_edge_start + t * BLOCK_SIZE + (int)threadIdx.x;
        if (e >= block_edge_end) break;

        int e_abs = base + e;

        int lo = hint, hi = num_block_verts;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (s_offsets_buf[mid] <= e_abs) lo = mid;
            else hi = mid - 1;
        }
        hint = lo;

        int src = v_start + lo;
        int dst = __ldg(&indices[e]);

        if (__ldg(&cluster[src]) != __ldg(&cluster[dst])) {
            local_sum += __ldg(&edge_weights[e]);
        }
    }

    double block_sum = BlockReduce(reduce_temp).Sum(local_sum);

    if (threadIdx.x == 0) {
        partials[blockIdx.x] = block_sum;
    }

    __threadfence();

    __shared__ bool am_last;
    __shared__ int num_blocks;

    if (threadIdx.x == 0) {
        num_blocks = (int)gridDim.x;
        unsigned int ticket = atomicAdd(counter, 1u);
        am_last = (ticket == (unsigned int)(gridDim.x - 1));
    }
    __syncthreads();

    if (am_last) {
        double s = 0.0;
        for (int i = (int)threadIdx.x; i < num_blocks; i += BLOCK_SIZE) {
            s += partials[i];
        }
        double total = BlockReduce(reduce_temp).Sum(s);
        if (threadIdx.x == 0) {
            out[0] = total * 0.5;
            *counter = 0u;
        }
    }
}

}  

double analyze_clustering_edge_cut_seg(const graph32_t& graph,
                                       const double* edge_weights,
                                       std::size_t num_clusters,
                                       const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    if (num_edges <= 0) {
        return 0.0;
    }

    constexpr int BLOCK_SIZE = 512;
    constexpr int IPT = 8;
    constexpr int EPB = BLOCK_SIZE * IPT;

    int required_blocks = (num_edges + EPB - 1) / EPB;
    cache.ensure_partials(required_blocks);

    int grid = (num_edges + EPB - 1) / EPB;

    edge_cut_flat_retire_kernel<BLOCK_SIZE, IPT><<<grid, BLOCK_SIZE>>>(
        graph.offsets, graph.indices, edge_weights, cluster_assignments,
        cache.result_dev, cache.partials, cache.counter,
        num_edges, num_vertices);

    double result;
    cudaMemcpy(&result, cache.result_dev, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

}  
