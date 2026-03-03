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
#include <climits>
#include <cstdint>
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    int* d_parent = nullptr;
    unsigned long long* d_min_edge = nullptr;
    int* d_src_array = nullptr;
    int* d_mst_srcs_tmp = nullptr;
    int* d_mst_dsts_tmp = nullptr;
    float* d_mst_weights_tmp = nullptr;
    long long* d_mst_count = nullptr;
    int* d_changed_flag = nullptr;
    int* h_changed_pinned = nullptr;
    long long* h_mst_count_pinned = nullptr;

    int max_vertices = 0;
    int max_edges_stored = 0;

    void ensure(int num_vertices, int num_edges) {
        if (num_vertices <= max_vertices && num_edges <= max_edges_stored) return;
        free_all();
        max_vertices = std::max(num_vertices, max_vertices);
        max_edges_stored = std::max(num_edges, max_edges_stored);

        cudaMalloc(&d_parent, max_vertices * sizeof(int));
        cudaMalloc(&d_min_edge, max_vertices * sizeof(unsigned long long));
        cudaMalloc(&d_src_array, max_edges_stored * sizeof(int));
        cudaMalloc(&d_mst_srcs_tmp, max_vertices * sizeof(int));
        cudaMalloc(&d_mst_dsts_tmp, max_vertices * sizeof(int));
        cudaMalloc(&d_mst_weights_tmp, max_vertices * sizeof(float));
        cudaMalloc(&d_mst_count, sizeof(long long));
        cudaMalloc(&d_changed_flag, sizeof(int));
        cudaMallocHost(&h_changed_pinned, sizeof(int));
        cudaMallocHost(&h_mst_count_pinned, sizeof(long long));
    }

    void free_all() {
        auto free_if = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        free_if(d_parent);
        free_if(d_min_edge);
        free_if(d_src_array);
        free_if(d_mst_srcs_tmp);
        free_if(d_mst_dsts_tmp);
        free_if(d_mst_weights_tmp);
        free_if(d_mst_count);
        free_if(d_changed_flag);
        if (h_changed_pinned) { cudaFreeHost(h_changed_pinned); h_changed_pinned = nullptr; }
        if (h_mst_count_pinned) { cudaFreeHost(h_mst_count_pinned); h_mst_count_pinned = nullptr; }
        max_vertices = 0;
        max_edges_stored = 0;
    }

    ~Cache() override { free_all(); }
};

__device__ __forceinline__ unsigned int float_to_ordered_uint(float f) {
    unsigned int i = __float_as_uint(f);
    unsigned int mask = (i & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return i ^ mask;
}

__global__ void compute_src_kernel(const int* offsets, int* src, int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int start = offsets[v];
    int end = offsets[v + 1];
    for (int e = start; e < end; e++) {
        src[e] = v;
    }
}

__global__ void init_kernel(int* parent, unsigned long long* min_edge, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        parent[tid] = tid;
        min_edge[tid] = ULLONG_MAX;
    }
}

__global__ void path_compress_and_init_kernel(int* parent, unsigned long long* min_edge, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int x = tid;
    int root = x;
    while (parent[root] != root) root = parent[root];
    while (parent[x] != root) {
        int next = parent[x];
        parent[x] = root;
        x = next;
    }
    min_edge[tid] = ULLONG_MAX;
}

__global__ void find_min_edge_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ weights,
    const int* __restrict__ parent,
    unsigned long long* __restrict__ min_edge,
    int num_vertices)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_vertices) return;

    int v = warp_id;
    int cv = parent[v];
    int start = offsets[v];
    int end = offsets[v + 1];

    unsigned long long local_min = ULLONG_MAX;

    for (int e = start + lane; e < end; e += 32) {
        int u = indices[e];
        int cu = parent[u];
        if (cv != cu) {
            unsigned int w = float_to_ordered_uint(weights[e]);
            unsigned long long packed = ((unsigned long long)w << 32) | (unsigned int)e;
            if (packed < local_min) local_min = packed;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        unsigned long long other = __shfl_down_sync(0xFFFFFFFF, local_min, offset);
        if (other < local_min) local_min = other;
    }

    if (lane == 0 && local_min != ULLONG_MAX) {
        atomicMin(&min_edge[cv], local_min);
    }
}

__global__ void merge_and_record_kernel(
    int* __restrict__ parent,
    const unsigned long long* __restrict__ min_edge,
    const int* __restrict__ indices,
    const int* __restrict__ src_array,
    const float* __restrict__ weights,
    int* __restrict__ mst_srcs,
    int* __restrict__ mst_dsts,
    float* __restrict__ mst_weights,
    long long* __restrict__ mst_count,
    int* __restrict__ changed_flag,
    int num_vertices,
    int max_mst_edges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_vertices) return;
    if (parent[i] != i) return;

    unsigned long long me = min_edge[i];
    if (me == ULLONG_MAX) return;

    int edge_idx = (int)(me & 0xFFFFFFFFu);
    int dst = indices[edge_idx];
    int j = parent[dst];

    if (j == i) return;

    if (i < j) {
        unsigned long long mj = min_edge[j];
        if (mj != ULLONG_MAX) {
            int j_edge_idx = (int)(mj & 0xFFFFFFFFu);
            int j_dst = indices[j_edge_idx];
            int j_target = parent[j_dst];
            if (j_target == i) return;
        }
    }

    int src_v = src_array[edge_idx];
    float w = weights[edge_idx];

    long long pos = (long long)atomicAdd((unsigned long long*)mst_count, 1ULL);
    if (pos < max_mst_edges) {
        mst_srcs[pos] = src_v;
        mst_dsts[pos] = dst;
        mst_weights[pos] = w;
    }

    parent[i] = j;
    *changed_flag = 1;
}

__global__ void symmetrize_kernel(
    const int* __restrict__ srcs_in,
    const int* __restrict__ dsts_in,
    const float* __restrict__ weights_in,
    int* __restrict__ srcs_out,
    int* __restrict__ dsts_out,
    float* __restrict__ weights_out,
    int num_edges)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_edges) return;

    int s = srcs_in[tid];
    int d = dsts_in[tid];
    float w = weights_in[tid];

    int64_t out_idx = 2LL * tid;
    srcs_out[out_idx] = s;
    dsts_out[out_idx] = d;
    weights_out[out_idx] = w;
    srcs_out[out_idx + 1] = d;
    dsts_out[out_idx + 1] = s;
    weights_out[out_idx + 1] = w;
}

}  

std::size_t minimum_spanning_tree_seg(const graph32_t& graph,
                                      const float* edge_weights,
                                      int32_t* mst_srcs,
                                      int32_t* mst_dsts,
                                      float* mst_weights,
                                      std::size_t max_edges) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges_graph = graph.number_of_edges;

    cache.ensure(num_vertices, num_edges_graph);

    cudaStream_t stream = nullptr;

    
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        compute_src_kernel<<<grid, block>>>(d_offsets, cache.d_src_array, num_vertices);
    }

    
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        init_kernel<<<grid, block>>>(cache.d_parent, cache.d_min_edge, num_vertices);
    }

    cudaMemsetAsync(cache.d_mst_count, 0, sizeof(long long), stream);

    int max_undirected_mst = num_vertices;

    for (int iter = 0; iter < 40; iter++) {
        if (iter > 0) {
            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            path_compress_and_init_kernel<<<grid, block>>>(cache.d_parent, cache.d_min_edge, num_vertices);
        }

        {
            int block = 256;
            int num_warps = num_vertices;
            int warps_per_block = block / 32;
            int grid = (num_warps + warps_per_block - 1) / warps_per_block;
            find_min_edge_kernel<<<grid, block>>>(d_offsets, d_indices, edge_weights,
                                                   cache.d_parent, cache.d_min_edge, num_vertices);
        }

        cudaMemsetAsync(cache.d_changed_flag, 0, sizeof(int), stream);

        {
            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            merge_and_record_kernel<<<grid, block>>>(
                cache.d_parent, cache.d_min_edge,
                d_indices, cache.d_src_array, edge_weights,
                cache.d_mst_srcs_tmp, cache.d_mst_dsts_tmp, cache.d_mst_weights_tmp,
                cache.d_mst_count, cache.d_changed_flag,
                num_vertices, max_undirected_mst);
        }

        cudaMemcpyAsync(cache.h_changed_pinned, cache.d_changed_flag, sizeof(int),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (!*cache.h_changed_pinned) break;
    }

    cudaMemcpyAsync(cache.h_mst_count_pinned, cache.d_mst_count, sizeof(long long),
                     cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int h_mst_count = (int)*cache.h_mst_count_pinned;

    if (h_mst_count > 0) {
        int block = 256;
        int grid = (h_mst_count + block - 1) / block;
        symmetrize_kernel<<<grid, block>>>(
            cache.d_mst_srcs_tmp, cache.d_mst_dsts_tmp, cache.d_mst_weights_tmp,
            mst_srcs, mst_dsts, mst_weights,
            h_mst_count);
    }

    return static_cast<std::size_t>(2 * h_mst_count);
}

}  
