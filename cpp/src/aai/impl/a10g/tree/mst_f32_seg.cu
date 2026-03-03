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
    int32_t* d_parent = nullptr;
    unsigned long long* d_min_edge = nullptr;
    int32_t* d_was_root = nullptr;
    int32_t* d_mst_count = nullptr;
    int32_t* d_changed_flag = nullptr;
    size_t alloc_vertices = 0;

    void ensure(int32_t num_vertices) {
        if ((size_t)num_vertices > alloc_vertices) {
            if (d_parent) cudaFree(d_parent);
            if (d_min_edge) cudaFree(d_min_edge);
            if (d_was_root) cudaFree(d_was_root);

            alloc_vertices = (size_t)num_vertices * 2;
            cudaMalloc(&d_parent, alloc_vertices * sizeof(int32_t));
            cudaMalloc(&d_min_edge, alloc_vertices * sizeof(unsigned long long));
            cudaMalloc(&d_was_root, alloc_vertices * sizeof(int32_t));
        }
        if (!d_mst_count) cudaMalloc(&d_mst_count, sizeof(int32_t));
        if (!d_changed_flag) cudaMalloc(&d_changed_flag, sizeof(int32_t));
    }

    ~Cache() override {
        if (d_parent) cudaFree(d_parent);
        if (d_min_edge) cudaFree(d_min_edge);
        if (d_was_root) cudaFree(d_was_root);
        if (d_mst_count) cudaFree(d_mst_count);
        if (d_changed_flag) cudaFree(d_changed_flag);
    }
};

__device__ __forceinline__ uint32_t float_to_sortable(float f) {
    uint32_t w = __float_as_uint(f);
    uint32_t mask = (-int32_t(w >> 31)) | 0x80000000u;
    return w ^ mask;
}

__device__ __forceinline__ unsigned long long encode_weight_edge(float weight, int32_t edge_idx) {
    uint32_t w = float_to_sortable(weight);
    return (((unsigned long long)w) << 32) | (unsigned long long)(uint32_t)edge_idx;
}

static constexpr unsigned long long INVALID_EDGE = 0xFFFFFFFFFFFFFFFFull;

__device__ __forceinline__ int32_t find_src_from_edge(
    const int32_t* __restrict__ offsets, int32_t num_vertices, int32_t edge_idx
) {
    int32_t lo = 0, hi = num_vertices;
    while (lo < hi) {
        int32_t mid = lo + (hi - lo + 1) / 2;
        if (__ldg(&offsets[mid]) <= edge_idx) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

__global__ void init_parent_kernel(int32_t* parent, int32_t num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) parent[v] = v;
}

__global__ void prepare_kernel(
    int32_t* __restrict__ parent,
    unsigned long long* __restrict__ min_edge,
    int32_t* __restrict__ was_root,
    int32_t n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int32_t r = idx;
    while (parent[r] != r) r = parent[r];
    parent[idx] = r;
    min_edge[idx] = INVALID_EDGE;
    was_root[idx] = (idx == r) ? 1 : 0;
}

__global__ void find_min_edge_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ parent,
    unsigned long long* __restrict__ min_edge,
    int32_t num_vertices
) {
    const int lane = threadIdx.x & 31;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int total_warps = (gridDim.x * blockDim.x) >> 5;

    for (int v = warp_id; v < num_vertices; v += total_warps) {
        int32_t root_v = __ldg(&parent[v]);
        int32_t start = __ldg(&offsets[v]);
        int32_t end = __ldg(&offsets[v + 1]);
        int32_t degree = end - start;

        unsigned long long warp_min = INVALID_EDGE;

        for (int32_t i = lane; i < degree; i += 32) {
            int32_t e = start + i;
            int32_t u = __ldg(&indices[e]);
            int32_t root_u = __ldg(&parent[u]);
            if (root_v != root_u) {
                float w = __ldg(&weights[e]);
                unsigned long long encoded = encode_weight_edge(w, e);
                if (encoded < warp_min) warp_min = encoded;
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            unsigned long long other = __shfl_down_sync(0xFFFFFFFF, warp_min, offset);
            if (other < warp_min) warp_min = other;
        }

        if (lane == 0 && warp_min != INVALID_EDGE) {
            atomicMin(&min_edge[root_v], warp_min);
        }
    }
}

__global__ void hook_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ parent,
    const unsigned long long* __restrict__ min_edge,
    const int32_t* __restrict__ was_root,
    int32_t* __restrict__ changed_flag,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    if (!was_root[v]) return;
    if (parent[v] != v) return;

    unsigned long long me = min_edge[v];
    if (me == INVALID_EDGE) return;

    int32_t edge_idx = (int32_t)(uint32_t)(me & 0xFFFFFFFFull);
    int32_t src = find_src_from_edge(offsets, num_vertices, edge_idx);
    int32_t dst = __ldg(&indices[edge_idx]);
    int32_t rs = parent[src];
    int32_t rd = parent[dst];

    if (rs == rd) return;

    int32_t succ = (rs == v) ? rd : rs;
    parent[v] = succ;
    *changed_flag = 1;
}

__global__ void min_pointer_jump_kernel(int32_t* parent, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int32_t p = parent[idx];
    if (p != idx) {
        int32_t pp = parent[p];
        int32_t new_p = p < pp ? p : pp;
        if (new_p < parent[idx]) parent[idx] = new_p;
    }
}

__global__ void collect_mst_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ parent,
    const unsigned long long* __restrict__ min_edge,
    const int32_t* __restrict__ was_root,
    int32_t* __restrict__ mst_srcs,
    int32_t* __restrict__ mst_dsts,
    float* __restrict__ mst_weights,
    int32_t* __restrict__ mst_count,
    int32_t num_vertices,
    int32_t max_mst_edges
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    if (!was_root[v]) return;
    if (parent[v] == v) return;

    unsigned long long me = min_edge[v];
    if (me == INVALID_EDGE) return;

    int32_t edge_idx = (int32_t)(uint32_t)(me & 0xFFFFFFFFull);
    int32_t src = find_src_from_edge(offsets, num_vertices, edge_idx);
    int32_t dst = __ldg(&indices[edge_idx]);
    float w = __ldg(&weights[edge_idx]);

    int32_t idx = atomicAdd(mst_count, 1);
    if (idx < max_mst_edges) {
        int32_t out = idx * 2;
        mst_srcs[out] = src;
        mst_dsts[out] = dst;
        mst_weights[out] = w;
        mst_srcs[out + 1] = dst;
        mst_dsts[out + 1] = src;
        mst_weights[out + 1] = w;
    }
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

    cudaStream_t stream = 0;
    cache.ensure(num_vertices);

    init_parent_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(cache.d_parent, num_vertices);
    cudaMemsetAsync(cache.d_mst_count, 0, sizeof(int32_t), stream);

    int32_t max_mst_edges = num_vertices;
    const int jump_rounds = 2;

    for (int iter = 0; iter < 40; iter++) {
        {
            int b = 256, g = (num_vertices + b - 1) / b;
            prepare_kernel<<<g, b, 0, stream>>>(cache.d_parent, cache.d_min_edge, cache.d_was_root, num_vertices);
        }

        {
            int threads_per_block = 256;
            int warps_per_block = threads_per_block / 32;
            int num_blocks = (num_vertices + warps_per_block - 1) / warps_per_block;
            num_blocks = num_blocks < 4096 ? num_blocks : 4096;
            find_min_edge_warp_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
                d_offsets, d_indices, edge_weights, cache.d_parent, cache.d_min_edge, num_vertices
            );
        }

        cudaMemsetAsync(cache.d_changed_flag, 0, sizeof(int32_t), stream);

        {
            int b = 256, g = (num_vertices + b - 1) / b;
            hook_kernel<<<g, b, 0, stream>>>(d_offsets, d_indices, cache.d_parent, cache.d_min_edge,
                                             cache.d_was_root, cache.d_changed_flag, num_vertices);
        }

        for (int j = 0; j < jump_rounds; j++) {
            int b = 512, g = (num_vertices + b - 1) / b;
            min_pointer_jump_kernel<<<g, b, 0, stream>>>(cache.d_parent, num_vertices);
        }

        {
            int b = 256, g = (num_vertices + b - 1) / b;
            collect_mst_kernel<<<g, b, 0, stream>>>(
                d_offsets, d_indices, edge_weights, cache.d_parent, cache.d_min_edge, cache.d_was_root,
                mst_srcs, mst_dsts, mst_weights, cache.d_mst_count,
                num_vertices, max_mst_edges
            );
        }

        int32_t h_changed = 0;
        cudaMemcpyAsync(&h_changed, cache.d_changed_flag, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (!h_changed) break;
    }

    int32_t h_mst_count = 0;
    cudaMemcpy(&h_mst_count, cache.d_mst_count, sizeof(int32_t), cudaMemcpyDeviceToHost);

    return static_cast<std::size_t>(h_mst_count) * 2;
}

}  
