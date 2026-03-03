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
    int32_t* d_component = nullptr;
    unsigned long long* d_comp_min_key = nullptr;
    int32_t* d_comp_min_vertex = nullptr;
    int32_t* d_v_min_dst = nullptr;
    double* d_v_min_wt = nullptr;
    int32_t* d_mst_src_temp = nullptr;
    int32_t* d_mst_dst_temp = nullptr;
    double* d_mst_wt_temp = nullptr;
    int32_t* d_mst_count = nullptr;
    int32_t* d_changed = nullptr;
    int32_t* h_changed_pinned = nullptr;
    size_t alloc_verts = 0;

    void ensure_scratch(int32_t nv) {
        if ((size_t)nv <= alloc_verts) return;
        free_scratch();
        alloc_verts = nv;
        cudaMalloc(&d_component, nv * sizeof(int32_t));
        cudaMalloc(&d_comp_min_key, nv * sizeof(unsigned long long));
        cudaMalloc(&d_comp_min_vertex, nv * sizeof(int32_t));
        cudaMalloc(&d_v_min_dst, nv * sizeof(int32_t));
        cudaMalloc(&d_v_min_wt, nv * sizeof(double));
        cudaMalloc(&d_mst_src_temp, nv * sizeof(int32_t));
        cudaMalloc(&d_mst_dst_temp, nv * sizeof(int32_t));
        cudaMalloc(&d_mst_wt_temp, nv * sizeof(double));
        cudaMalloc(&d_mst_count, sizeof(int32_t));
        cudaMalloc(&d_changed, sizeof(int32_t));
        cudaHostAlloc(&h_changed_pinned, sizeof(int32_t), cudaHostAllocDefault);
    }

    void free_scratch() {
        if (d_component) { cudaFree(d_component); d_component = nullptr; }
        if (d_comp_min_key) { cudaFree(d_comp_min_key); d_comp_min_key = nullptr; }
        if (d_comp_min_vertex) { cudaFree(d_comp_min_vertex); d_comp_min_vertex = nullptr; }
        if (d_v_min_dst) { cudaFree(d_v_min_dst); d_v_min_dst = nullptr; }
        if (d_v_min_wt) { cudaFree(d_v_min_wt); d_v_min_wt = nullptr; }
        if (d_mst_src_temp) { cudaFree(d_mst_src_temp); d_mst_src_temp = nullptr; }
        if (d_mst_dst_temp) { cudaFree(d_mst_dst_temp); d_mst_dst_temp = nullptr; }
        if (d_mst_wt_temp) { cudaFree(d_mst_wt_temp); d_mst_wt_temp = nullptr; }
        if (d_mst_count) { cudaFree(d_mst_count); d_mst_count = nullptr; }
        if (d_changed) { cudaFree(d_changed); d_changed = nullptr; }
        if (h_changed_pinned) { cudaFreeHost(h_changed_pinned); h_changed_pinned = nullptr; }
        alloc_verts = 0;
    }

    ~Cache() override {
        free_scratch();
    }
};

__device__ __forceinline__ unsigned long long double_to_sortable(double val) {
    unsigned long long bits = __double_as_longlong(val);
    unsigned long long mask = -((long long)(bits >> 63)) | 0x8000000000000000ULL;
    return bits ^ mask;
}

__global__ void k_init_components(int32_t* __restrict__ component, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) component[v] = v;
}

__global__ void k_init_iteration(
    unsigned long long* __restrict__ comp_min_key,
    int32_t* __restrict__ comp_min_vertex,
    int32_t n
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        comp_min_key[v] = 0xFFFFFFFFFFFFFFFFULL;
        comp_min_vertex[v] = -1;
    }
}

__global__ void k_find_min_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ component,
    unsigned long long* __restrict__ comp_min_key,
    int32_t* __restrict__ v_min_dst,
    double* __restrict__ v_min_wt,
    int32_t num_vertices
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= num_vertices) return;

    const int v = warp_id;
    const int comp_v = component[v];
    const int start = offsets[v];
    const int end = offsets[v + 1];

    unsigned long long my_best_key = 0xFFFFFFFFFFFFFFFFULL;
    int my_best_dst = -1;
    double my_best_wt = 0.0;

    for (int e = start + lane; e < end; e += 32) {
        int u = indices[e];
        if (component[u] != comp_v) {
            unsigned long long key = double_to_sortable(weights[e]);
            if (key < my_best_key) {
                my_best_key = key;
                my_best_dst = u;
                my_best_wt = weights[e];
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        unsigned long long other_key = __shfl_down_sync(0xFFFFFFFF, my_best_key, offset);
        int other_dst = __shfl_down_sync(0xFFFFFFFF, my_best_dst, offset);
        unsigned long long other_wt_bits = __shfl_down_sync(0xFFFFFFFF, __double_as_longlong(my_best_wt), offset);

        if (other_key < my_best_key) {
            my_best_key = other_key;
            my_best_dst = other_dst;
            my_best_wt = __longlong_as_double(other_wt_bits);
        }
    }

    if (lane == 0) {
        v_min_dst[v] = my_best_dst;
        v_min_wt[v] = my_best_wt;

        if (my_best_dst != -1) {
            atomicMin(&comp_min_key[comp_v], my_best_key);
        }
    }
}

__global__ void k_claim_vertex(
    const int32_t* __restrict__ component,
    const unsigned long long* __restrict__ comp_min_key,
    const double* __restrict__ v_min_wt,
    const int32_t* __restrict__ v_min_dst,
    int32_t* __restrict__ comp_min_vertex,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    if (v_min_dst[v] == -1) return;
    int rv = component[v];
    unsigned long long key = double_to_sortable(v_min_wt[v]);
    if (key == comp_min_key[rv]) {
        atomicCAS(&comp_min_vertex[rv], -1, v);
    }
}

__global__ void k_add_and_merge(
    int32_t* __restrict__ component,
    const int32_t* __restrict__ comp_min_vertex,
    const int32_t* __restrict__ v_min_dst,
    const double* __restrict__ v_min_wt,
    int32_t* __restrict__ mst_src,
    int32_t* __restrict__ mst_dst,
    double* __restrict__ mst_wt,
    int32_t* __restrict__ mst_count,
    int32_t* __restrict__ changed,
    int32_t num_vertices
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= num_vertices) return;
    if (component[c] != c) return;

    int sv = comp_min_vertex[c];
    if (sv == -1) return;
    int dst = v_min_dst[sv];
    double wt = v_min_wt[sv];
    if (dst == -1) return;

    while (true) {
        int ru = c;
        while (component[ru] != ru) ru = component[ru];
        int rv = dst;
        while (component[rv] != rv) rv = component[rv];

        if (ru == rv) break;

        int s = (ru < rv) ? ru : rv;
        int l = (ru < rv) ? rv : ru;

        int old = atomicCAS(&component[l], l, s);
        if (old == l) {
            int idx = atomicAdd(mst_count, 1);
            mst_src[idx] = sv;
            mst_dst[idx] = dst;
            mst_wt[idx] = wt;
            *changed = 1;
            break;
        }
    }
}

__global__ void k_path_compress(int32_t* __restrict__ component, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int root = v;
    while (component[root] != root) root = component[root];
    component[v] = root;
}

__global__ void k_symmetrize(
    const int32_t* __restrict__ src, const int32_t* __restrict__ dst,
    const double* __restrict__ wt,
    int32_t* __restrict__ out_src, int32_t* __restrict__ out_dst,
    double* __restrict__ out_wt, int32_t n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out_src[2*i] = src[i];
    out_dst[2*i] = dst[i];
    out_wt[2*i] = wt[i];
    out_src[2*i+1] = dst[i];
    out_dst[2*i+1] = src[i];
    out_wt[2*i+1] = wt[i];
}

}  

std::size_t minimum_spanning_tree_seg(const graph32_t& graph,
                                      const double* edge_weights,
                                      int32_t* mst_srcs,
                                      int32_t* mst_dsts,
                                      double* mst_weights,
                                      std::size_t max_edges) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    if (num_vertices <= 1 || num_edges == 0) {
        return 0;
    }

    cache.ensure_scratch(num_vertices);
    cudaStream_t stream = 0;

    k_init_components<<<(num_vertices + 255) / 256, 256, 0, stream>>>(cache.d_component, num_vertices);
    cudaMemsetAsync(cache.d_mst_count, 0, sizeof(int32_t), stream);

    const int CHECK_INTERVAL = 2;

    for (int iter = 0; iter < 40; iter++) {
        k_init_iteration<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
            cache.d_comp_min_key, cache.d_comp_min_vertex, num_vertices);

        {
            int64_t total_threads = (int64_t)num_vertices * 32;
            int block_size = 256;
            int grid_size = (int)((total_threads + block_size - 1) / block_size);
            k_find_min_warp<<<grid_size, block_size, 0, stream>>>(
                d_offsets, d_indices, edge_weights, cache.d_component,
                cache.d_comp_min_key, cache.d_v_min_dst, cache.d_v_min_wt,
                num_vertices);
        }

        k_claim_vertex<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
            cache.d_component, cache.d_comp_min_key, cache.d_v_min_wt,
            cache.d_v_min_dst, cache.d_comp_min_vertex, num_vertices);

        cudaMemsetAsync(cache.d_changed, 0, sizeof(int32_t), stream);

        k_add_and_merge<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
            cache.d_component, cache.d_comp_min_vertex, cache.d_v_min_dst,
            cache.d_v_min_wt, cache.d_mst_src_temp, cache.d_mst_dst_temp,
            cache.d_mst_wt_temp, cache.d_mst_count, cache.d_changed,
            num_vertices);

        k_path_compress<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
            cache.d_component, num_vertices);

        if ((iter + 1) % CHECK_INTERVAL == 0 || iter >= 38) {
            cudaMemcpyAsync(cache.h_changed_pinned, cache.d_changed, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (!*cache.h_changed_pinned) break;
        }
    }

    int32_t h_mst_count;
    cudaMemcpy(&h_mst_count, cache.d_mst_count, sizeof(int32_t), cudaMemcpyDeviceToHost);

    std::size_t num_directed = 2 * (std::size_t)h_mst_count;

    if (h_mst_count > 0) {
        k_symmetrize<<<(h_mst_count + 255) / 256, 256, 0, stream>>>(
            cache.d_mst_src_temp, cache.d_mst_dst_temp, cache.d_mst_wt_temp,
            mst_srcs, mst_dsts, mst_weights, h_mst_count);
    }

    cudaStreamSynchronize(stream);
    return num_directed;
}

}  
