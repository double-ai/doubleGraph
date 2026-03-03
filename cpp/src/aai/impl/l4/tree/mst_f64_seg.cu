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
#include <climits>
#include <math_constants.h>

namespace aai {

namespace {



__device__ __forceinline__ unsigned long long double_to_sortable(double val) {
    unsigned long long bits = __double_as_longlong(val);
    unsigned long long mask = ((unsigned long long)((long long)bits >> 63)) | 0x8000000000000000ULL;
    return bits ^ mask;
}

__device__ __forceinline__ double sortable_to_double(unsigned long long enc) {
    unsigned long long mask = ((enc >> 63) - 1) | 0x8000000000000000ULL;
    return __longlong_as_double(enc ^ mask);
}

__device__ __forceinline__ int find_root(int* __restrict__ parent, int x) {
    int p = parent[x];
    while (p != x) {
        int gp = parent[p];
        if (p != gp) parent[x] = gp;
        x = p;
        p = gp;
    }
    return x;
}

#define SENTINEL_ENC 0xFFFFFFFFFFFFFFFFULL
#define SENTINEL_SRC 0x7FFFFFFF



__global__ void init_kernel(
    int* __restrict__ parent,
    unsigned long long* __restrict__ comp_min_enc,
    int* __restrict__ comp_min_src,
    int n
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        parent[v] = v;
        comp_min_enc[v] = SENTINEL_ENC;
        comp_min_src[v] = SENTINEL_SRC;
    }
}

__global__ void find_min_edge_first_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const double* __restrict__ weights,
    unsigned long long* __restrict__ comp_min_enc,
    unsigned long long* __restrict__ vert_min_enc,
    int* __restrict__ vert_min_dst,
    int num_vertices
) {
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_global >= num_vertices) return;

    int v = warp_global;
    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);

    unsigned long long local_min_enc = SENTINEL_ENC;
    int local_min_dst = -1;

    for (int e = start + lane; e < end; e += 32) {
        double w = __ldg(&weights[e]);
        unsigned long long enc = double_to_sortable(w);
        if (enc < local_min_enc) {
            local_min_enc = enc;
            local_min_dst = __ldg(&indices[e]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        unsigned long long other_enc = __shfl_down_sync(0xFFFFFFFF, local_min_enc, offset);
        int other_dst = __shfl_down_sync(0xFFFFFFFF, local_min_dst, offset);
        if (other_enc < local_min_enc) {
            local_min_enc = other_enc;
            local_min_dst = other_dst;
        }
    }

    if (lane == 0) {
        vert_min_enc[v] = local_min_enc;
        vert_min_dst[v] = local_min_dst;
        if (local_min_dst != -1) {
            atomicMin(&comp_min_enc[v], local_min_enc);
        }
    }
}

__global__ void find_min_edge_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const double* __restrict__ weights,
    const int* __restrict__ parent,
    unsigned long long* __restrict__ comp_min_enc,
    unsigned long long* __restrict__ vert_min_enc,
    int* __restrict__ vert_min_dst,
    const int* __restrict__ active_list,
    const int* __restrict__ d_num_active,
    int num_vertices
) {
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    int num_active = *d_num_active;
    if (warp_global >= num_active) return;

    int v = active_list[warp_global];
    int comp_v = parent[v];
    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);

    unsigned long long local_min_enc = SENTINEL_ENC;
    int local_min_dst = -1;

    for (int e = start + lane; e < end; e += 32) {
        int u = __ldg(&indices[e]);
        int comp_u = parent[u];
        if (comp_v != comp_u) {
            double w = __ldg(&weights[e]);
            unsigned long long enc = double_to_sortable(w);
            if (enc < local_min_enc) {
                local_min_enc = enc;
                local_min_dst = u;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        unsigned long long other_enc = __shfl_down_sync(0xFFFFFFFF, local_min_enc, offset);
        int other_dst = __shfl_down_sync(0xFFFFFFFF, local_min_dst, offset);
        if (other_enc < local_min_enc) {
            local_min_enc = other_enc;
            local_min_dst = other_dst;
        }
    }

    if (lane == 0) {
        vert_min_enc[v] = local_min_enc;
        vert_min_dst[v] = local_min_dst;
        if (local_min_dst != -1) {
            atomicMin(&comp_min_enc[comp_v], local_min_enc);
        }
    }
}

__global__ void match_merge_output_kernel(
    int* __restrict__ parent,
    const unsigned long long* __restrict__ comp_min_enc,
    const unsigned long long* __restrict__ vert_min_enc,
    const int* __restrict__ vert_min_dst,
    int* __restrict__ comp_min_src,
    int* __restrict__ mst_srcs,
    int* __restrict__ mst_dsts,
    double* __restrict__ mst_weights,
    long long* __restrict__ mst_count,
    int num_vertices,
    int64_t max_edges
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    unsigned long long my_enc = vert_min_enc[v];
    if (my_enc == SENTINEL_ENC) return;

    int comp_v = parent[v];
    unsigned long long target_enc = comp_min_enc[comp_v];
    if (my_enc != target_enc) return;

    int old_src = atomicCAS(&comp_min_src[comp_v], SENTINEL_SRC, v);
    if (old_src != SENTINEL_SRC) return;

    int dst = vert_min_dst[v];
    double w = sortable_to_double(my_enc);

    int root_v = find_root(parent, v);
    int root_dst = find_root(parent, dst);

    while (root_v != root_dst) {
        int small = (root_v < root_dst) ? root_v : root_dst;
        int large = (root_v < root_dst) ? root_dst : root_v;

        int old = atomicCAS(&parent[large], large, small);
        if (old == large) {
            long long pos = (long long)atomicAdd((unsigned long long*)mst_count, 2ULL);
            if (pos + 1 < max_edges) {
                mst_srcs[pos] = v;
                mst_dsts[pos] = dst;
                mst_weights[pos] = w;
                mst_srcs[pos + 1] = dst;
                mst_dsts[pos + 1] = v;
                mst_weights[pos + 1] = w;
            }
            break;
        }
        root_v = find_root(parent, root_v);
        root_dst = find_root(parent, root_dst);
    }
}

__global__ void flatten_reset_compact_kernel(
    int* __restrict__ parent,
    unsigned long long* __restrict__ comp_min_enc,
    int* __restrict__ comp_min_src,
    const unsigned long long* __restrict__ vert_min_enc,
    int* __restrict__ active_list,
    int* __restrict__ d_num_active,
    int n
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    
    int root = v;
    while (parent[root] != root) root = parent[root];
    int x = v;
    while (x != root) {
        int next = parent[x];
        parent[x] = root;
        x = next;
    }

    
    comp_min_enc[v] = SENTINEL_ENC;
    comp_min_src[v] = SENTINEL_SRC;

    
    bool is_active = (vert_min_enc[v] != SENTINEL_ENC);

    unsigned mask = __ballot_sync(0xFFFFFFFF, is_active);
    int lane = threadIdx.x & 31;

    if (mask == 0) return;

    int warp_count = __popc(mask);
    int warp_offset;
    if (lane == 0) {
        warp_offset = atomicAdd(d_num_active, warp_count);
    }
    warp_offset = __shfl_sync(0xFFFFFFFF, warp_offset, 0);

    if (is_active) {
        int lane_offset = __popc(mask & ((1u << lane) - 1));
        active_list[warp_offset + lane_offset] = v;
    }
}



struct Cache : Cacheable {
    int* d_parent = nullptr;
    unsigned long long* d_comp_min_enc = nullptr;
    unsigned long long* d_vert_min_enc = nullptr;
    int* d_vert_min_dst = nullptr;
    int* d_comp_min_src = nullptr;
    int* d_active_list = nullptr;
    int* d_num_active = nullptr;
    long long* d_mst_count = nullptr;
    int alloc_verts = 0;

    void ensure_scratch(int nv) {
        if (nv > alloc_verts) {
            auto freeIfSet = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
            freeIfSet(d_parent);
            freeIfSet(d_comp_min_enc);
            freeIfSet(d_vert_min_enc);
            freeIfSet(d_vert_min_dst);
            freeIfSet(d_comp_min_src);
            freeIfSet(d_active_list);
            freeIfSet(d_num_active);
            size_t sz = (size_t)nv;
            cudaMalloc(&d_parent, sz * sizeof(int));
            cudaMalloc(&d_comp_min_enc, sz * sizeof(unsigned long long));
            cudaMalloc(&d_vert_min_enc, sz * sizeof(unsigned long long));
            cudaMalloc(&d_vert_min_dst, sz * sizeof(int));
            cudaMalloc(&d_comp_min_src, sz * sizeof(int));
            cudaMalloc(&d_active_list, sz * sizeof(int));
            cudaMalloc(&d_num_active, sizeof(int));
            alloc_verts = nv;
        }
        if (!d_mst_count) cudaMalloc(&d_mst_count, sizeof(long long));
    }

    ~Cache() override {
        auto f = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        f(d_parent);
        f(d_comp_min_enc);
        f(d_vert_min_enc);
        f(d_vert_min_dst);
        f(d_comp_min_src);
        f(d_active_list);
        f(d_num_active);
        f(d_mst_count);
    }
};

}  

std::size_t minimum_spanning_tree_seg(const graph32_t& graph,
                                      const double* edge_weights,
                                      int32_t* mst_srcs,
                                      int32_t* mst_dsts,
                                      double* mst_weights,
                                      std::size_t max_edges) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int nv = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const double* d_weights = edge_weights;

    cache.ensure_scratch(nv);

    init_kernel<<<(nv + 255) / 256, 256>>>(
        cache.d_parent, cache.d_comp_min_enc, cache.d_comp_min_src, nv);
    cudaMemset(cache.d_mst_count, 0, sizeof(long long));

    
    {
        int64_t total = (int64_t)nv * 32;
        find_min_edge_first_kernel<<<(total + 255) / 256, 256>>>(
            d_offsets, d_indices, d_weights,
            cache.d_comp_min_enc, cache.d_vert_min_enc, cache.d_vert_min_dst, nv);
    }
    match_merge_output_kernel<<<(nv + 255) / 256, 256>>>(
        cache.d_parent, cache.d_comp_min_enc, cache.d_vert_min_enc, cache.d_vert_min_dst,
        cache.d_comp_min_src, mst_srcs, mst_dsts, mst_weights,
        cache.d_mst_count, nv, (int64_t)max_edges);

    
    cudaMemset(cache.d_num_active, 0, sizeof(int));
    flatten_reset_compact_kernel<<<(nv + 255) / 256, 256>>>(
        cache.d_parent, cache.d_comp_min_enc, cache.d_comp_min_src,
        cache.d_vert_min_enc, cache.d_active_list, cache.d_num_active, nv);

    long long prev_count;
    cudaMemcpy(&prev_count, cache.d_mst_count, sizeof(long long), cudaMemcpyDeviceToHost);

    for (int round = 1; round < 40 && prev_count > 0; round++) {
        {
            int64_t total = (int64_t)nv * 32;
            find_min_edge_kernel<<<(total + 255) / 256, 256>>>(
                d_offsets, d_indices, d_weights, cache.d_parent,
                cache.d_comp_min_enc, cache.d_vert_min_enc, cache.d_vert_min_dst,
                cache.d_active_list, cache.d_num_active, nv);
        }
        match_merge_output_kernel<<<(nv + 255) / 256, 256>>>(
            cache.d_parent, cache.d_comp_min_enc, cache.d_vert_min_enc, cache.d_vert_min_dst,
            cache.d_comp_min_src, mst_srcs, mst_dsts, mst_weights,
            cache.d_mst_count, nv, (int64_t)max_edges);

        cudaMemset(cache.d_num_active, 0, sizeof(int));
        flatten_reset_compact_kernel<<<(nv + 255) / 256, 256>>>(
            cache.d_parent, cache.d_comp_min_enc, cache.d_comp_min_src,
            cache.d_vert_min_enc, cache.d_active_list, cache.d_num_active, nv);

        if (round & 1) {
            long long current_count;
            cudaMemcpy(&current_count, cache.d_mst_count, sizeof(long long), cudaMemcpyDeviceToHost);
            if (current_count == prev_count) break;
            prev_count = current_count;
        }
    }

    long long h_mst_count;
    cudaMemcpy(&h_mst_count, cache.d_mst_count, sizeof(long long), cudaMemcpyDeviceToHost);
    return (std::size_t)h_mst_count;
}

}  
