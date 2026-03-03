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
#include <float.h>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* frontier1 = nullptr;
    int32_t* frontier2 = nullptr;
    uint32_t* bitmap = nullptr;
    int32_t* sizes = nullptr;
    int32_t* h_frontier_count = nullptr;
    int32_t* d_changed = nullptr;
    int32_t frontier_capacity = 0;
    int32_t bitmap_capacity = 0;
    int32_t sizes_capacity = 0;
    bool changed_allocated = false;

    Cache() {
        cudaHostAlloc(&h_frontier_count, sizeof(int32_t), cudaHostAllocDefault);
    }

    ~Cache() override {
        if (frontier1) cudaFree(frontier1);
        if (frontier2) cudaFree(frontier2);
        if (bitmap) cudaFree(bitmap);
        if (sizes) cudaFree(sizes);
        if (h_frontier_count) cudaFreeHost(h_frontier_count);
        if (d_changed) cudaFree(d_changed);
    }

    void ensure(int32_t num_vertices) {
        if (frontier_capacity < num_vertices) {
            if (frontier1) cudaFree(frontier1);
            if (frontier2) cudaFree(frontier2);
            cudaMalloc(&frontier1, num_vertices * sizeof(int32_t));
            cudaMalloc(&frontier2, num_vertices * sizeof(int32_t));
            frontier_capacity = num_vertices;
        }
        int32_t bitmap_words = (num_vertices + 31) / 32;
        if (bitmap_capacity < bitmap_words) {
            if (bitmap) cudaFree(bitmap);
            cudaMalloc(&bitmap, bitmap_words * sizeof(uint32_t));
            bitmap_capacity = bitmap_words;
        }
        if (sizes_capacity < 2) {
            if (sizes) cudaFree(sizes);
            cudaMalloc(&sizes, 2 * sizeof(int32_t));
            sizes_capacity = 2;
        }
        if (!changed_allocated) {
            cudaMalloc(&d_changed, sizeof(int32_t));
            changed_allocated = true;
        }
    }
};

__device__ __forceinline__ double atomicMinDouble(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val) return __longlong_as_double(old);
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void init_sssp_kernel(
    double* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t num_vertices,
    int32_t source,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_size,
    uint32_t* __restrict__ bitmap,
    int32_t bitmap_words)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        distances[idx] = (idx == source) ? 0.0 : DBL_MAX;
        predecessors[idx] = -1;
    }
    if (idx < bitmap_words) {
        bitmap[idx] = 0;
    }
    if (idx == 0) {
        frontier[0] = source;
        *frontier_size = 1;
    }
}

__global__ void relax_edges_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ distances,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    uint32_t* __restrict__ in_frontier_bitmap,
    double cutoff)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t u = frontier[warp_id];
    double dist_u = distances[u];

    if (dist_u >= cutoff) return;

    int32_t start = offsets[u];
    int32_t end = offsets[u + 1];

    for (int32_t e = start + lane; e < end; e += 32) {
        if (!((edge_mask[e >> 5] >> (e & 31)) & 1)) continue;

        int32_t v = __ldg(&indices[e]);
        double w = __ldg(&weights[e]);
        double new_dist = dist_u + w;

        if (new_dist >= cutoff) continue;
        if (new_dist >= distances[v]) continue;

        double old_dist = atomicMinDouble(&distances[v], new_dist);

        if (new_dist < old_dist) {
            uint32_t bit = 1u << (v & 31);
            uint32_t old_word = atomicOr(&in_frontier_bitmap[v >> 5], bit);
            if (!(old_word & bit)) {
                int pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = v;
            }
        }
    }
}

__global__ void clear_bitmap_kernel(
    uint32_t* __restrict__ bitmap,
    const int32_t* __restrict__ vertices,
    int32_t count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        int32_t v = vertices[tid];
        uint32_t bit = 1u << (v & 31);
        atomicAnd(&bitmap[v >> 5], ~bit);
    }
}

__global__ void fix_predecessors_positive_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t num_vertices)
{
    int32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;

    double dist_u = distances[u];
    if (dist_u >= DBL_MAX * 0.5) return;

    int32_t estart = offsets[u];
    int32_t eend = offsets[u + 1];

    for (int32_t e = estart; e < eend; e++) {
        if (!((edge_mask[e >> 5] >> (e & 31)) & 1)) continue;
        int32_t v = indices[e];
        double w = weights[e];
        if (dist_u + w == distances[v] && dist_u < distances[v]) {
            predecessors[v] = u;
        }
    }
}

__global__ void fix_predecessors_zero_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t* __restrict__ changed,
    int32_t num_vertices,
    int32_t source)
{
    int32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;

    if (u != source && predecessors[u] == -1) return;

    double dist_u = distances[u];
    if (dist_u >= DBL_MAX * 0.5) return;

    int32_t estart = offsets[u];
    int32_t eend = offsets[u + 1];

    for (int32_t e = estart; e < eend; e++) {
        if (!((edge_mask[e >> 5] >> (e & 31)) & 1)) continue;
        int32_t v = indices[e];
        if (v == source) continue;
        double w = weights[e];
        if (dist_u + w == distances[v] && distances[v] == dist_u && predecessors[v] == -1) {
            predecessors[v] = u;
            *changed = 1;
        }
    }
}

}  

void sssp_seg_mask(const graph32_t& graph,
                   const double* edge_weights,
                   int32_t source,
                   double* distances,
                   int32_t* predecessors,
                   double cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cache.ensure(num_vertices);

    int32_t* d_frontier1 = cache.frontier1;
    int32_t* d_frontier2 = cache.frontier2;
    uint32_t* d_bitmap = cache.bitmap;
    int32_t* d_size_a = cache.sizes;
    int32_t* d_size_b = cache.sizes + 1;
    int32_t bitmap_words = (num_vertices + 31) / 32;

    {
        int block = 256;
        int n = num_vertices > bitmap_words ? num_vertices : bitmap_words;
        int grid = (n + block - 1) / block;
        init_sssp_kernel<<<grid, block>>>(distances, predecessors, num_vertices, source,
                                           d_frontier1, d_size_a, d_bitmap, bitmap_words);
    }

    int32_t frontier_size = 1;
    int32_t* d_cur = d_frontier1;
    int32_t* d_next = d_frontier2;
    int32_t* d_cur_sz = d_size_a;
    int32_t* d_next_sz = d_size_b;

    while (frontier_size > 0) {
        cudaMemsetAsync(d_next_sz, 0, sizeof(int32_t));

        {
            int tpb = 256;
            int64_t total = (int64_t)frontier_size * 32;
            int blocks = (int)((total + tpb - 1) / tpb);
            relax_edges_warp_kernel<<<blocks, tpb>>>(
                d_offsets, d_indices, edge_weights, d_edge_mask, distances,
                d_cur, frontier_size,
                d_next, d_next_sz,
                d_bitmap, cutoff);
        }

        cudaMemcpy(cache.h_frontier_count, d_next_sz, sizeof(int32_t), cudaMemcpyDeviceToHost);
        frontier_size = *cache.h_frontier_count;

        if (frontier_size > 0) {
            int block = 256;
            int grid = (frontier_size + block - 1) / block;
            clear_bitmap_kernel<<<grid, block>>>(d_bitmap, d_next, frontier_size);
        }

        int32_t* tmp_f = d_cur; d_cur = d_next; d_next = tmp_f;
        int32_t* tmp_s = d_cur_sz; d_cur_sz = d_next_sz; d_next_sz = tmp_s;
    }

    
    {
        int block = 256;
        int64_t grid64 = ((int64_t)num_vertices + block - 1) / block;
        int grid = (int)grid64;
        fix_predecessors_positive_kernel<<<grid, block>>>(
            d_offsets, d_indices, edge_weights, d_edge_mask,
            distances, predecessors, num_vertices);
    }

    
    {
        int block = 256;
        int64_t grid64 = ((int64_t)num_vertices + block - 1) / block;
        int grid = (int)grid64;
        for (int iter = 0; iter < num_vertices; iter++) {
            cudaMemset(cache.d_changed, 0, sizeof(int32_t));
            fix_predecessors_zero_kernel<<<grid, block>>>(
                d_offsets, d_indices, edge_weights, d_edge_mask,
                distances, predecessors, cache.d_changed, num_vertices, source);
            cudaMemcpy(cache.h_frontier_count, cache.d_changed, sizeof(int32_t),
                        cudaMemcpyDeviceToHost);
            if (*cache.h_frontier_count == 0) break;
        }
    }

    cudaDeviceSynchronize();
}

}  
