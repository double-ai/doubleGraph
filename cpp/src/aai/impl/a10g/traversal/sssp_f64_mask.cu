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

namespace aai {

namespace {

#define UNREACHABLE_DIST 1.7976931348623157e+308

struct Cache : Cacheable {
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    uint32_t* bitmap = nullptr;
    int32_t* counter = nullptr;
    int32_t* h_counter_pinned = nullptr;
    int32_t* d_changed = nullptr;
    int32_t capacity = 0;

    Cache() {
        cudaMalloc(&counter, sizeof(int32_t));
        cudaMalloc(&d_changed, sizeof(int32_t));
        cudaHostAlloc(&h_counter_pinned, sizeof(int32_t), cudaHostAllocDefault);
    }

    void ensure(int32_t num_vertices) {
        if (capacity < num_vertices) {
            if (frontier_a) cudaFree(frontier_a);
            if (frontier_b) cudaFree(frontier_b);
            if (bitmap) cudaFree(bitmap);
            cudaMalloc(&frontier_a, num_vertices * sizeof(int32_t));
            cudaMalloc(&frontier_b, num_vertices * sizeof(int32_t));
            int32_t bitmap_words = (num_vertices + 31) / 32;
            cudaMalloc(&bitmap, bitmap_words * sizeof(uint32_t));
            capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (bitmap) cudaFree(bitmap);
        if (counter) cudaFree(counter);
        if (d_changed) cudaFree(d_changed);
        if (h_counter_pinned) cudaFreeHost(h_counter_pinned);
    }
};

__device__ __forceinline__ double atomicMinDouble(double* addr, double val) {
    unsigned long long int* addr_as_ull = (unsigned long long int*)addr;
    unsigned long long int old = *addr_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val) return __longlong_as_double(assumed);
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void init_sssp_kernel(
    double* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t num_vertices,
    int32_t source
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        distances[idx] = (idx == source) ? 0.0 : UNREACHABLE_DIST;
        predecessors[idx] = -1;
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
    uint32_t* __restrict__ next_bitmap,
    double cutoff
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t u = frontier[warp_id];
    
    double dist_u;
    if (lane == 0) dist_u = distances[u];
    dist_u = __shfl_sync(0xFFFFFFFF, dist_u, 0);

    if (dist_u >= cutoff) return;

    int32_t start, end;
    if (lane == 0) {
        start = offsets[u];
        end = offsets[u + 1];
    }
    start = __shfl_sync(0xFFFFFFFF, start, 0);
    end = __shfl_sync(0xFFFFFFFF, end, 0);

    for (int32_t e = start + lane; e < end; e += 32) {
        uint32_t mask_word = edge_mask[e >> 5];
        if (!((mask_word >> (e & 31)) & 1)) continue;

        int32_t v = indices[e];
        double w = weights[e];
        double new_dist = dist_u + w;

        if (new_dist >= cutoff) continue;

        
        
        double cur_dist_v = distances[v];
        if (new_dist >= cur_dist_v) continue;

        double old_dist = atomicMinDouble(&distances[v], new_dist);
        if (new_dist < old_dist) {
            uint32_t bit = 1u << (v & 31);
            uint32_t old_word = atomicOr(&next_bitmap[v >> 5], bit);
            if (!(old_word & bit)) {
                int pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = v;
            }
        }
    }
}


__global__ void relax_edges_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ distances,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    uint32_t* __restrict__ next_bitmap,
    double cutoff
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int32_t u = frontier[tid];
    double dist_u = distances[u];
    if (dist_u >= cutoff) return;

    int32_t start = offsets[u];
    int32_t end = offsets[u + 1];

    for (int32_t e = start; e < end; e++) {
        uint32_t mask_word = edge_mask[e >> 5];
        if (!((mask_word >> (e & 31)) & 1)) continue;

        int32_t v = indices[e];
        double new_dist = dist_u + weights[e];
        if (new_dist >= cutoff) continue;

        
        if (new_dist >= distances[v]) continue;

        double old_dist = atomicMinDouble(&distances[v], new_dist);
        if (new_dist < old_dist) {
            uint32_t bit = 1u << (v & 31);
            uint32_t old_word = atomicOr(&next_bitmap[v >> 5], bit);
            if (!(old_word & bit)) {
                int pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = v;
            }
        }
    }
}



__global__ void compute_predecessors_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t num_vertices,
    int32_t source
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_vertices) return;
    int32_t u = warp_id;
    double dist_u = distances[u];
    if (dist_u >= UNREACHABLE_DIST) return;
    int32_t start = offsets[u];
    int32_t end = offsets[u + 1];
    for (int32_t e = start + lane; e < end; e += 32) {
        uint32_t mask_word = edge_mask[e >> 5];
        if (!((mask_word >> (e & 31)) & 1)) continue;
        int32_t v = indices[e];
        if (v == source) continue;
        double dist_v = distances[v];
        if (dist_v >= UNREACHABLE_DIST) continue;
        double expected = dist_u + weights[e];
        if (expected == dist_v && dist_u < dist_v) {
            predecessors[v] = u;
        }
    }
}


__global__ void zerow_pred_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t* __restrict__ changed,
    int32_t source, int32_t N
) {
    int32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N) return;
    double d_u = distances[u];
    if (d_u >= UNREACHABLE_DIST) return;
    if (u != source && predecessors[u] == -1) return;  
    int32_t es = offsets[u], ee = offsets[u + 1];
    for (int32_t e = es; e < ee; e++) {
        uint32_t mask_word = edge_mask[e >> 5];
        if (!((mask_word >> (e & 31)) & 1)) continue;
        int32_t v = indices[e];
        if (v == source) continue;
        double w = weights[e];
        if (d_u + w != distances[v]) continue;   
        if (d_u != distances[v]) continue;        
        if (predecessors[v] == -1) {
            predecessors[v] = u;
            *changed = 1;
        }
    }
}

}  

void sssp_mask(const graph32_t& graph,
               const double* edge_weights,
               int32_t source,
               double* distances,
               int32_t* predecessors,
               double cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cache.ensure(num_vertices);

    int32_t* d_fa = cache.frontier_a;
    int32_t* d_fb = cache.frontier_b;
    uint32_t* d_bitmap = cache.bitmap;
    int32_t* d_counter = cache.counter;
    int32_t bitmap_words = (num_vertices + 31) / 32;

    cudaStream_t stream = 0;

    init_sssp_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
        distances, predecessors, num_vertices, source);

    cudaMemcpyAsync(d_fa, &source, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    int32_t h_frontier_size = 1;

    int32_t* cur = d_fa;
    int32_t* nxt = d_fb;

    while (h_frontier_size > 0) {
        cudaMemsetAsync(d_bitmap, 0, bitmap_words * sizeof(uint32_t), stream);
        cudaMemsetAsync(d_counter, 0, sizeof(int32_t), stream);

        
        if (h_frontier_size >= 16) {
            int block = 256;
            int grid = (int)(((int64_t)h_frontier_size * 32 + block - 1) / block);
            relax_edges_warp_kernel<<<grid, block, 0, stream>>>(
                d_offsets, d_indices, edge_weights, d_edge_mask,
                distances, cur, h_frontier_size,
                nxt, d_counter, d_bitmap, cutoff);
        } else {
            int block = 256;
            int grid = (h_frontier_size + block - 1) / block;
            relax_edges_thread_kernel<<<grid, block, 0, stream>>>(
                d_offsets, d_indices, edge_weights, d_edge_mask,
                distances, cur, h_frontier_size,
                nxt, d_counter, d_bitmap, cutoff);
        }

        cudaMemcpyAsync(cache.h_counter_pinned, d_counter, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        h_frontier_size = *cache.h_counter_pinned;

        std::swap(cur, nxt);
    }

    
    {
        int grid = (int)(((int64_t)num_vertices * 32 + 255) / 256);
        compute_predecessors_kernel<<<grid, 256, 0, stream>>>(
            d_offsets, d_indices, edge_weights, d_edge_mask,
            distances, predecessors, num_vertices, source);
    }

    
    int32_t* d_ch = cache.d_changed;
    int32_t h_ch = 1;
    for (int ziter = 0; ziter < num_vertices && h_ch; ziter++) {
        cudaMemsetAsync(d_ch, 0, sizeof(int32_t), stream);
        int grid = (num_vertices + 255) / 256;
        zerow_pred_kernel<<<grid, 256, 0, stream>>>(
            d_offsets, d_indices, edge_weights, d_edge_mask,
            distances, predecessors, d_ch, source, num_vertices);
        cudaMemcpy(&h_ch, d_ch, sizeof(int32_t), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
}

}  
