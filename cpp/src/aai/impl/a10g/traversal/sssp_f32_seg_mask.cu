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
#include <math_constants.h>
#include <cstdint>

namespace aai {

namespace {

struct Cache : Cacheable {
    uint64_t* packed = nullptr;
    uint32_t* frontier_a = nullptr;
    uint32_t* frontier_b = nullptr;
    int32_t* d_changed = nullptr;
    size_t alloc_vertices = 0;

    void ensure(int32_t num_vertices) {
        size_t needed = (size_t)num_vertices;
        if (needed <= alloc_vertices) return;

        if (packed) cudaFree(packed);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (d_changed) cudaFree(d_changed);

        
        alloc_vertices = needed + needed / 2;
        size_t bm_words = (alloc_vertices + 31) / 32;

        cudaMalloc(&packed, alloc_vertices * sizeof(uint64_t));
        cudaMalloc(&frontier_a, bm_words * sizeof(uint32_t));
        cudaMalloc(&frontier_b, bm_words * sizeof(uint32_t));
        cudaMalloc(&d_changed, sizeof(int32_t));
    }

    ~Cache() override {
        if (packed) cudaFree(packed);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (d_changed) cudaFree(d_changed);
    }
};

__device__ __forceinline__ uint64_t pack_dp(float dist, int32_t pred) {
    return ((uint64_t)__float_as_uint(dist) << 32) | (uint32_t)pred;
}
__device__ __forceinline__ float unpack_d(uint64_t p) {
    return __uint_as_float((uint32_t)(p >> 32));
}
__device__ __forceinline__ int32_t unpack_p(uint64_t p) {
    return (int32_t)(uint32_t)(p & 0xFFFFFFFF);
}

__global__ void init_sssp(uint64_t* __restrict__ packed, int32_t n, int32_t src) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) packed[tid] = pack_dp((tid == src) ? 0.0f : CUDART_INF_F, -1);
}

__global__ void finalize_sssp(
    const uint64_t* __restrict__ packed, float* __restrict__ dist,
    int32_t* __restrict__ pred, int32_t n, int32_t src
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        uint64_t p = packed[tid];
        float d = unpack_d(p);
        dist[tid] = isinf(d) ? 3.4028235e+38f : d;
        pred[tid] = (tid == src) ? -1 : unpack_p(p);
    }
}

constexpr int BLOCK = 256;
constexpr int WPB = BLOCK / 32;


__device__ __forceinline__ void relax_edge(
    int32_t u, float nd, int32_t v,
    uint64_t* __restrict__ packed,
    uint32_t* __restrict__ nfrontier,
    int32_t* __restrict__ changed,
    float cutoff
) {
    if (nd >= cutoff) return;
    uint64_t np = pack_dp(nd, v);
    
    
    
    
    uint64_t cur = packed[u];
    uint32_t nd_bits = __float_as_uint(nd);
    if (nd_bits >= (uint32_t)(cur >> 32)) return;
    uint64_t old = atomicMin((unsigned long long*)&packed[u], (unsigned long long)np);
    
    
    
    if (nd_bits < (uint32_t)(old >> 32)) {
        atomicOr(&nfrontier[u >> 5], 1u << (u & 31));
        *changed = 1;
    }
}

__global__ __launch_bounds__(BLOCK)
void relax_edges(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ emask,
    uint64_t* __restrict__ packed,
    const uint32_t* __restrict__ frontier,
    uint32_t* __restrict__ nfrontier,
    int32_t* __restrict__ changed,
    int32_t n_high_blocks, int32_t n_mid_blocks,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3,
    float cutoff
) {
    if (blockIdx.x < n_high_blocks) {
        
        int32_t v = seg0 + blockIdx.x;
        if (!((frontier[v >> 5] >> (v & 31)) & 1)) return;
        float dist_v = unpack_d(packed[v]);
        if (dist_v >= cutoff) return;
        int32_t start = offsets[v], end = offsets[v + 1];
        for (int32_t e = start + threadIdx.x; e < end; e += BLOCK) {
            if (!((emask[e >> 5] >> (e & 31)) & 1)) continue;
            float nd = dist_v + __ldg(&weights[e]);
            relax_edge(__ldg(&indices[e]), nd, v, packed, nfrontier, changed, cutoff);
        }
    } else if (blockIdx.x < n_high_blocks + n_mid_blocks) {
        
        int32_t block_in_mid = blockIdx.x - n_high_blocks;
        int32_t warp_in_block = threadIdx.x >> 5;
        int32_t lane = threadIdx.x & 31;
        int32_t v = seg1 + block_in_mid * WPB + warp_in_block;
        if (v >= seg2) return;
        if (!((frontier[v >> 5] >> (v & 31)) & 1)) return;

        
        float dist_v;
        int32_t start, end;
        if (lane == 0) {
            dist_v = unpack_d(packed[v]);
            start = offsets[v];
            end = offsets[v + 1];
        }
        dist_v = __shfl_sync(0xFFFFFFFF, dist_v, 0);
        start = __shfl_sync(0xFFFFFFFF, start, 0);
        end = __shfl_sync(0xFFFFFFFF, end, 0);

        if (dist_v >= cutoff) return;
        for (int32_t e = start + lane; e < end; e += 32) {
            if (!((emask[e >> 5] >> (e & 31)) & 1)) continue;
            float nd = dist_v + __ldg(&weights[e]);
            relax_edge(__ldg(&indices[e]), nd, v, packed, nfrontier, changed, cutoff);
        }
    } else {
        
        int32_t block_in_low = blockIdx.x - n_high_blocks - n_mid_blocks;
        int32_t v = seg2 + block_in_low * BLOCK + threadIdx.x;
        if (v >= seg3) return;

        bool active = (frontier[v >> 5] >> (v & 31)) & 1;
        if (!__any_sync(0xFFFFFFFF, active)) return;
        if (!active) return;

        float dist_v = unpack_d(packed[v]);
        if (dist_v >= cutoff) return;
        int32_t start = offsets[v], end = offsets[v + 1];

        for (int32_t e = start; e < end; e++) {
            if (!((emask[e >> 5] >> (e & 31)) & 1)) continue;
            float nd = dist_v + __ldg(&weights[e]);
            relax_edge(__ldg(&indices[e]), nd, v, packed, nfrontier, changed, cutoff);
        }
    }
}

}  

void sssp_seg_mask(const graph32_t& graph,
                   const float* edge_weights,
                   int32_t source,
                   float* distances,
                   int32_t* predecessors,
                   float cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    const uint32_t* edge_mask = graph.edge_mask;

    
    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0];
    int32_t seg1 = seg[1];
    int32_t seg2 = seg[2];
    int32_t seg3 = seg[3];

    
    cache.ensure(num_vertices);

    int32_t bitmap_words = (num_vertices + 31) / 32;
    int grid_v = (num_vertices + BLOCK - 1) / BLOCK;

    int n_high_blocks = seg1 - seg0;
    int n_mid_blocks = ((seg2 - seg1) + WPB - 1) / WPB;
    int n_low_blocks = ((seg3 - seg2) + BLOCK - 1) / BLOCK;
    int total_blocks = n_high_blocks + n_mid_blocks + n_low_blocks;

    
    init_sssp<<<grid_v, BLOCK>>>(cache.packed, num_vertices, source);
    cudaMemsetAsync(cache.frontier_a, 0, bitmap_words * sizeof(uint32_t));
    cudaMemsetAsync(cache.frontier_b, 0, bitmap_words * sizeof(uint32_t));

    uint32_t src_bit = 1u << (source & 31);
    cudaMemcpyAsync(&cache.frontier_a[source >> 5], &src_bit, sizeof(uint32_t), cudaMemcpyHostToDevice);

    uint32_t* cur = cache.frontier_a;
    uint32_t* nxt = cache.frontier_b;
    int32_t h_changed;

    int max_iter = num_vertices;

    
    const int BATCH = 32;

    for (int iter = 0; iter < max_iter; iter += BATCH) {
        cudaMemsetAsync(cache.d_changed, 0, sizeof(int32_t));

        int batch_end = iter + BATCH;
        if (batch_end > max_iter) batch_end = max_iter;

        for (int it = iter; it < batch_end; it++) {
            if (total_blocks > 0) {
                relax_edges<<<total_blocks, BLOCK>>>(
                    offsets, indices, edge_weights, edge_mask,
                    cache.packed, cur, nxt, cache.d_changed,
                    n_high_blocks, n_mid_blocks,
                    seg0, seg1, seg2, seg3, cutoff
                );
            }
            uint32_t* tmp = cur; cur = nxt; nxt = tmp;
            cudaMemsetAsync(nxt, 0, bitmap_words * sizeof(uint32_t));
        }

        cudaMemcpy(&h_changed, cache.d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (!h_changed) break;
    }

    finalize_sssp<<<grid_v, BLOCK>>>(cache.packed, distances, predecessors, num_vertices, source);
}

}  
