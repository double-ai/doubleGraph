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
#include <cmath>
#include <limits>
#include <cstring>

namespace aai {

namespace {

#define UNREACHABLE_DIST 3.4028234663852886e+38f

struct Cache : Cacheable {
    int32_t* frontier1 = nullptr;
    int32_t* frontier2 = nullptr;
    int32_t* frontier_count = nullptr;
    uint32_t* bitmap = nullptr;

    int64_t frontier1_capacity = 0;
    int64_t frontier2_capacity = 0;
    int64_t frontier_count_capacity = 0;
    int64_t bitmap_capacity = 0;

    void ensure(int32_t n) {
        int32_t bm_words = (n + 31) / 32;

        if (frontier1_capacity < n) {
            if (frontier1) cudaFree(frontier1);
            cudaMalloc(&frontier1, (size_t)n * sizeof(int32_t));
            frontier1_capacity = n;
        }
        if (frontier2_capacity < n) {
            if (frontier2) cudaFree(frontier2);
            cudaMalloc(&frontier2, (size_t)n * sizeof(int32_t));
            frontier2_capacity = n;
        }
        if (frontier_count_capacity < 1) {
            if (frontier_count) cudaFree(frontier_count);
            cudaMalloc(&frontier_count, sizeof(int32_t));
            frontier_count_capacity = 1;
        }
        if (bitmap_capacity < bm_words) {
            if (bitmap) cudaFree(bitmap);
            cudaMalloc(&bitmap, (size_t)bm_words * sizeof(uint32_t));
            bitmap_capacity = bm_words;
        }
    }

    ~Cache() override {
        if (frontier1) cudaFree(frontier1);
        if (frontier2) cudaFree(frontier2);
        if (frontier_count) cudaFree(frontier_count);
        if (bitmap) cudaFree(bitmap);
    }
};

__global__ void init_kernel(float* __restrict__ distances, int32_t* __restrict__ predecessors,
                            int32_t n, int32_t source) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        distances[i] = (i == source) ? 0.0f : UNREACHABLE_DIST;
        predecessors[i] = -1;
    }
}

__global__ void relax_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    float* __restrict__ distances,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count,
    uint32_t* __restrict__ bitmap,
    unsigned int cutoff_uint
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t u = frontier[warp_id];
    float dist_u = distances[u];
    if (__float_as_uint(dist_u) >= cutoff_uint) return;

    int32_t start = __ldg(&offsets[u]);
    int32_t end = __ldg(&offsets[u + 1]);

    for (int32_t e = start + lane; e < end; e += 32) {
        int32_t v = __ldg(&indices[e]);
        float new_dist = dist_u + __ldg(&weights[e]);
        unsigned int nd_uint = __float_as_uint(new_dist);

        if (nd_uint < cutoff_uint) {
            unsigned int old = atomicMin((unsigned int*)&distances[v], nd_uint);

            if (nd_uint < old) {
                uint32_t wi = (uint32_t)v >> 5;
                uint32_t bit = 1u << (v & 31);
                if (!(atomicOr(&bitmap[wi], bit) & bit)) {
                    next_frontier[atomicAdd(next_count, 1)] = v;
                }
            }
        }
    }
}

__global__ void reset_bitmap_kernel(
    uint32_t* __restrict__ bitmap,
    const int32_t* __restrict__ frontier,
    int32_t n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int32_t v = frontier[i];
        atomicAnd(&bitmap[(uint32_t)v >> 5], ~(1u << (v & 31)));
    }
}

__global__ void fix_predecessors_positive(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t n,
    int32_t source,
    unsigned int unreachable_uint
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;

    for (int u = warp_id; u < n; u += total_warps) {
        float dist_u = distances[u];
        if (__float_as_uint(dist_u) >= unreachable_uint) continue;

        int32_t start = offsets[u];
        int32_t end = offsets[u + 1];

        for (int32_t e = start + lane; e < end; e += 32) {
            int32_t v = indices[e];
            if (v == source) continue;
            float w = weights[e];
            float dist_v = distances[v];
            if (__float_as_uint(dist_v) >= unreachable_uint) continue;

            if (__float_as_uint(dist_u + w) == __float_as_uint(dist_v) && dist_u < dist_v) {
                predecessors[v] = u;
            }
        }
    }
}

__global__ void fix_predecessors_zero(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t n,
    int32_t source,
    unsigned int unreachable_uint,
    int32_t* __restrict__ changed
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;
    if (u != source && predecessors[u] == -1) return;
    float dist_u = distances[u];
    if (__float_as_uint(dist_u) >= unreachable_uint) return;

    int32_t start = offsets[u];
    int32_t end = offsets[u + 1];

    for (int32_t e = start; e < end; e++) {
        int32_t v = indices[e];
        if (v == source) continue;
        float w = weights[e];

        if (dist_u + w == distances[v] && distances[v] == dist_u && predecessors[v] == -1) {
            predecessors[v] = u;
            *changed = 1;
        }
    }
}

}  

void sssp(const graph32_t& graph,
          const float* edge_weights,
          int32_t source,
          float* distances,
          int32_t* predecessors,
          float cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t n = graph.number_of_vertices;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    cache.ensure(n);

    if (std::isinf(cutoff)) cutoff = std::numeric_limits<float>::max();
    unsigned int cutoff_uint;
    std::memcpy(&cutoff_uint, &cutoff, sizeof(float));

    float unreachable = std::numeric_limits<float>::max();
    unsigned int unreachable_uint;
    std::memcpy(&unreachable_uint, &unreachable, sizeof(float));

    cudaStream_t stream = 0;
    int32_t bm_words = (n + 31) / 32;

    
    init_kernel<<<(n + 255) / 256, 256, 0, stream>>>(distances, predecessors, n, source);
    cudaMemsetAsync(cache.bitmap, 0, bm_words * sizeof(uint32_t), stream);
    cudaMemcpyAsync(cache.frontier1, &source, sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    int32_t* cur = cache.frontier1;
    int32_t* nxt = cache.frontier2;
    int32_t cur_size = 1;

    
    for (int iter = 0; cur_size > 0 && iter < n; iter++) {
        cudaMemsetAsync(cache.frontier_count, 0, sizeof(int32_t), stream);

        if (cur_size > 0) {
            int B = 256;
            int G = (cur_size + 7) / 8;  
            relax_kernel<<<G, B, 0, stream>>>(
                offsets, indices, edge_weights, distances,
                cur, cur_size, nxt, cache.frontier_count, cache.bitmap, cutoff_uint);
        }

        cudaMemcpyAsync(&cur_size, cache.frontier_count, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (cur_size > 0) {
            reset_bitmap_kernel<<<(cur_size + 255) / 256, 256, 0, stream>>>(cache.bitmap, nxt, cur_size);
        }

        int32_t* tmp = cur; cur = nxt; nxt = tmp;
    }

    
    {
        int B = 256;
        int G = (n + 7) / 8;
        if (G > 2048) G = 2048;
        fix_predecessors_positive<<<G, B, 0, stream>>>(
            offsets, indices, edge_weights, distances, predecessors, n, source, unreachable_uint);
    }

    
    {
        int32_t h_changed = 1;
        for (int iter = 0; iter < n && h_changed; iter++) {
            cudaMemsetAsync(cache.frontier_count, 0, sizeof(int32_t), stream);
            int B = 256;
            int G = (n + 255) / 256;
            fix_predecessors_zero<<<G, B, 0, stream>>>(
                offsets, indices, edge_weights, distances, predecessors, n, source, unreachable_uint, cache.frontier_count);
            cudaMemcpyAsync(&h_changed, cache.frontier_count, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
    }
}

}  
