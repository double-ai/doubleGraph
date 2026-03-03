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
#include <float.h>

namespace aai {

namespace {

struct Cache : Cacheable {
    unsigned long long* dist_pred = nullptr;
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* frontier_size_d = nullptr;
    int32_t* in_frontier = nullptr;

    int64_t dist_pred_capacity = 0;
    int64_t frontier_a_capacity = 0;
    int64_t frontier_b_capacity = 0;
    int64_t frontier_size_d_capacity = 0;
    int64_t in_frontier_capacity = 0;

    void ensure(int64_t num_vertices) {
        if (dist_pred_capacity < num_vertices) {
            if (dist_pred) cudaFree(dist_pred);
            cudaMalloc(&dist_pred, num_vertices * sizeof(unsigned long long));
            dist_pred_capacity = num_vertices;
        }
        if (frontier_a_capacity < num_vertices) {
            if (frontier_a) cudaFree(frontier_a);
            cudaMalloc(&frontier_a, num_vertices * sizeof(int32_t));
            frontier_a_capacity = num_vertices;
        }
        if (frontier_b_capacity < num_vertices) {
            if (frontier_b) cudaFree(frontier_b);
            cudaMalloc(&frontier_b, num_vertices * sizeof(int32_t));
            frontier_b_capacity = num_vertices;
        }
        if (frontier_size_d_capacity < 1) {
            if (frontier_size_d) cudaFree(frontier_size_d);
            cudaMalloc(&frontier_size_d, sizeof(int32_t));
            frontier_size_d_capacity = 1;
        }
        if (in_frontier_capacity < num_vertices) {
            if (in_frontier) cudaFree(in_frontier);
            cudaMalloc(&in_frontier, num_vertices * sizeof(int32_t));
            in_frontier_capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (dist_pred) cudaFree(dist_pred);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (frontier_size_d) cudaFree(frontier_size_d);
        if (in_frontier) cudaFree(in_frontier);
    }
};

__device__ __forceinline__ unsigned long long pack_dp(float dist, int pred) {
    return ((unsigned long long)__float_as_uint(dist) << 32) | (unsigned int)pred;
}

__device__ __forceinline__ float unpack_dist(unsigned long long packed) {
    return __uint_as_float((unsigned int)(packed >> 32));
}

__device__ __forceinline__ int unpack_pred(unsigned long long packed) {
    return (int)(unsigned int)(packed & 0xFFFFFFFFULL);
}

__global__ void sssp_init(
    unsigned long long* __restrict__ dist_pred,
    int num_vertices,
    int source
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        float d = (idx == source) ? 0.0f : FLT_MAX;
        dist_pred[idx] = pack_dp(d, -1);
    }
}

__global__ void sssp_relax(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ weights,
    unsigned long long* __restrict__ dist_pred,
    const int* __restrict__ frontier,
    int frontier_size,
    int* __restrict__ next_frontier,
    int* __restrict__ next_frontier_size,
    int* __restrict__ in_next_frontier,
    float cutoff
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int u = frontier[warp_id];
    float u_dist = unpack_dist(dist_pred[u]);

    if (u_dist >= cutoff) return;

    int start = offsets[u];
    int end = offsets[u + 1];

    for (int e = start + lane; e < end; e += 32) {
        int v = indices[e];
        float w = weights[e];
        float new_dist = u_dist + w;

        if (new_dist < cutoff) {
            unsigned long long cur = dist_pred[v];
            if (new_dist < unpack_dist(cur)) {
                unsigned long long new_packed = pack_dp(new_dist, u);
                unsigned long long old_packed = atomicMin(&dist_pred[v], new_packed);

                if (new_packed < old_packed) {
                    if (atomicExch(&in_next_frontier[v], 1) == 0) {
                        int pos = atomicAdd(next_frontier_size, 1);
                        next_frontier[pos] = v;
                    }
                }
            }
        }
    }
}

__global__ void sssp_clear_flags(
    int* __restrict__ in_next_frontier,
    const int* __restrict__ frontier,
    int frontier_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < frontier_size) {
        in_next_frontier[frontier[tid]] = 0;
    }
}

__global__ void sssp_extract(
    const unsigned long long* __restrict__ dist_pred,
    float* __restrict__ distances,
    int* __restrict__ predecessors,
    int num_vertices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        unsigned long long packed = dist_pred[idx];
        distances[idx] = unpack_dist(packed);
        predecessors[idx] = unpack_pred(packed);
    }
}

__global__ void fix_predecessors_positive_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ dist,
    int* __restrict__ pred,
    int num_vertices,
    int source
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;
    float dist_u = dist[u];
    if (dist_u >= FLT_MAX) return;
    int start = offsets[u];
    int end = offsets[u + 1];
    for (int e = start; e < end; e++) {
        int v = indices[e];
        if (v == source) continue;
        float w = weights[e];
        if (dist_u + w == dist[v] && dist_u < dist[v]) {
            pred[v] = u;
        }
    }
}

__global__ void fix_predecessors_zero_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ dist,
    int* __restrict__ pred,
    int num_vertices,
    int source,
    int* __restrict__ changed
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;
    if (u != source && pred[u] == -1) return;
    float dist_u = dist[u];
    if (dist_u >= FLT_MAX) return;
    int start = offsets[u];
    int end = offsets[u + 1];
    for (int e = start; e < end; e++) {
        int v = indices[e];
        if (v == source) continue;
        float w = weights[e];
        if (dist_u + w == dist[v] && dist[v] == dist_u && pred[v] == -1) {
            pred[v] = u;
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

    int32_t num_vertices = graph.number_of_vertices;

    cache.ensure(num_vertices);

    if (std::isinf(cutoff)) {
        cutoff = std::numeric_limits<float>::max();
    }

    const int BLOCK_SIZE = 256;

    int grid_init = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sssp_init<<<grid_init, BLOCK_SIZE>>>(cache.dist_pred, num_vertices, source);

    cudaMemset(cache.in_frontier, 0, num_vertices * sizeof(int32_t));

    cudaMemcpy(cache.frontier_a, &source, sizeof(int32_t), cudaMemcpyHostToDevice);
    int h_frontier_size = 1;

    int32_t* current_frontier = cache.frontier_a;
    int32_t* next_frontier = cache.frontier_b;

    while (h_frontier_size > 0) {
        cudaMemset(cache.frontier_size_d, 0, sizeof(int32_t));

        int64_t total_threads = (int64_t)h_frontier_size * 32;
        int grid_relax = (int)((total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
        sssp_relax<<<grid_relax, BLOCK_SIZE>>>(
            graph.offsets, graph.indices, edge_weights, cache.dist_pred,
            current_frontier, h_frontier_size,
            next_frontier, cache.frontier_size_d,
            cache.in_frontier, cutoff
        );

        cudaMemcpy(&h_frontier_size, cache.frontier_size_d, sizeof(int32_t), cudaMemcpyDeviceToHost);

        if (h_frontier_size > 0) {
            int grid_clear = (h_frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            sssp_clear_flags<<<grid_clear, BLOCK_SIZE>>>(
                cache.in_frontier, next_frontier, h_frontier_size
            );
        }

        int32_t* temp = current_frontier;
        current_frontier = next_frontier;
        next_frontier = temp;
    }

    sssp_extract<<<grid_init, BLOCK_SIZE>>>(cache.dist_pred, distances, predecessors, num_vertices);

    cudaMemsetAsync(predecessors, 0xFF, num_vertices * sizeof(int32_t));

    fix_predecessors_positive_kernel<<<grid_init, BLOCK_SIZE>>>(
        graph.offsets, graph.indices, edge_weights, distances, predecessors, num_vertices, source
    );

    int32_t h_changed = 1;
    for (int iter = 0; iter < num_vertices && h_changed; iter++) {
        h_changed = 0;
        cudaMemsetAsync(cache.frontier_size_d, 0, sizeof(int32_t));
        fix_predecessors_zero_kernel<<<grid_init, BLOCK_SIZE>>>(
            graph.offsets, graph.indices, edge_weights, distances, predecessors,
            num_vertices, source, cache.frontier_size_d
        );
        cudaMemcpy(&h_changed, cache.frontier_size_d, sizeof(int32_t), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
}

}  
