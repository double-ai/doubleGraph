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
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>

namespace aai {

namespace {

struct Cache : Cacheable {
    
    int32_t* h_counter = nullptr;

    
    void* dist_pred = nullptr;
    int32_t* frontier1 = nullptr;
    int32_t* frontier2 = nullptr;
    uint32_t* bitmap = nullptr;
    int32_t* counter = nullptr;

    int32_t capacity = 0;

    Cache() {
        cudaMallocHost(&h_counter, sizeof(int32_t));
        cudaMalloc(&counter, sizeof(int32_t));
    }

    void ensure(int32_t num_vertices) {
        if (capacity < num_vertices) {
            if (dist_pred) cudaFree(dist_pred);
            if (frontier1) cudaFree(frontier1);
            if (frontier2) cudaFree(frontier2);
            if (bitmap) cudaFree(bitmap);

            cudaMalloc(&dist_pred, (size_t)num_vertices * sizeof(int64_t));
            cudaMalloc(&frontier1, (size_t)num_vertices * sizeof(int32_t));
            cudaMalloc(&frontier2, (size_t)num_vertices * sizeof(int32_t));
            int32_t bitmap_words = (num_vertices + 31) / 32;
            cudaMalloc(&bitmap, (size_t)bitmap_words * sizeof(uint32_t));

            capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (h_counter) cudaFreeHost(h_counter);
        if (dist_pred) cudaFree(dist_pred);
        if (frontier1) cudaFree(frontier1);
        if (frontier2) cudaFree(frontier2);
        if (bitmap) cudaFree(bitmap);
        if (counter) cudaFree(counter);
    }
};



struct __align__(8) DistPred {
    float dist;
    int32_t pred;
};

__device__ __forceinline__ bool is_edge_active(const uint32_t* __restrict__ edge_mask, int32_t edge_idx) {
    return (edge_mask[edge_idx >> 5] >> (edge_idx & 31)) & 1;
}

__device__ __forceinline__ bool atomicRelax(DistPred* dp_ptr, float new_dist, int32_t new_pred) {
    unsigned long long* addr = reinterpret_cast<unsigned long long*>(dp_ptr);
    union { unsigned long long ull; DistPred dp; } new_val, old_val;
    new_val.dp.dist = new_dist;
    new_val.dp.pred = new_pred;
    old_val.ull = *addr;
    while (true) {
        if (old_val.dp.dist <= new_dist) return false;
        unsigned long long assumed = old_val.ull;
        old_val.ull = atomicCAS(addr, assumed, new_val.ull);
        if (old_val.ull == assumed) return true;
    }
}

__global__ void init_sssp_kernel(
    DistPred* __restrict__ dist_pred, int32_t num_vertices, int32_t source
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        DistPred dp;
        dp.dist = (idx == source) ? 0.0f : FLT_MAX;
        dp.pred = -1;
        dist_pred[idx] = dp;
    }
}




constexpr int BLOCK_SIZE = 256;
constexpr int VERTS_PER_BLOCK = 32;

__global__ void relax_balanced_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    DistPred* dist_pred,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_count,
    uint32_t* next_bitmap,
    float cutoff
) {
    __shared__ int32_t s_vertex[VERTS_PER_BLOCK];
    __shared__ float s_dist[VERTS_PER_BLOCK];
    __shared__ int32_t s_edge_start[VERTS_PER_BLOCK];
    __shared__ int32_t s_prefix[VERTS_PER_BLOCK + 1]; 

    int32_t block_start = blockIdx.x * VERTS_PER_BLOCK;
    int32_t block_end = block_start + VERTS_PER_BLOCK;
    if (block_end > frontier_size) block_end = frontier_size;
    int32_t num_verts = block_end - block_start;

    if (num_verts <= 0) return;

    
    if (threadIdx.x < num_verts) {
        int32_t u = __ldg(&frontier[block_start + threadIdx.x]);
        s_vertex[threadIdx.x] = u;
        int32_t start = __ldg(&offsets[u]);
        int32_t end = __ldg(&offsets[u + 1]);
        s_edge_start[threadIdx.x] = start;
        s_prefix[threadIdx.x] = end - start; 
        s_dist[threadIdx.x] = dist_pred[u].dist;
    } else if (threadIdx.x < VERTS_PER_BLOCK) {
        s_prefix[threadIdx.x] = 0;
    }
    __syncthreads();

    
    if (threadIdx.x == 0) {
        int32_t sum = 0;
        for (int i = 0; i < num_verts; i++) {
            int32_t deg = s_prefix[i];
            s_prefix[i] = sum;
            sum += deg;
        }
        s_prefix[num_verts] = sum;
    }
    __syncthreads();

    int32_t total_edges = s_prefix[num_verts];
    if (total_edges == 0) return;

    
    for (int32_t i = threadIdx.x; i < total_edges; i += BLOCK_SIZE) {
        
        int32_t lo = 0, hi = num_verts - 1;
        while (lo < hi) {
            int32_t mid = (lo + hi + 1) >> 1;
            if (s_prefix[mid] <= i) lo = mid;
            else hi = mid - 1;
        }

        int32_t vertex_idx = lo;
        int32_t u = s_vertex[vertex_idx];
        float dist_u = s_dist[vertex_idx];

        if (dist_u >= cutoff) continue;

        int32_t edge_offset = i - s_prefix[vertex_idx];
        int32_t e = s_edge_start[vertex_idx] + edge_offset;

        if (!is_edge_active(edge_mask, e)) continue;

        int32_t v = __ldg(&indices[e]);
        float w = __ldg(&weights[e]);
        float new_dist = dist_u + w;

        if (new_dist >= cutoff) continue;

        if (atomicRelax(&dist_pred[v], new_dist, u)) {
            uint32_t bit = 1u << (v & 31);
            uint32_t old_word = atomicOr(&next_bitmap[v >> 5], bit);
            if (!(old_word & bit)) {
                int32_t pos = atomicAdd(next_frontier_count, 1);
                next_frontier[pos] = v;
            }
        }
    }
}


__global__ void relax_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    DistPred* dist_pred,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_count,
    uint32_t* next_bitmap,
    float cutoff
) {
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;
    if (warp_id >= frontier_size) return;

    int32_t u = frontier[warp_id];
    float dist_u = dist_pred[u].dist;
    if (dist_u >= cutoff) return;

    int32_t start = __ldg(&offsets[u]);
    int32_t end = __ldg(&offsets[u + 1]);
    int32_t degree = end - start;

    for (int32_t i = lane; i < degree; i += 32) {
        int32_t e = start + i;
        if (!is_edge_active(edge_mask, e)) continue;

        int32_t v = __ldg(&indices[e]);
        float w = __ldg(&weights[e]);
        float new_dist = dist_u + w;
        if (new_dist >= cutoff) continue;

        if (atomicRelax(&dist_pred[v], new_dist, u)) {
            uint32_t bit = 1u << (v & 31);
            uint32_t old_word = atomicOr(&next_bitmap[v >> 5], bit);
            if (!(old_word & bit)) {
                int32_t pos = atomicAdd(next_frontier_count, 1);
                next_frontier[pos] = v;
            }
        }
    }
}


__global__ void relax_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    DistPred* dist_pred,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_count,
    uint32_t* next_bitmap,
    float cutoff
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int32_t u = frontier[tid];
    float dist_u = dist_pred[u].dist;
    if (dist_u >= cutoff) return;

    int32_t start = __ldg(&offsets[u]);
    int32_t end = __ldg(&offsets[u + 1]);

    for (int32_t e = start; e < end; e++) {
        if (!is_edge_active(edge_mask, e)) continue;
        int32_t v = __ldg(&indices[e]);
        float w = __ldg(&weights[e]);
        float new_dist = dist_u + w;
        if (new_dist >= cutoff) continue;

        if (atomicRelax(&dist_pred[v], new_dist, u)) {
            uint32_t bit = 1u << (v & 31);
            uint32_t old_word = atomicOr(&next_bitmap[v >> 5], bit);
            if (!(old_word & bit)) {
                int32_t pos = atomicAdd(next_frontier_count, 1);
                next_frontier[pos] = v;
            }
        }
    }
}

__global__ void extract_results_kernel(
    const DistPred* __restrict__ dist_pred,
    float* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t num_vertices
) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        DistPred dp = dist_pred[idx];
        distances[idx] = dp.dist;
        predecessors[idx] = dp.pred;
    }
}

__global__ void clear_frontier_bitmap_kernel(
    uint32_t* bitmap, const int32_t* __restrict__ frontier, int32_t frontier_size
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < frontier_size) {
        int32_t v = frontier[tid];
        atomicAnd(&bitmap[v >> 5], ~(1u << (v & 31)));
    }
}

}  

void sssp_mask(const graph32_t& graph,
               const float* edge_weights,
               int32_t source,
               float* distances,
               int32_t* predecessors,
               float cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    if (std::isinf(cutoff)) cutoff = std::numeric_limits<float>::max();

    cache.ensure(num_vertices);

    DistPred* d_dist_pred = static_cast<DistPred*>(cache.dist_pred);
    int32_t* d_frontier1 = cache.frontier1;
    int32_t* d_frontier2 = cache.frontier2;
    uint32_t* d_bitmap = cache.bitmap;
    int32_t* d_counter = cache.counter;
    int32_t* h_counter = cache.h_counter;

    int32_t bitmap_words = (num_vertices + 31) / 32;

    init_sssp_kernel<<<(num_vertices + 255) / 256, 256>>>(d_dist_pred, num_vertices, source);
    cudaMemcpy(d_frontier1, &source, sizeof(int32_t), cudaMemcpyHostToDevice);
    int32_t frontier_size = 1;

    int32_t* current_frontier = d_frontier1;
    int32_t* next_frontier = d_frontier2;

    cudaMemset(d_bitmap, 0, bitmap_words * sizeof(uint32_t));

    while (frontier_size > 0) {
        cudaMemset(d_counter, 0, sizeof(int32_t));

        if (frontier_size < 32) {
            
            relax_thread_kernel<<<(frontier_size + 255) / 256, 256>>>(
                d_offsets, d_indices, edge_weights, d_edge_mask,
                d_dist_pred,
                current_frontier, frontier_size,
                next_frontier, d_counter, d_bitmap, cutoff);
        } else if (frontier_size < 256) {
            
            int grid = (frontier_size * 32 + 255) / 256;
            relax_warp_kernel<<<grid, 256>>>(
                d_offsets, d_indices, edge_weights, d_edge_mask,
                d_dist_pred,
                current_frontier, frontier_size,
                next_frontier, d_counter, d_bitmap, cutoff);
        } else {
            
            int grid = (frontier_size + VERTS_PER_BLOCK - 1) / VERTS_PER_BLOCK;
            relax_balanced_kernel<<<grid, BLOCK_SIZE>>>(
                d_offsets, d_indices, edge_weights, d_edge_mask,
                d_dist_pred,
                current_frontier, frontier_size,
                next_frontier, d_counter, d_bitmap, cutoff);
        }

        cudaMemcpy(h_counter, d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
        frontier_size = *h_counter;

        if (frontier_size > 0) {
            if (frontier_size < bitmap_words) {
                clear_frontier_bitmap_kernel<<<(frontier_size + 255) / 256, 256>>>(
                    d_bitmap, next_frontier, frontier_size);
            } else {
                cudaMemset(d_bitmap, 0, bitmap_words * sizeof(uint32_t));
            }
        }

        std::swap(current_frontier, next_frontier);
    }

    extract_results_kernel<<<(num_vertices + 255) / 256, 256>>>(
        d_dist_pred, distances, predecessors, num_vertices);
}

}  
