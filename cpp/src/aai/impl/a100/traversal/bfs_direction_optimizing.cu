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
#include <cub/cub.cuh>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* d_frontier1 = nullptr;
    int32_t* d_frontier2 = nullptr;
    uint32_t* d_visited_bmp = nullptr;
    int32_t* d_counter = nullptr;
    int64_t* d_degree_sum = nullptr;
    int32_t* h_counter = nullptr;
    int64_t* h_degree_sum = nullptr;
    size_t frontier_capacity = 0;

    Cache() {
        frontier_capacity = 8000000;
        cudaMalloc(&d_frontier1, frontier_capacity * sizeof(int32_t));
        cudaMalloc(&d_frontier2, frontier_capacity * sizeof(int32_t));
        cudaMalloc(&d_visited_bmp, ((frontier_capacity + 31) / 32) * sizeof(uint32_t));
        cudaMalloc(&d_counter, sizeof(int32_t));
        cudaMalloc(&d_degree_sum, sizeof(int64_t));
        cudaMallocHost(&h_counter, sizeof(int32_t));
        cudaMallocHost(&h_degree_sum, sizeof(int64_t));
    }

    void ensure(size_t num_vertices) {
        if (num_vertices <= frontier_capacity) return;
        if (d_frontier1) cudaFree(d_frontier1);
        if (d_frontier2) cudaFree(d_frontier2);
        if (d_visited_bmp) cudaFree(d_visited_bmp);
        frontier_capacity = num_vertices;
        cudaMalloc(&d_frontier1, frontier_capacity * sizeof(int32_t));
        cudaMalloc(&d_frontier2, frontier_capacity * sizeof(int32_t));
        cudaMalloc(&d_visited_bmp, ((frontier_capacity + 31) / 32) * sizeof(uint32_t));
    }

    ~Cache() override {
        if (d_frontier1) cudaFree(d_frontier1);
        if (d_frontier2) cudaFree(d_frontier2);
        if (d_visited_bmp) cudaFree(d_visited_bmp);
        if (d_counter) cudaFree(d_counter);
        if (d_degree_sum) cudaFree(d_degree_sum);
        if (h_counter) cudaFreeHost(h_counter);
        if (h_degree_sum) cudaFreeHost(h_degree_sum);
    }
};




__device__ __forceinline__ bool bmp_test(const uint32_t* bmp, int v) {
    return (bmp[v >> 5] >> (v & 31)) & 1u;
}

__device__ __forceinline__ void bmp_set_atomic(uint32_t* bmp, int v) {
    atomicOr(&bmp[v >> 5], 1u << (v & 31));
}




__global__ void init_bfs_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bmp,
    int num_vertices,
    int compute_pred
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        distances[idx] = INT32_MAX;
        if (compute_pred) predecessors[idx] = -1;
    }
    int bmp_words = (num_vertices + 31) >> 5;
    for (int i = idx; i < bmp_words; i += blockDim.x * gridDim.x)
        visited_bmp[i] = 0;
}

__global__ void init_sources_kernel(
    const int32_t* __restrict__ sources, int n_sources,
    int32_t* __restrict__ distances, int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bmp,
    int32_t* __restrict__ frontier,
    int compute_pred
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_sources) {
        int s = sources[idx];
        distances[s] = 0;
        if (compute_pred) predecessors[s] = -1;
        bmp_set_atomic(visited_bmp, s);
        frontier[idx] = s;
    }
}




__device__ __forceinline__ void td_process_edge(
    int src, int dst,
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bmp,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int new_dist, int compute_pred
) {
    if (bmp_test(visited_bmp, dst)) return;
    int old = atomicCAS(&distances[dst], INT32_MAX, new_dist);
    if (old == INT32_MAX) {
        if (compute_pred) predecessors[dst] = src;
        bmp_set_atomic(visited_bmp, dst);
        int pos = atomicAdd(next_frontier_size, 1);
        next_frontier[pos] = dst;
    }
}




__global__ void bfs_td_small_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bmp,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int frontier_size,
    int new_dist, int compute_pred
) {
    __shared__ int32_t s_start[64];
    __shared__ int32_t s_end[64];
    __shared__ int32_t s_src[64];
    __shared__ int32_t s_total_edges;

    int fs = frontier_size < 64 ? frontier_size : 64;

    if (threadIdx.x < fs) {
        int v = frontier[threadIdx.x];
        s_src[threadIdx.x] = v;
        s_start[threadIdx.x] = offsets[v];
        s_end[threadIdx.x] = offsets[v + 1];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int i = 0; i < fs; i++)
            total += s_end[i] - s_start[i];
        s_total_edges = total;
    }
    __syncthreads();

    int total_edges = s_total_edges;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int edge_id = tid; edge_id < total_edges; edge_id += stride) {
        int cum = 0;
        int fid = 0;
        for (int i = 0; i < fs; i++) {
            int deg = s_end[i] - s_start[i];
            if (cum + deg > edge_id) { fid = i; break; }
            cum += deg;
        }
        int src = s_src[fid];
        int e = s_start[fid] + (edge_id - cum);
        int dst = indices[e];
        td_process_edge(src, dst, offsets, distances, predecessors, visited_bmp,
                        next_frontier, next_frontier_size, new_dist, compute_pred);
    }
}




__global__ void bfs_td_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited_bmp,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int frontier_size,
    int new_dist, int compute_pred
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;

    for (int fid = warp_id; fid < frontier_size; fid += total_warps) {
        int src = frontier[fid];
        int start = offsets[src];
        int end = offsets[src + 1];
        for (int e = start + lane; e < end; e += 32) {
            int dst = indices[e];
            td_process_edge(src, dst, offsets, distances, predecessors, visited_bmp,
                            next_frontier, next_frontier_size, new_dist, compute_pred);
        }
    }
}




__global__ void bfs_bu_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    const uint32_t* __restrict__ visited_bmp,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int num_vertices,
    int new_dist, int compute_pred
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    bool found = false;

    if (v < num_vertices && !bmp_test(visited_bmp, v)) {
        int start = offsets[v];
        int end = offsets[v + 1];
        for (int e = start; e < end; e++) {
            int u = indices[e];
            if (bmp_test(visited_bmp, u)) {
                int old = atomicCAS(&distances[v], INT32_MAX, new_dist);
                if (old == INT32_MAX) {
                    found = true;
                    if (compute_pred) predecessors[v] = u;
                }
                break;
            }
        }
    }

    unsigned mask = __ballot_sync(0xffffffff, found);
    if (mask) {
        int count = __popc(mask);
        int base;
        if (lane == 0) base = atomicAdd(next_frontier_size, count);
        base = __shfl_sync(0xffffffff, base, 0);
        if (found) {
            int idx = __popc(mask & ((1u << lane) - 1));
            next_frontier[base + idx] = v;
        }
    }
}




__global__ void update_bitmap_kernel(
    const int32_t* __restrict__ frontier, int frontier_size,
    uint32_t* __restrict__ visited_bmp
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < frontier_size) bmp_set_atomic(visited_bmp, frontier[idx]);
}




__global__ void sum_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ frontier,
    int frontier_size,
    int64_t* __restrict__ result
) {
    typedef cub::BlockReduce<int64_t, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int64_t sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < frontier_size;
         i += blockDim.x * gridDim.x) {
        int v = frontier[i];
        sum += offsets[v + 1] - offsets[v];
    }
    int64_t block_sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) atomicAdd((unsigned long long*)result, (unsigned long long)block_sum);
}




void launch_init_bfs(int32_t* distances, int32_t* predecessors, uint32_t* visited_bmp,
                     int num_vertices, int compute_pred) {
    int T = 256;
    int bmp_words = (num_vertices + 31) >> 5;
    int n = num_vertices > bmp_words ? num_vertices : bmp_words;
    init_bfs_kernel<<<(n + T - 1) / T, T>>>(distances, predecessors, visited_bmp, num_vertices, compute_pred);
}

void launch_init_sources(const int32_t* sources, int n_sources, int32_t* distances,
                         int32_t* predecessors, uint32_t* visited_bmp, int32_t* frontier,
                         int compute_pred) {
    int T = 256;
    init_sources_kernel<<<(n_sources + T - 1) / T + 1, T>>>(
        sources, n_sources, distances, predecessors, visited_bmp, frontier, compute_pred);
}

void launch_bfs_td(const int32_t* offsets, const int32_t* indices, int32_t* distances,
                   int32_t* predecessors, uint32_t* visited_bmp, const int32_t* frontier,
                   int32_t* next_frontier, int32_t* next_frontier_size,
                   int frontier_size, int new_dist, int compute_pred) {
    if (frontier_size <= 64) {
        int T = 256;
        int B = 108 * 4;
        bfs_td_small_kernel<<<B, T>>>(offsets, indices, distances, predecessors, visited_bmp,
                                       frontier, next_frontier, next_frontier_size,
                                       frontier_size, new_dist, compute_pred);
    } else {
        int T = 256;
        int warps_per_block = T / 32;
        int blocks = (frontier_size + warps_per_block - 1) / warps_per_block;
        if (blocks > 4096) blocks = 4096;
        if (blocks < 108) blocks = 108;
        bfs_td_warp_kernel<<<blocks, T>>>(offsets, indices, distances, predecessors, visited_bmp,
                                           frontier, next_frontier, next_frontier_size,
                                           frontier_size, new_dist, compute_pred);
    }
}

void launch_bfs_bu(const int32_t* offsets, const int32_t* indices, int32_t* distances,
                   int32_t* predecessors, const uint32_t* visited_bmp, int32_t* next_frontier,
                   int32_t* next_frontier_size, int num_vertices, int new_dist, int compute_pred) {
    int T = 256;
    int B = (num_vertices + T - 1) / T;
    bfs_bu_kernel<<<B, T>>>(offsets, indices, distances, predecessors, visited_bmp,
                             next_frontier, next_frontier_size, num_vertices, new_dist, compute_pred);
}

void launch_update_bitmap(const int32_t* frontier, int frontier_size, uint32_t* visited_bmp) {
    if (frontier_size == 0) return;
    int T = 256;
    update_bitmap_kernel<<<(frontier_size + T - 1) / T, T>>>(frontier, frontier_size, visited_bmp);
}

void launch_sum_degrees(const int32_t* offsets, const int32_t* frontier, int frontier_size,
                        int64_t* result) {
    if (frontier_size == 0) return;
    int T = 256;
    int B = (frontier_size + T - 1) / T;
    if (B > 256) B = 256;
    sum_degrees_kernel<<<B, T>>>(offsets, frontier, frontier_size, result);
}

}  

void bfs_direction_optimizing(const graph32_t& graph,
                              int32_t* distances,
                              int32_t* predecessors,
                              const int32_t* sources,
                              std::size_t n_sources,
                              int32_t depth_limit) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    int pred_flag = (predecessors != nullptr) ? 1 : 0;

    if (depth_limit < 0) depth_limit = INT32_MAX;
    cache.ensure((size_t)num_vertices);

    
    launch_init_bfs(distances, predecessors, cache.d_visited_bmp, num_vertices, pred_flag);
    launch_init_sources(sources, (int)n_sources, distances, predecessors,
                        cache.d_visited_bmp, cache.d_frontier1, pred_flag);

    int32_t* cur_frontier = cache.d_frontier1;
    int32_t* next_frontier = cache.d_frontier2;
    int frontier_size = (int)n_sources;
    int current_level = 0;

    
    double avg_degree = (num_vertices > 0) ? (double)num_edges / num_vertices : 0.0;
    double alpha = avg_degree * 0.25;
    if (alpha < 2.0) alpha = 2.0;
    const int beta = 24;
    bool top_down = true;
    int64_t m_u = num_edges;
    int64_t m_f = 0;

    
    cudaMemset(cache.d_degree_sum, 0, sizeof(int64_t));
    launch_sum_degrees(d_offsets, cur_frontier, frontier_size, cache.d_degree_sum);
    cudaMemcpy(cache.h_degree_sum, cache.d_degree_sum, sizeof(int64_t), cudaMemcpyDeviceToHost);
    m_f = *cache.h_degree_sum;
    m_u -= m_f;

    while (frontier_size > 0) {
        cudaMemsetAsync(cache.d_counter, 0, sizeof(int32_t), 0);

        int new_dist = current_level + 1;

        if (top_down) {
            launch_bfs_td(d_offsets, d_indices, distances, predecessors,
                          cache.d_visited_bmp, cur_frontier, next_frontier, cache.d_counter,
                          frontier_size, new_dist, pred_flag);
        } else {
            launch_bfs_bu(d_offsets, d_indices, distances, predecessors,
                          cache.d_visited_bmp, next_frontier, cache.d_counter,
                          num_vertices, new_dist, pred_flag);
        }

        
        cudaMemcpyAsync(cache.h_counter, cache.d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost, 0);
        cudaStreamSynchronize(0);
        int32_t next_size = *cache.h_counter;

        
        if (!top_down && next_size > 0) {
            launch_update_bitmap(next_frontier, next_size, cache.d_visited_bmp);
        }

        
        int64_t next_m_f = 0;
        if (next_size > 0) {
            cudaMemsetAsync(cache.d_degree_sum, 0, sizeof(int64_t), 0);
            launch_sum_degrees(d_offsets, next_frontier, next_size, cache.d_degree_sum);
            cudaMemcpyAsync(cache.h_degree_sum, cache.d_degree_sum, sizeof(int64_t), cudaMemcpyDeviceToHost, 0);
            cudaStreamSynchronize(0);
            next_m_f = *cache.h_degree_sum;
        }

        m_u -= next_m_f;
        if (m_u < 0) m_u = 0;

        
        if (top_down) {
            if (next_size > 0 && next_size >= frontier_size &&
                (double)next_m_f * alpha > (double)m_u) {
                top_down = false;
            }
        } else {
            int64_t unvisited_approx = (avg_degree > 0) ? m_u / (int64_t)avg_degree : 0;
            if (next_size < frontier_size &&
                (int64_t)next_size * beta < unvisited_approx) {
                top_down = true;
            }
        }

        
        int32_t* temp = cur_frontier;
        cur_frontier = next_frontier;
        next_frontier = temp;
        m_f = next_m_f;
        frontier_size = next_size;
        current_level++;
        if (depth_limit != INT32_MAX && current_level >= depth_limit) break;
    }
}

}  
