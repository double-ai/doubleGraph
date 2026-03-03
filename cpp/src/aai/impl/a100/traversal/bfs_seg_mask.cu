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
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* d_counters = nullptr;
    uint32_t* d_visited = nullptr;
    int32_t* d_frontier1 = nullptr;
    int32_t* d_frontier2 = nullptr;
    int32_t* d_high = nullptr;
    int32_t* d_mid = nullptr;
    int32_t* d_low = nullptr;

    int64_t counters_capacity = 0;
    int64_t visited_capacity = 0;
    int64_t frontier1_capacity = 0;
    int64_t frontier2_capacity = 0;
    int64_t high_capacity = 0;
    int64_t mid_capacity = 0;
    int64_t low_capacity = 0;

    void ensure(int32_t num_vertices) {
        int64_t bitmap_words = (num_vertices + 31) / 32;

        if (counters_capacity < 8) {
            if (d_counters) cudaFree(d_counters);
            cudaMalloc(&d_counters, sizeof(int32_t) * 8);
            counters_capacity = 8;
        }
        if (visited_capacity < bitmap_words) {
            if (d_visited) cudaFree(d_visited);
            cudaMalloc(&d_visited, bitmap_words * sizeof(uint32_t));
            visited_capacity = bitmap_words;
        }
        if (frontier1_capacity < num_vertices) {
            if (d_frontier1) cudaFree(d_frontier1);
            cudaMalloc(&d_frontier1, num_vertices * sizeof(int32_t));
            frontier1_capacity = num_vertices;
        }
        if (frontier2_capacity < num_vertices) {
            if (d_frontier2) cudaFree(d_frontier2);
            cudaMalloc(&d_frontier2, num_vertices * sizeof(int32_t));
            frontier2_capacity = num_vertices;
        }
        if (high_capacity < num_vertices) {
            if (d_high) cudaFree(d_high);
            cudaMalloc(&d_high, num_vertices * sizeof(int32_t));
            high_capacity = num_vertices;
        }
        if (mid_capacity < num_vertices) {
            if (d_mid) cudaFree(d_mid);
            cudaMalloc(&d_mid, num_vertices * sizeof(int32_t));
            mid_capacity = num_vertices;
        }
        if (low_capacity < num_vertices) {
            if (d_low) cudaFree(d_low);
            cudaMalloc(&d_low, num_vertices * sizeof(int32_t));
            low_capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (d_counters) cudaFree(d_counters);
        if (d_visited) cudaFree(d_visited);
        if (d_frontier1) cudaFree(d_frontier1);
        if (d_frontier2) cudaFree(d_frontier2);
        if (d_high) cudaFree(d_high);
        if (d_mid) cudaFree(d_mid);
        if (d_low) cudaFree(d_low);
    }
};


__global__ void bfs_init_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t num_vertices,
    bool compute_pred)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        distances[idx] = INT32_MAX;
        if (compute_pred) predecessors[idx] = -1;
    }
}


__global__ void bfs_sources_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    const int32_t* __restrict__ sources,
    int32_t n_sources,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_count,
    bool compute_pred)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sources) return;

    int32_t v = sources[idx];
    distances[v] = 0;
    if (compute_pred) predecessors[v] = -1;
    atomicOr(&visited[v >> 5], 1u << (v & 31));
    int pos = atomicAdd(frontier_count, 1);
    frontier[pos] = v;
}


__global__ void bfs_td_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count,
    int32_t depth,
    bool compute_pred)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    if (warp_id >= frontier_size) return;

    int32_t v = frontier[warp_id];
    int32_t row_start = offsets[v];
    int32_t row_end = offsets[v + 1];

    for (int32_t e = row_start + lane; e < row_end; e += 32) {
        bool discovered = false;
        int32_t neighbor = -1;

        
        uint32_t mask_word = edge_mask[e >> 5];
        bool active = (mask_word >> (e & 31)) & 1;

        if (active) {
            neighbor = indices[e];

            
            uint32_t v_word_idx = neighbor >> 5;
            uint32_t v_bit = 1u << (neighbor & 31);

            if (!(visited[v_word_idx] & v_bit)) {
                uint32_t old = atomicOr(&visited[v_word_idx], v_bit);
                if (!(old & v_bit)) {
                    discovered = true;
                    distances[neighbor] = depth;
                    if (compute_pred) predecessors[neighbor] = v;
                }
            }
        }

        
        uint32_t ballot = __ballot_sync(0xFFFFFFFF, discovered);
        if (ballot) {
            int count = __popc(ballot);
            int32_t base;
            if (lane == 0) {
                base = atomicAdd(next_count, count);
            }
            base = __shfl_sync(0xFFFFFFFF, base, 0);
            int offset = __popc(ballot & ((1u << lane) - 1));
            if (discovered) {
                next_frontier[base + offset] = neighbor;
            }
        }
    }
}


__global__ void bfs_td_block_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count,
    int32_t depth,
    bool compute_pred)
{
    if (blockIdx.x >= frontier_size) return;

    int32_t v = frontier[blockIdx.x];
    int32_t row_start = offsets[v];
    int32_t row_end = offsets[v + 1];
    int lane = threadIdx.x & 31;

    for (int32_t e = row_start + threadIdx.x; e < row_end; e += blockDim.x) {
        bool discovered = false;
        int32_t neighbor = -1;

        uint32_t mask_word = edge_mask[e >> 5];
        bool active = (mask_word >> (e & 31)) & 1;

        if (active) {
            neighbor = indices[e];
            uint32_t v_word_idx = neighbor >> 5;
            uint32_t v_bit = 1u << (neighbor & 31);

            if (!(visited[v_word_idx] & v_bit)) {
                uint32_t old = atomicOr(&visited[v_word_idx], v_bit);
                if (!(old & v_bit)) {
                    discovered = true;
                    distances[neighbor] = depth;
                    if (compute_pred) predecessors[neighbor] = v;
                }
            }
        }

        
        uint32_t ballot = __ballot_sync(0xFFFFFFFF, discovered);
        if (ballot) {
            int count = __popc(ballot);
            int32_t base;
            if (lane == 0) {
                base = atomicAdd(next_count, count);
            }
            base = __shfl_sync(0xFFFFFFFF, base, 0);
            int offset = __popc(ballot & ((1u << lane) - 1));
            if (discovered) {
                next_frontier[base + offset] = neighbor;
            }
        }
    }
}


__global__ void partition_frontier_kernel(
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ high_queue,
    int32_t* __restrict__ mid_queue,
    int32_t* __restrict__ low_queue,
    int32_t* __restrict__ high_count,
    int32_t* __restrict__ mid_count,
    int32_t* __restrict__ low_count,
    int32_t seg_high,    
    int32_t seg_mid)     
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int32_t v = frontier[tid];
    if (v < seg_high) {
        int pos = atomicAdd(high_count, 1);
        high_queue[pos] = v;
    } else if (v < seg_mid) {
        int pos = atomicAdd(mid_count, 1);
        mid_queue[pos] = v;
    } else {
        int pos = atomicAdd(low_count, 1);
        low_queue[pos] = v;
    }
}

void launch_bfs_init(int32_t* distances, int32_t* predecessors,
                     int32_t num_vertices, bool compute_pred) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    bfs_init_kernel<<<grid, block>>>(distances, predecessors, num_vertices, compute_pred);
}

void launch_bfs_sources(
    int32_t* distances, int32_t* predecessors, uint32_t* visited,
    const int32_t* sources, int32_t n_sources,
    int32_t* frontier, int32_t* frontier_count, bool compute_pred) {
    if (n_sources == 0) return;
    int block = 256;
    int grid = (n_sources + block - 1) / block;
    bfs_sources_kernel<<<grid, block>>>(distances, predecessors, visited,
                                         sources, n_sources,
                                         frontier, frontier_count, compute_pred);
}

void launch_bfs_td_warp(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    int32_t* distances, int32_t* predecessors, uint32_t* visited,
    const int32_t* frontier, int32_t frontier_size,
    int32_t* next_frontier, int32_t* next_count,
    int32_t depth, bool compute_pred) {
    if (frontier_size == 0) return;
    int block = 256;
    int warps_per_block = block / 32;
    int grid = (frontier_size + warps_per_block - 1) / warps_per_block;
    bfs_td_warp_kernel<<<grid, block>>>(
        offsets, indices, edge_mask,
        distances, predecessors, visited,
        frontier, frontier_size,
        next_frontier, next_count,
        depth, compute_pred);
}

void launch_bfs_td_block(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    int32_t* distances, int32_t* predecessors, uint32_t* visited,
    const int32_t* frontier, int32_t frontier_size,
    int32_t* next_frontier, int32_t* next_count,
    int32_t depth, bool compute_pred) {
    if (frontier_size == 0) return;
    int block = 512;
    bfs_td_block_kernel<<<frontier_size, block>>>(
        offsets, indices, edge_mask,
        distances, predecessors, visited,
        frontier, frontier_size,
        next_frontier, next_count,
        depth, compute_pred);
}

void launch_partition_frontier(
    const int32_t* frontier, int32_t frontier_size,
    int32_t* high_queue, int32_t* mid_queue, int32_t* low_queue,
    int32_t* high_count, int32_t* mid_count, int32_t* low_count,
    int32_t seg_high, int32_t seg_mid) {
    if (frontier_size == 0) return;
    int block = 256;
    int grid = (frontier_size + block - 1) / block;
    partition_frontier_kernel<<<grid, block>>>(
        frontier, frontier_size,
        high_queue, mid_queue, low_queue,
        high_count, mid_count, low_count,
        seg_high, seg_mid);
}

}  

void bfs_seg_mask(const graph32_t& graph,
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
    const uint32_t* d_edge_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg_high = seg[1];
    int32_t seg_mid = seg[2];

    bool compute_pred = (predecessors != nullptr);

    if (depth_limit < 0) depth_limit = INT32_MAX;

    cache.ensure(num_vertices);

    int32_t bitmap_words = (num_vertices + 31) / 32;
    uint32_t* d_visited = cache.d_visited;
    int32_t* d_cur = cache.d_frontier1;
    int32_t* d_next = cache.d_frontier2;
    int32_t* d_high = cache.d_high;
    int32_t* d_mid = cache.d_mid;
    int32_t* d_low = cache.d_low;

    int32_t* d_cnt_frontier = cache.d_counters;
    int32_t* d_cnt_high = cache.d_counters + 1;
    int32_t* d_cnt_mid = cache.d_counters + 2;
    int32_t* d_cnt_low = cache.d_counters + 3;

    
    cudaMemset(d_visited, 0, bitmap_words * sizeof(uint32_t));
    launch_bfs_init(distances, predecessors, num_vertices, compute_pred);

    
    cudaMemset(d_cnt_frontier, 0, sizeof(int32_t));
    launch_bfs_sources(distances, predecessors, d_visited,
                       sources, static_cast<int32_t>(n_sources),
                       d_cur, d_cnt_frontier, compute_pred);

    int32_t h_counters[4];
    cudaMemcpy(&h_counters[0], d_cnt_frontier, sizeof(int32_t), cudaMemcpyDeviceToHost);
    int32_t frontier_size = h_counters[0];

    
    int32_t depth = 1;
    while (frontier_size > 0 && depth <= depth_limit) {
        
        cudaMemset(d_cnt_frontier, 0, sizeof(int32_t));

        
        if (seg_high > 0 && frontier_size > 32) {
            
            cudaMemset(d_cnt_high, 0, sizeof(int32_t) * 3);
            launch_partition_frontier(d_cur, frontier_size,
                                      d_high, d_mid, d_low,
                                      d_cnt_high, d_cnt_mid, d_cnt_low,
                                      seg_high, seg_mid);

            
            cudaMemcpy(h_counters + 1, d_cnt_high, sizeof(int32_t) * 3, cudaMemcpyDeviceToHost);
            int32_t n_high = h_counters[1];
            int32_t n_mid = h_counters[2];
            int32_t n_low = h_counters[3];

            
            if (n_high > 0) {
                launch_bfs_td_block(d_offsets, d_indices, d_edge_mask,
                                    distances, predecessors, d_visited,
                                    d_high, n_high,
                                    d_next, d_cnt_frontier,
                                    depth, compute_pred);
            }
            if (n_mid > 0) {
                launch_bfs_td_warp(d_offsets, d_indices, d_edge_mask,
                                   distances, predecessors, d_visited,
                                   d_mid, n_mid,
                                   d_next, d_cnt_frontier,
                                   depth, compute_pred);
            }
            if (n_low > 0) {
                launch_bfs_td_warp(d_offsets, d_indices, d_edge_mask,
                                   distances, predecessors, d_visited,
                                   d_low, n_low,
                                   d_next, d_cnt_frontier,
                                   depth, compute_pred);
            }
        } else {
            
            launch_bfs_td_warp(d_offsets, d_indices, d_edge_mask,
                               distances, predecessors, d_visited,
                               d_cur, frontier_size,
                               d_next, d_cnt_frontier,
                               depth, compute_pred);
        }

        
        cudaMemcpy(&h_counters[0], d_cnt_frontier, sizeof(int32_t), cudaMemcpyDeviceToHost);
        frontier_size = h_counters[0];

        std::swap(d_cur, d_next);
        depth++;
    }
}

}  
