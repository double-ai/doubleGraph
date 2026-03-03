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

namespace aai {

namespace {





template <bool COMPUTE_PRED>
__global__ void bfs_init_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    int32_t num_vertices,
    int32_t visited_words
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        distances[idx] = INT32_MAX;
        if (COMPUTE_PRED) predecessors[idx] = -1;
    }
    if (idx < visited_words) visited[idx] = 0u;
}

template <bool COMPUTE_PRED>
__global__ void bfs_set_sources_kernel(
    const int32_t* __restrict__ sources,
    int32_t n_sources,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_sources) {
        int32_t v = sources[idx];
        distances[v] = 0;
        if (COMPUTE_PRED) predecessors[v] = -1;
        atomicOr(&visited[v >> 5], 1u << (v & 31));
        int pos = atomicAdd(frontier_count, 1);
        frontier[pos] = v;
    }
}

template <bool COMPUTE_PRED>
__global__ void bfs_topdown_kernel(
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
    int32_t next_depth
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid >> 5;
    int lane = global_tid & 31;

    if (warp_id >= frontier_size) return;

    int32_t v = frontier[warp_id];
    int32_t row_start = __ldg(&offsets[v]);
    int32_t row_end = __ldg(&offsets[v + 1]);

    for (int32_t e = row_start + lane; e < row_end; e += 32) {
        uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
        if (!((mask_word >> (e & 31)) & 1u)) continue;

        int32_t neighbor = __ldg(&indices[e]);
        uint32_t vis_word_idx = static_cast<uint32_t>(neighbor) >> 5;
        uint32_t vis_bit = 1u << (neighbor & 31);

        if (visited[vis_word_idx] & vis_bit) continue;

        uint32_t old = atomicOr(&visited[vis_word_idx], vis_bit);
        if (!(old & vis_bit)) {
            distances[neighbor] = next_depth;
            if (COMPUTE_PRED) predecessors[neighbor] = v;
            int pos = atomicAdd(next_count, 1);
            next_frontier[pos] = neighbor;
        }
    }
}

template <bool COMPUTE_PRED>
__global__ void bfs_topdown_persistent_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    const int32_t* __restrict__ frontier,
    const int32_t* __restrict__ frontier_size_ptr,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count,
    int32_t next_depth
) {
    int32_t frontier_size = *frontier_size_ptr;

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid >> 5;
    int lane = global_tid & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;

    for (int w = warp_id; w < frontier_size; w += total_warps) {
        int32_t v = frontier[w];
        int32_t row_start = __ldg(&offsets[v]);
        int32_t row_end = __ldg(&offsets[v + 1]);

        for (int32_t e = row_start + lane; e < row_end; e += 32) {
            uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
            if (!((mask_word >> (e & 31)) & 1u)) continue;

            int32_t neighbor = __ldg(&indices[e]);
            uint32_t vis_word_idx = static_cast<uint32_t>(neighbor) >> 5;
            uint32_t vis_bit = 1u << (neighbor & 31);

            if (visited[vis_word_idx] & vis_bit) continue;

            uint32_t old = atomicOr(&visited[vis_word_idx], vis_bit);
            if (!(old & vis_bit)) {
                distances[neighbor] = next_depth;
                if (COMPUTE_PRED) predecessors[neighbor] = v;
                int pos = atomicAdd(next_count, 1);
                next_frontier[pos] = neighbor;
            }
        }
    }
}





int g_max_grid = 0;
int g_num_sms = 0;

void ensure_grid_info() {
    if (g_max_grid == 0) {
        int max_bpsm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_bpsm, bfs_topdown_persistent_kernel<true>, 256, 0);
        cudaDeviceGetAttribute(&g_num_sms, cudaDevAttrMultiProcessorCount, 0);
        g_max_grid = max_bpsm * g_num_sms;
    }
}





void launch_bfs_init(
    int32_t* distances, int32_t* predecessors, uint32_t* visited,
    int32_t num_vertices, int32_t visited_words, bool compute_predecessors,
    cudaStream_t stream
) {
    int block = 512;
    int grid = (num_vertices + block - 1) / block;
    if (compute_predecessors)
        bfs_init_kernel<true><<<grid, block, 0, stream>>>(distances, predecessors, visited, num_vertices, visited_words);
    else
        bfs_init_kernel<false><<<grid, block, 0, stream>>>(distances, predecessors, visited, num_vertices, visited_words);
}

void launch_bfs_set_sources(
    const int32_t* sources, int32_t n_sources,
    int32_t* distances, int32_t* predecessors, uint32_t* visited,
    int32_t* frontier, int32_t* frontier_count,
    bool compute_predecessors, cudaStream_t stream
) {
    if (n_sources <= 0) return;
    int block = 256;
    int grid = (n_sources + block - 1) / block;
    if (compute_predecessors)
        bfs_set_sources_kernel<true><<<grid, block, 0, stream>>>(
            sources, n_sources, distances, predecessors, visited, frontier, frontier_count);
    else
        bfs_set_sources_kernel<false><<<grid, block, 0, stream>>>(
            sources, n_sources, distances, predecessors, visited, frontier, frontier_count);
}

void launch_bfs_topdown(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    int32_t* distances, int32_t* predecessors, uint32_t* visited,
    const int32_t* frontier, int32_t frontier_size,
    int32_t* next_frontier, int32_t* next_count,
    int32_t next_depth, bool compute_predecessors, cudaStream_t stream
) {
    if (frontier_size <= 0) return;
    int tpb = 256;
    int warps_per_block = tpb / 32;
    int grid = (frontier_size + warps_per_block - 1) / warps_per_block;
    if (compute_predecessors)
        bfs_topdown_kernel<true><<<grid, tpb, 0, stream>>>(
            offsets, indices, edge_mask, distances, predecessors, visited,
            frontier, frontier_size, next_frontier, next_count, next_depth);
    else
        bfs_topdown_kernel<false><<<grid, tpb, 0, stream>>>(
            offsets, indices, edge_mask, distances, predecessors, visited,
            frontier, frontier_size, next_frontier, next_count, next_depth);
}

void launch_bfs_topdown_persistent(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    int32_t* distances, int32_t* predecessors, uint32_t* visited,
    const int32_t* frontier, const int32_t* frontier_size_ptr,
    int32_t* next_frontier, int32_t* next_count,
    int32_t next_depth, bool compute_predecessors, cudaStream_t stream
) {
    ensure_grid_info();
    int tpb = 256;
    if (compute_predecessors)
        bfs_topdown_persistent_kernel<true><<<g_max_grid, tpb, 0, stream>>>(
            offsets, indices, edge_mask, distances, predecessors, visited,
            frontier, frontier_size_ptr, next_frontier, next_count, next_depth);
    else
        bfs_topdown_persistent_kernel<false><<<g_max_grid, tpb, 0, stream>>>(
            offsets, indices, edge_mask, distances, predecessors, visited,
            frontier, frontier_size_ptr, next_frontier, next_count, next_depth);
}





struct Cache : Cacheable {
    uint32_t* visited = nullptr;
    int32_t* frontier1 = nullptr;
    int32_t* frontier2 = nullptr;
    int32_t* counters = nullptr;
    int32_t* h_counter = nullptr;

    int32_t visited_capacity = 0;
    int32_t frontier1_capacity = 0;
    int32_t frontier2_capacity = 0;
    bool counters_allocated = false;
    bool h_counter_allocated = false;

    void ensure(int32_t num_vertices) {
        int32_t visited_words = (num_vertices + 31) / 32;

        if (visited_capacity < visited_words) {
            if (visited) cudaFree(visited);
            cudaMalloc(&visited, visited_words * sizeof(uint32_t));
            visited_capacity = visited_words;
        }
        if (frontier1_capacity < num_vertices) {
            if (frontier1) cudaFree(frontier1);
            cudaMalloc(&frontier1, num_vertices * sizeof(int32_t));
            frontier1_capacity = num_vertices;
        }
        if (frontier2_capacity < num_vertices) {
            if (frontier2) cudaFree(frontier2);
            cudaMalloc(&frontier2, num_vertices * sizeof(int32_t));
            frontier2_capacity = num_vertices;
        }
        if (!counters_allocated) {
            cudaMalloc(&counters, 2 * sizeof(int32_t));
            counters_allocated = true;
        }
        if (!h_counter_allocated) {
            cudaHostAlloc(&h_counter, sizeof(int32_t), cudaHostAllocDefault);
            h_counter_allocated = true;
        }
    }

    ~Cache() override {
        if (visited) cudaFree(visited);
        if (frontier1) cudaFree(frontier1);
        if (frontier2) cudaFree(frontier2);
        if (counters) cudaFree(counters);
        if (h_counter) cudaFreeHost(h_counter);
    }
};

}  





void bfs_mask(const graph32_t& graph,
              int32_t* distances,
              int32_t* predecessors,
              const int32_t* sources,
              std::size_t n_sources,
              int32_t depth_limit) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    const uint32_t* edge_mask = graph.edge_mask;

    bool compute_predecessors = (predecessors != nullptr);
    if (depth_limit < 0) depth_limit = INT32_MAX;

    cache.ensure(num_vertices);

    int32_t visited_words = (num_vertices + 31) / 32;
    cudaStream_t stream = 0;

    launch_bfs_init(distances, predecessors, cache.visited, num_vertices, visited_words,
                    compute_predecessors, stream);
    cudaMemsetAsync(cache.counters, 0, sizeof(int32_t), stream);

    int32_t* frontier_bufs[2] = {cache.frontier1, cache.frontier2};
    int32_t* counter_ptrs[2] = {cache.counters, cache.counters + 1};

    launch_bfs_set_sources(
        sources, static_cast<int32_t>(n_sources),
        distances, predecessors, cache.visited, frontier_bufs[0], counter_ptrs[0],
        compute_predecessors, stream);

    
    cudaMemcpyAsync(cache.h_counter, counter_ptrs[0], sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int32_t frontier_size = *cache.h_counter;

    int buf = 0;
    int32_t depth = 0;

    
    while (frontier_size > 0 && depth < depth_limit) {
        
        {
            int next_buf = 1 - buf;
            cudaMemsetAsync(counter_ptrs[next_buf], 0, sizeof(int32_t), stream);
            launch_bfs_topdown(offsets, indices, edge_mask, distances, predecessors, cache.visited,
                frontier_bufs[buf], frontier_size, frontier_bufs[next_buf], counter_ptrs[next_buf],
                depth + 1, compute_predecessors, stream);
            buf = next_buf;
            depth++;
        }

        
        int check_interval = 7; 
        int batch_end = depth + check_interval;
        if (batch_end > depth_limit) batch_end = depth_limit;

        while (depth < batch_end) {
            int next_buf = 1 - buf;
            cudaMemsetAsync(counter_ptrs[next_buf], 0, sizeof(int32_t), stream);
            launch_bfs_topdown_persistent(offsets, indices, edge_mask, distances, predecessors, cache.visited,
                frontier_bufs[buf], counter_ptrs[buf],
                frontier_bufs[next_buf], counter_ptrs[next_buf],
                depth + 1, compute_predecessors, stream);
            buf = next_buf;
            depth++;
        }

        
        cudaMemcpyAsync(cache.h_counter, counter_ptrs[buf], sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        frontier_size = *cache.h_counter;
    }
}

}  
