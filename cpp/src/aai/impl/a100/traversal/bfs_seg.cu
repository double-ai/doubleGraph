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
#include <limits>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* fsize_d = nullptr;
    uint32_t* visited = nullptr;
    int32_t capacity = 0;

    void ensure(int32_t num_vertices) {
        if (num_vertices <= capacity) return;

        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (fsize_d) cudaFree(fsize_d);
        if (visited) cudaFree(visited);

        cudaMalloc(&frontier_a, (size_t)num_vertices * sizeof(int32_t));
        cudaMalloc(&frontier_b, (size_t)num_vertices * sizeof(int32_t));
        cudaMalloc(&fsize_d, sizeof(int32_t));
        int32_t bm_words = (num_vertices + 31) / 32;
        cudaMalloc(&visited, (size_t)bm_words * sizeof(uint32_t));

        capacity = num_vertices;
    }

    ~Cache() override {
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (fsize_d) cudaFree(fsize_d);
        if (visited) cudaFree(visited);
    }
};

__global__ void bfs_init(
    int32_t* __restrict__ dist,
    int32_t* __restrict__ pred,
    int n,
    bool has_pred
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        dist[i] = 0x7FFFFFFF;
        if (has_pred) pred[i] = -1;
    }
}

__global__ void bfs_set_sources(
    int32_t* __restrict__ dist,
    int32_t* __restrict__ pred,
    const int32_t* __restrict__ srcs,
    int ns,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ fsize,
    uint32_t* __restrict__ visited,
    bool has_pred
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ns) {
        int s = srcs[i];
        dist[s] = 0;
        if (has_pred) pred[s] = -1;
        frontier[i] = s;
        atomicOr(&visited[s >> 5], 1u << (s & 31));
    }
    if (i == 0) *fsize = ns;
}

__global__ void bfs_td_warp(
    const int32_t* __restrict__ row_off,
    const int32_t* __restrict__ col_idx,
    int32_t* __restrict__ dist,
    int32_t* __restrict__ pred,
    const int32_t* __restrict__ frontier,
    int fsize,
    int32_t* __restrict__ next_f,
    int32_t* __restrict__ next_fsize,
    uint32_t* __restrict__ visited,
    int new_dist,
    bool has_pred
) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int num_warps = (gridDim.x * blockDim.x) >> 5;

    for (int fi = warp; fi < fsize; fi += num_warps) {
        int v = frontier[fi];
        int r_start = row_off[v];
        int r_end = row_off[v + 1];

        for (int e = r_start + lane; e < r_end; e += 32) {
            int nb = col_idx[e];
            unsigned bit = 1u << (nb & 31);
            unsigned wi = nb >> 5;

            if (visited[wi] & bit) continue;

            unsigned old = atomicOr(&visited[wi], bit);
            if (!(old & bit)) {
                dist[nb] = new_dist;
                if (has_pred) pred[nb] = v;
                int pos = atomicAdd(next_fsize, 1);
                next_f[pos] = nb;
            }
        }
    }
}

}  

void bfs_seg(const graph32_t& graph,
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
    bool has_pred = (predecessors != nullptr);

    if (depth_limit < 0) depth_limit = std::numeric_limits<int32_t>::max();

    cache.ensure(num_vertices);

    int32_t bm_words = (num_vertices + 31) / 32;
    const int BLK = 256;
    cudaStream_t s = 0;

    
    int g = (num_vertices + BLK - 1) / BLK;
    bfs_init<<<g, BLK, 0, s>>>(distances, predecessors, num_vertices, has_pred);

    
    cudaMemsetAsync(cache.visited, 0, bm_words * 4, s);

    
    g = ((int)n_sources + BLK - 1) / BLK;
    bfs_set_sources<<<g, BLK, 0, s>>>(
        distances, predecessors, sources, (int)n_sources,
        cache.frontier_a, cache.fsize_d, cache.visited, has_pred
    );

    
    int32_t* cur_f = cache.frontier_a;
    int32_t* nxt_f = cache.frontier_b;
    int h_fsize = (int)n_sources;
    int depth = 0;

    while (h_fsize > 0 && depth < depth_limit) {
        cudaMemsetAsync(cache.fsize_d, 0, 4, s);

        int warps_needed = h_fsize;
        g = (warps_needed * 32 + BLK - 1) / BLK;
        if (g < 1) g = 1;

        bfs_td_warp<<<g, BLK, 0, s>>>(
            offsets, indices, distances, predecessors,
            cur_f, h_fsize, nxt_f, cache.fsize_d, cache.visited,
            depth + 1, has_pred
        );

        cudaMemcpy(&h_fsize, cache.fsize_d, 4, cudaMemcpyDeviceToHost);

        int32_t* tmp = cur_f;
        cur_f = nxt_f;
        nxt_f = tmp;

        depth++;
    }

    cudaStreamSynchronize(s);
}

}  
