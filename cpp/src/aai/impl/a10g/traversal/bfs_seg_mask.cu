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
#include <limits>

namespace aai {

namespace {

struct Cache : Cacheable {
    uint32_t* visited = nullptr;
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* count_a = nullptr;
    int32_t* count_b = nullptr;
    int32_t capacity = 0;

    void ensure(int32_t num_vertices) {
        if (capacity < num_vertices) {
            if (visited) cudaFree(visited);
            if (frontier_a) cudaFree(frontier_a);
            if (frontier_b) cudaFree(frontier_b);
            if (count_a) cudaFree(count_a);
            if (count_b) cudaFree(count_b);

            int32_t bw = (num_vertices + 31) / 32;
            cudaMalloc(&visited, bw * sizeof(uint32_t));
            cudaMalloc(&frontier_a, num_vertices * sizeof(int32_t));
            cudaMalloc(&frontier_b, num_vertices * sizeof(int32_t));
            cudaMalloc(&count_a, sizeof(int32_t));
            cudaMalloc(&count_b, sizeof(int32_t));
            capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (visited) cudaFree(visited);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (count_a) cudaFree(count_a);
        if (count_b) cudaFree(count_b);
    }
};



__global__ void bfs_init_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    int32_t num_vertices, int32_t bitmap_words, bool has_pred)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        distances[idx] = INT32_MAX;
        if (has_pred) predecessors[idx] = -1;
    }
    if (idx < bitmap_words) visited[idx] = 0;
}

__global__ void bfs_set_sources_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ count,
    const int32_t* __restrict__ sources,
    int32_t n_sources, bool has_pred)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_sources) {
        int32_t s = sources[idx];
        distances[s] = 0;
        if (has_pred) predecessors[s] = -1;
        atomicOr(&visited[s >> 5], 1u << (s & 31));
        frontier[idx] = s;
    }
    
    if (idx == 0) *count = n_sources;
}



template <bool HAS_PRED>
__global__ void bfs_expand_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ visited,
    const int32_t* __restrict__ frontier,
    const int32_t* __restrict__ frontier_count,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count,
    int32_t depth)
{
    int32_t fsize = *frontier_count;
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int nwarps = (blockDim.x * gridDim.x) >> 5;

    for (int32_t wi = wid; wi < fsize; wi += nwarps) {
        int32_t v = frontier[wi];
        int32_t rs = offsets[v], re = offsets[v + 1];

        for (int32_t e = rs + lane; e < re; e += 32) {
            if (!((edge_mask[e >> 5] >> (e & 31)) & 1u)) continue;
            int32_t nbr = indices[e];
            uint32_t bit = 1u << (nbr & 31);
            uint32_t widx = nbr >> 5;
            if (visited[widx] & bit) continue;
            uint32_t old = atomicOr(&visited[widx], bit);
            if (old & bit) continue;
            distances[nbr] = depth;
            if constexpr (HAS_PRED) predecessors[nbr] = v;
            int32_t pos = atomicAdd(next_count, 1);
            next_frontier[pos] = nbr;
        }
    }
}


template <bool HAS_PRED>
__global__ void bfs_expand_block(
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
    int32_t depth)
{
    for (int32_t fi = blockIdx.x; fi < frontier_size; fi += gridDim.x) {
        int32_t v = frontier[fi];
        int32_t rs = offsets[v], re = offsets[v + 1];
        for (int32_t e = rs + threadIdx.x; e < re; e += blockDim.x) {
            if (!((edge_mask[e >> 5] >> (e & 31)) & 1u)) continue;
            int32_t nbr = indices[e];
            uint32_t bit = 1u << (nbr & 31);
            uint32_t widx = nbr >> 5;
            if (visited[widx] & bit) continue;
            uint32_t old = atomicOr(&visited[widx], bit);
            if (old & bit) continue;
            distances[nbr] = depth;
            if constexpr (HAS_PRED) predecessors[nbr] = v;
            int32_t pos = atomicAdd(next_count, 1);
            next_frontier[pos] = nbr;
        }
    }
}



void launch_bfs_init(int32_t* d, int32_t* p, uint32_t* v, int32_t nv, int32_t bw, bool hp, cudaStream_t s) {
    int n = (nv > bw) ? nv : bw;
    bfs_init_kernel<<<(n+255)/256, 256, 0, s>>>(d, p, v, nv, bw, hp);
}

void launch_bfs_set_sources(int32_t* d, int32_t* p, uint32_t* v,
    int32_t* f, int32_t* c, const int32_t* src, int32_t ns, bool hp, cudaStream_t s) {
    if (ns == 0) return;
    bfs_set_sources_kernel<<<(ns+255)/256, 256, 0, s>>>(d, p, v, f, c, src, ns, hp);
}

void launch_bfs_expand_warp(
    const int32_t* off, const int32_t* idx, const uint32_t* em,
    int32_t* d, int32_t* p, uint32_t* v,
    const int32_t* f, const int32_t* fc,
    int32_t* nf, int32_t* nc,
    int32_t dep, bool hp, int grid, cudaStream_t s)
{
    if (grid <= 0) return;
    if (hp) bfs_expand_warp<true><<<grid, 256, 0, s>>>(off,idx,em,d,p,v,f,fc,nf,nc,dep);
    else bfs_expand_warp<false><<<grid, 256, 0, s>>>(off,idx,em,d,p,v,f,fc,nf,nc,dep);
}

void launch_bfs_expand_block(
    const int32_t* off, const int32_t* idx, const uint32_t* em,
    int32_t* d, int32_t* p, uint32_t* v,
    const int32_t* f, int32_t fs,
    int32_t* nf, int32_t* nc,
    int32_t dep, bool hp, cudaStream_t s)
{
    if (fs <= 0) return;
    if (hp) bfs_expand_block<true><<<fs, 256, 0, s>>>(off,idx,em,d,p,v,f,fs,nf,nc,dep);
    else bfs_expand_block<false><<<fs, 256, 0, s>>>(off,idx,em,d,p,v,f,fs,nf,nc,dep);
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

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    const uint32_t* d_em = graph.edge_mask;
    bool compute_pred = (predecessors != nullptr);

    if (depth_limit < 0) depth_limit = std::numeric_limits<int32_t>::max();

    int32_t bw = (num_vertices + 31) / 32;
    cache.ensure(num_vertices);

    uint32_t* d_vis = cache.visited;
    int32_t* cur = cache.frontier_a;
    int32_t* nxt = cache.frontier_b;
    int32_t* ccnt = cache.count_a;
    int32_t* ncnt = cache.count_b;

    cudaStream_t stream = 0;

    
    
    int grid = std::min(std::max(num_vertices / 32 + 1, 80), 3200);

    launch_bfs_init(distances, predecessors, d_vis, num_vertices, bw, compute_pred, stream);
    launch_bfs_set_sources(distances, predecessors, d_vis, cur, ccnt, sources, (int32_t)n_sources, compute_pred, stream);

    
    int check_interval = 1;
    int32_t last_count = (int32_t)n_sources;

    for (int32_t depth = 1; depth <= depth_limit; ) {
        int32_t batch_end = std::min((int64_t)depth + check_interval, (int64_t)depth_limit + 1);

        for (int32_t d = depth; d < batch_end; d++) {
            cudaMemsetAsync(ncnt, 0, sizeof(int32_t), stream);
            launch_bfs_expand_warp(d_off, d_idx, d_em, distances, predecessors, d_vis,
                cur, ccnt, nxt, ncnt, d, compute_pred, grid, stream);
            auto* t = cur; cur = nxt; nxt = t;
            t = ccnt; ccnt = ncnt; ncnt = t;
        }

        int32_t h_count;
        cudaMemcpyAsync(&h_count, ccnt, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (h_count == 0) break;

        last_count = h_count;
        depth = batch_end;

        
        if (check_interval < 128) check_interval *= 2;
    }
}

}  
