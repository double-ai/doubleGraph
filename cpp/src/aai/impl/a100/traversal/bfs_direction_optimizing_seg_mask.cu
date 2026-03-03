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
#include <algorithm>

namespace aai {

namespace {

using int32 = int32_t;
using uint32 = uint32_t;

struct Cache : Cacheable {
    uint32_t* visited = nullptr;
    int32_t* frontier1 = nullptr;
    int32_t* frontier2 = nullptr;
    int32_t* counters = nullptr;
    unsigned long long* degree_sum = nullptr;

    int64_t visited_capacity = 0;
    int64_t frontier1_capacity = 0;
    int64_t frontier2_capacity = 0;
    bool counters_allocated = false;
    bool degree_sum_allocated = false;

    void ensure(int32_t nv) {
        int32_t bm_words = (nv + 31) / 32;
        if (visited_capacity < bm_words) {
            if (visited) cudaFree(visited);
            cudaMalloc(&visited, bm_words * sizeof(uint32_t));
            visited_capacity = bm_words;
        }
        if (frontier1_capacity < nv) {
            if (frontier1) cudaFree(frontier1);
            cudaMalloc(&frontier1, nv * sizeof(int32_t));
            frontier1_capacity = nv;
        }
        if (frontier2_capacity < nv) {
            if (frontier2) cudaFree(frontier2);
            cudaMalloc(&frontier2, nv * sizeof(int32_t));
            frontier2_capacity = nv;
        }
        if (!counters_allocated) {
            cudaMalloc(&counters, 4 * sizeof(int32_t));
            counters_allocated = true;
        }
        if (!degree_sum_allocated) {
            cudaMalloc(&degree_sum, sizeof(unsigned long long));
            degree_sum_allocated = true;
        }
    }

    ~Cache() override {
        if (visited) cudaFree(visited);
        if (frontier1) cudaFree(frontier1);
        if (frontier2) cudaFree(frontier2);
        if (counters) cudaFree(counters);
        if (degree_sum) cudaFree(degree_sum);
    }
};



__device__ __forceinline__ bool edge_active(const uint32* __restrict__ em, int32 e) {
    return (em[e >> 5] >> (e & 31)) & 1;
}

__device__ __forceinline__ bool bm_get(const uint32* __restrict__ bm, int32 v) {
    return (bm[v >> 5] >> (v & 31)) & 1;
}

__device__ __forceinline__ bool bm_try_set(uint32* bm, int32 v) {
    uint32 mask = 1u << (v & 31);
    uint32 old = atomicOr(&bm[v >> 5], mask);
    return !(old & mask);
}



__global__ void init_kernel(int32* __restrict__ dist, int32* __restrict__ pred,
                             uint32* __restrict__ visited, int32 nv,
                             int32 bm_words, bool do_pred) {
    int32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32 stride = gridDim.x * blockDim.x;
    for (int32 i = idx; i < nv; i += stride) {
        dist[i] = INT32_MAX;
        if (do_pred) pred[i] = -1;
    }
    for (int32 i = idx; i < bm_words; i += stride) {
        visited[i] = 0;
    }
}

__global__ void set_sources_kernel(const int32* __restrict__ src, int32 ns,
                                    int32* __restrict__ dist, int32* __restrict__ pred,
                                    uint32* __restrict__ visited, int32* __restrict__ frontier,
                                    int32* __restrict__ fsize, bool do_pred) {
    int32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ns) return;
    int32 s = src[idx];
    dist[s] = 0;
    if (do_pred) pred[s] = -1;
    if (bm_try_set(visited, s)) {
        int32 pos = atomicAdd(fsize, 1);
        frontier[pos] = s;
    }
}



__global__ void __launch_bounds__(256, 8)
top_down_warp_kernel(
    const int32* __restrict__ offsets,
    const int32* __restrict__ indices,
    const uint32* __restrict__ edge_mask,
    int32* __restrict__ dist,
    int32* __restrict__ pred,
    uint32* __restrict__ visited,
    const int32* __restrict__ frontier,
    int32 fsize,
    int32* __restrict__ next_frontier,
    int32* __restrict__ next_fsize,
    int32 depth,
    bool do_pred)
{
    int32 lane = threadIdx.x & 31;
    int32 warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32 total_warps = (gridDim.x * blockDim.x) >> 5;
    int32 nd = depth + 1;
    uint32 lane_mask_lt = (1u << lane) - 1;

    for (int32 fi = warp_id; fi < fsize; fi += total_warps) {
        int32 v = frontier[fi];
        int32 start = offsets[v];
        int32 end = offsets[v + 1];
        int32 degree = end - start;
        int32 iters = (degree + 31) >> 5;

        for (int32 iter = 0; iter < iters; iter++) {
            int32 e = start + iter * 32 + lane;
            bool found = false;
            int32 u = -1;

            if (e < end && edge_active(edge_mask, e)) {
                u = indices[e];
                found = bm_try_set(visited, u);
            }

            
            uint32 ballot = __ballot_sync(0xFFFFFFFF, found);
            if (ballot != 0) {
                
                int32 count = __popc(ballot);
                int32 warp_base;
                if (lane == 0) {
                    warp_base = atomicAdd(next_fsize, count);
                }
                warp_base = __shfl_sync(0xFFFFFFFF, warp_base, 0);

                if (found) {
                    int32 my_offset = __popc(ballot & lane_mask_lt);
                    next_frontier[warp_base + my_offset] = u;
                    dist[u] = nd;
                    if (do_pred) pred[u] = v;
                }
            }
        }
    }
}



__global__ void __launch_bounds__(256, 8)
bottom_up_kernel(
    const int32* __restrict__ offsets,
    const int32* __restrict__ indices,
    const uint32* __restrict__ edge_mask,
    int32* __restrict__ dist,
    int32* __restrict__ pred,
    const uint32* __restrict__ visited,
    int32* __restrict__ next_frontier,
    int32* __restrict__ next_fsize,
    int32 nv,
    int32 depth,
    bool do_pred)
{
    int32 v = blockIdx.x * blockDim.x + threadIdx.x;
    int32 lane = threadIdx.x & 31;
    uint32 lane_mask_lt = (1u << lane) - 1;
    int32 nd = depth + 1;

    bool found_parent = false;
    int32 parent = -1;

    if (v < nv && !bm_get(visited, v)) {
        int32 start = offsets[v];
        int32 end = offsets[v + 1];
        for (int32 e = start; e < end; e++) {
            if (!edge_active(edge_mask, e)) continue;
            int32 u = indices[e];
            if (bm_get(visited, u)) {
                found_parent = true;
                parent = u;
                break;
            }
        }
    }

    
    uint32 ballot = __ballot_sync(0xFFFFFFFF, found_parent);
    if (ballot != 0) {
        int32 count = __popc(ballot);
        int32 warp_base;
        if (lane == 0) {
            warp_base = atomicAdd(next_fsize, count);
        }
        warp_base = __shfl_sync(0xFFFFFFFF, warp_base, 0);

        if (found_parent) {
            int32 my_offset = __popc(ballot & lane_mask_lt);
            next_frontier[warp_base + my_offset] = v;
            dist[v] = nd;
            if (do_pred) pred[v] = parent;
        }
    }
}

__global__ void mark_visited_kernel(uint32* __restrict__ visited,
                                     const int32* __restrict__ frontier, int32 fsize) {
    int32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fsize) return;
    int32 v = frontier[idx];
    atomicOr(&visited[v >> 5], 1u << (v & 31));
}

__global__ void compute_degree_sum_kernel(
    const int32* __restrict__ offsets,
    const int32* __restrict__ frontier,
    int32 fsize,
    unsigned long long* __restrict__ result)
{
    unsigned long long sum = 0;
    int32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32 stride = gridDim.x * blockDim.x;
    for (int32 i = tid; i < fsize; i += stride) {
        int32 v = frontier[i];
        sum += (unsigned long long)(offsets[v + 1] - offsets[v]);
    }
    
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(result, sum);
}



void launch_init(int32* d, int32* p, uint32* v, int32 nv, int32 bw, bool dp) {
    int b = 256, n = nv > bw ? nv : bw;
    init_kernel<<<(n+b-1)/b, b>>>(d,p,v,nv,bw,dp);
}

void launch_set_sources(const int32* s, int32 ns, int32* d, int32* p,
                         uint32* v, int32* f, int32* fs, bool dp) {
    int b = 256;
    set_sources_kernel<<<(ns+b-1)/b, b>>>(s,ns,d,p,v,f,fs,dp);
}

void launch_top_down(const int32* off, const int32* idx, const uint32* em,
                      int32* d, int32* p, uint32* v,
                      const int32* f, int32 fs, int32* nf, int32* nfs,
                      int32 depth, bool dp) {
    if (fs == 0) return;
    int b = 256;
    int warps_per_block = b / 32;
    int grid = (fs + warps_per_block - 1) / warps_per_block;
    top_down_warp_kernel<<<grid, b>>>(off,idx,em,d,p,v,f,fs,nf,nfs,depth,dp);
}

void launch_bottom_up(const int32* off, const int32* idx, const uint32* em,
                       int32* d, int32* p, const uint32* v,
                       int32* nf, int32* nfs, int32 nv, int32 depth, bool dp) {
    int b = 256;
    bottom_up_kernel<<<(nv+b-1)/b, b>>>(off,idx,em,d,p,v,nf,nfs,nv,depth,dp);
}

void launch_mark_visited(uint32* v, const int32* f, int32 fs) {
    if (fs == 0) return;
    int b = 256;
    mark_visited_kernel<<<(fs+b-1)/b, b>>>(v,f,fs);
}

void launch_compute_degree_sum(const int32* off, const int32* f, int32 fs, unsigned long long* r) {
    if (fs == 0) return;
    int b = 256, g = (fs+b-1)/b;
    if (g > 512) g = 512;
    compute_degree_sum_kernel<<<g,b>>>(off,f,fs,r);
}

}  

void bfs_direction_optimizing_seg_mask(const graph32_t& graph,
                                       int32_t* distances,
                                       int32_t* predecessors,
                                       const int32_t* sources,
                                       std::size_t n_sources,
                                       int32_t depth_limit) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    const uint32_t* d_em = graph.edge_mask;

    bool do_pred = (predecessors != nullptr);

    if (depth_limit < 0) depth_limit = std::numeric_limits<int32_t>::max();

    cache.ensure(nv);

    int32_t* d_dist = distances;
    int32_t* d_pred = predecessors;

    int32_t bm_words = (nv + 31) / 32;
    uint32_t* d_vis = cache.visited;

    int32_t* d_fin = cache.frontier1;
    int32_t* d_fout = cache.frontier2;

    int32_t* d_cnt = cache.counters;
    unsigned long long* d_ds = cache.degree_sum;

    int32_t ns = static_cast<int32_t>(n_sources);

    
    launch_init(d_dist, d_pred, d_vis, nv, bm_words, do_pred);
    cudaMemset(d_cnt, 0, 4 * sizeof(int32_t));

    launch_set_sources(sources, ns, d_dist, d_pred, d_vis, d_fin, d_cnt, do_pred);

    int32_t h_fsize = 0;
    cudaMemcpy(&h_fsize, d_cnt, sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    double avg_deg = (nv > 0) ? static_cast<double>(ne) / nv : 0.0;
    double alpha = avg_deg * 0.5;
    constexpr int32_t beta = 24;

    bool is_td = true;
    int32_t depth = 0;

    while (h_fsize > 0) {
        cudaMemset(d_cnt + 1, 0, sizeof(int32_t));

        if (is_td) {
            launch_top_down(d_off, d_idx, d_em, d_dist, d_pred, d_vis,
                            d_fin, h_fsize, d_fout, d_cnt + 1, depth, do_pred);
        } else {
            launch_bottom_up(d_off, d_idx, d_em, d_dist, d_pred, d_vis,
                             d_fout, d_cnt + 1, nv, depth, do_pred);
        }

        int32_t h_next_fsize = 0;
        cudaMemcpy(&h_next_fsize, d_cnt + 1, sizeof(int32_t), cudaMemcpyDeviceToHost);

        if (!is_td && h_next_fsize > 0) {
            launch_mark_visited(d_vis, d_fout, h_next_fsize);
        }

        
        if (is_td && h_next_fsize >= h_fsize && h_next_fsize > 0) {
            
            cudaMemset(d_ds, 0, sizeof(unsigned long long));
            launch_compute_degree_sum(d_off, d_fout, h_next_fsize, d_ds);
            unsigned long long h_mf = 0;
            cudaMemcpy(&h_mf, d_ds, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            double m_f = static_cast<double>(h_mf);

            if (m_f * alpha > static_cast<double>(ne) * 0.5) {
                is_td = false;
            }
        } else if (!is_td && h_next_fsize < h_fsize) {
            if (static_cast<long long>(h_next_fsize) * beta < nv) {
                is_td = true;
            }
        }

        std::swap(d_fin, d_fout);
        h_fsize = h_next_fsize;
        depth++;
        if (depth_limit != std::numeric_limits<int32_t>::max() && depth >= depth_limit) break;
    }
}

}  
