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
#include <cub/cub.cuh>
#include <algorithm>

namespace aai {

namespace {


__device__ __forceinline__ bool edge_active(const uint32_t* __restrict__ mask, int32_t idx) {
    return (__ldg(&mask[idx >> 5]) >> (idx & 31)) & 1u;
}
__device__ __forceinline__ bool check_visited(const uint32_t* __restrict__ bm, int32_t v) {
    return (__ldg(&bm[v >> 5]) >> (v & 31)) & 1u;
}


__global__ void bfs_init_kernel(int32_t* __restrict__ dist, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        dist[i] = 0x7FFFFFFF;
    }
}


__global__ void bfs_set_sources_kernel(
    const int32_t* __restrict__ src, int32_t nsrc,
    int32_t* __restrict__ dist, int32_t* __restrict__ pred,
    uint32_t* __restrict__ visited, int32_t* __restrict__ frontier, bool do_pred
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nsrc) {
        int32_t s = src[i];
        dist[s] = 0;
        if (do_pred) pred[s] = -1;
        atomicOr(&visited[s >> 5], 1u << (s & 31));
        frontier[i] = s;
    }
}


__global__ __launch_bounds__(256, 8)
void bfs_td_warp_kernel(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ idx,
    const uint32_t* __restrict__ emask,
    int32_t* __restrict__ dist,
    int32_t* __restrict__ pred,
    uint32_t* __restrict__ visited,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_f,
    int32_t* __restrict__ next_f_size,
    int32_t f_size, int32_t depth, bool do_pred
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= f_size) return;

    int32_t src = frontier[warp_id];
    int32_t start = __ldg(&off[src]);
    int32_t end = __ldg(&off[src + 1]);
    int32_t nd = depth + 1;

    for (int32_t e = start + lane; e < end; e += 32) {
        uint32_t mask_word = __ldg(&emask[e >> 5]);
        if (!((mask_word >> (e & 31)) & 1u)) continue;

        int32_t dst = __ldg(&idx[e]);
        uint32_t wi = dst >> 5;
        uint32_t bm = 1u << (dst & 31);
        uint32_t old = atomicOr(&visited[wi], bm);
        if (!(old & bm)) {
            dist[dst] = nd;
            if (do_pred) pred[dst] = src;
            int32_t pos = atomicAdd(next_f_size, 1);
            next_f[pos] = dst;
        }
    }
}


__global__ __launch_bounds__(512, 4)
void bfs_bu_kernel(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ idx,
    const uint32_t* __restrict__ emask,
    int32_t* __restrict__ dist,
    int32_t* __restrict__ pred,
    const uint32_t* __restrict__ visited,
    int32_t* __restrict__ next_f,
    int32_t* __restrict__ next_f_size,
    int32_t num_v, int32_t depth, bool do_pred
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_v) return;
    if (check_visited(visited, v)) return;

    int32_t start = __ldg(&off[v]);
    int32_t end = __ldg(&off[v + 1]);
    if (start >= end) return;
    int32_t nd = depth + 1;

    for (int32_t e = start; e < end; e++) {
        if (!edge_active(emask, e)) continue;
        int32_t u = __ldg(&idx[e]);
        if (check_visited(visited, u)) {
            dist[v] = nd;
            if (do_pred) pred[v] = u;
            int32_t pos = atomicAdd(next_f_size, 1);
            next_f[pos] = v;
            return;
        }
    }
}


__global__ void bfs_mark_visited_kernel(
    const int32_t* __restrict__ frontier, int32_t f_size,
    uint32_t* __restrict__ visited
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < f_size) {
        int32_t v = frontier[i];
        atomicOr(&visited[v >> 5], 1u << (v & 31));
    }
}


__global__ void compute_mf_kernel(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ frontier,
    int32_t f_size, int64_t* __restrict__ result
) {
    typedef cub::BlockReduce<int64_t, 256> BR;
    __shared__ typename BR::TempStorage temp;
    int64_t sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < f_size; i += blockDim.x * gridDim.x) {
        int32_t v = frontier[i];
        sum += __ldg(&off[v + 1]) - __ldg(&off[v]);
    }
    int64_t bs = BR(temp).Sum(sum);
    if (threadIdx.x == 0) atomicAdd((unsigned long long*)result, (unsigned long long)bs);
}


__global__ void compute_mu_kernel(
    const int32_t* __restrict__ off,
    const uint32_t* __restrict__ visited,
    int32_t num_v, int64_t* __restrict__ result
) {
    typedef cub::BlockReduce<int64_t, 256> BR;
    __shared__ typename BR::TempStorage temp;
    int64_t sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_v; i += blockDim.x * gridDim.x) {
        if (!check_visited(visited, i)) {
            sum += __ldg(&off[i + 1]) - __ldg(&off[i]);
        }
    }
    int64_t bs = BR(temp).Sum(sum);
    if (threadIdx.x == 0) atomicAdd((unsigned long long*)result, (unsigned long long)bs);
}



void launch_bfs_init(int32_t* dist, int32_t n, cudaStream_t s) {
    if (n <= 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 2048) grid = 2048;
    bfs_init_kernel<<<grid, block, 0, s>>>(dist, n);
}

void launch_bfs_set_sources(const int32_t* src, int32_t nsrc,
    int32_t* dist, int32_t* pred, uint32_t* visited, int32_t* frontier,
    bool do_pred, cudaStream_t s) {
    if (nsrc <= 0) return;
    bfs_set_sources_kernel<<<(nsrc+255)/256, 256, 0, s>>>(src, nsrc, dist, pred, visited, frontier, do_pred);
}

void launch_bfs_td_warp(
    const int32_t* off, const int32_t* idx, const uint32_t* emask,
    int32_t* dist, int32_t* pred, uint32_t* visited,
    const int32_t* frontier, int32_t* next_f, int32_t* next_f_size,
    int32_t f_size, int32_t depth, bool do_pred, cudaStream_t s) {
    if (f_size <= 0) return;
    int threads = 256;
    int grid = (int)(((int64_t)f_size * 32 + threads - 1) / threads);
    bfs_td_warp_kernel<<<grid, threads, 0, s>>>(off, idx, emask, dist, pred, visited,
        frontier, next_f, next_f_size, f_size, depth, do_pred);
}

void launch_bfs_bu(
    const int32_t* off, const int32_t* idx, const uint32_t* emask,
    int32_t* dist, int32_t* pred, const uint32_t* visited,
    int32_t* next_f, int32_t* next_f_size,
    int32_t num_v, int32_t depth, bool do_pred, cudaStream_t s) {
    if (num_v <= 0) return;
    int block = 512;
    int grid = (num_v + block - 1) / block;
    bfs_bu_kernel<<<grid, block, 0, s>>>(off, idx, emask, dist, pred, visited,
        next_f, next_f_size, num_v, depth, do_pred);
}

void launch_bfs_mark_visited(const int32_t* frontier, int32_t f_size,
    uint32_t* visited, cudaStream_t s) {
    if (f_size <= 0) return;
    bfs_mark_visited_kernel<<<(f_size+255)/256, 256, 0, s>>>(frontier, f_size, visited);
}

void launch_compute_mf(const int32_t* off, const int32_t* frontier,
    int32_t f_size, int64_t* result, cudaStream_t s) {
    if (f_size <= 0) return;
    int grid = (f_size + 255) / 256;
    if (grid > 256) grid = 256;
    compute_mf_kernel<<<grid, 256, 0, s>>>(off, frontier, f_size, result);
}

void launch_compute_mu(const int32_t* off, const uint32_t* visited,
    int32_t num_v, int64_t* result, cudaStream_t s) {
    if (num_v <= 0) return;
    int grid = (num_v + 255) / 256;
    if (grid > 256) grid = 256;
    compute_mu_kernel<<<grid, 256, 0, s>>>(off, visited, num_v, result);
}



struct Cache : Cacheable {
    
    int32_t* h_counter = nullptr;
    int64_t* h_metrics = nullptr;

    
    uint32_t* visited = nullptr;
    int32_t* frontier = nullptr;
    int32_t* next_frontier = nullptr;
    int32_t* counter = nullptr;
    int64_t* d_metrics = nullptr;

    
    int32_t visited_capacity = 0;
    int32_t frontier_capacity = 0;
    int32_t next_frontier_capacity = 0;
    int32_t counter_capacity = 0;
    int32_t metrics_capacity = 0;

    Cache() {
        cudaHostAlloc(&h_counter, sizeof(int32_t), cudaHostAllocDefault);
        cudaHostAlloc(&h_metrics, 2 * sizeof(int64_t), cudaHostAllocDefault);
    }

    void ensure(int32_t num_v) {
        int32_t bm_size = (num_v + 31) / 32;

        if (visited_capacity < bm_size) {
            if (visited) cudaFree(visited);
            cudaMalloc(&visited, bm_size * sizeof(uint32_t));
            visited_capacity = bm_size;
        }
        if (frontier_capacity < num_v) {
            if (frontier) cudaFree(frontier);
            cudaMalloc(&frontier, num_v * sizeof(int32_t));
            frontier_capacity = num_v;
        }
        if (next_frontier_capacity < num_v) {
            if (next_frontier) cudaFree(next_frontier);
            cudaMalloc(&next_frontier, num_v * sizeof(int32_t));
            next_frontier_capacity = num_v;
        }
        if (counter_capacity < 1) {
            if (counter) cudaFree(counter);
            cudaMalloc(&counter, sizeof(int32_t));
            counter_capacity = 1;
        }
        if (metrics_capacity < 2) {
            if (d_metrics) cudaFree(d_metrics);
            cudaMalloc(&d_metrics, 2 * sizeof(int64_t));
            metrics_capacity = 2;
        }
    }

    ~Cache() override {
        if (h_counter) cudaFreeHost(h_counter);
        if (h_metrics) cudaFreeHost(h_metrics);
        if (visited) cudaFree(visited);
        if (frontier) cudaFree(frontier);
        if (next_frontier) cudaFree(next_frontier);
        if (counter) cudaFree(counter);
        if (d_metrics) cudaFree(d_metrics);
    }
};

}  

void bfs_direction_optimizing_seg_mask(const graph32_t& graph,
                                       int32_t* distances,
                                       int32_t* predecessors,
                                       const int32_t* sources,
                                       std::size_t n_sources,
                                       int32_t depth_limit) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_v = graph.number_of_vertices;
    int32_t num_e = graph.number_of_edges;
    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    const uint32_t* d_emask = graph.edge_mask;

    bool do_pred = (predecessors != nullptr);
    int32_t nsrc = static_cast<int32_t>(n_sources);
    if (depth_limit < 0) depth_limit = 0x7FFFFFFF;

    
    const auto& seg = graph.segment_offsets.value();
    int32_t num_nonzero = seg[3]; 

    cache.ensure(num_v);

    int32_t bm_size = (num_v + 31) / 32;
    uint32_t* d_visited = cache.visited;
    int32_t* d_frontier = cache.frontier;
    int32_t* d_next_f = cache.next_frontier;
    int32_t* d_cnt = cache.counter;
    int64_t* d_metrics_dev = cache.d_metrics;

    int32_t* d_dist = distances;
    int32_t* d_pred = predecessors;

    cudaStream_t stream = 0;

    
    cudaMemsetAsync(d_visited, 0, bm_size * sizeof(uint32_t), stream);
    launch_bfs_init(d_dist, num_v, stream);
    if (do_pred) {
        cudaMemsetAsync(d_pred, 0xFF, (int64_t)num_v * sizeof(int32_t), stream);
    }
    launch_bfs_set_sources(sources, nsrc, d_dist, d_pred, d_visited, d_frontier, do_pred, stream);

    int32_t frontier_size = nsrc;
    int32_t current_depth = 0;
    bool is_topdown = true;
    int32_t total_visited = nsrc;

    double alpha = (static_cast<double>(num_e) / num_v) * 0.267;
    constexpr int32_t beta = 24;

    while (frontier_size > 0) {
        cudaMemsetAsync(d_cnt, 0, sizeof(int32_t), stream);

        if (is_topdown) {
            launch_bfs_td_warp(d_off, d_idx, d_emask, d_dist, d_pred, d_visited,
                d_frontier, d_next_f, d_cnt, frontier_size, current_depth, do_pred, stream);
        } else {
            
            launch_bfs_bu(d_off, d_idx, d_emask, d_dist, d_pred, d_visited,
                d_next_f, d_cnt, num_nonzero, current_depth, do_pred, stream);
        }

        cudaMemcpyAsync(cache.h_counter, d_cnt, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int32_t next_fs = *cache.h_counter;

        if (!is_topdown && next_fs > 0) {
            launch_bfs_mark_visited(d_next_f, next_fs, d_visited, stream);
        }

        total_visited += next_fs;

        
        if (is_topdown && next_fs > 0 && next_fs >= frontier_size) {
            cudaMemsetAsync(d_metrics_dev, 0, 2 * sizeof(int64_t), stream);
            launch_compute_mf(d_off, d_next_f, next_fs, &d_metrics_dev[0], stream);
            launch_compute_mu(d_off, d_visited, num_v, &d_metrics_dev[1], stream);
            cudaMemcpyAsync(cache.h_metrics, d_metrics_dev, 2 * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (static_cast<double>(cache.h_metrics[0]) * alpha > static_cast<double>(cache.h_metrics[1])) {
                is_topdown = false;
            }
        } else if (!is_topdown && next_fs < frontier_size) {
            int32_t unvisited = num_v - total_visited;
            if (static_cast<int64_t>(next_fs) * beta < static_cast<int64_t>(unvisited)) {
                is_topdown = true;
            }
        }

        std::swap(d_frontier, d_next_f);
        frontier_size = next_fs;
        current_depth++;
        if (depth_limit != 0x7FFFFFFF && current_depth >= depth_limit) break;
    }
}

}  
