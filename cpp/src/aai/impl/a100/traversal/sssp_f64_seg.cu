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
#include <cfloat>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* in_frontier = nullptr;
    int32_t* frontier_size_dev = nullptr;
    int32_t* work_counter = nullptr;
    int32_t* h_frontier_size_pinned = nullptr;
    int num_sms = 0;

    int64_t frontier_a_capacity = 0;
    int64_t frontier_b_capacity = 0;
    int64_t in_frontier_capacity = 0;
    bool frontier_size_dev_allocated = false;
    bool work_counter_allocated = false;

    Cache() {
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        cudaMallocHost(&h_frontier_size_pinned, sizeof(int32_t));
        cudaMalloc(&frontier_size_dev, sizeof(int32_t));
        frontier_size_dev_allocated = true;
        cudaMalloc(&work_counter, sizeof(int32_t));
        work_counter_allocated = true;
    }

    void ensure(int64_t num_vertices) {
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
        if (in_frontier_capacity < num_vertices) {
            if (in_frontier) cudaFree(in_frontier);
            cudaMalloc(&in_frontier, num_vertices * sizeof(int32_t));
            in_frontier_capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (in_frontier) cudaFree(in_frontier);
        if (frontier_size_dev) cudaFree(frontier_size_dev);
        if (work_counter) cudaFree(work_counter);
        if (h_frontier_size_pinned) cudaFreeHost(h_frontier_size_pinned);
    }
};

__global__ __launch_bounds__(256, 8)
void relax_edges_worksteal_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    double* __restrict__ dist,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t* __restrict__ in_next_frontier,
    int32_t* __restrict__ work_counter,
    double cutoff
) {
    __shared__ int32_t s_block_start;

    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const unsigned long long cutoff_ull = __double_as_longlong(cutoff);

    const int BLOCK_CHUNK = warps_per_block * 4;

    while (true) {
        if (threadIdx.x == 0) {
            s_block_start = atomicAdd(work_counter, BLOCK_CHUNK);
        }
        __syncthreads();

        int32_t block_start = s_block_start;
        if (block_start >= frontier_size) break;

        int32_t block_end = block_start + BLOCK_CHUNK;
        if (block_end > frontier_size) block_end = frontier_size;

        for (int32_t vi = block_start + warp_in_block; vi < block_end; vi += warps_per_block) {
            int32_t u = frontier[vi];
            double dist_u = dist[u];

            if (dist_u >= cutoff) continue;

            int32_t start = __ldg(&offsets[u]);
            int32_t end = __ldg(&offsets[u + 1]);

            for (int32_t e = start + lane; e < end; e += 32) {
                int32_t v = __ldg(&indices[e]);
                double new_dist = dist_u + __ldg(&weights[e]);
                unsigned long long new_dist_ull = __double_as_longlong(new_dist);

                if (new_dist_ull < cutoff_ull) {
                    unsigned long long old_dist_ull = atomicMin(
                        (unsigned long long*)&dist[v], new_dist_ull);

                    if (old_dist_ull > new_dist_ull) {
                        int old_flag = atomicExch(&in_next_frontier[v], 1);
                        if (old_flag == 0) {
                            int pos = atomicAdd(next_frontier_size, 1);
                            next_frontier[pos] = v;
                        }
                    }
                }
            }
        }
    }
}

__global__ __launch_bounds__(256, 8)
void relax_edges_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    double* __restrict__ dist,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t* __restrict__ in_next_frontier,
    double cutoff
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t u = frontier[warp_id];
    double dist_u = dist[u];

    if (dist_u >= cutoff) return;

    int32_t start = __ldg(&offsets[u]);
    int32_t end = __ldg(&offsets[u + 1]);
    const unsigned long long cutoff_ull = __double_as_longlong(cutoff);

    for (int32_t e = start + lane; e < end; e += 32) {
        int32_t v = __ldg(&indices[e]);
        double new_dist = dist_u + __ldg(&weights[e]);
        unsigned long long new_dist_ull = __double_as_longlong(new_dist);

        if (new_dist_ull < cutoff_ull) {
            unsigned long long old_dist_ull = atomicMin(
                (unsigned long long*)&dist[v], new_dist_ull);

            if (old_dist_ull > new_dist_ull) {
                int old_flag = atomicExch(&in_next_frontier[v], 1);
                if (old_flag == 0) {
                    int pos = atomicAdd(next_frontier_size, 1);
                    next_frontier[pos] = v;
                }
            }
        }
    }
}

__global__ void init_sssp_kernel(
    double* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t num_vertices,
    int32_t source
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        dist[tid] = (tid == source) ? 0.0 : DBL_MAX;
        pred[tid] = -1;
    }
}

__global__ void clear_frontier_flags_kernel(
    int32_t* __restrict__ in_frontier,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < frontier_size) {
        in_frontier[frontier[tid]] = 0;
    }
}

__global__ __launch_bounds__(256, 8)
void compute_predecessors_positive_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t num_vertices,
    int32_t source
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int total_warps = (gridDim.x * blockDim.x) >> 5;

    for (int u = warp_id; u < num_vertices; u += total_warps) {
        double dist_u = dist[u];
        if (dist_u == DBL_MAX) continue;

        int32_t start = offsets[u];
        int32_t end = offsets[u + 1];

        for (int32_t e = start + lane; e < end; e += 32) {
            int32_t v = indices[e];
            if (v == source) continue;
            double w = weights[e];
            if (dist_u + w == dist[v] && dist_u < dist[v]) {
                pred[v] = u;
            }
        }
    }
}

__global__ void compute_predecessors_zero_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t num_vertices,
    int32_t source,
    int32_t* __restrict__ changed
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;
    if (u != source && pred[u] == -1) return;

    double dist_u = dist[u];
    if (dist_u == DBL_MAX) return;

    int32_t start = offsets[u];
    int32_t end = offsets[u + 1];

    for (int32_t e = start; e < end; e++) {
        int32_t v = indices[e];
        if (v == source) continue;
        double w = weights[e];
        if (dist_u + w == dist[v] && dist[v] == dist_u && pred[v] == -1) {
            pred[v] = u;
            *changed = 1;
        }
    }
}

}  

void sssp_seg(const graph32_t& graph,
              const double* edge_weights,
              int32_t source,
              double* distances,
              int32_t* predecessors,
              double cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    cache.ensure(num_vertices);

    int32_t* d_frontier_a = cache.frontier_a;
    int32_t* d_frontier_b = cache.frontier_b;
    int32_t* d_in_frontier = cache.in_frontier;
    int32_t* d_frontier_size = cache.frontier_size_dev;
    int32_t* d_work_counter = cache.work_counter;

    cudaStream_t stream = 0;

    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    init_sssp_kernel<<<grid, block, 0, stream>>>(distances, predecessors, num_vertices, source);
    cudaMemcpyAsync(d_frontier_a, &source, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(d_in_frontier, 0, num_vertices * sizeof(int32_t), stream);

    int32_t* frontier_current = d_frontier_a;
    int32_t* frontier_next = d_frontier_b;
    int32_t h_frontier_size = 1;

    const int ws_grid_size = cache.num_sms * 8;
    const int WS_THRESHOLD = 1024;

    for (int iter = 0; h_frontier_size > 0; iter++) {
        cudaMemsetAsync(d_frontier_size, 0, sizeof(int32_t), stream);

        if (h_frontier_size >= WS_THRESHOLD) {
            cudaMemsetAsync(d_work_counter, 0, sizeof(int32_t), stream);
            relax_edges_worksteal_kernel<<<ws_grid_size, 256, 0, stream>>>(
                d_offsets, d_indices, edge_weights, distances,
                frontier_current, h_frontier_size,
                frontier_next, d_frontier_size, d_in_frontier,
                d_work_counter, cutoff
            );
        } else {
            int threads_per_block = 256;
            int64_t total_threads = (int64_t)h_frontier_size * 32;
            int warp_grid = (int)((total_threads + threads_per_block - 1) / threads_per_block);
            relax_edges_warp_kernel<<<warp_grid, threads_per_block, 0, stream>>>(
                d_offsets, d_indices, edge_weights, distances,
                frontier_current, h_frontier_size,
                frontier_next, d_frontier_size, d_in_frontier,
                cutoff
            );
        }

        cudaMemcpyAsync(cache.h_frontier_size_pinned, d_frontier_size, sizeof(int32_t),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        h_frontier_size = *cache.h_frontier_size_pinned;

        if (h_frontier_size > 0) {
            int clear_grid = (h_frontier_size + block - 1) / block;
            clear_frontier_flags_kernel<<<clear_grid, block, 0, stream>>>(
                d_in_frontier, frontier_next, h_frontier_size);
        }

        int32_t* temp = frontier_current;
        frontier_current = frontier_next;
        frontier_next = temp;
    }

    
    {
        int pred_grid = (num_vertices + block - 1) / block;
        compute_predecessors_positive_kernel<<<pred_grid, block, 0, stream>>>(
            d_offsets, d_indices, edge_weights,
            distances, predecessors, num_vertices, source);
    }

    
    {
        int pred_grid = (num_vertices + block - 1) / block;
        int32_t h_changed = 1;
        for (int bfs_iter = 0; bfs_iter < num_vertices && h_changed; bfs_iter++) {
            cudaMemsetAsync(d_frontier_size, 0, sizeof(int32_t), stream);
            compute_predecessors_zero_kernel<<<pred_grid, block, 0, stream>>>(
                d_offsets, d_indices, edge_weights,
                distances, predecessors, num_vertices, source,
                d_frontier_size);
            cudaMemcpyAsync(&h_changed, d_frontier_size, sizeof(int32_t),
                             cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
    }

    cudaStreamSynchronize(stream);
}

}  
