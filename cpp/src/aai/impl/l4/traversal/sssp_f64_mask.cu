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
#include <cstdint>
#include <cub/cub.cuh>

namespace aai {

namespace {

#define UNREACHABLE_DIST DBL_MAX
#define BLOCK_SIZE 256
#define WARP_SIZE 32

struct Cache : Cacheable {
    int32_t* d_frontier1 = nullptr;
    int32_t* d_frontier2 = nullptr;
    int32_t* d_queued = nullptr;
    int32_t* d_size1 = nullptr;
    int32_t* d_size2 = nullptr;
    int32_t* d_work_counter = nullptr;
    int32_t* h_frontier_size = nullptr;

    int32_t frontier1_capacity = 0;
    int32_t frontier2_capacity = 0;
    int32_t queued_capacity = 0;

    Cache() {
        cudaMallocHost(&h_frontier_size, sizeof(int32_t));
        cudaMalloc(&d_size1, sizeof(int32_t));
        cudaMalloc(&d_size2, sizeof(int32_t));
        cudaMalloc(&d_work_counter, sizeof(int32_t));
    }

    ~Cache() override {
        if (d_frontier1) cudaFree(d_frontier1);
        if (d_frontier2) cudaFree(d_frontier2);
        if (d_queued) cudaFree(d_queued);
        if (d_size1) cudaFree(d_size1);
        if (d_size2) cudaFree(d_size2);
        if (d_work_counter) cudaFree(d_work_counter);
        if (h_frontier_size) cudaFreeHost(h_frontier_size);
    }

    void ensure(int32_t n) {
        if (frontier1_capacity < n) {
            if (d_frontier1) cudaFree(d_frontier1);
            cudaMalloc(&d_frontier1, n * sizeof(int32_t));
            frontier1_capacity = n;
        }
        if (frontier2_capacity < n) {
            if (d_frontier2) cudaFree(d_frontier2);
            cudaMalloc(&d_frontier2, n * sizeof(int32_t));
            frontier2_capacity = n;
        }
        if (queued_capacity < n) {
            if (d_queued) cudaFree(d_queued);
            cudaMalloc(&d_queued, n * sizeof(int32_t));
            queued_capacity = n;
        }
    }
};

__device__ __forceinline__ double atomicMinDouble(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val) return __longlong_as_double(old);
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void init_sssp_kernel(double* __restrict__ dist, int32_t* __restrict__ pred,
                                  int32_t* __restrict__ queued,
                                  int32_t n, int32_t source,
                                  int32_t* __restrict__ frontier, int32_t* __restrict__ frontier_size) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dist[i] = (i == source) ? 0.0 : UNREACHABLE_DIST;
        pred[i] = -1;
        queued[i] = 0;
    }
    if (i == 0) {
        frontier[0] = source;
        *frontier_size = 1;
        queued[source] = 1;
    }
}

__global__ void __launch_bounds__(BLOCK_SIZE)
relax_edges_blockcoop_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ dist,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t* __restrict__ queued,
    double cutoff,
    int32_t generation,
    int32_t* __restrict__ work_counter
) {
    typedef cub::BlockScan<int32_t, BLOCK_SIZE> BlockScan;

    __shared__ typename BlockScan::TempStorage scan_temp;
    __shared__ int32_t s_vertex[BLOCK_SIZE];
    __shared__ int32_t s_edge_start[BLOCK_SIZE];
    __shared__ int32_t s_prefix[BLOCK_SIZE + 1];
    __shared__ double  s_dist[BLOCK_SIZE];
    __shared__ int32_t s_chunk_base;

    const int32_t tid = threadIdx.x;

    while (true) {
        if (tid == 0) {
            s_chunk_base = atomicAdd(work_counter, BLOCK_SIZE);
        }
        __syncthreads();

        int32_t chunk_base = s_chunk_base;
        if (chunk_base >= frontier_size) break;

        int32_t my_idx = chunk_base + tid;
        int32_t seg_length = 0;

        if (my_idx < frontier_size) {
            int32_t u = frontier[my_idx];
            s_vertex[tid] = u;
            int32_t start = __ldg(&offsets[u]);
            int32_t end = __ldg(&offsets[u + 1]);
            s_edge_start[tid] = start;
            seg_length = end - start;
            s_dist[tid] = dist[u];
        } else {
            s_vertex[tid] = -1;
            s_edge_start[tid] = 0;
            s_dist[tid] = UNREACHABLE_DIST;
        }

        int32_t prefix;
        int32_t aggregate;
        BlockScan(scan_temp).ExclusiveSum(seg_length, prefix, aggregate);
        __syncthreads();

        s_prefix[tid] = prefix;
        if (tid == 0) {
            s_prefix[BLOCK_SIZE] = aggregate;
        }
        __syncthreads();

        if (aggregate == 0) continue;

        int32_t valid_segs = frontier_size - chunk_base;
        if (valid_segs > BLOCK_SIZE) valid_segs = BLOCK_SIZE;

        for (int32_t elem = tid; elem < aggregate; elem += BLOCK_SIZE) {
            int32_t lo = 0, hi = valid_segs;
            while (lo < hi) {
                int32_t mid = (lo + hi + 1) >> 1;
                if (s_prefix[mid] <= elem) lo = mid;
                else hi = mid - 1;
            }
            int32_t owner = lo;

            double d_u = s_dist[owner];
            if (d_u >= cutoff) continue;

            int32_t edge_within_seg = elem - s_prefix[owner];
            int32_t e = s_edge_start[owner] + edge_within_seg;

            if (!((edge_mask[e >> 5] >> (e & 31)) & 1u)) continue;

            int32_t v = indices[e];
            double w = weights[e];
            double new_dist = d_u + w;
            if (new_dist >= cutoff) continue;

            double old_dist = dist[v];
            if (new_dist >= old_dist) continue;

            old_dist = atomicMinDouble(&dist[v], new_dist);

            if (new_dist < old_dist) {
                int32_t old_gen = atomicMax(&queued[v], generation);
                if (old_gen < generation) {
                    int32_t pos = atomicAdd(next_frontier_size, 1);
                    next_frontier[pos] = v;
                }
            }
        }

        __syncthreads();
    }
}

__global__ void __launch_bounds__(BLOCK_SIZE)
relax_edges_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ dist,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t* __restrict__ queued,
    double cutoff,
    int32_t generation
) {
    const int32_t lane = threadIdx.x & 31;
    const int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int32_t num_warps = (gridDim.x * blockDim.x) >> 5;

    for (int32_t vid = warp_id; vid < frontier_size; vid += num_warps) {
        int32_t u = frontier[vid];
        double d_u = dist[u];
        if (d_u >= cutoff) continue;

        int32_t start = offsets[u];
        int32_t end = offsets[u + 1];

        for (int32_t e = start + lane; e < end; e += WARP_SIZE) {
            if (!((edge_mask[e >> 5] >> (e & 31)) & 1u)) continue;

            int32_t v = indices[e];
            double new_dist = d_u + weights[e];
            if (new_dist >= cutoff) continue;

            double old_dist = dist[v];
            if (new_dist >= old_dist) continue;

            old_dist = atomicMinDouble(&dist[v], new_dist);

            if (new_dist < old_dist) {
                int32_t old_gen = atomicMax(&queued[v], generation);
                if (old_gen < generation) {
                    int32_t pos = atomicAdd(next_frontier_size, 1);
                    next_frontier[pos] = v;
                }
            }
        }
    }
}

__global__ void __launch_bounds__(BLOCK_SIZE)
fix_predecessors_positive_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t num_vertices,
    int32_t source
) {
    const int32_t lane = threadIdx.x & 31;
    const int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int32_t num_warps = (gridDim.x * blockDim.x) >> 5;

    for (int32_t u = warp_id; u < num_vertices; u += num_warps) {
        double d_u = dist[u];
        if (d_u >= UNREACHABLE_DIST) continue;

        int32_t start = offsets[u];
        int32_t end = offsets[u + 1];

        for (int32_t e = start + lane; e < end; e += WARP_SIZE) {
            if (!((edge_mask[e >> 5] >> (e & 31)) & 1u)) continue;

            int32_t v = indices[e];
            if (v == source) continue;

            double d_v = dist[v];
            if (d_v >= UNREACHABLE_DIST) continue;

            double w = weights[e];
            if (d_u + w == d_v && d_u < d_v) {
                pred[v] = u;
            }
        }
    }
}

__global__ void fix_predecessors_zero_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t num_vertices,
    int32_t source,
    int32_t* __restrict__ changed
) {
    int32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;
    if (u != source && pred[u] == -1) return;
    double d_u = dist[u];
    if (d_u >= UNREACHABLE_DIST) return;

    int32_t start = offsets[u];
    int32_t end = offsets[u + 1];

    for (int32_t e = start; e < end; e++) {
        if (!((edge_mask[e >> 5] >> (e & 31)) & 1u)) continue;
        int32_t v = indices[e];
        if (v == source) continue;
        double w = weights[e];
        if (d_u + w == dist[v] && dist[v] == d_u && pred[v] == -1) {
            pred[v] = u;
            *changed = 1;
        }
    }
}

void launch_init_sssp(double* dist, int32_t* pred, int32_t* queued,
                      int32_t n, int32_t source,
                      int32_t* frontier, int32_t* frontier_size) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_sssp_kernel<<<grid, BLOCK_SIZE>>>(dist, pred, queued, n, source, frontier, frontier_size);
}

void launch_relax_edges(
    const int32_t* offsets, const int32_t* indices,
    const double* weights, const uint32_t* edge_mask,
    double* dist,
    const int32_t* frontier, int32_t frontier_size,
    int32_t* next_frontier, int32_t* next_frontier_size,
    int32_t* queued, double cutoff, int32_t generation,
    int32_t* work_counter
) {
    if (frontier_size <= 0) return;

    if (frontier_size >= 32) {
        int num_sms = 58;
        int blocks_per_sm = 6;
        int max_blocks = num_sms * blocks_per_sm;
        int min_blocks = (frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int grid = min_blocks < max_blocks ? min_blocks : max_blocks;
        if (grid < 1) grid = 1;

        relax_edges_blockcoop_kernel<<<grid, BLOCK_SIZE>>>(
            offsets, indices, weights, edge_mask,
            dist, frontier, frontier_size,
            next_frontier, next_frontier_size, queued, cutoff, generation,
            work_counter
        );
    } else {
        int warps_per_block = BLOCK_SIZE / WARP_SIZE;
        int grid = (frontier_size + warps_per_block - 1) / warps_per_block;
        if (grid < 1) grid = 1;

        relax_edges_warp_kernel<<<grid, BLOCK_SIZE>>>(
            offsets, indices, weights, edge_mask,
            dist, frontier, frontier_size,
            next_frontier, next_frontier_size, queued, cutoff, generation
        );
    }
}

void launch_fix_predecessors_positive(
    const int32_t* offsets, const int32_t* indices,
    const double* weights, const uint32_t* edge_mask,
    const double* dist, int32_t* pred,
    int32_t num_vertices, int32_t source
) {
    int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    int grid = (num_vertices + warps_per_block - 1) / warps_per_block;
    if (grid > 65535) grid = 65535;

    fix_predecessors_positive_kernel<<<grid, BLOCK_SIZE>>>(
        offsets, indices, weights, edge_mask, dist, pred, num_vertices, source
    );
}

void launch_fix_predecessors_zero(
    const int32_t* offsets, const int32_t* indices,
    const double* weights, const uint32_t* edge_mask,
    const double* dist, int32_t* pred,
    int32_t num_vertices, int32_t source,
    int32_t* changed
) {
    int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 65535) grid = 65535;

    fix_predecessors_zero_kernel<<<grid, BLOCK_SIZE>>>(
        offsets, indices, weights, edge_mask, dist, pred, num_vertices, source, changed
    );
}

}  

void sssp_mask(const graph32_t& graph,
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
    const uint32_t* d_edge_mask = graph.edge_mask;

    cache.ensure(num_vertices);

    int32_t* d_frontier1 = cache.d_frontier1;
    int32_t* d_frontier2 = cache.d_frontier2;
    int32_t* d_queued = cache.d_queued;
    int32_t* d_size1 = cache.d_size1;
    int32_t* d_size2 = cache.d_size2;
    int32_t* d_work_counter = cache.d_work_counter;

    launch_init_sssp(distances, predecessors, d_queued, num_vertices, source,
                     d_frontier1, d_size1);

    int32_t* d_cur_frontier = d_frontier1;
    int32_t* d_next_frontier = d_frontier2;
    int32_t* d_cur_size = d_size1;
    int32_t* d_next_size = d_size2;

    *cache.h_frontier_size = 1;
    int32_t generation = 1;

    while (*cache.h_frontier_size > 0) {
        generation++;

        cudaMemsetAsync(d_next_size, 0, sizeof(int32_t));
        cudaMemsetAsync(d_work_counter, 0, sizeof(int32_t));

        launch_relax_edges(
            d_offsets, d_indices, edge_weights, d_edge_mask,
            distances,
            d_cur_frontier, *cache.h_frontier_size,
            d_next_frontier, d_next_size,
            d_queued, cutoff, generation, d_work_counter
        );

        cudaMemcpy(cache.h_frontier_size, d_next_size, sizeof(int32_t), cudaMemcpyDeviceToHost);

        int32_t* tmp = d_cur_frontier;
        d_cur_frontier = d_next_frontier;
        d_next_frontier = tmp;

        int32_t* tmp_s = d_cur_size;
        d_cur_size = d_next_size;
        d_next_size = tmp_s;
    }

    
    launch_fix_predecessors_positive(
        d_offsets, d_indices, edge_weights, d_edge_mask,
        distances, predecessors, num_vertices, source
    );

    
    {
        int32_t h_changed = 1;
        for (int32_t iter = 0; iter < num_vertices && h_changed; iter++) {
            cudaMemsetAsync(d_work_counter, 0, sizeof(int32_t));
            launch_fix_predecessors_zero(
                d_offsets, d_indices, edge_weights, d_edge_mask,
                distances, predecessors, num_vertices, source, d_work_counter
            );
            cudaMemcpy(&h_changed, d_work_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
        }
    }

    cudaDeviceSynchronize();
}

}  
