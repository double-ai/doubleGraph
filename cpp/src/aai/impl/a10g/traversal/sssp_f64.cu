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

namespace aai {

namespace {

#define UNREACHABLE_DIST DBL_MAX
#define BLOCK_SIZE 512
#define WARP_SIZE 32

struct Cache : Cacheable {
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* gen_flags = nullptr;
    int32_t* counter = nullptr;
    int* h_counter = nullptr;
    int32_t* d_changed = nullptr;
    int64_t frontier_a_capacity = 0;
    int64_t frontier_b_capacity = 0;
    int64_t gen_flags_capacity = 0;
    bool counter_allocated = false;
    bool h_counter_allocated = false;
    bool changed_allocated = false;

    void ensure(int32_t num_vertices) {
        int64_t n = num_vertices;
        if (frontier_a_capacity < n) {
            if (frontier_a) cudaFree(frontier_a);
            cudaMalloc(&frontier_a, n * sizeof(int32_t));
            frontier_a_capacity = n;
        }
        if (frontier_b_capacity < n) {
            if (frontier_b) cudaFree(frontier_b);
            cudaMalloc(&frontier_b, n * sizeof(int32_t));
            frontier_b_capacity = n;
        }
        if (gen_flags_capacity < n) {
            if (gen_flags) cudaFree(gen_flags);
            cudaMalloc(&gen_flags, n * sizeof(int32_t));
            gen_flags_capacity = n;
        }
        if (!counter_allocated) {
            cudaMalloc(&counter, 2 * sizeof(int32_t));
            counter_allocated = true;
        }
        if (!h_counter_allocated) {
            cudaMallocHost(&h_counter, sizeof(int));
            h_counter_allocated = true;
        }
        if (!changed_allocated) {
            cudaMalloc(&d_changed, sizeof(int32_t));
            changed_allocated = true;
        }
    }

    ~Cache() override {
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (gen_flags) cudaFree(gen_flags);
        if (counter) cudaFree(counter);
        if (h_counter) cudaFreeHost(h_counter);
        if (d_changed) cudaFree(d_changed);
    }
};


__device__ __forceinline__ double atomicMinDouble(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val) return __longlong_as_double(old);
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void init_sssp_kernel(
    double* __restrict__ distances,
    int* __restrict__ predecessors,
    int* __restrict__ gen_flags,
    int num_vertices,
    int source,
    int* __restrict__ frontier
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        distances[tid] = (tid == source) ? 0.0 : UNREACHABLE_DIST;
        predecessors[tid] = -1;
        gen_flags[tid] = -1;
    }
    if (tid == 0) {
        frontier[0] = source;
    }
}


__global__ void __launch_bounds__(512, 3)
relax_kernel_warp(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const double* __restrict__ weights,
    double* __restrict__ distances,
    const int* __restrict__ frontier,
    const int frontier_size,
    int* __restrict__ next_frontier,
    int* __restrict__ next_frontier_count,
    int* __restrict__ gen_flags,
    const double cutoff,
    const int current_iteration
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    if (warp_id >= frontier_size) return;

    const int v = __ldg(&frontier[warp_id]);
    const double dist_v = distances[v];
    if (dist_v >= cutoff) return;

    const int start = __ldg(&offsets[v]);
    const int end = __ldg(&offsets[v + 1]);

    for (int e = start + lane; e < end; e += WARP_SIZE) {
        const int u = __ldg(&indices[e]);
        const double w = __ldg(&weights[e]);
        const double new_dist = dist_v + w;

        if (new_dist < cutoff && new_dist < distances[u]) {
            double old_dist = atomicMinDouble(&distances[u], new_dist);
            if (old_dist > new_dist) {
                int old_gen = atomicExch(&gen_flags[u], current_iteration);
                if (old_gen != current_iteration) {
                    int pos = atomicAdd(next_frontier_count, 1);
                    next_frontier[pos] = u;
                }
            }
        }
    }
}


__global__ void relax_kernel_simple(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const double* __restrict__ weights,
    double* __restrict__ distances,
    const int* __restrict__ frontier,
    const int frontier_size,
    int* __restrict__ next_frontier,
    int* __restrict__ next_frontier_count,
    int* __restrict__ gen_flags,
    const double cutoff,
    const int current_iteration
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int v = frontier[tid];
    double dist_v = distances[v];
    if (dist_v >= cutoff) return;

    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);

    for (int e = start; e < end; e++) {
        int u = __ldg(&indices[e]);
        double new_dist = dist_v + __ldg(&weights[e]);
        if (new_dist >= cutoff) continue;
        if (new_dist >= distances[u]) continue;

        double old_dist = atomicMinDouble(&distances[u], new_dist);
        if (old_dist > new_dist) {
            int old_gen = atomicExch(&gen_flags[u], current_iteration);
            if (old_gen != current_iteration) {
                int pos = atomicAdd(next_frontier_count, 1);
                next_frontier[pos] = u;
            }
        }
    }
}




__global__ void __launch_bounds__(512, 3)
compute_predecessors_warp_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ distances,
    int* __restrict__ predecessors,
    const int num_vertices,
    const int source
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);

    if (warp_id >= num_vertices) return;
    const int u = warp_id;

    const double dist_u = distances[u];
    if (dist_u >= UNREACHABLE_DIST) return;

    const int start = __ldg(&offsets[u]);
    const int end = __ldg(&offsets[u + 1]);

    for (int e = start + lane; e < end; e += WARP_SIZE) {
        const int v = __ldg(&indices[e]);
        if (v == source) continue;

        const double dist_v = distances[v];
        if (dist_v >= UNREACHABLE_DIST) continue;

        const double expected = dist_u + __ldg(&weights[e]);
        if (expected == dist_v && dist_u < dist_v) {
            predecessors[v] = u;
        }
    }
}


__global__ void zerow_pred_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ distances,
    int* __restrict__ predecessors,
    int* __restrict__ changed,
    int source, int N
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N) return;
    double d_u = distances[u];
    if (d_u >= UNREACHABLE_DIST) return;
    if (u != source && predecessors[u] == -1) return;  
    int es = offsets[u], ee = offsets[u + 1];
    for (int e = es; e < ee; e++) {
        int v = indices[e];
        if (v == source) continue;
        double w = weights[e];
        if (d_u + w != distances[v]) continue;   
        if (d_u != distances[v]) continue;        
        if (predecessors[v] == -1) {
            predecessors[v] = u;
            *changed = 1;
        }
    }
}

void launch_init_sssp(
    double* distances, int* predecessors, int* gen_flags,
    int num_vertices, int source, int* frontier
) {
    int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_sssp_kernel<<<grid, BLOCK_SIZE>>>(distances, predecessors, gen_flags,
                                            num_vertices, source, frontier);
}

void launch_relax(
    const int* offsets, const int* indices, const double* weights,
    double* distances,
    const int* frontier, int frontier_size,
    int* next_frontier, int* next_frontier_count,
    int* gen_flags, double cutoff, int current_iteration
) {
    if (frontier_size == 0) return;

    if (frontier_size >= 8) {
        int warps_needed = frontier_size;
        int threads_needed = warps_needed * WARP_SIZE;
        int grid = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
        relax_kernel_warp<<<grid, BLOCK_SIZE>>>(
            offsets, indices, weights, distances,
            frontier, frontier_size,
            next_frontier, next_frontier_count,
            gen_flags, cutoff, current_iteration
        );
    } else {
        relax_kernel_simple<<<1, BLOCK_SIZE>>>(
            offsets, indices, weights, distances,
            frontier, frontier_size,
            next_frontier, next_frontier_count,
            gen_flags, cutoff, current_iteration
        );
    }
}

void launch_compute_predecessors(
    const int* offsets, const int* indices, const double* weights,
    const double* distances, int* predecessors,
    int num_vertices, int source
) {
    int warps_needed = num_vertices;
    int threads_needed = warps_needed * WARP_SIZE;
    int grid = (threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_predecessors_warp_kernel<<<grid, BLOCK_SIZE>>>(
        offsets, indices, weights, distances, predecessors,
        num_vertices, source
    );
}

}  

void sssp(const graph32_t& graph,
          const double* edge_weights,
          int32_t source,
          double* distances,
          int32_t* predecessors,
          double cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    cache.ensure(num_vertices);

    int32_t* d_frontier_a = cache.frontier_a;
    int32_t* d_frontier_b = cache.frontier_b;
    int32_t* d_gen_flags = cache.gen_flags;
    int32_t* d_counter = cache.counter;

    
    launch_init_sssp(distances, predecessors, d_gen_flags,
                     num_vertices, source, d_frontier_a);

    
    
    int init_counters[2] = {1, 0};
    cudaMemcpyAsync(d_counter, init_counters, 2 * sizeof(int),
                    cudaMemcpyHostToDevice, 0);

    int* current_frontier = d_frontier_a;
    int* next_frontier = d_frontier_b;
    int current_size = 1;
    int iteration = 0;
    int cur_buf = 0;

    while (current_size > 0) {
        int nxt_buf = 1 - cur_buf;

        
        cudaMemsetAsync(&d_counter[nxt_buf], 0, sizeof(int), 0);

        
        launch_relax(
            offsets, indices, edge_weights,
            distances,
            current_frontier, current_size,
            next_frontier, &d_counter[nxt_buf],
            d_gen_flags, cutoff, iteration
        );

        
        cudaMemcpyAsync(cache.h_counter, &d_counter[nxt_buf], sizeof(int),
                       cudaMemcpyDeviceToHost, 0);
        cudaStreamSynchronize(0);
        current_size = *cache.h_counter;

        
        int* temp = current_frontier;
        current_frontier = next_frontier;
        next_frontier = temp;
        cur_buf = nxt_buf;
        iteration++;
    }

    
    launch_compute_predecessors(
        offsets, indices, edge_weights,
        distances, predecessors,
        num_vertices, source
    );

    
    int32_t* d_ch = cache.d_changed;
    int32_t h_ch = 1;
    for (int ziter = 0; ziter < num_vertices && h_ch; ziter++) {
        cudaMemsetAsync(d_ch, 0, sizeof(int32_t));
        int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        zerow_pred_kernel<<<grid, BLOCK_SIZE>>>(
            offsets, indices, edge_weights,
            distances, predecessors, d_ch,
            source, num_vertices);
        cudaMemcpy(&h_ch, d_ch, sizeof(int32_t), cudaMemcpyDeviceToHost);
    }
    cudaDeviceSynchronize();
}

}  
