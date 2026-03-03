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

struct Cache : Cacheable {
    int32_t* d_frontier[2] = {nullptr, nullptr};
    int32_t* d_frontier_size = nullptr;
    int32_t* d_in_next = nullptr;
    size_t alloc_vertices = 0;

    void ensure_capacity(size_t n) {
        if (n <= alloc_vertices) return;
        cleanup_buffers();
        alloc_vertices = n + 1000;
        cudaMalloc(&d_frontier[0], alloc_vertices * sizeof(int32_t));
        cudaMalloc(&d_frontier[1], alloc_vertices * sizeof(int32_t));
        cudaMalloc(&d_frontier_size, sizeof(int32_t));
        cudaMalloc(&d_in_next, alloc_vertices * sizeof(int32_t));
    }

    void cleanup_buffers() {
        if (d_frontier[0]) { cudaFree(d_frontier[0]); d_frontier[0] = nullptr; }
        if (d_frontier[1]) { cudaFree(d_frontier[1]); d_frontier[1] = nullptr; }
        if (d_frontier_size) { cudaFree(d_frontier_size); d_frontier_size = nullptr; }
        if (d_in_next) { cudaFree(d_in_next); d_in_next = nullptr; }
        alloc_vertices = 0;
    }

    ~Cache() override {
        cleanup_buffers();
    }
};



__device__ __forceinline__ bool atomicMinDouble(double* addr, double val) {
    unsigned long long int* addr_as_ull = (unsigned long long int*)addr;
    unsigned long long int old = *addr_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val) return false;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return true;
}

__global__ void init_sssp_kernel(double* distances, int32_t* predecessors,
                                  int32_t num_vertices, int32_t source) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        distances[idx] = (idx == source) ? 0.0 : UNREACHABLE_DIST;
        predecessors[idx] = -1;
    }
}


__global__ void relax_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    double* __restrict__ distances,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t* __restrict__ in_next,
    double cutoff
) {
    const int warp_id = ((int)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t src = frontier[warp_id];
    double src_dist = distances[src];

    int32_t start = offsets[src];
    int32_t end = offsets[src + 1];

    for (int32_t e = start + lane; e < end; e += 32) {
        int32_t dst = indices[e];
        double new_dist = src_dist + weights[e];

        if (new_dist >= cutoff) continue;

        bool improved = atomicMinDouble(&distances[dst], new_dist);
        if (improved) {
            int32_t old_flag = atomicExch(&in_next[dst], 1);
            if (old_flag == 0) {
                int32_t pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = dst;
            }
        }
    }
}


__global__ void relax_edges_simple_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    double* __restrict__ distances,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t* __restrict__ in_next,
    double cutoff
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int32_t src = frontier[tid];
    double src_dist = distances[src];

    int32_t start = offsets[src];
    int32_t end = offsets[src + 1];

    for (int32_t e = start; e < end; e++) {
        int32_t dst = indices[e];
        double new_dist = src_dist + weights[e];

        if (new_dist >= cutoff) continue;

        bool improved = atomicMinDouble(&distances[dst], new_dist);
        if (improved) {
            int32_t old_flag = atomicExch(&in_next[dst], 1);
            if (old_flag == 0) {
                int32_t pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = dst;
            }
        }
    }
}


__global__ void reset_flags_kernel(int32_t* in_next, const int32_t* frontier, int32_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        in_next[frontier[tid]] = 0;
    }
}

__global__ void fix_predecessors_positive_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t num_vertices,
    int32_t source
) {
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int warp_in_block = threadIdx.x >> 5;
    const int total_warps = gridDim.x * warps_per_block;

    for (int u = blockIdx.x * warps_per_block + warp_in_block; u < num_vertices; u += total_warps) {
        double u_dist = distances[u];
        if (u_dist >= UNREACHABLE_DIST) continue;

        int32_t start = offsets[u];
        int32_t end = offsets[u + 1];

        for (int32_t e = start + lane; e < end; e += 32) {
            int32_t v = indices[e];
            if (v == source) continue;
            double v_dist = distances[v];
            if (v_dist >= UNREACHABLE_DIST) continue;
            double w = weights[e];
            if (u_dist + w == v_dist && u_dist < v_dist) {
                predecessors[v] = u;
            }
        }
    }
}

__global__ void fix_predecessors_zero_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t num_vertices,
    int32_t source,
    int32_t* __restrict__ changed
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;
    if (u != source && predecessors[u] == -1) return;
    double u_dist = distances[u];
    if (u_dist >= UNREACHABLE_DIST) return;

    int32_t start = offsets[u];
    int32_t end = offsets[u + 1];

    for (int32_t e = start; e < end; e++) {
        int32_t v = indices[e];
        if (v == source) continue;
        double w = weights[e];
        if (u_dist + w == distances[v] && distances[v] == u_dist && predecessors[v] == -1) {
            predecessors[v] = u;
            *changed = 1;
        }
    }
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
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    cache.ensure_capacity(num_vertices);

    cudaStream_t stream = 0;

    
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        init_sssp_kernel<<<grid, block, 0, stream>>>(distances, predecessors, num_vertices, source);
    }

    
    cudaMemsetAsync(cache.d_in_next, 0, num_vertices * sizeof(int32_t), stream);

    
    cudaMemcpyAsync(cache.d_frontier[0], &source, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    int cur = 0;
    int32_t frontier_size = 1;
    int32_t h_fs;

    while (frontier_size > 0) {
        int next = 1 - cur;

        
        cudaMemsetAsync(cache.d_frontier_size, 0, sizeof(int32_t), stream);

        
        if (frontier_size <= 64) {
            int block = 256;
            int grid = (frontier_size + block - 1) / block;
            relax_edges_simple_kernel<<<grid, block, 0, stream>>>(
                d_offsets, d_indices, edge_weights, distances,
                cache.d_frontier[cur], frontier_size,
                cache.d_frontier[next], cache.d_frontier_size,
                cache.d_in_next, cutoff);
        } else {
            int block = 256;
            int warps_per_block = block / 32;
            int grid = (frontier_size + warps_per_block - 1) / warps_per_block;
            relax_edges_kernel<<<grid, block, 0, stream>>>(
                d_offsets, d_indices, edge_weights, distances,
                cache.d_frontier[cur], frontier_size,
                cache.d_frontier[next], cache.d_frontier_size,
                cache.d_in_next, cutoff);
        }

        
        cudaMemcpyAsync(&h_fs, cache.d_frontier_size, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        frontier_size = h_fs;

        
        if (frontier_size > 0) {
            int block = 256;
            int grid = (frontier_size + block - 1) / block;
            reset_flags_kernel<<<grid, block, 0, stream>>>(cache.d_in_next, cache.d_frontier[next], frontier_size);
        }

        cur = next;
    }

    
    {
        int block = 256;
        int warps_per_block = block / 32;
        int grid = (num_vertices + warps_per_block - 1) / warps_per_block;
        if (grid > 65535) grid = 65535;
        fix_predecessors_positive_kernel<<<grid, block, 0, stream>>>(
            d_offsets, d_indices, edge_weights, distances, predecessors, num_vertices, source);
    }

    
    {
        int32_t h_changed = 1;
        for (int iter = 0; iter < num_vertices && h_changed; iter++) {
            cudaMemsetAsync(cache.d_frontier_size, 0, sizeof(int32_t), stream);
            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            fix_predecessors_zero_kernel<<<grid, block, 0, stream>>>(
                d_offsets, d_indices, edge_weights, distances, predecessors, num_vertices, source, cache.d_frontier_size);
            cudaMemcpyAsync(&h_changed, cache.d_frontier_size, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
    }
}

}  
