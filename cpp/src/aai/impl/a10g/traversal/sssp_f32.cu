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
#include <cmath>
#include <limits>
#include <float.h>
#include <cub/cub.cuh>

namespace aai {

namespace {





struct Cache : Cacheable {
    unsigned long long* d_dist_pred = nullptr;
    int* d_frontier_0 = nullptr;
    int* d_frontier_1 = nullptr;
    int* d_frontier_size = nullptr;
    int* d_work_counter = nullptr;
    int* d_last_updated = nullptr;
    int* d_degrees = nullptr;
    int* d_prefix_sums = nullptr;
    void* d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;
    int* h_frontier_size = nullptr;
    int* h_total_edges = nullptr;

    int alloc_vertices = 0;

    void ensure(int nv) {
        if (nv > alloc_vertices) {
            free_all();
            alloc_vertices = nv;
            cudaMalloc(&d_dist_pred, (size_t)nv * sizeof(unsigned long long));
            cudaMalloc(&d_frontier_0, (size_t)nv * sizeof(int));
            cudaMalloc(&d_frontier_1, (size_t)nv * sizeof(int));
            cudaMalloc(&d_frontier_size, sizeof(int));
            cudaMalloc(&d_work_counter, sizeof(int));
            cudaMalloc(&d_last_updated, (size_t)nv * sizeof(int));
            cudaMalloc(&d_degrees, (size_t)nv * sizeof(int));
            cudaMalloc(&d_prefix_sums, (size_t)nv * sizeof(int));

            size_t tb = 0;
            cub::DeviceScan::InclusiveSum(nullptr, tb, (int*)nullptr, (int*)nullptr, nv);
            scan_temp_bytes = tb;
            cudaMalloc(&d_scan_temp, scan_temp_bytes);

            cudaMallocHost(&h_frontier_size, sizeof(int));
            cudaMallocHost(&h_total_edges, sizeof(int));
        }
    }

    void free_all() {
        if (d_dist_pred) cudaFree(d_dist_pred);
        if (d_frontier_0) cudaFree(d_frontier_0);
        if (d_frontier_1) cudaFree(d_frontier_1);
        if (d_frontier_size) cudaFree(d_frontier_size);
        if (d_work_counter) cudaFree(d_work_counter);
        if (d_last_updated) cudaFree(d_last_updated);
        if (d_degrees) cudaFree(d_degrees);
        if (d_prefix_sums) cudaFree(d_prefix_sums);
        if (d_scan_temp) cudaFree(d_scan_temp);
        if (h_frontier_size) cudaFreeHost(h_frontier_size);
        if (h_total_edges) cudaFreeHost(h_total_edges);
        d_dist_pred = nullptr;
        d_frontier_0 = d_frontier_1 = nullptr;
        d_frontier_size = d_work_counter = nullptr;
        d_last_updated = d_degrees = d_prefix_sums = nullptr;
        d_scan_temp = nullptr;
        h_frontier_size = h_total_edges = nullptr;
        scan_temp_bytes = 0;
        alloc_vertices = 0;
    }

    ~Cache() override {
        free_all();
    }
};





__device__ __forceinline__ unsigned long long pack_dist_pred(float dist, int pred) {
    unsigned long long d = __float_as_uint(dist);
    unsigned long long p = (unsigned int)pred;
    return (d << 32) | p;
}

__device__ __forceinline__ float unpack_dist(unsigned long long packed) {
    return __uint_as_float((unsigned int)(packed >> 32));
}

__device__ __forceinline__ int unpack_pred(unsigned long long packed) {
    return (int)(unsigned int)(packed & 0xFFFFFFFFULL);
}

__global__ void init_kernel(
    unsigned long long* __restrict__ dist_pred,
    int* __restrict__ last_updated,
    int num_vertices, int source
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        float d = (idx == source) ? 0.0f : FLT_MAX;
        dist_pred[idx] = pack_dist_pred(d, -1);
        last_updated[idx] = -1;
    }
}

__device__ __forceinline__ void relax_edge(
    int u, float dist_u, int v, float w,
    unsigned long long* __restrict__ dist_pred,
    int* __restrict__ next_frontier,
    int* __restrict__ next_frontier_size,
    int* __restrict__ last_updated,
    int next_iter, float cutoff
) {
    float new_dist = dist_u + w;
    if (new_dist >= cutoff) return;

    unsigned long long new_packed = pack_dist_pred(new_dist, u);
    unsigned long long old_packed = dist_pred[v];
    
    
    
    if ((new_packed >> 32) >= (old_packed >> 32)) return;

    while ((new_packed >> 32) < (old_packed >> 32)) {
        unsigned long long assumed = old_packed;
        old_packed = atomicCAS(&dist_pred[v], assumed, new_packed);
        if (old_packed == assumed) {
            if (atomicMax(&last_updated[v], next_iter) < next_iter) {
                int pos = atomicAdd(next_frontier_size, 1);
                next_frontier[pos] = v;
            }
            return;
        }
    }
}

__global__ void relax_worksteal_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ weights,
    unsigned long long* __restrict__ dist_pred,
    const int* __restrict__ frontier,
    int frontier_size,
    int* __restrict__ next_frontier,
    int* __restrict__ next_frontier_size,
    int* __restrict__ last_updated,
    int next_iter,
    float cutoff,
    int* __restrict__ work_counter
) {
    const int lane_id = threadIdx.x & 31;

    while (true) {
        int wi;
        if (lane_id == 0) {
            wi = atomicAdd(work_counter, 1);
        }
        wi = __shfl_sync(0xFFFFFFFF, wi, 0);
        if (wi >= frontier_size) return;

        int u = frontier[wi];
        unsigned long long packed_u = dist_pred[u];
        float dist_u = unpack_dist(packed_u);
        if (dist_u >= cutoff) continue;

        int start = __ldg(&offsets[u]);
        int end = __ldg(&offsets[u + 1]);

        for (int e = start + lane_id; e < end; e += 32) {
            int v = __ldg(&indices[e]);
            float w = __ldg(&weights[e]);
            relax_edge(u, dist_u, v, w, dist_pred, next_frontier, next_frontier_size, last_updated, next_iter, cutoff);
        }
    }
}


__global__ void relax_edge_parallel_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const float* __restrict__ weights,
    unsigned long long* __restrict__ dist_pred,
    const int* __restrict__ frontier,
    int frontier_size,
    const int* __restrict__ prefix_sums,
    int total_edges,
    int* __restrict__ next_frontier,
    int* __restrict__ next_frontier_size,
    int* __restrict__ last_updated,
    int next_iter,
    float cutoff
) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= total_edges) return;

    
    int lo = 0, hi = frontier_size;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (__ldg(&prefix_sums[mid]) <= eid) lo = mid + 1;
        else hi = mid;
    }
    int fi = lo;
    int u = __ldg(&frontier[fi]);
    int edge_base = (fi > 0) ? __ldg(&prefix_sums[fi - 1]) : 0;
    int e = __ldg(&offsets[u]) + (eid - edge_base);

    unsigned long long packed_u = dist_pred[u];
    float dist_u = unpack_dist(packed_u);
    if (dist_u >= cutoff) return;

    int v = __ldg(&indices[e]);
    float w = __ldg(&weights[e]);
    relax_edge(u, dist_u, v, w, dist_pred, next_frontier, next_frontier_size, last_updated, next_iter, cutoff);
}

__global__ void compute_degrees_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ frontier,
    int frontier_size,
    int* __restrict__ degrees
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < frontier_size) {
        int u = frontier[tid];
        degrees[tid] = __ldg(&offsets[u + 1]) - __ldg(&offsets[u]);
    }
}

__global__ void unpack_kernel(
    const unsigned long long* __restrict__ dist_pred,
    float* __restrict__ dist,
    int* __restrict__ pred,
    int num_vertices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        unsigned long long packed = dist_pred[idx];
        dist[idx] = unpack_dist(packed);
        pred[idx] = unpack_pred(packed);
    }
}

static constexpr int EDGE_PARALLEL_THRESHOLD = 4096;

}  

void sssp(const graph32_t& graph,
          const float* edge_weights,
          int32_t source,
          float* distances,
          int32_t* predecessors,
          float cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int num_vertices = graph.number_of_vertices;
    const int* d_offsets = graph.offsets;
    const int* d_indices = graph.indices;
    const float* d_weights = edge_weights;

    if (std::isinf(cutoff)) cutoff = std::numeric_limits<float>::max();

    cache.ensure(num_vertices);

    
    init_kernel<<<(num_vertices + 255) / 256, 256>>>(
        cache.d_dist_pred, cache.d_last_updated, num_vertices, source);

    
    cudaMemcpy(cache.d_frontier_0, &source, sizeof(int), cudaMemcpyHostToDevice);
    int frontier_size = 1;
    int cur = 0;
    int iter = 0;

    while (frontier_size > 0) {
        int nxt = 1 - cur;
        int* cur_frontier = (cur == 0) ? cache.d_frontier_0 : cache.d_frontier_1;
        int* nxt_frontier = (nxt == 0) ? cache.d_frontier_0 : cache.d_frontier_1;

        cudaMemsetAsync(cache.d_frontier_size, 0, sizeof(int));

        if (frontier_size >= EDGE_PARALLEL_THRESHOLD) {
            
            compute_degrees_kernel<<<(frontier_size + 255) / 256, 256>>>(
                d_offsets, cur_frontier, frontier_size, cache.d_degrees);

            cub::DeviceScan::InclusiveSum(
                cache.d_scan_temp, cache.scan_temp_bytes,
                cache.d_degrees, cache.d_prefix_sums, frontier_size);

            cudaMemcpyAsync(cache.h_total_edges,
                cache.d_prefix_sums + frontier_size - 1,
                sizeof(int), cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(0);
            int total_edges = *cache.h_total_edges;

            if (total_edges > 0) {
                int block = 256;
                int grid = (total_edges + block - 1) / block;
                relax_edge_parallel_kernel<<<grid, block>>>(
                    d_offsets, d_indices, d_weights, cache.d_dist_pred,
                    cur_frontier, frontier_size, cache.d_prefix_sums, total_edges,
                    nxt_frontier, cache.d_frontier_size, cache.d_last_updated,
                    iter + 1, cutoff);
            }
        } else {
            
            cudaMemsetAsync(cache.d_work_counter, 0, sizeof(int));
            relax_worksteal_kernel<<<1280, 256>>>(
                d_offsets, d_indices, d_weights, cache.d_dist_pred,
                cur_frontier, frontier_size, nxt_frontier, cache.d_frontier_size,
                cache.d_last_updated, iter + 1, cutoff, cache.d_work_counter);
        }

        cudaMemcpyAsync(cache.h_frontier_size, cache.d_frontier_size,
            sizeof(int), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(0);

        frontier_size = *cache.h_frontier_size;
        cur = nxt;
        iter++;
    }

    
    unpack_kernel<<<(num_vertices + 255) / 256, 256>>>(
        cache.d_dist_pred, distances, predecessors, num_vertices);
}

}  
