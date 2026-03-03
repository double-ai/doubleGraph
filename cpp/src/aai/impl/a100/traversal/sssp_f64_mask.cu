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

#define UNREACHABLE_DIST DBL_MAX

struct Cache : Cacheable {
    int32_t* fa = nullptr;
    int32_t* fb = nullptr;
    int* epoch = nullptr;
    int* ns = nullptr;
    int* d_changed = nullptr;

    int64_t fa_capacity = 0;
    int64_t fb_capacity = 0;
    int64_t epoch_capacity = 0;
    bool ns_allocated = false;
    bool d_changed_allocated = false;

    void ensure(int32_t nv) {
        if (fa_capacity < nv) {
            if (fa) cudaFree(fa);
            cudaMalloc(&fa, (size_t)nv * sizeof(int32_t));
            fa_capacity = nv;
        }
        if (fb_capacity < nv) {
            if (fb) cudaFree(fb);
            cudaMalloc(&fb, (size_t)nv * sizeof(int32_t));
            fb_capacity = nv;
        }
        if (epoch_capacity < nv) {
            if (epoch) cudaFree(epoch);
            cudaMalloc(&epoch, (size_t)nv * sizeof(int));
            epoch_capacity = nv;
        }
        if (!ns_allocated) {
            cudaMalloc(&ns, sizeof(int));
            ns_allocated = true;
        }
        if (!d_changed_allocated) {
            cudaMalloc(&d_changed, sizeof(int));
            d_changed_allocated = true;
        }
    }

    ~Cache() override {
        if (fa) cudaFree(fa);
        if (fb) cudaFree(fb);
        if (epoch) cudaFree(epoch);
        if (ns) cudaFree(ns);
        if (d_changed) cudaFree(d_changed);
    }
};

__device__ __forceinline__ bool atomicMinDouble(double* address, double val) {
    unsigned long long* addr_ull = (unsigned long long*)address;
    unsigned long long old = *addr_ull;
    while (true) {
        if (__longlong_as_double(old) <= val) return false;
        unsigned long long assumed = old;
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
        if (old == assumed) return true;
    }
}

__global__ void init_sssp(
    double* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t source,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dist[i] = (i == source) ? 0.0 : UNREACHABLE_DIST;
        pred[i] = -1;
    }
}

__global__ void relax_edges_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ mask,
    double* __restrict__ dist,
    const int32_t* __restrict__ frontier,
    int frontier_size,
    int32_t* __restrict__ next_frontier,
    int* __restrict__ next_size,
    int* __restrict__ epoch,
    int current_epoch,
    double cutoff
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int u = frontier[warp_id];
    double d_u = dist[u];

    int start = offsets[u];
    int end = offsets[u + 1];
    int degree = end - start;

    for (int i = lane_id; i < degree; i += 32) {
        int e = start + i;

        if (!((mask[e >> 5] >> (e & 31)) & 1u)) continue;

        int v = __ldg(&indices[e]);
        double w = __ldg(&weights[e]);
        double d_new = d_u + w;

        if (d_new < cutoff && atomicMinDouble(&dist[v], d_new)) {
            int old_epoch = atomicExch(&epoch[v], current_epoch);
            if (old_epoch != current_epoch) {
                next_frontier[atomicAdd(next_size, 1)] = v;
            }
        }
    }
}

__global__ void relax_edges_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ mask,
    double* __restrict__ dist,
    const int32_t* __restrict__ frontier,
    int frontier_size,
    int32_t* __restrict__ next_frontier,
    int* __restrict__ next_size,
    int* __restrict__ epoch,
    int current_epoch,
    double cutoff
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int u = frontier[tid];
    double d_u = dist[u];

    int start = offsets[u];
    int end = offsets[u + 1];

    for (int e = start; e < end; ++e) {
        if (!((mask[e >> 5] >> (e & 31)) & 1u)) continue;

        int v = __ldg(&indices[e]);
        double w = __ldg(&weights[e]);
        double d_new = d_u + w;

        if (d_new < cutoff && atomicMinDouble(&dist[v], d_new)) {
            int old_epoch = atomicExch(&epoch[v], current_epoch);
            if (old_epoch != current_epoch) {
                next_frontier[atomicAdd(next_size, 1)] = v;
            }
        }
    }
}

__global__ void set_predecessors_positive(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ mask,
    const double* __restrict__ dist,
    int32_t* __restrict__ pred,
    int n,
    int32_t source
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;

    double d_u = dist[u];
    if (d_u >= UNREACHABLE_DIST) return;

    int start = offsets[u];
    int end = offsets[u + 1];

    for (int e = start; e < end; ++e) {
        if (!((mask[e >> 5] >> (e & 31)) & 1u)) continue;

        int v = indices[e];
        if (v == source) continue;

        double d_v = dist[v];
        if (d_v >= UNREACHABLE_DIST) continue;

        double w = weights[e];
        if (d_u + w == d_v && d_u < d_v) {
            pred[v] = u;
        }
    }
}

__global__ void set_predecessors_zero(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ mask,
    const double* __restrict__ dist,
    int32_t* __restrict__ pred,
    int n,
    int32_t source,
    int* __restrict__ changed
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;

    double d_u = dist[u];
    if (d_u >= UNREACHABLE_DIST) return;

    if (u != source && pred[u] == -1) return;

    int start = offsets[u];
    int end = offsets[u + 1];

    for (int e = start; e < end; ++e) {
        if (!((mask[e >> 5] >> (e & 31)) & 1u)) continue;

        int v = indices[e];
        if (v == source) continue;

        double w = weights[e];
        if (d_u + w == dist[v] && dist[v] == d_u && pred[v] == -1) {
            pred[v] = u;
            *changed = 1;
        }
    }
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

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t nv = graph.number_of_vertices;
    const uint32_t* d_mask = graph.edge_mask;

    cache.ensure(nv);

    if (nv > 0) {
        init_sssp<<<(nv + 255) / 256, 256>>>(distances, predecessors, source, nv);
    }
    cudaMemsetAsync(cache.epoch, 0, (size_t)nv * sizeof(int));

    cudaMemcpyAsync(cache.fa, &source, sizeof(int32_t), cudaMemcpyHostToDevice);

    int32_t* cur = cache.fa;
    int32_t* nxt = cache.fb;
    int fsize = 1;
    int current_epoch = 1;

    const int WARP_THRESHOLD = 64;

    while (fsize > 0) {
        cudaMemsetAsync(cache.ns, 0, sizeof(int));

        if (fsize >= WARP_THRESHOLD) {
            int block = 256;
            int warps_per_block = block / 32;
            int grid = (fsize + warps_per_block - 1) / warps_per_block;
            relax_edges_warp<<<grid, block>>>(
                d_offsets, d_indices, edge_weights, d_mask, distances,
                cur, fsize, nxt, cache.ns, cache.epoch, current_epoch, cutoff
            );
        } else {
            int block = 256;
            int grid = (fsize + block - 1) / block;
            relax_edges_thread<<<grid, block>>>(
                d_offsets, d_indices, edge_weights, d_mask, distances,
                cur, fsize, nxt, cache.ns, cache.epoch, current_epoch, cutoff
            );
        }

        cudaMemcpy(&fsize, cache.ns, sizeof(int), cudaMemcpyDeviceToHost);

        std::swap(cur, nxt);
        current_epoch++;
    }

    
    if (nv > 0) {
        set_predecessors_positive<<<(nv + 255) / 256, 256>>>(
            d_offsets, d_indices, edge_weights, d_mask, distances,
            predecessors, nv, source
        );
    }

    
    int h_changed = 1;
    for (int iter = 0; iter < nv && h_changed; iter++) {
        cudaMemsetAsync(cache.d_changed, 0, sizeof(int));
        set_predecessors_zero<<<(nv + 255) / 256, 256>>>(
            d_offsets, d_indices, edge_weights, d_mask, distances,
            predecessors, nv, source, cache.d_changed
        );
        cudaMemcpy(&h_changed, cache.d_changed, sizeof(int), cudaMemcpyDeviceToHost);
    }
}

}  
