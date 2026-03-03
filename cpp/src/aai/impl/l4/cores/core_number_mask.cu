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

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* valid = nullptr;
    int32_t* remaining = nullptr;
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* delta_acc = nullptr;
    int32_t* candidates = nullptr;
    int32_t* sizes = nullptr;
    int32_t* h_pinned = nullptr;
    int64_t capacity = 0;

    Cache() {
        cudaMallocHost(&h_pinned, 4 * sizeof(int32_t));
        cudaMalloc(&sizes, 4 * sizeof(int32_t));
    }

    void ensure(int64_t n) {
        if (capacity < n) {
            if (valid) cudaFree(valid);
            if (remaining) cudaFree(remaining);
            if (frontier_a) cudaFree(frontier_a);
            if (frontier_b) cudaFree(frontier_b);
            if (delta_acc) cudaFree(delta_acc);
            if (candidates) cudaFree(candidates);
            cudaMalloc(&valid, n * sizeof(int32_t));
            cudaMalloc(&remaining, n * sizeof(int32_t));
            cudaMalloc(&frontier_a, n * sizeof(int32_t));
            cudaMalloc(&frontier_b, n * sizeof(int32_t));
            cudaMalloc(&delta_acc, n * sizeof(int32_t));
            cudaMalloc(&candidates, n * sizeof(int32_t));
            capacity = n;
        }
    }

    ~Cache() override {
        if (valid) cudaFree(valid);
        if (remaining) cudaFree(remaining);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (delta_acc) cudaFree(delta_acc);
        if (candidates) cudaFree(candidates);
        if (sizes) cudaFree(sizes);
        if (h_pinned) cudaFreeHost(h_pinned);
    }
};



__device__ __forceinline__ bool is_edge_active(const uint32_t* edge_mask, int j) {
    return (edge_mask[j >> 5] >> (j & 31)) & 1;
}

__global__ void compute_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ remaining,
    int32_t* __restrict__ valid,
    int32_t num_vertices,
    int32_t multiplier
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int deg = 0;
        for (int e = start; e < end; e++) {
            if (is_edge_active(edge_mask, e) && indices[e] != v) deg++;
        }
        core_numbers[v] = deg * multiplier;
        remaining[v] = (deg > 0) ? 1 : 0;
        valid[v] = 1;
    }
}

__global__ void build_frontier_kernel(
    const int32_t* __restrict__ core_numbers,
    const int32_t* __restrict__ remaining,
    const int32_t* __restrict__ valid,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_size,
    int32_t num_vertices,
    int32_t k
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        if (remaining[v] && valid[v] && core_numbers[v] < k) {
            frontier[atomicAdd(frontier_size, 1)] = v;
        }
    }
}

__global__ void mark_invalid_kernel(
    int32_t* __restrict__ valid,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < frontier_size; i += blockDim.x * gridDim.x) {
        valid[frontier[i]] = 0;
    }
}

__global__ void accumulate_deltas_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ valid,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ delta_acc,
    int32_t* __restrict__ candidates,
    int32_t* __restrict__ candidates_size,
    int32_t delta
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (blockDim.x * gridDim.x) >> 5;

    for (int i = warp_id; i < frontier_size; i += total_warps) {
        int v = frontier[i];
        int start = offsets[v];
        int end = offsets[v + 1];

        for (int e = start + lane; e < end; e += 32) {
            if (is_edge_active(edge_mask, e)) {
                int u = indices[e];
                if (u != v && valid[u]) {
                    int old = atomicAdd(&delta_acc[u], delta);
                    if (old == 0) {
                        candidates[atomicAdd(candidates_size, 1)] = u;
                    }
                }
            }
        }
    }
}

__global__ void apply_updates_kernel(
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ delta_acc,
    const int32_t* __restrict__ candidates,
    const int32_t* __restrict__ candidates_size_ptr,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t k,
    int32_t k_delta,
    int64_t k_first
) {
    int candidates_size = *candidates_size_ptr;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < candidates_size; i += blockDim.x * gridDim.x) {
        int u = candidates[i];
        int total_delta = delta_acc[u];
        delta_acc[u] = 0;

        int old_core = core_numbers[u];
        int new_core = old_core >= total_delta ? old_core - total_delta : 0;
        if (new_core < k_delta) new_core = k_delta;
        if (k_first > 0 && new_core < (int32_t)k_first) new_core = 0;
        core_numbers[u] = new_core;

        if (new_core < k) {
            next_frontier[atomicAdd(next_frontier_size, 1)] = u;
        }
    }
}

__global__ void preprocess_k_first_kernel(
    int32_t* __restrict__ core_numbers,
    const int32_t* __restrict__ remaining,
    int32_t num_vertices,
    int64_t k_first
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        if (remaining[v] && core_numbers[v] < (int32_t)k_first) core_numbers[v] = 0;
    }
}

__global__ void postprocess_k_first_kernel(
    int32_t* __restrict__ core_numbers,
    int32_t num_vertices,
    int64_t k_first
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        if (core_numbers[v] > 0 && core_numbers[v] < (int32_t)k_first) core_numbers[v] = 0;
    }
}

__global__ void find_min_core_kernel(
    const int32_t* __restrict__ core_numbers,
    const int32_t* __restrict__ valid,
    const int32_t* __restrict__ remaining,
    int32_t* __restrict__ min_core,
    int32_t num_vertices
) {
    __shared__ int32_t smin;
    if (threadIdx.x == 0) smin = INT32_MAX;
    __syncthreads();

    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        if (valid[v] && remaining[v]) atomicMin(&smin, core_numbers[v]);
    }
    __syncthreads();

    if (threadIdx.x == 0 && smin < INT32_MAX) atomicMin(min_core, smin);
}



void launch_compute_degrees(const int32_t* offsets, const int32_t* indices,
    const uint32_t* edge_mask, int32_t* core_numbers, int32_t* remaining,
    int32_t* valid, int32_t num_vertices, int32_t multiplier) {
    if (num_vertices == 0) return;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    if (grid > 2048) grid = 2048;
    compute_degrees_kernel<<<grid, block>>>(offsets, indices, edge_mask, core_numbers,
        remaining, valid, num_vertices, multiplier);
}

void launch_build_frontier(const int32_t* core_numbers, const int32_t* remaining,
    const int32_t* valid, int32_t* frontier, int32_t* frontier_size,
    int32_t num_vertices, int32_t k) {
    if (num_vertices == 0) return;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    if (grid > 2048) grid = 2048;
    build_frontier_kernel<<<grid, block>>>(core_numbers, remaining, valid, frontier,
        frontier_size, num_vertices, k);
}

void launch_mark_invalid(int32_t* valid, const int32_t* frontier, int32_t frontier_size) {
    if (frontier_size == 0) return;
    int block = 256;
    int grid = (frontier_size + block - 1) / block;
    mark_invalid_kernel<<<grid, block>>>(valid, frontier, frontier_size);
}

void launch_accumulate_and_apply(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    const int32_t* valid, const int32_t* frontier, int32_t frontier_size,
    int32_t* delta_acc, int32_t* candidates, int32_t* candidates_size,
    int32_t* core_numbers, int32_t* next_frontier, int32_t* next_frontier_size,
    int32_t k, int32_t k_delta, int64_t k_first, int32_t delta,
    int32_t num_vertices) {
    if (frontier_size == 0) return;

    int block = 256;
    int warps_needed = frontier_size;
    int grid = (int)(((int64_t)warps_needed * 32 + block - 1) / block);
    if (grid > 2048) grid = 2048;
    accumulate_deltas_kernel<<<grid, block>>>(offsets, indices, edge_mask, valid, frontier,
        frontier_size, delta_acc, candidates, candidates_size, delta);

    int apply_grid = (num_vertices + block - 1) / block;
    if (apply_grid > 512) apply_grid = 512;
    apply_updates_kernel<<<apply_grid, block>>>(core_numbers, delta_acc, candidates,
        candidates_size, next_frontier, next_frontier_size, k, k_delta, k_first);
}

void launch_preprocess_k_first(int32_t* core_numbers, const int32_t* remaining,
    int32_t num_vertices, int64_t k_first) {
    if (num_vertices == 0) return;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    if (grid > 2048) grid = 2048;
    preprocess_k_first_kernel<<<grid, block>>>(core_numbers, remaining, num_vertices, k_first);
}

void launch_postprocess_k_first(int32_t* core_numbers, int32_t num_vertices, int64_t k_first) {
    if (num_vertices == 0) return;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    if (grid > 2048) grid = 2048;
    postprocess_k_first_kernel<<<grid, block>>>(core_numbers, num_vertices, k_first);
}

void launch_find_min_core(const int32_t* core_numbers, const int32_t* valid,
    const int32_t* remaining, int32_t* min_core, int32_t num_vertices) {
    if (num_vertices == 0) return;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    if (grid > 1024) grid = 1024;
    find_min_core_kernel<<<grid, block>>>(core_numbers, valid, remaining, min_core, num_vertices);
}

}  

void core_number_mask(const graph32_t& graph,
                      int32_t* core_numbers,
                      int degree_type,
                      std::size_t k_first,
                      std::size_t k_last) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;
    int32_t num_vertices = graph.number_of_vertices;

    if (num_vertices == 0) return;

    cache.ensure(num_vertices);

    int32_t* d_valid = cache.valid;
    int32_t* d_remaining = cache.remaining;
    int32_t* d_frontier_a = cache.frontier_a;
    int32_t* d_frontier_b = cache.frontier_b;
    int32_t* d_delta_acc = cache.delta_acc;
    int32_t* d_candidates = cache.candidates;
    int32_t* d_sizes = cache.sizes;
    int32_t* h_pinned = cache.h_pinned;

    int32_t multiplier = (degree_type == 2) ? 2 : 1;
    int32_t delta = (degree_type == 2) ? 2 : 1;

    launch_compute_degrees(d_offsets, d_indices, d_edge_mask, core_numbers,
        d_remaining, d_valid, num_vertices, multiplier);

    cudaMemsetAsync(d_delta_acc, 0, (size_t)num_vertices * sizeof(int32_t));

    if (k_first > 1) {
        launch_preprocess_k_first(core_numbers, d_remaining, num_vertices, (int64_t)k_first);
    }

    size_t k = (k_first >= 2) ? k_first : 2;
    if (delta == 2 && (k % 2) == 1) k++;

    int32_t* d_cur_frontier = d_frontier_a;
    int32_t* d_next_frontier = d_frontier_b;

    while (k <= k_last) {
        
        cudaMemsetAsync(&d_sizes[0], 0, sizeof(int32_t));
        launch_build_frontier(core_numbers, d_remaining, d_valid, d_cur_frontier,
            &d_sizes[0], num_vertices, (int32_t)k);

        cudaMemcpyAsync(&h_pinned[0], &d_sizes[0], sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        int32_t h_frontier_size = h_pinned[0];

        if (h_frontier_size == 0) {
            int32_t h_min_core = INT32_MAX;
            cudaMemcpyAsync(&d_sizes[3], &h_min_core, sizeof(int32_t), cudaMemcpyHostToDevice);
            launch_find_min_core(core_numbers, d_valid, d_remaining, &d_sizes[3], num_vertices);
            cudaMemcpyAsync(&h_pinned[0], &d_sizes[3], sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            h_min_core = h_pinned[0];

            if (h_min_core == INT32_MAX) break;

            size_t new_k = (size_t)h_min_core + (size_t)delta;
            if (new_k > k + (size_t)delta) k = new_k;
            else k += (size_t)delta;
            if (delta == 2 && (k % 2) == 1) k++;
            continue;
        }

        
        while (h_frontier_size > 0) {
            launch_mark_invalid(d_valid, d_cur_frontier, h_frontier_size);

            cudaMemsetAsync(&d_sizes[1], 0, sizeof(int32_t));
            cudaMemsetAsync(&d_sizes[2], 0, sizeof(int32_t));

            launch_accumulate_and_apply(
                d_offsets, d_indices, d_edge_mask, d_valid,
                d_cur_frontier, h_frontier_size,
                d_delta_acc, d_candidates, &d_sizes[2],
                core_numbers, d_next_frontier, &d_sizes[1],
                (int32_t)k, (int32_t)(k - (size_t)delta), (int64_t)k_first, delta,
                num_vertices);

            cudaMemcpyAsync(&h_pinned[0], &d_sizes[1], sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            h_frontier_size = h_pinned[0];

            int32_t* tmp = d_cur_frontier;
            d_cur_frontier = d_next_frontier;
            d_next_frontier = tmp;
        }

        k += (size_t)delta;
    }

    if (k_first > 1) {
        launch_postprocess_k_first(core_numbers, num_vertices, (int64_t)k_first);
    }
}

}  
