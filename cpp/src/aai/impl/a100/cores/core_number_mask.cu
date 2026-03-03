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
#include <algorithm>
#include <climits>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* removed = nullptr;
    int32_t* delta_acc = nullptr;
    int32_t* buf1 = nullptr;
    int32_t* buf2 = nullptr;
    int32_t* frontier = nullptr;
    int32_t* dest_list = nullptr;
    int32_t* counters = nullptr;
    int32_t vertex_capacity = 0;

    void ensure(int32_t n) {
        if (vertex_capacity >= n) return;
        if (removed) cudaFree(removed);
        if (delta_acc) cudaFree(delta_acc);
        if (buf1) cudaFree(buf1);
        if (buf2) cudaFree(buf2);
        if (frontier) cudaFree(frontier);
        if (dest_list) cudaFree(dest_list);
        if (counters) cudaFree(counters);
        cudaMalloc(&removed, n * sizeof(int32_t));
        cudaMalloc(&delta_acc, n * sizeof(int32_t));
        cudaMalloc(&buf1, n * sizeof(int32_t));
        cudaMalloc(&buf2, n * sizeof(int32_t));
        cudaMalloc(&frontier, n * sizeof(int32_t));
        cudaMalloc(&dest_list, n * sizeof(int32_t));
        cudaMalloc(&counters, 8 * sizeof(int32_t));
        vertex_capacity = n;
    }

    ~Cache() override {
        if (removed) cudaFree(removed);
        if (delta_acc) cudaFree(delta_acc);
        if (buf1) cudaFree(buf1);
        if (buf2) cudaFree(buf2);
        if (frontier) cudaFree(frontier);
        if (dest_list) cudaFree(dest_list);
        if (counters) cudaFree(counters);
    }
};

__device__ inline bool check_edge_mask(const uint32_t* mask, int32_t idx) {
    return (mask[idx >> 5] >> (idx & 31)) & 1;
}

__global__ void compute_out_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ core_numbers,
    int32_t num_vertices,
    int32_t multiplier
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t deg = 0;

    for (int32_t e = start; e < end; e++) {
        if (indices[e] != v && check_edge_mask(edge_mask, e)) {
            deg++;
        }
    }

    core_numbers[v] = deg * multiplier;
}

__global__ void init_remaining_kernel(
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ remaining,
    int32_t* __restrict__ remaining_count,
    int32_t num_vertices,
    int32_t k_first
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int32_t cn = core_numbers[v];
    if (cn > 0) {
        if (k_first > 1 && cn < k_first) {
            core_numbers[v] = 0;
        }
        int pos = atomicAdd(remaining_count, 1);
        remaining[pos] = v;
    }
}

__global__ void partition_kernel(
    const int32_t* __restrict__ core_numbers,
    const int32_t* __restrict__ remaining,
    int32_t remaining_size,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_count,
    int32_t* __restrict__ new_remaining,
    int32_t* __restrict__ new_remaining_count,
    int32_t k
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= remaining_size) return;

    int32_t v = remaining[tid];
    if (core_numbers[v] < k) {
        int pos = atomicAdd(frontier_count, 1);
        frontier[pos] = v;
    } else {
        int pos = atomicAdd(new_remaining_count, 1);
        new_remaining[pos] = v;
    }
}

__global__ void mark_removed_kernel(
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ removed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    removed[frontier[tid]] = 1;
}

__global__ void scatter_delta_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    const int32_t* __restrict__ removed,
    int32_t* __restrict__ delta_acc,
    int32_t delta,
    int32_t* __restrict__ dest_list,
    int32_t* __restrict__ dest_count
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t v = frontier[warp_id];
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    for (int32_t e = start + lane; e < end; e += 32) {
        int32_t u = indices[e];
        if (u != v && check_edge_mask(edge_mask, e) && !removed[u]) {
            int32_t old_val = atomicAdd(&delta_acc[u], delta);
            if (old_val == 0) {
                int pos = atomicAdd(dest_count, 1);
                dest_list[pos] = u;
            }
        }
    }
}

__global__ void apply_and_build_frontier_kernel(
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ delta_acc,
    const int32_t* __restrict__ dest_list,
    int32_t dest_count,
    int32_t* __restrict__ new_frontier,
    int32_t* __restrict__ new_frontier_count,
    int32_t k,
    int32_t k_minus_delta,
    int32_t k_first
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= dest_count) return;

    int32_t v = dest_list[tid];
    int32_t acc = delta_acc[v];
    delta_acc[v] = 0;

    if (acc > 0) {
        int32_t old_val = core_numbers[v];
        int32_t new_val = old_val >= acc ? old_val - acc : 0;
        if (new_val < k_minus_delta) new_val = k_minus_delta;
        if (new_val < k_first) new_val = 0;
        core_numbers[v] = new_val;

        if (new_val < k) {
            int pos = atomicAdd(new_frontier_count, 1);
            new_frontier[pos] = v;
        }
    }
}

__global__ void compact_remaining_kernel(
    const int32_t* __restrict__ remaining_in,
    int32_t remaining_in_size,
    const int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ remaining_out,
    int32_t* __restrict__ remaining_out_count,
    int32_t k
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= remaining_in_size) return;

    int32_t v = remaining_in[tid];
    if (core_numbers[v] >= k) {
        int pos = atomicAdd(remaining_out_count, 1);
        remaining_out[pos] = v;
    }
}

__global__ void find_min_kernel(
    const int32_t* __restrict__ core_numbers,
    const int32_t* __restrict__ remaining,
    int32_t remaining_size,
    int32_t* __restrict__ min_result
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= remaining_size) return;
    atomicMin(min_result, core_numbers[remaining[tid]]);
}

void launch_compute_out_degrees(const int32_t* offsets, const int32_t* indices,
    const uint32_t* edge_mask, int32_t* core_numbers,
    int32_t num_vertices, int32_t multiplier, cudaStream_t stream) {
    if (num_vertices == 0) return;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    compute_out_degrees_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, edge_mask, core_numbers, num_vertices, multiplier);
}

void launch_init_remaining(int32_t* core_numbers, int32_t* remaining, int32_t* remaining_count,
    int32_t num_vertices, int32_t k_first, cudaStream_t stream) {
    if (num_vertices == 0) return;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    init_remaining_kernel<<<grid, block, 0, stream>>>(
        core_numbers, remaining, remaining_count, num_vertices, k_first);
}

void launch_partition(const int32_t* core_numbers, const int32_t* remaining, int32_t remaining_size,
    int32_t* frontier, int32_t* frontier_count,
    int32_t* new_remaining, int32_t* new_remaining_count,
    int32_t k, cudaStream_t stream) {
    if (remaining_size == 0) return;
    int block = 256;
    int grid = (remaining_size + block - 1) / block;
    partition_kernel<<<grid, block, 0, stream>>>(
        core_numbers, remaining, remaining_size,
        frontier, frontier_count, new_remaining, new_remaining_count, k);
}

void launch_mark_removed(const int32_t* frontier, int32_t frontier_size,
    int32_t* removed, cudaStream_t stream) {
    if (frontier_size == 0) return;
    int block = 256;
    int grid = (frontier_size + block - 1) / block;
    mark_removed_kernel<<<grid, block, 0, stream>>>(frontier, frontier_size, removed);
}

void launch_scatter_delta(const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    const int32_t* frontier, int32_t frontier_size,
    const int32_t* removed, int32_t* delta_acc, int32_t delta,
    int32_t* dest_list, int32_t* dest_count, cudaStream_t stream) {
    if (frontier_size == 0) return;
    int block = 256;
    int64_t threads_needed = (int64_t)frontier_size * 32;
    int grid = (int)((threads_needed + block - 1) / block);
    scatter_delta_warp_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, edge_mask, frontier, frontier_size,
        removed, delta_acc, delta, dest_list, dest_count);
}

void launch_apply_and_build_frontier(int32_t* core_numbers, int32_t* delta_acc,
    const int32_t* dest_list, int32_t dest_count,
    int32_t* new_frontier, int32_t* new_frontier_count,
    int32_t k, int32_t k_minus_delta, int32_t k_first, cudaStream_t stream) {
    if (dest_count == 0) return;
    int block = 256;
    int grid = (dest_count + block - 1) / block;
    apply_and_build_frontier_kernel<<<grid, block, 0, stream>>>(
        core_numbers, delta_acc, dest_list, dest_count,
        new_frontier, new_frontier_count, k, k_minus_delta, k_first);
}

void launch_compact_remaining(const int32_t* remaining_in, int32_t remaining_in_size,
    const int32_t* core_numbers, int32_t* remaining_out, int32_t* remaining_out_count,
    int32_t k, cudaStream_t stream) {
    if (remaining_in_size == 0) return;
    int block = 256;
    int grid = (remaining_in_size + block - 1) / block;
    compact_remaining_kernel<<<grid, block, 0, stream>>>(
        remaining_in, remaining_in_size, core_numbers, remaining_out, remaining_out_count, k);
}

void launch_find_min(const int32_t* core_numbers, const int32_t* remaining,
    int32_t remaining_size, int32_t* min_result, cudaStream_t stream) {
    if (remaining_size == 0) return;
    int block = 256;
    int grid = (remaining_size + block - 1) / block;
    find_min_kernel<<<grid, block, 0, stream>>>(
        core_numbers, remaining, remaining_size, min_result);
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

    cudaStream_t stream = 0;
    if (num_vertices == 0) return;

    cache.ensure(num_vertices);

    int32_t* d_core = core_numbers;

    bool is_symmetric = true;
    int32_t multiplier = (degree_type == 2 && is_symmetric) ? 2 : 1;
    int32_t delta = multiplier;

    int32_t* d_removed = cache.removed;
    int32_t* d_delta = cache.delta_acc;
    int32_t* d_buf1 = cache.buf1;
    int32_t* d_buf2 = cache.buf2;
    int32_t* d_frontier = cache.frontier;
    int32_t* d_dest_list = cache.dest_list;
    int32_t* d_counters = cache.counters;

    cudaMemsetAsync(d_removed, 0, num_vertices * sizeof(int32_t), stream);
    cudaMemsetAsync(d_delta, 0, num_vertices * sizeof(int32_t), stream);
    cudaMemsetAsync(d_counters, 0, 8 * sizeof(int32_t), stream);

    
    launch_compute_out_degrees(d_offsets, d_indices, d_edge_mask, d_core,
                               num_vertices, multiplier, stream);

    
    int32_t k_first_i32 = (k_first > (size_t)INT32_MAX) ? INT32_MAX : (int32_t)k_first;
    launch_init_remaining(d_core, d_buf1, &d_counters[0], num_vertices, k_first_i32, stream);

    int32_t h_remaining_size = 0;
    cudaMemcpyAsync(&h_remaining_size, &d_counters[0], sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (h_remaining_size == 0) return;

    int32_t* d_remaining = d_buf1;
    int32_t* d_remaining_alt = d_buf2;
    int32_t remaining_size = h_remaining_size;

    
    size_t k = (k_first < 2) ? 2 : k_first;
    if (is_symmetric && degree_type == 2 && (k % 2) == 1) k++;

    while (k <= k_last && remaining_size > 0) {
        
        cudaMemsetAsync(&d_counters[0], 0, 2 * sizeof(int32_t), stream);
        launch_partition(d_core, d_remaining, remaining_size,
                        d_frontier, &d_counters[0],
                        d_remaining_alt, &d_counters[1],
                        (int32_t)k, stream);

        int32_t h_counts[2];
        cudaMemcpyAsync(h_counts, &d_counters[0], 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        int32_t h_frontier_size = h_counts[0];
        std::swap(d_remaining, d_remaining_alt);
        remaining_size = h_counts[1];

        if (h_frontier_size > 0) {
            int32_t k_minus_delta = (int32_t)(k - delta);

            
            do {
                
                launch_mark_removed(d_frontier, h_frontier_size, d_removed, stream);
                cudaMemsetAsync(&d_counters[2], 0, sizeof(int32_t), stream);
                launch_scatter_delta(d_offsets, d_indices, d_edge_mask,
                                    d_frontier, h_frontier_size,
                                    d_removed, d_delta, delta,
                                    d_dest_list, &d_counters[2], stream);

                
                int32_t h_dest_count = 0;
                cudaMemcpyAsync(&h_dest_count, &d_counters[2], sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);

                if (h_dest_count == 0) {
                    h_frontier_size = 0;
                    break;
                }

                
                cudaMemsetAsync(&d_counters[3], 0, sizeof(int32_t), stream);
                launch_apply_and_build_frontier(
                    d_core, d_delta, d_dest_list, h_dest_count,
                    d_frontier, &d_counters[3],
                    (int32_t)k, k_minus_delta, k_first_i32, stream);

                cudaMemcpyAsync(&h_frontier_size, &d_counters[3], sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);

            } while (h_frontier_size > 0);

            
            cudaMemsetAsync(&d_counters[4], 0, sizeof(int32_t), stream);
            launch_compact_remaining(d_remaining, remaining_size, d_core,
                                    d_remaining_alt, &d_counters[4], (int32_t)k, stream);
            cudaMemcpyAsync(&remaining_size, &d_counters[4], sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            std::swap(d_remaining, d_remaining_alt);
            k += delta;
        } else {
            if (remaining_size == 0) break;

            int32_t h_min_core = INT32_MAX;
            cudaMemcpyAsync(&d_counters[5], &h_min_core, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
            launch_find_min(d_core, d_remaining, remaining_size, &d_counters[5], stream);
            cudaMemcpyAsync(&h_min_core, &d_counters[5], sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            size_t new_k = std::max(k + delta, (size_t)(h_min_core + delta));
            if (is_symmetric && degree_type == 2 && (new_k % 2) == 1) new_k++;
            k = new_k;
        }
    }
}

}  
