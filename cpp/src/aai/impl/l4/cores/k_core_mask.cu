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
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    
    int32_t* core_numbers_buf = nullptr;
    int32_t* degrees = nullptr;
    int32_t* valid = nullptr;
    int32_t* remaining = nullptr;
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* delta_buffer = nullptr;
    int32_t* modified_list = nullptr;
    int64_t vert_cap = 0;

    
    int32_t* counters = nullptr;   
    int64_t* edge_count = nullptr; 
    int64_t counters_cap = 0;
    int64_t edge_count_cap = 0;

    void ensure(int32_t num_vertices) {
        if (vert_cap < num_vertices) {
            if (core_numbers_buf) cudaFree(core_numbers_buf);
            if (degrees) cudaFree(degrees);
            if (valid) cudaFree(valid);
            if (remaining) cudaFree(remaining);
            if (frontier_a) cudaFree(frontier_a);
            if (frontier_b) cudaFree(frontier_b);
            if (delta_buffer) cudaFree(delta_buffer);
            if (modified_list) cudaFree(modified_list);

            cudaMalloc(&core_numbers_buf, (size_t)num_vertices * sizeof(int32_t));
            cudaMalloc(&degrees, (size_t)num_vertices * sizeof(int32_t));
            cudaMalloc(&valid, (size_t)num_vertices * sizeof(int32_t));
            cudaMalloc(&remaining, (size_t)num_vertices * sizeof(int32_t));
            cudaMalloc(&frontier_a, (size_t)num_vertices * sizeof(int32_t));
            cudaMalloc(&frontier_b, (size_t)num_vertices * sizeof(int32_t));
            cudaMalloc(&delta_buffer, (size_t)num_vertices * sizeof(int32_t));
            cudaMalloc(&modified_list, (size_t)num_vertices * sizeof(int32_t));
            vert_cap = num_vertices;
        }
        if (counters_cap < 4) {
            if (counters) cudaFree(counters);
            cudaMalloc(&counters, 4 * sizeof(int32_t));
            counters_cap = 4;
        }
        if (edge_count_cap < 1) {
            if (edge_count) cudaFree(edge_count);
            cudaMalloc(&edge_count, sizeof(int64_t));
            edge_count_cap = 1;
        }
    }

    ~Cache() override {
        if (core_numbers_buf) cudaFree(core_numbers_buf);
        if (degrees) cudaFree(degrees);
        if (valid) cudaFree(valid);
        if (remaining) cudaFree(remaining);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (delta_buffer) cudaFree(delta_buffer);
        if (modified_list) cudaFree(modified_list);
        if (counters) cudaFree(counters);
        if (edge_count) cudaFree(edge_count);
    }
};

__device__ __forceinline__ bool is_edge_active(const uint32_t* edge_mask, int32_t edge_idx) {
    return (edge_mask[edge_idx >> 5] >> (edge_idx & 31)) & 1;
}

__global__ void compute_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ degrees,
    int32_t num_vertices,
    int32_t degree_multiplier
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t deg = 0;
    for (int32_t e = start; e < end; e++) {
        if (is_edge_active(edge_mask, e) && indices[e] != v) {
            deg++;
        }
    }
    degrees[v] = deg * degree_multiplier;
}

__global__ void init_decomp_kernel(
    const int32_t* __restrict__ degrees,
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ valid,
    int32_t* __restrict__ remaining,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    core_numbers[v] = degrees[v];
    valid[v] = 1;
    remaining[v] = (degrees[v] > 0) ? 1 : 0;
}

__global__ void find_frontier_kernel(
    const int32_t* __restrict__ core_numbers,
    const int32_t* __restrict__ remaining,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_count,
    int32_t num_vertices,
    int32_t k
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    if (remaining[v] && core_numbers[v] < k) {
        int pos = atomicAdd(frontier_count, 1);
        frontier[pos] = v;
    }
}

__global__ void count_remaining_kernel(
    const int32_t* __restrict__ remaining,
    int32_t* __restrict__ count,
    int32_t num_vertices
) {
    extern __shared__ int32_t sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t local = (i < num_vertices && remaining[i]) ? 1 : 0;
    sdata[tid] = local;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(count, sdata[0]);
}

__global__ void mark_frontier_kernel(
    int32_t* __restrict__ valid,
    int32_t* __restrict__ remaining,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int32_t v = frontier[tid];
    valid[v] = 0;
    remaining[v] = 0;
}

__global__ void accumulate_delta_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ valid,
    int32_t* __restrict__ delta_buffer,
    int32_t* __restrict__ modified_list,
    int32_t* __restrict__ modified_count,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t delta
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int32_t v = frontier[tid];
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    for (int32_t e = start; e < end; e++) {
        if (!is_edge_active(edge_mask, e)) continue;
        int32_t u = indices[e];
        if (u == v) continue;
        if (!valid[u]) continue;

        int32_t old_delta = atomicAdd(&delta_buffer[u], delta);
        if (old_delta == 0) {
            int pos = atomicAdd(modified_count, 1);
            modified_list[pos] = u;
        }
    }
}

__global__ void update_and_next_frontier_kernel(
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ delta_buffer,
    const int32_t* __restrict__ modified_list,
    int32_t modified_count,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_count,
    int32_t k,
    int32_t k_minus_delta
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= modified_count) return;

    int32_t u = modified_list[tid];
    int32_t pushed_val = delta_buffer[u];
    delta_buffer[u] = 0;

    int32_t old_core = core_numbers[u];
    int32_t new_core = (old_core >= pushed_val) ? (old_core - pushed_val) : 0;
    if (new_core < k_minus_delta) new_core = k_minus_delta;

    core_numbers[u] = new_core;

    if (new_core < k) {
        int pos = atomicAdd(next_frontier_count, 1);
        next_frontier[pos] = u;
    }
}

__global__ void find_min_core_kernel(
    const int32_t* __restrict__ core_numbers,
    const int32_t* __restrict__ remaining,
    int32_t* __restrict__ min_val,
    int32_t num_vertices
) {
    extern __shared__ int32_t sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = INT32_MAX;
    if (i < num_vertices && remaining[i]) {
        sdata[tid] = core_numbers[i];
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0 && sdata[0] < INT32_MAX) {
        atomicMin(min_val, sdata[0]);
    }
}

__global__ void update_remaining_kernel(
    int32_t* __restrict__ remaining,
    const int32_t* __restrict__ core_numbers,
    int32_t num_vertices,
    int32_t k
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    if (remaining[v] && core_numbers[v] < k) {
        remaining[v] = 0;
    }
}


__global__ void init_removed_from_cn_kernel(
    const int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ removed,
    int32_t num_vertices,
    int32_t k
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    removed[v] = (core_numbers[v] < k) ? 1 : 0;
}

__global__ void extract_edges_cn_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ core_numbers,
    int32_t num_vertices,
    int32_t k,
    int32_t* __restrict__ edge_srcs,
    int32_t* __restrict__ edge_dsts,
    int64_t* __restrict__ edge_count
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    if (core_numbers[v] < k) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t local_count = 0;
    for (int32_t e = start; e < end; e++) {
        if (is_edge_active(edge_mask, e) && core_numbers[indices[e]] >= k) {
            local_count++;
        }
    }
    if (local_count > 0) {
        int64_t base = (int64_t)atomicAdd((unsigned long long*)edge_count, (unsigned long long)local_count);
        int64_t pos = base;
        for (int32_t e = start; e < end; e++) {
            if (is_edge_active(edge_mask, e) && core_numbers[indices[e]] >= k) {
                edge_srcs[pos] = v;
                edge_dsts[pos] = indices[e];
                pos++;
            }
        }
    }
}

}  

std::size_t k_core_mask(const graph32_t& graph,
                        std::size_t k,
                        int degree_type,
                        const int32_t* core_numbers,
                        int32_t* edge_srcs,
                        int32_t* edge_dsts,
                        std::size_t max_edges) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;
    bool is_symmetric = graph.is_symmetric;
    int32_t k_target = static_cast<int32_t>(k);

    if (num_vertices == 0 || num_edges == 0) {
        return 0;
    }

    cache.ensure(num_vertices);

    const int32_t* d_core_numbers;

    if (core_numbers != nullptr) {
        d_core_numbers = core_numbers;
    } else {
        
        bool is_inout = (degree_type == 2);
        int32_t degree_multiplier = (is_symmetric && is_inout) ? 2 : 1;
        int32_t delta = (is_symmetric && is_inout) ? 2 : 1;

        int32_t* d_core_buf = cache.core_numbers_buf;
        int32_t* d_degrees = cache.degrees;
        int32_t* d_valid = cache.valid;
        int32_t* d_remaining = cache.remaining;
        int32_t* d_frontier_a = cache.frontier_a;
        int32_t* d_frontier_b = cache.frontier_b;
        int32_t* d_delta_buffer = cache.delta_buffer;
        int32_t* d_modified_list = cache.modified_list;
        int32_t* d_counters = cache.counters;

        
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 0)
            compute_degrees_kernel<<<grid, block>>>(d_offsets, d_indices, d_edge_mask, d_degrees, num_vertices, degree_multiplier);

        
        if (grid > 0)
            init_decomp_kernel<<<grid, block>>>(d_degrees, d_core_buf, d_valid, d_remaining, num_vertices);

        
        cudaMemset(d_delta_buffer, 0, (size_t)num_vertices * sizeof(int32_t));

        
        int32_t k_decomp = 2;
        if (is_symmetric && is_inout && (k_decomp % 2) == 1) k_decomp++;

        while (true) {
            
            cudaMemset(d_counters + 3, 0, sizeof(int32_t));
            if (grid > 0)
                count_remaining_kernel<<<grid, block, block * sizeof(int32_t)>>>(d_remaining, d_counters + 3, num_vertices);
            int32_t h_remaining;
            cudaMemcpy(&h_remaining, d_counters + 3, sizeof(int32_t), cudaMemcpyDeviceToHost);
            if (h_remaining == 0) break;

            
            cudaMemset(d_counters, 0, sizeof(int32_t));
            if (grid > 0)
                find_frontier_kernel<<<grid, block>>>(d_core_buf, d_remaining, d_frontier_a, d_counters, num_vertices, k_decomp);
            int32_t h_frontier_size;
            cudaMemcpy(&h_frontier_size, d_counters, sizeof(int32_t), cudaMemcpyDeviceToHost);

            if (h_frontier_size == 0) {
                
                int32_t h_min_core = INT32_MAX;
                cudaMemcpy(d_counters + 3, &h_min_core, sizeof(int32_t), cudaMemcpyHostToDevice);
                if (grid > 0)
                    find_min_core_kernel<<<grid, block, block * sizeof(int32_t)>>>(d_core_buf, d_remaining, d_counters + 3, num_vertices);
                cudaMemcpy(&h_min_core, d_counters + 3, sizeof(int32_t), cudaMemcpyDeviceToHost);

                k_decomp = std::max(k_decomp + delta, h_min_core + delta);
                if (is_symmetric && is_inout && (k_decomp % 2) == 1) k_decomp++;
                continue;
            }

            
            {
                int fg = (h_frontier_size + block - 1) / block;
                if (fg > 0)
                    mark_frontier_kernel<<<fg, block>>>(d_valid, d_remaining, d_frontier_a, h_frontier_size);
            }

            
            int32_t* cur_frontier = d_frontier_a;
            int32_t* next_frontier = d_frontier_b;
            int32_t cur_frontier_size = h_frontier_size;

            while (cur_frontier_size > 0) {
                
                cudaMemset(d_counters + 1, 0, sizeof(int32_t));
                {
                    int fg = (cur_frontier_size + block - 1) / block;
                    if (fg > 0)
                        accumulate_delta_kernel<<<fg, block>>>(d_offsets, d_indices, d_edge_mask, d_valid,
                            d_delta_buffer, d_modified_list, d_counters + 1,
                            cur_frontier, cur_frontier_size, delta);
                }
                int32_t h_modified_count;
                cudaMemcpy(&h_modified_count, d_counters + 1, sizeof(int32_t), cudaMemcpyDeviceToHost);

                if (h_modified_count == 0) break;

                
                cudaMemset(d_counters + 2, 0, sizeof(int32_t));
                {
                    int fg = (h_modified_count + block - 1) / block;
                    if (fg > 0)
                        update_and_next_frontier_kernel<<<fg, block>>>(d_core_buf, d_delta_buffer,
                            d_modified_list, h_modified_count,
                            next_frontier, d_counters + 2,
                            k_decomp, k_decomp - delta);
                }
                int32_t h_next_frontier_size;
                cudaMemcpy(&h_next_frontier_size, d_counters + 2, sizeof(int32_t), cudaMemcpyDeviceToHost);

                
                if (h_next_frontier_size > 0) {
                    int fg = (h_next_frontier_size + block - 1) / block;
                    if (fg > 0)
                        mark_frontier_kernel<<<fg, block>>>(d_valid, d_remaining, next_frontier, h_next_frontier_size);
                }

                cur_frontier_size = h_next_frontier_size;
                int32_t* tmp = cur_frontier;
                cur_frontier = next_frontier;
                next_frontier = tmp;
            }

            
            if (grid > 0)
                update_remaining_kernel<<<grid, block>>>(d_remaining, d_core_buf, num_vertices, k_decomp);
            k_decomp += delta;
        }

        d_core_numbers = d_core_buf;
    }

    
    cudaMemset(cache.edge_count, 0, sizeof(int64_t));
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 0)
            extract_edges_cn_kernel<<<grid, block>>>(d_offsets, d_indices, d_edge_mask, d_core_numbers,
                num_vertices, k_target, edge_srcs, edge_dsts, cache.edge_count);
    }

    int64_t actual_edges;
    cudaMemcpy(&actual_edges, cache.edge_count, sizeof(int64_t), cudaMemcpyDeviceToHost);

    return static_cast<std::size_t>(actual_edges);
}

}  
