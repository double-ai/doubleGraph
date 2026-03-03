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
#include <cooperative_groups.h>
#include <cstdint>
#include <cstddef>

namespace aai {

namespace {

namespace cg = cooperative_groups;

struct Cache : Cacheable {
    int8_t* removed = nullptr;
    int32_t* delta_acc = nullptr;
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* affected = nullptr;
    int32_t* counters = nullptr;
    int32_t capacity = 0;

    void ensure(int32_t n) {
        if (capacity >= n) return;
        if (removed) cudaFree(removed);
        if (delta_acc) cudaFree(delta_acc);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (affected) cudaFree(affected);
        if (counters) cudaFree(counters);
        cudaMalloc(&removed, (size_t)n * sizeof(int8_t));
        cudaMalloc(&delta_acc, (size_t)n * sizeof(int32_t));
        cudaMalloc(&frontier_a, (size_t)n * sizeof(int32_t));
        cudaMalloc(&frontier_b, (size_t)n * sizeof(int32_t));
        cudaMalloc(&affected, (size_t)n * sizeof(int32_t));
        cudaMalloc(&counters, 8 * sizeof(int32_t));
        capacity = n;
    }

    ~Cache() override {
        if (removed) cudaFree(removed);
        if (delta_acc) cudaFree(delta_acc);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (affected) cudaFree(affected);
        if (counters) cudaFree(counters);
    }
};

__global__ void core_number_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ core_numbers,
    int8_t* __restrict__ removed,
    int32_t* __restrict__ delta_acc,
    int32_t* __restrict__ frontier_a,
    int32_t* __restrict__ frontier_b,
    int32_t* __restrict__ affected,
    int32_t* __restrict__ counters,
    int32_t num_vertices,
    int32_t delta,
    int32_t degree_multiplier,
    int32_t k_first_i32,
    int32_t k_last_i32,
    int32_t k_last_is_max
) {
    cg::grid_group grid = cg::this_grid();
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    const int lane = threadIdx.x & 31;
    const int warp_id_global = tid >> 5;
    const int num_warps = stride >> 5;

    
    for (int v = tid; v < num_vertices; v += stride) {
        int start = offsets[v], end = offsets[v + 1];
        int self_loops = 0;
        for (int e = start; e < end; e++) {
            if (indices[e] == v) self_loops++;
        }
        core_numbers[v] = (end - start - self_loops) * degree_multiplier;
        removed[v] = 0;
        delta_acc[v] = 0;
    }
    grid.sync();

    
    if (k_first_i32 > 1) {
        for (int v = tid; v < num_vertices; v += stride) {
            int32_t cn = core_numbers[v];
            if (cn > 0 && cn < k_first_i32) {
                core_numbers[v] = 0;
            }
        }
        grid.sync();
    }

    
    int32_t* cur_frontier = frontier_a;
    int32_t* nxt_frontier = frontier_b;

    int32_t k = (k_first_i32 > 2) ? k_first_i32 : 2;
    if (delta == 2 && (k & 1)) k++;

    for (;;) {
        if (!k_last_is_max && k > k_last_i32) break;

        
        if (tid == 0) counters[0] = 0;
        grid.sync();

        for (int v = tid; v < num_vertices; v += stride) {
            if (!removed[v] && core_numbers[v] < k) {
                int pos = atomicAdd(&counters[0], 1);
                cur_frontier[pos] = v;
            }
        }
        grid.sync();

        int32_t frontier_size = counters[0];

        if (frontier_size == 0) {
            
            if (tid == 0) counters[4] = 0x7FFFFFFF;
            grid.sync();

            for (int v = tid; v < num_vertices; v += stride) {
                if (!removed[v]) {
                    atomicMin(&counters[4], core_numbers[v]);
                }
            }
            grid.sync();

            int32_t min_core = counters[4];
            if (min_core == 0x7FFFFFFF) break;

            int32_t next_k = min_core + delta;
            if (delta == 2 && (next_k & 1)) next_k++;
            if (next_k <= k) next_k = k + delta;
            k = next_k;
            continue;
        }

        
        while (frontier_size > 0) {
            
            for (int i = tid; i < frontier_size; i += stride) {
                removed[cur_frontier[i]] = 1;
            }
            grid.sync();

            
            if (tid == 0) counters[2] = 0;
            grid.sync();

            
            for (int i = warp_id_global; i < frontier_size; i += num_warps) {
                int v = cur_frontier[i];
                int start = offsets[v];
                int end = offsets[v + 1];

                for (int e = start + lane; e < end; e += 32) {
                    int u = indices[e];
                    if (!removed[u]) {
                        int old = atomicAdd(&delta_acc[u], delta);
                        if (old == 0) {
                            int pos = atomicAdd(&counters[2], 1);
                            affected[pos] = u;
                        }
                    }
                }
            }
            grid.sync();

            int32_t affected_count = counters[2];

            
            if (tid == 0) counters[1] = 0;
            grid.sync();

            int32_t k_minus_delta = k - delta;
            for (int i = tid; i < affected_count; i += stride) {
                int u = affected[i];
                int32_t old_val = core_numbers[u];
                int32_t d = delta_acc[u];
                delta_acc[u] = 0;

                int32_t new_val = old_val >= d ? old_val - d : 0;
                if (new_val < k_minus_delta) new_val = k_minus_delta;
                if (new_val < k_first_i32) new_val = 0;
                core_numbers[u] = new_val;

                if (new_val < k) {
                    int pos = atomicAdd(&counters[1], 1);
                    nxt_frontier[pos] = u;
                }
            }
            grid.sync();

            frontier_size = counters[1];

            
            int32_t* tmp = cur_frontier;
            cur_frontier = nxt_frontier;
            nxt_frontier = tmp;
        }

        k += delta;
    }
}

void launch_core_number(
    const int32_t* offsets, const int32_t* indices,
    int32_t* core_numbers, int8_t* removed, int32_t* delta_acc,
    int32_t* frontier_a, int32_t* frontier_b, int32_t* affected,
    int32_t* counters,
    int32_t num_vertices, int32_t delta, int32_t degree_multiplier,
    int32_t k_first, int32_t k_last, int32_t k_last_is_max,
    int block_size
) {
    if (num_vertices == 0) return;

    int num_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm, core_number_kernel, block_size, 0);

    int device;
    cudaGetDevice(&device);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);

    int grid_size = num_blocks_per_sm * num_sms;
    if (grid_size < 1) grid_size = 1;

    void* args[] = {
        (void*)&offsets, (void*)&indices,
        (void*)&core_numbers, (void*)&removed, (void*)&delta_acc,
        (void*)&frontier_a, (void*)&frontier_b, (void*)&affected,
        (void*)&counters,
        (void*)&num_vertices, (void*)&delta, (void*)&degree_multiplier,
        (void*)&k_first, (void*)&k_last, (void*)&k_last_is_max
    };

    cudaLaunchCooperativeKernel(
        (void*)core_number_kernel,
        dim3(grid_size), dim3(block_size),
        args, 0
    );
}

}  

void core_number(const graph32_t& graph,
                 int32_t* core_numbers,
                 int degree_type,
                 std::size_t k_first,
                 std::size_t k_last) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;

    if (num_vertices == 0) return;

    bool is_inout_symmetric = graph.is_symmetric && (degree_type == 2);
    int32_t delta = is_inout_symmetric ? 2 : 1;
    int32_t degree_multiplier = is_inout_symmetric ? 2 : 1;

    int32_t k_first_i32 = (k_first > 0x7FFFFFFFU) ? 0x7FFFFFFF : (int32_t)k_first;
    int32_t k_last_i32 = (k_last > 0x7FFFFFFFU) ? 0x7FFFFFFF : (int32_t)k_last;
    int32_t k_last_is_max = (k_last >= 0x7FFFFFFFU) ? 1 : 0;

    cache.ensure(num_vertices);

    launch_core_number(
        offsets, indices,
        core_numbers,
        cache.removed,
        cache.delta_acc,
        cache.frontier_a,
        cache.frontier_b,
        cache.affected,
        cache.counters,
        num_vertices, delta, degree_multiplier,
        k_first_i32, k_last_i32, k_last_is_max,
        256
    );

    cudaDeviceSynchronize();
}

}  
