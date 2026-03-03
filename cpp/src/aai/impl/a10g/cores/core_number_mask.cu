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
#include <climits>

namespace aai {

namespace {

namespace cg = cooperative_groups;

__global__ void compute_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ core_numbers,
    int num_vertices,
    int degree_mult)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int deg = 0;
        for (int e = start; e < end; e++) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1u) {
                if (indices[e] != v) deg++;
            }
        }
        core_numbers[v] = deg * degree_mult;
    }
}



__global__ void core_number_coop_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ core_numbers,
    int8_t* __restrict__ removed,
    int32_t* __restrict__ push_count,
    int32_t* __restrict__ remaining_a,
    int32_t* __restrict__ remaining_b,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ affected,
    int32_t* __restrict__ counters,
    int32_t* __restrict__ block_min_buf,
    int num_vertices,
    int delta,
    int k_first,
    int64_t k_last_i64,
    int is_inout)
{
    auto grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = gridDim.x * blockDim.x;
    extern __shared__ int smem[];

    const size_t k_last = static_cast<size_t>(k_last_i64);
    const int warp_id_g = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = grid_size >> 5;

    
    if (tid == 0) counters[0] = 0;
    grid.sync();

    for (int v = tid; v < num_vertices; v += grid_size) {
        if (core_numbers[v] > 0)
            remaining_a[atomicAdd(&counters[0], 1)] = v;
    }
    grid.sync();

    int remaining_count = counters[0];

    
    if (k_first > 1) {
        for (int i = tid; i < remaining_count; i += grid_size) {
            int v = remaining_a[i];
            if (core_numbers[v] < k_first) core_numbers[v] = 0;
        }
        grid.sync();
    }

    
    size_t k = (static_cast<size_t>(k_first) > 2) ? static_cast<size_t>(k_first) : 2;
    if (is_inout && (k & 1)) k++;

    int32_t* cur_rem = remaining_a;
    int32_t* next_rem = remaining_b;

    while (k <= k_last && remaining_count > 0) {
        const int ki32 = (int)k;

        
        if (tid == 0) { counters[0] = 0; counters[1] = 0; }
        grid.sync();

        for (int i = tid; i < remaining_count; i += grid_size) {
            int v = cur_rem[i];
            if (removed[v]) continue;
            if (core_numbers[v] >= ki32)
                next_rem[atomicAdd(&counters[0], 1)] = v;
            else
                frontier[atomicAdd(&counters[1], 1)] = v;
        }
        grid.sync();

        remaining_count = counters[0];
        int frontier_count = counters[1];
        int32_t* tmp = cur_rem; cur_rem = next_rem; next_rem = tmp;

        if (frontier_count > 0) {
            while (frontier_count > 0) {
                
                
                if (tid == 0) counters[2] = 0;
                for (int i = tid; i < frontier_count; i += grid_size)
                    removed[frontier[i]] = 1;
                grid.sync();

                
                for (int i = warp_id_g; i < frontier_count; i += num_warps) {
                    int v = frontier[i];
                    int start = offsets[v];
                    int end = offsets[v + 1];

                    for (int e = start + lane; e < end; e += 32) {
                        if ((edge_mask[e >> 5] >> (e & 31)) & 1u) {
                            int u = indices[e];
                            if (u != v && !removed[u]) {
                                int old = atomicAdd(&push_count[u], delta);
                                if (old == 0)
                                    affected[atomicAdd(&counters[2], 1)] = u;
                            }
                        }
                    }
                }

                
                if (tid == 0) counters[3] = 0;
                grid.sync();

                int affected_count = counters[2];
                if (affected_count == 0) break;

                
                const int floor_val = ki32 - delta;
                for (int i = tid; i < affected_count; i += grid_size) {
                    int v = affected[i];
                    int pushed = push_count[v];
                    push_count[v] = 0;

                    int old_core = core_numbers[v];
                    int new_core = old_core >= pushed ? old_core - pushed : 0;
                    if (new_core < floor_val) new_core = floor_val;
                    if (new_core < k_first) new_core = 0;
                    core_numbers[v] = new_core;

                    if (new_core < ki32)
                        frontier[atomicAdd(&counters[3], 1)] = v;
                }

                
                grid.sync();

                frontier_count = counters[3];
            }
            k += delta;
        } else {
            
            int local_min = INT32_MAX;
            for (int i = tid; i < remaining_count; i += grid_size) {
                int c = core_numbers[cur_rem[i]];
                if (c < local_min) local_min = c;
            }
            smem[threadIdx.x] = local_min;
            __syncthreads();
            for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
                if (threadIdx.x < s) {
                    int o = smem[threadIdx.x + s];
                    if (o < smem[threadIdx.x]) smem[threadIdx.x] = o;
                }
                __syncthreads();
            }
            if (threadIdx.x == 0) block_min_buf[blockIdx.x] = smem[0];
            grid.sync();

            if (blockIdx.x == 0) {
                local_min = INT32_MAX;
                for (int i = threadIdx.x; i < (int)gridDim.x; i += (int)blockDim.x) {
                    int v = block_min_buf[i];
                    if (v < local_min) local_min = v;
                }
                smem[threadIdx.x] = local_min;
                __syncthreads();
                for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
                    if (threadIdx.x < s) {
                        int o = smem[threadIdx.x + s];
                        if (o < smem[threadIdx.x]) smem[threadIdx.x] = o;
                    }
                    __syncthreads();
                }
                if (threadIdx.x == 0) counters[4] = smem[0];
            }
            grid.sync();

            int min_val = counters[4];
            size_t new_k = k + delta;
            size_t jump_k = static_cast<size_t>(min_val) + delta;
            if (jump_k > new_k) new_k = jump_k;
            if (is_inout && (new_k & 1)) new_k++;
            k = new_k;
        }
    }
}

struct Cache : Cacheable {
    int8_t* removed = nullptr;
    int32_t* push_count = nullptr;
    int32_t* remaining_a = nullptr;
    int32_t* remaining_b = nullptr;
    int32_t* frontier_buf = nullptr;
    int32_t* affected_buf = nullptr;
    int32_t* counters = nullptr;
    int32_t* block_min_buf = nullptr;

    int32_t removed_cap = 0;
    int32_t push_count_cap = 0;
    int32_t remaining_a_cap = 0;
    int32_t remaining_b_cap = 0;
    int32_t frontier_cap = 0;
    int32_t affected_cap = 0;
    int32_t counters_cap = 0;
    int32_t block_min_cap = 0;

    int32_t grid_size = 0;

    void init_grid_size() {
        if (grid_size > 0) return;
        int block_size = 256;
        int smem_size = block_size * sizeof(int);
        int num_blocks_per_sm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &num_blocks_per_sm, core_number_coop_kernel, block_size, smem_size);
        int num_sms;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        
        
        int safe_bps = (num_blocks_per_sm > 1) ? (num_blocks_per_sm - 1) : num_blocks_per_sm;
        grid_size = safe_bps * num_sms;
        if (grid_size <= 0) grid_size = num_sms;
    }

    void ensure(int32_t n) {
        init_grid_size();
        if (removed_cap < n) {
            if (removed) cudaFree(removed);
            cudaMalloc((void**)&removed, (size_t)n);
            removed_cap = n;
        }
        if (push_count_cap < n) {
            if (push_count) cudaFree(push_count);
            cudaMalloc((void**)&push_count, (size_t)n * sizeof(int32_t));
            push_count_cap = n;
        }
        if (remaining_a_cap < n) {
            if (remaining_a) cudaFree(remaining_a);
            cudaMalloc((void**)&remaining_a, (size_t)n * sizeof(int32_t));
            remaining_a_cap = n;
        }
        if (remaining_b_cap < n) {
            if (remaining_b) cudaFree(remaining_b);
            cudaMalloc((void**)&remaining_b, (size_t)n * sizeof(int32_t));
            remaining_b_cap = n;
        }
        if (frontier_cap < n) {
            if (frontier_buf) cudaFree(frontier_buf);
            cudaMalloc((void**)&frontier_buf, (size_t)n * sizeof(int32_t));
            frontier_cap = n;
        }
        if (affected_cap < n) {
            if (affected_buf) cudaFree(affected_buf);
            cudaMalloc((void**)&affected_buf, (size_t)n * sizeof(int32_t));
            affected_cap = n;
        }
        if (counters_cap < 8) {
            if (counters) cudaFree(counters);
            cudaMalloc((void**)&counters, 8 * sizeof(int32_t));
            counters_cap = 8;
        }
        if (block_min_cap < grid_size) {
            if (block_min_buf) cudaFree(block_min_buf);
            cudaMalloc((void**)&block_min_buf, (size_t)grid_size * sizeof(int32_t));
            block_min_cap = grid_size;
        }
    }

    ~Cache() override {
        if (removed) cudaFree(removed);
        if (push_count) cudaFree(push_count);
        if (remaining_a) cudaFree(remaining_a);
        if (remaining_b) cudaFree(remaining_b);
        if (frontier_buf) cudaFree(frontier_buf);
        if (affected_buf) cudaFree(affected_buf);
        if (counters) cudaFree(counters);
        if (block_min_buf) cudaFree(block_min_buf);
    }
};

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
    int32_t num_vertices = graph.number_of_vertices;
    const uint32_t* d_edge_mask = graph.edge_mask;
    cudaStream_t stream = 0;

    if (num_vertices == 0) return;

    cache.ensure(num_vertices);

    cudaMemsetAsync(cache.removed, 0, num_vertices, stream);
    cudaMemsetAsync(cache.push_count, 0, (size_t)num_vertices * sizeof(int32_t), stream);

    
    int degree_mult = (degree_type == 2) ? 2 : 1;
    {
        int block = 256;
        int grid_dim = (num_vertices + block - 1) / block;
        if (grid_dim > 0)
            compute_degrees_kernel<<<grid_dim, block, 0, stream>>>(
                d_offsets, d_indices, d_edge_mask, core_numbers, num_vertices, degree_mult);
    }

    
    int delta = (degree_type == 2) ? 2 : 1;
    int k_first_i = static_cast<int>(k_first);
    int64_t k_last_i64 = static_cast<int64_t>(k_last);
    int is_inout = (degree_type == 2) ? 1 : 0;
    int nv = static_cast<int>(num_vertices);

    int block_size = 256;
    int smem_size = block_size * sizeof(int);

    void* args[] = {
        (void*)&d_offsets, (void*)&d_indices, (void*)&d_edge_mask,
        (void*)&core_numbers, (void*)&cache.removed, (void*)&cache.push_count,
        (void*)&cache.remaining_a, (void*)&cache.remaining_b,
        (void*)&cache.frontier_buf, (void*)&cache.affected_buf,
        (void*)&cache.counters, (void*)&cache.block_min_buf,
        (void*)&nv, (void*)&delta,
        (void*)&k_first_i, (void*)&k_last_i64,
        (void*)&is_inout
    };

    cudaLaunchCooperativeKernel(
        (void*)core_number_coop_kernel,
        dim3(cache.grid_size), dim3(block_size),
        args, smem_size, stream);

    cudaStreamSynchronize(stream);
}

}  
