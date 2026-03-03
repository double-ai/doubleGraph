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
#include <algorithm>

namespace aai {

namespace {

namespace cg = cooperative_groups;

#define BLOCK_SIZE 256
#define WARP_SIZE 32

struct Cache : Cacheable {
    int32_t* removed = nullptr;
    int32_t* frontier_0 = nullptr;
    int32_t* frontier_1 = nullptr;
    int64_t removed_capacity = 0;
    int64_t frontier_0_capacity = 0;
    int64_t frontier_1_capacity = 0;

    void ensure(int64_t num_vertices) {
        if (removed_capacity < num_vertices) {
            if (removed) cudaFree(removed);
            cudaMalloc(&removed, num_vertices * sizeof(int32_t));
            removed_capacity = num_vertices;
        }
        if (frontier_0_capacity < num_vertices) {
            if (frontier_0) cudaFree(frontier_0);
            cudaMalloc(&frontier_0, num_vertices * sizeof(int32_t));
            frontier_0_capacity = num_vertices;
        }
        if (frontier_1_capacity < num_vertices) {
            if (frontier_1) cudaFree(frontier_1);
            cudaMalloc(&frontier_1, num_vertices * sizeof(int32_t));
            frontier_1_capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (removed) cudaFree(removed);
        if (frontier_0) cudaFree(frontier_0);
        if (frontier_1) cudaFree(frontier_1);
    }
};


__device__ uint32_t d_frontier_size_0;
__device__ uint32_t d_frontier_size_1;
__device__ int32_t d_min_val;
__device__ int32_t d_k_val;  


__global__ void persistent_core_number_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ removed,
    int32_t* __restrict__ frontier_0,
    int32_t* __restrict__ frontier_1,
    int32_t num_vertices,
    int32_t delta,
    int32_t k_start,
    int32_t k_last_i32,
    int32_t k_first
) {
    extern __shared__ int32_t smem[];

    cg::grid_group grid = cg::this_grid();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int warp_id_global = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int num_warps = stride / WARP_SIZE;

    
    for (int v = tid; v < num_vertices; v += stride) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        int32_t deg = end - start;
        
        for (int32_t e = start; e < end; e++) {
            if (indices[e] == v) deg--;
        }
        core_numbers[v] = deg * delta;
        removed[v] = 0;
    }

    if (tid == 0) {
        d_frontier_size_0 = 0;
        d_frontier_size_1 = 0;
        d_k_val = k_start;
    }
    grid.sync();

    int cur_buf = 0;  
    int32_t* frontiers[2] = {frontier_0, frontier_1};

    int32_t k = k_start;

    while (k <= k_last_i32) {
        
        
        if (tid == 0) {
            if (cur_buf == 0) d_frontier_size_0 = 0;
            else d_frontier_size_1 = 0;
        }
        grid.sync();

        uint32_t* cur_size_ptr = (cur_buf == 0) ? &d_frontier_size_0 : &d_frontier_size_1;
        int32_t* cur_frontier = frontiers[cur_buf];

        
        for (int v = tid; v < num_vertices; v += stride) {
            if (!removed[v] && core_numbers[v] < k) {
                uint32_t pos = atomicAdd(cur_size_ptr, 1u);
                cur_frontier[pos] = v;
            }
        }
        grid.sync();

        uint32_t fs = *cur_size_ptr;

        if (fs == 0) {
            
            int32_t local_min = INT32_MAX;
            for (int v = tid; v < num_vertices; v += stride) {
                if (!removed[v]) {
                    int32_t cn = core_numbers[v];
                    if (cn < local_min) local_min = cn;
                }
            }

            
            smem[threadIdx.x] = local_min;
            __syncthreads();
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (threadIdx.x < s && smem[threadIdx.x + s] < smem[threadIdx.x])
                    smem[threadIdx.x] = smem[threadIdx.x + s];
                __syncthreads();
            }

            if (tid == 0) d_min_val = INT32_MAX;
            grid.sync();
            if (threadIdx.x == 0) atomicMin(&d_min_val, smem[0]);
            grid.sync();

            int32_t min_val = d_min_val;
            if (min_val == INT32_MAX) break;  

            int32_t new_k = k + delta;
            if (min_val + delta > new_k) new_k = min_val + delta;
            if (delta == 2 && (new_k & 1)) new_k++;
            k = new_k;
            continue;
        }

        
        while (fs > 0) {
            
            for (uint32_t i = tid; i < fs; i += stride) {
                removed[cur_frontier[i]] = 1;
            }
            grid.sync();

            
            int nxt_buf = 1 - cur_buf;
            uint32_t* nxt_size_ptr = (nxt_buf == 0) ? &d_frontier_size_0 : &d_frontier_size_1;
            int32_t* nxt_frontier = frontiers[nxt_buf];

            if (tid == 0) *nxt_size_ptr = 0;
            grid.sync();

            for (uint32_t wi = warp_id_global; wi < fs; wi += num_warps) {
                int32_t v = cur_frontier[wi];
                int32_t start = offsets[v];
                int32_t end = offsets[v + 1];
                int32_t degree = end - start;

                for (int32_t i = lane; i < degree; i += WARP_SIZE) {
                    int32_t u = indices[start + i];
                    if (u != v && !removed[u]) {
                        int32_t old_val = atomicSub(&core_numbers[u], delta);
                        int32_t new_val = old_val - delta;
                        if (old_val >= k && new_val < k) {
                            uint32_t pos = atomicAdd(nxt_size_ptr, 1u);
                            nxt_frontier[pos] = u;
                        }
                    }
                }
            }
            grid.sync();

            fs = *nxt_size_ptr;

            
            if (fs > 0) {
                int32_t kmd = k - delta;
                for (uint32_t i = tid; i < fs; i += stride) {
                    int32_t v = nxt_frontier[i];
                    if (core_numbers[v] < kmd) core_numbers[v] = kmd;
                }
                grid.sync();
            }

            
            cur_buf = nxt_buf;
            cur_frontier = nxt_frontier;
            cur_size_ptr = nxt_size_ptr;
        }

        k += delta;
    }

    
    if (k_first > 0) {
        for (int v = tid; v < num_vertices; v += stride) {
            if (core_numbers[v] > 0 && core_numbers[v] < k_first) {
                core_numbers[v] = 0;
            }
        }
    }
}

}  

void core_number_seg(const graph32_t& graph,
                     int32_t* core_numbers,
                     int degree_type,
                     std::size_t k_first,
                     std::size_t k_last) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    if (num_vertices == 0) return;

    cache.ensure(num_vertices);

    int32_t delta = (degree_type == 2) ? 2 : 1;
    int32_t k_first_i32 = static_cast<int32_t>(k_first);

    int64_t max_k = static_cast<int64_t>(2) * num_vertices + 10;
    int32_t k_last_i32;
    if (static_cast<int64_t>(k_last) < 0 || static_cast<int64_t>(k_last) > max_k) {
        k_last_i32 = static_cast<int32_t>(max_k);
    } else {
        k_last_i32 = static_cast<int32_t>(k_last);
    }

    int32_t k_start = std::max(k_first_i32, 2);
    if (delta == 2 && (k_start & 1)) k_start++;

    int numBlocksPerSm;
    int sharedMemSize = BLOCK_SIZE * sizeof(int32_t);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, persistent_core_number_kernel, BLOCK_SIZE, sharedMemSize);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    int gridSize = numBlocksPerSm * numSMs;

    const int32_t* offsets_ptr = graph.offsets;
    const int32_t* indices_ptr = graph.indices;

    void* args[] = {
        (void*)&offsets_ptr, (void*)&indices_ptr,
        (void*)&core_numbers, (void*)&cache.removed,
        (void*)&cache.frontier_0, (void*)&cache.frontier_1,
        (void*)&num_vertices, (void*)&delta,
        (void*)&k_start, (void*)&k_last_i32, (void*)&k_first_i32
    };

    cudaLaunchCooperativeKernel(
        (void*)persistent_core_number_kernel,
        dim3(gridSize), dim3(BLOCK_SIZE),
        args, sharedMemSize, 0
    );

    cudaDeviceSynchronize();
}

}  
