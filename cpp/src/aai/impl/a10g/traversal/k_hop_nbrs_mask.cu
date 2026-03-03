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

namespace aai {

namespace {

struct Cache : Cacheable {
    uint32_t* bitmaps = nullptr;
    int64_t bitmaps_capacity = 0;

    int64_t* counter = nullptr;
    bool counter_allocated = false;

    int64_t* front0 = nullptr;
    int64_t front0_capacity = 0;

    int64_t* front1 = nullptr;
    int64_t front1_capacity = 0;

    int64_t* counts = nullptr;
    int64_t counts_capacity = 0;

    int64_t* write_pos = nullptr;
    int64_t write_pos_capacity = 0;

    ~Cache() override {
        if (bitmaps) cudaFree(bitmaps);
        if (counter) cudaFree(counter);
        if (front0) cudaFree(front0);
        if (front1) cudaFree(front1);
        if (counts) cudaFree(counts);
        if (write_pos) cudaFree(write_pos);
    }

    void ensure_bitmaps(int64_t total) {
        if (bitmaps_capacity < total) {
            if (bitmaps) cudaFree(bitmaps);
            cudaMalloc(&bitmaps, total * sizeof(uint32_t));
            bitmaps_capacity = total;
        }
    }

    void ensure_counter() {
        if (!counter_allocated) {
            cudaMalloc(&counter, sizeof(int64_t));
            counter_allocated = true;
        }
    }

    void ensure_fronts(int64_t size) {
        if (front0_capacity < size) {
            if (front0) cudaFree(front0);
            cudaMalloc(&front0, size * sizeof(int64_t));
            front0_capacity = size;
        }
        if (front1_capacity < size) {
            if (front1) cudaFree(front1);
            cudaMalloc(&front1, size * sizeof(int64_t));
            front1_capacity = size;
        }
    }

    void ensure_counts(int64_t size) {
        if (counts_capacity < size) {
            if (counts) cudaFree(counts);
            cudaMalloc(&counts, size * sizeof(int64_t));
            counts_capacity = size;
        }
    }

    void ensure_write_pos(int64_t size) {
        if (write_pos_capacity < size) {
            if (write_pos) cudaFree(write_pos);
            cudaMalloc(&write_pos, size * sizeof(int64_t));
            write_pos_capacity = size;
        }
    }
};





__global__ void init_kernel(
    const int32_t* __restrict__ start_vertices,
    int num_start_vertices,
    int64_t* __restrict__ frontier
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_start_vertices) {
        frontier[tid] = ((int64_t)tid << 32) | (uint32_t)start_vertices[tid];
    }
}

__global__ void clear_bitmap_selective(
    const int64_t* __restrict__ frontier,
    int frontier_size,
    uint32_t* __restrict__ bitmaps,
    int64_t bitmap_words
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < frontier_size) {
        int64_t packed = frontier[tid];
        int32_t sv = (int32_t)(packed >> 32);
        int32_t v = (int32_t)(packed & 0xFFFFFFFF);
        int word = v >> 5;
        int bit = v & 31;
        atomicAnd(&bitmaps[(int64_t)sv * bitmap_words + word], ~(1U << bit));
    }
}

__global__ void __launch_bounds__(256, 6)
expand_frontier_warp(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const uint32_t* __restrict__ edge_mask,
    uint32_t* __restrict__ bitmaps,
    int64_t bitmap_words,
    const int64_t* __restrict__ frontier,
    int frontier_size,
    int64_t* __restrict__ new_frontier,
    int64_t* __restrict__ new_frontier_size,
    int64_t max_new_frontier_size
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid >> 5;
    int lane = global_tid & 31;

    if (warp_id >= frontier_size) return;

    int64_t packed = frontier[warp_id];
    int32_t sv_idx = (int32_t)(packed >> 32);
    int32_t v = (int32_t)(packed & 0xFFFFFFFF);

    int row_start = csr_offsets[v];
    int row_end = csr_offsets[v + 1];

    uint32_t* my_bitmap = bitmaps + (int64_t)sv_idx * bitmap_words;

    for (int e_base = row_start; e_base < row_end; e_base += 32) {
        int e = e_base + lane;
        bool in_range = (e < row_end);

        bool edge_active = false;
        if (in_range) {
            edge_active = (edge_mask[e >> 5] >> (e & 31)) & 1;
        }

        int32_t nbr = -1;
        bool is_new = false;

        if (edge_active) {
            nbr = csr_indices[e];
            int word = nbr >> 5;
            int bit = nbr & 31;
            uint32_t cur = my_bitmap[word];
            if (!((cur >> bit) & 1)) {
                uint32_t old = atomicOr(&my_bitmap[word], 1U << bit);
                is_new = !((old >> bit) & 1);
            }
        }

        unsigned ballot = __ballot_sync(0xffffffff, is_new);
        if (ballot) {
            int64_t base_idx;
            if (lane == 0) {
                base_idx = atomicAdd((unsigned long long*)new_frontier_size,
                                     (unsigned long long)__popc(ballot));
            }
            base_idx = __shfl_sync(0xffffffff, base_idx, 0);

            if (is_new) {
                unsigned lower_mask = (1U << lane) - 1;
                int64_t pos = base_idx + __popc(ballot & lower_mask);
                if (pos < max_new_frontier_size) {
                    new_frontier[pos] = ((int64_t)sv_idx << 32) | (uint32_t)nbr;
                }
            }
        }
    }
}

__global__ void count_per_sv_kernel(
    const int64_t* __restrict__ frontier,
    int frontier_size,
    int64_t* __restrict__ counts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & 31;
    bool valid = (tid < frontier_size);
    unsigned active = __ballot_sync(0xffffffff, valid);
    if (!valid) return;

    int sv = (int32_t)(frontier[tid] >> 32);
    unsigned match = __match_any_sync(active, sv);
    int leader = __ffs(match) - 1;
    if (lane == leader) {
        atomicAdd((unsigned long long*)&counts[sv], (unsigned long long)__popc(match));
    }
}

__global__ void build_offsets_kernel(
    const int64_t* __restrict__ counts,
    int64_t* __restrict__ offsets,
    int num_start_vertices
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int64_t sum = 0;
        offsets[0] = 0;
        for (int i = 0; i < num_start_vertices; i++) {
            sum += counts[i];
            offsets[i + 1] = sum;
        }
    }
}

__global__ void scatter_output_kernel(
    const int64_t* __restrict__ frontier,
    int frontier_size,
    int64_t* __restrict__ write_pos,
    int32_t* __restrict__ output
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & 31;
    bool valid = (tid < frontier_size);
    unsigned active = __ballot_sync(0xffffffff, valid);
    if (!valid) return;

    int64_t packed = frontier[tid];
    int sv = (int32_t)(packed >> 32);
    int32_t v = (int32_t)(packed & 0xFFFFFFFF);

    unsigned match = __match_any_sync(active, sv);
    int leader = __ffs(match) - 1;
    int count = __popc(match);

    int64_t base;
    if (lane == leader) {
        base = atomicAdd((unsigned long long*)&write_pos[sv], (unsigned long long)count);
    }
    base = __shfl_sync(match, base, leader);

    unsigned lower = match & ((1U << lane) - 1);
    int pos = __popc(lower);

    output[base + pos] = v;
}

}  

k_hop_nbrs_result_t k_hop_nbrs_mask(const graph32_t& graph,
                                     const int32_t* start_vertices,
                                     std::size_t num_start_vertices,
                                     std::size_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_csr_offsets = graph.offsets;
    const int32_t* d_csr_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    int nsv = static_cast<int>(num_start_vertices);
    int k_int = static_cast<int>(k);

    cudaStream_t stream = 0;

    if (k_int <= 0 || nsv <= 0) {
        int64_t* out_offsets;
        cudaMalloc(&out_offsets, (nsv + 1) * sizeof(int64_t));
        cudaMemsetAsync(out_offsets, 0, (nsv + 1) * sizeof(int64_t), stream);
        return {reinterpret_cast<std::size_t*>(out_offsets), nullptr,
                static_cast<std::size_t>(nsv + 1), 0};
    }

    
    int64_t bitmap_words = ((int64_t)num_vertices + 31) / 32;
    int64_t bitmap_total = (int64_t)nsv * bitmap_words;
    int64_t bitmap_bytes = bitmap_total * sizeof(uint32_t);

    cache.ensure_bitmaps(bitmap_total);
    cudaMemsetAsync(cache.bitmaps, 0, bitmap_bytes, stream);

    cache.ensure_counter();

    
    int64_t avg_degree = num_vertices > 0 ? (int64_t)num_edges / num_vertices + 1 : 1;
    int64_t est_per_sv = 1;
    for (int i = 0; i < k_int; i++) {
        est_per_sv *= avg_degree;
        if (est_per_sv > num_vertices) { est_per_sv = num_vertices; break; }
    }
    int64_t max_frontier = est_per_sv * nsv;
    if (max_frontier < 4096) max_frontier = 4096;
    int64_t cap = 128LL * 1024 * 1024;
    if (max_frontier > cap) max_frontier = cap;

    
    cache.ensure_fronts(max_frontier);
    int64_t* d_front[2] = {cache.front0, cache.front1};

    {
        int block = 256;
        int grid = (nsv + block - 1) / block;
        if (grid > 0)
            init_kernel<<<grid, block, 0, stream>>>(start_vertices, nsv, d_front[0]);
    }

    int curr = 0;
    int64_t frontier_size = nsv;

    for (int hop = 0; hop < k_int; hop++) {
        if (frontier_size == 0) break;

        int next = 1 - curr;

        
        if (hop > 0) {
            int64_t selective_cost = (int64_t)frontier_size * 8;
            if (selective_cost < bitmap_bytes) {
                int block = 256;
                int grid = (int)((frontier_size + block - 1) / block);
                if (frontier_size > 0)
                    clear_bitmap_selective<<<grid, block, 0, stream>>>(
                        d_front[curr], (int)frontier_size, cache.bitmaps, bitmap_words);
            } else {
                cudaMemsetAsync(cache.bitmaps, 0, bitmap_bytes, stream);
            }
        }

        cudaMemsetAsync(cache.counter, 0, sizeof(int64_t), stream);

        {
            int64_t threads_needed = (int64_t)frontier_size * 32;
            int block = 256;
            int grid = (int)((threads_needed + block - 1) / block);
            if (frontier_size > 0)
                expand_frontier_warp<<<grid, block, 0, stream>>>(
                    d_csr_offsets, d_csr_indices, d_edge_mask,
                    cache.bitmaps, bitmap_words,
                    d_front[curr], (int)frontier_size,
                    d_front[next], cache.counter, max_frontier);
        }

        cudaMemcpyAsync(&frontier_size, cache.counter, sizeof(int64_t),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (frontier_size > max_frontier) {
            max_frontier = frontier_size + (frontier_size >> 1);
            cache.ensure_fronts(max_frontier);
            d_front[0] = cache.front0;
            d_front[1] = cache.front1;

            cudaMemsetAsync(cache.bitmaps, 0, bitmap_bytes, stream);
            {
                int block = 256;
                int grid = (nsv + block - 1) / block;
                if (grid > 0)
                    init_kernel<<<grid, block, 0, stream>>>(start_vertices, nsv, d_front[0]);
            }
            curr = 0;
            frontier_size = nsv;
            hop = -1;
            continue;
        }

        curr = next;
    }

    
    int64_t* out_offsets;
    cudaMalloc(&out_offsets, (nsv + 1) * sizeof(int64_t));

    if (frontier_size == 0) {
        cudaMemsetAsync(out_offsets, 0, (nsv + 1) * sizeof(int64_t), stream);
        return {reinterpret_cast<std::size_t*>(out_offsets), nullptr,
                static_cast<std::size_t>(nsv + 1), 0};
    }

    cache.ensure_counts(nsv);
    cudaMemsetAsync(cache.counts, 0, nsv * sizeof(int64_t), stream);

    {
        int block = 256;
        int grid = (int)((frontier_size + block - 1) / block);
        if (frontier_size > 0)
            count_per_sv_kernel<<<grid, block, 0, stream>>>(
                d_front[curr], (int)frontier_size, cache.counts);
    }

    build_offsets_kernel<<<1, 1, 0, stream>>>(cache.counts, out_offsets, nsv);

    int32_t* out_neighbors;
    cudaMalloc(&out_neighbors, frontier_size * sizeof(int32_t));

    cache.ensure_write_pos(nsv);
    cudaMemcpyAsync(cache.write_pos, out_offsets, nsv * sizeof(int64_t),
                     cudaMemcpyDeviceToDevice, stream);

    {
        int block = 256;
        int grid = (int)((frontier_size + block - 1) / block);
        if (frontier_size > 0)
            scatter_output_kernel<<<grid, block, 0, stream>>>(
                d_front[curr], (int)frontier_size, cache.write_pos, out_neighbors);
    }

    return {reinterpret_cast<std::size_t*>(out_offsets), out_neighbors,
            static_cast<std::size_t>(nsv + 1), static_cast<std::size_t>(frontier_size)};
}

}  
