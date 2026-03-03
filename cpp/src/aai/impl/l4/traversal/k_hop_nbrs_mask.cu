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

    int* counter = nullptr;

    int64_t* front_0 = nullptr;
    int64_t front_0_capacity = 0;

    int64_t* front_1 = nullptr;
    int64_t front_1_capacity = 0;

    int64_t* counts = nullptr;
    int64_t counts_capacity = 0;

    int64_t* write_pos = nullptr;
    int64_t write_pos_capacity = 0;

    Cache() {
        cudaMalloc(&counter, sizeof(int));
    }

    ~Cache() override {
        if (bitmaps) cudaFree(bitmaps);
        if (counter) cudaFree(counter);
        if (front_0) cudaFree(front_0);
        if (front_1) cudaFree(front_1);
        if (counts) cudaFree(counts);
        if (write_pos) cudaFree(write_pos);
    }

    void ensure_bitmaps(int64_t size) {
        if (bitmaps_capacity < size) {
            if (bitmaps) cudaFree(bitmaps);
            cudaMalloc(&bitmaps, size * sizeof(uint32_t));
            bitmaps_capacity = size;
        }
    }

    void ensure_frontiers(int64_t size) {
        if (front_0_capacity < size) {
            if (front_0) cudaFree(front_0);
            cudaMalloc(&front_0, size * sizeof(int64_t));
            front_0_capacity = size;
        }
        if (front_1_capacity < size) {
            if (front_1) cudaFree(front_1);
            cudaMalloc(&front_1, size * sizeof(int64_t));
            front_1_capacity = size;
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
    int* __restrict__ new_frontier_size,
    int max_new_frontier_size
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
            int base_idx;
            if (lane == 0) {
                base_idx = atomicAdd(new_frontier_size, __popc(ballot));
            }
            base_idx = __shfl_sync(0xffffffff, base_idx, 0);

            if (is_new) {
                unsigned lower_mask = (1U << lane) - 1;
                int pos = base_idx + __popc(ballot & lower_mask);
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

    if (tid < frontier_size) {
        int sv = (int32_t)(frontier[tid] >> 32);
        unsigned match = __match_any_sync(0xffffffff, sv);
        int leader = __ffs(match) - 1;
        if (lane == leader) {
            atomicAdd((unsigned long long*)&counts[sv], (unsigned long long)__popc(match));
        }
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

    if (tid < frontier_size) {
        int64_t packed = frontier[tid];
        int sv = (int32_t)(packed >> 32);
        int32_t v = (int32_t)(packed & 0xFFFFFFFF);

        unsigned match = __match_any_sync(0xffffffff, sv);
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



void launch_init_kernel(
    const int32_t* start_vertices,
    int num_start_vertices,
    int64_t* frontier,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (num_start_vertices + block - 1) / block;
    if (grid > 0)
        init_kernel<<<grid, block, 0, stream>>>(start_vertices, num_start_vertices, frontier);
}

void launch_clear_bitmap_selective(
    const int64_t* frontier,
    int frontier_size,
    uint32_t* bitmaps,
    int64_t bitmap_words,
    cudaStream_t stream
) {
    if (frontier_size == 0) return;
    int block = 256;
    int grid = (frontier_size + block - 1) / block;
    clear_bitmap_selective<<<grid, block, 0, stream>>>(
        frontier, frontier_size, bitmaps, bitmap_words);
}

void launch_expand_frontier(
    const int32_t* csr_offsets,
    const int32_t* csr_indices,
    const uint32_t* edge_mask,
    uint32_t* bitmaps,
    int64_t bitmap_words,
    const int64_t* frontier,
    int frontier_size,
    int64_t* new_frontier,
    int* new_frontier_size,
    int max_new_frontier_size,
    cudaStream_t stream
) {
    if (frontier_size == 0) return;
    int block = 256;
    int warps_per_block = block / 32;
    int grid = (frontier_size + warps_per_block - 1) / warps_per_block;
    expand_frontier_warp<<<grid, block, 0, stream>>>(
        csr_offsets, csr_indices, edge_mask,
        bitmaps, bitmap_words,
        frontier, frontier_size,
        new_frontier, new_frontier_size, max_new_frontier_size
    );
}

void launch_count_per_sv(
    const int64_t* frontier,
    int frontier_size,
    int64_t* counts,
    cudaStream_t stream
) {
    if (frontier_size == 0) return;
    int block = 256;
    int grid = (frontier_size + block - 1) / block;
    count_per_sv_kernel<<<grid, block, 0, stream>>>(frontier, frontier_size, counts);
}

void launch_build_offsets(
    const int64_t* counts,
    int64_t* offsets,
    int num_start_vertices,
    cudaStream_t stream
) {
    build_offsets_kernel<<<1, 1, 0, stream>>>(counts, offsets, num_start_vertices);
}

void launch_scatter_output(
    const int64_t* frontier,
    int frontier_size,
    int64_t* write_pos,
    int32_t* output,
    cudaStream_t stream
) {
    if (frontier_size == 0) return;
    int block = 256;
    int grid = (frontier_size + block - 1) / block;
    scatter_output_kernel<<<grid, block, 0, stream>>>(
        frontier, frontier_size, write_pos, output);
}

}  

k_hop_nbrs_result_t k_hop_nbrs_mask(const graph32_t& graph,
                                    const int32_t* start_vertices,
                                    std::size_t num_start_vertices,
                                    std::size_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_csr_offsets = graph.offsets;
    const int32_t* d_csr_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;

    int nsv = static_cast<int>(num_start_vertices);
    int k_val = static_cast<int>(k);

    cudaStream_t stream = 0;

    
    int64_t bitmap_words = ((int64_t)num_vertices + 31) / 32;
    int64_t bitmap_total = (int64_t)nsv * bitmap_words;
    int64_t bitmap_bytes = bitmap_total * sizeof(uint32_t);

    cache.ensure_bitmaps(bitmap_total);
    uint32_t* d_bitmaps = cache.bitmaps;
    cudaMemsetAsync(d_bitmaps, 0, bitmap_bytes, stream);

    int* d_counter = cache.counter;

    
    int64_t avg_degree = num_vertices > 0 ? (int64_t)num_edges / num_vertices + 1 : 1;
    int64_t est_per_sv = 1;
    for (int i = 0; i < k_val; i++) {
        est_per_sv *= avg_degree;
        if (est_per_sv > num_vertices) { est_per_sv = num_vertices; break; }
    }
    int64_t max_frontier = est_per_sv * nsv;
    if (max_frontier < 4096) max_frontier = 4096;
    int64_t cap = 128LL * 1024 * 1024;
    if (max_frontier > cap) max_frontier = cap;

    
    cache.ensure_frontiers(max_frontier);
    int64_t* d_front[2] = {cache.front_0, cache.front_1};

    launch_init_kernel(start_vertices, nsv, d_front[0], stream);

    int curr = 0;
    int frontier_size = nsv;

    for (int hop = 0; hop < k_val; hop++) {
        if (frontier_size == 0) break;

        int next = 1 - curr;

        
        if (hop > 0) {
            int64_t selective_cost = (int64_t)frontier_size * 8;
            if (selective_cost < bitmap_bytes) {
                launch_clear_bitmap_selective(d_front[curr], frontier_size,
                                              d_bitmaps, bitmap_words, stream);
            } else {
                cudaMemsetAsync(d_bitmaps, 0, bitmap_bytes, stream);
            }
        }

        cudaMemsetAsync(d_counter, 0, sizeof(int), stream);

        launch_expand_frontier(
            d_csr_offsets, d_csr_indices, d_edge_mask,
            d_bitmaps, bitmap_words,
            d_front[curr], frontier_size,
            d_front[next], d_counter, (int)max_frontier,
            stream
        );

        cudaMemcpyAsync(&frontier_size, d_counter, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (frontier_size > (int)max_frontier) {
            max_frontier = (int64_t)frontier_size + ((int64_t)frontier_size >> 1);
            cache.ensure_frontiers(max_frontier);
            d_front[0] = cache.front_0;
            d_front[1] = cache.front_1;

            cudaMemsetAsync(d_bitmaps, 0, bitmap_bytes, stream);
            launch_init_kernel(start_vertices, nsv, d_front[0], stream);
            curr = 0;
            frontier_size = nsv;
            hop = -1;
            continue;
        }

        curr = next;
    }

    
    int64_t* d_out_offsets;
    cudaMalloc(&d_out_offsets, (nsv + 1) * sizeof(int64_t));

    if (frontier_size == 0) {
        cudaMemsetAsync(d_out_offsets, 0, (nsv + 1) * sizeof(int64_t), stream);
        return k_hop_nbrs_result_t{
            reinterpret_cast<std::size_t*>(d_out_offsets),
            nullptr,
            static_cast<std::size_t>(nsv + 1),
            0
        };
    }

    cache.ensure_counts(nsv);
    cache.ensure_write_pos(nsv);

    cudaMemsetAsync(cache.counts, 0, nsv * sizeof(int64_t), stream);
    launch_count_per_sv(d_front[curr], frontier_size, cache.counts, stream);

    launch_build_offsets(cache.counts, d_out_offsets, nsv, stream);

    int32_t* d_out_neighbors;
    cudaMalloc(&d_out_neighbors, (int64_t)frontier_size * sizeof(int32_t));

    cudaMemcpyAsync(cache.write_pos, d_out_offsets,
                    nsv * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream);

    launch_scatter_output(d_front[curr], frontier_size,
                         cache.write_pos, d_out_neighbors, stream);

    return k_hop_nbrs_result_t{
        reinterpret_cast<std::size_t*>(d_out_offsets),
        d_out_neighbors,
        static_cast<std::size_t>(nsv + 1),
        static_cast<std::size_t>(frontier_size)
    };
}

}  
