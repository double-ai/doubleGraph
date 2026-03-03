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
#include <cub/cub.cuh>
#include <cstdint>
#include <cstddef>
#include <algorithm>

namespace aai {

namespace {




struct Cache : Cacheable {
    uint32_t* packed_removed = nullptr;
    int32_t packed_removed_capacity = 0;

    int32_t* degrees = nullptr;
    int32_t degrees_capacity = 0;

    int32_t* queue1 = nullptr;
    int32_t queue1_capacity = 0;

    int32_t* queue2 = nullptr;
    int32_t queue2_capacity = 0;

    int32_t* counters = nullptr;
    bool counters_allocated = false;

    int32_t* counter = nullptr;
    bool counter_allocated = false;

    void ensure(int32_t num_vertices) {
        int32_t num_words = (num_vertices + 31) / 32;
        if (packed_removed_capacity < num_words) {
            if (packed_removed) cudaFree(packed_removed);
            cudaMalloc(&packed_removed, num_words * sizeof(uint32_t));
            packed_removed_capacity = num_words;
        }
        if (degrees_capacity < num_vertices) {
            if (degrees) cudaFree(degrees);
            cudaMalloc(&degrees, num_vertices * sizeof(int32_t));
            degrees_capacity = num_vertices;
        }
        if (queue1_capacity < num_vertices) {
            if (queue1) cudaFree(queue1);
            cudaMalloc(&queue1, num_vertices * sizeof(int32_t));
            queue1_capacity = num_vertices;
        }
        if (queue2_capacity < num_vertices) {
            if (queue2) cudaFree(queue2);
            cudaMalloc(&queue2, num_vertices * sizeof(int32_t));
            queue2_capacity = num_vertices;
        }
        if (!counters_allocated) {
            cudaMalloc(&counters, 2 * sizeof(int32_t));
            counters_allocated = true;
        }
        if (!counter_allocated) {
            cudaMalloc(&counter, sizeof(int32_t));
            counter_allocated = true;
        }
    }

    ~Cache() override {
        if (packed_removed) cudaFree(packed_removed);
        if (degrees) cudaFree(degrees);
        if (queue1) cudaFree(queue1);
        if (queue2) cudaFree(queue2);
        if (counters) cudaFree(counters);
        if (counter) cudaFree(counter);
    }
};




__device__ __forceinline__ int read_removed_bit(const uint32_t* __restrict__ packed, int v) {
    return (packed[v >> 5] >> (v & 31)) & 1;
}




__global__ void compute_degrees_and_init_peel_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ degrees,
    uint32_t* __restrict__ packed_removed,
    int32_t* __restrict__ queue,
    int32_t* __restrict__ queue_size,
    int32_t num_vertices,
    int32_t multiplier,
    int32_t k
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t deg = end - start;
    for (int32_t i = start; i < end; i++) {
        if (indices[i] == v) deg--;
    }
    int32_t eff_deg = deg * multiplier;
    degrees[v] = eff_deg;
    if (eff_deg < k) {
        atomicOr(&packed_removed[v >> 5], 1u << (v & 31));
        int pos = atomicAdd(queue_size, 1);
        queue[pos] = v;
    }
}




__global__ void init_from_core_numbers_kernel(
    const int32_t* __restrict__ core_numbers,
    uint32_t* __restrict__ packed_removed,
    int32_t num_vertices,
    int32_t k
) {
    
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base_v = word_idx * 32;
    if (base_v >= num_vertices) return;

    uint32_t word = 0;
    int end = base_v + 32;
    if (end > num_vertices) end = num_vertices;
    for (int v = base_v; v < end; v++) {
        if (core_numbers[v] < k) {
            word |= (1u << (v - base_v));
        }
    }
    packed_removed[word_idx] = word;
}




__global__ void peel_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ degrees,
    uint32_t* __restrict__ packed_removed,
    const int32_t* __restrict__ current_queue,
    int32_t current_queue_size,
    int32_t* __restrict__ next_queue,
    int32_t* __restrict__ next_queue_size,
    int32_t k,
    int32_t decrement
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= current_queue_size) return;
    int32_t u = current_queue[warp_id];
    int32_t start = offsets[u];
    int32_t end = offsets[u + 1];
    for (int32_t i = start + lane; i < end; i += 32) {
        int32_t v = indices[i];
        if (v == u) continue;
        if (read_removed_bit(packed_removed, v)) continue;
        int32_t old_deg = atomicSub(&degrees[v], decrement);
        if (old_deg >= k && (old_deg - decrement) < k) {
            atomicOr(&packed_removed[v >> 5], 1u << (v & 31));
            int pos = atomicAdd(next_queue_size, 1);
            next_queue[pos] = v;
        }
    }
}




__global__ void extract_edges_block_per_vertex(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ packed_removed,
    int32_t* __restrict__ edge_srcs,
    int32_t* __restrict__ edge_dsts,
    int32_t* __restrict__ global_counter,
    int32_t seg_start,
    int32_t seg_end
) {
    int v = seg_start + blockIdx.x;
    if (v >= seg_end) return;
    if (read_removed_bit(packed_removed, v)) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    
    int32_t local_count = 0;
    for (int32_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        if (!read_removed_bit(packed_removed, indices[i])) local_count++;
    }

    for (int delta = 16; delta > 0; delta >>= 1)
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, delta);

    __shared__ int32_t warp_counts[8];
    if (lane == 0) warp_counts[warp_id] = local_count;
    __syncthreads();

    __shared__ int32_t s_total, s_base;
    if (threadIdx.x == 0) {
        int32_t total = 0;
        for (int w = 0; w < num_warps; w++) total += warp_counts[w];
        s_total = total;
        if (total > 0) s_base = atomicAdd(global_counter, total);
    }
    __syncthreads();
    if (s_total == 0) return;
    int32_t base = s_base;

    
    __shared__ int32_t s_write_pos;
    if (threadIdx.x == 0) s_write_pos = 0;
    __syncthreads();

    for (int32_t chunk = start; chunk < end; chunk += blockDim.x) {
        int32_t i = chunk + threadIdx.x;
        int32_t u = 0;
        int valid = 0;
        if (i < end) {
            u = indices[i];
            valid = !read_removed_bit(packed_removed, u) ? 1 : 0;
        }
        unsigned ballot = __ballot_sync(0xFFFFFFFF, valid);
        int my_pos = __popc(ballot & ((1u << lane) - 1));
        int warp_count = __popc(ballot);
        int32_t warp_base = 0;
        if (lane == 0 && warp_count > 0)
            warp_base = atomicAdd(&s_write_pos, warp_count);
        warp_base = __shfl_sync(0xFFFFFFFF, warp_base, 0);
        if (valid) {
            edge_srcs[base + warp_base + my_pos] = v;
            edge_dsts[base + warp_base + my_pos] = u;
        }
    }
}




__global__ void extract_edges_warp_per_vertex(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ packed_removed,
    int32_t* __restrict__ edge_srcs,
    int32_t* __restrict__ edge_dsts,
    int32_t* __restrict__ global_counter,
    int32_t seg_start,
    int32_t seg_end
) {
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int v = seg_start + warp_global;
    if (v >= seg_end) return;
    if (read_removed_bit(packed_removed, v)) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    
    int32_t count = 0;
    for (int32_t i = start + lane; i < end; i += 32) {
        if (!read_removed_bit(packed_removed, indices[i])) count++;
    }
    for (int delta = 16; delta > 0; delta >>= 1)
        count += __shfl_down_sync(0xFFFFFFFF, count, delta);
    int32_t total = __shfl_sync(0xFFFFFFFF, count, 0);
    if (total == 0) return;

    int32_t base;
    if (lane == 0) base = atomicAdd(global_counter, total);
    base = __shfl_sync(0xFFFFFFFF, base, 0);

    
    int32_t write_offset = 0;
    for (int32_t chunk = start; chunk < end; chunk += 32) {
        int32_t i = chunk + lane;
        int32_t u = 0;
        int valid = 0;
        if (i < end) {
            u = indices[i];
            valid = !read_removed_bit(packed_removed, u) ? 1 : 0;
        }
        unsigned ballot = __ballot_sync(0xFFFFFFFF, valid);
        int my_pos = __popc(ballot & ((1u << lane) - 1));
        if (valid) {
            edge_srcs[base + write_offset + my_pos] = v;
            edge_dsts[base + write_offset + my_pos] = u;
        }
        write_offset += __popc(ballot);
    }
}





__global__ void extract_edges_thread_per_vertex(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ packed_removed,
    int32_t* __restrict__ edge_srcs,
    int32_t* __restrict__ edge_dsts,
    int32_t* __restrict__ global_counter,
    int32_t seg_start,
    int32_t seg_end
) {
    int v = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    
    int32_t valid_nbrs[31]; 
    int32_t count = 0;

    bool active = (v < seg_end);
    if (active) {
        
        int removed_v = read_removed_bit(packed_removed, v);
        if (removed_v) {
            active = false;
        } else {
            int32_t start = offsets[v];
            int32_t end = offsets[v + 1];
            for (int32_t i = start; i < end; i++) {
                int32_t u = indices[i];
                if (!read_removed_bit(packed_removed, u)) {
                    valid_nbrs[count++] = u;
                }
            }
            if (count == 0) active = false;
        }
    }

    
    int32_t my_count = active ? count : 0;
    int32_t prefix = my_count;
    #pragma unroll
    for (int delta = 1; delta < 32; delta <<= 1) {
        int32_t n = __shfl_up_sync(0xFFFFFFFF, prefix, delta);
        if (lane >= delta) prefix += n;
    }
    int32_t my_offset = prefix - my_count;
    int32_t warp_total = __shfl_sync(0xFFFFFFFF, prefix, 31);

    int32_t warp_base = 0;
    if (lane == 0 && warp_total > 0)
        warp_base = atomicAdd(global_counter, warp_total);
    warp_base = __shfl_sync(0xFFFFFFFF, warp_base, 0);

    
    if (active && count > 0) {
        int32_t write_pos = warp_base + my_offset;
        for (int32_t j = 0; j < count; j++) {
            edge_srcs[write_pos] = v;
            edge_dsts[write_pos] = valid_nbrs[j];
            write_pos++;
        }
    }
}





void launch_compute_degrees_and_init_peel(
    const int32_t* offsets, const int32_t* indices,
    int32_t* degrees, uint32_t* packed_removed,
    int32_t* queue, int32_t* queue_size,
    int32_t num_vertices, int32_t multiplier, int32_t k,
    cudaStream_t stream
) {
    if (num_vertices <= 0) return;
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    compute_degrees_and_init_peel_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, degrees, packed_removed, queue, queue_size, num_vertices, multiplier, k);
}

void launch_init_from_core_numbers(
    const int32_t* core_numbers, uint32_t* packed_removed,
    int32_t num_vertices, int32_t k, cudaStream_t stream
) {
    if (num_vertices <= 0) return;
    int num_words = (num_vertices + 31) / 32;
    int block = 256;
    int grid = (num_words + block - 1) / block;
    init_from_core_numbers_kernel<<<grid, block, 0, stream>>>(core_numbers, packed_removed, num_vertices, k);
}

void launch_peel(
    const int32_t* offsets, const int32_t* indices,
    int32_t* degrees, uint32_t* packed_removed,
    const int32_t* current_queue, int32_t current_queue_size,
    int32_t* next_queue, int32_t* next_queue_size,
    int32_t k, int32_t decrement, cudaStream_t stream
) {
    if (current_queue_size <= 0) return;
    int64_t threads_needed = (int64_t)current_queue_size * 32;
    int block = 256;
    int grid = (int)((threads_needed + block - 1) / block);
    peel_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, degrees, packed_removed,
        current_queue, current_queue_size,
        next_queue, next_queue_size, k, decrement);
}

void launch_extract_edges_segmented(
    const int32_t* offsets, const int32_t* indices,
    const uint32_t* packed_removed,
    int32_t* edge_srcs, int32_t* edge_dsts,
    int32_t* global_counter,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3,
    cudaStream_t stream
) {
    int32_t n_high = seg1 - seg0;
    if (n_high > 0)
        extract_edges_block_per_vertex<<<n_high, 256, 0, stream>>>(
            offsets, indices, packed_removed, edge_srcs, edge_dsts, global_counter, seg0, seg1);

    int32_t n_mid = seg2 - seg1;
    if (n_mid > 0) {
        int blocks = (n_mid + 7) / 8;
        extract_edges_warp_per_vertex<<<blocks, 256, 0, stream>>>(
            offsets, indices, packed_removed, edge_srcs, edge_dsts, global_counter, seg1, seg2);
    }

    int32_t n_low = seg3 - seg2;
    if (n_low > 0) {
        int blocks = (n_low + 255) / 256;
        extract_edges_thread_per_vertex<<<blocks, 256, 0, stream>>>(
            offsets, indices, packed_removed, edge_srcs, edge_dsts, global_counter, seg2, seg3);
    }
}

}  

std::size_t k_core_seg(const graph32_t& graph,
                       std::size_t k,
                       int degree_type,
                       const int32_t* core_numbers,
                       int32_t* edge_srcs,
                       int32_t* edge_dsts,
                       std::size_t max_edges) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg3 = seg[3];

    cudaStream_t stream = 0;

    if (num_vertices == 0 || num_edges == 0) {
        return 0;
    }

    cache.ensure(num_vertices);

    int32_t num_words = (num_vertices + 31) / 32;
    cudaMemsetAsync(cache.packed_removed, 0, num_words * sizeof(uint32_t), stream);

    int32_t k_int = static_cast<int32_t>(k);

    if (core_numbers != nullptr) {
        launch_init_from_core_numbers(
            core_numbers, cache.packed_removed,
            num_vertices, k_int, stream);
    } else {
        int32_t multiplier = (degree_type == 2) ? 2 : 1;
        int32_t decrement = multiplier;

        int32_t* d_counter_a = cache.counters;
        int32_t* d_counter_b = cache.counters + 1;

        cudaMemsetAsync(d_counter_a, 0, sizeof(int32_t), stream);

        launch_compute_degrees_and_init_peel(
            d_offsets, d_indices, cache.degrees, cache.packed_removed,
            cache.queue1, d_counter_a,
            num_vertices, multiplier, k_int, stream);

        int32_t h_queue_size = 0;
        cudaMemcpyAsync(&h_queue_size, d_counter_a, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        int32_t* current_queue = cache.queue1;
        int32_t* next_queue = cache.queue2;
        int32_t* next_counter = d_counter_b;

        while (h_queue_size > 0) {
            cudaMemsetAsync(next_counter, 0, sizeof(int32_t), stream);
            launch_peel(d_offsets, d_indices, cache.degrees, cache.packed_removed,
                       current_queue, h_queue_size,
                       next_queue, next_counter,
                       k_int, decrement, stream);

            cudaMemcpyAsync(&h_queue_size, next_counter, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            std::swap(current_queue, next_queue);
            next_counter = (next_counter == d_counter_b) ? d_counter_a : d_counter_b;
        }
    }

    
    cudaMemsetAsync(cache.counter, 0, sizeof(int32_t), stream);

    launch_extract_edges_segmented(
        d_offsets, d_indices, cache.packed_removed,
        edge_srcs, edge_dsts, cache.counter,
        seg0, seg1, seg2, seg3, stream);

    int32_t h_total_edges = 0;
    cudaMemcpyAsync(&h_total_edges, cache.counter, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return static_cast<std::size_t>(h_total_edges);
}

}  
