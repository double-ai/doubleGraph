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
#include <vector>
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    
    int* d_ego_size = nullptr;
    int64_t* d_reduce_result = nullptr;

    
    void* d_cub_temp = nullptr;
    size_t cub_temp_capacity = 0;

    int32_t* d_ego_buffer = nullptr;
    int32_t ego_buffer_capacity = 0;

    uint64_t* d_pack_keys = nullptr;
    int64_t pack_keys_capacity = 0;

    uint64_t* d_pack_out = nullptr;
    int64_t pack_out_capacity = 0;

    Cache() {
        cudaMalloc(&d_ego_size, sizeof(int));
        cudaMalloc(&d_reduce_result, sizeof(int64_t));
        cub_temp_capacity = 4 * 1024 * 1024;
        cudaMalloc(&d_cub_temp, cub_temp_capacity);
    }

    ~Cache() override {
        if (d_ego_size) cudaFree(d_ego_size);
        if (d_reduce_result) cudaFree(d_reduce_result);
        if (d_cub_temp) cudaFree(d_cub_temp);
        if (d_ego_buffer) cudaFree(d_ego_buffer);
        if (d_pack_keys) cudaFree(d_pack_keys);
        if (d_pack_out) cudaFree(d_pack_out);
    }

    void ensure_ego_buffer(int32_t n) {
        if (ego_buffer_capacity < n) {
            if (d_ego_buffer) cudaFree(d_ego_buffer);
            ego_buffer_capacity = n;
            cudaMalloc(&d_ego_buffer, (size_t)n * sizeof(int32_t));
        }
    }

    void ensure_cub_temp(size_t needed) {
        if (needed <= cub_temp_capacity) return;
        if (d_cub_temp) cudaFree(d_cub_temp);
        cub_temp_capacity = needed * 2;
        cudaMalloc(&d_cub_temp, cub_temp_capacity);
    }

    void ensure_pack_keys(int64_t max_edges) {
        if (pack_keys_capacity < max_edges) {
            if (d_pack_keys) cudaFree(d_pack_keys);
            pack_keys_capacity = max_edges;
            cudaMalloc(&d_pack_keys, (size_t)max_edges * sizeof(uint64_t));
        }
    }

    void ensure_pack_out(int64_t max_edges) {
        if (pack_out_capacity < max_edges) {
            if (d_pack_out) cudaFree(d_pack_out);
            pack_out_capacity = max_edges;
            cudaMalloc(&d_pack_out, (size_t)max_edges * sizeof(uint64_t));
        }
    }
};




__global__ void init_bfs_kernel(uint32_t* bitmap, int32_t* ego_buffer, int* ego_size, int32_t source) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        bitmap[source >> 5] |= (1u << (source & 31));
        ego_buffer[0] = source;
        *ego_size = 1;
    }
}

__global__ void bfs_expand_kernel(
    const int32_t* __restrict__ csr_offsets, const int32_t* __restrict__ csr_indices,
    uint32_t* __restrict__ bitmap, int32_t* __restrict__ ego_buffer,
    int* __restrict__ ego_size, int frontier_start, int frontier_end
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id = threadIdx.x & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;
    for (int fi = frontier_start + warp_id; fi < frontier_end; fi += total_warps) {
        int32_t v = ego_buffer[fi];
        int32_t rs = csr_offsets[v], re = csr_offsets[v + 1];
        for (int32_t i = rs + lane_id; i < re; i += 32) {
            int32_t n = csr_indices[i];
            uint32_t bit = 1u << (n & 31);
            uint32_t old = atomicOr(&bitmap[n >> 5], bit);
            if (!(old & bit)) { ego_buffer[atomicAdd(ego_size, 1)] = n; }
        }
    }
}




__global__ void count_edges_kernel(
    const int32_t* __restrict__ csr_offsets, const int32_t* __restrict__ csr_indices,
    const uint32_t* __restrict__ bitmap, const int32_t* __restrict__ ego_vertices,
    int ego_size, int64_t* __restrict__ edge_counts
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id = threadIdx.x & 31;
    if (warp_id >= ego_size) return;
    int32_t v = ego_vertices[warp_id];
    int32_t s = csr_offsets[v], e = csr_offsets[v + 1];
    int count = 0;
    for (int32_t i = s + lane_id; i < e; i += 32) {
        int32_t n = csr_indices[i];
        if (bitmap[n >> 5] & (1u << (n & 31))) count++;
    }
    for (int off = 16; off > 0; off >>= 1) count += __shfl_down_sync(0xffffffff, count, off);
    if (lane_id == 0) edge_counts[warp_id] = count;
}

__global__ void write_edges_kernel(
    const int32_t* __restrict__ csr_offsets, const int32_t* __restrict__ csr_indices,
    const uint32_t* __restrict__ bitmap, const int32_t* __restrict__ ego_vertices,
    int ego_size, const int64_t* __restrict__ write_offsets,
    int32_t* __restrict__ edge_srcs, int32_t* __restrict__ edge_dsts, int64_t global_offset
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane_id = threadIdx.x & 31;
    if (warp_id >= ego_size) return;
    int32_t v = ego_vertices[warp_id];
    int32_t s = csr_offsets[v], e = csr_offsets[v + 1];
    int32_t degree = e - s;
    int64_t base = global_offset + write_offsets[warp_id];
    for (int32_t chunk = 0; chunk < degree; chunk += 32) {
        int32_t pos = chunk + lane_id;
        bool active = pos < degree;
        int32_t n = active ? csr_indices[s + pos] : 0;
        bool match = active && ((bitmap[n >> 5] & (1u << (n & 31))) != 0);
        uint32_t ballot = __ballot_sync(0xffffffff, match);
        int prefix = __popc(ballot & ((1u << lane_id) - 1));
        if (match) { edge_srcs[base + prefix] = v; edge_dsts[base + prefix] = n; }
        base += __popc(ballot);
    }
}




__global__ void pack_edges_kernel(const int32_t* __restrict__ srcs, const int32_t* __restrict__ dsts,
                                   uint64_t* __restrict__ keys, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) keys[i] = ((uint64_t)(uint32_t)srcs[i] << 32) | (uint64_t)(uint32_t)dsts[i];
}

__global__ void unpack_edges_kernel(const uint64_t* __restrict__ keys,
                                     int32_t* __restrict__ srcs, int32_t* __restrict__ dsts, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { srcs[i] = (int32_t)(keys[i] >> 32); dsts[i] = (int32_t)(keys[i] & 0xFFFFFFFF); }
}




__global__ void reduce_int64_kernel(const int64_t* data, int n, int64_t* result) {
    typedef cub::BlockReduce<int64_t, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    int64_t val = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) val += data[i];
    int64_t s = BR(tmp).Sum(val);
    if (threadIdx.x == 0) atomicAdd((unsigned long long*)result, (unsigned long long)s);
}




int run_bfs(const int32_t* d_offsets, const int32_t* d_indices,
            uint32_t* bitmap, int32_t* ego_buffer, int* d_ego_size,
            int32_t source, int32_t radius, int bitmap_words, cudaStream_t stream) {
    cudaMemsetAsync(bitmap, 0, (size_t)bitmap_words * sizeof(uint32_t), stream);
    init_bfs_kernel<<<1, 1, 0, stream>>>(bitmap, ego_buffer, d_ego_size, source);
    int frontier_start = 0, cur = 1;
    for (int hop = 0; hop < radius; hop++) {
        int fe = cur;
        int n = fe - frontier_start;
        if (n > 0) {
            int bl = (int)(((int64_t)n * 32 + 255) / 256);
            if (bl > 65535) bl = 65535;
            bfs_expand_kernel<<<bl, 256, 0, stream>>>(d_offsets, d_indices, bitmap, ego_buffer,
                                                       d_ego_size, frontier_start, fe);
        }
        cudaMemcpyAsync(&cur, d_ego_size, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        frontier_start = fe;
    }
    return cur;
}

}  

extract_ego_result_t extract_ego(const graph32_t& graph,
                                 const int32_t* source_vertices,
                                 std::size_t n_sources,
                                 int32_t radius) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    cudaStream_t stream = 0;

    cache.ensure_ego_buffer(num_vertices);

    
    int end_bit = 1;
    { int v = num_vertices - 1; while (v > 0) { v >>= 1; end_bit++; } }
    if (end_bit > 32) end_bit = 32;
    int end_bit_64 = end_bit + 32;

    
    std::vector<int32_t> h_sources(n_sources);
    cudaMemcpy(h_sources.data(), source_vertices, n_sources * sizeof(int32_t), cudaMemcpyDeviceToHost);

    int bitmap_words = (num_vertices + 31) / 32;

    
    std::vector<uint32_t*> d_bitmaps(n_sources, nullptr);
    std::vector<int32_t*> d_sorted_egos(n_sources, nullptr);
    std::vector<int> h_ego_sizes(n_sources);
    std::vector<int64_t> h_edge_counts(n_sources, 0);

    
    
    
    for (std::size_t s = 0; s < n_sources; s++) {
        
        cudaMalloc(&d_bitmaps[s], (size_t)bitmap_words * sizeof(uint32_t));

        int ego_size = run_bfs(d_offsets, d_indices, d_bitmaps[s], cache.d_ego_buffer,
                                cache.d_ego_size, h_sources[s], radius, bitmap_words, stream);
        h_ego_sizes[s] = ego_size;

        
        cudaMalloc(&d_sorted_egos[s], (size_t)ego_size * sizeof(int32_t));

        if (ego_size <= 1) {
            if (ego_size == 1)
                cudaMemcpyAsync(d_sorted_egos[s], cache.d_ego_buffer, sizeof(int32_t),
                                cudaMemcpyDeviceToDevice, stream);
        } else {
            size_t sort_temp = 0;
            cub::DeviceRadixSort::SortKeys(nullptr, sort_temp, (int32_t*)0, (int32_t*)0,
                                           ego_size, 0, end_bit);
            cache.ensure_cub_temp(sort_temp);
            cub::DeviceRadixSort::SortKeys(cache.d_cub_temp, cache.cub_temp_capacity,
                                           cache.d_ego_buffer, d_sorted_egos[s],
                                           ego_size, 0, end_bit, stream);
        }

        
        int64_t* d_counts;
        cudaMalloc(&d_counts, (size_t)ego_size * sizeof(int64_t));

        if (ego_size > 0) {
            count_edges_kernel<<<(ego_size + 7) / 8, 256, 0, stream>>>(
                d_offsets, d_indices, d_bitmaps[s], d_sorted_egos[s], ego_size, d_counts);
        }

        
        cudaMemsetAsync(cache.d_reduce_result, 0, sizeof(int64_t), stream);
        if (ego_size > 0) {
            int bl = (ego_size + 255) / 256;
            if (bl > 1024) bl = 1024;
            reduce_int64_kernel<<<bl, 256, 0, stream>>>(d_counts, ego_size, cache.d_reduce_result);
        }
        cudaMemcpyAsync(&h_edge_counts[s], cache.d_reduce_result, sizeof(int64_t),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cudaFree(d_counts);
    }

    
    
    
    std::vector<int64_t> h_offsets(n_sources + 1, 0);
    for (std::size_t s = 0; s < n_sources; s++)
        h_offsets[s + 1] = h_offsets[s] + h_edge_counts[s];
    int64_t total_edges = h_offsets[n_sources];

    
    int32_t* d_edge_srcs = nullptr;
    int32_t* d_edge_dsts = nullptr;
    std::size_t* d_offsets_out = nullptr;

    cudaMalloc(&d_edge_srcs, (size_t)total_edges * sizeof(int32_t));
    cudaMalloc(&d_edge_dsts, (size_t)total_edges * sizeof(int32_t));
    cudaMalloc(&d_offsets_out, (n_sources + 1) * sizeof(std::size_t));

    
    std::vector<std::size_t> h_offsets_sz(n_sources + 1);
    for (std::size_t i = 0; i <= n_sources; i++)
        h_offsets_sz[i] = (std::size_t)h_offsets[i];
    cudaMemcpyAsync(d_offsets_out, h_offsets_sz.data(),
                    (n_sources + 1) * sizeof(std::size_t), cudaMemcpyHostToDevice, stream);

    
    int64_t max_edges = h_edge_counts.empty() ? 0 :
        *std::max_element(h_edge_counts.begin(), h_edge_counts.end());
    if (max_edges > 0) {
        cache.ensure_pack_keys(max_edges);
        cache.ensure_pack_out(max_edges);

        size_t sort64_temp = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, sort64_temp, (uint64_t*)0, (uint64_t*)0,
                                       (int)max_edges, 0, end_bit_64);
        cache.ensure_cub_temp(sort64_temp);
    }

    
    
    
    for (std::size_t s = 0; s < n_sources; s++) {
        if (h_edge_counts[s] == 0) continue;
        int ego_size = h_ego_sizes[s];

        
        int64_t* d_counts;
        cudaMalloc(&d_counts, (size_t)ego_size * sizeof(int64_t));

        count_edges_kernel<<<(ego_size + 7) / 8, 256, 0, stream>>>(
            d_offsets, d_indices, d_bitmaps[s], d_sorted_egos[s], ego_size, d_counts);

        
        size_t scan_temp = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, scan_temp, (int64_t*)0, (int64_t*)0, ego_size);
        cache.ensure_cub_temp(scan_temp);
        cub::DeviceScan::ExclusiveSum(cache.d_cub_temp, cache.cub_temp_capacity,
                                     d_counts, d_counts, ego_size, stream);

        
        write_edges_kernel<<<(ego_size + 7) / 8, 256, 0, stream>>>(
            d_offsets, d_indices, d_bitmaps[s], d_sorted_egos[s],
            ego_size, d_counts, d_edge_srcs, d_edge_dsts, h_offsets[s]);

        cudaFree(d_counts);

        
        int64_t ne = h_edge_counts[s];
        if (ne > 1) {
            int blk = (int)((ne + 255) / 256);
            pack_edges_kernel<<<blk, 256, 0, stream>>>(
                d_edge_srcs + h_offsets[s], d_edge_dsts + h_offsets[s],
                cache.d_pack_keys, ne);
            cub::DeviceRadixSort::SortKeys(cache.d_cub_temp, cache.cub_temp_capacity,
                                           cache.d_pack_keys, cache.d_pack_out,
                                           (int)ne, 0, end_bit_64, stream);
            unpack_edges_kernel<<<blk, 256, 0, stream>>>(
                cache.d_pack_out, d_edge_srcs + h_offsets[s],
                d_edge_dsts + h_offsets[s], ne);
        }
    }

    cudaStreamSynchronize(stream);

    
    for (std::size_t s = 0; s < n_sources; s++) {
        if (d_bitmaps[s]) cudaFree(d_bitmaps[s]);
        if (d_sorted_egos[s]) cudaFree(d_sorted_egos[s]);
    }

    return extract_ego_result_t{
        d_edge_srcs,
        d_edge_dsts,
        d_offsets_out,
        (std::size_t)total_edges,
        n_sources + 1
    };
}

}  
