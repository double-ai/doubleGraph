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
#include <cstddef>
#include <vector>
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/execution_policy.h>

namespace aai {

namespace {

struct Cache : Cacheable {
    uint32_t* d_bitmap = nullptr;
    int32_t* d_neighborhood = nullptr;
    int32_t* d_ns_counter = nullptr;
    int64_t* d_edge_counts = nullptr;
    int64_t* d_write_offsets = nullptr;
    void* d_scan_temp = nullptr;
    size_t scan_temp_bytes = 0;
    int32_t allocated_verts = 0;
    size_t bitmap_bytes = 0;

    void free_memory() {
        if (d_bitmap) { cudaFree(d_bitmap); d_bitmap = nullptr; }
        if (d_neighborhood) { cudaFree(d_neighborhood); d_neighborhood = nullptr; }
        if (d_ns_counter) { cudaFree(d_ns_counter); d_ns_counter = nullptr; }
        if (d_edge_counts) { cudaFree(d_edge_counts); d_edge_counts = nullptr; }
        if (d_write_offsets) { cudaFree(d_write_offsets); d_write_offsets = nullptr; }
        if (d_scan_temp) { cudaFree(d_scan_temp); d_scan_temp = nullptr; }
        allocated_verts = 0;
    }

    void ensure(int32_t num_vertices) {
        if (num_vertices <= allocated_verts) return;
        free_memory();

        size_t bitmap_words = ((size_t)num_vertices + 31) / 32;
        bitmap_bytes = bitmap_words * sizeof(uint32_t);
        cudaMalloc(&d_bitmap, bitmap_bytes);
        cudaMalloc(&d_neighborhood, (size_t)num_vertices * sizeof(int32_t));
        cudaMalloc(&d_ns_counter, sizeof(int32_t));
        cudaMalloc(&d_edge_counts, (size_t)num_vertices * sizeof(int64_t));
        cudaMalloc(&d_write_offsets, (size_t)num_vertices * sizeof(int64_t));
        scan_temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes,
                                      (int64_t*)nullptr, (int64_t*)nullptr, num_vertices);
        cudaMalloc(&d_scan_temp, scan_temp_bytes);
        allocated_verts = num_vertices;
    }

    ~Cache() override { free_memory(); }
};



__global__ void init_source_kernel(
    uint32_t* bitmap, int32_t* neighborhood, int32_t* neighborhood_size,
    int32_t source
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicOr(&bitmap[source >> 5], 1u << (source & 31));
        neighborhood[0] = source;
        *neighborhood_size = 1;
    }
}

__global__ void bfs_expand_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* neighborhood,
    int32_t frontier_start, int32_t frontier_end,
    int32_t* neighborhood_size,
    uint32_t* bitmap
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int frontier_size = frontier_end - frontier_start;
    if (warp_id >= frontier_size) return;

    int32_t v = neighborhood[frontier_start + warp_id];
    int32_t start = __ldg(&offsets[v]);
    int32_t end = __ldg(&offsets[v + 1]);

    for (int32_t e = start + lane; e < end; e += 32) {
        int32_t u = __ldg(&indices[e]);
        uint32_t bit = 1u << (u & 31);
        uint32_t old = atomicOr(&bitmap[u >> 5], bit);
        if (!(old & bit)) {
            int32_t idx = atomicAdd(neighborhood_size, 1);
            neighborhood[idx] = u;
        }
    }
}

__global__ void compute_edge_counts_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ neighborhood,
    int32_t neighborhood_size,
    const uint32_t* __restrict__ bitmap,
    int64_t* edge_counts
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= neighborhood_size) return;

    int32_t v = neighborhood[warp_id];
    int32_t start = __ldg(&offsets[v]);
    int32_t end = __ldg(&offsets[v + 1]);
    int count = 0;

    for (int32_t e = start + lane; e < end; e += 32) {
        int32_t u = __ldg(&indices[e]);
        if (__ldg(&bitmap[u >> 5]) & (1u << (u & 31))) {
            count++;
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        count += __shfl_down_sync(0xFFFFFFFF, count, offset);
    }

    if (lane == 0) edge_counts[warp_id] = (int64_t)count;
}

__global__ void extract_edges_warp_prefix_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ neighborhood,
    int32_t neighborhood_size,
    const uint32_t* __restrict__ bitmap,
    const int64_t* __restrict__ write_offsets,
    int32_t* edge_srcs, int32_t* edge_dsts, float* edge_weights
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= neighborhood_size) return;

    int32_t v = neighborhood[warp_id];
    int32_t start = __ldg(&offsets[v]);
    int32_t end = __ldg(&offsets[v + 1]);
    int64_t base_write = write_offsets[warp_id];
    int64_t local_write = 0;

    for (int32_t batch = start; batch < end; batch += 32) {
        int32_t e = batch + lane;
        bool valid = (e < end);
        int32_t u = valid ? __ldg(&indices[e]) : -1;
        bool match = valid && (__ldg(&bitmap[u >> 5]) & (1u << (u & 31)));
        float w = match ? __ldg(&weights[e]) : 0.0f;

        uint32_t ballot = __ballot_sync(0xFFFFFFFF, match);
        int prefix = __popc(ballot & ((1u << lane) - 1));

        if (match) {
            int64_t idx = base_write + local_write + prefix;
            edge_srcs[idx] = v;
            edge_dsts[idx] = u;
            edge_weights[idx] = w;
        }

        local_write += __popc(ballot);
    }
}



void launch_init_source(uint32_t* bitmap, int32_t* neighborhood,
                        int32_t* neighborhood_size, int32_t source) {
    init_source_kernel<<<1, 1>>>(bitmap, neighborhood, neighborhood_size, source);
}

void launch_bfs_expand(
    const int32_t* offsets, const int32_t* indices,
    int32_t* neighborhood, int32_t frontier_start, int32_t frontier_end,
    int32_t* neighborhood_size, uint32_t* bitmap
) {
    int frontier_size = frontier_end - frontier_start;
    if (frontier_size <= 0) return;
    int64_t total_threads = (int64_t)frontier_size * 32;
    int block = 256;
    int grid = (int)((total_threads + block - 1) / block);
    bfs_expand_warp_kernel<<<grid, block>>>(offsets, indices, neighborhood,
                                             frontier_start, frontier_end,
                                             neighborhood_size, bitmap);
}

void launch_compute_edge_counts(
    const int32_t* offsets, const int32_t* indices,
    const int32_t* neighborhood, int32_t neighborhood_size,
    const uint32_t* bitmap, int64_t* edge_counts
) {
    if (neighborhood_size <= 0) return;
    int64_t total_threads = (int64_t)neighborhood_size * 32;
    int block = 256;
    int grid = (int)((total_threads + block - 1) / block);
    compute_edge_counts_warp_kernel<<<grid, block>>>(offsets, indices, neighborhood,
                                                      neighborhood_size, bitmap, edge_counts);
}

void launch_exclusive_scan(
    const int64_t* input, int64_t* output, int32_t n,
    void* d_temp, size_t& temp_bytes
) {
    if (n <= 0) return;
    if (d_temp == nullptr) {
        cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, input, output, n);
        return;
    }
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, input, output, n);
}

void launch_extract_edges(
    const int32_t* offsets, const int32_t* indices, const float* weights,
    const int32_t* neighborhood, int32_t neighborhood_size,
    const uint32_t* bitmap, const int64_t* write_offsets,
    int32_t* edge_srcs, int32_t* edge_dsts, float* edge_weights
) {
    if (neighborhood_size <= 0) return;
    int64_t total_threads = (int64_t)neighborhood_size * 32;
    int block = 256;
    int grid = (int)((total_threads + block - 1) / block);
    extract_edges_warp_prefix_kernel<<<grid, block>>>(offsets, indices, weights,
                                                       neighborhood, neighborhood_size,
                                                       bitmap, write_offsets,
                                                       edge_srcs, edge_dsts, edge_weights);
}

void launch_sort_edges(int32_t* edge_srcs, int32_t* edge_dsts, float* edge_weights, int64_t count) {
    if (count <= 1) return;
    auto begin = thrust::make_zip_iterator(thrust::make_tuple(edge_srcs, edge_dsts, edge_weights));
    thrust::sort(thrust::device, begin, begin + count);
}



int32_t do_bfs(const int32_t* d_offsets, const int32_t* d_indices,
               int32_t source, int32_t radius,
               uint32_t* d_bitmap, size_t bitmap_bytes,
               int32_t* d_neighborhood, int32_t* d_ns_counter) {
    cudaMemsetAsync(d_bitmap, 0, bitmap_bytes);
    launch_init_source(d_bitmap, d_neighborhood, d_ns_counter, source);

    int32_t frontier_start = 0, frontier_end = 1, h_ns = 1;
    for (int32_t hop = 0; hop < radius; hop++) {
        if (frontier_start >= frontier_end) break;
        launch_bfs_expand(d_offsets, d_indices, d_neighborhood,
                         frontier_start, frontier_end, d_ns_counter, d_bitmap);
        frontier_start = frontier_end;
        cudaMemcpy(&h_ns, d_ns_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
        frontier_end = h_ns;
    }
    return h_ns;
}

}  

extract_ego_weighted_result_float_t extract_ego_weighted_f32(
    const graph32_t& graph,
    const float* edge_weights,
    const int32_t* source_vertices,
    std::size_t n_sources,
    int32_t radius) {

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;

    
    std::vector<int32_t> h_sources(n_sources);
    if (n_sources > 0) {
        cudaMemcpy(h_sources.data(), source_vertices,
                   n_sources * sizeof(int32_t), cudaMemcpyDeviceToHost);
    }

    cache.ensure(num_vertices);

    if (n_sources == 0) {
        extract_ego_weighted_result_float_t result;
        result.edge_srcs = nullptr;
        result.edge_dsts = nullptr;
        result.edge_weights = nullptr;
        cudaMalloc(&result.offsets, sizeof(std::size_t));
        std::size_t zero = 0;
        cudaMemcpy(result.offsets, &zero, sizeof(std::size_t), cudaMemcpyHostToDevice);
        result.num_edges = 0;
        result.num_offsets = 1;
        return result;
    }

    
    struct EgoResult {
        int32_t* srcs = nullptr;
        int32_t* dsts = nullptr;
        float* wts = nullptr;
        int64_t count = 0;
    };
    std::vector<EgoResult> results(n_sources);

    for (std::size_t s = 0; s < n_sources; s++) {
        int32_t ns = do_bfs(d_offsets, d_indices, h_sources[s], radius,
                            cache.d_bitmap, cache.bitmap_bytes,
                            cache.d_neighborhood, cache.d_ns_counter);

        if (ns <= 0) { results[s].count = 0; continue; }

        
        launch_compute_edge_counts(d_offsets, d_indices, cache.d_neighborhood, ns,
                                   cache.d_bitmap, cache.d_edge_counts);
        
        launch_exclusive_scan(cache.d_edge_counts, cache.d_write_offsets, ns,
                              cache.d_scan_temp, cache.scan_temp_bytes);

        
        int64_t last_offset, last_count;
        cudaMemcpy(&last_offset, cache.d_write_offsets + ns - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_count, cache.d_edge_counts + ns - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
        int64_t ec = last_offset + last_count;
        results[s].count = ec;

        if (ec <= 0) continue;

        
        cudaMalloc(&results[s].srcs, ec * sizeof(int32_t));
        cudaMalloc(&results[s].dsts, ec * sizeof(int32_t));
        cudaMalloc(&results[s].wts, ec * sizeof(float));

        launch_extract_edges(d_offsets, d_indices, edge_weights,
                            cache.d_neighborhood, ns, cache.d_bitmap, cache.d_write_offsets,
                            results[s].srcs, results[s].dsts, results[s].wts);

        
        launch_sort_edges(results[s].srcs, results[s].dsts, results[s].wts, ec);
    }

    
    std::vector<std::size_t> h_ego_offsets(n_sources + 1, 0);
    for (std::size_t s = 0; s < n_sources; s++) {
        h_ego_offsets[s + 1] = h_ego_offsets[s] + (std::size_t)results[s].count;
    }
    std::size_t total_edges = h_ego_offsets[n_sources];

    extract_ego_weighted_result_float_t result;
    result.num_edges = total_edges;
    result.num_offsets = n_sources + 1;

    if (total_edges > 0) {
        cudaMalloc(&result.edge_srcs, total_edges * sizeof(int32_t));
        cudaMalloc(&result.edge_dsts, total_edges * sizeof(int32_t));
        cudaMalloc(&result.edge_weights, total_edges * sizeof(float));
    } else {
        result.edge_srcs = nullptr;
        result.edge_dsts = nullptr;
        result.edge_weights = nullptr;
    }

    cudaMalloc(&result.offsets, (n_sources + 1) * sizeof(std::size_t));

    
    for (std::size_t s = 0; s < n_sources; s++) {
        if (results[s].count <= 0) continue;
        std::size_t off = h_ego_offsets[s];
        size_t bytes_i = results[s].count * sizeof(int32_t);
        size_t bytes_f = results[s].count * sizeof(float);
        cudaMemcpyAsync(result.edge_srcs + off, results[s].srcs, bytes_i, cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(result.edge_dsts + off, results[s].dsts, bytes_i, cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(result.edge_weights + off, results[s].wts, bytes_f, cudaMemcpyDeviceToDevice);
    }

    cudaMemcpy(result.offsets, h_ego_offsets.data(),
               (n_sources + 1) * sizeof(std::size_t), cudaMemcpyHostToDevice);

    
    for (std::size_t s = 0; s < n_sources; s++) {
        if (results[s].srcs) cudaFree(results[s].srcs);
        if (results[s].dsts) cudaFree(results[s].dsts);
        if (results[s].wts) cudaFree(results[s].wts);
    }

    return result;
}

}  
