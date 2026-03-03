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
#include <vector>
#include <algorithm>

namespace aai {

namespace {



__global__ void bfs_init_kernel(
    int32_t source, uint32_t* bitmap, int32_t* ego_vertices, int* ego_size
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        bitmap[(uint32_t)source >> 5] |= (1u << (source & 31));
        ego_vertices[0] = source;
        *ego_size = 1;
    }
}

__global__ void bfs_expand_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ ego_vertices,
    int frontier_start, int frontier_size,
    int* __restrict__ ego_size,
    uint32_t* __restrict__ bitmap
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= frontier_size) return;

    int32_t v = ego_vertices[frontier_start + warp_id];
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    for (int32_t e = start + lane; e < end; e += 32) {
        int32_t u = indices[e];
        uint32_t word_idx = (uint32_t)u >> 5;
        uint32_t bit_mask = 1u << (u & 31);
        uint32_t old = atomicOr(&bitmap[word_idx], bit_mask);
        if (!(old & bit_mask)) {
            int pos = atomicAdd(ego_size, 1);
            ego_vertices[pos] = u;
        }
    }
}



__global__ void count_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ bitmap,
    const int32_t* __restrict__ ego_vertices,
    int ego_size,
    int64_t* __restrict__ counts
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= ego_size) return;

    int32_t v = ego_vertices[warp_id];
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int count = 0;

    for (int32_t e = start + lane; e < end; e += 32) {
        int32_t u = indices[e];
        if (bitmap[(uint32_t)u >> 5] & (1u << (u & 31))) count++;
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        count += __shfl_down_sync(0xffffffff, count, s);

    if (lane == 0) counts[warp_id] = count;
}

__global__ void write_edges_atomic_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ bitmap,
    const int32_t* __restrict__ ego_vertices,
    int ego_size,
    int32_t* __restrict__ out_srcs,
    int32_t* __restrict__ out_dsts,
    double* __restrict__ out_weights,
    int* __restrict__ write_counter
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= ego_size) return;

    int32_t v = ego_vertices[warp_id];
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    for (int32_t e_base = start; e_base < end; e_base += 32) {
        int32_t e = e_base + lane;
        bool valid = e < end;
        int32_t u = valid ? indices[e] : -1;
        bool match = valid && (bitmap[(uint32_t)u >> 5] & (1u << (u & 31)));

        unsigned ballot = __ballot_sync(0xffffffff, match);
        int total = __popc(ballot);
        int rank = __popc(ballot & ((1u << lane) - 1));

        int base_pos = 0;
        if (lane == 0 && total > 0) base_pos = atomicAdd(write_counter, total);
        base_pos = __shfl_sync(0xffffffff, base_pos, 0);

        if (match) {
            int pos = base_pos + rank;
            out_srcs[pos] = v;
            out_dsts[pos] = u;
            out_weights[pos] = weights[e];
        }
    }
}



__global__ void clear_bitmap_kernel(
    const int32_t* __restrict__ ego_vertices, int ego_size, uint32_t* __restrict__ bitmap
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ego_size) return;
    int32_t v = ego_vertices[tid];
    atomicAnd(&bitmap[(uint32_t)v >> 5], ~(1u << (v & 31)));
}

__global__ void mark_bitmap_kernel(
    const int32_t* __restrict__ ego_vertices, int ego_size, uint32_t* __restrict__ bitmap
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ego_size) return;
    int32_t v = ego_vertices[tid];
    atomicOr(&bitmap[(uint32_t)v >> 5], 1u << (v & 31));
}



__global__ void create_sort_keys_kernel(
    const int32_t* __restrict__ srcs, const int32_t* __restrict__ dsts,
    uint64_t* __restrict__ keys, int64_t n
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    keys[tid] = ((uint64_t)(uint32_t)srcs[tid] << 32) | (uint32_t)dsts[tid];
}

__global__ void init_indices_kernel(int32_t* __restrict__ indices, int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    indices[tid] = (int32_t)tid;
}

__global__ void gather_edges_kernel(
    const int32_t* __restrict__ src_in, const int32_t* __restrict__ dst_in,
    const double* __restrict__ wt_in, const int32_t* __restrict__ perm,
    int32_t* __restrict__ src_out, int32_t* __restrict__ dst_out,
    double* __restrict__ wt_out, int64_t n
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int32_t idx = perm[tid];
    src_out[tid] = src_in[idx];
    dst_out[tid] = dst_in[idx];
    wt_out[tid] = wt_in[idx];
}

__global__ void create_weight_keys_kernel(
    const double* __restrict__ weights, uint64_t* __restrict__ keys, int64_t n
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint64_t bits;
    memcpy(&bits, &weights[tid], sizeof(uint64_t));
    uint64_t mask = (bits >> 63) ? 0xFFFFFFFFFFFFFFFFULL : 0x8000000000000000ULL;
    keys[tid] = bits ^ mask;
}

__global__ void create_keys_from_perm_kernel(
    const int32_t* __restrict__ srcs, const int32_t* __restrict__ dsts,
    const int32_t* __restrict__ perm, uint64_t* __restrict__ keys, int64_t n
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int32_t idx = perm[tid];
    keys[tid] = ((uint64_t)(uint32_t)srcs[idx] << 32) | (uint32_t)dsts[idx];
}



void launch_bfs_init(int32_t source, uint32_t* bitmap, int32_t* ego_vertices, int* ego_size, cudaStream_t stream) {
    bfs_init_kernel<<<1, 1, 0, stream>>>(source, bitmap, ego_vertices, ego_size);
}

void launch_bfs_expand(const int32_t* offsets, const int32_t* indices, int32_t* ego_vertices,
                       int frontier_start, int frontier_size, int* ego_size, uint32_t* bitmap, cudaStream_t stream) {
    if (frontier_size == 0) return;
    int64_t threads = (int64_t)frontier_size * 32;
    int block = 256;
    int grid = (int)((threads + block - 1) / block);
    bfs_expand_kernel<<<grid, block, 0, stream>>>(offsets, indices, ego_vertices, frontier_start, frontier_size, ego_size, bitmap);
}

void launch_count_edges(const int32_t* offsets, const int32_t* indices, const uint32_t* bitmap,
                        const int32_t* ego_vertices, int ego_size, int64_t* counts, cudaStream_t stream) {
    if (ego_size == 0) return;
    int64_t threads = (int64_t)ego_size * 32;
    int block = 256;
    int grid = (int)((threads + block - 1) / block);
    count_edges_kernel<<<grid, block, 0, stream>>>(offsets, indices, bitmap, ego_vertices, ego_size, counts);
}

void launch_write_edges_atomic(const int32_t* offsets, const int32_t* indices, const double* weights,
                               const uint32_t* bitmap, const int32_t* ego_vertices, int ego_size,
                               int32_t* out_srcs, int32_t* out_dsts, double* out_weights,
                               int* write_counter, cudaStream_t stream) {
    if (ego_size == 0) return;
    int64_t threads = (int64_t)ego_size * 32;
    int block = 256;
    int grid = (int)((threads + block - 1) / block);
    write_edges_atomic_kernel<<<grid, block, 0, stream>>>(offsets, indices, weights, bitmap, ego_vertices, ego_size,
                                                          out_srcs, out_dsts, out_weights, write_counter);
}

void launch_clear_bitmap(const int32_t* ego_vertices, int ego_size, uint32_t* bitmap, cudaStream_t stream) {
    if (ego_size == 0) return;
    int block = 256;
    int grid = (ego_size + block - 1) / block;
    clear_bitmap_kernel<<<grid, block, 0, stream>>>(ego_vertices, ego_size, bitmap);
}

void launch_mark_bitmap(const int32_t* ego_vertices, int ego_size, uint32_t* bitmap, cudaStream_t stream) {
    if (ego_size == 0) return;
    int block = 256;
    int grid = (ego_size + block - 1) / block;
    mark_bitmap_kernel<<<grid, block, 0, stream>>>(ego_vertices, ego_size, bitmap);
}

size_t get_reduce_temp_size(int n) {
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, temp_bytes, (int64_t*)nullptr, (int64_t*)nullptr, n);
    return temp_bytes;
}

void launch_reduce_sum(const int64_t* d_in, int64_t* d_out, int n, void* d_temp, size_t temp_bytes, cudaStream_t stream) {
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n, stream);
}

size_t get_sort_temp_size(int64_t n) {
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes,
        (uint64_t*)nullptr, (uint64_t*)nullptr,
        (int32_t*)nullptr, (int32_t*)nullptr,
        (int)n, 0, 64);
    return temp_bytes;
}

void launch_sort_edges_cub(
    int32_t* srcs, int32_t* dsts, double* weights, int64_t n,
    uint64_t* d_keys_in, uint64_t* d_keys_out,
    int32_t* d_idx_in, int32_t* d_idx_out,
    int32_t* d_tmp_src, int32_t* d_tmp_dst, double* d_tmp_wt,
    void* d_sort_temp, size_t sort_temp_bytes,
    bool is_multigraph, cudaStream_t stream
) {
    if (n <= 1) return;

    int block = 256;
    int grid = ((int)n + block - 1) / block;

    if (is_multigraph) {
        create_weight_keys_kernel<<<grid, block, 0, stream>>>(weights, d_keys_in, n);
        init_indices_kernel<<<grid, block, 0, stream>>>(d_idx_in, n);

        cub::DeviceRadixSort::SortPairs(d_sort_temp, sort_temp_bytes,
            d_keys_in, d_keys_out, d_idx_in, d_idx_out, (int)n, 0, 64, stream);

        create_keys_from_perm_kernel<<<grid, block, 0, stream>>>(srcs, dsts, d_idx_out, d_keys_in, n);

        cudaMemcpyAsync(d_idx_in, d_idx_out, n * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

        cub::DeviceRadixSort::SortPairs(d_sort_temp, sort_temp_bytes,
            d_keys_in, d_keys_out, d_idx_in, d_idx_out, (int)n, 0, 64, stream);
    } else {
        create_sort_keys_kernel<<<grid, block, 0, stream>>>(srcs, dsts, d_keys_in, n);
        init_indices_kernel<<<grid, block, 0, stream>>>(d_idx_in, n);

        cub::DeviceRadixSort::SortPairs(d_sort_temp, sort_temp_bytes,
            d_keys_in, d_keys_out, d_idx_in, d_idx_out, (int)n, 0, 64, stream);
    }

    gather_edges_kernel<<<grid, block, 0, stream>>>(srcs, dsts, weights, d_idx_out, d_tmp_src, d_tmp_dst, d_tmp_wt, n);

    cudaMemcpyAsync(srcs, d_tmp_src, n * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(dsts, d_tmp_dst, n * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(weights, d_tmp_wt, n * sizeof(double), cudaMemcpyDeviceToDevice, stream);
}



struct Cache : Cacheable {
    
    int* ego_size_counter = nullptr;
    int64_t* total = nullptr;

    
    uint32_t* bitmap = nullptr;
    int64_t bitmap_cap = 0;

    int32_t* ego_vertices = nullptr;
    int64_t ego_cap = 0;

    
    int64_t* counts = nullptr;
    int64_t counts_cap = 0;

    void* reduce_temp = nullptr;
    size_t reduce_temp_cap = 0;

    
    uint64_t* keys_in = nullptr;
    int64_t keys_in_cap = 0;
    uint64_t* keys_out = nullptr;
    int64_t keys_out_cap = 0;
    int32_t* idx_in = nullptr;
    int64_t idx_in_cap = 0;
    int32_t* idx_out = nullptr;
    int64_t idx_out_cap = 0;
    int32_t* tmp_src = nullptr;
    int64_t tmp_src_cap = 0;
    int32_t* tmp_dst = nullptr;
    int64_t tmp_dst_cap = 0;
    double* tmp_wt = nullptr;
    int64_t tmp_wt_cap = 0;
    void* sort_temp = nullptr;
    size_t sort_temp_cap = 0;

    Cache() {
        cudaMalloc(&ego_size_counter, sizeof(int));
        cudaMalloc(&total, sizeof(int64_t));
    }

    ~Cache() override {
        if (ego_size_counter) cudaFree(ego_size_counter);
        if (total) cudaFree(total);
        if (bitmap) cudaFree(bitmap);
        if (ego_vertices) cudaFree(ego_vertices);
        if (counts) cudaFree(counts);
        if (reduce_temp) cudaFree(reduce_temp);
        if (keys_in) cudaFree(keys_in);
        if (keys_out) cudaFree(keys_out);
        if (idx_in) cudaFree(idx_in);
        if (idx_out) cudaFree(idx_out);
        if (tmp_src) cudaFree(tmp_src);
        if (tmp_dst) cudaFree(tmp_dst);
        if (tmp_wt) cudaFree(tmp_wt);
        if (sort_temp) cudaFree(sort_temp);
    }
};

}  

extract_ego_weighted_result_double_t extract_ego_weighted_f64(
    const graph32_t& graph,
    const double* edge_weights,
    const int32_t* source_vertices,
    std::size_t n_sources,
    int32_t radius) {

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    bool is_multigraph = graph.is_multigraph;

    cudaStream_t stream = 0;

    
    std::vector<int32_t> h_sources(n_sources);
    cudaMemcpy(h_sources.data(), source_vertices, n_sources * sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    int64_t bitmap_words = ((int64_t)num_vertices + 31) / 32;
    if (cache.bitmap_cap < bitmap_words) {
        if (cache.bitmap) cudaFree(cache.bitmap);
        cudaMalloc(&cache.bitmap, bitmap_words * sizeof(uint32_t));
        cache.bitmap_cap = bitmap_words;
    }
    if (cache.ego_cap < (int64_t)num_vertices) {
        if (cache.ego_vertices) cudaFree(cache.ego_vertices);
        cudaMalloc(&cache.ego_vertices, (int64_t)num_vertices * sizeof(int32_t));
        cache.ego_cap = num_vertices;
    }
    cudaMemsetAsync(cache.bitmap, 0, (size_t)bitmap_words * sizeof(uint32_t), stream);

    
    int64_t max_counts = std::min((int64_t)num_vertices, (int64_t)10000000);
    if (cache.counts_cap < max_counts) {
        if (cache.counts) cudaFree(cache.counts);
        cudaMalloc(&cache.counts, max_counts * sizeof(int64_t));
        cache.counts_cap = max_counts;
    }
    {
        size_t needed = get_reduce_temp_size((int)max_counts);
        if (needed == 0) needed = 1;
        if (cache.reduce_temp_cap < needed) {
            if (cache.reduce_temp) cudaFree(cache.reduce_temp);
            cudaMalloc(&cache.reduce_temp, needed);
            cache.reduce_temp_cap = needed;
        }
    }

    
    std::vector<std::vector<int32_t>> saved_ego(n_sources);
    std::vector<int64_t> edge_counts(n_sources, 0);

    for (size_t s = 0; s < n_sources; s++) {
        launch_bfs_init(h_sources[s], cache.bitmap, cache.ego_vertices, cache.ego_size_counter, stream);

        int frontier_start = 0, frontier_size = 1, current_ego_size = 1;

        for (int hop = 0; hop < radius; hop++) {
            int prev = current_ego_size;
            launch_bfs_expand(d_offsets, d_indices, cache.ego_vertices, frontier_start, frontier_size, cache.ego_size_counter, cache.bitmap, stream);
            cudaMemcpyAsync(&current_ego_size, cache.ego_size_counter, sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            frontier_start = prev;
            frontier_size = current_ego_size - prev;
            if (frontier_size == 0) break;
        }

        if ((int64_t)current_ego_size > cache.counts_cap) {
            if (cache.counts) cudaFree(cache.counts);
            cudaMalloc(&cache.counts, (int64_t)current_ego_size * sizeof(int64_t));
            cache.counts_cap = current_ego_size;

            size_t needed = get_reduce_temp_size(current_ego_size);
            if (needed == 0) needed = 1;
            if (cache.reduce_temp_cap < needed) {
                if (cache.reduce_temp) cudaFree(cache.reduce_temp);
                cudaMalloc(&cache.reduce_temp, needed);
                cache.reduce_temp_cap = needed;
            }
        }

        launch_count_edges(d_offsets, d_indices, cache.bitmap, cache.ego_vertices, current_ego_size, cache.counts, stream);
        launch_reduce_sum(cache.counts, cache.total, current_ego_size, cache.reduce_temp, cache.reduce_temp_cap, stream);

        saved_ego[s].resize(current_ego_size);
        cudaMemcpyAsync(saved_ego[s].data(), cache.ego_vertices, current_ego_size * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&edge_counts[s], cache.total, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        launch_clear_bitmap(cache.ego_vertices, current_ego_size, cache.bitmap, stream);
    }

    
    int64_t total_edges = 0;
    std::vector<std::size_t> h_ego_offsets(n_sources + 1);
    h_ego_offsets[0] = 0;
    for (size_t s = 0; s < n_sources; s++) {
        total_edges += edge_counts[s];
        h_ego_offsets[s + 1] = (std::size_t)total_edges;
    }

    
    std::size_t* d_out_offsets = nullptr;
    cudaMalloc(&d_out_offsets, (n_sources + 1) * sizeof(std::size_t));
    cudaMemcpyAsync(d_out_offsets, h_ego_offsets.data(), (n_sources + 1) * sizeof(std::size_t), cudaMemcpyHostToDevice, stream);

    if (total_edges == 0) {
        cudaStreamSynchronize(stream);
        return {nullptr, nullptr, nullptr, d_out_offsets, 0, n_sources + 1};
    }

    
    int32_t* d_out_srcs = nullptr;
    int32_t* d_out_dsts = nullptr;
    double* d_out_weights = nullptr;
    cudaMalloc(&d_out_srcs, total_edges * sizeof(int32_t));
    cudaMalloc(&d_out_dsts, total_edges * sizeof(int32_t));
    cudaMalloc(&d_out_weights, total_edges * sizeof(double));

    
    int64_t max_ego_edges = 0;
    for (size_t s = 0; s < n_sources; s++)
        if (edge_counts[s] > max_ego_edges) max_ego_edges = edge_counts[s];

    int64_t sb = max_ego_edges > 0 ? max_ego_edges : (int64_t)1;

    if (cache.keys_in_cap < sb) {
        if (cache.keys_in) cudaFree(cache.keys_in);
        cudaMalloc(&cache.keys_in, sb * sizeof(uint64_t));
        cache.keys_in_cap = sb;
    }
    if (cache.keys_out_cap < sb) {
        if (cache.keys_out) cudaFree(cache.keys_out);
        cudaMalloc(&cache.keys_out, sb * sizeof(uint64_t));
        cache.keys_out_cap = sb;
    }
    if (cache.idx_in_cap < sb) {
        if (cache.idx_in) cudaFree(cache.idx_in);
        cudaMalloc(&cache.idx_in, sb * sizeof(int32_t));
        cache.idx_in_cap = sb;
    }
    if (cache.idx_out_cap < sb) {
        if (cache.idx_out) cudaFree(cache.idx_out);
        cudaMalloc(&cache.idx_out, sb * sizeof(int32_t));
        cache.idx_out_cap = sb;
    }
    if (cache.tmp_src_cap < sb) {
        if (cache.tmp_src) cudaFree(cache.tmp_src);
        cudaMalloc(&cache.tmp_src, sb * sizeof(int32_t));
        cache.tmp_src_cap = sb;
    }
    if (cache.tmp_dst_cap < sb) {
        if (cache.tmp_dst) cudaFree(cache.tmp_dst);
        cudaMalloc(&cache.tmp_dst, sb * sizeof(int32_t));
        cache.tmp_dst_cap = sb;
    }
    if (cache.tmp_wt_cap < sb) {
        if (cache.tmp_wt) cudaFree(cache.tmp_wt);
        cudaMalloc(&cache.tmp_wt, sb * sizeof(double));
        cache.tmp_wt_cap = sb;
    }
    {
        size_t sort_needed = get_sort_temp_size(sb);
        if (sort_needed == 0) sort_needed = 1;
        if (cache.sort_temp_cap < sort_needed) {
            if (cache.sort_temp) cudaFree(cache.sort_temp);
            cudaMalloc(&cache.sort_temp, sort_needed);
            cache.sort_temp_cap = sort_needed;
        }
    }

    
    for (size_t s = 0; s < n_sources; s++) {
        int ego_size = (int)saved_ego[s].size();
        if (ego_size == 0 || edge_counts[s] == 0) continue;

        cudaMemcpyAsync(cache.ego_vertices, saved_ego[s].data(), ego_size * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        launch_mark_bitmap(cache.ego_vertices, ego_size, cache.bitmap, stream);

        cudaMemsetAsync(cache.ego_size_counter, 0, sizeof(int), stream);
        launch_write_edges_atomic(d_offsets, d_indices, edge_weights, cache.bitmap,
                                  cache.ego_vertices, ego_size,
                                  d_out_srcs + h_ego_offsets[s],
                                  d_out_dsts + h_ego_offsets[s],
                                  d_out_weights + h_ego_offsets[s],
                                  cache.ego_size_counter, stream);

        if (edge_counts[s] > 1) {
            launch_sort_edges_cub(
                d_out_srcs + h_ego_offsets[s],
                d_out_dsts + h_ego_offsets[s],
                d_out_weights + h_ego_offsets[s],
                edge_counts[s],
                cache.keys_in, cache.keys_out,
                cache.idx_in, cache.idx_out,
                cache.tmp_src, cache.tmp_dst, cache.tmp_wt,
                cache.sort_temp, cache.sort_temp_cap,
                is_multigraph, stream
            );
        }

        launch_clear_bitmap(cache.ego_vertices, ego_size, cache.bitmap, stream);
    }

    cudaStreamSynchronize(stream);
    return {d_out_srcs, d_out_dsts, d_out_weights, d_out_offsets, (std::size_t)total_edges, n_sources + 1};
}

}  
