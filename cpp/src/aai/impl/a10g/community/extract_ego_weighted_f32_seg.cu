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

namespace aai {

namespace {

struct Cache : Cacheable {
    void* d_temp = nullptr;
    size_t temp_size = 0;

    ~Cache() override {
        if (d_temp) cudaFree(d_temp);
    }
};

static inline int compute_bits(int32_t val) {
    if (val <= 1) return 1;
    int bits = 0; val--;
    while (val > 0) { val >>= 1; bits++; }
    return bits;
}


__global__ void bfs_init_kernel(
    const int32_t* __restrict__ sources,
    uint32_t* __restrict__ visited,
    uint32_t* __restrict__ frontier,
    int32_t n_sources, int32_t bitmap_words
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_sources) return;
    int32_t src = sources[s];
    int64_t base = (int64_t)s * bitmap_words;
    visited[base + (src >> 5)] = 1u << (src & 31);
    frontier[base + (src >> 5)] = 1u << (src & 31);
}


__global__ void bfs_expand_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    uint32_t* __restrict__ visited,
    const uint32_t* __restrict__ frontier,
    uint32_t* __restrict__ next_frontier,
    int32_t num_vertices, int32_t n_sources, int32_t bitmap_words
) {
    int source_id = blockIdx.y;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid >> 5;
    int lane = threadIdx.x & 31;

    if (source_id >= n_sources || warp_id >= bitmap_words) return;

    int64_t base = (int64_t)source_id * bitmap_words;
    uint32_t word = frontier[base + warp_id];
    if (word == 0) return;

    uint32_t* sv = visited + base;
    uint32_t* sn = next_frontier + base;
    int bv = warp_id << 5;

    while (word) {
        int bit = __ffs(word) - 1;
        word &= word - 1;
        int v = bv + bit;
        if (v >= num_vertices) break;

        int es = csr_offsets[v], ee = csr_offsets[v + 1];
        for (int e = es + lane; e < ee; e += 32) {
            int nb = csr_indices[e];
            int nw = nb >> 5;
            uint32_t nm = 1u << (nb & 31);
            if (!(sv[nw] & nm)) {
                atomicOr(&sv[nw], nm);
                atomicOr(&sn[nw], nm);
            }
        }
    }
}


__global__ void find_nonzero_words_kernel(
    const uint32_t* __restrict__ visited,
    int32_t n_sources, int32_t bitmap_words,
    int32_t* __restrict__ nz_source_ids,
    int32_t* __restrict__ nz_word_indices,
    int32_t* __restrict__ nz_count
) {
    int source_id = blockIdx.y;
    int word_idx = blockIdx.x * blockDim.x + threadIdx.x;

    bool valid = (source_id < n_sources && word_idx < bitmap_words);
    bool nonzero = valid && (visited[(int64_t)source_id * bitmap_words + word_idx] != 0);

    
    uint32_t mask = __ballot_sync(0xFFFFFFFF, nonzero);
    int warp_count = __popc(mask);
    int lane = threadIdx.x & 31;

    int base_pos = 0;
    if (lane == 0 && warp_count > 0) {
        base_pos = atomicAdd(nz_count, warp_count);
    }
    base_pos = __shfl_sync(0xFFFFFFFF, base_pos, 0);

    if (nonzero) {
        int rank = __popc(mask & ((1u << lane) - 1));
        int pos = base_pos + rank;
        nz_source_ids[pos] = source_id;
        nz_word_indices[pos] = word_idx;
    }
}


__global__ void count_ego_edges_compact_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const uint32_t* __restrict__ visited,
    int32_t num_vertices, int32_t bitmap_words,
    const int32_t* __restrict__ nz_source_ids,
    const int32_t* __restrict__ nz_word_indices,
    int32_t total_nz,
    int64_t* __restrict__ edge_counts
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int work_idx = global_tid >> 5;
    int lane = threadIdx.x & 31;

    if (work_idx >= total_nz) return;

    int source_id = nz_source_ids[work_idx];
    int word_idx = nz_word_indices[work_idx];
    int64_t base = (int64_t)source_id * bitmap_words;
    const uint32_t* sv = visited + base;

    uint32_t word = sv[word_idx];
    int bv = word_idx << 5;
    int64_t count = 0;

    while (word) {
        int bit = __ffs(word) - 1;
        word &= word - 1;
        int v = bv + bit;
        if (v >= num_vertices) break;

        int es = csr_offsets[v], ee = csr_offsets[v + 1];
        for (int e = es + lane; e < ee; e += 32) {
            int nb = csr_indices[e];
            if (sv[nb >> 5] & (1u << (nb & 31))) count++;
        }
    }

    
    for (int off = 16; off > 0; off >>= 1)
        count += __shfl_down_sync(0xFFFFFFFF, count, off);

    if (lane == 0 && count > 0)
        atomicAdd((unsigned long long*)&edge_counts[source_id], (unsigned long long)count);
}


__global__ void extract_ego_edges_compact_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const float* __restrict__ edge_weights,
    const uint32_t* __restrict__ visited,
    int32_t num_vertices, int32_t bitmap_words,
    const int32_t* __restrict__ nz_source_ids,
    const int32_t* __restrict__ nz_word_indices,
    int32_t total_nz,
    uint64_t* __restrict__ out_keys,
    float* __restrict__ out_weights,
    const int64_t* __restrict__ ego_offsets,
    unsigned long long* __restrict__ write_pos,
    int32_t bpv
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int work_idx = global_tid >> 5;
    int lane = threadIdx.x & 31;

    if (work_idx >= total_nz) return;

    int source_id = nz_source_ids[work_idx];
    int word_idx = nz_word_indices[work_idx];
    int64_t base = (int64_t)source_id * bitmap_words;
    const uint32_t* sv = visited + base;

    uint32_t word = sv[word_idx];
    int bv = word_idx << 5;
    int64_t offset = ego_offsets[source_id];
    uint64_t ego_prefix = (uint64_t)source_id << (2 * bpv);

    while (word) {
        int bit = __ffs(word) - 1;
        word &= word - 1;
        int v = bv + bit;
        if (v >= num_vertices) break;

        int es = csr_offsets[v], ee = csr_offsets[v + 1];
        uint64_t src_part = (uint64_t)v << bpv;

        for (int e = es + lane; e < ee; e += 32) {
            int nb = csr_indices[e];
            if (sv[nb >> 5] & (1u << (nb & 31))) {
                unsigned long long pos = atomicAdd(&write_pos[source_id], 1ull);
                int64_t idx = offset + (int64_t)pos;
                out_keys[idx] = ego_prefix | src_part | (uint64_t)nb;
                out_weights[idx] = edge_weights[e];
            }
        }
    }
}


__global__ void unpack_sort_keys_kernel(
    const uint64_t* __restrict__ packed_keys,
    int32_t* __restrict__ srcs, int32_t* __restrict__ dsts,
    int64_t n, int32_t bpv
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t key = packed_keys[i];
    uint64_t mask = (1ull << bpv) - 1;
    dsts[i] = (int32_t)(key & mask);
    srcs[i] = (int32_t)((key >> bpv) & mask);
}



void cub_sort_pairs_uint64_float(void* dt, size_t* tb, const uint64_t* ki, uint64_t* ko,
    const float* vi, float* vo, int ni, int eb, cudaStream_t st) {
    cub::DeviceRadixSort::SortPairs(dt, *tb, ki, ko, vi, vo, ni, 0, eb, st);
}

void cub_sort_pairs_float_uint64(void* dt, size_t* tb, const float* ki, float* ko,
    const uint64_t* vi, uint64_t* vo, int ni, cudaStream_t st) {
    cub::DeviceRadixSort::SortPairs(dt, *tb, ki, ko, vi, vo, ni, 0, 32, st);
}

}  

extract_ego_weighted_result_float_t extract_ego_weighted_f32_seg(
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
    bool is_multigraph = graph.is_multigraph;
    int32_t ns = static_cast<int32_t>(n_sources);

    cudaStream_t stream = 0;

    int32_t bitmap_words = (num_vertices + 31) / 32;
    int64_t bitmap_total = (int64_t)ns * bitmap_words;
    int32_t bpv = compute_bits(num_vertices);
    int32_t ego_bits = compute_bits(ns);
    int end_bit = ego_bits + 2 * bpv;
    if (end_bit > 64) end_bit = 64;

    
    uint32_t* d_visited = nullptr;
    uint32_t* d_frontier = nullptr;
    uint32_t* d_next_frontier = nullptr;
    if (bitmap_total > 0) {
        cudaMalloc(&d_visited, bitmap_total * sizeof(uint32_t));
        cudaMalloc(&d_frontier, bitmap_total * sizeof(uint32_t));
        cudaMalloc(&d_next_frontier, bitmap_total * sizeof(uint32_t));

        cudaMemsetAsync(d_visited, 0, bitmap_total * sizeof(uint32_t), stream);
        cudaMemsetAsync(d_frontier, 0, bitmap_total * sizeof(uint32_t), stream);
    }

    if (ns > 0) {
        bfs_init_kernel<<<(ns + 255) / 256, 256, 0, stream>>>(
            source_vertices, d_visited, d_frontier, ns, bitmap_words);
    }

    for (int hop = 0; hop < radius; hop++) {
        if (bitmap_total > 0) {
            cudaMemsetAsync(d_next_frontier, 0, bitmap_total * sizeof(uint32_t), stream);
        }
        if (ns > 0 && bitmap_words > 0) {
            dim3 grid((bitmap_words + 7) / 8, ns);
            bfs_expand_kernel<<<grid, 256, 0, stream>>>(
                d_offsets, d_indices, d_visited, d_frontier, d_next_frontier,
                num_vertices, ns, bitmap_words);
        }
        std::swap(d_frontier, d_next_frontier);
    }

    
    int64_t max_nz = bitmap_total;
    int32_t* d_nz_sids = nullptr;
    int32_t* d_nz_wids = nullptr;
    int32_t* d_nz_count = nullptr;
    if (max_nz > 0) {
        cudaMalloc(&d_nz_sids, max_nz * sizeof(int32_t));
        cudaMalloc(&d_nz_wids, max_nz * sizeof(int32_t));
    }
    cudaMalloc(&d_nz_count, sizeof(int32_t));
    cudaMemsetAsync(d_nz_count, 0, sizeof(int32_t), stream);

    if (ns > 0 && bitmap_words > 0) {
        dim3 grid((bitmap_words + 255) / 256, ns);
        find_nonzero_words_kernel<<<grid, 256, 0, stream>>>(
            d_visited, ns, bitmap_words, d_nz_sids, d_nz_wids, d_nz_count);
    }

    int32_t h_total_nz = 0;
    cudaMemcpy(&h_total_nz, d_nz_count, sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    int64_t* d_edge_counts = nullptr;
    if (ns > 0) {
        cudaMalloc(&d_edge_counts, ns * sizeof(int64_t));
        cudaMemsetAsync(d_edge_counts, 0, ns * sizeof(int64_t), stream);
    }

    if (h_total_nz > 0) {
        int64_t threads = (int64_t)h_total_nz * 32;
        int block = 256;
        int grid_size = (int)((threads + block - 1) / block);
        count_ego_edges_compact_kernel<<<grid_size, block, 0, stream>>>(
            d_offsets, d_indices, d_visited, num_vertices, bitmap_words,
            d_nz_sids, d_nz_wids, h_total_nz, d_edge_counts);
    }

    std::vector<int64_t> h_counts(ns);
    if (ns > 0) {
        cudaMemcpy(h_counts.data(), d_edge_counts, ns * sizeof(int64_t), cudaMemcpyDeviceToHost);
    }

    std::vector<int64_t> h_offsets(ns + 1);
    h_offsets[0] = 0;
    for (int i = 0; i < ns; i++)
        h_offsets[i + 1] = h_offsets[i] + h_counts[i];
    int64_t total_edges = h_offsets[ns];

    
    int32_t* out_srcs = nullptr;
    int32_t* out_dsts = nullptr;
    float* out_wts = nullptr;
    int64_t* d_out_offs = nullptr;

    if (total_edges > 0) {
        cudaMalloc(&out_srcs, total_edges * sizeof(int32_t));
        cudaMalloc(&out_dsts, total_edges * sizeof(int32_t));
        cudaMalloc(&out_wts, total_edges * sizeof(float));
    }

    cudaMalloc(&d_out_offs, (ns + 1) * sizeof(int64_t));
    cudaMemcpyAsync(d_out_offs, h_offsets.data(),
                    (ns + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    if (total_edges > 0) {
        int num_items = (int)total_edges;

        uint64_t* d_keys1 = nullptr;
        uint64_t* d_keys2 = nullptr;
        float* d_wbuf = nullptr;
        cudaMalloc(&d_keys1, total_edges * sizeof(uint64_t));
        cudaMalloc(&d_keys2, total_edges * sizeof(uint64_t));
        cudaMalloc(&d_wbuf, total_edges * sizeof(float));

        unsigned long long* d_write_pos = nullptr;
        cudaMalloc(&d_write_pos, ns * sizeof(unsigned long long));
        cudaMemsetAsync(d_write_pos, 0, ns * sizeof(unsigned long long), stream);

        if (h_total_nz > 0) {
            int64_t threads = (int64_t)h_total_nz * 32;
            int block = 256;
            int grid_size = (int)((threads + block - 1) / block);
            extract_ego_edges_compact_kernel<<<grid_size, block, 0, stream>>>(
                d_offsets, d_indices, edge_weights, d_visited,
                num_vertices, bitmap_words,
                d_nz_sids, d_nz_wids, h_total_nz,
                d_keys1, out_wts, d_out_offs, d_write_pos, bpv);
        }

        if (is_multigraph) {
            float* d_fbuf = nullptr;
            cudaMalloc(&d_fbuf, total_edges * sizeof(float));

            size_t t1 = 0, t2 = 0;
            cub_sort_pairs_float_uint64(nullptr, &t1, out_wts, d_fbuf,
                d_keys1, d_keys2, num_items, stream);
            cub_sort_pairs_uint64_float(nullptr, &t2, d_keys2, d_keys1,
                d_fbuf, out_wts, num_items, end_bit, stream);

            size_t needed = (t1 > t2) ? t1 : t2;
            if (needed > cache.temp_size) {
                if (cache.d_temp) cudaFree(cache.d_temp);
                cudaMalloc(&cache.d_temp, needed);
                cache.temp_size = needed;
            }

            cub_sort_pairs_float_uint64(cache.d_temp, &t1, out_wts, d_fbuf,
                d_keys1, d_keys2, num_items, stream);
            cub_sort_pairs_uint64_float(cache.d_temp, &t2, d_keys2, d_keys1,
                d_fbuf, out_wts, num_items, end_bit, stream);

            unpack_sort_keys_kernel<<<(int)((total_edges + 255) / 256), 256, 0, stream>>>(
                d_keys1, out_srcs, out_dsts, total_edges, bpv);

            cudaFree(d_fbuf);
        } else {
            size_t t1 = 0;
            cub_sort_pairs_uint64_float(nullptr, &t1, d_keys1, d_keys2,
                out_wts, d_wbuf, num_items, end_bit, stream);

            if (t1 > cache.temp_size) {
                if (cache.d_temp) cudaFree(cache.d_temp);
                cudaMalloc(&cache.d_temp, t1);
                cache.temp_size = t1;
            }

            cub_sort_pairs_uint64_float(cache.d_temp, &t1, d_keys1, d_keys2,
                out_wts, d_wbuf, num_items, end_bit, stream);

            cudaMemcpyAsync(out_wts, d_wbuf,
                           total_edges * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            unpack_sort_keys_kernel<<<(int)((total_edges + 255) / 256), 256, 0, stream>>>(
                d_keys2, out_srcs, out_dsts, total_edges, bpv);
        }

        cudaFree(d_keys1);
        cudaFree(d_keys2);
        cudaFree(d_wbuf);
        cudaFree(d_write_pos);
    }

    
    if (d_visited) cudaFree(d_visited);
    if (d_frontier) cudaFree(d_frontier);
    if (d_next_frontier) cudaFree(d_next_frontier);
    if (d_nz_sids) cudaFree(d_nz_sids);
    if (d_nz_wids) cudaFree(d_nz_wids);
    cudaFree(d_nz_count);
    if (d_edge_counts) cudaFree(d_edge_counts);

    extract_ego_weighted_result_float_t result;
    result.edge_srcs = out_srcs;
    result.edge_dsts = out_dsts;
    result.edge_weights = out_wts;
    result.offsets = reinterpret_cast<std::size_t*>(d_out_offs);
    result.num_edges = static_cast<std::size_t>(total_edges);
    result.num_offsets = static_cast<std::size_t>(ns + 1);
    return result;
}

}  
