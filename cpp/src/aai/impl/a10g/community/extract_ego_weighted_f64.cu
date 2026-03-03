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
#include <cstring>
#include <vector>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* h_pinned = nullptr;

    
    int32_t cached_nv = 0;
    uint32_t* bitmap = nullptr;
    int32_t* fa = nullptr;
    int32_t* fb = nullptr;
    int32_t* ego = nullptr;
    int32_t* ego_alt = nullptr;
    int64_t* cnt = nullptr;
    int64_t* ps = nullptr;
    int32_t* fc = nullptr;
    uint8_t* scan_tmp = nullptr;
    uint8_t* ego_sort_tmp = nullptr;
    size_t scan_tb = 0;
    size_t ego_sort_tb = 0;
    int32_t bitmap_words = 0;

    
    int64_t sort_alloc_n = 0;
    uint64_t* s_ka = nullptr;
    uint64_t* s_kb = nullptr;
    int32_t* s_va = nullptr;
    int32_t* s_vb = nullptr;
    int32_t* s_ts = nullptr;
    int32_t* s_td = nullptr;
    double* s_tw = nullptr;
    uint8_t* s_tmp = nullptr;
    size_t sort_tb = 0;

    Cache() {
        cudaHostAlloc(&h_pinned, 64, cudaHostAllocDefault);
    }

    ~Cache() override {
        if (h_pinned) cudaFreeHost(h_pinned);
        if (bitmap) cudaFree(bitmap);
        if (fa) cudaFree(fa);
        if (fb) cudaFree(fb);
        if (ego) cudaFree(ego);
        if (ego_alt) cudaFree(ego_alt);
        if (cnt) cudaFree(cnt);
        if (ps) cudaFree(ps);
        if (fc) cudaFree(fc);
        if (scan_tmp) cudaFree(scan_tmp);
        if (ego_sort_tmp) cudaFree(ego_sort_tmp);
        if (s_ka) cudaFree(s_ka);
        if (s_kb) cudaFree(s_kb);
        if (s_va) cudaFree(s_va);
        if (s_vb) cudaFree(s_vb);
        if (s_ts) cudaFree(s_ts);
        if (s_td) cudaFree(s_td);
        if (s_tw) cudaFree(s_tw);
        if (s_tmp) cudaFree(s_tmp);
    }

    void ensure_scratch(int32_t num_vertices) {
        if (num_vertices <= cached_nv) return;
        cached_nv = num_vertices;
        bitmap_words = (num_vertices + 31) / 32;

        if (bitmap) cudaFree(bitmap);
        cudaMalloc(&bitmap, (size_t)bitmap_words * sizeof(uint32_t));

        if (fa) cudaFree(fa);
        cudaMalloc(&fa, (size_t)num_vertices * sizeof(int32_t));

        if (fb) cudaFree(fb);
        cudaMalloc(&fb, (size_t)num_vertices * sizeof(int32_t));

        if (ego) cudaFree(ego);
        cudaMalloc(&ego, (size_t)num_vertices * sizeof(int32_t));

        if (ego_alt) cudaFree(ego_alt);
        cudaMalloc(&ego_alt, (size_t)num_vertices * sizeof(int32_t));

        if (cnt) cudaFree(cnt);
        cudaMalloc(&cnt, (size_t)num_vertices * sizeof(int64_t));

        if (ps) cudaFree(ps);
        cudaMalloc(&ps, ((size_t)num_vertices + 1) * sizeof(int64_t));

        if (fc) cudaFree(fc);
        cudaMalloc(&fc, sizeof(int32_t));

        size_t new_scan_tb = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, new_scan_tb,
            (int64_t*)nullptr, (int64_t*)nullptr, num_vertices);
        if (new_scan_tb == 0) new_scan_tb = 1;
        scan_tb = new_scan_tb;
        if (scan_tmp) cudaFree(scan_tmp);
        cudaMalloc(&scan_tmp, scan_tb);

        size_t new_ego_sort_tb = 0;
        cub::DoubleBuffer<int32_t> keys_db;
        cub::DeviceRadixSort::SortKeys(nullptr, new_ego_sort_tb, keys_db, num_vertices);
        if (new_ego_sort_tb == 0) new_ego_sort_tb = 1;
        ego_sort_tb = new_ego_sort_tb;
        if (ego_sort_tmp) cudaFree(ego_sort_tmp);
        cudaMalloc(&ego_sort_tmp, ego_sort_tb);
    }

    void ensure_sort_scratch(int64_t n) {
        if (n <= sort_alloc_n) return;
        sort_alloc_n = n;

        if (s_ka) cudaFree(s_ka);
        cudaMalloc(&s_ka, (size_t)n * sizeof(uint64_t));

        if (s_kb) cudaFree(s_kb);
        cudaMalloc(&s_kb, (size_t)n * sizeof(uint64_t));

        if (s_va) cudaFree(s_va);
        cudaMalloc(&s_va, (size_t)n * sizeof(int32_t));

        if (s_vb) cudaFree(s_vb);
        cudaMalloc(&s_vb, (size_t)n * sizeof(int32_t));

        if (s_ts) cudaFree(s_ts);
        cudaMalloc(&s_ts, (size_t)n * sizeof(int32_t));

        if (s_td) cudaFree(s_td);
        cudaMalloc(&s_td, (size_t)n * sizeof(int32_t));

        if (s_tw) cudaFree(s_tw);
        cudaMalloc(&s_tw, (size_t)n * sizeof(double));

        size_t new_sort_tb = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, new_sort_tb,
            (uint64_t*)nullptr, (uint64_t*)nullptr,
            (int32_t*)nullptr, (int32_t*)nullptr, (int32_t)n);
        if (new_sort_tb == 0) new_sort_tb = 1;
        sort_tb = new_sort_tb;
        if (s_tmp) cudaFree(s_tmp);
        cudaMalloc(&s_tmp, sort_tb);
    }
};

static int compute_vertex_bits(int32_t num_vertices) {
    if (num_vertices <= 1) return 1;
    int bits = 0;
    int32_t v = num_vertices - 1;
    while (v > 0) { bits++; v >>= 1; }
    return bits;
}



__global__ void init_source_kernel(uint32_t* bitmap, int32_t* frontier, int32_t source) {
    bitmap[source >> 5] |= (1u << (source & 31));
    frontier[0] = source;
}

__global__ void bfs_expand_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    uint32_t* __restrict__ visited,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= frontier_size) return;

    int32_t v = frontier[warp_id];
    int32_t begin = csr_offsets[v];
    int32_t end = csr_offsets[v + 1];

    for (int32_t e = begin + lane; e < end; e += 32) {
        int32_t u = csr_indices[e];
        uint32_t word_idx = u >> 5;
        uint32_t bit_mask = 1u << (u & 31);
        uint32_t old_val = atomicOr(&visited[word_idx], bit_mask);
        if (!(old_val & bit_mask)) {
            int32_t pos = atomicAdd(next_count, 1);
            next_frontier[pos] = u;
        }
    }
}


__global__ void clear_bitmap_selective(
    uint32_t* __restrict__ bitmap,
    const int32_t* __restrict__ vertices,
    int32_t n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int32_t v = vertices[tid];
    atomicAnd(&bitmap[v >> 5], ~(1u << (v & 31)));
}

__global__ void count_ego_edges_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const int32_t* __restrict__ ego_vertices,
    int32_t ego_size,
    const uint32_t* __restrict__ visited,
    int64_t* __restrict__ edge_counts
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= ego_size) return;

    int32_t v = ego_vertices[warp_id];
    int32_t begin = csr_offsets[v];
    int32_t end = csr_offsets[v + 1];
    int count = 0;

    for (int32_t e = begin + lane; e < end; e += 32) {
        int32_t u = csr_indices[e];
        if ((visited[u >> 5] >> (u & 31)) & 1u) count++;
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        count += __shfl_down_sync(0xffffffff, count, offset);

    if (lane == 0) edge_counts[warp_id] = count;
}

__global__ void fill_ego_edges_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const double* __restrict__ csr_weights,
    const int32_t* __restrict__ ego_vertices,
    int32_t ego_size,
    const uint32_t* __restrict__ visited,
    const int64_t* __restrict__ edge_offsets,
    int64_t global_offset,
    int32_t* __restrict__ out_srcs,
    int32_t* __restrict__ out_dsts,
    double* __restrict__ out_weights
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= ego_size) return;

    int32_t v = ego_vertices[warp_id];
    int32_t begin = csr_offsets[v];
    int32_t end = csr_offsets[v + 1];
    int64_t base_wp = global_offset + edge_offsets[warp_id];
    int64_t wp_counter = 0;

    for (int32_t chunk_start = begin; chunk_start < end; chunk_start += 32) {
        int32_t e = chunk_start + lane;
        bool valid = (e < end);
        int32_t u = valid ? csr_indices[e] : -1;
        bool in_ego = valid && ((visited[u >> 5] >> (u & 31)) & 1u);

        uint32_t ballot = __ballot_sync(0xffffffff, in_ego);
        int count_before = __popc(ballot & ((1u << lane) - 1));
        int total = __popc(ballot);

        if (in_ego) {
            int64_t pos = base_wp + wp_counter + count_before;
            out_srcs[pos] = v;
            out_dsts[pos] = u;
            out_weights[pos] = csr_weights[e];
        }

        wp_counter += total;
    }
}

__global__ void check_sorted_kernel(
    const int32_t* __restrict__ srcs,
    const int32_t* __restrict__ dsts,
    const double* __restrict__ weights,
    int64_t n,
    int32_t* __restrict__ is_unsorted
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n - 1) return;

    bool bad = false;
    int32_t s0 = srcs[tid], s1 = srcs[tid + 1];
    if (s0 > s1) {
        bad = true;
    } else if (s0 == s1) {
        int32_t d0 = dsts[tid], d1 = dsts[tid + 1];
        if (d0 > d1) {
            bad = true;
        } else if (d0 == d1) {
            if (weights[tid] > weights[tid + 1]) {
                bad = true;
            }
        }
    }

    if (bad) atomicExch(is_unsorted, 1);
}

__global__ void create_sort_keys_and_iota(
    const int32_t* __restrict__ srcs,
    const int32_t* __restrict__ dsts,
    uint64_t* __restrict__ keys,
    int32_t* __restrict__ vals,
    int64_t n
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    keys[tid] = ((uint64_t)(uint32_t)srcs[tid] << 32) | (uint64_t)(uint32_t)dsts[tid];
    vals[tid] = (int32_t)tid;
}

__global__ void gather_edges(
    const int32_t* __restrict__ perm,
    const int32_t* __restrict__ in_srcs,
    const int32_t* __restrict__ in_dsts,
    const double* __restrict__ in_weights,
    int32_t* __restrict__ out_srcs,
    int32_t* __restrict__ out_dsts,
    double* __restrict__ out_weights,
    int64_t n
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    int32_t idx = perm[tid];
    out_srcs[tid] = in_srcs[idx];
    out_dsts[tid] = in_dsts[idx];
    out_weights[tid] = in_weights[idx];
}

__global__ void create_weight_keys_and_iota(
    const double* __restrict__ weights,
    uint64_t* __restrict__ keys,
    int32_t* __restrict__ vals,
    int64_t n
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint64_t bits;
    memcpy(&bits, &weights[tid], sizeof(uint64_t));
    uint64_t mask = -((int64_t)(bits >> 63)) | 0x8000000000000000ULL;
    keys[tid] = bits ^ mask;
    vals[tid] = (int32_t)tid;
}



static void launch_init_source(uint32_t* bitmap, int32_t* frontier, int32_t source) {
    init_source_kernel<<<1, 1>>>(bitmap, frontier, source);
}

static void launch_bfs_expand(
    const int32_t* csr_offsets, const int32_t* csr_indices,
    const int32_t* frontier, int32_t frontier_size,
    uint32_t* visited, int32_t* next_frontier, int32_t* next_count
) {
    if (frontier_size == 0) return;
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (frontier_size + warps_per_block - 1) / warps_per_block;
    bfs_expand_kernel<<<blocks, threads>>>(
        csr_offsets, csr_indices, frontier, frontier_size,
        visited, next_frontier, next_count
    );
}

static void launch_clear_bitmap(uint32_t* bitmap, const int32_t* vertices, int32_t n) {
    if (n == 0) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    clear_bitmap_selective<<<blocks, threads>>>(bitmap, vertices, n);
}

static void launch_count_ego_edges(
    const int32_t* csr_offsets, const int32_t* csr_indices,
    const int32_t* ego_vertices, int32_t ego_size,
    const uint32_t* visited, int64_t* edge_counts
) {
    if (ego_size == 0) return;
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (ego_size + warps_per_block - 1) / warps_per_block;
    count_ego_edges_kernel<<<blocks, threads>>>(
        csr_offsets, csr_indices, ego_vertices, ego_size, visited, edge_counts
    );
}

static void launch_fill_ego_edges(
    const int32_t* csr_offsets, const int32_t* csr_indices,
    const double* csr_weights, const int32_t* ego_vertices, int32_t ego_size,
    const uint32_t* visited, const int64_t* edge_offsets,
    int64_t global_offset,
    int32_t* out_srcs, int32_t* out_dsts, double* out_weights
) {
    if (ego_size == 0) return;
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (ego_size + warps_per_block - 1) / warps_per_block;
    fill_ego_edges_kernel<<<blocks, threads>>>(
        csr_offsets, csr_indices, csr_weights, ego_vertices, ego_size,
        visited, edge_offsets, global_offset, out_srcs, out_dsts, out_weights
    );
}

static void launch_check_sorted(
    const int32_t* srcs, const int32_t* dsts, const double* weights,
    int64_t n, int32_t* is_unsorted
) {
    if (n <= 1) return;
    int threads = 256;
    int blocks = ((int)(n - 1) + threads - 1) / threads;
    check_sorted_kernel<<<blocks, threads>>>(srcs, dsts, weights, n, is_unsorted);
}

static void launch_sort_int32(
    int32_t* data, int32_t* alt_buf, int32_t n,
    void* tmp, size_t tmp_bytes, int end_bit
) {
    if (n <= 1) return;
    cub::DoubleBuffer<int32_t> keys(data, alt_buf);
    size_t tb = tmp_bytes;
    cub::DeviceRadixSort::SortKeys(tmp, tb, keys, n, 0, end_bit);
    if (keys.Current() != data) {
        cudaMemcpyAsync(data, keys.Current(), (size_t)n * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    }
}

static void launch_sort_edges(
    int32_t* srcs, int32_t* dsts, double* weights, int32_t n,
    uint64_t* keys_a, uint64_t* keys_b,
    int32_t* vals_a, int32_t* vals_b,
    int32_t* tmp_s, int32_t* tmp_d, double* tmp_w,
    void* sort_tmp, size_t sort_tmp_bytes,
    bool multigraph, int end_bit
) {
    if (n <= 1) return;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    if (multigraph) {
        create_weight_keys_and_iota<<<blocks, threads>>>(weights, keys_a, vals_a, n);
        size_t tb = sort_tmp_bytes;
        cub::DeviceRadixSort::SortPairs(sort_tmp, tb, keys_a, keys_b, vals_a, vals_b, n);
        gather_edges<<<blocks, threads>>>(vals_b, srcs, dsts, weights, tmp_s, tmp_d, tmp_w, n);

        create_sort_keys_and_iota<<<blocks, threads>>>(tmp_s, tmp_d, keys_a, vals_a, n);
        tb = sort_tmp_bytes;
        cub::DeviceRadixSort::SortPairs(sort_tmp, tb, keys_a, keys_b, vals_a, vals_b, n, 0, end_bit);
        gather_edges<<<blocks, threads>>>(vals_b, tmp_s, tmp_d, tmp_w, srcs, dsts, weights, n);
    } else {
        create_sort_keys_and_iota<<<blocks, threads>>>(srcs, dsts, keys_a, vals_a, n);
        size_t tb = sort_tmp_bytes;
        cub::DeviceRadixSort::SortPairs(sort_tmp, tb, keys_a, keys_b, vals_a, vals_b, n, 0, end_bit);
        gather_edges<<<blocks, threads>>>(vals_b, srcs, dsts, weights, tmp_s, tmp_d, tmp_w, n);
        cudaMemcpyAsync(srcs, tmp_s, (size_t)n * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(dsts, tmp_d, (size_t)n * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(weights, tmp_w, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice);
    }
}

}  

extract_ego_weighted_result_double_t extract_ego_weighted_f64(
    const graph32_t& graph,
    const double* edge_weights,
    const int32_t* source_vertices,
    std::size_t n_sources,
    int32_t radius) {

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_csr_offsets = graph.offsets;
    const int32_t* d_csr_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    bool is_multigraph = graph.is_multigraph;

    int vertex_bits = compute_vertex_bits(num_vertices);
    int sort_end_bit = 32 + vertex_bits;
    if (sort_end_bit > 64) sort_end_bit = 64;

    std::vector<int32_t> h_sources(n_sources);
    cudaMemcpy(h_sources.data(), source_vertices,
               n_sources * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cache.ensure_scratch(num_vertices);

    struct PerSourceEdges {
        int32_t* srcs = nullptr;
        int32_t* dsts = nullptr;
        double* weights = nullptr;
        int64_t count = 0;
    };
    std::vector<PerSourceEdges> per_source(n_sources);
    std::vector<std::size_t> h_ego_offsets(n_sources + 1, 0);

    for (std::size_t s = 0; s < n_sources; s++) {
        int32_t source = h_sources[s];

        cudaMemsetAsync(cache.bitmap, 0, cache.bitmap_words * sizeof(uint32_t));
        cudaMemsetAsync(cache.fc, 0, sizeof(int32_t));

        launch_init_source(cache.bitmap, cache.fa, source);
        cudaMemcpyAsync(cache.ego, cache.fa, sizeof(int32_t), cudaMemcpyDeviceToDevice);
        int32_t ego_size = 1;

        int32_t* cur = cache.fa;
        int32_t* nxt = cache.fb;
        int32_t cur_size = 1;

        for (int32_t hop = 0; hop < radius; hop++) {
            cudaMemsetAsync(cache.fc, 0, sizeof(int32_t));
            launch_bfs_expand(d_csr_offsets, d_csr_indices, cur, cur_size,
                              cache.bitmap, nxt, cache.fc);

            cudaMemcpyAsync(cache.h_pinned, cache.fc, sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(0);
            int32_t next_size = cache.h_pinned[0];

            if (next_size > 0) {
                cudaMemcpyAsync(cache.ego + ego_size, nxt,
                                next_size * sizeof(int32_t), cudaMemcpyDeviceToDevice);
                ego_size += next_size;
            }
            std::swap(cur, nxt);
            cur_size = next_size;
            if (cur_size == 0) break;
        }

        if (ego_size == 0) {
            h_ego_offsets[s + 1] = h_ego_offsets[s];
            continue;
        }

        
        int32_t* ego_sorted = cache.ego;
        if (ego_size > 2) {
            cudaMemcpyAsync(cache.ego_alt, cache.ego,
                            ego_size * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            launch_sort_int32(cache.ego_alt, cache.ego, ego_size,
                              cache.ego_sort_tmp, cache.ego_sort_tb, vertex_bits);
            ego_sorted = cache.ego_alt;
        }

        
        launch_count_ego_edges(d_csr_offsets, d_csr_indices,
                               ego_sorted, ego_size, cache.bitmap, cache.cnt);
        {
            size_t tb = cache.scan_tb;
            cub::DeviceScan::ExclusiveSum(cache.scan_tmp, tb,
                                          cache.cnt, cache.ps, ego_size);
        }

        int64_t* h64 = reinterpret_cast<int64_t*>(cache.h_pinned);
        cudaMemcpyAsync(&h64[0], cache.ps + ego_size - 1,
                        sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(&h64[1], cache.cnt + ego_size - 1,
                        sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(0);
        int64_t total_edges = h64[0] + h64[1];

        if (total_edges == 0) {
            h_ego_offsets[s + 1] = h_ego_offsets[s];
            continue;
        }

        int32_t* es = nullptr;
        int32_t* ed = nullptr;
        double* ew = nullptr;
        cudaMalloc(&es, total_edges * sizeof(int32_t));
        cudaMalloc(&ed, total_edges * sizeof(int32_t));
        cudaMalloc(&ew, total_edges * sizeof(double));

        launch_fill_ego_edges(d_csr_offsets, d_csr_indices, edge_weights,
                              ego_sorted, ego_size, cache.bitmap, cache.ps, 0,
                              es, ed, ew);

        
        bool need_sort = false;
        if (total_edges > 1) {
            cudaMemsetAsync(cache.fc, 0, sizeof(int32_t));
            launch_check_sorted(es, ed, ew, total_edges, cache.fc);
            cudaMemcpyAsync(cache.h_pinned, cache.fc,
                            sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(0);
            need_sort = (cache.h_pinned[0] != 0);
        }

        if (need_sort) {
            cache.ensure_sort_scratch(total_edges);
            launch_sort_edges(
                es, ed, ew,
                (int32_t)total_edges,
                cache.s_ka, cache.s_kb,
                cache.s_va, cache.s_vb,
                cache.s_ts, cache.s_td, cache.s_tw,
                cache.s_tmp, cache.sort_tb,
                is_multigraph, sort_end_bit
            );
        }

        per_source[s] = {es, ed, ew, total_edges};
        h_ego_offsets[s + 1] = h_ego_offsets[s] + (std::size_t)total_edges;
    }

    std::size_t total = h_ego_offsets[n_sources];

    
    std::size_t* out_offsets = nullptr;
    cudaMalloc(&out_offsets, (n_sources + 1) * sizeof(std::size_t));
    cudaMemcpyAsync(out_offsets, h_ego_offsets.data(),
                    (n_sources + 1) * sizeof(std::size_t), cudaMemcpyHostToDevice);

    
    if (n_sources == 1) {
        return {per_source[0].srcs, per_source[0].dsts, per_source[0].weights,
                out_offsets, total, n_sources + 1};
    }

    if (total == 0) {
        return {nullptr, nullptr, nullptr, out_offsets, 0, n_sources + 1};
    }

    
    int32_t* out_srcs = nullptr;
    int32_t* out_dsts = nullptr;
    double* out_weights = nullptr;
    cudaMalloc(&out_srcs, total * sizeof(int32_t));
    cudaMalloc(&out_dsts, total * sizeof(int32_t));
    cudaMalloc(&out_weights, total * sizeof(double));

    
    std::size_t pos = 0;
    for (std::size_t i = 0; i < n_sources; i++) {
        int64_t n = per_source[i].count;
        if (n > 0) {
            cudaMemcpyAsync(out_srcs + pos, per_source[i].srcs,
                            n * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(out_dsts + pos, per_source[i].dsts,
                            n * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(out_weights + pos, per_source[i].weights,
                            n * sizeof(double), cudaMemcpyDeviceToDevice);
        }
        pos += (std::size_t)n;
    }

    
    for (std::size_t i = 0; i < n_sources; i++) {
        if (per_source[i].srcs) cudaFree(per_source[i].srcs);
        if (per_source[i].dsts) cudaFree(per_source[i].dsts);
        if (per_source[i].weights) cudaFree(per_source[i].weights);
    }

    return {out_srcs, out_dsts, out_weights, out_offsets, total, n_sources + 1};
}

}  
