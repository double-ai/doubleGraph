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
    void* d_temp = nullptr;
    size_t temp_capacity = 0;

    Cache() {
        temp_capacity = 128 * 1024 * 1024;
        cudaMalloc(&d_temp, temp_capacity);
    }

    ~Cache() override {
        if (d_temp) cudaFree(d_temp);
    }

    void ensure_temp(size_t needed) {
        if (needed <= temp_capacity) return;
        if (d_temp) cudaFree(d_temp);
        temp_capacity = std::max(needed, temp_capacity * 2);
        cudaMalloc(&d_temp, temp_capacity);
    }
};

static int compute_bits(int64_t n) {
    if (n <= 1) return 1;
    int bits = 0;
    int64_t x = n - 1;
    while (x > 0) { bits++; x >>= 1; }
    return bits;
}


__global__ void count_masked_nbrs_kernel(
    const int32_t* __restrict__ csr_offsets,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ verts,
    int64_t n,
    int64_t* __restrict__ counts)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int32_t v = verts[i];
    int32_t rs = __ldg(&csr_offsets[v]);
    int32_t re = __ldg(&csr_offsets[v + 1]);
    if (rs >= re) { counts[i] = 0; return; }
    int cnt = 0;
    int32_t fw = rs >> 5, lw = (re - 1) >> 5;
    if (fw == lw) {
        uint32_t m = __ldg(&edge_mask[fw]) >> (rs & 31);
        int bits = re - rs;
        if (bits < 32) m &= (1u << bits) - 1;
        cnt = __popc(m);
    } else {
        cnt = __popc(__ldg(&edge_mask[fw]) >> (rs & 31));
        for (int32_t w = fw + 1; w < lw; w++)
            cnt += __popc(__ldg(&edge_mask[w]));
        int hi = re & 31;
        cnt += (hi == 0) ? __popc(__ldg(&edge_mask[lw]))
                         : __popc(__ldg(&edge_mask[lw]) & ((1u << hi) - 1));
    }
    counts[i] = cnt;
}


__global__ void gather_k1_warp_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ start_verts,
    int64_t num_start,
    const int64_t* __restrict__ out_offsets,
    int32_t* __restrict__ out_nbrs)
{
    const int lane = threadIdx.x & 31;
    const int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp_id >= num_start) return;
    int32_t v = start_verts[warp_id];
    int32_t rs = __ldg(&csr_offsets[v]);
    int32_t re = __ldg(&csr_offsets[v + 1]);
    int64_t out_pos = out_offsets[warp_id];
    for (int32_t base = rs; base < re; base += 32) {
        int32_t e = base + lane;
        bool active = false;
        int32_t neighbor = 0;
        if (e < re) {
            active = (__ldg(&edge_mask[e >> 5]) >> (e & 31)) & 1;
            if (active) neighbor = __ldg(&csr_indices[e]);
        }
        uint32_t ballot = __ballot_sync(0xffffffff, active);
        int prefix = __popc(ballot & ((1u << lane) - 1));
        if (active) out_nbrs[out_pos + prefix] = neighbor;
        out_pos += __popc(ballot);
    }
}


__global__ void gather_tagged_warp_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ frontier_verts,
    const int32_t* __restrict__ frontier_tags,
    int64_t frontier_size,
    const int64_t* __restrict__ out_offsets,
    uint64_t* __restrict__ out_keys,
    int vert_bits)
{
    const int lane = threadIdx.x & 31;
    const int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp_id >= frontier_size) return;
    int32_t v = frontier_verts[warp_id];
    int32_t tag = frontier_tags[warp_id];
    int32_t rs = __ldg(&csr_offsets[v]);
    int32_t re = __ldg(&csr_offsets[v + 1]);
    int64_t out_pos = out_offsets[warp_id];
    uint64_t tag_shifted = (uint64_t)(uint32_t)tag << vert_bits;
    for (int32_t base = rs; base < re; base += 32) {
        int32_t e = base + lane;
        bool active = false;
        uint64_t key = 0;
        if (e < re) {
            active = (__ldg(&edge_mask[e >> 5]) >> (e & 31)) & 1;
            if (active) key = tag_shifted | (uint64_t)(uint32_t)__ldg(&csr_indices[e]);
        }
        uint32_t ballot = __ballot_sync(0xffffffff, active);
        int prefix = __popc(ballot & ((1u << lane) - 1));
        if (active) out_keys[out_pos + prefix] = key;
        out_pos += __popc(ballot);
    }
}


__global__ void extract_from_keys_kernel(
    const uint64_t* __restrict__ keys, int64_t n,
    int32_t* __restrict__ tags, int32_t* __restrict__ verts,
    int vert_bits)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t k = keys[i];
    uint64_t vert_mask = (1ULL << vert_bits) - 1;
    tags[i] = (int32_t)(k >> vert_bits);
    verts[i] = (int32_t)(k & vert_mask);
}


__global__ void build_offsets_kernel(
    const int32_t* __restrict__ tags, int64_t n_elems,
    int64_t num_start, int64_t* __restrict__ offsets)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i > num_start) return;
    int64_t lo = 0, hi = n_elems;
    int32_t target = (int32_t)i;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (tags[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    offsets[i] = lo;
}

__global__ void iota_kernel(int32_t* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (int32_t)i;
}


size_t get_inclusive_sum_temp_int64(int n) {
    size_t temp = 0;
    cub::DeviceScan::InclusiveSum((void*)nullptr, temp,
        (const int64_t*)nullptr, (int64_t*)nullptr, n);
    return temp;
}

void run_inclusive_sum_int64(
    void* d_temp, size_t temp_bytes,
    const int64_t* d_in, int64_t* d_out, int n, cudaStream_t stream)
{
    size_t tb = temp_bytes;
    cub::DeviceScan::InclusiveSum(d_temp, tb, d_in, d_out, n, stream);
}

size_t get_sort_temp_uint64(int n, int end_bit) {
    size_t temp = 0;
    cub::DeviceRadixSort::SortKeys((void*)nullptr, temp,
        (const uint64_t*)nullptr, (uint64_t*)nullptr, n, 0, end_bit);
    return temp;
}

void run_sort_uint64(
    void* d_temp, size_t temp_bytes,
    const uint64_t* d_in, uint64_t* d_out,
    int n, int begin_bit, int end_bit, cudaStream_t stream)
{
    size_t tb = temp_bytes;
    cub::DeviceRadixSort::SortKeys(d_temp, tb, d_in, d_out, n, begin_bit, end_bit, stream);
}

size_t get_unique_temp_uint64(int n) {
    size_t temp = 0;
    cub::DeviceSelect::Unique((void*)nullptr, temp,
        (const uint64_t*)nullptr, (uint64_t*)nullptr, (int*)nullptr, n);
    return temp;
}

void run_unique_uint64(
    void* d_temp, size_t temp_bytes,
    const uint64_t* d_in, uint64_t* d_out,
    int* d_num_selected, int n, cudaStream_t stream)
{
    size_t tb = temp_bytes;
    cub::DeviceSelect::Unique(d_temp, tb, d_in, d_out, d_num_selected, n, stream);
}


void launch_count_masked_nbrs(
    const int32_t* csr_offsets, const uint32_t* edge_mask,
    const int32_t* verts, int64_t n, int64_t* counts, cudaStream_t stream)
{
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    count_masked_nbrs_kernel<<<grid, block, 0, stream>>>(
        csr_offsets, edge_mask, verts, n, counts);
}

void launch_gather_k1_warp(
    const int32_t* csr_offsets, const int32_t* csr_indices, const uint32_t* edge_mask,
    const int32_t* start_verts, int64_t num_start,
    const int64_t* out_offsets, int32_t* out_nbrs, cudaStream_t stream)
{
    if (num_start == 0) return;
    int block = 256;
    int warps_per_block = block / 32;
    int grid = (int)((num_start + warps_per_block - 1) / warps_per_block);
    gather_k1_warp_kernel<<<grid, block, 0, stream>>>(
        csr_offsets, csr_indices, edge_mask, start_verts, num_start, out_offsets, out_nbrs);
}

void launch_gather_tagged_warp(
    const int32_t* csr_offsets, const int32_t* csr_indices, const uint32_t* edge_mask,
    const int32_t* frontier_verts, const int32_t* frontier_tags,
    int64_t frontier_size, const int64_t* out_offsets,
    uint64_t* out_keys, int vert_bits, cudaStream_t stream)
{
    if (frontier_size == 0) return;
    int block = 256;
    int warps_per_block = block / 32;
    int grid = (int)((frontier_size + warps_per_block - 1) / warps_per_block);
    gather_tagged_warp_kernel<<<grid, block, 0, stream>>>(
        csr_offsets, csr_indices, edge_mask, frontier_verts, frontier_tags,
        frontier_size, out_offsets, out_keys, vert_bits);
}

void launch_extract_from_keys(
    const uint64_t* keys, int64_t n, int32_t* tags, int32_t* verts,
    int vert_bits, cudaStream_t stream)
{
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    extract_from_keys_kernel<<<grid, block, 0, stream>>>(keys, n, tags, verts, vert_bits);
}

void launch_build_offsets(
    const int32_t* tags, int64_t n_elems, int64_t num_start,
    int64_t* offsets, cudaStream_t stream)
{
    int block = 256;
    int grid = (int)((num_start + 2 + block - 1) / block);
    build_offsets_kernel<<<grid, block, 0, stream>>>(tags, n_elems, num_start, offsets);
}

void launch_iota(int32_t* out, int64_t n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (int)((n + block - 1) / block);
    iota_kernel<<<grid, block, 0, stream>>>(out, n);
}


k_hop_nbrs_result_t k1_fast(
    Cache& cache,
    const int32_t* d_offsets, const int32_t* d_indices, const uint32_t* d_edge_mask,
    const int32_t* d_start, int64_t num_start)
{
    cudaStream_t stream = 0;

    int64_t* counts = nullptr;
    cudaMalloc(&counts, num_start * sizeof(int64_t));
    launch_count_masked_nbrs(d_offsets, d_edge_mask, d_start, num_start, counts, stream);

    int64_t* result_offsets = nullptr;
    cudaMalloc(&result_offsets, (num_start + 1) * sizeof(int64_t));
    cudaMemsetAsync(result_offsets, 0, sizeof(int64_t), stream);
    if (num_start > 0) {
        size_t scan_temp = get_inclusive_sum_temp_int64((int)num_start);
        cache.ensure_temp(scan_temp);
        run_inclusive_sum_int64(cache.d_temp, cache.temp_capacity,
                                counts, result_offsets + 1,
                                (int)num_start, stream);
    }

    int64_t total = 0;
    cudaMemcpyAsync(&total, result_offsets + num_start,
                     sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(counts);

    int32_t* result_nbrs = nullptr;
    if (total > 0) {
        cudaMalloc(&result_nbrs, total * sizeof(int32_t));
        launch_gather_k1_warp(d_offsets, d_indices, d_edge_mask, d_start, num_start,
                               result_offsets, result_nbrs, stream);
    }

    k_hop_nbrs_result_t result{};
    result.offsets = reinterpret_cast<std::size_t*>(result_offsets);
    result.neighbors = result_nbrs;
    result.num_offsets = (std::size_t)(num_start + 1);
    result.num_neighbors = (std::size_t)total;
    return result;
}


k_hop_nbrs_result_t k_general(
    Cache& cache,
    const int32_t* d_offsets, const int32_t* d_indices, const uint32_t* d_edge_mask,
    const int32_t* d_start, int64_t num_start, int64_t k_val,
    int32_t num_vertices)
{
    cudaStream_t stream = 0;

    int vert_bits = compute_bits((int64_t)num_vertices);
    int tag_bits = compute_bits(num_start);
    int total_sort_bits = vert_bits + tag_bits;

    int32_t* frontier_tags = nullptr;
    cudaMalloc(&frontier_tags, num_start * sizeof(int32_t));
    launch_iota(frontier_tags, num_start, stream);

    int32_t* frontier_verts_buf = nullptr;
    cudaMalloc(&frontier_verts_buf, num_start * sizeof(int32_t));
    cudaMemcpyAsync(frontier_verts_buf, d_start,
                     num_start * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

    int64_t frontier_size = num_start;

    for (int64_t hop = 0; hop < k_val; hop++) {
        if (frontier_size == 0) break;

        int64_t* counts = nullptr;
        cudaMalloc(&counts, frontier_size * sizeof(int64_t));
        launch_count_masked_nbrs(d_offsets, d_edge_mask,
                                  frontier_verts_buf, frontier_size,
                                  counts, stream);

        int64_t* gather_offsets = nullptr;
        cudaMalloc(&gather_offsets, (frontier_size + 1) * sizeof(int64_t));
        cudaMemsetAsync(gather_offsets, 0, sizeof(int64_t), stream);
        size_t scan_temp = get_inclusive_sum_temp_int64((int)frontier_size);
        cache.ensure_temp(scan_temp);
        run_inclusive_sum_int64(cache.d_temp, cache.temp_capacity,
                                counts, gather_offsets + 1,
                                (int)frontier_size, stream);

        int64_t expanded_size = 0;
        cudaMemcpyAsync(&expanded_size, gather_offsets + frontier_size,
                         sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cudaFree(counts);

        if (expanded_size == 0) {
            cudaFree(gather_offsets);
            frontier_size = 0;
            cudaFree(frontier_tags);
            cudaFree(frontier_verts_buf);
            frontier_tags = nullptr;
            frontier_verts_buf = nullptr;
            continue;
        }

        
        uint64_t* keys = nullptr;
        cudaMalloc(&keys, expanded_size * sizeof(uint64_t));
        launch_gather_tagged_warp(d_offsets, d_indices, d_edge_mask,
                                  frontier_verts_buf, frontier_tags,
                                  frontier_size, gather_offsets,
                                  keys, vert_bits, stream);

        cudaFree(gather_offsets);

        
        uint64_t* sorted_keys = nullptr;
        cudaMalloc(&sorted_keys, expanded_size * sizeof(uint64_t));
        size_t sort_temp = get_sort_temp_uint64((int)expanded_size, total_sort_bits);
        cache.ensure_temp(sort_temp);
        run_sort_uint64(cache.d_temp, cache.temp_capacity, keys, sorted_keys,
                         (int)expanded_size, 0, total_sort_bits, stream);

        cudaFree(keys);

        
        uint64_t* unique_keys = nullptr;
        cudaMalloc(&unique_keys, expanded_size * sizeof(uint64_t));
        int* d_num_unique = nullptr;
        cudaMalloc(&d_num_unique, sizeof(int));
        size_t unique_temp = get_unique_temp_uint64((int)expanded_size);
        cache.ensure_temp(unique_temp);
        run_unique_uint64(cache.d_temp, cache.temp_capacity, sorted_keys, unique_keys,
                           d_num_unique, (int)expanded_size, stream);

        cudaFree(sorted_keys);

        int num_unique_h = 0;
        cudaMemcpyAsync(&num_unique_h, d_num_unique, sizeof(int),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        cudaFree(d_num_unique);

        int64_t num_unique = (int64_t)num_unique_h;

        
        int32_t* old_tags = frontier_tags;
        int32_t* old_verts = frontier_verts_buf;
        frontier_tags = nullptr;
        frontier_verts_buf = nullptr;

        if (num_unique > 0) {
            cudaMalloc(&frontier_tags, num_unique * sizeof(int32_t));
            cudaMalloc(&frontier_verts_buf, num_unique * sizeof(int32_t));
            launch_extract_from_keys(unique_keys, num_unique,
                                      frontier_tags, frontier_verts_buf,
                                      vert_bits, stream);
        }

        cudaFree(unique_keys);
        cudaFree(old_tags);
        cudaFree(old_verts);

        frontier_size = num_unique;
    }

    int64_t* result_offsets = nullptr;
    cudaMalloc(&result_offsets, (num_start + 1) * sizeof(int64_t));

    if (frontier_size > 0) {
        launch_build_offsets(frontier_tags, frontier_size,
                              num_start, result_offsets, stream);
    } else {
        cudaMemsetAsync(result_offsets, 0,
                         (num_start + 1) * sizeof(int64_t), stream);
    }

    if (frontier_tags) cudaFree(frontier_tags);

    k_hop_nbrs_result_t result{};
    result.offsets = reinterpret_cast<std::size_t*>(result_offsets);
    result.neighbors = frontier_verts_buf;
    result.num_offsets = (std::size_t)(num_start + 1);
    result.num_neighbors = (std::size_t)frontier_size;
    return result;
}

}  

k_hop_nbrs_result_t k_hop_nbrs_seg_mask(const graph32_t& graph,
                                        const int32_t* start_vertices,
                                        std::size_t num_start_vertices,
                                        std::size_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;
    const int32_t* d_start = start_vertices;
    int32_t num_vertices = graph.number_of_vertices;
    bool is_multigraph = graph.is_multigraph;
    int64_t num_start = (int64_t)num_start_vertices;
    int64_t k_val = (int64_t)k;

    if (k_val == 1 && !is_multigraph) {
        return k1_fast(cache, d_offsets, d_indices, d_edge_mask, d_start, num_start);
    } else {
        return k_general(cache, d_offsets, d_indices, d_edge_mask,
                          d_start, num_start, k_val, num_vertices);
    }
}

}  
