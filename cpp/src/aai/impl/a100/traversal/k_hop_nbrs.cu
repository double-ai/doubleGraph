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

namespace aai {

namespace {

struct Cache : Cacheable {
    void* cub_temp = nullptr;
    size_t cub_temp_bytes = 0;

    Cache() {
        cub_temp_bytes = 1 << 24;  
        cudaMalloc(&cub_temp, cub_temp_bytes);
    }

    ~Cache() override {
        if (cub_temp) cudaFree(cub_temp);
    }

    void ensure_temp(size_t needed) {
        if (needed > cub_temp_bytes) {
            if (cub_temp) cudaFree(cub_temp);
            cub_temp_bytes = needed * 2;
            cudaMalloc(&cub_temp, cub_temp_bytes);
        }
    }
};



__global__ void compute_degrees_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ frontier,
    int64_t* __restrict__ degrees,
    int64_t n)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int32_t v = frontier[i];
        degrees[i] = (int64_t)(csr_offsets[v + 1] - csr_offsets[v]);
    }
}

__global__ void expand_warp_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const int32_t* __restrict__ frontier,
    const int64_t* __restrict__ prefix,
    int32_t* __restrict__ output,
    int64_t frontier_size)
{
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t v = frontier[warp_id];
    int32_t start = csr_offsets[v];
    int32_t end = csr_offsets[v + 1];
    int32_t degree = end - start;
    int64_t write_pos = prefix[warp_id];

    for (int32_t i = lane; i < degree; i += 32) {
        output[write_pos + i] = csr_indices[start + i];
    }
}

__global__ void expand_bsearch_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const int32_t* __restrict__ frontier,
    const int64_t* __restrict__ prefix,
    int32_t* __restrict__ output,
    int frontier_size,
    int64_t total_outputs)
{
    int64_t out_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_outputs) return;

    int lo = 0, hi = frontier_size;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (prefix[mid] <= out_idx) lo = mid;
        else hi = mid - 1;
    }

    int32_t v = frontier[lo];
    int32_t csr_start = csr_offsets[v];
    int32_t local_idx = (int32_t)(out_idx - prefix[lo]);
    output[out_idx] = csr_indices[csr_start + local_idx];
}

__global__ void extract_seg_offsets_kernel(
    const int64_t* __restrict__ prefix,
    const int64_t* __restrict__ old_seg,
    int64_t* __restrict__ new_seg,
    int64_t n)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        new_seg[i] = prefix[old_seg[i]];
    }
}

__global__ void iota_kernel(int64_t* __restrict__ out, int64_t n)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}



__global__ void mark_unique_kernel(
    const int32_t* __restrict__ sorted,
    const int64_t* __restrict__ seg_offsets,
    int32_t* __restrict__ marks,
    int64_t total,
    int num_segments)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    if (i == 0) { marks[i] = 1; return; }

    int lo = 0, hi = num_segments;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (seg_offsets[mid] <= i) lo = mid;
        else hi = mid - 1;
    }

    marks[i] = (seg_offsets[lo] == i || sorted[i] != sorted[i - 1]) ? 1 : 0;
}

__global__ void scatter_unique_kernel(
    const int32_t* __restrict__ sorted,
    const int32_t* __restrict__ marks,
    const int64_t* __restrict__ compact_pos,
    int32_t* __restrict__ compacted,
    int64_t total)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    if (marks[i]) {
        compacted[compact_pos[i]] = sorted[i];
    }
}



struct HopResult {
    int32_t* frontier;
    int64_t size;
    int64_t* seg_offsets;
};

HopResult do_hop(Cache& cache,
                 const int32_t* d_csr_offsets, const int32_t* d_csr_indices,
                 const int32_t* frontier, int64_t frontier_size,
                 const int64_t* d_seg, int64_t num_start,
                 bool need_dedup, cudaStream_t stream)
{
    
    int64_t* d_deg;
    cudaMalloc(&d_deg, (frontier_size + 1) * sizeof(int64_t));
    cudaMemsetAsync(d_deg + frontier_size, 0, sizeof(int64_t), stream);
    if (frontier_size > 0) {
        compute_degrees_kernel<<<(int)((frontier_size + 255) / 256), 256, 0, stream>>>(
            d_csr_offsets, frontier, d_deg, frontier_size);
    }

    
    int psz = (int)(frontier_size + 1);
    {
        size_t needed = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, needed, (int64_t*)nullptr, (int64_t*)nullptr, psz);
        cache.ensure_temp(needed);
    }
    int64_t* d_prefix;
    cudaMalloc(&d_prefix, (frontier_size + 1) * sizeof(int64_t));
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, cache.cub_temp_bytes, d_deg, d_prefix, psz, stream);

    cudaFree(d_deg);

    
    int64_t raw_total;
    cudaMemcpy(&raw_total, d_prefix + frontier_size, sizeof(int64_t), cudaMemcpyDeviceToHost);

    
    int64_t* d_raw_seg;
    cudaMalloc(&d_raw_seg, (num_start + 1) * sizeof(int64_t));
    extract_seg_offsets_kernel<<<(int)((num_start + 1 + 255) / 256), 256, 0, stream>>>(
        d_prefix, d_seg, d_raw_seg, num_start + 1);

    if (raw_total == 0) {
        cudaFree(d_prefix);
        return {nullptr, 0, d_raw_seg};
    }

    
    int32_t* d_raw;
    cudaMalloc(&d_raw, raw_total * sizeof(int32_t));
    {
        int threads = 256;
        int64_t total_threads = frontier_size * 32;
        int grid = (int)((total_threads + threads - 1) / threads);
        expand_warp_kernel<<<grid, threads, 0, stream>>>(
            d_csr_offsets, d_csr_indices, frontier, d_prefix, d_raw, frontier_size);
    }

    cudaFree(d_prefix);

    if (!need_dedup) {
        return {d_raw, raw_total, d_raw_seg};
    }

    
    int rt = (int)raw_total;
    int ns = (int)num_start;

    
    {
        size_t needed = 0;
        cub::DeviceSegmentedSort::SortKeys(nullptr, needed, (int32_t*)nullptr, (int32_t*)nullptr,
            rt, ns, (int64_t*)nullptr, (int64_t*)nullptr);
        cache.ensure_temp(needed);
    }
    int32_t* d_sorted;
    cudaMalloc(&d_sorted, raw_total * sizeof(int32_t));
    cub::DeviceSegmentedSort::SortKeys(cache.cub_temp, cache.cub_temp_bytes,
        d_raw, d_sorted, rt, ns, d_raw_seg, d_raw_seg + 1, stream);

    cudaFree(d_raw);

    
    int32_t* d_marks;
    cudaMalloc(&d_marks, (raw_total + 1) * sizeof(int32_t));
    cudaMemsetAsync(d_marks + raw_total, 0, sizeof(int32_t), stream);
    mark_unique_kernel<<<(int)((raw_total + 255) / 256), 256, 0, stream>>>(
        d_sorted, d_raw_seg, d_marks, raw_total, ns);

    
    {
        size_t needed = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, needed, (int32_t*)nullptr, (int64_t*)nullptr, rt + 1);
        cache.ensure_temp(needed);
    }
    int64_t* d_cpos;
    cudaMalloc(&d_cpos, (raw_total + 1) * sizeof(int64_t));
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, cache.cub_temp_bytes, d_marks, d_cpos, rt + 1, stream);

    
    int64_t total_unique;
    cudaMemcpy(&total_unique, d_cpos + raw_total, sizeof(int64_t), cudaMemcpyDeviceToHost);

    
    int32_t* d_comp = nullptr;
    if (total_unique > 0) {
        cudaMalloc(&d_comp, total_unique * sizeof(int32_t));
        scatter_unique_kernel<<<(int)((raw_total + 255) / 256), 256, 0, stream>>>(
            d_sorted, d_marks, d_cpos, d_comp, raw_total);
    }

    cudaFree(d_sorted);
    cudaFree(d_marks);

    
    int64_t* d_dedup_seg;
    cudaMalloc(&d_dedup_seg, (num_start + 1) * sizeof(int64_t));
    extract_seg_offsets_kernel<<<(int)((num_start + 1 + 255) / 256), 256, 0, stream>>>(
        d_cpos, d_raw_seg, d_dedup_seg, num_start + 1);

    cudaFree(d_cpos);
    cudaFree(d_raw_seg);

    return {d_comp, total_unique, d_dedup_seg};
}

}  

k_hop_nbrs_result_t k_hop_nbrs(const graph32_t& graph,
                               const int32_t* start_vertices,
                               std::size_t num_start_vertices,
                               std::size_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_csr_offsets = graph.offsets;
    const int32_t* d_csr_indices = graph.indices;
    bool is_multigraph = graph.is_multigraph;
    int64_t num_start = (int64_t)num_start_vertices;
    cudaStream_t stream = 0;

    
    int64_t* seg_owned;
    cudaMalloc(&seg_owned, (num_start + 1) * sizeof(int64_t));
    iota_kernel<<<(int)((num_start + 1 + 255) / 256), 256, 0, stream>>>(seg_owned, num_start + 1);

    const int32_t* frontier = start_vertices;
    int64_t frontier_size = num_start;
    int32_t* frontier_owned = nullptr;

    for (int64_t hop = 0; hop < (int64_t)k; hop++) {
        bool need_dedup = is_multigraph || (k > 1);

        HopResult result = do_hop(cache, d_csr_offsets, d_csr_indices,
                                  frontier, frontier_size, seg_owned, num_start,
                                  need_dedup, stream);

        
        if (frontier_owned) cudaFree(frontier_owned);
        cudaFree(seg_owned);

        if (hop == (int64_t)k - 1) {
            
            k_hop_nbrs_result_t out;
            out.offsets = reinterpret_cast<std::size_t*>(result.seg_offsets);
            out.neighbors = result.frontier;
            out.num_offsets = (std::size_t)(num_start + 1);
            out.num_neighbors = (std::size_t)result.size;
            return out;
        }

        
        frontier_owned = result.frontier;
        frontier = (result.size > 0) ? frontier_owned : nullptr;
        frontier_size = result.size;
        seg_owned = result.seg_offsets;
    }

    
    k_hop_nbrs_result_t out;
    out.offsets = nullptr;
    out.neighbors = nullptr;
    out.num_offsets = 0;
    out.num_neighbors = 0;
    return out;
}

}  
