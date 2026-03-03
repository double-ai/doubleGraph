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
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    uint32_t* d_bitmaps = nullptr;
    size_t bitmap_alloc_size = 0;

    void ensure_bitmap(int64_t num_start_vertices, int64_t bitmap_words) {
        size_t needed = (size_t)num_start_vertices * bitmap_words * sizeof(uint32_t);
        if (needed > bitmap_alloc_size) {
            if (d_bitmaps) cudaFree(d_bitmaps);
            cudaMalloc(&d_bitmaps, needed);
            bitmap_alloc_size = needed;
        }
    }

    ~Cache() override {
        if (d_bitmaps) { cudaFree(d_bitmaps); d_bitmaps = nullptr; }
    }
};





__global__ void compute_degrees_kernel(
    const int32_t* __restrict__ graph_offsets,
    const int32_t* __restrict__ frontier,
    int64_t* __restrict__ degrees,
    int64_t frontier_size
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < frontier_size) {
        int32_t v = frontier[idx];
        degrees[idx] = (int64_t)(graph_offsets[v + 1] - graph_offsets[v]);
    }
}

__global__ void expand_frontier_kernel(
    const int32_t* __restrict__ graph_offsets,
    const int32_t* __restrict__ graph_indices,
    const int32_t* __restrict__ frontier,
    const int64_t* __restrict__ element_offsets,
    int32_t* __restrict__ new_frontier,
    int64_t frontier_size,
    int64_t total_output
) {
    int64_t out_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= total_output) return;
    int64_t lo = 0, hi = frontier_size - 1;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo + 1) / 2;
        if (element_offsets[mid] <= out_idx) lo = mid;
        else hi = mid - 1;
    }
    int32_t v = frontier[lo];
    int32_t pos = (int32_t)(out_idx - element_offsets[lo]);
    new_frontier[out_idx] = graph_indices[graph_offsets[v] + pos];
}

__global__ void update_seg_offsets_kernel(
    const int64_t* __restrict__ src_offsets,
    const int64_t* __restrict__ old_seg_offsets,
    int64_t* __restrict__ new_seg_offsets,
    int64_t num_start_vertices
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= num_start_vertices)
        new_seg_offsets[idx] = src_offsets[old_seg_offsets[idx]];
}

__global__ void init_sequence_kernel(int64_t* data, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= n) data[idx] = idx;
}


__global__ void set_segment_starts_kernel(const int64_t* __restrict__ seg_offsets,
    int32_t* __restrict__ is_start, int64_t num_segments, int64_t total) {
    int64_t s = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (s < num_segments && seg_offsets[s] < total) is_start[seg_offsets[s]] = 1;
}

__global__ void mark_unique_kernel(const int32_t* __restrict__ sorted,
    const int32_t* __restrict__ is_start, int64_t* __restrict__ is_unique, int64_t total) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    is_unique[idx] = (is_start[idx] || (idx > 0 && sorted[idx] != sorted[idx - 1])) ? 1 : 0;
}

__global__ void compact_kernel(const int32_t* __restrict__ sorted,
    const int64_t* __restrict__ is_unique, const int64_t* __restrict__ prefix,
    int32_t* __restrict__ compacted, int64_t total) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    if (is_unique[idx]) compacted[prefix[idx]] = sorted[idx];
}




constexpr int BM_BLOCK = 256;

__global__ void bitmap_hop_smem_kernel(
    const int32_t* __restrict__ graph_offsets,
    const int32_t* __restrict__ graph_indices,
    const int32_t* __restrict__ frontier,
    const int64_t* __restrict__ seg_offsets,
    uint32_t* __restrict__ global_bitmaps,
    int64_t* __restrict__ counts,
    int64_t num_start_vertices,
    int64_t bitmap_words
) {
    extern __shared__ uint32_t s_bitmap[];

    int64_t sv = blockIdx.x;
    if (sv >= num_start_vertices) return;

    for (int64_t w = threadIdx.x; w < bitmap_words; w += BM_BLOCK) {
        s_bitmap[w] = 0;
    }
    __syncthreads();

    int64_t seg_start = seg_offsets[sv];
    int64_t seg_end = seg_offsets[sv + 1];

    for (int64_t i = seg_start + threadIdx.x; i < seg_end; i += BM_BLOCK) {
        int32_t v = frontier[i];
        int32_t adj_start = graph_offsets[v];
        int32_t adj_end = graph_offsets[v + 1];

        for (int32_t j = adj_start; j < adj_end; j++) {
            int32_t nbr = graph_indices[j];
            atomicOr(&s_bitmap[nbr >> 5], 1U << (nbr & 31));
        }
    }
    __syncthreads();

    uint32_t* g_bitmap = global_bitmaps + sv * bitmap_words;
    int my_count = 0;
    for (int64_t w = threadIdx.x; w < bitmap_words; w += BM_BLOCK) {
        uint32_t word = s_bitmap[w];
        g_bitmap[w] = word;
        my_count += __popc(word);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        my_count += __shfl_down_sync(0xffffffff, my_count, offset);

    __shared__ int warp_sums[BM_BLOCK / 32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;
    if (lane_id == 0) warp_sums[warp_id] = my_count;
    __syncthreads();

    if (threadIdx.x < (BM_BLOCK / 32)) {
        int val = warp_sums[threadIdx.x];
        #pragma unroll
        for (int offset = (BM_BLOCK / 64); offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x == 0) counts[sv] = val;
    }
}

__global__ void bitmap_hop_global_kernel(
    const int32_t* __restrict__ graph_offsets,
    const int32_t* __restrict__ graph_indices,
    const int32_t* __restrict__ frontier,
    const int64_t* __restrict__ seg_offsets,
    uint32_t* __restrict__ bitmaps,
    int64_t* __restrict__ counts,
    int64_t num_start_vertices,
    int64_t bitmap_words
) {
    int64_t sv = blockIdx.x;
    if (sv >= num_start_vertices) return;
    uint32_t* my_bitmap = bitmaps + sv * bitmap_words;

    for (int64_t w = threadIdx.x; w < bitmap_words; w += BM_BLOCK)
        my_bitmap[w] = 0;
    __syncthreads();

    int64_t seg_start = seg_offsets[sv];
    int64_t seg_end = seg_offsets[sv + 1];
    for (int64_t i = seg_start + threadIdx.x; i < seg_end; i += BM_BLOCK) {
        int32_t v = frontier[i];
        int32_t adj_start = graph_offsets[v];
        int32_t adj_end = graph_offsets[v + 1];
        for (int32_t j = adj_start; j < adj_end; j++) {
            int32_t nbr = graph_indices[j];
            atomicOr(&my_bitmap[nbr >> 5], 1U << (nbr & 31));
        }
    }
    __syncthreads();

    int my_count = 0;
    for (int64_t w = threadIdx.x; w < bitmap_words; w += BM_BLOCK)
        my_count += __popc(my_bitmap[w]);

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        my_count += __shfl_down_sync(0xffffffff, my_count, offset);

    __shared__ int warp_sums[BM_BLOCK / 32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x & 31;
    if (lane_id == 0) warp_sums[warp_id] = my_count;
    __syncthreads();

    if (threadIdx.x < (BM_BLOCK / 32)) {
        int val = warp_sums[threadIdx.x];
        #pragma unroll
        for (int offset = (BM_BLOCK / 64); offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x == 0) counts[sv] = val;
    }
}

__global__ void bitmap_extract_kernel(
    const uint32_t* __restrict__ bitmaps,
    const int64_t* __restrict__ new_seg_offsets,
    int32_t* __restrict__ new_frontier,
    int64_t num_start_vertices,
    int64_t bitmap_words
) {
    int64_t sv = blockIdx.x;
    if (sv >= num_start_vertices) return;

    const uint32_t* my_bitmap = bitmaps + sv * bitmap_words;
    int64_t out_base = new_seg_offsets[sv];

    int64_t words_per_thread = (bitmap_words + BM_BLOCK - 1) / BM_BLOCK;
    int64_t w_start = (int64_t)threadIdx.x * words_per_thread;
    int64_t w_end = w_start + words_per_thread;
    if (w_end > bitmap_words) w_end = bitmap_words;
    if (w_start > bitmap_words) w_start = bitmap_words;

    int my_count = 0;
    for (int64_t w = w_start; w < w_end; w++)
        my_count += __popc(my_bitmap[w]);

    typedef cub::BlockScan<int, BM_BLOCK> BlockScanT;
    __shared__ typename BlockScanT::TempStorage scan_temp;
    int prefix;
    BlockScanT(scan_temp).ExclusiveSum(my_count, prefix);

    int out_idx = prefix;
    for (int64_t w = w_start; w < w_end; w++) {
        uint32_t word = my_bitmap[w];
        while (word) {
            int bit = __ffs(word) - 1;
            new_frontier[out_base + out_idx] = (int32_t)(w * 32 + bit);
            out_idx++;
            word &= word - 1;
        }
    }
}






void do_expand(const int32_t* d_go, const int32_t* d_gi,
               int32_t*& frontier, int64_t& frontier_size, bool& frontier_owned,
               int64_t*& seg_offsets, int64_t num_sv) {
    
    int64_t* degrees = nullptr;
    cudaMalloc(&degrees, frontier_size * sizeof(int64_t));
    compute_degrees_kernel<<<(int)((frontier_size + 255) / 256), 256>>>(
        d_go, frontier, degrees, frontier_size);

    
    int64_t* eo = nullptr;
    cudaMalloc(&eo, (frontier_size + 1) * sizeof(int64_t));
    cudaMemset(eo, 0, sizeof(int64_t));
    {
        size_t tb = 0;
        cub::DeviceScan::InclusiveSum(nullptr, tb, (const int64_t*)nullptr, (int64_t*)nullptr, frontier_size);
        void* tmp = nullptr;
        cudaMalloc(&tmp, std::max(tb, (size_t)1));
        cub::DeviceScan::InclusiveSum(tmp, tb, degrees, eo + 1, frontier_size);
        cudaFree(tmp);
    }
    cudaFree(degrees);

    int64_t total;
    cudaMemcpy(&total, eo + frontier_size, sizeof(int64_t), cudaMemcpyDeviceToHost);

    
    int64_t* new_seg = nullptr;
    cudaMalloc(&new_seg, (num_sv + 1) * sizeof(int64_t));
    update_seg_offsets_kernel<<<(int)((num_sv + 1 + 255) / 256), 256>>>(
        eo, seg_offsets, new_seg, num_sv);

    
    int32_t* new_frontier = nullptr;
    if (total > 0) {
        cudaMalloc(&new_frontier, total * sizeof(int32_t));
        expand_frontier_kernel<<<(int)((total + 255) / 256), 256>>>(
            d_go, d_gi, frontier, eo, new_frontier, frontier_size, total);
    }

    cudaFree(eo);

    
    if (frontier_owned && frontier) cudaFree(frontier);
    cudaFree(seg_offsets);

    frontier = new_frontier;
    frontier_size = total;
    frontier_owned = true;
    seg_offsets = new_seg;
}


void do_sort_dedup(int32_t*& frontier, int64_t& frontier_size,
                   int64_t*& seg_offsets, int64_t num_sv) {
    if (frontier_size == 0) return;

    
    int32_t* sorted = nullptr;
    cudaMalloc(&sorted, frontier_size * sizeof(int32_t));
    {
        size_t tb = 0;
        cub::DeviceSegmentedSort::SortKeys(nullptr, tb,
            (const int32_t*)nullptr, (int32_t*)nullptr, frontier_size, (int)num_sv,
            (const int64_t*)nullptr, (const int64_t*)nullptr);
        void* tmp = nullptr;
        cudaMalloc(&tmp, std::max(tb, (size_t)1));
        cub::DeviceSegmentedSort::SortKeys(tmp, tb, frontier, sorted, frontier_size, (int)num_sv,
            seg_offsets, seg_offsets + 1);
        cudaFree(tmp);
    }

    
    int32_t* is_start = nullptr;
    cudaMalloc(&is_start, frontier_size * sizeof(int32_t));
    cudaMemset(is_start, 0, frontier_size * sizeof(int32_t));
    if (num_sv > 0) {
        set_segment_starts_kernel<<<(int)((num_sv + 255) / 256), 256>>>(
            seg_offsets, is_start, num_sv, frontier_size);
    }

    
    int64_t* is_unique = nullptr;
    cudaMalloc(&is_unique, frontier_size * sizeof(int64_t));
    mark_unique_kernel<<<(int)((frontier_size + 255) / 256), 256>>>(
        sorted, is_start, is_unique, frontier_size);

    cudaFree(is_start);

    
    int64_t* prefix = nullptr;
    cudaMalloc(&prefix, (frontier_size + 1) * sizeof(int64_t));
    cudaMemset(prefix, 0, sizeof(int64_t));
    {
        size_t tb = 0;
        cub::DeviceScan::InclusiveSum(nullptr, tb, (const int64_t*)nullptr, (int64_t*)nullptr, frontier_size);
        void* tmp = nullptr;
        cudaMalloc(&tmp, std::max(tb, (size_t)1));
        cub::DeviceScan::InclusiveSum(tmp, tb, is_unique, prefix + 1, frontier_size);
        cudaFree(tmp);
    }

    int64_t total_unique;
    cudaMemcpy(&total_unique, prefix + frontier_size, sizeof(int64_t), cudaMemcpyDeviceToHost);

    
    int32_t* compacted = nullptr;
    if (total_unique > 0) {
        cudaMalloc(&compacted, total_unique * sizeof(int32_t));
        compact_kernel<<<(int)((frontier_size + 255) / 256), 256>>>(
            sorted, is_unique, prefix, compacted, frontier_size);
    }

    cudaFree(sorted);
    cudaFree(is_unique);

    
    int64_t* new_seg = nullptr;
    cudaMalloc(&new_seg, (num_sv + 1) * sizeof(int64_t));
    update_seg_offsets_kernel<<<(int)((num_sv + 1 + 255) / 256), 256>>>(
        prefix, seg_offsets, new_seg, num_sv);

    cudaFree(prefix);

    
    if (frontier) cudaFree(frontier);
    cudaFree(seg_offsets);

    frontier = compacted;
    frontier_size = total_unique;
    seg_offsets = new_seg;
}


void do_bitmap_hop(const int32_t* d_go, const int32_t* d_gi,
                   int32_t*& frontier, int64_t& frontier_size,
                   int64_t*& seg_offsets, int64_t num_sv,
                   int64_t num_vertices, Cache& cache) {
    int64_t bitmap_words = (num_vertices + 31) / 32;
    cache.ensure_bitmap(num_sv, bitmap_words);

    
    int64_t* counts = nullptr;
    cudaMalloc(&counts, num_sv * sizeof(int64_t));

    
    if (num_sv > 0) {
        size_t smem_size = bitmap_words * sizeof(uint32_t);
        if (smem_size <= 98304) {
            cudaFuncSetAttribute(bitmap_hop_smem_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_size);
            bitmap_hop_smem_kernel<<<(int)num_sv, BM_BLOCK, smem_size>>>(
                d_go, d_gi, frontier, seg_offsets, cache.d_bitmaps,
                counts, num_sv, bitmap_words);
        } else {
            bitmap_hop_global_kernel<<<(int)num_sv, BM_BLOCK>>>(
                d_go, d_gi, frontier, seg_offsets, cache.d_bitmaps,
                counts, num_sv, bitmap_words);
        }
    }

    
    int64_t* new_seg = nullptr;
    cudaMalloc(&new_seg, (num_sv + 1) * sizeof(int64_t));
    cudaMemset(new_seg, 0, sizeof(int64_t));
    {
        size_t tb = 0;
        cub::DeviceScan::InclusiveSum(nullptr, tb, (const int64_t*)nullptr, (int64_t*)nullptr, num_sv);
        void* tmp = nullptr;
        cudaMalloc(&tmp, std::max(tb, (size_t)1));
        cub::DeviceScan::InclusiveSum(tmp, tb, counts, new_seg + 1, num_sv);
        cudaFree(tmp);
    }
    cudaFree(counts);

    
    int64_t total_unique;
    cudaMemcpy(&total_unique, new_seg + num_sv, sizeof(int64_t), cudaMemcpyDeviceToHost);

    
    int32_t* nf = nullptr;
    if (total_unique > 0) {
        cudaMalloc(&nf, total_unique * sizeof(int32_t));
        if (num_sv > 0) {
            bitmap_extract_kernel<<<(int)num_sv, BM_BLOCK>>>(
                cache.d_bitmaps, new_seg, nf, num_sv, bitmap_words);
        }
    }

    
    if (frontier) cudaFree(frontier);
    cudaFree(seg_offsets);

    frontier = nf;
    frontier_size = total_unique;
    seg_offsets = new_seg;
}

}  

k_hop_nbrs_result_t k_hop_nbrs(const graph32_t& graph,
                               const int32_t* start_vertices,
                               std::size_t num_start_vertices,
                               std::size_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_go = graph.offsets;
    const int32_t* d_gi = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    bool is_multigraph = graph.is_multigraph;
    int64_t num_sv = (int64_t)num_start_vertices;
    int64_t k_val = (int64_t)k;

    
    int32_t* frontier = const_cast<int32_t*>(start_vertices);
    int64_t frontier_size = num_sv;
    bool frontier_owned = false;

    
    int64_t* seg_offsets = nullptr;
    cudaMalloc(&seg_offsets, (num_sv + 1) * sizeof(int64_t));
    init_sequence_kernel<<<(int)((num_sv + 1 + 255) / 256), 256>>>(seg_offsets, num_sv);

    if (k_val == 1) {
        
        do_expand(d_go, d_gi, frontier, frontier_size, frontier_owned, seg_offsets, num_sv);
        
        if (is_multigraph) {
            do_sort_dedup(frontier, frontier_size, seg_offsets, num_sv);
        }
    } else {
        
        do_expand(d_go, d_gi, frontier, frontier_size, frontier_owned, seg_offsets, num_sv);
        if (is_multigraph && frontier_size > 0) {
            do_sort_dedup(frontier, frontier_size, seg_offsets, num_sv);
        }

        
        for (int64_t hop = 1; hop < k_val; hop++) {
            if (frontier_size == 0) {
                cudaMemset(seg_offsets, 0, (num_sv + 1) * sizeof(int64_t));
                break;
            }
            do_bitmap_hop(d_go, d_gi, frontier, frontier_size, seg_offsets,
                         num_sv, num_vertices, cache);
        }
    }

    
    
    k_hop_nbrs_result_t result;
    result.offsets = reinterpret_cast<std::size_t*>(seg_offsets);
    result.neighbors = frontier;
    result.num_offsets = (std::size_t)(num_sv + 1);
    result.num_neighbors = (std::size_t)frontier_size;

    return result;
}

}  
