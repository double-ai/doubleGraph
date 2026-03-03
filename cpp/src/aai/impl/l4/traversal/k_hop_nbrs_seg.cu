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

namespace aai {

namespace {

static_assert(sizeof(std::size_t) == sizeof(int64_t), "size_t must be 64-bit");

static inline __host__ int calc_grid(int64_t n, int block) {
    int64_t g = (n + block - 1) / block;
    return (int)(g > 65535 ? 65535 : g);
}





size_t get_scan_temp_size(int64_t n) {
    if (n <= 0) return 256;
    size_t temp_size = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_size, (int64_t*)nullptr, (int64_t*)nullptr, n);
    return temp_size;
}

size_t get_sort_temp_size(int64_t n) {
    if (n <= 0) return 256;
    size_t temp_size = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_size, (uint64_t*)nullptr, (uint64_t*)nullptr, n);
    return temp_size;
}

size_t get_unique_temp_size(int64_t n) {
    if (n <= 0) return 256;
    size_t temp_size = 0;
    cub::DeviceSelect::Unique(nullptr, temp_size, (uint64_t*)nullptr, (uint64_t*)nullptr, (int64_t*)nullptr, n);
    return temp_size;
}





struct Cache : Cacheable {
    void* scan_temp = nullptr;
    size_t scan_temp_size = 0;
    void* sort_temp = nullptr;
    size_t sort_temp_size = 0;
    void* unique_temp = nullptr;
    size_t unique_temp_size = 0;
    int64_t* d_num_selected = nullptr;

    Cache() {
        int64_t initial_size = 1 << 24;
        scan_temp_size = get_scan_temp_size(initial_size);
        sort_temp_size = get_sort_temp_size(initial_size);
        unique_temp_size = get_unique_temp_size(initial_size);
        cudaMalloc(&scan_temp, scan_temp_size);
        cudaMalloc(&sort_temp, sort_temp_size);
        cudaMalloc(&unique_temp, unique_temp_size);
        cudaMalloc(&d_num_selected, sizeof(int64_t));
    }

    ~Cache() override {
        if (scan_temp) cudaFree(scan_temp);
        if (sort_temp) cudaFree(sort_temp);
        if (unique_temp) cudaFree(unique_temp);
        if (d_num_selected) cudaFree(d_num_selected);
    }

    void ensure_scan(size_t needed) {
        if (needed > scan_temp_size) {
            if (scan_temp) cudaFree(scan_temp);
            scan_temp_size = needed;
            cudaMalloc(&scan_temp, scan_temp_size);
        }
    }

    void ensure_sort(size_t needed) {
        if (needed > sort_temp_size) {
            if (sort_temp) cudaFree(sort_temp);
            sort_temp_size = needed;
            cudaMalloc(&sort_temp, sort_temp_size);
        }
    }

    void ensure_unique(size_t needed) {
        if (needed > unique_temp_size) {
            if (unique_temp) cudaFree(unique_temp);
            unique_temp_size = needed;
            cudaMalloc(&unique_temp, unique_temp_size);
        }
    }
};





__global__ void compute_degrees_start_kernel(
    const int32_t* __restrict__ start_vertices,
    int64_t num_start,
    const int32_t* __restrict__ csr_offsets,
    int64_t* __restrict__ degrees)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_start; i += (int64_t)gridDim.x * blockDim.x) {
        int32_t v = start_vertices[i];
        degrees[i] = (int64_t)(csr_offsets[v + 1] - csr_offsets[v]);
    }
}


constexpr int EXPAND_BLK = 64;
constexpr int ITEMS_PER_THR = 4;
constexpr int OUTS_PER_BLK = EXPAND_BLK * ITEMS_PER_THR;

__global__ void expand_balanced_kernel(
    const int32_t* __restrict__ start_vertices,
    const int64_t* __restrict__ offsets,
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    int32_t* __restrict__ neighbors,
    int64_t num_start,
    int64_t total_output)
{
    int64_t block_out_start = (int64_t)blockIdx.x * OUTS_PER_BLK;
    if (block_out_start >= total_output) return;
    int64_t block_out_end = block_out_start + OUTS_PER_BLK;
    if (block_out_end > total_output) block_out_end = total_output;

    
    __shared__ int64_t s_seg_start, s_seg_end;
    if (threadIdx.x == 0) {
        
        int64_t lo = 0, hi = num_start;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo + 1) / 2;
            if (offsets[mid] <= block_out_start) lo = mid;
            else hi = mid - 1;
        }
        s_seg_start = lo;

        
        hi = num_start;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo + 1) / 2;
            if (offsets[mid] <= block_out_end - 1) lo = mid;
            else hi = mid - 1;
        }
        s_seg_end = lo + 1;
    }
    __syncthreads();

    int64_t seg_start = s_seg_start;
    int64_t seg_end = s_seg_end;
    int64_t num_segs = seg_end - seg_start;

    
    extern __shared__ char smem[];
    int64_t* s_offsets = (int64_t*)smem;
    int32_t* s_csr_start = (int32_t*)(s_offsets + num_segs + 1);

    for (int64_t i = threadIdx.x; i <= num_segs; i += blockDim.x) {
        int64_t si = seg_start + i;
        s_offsets[i] = offsets[si];
        if (i < num_segs) {
            int32_t v = start_vertices[si];
            s_csr_start[i] = csr_offsets[v];
        }
    }
    __syncthreads();

    
    int64_t prev_seg = 0;
    #pragma unroll
    for (int item = 0; item < ITEMS_PER_THR; item++) {
        int64_t out_idx = block_out_start + threadIdx.x + item * EXPAND_BLK;
        if (out_idx >= block_out_end) break;

        
        int64_t lo = prev_seg, hi = num_segs - 1;
        while (lo < hi) {
            int64_t mid = lo + (hi - lo + 1) / 2;
            if (s_offsets[mid] <= out_idx) lo = mid;
            else hi = mid - 1;
        }
        prev_seg = lo;

        int64_t local = out_idx - s_offsets[lo];
        neighbors[out_idx] = csr_indices[s_csr_start[lo] + local];
    }
}





__global__ void init_packed_kernel(
    const int32_t* __restrict__ start_vertices,
    uint64_t* __restrict__ packed,
    int64_t num_start)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_start; i += (int64_t)gridDim.x * blockDim.x) {
        uint64_t v = (uint64_t)(uint32_t)start_vertices[i];
        packed[i] = (v << 32) | (uint64_t)(uint32_t)i;
    }
}

__global__ void compute_degrees_packed_kernel(
    const uint64_t* __restrict__ packed,
    int64_t n,
    const int32_t* __restrict__ csr_offsets,
    int64_t* __restrict__ degrees)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)gridDim.x * blockDim.x) {
        int32_t v = (int32_t)(packed[i] >> 32);
        degrees[i] = (int64_t)(csr_offsets[v + 1] - csr_offsets[v]);
    }
}

__global__ void expand_packed_kernel(
    const uint64_t* __restrict__ packed,
    const int64_t* __restrict__ write_pos,
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    uint64_t* __restrict__ new_packed,
    int64_t n)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)gridDim.x * blockDim.x) {
        uint64_t p = packed[i];
        int32_t v = (int32_t)(p >> 32);
        uint32_t tag = (uint32_t)(p & 0xFFFFFFFFULL);
        int32_t start = csr_offsets[v];
        int32_t end = csr_offsets[v + 1];
        int64_t wp = write_pos[i];
        for (int32_t j = start; j < end; j++) {
            uint64_t nbr = (uint64_t)(uint32_t)csr_indices[j];
            new_packed[wp + (j - start)] = (nbr << 32) | (uint64_t)tag;
        }
    }
}

__global__ void repack_kernel(
    const uint64_t* __restrict__ packed,
    uint64_t* __restrict__ repacked,
    int64_t n)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)gridDim.x * blockDim.x) {
        uint64_t p = packed[i];
        uint32_t v = (uint32_t)(p >> 32);
        uint32_t tag = (uint32_t)(p & 0xFFFFFFFFULL);
        repacked[i] = ((uint64_t)tag << 32) | (uint64_t)v;
    }
}

__global__ void extract_offsets_kernel(
    const uint64_t* __restrict__ repacked,
    int64_t n,
    int64_t num_start,
    int64_t* __restrict__ offsets)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i <= num_start; i += (int64_t)gridDim.x * blockDim.x) {
        if (i == num_start) {
            offsets[i] = n;
        } else {
            uint64_t target = (uint64_t)(uint32_t)i << 32;
            int64_t lo = 0, hi = n;
            while (lo < hi) {
                int64_t mid = lo + (hi - lo) / 2;
                if (repacked[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            offsets[i] = lo;
        }
    }
}

__global__ void extract_neighbors_kernel(
    const uint64_t* __restrict__ repacked,
    int32_t* __restrict__ neighbors,
    int64_t n)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += (int64_t)gridDim.x * blockDim.x) {
        neighbors[i] = (int32_t)(uint32_t)(repacked[i] & 0xFFFFFFFFULL);
    }
}






__global__ void expand_start_to_bitmap_kernel(
    const int32_t* __restrict__ start_vertices,
    int64_t num_start,
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    uint32_t* __restrict__ bitmap,
    int32_t bitmap_words)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < num_start; i += (int64_t)gridDim.x * blockDim.x) {
        int32_t v = start_vertices[i];
        uint32_t tag = (uint32_t)i;
        int32_t start_e = csr_offsets[v];
        int32_t end_e = csr_offsets[v + 1];
        for (int32_t j = start_e; j < end_e; j++) {
            int32_t nbr = csr_indices[j];
            uint32_t word_idx = tag * (uint32_t)bitmap_words + ((uint32_t)nbr >> 5);
            uint32_t bit = 1u << ((uint32_t)nbr & 31);
            atomicOr(bitmap + word_idx, bit);
        }
    }
}


__global__ void expand_frontier_to_bitmap_kernel(
    const uint64_t* __restrict__ frontier,
    int64_t frontier_size,
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    uint32_t* __restrict__ bitmap,
    int32_t bitmap_words)
{
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < frontier_size; i += (int64_t)gridDim.x * blockDim.x) {
        uint64_t p = frontier[i];
        int32_t v = (int32_t)(p >> 32);
        uint32_t tag = (uint32_t)(p & 0xFFFFFFFFULL);
        int32_t start_e = csr_offsets[v];
        int32_t end_e = csr_offsets[v + 1];
        for (int32_t j = start_e; j < end_e; j++) {
            int32_t nbr = csr_indices[j];
            uint32_t word_idx = tag * (uint32_t)bitmap_words + ((uint32_t)nbr >> 5);
            uint32_t bit = 1u << ((uint32_t)nbr & 31);
            atomicOr(bitmap + word_idx, bit);
        }
    }
}


__global__ void count_bits_per_tag_kernel(
    const uint32_t* __restrict__ bitmap,
    int32_t bitmap_words,
    int64_t num_start,
    int64_t* __restrict__ counts)
{
    typedef cub::BlockReduce<int64_t, 64> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int64_t tag = blockIdx.x;
    if (tag >= num_start) return;

    int64_t count = 0;
    const uint32_t* tag_bitmap = bitmap + tag * bitmap_words;
    for (int w = threadIdx.x; w < bitmap_words; w += blockDim.x) {
        count += __popc(tag_bitmap[w]);
    }

    int64_t total = BlockReduce(temp_storage).Sum(count);
    if (threadIdx.x == 0) counts[tag] = total;
}


__global__ void compact_bitmap_to_frontier_kernel(
    const uint32_t* __restrict__ bitmap,
    int32_t bitmap_words,
    int64_t num_start,
    const int64_t* __restrict__ tag_offsets,
    uint64_t* __restrict__ frontier)
{
    int64_t tag = blockIdx.x;
    if (tag >= num_start) return;

    int64_t base = tag_offsets[tag];
    __shared__ int shared_idx;
    if (threadIdx.x == 0) shared_idx = 0;
    __syncthreads();

    const uint32_t* tag_bitmap = bitmap + tag * bitmap_words;
    for (int w = threadIdx.x; w < bitmap_words; w += blockDim.x) {
        uint32_t word = tag_bitmap[w];
        int nbits = __popc(word);
        if (nbits > 0) {
            int start_pos = atomicAdd(&shared_idx, nbits);
            int idx = 0;
            while (word) {
                int bit = __ffs(word) - 1;
                int32_t vertex = w * 32 + bit;
                frontier[base + start_pos + idx] = ((uint64_t)(uint32_t)vertex << 32) | (uint64_t)(uint32_t)tag;
                idx++;
                word &= word - 1;
            }
        }
    }
}


__global__ void bitmap_to_neighbors_kernel(
    const uint32_t* __restrict__ bitmap,
    int32_t bitmap_words,
    int64_t num_start,
    const int64_t* __restrict__ tag_offsets,
    int32_t* __restrict__ neighbors)
{
    int64_t tag = blockIdx.x;
    if (tag >= num_start) return;

    int64_t base = tag_offsets[tag];
    __shared__ int shared_idx;
    if (threadIdx.x == 0) shared_idx = 0;
    __syncthreads();

    const uint32_t* tag_bitmap = bitmap + tag * bitmap_words;
    for (int w = threadIdx.x; w < bitmap_words; w += blockDim.x) {
        uint32_t word = tag_bitmap[w];
        int nbits = __popc(word);
        if (nbits > 0) {
            int start_pos = atomicAdd(&shared_idx, nbits);
            int idx = 0;
            while (word) {
                int bit = __ffs(word) - 1;
                int32_t vertex = w * 32 + bit;
                neighbors[base + start_pos + idx] = vertex;
                idx++;
                word &= word - 1;
            }
        }
    }
}






void launch_compute_degrees_start(const int32_t* start_vertices, int64_t num_start,
                                  const int32_t* csr_offsets, int64_t* degrees, cudaStream_t s) {
    if (num_start == 0) return;
    compute_degrees_start_kernel<<<calc_grid(num_start, 64), 64, 0, s>>>(
        start_vertices, num_start, csr_offsets, degrees);
}

void launch_expand_balanced(const int32_t* start_vertices, const int64_t* offsets,
                            const int32_t* csr_offsets, const int32_t* csr_indices,
                            int32_t* neighbors, int64_t num_start, int64_t total_output,
                            cudaStream_t s) {
    if (total_output == 0) return;
    int64_t num_blocks = (total_output + OUTS_PER_BLK - 1) / OUTS_PER_BLK;
    if (num_blocks > 2147483647LL) num_blocks = 2147483647LL;
    
    
    size_t smem_bytes = (OUTS_PER_BLK + 1) * sizeof(int64_t) + OUTS_PER_BLK * sizeof(int32_t);
    expand_balanced_kernel<<<(int)num_blocks, EXPAND_BLK, smem_bytes, s>>>(
        start_vertices, offsets, csr_offsets, csr_indices, neighbors, num_start, total_output);
}


void launch_exclusive_scan(int64_t* d_in, int64_t* d_out, int64_t n, void* temp, size_t temp_size, cudaStream_t s) {
    if (n <= 0) return;
    cub::DeviceScan::ExclusiveSum(temp, temp_size, d_in, d_out, n, s);
}

void launch_inclusive_scan(int64_t* d_in, int64_t* d_out, int64_t n, void* temp, size_t temp_size, cudaStream_t s) {
    if (n <= 0) return;
    cub::DeviceScan::InclusiveSum(temp, temp_size, d_in, d_out, n, s);
}


void launch_init_packed(const int32_t* sv, uint64_t* p, int64_t n, cudaStream_t s) {
    if (n == 0) return;
    init_packed_kernel<<<calc_grid(n, 64), 64, 0, s>>>(sv, p, n);
}

void launch_compute_degrees_packed(const uint64_t* p, int64_t n, const int32_t* csr_offsets, int64_t* d, cudaStream_t s) {
    if (n == 0) return;
    compute_degrees_packed_kernel<<<calc_grid(n, 64), 64, 0, s>>>(p, n, csr_offsets, d);
}

void launch_expand_packed(const uint64_t* p, const int64_t* wp, const int32_t* co, const int32_t* ci, uint64_t* np, int64_t n, cudaStream_t s) {
    if (n == 0) return;
    expand_packed_kernel<<<calc_grid(n, 64), 64, 0, s>>>(p, wp, co, ci, np, n);
}

void launch_radix_sort(uint64_t* d_in, uint64_t* d_out, int64_t n, void* temp, size_t temp_size, cudaStream_t s) {
    if (n <= 0) return;
    cub::DeviceRadixSort::SortKeys(temp, temp_size, d_in, d_out, n, 0, 64, s);
}

void launch_unique(uint64_t* d_in, uint64_t* d_out, int64_t* d_num, int64_t n, void* temp, size_t temp_size, cudaStream_t s) {
    if (n <= 0) return;
    cub::DeviceSelect::Unique(temp, temp_size, d_in, d_out, d_num, n, s);
}

void launch_repack(const uint64_t* p, uint64_t* r, int64_t n, cudaStream_t s) {
    if (n == 0) return;
    repack_kernel<<<calc_grid(n, 64), 64, 0, s>>>(p, r, n);
}

void launch_extract_offsets(const uint64_t* r, int64_t n, int64_t ns, int64_t* o, cudaStream_t s) {
    extract_offsets_kernel<<<calc_grid(ns + 1, 64), 64, 0, s>>>(r, n, ns, o);
}

void launch_extract_neighbors(const uint64_t* r, int32_t* nb, int64_t n, cudaStream_t s) {
    if (n == 0) return;
    extract_neighbors_kernel<<<calc_grid(n, 64), 64, 0, s>>>(r, nb, n);
}


void launch_expand_start_to_bitmap(const int32_t* sv, int64_t ns, const int32_t* co, const int32_t* ci,
                                   uint32_t* bm, int32_t bw, cudaStream_t s) {
    if (ns == 0) return;
    expand_start_to_bitmap_kernel<<<calc_grid(ns, 64), 64, 0, s>>>(sv, ns, co, ci, bm, bw);
}

void launch_expand_frontier_to_bitmap(const uint64_t* f, int64_t fs, const int32_t* co, const int32_t* ci,
                                      uint32_t* bm, int32_t bw, cudaStream_t s) {
    if (fs == 0) return;
    expand_frontier_to_bitmap_kernel<<<calc_grid(fs, 64), 64, 0, s>>>(f, fs, co, ci, bm, bw);
}

void launch_count_bits(const uint32_t* bm, int32_t bw, int64_t ns, int64_t* counts, cudaStream_t s) {
    if (ns == 0) return;
    count_bits_per_tag_kernel<<<(int)ns, 64, 0, s>>>(bm, bw, ns, counts);
}

void launch_compact_bitmap(const uint32_t* bm, int32_t bw, int64_t ns, const int64_t* offsets,
                           uint64_t* frontier, cudaStream_t s) {
    if (ns == 0) return;
    compact_bitmap_to_frontier_kernel<<<(int)ns, 64, 0, s>>>(bm, bw, ns, offsets, frontier);
}

void launch_bitmap_to_neighbors(const uint32_t* bm, int32_t bw, int64_t ns, const int64_t* offsets,
                                int32_t* neighbors, cudaStream_t s) {
    if (ns == 0) return;
    bitmap_to_neighbors_kernel<<<(int)ns, 64, 0, s>>>(bm, bw, ns, offsets, neighbors);
}

}  

k_hop_nbrs_result_t k_hop_nbrs_seg(const graph32_t& graph,
                                   const int32_t* start_vertices,
                                   std::size_t num_start_vertices,
                                   std::size_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* csr_offsets = graph.offsets;
    const int32_t* csr_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    bool is_multigraph = graph.is_multigraph;
    int64_t num_start = (int64_t)num_start_vertices;
    int64_t k_val = (int64_t)k;
    cudaStream_t stream = 0;

    
    
    
    if (k_val == 1 && !is_multigraph) {
        int64_t* d_offsets = nullptr;
        cudaMalloc(&d_offsets, (num_start + 1) * sizeof(int64_t));

        
        launch_compute_degrees_start(start_vertices, num_start, csr_offsets,
                                     d_offsets + 1, stream);

        
        cache.ensure_scan(get_scan_temp_size(num_start));
        launch_inclusive_scan(d_offsets + 1, d_offsets + 1, num_start,
                             cache.scan_temp, cache.scan_temp_size, stream);

        
        cudaMemsetAsync(d_offsets, 0, sizeof(int64_t), stream);

        
        int64_t total;
        cudaMemcpyAsync(&total, d_offsets + num_start, sizeof(int64_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        
        int32_t* d_neighbors = nullptr;
        if (total > 0) {
            cudaMalloc(&d_neighbors, total * sizeof(int32_t));
            launch_expand_balanced(start_vertices, d_offsets,
                                   csr_offsets, csr_indices,
                                   d_neighbors, num_start, total, stream);
        }

        k_hop_nbrs_result_t result;
        result.offsets = reinterpret_cast<std::size_t*>(d_offsets);
        result.neighbors = d_neighbors;
        result.num_offsets = (std::size_t)(num_start + 1);
        result.num_neighbors = (std::size_t)total;
        return result;
    }

    
    
    
    int32_t bitmap_words = (num_vertices + 31) / 32;
    int64_t bitmap_bytes = (int64_t)num_start * bitmap_words * sizeof(uint32_t);

    
    
    
    if (k_val > 1 && bitmap_bytes <= 2LL * 1024 * 1024 * 1024) {
        int64_t bitmap_total_words = (int64_t)num_start * bitmap_words;

        
        uint32_t* bitmap_A = nullptr;
        uint32_t* bitmap_B = nullptr;
        cudaMalloc(&bitmap_A, bitmap_total_words * sizeof(uint32_t));
        cudaMalloc(&bitmap_B, bitmap_total_words * sizeof(uint32_t));

        
        int64_t* d_counts = nullptr;
        cudaMalloc(&d_counts, num_start * sizeof(int64_t));
        int64_t* d_tag_offsets = nullptr;
        cudaMalloc(&d_tag_offsets, (num_start + 1) * sizeof(int64_t));

        uint32_t* cur_bitmap = bitmap_A;
        uint32_t* next_bitmap = bitmap_B;

        
        cudaMemsetAsync(cur_bitmap, 0, bitmap_total_words * sizeof(uint32_t), stream);
        launch_expand_start_to_bitmap(start_vertices, num_start, csr_offsets, csr_indices,
                                      cur_bitmap, bitmap_words, stream);

        
        uint64_t* d_frontier = nullptr;
        int64_t frontier_size = 0;

        for (int64_t hop = 1; hop < k_val; hop++) {
            
            launch_count_bits(cur_bitmap, bitmap_words, num_start, d_counts, stream);

            cache.ensure_scan(get_scan_temp_size(num_start));
            launch_exclusive_scan(d_counts, d_tag_offsets, num_start,
                                  cache.scan_temp, cache.scan_temp_size, stream);

            
            int64_t last_off, last_cnt;
            cudaMemcpyAsync(&last_off, d_tag_offsets + num_start - 1,
                            sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(&last_cnt, d_counts + num_start - 1,
                            sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            frontier_size = last_off + last_cnt;

            if (frontier_size == 0) break;

            
            if (d_frontier) cudaFree(d_frontier);
            cudaMalloc(&d_frontier, frontier_size * sizeof(uint64_t));
            launch_compact_bitmap(cur_bitmap, bitmap_words, num_start,
                                  d_tag_offsets, d_frontier, stream);

            
            cudaMemsetAsync(next_bitmap, 0, bitmap_total_words * sizeof(uint32_t), stream);
            launch_expand_frontier_to_bitmap(d_frontier, frontier_size, csr_offsets, csr_indices,
                                             next_bitmap, bitmap_words, stream);

            
            uint32_t* tmp = cur_bitmap;
            cur_bitmap = next_bitmap;
            next_bitmap = tmp;
        }

        
        launch_count_bits(cur_bitmap, bitmap_words, num_start, d_counts, stream);

        
        int64_t* d_offsets_out = nullptr;
        cudaMalloc(&d_offsets_out, (num_start + 1) * sizeof(int64_t));
        cache.ensure_scan(get_scan_temp_size(num_start));
        launch_exclusive_scan(d_counts, d_offsets_out, num_start,
                              cache.scan_temp, cache.scan_temp_size, stream);

        
        int64_t last_off, last_cnt;
        cudaMemcpyAsync(&last_off, d_offsets_out + num_start - 1,
                        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&last_cnt, d_counts + num_start - 1,
                        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int64_t total_neighbors = last_off + last_cnt;

        
        cudaMemcpyAsync(d_offsets_out + num_start, &total_neighbors,
                        sizeof(int64_t), cudaMemcpyHostToDevice, stream);

        
        int32_t* d_neighbors_out = nullptr;
        if (total_neighbors > 0) {
            cudaMalloc(&d_neighbors_out, total_neighbors * sizeof(int32_t));
            launch_bitmap_to_neighbors(cur_bitmap, bitmap_words, num_start,
                                       d_offsets_out, d_neighbors_out, stream);
        }

        
        cudaFree(bitmap_A);
        cudaFree(bitmap_B);
        cudaFree(d_counts);
        cudaFree(d_tag_offsets);
        if (d_frontier) cudaFree(d_frontier);

        k_hop_nbrs_result_t result;
        result.offsets = reinterpret_cast<std::size_t*>(d_offsets_out);
        result.neighbors = d_neighbors_out;
        result.num_offsets = (std::size_t)(num_start + 1);
        result.num_neighbors = (std::size_t)total_neighbors;
        return result;
    }

    
    
    
    {
        int64_t frontier_size = num_start;

        uint64_t* d_packed = nullptr;
        cudaMalloc(&d_packed, frontier_size * sizeof(uint64_t));
        launch_init_packed(start_vertices, d_packed, frontier_size, stream);

        for (int64_t hop = 0; hop < k_val; hop++) {
            if (frontier_size == 0) break;

            
            int64_t* d_degrees = nullptr;
            cudaMalloc(&d_degrees, frontier_size * sizeof(int64_t));
            launch_compute_degrees_packed(d_packed, frontier_size, csr_offsets,
                                          d_degrees, stream);

            
            int64_t* d_write_pos = nullptr;
            cudaMalloc(&d_write_pos, frontier_size * sizeof(int64_t));
            cache.ensure_scan(get_scan_temp_size(frontier_size));
            launch_exclusive_scan(d_degrees, d_write_pos, frontier_size,
                                  cache.scan_temp, cache.scan_temp_size, stream);

            
            int64_t last_wp, last_deg;
            cudaMemcpyAsync(&last_wp, d_write_pos + frontier_size - 1,
                            sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(&last_deg, d_degrees + frontier_size - 1,
                            sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            int64_t total_expanded = last_wp + last_deg;

            if (total_expanded == 0) {
                cudaFree(d_degrees);
                cudaFree(d_write_pos);
                frontier_size = 0;
                break;
            }

            
            uint64_t* d_expanded = nullptr;
            cudaMalloc(&d_expanded, total_expanded * sizeof(uint64_t));
            launch_expand_packed(d_packed, d_write_pos, csr_offsets, csr_indices,
                                 d_expanded, frontier_size, stream);

            
            uint64_t* d_sorted = nullptr;
            cudaMalloc(&d_sorted, total_expanded * sizeof(uint64_t));
            cache.ensure_sort(get_sort_temp_size(total_expanded));
            launch_radix_sort(d_expanded, d_sorted, total_expanded,
                              cache.sort_temp, cache.sort_temp_size, stream);

            
            uint64_t* d_unique = nullptr;
            cudaMalloc(&d_unique, total_expanded * sizeof(uint64_t));
            cache.ensure_unique(get_unique_temp_size(total_expanded));
            launch_unique(d_sorted, d_unique, cache.d_num_selected, total_expanded,
                          cache.unique_temp, cache.unique_temp_size, stream);

            int64_t new_frontier_size;
            cudaMemcpy(&new_frontier_size, cache.d_num_selected,
                       sizeof(int64_t), cudaMemcpyDeviceToHost);

            
            cudaFree(d_degrees);
            cudaFree(d_write_pos);
            cudaFree(d_expanded);
            cudaFree(d_sorted);

            
            cudaFree(d_packed);
            d_packed = d_unique;
            frontier_size = new_frontier_size;
        }

        if (frontier_size == 0) {
            int64_t* d_offsets_out = nullptr;
            cudaMalloc(&d_offsets_out, (num_start + 1) * sizeof(int64_t));
            cudaMemsetAsync(d_offsets_out, 0, (num_start + 1) * sizeof(int64_t), stream);
            cudaFree(d_packed);

            k_hop_nbrs_result_t result;
            result.offsets = reinterpret_cast<std::size_t*>(d_offsets_out);
            result.neighbors = nullptr;
            result.num_offsets = (std::size_t)(num_start + 1);
            result.num_neighbors = 0;
            return result;
        }

        
        uint64_t* d_repacked = nullptr;
        cudaMalloc(&d_repacked, frontier_size * sizeof(uint64_t));
        launch_repack(d_packed, d_repacked, frontier_size, stream);

        
        uint64_t* d_sorted_final = nullptr;
        cudaMalloc(&d_sorted_final, frontier_size * sizeof(uint64_t));
        cache.ensure_sort(get_sort_temp_size(frontier_size));
        launch_radix_sort(d_repacked, d_sorted_final, frontier_size,
                          cache.sort_temp, cache.sort_temp_size, stream);

        
        int64_t* d_offsets_out = nullptr;
        cudaMalloc(&d_offsets_out, (num_start + 1) * sizeof(int64_t));
        launch_extract_offsets(d_sorted_final, frontier_size, num_start,
                               d_offsets_out, stream);

        
        int32_t* d_neighbors_out = nullptr;
        cudaMalloc(&d_neighbors_out, frontier_size * sizeof(int32_t));
        launch_extract_neighbors(d_sorted_final, d_neighbors_out, frontier_size, stream);

        
        cudaFree(d_packed);
        cudaFree(d_repacked);
        cudaFree(d_sorted_final);

        k_hop_nbrs_result_t result;
        result.offsets = reinterpret_cast<std::size_t*>(d_offsets_out);
        result.neighbors = d_neighbors_out;
        result.num_offsets = (std::size_t)(num_start + 1);
        result.num_neighbors = (std::size_t)frontier_size;
        return result;
    }
}

}  
