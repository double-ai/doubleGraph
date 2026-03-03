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





struct Cache : Cacheable {
    void* cub_temp = nullptr;
    size_t cub_temp_cap = 0;

    int64_t* d_num_selected = nullptr;

    uint8_t* bitmap_a = nullptr;
    size_t bitmap_a_cap = 0;

    uint8_t* bitmap_b = nullptr;
    size_t bitmap_b_cap = 0;

    Cache() {
        cudaMalloc(&d_num_selected, sizeof(int64_t));
    }

    ~Cache() override {
        if (cub_temp) cudaFree(cub_temp);
        if (d_num_selected) cudaFree(d_num_selected);
        if (bitmap_a) cudaFree(bitmap_a);
        if (bitmap_b) cudaFree(bitmap_b);
    }

    void ensure_cub_temp(size_t needed) {
        if (needed > cub_temp_cap) {
            if (cub_temp) cudaFree(cub_temp);
            cub_temp_cap = needed * 2;
            cudaMalloc(&cub_temp, cub_temp_cap);
        }
    }

    void ensure_bitmap_a(size_t needed) {
        if (needed > bitmap_a_cap) {
            if (bitmap_a) cudaFree(bitmap_a);
            bitmap_a_cap = needed;
            cudaMalloc(&bitmap_a, bitmap_a_cap);
        }
    }

    void ensure_bitmap_b(size_t needed) {
        if (needed > bitmap_b_cap) {
            if (bitmap_b) cudaFree(bitmap_b);
            bitmap_b_cap = needed;
            cudaMalloc(&bitmap_b, bitmap_b_cap);
        }
    }

    void inclusive_sum(const int64_t* d_in, int64_t* d_out, int64_t n) {
        size_t bytes = 0;
        cub::DeviceScan::InclusiveSum(nullptr, bytes, d_in, d_out, n);
        ensure_cub_temp(bytes);
        cub::DeviceScan::InclusiveSum(cub_temp, bytes, d_in, d_out, n);
    }

    void sort_keys(const int64_t* d_in, int64_t* d_out, int64_t n) {
        size_t bytes = 0;
        cub::DeviceRadixSort::SortKeys(nullptr, bytes, d_in, d_out, n);
        ensure_cub_temp(bytes);
        cub::DeviceRadixSort::SortKeys(cub_temp, bytes, d_in, d_out, n);
    }

    int64_t unique_keys(const int64_t* d_in, int64_t* d_out, int64_t n) {
        size_t bytes = 0;
        cub::DeviceSelect::Unique(nullptr, bytes, d_in, d_out, d_num_selected, n);
        ensure_cub_temp(bytes);
        cub::DeviceSelect::Unique(cub_temp, bytes, d_in, d_out, d_num_selected, n);
        int64_t result;
        cudaMemcpy(&result, d_num_selected, sizeof(int64_t), cudaMemcpyDeviceToHost);
        return result;
    }
};







__global__ void k1_count_kernel(
    const int32_t* __restrict__ csr_offsets,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ start_vertices,
    int num_starts,
    int64_t* __restrict__ counts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_starts) return;

    int32_t v = __ldg(&start_vertices[tid]);
    int32_t ebegin = __ldg(&csr_offsets[v]);
    int32_t eend = __ldg(&csr_offsets[v + 1]);

    if (ebegin >= eend) { counts[tid] = 0; return; }

    int cnt = 0;
    int32_t first_word = ebegin >> 5;
    int32_t last_word = (eend - 1) >> 5;
    int first_bit = ebegin & 31;
    int last_bit = eend & 31;

    if (first_word == last_word) {
        uint32_t mask = __ldg(&edge_mask[first_word]);
        mask >>= first_bit;
        int bits = eend - ebegin;
        if (bits < 32) mask &= (1u << bits) - 1;
        cnt = __popc(mask);
    } else {
        cnt = __popc(__ldg(&edge_mask[first_word]) >> first_bit);
        for (int32_t w = first_word + 1; w < last_word; w++) {
            cnt += __popc(__ldg(&edge_mask[w]));
        }
        if (last_bit == 0) cnt += __popc(__ldg(&edge_mask[last_word]));
        else cnt += __popc(__ldg(&edge_mask[last_word]) & ((1u << last_bit) - 1));
    }
    counts[tid] = cnt;
}

__global__ void k1_write_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ start_vertices,
    int num_starts,
    const int64_t* __restrict__ out_offsets,
    int32_t* __restrict__ out_neighbors
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_starts) return;

    int32_t v = __ldg(&start_vertices[warp_id]);
    int32_t ebegin = __ldg(&csr_offsets[v]);
    int32_t eend = __ldg(&csr_offsets[v + 1]);
    int64_t base_pos = out_offsets[warp_id];

    for (int32_t chunk = ebegin; chunk < eend; chunk += 32) {
        int32_t e = chunk + lane;
        bool valid = (e < eend);
        bool active = valid && ((__ldg(&edge_mask[e >> 5]) >> (e & 31)) & 1u);

        uint32_t ballot = __ballot_sync(0xFFFFFFFF, active);
        int prefix = __popc(ballot & ((1u << lane) - 1));
        int total = __popc(ballot);

        if (active) {
            out_neighbors[base_pos + prefix] = __ldg(&csr_indices[e]);
        }
        base_pos += total;
    }
}



__global__ void mark_start_vertices_kernel(
    const int32_t* __restrict__ start_vertices,
    int num_starts,
    uint32_t* __restrict__ bitmap,
    int bitmap_words
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_starts) return;
    int32_t v = __ldg(&start_vertices[tid]);
    size_t base = (size_t)tid * bitmap_words;
    atomicOr(&bitmap[base + (v >> 5)], 1u << (v & 31));
}

__global__ void expand_bitmap_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const uint32_t* __restrict__ edge_mask,
    const uint32_t* __restrict__ current_bitmap,
    uint32_t* __restrict__ next_bitmap,
    int num_starts,
    int num_vertices,
    int bitmap_words
) {
    int start_idx = blockIdx.x;
    if (start_idx >= num_starts) return;

    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;

    size_t base = (size_t)start_idx * bitmap_words;
    const uint32_t* my_frontier = current_bitmap + base;
    uint32_t* my_next = next_bitmap + base;

    for (int w = warp_id; w < bitmap_words; w += warps_per_block) {
        uint32_t word = __ldg(&my_frontier[w]);

        while (word) {
            int vertex;
            if (lane == 0) {
                int bit = __ffs(word) - 1;
                word &= word - 1;
                vertex = w * 32 + bit;
            }
            vertex = __shfl_sync(0xFFFFFFFF, vertex, 0);
            word = __shfl_sync(0xFFFFFFFF, word, 0);

            if (vertex >= num_vertices) break;

            int ebegin = __ldg(&csr_offsets[vertex]);
            int eend = __ldg(&csr_offsets[vertex + 1]);
            for (int e = ebegin + lane; e < eend; e += 32) {
                if ((__ldg(&edge_mask[e >> 5]) >> (e & 31)) & 1u) {
                    int neighbor = __ldg(&csr_indices[e]);
                    atomicOr(&my_next[neighbor >> 5], 1u << (neighbor & 31));
                }
            }
        }
    }
}

__global__ void count_bitmap_kernel(
    const uint32_t* __restrict__ bitmap,
    int num_starts,
    int num_vertices,
    int bitmap_words,
    int64_t* __restrict__ counts
) {
    int start_idx = blockIdx.x;
    if (start_idx >= num_starts) return;

    size_t base = (size_t)start_idx * bitmap_words;

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;

    int64_t local_count = 0;
    for (int w = threadIdx.x; w < bitmap_words; w += blockDim.x) {
        uint32_t word = bitmap[base + w];
        int start_bit = w * 32;
        if (start_bit + 32 > num_vertices) {
            int valid = num_vertices - start_bit;
            if (valid > 0 && valid < 32) word &= (1u << valid) - 1;
            else if (valid <= 0) word = 0;
        }
        local_count += __popc(word);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, offset);

    __shared__ int64_t warp_counts[32];
    if (lane == 0) warp_counts[warp_id] = local_count;
    __syncthreads();

    if (warp_id == 0) {
        local_count = (lane < warps) ? warp_counts[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1)
            local_count += __shfl_down_sync(0xFFFFFFFF, local_count, offset);
        if (lane == 0) counts[start_idx] = local_count;
    }
}

__global__ void extract_bitmap_kernel(
    const uint32_t* __restrict__ bitmap,
    int num_starts,
    int num_vertices,
    int bitmap_words,
    const int64_t* __restrict__ offsets,
    int32_t* __restrict__ neighbors
) {
    int start_idx = blockIdx.x;
    if (start_idx >= num_starts) return;

    size_t base = (size_t)start_idx * bitmap_words;
    int64_t out_base = offsets[start_idx];

    extern __shared__ int smem_extract[];
    int64_t running_offset = 0;

    for (int chunk = 0; chunk < bitmap_words; chunk += blockDim.x) {
        int w = chunk + threadIdx.x;
        uint32_t word = 0;
        int my_count = 0;

        if (w < bitmap_words) {
            word = bitmap[base + w];
            int start_bit = w * 32;
            if (start_bit + 32 > num_vertices) {
                int valid = num_vertices - start_bit;
                if (valid > 0 && valid < 32) word &= (1u << valid) - 1;
                else if (valid <= 0) word = 0;
            }
            my_count = __popc(word);
        }

        smem_extract[threadIdx.x] = my_count;
        __syncthreads();

        for (int stride = 1; stride < (int)blockDim.x; stride *= 2) {
            int val = 0;
            if ((int)threadIdx.x >= stride) val = smem_extract[threadIdx.x - stride];
            __syncthreads();
            smem_extract[threadIdx.x] += val;
            __syncthreads();
        }

        int my_exclusive = smem_extract[threadIdx.x] - my_count;
        int my_pos = (int)running_offset + my_exclusive;

        if (w < bitmap_words) {
            int64_t pos = out_base + my_pos;
            while (word) {
                int bit = __ffs(word) - 1;
                word &= word - 1;
                int vertex = w * 32 + bit;
                if (vertex < num_vertices) {
                    neighbors[pos++] = vertex;
                }
            }
        }

        running_offset += smem_extract[blockDim.x - 1];
        __syncthreads();
    }
}



__global__ void init_frontier_kernel(
    const int32_t* __restrict__ start_vertices,
    int num_starts, int64_t num_vertices,
    int64_t* __restrict__ frontier_keys
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_starts) return;
    frontier_keys[tid] = (int64_t)tid * num_vertices + __ldg(&start_vertices[tid]);
}

__global__ void count_from_keys_kernel(
    const int32_t* __restrict__ csr_offsets,
    const uint32_t* __restrict__ edge_mask,
    const int64_t* __restrict__ frontier_keys,
    int num_frontier, int64_t num_vertices,
    int64_t* __restrict__ counts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_frontier) return;
    int32_t v = (int32_t)(frontier_keys[tid] % num_vertices);
    int32_t ebegin = __ldg(&csr_offsets[v]);
    int32_t eend = __ldg(&csr_offsets[v + 1]);
    int64_t cnt = 0;
    for (int32_t e = ebegin; e < eend; e++) {
        if ((__ldg(&edge_mask[e >> 5]) >> (e & 31)) & 1u) cnt++;
    }
    counts[tid] = cnt;
}

__global__ void expand_from_keys_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const uint32_t* __restrict__ edge_mask,
    const int64_t* __restrict__ frontier_keys,
    int num_frontier, int64_t num_vertices,
    const int64_t* __restrict__ inc_sums,
    int64_t* __restrict__ output_keys
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_frontier) return;
    int64_t key = frontier_keys[tid];
    int32_t idx = (int32_t)(key / num_vertices);
    int32_t v = (int32_t)(key % num_vertices);
    int32_t ebegin = __ldg(&csr_offsets[v]);
    int32_t eend = __ldg(&csr_offsets[v + 1]);
    int64_t pos = (tid == 0) ? 0 : inc_sums[tid - 1];
    for (int32_t e = ebegin; e < eend; e++) {
        if ((__ldg(&edge_mask[e >> 5]) >> (e & 31)) & 1u) {
            output_keys[pos++] = (int64_t)idx * num_vertices + __ldg(&csr_indices[e]);
        }
    }
}

__global__ void build_offsets_kernel(
    const int64_t* __restrict__ keys, int num_keys,
    int64_t num_vertices, int num_starts,
    int64_t* __restrict__ offsets
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > num_starts) return;
    if (num_keys == 0) { offsets[i] = 0; return; }
    if (i == num_starts) { offsets[i] = num_keys; return; }
    int64_t target = (int64_t)i * num_vertices;
    int lo = 0, hi = num_keys;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (__ldg(&keys[mid]) < target) lo = mid + 1;
        else hi = mid;
    }
    offsets[i] = lo;
}

__global__ void extract_neighbors_kernel(
    const int64_t* __restrict__ keys, int num_keys,
    int64_t num_vertices, int32_t* __restrict__ neighbors
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_keys) return;
    neighbors[tid] = (int32_t)(__ldg(&keys[tid]) % num_vertices);
}





void launch_k1_count(const int32_t* offsets, const uint32_t* edge_mask,
                     const int32_t* start_vertices, int num_starts, int64_t* counts) {
    int block = 256;
    int grid = (num_starts + block - 1) / block;
    k1_count_kernel<<<grid, block>>>(offsets, edge_mask, start_vertices, num_starts, counts);
}

void launch_k1_write(const int32_t* offsets, const int32_t* indices,
                     const uint32_t* edge_mask, const int32_t* start_vertices,
                     int num_starts, const int64_t* out_offsets, int32_t* out_neighbors) {
    int threads = 256;
    int warps_per_block = threads / 32;
    int grid = (num_starts + warps_per_block - 1) / warps_per_block;
    k1_write_kernel<<<grid, threads>>>(offsets, indices, edge_mask, start_vertices,
                                        num_starts, out_offsets, out_neighbors);
}

void launch_mark_starts(const int32_t* start_vertices, int num_starts,
                        uint32_t* bitmap, int bitmap_words) {
    int block = 256;
    int grid = (num_starts + block - 1) / block;
    mark_start_vertices_kernel<<<grid, block>>>(start_vertices, num_starts, bitmap, bitmap_words);
}

void launch_expand_bitmap(const int32_t* offsets, const int32_t* indices,
                          const uint32_t* edge_mask,
                          const uint32_t* current, uint32_t* next,
                          int num_starts, int num_vertices, int bitmap_words) {
    expand_bitmap_kernel<<<num_starts, 256>>>(offsets, indices, edge_mask,
                                               current, next,
                                               num_starts, num_vertices, bitmap_words);
}

void launch_count_bitmap(const uint32_t* bitmap, int num_starts, int num_vertices,
                         int bitmap_words, int64_t* counts) {
    count_bitmap_kernel<<<num_starts, 256>>>(bitmap, num_starts, num_vertices,
                                              bitmap_words, counts);
}

void launch_extract_bitmap(const uint32_t* bitmap, int num_starts, int num_vertices,
                           int bitmap_words, const int64_t* offsets, int32_t* neighbors) {
    int block = 256;
    int smem = block * sizeof(int);
    extract_bitmap_kernel<<<num_starts, block, smem>>>(bitmap, num_starts, num_vertices,
                                                        bitmap_words, offsets, neighbors);
}

void launch_init_frontier(const int32_t* start_vertices, int num_starts,
                          int64_t num_vertices, int64_t* frontier_keys) {
    int block = 256;
    int grid = (num_starts + block - 1) / block;
    init_frontier_kernel<<<grid, block>>>(start_vertices, num_starts, num_vertices, frontier_keys);
}

void launch_count_from_keys(const int32_t* offsets, const uint32_t* edge_mask,
                            const int64_t* keys, int num_frontier,
                            int64_t num_vertices, int64_t* counts) {
    int block = 256;
    int grid = (num_frontier + block - 1) / block;
    count_from_keys_kernel<<<grid, block>>>(offsets, edge_mask, keys, num_frontier,
                                             num_vertices, counts);
}

void launch_expand_from_keys(const int32_t* offsets, const int32_t* indices,
                              const uint32_t* edge_mask,
                              const int64_t* keys, int num_frontier,
                              int64_t num_vertices,
                              const int64_t* inc_sums, int64_t* output_keys) {
    int block = 256;
    int grid = (num_frontier + block - 1) / block;
    expand_from_keys_kernel<<<grid, block>>>(offsets, indices, edge_mask,
                                              keys, num_frontier, num_vertices,
                                              inc_sums, output_keys);
}

void launch_build_offsets(const int64_t* keys, int num_keys,
                         int64_t num_vertices, int num_starts, int64_t* offsets) {
    int block = 256;
    int grid = (num_starts + 1 + block - 1) / block;
    build_offsets_kernel<<<grid, block>>>(keys, num_keys, num_vertices, num_starts, offsets);
}

void launch_extract_neighbors(const int64_t* keys, int num_keys,
                              int64_t num_vertices, int32_t* neighbors) {
    if (num_keys == 0) return;
    int block = 256;
    int grid = (num_keys + block - 1) / block;
    extract_neighbors_kernel<<<grid, block>>>(keys, num_keys, num_vertices, neighbors);
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
    int32_t num_vertices = graph.number_of_vertices;
    bool is_multigraph = graph.is_multigraph;
    const uint32_t* d_edge_mask = graph.edge_mask;
    int num_starts = (int)num_start_vertices;
    int64_t V = (int64_t)num_vertices;

    if (k == 1 && !is_multigraph) {
        
        int64_t* d_offsets_out = nullptr;
        cudaMalloc(&d_offsets_out, (num_starts + 1) * sizeof(int64_t));
        int64_t* d_counts = d_offsets_out + 1;

        launch_k1_count(d_offsets, d_edge_mask, start_vertices, num_starts, d_counts);

        cache.inclusive_sum(d_counts, d_counts, num_starts);
        cudaMemsetAsync(d_offsets_out, 0, sizeof(int64_t));

        int64_t total;
        cudaMemcpy(&total, d_offsets_out + num_starts,
                   sizeof(int64_t), cudaMemcpyDeviceToHost);

        int32_t* d_neighbors = nullptr;
        if (total > 0) {
            cudaMalloc(&d_neighbors, total * sizeof(int32_t));
            launch_k1_write(d_offsets, d_indices, d_edge_mask, start_vertices, num_starts,
                           d_offsets_out, d_neighbors);
        }

        return k_hop_nbrs_result_t{
            reinterpret_cast<std::size_t*>(d_offsets_out),
            d_neighbors,
            (std::size_t)(num_starts + 1),
            (std::size_t)total
        };

    } else if (k >= 2 && !is_multigraph) {
        
        int bitmap_words = (num_vertices + 31) / 32;
        int64_t total_words = (int64_t)num_starts * bitmap_words;
        int64_t total_bytes = total_words * 4;

        if (total_bytes * 2 < 4LL * 1024 * 1024 * 1024) {
            cache.ensure_bitmap_a(total_bytes);
            cache.ensure_bitmap_b(total_bytes);
            uint32_t* bmp_current = reinterpret_cast<uint32_t*>(cache.bitmap_a);
            uint32_t* bmp_next = reinterpret_cast<uint32_t*>(cache.bitmap_b);

            cudaMemsetAsync(bmp_current, 0, total_bytes);
            cudaMemsetAsync(bmp_next, 0, total_bytes);
            launch_mark_starts(start_vertices, num_starts, bmp_current, bitmap_words);

            for (std::size_t hop = 0; hop < k; hop++) {
                cudaMemsetAsync(bmp_next, 0, total_bytes);
                launch_expand_bitmap(d_offsets, d_indices, d_edge_mask,
                                    bmp_current, bmp_next,
                                    num_starts, num_vertices, bitmap_words);
                uint32_t* tmp = bmp_current; bmp_current = bmp_next; bmp_next = tmp;
            }

            int64_t* d_counts = nullptr;
            cudaMalloc(&d_counts, num_starts * sizeof(int64_t));
            launch_count_bitmap(bmp_current, num_starts, num_vertices, bitmap_words,
                               d_counts);

            int64_t* d_offsets_out = nullptr;
            cudaMalloc(&d_offsets_out, (num_starts + 1) * sizeof(int64_t));
            cudaMemsetAsync(d_offsets_out, 0, sizeof(int64_t));
            cache.inclusive_sum(d_counts, d_offsets_out + 1, num_starts);
            cudaFree(d_counts);

            int64_t total;
            cudaMemcpy(&total, d_offsets_out + num_starts,
                       sizeof(int64_t), cudaMemcpyDeviceToHost);

            int32_t* d_neighbors = nullptr;
            if (total > 0) {
                cudaMalloc(&d_neighbors, total * sizeof(int32_t));
                launch_extract_bitmap(bmp_current, num_starts, num_vertices, bitmap_words,
                                     d_offsets_out, d_neighbors);
            }

            return k_hop_nbrs_result_t{
                reinterpret_cast<std::size_t*>(d_offsets_out),
                d_neighbors,
                (std::size_t)(num_starts + 1),
                (std::size_t)total
            };
        }
        
    }

    
    int64_t* d_frontier = nullptr;
    cudaMalloc(&d_frontier, num_starts * sizeof(int64_t));
    launch_init_frontier(start_vertices, num_starts, V, d_frontier);
    int64_t frontier_size = num_starts;

    for (std::size_t hop = 0; hop < k; hop++) {
        if (frontier_size == 0) break;

        int64_t* d_counts = nullptr;
        cudaMalloc(&d_counts, frontier_size * sizeof(int64_t));
        launch_count_from_keys(d_offsets, d_edge_mask,
                               d_frontier, (int)frontier_size,
                               V, d_counts);

        int64_t* d_inc_sums = nullptr;
        cudaMalloc(&d_inc_sums, frontier_size * sizeof(int64_t));
        cache.inclusive_sum(d_counts, d_inc_sums, frontier_size);
        cudaFree(d_counts);

        int64_t expanded_size;
        cudaMemcpy(&expanded_size, d_inc_sums + frontier_size - 1,
                   sizeof(int64_t), cudaMemcpyDeviceToHost);

        if (expanded_size == 0) {
            cudaFree(d_inc_sums);
            frontier_size = 0;
            break;
        }

        int64_t* d_expanded = nullptr;
        cudaMalloc(&d_expanded, expanded_size * sizeof(int64_t));
        launch_expand_from_keys(d_offsets, d_indices, d_edge_mask,
                                 d_frontier, (int)frontier_size,
                                 V, d_inc_sums, d_expanded);
        cudaFree(d_inc_sums);
        cudaFree(d_frontier);

        int64_t* d_sorted = nullptr;
        cudaMalloc(&d_sorted, expanded_size * sizeof(int64_t));
        cache.sort_keys(d_expanded, d_sorted, expanded_size);
        cudaFree(d_expanded);

        int64_t* d_unique = nullptr;
        cudaMalloc(&d_unique, expanded_size * sizeof(int64_t));
        int64_t new_size = cache.unique_keys(d_sorted, d_unique, expanded_size);
        cudaFree(d_sorted);

        d_frontier = d_unique;
        frontier_size = new_size;
    }

    int64_t* d_offsets_out = nullptr;
    cudaMalloc(&d_offsets_out, (num_starts + 1) * sizeof(int64_t));
    launch_build_offsets(d_frontier, (int)frontier_size,
                        V, num_starts, d_offsets_out);

    int32_t* d_neighbors = nullptr;
    if (frontier_size > 0) {
        cudaMalloc(&d_neighbors, frontier_size * sizeof(int32_t));
    }
    launch_extract_neighbors(d_frontier, (int)frontier_size,
                            V, d_neighbors);
    cudaFree(d_frontier);

    return k_hop_nbrs_result_t{
        reinterpret_cast<std::size_t*>(d_offsets_out),
        d_neighbors,
        (std::size_t)(num_starts + 1),
        (std::size_t)frontier_size
    };
}

}  
