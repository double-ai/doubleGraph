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
    int32_t* src_array = nullptr;
    uint32_t* active_mask = nullptr;
    int32_t* changed_flag = nullptr;
    int32_t* flags = nullptr;
    int32_t* positions = nullptr;
    uint8_t* dirty_a = nullptr;
    uint8_t* dirty_b = nullptr;
    void* scan_temp = nullptr;

    int64_t src_array_cap = 0;
    int64_t active_mask_cap = 0;
    bool changed_flag_alloc = false;
    int64_t flags_cap = 0;
    int64_t positions_cap = 0;
    int64_t dirty_a_cap = 0;
    int64_t dirty_b_cap = 0;
    size_t scan_temp_cap = 0;

    void ensure(int32_t num_edges, int32_t num_vertices, size_t temp_size) {
        int64_t ne = num_edges;
        int64_t mw = (num_edges + 31) / 32;
        int64_t nv = num_vertices;

        if (src_array_cap < ne) {
            if (src_array) cudaFree(src_array);
            cudaMalloc(&src_array, ne * sizeof(int32_t));
            src_array_cap = ne;
        }
        if (active_mask_cap < mw) {
            if (active_mask) cudaFree(active_mask);
            cudaMalloc(&active_mask, mw * sizeof(uint32_t));
            active_mask_cap = mw;
        }
        if (!changed_flag_alloc) {
            cudaMalloc(&changed_flag, sizeof(int32_t));
            changed_flag_alloc = true;
        }
        if (flags_cap < ne) {
            if (flags) cudaFree(flags);
            cudaMalloc(&flags, ne * sizeof(int32_t));
            flags_cap = ne;
        }
        if (positions_cap < ne) {
            if (positions) cudaFree(positions);
            cudaMalloc(&positions, ne * sizeof(int32_t));
            positions_cap = ne;
        }
        if (dirty_a_cap < nv) {
            if (dirty_a) cudaFree(dirty_a);
            cudaMalloc(&dirty_a, nv * sizeof(uint8_t));
            dirty_a_cap = nv;
        }
        if (dirty_b_cap < nv) {
            if (dirty_b) cudaFree(dirty_b);
            cudaMalloc(&dirty_b, nv * sizeof(uint8_t));
            dirty_b_cap = nv;
        }
        if (scan_temp_cap < temp_size) {
            if (scan_temp) cudaFree(scan_temp);
            cudaMalloc(&scan_temp, temp_size);
            scan_temp_cap = temp_size;
        }
    }

    ~Cache() override {
        if (src_array) cudaFree(src_array);
        if (active_mask) cudaFree(active_mask);
        if (changed_flag) cudaFree(changed_flag);
        if (flags) cudaFree(flags);
        if (positions) cudaFree(positions);
        if (dirty_a) cudaFree(dirty_a);
        if (dirty_b) cudaFree(dirty_b);
        if (scan_temp) cudaFree(scan_temp);
    }
};


__global__ void compute_src_array(
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ src_array,
    int32_t num_vertices
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;
    int start = offsets[u];
    int end = offsets[u + 1];
    for (int j = start; j < end; j++) {
        src_array[j] = u;
    }
}


__global__ void count_and_remove(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ src_array,
    uint32_t* active_mask,
    const uint8_t* __restrict__ dirty_prev,
    uint8_t* dirty_curr,
    int32_t threshold,
    int32_t num_edges,
    int32_t* changed_flag
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_edges) return;

    int edge_idx = warp_id;

    
    uint32_t mask_word = active_mask[edge_idx >> 5];
    if (!((mask_word >> (edge_idx & 31)) & 1u)) return;

    int u = src_array[edge_idx];
    int v = indices[edge_idx];

    
    if (u == v) {
        if (lane == 0) {
            atomicAnd(&active_mask[edge_idx >> 5], ~(1u << (edge_idx & 31)));
            dirty_curr[u] = 1;
            atomicOr(changed_flag, 1);
        }
        return;
    }

    
    if (!dirty_prev[u] && !dirty_prev[v]) return;

    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int v_start = offsets[v];
    int v_end = offsets[v + 1];
    int u_deg = u_end - u_start;
    int v_deg = v_end - v_start;

    
    int iter_start, iter_end, search_start, search_end;
    if (u_deg <= v_deg) {
        iter_start = u_start; iter_end = u_end;
        search_start = v_start; search_end = v_end;
    } else {
        iter_start = v_start; iter_end = v_end;
        search_start = u_start; search_end = u_end;
    }

    int count = 0;

    if (iter_end > iter_start && search_end > search_start) {
        int search_min = indices[search_start];
        int search_max = indices[search_end - 1];

        int iter_len = iter_end - iter_start;
        int num_chunks = (iter_len + 31) >> 5;

        for (int c = 0; c < num_chunks; c++) {
            int i = iter_start + (c << 5) + lane;
            int local_match = 0;

            if (i < iter_end) {
                if ((active_mask[i >> 5] >> (i & 31)) & 1u) {
                    int w = indices[i];
                    if (w != u && w != v && w >= search_min && w <= search_max) {
                        
                        int lo = search_start, hi = search_end;
                        while (lo < hi) {
                            int mid = (lo + hi) >> 1;
                            if (indices[mid] < w) lo = mid + 1;
                            else hi = mid;
                        }
                        if (lo < search_end && indices[lo] == w) {
                            if ((active_mask[lo >> 5] >> (lo & 31)) & 1u) {
                                local_match = 1;
                            }
                        }
                    }
                }
            }

            count += local_match;

            
            if (threshold == 1) {
                
                if (__any_sync(0xffffffff, count > 0)) goto done;
            } else if (threshold == 2) {
                
                uint32_t vote = __ballot_sync(0xffffffff, count > 0);
                if (__popc(vote) >= 2) goto done;
                if (__any_sync(0xffffffff, count >= 2)) goto done;
            } else {
                
                int total = count;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    total += __shfl_down_sync(0xffffffff, total, off);
                total = __shfl_sync(0xffffffff, total, 0);
                if (total >= threshold) goto done;
            }
        }
    }

    
    {
        int total = count;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            total += __shfl_down_sync(0xffffffff, total, off);

        if (lane == 0 && total < threshold) {
            atomicAnd(&active_mask[edge_idx >> 5], ~(1u << (edge_idx & 31)));
            dirty_curr[u] = 1;
            dirty_curr[v] = 1;
            atomicOr(changed_flag, 1);
        }
    }

done:
    ;
}


__global__ void count_and_remove_first(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ src_array,
    uint32_t* active_mask,
    uint8_t* dirty_curr,
    int32_t threshold,
    int32_t num_edges,
    int32_t* changed_flag
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_edges) return;

    int edge_idx = warp_id;

    if (!((active_mask[edge_idx >> 5] >> (edge_idx & 31)) & 1u)) return;

    int u = src_array[edge_idx];
    int v = indices[edge_idx];

    if (u == v) {
        if (lane == 0) {
            atomicAnd(&active_mask[edge_idx >> 5], ~(1u << (edge_idx & 31)));
            dirty_curr[u] = 1;
            atomicOr(changed_flag, 1);
        }
        return;
    }

    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int v_start = offsets[v];
    int v_end = offsets[v + 1];
    int u_deg = u_end - u_start;
    int v_deg = v_end - v_start;

    int iter_start, iter_end, search_start, search_end;
    if (u_deg <= v_deg) {
        iter_start = u_start; iter_end = u_end;
        search_start = v_start; search_end = v_end;
    } else {
        iter_start = v_start; iter_end = v_end;
        search_start = u_start; search_end = u_end;
    }

    int count = 0;

    if (iter_end > iter_start && search_end > search_start) {
        int search_min = indices[search_start];
        int search_max = indices[search_end - 1];

        int iter_len = iter_end - iter_start;
        int num_chunks = (iter_len + 31) >> 5;

        for (int c = 0; c < num_chunks; c++) {
            int i = iter_start + (c << 5) + lane;
            int local_match = 0;

            if (i < iter_end) {
                if ((active_mask[i >> 5] >> (i & 31)) & 1u) {
                    int w = indices[i];
                    if (w != u && w != v && w >= search_min && w <= search_max) {
                        int lo = search_start, hi = search_end;
                        while (lo < hi) {
                            int mid = (lo + hi) >> 1;
                            if (indices[mid] < w) lo = mid + 1;
                            else hi = mid;
                        }
                        if (lo < search_end && indices[lo] == w) {
                            if ((active_mask[lo >> 5] >> (lo & 31)) & 1u) {
                                local_match = 1;
                            }
                        }
                    }
                }
            }

            count += local_match;

            if (threshold == 1) {
                if (__any_sync(0xffffffff, count > 0)) goto done_first;
            } else if (threshold == 2) {
                uint32_t vote = __ballot_sync(0xffffffff, count > 0);
                if (__popc(vote) >= 2) goto done_first;
                if (__any_sync(0xffffffff, count >= 2)) goto done_first;
            } else {
                int total = count;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    total += __shfl_down_sync(0xffffffff, total, off);
                total = __shfl_sync(0xffffffff, total, 0);
                if (total >= threshold) goto done_first;
            }
        }
    }

    {
        int total = count;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            total += __shfl_down_sync(0xffffffff, total, off);

        if (lane == 0 && total < threshold) {
            atomicAnd(&active_mask[edge_idx >> 5], ~(1u << (edge_idx & 31)));
            dirty_curr[u] = 1;
            dirty_curr[v] = 1;
            atomicOr(changed_flag, 1);
        }
    }

done_first:
    ;
}


__global__ void mask_to_flags(
    const uint32_t* __restrict__ active_mask,
    int32_t* __restrict__ flags,
    int32_t num_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    flags[idx] = (active_mask[idx >> 5] >> (idx & 31)) & 1;
}


__global__ void extract_edges(
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ src_array,
    const uint32_t* __restrict__ active_mask,
    const int32_t* __restrict__ positions,
    int32_t* __restrict__ out_src,
    int32_t* __restrict__ out_dst,
    int32_t num_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;

    if ((active_mask[idx >> 5] >> (idx & 31)) & 1u) {
        int pos = positions[idx];
        out_src[pos] = src_array[idx];
        out_dst[pos] = indices[idx];
    }
}

}  

k_truss_result_t k_truss_seg_mask(const graph32_t& graph,
                                  int32_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;

    int32_t threshold = k - 2;
    int32_t mask_words = (num_edges + 31) / 32;

    if (num_edges == 0) {
        k_truss_result_t result;
        result.edge_srcs = nullptr;
        result.edge_dsts = nullptr;
        result.num_edges = 0;
        return result;
    }

    
    size_t temp_size = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_size, (int32_t*)nullptr, (int32_t*)nullptr, (int)num_edges);

    cache.ensure(num_edges, num_vertices, temp_size);

    int32_t* d_src = cache.src_array;
    uint32_t* d_mask = cache.active_mask;
    int32_t* d_flag = cache.changed_flag;
    int32_t* d_fl = cache.flags;
    int32_t* d_pos = cache.positions;
    uint8_t* d_dirty_a = cache.dirty_a;
    uint8_t* d_dirty_b = cache.dirty_b;
    void* d_temp = cache.scan_temp;

    
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (num_vertices > 0)
            compute_src_array<<<grid, block>>>(d_offsets, d_src, num_vertices);
    }

    
    cudaMemcpyAsync(d_mask, d_edge_mask, mask_words * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    
    cudaMemsetAsync(d_flag, 0, sizeof(int32_t));
    cudaMemsetAsync(d_dirty_a, 0, num_vertices);

    {
        int warps_per_block = 8;
        int threads_per_block = warps_per_block * 32;
        int grid = (num_edges + warps_per_block - 1) / warps_per_block;
        if (num_edges > 0)
            count_and_remove_first<<<grid, threads_per_block>>>(
                d_offsets, d_indices, d_src, d_mask,
                d_dirty_a, threshold, num_edges, d_flag);
    }

    int32_t h_flag;
    cudaMemcpy(&h_flag, d_flag, sizeof(int32_t), cudaMemcpyDeviceToHost);

    if (h_flag) {
        
        uint8_t* dirty_prev = d_dirty_a;
        uint8_t* dirty_curr = d_dirty_b;

        while (h_flag) {
            cudaMemsetAsync(d_flag, 0, sizeof(int32_t));
            cudaMemsetAsync(dirty_curr, 0, num_vertices);

            {
                int warps_per_block = 8;
                int threads_per_block = warps_per_block * 32;
                int grid = (num_edges + warps_per_block - 1) / warps_per_block;
                if (num_edges > 0)
                    count_and_remove<<<grid, threads_per_block>>>(
                        d_offsets, d_indices, d_src, d_mask,
                        dirty_prev, dirty_curr,
                        threshold, num_edges, d_flag);
            }

            cudaMemcpy(&h_flag, d_flag, sizeof(int32_t), cudaMemcpyDeviceToHost);
            if (!h_flag) break;

            
            uint8_t* tmp = dirty_prev;
            dirty_prev = dirty_curr;
            dirty_curr = tmp;
        }
    }

    
    {
        int block = 256;
        int grid = (num_edges + block - 1) / block;
        if (num_edges > 0)
            mask_to_flags<<<grid, block>>>(d_mask, d_fl, num_edges);
    }

    cub::DeviceScan::ExclusiveSum(d_temp, temp_size, d_fl, d_pos, (int)num_edges);

    int32_t last_flag = 0, last_pos = 0;
    cudaMemcpy(&last_flag, d_fl + num_edges - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_pos, d_pos + num_edges - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
    int32_t total_edges = last_flag + last_pos;

    int32_t* out_src = nullptr;
    int32_t* out_dst = nullptr;

    if (total_edges > 0) {
        cudaMalloc(&out_src, total_edges * sizeof(int32_t));
        cudaMalloc(&out_dst, total_edges * sizeof(int32_t));

        int block = 256;
        int grid = (num_edges + block - 1) / block;
        extract_edges<<<grid, block>>>(d_indices, d_src, d_mask, d_pos,
                                       out_src, out_dst, num_edges);
    }

    cudaDeviceSynchronize();

    k_truss_result_t result;
    result.edge_srcs = out_src;
    result.edge_dsts = out_dst;
    result.num_edges = static_cast<std::size_t>(total_edges);
    return result;
}

}  
