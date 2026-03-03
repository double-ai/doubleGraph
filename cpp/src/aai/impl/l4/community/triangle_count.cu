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
    int32_t* sort_flag = nullptr;

    int32_t* sorted_indices = nullptr;
    int64_t sorted_indices_capacity = 0;

    uint8_t* sort_temp = nullptr;
    int64_t sort_temp_capacity = 0;

    int32_t* upper_counts = nullptr;
    int64_t upper_counts_capacity = 0;

    int32_t* upper_prefix = nullptr;
    int64_t upper_prefix_capacity = 0;

    uint8_t* scan_temp = nullptr;
    int64_t scan_temp_capacity = 0;

    int32_t* edge_src = nullptr;
    int64_t edge_src_capacity = 0;

    int32_t* edge_dst = nullptr;
    int64_t edge_dst_capacity = 0;

    Cache() {
        cudaMalloc(&sort_flag, sizeof(int32_t));
    }

    ~Cache() override {
        if (sort_flag) cudaFree(sort_flag);
        if (sorted_indices) cudaFree(sorted_indices);
        if (sort_temp) cudaFree(sort_temp);
        if (upper_counts) cudaFree(upper_counts);
        if (upper_prefix) cudaFree(upper_prefix);
        if (scan_temp) cudaFree(scan_temp);
        if (edge_src) cudaFree(edge_src);
        if (edge_dst) cudaFree(edge_dst);
    }

    void ensure_sorted_indices(int64_t n) {
        if (sorted_indices_capacity < n) {
            if (sorted_indices) cudaFree(sorted_indices);
            cudaMalloc(&sorted_indices, n * sizeof(int32_t));
            sorted_indices_capacity = n;
        }
    }

    void ensure_sort_temp(int64_t n) {
        if (sort_temp_capacity < n) {
            if (sort_temp) cudaFree(sort_temp);
            cudaMalloc(&sort_temp, n);
            sort_temp_capacity = n;
        }
    }

    void ensure_upper_counts(int64_t n) {
        if (upper_counts_capacity < n) {
            if (upper_counts) cudaFree(upper_counts);
            cudaMalloc(&upper_counts, n * sizeof(int32_t));
            upper_counts_capacity = n;
        }
    }

    void ensure_upper_prefix(int64_t n) {
        if (upper_prefix_capacity < n) {
            if (upper_prefix) cudaFree(upper_prefix);
            cudaMalloc(&upper_prefix, n * sizeof(int32_t));
            upper_prefix_capacity = n;
        }
    }

    void ensure_scan_temp(int64_t n) {
        if (scan_temp_capacity < n) {
            if (scan_temp) cudaFree(scan_temp);
            cudaMalloc(&scan_temp, n);
            scan_temp_capacity = n;
        }
    }

    void ensure_edge_src(int64_t n) {
        if (edge_src_capacity < n) {
            if (edge_src) cudaFree(edge_src);
            cudaMalloc(&edge_src, n * sizeof(int32_t));
            edge_src_capacity = n;
        }
    }

    void ensure_edge_dst(int64_t n) {
        if (edge_dst_capacity < n) {
            if (edge_dst) cudaFree(edge_dst);
            cudaMalloc(&edge_dst, n * sizeof(int32_t));
            edge_dst_capacity = n;
        }
    }
};





__device__ __forceinline__ int binary_search_from(
    const int32_t* __restrict__ arr, int32_t len, int32_t target, int32_t& bs_lo
) {
    int32_t lo = bs_lo, hi = len;
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    bs_lo = lo;
    return (lo < len && arr[lo] == target) ? 1 : 0;
}





__global__ void check_sorted_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ result,
    int32_t num_vertices
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;

    int32_t start = offsets[u];
    int32_t end = offsets[u + 1];

    for (int32_t i = start + 1; i < end; i++) {
        if (indices[i] < indices[i-1]) {
            atomicExch(result, 0);
            return;
        }
    }
}





__global__ void count_upper_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ upper_counts,
    int32_t num_vertices
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;

    int32_t start = offsets[u];
    int32_t end = offsets[u + 1];

    int32_t lo = start, hi = end;
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (indices[mid] <= u) lo = mid + 1;
        else hi = mid;
    }

    upper_counts[u] = end - lo;
}

__global__ void build_upper_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ upper_prefix,
    int32_t* __restrict__ edge_src,
    int32_t* __restrict__ edge_dst,
    int32_t num_vertices
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;

    int32_t start = offsets[u];
    int32_t end = offsets[u + 1];

    int32_t lo = start, hi = end;
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (indices[mid] <= u) lo = mid + 1;
        else hi = mid;
    }

    int32_t out_pos = upper_prefix[u];
    for (int32_t i = lo; i < end; i++) {
        edge_src[out_pos] = u;
        edge_dst[out_pos] = indices[i];
        out_pos++;
    }
}





__global__ void triangle_count_edge_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ edge_src,
    const int32_t* __restrict__ edge_dst,
    int32_t* __restrict__ counts,
    int32_t total_upper_edges
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= total_upper_edges) return;

    int32_t u = edge_src[warp_id];
    int32_t v = edge_dst[warp_id];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t u_deg = u_end - u_start;

    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];
    int32_t v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) return;

    const int32_t* search_arr;
    int32_t search_len;
    const int32_t* lookup_arr;
    int32_t lookup_len;

    if (u_deg <= v_deg) {
        search_arr = indices + u_start;
        search_len = u_deg;
        lookup_arr = indices + v_start;
        lookup_len = v_deg;
    } else {
        search_arr = indices + v_start;
        search_len = v_deg;
        lookup_arr = indices + u_start;
        lookup_len = u_deg;
    }

    if (search_arr[search_len - 1] < lookup_arr[0] ||
        lookup_arr[lookup_len - 1] < search_arr[0]) return;

    int32_t count = 0;
    int32_t bs_lo = 0;

    for (int32_t j = lane; j < search_len; j += 32) {
        int32_t w = search_arr[j];
        if (w != u && w != v) {
            count += binary_search_from(lookup_arr, lookup_len, w, bs_lo);
        } else {
            
            if (w <= lookup_arr[lookup_len - 1]) {
                int32_t lo2 = bs_lo, hi2 = lookup_len;
                while (lo2 < hi2) {
                    int32_t mid = lo2 + ((hi2 - lo2) >> 1);
                    if (lookup_arr[mid] < w) lo2 = mid + 1;
                    else hi2 = mid;
                }
                bs_lo = lo2;
            }
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        count += __shfl_down_sync(0xffffffff, count, offset);
    }

    if (lane == 0 && count > 0) {
        atomicAdd(&counts[u], count);
        atomicAdd(&counts[v], count);
    }
}

__global__ void triangle_count_vertex_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ counts,
    const int32_t* __restrict__ vertices,
    int32_t n_query
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= n_query) return;

    int32_t u = vertices[warp_id];
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t u_deg = u_end - u_start;

    if (u_deg == 0) {
        if (lane == 0) counts[warp_id] = 0;
        return;
    }

    int32_t total = 0;

    for (int32_t i = u_start; i < u_end; i++) {
        int32_t v = indices[i];
        if (v == u) continue;

        int32_t v_start = offsets[v];
        int32_t v_end = offsets[v + 1];
        int32_t v_deg = v_end - v_start;
        if (v_deg == 0) continue;

        const int32_t* search_arr;
        int32_t search_len;
        const int32_t* lookup_arr;
        int32_t lookup_len;

        if (u_deg <= v_deg) {
            search_arr = indices + u_start;
            search_len = u_deg;
            lookup_arr = indices + v_start;
            lookup_len = v_deg;
        } else {
            search_arr = indices + v_start;
            search_len = v_deg;
            lookup_arr = indices + u_start;
            lookup_len = u_deg;
        }

        if (search_arr[search_len - 1] < lookup_arr[0] ||
            lookup_arr[lookup_len - 1] < search_arr[0]) continue;

        int32_t count = 0;
        int32_t bs_lo = 0;
        for (int32_t j = lane; j < search_len; j += 32) {
            int32_t w = search_arr[j];
            if (w != u && w != v) {
                count += binary_search_from(lookup_arr, lookup_len, w, bs_lo);
            }
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            count += __shfl_down_sync(0xffffffff, count, offset);
        }

        if (lane == 0) total += count;
    }

    if (lane == 0) counts[warp_id] = total / 2;
}

__global__ void divide_by_two_kernel(int32_t* counts, int32_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) counts[tid] /= 2;
}

}  





void triangle_count(const graph32_t& graph,
                    int32_t* counts,
                    const int32_t* vertices,
                    std::size_t n_vertices) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    
    const int32_t* d_sorted = d_indices;

    if (num_edges > 0) {
        int32_t one = 1;
        cudaMemcpy(cache.sort_flag, &one, sizeof(int32_t), cudaMemcpyHostToDevice);

        int t = 256, b = (num_vertices + t - 1) / t;
        check_sorted_kernel<<<b, t>>>(d_offsets, d_indices, cache.sort_flag, num_vertices);

        int32_t is_sorted;
        cudaMemcpy(&is_sorted, cache.sort_flag, sizeof(int32_t), cudaMemcpyDeviceToHost);

        if (!is_sorted) {
            cache.ensure_sorted_indices(num_edges);
            d_sorted = cache.sorted_indices;

            size_t sort_tb = 0;
            cub::DeviceSegmentedSort::SortKeys(nullptr, sort_tb, d_indices,
                cache.sorted_indices, (int)num_edges, (int)num_vertices,
                d_offsets, d_offsets + 1);

            cache.ensure_sort_temp((int64_t)sort_tb);
            cub::DeviceSegmentedSort::SortKeys(cache.sort_temp, sort_tb, d_indices,
                cache.sorted_indices, (int)num_edges, (int)num_vertices,
                d_offsets, d_offsets + 1);
        }
    }

    
    if (vertices != nullptr) {
        int32_t n_query = (int32_t)n_vertices;
        if (n_query == 0) return;
        int t = 256, wpb = t / 32, b = (n_query + wpb - 1) / wpb;
        triangle_count_vertex_kernel<<<b, t>>>(d_offsets, d_sorted, counts, vertices, n_query);
        return;
    }

    
    if (num_vertices == 0) return;

    
    cache.ensure_upper_counts(num_vertices);
    {
        int t = 256, b = (num_vertices + t - 1) / t;
        count_upper_edges_kernel<<<b, t>>>(d_offsets, d_sorted, cache.upper_counts, num_vertices);
    }

    
    cache.ensure_upper_prefix(num_vertices);
    {
        size_t scan_tb = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, scan_tb, cache.upper_counts,
            cache.upper_prefix, (int)num_vertices);
        cache.ensure_scan_temp((int64_t)scan_tb);
        cub::DeviceScan::ExclusiveSum(cache.scan_temp, scan_tb, cache.upper_counts,
            cache.upper_prefix, (int)num_vertices);
    }

    
    int32_t last_p, last_c;
    cudaMemcpy(&last_p, cache.upper_prefix + num_vertices - 1,
               sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&last_c, cache.upper_counts + num_vertices - 1,
               sizeof(int32_t), cudaMemcpyDeviceToHost);
    int32_t total_upper = last_p + last_c;

    
    if (total_upper > 0) {
        cache.ensure_edge_src(total_upper);
        cache.ensure_edge_dst(total_upper);
        {
            int t = 256, b = (num_vertices + t - 1) / t;
            build_upper_edges_kernel<<<b, t>>>(d_offsets, d_sorted, cache.upper_prefix,
                cache.edge_src, cache.edge_dst, num_vertices);
        }
    }

    
    cudaMemset(counts, 0, num_vertices * sizeof(int32_t));

    if (total_upper > 0) {
        int t = 256, wpb = t / 32, b = (total_upper + wpb - 1) / wpb;
        triangle_count_edge_kernel<<<b, t>>>(d_offsets, d_sorted,
            cache.edge_src, cache.edge_dst, counts, total_upper);
    }

    
    {
        int t = 256, b = (num_vertices + t - 1) / t;
        divide_by_two_kernel<<<b, t>>>(counts, num_vertices);
    }
}

}  
