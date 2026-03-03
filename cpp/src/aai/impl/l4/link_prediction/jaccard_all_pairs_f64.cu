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
#include <cstdint>
#include <optional>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/remove.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/functional.h>

namespace aai {

namespace {

struct Cache : Cacheable {};





__global__ void count_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ counts
) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_seeds) return;
    int u = seeds[s];
    int64_t cnt = 0;
    for (int i = offsets[u]; i < offsets[u + 1]; i++) {
        int w = indices[i];
        cnt += offsets[w + 1] - offsets[w];
    }
    counts[s] = cnt;
}

__global__ void gen_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ pair_offsets,
    uint64_t* __restrict__ keys
) {
    int s = blockIdx.x;
    if (s >= num_seeds) return;
    int u = seeds[s];
    int u_start = offsets[u], u_end = offsets[u + 1];
    int64_t base = pair_offsets[s];
    int64_t cumulative = 0;

    for (int i = u_start; i < u_end; i++) {
        int w = indices[i];
        int w_start = offsets[w], w_end = offsets[w + 1];
        int w_deg = w_end - w_start;
        for (int j = threadIdx.x; j < w_deg; j += blockDim.x) {
            keys[base + cumulative + j] = ((uint64_t)(uint32_t)u << 32) | (uint64_t)(uint32_t)indices[w_start + j];
        }
        cumulative += w_deg;
    }
}

struct IsSelfPair {
    __device__ bool operator()(uint64_t key) const {
        return (uint32_t)(key >> 32) == (uint32_t)(key & 0xFFFFFFFFULL);
    }
};

__device__ __forceinline__ int lower_bound_dev(const int32_t* arr, int size, int32_t target) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ double warp_reduce_sum(double val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}


__global__ void jaccard_kernel_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const uint64_t* __restrict__ keys,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    double* __restrict__ out_scores,
    int64_t num_pairs
) {
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    uint64_t key = keys[warp_id];
    int32_t u = (int32_t)(key >> 32);
    int32_t v = (int32_t)(key & 0xFFFFFFFFULL);

    if (lane == 0) {
        out_first[warp_id] = u;
        out_second[warp_id] = v;
    }

    int u_start = offsets[u], u_end = offsets[u + 1], u_deg = u_end - u_start;
    int v_start = offsets[v], v_end = offsets[v + 1], v_deg = v_end - v_start;

    
    double local_sum = 0.0;
    for (int i = u_start + lane; i < u_end; i += 32)
        local_sum += edge_weights[i];
    double sum_u = warp_reduce_sum(local_sum);
    sum_u = __shfl_sync(0xffffffff, sum_u, 0);

    local_sum = 0.0;
    for (int i = v_start + lane; i < v_end; i += 32)
        local_sum += edge_weights[i];
    double sum_v = warp_reduce_sum(local_sum);
    sum_v = __shfl_sync(0xffffffff, sum_v, 0);

    
    const int32_t* small_nbrs, *large_nbrs;
    const double* small_wts, *large_wts;
    int small_deg, large_deg;

    if (u_deg <= v_deg) {
        small_nbrs = indices + u_start; small_wts = edge_weights + u_start; small_deg = u_deg;
        large_nbrs = indices + v_start; large_wts = edge_weights + v_start; large_deg = v_deg;
    } else {
        small_nbrs = indices + v_start; small_wts = edge_weights + v_start; small_deg = v_deg;
        large_nbrs = indices + u_start; large_wts = edge_weights + u_start; large_deg = u_deg;
    }

    double local_intersection = 0.0;

    if (small_deg > 0 && large_deg > 0) {
        int32_t small_min = small_nbrs[0];
        int32_t small_max = small_nbrs[small_deg - 1];
        int32_t large_min = large_nbrs[0];
        int32_t large_max = large_nbrs[large_deg - 1];

        if (small_min <= large_max && large_min <= small_max) {
            int search_lo = lower_bound_dev(large_nbrs, large_deg, small_min);
            int search_hi = large_deg;
            if (small_max < large_max) {
                search_hi = lower_bound_dev(large_nbrs + search_lo, large_deg - search_lo, small_max + 1) + search_lo;
            }

            for (int i = lane; i < small_deg; i += 32) {
                int32_t target = small_nbrs[i];
                if (target < large_min || target > large_max) continue;

                int lo = search_lo, hi = search_hi;
                while (lo < hi) {
                    int mid = lo + ((hi - lo) >> 1);
                    if (large_nbrs[mid] < target) lo = mid + 1;
                    else hi = mid;
                }

                if (lo < search_hi && large_nbrs[lo] == target) {
                    local_intersection += fmin(small_wts[i], large_wts[lo]);
                }
            }
        }
    }

    double intersection = warp_reduce_sum(local_intersection);

    if (lane == 0) {
        double denom = sum_u + sum_v - intersection;
        out_scores[warp_id] = (denom > 0.0) ? intersection / denom : 0.0;
    }
}

__global__ void fill_seq_kernel(int32_t* data, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = idx;
}

}  

similarity_result_double_t jaccard_all_pairs_similarity(const graph32_t& graph,
                                                        const double* edge_weights,
                                                        const int32_t* vertices,
                                                        std::size_t num_vertices,
                                                        std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t graph_num_vertices = graph.number_of_vertices;

    similarity_result_double_t result = {nullptr, nullptr, nullptr, 0};

    
    const int32_t* d_seeds = nullptr;
    int32_t num_seeds;
    bool allocated_seeds = false;

    if (vertices == nullptr) {
        num_seeds = graph_num_vertices;
        cudaMalloc(&d_seeds, (size_t)graph_num_vertices * sizeof(int32_t));
        allocated_seeds = true;
        fill_seq_kernel<<<(graph_num_vertices + 255) / 256, 256>>>(
            const_cast<int32_t*>(d_seeds), graph_num_vertices);
    } else {
        num_seeds = static_cast<int32_t>(num_vertices);
        d_seeds = vertices;
    }

    if (num_seeds == 0) {
        if (allocated_seeds) cudaFree(const_cast<int32_t*>(d_seeds));
        return result;
    }

    const int BLK = 256;

    
    int64_t* d_counts;
    cudaMalloc(&d_counts, (size_t)(num_seeds + 1) * sizeof(int64_t));
    cudaMemset(d_counts, 0, sizeof(int64_t));
    count_pairs_kernel<<<(num_seeds + BLK - 1) / BLK, BLK>>>(
        d_offsets, d_indices, d_seeds, num_seeds, d_counts + 1);

    
    thrust::inclusive_scan(thrust::device,
        thrust::device_pointer_cast(d_counts + 1),
        thrust::device_pointer_cast(d_counts + 1 + num_seeds),
        thrust::device_pointer_cast(d_counts + 1));

    
    int64_t total_raw;
    cudaMemcpy(&total_raw, d_counts + num_seeds, sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (total_raw <= 0) {
        cudaFree(d_counts);
        if (allocated_seeds) cudaFree(const_cast<int32_t*>(d_seeds));
        return result;
    }

    
    uint64_t* d_keys;
    cudaMalloc(&d_keys, (size_t)total_raw * sizeof(uint64_t));
    gen_pairs_kernel<<<num_seeds, BLK>>>(
        d_offsets, d_indices, d_seeds, num_seeds, d_counts, d_keys);
    cudaFree(d_counts);

    if (allocated_seeds) cudaFree(const_cast<int32_t*>(d_seeds));

    
    thrust::sort(thrust::device,
        thrust::device_pointer_cast(d_keys),
        thrust::device_pointer_cast(d_keys + total_raw));

    
    auto unique_end = thrust::unique(thrust::device,
        thrust::device_pointer_cast(d_keys),
        thrust::device_pointer_cast(d_keys + total_raw));
    int64_t num_unique = unique_end - thrust::device_pointer_cast(d_keys);

    
    auto non_self_end = thrust::remove_if(thrust::device,
        thrust::device_pointer_cast(d_keys),
        thrust::device_pointer_cast(d_keys + num_unique),
        IsSelfPair());
    int64_t num_pairs = non_self_end - thrust::device_pointer_cast(d_keys);

    if (num_pairs == 0) {
        cudaFree(d_keys);
        return result;
    }

    
    cudaMalloc(&result.first, (size_t)num_pairs * sizeof(int32_t));
    cudaMalloc(&result.second, (size_t)num_pairs * sizeof(int32_t));
    cudaMalloc(&result.scores, (size_t)num_pairs * sizeof(double));

    
    int64_t warps_needed = num_pairs;
    int64_t threads_total = warps_needed * 32;
    int blocks_needed = (int)((threads_total + BLK - 1) / BLK);

    jaccard_kernel_warp<<<blocks_needed, BLK>>>(
        d_offsets, d_indices, edge_weights,
        d_keys, result.first, result.second, result.scores, num_pairs);

    cudaFree(d_keys);

    
    if (topk.has_value() && (int64_t)topk.value() < num_pairs) {
        thrust::sort_by_key(thrust::device,
            thrust::device_pointer_cast(result.scores),
            thrust::device_pointer_cast(result.scores + num_pairs),
            thrust::make_zip_iterator(thrust::make_tuple(
                thrust::device_pointer_cast(result.first),
                thrust::device_pointer_cast(result.second))),
            thrust::greater<double>());
        num_pairs = (int64_t)topk.value();
    }

    result.count = (std::size_t)num_pairs;
    return result;
}

}  
