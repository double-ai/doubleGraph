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
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <cstdint>
#include <optional>

namespace aai {

namespace {

struct Cache : Cacheable {};




__global__ void count_expanded_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;

    int32_t u = seeds[idx];
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];

    int64_t count = 0;
    for (int32_t i = u_start; i < u_end; i++) {
        int32_t w = indices[i];
        count += (int64_t)(offsets[w + 1] - offsets[w]);
    }
    counts[idx] = count;
}




__global__ void expand_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ write_offsets,
    int64_t* __restrict__ out_keys
) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    int32_t u = seeds[seed_idx];
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];

    int64_t base = write_offsets[seed_idx];

    int64_t local_offset = 0;
    for (int32_t i = u_start; i < u_end; i++) {
        int32_t w = indices[i];
        int32_t w_start = offsets[w];
        int32_t w_end = offsets[w + 1];
        int32_t w_deg = w_end - w_start;

        for (int32_t j = threadIdx.x; j < w_deg; j += blockDim.x) {
            int32_t v = indices[w_start + j];
            out_keys[base + local_offset + j] = ((int64_t)u << 32) | ((int64_t)(uint32_t)v);
        }
        local_offset += w_deg;
    }
}




__device__ __forceinline__ int lower_bound_dev(const int32_t* arr, int size, int32_t target) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}




__device__ __forceinline__ int galloping_search(
    const int32_t* arr, int start, int end, int32_t target
) {
    if (start >= end) return end;
    if (arr[start] >= target) {
        int lo = start, hi = start;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (arr[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
    int pos = start;
    int step = 1;
    while (pos + step < end && arr[pos + step] < target) {
        pos += step;
        step *= 2;
    }
    int lo = pos;
    int hi = (pos + step + 1 < end) ? pos + step + 1 : end;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}




__device__ int intersect_count_optimized(
    const int32_t* a, int size_a,
    const int32_t* b, int size_b
) {
    if (size_a == 0 || size_b == 0) return 0;

    const int32_t* smaller;
    const int32_t* larger;
    int size_smaller, size_larger;
    if (size_a <= size_b) {
        smaller = a; size_smaller = size_a;
        larger = b; size_larger = size_b;
    } else {
        smaller = b; size_smaller = size_b;
        larger = a; size_larger = size_a;
    }

    int ratio = size_larger / (size_smaller > 0 ? size_smaller : 1);

    if (ratio > 10) {
        int count = 0;
        int j = 0;
        for (int i = 0; i < size_smaller && j < size_larger; i++) {
            int32_t target = smaller[i];
            j = galloping_search(larger, j, size_larger, target);
            if (j < size_larger && larger[j] == target) {
                count++;
                j++;
            }
        }
        return count;
    }

    int start_s = lower_bound_dev(smaller, size_smaller, larger[0]);
    if (start_s >= size_smaller) return 0;
    int start_l = lower_bound_dev(larger, size_larger, smaller[start_s]);
    if (start_l >= size_larger) return 0;

    int32_t max_s = smaller[size_smaller - 1];
    int32_t max_l = larger[size_larger - 1];
    int32_t min_max = (max_s < max_l) ? max_s : max_l;

    int count = 0;
    int i = start_s, j = start_l;
    while (i < size_smaller && j < size_larger) {
        int32_t va = smaller[i];
        int32_t vb = larger[j];
        if (va == vb) {
            count++;
            i++;
            j++;
        } else if (va < vb) {
            i++;
        } else {
            j++;
        }
        if (va > min_max || vb > min_max) break;
    }
    return count;
}




__global__ void compute_jaccard_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pair_first,
    const int32_t* __restrict__ pair_second,
    int64_t num_pairs,
    float* __restrict__ scores
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int32_t u = pair_first[idx];
    int32_t v = pair_second[idx];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];

    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;

    int intersection = intersect_count_optimized(
        indices + u_start, deg_u,
        indices + v_start, deg_v);

    int32_t union_size = deg_u + deg_v - intersection;
    scores[idx] = (union_size > 0) ? ((float)intersection / (float)union_size) : 0.0f;
}

}  

similarity_result_float_t jaccard_all_pairs_similarity(const graph32_t& graph,
                                                       const int32_t* vertices,
                                                       std::size_t num_vertices,
                                                       std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t nv = graph.number_of_vertices;

    
    int32_t* d_seeds_alloc = nullptr;
    const int32_t* d_seeds;
    int32_t num_seeds;

    if (vertices == nullptr || num_vertices == 0) {
        num_seeds = nv;
        cudaMalloc(&d_seeds_alloc, (size_t)num_seeds * sizeof(int32_t));
        thrust::sequence(thrust::device,
                        thrust::device_pointer_cast(d_seeds_alloc),
                        thrust::device_pointer_cast(d_seeds_alloc) + num_seeds);
        d_seeds = d_seeds_alloc;
    } else {
        d_seeds = vertices;
        num_seeds = (int32_t)num_vertices;
    }

    
    int64_t* d_counts;
    cudaMalloc(&d_counts, (size_t)num_seeds * sizeof(int64_t));

    {
        int block = 256;
        int grid = (num_seeds + block - 1) / block;
        count_expanded_kernel<<<grid, block>>>(d_offsets, d_indices, d_seeds, num_seeds, d_counts);
    }

    
    int64_t* d_write_offsets;
    cudaMalloc(&d_write_offsets, (size_t)(num_seeds + 1) * sizeof(int64_t));
    cudaMemset(d_write_offsets, 0, sizeof(int64_t));

    thrust::inclusive_scan(thrust::device,
                          thrust::device_pointer_cast(d_counts),
                          thrust::device_pointer_cast(d_counts) + num_seeds,
                          thrust::device_pointer_cast(d_write_offsets + 1));

    int64_t total_expanded;
    cudaMemcpy(&total_expanded, d_write_offsets + num_seeds, sizeof(int64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_counts);

    if (total_expanded <= 0) {
        cudaFree(d_write_offsets);
        if (d_seeds_alloc) cudaFree(d_seeds_alloc);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t* d_keys;
    cudaMalloc(&d_keys, total_expanded * sizeof(int64_t));

    {
        int block = 256;
        int grid = num_seeds;
        expand_pairs_kernel<<<grid, block>>>(d_offsets, d_indices, d_seeds, num_seeds, d_write_offsets, d_keys);
    }

    cudaFree(d_write_offsets);

    
    thrust::sort(thrust::device,
                thrust::device_pointer_cast(d_keys),
                thrust::device_pointer_cast(d_keys) + total_expanded);

    auto unique_end = thrust::unique(thrust::device,
                                     thrust::device_pointer_cast(d_keys),
                                     thrust::device_pointer_cast(d_keys) + total_expanded);
    int64_t unique_count = unique_end - thrust::device_pointer_cast(d_keys);

    
    auto non_self_end = thrust::remove_if(thrust::device,
                                           thrust::device_pointer_cast(d_keys),
                                           thrust::device_pointer_cast(d_keys) + unique_count,
                                           [] __device__ (int64_t key) {
                                               int32_t u = (int32_t)(key >> 32);
                                               int32_t v = (int32_t)(key & 0xFFFFFFFF);
                                               return u == v;
                                           });
    int64_t num_pairs = non_self_end - thrust::device_pointer_cast(d_keys);

    if (num_pairs == 0) {
        cudaFree(d_keys);
        if (d_seeds_alloc) cudaFree(d_seeds_alloc);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t* d_first;
    int32_t* d_second;
    float* d_scores;
    cudaMalloc(&d_first, num_pairs * sizeof(int32_t));
    cudaMalloc(&d_second, num_pairs * sizeof(int32_t));
    cudaMalloc(&d_scores, num_pairs * sizeof(float));

    thrust::transform(thrust::device,
                     thrust::device_pointer_cast(d_keys),
                     thrust::device_pointer_cast(d_keys) + num_pairs,
                     thrust::device_pointer_cast(d_first),
                     [] __device__ (int64_t key) -> int32_t {
                         return (int32_t)(key >> 32);
                     });

    thrust::transform(thrust::device,
                     thrust::device_pointer_cast(d_keys),
                     thrust::device_pointer_cast(d_keys) + num_pairs,
                     thrust::device_pointer_cast(d_second),
                     [] __device__ (int64_t key) -> int32_t {
                         return (int32_t)(key & 0xFFFFFFFF);
                     });

    cudaFree(d_keys);

    
    {
        int block = 256;
        int grid = (int)((num_pairs + block - 1) / block);
        compute_jaccard_kernel<<<grid, block>>>(d_offsets, d_indices, d_first, d_second, num_pairs, d_scores);
    }

    
    if (topk.has_value() && num_pairs > (int64_t)topk.value()) {
        auto key_begin = thrust::device_pointer_cast(d_scores);
        auto val_begin = thrust::make_zip_iterator(
            thrust::make_tuple(
                thrust::device_pointer_cast(d_first),
                thrust::device_pointer_cast(d_second)
            ));
        thrust::sort_by_key(thrust::device, key_begin, key_begin + num_pairs,
                           val_begin, thrust::greater<float>());
        num_pairs = (int64_t)topk.value();
    }

    if (d_seeds_alloc) cudaFree(d_seeds_alloc);

    return {d_first, d_second, d_scores, (std::size_t)num_pairs};
}

}  
