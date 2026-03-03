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
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <cstdint>
#include <climits>
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* d_weighted_deg = nullptr;
    int32_t* d_seeds_all = nullptr;
    size_t weighted_deg_cap = 0;
    size_t seeds_all_cap = 0;

    ~Cache() override {
        if (d_weighted_deg) cudaFree(d_weighted_deg);
        if (d_seeds_all) cudaFree(d_seeds_all);
    }
};




__global__ void compute_weighted_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ weights,
    float* __restrict__ weighted_deg,
    int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int start = offsets[v];
    int end = offsets[v + 1];
    float sum = 0.0f;
    for (int i = start; i < end; i++) sum += weights[i];
    weighted_deg[v] = sum;
}




__global__ void count_raw_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    int64_t* __restrict__ counts
) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_seeds) return;

    int u = seeds[sid];
    int u_start = offsets[u];
    int u_end = offsets[u + 1];

    int64_t count = 0;
    for (int i = u_start; i < u_end; i++) {
        int w = indices[i];
        count += offsets[w + 1] - offsets[w];
    }
    counts[sid] = count;
}




__global__ void fill_raw_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    const int64_t* __restrict__ prefix_sums,
    int64_t* __restrict__ keys,
    int64_t max_v_plus_1
) {
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int u = seeds[sid];
    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int u_deg = u_end - u_start;

    int64_t base = prefix_sums[sid];
    int64_t offset = 0;

    for (int ni = 0; ni < u_deg; ni++) {
        int w = indices[u_start + ni];
        int w_start = offsets[w];
        int w_end = offsets[w + 1];
        int w_deg = w_end - w_start;

        for (int nj = threadIdx.x; nj < w_deg; nj += blockDim.x) {
            int v = indices[w_start + nj];
            int64_t key;
            if (v == u) {
                key = INT64_MAX;  
            } else {
                key = (int64_t)sid * max_v_plus_1 + (int64_t)v;
            }
            keys[base + offset + nj] = key;
        }
        offset += w_deg;
    }
}




__global__ void compute_intersection_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ weighted_deg,
    const int32_t* __restrict__ seeds,
    const int64_t* __restrict__ unique_keys,
    int64_t num_unique_pairs,
    int64_t max_v_plus_1,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores
) {
    
    int64_t warp_global_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane_id = threadIdx.x % 32;

    if (warp_global_id >= num_unique_pairs) return;

    int64_t key = unique_keys[warp_global_id];
    int sid = (int)(key / max_v_plus_1);
    int v = (int)(key % max_v_plus_1);
    int u = seeds[sid];

    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int u_deg = u_end - u_start;
    int v_start = offsets[v];
    int v_end = offsets[v + 1];
    int v_deg = v_end - v_start;

    float u_wdeg = weighted_deg[u];
    float v_wdeg = weighted_deg[v];

    
    int s_start, s_end, l_start, l_end, s_deg, l_deg;
    if (u_deg <= v_deg) {
        s_start = u_start; s_end = u_end; s_deg = u_deg;
        l_start = v_start; l_end = v_end; l_deg = v_deg;
    } else {
        s_start = v_start; s_end = v_end; s_deg = v_deg;
        l_start = u_start; l_end = u_end; l_deg = u_deg;
    }

    
    float intersection = 0.0f;

    for (int batch = 0; batch < s_deg; batch += 32) {
        int idx = batch + lane_id;
        float contrib = 0.0f;

        if (idx < s_deg) {
            int target = indices[s_start + idx];
            float s_weight = weights[s_start + idx];

            
            int lo = l_start, hi = l_end;
            while (lo < hi) {
                int mid = lo + (hi - lo) / 2;
                if (indices[mid] < target) lo = mid + 1; else hi = mid;
            }

            if (lo < l_end && indices[lo] == target) {
                contrib = fminf(s_weight, weights[lo]);
            }
        }

        
        #pragma unroll
        for (int off = 16; off > 0; off /= 2) {
            contrib += __shfl_down_sync(0xffffffff, contrib, off);
        }

        if (lane_id == 0) intersection += contrib;
    }

    
    if (lane_id == 0) {
        float denom = u_wdeg + v_wdeg;
        float score = (denom > 0.0f) ? (2.0f * intersection / denom) : 0.0f;
        out_first[warp_global_id] = u;
        out_second[warp_global_id] = v;
        out_scores[warp_global_id] = score;
    }
}




__global__ void iota_kernel(int32_t* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = i;
}

}  

similarity_result_float_t sorensen_all_pairs_similarity(const graph32_t& graph,
                                                        const float* edge_weights,
                                                        const int32_t* vertices,
                                                        std::size_t num_vertices,
                                                        std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n_vertices = graph.number_of_vertices;
    const float* d_weights = edge_weights;

    cudaStream_t stream = 0;

    
    if ((size_t)n_vertices > cache.weighted_deg_cap) {
        if (cache.d_weighted_deg) cudaFree(cache.d_weighted_deg);
        cudaMalloc(&cache.d_weighted_deg, (size_t)n_vertices * sizeof(float));
        cache.weighted_deg_cap = n_vertices;
    }
    {
        int block = 256;
        int grid = (n_vertices + block - 1) / block;
        if (grid > 0)
            compute_weighted_degrees_kernel<<<grid, block, 0, stream>>>(
                d_offsets, d_weights, cache.d_weighted_deg, n_vertices);
    }

    
    const int32_t* d_seeds;
    int actual_num_seeds;
    if (vertices == nullptr) {
        actual_num_seeds = n_vertices;
        if ((size_t)n_vertices > cache.seeds_all_cap) {
            if (cache.d_seeds_all) cudaFree(cache.d_seeds_all);
            cudaMalloc(&cache.d_seeds_all, (size_t)n_vertices * sizeof(int32_t));
            cache.seeds_all_cap = n_vertices;
        }
        {
            int block = 256;
            int grid = (n_vertices + block - 1) / block;
            if (grid > 0)
                iota_kernel<<<grid, block, 0, stream>>>(cache.d_seeds_all, n_vertices);
        }
        d_seeds = cache.d_seeds_all;
    } else {
        d_seeds = vertices;
        actual_num_seeds = (int)num_vertices;
    }

    
    int64_t* d_counts = nullptr;
    int64_t* d_prefix = nullptr;
    cudaMalloc(&d_counts, (size_t)actual_num_seeds * sizeof(int64_t));
    cudaMalloc(&d_prefix, (size_t)actual_num_seeds * sizeof(int64_t));

    {
        int block = 256;
        int grid = (actual_num_seeds + block - 1) / block;
        if (grid > 0)
            count_raw_pairs_kernel<<<grid, block, 0, stream>>>(
                d_offsets, d_indices, d_seeds, actual_num_seeds, d_counts);
    }

    
    size_t scan_temp_size = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_size,
        (int64_t*)nullptr, (int64_t*)nullptr, actual_num_seeds);

    void* d_scan_temp = nullptr;
    cudaMalloc(&d_scan_temp, scan_temp_size);

    cub::DeviceScan::ExclusiveSum(d_scan_temp, scan_temp_size,
        d_counts, d_prefix, actual_num_seeds, stream);

    cudaFree(d_scan_temp);

    
    int64_t last_prefix = 0, last_count = 0;
    if (actual_num_seeds > 0) {
        cudaMemcpyAsync(&last_prefix, d_prefix + actual_num_seeds - 1,
            sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&last_count, d_counts + actual_num_seeds - 1,
            sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);
    int64_t total_raw = last_prefix + last_count;

    if (total_raw == 0) {
        cudaFree(d_counts);
        cudaFree(d_prefix);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t max_v_plus_1 = (int64_t)n_vertices + 1;
    int64_t* d_keys = nullptr;
    cudaMalloc(&d_keys, (size_t)total_raw * sizeof(int64_t));

    if (actual_num_seeds > 0)
        fill_raw_pairs_kernel<<<actual_num_seeds, 256, 0, stream>>>(
            d_offsets, d_indices, d_seeds, actual_num_seeds,
            d_prefix, d_keys, max_v_plus_1);

    cudaFree(d_counts);
    cudaFree(d_prefix);

    
    int64_t* d_sorted_keys = nullptr;
    cudaMalloc(&d_sorted_keys, (size_t)total_raw * sizeof(int64_t));

    size_t sort_temp_size = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, sort_temp_size,
        d_keys, d_sorted_keys, total_raw, 0, 63, stream);

    void* d_sort_temp = nullptr;
    cudaMalloc(&d_sort_temp, sort_temp_size);

    cub::DeviceRadixSort::SortKeys(d_sort_temp, sort_temp_size,
        d_keys, d_sorted_keys, total_raw, 0, 63, stream);

    cudaFree(d_sort_temp);
    cudaFree(d_keys);

    
    thrust::device_ptr<int64_t> begin(d_sorted_keys);
    thrust::device_ptr<int64_t> end_ptr = thrust::unique(
        thrust::cuda::par.on(stream), begin, begin + total_raw);

    int64_t num_unique = end_ptr - begin;

    
    while (num_unique > 0) {
        int64_t last_val;
        cudaMemcpyAsync(&last_val, d_sorted_keys + num_unique - 1,
            sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (last_val == INT64_MAX) {
            num_unique--;
        } else {
            break;
        }
    }

    if (num_unique <= 0) {
        cudaFree(d_sorted_keys);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;
    cudaMalloc(&out_first, (size_t)num_unique * sizeof(int32_t));
    cudaMalloc(&out_second, (size_t)num_unique * sizeof(int32_t));
    cudaMalloc(&out_scores, (size_t)num_unique * sizeof(float));

    {
        int threads_per_block = 256;
        int warps_per_block = threads_per_block / 32;
        int64_t grid64 = (num_unique + warps_per_block - 1) / warps_per_block;
        int grid = (int)min(grid64, (int64_t)INT_MAX);
        compute_intersection_kernel<<<grid, threads_per_block, 0, stream>>>(
            d_offsets, d_indices, d_weights, cache.d_weighted_deg, d_seeds,
            d_sorted_keys, num_unique, max_v_plus_1,
            out_first, out_second, out_scores);
    }

    cudaFree(d_sorted_keys);

    
    int64_t result_count = num_unique;
    if (topk.has_value() && (int64_t)topk.value() < num_unique) {
        thrust::device_ptr<float> scores_ptr(out_scores);
        thrust::device_ptr<int32_t> first_ptr(out_first);
        thrust::device_ptr<int32_t> second_ptr(out_second);
        auto values = thrust::make_zip_iterator(thrust::make_tuple(first_ptr, second_ptr));

        thrust::transform(thrust::cuda::par.on(stream),
            scores_ptr, scores_ptr + num_unique, scores_ptr, thrust::negate<float>());
        thrust::sort_by_key(thrust::cuda::par.on(stream),
            scores_ptr, scores_ptr + num_unique, values);
        thrust::transform(thrust::cuda::par.on(stream),
            scores_ptr, scores_ptr + num_unique, scores_ptr, thrust::negate<float>());

        result_count = (int64_t)topk.value();
    }

    
    if (result_count == num_unique) {
        cudaStreamSynchronize(stream);
        return {out_first, out_second, out_scores, (std::size_t)result_count};
    }

    
    int32_t* r1 = nullptr;
    int32_t* r2 = nullptr;
    float* r3 = nullptr;
    cudaMalloc(&r1, (size_t)result_count * sizeof(int32_t));
    cudaMalloc(&r2, (size_t)result_count * sizeof(int32_t));
    cudaMalloc(&r3, (size_t)result_count * sizeof(float));

    if (result_count > 0) {
        cudaMemcpyAsync(r1, out_first,
            result_count * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(r2, out_second,
            result_count * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(r3, out_scores,
            result_count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    cudaFree(out_first);
    cudaFree(out_second);
    cudaFree(out_scores);

    return {r1, r2, r3, (std::size_t)result_count};
}

}  
