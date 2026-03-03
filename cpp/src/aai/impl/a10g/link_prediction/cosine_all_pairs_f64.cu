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
#include <cstddef>
#include <algorithm>
#include <optional>
#include <cmath>
#include <cub/cub.cuh>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

namespace aai {

namespace {




__global__ void iota_kernel(int32_t* data, int32_t n) {
    for (int32_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
        data[idx] = idx;
}




__global__ void compute_work_offsets_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int32_t* __restrict__ work_offsets
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int32_t sum = 0;
        for (int i = 0; i < num_seeds; i++) {
            work_offsets[i] = sum;
            int32_t u = seeds[i];
            sum += offsets[u + 1] - offsets[u];
        }
        work_offsets[num_seeds] = sum;
    }
}





__global__ void expand_grid_stride(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    uint64_t num_vertices,
    const int32_t* __restrict__ work_offsets,
    uint64_t* __restrict__ pair_keys,
    int32_t* __restrict__ write_counter,
    int32_t max_pairs
) {
    int32_t total_work = work_offsets[num_seeds];

    for (int32_t work_idx = blockIdx.x; work_idx < total_work; work_idx += gridDim.x) {
        
        int32_t lo = 0, hi = num_seeds;
        while (lo < hi) {
            int32_t mid = (lo + hi + 1) >> 1;
            if (work_offsets[mid] <= work_idx) lo = mid;
            else hi = mid - 1;
        }
        int32_t seed_idx = lo;
        int32_t u = seeds[seed_idx];
        int32_t ki_local = work_idx - work_offsets[seed_idx];

        int32_t u_start = offsets[u];
        int32_t k = indices[u_start + ki_local];
        int32_t k_start = offsets[k];
        int32_t k_end = offsets[k + 1];
        uint64_t base_key = (uint64_t)seed_idx * num_vertices;

        for (int32_t j = k_start + threadIdx.x; j < k_end; j += blockDim.x) {
            int32_t v = indices[j];
            if (v != u) {
                int32_t pos = atomicAdd(write_counter, 1);
                if (pos < max_pairs) {
                    pair_keys[pos] = base_key + (uint64_t)v;
                }
            }
        }
    }
}




__device__ __forceinline__ int32_t lower_bound_dev(
    const int32_t* arr, int32_t size, int32_t target
) {
    int32_t lo = 0, hi = size;
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}




__global__ void compute_cosine_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint64_t* __restrict__ pair_keys,
    int64_t num_pairs,
    uint64_t num_vertices,
    const int32_t* __restrict__ seeds,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    double* __restrict__ out_scores
) {
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    uint64_t key = pair_keys[warp_id];
    int32_t seed_idx = (int32_t)(key / num_vertices);
    int32_t v = (int32_t)(key % num_vertices);
    int32_t u = seeds[seed_idx];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];
    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    const int32_t* small_nbrs, *large_nbrs;
    const double* small_wts, *large_wts;
    int32_t small_size, large_size;
    bool u_is_small;

    if (u_deg <= v_deg) {
        small_nbrs = indices + u_start; large_nbrs = indices + v_start;
        small_wts = weights + u_start;  large_wts = weights + v_start;
        small_size = u_deg; large_size = v_deg;
        u_is_small = true;
    } else {
        small_nbrs = indices + v_start; large_nbrs = indices + u_start;
        small_wts = weights + v_start;  large_wts = weights + u_start;
        small_size = v_deg; large_size = u_deg;
        u_is_small = false;
    }

    double local_dot = 0.0, local_nu = 0.0, local_nv = 0.0;

    for (int32_t i = lane; i < small_size; i += 32) {
        int32_t target = small_nbrs[i];
        double w_small = small_wts[i];

        int32_t pos = lower_bound_dev(large_nbrs, large_size, target);

        if (pos < large_size && large_nbrs[pos] == target) {
            double w_large = large_wts[pos];
            double wu = u_is_small ? w_small : w_large;
            double wv = u_is_small ? w_large : w_small;
            local_dot += wu * wv;
            local_nu += wu * wu;
            local_nv += wv * wv;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_dot += __shfl_down_sync(0xffffffff, local_dot, offset);
        local_nu += __shfl_down_sync(0xffffffff, local_nu, offset);
        local_nv += __shfl_down_sync(0xffffffff, local_nv, offset);
    }

    if (lane == 0) {
        double denom = sqrt(local_nu) * sqrt(local_nv);
        out_scores[warp_id] = (denom > 0.0) ? local_dot / denom : 0.0;
        out_first[warp_id] = u;
        out_second[warp_id] = v;
    }
}


__global__ void compute_cosine_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint64_t* __restrict__ pair_keys,
    int64_t num_pairs,
    uint64_t num_vertices,
    const int32_t* __restrict__ seeds,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    double* __restrict__ out_scores
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    uint64_t key = pair_keys[idx];
    int32_t seed_idx = (int32_t)(key / num_vertices);
    int32_t v = (int32_t)(key % num_vertices);
    int32_t u = seeds[seed_idx];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];
    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        out_scores[idx] = 0.0; out_first[idx] = u; out_second[idx] = v; return;
    }

    const int32_t* u_nbrs = indices + u_start;
    const int32_t* v_nbrs = indices + v_start;
    const double* u_wts = weights + u_start;
    const double* v_wts = weights + v_start;

    int32_t i = lower_bound_dev(u_nbrs, u_deg, v_nbrs[0]);
    int32_t j = (i < u_deg) ? lower_bound_dev(v_nbrs, v_deg, u_nbrs[i]) : v_deg;

    double dot = 0.0, nu = 0.0, nv = 0.0;
    while (i < u_deg && j < v_deg) {
        int32_t ui = u_nbrs[i], vj = v_nbrs[j];
        if (ui == vj) {
            double wu = u_wts[i], wv = v_wts[j];
            dot += wu * wv; nu += wu * wu; nv += wv * wv;
            i++; j++;
        } else if (ui < vj) i++;
        else j++;
    }

    double denom = sqrt(nu) * sqrt(nv);
    out_scores[idx] = (denom > 0.0) ? dot / denom : 0.0;
    out_first[idx] = u; out_second[idx] = v;
}





void launch_iota(int32_t* data, int32_t n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256, grid = (n + block - 1) / block;
    if (grid > 2048) grid = 2048;
    iota_kernel<<<grid, block, 0, stream>>>(data, n);
}

void launch_compute_work_offsets(const int32_t* offsets, const int32_t* seeds,
                                 int32_t num_seeds, int32_t* work_offsets, cudaStream_t stream) {
    if (num_seeds == 0) return;
    compute_work_offsets_kernel<<<1, 1, 0, stream>>>(offsets, seeds, num_seeds, work_offsets);
}

void launch_expand(const int32_t* offsets, const int32_t* indices,
                   const int32_t* seeds, int32_t num_seeds, uint64_t num_vertices,
                   const int32_t* work_offsets,
                   uint64_t* pair_keys, int32_t* write_counter, int32_t max_pairs,
                   cudaStream_t stream) {
    cudaMemsetAsync(write_counter, 0, sizeof(int32_t), stream);
    int grid = 2560, block = 128;
    expand_grid_stride<<<grid, block, 0, stream>>>(
        offsets, indices, seeds, num_seeds, num_vertices,
        work_offsets, pair_keys, write_counter, max_pairs);
}

size_t get_sort_temp_bytes(int32_t max_items) {
    size_t temp = 0;
    cub::DoubleBuffer<uint64_t> d_keys;
    cub::DeviceRadixSort::SortKeys(nullptr, temp, d_keys, max_items, 0, 64);
    return temp;
}

void launch_cub_sort(void* d_temp, size_t temp_bytes,
                     uint64_t* d_buf1, uint64_t* d_buf2,
                     int32_t num_items, int end_bit, int* selector_out,
                     cudaStream_t stream) {
    cub::DoubleBuffer<uint64_t> d_keys(d_buf1, d_buf2);
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys, num_items, 0, end_bit, stream);
    *selector_out = d_keys.selector;
}

size_t get_unique_temp_bytes(int32_t max_items) {
    size_t temp = 0;
    cub::DeviceSelect::Unique(nullptr, temp, (uint64_t*)nullptr, (uint64_t*)nullptr, (int32_t*)nullptr, max_items);
    return temp;
}

void launch_cub_unique(void* d_temp, size_t temp_bytes,
                       const uint64_t* d_in, uint64_t* d_out, int32_t* d_num_selected,
                       int32_t num_items, cudaStream_t stream) {
    cub::DeviceSelect::Unique(d_temp, temp_bytes, d_in, d_out, d_num_selected, num_items, stream);
}

void launch_compute_cosine(
    const int32_t* offsets, const int32_t* indices, const double* weights,
    const uint64_t* pair_keys, int64_t num_pairs, uint64_t num_vertices,
    const int32_t* seeds,
    int32_t* out_first, int32_t* out_second, double* out_scores,
    bool use_warp, cudaStream_t stream
) {
    if (num_pairs == 0) return;
    if (use_warp) {
        int warps_per_block = 8;
        int tpb = warps_per_block * 32;
        int grid = (int)((num_pairs + warps_per_block - 1) / warps_per_block);
        compute_cosine_warp<<<grid, tpb, 0, stream>>>(
            offsets, indices, weights, pair_keys, num_pairs, num_vertices, seeds,
            out_first, out_second, out_scores);
    } else {
        int block = 256;
        int grid = (int)((num_pairs + block - 1) / block);
        compute_cosine_thread<<<grid, block, 0, stream>>>(
            offsets, indices, weights, pair_keys, num_pairs, num_vertices, seeds,
            out_first, out_second, out_scores);
    }
}

void sort_scores_desc(double* scores, int32_t* first, int32_t* second,
                      int64_t count, cudaStream_t stream) {
    if (count <= 1) return;
    auto sp = thrust::device_pointer_cast(scores);
    auto fp = thrust::device_pointer_cast(first);
    auto sec = thrust::device_pointer_cast(second);
    auto vals = thrust::make_zip_iterator(thrust::make_tuple(fp, sec));
    thrust::sort_by_key(thrust::cuda::par.on(stream), sp, sp + count, vals, thrust::greater<double>());
}

int compute_end_bit(uint64_t max_key) {
    if (max_key == 0) return 1;
    int bits = 64 - __builtin_clzll(max_key);
    return bits;
}





static constexpr int32_t INITIAL_PAIR_CAPACITY = 32 * 1024 * 1024;

struct Cache : Cacheable {
    uint64_t* pair_buf1 = nullptr;
    uint64_t* pair_buf2 = nullptr;
    int32_t* counter = nullptr;
    int32_t* work_offsets = nullptr;
    int32_t* seeds_buf = nullptr;
    void* cub_temp = nullptr;

    size_t sort_temp_bytes = 0;
    size_t unique_temp_bytes = 0;
    size_t cub_temp_size = 0;
    int32_t pair_buf_capacity = 0;
    int32_t max_vertices = 0;
    int32_t work_offsets_capacity = 0;
    bool initialized = false;

    void ensure_pair_buffers(int32_t needed) {
        if (needed <= pair_buf_capacity) return;
        if (pair_buf1) cudaFree(pair_buf1);
        if (pair_buf2) cudaFree(pair_buf2);
        pair_buf_capacity = needed;
        cudaMalloc(&pair_buf1, (size_t)pair_buf_capacity * sizeof(uint64_t));
        cudaMalloc(&pair_buf2, (size_t)pair_buf_capacity * sizeof(uint64_t));

        sort_temp_bytes = get_sort_temp_bytes(pair_buf_capacity);
        unique_temp_bytes = get_unique_temp_bytes(pair_buf_capacity);
        size_t new_cub_size = (sort_temp_bytes > unique_temp_bytes) ? sort_temp_bytes : unique_temp_bytes;
        new_cub_size = ((new_cub_size + 255) / 256) * 256;
        if (new_cub_size > cub_temp_size) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, new_cub_size);
            cub_temp_size = new_cub_size;
        }
    }

    void ensure_work_offsets(int32_t num_seeds) {
        int32_t needed = num_seeds + 1;
        if (needed <= work_offsets_capacity) return;
        if (work_offsets) cudaFree(work_offsets);
        work_offsets_capacity = needed;
        cudaMalloc(&work_offsets, (size_t)work_offsets_capacity * sizeof(int32_t));
    }

    void lazy_init() {
        if (initialized) return;
        ensure_pair_buffers(INITIAL_PAIR_CAPACITY);
        cudaMalloc(&counter, 8);
        initialized = true;
    }

    ~Cache() override {
        if (pair_buf1) cudaFree(pair_buf1);
        if (pair_buf2) cudaFree(pair_buf2);
        if (counter) cudaFree(counter);
        if (work_offsets) cudaFree(work_offsets);
        if (seeds_buf) cudaFree(seeds_buf);
        if (cub_temp) cudaFree(cub_temp);
    }
};

}  

similarity_result_double_t cosine_all_pairs_similarity(const graph32_t& graph,
                                                       const double* edge_weights,
                                                       const int32_t* vertices,
                                                       std::size_t num_vertices,
                                                       std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    cache.lazy_init();

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n = graph.number_of_vertices;
    const double* d_weights = edge_weights;

    cudaStream_t stream = 0;

    int32_t num_seeds;
    const int32_t* d_seeds;

    if (vertices != nullptr) {
        num_seeds = (int32_t)num_vertices;
        d_seeds = vertices;
    } else {
        num_seeds = n;
        if (n > cache.max_vertices) {
            if (cache.seeds_buf) cudaFree(cache.seeds_buf);
            cudaMalloc(&cache.seeds_buf, (size_t)n * sizeof(int32_t));
            cache.max_vertices = n;
        }
        launch_iota(cache.seeds_buf, n, stream);
        d_seeds = cache.seeds_buf;
    }

    if (num_seeds == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    uint64_t max_key = (uint64_t)(num_seeds - 1) * (uint64_t)n + (uint64_t)(n - 1);
    int end_bit = compute_end_bit(max_key);

    
    cache.ensure_work_offsets(num_seeds);

    
    launch_compute_work_offsets(d_offsets, d_seeds, num_seeds, cache.work_offsets, stream);

    
    launch_expand(d_offsets, d_indices, d_seeds, num_seeds, (uint64_t)n,
                  cache.work_offsets, cache.pair_buf1, cache.counter, cache.pair_buf_capacity, stream);

    
    int32_t actual_count;
    cudaMemcpyAsync(&actual_count, cache.counter, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (actual_count == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    if (actual_count > cache.pair_buf_capacity) {
        cache.ensure_pair_buffers(actual_count);
        launch_expand(d_offsets, d_indices, d_seeds, num_seeds, (uint64_t)n,
                      cache.work_offsets, cache.pair_buf1, cache.counter, cache.pair_buf_capacity, stream);
    }

    
    int selector;
    launch_cub_sort(cache.cub_temp, cache.sort_temp_bytes,
                    cache.pair_buf1, cache.pair_buf2,
                    actual_count, end_bit, &selector, stream);

    uint64_t* d_sorted = (selector == 0) ? cache.pair_buf1 : cache.pair_buf2;
    uint64_t* d_unique_out = (selector == 0) ? cache.pair_buf2 : cache.pair_buf1;

    
    launch_cub_unique(cache.cub_temp, cache.unique_temp_bytes,
                      d_sorted, d_unique_out, cache.counter,
                      actual_count, stream);

    
    int32_t num_unique;
    cudaMemcpyAsync(&num_unique, cache.counter, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (num_unique == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    bool use_warp = (num_unique > 500);
    int32_t* d_first;
    int32_t* d_second;
    double* d_scores;
    cudaMalloc(&d_first, (size_t)num_unique * sizeof(int32_t));
    cudaMalloc(&d_second, (size_t)num_unique * sizeof(int32_t));
    cudaMalloc(&d_scores, (size_t)num_unique * sizeof(double));

    launch_compute_cosine(d_offsets, d_indices, d_weights,
                          d_unique_out, (int64_t)num_unique, (uint64_t)n,
                          d_seeds,
                          d_first, d_second, d_scores,
                          use_warp, stream);

    
    int64_t output_count = num_unique;
    if (topk.has_value()) {
        sort_scores_desc(d_scores, d_first, d_second, num_unique, stream);
        output_count = ((int64_t)topk.value() < (int64_t)num_unique) ? (int64_t)topk.value() : (int64_t)num_unique;
    }

    if (output_count < num_unique) {
        int32_t* f;
        int32_t* s;
        double* sc;
        cudaMalloc(&f, (size_t)output_count * sizeof(int32_t));
        cudaMalloc(&s, (size_t)output_count * sizeof(int32_t));
        cudaMalloc(&sc, (size_t)output_count * sizeof(double));
        cudaMemcpyAsync(f, d_first, (size_t)output_count * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(s, d_second, (size_t)output_count * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(sc, d_scores, (size_t)output_count * sizeof(double), cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
        cudaFree(d_first);
        cudaFree(d_second);
        cudaFree(d_scores);
        return {f, s, sc, (std::size_t)output_count};
    }

    cudaStreamSynchronize(stream);
    return {d_first, d_second, d_scores, (std::size_t)output_count};
}

}  
