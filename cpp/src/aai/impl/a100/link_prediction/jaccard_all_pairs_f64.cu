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
#include <optional>
#include <vector>
#include <algorithm>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

namespace aai {

namespace {

struct Cache : Cacheable {
    
    double* weight_sums = nullptr;
    int64_t weight_sums_capacity = 0;

    
    int32_t* seeds = nullptr;
    int64_t seeds_capacity = 0;

    
    int64_t* per_seed_bound = nullptr;
    int64_t per_seed_bound_capacity = 0;

    
    int32_t* per_seed_degree = nullptr;
    int64_t per_seed_degree_capacity = 0;

    
    int64_t* total_bound = nullptr;
    bool total_bound_allocated = false;

    
    int32_t* deg_prefix = nullptr;
    int64_t deg_prefix_capacity = 0;

    
    int64_t* unique_offsets = nullptr;
    int64_t unique_offsets_capacity = 0;

    
    int32_t* unique_caps = nullptr;
    int64_t unique_caps_capacity = 0;

    
    uint32_t* bitmaps = nullptr;
    int64_t bitmaps_capacity = 0;

    
    int32_t* all_unique = nullptr;
    int64_t all_unique_capacity = 0;

    
    int32_t* per_seed_count = nullptr;
    int64_t per_seed_count_capacity = 0;

    
    int64_t* unique_prefix = nullptr;
    int64_t unique_prefix_capacity = 0;

    ~Cache() override {
        if (weight_sums) cudaFree(weight_sums);
        if (seeds) cudaFree(seeds);
        if (per_seed_bound) cudaFree(per_seed_bound);
        if (per_seed_degree) cudaFree(per_seed_degree);
        if (total_bound) cudaFree(total_bound);
        if (deg_prefix) cudaFree(deg_prefix);
        if (unique_offsets) cudaFree(unique_offsets);
        if (unique_caps) cudaFree(unique_caps);
        if (bitmaps) cudaFree(bitmaps);
        if (all_unique) cudaFree(all_unique);
        if (per_seed_count) cudaFree(per_seed_count);
        if (unique_prefix) cudaFree(unique_prefix);
    }

    void ensure_weight_sums(int64_t n) {
        if (weight_sums_capacity < n) {
            if (weight_sums) cudaFree(weight_sums);
            cudaMalloc(&weight_sums, n * sizeof(double));
            weight_sums_capacity = n;
        }
    }

    void ensure_seeds(int64_t n) {
        if (seeds_capacity < n) {
            if (seeds) cudaFree(seeds);
            cudaMalloc(&seeds, n * sizeof(int32_t));
            seeds_capacity = n;
        }
    }

    void ensure_per_seed_bound(int64_t n) {
        if (per_seed_bound_capacity < n) {
            if (per_seed_bound) cudaFree(per_seed_bound);
            cudaMalloc(&per_seed_bound, n * sizeof(int64_t));
            per_seed_bound_capacity = n;
        }
    }

    void ensure_per_seed_degree(int64_t n) {
        if (per_seed_degree_capacity < n) {
            if (per_seed_degree) cudaFree(per_seed_degree);
            cudaMalloc(&per_seed_degree, n * sizeof(int32_t));
            per_seed_degree_capacity = n;
        }
    }

    void ensure_total_bound() {
        if (!total_bound_allocated) {
            cudaMalloc(&total_bound, sizeof(int64_t));
            total_bound_allocated = true;
        }
    }

    void ensure_deg_prefix(int64_t n) {
        if (deg_prefix_capacity < n) {
            if (deg_prefix) cudaFree(deg_prefix);
            cudaMalloc(&deg_prefix, n * sizeof(int32_t));
            deg_prefix_capacity = n;
        }
    }

    void ensure_unique_offsets(int64_t n) {
        if (unique_offsets_capacity < n) {
            if (unique_offsets) cudaFree(unique_offsets);
            cudaMalloc(&unique_offsets, n * sizeof(int64_t));
            unique_offsets_capacity = n;
        }
    }

    void ensure_unique_caps(int64_t n) {
        if (unique_caps_capacity < n) {
            if (unique_caps) cudaFree(unique_caps);
            cudaMalloc(&unique_caps, n * sizeof(int32_t));
            unique_caps_capacity = n;
        }
    }

    void ensure_bitmaps(int64_t n) {
        if (bitmaps_capacity < n) {
            if (bitmaps) cudaFree(bitmaps);
            cudaMalloc(&bitmaps, n * sizeof(uint32_t));
            bitmaps_capacity = n;
        }
    }

    void ensure_all_unique(int64_t n) {
        if (all_unique_capacity < n) {
            if (all_unique) cudaFree(all_unique);
            cudaMalloc(&all_unique, n * sizeof(int32_t));
            all_unique_capacity = n;
        }
    }

    void ensure_per_seed_count(int64_t n) {
        if (per_seed_count_capacity < n) {
            if (per_seed_count) cudaFree(per_seed_count);
            cudaMalloc(&per_seed_count, n * sizeof(int32_t));
            per_seed_count_capacity = n;
        }
    }

    void ensure_unique_prefix(int64_t n) {
        if (unique_prefix_capacity < n) {
            if (unique_prefix) cudaFree(unique_prefix);
            cudaMalloc(&unique_prefix, n * sizeof(int64_t));
            unique_prefix_capacity = n;
        }
    }
};




__global__ void k_compute_weight_sums(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ edge_weights,
    double* __restrict__ weight_sums,
    int32_t num_vertices
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += gridDim.x * blockDim.x) {
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        double sum = 0.0;
        for (int32_t i = start; i < end; i++) {
            sum += edge_weights[i];
        }
        weight_sums[v] = sum;
    }
}




__global__ void k_compute_bounds(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ per_seed_bound,
    int32_t* __restrict__ per_seed_degree,
    int64_t* __restrict__ total_bound
) {
    extern __shared__ int64_t shared_sum[];
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int64_t my_sum = 0;
    for (int s = tid; s < num_seeds; s += block_size) {
        int32_t u = seeds[s];
        int32_t u_start = offsets[u];
        int32_t u_end = offsets[u + 1];
        int32_t deg = u_end - u_start;
        per_seed_degree[s] = deg;

        int64_t seed_sum = 0;
        for (int32_t i = u_start; i < u_end; i++) {
            int32_t c = indices[i];
            seed_sum += offsets[c + 1] - offsets[c];
        }
        per_seed_bound[s] = seed_sum;
        my_sum += seed_sum;
    }

    shared_sum[tid] = my_sum;
    __syncthreads();
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
        __syncthreads();
    }
    if (tid == 0) *total_bound = shared_sum[0];
}




__global__ void k_expand_dedup_all(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int32_t* __restrict__ degree_prefix,
    int32_t total_neighbors,
    uint32_t* __restrict__ bitmaps,
    int32_t bitmap_stride,
    int32_t* __restrict__ all_unique,
    const int64_t* __restrict__ unique_offsets,
    const int32_t* __restrict__ unique_caps,
    int32_t* __restrict__ per_seed_count
) {
    for (int bi = blockIdx.x; bi < total_neighbors; bi += gridDim.x) {
        int lo = 0, hi = num_seeds - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (degree_prefix[mid] <= bi) lo = mid;
            else hi = mid - 1;
        }
        int seed_idx = lo;
        int neighbor_offset = bi - degree_prefix[seed_idx];

        int32_t u = seeds[seed_idx];
        int32_t u_start = offsets[u];
        int32_t c = indices[u_start + neighbor_offset];
        int32_t c_start = offsets[c];
        int32_t c_end = offsets[c + 1];

        uint32_t* my_bitmap = bitmaps + (int64_t)seed_idx * bitmap_stride;
        int32_t* my_unique = all_unique + unique_offsets[seed_idx];
        int32_t my_cap = unique_caps[seed_idx];
        int32_t* my_count = per_seed_count + seed_idx;

        for (int32_t j = c_start + threadIdx.x; j < c_end; j += blockDim.x) {
            int32_t v = indices[j];
            if (v == u) continue;

            uint32_t bit = 1u << (v & 31);
            uint32_t old = atomicOr(&my_bitmap[v >> 5], bit);
            if (!(old & bit)) {
                int pos = atomicAdd(my_count, 1);
                if (pos < my_cap) {
                    my_unique[pos] = v;
                }
            }
        }
    }
}




__global__ void k_prefix_sum_small(
    const int32_t* __restrict__ counts,
    int64_t* __restrict__ prefix,
    int32_t n
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int64_t sum = 0;
        for (int i = 0; i < n; i++) {
            prefix[i] = sum;
            sum += counts[i];
        }
        prefix[n] = sum;
    }
}




__global__ void k_compute_jaccard_all(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const double* __restrict__ weight_sums,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ unique_prefix,
    const int32_t* __restrict__ all_unique,
    const int64_t* __restrict__ unique_offsets,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    double* __restrict__ out_scores
) {
    int64_t total_pairs = unique_prefix[num_seeds];
    int warps_per_block = blockDim.x / 32;
    int lane = threadIdx.x % 32;
    int warp_in_block = threadIdx.x / 32;
    int64_t global_warp = (int64_t)blockIdx.x * warps_per_block + warp_in_block;
    int64_t total_warps = (int64_t)gridDim.x * warps_per_block;

    for (int64_t warp_id = global_warp; warp_id < total_pairs; warp_id += total_warps) {
        int lo = 0, hi = num_seeds - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >> 1;
            if (unique_prefix[mid] <= warp_id) lo = mid;
            else hi = mid - 1;
        }
        int seed_idx = lo;
        int pair_offset = (int)(warp_id - unique_prefix[seed_idx]);

        int32_t u = seeds[seed_idx];
        int32_t v = all_unique[unique_offsets[seed_idx] + pair_offset];

        int32_t u_start = offsets[u];
        int32_t u_end = offsets[u + 1];
        int32_t v_start = offsets[v];
        int32_t v_end = offsets[v + 1];
        int32_t u_deg = u_end - u_start;
        int32_t v_deg = v_end - v_start;

        const int32_t* A_base;
        const double* A_w;
        int32_t A_len;
        const int32_t* B_base;
        const double* B_w;
        int32_t B_len;

        if (u_deg <= v_deg) {
            A_base = indices + u_start; A_w = edge_weights + u_start; A_len = u_deg;
            B_base = indices + v_start; B_w = edge_weights + v_start; B_len = v_deg;
        } else {
            A_base = indices + v_start; A_w = edge_weights + v_start; A_len = v_deg;
            B_base = indices + u_start; B_w = edge_weights + u_start; B_len = u_deg;
        }

        double my_intersection = 0.0;
        for (int i = lane; i < A_len; i += 32) {
            int32_t target = A_base[i];
            double wa = A_w[i];

            int blo = 0, bhi = B_len;
            while (blo < bhi) {
                int mid = (blo + bhi) >> 1;
                if (B_base[mid] < target) blo = mid + 1;
                else bhi = mid;
            }

            if (blo < B_len && B_base[blo] == target) {
                my_intersection += fmin(wa, B_w[blo]);
            }
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            my_intersection += __shfl_down_sync(0xffffffff, my_intersection, offset);
        }

        if (lane == 0) {
            double ws_u = weight_sums[u];
            double ws_v = weight_sums[v];
            double denom = ws_u + ws_v - my_intersection;
            double score = (denom > 0.0) ? (my_intersection / denom) : 0.0;

            out_first[warp_id] = u;
            out_second[warp_id] = v;
            out_scores[warp_id] = score;
        }
    }
}




__global__ void k_iota(int32_t* out, int32_t n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        out[i] = i;
}

__global__ void k_prefix_sum_i32(const int32_t* input, int32_t* output, int32_t n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int32_t sum = 0;
        for (int i = 0; i < n; i++) {
            output[i] = sum;
            sum += input[i];
        }
        output[n] = sum;
    }
}

}  

similarity_result_double_t jaccard_all_pairs_similarity(const graph32_t& graph,
                                                        const double* edge_weights,
                                                        const int32_t* vertices,
                                                        std::size_t num_vertices_param,
                                                        std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;

    if (nv == 0 || ne == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t num_seeds;
    const int32_t* d_seeds;
    if (vertices != nullptr && num_vertices_param > 0) {
        num_seeds = (int32_t)num_vertices_param;
        d_seeds = vertices;
    } else {
        num_seeds = nv;
        cache.ensure_seeds(nv);
        int block = 256;
        int grid = (nv + block - 1) / block;
        grid = grid < 1024 ? grid : 1024;
        k_iota<<<grid, block>>>(cache.seeds, nv);
        d_seeds = cache.seeds;
    }

    if (num_seeds == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    cache.ensure_weight_sums(nv);
    {
        int block = 256;
        int grid = (nv + block - 1) / block;
        grid = grid < 1024 ? grid : 1024;
        k_compute_weight_sums<<<grid, block>>>(d_offsets, edge_weights, cache.weight_sums, nv);
    }

    
    cache.ensure_per_seed_bound(num_seeds);
    cache.ensure_per_seed_degree(num_seeds);
    cache.ensure_total_bound();
    {
        int block = 256;
        k_compute_bounds<<<1, block, block * sizeof(int64_t)>>>(
            d_offsets, d_indices, d_seeds, num_seeds,
            cache.per_seed_bound, cache.per_seed_degree, cache.total_bound);
    }

    
    std::vector<int64_t> h_bounds(num_seeds);
    std::vector<int32_t> h_degrees(num_seeds);
    int64_t h_total_bound;
    cudaMemcpy(h_bounds.data(), cache.per_seed_bound,
               num_seeds * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_degrees.data(), cache.per_seed_degree,
               num_seeds * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_total_bound, cache.total_bound,
               sizeof(int64_t), cudaMemcpyDeviceToHost);

    
    std::vector<int32_t> h_deg_prefix(num_seeds + 1);
    h_deg_prefix[0] = 0;
    for (int s = 0; s < num_seeds; s++)
        h_deg_prefix[s + 1] = h_deg_prefix[s] + h_degrees[s];
    int32_t total_neighbors = h_deg_prefix[num_seeds];

    
    std::vector<int64_t> h_unique_offsets(num_seeds + 1);
    std::vector<int32_t> h_unique_caps(num_seeds);
    h_unique_offsets[0] = 0;
    for (int s = 0; s < num_seeds; s++) {
        int64_t cap = h_bounds[s];
        if (cap > (int64_t)nv) cap = nv;
        h_unique_caps[s] = (int32_t)cap;
        h_unique_offsets[s + 1] = h_unique_offsets[s] + cap;
    }
    int64_t total_unique_buf = h_unique_offsets[num_seeds];
    if (total_unique_buf < 1) total_unique_buf = 1;

    
    cache.ensure_deg_prefix(num_seeds + 1);
    cache.ensure_unique_offsets(num_seeds + 1);
    cache.ensure_unique_caps(num_seeds);
    cudaMemcpy(cache.deg_prefix, h_deg_prefix.data(),
               (num_seeds + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cache.unique_offsets, h_unique_offsets.data(),
               (num_seeds + 1) * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(cache.unique_caps, h_unique_caps.data(),
               num_seeds * sizeof(int32_t), cudaMemcpyHostToDevice);

    
    int32_t bitmap_stride = (nv + 31) / 32;
    int64_t bitmap_total = (int64_t)num_seeds * bitmap_stride;
    cache.ensure_bitmaps(bitmap_total);
    cudaMemset(cache.bitmaps, 0, bitmap_total * sizeof(uint32_t));

    
    cache.ensure_all_unique(total_unique_buf);
    cache.ensure_per_seed_count(num_seeds);
    cudaMemset(cache.per_seed_count, 0, num_seeds * sizeof(int32_t));

    
    if (total_neighbors > 0) {
        int block = 256;
        int grid = total_neighbors < 2048 ? total_neighbors : 2048;
        k_expand_dedup_all<<<grid, block>>>(
            d_offsets, d_indices, d_seeds, num_seeds,
            cache.deg_prefix, total_neighbors,
            cache.bitmaps, bitmap_stride,
            cache.all_unique,
            cache.unique_offsets,
            cache.unique_caps,
            cache.per_seed_count);
    }

    
    cache.ensure_unique_prefix(num_seeds + 1);
    k_prefix_sum_small<<<1, 1>>>(cache.per_seed_count,
                                  cache.unique_prefix, num_seeds);

    
    int64_t total_pairs;
    cudaMemcpy(&total_pairs, cache.unique_prefix + num_seeds,
               sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (total_pairs <= 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    
    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    double* out_scores = nullptr;
    cudaMalloc(&out_first, total_pairs * sizeof(int32_t));
    cudaMalloc(&out_second, total_pairs * sizeof(int32_t));
    cudaMalloc(&out_scores, total_pairs * sizeof(double));

    {
        int warps_per_block = 8;
        int block = warps_per_block * 32;
        int64_t num_blocks = (total_pairs + warps_per_block - 1) / warps_per_block;
        if (num_blocks > 2048) num_blocks = 2048;
        k_compute_jaccard_all<<<(int)num_blocks, block>>>(
            d_offsets, d_indices, edge_weights, cache.weight_sums,
            d_seeds, num_seeds,
            cache.unique_prefix, cache.all_unique, cache.unique_offsets,
            out_first, out_second, out_scores);
    }

    
    if (topk.has_value() && total_pairs > (int64_t)topk.value()) {
        int64_t k = (int64_t)topk.value();

        
        thrust::device_ptr<double> d_scores_ptr(out_scores);
        auto vals = thrust::make_zip_iterator(
            thrust::make_tuple(thrust::device_ptr<int32_t>(out_first),
                               thrust::device_ptr<int32_t>(out_second)));
        thrust::sort_by_key(d_scores_ptr, d_scores_ptr + total_pairs, vals, thrust::greater<double>());

        
        int32_t* trunc_first = nullptr;
        int32_t* trunc_second = nullptr;
        double* trunc_scores = nullptr;
        cudaMalloc(&trunc_first, k * sizeof(int32_t));
        cudaMalloc(&trunc_second, k * sizeof(int32_t));
        cudaMalloc(&trunc_scores, k * sizeof(double));
        cudaMemcpy(trunc_first, out_first, k * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(trunc_second, out_second, k * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(trunc_scores, out_scores, k * sizeof(double), cudaMemcpyDeviceToDevice);

        
        cudaFree(out_first);
        cudaFree(out_second);
        cudaFree(out_scores);

        return {trunc_first, trunc_second, trunc_scores, (std::size_t)k};
    }

    return {out_first, out_second, out_scores, (std::size_t)total_pairs};
}

}  
