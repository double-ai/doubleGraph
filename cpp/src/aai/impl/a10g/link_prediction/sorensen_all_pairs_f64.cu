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
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>

namespace aai {

namespace {

constexpr int BLOCK_SIZE = 256;

struct Cache : Cacheable {
    double* w_degrees = nullptr;
    int64_t w_degrees_capacity = 0;

    int64_t* counts = nullptr;
    int64_t counts_capacity = 0;

    int64_t* seed_offsets = nullptr;
    int64_t seed_offsets_capacity = 0;

    int32_t* seeds_buf = nullptr;
    int64_t seeds_buf_capacity = 0;

    int64_t* pair_keys = nullptr;
    int64_t pair_keys_capacity = 0;

    void ensure_w_degrees(int64_t n) {
        if (w_degrees_capacity < n) {
            if (w_degrees) cudaFree(w_degrees);
            cudaMalloc(&w_degrees, n * sizeof(double));
            w_degrees_capacity = n;
        }
    }

    void ensure_counts(int64_t n) {
        if (counts_capacity < n) {
            if (counts) cudaFree(counts);
            cudaMalloc(&counts, n * sizeof(int64_t));
            counts_capacity = n;
        }
    }

    void ensure_seed_offsets(int64_t n) {
        if (seed_offsets_capacity < n) {
            if (seed_offsets) cudaFree(seed_offsets);
            cudaMalloc(&seed_offsets, n * sizeof(int64_t));
            seed_offsets_capacity = n;
        }
    }

    void ensure_seeds_buf(int64_t n) {
        if (seeds_buf_capacity < n) {
            if (seeds_buf) cudaFree(seeds_buf);
            cudaMalloc(&seeds_buf, n * sizeof(int32_t));
            seeds_buf_capacity = n;
        }
    }

    void ensure_pair_keys(int64_t n) {
        if (pair_keys_capacity < n) {
            if (pair_keys) cudaFree(pair_keys);
            cudaMalloc(&pair_keys, n * sizeof(int64_t));
            pair_keys_capacity = n;
        }
    }

    ~Cache() override {
        if (w_degrees) cudaFree(w_degrees);
        if (counts) cudaFree(counts);
        if (seed_offsets) cudaFree(seed_offsets);
        if (seeds_buf) cudaFree(seeds_buf);
        if (pair_keys) cudaFree(pair_keys);
    }
};



__device__ __forceinline__ int lower_bound_dev(
    const int32_t* __restrict__ arr, int lo, int hi, int target)
{
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}



__global__ void kernel_weighted_degrees(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ weights,
    double* __restrict__ w_degrees,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int s = offsets[v], e = offsets[v + 1];
    double sum = 0.0;
    for (int i = s; i < e; i++) sum += weights[i];
    w_degrees[v] = sum;
}

__global__ void kernel_count_candidates(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ counts)
{
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int u = seeds[sid];
    int u_s = offsets[u], u_e = offsets[u + 1];
    int64_t cnt = 0;
    for (int i = u_s + (int)threadIdx.x; i < u_e; i += blockDim.x) {
        int w = indices[i];
        cnt += (int64_t)(offsets[w + 1] - offsets[w]) - 1;  
    }
    typedef cub::BlockReduce<int64_t, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage tmp;
    int64_t total = BR(tmp).Sum(cnt);
    if (threadIdx.x == 0) counts[sid] = total;
}

__global__ void kernel_generate_candidates(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t num_vertices_64,
    const int64_t* __restrict__ seed_offsets,
    int64_t* __restrict__ pair_keys)
{
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int u = seeds[sid];
    int u_s = offsets[u], u_e = offsets[u + 1];

    __shared__ int64_t write_pos;
    if (threadIdx.x == 0) write_pos = seed_offsets[sid];
    __syncthreads();

    for (int i = u_s + (int)threadIdx.x; i < u_e; i += blockDim.x) {
        int w = indices[i];
        int w_s = offsets[w], w_e = offsets[w + 1];
        int w_deg = w_e - w_s;
        int cnt = w_deg - 1;  
        if (cnt > 0) {
            int64_t pos = atomicAdd((unsigned long long*)&write_pos, (unsigned long long)cnt);
            for (int j = w_s; j < w_e; j++) {
                int v = indices[j];
                if (v != u) {
                    pair_keys[pos++] = (int64_t)u * num_vertices_64 + (int64_t)v;
                }
            }
        }
    }
}


__device__ __forceinline__ double intersect_weighted(
    const int32_t* __restrict__ a_idx, const double* __restrict__ a_wt, int a_n,
    const int32_t* __restrict__ b_idx, const double* __restrict__ b_wt, int b_n)
{
    if (a_n == 0 || b_n == 0) return 0.0;

    
    if (a_n > b_n) {
        
        const int32_t* ti = a_idx; a_idx = b_idx; b_idx = ti;
        const double* tw = a_wt; a_wt = b_wt; b_wt = tw;
        int tn = a_n; a_n = b_n; b_n = tn;
    }

    
    if (a_idx[a_n - 1] < b_idx[0] || b_idx[b_n - 1] < a_idx[0]) return 0.0;

    double sum_min = 0.0;

    if (b_n > 10 * a_n) {
        
        int j = 0;
        for (int i = 0; i < a_n && j < b_n; i++) {
            int target = a_idx[i];
            
            int pos = j, step = 1;
            while (pos + step < b_n && b_idx[pos + step] < target) {
                pos += step;
                step <<= 1;
            }
            int lo = pos, hi = (pos + step < b_n) ? pos + step + 1 : b_n;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (b_idx[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            j = lo;
            if (j < b_n && b_idx[j] == target) {
                double wa = a_wt[i], wb = b_wt[j];
                sum_min += (wa < wb) ? wa : wb;
                j++;
            }
        }
    } else {
        
        int i = lower_bound_dev(a_idx, 0, a_n, b_idx[0]);
        int j = (i < a_n) ? lower_bound_dev(b_idx, 0, b_n, a_idx[i]) : b_n;

        while (i < a_n && j < b_n) {
            int ai = a_idx[i], bj = b_idx[j];
            if (ai == bj) {
                double wa = a_wt[i], wb = b_wt[j];
                sum_min += (wa < wb) ? wa : wb;
                i++; j++;
            } else if (ai < bj) {
                i++;
            } else {
                j++;
            }
        }
    }

    return sum_min;
}

__global__ void kernel_compute_sorensen(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ w_degrees,
    const int64_t* __restrict__ pair_keys,
    int64_t num_vertices_64,
    int64_t num_pairs,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    double* __restrict__ out_scores)
{
    int64_t pid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= num_pairs) return;

    int64_t key = pair_keys[pid];
    int32_t u = (int32_t)(key / num_vertices_64);
    int32_t v = (int32_t)(key % num_vertices_64);

    out_first[pid] = u;
    out_second[pid] = v;

    int u_s = offsets[u], u_e = offsets[u + 1];
    int v_s = offsets[v], v_e = offsets[v + 1];

    double sum_min = intersect_weighted(
        indices + u_s, weights + u_s, u_e - u_s,
        indices + v_s, weights + v_s, v_e - v_s);

    double denom = w_degrees[u] + w_degrees[v];
    out_scores[pid] = (denom > 0.0) ? (2.0 * sum_min / denom) : 0.0;
}

}  

similarity_result_double_t sorensen_all_pairs_similarity(const graph32_t& graph,
                                                         const double* edge_weights,
                                                         const int32_t* vertices,
                                                         std::size_t num_vertices,
                                                         std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n_verts = graph.number_of_vertices;
    const double* d_weights = edge_weights;

    
    const int32_t* d_seeds;
    int32_t num_seeds;

    if (vertices != nullptr) {
        d_seeds = vertices;
        num_seeds = (int32_t)num_vertices;
    } else {
        num_seeds = n_verts;
        cache.ensure_seeds_buf(num_seeds);
        thrust::device_ptr<int32_t> d(cache.seeds_buf);
        thrust::sequence(d, d + num_seeds);
        d_seeds = cache.seeds_buf;
    }

    if (num_seeds == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    cache.ensure_w_degrees(n_verts);
    {
        int grid = (n_verts + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kernel_weighted_degrees<<<grid, BLOCK_SIZE>>>(d_offsets, d_weights, cache.w_degrees, n_verts);
    }

    
    cache.ensure_counts(num_seeds);
    if (num_seeds > 0) {
        kernel_count_candidates<<<num_seeds, BLOCK_SIZE>>>(d_offsets, d_indices, d_seeds, num_seeds, cache.counts);
    }

    
    cache.ensure_seed_offsets(num_seeds);
    int64_t total_raw;
    {
        thrust::device_ptr<int64_t> dc(cache.counts);
        thrust::device_ptr<int64_t> dout(cache.seed_offsets);
        thrust::exclusive_scan(dc, dc + num_seeds, dout);
        int64_t last_off, last_cnt;
        cudaMemcpy(&last_off, cache.seed_offsets + num_seeds - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_cnt, cache.counts + num_seeds - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
        total_raw = last_off + last_cnt;
    }

    if (total_raw == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    cache.ensure_pair_keys(total_raw);
    if (num_seeds > 0) {
        kernel_generate_candidates<<<num_seeds, BLOCK_SIZE>>>(
            d_offsets, d_indices, d_seeds, num_seeds,
            (int64_t)n_verts, cache.seed_offsets, cache.pair_keys);
    }

    
    int64_t num_unique;
    {
        thrust::device_ptr<int64_t> dk(cache.pair_keys);
        thrust::sort(dk, dk + total_raw);
        auto new_end = thrust::unique(dk, dk + total_raw);
        num_unique = new_end - dk;
    }

    if (num_unique == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    double* out_scores = nullptr;
    cudaMalloc(&out_first, num_unique * sizeof(int32_t));
    cudaMalloc(&out_second, num_unique * sizeof(int32_t));
    cudaMalloc(&out_scores, num_unique * sizeof(double));

    
    {
        int grid = (int)((num_unique + BLOCK_SIZE - 1) / BLOCK_SIZE);
        kernel_compute_sorensen<<<grid, BLOCK_SIZE>>>(
            d_offsets, d_indices, d_weights, cache.w_degrees,
            cache.pair_keys, (int64_t)n_verts, num_unique,
            out_first, out_second, out_scores);
    }

    
    if (topk.has_value() && num_unique > (int64_t)topk.value()) {
        int64_t k = (int64_t)topk.value();

        
        thrust::device_ptr<double> dk(out_scores);
        thrust::device_ptr<int32_t> df(out_first);
        thrust::device_ptr<int32_t> ds(out_second);
        auto vals = thrust::make_zip_iterator(thrust::make_tuple(df, ds));
        thrust::sort_by_key(dk, dk + num_unique, vals, thrust::greater<double>());

        if (k == 0) {
            cudaFree(out_first);
            cudaFree(out_second);
            cudaFree(out_scores);
            return {nullptr, nullptr, nullptr, 0};
        }

        
        int32_t* final_first = nullptr;
        int32_t* final_second = nullptr;
        double* final_scores = nullptr;
        cudaMalloc(&final_first, k * sizeof(int32_t));
        cudaMalloc(&final_second, k * sizeof(int32_t));
        cudaMalloc(&final_scores, k * sizeof(double));

        cudaMemcpyAsync(final_first, out_first, k * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(final_second, out_second, k * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(final_scores, out_scores, k * sizeof(double), cudaMemcpyDeviceToDevice);

        
        cudaFree(out_first);
        cudaFree(out_second);
        cudaFree(out_scores);

        return {final_first, final_second, final_scores, (std::size_t)k};
    }

    return {out_first, out_second, out_scores, (std::size_t)num_unique};
}

}  
