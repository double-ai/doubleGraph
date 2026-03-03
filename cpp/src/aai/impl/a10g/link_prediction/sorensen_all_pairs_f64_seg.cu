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
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <cstdint>
#include <algorithm>
#include <optional>

namespace aai {

namespace {

struct Cache : Cacheable {
    double* w_degrees = nullptr;
    int64_t w_degrees_cap = 0;

    int32_t* seeds = nullptr;
    int64_t seeds_cap = 0;

    uint32_t* bitmap = nullptr;
    int64_t bitmap_cap = 0;

    int64_t* pair_counts = nullptr;
    int64_t pair_counts_cap = 0;

    int64_t* pair_offsets = nullptr;
    int64_t pair_offsets_cap = 0;

    int64_t* raw_keys = nullptr;
    int64_t raw_keys_cap = 0;

    void ensure_w_degrees(int64_t n) {
        if (w_degrees_cap < n) {
            if (w_degrees) cudaFree(w_degrees);
            cudaMalloc(&w_degrees, n * sizeof(double));
            w_degrees_cap = n;
        }
    }

    void ensure_seeds(int64_t n) {
        if (seeds_cap < n) {
            if (seeds) cudaFree(seeds);
            cudaMalloc(&seeds, n * sizeof(int32_t));
            seeds_cap = n;
        }
    }

    void ensure_bitmap(int64_t n) {
        if (bitmap_cap < n) {
            if (bitmap) cudaFree(bitmap);
            cudaMalloc(&bitmap, n * sizeof(uint32_t));
            bitmap_cap = n;
        }
    }

    void ensure_pair_counts(int64_t n) {
        if (pair_counts_cap < n) {
            if (pair_counts) cudaFree(pair_counts);
            cudaMalloc(&pair_counts, n * sizeof(int64_t));
            pair_counts_cap = n;
        }
    }

    void ensure_pair_offsets(int64_t n) {
        if (pair_offsets_cap < n) {
            if (pair_offsets) cudaFree(pair_offsets);
            cudaMalloc(&pair_offsets, n * sizeof(int64_t));
            pair_offsets_cap = n;
        }
    }

    void ensure_raw_keys(int64_t n) {
        if (raw_keys_cap < n) {
            if (raw_keys) cudaFree(raw_keys);
            cudaMalloc(&raw_keys, n * sizeof(int64_t));
            raw_keys_cap = n;
        }
    }

    ~Cache() override {
        if (w_degrees) cudaFree(w_degrees);
        if (seeds) cudaFree(seeds);
        if (bitmap) cudaFree(bitmap);
        if (pair_counts) cudaFree(pair_counts);
        if (pair_offsets) cudaFree(pair_offsets);
        if (raw_keys) cudaFree(raw_keys);
    }
};




__global__ void compute_weighted_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ edge_weights,
    double* __restrict__ weighted_degrees,
    int32_t num_vertices)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    double sum = 0.0;
    for (int32_t e = offsets[v]; e < offsets[v + 1]; e++)
        sum += edge_weights[e];
    weighted_degrees[v] = sum;
}





__global__ void bitmap_mark_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    uint32_t* __restrict__ bitmaps,
    int32_t bitmap_words)
{
    int32_t sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int32_t u = seeds[sid];
    uint32_t* bm = bitmaps + (int64_t)sid * bitmap_words;

    
    for (int32_t i = threadIdx.x; i < bitmap_words; i += blockDim.x)
        bm[i] = 0;
    __syncthreads();

    
    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    for (int32_t wi = u_start; wi < u_end; wi++) {
        int32_t w = indices[wi];
        int32_t w_start = offsets[w], w_end = offsets[w + 1];
        for (int32_t vi = w_start + threadIdx.x; vi < w_end; vi += blockDim.x) {
            int32_t v = indices[vi];
            if (v != u) {
                atomicOr(&bm[v >> 5], 1u << (v & 31));
            }
        }
    }
}




__global__ void bitmap_count_kernel(
    const uint32_t* __restrict__ bitmaps,
    int32_t num_seeds,
    int32_t bitmap_words,
    int64_t* __restrict__ counts)
{
    int32_t sid = blockIdx.x;
    if (sid >= num_seeds) return;

    const uint32_t* bm = bitmaps + (int64_t)sid * bitmap_words;

    int32_t count = 0;
    for (int32_t i = threadIdx.x; i < bitmap_words; i += blockDim.x)
        count += __popc(bm[i]);

    
    for (int offset = 16; offset > 0; offset >>= 1)
        count += __shfl_down_sync(0xFFFFFFFF, count, offset);

    
    __shared__ int32_t warp_counts[8]; 
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (lane == 0) warp_counts[warp_id] = count;
    __syncthreads();

    if (threadIdx.x == 0) {
        int32_t total = 0;
        int num_warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < num_warps; i++)
            total += warp_counts[i];
        counts[sid] = total;
    }
}




__global__ void bitmap_compact_kernel(
    const uint32_t* __restrict__ bitmaps,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int32_t bitmap_words,
    const int64_t* __restrict__ pair_offsets,
    int32_t* __restrict__ out_v1,
    int32_t* __restrict__ out_v2)
{
    int32_t sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int32_t u = seeds[sid];
    const uint32_t* bm = bitmaps + (int64_t)sid * bitmap_words;
    int64_t base = pair_offsets[sid];

    __shared__ int32_t s_write_pos;
    if (threadIdx.x == 0) s_write_pos = 0;
    __syncthreads();

    for (int32_t i = threadIdx.x; i < bitmap_words; i += blockDim.x) {
        uint32_t word = bm[i];
        int32_t base_v = i * 32;
        while (word != 0) {
            int bit = __ffs(word) - 1;
            word &= word - 1; 
            int32_t v = base_v + bit;
            int32_t pos = atomicAdd(&s_write_pos, 1);
            out_v1[base + pos] = u;
            out_v2[base + pos] = v;
        }
    }
}




__global__ void count_two_hop_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ pair_counts)
{
    int32_t sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_seeds) return;
    int32_t u = seeds[sid];
    int64_t count = 0;
    for (int32_t wi = offsets[u]; wi < offsets[u + 1]; wi++) {
        int32_t w = indices[wi];
        count += (int64_t)(offsets[w + 1] - offsets[w]);
    }
    pair_counts[sid] = count;
}

__global__ void write_two_hop_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ pair_offsets,
    int64_t* __restrict__ out_keys)
{
    int32_t sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int32_t u = seeds[sid];
    int64_t pos = pair_offsets[sid];
    int64_t u_shifted = ((int64_t)(uint32_t)u) << 32;
    for (int32_t wi = offsets[u]; wi < offsets[u + 1]; wi++) {
        int32_t w = indices[wi];
        int32_t w_start = offsets[w], w_end = offsets[w + 1];
        int32_t deg_w = w_end - w_start;
        for (int32_t vi = threadIdx.x; vi < deg_w; vi += blockDim.x) {
            out_keys[pos + vi] = u_shifted | (uint32_t)indices[w_start + vi];
        }
        pos += deg_w;
    }
}




__device__ __forceinline__ int32_t lower_bound_device(
    const int32_t* __restrict__ indices,
    int32_t lo, int32_t hi, int32_t target)
{
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (indices[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}




__global__ void compute_sorensen_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ pair_v1,
    const int32_t* __restrict__ pair_v2,
    const double* __restrict__ weighted_degrees,
    int64_t num_pairs,
    double* __restrict__ scores)
{
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int32_t u = pair_v1[warp_id];
    int32_t v = pair_v2[warp_id];

    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int32_t v_start = offsets[v], v_end = offsets[v + 1];
    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;

    double my_sum = 0.0;

    if (deg_u > 0 && deg_v > 0) {
        int32_t small_start, small_end, small_deg;
        int32_t large_start, large_end, large_deg;

        if (deg_u <= deg_v) {
            small_start = u_start; small_end = u_end; small_deg = deg_u;
            large_start = v_start; large_end = v_end; large_deg = deg_v;
        } else {
            small_start = v_start; small_end = v_end; small_deg = deg_v;
            large_start = u_start; large_end = u_end; large_deg = deg_u;
        }

        int32_t small_first = indices[small_start];
        int32_t small_last = indices[small_end - 1];
        int32_t large_first = indices[large_start];
        int32_t large_last = indices[large_end - 1];

        if (small_first <= large_last && large_first <= small_last) {
            int32_t l_lo = lower_bound_device(indices, large_start, large_end, small_first);
            int32_t l_hi = large_end;
            if (small_last < large_last)
                l_hi = lower_bound_device(indices, l_lo, large_end, small_last + 1);

            for (int32_t i = lane; i < small_deg; i += 32) {
                int32_t target = indices[small_start + i];
                if (target >= large_first && target <= large_last) {
                    int32_t pos = lower_bound_device(indices, l_lo, l_hi, target);
                    if (pos < l_hi && indices[pos] == target) {
                        double w_small = edge_weights[small_start + i];
                        double w_large = edge_weights[pos];
                        my_sum += fmin(w_small, w_large);
                    }
                }
            }
        }
    }

    
    for (int offset = 16; offset > 0; offset >>= 1)
        my_sum += __shfl_down_sync(0xFFFFFFFF, my_sum, offset);

    if (lane == 0) {
        double weight_a = weighted_degrees[u];
        double weight_b = weighted_degrees[v];
        scores[warp_id] = (weight_a + weight_b <= 2.2250738585072014e-308)
            ? 0.0 : 2.0 * my_sum / (weight_a + weight_b);
    }
}

__global__ void decode_keys_kernel(
    const int64_t* __restrict__ keys, int64_t num_keys,
    int32_t* __restrict__ v1, int32_t* __restrict__ v2)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_keys) return;
    int64_t key = keys[i];
    v1[i] = (int32_t)((uint64_t)key >> 32);
    v2[i] = (int32_t)(key & 0xFFFFFFFF);
}

}  

similarity_result_double_t sorensen_all_pairs_similarity_seg(const graph32_t& graph,
                                                             const double* edge_weights,
                                                             const int32_t* vertices,
                                                             std::size_t num_vertices,
                                                             std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t n_verts = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    
    cache.ensure_w_degrees(n_verts);
    if (n_verts > 0) {
        compute_weighted_degrees_kernel<<<(n_verts + 255) / 256, 256>>>(
            d_offsets, edge_weights, cache.w_degrees, n_verts);
    }

    
    int32_t num_seeds;
    const int32_t* d_seeds;
    if (vertices != nullptr && num_vertices > 0) {
        num_seeds = (int32_t)num_vertices;
        d_seeds = vertices;
    } else {
        num_seeds = n_verts;
        cache.ensure_seeds(n_verts);
        thrust::sequence(thrust::device,
                         thrust::device_ptr<int32_t>(cache.seeds),
                         thrust::device_ptr<int32_t>(cache.seeds + n_verts),
                         (int32_t)0);
        d_seeds = cache.seeds;
    }

    if (num_seeds == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t bitmap_words = (n_verts + 31) / 32;
    int64_t bitmap_total_bytes = (int64_t)num_seeds * bitmap_words * sizeof(uint32_t);
    bool use_bitmap = (bitmap_total_bytes < 200LL * 1024 * 1024); 

    int64_t num_unique_pairs;
    int32_t* d_v1;
    int32_t* d_v2;

    if (use_bitmap) {
        
        int64_t bitmap_total_words = (int64_t)num_seeds * bitmap_words;
        cache.ensure_bitmap(bitmap_total_words);

        
        bitmap_mark_kernel<<<num_seeds, 256>>>(
            d_offsets, d_indices, d_seeds, num_seeds, cache.bitmap, bitmap_words);

        
        cache.ensure_pair_counts((int64_t)num_seeds + 1);
        cudaMemsetAsync(cache.pair_counts + num_seeds, 0, sizeof(int64_t));
        bitmap_count_kernel<<<num_seeds, 256>>>(
            cache.bitmap, num_seeds, bitmap_words, cache.pair_counts);

        
        cache.ensure_pair_offsets((int64_t)num_seeds + 1);
        thrust::exclusive_scan(thrust::device,
            thrust::device_ptr<const int64_t>(cache.pair_counts),
            thrust::device_ptr<const int64_t>(cache.pair_counts + num_seeds + 1),
            thrust::device_ptr<int64_t>(cache.pair_offsets),
            (int64_t)0);

        int64_t total;
        cudaMemcpy(&total, cache.pair_offsets + num_seeds, sizeof(int64_t), cudaMemcpyDeviceToHost);

        if (total == 0) {
            return {nullptr, nullptr, nullptr, 0};
        }

        num_unique_pairs = total;
        cudaMalloc(&d_v1, total * sizeof(int32_t));
        cudaMalloc(&d_v2, total * sizeof(int32_t));
        bitmap_compact_kernel<<<num_seeds, 256>>>(
            cache.bitmap, d_seeds, num_seeds, bitmap_words,
            cache.pair_offsets, d_v1, d_v2);
    } else {
        
        cache.ensure_pair_counts((int64_t)num_seeds + 1);
        cudaMemsetAsync(cache.pair_counts + num_seeds, 0, sizeof(int64_t));
        count_two_hop_pairs_kernel<<<(num_seeds + 255) / 256, 256>>>(
            d_offsets, d_indices, d_seeds, num_seeds, cache.pair_counts);

        cache.ensure_pair_offsets((int64_t)num_seeds + 1);
        thrust::exclusive_scan(thrust::device,
            thrust::device_ptr<const int64_t>(cache.pair_counts),
            thrust::device_ptr<const int64_t>(cache.pair_counts + num_seeds + 1),
            thrust::device_ptr<int64_t>(cache.pair_offsets),
            (int64_t)0);

        int64_t total_raw;
        cudaMemcpy(&total_raw, cache.pair_offsets + num_seeds, sizeof(int64_t), cudaMemcpyDeviceToHost);

        if (total_raw == 0) {
            return {nullptr, nullptr, nullptr, 0};
        }

        cache.ensure_raw_keys(total_raw);
        write_two_hop_pairs_kernel<<<num_seeds, 256>>>(
            d_offsets, d_indices, d_seeds, num_seeds,
            cache.pair_offsets, cache.raw_keys);

        
        thrust::device_ptr<int64_t> p(cache.raw_keys);
        thrust::sort(thrust::device, p, p + total_raw);
        auto end = thrust::unique(thrust::device, p, p + total_raw);
        int64_t uc = end - p;
        auto end2 = thrust::remove_if(thrust::device, p, p + uc,
            [] __device__ (int64_t key) {
                return ((uint32_t)((uint64_t)key >> 32)) == ((uint32_t)(key & 0xFFFFFFFFLL));
            });
        num_unique_pairs = end2 - p;

        if (num_unique_pairs == 0) {
            return {nullptr, nullptr, nullptr, 0};
        }

        cudaMalloc(&d_v1, num_unique_pairs * sizeof(int32_t));
        cudaMalloc(&d_v2, num_unique_pairs * sizeof(int32_t));
        decode_keys_kernel<<<(int)((num_unique_pairs + 255) / 256), 256>>>(
            cache.raw_keys, num_unique_pairs, d_v1, d_v2);
    }

    
    double* d_scores;
    cudaMalloc(&d_scores, num_unique_pairs * sizeof(double));
    {
        int tpb = 256;
        int wpb = tpb / 32;
        compute_sorensen_warp_kernel<<<(int)((num_unique_pairs + wpb - 1) / wpb), tpb>>>(
            d_offsets, d_indices, edge_weights, d_v1, d_v2,
            cache.w_degrees, num_unique_pairs, d_scores);
    }

    
    if (topk.has_value()) {
        thrust::device_ptr<double> sp(d_scores);
        thrust::device_ptr<int32_t> v1p(d_v1), v2p(d_v2);
        thrust::sort_by_key(thrust::device, sp, sp + num_unique_pairs,
            thrust::make_zip_iterator(thrust::make_tuple(v1p, v2p)),
            thrust::greater<double>());

        int64_t keep = std::min((int64_t)topk.value(), num_unique_pairs);
        if (keep < num_unique_pairs) {
            int32_t* out_v1;
            int32_t* out_v2;
            double* out_scores;
            cudaMalloc(&out_v1, keep * sizeof(int32_t));
            cudaMalloc(&out_v2, keep * sizeof(int32_t));
            cudaMalloc(&out_scores, keep * sizeof(double));
            cudaMemcpyAsync(out_v1, d_v1, keep * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(out_v2, d_v2, keep * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(out_scores, d_scores, keep * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaFree(d_v1);
            cudaFree(d_v2);
            cudaFree(d_scores);
            return {out_v1, out_v2, out_scores, (std::size_t)keep};
        }
    }

    return {d_v1, d_v2, d_scores, (std::size_t)num_unique_pairs};
}

}  
