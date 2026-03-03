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
#include <optional>

namespace aai {

namespace {

static constexpr int64_t MAX_BITMAP_BYTES = 256LL * 1024 * 1024;





template <typename T>
static void ensure_buf(T*& ptr, int64_t& cap, int64_t needed) {
    if (cap < needed) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, needed * sizeof(T));
        cap = needed;
    }
}

static void ensure_void(void*& ptr, size_t& cap, size_t needed) {
    if (cap < needed) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(&ptr, needed);
        cap = needed;
    }
}





struct Cache : Cacheable {
    
    int32_t* seeds_buf = nullptr;
    int64_t seeds_cap = 0;

    
    uint32_t* bitmap_buf = nullptr;
    int64_t bitmap_cap = 0;

    
    int64_t* counts_buf = nullptr;
    int64_t counts_cap = 0;

    int64_t* offsets_buf = nullptr;
    int64_t offsets_cap = 0;

    
    double* seed_deg_buf = nullptr;
    int64_t seed_deg_cap = 0;

    
    int64_t* pair_keys_buf = nullptr;
    int64_t pair_keys_cap = 0;

    int64_t* sorted_keys_buf = nullptr;
    int64_t sorted_keys_cap = 0;

    int64_t* unique_keys_buf = nullptr;
    int64_t unique_keys_cap = 0;

    int32_t* nsel_buf = nullptr;

    
    int32_t* perm_buf = nullptr;
    int64_t perm_cap = 0;

    int32_t* perm_sorted_buf = nullptr;
    int64_t perm_sorted_cap = 0;

    double* scores_sorted_buf = nullptr;
    int64_t scores_sorted_cap = 0;

    
    void* cub_temp = nullptr;
    size_t cub_temp_cap = 0;

    ~Cache() override {
        if (seeds_buf) cudaFree(seeds_buf);
        if (bitmap_buf) cudaFree(bitmap_buf);
        if (counts_buf) cudaFree(counts_buf);
        if (offsets_buf) cudaFree(offsets_buf);
        if (seed_deg_buf) cudaFree(seed_deg_buf);
        if (pair_keys_buf) cudaFree(pair_keys_buf);
        if (sorted_keys_buf) cudaFree(sorted_keys_buf);
        if (unique_keys_buf) cudaFree(unique_keys_buf);
        if (nsel_buf) cudaFree(nsel_buf);
        if (perm_buf) cudaFree(perm_buf);
        if (perm_sorted_buf) cudaFree(perm_sorted_buf);
        if (scores_sorted_buf) cudaFree(scores_sorted_buf);
        if (cub_temp) cudaFree(cub_temp);
    }
};





__global__ void bitmap_mark_count_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    uint32_t* __restrict__ bitmaps,
    int bitmap_words,
    int64_t* __restrict__ unique_counts
) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    int32_t u = seeds[seed_idx];
    uint32_t* my_bitmap = bitmaps + (int64_t)seed_idx * bitmap_words;

    for (int w = threadIdx.x; w < bitmap_words; w += blockDim.x)
        my_bitmap[w] = 0;
    __syncthreads();

    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int local_count = 0;

    for (int32_t i = u_start; i < u_end; i++) {
        int32_t m = indices[i];
        int32_t m_start = offsets[m], m_end = offsets[m + 1];
        for (int32_t j = m_start + threadIdx.x; j < m_end; j += blockDim.x) {
            int32_t v = indices[j];
            if (v != u) {
                uint32_t bit = 1u << (v & 31);
                uint32_t old = atomicOr(&my_bitmap[v >> 5], bit);
                if ((old & bit) == 0) local_count++;
            }
        }
    }

    __shared__ int s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();
    atomicAdd(&s_count, local_count);
    __syncthreads();
    if (threadIdx.x == 0) unique_counts[seed_idx] = s_count;
}

__global__ void bitmap_extract_pairs_kernel(
    const uint32_t* __restrict__ bitmaps,
    int bitmap_words, int num_seeds,
    const int32_t* __restrict__ seeds,
    const int64_t* __restrict__ pair_offsets,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second
) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;
    int32_t u = seeds[seed_idx];
    const uint32_t* my_bitmap = bitmaps + (int64_t)seed_idx * bitmap_words;
    int64_t base = pair_offsets[seed_idx];
    __shared__ int s_counter;
    if (threadIdx.x == 0) s_counter = 0;
    __syncthreads();
    for (int w = threadIdx.x; w < bitmap_words; w += blockDim.x) {
        uint32_t bits = my_bitmap[w];
        while (bits) {
            int bit = __ffs(bits) - 1;
            int32_t v = w * 32 + bit;
            int pos = atomicAdd(&s_counter, 1);
            out_first[base + pos] = u;
            out_second[base + pos] = v;
            bits &= ~(1u << bit);
        }
    }
}





__global__ void count_2hop_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ counts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_seeds) return;
    int32_t u = seeds[tid];
    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int64_t count = 0;
    for (int32_t i = u_start; i < u_end; i++) {
        int32_t m = indices[i];
        count += (int64_t)(offsets[m + 1] - offsets[m]) - 1;
    }
    counts[tid] = (count > 0) ? count : 0;
}

__global__ void generate_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ pair_offsets,
    int64_t* __restrict__ pair_keys
) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;
    int32_t u = seeds[seed_idx];
    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int64_t out_base = pair_offsets[seed_idx];
    __shared__ int s_counter;
    if (threadIdx.x == 0) s_counter = 0;
    __syncthreads();
    for (int32_t i = u_start; i < u_end; i++) {
        int32_t m = indices[i];
        int32_t m_start = offsets[m], m_end = offsets[m + 1];
        for (int32_t j = m_start + threadIdx.x; j < m_end; j += blockDim.x) {
            int32_t v = indices[j];
            if (v != u) {
                int pos = atomicAdd(&s_counter, 1);
                pair_keys[out_base + pos] = ((int64_t)(uint32_t)u << 32) | (int64_t)(uint32_t)v;
            }
        }
    }
}





__global__ void precompute_seed_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ weights,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    double* __restrict__ seed_degrees
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_seeds) return;

    int32_t u = seeds[warp_id];
    int32_t start = offsets[u], end = offsets[u + 1];

    double local = 0.0;
    for (int i = start + lane; i < end; i += 32) local += weights[i];
    for (int d = 16; d > 0; d >>= 1) local += __shfl_down_sync(0xFFFFFFFF, local, d);
    if (lane == 0) seed_degrees[warp_id] = local;
}





__global__ void compute_scores_warp_grouped_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ first_arr,
    const int32_t* __restrict__ second_arr,
    int num_pairs,
    const double* __restrict__ seed_degrees,
    const int64_t* __restrict__ pair_offsets,
    int num_seeds,
    double* __restrict__ out_scores
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_pairs) return;

    int32_t u = first_arr[warp_id];
    int32_t v = second_arr[warp_id];

    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int32_t v_start = offsets[v], v_end = offsets[v + 1];
    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    int lo = 0, hi = num_seeds;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (pair_offsets[mid] <= (int64_t)warp_id) lo = mid + 1;
        else hi = mid;
    }
    int seed_idx = lo - 1;
    double deg_u = seed_degrees[seed_idx];

    double local_deg_v = 0.0;
    for (int i = v_start + lane; i < v_end; i += 32) local_deg_v += weights[i];
    for (int d = 16; d > 0; d >>= 1) local_deg_v += __shfl_down_sync(0xFFFFFFFF, local_deg_v, d);
    double deg_v = __shfl_sync(0xFFFFFFFF, local_deg_v, 0);

    double min_deg = fmin(deg_u, deg_v);

    int32_t short_start, short_end, long_start, long_end;
    if (u_deg <= v_deg) {
        short_start = u_start; short_end = u_end;
        long_start = v_start; long_end = v_end;
    } else {
        short_start = v_start; short_end = v_end;
        long_start = u_start; long_end = u_end;
    }

    double local_intersection = 0.0;
    for (int i = short_start + lane; i < short_end; i += 32) {
        int32_t target = indices[i];
        double w_short = weights[i];
        int lo2 = long_start, hi2 = long_end;
        while (lo2 < hi2) {
            int mid = (lo2 + hi2) >> 1;
            if (indices[mid] < target) lo2 = mid + 1;
            else hi2 = mid;
        }
        if (lo2 < long_end && indices[lo2] == target)
            local_intersection += fmin(w_short, weights[lo2]);
    }

    for (int d = 16; d > 0; d >>= 1)
        local_intersection += __shfl_down_sync(0xFFFFFFFF, local_intersection, d);

    if (lane == 0)
        out_scores[warp_id] = (min_deg > 0.0) ? (local_intersection / min_deg) : 0.0;
}


__global__ void compute_scores_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ first_arr,
    const int32_t* __restrict__ second_arr,
    int num_pairs,
    double* __restrict__ out_scores
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_pairs) return;

    int32_t u = first_arr[warp_id];
    int32_t v = second_arr[warp_id];
    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int32_t v_start = offsets[v], v_end = offsets[v + 1];
    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    double local_deg_u = 0.0;
    for (int i = u_start + lane; i < u_end; i += 32) local_deg_u += weights[i];
    double local_deg_v = 0.0;
    for (int i = v_start + lane; i < v_end; i += 32) local_deg_v += weights[i];
    for (int d = 16; d > 0; d >>= 1) {
        local_deg_u += __shfl_down_sync(0xFFFFFFFF, local_deg_u, d);
        local_deg_v += __shfl_down_sync(0xFFFFFFFF, local_deg_v, d);
    }
    double deg_u = __shfl_sync(0xFFFFFFFF, local_deg_u, 0);
    double deg_v = __shfl_sync(0xFFFFFFFF, local_deg_v, 0);
    double min_deg = fmin(deg_u, deg_v);

    int32_t short_start, short_end, long_start, long_end;
    if (u_deg <= v_deg) {
        short_start = u_start; short_end = u_end;
        long_start = v_start; long_end = v_end;
    } else {
        short_start = v_start; short_end = v_end;
        long_start = u_start; long_end = u_end;
    }

    double local_intersection = 0.0;
    for (int i = short_start + lane; i < short_end; i += 32) {
        int32_t target = indices[i];
        double w_short = weights[i];
        int lo = long_start, hi = long_end;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (indices[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        if (lo < long_end && indices[lo] == target)
            local_intersection += fmin(w_short, weights[lo]);
    }
    for (int d = 16; d > 0; d >>= 1)
        local_intersection += __shfl_down_sync(0xFFFFFFFF, local_intersection, d);
    if (lane == 0)
        out_scores[warp_id] = (min_deg > 0.0) ? (local_intersection / min_deg) : 0.0;
}


__global__ void compute_scores_from_keys_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int64_t* __restrict__ unique_keys,
    int num_pairs,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    double* __restrict__ out_scores
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_pairs) return;

    int64_t key = unique_keys[warp_id];
    int32_t u = (int32_t)((uint64_t)key >> 32);
    int32_t v = (int32_t)(key & 0xFFFFFFFFULL);
    if (lane == 0) { out_first[warp_id] = u; out_second[warp_id] = v; }

    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int32_t v_start = offsets[v], v_end = offsets[v + 1];
    int32_t u_deg = u_end - u_start, v_deg = v_end - v_start;

    double local_deg_u = 0.0;
    for (int i = u_start + lane; i < u_end; i += 32) local_deg_u += weights[i];
    double local_deg_v = 0.0;
    for (int i = v_start + lane; i < v_end; i += 32) local_deg_v += weights[i];
    for (int d = 16; d > 0; d >>= 1) {
        local_deg_u += __shfl_down_sync(0xFFFFFFFF, local_deg_u, d);
        local_deg_v += __shfl_down_sync(0xFFFFFFFF, local_deg_v, d);
    }
    double deg_u = __shfl_sync(0xFFFFFFFF, local_deg_u, 0);
    double deg_v = __shfl_sync(0xFFFFFFFF, local_deg_v, 0);
    double min_deg = fmin(deg_u, deg_v);

    int32_t ss, se, ls, le;
    if (u_deg <= v_deg) { ss = u_start; se = u_end; ls = v_start; le = v_end; }
    else { ss = v_start; se = v_end; ls = u_start; le = u_end; }

    double li = 0.0;
    for (int i = ss + lane; i < se; i += 32) {
        int32_t target = indices[i];
        double ws = weights[i];
        int lo = ls, hi = le;
        while (lo < hi) { int mid = (lo + hi) >> 1; if (indices[mid] < target) lo = mid + 1; else hi = mid; }
        if (lo < le && indices[lo] == target) li += fmin(ws, weights[lo]);
    }
    for (int d = 16; d > 0; d >>= 1) li += __shfl_down_sync(0xFFFFFFFF, li, d);
    if (lane == 0) out_scores[warp_id] = (min_deg > 0.0) ? (li / min_deg) : 0.0;
}





__global__ void init_sequence_kernel(int32_t* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = idx;
}

__global__ void gather_topk_kernel(
    const int32_t* __restrict__ first, const int32_t* __restrict__ second,
    const double* __restrict__ scores_sorted, const int32_t* __restrict__ perm,
    int32_t* __restrict__ out_first, int32_t* __restrict__ out_second,
    double* __restrict__ out_scores, int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= k) return;
    int32_t src_idx = perm[idx];
    out_first[idx] = first[src_idx];
    out_second[idx] = second[src_idx];
    out_scores[idx] = scores_sorted[idx];
}

__global__ void make_sequence_kernel(int32_t* arr, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = idx;
}





static void prefix_sum(Cache& cache, const int64_t* in, int64_t* out, int n) {
    size_t ts = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, ts, in, out, n);
    ensure_void(cache.cub_temp, cache.cub_temp_cap, ts);
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, ts, in, out, n);
}

static int64_t get_total(const int64_t* d_off, const int64_t* d_cnt, int n) {
    int64_t lo, lc;
    cudaMemcpy(&lo, d_off + n - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lc, d_cnt + n - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
    return lo + lc;
}

static void do_topk(Cache& cache,
                    int32_t* first, int32_t* second, double* scores,
                    int num_pairs, std::optional<std::size_t> topk,
                    int32_t*& out_first, int32_t*& out_second,
                    double*& out_scores, std::size_t& output_count) {
    if (topk.has_value() && *topk < (std::size_t)num_pairs) {
        output_count = *topk;

        ensure_buf(cache.perm_buf, cache.perm_cap, (int64_t)num_pairs);
        ensure_buf(cache.perm_sorted_buf, cache.perm_sorted_cap, (int64_t)num_pairs);
        ensure_buf(cache.scores_sorted_buf, cache.scores_sorted_cap, (int64_t)num_pairs);

        init_sequence_kernel<<<(num_pairs + 255) / 256, 256>>>(cache.perm_buf, num_pairs);

        size_t ts = 0;
        cub::DeviceRadixSort::SortPairsDescending(nullptr, ts,
            (double*)nullptr, (double*)nullptr,
            (int32_t*)nullptr, (int32_t*)nullptr, num_pairs, 0, 64);
        ensure_void(cache.cub_temp, cache.cub_temp_cap, ts);
        cub::DeviceRadixSort::SortPairsDescending(cache.cub_temp, ts,
            scores, cache.scores_sorted_buf,
            cache.perm_buf, cache.perm_sorted_buf, num_pairs, 0, 64);

        cudaMalloc(&out_first, output_count * sizeof(int32_t));
        cudaMalloc(&out_second, output_count * sizeof(int32_t));
        cudaMalloc(&out_scores, output_count * sizeof(double));

        gather_topk_kernel<<<((int)output_count + 255) / 256, 256>>>(
            first, second, cache.scores_sorted_buf, cache.perm_sorted_buf,
            out_first, out_second, out_scores, (int)output_count);

        cudaFree(first);
        cudaFree(second);
        cudaFree(scores);
    } else {
        output_count = num_pairs;
        out_first = first;
        out_second = second;
        out_scores = scores;
    }
}

}  





similarity_result_double_t overlap_all_pairs_similarity(
    const graph32_t& graph,
    const double* edge_weights,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    const double* d_wt = edge_weights;

    
    const int32_t* d_seeds;
    int32_t ns;
    if (vertices != nullptr && num_vertices > 0) {
        d_seeds = vertices;
        ns = (int32_t)num_vertices;
    } else {
        ns = nv;
        ensure_buf(cache.seeds_buf, cache.seeds_cap, (int64_t)ns);
        make_sequence_kernel<<<(ns + 255) / 256, 256>>>(cache.seeds_buf, ns);
        d_seeds = cache.seeds_buf;
    }

    if (ns == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    int bw = (nv + 31) / 32;
    int64_t bb = (int64_t)bw * 4 * ns;
    bool use_bitmap = (bb <= MAX_BITMAP_BYTES);

    int32_t* fo = nullptr;
    int32_t* so = nullptr;
    double* sco = nullptr;
    std::size_t oc = 0;

    if (use_bitmap) {
        
        ensure_buf(cache.bitmap_buf, cache.bitmap_cap, (int64_t)bw * ns);
        ensure_buf(cache.counts_buf, cache.counts_cap, (int64_t)ns);

        bitmap_mark_count_kernel<<<ns, 512>>>(d_off, d_idx, d_seeds, ns,
            cache.bitmap_buf, bw, cache.counts_buf);

        ensure_buf(cache.offsets_buf, cache.offsets_cap, (int64_t)ns);
        prefix_sum(cache, cache.counts_buf, cache.offsets_buf, ns);
        int64_t tu = get_total(cache.offsets_buf, cache.counts_buf, ns);

        if (tu == 0) {
            return {nullptr, nullptr, nullptr, 0};
        }

        
        int32_t* ft = nullptr;
        int32_t* st = nullptr;
        cudaMalloc(&ft, tu * sizeof(int32_t));
        cudaMalloc(&st, tu * sizeof(int32_t));

        bitmap_extract_pairs_kernel<<<ns, 256>>>(cache.bitmap_buf, bw, ns,
            d_seeds, cache.offsets_buf, ft, st);

        ensure_buf(cache.seed_deg_buf, cache.seed_deg_cap, (int64_t)ns);
        {
            int blocks = (int)(((int64_t)ns * 32 + 255) / 256);
            precompute_seed_degrees_kernel<<<blocks, 256>>>(d_off, d_wt, d_seeds, ns,
                cache.seed_deg_buf);
        }

        double* sc = nullptr;
        cudaMalloc(&sc, tu * sizeof(double));
        {
            int blocks = (int)(((int64_t)tu * 32 + 255) / 256);
            compute_scores_warp_grouped_kernel<<<blocks, 256>>>(d_off, d_idx, d_wt,
                ft, st, (int)tu, cache.seed_deg_buf, cache.offsets_buf, ns, sc);
        }

        do_topk(cache, ft, st, sc, (int)tu, topk, fo, so, sco, oc);

    } else {
        
        ensure_buf(cache.counts_buf, cache.counts_cap, (int64_t)ns);
        count_2hop_kernel<<<(ns + 255) / 256, 256>>>(d_off, d_idx, d_seeds, ns,
            cache.counts_buf);

        ensure_buf(cache.offsets_buf, cache.offsets_cap, (int64_t)ns);
        prefix_sum(cache, cache.counts_buf, cache.offsets_buf, ns);
        int64_t tp = get_total(cache.offsets_buf, cache.counts_buf, ns);

        if (tp == 0) {
            return {nullptr, nullptr, nullptr, 0};
        }

        
        ensure_buf(cache.pair_keys_buf, cache.pair_keys_cap, tp);
        generate_pairs_kernel<<<ns, 256>>>(d_off, d_idx, d_seeds, ns,
            cache.offsets_buf, cache.pair_keys_buf);

        
        int bits = 1;
        { int v = nv - 1; while (v > 0) { bits++; v >>= 1; } }
        int eb = 2 * bits;
        if (eb > 64) eb = 64;

        ensure_buf(cache.sorted_keys_buf, cache.sorted_keys_cap, tp);
        {
            size_t sts = 0;
            cub::DeviceRadixSort::SortKeys(nullptr, sts,
                (int64_t*)nullptr, (int64_t*)nullptr, (int)tp, 0, eb);
            ensure_void(cache.cub_temp, cache.cub_temp_cap, sts);
            cub::DeviceRadixSort::SortKeys(cache.cub_temp, sts,
                cache.pair_keys_buf, cache.sorted_keys_buf, (int)tp, 0, eb);
        }

        
        ensure_buf(cache.unique_keys_buf, cache.unique_keys_cap, tp);
        if (!cache.nsel_buf) cudaMalloc(&cache.nsel_buf, sizeof(int32_t));
        {
            size_t uts = 0;
            cub::DeviceSelect::Unique(nullptr, uts,
                (int64_t*)nullptr, (int64_t*)nullptr, (int*)nullptr, (int)tp);
            ensure_void(cache.cub_temp, cache.cub_temp_cap, uts);
            cub::DeviceSelect::Unique(cache.cub_temp, uts,
                cache.sorted_keys_buf, cache.unique_keys_buf,
                cache.nsel_buf, (int)tp);
        }

        int nu;
        cudaMemcpy(&nu, cache.nsel_buf, sizeof(int32_t), cudaMemcpyDeviceToHost);

        if (nu == 0) {
            return {nullptr, nullptr, nullptr, 0};
        }

        
        int32_t* ft = nullptr;
        int32_t* st = nullptr;
        double* sc = nullptr;
        cudaMalloc(&ft, nu * sizeof(int32_t));
        cudaMalloc(&st, nu * sizeof(int32_t));
        cudaMalloc(&sc, nu * sizeof(double));

        {
            int blocks = (int)(((int64_t)nu * 32 + 255) / 256);
            compute_scores_from_keys_kernel<<<blocks, 256>>>(d_off, d_idx, d_wt,
                cache.unique_keys_buf, nu, ft, st, sc);
        }

        do_topk(cache, ft, st, sc, nu, topk, fo, so, sco, oc);
    }

    return {fo, so, sco, oc};
}

}  
