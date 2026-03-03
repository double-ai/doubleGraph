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

namespace aai {

namespace {

struct Cache : Cacheable {};



__device__ __forceinline__ int lower_bound_dev(const int32_t* arr, int size, int32_t target)
{
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}



__global__ void weighted_degrees_kernel(const int32_t* __restrict__ offsets,
                                       const float* __restrict__ weights,
                                       float* __restrict__ degrees,
                                       int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    float sum = 0.0f;
    int start = offsets[v];
    int end = offsets[v + 1];
    for (int e = start; e < end; e++) sum += weights[e];
    degrees[v] = sum;
}

__global__ void weighted_degrees_vertices_kernel(const int32_t* __restrict__ offsets,
                                               const float* __restrict__ weights,
                                               const int32_t* __restrict__ vertices,
                                               int nverts,
                                               float* __restrict__ degrees)
{
    int vidx = blockIdx.x;
    if (vidx >= nverts) return;

    int32_t v = vertices[vidx];
    int start = offsets[v];
    int end = offsets[v + 1];

    float sum = 0.0f;
    for (int e = start + threadIdx.x; e < end; e += blockDim.x) sum += weights[e];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    __shared__ float warp_sums[8];
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();

    if (warp == 0) {
        float ws = (lane < 8) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            ws += __shfl_down_sync(0xffffffff, ws, offset);
        }
        if (lane == 0) degrees[v] = ws;
    }
}

__global__ void count_pairs_kernel(const int32_t* __restrict__ offsets,
                                  const int32_t* __restrict__ indices,
                                  const int32_t* __restrict__ seeds,
                                  int num_seeds,
                                  int64_t* __restrict__ counts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;

    int32_t u = seeds ? seeds[idx] : idx;

    int64_t count = 0;
    int u_start = offsets[u];
    int u_end = offsets[u + 1];

    for (int e = u_start; e < u_end; e++) {
        int32_t x = indices[e];
        count += (int64_t)(offsets[x + 1] - offsets[x] - 1);
    }

    counts[idx] = count;
}

__global__ void generate_pairs_block_i64_kernel(const int32_t* __restrict__ offsets,
                                               const int32_t* __restrict__ indices,
                                               const int32_t* __restrict__ seeds,
                                               int num_seeds,
                                               uint32_t key_bits,
                                               const int64_t* __restrict__ pair_offsets,
                                               int64_t* __restrict__ keys)
{
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    int32_t u = seeds ? seeds[seed_idx] : seed_idx;
    int64_t base_pos = pair_offsets[seed_idx];

    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int u_deg = u_end - u_start;

    __shared__ int write_counter;
    if (threadIdx.x == 0) write_counter = 0;
    __syncthreads();

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    for (int ne = warp; ne < u_deg; ne += warps_per_block) {
        int32_t x = indices[u_start + ne];
        int x_start = offsets[x];
        int x_end = offsets[x + 1];
        int x_deg = x_end - x_start;
        int len = x_deg - 1;

        int base = 0;
        int pos_u = 0;
        if (lane == 0) {
            base = atomicAdd(&write_counter, len);
            pos_u = lower_bound_dev(indices + x_start, x_deg, u);
        }
        base = __shfl_sync(0xffffffff, base, 0);
        pos_u = __shfl_sync(0xffffffff, pos_u, 0);

        for (int f = lane; f < x_deg; f += 32) {
            if (f == pos_u) continue;
            int out_f = f - (f > pos_u);
            int32_t v = indices[x_start + f];
            keys[base_pos + (int64_t)base + out_f] = (int64_t)(((uint64_t)u << key_bits) | (uint32_t)v);
        }
    }
}

__global__ void generate_pairs_block_u32_kernel(const int32_t* __restrict__ offsets,
                                               const int32_t* __restrict__ indices,
                                               const int32_t* __restrict__ seeds,
                                               int num_seeds,
                                               uint32_t key_bits,
                                               const int64_t* __restrict__ pair_offsets,
                                               uint32_t* __restrict__ keys)
{
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    int32_t u = seeds[seed_idx];
    int64_t base_pos = pair_offsets[seed_idx];

    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int u_deg = u_end - u_start;

    __shared__ int write_counter;
    if (threadIdx.x == 0) write_counter = 0;
    __syncthreads();

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    uint32_t seed_prefix = ((uint32_t)seed_idx) << key_bits;

    for (int ne = warp; ne < u_deg; ne += warps_per_block) {
        int32_t x = indices[u_start + ne];
        int x_start = offsets[x];
        int x_end = offsets[x + 1];
        int x_deg = x_end - x_start;
        int len = x_deg - 1;

        int base = 0;
        int pos_u = 0;
        if (lane == 0) {
            base = atomicAdd(&write_counter, len);
            pos_u = lower_bound_dev(indices + x_start, x_deg, u);
        }
        base = __shfl_sync(0xffffffff, base, 0);
        pos_u = __shfl_sync(0xffffffff, pos_u, 0);

        for (int f = lane; f < x_deg; f += 32) {
            if (f == pos_u) continue;
            int out_f = f - (f > pos_u);
            uint32_t v = (uint32_t)indices[x_start + f];
            keys[base_pos + (int64_t)base + out_f] = seed_prefix | v;
        }
    }
}

__global__ void extract_v_i64_kernel(const int64_t* __restrict__ pair_keys,
                                    int32_t* __restrict__ out_v,
                                    int n,
                                    uint32_t key_mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint64_t key = (uint64_t)pair_keys[idx];
    out_v[idx] = (int32_t)((uint32_t)key & key_mask);
}

__global__ void extract_v_u32_kernel(const uint32_t* __restrict__ pair_keys,
                                    int32_t* __restrict__ out_v,
                                    int n,
                                    uint32_t key_mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out_v[idx] = (int32_t)(pair_keys[idx] & key_mask);
}

__global__ void compute_overlap_i64_kernel(const int32_t* __restrict__ offsets,
                                          const int32_t* __restrict__ indices,
                                          const float* __restrict__ weights,
                                          const float* __restrict__ weighted_degrees,
                                          uint32_t key_bits,
                                          uint32_t key_mask,
                                          const int64_t* __restrict__ pair_keys,
                                          int32_t* __restrict__ out_first,
                                          int32_t* __restrict__ out_second,
                                          float* __restrict__ out_scores,
                                          int64_t num_pairs)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    uint64_t key = (uint64_t)pair_keys[idx];
    int32_t u = (int32_t)(key >> key_bits);
    int32_t v = (int32_t)((uint32_t)key & key_mask);

    out_first[idx] = u;
    out_second[idx] = v;

    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int v_start = offsets[v];
    int v_end = offsets[v + 1];

    int u_deg = u_end - u_start;
    int v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        out_scores[idx] = 0.0f;
        return;
    }

    const int32_t* u_nbrs = indices + u_start;
    const int32_t* v_nbrs = indices + v_start;
    const float* u_wts = weights + u_start;
    const float* v_wts = weights + v_start;

    int i_start = lower_bound_dev(u_nbrs, u_deg, v_nbrs[0]);
    if (i_start >= u_deg) {
        out_scores[idx] = 0.0f;
        return;
    }
    int j_start = lower_bound_dev(v_nbrs, v_deg, u_nbrs[i_start]);
    if (j_start >= v_deg) {
        out_scores[idx] = 0.0f;
        return;
    }

    int32_t max_val = (u_nbrs[u_deg - 1] < v_nbrs[v_deg - 1]) ? u_nbrs[u_deg - 1] : v_nbrs[v_deg - 1];
    int i_end = lower_bound_dev(u_nbrs + i_start, u_deg - i_start, max_val + 1) + i_start;
    int j_end = lower_bound_dev(v_nbrs + j_start, v_deg - j_start, max_val + 1) + j_start;

    int size_a = i_end - i_start;
    int size_b = j_end - j_start;

    float w_intersect = 0.0f;

    if (size_a > 0 && size_b > 0) {
        const int32_t* a_nbrs;
        const float* a_wts;
        int a_size;
        const int32_t* b_nbrs;
        const float* b_wts;
        int b_size;

        if (size_a <= size_b) {
            a_nbrs = u_nbrs + i_start;
            a_wts = u_wts + i_start;
            a_size = size_a;
            b_nbrs = v_nbrs + j_start;
            b_wts = v_wts + j_start;
            b_size = size_b;
        } else {
            a_nbrs = v_nbrs + j_start;
            a_wts = v_wts + j_start;
            a_size = size_b;
            b_nbrs = u_nbrs + i_start;
            b_wts = u_wts + i_start;
            b_size = size_a;
        }

        int ratio = b_size / (a_size > 0 ? a_size : 1);

        if (ratio > 16) {
            int j = 0;
            for (int i = 0; i < a_size && j < b_size; i++) {
                int32_t target = a_nbrs[i];
                int pos = j;
                int step = 1;
                while (pos + step < b_size && b_nbrs[pos + step] < target) {
                    pos += step;
                    step <<= 1;
                }
                int lo = pos;
                int hi = (pos + step < b_size) ? (pos + step + 1) : b_size;
                while (lo < hi) {
                    int mid = lo + ((hi - lo) >> 1);
                    if (b_nbrs[mid] < target) lo = mid + 1;
                    else hi = mid;
                }
                j = lo;
                if (j < b_size && b_nbrs[j] == target) {
                    w_intersect += fminf(a_wts[i], b_wts[j]);
                    j++;
                }
            }
        } else {
            int i = 0;
            int j = 0;
            while (i < a_size && j < b_size) {
                int32_t aa = a_nbrs[i];
                int32_t bb = b_nbrs[j];
                if (aa == bb) {
                    w_intersect += fminf(a_wts[i], b_wts[j]);
                    i++;
                    j++;
                } else if (aa < bb) {
                    i++;
                } else {
                    j++;
                }
            }
        }
    }

    float min_deg = fminf(weighted_degrees[u], weighted_degrees[v]);
    out_scores[idx] = (min_deg > 0.0f) ? (w_intersect / min_deg) : 0.0f;
}

__global__ void compute_overlap_u32_kernel(const int32_t* __restrict__ offsets,
                                          const int32_t* __restrict__ indices,
                                          const float* __restrict__ weights,
                                          const float* __restrict__ weighted_degrees,
                                          uint32_t key_bits,
                                          uint32_t key_mask,
                                          const uint32_t* __restrict__ pair_keys,
                                          const int32_t* __restrict__ seeds,
                                          int32_t* __restrict__ out_first,
                                          int32_t* __restrict__ out_second,
                                          float* __restrict__ out_scores,
                                          int64_t num_pairs)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    uint32_t key = pair_keys[idx];
    int32_t seed_idx = (int32_t)(key >> key_bits);
    int32_t v = (int32_t)(key & key_mask);
    int32_t u = seeds[seed_idx];

    out_first[idx] = u;
    out_second[idx] = v;

    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int v_start = offsets[v];
    int v_end = offsets[v + 1];

    int u_deg = u_end - u_start;
    int v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        out_scores[idx] = 0.0f;
        return;
    }

    const int32_t* u_nbrs = indices + u_start;
    const int32_t* v_nbrs = indices + v_start;
    const float* u_wts = weights + u_start;
    const float* v_wts = weights + v_start;

    int i_start = lower_bound_dev(u_nbrs, u_deg, v_nbrs[0]);
    if (i_start >= u_deg) {
        out_scores[idx] = 0.0f;
        return;
    }
    int j_start = lower_bound_dev(v_nbrs, v_deg, u_nbrs[i_start]);
    if (j_start >= v_deg) {
        out_scores[idx] = 0.0f;
        return;
    }

    int32_t max_val = (u_nbrs[u_deg - 1] < v_nbrs[v_deg - 1]) ? u_nbrs[u_deg - 1] : v_nbrs[v_deg - 1];
    int i_end = lower_bound_dev(u_nbrs + i_start, u_deg - i_start, max_val + 1) + i_start;
    int j_end = lower_bound_dev(v_nbrs + j_start, v_deg - j_start, max_val + 1) + j_start;

    int size_a = i_end - i_start;
    int size_b = j_end - j_start;

    float w_intersect = 0.0f;

    if (size_a > 0 && size_b > 0) {
        const int32_t* a_nbrs;
        const float* a_wts;
        int a_size;
        const int32_t* b_nbrs;
        const float* b_wts;
        int b_size;

        if (size_a <= size_b) {
            a_nbrs = u_nbrs + i_start;
            a_wts = u_wts + i_start;
            a_size = size_a;
            b_nbrs = v_nbrs + j_start;
            b_wts = v_wts + j_start;
            b_size = size_b;
        } else {
            a_nbrs = v_nbrs + j_start;
            a_wts = v_wts + j_start;
            a_size = size_b;
            b_nbrs = u_nbrs + i_start;
            b_wts = u_wts + i_start;
            b_size = size_a;
        }

        int ratio = b_size / (a_size > 0 ? a_size : 1);

        if (ratio > 16) {
            int j = 0;
            for (int i = 0; i < a_size && j < b_size; i++) {
                int32_t target = a_nbrs[i];
                int pos = j;
                int step = 1;
                while (pos + step < b_size && b_nbrs[pos + step] < target) {
                    pos += step;
                    step <<= 1;
                }
                int lo = pos;
                int hi = (pos + step < b_size) ? (pos + step + 1) : b_size;
                while (lo < hi) {
                    int mid = lo + ((hi - lo) >> 1);
                    if (b_nbrs[mid] < target) lo = mid + 1;
                    else hi = mid;
                }
                j = lo;
                if (j < b_size && b_nbrs[j] == target) {
                    w_intersect += fminf(a_wts[i], b_wts[j]);
                    j++;
                }
            }
        } else {
            int i = 0;
            int j = 0;
            while (i < a_size && j < b_size) {
                int32_t aa = a_nbrs[i];
                int32_t bb = b_nbrs[j];
                if (aa == bb) {
                    w_intersect += fminf(a_wts[i], b_wts[j]);
                    i++;
                    j++;
                } else if (aa < bb) {
                    i++;
                } else {
                    j++;
                }
            }
        }
    }

    float min_deg = fminf(weighted_degrees[u], weighted_degrees[v]);
    out_scores[idx] = (min_deg > 0.0f) ? (w_intersect / min_deg) : 0.0f;
}

__global__ void iota_kernel(int32_t* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = idx;
}

__global__ void gather_results_kernel(const int32_t* __restrict__ src_first,
                                     const int32_t* __restrict__ src_second,
                                     const float* __restrict__ src_scores,
                                     const int32_t* __restrict__ gather_idx,
                                     int32_t* __restrict__ dst_first,
                                     int32_t* __restrict__ dst_second,
                                     float* __restrict__ dst_scores,
                                     int64_t count)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int si = gather_idx[idx];
    dst_first[idx] = src_first[si];
    dst_second[idx] = src_second[si];
    dst_scores[idx] = src_scores[si];
}

constexpr int TOPK_HIST_BINS = 1024;
constexpr int TOPK_HIST_SHIFT = 22;

__global__ void hist_scores_kernel(const float* __restrict__ scores, int n, uint32_t* __restrict__ hist)
{
    __shared__ uint32_t sh[TOPK_HIST_BINS];
    for (int i = threadIdx.x; i < TOPK_HIST_BINS; i += blockDim.x) sh[i] = 0;
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        uint32_t b = __float_as_uint(scores[i]) >> TOPK_HIST_SHIFT;
        atomicAdd(&sh[b], 1u);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < TOPK_HIST_BINS; i += blockDim.x) {
        uint32_t v = sh[i];
        if (v) atomicAdd(&hist[i], v);
    }
}

__global__ void find_cutoff_kernel(const uint32_t* __restrict__ hist, int k, int* __restrict__ cutoff_bin, int* __restrict__ cand_count)
{
    if (threadIdx.x == 0) {
        int cum = 0;
        int cut = 0;
        for (int b = TOPK_HIST_BINS - 1; b >= 0; --b) {
            cum += (int)hist[b];
            if (cum >= k) { cut = b; break; }
        }
        *cutoff_bin = cut;
        *cand_count = cum;
    }
}

__global__ void make_flags_kernel(const float* __restrict__ scores, int n, int cutoff_bin, uint8_t* __restrict__ flags)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t b = __float_as_uint(scores[idx]) >> TOPK_HIST_SHIFT;
    flags[idx] = (uint8_t)(b >= (uint32_t)cutoff_bin);
}



static uint32_t ceil_log2_u32(uint32_t x)
{
    if (x <= 1) return 0;
    uint32_t v = x - 1;
    uint32_t bits = 0;
    while (v) {
        bits++;
        v >>= 1;
    }
    return bits;
}

}  

similarity_result_float_t overlap_all_pairs_similarity(
    const graph32_t& graph,
    const float* edge_weights,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n_verts = graph.number_of_vertices;
    const float* d_weights = edge_weights;

    int num_seeds;
    const int32_t* d_seeds;
    if (vertices != nullptr) {
        num_seeds = (int)num_vertices;
        d_seeds = vertices;
    } else {
        num_seeds = n_verts;
        d_seeds = nullptr;
    }

    uint32_t key_bits = ceil_log2_u32((uint32_t)n_verts);
    uint32_t key_mask = (key_bits == 0) ? 0u : ((key_bits == 32) ? 0xFFFFFFFFu : ((1u << key_bits) - 1u));

    uint32_t seed_bits = (d_seeds != nullptr) ? ceil_log2_u32((uint32_t)num_seeds) : 0;
    bool use_u32_keys = (d_seeds != nullptr) && ((key_bits + seed_bits) <= 32);

    
    int64_t* d_counts = nullptr;
    cudaMalloc(&d_counts, (num_seeds + 1) * sizeof(int64_t));

    {
        int block = 256;
        int grid = (num_seeds + block - 1) / block;
        count_pairs_kernel<<<grid, block>>>(d_offsets, d_indices, d_seeds, num_seeds, d_counts);
    }
    cudaMemsetAsync(d_counts + num_seeds, 0, sizeof(int64_t));

    size_t scan_temp_sz = 0;
    cub::DeviceScan::ExclusiveSum((void*)nullptr, scan_temp_sz, (int64_t*)nullptr, (int64_t*)nullptr, num_seeds + 1);
    void* scan_temp = nullptr;
    cudaMalloc(&scan_temp, scan_temp_sz + 8);
    cub::DeviceScan::ExclusiveSum(scan_temp, scan_temp_sz, d_counts, d_counts, num_seeds + 1);
    cudaFree(scan_temp);

    int64_t total_pairs = 0;
    cudaMemcpy(&total_pairs, d_counts + num_seeds, sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (total_pairs <= 0) {
        cudaFree(d_counts);
        return {nullptr, nullptr, nullptr, 0};
    }

    int n_paths = (int)total_pairs;

    
    int num_unique = 0;
    uint32_t* unique_keys_u32 = nullptr;
    int64_t* unique_keys_i64 = nullptr;

    if (use_u32_keys) {
        uint32_t* keys = nullptr;
        cudaMalloc(&keys, total_pairs * sizeof(uint32_t));
        generate_pairs_block_u32_kernel<<<num_seeds, 256>>>(d_offsets, d_indices, d_seeds, num_seeds,
                                                             key_bits, d_counts, keys);
        cudaFree(d_counts);

        int end_bit = (int)(key_bits + seed_bits);
        size_t sort_temp_sz = 0;
        cub::DeviceRadixSort::SortKeys((void*)nullptr, sort_temp_sz, (uint32_t*)nullptr, (uint32_t*)nullptr, n_paths);
        void* sort_temp = nullptr;
        cudaMalloc(&sort_temp, sort_temp_sz + 8);
        uint32_t* sorted_keys = nullptr;
        cudaMalloc(&sorted_keys, total_pairs * sizeof(uint32_t));
        cub::DeviceRadixSort::SortKeys(sort_temp, sort_temp_sz, keys, sorted_keys, n_paths, 0, end_bit);
        cudaFree(keys);
        cudaFree(sort_temp);

        size_t unique_temp_sz = 0;
        cub::DeviceSelect::Unique((void*)nullptr, unique_temp_sz, (uint32_t*)nullptr, (uint32_t*)nullptr, (int*)nullptr, n_paths);
        void* unique_temp = nullptr;
        cudaMalloc(&unique_temp, unique_temp_sz + 8);
        cudaMalloc(&unique_keys_u32, total_pairs * sizeof(uint32_t));
        int* num_sel = nullptr;
        cudaMalloc(&num_sel, sizeof(int));
        cub::DeviceSelect::Unique(unique_temp, unique_temp_sz, sorted_keys, unique_keys_u32, num_sel, n_paths);
        cudaMemcpy(&num_unique, num_sel, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(sorted_keys);
        cudaFree(unique_temp);
        cudaFree(num_sel);
    } else {
        int64_t* keys = nullptr;
        cudaMalloc(&keys, total_pairs * sizeof(int64_t));
        generate_pairs_block_i64_kernel<<<num_seeds, 256>>>(d_offsets, d_indices, d_seeds, num_seeds,
                                                             key_bits, d_counts, keys);
        cudaFree(d_counts);

        int end_bit = (int)(key_bits * 2);
        size_t sort_temp_sz = 0;
        cub::DeviceRadixSort::SortKeys((void*)nullptr, sort_temp_sz, (int64_t*)nullptr, (int64_t*)nullptr, n_paths);
        void* sort_temp = nullptr;
        cudaMalloc(&sort_temp, sort_temp_sz + 8);
        int64_t* sorted_keys = nullptr;
        cudaMalloc(&sorted_keys, total_pairs * sizeof(int64_t));
        cub::DeviceRadixSort::SortKeys(sort_temp, sort_temp_sz, keys, sorted_keys, n_paths, 0, end_bit);
        cudaFree(keys);
        cudaFree(sort_temp);

        size_t unique_temp_sz = 0;
        cub::DeviceSelect::Unique((void*)nullptr, unique_temp_sz, (int64_t*)nullptr, (int64_t*)nullptr, (int*)nullptr, n_paths);
        void* unique_temp = nullptr;
        cudaMalloc(&unique_temp, unique_temp_sz + 8);
        cudaMalloc(&unique_keys_i64, total_pairs * sizeof(int64_t));
        int* num_sel = nullptr;
        cudaMalloc(&num_sel, sizeof(int));
        cub::DeviceSelect::Unique(unique_temp, unique_temp_sz, sorted_keys, unique_keys_i64, num_sel, n_paths);
        cudaMemcpy(&num_unique, num_sel, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(sorted_keys);
        cudaFree(unique_temp);
        cudaFree(num_sel);
    }

    if (num_unique == 0) {
        if (unique_keys_u32) cudaFree(unique_keys_u32);
        if (unique_keys_i64) cudaFree(unique_keys_i64);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    bool use_touched_degrees = (vertices != nullptr) && (n_verts >= 500000);
    float* deg = nullptr;
    cudaMalloc(&deg, n_verts * sizeof(float));

    if (!use_touched_degrees) {
        int block = 256;
        int grid = (n_verts + block - 1) / block;
        weighted_degrees_kernel<<<grid, block>>>(d_offsets, d_weights, deg, n_verts);
    } else {
        weighted_degrees_vertices_kernel<<<num_seeds, 256>>>(d_offsets, d_weights, d_seeds, num_seeds, deg);

        int32_t* v_all = nullptr;
        cudaMalloc(&v_all, num_unique * sizeof(int32_t));
        {
            int block = 256;
            int grid = (num_unique + block - 1) / block;
            if (use_u32_keys) {
                extract_v_u32_kernel<<<grid, block>>>(unique_keys_u32, v_all, num_unique, key_mask);
            } else {
                extract_v_i64_kernel<<<grid, block>>>(unique_keys_i64, v_all, num_unique, key_mask);
            }
        }

        size_t sort_v_temp_sz = 0;
        cub::DeviceRadixSort::SortKeys((void*)nullptr, sort_v_temp_sz, (uint32_t*)nullptr, (uint32_t*)nullptr, num_unique);
        void* sort_v_temp = nullptr;
        cudaMalloc(&sort_v_temp, sort_v_temp_sz + 8);
        int32_t* v_sorted = nullptr;
        cudaMalloc(&v_sorted, num_unique * sizeof(int32_t));
        cub::DeviceRadixSort::SortKeys(sort_v_temp, sort_v_temp_sz,
                                        reinterpret_cast<const uint32_t*>(v_all),
                                        reinterpret_cast<uint32_t*>(v_sorted),
                                        num_unique, 0, (int)key_bits);
        cudaFree(v_all);
        cudaFree(sort_v_temp);

        size_t uniq_v_temp_sz = 0;
        cub::DeviceSelect::Unique((void*)nullptr, uniq_v_temp_sz, (int32_t*)nullptr, (int32_t*)nullptr, (int*)nullptr, num_unique);
        void* uniq_v_temp = nullptr;
        cudaMalloc(&uniq_v_temp, uniq_v_temp_sz + 8);
        int32_t* v_unique = nullptr;
        cudaMalloc(&v_unique, num_unique * sizeof(int32_t));
        int* v_num_sel = nullptr;
        cudaMalloc(&v_num_sel, sizeof(int));
        cub::DeviceSelect::Unique(uniq_v_temp, uniq_v_temp_sz, v_sorted, v_unique, v_num_sel, num_unique);

        int num_unique_v = 0;
        cudaMemcpy(&num_unique_v, v_num_sel, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(v_sorted);
        cudaFree(uniq_v_temp);
        cudaFree(v_num_sel);

        weighted_degrees_vertices_kernel<<<num_unique_v, 256>>>(d_offsets, d_weights, v_unique, num_unique_v, deg);
        cudaFree(v_unique);
    }

    
    int32_t* first = nullptr;
    int32_t* second = nullptr;
    float* scores = nullptr;
    cudaMalloc(&first, num_unique * sizeof(int32_t));
    cudaMalloc(&second, num_unique * sizeof(int32_t));
    cudaMalloc(&scores, num_unique * sizeof(float));

    {
        int block = 256;
        int grid = (int)(((int64_t)num_unique + block - 1) / block);
        if (use_u32_keys) {
            compute_overlap_u32_kernel<<<grid, block>>>(d_offsets, d_indices, d_weights, deg,
                                                         key_bits, key_mask, unique_keys_u32, d_seeds,
                                                         first, second, scores, num_unique);
        } else {
            compute_overlap_i64_kernel<<<grid, block>>>(d_offsets, d_indices, d_weights, deg,
                                                         key_bits, key_mask, unique_keys_i64,
                                                         first, second, scores, num_unique);
        }
    }

    if (unique_keys_u32) cudaFree(unique_keys_u32);
    if (unique_keys_i64) cudaFree(unique_keys_i64);
    cudaFree(deg);

    
    int64_t result_count = num_unique;

    if (topk.has_value() && num_unique > (int)topk.value()) {
        int k = (int)topk.value();
        bool use_partial_topk = (k <= 1024) && (num_unique >= 200000);

        if (use_partial_topk) {
            uint32_t* hist = nullptr;
            cudaMalloc(&hist, 1024 * sizeof(uint32_t));
            cudaMemsetAsync(hist, 0, 1024 * sizeof(uint32_t));
            hist_scores_kernel<<<120, 256>>>(scores, num_unique, hist);

            int* cutoff_d = nullptr;
            int* cand_count_d = nullptr;
            cudaMalloc(&cutoff_d, sizeof(int));
            cudaMalloc(&cand_count_d, sizeof(int));
            find_cutoff_kernel<<<1, 32>>>(hist, k, cutoff_d, cand_count_d);

            int cutoff_bin = 0;
            int cand_count = 0;
            cudaMemcpy(&cutoff_bin, cutoff_d, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&cand_count, cand_count_d, sizeof(int), cudaMemcpyDeviceToHost);
            (void)cand_count;
            cudaFree(cutoff_d);
            cudaFree(cand_count_d);

            uint8_t* flags = nullptr;
            cudaMalloc(&flags, num_unique * sizeof(uint8_t));
            {
                int block = 256;
                int grid = (num_unique + block - 1) / block;
                make_flags_kernel<<<grid, block>>>(scores, num_unique, cutoff_bin, flags);
            }

            int32_t* idx = nullptr;
            cudaMalloc(&idx, num_unique * sizeof(int32_t));
            {
                int block = 256;
                int grid = (num_unique + block - 1) / block;
                iota_kernel<<<grid, block>>>(idx, num_unique);
            }

            size_t sel_temp_sz = 0;
            cub::DeviceSelect::Flagged((void*)nullptr, sel_temp_sz,
                                        (int32_t*)nullptr, (uint8_t*)nullptr,
                                        (int32_t*)nullptr, (int*)nullptr, num_unique);
            void* sel_temp = nullptr;
            cudaMalloc(&sel_temp, sel_temp_sz + 8);
            int32_t* sel_idx = nullptr;
            cudaMalloc(&sel_idx, num_unique * sizeof(int32_t));
            int* sel_n = nullptr;
            cudaMalloc(&sel_n, sizeof(int));

            cub::DeviceSelect::Flagged(sel_temp, sel_temp_sz, idx, flags, sel_idx, sel_n, num_unique);

            int num_cand = 0;
            cudaMemcpy(&num_cand, sel_n, sizeof(int), cudaMemcpyDeviceToHost);

            if (num_cand < k) {
                use_partial_topk = false;
            }

            if (use_partial_topk && num_cand > num_unique / 2) {
                use_partial_topk = false;
            }

            if (use_partial_topk) {
                int32_t* cand_first = nullptr;
                int32_t* cand_second = nullptr;
                float* cand_scores = nullptr;
                cudaMalloc(&cand_first, num_cand * sizeof(int32_t));
                cudaMalloc(&cand_second, num_cand * sizeof(int32_t));
                cudaMalloc(&cand_scores, num_cand * sizeof(float));

                {
                    int block = 256;
                    int grid = (int)(((int64_t)num_cand + block - 1) / block);
                    gather_results_kernel<<<grid, block>>>(first, second, scores, sel_idx,
                                                            cand_first, cand_second, cand_scores, num_cand);
                }

                int32_t* cand_iota = nullptr;
                cudaMalloc(&cand_iota, num_cand * sizeof(int32_t));
                {
                    int block = 256;
                    int grid = (num_cand + block - 1) / block;
                    iota_kernel<<<grid, block>>>(cand_iota, num_cand);
                }

                size_t topk_temp_sz = 0;
                cub::DeviceRadixSort::SortPairsDescending((void*)nullptr, topk_temp_sz,
                                                           (float*)nullptr, (float*)nullptr,
                                                           (int32_t*)nullptr, (int32_t*)nullptr, num_cand);
                void* topk_temp = nullptr;
                cudaMalloc(&topk_temp, topk_temp_sz + 8);
                int32_t* sorted_idx = nullptr;
                float* sorted_scores = nullptr;
                cudaMalloc(&sorted_idx, num_cand * sizeof(int32_t));
                cudaMalloc(&sorted_scores, num_cand * sizeof(float));

                cub::DeviceRadixSort::SortPairsDescending(topk_temp, topk_temp_sz,
                                                           cand_scores, sorted_scores,
                                                           cand_iota, sorted_idx, num_cand);

                int32_t* final_first = nullptr;
                int32_t* final_second = nullptr;
                float* final_scores = nullptr;
                cudaMalloc(&final_first, k * sizeof(int32_t));
                cudaMalloc(&final_second, k * sizeof(int32_t));
                cudaMalloc(&final_scores, k * sizeof(float));

                {
                    int block = 256;
                    int grid = (k + block - 1) / block;
                    gather_results_kernel<<<grid, block>>>(cand_first, cand_second, cand_scores, sorted_idx,
                                                            final_first, final_second, final_scores, k);
                }

                cudaFree(first);
                cudaFree(second);
                cudaFree(scores);
                first = final_first;
                second = final_second;
                scores = final_scores;
                result_count = k;

                cudaFree(cand_first);
                cudaFree(cand_second);
                cudaFree(cand_scores);
                cudaFree(cand_iota);
                cudaFree(topk_temp);
                cudaFree(sorted_idx);
                cudaFree(sorted_scores);
            }

            cudaFree(hist);
            cudaFree(flags);
            cudaFree(idx);
            cudaFree(sel_temp);
            cudaFree(sel_idx);
            cudaFree(sel_n);
        }

        if (!use_partial_topk) {
            int32_t* idx = nullptr;
            cudaMalloc(&idx, num_unique * sizeof(int32_t));
            {
                int block = 256;
                int grid = (num_unique + block - 1) / block;
                iota_kernel<<<grid, block>>>(idx, num_unique);
            }

            size_t topk_temp_sz = 0;
            cub::DeviceRadixSort::SortPairsDescending((void*)nullptr, topk_temp_sz,
                                                       (float*)nullptr, (float*)nullptr,
                                                       (int32_t*)nullptr, (int32_t*)nullptr, num_unique);
            void* topk_temp = nullptr;
            cudaMalloc(&topk_temp, topk_temp_sz + 8);
            float* sorted_scores = nullptr;
            int32_t* sorted_idx = nullptr;
            cudaMalloc(&sorted_scores, num_unique * sizeof(float));
            cudaMalloc(&sorted_idx, num_unique * sizeof(int32_t));

            cub::DeviceRadixSort::SortPairsDescending(topk_temp, topk_temp_sz,
                                                       scores, sorted_scores,
                                                       idx, sorted_idx, num_unique);

            int32_t* final_first = nullptr;
            int32_t* final_second = nullptr;
            float* final_scores = nullptr;
            cudaMalloc(&final_first, k * sizeof(int32_t));
            cudaMalloc(&final_second, k * sizeof(int32_t));
            cudaMalloc(&final_scores, k * sizeof(float));

            {
                int block = 256;
                int grid = (k + block - 1) / block;
                gather_results_kernel<<<grid, block>>>(first, second, scores, sorted_idx,
                                                        final_first, final_second, final_scores, k);
            }

            cudaFree(first);
            cudaFree(second);
            cudaFree(scores);
            first = final_first;
            second = final_second;
            scores = final_scores;
            result_count = k;

            cudaFree(idx);
            cudaFree(topk_temp);
            cudaFree(sorted_scores);
            cudaFree(sorted_idx);
        }
    }

    return {first, second, scores, (std::size_t)result_count};
}

}  
