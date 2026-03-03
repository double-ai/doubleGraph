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
#include <optional>

namespace aai {

namespace {

struct Cache : Cacheable {};





template <typename T>
__device__ __forceinline__ int lower_bound_dev_t(const T* __restrict__ arr, int n, T key)
{
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    int mid = lo + ((hi - lo) >> 1);
    T v = arr[mid];
    if (v < key)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

__device__ __forceinline__ int lower_bound_i32(const int32_t* __restrict__ arr, int n, int32_t key)
{
  return lower_bound_dev_t<int32_t>(arr, n, key);
}





__device__ __forceinline__ int warp_intersect_count(const int32_t* __restrict__ a,
                                                    int size_a,
                                                    const int32_t* __restrict__ b,
                                                    int size_b,
                                                    bool is_multigraph)
{
  int lane = threadIdx.x & 31;

  if (size_a == 0 || size_b == 0) return 0;
  if (a[size_a - 1] < b[0] || b[size_b - 1] < a[0]) return 0;

  const int32_t* probe;
  const int32_t* search;
  int probe_size;
  int search_size;
  if (size_a <= size_b) {
    probe = a;
    search = b;
    probe_size = size_a;
    search_size = size_b;
  } else {
    probe = b;
    search = a;
    probe_size = size_b;
    search_size = size_a;
  }

  int local = 0;
  for (int i = lane; i < probe_size; i += 32) {
    int32_t target = probe[i];

    int rank = 0;
    if (is_multigraph) {
      int lo = 0;
      int hi = i;
      while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (probe[mid] < target) lo = mid + 1;
        else hi = mid;
      }
      rank = i - lo;
    }

    int lb = lower_bound_i32(search, search_size, target);
    int pos = lb + rank;
    if (pos < search_size && search[pos] == target) local++;
  }

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    local += __shfl_down_sync(0xffffffff, local, off);
  }
  return local;
}





__global__ void gather_seed_deg64_kernel(const int32_t* __restrict__ offsets,
                                         const int32_t* __restrict__ seeds,
                                         int32_t num_seeds,
                                         bool all_seeds,
                                         int64_t* __restrict__ deg_out)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < num_seeds) {
    int32_t u = all_seeds ? tid : seeds[tid];
    deg_out[tid] = (int64_t)(offsets[u + 1] - offsets[u]);
  }
}

__global__ void fill_wedges_kernel(const int32_t* __restrict__ offsets,
                                  const int32_t* __restrict__ indices,
                                  const int32_t* __restrict__ seeds,
                                  int32_t num_seeds,
                                  bool all_seeds,
                                  const int64_t* __restrict__ seed_nb_offsets,
                                  int32_t* __restrict__ wedge_seed_idx,
                                  int32_t* __restrict__ wedge_w,
                                  int64_t* __restrict__ wedge_sizes)
{
  int seed_idx = blockIdx.x;
  if (seed_idx >= num_seeds) return;

  int32_t u = all_seeds ? seed_idx : seeds[seed_idx];
  int32_t start = offsets[u];
  int32_t end = offsets[u + 1];
  int32_t deg_u = end - start;
  int64_t out_base = seed_nb_offsets[seed_idx];

  for (int i = threadIdx.x; i < deg_u; i += blockDim.x) {
    int32_t w = indices[start + i];
    int64_t o = out_base + i;
    wedge_seed_idx[o] = seed_idx;
    wedge_w[o] = w;
    wedge_sizes[o] = (int64_t)(offsets[w + 1] - offsets[w]);
  }
}

__device__ __forceinline__ int32_t find_wedge_start(const int64_t* __restrict__ exp_offsets,
                                                    int32_t num_wedges,
                                                    int64_t out_idx)
{
  int32_t lo = 0;
  int32_t hi = num_wedges;
  while (lo < hi) {
    int32_t mid = (lo + hi + 1) >> 1;
    int64_t v = exp_offsets[mid];
    if (v <= out_idx) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

template <typename KeyT, int THREADS, int ITEMS>
__global__ void expand_keys_partitioned_kernel(const int32_t* __restrict__ offsets,
                                               const int32_t* __restrict__ indices,
                                               const int32_t* __restrict__ wedge_seed_idx,
                                               const int32_t* __restrict__ wedge_w,
                                               const int64_t* __restrict__ exp_offsets,
                                               int32_t num_wedges,
                                               int64_t total_expansion,
                                               int shift_v,
                                               KeyT* __restrict__ out_keys)
{
  constexpr int OUTPUTS_PER_BLOCK = THREADS * ITEMS;
  int64_t block_out_start = (int64_t)blockIdx.x * OUTPUTS_PER_BLOCK;
  if (block_out_start >= total_expansion) return;

  int64_t block_out_end = block_out_start + OUTPUTS_PER_BLOCK;
  if (block_out_end > total_expansion) block_out_end = total_expansion;

  __shared__ int32_t s_wedge_start;
  __shared__ int32_t s_wedge_end;
  if (threadIdx.x == 0) {
    int32_t ws = find_wedge_start(exp_offsets, num_wedges, block_out_start);
    int32_t we = find_wedge_start(exp_offsets, num_wedges, block_out_end - 1) + 1;
    if (we > num_wedges) we = num_wedges;
    s_wedge_start = ws;
    s_wedge_end = we;
  }
  __syncthreads();

  int32_t wedge_start = s_wedge_start;
  int32_t wedge_end = s_wedge_end;
  int32_t num_block_wedges = wedge_end - wedge_start;

  if (num_block_wedges > OUTPUTS_PER_BLOCK) {
    int t = threadIdx.x;
#pragma unroll
    for (int it = 0; it < ITEMS; ++it) {
      int64_t out_idx = block_out_start + (int64_t)t + (int64_t)it * THREADS;
      if (out_idx >= block_out_end) continue;
      int32_t wi = find_wedge_start(exp_offsets, num_wedges, out_idx);
      int64_t base = exp_offsets[wi];
      int32_t u_rank = wedge_seed_idx[wi];
      int32_t w = wedge_w[wi];
      int32_t w_start = offsets[w];
      int32_t pos = (int32_t)(out_idx - base);
      int32_t v = indices[w_start + pos];
      out_keys[out_idx] = (KeyT)(((uint64_t)(uint32_t)u_rank << shift_v) | (uint64_t)(uint32_t)v);
    }
    return;
  }

  extern __shared__ uint8_t smem[];
  int64_t* s_off = reinterpret_cast<int64_t*>(smem);
  int32_t* s_u = reinterpret_cast<int32_t*>(s_off + (OUTPUTS_PER_BLOCK + 1));
  int32_t* s_w = s_u + OUTPUTS_PER_BLOCK;

  for (int i = threadIdx.x; i <= num_block_wedges; i += THREADS) {
    s_off[i] = exp_offsets[wedge_start + i];
  }
  for (int i = threadIdx.x; i < num_block_wedges; i += THREADS) {
    int32_t wi = wedge_start + i;
    s_u[i] = wedge_seed_idx[wi];
    s_w[i] = wedge_w[wi];
  }
  __syncthreads();

  int t = threadIdx.x;
#pragma unroll
  for (int it = 0; it < ITEMS; ++it) {
    int64_t out_idx = block_out_start + (int64_t)t + (int64_t)it * THREADS;
    if (out_idx >= block_out_end) continue;

    int32_t lo = 0;
    int32_t hi = num_block_wedges - 1;
    while (lo < hi) {
      int32_t mid = (lo + hi + 1) >> 1;
      if (s_off[mid] <= out_idx) lo = mid;
      else hi = mid - 1;
    }

    int32_t l = lo;
    int64_t base = s_off[l];
    int32_t u_rank = s_u[l];
    int32_t w = s_w[l];

    int32_t w_start = offsets[w];
    int32_t pos = (int32_t)(out_idx - base);
    int32_t v = indices[w_start + pos];

    out_keys[out_idx] = (KeyT)(((uint64_t)(uint32_t)u_rank << shift_v) | (uint64_t)(uint32_t)v);
  }
}

__global__ void iota_i32_kernel(int32_t* out, int32_t n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) out[tid] = tid;
}

template <typename KeyT>
__global__ void remove_self_rle_inplace(KeyT* __restrict__ keys,
                                        int32_t* __restrict__ counts,
                                        int32_t* __restrict__ n_inout,
                                        const int32_t* __restrict__ seeds,
                                        bool all_seeds,
                                        int32_t num_seeds,
                                        int shift_v,
                                        uint64_t mask_v)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int32_t n = *n_inout;

  for (int32_t u_rank = num_seeds - 1; u_rank >= 0; --u_rank) {
    int32_t u = all_seeds ? u_rank : seeds[u_rank];
    uint64_t self_u64 = ((uint64_t)(uint32_t)u_rank << shift_v) | (uint64_t)(uint32_t)u;
    KeyT self_key = (KeyT)self_u64;

    int pos = lower_bound_dev_t<KeyT>(keys, n, self_key);
    if (pos < n && keys[pos] == self_key) {
      int last = n - 1;
      keys[pos] = keys[last];
      counts[pos] = counts[last];
      n = last;
    }
  }
  *n_inout = n;
}

template <typename KeyT>
__global__ void remove_self_unique_inplace(KeyT* __restrict__ keys,
                                           int32_t* __restrict__ n_inout,
                                           const int32_t* __restrict__ seeds,
                                           bool all_seeds,
                                           int32_t num_seeds,
                                           int shift_v)
{
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int32_t n = *n_inout;

  for (int32_t u_rank = num_seeds - 1; u_rank >= 0; --u_rank) {
    int32_t u = all_seeds ? u_rank : seeds[u_rank];
    uint64_t self_u64 = ((uint64_t)(uint32_t)u_rank << shift_v) | (uint64_t)(uint32_t)u;
    KeyT self_key = (KeyT)self_u64;

    int pos = lower_bound_dev_t<KeyT>(keys, n, self_key);
    if (pos < n && keys[pos] == self_key) {
      int last = n - 1;
      keys[pos] = keys[last];
      n = last;
    }
  }
  *n_inout = n;
}

template <typename KeyT>
__global__ void scores_from_counts_kernel(const int32_t* __restrict__ offsets,
                                          const int32_t* __restrict__ seeds,
                                          bool all_seeds,
                                          const KeyT* __restrict__ keys,
                                          const int32_t* __restrict__ counts,
                                          int32_t n,
                                          int shift_v,
                                          uint64_t mask_v,
                                          float* __restrict__ out_scores)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  uint64_t key = (uint64_t)keys[tid];
  int32_t u_rank = (int32_t)(key >> shift_v);
  int32_t u = all_seeds ? u_rank : seeds[u_rank];
  int32_t v = (int32_t)(key & mask_v);

  int du = offsets[u + 1] - offsets[u];
  int dv = offsets[v + 1] - offsets[v];
  int md = du < dv ? du : dv;
  int32_t c = counts[tid];
  out_scores[tid] = (md > 0 && c > 0) ? ((float)c / (float)md) : 0.0f;
}

template <typename KeyT>
__global__ void fused_from_counts_kernel(const int32_t* __restrict__ offsets,
                                         const int32_t* __restrict__ seeds,
                                         bool all_seeds,
                                         const KeyT* __restrict__ keys,
                                         const int32_t* __restrict__ counts,
                                         int32_t n,
                                         int shift_v,
                                         uint64_t mask_v,
                                         int32_t* __restrict__ out_first,
                                         int32_t* __restrict__ out_second,
                                         float* __restrict__ out_scores)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;

  uint64_t key = (uint64_t)keys[tid];
  int32_t u_rank = (int32_t)(key >> shift_v);
  int32_t u = all_seeds ? u_rank : seeds[u_rank];
  int32_t v = (int32_t)(key & mask_v);

  out_first[tid] = u;
  out_second[tid] = v;

  int du = offsets[u + 1] - offsets[u];
  int dv = offsets[v + 1] - offsets[v];
  int md = du < dv ? du : dv;
  int32_t c = counts[tid];
  out_scores[tid] = (md > 0 && c > 0) ? ((float)c / (float)md) : 0.0f;
}

template <typename KeyT>
__global__ void overlap_scores_warp_kernel(const int32_t* __restrict__ offsets,
                                           const int32_t* __restrict__ indices,
                                           const int32_t* __restrict__ seeds,
                                           bool all_seeds,
                                           const KeyT* __restrict__ keys,
                                           int32_t n,
                                           int shift_v,
                                           uint64_t mask_v,
                                           bool is_multigraph,
                                           float* __restrict__ out_scores)
{
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (warp >= n) return;

  uint64_t key = (uint64_t)keys[warp];
  int32_t u_rank = (int32_t)(key >> shift_v);
  int32_t u = all_seeds ? u_rank : seeds[u_rank];
  int32_t v = (int32_t)(key & mask_v);

  int32_t us = offsets[u];
  int32_t ue = offsets[u + 1];
  int32_t vs = offsets[v];
  int32_t ve = offsets[v + 1];

  int du = ue - us;
  int dv = ve - vs;
  int md = du < dv ? du : dv;
  if (md <= 0) {
    if (lane == 0) out_scores[warp] = 0.0f;
    return;
  }

  int c = warp_intersect_count(indices + us, du, indices + vs, dv, is_multigraph);
  if (lane == 0) out_scores[warp] = (float)c / (float)md;
}

template <typename KeyT>
__global__ void overlap_fused_warp_kernel(const int32_t* __restrict__ offsets,
                                          const int32_t* __restrict__ indices,
                                          const int32_t* __restrict__ seeds,
                                          bool all_seeds,
                                          const KeyT* __restrict__ keys,
                                          int32_t n,
                                          int shift_v,
                                          uint64_t mask_v,
                                          bool is_multigraph,
                                          int32_t* __restrict__ out_first,
                                          int32_t* __restrict__ out_second,
                                          float* __restrict__ out_scores)
{
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (warp >= n) return;

  uint64_t key = (uint64_t)keys[warp];
  int32_t u_rank = (int32_t)(key >> shift_v);
  int32_t u = all_seeds ? u_rank : seeds[u_rank];
  int32_t v = (int32_t)(key & mask_v);

  if (lane == 0) {
    out_first[warp] = u;
    out_second[warp] = v;
  }

  int32_t us = offsets[u];
  int32_t ue = offsets[u + 1];
  int32_t vs = offsets[v];
  int32_t ve = offsets[v + 1];

  int du = ue - us;
  int dv = ve - vs;
  int md = du < dv ? du : dv;
  if (md <= 0) {
    if (lane == 0) out_scores[warp] = 0.0f;
    return;
  }

  int c = warp_intersect_count(indices + us, du, indices + vs, dv, is_multigraph);
  if (lane == 0) out_scores[warp] = (float)c / (float)md;
}

template <typename KeyT>
__global__ void gather_topk_kernel(const KeyT* __restrict__ keys,
                                   const int32_t* __restrict__ seeds,
                                   bool all_seeds,
                                   const float* __restrict__ sorted_scores,
                                   const int32_t* __restrict__ sorted_idx,
                                   int32_t k,
                                   int shift_v,
                                   uint64_t mask_v,
                                   int32_t* __restrict__ out_first,
                                   int32_t* __restrict__ out_second,
                                   float* __restrict__ out_scores)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= k) return;

  int32_t src = sorted_idx[tid];
  uint64_t key = (uint64_t)keys[src];
  int32_t u_rank = (int32_t)(key >> shift_v);
  int32_t u = all_seeds ? u_rank : seeds[u_rank];
  int32_t v = (int32_t)(key & mask_v);

  out_first[tid] = u;
  out_second[tid] = v;
  out_scores[tid] = sorted_scores[tid];
}





size_t get_exclusive_sum_temp_bytes(int num_items)
{
  size_t temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, (const int64_t*)nullptr, (int64_t*)nullptr, num_items);
  return temp_bytes;
}

void launch_exclusive_sum(const int64_t* in, int64_t* out, int num_items, void* temp, size_t temp_bytes)
{
  cub::DeviceScan::ExclusiveSum(temp, temp_bytes, in, out, num_items);
}

size_t get_sort_keys_temp_bytes_u32(int num_items)
{
  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, (const uint32_t*)nullptr, (uint32_t*)nullptr, num_items);
  return temp_bytes;
}

size_t get_sort_keys_temp_bytes_u64(int num_items)
{
  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, (const uint64_t*)nullptr, (uint64_t*)nullptr, num_items);
  return temp_bytes;
}

void launch_sort_keys_u32(const uint32_t* in, uint32_t* out, int num_items, int end_bit, void* temp, size_t temp_bytes)
{
  cub::DeviceRadixSort::SortKeys(temp, temp_bytes, in, out, num_items, 0, end_bit);
}

void launch_sort_keys_u64(const uint64_t* in, uint64_t* out, int num_items, int end_bit, void* temp, size_t temp_bytes)
{
  cub::DeviceRadixSort::SortKeys(temp, temp_bytes, in, out, num_items, 0, end_bit);
}

size_t get_unique_temp_bytes_u32(int num_items)
{
  size_t temp_bytes = 0;
  cub::DeviceSelect::Unique(nullptr, temp_bytes,
                            (const uint32_t*)nullptr,
                            (uint32_t*)nullptr,
                            (int*)nullptr,
                            num_items);
  return temp_bytes;
}

size_t get_unique_temp_bytes_u64(int num_items)
{
  size_t temp_bytes = 0;
  cub::DeviceSelect::Unique(nullptr, temp_bytes,
                            (const uint64_t*)nullptr,
                            (uint64_t*)nullptr,
                            (int*)nullptr,
                            num_items);
  return temp_bytes;
}

void launch_unique_u32(const uint32_t* in, uint32_t* out, int* out_count, int num_items, void* temp, size_t temp_bytes)
{
  cub::DeviceSelect::Unique(temp, temp_bytes, in, out, out_count, num_items);
}

void launch_unique_u64(const uint64_t* in, uint64_t* out, int* out_count, int num_items, void* temp, size_t temp_bytes)
{
  cub::DeviceSelect::Unique(temp, temp_bytes, in, out, out_count, num_items);
}

size_t get_rle_temp_bytes_u32(int num_items)
{
  size_t temp_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(nullptr, temp_bytes,
                                     (const uint32_t*)nullptr,
                                     (uint32_t*)nullptr,
                                     (int*)nullptr,
                                     (int*)nullptr,
                                     num_items);
  return temp_bytes;
}

size_t get_rle_temp_bytes_u64(int num_items)
{
  size_t temp_bytes = 0;
  cub::DeviceRunLengthEncode::Encode(nullptr, temp_bytes,
                                     (const uint64_t*)nullptr,
                                     (uint64_t*)nullptr,
                                     (int*)nullptr,
                                     (int*)nullptr,
                                     num_items);
  return temp_bytes;
}

void launch_rle_u32(const uint32_t* in, uint32_t* out_keys, int* out_counts, int* out_runs, int num_items, void* temp, size_t temp_bytes)
{
  cub::DeviceRunLengthEncode::Encode(temp, temp_bytes, in, out_keys, out_counts, out_runs, num_items);
}

void launch_rle_u64(const uint64_t* in, uint64_t* out_keys, int* out_counts, int* out_runs, int num_items, void* temp, size_t temp_bytes)
{
  cub::DeviceRunLengthEncode::Encode(temp, temp_bytes, in, out_keys, out_counts, out_runs, num_items);
}

size_t get_sort_pairs_temp_bytes(int num_items)
{
  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_bytes,
                                            (const float*)nullptr,
                                            (float*)nullptr,
                                            (const int32_t*)nullptr,
                                            (int32_t*)nullptr,
                                            num_items);
  return temp_bytes;
}

void launch_sort_pairs_desc(const float* in_keys, float* out_keys,
                            const int32_t* in_vals, int32_t* out_vals,
                            int num_items, void* temp, size_t temp_bytes)
{
  cub::DeviceRadixSort::SortPairsDescending(temp, temp_bytes,
                                           in_keys, out_keys,
                                           in_vals, out_vals,
                                           num_items);
}





void launch_gather_seed_deg64(const int32_t* offsets, const int32_t* seeds, int32_t num_seeds, bool all_seeds, int64_t* deg_out)
{
  int block = 256;
  int grid = (num_seeds + block - 1) / block;
  gather_seed_deg64_kernel<<<grid, block>>>(offsets, seeds, num_seeds, all_seeds, deg_out);
}

void launch_fill_wedges(const int32_t* offsets, const int32_t* indices,
                        const int32_t* seeds, int32_t num_seeds, bool all_seeds,
                        const int64_t* seed_nb_offsets,
                        int32_t* wedge_seed_idx, int32_t* wedge_w, int64_t* wedge_sizes)
{
  fill_wedges_kernel<<<num_seeds, 256>>>(offsets, indices, seeds, num_seeds, all_seeds,
                                        seed_nb_offsets, wedge_seed_idx, wedge_w, wedge_sizes);
}

void launch_expand_keys_u32(const int32_t* offsets, const int32_t* indices,
                            const int32_t* wedge_seed_idx, const int32_t* wedge_w,
                            const int64_t* exp_offsets, int32_t num_wedges,
                            int64_t total_expansion, int shift_v,
                            uint32_t* out_keys)
{
  constexpr int THREADS = 256;
  constexpr int ITEMS = 8;
  constexpr int OUTPUTS_PER_BLOCK = THREADS * ITEMS;
  int grid = (int)((total_expansion + OUTPUTS_PER_BLOCK - 1) / OUTPUTS_PER_BLOCK);
  size_t smem = (OUTPUTS_PER_BLOCK + 1) * sizeof(int64_t) + OUTPUTS_PER_BLOCK * sizeof(int32_t) * 2;
  expand_keys_partitioned_kernel<uint32_t, THREADS, ITEMS><<<grid, THREADS, smem>>>(
      offsets, indices, wedge_seed_idx, wedge_w, exp_offsets, num_wedges, total_expansion, shift_v, out_keys);
}

void launch_expand_keys_u64(const int32_t* offsets, const int32_t* indices,
                            const int32_t* wedge_seed_idx, const int32_t* wedge_w,
                            const int64_t* exp_offsets, int32_t num_wedges,
                            int64_t total_expansion, int shift_v,
                            uint64_t* out_keys)
{
  constexpr int THREADS = 256;
  constexpr int ITEMS = 8;
  constexpr int OUTPUTS_PER_BLOCK = THREADS * ITEMS;
  int grid = (int)((total_expansion + OUTPUTS_PER_BLOCK - 1) / OUTPUTS_PER_BLOCK);
  size_t smem = (OUTPUTS_PER_BLOCK + 1) * sizeof(int64_t) + OUTPUTS_PER_BLOCK * sizeof(int32_t) * 2;
  expand_keys_partitioned_kernel<uint64_t, THREADS, ITEMS><<<grid, THREADS, smem>>>(
      offsets, indices, wedge_seed_idx, wedge_w, exp_offsets, num_wedges, total_expansion, shift_v, out_keys);
}

void launch_iota_i32(int32_t* out, int32_t n)
{
  int block = 256;
  int grid = (n + block - 1) / block;
  iota_i32_kernel<<<grid, block>>>(out, n);
}

void launch_remove_self_rle_u32(uint32_t* keys, int32_t* counts, int32_t* n_inout,
                                const int32_t* seeds, bool all_seeds, int32_t num_seeds,
                                int shift_v, uint64_t mask_v)
{
  remove_self_rle_inplace<uint32_t><<<1, 1>>>(keys, counts, n_inout, seeds, all_seeds, num_seeds, shift_v, mask_v);
}

void launch_remove_self_rle_u64(uint64_t* keys, int32_t* counts, int32_t* n_inout,
                                const int32_t* seeds, bool all_seeds, int32_t num_seeds,
                                int shift_v, uint64_t mask_v)
{
  remove_self_rle_inplace<uint64_t><<<1, 1>>>(keys, counts, n_inout, seeds, all_seeds, num_seeds, shift_v, mask_v);
}

void launch_remove_self_unique_u32(uint32_t* keys, int32_t* n_inout,
                                   const int32_t* seeds, bool all_seeds, int32_t num_seeds,
                                   int shift_v)
{
  remove_self_unique_inplace<uint32_t><<<1, 1>>>(keys, n_inout, seeds, all_seeds, num_seeds, shift_v);
}

void launch_remove_self_unique_u64(uint64_t* keys, int32_t* n_inout,
                                   const int32_t* seeds, bool all_seeds, int32_t num_seeds,
                                   int shift_v)
{
  remove_self_unique_inplace<uint64_t><<<1, 1>>>(keys, n_inout, seeds, all_seeds, num_seeds, shift_v);
}

void launch_scores_from_counts_u32(const int32_t* offsets, const int32_t* seeds, bool all_seeds,
                                   const uint32_t* keys, const int32_t* counts, int32_t n,
                                   int shift_v, uint64_t mask_v, float* out_scores)
{
  int block = 256;
  int grid = (n + block - 1) / block;
  scores_from_counts_kernel<uint32_t><<<grid, block>>>(offsets, seeds, all_seeds, keys, counts, n, shift_v, mask_v, out_scores);
}

void launch_scores_from_counts_u64(const int32_t* offsets, const int32_t* seeds, bool all_seeds,
                                   const uint64_t* keys, const int32_t* counts, int32_t n,
                                   int shift_v, uint64_t mask_v, float* out_scores)
{
  int block = 256;
  int grid = (n + block - 1) / block;
  scores_from_counts_kernel<uint64_t><<<grid, block>>>(offsets, seeds, all_seeds, keys, counts, n, shift_v, mask_v, out_scores);
}

void launch_fused_from_counts_u32(const int32_t* offsets, const int32_t* seeds, bool all_seeds,
                                  const uint32_t* keys, const int32_t* counts, int32_t n,
                                  int shift_v, uint64_t mask_v,
                                  int32_t* out_first, int32_t* out_second, float* out_scores)
{
  int block = 256;
  int grid = (n + block - 1) / block;
  fused_from_counts_kernel<uint32_t><<<grid, block>>>(offsets, seeds, all_seeds, keys, counts, n, shift_v, mask_v,
                                                      out_first, out_second, out_scores);
}

void launch_fused_from_counts_u64(const int32_t* offsets, const int32_t* seeds, bool all_seeds,
                                  const uint64_t* keys, const int32_t* counts, int32_t n,
                                  int shift_v, uint64_t mask_v,
                                  int32_t* out_first, int32_t* out_second, float* out_scores)
{
  int block = 256;
  int grid = (n + block - 1) / block;
  fused_from_counts_kernel<uint64_t><<<grid, block>>>(offsets, seeds, all_seeds, keys, counts, n, shift_v, mask_v,
                                                      out_first, out_second, out_scores);
}

void launch_overlap_scores_u32(const int32_t* offsets, const int32_t* indices, const int32_t* seeds, bool all_seeds,
                               const uint32_t* keys, int32_t n, int shift_v, uint64_t mask_v, bool is_multigraph,
                               float* out_scores)
{
  int warps_per_block = 8;
  int block = warps_per_block * 32;
  int grid = (n + warps_per_block - 1) / warps_per_block;
  overlap_scores_warp_kernel<uint32_t><<<grid, block>>>(offsets, indices, seeds, all_seeds, keys, n, shift_v, mask_v, is_multigraph, out_scores);
}

void launch_overlap_scores_u64(const int32_t* offsets, const int32_t* indices, const int32_t* seeds, bool all_seeds,
                               const uint64_t* keys, int32_t n, int shift_v, uint64_t mask_v, bool is_multigraph,
                               float* out_scores)
{
  int warps_per_block = 8;
  int block = warps_per_block * 32;
  int grid = (n + warps_per_block - 1) / warps_per_block;
  overlap_scores_warp_kernel<uint64_t><<<grid, block>>>(offsets, indices, seeds, all_seeds, keys, n, shift_v, mask_v, is_multigraph, out_scores);
}

void launch_overlap_fused_u32(const int32_t* offsets, const int32_t* indices, const int32_t* seeds, bool all_seeds,
                              const uint32_t* keys, int32_t n, int shift_v, uint64_t mask_v, bool is_multigraph,
                              int32_t* out_first, int32_t* out_second, float* out_scores)
{
  int warps_per_block = 8;
  int block = warps_per_block * 32;
  int grid = (n + warps_per_block - 1) / warps_per_block;
  overlap_fused_warp_kernel<uint32_t><<<grid, block>>>(offsets, indices, seeds, all_seeds, keys, n, shift_v, mask_v, is_multigraph,
                                                       out_first, out_second, out_scores);
}

void launch_overlap_fused_u64(const int32_t* offsets, const int32_t* indices, const int32_t* seeds, bool all_seeds,
                              const uint64_t* keys, int32_t n, int shift_v, uint64_t mask_v, bool is_multigraph,
                              int32_t* out_first, int32_t* out_second, float* out_scores)
{
  int warps_per_block = 8;
  int block = warps_per_block * 32;
  int grid = (n + warps_per_block - 1) / warps_per_block;
  overlap_fused_warp_kernel<uint64_t><<<grid, block>>>(offsets, indices, seeds, all_seeds, keys, n, shift_v, mask_v, is_multigraph,
                                                       out_first, out_second, out_scores);
}

void launch_gather_topk_u32(const uint32_t* keys, const int32_t* seeds, bool all_seeds,
                            const float* sorted_scores, const int32_t* sorted_idx,
                            int32_t k, int shift_v, uint64_t mask_v,
                            int32_t* out_first, int32_t* out_second, float* out_scores)
{
  int block = 256;
  int grid = (k + block - 1) / block;
  gather_topk_kernel<uint32_t><<<grid, block>>>(keys, seeds, all_seeds, sorted_scores, sorted_idx, k, shift_v, mask_v,
                                                out_first, out_second, out_scores);
}

void launch_gather_topk_u64(const uint64_t* keys, const int32_t* seeds, bool all_seeds,
                            const float* sorted_scores, const int32_t* sorted_idx,
                            int32_t k, int shift_v, uint64_t mask_v,
                            int32_t* out_first, int32_t* out_second, float* out_scores)
{
  int block = 256;
  int grid = (k + block - 1) / block;
  gather_topk_kernel<uint64_t><<<grid, block>>>(keys, seeds, all_seeds, sorted_scores, sorted_idx, k, shift_v, mask_v,
                                                out_first, out_second, out_scores);
}





inline int bits_needed_u32(uint32_t x)
{
  if (x <= 1) return 1;
  return 32 - __builtin_clz(x - 1);
}

}  





similarity_result_float_t overlap_all_pairs_similarity(const graph32_t& graph,
                                                       const int32_t* vertices,
                                                       std::size_t num_vertices,
                                                       std::optional<std::size_t> topk) {
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);
  (void)cache;

  const int32_t* d_offsets = graph.offsets;
  const int32_t* d_indices = graph.indices;
  int32_t nv = graph.number_of_vertices;
  bool is_multigraph = graph.is_multigraph;

  
  const int32_t* d_seeds = nullptr;
  int32_t num_seeds = 0;
  bool all_seeds = false;

  if (num_vertices > 0 && vertices != nullptr) {
    d_seeds = vertices;
    num_seeds = (int32_t)num_vertices;
    all_seeds = false;
  } else {
    d_seeds = nullptr;
    num_seeds = nv;
    all_seeds = true;
  }

  if (num_seeds <= 0) {
    return {nullptr, nullptr, nullptr, 0};
  }

  int64_t topk_raw = topk.has_value() ? (int64_t)topk.value() : -1;

  int shift_v = bits_needed_u32((uint32_t)nv);
  uint64_t mask_v = (shift_v == 64) ? 0xFFFFFFFFFFFFFFFFull : ((1ull << shift_v) - 1ull);
  int bits_u = bits_needed_u32((uint32_t)num_seeds);
  int key_bits = shift_v + bits_u;

  bool use_u32 = (key_bits <= 32);

  int end_bit = key_bits;
  if (end_bit > 64) end_bit = 64;

  
  int64_t* seed_deg = nullptr;
  int64_t* seed_nb_offsets_d = nullptr;
  cudaMalloc(&seed_deg, ((size_t)num_seeds + 1) * sizeof(int64_t));
  cudaMalloc(&seed_nb_offsets_d, ((size_t)num_seeds + 1) * sizeof(int64_t));
  cudaMemset(seed_deg + num_seeds, 0, sizeof(int64_t));

  launch_gather_seed_deg64(d_offsets, d_seeds, num_seeds, all_seeds, seed_deg);

  size_t scan_tmp_bytes = get_exclusive_sum_temp_bytes(num_seeds + 1);
  void* scan_tmp = nullptr;
  cudaMalloc(&scan_tmp, scan_tmp_bytes);
  launch_exclusive_sum(seed_deg, seed_nb_offsets_d, num_seeds + 1, scan_tmp, scan_tmp_bytes);
  cudaFree(scan_tmp);
  cudaFree(seed_deg);

  int64_t total_wedges = 0;
  cudaMemcpy(&total_wedges, seed_nb_offsets_d + num_seeds, sizeof(int64_t), cudaMemcpyDeviceToHost);
  if (total_wedges <= 0) {
    cudaFree(seed_nb_offsets_d);
    return {nullptr, nullptr, nullptr, 0};
  }

  int32_t total_wedges_i32 = (int32_t)total_wedges;

  
  int32_t* wedge_seed_idx = nullptr;
  int32_t* wedge_w = nullptr;
  int64_t* wedge_sizes = nullptr;
  cudaMalloc(&wedge_seed_idx, (size_t)total_wedges_i32 * sizeof(int32_t));
  cudaMalloc(&wedge_w, (size_t)total_wedges_i32 * sizeof(int32_t));
  cudaMalloc(&wedge_sizes, ((size_t)total_wedges_i32 + 1) * sizeof(int64_t));
  cudaMemset(wedge_sizes + total_wedges_i32, 0, sizeof(int64_t));

  launch_fill_wedges(d_offsets, d_indices, d_seeds, num_seeds, all_seeds,
                     seed_nb_offsets_d, wedge_seed_idx, wedge_w, wedge_sizes);
  cudaFree(seed_nb_offsets_d);

  
  int64_t* exp_offsets = nullptr;
  cudaMalloc(&exp_offsets, ((size_t)total_wedges_i32 + 1) * sizeof(int64_t));
  size_t scan2_tmp_bytes = get_exclusive_sum_temp_bytes(total_wedges_i32 + 1);
  void* scan2_tmp = nullptr;
  cudaMalloc(&scan2_tmp, scan2_tmp_bytes);
  launch_exclusive_sum(wedge_sizes, exp_offsets, total_wedges_i32 + 1, scan2_tmp, scan2_tmp_bytes);
  cudaFree(scan2_tmp);
  cudaFree(wedge_sizes);

  int64_t total_expansion = 0;
  cudaMemcpy(&total_expansion, exp_offsets + total_wedges_i32, sizeof(int64_t), cudaMemcpyDeviceToHost);
  if (total_expansion <= 0) {
    cudaFree(wedge_seed_idx);
    cudaFree(wedge_w);
    cudaFree(exp_offsets);
    return {nullptr, nullptr, nullptr, 0};
  }

  int32_t total_exp_i32 = (int32_t)total_expansion;

  
  void* raw_keys = nullptr;
  if (use_u32) {
    cudaMalloc(&raw_keys, (size_t)total_exp_i32 * sizeof(uint32_t));
    launch_expand_keys_u32(d_offsets, d_indices, wedge_seed_idx, wedge_w,
                           exp_offsets, total_wedges_i32, total_expansion, shift_v,
                           (uint32_t*)raw_keys);
  } else {
    cudaMalloc(&raw_keys, (size_t)total_exp_i32 * sizeof(uint64_t));
    launch_expand_keys_u64(d_offsets, d_indices, wedge_seed_idx, wedge_w,
                           exp_offsets, total_wedges_i32, total_expansion, shift_v,
                           (uint64_t*)raw_keys);
  }
  cudaFree(wedge_seed_idx);
  cudaFree(wedge_w);
  cudaFree(exp_offsets);

  
  void* sorted_keys = nullptr;
  if (use_u32) {
    cudaMalloc(&sorted_keys, (size_t)total_exp_i32 * sizeof(uint32_t));
    size_t sort_tmp_bytes = get_sort_keys_temp_bytes_u32(total_exp_i32);
    void* sort_tmp = nullptr;
    cudaMalloc(&sort_tmp, sort_tmp_bytes);
    launch_sort_keys_u32((const uint32_t*)raw_keys, (uint32_t*)sorted_keys,
                         total_exp_i32, end_bit, sort_tmp, sort_tmp_bytes);
    cudaFree(sort_tmp);
  } else {
    cudaMalloc(&sorted_keys, (size_t)total_exp_i32 * sizeof(uint64_t));
    size_t sort_tmp_bytes = get_sort_keys_temp_bytes_u64(total_exp_i32);
    void* sort_tmp = nullptr;
    cudaMalloc(&sort_tmp, sort_tmp_bytes);
    launch_sort_keys_u64((const uint64_t*)raw_keys, (uint64_t*)sorted_keys,
                         total_exp_i32, end_bit, sort_tmp, sort_tmp_bytes);
    cudaFree(sort_tmp);
  }
  cudaFree(raw_keys);

  if (!is_multigraph) {
    
    void* rle_keys = nullptr;
    int32_t* rle_counts = nullptr;
    int32_t* rle_runs_d = nullptr;

    if (use_u32) {
      cudaMalloc(&rle_keys, (size_t)total_exp_i32 * sizeof(uint32_t));
    } else {
      cudaMalloc(&rle_keys, (size_t)total_exp_i32 * sizeof(uint64_t));
    }
    cudaMalloc(&rle_counts, (size_t)total_exp_i32 * sizeof(int32_t));
    cudaMalloc(&rle_runs_d, sizeof(int32_t));

    if (use_u32) {
      size_t rle_tmp_bytes = get_rle_temp_bytes_u32(total_exp_i32);
      void* rle_tmp = nullptr;
      cudaMalloc(&rle_tmp, rle_tmp_bytes);
      launch_rle_u32((const uint32_t*)sorted_keys, (uint32_t*)rle_keys,
                     (int*)rle_counts, (int*)rle_runs_d, total_exp_i32,
                     rle_tmp, rle_tmp_bytes);
      cudaFree(rle_tmp);
      launch_remove_self_rle_u32((uint32_t*)rle_keys, rle_counts, rle_runs_d,
                                 d_seeds, all_seeds, num_seeds, shift_v, mask_v);
    } else {
      size_t rle_tmp_bytes = get_rle_temp_bytes_u64(total_exp_i32);
      void* rle_tmp = nullptr;
      cudaMalloc(&rle_tmp, rle_tmp_bytes);
      launch_rle_u64((const uint64_t*)sorted_keys, (uint64_t*)rle_keys,
                     (int*)rle_counts, (int*)rle_runs_d, total_exp_i32,
                     rle_tmp, rle_tmp_bytes);
      cudaFree(rle_tmp);
      launch_remove_self_rle_u64((uint64_t*)rle_keys, rle_counts, rle_runs_d,
                                 d_seeds, all_seeds, num_seeds, shift_v, mask_v);
    }
    cudaFree(sorted_keys);

    int32_t num_pairs = 0;
    cudaMemcpy(&num_pairs, rle_runs_d, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(rle_runs_d);

    if (num_pairs <= 0) {
      cudaFree(rle_keys);
      cudaFree(rle_counts);
      return {nullptr, nullptr, nullptr, 0};
    }

    bool do_topk = (topk_raw >= 0 && topk_raw < (int64_t)num_pairs);
    std::size_t out_count = do_topk ? (std::size_t)topk_raw : (std::size_t)num_pairs;

    if (do_topk) {
      float* scores = nullptr;
      cudaMalloc(&scores, (size_t)num_pairs * sizeof(float));
      if (use_u32) {
        launch_scores_from_counts_u32(d_offsets, d_seeds, all_seeds,
                                      (const uint32_t*)rle_keys, rle_counts,
                                      num_pairs, shift_v, mask_v, scores);
      } else {
        launch_scores_from_counts_u64(d_offsets, d_seeds, all_seeds,
                                      (const uint64_t*)rle_keys, rle_counts,
                                      num_pairs, shift_v, mask_v, scores);
      }
      cudaFree(rle_counts);

      int32_t* idx_in = nullptr;
      cudaMalloc(&idx_in, (size_t)num_pairs * sizeof(int32_t));
      launch_iota_i32(idx_in, num_pairs);

      float* scores_sorted = nullptr;
      int32_t* idx_sorted = nullptr;
      cudaMalloc(&scores_sorted, (size_t)num_pairs * sizeof(float));
      cudaMalloc(&idx_sorted, (size_t)num_pairs * sizeof(int32_t));
      size_t sp_tmp_bytes = get_sort_pairs_temp_bytes(num_pairs);
      void* sp_tmp = nullptr;
      cudaMalloc(&sp_tmp, sp_tmp_bytes);
      launch_sort_pairs_desc(scores, scores_sorted, idx_in, idx_sorted,
                             num_pairs, sp_tmp, sp_tmp_bytes);
      cudaFree(sp_tmp);
      cudaFree(scores);
      cudaFree(idx_in);

      int32_t k = (int32_t)topk_raw;
      int32_t* out_first = nullptr;
      int32_t* out_second = nullptr;
      float* out_scores = nullptr;
      cudaMalloc(&out_first, (size_t)k * sizeof(int32_t));
      cudaMalloc(&out_second, (size_t)k * sizeof(int32_t));
      cudaMalloc(&out_scores, (size_t)k * sizeof(float));

      if (use_u32) {
        launch_gather_topk_u32((const uint32_t*)rle_keys, d_seeds, all_seeds,
                               scores_sorted, idx_sorted, k, shift_v, mask_v,
                               out_first, out_second, out_scores);
      } else {
        launch_gather_topk_u64((const uint64_t*)rle_keys, d_seeds, all_seeds,
                               scores_sorted, idx_sorted, k, shift_v, mask_v,
                               out_first, out_second, out_scores);
      }
      cudaFree(scores_sorted);
      cudaFree(idx_sorted);
      cudaFree(rle_keys);

      return {out_first, out_second, out_scores, out_count};

    } else {
      int32_t* out_first = nullptr;
      int32_t* out_second = nullptr;
      float* out_scores = nullptr;
      cudaMalloc(&out_first, (size_t)num_pairs * sizeof(int32_t));
      cudaMalloc(&out_second, (size_t)num_pairs * sizeof(int32_t));
      cudaMalloc(&out_scores, (size_t)num_pairs * sizeof(float));

      if (use_u32) {
        launch_fused_from_counts_u32(d_offsets, d_seeds, all_seeds,
                                     (const uint32_t*)rle_keys, rle_counts,
                                     num_pairs, shift_v, mask_v,
                                     out_first, out_second, out_scores);
      } else {
        launch_fused_from_counts_u64(d_offsets, d_seeds, all_seeds,
                                     (const uint64_t*)rle_keys, rle_counts,
                                     num_pairs, shift_v, mask_v,
                                     out_first, out_second, out_scores);
      }
      cudaFree(rle_keys);
      cudaFree(rle_counts);

      return {out_first, out_second, out_scores, out_count};
    }

  } else {
    
    void* uniq_keys = nullptr;
    int32_t* uniq_count_d = nullptr;

    if (use_u32) {
      cudaMalloc(&uniq_keys, (size_t)total_exp_i32 * sizeof(uint32_t));
    } else {
      cudaMalloc(&uniq_keys, (size_t)total_exp_i32 * sizeof(uint64_t));
    }
    cudaMalloc(&uniq_count_d, sizeof(int32_t));

    if (use_u32) {
      size_t uniq_tmp_bytes = get_unique_temp_bytes_u32(total_exp_i32);
      void* uniq_tmp = nullptr;
      cudaMalloc(&uniq_tmp, uniq_tmp_bytes);
      launch_unique_u32((const uint32_t*)sorted_keys, (uint32_t*)uniq_keys,
                        uniq_count_d, total_exp_i32, uniq_tmp, uniq_tmp_bytes);
      cudaFree(uniq_tmp);
      launch_remove_self_unique_u32((uint32_t*)uniq_keys, uniq_count_d,
                                    d_seeds, all_seeds, num_seeds, shift_v);
    } else {
      size_t uniq_tmp_bytes = get_unique_temp_bytes_u64(total_exp_i32);
      void* uniq_tmp = nullptr;
      cudaMalloc(&uniq_tmp, uniq_tmp_bytes);
      launch_unique_u64((const uint64_t*)sorted_keys, (uint64_t*)uniq_keys,
                        uniq_count_d, total_exp_i32, uniq_tmp, uniq_tmp_bytes);
      cudaFree(uniq_tmp);
      launch_remove_self_unique_u64((uint64_t*)uniq_keys, uniq_count_d,
                                    d_seeds, all_seeds, num_seeds, shift_v);
    }
    cudaFree(sorted_keys);

    int32_t num_pairs = 0;
    cudaMemcpy(&num_pairs, uniq_count_d, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(uniq_count_d);

    if (num_pairs <= 0) {
      cudaFree(uniq_keys);
      return {nullptr, nullptr, nullptr, 0};
    }

    bool do_topk = (topk_raw >= 0 && topk_raw < (int64_t)num_pairs);
    std::size_t out_count = do_topk ? (std::size_t)topk_raw : (std::size_t)num_pairs;

    if (do_topk) {
      float* scores = nullptr;
      cudaMalloc(&scores, (size_t)num_pairs * sizeof(float));
      if (use_u32) {
        launch_overlap_scores_u32(d_offsets, d_indices, d_seeds, all_seeds,
                                  (const uint32_t*)uniq_keys, num_pairs,
                                  shift_v, mask_v, true, scores);
      } else {
        launch_overlap_scores_u64(d_offsets, d_indices, d_seeds, all_seeds,
                                  (const uint64_t*)uniq_keys, num_pairs,
                                  shift_v, mask_v, true, scores);
      }

      int32_t* idx_in = nullptr;
      cudaMalloc(&idx_in, (size_t)num_pairs * sizeof(int32_t));
      launch_iota_i32(idx_in, num_pairs);

      float* scores_sorted = nullptr;
      int32_t* idx_sorted = nullptr;
      cudaMalloc(&scores_sorted, (size_t)num_pairs * sizeof(float));
      cudaMalloc(&idx_sorted, (size_t)num_pairs * sizeof(int32_t));
      size_t sp_tmp_bytes = get_sort_pairs_temp_bytes(num_pairs);
      void* sp_tmp = nullptr;
      cudaMalloc(&sp_tmp, sp_tmp_bytes);
      launch_sort_pairs_desc(scores, scores_sorted, idx_in, idx_sorted,
                             num_pairs, sp_tmp, sp_tmp_bytes);
      cudaFree(sp_tmp);
      cudaFree(scores);
      cudaFree(idx_in);

      int32_t k = (int32_t)topk_raw;
      int32_t* out_first = nullptr;
      int32_t* out_second = nullptr;
      float* out_scores = nullptr;
      cudaMalloc(&out_first, (size_t)k * sizeof(int32_t));
      cudaMalloc(&out_second, (size_t)k * sizeof(int32_t));
      cudaMalloc(&out_scores, (size_t)k * sizeof(float));

      if (use_u32) {
        launch_gather_topk_u32((const uint32_t*)uniq_keys, d_seeds, all_seeds,
                               scores_sorted, idx_sorted, k, shift_v, mask_v,
                               out_first, out_second, out_scores);
      } else {
        launch_gather_topk_u64((const uint64_t*)uniq_keys, d_seeds, all_seeds,
                               scores_sorted, idx_sorted, k, shift_v, mask_v,
                               out_first, out_second, out_scores);
      }
      cudaFree(scores_sorted);
      cudaFree(idx_sorted);
      cudaFree(uniq_keys);

      return {out_first, out_second, out_scores, out_count};

    } else {
      int32_t* out_first = nullptr;
      int32_t* out_second = nullptr;
      float* out_scores = nullptr;
      cudaMalloc(&out_first, (size_t)num_pairs * sizeof(int32_t));
      cudaMalloc(&out_second, (size_t)num_pairs * sizeof(int32_t));
      cudaMalloc(&out_scores, (size_t)num_pairs * sizeof(float));

      if (use_u32) {
        launch_overlap_fused_u32(d_offsets, d_indices, d_seeds, all_seeds,
                                 (const uint32_t*)uniq_keys, num_pairs,
                                 shift_v, mask_v, true,
                                 out_first, out_second, out_scores);
      } else {
        launch_overlap_fused_u64(d_offsets, d_indices, d_seeds, all_seeds,
                                 (const uint64_t*)uniq_keys, num_pairs,
                                 shift_v, mask_v, true,
                                 out_first, out_second, out_scores);
      }
      cudaFree(uniq_keys);

      return {out_first, out_second, out_scores, out_count};
    }
  }
}

}  
