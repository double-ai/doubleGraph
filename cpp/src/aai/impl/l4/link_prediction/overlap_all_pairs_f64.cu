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
#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

namespace aai {

namespace {





constexpr int32_t kBlock = 256;
constexpr int64_t kMaxRawPerBatch = 58LL * (1LL << 15);
constexpr int32_t kFastMaxSeeds = 512;
constexpr int32_t kFastMaxCap = 65536;

static inline int64_t next_pow2_i64(int64_t v)
{
  if (v <= 1) return 1;
  --v;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  return v + 1;
}





struct Cache : Cacheable {
};





__device__ __forceinline__ int32_t lower_bound_dev(const int32_t* __restrict__ arr, int32_t size, int32_t target)
{
  int32_t lo = 0, hi = size;
  while (lo < hi) {
    int32_t mid = lo + ((hi - lo) >> 1);
    int32_t v = arr[mid];
    if (v < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

__device__ __forceinline__ int32_t gallop_lower_bound(const int32_t* __restrict__ arr, int32_t start, int32_t end, int32_t target)
{
  if (start >= end) return end;
  if (arr[start] >= target) return start;
  int32_t pos = start;
  int32_t step = 1;
  while (pos + step < end && arr[pos + step] < target) {
    pos += step;
    step <<= 1;
  }
  int32_t lo = pos;
  int32_t hi = (pos + step < end) ? (pos + step + 1) : end;
  while (lo < hi) {
    int32_t mid = lo + ((hi - lo) >> 1);
    if (arr[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

__device__ __forceinline__ uint32_t hash_u32(uint32_t x)
{
  x ^= x >> 16;
  x *= 0x85ebca6bU;
  x ^= x >> 13;
  x *= 0xc2b2ae35U;
  x ^= x >> 16;
  return x;
}

__device__ __forceinline__ bool hash_insert_i32(int32_t* __restrict__ table, int cap, int32_t key)
{
  uint32_t h = hash_u32(static_cast<uint32_t>(key));
  int slot = static_cast<int>(h & static_cast<uint32_t>(cap - 1));
  #pragma unroll 1
  for (int it = 0; it < 64; ++it) {
    int32_t prev = atomicCAS(&table[slot], -1, key);
    if (prev == -1) return true;
    if (prev == key) return false;
    slot = (slot + 1) & (cap - 1);
  }
  for (int it = 64; it < cap; ++it) {
    int32_t prev = atomicCAS(&table[slot], -1, key);
    if (prev == -1) return true;
    if (prev == key) return false;
    slot = (slot + 1) & (cap - 1);
  }
  return false;
}

__device__ __forceinline__ unsigned long long seed_map_lookup_bits(const int32_t* __restrict__ map_keys,
                                                                   const unsigned long long* __restrict__ map_vals,
                                                                   int cap,
                                                                   int32_t u)
{
  uint32_t h = hash_u32(static_cast<uint32_t>(u));
  int slot = static_cast<int>(h & static_cast<uint32_t>(cap - 1));
  #pragma unroll 1
  for (int it = 0; it < cap; ++it) {
    int32_t k = map_keys[slot];
    if (k == u) return map_vals[slot];
    if (k == -1) return 0ull;
    slot = (slot + 1) & (cap - 1);
  }
  return 0ull;
}





__global__ void iota_i32_kernel(int32_t* data, int32_t n)
{
  int32_t i = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) data[i] = i;
}

__global__ void weight_sums_kernel(const int32_t* __restrict__ offsets,
                                  const double* __restrict__ weights,
                                  double* __restrict__ out_wsum,
                                  int32_t n)
{
  int32_t v = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (v >= n) return;
  int32_t s = offsets[v];
  int32_t e = offsets[v + 1];
  double sum = 0.0;
  for (int32_t i = s; i < e; ++i) sum += weights[i];
  out_wsum[v] = sum;
}

__global__ void count_raw_kernel(const int32_t* __restrict__ offsets,
                                const int32_t* __restrict__ indices,
                                const int32_t* __restrict__ seeds,
                                int32_t num_seeds,
                                int64_t* __restrict__ out_counts)
{
  int32_t sid = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (sid >= num_seeds) return;
  int32_t u = seeds[sid];
  int32_t us = offsets[u];
  int32_t ue = offsets[u + 1];
  int64_t sum = 0;
  for (int32_t i = us; i < ue; ++i) {
    int32_t w = indices[i];
    sum += static_cast<int64_t>(offsets[w + 1] - offsets[w]) - 1;
  }
  if (sum < 0) sum = 0;
  out_counts[sid] = sum;
}

__global__ void generate_raw_keys_kernel(const int32_t* __restrict__ offsets,
                                        const int32_t* __restrict__ indices,
                                        const int32_t* __restrict__ seeds,
                                        int32_t num_seeds,
                                        const int64_t* __restrict__ seed_offsets,
                                        int64_t* __restrict__ out_keys)
{
  int32_t sid = static_cast<int32_t>(blockIdx.x);
  if (sid >= num_seeds) return;
  int32_t u = seeds[sid];
  int32_t us = offsets[u];
  int32_t ue = offsets[u + 1];
  int64_t base = seed_offsets[sid];

  __shared__ int32_t s_ctr;
  if (threadIdx.x == 0) s_ctr = 0;
  __syncthreads();

  for (int32_t i = us; i < ue; ++i) {
    int32_t w = indices[i];
    int32_t ws = offsets[w];
    int32_t we = offsets[w + 1];
    for (int32_t j = ws + threadIdx.x; j < we; j += blockDim.x) {
      int32_t v = indices[j];
      if (v == u) continue;
      int32_t pos = atomicAdd(&s_ctr, 1);
      out_keys[base + static_cast<int64_t>(pos)] = (static_cast<int64_t>(static_cast<uint32_t>(u)) << 32) |
                                                   static_cast<int64_t>(static_cast<uint32_t>(v));
    }
  }
}

__global__ void scores_from_keys_kernel(const int32_t* __restrict__ offsets,
                                       const int32_t* __restrict__ indices,
                                       const double* __restrict__ weights,
                                       const double* __restrict__ wsum,
                                       const int64_t* __restrict__ keys,
                                       int n,
                                       int32_t* __restrict__ out_first,
                                       int32_t* __restrict__ out_second,
                                       double* __restrict__ out_scores)
{
  int warp_id = (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (warp_id >= n) return;

  int64_t key = keys[warp_id];
  int32_t u = static_cast<int32_t>(static_cast<uint64_t>(key) >> 32);
  int32_t v = static_cast<int32_t>(static_cast<uint32_t>(key));

  if (lane == 0) {
    out_first[warp_id] = u;
    out_second[warp_id] = v;
  }

  double denom = fmin(wsum[u], wsum[v]);
  if (denom <= 0.0) {
    if (lane == 0) out_scores[warp_id] = 0.0;
    return;
  }

  int32_t us = offsets[u];
  int32_t ue = offsets[u + 1];
  int32_t vs = offsets[v];
  int32_t ve = offsets[v + 1];
  int32_t u_deg = ue - us;
  int32_t v_deg = ve - vs;

  int32_t ss, se, ls, le;
  if (u_deg <= v_deg) {
    ss = us; se = ue; ls = vs; le = ve;
  } else {
    ss = vs; se = ve; ls = us; le = ue;
  }

  int32_t s_sz = se - ss;
  int32_t l_sz = le - ls;

  double local = 0.0;
  if (l_sz > 10 * s_sz) {
    int32_t j = 0;
    for (int32_t i = ss + lane; i < se; i += 32) {
      int32_t t = indices[i];
      double wa = weights[i];
      j = gallop_lower_bound(indices + ls, j, l_sz, t);
      if (j < l_sz && indices[ls + j] == t) {
        double wb = weights[ls + j];
        local += (wa < wb) ? wa : wb;
        ++j;
      }
    }
  } else {
    for (int32_t i = ss + lane; i < se; i += 32) {
      int32_t t = indices[i];
      double wa = weights[i];
      int32_t lo = 0, hi = l_sz;
      const int32_t* l_arr = indices + ls;
      while (lo < hi) {
        int32_t mid = (lo + hi) >> 1;
        if (l_arr[mid] < t) lo = mid + 1;
        else hi = mid;
      }
      if (lo < l_sz && l_arr[lo] == t) {
        double wb = weights[ls + lo];
        local += (wa < wb) ? wa : wb;
      }
    }
  }

  for (int d = 16; d > 0; d >>= 1) {
    local += __shfl_down_sync(0xffffffff, local, d);
  }
  if (lane == 0) out_scores[warp_id] = local / denom;
}

__global__ void iota_perm_kernel(int32_t* data, int32_t n)
{
  int32_t i = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) data[i] = i;
}

__global__ void gather_topk_kernel(const int32_t* __restrict__ first,
                                  const int32_t* __restrict__ second,
                                  const double* __restrict__ scores_sorted,
                                  const int32_t* __restrict__ perm,
                                  int32_t* __restrict__ out_first,
                                  int32_t* __restrict__ out_second,
                                  double* __restrict__ out_scores,
                                  int k)
{
  int32_t i = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= k) return;
  int32_t src = perm[i];
  out_first[i] = first[src];
  out_second[i] = second[src];
  out_scores[i] = scores_sorted[i];
}

__global__ void encode_keys_kernel(const int32_t* __restrict__ first,
                                  const int32_t* __restrict__ second,
                                  int64_t* __restrict__ keys,
                                  int n)
{
  int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    keys[i] = (static_cast<int64_t>(static_cast<uint32_t>(first[i])) << 32) |
              static_cast<int64_t>(static_cast<uint32_t>(second[i]));
  }
}

__global__ void merge_topk_kernel(double* __restrict__ global_scores,
                                 int64_t* __restrict__ global_keys,
                                 int k,
                                 const double* __restrict__ batch_scores,
                                 const int64_t* __restrict__ batch_keys,
                                 int batch_k,
                                 double* __restrict__ tmp_scores,
                                 int64_t* __restrict__ tmp_keys)
{
  if (threadIdx.x != 0) return;

  if (batch_k == 0) {
    for (int i = 0; i < k; ++i) {
      global_scores[i] = -1.0;
      global_keys[i] = 0;
    }
    return;
  }

  int i = 0, j = 0;

  for (int out = 0; out < k; ++out) {
    double gs = (i < k) ? global_scores[i] : -1.0;
    double bs = (j < batch_k) ? batch_scores[j] : -1.0;
    if (bs > gs) {
      tmp_scores[out] = bs;
      tmp_keys[out] = batch_keys[j];
      ++j;
    } else {
      tmp_scores[out] = gs;
      tmp_keys[out] = global_keys[i];
      ++i;
    }
  }

  for (int out = 0; out < k; ++out) {
    global_scores[out] = tmp_scores[out];
    global_keys[out] = tmp_keys[out];
  }
}

__global__ void count_valid_topk_kernel(const double* __restrict__ scores, int k, int* __restrict__ out_count)
{
  if (threadIdx.x != 0) return;
  int lo = 0;
  int hi = k;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (scores[mid] >= 0.0) lo = mid + 1;
    else hi = mid;
  }
  *out_count = lo;
}

__global__ void decode_keys_kernel(const int64_t* __restrict__ keys, int32_t* __restrict__ first, int32_t* __restrict__ second, int n)
{
  int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    int64_t key = keys[i];
    first[i] = static_cast<int32_t>(static_cast<uint64_t>(key) >> 32);
    second[i] = static_cast<int32_t>(static_cast<uint32_t>(key));
  }
}

__global__ void build_candidates_hash_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    int cap,
    int32_t* __restrict__ hash_tables,
    int32_t* __restrict__ cand_lists,
    int32_t* __restrict__ cand_counts,
    int32_t* __restrict__ overflow_flag)
{
  int sid = static_cast<int>(blockIdx.x);
  if (sid >= num_seeds) return;

  int32_t u = seeds[sid];
  int32_t us = offsets[u];
  int32_t ue = offsets[u + 1];

  int32_t* table = hash_tables + static_cast<int64_t>(sid) * cap;
  int32_t* out = cand_lists + static_cast<int64_t>(sid) * cap;

  for (int32_t i = us; i < ue; ++i) {
    int32_t w = indices[i];
    int32_t ws = offsets[w];
    int32_t we = offsets[w + 1];

    for (int32_t j = ws + threadIdx.x; j < we; j += blockDim.x) {
      int32_t v = indices[j];
      if (v == u) continue;

      bool inserted = hash_insert_i32(table, cap, v);
      if (inserted) {
        int pos = atomicAdd(&cand_counts[sid], 1);
        if (pos < cap) {
          out[pos] = v;
        } else {
          atomicExch(overflow_flag, 1);
        }
      }
    }
  }

  if (threadIdx.x == 0) {
    int c = cand_counts[sid];
    if (c > cap) cand_counts[sid] = cap;
  }
}

__global__ void i32_to_i64_kernel(const int32_t* __restrict__ in, int64_t* __restrict__ out, int n)
{
  int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = static_cast<int64_t>(in[i]);
}

__global__ void build_keys_from_candidates_kernel(
    const int32_t* __restrict__ seeds,
    int num_seeds,
    int cap,
    const int32_t* __restrict__ cand_lists,
    const int32_t* __restrict__ cand_counts,
    const int64_t* __restrict__ offsets64,
    int64_t* __restrict__ out_keys)
{
  int sid = static_cast<int>(blockIdx.x);
  if (sid >= num_seeds) return;

  int32_t u = seeds[sid];
  int32_t c = cand_counts[sid];
  int64_t base = offsets64[sid];
  const int32_t* cand = cand_lists + static_cast<int64_t>(sid) * cap;

  for (int i = threadIdx.x; i < c; i += blockDim.x) {
    int32_t v = cand[i];
    out_keys[base + i] = (static_cast<int64_t>(static_cast<uint32_t>(u)) << 32) |
                         static_cast<int64_t>(static_cast<uint32_t>(v));
  }
}

__global__ void seed_weight_sums_kernel(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ weights,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    double* __restrict__ seed_wsum)
{
  int warp_id = (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (warp_id >= num_seeds) return;

  int32_t u = seeds[warp_id];
  int32_t s = offsets[u];
  int32_t e = offsets[u + 1];
  double local = 0.0;
  for (int32_t i = s + lane; i < e; i += 32) local += weights[i];
  for (int d = 16; d > 0; d >>= 1) local += __shfl_down_sync(0xffffffff, local, d);
  if (lane == 0) seed_wsum[warp_id] = local;
}

__global__ void build_seed_map_kernel(const int32_t* __restrict__ seeds,
                                     const double* __restrict__ seed_wsum,
                                     int n,
                                     int cap,
                                     int32_t* __restrict__ map_keys,
                                     unsigned long long* __restrict__ map_vals)
{
  int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;

  int32_t u = seeds[i];
  unsigned long long val = __double_as_longlong(seed_wsum[i]);
  uint32_t h = hash_u32(static_cast<uint32_t>(u));
  int slot = static_cast<int>(h & static_cast<uint32_t>(cap - 1));

  #pragma unroll 1
  for (int it = 0; it < cap; ++it) {
    int32_t prev = atomicCAS(&map_keys[slot], -1, u);
    if (prev == -1 || prev == u) {
      map_vals[slot] = val;
      return;
    }
    slot = (slot + 1) & (cap - 1);
  }
}

__global__ void scores_seedmap_from_keys_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ map_keys,
    const unsigned long long* __restrict__ map_vals,
    int map_cap,
    const int64_t* __restrict__ keys,
    int n,
    double* __restrict__ out_scores)
{
  int warp_id = (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
  int lane = threadIdx.x & 31;
  if (warp_id >= n) return;

  int64_t key = keys[warp_id];
  int32_t u = static_cast<int32_t>(static_cast<uint64_t>(key) >> 32);
  int32_t v = static_cast<int32_t>(static_cast<uint32_t>(key));

  unsigned long long deg_u_bits = 0ull;
  if (lane == 0) deg_u_bits = seed_map_lookup_bits(map_keys, map_vals, map_cap, u);
  deg_u_bits = __shfl_sync(0xffffffff, deg_u_bits, 0);
  double deg_u = __longlong_as_double(static_cast<long long>(deg_u_bits));

  int32_t vs = offsets[v];
  int32_t ve = offsets[v + 1];
  double local_deg_v = 0.0;
  for (int32_t i = vs + lane; i < ve; i += 32) local_deg_v += weights[i];
  for (int d = 16; d > 0; d >>= 1) local_deg_v += __shfl_down_sync(0xffffffff, local_deg_v, d);
  double deg_v = __shfl_sync(0xffffffff, local_deg_v, 0);

  double denom = fmin(deg_u, deg_v);
  if (denom <= 0.0) {
    if (lane == 0) out_scores[warp_id] = 0.0;
    return;
  }

  int32_t us = offsets[u];
  int32_t ue = offsets[u + 1];
  int32_t u_deg = ue - us;
  int32_t v_deg = ve - vs;

  int32_t ss, se, ls, le;
  if (u_deg <= v_deg) {
    ss = us; se = ue; ls = vs; le = ve;
  } else {
    ss = vs; se = ve; ls = us; le = ue;
  }
  int32_t s_sz = se - ss;
  int32_t l_sz = le - ls;

  double local = 0.0;
  if (l_sz > 10 * s_sz) {
    int32_t j = 0;
    for (int32_t i = ss + lane; i < se; i += 32) {
      int32_t t = indices[i];
      double wa = weights[i];
      j = gallop_lower_bound(indices + ls, j, l_sz, t);
      if (j < l_sz && indices[ls + j] == t) {
        double wb = weights[ls + j];
        local += (wa < wb) ? wa : wb;
        ++j;
      }
    }
  } else {
    for (int32_t i = ss + lane; i < se; i += 32) {
      int32_t t = indices[i];
      double wa = weights[i];
      int32_t lo = 0, hi = l_sz;
      const int32_t* l_arr = indices + ls;
      while (lo < hi) {
        int32_t mid = (lo + hi) >> 1;
        if (l_arr[mid] < t) lo = mid + 1;
        else hi = mid;
      }
      if (lo < l_sz && l_arr[lo] == t) {
        double wb = weights[ls + lo];
        local += (wa < wb) ? wa : wb;
      }
    }
  }

  for (int d = 16; d > 0; d >>= 1) local += __shfl_down_sync(0xffffffff, local, d);
  if (lane == 0) out_scores[warp_id] = local / denom;
}





static size_t cub_scan_temp_bytes(int n)
{
  size_t bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, bytes, (const int64_t*)nullptr, (int64_t*)nullptr, n);
  return bytes;
}

static void cub_scan_exclusive_sum(void* temp, size_t temp_bytes, const int64_t* in, int64_t* out, int n, cudaStream_t stream)
{
  cub::DeviceScan::ExclusiveSum(temp, temp_bytes, in, out, n, stream);
}

static size_t cub_sort_keys_temp_bytes(int n)
{
  size_t bytes = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, bytes, (const int64_t*)nullptr, (int64_t*)nullptr, n, 0, 64);
  return bytes;
}

static void cub_sort_keys_i64(void* temp, size_t temp_bytes, const int64_t* keys_in, int64_t* keys_out, int n, cudaStream_t stream)
{
  cub::DeviceRadixSort::SortKeys(temp, temp_bytes, keys_in, keys_out, n, 0, 64, stream);
}

static size_t cub_unique_temp_bytes(int n)
{
  size_t bytes = 0;
  cub::DeviceSelect::Unique(nullptr, bytes, (const int64_t*)nullptr, (int64_t*)nullptr, (int*)nullptr, n);
  return bytes;
}

static void cub_unique_i64(void* temp, size_t temp_bytes, const int64_t* in, int64_t* out, int* d_num_selected, int n, cudaStream_t stream)
{
  cub::DeviceSelect::Unique(temp, temp_bytes, in, out, d_num_selected, n, stream);
}

static size_t cub_sort_pairs_desc_temp_bytes(int n)
{
  size_t bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(nullptr, bytes,
                                           (const double*)nullptr, (double*)nullptr,
                                           (const int32_t*)nullptr, (int32_t*)nullptr,
                                           n, 0, 64);
  return bytes;
}

static void cub_sort_pairs_desc_double_i32(void* temp, size_t temp_bytes,
                                           const double* keys_in, double* keys_out,
                                           const int32_t* vals_in, int32_t* vals_out,
                                           int n, cudaStream_t stream)
{
  cub::DeviceRadixSort::SortPairsDescending(temp, temp_bytes, keys_in, keys_out, vals_in, vals_out, n, 0, 64, stream);
}

static size_t cub_sort_pairs_desc_i64_temp_bytes(int n)
{
  size_t bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(nullptr, bytes,
                                           (const double*)nullptr, (double*)nullptr,
                                           (const int64_t*)nullptr, (int64_t*)nullptr,
                                           n, 0, 64);
  return bytes;
}

static void cub_sort_pairs_desc_double_i64(void* temp, size_t temp_bytes,
                                           const double* keys_in, double* keys_out,
                                           const int64_t* vals_in, int64_t* vals_out,
                                           int n, cudaStream_t stream)
{
  cub::DeviceRadixSort::SortPairsDescending(temp, temp_bytes, keys_in, keys_out, vals_in, vals_out, n, 0, 64, stream);
}





static void launch_iota_i32(int32_t* data, int32_t n, cudaStream_t stream)
{
  if (n <= 0) return;
  int grid = (n + 255) / 256;
  iota_i32_kernel<<<grid, 256, 0, stream>>>(data, n);
}

static void launch_weight_sums(const int32_t* offsets, const double* weights, double* out_wsum,
                               int32_t num_vertices, cudaStream_t stream)
{
  if (num_vertices <= 0) return;
  int grid = (num_vertices + 255) / 256;
  weight_sums_kernel<<<grid, 256, 0, stream>>>(offsets, weights, out_wsum, num_vertices);
}

static void launch_count_raw(const int32_t* offsets, const int32_t* indices,
                             const int32_t* seeds, int32_t num_seeds,
                             int64_t* out_counts, cudaStream_t stream)
{
  if (num_seeds <= 0) return;
  int grid = (num_seeds + 255) / 256;
  count_raw_kernel<<<grid, 256, 0, stream>>>(offsets, indices, seeds, num_seeds, out_counts);
}

static void launch_generate_raw_keys(const int32_t* offsets, const int32_t* indices,
                                     const int32_t* seeds, int32_t num_seeds,
                                     const int64_t* seed_offsets,
                                     int64_t* out_keys, cudaStream_t stream)
{
  if (num_seeds <= 0) return;
  generate_raw_keys_kernel<<<num_seeds, 256, 0, stream>>>(offsets, indices, seeds, num_seeds, seed_offsets, out_keys);
}

static void launch_scores_from_keys(const int32_t* offsets, const int32_t* indices, const double* weights,
                                    const double* wsum,
                                    const int64_t* keys, int n,
                                    int32_t* out_first, int32_t* out_second, double* out_scores,
                                    cudaStream_t stream)
{
  if (n <= 0) return;
  int threads = 256;
  int warps_per_block = threads / 32;
  int blocks = (n + warps_per_block - 1) / warps_per_block;
  scores_from_keys_kernel<<<blocks, threads, 0, stream>>>(offsets, indices, weights, wsum, keys, n, out_first, out_second, out_scores);
}

static void launch_iota_perm(int32_t* perm, int32_t n, cudaStream_t stream)
{
  if (n <= 0) return;
  int grid = (n + 255) / 256;
  iota_perm_kernel<<<grid, 256, 0, stream>>>(perm, n);
}

static void launch_gather_topk(const int32_t* first, const int32_t* second, const double* scores_sorted,
                               const int32_t* perm, int32_t* out_first, int32_t* out_second, double* out_scores,
                               int k, cudaStream_t stream)
{
  if (k <= 0) return;
  int grid = (k + 255) / 256;
  gather_topk_kernel<<<grid, 256, 0, stream>>>(first, second, scores_sorted, perm, out_first, out_second, out_scores, k);
}

static void launch_encode_keys(const int32_t* first, const int32_t* second, int64_t* keys, int n, cudaStream_t stream)
{
  if (n <= 0) return;
  int grid = (n + 255) / 256;
  encode_keys_kernel<<<grid, 256, 0, stream>>>(first, second, keys, n);
}

static void launch_merge_topk(double* global_scores, int64_t* global_keys, int k,
                              const double* batch_scores, const int64_t* batch_keys, int batch_k,
                              double* tmp_scores, int64_t* tmp_keys,
                              cudaStream_t stream)
{
  merge_topk_kernel<<<1, 32, 0, stream>>>(global_scores, global_keys, k, batch_scores, batch_keys, batch_k, tmp_scores, tmp_keys);
}

static void launch_count_valid_topk(const double* global_scores, int k, int* out_count, cudaStream_t stream)
{
  count_valid_topk_kernel<<<1, 32, 0, stream>>>(global_scores, k, out_count);
}

static void launch_decode_keys(const int64_t* keys, int32_t* first, int32_t* second, int n, cudaStream_t stream)
{
  if (n <= 0) return;
  int grid = (n + 255) / 256;
  decode_keys_kernel<<<grid, 256, 0, stream>>>(keys, first, second, n);
}

static void launch_build_candidates_hash(const int32_t* offsets, const int32_t* indices,
                                         const int32_t* seeds, int num_seeds,
                                         int cap, int32_t* hash_tables,
                                         int32_t* cand_lists, int32_t* cand_counts,
                                         int32_t* overflow_flag,
                                         cudaStream_t stream)
{
  if (num_seeds <= 0) return;
  build_candidates_hash_kernel<<<num_seeds, 256, 0, stream>>>(
      offsets, indices, seeds, num_seeds, cap, hash_tables, cand_lists, cand_counts, overflow_flag);
}

static void launch_i32_to_i64(const int32_t* in, int64_t* out, int n, cudaStream_t stream)
{
  if (n <= 0) return;
  int grid = (n + 255) / 256;
  i32_to_i64_kernel<<<grid, 256, 0, stream>>>(in, out, n);
}

static void launch_build_keys_from_candidates(const int32_t* seeds, int num_seeds,
                                              int cap, const int32_t* cand_lists,
                                              const int32_t* cand_counts,
                                              const int64_t* offsets64,
                                              int64_t* out_keys, cudaStream_t stream)
{
  if (num_seeds <= 0) return;
  build_keys_from_candidates_kernel<<<num_seeds, 256, 0, stream>>>(
      seeds, num_seeds, cap, cand_lists, cand_counts, offsets64, out_keys);
}

static void launch_seed_weight_sums(const int32_t* offsets, const double* weights,
                                    const int32_t* seeds, int num_seeds,
                                    double* seed_wsum, cudaStream_t stream)
{
  if (num_seeds <= 0) return;
  int threads = 256;
  int warps_per_block = threads / 32;
  int blocks = (num_seeds + warps_per_block - 1) / warps_per_block;
  seed_weight_sums_kernel<<<blocks, threads, 0, stream>>>(offsets, weights, seeds, num_seeds, seed_wsum);
}

static void launch_build_seed_map(const int32_t* seeds, const double* seed_wsum,
                                  int n, int cap,
                                  int32_t* map_keys, unsigned long long* map_vals,
                                  cudaStream_t stream)
{
  if (n <= 0) return;
  int grid = (n + 255) / 256;
  build_seed_map_kernel<<<grid, 256, 0, stream>>>(seeds, seed_wsum, n, cap, map_keys, map_vals);
}

static void launch_scores_seedmap_from_keys(const int32_t* offsets, const int32_t* indices,
                                            const double* weights,
                                            const int32_t* map_keys,
                                            const unsigned long long* map_vals,
                                            int map_cap,
                                            const int64_t* keys, int n,
                                            double* out_scores,
                                            cudaStream_t stream)
{
  if (n <= 0) return;
  int threads = 256;
  int warps_per_block = threads / 32;
  int blocks = (n + warps_per_block - 1) / warps_per_block;
  scores_seedmap_from_keys_kernel<<<blocks, threads, 0, stream>>>(
      offsets, indices, weights, map_keys, map_vals, map_cap, keys, n, out_scores);
}





static bool run_topk_fastpath(const int32_t* d_offsets,
                             const int32_t* d_indices,
                             const double* d_weights,
                             const int32_t* d_seeds,
                             int32_t num_seeds,
                             int32_t num_vertices,
                             int topk,
                             cudaStream_t stream,
                             similarity_result_double_t& out)
{
  if (num_seeds <= 0 || topk <= 0) return false;
  if (num_seeds > kFastMaxSeeds) return false;

  int64_t* raw_counts = nullptr;
  cudaMalloc(&raw_counts, static_cast<size_t>(num_seeds) * sizeof(int64_t));
  launch_count_raw(d_offsets, d_indices, d_seeds, num_seeds, raw_counts, stream);

  std::vector<int64_t> h_raw(num_seeds);
  cudaMemcpyAsync(h_raw.data(), raw_counts, sizeof(int64_t) * num_seeds, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaFree(raw_counts);

  int64_t max_raw = 0;
  for (auto c : h_raw) max_raw = std::max(max_raw, c);

  int64_t est = std::min<int64_t>(static_cast<int64_t>(num_vertices), max_raw * 2 + 16);
  int64_t cap64 = next_pow2_i64(est);
  if (cap64 < 128) cap64 = 128;
  if (num_vertices > kFastMaxCap && cap64 > kFastMaxCap) return false;
  int32_t cap = static_cast<int32_t>(cap64);

  int32_t* hash_buf = nullptr;
  int32_t* cand_buf = nullptr;
  int32_t* cand_counts = nullptr;
  int32_t* overflow_flag = nullptr;
  cudaMalloc(&hash_buf, static_cast<size_t>(num_seeds) * cap * sizeof(int32_t));
  cudaMalloc(&cand_buf, static_cast<size_t>(num_seeds) * cap * sizeof(int32_t));
  cudaMalloc(&cand_counts, static_cast<size_t>(num_seeds) * sizeof(int32_t));
  cudaMalloc(&overflow_flag, sizeof(int32_t));

  cudaMemsetAsync(hash_buf, 0xFF, sizeof(int32_t) * static_cast<size_t>(num_seeds) * cap, stream);
  cudaMemsetAsync(cand_counts, 0, sizeof(int32_t) * static_cast<size_t>(num_seeds), stream);
  cudaMemsetAsync(overflow_flag, 0, sizeof(int32_t), stream);

  launch_build_candidates_hash(d_offsets, d_indices, d_seeds, num_seeds, cap,
                              hash_buf, cand_buf, cand_counts, overflow_flag, stream);

  int32_t h_overflow = 0;
  cudaMemcpyAsync(&h_overflow, overflow_flag, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaFree(overflow_flag);

  if (h_overflow != 0) {
    cudaFree(hash_buf);
    cudaFree(cand_buf);
    cudaFree(cand_counts);
    return false;
  }
  cudaFree(hash_buf);

  int64_t* counts64 = nullptr;
  cudaMalloc(&counts64, static_cast<size_t>(num_seeds) * sizeof(int64_t));
  launch_i32_to_i64(cand_counts, counts64, num_seeds, stream);

  int64_t* offs64 = nullptr;
  cudaMalloc(&offs64, static_cast<size_t>(num_seeds) * sizeof(int64_t));
  {
    size_t tmp_bytes = cub_scan_temp_bytes(num_seeds);
    void* tmp = nullptr;
    cudaMalloc(&tmp, tmp_bytes);
    cub_scan_exclusive_sum(tmp, tmp_bytes, counts64, offs64, num_seeds, stream);
    cudaFree(tmp);
  }

  int64_t last_off = 0;
  int64_t last_cnt = 0;
  cudaMemcpyAsync(&last_off, offs64 + (num_seeds - 1), sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&last_cnt, counts64 + (num_seeds - 1), sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  int64_t total_pairs = last_off + last_cnt;
  if (total_pairs <= 0) {
    cudaFree(cand_buf);
    cudaFree(cand_counts);
    cudaFree(counts64);
    cudaFree(offs64);
    out = {nullptr, nullptr, nullptr, 0};
    return true;
  }
  if (total_pairs > std::numeric_limits<int>::max()) {
    cudaFree(cand_buf);
    cudaFree(cand_counts);
    cudaFree(counts64);
    cudaFree(offs64);
    return false;
  }

  cudaFree(counts64);

  int64_t* keys = nullptr;
  cudaMalloc(&keys, static_cast<size_t>(total_pairs) * sizeof(int64_t));
  launch_build_keys_from_candidates(d_seeds, num_seeds, cap, cand_buf, cand_counts, offs64, keys, stream);
  cudaFree(cand_buf);
  cudaFree(cand_counts);
  cudaFree(offs64);

  double* seed_wsum = nullptr;
  cudaMalloc(&seed_wsum, static_cast<size_t>(num_seeds) * sizeof(double));
  launch_seed_weight_sums(d_offsets, d_weights, d_seeds, num_seeds, seed_wsum, stream);

  int map_cap = 1;
  while (map_cap < (num_seeds * 4)) map_cap <<= 1;
  if (map_cap < 128) map_cap = 128;

  int32_t* map_keys = nullptr;
  unsigned long long* map_vals = nullptr;
  cudaMalloc(&map_keys, static_cast<size_t>(map_cap) * sizeof(int32_t));
  cudaMalloc(&map_vals, static_cast<size_t>(map_cap) * sizeof(unsigned long long));
  cudaMemsetAsync(map_keys, 0xFF, sizeof(int32_t) * static_cast<size_t>(map_cap), stream);
  launch_build_seed_map(d_seeds, seed_wsum, num_seeds, map_cap, map_keys, map_vals, stream);
  cudaFree(seed_wsum);

  double* scores = nullptr;
  cudaMalloc(&scores, static_cast<size_t>(total_pairs) * sizeof(double));
  launch_scores_seedmap_from_keys(d_offsets, d_indices, d_weights,
                                  map_keys, map_vals, map_cap,
                                  keys, static_cast<int>(total_pairs), scores, stream);
  cudaFree(map_keys);
  cudaFree(map_vals);

  int64_t out_count64 = std::min<int64_t>(static_cast<int64_t>(topk), total_pairs);
  int out_count = static_cast<int>(out_count64);

  int32_t* first_out = nullptr;
  int32_t* second_out = nullptr;
  double* scores_out = nullptr;
  cudaMalloc(&first_out, static_cast<size_t>(out_count) * sizeof(int32_t));
  cudaMalloc(&second_out, static_cast<size_t>(out_count) * sizeof(int32_t));
  cudaMalloc(&scores_out, static_cast<size_t>(out_count) * sizeof(double));

  if (total_pairs > topk) {
    double* sorted_scores = nullptr;
    int64_t* sorted_keys = nullptr;
    cudaMalloc(&sorted_scores, static_cast<size_t>(total_pairs) * sizeof(double));
    cudaMalloc(&sorted_keys, static_cast<size_t>(total_pairs) * sizeof(int64_t));

    size_t tmp_bytes = cub_sort_pairs_desc_i64_temp_bytes(static_cast<int>(total_pairs));
    void* tmp = nullptr;
    cudaMalloc(&tmp, tmp_bytes);
    cub_sort_pairs_desc_double_i64(tmp, tmp_bytes, scores, sorted_scores, keys, sorted_keys,
                                   static_cast<int>(total_pairs), stream);
    cudaFree(tmp);

    launch_decode_keys(sorted_keys, first_out, second_out, out_count, stream);
    cudaMemcpyAsync(scores_out, sorted_scores, sizeof(double) * out_count, cudaMemcpyDeviceToDevice, stream);

    cudaFree(sorted_scores);
    cudaFree(sorted_keys);
  } else {
    launch_decode_keys(keys, first_out, second_out, out_count, stream);
    cudaMemcpyAsync(scores_out, scores, sizeof(double) * out_count, cudaMemcpyDeviceToDevice, stream);
  }

  cudaFree(keys);
  cudaFree(scores);

  out = {first_out, second_out, scores_out, static_cast<std::size_t>(out_count)};
  return true;
}





static similarity_result_double_t run_all(const int32_t* d_offsets,
                                         const int32_t* d_indices,
                                         const double* d_weights,
                                         const double* d_wsum,
                                         const int32_t* d_seeds,
                                         int32_t num_seeds,
                                         cudaStream_t stream)
{
  int64_t* counts = nullptr;
  cudaMalloc(&counts, static_cast<size_t>(num_seeds) * sizeof(int64_t));
  launch_count_raw(d_offsets, d_indices, d_seeds, num_seeds, counts, stream);

  int64_t* scan_offsets = nullptr;
  cudaMalloc(&scan_offsets, static_cast<size_t>(num_seeds) * sizeof(int64_t));
  {
    size_t tmp_bytes = cub_scan_temp_bytes(num_seeds);
    void* tmp = nullptr;
    cudaMalloc(&tmp, tmp_bytes);
    cub_scan_exclusive_sum(tmp, tmp_bytes, counts, scan_offsets, num_seeds, stream);
    cudaFree(tmp);
  }

  int64_t last_off = 0, last_cnt = 0;
  cudaMemcpyAsync(&last_off, scan_offsets + (num_seeds - 1), sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaMemcpyAsync(&last_cnt, counts + (num_seeds - 1), sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  int64_t total_raw = last_off + last_cnt;

  if (total_raw == 0) {
    cudaFree(counts);
    cudaFree(scan_offsets);
    return {nullptr, nullptr, nullptr, 0};
  }

  int64_t* raw_keys = nullptr;
  cudaMalloc(&raw_keys, static_cast<size_t>(total_raw) * sizeof(int64_t));
  launch_generate_raw_keys(d_offsets, d_indices, d_seeds, num_seeds, scan_offsets, raw_keys, stream);
  cudaFree(scan_offsets);
  cudaFree(counts);

  int64_t* sorted_keys = nullptr;
  cudaMalloc(&sorted_keys, static_cast<size_t>(total_raw) * sizeof(int64_t));
  {
    size_t tmp_bytes = cub_sort_keys_temp_bytes(static_cast<int>(total_raw));
    void* tmp = nullptr;
    cudaMalloc(&tmp, tmp_bytes);
    cub_sort_keys_i64(tmp, tmp_bytes, raw_keys, sorted_keys, static_cast<int>(total_raw), stream);
    cudaFree(tmp);
  }
  cudaFree(raw_keys);

  int64_t* uniq_keys = nullptr;
  cudaMalloc(&uniq_keys, static_cast<size_t>(total_raw) * sizeof(int64_t));
  int32_t* d_num_uniq = nullptr;
  cudaMalloc(&d_num_uniq, sizeof(int32_t));
  {
    size_t tmp_bytes = cub_unique_temp_bytes(static_cast<int>(total_raw));
    void* tmp = nullptr;
    cudaMalloc(&tmp, tmp_bytes);
    cub_unique_i64(tmp, tmp_bytes, sorted_keys, uniq_keys, d_num_uniq, static_cast<int>(total_raw), stream);
    cudaFree(tmp);
  }
  cudaFree(sorted_keys);

  int32_t num_pairs = 0;
  cudaMemcpyAsync(&num_pairs, d_num_uniq, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaFree(d_num_uniq);

  if (num_pairs <= 0) {
    cudaFree(uniq_keys);
    return {nullptr, nullptr, nullptr, 0};
  }

  int32_t* first = nullptr;
  int32_t* second = nullptr;
  double* scores = nullptr;
  cudaMalloc(&first, static_cast<size_t>(num_pairs) * sizeof(int32_t));
  cudaMalloc(&second, static_cast<size_t>(num_pairs) * sizeof(int32_t));
  cudaMalloc(&scores, static_cast<size_t>(num_pairs) * sizeof(double));

  launch_scores_from_keys(d_offsets, d_indices, d_weights, d_wsum,
                          uniq_keys, num_pairs, first, second, scores, stream);
  cudaFree(uniq_keys);

  return {first, second, scores, static_cast<std::size_t>(num_pairs)};
}





static similarity_result_double_t run_topk(const int32_t* d_offsets,
                                          const int32_t* d_indices,
                                          const double* d_weights,
                                          const double* d_wsum,
                                          const int32_t* d_seeds,
                                          int32_t num_seeds,
                                          int topk,
                                          cudaStream_t stream)
{
  if (topk <= 0) {
    return {nullptr, nullptr, nullptr, 0};
  }

  int64_t* counts_all = nullptr;
  cudaMalloc(&counts_all, static_cast<size_t>(num_seeds) * sizeof(int64_t));
  launch_count_raw(d_offsets, d_indices, d_seeds, num_seeds, counts_all, stream);

  std::vector<int64_t> h_counts(num_seeds);
  cudaMemcpyAsync(h_counts.data(), counts_all, sizeof(int64_t) * num_seeds, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  std::vector<int32_t> batch_starts;
  batch_starts.reserve(128);
  batch_starts.push_back(0);
  int64_t acc = 0;
  for (int32_t i = 0; i < num_seeds; ++i) {
    int64_t c = h_counts[i];
    if (acc > 0 && acc + c > kMaxRawPerBatch) {
      batch_starts.push_back(i);
      acc = 0;
    }
    acc += c;
  }
  batch_starts.push_back(num_seeds);

  double* global_scores = nullptr;
  int64_t* global_keys = nullptr;
  double* merge_tmp_scores = nullptr;
  int64_t* merge_tmp_keys = nullptr;
  cudaMalloc(&global_scores, static_cast<size_t>(topk) * sizeof(double));
  cudaMalloc(&global_keys, static_cast<size_t>(topk) * sizeof(int64_t));
  cudaMalloc(&merge_tmp_scores, static_cast<size_t>(topk) * sizeof(double));
  cudaMalloc(&merge_tmp_keys, static_cast<size_t>(topk) * sizeof(int64_t));

  launch_merge_topk(global_scores, global_keys, topk,
                    nullptr, nullptr, 0,
                    merge_tmp_scores, merge_tmp_keys, stream);

  for (size_t bi = 0; bi + 1 < batch_starts.size(); ++bi) {
    int32_t b0 = batch_starts[bi];
    int32_t b1 = batch_starts[bi + 1];
    int32_t bsz = b1 - b0;
    if (bsz <= 0) continue;

    const int32_t* batch_seeds = d_seeds + b0;

    int64_t* b_counts = nullptr;
    cudaMalloc(&b_counts, static_cast<size_t>(bsz) * sizeof(int64_t));
    cudaMemcpyAsync(b_counts, counts_all + b0, sizeof(int64_t) * bsz, cudaMemcpyDeviceToDevice, stream);

    int64_t* b_off = nullptr;
    cudaMalloc(&b_off, static_cast<size_t>(bsz) * sizeof(int64_t));
    {
      size_t tmp_bytes = cub_scan_temp_bytes(bsz);
      void* tmp = nullptr;
      cudaMalloc(&tmp, tmp_bytes);
      cub_scan_exclusive_sum(tmp, tmp_bytes, b_counts, b_off, bsz, stream);
      cudaFree(tmp);
    }

    int64_t last_off = 0, last_cnt = 0;
    cudaMemcpyAsync(&last_off, b_off + (bsz - 1), sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&last_cnt, b_counts + (bsz - 1), sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int64_t total_raw = last_off + last_cnt;

    cudaFree(b_counts);

    if (total_raw <= 0) {
      cudaFree(b_off);
      continue;
    }

    int64_t* raw_keys = nullptr;
    cudaMalloc(&raw_keys, static_cast<size_t>(total_raw) * sizeof(int64_t));
    launch_generate_raw_keys(d_offsets, d_indices, batch_seeds, bsz, b_off, raw_keys, stream);
    cudaFree(b_off);

    int64_t* sorted_keys = nullptr;
    cudaMalloc(&sorted_keys, static_cast<size_t>(total_raw) * sizeof(int64_t));
    {
      size_t tmp_bytes = cub_sort_keys_temp_bytes(static_cast<int>(total_raw));
      void* tmp = nullptr;
      cudaMalloc(&tmp, tmp_bytes);
      cub_sort_keys_i64(tmp, tmp_bytes, raw_keys, sorted_keys, static_cast<int>(total_raw), stream);
      cudaFree(tmp);
    }
    cudaFree(raw_keys);

    int64_t* uniq_keys = nullptr;
    cudaMalloc(&uniq_keys, static_cast<size_t>(total_raw) * sizeof(int64_t));
    int32_t* d_num_uniq = nullptr;
    cudaMalloc(&d_num_uniq, sizeof(int32_t));
    {
      size_t tmp_bytes = cub_unique_temp_bytes(static_cast<int>(total_raw));
      void* tmp = nullptr;
      cudaMalloc(&tmp, tmp_bytes);
      cub_unique_i64(tmp, tmp_bytes, sorted_keys, uniq_keys, d_num_uniq, static_cast<int>(total_raw), stream);
      cudaFree(tmp);
    }
    cudaFree(sorted_keys);

    int32_t num_pairs = 0;
    cudaMemcpyAsync(&num_pairs, d_num_uniq, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_num_uniq);

    if (num_pairs <= 0) {
      cudaFree(uniq_keys);
      continue;
    }

    int32_t* b_first = nullptr;
    int32_t* b_second = nullptr;
    double* b_scores = nullptr;
    cudaMalloc(&b_first, static_cast<size_t>(num_pairs) * sizeof(int32_t));
    cudaMalloc(&b_second, static_cast<size_t>(num_pairs) * sizeof(int32_t));
    cudaMalloc(&b_scores, static_cast<size_t>(num_pairs) * sizeof(double));
    launch_scores_from_keys(d_offsets, d_indices, d_weights, d_wsum,
                            uniq_keys, num_pairs, b_first, b_second, b_scores, stream);
    cudaFree(uniq_keys);

    int32_t* perm_in = nullptr;
    int32_t* perm_out = nullptr;
    double* scores_sorted = nullptr;
    cudaMalloc(&perm_in, static_cast<size_t>(num_pairs) * sizeof(int32_t));
    cudaMalloc(&perm_out, static_cast<size_t>(num_pairs) * sizeof(int32_t));
    cudaMalloc(&scores_sorted, static_cast<size_t>(num_pairs) * sizeof(double));
    launch_iota_perm(perm_in, num_pairs, stream);
    {
      size_t tmp_bytes = cub_sort_pairs_desc_temp_bytes(num_pairs);
      void* tmp = nullptr;
      cudaMalloc(&tmp, tmp_bytes);
      cub_sort_pairs_desc_double_i32(tmp, tmp_bytes,
                                     b_scores, scores_sorted,
                                     perm_in, perm_out,
                                     num_pairs, stream);
      cudaFree(tmp);
    }
    cudaFree(perm_in);

    int batch_k = std::min<int>(topk, num_pairs);

    int32_t* batch_first = nullptr;
    int32_t* batch_second = nullptr;
    double* batch_scores = nullptr;
    cudaMalloc(&batch_first, static_cast<size_t>(batch_k) * sizeof(int32_t));
    cudaMalloc(&batch_second, static_cast<size_t>(batch_k) * sizeof(int32_t));
    cudaMalloc(&batch_scores, static_cast<size_t>(batch_k) * sizeof(double));
    launch_gather_topk(b_first, b_second, scores_sorted, perm_out,
                      batch_first, batch_second, batch_scores, batch_k, stream);
    cudaFree(perm_out);
    cudaFree(scores_sorted);
    cudaFree(b_first);
    cudaFree(b_second);
    cudaFree(b_scores);

    int64_t* batch_keys = nullptr;
    cudaMalloc(&batch_keys, static_cast<size_t>(batch_k) * sizeof(int64_t));
    launch_encode_keys(batch_first, batch_second, batch_keys, batch_k, stream);

    launch_merge_topk(global_scores, global_keys, topk,
                      batch_scores, batch_keys, batch_k,
                      merge_tmp_scores, merge_tmp_keys, stream);

    cudaFree(batch_first);
    cudaFree(batch_second);
    cudaFree(batch_scores);
    cudaFree(batch_keys);
  }

  cudaFree(counts_all);
  cudaFree(merge_tmp_scores);
  cudaFree(merge_tmp_keys);

  int32_t* d_count_i32 = nullptr;
  cudaMalloc(&d_count_i32, sizeof(int32_t));
  launch_count_valid_topk(global_scores, topk, d_count_i32, stream);
  int32_t out_count_i32 = 0;
  cudaMemcpyAsync(&out_count_i32, d_count_i32, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaFree(d_count_i32);

  int32_t out_count = std::max<int32_t>(0, std::min<int32_t>(topk, out_count_i32));

  if (out_count <= 0) {
    cudaFree(global_scores);
    cudaFree(global_keys);
    return {nullptr, nullptr, nullptr, 0};
  }

  int32_t* first_out = nullptr;
  int32_t* second_out = nullptr;
  double* scores_out = nullptr;
  cudaMalloc(&first_out, static_cast<size_t>(out_count) * sizeof(int32_t));
  cudaMalloc(&second_out, static_cast<size_t>(out_count) * sizeof(int32_t));
  cudaMalloc(&scores_out, static_cast<size_t>(out_count) * sizeof(double));

  launch_decode_keys(global_keys, first_out, second_out, out_count, stream);
  cudaMemcpyAsync(scores_out, global_scores, sizeof(double) * out_count, cudaMemcpyDeviceToDevice, stream);

  cudaFree(global_scores);
  cudaFree(global_keys);

  return {first_out, second_out, scores_out, static_cast<std::size_t>(out_count)};
}

}  





similarity_result_double_t overlap_all_pairs_similarity(const graph32_t& graph,
                                                        const double* edge_weights,
                                                        const int32_t* vertices,
                                                        std::size_t num_vertices_param,
                                                        std::optional<std::size_t> topk)
{
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);
  (void)cache;

  cudaStream_t stream = 0;

  const int32_t* d_offsets = graph.offsets;
  const int32_t* d_indices = graph.indices;
  const double* d_weights = edge_weights;
  int32_t num_verts = graph.number_of_vertices;

  const int32_t* d_seeds = nullptr;
  int32_t num_seeds = 0;
  int32_t* seeds_buf = nullptr;

  if (vertices != nullptr && num_vertices_param > 0) {
    d_seeds = vertices;
    num_seeds = static_cast<int32_t>(num_vertices_param);
  } else {
    num_seeds = num_verts;
    cudaMalloc(&seeds_buf, static_cast<size_t>(num_verts) * sizeof(int32_t));
    launch_iota_i32(seeds_buf, num_verts, stream);
    d_seeds = seeds_buf;
  }

  if (num_verts <= 0 || num_seeds <= 0) {
    if (seeds_buf) cudaFree(seeds_buf);
    return {nullptr, nullptr, nullptr, 0};
  }

  int64_t topk_raw = topk.has_value() ? static_cast<int64_t>(topk.value()) : -1;

  
  if (topk_raw >= 0 && vertices != nullptr && num_vertices_param > 0 &&
      num_seeds <= kFastMaxSeeds && static_cast<int>(topk_raw) <= 2048) {
    similarity_result_double_t fast_result;
    if (run_topk_fastpath(d_offsets, d_indices, d_weights, d_seeds, num_seeds,
                          num_verts, static_cast<int>(topk_raw), stream, fast_result)) {
      if (seeds_buf) cudaFree(seeds_buf);
      return fast_result;
    }
  }

  
  double* wsum = nullptr;
  cudaMalloc(&wsum, static_cast<size_t>(num_verts) * sizeof(double));
  launch_weight_sums(d_offsets, d_weights, wsum, num_verts, stream);

  similarity_result_double_t result;
  if (topk_raw >= 0) {
    result = run_topk(d_offsets, d_indices, d_weights, wsum, d_seeds, num_seeds,
                      static_cast<int>(topk_raw), stream);
  } else {
    result = run_all(d_offsets, d_indices, d_weights, wsum, d_seeds, num_seeds, stream);
  }

  cudaFree(wsum);
  if (seeds_buf) cudaFree(seeds_buf);
  return result;
}

}  
