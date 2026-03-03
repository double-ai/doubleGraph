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
#include <math_constants.h>

namespace aai {

namespace {

#ifndef MAX_SMEM_PER_WARP
#define MAX_SMEM_PER_WARP 256
#endif

#ifndef CACHE_MIN_LEN
#define CACHE_MIN_LEN 1000000000
#endif





__device__ __forceinline__ int32_t d_lower_bound(const int32_t* __restrict__ a,
                                                 int32_t lo,
                                                 int32_t hi,
                                                 int32_t target)
{
  while (lo < hi) {
    int32_t mid = lo + ((hi - lo) >> 1);
    int32_t v = __ldg(a + mid);
    if (v < target)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

__device__ __forceinline__ int32_t d_upper_bound(const int32_t* __restrict__ a,
                                                 int32_t lo,
                                                 int32_t hi,
                                                 int32_t target)
{
  while (lo < hi) {
    int32_t mid = lo + ((hi - lo) >> 1);
    int32_t v = __ldg(a + mid);
    if (v <= target)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

__device__ __forceinline__ bool trim_ranges(const int32_t* __restrict__ indices,
                                            int32_t& a0,
                                            int32_t& a1,
                                            int32_t& b0,
                                            int32_t& b1)
{
  
  int32_t a_first = __ldg(indices + a0);
  int32_t a_last = __ldg(indices + (a1 - 1));
  int32_t b_first = __ldg(indices + b0);
  int32_t b_last = __ldg(indices + (b1 - 1));

  if (a_last < b_first || b_last < a_first) return false;

  if (a_first < b_first) {
    a0 = d_lower_bound(indices, a0, a1, b_first);
    if (a0 >= a1) return false;
  } else if (b_first < a_first) {
    b0 = d_lower_bound(indices, b0, b1, a_first);
    if (b0 >= b1) return false;
  }

  a_last = __ldg(indices + (a1 - 1));
  b_last = __ldg(indices + (b1 - 1));
  int32_t min_last = (a_last < b_last) ? a_last : b_last;

  if (a_last > min_last) a1 = d_upper_bound(indices, a0, a1, min_last);
  if (b_last > min_last) b1 = d_upper_bound(indices, b0, b1, min_last);

  return (a0 < a1) && (b0 < b1);
}

__device__ __forceinline__ double qnan64() { return __longlong_as_double(0x7FF8000000000000ULL); }


__device__ __forceinline__ int32_t dup_rank_before(const int32_t* __restrict__ indices,
                                                   int32_t base,
                                                   int32_t j,
                                                   int32_t target)
{
  if (j <= 0) return 0;
  if (__ldg(indices + (base + j - 1)) != target) return 0;
  int32_t rank = 1;
  for (int32_t k = j - 2; k >= 0; --k) {
    if (__ldg(indices + (base + k)) != target) break;
    ++rank;
  }
  return rank;
}






__global__ __launch_bounds__(64, 16) void cosine_similarity_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ scores,
    int64_t num_pairs)
{
  constexpr int WARP = 32;
  int32_t lane = threadIdx.x & (WARP - 1);
  int32_t warp_in_block = threadIdx.x >> 5;
  int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp_id >= num_pairs) return;

  
  extern __shared__ int32_t smem_i[];
  int32_t* cache_i = smem_i + warp_in_block * MAX_SMEM_PER_WARP;

  int32_t u = __ldg(first + warp_id);
  int32_t v = __ldg(second + warp_id);

  int32_t a0 = __ldg(offsets + u);
  int32_t a1 = __ldg(offsets + u + 1);
  int32_t b0 = __ldg(offsets + v);
  int32_t b1 = __ldg(offsets + v + 1);

  
  if (a0 == a1 || b0 == b1) {
    if (lane == 0) scores[warp_id] = qnan64();
    return;
  }

  
  bool ok = true;
  if (lane == 0) {
    ok = trim_ranges(indices, a0, a1, b0, b1);
  }
  ok = __shfl_sync(0xffffffff, ok, 0);
  if (!ok) {
    if (lane == 0) scores[warp_id] = qnan64();
    return;
  }
  a0 = __shfl_sync(0xffffffff, a0, 0);
  a1 = __shfl_sync(0xffffffff, a1, 0);
  b0 = __shfl_sync(0xffffffff, b0, 0);
  b1 = __shfl_sync(0xffffffff, b1, 0);

  int32_t alen = a1 - a0;
  int32_t blen = b1 - b0;

  
  int32_t s0, slen, l0, llen;
  if (alen <= blen) {
    s0 = a0;
    slen = alen;
    l0 = b0;
    llen = blen;
  } else {
    s0 = b0;
    slen = blen;
    l0 = a0;
    llen = alen;
  }

  
  bool cached = (llen <= MAX_SMEM_PER_WARP) && (llen >= CACHE_MIN_LEN);
  if (cached) {
    for (int i = lane; i < llen; i += WARP) {
      cache_i[i] = __ldg(indices + (l0 + i));
    }
    __syncwarp();
  }

  double dot = 0.0;
  double ns = 0.0;
  double nl = 0.0;

  int32_t search_lo = 0;

  for (int32_t j = lane; j < slen; j += WARP) {
    int32_t target = __ldg(indices + (s0 + j));
    int32_t rank = dup_rank_before(indices, s0, j, target);

    int32_t lo = search_lo;
    int32_t hi = llen;

    if (cached) {
      while (lo < hi) {
        int32_t mid = (lo + hi) >> 1;
        int32_t val = cache_i[mid];
        if (val < target)
          lo = mid + 1;
        else
          hi = mid;
      }
    } else {
      while (lo < hi) {
        int32_t mid = (lo + hi) >> 1;
        int32_t val = __ldg(indices + (l0 + mid));
        if (val < target)
          lo = mid + 1;
        else
          hi = mid;
      }
    }

    search_lo = lo;
    int32_t pos = lo + rank;

    if (pos < llen) {
      int32_t val = cached ? cache_i[pos] : __ldg(indices + (l0 + pos));
      if (val == target) {
        double ws = __ldg(edge_weights + (s0 + j));
        double wl = __ldg(edge_weights + (l0 + pos));
        dot = __fma_rn(ws, wl, dot);
        ns = __fma_rn(ws, ws, ns);
        nl = __fma_rn(wl, wl, nl);
      }
    }
  }

  
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    dot += __shfl_down_sync(0xffffffff, dot, offset);
    ns += __shfl_down_sync(0xffffffff, ns, offset);
    nl += __shfl_down_sync(0xffffffff, nl, offset);
  }

  if (lane == 0) {
    if (ns == 0.0) {
      scores[warp_id] = qnan64();
    } else {
      scores[warp_id] = dot / (sqrt(ns) * sqrt(nl));
    }
  }
}

}  

void cosine_similarity(const graph32_t& graph,
                       const double* edge_weights,
                       const int32_t* vertex_pairs_first,
                       const int32_t* vertex_pairs_second,
                       std::size_t num_pairs,
                       double* similarity_scores) {
  if (num_pairs == 0) return;

  const int32_t* offsets = graph.offsets;
  const int32_t* indices = graph.indices;

  constexpr int BLOCK = 64;
  constexpr int WARPS_PER_BLOCK = BLOCK / 32;
  int grid = (int)((num_pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

  size_t smem_bytes = (size_t)WARPS_PER_BLOCK * (size_t)MAX_SMEM_PER_WARP * sizeof(int32_t);
  cosine_similarity_warp_kernel<<<grid, BLOCK, smem_bytes>>>(
      offsets, indices, edge_weights, vertex_pairs_first, vertex_pairs_second,
      similarity_scores, (int64_t)num_pairs);
}

}  
