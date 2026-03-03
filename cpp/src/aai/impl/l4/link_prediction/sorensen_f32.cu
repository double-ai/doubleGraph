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

namespace aai {

namespace {

struct Cache : Cacheable {};

#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 8
#endif
#ifndef CACHE_EDGES
#define CACHE_EDGES 256
#endif

static __device__ __forceinline__ int lb_binary(const int32_t* __restrict__ arr, int lo, int hi, int32_t target)
{
  while (lo < hi) {
    int mid = lo + ((hi - lo) >> 1);
    int v   = arr[mid];
    if (v < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

static __device__ __forceinline__ int lb_galloping(const int32_t* __restrict__ arr,
                                                   int n,
                                                   int start,
                                                   int32_t target)
{
  if (start >= n) return n;
  int v0 = arr[start];
  if (v0 >= target) return start;

  int pos  = start;
  int step = 1;
  while (true) {
    int nxt = pos + step;
    if (nxt >= n) break;
    int vv = arr[nxt];
    if (vv < target) {
      pos = nxt;
      step <<= 1;
    } else {
      break;
    }
  }

  int lo = pos + 1;
  int hi = pos + step + 1;
  if (hi > n) hi = n;
  while (lo < hi) {
    int mid = lo + ((hi - lo) >> 1);
    int vm  = arr[mid];
    if (vm < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

__global__ void sorensen_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ out,
    int64_t num_pairs)
{
  extern __shared__ int32_t smem_i[];

  int lane          = threadIdx.x & 31;
  int warp_in_block = threadIdx.x >> 5;
  int64_t warp_id   = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp_id >= num_pairs) return;

  int32_t u = first[warp_id];
  int32_t v = second[warp_id];

  int u_start = offsets[u];
  int u_end   = offsets[u + 1];
  int v_start = offsets[v];
  int v_end   = offsets[v + 1];

  int u_deg = u_end - u_start;
  int v_deg = v_end - v_start;

  if (u_deg == 0 || v_deg == 0) {
    if (lane == 0) out[warp_id] = 0.0f;
    return;
  }

  if (u == v) {
    if (lane == 0) out[warp_id] = 1.0f;
    return;
  }

  const int32_t* short_idx;
  const float* short_wt;
  int short_deg;
  const int32_t* long_idx;
  const float* long_wt;
  int long_deg;

  if (u_deg <= v_deg) {
    short_idx = indices + u_start;
    short_wt  = edge_weights + u_start;
    short_deg = u_deg;
    long_idx  = indices + v_start;
    long_wt   = edge_weights + v_start;
    long_deg  = v_deg;
  } else {
    short_idx = indices + v_start;
    short_wt  = edge_weights + v_start;
    short_deg = v_deg;
    long_idx  = indices + u_start;
    long_wt   = edge_weights + u_start;
    long_deg  = u_deg;
  }

  int32_t long_min  = long_idx[0];
  int32_t long_max  = long_idx[long_deg - 1];
  int32_t short_min = short_idx[0];
  int32_t short_max = short_idx[short_deg - 1];
  if (short_max < long_min || long_max < short_min) {
    if (lane == 0) out[warp_id] = 0.0f;
    return;
  }

  int short_start = 0;
  if (short_min < long_min) {
    short_start = lb_binary(short_idx, 0, short_deg, long_min);
    if (short_start >= short_deg) {
      if (lane == 0) out[warp_id] = 0.0f;
      return;
    }
  }

  int search_init = 0;
  if (short_idx[short_start] > long_min) {
    search_init = lb_binary(long_idx, 0, long_deg, short_idx[short_start]);
    if (search_init >= long_deg) {
      if (lane == 0) out[warp_id] = 0.0f;
      return;
    }
  }

  float local_inter = 0.0f;

  int cache_base = warp_in_block * CACHE_EDGES;
  if (long_deg <= CACHE_EDGES) {
    for (int i = lane; i < long_deg; i += 32) {
      smem_i[cache_base + i] = long_idx[i];
    }
    __syncwarp();

    int search_lo = search_init;
    for (int i = short_start + lane; i < short_deg; i += 32) {
      int32_t key = short_idx[i];
      if (key > long_max) break;

      int rank = 0;
      int p = i - 1;
      while (p >= short_start && short_idx[p] == key) {
        ++rank;
        --p;
      }

      int lb = lb_binary(smem_i + cache_base, search_lo, long_deg, key);
      search_lo = lb;
      int pos = lb + rank;
      if (pos < long_deg && smem_i[cache_base + pos] == key) {
        float w_l = long_wt[pos];
        float w_s = short_wt[i];
        local_inter += (w_s < w_l) ? w_s : w_l;
      }
    }
  } else {
    int search_lo = search_init;
    for (int i = short_start + lane; i < short_deg; i += 32) {
      int32_t key = short_idx[i];
      if (key > long_max) break;

      int rank = 0;
      int p = i - 1;
      while (p >= short_start && short_idx[p] == key) {
        ++rank;
        --p;
      }

      int lb = lb_galloping(long_idx, long_deg, search_lo, key);
      search_lo = lb;
      int pos = lb + rank;
      if (pos < long_deg && long_idx[pos] == key) {
        float w_l = long_wt[pos];
        float w_s = short_wt[i];
        local_inter += (w_s < w_l) ? w_s : w_l;
      }
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    local_inter += __shfl_down_sync(0xffffffff, local_inter, offset);
  }
  float inter = __shfl_sync(0xffffffff, local_inter, 0);

  if (inter == 0.0f) {
    if (lane == 0) out[warp_id] = 0.0f;
    return;
  }

  float deg_u = 0.0f;
  for (int i = lane; i < u_deg; i += 32) deg_u += edge_weights[u_start + i];
  for (int offset = 16; offset > 0; offset >>= 1) deg_u += __shfl_down_sync(0xffffffff, deg_u, offset);
  deg_u = __shfl_sync(0xffffffff, deg_u, 0);

  float deg_v = 0.0f;
  for (int i = lane; i < v_deg; i += 32) deg_v += edge_weights[v_start + i];
  for (int offset = 16; offset > 0; offset >>= 1) deg_v += __shfl_down_sync(0xffffffff, deg_v, offset);
  deg_v = __shfl_sync(0xffffffff, deg_v, 0);

  float denom = deg_u + deg_v;
  float score = (denom <= 1.1754944e-38f) ? 0.0f : (2.0f * inter) / denom;
  if (lane == 0) out[warp_id] = score;
}

void launch_sorensen_warp(
    const int32_t* offsets,
    const int32_t* indices,
    const float* edge_weights,
    const int32_t* first,
    const int32_t* second,
    float* out,
    int64_t num_pairs,
    cudaStream_t stream)
{
  constexpr int block = WARPS_PER_BLOCK * 32;
  int64_t blocks = (num_pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
  if (blocks > 2147483647LL) blocks = 2147483647LL;
  size_t smem_bytes = (size_t)WARPS_PER_BLOCK * (size_t)CACHE_EDGES * sizeof(int32_t);
  sorensen_warp_kernel<<<(int)blocks, block, smem_bytes, stream>>>(
      offsets, indices, edge_weights, first, second, out, num_pairs);
}

}  

void sorensen_similarity(const graph32_t& graph,
                         const float* edge_weights,
                         const int32_t* vertex_pairs_first,
                         const int32_t* vertex_pairs_second,
                         std::size_t num_pairs,
                         float* similarity_scores) {
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);
  (void)cache;

  launch_sorensen_warp(
      graph.offsets,
      graph.indices,
      edge_weights,
      vertex_pairs_first,
      vertex_pairs_second,
      similarity_scores,
      static_cast<int64_t>(num_pairs),
      0);
}

}  
