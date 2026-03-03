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

#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

static __device__ __forceinline__ int32_t ldg_i32(const int32_t* p) {
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}

static __device__ __forceinline__ double ldg_f64(const double* p) {
#if __CUDA_ARCH__ >= 350
  return __ldg(p);
#else
  return *p;
#endif
}

static __device__ __forceinline__ int32_t lower_bound_int(const int32_t* __restrict__ arr, int32_t lo, int32_t hi, int32_t target)
{
  while (lo < hi) {
    int32_t mid = lo + ((hi - lo) >> 1);
    int32_t v = ldg_i32(arr + mid);
    if (v < target)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}


static __device__ __forceinline__ int32_t gallop_lower_bound(const int32_t* __restrict__ arr, int32_t start, int32_t end, int32_t target)
{
  if (start >= end) return end;
  int32_t first = ldg_i32(arr + start);
  if (first >= target) return start;
  int32_t pos = start;
  int32_t step = 1;
  while (pos + step < end && ldg_i32(arr + (pos + step)) < target) {
    pos += step;
    step <<= 1;
  }
  int32_t lo = pos + 1;
  int32_t hi = pos + step + 1;
  if (hi > end) hi = end;
  
  while (lo < hi) {
    int32_t mid = lo + ((hi - lo) >> 1);
    int32_t v = ldg_i32(arr + mid);
    if (v < target)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}





__global__ void jaccard_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ scores,
    int64_t num_pairs)
{
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_pairs) return;

  int32_t u = ldg_i32(first + tid);
  int32_t v = ldg_i32(second + tid);

  if (u == v) {
    int32_t deg = ldg_i32(offsets + u + 1) - ldg_i32(offsets + u);
    scores[tid] = (deg > 0) ? 1.0 : 0.0;
    return;
  }

  int32_t u_start = ldg_i32(offsets + u);
  int32_t u_end = ldg_i32(offsets + u + 1);
  int32_t v_start = ldg_i32(offsets + v);
  int32_t v_end = ldg_i32(offsets + v + 1);
  int32_t du = u_end - u_start;
  int32_t dv = v_end - v_start;

  if (du <= 0 || dv <= 0) { scores[tid] = 0.0; return; }

  int32_t s_start, s_end, l_start, l_end;
  int32_t s_deg, l_deg;
  if (du <= dv) {
    s_start = u_start; s_end = u_end; s_deg = du;
    l_start = v_start; l_end = v_end; l_deg = dv;
  } else {
    s_start = v_start; s_end = v_end; s_deg = dv;
    l_start = u_start; l_end = u_end; l_deg = du;
  }

  
  int32_t s_first = ldg_i32(indices + s_start);
  int32_t s_last = ldg_i32(indices + (s_end - 1));
  int32_t l_first = ldg_i32(indices + l_start);
  int32_t l_last = ldg_i32(indices + (l_end - 1));
  if (s_last < l_first || l_last < s_first) { scores[tid] = 0.0; return; }

  
  int found = 0;
  {
    int32_t l_pos0 = l_start;
    const bool use_gallop0 = (l_deg > (s_deg << 4)); 
    for (int32_t sp = s_start; sp < s_end && l_pos0 < l_end; ++sp) {
      int32_t t = ldg_i32(indices + sp);
      if ((uint32_t)(t - l_first) > (uint32_t)(l_last - l_first)) continue;
      if (use_gallop0) {
        l_pos0 = gallop_lower_bound(indices, l_pos0, l_end, t);
      } else {
        l_pos0 = lower_bound_int(indices, l_pos0, l_end, t);
      }
      if (l_pos0 < l_end && ldg_i32(indices + l_pos0) == t) { found = 1; break; }
    }
  }
  if (!found) { scores[tid] = 0.0; return; }

  
  double sum_s = 0.0;
  double inter = 0.0;

  int32_t l_pos = l_start;
  const bool use_gallop = (l_deg > (s_deg << 4)); 

  
  
  for (int32_t sp = s_start; sp < s_end; ++sp) {
    int32_t t = ldg_i32(indices + sp);
    double w_s = ldg_f64(edge_weights + sp);
    sum_s += w_s;

    if (l_pos >= l_end) continue;

    
    if ((uint32_t)(t - l_first) > (uint32_t)(l_last - l_first)) continue;

    if (use_gallop) {
      l_pos = gallop_lower_bound(indices, l_pos, l_end, t);
    } else {
      l_pos = lower_bound_int(indices, l_pos, l_end, t);
    }

    if (l_pos < l_end && ldg_i32(indices + l_pos) == t) {
      double w_l = ldg_f64(edge_weights + l_pos);
      inter += fmin(w_s, w_l);
      l_pos++; 
    }
  }

  if (inter == 0.0) { scores[tid] = 0.0; return; }

  double sum_l = 0.0;
  
  for (int32_t lp = l_start; lp < l_end; ++lp) {
    sum_l += ldg_f64(edge_weights + lp);
  }

  double denom = sum_s + sum_l - inter;
  scores[tid] = (denom != 0.0) ? (inter / denom) : 0.0;
}









__global__ void jaccard_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ scores,
    int64_t num_pairs)
{
  const int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp_id >= num_pairs) return;
  const int lane = threadIdx.x & 31;
  const unsigned mask = 0xffffffffu;

  int32_t u = 0, v = 0;
  if (lane == 0) {
    u = ldg_i32(first + warp_id);
    v = ldg_i32(second + warp_id);
  }
  u = __shfl_sync(mask, u, 0);
  v = __shfl_sync(mask, v, 0);

  if (u == v) {
    if (lane == 0) {
      int32_t deg = ldg_i32(offsets + u + 1) - ldg_i32(offsets + u);
      scores[warp_id] = (deg > 0) ? 1.0 : 0.0;
    }
    return;
  }

  int32_t u_start = 0, u_end = 0, v_start = 0, v_end = 0;
  if (lane == 0) {
    u_start = ldg_i32(offsets + u);
    u_end   = ldg_i32(offsets + u + 1);
    v_start = ldg_i32(offsets + v);
    v_end   = ldg_i32(offsets + v + 1);
  }
  u_start = __shfl_sync(mask, u_start, 0);
  u_end   = __shfl_sync(mask, u_end, 0);
  v_start = __shfl_sync(mask, v_start, 0);
  v_end   = __shfl_sync(mask, v_end, 0);

  int32_t du = u_end - u_start;
  int32_t dv = v_end - v_start;
  if (du <= 0 || dv <= 0) {
    if (lane == 0) scores[warp_id] = 0.0;
    return;
  }

  int32_t s_start, s_deg, l_start, l_deg;
  if (du <= dv) {
    s_start = u_start; s_deg = du;
    l_start = v_start; l_deg = dv;
  } else {
    s_start = v_start; s_deg = dv;
    l_start = u_start; l_deg = du;
  }

  
  int32_t s_first = 0, s_last = 0, l_first = 0, l_last = 0;
  if (lane == 0) {
    s_first = ldg_i32(indices + s_start);
    s_last  = ldg_i32(indices + (s_start + s_deg - 1));
    l_first = ldg_i32(indices + l_start);
    l_last  = ldg_i32(indices + (l_start + l_deg - 1));
  }
  s_first = __shfl_sync(mask, s_first, 0);
  s_last  = __shfl_sync(mask, s_last, 0);
  l_first = __shfl_sync(mask, l_first, 0);
  l_last  = __shfl_sync(mask, l_last, 0);
  if (s_last < l_first || l_last < s_first) {
    if (lane == 0) scores[warp_id] = 0.0;
    return;
  }

  
  
  
  int found = 0;
  for (int32_t si = lane; si < s_deg; si += 32) {
    int32_t t = ldg_i32(indices + (s_start + si));
    if ((uint32_t)(t - l_first) > (uint32_t)(l_last - l_first)) continue;
    int32_t lo = 0;
    int32_t hi = l_deg;
    while (lo < hi) {
      int32_t mid = lo + ((hi - lo) >> 1);
      int32_t v_mid = ldg_i32(indices + (l_start + mid));
      if (v_mid < t) lo = mid + 1;
      else hi = mid;
    }
    if (lo < l_deg && ldg_i32(indices + (l_start + lo)) == t) { found = 1; break; }
  }
  if (!__any_sync(mask, found)) {
    if (lane == 0) scores[warp_id] = 0.0;
    return;
  }

  
  
  
  double sum_s = 0.0;
  double inter = 0.0;

  if (s_deg <= 32) {
    
    int active = (lane < s_deg);
    int32_t t = active ? ldg_i32(indices + (s_start + lane)) : (int32_t)(0x80000000u + (unsigned)lane);
    double w_s = active ? ldg_f64(edge_weights + (s_start + lane)) : 0.0;
    sum_s = w_s;

    unsigned eq = __match_any_sync(mask, t);
    unsigned lower = eq & ((lane == 0) ? 0u : ((1u << lane) - 1u));
    int dup_rank = active ? __popc(lower) : 0;

    if (active && (uint32_t)(t - l_first) <= (uint32_t)(l_last - l_first)) {
      int32_t lo = 0;
      int32_t hi = l_deg;
      while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        int32_t v_mid = ldg_i32(indices + (l_start + mid));
        if (v_mid < t) lo = mid + 1;
        else hi = mid;
      }
      int32_t lpos = lo + dup_rank;
      if (lpos < l_deg && ldg_i32(indices + (l_start + lpos)) == t) {
        double w_l = ldg_f64(edge_weights + (l_start + lpos));
        inter = fmin(w_s, w_l);
      }
    }

    
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sum_s += __shfl_down_sync(mask, sum_s, off);
      inter += __shfl_down_sync(mask, inter, off);
    }
  } else {
    
    for (int32_t si = lane; si < s_deg; si += 32) {
      int32_t pos = s_start + si;
      int32_t t = ldg_i32(indices + pos);
      double w = ldg_f64(edge_weights + pos);
      sum_s += w;

      if ((uint32_t)(t - l_first) > (uint32_t)(l_last - l_first)) continue;

      int32_t dup_rank = 0;
      if (pos > s_start && ldg_i32(indices + (pos - 1)) == t) {
        int32_t p = pos - 1;
        while (p >= s_start && ldg_i32(indices + p) == t) { dup_rank++; p--; }
      }

      int32_t lo = 0;
      int32_t hi = l_deg;
      while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        int32_t v_mid = ldg_i32(indices + (l_start + mid));
        if (v_mid < t) lo = mid + 1;
        else hi = mid;
      }
      int32_t lpos = lo + dup_rank;
      if (lpos < l_deg && ldg_i32(indices + (l_start + lpos)) == t) {
        double w_l = ldg_f64(edge_weights + (l_start + lpos));
        inter += fmin(w, w_l);
      }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      sum_s += __shfl_down_sync(mask, sum_s, off);
      inter += __shfl_down_sync(mask, inter, off);
    }
  }

  double inter0 = __shfl_sync(mask, inter, 0);
  if (inter0 == 0.0) {
    if (lane == 0) scores[warp_id] = 0.0;
    return;
  }

  
  double sum_l = 0.0;
  for (int32_t li = lane; li < l_deg; li += 32) {
    sum_l += ldg_f64(edge_weights + (l_start + li));
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sum_l += __shfl_down_sync(mask, sum_l, off);
  }

  if (lane == 0) {
    double denom = sum_s + sum_l - inter0;
    scores[warp_id] = (denom != 0.0) ? (inter0 / denom) : 0.0;
  }
}

}  

void jaccard_similarity(const graph32_t& graph,
                        const double* edge_weights,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        double* similarity_scores) {
  if (num_pairs == 0) return;

  const int32_t* offsets = graph.offsets;
  const int32_t* indices = graph.indices;

  int block = 128;
  int warps_per_block = block / 32;
  int grid = (int)(((int64_t)num_pairs + warps_per_block - 1) / warps_per_block);
  jaccard_warp_kernel<<<grid, block, 0, cudaStreamPerThread>>>(
      offsets, indices, edge_weights,
      vertex_pairs_first, vertex_pairs_second,
      similarity_scores, (int64_t)num_pairs);

  cudaStreamSynchronize(cudaStreamPerThread);
}

}  
