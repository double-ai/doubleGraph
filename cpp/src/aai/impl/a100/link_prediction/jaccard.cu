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
#include <cuda_pipeline_primitives.h>
#include <cstdint>

namespace aai {

namespace {








#ifndef GROUP_SIZE
#define GROUP_SIZE 16
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#ifndef SMEM_B_MAX
#define SMEM_B_MAX 128
#endif

static_assert((GROUP_SIZE & (GROUP_SIZE - 1)) == 0, "GROUP_SIZE must be power of 2");

__device__ __forceinline__ int lb_smem_128(const int* arr, int size, int target)
{
  
  int pos = 0;
  if (pos + 64 < size && arr[pos + 64] < target) pos += 64;
  if (pos + 32 < size && arr[pos + 32] < target) pos += 32;
  if (pos + 16 < size && arr[pos + 16] < target) pos += 16;
  if (pos + 8 < size && arr[pos + 8] < target) pos += 8;
  if (pos + 4 < size && arr[pos + 4] < target) pos += 4;
  if (pos + 2 < size && arr[pos + 2] < target) pos += 2;
  if (pos + 1 < size && arr[pos + 1] < target) pos += 1;
  return pos + (pos < size && arr[pos] < target);
}

__device__ __forceinline__ int lb_global_branchless(const int* __restrict__ arr, int size, int target)
{
  
  const int* base = arr;
  int n = size;
  while (n > 1) {
    int half = n >> 1;
    int v = __ldg(base + half);
    base += (v < target) ? half : 0;
    n -= half;
  }
  int v0 = __ldg(base);
  return (int)(base - arr) + (n > 0 && v0 < target);
}

__device__ __forceinline__ int upper_bound_smem_128(const int* arr, int size, int target)
{
  
  
  return lb_smem_128(arr, size, target + 1);
}

__device__ __forceinline__ int upper_bound_global(const int* __restrict__ arr, int size, int target)
{
  return lb_global_branchless(arr, size, target + 1);
}

__global__ __launch_bounds__(BLOCK_SIZE, 8)
void jaccard_groups_simple(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs)
{
  constexpr int GROUPS_PER_BLOCK = BLOCK_SIZE / GROUP_SIZE;
  const int lane = threadIdx.x & (GROUP_SIZE - 1);
  const int group_in_block = threadIdx.x / GROUP_SIZE;
  const int64_t pair_id = (int64_t)blockIdx.x * GROUPS_PER_BLOCK + group_in_block;
  if (pair_id >= num_pairs) return;

  
  extern __shared__ int smem[];
  int* sb = smem + group_in_block * SMEM_B_MAX;

  
  const int warp_lane = threadIdx.x & 31;
  const int group_base_in_warp = (warp_lane / GROUP_SIZE) * GROUP_SIZE;
  const unsigned mask = ((1u << GROUP_SIZE) - 1u) << group_base_in_warp;

  int u = __ldg(&first[pair_id]);
  int v = __ldg(&second[pair_id]);

  int u_start = __ldg(&offsets[u]);
  int u_end = __ldg(&offsets[u + 1]);
  int v_start = __ldg(&offsets[v]);
  int v_end = __ldg(&offsets[v + 1]);

  int du = u_end - u_start;
  int dv = v_end - v_start;
  if (du == 0 || dv == 0) {
    if (lane == 0) scores[pair_id] = 0.0f;
    return;
  }

  const int* a = (du <= dv) ? (indices + u_start) : (indices + v_start);
  const int* b = (du <= dv) ? (indices + v_start) : (indices + u_start);
  int na = (du <= dv) ? du : dv;
  int nb = (du <= dv) ? dv : du;

  
  int a0 = __ldg(a);
  int a1 = __ldg(a + na - 1);
  int b0 = __ldg(b);
  int b1 = __ldg(b + nb - 1);
  if (a1 < b0 || b1 < a0) {
    if (lane == 0) scores[pair_id] = 0.0f;
    return;
  }

  
  int lo_a = 0;
  if (a0 < b0 && na > 32) lo_a = lb_global_branchless(a, na, b0);
  int hi_a = na;
  if (a1 > b1 && na > 32) hi_a = upper_bound_global(a, na, b1);

  
  int i0 = lo_a;
  int i = i0 + ((lane - i0) & (GROUP_SIZE - 1));

  int count = 0;

  if (nb <= SMEM_B_MAX) {
    
    for (int j = lane; j < nb; j += GROUP_SIZE) {
      __pipeline_memcpy_async(sb + j, b + j, sizeof(int));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncwarp(mask);

    
    for (; i < hi_a; i += GROUP_SIZE) {
      int val = __ldg(a + i);
      
      int pos = lb_smem_128(sb, nb, val);
      count += (pos < nb && sb[pos] == val);
    }
  } else {
    
    int search_lo = 0;
    
    if (i < hi_a) {
      int first_val = __ldg(a + i);
      if (b0 < first_val) search_lo = lb_global_branchless(b, nb, first_val);
    }
    for (; i < hi_a; i += GROUP_SIZE) {
      int val = __ldg(a + i);
      int rem = nb - search_lo;
      int pos = search_lo + lb_global_branchless(b + search_lo, rem, val);
      count += (pos < nb && __ldg(b + pos) == val);
      search_lo = pos;
    }
  }

  
#pragma unroll
  for (int offset = GROUP_SIZE >> 1; offset > 0; offset >>= 1) {
    count += __shfl_down_sync(mask, count, offset);
  }

  if (lane == 0) {
    int inter = count;
    int uni = du + dv - inter;
    scores[pair_id] = (uni > 0) ? __int2float_rn(inter) / __int2float_rn(uni) : 0.0f;
  }
}

__global__ __launch_bounds__(BLOCK_SIZE, 8)
void jaccard_groups_multi(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs)
{
  constexpr int GROUPS_PER_BLOCK = BLOCK_SIZE / GROUP_SIZE;
  const int lane = threadIdx.x & (GROUP_SIZE - 1);
  const int group_in_block = threadIdx.x / GROUP_SIZE;
  const int64_t pair_id = (int64_t)blockIdx.x * GROUPS_PER_BLOCK + group_in_block;
  if (pair_id >= num_pairs) return;

  extern __shared__ int smem[];
  int* sb = smem + group_in_block * SMEM_B_MAX;

  const int warp_lane = threadIdx.x & 31;
  const int group_base_in_warp = (warp_lane / GROUP_SIZE) * GROUP_SIZE;
  const unsigned mask = ((1u << GROUP_SIZE) - 1u) << group_base_in_warp;

  int u = __ldg(&first[pair_id]);
  int v = __ldg(&second[pair_id]);

  int u_start = __ldg(&offsets[u]);
  int u_end = __ldg(&offsets[u + 1]);
  int v_start = __ldg(&offsets[v]);
  int v_end = __ldg(&offsets[v + 1]);

  int du = u_end - u_start;
  int dv = v_end - v_start;
  if (du == 0 || dv == 0) {
    if (lane == 0) scores[pair_id] = 0.0f;
    return;
  }

  const int* a = (du <= dv) ? (indices + u_start) : (indices + v_start);
  const int* b = (du <= dv) ? (indices + v_start) : (indices + u_start);
  int na = (du <= dv) ? du : dv;
  int nb = (du <= dv) ? dv : du;

  int a0 = __ldg(a);
  int a1 = __ldg(a + na - 1);
  int b0 = __ldg(b);
  int b1 = __ldg(b + nb - 1);
  if (a1 < b0 || b1 < a0) {
    if (lane == 0) scores[pair_id] = 0.0f;
    return;
  }

  int lo_a = 0;
  if (a0 < b0 && na > 32) lo_a = lb_global_branchless(a, na, b0);
  int hi_a = na;
  if (a1 > b1 && na > 32) hi_a = upper_bound_global(a, na, b1);

  int i0 = lo_a;
  int i = i0 + ((lane - i0) & (GROUP_SIZE - 1));

  int count = 0;

  if (nb <= SMEM_B_MAX) {
    for (int j = lane; j < nb; j += GROUP_SIZE) {
      __pipeline_memcpy_async(sb + j, b + j, sizeof(int));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncwarp(mask);

    for (; i < hi_a; i += GROUP_SIZE) {
      int val = __ldg(a + i);
      int pos = lb_smem_128(sb, nb, val);
      if (pos < nb && sb[pos] == val) {
        
        int prev = (i > 0) ? __ldg(a + i - 1) : (val - 1);
        if (i == 0 || prev != val) {
          count++;
        } else {
          int lb_a = lb_global_branchless(a, i, val);
          int rank = i - lb_a;
          int ub_b = upper_bound_smem_128(sb + pos, nb - pos, val) + pos;
          int mult_b = ub_b - pos;
          if (rank < mult_b) count++;
        }
      }
    }
  } else {
    int search_lo = 0;
    if (i < hi_a) {
      int first_val = __ldg(a + i);
      if (b0 < first_val) search_lo = lb_global_branchless(b, nb, first_val);
    }

    for (; i < hi_a; i += GROUP_SIZE) {
      int val = __ldg(a + i);
      int rem = nb - search_lo;
      int pos = search_lo + lb_global_branchless(b + search_lo, rem, val);
      if (pos < nb && __ldg(b + pos) == val) {
        int prev = (i > 0) ? __ldg(a + i - 1) : (val - 1);
        if (i == 0 || prev != val) {
          count++;
        } else {
          int lb_a = lb_global_branchless(a, i, val);
          int rank = i - lb_a;
          int ub_b = pos + upper_bound_global(b + pos, nb - pos, val);
          int mult_b = ub_b - pos;
          if (rank < mult_b) count++;
        }
      }
      search_lo = pos;
    }
  }

#pragma unroll
  for (int offset = GROUP_SIZE >> 1; offset > 0; offset >>= 1) {
    count += __shfl_down_sync(mask, count, offset);
  }

  if (lane == 0) {
    int inter = count;
    int uni = du + dv - inter;
    scores[pair_id] = (uni > 0) ? __int2float_rn(inter) / __int2float_rn(uni) : 0.0f;
  }
}

}  

void jaccard_similarity(const graph32_t& graph,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        float* similarity_scores) {
  if (num_pairs <= 0) return;

  constexpr int GROUPS_PER_BLOCK = BLOCK_SIZE / GROUP_SIZE;
  int64_t grid = ((int64_t)num_pairs + GROUPS_PER_BLOCK - 1) / GROUPS_PER_BLOCK;
  size_t smem_bytes = (size_t)GROUPS_PER_BLOCK * SMEM_B_MAX * sizeof(int);

  if (graph.is_multigraph) {
    jaccard_groups_multi<<<(int)grid, BLOCK_SIZE, smem_bytes>>>(
        graph.offsets, graph.indices,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, (int64_t)num_pairs);
  } else {
    jaccard_groups_simple<<<(int)grid, BLOCK_SIZE, smem_bytes>>>(
        graph.offsets, graph.indices,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, (int64_t)num_pairs);
  }
}

}  
