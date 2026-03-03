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
#include <cfloat>
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {};

struct TempBuf {
  void* ptr{nullptr};
  size_t bytes{0};

  TempBuf() = default;
  explicit TempBuf(size_t bytes_) : bytes(bytes_) {
    if (bytes > 0) cudaMalloc(&ptr, bytes);
  }
  ~TempBuf() {
    if (ptr) cudaFree(ptr);
  }
  TempBuf(const TempBuf&) = delete;
  TempBuf& operator=(const TempBuf&) = delete;
  TempBuf(TempBuf&& o) noexcept : ptr(o.ptr), bytes(o.bytes) {
    o.ptr = nullptr;
    o.bytes = 0;
  }
  TempBuf& operator=(TempBuf&& o) noexcept {
    if (this != &o) {
      if (ptr) cudaFree(ptr);
      ptr = o.ptr;
      bytes = o.bytes;
      o.ptr = nullptr;
      o.bytes = 0;
    }
    return *this;
  }

  template <typename T>
  T* as() { return static_cast<T*>(ptr); }
};


__global__ void iota_int64(int64_t* out, int64_t n) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = idx;
}


__global__ void compute_weighted_degrees_kernel(const int32_t* __restrict__ offsets,
                                                const double* __restrict__ edge_weights,
                                                double* __restrict__ weighted_degrees,
                                                int32_t num_vertices)
{
  int v = (int)((int64_t)blockIdx.x * blockDim.x + threadIdx.x);
  if (v >= num_vertices) return;

  int start = offsets[v];
  int end = offsets[v + 1];

  double sum = 0.0;
  for (int i = start; i < end; i++) sum += edge_weights[i];
  weighted_degrees[v] = sum;
}


__global__ void compute_weighted_degrees_high_kernel(const int32_t* __restrict__ offsets,
                                                     const double* __restrict__ edge_weights,
                                                     double* __restrict__ weighted_degrees,
                                                     int32_t end_v)
{
  int v = (int)blockIdx.x;
  if (v >= end_v) return;
  int start = offsets[v];
  int end = offsets[v + 1];
  double sum = 0.0;
  for (int i = start + threadIdx.x; i < end; i += blockDim.x) sum += edge_weights[i];
  using BlockReduce = cub::BlockReduce<double, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;
  sum = BlockReduce(tmp).Sum(sum);
  if (threadIdx.x == 0) weighted_degrees[v] = sum;
}


__global__ void compute_weighted_degrees_mid_kernel(const int32_t* __restrict__ offsets,
                                                    const double* __restrict__ edge_weights,
                                                    double* __restrict__ weighted_degrees,
                                                    int32_t start_v,
                                                    int32_t end_v)
{
  int warp_global = ((int)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int lane = (int)(threadIdx.x & 31);
  int v = start_v + warp_global;
  bool active = (v < end_v);

  double sum = 0.0;
  if (active) {
    int start = offsets[v];
    int end = offsets[v + 1];
    for (int i = start + lane; i < end; i += 32) sum += edge_weights[i];
  }

  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
  if (lane == 0 && active) weighted_degrees[v] = sum;
}


__global__ void compute_weighted_degrees_range_kernel(const int32_t* __restrict__ offsets,
                                                      const double* __restrict__ edge_weights,
                                                      double* __restrict__ weighted_degrees,
                                                      int32_t start_v,
                                                      int32_t end_v)
{
  int v = start_v + (int)((int64_t)blockIdx.x * blockDim.x + threadIdx.x);
  if (v >= end_v) return;
  int start = offsets[v];
  int end = offsets[v + 1];
  double sum = 0.0;
  for (int i = start; i < end; i++) sum += edge_weights[i];
  weighted_degrees[v] = sum;
}


__global__ void count_twohop_pairs_kernel(const int32_t* __restrict__ offsets,
                                         const int32_t* __restrict__ indices,
                                         const int32_t* __restrict__ seeds, 
                                         int32_t num_seeds,
                                         int64_t* __restrict__ pair_counts)
{
  int seed_idx = blockIdx.x;
  if (seed_idx >= num_seeds) return;

  int u = seeds ? seeds[seed_idx] : seed_idx;
  int u_start = offsets[u];
  int u_end = offsets[u + 1];
  int u_deg = u_end - u_start;

  int64_t count = 0;
  for (int i = threadIdx.x; i < u_deg; i += blockDim.x) {
    int w = indices[u_start + i];
    count += (int64_t)(offsets[w + 1] - offsets[w]);
  }

  using BlockReduce = cub::BlockReduce<int64_t, 256>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int64_t total = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0) pair_counts[seed_idx] = total;
}


__global__ void enumerate_twohop_pairs_kernel(const int32_t* __restrict__ offsets,
                                             const int32_t* __restrict__ indices,
                                             const int32_t* __restrict__ seeds, 
                                             int32_t num_seeds,
                                             const int64_t* __restrict__ pair_offsets,
                                             int64_t* __restrict__ pair_keys) 
{
  int seed_idx = blockIdx.x;
  if (seed_idx >= num_seeds) return;

  int u = seeds ? seeds[seed_idx] : seed_idx;
  int u_start = offsets[u];
  int u_end = offsets[u + 1];
  int u_deg = u_end - u_start;
  int64_t base_offset = pair_offsets[seed_idx];

  
  int64_t my_count = 0;
  for (int i = threadIdx.x; i < u_deg; i += blockDim.x) {
    int w = indices[u_start + i];
    my_count += (int64_t)(offsets[w + 1] - offsets[w]);
  }

  using BlockScan = cub::BlockScan<int64_t, 256>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  int64_t my_off;
  BlockScan(scan_storage).ExclusiveSum(my_count, my_off);

  int64_t write_pos = base_offset + my_off;
  int64_t seed_high = ((int64_t)seed_idx) << 32;

  for (int i = threadIdx.x; i < u_deg; i += blockDim.x) {
    int w = indices[u_start + i];
    int w_start = offsets[w];
    int w_end = offsets[w + 1];
    for (int j = w_start; j < w_end; j++) {
      int v = indices[j];
      pair_keys[write_pos++] = seed_high | (int64_t)(uint32_t)v;
    }
  }
}


__global__ void mark_unique_pairs_kernel(const int64_t* __restrict__ sorted_keys,
                                        const int32_t* __restrict__ seeds, 
                                        int32_t* __restrict__ flags,
                                        int64_t num_pairs)
{
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_pairs) return;

  int64_t key = sorted_keys[idx];
  int32_t seed_idx = (int32_t)(key >> 32);
  int32_t v = (int32_t)(key & 0xFFFFFFFF);
  int32_t u = seeds ? seeds[seed_idx] : seed_idx;

  if (v == u) {
    flags[idx] = 0;
    return;
  }
  if (idx > 0 && sorted_keys[idx - 1] == key) {
    flags[idx] = 0;
    return;
  }
  flags[idx] = 1;
}



__device__ __forceinline__ double warp_intersection_weighted(const int32_t* __restrict__ a_n,
                                                             const double* __restrict__ a_w,
                                                             int a_deg,
                                                             const int32_t* __restrict__ b_n,
                                                             const double* __restrict__ b_w,
                                                             int b_deg,
                                                             int lane)
{
  if (a_deg == 0 || b_deg == 0) return 0.0;

  if (a_deg > b_deg) {
    const int32_t* tn = a_n;
    a_n = b_n;
    b_n = tn;
    const double* tw = a_w;
    a_w = b_w;
    b_w = tw;
    int td = a_deg;
    a_deg = b_deg;
    b_deg = td;
  }

  double sum = 0.0;
  for (int i = lane; i < a_deg; i += 32) {
    int32_t target = a_n[i];
    double w_a = a_w[i];

    int lo = 0, hi = b_deg;
    while (lo < hi) {
      int mid = lo + ((hi - lo) >> 1);
      if (b_n[mid] < target)
        lo = mid + 1;
      else
        hi = mid;
    }

    if (lo < b_deg && b_n[lo] == target) {
      double w_b = b_w[lo];
      sum += (w_a < w_b) ? w_a : w_b;
    }
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  return sum;
}

__global__ void compute_intersection_scores_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const double* __restrict__ weighted_degrees,
    const int32_t* __restrict__ seeds, 
    const int64_t* __restrict__ unique_pair_keys,
    int64_t num_unique_pairs,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    double* __restrict__ out_scores)
{
  int64_t warp_idx = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int lane = threadIdx.x & 31;
  if (warp_idx >= num_unique_pairs) return;

  int64_t key = unique_pair_keys[warp_idx];
  int32_t seed_idx = (int32_t)(key >> 32);
  int32_t v = (int32_t)(key & 0xFFFFFFFF);
  int32_t u = seeds ? seeds[seed_idx] : seed_idx;

  int u_start = offsets[u];
  int u_deg = offsets[u + 1] - u_start;
  int v_start = offsets[v];
  int v_deg = offsets[v + 1] - v_start;

  double inter = warp_intersection_weighted(indices + u_start,
                                            edge_weights + u_start,
                                            u_deg,
                                            indices + v_start,
                                            edge_weights + v_start,
                                            v_deg,
                                            lane);

  if (lane == 0) {
    double denom = weighted_degrees[u] + weighted_degrees[v];
    double score = (denom <= DBL_MIN) ? 0.0 : (2.0 * inter / denom);
    out_first[warp_idx] = u;
    out_second[warp_idx] = v;
    out_scores[warp_idx] = score;
  }
}

__global__ void compute_intersection_scores_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const double* __restrict__ weighted_degrees,
    const int32_t* __restrict__ seeds, 
    const int64_t* __restrict__ unique_pair_keys,
    int64_t num_unique_pairs,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    double* __restrict__ out_scores)
{
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_unique_pairs) return;

  int64_t key = unique_pair_keys[idx];
  int32_t seed_idx = (int32_t)(key >> 32);
  int32_t v = (int32_t)(key & 0xFFFFFFFF);
  int32_t u = seeds ? seeds[seed_idx] : seed_idx;

  int u_start = offsets[u];
  int u_deg = offsets[u + 1] - u_start;
  int v_start = offsets[v];
  int v_deg = offsets[v + 1] - v_start;

  const int32_t* u_nbrs = indices + u_start;
  const int32_t* v_nbrs = indices + v_start;
  const double* u_wts = edge_weights + u_start;
  const double* v_wts = edge_weights + v_start;

  double intersection_weight = 0.0;

  if (u_deg > 0 && v_deg > 0) {
    
    const int32_t* a_nbrs = u_nbrs;
    const double* a_wts = u_wts;
    int a_deg = u_deg;

    const int32_t* b_nbrs = v_nbrs;
    const double* b_wts = v_wts;
    int b_deg = v_deg;

    if (a_deg > b_deg) {
      a_nbrs = v_nbrs;
      a_wts = v_wts;
      a_deg = v_deg;
      b_nbrs = u_nbrs;
      b_wts = u_wts;
      b_deg = u_deg;
    }

    
    int j = 0;
    for (int i = 0; i < a_deg; i++) {
      int32_t target = a_nbrs[i];

      if (j < b_deg && b_nbrs[j] < target) {
        
        int step = 1;
        int pos = j;
        while (pos + step < b_deg && b_nbrs[pos + step] < target) {
          pos += step;
          step <<= 1;
        }
        
        int lo = pos;
        int hi = (pos + step < b_deg) ? (pos + step) : b_deg;
        while (lo < hi) {
          int mid = lo + ((hi - lo) >> 1);
          if (b_nbrs[mid] < target)
            lo = mid + 1;
          else
            hi = mid;
        }
        j = lo;
      }

      if (j < b_deg && b_nbrs[j] == target) {
        double wa = a_wts[i];
        double wb = b_wts[j];
        intersection_weight += (wa < wb) ? wa : wb;
        j++;
      }
    }
  }

  double denom = weighted_degrees[u] + weighted_degrees[v];
  double score = (denom <= DBL_MIN) ? 0.0 : (2.0 * intersection_weight / denom);

  out_first[idx] = u;
  out_second[idx] = v;
  out_scores[idx] = score;
}


__global__ void gather_topk_kernel(const int32_t* __restrict__ first_in,
                                  const int32_t* __restrict__ second_in,
                                  const double* __restrict__ scores_sorted,
                                  const int64_t* __restrict__ sort_indices,
                                  int32_t* __restrict__ first_out,
                                  int32_t* __restrict__ second_out,
                                  double* __restrict__ scores_out,
                                  int64_t count)
{
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  int64_t src = sort_indices[idx];
  first_out[idx] = first_in[src];
  second_out[idx] = second_in[src];
  scores_out[idx] = scores_sorted[idx];
}



__global__ void dedup_2hop_hash_kernel(const int32_t* __restrict__ offsets,
                                      const int32_t* __restrict__ indices,
                                      const int32_t* __restrict__ seeds,  
                                      int32_t num_seeds,
                                      int64_t* __restrict__ out_keys,
                                      unsigned long long* __restrict__ global_counter,
                                      unsigned long long max_output,
                                      int* __restrict__ overflow_count,
                                      int hash_capacity)
{
  extern __shared__ int32_t hash_keys[];

  int32_t seed_idx = (int32_t)blockIdx.x;
  if (seed_idx >= num_seeds) return;

  int32_t u = seeds ? seeds[seed_idx] : seed_idx;
  int32_t u_start = offsets[u];
  int32_t u_end = offsets[u + 1];

  
  for (int i = threadIdx.x; i < hash_capacity; i += blockDim.x) hash_keys[i] = -1;

  __shared__ int overflow;
  if (threadIdx.x == 0) overflow = 0;
  __syncthreads();

  int mask = hash_capacity - 1;

  for (int32_t ei = u_start; ei < u_end; ++ei) {
    if (overflow) break;
    int32_t w = indices[ei];
    int32_t w_start = offsets[w];
    int32_t w_end = offsets[w + 1];

    for (int32_t j = w_start + threadIdx.x; j < w_end; j += blockDim.x) {
      if (overflow) break;
      int32_t v = indices[j];
      if (v == u) continue;

      uint32_t slot = ((uint32_t)v * 2654435761u) & (uint32_t)mask;
      int probes = 0;
      while (true) {
        int32_t old = atomicCAS(&hash_keys[slot], -1, v);
        if (old == -1 || old == v) break;
        slot = (slot + 1) & (uint32_t)mask;
        if (++probes > 128) {
          overflow = 1;
          break;
        }
      }
    }

    __syncthreads();
  }

  if (overflow) {
    if (threadIdx.x == 0) atomicAdd(overflow_count, 1);
    return;
  }

  
  int lane = (int)(threadIdx.x & 31);
  int64_t seed_high = ((int64_t)seed_idx) << 32;

  for (int slot = threadIdx.x; slot < hash_capacity; slot += blockDim.x) {
    int32_t v = hash_keys[slot];
    bool pred = (v != -1);

    unsigned maskw = __ballot_sync(0xffffffff, pred);
    int nsel = __popc(maskw);
    if (!nsel) continue;

    int leader = __ffs(maskw) - 1;
    unsigned long long base = 0;
    if (lane == leader) {
      base = atomicAdd(global_counter, (unsigned long long)nsel);
    }
    base = __shfl_sync(0xffffffff, base, leader);

    int off = __popc(maskw & ((1u << lane) - 1u));
    unsigned long long pos = base + (unsigned long long)off;

    if (pred && pos < max_output) {
      out_keys[pos] = seed_high | (int64_t)(uint32_t)v;
    }
  }
}



__global__ void init_radix_select_kernel(uint64_t* prefix, uint64_t* k_rem, uint64_t k) {
  if (threadIdx.x == 0) {
    *prefix = 0ull;
    *k_rem = k;
  }
}

__global__ void histogram_byte_kernel(const double* __restrict__ scores,
                                      int64_t n,
                                      const uint64_t* __restrict__ prefix,
                                      int pass,
                                      uint32_t* __restrict__ hist)
{
  __shared__ unsigned int shist[256];
  for (int i = threadIdx.x; i < 256; i += blockDim.x) shist[i] = 0;
  __syncthreads();

  uint64_t pre = *prefix;
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;

  int shift_above = (pass + 1) * 8;
  int shift_byte = pass * 8;

  if (pass == 7) {
    for (int64_t i = tid; i < n; i += stride) {
      uint64_t key = (uint64_t)__double_as_longlong(scores[i]);
      unsigned int b = (unsigned int)((key >> shift_byte) & 0xFFull);
      atomicAdd(&shist[b], 1u);
    }
  } else {
    for (int64_t i = tid; i < n; i += stride) {
      uint64_t key = (uint64_t)__double_as_longlong(scores[i]);
      if ((key >> shift_above) != pre) continue;
      unsigned int b = (unsigned int)((key >> shift_byte) & 0xFFull);
      atomicAdd(&shist[b], 1u);
    }
  }

  __syncthreads();
  if (threadIdx.x < 256) {
    unsigned int v = shist[threadIdx.x];
    if (v) atomicAdd(&hist[threadIdx.x], v);
  }
}

__global__ void select_byte_kernel(const uint32_t* __restrict__ hist,
                                  uint64_t* __restrict__ prefix,
                                  uint64_t* __restrict__ k_rem)
{
  if (threadIdx.x == 0) {
    uint64_t k = *k_rem;
    uint64_t cum = 0;
    unsigned int chosen = 0;
    
    #pragma unroll
    for (int b = 255; b >= 0; --b) {
      uint64_t h = (uint64_t)hist[b];
      if (cum + h >= k) {
        chosen = (unsigned)b;
        k -= cum;
        break;
      }
      cum += h;
    }
    *prefix = (*prefix << 8) | (uint64_t)chosen;
    *k_rem = k;
  }
}

__device__ __forceinline__ int lane_id() { return (int)(threadIdx.x & 31); }

__global__ void select_greater_kernel(const double* __restrict__ scores,
                                     int64_t n,
                                     const uint64_t* __restrict__ threshold_key,
                                     int64_t* __restrict__ out_idx,
                                     int64_t* __restrict__ out_count,
                                     int64_t kmax)
{
  uint64_t thr = *threshold_key;
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  int lane = lane_id();

  for (int64_t i = tid; i < n; i += stride) {
    uint64_t key = (uint64_t)__double_as_longlong(scores[i]);
    bool pred = (key > thr);
    unsigned mask = __ballot_sync(0xffffffff, pred);
    int nsel = __popc(mask);
    if (!nsel) continue;
    int leader = __ffs(mask) - 1;
    int64_t base = 0;
    if (lane == leader) base = (int64_t)atomicAdd((unsigned long long*)out_count, (unsigned long long)nsel);
    base = __shfl_sync(0xffffffff, base, leader);
    int offset = __popc(mask & ((1u << lane) - 1u));
    int64_t pos = base + offset;
    if (pred && pos < kmax) out_idx[pos] = i;
  }
}

__global__ void select_equal_kernel(const double* __restrict__ scores,
                                   int64_t n,
                                   const uint64_t* __restrict__ threshold_key,
                                   int64_t* __restrict__ out_idx,
                                   const int64_t* __restrict__ greater_count,
                                   int64_t* __restrict__ eq_count,
                                   int64_t kmax)
{
  uint64_t thr = *threshold_key;
  int64_t base_off = *greater_count;
  if (base_off >= kmax) return;
  int64_t need = kmax - base_off;

  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  int lane = lane_id();

  for (int64_t i = tid; i < n; i += stride) {
    uint64_t key = (uint64_t)__double_as_longlong(scores[i]);
    bool pred = (key == thr);
    unsigned mask = __ballot_sync(0xffffffff, pred);
    int nsel = __popc(mask);
    if (!nsel) continue;
    int leader = __ffs(mask) - 1;
    int64_t base = 0;
    if (lane == leader) base = (int64_t)atomicAdd((unsigned long long*)eq_count, (unsigned long long)nsel);
    base = __shfl_sync(0xffffffff, base, leader);
    int offset = __popc(mask & ((1u << lane) - 1u));
    int64_t pos = base + offset;
    if (pred && pos < need) out_idx[base_off + pos] = i;
  }
}

__global__ void gather_scores_kernel(const double* __restrict__ scores,
                                    const int64_t* __restrict__ idx,
                                    double* __restrict__ out_scores,
                                    int64_t k)
{
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < k) {
    int64_t src = idx[i];
    out_scores[i] = scores[src];
  }
}



void launch_compute_weighted_degrees(const int32_t* offsets,
                                     const double* edge_weights,
                                     double* weighted_degrees,
                                     int32_t num_vertices,
                                     cudaStream_t stream)
{
  int block = 256;
  int grid = (num_vertices + block - 1) / block;
  compute_weighted_degrees_kernel<<<grid, block, 0, stream>>>(offsets, edge_weights, weighted_degrees, num_vertices);
}

void launch_compute_weighted_degrees_segmented(const int32_t* offsets,
                                               const double* edge_weights,
                                               double* weighted_degrees,
                                               int32_t seg1,
                                               int32_t seg2,
                                               int32_t num_vertices,
                                               cudaStream_t stream)
{
  
  if (seg1 > 0) {
    compute_weighted_degrees_high_kernel<<<seg1, 256, 0, stream>>>(offsets, edge_weights, weighted_degrees, seg1);
  }
  
  int32_t mid_n = seg2 - seg1;
  if (mid_n > 0) {
    constexpr int warps_per_block = 8;
    int block = warps_per_block * 32;
    int grid = (mid_n + warps_per_block - 1) / warps_per_block;
    compute_weighted_degrees_mid_kernel<<<grid, block, 0, stream>>>(offsets, edge_weights, weighted_degrees, seg1, seg2);
  }
  
  if (seg2 < num_vertices) {
    int32_t start_v = seg2;
    int32_t n = num_vertices - start_v;
    int block = 256;
    int grid = (n + block - 1) / block;
    compute_weighted_degrees_range_kernel<<<grid, block, 0, stream>>>(offsets, edge_weights, weighted_degrees, start_v, num_vertices);
  }
}

void launch_count_twohop_pairs(const int32_t* offsets,
                               const int32_t* indices,
                               const int32_t* seeds,
                               int32_t num_seeds,
                               int64_t* pair_counts,
                               cudaStream_t stream)
{
  if (num_seeds <= 0) return;
  count_twohop_pairs_kernel<<<num_seeds, 256, 0, stream>>>(offsets, indices, seeds, num_seeds, pair_counts);
}

void launch_enumerate_twohop_pairs(const int32_t* offsets,
                                  const int32_t* indices,
                                  const int32_t* seeds,
                                  int32_t num_seeds,
                                  const int64_t* pair_offsets,
                                  int64_t* pair_keys,
                                  cudaStream_t stream)
{
  if (num_seeds <= 0) return;
  enumerate_twohop_pairs_kernel<<<num_seeds, 256, 0, stream>>>(
      offsets, indices, seeds, num_seeds, pair_offsets, pair_keys);
}

void launch_mark_unique_pairs(const int64_t* sorted_keys,
                              const int32_t* seeds,
                              int32_t* flags,
                              int64_t num_pairs,
                              cudaStream_t stream)
{
  int block = 256;
  int grid = (int)((num_pairs + block - 1) / block);
  mark_unique_pairs_kernel<<<grid, block, 0, stream>>>(sorted_keys, seeds, flags, num_pairs);
}

void launch_compute_intersection_scores(const int32_t* offsets,
                                        const int32_t* indices,
                                        const double* edge_weights,
                                        const double* weighted_degrees,
                                        const int32_t* seeds,
                                        const int64_t* unique_pair_keys,
                                        int64_t num_unique_pairs,
                                        int32_t* out_first,
                                        int32_t* out_second,
                                        double* out_scores,
                                        bool use_warp,
                                        cudaStream_t stream)
{
  if (num_unique_pairs <= 0) return;
  if (use_warp) {
    constexpr int warps_per_block = 8;
    int block = warps_per_block * 32;
    int grid = (int)((num_unique_pairs + warps_per_block - 1) / warps_per_block);
    compute_intersection_scores_warp_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, edge_weights, weighted_degrees,
        seeds, unique_pair_keys, num_unique_pairs,
        out_first, out_second, out_scores);
  } else {
    int block = 256;
    int grid = (int)((num_unique_pairs + block - 1) / block);
    compute_intersection_scores_thread_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, edge_weights, weighted_degrees,
        seeds, unique_pair_keys, num_unique_pairs,
        out_first, out_second, out_scores);
  }
}

void launch_iota_int64(int64_t* out, int64_t n, cudaStream_t stream)
{
  if (n <= 0) return;
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  iota_int64<<<grid, block, 0, stream>>>(out, n);
}

void launch_dedup_2hop_hash(const int32_t* offsets,
                            const int32_t* indices,
                            const int32_t* seeds,
                            int32_t num_seeds,
                            int64_t* out_keys,
                            unsigned long long* global_counter,
                            unsigned long long max_output,
                            int* overflow_count,
                            int hash_capacity,
                            cudaStream_t stream)
{
  if (num_seeds <= 0) return;
  int block = 256;
  size_t shmem = (size_t)hash_capacity * sizeof(int32_t);
  static int last_shmem = 0;
  if ((int)shmem != last_shmem) {
    cudaFuncSetAttribute(dedup_2hop_hash_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    last_shmem = (int)shmem;
  }
  dedup_2hop_hash_kernel<<<num_seeds, block, shmem, stream>>>(
      offsets, indices, seeds, num_seeds, out_keys, global_counter, max_output, overflow_count, hash_capacity);
}

void launch_gather_topk(const int32_t* first_in,
                        const int32_t* second_in,
                        const double* scores_sorted,
                        const int64_t* sort_indices,
                        int32_t* first_out,
                        int32_t* second_out,
                        double* scores_out,
                        int64_t count,
                        cudaStream_t stream)
{
  if (count <= 0) return;
  int block = 256;
  int grid = (int)((count + block - 1) / block);
  gather_topk_kernel<<<grid, block, 0, stream>>>(
      first_in, second_in, scores_sorted, sort_indices, first_out, second_out, scores_out, count);
}

void launch_init_radix_select(uint64_t* prefix, uint64_t* k_rem, uint64_t k, cudaStream_t stream)
{
  init_radix_select_kernel<<<1, 1, 0, stream>>>(prefix, k_rem, k);
}

void launch_histogram_byte(const double* scores,
                           int64_t n,
                           const uint64_t* prefix,
                           int pass,
                           uint32_t* hist,
                           cudaStream_t stream)
{
  if (n <= 0) return;
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  if (grid > 4096) grid = 4096;
  histogram_byte_kernel<<<grid, block, 0, stream>>>(scores, n, prefix, pass, hist);
}

void launch_select_byte(const uint32_t* hist, uint64_t* prefix, uint64_t* k_rem, cudaStream_t stream)
{
  select_byte_kernel<<<1, 1, 0, stream>>>(hist, prefix, k_rem);
}

void launch_select_greater(const double* scores,
                           int64_t n,
                           const uint64_t* threshold_key,
                           int64_t* out_idx,
                           int64_t* out_count,
                           int64_t kmax,
                           cudaStream_t stream)
{
  if (n <= 0 || kmax <= 0) return;
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  if (grid > 4096) grid = 4096;
  select_greater_kernel<<<grid, block, 0, stream>>>(scores, n, threshold_key, out_idx, out_count, kmax);
}

void launch_select_equal(const double* scores,
                         int64_t n,
                         const uint64_t* threshold_key,
                         int64_t* out_idx,
                         const int64_t* greater_count,
                         int64_t* eq_count,
                         int64_t kmax,
                         cudaStream_t stream)
{
  if (n <= 0 || kmax <= 0) return;
  int block = 256;
  int grid = (int)((n + block - 1) / block);
  if (grid > 4096) grid = 4096;
  select_equal_kernel<<<grid, block, 0, stream>>>(scores, n, threshold_key, out_idx, greater_count, eq_count, kmax);
}

void launch_gather_scores(const double* scores,
                          const int64_t* idx,
                          double* out_scores,
                          int64_t k,
                          cudaStream_t stream)
{
  if (k <= 0) return;
  int block = 256;
  int grid = (int)((k + block - 1) / block);
  gather_scores_kernel<<<grid, block, 0, stream>>>(scores, idx, out_scores, k);
}


size_t get_sort_keys_temp_size(int64_t num_items)
{
  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, (int64_t*)nullptr, (int64_t*)nullptr, (int)num_items, 0, 64);
  return temp_bytes;
}

void launch_sort_keys(void* temp,
                      size_t temp_bytes,
                      const int64_t* keys_in,
                      int64_t* keys_out,
                      int64_t num_items,
                      int end_bit,
                      cudaStream_t stream)
{
  cub::DeviceRadixSort::SortKeys(temp, temp_bytes, keys_in, keys_out, (int)num_items, 0, end_bit, stream);
}

size_t get_prefix_sum_temp_size(int num_items)
{
  size_t temp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, (int64_t*)nullptr, (int64_t*)nullptr, num_items);
  return temp_bytes;
}

void launch_prefix_sum(void* temp,
                       size_t temp_bytes,
                       const int64_t* in,
                       int64_t* out,
                       int num_items,
                       cudaStream_t stream)
{
  cub::DeviceScan::ExclusiveSum(temp, temp_bytes, in, out, num_items, stream);
}

size_t get_select_flagged_temp_size(int64_t num_items)
{
  size_t temp_bytes = 0;
  cub::DeviceSelect::Flagged(nullptr,
                             temp_bytes,
                             (int64_t*)nullptr,
                             (int32_t*)nullptr,
                             (int64_t*)nullptr,
                             (int64_t*)nullptr,
                             (int)num_items);
  return temp_bytes;
}

void launch_select_flagged(void* temp,
                           size_t temp_bytes,
                           const int64_t* keys_in,
                           const int32_t* flags,
                           int64_t* keys_out,
                           int64_t* num_selected,
                           int64_t num_items,
                           cudaStream_t stream)
{
  cub::DeviceSelect::Flagged(temp, temp_bytes, keys_in, flags, keys_out, num_selected, (int)num_items, stream);
}

size_t get_sort_pairs_desc_temp_size(int64_t num_items)
{
  size_t temp_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(nullptr,
                                            temp_bytes,
                                            (double*)nullptr,
                                            (double*)nullptr,
                                            (int64_t*)nullptr,
                                            (int64_t*)nullptr,
                                            (int)num_items);
  return temp_bytes;
}

void launch_sort_pairs_desc(void* temp,
                            size_t temp_bytes,
                            const double* keys_in,
                            double* keys_out,
                            const int64_t* vals_in,
                            int64_t* vals_out,
                            int64_t num_items,
                            cudaStream_t stream)
{
  cub::DeviceRadixSort::SortPairsDescending(temp, temp_bytes, keys_in, keys_out, vals_in, vals_out, (int)num_items, 0, 64, stream);
}



similarity_result_double_t do_topk(int32_t* first,
                                   int32_t* second,
                                   double* scores,
                                   int64_t num_unique,
                                   int64_t topk_val,
                                   cudaStream_t stream)
{
  TempBuf prefix_buf(sizeof(uint64_t));
  TempBuf krem_buf(sizeof(uint64_t));
  TempBuf hist_buf(256 * sizeof(uint32_t));

  launch_init_radix_select(prefix_buf.as<uint64_t>(), krem_buf.as<uint64_t>(), (uint64_t)topk_val, stream);
  for (int pass = 7; pass >= 0; --pass) {
    cudaMemsetAsync(hist_buf.ptr, 0, 256 * sizeof(uint32_t), stream);
    launch_histogram_byte(scores, num_unique, prefix_buf.as<uint64_t>(), pass, hist_buf.as<uint32_t>(), stream);
    launch_select_byte(hist_buf.as<uint32_t>(), prefix_buf.as<uint64_t>(), krem_buf.as<uint64_t>(), stream);
  }

  TempBuf selected_idx_buf((size_t)topk_val * sizeof(int64_t));
  TempBuf gt_count_buf(sizeof(int64_t));
  TempBuf eq_count_buf(sizeof(int64_t));
  cudaMemsetAsync(gt_count_buf.ptr, 0, sizeof(int64_t), stream);
  launch_select_greater(scores, num_unique, prefix_buf.as<uint64_t>(),
                        selected_idx_buf.as<int64_t>(), gt_count_buf.as<int64_t>(), topk_val, stream);
  cudaMemsetAsync(eq_count_buf.ptr, 0, sizeof(int64_t), stream);
  launch_select_equal(scores, num_unique, prefix_buf.as<uint64_t>(),
                      selected_idx_buf.as<int64_t>(), gt_count_buf.as<int64_t>(),
                      eq_count_buf.as<int64_t>(), topk_val, stream);

  TempBuf selected_scores_buf((size_t)topk_val * sizeof(double));
  TempBuf scores_sorted_buf((size_t)topk_val * sizeof(double));
  TempBuf idx_sorted_buf((size_t)topk_val * sizeof(int64_t));

  launch_gather_scores(scores, selected_idx_buf.as<int64_t>(), selected_scores_buf.as<double>(), topk_val, stream);
  {
    size_t tb = get_sort_pairs_desc_temp_size(topk_val);
    TempBuf temp(tb);
    launch_sort_pairs_desc(temp.ptr, tb,
                           selected_scores_buf.as<double>(), scores_sorted_buf.as<double>(),
                           selected_idx_buf.as<int64_t>(), idx_sorted_buf.as<int64_t>(),
                           topk_val, stream);
  }

  int32_t* tf = nullptr;
  int32_t* ts = nullptr;
  double* tsc = nullptr;
  cudaMalloc(&tf, topk_val * sizeof(int32_t));
  cudaMalloc(&ts, topk_val * sizeof(int32_t));
  cudaMalloc(&tsc, topk_val * sizeof(double));

  launch_gather_topk(first, second,
                     scores_sorted_buf.as<double>(), idx_sorted_buf.as<int64_t>(),
                     tf, ts, tsc, topk_val, stream);

  
  cudaFree(first);
  cudaFree(second);
  cudaFree(scores);

  return {tf, ts, tsc, (std::size_t)topk_val};
}

}  



similarity_result_double_t sorensen_all_pairs_similarity_seg(
    const graph32_t& graph,
    const double* edge_weights,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk)
{
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);
  (void)cache;

  cudaStream_t stream = 0;

  const int32_t n_verts = graph.number_of_vertices;
  const int32_t n_edges = graph.number_of_edges;
  const int32_t* d_offsets = graph.offsets;
  const int32_t* d_indices = graph.indices;
  const double* d_edge_weights = edge_weights;

  const int32_t* d_seeds = vertices;
  const int32_t num_seeds = vertices ? (int32_t)num_vertices : n_verts;

  if (n_verts <= 0 || num_seeds <= 0 || n_edges <= 0) {
    return {nullptr, nullptr, nullptr, 0};
  }

  
  const auto& seg = graph.segment_offsets.value();
  int32_t seg1 = seg[1];
  int32_t seg2 = seg[2];

  
  TempBuf wdeg_buf((size_t)n_verts * sizeof(double));
  launch_compute_weighted_degrees_segmented(d_offsets, d_edge_weights, wdeg_buf.as<double>(),
                                            seg1, seg2, n_verts, stream);

  
  
  if (num_seeds <= 512) {
    constexpr int HASH_CAP = 16384;
    unsigned long long max_out = (unsigned long long)num_seeds * (unsigned long long)HASH_CAP;

    TempBuf keys_hash((size_t)max_out * sizeof(int64_t));
    TempBuf counter(sizeof(unsigned long long));
    TempBuf overflow_buf(sizeof(int));

    cudaMemsetAsync(counter.ptr, 0, sizeof(unsigned long long), stream);
    cudaMemsetAsync(overflow_buf.ptr, 0, sizeof(int), stream);

    launch_dedup_2hop_hash(d_offsets, d_indices, d_seeds, num_seeds,
                           keys_hash.as<int64_t>(),
                           (unsigned long long*)counter.ptr,
                           max_out,
                           overflow_buf.as<int>(),
                           HASH_CAP, stream);

    int h_over = 0;
    unsigned long long h_count = 0;
    cudaMemcpyAsync(&h_over, overflow_buf.as<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_count, (unsigned long long*)counter.ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (h_over == 0 && h_count > 0) {
      int64_t num_unique = (int64_t)h_count;

      int32_t* first = nullptr;
      int32_t* second = nullptr;
      double* scores = nullptr;
      cudaMalloc(&first, num_unique * sizeof(int32_t));
      cudaMalloc(&second, num_unique * sizeof(int32_t));
      cudaMalloc(&scores, num_unique * sizeof(double));

      bool use_warp = (num_unique < 500000);
      launch_compute_intersection_scores(d_offsets, d_indices, d_edge_weights,
                                         wdeg_buf.as<double>(), d_seeds,
                                         keys_hash.as<int64_t>(), num_unique,
                                         first, second, scores,
                                         use_warp, stream);

      
      if (topk.has_value() && num_unique > (int64_t)topk.value()) {
        return do_topk(first, second, scores, num_unique, (int64_t)topk.value(), stream);
      }

      return {first, second, scores, (std::size_t)num_unique};
    }
  }

  
  TempBuf counts_buf((size_t)num_seeds * sizeof(int64_t));
  launch_count_twohop_pairs(d_offsets, d_indices, d_seeds, num_seeds,
                            counts_buf.as<int64_t>(), stream);

  
  TempBuf pfx_buf((size_t)num_seeds * sizeof(int64_t));
  {
    size_t tb = get_prefix_sum_temp_size(num_seeds);
    TempBuf temp(tb);
    launch_prefix_sum(temp.ptr, tb, counts_buf.as<int64_t>(), pfx_buf.as<int64_t>(),
                      num_seeds, stream);
  }

  
  int64_t total_pairs = 0;
  {
    int64_t last_off = 0;
    int64_t last_cnt = 0;
    cudaMemcpyAsync(&last_off, pfx_buf.as<int64_t>() + (num_seeds - 1),
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&last_cnt, counts_buf.as<int64_t>() + (num_seeds - 1),
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    total_pairs = last_off + last_cnt;
  }

  if (total_pairs <= 0) {
    return {nullptr, nullptr, nullptr, 0};
  }

  
  TempBuf keys_buf((size_t)total_pairs * sizeof(int64_t));
  launch_enumerate_twohop_pairs(d_offsets, d_indices, d_seeds, num_seeds,
                                pfx_buf.as<int64_t>(), keys_buf.as<int64_t>(), stream);

  
  TempBuf sorted_keys_buf((size_t)total_pairs * sizeof(int64_t));
  {
    int s_bits = 1;
    for (int tmp = num_seeds - 1; tmp > 0; tmp >>= 1) ++s_bits;
    int end_bit = 32 + s_bits;
    if (end_bit > 64) end_bit = 64;

    size_t tb = get_sort_keys_temp_size(total_pairs);
    TempBuf temp(tb);
    launch_sort_keys(temp.ptr, tb, keys_buf.as<int64_t>(), sorted_keys_buf.as<int64_t>(),
                     total_pairs, end_bit, stream);
  }

  
  TempBuf flags_buf((size_t)total_pairs * sizeof(int32_t));
  launch_mark_unique_pairs(sorted_keys_buf.as<int64_t>(), d_seeds,
                           flags_buf.as<int32_t>(), total_pairs, stream);

  TempBuf nsel_buf(sizeof(int64_t));
  {
    size_t tb = get_select_flagged_temp_size(total_pairs);
    TempBuf temp(tb);
    
    launch_select_flagged(temp.ptr, tb,
                          sorted_keys_buf.as<int64_t>(),
                          flags_buf.as<int32_t>(),
                          keys_buf.as<int64_t>(),
                          nsel_buf.as<int64_t>(),
                          total_pairs, stream);
  }

  int64_t num_unique = 0;
  cudaMemcpyAsync(&num_unique, nsel_buf.as<int64_t>(), sizeof(int64_t),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  if (num_unique <= 0) {
    return {nullptr, nullptr, nullptr, 0};
  }

  
  int32_t* first = nullptr;
  int32_t* second = nullptr;
  double* scores = nullptr;
  cudaMalloc(&first, num_unique * sizeof(int32_t));
  cudaMalloc(&second, num_unique * sizeof(int32_t));
  cudaMalloc(&scores, num_unique * sizeof(double));

  bool use_warp = (num_unique < 500000);

  launch_compute_intersection_scores(d_offsets, d_indices, d_edge_weights,
                                     wdeg_buf.as<double>(), d_seeds,
                                     keys_buf.as<int64_t>(), num_unique,
                                     first, second, scores,
                                     use_warp, stream);

  
  if (topk.has_value() && num_unique > (int64_t)topk.value()) {
    return do_topk(first, second, scores, num_unique, (int64_t)topk.value(), stream);
  }

  return {first, second, scores, (std::size_t)num_unique};
}

}  
