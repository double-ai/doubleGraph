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



struct Cache : Cacheable {
  uint8_t* scratch = nullptr;
  int64_t scratch_capacity = 0;

  uint8_t* ensure_scratch(size_t bytes) {
    if (scratch_capacity < (int64_t)bytes) {
      if (scratch) cudaFree(scratch);
      scratch_capacity = (int64_t)bytes + 256;
      cudaMalloc(&scratch, scratch_capacity);
    }
    return scratch;
  }

  ~Cache() override {
    if (scratch) cudaFree(scratch);
  }
};



static inline uint32_t ceil_log2_u32(uint32_t x)
{
  uint32_t r = 0;
  uint32_t v = x > 1 ? x - 1 : 0;
  while (v > 0) {
    v >>= 1;
    r++;
  }
  return r;
}

__global__ void iota_kernel(int32_t* out, int32_t n)
{
  int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) out[i] = i;
}


__global__ void compute_sizes_kernel(const int32_t* __restrict__ offsets,
                                     const int32_t* __restrict__ indices,
                                     const int32_t* __restrict__ seeds,
                                     bool seeds_is_iota,
                                     int32_t num_seeds,
                                     int64_t* __restrict__ sizes)
{
  int32_t warp = static_cast<int32_t>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  if (warp >= num_seeds) return;
  int32_t u = seeds_is_iota ? warp : __ldg(seeds + warp);
  int32_t s = __ldg(offsets + u);
  int32_t e = __ldg(offsets + u + 1);
  int32_t deg = e - s;
  int64_t total = 0;
  for (int32_t i = lane; i < deg; i += 32) {
    int32_t w = __ldg(indices + s + i);
    int32_t ws = __ldg(offsets + w);
    int32_t we = __ldg(offsets + w + 1);
    total += (int64_t)(we - ws);
  }
  
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    total += __shfl_down_sync(0xffffffff, total, off);
  }
  if (lane == 0) sizes[warp] = total;
}

__global__ void compute_total_pairs_kernel(const int64_t* __restrict__ sizes,
                                           const int64_t* __restrict__ offsets,
                                           int32_t n,
                                           int64_t* __restrict__ out_total)
{
  if (n <= 0) {
    if (threadIdx.x == 0) *out_total = 0;
    return;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    int64_t last = offsets[n - 1];
    int64_t sz = sizes[n - 1];
    *out_total = last + sz;
  }
}



__global__ void fill_keys_kernel(const int32_t* __restrict__ offsets,
                                 const int32_t* __restrict__ indices,
                                 const int32_t* __restrict__ seeds,
                                 bool seeds_is_iota,
                                 int32_t num_seeds,
                                 const int64_t* __restrict__ write_offsets,
                                 int64_t* __restrict__ keys,
                                 int64_t stride)
{
  int32_t sid = static_cast<int32_t>(blockIdx.x);
  if (sid >= num_seeds) return;
  int32_t u = seeds_is_iota ? sid : __ldg(seeds + sid);
  int32_t u_s = __ldg(offsets + u);
  int32_t u_e = __ldg(offsets + u + 1);
  int64_t base = __ldg(write_offsets + sid);
  int64_t pos = 0;
  uint64_t sid_stride = (uint64_t)(uint32_t)sid * (uint64_t)stride;
  for (int32_t ni = u_s; ni < u_e; ++ni) {
    int32_t w = __ldg(indices + ni);
    int32_t w_s = __ldg(offsets + w);
    int32_t w_e = __ldg(offsets + w + 1);
    int32_t deg_w = w_e - w_s;
    for (int32_t j = static_cast<int32_t>(threadIdx.x); j < deg_w; j += static_cast<int32_t>(blockDim.x)) {
      int32_t v = __ldg(indices + w_s + j);
      keys[base + pos + j] = (int64_t)(sid_stride + (uint64_t)(uint32_t)v);
    }
    pos += deg_w;
  }
}




__global__ void fill_keys_u32_kernel(const int32_t* __restrict__ offsets,
                                     const int32_t* __restrict__ indices,
                                     const int32_t* __restrict__ seeds,
                                     bool seeds_is_iota,
                                     int32_t num_seeds,
                                     const int64_t* __restrict__ write_offsets,
                                     uint32_t* __restrict__ keys,
                                     uint32_t stride_bits)
{
  int32_t sid = static_cast<int32_t>(blockIdx.x);
  if (sid >= num_seeds) return;
  int32_t u = seeds_is_iota ? sid : __ldg(seeds + sid);
  int32_t u_s = __ldg(offsets + u);
  int32_t u_e = __ldg(offsets + u + 1);
  int64_t base = __ldg(write_offsets + sid);
  int64_t pos = 0;
  uint32_t sid_part = static_cast<uint32_t>(sid) << stride_bits;
  for (int32_t ni = u_s; ni < u_e; ++ni) {
    int32_t w = __ldg(indices + ni);
    int32_t w_s = __ldg(offsets + w);
    int32_t w_e = __ldg(offsets + w + 1);
    int32_t deg_w = w_e - w_s;
    for (int32_t j = static_cast<int32_t>(threadIdx.x); j < deg_w; j += static_cast<int32_t>(blockDim.x)) {
      int32_t v = __ldg(indices + w_s + j);
      keys[base + pos + j] = sid_part | static_cast<uint32_t>(v);
    }
    pos += deg_w;
  }
}



__global__ void make_valid_flags_kernel(const int64_t* __restrict__ unique_keys,
                                        int32_t num_runs,
                                        const int32_t* __restrict__ seeds,
                                        bool seeds_is_iota,
                                        int64_t stride,
                                        uint32_t stride_bits,
                                        int32_t* __restrict__ flags)
{
  int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= num_runs) return;
  uint64_t key = static_cast<uint64_t>(unique_keys[i]);
  uint64_t stride_u = static_cast<uint64_t>(stride);
  uint32_t sid = static_cast<uint32_t>(key >> stride_bits);
  uint32_t v = static_cast<uint32_t>(key & (stride_u - 1));
  int32_t u = seeds_is_iota ? static_cast<int32_t>(sid) : __ldg(seeds + sid);
  flags[i] = (static_cast<int32_t>(v) != u) ? 1 : 0;
}

__global__ void make_valid_flags_u32_kernel(const uint32_t* __restrict__ unique_keys,
                                            int32_t num_runs,
                                            const int32_t* __restrict__ seeds,
                                            bool seeds_is_iota,
                                            uint32_t stride_bits,
                                            int32_t* __restrict__ flags)
{
  int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= num_runs) return;
  uint32_t key = __ldg(unique_keys + i);
  uint32_t sid = key >> stride_bits;
  uint32_t mask = (1u << stride_bits) - 1u;
  uint32_t v = key & mask;
  int32_t u = seeds_is_iota ? static_cast<int32_t>(sid) : __ldg(seeds + sid);
  flags[i] = (static_cast<int32_t>(v) != u) ? 1 : 0;
}



__device__ __forceinline__ int lower_bound_d(const int32_t* arr, int size, int target)
{
  int lo = 0, hi = size;
  while (lo < hi) {
    int mid = lo + ((hi - lo) >> 1);
    int v = __ldg(arr + mid);
    if (v < target)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

__device__ __forceinline__ int intersect_merge(const int32_t* __restrict__ a, int sa,
                                               const int32_t* __restrict__ b, int sb)
{
  if (sa == 0 || sb == 0) return 0;
  int ia = lower_bound_d(a, sa, __ldg(b));
  if (ia >= sa) return 0;
  int ib = lower_bound_d(b, sb, __ldg(a + ia));
  if (ib >= sb) return 0;
  int count = 0;
  while (ia < sa && ib < sb) {
    int va = __ldg(a + ia);
    int vb = __ldg(b + ib);
    if (va == vb) {
      count++;
      ia++;
      ib++;
    } else if (va < vb) {
      ia++;
    } else {
      ib++;
    }
  }
  return count;
}

__device__ __forceinline__ int intersect_gallop(const int32_t* __restrict__ sm, int ssm,
                                                const int32_t* __restrict__ lg, int slg)
{
  int count = 0;
  int j = 0;
  for (int i = 0; i < ssm && j < slg; ++i) {
    int target = __ldg(sm + i);
    if (__ldg(lg + j) < target) {
      int step = 1;
      int pos = j;
      while (pos + step < slg && __ldg(lg + pos + step) < target) {
        pos += step;
        step <<= 1;
      }
      int lo = pos;
      int hi = (pos + step < slg) ? pos + step + 1 : slg;
      while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (__ldg(lg + mid) < target)
          lo = mid + 1;
        else
          hi = mid;
      }
      j = lo;
    }
    if (j < slg && __ldg(lg + j) == target) {
      count++;
      j++;
    }
  }
  return count;
}

__device__ __forceinline__ int intersect_opt(const int32_t* a, int sa, const int32_t* b, int sb)
{
  if (sa == 0 || sb == 0) return 0;
  const int32_t *sm, *lg;
  int ssm, slg;
  if (sa <= sb) {
    sm = a;
    ssm = sa;
    lg = b;
    slg = sb;
  } else {
    sm = b;
    ssm = sb;
    lg = a;
    slg = sa;
  }
  if (slg > 10 * ssm) {
    return intersect_gallop(sm, ssm, lg, slg);
  }
  return intersect_merge(a, sa, b, sb);
}



__global__ void score_scatter_kernel(const int64_t* __restrict__ unique_keys,
                                     const int32_t* __restrict__ counts,
                                     int32_t num_runs,
                                     const int32_t* __restrict__ offsets,
                                     const int32_t* __restrict__ indices,
                                     const int32_t* __restrict__ seeds,
                                     bool seeds_is_iota,
                                     int64_t stride,
                                     uint32_t stride_bits,
                                     const int32_t* __restrict__ scan,
                                     int32_t* __restrict__ out_first,
                                     int32_t* __restrict__ out_second,
                                     float* __restrict__ out_scores,
                                     bool is_multigraph)
{
  int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= num_runs) return;
  uint64_t key = static_cast<uint64_t>(unique_keys[i]);
  uint64_t stride_u = static_cast<uint64_t>(stride);
  uint32_t sid = static_cast<uint32_t>(key >> stride_bits);
  uint32_t v_u32 = static_cast<uint32_t>(key & (stride_u - 1));
  int32_t v = static_cast<int32_t>(v_u32);
  int32_t u = seeds_is_iota ? static_cast<int32_t>(sid) : __ldg(seeds + sid);
  if (v == u) return;
  int32_t pos = scan[i];

  int32_t u_s = __ldg(offsets + u);
  int32_t u_e = __ldg(offsets + u + 1);
  int32_t v_s = __ldg(offsets + v);
  int32_t v_e = __ldg(offsets + v + 1);
  int32_t deg_u = u_e - u_s;
  int32_t deg_v = v_e - v_s;
  int32_t inter = is_multigraph ? intersect_opt(indices + u_s, deg_u, indices + v_s, deg_v) : counts[i];
  float denom = static_cast<float>(deg_u + deg_v);
  float score = (denom > 0.0f) ? (2.0f * static_cast<float>(inter) / denom) : 0.0f;
  out_first[pos] = u;
  out_second[pos] = v;
  out_scores[pos] = score;
}

__global__ void score_scatter_u32_kernel(const uint32_t* __restrict__ unique_keys,
                                         const int32_t* __restrict__ counts,
                                         int32_t num_runs,
                                         const int32_t* __restrict__ offsets,
                                         const int32_t* __restrict__ indices,
                                         const int32_t* __restrict__ seeds,
                                         bool seeds_is_iota,
                                         uint32_t stride_bits,
                                         const int32_t* __restrict__ scan,
                                         int32_t* __restrict__ out_first,
                                         int32_t* __restrict__ out_second,
                                         float* __restrict__ out_scores,
                                         bool is_multigraph)
{
  int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= num_runs) return;
  uint32_t key = __ldg(unique_keys + i);
  uint32_t sid = key >> stride_bits;
  uint32_t mask = (1u << stride_bits) - 1u;
  uint32_t v_u32 = key & mask;
  int32_t v = static_cast<int32_t>(v_u32);
  int32_t u = seeds_is_iota ? static_cast<int32_t>(sid) : __ldg(seeds + sid);
  if (v == u) return;
  int32_t pos = scan[i];

  int32_t u_s = __ldg(offsets + u);
  int32_t u_e = __ldg(offsets + u + 1);
  int32_t v_s = __ldg(offsets + v);
  int32_t v_e = __ldg(offsets + v + 1);
  int32_t deg_u = u_e - u_s;
  int32_t deg_v = v_e - v_s;
  int32_t inter = is_multigraph ? intersect_opt(indices + u_s, deg_u, indices + v_s, deg_v) : counts[i];
  float denom = static_cast<float>(deg_u + deg_v);
  float score = (denom > 0.0f) ? (2.0f * static_cast<float>(inter) / denom) : 0.0f;
  out_first[pos] = u;
  out_second[pos] = v;
  out_scores[pos] = score;
}

__global__ void gather_i32_kernel(const int32_t* __restrict__ in,
                                  const int32_t* __restrict__ idx,
                                  int32_t* __restrict__ out,
                                  int32_t n)
{
  int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) out[i] = __ldg(in + __ldg(idx + i));
}

}  

similarity_result_float_t sorensen_all_pairs_similarity_seg(const graph32_t& graph,
                                                            const int32_t* vertices,
                                                            std::size_t num_vertices,
                                                            std::optional<std::size_t> topk) {
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);

  const int32_t* d_offsets = graph.offsets;
  const int32_t* d_indices = graph.indices;
  int32_t num_verts = graph.number_of_vertices;
  bool is_multigraph = graph.is_multigraph;
  cudaStream_t stream = 0;

  
  const int32_t* d_seeds = nullptr;
  bool seeds_is_iota = false;
  int32_t num_seeds = 0;
  if (vertices == nullptr || num_vertices == 0) {
    seeds_is_iota = true;
    d_seeds = nullptr;
    num_seeds = num_verts;
  } else {
    d_seeds = vertices;
    num_seeds = static_cast<int32_t>(num_vertices);
  }

  if (num_seeds <= 0) {
    return {nullptr, nullptr, nullptr, 0};
  }

  
  uint32_t stride_bits = ceil_log2_u32(static_cast<uint32_t>(num_verts));
  int64_t stride64 = 1ull << stride_bits;

  
  uint32_t sid_bits = ceil_log2_u32(static_cast<uint32_t>(num_seeds));
  uint32_t total_bits = stride_bits + sid_bits;
  bool use_u32_keys = (total_bits <= 32);

  
  int64_t* d_sizes = nullptr;
  cudaMalloc(&d_sizes, num_seeds * sizeof(int64_t));

  {
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (num_seeds + warps_per_block - 1) / warps_per_block;
    compute_sizes_kernel<<<blocks, threads, 0, stream>>>(d_offsets, d_indices, d_seeds, seeds_is_iota, num_seeds, d_sizes);
  }

  
  int64_t* d_write_offsets = nullptr;
  cudaMalloc(&d_write_offsets, num_seeds * sizeof(int64_t));
  {
    size_t ts = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, ts, (const int64_t*)nullptr, (int64_t*)nullptr, num_seeds);
    uint8_t* tmp = cache.ensure_scratch(ts);
    cub::DeviceScan::ExclusiveSum(tmp, ts, d_sizes, d_write_offsets, num_seeds, stream);
  }

  
  int64_t* d_total_pairs = nullptr;
  cudaMalloc(&d_total_pairs, sizeof(int64_t));
  compute_total_pairs_kernel<<<1, 32, 0, stream>>>(d_sizes, d_write_offsets, num_seeds, d_total_pairs);

  int64_t total_pairs = 0;
  cudaMemcpyAsync(&total_pairs, d_total_pairs, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaFree(d_total_pairs);

  if (total_pairs == 0) {
    cudaFree(d_sizes);
    cudaFree(d_write_offsets);
    return {nullptr, nullptr, nullptr, 0};
  }

  int32_t n_pairs = static_cast<int32_t>(total_pairs);

  
  int32_t* d_counts = nullptr;
  cudaMalloc(&d_counts, total_pairs * sizeof(int32_t));
  int32_t* d_num_runs = nullptr;
  cudaMalloc(&d_num_runs, sizeof(int32_t));

  int64_t* d_unique_keys_i64 = nullptr;
  uint32_t* d_unique_keys_u32 = nullptr;

  if (use_u32_keys) {
    
    uint32_t* d_keys_a = nullptr;
    uint32_t* d_keys_b = nullptr;
    cudaMalloc(&d_keys_a, total_pairs * sizeof(uint32_t));
    cudaMalloc(&d_keys_b, total_pairs * sizeof(uint32_t));

    fill_keys_u32_kernel<<<num_seeds, 512, 0, stream>>>(d_offsets, d_indices, d_seeds, seeds_is_iota, num_seeds,
                                                         d_write_offsets, d_keys_a, stride_bits);

    {
      size_t ts = 0;
      cub::DeviceRadixSort::SortKeys(nullptr, ts, (const uint32_t*)nullptr, (uint32_t*)nullptr, n_pairs, 0, static_cast<int>(total_bits));
      uint8_t* tmp = cache.ensure_scratch(ts);
      cub::DeviceRadixSort::SortKeys(tmp, ts, d_keys_a, d_keys_b, n_pairs, 0, static_cast<int>(total_bits), stream);
    }

    
    d_unique_keys_u32 = d_keys_a;
    {
      size_t ts = 0;
      cub::DeviceRunLengthEncode::Encode(nullptr, ts, (const uint32_t*)nullptr, (uint32_t*)nullptr, (int32_t*)nullptr, (int32_t*)nullptr, n_pairs);
      uint8_t* tmp = cache.ensure_scratch(ts);
      cub::DeviceRunLengthEncode::Encode(tmp, ts, d_keys_b, d_unique_keys_u32, d_counts, d_num_runs, n_pairs, stream);
    }

    cudaFree(d_keys_b);
  } else {
    
    int64_t* d_keys_a = nullptr;
    int64_t* d_keys_b = nullptr;
    cudaMalloc(&d_keys_a, total_pairs * sizeof(int64_t));
    cudaMalloc(&d_keys_b, total_pairs * sizeof(int64_t));

    fill_keys_kernel<<<num_seeds, 512, 0, stream>>>(d_offsets, d_indices, d_seeds, seeds_is_iota, num_seeds,
                                                     d_write_offsets, d_keys_a, stride64);

    int end_bit = static_cast<int>(total_bits);
    if (end_bit > 64) end_bit = 64;
    {
      size_t ts = 0;
      cub::DeviceRadixSort::SortKeys(nullptr, ts, (const int64_t*)nullptr, (int64_t*)nullptr, n_pairs, 0, end_bit);
      uint8_t* tmp = cache.ensure_scratch(ts);
      cub::DeviceRadixSort::SortKeys(tmp, ts, d_keys_a, d_keys_b, n_pairs, 0, end_bit, stream);
    }

    
    d_unique_keys_i64 = d_keys_a;
    {
      size_t ts = 0;
      cub::DeviceRunLengthEncode::Encode(nullptr, ts, (const int64_t*)nullptr, (int64_t*)nullptr, (int32_t*)nullptr, (int32_t*)nullptr, n_pairs);
      uint8_t* tmp = cache.ensure_scratch(ts);
      cub::DeviceRunLengthEncode::Encode(tmp, ts, d_keys_b, d_unique_keys_i64, d_counts, d_num_runs, n_pairs, stream);
    }

    cudaFree(d_keys_b);
  }

  cudaFree(d_sizes);
  cudaFree(d_write_offsets);

  int32_t num_runs = 0;
  cudaMemcpyAsync(&num_runs, d_num_runs, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaFree(d_num_runs);

  if (num_runs <= 0) {
    cudaFree(d_counts);
    if (d_unique_keys_i64) cudaFree(d_unique_keys_i64);
    if (d_unique_keys_u32) cudaFree(d_unique_keys_u32);
    return {nullptr, nullptr, nullptr, 0};
  }

  
  int32_t* d_flags = nullptr;
  cudaMalloc(&d_flags, num_runs * sizeof(int32_t));
  if (use_u32_keys) {
    int threads = 256;
    int blocks = (num_runs + threads - 1) / threads;
    make_valid_flags_u32_kernel<<<blocks, threads, 0, stream>>>(d_unique_keys_u32, num_runs, d_seeds, seeds_is_iota, stride_bits, d_flags);
  } else {
    int threads = 256;
    int blocks = (num_runs + threads - 1) / threads;
    make_valid_flags_kernel<<<blocks, threads, 0, stream>>>(d_unique_keys_i64, num_runs, d_seeds, seeds_is_iota, stride64, stride_bits, d_flags);
  }

  int32_t* d_scan = nullptr;
  cudaMalloc(&d_scan, num_runs * sizeof(int32_t));
  {
    size_t ts = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, ts, (const int32_t*)nullptr, (int32_t*)nullptr, num_runs);
    uint8_t* tmp = cache.ensure_scratch(ts);
    cub::DeviceScan::ExclusiveSum(tmp, ts, d_flags, d_scan, num_runs, stream);
  }

  
  int32_t out_count = 0;
  {
    int32_t last_scan = 0;
    int32_t last_flag = 0;
    cudaMemcpyAsync(&last_scan, d_scan + (num_runs - 1), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&last_flag, d_flags + (num_runs - 1), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    out_count = last_scan + last_flag;
  }

  cudaFree(d_flags);

  if (out_count <= 0) {
    cudaFree(d_counts);
    cudaFree(d_scan);
    if (d_unique_keys_i64) cudaFree(d_unique_keys_i64);
    if (d_unique_keys_u32) cudaFree(d_unique_keys_u32);
    return {nullptr, nullptr, nullptr, 0};
  }

  
  int32_t* d_out_first = nullptr;
  int32_t* d_out_second = nullptr;
  float* d_out_scores = nullptr;
  cudaMalloc(&d_out_first, out_count * sizeof(int32_t));
  cudaMalloc(&d_out_second, out_count * sizeof(int32_t));
  cudaMalloc(&d_out_scores, out_count * sizeof(float));

  if (use_u32_keys) {
    int threads = 256;
    int blocks = (num_runs + threads - 1) / threads;
    score_scatter_u32_kernel<<<blocks, threads, 0, stream>>>(d_unique_keys_u32, d_counts, num_runs,
                                                              d_offsets, d_indices, d_seeds, seeds_is_iota,
                                                              stride_bits, d_scan,
                                                              d_out_first, d_out_second, d_out_scores,
                                                              is_multigraph);
  } else {
    int threads = 256;
    int blocks = (num_runs + threads - 1) / threads;
    score_scatter_kernel<<<blocks, threads, 0, stream>>>(d_unique_keys_i64, d_counts, num_runs,
                                                          d_offsets, d_indices, d_seeds, seeds_is_iota,
                                                          stride64, stride_bits, d_scan,
                                                          d_out_first, d_out_second, d_out_scores,
                                                          is_multigraph);
  }

  cudaFree(d_counts);
  cudaFree(d_scan);
  if (d_unique_keys_i64) cudaFree(d_unique_keys_i64);
  if (d_unique_keys_u32) cudaFree(d_unique_keys_u32);

  
  if (topk.has_value() && static_cast<int64_t>(topk.value()) < static_cast<int64_t>(out_count)) {
    int32_t topk_val = static_cast<int32_t>(topk.value());
    if (topk_val <= 0) {
      cudaFree(d_out_first);
      cudaFree(d_out_second);
      cudaFree(d_out_scores);
      return {nullptr, nullptr, nullptr, 0};
    }

    int32_t* d_idx = nullptr;
    cudaMalloc(&d_idx, out_count * sizeof(int32_t));
    iota_kernel<<<(out_count + 255) / 256, 256, 0, stream>>>(d_idx, out_count);

    float* d_sorted_scores = nullptr;
    int32_t* d_sorted_idx = nullptr;
    cudaMalloc(&d_sorted_scores, out_count * sizeof(float));
    cudaMalloc(&d_sorted_idx, out_count * sizeof(int32_t));
    {
      size_t ts = 0;
      cub::DeviceRadixSort::SortPairsDescending(nullptr, ts,
                                                 (const float*)nullptr, (float*)nullptr,
                                                 (const int32_t*)nullptr, (int32_t*)nullptr,
                                                 out_count);
      uint8_t* tmp = cache.ensure_scratch(ts);
      cub::DeviceRadixSort::SortPairsDescending(tmp, ts,
                                                 d_out_scores, d_sorted_scores,
                                                 d_idx, d_sorted_idx,
                                                 out_count, 0, 32, stream);
    }

    cudaFree(d_idx);

    int32_t* d_topk_first = nullptr;
    int32_t* d_topk_second = nullptr;
    float* d_topk_scores = nullptr;
    cudaMalloc(&d_topk_first, topk_val * sizeof(int32_t));
    cudaMalloc(&d_topk_second, topk_val * sizeof(int32_t));
    cudaMalloc(&d_topk_scores, topk_val * sizeof(float));

    {
      int threads = 256;
      int blocks = (topk_val + threads - 1) / threads;
      gather_i32_kernel<<<blocks, threads, 0, stream>>>(d_out_first, d_sorted_idx, d_topk_first, topk_val);
      gather_i32_kernel<<<blocks, threads, 0, stream>>>(d_out_second, d_sorted_idx, d_topk_second, topk_val);
    }
    cudaMemcpyAsync(d_topk_scores, d_sorted_scores, sizeof(float) * topk_val, cudaMemcpyDeviceToDevice, stream);

    cudaFree(d_out_first);
    cudaFree(d_out_second);
    cudaFree(d_out_scores);
    cudaFree(d_sorted_scores);
    cudaFree(d_sorted_idx);

    return {d_topk_first, d_topk_second, d_topk_scores, static_cast<std::size_t>(topk_val)};
  }

  return {d_out_first, d_out_second, d_out_scores, static_cast<std::size_t>(out_count)};
}

}  
