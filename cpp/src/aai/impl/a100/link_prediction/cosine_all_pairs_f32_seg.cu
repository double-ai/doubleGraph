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
#include <climits>
#include <optional>

namespace aai {

namespace {





struct Cache : Cacheable {
  uint8_t* cub_tmp = nullptr;
  int64_t cub_tmp_capacity = 0;
  bool pool_initialized = false;

  uint8_t* ensure_cub_tmp(size_t bytes) {
    if (bytes == 0) return nullptr;
    if (cub_tmp_capacity < (int64_t)bytes) {
      if (cub_tmp) cudaFree(cub_tmp);
      cudaMalloc(&cub_tmp, bytes);
      cub_tmp_capacity = (int64_t)bytes;
    }
    return cub_tmp;
  }

  void ensure_pool() {
    if (!pool_initialized) {
      cudaMemPool_t pool;
      cudaDeviceGetDefaultMemPool(&pool, 0);
      uint64_t threshold = UINT64_MAX;
      cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
      pool_initialized = true;
    }
  }

  ~Cache() override {
    if (cub_tmp) { cudaFree(cub_tmp); cub_tmp = nullptr; }
  }
};





__global__ void iota_kernel(int32_t* out, int32_t n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) out[tid] = tid;
}

void launch_iota(int32_t* out, int32_t n) {
  if (n <= 0) return;
  iota_kernel<<<(n + 255) / 256, 256>>>(out, n);
}

__global__ void count_expansion_kernel(const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
                                       const int32_t* __restrict__ seeds, int32_t num_seeds,
                                       int64_t* __restrict__ counts) {
  int sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= num_seeds) return;

  int32_t u = seeds[sid];
  int32_t us = __ldg(&offsets[u]);
  int32_t ue = __ldg(&offsets[u + 1]);

  int64_t c = 0;
  for (int32_t i = us; i < ue; i++) {
    int32_t k = __ldg(&indices[i]);
    int32_t ks = __ldg(&offsets[k]);
    int32_t ke = __ldg(&offsets[k + 1]);
    c += (int64_t)(ke - ks);
  }
  counts[sid] = c;
}

void launch_count_expansion(const int32_t* offsets, const int32_t* indices, const int32_t* seeds,
                             int32_t num_seeds, int64_t* counts) {
  if (num_seeds <= 0) return;
  count_expansion_kernel<<<(num_seeds + 255) / 256, 256>>>(offsets, indices, seeds, num_seeds, counts);
}

template <typename KeyT>
__global__ void fill_expansion_keys_kernel(const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
                                          const int32_t* __restrict__ seeds, int32_t num_seeds,
                                          const int64_t* __restrict__ seed_prefix, int v_bits, KeyT v_mask,
                                          KeyT* __restrict__ out_keys) {
  int sid = (int)blockIdx.x;
  if (sid >= num_seeds) return;

  int32_t u = seeds[sid];
  int32_t us = __ldg(&offsets[u]);
  int32_t ue = __ldg(&offsets[u + 1]);

  int64_t out_base = seed_prefix[sid];
  KeyT sid_shifted = (KeyT)((uint64_t)sid << v_bits);

  int64_t local_offset = 0;
  for (int32_t i = us; i < ue; i++) {
    int32_t k = __ldg(&indices[i]);
    int32_t ks = __ldg(&offsets[k]);
    int32_t ke = __ldg(&offsets[k + 1]);
    int32_t kd = ke - ks;

    for (int32_t j = threadIdx.x; j < kd; j += (int32_t)blockDim.x) {
      int32_t v = __ldg(&indices[ks + j]);
      int64_t out_idx = out_base + local_offset + (int64_t)j;
      out_keys[out_idx] = sid_shifted | ((KeyT)v & v_mask);
    }

    local_offset += (int64_t)kd;
  }
}

void launch_fill_expansion_keys_u32(const int32_t* offsets, const int32_t* indices, const int32_t* seeds,
                                    int32_t num_seeds, const int64_t* seed_prefix, int v_bits,
                                    uint32_t v_mask, uint32_t* out_keys) {
  if (num_seeds <= 0) return;
  fill_expansion_keys_kernel<uint32_t><<<num_seeds, 256>>>(offsets, indices, seeds, num_seeds, seed_prefix, v_bits,
                                                           v_mask, out_keys);
}

void launch_fill_expansion_keys_u64(const int32_t* offsets, const int32_t* indices, const int32_t* seeds,
                                    int32_t num_seeds, const int64_t* seed_prefix, int v_bits,
                                    uint64_t v_mask, uint64_t* out_keys) {
  if (num_seeds <= 0) return;
  fill_expansion_keys_kernel<uint64_t><<<num_seeds, 256>>>(offsets, indices, seeds, num_seeds, seed_prefix, v_bits,
                                                           v_mask, out_keys);
}

template <typename KeyT>
struct NotSelfPair {
  int v_bits;
  KeyT v_mask;
  const int32_t* seeds;

  __device__ __forceinline__ bool operator()(KeyT key) const {
    int32_t sid = (int32_t)((uint64_t)key >> v_bits);
    int32_t v = (int32_t)((uint64_t)key & (uint64_t)v_mask);
    int32_t u = __ldg(&seeds[sid]);
    return u != v;
  }
};

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) x += __shfl_down_sync(0xffffffff, x, offset);
  return x;
}

template <typename KeyT>
__global__ void compute_scores_kernel(const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
                                      const float* __restrict__ weights, const int32_t* __restrict__ seeds,
                                      const KeyT* __restrict__ keys, int32_t num_pairs, int v_bits, KeyT v_mask,
                                      float* __restrict__ out_scores) {
  int warp_id = (int)(((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  int lane = threadIdx.x & 31;
  if (warp_id >= num_pairs) return;

  KeyT key = keys[warp_id];
  int32_t sid = (int32_t)((uint64_t)key >> v_bits);
  int32_t u = __ldg(&seeds[sid]);
  int32_t v = (int32_t)((uint64_t)key & (uint64_t)v_mask);

  int32_t us = __ldg(&offsets[u]);
  int32_t ue = __ldg(&offsets[u + 1]);
  int32_t vs = __ldg(&offsets[v]);
  int32_t ve = __ldg(&offsets[v + 1]);

  int32_t udeg = ue - us;
  int32_t vdeg = ve - vs;

  const int32_t* small_idx;
  const float* small_w;
  int32_t small_deg;
  const int32_t* big_idx;
  const float* big_w;
  int32_t big_deg;
  bool small_is_u;

  if (udeg <= vdeg) {
    small_idx = indices + us;
    small_w = weights + us;
    small_deg = udeg;
    big_idx = indices + vs;
    big_w = weights + vs;
    big_deg = vdeg;
    small_is_u = true;
  } else {
    small_idx = indices + vs;
    small_w = weights + vs;
    small_deg = vdeg;
    big_idx = indices + us;
    big_w = weights + us;
    big_deg = udeg;
    small_is_u = false;
  }

  float dot = 0.0f, nu = 0.0f, nv = 0.0f;

  for (int i = lane; i < small_deg; i += 32) {
    int32_t k = __ldg(&small_idx[i]);
    int lo = 0, hi = big_deg;
    while (lo < hi) {
      int mid = (lo + hi) >> 1;
      int32_t x = __ldg(&big_idx[mid]);
      if (x < k)
        lo = mid + 1;
      else
        hi = mid;
    }
    if (lo < big_deg && __ldg(&big_idx[lo]) == k) {
      float ws = __ldg(&small_w[i]);
      float wb = __ldg(&big_w[lo]);
      dot += ws * wb;
      if (small_is_u) {
        nu += ws * ws;
        nv += wb * wb;
      } else {
        nu += wb * wb;
        nv += ws * ws;
      }
    }
  }

  dot = warp_reduce_sum(dot);
  nu = warp_reduce_sum(nu);
  nv = warp_reduce_sum(nv);

  if (lane == 0) {
    float denom2 = nu * nv;
    float score = (denom2 > 0.0f) ? (dot * rsqrtf(denom2)) : 0.0f;
    out_scores[warp_id] = score;
  }
}

void launch_compute_scores_u32(const int32_t* offsets, const int32_t* indices, const float* weights,
                                const int32_t* seeds, const uint32_t* keys, int32_t num_pairs, int v_bits,
                                uint32_t v_mask, float* out_scores) {
  if (num_pairs <= 0) return;
  int threads = 256;
  int warps_per_block = threads / 32;
  int blocks = (num_pairs + warps_per_block - 1) / warps_per_block;
  compute_scores_kernel<uint32_t><<<blocks, threads>>>(offsets, indices, weights, seeds, keys, num_pairs, v_bits, v_mask,
                                                       out_scores);
}

void launch_compute_scores_u64(const int32_t* offsets, const int32_t* indices, const float* weights,
                                const int32_t* seeds, const uint64_t* keys, int32_t num_pairs, int v_bits,
                                uint64_t v_mask, float* out_scores) {
  if (num_pairs <= 0) return;
  int threads = 256;
  int warps_per_block = threads / 32;
  int blocks = (num_pairs + warps_per_block - 1) / warps_per_block;
  compute_scores_kernel<uint64_t><<<blocks, threads>>>(offsets, indices, weights, seeds, keys, num_pairs, v_bits, v_mask,
                                                       out_scores);
}

template <typename KeyT>
__global__ void decode_results_kernel(const KeyT* __restrict__ keys, const float* __restrict__ scores, int32_t n,
                                      const int32_t* __restrict__ seeds, int v_bits, KeyT v_mask,
                                      int32_t* __restrict__ out_u, int32_t* __restrict__ out_v,
                                      float* __restrict__ out_s) {
  int tid = (int)((int64_t)blockIdx.x * blockDim.x + threadIdx.x);
  if (tid >= n) return;
  KeyT key = keys[tid];
  int32_t sid = (int32_t)((uint64_t)key >> v_bits);
  out_u[tid] = __ldg(&seeds[sid]);
  out_v[tid] = (int32_t)((uint64_t)key & (uint64_t)v_mask);
  out_s[tid] = scores[tid];
}

void launch_decode_results_u32(const uint32_t* keys, const float* scores, int32_t n, const int32_t* seeds,
                                int v_bits, uint32_t v_mask, int32_t* out_u, int32_t* out_v,
                                float* out_s) {
  if (n <= 0) return;
  decode_results_kernel<uint32_t><<<((int64_t)n + 255) / 256, 256>>>(keys, scores, n, seeds, v_bits, v_mask, out_u,
                                                                    out_v, out_s);
}

void launch_decode_results_u64(const uint64_t* keys, const float* scores, int32_t n, const int32_t* seeds,
                                int v_bits, uint64_t v_mask, int32_t* out_u, int32_t* out_v,
                                float* out_s) {
  if (n <= 0) return;
  decode_results_kernel<uint64_t><<<((int64_t)n + 255) / 256, 256>>>(keys, scores, n, seeds, v_bits, v_mask, out_u,
                                                                    out_v, out_s);
}





size_t cub_exclusive_scan_temp_bytes_i64(int32_t n) {
  size_t temp = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, temp, (int64_t*)nullptr, (int64_t*)nullptr, n);
  return temp;
}

void cub_exclusive_scan_i64(void* temp, size_t temp_bytes, const int64_t* in, int64_t* out, int32_t n) {
  cub::DeviceScan::ExclusiveSum(temp, temp_bytes, in, out, n);
}

size_t cub_sort_keys_temp_bytes_u32(int32_t n, int begin_bit, int end_bit) {
  size_t temp = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, temp, (uint32_t*)nullptr, (uint32_t*)nullptr, n, begin_bit, end_bit);
  return temp;
}

void cub_sort_keys_u32(void* temp, size_t temp_bytes, const uint32_t* keys_in, uint32_t* keys_out, int32_t n,
                        int begin_bit, int end_bit) {
  cub::DeviceRadixSort::SortKeys(temp, temp_bytes, keys_in, keys_out, n, begin_bit, end_bit);
}

size_t cub_sort_keys_temp_bytes_u64(int32_t n, int begin_bit, int end_bit) {
  size_t temp = 0;
  cub::DeviceRadixSort::SortKeys(nullptr, temp, (uint64_t*)nullptr, (uint64_t*)nullptr, n, begin_bit, end_bit);
  return temp;
}

void cub_sort_keys_u64(void* temp, size_t temp_bytes, const uint64_t* keys_in, uint64_t* keys_out, int32_t n,
                        int begin_bit, int end_bit) {
  cub::DeviceRadixSort::SortKeys(temp, temp_bytes, keys_in, keys_out, n, begin_bit, end_bit);
}

size_t cub_unique_temp_bytes_u32(int32_t n) {
  size_t temp = 0;
  cub::DeviceSelect::Unique(nullptr, temp, (uint32_t*)nullptr, (uint32_t*)nullptr, (int32_t*)nullptr, n);
  return temp;
}

void cub_unique_u32(void* temp, size_t temp_bytes, const uint32_t* in, uint32_t* out, int32_t* d_num_selected,
                     int32_t n) {
  cub::DeviceSelect::Unique(temp, temp_bytes, in, out, d_num_selected, n);
}

size_t cub_unique_temp_bytes_u64(int32_t n) {
  size_t temp = 0;
  cub::DeviceSelect::Unique(nullptr, temp, (uint64_t*)nullptr, (uint64_t*)nullptr, (int32_t*)nullptr, n);
  return temp;
}

void cub_unique_u64(void* temp, size_t temp_bytes, const uint64_t* in, uint64_t* out, int32_t* d_num_selected,
                     int32_t n) {
  cub::DeviceSelect::Unique(temp, temp_bytes, in, out, d_num_selected, n);
}

size_t cub_select_not_self_temp_bytes_u32(int32_t n) {
  size_t temp = 0;
  NotSelfPair<uint32_t> pred{1, 1u, nullptr};
  cub::DeviceSelect::If(nullptr, temp, (uint32_t*)nullptr, (uint32_t*)nullptr, (int32_t*)nullptr, n, pred);
  return temp;
}

void cub_select_not_self_u32(void* temp, size_t temp_bytes, const uint32_t* in, uint32_t* out,
                              int32_t* d_num_selected, int32_t n, const int32_t* seeds, int v_bits,
                              uint32_t v_mask) {
  NotSelfPair<uint32_t> pred{v_bits, v_mask, seeds};
  cub::DeviceSelect::If(temp, temp_bytes, in, out, d_num_selected, n, pred);
}

size_t cub_select_not_self_temp_bytes_u64(int32_t n) {
  size_t temp = 0;
  NotSelfPair<uint64_t> pred{1, 1ull, nullptr};
  cub::DeviceSelect::If(nullptr, temp, (uint64_t*)nullptr, (uint64_t*)nullptr, (int32_t*)nullptr, n, pred);
  return temp;
}

void cub_select_not_self_u64(void* temp, size_t temp_bytes, const uint64_t* in, uint64_t* out,
                              int32_t* d_num_selected, int32_t n, const int32_t* seeds, int v_bits,
                              uint64_t v_mask) {
  NotSelfPair<uint64_t> pred{v_bits, v_mask, seeds};
  cub::DeviceSelect::If(temp, temp_bytes, in, out, d_num_selected, n, pred);
}

size_t cub_sort_pairs_desc_temp_bytes_f32_u32(int32_t n) {
  size_t temp = 0;
  cub::DeviceRadixSort::SortPairsDescending(nullptr, temp, (float*)nullptr, (float*)nullptr, (uint32_t*)nullptr,
                                            (uint32_t*)nullptr, n);
  return temp;
}

void cub_sort_pairs_desc_f32_u32(void* temp, size_t temp_bytes, const float* keys_in, float* keys_out,
                                  const uint32_t* vals_in, uint32_t* vals_out, int32_t n) {
  cub::DeviceRadixSort::SortPairsDescending(temp, temp_bytes, keys_in, keys_out, vals_in, vals_out, n);
}

size_t cub_sort_pairs_desc_temp_bytes_f32_u64(int32_t n) {
  size_t temp = 0;
  cub::DeviceRadixSort::SortPairsDescending(nullptr, temp, (float*)nullptr, (float*)nullptr, (uint64_t*)nullptr,
                                            (uint64_t*)nullptr, n);
  return temp;
}

void cub_sort_pairs_desc_f32_u64(void* temp, size_t temp_bytes, const float* keys_in, float* keys_out,
                                  const uint64_t* vals_in, uint64_t* vals_out, int32_t n) {
  cub::DeviceRadixSort::SortPairsDescending(temp, temp_bytes, keys_in, keys_out, vals_in, vals_out, n);
}

}  





similarity_result_float_t cosine_all_pairs_similarity_seg(const graph32_t& graph,
                                                          const float* edge_weights,
                                                          const int32_t* vertices,
                                                          std::size_t num_vertices,
                                                          std::optional<std::size_t> topk) {
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);
  cache.ensure_pool();

  const int32_t* d_offsets = graph.offsets;
  const int32_t* d_indices = graph.indices;
  int32_t num_verts = graph.number_of_vertices;
  const float* d_weights = edge_weights;

  similarity_result_float_t empty_result{nullptr, nullptr, nullptr, 0};

  
  int32_t num_seeds;
  const int32_t* d_seeds;
  int32_t* seeds_alloc = nullptr;
  if (vertices != nullptr && num_vertices > 0) {
    num_seeds = static_cast<int32_t>(num_vertices);
    d_seeds = vertices;
  } else {
    num_seeds = num_verts;
    cudaMalloc(&seeds_alloc, (size_t)num_verts * sizeof(int32_t));
    launch_iota(seeds_alloc, num_verts);
    d_seeds = seeds_alloc;
  }

  if (num_seeds <= 0) {
    if (seeds_alloc) cudaFree(seeds_alloc);
    return empty_result;
  }

  
  int v_bits = 0;
  while ((1ULL << v_bits) < (uint64_t)num_verts) v_bits++;
  uint64_t v_mask_u64 = (v_bits == 64) ? ~0ull : ((1ULL << v_bits) - 1ULL);
  uint32_t v_mask_u32 = (v_bits >= 32) ? 0xFFFFFFFFu : ((1u << v_bits) - 1u);

  int u_bits = 0;
  while ((1ULL << u_bits) < (uint64_t)num_seeds) u_bits++;
  int key_bits = u_bits + v_bits;

  bool use_u32_keys = (key_bits <= 32);
  int end_bit = key_bits;
  if (end_bit < 1) end_bit = 1;

  
  int64_t* d_counts = nullptr;
  cudaMalloc(&d_counts, (size_t)(num_seeds + 1) * sizeof(int64_t));
  cudaMemsetAsync(d_counts + num_seeds, 0, sizeof(int64_t));
  launch_count_expansion(d_offsets, d_indices, d_seeds, num_seeds, d_counts);

  
  int64_t* d_prefix = nullptr;
  cudaMalloc(&d_prefix, (size_t)(num_seeds + 1) * sizeof(int64_t));
  size_t scan_tmp_bytes = cub_exclusive_scan_temp_bytes_i64(num_seeds + 1);
  uint8_t* tmp = cache.ensure_cub_tmp(scan_tmp_bytes);
  cub_exclusive_scan_i64(tmp, scan_tmp_bytes, d_counts, d_prefix, num_seeds + 1);

  int64_t total_expansion = 0;
  cudaMemcpy(&total_expansion, d_prefix + num_seeds, sizeof(int64_t), cudaMemcpyDeviceToHost);

  cudaFree(d_counts);
  d_counts = nullptr;

  if (total_expansion <= 0) {
    cudaFree(d_prefix);
    if (seeds_alloc) cudaFree(seeds_alloc);
    return empty_result;
  }

  

  int32_t num_pairs = 0;

  if (use_u32_keys) {
    uint32_t* d_keys = nullptr;
    cudaMalloc(&d_keys, (size_t)total_expansion * sizeof(uint32_t));
    launch_fill_expansion_keys_u32(d_offsets, d_indices, d_seeds, num_seeds, d_prefix, v_bits,
                                   v_mask_u32, d_keys);

    cudaFree(d_prefix);
    d_prefix = nullptr;

    uint32_t* d_sorted_keys = nullptr;
    cudaMalloc(&d_sorted_keys, (size_t)total_expansion * sizeof(uint32_t));
    size_t sort_tmp_bytes = cub_sort_keys_temp_bytes_u32((int32_t)total_expansion, 0, end_bit);
    tmp = cache.ensure_cub_tmp(sort_tmp_bytes);
    cub_sort_keys_u32(tmp, sort_tmp_bytes, d_keys, d_sorted_keys, (int32_t)total_expansion, 0, end_bit);

    cudaFree(d_keys);
    d_keys = nullptr;

    uint32_t* d_unique_keys = nullptr;
    cudaMalloc(&d_unique_keys, (size_t)total_expansion * sizeof(uint32_t));
    int32_t* d_num_unique_dev = nullptr;
    cudaMalloc(&d_num_unique_dev, sizeof(int32_t));
    size_t uniq_tmp_bytes = cub_unique_temp_bytes_u32((int32_t)total_expansion);
    tmp = cache.ensure_cub_tmp(uniq_tmp_bytes);
    cub_unique_u32(tmp, uniq_tmp_bytes, d_sorted_keys, d_unique_keys, d_num_unique_dev, (int32_t)total_expansion);

    cudaFree(d_sorted_keys);
    d_sorted_keys = nullptr;

    int32_t num_unique = 0;
    cudaMemcpy(&num_unique, d_num_unique_dev, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_num_unique_dev);
    d_num_unique_dev = nullptr;

    if (num_unique <= 0) {
      cudaFree(d_unique_keys);
      if (seeds_alloc) cudaFree(seeds_alloc);
      return empty_result;
    }

    uint32_t* d_filtered_keys = nullptr;
    cudaMalloc(&d_filtered_keys, (size_t)num_unique * sizeof(uint32_t));
    int32_t* d_num_pairs_dev = nullptr;
    cudaMalloc(&d_num_pairs_dev, sizeof(int32_t));
    size_t filt_tmp_bytes = cub_select_not_self_temp_bytes_u32(num_unique);
    tmp = cache.ensure_cub_tmp(filt_tmp_bytes);
    cub_select_not_self_u32(tmp, filt_tmp_bytes, d_unique_keys, d_filtered_keys, d_num_pairs_dev, num_unique,
                            d_seeds, v_bits, v_mask_u32);

    cudaFree(d_unique_keys);
    d_unique_keys = nullptr;

    cudaMemcpy(&num_pairs, d_num_pairs_dev, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_num_pairs_dev);
    d_num_pairs_dev = nullptr;

    if (num_pairs <= 0) {
      cudaFree(d_filtered_keys);
      if (seeds_alloc) cudaFree(seeds_alloc);
      return empty_result;
    }

    float* d_scores = nullptr;
    cudaMalloc(&d_scores, (size_t)num_pairs * sizeof(float));
    launch_compute_scores_u32(d_offsets, d_indices, d_weights, d_seeds, d_filtered_keys,
                              num_pairs, v_bits, v_mask_u32, d_scores);

    int32_t out_n = num_pairs;
    bool do_topk = topk.has_value() && (*topk < (std::size_t)num_pairs);
    if (do_topk) out_n = (int32_t)*topk;

    const uint32_t* final_keys = d_filtered_keys;
    const float* final_scores = d_scores;
    float* d_sorted_scores = nullptr;
    uint32_t* d_sorted_pair_keys = nullptr;

    if (do_topk) {
      cudaMalloc(&d_sorted_scores, (size_t)num_pairs * sizeof(float));
      cudaMalloc(&d_sorted_pair_keys, (size_t)num_pairs * sizeof(uint32_t));
      size_t sp_tmp_bytes = cub_sort_pairs_desc_temp_bytes_f32_u32(num_pairs);
      tmp = cache.ensure_cub_tmp(sp_tmp_bytes);
      cub_sort_pairs_desc_f32_u32(tmp, sp_tmp_bytes, d_scores, d_sorted_scores,
                                  d_filtered_keys, d_sorted_pair_keys, num_pairs);
      final_keys = d_sorted_pair_keys;
      final_scores = d_sorted_scores;
    }

    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;
    cudaMalloc(&out_first, (size_t)out_n * sizeof(int32_t));
    cudaMalloc(&out_second, (size_t)out_n * sizeof(int32_t));
    cudaMalloc(&out_scores, (size_t)out_n * sizeof(float));

    launch_decode_results_u32(final_keys, final_scores, out_n, d_seeds, v_bits, v_mask_u32,
                              out_first, out_second, out_scores);

    cudaFree(d_filtered_keys);
    cudaFree(d_scores);
    if (d_sorted_scores) cudaFree(d_sorted_scores);
    if (d_sorted_pair_keys) cudaFree(d_sorted_pair_keys);
    if (seeds_alloc) cudaFree(seeds_alloc);

    return {out_first, out_second, out_scores, (std::size_t)out_n};

  } else {
    uint64_t* d_keys = nullptr;
    cudaMalloc(&d_keys, (size_t)total_expansion * sizeof(uint64_t));
    launch_fill_expansion_keys_u64(d_offsets, d_indices, d_seeds, num_seeds, d_prefix, v_bits,
                                   v_mask_u64, d_keys);

    cudaFree(d_prefix);
    d_prefix = nullptr;

    uint64_t* d_sorted_keys = nullptr;
    cudaMalloc(&d_sorted_keys, (size_t)total_expansion * sizeof(uint64_t));
    size_t sort_tmp_bytes = cub_sort_keys_temp_bytes_u64((int32_t)total_expansion, 0, end_bit);
    tmp = cache.ensure_cub_tmp(sort_tmp_bytes);
    cub_sort_keys_u64(tmp, sort_tmp_bytes, d_keys, d_sorted_keys, (int32_t)total_expansion, 0, end_bit);

    cudaFree(d_keys);
    d_keys = nullptr;

    uint64_t* d_unique_keys = nullptr;
    cudaMalloc(&d_unique_keys, (size_t)total_expansion * sizeof(uint64_t));
    int32_t* d_num_unique_dev = nullptr;
    cudaMalloc(&d_num_unique_dev, sizeof(int32_t));
    size_t uniq_tmp_bytes = cub_unique_temp_bytes_u64((int32_t)total_expansion);
    tmp = cache.ensure_cub_tmp(uniq_tmp_bytes);
    cub_unique_u64(tmp, uniq_tmp_bytes, d_sorted_keys, d_unique_keys, d_num_unique_dev, (int32_t)total_expansion);

    cudaFree(d_sorted_keys);
    d_sorted_keys = nullptr;

    int32_t num_unique = 0;
    cudaMemcpy(&num_unique, d_num_unique_dev, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_num_unique_dev);
    d_num_unique_dev = nullptr;

    if (num_unique <= 0) {
      cudaFree(d_unique_keys);
      if (seeds_alloc) cudaFree(seeds_alloc);
      return empty_result;
    }

    uint64_t* d_filtered_keys = nullptr;
    cudaMalloc(&d_filtered_keys, (size_t)num_unique * sizeof(uint64_t));
    int32_t* d_num_pairs_dev = nullptr;
    cudaMalloc(&d_num_pairs_dev, sizeof(int32_t));
    size_t filt_tmp_bytes = cub_select_not_self_temp_bytes_u64(num_unique);
    tmp = cache.ensure_cub_tmp(filt_tmp_bytes);
    cub_select_not_self_u64(tmp, filt_tmp_bytes, d_unique_keys, d_filtered_keys, d_num_pairs_dev, num_unique,
                            d_seeds, v_bits, v_mask_u64);

    cudaFree(d_unique_keys);
    d_unique_keys = nullptr;

    cudaMemcpy(&num_pairs, d_num_pairs_dev, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_num_pairs_dev);
    d_num_pairs_dev = nullptr;

    if (num_pairs <= 0) {
      cudaFree(d_filtered_keys);
      if (seeds_alloc) cudaFree(seeds_alloc);
      return empty_result;
    }

    float* d_scores = nullptr;
    cudaMalloc(&d_scores, (size_t)num_pairs * sizeof(float));
    launch_compute_scores_u64(d_offsets, d_indices, d_weights, d_seeds, (uint64_t*)d_filtered_keys, num_pairs,
                              v_bits, v_mask_u64, d_scores);

    int32_t out_n = num_pairs;
    bool do_topk = topk.has_value() && (*topk < (std::size_t)num_pairs);
    if (do_topk) out_n = (int32_t)*topk;

    const uint64_t* final_keys = d_filtered_keys;
    const float* final_scores = d_scores;
    float* d_sorted_scores = nullptr;
    uint64_t* d_sorted_pair_keys = nullptr;

    if (do_topk) {
      cudaMalloc(&d_sorted_scores, (size_t)num_pairs * sizeof(float));
      cudaMalloc(&d_sorted_pair_keys, (size_t)num_pairs * sizeof(uint64_t));
      size_t sp_tmp_bytes = cub_sort_pairs_desc_temp_bytes_f32_u64(num_pairs);
      tmp = cache.ensure_cub_tmp(sp_tmp_bytes);
      cub_sort_pairs_desc_f32_u64(tmp, sp_tmp_bytes, d_scores, d_sorted_scores,
                                  d_filtered_keys, d_sorted_pair_keys, num_pairs);
      final_keys = d_sorted_pair_keys;
      final_scores = d_sorted_scores;
    }

    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;
    cudaMalloc(&out_first, (size_t)out_n * sizeof(int32_t));
    cudaMalloc(&out_second, (size_t)out_n * sizeof(int32_t));
    cudaMalloc(&out_scores, (size_t)out_n * sizeof(float));

    launch_decode_results_u64(final_keys, final_scores, out_n, d_seeds, v_bits, v_mask_u64,
                              out_first, out_second, out_scores);

    cudaFree(d_filtered_keys);
    cudaFree(d_scores);
    if (d_sorted_scores) cudaFree(d_sorted_scores);
    if (d_sorted_pair_keys) cudaFree(d_sorted_pair_keys);
    if (seeds_alloc) cudaFree(seeds_alloc);

    return {out_first, out_second, out_scores, (std::size_t)out_n};
  }
}

}  
