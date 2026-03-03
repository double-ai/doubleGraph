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
#include <cstddef>
#include <optional>
#include <math_constants.h>

#include <cub/block/block_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>

namespace aai {

namespace {

struct Triple {
  double dot;
  double nu;
  double nv;
};
static_assert(sizeof(Triple) == 24, "Triple must be 24 bytes");

struct TripleSum {
  __host__ __device__ __forceinline__ Triple operator()(const Triple& a, const Triple& b) const {
    return {a.dot + b.dot, a.nu + b.nu, a.nv + b.nv};
  }
};

struct KeyScore {
  uint64_t key;
  double score;
};
static_assert(sizeof(KeyScore) == 16, "KeyScore must be 16 bytes");



static void get_scan_i64_temp_bytes(size_t* out_bytes, int32_t n) {
  size_t tmp = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, tmp, (const int64_t*)nullptr, (int64_t*)nullptr, n);
  *out_bytes = tmp;
}

static void get_sort_u64_triple_temp_bytes(size_t* out_bytes, int32_t n, int32_t end_bit) {
  size_t tmp = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, tmp,
                                  (const uint64_t*)nullptr, (uint64_t*)nullptr,
                                  (const Triple*)nullptr, (Triple*)nullptr,
                                  n, 0, end_bit);
  *out_bytes = tmp;
}

static void get_reduce_by_key_u64_triple_temp_bytes(size_t* out_bytes, int32_t n) {
  size_t tmp = 0;
  cub::DeviceReduce::ReduceByKey(nullptr, tmp,
                                 (const uint64_t*)nullptr, (uint64_t*)nullptr,
                                 (const Triple*)nullptr, (Triple*)nullptr,
                                 (int32_t*)nullptr,
                                 TripleSum{}, n);
  *out_bytes = tmp;
}

static void get_select_keyscore_temp_bytes(size_t* out_bytes, int32_t n) {
  size_t tmp = 0;
  cub::DeviceSelect::Flagged(nullptr, tmp,
                             (const KeyScore*)nullptr, (const uint8_t*)nullptr,
                             (KeyScore*)nullptr, (int32_t*)nullptr,
                             n);
  *out_bytes = tmp;
}

static void get_sort_score_u64_temp_bytes(size_t* out_bytes, int32_t n) {
  size_t tmp = 0;
  cub::DeviceRadixSort::SortPairsDescending(nullptr, tmp,
                                            (const double*)nullptr, (double*)nullptr,
                                            (const uint64_t*)nullptr, (uint64_t*)nullptr,
                                            n);
  *out_bytes = tmp;
}



__device__ __forceinline__ int32_t binary_search_sorted(
    const int32_t* __restrict__ arr,
    int32_t lo, int32_t hi,
    int32_t key) {
  while (lo < hi) {
    int32_t mid = lo + ((hi - lo) >> 1);
    int32_t v = __ldg(arr + mid);
    if (v < key) lo = mid + 1;
    else if (v > key) hi = mid;
    else return mid;
  }
  return -1;
}

__global__ void seed_degrees_i64_kernel(const int32_t* __restrict__ offsets,
                                       const int32_t* __restrict__ seeds,
                                       int32_t num_seeds,
                                       int64_t* __restrict__ seed_deg) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int stride = (int)(gridDim.x * blockDim.x);
  for (; i < num_seeds; i += stride) {
    int32_t u = seeds ? __ldg(seeds + i) : i;
    int32_t start = __ldg(offsets + u);
    int32_t end = __ldg(offsets + (u + 1));
    seed_deg[i] = (int64_t)(end - start);
  }
}

__global__ void finish_offsets_kernel(const int64_t* __restrict__ deg,
                                     const int64_t* __restrict__ scan_out,
                                     int64_t* __restrict__ offsets_out,
                                     int32_t n) {
  if (n == 0) return;
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int stride = (int)(gridDim.x * blockDim.x);
  for (; i < n; i += stride) offsets_out[i] = scan_out[i];
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    offsets_out[n] = scan_out[n - 1] + deg[n - 1];
  }
}

__global__ void build_uk_edges_kernel(const int32_t* __restrict__ offsets,
                                     const int32_t* __restrict__ indices,
                                     const int32_t* __restrict__ seeds,
                                     int32_t num_seeds,
                                     const int64_t* __restrict__ seed_edge_offsets,
                                     int32_t* __restrict__ out_u,
                                     int32_t* __restrict__ out_eidx) {
  (void)indices;
  int seed_i = (int)blockIdx.x;
  if (seed_i >= num_seeds) return;

  int32_t u = seeds ? __ldg(seeds + seed_i) : seed_i;
  int32_t u_start = __ldg(offsets + u);
  int32_t u_end = __ldg(offsets + (u + 1));
  int32_t deg_u = u_end - u_start;
  int64_t base = seed_edge_offsets[seed_i];

  int t = (int)threadIdx.x;
  for (int32_t j = t; j < deg_u; j += (int)blockDim.x) {
    int64_t idx = base + j;
    out_u[idx] = u;
    out_eidx[idx] = u_start + j;
  }
}

__global__ void uk_degk_i64_kernel(const int32_t* __restrict__ offsets,
                                  const int32_t* __restrict__ indices,
                                  const int32_t* __restrict__ uk_eidx,
                                  int32_t m,
                                  int64_t* __restrict__ out_degk) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int stride = (int)(gridDim.x * blockDim.x);
  for (; i < m; i += stride) {
    int32_t e = __ldg(uk_eidx + i);
    int32_t k = __ldg(indices + e);
    int32_t k_start = __ldg(offsets + k);
    int32_t k_end = __ldg(offsets + (k + 1));
    out_degk[i] = (int64_t)(k_end - k_start);
  }
}

__global__ void build_wedges_kernel(const int32_t* __restrict__ offsets,
                                   const int32_t* __restrict__ indices,
                                   const double* __restrict__ weights,
                                   const int32_t* __restrict__ uk_u,
                                   const int32_t* __restrict__ uk_eidx,
                                   int32_t m,
                                   const int64_t* __restrict__ uk_wedge_offsets,
                                   uint64_t* __restrict__ out_keys,
                                   Triple* __restrict__ out_vals) {
  int edge_i = (int)blockIdx.x;
  if (edge_i >= m) return;

  int32_t u = __ldg(uk_u + edge_i);
  int32_t e_uk = __ldg(uk_eidx + edge_i);
  int32_t k = __ldg(indices + e_uk);
  double w_uk = __ldg(weights + e_uk);
  double w_uk_sq = w_uk * w_uk;

  int32_t k_start = __ldg(offsets + k);
  int32_t k_end = __ldg(offsets + (k + 1));
  int32_t deg_k = k_end - k_start;
  int64_t base = uk_wedge_offsets[edge_i];

  int t = (int)threadIdx.x;
  for (int32_t j = t; j < deg_k; j += (int)blockDim.x) {
    int32_t e_kv = k_start + j;
    int32_t v = __ldg(indices + e_kv);

    int32_t v_start = __ldg(offsets + v);
    int32_t v_end = __ldg(offsets + (v + 1));
    int32_t pos_vk = binary_search_sorted(indices, v_start, v_end, k);
    if (pos_vk < 0) continue;
    double w_vk = __ldg(weights + pos_vk);

    uint64_t key = (uint64_t)(uint32_t)u;
    key = (key << 32) | (uint64_t)(uint32_t)v;

    int64_t out_idx = base + j;
    out_keys[out_idx] = key;
    out_vals[out_idx] = {w_uk * w_vk, w_uk_sq, w_vk * w_vk};
  }
}

__global__ void pack_keyscore_kernel(const uint64_t* __restrict__ keys,
                                    const Triple* __restrict__ triples,
                                    int32_t n,
                                    KeyScore* __restrict__ out_items,
                                    uint8_t* __restrict__ out_flags) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int stride = (int)(gridDim.x * blockDim.x);
  for (; i < n; i += stride) {
    uint64_t key = keys[i];
    uint32_t u = (uint32_t)(key >> 32);
    uint32_t v = (uint32_t)(key & 0xffffffffu);

    Triple t = triples[i];
    double score = 0.0;
    double prod = t.nu * t.nv;
    if (prod > 0.0) score = t.dot * rsqrt(prod);

    out_items[i] = {key, score};
    out_flags[i] = (u != v) ? 1 : 0;
  }
}

__global__ void unpack_keyscore_kernel(const KeyScore* __restrict__ items,
                                      int32_t n,
                                      int32_t* __restrict__ out_first,
                                      int32_t* __restrict__ out_second,
                                      double* __restrict__ out_scores) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int stride = (int)(gridDim.x * blockDim.x);
  for (; i < n; i += stride) {
    uint64_t key = items[i].key;
    out_first[i] = (int32_t)(key >> 32);
    out_second[i] = (int32_t)(key & 0xffffffffu);
    out_scores[i] = items[i].score;
  }
}

__global__ void extract_score_key_kernel(const KeyScore* __restrict__ items,
                                        int32_t n,
                                        double* __restrict__ scores,
                                        uint64_t* __restrict__ keys) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int stride = (int)(gridDim.x * blockDim.x);
  for (; i < n; i += stride) {
    scores[i] = items[i].score;
    keys[i] = items[i].key;
  }
}

__global__ void unpack_topk_kernel(const double* __restrict__ scores,
                                  const uint64_t* __restrict__ keys,
                                  int32_t k,
                                  int32_t* __restrict__ out_first,
                                  int32_t* __restrict__ out_second,
                                  double* __restrict__ out_scores) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int stride = (int)(gridDim.x * blockDim.x);
  for (; i < k; i += stride) {
    uint64_t key = keys[i];
    out_first[i] = (int32_t)(key >> 32);
    out_second[i] = (int32_t)(key & 0xffffffffu);
    out_scores[i] = scores[i];
  }
}

template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void block_topk_candidates_score_kernel(const KeyScore* __restrict__ items,
                                                  int32_t n,
                                                  double* __restrict__ out_scores,
                                                  uint64_t* __restrict__ out_keys,
                                                  int32_t k) {
  using BlockSort = cub::BlockRadixSort<double, BLOCK_THREADS, ITEMS_PER_THREAD, uint64_t>;
  __shared__ typename BlockSort::TempStorage temp_storage;

  double keys[ITEMS_PER_THREAD];
  uint64_t vals[ITEMS_PER_THREAD];

  int32_t block_items = BLOCK_THREADS * ITEMS_PER_THREAD;
  int32_t base = (int32_t)blockIdx.x * block_items;

#pragma unroll
  for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
    int32_t idx = base + (int32_t)threadIdx.x * ITEMS_PER_THREAD + j;
    if (idx < n) {
      keys[j] = items[idx].score;
      vals[j] = items[idx].key;
    } else {
      keys[j] = -CUDART_INF;
      vals[j] = 0ULL;
    }
  }

  BlockSort(temp_storage).SortDescending(keys, vals);

  int32_t out_base = (int32_t)blockIdx.x * k;
#pragma unroll
  for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
    int32_t rank = (int32_t)threadIdx.x * ITEMS_PER_THREAD + j;
    if (rank < k) {
      out_scores[out_base + rank] = keys[j];
      out_keys[out_base + rank] = vals[j];
    }
  }
}



static void launch_scan_exclusive_i64(void* d_temp, size_t temp_bytes,
                                      const int64_t* d_in, int64_t* d_out, int32_t n,
                                      cudaStream_t stream) {
  size_t bytes = temp_bytes;
  cub::DeviceScan::ExclusiveSum(d_temp, bytes, d_in, d_out, n, stream);
}

static void launch_sort_u64_triple(void* d_temp, size_t temp_bytes,
                                   const uint64_t* keys_in, uint64_t* keys_out,
                                   const void* vals_in, void* vals_out,
                                   int32_t n, int32_t end_bit, cudaStream_t stream) {
  size_t bytes = temp_bytes;
  cub::DeviceRadixSort::SortPairs(d_temp, bytes,
                                  keys_in, keys_out,
                                  (const Triple*)vals_in, (Triple*)vals_out,
                                  n, 0, end_bit, stream);
}

static void launch_reduce_by_key_u64_triple(void* d_temp, size_t temp_bytes,
                                            const uint64_t* keys_in, uint64_t* unique_out,
                                            const void* vals_in, void* aggregates_out,
                                            int32_t* d_num_runs,
                                            int32_t n, cudaStream_t stream) {
  size_t bytes = temp_bytes;
  cub::DeviceReduce::ReduceByKey(d_temp, bytes,
                                 keys_in, unique_out,
                                 (const Triple*)vals_in, (Triple*)aggregates_out,
                                 d_num_runs,
                                 TripleSum{}, n, stream);
}

static void launch_select_keyscore(void* d_temp, size_t temp_bytes,
                                   const void* in_items, const uint8_t* flags,
                                   void* out_items, int32_t* d_num_selected,
                                   int32_t n, cudaStream_t stream) {
  size_t bytes = temp_bytes;
  cub::DeviceSelect::Flagged(d_temp, bytes,
                             (const KeyScore*)in_items, flags,
                             (KeyScore*)out_items, d_num_selected,
                             n, stream);
}

static void launch_sort_score_u64_desc(void* d_temp, size_t temp_bytes,
                                       const double* scores_in, double* scores_out,
                                       const uint64_t* keys_in, uint64_t* keys_out,
                                       int32_t n, cudaStream_t stream) {
  size_t bytes = temp_bytes;
  cub::DeviceRadixSort::SortPairsDescending(d_temp, bytes,
                                            scores_in, scores_out,
                                            keys_in, keys_out,
                                            n, 0, 64, stream);
}

static void launch_seed_degrees_i64(const int32_t* offsets, const int32_t* seeds, int32_t num_seeds,
                                    int64_t* seed_deg, cudaStream_t stream) {
  int block = 256;
  int grid = (num_seeds + block - 1) / block;
  if (grid > 65535) grid = 65535;
  seed_degrees_i64_kernel<<<grid, block, 0, stream>>>(offsets, seeds, num_seeds, seed_deg);
}

static void launch_finish_offsets(const int64_t* deg, const int64_t* scan_out, int64_t* offsets_out, int32_t n,
                                  cudaStream_t stream) {
  if (n <= 0) return;
  int block = 256;
  int grid = (n + block - 1) / block;
  if (grid > 65535) grid = 65535;
  finish_offsets_kernel<<<grid, block, 0, stream>>>(deg, scan_out, offsets_out, n);
}

static void launch_build_uk_edges(const int32_t* offsets, const int32_t* indices,
                                  const int32_t* seeds, int32_t num_seeds,
                                  const int64_t* seed_edge_offsets,
                                  int32_t* out_u, int32_t* out_eidx,
                                  cudaStream_t stream) {
  (void)indices;
  dim3 block(256);
  dim3 grid((unsigned)num_seeds);
  build_uk_edges_kernel<<<grid, block, 0, stream>>>(offsets, indices, seeds, num_seeds, seed_edge_offsets, out_u, out_eidx);
}

static void launch_uk_degk_i64(const int32_t* offsets, const int32_t* indices,
                                const int32_t* uk_eidx, int32_t m,
                                int64_t* out_degk, cudaStream_t stream) {
  int block = 256;
  int grid = (m + block - 1) / block;
  if (grid > 65535) grid = 65535;
  uk_degk_i64_kernel<<<grid, block, 0, stream>>>(offsets, indices, uk_eidx, m, out_degk);
}

static void launch_build_wedges(const int32_t* offsets, const int32_t* indices, const double* weights,
                                const int32_t* uk_u, const int32_t* uk_eidx, int32_t m,
                                const int64_t* uk_wedge_offsets,
                                uint64_t* out_keys, void* out_vals,
                                cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((unsigned)m);
  build_wedges_kernel<<<grid, block, 0, stream>>>(offsets, indices, weights, uk_u, uk_eidx, m, uk_wedge_offsets, out_keys, (Triple*)out_vals);
}

static void launch_pack_keyscore(const uint64_t* keys, const void* triples, int32_t n,
                                 void* out_keyscore, uint8_t* out_flags,
                                 cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  if (grid > 65535) grid = 65535;
  pack_keyscore_kernel<<<grid, block, 0, stream>>>(keys, (const Triple*)triples, n, (KeyScore*)out_keyscore, out_flags);
}

static void launch_unpack_keyscore(const void* keyscore, int32_t n,
                                   int32_t* out_first, int32_t* out_second, double* out_scores,
                                   cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  if (grid > 65535) grid = 65535;
  unpack_keyscore_kernel<<<grid, block, 0, stream>>>((const KeyScore*)keyscore, n, out_first, out_second, out_scores);
}

static void launch_extract_score_key(const void* keyscore, int32_t n, double* scores, uint64_t* keys,
                                     cudaStream_t stream) {
  int block = 256;
  int grid = (n + block - 1) / block;
  if (grid > 65535) grid = 65535;
  extract_score_key_kernel<<<grid, block, 0, stream>>>((const KeyScore*)keyscore, n, scores, keys);
}

static void launch_unpack_topk(const double* scores, const uint64_t* keys, int32_t k,
                                int32_t* out_first, int32_t* out_second, double* out_scores,
                                cudaStream_t stream) {
  int block = 256;
  int grid = (k + block - 1) / block;
  if (grid > 65535) grid = 65535;
  unpack_topk_kernel<<<grid, block, 0, stream>>>(scores, keys, k, out_first, out_second, out_scores);
}

static void launch_block_topk_candidates_score_u64(const void* keyscore, int32_t n,
                                                   double* out_scores, uint64_t* out_keys,
                                                   int32_t k, cudaStream_t stream) {
  constexpr int BLOCK_THREADS = 256;
  constexpr int ITEMS_PER_THREAD = 4;
  int32_t block_items = BLOCK_THREADS * ITEMS_PER_THREAD;
  int32_t grid = (n + block_items - 1) / block_items;
  block_topk_candidates_score_kernel<BLOCK_THREADS, ITEMS_PER_THREAD><<<grid, BLOCK_THREADS, 0, stream>>>(
      (const KeyScore*)keyscore, n, out_scores, out_keys, k);
}



struct CubTempCache {
  int32_t n = 0;
  int32_t end_bit = 0;
  size_t bytes = 0;
};

struct HostCounts {
  int64_t m;
  int64_t l;
  int32_t unique;
  int32_t selected;
};

struct Cache : Cacheable {
  HostCounts* h_counts = nullptr;

  void* cub_temp = nullptr;
  size_t cub_temp_bytes = 0;

  CubTempCache scan_cache;
  CubTempCache sort_key_cache;
  CubTempCache reduce_cache;
  CubTempCache select_cache;
  CubTempCache sort_score_cache;

  
  int32_t cap_seeds = 0;
  int64_t* seed_deg = nullptr;
  int64_t* seed_scan = nullptr;
  int64_t* seed_edge_offsets = nullptr;

  
  int32_t cap_m = 0;
  int32_t* uk_u = nullptr;
  int32_t* uk_eidx = nullptr;
  int64_t* degk = nullptr;
  int64_t* degk_scan = nullptr;
  int64_t* uk_wedge_offsets = nullptr;

  
  int32_t cap_l = 0;
  uint64_t* keys_in = nullptr;
  uint64_t* keys_out = nullptr;
  void* vals_in = nullptr;
  void* vals_out = nullptr;

  
  int32_t cap_unique = 0;
  uint64_t* unique_keys = nullptr;
  void* agg_vals = nullptr;
  int32_t* num_runs_d = nullptr;

  
  int32_t cap_selected = 0;
  void* keyscore_in = nullptr;
  void* keyscore_out = nullptr;
  uint8_t* flags = nullptr;
  int32_t* num_selected_d = nullptr;
  double* scores_a = nullptr;
  uint64_t* keys_a = nullptr;
  double* scores_b = nullptr;
  uint64_t* keys_b = nullptr;

  
  int32_t cap_cand = 0;
  double* cand_scores_a = nullptr;
  uint64_t* cand_keys_a = nullptr;
  double* cand_scores_b = nullptr;
  uint64_t* cand_keys_b = nullptr;

  Cache() {
    cudaHostAlloc((void**)&h_counts, sizeof(HostCounts), cudaHostAllocPortable);
  }

  ~Cache() override {
    if (h_counts) cudaFreeHost(h_counts);
    if (cub_temp) cudaFree(cub_temp);
    if (seed_deg) cudaFree(seed_deg);
    if (seed_scan) cudaFree(seed_scan);
    if (seed_edge_offsets) cudaFree(seed_edge_offsets);
    if (uk_u) cudaFree(uk_u);
    if (uk_eidx) cudaFree(uk_eidx);
    if (degk) cudaFree(degk);
    if (degk_scan) cudaFree(degk_scan);
    if (uk_wedge_offsets) cudaFree(uk_wedge_offsets);
    if (keys_in) cudaFree(keys_in);
    if (keys_out) cudaFree(keys_out);
    if (vals_in) cudaFree(vals_in);
    if (vals_out) cudaFree(vals_out);
    if (unique_keys) cudaFree(unique_keys);
    if (agg_vals) cudaFree(agg_vals);
    if (num_runs_d) cudaFree(num_runs_d);
    if (keyscore_in) cudaFree(keyscore_in);
    if (keyscore_out) cudaFree(keyscore_out);
    if (flags) cudaFree(flags);
    if (num_selected_d) cudaFree(num_selected_d);
    if (scores_a) cudaFree(scores_a);
    if (keys_a) cudaFree(keys_a);
    if (scores_b) cudaFree(scores_b);
    if (keys_b) cudaFree(keys_b);
    if (cand_scores_a) cudaFree(cand_scores_a);
    if (cand_keys_a) cudaFree(cand_keys_a);
    if (cand_scores_b) cudaFree(cand_scores_b);
    if (cand_keys_b) cudaFree(cand_keys_b);
  }

  void ensure_cub_temp(size_t bytes) {
    if (bytes <= cub_temp_bytes) return;
    if (cub_temp) cudaFree(cub_temp);
    cub_temp_bytes = bytes;
    cudaMalloc(&cub_temp, cub_temp_bytes);
  }

  void ensure_seed_capacity(int32_t n) {
    if (n <= cap_seeds) return;
    cap_seeds = n;
    if (seed_deg) cudaFree(seed_deg);
    cudaMalloc(&seed_deg, (size_t)cap_seeds * sizeof(int64_t));
    if (seed_scan) cudaFree(seed_scan);
    cudaMalloc(&seed_scan, (size_t)cap_seeds * sizeof(int64_t));
    if (seed_edge_offsets) cudaFree(seed_edge_offsets);
    cudaMalloc(&seed_edge_offsets, (size_t)(cap_seeds + 1) * sizeof(int64_t));
  }

  void ensure_m_capacity(int32_t m) {
    if (m <= cap_m) return;
    cap_m = m;
    if (uk_u) cudaFree(uk_u);
    cudaMalloc(&uk_u, (size_t)cap_m * sizeof(int32_t));
    if (uk_eidx) cudaFree(uk_eidx);
    cudaMalloc(&uk_eidx, (size_t)cap_m * sizeof(int32_t));
    if (degk) cudaFree(degk);
    cudaMalloc(&degk, (size_t)cap_m * sizeof(int64_t));
    if (degk_scan) cudaFree(degk_scan);
    cudaMalloc(&degk_scan, (size_t)cap_m * sizeof(int64_t));
    if (uk_wedge_offsets) cudaFree(uk_wedge_offsets);
    cudaMalloc(&uk_wedge_offsets, (size_t)(cap_m + 1) * sizeof(int64_t));
  }

  void ensure_l_capacity(int32_t l) {
    if (l <= cap_l) return;
    cap_l = l;
    if (keys_in) cudaFree(keys_in);
    cudaMalloc(&keys_in, (size_t)cap_l * sizeof(uint64_t));
    if (keys_out) cudaFree(keys_out);
    cudaMalloc(&keys_out, (size_t)cap_l * sizeof(uint64_t));
    constexpr int TRIPLE_BYTES = 24;
    if (vals_in) cudaFree(vals_in);
    cudaMalloc(&vals_in, (size_t)cap_l * TRIPLE_BYTES);
    if (vals_out) cudaFree(vals_out);
    cudaMalloc(&vals_out, (size_t)cap_l * TRIPLE_BYTES);
  }

  void ensure_unique_capacity(int32_t u) {
    if (u <= cap_unique) return;
    cap_unique = u;
    if (unique_keys) cudaFree(unique_keys);
    cudaMalloc(&unique_keys, (size_t)cap_unique * sizeof(uint64_t));
    constexpr int TRIPLE_BYTES = 24;
    if (agg_vals) cudaFree(agg_vals);
    cudaMalloc(&agg_vals, (size_t)cap_unique * TRIPLE_BYTES);
    if (num_runs_d) cudaFree(num_runs_d);
    cudaMalloc(&num_runs_d, sizeof(int32_t));
  }

  void ensure_selected_capacity(int32_t n) {
    if (n <= cap_selected) return;
    cap_selected = n;
    constexpr int KEYSCORE_BYTES = 16;
    if (keyscore_in) cudaFree(keyscore_in);
    cudaMalloc(&keyscore_in, (size_t)cap_selected * KEYSCORE_BYTES);
    if (keyscore_out) cudaFree(keyscore_out);
    cudaMalloc(&keyscore_out, (size_t)cap_selected * KEYSCORE_BYTES);
    if (flags) cudaFree(flags);
    cudaMalloc(&flags, (size_t)cap_selected * sizeof(uint8_t));
    if (num_selected_d) cudaFree(num_selected_d);
    cudaMalloc(&num_selected_d, sizeof(int32_t));
    if (scores_a) cudaFree(scores_a);
    cudaMalloc(&scores_a, (size_t)cap_selected * sizeof(double));
    if (keys_a) cudaFree(keys_a);
    cudaMalloc(&keys_a, (size_t)cap_selected * sizeof(uint64_t));
    if (scores_b) cudaFree(scores_b);
    cudaMalloc(&scores_b, (size_t)cap_selected * sizeof(double));
    if (keys_b) cudaFree(keys_b);
    cudaMalloc(&keys_b, (size_t)cap_selected * sizeof(uint64_t));
  }

  void ensure_cand_capacity(int32_t n) {
    if (n <= cap_cand) return;
    cap_cand = n;
    if (cand_scores_a) cudaFree(cand_scores_a);
    cudaMalloc(&cand_scores_a, (size_t)cap_cand * sizeof(double));
    if (cand_keys_a) cudaFree(cand_keys_a);
    cudaMalloc(&cand_keys_a, (size_t)cap_cand * sizeof(uint64_t));
    if (cand_scores_b) cudaFree(cand_scores_b);
    cudaMalloc(&cand_scores_b, (size_t)cap_cand * sizeof(double));
    if (cand_keys_b) cudaFree(cand_keys_b);
    cudaMalloc(&cand_keys_b, (size_t)cap_cand * sizeof(uint64_t));
  }

  void ensure_scan_temp(int32_t n) {
    if (n <= scan_cache.n && scan_cache.bytes > 0) {
      ensure_cub_temp(scan_cache.bytes);
      return;
    }
    size_t bytes = 0;
    get_scan_i64_temp_bytes(&bytes, n);
    scan_cache.n = n;
    scan_cache.bytes = bytes;
    ensure_cub_temp(bytes);
  }

  void ensure_sort_key_temp(int32_t n, int32_t end_bit) {
    if (n <= sort_key_cache.n && end_bit <= sort_key_cache.end_bit && sort_key_cache.bytes > 0) {
      ensure_cub_temp(sort_key_cache.bytes);
      return;
    }
    size_t bytes = 0;
    get_sort_u64_triple_temp_bytes(&bytes, n, end_bit);
    sort_key_cache.n = n;
    sort_key_cache.end_bit = end_bit;
    sort_key_cache.bytes = bytes;
    ensure_cub_temp(bytes);
  }

  void ensure_reduce_temp(int32_t n) {
    if (n <= reduce_cache.n && reduce_cache.bytes > 0) {
      ensure_cub_temp(reduce_cache.bytes);
      return;
    }
    size_t bytes = 0;
    get_reduce_by_key_u64_triple_temp_bytes(&bytes, n);
    reduce_cache.n = n;
    reduce_cache.bytes = bytes;
    ensure_cub_temp(bytes);
  }

  void ensure_select_temp(int32_t n) {
    if (n <= select_cache.n && select_cache.bytes > 0) {
      ensure_cub_temp(select_cache.bytes);
      return;
    }
    size_t bytes = 0;
    get_select_keyscore_temp_bytes(&bytes, n);
    select_cache.n = n;
    select_cache.bytes = bytes;
    ensure_cub_temp(bytes);
  }

  void ensure_sort_score_temp(int32_t n) {
    if (n <= sort_score_cache.n && sort_score_cache.bytes > 0) {
      ensure_cub_temp(sort_score_cache.bytes);
      return;
    }
    size_t bytes = 0;
    get_sort_score_u64_temp_bytes(&bytes, n);
    sort_score_cache.n = n;
    sort_score_cache.bytes = bytes;
    ensure_cub_temp(bytes);
  }
};

}  

similarity_result_double_t cosine_all_pairs_similarity_seg(const graph32_t& graph,
                                                           const double* edge_weights,
                                                           const int32_t* vertices,
                                                           std::size_t num_vertices,
                                                           std::optional<std::size_t> topk) {
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);

  cudaStream_t stream = 0;

  const int32_t num_verts = graph.number_of_vertices;
  const int32_t* d_offsets = graph.offsets;
  const int32_t* d_indices = graph.indices;
  const double* d_weights = edge_weights;

  const int32_t num_seeds = (vertices != nullptr && num_vertices > 0)
                                ? (int32_t)num_vertices
                                : num_verts;
  const int32_t* d_seeds = (vertices != nullptr && num_vertices > 0)
                               ? vertices
                               : nullptr;

  std::optional<int32_t> topk_opt;
  if (topk.has_value()) topk_opt = (int32_t)topk.value();

  cache.ensure_seed_capacity(num_seeds);

  
  launch_seed_degrees_i64(d_offsets, d_seeds, num_seeds, cache.seed_deg, stream);

  
  cache.ensure_scan_temp(num_seeds);
  launch_scan_exclusive_i64(cache.cub_temp, cache.cub_temp_bytes,
                            cache.seed_deg, cache.seed_scan, num_seeds, stream);

  
  launch_finish_offsets(cache.seed_deg, cache.seed_scan,
                        cache.seed_edge_offsets, num_seeds, stream);

  cudaMemcpyAsync(&cache.h_counts->m, cache.seed_edge_offsets + num_seeds,
                  sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  int32_t m = (cache.h_counts->m > 0) ? (int32_t)cache.h_counts->m : 0;

  if (m == 0) {
    return {nullptr, nullptr, nullptr, 0};
  }

  cache.ensure_m_capacity(m);

  
  launch_build_uk_edges(d_offsets, d_indices,
                        d_seeds, num_seeds,
                        cache.seed_edge_offsets,
                        cache.uk_u, cache.uk_eidx,
                        stream);

  
  launch_uk_degk_i64(d_offsets, d_indices, cache.uk_eidx, m, cache.degk, stream);

  
  cache.ensure_scan_temp(m);
  launch_scan_exclusive_i64(cache.cub_temp, cache.cub_temp_bytes,
                            cache.degk, cache.degk_scan, m, stream);

  launch_finish_offsets(cache.degk, cache.degk_scan,
                        cache.uk_wedge_offsets, m, stream);

  cudaMemcpyAsync(&cache.h_counts->l, cache.uk_wedge_offsets + m,
                  sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  int32_t l = (cache.h_counts->l > 0) ? (int32_t)cache.h_counts->l : 0;

  if (l == 0) {
    return {nullptr, nullptr, nullptr, 0};
  }

  cache.ensure_l_capacity(l);

  
  launch_build_wedges(d_offsets, d_indices, d_weights,
                      cache.uk_u, cache.uk_eidx, m,
                      cache.uk_wedge_offsets,
                      cache.keys_in,
                      cache.vals_in,
                      stream);

  
  int32_t bits = 1;
  if (num_verts > 1) {
    uint32_t x = (uint32_t)(num_verts - 1);
    bits = 32 - __builtin_clz(x);
  }
  int32_t end_bit = 32 + bits;

  cache.ensure_sort_key_temp(l, end_bit);
  launch_sort_u64_triple(cache.cub_temp, cache.cub_temp_bytes,
                         cache.keys_in,
                         cache.keys_out,
                         cache.vals_in, cache.vals_out,
                         l, end_bit, stream);

  
  cache.ensure_unique_capacity(l);
  cache.ensure_reduce_temp(l);
  launch_reduce_by_key_u64_triple(cache.cub_temp, cache.cub_temp_bytes,
                                  cache.keys_out,
                                  cache.unique_keys,
                                  cache.vals_out, cache.agg_vals,
                                  cache.num_runs_d,
                                  l, stream);

  cudaMemcpyAsync(&cache.h_counts->unique, cache.num_runs_d, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  int32_t h_unique = cache.h_counts->unique;

  if (h_unique <= 0) {
    return {nullptr, nullptr, nullptr, 0};
  }

  cache.ensure_selected_capacity(h_unique);

  
  launch_pack_keyscore(cache.unique_keys,
                       cache.agg_vals, h_unique,
                       cache.keyscore_in, cache.flags,
                       stream);

  
  cache.ensure_select_temp(h_unique);
  launch_select_keyscore(cache.cub_temp, cache.cub_temp_bytes,
                         cache.keyscore_in, cache.flags,
                         cache.keyscore_out, cache.num_selected_d,
                         h_unique, stream);

  cudaMemcpyAsync(&cache.h_counts->selected, cache.num_selected_d, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  int32_t h_selected = cache.h_counts->selected;

  if (h_selected <= 0) {
    return {nullptr, nullptr, nullptr, 0};
  }

  
  if (topk_opt.has_value() && h_selected > topk_opt.value()) {
    int32_t k = topk_opt.value();

    
    if (k <= 512 && h_selected >= 2048) {
      constexpr int BLOCK_ITEMS = 1024;
      int32_t num_blocks = (h_selected + BLOCK_ITEMS - 1) / BLOCK_ITEMS;
      int32_t cand_n = num_blocks * k;
      cache.ensure_cand_capacity(cand_n);

      launch_block_topk_candidates_score_u64(cache.keyscore_out, h_selected,
                                             cache.cand_scores_a,
                                             cache.cand_keys_a,
                                             k, stream);

      if (num_blocks == 1) {
        int32_t* out_first = nullptr;
        int32_t* out_second = nullptr;
        double* out_scores = nullptr;
        cudaMalloc(&out_first, (size_t)k * sizeof(int32_t));
        cudaMalloc(&out_second, (size_t)k * sizeof(int32_t));
        cudaMalloc(&out_scores, (size_t)k * sizeof(double));
        launch_unpack_topk(cache.cand_scores_a,
                           cache.cand_keys_a,
                           k,
                           out_first, out_second, out_scores,
                           stream);
        return {out_first, out_second, out_scores, (std::size_t)k};
      }

      cache.ensure_sort_score_temp(cand_n);
      launch_sort_score_u64_desc(cache.cub_temp, cache.cub_temp_bytes,
                                 cache.cand_scores_a, cache.cand_scores_b,
                                 cache.cand_keys_a,
                                 cache.cand_keys_b,
                                 cand_n, stream);

      int32_t* out_first = nullptr;
      int32_t* out_second = nullptr;
      double* out_scores = nullptr;
      cudaMalloc(&out_first, (size_t)k * sizeof(int32_t));
      cudaMalloc(&out_second, (size_t)k * sizeof(int32_t));
      cudaMalloc(&out_scores, (size_t)k * sizeof(double));

      launch_unpack_topk(cache.cand_scores_b,
                         cache.cand_keys_b,
                         k,
                         out_first, out_second, out_scores,
                         stream);

      return {out_first, out_second, out_scores, (std::size_t)k};
    }

    
    launch_extract_score_key(cache.keyscore_out, h_selected,
                             cache.scores_a,
                             cache.keys_a,
                             stream);

    cache.ensure_sort_score_temp(h_selected);
    launch_sort_score_u64_desc(cache.cub_temp, cache.cub_temp_bytes,
                               cache.scores_a, cache.scores_b,
                               cache.keys_a,
                               cache.keys_b,
                               h_selected, stream);

    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    double* out_scores = nullptr;
    cudaMalloc(&out_first, (size_t)k * sizeof(int32_t));
    cudaMalloc(&out_second, (size_t)k * sizeof(int32_t));
    cudaMalloc(&out_scores, (size_t)k * sizeof(double));

    launch_unpack_topk(cache.scores_b,
                       cache.keys_b,
                       k,
                       out_first, out_second, out_scores,
                       stream);

    return {out_first, out_second, out_scores, (std::size_t)k};
  }

  
  int32_t* out_first = nullptr;
  int32_t* out_second = nullptr;
  double* out_scores = nullptr;
  cudaMalloc(&out_first, (size_t)h_selected * sizeof(int32_t));
  cudaMalloc(&out_second, (size_t)h_selected * sizeof(int32_t));
  cudaMalloc(&out_scores, (size_t)h_selected * sizeof(double));

  launch_unpack_keyscore(cache.keyscore_out, h_selected,
                         out_first, out_second, out_scores,
                         stream);

  return {out_first, out_second, out_scores, (std::size_t)h_selected};
}

}  
