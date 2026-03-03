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
  int64_t* h_scalars = nullptr;

  Cache() {
    cudaMallocHost(&h_scalars, 2 * sizeof(int64_t));
  }

  ~Cache() override {
    if (h_scalars) {
      cudaFreeHost(h_scalars);
      h_scalars = nullptr;
    }
  }
};





__device__ __forceinline__ int32_t lower_bound_int32(const int32_t* __restrict__ a,
                                                     int32_t n,
                                                     int32_t x)
{
  int32_t l = 0, r = n;
  while (l < r) {
    int32_t m = (l + r) >> 1;
    int32_t v = a[m];
    if (v < x) {
      l = m + 1;
    } else {
      r = m;
    }
  }
  return l;
}

__device__ __forceinline__ int64_t pack_pair_u32(int32_t u, int32_t v)
{
  return static_cast<int64_t>((uint64_t(uint32_t(u)) << 32) | uint32_t(v));
}

__device__ __forceinline__ int32_t unpack_u(int64_t key)
{
  return int32_t(uint32_t(uint64_t(key) >> 32));
}
__device__ __forceinline__ int32_t unpack_v(int64_t key)
{
  return int32_t(uint32_t(uint64_t(key)));
}





__global__ void k_fill_sequence_int32(int32_t* out, int32_t n)
{
  int32_t idx = int32_t(blockIdx.x) * int32_t(blockDim.x) + int32_t(threadIdx.x);
  if (idx < n) out[idx] = idx;
}

__global__ void k_compute_deg_for_vertices(const int32_t* __restrict__ vertices,
                                           int32_t n,
                                           const int32_t* __restrict__ offsets,
                                           int32_t* __restrict__ deg_out)
{
  int32_t i = int32_t(blockIdx.x) * int32_t(blockDim.x) + int32_t(threadIdx.x);
  if (i < n) {
    int32_t v = vertices[i];
    deg_out[i] = offsets[v + 1] - offsets[v];
  }
}

__global__ void k_set_last_int32(int32_t* a, int64_t idx, int32_t val)
{
  if (blockIdx.x == 0 && threadIdx.x == 0) { a[idx] = val; }
}

__global__ void k_fill_seed_edge_tasks(const int32_t* __restrict__ seeds,
                                       int32_t S,
                                       const int64_t* __restrict__ seed_edge_offsets,
                                       const int32_t* __restrict__ offsets,
                                       const int32_t* __restrict__ indices,
                                       int32_t* __restrict__ task_seed,
                                       int32_t* __restrict__ task_w)
{
  constexpr int WARPS_PER_BLOCK = 8;
  int warp_id = int(threadIdx.x) >> 5;
  int lane    = int(threadIdx.x) & 31;
  int global_warp = int(blockIdx.x) * WARPS_PER_BLOCK + warp_id;
  if (global_warp >= S) return;

  int32_t seed_idx = global_warp;
  int32_t u = seeds[seed_idx];
  int32_t row_start = offsets[u];
  int32_t row_end   = offsets[u + 1];
  int32_t deg_u     = row_end - row_start;
  int64_t out_base  = seed_edge_offsets[seed_idx];

  for (int32_t j = lane; j < deg_u; j += 32) {
    int32_t w = indices[row_start + j];
    int64_t pos = out_base + int64_t(j);
    task_seed[pos] = seed_idx;
    task_w[pos] = w;
  }
}

__global__ void k_compute_deg_for_tasks(const int32_t* __restrict__ task_w,
                                        int64_t n_tasks,
                                        const int32_t* __restrict__ offsets,
                                        int32_t* __restrict__ deg_w)
{
  int64_t t = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if (t < n_tasks) {
    int32_t w = task_w[t];
    deg_w[t] = offsets[w + 1] - offsets[w];
  }
}

__global__ void k_fill_twohop_keys(const int32_t* __restrict__ task_seed,
                                  const int32_t* __restrict__ task_w,
                                  int64_t n_tasks,
                                  const int64_t* __restrict__ task_out_offsets,
                                  const int32_t* __restrict__ offsets,
                                  const int32_t* __restrict__ indices,
                                  int64_t* __restrict__ out_keys)
{
  constexpr int WARPS_PER_BLOCK = 8;
  int warp_id = int(threadIdx.x) >> 5;
  int lane    = int(threadIdx.x) & 31;
  int64_t global_warp = int64_t(blockIdx.x) * WARPS_PER_BLOCK + int64_t(warp_id);
  if (global_warp >= n_tasks) return;

  int32_t seed_idx = task_seed[global_warp];
  int32_t w = task_w[global_warp];
  int32_t row_start = offsets[w];
  int32_t row_end   = offsets[w + 1];
  int32_t deg_w     = row_end - row_start;
  int64_t out_base  = task_out_offsets[global_warp];

  for (int32_t j = lane; j < deg_w; j += 32) {
    int32_t v = indices[row_start + j];
    out_keys[out_base + int64_t(j)] = pack_pair_u32(v, seed_idx);
  }
}

__global__ void k_predicate_not_self(const int64_t* __restrict__ in,
                                     const int32_t* __restrict__ seeds,
                                     uint8_t* __restrict__ flags,
                                     int64_t n)
{
  int64_t i = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if (i < n) {
    int64_t key = in[i];
    int32_t v = unpack_u(key);
    int32_t seed_idx = unpack_v(key);
    int32_t u = seeds[seed_idx];
    flags[i] = (u != v) ? uint8_t{1} : uint8_t{0};
  }
}

__global__ void k_extract_v_from_keys(const int64_t* __restrict__ keys,
                                     int64_t n,
                                     int32_t* __restrict__ v_out)
{
  int64_t i = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if (i < n) {
    v_out[i] = unpack_u(keys[i]);
  }
}


__global__ void k_compute_weight_sums_for_unique_verts(const int32_t* __restrict__ verts,
                                                      int32_t n,
                                                      const int32_t* __restrict__ offsets,
                                                      const double* __restrict__ weights,
                                                      double* __restrict__ out_sums)
{
  constexpr int WARPS_PER_BLOCK = 8;
  int warp_id = int(threadIdx.x) >> 5;
  int lane    = int(threadIdx.x) & 31;
  int global_warp = int(blockIdx.x) * WARPS_PER_BLOCK + warp_id;
  if (global_warp >= n) return;

  int32_t v = verts[global_warp];
  int32_t row_start = offsets[v];
  int32_t row_end   = offsets[v + 1];
  double sum = 0.0;
  for (int32_t i = row_start + lane; i < row_end; i += 32) {
    sum += weights[i];
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  if (lane == 0) out_sums[global_warp] = sum;
}

__global__ void k_compute_scores(const int64_t* __restrict__ keys,
                                int64_t n_pairs,
                                const int32_t* __restrict__ seeds,
                                const double* __restrict__ seed_wsum,
                                const int32_t* __restrict__ unique_v,
                                int32_t n_unique_v,
                                const double* __restrict__ unique_v_wsum,
                                const int32_t* __restrict__ offsets,
                                const int32_t* __restrict__ indices,
                                const double* __restrict__ weights,
                                double* __restrict__ out_scores)
{
  int64_t i = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if (i >= n_pairs) return;
  int64_t key = keys[i];
  int32_t v = unpack_u(key);
  int32_t seed_idx = unpack_v(key);
  int32_t u = seeds[seed_idx];

  double wu = seed_wsum[seed_idx];
  int32_t iv = lower_bound_int32(unique_v, n_unique_v, v);
  double wv = unique_v_wsum[iv];
  double denom = wu + wv;

  int32_t u0 = offsets[u];
  int32_t u1 = offsets[u + 1];
  int32_t v0 = offsets[v];
  int32_t v1 = offsets[v + 1];

  int32_t pu = u0;
  int32_t pv = v0;
  double numer = 0.0;

  while (pu < u1 && pv < v1) {
    int32_t a = indices[pu];
    int32_t b = indices[pv];
    if (a < b) {
      ++pu;
    } else if (b < a) {
      ++pv;
    } else {
      double wa = weights[pu];
      double wb = weights[pv];
      numer += (wa < wb) ? wa : wb;
      ++pu;
      ++pv;
    }
  }

  out_scores[i] = (denom <= 2.2250738585072014e-308) ? 0.0 : (2.0 * numer) / denom;
}

__global__ void k_write_outputs(const int64_t* __restrict__ keys,
                               const double* __restrict__ scores,
                               int64_t n,
                               const int32_t* __restrict__ seeds,
                               int32_t* __restrict__ out_first,
                               int32_t* __restrict__ out_second,
                               double* __restrict__ out_scores)
{
  int64_t i = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if (i < n) {
    int64_t key = keys[i];
    int32_t v = unpack_u(key);
    int32_t seed_idx = unpack_v(key);
    out_first[i] = seeds[seed_idx];
    out_second[i] = v;
    out_scores[i] = scores[i];
  }
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
  int64_t* h_scalars = cache.h_scalars;

  const int32_t* d_offsets = graph.offsets;
  const int32_t* d_indices = graph.indices;
  int32_t num_verts = graph.number_of_vertices;
  const double* d_weights = edge_weights;

  cudaStream_t stream = 0;

  int64_t topk_raw = topk.has_value() ? static_cast<int64_t>(topk.value()) : -1;

  
  int32_t* d_seeds_alloc = nullptr;
  const int32_t* d_seeds;
  int32_t S;
  if (vertices != nullptr && num_vertices > 0) {
    d_seeds = vertices;
    S = static_cast<int32_t>(num_vertices);
  } else {
    S = num_verts;
    cudaMalloc(&d_seeds_alloc, int64_t(S) * sizeof(int32_t));
    int threads = 256;
    int blocks = (S + threads - 1) / threads;
    k_fill_sequence_int32<<<blocks, threads, 0, stream>>>(d_seeds_alloc, S);
    d_seeds = d_seeds_alloc;
  }

  
  double* d_seed_wsum = nullptr;
  cudaMalloc(&d_seed_wsum, int64_t(S) * sizeof(double));
  {
    int threads = 256;
    int blocks = (S + 8 - 1) / 8;
    k_compute_weight_sums_for_unique_verts<<<blocks, threads, 0, stream>>>(
        d_seeds, S, d_offsets, d_weights, d_seed_wsum);
  }

  
  int32_t* d_deg_u = nullptr;
  cudaMalloc(&d_deg_u, int64_t(S + 1) * sizeof(int32_t));
  {
    int threads = 256;
    int blocks = (S + threads - 1) / threads;
    k_compute_deg_for_vertices<<<blocks, threads, 0, stream>>>(d_seeds, S, d_offsets, d_deg_u);
    k_set_last_int32<<<1, 1, 0, stream>>>(d_deg_u, S, 0);
  }

  
  int64_t* d_seed_edge_offsets = nullptr;
  cudaMalloc(&d_seed_edge_offsets, int64_t(S + 1) * sizeof(int64_t));
  {
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_deg_u, d_seed_edge_offsets, S + 1, stream);
    void* d_temp = nullptr;
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_deg_u, d_seed_edge_offsets, S + 1, stream);
    cudaFree(d_temp);
  }
  cudaFree(d_deg_u);

  
  cudaMemcpyAsync(h_scalars + 1, d_seed_edge_offsets + S, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  int64_t total_seed_edges = h_scalars[1];

  if (total_seed_edges == 0 || topk_raw == 0) {
    if (d_seeds_alloc) cudaFree(d_seeds_alloc);
    cudaFree(d_seed_wsum);
    cudaFree(d_seed_edge_offsets);
    return {nullptr, nullptr, nullptr, 0};
  }

  
  int32_t* d_task_seed = nullptr;
  int32_t* d_task_w = nullptr;
  cudaMalloc(&d_task_seed, total_seed_edges * sizeof(int32_t));
  cudaMalloc(&d_task_w, total_seed_edges * sizeof(int32_t));

  
  {
    int threads = 256;
    int blocks = (S + 8 - 1) / 8;
    k_fill_seed_edge_tasks<<<blocks, threads, 0, stream>>>(
        d_seeds, S, d_seed_edge_offsets, d_offsets, d_indices,
        d_task_seed, d_task_w);
  }
  cudaFree(d_seed_edge_offsets);

  
  int32_t* d_deg_w = nullptr;
  cudaMalloc(&d_deg_w, (total_seed_edges + 1) * sizeof(int32_t));
  {
    int threads = 256;
    int blocks = int((total_seed_edges + threads - 1) / threads);
    k_compute_deg_for_tasks<<<blocks, threads, 0, stream>>>(d_task_w, total_seed_edges, d_offsets, d_deg_w);
    k_set_last_int32<<<1, 1, 0, stream>>>(d_deg_w, total_seed_edges, 0);
  }

  
  int64_t* d_task_out_offsets = nullptr;
  cudaMalloc(&d_task_out_offsets, (total_seed_edges + 1) * sizeof(int64_t));
  {
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_deg_w, d_task_out_offsets, total_seed_edges + 1, stream);
    void* d_temp = nullptr;
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_deg_w, d_task_out_offsets, total_seed_edges + 1, stream);
    cudaFree(d_temp);
  }
  cudaFree(d_deg_w);

  cudaMemcpyAsync(h_scalars + 1, d_task_out_offsets + total_seed_edges, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  int64_t total_paths = h_scalars[1];

  if (total_paths == 0) {
    if (d_seeds_alloc) cudaFree(d_seeds_alloc);
    cudaFree(d_seed_wsum);
    cudaFree(d_task_seed);
    cudaFree(d_task_w);
    cudaFree(d_task_out_offsets);
    return {nullptr, nullptr, nullptr, 0};
  }

  
  int64_t* d_raw_keys = nullptr;
  cudaMalloc(&d_raw_keys, total_paths * sizeof(int64_t));

  {
    int threads = 256;
    int blocks = int((total_seed_edges + 8 - 1) / 8);
    k_fill_twohop_keys<<<blocks, threads, 0, stream>>>(
        d_task_seed, d_task_w, total_seed_edges,
        d_task_out_offsets, d_offsets, d_indices, d_raw_keys);
  }
  cudaFree(d_task_seed);
  cudaFree(d_task_w);
  cudaFree(d_task_out_offsets);

  
  int64_t* d_keys_tmp = nullptr;
  cudaMalloc(&d_keys_tmp, total_paths * sizeof(int64_t));

  cub::DoubleBuffer<int64_t> d_keys(d_raw_keys, d_keys_tmp);
  {
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, d_keys, total_paths, 0, 64, stream);
    void* d_temp = nullptr;
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_keys, total_paths, 0, 64, stream);
    cudaFree(d_temp);
  }
  int64_t* d_sorted_keys = d_keys.Current();

  
  int64_t* d_unique_keys = nullptr;
  int64_t* d_num_unique = nullptr;
  cudaMalloc(&d_unique_keys, total_paths * sizeof(int64_t));
  cudaMalloc(&d_num_unique, sizeof(int64_t));

  {
    size_t temp_bytes = 0;
    cub::DeviceSelect::Unique(nullptr, temp_bytes, d_sorted_keys, d_unique_keys, d_num_unique, total_paths, stream);
    void* d_temp = nullptr;
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceSelect::Unique(d_temp, temp_bytes, d_sorted_keys, d_unique_keys, d_num_unique, total_paths, stream);
    cudaFree(d_temp);
  }
  cudaFree(d_raw_keys);
  cudaFree(d_keys_tmp);

  cudaMemcpyAsync(h_scalars + 1, d_num_unique, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  int64_t num_unique_pairs_including_self = h_scalars[1];
  cudaFree(d_num_unique);

  if (num_unique_pairs_including_self == 0) {
    if (d_seeds_alloc) cudaFree(d_seeds_alloc);
    cudaFree(d_seed_wsum);
    cudaFree(d_unique_keys);
    return {nullptr, nullptr, nullptr, 0};
  }

  
  uint8_t* d_flags = nullptr;
  cudaMalloc(&d_flags, num_unique_pairs_including_self * sizeof(uint8_t));
  {
    int threads = 256;
    int blocks = int((num_unique_pairs_including_self + threads - 1) / threads);
    k_predicate_not_self<<<blocks, threads, 0, stream>>>(
        d_unique_keys, d_seeds, d_flags, num_unique_pairs_including_self);
  }

  int64_t* d_filtered_keys = nullptr;
  int64_t* d_num_filtered = nullptr;
  cudaMalloc(&d_filtered_keys, num_unique_pairs_including_self * sizeof(int64_t));
  cudaMalloc(&d_num_filtered, sizeof(int64_t));

  {
    size_t temp_bytes = 0;
    cub::DeviceSelect::Flagged(nullptr, temp_bytes, d_unique_keys, d_flags, d_filtered_keys, d_num_filtered, num_unique_pairs_including_self, stream);
    void* d_temp = nullptr;
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceSelect::Flagged(d_temp, temp_bytes, d_unique_keys, d_flags, d_filtered_keys, d_num_filtered, num_unique_pairs_including_self, stream);
    cudaFree(d_temp);
  }
  cudaFree(d_unique_keys);
  cudaFree(d_flags);

  cudaMemcpyAsync(h_scalars + 1, d_num_filtered, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  int64_t num_pairs = h_scalars[1];
  cudaFree(d_num_filtered);

  if (num_pairs == 0) {
    if (d_seeds_alloc) cudaFree(d_seeds_alloc);
    cudaFree(d_seed_wsum);
    cudaFree(d_filtered_keys);
    return {nullptr, nullptr, nullptr, 0};
  }

  
  int32_t* d_v_list = nullptr;
  cudaMalloc(&d_v_list, num_pairs * sizeof(int32_t));
  {
    int threads = 256;
    int blocks = int((num_pairs + threads - 1) / threads);
    k_extract_v_from_keys<<<blocks, threads, 0, stream>>>(d_filtered_keys, num_pairs, d_v_list);
  }

  int32_t* d_sorted_v = d_v_list;

  int32_t* d_unique_v = nullptr;
  int64_t* d_num_unique_v = nullptr;
  cudaMalloc(&d_unique_v, num_pairs * sizeof(int32_t));
  cudaMalloc(&d_num_unique_v, sizeof(int64_t));
  {
    size_t temp_bytes = 0;
    cub::DeviceSelect::Unique(nullptr, temp_bytes, d_sorted_v, d_unique_v, d_num_unique_v, num_pairs, stream);
    void* d_temp = nullptr;
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceSelect::Unique(d_temp, temp_bytes, d_sorted_v, d_unique_v, d_num_unique_v, num_pairs, stream);
    cudaFree(d_temp);
  }
  cudaFree(d_v_list);

  cudaMemcpyAsync(h_scalars + 1, d_num_unique_v, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  int32_t num_unique_v = int32_t(h_scalars[1]);
  cudaFree(d_num_unique_v);

  
  double* d_unique_v_wsum = nullptr;
  cudaMalloc(&d_unique_v_wsum, int64_t(num_unique_v) * sizeof(double));
  {
    int threads = 256;
    int blocks = (num_unique_v + 8 - 1) / 8;
    k_compute_weight_sums_for_unique_verts<<<blocks, threads, 0, stream>>>(
        d_unique_v, num_unique_v, d_offsets, d_weights, d_unique_v_wsum);
  }

  
  double* d_scores = nullptr;
  cudaMalloc(&d_scores, num_pairs * sizeof(double));
  {
    int threads = 256;
    int blocks = int((num_pairs + threads - 1) / threads);
    k_compute_scores<<<blocks, threads, 0, stream>>>(
        d_filtered_keys,
        num_pairs,
        d_seeds,
        d_seed_wsum,
        d_unique_v,
        num_unique_v,
        d_unique_v_wsum,
        d_offsets,
        d_indices,
        d_weights,
        d_scores);
  }
  cudaFree(d_unique_v);
  cudaFree(d_unique_v_wsum);
  cudaFree(d_seed_wsum);

  int64_t* final_keys = d_filtered_keys;
  double* final_scores = d_scores;
  int64_t out_count = num_pairs;

  int64_t* d_sorted_keys_buf = nullptr;
  double* d_sorted_scores_buf = nullptr;
  if (topk_raw >= 0 && topk_raw < num_pairs) {
    cudaMalloc(&d_sorted_scores_buf, num_pairs * sizeof(double));
    cudaMalloc(&d_sorted_keys_buf, num_pairs * sizeof(int64_t));

    cub::DoubleBuffer<double> db_scores(d_scores, d_sorted_scores_buf);
    cub::DoubleBuffer<int64_t> db_vals(d_filtered_keys, d_sorted_keys_buf);

    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(nullptr, temp_bytes, db_scores, db_vals, num_pairs, 0, 64, stream);
    void* d_temp = nullptr;
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceRadixSort::SortPairsDescending(d_temp, temp_bytes, db_scores, db_vals, num_pairs, 0, 64, stream);
    cudaFree(d_temp);

    final_scores = db_scores.Current();
    final_keys = db_vals.Current();
    out_count = topk_raw;
  }

  
  int32_t* d_out_first = nullptr;
  int32_t* d_out_second = nullptr;
  double* d_out_scores = nullptr;
  cudaMalloc(&d_out_first, out_count * sizeof(int32_t));
  cudaMalloc(&d_out_second, out_count * sizeof(int32_t));
  cudaMalloc(&d_out_scores, out_count * sizeof(double));

  {
    int threads = 256;
    int blocks = int((out_count + threads - 1) / threads);
    k_write_outputs<<<blocks, threads, 0, stream>>>(
        final_keys,
        final_scores,
        out_count,
        d_seeds,
        d_out_first,
        d_out_second,
        d_out_scores);
  }

  
  if (d_seeds_alloc) cudaFree(d_seeds_alloc);
  cudaFree(d_filtered_keys);
  cudaFree(d_scores);
  if (d_sorted_keys_buf) cudaFree(d_sorted_keys_buf);
  if (d_sorted_scores_buf) cudaFree(d_sorted_scores_buf);

  return {d_out_first, d_out_second, d_out_scores, static_cast<std::size_t>(out_count)};
}

}  
