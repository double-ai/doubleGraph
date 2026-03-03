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
#include <cmath>
#include <optional>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/tuple.h>

namespace aai {

namespace {

struct Cache : Cacheable {
    Cache() {
        cudaMemPool_t pool;
        cudaDeviceGetDefaultMemPool(&pool, 0);
        uint64_t thr = UINT64_MAX;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr);
    }
    ~Cache() override = default;
};





__global__ void count_candidates_kernel(const int32_t* __restrict__ offsets,
                                        const int32_t* __restrict__ indices,
                                        const int32_t* __restrict__ seeds,
                                        int num_seeds,
                                        int64_t* __restrict__ counts)
{
  int sid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (sid >= num_seeds) return;
  int32_t u = seeds[sid];
  int64_t c = 0;
  int32_t u_start = offsets[u];
  int32_t u_end = offsets[u + 1];
  for (int32_t i = u_start; i < u_end; ++i) {
    int32_t k = indices[i];
    c += (int64_t)(offsets[k + 1] - offsets[k]);
  }
  counts[sid] = c;
}


__global__ void write_candidates_i64_kernel(const int32_t* __restrict__ offsets,
                                            const int32_t* __restrict__ indices,
                                            const int32_t* __restrict__ seeds,
                                            int num_seeds,
                                            const int64_t* __restrict__ seed_offsets,
                                            int64_t* __restrict__ out_keys)
{
  int sid = (int)blockIdx.x;
  if (sid >= num_seeds) return;
  int32_t u = seeds[sid];
  int32_t u_start = offsets[u];
  int32_t u_end = offsets[u + 1];
  int64_t base = seed_offsets[sid];
  int64_t write_pos = 0;

  for (int32_t i = u_start; i < u_end; ++i) {
    int32_t k = indices[i];
    int32_t ks = offsets[k];
    int32_t ke = offsets[k + 1];
    int32_t kd = ke - ks;
    for (int32_t t = (int32_t)threadIdx.x; t < kd; t += (int32_t)blockDim.x) {
      int32_t v = indices[ks + t];
      out_keys[base + write_pos + t] = (int64_t(uint32_t(u)) << 32) | int64_t(uint32_t(v));
    }
    write_pos += kd;
  }
}


__global__ void write_candidates_u32_kernel(const int32_t* __restrict__ offsets,
                                            const int32_t* __restrict__ indices,
                                            const int32_t* __restrict__ seeds,
                                            int num_seeds,
                                            const int64_t* __restrict__ seed_offsets,
                                            int vbits,
                                            uint32_t* __restrict__ out_keys)
{
  int sid = (int)blockIdx.x;
  if (sid >= num_seeds) return;
  int32_t u = seeds[sid];
  int32_t u_start = offsets[u];
  int32_t u_end = offsets[u + 1];
  int64_t base = seed_offsets[sid];
  int64_t write_pos = 0;
  uint32_t sid_shifted = (uint32_t)sid << vbits;

  for (int32_t i = u_start; i < u_end; ++i) {
    int32_t k = indices[i];
    int32_t ks = offsets[k];
    int32_t ke = offsets[k + 1];
    int32_t kd = ke - ks;
    for (int32_t t = (int32_t)threadIdx.x; t < kd; t += (int32_t)blockDim.x) {
      uint32_t v = (uint32_t)indices[ks + t];
      out_keys[(size_t)(base + write_pos + t)] = sid_shifted | v;
    }
    write_pos += kd;
  }
}

__global__ void unpack_keys_i64_kernel(const int64_t* __restrict__ keys,
                                       int64_t n,
                                       int32_t* __restrict__ first,
                                       int32_t* __restrict__ second)
{
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t k = (uint64_t)keys[i];
  first[i] = (int32_t)(k >> 32);
  second[i] = (int32_t)(k & 0xFFFFFFFFu);
}

__global__ void unpack_keys_u32_kernel(const uint32_t* __restrict__ keys,
                                       int64_t n,
                                       const int32_t* __restrict__ seeds,
                                       int vbits,
                                       int32_t* __restrict__ first,
                                       int32_t* __restrict__ second)
{
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t k = keys[i];
  uint32_t seed_idx = k >> vbits;
  uint32_t mask = (vbits == 32) ? 0xFFFFFFFFu : ((1u << vbits) - 1u);
  uint32_t v = k & mask;
  first[i] = seeds[seed_idx];
  second[i] = (int32_t)v;
}

__global__ void fill_sequence_kernel(int32_t* __restrict__ out, int n)
{
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < n) out[i] = i;
}





#ifndef SMEM_MAX_DEGREE
#define SMEM_MAX_DEGREE 1024
#endif

__device__ __forceinline__ int binary_search_idx(const int32_t* __restrict__ arr,
                                                 int len,
                                                 int32_t target)
{
  int lo = 0;
  int hi = len - 1;
  while (lo <= hi) {
    int mid = (lo + hi) >> 1;
    int32_t v = arr[mid];
    if (v == target) return mid;
    if (v < target)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return -1;
}

__global__ void cosine_sim_hybrid_kernel(const int32_t* __restrict__ offsets,
                                         const int32_t* __restrict__ indices,
                                         const double* __restrict__ weights,
                                         const int32_t* __restrict__ pair_first,
                                         const int32_t* __restrict__ pair_second,
                                         int64_t num_pairs,
                                         double* __restrict__ scores)
{
  int warps_per_block = (int)(blockDim.x >> 5);
  int warp_id = (int)(threadIdx.x >> 5);
  int lane = (int)(threadIdx.x & 31);
  int64_t block_pair_start = (int64_t)blockIdx.x * warps_per_block;
  if (block_pair_start >= num_pairs) return;

  int32_t first_u = pair_first[block_pair_start];
  int64_t last_idx = block_pair_start + warps_per_block - 1;
  if (last_idx >= num_pairs) last_idx = num_pairs - 1;
  int32_t last_u = pair_first[last_idx];
  bool same_u = (first_u == last_u);

  extern __shared__ __align__(16) unsigned char smem[];
  double* cached_wt = reinterpret_cast<double*>(smem);
  int32_t* cached_idx = reinterpret_cast<int32_t*>(cached_wt + SMEM_MAX_DEGREE);
  int32_t cached_u = -1;
  int32_t cached_deg = 0;

  if (same_u) {
    int32_t u_start = offsets[first_u];
    int32_t u_end = offsets[first_u + 1];
    int32_t u_deg = u_end - u_start;
    if (u_deg <= SMEM_MAX_DEGREE) {
      cached_u = first_u;
      cached_deg = u_deg;
      for (int i = (int)threadIdx.x; i < u_deg; i += (int)blockDim.x) {
        cached_idx[i] = indices[u_start + i];
        cached_wt[i] = weights[u_start + i];
      }
    }
  }
  __syncthreads();

  int64_t pair_idx = block_pair_start + warp_id;
  if (pair_idx >= num_pairs) return;

  int32_t u = pair_first[pair_idx];
  int32_t v = pair_second[pair_idx];
  int32_t u_start = offsets[u];
  int32_t u_end = offsets[u + 1];
  int32_t v_start = offsets[v];
  int32_t v_end = offsets[v + 1];
  int32_t u_deg = u_end - u_start;
  int32_t v_deg = v_end - v_start;

  double dot = 0.0;
  double nu_sq = 0.0;
  double nv_sq = 0.0;

  bool use_cached = (u == cached_u) && (cached_deg > 0);
  if (use_cached) {
    if (u_deg <= v_deg) {
      for (int t = lane; t < u_deg; t += 32) {
        int32_t k = cached_idx[t];
        double wu = cached_wt[t];
        int pos = binary_search_idx(indices + v_start, v_deg, k);
        if (pos >= 0) {
          double wv = weights[v_start + pos];
          dot += wu * wv;
          nu_sq += wu * wu;
          nv_sq += wv * wv;
        }
      }
    } else {
      for (int t = lane; t < v_deg; t += 32) {
        int32_t k = indices[v_start + t];
        double wv = weights[v_start + t];
        int pos = binary_search_idx(cached_idx, u_deg, k);
        if (pos >= 0) {
          double wu = cached_wt[pos];
          dot += wu * wv;
          nu_sq += wu * wu;
          nv_sq += wv * wv;
        }
      }
    }
  } else {
    const int32_t* s_idx;
    const double* s_wt;
    int s_len;
    const int32_t* l_idx;
    const double* l_wt;
    int l_len;
    bool u_is_small;
    if (u_deg <= v_deg) {
      s_idx = indices + u_start;
      s_wt = weights + u_start;
      s_len = u_deg;
      l_idx = indices + v_start;
      l_wt = weights + v_start;
      l_len = v_deg;
      u_is_small = true;
    } else {
      s_idx = indices + v_start;
      s_wt = weights + v_start;
      s_len = v_deg;
      l_idx = indices + u_start;
      l_wt = weights + u_start;
      l_len = u_deg;
      u_is_small = false;
    }

    for (int t = lane; t < s_len; t += 32) {
      int32_t k = s_idx[t];
      double ws = s_wt[t];
      int pos = binary_search_idx(l_idx, l_len, k);
      if (pos >= 0) {
        double wl = l_wt[pos];
        double wu = u_is_small ? ws : wl;
        double wv = u_is_small ? wl : ws;
        dot += wu * wv;
        nu_sq += wu * wu;
        nv_sq += wv * wv;
      }
    }
  }

  for (int off = 16; off > 0; off >>= 1) {
    dot += __shfl_down_sync(0xffffffffu, dot, off);
    nu_sq += __shfl_down_sync(0xffffffffu, nu_sq, off);
    nv_sq += __shfl_down_sync(0xffffffffu, nv_sq, off);
  }

  if (lane == 0) {
    double denom = sqrt(nu_sq) * sqrt(nv_sq);
    scores[pair_idx] = (denom > 0.0) ? (dot / denom) : 0.0;
  }
}





struct is_self_loop_i64 {
  __host__ __device__ bool operator()(int64_t key) const
  {
    uint32_t u = (uint32_t)((uint64_t)key >> 32);
    uint32_t v = (uint32_t)((uint64_t)key & 0xFFFFFFFFu);
    return u == v;
  }
};

struct is_self_loop_u32 {
  const int32_t* seeds;
  int vbits;

  __host__ __device__ bool operator()(uint32_t key) const
  {
    uint32_t seed_idx = key >> vbits;
    uint32_t mask = (vbits == 32) ? 0xFFFFFFFFu : ((1u << vbits) - 1u);
    uint32_t v = key & mask;
    uint32_t u = (uint32_t)seeds[seed_idx];
    return u == v;
  }
};

void thrust_inclusive_scan_int64(int64_t* in, int64_t* out, int64_t n)
{
  thrust::inclusive_scan(thrust::cuda::par_nosync,
                         thrust::device_ptr<int64_t>(in),
                         thrust::device_ptr<int64_t>(in) + n,
                         thrust::device_ptr<int64_t>(out));
}

void thrust_sort_i64(int64_t* data, int64_t n)
{
  thrust::sort(thrust::cuda::par_nosync,
               thrust::device_ptr<int64_t>(data),
               thrust::device_ptr<int64_t>(data) + n);
}

int64_t thrust_unique_i64(int64_t* data, int64_t n)
{
  auto p = thrust::device_ptr<int64_t>(data);
  return (int64_t)(thrust::unique(thrust::cuda::par_nosync, p, p + n) - p);
}

int64_t thrust_remove_self_loops_i64(int64_t* data, int64_t n)
{
  auto p = thrust::device_ptr<int64_t>(data);
  return (int64_t)(thrust::remove_if(thrust::cuda::par_nosync, p, p + n, is_self_loop_i64()) - p);
}

void thrust_sort_u32(uint32_t* data, int64_t n)
{
  thrust::sort(thrust::cuda::par_nosync,
               thrust::device_ptr<uint32_t>(data),
               thrust::device_ptr<uint32_t>(data) + n);
}

int64_t thrust_unique_u32(uint32_t* data, int64_t n)
{
  auto p = thrust::device_ptr<uint32_t>(data);
  return (int64_t)(thrust::unique(thrust::cuda::par_nosync, p, p + n) - p);
}

int64_t thrust_remove_self_loops_u32(uint32_t* data, int64_t n, const int32_t* seeds, int vbits)
{
  auto p = thrust::device_ptr<uint32_t>(data);
  is_self_loop_u32 pred{seeds, vbits};
  return (int64_t)(thrust::remove_if(thrust::cuda::par_nosync, p, p + n, pred) - p);
}

void thrust_sort_pairs_by_score_desc(double* scores, int32_t* first, int32_t* second, int64_t n)
{
  auto sp = thrust::device_ptr<double>(scores);
  auto fp = thrust::device_ptr<int32_t>(first);
  auto sp2 = thrust::device_ptr<int32_t>(second);
  auto zip = thrust::make_zip_iterator(thrust::make_tuple(fp, sp2));
  thrust::sort_by_key(thrust::cuda::par_nosync, sp, sp + n, zip, thrust::greater<double>());
}





void launch_count_candidates(const int32_t* offsets,
                             const int32_t* indices,
                             const int32_t* seeds,
                             int num_seeds,
                             int64_t* counts)
{
  if (num_seeds <= 0) return;
  int threads = 256;
  int blocks = (num_seeds + threads - 1) / threads;
  count_candidates_kernel<<<blocks, threads>>>(offsets, indices, seeds, num_seeds, counts);
}

void launch_write_candidates_i64(const int32_t* offsets,
                                 const int32_t* indices,
                                 const int32_t* seeds,
                                 int num_seeds,
                                 const int64_t* seed_offsets,
                                 int64_t* out_keys)
{
  if (num_seeds <= 0) return;
  int threads = 256;
  write_candidates_i64_kernel<<<num_seeds, threads>>>(offsets, indices, seeds, num_seeds, seed_offsets, out_keys);
}

void launch_write_candidates_u32(const int32_t* offsets,
                                 const int32_t* indices,
                                 const int32_t* seeds,
                                 int num_seeds,
                                 const int64_t* seed_offsets,
                                 int vbits,
                                 uint32_t* out_keys)
{
  if (num_seeds <= 0) return;
  int threads = 256;
  write_candidates_u32_kernel<<<num_seeds, threads>>>(offsets, indices, seeds, num_seeds, seed_offsets, vbits, out_keys);
}

void launch_unpack_keys_i64(const int64_t* keys, int64_t n, int32_t* first, int32_t* second)
{
  if (n <= 0) return;
  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);
  unpack_keys_i64_kernel<<<blocks, threads>>>(keys, n, first, second);
}

void launch_unpack_keys_u32(const uint32_t* keys,
                            int64_t n,
                            const int32_t* seeds,
                            int vbits,
                            int32_t* first,
                            int32_t* second)
{
  if (n <= 0) return;
  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);
  unpack_keys_u32_kernel<<<blocks, threads>>>(keys, n, seeds, vbits, first, second);
}

void launch_cosine_sim_hybrid(const int32_t* offsets,
                              const int32_t* indices,
                              const double* weights,
                              const int32_t* pair_first,
                              const int32_t* pair_second,
                              int64_t num_pairs,
                              double* scores)
{
  if (num_pairs <= 0) return;
  constexpr int warps_per_block = 8;
  int threads = warps_per_block * 32;
  int blocks = (int)((num_pairs + warps_per_block - 1) / warps_per_block);
  int smem_bytes = SMEM_MAX_DEGREE * (int)(sizeof(double) + sizeof(int32_t));
  cosine_sim_hybrid_kernel<<<blocks, threads, smem_bytes>>>(offsets, indices, weights, pair_first, pair_second, num_pairs, scores);
}

int ilog2_ceil_u32(uint32_t x)
{
  if (x <= 1) return 0;
  return 32 - __builtin_clz(x - 1);
}

}  

similarity_result_double_t cosine_all_pairs_similarity(const graph32_t& graph,
                                                       const double* edge_weights,
                                                       const int32_t* vertices,
                                                       std::size_t num_vertices,
                                                       std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_verts = graph.number_of_vertices;
    const double* d_weights = edge_weights;

    
    int num_seeds = 0;
    int32_t* d_seeds_buf = nullptr;
    const int32_t* d_seeds = nullptr;

    if (vertices != nullptr && num_vertices > 0) {
        num_seeds = (int)num_vertices;
        d_seeds = vertices;
    } else {
        num_seeds = num_verts;
        cudaMalloc(&d_seeds_buf, (size_t)num_verts * sizeof(int32_t));
        int threads = 256;
        int blocks = (num_verts + threads - 1) / threads;
        fill_sequence_kernel<<<blocks, threads>>>(d_seeds_buf, num_verts);
        d_seeds = d_seeds_buf;
    }

    if (num_seeds <= 0 || num_verts <= 0) {
        if (d_seeds_buf) cudaFree(d_seeds_buf);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t* d_counts;
    cudaMalloc(&d_counts, (size_t)num_seeds * sizeof(int64_t));
    launch_count_candidates(d_offsets, d_indices, d_seeds, num_seeds, d_counts);

    
    int64_t* d_seed_off;
    cudaMalloc(&d_seed_off, (size_t)(num_seeds + 1) * sizeof(int64_t));
    cudaMemset(d_seed_off, 0, sizeof(int64_t));
    thrust_inclusive_scan_int64(d_counts, d_seed_off + 1, num_seeds);

    int64_t total_candidates = 0;
    cudaMemcpy(&total_candidates, d_seed_off + num_seeds, sizeof(int64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_counts);

    if (total_candidates <= 0) {
        cudaFree(d_seed_off);
        if (d_seeds_buf) cudaFree(d_seeds_buf);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int vbits = ilog2_ceil_u32((uint32_t)num_verts);
    bool use_u32_keys = false;
    if (vbits > 0 && vbits < 32) {
        int seed_bits_cap = 32 - vbits;
        if (seed_bits_cap > 0 && num_seeds <= (int)(1u << seed_bits_cap)) {
            use_u32_keys = true;
        }
    }

    int64_t num_pairs = 0;
    int32_t* d_first = nullptr;
    int32_t* d_second = nullptr;

    if (use_u32_keys) {
        
        uint32_t* d_keys;
        cudaMalloc(&d_keys, (size_t)total_candidates * sizeof(uint32_t));
        launch_write_candidates_u32(d_offsets, d_indices, d_seeds, num_seeds, d_seed_off, vbits, d_keys);

        
        thrust_sort_u32(d_keys, total_candidates);
        int64_t num_unique = thrust_unique_u32(d_keys, total_candidates);
        num_pairs = thrust_remove_self_loops_u32(d_keys, num_unique, d_seeds, vbits);

        if (num_pairs <= 0) {
            cudaFree(d_keys);
            cudaFree(d_seed_off);
            if (d_seeds_buf) cudaFree(d_seeds_buf);
            return {nullptr, nullptr, nullptr, 0};
        }

        
        cudaMalloc(&d_first, (size_t)num_pairs * sizeof(int32_t));
        cudaMalloc(&d_second, (size_t)num_pairs * sizeof(int32_t));
        launch_unpack_keys_u32(d_keys, num_pairs, d_seeds, vbits, d_first, d_second);
        cudaFree(d_keys);
    } else {
        
        int64_t* d_keys;
        cudaMalloc(&d_keys, (size_t)total_candidates * sizeof(int64_t));
        launch_write_candidates_i64(d_offsets, d_indices, d_seeds, num_seeds, d_seed_off, d_keys);

        thrust_sort_i64(d_keys, total_candidates);
        int64_t num_unique = thrust_unique_i64(d_keys, total_candidates);
        num_pairs = thrust_remove_self_loops_i64(d_keys, num_unique);

        if (num_pairs <= 0) {
            cudaFree(d_keys);
            cudaFree(d_seed_off);
            if (d_seeds_buf) cudaFree(d_seeds_buf);
            return {nullptr, nullptr, nullptr, 0};
        }

        cudaMalloc(&d_first, (size_t)num_pairs * sizeof(int32_t));
        cudaMalloc(&d_second, (size_t)num_pairs * sizeof(int32_t));
        launch_unpack_keys_i64(d_keys, num_pairs, d_first, d_second);
        cudaFree(d_keys);
    }

    cudaFree(d_seed_off);
    if (d_seeds_buf) cudaFree(d_seeds_buf);

    
    double* d_scores;
    cudaMalloc(&d_scores, (size_t)num_pairs * sizeof(double));
    launch_cosine_sim_hybrid(d_offsets, d_indices, d_weights, d_first, d_second, num_pairs, d_scores);

    
    if (topk.has_value() && (int64_t)topk.value() < num_pairs) {
        int64_t k = (int64_t)topk.value();
        thrust_sort_pairs_by_score_desc(d_scores, d_first, d_second, num_pairs);

        int32_t* d_first_o;
        int32_t* d_second_o;
        double* d_scores_o;
        cudaMalloc(&d_first_o, (size_t)k * sizeof(int32_t));
        cudaMalloc(&d_second_o, (size_t)k * sizeof(int32_t));
        cudaMalloc(&d_scores_o, (size_t)k * sizeof(double));
        cudaMemcpy(d_first_o, d_first, (size_t)k * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_second_o, d_second, (size_t)k * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_scores_o, d_scores, (size_t)k * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaFree(d_first);
        cudaFree(d_second);
        cudaFree(d_scores);
        return {d_first_o, d_second_o, d_scores_o, (std::size_t)k};
    }

    return {d_first, d_second, d_scores, (std::size_t)num_pairs};
}

}  
