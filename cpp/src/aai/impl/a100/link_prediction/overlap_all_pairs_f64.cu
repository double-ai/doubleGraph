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

#include <cub/block/block_reduce.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/unique.h>

#include <cfloat>
#include <cstdint>
#include <optional>

namespace aai {

namespace {

struct Cache : Cacheable {
    double* seed_wdeg = nullptr;
    int64_t seed_wdeg_cap = 0;

    int64_t* raw_counts = nullptr;
    int64_t raw_counts_cap = 0;

    int64_t* pair_keys = nullptr;
    int64_t pair_keys_cap = 0;

    int32_t* temp_first = nullptr;
    int64_t temp_first_cap = 0;

    int32_t* temp_second = nullptr;
    int64_t temp_second_cap = 0;

    double* temp_scores = nullptr;
    int64_t temp_scores_cap = 0;

    ~Cache() override {
        if (seed_wdeg) cudaFree(seed_wdeg);
        if (raw_counts) cudaFree(raw_counts);
        if (pair_keys) cudaFree(pair_keys);
        if (temp_first) cudaFree(temp_first);
        if (temp_second) cudaFree(temp_second);
        if (temp_scores) cudaFree(temp_scores);
    }
};





__global__ void compute_seed_wdeg_kernel(const int32_t* __restrict__ offsets,
                                        const double* __restrict__ weights,
                                        const int32_t* __restrict__ seeds,
                                        int32_t num_seeds,
                                        double* __restrict__ seed_wdeg)
{
  int32_t sid = blockIdx.x * blockDim.x + threadIdx.x;
  if (sid >= num_seeds) return;
  int32_t u = seeds ? seeds[sid] : sid;
  int32_t s = offsets[u];
  int32_t e = offsets[u + 1];
  double sum = 0.0;
  for (int32_t i = s; i < e; ++i) sum += weights[i];
  seed_wdeg[sid] = sum;
}





__global__ void count_raw_pairs_kernel(const int32_t* __restrict__ offsets,
                                      const int32_t* __restrict__ indices,
                                      const int32_t* __restrict__ seeds,
                                      int32_t num_seeds,
                                      int64_t* __restrict__ counts)
{
  int32_t sid = blockIdx.x;
  if (sid >= num_seeds) return;
  int32_t u = seeds ? seeds[sid] : sid;
  int32_t u_s = offsets[u];
  int32_t u_e = offsets[u + 1];
  int32_t u_deg = u_e - u_s;

  int64_t local = 0;
  for (int32_t i = threadIdx.x; i < u_deg; i += blockDim.x) {
    int32_t w = indices[u_s + i];
    local += static_cast<int64_t>(offsets[w + 1] - offsets[w]);
  }

  using BlockReduce = cub::BlockReduce<int64_t, 256>;
  __shared__ typename BlockReduce::TempStorage tmp;
  int64_t total = BlockReduce(tmp).Sum(local);
  if (threadIdx.x == 0) counts[sid] = total;
}






__global__ void write_raw_pairs_kernel(const int32_t* __restrict__ offsets,
                                      const int32_t* __restrict__ indices,
                                      const int32_t* __restrict__ seeds,
                                      int32_t num_seeds,
                                      const int64_t* __restrict__ pair_offsets,
                                      int64_t* __restrict__ pair_keys)
{
  int32_t sid = blockIdx.x;
  if (sid >= num_seeds) return;
  int32_t u = seeds ? seeds[sid] : sid;
  int32_t u_s = offsets[u];
  int32_t u_e = offsets[u + 1];
  int32_t u_deg = u_e - u_s;

  __shared__ int32_t s_write;
  if (threadIdx.x == 0) s_write = 0;
  __syncthreads();

  int64_t base = pair_offsets[sid];
  for (int32_t ni = threadIdx.x; ni < u_deg; ni += blockDim.x) {
    int32_t w = indices[u_s + ni];
    int32_t w_s = offsets[w];
    int32_t w_e = offsets[w + 1];
    int32_t w_deg = w_e - w_s;
    int32_t my_off = atomicAdd(&s_write, w_deg);
    for (int32_t j = 0; j < w_deg; ++j) {
      int32_t v = indices[w_s + j];
      pair_keys[base + static_cast<int64_t>(my_off + j)] = (static_cast<int64_t>(u) << 32) |
                                                          static_cast<int64_t>(static_cast<uint32_t>(v));
    }
  }
}




__device__ __forceinline__ int32_t lower_bound_dev(const int32_t* __restrict__ arr,
                                                   int32_t size,
                                                   int32_t target)
{
  int32_t lo = 0;
  int32_t hi = size;
  while (lo < hi) {
    int32_t mid = lo + ((hi - lo) >> 1);
    int32_t v = arr[mid];
    if (v < target)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

__device__ __forceinline__ int32_t galloping_search(const int32_t* __restrict__ arr,
                                                    int32_t start,
                                                    int32_t end,
                                                    int32_t target)
{
  if (start >= end || arr[start] >= target) return start;
  int32_t pos = start;
  int32_t step = 1;
  while ((pos + step) < end && arr[pos + step] < target) {
    pos += step;
    step <<= 1;
  }
  int32_t lo = pos;
  int32_t hi = (pos + step < end) ? (pos + step + 1) : end;
  while (lo < hi) {
    int32_t mid = lo + ((hi - lo) >> 1);
    if (arr[mid] < target)
      lo = mid + 1;
    else
      hi = mid;
  }
  return lo;
}

__device__ double weighted_intersect_gallop(const int32_t* __restrict__ s_nbrs,
                                            const double* __restrict__ s_wts,
                                            int32_t s_size,
                                            const int32_t* __restrict__ l_nbrs,
                                            const double* __restrict__ l_wts,
                                            int32_t l_size)
{
  double iw = 0.0;
  int32_t j = 0;
  for (int32_t i = 0; i < s_size && j < l_size; ++i) {
    int32_t target = s_nbrs[i];
    j = galloping_search(l_nbrs, j, l_size, target);
    if (j < l_size && l_nbrs[j] == target) {
      double wa = s_wts[i];
      double wb = l_wts[j];
      iw += (wa < wb ? wa : wb);
      ++j;
    }
  }
  return iw;
}

__device__ double weighted_intersect_merge(const int32_t* __restrict__ a_nbrs,
                                           const double* __restrict__ a_wts,
                                           int32_t a_size,
                                           const int32_t* __restrict__ b_nbrs,
                                           const double* __restrict__ b_wts,
                                           int32_t b_size)
{
  if (a_size == 0 || b_size == 0) return 0.0;

  int32_t i_start = lower_bound_dev(a_nbrs, a_size, b_nbrs[0]);
  if (i_start >= a_size) return 0.0;
  int32_t j_start = lower_bound_dev(b_nbrs, b_size, a_nbrs[i_start]);
  if (j_start >= b_size) return 0.0;

  int32_t a_max = a_nbrs[a_size - 1];
  int32_t b_max = b_nbrs[b_size - 1];
  int32_t i_end = a_size;
  int32_t j_end = b_size;
  if (a_max < b_max) {
    j_end = lower_bound_dev(b_nbrs + j_start, b_size - j_start, a_max + 1) + j_start;
  } else if (b_max < a_max) {
    i_end = lower_bound_dev(a_nbrs + i_start, a_size - i_start, b_max + 1) + i_start;
  }

  double iw = 0.0;
  int32_t i = i_start;
  int32_t j = j_start;
  while (i < i_end && j < j_end) {
    int32_t av = a_nbrs[i];
    int32_t bv = b_nbrs[j];
    if (av < bv)
      ++i;
    else if (av > bv)
      ++j;
    else {
      double wa = a_wts[i];
      double wb = b_wts[j];
      iw += (wa < wb ? wa : wb);
      ++i;
      ++j;
    }
  }
  return iw;
}





__global__ void compute_overlap_kernel(const int32_t* __restrict__ offsets,
                                      const int32_t* __restrict__ indices,
                                      const double* __restrict__ weights,
                                      const int32_t* __restrict__ pair_first,
                                      const int32_t* __restrict__ pair_second,
                                      const int32_t* __restrict__ seeds,
                                      const double* __restrict__ seed_wdeg,
                                      int32_t num_seeds,
                                      double* __restrict__ out_scores,
                                      int64_t num_pairs)
{
  int64_t pid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pid >= num_pairs) return;

  int32_t u = pair_first[pid];
  int32_t v = pair_second[pid];

  int32_t u_s = offsets[u];
  int32_t u_e = offsets[u + 1];
  int32_t v_s = offsets[v];
  int32_t v_e = offsets[v + 1];

  int32_t u_deg = u_e - u_s;
  int32_t v_deg = v_e - v_s;

  if (u_deg == 0 || v_deg == 0) {
    out_scores[pid] = 0.0;
    return;
  }

  double wdeg_u;
  if (seeds) {
    int32_t sid = lower_bound_dev(seeds, num_seeds, u);
    wdeg_u = seed_wdeg[sid];
  } else {
    wdeg_u = seed_wdeg[u];
  }

  double wdeg_v = 0.0;
  for (int32_t j = v_s; j < v_e; ++j) wdeg_v += weights[j];

  const int32_t* u_nbrs = indices + u_s;
  const int32_t* v_nbrs = indices + v_s;
  const double* u_wts = weights + u_s;
  const double* v_wts = weights + v_s;

  double iw;
  int32_t ratio = (u_deg > v_deg) ? (u_deg / (v_deg > 0 ? v_deg : 1)) : (v_deg / (u_deg > 0 ? u_deg : 1));
  if (ratio > 8) {
    if (u_deg <= v_deg)
      iw = weighted_intersect_gallop(u_nbrs, u_wts, u_deg, v_nbrs, v_wts, v_deg);
    else
      iw = weighted_intersect_gallop(v_nbrs, v_wts, v_deg, u_nbrs, u_wts, u_deg);
  } else {
    iw = weighted_intersect_merge(u_nbrs, u_wts, u_deg, v_nbrs, v_wts, v_deg);
  }

  double denom = (wdeg_u < wdeg_v) ? wdeg_u : wdeg_v;
  out_scores[pid] = (denom <= DBL_MIN) ? 0.0 : (iw / denom);
}

struct IsSelfLoop {
  __host__ __device__ bool operator()(int64_t key) const
  {
    return static_cast<int32_t>(key >> 32) == static_cast<int32_t>(key & 0xFFFFFFFF);
  }
};

}  

similarity_result_double_t overlap_all_pairs_similarity(const graph32_t& graph,
                                                        const double* edge_weights,
                                                        const int32_t* vertices,
                                                        std::size_t num_vertices,
                                                        std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const double* d_weights = edge_weights;

    
    const int32_t* d_seeds = nullptr;
    int32_t num_seeds = 0;
    if (vertices != nullptr && num_vertices > 0) {
        d_seeds = vertices;
        num_seeds = static_cast<int32_t>(num_vertices);
    } else {
        d_seeds = nullptr;
        num_seeds = graph.number_of_vertices;
    }

    if (num_seeds <= 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    if (cache.seed_wdeg_cap < num_seeds) {
        if (cache.seed_wdeg) cudaFree(cache.seed_wdeg);
        cudaMalloc(&cache.seed_wdeg, static_cast<size_t>(num_seeds) * sizeof(double));
        cache.seed_wdeg_cap = num_seeds;
    }
    {
        int32_t block = 256;
        int32_t grid = (num_seeds + block - 1) / block;
        compute_seed_wdeg_kernel<<<grid, block>>>(d_offsets, d_weights, d_seeds, num_seeds, cache.seed_wdeg);
    }

    
    int64_t counts_size = static_cast<int64_t>(num_seeds) + 1;
    if (cache.raw_counts_cap < counts_size) {
        if (cache.raw_counts) cudaFree(cache.raw_counts);
        cudaMalloc(&cache.raw_counts, static_cast<size_t>(counts_size) * sizeof(int64_t));
        cache.raw_counts_cap = counts_size;
    }
    cudaMemsetAsync(cache.raw_counts, 0, sizeof(int64_t));
    if (num_seeds > 0) {
        count_raw_pairs_kernel<<<num_seeds, 256>>>(d_offsets, d_indices, d_seeds, num_seeds, cache.raw_counts + 1);
    }

    
    {
        thrust::device_ptr<int64_t> ptr(cache.raw_counts + 1);
        thrust::inclusive_scan(thrust::device, ptr, ptr + num_seeds, ptr);
    }

    
    int64_t total_raw = 0;
    cudaMemcpy(&total_raw, cache.raw_counts + num_seeds, sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (total_raw <= 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    if (cache.pair_keys_cap < total_raw) {
        if (cache.pair_keys) cudaFree(cache.pair_keys);
        cudaMalloc(&cache.pair_keys, static_cast<size_t>(total_raw) * sizeof(int64_t));
        cache.pair_keys_cap = total_raw;
    }
    if (num_seeds > 0) {
        write_raw_pairs_kernel<<<num_seeds, 256>>>(d_offsets, d_indices, d_seeds, num_seeds, cache.raw_counts, cache.pair_keys);
    }

    
    int64_t num_pairs;
    {
        thrust::device_ptr<int64_t> ptr(cache.pair_keys);
        thrust::sort(thrust::device, ptr, ptr + total_raw);
        auto unique_end = thrust::unique(thrust::device, ptr, ptr + total_raw);
        int64_t num_unique = unique_end - ptr;
        auto no_self = thrust::remove_if(thrust::device, ptr, ptr + num_unique, IsSelfLoop{});
        num_pairs = no_self - ptr;
    }

    if (num_pairs <= 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    if (cache.temp_first_cap < num_pairs) {
        if (cache.temp_first) cudaFree(cache.temp_first);
        cudaMalloc(&cache.temp_first, static_cast<size_t>(num_pairs) * sizeof(int32_t));
        cache.temp_first_cap = num_pairs;
    }
    if (cache.temp_second_cap < num_pairs) {
        if (cache.temp_second) cudaFree(cache.temp_second);
        cudaMalloc(&cache.temp_second, static_cast<size_t>(num_pairs) * sizeof(int32_t));
        cache.temp_second_cap = num_pairs;
    }
    {
        int64_t* keys = cache.pair_keys;
        int32_t* first = cache.temp_first;
        int32_t* second = cache.temp_second;
        thrust::for_each_n(thrust::device,
                           thrust::counting_iterator<int64_t>(0),
                           num_pairs,
                           [=] __device__(int64_t i) {
                               int64_t key = keys[i];
                               first[i] = static_cast<int32_t>(key >> 32);
                               second[i] = static_cast<int32_t>(key & 0xFFFFFFFF);
                           });
    }

    
    if (cache.temp_scores_cap < num_pairs) {
        if (cache.temp_scores) cudaFree(cache.temp_scores);
        cudaMalloc(&cache.temp_scores, static_cast<size_t>(num_pairs) * sizeof(double));
        cache.temp_scores_cap = num_pairs;
    }
    {
        int blk = 256;
        int grd = static_cast<int>((num_pairs + blk - 1) / blk);
        compute_overlap_kernel<<<grd, blk>>>(d_offsets,
                                             d_indices,
                                             d_weights,
                                             cache.temp_first,
                                             cache.temp_second,
                                             d_seeds,
                                             cache.seed_wdeg,
                                             num_seeds,
                                             cache.temp_scores,
                                             num_pairs);
    }

    
    int64_t output_count = num_pairs;
    if (topk.has_value() && static_cast<int64_t>(topk.value()) < num_pairs) {
        thrust::device_ptr<double> sc(cache.temp_scores);
        auto zip = thrust::make_zip_iterator(
            thrust::make_tuple(thrust::device_ptr<int32_t>(cache.temp_first),
                               thrust::device_ptr<int32_t>(cache.temp_second)));
        thrust::sort_by_key(thrust::device, sc, sc + num_pairs, zip, thrust::greater<double>());
        output_count = static_cast<int64_t>(topk.value());
    }

    if (output_count <= 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    double* out_scores = nullptr;
    cudaMalloc(&out_first, static_cast<size_t>(output_count) * sizeof(int32_t));
    cudaMalloc(&out_second, static_cast<size_t>(output_count) * sizeof(int32_t));
    cudaMalloc(&out_scores, static_cast<size_t>(output_count) * sizeof(double));

    cudaMemcpyAsync(out_first, cache.temp_first, static_cast<size_t>(output_count) * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(out_second, cache.temp_second, static_cast<size_t>(output_count) * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(out_scores, cache.temp_scores, static_cast<size_t>(output_count) * sizeof(double), cudaMemcpyDeviceToDevice);

    return {out_first, out_second, out_scores, static_cast<std::size_t>(output_count)};
}

}  
