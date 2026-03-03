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
#include <climits>
#include <optional>
#include <cub/cub.cuh>

namespace aai {

namespace {

struct Cache : Cacheable {
  void* scratch_ptrs[32]{};
  size_t scratch_caps[32]{};
  float* sumw = nullptr;
  int64_t sumw_capacity = 0;
  int64_t* count_buf = nullptr;

  void ensure_sumw(int64_t n) {
    if (sumw_capacity < n) {
      if (sumw) cudaFree(sumw);
      cudaMalloc(&sumw, n * sizeof(float));
      sumw_capacity = n;
    }
  }

  void ensure_count_buf() {
    if (!count_buf) {
      cudaMalloc(&count_buf, sizeof(int64_t));
    }
  }

  ~Cache() override {
    for (int i = 0; i < 32; ++i) {
      if (scratch_ptrs[i]) {
        cudaFree(scratch_ptrs[i]);
        scratch_ptrs[i] = nullptr;
        scratch_caps[i] = 0;
      }
    }
    if (sumw) { cudaFree(sumw); sumw = nullptr; }
    if (count_buf) { cudaFree(count_buf); count_buf = nullptr; }
  }
};





template <typename T>
static T* get_scratch(void** ptrs, size_t* caps, int idx, size_t count) {
  size_t bytes = count * sizeof(T);
  if (bytes > caps[idx]) {
    if (ptrs[idx]) cudaFree(ptrs[idx]);
    cudaMalloc(&ptrs[idx], bytes);
    caps[idx] = bytes;
  }
  return reinterpret_cast<T*>(ptrs[idx]);
}

static void* get_scratch_bytes(void** ptrs, size_t* caps, int idx, size_t bytes) {
  if (bytes > caps[idx]) {
    if (ptrs[idx]) cudaFree(ptrs[idx]);
    cudaMalloc(&ptrs[idx], bytes);
    caps[idx] = bytes;
  }
  return ptrs[idx];
}





__device__ __forceinline__ int lower_bound_int32(const int32_t* __restrict__ a, int n, int32_t x) {
  int lo = 0;
  int hi = n;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    int32_t v = a[mid];
    if (v < x) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}





__global__ void iota_int32_kernel(int32_t* __restrict__ out, int32_t n) {
  int32_t i = static_cast<int32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = i;
}


__global__ void sumw_warp_kernel(const int32_t* __restrict__ offsets,
                                const float* __restrict__ weights,
                                float* __restrict__ sumw,
                                int32_t n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp = tid >> 5;
  int lane = tid & 31;
  if (warp >= n) return;
  int32_t v = warp;
  int32_t start = offsets[v];
  int32_t end = offsets[v + 1];
  float s = 0.0f;
  for (int32_t e = start + lane; e < end; e += 32) {
    s += weights[e];
  }
  
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    s += __shfl_down_sync(0xffffffff, s, off);
  }
  if (lane == 0) sumw[v] = s;
}

__global__ void count_raw_pairs_kernel(const int32_t* __restrict__ offsets,
                                      const int32_t* __restrict__ indices,
                                      const int32_t* __restrict__ seeds,
                                      int32_t num_seeds,
                                      int64_t* __restrict__ counts) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= num_seeds) return;
  int u = seeds[s];
  int64_t c = 0;
  int us = offsets[u];
  int ue = offsets[u + 1];
  for (int e = us; e < ue; e++) {
    int x = indices[e];
    c += (int64_t)(offsets[x + 1] - offsets[x]);
  }
  counts[s] = c;
}


__global__ void max_seed_degree_kernel(const int32_t* __restrict__ offsets,
                                     const int32_t* __restrict__ seeds,
                                     int32_t num_seeds,
                                     int* __restrict__ out_max_deg) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int local = 0;
  if (tid < num_seeds) {
    int u = seeds[tid];
    local = offsets[u + 1] - offsets[u];
  }
  __shared__ int smax;
  if (threadIdx.x == 0) smax = 0;
  __syncthreads();
  atomicMax(&smax, local);
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(out_max_deg, smax);
}



__global__ void generate_raw_pairs_kernel(const int32_t* __restrict__ offsets,
                                         const int32_t* __restrict__ indices,
                                         const int32_t* __restrict__ seeds,
                                         int32_t num_seeds,
                                         const int64_t* __restrict__ pair_offsets,
                                         int64_t* __restrict__ keys,
                                         int64_t nv) {
  int si = blockIdx.x;
  if (si >= num_seeds) return;
  int u = seeds[si];
  int us = offsets[u];
  int ue = offsets[u + 1];
  int ud = ue - us;
  if (ud == 0) return;

  extern __shared__ int sp[];  

  for (int i = threadIdx.x; i < ud; i += blockDim.x) {
    int x = indices[us + i];
    sp[i] = offsets[x + 1] - offsets[x];
  }
  __syncthreads();

  __shared__ int total;
  if (threadIdx.x == 0) {
    int s = 0;
    for (int i = 0; i < ud; i++) {
      int d = sp[i];
      sp[i] = s;
      s += d;
    }
    sp[ud] = s;
    total = s;
  }
  __syncthreads();

  int tot = total;
  int64_t base = pair_offsets[si];

  for (int f = threadIdx.x; f < tot; f += blockDim.x) {
    
    int lo = 0, hi = ud;
    while (lo < hi) {
      int m = (lo + hi + 1) >> 1;
      if (sp[m] <= f) lo = m;
      else hi = m - 1;
    }
    int x = indices[us + lo];
    int xs = offsets[x];
    int v = indices[xs + (f - sp[lo])];
    keys[base + f] = (v != u) ? ((int64_t)u * nv + v) : INT64_MAX;
  }
}


__global__ void decode_keys_kernel(const int64_t* __restrict__ keys,
                                  int32_t* __restrict__ first,
                                  int32_t* __restrict__ second,
                                  int64_t n,
                                  int64_t nv) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int64_t k = keys[i];
  first[i] = (int32_t)(k / nv);
  second[i] = (int32_t)(k - (int64_t)first[i] * nv);
}


__global__ void init_write_pos_kernel(const int64_t* __restrict__ pair_offsets,
                                    unsigned long long* __restrict__ write_pos,
                                    int32_t num_seeds) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_seeds) write_pos[i] = (unsigned long long)pair_offsets[i];
}

__global__ void generate_raw_pairs_atomic_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds, int32_t num_seeds,
    unsigned long long* __restrict__ write_pos, int64_t* __restrict__ keys, int64_t nv) {
  int si = blockIdx.x;
  if (si >= num_seeds) return;
  int u = seeds[si];
  int us = offsets[u];
  int ue = offsets[u + 1];

  for (int e = us + threadIdx.x; e < ue; e += blockDim.x) {
    int x = indices[e];
    int xs = offsets[x];
    int xe = offsets[x + 1];
    for (int f = xs; f < xe; ++f) {
      int v = indices[f];
      unsigned long long pos = atomicAdd(&write_pos[si], 1ULL);
      keys[pos] = (v != u) ? ((int64_t)u * nv + v) : INT64_MAX;
    }
  }
}

__global__ void pack_pairs_kernel(const int32_t* __restrict__ first,
                                 const int32_t* __restrict__ second,
                                 uint64_t* __restrict__ packed,
                                 int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  packed[i] = (uint64_t)((uint32_t)first[i]) << 32 | (uint32_t)second[i];
}

__global__ void unpack_pairs_kernel(const uint64_t* __restrict__ packed,
                                   int32_t* __restrict__ first,
                                   int32_t* __restrict__ second,
                                   int64_t n) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint64_t p = packed[i];
  first[i] = (int32_t)(p >> 32);
  second[i] = (int32_t)(uint32_t)p;
}


__global__ void overlap_warp_kernel(const int32_t* __restrict__ offsets,
                                   const int32_t* __restrict__ indices,
                                   const float* __restrict__ weights,
                                   const float* __restrict__ sumw,
                                   const int32_t* __restrict__ first,
                                   const int32_t* __restrict__ second,
                                   float* __restrict__ scores,
                                   int64_t num_pairs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int wid = tid >> 5;
  int lane = tid & 31;
  if (wid >= num_pairs) return;

  int u = first[wid];
  int v = second[wid];
  int us = offsets[u];
  int ue = offsets[u + 1];
  int vs = offsets[v];
  int ve = offsets[v + 1];
  int ul = ue - us;
  int vl = ve - vs;

  
  const int32_t* si;
  const float* sw;
  int sl;
  const int32_t* li;
  const float* lw;
  int ll;
  if (ul <= vl) {
    si = indices + us;
    sw = weights + us;
    sl = ul;
    li = indices + vs;
    lw = weights + vs;
    ll = vl;
  } else {
    si = indices + vs;
    sw = weights + vs;
    sl = vl;
    li = indices + us;
    lw = weights + us;
    ll = ul;
  }

  float num = 0.0f;
  for (int i = lane; i < sl; i += 32) {
    int32_t t = si[i];
    float tw = sw[i];
    int p = lower_bound_int32(li, ll, t);
    if (p < ll && li[p] == t) {
      num += fminf(tw, lw[p]);
    }
  }

  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    num += __shfl_down_sync(0xffffffff, num, off);
  }

  if (lane == 0) {
    float d = fminf(sumw[u], sumw[v]);
    scores[wid] = (d > 0.0f) ? (num / d) : 0.0f;
  }
}




template <int HASH_CAP, int U_CACHE_MAX>
__global__ void subset_overlap_kernel(const int32_t* __restrict__ offsets,
                                     const int32_t* __restrict__ indices,
                                     const float* __restrict__ weights,
                                     const float* __restrict__ sumw,
                                     const int32_t* __restrict__ seeds,
                                     int32_t num_seeds,
                                     int32_t* __restrict__ out_first,
                                     int32_t* __restrict__ out_second,
                                     float* __restrict__ out_scores,
                                     unsigned long long* __restrict__ out_count) {
  int si = blockIdx.x;
  if (si >= num_seeds) return;

  int u = seeds[si];
  int us = offsets[u];
  int ue = offsets[u + 1];
  int du = ue - us;

  __shared__ int32_t h_keys[HASH_CAP];
  __shared__ int32_t u_idx[U_CACHE_MAX];
  __shared__ float u_w[U_CACHE_MAX];

  for (int i = threadIdx.x; i < HASH_CAP; i += blockDim.x) {
    h_keys[i] = -1;
  }

  if (du <= U_CACHE_MAX) {
    for (int i = threadIdx.x; i < du; i += blockDim.x) {
      u_idx[i] = indices[us + i];
      u_w[i] = weights[us + i];
    }
  }
  __syncthreads();

  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int warps = blockDim.x >> 5;

  
  for (int e = us + warp; e < ue; e += warps) {
    int x = indices[e];
    int xs = offsets[x];
    int xe = offsets[x + 1];
    for (int f = xs + lane; f < xe; f += 32) {
      int v = indices[f];
      if (v == u) continue;
      uint32_t h = (uint32_t)v * 0x9e3779b1u;
      int slot = (int)(h & (HASH_CAP - 1));
      #pragma unroll 1
      for (int it = 0; it < HASH_CAP; ++it) {
        int old = atomicCAS(&h_keys[slot], -1, v);
        if (old == -1 || old == v) break;
        slot = (slot + 1) & (HASH_CAP - 1);
      }
    }
  }
  __syncthreads();

  
  for (int slot = warp; slot < HASH_CAP; slot += warps) {
    int v = h_keys[slot];
    if (v < 0) continue;

    int vs = offsets[v];
    int ve = offsets[v + 1];
    int dv = ve - vs;

    
    const int32_t* si_ptr;
    const float* sw_ptr;
    int sl;
    const int32_t* li_ptr;
    const float* lw_ptr;
    int ll;

    
    const int32_t* uip = (du <= U_CACHE_MAX) ? u_idx : (indices + us);
    const float* uwp = (du <= U_CACHE_MAX) ? u_w : (weights + us);

    if (du <= dv) {
      si_ptr = uip;
      sw_ptr = uwp;
      sl = du;
      li_ptr = indices + vs;
      lw_ptr = weights + vs;
      ll = dv;
    } else {
      si_ptr = indices + vs;
      sw_ptr = weights + vs;
      sl = dv;
      li_ptr = uip;
      lw_ptr = uwp;
      ll = du;
    }

    float num = 0.0f;
    for (int i = lane; i < sl; i += 32) {
      int32_t t = si_ptr[i];
      float tw = sw_ptr[i];
      int p = lower_bound_int32(li_ptr, ll, t);
      if (p < ll && li_ptr[p] == t) {
        num += fminf(tw, lw_ptr[p]);
      }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      num += __shfl_down_sync(0xffffffff, num, off);
    }

    if (lane == 0) {
      float denom = fminf(sumw[u], sumw[v]);
      float score = (denom > 0.0f) ? (num / denom) : 0.0f;
      unsigned long long pos = atomicAdd(out_count, 1ULL);
      out_first[pos] = u;
      out_second[pos] = v;
      out_scores[pos] = score;
    }
  }
}





static void overlap_run(
    const int32_t* d_offsets,
    const int32_t* d_indices,
    const float* d_edge_weights,
    int32_t num_vertices,
    int32_t ,
    const int32_t* d_vertices,
    int32_t num_subset_vertices,
    int64_t topk,
    float* d_sumw,
    int32_t* d_first,
    int32_t* d_second,
    float* d_scores,
    int64_t* d_count,
    void** scratch_ptrs,
    size_t* scratch_caps) {
  cudaStream_t stream = 0;

  
  {
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (num_vertices + warps_per_block - 1) / warps_per_block;
    sumw_warp_kernel<<<blocks, threads, 0, stream>>>(d_offsets, d_edge_weights, d_sumw, num_vertices);
  }

  
  cudaMemsetAsync(d_count, 0, sizeof(int64_t), stream);

  
  
  

  const int B = 256;

  int32_t num_seeds = (num_subset_vertices > 0 && d_vertices != nullptr) ? num_subset_vertices : num_vertices;
  const int32_t* d_seeds = d_vertices;
  int32_t* d_seeds_buf = nullptr;
  if (num_subset_vertices <= 0 || d_vertices == nullptr) {
    d_seeds_buf = get_scratch<int32_t>(scratch_ptrs, scratch_caps, 8, (size_t)num_seeds);
    int g = (num_seeds + B - 1) / B;
    iota_int32_kernel<<<g, B, 0, stream>>>(d_seeds_buf, num_seeds);
    d_seeds = d_seeds_buf;
  }

  int64_t* d_counts = get_scratch<int64_t>(scratch_ptrs, scratch_caps, 9, (size_t)num_seeds);
  {
    int g = (num_seeds + B - 1) / B;
    count_raw_pairs_kernel<<<g, B, 0, stream>>>(d_offsets, d_indices, d_seeds, num_seeds, d_counts);
  }

  int64_t* d_pair_offsets = get_scratch<int64_t>(scratch_ptrs, scratch_caps, 10, (size_t)num_seeds + 1);
  cudaMemsetAsync(d_pair_offsets, 0, sizeof(int64_t), stream);
  {
    size_t temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, d_counts, d_pair_offsets + 1, num_seeds, stream);
    void* temp = get_scratch_bytes(scratch_ptrs, scratch_caps, 11, temp_bytes);
    cub::DeviceScan::InclusiveSum(temp, temp_bytes, d_counts, d_pair_offsets + 1, num_seeds, stream);
  }

  int64_t total_raw = 0;
  cudaMemcpyAsync(&total_raw, d_pair_offsets + num_seeds, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  if (total_raw == 0) {
    int64_t z = 0;
    cudaMemcpyAsync(d_count, &z, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return;
  }

  int64_t* d_keys_in = get_scratch<int64_t>(scratch_ptrs, scratch_caps, 12, (size_t)total_raw);
  int64_t* d_keys_out = get_scratch<int64_t>(scratch_ptrs, scratch_caps, 13, (size_t)total_raw);



  {
    
    int* d_max_deg = get_scratch<int>(scratch_ptrs, scratch_caps, 0, 1);
    cudaMemsetAsync(d_max_deg, 0, sizeof(int), stream);
    {
      int threads = 256;
      int blocks = (num_seeds + threads - 1) / threads;
      max_seed_degree_kernel<<<blocks, threads, 0, stream>>>(d_offsets, d_seeds, num_seeds, d_max_deg);
    }
    int h_max_deg = 0;
    cudaMemcpyAsync(&h_max_deg, d_max_deg, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (h_max_deg > 24575) {
      
      unsigned long long* d_write_pos = get_scratch<unsigned long long>(scratch_ptrs, scratch_caps, 1, (size_t)num_seeds);
      {
        int threads = 256;
        int blocks = (num_seeds + threads - 1) / threads;
        init_write_pos_kernel<<<blocks, threads, 0, stream>>>(d_pair_offsets, d_write_pos, num_seeds);
      }
      generate_raw_pairs_atomic_kernel<<<num_seeds, B, 0, stream>>>(
          d_offsets, d_indices, d_seeds, num_seeds, d_write_pos, d_keys_in, (int64_t)num_vertices);
    } else {
      size_t sm = 96 * 1024;
      cudaFuncSetAttribute(generate_raw_pairs_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
      generate_raw_pairs_kernel<<<num_seeds, B, sm, stream>>>(
          d_offsets, d_indices, d_seeds, num_seeds, d_pair_offsets, d_keys_in, (int64_t)num_vertices);
    }
  }

  
  int end_bit = 1;
  {
    long long mk = (long long)num_vertices * (long long)num_vertices;
    while ((1LL << end_bit) < mk && end_bit < 63) end_bit++;
  }

  cub::DoubleBuffer<int64_t> db(d_keys_in, d_keys_out);
  {
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, db, total_raw, 0, end_bit, stream);
    void* temp = get_scratch_bytes(scratch_ptrs, scratch_caps, 14, temp_bytes);
    cub::DeviceRadixSort::SortKeys(temp, temp_bytes, db, total_raw, 0, end_bit, stream);
  }

  int64_t* d_sorted = db.Current();

  
  int64_t* d_unique = get_scratch<int64_t>(scratch_ptrs, scratch_caps, 15, (size_t)total_raw);
  int* d_num_unique = get_scratch<int>(scratch_ptrs, scratch_caps, 3, 1);
  {
    size_t temp_bytes = 0;
    cub::DeviceSelect::Unique(nullptr, temp_bytes, d_sorted, d_unique, d_num_unique, total_raw, stream);
    void* temp = get_scratch_bytes(scratch_ptrs, scratch_caps, 2, temp_bytes);
    cub::DeviceSelect::Unique(temp, temp_bytes, d_sorted, d_unique, d_num_unique, total_raw, stream);
  }

  int num_unique_h = 0;
  cudaMemcpyAsync(&num_unique_h, d_num_unique, sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  int64_t nu = (int64_t)num_unique_h;
  if (nu > 0) {
    int64_t last = 0;
    cudaMemcpyAsync(&last, d_unique + (nu - 1), sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (last == INT64_MAX) nu--;
  }

  if (nu <= 0) {
    int64_t z = 0;
    cudaMemcpyAsync(d_count, &z, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return;
  }

  
  int32_t* d_out_first = nullptr;
  int32_t* d_out_second = nullptr;
  float* d_out_scores = nullptr;
  cudaMalloc(&d_out_first, (size_t)nu * sizeof(int32_t));
  cudaMalloc(&d_out_second, (size_t)nu * sizeof(int32_t));
  cudaMalloc(&d_out_scores, (size_t)nu * sizeof(float));

  {
    int g = (int)((nu + B - 1) / B);
    decode_keys_kernel<<<g, B, 0, stream>>>(d_unique, d_out_first, d_out_second, nu, (int64_t)num_vertices);
  }

  {
    int wpb = B / 32;
    int g = (int)((nu + wpb - 1) / wpb);
    overlap_warp_kernel<<<g, B, 0, stream>>>(
        d_offsets, d_indices, d_edge_weights, d_sumw, d_out_first, d_out_second, d_out_scores, nu);
  }

  int64_t rc = nu;
  if (topk >= 0 && topk < nu) {
    
    uint64_t* d_packed_in = get_scratch<uint64_t>(scratch_ptrs, scratch_caps, 4, (size_t)nu);
    uint64_t* d_packed_out = get_scratch<uint64_t>(scratch_ptrs, scratch_caps, 5, (size_t)nu);
    int g = (int)((nu + B - 1) / B);
    pack_pairs_kernel<<<g, B, 0, stream>>>(d_out_first, d_out_second, d_packed_in, nu);

    float* d_scores_alt = get_scratch<float>(scratch_ptrs, scratch_caps, 6, (size_t)nu);

    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairsDescending(nullptr,
                                              temp_bytes,
                                              d_out_scores,
                                              d_scores_alt,
                                              d_packed_in,
                                              d_packed_out,
                                              (int)nu,
                                              0,
                                              32,
                                              stream);
    void* temp = get_scratch_bytes(scratch_ptrs, scratch_caps, 7, temp_bytes);
    cub::DeviceRadixSort::SortPairsDescending(temp,
                                              temp_bytes,
                                              d_out_scores,
                                              d_scores_alt,
                                              d_packed_in,
                                              d_packed_out,
                                              (int)nu,
                                              0,
                                              32,
                                              stream);

    cudaMemcpyAsync(d_out_scores, d_scores_alt, (size_t)nu * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    unpack_pairs_kernel<<<g, B, 0, stream>>>(d_packed_out, d_out_first, d_out_second, nu);
    rc = topk;
  }

  
  scratch_ptrs[24] = d_out_first;
  scratch_ptrs[25] = d_out_second;
  scratch_ptrs[26] = d_out_scores;

  cudaMemcpyAsync(d_count, &rc, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  cudaStreamSynchronize(stream);
}

}  

similarity_result_float_t overlap_all_pairs_similarity_seg(const graph32_t& graph,
                                                           const float* edge_weights,
                                                           const int32_t* vertices,
                                                           std::size_t num_vertices,
                                                           std::optional<std::size_t> topk) {
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);

  cache.ensure_sumw(graph.number_of_vertices);
  cache.ensure_count_buf();

  int32_t num_subset = static_cast<int32_t>(num_vertices);
  int64_t topk_val = topk.has_value() ? static_cast<int64_t>(topk.value()) : -1;

  overlap_run(
      graph.offsets,
      graph.indices,
      edge_weights,
      graph.number_of_vertices,
      graph.number_of_edges,
      vertices,
      num_subset,
      topk_val,
      cache.sumw,
      nullptr,
      nullptr,
      nullptr,
      cache.count_buf,
      cache.scratch_ptrs,
      cache.scratch_caps);

  int64_t count_host = 0;
  cudaMemcpy(&count_host, cache.count_buf, sizeof(int64_t), cudaMemcpyDeviceToHost);

  similarity_result_float_t result;
  result.count = static_cast<std::size_t>(count_host);
  if (count_host > 0) {
    result.first = static_cast<int32_t*>(cache.scratch_ptrs[24]);
    result.second = static_cast<int32_t*>(cache.scratch_ptrs[25]);
    result.scores = static_cast<float*>(cache.scratch_ptrs[26]);
    cache.scratch_ptrs[24] = nullptr;
    cache.scratch_ptrs[25] = nullptr;
    cache.scratch_ptrs[26] = nullptr;
  } else {
    result.first = nullptr;
    result.second = nullptr;
    result.scores = nullptr;
  }
  return result;
}

}  
