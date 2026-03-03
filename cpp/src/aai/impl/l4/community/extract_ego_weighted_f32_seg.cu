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
#include <vector>
#include <algorithm>

namespace aai {

namespace {










__global__ void init_multi_bounds_kernel(const int32_t* __restrict__ sources,
                                         int32_t n_sources,
                                         int32_t num_vertices,
                                         uint32_t* __restrict__ bitmaps,
                                         int32_t bitmap_words,
                                         int32_t* __restrict__ ego_vertices_all,
                                         int32_t* __restrict__ ego_sizes,
                                         int32_t* __restrict__ frontier_start,
                                         int32_t* __restrict__ frontier_end)
{
  int s = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_sources) return;
  int32_t src_v = sources[s];

  uint32_t* bm = bitmaps + (size_t)s * (size_t)bitmap_words;
  bm[(uint32_t)src_v >> 5] = (1u << (src_v & 31));

  int32_t* ego = ego_vertices_all + (size_t)s * (size_t)num_vertices;
  ego[0] = src_v;
  ego_sizes[s] = 1;

  frontier_start[s] = 0;
  frontier_end[s] = 1;
}

__global__ void update_bounds_multi_kernel(const int32_t* __restrict__ ego_sizes,
                                           int32_t* __restrict__ frontier_start,
                                           int32_t* __restrict__ frontier_end,
                                           int32_t n_sources)
{
  int s = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= n_sources) return;
  frontier_start[s] = frontier_end[s];
  frontier_end[s] = ego_sizes[s];
}

__global__ void bfs_expand_bounds_multi_kernel(const int32_t* __restrict__ offsets,
                                               const int32_t* __restrict__ indices,
                                               uint32_t* __restrict__ bitmaps,
                                               int32_t bitmap_words,
                                               int32_t* __restrict__ ego_vertices_all,
                                               int32_t* __restrict__ ego_sizes,
                                               const int32_t* __restrict__ frontier_start,
                                               const int32_t* __restrict__ frontier_end,
                                               int32_t num_vertices,
                                               int32_t n_sources)
{
  for (int s = (int)blockIdx.y; s < n_sources; s += (int)gridDim.y) {
    const int32_t fstart = frontier_start[s];
    const int32_t fend = frontier_end[s];
    const int32_t fsz = fend - fstart;
    if (fsz <= 0) continue;

    uint32_t* bm = bitmaps + (size_t)s * (size_t)bitmap_words;
    int32_t* ego = ego_vertices_all + (size_t)s * (size_t)num_vertices;
    int32_t* ego_size_ptr = ego_sizes + s;

    int lane = threadIdx.x & 31;
    int warp0 = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    int warps_total = (int)((gridDim.x * blockDim.x) >> 5);

    for (int w = warp0; w < fsz; w += warps_total) {
      int32_t v = ego[fstart + w];
      int32_t start = offsets[v];
      int32_t end = offsets[v + 1];

      for (int32_t base_e = start; base_e < end; base_e += 32) {
        int32_t e = base_e + lane;
        bool active = (e < end);

        int32_t u = 0;
        bool is_new = false;
        if (active) {
          u = indices[e];
          uint32_t wi = ((uint32_t)u) >> 5;
          uint32_t bit = 1u << (u & 31);
          uint32_t old = atomicOr(&bm[wi], bit);
          is_new = !(old & bit);
        }

        uint32_t ballot = __ballot_sync(0xffffffff, is_new);
        if (!ballot) continue;

        int32_t n_new = __popc(ballot);
        int32_t ego_base = 0;
        if (lane == 0) {
          ego_base = atomicAdd(ego_size_ptr, n_new);
        }
        ego_base = __shfl_sync(0xffffffff, ego_base, 0);

        if (is_new) {
          uint32_t lower = (lane == 0) ? 0u : ((1u << lane) - 1);
          int32_t off = __popc(ballot & lower);
          int32_t pos = ego_base + off;
          ego[pos] = u;
        }
      }
    }
  }
}





__global__ void count_edges_multi_kernel(const int32_t* __restrict__ offsets,
                                         const int32_t* __restrict__ indices,
                                         const uint32_t* __restrict__ bitmaps,
                                         int32_t bitmap_words,
                                         const int32_t* __restrict__ ego_vertices_all,
                                         const int32_t* __restrict__ ego_sizes,
                                         int32_t num_vertices,
                                         int32_t n_sources,
                                         int64_t* __restrict__ edge_counts)
{
  for (int s = (int)blockIdx.y; s < n_sources; s += (int)gridDim.y) {
    const uint32_t* bm = bitmaps + (size_t)s * (size_t)bitmap_words;
    const int32_t* ego = ego_vertices_all + (size_t)s * (size_t)num_vertices;
    int32_t ego_size = ego_sizes[s];

    int lane = threadIdx.x & 31;
    int warp0 = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    int warps_total = (int)((gridDim.x * blockDim.x) >> 5);

    int64_t thread_sum = 0;
    for (int i = warp0; i < ego_size; i += warps_total) {
      int32_t v = ego[i];
      int32_t start = offsets[v];
      int32_t end = offsets[v + 1];
      int local = 0;
      for (int e = start + lane; e < end; e += 32) {
        int32_t u = indices[e];
        local += (bm[((uint32_t)u) >> 5] & (1u << (u & 31))) ? 1 : 0;
      }
      for (int off = 16; off > 0; off >>= 1) {
        local += __shfl_down_sync(0xffffffff, local, off);
      }
      if (lane == 0) thread_sum += (int64_t)local;
    }

    using BlockReduce = cub::BlockReduce<int64_t, 256>;
    __shared__ typename BlockReduce::TempStorage temp;
    int64_t block_sum = BlockReduce(temp).Sum(thread_sum);
    if (threadIdx.x == 0 && block_sum) {
      atomicAdd((unsigned long long*)(edge_counts + s), (unsigned long long)block_sum);
    }
  }
}

__global__ void init_write_pos_kernel(const int64_t* __restrict__ offsets,
                                      uint64_t* __restrict__ write_pos,
                                      int32_t n_sources)
{
  int s = (int)blockIdx.x * blockDim.x + threadIdx.x;
  if (s < n_sources) write_pos[s] = (uint64_t)offsets[s];
}

__global__ void extract_edges_multi_kernel(const int32_t* __restrict__ offsets,
                                           const int32_t* __restrict__ indices,
                                           const float* __restrict__ weights,
                                           const uint32_t* __restrict__ bitmaps,
                                           int32_t bitmap_words,
                                           const int32_t* __restrict__ ego_vertices_all,
                                           const int32_t* __restrict__ ego_sizes,
                                           int32_t num_vertices,
                                           int32_t n_sources,
                                           int32_t* __restrict__ out_srcs,
                                           int32_t* __restrict__ out_dsts,
                                           float* __restrict__ out_weights,
                                           uint64_t* __restrict__ write_pos)
{
  for (int s = (int)blockIdx.y; s < n_sources; s += (int)gridDim.y) {
    const uint32_t* bm = bitmaps + (size_t)s * (size_t)bitmap_words;
    const int32_t* ego = ego_vertices_all + (size_t)s * (size_t)num_vertices;
    int32_t ego_size = ego_sizes[s];

    int lane = threadIdx.x & 31;
    int warp0 = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    int warps_total = (int)((gridDim.x * blockDim.x) >> 5);

    for (int i = warp0; i < ego_size; i += warps_total) {
      int32_t v = ego[i];
      int32_t start = offsets[v];
      int32_t end = offsets[v + 1];

      for (int32_t base_e = start; base_e < end; base_e += 32) {
        int32_t e = base_e + lane;
        bool active = (e < end);

        int32_t u = 0;
        float w = 0.0f;
        bool hit = false;
        if (active) {
          u = indices[e];
          hit = (bm[((uint32_t)u) >> 5] & (1u << (u & 31))) != 0;
          if (hit) w = weights[e];
        }

        uint32_t ballot = __ballot_sync(0xffffffff, hit);
        if (!ballot) continue;

        uint64_t base = 0;
        if (lane == 0) {
          base = atomicAdd((unsigned long long*)(write_pos + s), (unsigned long long)__popc(ballot));
        }
        base = __shfl_sync(0xffffffff, base, 0);

        if (hit) {
          uint32_t lower = (lane == 0) ? 0u : ((1u << lane) - 1);
          int off = __popc(ballot & lower);
          uint64_t pos = base + (uint64_t)off;
          out_srcs[pos] = v;
          out_dsts[pos] = u;
          out_weights[pos] = w;
        }
      }
    }
  }
}

__global__ void set_last_offset_kernel(int64_t* __restrict__ offsets,
                                       const int64_t* __restrict__ counts,
                                       int32_t n_sources)
{
  if (threadIdx.x == 0) {
    if (n_sources > 0) offsets[n_sources] = offsets[n_sources - 1] + counts[n_sources - 1];
    else offsets[0] = 0;
  }
}





__global__ void init_bfs_kernel(uint32_t* __restrict__ visited,
                                int32_t* __restrict__ ego_vertices,
                                int32_t* __restrict__ ego_size,
                                int32_t* __restrict__ frontier_bounds,
                                int32_t source)
{
  if (threadIdx.x == 0) {
    visited[(uint32_t)source >> 5] |= (1u << (source & 31));
    ego_vertices[0] = source;
    *ego_size = 1;
    frontier_bounds[0] = 0;
    frontier_bounds[1] = 1;
  }
}

__global__ void bfs_expand_warp_kernel(const int32_t* __restrict__ offsets,
                                       const int32_t* __restrict__ indices,
                                       int32_t* __restrict__ ego_vertices,
                                       int32_t* __restrict__ ego_size,
                                       const int32_t* __restrict__ frontier_bounds,
                                       uint32_t* __restrict__ visited)
{
  const int32_t fstart = frontier_bounds[0];
  const int32_t fend = frontier_bounds[1];
  const int32_t frontier_size = fend - fstart;

  const int32_t lane_id = threadIdx.x & 31;
  const int32_t warp_id0 = (int32_t)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int32_t total_warps = (int32_t)((gridDim.x * blockDim.x) >> 5);

  for (int32_t warp_id = warp_id0; warp_id < frontier_size; warp_id += total_warps) {
    const int32_t v = ego_vertices[fstart + warp_id];
    const int32_t start = offsets[v];
    const int32_t end = offsets[v + 1];

    for (int32_t base_e = start; base_e < end; base_e += 32) {
      int32_t e = base_e + lane_id;
      bool active = (e < end);

      int32_t u = 0;
      bool is_new = false;
      if (active) {
        u = indices[e];
        uint32_t word_idx = ((uint32_t)u) >> 5;
        uint32_t bit = 1u << (u & 31);
        uint32_t old = atomicOr(&visited[word_idx], bit);
        is_new = !(old & bit);
      }

      uint32_t ballot = __ballot_sync(0xffffffff, is_new);
      if (!ballot) continue;

      int32_t warp_base = 0;
      if (lane_id == 0) warp_base = atomicAdd(ego_size, __popc(ballot));
      warp_base = __shfl_sync(0xffffffff, warp_base, 0);

      if (is_new) {
        uint32_t lower_mask = (lane_id == 0) ? 0u : ((1u << lane_id) - 1);
        int32_t my_off = __popc(ballot & lower_mask);
        ego_vertices[warp_base + my_off] = u;
      }
    }
  }
}

__global__ void update_bounds_kernel(int32_t* __restrict__ frontier_bounds,
                                     const int32_t* __restrict__ ego_size)
{
  if (threadIdx.x == 0) {
    frontier_bounds[0] = frontier_bounds[1];
    frontier_bounds[1] = *ego_size;
  }
}

__global__ void count_edges_warp_kernel(const int32_t* __restrict__ offsets,
                                        const int32_t* __restrict__ indices,
                                        const uint32_t* __restrict__ visited,
                                        const int32_t* __restrict__ ego_vertices,
                                        const int32_t* __restrict__ ego_size_ptr,
                                        unsigned long long* __restrict__ edge_count)
{
  const int32_t ego_size = *ego_size_ptr;
  const int32_t lane_id = threadIdx.x & 31;
  const int32_t warp_id0 = (int32_t)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int32_t total_warps = (int32_t)((gridDim.x * blockDim.x) >> 5);

  for (int32_t warp_id = warp_id0; warp_id < ego_size; warp_id += total_warps) {
    const int32_t v = ego_vertices[warp_id];
    const int32_t start = offsets[v];
    const int32_t end = offsets[v + 1];

    int local_count = 0;
    for (int32_t e = start + lane_id; e < end; e += 32) {
      int32_t u = indices[e];
      local_count += (visited[((uint32_t)u) >> 5] & (1u << (u & 31))) ? 1 : 0;
    }

    for (int offset = 16; offset > 0; offset >>= 1) local_count += __shfl_down_sync(0xffffffff, local_count, offset);

    if (lane_id == 0 && local_count) atomicAdd(edge_count, (unsigned long long)local_count);
  }
}

__global__ void collect_bitmap_atomic_kernel(const uint32_t* __restrict__ bitmap,
                                             int32_t bitmap_words,
                                             int32_t* __restrict__ vertices,
                                             int32_t* __restrict__ count)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = tid; i < bitmap_words; i += stride) {
    uint32_t word = bitmap[i];
    while (word) {
      int bit = __ffs(word) - 1;
      int32_t pos = atomicAdd(count, 1);
      vertices[pos] = i * 32 + bit;
      word &= word - 1;
    }
  }
}

__global__ void extract_edges_warp_kernel(const int32_t* __restrict__ offsets,
                                          const int32_t* __restrict__ indices,
                                          const float* __restrict__ weights,
                                          const uint32_t* __restrict__ visited,
                                          const int32_t* __restrict__ ego_vertices,
                                          const int32_t* __restrict__ ego_size_ptr,
                                          int32_t* __restrict__ out_srcs,
                                          int32_t* __restrict__ out_dsts,
                                          float* __restrict__ out_weights,
                                          unsigned long long* __restrict__ write_pos)
{
  const int32_t ego_size = *ego_size_ptr;
  const int32_t lane_id = threadIdx.x & 31;
  const int32_t warp_id0 = (int32_t)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int32_t total_warps = (int32_t)((gridDim.x * blockDim.x) >> 5);

  for (int32_t warp_id = warp_id0; warp_id < ego_size; warp_id += total_warps) {
    const int32_t v = ego_vertices[warp_id];
    const int32_t start = offsets[v];
    const int32_t end = offsets[v + 1];

    for (int32_t base_e = start; base_e < end; base_e += 32) {
      int32_t e = base_e + lane_id;
      bool active = (e < end);

      int32_t u = 0;
      float w = 0.0f;
      bool is_ego = false;
      if (active) {
        u = indices[e];
        is_ego = (visited[((uint32_t)u) >> 5] & (1u << (u & 31))) != 0;
        w = is_ego ? weights[e] : 0.0f;
      }

      uint32_t ballot = __ballot_sync(0xffffffff, is_ego);
      if (!ballot) continue;

      unsigned long long warp_base = 0;
      if (lane_id == 0) warp_base = atomicAdd(write_pos, (unsigned long long)__popc(ballot));
      warp_base = __shfl_sync(0xffffffff, warp_base, 0);

      if (is_ego) {
        uint32_t lower_mask = (lane_id == 0) ? 0u : ((1u << lane_id) - 1);
        int my_offset = __popc(ballot & lower_mask);
        unsigned long long out_i = warp_base + (unsigned long long)my_offset;
        out_srcs[out_i] = v;
        out_dsts[out_i] = u;
        out_weights[out_i] = w;
      }
    }
  }
}





__global__ void encode_keys_kernel(const int32_t* __restrict__ srcs,
                                   const int32_t* __restrict__ dsts,
                                   const float* __restrict__ weights,
                                   uint32_t* __restrict__ wkeys,
                                   uint64_t* __restrict__ pkeys,
                                   int64_t n)
{
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t i = tid; i < n; i += stride) {
    uint32_t s = (uint32_t)srcs[i];
    uint32_t d = (uint32_t)dsts[i];
    pkeys[i] = ((uint64_t)s << 32) | (uint64_t)d;
    uint32_t bits = __float_as_uint(weights[i]);
    uint32_t mask = (bits >> 31) ? 0xFFFFFFFFu : 0x80000000u;
    wkeys[i] = bits ^ mask;
  }
}

__global__ void decode_keys_kernel(const uint32_t* __restrict__ wkeys,
                                   const uint64_t* __restrict__ pkeys,
                                   int32_t* __restrict__ srcs,
                                   int32_t* __restrict__ dsts,
                                   float* __restrict__ weights,
                                   int64_t n)
{
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t i = tid; i < n; i += stride) {
    uint64_t k = pkeys[i];
    srcs[i] = (int32_t)(k >> 32);
    dsts[i] = (int32_t)(k & 0xffffffffu);
    uint32_t wk = wkeys[i];
    uint32_t mask = (wk >> 31) ? 0x80000000u : 0xFFFFFFFFu;
    weights[i] = __uint_as_float(wk ^ mask);
  }
}





void launch_init_multi_bounds(const int32_t* sources, int32_t n_sources, int32_t num_vertices,
                              uint32_t* bitmaps, int32_t bitmap_words,
                              int32_t* ego_vertices_all, int32_t* ego_sizes,
                              int32_t* frontier_start, int32_t* frontier_end,
                              cudaStream_t stream)
{
  int t = 256;
  int b = (n_sources + t - 1) / t;
  if (b < 1) b = 1;
  init_multi_bounds_kernel<<<b, t, 0, stream>>>(sources, n_sources, num_vertices, bitmaps, bitmap_words,
                                                ego_vertices_all, ego_sizes, frontier_start, frontier_end);
}

void launch_update_bounds_multi(const int32_t* ego_sizes, int32_t* frontier_start, int32_t* frontier_end,
                                int32_t n_sources, cudaStream_t stream)
{
  int t = 256;
  int b = (n_sources + t - 1) / t;
  if (b < 1) b = 1;
  update_bounds_multi_kernel<<<b, t, 0, stream>>>(ego_sizes, frontier_start, frontier_end, n_sources);
}

void launch_bfs_expand_bounds_multi(const int32_t* offsets, const int32_t* indices,
                                    uint32_t* bitmaps, int32_t bitmap_words,
                                    int32_t* ego_vertices_all, int32_t* ego_sizes,
                                    const int32_t* frontier_start, const int32_t* frontier_end,
                                    int32_t num_vertices, int32_t n_sources,
                                    int32_t blocks_per_source, cudaStream_t stream)
{
  dim3 block(256, 1, 1);
  unsigned gy = (unsigned)n_sources; if (gy > 65535u) gy = 65535u;
  dim3 grid((unsigned)blocks_per_source, gy, 1);
  bfs_expand_bounds_multi_kernel<<<grid, block, 0, stream>>>(offsets, indices, bitmaps, bitmap_words,
                                                             ego_vertices_all, ego_sizes,
                                                             frontier_start, frontier_end,
                                                             num_vertices, n_sources);
}

void launch_count_edges_multi(const int32_t* offsets, const int32_t* indices,
                              const uint32_t* bitmaps, int32_t bitmap_words,
                              const int32_t* ego_vertices_all, const int32_t* ego_sizes,
                              int32_t num_vertices, int32_t n_sources,
                              int64_t* edge_counts, int32_t blocks_per_source, cudaStream_t stream)
{
  dim3 block(256, 1, 1);
  unsigned gy = (unsigned)n_sources; if (gy > 65535u) gy = 65535u;
  dim3 grid((unsigned)blocks_per_source, gy, 1);
  count_edges_multi_kernel<<<grid, block, 0, stream>>>(offsets, indices, bitmaps, bitmap_words,
                                                       ego_vertices_all, ego_sizes,
                                                       num_vertices, n_sources, edge_counts);
}

void launch_init_write_pos(const int64_t* offsets, uint64_t* write_pos, int32_t n_sources, cudaStream_t stream)
{
  int t = 256;
  int b = (n_sources + t - 1) / t;
  if (b < 1) b = 1;
  init_write_pos_kernel<<<b, t, 0, stream>>>(offsets, write_pos, n_sources);
}

void launch_extract_edges_multi(const int32_t* offsets, const int32_t* indices, const float* weights,
                                const uint32_t* bitmaps, int32_t bitmap_words,
                                const int32_t* ego_vertices_all, const int32_t* ego_sizes,
                                int32_t num_vertices, int32_t n_sources,
                                int32_t* out_srcs, int32_t* out_dsts, float* out_weights,
                                uint64_t* write_pos, int32_t blocks_per_source, cudaStream_t stream)
{
  dim3 block(256, 1, 1);
  unsigned gy = (unsigned)n_sources; if (gy > 65535u) gy = 65535u;
  dim3 grid((unsigned)blocks_per_source, gy, 1);
  extract_edges_multi_kernel<<<grid, block, 0, stream>>>(offsets, indices, weights,
                                                         bitmaps, bitmap_words,
                                                         ego_vertices_all, ego_sizes,
                                                         num_vertices, n_sources,
                                                         out_srcs, out_dsts, out_weights,
                                                         write_pos);
}

void launch_set_last_offset(int64_t* offsets, const int64_t* counts, int32_t n_sources, cudaStream_t stream)
{
  set_last_offset_kernel<<<1, 1, 0, stream>>>(offsets, counts, n_sources);
}

void launch_init_bfs(uint32_t* visited, int32_t* ego_vertices, int32_t* ego_size,
                     int32_t* frontier_bounds, int32_t source, cudaStream_t stream)
{
  init_bfs_kernel<<<1, 1, 0, stream>>>(visited, ego_vertices, ego_size, frontier_bounds, source);
}

void launch_bfs_expand(const int32_t* offsets, const int32_t* indices,
                       int32_t* ego_vertices, int32_t* ego_size,
                       const int32_t* frontier_bounds, uint32_t* visited,
                       int32_t max_warps, cudaStream_t stream)
{
  int threads = 256;
  int blocks = (int)(((int64_t)max_warps * 32 + threads - 1) / threads);
  if (blocks < 1) blocks = 1;
  if (blocks > 4096) blocks = 4096;
  bfs_expand_warp_kernel<<<blocks, threads, 0, stream>>>(offsets, indices, ego_vertices, ego_size, frontier_bounds, visited);
}

void launch_update_bounds(int32_t* frontier_bounds, int32_t* ego_size, cudaStream_t stream)
{
  update_bounds_kernel<<<1, 1, 0, stream>>>(frontier_bounds, ego_size);
}

void launch_count_edges(const int32_t* offsets, const int32_t* indices,
                        const uint32_t* visited, const int32_t* ego_vertices,
                        const int32_t* ego_size_ptr, unsigned long long* edge_count,
                        int32_t max_warps, cudaStream_t stream)
{
  int threads = 256;
  int blocks = (int)(((int64_t)max_warps * 32 + threads - 1) / threads);
  if (blocks < 1) blocks = 1;
  if (blocks > 4096) blocks = 4096;
  count_edges_warp_kernel<<<blocks, threads, 0, stream>>>(offsets, indices, visited, ego_vertices, ego_size_ptr, edge_count);
}

void launch_collect_bitmap_atomic(const uint32_t* bitmap, int32_t bitmap_words,
                                 int32_t* vertices, int32_t* count, cudaStream_t stream)
{
  int threads = 256;
  int blocks = (bitmap_words + threads - 1) / threads;
  if (blocks < 1) blocks = 1;
  if (blocks > 4096) blocks = 4096;
  collect_bitmap_atomic_kernel<<<blocks, threads, 0, stream>>>(bitmap, bitmap_words, vertices, count);
}

void launch_extract_edges(const int32_t* offsets, const int32_t* indices, const float* weights,
                          const uint32_t* visited, const int32_t* ego_vertices,
                          const int32_t* ego_size_ptr,
                          int32_t* out_srcs, int32_t* out_dsts, float* out_weights,
                          unsigned long long* write_pos,
                          int32_t max_warps, cudaStream_t stream)
{
  int threads = 256;
  int blocks = (int)(((int64_t)max_warps * 32 + threads - 1) / threads);
  if (blocks < 1) blocks = 1;
  if (blocks > 4096) blocks = 4096;
  extract_edges_warp_kernel<<<blocks, threads, 0, stream>>>(offsets, indices, weights, visited, ego_vertices, ego_size_ptr,
                                                           out_srcs, out_dsts, out_weights, write_pos);
}





size_t get_scan_temp_bytes_i64(int n)
{
  size_t temp = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, temp, (const int64_t*)nullptr, (int64_t*)nullptr, n, 0);
  return temp;
}

void do_scan_exclusive_i64(void* temp, size_t temp_bytes, const int64_t* in, int64_t* out, int n, cudaStream_t stream)
{
  cub::DeviceScan::ExclusiveSum(temp, temp_bytes, in, out, n, stream);
}

size_t get_sort_temp_bytes_u32_u64(int64_t n)
{
  size_t t1 = 0, t2 = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, t1,
                                 (uint32_t*)nullptr, (uint32_t*)nullptr,
                                 (uint64_t*)nullptr, (uint64_t*)nullptr,
                                 (int)n, 0, 32, 0);
  cub::DeviceRadixSort::SortPairs(nullptr, t2,
                                 (uint64_t*)nullptr, (uint64_t*)nullptr,
                                 (uint32_t*)nullptr, (uint32_t*)nullptr,
                                 (int)n, 0, 64, 0);
  return (t1 > t2) ? t1 : t2;
}

void launch_sort_edges_u32_u64(int32_t* srcs, int32_t* dsts, float* weights, int64_t n,
                               uint32_t* w0, uint32_t* w1, uint64_t* k0, uint64_t* k1,
                               void* temp, size_t temp_bytes, cudaStream_t stream)
{
  if (n <= 1) return;

  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);
  if (blocks < 1) blocks = 1;
  if (blocks > 4096) blocks = 4096;

  encode_keys_kernel<<<blocks, threads, 0, stream>>>(srcs, dsts, weights, w0, k0, n);
  cub::DeviceRadixSort::SortPairs(temp, temp_bytes, w0, w1, k0, k1, (int)n, 0, 32, stream);
  cub::DeviceRadixSort::SortPairs(temp, temp_bytes, k1, k0, w1, w0, (int)n, 0, 64, stream);
  decode_keys_kernel<<<blocks, threads, 0, stream>>>(w0, k0, srcs, dsts, weights, n);
}





struct Cache : Cacheable {
  
  uint32_t* d_bitmaps_ = nullptr;
  int32_t* d_ego_all_ = nullptr;
  int32_t* d_ego_sizes_ = nullptr;
  int32_t* d_frontier_start_ = nullptr;
  int32_t* d_frontier_end_ = nullptr;
  int64_t* d_edge_counts_i64_ = nullptr;
  uint64_t* d_write_pos_u64_ = nullptr;

  void* d_scan_temp_ = nullptr;
  size_t scan_temp_bytes_ = 0;

  
  uint32_t* d_w0_ = nullptr;
  uint32_t* d_w1_ = nullptr;
  uint64_t* d_k0_ = nullptr;
  uint64_t* d_k1_ = nullptr;
  void* d_sort_temp_ = nullptr;
  size_t sort_temp_bytes_ = 0;
  int64_t sort_capacity_ = 0;

  
  uint32_t* d_all_bitmaps_ = nullptr;
  int32_t* d_ego_vertices_ = nullptr;
  int32_t* d_ego_size_ = nullptr;
  int32_t* d_frontier_bounds_ = nullptr;
  unsigned long long* d_edge_counts_ = nullptr;
  unsigned long long* d_write_pos_ = nullptr;

  
  size_t cap_bitmap_words_total_ = 0;
  size_t cap_product_ = 0;
  int32_t cap_nsrc_ = 0;

  size_t cap_bitmap_total_words_legacy_ = 0;
  int32_t cap_nv_legacy_ = 0;
  int32_t cap_nsrc_legacy_ = 0;

  void ensure_multi_scratch(int32_t nv, int32_t nsrc) {
    const int32_t bw = (nv + 31) / 32;
    const size_t bitmap_total = (size_t)nsrc * (size_t)bw;
    const size_t product = (size_t)nsrc * (size_t)nv;

    if (bitmap_total > cap_bitmap_words_total_) {
      if (d_bitmaps_) cudaFree(d_bitmaps_);
      cudaMalloc(&d_bitmaps_, bitmap_total * sizeof(uint32_t));
      cap_bitmap_words_total_ = bitmap_total;
    }

    if (product > cap_product_) {
      if (d_ego_all_) cudaFree(d_ego_all_);
      cudaMalloc(&d_ego_all_, product * sizeof(int32_t));
      cap_product_ = product;
    }

    if (nsrc > cap_nsrc_) {
      auto freep = [](auto& p) {
        if (p) cudaFree(p);
        p = nullptr;
      };
      freep(d_ego_sizes_);
      freep(d_frontier_start_);
      freep(d_frontier_end_);
      freep(d_edge_counts_i64_);
      freep(d_write_pos_u64_);

      cudaMalloc(&d_ego_sizes_, (size_t)nsrc * sizeof(int32_t));
      cudaMalloc(&d_frontier_start_, (size_t)nsrc * sizeof(int32_t));
      cudaMalloc(&d_frontier_end_, (size_t)nsrc * sizeof(int32_t));
      cudaMalloc(&d_edge_counts_i64_, (size_t)nsrc * sizeof(int64_t));
      cudaMalloc(&d_write_pos_u64_, (size_t)nsrc * sizeof(uint64_t));
      cap_nsrc_ = nsrc;

      size_t need = get_scan_temp_bytes_i64(nsrc);
      if (need > scan_temp_bytes_) {
        if (d_scan_temp_) cudaFree(d_scan_temp_);
        scan_temp_bytes_ = need + 1024;
        cudaMalloc(&d_scan_temp_, scan_temp_bytes_);
      }
    }
  }

  void ensure_legacy_scratch(int32_t nv, int32_t nsrc) {
    const int32_t bw = (nv + 31) / 32;
    const size_t bitmap_total_words = (size_t)nsrc * (size_t)bw;

    if (bitmap_total_words > cap_bitmap_total_words_legacy_) {
      if (d_all_bitmaps_) cudaFree(d_all_bitmaps_);
      cudaMalloc(&d_all_bitmaps_, bitmap_total_words * sizeof(uint32_t));
      cap_bitmap_total_words_legacy_ = bitmap_total_words;
    }

    if (nv > cap_nv_legacy_) {
      if (d_ego_vertices_) cudaFree(d_ego_vertices_);
      cudaMalloc(&d_ego_vertices_, (size_t)nv * sizeof(int32_t));
      cap_nv_legacy_ = nv;
    }

    if (nsrc > cap_nsrc_legacy_) {
      if (d_edge_counts_) cudaFree(d_edge_counts_);
      cudaMalloc(&d_edge_counts_, (size_t)nsrc * sizeof(unsigned long long));
      cap_nsrc_legacy_ = nsrc;
    }

    if (!d_ego_size_) cudaMalloc(&d_ego_size_, sizeof(int32_t));
    if (!d_frontier_bounds_) cudaMalloc(&d_frontier_bounds_, 2 * sizeof(int32_t));
    if (!d_write_pos_) cudaMalloc(&d_write_pos_, sizeof(unsigned long long));
  }

  void ensure_sort_scratch(int64_t n) {
    if (n <= 1) return;
    if (n <= sort_capacity_) return;

    if (d_w0_) cudaFree(d_w0_);
    if (d_w1_) cudaFree(d_w1_);
    if (d_k0_) cudaFree(d_k0_);
    if (d_k1_) cudaFree(d_k1_);
    if (d_sort_temp_) cudaFree(d_sort_temp_);

    sort_capacity_ = n;
    cudaMalloc(&d_w0_, (size_t)n * sizeof(uint32_t));
    cudaMalloc(&d_w1_, (size_t)n * sizeof(uint32_t));
    cudaMalloc(&d_k0_, (size_t)n * sizeof(uint64_t));
    cudaMalloc(&d_k1_, (size_t)n * sizeof(uint64_t));

    sort_temp_bytes_ = get_sort_temp_bytes_u32_u64(n);
    cudaMalloc(&d_sort_temp_, sort_temp_bytes_);
  }

  ~Cache() override {
    auto freep = [](auto& p) {
      if (p) cudaFree(p);
      p = nullptr;
    };

    freep(d_bitmaps_);
    freep(d_ego_all_);
    freep(d_ego_sizes_);
    freep(d_frontier_start_);
    freep(d_frontier_end_);
    freep(d_edge_counts_i64_);
    freep(d_write_pos_u64_);
    freep(d_scan_temp_);

    freep(d_w0_);
    freep(d_w1_);
    freep(d_k0_);
    freep(d_k1_);
    freep(d_sort_temp_);

    freep(d_all_bitmaps_);
    freep(d_ego_vertices_);
    freep(d_ego_size_);
    freep(d_frontier_bounds_);
    freep(d_edge_counts_);
    freep(d_write_pos_);
  }
};

}  

extract_ego_weighted_result_float_t extract_ego_weighted_f32_seg(
    const graph32_t& graph,
    const float* edge_weights,
    const int32_t* source_vertices,
    std::size_t n_sources,
    int32_t radius) {

  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);

  const int32_t nv = graph.number_of_vertices;
  const int32_t nsrc = (int32_t)n_sources;
  const bool is_multigraph = graph.is_multigraph;

  const int32_t bw = (nv + 31) / 32;
  const int32_t* d_offsets = graph.offsets;
  const int32_t* d_indices = graph.indices;
  const float* d_weights = edge_weights;
  const int32_t* d_sources = source_vertices;

  cudaStream_t stream = 0;

  
  const size_t product = (size_t)nsrc * (size_t)nv;
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);

  size_t scan_need = cache.scan_temp_bytes_;
  if (scan_need == 0) scan_need = get_scan_temp_bytes_i64(nsrc);
  const size_t need_b = (size_t)nsrc * (size_t)bw * sizeof(uint32_t) +
                        product * sizeof(int32_t) +
                        (size_t)nsrc * (sizeof(int32_t) * 3 + sizeof(int64_t) + sizeof(uint64_t)) +
                        scan_need;

  if (need_b > free_b) {
    
    cache.ensure_legacy_scratch(nv, nsrc);

    std::vector<int32_t> h_sources((size_t)nsrc);
    if (nsrc > 0) {
      cudaMemcpyAsync(h_sources.data(), d_sources, (size_t)nsrc * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    }
    const int32_t max_warps = 32768;

    cudaMemsetAsync(cache.d_all_bitmaps_, 0, (size_t)nsrc * (size_t)bw * sizeof(uint32_t), stream);
    cudaMemsetAsync(cache.d_edge_counts_, 0, (size_t)nsrc * sizeof(unsigned long long), stream);

    for (int s = 0; s < nsrc; s++) {
      uint32_t* bm = cache.d_all_bitmaps_ + (size_t)s * (size_t)bw;
      launch_init_bfs(bm, cache.d_ego_vertices_, cache.d_ego_size_, cache.d_frontier_bounds_, h_sources[(size_t)s], stream);
      for (int hop = 0; hop < radius; hop++) {
        launch_bfs_expand(d_offsets, d_indices, cache.d_ego_vertices_, cache.d_ego_size_, cache.d_frontier_bounds_, bm, max_warps, stream);
        if (hop < radius - 1) launch_update_bounds(cache.d_frontier_bounds_, cache.d_ego_size_, stream);
      }
      launch_count_edges(d_offsets, d_indices, bm, cache.d_ego_vertices_, cache.d_ego_size_, &cache.d_edge_counts_[s], max_warps, stream);
    }

    std::vector<unsigned long long> h_edge_counts((size_t)nsrc);
    if (nsrc > 0) {
      cudaMemcpyAsync(h_edge_counts.data(), cache.d_edge_counts_, (size_t)nsrc * sizeof(unsigned long long), cudaMemcpyDeviceToHost, stream);
    }
    cudaStreamSynchronize(stream);

    std::vector<int64_t> h_offsets((size_t)nsrc + 1, 0);
    unsigned long long max_ecnt = 0;
    for (int s = 0; s < nsrc; s++) {
      auto ecnt = h_edge_counts[(size_t)s];
      h_offsets[(size_t)s + 1] = h_offsets[(size_t)s] + (int64_t)ecnt;
      max_ecnt = std::max(max_ecnt, ecnt);
    }
    int64_t total_edges = h_offsets[(size_t)nsrc];

    int32_t* out_srcs = nullptr;
    int32_t* out_dsts = nullptr;
    float* out_w = nullptr;
    int64_t* out_off = nullptr;
    if (total_edges > 0) {
      cudaMalloc(&out_srcs, (size_t)total_edges * sizeof(int32_t));
      cudaMalloc(&out_dsts, (size_t)total_edges * sizeof(int32_t));
      cudaMalloc(&out_w, (size_t)total_edges * sizeof(float));
    }
    cudaMalloc(&out_off, ((size_t)nsrc + 1) * sizeof(int64_t));

    if (is_multigraph && max_ecnt > 1) cache.ensure_sort_scratch((int64_t)max_ecnt);

    for (int s = 0; s < nsrc; s++) {
      auto ecnt = h_edge_counts[(size_t)s];
      if (!ecnt) continue;
      uint32_t* bm = cache.d_all_bitmaps_ + (size_t)s * (size_t)bw;
      cudaMemsetAsync(cache.d_ego_size_, 0, sizeof(int32_t), stream);
      launch_collect_bitmap_atomic(bm, bw, cache.d_ego_vertices_, cache.d_ego_size_, stream);
      cudaMemsetAsync(cache.d_write_pos_, 0, sizeof(unsigned long long), stream);
      int32_t* seg_s = out_srcs + h_offsets[(size_t)s];
      int32_t* seg_d = out_dsts + h_offsets[(size_t)s];
      float* seg_w = out_w + h_offsets[(size_t)s];
      launch_extract_edges(d_offsets, d_indices, d_weights, bm, cache.d_ego_vertices_, cache.d_ego_size_, seg_s, seg_d, seg_w, cache.d_write_pos_, max_warps, stream);
      if (is_multigraph && ecnt > 1) {
        launch_sort_edges_u32_u64(seg_s, seg_d, seg_w, (int64_t)ecnt, cache.d_w0_, cache.d_w1_, cache.d_k0_, cache.d_k1_, cache.d_sort_temp_, cache.sort_temp_bytes_, stream);
      }
    }

    cudaMemcpyAsync(out_off, h_offsets.data(), ((size_t)nsrc + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    return extract_ego_weighted_result_float_t{
      out_srcs,
      out_dsts,
      out_w,
      reinterpret_cast<std::size_t*>(out_off),
      (std::size_t)total_edges,
      (std::size_t)nsrc + 1,
    };
  }

  
  cache.ensure_multi_scratch(nv, nsrc);

  cudaMemsetAsync(cache.d_bitmaps_, 0, (size_t)nsrc * (size_t)bw * sizeof(uint32_t), stream);
  cudaMemsetAsync(cache.d_ego_sizes_, 0, (size_t)nsrc * sizeof(int32_t), stream);

  launch_init_multi_bounds(d_sources, nsrc, nv, cache.d_bitmaps_, bw, cache.d_ego_all_, cache.d_ego_sizes_, cache.d_frontier_start_, cache.d_frontier_end_, stream);

  const int32_t bfs_blocks = 1024;
  for (int hop = 0; hop < radius; hop++) {
    launch_bfs_expand_bounds_multi(d_offsets, d_indices, cache.d_bitmaps_, bw, cache.d_ego_all_, cache.d_ego_sizes_, cache.d_frontier_start_, cache.d_frontier_end_, nv, nsrc,
                                   bfs_blocks, stream);
    if (hop < radius - 1) {
      launch_update_bounds_multi(cache.d_ego_sizes_, cache.d_frontier_start_, cache.d_frontier_end_, nsrc, stream);
    }
  }

  cudaMemsetAsync(cache.d_edge_counts_i64_, 0, (size_t)nsrc * sizeof(int64_t), stream);
  const int32_t count_blocks = 256;
  launch_count_edges_multi(d_offsets, d_indices, cache.d_bitmaps_, bw, cache.d_ego_all_, cache.d_ego_sizes_, nv, nsrc, cache.d_edge_counts_i64_, count_blocks, stream);

  int64_t* d_out_offsets = nullptr;
  cudaMalloc(&d_out_offsets, ((size_t)nsrc + 1) * sizeof(int64_t));

  if (nsrc > 0) {
    do_scan_exclusive_i64(cache.d_scan_temp_, cache.scan_temp_bytes_, cache.d_edge_counts_i64_, d_out_offsets, nsrc, stream);
    launch_set_last_offset(d_out_offsets, cache.d_edge_counts_i64_, nsrc, stream);
  } else {
    cudaMemsetAsync(d_out_offsets, 0, sizeof(int64_t), stream);
  }

  int64_t total_edges = 0;
  if (nsrc > 0) {
    cudaMemcpyAsync(&total_edges, d_out_offsets + nsrc, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  }
  cudaStreamSynchronize(stream);

  int32_t* out_srcs = nullptr;
  int32_t* out_dsts = nullptr;
  float* out_w = nullptr;
  if (total_edges > 0) {
    cudaMalloc(&out_srcs, (size_t)total_edges * sizeof(int32_t));
    cudaMalloc(&out_dsts, (size_t)total_edges * sizeof(int32_t));
    cudaMalloc(&out_w, (size_t)total_edges * sizeof(float));
  }

  launch_init_write_pos(d_out_offsets, cache.d_write_pos_u64_, nsrc, stream);

  const int32_t extract_blocks = 256;
  launch_extract_edges_multi(d_offsets, d_indices, d_weights, cache.d_bitmaps_, bw, cache.d_ego_all_, cache.d_ego_sizes_, nv, nsrc,
                             out_srcs, out_dsts, out_w, cache.d_write_pos_u64_,
                             extract_blocks, stream);

  if (is_multigraph && total_edges > 1) {
    std::vector<int64_t> h_off((size_t)nsrc + 1);
    cudaMemcpyAsync(h_off.data(), d_out_offsets, ((size_t)nsrc + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int64_t max_len = 0;
    for (int s = 0; s < nsrc; s++) max_len = std::max(max_len, h_off[(size_t)s + 1] - h_off[(size_t)s]);
    if (max_len > 1) cache.ensure_sort_scratch(max_len);

    for (int s = 0; s < nsrc; s++) {
      int64_t beg = h_off[(size_t)s];
      int64_t end = h_off[(size_t)s + 1];
      int64_t len = end - beg;
      if (len <= 1) continue;
      launch_sort_edges_u32_u64(out_srcs + beg,
                                out_dsts + beg,
                                out_w + beg,
                                len, cache.d_w0_, cache.d_w1_, cache.d_k0_, cache.d_k1_, cache.d_sort_temp_, cache.sort_temp_bytes_, stream);
    }
  }

  cudaStreamSynchronize(stream);

  return extract_ego_weighted_result_float_t{
    out_srcs,
    out_dsts,
    out_w,
    reinterpret_cast<std::size_t*>(d_out_offsets),
    (std::size_t)total_edges,
    (std::size_t)nsrc + 1,
  };
}

}  
