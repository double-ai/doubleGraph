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

namespace aai {

namespace {

struct Cache : Cacheable {};

__device__ __forceinline__ int32_t find_root(int32_t* __restrict__ parent, int32_t x) {
  int32_t orig = x;
  int32_t p = parent[x];
  while (p != x) {
    x = p;
    p = parent[x];
  }
  
  int32_t old = parent[orig];
  if (x < old) atomicMin(&parent[orig], x);
  return x;
}

__device__ __forceinline__ void unite(int32_t* __restrict__ parent, int32_t x, int32_t y) {
  while (true) {
    x = find_root(parent, x);
    y = find_root(parent, y);
    if (x == y) return;
    if (x > y) {
      int32_t t = x;
      x = y;
      y = t;
    }
    if (atomicCAS(&parent[y], y, x) == y) return;
  }
}

__global__ void init_kernel(int32_t* __restrict__ parent, int32_t n) {
  for (int32_t v = (int32_t)blockIdx.x * blockDim.x + threadIdx.x; v < n;
       v += (int32_t)blockDim.x * gridDim.x) {
    parent[v] = v;
  }
}

__global__ void union_warp_vertex(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ parent,
    int32_t n) {

  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const int warp_global = (int)blockIdx.x * warps_per_block + warp_in_block;
  const int total_warps = (int)gridDim.x * warps_per_block;

  for (int32_t v = warp_global; v < n; v += total_warps) {
    int32_t start, end;
    if (lane == 0) {
      start = __ldg(&offsets[v]);
      end = __ldg(&offsets[v + 1]);
    }
    start = __shfl_sync(0xffffffffu, start, 0);
    end = __shfl_sync(0xffffffffu, end, 0);

    int32_t chunk = start & ~31;
    for (; chunk < end; chunk += 32) {
      uint32_t word;
      if (lane == 0) word = __ldg(&edge_mask[(uint32_t)chunk >> 5]);
      word = __shfl_sync(0xffffffffu, word, 0);
      if (word == 0u) continue;

      int32_t e = chunk + lane;
      const bool valid = (e >= start) & (e < end);
      int32_t u = 0;
      if (valid) u = __ldg(&indices[e]);

      if (valid && ((word >> lane) & 1u)) {
        
        
        int32_t pv = parent[v];
        int32_t pu = parent[u];
        if (pv != pu) {
          unite(parent, v, u);
        }
      }
    }
  }
}

__global__ void flatten(int32_t* __restrict__ parent, int32_t n) {
  for (int32_t v = (int32_t)blockIdx.x * blockDim.x + threadIdx.x; v < n;
       v += (int32_t)blockDim.x * gridDim.x) {
    parent[v] = find_root(parent, v);
  }
}

}  

void weakly_connected_components_mask(const graph32_t& graph,
                                      int32_t* components) {
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);
  (void)cache;

  const int32_t* offsets = graph.offsets;
  const int32_t* indices = graph.indices;
  const uint32_t* edge_mask = graph.edge_mask;
  int32_t num_vertices = graph.number_of_vertices;

  if (num_vertices <= 0) return;

  constexpr int block = 256;
  constexpr int SMS = 108;
  constexpr int warps_per_block = block / 32;
  constexpr int max_blocks = SMS * (64 / warps_per_block);

  int grid = (num_vertices + block - 1) / block;
  if (grid > max_blocks) grid = max_blocks;

  init_kernel<<<grid, block>>>(components, num_vertices);
  union_warp_vertex<<<grid, block>>>(offsets, indices, edge_mask, components, num_vertices);
  flatten<<<grid, block>>>(components, num_vertices);
}

}  
