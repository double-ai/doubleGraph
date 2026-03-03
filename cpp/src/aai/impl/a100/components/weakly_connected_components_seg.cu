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

struct Cache : Cacheable {
    int* h_tmp_pinned = nullptr;

    Cache() {
        cudaMallocHost(&h_tmp_pinned, sizeof(int));
    }

    ~Cache() override {
        if (h_tmp_pinned) {
            cudaFreeHost(h_tmp_pinned);
            h_tmp_pinned = nullptr;
        }
    }
};







static __forceinline__ __device__ int32_t find_root(const int32_t* __restrict__ parent, int32_t v) {
  int32_t p = parent[v];
#pragma unroll 1
  while (p != v) {
    
    int32_t gp = parent[p];
    v = gp;
    p = parent[v];
  }
  return v;
}

static __forceinline__ __device__ void unite(int32_t* __restrict__ parent, int32_t a, int32_t b) {
  int32_t ra = find_root(parent, a);
  int32_t rb = find_root(parent, b);

#pragma unroll 1
  while (ra != rb) {
    int32_t hi = (ra > rb) ? ra : rb;
    int32_t lo = (ra > rb) ? rb : ra;

    int old = atomicCAS((int*)(&parent[hi]), (int)hi, (int)lo);
    if (old == hi) return;

    
    ra = find_root(parent, ra);
    rb = find_root(parent, rb);
  }
}





__global__ void init_kernel(int32_t* __restrict__ parent, int32_t n) {
  int32_t tid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t stride = (int32_t)(gridDim.x * blockDim.x);
  for (int32_t v = tid; v < n; v += stride) parent[v] = v;
}

__global__ void compress_kernel(int32_t* __restrict__ parent, int32_t n) {
  int32_t tid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t stride = (int32_t)(gridDim.x * blockDim.x);
  for (int32_t v = tid; v < n; v += stride) {
    int32_t r = find_root(parent, v);
    parent[v] = r;
  }
}


__global__ void hook_sample_kernel(const int32_t* __restrict__ offsets,
                                  const int32_t* __restrict__ indices,
                                  int32_t* __restrict__ parent,
                                  int32_t n,
                                  int32_t k) {
  int32_t tid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t stride = (int32_t)(gridDim.x * blockDim.x);

  for (int32_t v = tid; v < n; v += stride) {
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t limit = start + k;
    if (limit > end) limit = end;

    for (int32_t e = start; e < limit; ++e) {
      int32_t u = indices[e];
      if (u == v) continue;
      unite(parent, v, u);
    }
  }
}


__global__ void hook_remaining_kernel(const int32_t* __restrict__ offsets,
                                     const int32_t* __restrict__ indices,
                                     int32_t* __restrict__ parent,
                                     int32_t n,
                                     int32_t L) {
  int32_t tid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t stride = (int32_t)(gridDim.x * blockDim.x);

  for (int32_t v = tid; v < n; v += stride) {
    if (parent[v] == L) continue;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    for (int32_t e = start; e < end; ++e) {
      int32_t u = indices[e];
      if (u == v) continue;
      unite(parent, v, u);
    }
  }
}



__global__ void hook_high_degree(const int32_t* __restrict__ offsets,
                                const int32_t* __restrict__ indices,
                                int32_t* __restrict__ parent,
                                int32_t start_v,
                                int32_t end_v) {
  int32_t v = start_v + (int32_t)blockIdx.x;
  if (v >= end_v) return;

  int32_t edge_start = offsets[v];
  int32_t edge_end = offsets[v + 1];

  for (int32_t e = edge_start + (int32_t)threadIdx.x; e < edge_end; e += (int32_t)blockDim.x) {
    int32_t u = indices[e];
    if (u <= v) continue;
    unite(parent, v, u);
  }
}


__global__ void hook_mid_degree(const int32_t* __restrict__ offsets,
                               const int32_t* __restrict__ indices,
                               int32_t* __restrict__ parent,
                               int32_t start_v,
                               int32_t end_v) {
  int32_t warps_per_block = (int32_t)(blockDim.x >> 5);
  int32_t warp_id = (int32_t)(threadIdx.x >> 5);
  int32_t lane = (int32_t)(threadIdx.x & 31);

  int32_t v = start_v + (int32_t)blockIdx.x * warps_per_block + warp_id;
  if (v >= end_v) return;

  int32_t edge_start = offsets[v];
  int32_t edge_end = offsets[v + 1];

  for (int32_t e = edge_start + lane; e < edge_end; e += 32) {
    int32_t u = indices[e];
    if (u <= v) continue;
    unite(parent, v, u);
  }
}


__global__ void hook_low_degree(const int32_t* __restrict__ offsets,
                               const int32_t* __restrict__ indices,
                               int32_t* __restrict__ parent,
                               int32_t start_v,
                               int32_t end_v) {
  int32_t tid = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
  int32_t stride = (int32_t)(gridDim.x * blockDim.x);

  for (int32_t v = start_v + tid; v < end_v; v += stride) {
    int32_t edge_start = offsets[v];
    int32_t edge_end = offsets[v + 1];

    for (int32_t e = edge_start; e < edge_end; ++e) {
      int32_t u = indices[e];
      if (u <= v) continue;
      unite(parent, v, u);
    }
  }
}





static void launch_wcc_afforest(const int32_t* offsets, const int32_t* indices,
                                int32_t* parent,
                                int32_t n, int32_t m,
                                int32_t seg1, int32_t seg2, int32_t seg3,
                                int* h_tmp_pinned,
                                cudaStream_t stream) {
  if (n <= 0) return;

  constexpr int BLOCK = 256;

  
  int grid = (n + BLOCK - 1) / BLOCK;
  if (grid > 4096) grid = 4096;
  if (grid < 1) grid = 1;

  init_kernel<<<grid, BLOCK, 0, stream>>>(parent, n);

  float avg_degree = (n > 0) ? ((float)m / (float)n) : 0.0f;

  if (avg_degree >= 6.0f && n >= 5000) {
    
    constexpr int k = 2;

    hook_sample_kernel<<<grid, BLOCK, 0, stream>>>(offsets, indices, parent, n, k);
    compress_kernel<<<grid, BLOCK, 0, stream>>>(parent, n);

    
    cudaMemcpyAsync(h_tmp_pinned, parent, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int32_t L = *(int32_t*)h_tmp_pinned;

    hook_remaining_kernel<<<grid, BLOCK, 0, stream>>>(offsets, indices, parent, n, L);
    compress_kernel<<<grid, BLOCK, 0, stream>>>(parent, n);
  } else {
    
    int32_t num_high = seg1;
    int32_t num_mid = seg2 - seg1;
    int32_t num_low = seg3 - seg2;

    if (num_high > 0) {
      hook_high_degree<<<num_high, BLOCK, 0, stream>>>(offsets, indices, parent, 0, seg1);
    }
    if (num_mid > 0) {
      int warps_per_block = BLOCK / 32;
      int grid_mid = (num_mid + warps_per_block - 1) / warps_per_block;
      if (grid_mid > 4096) grid_mid = 4096;
      hook_mid_degree<<<grid_mid, BLOCK, 0, stream>>>(offsets, indices, parent, seg1, seg2);
    }
    if (num_low > 0) {
      int grid_low = (num_low + BLOCK - 1) / BLOCK;
      if (grid_low > 4096) grid_low = 4096;
      hook_low_degree<<<grid_low, BLOCK, 0, stream>>>(offsets, indices, parent, seg2, seg3);
    }

    compress_kernel<<<grid, BLOCK, 0, stream>>>(parent, n);
  }
}

}  

void weakly_connected_components_seg(const graph32_t& graph,
                                     int32_t* components) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t n = graph.number_of_vertices;
    int32_t m = graph.number_of_edges;

    if (n == 0) return;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg1 = seg[1];
    int32_t seg2 = seg[2];
    int32_t seg3 = seg[3];

    cudaStream_t stream = 0;

    launch_wcc_afforest(offsets, indices, components,
                        n, m, seg1, seg2, seg3,
                        cache.h_tmp_pinned, stream);
}

}  
