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
#include <algorithm>
#include <cub/block/block_reduce.cuh>

namespace aai {

namespace {

struct Cache : Cacheable {
    
    double* result = nullptr;
    int32_t* hub_count = nullptr;
    int32_t* hub_iter = nullptr;

    
    int32_t* sizes = nullptr;
    int64_t sizes_capacity = 0;

    int32_t* hub_q = nullptr;
    int64_t hub_q_capacity = 0;

    uint8_t* packed_u8 = nullptr;
    int64_t packed_u8_capacity = 0;

    int16_t* packed_i16 = nullptr;
    int64_t packed_i16_capacity = 0;

    Cache() {
        cudaMalloc(&result, sizeof(double));
        cudaMalloc(&hub_count, sizeof(int32_t));
        cudaMalloc(&hub_iter, sizeof(int32_t));
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 16 * 1024 * 1024);
    }

    ~Cache() override {
        if (result) cudaFree(result);
        if (hub_count) cudaFree(hub_count);
        if (hub_iter) cudaFree(hub_iter);
        if (sizes) cudaFree(sizes);
        if (hub_q) cudaFree(hub_q);
        if (packed_u8) cudaFree(packed_u8);
        if (packed_i16) cudaFree(packed_i16);
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
    }

    void ensure(int32_t nv, int32_t nc) {
        if (sizes_capacity < nc) {
            if (sizes) cudaFree(sizes);
            cudaMalloc(&sizes, (size_t)nc * sizeof(int32_t));
            sizes_capacity = nc;
        }
        if (hub_q_capacity < nv) {
            if (hub_q) cudaFree(hub_q);
            cudaMalloc(&hub_q, (size_t)nv * sizeof(int32_t));
            hub_q_capacity = nv;
        }
        if (nc <= 255) {
            if (packed_u8_capacity < nv) {
                if (packed_u8) cudaFree(packed_u8);
                cudaMalloc(&packed_u8, (size_t)nv * sizeof(uint8_t));
                packed_u8_capacity = nv;
            }
        } else if (nc <= 32767) {
            if (packed_i16_capacity < nv) {
                if (packed_i16) cudaFree(packed_i16);
                cudaMalloc(&packed_i16, (size_t)nv * sizeof(int16_t));
                packed_i16_capacity = nv;
            }
        }
    }
};





__global__ void pack_u8_kernel(const int32_t* __restrict__ in, uint8_t* __restrict__ out, int32_t n, int32_t nc) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    int32_t c = in[i];
    if ((uint32_t)c >= (uint32_t)nc) c = (c < 0) ? 0 : (nc - 1);
    out[i] = static_cast<uint8_t>(c);
  }
}

__global__ void pack_i16_kernel(const int32_t* __restrict__ in, int16_t* __restrict__ out, int32_t n, int32_t nc) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    int32_t c = in[i];
    if ((uint32_t)c >= (uint32_t)nc) c = (c < 0) ? 0 : (nc - 1);
    out[i] = static_cast<int16_t>(c);
  }
}

template <typename ClusterT, int BLOCK>
__global__ __launch_bounds__(BLOCK, 8) void cluster_sizes_shared_kernel(const ClusterT* __restrict__ clusters,
                                                                       int32_t* __restrict__ sizes, int32_t nv,
                                                                       int32_t nc) {
  extern __shared__ int32_t s_sizes[];
  for (int i = threadIdx.x; i < nc; i += BLOCK) s_sizes[i] = 0;
  __syncthreads();

  for (int u = (int)blockIdx.x * BLOCK + threadIdx.x; u < nv; u += (int)gridDim.x * BLOCK) {
    int c = (int)clusters[u];
    atomicAdd(&s_sizes[c], 1);
  }

  __syncthreads();
  for (int i = threadIdx.x; i < nc; i += BLOCK) {
    int v = s_sizes[i];
    if (v) atomicAdd(&sizes[i], v);
  }
}

template <typename ClusterT, int BLOCK>
__global__ __launch_bounds__(BLOCK, 8) void cluster_sizes_global_kernel(const ClusterT* __restrict__ clusters,
                                                                       int32_t* __restrict__ sizes, int32_t nv) {
  for (int u = (int)blockIdx.x * BLOCK + threadIdx.x; u < nv; u += (int)gridDim.x * BLOCK) {
    int c = (int)clusters[u];
    atomicAdd(&sizes[c], 1);
  }
}

__device__ __forceinline__ double warp_reduce_sum(double v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, offset);
  return v;
}

template <typename ClusterT, int BLOCK>
__global__ __launch_bounds__(BLOCK, 6) void ratio_cut_tpv_kernel(const int32_t* __restrict__ offsets,
                                                                 const int32_t* __restrict__ indices,
                                                                 const float* __restrict__ ew,
                                                                 const ClusterT* __restrict__ clusters,
                                                                 const int32_t* __restrict__ sizes, int32_t nv,
                                                                 int32_t hub_thresh, int32_t* __restrict__ hub_q,
                                                                 int32_t* __restrict__ hub_count, int32_t max_hubs,
                                                                 double* __restrict__ out) {
  double thread_sum = 0.0;
  for (int u = (int)blockIdx.x * BLOCK + threadIdx.x; u < nv; u += (int)gridDim.x * BLOCK) {
    int start = __ldg(&offsets[u]);
    int end = __ldg(&offsets[u + 1]);
    int deg = end - start;

    if (deg >= hub_thresh) {
      int pos = atomicAdd(hub_count, 1);
      if (pos < max_hubs) hub_q[pos] = u;
      continue;
    }

    int cu = (int)__ldg(&clusters[u]);
    int sz = __ldg(&sizes[cu]);

    float local_cut = 0.0f;
    for (int e = start; e < end; ++e) {
      int v = __ldg(&indices[e]);
      int cv = (int)__ldg(&clusters[v]);
      if (cv != cu) local_cut += __ldg(&ew[e]);
    }
    thread_sum += (double)local_cut / (double)sz;
  }

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  double v = warp_reduce_sum(thread_sum);
  __shared__ double warp_sums[8]; 
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  if (warp == 0) {
    double sum = (lane < (BLOCK >> 5)) ? warp_sums[lane] : 0.0;
    sum = warp_reduce_sum(sum);
    if (lane == 0) atomicAdd(out, sum);
  }
}

template <typename ClusterT, int BLOCK>
__global__ __launch_bounds__(BLOCK, 6) void ratio_cut_wpv_kernel(const int32_t* __restrict__ offsets,
                                                                 const int32_t* __restrict__ indices,
                                                                 const float* __restrict__ ew,
                                                                 const ClusterT* __restrict__ clusters,
                                                                 const int32_t* __restrict__ sizes, int32_t nv,
                                                                 int32_t hub_thresh, int32_t* __restrict__ hub_q,
                                                                 int32_t* __restrict__ hub_count, int32_t max_hubs,
                                                                 double* __restrict__ out) {
  constexpr int WARPS_PER_BLOCK = BLOCK >> 5;
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int global_warp = (int)blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
  const int total_warps = (int)gridDim.x * WARPS_PER_BLOCK;

  double warp_accum = 0.0;
  for (int u = global_warp; u < nv; u += total_warps) {
    int start = __ldg(&offsets[u]);
    int end = __ldg(&offsets[u + 1]);
    int deg = end - start;

    if (deg >= hub_thresh) {
      if (lane == 0) {
        int pos = atomicAdd(hub_count, 1);
        if (pos < max_hubs) hub_q[pos] = u;
      }
      continue;
    }

    int cu = (int)__ldg(&clusters[u]);
    int sz = __ldg(&sizes[cu]);

    float local_cut = 0.0f;
    for (int e = start + lane; e < end; e += 32) {
      int v = __ldg(&indices[e]);
      int cv = (int)__ldg(&clusters[v]);
      if (cv != cu) local_cut += __ldg(&ew[e]);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) local_cut += __shfl_down_sync(0xFFFFFFFF, local_cut, offset);

    if (lane == 0) warp_accum += (double)local_cut / (double)sz;
  }

  __shared__ double warp_sums[WARPS_PER_BLOCK];
  if (lane == 0) warp_sums[warp_in_block] = warp_accum;
  __syncthreads();

  if (threadIdx.x == 0) {
    double block_sum = 0.0;
#pragma unroll
    for (int w = 0; w < WARPS_PER_BLOCK; ++w) block_sum += warp_sums[w];
    if (block_sum != 0.0) atomicAdd(out, block_sum);
  }
}

template <typename ClusterT, int BLOCK>
__global__ __launch_bounds__(BLOCK, 4) void ratio_cut_hubs_kernel(const int32_t* __restrict__ offsets,
                                                                  const int32_t* __restrict__ indices,
                                                                  const float* __restrict__ ew,
                                                                  const ClusterT* __restrict__ clusters,
                                                                  const int32_t* __restrict__ sizes,
                                                                  const int32_t* __restrict__ hub_q,
                                                                  const int32_t* __restrict__ hub_count,
                                                                  int32_t* __restrict__ hub_iter,
                                                                  double* __restrict__ out) {
  using BlockReduce = cub::BlockReduce<float, BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ int n_hubs;
  __shared__ int hub_idx;

  if (threadIdx.x == 0) n_hubs = *hub_count;
  __syncthreads();

  while (true) {
    if (threadIdx.x == 0) hub_idx = atomicAdd(hub_iter, 1);
    __syncthreads();
    int idx = hub_idx;
    if (idx >= n_hubs) break;

    int u = __ldg(&hub_q[idx]);
    int cu = (int)__ldg(&clusters[u]);
    int sz = __ldg(&sizes[cu]);
    int start = __ldg(&offsets[u]);
    int end = __ldg(&offsets[u + 1]);

    float local_cut = 0.0f;
    for (int e = start + threadIdx.x; e < end; e += BLOCK) {
      int v = __ldg(&indices[e]);
      int cv = (int)__ldg(&clusters[v]);
      if (cv != cu) local_cut += __ldg(&ew[e]);
    }

    float block_cut = BlockReduce(temp_storage).Sum(local_cut);
    if (threadIdx.x == 0) atomicAdd(out, (double)block_cut / (double)sz);
    __syncthreads();
  }
}





void launch_pack_u8(const int32_t* in, uint8_t* out, int32_t n, int32_t nc) {
  int block = 256;
  int grid = (n + block - 1) / block;
  if (grid > 4096) grid = 4096;
  pack_u8_kernel<<<grid, block>>>(in, out, n, nc);
}

void launch_pack_i16(const int32_t* in, int16_t* out, int32_t n, int32_t nc) {
  int block = 256;
  int grid = (n + block - 1) / block;
  if (grid > 4096) grid = 4096;
  pack_i16_kernel<<<grid, block>>>(in, out, n, nc);
}

void launch_cluster_sizes_u8(const uint8_t* clusters, int32_t* sizes, int32_t nv, int32_t nc) {
  int block = 256;
  int grid = (nv + block - 1) / block;
  if (grid > 108 * 8) grid = 108 * 8;
  if (nc <= 256) {
    cluster_sizes_shared_kernel<uint8_t, 256><<<grid, block, nc * sizeof(int32_t)>>>(clusters, sizes, nv, nc);
  } else {
    cluster_sizes_global_kernel<uint8_t, 256><<<grid, block>>>(clusters, sizes, nv);
  }
}

void launch_cluster_sizes_i16(const int16_t* clusters, int32_t* sizes, int32_t nv, int32_t nc) {
  int block = 256;
  int grid = (nv + block - 1) / block;
  if (grid > 108 * 8) grid = 108 * 8;
  if (nc <= 512) {
    cluster_sizes_shared_kernel<int16_t, 256><<<grid, block, nc * sizeof(int32_t)>>>(clusters, sizes, nv, nc);
  } else {
    cluster_sizes_global_kernel<int16_t, 256><<<grid, block>>>(clusters, sizes, nv);
  }
}

void launch_cluster_sizes_i32(const int32_t* clusters, int32_t* sizes, int32_t nv, int32_t nc) {
  int block = 256;
  int grid = (nv + block - 1) / block;
  if (grid > 108 * 8) grid = 108 * 8;
  if (nc <= 1024) {
    cluster_sizes_shared_kernel<int32_t, 256><<<grid, block, nc * sizeof(int32_t)>>>(clusters, sizes, nv, nc);
  } else {
    cluster_sizes_global_kernel<int32_t, 256><<<grid, block>>>(clusters, sizes, nv);
  }
}

void launch_ratio_cut_u8(const int32_t* offsets, const int32_t* indices, const float* ew, const uint8_t* clusters,
                         const int32_t* sizes, int32_t nv, int32_t gs, int32_t hub_thresh, int32_t* hub_q,
                         int32_t* hub_count, int32_t max_hubs, double* out) {
  int block = 256;
  int grid = gs > 0 ? gs : -gs;
  if (gs > 0)
    ratio_cut_wpv_kernel<uint8_t, 256><<<grid, block>>>(offsets, indices, ew, clusters, sizes, nv, hub_thresh, hub_q,
                                                        hub_count, max_hubs, out);
  else
    ratio_cut_tpv_kernel<uint8_t, 256><<<grid, block>>>(offsets, indices, ew, clusters, sizes, nv, hub_thresh, hub_q,
                                                        hub_count, max_hubs, out);
}

void launch_ratio_cut_i16(const int32_t* offsets, const int32_t* indices, const float* ew, const int16_t* clusters,
                          const int32_t* sizes, int32_t nv, int32_t gs, int32_t hub_thresh, int32_t* hub_q,
                          int32_t* hub_count, int32_t max_hubs, double* out) {
  int block = 256;
  int grid = gs > 0 ? gs : -gs;
  if (gs > 0)
    ratio_cut_wpv_kernel<int16_t, 256><<<grid, block>>>(offsets, indices, ew, clusters, sizes, nv, hub_thresh, hub_q,
                                                        hub_count, max_hubs, out);
  else
    ratio_cut_tpv_kernel<int16_t, 256><<<grid, block>>>(offsets, indices, ew, clusters, sizes, nv, hub_thresh, hub_q,
                                                        hub_count, max_hubs, out);
}

void launch_ratio_cut_i32(const int32_t* offsets, const int32_t* indices, const float* ew, const int32_t* clusters,
                          const int32_t* sizes, int32_t nv, int32_t gs, int32_t hub_thresh, int32_t* hub_q,
                          int32_t* hub_count, int32_t max_hubs, double* out) {
  int block = 256;
  int grid = gs > 0 ? gs : -gs;
  if (gs > 0)
    ratio_cut_wpv_kernel<int32_t, 256><<<grid, block>>>(offsets, indices, ew, clusters, sizes, nv, hub_thresh, hub_q,
                                                        hub_count, max_hubs, out);
  else
    ratio_cut_tpv_kernel<int32_t, 256><<<grid, block>>>(offsets, indices, ew, clusters, sizes, nv, hub_thresh, hub_q,
                                                        hub_count, max_hubs, out);
}

void launch_ratio_cut_hubs_u8(const int32_t* offsets, const int32_t* indices, const float* ew, const uint8_t* clusters,
                              const int32_t* sizes, const int32_t* hub_q, const int32_t* hub_count, int32_t* hub_iter,
                              int32_t grid, double* out) {
  ratio_cut_hubs_kernel<uint8_t, 256><<<grid, 256>>>(offsets, indices, ew, clusters, sizes, hub_q, hub_count, hub_iter,
                                                     out);
}

void launch_ratio_cut_hubs_i16(const int32_t* offsets, const int32_t* indices, const float* ew, const int16_t* clusters,
                               const int32_t* sizes, const int32_t* hub_q, const int32_t* hub_count, int32_t* hub_iter,
                               int32_t grid, double* out) {
  ratio_cut_hubs_kernel<int16_t, 256><<<grid, 256>>>(offsets, indices, ew, clusters, sizes, hub_q, hub_count, hub_iter,
                                                     out);
}

void launch_ratio_cut_hubs_i32(const int32_t* offsets, const int32_t* indices, const float* ew, const int32_t* clusters,
                               const int32_t* sizes, const int32_t* hub_q, const int32_t* hub_count, int32_t* hub_iter,
                               int32_t grid, double* out) {
  ratio_cut_hubs_kernel<int32_t, 256><<<grid, 256>>>(offsets, indices, ew, clusters, sizes, hub_q, hub_count, hub_iter,
                                                     out);
}

}  

double analyze_clustering_ratio_cut(const graph32_t& graph,
                                    const float* edge_weights,
                                    std::size_t num_clusters,
                                    const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t nv = graph.number_of_vertices;
    const int32_t ne = graph.number_of_edges;
    const int32_t nc = static_cast<int32_t>(num_clusters);

    cache.ensure(nv, nc);

    const int32_t* off_ptr = graph.offsets;
    const int32_t* ind_ptr = graph.indices;
    const float* ew_ptr = edge_weights;

    cudaMemsetAsync(cache.sizes, 0, (size_t)nc * sizeof(int32_t));
    cudaMemsetAsync(cache.hub_count, 0, sizeof(int32_t));
    cudaMemsetAsync(cache.hub_iter, 0, sizeof(int32_t));
    cudaMemsetAsync(cache.result, 0, sizeof(double));

    const float avg_degree = (nv > 0) ? ((float)ne / (float)nv) : 0.0f;
    const bool use_warp = (avg_degree >= 16.0f);

    int32_t grid_size;
    if (use_warp) {
      constexpr int warps_per_block = 8; 
      grid_size = (nv + warps_per_block - 1) / warps_per_block;
    } else {
      grid_size = (nv + 255) / 256;
    }
    grid_size = std::min(grid_size, 108 * 8);
    grid_size = std::max(grid_size, 1);

    int32_t hub_thresh = 1024;

    auto set_l2_persist = [](const void* ptr, size_t bytes) {
      cudaStreamAttrValue attr{};
      attr.accessPolicyWindow.base_ptr = const_cast<void*>(ptr);
      attr.accessPolicyWindow.num_bytes = std::min(bytes, (size_t)(16 * 1024 * 1024));
      attr.accessPolicyWindow.hitRatio = 1.0f;
      attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
      attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
      cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
    };

    const int32_t max_hubs = nv;

    if (nc <= 255) {
      launch_pack_u8(cluster_assignments, cache.packed_u8, nv, nc);
      set_l2_persist(cache.packed_u8, (size_t)nv);

      launch_cluster_sizes_u8(cache.packed_u8, cache.sizes, nv, nc);
      launch_ratio_cut_u8(off_ptr, ind_ptr, ew_ptr, cache.packed_u8, cache.sizes, nv,
                          use_warp ? grid_size : -grid_size, hub_thresh, cache.hub_q,
                          cache.hub_count, max_hubs, cache.result);

      launch_ratio_cut_hubs_u8(off_ptr, ind_ptr, ew_ptr, cache.packed_u8, cache.sizes,
                               cache.hub_q, cache.hub_count,
                               cache.hub_iter, 108 * 4, cache.result);

    } else if (nc <= 32767) {
      launch_pack_i16(cluster_assignments, cache.packed_i16, nv, nc);
      set_l2_persist(cache.packed_i16, (size_t)nv * 2);

      launch_cluster_sizes_i16(cache.packed_i16, cache.sizes, nv, nc);
      launch_ratio_cut_i16(off_ptr, ind_ptr, ew_ptr, cache.packed_i16, cache.sizes, nv,
                           use_warp ? grid_size : -grid_size, hub_thresh, cache.hub_q,
                           cache.hub_count, max_hubs, cache.result);

      launch_ratio_cut_hubs_i16(off_ptr, ind_ptr, ew_ptr, cache.packed_i16, cache.sizes,
                                cache.hub_q, cache.hub_count,
                                cache.hub_iter, 108 * 4, cache.result);

    } else {
      set_l2_persist(cluster_assignments, (size_t)nv * 4);

      launch_cluster_sizes_i32(cluster_assignments, cache.sizes, nv, nc);
      launch_ratio_cut_i32(off_ptr, ind_ptr, ew_ptr, cluster_assignments, cache.sizes, nv,
                           use_warp ? grid_size : -grid_size, hub_thresh, cache.hub_q,
                           cache.hub_count, max_hubs, cache.result);

      launch_ratio_cut_hubs_i32(off_ptr, ind_ptr, ew_ptr, cluster_assignments, cache.sizes,
                                cache.hub_q, cache.hub_count,
                                cache.hub_iter, 108 * 2, cache.result);
    }

    
    cudaStreamAttrValue reset{};
    reset.accessPolicyWindow.num_bytes = 0;
    cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &reset);

    
    double host_result;
    cudaMemcpy(&host_result, cache.result, sizeof(double), cudaMemcpyDeviceToHost);
    return host_result;
}

}  
