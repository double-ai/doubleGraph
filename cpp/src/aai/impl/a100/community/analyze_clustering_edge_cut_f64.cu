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
#include <cub/block/block_reduce.cuh>

namespace aai {

namespace {

struct Cache : Cacheable {
  double* d_result = nullptr;

  Cache() {
    cudaMalloc(&d_result, sizeof(double));
  }

  ~Cache() override {
    if (d_result) cudaFree(d_result);
  }
};

static constexpr int kSMs = 108;          
static constexpr int kMaxBlocks = 8192;  

constexpr int BLOCK = 512;
constexpr int WARPS_PER_BLOCK = BLOCK / 32;

__global__ __launch_bounds__(BLOCK, 4)
void edge_cut_kernel_512(const int32_t* __restrict__ offsets,
                         const int32_t* __restrict__ indices,
                         const double* __restrict__ weights,
                         const int32_t* __restrict__ cluster,
                         int32_t num_vertices,
                         double* __restrict__ out)
{
  using BlockReduce = cub::BlockReduce<double, BLOCK>;
  __shared__ typename BlockReduce::TempStorage temp;

  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int global_warp = ((int)blockIdx.x * WARPS_PER_BLOCK) + warp;
  const int total_warps = (int)gridDim.x * WARPS_PER_BLOCK;

  double sum = 0.0;

  for (int32_t v = (int32_t)global_warp; v < num_vertices; v += (int32_t)total_warps) {
    int32_t row_start, row_end, src_c;
    if (lane == 0) {
      row_start = offsets[v];
      row_end = offsets[v + 1];
      src_c = cluster[v];
    }
    row_start = __shfl_sync(0xffffffff, row_start, 0);
    row_end = __shfl_sync(0xffffffff, row_end, 0);
    src_c = __shfl_sync(0xffffffff, src_c, 0);

    for (int32_t e = row_start + lane; e < row_end; e += 32) {
      int32_t dst = indices[e];
      double w = weights[e];
      int32_t dst_c = cluster[dst];
      sum += (dst_c != src_c) ? w : 0.0;
    }
  }

  double block_sum = BlockReduce(temp).Sum(sum);
  if (tid == 0) atomicAdd(out, block_sum * 0.5);
}

}  

double analyze_clustering_edge_cut(const graph32_t& graph,
                                   const double* edge_weights,
                                   std::size_t num_clusters,
                                   const int32_t* cluster_assignments) {
  static int tag;
  auto& cache = cache_pool().acquire<Cache>(&tag);

  int32_t num_vertices = graph.number_of_vertices;
  int32_t num_edges = graph.number_of_edges;

  cudaMemset(cache.d_result, 0, sizeof(double));
  if (num_vertices <= 0 || num_edges <= 0) {
    double result = 0.0;
    return result;
  }

  int blocks = (int)((num_vertices + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
  int min_blocks = kSMs * 6;
  if (blocks < min_blocks) blocks = min_blocks;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;

  edge_cut_kernel_512<<<blocks, BLOCK>>>(graph.offsets, graph.indices,
                                         edge_weights, cluster_assignments,
                                         num_vertices, cache.d_result);

  double result;
  cudaMemcpy(&result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);
  return result;
}

}  
