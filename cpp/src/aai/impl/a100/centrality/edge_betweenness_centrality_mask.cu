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
#include <vector>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* distances = nullptr;
    int32_t* sigma = nullptr;
    float* delta = nullptr;
    int32_t* bfs_queue = nullptr;
    int32_t* counter_d = nullptr;
    int32_t vertex_capacity = 0;
    bool counter_allocated = false;

    void ensure(int32_t num_vertices) {
        if (vertex_capacity < num_vertices) {
            if (distances) cudaFree(distances);
            if (sigma) cudaFree(sigma);
            if (delta) cudaFree(delta);
            if (bfs_queue) cudaFree(bfs_queue);
            cudaMalloc(&distances, (size_t)num_vertices * sizeof(int32_t));
            cudaMalloc(&sigma, (size_t)num_vertices * sizeof(int32_t));
            cudaMalloc(&delta, (size_t)num_vertices * sizeof(float));
            cudaMalloc(&bfs_queue, (size_t)num_vertices * sizeof(int32_t));
            vertex_capacity = num_vertices;
        }
        if (!counter_allocated) {
            cudaMalloc(&counter_d, sizeof(int32_t));
            counter_allocated = true;
        }
    }

    ~Cache() override {
        if (distances) cudaFree(distances);
        if (sigma) cudaFree(sigma);
        if (delta) cudaFree(delta);
        if (bfs_queue) cudaFree(bfs_queue);
        if (counter_d) cudaFree(counter_d);
    }
};

__device__ __forceinline__ bool is_edge_active(const uint32_t* edge_mask, int32_t edge_idx) {
    return (edge_mask[edge_idx >> 5] >> (edge_idx & 31)) & 1;
}

__global__ void init_arrays_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ sigma,
    float* __restrict__ delta,
    int32_t num_vertices
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_vertices) {
        distances[tid] = 0x7FFFFFFF;
        sigma[tid] = 0;
        delta[tid] = 0.0f;
    }
}

__global__ void sparse_reset_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ sigma,
    float* __restrict__ delta,
    const int32_t* __restrict__ bfs_queue,
    int32_t queue_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < queue_size) {
        int32_t v = bfs_queue[tid];
        distances[v] = 0x7FFFFFFF;
        sigma[v] = 0;
        delta[v] = 0.0f;
    }
}

__global__ void set_source_kernel(
    int32_t* __restrict__ distances,
    int32_t* __restrict__ sigma,
    int32_t* __restrict__ bfs_queue,
    int32_t source
) {
    distances[source] = 0;
    sigma[source] = 1;
    bfs_queue[0] = source;
}

__global__ void bfs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ sigma,
    int32_t* __restrict__ bfs_queue,
    int32_t queue_start,
    int32_t queue_end,
    int32_t write_pos,
    int32_t* __restrict__ new_count,
    int32_t current_level
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int frontier_size = queue_end - queue_start;

    if (warp_id >= frontier_size) return;

    int32_t u = bfs_queue[queue_start + warp_id];
    int32_t sigma_u = sigma[u];
    int32_t start = offsets[u];
    int32_t end = offsets[u + 1];
    int32_t next_level = current_level + 1;
    int32_t degree = end - start;

    for (int32_t i = lane; i < degree; i += 32) {
        int32_t e = start + i;
        if (!is_edge_active(edge_mask, e)) continue;
        int32_t v = indices[e];

        int32_t old_dist = atomicCAS(&distances[v], 0x7FFFFFFF, next_level);

        if (old_dist != 0x7FFFFFFF && old_dist != next_level) continue;

        atomicAdd(&sigma[v], sigma_u);

        if (old_dist == 0x7FFFFFFF) {
            int32_t pos = atomicAdd(new_count, 1);
            bfs_queue[write_pos + pos] = v;
        }
    }
}

__global__ void bfs_kernel_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    int32_t* __restrict__ sigma,
    int32_t* __restrict__ bfs_queue,
    int32_t queue_start,
    int32_t queue_end,
    int32_t write_pos,
    int32_t* __restrict__ new_count,
    int32_t current_level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int frontier_size = queue_end - queue_start;
    if (tid >= frontier_size) return;

    int32_t u = bfs_queue[queue_start + tid];
    int32_t sigma_u = sigma[u];
    int32_t start = offsets[u];
    int32_t end = offsets[u + 1];
    int32_t next_level = current_level + 1;

    for (int32_t e = start; e < end; e++) {
        if (!is_edge_active(edge_mask, e)) continue;
        int32_t v = indices[e];

        int32_t old_dist = atomicCAS(&distances[v], 0x7FFFFFFF, next_level);
        if (old_dist != 0x7FFFFFFF && old_dist != next_level) continue;
        atomicAdd(&sigma[v], sigma_u);
        if (old_dist == 0x7FFFFFFF) {
            int32_t pos = atomicAdd(new_count, 1);
            bfs_queue[write_pos + pos] = v;
        }
    }
}

__global__ void backprop_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ distances,
    const int32_t* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ edge_bc,
    const int32_t* __restrict__ bfs_queue,
    int32_t queue_start,
    int32_t queue_end,
    int32_t parent_level
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int frontier_size = queue_end - queue_start;

    if (warp_id >= frontier_size) return;

    int32_t u = bfs_queue[queue_start + warp_id];
    float sigma_u = __int2float_rn(sigma[u]);
    int32_t start = offsets[u];
    int32_t end = offsets[u + 1];
    int32_t child_level = parent_level + 1;
    int32_t degree = end - start;
    float local_delta = 0.0f;

    for (int32_t i = lane; i < degree; i += 32) {
        int32_t e = start + i;
        if (!is_edge_active(edge_mask, e)) continue;
        int32_t v = indices[e];

        if (distances[v] == child_level) {
            float sigma_v = __int2float_rn(sigma[v]);
            float c = (sigma_u / sigma_v) * (1.0f + delta[v]);
            edge_bc[e] += c;
            local_delta += c;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_delta += __shfl_down_sync(0xffffffff, local_delta, offset);
    }

    if (lane == 0) {
        delta[u] = local_delta;
    }
}

__global__ void backprop_kernel_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ distances,
    const int32_t* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ edge_bc,
    const int32_t* __restrict__ bfs_queue,
    int32_t queue_start,
    int32_t queue_end,
    int32_t parent_level
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int frontier_size = queue_end - queue_start;
    if (tid >= frontier_size) return;

    int32_t u = bfs_queue[queue_start + tid];
    float sigma_u = __int2float_rn(sigma[u]);
    int32_t start = offsets[u];
    int32_t end = offsets[u + 1];
    int32_t child_level = parent_level + 1;
    float local_delta = 0.0f;

    for (int32_t e = start; e < end; e++) {
        if (!is_edge_active(edge_mask, e)) continue;
        int32_t v = indices[e];
        if (distances[v] == child_level) {
            float sigma_v = __int2float_rn(sigma[v]);
            float c = (sigma_u / sigma_v) * (1.0f + delta[v]);
            edge_bc[e] += c;
            local_delta += c;
        }
    }
    delta[u] = local_delta;
}

__global__ void scale_kernel(float* __restrict__ edge_bc, int32_t num_edges, float scale) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_edges) {
        edge_bc[tid] /= scale;
    }
}

static constexpr int32_t WARP_THRESHOLD = 256;

void launch_edge_betweenness(
    const int32_t* offsets,
    const int32_t* indices,
    const uint32_t* edge_mask,
    float* edge_bc,
    int32_t num_vertices,
    int32_t num_edges,
    bool normalized,
    bool is_symmetric,
    const int32_t* sample_vertices_d,
    int64_t num_samples,
    int32_t* distances,
    int32_t* sigma,
    float* delta,
    int32_t* bfs_queue,
    int32_t* counter_d
) {
    const int BLOCK = 256;
    cudaStream_t stream = 0;

    cudaMemsetAsync(edge_bc, 0, num_edges * sizeof(float), stream);

    std::vector<int32_t> samples_h(num_samples);
    if (num_samples > 0) {
        cudaMemcpy(samples_h.data(), sample_vertices_d, num_samples * sizeof(int32_t), cudaMemcpyDeviceToHost);
    }

    bool first_source = true;
    int32_t prev_queue_end = 0;

    for (int64_t s = 0; s < num_samples; s++) {
        int32_t source = samples_h[s];

        if (first_source) {
            int grid_init = (num_vertices + BLOCK - 1) / BLOCK;
            init_arrays_kernel<<<grid_init, BLOCK, 0, stream>>>(distances, sigma, delta, num_vertices);
            first_source = false;
        } else {
            if (prev_queue_end > 0) {
                int grid_reset = (prev_queue_end + BLOCK - 1) / BLOCK;
                sparse_reset_kernel<<<grid_reset, BLOCK, 0, stream>>>(
                    distances, sigma, delta, bfs_queue, prev_queue_end);
            }
        }

        set_source_kernel<<<1, 1, 0, stream>>>(distances, sigma, bfs_queue, source);

        std::vector<int32_t> level_starts;
        level_starts.reserve(1024);
        level_starts.push_back(0);
        level_starts.push_back(1);
        int32_t queue_end = 1;
        int32_t current_level = 0;

        while (true) {
            int32_t frontier_start = level_starts[current_level];
            int32_t frontier_end = level_starts[current_level + 1];
            int32_t frontier_size = frontier_end - frontier_start;

            if (frontier_size == 0) break;

            cudaMemsetAsync(counter_d, 0, sizeof(int32_t), stream);

            if (frontier_size > WARP_THRESHOLD) {
                int num_warps = frontier_size;
                int64_t threads_needed = (int64_t)num_warps * 32;
                int grid = (int)((threads_needed + BLOCK - 1) / BLOCK);
                bfs_kernel<<<grid, BLOCK, 0, stream>>>(
                    offsets, indices, edge_mask,
                    distances, sigma,
                    bfs_queue, frontier_start, frontier_end,
                    queue_end, counter_d, current_level);
            } else {
                int grid = (frontier_size + BLOCK - 1) / BLOCK;
                bfs_kernel_thread<<<grid, BLOCK, 0, stream>>>(
                    offsets, indices, edge_mask,
                    distances, sigma,
                    bfs_queue, frontier_start, frontier_end,
                    queue_end, counter_d, current_level);
            }

            int32_t new_count;
            cudaMemcpyAsync(&new_count, counter_d, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            queue_end += new_count;
            level_starts.push_back(queue_end);
            current_level++;
        }

        int32_t max_level = current_level;
        prev_queue_end = queue_end;

        for (int32_t level = max_level - 1; level >= 0; level--) {
            int32_t frontier_start = level_starts[level];
            int32_t frontier_end = level_starts[level + 1];
            int32_t frontier_size = frontier_end - frontier_start;

            if (frontier_size == 0) continue;

            if (frontier_size > WARP_THRESHOLD) {
                int num_warps = frontier_size;
                int64_t threads_needed = (int64_t)num_warps * 32;
                int grid = (int)((threads_needed + BLOCK - 1) / BLOCK);
                backprop_kernel<<<grid, BLOCK, 0, stream>>>(
                    offsets, indices, edge_mask,
                    distances, sigma, delta, edge_bc,
                    bfs_queue, frontier_start, frontier_end, level);
            } else {
                int grid = (frontier_size + BLOCK - 1) / BLOCK;
                backprop_kernel_thread<<<grid, BLOCK, 0, stream>>>(
                    offsets, indices, edge_mask,
                    distances, sigma, delta, edge_bc,
                    bfs_queue, frontier_start, frontier_end, level);
            }
        }
    }

    float n = static_cast<float>(num_vertices);
    float k = static_cast<float>(num_samples);
    bool need_scale = false;
    float scale = 1.0f;

    if (normalized) {
        scale = n * (n - 1.0f);
        if (k < n) scale *= k / n;
        need_scale = true;
    } else if (is_symmetric) {
        scale = 2.0f;
        if (k < n) scale *= k / n;
        need_scale = true;
    }

    if (need_scale && num_vertices > 1) {
        int grid = (num_edges + BLOCK - 1) / BLOCK;
        scale_kernel<<<grid, BLOCK, 0, stream>>>(edge_bc, num_edges, scale);
    }

    cudaStreamSynchronize(stream);
}

}  

void edge_betweenness_centrality_mask(const graph32_t& graph,
                                       float* edge_centralities,
                                       bool normalized,
                                       const int32_t* sample_vertices,
                                       std::size_t num_samples) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    bool is_symmetric = graph.is_symmetric;
    const uint32_t* edge_mask = graph.edge_mask;

    cache.ensure(num_vertices);

    const int32_t* effective_samples = sample_vertices;
    int64_t effective_num_samples = static_cast<int64_t>(num_samples);
    int32_t* all_vertices_d = nullptr;

    if (sample_vertices == nullptr) {
        effective_num_samples = num_vertices;
        std::vector<int32_t> all_v(num_vertices);
        for (int32_t i = 0; i < num_vertices; i++) all_v[i] = i;
        cudaMalloc(&all_vertices_d, (size_t)num_vertices * sizeof(int32_t));
        cudaMemcpy(all_vertices_d, all_v.data(),
                   (size_t)num_vertices * sizeof(int32_t), cudaMemcpyHostToDevice);
        effective_samples = all_vertices_d;
    }

    launch_edge_betweenness(
        offsets, indices, edge_mask,
        edge_centralities,
        num_vertices, num_edges,
        normalized, is_symmetric,
        effective_samples, effective_num_samples,
        cache.distances, cache.sigma, cache.delta,
        cache.bfs_queue, cache.counter_d
    );

    if (all_vertices_d) cudaFree(all_vertices_d);
}

}  
