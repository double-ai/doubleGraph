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
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <vector>

namespace aai {

namespace {

namespace cg = cooperative_groups;



__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_counts,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int start = offsets[v];
    int end = offsets[v + 1];
    int count = 0;
    for (int e = start; e < end; e++)
        count += (edge_mask[e >> 5] >> (e & 31)) & 1u;
    active_counts[v] = count;
}

__global__ void compute_last_offset_kernel(
    int32_t* __restrict__ new_offsets,
    const int32_t* __restrict__ active_counts,
    int32_t num_vertices)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (num_vertices > 0)
            new_offsets[num_vertices] = new_offsets[num_vertices - 1] + active_counts[num_vertices - 1];
        else
            new_offsets[0] = 0;
    }
}

__global__ void scatter_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int start = offsets[v];
    int end = offsets[v + 1];
    int write_pos = new_offsets[v];
    for (int e = start; e < end; e++) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1u)
            new_indices[write_pos++] = indices[e];
    }
}



__global__ void __launch_bounds__(256, 6)
brandes_persistent_kernel(
    const int32_t* __restrict__ comp_offsets,
    const int32_t* __restrict__ comp_indices,
    int32_t* __restrict__ distances,
    float* __restrict__ sigma,
    float* __restrict__ delta,
    int32_t* __restrict__ bfs_stack,
    int32_t* __restrict__ d_counter,
    int32_t* __restrict__ level_starts,
    float* __restrict__ centrality,
    int32_t source,
    int32_t num_vertices,
    bool include_endpoints)
{
    auto grid = cg::this_grid();
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = gridDim.x * blockDim.x;
    const int warp_id = gtid >> 5;
    const int lane_id = gtid & 31;
    const int num_warps = grid_size >> 5;

    
    for (int i = gtid; i < num_vertices; i += grid_size) {
        distances[i] = (i == source) ? 0 : -1;
        sigma[i] = (i == source) ? 1.0f : 0.0f;
        delta[i] = 0.0f;
    }
    if (gtid == 0) {
        bfs_stack[0] = source;
        d_counter[0] = 1;
        level_starts[0] = 0;
    }
    grid.sync();

    
    int current_start = 0;
    int current_end = 1;
    int level = 0;
    int prev_total = 1;

    while (current_end > current_start) {
        const int frontier_size = current_end - current_start;
        const int new_dist = level + 1;

        
        for (int fi = warp_id; fi < frontier_size; fi += num_warps) {
            const int src = bfs_stack[current_start + fi];
            const int start = __ldg(&comp_offsets[src]);
            const int end = __ldg(&comp_offsets[src + 1]);
            const float src_sigma = sigma[src];

            for (int e = start + lane_id; e < end; e += 32) {
                const int dst = __ldg(&comp_indices[e]);
                const int old_dist = atomicCAS(&distances[dst], -1, new_dist);
                if (old_dist != -1 && old_dist != new_dist) continue;
                atomicAdd(&sigma[dst], src_sigma);
                if (old_dist == -1) {
                    const int pos = atomicAdd(&d_counter[0], 1);
                    bfs_stack[pos] = dst;
                }
            }
        }

        grid.sync();  

        
        const int total = d_counter[0];
        const int new_count = total - prev_total;
        prev_total = total;

        current_start = current_end;
        current_end = current_start + new_count;
        level++;

        
        if (gtid == 0) level_starts[level] = current_start;
    }

    const int max_depth = level;
    const int total_visited = current_start;

    
    grid.sync();

    
    for (int L = max_depth - 2; L >= 0; L--) {
        const int lstart = level_starts[L];
        const int lsize = level_starts[L + 1] - level_starts[L];
        const int next_level = L + 1;

        for (int fi = warp_id; fi < lsize; fi += num_warps) {
            const int v = bfs_stack[lstart + fi];
            const int start = __ldg(&comp_offsets[v]);
            const int end = __ldg(&comp_offsets[v + 1]);
            const float sigma_v = sigma[v];
            float sum = 0.0f;

            for (int e = start + lane_id; e < end; e += 32) {
                const int w = __ldg(&comp_indices[e]);
                if (__ldg(&distances[w]) == next_level) {
                    const float sw = sigma[w];
                    sum += (sigma_v / sw) * (1.0f + delta[w]);
                }
            }

            
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

            if (lane_id == 0) delta[v] = sum;
        }

        grid.sync();
    }

    
    if (include_endpoints && gtid == 0)
        centrality[source] += (float)(total_visited - 1);

    for (int fi = gtid; fi < total_visited; fi += grid_size) {
        const int v = bfs_stack[fi];
        if (v != source) {
            float d = delta[v];
            if (include_endpoints) d += 1.0f;
            centrality[v] += d;
        }
    }
}



__global__ void mark_sources_zero_kernel(uint8_t* is_source, int32_t n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) is_source[tid] = 0;
}

__global__ void mark_sources_set_kernel(uint8_t* is_source, const int32_t* sv, int32_t ns) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < ns) is_source[sv[tid]] = 1;
}

__global__ void normalize_kernel(
    float* __restrict__ centrality, const uint8_t* __restrict__ is_source,
    int32_t nv, int32_t ns, bool normalized, bool include_endpoints, bool is_symmetric)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv) return;
    float bc = centrality[v];

    int n = nv, k = ns;
    int adj = include_endpoints ? n : (n - 1);
    bool all_srcs = (k == adj) || include_endpoints;

    if (all_srcs) {
        float scale;
        if (normalized) scale = (float)k * (float)(adj - 1);
        else if (is_symmetric) scale = (float)k * 2.0f / (float)adj;
        else scale = (float)k / (float)adj;
        if (scale != 0.0f) bc /= scale;
    } else if (normalized) {
        bool is_src = (is_source[v] != 0);
        float scale = is_src ? (float)(k-1) * (float)(adj-1) : (float)k * (float)(adj-1);
        bc /= scale;
    } else {
        bool is_src = (is_source[v] != 0);
        float s = is_src ? (float)(k-1) / (float)adj : (float)k / (float)adj;
        if (is_symmetric) s *= 2.0f;
        bc /= s;
    }
    centrality[v] = bc;
}



struct Cache : Cacheable {
    int32_t* active_counts = nullptr;  int64_t active_counts_cap = 0;
    int32_t* new_offsets = nullptr;    int64_t new_offsets_cap = 0;
    int32_t* new_indices = nullptr;    int64_t new_indices_cap = 0;
    int32_t* distances = nullptr;      int64_t distances_cap = 0;
    float* sigma = nullptr;            int64_t sigma_cap = 0;
    float* delta = nullptr;            int64_t delta_cap = 0;
    int32_t* bfs_stack = nullptr;      int64_t bfs_stack_cap = 0;
    int32_t* counter = nullptr;
    int32_t* level_starts = nullptr;   int64_t level_starts_cap = 0;
    uint8_t* is_source = nullptr;      int64_t is_source_cap = 0;
    void* cub_temp = nullptr;          size_t cub_temp_cap = 0;
    int brandes_grid_size = 0;

    void ensure(int32_t V, int32_t E) {
        if (brandes_grid_size == 0) {
            int bpsm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsm, brandes_persistent_kernel, 256, 0);
            int nsm = 0;
            cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
            brandes_grid_size = bpsm * nsm;
        }
        if (!counter) {
            cudaMalloc(&counter, sizeof(int32_t));
        }
        if (active_counts_cap < V) {
            if (active_counts) cudaFree(active_counts);
            cudaMalloc(&active_counts, (size_t)V * sizeof(int32_t));
            active_counts_cap = V;
        }
        if (new_offsets_cap < (int64_t)(V + 1)) {
            if (new_offsets) cudaFree(new_offsets);
            cudaMalloc(&new_offsets, ((size_t)V + 1) * sizeof(int32_t));
            new_offsets_cap = V + 1;
        }
        {
            int64_t ni = E > 0 ? (int64_t)E : 1;
            if (new_indices_cap < ni) {
                if (new_indices) cudaFree(new_indices);
                cudaMalloc(&new_indices, (size_t)ni * sizeof(int32_t));
                new_indices_cap = ni;
            }
        }
        if (distances_cap < V) {
            if (distances) cudaFree(distances);
            cudaMalloc(&distances, (size_t)V * sizeof(int32_t));
            distances_cap = V;
        }
        if (sigma_cap < V) {
            if (sigma) cudaFree(sigma);
            cudaMalloc(&sigma, (size_t)V * sizeof(float));
            sigma_cap = V;
        }
        if (delta_cap < V) {
            if (delta) cudaFree(delta);
            cudaMalloc(&delta, (size_t)V * sizeof(float));
            delta_cap = V;
        }
        if (bfs_stack_cap < V) {
            if (bfs_stack) cudaFree(bfs_stack);
            cudaMalloc(&bfs_stack, (size_t)V * sizeof(int32_t));
            bfs_stack_cap = V;
        }
        if (level_starts_cap < (int64_t)(V + 2)) {
            if (level_starts) cudaFree(level_starts);
            cudaMalloc(&level_starts, ((size_t)V + 2) * sizeof(int32_t));
            level_starts_cap = V + 2;
        }
        if (is_source_cap < V) {
            if (is_source) cudaFree(is_source);
            cudaMalloc(&is_source, (size_t)V * sizeof(uint8_t));
            is_source_cap = V;
        }
        {
            size_t needed = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, needed, (int32_t*)nullptr, (int32_t*)nullptr, V);
            if (cub_temp_cap < needed) {
                if (cub_temp) cudaFree(cub_temp);
                cudaMalloc(&cub_temp, needed + 16);
                cub_temp_cap = needed + 16;
            }
        }
    }

    ~Cache() override {
        if (active_counts) cudaFree(active_counts);
        if (new_offsets) cudaFree(new_offsets);
        if (new_indices) cudaFree(new_indices);
        if (distances) cudaFree(distances);
        if (sigma) cudaFree(sigma);
        if (delta) cudaFree(delta);
        if (bfs_stack) cudaFree(bfs_stack);
        if (counter) cudaFree(counter);
        if (level_starts) cudaFree(level_starts);
        if (is_source) cudaFree(is_source);
        if (cub_temp) cudaFree(cub_temp);
    }
};

}  

void betweenness_centrality_seg_mask(const graph32_t& graph,
                                      float* centralities,
                                      bool normalized,
                                      bool include_endpoints,
                                      const int32_t* sample_vertices,
                                      std::size_t num_samples) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t V = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;
    bool is_symmetric = graph.is_symmetric;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cudaStream_t stream = 0;
    int32_t K = (int32_t)num_samples;

    cudaMemsetAsync(centralities, 0, (size_t)V * sizeof(float), stream);

    if (V == 0 || K == 0) {
        cudaStreamSynchronize(stream);
        return;
    }

    
    std::vector<int32_t> h_samples(K);
    cudaMemcpy(h_samples.data(), sample_vertices, (size_t)K * sizeof(int32_t), cudaMemcpyDeviceToHost);

    cache.ensure(V, E);

    
    count_active_edges_kernel<<<(V + 255) / 256, 256, 0, stream>>>(
        d_offsets, d_edge_mask, cache.active_counts, V);

    size_t cub_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_bytes, (int32_t*)nullptr, (int32_t*)nullptr, V);
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, cub_bytes,
        cache.active_counts, cache.new_offsets, V, stream);

    compute_last_offset_kernel<<<1, 1, 0, stream>>>(cache.new_offsets, cache.active_counts, V);

    scatter_active_edges_kernel<<<(V + 255) / 256, 256, 0, stream>>>(
        d_offsets, d_indices, d_edge_mask, cache.new_offsets, cache.new_indices, V);

    
    int num_blocks = cache.brandes_grid_size;

    const int32_t* comp_offsets_ptr = cache.new_offsets;
    const int32_t* comp_indices_ptr = cache.new_indices;
    int32_t* dist_ptr = cache.distances;
    float* sigma_ptr = cache.sigma;
    float* delta_ptr = cache.delta;
    int32_t* stk_ptr = cache.bfs_stack;
    int32_t* cnt_ptr = cache.counter;
    int32_t* ls_ptr = cache.level_starts;
    float* bc_ptr = centralities;

    for (int32_t si = 0; si < K; si++) {
        int32_t src = h_samples[si];
        void* args[] = {
            (void*)&comp_offsets_ptr, (void*)&comp_indices_ptr,
            (void*)&dist_ptr, (void*)&sigma_ptr, (void*)&delta_ptr,
            (void*)&stk_ptr, (void*)&cnt_ptr, (void*)&ls_ptr,
            (void*)&bc_ptr,
            (void*)&src, (void*)&V, (void*)&include_endpoints
        };
        cudaLaunchCooperativeKernel((void*)brandes_persistent_kernel,
            dim3(num_blocks), dim3(256), args, 0, stream);
    }

    
    int t = 256;
    if (V > 0) mark_sources_zero_kernel<<<(V + t - 1) / t, t, 0, stream>>>(cache.is_source, V);
    if (K > 0) mark_sources_set_kernel<<<(K + t - 1) / t, t, 0, stream>>>(cache.is_source, sample_vertices, K);
    if (V > 0) normalize_kernel<<<(V + 255) / 256, 256, 0, stream>>>(
        centralities, cache.is_source, V, K, normalized, include_endpoints, is_symmetric);

    cudaStreamSynchronize(stream);
}

}  
