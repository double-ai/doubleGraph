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

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* status = nullptr;
    int32_t* delta_count = nullptr;
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* d_counters = nullptr;
    int32_t* h_counters = nullptr;
    int32_t capacity = 0;

    Cache() {
        cudaMalloc(&d_counters, 4 * sizeof(int32_t));
        cudaMallocHost(&h_counters, 4 * sizeof(int32_t));
    }

    ~Cache() override {
        if (status) cudaFree(status);
        if (delta_count) cudaFree(delta_count);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (d_counters) cudaFree(d_counters);
        if (h_counters) cudaFreeHost(h_counters);
    }

    void ensure(int32_t n) {
        if (capacity < n) {
            if (status) cudaFree(status);
            if (delta_count) cudaFree(delta_count);
            if (frontier_a) cudaFree(frontier_a);
            if (frontier_b) cudaFree(frontier_b);
            cudaMalloc(&status, (size_t)n * sizeof(int32_t));
            cudaMalloc(&delta_count, (size_t)n * sizeof(int32_t));
            cudaMalloc(&frontier_a, (size_t)n * sizeof(int32_t));
            cudaMalloc(&frontier_b, (size_t)n * sizeof(int32_t));
            capacity = n;
        }
    }
};


__device__ __forceinline__ bool is_edge_active(const uint32_t* edge_mask, int32_t e) {
    return (edge_mask[e >> 5] >> (e & 31)) & 1;
}


__global__ void compute_degrees_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ core_numbers,
    int32_t start_v, int32_t end_v,
    int32_t delta_factor)
{
    int32_t v = start_v + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= end_v) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t d = 0;
    for (int32_t e = start; e < end; e++) {
        if (is_edge_active(edge_mask, e) && __ldg(&indices[e]) != v) d++;
    }
    core_numbers[v] = d * delta_factor;
}


__global__ void compute_degrees_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ core_numbers,
    int32_t start_v, int32_t end_v,
    int32_t delta_factor)
{
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;
    int32_t v = start_v + warp_id;
    if (v >= end_v) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t d = 0;
    for (int32_t e = start + lane; e < end; e += 32) {
        if (is_edge_active(edge_mask, e) && __ldg(&indices[e]) != v) d++;
    }
    for (int32_t s = 16; s > 0; s >>= 1)
        d += __shfl_down_sync(0xFFFFFFFF, d, s);
    if (lane == 0) core_numbers[v] = d * delta_factor;
}


__global__ void compute_degrees_block_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ core_numbers,
    int32_t start_v, int32_t end_v,
    int32_t delta_factor)
{
    extern __shared__ int32_t smem_block[];
    int32_t v = start_v + blockIdx.x;
    if (v >= end_v) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t d = 0;
    for (int32_t e = start + threadIdx.x; e < end; e += blockDim.x) {
        if (is_edge_active(edge_mask, e) && __ldg(&indices[e]) != v) d++;
    }

    
    for (int32_t s = 16; s > 0; s >>= 1)
        d += __shfl_down_sync(0xFFFFFFFF, d, s);

    int32_t warp_id = threadIdx.x >> 5;
    int32_t lane = threadIdx.x & 31;
    if (lane == 0) smem_block[warp_id] = d;
    __syncthreads();

    if (warp_id == 0) {
        int32_t nwarps = blockDim.x >> 5;
        d = (lane < nwarps) ? smem_block[lane] : 0;
        for (int32_t s = 16; s > 0; s >>= 1)
            d += __shfl_down_sync(0xFFFFFFFF, d, s);
        if (lane == 0) core_numbers[v] = d * delta_factor;
    }
}

__global__ void init_flags_kernel(
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ status,
    int32_t n,
    int32_t k_first)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int32_t cn = core_numbers[v];
    if (cn == 0) {
        status[v] = 0;
    } else if (k_first > 1 && cn < k_first) {
        core_numbers[v] = 0;
        status[v] = 1;
    } else {
        status[v] = 1;
    }
}

__global__ void build_frontier_kernel(
    const int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ status,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_size,
    int32_t n,
    int32_t k)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n || status[v] != 1) return;
    if (core_numbers[v] < k) {
        status[v] = 2;
        int32_t pos = atomicAdd(frontier_size, 1);
        frontier[pos] = v;
    }
}


__global__ void process_frontier_warp_kernel(
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ delta_count,
    const int32_t* __restrict__ status,
    int32_t* __restrict__ candidates,
    int32_t* __restrict__ candidate_count)
{
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane_id = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t v = frontier[warp_id];
    int32_t start = __ldg(&offsets[v]);
    int32_t end = __ldg(&offsets[v + 1]);

    for (int32_t e = start + lane_id; e < end; e += 32) {
        if (is_edge_active(edge_mask, e)) {
            int32_t u = __ldg(&indices[e]);
            if (u != v && status[u] < 2) {
                int32_t old = atomicAdd(&delta_count[u], 1);
                if (old == 0) {
                    int32_t pos = atomicAdd(candidate_count, 1);
                    candidates[pos] = u;
                }
            }
        }
    }
}


__global__ void apply_and_build_auto_kernel(
    const int32_t* __restrict__ candidates,
    const int32_t* __restrict__ candidate_count_ptr,
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ delta_count,
    int32_t* __restrict__ status,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t k,
    int32_t delta,
    int32_t k_first)
{
    int32_t candidate_count = *candidate_count_ptr;
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= candidate_count) return;

    int32_t u = candidates[idx];
    int32_t dec = delta_count[u] * delta;
    delta_count[u] = 0;

    int32_t cn = core_numbers[u];
    int32_t new_cn = cn - dec;
    int32_t floor_val = k - delta;
    if (new_cn < floor_val) new_cn = floor_val;
    if (new_cn < k_first) new_cn = 0;
    core_numbers[u] = new_cn;

    if (new_cn < k) {
        status[u] = 2;
        int32_t pos = atomicAdd(next_frontier_size, 1);
        next_frontier[pos] = u;
    }
}

__global__ void find_min_kernel(
    const int32_t* __restrict__ core_numbers,
    const int32_t* __restrict__ status,
    int32_t* __restrict__ result,
    int32_t n)
{
    extern __shared__ int32_t smem[];
    int32_t tid = threadIdx.x;
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;

    int32_t local_min = INT32_MAX;
    for (int32_t i = idx; i < n; i += stride) {
        if (status[i] == 1) {
            int32_t val = core_numbers[i];
            if (val < local_min) local_min = val;
        }
    }
    smem[tid] = local_min;
    __syncthreads();
    for (int32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] < smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicMin(result, smem[0]);
}

void launch_core_number(
    const int32_t* offsets,
    const int32_t* indices,
    const uint32_t* edge_mask,
    int32_t* core_numbers,
    int32_t num_vertices,
    int32_t num_edges,
    int32_t degree_type,
    int64_t k_first_64,
    int64_t k_last_64,
    int32_t* status,
    int32_t* delta_count,
    int32_t* frontier_a,
    int32_t* frontier_b,
    int32_t* d_counters,
    int32_t* h_counters,
    const int32_t* seg_offsets_host,
    cudaStream_t stream)
{
    if (num_vertices == 0) return;

    int32_t delta = (degree_type == 2) ? 2 : 1;
    size_t k_first = static_cast<size_t>(k_first_64);
    size_t k_last = static_cast<size_t>(k_last_64);

    const int32_t BLOCK = 256;
    int32_t grid_v = (num_vertices + BLOCK - 1) / BLOCK;

    
    int32_t seg0 = seg_offsets_host[0];
    int32_t seg1 = seg_offsets_host[1];
    int32_t seg2 = seg_offsets_host[2];
    int32_t seg3 = seg_offsets_host[3];
    int32_t seg4 = seg_offsets_host[4];

    
    if (seg1 > seg0) {
        int32_t n_high = seg1 - seg0;
        compute_degrees_block_kernel<<<n_high, BLOCK, (BLOCK/32)*sizeof(int32_t), stream>>>(
            offsets, indices, edge_mask, core_numbers, seg0, seg1, delta);
    }

    
    if (seg2 > seg1) {
        int32_t n_mid = seg2 - seg1;
        int32_t warps = n_mid;
        int64_t threads = (int64_t)warps * 32;
        int32_t grid = (int32_t)((threads + BLOCK - 1) / BLOCK);
        compute_degrees_warp_kernel<<<grid, BLOCK, 0, stream>>>(
            offsets, indices, edge_mask, core_numbers, seg1, seg2, delta);
    }

    
    if (seg3 > seg2) {
        int32_t n_low = seg3 - seg2;
        int32_t grid = (n_low + BLOCK - 1) / BLOCK;
        compute_degrees_thread_kernel<<<grid, BLOCK, 0, stream>>>(
            offsets, indices, edge_mask, core_numbers, seg2, seg3, delta);
    }

    
    if (seg4 > seg3) {
        cudaMemsetAsync(core_numbers + seg3, 0, (size_t)(seg4 - seg3) * sizeof(int32_t), stream);
    }

    
    init_flags_kernel<<<grid_v, BLOCK, 0, stream>>>(
        core_numbers, status, num_vertices, (int32_t)k_first);

    cudaMemsetAsync(delta_count, 0, (size_t)num_vertices * sizeof(int32_t), stream);

    size_t k = (k_first >= 2) ? k_first : 2;
    if (delta == 2 && (k % 2) == 1) k++;

    
    int32_t max_apply_grid = grid_v;  

    while (k <= k_last) {
        
        cudaMemsetAsync(d_counters, 0, sizeof(int32_t), stream);
        build_frontier_kernel<<<grid_v, BLOCK, 0, stream>>>(
            core_numbers, status, frontier_a, d_counters, num_vertices, (int32_t)k);
        cudaMemcpyAsync(h_counters, d_counters, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        int32_t frontier_size = h_counters[0];

        if (frontier_size == 0) {
            h_counters[2] = INT32_MAX;
            cudaMemcpyAsync(d_counters + 2, h_counters + 2, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
            int32_t min_grid = grid_v < 256 ? grid_v : 256;
            find_min_kernel<<<min_grid, BLOCK, BLOCK * sizeof(int32_t), stream>>>(
                core_numbers, status, d_counters + 2, num_vertices);
            cudaMemcpyAsync(h_counters + 2, d_counters + 2, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            int32_t min_cn = h_counters[2];
            if (min_cn == INT32_MAX) break;

            size_t new_k = (size_t)min_cn;
            if (delta == 2 && (new_k % 2) == 1) new_k++;
            k = (new_k > k + (size_t)delta) ? new_k : k + (size_t)delta;
            continue;
        }

        
        while (frontier_size > 0) {
            
            cudaMemsetAsync(d_counters, 0, 2 * sizeof(int32_t), stream);

            
            int64_t threads_needed = (int64_t)frontier_size * 32;
            int32_t grid_f = (int32_t)((threads_needed + BLOCK - 1) / BLOCK);

            process_frontier_warp_kernel<<<grid_f, BLOCK, 0, stream>>>(
                frontier_a, frontier_size,
                offsets, indices, edge_mask,
                delta_count, status,
                frontier_b, d_counters + 1);

            
            apply_and_build_auto_kernel<<<max_apply_grid, BLOCK, 0, stream>>>(
                frontier_b, d_counters + 1,
                core_numbers, delta_count, status,
                frontier_a, d_counters,
                (int32_t)k, delta, (int32_t)k_first);

            
            cudaMemcpyAsync(h_counters, d_counters, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            frontier_size = h_counters[0];
        }

        k += (size_t)delta;
    }
}

}  

void core_number_seg_mask(const graph32_t& graph,
                          int32_t* core_numbers,
                          int degree_type,
                          std::size_t k_first,
                          std::size_t k_last) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    if (num_vertices == 0) return;

    cache.ensure(num_vertices);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    const uint32_t* edge_mask = graph.edge_mask;
    const auto& seg = graph.segment_offsets.value();

    launch_core_number(
        offsets, indices, edge_mask,
        core_numbers, num_vertices, num_edges,
        degree_type, (int64_t)k_first, (int64_t)k_last,
        cache.status, cache.delta_count,
        cache.frontier_a, cache.frontier_b,
        cache.d_counters, cache.h_counters,
        seg.data(),
        0);
}

}  
