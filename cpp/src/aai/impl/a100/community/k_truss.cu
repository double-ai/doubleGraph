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
    
    int32_t* src_vertex = nullptr;
    int32_t* reverse_map = nullptr;
    int8_t* alive = nullptr;
    int32_t* support = nullptr;
    int32_t* work_queue = nullptr;
    int64_t edge_capacity = 0;

    
    int32_t* degrees = nullptr;
    int64_t vertex_capacity = 0;

    
    int32_t* queue_size_buf = nullptr;
    int32_t* flag_buf = nullptr;
    int32_t* out_count_buf = nullptr;
    bool scalars_allocated = false;

    void ensure(int32_t num_edges, int32_t num_vertices) {
        if (edge_capacity < num_edges) {
            if (src_vertex) cudaFree(src_vertex);
            if (reverse_map) cudaFree(reverse_map);
            if (alive) cudaFree(alive);
            if (support) cudaFree(support);
            if (work_queue) cudaFree(work_queue);
            cudaMalloc(&src_vertex, (size_t)num_edges * sizeof(int32_t));
            cudaMalloc(&reverse_map, (size_t)num_edges * sizeof(int32_t));
            cudaMalloc(&alive, (size_t)num_edges * sizeof(int8_t));
            cudaMalloc(&support, (size_t)num_edges * sizeof(int32_t));
            cudaMalloc(&work_queue, (size_t)num_edges * sizeof(int32_t));
            edge_capacity = num_edges;
        }
        if (vertex_capacity < num_vertices) {
            if (degrees) cudaFree(degrees);
            cudaMalloc(&degrees, (size_t)num_vertices * sizeof(int32_t));
            vertex_capacity = num_vertices;
        }
        if (!scalars_allocated) {
            cudaMalloc(&queue_size_buf, sizeof(int32_t));
            cudaMalloc(&flag_buf, sizeof(int32_t));
            cudaMalloc(&out_count_buf, sizeof(int32_t));
            scalars_allocated = true;
        }
    }

    ~Cache() override {
        if (src_vertex) cudaFree(src_vertex);
        if (reverse_map) cudaFree(reverse_map);
        if (alive) cudaFree(alive);
        if (support) cudaFree(support);
        if (work_queue) cudaFree(work_queue);
        if (degrees) cudaFree(degrees);
        if (queue_size_buf) cudaFree(queue_size_buf);
        if (flag_buf) cudaFree(flag_buf);
        if (out_count_buf) cudaFree(out_count_buf);
    }
};





__device__ int32_t find_src_vertex(const int32_t* offsets, int32_t num_vertices, int32_t eid) {
    int32_t lo = 0, hi = num_vertices - 1;
    while (lo < hi) {
        int32_t mid = (lo + hi) >> 1;
        if (offsets[mid + 1] <= eid) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__device__ int32_t warp_intersect_count(
    const int32_t* __restrict__ indices,
    const int8_t* __restrict__ alive,
    int32_t a_start, int32_t a_end,
    int32_t b_start, int32_t b_end,
    int32_t lane_id)
{
    int32_t a_len = a_end - a_start;
    int32_t b_len = b_end - b_start;
    if (a_len == 0 || b_len == 0) return 0;

    
    int32_t s_start, s_end, l_start, l_end;
    if (a_len <= b_len) {
        s_start = a_start; s_end = a_end;
        l_start = b_start; l_end = b_end;
    } else {
        s_start = b_start; s_end = b_end;
        l_start = a_start; l_end = a_end;
    }
    int32_t s_len = s_end - s_start;

    int32_t count = 0;
    for (int32_t i = lane_id; i < s_len; i += 32) {
        int32_t s_pos = s_start + i;
        if (!alive[s_pos]) continue;
        int32_t target = indices[s_pos];

        
        int32_t lo = l_start, hi = l_end;
        while (lo < hi) {
            int32_t mid = (lo + hi) >> 1;
            if (indices[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        if (lo < l_end && indices[lo] == target && alive[lo]) {
            count++;
        }
    }

    
    for (int32_t offset = 16; offset > 0; offset >>= 1) {
        count += __shfl_down_sync(0xffffffff, count, offset);
    }
    return count;
}





__global__ void kernel_compute_src_vertices(
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ src_vertex,
    int32_t num_vertices, int32_t num_edges)
{
    int32_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges) return;
    src_vertex[eid] = find_src_vertex(offsets, num_vertices, eid);
}

__global__ void kernel_compute_reverse_map(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ src_vertex,
    int32_t* __restrict__ reverse_map,
    int32_t num_edges)
{
    int32_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges) return;
    int32_t u = src_vertex[eid];
    int32_t v = indices[eid];
    int32_t lo = offsets[v], hi = offsets[v + 1];
    while (lo < hi) {
        int32_t mid = (lo + hi) >> 1;
        if (indices[mid] < u) lo = mid + 1;
        else hi = mid;
    }
    reverse_map[eid] = lo;
}





__global__ void kernel_compute_degrees(
    const int32_t* __restrict__ offsets,
    const int8_t* __restrict__ alive,
    int32_t* __restrict__ degrees,
    int32_t num_vertices)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int32_t count = 0;
    for (int32_t i = offsets[v]; i < offsets[v + 1]; i++) {
        if (alive[i]) count++;
    }
    degrees[v] = count;
}

__global__ void kernel_remove_low_degree_edges(
    const int32_t* __restrict__ src_vertex,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ degrees,
    int8_t* __restrict__ alive,
    int32_t min_degree,
    int32_t* __restrict__ removed_flag,
    int32_t num_edges)
{
    int32_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges) return;
    if (!alive[eid]) return;

    int32_t u = src_vertex[eid];
    int32_t v = indices[eid];
    if (degrees[u] < min_degree || degrees[v] < min_degree) {
        alive[eid] = 0;
        *removed_flag = 1;
    }
}





__global__ void kernel_build_work_queue(
    const int32_t* __restrict__ src_vertex,
    const int32_t* __restrict__ indices,
    const int8_t* __restrict__ alive,
    int32_t* __restrict__ queue,
    int32_t* __restrict__ queue_size,
    int32_t num_edges)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t lane = tid & 31;

    bool qualify = false;
    if (tid < num_edges) {
        qualify = alive[tid] && (src_vertex[tid] < indices[tid]);
    }

    unsigned int mask = __ballot_sync(0xffffffff, qualify);
    int warp_count = __popc(mask);

    int32_t base = 0;
    if (lane == 0 && warp_count > 0) {
        base = atomicAdd(queue_size, warp_count);
    }
    base = __shfl_sync(0xffffffff, base, 0);

    if (qualify) {
        int32_t offset = __popc(mask & ((1u << lane) - 1));
        queue[base + offset] = tid;
    }
}





__global__ void kernel_count_support_queue(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ src_vertex,
    const int32_t* __restrict__ reverse_map,
    const int8_t* __restrict__ alive,
    int32_t* __restrict__ support,
    const int32_t* __restrict__ work_queue,
    int32_t queue_size)
{
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane_id = threadIdx.x & 31;

    if (warp_id >= queue_size) return;

    int32_t eid = work_queue[warp_id];
    int32_t u = src_vertex[eid];
    int32_t v = indices[eid];

    int32_t count = warp_intersect_count(
        indices, alive,
        offsets[u], offsets[u + 1],
        offsets[v], offsets[v + 1],
        lane_id);

    if (lane_id == 0) {
        support[eid] = count;
        support[reverse_map[eid]] = count;
    }
}





__global__ void kernel_remove_low_support(
    const int32_t* __restrict__ support,
    const int32_t* __restrict__ reverse_map,
    int8_t* __restrict__ alive,
    int32_t threshold,
    int32_t* __restrict__ removed_flag,
    int32_t num_edges)
{
    int32_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges) return;
    if (alive[eid] && support[eid] < threshold) {
        alive[eid] = 0;
        alive[reverse_map[eid]] = 0;
        *removed_flag = 1;
    }
}





__global__ void kernel_collect_edges(
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ src_vertex,
    const int8_t* __restrict__ alive,
    int32_t* __restrict__ out_srcs,
    int32_t* __restrict__ out_dsts,
    int32_t* __restrict__ out_count,
    int32_t num_edges)
{
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t lane = tid & 31;

    bool qualify = false;
    if (tid < num_edges) {
        qualify = (alive[tid] != 0);
    }

    unsigned int mask = __ballot_sync(0xffffffff, qualify);
    int warp_count = __popc(mask);

    int32_t base = 0;
    if (lane == 0 && warp_count > 0) {
        base = atomicAdd(out_count, warp_count);
    }
    base = __shfl_sync(0xffffffff, base, 0);

    if (qualify) {
        int32_t offset = __popc(mask & ((1u << lane) - 1));
        int32_t pos = base + offset;
        out_srcs[pos] = src_vertex[tid];
        out_dsts[pos] = indices[tid];
    }
}

}  

k_truss_result_t k_truss(const graph32_t& graph,
                         int32_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    int32_t threshold = k - 2;

    
    if (num_edges == 0) {
        return k_truss_result_t{nullptr, nullptr, 0};
    }

    cache.ensure(num_edges, num_vertices);

    int32_t* d_src_vertex = cache.src_vertex;
    int32_t* d_reverse_map = cache.reverse_map;
    int8_t* d_alive = cache.alive;
    int32_t* d_support = cache.support;
    int32_t* d_degrees = cache.degrees;
    int32_t* d_work_queue = cache.work_queue;
    int32_t* d_queue_size = cache.queue_size_buf;
    int32_t* d_flag = cache.flag_buf;

    constexpr int block = 256;

    
    cudaMemset(d_alive, 1, (size_t)num_edges * sizeof(int8_t));
    cudaMemset(d_support, 0, (size_t)num_edges * sizeof(int32_t));

    
    {
        int grid = (num_edges + block - 1) / block;
        if (grid > 0) kernel_compute_src_vertices<<<grid, block>>>(d_offsets, d_src_vertex, num_vertices, num_edges);
    }
    {
        int grid = (num_edges + block - 1) / block;
        if (grid > 0) kernel_compute_reverse_map<<<grid, block>>>(d_offsets, d_indices, d_src_vertex, d_reverse_map, num_edges);
    }

    
    
    
    int32_t min_degree = k - 1;
    while (true) {
        {
            int grid = (num_vertices + block - 1) / block;
            if (grid > 0) kernel_compute_degrees<<<grid, block>>>(d_offsets, d_alive, d_degrees, num_vertices);
        }

        cudaMemset(d_flag, 0, sizeof(int32_t));
        {
            int grid = (num_edges + block - 1) / block;
            if (grid > 0) kernel_remove_low_degree_edges<<<grid, block>>>(d_src_vertex, d_indices, d_degrees,
                                                                          d_alive, min_degree, d_flag, num_edges);
        }

        int32_t h_flag;
        cudaMemcpy(&h_flag, d_flag, sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (h_flag == 0) break;
    }

    
    
    
    while (true) {
        
        cudaMemset(d_queue_size, 0, sizeof(int32_t));
        {
            int grid = (num_edges + block - 1) / block;
            if (grid > 0) kernel_build_work_queue<<<grid, block>>>(d_src_vertex, d_indices, d_alive,
                                                                    d_work_queue, d_queue_size, num_edges);
        }

        int32_t h_queue_size;
        cudaMemcpy(&h_queue_size, d_queue_size, sizeof(int32_t), cudaMemcpyDeviceToHost);

        if (h_queue_size == 0) break;  

        
        {
            int warps_per_block = 8;
            int blk = warps_per_block * 32;
            int grid = (h_queue_size + warps_per_block - 1) / warps_per_block;
            if (grid > 0) kernel_count_support_queue<<<grid, blk>>>(d_offsets, d_indices, d_src_vertex, d_reverse_map,
                                                                     d_alive, d_support, d_work_queue, h_queue_size);
        }

        
        cudaMemset(d_flag, 0, sizeof(int32_t));
        {
            int grid = (num_edges + block - 1) / block;
            if (grid > 0) kernel_remove_low_support<<<grid, block>>>(d_support, d_reverse_map, d_alive,
                                                                      threshold, d_flag, num_edges);
        }

        int32_t h_flag;
        cudaMemcpy(&h_flag, d_flag, sizeof(int32_t), cudaMemcpyDeviceToHost);
        if (h_flag == 0) break;
    }

    
    
    
    int32_t* d_out_count = cache.out_count_buf;
    cudaMemset(d_out_count, 0, sizeof(int32_t));

    
    int32_t* d_edge_srcs = nullptr;
    int32_t* d_edge_dsts = nullptr;
    cudaMalloc(&d_edge_srcs, (size_t)num_edges * sizeof(int32_t));
    cudaMalloc(&d_edge_dsts, (size_t)num_edges * sizeof(int32_t));

    {
        int grid = (num_edges + block - 1) / block;
        if (grid > 0) kernel_collect_edges<<<grid, block>>>(d_indices, d_src_vertex, d_alive,
                                                             d_edge_srcs, d_edge_dsts, d_out_count, num_edges);
    }

    int32_t h_out_count;
    cudaMemcpy(&h_out_count, d_out_count, sizeof(int32_t), cudaMemcpyDeviceToHost);

    return k_truss_result_t{d_edge_srcs, d_edge_dsts, static_cast<std::size_t>(h_out_count)};
}

}  
