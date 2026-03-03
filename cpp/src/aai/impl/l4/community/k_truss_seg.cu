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

namespace aai {

namespace {

struct Cache : Cacheable {
    
    int32_t* src_a = nullptr;
    int32_t* offsets_b = nullptr;
    int32_t* indices_b = nullptr;
    int32_t* src_b = nullptr;
    int32_t* offsets_c = nullptr;
    int32_t* indices_c = nullptr;
    int32_t* src_c = nullptr;
    int32_t* support = nullptr;
    int32_t* degree_buf = nullptr;

    
    void* d_cub_temp = nullptr;
    size_t cub_temp_bytes = 0;

    
    int64_t edge_capacity = 0;
    int64_t vertex_capacity = 0;
    int64_t src_a_capacity = 0;

    void ensure_edges(int64_t num_edges) {
        if (edge_capacity < num_edges) {
            if (indices_b) cudaFree(indices_b);
            if (src_b) cudaFree(src_b);
            if (indices_c) cudaFree(indices_c);
            if (src_c) cudaFree(src_c);
            if (support) cudaFree(support);

            cudaMalloc(&indices_b, num_edges * sizeof(int32_t));
            cudaMalloc(&src_b, num_edges * sizeof(int32_t));
            cudaMalloc(&indices_c, num_edges * sizeof(int32_t));
            cudaMalloc(&src_c, num_edges * sizeof(int32_t));
            cudaMalloc(&support, num_edges * sizeof(int32_t));

            edge_capacity = num_edges;
        }
    }

    void ensure_vertices(int64_t num_vertices) {
        if (vertex_capacity < num_vertices) {
            if (offsets_b) cudaFree(offsets_b);
            if (offsets_c) cudaFree(offsets_c);
            if (degree_buf) cudaFree(degree_buf);

            cudaMalloc(&offsets_b, (num_vertices + 1) * sizeof(int32_t));
            cudaMalloc(&offsets_c, (num_vertices + 1) * sizeof(int32_t));
            cudaMalloc(&degree_buf, (num_vertices + 1) * sizeof(int32_t));

            vertex_capacity = num_vertices;
        }
    }

    void ensure_src_a(int64_t num_edges) {
        if (src_a_capacity < num_edges) {
            if (src_a) cudaFree(src_a);
            cudaMalloc(&src_a, num_edges * sizeof(int32_t));
            src_a_capacity = num_edges;
        }
    }

    void ensure_cub_temp(size_t needed) {
        if (needed > cub_temp_bytes) {
            if (d_cub_temp) cudaFree(d_cub_temp);
            cub_temp_bytes = needed * 2;
            cudaMalloc(&d_cub_temp, cub_temp_bytes);
        }
    }

    ~Cache() override {
        if (src_a) cudaFree(src_a);
        if (offsets_b) cudaFree(offsets_b);
        if (indices_b) cudaFree(indices_b);
        if (src_b) cudaFree(src_b);
        if (offsets_c) cudaFree(offsets_c);
        if (indices_c) cudaFree(indices_c);
        if (src_c) cudaFree(src_c);
        if (support) cudaFree(support);
        if (degree_buf) cudaFree(degree_buf);
        if (d_cub_temp) cudaFree(d_cub_temp);
    }
};

__device__ __forceinline__ int lower_bound_dev(const int* arr, int lo, int hi, int target) {
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void compute_src_vertices_kernel(
    const int* __restrict__ offsets, int* __restrict__ src_vertex,
    int num_vertices, int num_edges
) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges) return;
    int lo = 0, hi = num_vertices;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (offsets[mid] <= eid) lo = mid;
        else hi = mid - 1;
    }
    src_vertex[eid] = lo;
}

__global__ void count_triangles_upper_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const int* __restrict__ src_vertex,
    int* __restrict__ support,
    int num_edges
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_edges) return;
    int eid = warp_id;

    int u = src_vertex[eid];
    int v = indices[eid];

    if (u >= v) {
        if (lane == 0) support[eid] = -1;
        return;
    }

    int u_start = offsets[u], u_end = offsets[u + 1];
    int v_start = offsets[v], v_end = offsets[v + 1];
    int u_deg = u_end - u_start;
    int v_deg = v_end - v_start;

    int small_start, small_end, large_start, large_end;
    if (u_deg <= v_deg) {
        small_start = u_start; small_end = u_end;
        large_start = v_start; large_end = v_end;
    } else {
        small_start = v_start; small_end = v_end;
        large_start = u_start; large_end = u_end;
    }
    int small_size = small_end - small_start;

    int count = 0;
    for (int idx = lane; idx < small_size; idx += 32) {
        int w = indices[small_start + idx];
        if (w == u || w == v) continue;
        int found_pos = lower_bound_dev(indices, large_start, large_end, w);
        if (found_pos < large_end && indices[found_pos] == w) count++;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        count += __shfl_down_sync(0xFFFFFFFF, count, offset);

    if (lane == 0) support[eid] = count;
}

__global__ void copy_support_reverse_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const int* __restrict__ src_vertex,
    int* __restrict__ support,
    int num_edges
) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges) return;
    if (support[eid] != -1) return;

    int u = src_vertex[eid];
    int v = indices[eid];

    if (u == v) { support[eid] = 0; return; }

    int lo = offsets[v], hi = offsets[v + 1];
    int rev_pos = lower_bound_dev(indices, lo, hi, u);

    if (rev_pos < hi && indices[rev_pos] == u) {
        support[eid] = support[rev_pos];
    } else {
        support[eid] = 0;
    }
}

__global__ void count_surviving_degrees_kernel(
    const int* __restrict__ offsets, const int* __restrict__ support,
    int* __restrict__ new_degrees, int threshold, int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int count = 0;
    for (int i = offsets[v]; i < offsets[v + 1]; i++) {
        if (support[i] >= threshold) count++;
    }
    new_degrees[v] = count;
}

__global__ void scatter_surviving_edges_kernel(
    const int* __restrict__ old_offsets, const int* __restrict__ old_indices,
    const int* __restrict__ support, int threshold,
    const int* __restrict__ new_offsets,
    int* __restrict__ new_indices, int* __restrict__ new_src,
    int num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int write_pos = new_offsets[v];
    for (int i = old_offsets[v]; i < old_offsets[v + 1]; i++) {
        if (support[i] >= threshold) {
            new_indices[write_pos] = old_indices[i];
            new_src[write_pos] = v;
            write_pos++;
        }
    }
}

__global__ void write_output_kernel(
    const int* __restrict__ src_vertex, const int* __restrict__ indices,
    int* __restrict__ out_src, int* __restrict__ out_dst, int num_edges
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;
    out_src[i] = src_vertex[i];
    out_dst[i] = indices[i];
}

}  

k_truss_result_t k_truss_seg(const graph32_t& graph,
                             int32_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int num_vertices = graph.number_of_vertices;
    int num_edges = graph.number_of_edges;

    int threshold = k - 2;

    if (num_edges == 0 || threshold <= 0) {
        int32_t* out_src = nullptr;
        int32_t* out_dst = nullptr;
        if (num_edges > 0) {
            cudaMalloc(&out_src, num_edges * sizeof(int32_t));
            cudaMalloc(&out_dst, num_edges * sizeof(int32_t));
            cache.ensure_src_a(num_edges);
            compute_src_vertices_kernel<<<(num_edges + 255) / 256, 256>>>(
                offsets, cache.src_a, num_vertices, num_edges);
            cudaMemcpyAsync(out_src, cache.src_a, num_edges * sizeof(int), cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(out_dst, indices, num_edges * sizeof(int), cudaMemcpyDeviceToDevice);
        }
        return k_truss_result_t{out_src, out_dst, static_cast<std::size_t>(num_edges)};
    }

    
    cache.ensure_edges(num_edges);
    cache.ensure_vertices(num_vertices);
    cache.ensure_src_a(num_edges);

    
    const int* cur_offsets = offsets;
    const int* cur_indices = indices;
    int* cur_src = cache.src_a;
    compute_src_vertices_kernel<<<(num_edges + 255) / 256, 256>>>(
        cur_offsets, cur_src, num_vertices, num_edges);

    int* buf_offsets[2] = {cache.offsets_b, cache.offsets_c};
    int* buf_indices[2] = {cache.indices_b, cache.indices_c};
    int* buf_src[2] = {cache.src_b, cache.src_c};

    int cur_num_edges = num_edges;
    int cur_buf = 0;

    while (true) {
        if (cur_num_edges == 0) break;

        
        {
            int grid = ((int64_t)cur_num_edges * 32 + 255) / 256;
            count_triangles_upper_kernel<<<grid, 256>>>(
                cur_offsets, cur_indices, cur_src, cache.support, cur_num_edges);
            copy_support_reverse_kernel<<<(cur_num_edges + 255) / 256, 256>>>(
                cur_offsets, cur_indices, cur_src, cache.support, cur_num_edges);
        }

        
        int new_num_edges;
        {
            int grid_v = (num_vertices + 255) / 256;
            count_surviving_degrees_kernel<<<grid_v, 256>>>(
                cur_offsets, cache.support, cache.degree_buf, threshold, num_vertices);

            cudaMemsetAsync(cache.degree_buf + num_vertices, 0, sizeof(int));
            size_t temp_needed = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, temp_needed,
                cache.degree_buf, buf_offsets[cur_buf], num_vertices + 1);
            cache.ensure_cub_temp(temp_needed);
            cub::DeviceScan::ExclusiveSum(cache.d_cub_temp, cache.cub_temp_bytes,
                cache.degree_buf, buf_offsets[cur_buf], num_vertices + 1);

            cudaMemcpy(&new_num_edges, buf_offsets[cur_buf] + num_vertices,
                       sizeof(int), cudaMemcpyDeviceToHost);

            if (new_num_edges > 0 && new_num_edges != cur_num_edges) {
                scatter_surviving_edges_kernel<<<grid_v, 256>>>(
                    cur_offsets, cur_indices, cache.support, threshold,
                    buf_offsets[cur_buf], buf_indices[cur_buf], buf_src[cur_buf],
                    num_vertices);
            }
        }

        if (new_num_edges == 0) {
            cur_num_edges = 0;
            break;
        }
        if (new_num_edges == cur_num_edges) break;

        cur_offsets = buf_offsets[cur_buf];
        cur_indices = buf_indices[cur_buf];
        cur_src = buf_src[cur_buf];
        cur_num_edges = new_num_edges;
        cur_buf = 1 - cur_buf;
    }

    
    int32_t* out_src = nullptr;
    int32_t* out_dst = nullptr;
    if (cur_num_edges > 0) {
        cudaMalloc(&out_src, cur_num_edges * sizeof(int32_t));
        cudaMalloc(&out_dst, cur_num_edges * sizeof(int32_t));
        write_output_kernel<<<(cur_num_edges + 255) / 256, 256>>>(
            cur_src, cur_indices, out_src, out_dst, cur_num_edges);
    }

    return k_truss_result_t{out_src, out_dst, static_cast<std::size_t>(cur_num_edges)};
}

}  
