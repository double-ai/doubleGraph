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
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <cstdint>
#include <cstddef>
#include <climits>

namespace aai {

namespace {

struct Cache : Cacheable {
    
    int32_t* valid = nullptr;
    int32_t* pushed_val = nullptr;
    int32_t* frontier_buf0 = nullptr;
    int32_t* frontier_buf1 = nullptr;
    int32_t* d_frontier_count = nullptr;
    int64_t decomp_capacity = 0;

    
    int32_t* in_kcore = nullptr;
    int32_t* edge_counts = nullptr;
    int32_t* write_offsets = nullptr;
    int64_t extract_capacity = 0;

    
    int32_t* core_numbers_buf = nullptr;
    int64_t cn_capacity = 0;

    
    void* cub_temp = nullptr;
    size_t cub_temp_bytes = 0;

    void ensure_decomp(int32_t n) {
        if (decomp_capacity < n) {
            if (valid) cudaFree(valid);
            if (pushed_val) cudaFree(pushed_val);
            if (frontier_buf0) cudaFree(frontier_buf0);
            if (frontier_buf1) cudaFree(frontier_buf1);
            if (d_frontier_count) cudaFree(d_frontier_count);
            cudaMalloc(&valid, n * sizeof(int32_t));
            cudaMalloc(&pushed_val, n * sizeof(int32_t));
            cudaMalloc(&frontier_buf0, n * sizeof(int32_t));
            cudaMalloc(&frontier_buf1, n * sizeof(int32_t));
            cudaMalloc(&d_frontier_count, sizeof(int32_t));
            decomp_capacity = n;
        }
    }

    void ensure_extract(int32_t n) {
        if (extract_capacity < n) {
            if (in_kcore) cudaFree(in_kcore);
            if (edge_counts) cudaFree(edge_counts);
            if (write_offsets) cudaFree(write_offsets);
            cudaMalloc(&in_kcore, n * sizeof(int32_t));
            cudaMalloc(&edge_counts, n * sizeof(int32_t));
            cudaMalloc(&write_offsets, n * sizeof(int32_t));
            extract_capacity = n;
        }
    }

    void ensure_cn(int32_t n) {
        if (cn_capacity < n) {
            if (core_numbers_buf) cudaFree(core_numbers_buf);
            cudaMalloc(&core_numbers_buf, n * sizeof(int32_t));
            cn_capacity = n;
        }
    }

    void ensure_cub_temp(int32_t n) {
        size_t temp_bytes_needed = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes_needed,
                                       (int32_t*)nullptr, (int32_t*)nullptr, n);
        if (temp_bytes_needed > cub_temp_bytes) {
            if (cub_temp) cudaFree(cub_temp);
            cub_temp_bytes = temp_bytes_needed * 2;
            cudaMalloc(&cub_temp, cub_temp_bytes);
        }
    }

    ~Cache() override {
        if (valid) cudaFree(valid);
        if (pushed_val) cudaFree(pushed_val);
        if (frontier_buf0) cudaFree(frontier_buf0);
        if (frontier_buf1) cudaFree(frontier_buf1);
        if (d_frontier_count) cudaFree(d_frontier_count);
        if (in_kcore) cudaFree(in_kcore);
        if (edge_counts) cudaFree(edge_counts);
        if (write_offsets) cudaFree(write_offsets);
        if (core_numbers_buf) cudaFree(core_numbers_buf);
        if (cub_temp) cudaFree(cub_temp);
    }
};


__device__ __forceinline__ bool is_edge_active(const uint32_t* edge_mask, int32_t edge_id) {
    return (edge_mask[edge_id >> 5] >> (edge_id & 31)) & 1;
}


__global__ void compute_degrees_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask, int32_t num_vertices,
    int32_t* __restrict__ degrees, int32_t degree_multiplier
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int start = offsets[v], end = offsets[v + 1], deg = 0;
    for (int e = start; e < end; e++) {
        if (indices[e] != v && is_edge_active(edge_mask, e)) deg++;
    }
    degrees[v] = deg * degree_multiplier;
}

__global__ void init_ones_kernel(int32_t* arr, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = 1;
}

__global__ void mark_invalid_kernel(const int32_t* frontier, int32_t count, int32_t* valid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) valid[frontier[idx]] = 0;
}

__global__ void accumulate_pushed_val_warp_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask, const int32_t* __restrict__ frontier,
    int32_t frontier_size, int32_t* __restrict__ pushed_val,
    const int32_t* __restrict__ valid, int32_t delta
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (warp_id >= frontier_size) return;
    int v = frontier[warp_id];
    int start = offsets[v], end = offsets[v + 1];
    for (int e = start + lane; e < end; e += 32) {
        int u = indices[e];
        if (u != v && is_edge_active(edge_mask, e) && valid[u])
            atomicAdd(&pushed_val[u], delta);
    }
}

__global__ void accumulate_pushed_val_simple_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask, const int32_t* __restrict__ frontier,
    int32_t frontier_size, int32_t* __restrict__ pushed_val,
    const int32_t* __restrict__ valid, int32_t delta
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;
    int v = frontier[idx];
    int start = offsets[v], end = offsets[v + 1];
    for (int e = start; e < end; e++) {
        int u = indices[e];
        if (u != v && is_edge_active(edge_mask, e) && valid[u])
            atomicAdd(&pushed_val[u], delta);
    }
}

__global__ void update_cn_find_frontier_kernel(
    int32_t num_vertices, int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ pushed_val, int32_t* __restrict__ valid,
    int32_t k, int32_t k_minus_delta,
    int32_t* __restrict__ next_frontier, int32_t* __restrict__ next_frontier_count
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int pv = pushed_val[v];
    if (pv == 0) return;
    pushed_val[v] = 0;
    int cn = core_numbers[v];
    int new_cn = cn >= pv ? cn - pv : 0;
    if (new_cn < k_minus_delta) new_cn = k_minus_delta;
    if (new_cn < 0) new_cn = 0;
    core_numbers[v] = new_cn;
    if (new_cn < k && valid[v]) {
        int pos = atomicAdd(next_frontier_count, 1);
        next_frontier[pos] = v;
    }
}

__global__ void mark_kcore_kernel(int32_t num_vertices, const int32_t* core_numbers, int32_t k, int32_t* in_kcore) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    in_kcore[v] = (core_numbers[v] >= k) ? 1 : 0;
}

__global__ void count_kcore_edges_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask, const int32_t* __restrict__ in_kcore,
    int32_t num_vertices, int32_t* __restrict__ edge_counts
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    if (!in_kcore[v]) { edge_counts[v] = 0; return; }
    int start = offsets[v], end = offsets[v + 1], count = 0;
    for (int e = start; e < end; e++) {
        if (is_edge_active(edge_mask, e) && in_kcore[indices[e]]) count++;
    }
    edge_counts[v] = count;
}

__global__ void write_kcore_edges_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask, const int32_t* __restrict__ in_kcore,
    int32_t num_vertices, const int32_t* __restrict__ write_offsets,
    int32_t* __restrict__ edge_srcs, int32_t* __restrict__ edge_dsts
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices || !in_kcore[v]) return;
    int start = offsets[v], end = offsets[v + 1], pos = write_offsets[v];
    for (int e = start; e < end; e++) {
        int u = indices[e];
        if (is_edge_active(edge_mask, e) && in_kcore[u]) {
            edge_srcs[pos] = v; edge_dsts[pos] = u; pos++;
        }
    }
}


struct FindFrontierPred {
    const int32_t* core_numbers;
    const int32_t* valid;
    int32_t k;
    __device__ bool operator()(int v) const {
        return core_numbers[v] > 0 && core_numbers[v] < k && valid[v];
    }
};

struct RemainingPred {
    const int32_t* valid;
    const int32_t* core_numbers;
    __device__ bool operator()(int v) const {
        return valid[v] && core_numbers[v] > 0;
    }
};

struct MinCnTransform {
    const int32_t* valid;
    const int32_t* core_numbers;
    __device__ int32_t operator()(int v) const {
        return (valid[v] && core_numbers[v] > 0) ? core_numbers[v] : INT_MAX;
    }
};


static void core_number_decomposition(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    int32_t num_vertices, int32_t num_edges,
    int32_t* core_numbers, int32_t degree_multiplier, int32_t delta,
    int32_t* valid, int32_t* pushed_val,
    int32_t* frontier_buf0, int32_t* frontier_buf1,
    int32_t* d_frontier_count,
    cudaStream_t stream
) {
    const int BLOCK = 256;
    if (num_vertices == 0) return;

    
    int grid = (num_vertices + BLOCK - 1) / BLOCK;
    compute_degrees_kernel<<<grid, BLOCK, 0, stream>>>(
        offsets, indices, edge_mask, num_vertices, core_numbers, degree_multiplier);

    
    init_ones_kernel<<<grid, BLOCK, 0, stream>>>(valid, num_vertices);

    
    cudaMemsetAsync(pushed_val, 0, num_vertices * sizeof(int32_t), stream);

    
    RemainingPred remaining_pred{valid, core_numbers};
    int32_t h_remaining = (int32_t)thrust::count_if(
        thrust::cuda::par.on(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_vertices),
        remaining_pred);

    
    int32_t k = 2;
    if (delta == 2 && (k % 2 == 1)) k++;

    int32_t* frontier_buf[2] = {frontier_buf0, frontier_buf1};
    int cur_buf = 0;

    while (h_remaining > 0) {
        
        FindFrontierPred find_pred{core_numbers, valid, k};
        auto frontier_end = thrust::copy_if(
            thrust::cuda::par.on(stream),
            thrust::make_counting_iterator(0),
            thrust::make_counting_iterator(num_vertices),
            frontier_buf[cur_buf],
            find_pred);
        int32_t h_frontier_count = (int32_t)(frontier_end - frontier_buf[cur_buf]);

        if (h_frontier_count > 0) {
            while (h_frontier_count > 0) {
                
                int fg = (h_frontier_count + BLOCK - 1) / BLOCK;
                mark_invalid_kernel<<<fg, BLOCK, 0, stream>>>(
                    frontier_buf[cur_buf], h_frontier_count, valid);

                
                int64_t threads = (int64_t)h_frontier_count * 32;
                int ag = (int)((threads + BLOCK - 1) / BLOCK);
                accumulate_pushed_val_warp_kernel<<<ag, BLOCK, 0, stream>>>(
                    offsets, indices, edge_mask,
                    frontier_buf[cur_buf], h_frontier_count,
                    pushed_val, valid, delta);

                
                int next_buf = 1 - cur_buf;
                cudaMemsetAsync(d_frontier_count, 0, sizeof(int32_t), stream);
                int ug = (num_vertices + BLOCK - 1) / BLOCK;
                int32_t k_minus_delta = k - delta;
                if (k_minus_delta < 0) k_minus_delta = 0;
                update_cn_find_frontier_kernel<<<ug, BLOCK, 0, stream>>>(
                    num_vertices, core_numbers, pushed_val, valid,
                    k, k_minus_delta,
                    frontier_buf[next_buf], d_frontier_count);

                cudaMemcpyAsync(&h_frontier_count, d_frontier_count, sizeof(int32_t),
                                cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                cur_buf = next_buf;
            }

            
            remaining_pred = RemainingPred{valid, core_numbers};
            h_remaining = (int32_t)thrust::count_if(
                thrust::cuda::par.on(stream),
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(num_vertices),
                remaining_pred);
        } else {
            
            MinCnTransform min_xform{valid, core_numbers};
            int32_t min_cn = thrust::transform_reduce(
                thrust::cuda::par.on(stream),
                thrust::make_counting_iterator(0),
                thrust::make_counting_iterator(num_vertices),
                min_xform,
                (int32_t)INT_MAX,
                thrust::minimum<int32_t>());

            if (min_cn == INT_MAX) break;

            int32_t new_k = min_cn;
            if (delta == 2 && (new_k % 2 == 1)) new_k++;
            if (new_k <= k) new_k = k + delta;
            k = new_k;
            continue;
        }

        k += delta;
    }
}

}  

std::size_t k_core_seg_mask(const graph32_t& graph,
                            std::size_t k,
                            int degree_type,
                            const int32_t* core_numbers,
                            int32_t* edge_srcs,
                            int32_t* edge_dsts,
                            std::size_t max_edges) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cudaStream_t stream = 0;
    const int BLOCK = 256;

    cache.ensure_extract(num_vertices);

    int32_t* d_in_kcore = cache.in_kcore;

    if (core_numbers != nullptr) {
        
        mark_kcore_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
            num_vertices, core_numbers, static_cast<int32_t>(k), d_in_kcore);
    } else {
        
        int32_t degree_multiplier = (degree_type == 2) ? 2 : 1;
        int32_t delta = (degree_type == 2) ? 2 : 1;

        cache.ensure_cn(num_vertices);
        cache.ensure_decomp(num_vertices);

        core_number_decomposition(
            d_offsets, d_indices, d_edge_mask,
            num_vertices, num_edges,
            cache.core_numbers_buf, degree_multiplier, delta,
            cache.valid, cache.pushed_val,
            cache.frontier_buf0, cache.frontier_buf1,
            cache.d_frontier_count,
            stream);

        
        mark_kcore_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
            num_vertices, cache.core_numbers_buf, static_cast<int32_t>(k), d_in_kcore);
    }

    
    int32_t* d_edge_counts = cache.edge_counts;
    int32_t* d_write_offsets = cache.write_offsets;

    count_kcore_edges_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
        d_offsets, d_indices, d_edge_mask, d_in_kcore, num_vertices, d_edge_counts);

    cache.ensure_cub_temp(num_vertices);
    size_t temp_bytes = cache.cub_temp_bytes;
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, temp_bytes,
                                   d_edge_counts, d_write_offsets, num_vertices, stream);

    int32_t h_total_edges = 0;
    if (num_vertices > 0) {
        int32_t h_last_offset = 0, h_last_count = 0;
        cudaMemcpyAsync(&h_last_offset, d_write_offsets + num_vertices - 1, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&h_last_count, d_edge_counts + num_vertices - 1, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        h_total_edges = h_last_offset + h_last_count;
    }

    std::size_t result_edges = static_cast<std::size_t>(h_total_edges);

    if (h_total_edges > 0) {
        write_kcore_edges_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
            d_offsets, d_indices, d_edge_mask, d_in_kcore,
            num_vertices, d_write_offsets,
            edge_srcs, edge_dsts);
    }

    cudaStreamSynchronize(stream);

    return result_edges;
}

}  
