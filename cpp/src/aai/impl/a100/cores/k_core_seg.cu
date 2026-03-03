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
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    uint8_t* in_kcore = nullptr;
    int32_t* degrees = nullptr;
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* frontier_sizes = nullptr;
    int32_t* counter = nullptr;

    int64_t in_kcore_capacity = 0;
    int64_t degrees_capacity = 0;
    int64_t frontier_a_capacity = 0;
    int64_t frontier_b_capacity = 0;

    Cache() {
        cudaMalloc(&frontier_sizes, 2 * sizeof(int32_t));
        cudaMalloc(&counter, sizeof(int32_t));
    }

    ~Cache() override {
        if (in_kcore) cudaFree(in_kcore);
        if (degrees) cudaFree(degrees);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (frontier_sizes) cudaFree(frontier_sizes);
        if (counter) cudaFree(counter);
    }

    void ensure(int64_t nv) {
        if (in_kcore_capacity < nv) {
            if (in_kcore) cudaFree(in_kcore);
            cudaMalloc(&in_kcore, nv * sizeof(uint8_t));
            in_kcore_capacity = nv;
        }
        if (degrees_capacity < nv) {
            if (degrees) cudaFree(degrees);
            cudaMalloc(&degrees, nv * sizeof(int32_t));
            degrees_capacity = nv;
        }
        if (frontier_a_capacity < nv) {
            if (frontier_a) cudaFree(frontier_a);
            cudaMalloc(&frontier_a, nv * sizeof(int32_t));
            frontier_a_capacity = nv;
        }
        if (frontier_b_capacity < nv) {
            if (frontier_b) cudaFree(frontier_b);
            cudaMalloc(&frontier_b, nv * sizeof(int32_t));
            frontier_b_capacity = nv;
        }
    }
};






__global__ void compute_degrees_block_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ degrees,
    int32_t multiplier,
    int32_t v_start,
    int32_t v_count)
{
    if (blockIdx.x >= (unsigned)v_count) return;
    int v = v_start + blockIdx.x;
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    
    int32_t my_self_loops = 0;
    for (int32_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        if (indices[i] == v) my_self_loops++;
    }

    
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        my_self_loops += __shfl_down_sync(0xFFFFFFFF, my_self_loops, s);

    __shared__ int32_t warp_sums[8];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = my_self_loops;
    __syncthreads();

    if (threadIdx.x == 0) {
        int32_t total_self = 0;
        int nwarps = blockDim.x >> 5;
        for (int i = 0; i < nwarps; i++)
            total_self += warp_sums[i];
        degrees[v] = (end - start - total_self) * multiplier;
    }
}


__global__ void compute_degrees_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ degrees,
    int32_t multiplier,
    int32_t v_start,
    int32_t v_count)
{
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_global >= v_count) return;

    int v = v_start + warp_global;
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    int32_t my_self_loops = 0;
    for (int32_t i = start + lane; i < end; i += 32) {
        if (indices[i] == v) my_self_loops++;
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        my_self_loops += __shfl_down_sync(0xFFFFFFFF, my_self_loops, s);

    if (lane == 0) {
        degrees[v] = (end - start - my_self_loops) * multiplier;
    }
}


__global__ void compute_degrees_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ degrees,
    int32_t multiplier,
    int32_t v_start,
    int32_t v_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= v_count) return;
    int v = v_start + tid;
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t self_loops = 0;
    for (int32_t i = start; i < end; i++) {
        if (indices[i] == v) self_loops++;
    }
    degrees[v] = (end - start - self_loops) * multiplier;
}




__global__ void build_frontier_kernel(
    const int32_t* __restrict__ degrees,
    uint8_t* __restrict__ in_kcore,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_size,
    int32_t k,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    if (degrees[v] < k) {
        in_kcore[v] = 0;
        frontier[atomicAdd(frontier_size, 1)] = v;
    } else {
        in_kcore[v] = 1;
    }
}

__global__ void process_frontier_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ degrees,
    uint8_t* __restrict__ in_kcore,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t k,
    int32_t multiplier)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int v = frontier[tid];
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    for (int32_t i = start; i < end; i++) {
        int32_t u = indices[i];
        if (u == v) continue;
        if (!in_kcore[u]) continue;
        int32_t old_deg = atomicSub(&degrees[u], multiplier);
        if (old_deg >= k && (old_deg - multiplier) < k) {
            in_kcore[u] = 0;
            next_frontier[atomicAdd(next_frontier_size, 1)] = u;
        }
    }
}

__global__ void mark_kcore_kernel(
    const int32_t* __restrict__ core_numbers,
    uint8_t* __restrict__ in_kcore,
    int32_t k,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    in_kcore[v] = (core_numbers[v] >= k) ? 1 : 0;
}






__global__ void extract_edges_block_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ in_kcore,
    int32_t* __restrict__ edge_srcs,
    int32_t* __restrict__ edge_dsts,
    int32_t* __restrict__ counter,
    int32_t v_start,
    int32_t v_count)
{
    if (blockIdx.x >= (unsigned)v_count) return;
    int v = v_start + blockIdx.x;
    if (!in_kcore[v]) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    int32_t my_count = 0;
    for (int32_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        if (in_kcore[indices[i]]) my_count++;
    }

    
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        my_count += __shfl_down_sync(0xFFFFFFFF, my_count, s);

    __shared__ int32_t warp_sums[8];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = my_count;
    __syncthreads();

    __shared__ int32_t sh_total;
    __shared__ int32_t sh_base;
    if (threadIdx.x == 0) {
        int32_t total = 0;
        int nwarps = blockDim.x >> 5;
        for (int i = 0; i < nwarps; i++) total += warp_sums[i];
        sh_total = total;
        if (total > 0) sh_base = atomicAdd(counter, total);
    }
    __syncthreads();
    if (sh_total == 0) return;

    __shared__ int32_t write_pos;
    if (threadIdx.x == 0) write_pos = sh_base;
    __syncthreads();

    for (int32_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        int32_t u = indices[i];
        if (in_kcore[u]) {
            int32_t pos = atomicAdd(&write_pos, 1);
            edge_srcs[pos] = v;
            edge_dsts[pos] = u;
        }
    }
}


__global__ void extract_edges_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ in_kcore,
    int32_t* __restrict__ edge_srcs,
    int32_t* __restrict__ edge_dsts,
    int32_t* __restrict__ counter,
    int32_t v_start,
    int32_t v_count)
{
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_global >= v_count) return;

    int v = v_start + warp_global;
    if (!in_kcore[v]) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    int32_t my_count = 0;
    for (int32_t i = start + lane; i < end; i += 32) {
        if (in_kcore[indices[i]]) my_count++;
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        my_count += __shfl_down_sync(0xFFFFFFFF, my_count, s);

    int32_t count = __shfl_sync(0xFFFFFFFF, my_count, 0);
    if (count == 0) return;

    int32_t base_pos;
    if (lane == 0) base_pos = atomicAdd(counter, count);
    base_pos = __shfl_sync(0xFFFFFFFF, base_pos, 0);

    for (int32_t chunk = start; chunk < end; chunk += 32) {
        int32_t i = chunk + lane;
        int pred = 0;
        int32_t u = 0;
        if (i < end) {
            u = indices[i];
            pred = in_kcore[u] ? 1 : 0;
        }
        unsigned int mask = __ballot_sync(0xFFFFFFFF, pred);
        int pos_in_chunk = __popc(mask & ((1u << lane) - 1));
        if (pred) {
            edge_srcs[base_pos + pos_in_chunk] = v;
            edge_dsts[base_pos + pos_in_chunk] = u;
        }
        base_pos += __popc(mask);
    }
}


__global__ void extract_edges_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ in_kcore,
    int32_t* __restrict__ edge_srcs,
    int32_t* __restrict__ edge_dsts,
    int32_t* __restrict__ counter,
    int32_t v_start,
    int32_t v_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= v_count) return;
    int v = v_start + tid;
    if (!in_kcore[v]) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    int32_t count = 0;
    for (int32_t i = start; i < end; i++) {
        if (in_kcore[indices[i]]) count++;
    }
    if (count == 0) return;
    int32_t pos = atomicAdd(counter, count);
    for (int32_t i = start; i < end; i++) {
        int32_t u = indices[i];
        if (in_kcore[u]) {
            edge_srcs[pos] = v;
            edge_dsts[pos] = u;
            pos++;
        }
    }
}

}  

std::size_t k_core_seg(const graph32_t& graph,
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
    cudaStream_t stream = 0;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2];
    int32_t n_high = seg1 - seg0;
    int32_t n_mid = seg2 - seg1;
    int32_t n_low_and_zero = num_vertices - seg2;

    int32_t k_val = static_cast<int32_t>(k);

    int32_t multiplier = (degree_type == 2) ? 2 : 1;
    if (degree_type < 0 && core_numbers == nullptr)
        multiplier = 2;

    cache.ensure(num_vertices);
    uint8_t* d_in_kcore = cache.in_kcore;

    if (core_numbers != nullptr) {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        if (grid > 0)
            mark_kcore_kernel<<<grid, block, 0, stream>>>(
                core_numbers, d_in_kcore, k_val, num_vertices);
    } else {
        int32_t* d_degrees = cache.degrees;

        if (n_high > 0)
            compute_degrees_block_kernel<<<n_high, 256, 0, stream>>>(
                d_offsets, d_indices, d_degrees, multiplier, seg0, n_high);

        if (n_mid > 0) {
            int warps_per_block = 8;
            int grid = (n_mid + warps_per_block - 1) / warps_per_block;
            compute_degrees_warp_kernel<<<grid, warps_per_block * 32, 0, stream>>>(
                d_offsets, d_indices, d_degrees, multiplier, seg1, n_mid);
        }

        if (n_low_and_zero > 0) {
            int block = 256;
            int grid = (n_low_and_zero + block - 1) / block;
            compute_degrees_thread_kernel<<<grid, block, 0, stream>>>(
                d_offsets, d_indices, d_degrees, multiplier, seg2, n_low_and_zero);
        }

        int32_t* d_fa = cache.frontier_a;
        int32_t* d_fb = cache.frontier_b;
        int32_t* d_sa = cache.frontier_sizes;
        int32_t* d_sb = cache.frontier_sizes + 1;

        cudaMemsetAsync(d_sa, 0, 2 * sizeof(int32_t), stream);

        {
            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            if (grid > 0)
                build_frontier_kernel<<<grid, block, 0, stream>>>(
                    d_degrees, d_in_kcore, d_fa, d_sa, k_val, num_vertices);
        }

        int32_t h_fs;
        cudaMemcpyAsync(&h_fs, d_sa, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        int32_t* d_cf = d_fa;
        int32_t* d_nf = d_fb;
        int32_t* d_ns = d_sb;

        while (h_fs > 0) {
            cudaMemsetAsync(d_ns, 0, sizeof(int32_t), stream);
            {
                int block = 256;
                int grid = (h_fs + block - 1) / block;
                if (grid > 0)
                    process_frontier_kernel<<<grid, block, 0, stream>>>(
                        d_offsets, d_indices, d_cf, h_fs,
                        d_degrees, d_in_kcore, d_nf, d_ns, k_val, multiplier);
            }
            cudaMemcpyAsync(&h_fs, d_ns, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            std::swap(d_cf, d_nf);
            d_ns = (d_cf == d_fa) ? d_sb : d_sa;
        }
    }

    
    int32_t* d_cnt = cache.counter;
    cudaMemsetAsync(d_cnt, 0, sizeof(int32_t), stream);

    if (n_high > 0)
        extract_edges_block_kernel<<<n_high, 256, 0, stream>>>(
            d_offsets, d_indices, d_in_kcore, edge_srcs, edge_dsts, d_cnt, seg0, n_high);

    if (n_mid > 0) {
        int warps_per_block = 8;
        int grid = (n_mid + warps_per_block - 1) / warps_per_block;
        extract_edges_warp_kernel<<<grid, warps_per_block * 32, 0, stream>>>(
            d_offsets, d_indices, d_in_kcore, edge_srcs, edge_dsts, d_cnt, seg1, n_mid);
    }

    if (n_low_and_zero > 0) {
        int block = 256;
        int grid = (n_low_and_zero + block - 1) / block;
        extract_edges_thread_kernel<<<grid, block, 0, stream>>>(
            d_offsets, d_indices, d_in_kcore, edge_srcs, edge_dsts, d_cnt, seg2, n_low_and_zero);
    }

    int32_t h_count;
    cudaMemcpyAsync(&h_count, d_cnt, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    return static_cast<std::size_t>(h_count);
}

}  
