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
#include <climits>
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* d_degrees = nullptr;
    int32_t* d_peeled = nullptr;
    int32_t* d_dirty = nullptr;
    int32_t* d_frontier_a = nullptr;
    int32_t* d_frontier_b = nullptr;
    int32_t* d_candidates = nullptr;
    int32_t* d_remaining_a = nullptr;
    int32_t* d_remaining_b = nullptr;
    int32_t* d_counter = nullptr;
    int32_t* h_counter = nullptr;
    cudaStream_t stream = nullptr;
    size_t max_vertices = 0;

    Cache() {
        cudaStreamCreate(&stream);
        cudaMallocHost(&h_counter, 16 * sizeof(int32_t));
        cudaMalloc(&d_counter, 16 * sizeof(int32_t));
    }

    ~Cache() override {
        if (d_degrees) cudaFree(d_degrees);
        if (d_peeled) cudaFree(d_peeled);
        if (d_dirty) cudaFree(d_dirty);
        if (d_frontier_a) cudaFree(d_frontier_a);
        if (d_frontier_b) cudaFree(d_frontier_b);
        if (d_candidates) cudaFree(d_candidates);
        if (d_remaining_a) cudaFree(d_remaining_a);
        if (d_remaining_b) cudaFree(d_remaining_b);
        if (d_counter) cudaFree(d_counter);
        if (h_counter) cudaFreeHost(h_counter);
        if (stream) cudaStreamDestroy(stream);
    }

    void ensure_scratch(int32_t num_vertices) {
        if ((size_t)num_vertices > max_vertices) {
            if (d_degrees) cudaFree(d_degrees);
            if (d_peeled) cudaFree(d_peeled);
            if (d_dirty) cudaFree(d_dirty);
            if (d_frontier_a) cudaFree(d_frontier_a);
            if (d_frontier_b) cudaFree(d_frontier_b);
            if (d_candidates) cudaFree(d_candidates);
            if (d_remaining_a) cudaFree(d_remaining_a);
            if (d_remaining_b) cudaFree(d_remaining_b);

            max_vertices = num_vertices;
            cudaMalloc(&d_degrees, num_vertices * sizeof(int32_t));
            cudaMalloc(&d_peeled, num_vertices * sizeof(int32_t));
            cudaMalloc(&d_dirty, num_vertices * sizeof(int32_t));
            cudaMalloc(&d_frontier_a, num_vertices * sizeof(int32_t));
            cudaMalloc(&d_frontier_b, num_vertices * sizeof(int32_t));
            cudaMalloc(&d_candidates, num_vertices * sizeof(int32_t));
            cudaMalloc(&d_remaining_a, num_vertices * sizeof(int32_t));
            cudaMalloc(&d_remaining_b, num_vertices * sizeof(int32_t));
        }
    }
};




__global__ void compute_masked_degrees(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ degrees,
    int32_t num_vertices)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int count = 0;
        for (int e = start; e < end; e++) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1) {
                if (__ldg(&indices[e]) != v) count++;
            }
        }
        degrees[v] = count;
    }
}




__global__ void init_state_and_remaining(
    const int32_t* __restrict__ degrees,
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ peeled,
    int32_t* __restrict__ remaining,
    int32_t* __restrict__ remaining_count,
    int32_t num_vertices,
    int32_t delta,
    int64_t k_first)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        int cn = degrees[v] * delta;
        if (k_first > 1 && cn > 0 && cn < (int32_t)k_first) cn = 0;
        core_numbers[v] = cn;
        peeled[v] = 0;
        
        if (degrees[v] > 0) {
            int idx = atomicAdd(remaining_count, 1);
            remaining[idx] = v;
        }
    }
}




__global__ void partition_remaining(
    const int32_t* __restrict__ core_numbers,
    const int32_t* __restrict__ peeled,
    const int32_t* __restrict__ remaining_in,
    int32_t remaining_size,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_count,
    int32_t* __restrict__ remaining_out,
    int32_t* __restrict__ remaining_out_count,
    int32_t k)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < remaining_size; i += blockDim.x * gridDim.x) {
        int v = remaining_in[i];
        if (peeled[v]) continue;  
        int cn = core_numbers[v];
        if (cn < k) {
            int idx = atomicAdd(frontier_count, 1);
            frontier[idx] = v;
        } else {
            int idx = atomicAdd(remaining_out_count, 1);
            remaining_out[idx] = v;
        }
    }
}




__global__ void mark_peeled_dev(
    int32_t* __restrict__ peeled,
    const int32_t* __restrict__ frontier,
    const int32_t* __restrict__ frontier_size_ptr)
{
    int frontier_size = *frontier_size_ptr;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < frontier_size; i += blockDim.x * gridDim.x) {
        peeled[frontier[i]] = 1;
    }
}




__global__ void peel_edges_warp_dev(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ core_numbers,
    const int32_t* __restrict__ peeled,
    int32_t* __restrict__ dirty,
    const int32_t* __restrict__ frontier,
    const int32_t* __restrict__ frontier_size_ptr,
    int32_t* __restrict__ candidates,
    int32_t* __restrict__ candidate_count,
    int32_t delta)
{
    int frontier_size = *frontier_size_ptr;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int total_warps = (blockDim.x * gridDim.x) >> 5;

    for (int wi = warp_id; wi < frontier_size; wi += total_warps) {
        int v = frontier[wi];
        int start = offsets[v];
        int end = offsets[v + 1];
        int degree = end - start;

        for (int i = lane; i < degree; i += 32) {
            int e = start + i;
            if (!((edge_mask[e >> 5] >> (e & 31)) & 1)) continue;
            int u = indices[e];
            if (u == v) continue;
            if (peeled[u]) continue;

            atomicSub(&core_numbers[u], delta);
            if (atomicExch(&dirty[u], 1) == 0) {
                int idx = atomicAdd(candidate_count, 1);
                candidates[idx] = u;
            }
        }
    }
}




__global__ void evaluate_candidates_dev(
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ dirty,
    const int32_t* __restrict__ candidates,
    const int32_t* __restrict__ candidate_count_ptr,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_count,
    int32_t k,
    int32_t k_minus_delta,
    int64_t k_first)
{
    int count = *candidate_count_ptr;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < count; tid += blockDim.x * gridDim.x) {
        int u = candidates[tid];
        int val = core_numbers[u];
        if (val < k_minus_delta) { val = k_minus_delta; core_numbers[u] = val; }
        if (k_first > 1 && val > 0 && val < (int32_t)k_first) { val = 0; core_numbers[u] = 0; }
        dirty[u] = 0;
        if (val < k) {
            int idx = atomicAdd(next_frontier_count, 1);
            next_frontier[idx] = u;
        }
    }
}




__global__ void swap_counters(int32_t* frontier_size, int32_t* cand_count, int32_t* next_count) {
    *frontier_size = *next_count;
    *cand_count = 0;
    *next_count = 0;
}




__global__ void compact_remaining(
    const int32_t* __restrict__ peeled,
    const int32_t* __restrict__ remaining_in,
    int32_t remaining_size,
    int32_t* __restrict__ remaining_out,
    int32_t* __restrict__ remaining_out_count)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < remaining_size; i += blockDim.x * gridDim.x) {
        int v = remaining_in[i];
        if (!peeled[v]) {
            int idx = atomicAdd(remaining_out_count, 1);
            remaining_out[idx] = v;
        }
    }
}




__global__ void find_min_remaining(
    const int32_t* __restrict__ core_numbers,
    const int32_t* __restrict__ remaining,
    int32_t remaining_size,
    int32_t* __restrict__ min_result,
    int32_t* __restrict__ remaining_count_out)
{
    typedef cub::BlockReduce<int32_t, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage ts1;
    __shared__ typename BlockReduce::TempStorage ts2;

    int32_t local_min = INT32_MAX;
    int32_t local_count = 0;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < remaining_size; i += blockDim.x * gridDim.x) {
        int cn = core_numbers[remaining[i]];
        if (cn < local_min) local_min = cn;
        local_count++;
    }

    struct MinOp { __device__ int32_t operator()(int32_t a, int32_t b) const { return a < b ? a : b; } };
    int32_t block_min = BlockReduce(ts1).Reduce(local_min, MinOp{});
    __syncthreads();
    int32_t block_count = BlockReduce(ts2).Sum(local_count);

    if (threadIdx.x == 0) {
        atomicMin(min_result, block_min);
        atomicAdd(remaining_count_out, block_count);
    }
}





void launch_compute_masked_degrees(const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    int32_t* degrees, int32_t num_vertices, cudaStream_t s) {
    int block = 256, grid = (num_vertices + block - 1) / block;
    if (grid > 4096) grid = 4096;
    compute_masked_degrees<<<grid, block, 0, s>>>(offsets, indices, edge_mask, degrees, num_vertices);
}

void launch_init_state_and_remaining(const int32_t* degrees, int32_t* core_numbers, int32_t* peeled,
    int32_t* remaining, int32_t* remaining_count, int32_t num_vertices, int32_t delta, int64_t k_first, cudaStream_t s) {
    int block = 256, grid = (num_vertices + block - 1) / block;
    if (grid > 4096) grid = 4096;
    init_state_and_remaining<<<grid, block, 0, s>>>(degrees, core_numbers, peeled, remaining, remaining_count,
        num_vertices, delta, k_first);
}

void launch_partition_remaining(const int32_t* core_numbers, const int32_t* peeled,
    const int32_t* remaining_in, int32_t remaining_size,
    int32_t* frontier, int32_t* frontier_count, int32_t* remaining_out, int32_t* remaining_out_count,
    int32_t k, cudaStream_t s) {
    int block = 256, grid = (remaining_size + block - 1) / block;
    if (grid > 4096) grid = 4096;
    if (grid < 1) grid = 1;
    partition_remaining<<<grid, block, 0, s>>>(core_numbers, peeled, remaining_in, remaining_size,
        frontier, frontier_count, remaining_out, remaining_out_count, k);
}

void launch_peel_iteration(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    int32_t* core_numbers, int32_t* peeled, int32_t* dirty,
    int32_t* cur_frontier, const int32_t* frontier_size_ptr,
    int32_t* candidates, int32_t* candidate_count_ptr,
    int32_t* next_frontier, int32_t* next_frontier_count_ptr,
    int32_t k, int32_t k_minus_delta, int64_t k_first, int32_t delta,
    int32_t max_grid, cudaStream_t s)
{
    int block = 256;
    mark_peeled_dev<<<max_grid, block, 0, s>>>(peeled, cur_frontier, frontier_size_ptr);
    peel_edges_warp_dev<<<max_grid, block, 0, s>>>(
        offsets, indices, edge_mask, core_numbers, peeled, dirty,
        cur_frontier, frontier_size_ptr, candidates, candidate_count_ptr, delta);
    evaluate_candidates_dev<<<max_grid, block, 0, s>>>(
        core_numbers, dirty, candidates, candidate_count_ptr,
        next_frontier, next_frontier_count_ptr, k, k_minus_delta, k_first);
    swap_counters<<<1, 1, 0, s>>>(
        const_cast<int32_t*>(frontier_size_ptr), candidate_count_ptr, next_frontier_count_ptr);
}

void launch_compact_remaining(const int32_t* peeled, const int32_t* remaining_in, int32_t remaining_size,
    int32_t* remaining_out, int32_t* remaining_out_count, cudaStream_t s) {
    int block = 256, grid = (remaining_size + block - 1) / block;
    if (grid > 4096) grid = 4096;
    if (grid < 1) grid = 1;
    compact_remaining<<<grid, block, 0, s>>>(peeled, remaining_in, remaining_size, remaining_out, remaining_out_count);
}

void launch_find_min_remaining(const int32_t* core_numbers, const int32_t* remaining, int32_t remaining_size,
    int32_t* min_result, int32_t* remaining_count, cudaStream_t s) {
    int block = 256, grid = (remaining_size + block - 1) / block;
    if (grid > 1024) grid = 1024;
    if (grid < 1) grid = 1;
    find_min_remaining<<<grid, block, 0, s>>>(core_numbers, remaining, remaining_size, min_result, remaining_count);
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
    if (num_vertices == 0) return;

    cache.ensure_scratch(num_vertices);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;
    bool is_symmetric = graph.is_symmetric;
    int32_t delta = (is_symmetric && degree_type == 2) ? 2 : 1;

    cudaStream_t stream = cache.stream;
    int32_t* h_counter = cache.h_counter;
    int32_t* d_counter = cache.d_counter;
    int32_t max_grid = 4096;

    
    launch_compute_masked_degrees(d_offsets, d_indices, d_edge_mask, cache.d_degrees, num_vertices, stream);

    
    cudaMemsetAsync(d_counter, 0, 16 * sizeof(int32_t), stream);
    cudaMemsetAsync(cache.d_dirty, 0, num_vertices * sizeof(int32_t), stream);
    launch_init_state_and_remaining(cache.d_degrees, core_numbers, cache.d_peeled, cache.d_remaining_a, d_counter + 5,
                                     num_vertices, delta, (int64_t)k_first, stream);
    
    cudaMemcpyAsync(h_counter + 5, d_counter + 5, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int32_t remaining_size = h_counter[5];

    int32_t* cur_remaining = cache.d_remaining_a;
    int32_t* other_remaining = cache.d_remaining_b;

    
    size_t k = (k_first < 2) ? 2 : k_first;
    if (is_symmetric && degree_type == 2 && (k % 2 == 1)) k++;

    int32_t* cur_frontier = cache.d_frontier_a;
    int32_t* next_frontier = cache.d_frontier_b;
    int32_t* frontier_size_ptr = d_counter + 0;
    int32_t* candidate_count_ptr = d_counter + 1;
    int32_t* next_frontier_count_ptr = d_counter + 4;

    while (k <= k_last && remaining_size > 0) {
        
        h_counter[6] = 0; 
        h_counter[7] = 0; 
        cudaMemcpyAsync(d_counter + 6, h_counter + 6, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        launch_partition_remaining(core_numbers, cache.d_peeled, cur_remaining, remaining_size,
                                   cur_frontier, d_counter + 6, other_remaining, d_counter + 7,
                                   (int32_t)k, stream);
        cudaMemcpyAsync(h_counter + 6, d_counter + 6, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int32_t frontier_size = h_counter[6];
        remaining_size = h_counter[7];
        std::swap(cur_remaining, other_remaining);

        if (frontier_size == 0) {
            
            h_counter[2] = INT32_MAX;
            h_counter[3] = 0;
            cudaMemcpyAsync(d_counter + 2, h_counter + 2, 2 * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
            launch_find_min_remaining(core_numbers, cur_remaining, remaining_size,
                                      d_counter + 2, d_counter + 3, stream);
            cudaMemcpyAsync(h_counter + 2, d_counter + 2, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            if (h_counter[3] == 0) break;
            int32_t min_core = h_counter[2];
            size_t new_k = (size_t)min_core + (size_t)delta;
            k = (new_k > k + (size_t)delta) ? new_k : k + (size_t)delta;
            if (is_symmetric && degree_type == 2 && (k % 2 == 1)) k++;
            continue;
        }

        
        h_counter[0] = frontier_size;
        h_counter[1] = 0;
        h_counter[4] = 0;
        cudaMemcpyAsync(d_counter, h_counter, 5 * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

        
        int iter = 0;
        while (true) {
            launch_peel_iteration(
                d_offsets, d_indices, d_edge_mask,
                core_numbers, cache.d_peeled, cache.d_dirty,
                cur_frontier, frontier_size_ptr,
                cache.d_candidates, candidate_count_ptr,
                next_frontier, next_frontier_count_ptr,
                (int32_t)k, (int32_t)(k - delta), (int64_t)k_first, delta,
                max_grid, stream);

            std::swap(cur_frontier, next_frontier);
            iter++;

            
            if (iter <= 2 || iter % 4 == 0) {
                cudaMemcpyAsync(h_counter, frontier_size_ptr, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                if (h_counter[0] == 0) break;
            }
        }

        
        h_counter[5] = 0;
        cudaMemcpyAsync(d_counter + 5, h_counter + 5, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        launch_compact_remaining(cache.d_peeled, cur_remaining, remaining_size,
                                 other_remaining, d_counter + 5, stream);
        cudaMemcpyAsync(h_counter + 5, d_counter + 5, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        remaining_size = h_counter[5];
        std::swap(cur_remaining, other_remaining);

        k += delta;
        if (is_symmetric && degree_type == 2 && (k % 2 == 1)) k++;
    }
}

}  
