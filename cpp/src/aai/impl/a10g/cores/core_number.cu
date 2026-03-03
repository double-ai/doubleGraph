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
#include <cstdint>
#include <climits>

namespace aai {

namespace {

namespace cg = cooperative_groups;

struct Cache : Cacheable {
    int32_t* valid = nullptr;
    int32_t* in_next = nullptr;
    int32_t* rem_a = nullptr;
    int32_t* rem_b = nullptr;
    int32_t* front_a = nullptr;
    int32_t* front_b = nullptr;
    int32_t* counters = nullptr;
    int32_t capacity = 0;

    void ensure(int32_t n) {
        if (capacity < n) {
            if (valid) cudaFree(valid);
            if (in_next) cudaFree(in_next);
            if (rem_a) cudaFree(rem_a);
            if (rem_b) cudaFree(rem_b);
            if (front_a) cudaFree(front_a);
            if (front_b) cudaFree(front_b);
            if (counters) cudaFree(counters);
            cudaMalloc(&valid, n * sizeof(int32_t));
            cudaMalloc(&in_next, n * sizeof(int32_t));
            cudaMalloc(&rem_a, n * sizeof(int32_t));
            cudaMalloc(&rem_b, n * sizeof(int32_t));
            cudaMalloc(&front_a, n * sizeof(int32_t));
            cudaMalloc(&front_b, n * sizeof(int32_t));
            cudaMalloc(&counters, 8 * sizeof(int32_t));
            capacity = n;
        }
    }

    ~Cache() override {
        if (valid) cudaFree(valid);
        if (in_next) cudaFree(in_next);
        if (rem_a) cudaFree(rem_a);
        if (rem_b) cudaFree(rem_b);
        if (front_a) cudaFree(front_a);
        if (front_b) cudaFree(front_b);
        if (counters) cudaFree(counters);
    }
};

__global__ void init_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ valid,
    int32_t degree_mul,
    int32_t k_first,
    int32_t n)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int deg = 0;
        for (int e = start; e < end; e++) {
            if (indices[e] != v) deg++;
        }
        int cn = deg * degree_mul;
        int has_edges = (cn > 0) ? 1 : 0;
        if (cn > 0 && k_first > 1 && cn < k_first) {
            cn = 0;
        }
        core_numbers[v] = cn;
        valid[v] = has_edges;
    }
}

__global__ void build_remaining_kernel(
    const int32_t* __restrict__ valid,
    int32_t* __restrict__ remaining,
    int32_t* __restrict__ remaining_size,
    int32_t n)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n && valid[v]) {
        int pos = atomicAdd(remaining_size, 1);
        remaining[pos] = v;
    }
}

__global__ void process_all_levels_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ valid,
    int32_t* __restrict__ in_next,
    int32_t* __restrict__ remaining_a,
    int32_t* __restrict__ remaining_b,
    int32_t* __restrict__ frontier_a,
    int32_t* __restrict__ frontier_b,
    int32_t* __restrict__ counters,
    int32_t initial_rem_count,
    int32_t k_start,
    int32_t delta,
    int32_t k_first,
    int32_t k_last_val)
{
    cg::grid_group grid = cg::this_grid();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int total_warps = stride >> 5;

    int32_t* cur_rem = remaining_a;
    int32_t* nxt_rem = remaining_b;
    int32_t rem_count = initial_rem_count;

    int32_t* cur_front = frontier_a;
    int32_t* nxt_front = frontier_b;

    int32_t k = k_start;

    while (k <= k_last_val && rem_count > 0) {
        
        if (tid == 0) { counters[0] = 0; counters[1] = 0; }
        grid.sync();

        for (int idx = tid; idx < rem_count; idx += stride) {
            int v = cur_rem[idx];
            if (valid[v]) {
                if (core_numbers[v] < k) {
                    valid[v] = 0;
                    int pos = atomicAdd(&counters[1], 1);
                    cur_front[pos] = v;
                } else {
                    int pos = atomicAdd(&counters[0], 1);
                    nxt_rem[pos] = v;
                }
            }
        }
        grid.sync();

        int32_t new_rem = counters[0];
        int32_t fsize = counters[1];

        
        { int32_t* t = cur_rem; cur_rem = nxt_rem; nxt_rem = t; }
        rem_count = new_rem;

        if (fsize > 0) {
            
            int32_t k_minus_delta = k - delta;

            while (fsize > 0) {
                if (tid == 0) counters[2] = 0;
                grid.sync();

                for (int fidx = warp_id; fidx < fsize; fidx += total_warps) {
                    int v = cur_front[fidx];
                    int start = offsets[v];
                    int end = offsets[v + 1];
                    int degree = end - start;

                    for (int i = lane; i < degree; i += 32) {
                        int u = indices[start + i];
                        if (u != v && valid[u]) {
                            int old_val = atomicSub(&core_numbers[u], delta);
                            if (old_val >= k && old_val - delta < k) {
                                int was_zero = atomicCAS(&in_next[u], 0, 1);
                                if (was_zero == 0) {
                                    int pos = atomicAdd(&counters[2], 1);
                                    nxt_front[pos] = u;
                                }
                            }
                        }
                    }
                }
                grid.sync();

                int32_t next_fsize = counters[2];

                for (int idx = tid; idx < next_fsize; idx += stride) {
                    int v = nxt_front[idx];
                    valid[v] = 0;
                    in_next[v] = 0;
                    int cn = core_numbers[v];
                    if (cn < k_minus_delta) cn = k_minus_delta;
                    if (k_first > 0 && cn < k_first) cn = 0;
                    core_numbers[v] = cn;
                }
                grid.sync();

                { int32_t* t = cur_front; cur_front = nxt_front; nxt_front = t; }
                fsize = next_fsize;
            }

            k += delta;
            if (delta == 2 && (k & 1)) k++;
        } else {
            
            if (rem_count == 0) break;

            if (tid == 0) counters[3] = INT32_MAX;
            grid.sync();

            int32_t local_min = INT32_MAX;
            for (int idx = tid; idx < rem_count; idx += stride) {
                int32_t cn = core_numbers[cur_rem[idx]];
                if (cn < local_min) local_min = cn;
            }
            for (int d = 16; d >= 1; d >>= 1) {
                int32_t other = __shfl_xor_sync(0xffffffff, local_min, d);
                if (other < local_min) local_min = other;
            }
            if (lane == 0 && local_min < INT32_MAX) atomicMin(&counters[3], local_min);
            grid.sync();

            int32_t min_val = counters[3];
            if (min_val == INT32_MAX) break;

            int32_t new_k = k + delta;
            int32_t jump_k = min_val + delta;
            k = (new_k > jump_k) ? new_k : jump_k;
            if (delta == 2 && (k & 1)) k++;
        }
    }
}

}  

void core_number(const graph32_t& graph,
                 int32_t* core_numbers,
                 int degree_type,
                 std::size_t k_first,
                 std::size_t k_last) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    if (num_vertices == 0) return;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    bool is_inout = (degree_type == 2);
    int32_t degree_mul = is_inout ? 2 : 1;
    int32_t delta = is_inout ? 2 : 1;

    cache.ensure(num_vertices);

    cudaMemsetAsync(cache.in_next, 0, num_vertices * sizeof(int32_t), 0);

    int32_t k_first_i32 = (int32_t)k_first;
    init_kernel<<<(num_vertices + 255) / 256, 256, 0, 0>>>(
        offsets, indices, core_numbers, cache.valid,
        degree_mul, k_first_i32, num_vertices);

    
    int32_t h_zero = 0;
    int32_t* d_cnt = cache.counters;
    cudaMemcpyAsync(d_cnt, &h_zero, sizeof(int32_t), cudaMemcpyHostToDevice, 0);
    build_remaining_kernel<<<(num_vertices + 255) / 256, 256, 0, 0>>>(
        cache.valid, cache.rem_a, d_cnt, num_vertices);
    int32_t h_rem_count;
    cudaMemcpyAsync(&h_rem_count, d_cnt, sizeof(int32_t), cudaMemcpyDeviceToHost, 0);
    cudaStreamSynchronize(0);

    if (h_rem_count == 0) return;

    int32_t k_start = (k_first >= 2) ? (int32_t)k_first : 2;
    if (is_inout && (k_start % 2 == 1)) k_start++;
    int32_t k_last_val = (k_last > (std::size_t)INT32_MAX) ? INT32_MAX : (int32_t)k_last;

    
    int block_size = 256;
    int num_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm, process_all_levels_kernel, block_size, 0);

    int device;
    cudaGetDevice(&device);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);

    int grid_size = num_blocks_per_sm * num_sms;

    const int32_t* offsets_ptr = offsets;
    const int32_t* indices_ptr = indices;
    int32_t* core_numbers_ptr = core_numbers;
    int32_t* valid_ptr = cache.valid;
    int32_t* in_next_ptr = cache.in_next;
    int32_t* rem_a_ptr = cache.rem_a;
    int32_t* rem_b_ptr = cache.rem_b;
    int32_t* front_a_ptr = cache.front_a;
    int32_t* front_b_ptr = cache.front_b;
    int32_t* counters_ptr = cache.counters;

    void* args[] = {
        (void*)&offsets_ptr, (void*)&indices_ptr,
        (void*)&core_numbers_ptr, (void*)&valid_ptr, (void*)&in_next_ptr,
        (void*)&rem_a_ptr, (void*)&rem_b_ptr,
        (void*)&front_a_ptr, (void*)&front_b_ptr,
        (void*)&counters_ptr,
        (void*)&h_rem_count,
        (void*)&k_start, (void*)&delta, (void*)&k_first_i32, (void*)&k_last_val
    };

    cudaLaunchCooperativeKernel(
        (void*)process_all_levels_kernel,
        dim3(grid_size), dim3(block_size),
        args, 0, 0);

    cudaDeviceSynchronize();
}

}  
