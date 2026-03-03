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
    int32_t* remaining_a = nullptr;
    int32_t* remaining_b = nullptr;
    int32_t* frontier_a = nullptr;
    int32_t* frontier_b = nullptr;
    int32_t* d_counters = nullptr;
    int32_t* h_counters = nullptr;

    int64_t valid_cap = 0;
    int64_t rem_a_cap = 0;
    int64_t rem_b_cap = 0;
    int64_t front_a_cap = 0;
    int64_t front_b_cap = 0;
    bool counters_init = false;

    void ensure(int64_t n) {
        if (valid_cap < n) {
            if (valid) cudaFree(valid);
            cudaMalloc(&valid, n * sizeof(int32_t));
            valid_cap = n;
        }
        if (rem_a_cap < n) {
            if (remaining_a) cudaFree(remaining_a);
            cudaMalloc(&remaining_a, n * sizeof(int32_t));
            rem_a_cap = n;
        }
        if (rem_b_cap < n) {
            if (remaining_b) cudaFree(remaining_b);
            cudaMalloc(&remaining_b, n * sizeof(int32_t));
            rem_b_cap = n;
        }
        if (front_a_cap < n) {
            if (frontier_a) cudaFree(frontier_a);
            cudaMalloc(&frontier_a, n * sizeof(int32_t));
            front_a_cap = n;
        }
        if (front_b_cap < n) {
            if (frontier_b) cudaFree(frontier_b);
            cudaMalloc(&frontier_b, n * sizeof(int32_t));
            front_b_cap = n;
        }
        if (!counters_init) {
            cudaMalloc(&d_counters, 4 * sizeof(int32_t));
            cudaHostAlloc(&h_counters, 4 * sizeof(int32_t), cudaHostAllocDefault);
            counters_init = true;
        }
    }

    ~Cache() override {
        if (valid) cudaFree(valid);
        if (remaining_a) cudaFree(remaining_a);
        if (remaining_b) cudaFree(remaining_b);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (d_counters) cudaFree(d_counters);
        if (h_counters) cudaFreeHost(h_counters);
    }
};

__global__ void init_core_numbers_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ valid,
    int32_t num_vertices,
    int32_t multiplier
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int start = __ldg(&offsets[v]);
        int end = __ldg(&offsets[v + 1]);
        int deg = end - start;
        for (int i = start; i < end; i++) {
            if (__ldg(&indices[i]) == v) deg--;
        }
        core_numbers[v] = deg * multiplier;
        valid[v] = (deg > 0) ? 1 : 0;
    }
}

__global__ void apply_k_first_kernel(
    int32_t* __restrict__ core_numbers,
    int32_t num_vertices,
    int32_t k_first
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int cn = core_numbers[v];
        if (cn > 0 && cn < k_first) core_numbers[v] = 0;
    }
}

__global__ void build_remaining_kernel(
    const int32_t* __restrict__ valid,
    int32_t* __restrict__ remaining,
    int32_t* __restrict__ remaining_count,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices && valid[v]) {
        int pos = atomicAdd(remaining_count, 1);
        remaining[pos] = v;
    }
}

__device__ __forceinline__ void process_edges_device(
    cg::grid_group& grid,
    const int32_t* __restrict__ fcur, int fc,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ core_numbers,
    const int32_t* __restrict__ valid,
    int32_t* __restrict__ fnxt,
    int32_t* __restrict__ nxt_cnt_ptr,
    int32_t k, int32_t delta
) {
    const int tid = grid.thread_rank();
    const int gs = grid.size();
    const int num_warps = gs >> 5;
    const int warp_idx = tid >> 5;
    const int lane = tid & 31;

    for (int w = warp_idx; w < fc; w += num_warps) {
        int v = fcur[w];
        int start = __ldg(&offsets[v]);
        int end = __ldg(&offsets[v + 1]);
        for (int i = start + lane; i < end; i += 32) {
            int u = __ldg(&indices[i]);
            if (u != v && valid[u]) {
                int old_cn = atomicSub(&core_numbers[u], delta);
                if (old_cn >= k && (old_cn - delta) < k) {
                    fnxt[atomicAdd(nxt_cnt_ptr, 1)] = u;
                }
            }
        }
    }
}

__launch_bounds__(1024, 1)
__global__ void batched_core_number_coop(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ valid,
    int32_t* __restrict__ rem_a,
    int32_t* __restrict__ rem_b,
    int32_t* __restrict__ front_a,
    int32_t* __restrict__ front_b,
    int32_t* __restrict__ counters,
    int32_t rem_count_init,
    int32_t k_start,
    int32_t k_last_i32,
    int32_t delta,
    int32_t k_first
) {
    cg::grid_group grid = cg::this_grid();
    const int tid = grid.thread_rank();
    const int gs = grid.size();

    int32_t* rem_in = rem_a;
    int32_t* rem_out = rem_b;
    int32_t* fronts[2] = {front_a, front_b};

    int remaining_count = rem_count_init;
    int32_t k = k_start;
    int buf_cur = 0;

    while (k <= k_last_i32 && remaining_count > 0) {
        if (tid == 0) {
            counters[0] = 0;
            counters[1 + buf_cur] = 0;
        }
        grid.sync();

        int32_t* fcur = fronts[buf_cur];
        int32_t* fcnt_cur = &counters[1 + buf_cur];

        for (int i = tid; i < remaining_count; i += gs) {
            int v = rem_in[i];
            if (!valid[v]) continue;
            if (core_numbers[v] < k) {
                fcur[atomicAdd(fcnt_cur, 1)] = v;
            } else {
                rem_out[atomicAdd(&counters[0], 1)] = v;
            }
        }
        grid.sync();

        remaining_count = counters[0];
        int fc = *fcnt_cur;

        { int32_t* t = rem_in; rem_in = rem_out; rem_out = t; }

        if (fc == 0) {
            if (tid == 0) counters[3] = INT32_MAX;
            grid.sync();

            int local_min = INT32_MAX;
            for (int i = tid; i < remaining_count; i += gs) {
                int cn = core_numbers[rem_in[i]];
                if (cn < local_min) local_min = cn;
            }
            for (int off = 16; off > 0; off >>= 1) {
                int o = __shfl_down_sync(0xffffffff, local_min, off);
                if (o < local_min) local_min = o;
            }
            if ((tid & 31) == 0 && local_min < INT32_MAX)
                atomicMin(&counters[3], local_min);
            grid.sync();

            int32_t mn = counters[3];
            int32_t nk = mn + delta;
            if (delta == 2 && (nk & 1)) nk++;
            k = (nk > k + delta) ? nk : k + delta;
            continue;
        }

        int buf_nxt = 1 - buf_cur;
        int32_t* fnxt = fronts[buf_nxt];
        int32_t* fcnt_nxt = &counters[1 + buf_nxt];

        if (tid == 0) *fcnt_nxt = 0;
        for (int i = tid; i < fc; i += gs) valid[fcur[i]] = 0;
        grid.sync();

        process_edges_device(grid, fcur, fc, offsets, indices,
                           core_numbers, valid, fnxt, fcnt_nxt, k, delta);
        grid.sync();

        int nc = *fcnt_nxt;
        int32_t kmd = (k > delta) ? (k - delta) : 0;

        while (nc > 0) {
            int32_t* fcnt_newnxt = fcnt_cur;
            if (tid == 0) *fcnt_newnxt = 0;
            for (int i = tid; i < nc; i += gs) {
                int v = fnxt[i];
                int cn = core_numbers[v];
                if (cn < kmd) cn = kmd;
                if (cn < k_first) cn = 0;
                core_numbers[v] = cn;
                valid[v] = 0;
            }
            grid.sync();

            { int32_t* t = fcur; fcur = fnxt; fnxt = t; }
            { int32_t* t2 = fcnt_cur; fcnt_cur = fcnt_nxt; fcnt_nxt = t2; }
            fc = nc;

            process_edges_device(grid, fcur, fc, offsets, indices,
                               core_numbers, valid, fnxt, fcnt_nxt, k, delta);
            grid.sync();

            nc = *fcnt_nxt;
        }

        buf_cur = (fcur == front_a) ? 1 : 0;

        k += delta;
    }

    if (tid == 0) counters[0] = remaining_count;
}

}  

void core_number_seg(const graph32_t& graph,
                     int32_t* core_numbers,
                     int degree_type,
                     std::size_t k_first,
                     std::size_t k_last) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    bool is_symmetric = graph.is_symmetric;

    if (num_vertices == 0) return;

    cache.ensure(num_vertices);

    int32_t* valid = cache.valid;
    int32_t* remaining_a = cache.remaining_a;
    int32_t* remaining_b = cache.remaining_b;
    int32_t* frontier_a = cache.frontier_a;
    int32_t* frontier_b = cache.frontier_b;
    int32_t* d_counters = cache.d_counters;
    int32_t* h_counters = cache.h_counters;

    int32_t k_first_i32 = (k_first > (std::size_t)INT32_MAX) ? INT32_MAX : (int32_t)k_first;
    int32_t k_last_i32 = (k_last > (std::size_t)INT32_MAX) ? INT32_MAX : (int32_t)k_last;

    int32_t multiplier = 1;
    int32_t delta = 1;
    if (is_symmetric && degree_type == 2) {
        multiplier = 2;
        delta = 2;
    }

    const int BLOCK = 1024;

    
    {
        int grid = (num_vertices + BLOCK - 1) / BLOCK;
        init_core_numbers_kernel<<<grid, BLOCK>>>(
            offsets, indices, core_numbers, valid, num_vertices, multiplier);
    }

    
    cudaMemsetAsync(d_counters, 0, 4 * sizeof(int32_t));
    {
        int grid = (num_vertices + BLOCK - 1) / BLOCK;
        build_remaining_kernel<<<grid, BLOCK>>>(
            valid, remaining_a, &d_counters[0], num_vertices);
    }

    
    if (k_first > 1) {
        int grid = (num_vertices + BLOCK - 1) / BLOCK;
        apply_k_first_kernel<<<grid, BLOCK>>>(core_numbers, num_vertices, k_first_i32);
    }

    cudaMemcpyAsync(h_counters, d_counters, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);

    int32_t remaining_count = h_counters[0];
    if (remaining_count == 0) return;

    int coop_block = 1024;
    int nbps = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &nbps, batched_core_number_coop, coop_block, 0);
    int device = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int coop_grid = nbps * prop.multiProcessorCount;
    if (coop_grid <= 0) coop_grid = 1;

    int32_t k_start = (k_first_i32 >= 2) ? k_first_i32 : 2;
    if (delta == 2 && (k_start % 2 == 1)) k_start++;

    void* args[] = {
        (void*)&offsets, (void*)&indices,
        (void*)&core_numbers, (void*)&valid,
        (void*)&remaining_a, (void*)&remaining_b,
        (void*)&frontier_a, (void*)&frontier_b,
        (void*)&d_counters,
        (void*)&remaining_count,
        (void*)&k_start, (void*)&k_last_i32,
        (void*)&delta, (void*)&k_first_i32
    };

    cudaLaunchCooperativeKernel(
        (void*)batched_core_number_coop,
        dim3(coop_grid), dim3(coop_block),
        args, 0, 0);

    cudaDeviceSynchronize();
}

}  
