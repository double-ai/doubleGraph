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
#include <cfloat>
#include <cstdint>

namespace aai {

namespace {




static constexpr size_t CUB_TEMP_BYTES = 8 * 1024 * 1024;

struct Cache : Cacheable {
    int32_t* filt_offsets = nullptr;
    int32_t* filt_indices = nullptr;
    int32_t* csr_offsets = nullptr;
    int32_t* csr_indices = nullptr;
    int32_t* counts = nullptr;
    float* hubs_alt = nullptr;
    float* accum = nullptr;
    void* cub_temp = nullptr;

    int64_t filt_offsets_cap = 0;
    int64_t csr_offsets_cap = 0;
    int64_t counts_cap = 0;
    int64_t filt_indices_cap = 0;
    int64_t csr_indices_cap = 0;
    int64_t hubs_alt_cap = 0;

    Cache() {
        cudaMalloc(&accum, 4 * sizeof(float));
        cudaMalloc(&cub_temp, CUB_TEMP_BYTES);
    }

    void ensure(int32_t num_vertices, int32_t num_edges) {
        int64_t nv1 = (int64_t)num_vertices + 1;
        int64_t ne = num_edges > 0 ? (int64_t)num_edges : 1;

        if (filt_offsets_cap < nv1) {
            if (filt_offsets) cudaFree(filt_offsets);
            cudaMalloc(&filt_offsets, nv1 * sizeof(int32_t));
            filt_offsets_cap = nv1;
        }
        if (csr_offsets_cap < nv1) {
            if (csr_offsets) cudaFree(csr_offsets);
            cudaMalloc(&csr_offsets, nv1 * sizeof(int32_t));
            csr_offsets_cap = nv1;
        }
        if (counts_cap < nv1) {
            if (counts) cudaFree(counts);
            cudaMalloc(&counts, nv1 * sizeof(int32_t));
            counts_cap = nv1;
        }
        if (filt_indices_cap < ne) {
            if (filt_indices) cudaFree(filt_indices);
            cudaMalloc(&filt_indices, ne * sizeof(int32_t));
            filt_indices_cap = ne;
        }
        if (csr_indices_cap < ne) {
            if (csr_indices) cudaFree(csr_indices);
            cudaMalloc(&csr_indices, ne * sizeof(int32_t));
            csr_indices_cap = ne;
        }
        if (hubs_alt_cap < num_vertices) {
            if (hubs_alt) cudaFree(hubs_alt);
            cudaMalloc(&hubs_alt, (int64_t)num_vertices * sizeof(float));
            hubs_alt_cap = num_vertices;
        }
    }

    ~Cache() override {
        if (filt_offsets) cudaFree(filt_offsets);
        if (filt_indices) cudaFree(filt_indices);
        if (csr_offsets) cudaFree(csr_offsets);
        if (csr_indices) cudaFree(csr_indices);
        if (counts) cudaFree(counts);
        if (hubs_alt) cudaFree(hubs_alt);
        if (accum) cudaFree(accum);
        if (cub_temp) cudaFree(cub_temp);
    }
};




struct MaxOp { __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); } };

__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    if (val <= 0.0f) return;
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) return;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(val));
    } while (assumed != old);
}




__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets, const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ counts, int32_t num_vertices)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        int start = offsets[v], end = offsets[v + 1], count = 0;
        for (int e = start; e < end; e++)
            if ((edge_mask[e >> 5] >> (e & 31)) & 1) count++;
        counts[v] = count;
    }
}

__global__ void scatter_active_edges_kernel(
    const int32_t* __restrict__ old_offsets, const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets, int32_t* __restrict__ new_indices,
    int32_t num_vertices)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        int old_start = old_offsets[v], old_end = old_offsets[v + 1], new_pos = new_offsets[v];
        for (int e = old_start; e < old_end; e++)
            if ((edge_mask[e >> 5] >> (e & 31)) & 1)
                new_indices[new_pos++] = old_indices[e];
    }
}




__global__ void count_per_row_kernel(
    const int32_t* __restrict__ indices, int32_t num_edges, int32_t* __restrict__ row_counts)
{
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < num_edges; e += blockDim.x * gridDim.x)
        atomicAdd(&row_counts[indices[e]], 1);
}

__global__ void csc_to_csr_scatter_kernel(
    const int32_t* __restrict__ csc_offsets, const int32_t* __restrict__ csc_indices,
    int32_t num_vertices, int32_t* __restrict__ csr_indices, int32_t* __restrict__ write_pos)
{
    for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < num_vertices; col += blockDim.x * gridDim.x) {
        int start = csc_offsets[col], end = csc_offsets[col + 1];
        for (int e = start; e < end; e++) {
            int row = csc_indices[e];
            csr_indices[atomicAdd(&write_pos[row], 1)] = col;
        }
    }
}




__global__ void spmv_thread_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ x, float* __restrict__ y, int32_t num_rows)
{
    for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < num_rows; row += blockDim.x * gridDim.x) {
        int start = offsets[row], end = offsets[row + 1];
        float sum = 0.0f;
        for (int e = start; e < end; e++) sum += x[indices[e]];
        y[row] = sum;
    }
}

__global__ void spmv_warp_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ x, float* __restrict__ y, int32_t num_rows)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    int warp_id = tid >> 5, lane = tid & 31;
    int total_warps = total_threads >> 5;
    for (int row = warp_id; row < num_rows; row += total_warps) {
        int start = offsets[row], end = offsets[row + 1];
        float sum = 0.0f;
        for (int e = start + lane; e < end; e += 32) sum += x[indices[e]];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) y[row] = sum;
    }
}




__global__ void fused_max_kernel(
    const float* __restrict__ auth, const float* __restrict__ hub,
    float* __restrict__ accum, int32_t n)
{
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage t1, t2;
    float la = 0.0f, lh = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        la = fmaxf(la, auth[i]); lh = fmaxf(lh, hub[i]);
    }
    la = BR(t1).Reduce(la, MaxOp()); lh = BR(t2).Reduce(lh, MaxOp());
    if (threadIdx.x == 0) { atomicMaxFloat(&accum[0], la); atomicMaxFloat(&accum[1], lh); }
}

__global__ void fused_normalize_diff_kernel(
    float* __restrict__ auth, float* __restrict__ hub,
    const float* __restrict__ old_hub, float* __restrict__ accum, int32_t n)
{
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage temp;
    float am = accum[0], hm = accum[1];
    float ai = (am > 0.0f) ? (1.0f / am) : 1.0f;
    float hi = (hm > 0.0f) ? (1.0f / hm) : 1.0f;
    float ld = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float a = auth[i] * ai, h = hub[i] * hi;
        auth[i] = a; hub[i] = h;
        ld += fabsf(h - old_hub[i]);
    }
    ld = BR(temp).Sum(ld);
    if (threadIdx.x == 0) atomicAdd(&accum[2], ld);
}




__global__ void init_uniform_kernel(float* data, float val, int32_t n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) data[i] = val;
}

__global__ void fused_abs_sum_kernel(const float* __restrict__ data, float* __restrict__ out, int32_t n) {
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage temp;
    float s = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) s += fabsf(data[i]);
    s = BR(temp).Sum(s);
    if (threadIdx.x == 0) atomicAdd(out, s);
}

__global__ void l1_normalize_kernel(float* data, const float* d_sum, int32_t n) {
    float s = *d_sum;
    if (s <= 0.0f) return;
    float inv = 1.0f / s;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) data[i] *= inv;
}

}  




HitsResult hits_mask(const graph32_t& graph,
                     float* hubs,
                     float* authorities,
                     float epsilon,
                     std::size_t max_iterations,
                     bool has_initial_hubs_guess,
                     bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    if (num_vertices == 0) {
        return HitsResult{max_iterations, false, FLT_MAX};
    }

    cache.ensure(num_vertices, num_edges);

    int32_t* d_filt_offsets = cache.filt_offsets;
    int32_t* d_filt_indices = cache.filt_indices;
    int32_t* d_csr_offsets = cache.csr_offsets;
    int32_t* d_csr_indices = cache.csr_indices;
    int32_t* d_counts = cache.counts;
    float* d_hubs_alt = cache.hubs_alt;
    float* d_accum = cache.accum;
    void* d_cub_temp = cache.cub_temp;
    size_t cub_temp_size = CUB_TEMP_BYTES;

    const int BLOCK = 256;
    const int MAX_BLOCKS = 108 * 8;
    auto grid_for = [&](int64_t n) -> int { int g = (int)((n + BLOCK - 1) / BLOCK); return g < MAX_BLOCKS ? g : MAX_BLOCKS; };
    float tolerance = (float)num_vertices * epsilon;
    int v_grid = grid_for(num_vertices);

    
    cudaMemsetAsync(d_counts + num_vertices, 0, sizeof(int32_t));
    count_active_edges_kernel<<<v_grid, BLOCK>>>(d_offsets, d_edge_mask, d_counts, num_vertices);

    size_t needed = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, needed, d_counts, d_filt_offsets, num_vertices + 1);
    needed = needed < cub_temp_size ? needed : cub_temp_size;
    cub::DeviceScan::ExclusiveSum(d_cub_temp, needed, d_counts, d_filt_offsets, num_vertices + 1);

    int32_t num_active_edges;
    cudaMemcpy(&num_active_edges, d_filt_offsets + num_vertices, sizeof(int32_t), cudaMemcpyDeviceToHost);

    if (num_active_edges > 0)
        scatter_active_edges_kernel<<<v_grid, BLOCK>>>(d_offsets, d_indices, d_edge_mask, d_filt_offsets, d_filt_indices, num_vertices);

    
    cudaMemsetAsync(d_counts, 0, ((int64_t)num_vertices + 1) * sizeof(int32_t));
    if (num_active_edges > 0)
        count_per_row_kernel<<<grid_for(num_active_edges), BLOCK>>>(d_filt_indices, num_active_edges, d_counts);

    needed = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, needed, d_counts, d_csr_offsets, num_vertices + 1);
    needed = needed < cub_temp_size ? needed : cub_temp_size;
    cub::DeviceScan::ExclusiveSum(d_cub_temp, needed, d_counts, d_csr_offsets, num_vertices + 1);

    
    cudaMemcpyAsync(d_counts, d_csr_offsets, (int64_t)num_vertices * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    if (num_active_edges > 0)
        csc_to_csr_scatter_kernel<<<v_grid, BLOCK>>>(d_filt_offsets, d_filt_indices, num_vertices, d_csr_indices, d_counts);

    
    if (has_initial_hubs_guess) {
        cudaMemsetAsync(d_accum, 0, sizeof(float));
        fused_abs_sum_kernel<<<v_grid, BLOCK>>>(hubs, d_accum, num_vertices);
        l1_normalize_kernel<<<v_grid, BLOCK>>>(hubs, d_accum, num_vertices);
    } else {
        init_uniform_kernel<<<v_grid, BLOCK>>>(hubs, 1.0f / (float)num_vertices, num_vertices);
    }

    
    bool use_warp = (num_vertices > 0 && (num_active_edges / num_vertices) >= 32);
    int spmv_grid;
    if (use_warp) {
        spmv_grid = grid_for((int64_t)num_vertices * 32 / BLOCK * BLOCK); 
        int warps_per_block = BLOCK / 32;
        spmv_grid = (int)((num_vertices + warps_per_block - 1) / warps_per_block);
        if (spmv_grid > MAX_BLOCKS) spmv_grid = MAX_BLOCKS;
    } else {
        spmv_grid = v_grid;
    }
    int reduce_grid = v_grid;

    float* hub_buffers[2] = {hubs, d_hubs_alt};
    int current_hub = 0;
    float diff_sum = FLT_MAX;
    size_t iter = 0;

    for (size_t it = 0; it < max_iterations; it++) {
        float* prev_hubs = hub_buffers[current_hub];
        float* curr_hubs = hub_buffers[1 - current_hub];

        if (use_warp) {
            spmv_warp_kernel<<<spmv_grid, BLOCK>>>(d_filt_offsets, d_filt_indices, prev_hubs, authorities, num_vertices);
            spmv_warp_kernel<<<spmv_grid, BLOCK>>>(d_csr_offsets, d_csr_indices, authorities, curr_hubs, num_vertices);
        } else {
            spmv_thread_kernel<<<spmv_grid, BLOCK>>>(d_filt_offsets, d_filt_indices, prev_hubs, authorities, num_vertices);
            spmv_thread_kernel<<<spmv_grid, BLOCK>>>(d_csr_offsets, d_csr_indices, authorities, curr_hubs, num_vertices);
        }

        cudaMemsetAsync(d_accum, 0, 3 * sizeof(float));
        fused_max_kernel<<<reduce_grid, BLOCK>>>(authorities, curr_hubs, d_accum, num_vertices);
        fused_normalize_diff_kernel<<<reduce_grid, BLOCK>>>(authorities, curr_hubs, prev_hubs, d_accum, num_vertices);

        current_hub = 1 - current_hub;
        iter = it + 1;

        cudaMemcpy(&diff_sum, &d_accum[2], sizeof(float), cudaMemcpyDeviceToHost);
        if (diff_sum < tolerance) break;
    }

    
    float* final_hubs = hub_buffers[current_hub];
    if (normalize) {
        cudaMemsetAsync(d_accum, 0, sizeof(float));
        fused_abs_sum_kernel<<<reduce_grid, BLOCK>>>(final_hubs, d_accum, num_vertices);
        l1_normalize_kernel<<<reduce_grid, BLOCK>>>(final_hubs, d_accum, num_vertices);

        cudaMemsetAsync(d_accum, 0, sizeof(float));
        fused_abs_sum_kernel<<<reduce_grid, BLOCK>>>(authorities, d_accum, num_vertices);
        l1_normalize_kernel<<<reduce_grid, BLOCK>>>(authorities, d_accum, num_vertices);
    }

    if (final_hubs != hubs)
        cudaMemcpyAsync(hubs, final_hubs, (int64_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaDeviceSynchronize();

    return HitsResult{iter, iter < max_iterations, diff_sum};
}

}  
