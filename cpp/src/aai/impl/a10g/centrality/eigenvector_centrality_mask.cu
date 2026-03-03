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
#include <cmath>
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* h_diff = nullptr;

    int32_t* active_counts = nullptr;
    int32_t* new_offsets = nullptr;
    void* scan_temp = nullptr;
    int32_t* new_indices = nullptr;
    float* scratch = nullptr;
    float* x_buf0 = nullptr;
    float* x_buf1 = nullptr;

    int64_t active_counts_cap = 0;
    int64_t new_offsets_cap = 0;
    size_t scan_temp_cap = 0;
    int64_t new_indices_cap = 0;
    int64_t scratch_cap = 0;
    int64_t x_buf0_cap = 0;
    int64_t x_buf1_cap = 0;

    Cache() {
        cudaMallocHost(&h_diff, sizeof(float));
    }

    ~Cache() override {
        if (h_diff) cudaFreeHost(h_diff);
        if (active_counts) cudaFree(active_counts);
        if (new_offsets) cudaFree(new_offsets);
        if (scan_temp) cudaFree(scan_temp);
        if (new_indices) cudaFree(new_indices);
        if (scratch) cudaFree(scratch);
        if (x_buf0) cudaFree(x_buf0);
        if (x_buf1) cudaFree(x_buf1);
    }

    void ensure(int32_t nv, int32_t ne, size_t stb, int32_t ss) {
        if (active_counts_cap < nv) {
            if (active_counts) cudaFree(active_counts);
            cudaMalloc(&active_counts, (size_t)nv * sizeof(int32_t));
            active_counts_cap = nv;
        }
        if (new_offsets_cap < (int64_t)(nv + 1)) {
            if (new_offsets) cudaFree(new_offsets);
            cudaMalloc(&new_offsets, (size_t)(nv + 1) * sizeof(int32_t));
            new_offsets_cap = nv + 1;
        }
        if (scan_temp_cap < stb) {
            if (scan_temp) cudaFree(scan_temp);
            cudaMalloc(&scan_temp, stb);
            scan_temp_cap = stb;
        }
        if (new_indices_cap < ne) {
            if (new_indices) cudaFree(new_indices);
            cudaMalloc(&new_indices, (size_t)ne * sizeof(int32_t));
            new_indices_cap = ne;
        }
        if (scratch_cap < ss) {
            if (scratch) cudaFree(scratch);
            cudaMalloc(&scratch, (size_t)ss * sizeof(float));
            scratch_cap = ss;
        }
        if (x_buf0_cap < nv) {
            if (x_buf0) cudaFree(x_buf0);
            cudaMalloc(&x_buf0, (size_t)nv * sizeof(float));
            x_buf0_cap = nv;
        }
        if (x_buf1_cap < nv) {
            if (x_buf1) cudaFree(x_buf1);
            cudaMalloc(&x_buf1, (size_t)nv * sizeof(float));
            x_buf1_cap = nv;
        }
    }
};



__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_counts,
    int32_t num_vertices)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += gridDim.x * blockDim.x) {
        int32_t start = offsets[v], end = offsets[v + 1], count = 0;
        for (int32_t e = start; e < end; e++)
            if ((edge_mask[e >> 5] >> (e & 31)) & 1) count++;
        active_counts[v] = count;
    }
}

__global__ void compact_edges_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t num_vertices)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += gridDim.x * blockDim.x) {
        int32_t old_start = old_offsets[v], old_end = old_offsets[v + 1];
        int32_t write_pos = new_offsets[v];
        for (int32_t e = old_start; e < old_end; e++)
            if ((edge_mask[e >> 5] >> (e & 31)) & 1)
                new_indices[write_pos++] = old_indices[e];
    }
}

__global__ void set_last_offset_kernel(
    const int32_t* __restrict__ scan_out, const int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets_out, int32_t num_vertices)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        offsets_out[num_vertices] = scan_out[num_vertices - 1] + counts[num_vertices - 1];
}

__global__ void init_degree_kernel(
    const int32_t* __restrict__ offsets,
    float* __restrict__ c,
    float* __restrict__ partials,
    float* __restrict__ norm_sq_out,
    unsigned int* __restrict__ retire_counter,
    int32_t n)
{
    float thread_norm_sq = 0.0f;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += gridDim.x * blockDim.x) {
        float deg = (float)(offsets[v+1] - offsets[v]) + 1.0f;
        float val = sqrtf(deg);
        c[v] = val;
        thread_norm_sq += val * val;
    }

    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage ts;
    float bs = BR(ts).Sum(thread_norm_sq);
    if (threadIdx.x == 0) partials[blockIdx.x] = bs;
    __threadfence();
    __shared__ bool am_last;
    if (threadIdx.x == 0) am_last = (atomicAdd(retire_counter, 1) == gridDim.x - 1);
    __syncthreads();
    if (am_last) {
        float s = 0.0f;
        for (int i = threadIdx.x; i < (int)gridDim.x; i += blockDim.x) s += partials[i];
        s = BR(ts).Sum(s);
        if (threadIdx.x == 0) { norm_sq_out[0] = s; *retire_counter = 0; }
    }
}

__global__ void normalize_kernel(float* __restrict__ x, const float* __restrict__ norm_sq_ptr, int32_t n)
{
    float inv_norm = rsqrtf(norm_sq_ptr[0]);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        x[i] *= inv_norm;
}

__global__ void init_uniform_kernel(float* __restrict__ c, int32_t n)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x)
        c[idx] = 1.0f / (float)n;
}

__global__ void spmv_warp_l2_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ x_old, float* __restrict__ x_new,
    float* __restrict__ partials, float* __restrict__ norm_sq_out,
    unsigned int* __restrict__ retire_counter, int32_t num_vertices)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int total_warps = (gridDim.x * blockDim.x) >> 5;
    float thread_norm_sq = 0.0f;

    for (int v = warp_id; v < num_vertices; v += total_warps) {
        int32_t row_start = __ldg(&offsets[v]);
        int32_t row_end = __ldg(&offsets[v + 1]);
        float sum = 0.0f;
        for (int32_t e = row_start + lane; e < row_end; e += 32)
            sum += __ldg(&x_old[__ldg(&indices[e])]);
        sum += __shfl_down_sync(0xffffffff, sum, 16);
        sum += __shfl_down_sync(0xffffffff, sum, 8);
        sum += __shfl_down_sync(0xffffffff, sum, 4);
        sum += __shfl_down_sync(0xffffffff, sum, 2);
        sum += __shfl_down_sync(0xffffffff, sum, 1);
        if (lane == 0) {
            float val = sum + __ldg(&x_old[v]);
            x_new[v] = val;
            thread_norm_sq += val * val;
        }
    }

    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage ts;
    float bs = BR(ts).Sum(thread_norm_sq);
    if (threadIdx.x == 0) partials[blockIdx.x] = bs;
    __threadfence();
    __shared__ bool am_last;
    if (threadIdx.x == 0) am_last = (atomicAdd(retire_counter, 1) == gridDim.x - 1);
    __syncthreads();
    if (am_last) {
        float s = 0.0f;
        for (int i = threadIdx.x; i < (int)gridDim.x; i += blockDim.x) s += partials[i];
        s = BR(ts).Sum(s);
        if (threadIdx.x == 0) { norm_sq_out[0] = s; *retire_counter = 0; }
    }
}

__global__ void spmv_thread_l2_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ x_old, float* __restrict__ x_new,
    float* __restrict__ partials, float* __restrict__ norm_sq_out,
    unsigned int* __restrict__ retire_counter, int32_t num_vertices)
{
    float thread_norm_sq = 0.0f;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += gridDim.x * blockDim.x) {
        int32_t row_start = __ldg(&offsets[v]);
        int32_t row_end = __ldg(&offsets[v + 1]);
        float sum = __ldg(&x_old[v]);
        for (int32_t e = row_start; e < row_end; e++)
            sum += __ldg(&x_old[__ldg(&indices[e])]);
        x_new[v] = sum;
        thread_norm_sq += sum * sum;
    }
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage ts;
    float bs = BR(ts).Sum(thread_norm_sq);
    if (threadIdx.x == 0) partials[blockIdx.x] = bs;
    __threadfence();
    __shared__ bool am_last;
    if (threadIdx.x == 0) am_last = (atomicAdd(retire_counter, 1) == gridDim.x - 1);
    __syncthreads();
    if (am_last) {
        float s = 0.0f;
        for (int i = threadIdx.x; i < (int)gridDim.x; i += blockDim.x) s += partials[i];
        s = BR(ts).Sum(s);
        if (threadIdx.x == 0) { norm_sq_out[0] = s; *retire_counter = 0; }
    }
}

__global__ void normalize_l1diff_kernel(
    float* __restrict__ x_new, const float* __restrict__ x_old,
    const float* __restrict__ norm_sq_ptr, float* __restrict__ partials,
    float* __restrict__ diff_out, unsigned int* __restrict__ retire_counter,
    int32_t num_vertices)
{
    float inv_norm = rsqrtf(norm_sq_ptr[0]);
    float thread_diff = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vertices; i += gridDim.x * blockDim.x) {
        float normalized = x_new[i] * inv_norm;
        x_new[i] = normalized;
        thread_diff += fabsf(normalized - __ldg(&x_old[i]));
    }
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage ts;
    float bs = BR(ts).Sum(thread_diff);
    if (threadIdx.x == 0) partials[blockIdx.x] = bs;
    __threadfence();
    __shared__ bool am_last;
    if (threadIdx.x == 0) am_last = (atomicAdd(retire_counter, 1) == gridDim.x - 1);
    __syncthreads();
    if (am_last) {
        float s = 0.0f;
        for (int i = threadIdx.x; i < (int)gridDim.x; i += blockDim.x) s += partials[i];
        s = BR(ts).Sum(s);
        if (threadIdx.x == 0) { diff_out[0] = s; *retire_counter = 0; }
    }
}

}  

eigenvector_centrality_result_t eigenvector_centrality_mask(const graph32_t& graph,
                                  float* centralities,
                                  float epsilon,
                                  std::size_t max_iterations,
                                  const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;

    float avg_degree = (num_vertices > 0) ? ((float)num_edges * 0.7f / (float)num_vertices) : 0.0f;
    bool use_warp = (avg_degree >= 4.0f);

    
    int compact_grid = std::min((num_vertices + 255) / 256, 4096);

    
    size_t scan_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes,
                                  (int32_t*)nullptr, (int32_t*)nullptr, num_vertices);

    
    int spmv_grid;
    if (use_warp) {
        spmv_grid = (num_vertices + 7) / 8;
        spmv_grid = std::max(80, std::min(spmv_grid, 2048));
    } else {
        spmv_grid = (num_vertices + 255) / 256;
        spmv_grid = std::max(80, std::min(spmv_grid, 4096));
    }
    int norm_grid = (num_vertices + 255) / 256;
    norm_grid = std::max(80, std::min(norm_grid, 2048));
    int max_grid = std::max(spmv_grid, norm_grid);
    int total_scratch = max_grid + 4;

    
    cache.ensure(num_vertices, num_edges, scan_temp_bytes, total_scratch);

    
    count_active_edges_kernel<<<compact_grid, 256>>>(d_offsets, d_edge_mask,
                                                      cache.active_counts, num_vertices);

    cub::DeviceScan::ExclusiveSum(cache.scan_temp, scan_temp_bytes,
                                  cache.active_counts, cache.new_offsets, num_vertices);

    set_last_offset_kernel<<<1, 1>>>(cache.new_offsets, cache.active_counts,
                                     cache.new_offsets, num_vertices);

    compact_edges_kernel<<<compact_grid, 256>>>(d_offsets, d_indices, d_edge_mask,
                                                 cache.new_offsets, cache.new_indices, num_vertices);

    
    float* d_scratch = cache.scratch;
    float* d_partials = d_scratch;
    float* d_norm_sq = d_scratch + max_grid;
    float* d_diff = d_scratch + max_grid + 1;
    unsigned int* d_retire1 = reinterpret_cast<unsigned int*>(d_scratch + max_grid + 2);
    unsigned int* d_retire2 = d_retire1 + 1;
    cudaMemsetAsync(d_retire1, 0, 2 * sizeof(unsigned int), 0);

    
    float* d_x_old = cache.x_buf0;
    float* d_x_new = cache.x_buf1;

    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(d_x_old, initial_centralities,
                        (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, 0);
    } else {
        init_degree_kernel<<<norm_grid, 256>>>(cache.new_offsets, d_x_old,
                                                d_partials, d_norm_sq, d_retire1, num_vertices);
        normalize_kernel<<<norm_grid, 256>>>(d_x_old, d_norm_sq, num_vertices);
    }

    
    float threshold = num_vertices * epsilon;
    std::size_t iterations = 0;
    bool converged = false;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        if (use_warp) {
            spmv_warp_l2_kernel<<<spmv_grid, 256>>>(cache.new_offsets, cache.new_indices,
                d_x_old, d_x_new, d_partials, d_norm_sq, d_retire1, num_vertices);
        } else {
            spmv_thread_l2_kernel<<<spmv_grid, 256>>>(cache.new_offsets, cache.new_indices,
                d_x_old, d_x_new, d_partials, d_norm_sq, d_retire1, num_vertices);
        }

        normalize_l1diff_kernel<<<norm_grid, 256>>>(d_x_new, d_x_old, d_norm_sq,
            d_partials, d_diff, d_retire2, num_vertices);

        iterations++;

        cudaMemcpyAsync(cache.h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost, 0);
        cudaStreamSynchronize(0);

        if (*cache.h_diff < threshold) {
            converged = true;
            std::swap(d_x_old, d_x_new);
            break;
        }
        std::swap(d_x_old, d_x_new);
    }

    
    cudaMemcpyAsync(centralities, d_x_old,
                    (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, 0);

    return {iterations, converged};
}

}  
