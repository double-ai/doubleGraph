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
#include <cstddef>
#include <algorithm>
#include <limits>

namespace aai {

namespace {





__global__ void spmv_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t N)
{
    int32_t warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;
    int32_t total_warps = (blockDim.x * gridDim.x) >> 5;

    for (int32_t row = warp_global; row < N; row += total_warps) {
        int32_t start = __ldg(&offsets[row]);
        int32_t end = __ldg(&offsets[row + 1]);
        double sum = 0.0;
        for (int32_t j = start + lane; j < end; j += 32) {
            sum += x[__ldg(&indices[j])];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (lane == 0) {
            y[row] = sum;
        }
    }
}

__global__ void spmv_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t N)
{
    for (int32_t row = blockIdx.x * blockDim.x + threadIdx.x; row < N;
         row += blockDim.x * gridDim.x) {
        int32_t start = __ldg(&offsets[row]);
        int32_t end = __ldg(&offsets[row + 1]);
        double sum = 0.0;
        for (int32_t j = start; j < end; j++) {
            sum += x[__ldg(&indices[j])];
        }
        y[row] = sum;
    }
}





__global__ void count_degrees_kernel(
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ counts,
    int32_t E)
{
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < E;
         i += blockDim.x * gridDim.x) {
        atomicAdd(&counts[indices[i]], 1);
    }
}

__global__ void scatter_edges_kernel(
    const int32_t* __restrict__ csc_offsets,
    const int32_t* __restrict__ csc_indices,
    int32_t* __restrict__ write_pos,
    int32_t* __restrict__ csr_indices,
    int32_t N)
{
    for (int32_t col = blockIdx.x * blockDim.x + threadIdx.x; col < N;
         col += blockDim.x * gridDim.x) {
        int32_t start = csc_offsets[col];
        int32_t end = csc_offsets[col + 1];
        for (int32_t j = start; j < end; j++) {
            int32_t row = csc_indices[j];
            int32_t pos = atomicAdd(&write_pos[row], 1);
            csr_indices[pos] = col;
        }
    }
}





__global__ void init_uniform_kernel(double* __restrict__ hubs, int32_t N)
{
    double val = 1.0 / (double)N;
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        hubs[i] = val;
    }
}

__global__ void divide_by_scalar_kernel(
    double* __restrict__ x,
    const double* __restrict__ scalar,
    int32_t N)
{
    double s = *scalar;
    if (s <= 0.0) return;
    double inv_s = 1.0 / s;
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        x[i] *= inv_s;
    }
}

__global__ void normalize_and_diff_kernel(
    double* __restrict__ x,
    const double* __restrict__ x_max,
    const double* __restrict__ prev,
    double* __restrict__ diff_buf,
    int32_t N)
{
    double mx = *x_max;
    double inv_mx = (mx > 0.0) ? (1.0 / mx) : 0.0;
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        double val = x[i] * inv_mx;
        x[i] = val;
        diff_buf[i] = fabs(val - prev[i]);
    }
}





void launch_spmv(const int32_t* offsets, const int32_t* indices,
                 const double* x, double* y, int32_t N, int32_t avg_degree,
                 cudaStream_t stream)
{
    if (N == 0) return;
    int threads = 256;
    if (avg_degree <= 4) {
        int blocks = (N + threads - 1) / threads;
        spmv_thread_kernel<<<blocks, threads, 0, stream>>>(offsets, indices, x, y, N);
    } else {
        int warps_per_block = threads / 32;
        int blocks = (N + warps_per_block - 1) / warps_per_block;
        spmv_warp_kernel<<<blocks, threads, 0, stream>>>(offsets, indices, x, y, N);
    }
}

void launch_count_degrees(const int32_t* indices, int32_t* counts,
                          int32_t E, cudaStream_t stream)
{
    if (E == 0) return;
    int threads = 256;
    int blocks = (E + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    count_degrees_kernel<<<blocks, threads, 0, stream>>>(indices, counts, E);
}

void launch_scatter_edges(const int32_t* csc_offsets, const int32_t* csc_indices,
                          int32_t* write_pos, int32_t* csr_indices,
                          int32_t N, cudaStream_t stream)
{
    if (N == 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    scatter_edges_kernel<<<blocks, threads, 0, stream>>>(
        csc_offsets, csc_indices, write_pos, csr_indices, N);
}

void launch_init_uniform(double* hubs, int32_t N, cudaStream_t stream)
{
    if (N == 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    init_uniform_kernel<<<blocks, threads, 0, stream>>>(hubs, N);
}

void launch_divide_by_scalar(double* x, const double* scalar,
                             int32_t N, cudaStream_t stream)
{
    if (N == 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    divide_by_scalar_kernel<<<blocks, threads, 0, stream>>>(x, scalar, N);
}

void launch_normalize_and_diff(double* x, const double* x_max,
                               const double* prev, double* diff_buf,
                               int32_t N, cudaStream_t stream)
{
    if (N == 0) return;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    normalize_and_diff_kernel<<<blocks, threads, 0, stream>>>(
        x, x_max, prev, diff_buf, N);
}

size_t get_cub_reduce_temp_bytes(int32_t N)
{
    size_t temp1 = 0, temp2 = 0;
    double* dp = nullptr;
    cub::DeviceReduce::Max(nullptr, temp1, dp, dp, N);
    cub::DeviceReduce::Sum(nullptr, temp2, dp, dp, N);
    return (temp1 > temp2) ? temp1 : temp2;
}

void launch_cub_max(double* d_in, double* d_out, int32_t N,
                    void* temp, size_t temp_bytes, cudaStream_t stream)
{
    cub::DeviceReduce::Max(temp, temp_bytes, d_in, d_out, N, stream);
}

void launch_cub_sum(double* d_in, double* d_out, int32_t N,
                    void* temp, size_t temp_bytes, cudaStream_t stream)
{
    cub::DeviceReduce::Sum(temp, temp_bytes, d_in, d_out, N, stream);
}

size_t get_cub_scan_temp_bytes(int32_t N)
{
    size_t temp_bytes = 0;
    int32_t* dp = nullptr;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, dp, dp, N);
    return temp_bytes;
}

void launch_cub_exclusive_sum(int32_t* d_in, int32_t* d_out, int32_t N,
                              void* temp, size_t temp_bytes,
                              cudaStream_t stream)
{
    cub::DeviceScan::ExclusiveSum(temp, temp_bytes, d_in, d_out, N, stream);
}





struct Cache : Cacheable {
    int32_t* csr_offsets = nullptr;
    int64_t csr_offsets_cap = 0;

    int32_t* csr_indices = nullptr;
    int64_t csr_indices_cap = 0;

    int32_t* degree = nullptr;
    int64_t degree_cap = 0;

    double* temp_hubs = nullptr;
    int64_t temp_hubs_cap = 0;

    double* diff_buf = nullptr;
    int64_t diff_buf_cap = 0;

    void* cub_temp = nullptr;
    size_t cub_temp_cap = 0;

    double* scalars = nullptr;
    bool scalars_allocated = false;

    void ensure(int32_t N, int32_t E) {
        int64_t offsets_need = (int64_t)(N + 1);

        if (csr_offsets_cap < offsets_need) {
            if (csr_offsets) cudaFree(csr_offsets);
            cudaMalloc(&csr_offsets, offsets_need * sizeof(int32_t));
            csr_offsets_cap = offsets_need;
        }
        if (csr_indices_cap < (int64_t)E) {
            if (csr_indices) cudaFree(csr_indices);
            cudaMalloc(&csr_indices, (size_t)E * sizeof(int32_t));
            csr_indices_cap = E;
        }
        if (degree_cap < offsets_need) {
            if (degree) cudaFree(degree);
            cudaMalloc(&degree, offsets_need * sizeof(int32_t));
            degree_cap = offsets_need;
        }
        if (temp_hubs_cap < (int64_t)N) {
            if (temp_hubs) cudaFree(temp_hubs);
            cudaMalloc(&temp_hubs, (size_t)N * sizeof(double));
            temp_hubs_cap = N;
        }
        if (diff_buf_cap < (int64_t)N) {
            if (diff_buf) cudaFree(diff_buf);
            cudaMalloc(&diff_buf, (size_t)N * sizeof(double));
            diff_buf_cap = N;
        }

        size_t scan_temp = get_cub_scan_temp_bytes(N + 1);
        size_t reduce_temp = get_cub_reduce_temp_bytes(N);
        size_t needed = ((scan_temp > reduce_temp) ? scan_temp : reduce_temp) + 64;
        if (cub_temp_cap < needed) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, needed);
            cub_temp_cap = needed;
        }

        if (!scalars_allocated) {
            cudaMalloc(&scalars, 4 * sizeof(double));
            scalars_allocated = true;
        }
    }

    ~Cache() override {
        if (csr_offsets) cudaFree(csr_offsets);
        if (csr_indices) cudaFree(csr_indices);
        if (degree) cudaFree(degree);
        if (temp_hubs) cudaFree(temp_hubs);
        if (diff_buf) cudaFree(diff_buf);
        if (cub_temp) cudaFree(cub_temp);
        if (scalars) cudaFree(scalars);
    }
};

}  

HitsResultDouble hits(const graph32_t& graph,
                      double* hubs,
                      double* authorities,
                      double epsilon,
                      std::size_t max_iterations,
                      bool has_initial_hubs_guess,
                      bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t N = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;

    if (N == 0) {
        return HitsResultDouble{max_iterations, false,
                                std::numeric_limits<double>::max()};
    }

    double tolerance = static_cast<double>(N) * epsilon;
    int32_t avg_degree = (N > 0) ? (E / N) : 0;
    cudaStream_t stream = 0;

    cache.ensure(N, E);

    size_t scan_temp_bytes = get_cub_scan_temp_bytes(N + 1);
    size_t reduce_temp_bytes = get_cub_reduce_temp_bytes(N);

    
    cudaMemsetAsync(cache.degree, 0, (N + 1) * sizeof(int32_t), stream);
    launch_count_degrees(graph.indices, cache.degree, E, stream);
    launch_cub_exclusive_sum(cache.degree, cache.csr_offsets, N + 1,
                             cache.cub_temp, scan_temp_bytes, stream);
    cudaMemcpyAsync(cache.degree, cache.csr_offsets, (N + 1) * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice, stream);
    launch_scatter_edges(graph.offsets, graph.indices, cache.degree,
                         cache.csr_indices, N, stream);

    
    double* d_hub_max = cache.scalars;
    double* d_diff_sum = cache.scalars + 1;
    double* d_sum = cache.scalars + 2;

    
    double* prev_hubs = hubs;
    double* curr_hubs = cache.temp_hubs;

    if (has_initial_hubs_guess) {
        
        launch_cub_sum(prev_hubs, d_sum, N, cache.cub_temp,
                       reduce_temp_bytes, stream);
        launch_divide_by_scalar(prev_hubs, d_sum, N, stream);
    } else {
        launch_init_uniform(prev_hubs, N, stream);
    }

    
    
    
    
    
    double diff_sum = std::numeric_limits<double>::max();
    std::size_t iterations = 0;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        launch_spmv(graph.offsets, graph.indices, prev_hubs, authorities,
                    N, avg_degree, stream);

        
        launch_spmv(cache.csr_offsets, cache.csr_indices, authorities,
                    curr_hubs, N, avg_degree, stream);

        
        launch_cub_max(curr_hubs, d_hub_max, N, cache.cub_temp,
                       reduce_temp_bytes, stream);
        launch_normalize_and_diff(curr_hubs, d_hub_max, prev_hubs,
                                  cache.diff_buf, N, stream);

        
        launch_cub_sum(cache.diff_buf, d_diff_sum, N, cache.cub_temp,
                       reduce_temp_bytes, stream);

        
        std::swap(prev_hubs, curr_hubs);
        iterations++;

        
        cudaMemcpy(&diff_sum, d_diff_sum, sizeof(double), cudaMemcpyDeviceToHost);
        if (diff_sum < tolerance) break;
    }

    bool converged = (diff_sum < tolerance);

    
    
    
    launch_cub_max(authorities, d_hub_max, N, cache.cub_temp,
                   reduce_temp_bytes, stream);
    launch_divide_by_scalar(authorities, d_hub_max, N, stream);

    
    if (normalize) {
        launch_cub_sum(prev_hubs, d_sum, N, cache.cub_temp,
                       reduce_temp_bytes, stream);
        launch_divide_by_scalar(prev_hubs, d_sum, N, stream);
        launch_cub_sum(authorities, d_sum, N, cache.cub_temp,
                       reduce_temp_bytes, stream);
        launch_divide_by_scalar(authorities, d_sum, N, stream);
    }

    
    if (prev_hubs != hubs) {
        cudaMemcpyAsync(hubs, prev_hubs, N * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    return HitsResultDouble{iterations, converged, diff_sum};
}

}  
