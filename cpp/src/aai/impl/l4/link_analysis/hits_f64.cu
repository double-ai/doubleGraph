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
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <cstdint>
#include <algorithm>

namespace aai {

namespace {





struct Cache : Cacheable {
    double* hubs_scratch = nullptr;
    double* scalar = nullptr;       
    uint8_t* cub_temp = nullptr;
    int32_t* csr_offsets = nullptr;
    int32_t* csr_indices = nullptr;
    int32_t* csr_temp = nullptr;

    int64_t hubs_scratch_cap = 0;
    size_t cub_temp_cap = 0;
    int64_t csr_offsets_cap = 0;
    int64_t csr_indices_cap = 0;
    int64_t csr_temp_cap = 0;
    bool scalar_allocated = false;

    void ensure(int32_t N, int32_t E, bool need_csr, size_t temp_bytes) {
        if (hubs_scratch_cap < N) {
            if (hubs_scratch) cudaFree(hubs_scratch);
            cudaMalloc(&hubs_scratch, (size_t)N * sizeof(double));
            hubs_scratch_cap = N;
        }
        if (!scalar_allocated) {
            cudaMalloc(&scalar, 2 * sizeof(double));
            scalar_allocated = true;
        }
        if (cub_temp_cap < temp_bytes) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, temp_bytes + 16);
            cub_temp_cap = temp_bytes + 16;
        }
        if (need_csr) {
            if (csr_offsets_cap < (int64_t)(N + 1)) {
                if (csr_offsets) cudaFree(csr_offsets);
                cudaMalloc(&csr_offsets, (size_t)(N + 1) * sizeof(int32_t));
                csr_offsets_cap = N + 1;
            }
            if (csr_indices_cap < E) {
                if (csr_indices) cudaFree(csr_indices);
                cudaMalloc(&csr_indices, (size_t)E * sizeof(int32_t));
                csr_indices_cap = E;
            }
            if (csr_temp_cap < (int64_t)(N + 1)) {
                if (csr_temp) cudaFree(csr_temp);
                cudaMalloc(&csr_temp, (size_t)(N + 1) * sizeof(int32_t));
                csr_temp_cap = N + 1;
            }
        }
    }

    ~Cache() override {
        if (hubs_scratch) cudaFree(hubs_scratch);
        if (scalar) cudaFree(scalar);
        if (cub_temp) cudaFree(cub_temp);
        if (csr_offsets) cudaFree(csr_offsets);
        if (csr_indices) cudaFree(csr_indices);
        if (csr_temp) cudaFree(csr_temp);
    }
};




__global__ void spmv_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_rows)
{
    for (int row = blockIdx.x * blockDim.x + threadIdx.x;
         row < num_rows;
         row += blockDim.x * gridDim.x) {
        int start = offsets[row];
        int end = offsets[row + 1];
        double sum = 0.0;
        for (int j = start; j < end; j++) {
            sum += x[indices[j]];
        }
        y[row] = sum;
    }
}

void launch_spmv(
    const int* offsets, const int* indices,
    const double* x, double* y, int num_rows)
{
    int block = 256;
    int grid = (num_rows + block - 1) / block;
    if (grid > 65536) grid = 65536;
    spmv_kernel<<<grid, block>>>(offsets, indices, x, y, num_rows);
}




__global__ void fill_kernel(double* arr, int n, double val) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        arr[i] = val;
    }
}

void launch_fill(double* arr, int n, double val) {
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 65536) grid = 65536;
    fill_kernel<<<grid, block>>>(arr, n, val);
}




__global__ void normalize_kernel(
    double* __restrict__ arr,
    const double* __restrict__ scalar_ptr,
    int n)
{
    double s = *scalar_ptr;
    if (s == 0.0) return;
    double inv = 1.0 / s;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        arr[i] *= inv;
    }
}

void launch_normalize(double* arr, const double* scalar_ptr, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 65536) grid = 65536;
    normalize_kernel<<<grid, block>>>(arr, scalar_ptr, n);
}





struct AbsFunctor {
    __host__ __device__ double operator()(double x) const { return fabs(x); }
};

struct MaxOp {
    __host__ __device__ double operator()(double a, double b) const {
        return (a > b) ? a : b;
    }
};

struct AbsDiffFunctor {
    const double* a;
    const double* b;
    __host__ __device__ double operator()(int i) const { return fabs(a[i] - b[i]); }
};


void launch_reduce_max_abs(
    const double* d_in, double* d_out, int n,
    void* d_temp, size_t temp_bytes)
{
    auto abs_iter = thrust::make_transform_iterator(d_in, AbsFunctor());
    cub::DeviceReduce::Reduce(d_temp, temp_bytes, abs_iter, d_out, n, MaxOp(), 0.0);
}


void launch_reduce_l1_diff(
    const double* a, const double* b, double* d_out, int n,
    void* d_temp, size_t temp_bytes)
{
    auto diff_iter = thrust::make_transform_iterator(
        thrust::counting_iterator<int>(0), AbsDiffFunctor{a, b});
    cub::DeviceReduce::Sum(d_temp, temp_bytes, diff_iter, d_out, n);
}


void launch_reduce_sum(
    const double* d_in, double* d_out, int n,
    void* d_temp, size_t temp_bytes)
{
    cub::DeviceReduce::Sum(d_temp, temp_bytes, d_in, d_out, n);
}


size_t query_reduce_temp_bytes(int n) {
    size_t t1 = 0, t2 = 0, t3 = 0, t4 = 0;

    auto abs_iter = thrust::make_transform_iterator((const double*)nullptr, AbsFunctor());
    cub::DeviceReduce::Reduce(nullptr, t1, abs_iter, (double*)nullptr, n, MaxOp(), 0.0);

    auto diff_iter = thrust::make_transform_iterator(
        thrust::counting_iterator<int>(0), AbsDiffFunctor{nullptr, nullptr});
    cub::DeviceReduce::Sum(nullptr, t2, diff_iter, (double*)nullptr, n);

    cub::DeviceReduce::Sum(nullptr, t3, (const double*)nullptr, (double*)nullptr, n);

    cub::DeviceScan::ExclusiveSum(nullptr, t4, (int*)nullptr, (int*)nullptr, n + 1);

    size_t mx = t1;
    if (t2 > mx) mx = t2;
    if (t3 > mx) mx = t3;
    if (t4 > mx) mx = t4;
    return mx;
}





__global__ void count_degrees_kernel(
    const int* __restrict__ indices,
    int* __restrict__ degrees,
    int num_edges)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_edges; i += blockDim.x * gridDim.x) {
        atomicAdd(&degrees[indices[i]], 1);
    }
}

__global__ void scatter_csr_indices_kernel(
    const int* __restrict__ csc_offsets,
    const int* __restrict__ csc_indices,
    int* __restrict__ csr_write_pos,
    int* __restrict__ csr_indices,
    int num_cols)
{
    for (int col = blockIdx.x * blockDim.x + threadIdx.x;
         col < num_cols; col += blockDim.x * gridDim.x) {
        int start = csc_offsets[col];
        int end = csc_offsets[col + 1];
        for (int j = start; j < end; j++) {
            int row = csc_indices[j];
            int pos = atomicAdd(&csr_write_pos[row], 1);
            csr_indices[pos] = col;
        }
    }
}

void launch_build_csr_from_csc(
    const int* csc_offsets, const int* csc_indices,
    int* csr_offsets, int* csr_indices, int* temp_degrees,
    int num_vertices, int num_edges,
    void* cub_temp, size_t cub_temp_bytes)
{
    cudaMemset(temp_degrees, 0, num_vertices * sizeof(int));
    int block = 256;
    int grid1 = (num_edges + block - 1) / block;
    if (grid1 > 65536) grid1 = 65536;
    count_degrees_kernel<<<grid1, block>>>(csc_indices, temp_degrees, num_edges);
    cub::DeviceScan::ExclusiveSum(cub_temp, cub_temp_bytes, temp_degrees, csr_offsets, num_vertices + 1);
    cudaMemcpy(temp_degrees, csr_offsets, num_vertices * sizeof(int), cudaMemcpyDeviceToDevice);
    int grid2 = (num_vertices + block - 1) / block;
    if (grid2 > 65536) grid2 = 65536;
    scatter_csr_indices_kernel<<<grid2, block>>>(
        csc_offsets, csc_indices, temp_degrees, csr_indices, num_vertices);
}

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
    bool is_symmetric = graph.is_symmetric;
    const int32_t* csc_offsets = graph.offsets;
    const int32_t* csc_indices = graph.indices;

    
    int max_n = (N > E) ? N : E;
    size_t temp_bytes = query_reduce_temp_bytes(max_n);

    
    cache.ensure(N, E, !is_symmetric, temp_bytes);

    double* hubs_a = hubs;
    double* hubs_b = cache.hubs_scratch;
    double* auth = authorities;
    double* d_max = cache.scalar;
    double* d_diff = cache.scalar + 1;
    void* d_cub_temp = cache.cub_temp;

    
    const int32_t* csr_offsets_ptr = csc_offsets;
    const int32_t* csr_indices_ptr = csc_indices;

    if (!is_symmetric) {
        launch_build_csr_from_csc(
            csc_offsets, csc_indices,
            cache.csr_offsets, cache.csr_indices, cache.csr_temp,
            N, E, d_cub_temp, temp_bytes);

        csr_offsets_ptr = cache.csr_offsets;
        csr_indices_ptr = cache.csr_indices;
    }

    
    if (has_initial_hubs_guess) {
        
        launch_reduce_sum(hubs_a, d_max, N, d_cub_temp, temp_bytes);
        launch_normalize(hubs_a, d_max, N);
    } else {
        launch_fill(hubs_a, N, 1.0 / N);
    }

    
    double* hubs_cur = hubs_a;
    double* hubs_new = hubs_b;
    double h_diff = 0.0;
    size_t iterations = 0;
    bool converged = false;
    double conv_threshold = epsilon * N;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        
        launch_spmv(csc_offsets, csc_indices, hubs_cur, auth, N);
        
        launch_reduce_max_abs(auth, d_max, N, d_cub_temp, temp_bytes);
        launch_normalize(auth, d_max, N);

        
        launch_spmv(csr_offsets_ptr, csr_indices_ptr, auth, hubs_new, N);
        
        launch_reduce_max_abs(hubs_new, d_max, N, d_cub_temp, temp_bytes);
        launch_normalize(hubs_new, d_max, N);

        
        launch_reduce_l1_diff(hubs_new, hubs_cur, d_diff, N, d_cub_temp, temp_bytes);
        cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);

        iterations = iter + 1;

        if (h_diff < conv_threshold) {
            converged = true;
            std::swap(hubs_cur, hubs_new);
            break;
        }

        std::swap(hubs_cur, hubs_new);
    }

    
    if (normalize) {
        launch_reduce_sum(hubs_cur, d_max, N, d_cub_temp, temp_bytes);
        launch_normalize(hubs_cur, d_max, N);
        launch_reduce_sum(auth, d_max, N, d_cub_temp, temp_bytes);
        launch_normalize(auth, d_max, N);
    }

    
    if (hubs_cur != hubs) {
        cudaMemcpy(hubs, hubs_cur, (size_t)N * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    return HitsResultDouble{iterations, converged, h_diff};
}

}  
