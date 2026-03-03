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



__device__ __forceinline__ double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ double warp_reduce_max(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ void atomicMaxDouble(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= val) return;
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}



__global__ void count_out_degree_kernel(
    const int32_t* __restrict__ csc_offsets,
    const int32_t* __restrict__ csc_indices,
    const uint32_t* __restrict__ mask,
    int32_t* __restrict__ degree,
    int32_t N)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= N) return;

    int start = __ldg(&csc_offsets[warp_id]);
    int end = __ldg(&csc_offsets[warp_id + 1]);

    for (int e = start + lane; e < end; e += 32) {
        uint32_t word = __ldg(&mask[e >> 5]);
        if ((word >> (e & 31)) & 1) {
            atomicAdd(&degree[__ldg(&csc_indices[e])], 1);
        }
    }
}

__global__ void scatter_csr_kernel(
    const int32_t* __restrict__ csc_offsets,
    const int32_t* __restrict__ csc_indices,
    const uint32_t* __restrict__ mask,
    int32_t* __restrict__ write_pos,
    int32_t* __restrict__ csr_indices,
    int32_t N)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= N) return;

    int32_t dst = warp_id;
    int start = __ldg(&csc_offsets[warp_id]);
    int end = __ldg(&csc_offsets[warp_id + 1]);

    for (int e = start + lane; e < end; e += 32) {
        uint32_t word = __ldg(&mask[e >> 5]);
        if ((word >> (e & 31)) & 1) {
            int src = __ldg(&csc_indices[e]);
            int pos = atomicAdd(&write_pos[src], 1);
            csr_indices[pos] = dst;
        }
    }
}



__global__ void csc_spmv_masked_warp_kernel(
    const int32_t* __restrict__ csc_offsets,
    const int32_t* __restrict__ csc_indices,
    const uint32_t* __restrict__ mask,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t N)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= N) return;

    int start = __ldg(&csc_offsets[warp_id]);
    int end = __ldg(&csc_offsets[warp_id + 1]);

    double sum = 0.0;
    for (int e = start + lane; e < end; e += 32) {
        uint32_t word = __ldg(&mask[e >> 5]);
        if ((word >> (e & 31)) & 1) {
            sum += __ldg(&x[__ldg(&csc_indices[e])]);
        }
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) y[warp_id] = sum;
}

__global__ void csr_spmv_warp_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t N)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= N) return;

    int start = __ldg(&csr_offsets[warp_id]);
    int end = __ldg(&csr_offsets[warp_id + 1]);

    double sum = 0.0;
    for (int e = start + lane; e < end; e += 32) {
        sum += __ldg(&x[__ldg(&csr_indices[e])]);
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) y[warp_id] = sum;
}



__global__ void csc_spmv_masked_thread_kernel(
    const int32_t* __restrict__ csc_offsets,
    const int32_t* __restrict__ csc_indices,
    const uint32_t* __restrict__ mask,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t N)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;

    int start = __ldg(&csc_offsets[v]);
    int end = __ldg(&csc_offsets[v + 1]);

    double sum = 0.0;
    for (int e = start; e < end; e++) {
        uint32_t word = __ldg(&mask[e >> 5]);
        if ((word >> (e & 31)) & 1) {
            sum += __ldg(&x[__ldg(&csc_indices[e])]);
        }
    }
    y[v] = sum;
}

__global__ void csr_spmv_thread_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t N)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;

    int start = __ldg(&csr_offsets[v]);
    int end = __ldg(&csr_offsets[v + 1]);

    double sum = 0.0;
    for (int e = start; e < end; e++) {
        sum += __ldg(&x[__ldg(&csr_indices[e])]);
    }
    y[v] = sum;
}



__global__ void fused_max_kernel(
    const double* __restrict__ arr1,
    const double* __restrict__ arr2,
    double* __restrict__ max1,
    double* __restrict__ max2,
    int32_t N)
{
    __shared__ double s_max1[32], s_max2[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = blockDim.x >> 5;
    double local_max1 = 0.0, local_max2 = 0.0;
    for (int idx = blockIdx.x * blockDim.x + tid; idx < N; idx += blockDim.x * gridDim.x) {
        local_max1 = fmax(local_max1, arr1[idx]);
        local_max2 = fmax(local_max2, arr2[idx]);
    }
    local_max1 = warp_reduce_max(local_max1);
    local_max2 = warp_reduce_max(local_max2);
    if (lane == 0) { s_max1[warp] = local_max1; s_max2[warp] = local_max2; }
    __syncthreads();
    if (warp == 0) {
        local_max1 = (lane < num_warps) ? s_max1[lane] : 0.0;
        local_max2 = (lane < num_warps) ? s_max2[lane] : 0.0;
        local_max1 = warp_reduce_max(local_max1);
        local_max2 = warp_reduce_max(local_max2);
        if (lane == 0) { atomicMaxDouble(max1, local_max1); atomicMaxDouble(max2, local_max2); }
    }
}

__global__ void normalize_diff_kernel(
    double* __restrict__ curr_hubs,
    const double* __restrict__ prev_hubs,
    double* __restrict__ auth,
    const double* __restrict__ d_hub_max,
    const double* __restrict__ d_auth_max,
    double* __restrict__ d_diff_sum,
    int32_t N)
{
    __shared__ double s_sum[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = blockDim.x >> 5;
    double hub_max = *d_hub_max;
    double auth_max = *d_auth_max;
    double inv_hub = (hub_max > 0.0) ? 1.0 / hub_max : 0.0;
    double inv_auth = (auth_max > 0.0) ? 1.0 / auth_max : 0.0;
    double local_diff = 0.0;
    for (int idx = blockIdx.x * blockDim.x + tid; idx < N; idx += blockDim.x * gridDim.x) {
        double h = curr_hubs[idx] * inv_hub;
        curr_hubs[idx] = h;
        auth[idx] *= inv_auth;
        local_diff += fabs(h - prev_hubs[idx]);
    }
    local_diff = warp_reduce_sum(local_diff);
    if (lane == 0) s_sum[warp] = local_diff;
    __syncthreads();
    if (warp == 0) {
        local_diff = (lane < num_warps) ? s_sum[lane] : 0.0;
        local_diff = warp_reduce_sum(local_diff);
        if (lane == 0) atomicAdd(d_diff_sum, local_diff);
    }
}



__global__ void init_uniform_kernel(double* arr, double val, int32_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) arr[idx] = val;
}

__global__ void sum_reduce_kernel(const double* __restrict__ arr, double* __restrict__ out, int32_t N) {
    __shared__ double s_sum[32];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int num_warps = blockDim.x >> 5;
    double local_sum = 0.0;
    for (int idx = blockIdx.x * blockDim.x + tid; idx < N; idx += blockDim.x * gridDim.x)
        local_sum += arr[idx];
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) s_sum[warp] = local_sum;
    __syncthreads();
    if (warp == 0) {
        local_sum = (lane < num_warps) ? s_sum[lane] : 0.0;
        local_sum = warp_reduce_sum(local_sum);
        if (lane == 0) atomicAdd(out, local_sum);
    }
}

__global__ void scale_kernel(double* arr, const double* scale_val, int32_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double s = *scale_val;
        if (s > 0.0) arr[idx] /= s;
    }
}



void launch_count_out_degree(const int32_t* offsets, const int32_t* indices,
    const uint32_t* mask, int32_t* degree, int32_t N, cudaStream_t stream) {
    if (N == 0) return;
    int block = 256;
    int grid = (N + 7) / 8;
    count_out_degree_kernel<<<grid, block, 0, stream>>>(offsets, indices, mask, degree, N);
}

void launch_scatter_csr(const int32_t* offsets, const int32_t* indices,
    const uint32_t* mask, int32_t* write_pos, int32_t* csr_indices, int32_t N, cudaStream_t stream) {
    if (N == 0) return;
    int block = 256;
    int grid = (N + 7) / 8;
    scatter_csr_kernel<<<grid, block, 0, stream>>>(offsets, indices, mask, write_pos, csr_indices, N);
}

void launch_prefix_sum(int32_t* d_in, int32_t* d_out, int32_t count, void* temp, size_t temp_bytes, cudaStream_t stream) {
    cub::DeviceScan::ExclusiveSum(temp, temp_bytes, d_in, d_out, count, stream);
}

void launch_csc_spmv_masked(const int32_t* offsets, const int32_t* indices,
    const uint32_t* mask, const double* x, double* y, int32_t N, bool use_thread, cudaStream_t stream) {
    if (N == 0) return;
    if (use_thread) {
        int block = 256;
        int grid = (N + block - 1) / block;
        csc_spmv_masked_thread_kernel<<<grid, block, 0, stream>>>(offsets, indices, mask, x, y, N);
    } else {
        int block = 256;
        int grid = (N + 7) / 8;
        csc_spmv_masked_warp_kernel<<<grid, block, 0, stream>>>(offsets, indices, mask, x, y, N);
    }
}

void launch_csr_spmv(const int32_t* offsets, const int32_t* indices,
    const double* x, double* y, int32_t N, bool use_thread, cudaStream_t stream) {
    if (N == 0) return;
    if (use_thread) {
        int block = 256;
        int grid = (N + block - 1) / block;
        csr_spmv_thread_kernel<<<grid, block, 0, stream>>>(offsets, indices, x, y, N);
    } else {
        int block = 256;
        int grid = (N + 7) / 8;
        csr_spmv_warp_kernel<<<grid, block, 0, stream>>>(offsets, indices, x, y, N);
    }
}

void launch_fused_max(const double* arr1, const double* arr2,
    double* max1, double* max2, int32_t N, cudaStream_t stream) {
    if (N == 0) return;
    int block = 256;
    int grid = (N + block * 4 - 1) / (block * 4);
    if (grid > 1024) grid = 1024;
    if (grid < 1) grid = 1;
    fused_max_kernel<<<grid, block, 0, stream>>>(arr1, arr2, max1, max2, N);
}

void launch_normalize_diff(double* curr_hubs, const double* prev_hubs,
    double* auth, const double* hub_max, const double* auth_max,
    double* diff_sum, int32_t N, cudaStream_t stream) {
    if (N == 0) return;
    int block = 256;
    int grid = (N + block * 4 - 1) / (block * 4);
    if (grid > 1024) grid = 1024;
    if (grid < 1) grid = 1;
    normalize_diff_kernel<<<grid, block, 0, stream>>>(
        curr_hubs, prev_hubs, auth, hub_max, auth_max, diff_sum, N);
}

void launch_init_uniform(double* arr, double val, int32_t N, cudaStream_t stream) {
    if (N == 0) return;
    int block = 256;
    int grid = (N + block - 1) / block;
    init_uniform_kernel<<<grid, block, 0, stream>>>(arr, val, N);
}

void launch_sum_reduce(const double* arr, double* out, int32_t N, cudaStream_t stream) {
    if (N == 0) return;
    int block = 256;
    int grid = (N + block * 4 - 1) / (block * 4);
    if (grid > 1024) grid = 1024;
    if (grid < 1) grid = 1;
    sum_reduce_kernel<<<grid, block, 0, stream>>>(arr, out, N);
}

void launch_scale(double* arr, const double* scale_val, int32_t N, cudaStream_t stream) {
    if (N == 0) return;
    int block = 256;
    int grid = (N + block - 1) / block;
    scale_kernel<<<grid, block, 0, stream>>>(arr, scale_val, N);
}



struct Cache : Cacheable {
    
    int32_t* degree = nullptr;
    int32_t* csr_offsets = nullptr;
    int32_t* csr_indices = nullptr;
    int32_t* write_pos = nullptr;
    void* prefix_temp = nullptr;

    
    double* hubs_temp = nullptr;
    double* scratch = nullptr;  

    
    int64_t degree_capacity = 0;       
    int64_t csr_offsets_capacity = 0;  
    int64_t csr_indices_capacity = 0;  
    int64_t write_pos_capacity = 0;    
    size_t prefix_temp_capacity = 0;   
    int64_t hubs_temp_capacity = 0;    
    bool scratch_allocated = false;

    void ensure_preprocessing(int64_t n_plus_1, size_t prefix_bytes) {
        if (degree_capacity < n_plus_1) {
            if (degree) cudaFree(degree);
            cudaMalloc(&degree, n_plus_1 * sizeof(int32_t));
            degree_capacity = n_plus_1;
        }
        if (csr_offsets_capacity < n_plus_1) {
            if (csr_offsets) cudaFree(csr_offsets);
            cudaMalloc(&csr_offsets, n_plus_1 * sizeof(int32_t));
            csr_offsets_capacity = n_plus_1;
        }
        if (write_pos_capacity < n_plus_1) {
            if (write_pos) cudaFree(write_pos);
            cudaMalloc(&write_pos, n_plus_1 * sizeof(int32_t));
            write_pos_capacity = n_plus_1;
        }
        if (prefix_temp_capacity < prefix_bytes) {
            if (prefix_temp) cudaFree(prefix_temp);
            cudaMalloc(&prefix_temp, prefix_bytes + 16);
            prefix_temp_capacity = prefix_bytes;
        }
    }

    void ensure_csr_indices(int64_t e_active) {
        if (csr_indices_capacity < e_active) {
            if (csr_indices) cudaFree(csr_indices);
            cudaMalloc(&csr_indices, e_active * sizeof(int32_t));
            csr_indices_capacity = e_active;
        }
    }

    void ensure_iteration(int64_t n) {
        if (hubs_temp_capacity < n) {
            if (hubs_temp) cudaFree(hubs_temp);
            cudaMalloc(&hubs_temp, n * sizeof(double));
            hubs_temp_capacity = n;
        }
        if (!scratch_allocated) {
            cudaMalloc(&scratch, 4 * sizeof(double));
            scratch_allocated = true;
        }
    }

    ~Cache() override {
        if (degree) cudaFree(degree);
        if (csr_offsets) cudaFree(csr_offsets);
        if (csr_indices) cudaFree(csr_indices);
        if (write_pos) cudaFree(write_pos);
        if (prefix_temp) cudaFree(prefix_temp);
        if (hubs_temp) cudaFree(hubs_temp);
        if (scratch) cudaFree(scratch);
    }
};

}  

HitsResultDouble hits_mask(const graph32_t& graph,
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
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_mask = graph.edge_mask;

    cudaStream_t stream = 0;
    double tolerance = static_cast<double>(N) * epsilon;

    double* d_hubs = hubs;
    double* d_auth = authorities;

    if (N == 0) {
        return HitsResultDouble{max_iterations, false, std::numeric_limits<double>::max()};
    }

    
    size_t prefix_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, prefix_temp_bytes, (int32_t*)nullptr, (int32_t*)nullptr, N + 1);

    cache.ensure_preprocessing(N + 1, prefix_temp_bytes);
    cache.ensure_iteration(N);

    int32_t* d_degree = cache.degree;
    int32_t* d_csr_offsets = cache.csr_offsets;
    int32_t* d_write_pos = cache.write_pos;

    cudaMemsetAsync(d_degree, 0, (N + 1) * sizeof(int32_t), stream);
    launch_count_out_degree(d_offsets, d_indices, d_mask, d_degree, N, stream);

    launch_prefix_sum(d_degree, d_csr_offsets, N + 1, cache.prefix_temp, prefix_temp_bytes, stream);

    int32_t E_active = 0;
    cudaMemcpyAsync(&E_active, &d_csr_offsets[N], sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int64_t alloc_E = (E_active > 0) ? (int64_t)E_active : 1;
    cache.ensure_csr_indices(alloc_E);
    int32_t* d_csr_indices = cache.csr_indices;

    cudaMemcpyAsync(d_write_pos, d_csr_offsets, (N + 1) * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

    if (E_active > 0) {
        launch_scatter_csr(d_offsets, d_indices, d_mask, d_write_pos, d_csr_indices, N, stream);
    }

    
    double avg_degree = (N > 0) ? (double)E / N : 0;
    bool use_thread_per_row = (avg_degree < 8.0);

    
    double* d_hubs_temp = cache.hubs_temp;
    double* d_scratch = cache.scratch;
    double* d_max_hub = d_scratch;
    double* d_max_auth = d_scratch + 1;
    double* d_diff_sum = d_scratch + 2;
    double* d_sum_val = d_scratch + 3;

    if (has_initial_hubs_guess) {
        cudaMemsetAsync(d_sum_val, 0, sizeof(double), stream);
        launch_sum_reduce(d_hubs, d_sum_val, N, stream);
        launch_scale(d_hubs, d_sum_val, N, stream);
    } else {
        launch_init_uniform(d_hubs, 1.0 / N, N, stream);
    }

    
    double* prev_hubs = d_hubs;
    double* curr_hubs = d_hubs_temp;

    double h_diff_sum = std::numeric_limits<double>::max();
    size_t iter = 0;
    const int CHECK_INTERVAL = 1;

    if (max_iterations == 0) goto done;

    while (true) {
        
        launch_csc_spmv_masked(d_offsets, d_indices, d_mask, prev_hubs, d_auth, N, use_thread_per_row, stream);

        
        launch_csr_spmv(d_csr_offsets, d_csr_indices, d_auth, curr_hubs, N, use_thread_per_row, stream);

        
        cudaMemsetAsync(d_scratch, 0, 3 * sizeof(double), stream);
        launch_fused_max(curr_hubs, d_auth, d_max_hub, d_max_auth, N, stream);
        launch_normalize_diff(curr_hubs, prev_hubs, d_auth, d_max_hub, d_max_auth, d_diff_sum, N, stream);

        
        std::swap(prev_hubs, curr_hubs);
        iter++;

        
        bool should_check = (iter % CHECK_INTERVAL == 0) || (iter >= max_iterations);
        if (should_check) {
            cudaMemcpyAsync(&h_diff_sum, d_diff_sum, sizeof(double), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (h_diff_sum < tolerance) break;
            if (iter >= max_iterations) break;
        }
    }

done:
    
    if (normalize) {
        cudaMemsetAsync(d_sum_val, 0, sizeof(double), stream);
        launch_sum_reduce(prev_hubs, d_sum_val, N, stream);
        launch_scale(prev_hubs, d_sum_val, N, stream);

        cudaMemsetAsync(d_sum_val, 0, sizeof(double), stream);
        launch_sum_reduce(d_auth, d_sum_val, N, stream);
        launch_scale(d_auth, d_sum_val, N, stream);
    }

    if (prev_hubs != d_hubs) {
        cudaMemcpyAsync(d_hubs, prev_hubs, N * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    bool converged = (iter < max_iterations);
    return HitsResultDouble{iter, converged, h_diff_sum};
}

}  
