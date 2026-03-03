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

namespace aai {

namespace {





struct Cache : Cacheable {
    double* temp_hubs = nullptr;
    int32_t* filt_csc_offsets = nullptr;
    int32_t* filt_csr_offsets = nullptr;
    int32_t* temp_counts = nullptr;
    double* scalars = nullptr;
    int32_t* filt_csc_indices = nullptr;
    int32_t* filt_csr_indices = nullptr;
    void* cub_temp = nullptr;

    int32_t vert_cap = 0;
    int32_t edge_cap = 0;
    size_t cub_cap = 0;

    void ensure(int32_t N, int32_t E, size_t cub_bytes) {
        if (vert_cap < N) {
            if (temp_hubs) cudaFree(temp_hubs);
            cudaMalloc(&temp_hubs, (size_t)N * sizeof(double));
            if (filt_csc_offsets) cudaFree(filt_csc_offsets);
            cudaMalloc(&filt_csc_offsets, (size_t)(N + 1) * sizeof(int32_t));
            if (filt_csr_offsets) cudaFree(filt_csr_offsets);
            cudaMalloc(&filt_csr_offsets, (size_t)(N + 1) * sizeof(int32_t));
            if (temp_counts) cudaFree(temp_counts);
            cudaMalloc(&temp_counts, (size_t)(N + 1) * sizeof(int32_t));
            if (scalars) cudaFree(scalars);
            cudaMalloc(&scalars, 8 * sizeof(double));
            vert_cap = N;
        }
        int32_t needed_edges = E > 0 ? E : 1;
        if (edge_cap < needed_edges) {
            if (filt_csc_indices) cudaFree(filt_csc_indices);
            cudaMalloc(&filt_csc_indices, (size_t)needed_edges * sizeof(int32_t));
            if (filt_csr_indices) cudaFree(filt_csr_indices);
            cudaMalloc(&filt_csr_indices, (size_t)needed_edges * sizeof(int32_t));
            edge_cap = needed_edges;
        }
        if (cub_cap < cub_bytes) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, cub_bytes);
            cub_cap = cub_bytes;
        }
    }

    ~Cache() override {
        if (temp_hubs) cudaFree(temp_hubs);
        if (filt_csc_offsets) cudaFree(filt_csc_offsets);
        if (filt_csc_indices) cudaFree(filt_csc_indices);
        if (filt_csr_offsets) cudaFree(filt_csr_offsets);
        if (filt_csr_indices) cudaFree(filt_csr_indices);
        if (temp_counts) cudaFree(temp_counts);
        if (scalars) cudaFree(scalars);
        if (cub_temp) cudaFree(cub_temp);
    }
};





__global__ void k_count_active_edges(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ mask,
    int32_t* __restrict__ counts,
    int32_t N
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < N; v += blockDim.x * gridDim.x) {
        int start = offsets[v];
        int end = offsets[v + 1];
        if (start >= end) { counts[v] = 0; continue; }
        int count = 0;
        int first_word = start >> 5;
        int last_word = (end - 1) >> 5;
        int first_bit = start & 31;
        int last_bit = (end - 1) & 31;
        if (first_word == last_word) {
            uint32_t w = mask[first_word] >> first_bit;
            int nbits = end - start;
            if (nbits < 32) w &= (1u << nbits) - 1;
            count = __popc(w);
        } else {
            count = __popc(mask[first_word] >> first_bit);
            for (int wi = first_word + 1; wi < last_word; wi++)
                count += __popc(mask[wi]);
            count += __popc(mask[last_word] & ((2u << last_bit) - 1));
        }
        counts[v] = count;
    }
}

__global__ void k_compact_csc(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t N
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < N; v += blockDim.x * gridDim.x) {
        int old_start = old_offsets[v];
        int old_end = old_offsets[v + 1];
        int write_pos = new_offsets[v];
        for (int j = old_start; j < old_end; j++) {
            if ((mask[j >> 5] >> (j & 31)) & 1)
                new_indices[write_pos++] = old_indices[j];
        }
    }
}

__global__ void k_count_per_source(
    const int32_t* __restrict__ csc_offsets,
    const int32_t* __restrict__ csc_indices,
    int32_t* __restrict__ source_counts,
    int32_t N
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < N; v += blockDim.x * gridDim.x) {
        int start = csc_offsets[v];
        int end = csc_offsets[v + 1];
        for (int j = start; j < end; j++)
            atomicAdd(&source_counts[csc_indices[j]], 1);
    }
}

__global__ void k_fill_csr(
    const int32_t* __restrict__ csc_offsets,
    const int32_t* __restrict__ csc_indices,
    const int32_t* __restrict__ csr_offsets,
    int32_t* __restrict__ csr_indices,
    int32_t* __restrict__ write_counters,
    int32_t N
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < N; v += blockDim.x * gridDim.x) {
        int start = csc_offsets[v];
        int end = csc_offsets[v + 1];
        for (int j = start; j < end; j++) {
            int src = csc_indices[j];
            int pos = atomicAdd(&write_counters[src], 1);
            csr_indices[csr_offsets[src] + pos] = v;
        }
    }
}





__device__ __forceinline__ void atomicMaxDouble(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= val) break;
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

__global__ __launch_bounds__(256, 8)
void k_spmv_max(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ x,
    double* __restrict__ y,
    double* __restrict__ d_max,
    int32_t N
) {
    double local_max = 0.0;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < N; v += blockDim.x * gridDim.x) {
        int start = __ldg(&offsets[v]);
        int end = __ldg(&offsets[v + 1]);
        double sum = 0.0;
        for (int j = start; j < end; j++)
            sum += __ldg(&x[__ldg(&indices[j])]);
        y[v] = sum;
        local_max = fmax(local_max, fabs(sum));
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmax(local_max, __shfl_down_sync(0xffffffff, local_max, offset));

    
    if ((threadIdx.x & 31) == 0 && local_max > 0.0)
        atomicMaxDouble(d_max, local_max);
}





__global__ __launch_bounds__(256, 8)
void k_normalize_diff(
    double* __restrict__ hubs,
    double* __restrict__ auth,
    const double* __restrict__ prev_hubs,
    const double* __restrict__ d_hub_max,
    const double* __restrict__ d_auth_max,
    double* __restrict__ d_diff,
    int32_t N
) {
    double hub_max = *d_hub_max;
    double auth_max = *d_auth_max;
    double inv_hmax = (hub_max > 0.0) ? 1.0 / hub_max : 1.0;
    double inv_amax = (auth_max > 0.0) ? 1.0 / auth_max : 1.0;

    double local_diff = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        double h = hubs[i] * inv_hmax;
        double a = auth[i] * inv_amax;
        hubs[i] = h;
        auth[i] = a;
        local_diff += fabs(h - prev_hubs[i]);
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_diff += __shfl_down_sync(0xffffffff, local_diff, offset);

    if ((threadIdx.x & 31) == 0)
        atomicAdd(d_diff, local_diff);
}





__global__ void k_fill_val(double* __restrict__ data, double val, int32_t N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        data[i] = val;
}

__global__ __launch_bounds__(256, 8)
void k_l1_sum(const double* __restrict__ data, double* __restrict__ d_sum, int32_t N) {
    double local_sum = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        local_sum += data[i];
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(d_sum, local_sum);
}

__global__ void k_div_scalar(double* __restrict__ data, const double* __restrict__ d_scalar, int32_t N) {
    double s = *d_scalar;
    if (s <= 0.0) return;
    double inv_s = 1.0 / s;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        data[i] *= inv_s;
}





static size_t get_cub_scan_temp_bytes(int N) {
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, (int32_t*)nullptr, (int32_t*)nullptr, N);
    return temp_bytes;
}

static void cub_exclusive_sum(int32_t* d_in, int32_t* d_out, int N, void* d_temp, size_t temp_bytes) {
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_in, d_out, N);
}

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

    const int32_t N = graph.number_of_vertices;
    const int32_t num_edges = graph.number_of_edges;

    if (N == 0) {
        return {max_iterations, false, 0.0};
    }

    size_t cub_temp_bytes = get_cub_scan_temp_bytes(N + 1);
    if (cub_temp_bytes < 256) cub_temp_bytes = 256;

    cache.ensure(N, num_edges, cub_temp_bytes);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    double* d_temp_hubs = cache.temp_hubs;
    int32_t* d_filt_csc_offsets = cache.filt_csc_offsets;
    int32_t* d_filt_csc_indices = cache.filt_csc_indices;
    int32_t* d_filt_csr_offsets = cache.filt_csr_offsets;
    int32_t* d_filt_csr_indices = cache.filt_csr_indices;
    int32_t* d_temp_counts = cache.temp_counts;
    double* d_scalars = cache.scalars;
    void* d_cub_temp = cache.cub_temp;

    const int BLOCK = 256;
    const int grid_N = (N + BLOCK - 1) / BLOCK;
    const int grid_cap = 2048;
    const int grid = (grid_N < grid_cap) ? grid_N : grid_cap;
    double tolerance = (double)N * epsilon;

    
    
    
    cudaMemsetAsync(&d_temp_counts[N], 0, sizeof(int32_t));
    k_count_active_edges<<<grid, BLOCK>>>(d_offsets, d_edge_mask, d_temp_counts, N);
    cub_exclusive_sum(d_temp_counts, d_filt_csc_offsets, N + 1, d_cub_temp, cub_temp_bytes);

    int32_t total_active = 0;
    cudaMemcpy(&total_active, &d_filt_csc_offsets[N], sizeof(int32_t), cudaMemcpyDeviceToHost);

    if (total_active == 0) {
        k_fill_val<<<grid, BLOCK>>>(hubs, 1.0 / N, N);
        cudaMemset(authorities, 0, N * sizeof(double));
        cudaDeviceSynchronize();
        return {1, true, 0.0};
    }

    k_compact_csc<<<grid, BLOCK>>>(d_offsets, d_indices, d_edge_mask,
                                    d_filt_csc_offsets, d_filt_csc_indices, N);

    cudaMemset(d_temp_counts, 0, (N + 1) * sizeof(int32_t));
    k_count_per_source<<<grid, BLOCK>>>(d_filt_csc_offsets, d_filt_csc_indices, d_temp_counts, N);
    cub_exclusive_sum(d_temp_counts, d_filt_csr_offsets, N + 1, d_cub_temp, cub_temp_bytes);
    cudaMemset(d_temp_counts, 0, (N + 1) * sizeof(int32_t));
    k_fill_csr<<<grid, BLOCK>>>(d_filt_csc_offsets, d_filt_csc_indices,
                                 d_filt_csr_offsets, d_filt_csr_indices, d_temp_counts, N);

    
    
    
    if (has_initial_hubs_guess) {
        cudaMemsetAsync(&d_scalars[5], 0, sizeof(double));
        k_l1_sum<<<grid, BLOCK>>>(hubs, &d_scalars[5], N);
        k_div_scalar<<<grid, BLOCK>>>(hubs, &d_scalars[5], N);
    } else {
        k_fill_val<<<grid, BLOCK>>>(hubs, 1.0 / (double)N, N);
    }

    
    
    
    double* prev_hubs = hubs;
    double* curr_hubs = d_temp_hubs;
    double* d_hub_max = &d_scalars[0];
    double* d_auth_max = &d_scalars[1];
    double* d_diff = &d_scalars[2];

    double diff_sum = 1e30;
    size_t iter = 0;

    while (iter < max_iterations) {
        
        cudaMemsetAsync(d_scalars, 0, 3 * sizeof(double));

        
        k_spmv_max<<<grid, BLOCK>>>(d_filt_csc_offsets, d_filt_csc_indices,
                                     prev_hubs, authorities, d_auth_max, N);

        
        k_spmv_max<<<grid, BLOCK>>>(d_filt_csr_offsets, d_filt_csr_indices,
                                     authorities, curr_hubs, d_hub_max, N);

        
        k_normalize_diff<<<grid, BLOCK>>>(curr_hubs, authorities, prev_hubs,
                                           d_hub_max, d_auth_max, d_diff, N);

        
        double* tmp = prev_hubs; prev_hubs = curr_hubs; curr_hubs = tmp;
        iter++;

        
        cudaMemcpy(&diff_sum, d_diff, sizeof(double), cudaMemcpyDeviceToHost);
        if (diff_sum < tolerance) break;
    }

    
    if (prev_hubs != hubs)
        cudaMemcpy(hubs, prev_hubs, N * sizeof(double), cudaMemcpyDeviceToDevice);

    
    if (normalize) {
        cudaMemsetAsync(&d_scalars[3], 0, 2 * sizeof(double));
        k_l1_sum<<<grid, BLOCK>>>(hubs, &d_scalars[3], N);
        k_l1_sum<<<grid, BLOCK>>>(authorities, &d_scalars[4], N);
        k_div_scalar<<<grid, BLOCK>>>(hubs, &d_scalars[3], N);
        k_div_scalar<<<grid, BLOCK>>>(authorities, &d_scalars[4], N);
    }

    cudaDeviceSynchronize();
    return {iter, diff_sum < tolerance, diff_sum};
}

}  
