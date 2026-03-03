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
#include <cfloat>

namespace aai {

namespace {

constexpr int BLOCK_SIZE = 256;






__global__ void fwd_block(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t v_start, int32_t v_count)
{
    if ((int)blockIdx.x >= v_count) return;
    int v = v_start + (int)blockIdx.x;
    int start = offsets[v];
    int end = offsets[v + 1];

    double sum = 0.0;
    for (int e = start + (int)threadIdx.x; e < end; e += BLOCK_SIZE) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1u)
            sum += x[indices[e]];
    }

    typedef cub::BlockReduce<double, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage tmp;
    sum = BR(tmp).Sum(sum);
    if (threadIdx.x == 0) y[v] = sum;
}


__global__ void fwd_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t v_start, int32_t v_count)
{
    int warp_id = ((int)blockIdx.x * BLOCK_SIZE + (int)threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= v_count) return;

    int v = v_start + warp_id;
    int start = offsets[v];
    int end = offsets[v + 1];

    double sum = 0.0;
    for (int e = start + lane; e < end; e += 32) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1u)
            sum += x[indices[e]];
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, s);

    if (lane == 0) y[v] = sum;
}


__global__ void fwd_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t v_start, int32_t v_count)
{
    int tid = (int)blockIdx.x * BLOCK_SIZE + (int)threadIdx.x;
    if (tid >= v_count) return;

    int v = v_start + tid;
    int start = offsets[v];
    int end = offsets[v + 1];

    double sum = 0.0;
    for (int e = start; e < end; e++) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1u)
            sum += x[indices[e]];
    }
    y[v] = sum;
}





__global__ void rev_block(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t v_start, int32_t v_count)
{
    if ((int)blockIdx.x >= v_count) return;
    int v = v_start + (int)blockIdx.x;
    int start = offsets[v];
    int end = offsets[v + 1];
    double val = x[v];

    for (int e = start + (int)threadIdx.x; e < end; e += BLOCK_SIZE) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1u)
            atomicAdd(&y[indices[e]], val);
    }
}

__global__ void rev_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t v_start, int32_t v_count)
{
    int warp_id = ((int)blockIdx.x * BLOCK_SIZE + (int)threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= v_count) return;

    int v = v_start + warp_id;
    int start = offsets[v];
    int end = offsets[v + 1];
    double val = x[v];

    for (int e = start + lane; e < end; e += 32) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1u)
            atomicAdd(&y[indices[e]], val);
    }
}

__global__ void rev_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t v_start, int32_t v_count)
{
    int tid = (int)blockIdx.x * BLOCK_SIZE + (int)threadIdx.x;
    if (tid >= v_count) return;

    int v = v_start + tid;
    int start = offsets[v];
    int end = offsets[v + 1];
    double val = x[v];

    for (int e = start; e < end; e++) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1u)
            atomicAdd(&y[indices[e]], val);
    }
}





__global__ void fill_kernel(double* arr, double val, int32_t n) {
    int idx = (int)blockIdx.x * BLOCK_SIZE + (int)threadIdx.x;
    if (idx < n) arr[idx] = val;
}


__global__ void fused_normalize_diff(
    double* __restrict__ new_hubs,
    const double* __restrict__ old_hubs,
    double* __restrict__ auth,
    const double* __restrict__ max_h_ptr,
    const double* __restrict__ max_a_ptr,
    double* __restrict__ diff_out,
    int32_t n)
{
    double max_h = *max_h_ptr;
    double max_a = *max_a_ptr;
    double inv_max_h = (max_h > 0.0) ? (1.0 / max_h) : 0.0;
    double inv_max_a = (max_a > 0.0) ? (1.0 / max_a) : 0.0;

    typedef cub::BlockReduce<double, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage tmp;

    int idx = (int)blockIdx.x * BLOCK_SIZE + (int)threadIdx.x;
    double d = 0.0;
    if (idx < n) {
        double h = new_hubs[idx] * inv_max_h;
        new_hubs[idx] = h;
        d = fabs(h - old_hubs[idx]);
        auth[idx] *= inv_max_a;
    }

    d = BR(tmp).Sum(d);
    if (threadIdx.x == 0 && d > 0.0)
        atomicAdd(diff_out, d);
}


__global__ void l1_normalize(double* arr, const double* sum_ptr, int32_t n) {
    int idx = (int)blockIdx.x * BLOCK_SIZE + (int)threadIdx.x;
    if (idx < n) {
        double s = *sum_ptr;
        if (s > 0.0) arr[idx] /= s;
    }
}





static void dispatch_fwd(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    const double* x, double* y,
    int seg0, int seg1, int seg2, int seg3, cudaStream_t stream)
{
    int n_high = seg1 - seg0;
    int n_mid  = seg2 - seg1;
    int n_low  = seg3 - seg2;

    if (n_high > 0)
        fwd_block<<<n_high, BLOCK_SIZE, 0, stream>>>(offsets, indices, edge_mask, x, y, seg0, n_high);
    if (n_mid > 0) {
        int warps_per_block = BLOCK_SIZE / 32;
        int blocks = (n_mid + warps_per_block - 1) / warps_per_block;
        fwd_warp<<<blocks, BLOCK_SIZE, 0, stream>>>(offsets, indices, edge_mask, x, y, seg1, n_mid);
    }
    if (n_low > 0) {
        int blocks = (n_low + BLOCK_SIZE - 1) / BLOCK_SIZE;
        fwd_thread<<<blocks, BLOCK_SIZE, 0, stream>>>(offsets, indices, edge_mask, x, y, seg2, n_low);
    }
}

static void dispatch_rev(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    const double* x, double* y,
    int seg0, int seg1, int seg2, int seg3, cudaStream_t stream)
{
    int n_high = seg1 - seg0;
    int n_mid  = seg2 - seg1;
    int n_low  = seg3 - seg2;

    if (n_high > 0)
        rev_block<<<n_high, BLOCK_SIZE, 0, stream>>>(offsets, indices, edge_mask, x, y, seg0, n_high);
    if (n_mid > 0) {
        int warps_per_block = BLOCK_SIZE / 32;
        int blocks = (n_mid + warps_per_block - 1) / warps_per_block;
        rev_warp<<<blocks, BLOCK_SIZE, 0, stream>>>(offsets, indices, edge_mask, x, y, seg1, n_mid);
    }
    if (n_low > 0) {
        int blocks = (n_low + BLOCK_SIZE - 1) / BLOCK_SIZE;
        rev_thread<<<blocks, BLOCK_SIZE, 0, stream>>>(offsets, indices, edge_mask, x, y, seg2, n_low);
    }
}




static size_t compute_cub_temp_bytes(int32_t num_vertices) {
    size_t b1 = 0, b2 = 0;
    cub::DeviceReduce::Max(nullptr, b1, (double*)nullptr, (double*)nullptr, (int)num_vertices);
    cub::DeviceReduce::Sum(nullptr, b2, (double*)nullptr, (double*)nullptr, (int)num_vertices);
    return (b1 > b2) ? b1 : b2;
}




struct Cache : Cacheable {
    
    double* d_max_h = nullptr;
    double* d_max_a = nullptr;
    double* d_diff = nullptr;
    double* d_sum = nullptr;

    
    double* temp_hubs = nullptr;
    int64_t temp_hubs_capacity = 0;

    void* cub_temp = nullptr;
    size_t cub_temp_capacity = 0;

    Cache() {
        cudaMalloc(&d_max_h, sizeof(double));
        cudaMalloc(&d_max_a, sizeof(double));
        cudaMalloc(&d_diff, sizeof(double));
        cudaMalloc(&d_sum, sizeof(double));
    }

    ~Cache() override {
        if (d_max_h) cudaFree(d_max_h);
        if (d_max_a) cudaFree(d_max_a);
        if (d_diff) cudaFree(d_diff);
        if (d_sum) cudaFree(d_sum);
        if (temp_hubs) cudaFree(temp_hubs);
        if (cub_temp) cudaFree(cub_temp);
    }

    void ensure(int32_t num_vertices) {
        if (temp_hubs_capacity < num_vertices) {
            if (temp_hubs) cudaFree(temp_hubs);
            cudaMalloc(&temp_hubs, (size_t)num_vertices * sizeof(double));
            temp_hubs_capacity = num_vertices;
        }

        size_t needed = compute_cub_temp_bytes(num_vertices);
        if (needed < 1024) needed = 1024;
        if (cub_temp_capacity < needed) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, needed);
            cub_temp_capacity = needed;
        }
    }
};

}  

HitsResultDouble hits_seg_mask(const graph32_t& graph,
                               double* hubs,
                               double* authorities,
                               double epsilon,
                               std::size_t max_iterations,
                               bool has_initial_hubs_guess,
                               bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    const uint32_t* edge_mask = graph.edge_mask;
    const auto& seg = graph.segment_offsets.value();

    cudaStream_t stream = 0;

    if (num_vertices == 0) {
        return HitsResultDouble{max_iterations, false, DBL_MAX};
    }

    if (max_iterations == 0) {
        return HitsResultDouble{0, false, 1e30};
    }

    cache.ensure(num_vertices);

    double tolerance = (double)num_vertices * epsilon;
    int seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg3 = seg[3];
    int blocks = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;

    size_t cub_bytes = cache.cub_temp_capacity;

    
    if (has_initial_hubs_guess) {
        
        cub::DeviceReduce::Sum(cache.cub_temp, cub_bytes, hubs, cache.d_sum, num_vertices, stream);
        l1_normalize<<<blocks, BLOCK_SIZE, 0, stream>>>(hubs, cache.d_sum, num_vertices);
    } else {
        double init_val = 1.0 / (double)num_vertices;
        fill_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(hubs, init_val, num_vertices);
    }

    
    int n_zero = num_vertices - seg3;
    if (n_zero > 0) {
        cudaMemsetAsync(authorities + seg3, 0, (size_t)n_zero * sizeof(double), stream);
    }

    double* prev_hubs = hubs;
    double* curr_hubs = cache.temp_hubs;

    double diff_sum = DBL_MAX;
    size_t iter = 0;

    while (iter < max_iterations) {
        
        dispatch_fwd(offsets, indices, edge_mask, prev_hubs, authorities,
                     seg0, seg1, seg2, seg3, stream);

        
        cudaMemsetAsync(curr_hubs, 0, (size_t)num_vertices * sizeof(double), stream);

        
        dispatch_rev(offsets, indices, edge_mask, authorities, curr_hubs,
                     seg0, seg1, seg2, seg3, stream);

        
        cub::DeviceReduce::Max(cache.cub_temp, cub_bytes, curr_hubs, cache.d_max_h, num_vertices, stream);
        cub::DeviceReduce::Max(cache.cub_temp, cub_bytes, authorities, cache.d_max_a, num_vertices, stream);

        
        cudaMemsetAsync(cache.d_diff, 0, sizeof(double), stream);
        fused_normalize_diff<<<blocks, BLOCK_SIZE, 0, stream>>>(
            curr_hubs, prev_hubs, authorities, cache.d_max_h, cache.d_max_a, cache.d_diff, num_vertices);

        
        cudaMemcpy(&diff_sum, cache.d_diff, sizeof(double), cudaMemcpyDeviceToHost);

        
        double* tmp = prev_hubs;
        prev_hubs = curr_hubs;
        curr_hubs = tmp;

        iter++;
        if (diff_sum < tolerance) break;
    }

    
    if (normalize) {
        cub::DeviceReduce::Sum(cache.cub_temp, cub_bytes, prev_hubs, cache.d_sum, num_vertices, stream);
        l1_normalize<<<blocks, BLOCK_SIZE, 0, stream>>>(prev_hubs, cache.d_sum, num_vertices);

        cub::DeviceReduce::Sum(cache.cub_temp, cub_bytes, authorities, cache.d_sum, num_vertices, stream);
        l1_normalize<<<blocks, BLOCK_SIZE, 0, stream>>>(authorities, cache.d_sum, num_vertices);
    }

    
    if (prev_hubs != hubs) {
        cudaMemcpyAsync(hubs, prev_hubs, (size_t)num_vertices * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    bool converged = (iter < max_iterations);
    return HitsResultDouble{iter, converged, diff_sum};
}

}  
