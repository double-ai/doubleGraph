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
#include <cub/device/device_scan.cuh>
#include <cstdint>

namespace aai {

namespace {



struct Cache : Cacheable {
    
    int32_t* count = nullptr;
    int32_t* new_offsets = nullptr;
    uint8_t* scan_temp = nullptr;
    int32_t* new_indices = nullptr;

    int64_t count_capacity = 0;
    int64_t new_offsets_capacity = 0;
    int64_t scan_temp_capacity = 0;
    int64_t new_indices_capacity = 0;

    
    float* hubs_a = nullptr;
    float* hubs_b = nullptr;
    float* auth = nullptr;
    float* scalars = nullptr;  

    int64_t hubs_a_capacity = 0;
    int64_t hubs_b_capacity = 0;
    int64_t auth_capacity = 0;
    bool scalars_allocated = false;

    void ensure_compaction(int64_t n_plus_1, int64_t scan_bytes) {
        if (count_capacity < n_plus_1) {
            if (count) cudaFree(count);
            cudaMalloc(&count, n_plus_1 * sizeof(int32_t));
            count_capacity = n_plus_1;
        }
        if (new_offsets_capacity < n_plus_1) {
            if (new_offsets) cudaFree(new_offsets);
            cudaMalloc(&new_offsets, n_plus_1 * sizeof(int32_t));
            new_offsets_capacity = n_plus_1;
        }
        if (scan_temp_capacity < scan_bytes) {
            if (scan_temp) cudaFree(scan_temp);
            cudaMalloc(&scan_temp, scan_bytes);
            scan_temp_capacity = scan_bytes;
        }
    }

    void ensure_indices(int64_t active_edges) {
        if (new_indices_capacity < active_edges) {
            if (new_indices) cudaFree(new_indices);
            cudaMalloc(&new_indices, active_edges * sizeof(int32_t));
            new_indices_capacity = active_edges;
        }
    }

    void ensure_iteration(int64_t n) {
        if (hubs_a_capacity < n) {
            if (hubs_a) cudaFree(hubs_a);
            cudaMalloc(&hubs_a, n * sizeof(float));
            hubs_a_capacity = n;
        }
        if (hubs_b_capacity < n) {
            if (hubs_b) cudaFree(hubs_b);
            cudaMalloc(&hubs_b, n * sizeof(float));
            hubs_b_capacity = n;
        }
        if (auth_capacity < n) {
            if (auth) cudaFree(auth);
            cudaMalloc(&auth, n * sizeof(float));
            auth_capacity = n;
        }
        if (!scalars_allocated) {
            cudaMalloc(&scalars, 4 * sizeof(float));
            scalars_allocated = true;
        }
    }

    ~Cache() override {
        if (count) cudaFree(count);
        if (new_offsets) cudaFree(new_offsets);
        if (scan_temp) cudaFree(scan_temp);
        if (new_indices) cudaFree(new_indices);
        if (hubs_a) cudaFree(hubs_a);
        if (hubs_b) cudaFree(hubs_b);
        if (auth) cudaFree(auth);
        if (scalars) cudaFree(scalars);
    }
};



__global__ void compact_count_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ count,
    int32_t n)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int start = offsets[v];
    int end = offsets[v + 1];
    int cnt = 0;
    int e = start;
    for (; e < end && (e & 31) != 0; e++)
        if ((edge_mask[e >> 5] >> (e & 31)) & 1) cnt++;
    for (; e + 32 <= end; e += 32)
        cnt += __popc(edge_mask[e >> 5]);
    for (; e < end; e++)
        if ((edge_mask[e >> 5] >> (e & 31)) & 1) cnt++;
    count[v] = cnt;
}

__global__ void compact_scatter_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t n)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int old_start = old_offsets[v];
    int old_end = old_offsets[v + 1];
    int new_pos = new_offsets[v];
    for (int e = old_start; e < old_end; e++) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1) {
            new_indices[new_pos++] = old_indices[e];
        }
    }
}



__global__ void spmv_forward_and_zero(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ zero_buf,
    int32_t n)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    float sum = 0.0f;
    int start = offsets[v];
    int end = offsets[v + 1];
    for (int e = start; e < end; e++) {
        sum += x[__ldg(&indices[e])];
    }
    y[v] = sum;
    zero_buf[v] = 0.0f;
}

__global__ void spmv_transpose_scatter(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t n)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    float val = x[v];
    int start = offsets[v];
    int end = offsets[v + 1];
    for (int e = start; e < end; e++) {
        atomicAdd(&y[__ldg(&indices[e])], val);
    }
}



__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    if (val <= 0.0f) return;
    int* addr_as_i = (int*)addr;
    int old = *addr_as_i, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) return;
        old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
    } while (assumed != old);
}

__global__ void reduce_two_max_abs_kernel(
    const float* __restrict__ data1, float* __restrict__ result1,
    const float* __restrict__ data2, float* __restrict__ result2,
    int32_t n)
{
    __shared__ float s1[256];
    __shared__ float s2[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 4) + tid;

    float v1 = 0.0f, v2 = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int idx = i + j * 256;
        if (idx < n) {
            v1 = fmaxf(v1, fabsf(data1[idx]));
            v2 = fmaxf(v2, fabsf(data2[idx]));
        }
    }
    s1[tid] = v1; s2[tid] = v2;
    __syncthreads();

    if (tid < 128) { s1[tid] = fmaxf(s1[tid], s1[tid+128]); s2[tid] = fmaxf(s2[tid], s2[tid+128]); } __syncthreads();
    if (tid < 64)  { s1[tid] = fmaxf(s1[tid], s1[tid+64]);  s2[tid] = fmaxf(s2[tid], s2[tid+64]);  } __syncthreads();
    if (tid < 32) {
        float a = fmaxf(s1[tid], s1[tid+32]);
        float b = fmaxf(s2[tid], s2[tid+32]);
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 16));
        b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, 16));
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 8));
        b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, 8));
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 4));
        b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, 4));
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 2));
        b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, 2));
        a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 1));
        b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, 1));
        if (tid == 0) { atomicMaxFloat(result1, a); atomicMaxFloat(result2, b); }
    }
}

__global__ void normalize_diff_sum_kernel(
    float* __restrict__ hubs,
    const float* __restrict__ old_hubs,
    float* __restrict__ auth,
    const float* __restrict__ hub_max_ptr,
    const float* __restrict__ auth_max_ptr,
    float* __restrict__ diff_sum_ptr,
    int32_t n)
{
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float hub_max = *hub_max_ptr;
    float auth_max = *auth_max_ptr;
    float local_diff = 0.0f;

    if (i < n) {
        float h = hubs[i];
        float new_h = (hub_max > 0.0f) ? (h / hub_max) : h;
        hubs[i] = new_h;
        local_diff = fabsf(new_h - old_hubs[i]);

        float a = auth[i];
        if (auth_max > 0.0f) auth[i] = a / auth_max;
    }

    sdata[tid] = local_diff;
    __syncthreads();

    if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads();
    if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads();
    if (tid < 32) {
        float v = sdata[tid] + sdata[tid + 32];
        v += __shfl_xor_sync(0xffffffff, v, 16);
        v += __shfl_xor_sync(0xffffffff, v, 8);
        v += __shfl_xor_sync(0xffffffff, v, 4);
        v += __shfl_xor_sync(0xffffffff, v, 2);
        v += __shfl_xor_sync(0xffffffff, v, 1);
        if (tid == 0) atomicAdd(diff_sum_ptr, v);
    }
}

__global__ void reduce_sum_kernel(const float* __restrict__ data, float* __restrict__ result, int32_t n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 4) + tid;
    float val = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int idx = i + j * 256;
        if (idx < n) val += data[idx];
    }
    sdata[tid] = val;
    __syncthreads();
    if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads();
    if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads();
    if (tid < 32) {
        float v = sdata[tid] + sdata[tid + 32];
        v += __shfl_xor_sync(0xffffffff, v, 16);
        v += __shfl_xor_sync(0xffffffff, v, 8);
        v += __shfl_xor_sync(0xffffffff, v, 4);
        v += __shfl_xor_sync(0xffffffff, v, 2);
        v += __shfl_xor_sync(0xffffffff, v, 1);
        if (tid == 0) atomicAdd(result, v);
    }
}

__global__ void fill_value_kernel(float* arr, float val, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = val;
}

__global__ void scale_by_sum_kernel(float* __restrict__ vec, const float* __restrict__ sum_ptr, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = *sum_ptr;
    if (s > 0.0f) vec[i] /= s;
}



static void launch_compact_count(const int32_t* offsets, const uint32_t* edge_mask,
                                  int32_t* count, int32_t n, cudaStream_t stream) {
    int block = 256; int grid = (n + block - 1) / block;
    compact_count_kernel<<<grid, block, 0, stream>>>(offsets, edge_mask, count, n);
}

static void launch_compact_scatter(const int32_t* old_offsets, const int32_t* old_indices,
                                    const uint32_t* edge_mask, const int32_t* new_offsets,
                                    int32_t* new_indices, int32_t n, cudaStream_t stream) {
    int block = 256; int grid = (n + block - 1) / block;
    compact_scatter_kernel<<<grid, block, 0, stream>>>(old_offsets, old_indices, edge_mask,
                                                        new_offsets, new_indices, n);
}

static void launch_spmv_forward_and_zero(const int32_t* offsets, const int32_t* indices,
                                          const float* x, float* y, float* zero_buf,
                                          int32_t n, cudaStream_t stream) {
    int block = 256; int grid = (n + block - 1) / block;
    spmv_forward_and_zero<<<grid, block, 0, stream>>>(offsets, indices, x, y, zero_buf, n);
}

static void launch_spmv_transpose(const int32_t* offsets, const int32_t* indices,
                                   const float* x, float* y, int32_t n, cudaStream_t stream) {
    int block = 256; int grid = (n + block - 1) / block;
    spmv_transpose_scatter<<<grid, block, 0, stream>>>(offsets, indices, x, y, n);
}

static void launch_two_max_abs(const float* d1, float* r1, const float* d2, float* r2,
                                int32_t n, cudaStream_t stream) {
    int block = 256; int grid = (n + block * 4 - 1) / (block * 4);
    reduce_two_max_abs_kernel<<<grid, block, 0, stream>>>(d1, r1, d2, r2, n);
}

static void launch_normalize_diff_sum(float* hubs, const float* old_hubs, float* auth,
                                       float* hub_max, float* auth_max, float* diff_sum,
                                       int32_t n, cudaStream_t stream) {
    int block = 256; int grid = (n + block - 1) / block;
    normalize_diff_sum_kernel<<<grid, block, 0, stream>>>(hubs, old_hubs, auth,
                                                           hub_max, auth_max, diff_sum, n);
}

static void launch_sum(const float* d_in, float* d_out, int32_t n, cudaStream_t stream) {
    cudaMemsetAsync(d_out, 0, sizeof(float), stream);
    int block = 256; int grid = (n + block * 4 - 1) / (block * 4);
    reduce_sum_kernel<<<grid, block, 0, stream>>>(d_in, d_out, n);
}

static void launch_fill(float* arr, float val, int32_t n, cudaStream_t stream) {
    int block = 256; int grid = (n + block - 1) / block;
    fill_value_kernel<<<grid, block, 0, stream>>>(arr, val, n);
}

static void launch_scale_by_sum(float* vec, const float* sum_ptr, int32_t n, cudaStream_t stream) {
    int block = 256; int grid = (n + block - 1) / block;
    scale_by_sum_kernel<<<grid, block, 0, stream>>>(vec, sum_ptr, n);
}


static void run_iteration(
    cudaStream_t stream,
    const int32_t* offsets, const int32_t* indices,
    float* hubs_read, float* hubs_write, float* auth,
    float* hub_max, float* auth_max, float* diff_sum,
    int32_t n)
{
    
    launch_spmv_forward_and_zero(offsets, indices, hubs_read, auth, hubs_write, n, stream);
    
    launch_spmv_transpose(offsets, indices, auth, hubs_write, n, stream);
    
    cudaMemsetAsync(hub_max, 0, 3 * sizeof(float), stream); 
    launch_two_max_abs(hubs_write, hub_max, auth, auth_max, n, stream);
    
    launch_normalize_diff_sum(hubs_write, hubs_read, auth, hub_max, auth_max, diff_sum, n, stream);
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

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n = graph.number_of_vertices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cudaStream_t stream = 0;

    
    cache.ensure_compaction(n + 1, 0);

    int32_t* d_count = cache.count;
    cudaMemsetAsync(d_count + n, 0, sizeof(int32_t), stream);
    launch_compact_count(d_offsets, d_edge_mask, d_count, n, stream);

    
    size_t scan_temp_bytes = 0;
    {
        int32_t* d = nullptr;
        cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes, d, d, n + 1);
    }
    cache.ensure_compaction(n + 1, scan_temp_bytes);

    int32_t* d_new_offsets = cache.new_offsets;
    cub::DeviceScan::ExclusiveSum(cache.scan_temp, scan_temp_bytes,
                                   d_count, d_new_offsets, n + 1, stream);

    int32_t h_num_active;
    cudaMemcpyAsync(&h_num_active, d_new_offsets + n, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int64_t active_edges = (h_num_active > 0) ? h_num_active : 1;
    cache.ensure_indices(active_edges);
    int32_t* d_new_indices = cache.new_indices;
    if (h_num_active > 0)
        launch_compact_scatter(d_offsets, d_indices, d_edge_mask, d_new_offsets, d_new_indices, n, stream);

    
    cache.ensure_iteration(n);

    float* d_a = cache.hubs_a;
    float* d_b = cache.hubs_b;
    float* d_auth = cache.auth;

    float* d_scalars = cache.scalars;
    float* d_hub_max = d_scalars;
    float* d_auth_max = d_scalars + 1;
    float* d_diff_sum = d_scalars + 2;
    float* d_temp_scalar = d_scalars + 3;

    
    if (has_initial_hubs_guess) {
        cudaMemcpyAsync(d_a, hubs, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        launch_sum(d_a, d_temp_scalar, n, stream);
        launch_scale_by_sum(d_a, d_temp_scalar, n, stream);
    } else {
        launch_fill(d_a, 1.0f / n, n, stream);
    }
    cudaStreamSynchronize(stream);

    float h_diff_sum = 0.0f;
    float tol = epsilon * n;
    bool converged = false;
    int64_t total_iters = 0;

    
    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        if (iter % 2 == 0)
            run_iteration(stream, d_new_offsets, d_new_indices, d_a, d_b, d_auth,
                         d_hub_max, d_auth_max, d_diff_sum, n);
        else
            run_iteration(stream, d_new_offsets, d_new_indices, d_b, d_a, d_auth,
                         d_hub_max, d_auth_max, d_diff_sum, n);
        total_iters++;

        cudaMemcpyAsync(&h_diff_sum, d_diff_sum, sizeof(float),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (h_diff_sum < tol) {
            converged = true;
            break;
        }
    }

    
    float* d_final_hubs = (total_iters % 2 == 0) ? d_a : d_b;

    
    if (normalize) {
        launch_sum(d_final_hubs, d_temp_scalar, n, stream);
        launch_scale_by_sum(d_final_hubs, d_temp_scalar, n, stream);
        launch_sum(d_auth, d_temp_scalar, n, stream);
        launch_scale_by_sum(d_auth, d_temp_scalar, n, stream);
    }

    
    cudaMemcpyAsync(hubs, d_final_hubs, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(authorities, d_auth, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    HitsResult result;
    result.iterations = static_cast<std::size_t>(total_iters);
    result.converged = converged;
    result.final_norm = h_diff_sum;
    return result;
}

}  
