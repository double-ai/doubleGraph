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
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* temp_hubs = nullptr;
    float* scratch = nullptr;
    int64_t temp_hubs_capacity = 0;
    int64_t scratch_capacity = 0;

    void ensure(int32_t num_vertices) {
        if (temp_hubs_capacity < num_vertices) {
            if (temp_hubs) cudaFree(temp_hubs);
            cudaMalloc(&temp_hubs, num_vertices * sizeof(float));
            temp_hubs_capacity = num_vertices;
        }
        if (scratch_capacity < 4) {
            if (scratch) cudaFree(scratch);
            cudaMalloc(&scratch, 4 * sizeof(float));
            scratch_capacity = 4;
        }
    }

    ~Cache() override {
        if (temp_hubs) cudaFree(temp_hubs);
        if (scratch) cudaFree(scratch);
    }
};





__global__ void spmv_forward_high(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t seg_start, int32_t seg_end)
{
    int v = blockIdx.x + seg_start;
    if (v >= seg_end) return;

    int start = offsets[v];
    int end = offsets[v + 1];

    float sum = 0.0f;
    for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
        sum += x[__ldg(&indices[k])];
    }

    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    sum = BlockReduce(temp_storage).Sum(sum);

    if (threadIdx.x == 0) {
        y[v] = sum;
    }
}

__global__ void spmv_forward_mid(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t seg_start, int32_t seg_end)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int v = warp_id + seg_start;
    if (v >= seg_end) return;

    int start = offsets[v];
    int end = offsets[v + 1];

    float sum = 0.0f;
    for (int k = start + lane; k < end; k += 32) {
        sum += x[__ldg(&indices[k])];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        y[v] = sum;
    }
}

__global__ void spmv_forward_low(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t seg_start, int32_t seg_end)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x + seg_start;
    if (v >= seg_end) return;

    int start = offsets[v];
    int end = offsets[v + 1];

    float sum = 0.0f;
    for (int k = start; k < end; ++k) {
        sum += x[__ldg(&indices[k])];
    }
    y[v] = sum;
}





__global__ void spmv_transpose_high(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t seg_start, int32_t seg_end)
{
    int v = blockIdx.x + seg_start;
    if (v >= seg_end) return;

    int start = offsets[v];
    int end = offsets[v + 1];
    float val = x[v];

    for (int k = start + threadIdx.x; k < end; k += blockDim.x) {
        atomicAdd(&y[__ldg(&indices[k])], val);
    }
}

__global__ void spmv_transpose_mid(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t seg_start, int32_t seg_end)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int v = warp_id + seg_start;
    if (v >= seg_end) return;

    int start = offsets[v];
    int end = offsets[v + 1];
    float val = x[v];

    for (int k = start + lane; k < end; k += 32) {
        atomicAdd(&y[__ldg(&indices[k])], val);
    }
}

__global__ void spmv_transpose_low(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t seg_start, int32_t seg_end)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x + seg_start;
    if (v >= seg_end) return;

    int start = offsets[v];
    int end = offsets[v + 1];
    float val = x[v];

    for (int k = start; k < end; ++k) {
        atomicAdd(&y[__ldg(&indices[k])], val);
    }
}





__global__ void reduce_max_two(
    const float* __restrict__ hubs,
    const float* __restrict__ auth,
    int* __restrict__ hub_max_int,
    int* __restrict__ auth_max_int,
    int32_t n)
{
    float h_max = 0.0f;
    float a_max = 0.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        h_max = fmaxf(h_max, hubs[i]);
        a_max = fmaxf(a_max, auth[i]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        h_max = fmaxf(h_max, __shfl_down_sync(0xffffffff, h_max, offset));
        a_max = fmaxf(a_max, __shfl_down_sync(0xffffffff, a_max, offset));
    }

    if ((threadIdx.x & 31) == 0) {
        atomicMax(hub_max_int, __float_as_int(h_max));
        atomicMax(auth_max_int, __float_as_int(a_max));
    }
}

__global__ void normalize_and_diff(
    float* __restrict__ curr_hubs,
    const float* __restrict__ prev_hubs,
    float* __restrict__ auth,
    const int* __restrict__ hub_max_int_ptr,
    const int* __restrict__ auth_max_int_ptr,
    float* __restrict__ diff_out,
    int32_t n)
{
    float hub_max = __int_as_float(*hub_max_int_ptr);
    float auth_max = __int_as_float(*auth_max_int_ptr);

    float inv_hub_max = (hub_max > 0.0f) ? (1.0f / hub_max) : 0.0f;
    float inv_auth_max = (auth_max > 0.0f) ? (1.0f / auth_max) : 0.0f;

    float local_diff = 0.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float h = curr_hubs[i] * inv_hub_max;
        curr_hubs[i] = h;
        auth[i] *= inv_auth_max;
        local_diff += fabsf(h - prev_hubs[i]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_diff += __shfl_down_sync(0xffffffff, local_diff, offset);
    }

    if ((threadIdx.x & 31) == 0) {
        atomicAdd(diff_out, local_diff);
    }
}





__global__ void fill_uniform(float* data, int32_t n, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = val;
    }
}

__global__ void reduce_sum_kernel(const float* __restrict__ data, float* __restrict__ result, int32_t n) {
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        sum += data[i];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if ((threadIdx.x & 31) == 0) {
        atomicAdd(result, sum);
    }
}

__global__ void divide_by_scalar(float* data, const float* scalar_ptr, int32_t n) {
    float inv_s = 1.0f / (*scalar_ptr);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] *= inv_s;
    }
}

__global__ void zero_range_kernel(float* y, int32_t start, int32_t end) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (i < end) {
        y[i] = 0.0f;
    }
}





static void launch_spmv_forward(
    const int32_t* offsets, const int32_t* indices,
    const float* x, float* y, int32_t num_vertices,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3, int32_t seg4,
    cudaStream_t stream)
{
    int n_high = seg1 - seg0;
    if (n_high > 0) {
        spmv_forward_high<<<n_high, 256, 0, stream>>>(offsets, indices, x, y, seg0, seg1);
    }

    int n_mid = seg2 - seg1;
    if (n_mid > 0) {
        int warps_per_block = 8;
        int blocks = (n_mid + warps_per_block - 1) / warps_per_block;
        spmv_forward_mid<<<blocks, 256, 0, stream>>>(offsets, indices, x, y, seg1, seg2);
    }

    int n_low = seg3 - seg2;
    if (n_low > 0) {
        int blocks = (n_low + 255) / 256;
        spmv_forward_low<<<blocks, 256, 0, stream>>>(offsets, indices, x, y, seg2, seg3);
    }

    int n_zero = seg4 - seg3;
    if (n_zero > 0) {
        int blocks = (n_zero + 255) / 256;
        zero_range_kernel<<<blocks, 256, 0, stream>>>(y, seg3, seg4);
    }
}

static void launch_spmv_transpose(
    const int32_t* offsets, const int32_t* indices,
    const float* x, float* y, int32_t num_vertices,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3,
    cudaStream_t stream)
{
    int n_high = seg1 - seg0;
    if (n_high > 0) {
        spmv_transpose_high<<<n_high, 256, 0, stream>>>(offsets, indices, x, y, seg0, seg1);
    }

    int n_mid = seg2 - seg1;
    if (n_mid > 0) {
        int warps_per_block = 8;
        int blocks = (n_mid + warps_per_block - 1) / warps_per_block;
        spmv_transpose_mid<<<blocks, 256, 0, stream>>>(offsets, indices, x, y, seg1, seg2);
    }

    int n_low = seg3 - seg2;
    if (n_low > 0) {
        int blocks = (n_low + 255) / 256;
        spmv_transpose_low<<<blocks, 256, 0, stream>>>(offsets, indices, x, y, seg2, seg3);
    }
}

static void launch_reduce_max_two(
    const float* hubs, const float* auth,
    int* hub_max, int* auth_max,
    int32_t n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    reduce_max_two<<<blocks, threads, 0, stream>>>(hubs, auth, hub_max, auth_max, n);
}

static void launch_normalize_and_diff(
    float* curr_hubs, const float* prev_hubs, float* auth,
    const int* hub_max, const int* auth_max, float* diff_out,
    int32_t n, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    normalize_and_diff<<<blocks, threads, 0, stream>>>(
        curr_hubs, prev_hubs, auth, hub_max, auth_max, diff_out, n);
}

static void launch_fill_uniform(float* data, int32_t n, float val, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    fill_uniform<<<blocks, 256, 0, stream>>>(data, n, val);
}

static void launch_reduce_sum(const float* data, float* result, int32_t n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;
    reduce_sum_kernel<<<blocks, threads, 0, stream>>>(data, result, n);
}

static void launch_divide_by_scalar(float* data, const float* scalar_ptr, int32_t n, cudaStream_t stream) {
    int blocks = (n + 255) / 256;
    divide_by_scalar<<<blocks, 256, 0, stream>>>(data, scalar_ptr, n);
}

}  

HitsResult hits_seg(const graph32_t& graph,
                    float* hubs,
                    float* authorities,
                    float epsilon,
                    std::size_t max_iterations,
                    bool has_initial_hubs_guess,
                    bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;

    if (num_vertices == 0) {
        return HitsResult{max_iterations, false, FLT_MAX};
    }

    cache.ensure(num_vertices);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0];
    int32_t seg1 = seg[1];
    int32_t seg2 = seg[2];
    int32_t seg3 = seg[3];
    int32_t seg4 = seg[4];

    cudaStream_t stream = 0;

    float* d_hubs = hubs;
    float* d_temp = cache.temp_hubs;
    float* d_auth = authorities;
    float* d_scratch = cache.scratch;
    int* d_scratch_int = reinterpret_cast<int*>(d_scratch);

    float tolerance = static_cast<float>(num_vertices) * epsilon;

    float* prev_h = d_hubs;
    float* curr_h = d_temp;

    if (has_initial_hubs_guess) {
        
        cudaMemsetAsync(d_scratch, 0, sizeof(float), stream);
        launch_reduce_sum(prev_h, d_scratch, num_vertices, stream);
        launch_divide_by_scalar(prev_h, d_scratch, num_vertices, stream);
    } else {
        float init_val = 1.0f / static_cast<float>(num_vertices);
        launch_fill_uniform(prev_h, num_vertices, init_val, stream);
    }

    std::size_t iterations = max_iterations;
    float final_norm = FLT_MAX;

    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        
        launch_spmv_forward(d_offsets, d_indices, prev_h, d_auth,
                           num_vertices, seg0, seg1, seg2, seg3, seg4, stream);

        
        cudaMemsetAsync(curr_h, 0, num_vertices * sizeof(float), stream);

        
        launch_spmv_transpose(d_offsets, d_indices, d_auth, curr_h,
                             num_vertices, seg0, seg1, seg2, seg3, stream);

        
        cudaMemsetAsync(d_scratch, 0, 3 * sizeof(float), stream);

        
        launch_reduce_max_two(curr_h, d_auth, d_scratch_int, d_scratch_int + 1,
                             num_vertices, stream);

        
        launch_normalize_and_diff(curr_h, prev_h, d_auth,
                                 d_scratch_int, d_scratch_int + 1,
                                 d_scratch + 2, num_vertices, stream);

        
        float diff;
        cudaMemcpyAsync(&diff, d_scratch + 2, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::swap(prev_h, curr_h);

        final_norm = diff;
        if (diff < tolerance) {
            iterations = iter + 1;
            break;
        }
    }

    
    if (normalize) {
        cudaMemsetAsync(d_scratch, 0, sizeof(float), stream);
        launch_reduce_sum(prev_h, d_scratch, num_vertices, stream);
        launch_divide_by_scalar(prev_h, d_scratch, num_vertices, stream);

        cudaMemsetAsync(d_scratch, 0, sizeof(float), stream);
        launch_reduce_sum(d_auth, d_scratch, num_vertices, stream);
        launch_divide_by_scalar(d_auth, d_scratch, num_vertices, stream);
    }

    
    if (prev_h != d_hubs) {
        cudaMemcpyAsync(d_hubs, prev_h, num_vertices * sizeof(float),
                       cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    bool converged = (final_norm < tolerance);
    return HitsResult{iterations, converged, final_norm};
}

}  
