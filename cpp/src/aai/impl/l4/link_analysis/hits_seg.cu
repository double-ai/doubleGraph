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

struct Cache : Cacheable {
    float* prev_hubs = nullptr;
    float* scratch = nullptr;
    int64_t prev_hubs_capacity = 0;
    int64_t scratch_capacity = 0;

    void ensure(int64_t N) {
        if (prev_hubs_capacity < N) {
            if (prev_hubs) cudaFree(prev_hubs);
            cudaMalloc(&prev_hubs, N * sizeof(float));
            prev_hubs_capacity = N;
        }
        if (scratch_capacity < 8) {
            if (scratch) cudaFree(scratch);
            cudaMalloc(&scratch, 8 * sizeof(float));
            scratch_capacity = 8;
        }
    }

    ~Cache() override {
        if (prev_hubs) cudaFree(prev_hubs);
        if (scratch) cudaFree(scratch);
    }
};







template<int BLOCK_SIZE>
__global__ void spmv_gather_block_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t seg_start, int32_t seg_end)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int v = blockIdx.x + seg_start;
    if (v >= seg_end) return;

    int start = offsets[v];
    int end = offsets[v + 1];

    float sum = 0.0f;
    for (int k = start + threadIdx.x; k < end; k += BLOCK_SIZE) {
        sum += x[indices[k]];
    }

    float block_sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) y[v] = block_sum;
}


__global__ void spmv_gather_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t seg_start, int32_t seg_end)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int v = wid + seg_start;

    if (v >= seg_end) return;

    int start = offsets[v];
    int end = offsets[v + 1];

    float sum = 0.0f;
    for (int k = start + lane; k < end; k += 32) {
        sum += x[indices[k]];
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, s);

    if (lane == 0) y[v] = sum;
}


__global__ void spmv_gather_thread_kernel(
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
    for (int k = start; k < end; k++) {
        sum += x[indices[k]];
    }
    y[v] = sum;
}







template<int BLOCK_SIZE>
__global__ void spmv_scatter_block_kernel(
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

    if (val != 0.0f) {
        for (int k = start + threadIdx.x; k < end; k += BLOCK_SIZE) {
            atomicAdd(&y[indices[k]], val);
        }
    }
}


__global__ void spmv_scatter_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t seg_start, int32_t seg_end)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int v = wid + seg_start;

    if (v >= seg_end) return;

    int start = offsets[v];
    int end = offsets[v + 1];
    float val = x[v];

    if (val != 0.0f) {
        for (int k = start + lane; k < end; k += 32) {
            atomicAdd(&y[indices[k]], val);
        }
    }
}


__global__ void spmv_scatter_thread_kernel(
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

    if (val != 0.0f) {
        for (int k = start; k < end; k++) {
            atomicAdd(&y[indices[k]], val);
        }
    }
}





__global__ void max_abs_kernel(
    const float* __restrict__ data,
    int* __restrict__ d_max_int,
    int N)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float thread_max = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        thread_max = fmaxf(thread_max, fabsf(data[i]));
    }

    float block_max = BlockReduce(temp).Reduce(thread_max,
        [] __device__ (float a, float b) { return fmaxf(a, b); });

    if (threadIdx.x == 0 && block_max > 0.0f) {
        atomicMax(d_max_int, __float_as_int(block_max));
    }
}

__global__ void normalize_and_diff_kernel(
    float* __restrict__ curr,
    const float* __restrict__ prev,
    const float* __restrict__ max_val_ptr,
    float* __restrict__ diff_out,
    int N)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float m = *max_val_ptr;
    float inv_max = (m > 0.0f) ? (1.0f / m) : 1.0f;

    float thread_diff = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        float val = curr[i] * inv_max;
        curr[i] = val;
        thread_diff += fabsf(val - prev[i]);
    }

    float block_diff = BlockReduce(temp).Sum(thread_diff);
    if (threadIdx.x == 0) atomicAdd(diff_out, block_diff);
}

__global__ void max_normalize_kernel(
    float* __restrict__ data,
    const float* __restrict__ max_val_ptr,
    int N)
{
    float m = *max_val_ptr;
    float inv_max = (m > 0.0f) ? (1.0f / m) : 1.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        data[i] *= inv_max;
    }
}

__global__ void l1_sum_kernel(
    const float* __restrict__ data,
    float* __restrict__ result,
    int N)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float thread_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        thread_sum += data[i];
    }

    float block_sum = BlockReduce(temp).Sum(thread_sum);
    if (threadIdx.x == 0) atomicAdd(result, block_sum);
}

__global__ void divide_kernel(
    float* __restrict__ data,
    const float* __restrict__ scalar,
    int N)
{
    float s = *scalar;
    float inv_s = (s > 0.0f) ? (1.0f / s) : 1.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        data[i] *= inv_s;
    }
}




static inline void dispatch_spmv_gather(
    const int32_t* offsets, const int32_t* indices,
    const float* x, float* y,
    const int32_t* seg,
    cudaStream_t stream)
{
    const int BLOCK = 256;

    int n_high = seg[1] - seg[0];
    if (n_high > 0) {
        spmv_gather_block_kernel<BLOCK><<<n_high, BLOCK, 0, stream>>>(
            offsets, indices, x, y, seg[0], seg[1]);
    }

    int n_mid = seg[2] - seg[1];
    if (n_mid > 0) {
        int warps_per_block = BLOCK / 32;
        int grid = (n_mid + warps_per_block - 1) / warps_per_block;
        spmv_gather_warp_kernel<<<grid, BLOCK, 0, stream>>>(
            offsets, indices, x, y, seg[1], seg[2]);
    }

    int n_low = seg[3] - seg[2];
    if (n_low > 0) {
        int grid = (n_low + BLOCK - 1) / BLOCK;
        spmv_gather_thread_kernel<<<grid, BLOCK, 0, stream>>>(
            offsets, indices, x, y, seg[2], seg[3]);
    }

    int n_zero = seg[4] - seg[3];
    if (n_zero > 0) {
        cudaMemsetAsync(y + seg[3], 0, n_zero * sizeof(float), stream);
    }
}

static inline void dispatch_spmv_scatter(
    const int32_t* offsets, const int32_t* indices,
    const float* x, float* y, int32_t N,
    const int32_t* seg,
    cudaStream_t stream)
{
    const int BLOCK = 256;

    cudaMemsetAsync(y, 0, N * sizeof(float), stream);

    int n_high = seg[1] - seg[0];
    if (n_high > 0) {
        spmv_scatter_block_kernel<BLOCK><<<n_high, BLOCK, 0, stream>>>(
            offsets, indices, x, y, seg[0], seg[1]);
    }

    int n_mid = seg[2] - seg[1];
    if (n_mid > 0) {
        int warps_per_block = BLOCK / 32;
        int grid = (n_mid + warps_per_block - 1) / warps_per_block;
        spmv_scatter_warp_kernel<<<grid, BLOCK, 0, stream>>>(
            offsets, indices, x, y, seg[1], seg[2]);
    }

    int n_low = seg[3] - seg[2];
    if (n_low > 0) {
        int grid = (n_low + BLOCK - 1) / BLOCK;
        spmv_scatter_thread_kernel<<<grid, BLOCK, 0, stream>>>(
            offsets, indices, x, y, seg[2], seg[3]);
    }
}




__global__ void fill_kernel(float* data, float val, int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        data[i] = val;
    }
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

    const int32_t* offsets_csc = graph.offsets;
    const int32_t* indices_csc = graph.indices;
    int32_t N = graph.number_of_vertices;

    if (N == 0) {
        return HitsResult{
            max_iterations > static_cast<std::size_t>(INT64_MAX)
                ? static_cast<std::size_t>(INT64_MAX) : max_iterations,
            false,
            FLT_MAX
        };
    }

    cache.ensure(N);

    const auto& seg_vec = graph.segment_offsets.value();
    int32_t seg[5] = {seg_vec[0], seg_vec[1], seg_vec[2], seg_vec[3], seg_vec[4]};

    cudaStream_t stream = 0;
    float tolerance = static_cast<float>(N) * epsilon;

    const int BLOCK = 256;
    int reduce_grid = (N + BLOCK - 1) / BLOCK;
    if (reduce_grid > 1024) reduce_grid = 1024;
    int norm_grid = (N + BLOCK - 1) / BLOCK;

    float* scratch = cache.scratch;
    float* d_max_val = scratch;
    float* d_diff    = scratch + 1;
    float* d_l1_sum  = scratch + 2;

    
    
    
    float* prev = hubs;
    float* curr = cache.prev_hubs;

    if (has_initial_hubs_guess) {
        
        cudaMemsetAsync(d_l1_sum, 0, sizeof(float), stream);
        l1_sum_kernel<<<reduce_grid, BLOCK, 0, stream>>>(prev, d_l1_sum, N);
        divide_kernel<<<norm_grid, BLOCK, 0, stream>>>(prev, d_l1_sum, N);
    } else {
        float inv_n = 1.0f / static_cast<float>(N);
        int fill_grid = (N + BLOCK - 1) / BLOCK;
        if (fill_grid > 65535) fill_grid = 65535;
        fill_kernel<<<fill_grid, BLOCK, 0, stream>>>(prev, inv_n, N);
    }

    
    
    
    float diff_sum = FLT_MAX;
    std::size_t iter = 0;

    while (iter < max_iterations) {
        
        dispatch_spmv_gather(offsets_csc, indices_csc, prev, authorities, seg, stream);

        
        dispatch_spmv_scatter(offsets_csc, indices_csc, authorities, curr, N, seg, stream);

        
        cudaMemsetAsync(d_max_val, 0, sizeof(float), stream);
        max_abs_kernel<<<reduce_grid, BLOCK, 0, stream>>>(curr, (int*)d_max_val, N);

        cudaMemsetAsync(d_diff, 0, sizeof(float), stream);
        normalize_and_diff_kernel<<<reduce_grid, BLOCK, 0, stream>>>(
            curr, prev, d_max_val, d_diff, N);

        
        cudaMemcpyAsync(&diff_sum, d_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        
        float* tmp = prev; prev = curr; curr = tmp;

        iter++;
        if (diff_sum < tolerance) break;
    }

    
    
    

    
    cudaMemsetAsync(d_max_val, 0, sizeof(float), stream);
    max_abs_kernel<<<reduce_grid, BLOCK, 0, stream>>>(authorities, (int*)d_max_val, N);
    max_normalize_kernel<<<norm_grid, BLOCK, 0, stream>>>(authorities, d_max_val, N);

    if (normalize) {
        cudaMemsetAsync(d_l1_sum, 0, sizeof(float), stream);
        l1_sum_kernel<<<reduce_grid, BLOCK, 0, stream>>>(prev, d_l1_sum, N);
        divide_kernel<<<norm_grid, BLOCK, 0, stream>>>(prev, d_l1_sum, N);

        cudaMemsetAsync(d_l1_sum, 0, sizeof(float), stream);
        l1_sum_kernel<<<reduce_grid, BLOCK, 0, stream>>>(authorities, d_l1_sum, N);
        divide_kernel<<<norm_grid, BLOCK, 0, stream>>>(authorities, d_l1_sum, N);
    }

    if (prev != hubs) {
        cudaMemcpyAsync(hubs, prev, N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    return HitsResult{iter, diff_sum < tolerance, diff_sum};
}

}  
