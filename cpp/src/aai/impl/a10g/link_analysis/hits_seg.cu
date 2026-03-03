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
#include <limits>

namespace aai {

namespace {





struct Cache : Cacheable {
    float* hubs_scratch = nullptr;
    float* d_scalars = nullptr;  
    int64_t hubs_scratch_capacity = 0;
    bool scalars_allocated = false;

    void ensure(int32_t num_vertices) {
        if (hubs_scratch_capacity < num_vertices) {
            if (hubs_scratch) cudaFree(hubs_scratch);
            cudaMalloc(&hubs_scratch, (size_t)num_vertices * sizeof(float));
            hubs_scratch_capacity = num_vertices;
        }
        if (!scalars_allocated) {
            cudaMalloc(&d_scalars, 2 * sizeof(float));
            scalars_allocated = true;
        }
    }

    ~Cache() override {
        if (hubs_scratch) cudaFree(hubs_scratch);
        if (d_scalars) cudaFree(d_scalars);
    }
};






__global__ void spmv_gather_block(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t start_v, int32_t num_v
) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    if (blockIdx.x >= (unsigned)num_v) return;
    int v = blockIdx.x + start_v;
    int row_start = __ldg(&offsets[v]);
    int row_end = __ldg(&offsets[v + 1]);
    float sum = 0.0f;
    for (int j = row_start + threadIdx.x; j < row_end; j += 256)
        sum += __ldg(&x[__ldg(&indices[j])]);
    float block_sum = BlockReduce(temp_storage).Sum(sum);
    if (threadIdx.x == 0) y[v] = block_sum;
}


__global__ void spmv_gather_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t start_v, int32_t num_v
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (warp_id >= num_v) return;
    int v = warp_id + start_v;
    int row_start = __ldg(&offsets[v]);
    int row_end = __ldg(&offsets[v + 1]);
    float sum = 0.0f;
    for (int j = row_start + lane; j < row_end; j += 32)
        sum += __ldg(&x[__ldg(&indices[j])]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    if (lane == 0) y[v] = sum;
}


__global__ void spmv_gather_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t start_v, int32_t num_v
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_v) return;
    int v = tid + start_v;
    int row_start = __ldg(&offsets[v]);
    int row_end = __ldg(&offsets[v + 1]);
    float sum = 0.0f;
    for (int j = row_start; j < row_end; j++)
        sum += __ldg(&x[__ldg(&indices[j])]);
    y[v] = sum;
}





__global__ void spmv_scatter_block(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t start_v, int32_t num_v
) {
    if (blockIdx.x >= (unsigned)num_v) return;
    int v = blockIdx.x + start_v;
    int row_start = __ldg(&offsets[v]);
    int row_end = __ldg(&offsets[v + 1]);
    float val = x[v];
    for (int j = row_start + threadIdx.x; j < row_end; j += 256)
        atomicAdd(&y[__ldg(&indices[j])], val);
}

__global__ void spmv_scatter_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t start_v, int32_t num_v
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (warp_id >= num_v) return;
    int v = warp_id + start_v;
    int row_start = __ldg(&offsets[v]);
    int row_end = __ldg(&offsets[v + 1]);
    float val = x[v];
    for (int j = row_start + lane; j < row_end; j += 32)
        atomicAdd(&y[__ldg(&indices[j])], val);
}

__global__ void spmv_scatter_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ y,
    int32_t start_v, int32_t num_v
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_v) return;
    int v = tid + start_v;
    int row_start = __ldg(&offsets[v]);
    int row_end = __ldg(&offsets[v + 1]);
    float val = x[v];
    for (int j = row_start; j < row_end; j++)
        atomicAdd(&y[__ldg(&indices[j])], val);
}





__device__ float atomicMaxFloat(float* address, float val) {
    int* addr_as_i = (int*)address;
    int old = *addr_as_i, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= val) return __int_as_float(old);
        old = atomicCAS(addr_as_i, assumed, __float_as_int(val));
    } while (assumed != old);
    return __int_as_float(old);
}

template<int BLOCK_SIZE>
__global__ void max_abs_kernel(const float* __restrict__ arr, float* __restrict__ result, int32_t n) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float local_max = 0.0f;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        float v = fabsf(arr[i]);
        local_max = fmaxf(local_max, v);
    }
    float block_max = BlockReduce(temp_storage).Reduce(local_max, ::cuda::maximum<float>{});
    if (threadIdx.x == 0 && block_max > 0.0f)
        atomicMaxFloat(result, block_max);
}

template<int BLOCK_SIZE>
__global__ void normalize_diff_kernel(
    float* __restrict__ curr,
    const float* __restrict__ prev,
    const float* __restrict__ d_max,
    float* __restrict__ d_diff,
    int32_t n
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float max_val = *d_max;
    float inv_max = (max_val > 0.0f) ? (1.0f / max_val) : 1.0f;
    float local_diff = 0.0f;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        float normalized = curr[i] * inv_max;
        curr[i] = normalized;
        local_diff += fabsf(normalized - prev[i]);
    }
    float block_sum = BlockReduce(temp_storage).Sum(local_diff);
    if (threadIdx.x == 0 && block_sum > 0.0f)
        atomicAdd(d_diff, block_sum);
}

__global__ void init_val_kernel(float* __restrict__ arr, float val, int32_t n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        arr[i] = val;
}

__global__ void scale_kernel(float* __restrict__ arr, float scale, int32_t n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        arr[i] *= scale;
}

template<int BLOCK_SIZE>
__global__ void sum_kernel(const float* __restrict__ arr, float* __restrict__ result, int32_t n) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    float local_sum = 0.0f;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE)
        local_sum += arr[i];
    float block_sum = BlockReduce(temp_storage).Sum(local_sum);
    if (threadIdx.x == 0 && block_sum != 0.0f)
        atomicAdd(result, block_sum);
}





static inline int capped_grid(int n, int block_size) {
    int full = (n + block_size - 1) / block_size;
    return (full < 240) ? full : 240;
}

static void launch_spmv_gather_segmented(
    const int32_t* offsets, const int32_t* indices,
    const float* x, float* y, int32_t num_vertices,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3, int32_t seg4
) {
    int n_high = seg1 - seg0;
    if (n_high > 0)
        spmv_gather_block<<<n_high, 256>>>(offsets, indices, x, y, seg0, n_high);
    int n_mid = seg2 - seg1;
    if (n_mid > 0) {
        int wpb = 8;
        spmv_gather_warp<<<(n_mid + wpb - 1) / wpb, wpb * 32>>>(offsets, indices, x, y, seg1, n_mid);
    }
    int n_low = seg4 - seg2;
    if (n_low > 0)
        spmv_gather_thread<<<(n_low + 255) / 256, 256>>>(offsets, indices, x, y, seg2, n_low);
}

static void launch_spmv_scatter_segmented(
    const int32_t* offsets, const int32_t* indices,
    const float* x, float* y, int32_t num_vertices,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3, int32_t seg4
) {
    int n_high = seg1 - seg0;
    if (n_high > 0)
        spmv_scatter_block<<<n_high, 256>>>(offsets, indices, x, y, seg0, n_high);
    int n_mid = seg2 - seg1;
    if (n_mid > 0) {
        int wpb = 8;
        spmv_scatter_warp<<<(n_mid + wpb - 1) / wpb, wpb * 32>>>(offsets, indices, x, y, seg1, n_mid);
    }
    int n_low = seg3 - seg2;
    if (n_low > 0)
        spmv_scatter_thread<<<(n_low + 255) / 256, 256>>>(offsets, indices, x, y, seg2, n_low);
}

static void launch_init_val(float* arr, float val, int32_t n) {
    if (n <= 0) return;
    init_val_kernel<<<capped_grid(n, 256), 256>>>(arr, val, n);
}

static void launch_max_abs(const float* arr, float* result, int32_t n) {
    if (n <= 0) return;
    max_abs_kernel<256><<<capped_grid(n, 256), 256>>>(arr, result, n);
}

static void launch_normalize_diff(float* curr, const float* prev,
                                   const float* d_max, float* d_diff, int32_t n) {
    if (n <= 0) return;
    normalize_diff_kernel<256><<<capped_grid(n, 256), 256>>>(curr, prev, d_max, d_diff, n);
}

static void launch_scale(float* arr, float scale, int32_t n) {
    if (n <= 0) return;
    scale_kernel<<<capped_grid(n, 256), 256>>>(arr, scale, n);
}

static void launch_sum(const float* arr, float* result, int32_t n) {
    if (n <= 0) return;
    sum_kernel<256><<<capped_grid(n, 256), 256>>>(arr, result, n);
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
        return HitsResult{max_iterations, false, std::numeric_limits<float>::max()};
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

    float tolerance = static_cast<float>(num_vertices) * epsilon;

    float* d_max = cache.d_scalars;
    float* d_diff = cache.d_scalars + 1;

    
    float* hubs_buf[2] = {hubs, cache.hubs_scratch};
    float* d_auth = authorities;
    int curr = 0;

    
    if (has_initial_hubs_guess) {
        
        
        cudaMemsetAsync(d_diff, 0, sizeof(float));
        launch_sum(hubs_buf[curr], d_diff, num_vertices);
        float h_sum;
        cudaMemcpy(&h_sum, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_sum > 0.0f) {
            launch_scale(hubs_buf[curr], 1.0f / h_sum, num_vertices);
        }
    } else {
        launch_init_val(hubs_buf[curr], 1.0f / num_vertices, num_vertices);
    }

    std::size_t iterations = 0;
    bool converged = false;
    float final_norm = 0.0f;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        int next = 1 - curr;

        
        launch_spmv_gather_segmented(d_offsets, d_indices, hubs_buf[curr], d_auth,
                                     num_vertices, seg0, seg1, seg2, seg3, seg4);

        
        cudaMemsetAsync(hubs_buf[next], 0, num_vertices * sizeof(float));
        launch_spmv_scatter_segmented(d_offsets, d_indices, d_auth, hubs_buf[next],
                                      num_vertices, seg0, seg1, seg2, seg3, seg4);

        
        cudaMemsetAsync(d_max, 0, sizeof(float));
        launch_max_abs(hubs_buf[next], d_max, num_vertices);

        
        cudaMemsetAsync(d_diff, 0, sizeof(float));
        launch_normalize_diff(hubs_buf[next], hubs_buf[curr], d_max, d_diff, num_vertices);

        
        float h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);

        curr = next;
        iterations = iter + 1;
        final_norm = h_diff;

        if (h_diff < tolerance) {
            converged = true;
            break;
        }
    }

    
    cudaMemsetAsync(d_max, 0, sizeof(float));
    launch_max_abs(d_auth, d_max, num_vertices);
    float h_auth_max;
    cudaMemcpy(&h_auth_max, d_max, sizeof(float), cudaMemcpyDeviceToHost);
    if (h_auth_max > 0.0f) {
        launch_scale(d_auth, 1.0f / h_auth_max, num_vertices);
    }

    
    if (normalize) {
        cudaMemsetAsync(d_diff, 0, sizeof(float));
        launch_sum(hubs_buf[curr], d_diff, num_vertices);
        float h_sum;
        cudaMemcpy(&h_sum, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_sum > 0.0f) {
            launch_scale(hubs_buf[curr], 1.0f / h_sum, num_vertices);
        }

        cudaMemsetAsync(d_diff, 0, sizeof(float));
        launch_sum(d_auth, d_diff, num_vertices);
        cudaMemcpy(&h_sum, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_sum > 0.0f) {
            launch_scale(d_auth, 1.0f / h_sum, num_vertices);
        }
    }

    
    if (hubs_buf[curr] != hubs) {
        cudaMemcpy(hubs, hubs_buf[curr],
                   num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return HitsResult{iterations, converged, final_norm};
}

}  
