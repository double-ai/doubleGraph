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
#include <algorithm>

namespace aai {

namespace {





struct Cache : Cacheable {
    float* hub_tmp = nullptr;
    float* diff = nullptr;
    float* scalars = nullptr;
    void* cub_temp = nullptr;

    int64_t hub_tmp_cap = 0;
    int64_t diff_cap = 0;
    size_t cub_temp_cap = 0;

    void ensure(int32_t N, size_t cub_size) {
        if (hub_tmp_cap < N) {
            if (hub_tmp) cudaFree(hub_tmp);
            cudaMalloc(&hub_tmp, (size_t)N * sizeof(float));
            hub_tmp_cap = N;
        }
        if (diff_cap < N) {
            if (diff) cudaFree(diff);
            cudaMalloc(&diff, (size_t)N * sizeof(float));
            diff_cap = N;
        }
        if (!scalars) {
            cudaMalloc(&scalars, 2 * sizeof(float));
        }
        if (cub_temp_cap < cub_size) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, cub_size);
            cub_temp_cap = cub_size;
        }
    }

    ~Cache() override {
        if (hub_tmp) cudaFree(hub_tmp);
        if (diff) cudaFree(diff);
        if (scalars) cudaFree(scalars);
        if (cub_temp) cudaFree(cub_temp);
    }
};





__device__ __forceinline__ bool edge_active(const uint32_t* mask, int32_t e) {
    return (mask[e >> 5] >> (e & 31)) & 1u;
}

__global__ void fused_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ mask,
    const float* __restrict__ hubs,
    float* __restrict__ auth,
    float* __restrict__ new_hubs,
    int32_t vstart, int32_t vend)
{
    int32_t v = vstart + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= vend) return;
    int32_t s = offsets[v], e = offsets[v + 1];
    float sum = 0.0f;
    for (int32_t i = s; i < e; i++)
        if (edge_active(mask, i)) sum += hubs[indices[i]];
    auth[v] = sum;
    for (int32_t i = s; i < e; i++)
        if (edge_active(mask, i)) atomicAdd(&new_hubs[indices[i]], sum);
}

__global__ void fused_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ mask,
    const float* __restrict__ hubs,
    float* __restrict__ auth,
    float* __restrict__ new_hubs,
    int32_t vstart, int32_t vend)
{
    int32_t wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;
    int32_t v = vstart + wid;
    if (v >= vend) return;
    int32_t s = offsets[v], e = offsets[v + 1];
    float sum = 0.0f;
    for (int32_t i = s + lane; i < e; i += 32)
        if (edge_active(mask, i)) sum += hubs[indices[i]];
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
    float a = __shfl_sync(0xFFFFFFFF, sum, 0);
    if (lane == 0) auth[v] = a;
    for (int32_t i = s + lane; i < e; i += 32)
        if (edge_active(mask, i)) atomicAdd(&new_hubs[indices[i]], a);
}

template <int BS>
__global__ void fused_block_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ mask,
    const float* __restrict__ hubs,
    float* __restrict__ auth,
    float* __restrict__ new_hubs,
    int32_t vstart, int32_t vend)
{
    typedef cub::BlockReduce<float, BS> BR;
    __shared__ typename BR::TempStorage tmp;
    __shared__ float s_auth;
    int32_t v = vstart + blockIdx.x;
    if (v >= vend) return;
    int32_t s = offsets[v], e = offsets[v + 1];
    float sum = 0.0f;
    for (int32_t i = s + threadIdx.x; i < e; i += BS)
        if (edge_active(mask, i)) sum += hubs[indices[i]];
    float total = BR(tmp).Sum(sum);
    if (threadIdx.x == 0) { auth[v] = total; s_auth = total; }
    __syncthreads();
    float a = s_auth;
    for (int32_t i = s + threadIdx.x; i < e; i += BS)
        if (edge_active(mask, i)) atomicAdd(&new_hubs[indices[i]], a);
}

__global__ void normalize_diff_kernel(
    float* __restrict__ data,
    const float* __restrict__ old_data,
    float* __restrict__ diff,
    int32_t n,
    const float* __restrict__ d_max_val)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float mx = *d_max_val;
    float val = (mx > 0.0f) ? data[i] / mx : data[i];
    data[i] = val;
    diff[i] = fabsf(val - old_data[i]);
}

__global__ void normalize_by_device_scalar(
    float* __restrict__ data, int32_t n, const float* __restrict__ d_div)
{
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float d = *d_div;
    if (d > 0.0f) data[i] /= d;
}

__global__ void scale_kernel(float* __restrict__ data, int32_t n, float s) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= s;
}

__global__ void fill_kernel(float* __restrict__ data, int32_t n, float v) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = v;
}





void launch_fused_auth_hub(
    const int32_t* offsets, const int32_t* indices, const uint32_t* mask,
    const float* hubs, float* auth, float* new_hubs,
    const int32_t* seg, cudaStream_t s)
{
    int32_t n;
    n = seg[1] - seg[0];
    if (n > 0)
        fused_block_kernel<256><<<n, 256, 0, s>>>(offsets, indices, mask, hubs, auth, new_hubs, seg[0], seg[1]);
    n = seg[2] - seg[1];
    if (n > 0) {
        int wpb = 8, grid = (n + wpb - 1) / wpb;
        fused_warp_kernel<<<grid, wpb * 32, 0, s>>>(offsets, indices, mask, hubs, auth, new_hubs, seg[1], seg[2]);
    }
    n = seg[3] - seg[2];
    if (n > 0) {
        int blk = 256, grid = (n + blk - 1) / blk;
        fused_thread_kernel<<<grid, blk, 0, s>>>(offsets, indices, mask, hubs, auth, new_hubs, seg[2], seg[3]);
    }
    n = seg[4] - seg[3];
    if (n > 0) {
        int blk = 256, grid = (n + blk - 1) / blk;
        fill_kernel<<<grid, blk, 0, s>>>(auth + seg[3], n, 0.0f);
    }
}

size_t get_cub_temp_size(int32_t n) {
    size_t s1 = 0, s2 = 0;
    cub::DeviceReduce::Max(nullptr, s1, (float*)nullptr, (float*)nullptr, n);
    cub::DeviceReduce::Sum(nullptr, s2, (float*)nullptr, (float*)nullptr, n);
    return (s1 > s2) ? s1 : s2;
}

void launch_reduce_max(const float* data, float* result, void* temp, size_t temp_size,
    int32_t n, cudaStream_t s) {
    cub::DeviceReduce::Max(temp, temp_size, data, result, n, s);
}

void launch_reduce_sum(const float* data, float* result, void* temp, size_t temp_size,
    int32_t n, cudaStream_t s) {
    cub::DeviceReduce::Sum(temp, temp_size, data, result, n, s);
}

void launch_normalize_diff(float* data, const float* old_data, float* diff,
    int32_t n, const float* d_max, cudaStream_t s) {
    int blk = 256, grid = (n + blk - 1) / blk;
    normalize_diff_kernel<<<grid, blk, 0, s>>>(data, old_data, diff, n, d_max);
}

void launch_normalize_by_scalar(float* data, int32_t n, const float* d_div, cudaStream_t s) {
    int blk = 256, grid = (n + blk - 1) / blk;
    normalize_by_device_scalar<<<grid, blk, 0, s>>>(data, n, d_div);
}

void launch_fill(float* data, int32_t n, float val, cudaStream_t s) {
    if (n <= 0) return;
    int blk = 256, grid = (n + blk - 1) / blk;
    fill_kernel<<<grid, blk, 0, s>>>(data, n, val);
}

void launch_scale(float* data, int32_t n, float sv, cudaStream_t s) {
    int blk = 256, grid = (n + blk - 1) / blk;
    scale_kernel<<<grid, blk, 0, s>>>(data, n, sv);
}

}  





HitsResult hits_seg_mask(const graph32_t& graph,
                         float* hubs,
                         float* authorities,
                         float epsilon,
                         std::size_t max_iterations,
                         bool has_initial_hubs_guess,
                         bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t N = graph.number_of_vertices;

    if (N == 0) {
        return HitsResult{0, false, 0.0f};
    }

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    const uint32_t* d_mask = graph.edge_mask;

    const auto& seg_vec = graph.segment_offsets.value();
    int32_t seg[5] = {seg_vec[0], seg_vec[1], seg_vec[2], seg_vec[3], seg_vec[4]};

    size_t cub_size = get_cub_temp_size(N);
    if (cub_size < 256) cub_size = 256;
    cache.ensure(N, cub_size + 64);

    float* d_ha = hubs;
    float* d_hb = cache.hub_tmp;
    float* d_auth = authorities;
    float* d_diff = cache.diff;
    float* d_s1 = cache.scalars;
    float* d_s2 = cache.scalars + 1;
    void* d_cub = cache.cub_temp;

    cudaStream_t stream = nullptr;

    float* d_curr = d_ha;
    float* d_next = d_hb;

    if (has_initial_hubs_guess) {
        launch_reduce_sum(d_curr, d_s1, d_cub, cub_size, N, stream);
        float hub_sum;
        cudaMemcpy(&hub_sum, d_s1, sizeof(float), cudaMemcpyDeviceToHost);
        if (hub_sum > 0.0f) launch_scale(d_curr, N, 1.0f / hub_sum, stream);
    } else {
        launch_fill(d_curr, N, 1.0f / N, stream);
    }

    float eff_eps = epsilon * N;
    float final_norm = 0.0f;
    size_t iterations = 0;

    for (size_t it = 0; it < max_iterations; it++) {
        cudaMemsetAsync(d_next, 0, N * sizeof(float), stream);

        launch_fused_auth_hub(d_off, d_idx, d_mask, d_curr, d_auth, d_next, seg, stream);

        launch_reduce_max(d_next, d_s1, d_cub, cub_size, N, stream);

        launch_normalize_diff(d_next, d_curr, d_diff, N, d_s1, stream);

        launch_reduce_sum(d_diff, d_s2, d_cub, cub_size, N, stream);

        float diff_val;
        cudaMemcpy(&diff_val, d_s2, sizeof(float), cudaMemcpyDeviceToHost);

        std::swap(d_curr, d_next);
        iterations = it + 1;
        final_norm = diff_val;

        if (diff_val < eff_eps) break;
    }

    launch_reduce_max(d_auth, d_s1, d_cub, cub_size, N, stream);
    launch_normalize_by_scalar(d_auth, N, d_s1, stream);

    if (normalize) {
        launch_reduce_sum(d_curr, d_s1, d_cub, cub_size, N, stream);
        launch_normalize_by_scalar(d_curr, N, d_s1, stream);
        launch_reduce_sum(d_auth, d_s1, d_cub, cub_size, N, stream);
        launch_normalize_by_scalar(d_auth, N, d_s1, stream);
    }

    if (d_curr != hubs) {
        cudaMemcpyAsync(hubs, d_curr, (size_t)N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    bool converged = (iterations > 0) && (final_norm < eff_eps);
    return HitsResult{iterations, converged, final_norm};
}

}  
