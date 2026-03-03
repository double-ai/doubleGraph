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
#include <limits>

namespace aai {

namespace {

static constexpr int MAX_REDUCE_BLOCKS = 512;
static constexpr int REDUCE_BLOCKS = 512;
static constexpr int BLOCK = 256;


__device__ __forceinline__ bool is_active(const uint32_t* mask, int32_t e) {
    return (mask[e >> 5] >> (e & 31)) & 1;
}



template <int BLOCK_SIZE>
__global__ void auth_block(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ mask,
    const float* __restrict__ hubs,
    float* __restrict__ auth,
    int32_t v_start, int32_t v_end)
{
    int v = v_start + blockIdx.x;
    if (v >= v_end) return;
    int s = offsets[v], e = offsets[v+1];
    float sum = 0.0f;
    for (int i = s + threadIdx.x; i < e; i += BLOCK_SIZE) {
        if (is_active(mask, i))
            sum += hubs[indices[i]];
    }
    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage tmp;
    sum = BR(tmp).Sum(sum);
    if (threadIdx.x == 0) auth[v] = sum;
}

__global__ void auth_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ mask,
    const float* __restrict__ hubs,
    float* __restrict__ auth,
    int32_t v_start, int32_t v_end)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int v = v_start + wid;
    if (v >= v_end) return;
    int s = offsets[v], e = offsets[v+1];
    float sum = 0.0f;
    for (int i = s + lane; i < e; i += 32) {
        if (is_active(mask, i))
            sum += hubs[indices[i]];
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, off);
    if (lane == 0) auth[v] = sum;
}

__global__ void auth_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ mask,
    const float* __restrict__ hubs,
    float* __restrict__ auth,
    int32_t v_start, int32_t v_end)
{
    int v = v_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= v_end) return;
    int s = offsets[v], e = offsets[v+1];
    float sum = 0.0f;
    for (int i = s; i < e; i++) {
        if (is_active(mask, i))
            sum += hubs[indices[i]];
    }
    auth[v] = sum;
}



template <int BLOCK_SIZE>
__global__ void hub_block(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ mask,
    const float* __restrict__ auth,
    float* __restrict__ hubs,
    int32_t v_start, int32_t v_end)
{
    int v = v_start + blockIdx.x;
    if (v >= v_end) return;
    int s = offsets[v], e = offsets[v+1];
    float a = auth[v];
    if (a == 0.0f) return;
    for (int i = s + threadIdx.x; i < e; i += BLOCK_SIZE) {
        if (is_active(mask, i))
            atomicAdd(&hubs[indices[i]], a);
    }
}

__global__ void hub_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ mask,
    const float* __restrict__ auth,
    float* __restrict__ hubs,
    int32_t v_start, int32_t v_end)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int v = v_start + wid;
    if (v >= v_end) return;
    int s = offsets[v], e = offsets[v+1];
    float a = auth[v];
    if (a == 0.0f) return;
    for (int i = s + lane; i < e; i += 32) {
        if (is_active(mask, i))
            atomicAdd(&hubs[indices[i]], a);
    }
}

__global__ void hub_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ mask,
    const float* __restrict__ auth,
    float* __restrict__ hubs,
    int32_t v_start, int32_t v_end)
{
    int v = v_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= v_end) return;
    int s = offsets[v], e = offsets[v+1];
    float a = auth[v];
    if (a == 0.0f) return;
    for (int i = s; i < e; i++) {
        if (is_active(mask, i))
            atomicAdd(&hubs[indices[i]], a);
    }
}



template <int BLOCK_SIZE>
__global__ void fused_abs_max_2(
    const float* __restrict__ arr_a,
    const float* __restrict__ arr_b,
    int32_t n,
    float* __restrict__ partials_a,
    float* __restrict__ partials_b,
    unsigned int* __restrict__ retire_count,
    float* __restrict__ out_max)
{
    float max_a = 0.0f, max_b = 0.0f;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        float va = fabsf(arr_a[i]);
        float vb = fabsf(arr_b[i]);
        if (va > max_a) max_a = va;
        if (vb > max_b) max_b = vb;
    }

    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage tmp;

    max_a = BR(tmp).Reduce(max_a, cuda::maximum<float>{});
    __syncthreads();
    max_b = BR(tmp).Reduce(max_b, cuda::maximum<float>{});

    if (threadIdx.x == 0) {
        partials_a[blockIdx.x] = max_a;
        partials_b[blockIdx.x] = max_b;
    }

    __threadfence();

    __shared__ bool am_last;
    if (threadIdx.x == 0) {
        unsigned int ticket = atomicAdd(retire_count, 1);
        am_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (am_last) {
        float local_max_a = 0.0f, local_max_b = 0.0f;
        for (int i = threadIdx.x; i < (int)gridDim.x; i += BLOCK_SIZE) {
            float pa = partials_a[i], pb = partials_b[i];
            if (pa > local_max_a) local_max_a = pa;
            if (pb > local_max_b) local_max_b = pb;
        }
        local_max_a = BR(tmp).Reduce(local_max_a, cuda::maximum<float>{});
        __syncthreads();
        local_max_b = BR(tmp).Reduce(local_max_b, cuda::maximum<float>{});
        if (threadIdx.x == 0) {
            out_max[0] = local_max_a;
            out_max[1] = local_max_b;
            *retire_count = 0;
        }
    }
}

template <int BLOCK_SIZE>
__global__ void fused_norm_diff_sum(
    float* __restrict__ arr_a,
    float* __restrict__ arr_b,
    const float* __restrict__ old_arr,
    const float* __restrict__ max_vals,
    int32_t n,
    float* __restrict__ partials,
    unsigned int* __restrict__ retire_count,
    float* __restrict__ out_diff_sum)
{
    float max_a = max_vals[0];
    float max_b = max_vals[1];
    float inv_a = (max_a > 0.0f) ? 1.0f / max_a : 1.0f;
    float inv_b = (max_b > 0.0f) ? 1.0f / max_b : 1.0f;

    float diff_sum = 0.0f;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        float va = arr_a[i] * inv_a;
        arr_a[i] = va;
        arr_b[i] *= inv_b;
        diff_sum += fabsf(va - old_arr[i]);
    }

    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage tmp;
    diff_sum = BR(tmp).Sum(diff_sum);

    if (threadIdx.x == 0) partials[blockIdx.x] = diff_sum;

    __threadfence();

    __shared__ bool am_last;
    if (threadIdx.x == 0) {
        unsigned int ticket = atomicAdd(retire_count, 1);
        am_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (am_last) {
        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < (int)gridDim.x; i += BLOCK_SIZE)
            local_sum += partials[i];
        local_sum = BR(tmp).Sum(local_sum);
        if (threadIdx.x == 0) {
            *out_diff_sum = local_sum;
            *retire_count = 0;
        }
    }
}

template <int BLOCK_SIZE>
__global__ void fused_abs_sum(
    const float* __restrict__ arr,
    int32_t n,
    float* __restrict__ partials,
    unsigned int* __restrict__ retire_count,
    float* __restrict__ out_sum)
{
    float sum = 0.0f;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE)
        sum += fabsf(arr[i]);

    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage tmp;
    sum = BR(tmp).Sum(sum);

    if (threadIdx.x == 0) partials[blockIdx.x] = sum;
    __threadfence();

    __shared__ bool am_last;
    if (threadIdx.x == 0) {
        unsigned int ticket = atomicAdd(retire_count, 1);
        am_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (am_last) {
        float local_sum = 0.0f;
        for (int i = threadIdx.x; i < (int)gridDim.x; i += BLOCK_SIZE)
            local_sum += partials[i];
        local_sum = BR(tmp).Sum(local_sum);
        if (threadIdx.x == 0) {
            *out_sum = local_sum;
            *retire_count = 0;
        }
    }
}

template <int BLOCK_SIZE>
__global__ void fused_abs_sum_2_kernel(
    const float* __restrict__ arr_a,
    const float* __restrict__ arr_b,
    int32_t n,
    float* __restrict__ partials_a,
    float* __restrict__ partials_b,
    unsigned int* __restrict__ retire_count,
    float* __restrict__ out_sums)
{
    float sum_a = 0.0f, sum_b = 0.0f;
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        sum_a += fabsf(arr_a[i]);
        sum_b += fabsf(arr_b[i]);
    }

    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage tmp;
    sum_a = BR(tmp).Sum(sum_a);
    __syncthreads();
    sum_b = BR(tmp).Sum(sum_b);

    if (threadIdx.x == 0) {
        partials_a[blockIdx.x] = sum_a;
        partials_b[blockIdx.x] = sum_b;
    }
    __threadfence();

    __shared__ bool am_last;
    if (threadIdx.x == 0) {
        unsigned int ticket = atomicAdd(retire_count, 1);
        am_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (am_last) {
        float la = 0.0f, lb = 0.0f;
        for (int i = threadIdx.x; i < (int)gridDim.x; i += BLOCK_SIZE) {
            la += partials_a[i];
            lb += partials_b[i];
        }
        la = BR(tmp).Sum(la);
        __syncthreads();
        lb = BR(tmp).Sum(lb);
        if (threadIdx.x == 0) {
            out_sums[0] = la;
            out_sums[1] = lb;
            *retire_count = 0;
        }
    }
}

__global__ void l1_norm_2(float* __restrict__ a, float* __restrict__ b,
                          const float* __restrict__ sums, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sa = sums[0], sb = sums[1];
    if (sa > 0.0f) a[i] /= sa;
    if (sb > 0.0f) b[i] /= sb;
}



__global__ void fill_val(float* arr, int32_t n, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = val;
}

__global__ void l1_norm_kernel(float* arr, float sum, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (sum > 0.0f) arr[i] /= sum;
}



void launch_compute_auth(
    const int32_t* offsets, const int32_t* indices, const uint32_t* mask,
    const float* hubs, float* auth,
    const int32_t* segs)
{
    int32_t s0 = segs[0], s1 = segs[1], s2 = segs[2], s4 = segs[4];

    int n_high = s1 - s0;
    if (n_high > 0)
        auth_block<BLOCK><<<n_high, BLOCK>>>(offsets, indices, mask, hubs, auth, s0, s1);

    int n_mid = s2 - s1;
    if (n_mid > 0) {
        int blocks = (int)(((int64_t)n_mid * 32 + BLOCK - 1) / BLOCK);
        auth_warp<<<blocks, BLOCK>>>(offsets, indices, mask, hubs, auth, s1, s2);
    }

    int n_low = s4 - s2;
    if (n_low > 0) {
        int blocks = (n_low + BLOCK - 1) / BLOCK;
        auth_thread<<<blocks, BLOCK>>>(offsets, indices, mask, hubs, auth, s2, s4);
    }
}

void launch_compute_hubs_scatter(
    const int32_t* offsets, const int32_t* indices, const uint32_t* mask,
    const float* auth, float* hubs,
    const int32_t* segs, int32_t num_vertices)
{
    cudaMemsetAsync(hubs, 0, num_vertices * sizeof(float));

    int32_t s0 = segs[0], s1 = segs[1], s2 = segs[2], s3 = segs[3];

    int n_high = s1 - s0;
    if (n_high > 0)
        hub_block<BLOCK><<<n_high, BLOCK>>>(offsets, indices, mask, auth, hubs, s0, s1);

    int n_mid = s2 - s1;
    if (n_mid > 0) {
        int blocks = (int)(((int64_t)n_mid * 32 + BLOCK - 1) / BLOCK);
        hub_warp<<<blocks, BLOCK>>>(offsets, indices, mask, auth, hubs, s1, s2);
    }

    int n_low = s3 - s2;
    if (n_low > 0) {
        int blocks = (n_low + BLOCK - 1) / BLOCK;
        hub_thread<<<blocks, BLOCK>>>(offsets, indices, mask, auth, hubs, s2, s3);
    }
}

void launch_fused_abs_max(float* hubs, float* auth, int32_t n,
                          float* partials_a, float* partials_b,
                          unsigned int* retire_count, float* d_maxvals)
{
    int num_blocks = (n + BLOCK * 4 - 1) / (BLOCK * 4);
    if (num_blocks < 64) num_blocks = 64;
    if (num_blocks > REDUCE_BLOCKS) num_blocks = REDUCE_BLOCKS;
    fused_abs_max_2<BLOCK><<<num_blocks, BLOCK>>>(hubs, auth, n, partials_a, partials_b, retire_count, d_maxvals);
}

void launch_fused_norm_diff(float* hubs, float* auth, const float* prev_hubs,
                            const float* d_maxvals, int32_t n,
                            float* partials, unsigned int* retire_count, float* d_diff_sum)
{
    int num_blocks = (n + BLOCK * 4 - 1) / (BLOCK * 4);
    if (num_blocks < 64) num_blocks = 64;
    if (num_blocks > REDUCE_BLOCKS) num_blocks = REDUCE_BLOCKS;
    fused_norm_diff_sum<BLOCK><<<num_blocks, BLOCK>>>(hubs, auth, prev_hubs, d_maxvals, n,
                                                       partials, retire_count, d_diff_sum);
}

void launch_fused_abs_sum_1(const float* arr, int32_t n,
                            float* partials, unsigned int* retire_count, float* d_out)
{
    int num_blocks = (n + BLOCK * 4 - 1) / (BLOCK * 4);
    if (num_blocks < 64) num_blocks = 64;
    if (num_blocks > REDUCE_BLOCKS) num_blocks = REDUCE_BLOCKS;
    fused_abs_sum<BLOCK><<<num_blocks, BLOCK>>>(arr, n, partials, retire_count, d_out);
}

void launch_fused_abs_sum_2(const float* a, const float* b, int32_t n,
                            float* partials_a, float* partials_b,
                            unsigned int* retire_count, float* d_sums)
{
    int num_blocks = (n + BLOCK * 4 - 1) / (BLOCK * 4);
    if (num_blocks < 64) num_blocks = 64;
    if (num_blocks > REDUCE_BLOCKS) num_blocks = REDUCE_BLOCKS;
    fused_abs_sum_2_kernel<BLOCK><<<num_blocks, BLOCK>>>(a, b, n, partials_a, partials_b, retire_count, d_sums);
}

void launch_l1_norm_2(float* a, float* b, const float* sums, int32_t n) {
    if (n > 0)
        l1_norm_2<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(a, b, sums, n);
}

void launch_fill(float* arr, int32_t n, float val) {
    if (n > 0) fill_val<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(arr, n, val);
}

void launch_l1_norm_1(float* arr, float sum, int32_t n) {
    if (n > 0) l1_norm_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK>>>(arr, sum, n);
}



struct Cache : Cacheable {
    float* hubs_buf = nullptr;
    float* partials_a = nullptr;
    float* partials_b = nullptr;
    float* scalars = nullptr;
    unsigned int* retire_count = nullptr;

    int32_t hubs_buf_capacity = 0;
    int32_t partials_a_capacity = 0;
    int32_t partials_b_capacity = 0;
    int32_t scalars_capacity = 0;
    int32_t retire_count_capacity = 0;

    void ensure(int32_t num_vertices) {
        if (hubs_buf_capacity < num_vertices) {
            if (hubs_buf) cudaFree(hubs_buf);
            cudaMalloc(&hubs_buf, (std::size_t)num_vertices * sizeof(float));
            hubs_buf_capacity = num_vertices;
        }
        if (partials_a_capacity < MAX_REDUCE_BLOCKS) {
            if (partials_a) cudaFree(partials_a);
            cudaMalloc(&partials_a, MAX_REDUCE_BLOCKS * sizeof(float));
            partials_a_capacity = MAX_REDUCE_BLOCKS;
        }
        if (partials_b_capacity < MAX_REDUCE_BLOCKS) {
            if (partials_b) cudaFree(partials_b);
            cudaMalloc(&partials_b, MAX_REDUCE_BLOCKS * sizeof(float));
            partials_b_capacity = MAX_REDUCE_BLOCKS;
        }
        if (scalars_capacity < 4) {
            if (scalars) cudaFree(scalars);
            cudaMalloc(&scalars, 4 * sizeof(float));
            scalars_capacity = 4;
        }
        if (retire_count_capacity < 1) {
            if (retire_count) cudaFree(retire_count);
            cudaMalloc(&retire_count, sizeof(unsigned int));
            retire_count_capacity = 1;
        }
    }

    ~Cache() override {
        if (hubs_buf) cudaFree(hubs_buf);
        if (partials_a) cudaFree(partials_a);
        if (partials_b) cudaFree(partials_b);
        if (scalars) cudaFree(scalars);
        if (retire_count) cudaFree(retire_count);
    }
};

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

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    const uint32_t* d_mask = graph.edge_mask;

    const auto& seg_vec = graph.segment_offsets.value();
    int32_t segs[5] = {seg_vec[0], seg_vec[1], seg_vec[2], seg_vec[3], seg_vec[4]};

    cache.ensure(num_vertices);

    float* d_hubs = hubs;
    float* d_auth = authorities;
    float* d_hubs_buf = cache.hubs_buf;
    float* d_partials_a = cache.partials_a;
    float* d_partials_b = cache.partials_b;
    float* d_scalars = cache.scalars;
    unsigned int* d_retire = cache.retire_count;

    cudaMemsetAsync(d_retire, 0, sizeof(unsigned int));

    float tolerance = static_cast<float>(num_vertices) * epsilon;

    float* prev = d_hubs;
    float* curr = d_hubs_buf;

    if (has_initial_hubs_guess) {
        launch_fused_abs_sum_1(prev, num_vertices, d_partials_a, d_retire, d_scalars);
        float h_sum;
        cudaMemcpy(&h_sum, d_scalars, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_sum > 0.0f)
            launch_l1_norm_1(prev, h_sum, num_vertices);
    } else {
        launch_fill(prev, num_vertices, 1.0f / num_vertices);
    }

    float diff_sum = std::numeric_limits<float>::max();
    std::size_t iter = 0;
    std::size_t max_iter = max_iterations;

    int check_interval = 1;

    while (iter < max_iter) {
        std::size_t batch_end = iter + check_interval;
        if (batch_end > max_iter) batch_end = max_iter;

        for (; iter < batch_end; iter++) {
            launch_compute_auth(d_offsets, d_indices, d_mask, prev, d_auth, segs);
            launch_compute_hubs_scatter(d_offsets, d_indices, d_mask, d_auth, curr, segs, num_vertices);
            launch_fused_abs_max(curr, d_auth, num_vertices,
                                d_partials_a, d_partials_b, d_retire, d_scalars);
            launch_fused_norm_diff(curr, d_auth, prev, d_scalars, num_vertices,
                                   d_partials_a, d_retire, d_scalars + 2);
            std::swap(prev, curr);
        }

        cudaMemcpy(&diff_sum, d_scalars + 2, sizeof(float), cudaMemcpyDeviceToHost);
        if (diff_sum < tolerance) break;
    }

    if (normalize) {
        launch_fused_abs_sum_2(prev, d_auth, num_vertices,
                               d_partials_a, d_partials_b, d_retire, d_scalars);
        launch_l1_norm_2(prev, d_auth, d_scalars, num_vertices);
    }

    if (prev != d_hubs) {
        cudaMemcpyAsync(d_hubs, prev, num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();

    return HitsResult{iter, (iter < max_iter), diff_sum};
}

}  
