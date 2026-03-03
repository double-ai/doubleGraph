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
#include <utility>

namespace aai {

namespace {

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }
static inline int hmin(int a, int b) { return a < b ? a : b; }



struct Cache : Cacheable {
    double* hubs_tmp = nullptr;
    double* scalars = nullptr;
    int64_t hubs_tmp_capacity = 0;
    int64_t scalars_capacity = 0;

    void ensure(int32_t V) {
        if (hubs_tmp_capacity < V) {
            if (hubs_tmp) cudaFree(hubs_tmp);
            cudaMalloc(&hubs_tmp, (size_t)V * sizeof(double));
            hubs_tmp_capacity = V;
        }
        if (scalars_capacity < 4) {
            if (scalars) cudaFree(scalars);
            cudaMalloc(&scalars, 4 * sizeof(double));
            scalars_capacity = 4;
        }
    }

    ~Cache() override {
        if (hubs_tmp) cudaFree(hubs_tmp);
        if (scalars) cudaFree(scalars);
    }
};




__global__ void spmv_block_k(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    const double* __restrict__ x, double* __restrict__ y, int32_t start_v)
{
    typedef cub::BlockReduce<double, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    int v = start_v + blockIdx.x;
    int s = off[v], e = off[v+1];
    double sum = 0.0;
    for (int j = s + threadIdx.x; j < e; j += 256)
        sum += x[idx[j]];
    sum = BR(tmp).Sum(sum);
    if (threadIdx.x == 0) y[v] = sum;
}


__global__ void spmv_warp_k(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    const double* __restrict__ x, double* __restrict__ y,
    int32_t start_v, int32_t cnt)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= cnt) return;
    int v = start_v + wid;
    int s = off[v], e = off[v+1];
    double sum = 0.0;
    for (int j = s + lane; j < e; j += 32)
        sum += x[idx[j]];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, o);
    if (lane == 0) y[v] = sum;
}


__global__ void spmv_thread_k(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    const double* __restrict__ x, double* __restrict__ y,
    int32_t start_v, int32_t cnt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cnt) return;
    int v = start_v + i;
    int s = off[v], e = off[v+1];
    double sum = 0.0;
    for (int j = s; j < e; j++)
        sum += x[idx[j]];
    y[v] = sum;
}



__global__ void scatter_block_k(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    const double* __restrict__ auth, double* __restrict__ hubs, int32_t start_v)
{
    int v = start_v + blockIdx.x;
    int s = off[v], e = off[v+1];
    double a = auth[v];
    for (int j = s + threadIdx.x; j < e; j += 256)
        atomicAdd(&hubs[idx[j]], a);
}

__global__ void scatter_warp_k(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    const double* __restrict__ auth, double* __restrict__ hubs,
    int32_t start_v, int32_t cnt)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= cnt) return;
    int v = start_v + wid;
    int s = off[v], e = off[v+1];
    double a = auth[v];
    for (int j = s + lane; j < e; j += 32)
        atomicAdd(&hubs[idx[j]], a);
}

__global__ void scatter_thread_k(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    const double* __restrict__ auth, double* __restrict__ hubs,
    int32_t start_v, int32_t cnt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cnt) return;
    int v = start_v + i;
    int s = off[v], e = off[v+1];
    double a = auth[v];
    for (int j = s; j < e; j++)
        atomicAdd(&hubs[idx[j]], a);
}



__global__ void fill_k(double* __restrict__ data, double val, int32_t n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        data[i] = val;
}

__global__ void abs_max_k(const double* __restrict__ data, double* __restrict__ out, int32_t n)
{
    typedef cub::BlockReduce<double, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    double mx = 0.0;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256)
        mx = fmax(mx, fabs(data[i]));
    mx = BR(tmp).Reduce(mx, [] __device__ (double a, double b) { return fmax(a, b); });
    if (threadIdx.x == 0) {
        unsigned long long* addr = (unsigned long long*)out;
        unsigned long long old_val = *addr, assumed;
        do {
            assumed = old_val;
            if (__longlong_as_double(assumed) >= mx) break;
            old_val = atomicCAS(addr, assumed, __double_as_longlong(mx));
        } while (assumed != old_val);
    }
}


__global__ void norm_diff_sum_k(
    double* __restrict__ hubs, const double* __restrict__ prev,
    const double* __restrict__ max_ptr, double* __restrict__ sum_out, int32_t n)
{
    typedef cub::BlockReduce<double, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    double inv = (*max_ptr > 0.0) ? (1.0 / *max_ptr) : 0.0;
    double s = 0.0;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        double h = hubs[i] * inv;
        hubs[i] = h;
        s += fabs(h - prev[i]);
    }
    s = BR(tmp).Sum(s);
    if (threadIdx.x == 0) atomicAdd(sum_out, s);
}

__global__ void sum_k(const double* __restrict__ data, double* __restrict__ out, int32_t n)
{
    typedef cub::BlockReduce<double, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    double s = 0.0;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256)
        s += data[i];
    s = BR(tmp).Sum(s);
    if (threadIdx.x == 0) atomicAdd(out, s);
}

__global__ void div_by_ptr_k(double* __restrict__ data, const double* __restrict__ d, int32_t n)
{
    double val = *d;
    double inv = (val > 0.0) ? (1.0 / val) : 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        data[i] *= inv;
}



void launch_spmv_gather(
    const int32_t* off, const int32_t* idx, const double* x, double* y,
    int32_t s0, int32_t s1, int32_t s2, int32_t s3, int32_t s4)
{
    int high = s1 - s0;
    if (high > 0)
        spmv_block_k<<<high, 256>>>(off, idx, x, y, s0);
    int mid = s2 - s1;
    if (mid > 0)
        spmv_warp_k<<<cdiv(mid, 8), 256>>>(off, idx, x, y, s1, mid);
    int low = s3 - s2;
    if (low > 0)
        spmv_thread_k<<<cdiv(low, 256), 256>>>(off, idx, x, y, s2, low);
    int zero = s4 - s3;
    if (zero > 0)
        fill_k<<<cdiv(zero, 256), 256>>>(y + s3, 0.0, zero);
}

void launch_scatter(
    const int32_t* off, const int32_t* idx, const double* auth, double* hubs,
    int32_t s0, int32_t s1, int32_t s2, int32_t s3, int32_t nv)
{
    cudaMemsetAsync(hubs, 0, (size_t)nv * sizeof(double));
    int high = s1 - s0;
    if (high > 0)
        scatter_block_k<<<high, 256>>>(off, idx, auth, hubs, s0);
    int mid = s2 - s1;
    if (mid > 0)
        scatter_warp_k<<<cdiv(mid, 8), 256>>>(off, idx, auth, hubs, s1, mid);
    int low = s3 - s2;
    if (low > 0)
        scatter_thread_k<<<cdiv(low, 256), 256>>>(off, idx, auth, hubs, s2, low);
}

void launch_abs_max(const double* data, double* out, int32_t n)
{
    cudaMemsetAsync(out, 0, sizeof(double));
    int blocks = hmin(cdiv(n, 256), 512);
    abs_max_k<<<blocks, 256>>>(data, out, n);
}

void launch_norm_diff_sum(double* hubs, const double* prev, const double* max_ptr, double* sum_out, int32_t n)
{
    cudaMemsetAsync(sum_out, 0, sizeof(double));
    int blocks = hmin(cdiv(n, 256), 512);
    norm_diff_sum_k<<<blocks, 256>>>(hubs, prev, max_ptr, sum_out, n);
}

void launch_fill(double* data, double val, int32_t n)
{
    fill_k<<<cdiv(n, 256), 256>>>(data, val, n);
}

void launch_sum(const double* data, double* out, int32_t n)
{
    cudaMemsetAsync(out, 0, sizeof(double));
    int blocks = hmin(cdiv(n, 256), 512);
    sum_k<<<blocks, 256>>>(data, out, n);
}

void launch_div_by_ptr(double* data, const double* d, int32_t n)
{
    div_by_ptr_k<<<cdiv(n, 256), 256>>>(data, d, n);
}

}  

HitsResultDouble hits_seg(const graph32_t& graph,
                          double* hubs,
                          double* authorities,
                          double epsilon,
                          std::size_t max_iterations,
                          bool has_initial_hubs_guess,
                          bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t V = graph.number_of_vertices;

    const auto& seg = graph.segment_offsets.value();
    int32_t s0 = seg[0], s1 = seg[1], s2 = seg[2], s3 = seg[3], s4 = seg[4];

    cache.ensure(V);

    double* d_ha = hubs;
    double* d_hb = cache.hubs_tmp;
    double* d_auth = authorities;
    double* d_max = cache.scalars;
    double* d_sum = cache.scalars + 1;

    
    if (has_initial_hubs_guess) {
        
        
        launch_sum(d_ha, d_sum, V);
        launch_div_by_ptr(d_ha, d_sum, V);
    } else {
        launch_fill(d_ha, 1.0 / V, V);
    }

    double* cur = d_ha;
    double* nxt = d_hb;
    double threshold = (double)V * epsilon;
    double h_delta = 0.0;
    bool converged = false;
    size_t iterations = 0;

    for (size_t it = 0; it < max_iterations; it++) {
        
        launch_spmv_gather(d_off, d_idx, cur, d_auth, s0, s1, s2, s3, s4);

        
        launch_scatter(d_off, d_idx, d_auth, nxt, s0, s1, s2, s3, V);

        
        launch_abs_max(nxt, d_max, V);
        launch_norm_diff_sum(nxt, cur, d_max, d_sum, V);

        
        cudaMemcpy(&h_delta, d_sum, sizeof(double), cudaMemcpyDeviceToHost);

        std::swap(cur, nxt);
        iterations = it + 1;

        if (h_delta < threshold) {
            converged = true;
            break;
        }
    }

    
    if (cur != d_ha)
        cudaMemcpy(d_ha, cur, (size_t)V * sizeof(double), cudaMemcpyDeviceToDevice);

    
    launch_abs_max(d_auth, d_max, V);
    launch_div_by_ptr(d_auth, d_max, V);

    
    if (normalize) {
        launch_sum(d_ha, d_sum, V);
        launch_div_by_ptr(d_ha, d_sum, V);
        launch_sum(d_auth, d_sum, V);
        launch_div_by_ptr(d_auth, d_sum, V);
    }

    cudaDeviceSynchronize();

    return HitsResultDouble{iterations, converged, h_delta};
}

}  
