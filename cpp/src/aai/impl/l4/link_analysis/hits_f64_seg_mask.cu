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
#include <cstdint>

namespace aai {

namespace {

struct Cache : Cacheable {
    double* d_temp = nullptr;
    double* d_sc = nullptr;
    int64_t temp_capacity = 0;
    int64_t sc_capacity = 0;

    void ensure(int64_t N) {
        if (temp_capacity < N) {
            if (d_temp) cudaFree(d_temp);
            cudaMalloc(&d_temp, N * sizeof(double));
            temp_capacity = N;
        }
        if (sc_capacity < 4) {
            if (d_sc) cudaFree(d_sc);
            cudaMalloc(&d_sc, 4 * sizeof(double));
            sc_capacity = 4;
        }
    }

    ~Cache() override {
        if (d_temp) cudaFree(d_temp);
        if (d_sc) cudaFree(d_sc);
    }
};


__device__ __forceinline__ bool edge_active(const uint32_t* mask, int j) {
    return (mask[j >> 5] >> (j & 31)) & 1;
}

__device__ void atomicMaxDouble(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull, assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= val) return;
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}




__global__ void auth_thread(
    const int* __restrict__ off, const int* __restrict__ idx,
    const uint32_t* __restrict__ mask, const double* __restrict__ hubs,
    double* __restrict__ auth, int sv, int cnt)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cnt) return;
    int v = sv + t;
    double s = 0.0;
    for (int j = off[v]; j < off[v+1]; j++)
        if (edge_active(mask, j)) s += hubs[idx[j]];
    auth[v] = s;
}


__global__ void auth_warp(
    const int* __restrict__ off, const int* __restrict__ idx,
    const uint32_t* __restrict__ mask, const double* __restrict__ hubs,
    double* __restrict__ auth, int sv, int cnt)
{
    int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= cnt) return;
    int v = sv + w;
    double s = 0.0;
    for (int j = off[v] + lane; j < off[v+1]; j += 32)
        if (edge_active(mask, j)) s += hubs[idx[j]];
    #pragma unroll
    for (int d = 16; d; d >>= 1) s += __shfl_down_sync(0xffffffff, s, d);
    if (!lane) auth[v] = s;
}


__global__ void auth_block(
    const int* __restrict__ off, const int* __restrict__ idx,
    const uint32_t* __restrict__ mask, const double* __restrict__ hubs,
    double* __restrict__ auth, int sv, int cnt)
{
    extern __shared__ double smem[];
    if (blockIdx.x >= (unsigned)cnt) return;
    int v = sv + blockIdx.x;
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;

    double s = 0.0;
    for (int j = off[v] + threadIdx.x; j < off[v+1]; j += blockDim.x)
        if (edge_active(mask, j)) s += hubs[idx[j]];

    #pragma unroll
    for (int d = 16; d; d >>= 1) s += __shfl_down_sync(0xffffffff, s, d);
    if (!lane) smem[wid] = s;
    __syncthreads();

    if (threadIdx.x < nw) s = smem[threadIdx.x]; else s = 0.0;
    if (threadIdx.x < 32) {
        #pragma unroll
        for (int d = 16; d; d >>= 1) s += __shfl_down_sync(0xffffffff, s, d);
    }
    if (!threadIdx.x) auth[v] = s;
}



__global__ void hub_thread(
    const int* __restrict__ off, const int* __restrict__ idx,
    const uint32_t* __restrict__ mask, const double* __restrict__ auth,
    double* __restrict__ hubs, int sv, int cnt)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cnt) return;
    int v = sv + t;
    double a = auth[v];
    if (a == 0.0) return;
    for (int j = off[v]; j < off[v+1]; j++)
        if (edge_active(mask, j)) atomicAdd(&hubs[idx[j]], a);
}

__global__ void hub_warp(
    const int* __restrict__ off, const int* __restrict__ idx,
    const uint32_t* __restrict__ mask, const double* __restrict__ auth,
    double* __restrict__ hubs, int sv, int cnt)
{
    int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (w >= cnt) return;
    int v = sv + w;
    double a = lane ? 0.0 : auth[v];
    a = __shfl_sync(0xffffffff, a, 0);
    if (a == 0.0) return;
    for (int j = off[v] + lane; j < off[v+1]; j += 32)
        if (edge_active(mask, j)) atomicAdd(&hubs[idx[j]], a);
}

__global__ void hub_block(
    const int* __restrict__ off, const int* __restrict__ idx,
    const uint32_t* __restrict__ mask, const double* __restrict__ auth,
    double* __restrict__ hubs, int sv, int cnt)
{
    if (blockIdx.x >= (unsigned)cnt) return;
    int v = sv + blockIdx.x;
    __shared__ double av;
    if (!threadIdx.x) av = auth[v];
    __syncthreads();
    if (av == 0.0) return;
    for (int j = off[v] + threadIdx.x; j < off[v+1]; j += blockDim.x)
        if (edge_active(mask, j)) atomicAdd(&hubs[idx[j]], av);
}



__global__ void k_set_zero(double* data, int sv, int cnt) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < cnt) data[sv + t] = 0.0;
}

__global__ void k_init_uniform(double* data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) data[i] = 1.0 / (double)N;
}


__global__ void k_max_reduce(const double* __restrict__ data, double* __restrict__ result, int N) {
    extern __shared__ double smem[];
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;
    double mx = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        double v = data[i];
        if (v > mx) mx = v;
    }
    #pragma unroll
    for (int d = 16; d; d >>= 1) { double o = __shfl_down_sync(0xffffffff, mx, d); if (o > mx) mx = o; }
    if (!lane) smem[wid] = mx;
    __syncthreads();
    mx = (threadIdx.x < (unsigned)nw) ? smem[threadIdx.x] : 0.0;
    if (threadIdx.x < 32) {
        #pragma unroll
        for (int d = 16; d; d >>= 1) { double o = __shfl_down_sync(0xffffffff, mx, d); if (o > mx) mx = o; }
    }
    if (!threadIdx.x) atomicMaxDouble(result, mx);
}


__global__ void k_sum_reduce(const double* __restrict__ data, double* __restrict__ result, int N) {
    extern __shared__ double smem[];
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;
    double s = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        s += data[i];
    #pragma unroll
    for (int d = 16; d; d >>= 1) s += __shfl_down_sync(0xffffffff, s, d);
    if (!lane) smem[wid] = s;
    __syncthreads();
    s = (threadIdx.x < (unsigned)nw) ? smem[threadIdx.x] : 0.0;
    if (threadIdx.x < 32) {
        #pragma unroll
        for (int d = 16; d; d >>= 1) s += __shfl_down_sync(0xffffffff, s, d);
    }
    if (!threadIdx.x) atomicAdd(result, s);
}


__global__ void k_normalize_and_diff(
    double* __restrict__ new_data, const double* __restrict__ old_data,
    const double* __restrict__ max_val, double* __restrict__ diff_sum, int N)
{
    extern __shared__ double smem[];
    int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int nw = blockDim.x >> 5;
    double m = *max_val;
    double ds = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        double h = (m > 0.0) ? new_data[i] / m : new_data[i];
        new_data[i] = h;
        ds += fabs(h - old_data[i]);
    }
    #pragma unroll
    for (int d = 16; d; d >>= 1) ds += __shfl_down_sync(0xffffffff, ds, d);
    if (!lane) smem[wid] = ds;
    __syncthreads();
    ds = (threadIdx.x < (unsigned)nw) ? smem[threadIdx.x] : 0.0;
    if (threadIdx.x < 32) {
        #pragma unroll
        for (int d = 16; d; d >>= 1) ds += __shfl_down_sync(0xffffffff, ds, d);
    }
    if (!threadIdx.x && ds != 0.0) atomicAdd(diff_sum, ds);
}


__global__ void k_normalize_by_val(double* __restrict__ data, const double* __restrict__ val, int N) {
    double v = *val;
    if (v <= 0.0) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        data[i] /= v;
}

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

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int N = graph.number_of_vertices;
    const uint32_t* d_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int s0 = seg[0], s1 = seg[1], s2 = seg[2], s3 = seg[3], s4 = seg[4];

    if (N == 0) {
        return HitsResultDouble{max_iterations, false, 1e30};
    }

    cache.ensure(N);
    double* d_temp = cache.d_temp;
    double* d_sc = cache.d_sc;

    const int BLK = 256;
    int gv = (N + BLK - 1) / BLK;
    int gr = (gv < 512) ? gv : 512;
    int smem = (BLK / 32) * sizeof(double);

    double tolerance = (double)N * epsilon;

    int n_high = s1 - s0;
    int n_mid  = s2 - s1;
    int n_low  = s3 - s2;
    int n_zero = s4 - s3;

    
    if (has_initial_hubs_guess) {
        cudaMemset(d_sc, 0, sizeof(double));
        k_sum_reduce<<<gr, BLK, smem>>>(hubs, d_sc, N);
        k_normalize_by_val<<<gv, BLK>>>(hubs, d_sc, N);
    } else {
        k_init_uniform<<<gv, BLK>>>(hubs, N);
    }

    double* prev = hubs;
    double* curr = d_temp;
    double diff = 1e30;
    size_t it = 0;

    while (it < max_iterations) {
        
        if (n_high > 0)
            auth_block<<<n_high, BLK, smem>>>(d_off, d_idx, d_mask, prev, authorities, s0, n_high);
        if (n_mid > 0) {
            int g = (int)(((int64_t)n_mid * 32 + BLK - 1) / BLK);
            auth_warp<<<g, BLK>>>(d_off, d_idx, d_mask, prev, authorities, s1, n_mid);
        }
        if (n_low > 0) {
            int g = (n_low + BLK - 1) / BLK;
            auth_thread<<<g, BLK>>>(d_off, d_idx, d_mask, prev, authorities, s2, n_low);
        }
        if (n_zero > 0) {
            int g = (n_zero + BLK - 1) / BLK;
            k_set_zero<<<g, BLK>>>(authorities, s3, n_zero);
        }

        
        cudaMemset(curr, 0, N * sizeof(double));

        
        if (n_high > 0)
            hub_block<<<n_high, BLK>>>(d_off, d_idx, d_mask, authorities, curr, s0, n_high);
        if (n_mid > 0) {
            int g = (int)(((int64_t)n_mid * 32 + BLK - 1) / BLK);
            hub_warp<<<g, BLK>>>(d_off, d_idx, d_mask, authorities, curr, s1, n_mid);
        }
        if (n_low > 0) {
            int g = (n_low + BLK - 1) / BLK;
            hub_thread<<<g, BLK>>>(d_off, d_idx, d_mask, authorities, curr, s2, n_low);
        }

        
        cudaMemset(d_sc, 0, sizeof(double));     
        k_max_reduce<<<gr, BLK, smem>>>(curr, d_sc, N);
        cudaMemset(d_sc + 2, 0, sizeof(double)); 
        k_normalize_and_diff<<<gr, BLK, smem>>>(curr, prev, d_sc, d_sc + 2, N);

        
        cudaMemset(d_sc + 1, 0, sizeof(double)); 
        k_max_reduce<<<gr, BLK, smem>>>(authorities, d_sc + 1, N);
        k_normalize_by_val<<<gv, BLK>>>(authorities, d_sc + 1, N);

        
        cudaMemcpy(&diff, d_sc + 2, sizeof(double), cudaMemcpyDeviceToHost);

        
        double* t = prev; prev = curr; curr = t;
        it++;

        if (diff < tolerance) break;
    }

    
    if (normalize) {
        cudaMemset(d_sc, 0, sizeof(double));
        k_sum_reduce<<<gr, BLK, smem>>>(prev, d_sc, N);
        k_normalize_by_val<<<gv, BLK>>>(prev, d_sc, N);

        cudaMemset(d_sc, 0, sizeof(double));
        k_sum_reduce<<<gr, BLK, smem>>>(authorities, d_sc, N);
        k_normalize_by_val<<<gv, BLK>>>(authorities, d_sc, N);
    }

    
    if (prev != hubs) {
        cudaMemcpy(hubs, prev, N * sizeof(double), cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();

    return HitsResultDouble{it, (diff < tolerance), diff};
}

}  
