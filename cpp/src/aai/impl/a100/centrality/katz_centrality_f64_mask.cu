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
#include <algorithm>
#include <cmath>

namespace aai {

namespace {




struct Cache : Cacheable {
    double* buf0 = nullptr;
    double* buf1 = nullptr;
    double* d_scratch = nullptr;
    int64_t buf0_capacity = 0;
    int64_t buf1_capacity = 0;
    bool scratch_allocated = false;

    void ensure(int32_t nv) {
        int64_t need = static_cast<int64_t>(nv);
        if (buf0_capacity < need) {
            if (buf0) cudaFree(buf0);
            cudaMalloc(&buf0, need * sizeof(double));
            buf0_capacity = need;
        }
        if (buf1_capacity < need) {
            if (buf1) cudaFree(buf1);
            cudaMalloc(&buf1, need * sizeof(double));
            buf1_capacity = need;
        }
        if (!scratch_allocated) {
            cudaMalloc(&d_scratch, 2 * sizeof(double));
            scratch_allocated = true;
        }
    }

    ~Cache() override {
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (d_scratch) cudaFree(d_scratch);
    }
};




static constexpr int BLK = 256;
static constexpr int WPB = BLK / 32;

__global__ void spmv_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double*  __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double*  __restrict__ x_old,
    double*        __restrict__ x_new,
    double alpha,
    double beta_scalar,
    const double*  __restrict__ betas,
    int32_t nv,
    double*        __restrict__ d_l1)
{
    typedef cub::BlockReduce<double, BLK> BR;
    __shared__ typename BR::TempStorage ts;

    int warp = threadIdx.x >> 5;
    int lane  = threadIdx.x & 31;
    int v = blockIdx.x * WPB + warp;

    double diff = 0.0;
    if (v < nv) {
        int32_t s = offsets[v], e = offsets[v + 1];
        double sum = 0.0;
        for (int32_t i = s + lane; i < e; i += 32) {
            uint32_t mw = edge_mask[i >> 5];
            if ((mw >> (i & 31)) & 1u)
                sum += weights[i] * x_old[indices[i]];
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, o);

        if (lane == 0) {
            double bv = betas ? betas[v] : beta_scalar;
            double nval = alpha * sum + bv;
            x_new[v] = nval;
            diff = fabs(nval - x_old[v]);
        }
    }

    double bs = BR(ts).Sum(diff);
    if (threadIdx.x == 0 && bs > 0.0)
        atomicAdd(d_l1, bs);
}




__global__ void spmv_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double*  __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double*  __restrict__ x_old,
    double*        __restrict__ x_new,
    double alpha,
    double beta_scalar,
    const double*  __restrict__ betas,
    int32_t nv,
    double*        __restrict__ d_l1)
{
    typedef cub::BlockReduce<double, BLK> BR;
    __shared__ typename BR::TempStorage ts;

    int v = blockIdx.x * BLK + threadIdx.x;

    double diff = 0.0;
    if (v < nv) {
        int32_t s = offsets[v], e = offsets[v + 1];
        double sum = 0.0;
        for (int32_t i = s; i < e; i++) {
            uint32_t mw = edge_mask[i >> 5];
            if ((mw >> (i & 31)) & 1u)
                sum += weights[i] * x_old[indices[i]];
        }
        double bv = betas ? betas[v] : beta_scalar;
        double nval = alpha * sum + bv;
        x_new[v] = nval;
        diff = fabs(nval - x_old[v]);
    }

    double bs = BR(ts).Sum(diff);
    if (threadIdx.x == 0 && bs > 0.0)
        atomicAdd(d_l1, bs);
}




__global__ void l2_sq_kernel(const double* __restrict__ x, double* __restrict__ out, int32_t n) {
    typedef cub::BlockReduce<double, 256> BR;
    __shared__ typename BR::TempStorage ts;
    double s = 0.0;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256) {
        double v = x[i]; s += v * v;
    }
    double bs = BR(ts).Sum(s);
    if (threadIdx.x == 0 && bs > 0.0) atomicAdd(out, bs);
}

__global__ void norm_kernel(double* __restrict__ x, const double* __restrict__ nsq, int32_t n) {
    double inv = rsqrt(*nsq);
    for (int i = blockIdx.x * 256 + threadIdx.x; i < n; i += gridDim.x * 256)
        x[i] *= inv;
}




void launch_spmv_warp_impl(const int32_t* off, const int32_t* idx, const double* w,
    const uint32_t* mask, const double* x_old, double* x_new,
    double alpha, double beta, const double* betas,
    int32_t nv, double* d_l1, cudaStream_t s)
{
    cudaMemsetAsync(d_l1, 0, sizeof(double), s);
    int g = (nv + WPB - 1) / WPB;
    if (g > 0) spmv_warp_kernel<<<g, BLK, 0, s>>>(off, idx, w, mask, x_old, x_new, alpha, beta, betas, nv, d_l1);
}

void launch_spmv_thread_impl(const int32_t* off, const int32_t* idx, const double* w,
    const uint32_t* mask, const double* x_old, double* x_new,
    double alpha, double beta, const double* betas,
    int32_t nv, double* d_l1, cudaStream_t s)
{
    cudaMemsetAsync(d_l1, 0, sizeof(double), s);
    int g = (nv + BLK - 1) / BLK;
    if (g > 0) spmv_thread_kernel<<<g, BLK, 0, s>>>(off, idx, w, mask, x_old, x_new, alpha, beta, betas, nv, d_l1);
}

void launch_normalize_impl(double* x, double* nsq, int32_t n, cudaStream_t s)
{
    cudaMemsetAsync(nsq, 0, sizeof(double), s);
    int g = (n + 255) / 256;
    if (g > 2048) g = 2048;
    l2_sq_kernel<<<g, 256, 0, s>>>(x, nsq, n);
    norm_kernel<<<g, 256, 0, s>>>(x, nsq, n);
}

}  

katz_centrality_result_t katz_centrality_mask(const graph32_t& graph,
                           const double* edge_weights,
                           double* centralities,
                           double alpha,
                           double beta,
                           const double* betas,
                           double epsilon,
                           std::size_t max_iterations,
                           bool has_initial_guess,
                           bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;

    const int32_t*  d_off  = graph.offsets;
    const int32_t*  d_idx  = graph.indices;
    const uint32_t* d_mask = graph.edge_mask;

    cache.ensure(nv);

    cudaStream_t stream = 0;

    
    int avg_deg = (nv > 0) ? (ne / nv) : 0;
    bool use_thread_kernel = (avg_deg <= 8);

    
    double* x_old = cache.buf0;
    double* x_new = cache.buf1;

    if (has_initial_guess) {
        cudaMemcpyAsync(x_old, centralities,
            nv * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemsetAsync(x_old, 0, nv * sizeof(double), stream);
    }

    double* d_l1  = &cache.d_scratch[0];
    double* d_nsq = &cache.d_scratch[1];

    bool converged = false;
    std::size_t iterations = 0;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        if (use_thread_kernel) {
            launch_spmv_thread_impl(d_off, d_idx, edge_weights, d_mask,
                x_old, x_new, alpha, beta, betas, nv, d_l1, stream);
        } else {
            launch_spmv_warp_impl(d_off, d_idx, edge_weights, d_mask,
                x_old, x_new, alpha, beta, betas, nv, d_l1, stream);
        }

        double h_l1;
        cudaMemcpy(&h_l1, d_l1, sizeof(double), cudaMemcpyDeviceToHost);

        std::swap(x_old, x_new);
        iterations = iter + 1;

        if (h_l1 < epsilon) {
            converged = true;
            break;
        }
    }

    if (normalize) {
        launch_normalize_impl(x_old, d_nsq, nv, stream);
    }

    
    cudaMemcpyAsync(centralities, x_old,
        nv * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return {iterations, converged};
}

}  
