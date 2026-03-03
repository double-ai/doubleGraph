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
#include <vector>

namespace aai {

namespace {

#define BS 256

struct Cache : Cacheable {
    int32_t* out_deg = nullptr;
    float* x = nullptr;
    float* pr_buf0 = nullptr;
    float* pr_buf1 = nullptr;
    float* scalars = nullptr;
    int32_t vertex_cap = 0;

    float* pers_norm = nullptr;
    int64_t pers_cap = 0;

    void ensure(int32_t n_verts, int64_t n_pers) {
        if (vertex_cap < n_verts) {
            if (out_deg) cudaFree(out_deg);
            if (x) cudaFree(x);
            if (pr_buf0) cudaFree(pr_buf0);
            if (pr_buf1) cudaFree(pr_buf1);
            if (scalars) cudaFree(scalars);
            cudaMalloc(&out_deg, (size_t)n_verts * sizeof(int32_t));
            cudaMalloc(&x, (size_t)n_verts * sizeof(float));
            cudaMalloc(&pr_buf0, (size_t)n_verts * sizeof(float));
            cudaMalloc(&pr_buf1, (size_t)n_verts * sizeof(float));
            cudaMalloc(&scalars, 2 * sizeof(float));
            vertex_cap = n_verts;
        }
        if (pers_cap < n_pers) {
            if (pers_norm) cudaFree(pers_norm);
            cudaMalloc(&pers_norm, (size_t)n_pers * sizeof(float));
            pers_cap = n_pers;
        }
    }

    ~Cache() override {
        if (out_deg) cudaFree(out_deg);
        if (x) cudaFree(x);
        if (pr_buf0) cudaFree(pr_buf0);
        if (pr_buf1) cudaFree(pr_buf1);
        if (scalars) cudaFree(scalars);
        if (pers_norm) cudaFree(pers_norm);
    }
};


__global__ void compute_out_degrees_kernel(
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ out_deg,
    int32_t num_edges)
{
    int idx = blockIdx.x * BS + threadIdx.x;
    if (idx < num_edges) {
        atomicAdd(&out_deg[indices[idx]], 1);
    }
}


__global__ void compute_x_dangling_diff_kernel(
    const float* __restrict__ pr,
    const int32_t* __restrict__ out_deg,
    float* __restrict__ x,
    float* __restrict__ d_dangling_sum,
    float* __restrict__ d_diff_sum,
    const float* __restrict__ pr_other,
    int32_t N)
{
    typedef cub::BlockReduce<float, BS> BR;
    __shared__ union {
        typename BR::TempStorage ts[2];
    } shared;

    int idx = blockIdx.x * BS + threadIdx.x;
    float dval = 0.0f;
    float diff = 0.0f;

    if (idx < N) {
        float p = pr[idx];
        int32_t d = out_deg[idx];
        if (d > 0) {
            x[idx] = p / (float)d;
        } else {
            x[idx] = 0.0f;
            dval = p;
        }
        if (pr_other) {
            diff = fabsf(p - pr_other[idx]);
        }
    }

    float bsum_d = BR(shared.ts[0]).Sum(dval);
    if (threadIdx.x == 0 && bsum_d != 0.0f) {
        atomicAdd(d_dangling_sum, bsum_d);
    }

    if (pr_other) {
        __syncthreads();
        float bsum_diff = BR(shared.ts[1]).Sum(diff);
        if (threadIdx.x == 0 && bsum_diff != 0.0f) {
            atomicAdd(d_diff_sum, bsum_diff);
        }
    }
}


__global__ void spmv_high_kernel(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ ind,
    const float* __restrict__ x,
    float* __restrict__ out,
    float alpha,
    int32_t vertex_start,
    int32_t count)
{
    typedef cub::BlockReduce<float, BS> BR;
    __shared__ typename BR::TempStorage ts;

    if ((int)blockIdx.x >= count) return;
    int32_t v = vertex_start + blockIdx.x;
    int32_t s = off[v], e = off[v + 1];

    float sum = 0.0f;
    for (int32_t j = s + threadIdx.x; j < e; j += BS) {
        sum += x[ind[j]];
    }

    float bsum = BR(ts).Sum(sum);
    if (threadIdx.x == 0) {
        out[v] = alpha * bsum;
    }
}


__global__ void spmv_mid_kernel(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ ind,
    const float* __restrict__ x,
    float* __restrict__ out,
    float alpha,
    int32_t vertex_start,
    int32_t count)
{
    int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (wid >= count) return;
    int32_t v = vertex_start + wid;
    int32_t s = off[v], e = off[v + 1];

    float sum = 0.0f;
    for (int32_t j = s + lane; j < e; j += 32) {
        sum += x[ind[j]];
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, o);
    }

    if (lane == 0) {
        out[v] = alpha * sum;
    }
}


__global__ void spmv_low_zero_kernel(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ ind,
    const float* __restrict__ x,
    float* __restrict__ out,
    float alpha,
    int32_t vertex_start,
    int32_t count)
{
    int tid = blockIdx.x * BS + threadIdx.x;
    if (tid >= count) return;
    int32_t v = vertex_start + tid;
    int32_t s = off[v], e = off[v + 1];

    float sum = 0.0f;
    for (int32_t j = s; j < e; j++) {
        sum += x[ind[j]];
    }

    out[v] = alpha * sum;
}


__global__ void add_pers_kernel(
    float* __restrict__ pr,
    const int32_t* __restrict__ pers_verts,
    const float* __restrict__ pers_norm,
    const float* __restrict__ d_dangling_sum,
    float alpha,
    int32_t pers_size)
{
    int idx = blockIdx.x * BS + threadIdx.x;
    if (idx >= pers_size) return;
    float base = alpha * d_dangling_sum[0] + (1.0f - alpha);
    pr[pers_verts[idx]] += base * pers_norm[idx];
}


__global__ void init_pr_kernel(float* __restrict__ pr, float val, int32_t N) {
    int idx = blockIdx.x * BS + threadIdx.x;
    if (idx < N) pr[idx] = val;
}


__global__ void compute_diff_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ d_diff_sum,
    int32_t N)
{
    typedef cub::BlockReduce<float, BS> BR;
    __shared__ typename BR::TempStorage ts;

    int idx = blockIdx.x * BS + threadIdx.x;
    float d = 0.0f;
    if (idx < N) d = fabsf(a[idx] - b[idx]);
    float bsum = BR(ts).Sum(d);
    if (threadIdx.x == 0 && bsum != 0.0f) {
        atomicAdd(d_diff_sum, bsum);
    }
}

static void launch_compute_out_degrees(const int32_t* indices, int32_t* out_deg, int32_t ne) {
    if (ne <= 0) return;
    compute_out_degrees_kernel<<<(ne + BS - 1) / BS, BS>>>(indices, out_deg, ne);
}

static void launch_compute_x_dangling_diff(const float* pr, const int32_t* od, float* x,
    float* d_dang, float* d_diff, const float* pr_other, int32_t N) {
    if (N <= 0) return;
    compute_x_dangling_diff_kernel<<<(N + BS - 1) / BS, BS>>>(pr, od, x, d_dang, d_diff, pr_other, N);
}

static void launch_spmv_high(const int32_t* off, const int32_t* ind, const float* x,
    float* out, float alpha, int32_t vs, int32_t cnt) {
    if (cnt <= 0) return;
    spmv_high_kernel<<<cnt, BS>>>(off, ind, x, out, alpha, vs, cnt);
}

static void launch_spmv_mid(const int32_t* off, const int32_t* ind, const float* x,
    float* out, float alpha, int32_t vs, int32_t cnt) {
    if (cnt <= 0) return;
    int wpb = BS / 32;
    int grid = (cnt + wpb - 1) / wpb;
    spmv_mid_kernel<<<grid, BS>>>(off, ind, x, out, alpha, vs, cnt);
}

static void launch_spmv_low_zero(const int32_t* off, const int32_t* ind, const float* x,
    float* out, float alpha, int32_t vs, int32_t cnt) {
    if (cnt <= 0) return;
    spmv_low_zero_kernel<<<(cnt + BS - 1) / BS, BS>>>(off, ind, x, out, alpha, vs, cnt);
}

static void launch_add_pers(float* pr, const int32_t* pv, const float* pn, const float* dd,
    float alpha, int32_t ps) {
    if (ps <= 0) return;
    add_pers_kernel<<<(ps + BS - 1) / BS, BS>>>(pr, pv, pn, dd, alpha, ps);
}

static void launch_init_pr(float* pr, float val, int32_t N) {
    if (N <= 0) return;
    init_pr_kernel<<<(N + BS - 1) / BS, BS>>>(pr, val, N);
}

static void launch_compute_diff(const float* a, const float* b, float* ds, int32_t N) {
    if (N <= 0) return;
    compute_diff_kernel<<<(N + BS - 1) / BS, BS>>>(a, b, ds, N);
}

}  

PageRankResult personalized_pagerank_seg(const graph32_t& graph,
                                         const int32_t* personalization_vertices,
                                         const float* personalization_values,
                                         std::size_t personalization_size,
                                         float* pageranks,
                                         const float* precomputed_vertex_out_weight_sums,
                                         float alpha,
                                         float epsilon,
                                         std::size_t max_iterations,
                                         const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    const auto& seg_opt = graph.segment_offsets.value();
    int32_t seg[5] = {seg_opt[0], seg_opt[1], seg_opt[2], seg_opt[3], seg_opt[4]};

    int64_t pers_size = static_cast<int64_t>(personalization_size);

    cache.ensure(num_vertices, pers_size);

    int32_t* d_out_deg = cache.out_deg;
    float* d_x = cache.x;
    float* d_pr0 = cache.pr_buf0;
    float* d_pr1 = cache.pr_buf1;
    float* d_pers_norm = cache.pers_norm;
    float* d_dangling = cache.scalars;
    float* d_diff = cache.scalars + 1;

    
    cudaMemset(d_out_deg, 0, (size_t)num_vertices * sizeof(int32_t));
    launch_compute_out_degrees(d_indices, d_out_deg, num_edges);

    
    std::vector<float> h_pers(pers_size);
    cudaMemcpy(h_pers.data(), personalization_values, pers_size * sizeof(float), cudaMemcpyDeviceToHost);
    double pers_sum = 0.0;
    for (int64_t i = 0; i < pers_size; i++) pers_sum += (double)h_pers[i];
    for (int64_t i = 0; i < pers_size; i++) h_pers[i] = (float)((double)h_pers[i] / pers_sum);
    cudaMemcpy(d_pers_norm, h_pers.data(), pers_size * sizeof(float), cudaMemcpyHostToDevice);

    
    float* pr_src = d_pr0;
    float* pr_dst = d_pr1;

    if (initial_pageranks != nullptr) {
        cudaMemcpy(pr_src, initial_pageranks,
                   (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        float init_val = 1.0f / (float)num_vertices;
        launch_init_pr(pr_src, init_val, num_vertices);
    }

    int32_t n_high = seg[1] - seg[0];
    int32_t n_mid  = seg[2] - seg[1];
    int32_t n_low_zero = seg[4] - seg[2];

    
    bool converged = false;
    size_t iterations = 0;
    float* result_ptr = pr_src;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        
        cudaMemsetAsync(d_dangling, 0, 2 * sizeof(float), 0);

        
        if (iter == 0) {
            launch_compute_x_dangling_diff(pr_src, d_out_deg, d_x, d_dangling, d_diff, nullptr, num_vertices);
        } else {
            
            launch_compute_x_dangling_diff(pr_src, d_out_deg, d_x, d_dangling, d_diff, pr_dst, num_vertices);

            
            float h_diff;
            cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_diff < epsilon) {
                converged = true;
                iterations = iter;
                result_ptr = pr_src;
                break;
            }
        }

        
        launch_spmv_high(d_offsets, d_indices, d_x, pr_dst, alpha, seg[0], n_high);
        launch_spmv_mid(d_offsets, d_indices, d_x, pr_dst, alpha, seg[1], n_mid);
        launch_spmv_low_zero(d_offsets, d_indices, d_x, pr_dst, alpha, seg[2], n_low_zero);

        
        launch_add_pers(pr_dst, personalization_vertices, d_pers_norm, d_dangling, alpha, (int32_t)pers_size);

        result_ptr = pr_dst;
        iterations = iter + 1;

        
        std::swap(pr_src, pr_dst);
    }

    
    if (!converged && iterations > 0) {
        
        cudaMemsetAsync(d_diff, 0, sizeof(float), 0);
        launch_compute_diff(pr_src, pr_dst, d_diff, num_vertices);
        float h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_diff < epsilon) {
            converged = true;
            result_ptr = pr_src;
        }
    }

    
    cudaMemcpy(pageranks, result_ptr, (size_t)num_vertices * sizeof(float), cudaMemcpyDeviceToDevice);

    return PageRankResult{iterations, converged};
}

}  
