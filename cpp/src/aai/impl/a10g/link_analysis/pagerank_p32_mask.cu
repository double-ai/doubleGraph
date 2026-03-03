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
#include <vector>

namespace aai {

namespace {


__global__ void count_and_outdeg_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ counts,
    float* __restrict__ out_degrees,
    int32_t nv)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < nv; v += blockDim.x * gridDim.x) {
        int start = offsets[v], end = offsets[v + 1], cnt = 0;
        for (int e = start; e < end; e++) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1u) {
                cnt++;
                atomicAdd(&out_degrees[indices[e]], 1.0f);
            }
        }
        counts[v] = cnt;
    }
}


__global__ void compact_edges_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t nv)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < nv; v += blockDim.x * gridDim.x) {
        int old_s = old_offsets[v], old_e = old_offsets[v + 1];
        int w = new_offsets[v];
        for (int e = old_s; e < old_e; e++) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1u)
                new_indices[w++] = old_indices[e];
        }
    }
}


__global__ void init_uniform_kernel(float* __restrict__ pr, int32_t n) {
    float val = 1.0f / (float)n;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        pr[i] = val;
}


__global__ void compute_x_dangling_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ out_deg,
    float* __restrict__ x,
    float* __restrict__ dangling_sum,
    int32_t n)
{
    float local_d = 0.0f;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += blockDim.x * gridDim.x) {
        float od = out_deg[v];
        float p = pr[v];
        x[v] = (od > 0.0f) ? (p / od) : 0.0f;
        if (od == 0.0f) local_d += p;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_d += __shfl_down_sync(0xffffffff, local_d, off);
    if ((threadIdx.x & 31) == 0 && local_d != 0.0f)
        atomicAdd(dangling_sum, local_d);
}


__global__ void spmv_filtered_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ result,
    float alpha,
    int32_t num_vertices)
{
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices; v += blockDim.x * gridDim.x) {
        int start = offsets[v], end = offsets[v + 1];
        float sum = 0.0f;
        for (int e = start; e < end; e++)
            sum += x[indices[e]];
        result[v] = alpha * sum;
    }
}

__global__ void spmv_filtered_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x,
    float* __restrict__ result,
    float alpha,
    int32_t num_vertices)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (blockDim.x * gridDim.x) >> 5;

    for (int v = warp_id; v < num_vertices; v += total_warps) {
        int start = offsets[v], end = offsets[v + 1];
        float sum = 0.0f;
        for (int e = start + lane; e < end; e += 32)
            sum += x[indices[e]];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, off);
        if (lane == 0) result[v] = alpha * sum;
    }
}


__global__ void teleport_sparse_kernel(
    float* __restrict__ new_pr,
    const int32_t* __restrict__ pers_verts,
    const float* __restrict__ pers_norm_vals,
    int32_t pers_size,
    float alpha,
    float one_minus_alpha,
    const float* __restrict__ dangling_sum_ptr)
{
    float base = alpha * (*dangling_sum_ptr) + one_minus_alpha;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < pers_size; i += blockDim.x * gridDim.x)
        new_pr[pers_verts[i]] += base * pers_norm_vals[i];
}


__global__ void diff_kernel(
    const float* __restrict__ new_pr,
    const float* __restrict__ old_pr,
    float* __restrict__ diff_sum,
    int32_t n)
{
    float local_diff = 0.0f;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += blockDim.x * gridDim.x)
        local_diff += fabsf(new_pr[v] - old_pr[v]);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_diff += __shfl_down_sync(0xffffffff, local_diff, off);
    if ((threadIdx.x & 31) == 0 && local_diff != 0.0f)
        atomicAdd(diff_sum, local_diff);
}


static inline int grid_for(int n, int B, int maxG) {
    int G = (n + B - 1) / B;
    return G < maxG ? G : maxG;
}

static size_t get_prefix_sum_temp_bytes(int32_t n) {
    size_t tb = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tb, (int32_t*)nullptr, (int32_t*)nullptr, n);
    return tb;
}


struct Cache : Cacheable {
    
    float* out_deg = nullptr;
    int32_t* counts = nullptr;
    int32_t* new_off = nullptr;
    float* pr_a = nullptr;
    float* pr_b = nullptr;
    float* x = nullptr;
    float* scalars = nullptr;
    void* ps_tmp = nullptr;
    size_t ps_bytes = 0;
    int64_t nv_cap = 0;

    
    int32_t* new_ind = nullptr;
    int64_t ne_cap = 0;

    
    float* pnv = nullptr;
    int64_t ps_cap = 0;

    ~Cache() override {
        if (out_deg) cudaFree(out_deg);
        if (counts) cudaFree(counts);
        if (new_off) cudaFree(new_off);
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (x) cudaFree(x);
        if (scalars) cudaFree(scalars);
        if (ps_tmp) cudaFree(ps_tmp);
        if (new_ind) cudaFree(new_ind);
        if (pnv) cudaFree(pnv);
    }

    void ensure_nv(int32_t nv) {
        if (nv_cap < nv) {
            if (out_deg) cudaFree(out_deg);
            if (counts) cudaFree(counts);
            if (new_off) cudaFree(new_off);
            if (pr_a) cudaFree(pr_a);
            if (pr_b) cudaFree(pr_b);
            if (x) cudaFree(x);
            if (scalars) cudaFree(scalars);
            if (ps_tmp) cudaFree(ps_tmp);

            cudaMalloc(&out_deg, nv * sizeof(float));
            cudaMalloc(&counts, ((int64_t)nv + 1) * sizeof(int32_t));
            cudaMalloc(&new_off, ((int64_t)nv + 1) * sizeof(int32_t));
            cudaMalloc(&pr_a, nv * sizeof(float));
            cudaMalloc(&pr_b, nv * sizeof(float));
            cudaMalloc(&x, nv * sizeof(float));
            cudaMalloc(&scalars, 2 * sizeof(float));

            ps_bytes = get_prefix_sum_temp_bytes(nv + 1);
            cudaMalloc(&ps_tmp, ps_bytes);

            nv_cap = nv;
        }
    }

    void ensure_ne(int32_t new_ne) {
        int64_t sz = new_ne > 0 ? new_ne : 1;
        if (ne_cap < sz) {
            if (new_ind) cudaFree(new_ind);
            cudaMalloc(&new_ind, sz * sizeof(int32_t));
            ne_cap = sz;
        }
    }

    void ensure_ps(int64_t ps) {
        if (ps_cap < ps) {
            if (pnv) cudaFree(pnv);
            cudaMalloc(&pnv, ps * sizeof(float));
            ps_cap = ps;
        }
    }
};

}  

PageRankResult personalized_pagerank_mask(const graph32_t& graph,
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

    const int32_t* d_off = graph.offsets;
    const int32_t* d_ind = graph.indices;
    int32_t nv = graph.number_of_vertices;
    const uint32_t* d_mask = graph.edge_mask;

    const int32_t* d_pv = personalization_vertices;
    int64_t ps = static_cast<int64_t>(personalization_size);

    cache.ensure_nv(nv);

    float* d_od = cache.out_deg;
    int32_t* d_counts = cache.counts;

    
    cudaMemset(d_od, 0, nv * sizeof(float));
    cudaMemsetAsync(d_counts + nv, 0, sizeof(int32_t));
    count_and_outdeg_kernel<<<grid_for(nv, 256, 4096), 256>>>(d_off, d_ind, d_mask, d_counts, d_od, nv);

    
    int32_t* d_new_off = cache.new_off;
    cub::DeviceScan::ExclusiveSum(cache.ps_tmp, cache.ps_bytes, d_counts, d_new_off, nv + 1);

    
    int32_t new_ne;
    cudaMemcpy(&new_ne, d_new_off + nv, sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    cache.ensure_ne(new_ne);
    int32_t* d_new_ind = cache.new_ind;
    compact_edges_kernel<<<grid_for(nv, 256, 4096), 256>>>(d_off, d_ind, d_mask, d_new_off, d_new_ind, nv);

    
    std::vector<float> h_pv(ps);
    cudaMemcpy(h_pv.data(), personalization_values, ps * sizeof(float), cudaMemcpyDeviceToHost);
    double psum = 0.0;
    for (int64_t i = 0; i < ps; i++) psum += (double)h_pv[i];
    float inv_psum = (psum > 0.0) ? (float)(1.0 / psum) : 0.0f;
    for (int64_t i = 0; i < ps; i++) h_pv[i] *= inv_psum;

    cache.ensure_ps(ps);
    float* d_pnv = cache.pnv;
    cudaMemcpy(d_pnv, h_pv.data(), ps * sizeof(float), cudaMemcpyHostToDevice);

    
    float* d_pra = cache.pr_a;
    float* d_prb = cache.pr_b;
    float* d_x = cache.x;
    float* d_ds = cache.scalars;
    float* d_dfs = cache.scalars + 1;

    
    if (initial_pageranks) {
        cudaMemcpy(d_pra, initial_pageranks, nv * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        init_uniform_kernel<<<grid_for(nv, 256, 4096), 256>>>(d_pra, nv);
    }

    
    float* cur = d_pra;
    float* nxt = d_prb;
    float oma = 1.0f - alpha;
    bool converged = false;
    size_t iters = 0;
    const int CHECK_INTERVAL = 8;

    for (size_t it = 0; it < max_iterations; it++) {
        iters = it + 1;
        bool do_check = ((it + 1) % CHECK_INTERVAL == 0) || (it + 1 == max_iterations);

        
        cudaMemsetAsync(d_ds, 0, sizeof(float));

        
        compute_x_dangling_kernel<<<grid_for(nv, 256, 4096), 256>>>(cur, d_od, d_x, d_ds, nv);

        
        {
            int B = 256;
            float avg_deg = (float)new_ne / (float)(nv > 0 ? nv : 1);
            if (avg_deg <= 8.0f) {
                spmv_filtered_thread_kernel<<<grid_for(nv, B, 65535), B>>>(d_new_off, d_new_ind, d_x, nxt, alpha, nv);
            } else {
                int wpc = B / 32;
                spmv_filtered_warp_kernel<<<grid_for(nv, wpc, 65535), B>>>(d_new_off, d_new_ind, d_x, nxt, alpha, nv);
            }
        }

        
        if (ps > 0) {
            int B = 64;
            int G = ((int)ps + B - 1) / B;
            if (G < 1) G = 1;
            teleport_sparse_kernel<<<G, B>>>(nxt, d_pv, d_pnv, (int32_t)ps, alpha, oma, d_ds);
        }

        if (do_check) {
            
            cudaMemsetAsync(d_dfs, 0, sizeof(float));
            diff_kernel<<<grid_for(nv, 256, 4096), 256>>>(nxt, cur, d_dfs, nv);
            float h_diff;
            cudaMemcpy(&h_diff, d_dfs, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_diff < epsilon) {
                converged = true;
                break;
            }
        }

        std::swap(cur, nxt);
    }

    
    float* result_ptr = converged ? nxt : cur;
    cudaMemcpy(pageranks, result_ptr, nv * sizeof(float), cudaMemcpyDeviceToDevice);

    return PageRankResult{iters, converged};
}

}  
