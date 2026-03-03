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

namespace aai {

namespace {





struct Cache : Cacheable {
    
    int32_t* active_counts = nullptr;
    int32_t* new_offsets = nullptr;
    float* out_weight = nullptr;
    float* out_weight_inv = nullptr;
    float* pr_norm = nullptr;
    float* spmv_out = nullptr;
    int64_t n_capacity = 0;

    
    double* dangling = nullptr;   
    double* diff = nullptr;       
    bool fixed_allocated = false;

    
    int32_t* new_indices = nullptr;
    int64_t nidx_capacity = 0;

    uint8_t* scan_workspace = nullptr;
    size_t scan_ws_capacity = 0;

    void ensure_n(int32_t N) {
        if (n_capacity < N) {
            if (active_counts) cudaFree(active_counts);
            if (new_offsets) cudaFree(new_offsets);
            if (out_weight) cudaFree(out_weight);
            if (out_weight_inv) cudaFree(out_weight_inv);
            if (pr_norm) cudaFree(pr_norm);
            if (spmv_out) cudaFree(spmv_out);

            cudaMalloc(&active_counts, (int64_t)(N + 1) * sizeof(int32_t));
            cudaMalloc(&new_offsets, (int64_t)(N + 1) * sizeof(int32_t));
            cudaMalloc(&out_weight, (int64_t)N * sizeof(float));
            cudaMalloc(&out_weight_inv, (int64_t)N * sizeof(float));
            cudaMalloc(&pr_norm, (int64_t)N * sizeof(float));
            cudaMalloc(&spmv_out, (int64_t)N * sizeof(float));

            n_capacity = N;
        }
    }

    void ensure_fixed() {
        if (!fixed_allocated) {
            cudaMalloc(&dangling, 2 * sizeof(double));
            cudaMalloc(&diff, sizeof(double));
            fixed_allocated = true;
        }
    }

    void ensure_nidx(int32_t h_total) {
        int64_t need = h_total > 0 ? (int64_t)h_total : 1LL;
        if (nidx_capacity < need) {
            if (new_indices) cudaFree(new_indices);
            cudaMalloc(&new_indices, need * sizeof(int32_t));
            nidx_capacity = need;
        }
    }

    void ensure_scan_ws(size_t ws_size) {
        if (scan_ws_capacity < ws_size) {
            if (scan_workspace) cudaFree(scan_workspace);
            cudaMalloc(&scan_workspace, ws_size);
            scan_ws_capacity = ws_size;
        }
    }

    ~Cache() override {
        if (active_counts) cudaFree(active_counts);
        if (new_offsets) cudaFree(new_offsets);
        if (out_weight) cudaFree(out_weight);
        if (out_weight_inv) cudaFree(out_weight_inv);
        if (pr_norm) cudaFree(pr_norm);
        if (spmv_out) cudaFree(spmv_out);
        if (dangling) cudaFree(dangling);
        if (diff) cudaFree(diff);
        if (new_indices) cudaFree(new_indices);
        if (scan_workspace) cudaFree(scan_workspace);
    }
};





__global__ void count_active_edges(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_counts,
    int32_t num_vertices)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int32_t start = offsets[v], end = offsets[v + 1];
    int32_t count = 0;
    for (int32_t e = start; e < end; e++)
        count += (edge_mask[e >> 5] >> (e & 31)) & 1;
    active_counts[v] = count;
}

__global__ void compact_edges(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t num_vertices)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int32_t old_start = old_offsets[v], old_end = old_offsets[v + 1];
    int32_t write_pos = new_offsets[v];
    for (int32_t e = old_start; e < old_end; e++)
        if ((edge_mask[e >> 5] >> (e & 31)) & 1)
            new_indices[write_pos++] = old_indices[e];
}

__global__ void compute_out_weights_compact(
    const int32_t* __restrict__ indices, float* __restrict__ out_weight, int32_t E)
{
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    atomicAdd(&out_weight[indices[e]], 1.0f);
}

__global__ void compute_out_weight_inv(
    const float* __restrict__ out_weight, float* __restrict__ out_weight_inv, int32_t N)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    float w = out_weight[v];
    out_weight_inv[v] = (w > 0.0f) ? __frcp_rn(w) : 0.0f;
}

__global__ void init_pr(float* __restrict__ pr, float val, int32_t n)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) pr[v] = val;
}





__global__ void compute_initial_pr_norm_dangling(
    const float* __restrict__ pr,
    const float* __restrict__ out_weight_inv,
    float* __restrict__ pr_norm,
    double* __restrict__ d_dangling_sum,
    int32_t N)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int32_t v = blockIdx.x * 256 + threadIdx.x;
    double my_dang = 0.0;
    if (v < N) {
        float inv = out_weight_inv[v];
        float prv = pr[v];
        pr_norm[v] = prv * inv;
        if (inv == 0.0f) my_dang = (double)prv;
    }
    double bs = BlockReduce(temp).Sum(my_dang);
    if (threadIdx.x == 0 && bs != 0.0)
        atomicAdd(d_dangling_sum, bs);
}





__global__ void spmv_high(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_norm,
    float* __restrict__ spmv_out,
    int32_t seg_start, int32_t seg_end)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int32_t v = seg_start + blockIdx.x;
    if (v >= seg_end) return;
    int32_t start = offsets[v], end = offsets[v + 1];

    float sum = 0.0f;
    for (int32_t e = start + threadIdx.x; e < end; e += 256)
        sum += pr_norm[indices[e]];

    float bs = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) spmv_out[v] = bs;
}

__global__ void spmv_mid(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_norm,
    float* __restrict__ spmv_out,
    int32_t seg_start, int32_t seg_end)
{
    int32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t warp_id = gtid >> 5, lane = gtid & 31;
    int32_t v = seg_start + warp_id;
    if (v >= seg_end) return;

    int32_t start = offsets[v], end = offsets[v + 1];
    float sum = 0.0f;
    for (int32_t e = start + lane; e < end; e += 32)
        sum += pr_norm[indices[e]];

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, o);

    if (lane == 0) spmv_out[v] = sum;
}

__global__ void spmv_low(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ pr_norm,
    float* __restrict__ spmv_out,
    int32_t seg_start, int32_t seg_end)
{
    int32_t v = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= seg_end) return;
    int32_t start = offsets[v], end = offsets[v + 1];
    float sum = 0.0f;
    for (int32_t e = start; e < end; e++)
        sum += pr_norm[indices[e]];
    spmv_out[v] = sum;
}





__global__ void fused_update_prep(
    float* __restrict__ pr,
    const float* __restrict__ spmv_out,
    const double* __restrict__ d_dangling_curr,
    float* __restrict__ pr_norm,
    double* __restrict__ d_dangling_next,
    const float* __restrict__ out_weight_inv,
    float alpha, float omaN, float aoN, int32_t N)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int32_t v = blockIdx.x * 256 + threadIdx.x;
    double my_dang = 0.0;
    if (v < N) {
        float dc = (float)(*d_dangling_curr) * aoN;
        float new_val = omaN + alpha * spmv_out[v] + dc;
        pr[v] = new_val;

        float inv = out_weight_inv[v];
        pr_norm[v] = new_val * inv;
        if (inv == 0.0f) my_dang = (double)new_val;
    }
    double bs = BlockReduce(temp).Sum(my_dang);
    if (threadIdx.x == 0 && bs != 0.0)
        atomicAdd(d_dangling_next, bs);
}





__global__ void fused_update_diff_prep(
    float* __restrict__ pr,
    const float* __restrict__ spmv_out,
    const double* __restrict__ d_dangling_curr,
    float* __restrict__ pr_norm,
    double* __restrict__ d_dangling_next,
    double* __restrict__ d_diff_sum,
    const float* __restrict__ out_weight_inv,
    float alpha, float omaN, float aoN, int32_t N)
{
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ union {
        typename BlockReduce::TempStorage dang;
        typename BlockReduce::TempStorage diff;
    } temp;

    int32_t v = blockIdx.x * 256 + threadIdx.x;
    double my_dang = 0.0;
    double my_diff = 0.0;
    if (v < N) {
        float dc = (float)(*d_dangling_curr) * aoN;
        float new_val = omaN + alpha * spmv_out[v] + dc;
        float old_val = pr[v];
        my_diff = (double)fabsf(new_val - old_val);
        pr[v] = new_val;

        float inv = out_weight_inv[v];
        pr_norm[v] = new_val * inv;
        if (inv == 0.0f) my_dang = (double)new_val;
    }

    double dang_bs = BlockReduce(temp.dang).Sum(my_dang);
    if (threadIdx.x == 0 && dang_bs != 0.0)
        atomicAdd(d_dangling_next, dang_bs);

    __syncthreads();

    double diff_bs = BlockReduce(temp.diff).Sum(my_diff);
    if (threadIdx.x == 0 && diff_bs != 0.0)
        atomicAdd(d_diff_sum, diff_bs);
}

}  





PageRankResult pagerank_seg_mask(const graph32_t& graph,
                                 float* pageranks,
                                 const float* precomputed_vertex_out_weight_sums,
                                 float alpha,
                                 float epsilon,
                                 std::size_t max_iterations,
                                 const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t N = graph.number_of_vertices;
    const uint32_t* d_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int32_t s0 = seg[0], s1 = seg[1], s2 = seg[2], s4 = seg[4];

    cache.ensure_n(N);
    cache.ensure_fixed();

    
    int32_t* d_ac = cache.active_counts;
    cudaMemsetAsync(d_ac + N, 0, sizeof(int32_t));
    if (N > 0)
        count_active_edges<<<(N + 255) / 256, 256>>>(d_off, d_mask, d_ac, N);

    int32_t* d_noff = cache.new_offsets;
    size_t scan_ws = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_ws, (int32_t*)nullptr, (int32_t*)nullptr, N + 1);
    cache.ensure_scan_ws(scan_ws);
    cub::DeviceScan::ExclusiveSum(cache.scan_workspace, scan_ws, d_ac, d_noff, N + 1);

    int32_t h_total;
    cudaMemcpy(&h_total, d_noff + N, sizeof(int32_t), cudaMemcpyDeviceToHost);

    cache.ensure_nidx(h_total);
    int32_t* d_nidx = cache.new_indices;
    if (N > 0)
        compact_edges<<<(N + 255) / 256, 256>>>(d_off, d_idx, d_mask, d_noff, d_nidx, N);

    
    float* d_ow = cache.out_weight;
    float* d_owi = cache.out_weight_inv;
    cudaMemset(d_ow, 0, N * sizeof(float));
    if (h_total > 0)
        compute_out_weights_compact<<<(h_total + 255) / 256, 256>>>(d_nidx, d_ow, h_total);
    if (N > 0)
        compute_out_weight_inv<<<(N + 255) / 256, 256>>>(d_ow, d_owi, N);

    
    float* d_pr = pageranks;
    if (initial_pageranks != nullptr)
        cudaMemcpy(d_pr, initial_pageranks, N * sizeof(float), cudaMemcpyDeviceToDevice);
    else if (N > 0)
        init_pr<<<(N + 255) / 256, 256>>>(d_pr, 1.0f / (float)N, N);

    
    float* d_prn = cache.pr_norm;
    float* d_spmv = cache.spmv_out;
    double* d_dang = cache.dangling;
    double* d_diff = cache.diff;

    double* d_dang_curr = d_dang;
    double* d_dang_next = d_dang + 1;

    float omaN = (1.0f - alpha) / (float)N;
    float aoN = alpha / (float)N;

    
    cudaMemset(d_dang_curr, 0, sizeof(double));
    if (N > 0)
        compute_initial_pr_norm_dangling<<<(N + 255) / 256, 256>>>(d_pr, d_owi, d_prn, d_dang_curr, N);

    
    std::size_t iterations = 0;
    bool converged = false;
    const int64_t CHECK_INTERVAL = 5;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        int32_t nh = s1 - s0;
        if (nh > 0) spmv_high<<<nh, 256>>>(d_noff, d_nidx, d_prn, d_spmv, s0, s1);
        int32_t nm = s2 - s1;
        if (nm > 0) {
            int wpb = 8;
            spmv_mid<<<(nm + wpb - 1) / wpb, wpb * 32>>>(d_noff, d_nidx, d_prn, d_spmv, s1, s2);
        }
        int32_t nl = s4 - s2;
        if (nl > 0) spmv_low<<<(nl + 255) / 256, 256>>>(d_noff, d_nidx, d_prn, d_spmv, s2, s4);

        
        cudaMemsetAsync(d_dang_next, 0, sizeof(double));

        bool do_check = ((iter + 1) % CHECK_INTERVAL == 0) || (iter + 1 == max_iterations);

        if (do_check) {
            cudaMemsetAsync(d_diff, 0, sizeof(double));
            if (N > 0)
                fused_update_diff_prep<<<(N + 255) / 256, 256>>>(d_pr, d_spmv, d_dang_curr,
                    d_prn, d_dang_next, d_diff, d_owi, alpha, omaN, aoN, N);

            double h_diff;
            cudaMemcpy(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost);
            iterations = iter + 1;
            if (h_diff < (double)epsilon) {
                converged = true;
                break;
            }
        } else {
            if (N > 0)
                fused_update_prep<<<(N + 255) / 256, 256>>>(d_pr, d_spmv, d_dang_curr,
                    d_prn, d_dang_next, d_owi, alpha, omaN, aoN, N);
            iterations = iter + 1;
        }

        
        double* tmp = d_dang_curr;
        d_dang_curr = d_dang_next;
        d_dang_next = tmp;
    }

    return PageRankResult{iterations, converged};
}

}  
