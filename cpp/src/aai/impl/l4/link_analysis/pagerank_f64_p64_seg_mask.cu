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

namespace aai {

namespace {

struct Cache : Cacheable {
    
    int32_t* active_count = nullptr;
    int32_t* new_offsets = nullptr;
    double* out_weights = nullptr;
    int32_t* dangling_list = nullptr;
    int32_t* dangling_count = nullptr;
    double* pers_norm = nullptr;
    double* pr_a64 = nullptr;
    double* pr_b64 = nullptr;
    float* pr_a32 = nullptr;
    float* pr_b32 = nullptr;
    double* scalars = nullptr;
    int32_t vertex_cap = 0;

    
    int32_t* new_indices = nullptr;
    double* new_weights = nullptr;
    float* norm_weights_f32 = nullptr;
    int32_t edge_cap = 0;

    
    void* cub_temp = nullptr;
    size_t cub_cap = 0;

    void ensure_vertex(int32_t N) {
        if (vertex_cap < N) {
            if (active_count) cudaFree(active_count);
            if (new_offsets) cudaFree(new_offsets);
            if (out_weights) cudaFree(out_weights);
            if (dangling_list) cudaFree(dangling_list);
            if (dangling_count) cudaFree(dangling_count);
            if (pers_norm) cudaFree(pers_norm);
            if (pr_a64) cudaFree(pr_a64);
            if (pr_b64) cudaFree(pr_b64);
            if (pr_a32) cudaFree(pr_a32);
            if (pr_b32) cudaFree(pr_b32);
            if (scalars) cudaFree(scalars);

            cudaMalloc(&active_count, N * sizeof(int32_t));
            cudaMalloc(&new_offsets, ((int64_t)N + 1) * sizeof(int32_t));
            cudaMalloc(&out_weights, N * sizeof(double));
            cudaMalloc(&dangling_list, N * sizeof(int32_t));
            cudaMalloc(&dangling_count, sizeof(int32_t));
            cudaMalloc(&pers_norm, N * sizeof(double));
            cudaMalloc(&pr_a64, N * sizeof(double));
            cudaMalloc(&pr_b64, N * sizeof(double));
            cudaMalloc(&pr_a32, N * sizeof(float));
            cudaMalloc(&pr_b32, N * sizeof(float));
            cudaMalloc(&scalars, 2 * sizeof(double));

            vertex_cap = N;
        }
    }

    void ensure_edge(int32_t ta) {
        if (edge_cap < ta) {
            if (new_indices) cudaFree(new_indices);
            if (new_weights) cudaFree(new_weights);
            if (norm_weights_f32) cudaFree(norm_weights_f32);

            cudaMalloc(&new_indices, ta * sizeof(int32_t));
            cudaMalloc(&new_weights, ta * sizeof(double));
            cudaMalloc(&norm_weights_f32, ta * sizeof(float));

            edge_cap = ta;
        }
    }

    void ensure_cub(size_t tb) {
        if (cub_cap < tb) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, tb);
            cub_cap = tb;
        }
    }

    ~Cache() override {
        if (active_count) cudaFree(active_count);
        if (new_offsets) cudaFree(new_offsets);
        if (out_weights) cudaFree(out_weights);
        if (dangling_list) cudaFree(dangling_list);
        if (dangling_count) cudaFree(dangling_count);
        if (pers_norm) cudaFree(pers_norm);
        if (pr_a64) cudaFree(pr_a64);
        if (pr_b64) cudaFree(pr_b64);
        if (pr_a32) cudaFree(pr_a32);
        if (pr_b32) cudaFree(pr_b32);
        if (scalars) cudaFree(scalars);
        if (new_indices) cudaFree(new_indices);
        if (new_weights) cudaFree(new_weights);
        if (norm_weights_f32) cudaFree(norm_weights_f32);
        if (cub_temp) cudaFree(cub_temp);
    }
};




__device__ __forceinline__ bool edge_active(const uint32_t* mask, int32_t e) {
    return (mask[e >> 5] >> (e & 31)) & 1;
}




__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets, const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_count, int32_t N)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    int32_t s = offsets[v], e = offsets[v + 1], cnt = 0;
    int32_t i = s;
    while (i < e && (i & 31)) { if (edge_active(edge_mask, i)) cnt++; i++; }
    while (i + 32 <= e) { cnt += __popc(edge_mask[i >> 5]); i += 32; }
    while (i < e) { if (edge_active(edge_mask, i)) cnt++; i++; }
    active_count[v] = cnt;
}

__global__ void set_last_offset_kernel(int32_t* no, const int32_t* cnt, int32_t N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (N > 0) no[N] = no[N-1] + cnt[N-1]; else no[0] = 0;
    }
}

__global__ void compact_edges_kernel(
    const int32_t* __restrict__ oo, const int32_t* __restrict__ idx,
    const double* __restrict__ w, const uint32_t* __restrict__ mask,
    const int32_t* __restrict__ no,
    int32_t* __restrict__ ni, double* __restrict__ nw, int32_t N)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    int32_t os = oo[v], oe = oo[v+1], pos = no[v];
    for (int32_t i = os; i < oe; i++) {
        if (edge_active(mask, i)) { ni[pos] = idx[i]; nw[pos] = w[i]; pos++; }
    }
}




__global__ void compute_out_weights_kernel(
    const int32_t* __restrict__ idx, const double* __restrict__ w,
    double* __restrict__ ow, int32_t E)
{
    int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    atomicAdd(&ow[idx[e]], w[e]);
}

__global__ void precompute_norm_weights_kernel(
    const int32_t* __restrict__ idx, const double* __restrict__ w,
    const double* __restrict__ ow, float* __restrict__ nwf, int32_t E)
{
    int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= E) return;
    double o = ow[idx[e]];
    nwf[e] = (o > 0.0) ? (float)(w[e] / o) : 0.0f;
}

__global__ void build_dangling_list_kernel(
    const double* __restrict__ ow, int32_t* __restrict__ dl,
    int32_t* __restrict__ nd, int32_t N)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    if (ow[v] == 0.0) { int32_t p = atomicAdd(nd, 1); dl[p] = v; }
}




__global__ void init_pr_uniform_kernel(double* pr, float* prf, int32_t N) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N) { double val = 1.0 / (double)N; pr[v] = val; prf[v] = (float)val; }
}

__global__ void copy_and_convert_kernel(const double* __restrict__ src, double* __restrict__ dst, float* __restrict__ dstf, int32_t N) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N) { double val = src[v]; dst[v] = val; dstf[v] = (float)val; }
}

__global__ void pers_sum_and_scatter_kernel(
    const int32_t* __restrict__ pv, const double* __restrict__ vals,
    double* __restrict__ norm, int32_t sz)
{
    __shared__ double s_sum;
    if (threadIdx.x == 0) {
        double sum = 0.0;
        for (int i = 0; i < sz; i++) sum += vals[i];
        s_sum = sum;
    }
    __syncthreads();
    double inv = 1.0 / s_sum;
    for (int i = threadIdx.x; i < sz; i += blockDim.x)
        norm[pv[i]] = vals[i] * inv;
}




__global__ void dangling_sum_kernel(
    const double* __restrict__ pr, const int32_t* __restrict__ dl,
    double* __restrict__ ds, int32_t nd)
{
    typedef cub::BlockReduce<double, 256> BR;
    __shared__ typename BR::TempStorage temp;
    int32_t i = blockIdx.x * 256 + threadIdx.x;
    double val = (i < nd) ? pr[dl[i]] : 0.0;
    double bs = BR(temp).Sum(val);
    if (threadIdx.x == 0 && bs != 0.0) atomicAdd(ds, bs);
}







template <bool DIFF>
__global__ void spmv_block_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ nw,
    const float* __restrict__ pr_old_f32,
    const double* __restrict__ pr_old_f64,
    double* __restrict__ pr_new_f64,
    float* __restrict__ pr_new_f32,
    const double* __restrict__ pn,
    double alpha, const double* __restrict__ ds_ptr,
    int32_t v_start, double* __restrict__ diff_sum)
{
    typedef cub::BlockReduce<double, 256> BR;
    __shared__ typename BR::TempStorage temp;
    int32_t v = v_start + blockIdx.x;
    int32_t s = offsets[v], e = offsets[v+1];
    double sum = 0.0;
    for (int32_t i = s + threadIdx.x; i < e; i += 256)
        sum += (double)nw[i] * (double)pr_old_f32[indices[i]];
    double bs = BR(temp).Sum(sum);
    if (threadIdx.x == 0) {
        double bf = alpha * (*ds_ptr) + (1.0 - alpha);
        double nv = alpha * bs + bf * pn[v];
        pr_new_f64[v] = nv;
        pr_new_f32[v] = (float)nv;
        if constexpr (DIFF) {
            double d = nv - pr_old_f64[v];
            atomicAdd(diff_sum, d > 0 ? d : -d);
        }
    }
}


template <bool DIFF>
__global__ void spmv_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ nw,
    const float* __restrict__ pr_old_f32,
    const double* __restrict__ pr_old_f64,
    double* __restrict__ pr_new_f64,
    float* __restrict__ pr_new_f32,
    const double* __restrict__ pn,
    double alpha, const double* __restrict__ ds_ptr,
    int32_t v_start, int32_t v_end, double* __restrict__ diff_sum)
{
    typedef cub::BlockReduce<double, 256> BR;
    __shared__ typename BR::TempStorage temp;
    int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    int32_t v = v_start + blockIdx.x * 8 + w;
    double ld = 0.0;
    if (v < v_end) {
        int32_t s = offsets[v], e = offsets[v+1];
        double sum = 0.0;
        for (int32_t i = s + l; i < e; i += 32)
            sum += (double)nw[i] * (double)pr_old_f32[indices[i]];
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, off);
        if (l == 0) {
            double bf = alpha * (*ds_ptr) + (1.0 - alpha);
            double nv = alpha * sum + bf * pn[v];
            pr_new_f64[v] = nv;
            pr_new_f32[v] = (float)nv;
            if constexpr (DIFF) {
                double d = nv - pr_old_f64[v];
                ld = d > 0 ? d : -d;
            }
        }
    }
    if constexpr (DIFF) {
        double bd = BR(temp).Sum(ld);
        if (threadIdx.x == 0 && bd != 0.0) atomicAdd(diff_sum, bd);
    }
}


template <bool DIFF>
__global__ void spmv_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ nw,
    const float* __restrict__ pr_old_f32,
    const double* __restrict__ pr_old_f64,
    double* __restrict__ pr_new_f64,
    float* __restrict__ pr_new_f32,
    const double* __restrict__ pn,
    double alpha, const double* __restrict__ ds_ptr,
    int32_t v_start, int32_t v_end, double* __restrict__ diff_sum)
{
    typedef cub::BlockReduce<double, 256> BR;
    __shared__ typename BR::TempStorage temp;
    int32_t v = v_start + blockIdx.x * 256 + threadIdx.x;
    double ld = 0.0;
    if (v < v_end) {
        int32_t s = offsets[v], e = offsets[v+1];
        double sum = 0.0;
        for (int32_t i = s; i < e; i++)
            sum += (double)nw[i] * (double)pr_old_f32[indices[i]];
        double bf = alpha * (*ds_ptr) + (1.0 - alpha);
        double nv = alpha * sum + bf * pn[v];
        pr_new_f64[v] = nv;
        pr_new_f32[v] = (float)nv;
        if constexpr (DIFF) {
            double d = nv - pr_old_f64[v];
            ld = d > 0 ? d : -d;
        }
    }
    if constexpr (DIFF) {
        double bd = BR(temp).Sum(ld);
        if (threadIdx.x == 0 && bd != 0.0) atomicAdd(diff_sum, bd);
    }
}

}  

PageRankResult personalized_pagerank_seg_mask(const graph32_t& graph,
                                              const double* edge_weights,
                                              const int32_t* personalization_vertices,
                                              const double* personalization_values,
                                              std::size_t personalization_size,
                                              double* pageranks,
                                              const double* precomputed_vertex_out_weight_sums,
                                              double alpha,
                                              double epsilon,
                                              std::size_t max_iterations,
                                              const double* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t N = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_mask = graph.edge_mask;
    const double* d_weights = edge_weights;
    cudaStream_t stream = 0;

    const auto& seg_vec = graph.segment_offsets.value();
    int32_t seg[5] = {seg_vec[0], seg_vec[1], seg_vec[2], seg_vec[3], seg_vec[4]};

    cache.ensure_vertex(N);

    
    int32_t* d_ac = cache.active_count;
    if (N > 0)
        count_active_edges_kernel<<<(N+255)/256, 256, 0, stream>>>(d_offsets, d_mask, d_ac, N);

    int32_t* d_no = cache.new_offsets;
    size_t tb = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, tb, d_ac, d_no, N, stream);
    cache.ensure_cub(tb);
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, tb, d_ac, d_no, N, stream);
    set_last_offset_kernel<<<1, 1, 0, stream>>>(d_no, d_ac, N);

    int32_t ta = 0;
    cudaMemcpy(&ta, d_no + N, sizeof(int32_t), cudaMemcpyDeviceToHost);
    int32_t ta_alloc = (ta > 0) ? ta : 1;

    cache.ensure_edge(ta_alloc);
    int32_t* d_ni = cache.new_indices;
    double* d_nw = cache.new_weights;
    if (N > 0)
        compact_edges_kernel<<<(N+255)/256, 256, 0, stream>>>(d_offsets, d_indices, d_weights, d_mask, d_no, d_ni, d_nw, N);

    
    double* d_ow = cache.out_weights;
    cudaMemsetAsync(d_ow, 0, N * sizeof(double), stream);
    if (ta > 0)
        compute_out_weights_kernel<<<(ta+255)/256, 256, 0, stream>>>(d_ni, d_nw, d_ow, ta);

    float* d_nwf = cache.norm_weights_f32;
    if (ta > 0)
        precompute_norm_weights_kernel<<<(ta+255)/256, 256, 0, stream>>>(d_ni, d_nw, d_ow, d_nwf, ta);

    
    int32_t* d_dl = cache.dangling_list;
    int32_t* d_nd = cache.dangling_count;
    cudaMemsetAsync(d_nd, 0, sizeof(int32_t), stream);
    if (N > 0)
        build_dangling_list_kernel<<<(N+255)/256, 256, 0, stream>>>(d_ow, d_dl, d_nd, N);
    int32_t h_nd = 0;
    cudaMemcpy(&h_nd, d_nd, sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    double* d_pn = cache.pers_norm;
    cudaMemsetAsync(d_pn, 0, N * sizeof(double), stream);
    if (personalization_size > 0)
        pers_sum_and_scatter_kernel<<<1, 256, 0, stream>>>(personalization_vertices, personalization_values, d_pn, (int32_t)personalization_size);

    
    double* d_pra64 = cache.pr_a64;
    double* d_prb64 = cache.pr_b64;
    float* d_pra32 = cache.pr_a32;
    float* d_prb32 = cache.pr_b32;

    if (initial_pageranks != nullptr) {
        if (N > 0)
            copy_and_convert_kernel<<<(N+255)/256, 256, 0, stream>>>(initial_pageranks, d_pra64, d_pra32, N);
    } else {
        if (N > 0)
            init_pr_uniform_kernel<<<(N+255)/256, 256, 0, stream>>>(d_pra64, d_pra32, N);
    }

    
    double* d_ds = cache.scalars;
    double* d_df = d_ds + 1;

    int32_t hs = seg[0], hc = seg[1]-seg[0];
    int32_t ms = seg[1], mc = seg[2]-seg[1];
    int32_t ls = seg[2], lc = seg[4]-seg[2];

    
    double* pr_old64 = d_pra64, *pr_new64 = d_prb64;
    float* pr_old32 = d_pra32, *pr_new32 = d_prb32;
    bool converged = false;
    std::size_t iters = 0;
    const int CHK = 4;

    for (std::size_t it = 0; it < max_iterations; it++) {
        bool check = (it % CHK == CHK-1) || (it == max_iterations-1);
        cudaMemsetAsync(d_ds, 0, check ? 16 : 8, stream);

        if (h_nd > 0)
            dangling_sum_kernel<<<(h_nd+255)/256, 256, 0, stream>>>(pr_old64, d_dl, d_ds, h_nd);

        if (check) {
            if (hc > 0)
                spmv_block_kernel<true><<<hc, 256, 0, stream>>>(d_no, d_ni, d_nwf, pr_old32, pr_old64, pr_new64, pr_new32, d_pn, alpha, d_ds, hs, d_df);
            if (mc > 0)
                spmv_warp_kernel<true><<<(mc+7)/8, 256, 0, stream>>>(d_no, d_ni, d_nwf, pr_old32, pr_old64, pr_new64, pr_new32, d_pn, alpha, d_ds, ms, ms+mc, d_df);
            if (lc > 0)
                spmv_thread_kernel<true><<<(lc+255)/256, 256, 0, stream>>>(d_no, d_ni, d_nwf, pr_old32, pr_old64, pr_new64, pr_new32, d_pn, alpha, d_ds, ls, ls+lc, d_df);
        } else {
            if (hc > 0)
                spmv_block_kernel<false><<<hc, 256, 0, stream>>>(d_no, d_ni, d_nwf, pr_old32, pr_old64, pr_new64, pr_new32, d_pn, alpha, d_ds, hs, nullptr);
            if (mc > 0)
                spmv_warp_kernel<false><<<(mc+7)/8, 256, 0, stream>>>(d_no, d_ni, d_nwf, pr_old32, pr_old64, pr_new64, pr_new32, d_pn, alpha, d_ds, ms, ms+mc, nullptr);
            if (lc > 0)
                spmv_thread_kernel<false><<<(lc+255)/256, 256, 0, stream>>>(d_no, d_ni, d_nwf, pr_old32, pr_old64, pr_new64, pr_new32, d_pn, alpha, d_ds, ls, ls+lc, nullptr);
        }

        iters = it + 1;
        if (check) {
            double hd;
            cudaMemcpy(&hd, d_df, sizeof(double), cudaMemcpyDeviceToHost);
            if (hd < epsilon) { converged = true; break; }
        }
        std::swap(pr_old64, pr_new64);
        std::swap(pr_old32, pr_new32);
    }

    
    double* result = converged ? pr_new64 : pr_old64;
    if (N > 0)
        cudaMemcpyAsync(pageranks, result, (size_t)N * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return PageRankResult{iters, converged};
}

}  
