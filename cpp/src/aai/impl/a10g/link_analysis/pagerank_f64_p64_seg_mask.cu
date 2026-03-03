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
    
    int32_t* d_ac = nullptr;        
    int64_t d_ac_cap = 0;

    int32_t* d_no = nullptr;        
    int64_t d_no_cap = 0;

    int32_t* d_ni = nullptr;        
    int64_t d_ni_cap = 0;

    double* d_nw = nullptr;         
    int64_t d_nw_cap = 0;

    uint8_t* d_cub_tmp = nullptr;   
    size_t d_cub_tmp_cap = 0;

    
    double* d_ow = nullptr;         
    int64_t d_ow_cap = 0;

    float* d_nwf = nullptr;         
    int64_t d_nwf_cap = 0;

    
    int32_t* d_dl = nullptr;        
    int64_t d_dl_cap = 0;

    int32_t* d_nd = nullptr;        
    bool d_nd_alloc = false;

    
    double* d_pn = nullptr;         
    int64_t d_pn_cap = 0;

    
    double* d_pra64 = nullptr;
    double* d_prb64 = nullptr;
    float* d_pra32 = nullptr;
    float* d_prb32 = nullptr;
    int64_t d_pr_cap = 0;

    
    double* d_sc = nullptr;         
    bool d_sc_alloc = false;

    void ensure(int32_t N, int64_t max_edges) {
        int64_t N64 = (int64_t)N;
        int64_t Np1 = N64 + 1;
        int64_t E64 = (max_edges > 0) ? max_edges : 1;

        if (d_ac_cap < N64) {
            if (d_ac) cudaFree(d_ac);
            cudaMalloc(&d_ac, N64 * sizeof(int32_t));
            d_ac_cap = N64;
        }
        if (d_no_cap < Np1) {
            if (d_no) cudaFree(d_no);
            cudaMalloc(&d_no, Np1 * sizeof(int32_t));
            d_no_cap = Np1;
        }
        if (d_ni_cap < E64) {
            if (d_ni) cudaFree(d_ni);
            cudaMalloc(&d_ni, E64 * sizeof(int32_t));
            d_ni_cap = E64;
        }
        if (d_nw_cap < E64) {
            if (d_nw) cudaFree(d_nw);
            cudaMalloc(&d_nw, E64 * sizeof(double));
            d_nw_cap = E64;
        }
        if (d_ow_cap < N64) {
            if (d_ow) cudaFree(d_ow);
            cudaMalloc(&d_ow, N64 * sizeof(double));
            d_ow_cap = N64;
        }
        if (d_nwf_cap < E64) {
            if (d_nwf) cudaFree(d_nwf);
            cudaMalloc(&d_nwf, E64 * sizeof(float));
            d_nwf_cap = E64;
        }
        if (d_dl_cap < N64) {
            if (d_dl) cudaFree(d_dl);
            cudaMalloc(&d_dl, N64 * sizeof(int32_t));
            d_dl_cap = N64;
        }
        if (!d_nd_alloc) {
            cudaMalloc(&d_nd, sizeof(int32_t));
            d_nd_alloc = true;
        }
        if (d_pn_cap < N64) {
            if (d_pn) cudaFree(d_pn);
            cudaMalloc(&d_pn, N64 * sizeof(double));
            d_pn_cap = N64;
        }
        if (d_pr_cap < N64) {
            if (d_pra64) cudaFree(d_pra64);
            if (d_prb64) cudaFree(d_prb64);
            if (d_pra32) cudaFree(d_pra32);
            if (d_prb32) cudaFree(d_prb32);
            cudaMalloc(&d_pra64, N64 * sizeof(double));
            cudaMalloc(&d_prb64, N64 * sizeof(double));
            cudaMalloc(&d_pra32, N64 * sizeof(float));
            cudaMalloc(&d_prb32, N64 * sizeof(float));
            d_pr_cap = N64;
        }
        if (!d_sc_alloc) {
            cudaMalloc(&d_sc, 2 * sizeof(double));
            d_sc_alloc = true;
        }
    }

    void ensure_cub_tmp(size_t needed) {
        if (d_cub_tmp_cap < needed) {
            if (d_cub_tmp) cudaFree(d_cub_tmp);
            cudaMalloc(&d_cub_tmp, needed);
            d_cub_tmp_cap = needed;
        }
    }

    ~Cache() override {
        if (d_ac) cudaFree(d_ac);
        if (d_no) cudaFree(d_no);
        if (d_ni) cudaFree(d_ni);
        if (d_nw) cudaFree(d_nw);
        if (d_cub_tmp) cudaFree(d_cub_tmp);
        if (d_ow) cudaFree(d_ow);
        if (d_nwf) cudaFree(d_nwf);
        if (d_dl) cudaFree(d_dl);
        if (d_nd) cudaFree(d_nd);
        if (d_pn) cudaFree(d_pn);
        if (d_pra64) cudaFree(d_pra64);
        if (d_prb64) cudaFree(d_prb64);
        if (d_pra32) cudaFree(d_pra32);
        if (d_prb32) cudaFree(d_prb32);
        if (d_sc) cudaFree(d_sc);
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

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t N = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const uint32_t* d_mask = graph.edge_mask;
    cudaStream_t stream = 0;

    
    const auto& seg_vec = graph.segment_offsets.value();
    int32_t seg[5];
    for (int i = 0; i < 5; i++) seg[i] = seg_vec[i];

    
    cache.ensure(N, (int64_t)num_edges);

    
    
    
    if (N > 0)
        count_active_edges_kernel<<<(N+255)/256, 256, 0, stream>>>(d_offsets, d_mask, cache.d_ac, N);

    
    size_t cub_tmp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_tmp_bytes, cache.d_ac, cache.d_no, N, stream);
    cache.ensure_cub_tmp(cub_tmp_bytes);
    cub::DeviceScan::ExclusiveSum(cache.d_cub_tmp, cub_tmp_bytes, cache.d_ac, cache.d_no, N, stream);

    set_last_offset_kernel<<<1, 1, 0, stream>>>(cache.d_no, cache.d_ac, N);

    int32_t ta = 0;
    cudaMemcpy(&ta, cache.d_no + N, sizeof(int32_t), cudaMemcpyDeviceToHost);
    int64_t ta64 = (ta > 0) ? (int64_t)ta : 1;

    
    if (cache.d_ni_cap < ta64) {
        if (cache.d_ni) cudaFree(cache.d_ni);
        cudaMalloc(&cache.d_ni, ta64 * sizeof(int32_t));
        cache.d_ni_cap = ta64;
    }
    if (cache.d_nw_cap < ta64) {
        if (cache.d_nw) cudaFree(cache.d_nw);
        cudaMalloc(&cache.d_nw, ta64 * sizeof(double));
        cache.d_nw_cap = ta64;
    }
    if (cache.d_nwf_cap < ta64) {
        if (cache.d_nwf) cudaFree(cache.d_nwf);
        cudaMalloc(&cache.d_nwf, ta64 * sizeof(float));
        cache.d_nwf_cap = ta64;
    }

    if (N > 0)
        compact_edges_kernel<<<(N+255)/256, 256, 0, stream>>>(
            d_offsets, d_indices, edge_weights, d_mask, cache.d_no,
            cache.d_ni, cache.d_nw, N);

    
    
    
    cudaMemsetAsync(cache.d_ow, 0, N * sizeof(double), stream);
    if (ta > 0)
        compute_out_weights_kernel<<<(ta+255)/256, 256, 0, stream>>>(
            cache.d_ni, cache.d_nw, cache.d_ow, ta);

    if (ta > 0)
        precompute_norm_weights_kernel<<<(ta+255)/256, 256, 0, stream>>>(
            cache.d_ni, cache.d_nw, cache.d_ow, cache.d_nwf, ta);

    
    
    
    cudaMemsetAsync(cache.d_nd, 0, sizeof(int32_t), stream);
    if (N > 0)
        build_dangling_list_kernel<<<(N+255)/256, 256, 0, stream>>>(
            cache.d_ow, cache.d_dl, cache.d_nd, N);
    int32_t h_nd = 0;
    cudaMemcpy(&h_nd, cache.d_nd, sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    
    
    cudaMemsetAsync(cache.d_pn, 0, N * sizeof(double), stream);
    if (personalization_size > 0)
        pers_sum_and_scatter_kernel<<<1, 256, 0, stream>>>(
            personalization_vertices, personalization_values,
            cache.d_pn, (int32_t)personalization_size);

    
    
    
    double* d_pra64 = cache.d_pra64;
    double* d_prb64 = cache.d_prb64;
    float* d_pra32 = cache.d_pra32;
    float* d_prb32 = cache.d_prb32;

    if (initial_pageranks != nullptr) {
        if (N > 0)
            copy_and_convert_kernel<<<(N+255)/256, 256, 0, stream>>>(
                initial_pageranks, d_pra64, d_pra32, N);
    } else {
        if (N > 0)
            init_pr_uniform_kernel<<<(N+255)/256, 256, 0, stream>>>(d_pra64, d_pra32, N);
    }

    
    
    
    double* d_ds = cache.d_sc;
    double* d_df = cache.d_sc + 1;

    int32_t hs = seg[0], hc = seg[1] - seg[0];
    int32_t ms = seg[1], mc = seg[2] - seg[1];
    int32_t ls = seg[2], lc = seg[4] - seg[2];

    
    
    
    double* pr_old64 = d_pra64;
    double* pr_new64 = d_prb64;
    float* pr_old32 = d_pra32;
    float* pr_new32 = d_prb32;
    bool converged = false;
    std::size_t iters = 0;
    const int CHK = 4;

    for (std::size_t it = 0; it < max_iterations; it++) {
        bool check = (it % CHK == CHK - 1) || (it == max_iterations - 1);
        cudaMemsetAsync(d_ds, 0, check ? 16 : 8, stream);

        if (h_nd > 0)
            dangling_sum_kernel<<<(h_nd+255)/256, 256, 0, stream>>>(pr_old64, cache.d_dl, d_ds, h_nd);

        if (check) {
            if (hc > 0)
                spmv_block_kernel<true><<<hc, 256, 0, stream>>>(
                    cache.d_no, cache.d_ni, cache.d_nwf, pr_old32, pr_old64,
                    pr_new64, pr_new32, cache.d_pn, alpha, d_ds, hs, d_df);
            if (mc > 0)
                spmv_warp_kernel<true><<<(mc+7)/8, 256, 0, stream>>>(
                    cache.d_no, cache.d_ni, cache.d_nwf, pr_old32, pr_old64,
                    pr_new64, pr_new32, cache.d_pn, alpha, d_ds, ms, ms+mc, d_df);
            if (lc > 0)
                spmv_thread_kernel<true><<<(lc+255)/256, 256, 0, stream>>>(
                    cache.d_no, cache.d_ni, cache.d_nwf, pr_old32, pr_old64,
                    pr_new64, pr_new32, cache.d_pn, alpha, d_ds, ls, ls+lc, d_df);
        } else {
            if (hc > 0)
                spmv_block_kernel<false><<<hc, 256, 0, stream>>>(
                    cache.d_no, cache.d_ni, cache.d_nwf, pr_old32, pr_old64,
                    pr_new64, pr_new32, cache.d_pn, alpha, d_ds, hs, nullptr);
            if (mc > 0)
                spmv_warp_kernel<false><<<(mc+7)/8, 256, 0, stream>>>(
                    cache.d_no, cache.d_ni, cache.d_nwf, pr_old32, pr_old64,
                    pr_new64, pr_new32, cache.d_pn, alpha, d_ds, ms, ms+mc, nullptr);
            if (lc > 0)
                spmv_thread_kernel<false><<<(lc+255)/256, 256, 0, stream>>>(
                    cache.d_no, cache.d_ni, cache.d_nwf, pr_old32, pr_old64,
                    pr_new64, pr_new32, cache.d_pn, alpha, d_ds, ls, ls+lc, nullptr);
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

    
    
    
    const double* result_ptr = converged ? pr_new64 : pr_old64;
    if (result_ptr != pageranks) {
        cudaMemcpyAsync(pageranks, result_ptr, (std::size_t)N * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    return PageRankResult{iters, converged};
}

}  
