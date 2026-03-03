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
    
    double* ow = nullptr;
    int32_t* dl = nullptr;
    double* pn = nullptr;
    double* pra64 = nullptr;
    double* prb64 = nullptr;
    float* pra32 = nullptr;
    float* prb32 = nullptr;
    int32_t N_cap = 0;

    
    float* nwf = nullptr;
    int32_t E_cap = 0;

    
    int32_t* nd = nullptr;
    double* sc = nullptr;
    bool small_alloc = false;

    void ensure(int32_t N, int32_t E) {
        if (N_cap < N) {
            if (ow) cudaFree(ow);
            if (dl) cudaFree(dl);
            if (pn) cudaFree(pn);
            if (pra64) cudaFree(pra64);
            if (prb64) cudaFree(prb64);
            if (pra32) cudaFree(pra32);
            if (prb32) cudaFree(prb32);
            cudaMalloc(&ow, (size_t)N * sizeof(double));
            cudaMalloc(&dl, (size_t)N * sizeof(int32_t));
            cudaMalloc(&pn, (size_t)N * sizeof(double));
            cudaMalloc(&pra64, (size_t)N * sizeof(double));
            cudaMalloc(&prb64, (size_t)N * sizeof(double));
            cudaMalloc(&pra32, (size_t)N * sizeof(float));
            cudaMalloc(&prb32, (size_t)N * sizeof(float));
            N_cap = N;
        }
        if (E_cap < E) {
            if (nwf) cudaFree(nwf);
            int64_t E64 = (E > 0) ? (int64_t)E : 1;
            cudaMalloc(&nwf, E64 * sizeof(float));
            E_cap = E;
        }
        if (!small_alloc) {
            cudaMalloc(&nd, sizeof(int32_t));
            cudaMalloc(&sc, 2 * sizeof(double));
            small_alloc = true;
        }
    }

    ~Cache() override {
        if (ow) cudaFree(ow);
        if (dl) cudaFree(dl);
        if (pn) cudaFree(pn);
        if (pra64) cudaFree(pra64);
        if (prb64) cudaFree(prb64);
        if (pra32) cudaFree(pra32);
        if (prb32) cudaFree(prb32);
        if (nwf) cudaFree(nwf);
        if (nd) cudaFree(nd);
        if (sc) cudaFree(sc);
    }
};




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
    int32_t s = offsets[v], e = offsets[v + 1];
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
        int32_t s = offsets[v], e = offsets[v + 1];
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
        int32_t s = offsets[v], e = offsets[v + 1];
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





void launch_compute_out_weights(const int32_t* idx, const double* w,
    double* ow, int32_t E, cudaStream_t s) {
    if (E > 0) compute_out_weights_kernel<<<(E + 255) / 256, 256, 0, s>>>(idx, w, ow, E);
}
void launch_precompute_norm_weights(const int32_t* idx, const double* w,
    const double* ow, float* nwf, int32_t E, cudaStream_t s) {
    if (E > 0) precompute_norm_weights_kernel<<<(E + 255) / 256, 256, 0, s>>>(idx, w, ow, nwf, E);
}
void launch_build_dangling_list(const double* ow, int32_t* dl,
    int32_t* nd, int32_t N, cudaStream_t s) {
    if (N > 0) build_dangling_list_kernel<<<(N + 255) / 256, 256, 0, s>>>(ow, dl, nd, N);
}
void launch_init_pr_uniform(double* pr, float* prf, int32_t N, cudaStream_t s) {
    if (N > 0) init_pr_uniform_kernel<<<(N + 255) / 256, 256, 0, s>>>(pr, prf, N);
}
void launch_copy_and_convert(const double* src, double* dst, float* dstf, int32_t N, cudaStream_t s) {
    if (N > 0) copy_and_convert_kernel<<<(N + 255) / 256, 256, 0, s>>>(src, dst, dstf, N);
}
void launch_pers(const int32_t* v, const double* val, double* n, int32_t sz, cudaStream_t s) {
    if (sz > 0) pers_sum_and_scatter_kernel<<<1, 256, 0, s>>>(v, val, n, sz);
}
void launch_dangling_sum(const double* pr, const int32_t* dl, double* ds, int32_t nd, cudaStream_t s) {
    if (nd > 0) dangling_sum_kernel<<<(nd + 255) / 256, 256, 0, s>>>(pr, dl, ds, nd);
}


void launch_spmv_block_d(const int32_t* off, const int32_t* idx, const float* nw,
    const float* pof, const double* pod, double* pnd, float* pnf,
    const double* pn, double a, const double* ds,
    int32_t vs, int32_t vc, double* diff, cudaStream_t s) {
    if (vc > 0) spmv_block_kernel<true><<<vc, 256, 0, s>>>(off, idx, nw, pof, pod, pnd, pnf, pn, a, ds, vs, diff);
}
void launch_spmv_warp_d(const int32_t* off, const int32_t* idx, const float* nw,
    const float* pof, const double* pod, double* pnd, float* pnf,
    const double* pn, double a, const double* ds,
    int32_t vs, int32_t vc, double* diff, cudaStream_t s) {
    if (vc > 0) spmv_warp_kernel<true><<<(vc + 7) / 8, 256, 0, s>>>(off, idx, nw, pof, pod, pnd, pnf, pn, a, ds, vs, vs + vc, diff);
}
void launch_spmv_thread_d(const int32_t* off, const int32_t* idx, const float* nw,
    const float* pof, const double* pod, double* pnd, float* pnf,
    const double* pn, double a, const double* ds,
    int32_t vs, int32_t vc, double* diff, cudaStream_t s) {
    if (vc > 0) spmv_thread_kernel<true><<<(vc + 255) / 256, 256, 0, s>>>(off, idx, nw, pof, pod, pnd, pnf, pn, a, ds, vs, vs + vc, diff);
}

void launch_spmv_block_n(const int32_t* off, const int32_t* idx, const float* nw,
    const float* pof, const double* pod, double* pnd, float* pnf,
    const double* pn, double a, const double* ds,
    int32_t vs, int32_t vc, cudaStream_t s) {
    if (vc > 0) spmv_block_kernel<false><<<vc, 256, 0, s>>>(off, idx, nw, pof, pod, pnd, pnf, pn, a, ds, vs, nullptr);
}
void launch_spmv_warp_n(const int32_t* off, const int32_t* idx, const float* nw,
    const float* pof, const double* pod, double* pnd, float* pnf,
    const double* pn, double a, const double* ds,
    int32_t vs, int32_t vc, cudaStream_t s) {
    if (vc > 0) spmv_warp_kernel<false><<<(vc + 7) / 8, 256, 0, s>>>(off, idx, nw, pof, pod, pnd, pnf, pn, a, ds, vs, vs + vc, nullptr);
}
void launch_spmv_thread_n(const int32_t* off, const int32_t* idx, const float* nw,
    const float* pof, const double* pod, double* pnd, float* pnf,
    const double* pn, double a, const double* ds,
    int32_t vs, int32_t vc, cudaStream_t s) {
    if (vc > 0) spmv_thread_kernel<false><<<(vc + 255) / 256, 256, 0, s>>>(off, idx, nw, pof, pod, pnd, pnf, pn, a, ds, vs, vs + vc, nullptr);
}

}  

PageRankResult personalized_pagerank_seg(const graph32_t& graph,
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
    int32_t E = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const double* d_weights = edge_weights;

    const auto& seg_v = graph.segment_offsets.value();
    int32_t seg[5] = {seg_v[0], seg_v[1], seg_v[2], seg_v[3], seg_v[4]};

    cache.ensure(N, E);
    cudaStream_t stream = 0;

    
    cudaMemsetAsync(cache.ow, 0, (size_t)N * sizeof(double), stream);
    launch_compute_out_weights(d_indices, d_weights, cache.ow, E, stream);

    
    launch_precompute_norm_weights(d_indices, d_weights, cache.ow, cache.nwf, E, stream);

    
    cudaMemsetAsync(cache.nd, 0, sizeof(int32_t), stream);
    launch_build_dangling_list(cache.ow, cache.dl, cache.nd, N, stream);
    int32_t h_nd = 0;
    cudaMemcpy(&h_nd, cache.nd, sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    cudaMemsetAsync(cache.pn, 0, (size_t)N * sizeof(double), stream);
    launch_pers(personalization_vertices, personalization_values,
        cache.pn, (int32_t)personalization_size, stream);

    
    if (initial_pageranks != nullptr)
        launch_copy_and_convert(initial_pageranks, cache.pra64, cache.pra32, N, stream);
    else
        launch_init_pr_uniform(cache.pra64, cache.pra32, N, stream);

    
    double* d_ds = cache.sc;
    double* d_df = cache.sc + 1;

    
    int32_t hs = seg[0], hc = seg[1] - seg[0];
    int32_t ms = seg[1], mc = seg[2] - seg[1];
    int32_t ls = seg[2], lc = seg[4] - seg[2];

    
    double* pr_old64 = cache.pra64, *pr_new64 = cache.prb64;
    float* pr_old32 = cache.pra32, *pr_new32 = cache.prb32;
    bool converged = false;
    std::size_t iters = 0;
    const int CHK = 4;

    for (std::size_t it = 0; it < max_iterations; it++) {
        bool check = (it % CHK == CHK - 1) || (it == max_iterations - 1);
        cudaMemsetAsync(d_ds, 0, check ? 16 : 8, stream);

        if (h_nd > 0)
            launch_dangling_sum(pr_old64, cache.dl, d_ds, h_nd, stream);

        if (check) {
            launch_spmv_block_d(d_offsets, d_indices, cache.nwf, pr_old32, pr_old64, pr_new64, pr_new32, cache.pn, alpha, d_ds, hs, hc, d_df, stream);
            launch_spmv_warp_d(d_offsets, d_indices, cache.nwf, pr_old32, pr_old64, pr_new64, pr_new32, cache.pn, alpha, d_ds, ms, mc, d_df, stream);
            launch_spmv_thread_d(d_offsets, d_indices, cache.nwf, pr_old32, pr_old64, pr_new64, pr_new32, cache.pn, alpha, d_ds, ls, lc, d_df, stream);
        } else {
            launch_spmv_block_n(d_offsets, d_indices, cache.nwf, pr_old32, pr_old64, pr_new64, pr_new32, cache.pn, alpha, d_ds, hs, hc, stream);
            launch_spmv_warp_n(d_offsets, d_indices, cache.nwf, pr_old32, pr_old64, pr_new64, pr_new32, cache.pn, alpha, d_ds, ms, mc, stream);
            launch_spmv_thread_n(d_offsets, d_indices, cache.nwf, pr_old32, pr_old64, pr_new64, pr_new32, cache.pn, alpha, d_ds, ls, lc, stream);
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

    
    double* result_ptr;
    if (converged) {
        result_ptr = pr_new64;
    } else {
        result_ptr = pr_old64;
    }
    cudaMemcpyAsync(pageranks, result_ptr, (size_t)N * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return PageRankResult{iters, converged};
}

}  
