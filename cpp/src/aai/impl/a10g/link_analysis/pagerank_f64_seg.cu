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
#include <cusparse.h>
#include <cub/cub.cuh>
#include <cstdint>
#include <cmath>

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define WARP_SIZE 32


#define CUSPARSE_THRESHOLD 500000

struct Cache : Cacheable {
    cusparseHandle_t handle = nullptr;

    
    float* pr_a = nullptr;
    float* pr_b = nullptr;
    float* nw = nullptr;
    double* ow = nullptr;
    float* scratch = nullptr;
    int32_t* dl = nullptr;
    int32_t* dc = nullptr;

    int64_t pr_capacity = 0;
    int64_t nw_capacity = 0;
    int64_t ow_capacity = 0;
    int64_t dl_capacity = 0;

    
    void* spmv_buf = nullptr;
    size_t spmv_buf_size = 0;

    Cache() {
        cusparseCreate(&handle);
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
        cudaMalloc(&scratch, 8 * sizeof(float));
        cudaMalloc(&dc, sizeof(int32_t));
    }

    ~Cache() override {
        if (handle) cusparseDestroy(handle);
        if (pr_a) cudaFree(pr_a);
        if (pr_b) cudaFree(pr_b);
        if (nw) cudaFree(nw);
        if (ow) cudaFree(ow);
        if (scratch) cudaFree(scratch);
        if (dl) cudaFree(dl);
        if (dc) cudaFree(dc);
        if (spmv_buf) cudaFree(spmv_buf);
    }

    void ensure_pr(int64_t n) {
        if (pr_capacity < n) {
            if (pr_a) cudaFree(pr_a);
            if (pr_b) cudaFree(pr_b);
            cudaMalloc(&pr_a, n * sizeof(float));
            cudaMalloc(&pr_b, n * sizeof(float));
            pr_capacity = n;
        }
    }

    void ensure_nw(int64_t e) {
        if (nw_capacity < e) {
            if (nw) cudaFree(nw);
            cudaMalloc(&nw, e * sizeof(float));
            nw_capacity = e;
        }
    }

    void ensure_ow(int64_t n) {
        if (ow_capacity < n) {
            if (ow) cudaFree(ow);
            cudaMalloc(&ow, n * sizeof(double));
            ow_capacity = n;
        }
    }

    void ensure_dl(int64_t n) {
        if (dl_capacity < n) {
            if (dl) cudaFree(dl);
            cudaMalloc(&dl, n * sizeof(int32_t));
            dl_capacity = n;
        }
    }

    void ensure_spmv_buf(size_t sz) {
        if (spmv_buf_size < sz) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, sz);
            spmv_buf_size = sz;
        }
    }
};




__global__ void compute_out_weights_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    double* __restrict__ out_weights,
    int32_t num_edges)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges) {
        atomicAdd(&out_weights[indices[idx]], weights[idx]);
    }
}

__global__ void compute_norm_weights_fp32_kernel(
    const double* __restrict__ weights,
    const double* __restrict__ out_weights,
    const int32_t* __restrict__ indices,
    float* __restrict__ norm_weights,
    int32_t num_edges)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_edges) {
        double ow = out_weights[indices[idx]];
        norm_weights[idx] = (ow > 0.0) ? (float)(weights[idx] / ow) : 0.0f;
    }
}

__global__ void build_dangling_list_kernel(
    const double* __restrict__ out_weights,
    int32_t* __restrict__ dangling_list,
    int32_t* __restrict__ dangling_count,
    int32_t num_vertices)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices && out_weights[idx] == 0.0) {
        int32_t pos = atomicAdd(dangling_count, 1);
        dangling_list[pos] = idx;
    }
}

__global__ void init_pr_f32_kernel(float* __restrict__ pr, int32_t N, float init_val)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) pr[idx] = init_val;
}




__global__ void compute_dangling_sum_f32_kernel(
    const float* __restrict__ pr,
    const int32_t* __restrict__ dangling_list,
    int32_t num_dangling,
    float* __restrict__ dangling_sum_ptr)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < num_dangling) ? pr[dangling_list[idx]] : 0.0f;
    float block_sum = BlockReduce(temp_storage).Sum(val);
    if (threadIdx.x == 0) atomicAdd(dangling_sum_ptr, block_sum);
}


__global__ void add_base_diff_f32_kernel(
    float* __restrict__ new_pr,
    const float* __restrict__ old_pr,
    float* __restrict__ diff_ptr,
    const float* __restrict__ dangling_ptr,
    float one_minus_alpha_div_n,
    float alpha_div_n,
    int32_t N)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float my_diff = 0.0f;
    if (idx < N) {
        float base = one_minus_alpha_div_n + alpha_div_n * (*dangling_ptr);
        float nv = new_pr[idx] + base;
        new_pr[idx] = nv;
        my_diff = fabsf(nv - old_pr[idx]);
    }
    float block_sum = BlockReduce(temp_storage).Sum(my_diff);
    if (threadIdx.x == 0) atomicAdd(diff_ptr, block_sum);
}


__global__ void spmv_high_f32_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ nw, const float* __restrict__ pr,
    float* __restrict__ new_pr, float* __restrict__ diff_ptr, const float* __restrict__ dang,
    float omad, float ad, float alpha, int32_t start_v, int32_t end_v)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage ts;
    int32_t v = start_v + blockIdx.x;
    if (v >= end_v) return;
    int32_t s = offsets[v], e = offsets[v+1];
    float sum = 0.0f;
    for (int32_t i = s + threadIdx.x; i < e; i += BLOCK_SIZE)
        sum += pr[indices[i]] * nw[i];
    sum = BlockReduce(ts).Sum(sum);
    if (threadIdx.x == 0) {
        float base = omad + ad * (*dang);
        float nv = base + alpha * sum;
        new_pr[v] = nv;
        atomicAdd(diff_ptr, fabsf(nv - pr[v]));
    }
}

__global__ void spmv_mid_f32_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ nw, const float* __restrict__ pr,
    float* __restrict__ new_pr, float* __restrict__ diff_ptr, const float* __restrict__ dang,
    float omad, float ad, float alpha, int32_t start_v, int32_t end_v)
{
    int32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t wid = gtid / WARP_SIZE, lid = gtid % WARP_SIZE;
    int32_t v = start_v + wid;
    if (v >= end_v) return;
    int32_t s = offsets[v], e = offsets[v+1];
    float sum = 0.0f;
    for (int32_t i = s + lid; i < e; i += WARP_SIZE)
        sum += pr[indices[i]] * nw[i];
    for (int32_t off = WARP_SIZE/2; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lid == 0) {
        float base = omad + ad * (*dang);
        float nv = base + alpha * sum;
        new_pr[v] = nv;
        atomicAdd(diff_ptr, fabsf(nv - pr[v]));
    }
}

__global__ void spmv_low_zero_f32_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ nw, const float* __restrict__ pr,
    float* __restrict__ new_pr, float* __restrict__ diff_ptr, const float* __restrict__ dang,
    float omad, float ad, float alpha, int32_t start_v, int32_t end_v)
{
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage ts;
    int32_t v = start_v + blockIdx.x * blockDim.x + threadIdx.x;
    float base = omad + ad * (*dang);
    float my_diff = 0.0f;
    if (v < end_v) {
        int32_t s = offsets[v], e = offsets[v+1];
        float sum = 0.0f;
        for (int32_t i = s; i < e; i++)
            sum += pr[indices[i]] * nw[i];
        float nv = base + alpha * sum;
        new_pr[v] = nv;
        my_diff = fabsf(nv - pr[v]);
    }
    float block_diff = BlockReduce(ts).Sum(my_diff);
    if (threadIdx.x == 0) atomicAdd(diff_ptr, block_diff);
}

__global__ void convert_f32_to_f64_kernel(
    const float* __restrict__ src, double* __restrict__ dst, int32_t N)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = (double)src[idx];
}

__global__ void convert_f64_to_f32_kernel(
    const double* __restrict__ src, float* __restrict__ dst, int32_t N)
{
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) dst[idx] = (float)src[idx];
}

}  

PageRankResult pagerank_seg(const graph32_t& graph,
                            const double* edge_weights,
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
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    if (N == 0) {
        return PageRankResult{0, true};
    }

    const auto& seg = graph.segment_offsets.value();

    float f_alpha = (float)alpha;
    float f_eps = (float)epsilon;
    float inv_n = 1.0f / (float)N;
    float omad = (1.0f - f_alpha) * inv_n;
    float ad = f_alpha * inv_n;

    
    cache.ensure_pr((int64_t)N);
    cache.ensure_nw((int64_t)E);
    cache.ensure_ow((int64_t)N);
    cache.ensure_dl((int64_t)N);

    float* pr_a = cache.pr_a;
    float* pr_b = cache.pr_b;
    float* nw = cache.nw;
    float* scratch = cache.scratch;

    
    float h_vals[8] = {0,0,f_alpha,0, 0,0,0,0};
    cudaMemcpy(scratch, h_vals, 8*sizeof(float), cudaMemcpyHostToDevice);

    
    int ge = ((int64_t)E + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int gv = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMemsetAsync(cache.ow, 0, (int64_t)N * sizeof(double));
    if (E > 0) {
        compute_out_weights_kernel<<<ge, BLOCK_SIZE>>>(indices, edge_weights, cache.ow, E);
        compute_norm_weights_fp32_kernel<<<ge, BLOCK_SIZE>>>(edge_weights, cache.ow, indices, nw, E);
    }
    cudaMemsetAsync(cache.dc, 0, sizeof(int32_t));
    build_dangling_list_kernel<<<gv, BLOCK_SIZE>>>(cache.ow, cache.dl, cache.dc, N);

    int32_t h_nd;
    cudaMemcpy(&h_nd, cache.dc, sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    if (initial_pageranks != nullptr) {
        convert_f64_to_f32_kernel<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(initial_pageranks, pr_a, N);
    } else {
        init_pr_f32_kernel<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(pr_a, N, inv_n);
    }

    
    int avg_deg = (N > 0) ? (E / N) : 0;
    bool use_cusparse = (E >= 500000) && (avg_deg >= 12);

    
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;

    if (use_cusparse) {
        cusparseCreateCsr(&matA, N, N, E,
            (void*)offsets, (void*)indices, (void*)nw,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cusparseCreateDnVec(&vecX, N, pr_a, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, N, pr_b, CUDA_R_32F);
        float h_a = f_alpha, h_z = 0.0f;
        size_t bsz = 0;
        cusparseSpMV_bufferSize(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_a, matA, vecX, &h_z, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bsz);
        cache.ensure_spmv_buf(bsz);
        if (E >= 4000000) {
            cusparseSpMV_preprocess(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &h_a, matA, vecX, &h_z, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);
        }
    }

    float* cur = pr_a;
    float* nxt = pr_b;
    size_t iter = 0;
    bool conv = false;
    float h_diff;

    for (iter = 0; iter < max_iterations; iter++) {
        cudaMemsetAsync(scratch, 0, 2*sizeof(float));
        if (h_nd > 0) {
            compute_dangling_sum_f32_kernel<<<(h_nd+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
                cur, cache.dl, h_nd, &scratch[0]);
        }

        if (use_cusparse) {
            cusparseDnVecSetValues(vecX, cur);
            cusparseDnVecSetValues(vecY, nxt);
            cusparseSpMV(cache.handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &scratch[2], matA, vecX, &scratch[3], vecY,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);
            add_base_diff_f32_kernel<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(
                nxt, cur, &scratch[1], &scratch[0], omad, ad, N);
        } else {
            int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg3 = seg[3], seg4 = seg[4];
            int nvh = seg0, nh = seg1-seg0, nm = seg2-seg1, nl = seg3-seg2, nz = seg4-seg3;
            if (nvh > 0)
                spmv_high_f32_kernel<<<nvh, BLOCK_SIZE>>>(offsets, indices, nw, cur, nxt, &scratch[1], &scratch[0], omad, ad, f_alpha, 0, seg0);
            if (nh > 0)
                spmv_high_f32_kernel<<<nh, BLOCK_SIZE>>>(offsets, indices, nw, cur, nxt, &scratch[1], &scratch[0], omad, ad, f_alpha, seg0, seg1);
            if (nm > 0) {
                int gm = ((int64_t)nm * WARP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
                spmv_mid_f32_kernel<<<gm, BLOCK_SIZE>>>(offsets, indices, nw, cur, nxt, &scratch[1], &scratch[0], omad, ad, f_alpha, seg1, seg2);
            }
            int nlz = nl + nz;
            if (nlz > 0) {
                int glz = (nlz + BLOCK_SIZE - 1) / BLOCK_SIZE;
                spmv_low_zero_f32_kernel<<<glz, BLOCK_SIZE>>>(offsets, indices, nw, cur, nxt, &scratch[1], &scratch[0], omad, ad, f_alpha, seg2, seg4);
            }
        }

        cudaMemcpy(&h_diff, &scratch[1], sizeof(float), cudaMemcpyDeviceToHost);
        float* tmp = cur; cur = nxt; nxt = tmp;
        if (h_diff < f_eps) { conv = true; iter++; break; }
    }

    
    convert_f32_to_f64_kernel<<<(N+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(cur, pageranks, N);
    cudaDeviceSynchronize();

    if (use_cusparse) {
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cusparseDestroySpMat(matA);
    }

    return PageRankResult{iter, conv};
}

}  
