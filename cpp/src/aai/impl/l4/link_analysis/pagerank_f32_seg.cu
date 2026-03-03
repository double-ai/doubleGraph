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
#include <cub/block/block_reduce.cuh>
#include <cstdint>
#include <algorithm>

namespace aai {

namespace {

static constexpr int BLOCK = 256;

__global__ void fill_f32_kernel(float* a, float val, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) a[i] = val;
}

__global__ void scatter_add_kernel(const int* __restrict__ idx, const float* __restrict__ val,
                                    float* __restrict__ out, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) atomicAdd(&out[idx[i]], val[i]);
}

__global__ void norm_weights_kernel(const int* __restrict__ idx, const float* __restrict__ w,
                                     const float* __restrict__ ow, float* __restrict__ nw, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n) nw[i] = w[i] / ow[idx[i]];
}

__global__ void find_dangling_kernel(const float* __restrict__ ow, int* __restrict__ dang_idx,
                                      int* __restrict__ num_dang, int n) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < n && ow[i] == 0.0f) {
        int pos = atomicAdd(num_dang, 1);
        dang_idx[pos] = i;
    }
}

__global__ void dangling_sum_kernel(const float* __restrict__ pr, const int* __restrict__ dang_idx,
                                     int num_dang, float* __restrict__ result) {
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage ts;
    int i = blockIdx.x * BLOCK + threadIdx.x;
    float val = (i < num_dang) ? pr[dang_idx[i]] : 0.0f;
    float s = BR(ts).Sum(val);
    if (threadIdx.x == 0) atomicAdd(result, s);
}


__global__ void update_diff_kernel(
    const float* __restrict__ spmv, const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    float alpha, const float* __restrict__ d_dang, float br, float invN,
    float* __restrict__ d_diff, int V
) {
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage ts;

    float dang_c = alpha * (*d_dang) * invN;
    int i = blockIdx.x * BLOCK + threadIdx.x;
    float my_diff = 0.0f;
    if (i < V) {
        float new_val = br + dang_c + alpha * spmv[i];
        my_diff = fabsf(new_val - pr_old[i]);
        pr_new[i] = new_val;
    }
    float block_diff = BR(ts).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff > 0.0f) atomicAdd(d_diff, block_diff);
}


__global__ void spmv_large_kernel(
    const int* __restrict__ off, const int* __restrict__ idx, const float* __restrict__ nw,
    const float* __restrict__ pr_old, float* __restrict__ pr_new,
    float alpha, const float* __restrict__ d_dang, float br, float invN,
    float* __restrict__ d_diff, int seg_start, int seg_end
) {
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage ts;
    int v = blockIdx.x + seg_start;
    if (v >= seg_end) return;
    int start = off[v], end = off[v + 1];
    float sum = 0.0f;
    for (int j = start + threadIdx.x; j < end; j += BLOCK)
        sum += nw[j] * pr_old[idx[j]];
    sum = BR(ts).Sum(sum);
    if (threadIdx.x == 0) {
        float dang_c = alpha * (*d_dang) * invN;
        float new_val = br + dang_c + alpha * sum;
        atomicAdd(d_diff, fabsf(new_val - pr_old[v]));
        pr_new[v] = new_val;
    }
}

__global__ void spmv_medium_kernel(
    const int* __restrict__ off, const int* __restrict__ idx, const float* __restrict__ nw,
    const float* __restrict__ pr_old, float* __restrict__ pr_new,
    float alpha, const float* __restrict__ d_dang, float br, float invN,
    float* __restrict__ d_diff, int seg_start, int num_verts
) {
    constexpr int WPB = BLOCK / 32;
    int warp_id = (blockIdx.x * WPB) + (threadIdx.x / 32);
    int lane = threadIdx.x & 31;
    float my_diff = 0.0f;
    if (warp_id < num_verts) {
        int v = warp_id + seg_start;
        int start = off[v], end = off[v + 1];
        float sum = 0.0f;
        for (int j = start + lane; j < end; j += 32)
            sum += nw[j] * pr_old[idx[j]];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            sum += __shfl_down_sync(0xFFFFFFFF, sum, o);
        if (lane == 0) {
            float dang_c = alpha * (*d_dang) * invN;
            float new_val = br + dang_c + alpha * sum;
            my_diff = fabsf(new_val - pr_old[v]);
            pr_new[v] = new_val;
        }
    }
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage ts;
    float block_diff = BR(ts).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff > 0.0f) atomicAdd(d_diff, block_diff);
}

__global__ void spmv_small_kernel(
    const int* __restrict__ off, const int* __restrict__ idx, const float* __restrict__ nw,
    const float* __restrict__ pr_old, float* __restrict__ pr_new,
    float alpha, const float* __restrict__ d_dang, float br, float invN,
    float* __restrict__ d_diff, int seg_start, int num_verts
) {
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage ts;
    int tid = blockIdx.x * BLOCK + threadIdx.x;
    float my_diff = 0.0f;
    if (tid < num_verts) {
        int v = tid + seg_start;
        int start = off[v], end = off[v + 1];
        float sum = 0.0f;
        for (int j = start; j < end; j++)
            sum += nw[j] * pr_old[idx[j]];
        float dang_c = alpha * (*d_dang) * invN;
        float new_val = br + dang_c + alpha * sum;
        my_diff = fabsf(new_val - pr_old[v]);
        pr_new[v] = new_val;
    }
    float block_diff = BR(ts).Sum(my_diff);
    if (threadIdx.x == 0 && block_diff > 0.0f) atomicAdd(d_diff, block_diff);
}

struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    float* pr_scratch = nullptr;
    float* ow = nullptr;
    float* nw = nullptr;
    int32_t* di = nullptr;
    float* sc = nullptr;
    int32_t* nd = nullptr;
    float* spmv_out = nullptr;
    void* spmv_buf = nullptr;

    int64_t pr_scratch_cap = 0;
    int64_t ow_cap = 0;
    int64_t nw_cap = 0;
    int64_t di_cap = 0;
    bool sc_allocated = false;
    bool nd_allocated = false;
    int64_t spmv_out_cap = 0;
    int64_t spmv_buf_cap = 0;

    Cache() { cusparseCreate(&cusparse_handle); }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (pr_scratch) cudaFree(pr_scratch);
        if (ow) cudaFree(ow);
        if (nw) cudaFree(nw);
        if (di) cudaFree(di);
        if (sc) cudaFree(sc);
        if (nd) cudaFree(nd);
        if (spmv_out) cudaFree(spmv_out);
        if (spmv_buf) cudaFree(spmv_buf);
    }

    void ensure_pr_scratch(int64_t n) {
        if (pr_scratch_cap < n) {
            if (pr_scratch) cudaFree(pr_scratch);
            cudaMalloc(&pr_scratch, n * sizeof(float));
            pr_scratch_cap = n;
        }
    }

    void ensure_ow(int64_t n) {
        if (ow_cap < n) {
            if (ow) cudaFree(ow);
            cudaMalloc(&ow, n * sizeof(float));
            ow_cap = n;
        }
    }

    void ensure_nw(int64_t n) {
        if (nw_cap < n) {
            if (nw) cudaFree(nw);
            cudaMalloc(&nw, n * sizeof(float));
            nw_cap = n;
        }
    }

    void ensure_di(int64_t n) {
        if (di_cap < n) {
            if (di) cudaFree(di);
            cudaMalloc(&di, n * sizeof(int32_t));
            di_cap = n;
        }
    }

    void ensure_sc() {
        if (!sc_allocated) {
            cudaMalloc(&sc, 2 * sizeof(float));
            sc_allocated = true;
        }
    }

    void ensure_nd() {
        if (!nd_allocated) {
            cudaMalloc(&nd, sizeof(int32_t));
            nd_allocated = true;
        }
    }

    void ensure_spmv_out(int64_t n) {
        if (spmv_out_cap < n) {
            if (spmv_out) cudaFree(spmv_out);
            cudaMalloc(&spmv_out, n * sizeof(float));
            spmv_out_cap = n;
        }
    }

    void ensure_spmv_buf(int64_t n) {
        if (spmv_buf_cap < n) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, n);
            spmv_buf_cap = n;
        }
    }
};

}  

PageRankResult pagerank_seg(const graph32_t& graph,
                            const float* edge_weights,
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
    int32_t V = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;

    const auto& seg_opt = graph.segment_offsets.value();
    int32_t seg[5];
    for (int i = 0; i < 5; i++) seg[i] = seg_opt[i];

    
    cache.ensure_pr_scratch(V);
    cache.ensure_nw(E);
    cache.ensure_di(V);
    cache.ensure_sc();
    cache.ensure_nd();

    float* nw = cache.nw;
    int32_t* di = cache.di;
    float* d_dang = cache.sc;
    float* d_diff = cache.sc + 1;
    int32_t* d_nd = cache.nd;

    cache.ensure_ow(V);
    cudaMemset(cache.ow, 0, V * sizeof(float));
    if (E > 0) scatter_add_kernel<<<(E + BLOCK - 1) / BLOCK, BLOCK>>>(d_idx, edge_weights, cache.ow, E);
    const float* ow_ptr = cache.ow;

    
    if (initial_pageranks != nullptr) {
        cudaMemcpy(pageranks, initial_pageranks, V * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        if (V > 0) fill_f32_kernel<<<(V + BLOCK - 1) / BLOCK, BLOCK>>>(pageranks, 1.0f / V, V);
    }

    
    if (E > 0) norm_weights_kernel<<<(E + BLOCK - 1) / BLOCK, BLOCK>>>(d_idx, edge_weights, ow_ptr, nw, E);

    
    cudaMemset(d_nd, 0, sizeof(int32_t));
    if (V > 0) find_dangling_kernel<<<(V + BLOCK - 1) / BLOCK, BLOCK>>>(ow_ptr, di, d_nd, V);
    int h_nd;
    cudaMemcpy(&h_nd, d_nd, sizeof(int), cudaMemcpyDeviceToHost);

    float br = (1.0f - alpha) / V;
    float invN = 1.0f / V;
    int n_medium = seg[2] - seg[1];
    int n_small_zero = seg[4] - seg[2];

    
    bool use_cusparse = (E > 500000 && (float)E / V > 8.0f);

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    float* d_spmv = nullptr;

    if (use_cusparse) {
        cache.ensure_spmv_out(V);
        d_spmv = cache.spmv_out;

        cusparseCreateCsr(&matA, V, V, E, (void*)d_off, (void*)d_idx, (void*)nw,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cusparseCreateDnVec(&vecX, V, pageranks, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, V, d_spmv, CUDA_R_32F);

        float one = 1.0f, zero = 0.0f;
        size_t bsz;
        cusparseSpMV_bufferSize(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, matA, vecX, &zero, vecY, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, &bsz);
        int64_t buf_size = std::max((int64_t)bsz, (int64_t)16);
        cache.ensure_spmv_buf(buf_size);
        cusparseSpMV_preprocess(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, matA, vecX, &zero, vecY, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, cache.spmv_buf);
    }

    float* pr_curr = pageranks;
    float* pr_next = cache.pr_scratch;
    std::size_t iters = 0;
    bool conv = false;
    float h_diff;

    for (std::size_t i = 0; i < max_iterations; i++) {
        cudaMemsetAsync(d_dang, 0, 2 * sizeof(float), 0);
        if (h_nd > 0) dangling_sum_kernel<<<(h_nd + BLOCK - 1) / BLOCK, BLOCK>>>(pr_curr, di, h_nd, d_dang);

        if (use_cusparse) {
            float one = 1.0f, zero = 0.0f;
            cusparseDnVecSetValues(vecX, pr_curr);
            cusparseSpMV(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, matA, vecX, &zero, vecY, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG2, cache.spmv_buf);
            if (V > 0) update_diff_kernel<<<(V + BLOCK - 1) / BLOCK, BLOCK>>>(d_spmv, pr_curr, pr_next, alpha, d_dang, br, invN, d_diff, V);
        } else {
            if (seg[1] > seg[0])
                spmv_large_kernel<<<seg[1] - seg[0], BLOCK>>>(d_off, d_idx, nw, pr_curr, pr_next, alpha, d_dang, br, invN, d_diff, seg[0], seg[1]);
            if (n_medium > 0) {
                int g = (n_medium + BLOCK / 32 - 1) / (BLOCK / 32);
                spmv_medium_kernel<<<g, BLOCK>>>(d_off, d_idx, nw, pr_curr, pr_next, alpha, d_dang, br, invN, d_diff, seg[1], n_medium);
            }
            if (n_small_zero > 0)
                spmv_small_kernel<<<(n_small_zero + BLOCK - 1) / BLOCK, BLOCK>>>(d_off, d_idx, nw, pr_curr, pr_next, alpha, d_dang, br, invN, d_diff, seg[2], n_small_zero);
        }

        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        iters = i + 1;
        if (h_diff < epsilon) { conv = true; break; }
        float* tmp = pr_curr; pr_curr = pr_next; pr_next = tmp;
    }

    if (use_cusparse) {
        cusparseDestroyDnVec(vecX);
        cusparseDestroyDnVec(vecY);
        cusparseDestroySpMat(matA);
    }

    
    if (pr_curr != pageranks) {
        cudaMemcpy(pageranks, pr_curr, V * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return PageRankResult{iters, conv};
}

}  
