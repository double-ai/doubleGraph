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
#include <cstddef>
#include <climits>

namespace aai {

namespace {

constexpr int BLOCK = 256;
constexpr int MAX_GRID = 2048;

static inline int calc_grid(int n) {
    int g = (n + BLOCK - 1) / BLOCK;
    return g < MAX_GRID ? g : MAX_GRID;
}



__global__ void compute_out_weights_kernel(
    const int32_t* __restrict__ indices, const float* __restrict__ weights,
    float* out_weights, int32_t num_edges) {
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < num_edges; i += gridDim.x * BLOCK)
        atomicAdd(&out_weights[indices[i]], weights[i]);
}

__global__ void prescale_weights_kernel(
    const float* __restrict__ weights, const int32_t* __restrict__ indices,
    const float* __restrict__ out_weights, float* __restrict__ scaled,
    float alpha, int32_t num_edges) {
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < num_edges; i += gridDim.x * BLOCK) {
        float ow = out_weights[indices[i]];
        scaled[i] = (ow > 0.0f) ? alpha * weights[i] / ow : 0.0f;
    }
}

__global__ void reduce_sum_kernel(const float* __restrict__ data, int n, float* result) {
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage temp;
    float s = 0.0f;
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK)
        s += data[i];
    s = BR(temp).Sum(s);
    if (threadIdx.x == 0) atomicAdd(result, s);
}

__global__ void build_normalized_pers_kernel(
    const int32_t* __restrict__ verts, const float* __restrict__ vals,
    float* __restrict__ dense, const float* __restrict__ sum_ptr, int32_t size) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < size) {
        float s = *sum_ptr;
        dense[verts[i]] = (s > 0.0f) ? vals[i] / s : 0.0f;
    }
}

__global__ void normalize_compact_kernel(
    const float* __restrict__ vals, const float* __restrict__ sum_ptr,
    float* __restrict__ norm, int32_t size) {
    int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i < size) {
        float s = *sum_ptr;
        norm[i] = (s > 0.0f) ? vals[i] / s : 0.0f;
    }
}

__global__ void find_dangling_kernel(
    const float* __restrict__ out_weights, int32_t* dangling_list,
    int32_t* dangling_count, int32_t num_vertices) {
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < num_vertices; i += gridDim.x * BLOCK) {
        if (out_weights[i] == 0.0f) {
            int pos = atomicAdd(dangling_count, 1);
            dangling_list[pos] = i;
        }
    }
}

__global__ void init_uniform_kernel(float* data, float val, int32_t n) {
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < n; i += gridDim.x * BLOCK)
        data[i] = val;
}



__global__ void dangling_sum_kernel(
    const float* __restrict__ pageranks, const int32_t* __restrict__ dangling_list,
    int32_t dangling_count, float* result) {
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage temp;
    float s = 0.0f;
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < dangling_count; i += gridDim.x * BLOCK)
        s += pageranks[dangling_list[i]];
    s = BR(temp).Sum(s);
    if (threadIdx.x == 0) atomicAdd(result, s);
}

__global__ void update_diff_kernel(
    float* __restrict__ spmv_result, const float* __restrict__ cur_pr,
    const float* __restrict__ pers_dense, const float* __restrict__ dsum_ptr,
    float alpha, float oma, float* diff_sum, int32_t V) {
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage temp;
    float dsum = *dsum_ptr;
    float base = __fmaf_rn(alpha, dsum, oma);
    float td = 0.0f;
    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < V; i += gridDim.x * BLOCK) {
        float nv = __fmaf_rn(base, pers_dense[i], spmv_result[i]);
        spmv_result[i] = nv;
        td += fabsf(nv - cur_pr[i]);
    }
    float bd = BR(temp).Sum(td);
    if (threadIdx.x == 0) atomicAdd(diff_sum, bd);
}

__global__ void diff_only_kernel(
    const float* __restrict__ nxt, const float* __restrict__ cur,
    float* diff_sum, int32_t V) {
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage temp;
    float td = 0.0f;

    int V4 = V >> 2;
    const float4* nxt4 = reinterpret_cast<const float4*>(nxt);
    const float4* cur4 = reinterpret_cast<const float4*>(cur);

    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < V4; i += gridDim.x * BLOCK) {
        float4 n = nxt4[i];
        float4 c = cur4[i];
        td += fabsf(n.x - c.x) + fabsf(n.y - c.y) + fabsf(n.z - c.z) + fabsf(n.w - c.w);
    }

    int rem = V4 << 2;
    for (int i = rem + blockIdx.x * BLOCK + threadIdx.x; i < V; i += gridDim.x * BLOCK)
        td += fabsf(nxt[i] - cur[i]);

    float bd = BR(temp).Sum(td);
    if (threadIdx.x == 0) atomicAdd(diff_sum, bd);
}

__global__ void pers_scatter_adjust_kernel(
    float* __restrict__ nxt, const float* __restrict__ cur,
    const int32_t* __restrict__ pers_verts,
    const float* __restrict__ pers_norm,
    int32_t pers_size,
    const float* __restrict__ dsum_ptr,
    float alpha, float oma, float* diff_sum) {
    typedef cub::BlockReduce<float, BLOCK> BR;
    __shared__ typename BR::TempStorage temp;

    float base = __fmaf_rn(alpha, *dsum_ptr, oma);
    float adj = 0.0f;

    for (int i = blockIdx.x * BLOCK + threadIdx.x; i < pers_size; i += gridDim.x * BLOCK) {
        int v = pers_verts[i];
        float sv = nxt[v];
        float cv = cur[v];
        float old_d = fabsf(sv - cv);
        float nv = __fmaf_rn(base, pers_norm[i], sv);
        nxt[v] = nv;
        adj += fabsf(nv - cv) - old_d;
    }

    float total = BR(temp).Sum(adj);
    if (threadIdx.x == 0) atomicAdd(diff_sum, total);
}


struct Cache : Cacheable {
    cusparseHandle_t cusparse = nullptr;

    
    float* out_w = nullptr;
    float* sw = nullptr;
    float* ps = nullptr;
    float* pd = nullptr;
    float* pn = nullptr;
    int32_t* dl = nullptr;
    int32_t* dc = nullptr;
    float* pa = nullptr;
    float* pb = nullptr;
    float* dsum = nullptr;
    float* diff = nullptr;
    void* spmv_buf = nullptr;

    
    int64_t out_w_cap = 0;
    int64_t sw_cap = 0;
    int64_t pd_cap = 0;
    int64_t pn_cap = 0;
    int64_t dl_cap = 0;
    int64_t pa_cap = 0;
    int64_t pb_cap = 0;
    size_t spmv_buf_cap = 0;

    
    bool scalars_allocated = false;

    Cache() {
        cusparseCreate(&cusparse);
    }

    ~Cache() override {
        if (cusparse) cusparseDestroy(cusparse);
        if (out_w) cudaFree(out_w);
        if (sw) cudaFree(sw);
        if (ps) cudaFree(ps);
        if (pd) cudaFree(pd);
        if (pn) cudaFree(pn);
        if (dl) cudaFree(dl);
        if (dc) cudaFree(dc);
        if (pa) cudaFree(pa);
        if (pb) cudaFree(pb);
        if (dsum) cudaFree(dsum);
        if (diff) cudaFree(diff);
        if (spmv_buf) cudaFree(spmv_buf);
    }

    void ensure_scalars() {
        if (!scalars_allocated) {
            cudaMalloc(&ps, sizeof(float));
            cudaMalloc(&dc, sizeof(int32_t));
            cudaMalloc(&dsum, sizeof(float));
            cudaMalloc(&diff, sizeof(float));
            scalars_allocated = true;
        }
    }

    void ensure_V(int64_t V) {
        if (out_w_cap < V) {
            if (out_w) cudaFree(out_w);
            cudaMalloc(&out_w, V * sizeof(float));
            out_w_cap = V;
        }
        if (dl_cap < V) {
            if (dl) cudaFree(dl);
            cudaMalloc(&dl, V * sizeof(int32_t));
            dl_cap = V;
        }
        if (pa_cap < V) {
            if (pa) cudaFree(pa);
            cudaMalloc(&pa, V * sizeof(float));
            pa_cap = V;
        }
        if (pb_cap < V) {
            if (pb) cudaFree(pb);
            cudaMalloc(&pb, V * sizeof(float));
            pb_cap = V;
        }
    }

    void ensure_E(int64_t E) {
        if (sw_cap < E) {
            if (sw) cudaFree(sw);
            cudaMalloc(&sw, E * sizeof(float));
            sw_cap = E;
        }
    }

    void ensure_pd(int64_t V) {
        if (pd_cap < V) {
            if (pd) cudaFree(pd);
            cudaMalloc(&pd, V * sizeof(float));
            pd_cap = V;
        }
    }

    void ensure_pn(int64_t sz) {
        if (pn_cap < sz) {
            if (pn) cudaFree(pn);
            cudaMalloc(&pn, sz * sizeof(float));
            pn_cap = sz;
        }
    }

    void ensure_spmv_buf(size_t sz) {
        if (spmv_buf_cap < sz) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, sz);
            spmv_buf_cap = sz;
        }
    }
};

}  

PageRankResult personalized_pagerank_seg(const graph32_t& graph,
                                         const float* edge_weights,
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
    int32_t V = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;
    const float* d_weights = edge_weights;
    const int32_t* d_pv = personalization_vertices;
    const float* d_pp = personalization_values;
    int64_t pers_size = static_cast<int64_t>(personalization_size);
    float oma = 1.0f - alpha;
    cudaStream_t stream = 0;

    bool has_initial = (initial_pageranks != nullptr);
    bool use_split = (pers_size < (int64_t)V / 3);

    
    cache.ensure_scalars();
    cache.ensure_V(V);
    cache.ensure_E(E);

    float* d_out_w = cache.out_w;
    float* d_sw = cache.sw;
    float* d_ps = cache.ps;
    int32_t* d_dl = cache.dl;
    int32_t* d_dc = cache.dc;
    float* d_pa = cache.pa;
    float* d_pb = cache.pb;
    float* d_dsum = cache.dsum;
    float* d_diff = cache.diff;

    

    
    if (precomputed_vertex_out_weight_sums) {
        cudaMemcpyAsync(d_out_w, precomputed_vertex_out_weight_sums,
                       (size_t)V * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        cudaMemsetAsync(d_out_w, 0, (size_t)V * sizeof(float), stream);
        if (E > 0) {
            compute_out_weights_kernel<<<calc_grid(E), BLOCK, 0, stream>>>(d_indices, d_weights, d_out_w, E);
        }
    }

    
    if (E > 0) {
        prescale_weights_kernel<<<calc_grid(E), BLOCK, 0, stream>>>(d_weights, d_indices, d_out_w, d_sw, alpha, E);
    }

    
    cudaMemsetAsync(d_ps, 0, sizeof(float), stream);
    if (pers_size > 0) {
        reduce_sum_kernel<<<calc_grid((int)pers_size), BLOCK, 0, stream>>>(d_pp, (int)pers_size, d_ps);
    }

    
    float* d_pd = nullptr;
    float* d_pn = nullptr;

    if (use_split) {
        cache.ensure_pn(pers_size);
        d_pn = cache.pn;
        if (pers_size > 0) {
            normalize_compact_kernel<<<((int)pers_size + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(d_pp, d_ps, d_pn, (int32_t)pers_size);
        }
    } else {
        cache.ensure_pd(V);
        d_pd = cache.pd;
        cudaMemsetAsync(d_pd, 0, (size_t)V * sizeof(float), stream);
        if (pers_size > 0) {
            build_normalized_pers_kernel<<<((int)pers_size + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(d_pv, d_pp, d_pd, d_ps, (int32_t)pers_size);
        }
    }

    
    cudaMemsetAsync(d_dc, 0, sizeof(int32_t), stream);
    if (V > 0) {
        find_dangling_kernel<<<calc_grid(V), BLOCK, 0, stream>>>(d_out_w, d_dl, d_dc, V);
    }
    int32_t h_dc;
    cudaMemcpy(&h_dc, d_dc, sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    if (has_initial) {
        cudaMemcpyAsync(d_pa, initial_pageranks,
                       (size_t)V * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        if (V > 0) {
            init_uniform_kernel<<<calc_grid(V), BLOCK, 0, stream>>>(d_pa, 1.0f / V, V);
        }
    }

    
    cusparseSpMatDescr_t mat = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    float one = 1.0f, zero_f = 0.0f;

    if (E > 0) {
        cusparseCreateCsr(&mat, V, V, E,
            const_cast<int32_t*>(d_offsets), const_cast<int32_t*>(d_indices), d_sw,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        cusparseCreateDnVec(&vecX, V, d_pa, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, V, d_pb, CUDA_R_32F);

        size_t buf_size = 0;
        cusparseSpMV_bufferSize(cache.cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, mat, vecX, &zero_f, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size);

        if (buf_size > 0) {
            cache.ensure_spmv_buf(buf_size);
        }

        cusparseSpMV_preprocess(cache.cusparse,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &one, mat, vecX, &zero_f, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);
    }

    
    float* cur = d_pa;
    float* nxt = d_pb;
    int cur_idx = 0;
    size_t iterations = 0;
    bool converged = false;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        
        if (E > 0) {
            cusparseDnVecSetValues(vecX, cur);
            cusparseDnVecSetValues(vecY, nxt);
            cusparseSpMV(cache.cusparse,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &one, mat, vecX, &zero_f, vecY,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buf);
        } else {
            cudaMemsetAsync(nxt, 0, (size_t)V * sizeof(float), stream);
        }

        
        cudaMemsetAsync(d_dsum, 0, sizeof(float), stream);
        cudaMemsetAsync(d_diff, 0, sizeof(float), stream);

        
        if (h_dc > 0) {
            dangling_sum_kernel<<<calc_grid(h_dc), BLOCK, 0, stream>>>(cur, d_dl, h_dc, d_dsum);
        }

        if (use_split) {
            int V4 = V >> 2;
            int g = calc_grid(V4 > 0 ? V4 : V);
            if (V > 0) {
                diff_only_kernel<<<g, BLOCK, 0, stream>>>(nxt, cur, d_diff, V);
            }
            if (pers_size > 0) {
                int pg = ((int)pers_size + BLOCK - 1) / BLOCK;
                if (pg > 32) pg = 32;
                pers_scatter_adjust_kernel<<<pg, BLOCK, 0, stream>>>(nxt, cur, d_pv, d_pn, (int32_t)pers_size,
                                          d_dsum, alpha, oma, d_diff);
            }
        } else {
            if (V > 0) {
                update_diff_kernel<<<calc_grid(V), BLOCK, 0, stream>>>(nxt, cur, d_pd, d_dsum, alpha, oma, d_diff, V);
            }
        }

        
        float h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);

        iterations++;

        float* tmp = cur; cur = nxt; nxt = tmp;
        cur_idx = 1 - cur_idx;

        if (h_diff < epsilon) {
            converged = true;
            break;
        }
    }

    
    
    float* result_ptr = (cur_idx == 0) ? d_pa : d_pb;
    if (result_ptr != pageranks) {
        cudaMemcpyAsync(pageranks, result_ptr,
                       (size_t)V * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaStreamSynchronize(stream);
    }

    if (mat) cusparseDestroySpMat(mat);
    if (vecX) cusparseDestroyDnVec(vecX);
    if (vecY) cusparseDestroyDnVec(vecY);

    return PageRankResult{iterations, converged};
}

}  
