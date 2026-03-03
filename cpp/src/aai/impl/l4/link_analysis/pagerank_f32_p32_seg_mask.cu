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
#include <cstddef>

namespace aai {

namespace {

constexpr int BLOCK_SIZE = 128;



struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;

    
    int32_t* new_offsets = nullptr;   
    float* out_weights = nullptr;     
    uint8_t* dangling_mask = nullptr; 
    float* pers_full = nullptr;       
    float* pr_buf = nullptr;          
    int64_t nv_cap = 0;

    
    int32_t* new_indices = nullptr;   
    float* new_weights = nullptr;     
    int* flags = nullptr;             
    int* prefix = nullptr;            
    int64_t ne_cap = 0;

    
    float* ds0 = nullptr;
    float* ds1 = nullptr;
    float* diff_sum = nullptr;
    float* pers_sum_dev = nullptr;
    bool small_allocated = false;

    
    void* scan_temp = nullptr;
    size_t scan_temp_cap = 0;

    void* reduce_pers_temp = nullptr;
    size_t reduce_pers_cap = 0;

    void* spmv_buf = nullptr;
    size_t spmv_buf_cap = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        cudaFree(new_offsets);
        cudaFree(out_weights);
        cudaFree(dangling_mask);
        cudaFree(pers_full);
        cudaFree(pr_buf);
        cudaFree(new_indices);
        cudaFree(new_weights);
        cudaFree(flags);
        cudaFree(prefix);
        cudaFree(ds0);
        cudaFree(ds1);
        cudaFree(diff_sum);
        cudaFree(pers_sum_dev);
        cudaFree(scan_temp);
        cudaFree(reduce_pers_temp);
        cudaFree(spmv_buf);
    }

    void ensure_nv(int32_t nv) {
        int64_t needed = (int64_t)nv + 1;
        if (nv_cap < needed) {
            cudaFree(new_offsets);
            cudaFree(out_weights);
            cudaFree(dangling_mask);
            cudaFree(pers_full);
            cudaFree(pr_buf);
            cudaMalloc(&new_offsets, needed * sizeof(int32_t));
            cudaMalloc(&out_weights, (nv > 0 ? nv : 1) * sizeof(float));
            cudaMalloc(&dangling_mask, (nv > 0 ? nv : 1) * sizeof(uint8_t));
            cudaMalloc(&pers_full, (nv > 0 ? nv : 1) * sizeof(float));
            cudaMalloc(&pr_buf, (nv > 0 ? nv : 1) * sizeof(float));
            nv_cap = needed;
        }
    }

    void ensure_ne(int32_t ne) {
        int64_t needed = (int64_t)ne + 1;
        if (ne_cap < needed) {
            cudaFree(new_indices);
            cudaFree(new_weights);
            cudaFree(flags);
            cudaFree(prefix);
            cudaMalloc(&new_indices, (ne > 0 ? ne : 1) * sizeof(int32_t));
            cudaMalloc(&new_weights, (ne > 0 ? ne : 1) * sizeof(float));
            cudaMalloc(&flags, needed * sizeof(int));
            cudaMalloc(&prefix, needed * sizeof(int));
            ne_cap = needed;
        }
    }

    void ensure_small() {
        if (!small_allocated) {
            cudaMalloc(&ds0, sizeof(float));
            cudaMalloc(&ds1, sizeof(float));
            cudaMalloc(&diff_sum, sizeof(float));
            cudaMalloc(&pers_sum_dev, sizeof(float));
            small_allocated = true;
        }
    }

    void ensure_scan_temp(size_t bytes) {
        if (scan_temp_cap < bytes) {
            cudaFree(scan_temp);
            cudaMalloc(&scan_temp, bytes > 0 ? bytes : 1);
            scan_temp_cap = bytes;
        }
    }

    void ensure_reduce_pers_temp(size_t bytes) {
        if (reduce_pers_cap < bytes) {
            cudaFree(reduce_pers_temp);
            cudaMalloc(&reduce_pers_temp, bytes > 0 ? bytes : 1);
            reduce_pers_cap = bytes;
        }
    }

    void ensure_spmv_buf(size_t bytes) {
        if (spmv_buf_cap < bytes) {
            cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, bytes > 0 ? bytes : 1);
            spmv_buf_cap = bytes;
        }
    }
};



__global__ void build_flags_kernel(const uint32_t* __restrict__ mask, int* __restrict__ flags, int ne) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ne) flags[idx] = (mask[idx >> 5] >> (idx & 31)) & 1;
}

__global__ void compact_edges_kernel(const int32_t* __restrict__ old_indices, const float* __restrict__ old_weights,
    const int* __restrict__ flags, const int* __restrict__ prefix,
    int32_t* __restrict__ new_indices, float* __restrict__ new_weights, int ne) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ne && flags[idx]) {
        int p = prefix[idx];
        new_indices[p] = old_indices[idx];
        new_weights[p] = old_weights[idx];
    }
}

__global__ void build_new_offsets_kernel(const int32_t* __restrict__ old_offsets, const int* __restrict__ prefix,
    int32_t* __restrict__ new_offsets, int nv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= nv) new_offsets[idx] = prefix[old_offsets[idx]];
}

__global__ void compute_out_weights_kernel(const int32_t* __restrict__ indices, const float* __restrict__ weights,
    float* __restrict__ out_weights, int ne) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ne) atomicAdd(&out_weights[indices[idx]], weights[idx]);
}

__global__ void normalize_weights_kernel(float* __restrict__ weights, const int32_t* __restrict__ indices,
    const float* __restrict__ out_weights, int ne) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ne) {
        float ow = out_weights[indices[idx]];
        weights[idx] = (ow > 0.0f) ? weights[idx] / ow : 0.0f;
    }
}

__global__ void build_dangling_mask_kernel(const float* __restrict__ out_weights,
    uint8_t* __restrict__ dangling_mask, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dangling_mask[idx] = (out_weights[idx] == 0.0f) ? 1 : 0;
}

__global__ void scatter_pers_kernel(const int32_t* __restrict__ pv, const float* __restrict__ pp,
    float* __restrict__ pers, int ps, float inv) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ps) pers[pv[idx]] = pp[idx] * inv;
}

__global__ void init_uniform_kernel(float* __restrict__ pr, int n, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) pr[idx] = val;
}

__global__ void dangling_dot_kernel(const float* __restrict__ pr, const uint8_t* __restrict__ dmask,
    float* __restrict__ d_sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n && dmask[idx]) ? pr[idx] : 0.0f;
    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage ts;
    float s = BR(ts).Sum(val);
    if (threadIdx.x == 0 && s != 0.0f) atomicAdd(d_sum, s);
}

__global__ void apply_pers_fused_kernel(
    float* __restrict__ pr_new,
    const float* __restrict__ pr_old,
    const float* __restrict__ pers_full,
    const uint8_t* __restrict__ dmask,
    float alpha,
    const float* __restrict__ d_dangling_sum,
    float one_minus_alpha,
    float* __restrict__ d_dangling_next,
    float* __restrict__ d_diff_sum,
    int n)
{
    float base = alpha * d_dangling_sum[0] + one_minus_alpha;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f, dcont = 0.0f, dval = 0.0f;
    if (idx < n) {
        val = pr_new[idx] + base * pers_full[idx];
        pr_new[idx] = val;
        if (dmask[idx]) dcont = val;
        if (d_diff_sum) dval = fabsf(val - pr_old[idx]);
    }
    typedef cub::BlockReduce<float, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage ts;
    float bs = BR(ts).Sum(dcont);
    if (threadIdx.x == 0 && bs != 0.0f) atomicAdd(d_dangling_next, bs);
    if (d_diff_sum) {
        __syncthreads();
        float bd = BR(ts).Sum(dval);
        if (threadIdx.x == 0 && bd != 0.0f) atomicAdd(d_diff_sum, bd);
    }
}

}  

PageRankResult personalized_pagerank_seg_mask(const graph32_t& graph,
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
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    const uint32_t* d_mask = graph.edge_mask;

    cudaStream_t stream = 0;

    cache.ensure_nv(nv);
    cache.ensure_ne(ne);
    cache.ensure_small();

    
    int32_t new_ne;
    float h_pers_sum;

    int32_t* d_new_offsets = cache.new_offsets;
    int32_t* d_new_indices = cache.new_indices;
    float* d_new_weights = cache.new_weights;

    {
        int* d_flags = cache.flags;
        int* d_prefix = cache.prefix;

        cudaMemsetAsync(d_flags + ne, 0, sizeof(int), stream);
        if (ne > 0) {
            build_flags_kernel<<<(ne + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                d_mask, d_flags, ne);
        }

        size_t scan_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, (int*)nullptr, (int*)nullptr, ne + 1);
        cache.ensure_scan_temp(scan_bytes);
        cub::DeviceScan::ExclusiveSum(cache.scan_temp, scan_bytes, d_flags, d_prefix, ne + 1, stream);

        build_new_offsets_kernel<<<(nv + 1 + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            d_offsets, d_prefix, d_new_offsets, nv);
        if (ne > 0) {
            compact_edges_kernel<<<(ne + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                d_indices, edge_weights, d_flags, d_prefix,
                d_new_indices, d_new_weights, ne);
        }

        
        size_t reduce_pers_bytes = 0;
        cub::DeviceReduce::Sum(nullptr, reduce_pers_bytes, (float*)nullptr, (float*)nullptr, (int)personalization_size);
        cache.ensure_reduce_pers_temp(reduce_pers_bytes);
        cub::DeviceReduce::Sum(cache.reduce_pers_temp, reduce_pers_bytes,
                               personalization_values, cache.pers_sum_dev, (int)personalization_size, stream);

        
        cudaMemcpyAsync(&new_ne, d_prefix + ne, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&h_pers_sum, cache.pers_sum_dev, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    float pers_sum_inv = (h_pers_sum > 0.0f) ? 1.0f / h_pers_sum : 0.0f;

    
    float* d_out_weights = cache.out_weights;
    cudaMemsetAsync(d_out_weights, 0, nv * sizeof(float), stream);
    if (new_ne > 0) {
        compute_out_weights_kernel<<<(new_ne + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            d_new_indices, d_new_weights, d_out_weights, new_ne);
    }

    if (new_ne > 0) {
        normalize_weights_kernel<<<(new_ne + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            d_new_weights, d_new_indices, d_out_weights, new_ne);
    }

    
    uint8_t* d_dangling_mask = cache.dangling_mask;
    if (nv > 0) {
        build_dangling_mask_kernel<<<(nv + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            d_out_weights, d_dangling_mask, nv);
    }

    
    float* d_pers_full = cache.pers_full;
    cudaMemsetAsync(d_pers_full, 0, nv * sizeof(float), stream);
    if (personalization_size > 0) {
        int ps = (int)personalization_size;
        scatter_pers_kernel<<<(ps + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            personalization_vertices, personalization_values, d_pers_full, ps, pers_sum_inv);
    }

    
    float* d_pr0 = pageranks;
    float* d_pr1 = cache.pr_buf;

    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(d_pr0, initial_pageranks, nv * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        if (nv > 0) {
            init_uniform_kernel<<<(nv + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                d_pr0, nv, 1.0f / nv);
        }
    }

    
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, nv, nv, new_ne,
                      d_new_offsets, d_new_indices, d_new_weights,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, nv, d_pr0, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, nv, d_pr1, CUDA_R_32F);

    float spmv_alpha = alpha;
    float spmv_beta = 0.0f;

    size_t spmv_buffer_size = 0;
    cusparseSpMV_bufferSize(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &spmv_alpha, matA, vecX, &spmv_beta, vecY,
                            CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, &spmv_buffer_size);
    cache.ensure_spmv_buf(spmv_buffer_size);

    cusparseSpMV_preprocess(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &spmv_alpha, matA, vecX, &spmv_beta, vecY,
                             CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, cache.spmv_buf);

    
    float* d_ds[2] = {cache.ds0, cache.ds1};
    float* d_diff_sum = cache.diff_sum;

    
    cudaMemsetAsync(d_ds[0], 0, sizeof(float), stream);
    if (nv > 0) {
        dangling_dot_kernel<<<(nv + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
            d_pr0, d_dangling_mask, d_ds[0], nv);
    }

    float one_minus_alpha = 1.0f - alpha;
    float* d_pr_old = d_pr0;
    float* d_pr_new = d_pr1;
    int ds_cur = 0;
    int ds_nxt = 1;

    constexpr int CHECK_INTERVAL = 2;
    std::size_t iterations = 0;
    bool converged = false;

    
    for (std::size_t iter = 0; iter < max_iterations; ++iter) {
        
        cusparseDnVecSetValues(vecX, d_pr_old);
        cusparseDnVecSetValues(vecY, d_pr_new);
        cusparseSpMV(cache.cusparse_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &spmv_alpha, matA, vecX, &spmv_beta, vecY,
                     CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, cache.spmv_buf);

        bool do_check = ((iter + 1) % CHECK_INTERVAL == 0) || (iter + 1 >= max_iterations);

        
        cudaMemsetAsync(d_ds[ds_nxt], 0, sizeof(float), stream);
        if (do_check) {
            cudaMemsetAsync(d_diff_sum, 0, sizeof(float), stream);
        }

        
        if (nv > 0) {
            apply_pers_fused_kernel<<<(nv + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                d_pr_new, d_pr_old, d_pers_full, d_dangling_mask,
                alpha, d_ds[ds_cur], one_minus_alpha,
                d_ds[ds_nxt],
                do_check ? d_diff_sum : nullptr,
                nv);
        }

        iterations = iter + 1;

        if (do_check) {
            float h_diff;
            cudaMemcpy(&h_diff, d_diff_sum, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_diff < epsilon) {
                converged = true;
                break;
            }
        }

        
        float* tmp = d_pr_old; d_pr_old = d_pr_new; d_pr_new = tmp;
        int tmp_ds = ds_cur; ds_cur = ds_nxt; ds_nxt = tmp_ds;
    }

    
    float* d_result = converged ? d_pr_new : d_pr_old;

    
    if (d_result != pageranks) {
        cudaMemcpyAsync(pageranks, d_result, nv * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    return PageRankResult{iterations, converged};
}

}  
