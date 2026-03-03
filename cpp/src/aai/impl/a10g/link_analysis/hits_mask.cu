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
#include <algorithm>
#include <limits>
#include <math_constants.h>

namespace aai {

namespace {

struct MaxOp {
    __device__ __forceinline__ float operator()(float a, float b) const {
        return fmaxf(a, b);
    }
};





__global__ void compute_edge_flags(const uint32_t* __restrict__ edge_mask,
                                    int* __restrict__ flags, int num_edges) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < num_edges) {
        flags[e] = (edge_mask[e >> 5] >> (e & 31)) & 1;
    }
}

__global__ void compact_indices(const int* __restrict__ old_indices,
                                const int* __restrict__ positions,
                                const int* __restrict__ flags,
                                int* __restrict__ new_indices, int num_edges) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < num_edges && flags[e]) {
        new_indices[positions[e]] = old_indices[e];
    }
}

__global__ void build_compact_offsets(const int* __restrict__ old_offsets,
                                       const int* __restrict__ positions,
                                       int* __restrict__ new_offsets,
                                       int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v <= num_vertices) {
        new_offsets[v] = positions[old_offsets[v]];
    }
}





__global__ void expand_offsets_to_dst(const int* __restrict__ offsets,
                                       int* __restrict__ dst,
                                       int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];
        for (int e = start; e < end; e++) {
            dst[e] = v;
        }
    }
}

__global__ void build_offsets_from_sorted(const int* __restrict__ sorted_keys,
                                           int* __restrict__ offsets,
                                           int num_vertices, int num_edges) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v <= num_vertices) {
        int lo = 0, hi = num_edges;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (sorted_keys[mid] < v) lo = mid + 1;
            else hi = mid;
        }
        offsets[v] = lo;
    }
}




__global__ void fill_ones(float* __restrict__ arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = 1.0f;
}





__global__ void init_uniform(float* __restrict__ hubs, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) hubs[i] = 1.0f / (float)n;
}





__global__ void l1_sum_reduce(const float* __restrict__ arr,
                               float* __restrict__ partial_sums,
                               unsigned int* __restrict__ retire_count,
                               float* __restrict__ result,
                               int n) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        sum += fabsf(arr[i]);
    }

    float block_sum = BlockReduce(temp).Sum(sum);

    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = block_sum;
        __threadfence();
        unsigned int ticket = atomicInc(retire_count, gridDim.x - 1);
        if (ticket == gridDim.x - 1) {
            float total = 0.0f;
            for (int i = 0; i < (int)gridDim.x; i++) {
                total += partial_sums[i];
            }
            *result = total;
        }
    }
}

__global__ void divide_by_value(float* __restrict__ arr, const float* __restrict__ divisor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float d = *divisor;
        if (d > 0.0f) arr[i] /= d;
    }
}





__global__ void compute_max_phase(const float* __restrict__ new_hubs,
                                    const float* __restrict__ auth,
                                    float* __restrict__ partial_hub_max,
                                    float* __restrict__ partial_auth_max,
                                    unsigned int* __restrict__ retire_count,
                                    float* __restrict__ hub_max_out,
                                    float* __restrict__ auth_max_out,
                                    int n) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp1;
    __shared__ typename BlockReduce::TempStorage temp2;

    float h_max = 0.0f, a_max = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        h_max = fmaxf(h_max, fabsf(new_hubs[i]));
        a_max = fmaxf(a_max, fabsf(auth[i]));
    }

    float bh = BlockReduce(temp1).Reduce(h_max, MaxOp());
    float ba = BlockReduce(temp2).Reduce(a_max, MaxOp());

    if (threadIdx.x == 0) {
        partial_hub_max[blockIdx.x] = bh;
        partial_auth_max[blockIdx.x] = ba;
        __threadfence();
        unsigned int ticket = atomicInc(retire_count, gridDim.x - 1);
        if (ticket == gridDim.x - 1) {
            float gh = 0.0f, ga = 0.0f;
            for (int i = 0; i < (int)gridDim.x; i++) {
                gh = fmaxf(gh, partial_hub_max[i]);
                ga = fmaxf(ga, partial_auth_max[i]);
            }
            *hub_max_out = gh;
            *auth_max_out = ga;
        }
    }
}

__global__ void normalize_diff_phase(float* __restrict__ new_hubs,
                                      float* __restrict__ auth,
                                      const float* __restrict__ old_hubs,
                                      const float* __restrict__ hub_max_in,
                                      const float* __restrict__ auth_max_in,
                                      float* __restrict__ partial_diff,
                                      unsigned int* __restrict__ retire_count,
                                      float* __restrict__ diff_out,
                                      int n) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float hm = *hub_max_in;
    float am = *auth_max_in;

    float local_diff = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float h = new_hubs[i];
        float a = auth[i];

        if (hm > 0.0f) h /= hm;
        if (am > 0.0f) a /= am;

        new_hubs[i] = h;
        auth[i] = a;

        local_diff += fabsf(h - old_hubs[i]);
    }

    float block_diff = BlockReduce(temp).Sum(local_diff);

    if (threadIdx.x == 0) {
        partial_diff[blockIdx.x] = block_diff;
        __threadfence();
        unsigned int ticket = atomicInc(retire_count, gridDim.x - 1);
        if (ticket == gridDim.x - 1) {
            float total = 0.0f;
            for (int i = 0; i < (int)gridDim.x; i++) {
                total += partial_diff[i];
            }
            *diff_out = total;
        }
    }
}




__global__ void l1_normalize_two(float* __restrict__ hubs,
                                  float* __restrict__ auth,
                                  float* __restrict__ partial_h,
                                  float* __restrict__ partial_a,
                                  unsigned int* __restrict__ retire_count,
                                  float* __restrict__ h_sum_out,
                                  float* __restrict__ a_sum_out,
                                  int n) {
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp1;
    __shared__ typename BlockReduce::TempStorage temp2;

    float hs = 0.0f, as = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        hs += fabsf(hubs[i]);
        as += fabsf(auth[i]);
    }

    float bh = BlockReduce(temp1).Sum(hs);
    float ba = BlockReduce(temp2).Sum(as);

    if (threadIdx.x == 0) {
        partial_h[blockIdx.x] = bh;
        partial_a[blockIdx.x] = ba;
        __threadfence();
        unsigned int ticket = atomicInc(retire_count, gridDim.x - 1);
        if (ticket == gridDim.x - 1) {
            float th = 0.0f, ta = 0.0f;
            for (int i = 0; i < (int)gridDim.x; i++) {
                th += partial_h[i];
                ta += partial_a[i];
            }
            *h_sum_out = th;
            *a_sum_out = ta;
        }
    }
}

__global__ void divide_two_by_values(float* __restrict__ hubs,
                                      float* __restrict__ auth,
                                      const float* __restrict__ h_div,
                                      const float* __restrict__ a_div,
                                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float hd = *h_div;
        float ad = *a_div;
        if (hd > 0.0f) hubs[i] /= hd;
        if (ad > 0.0f) auth[i] /= ad;
    }
}





struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;
    float* h_diff_pinned = nullptr;

    
    int* d_flags = nullptr;
    int64_t flags_capacity = 0;

    int* d_positions = nullptr;
    int64_t positions_capacity = 0;

    int* d_csc_offsets = nullptr;
    int64_t csc_offsets_capacity = 0;

    int* d_csc_indices = nullptr;
    int64_t csc_indices_capacity = 0;

    int* d_csr_offsets = nullptr;
    int64_t csr_offsets_capacity = 0;

    int* d_dst = nullptr;
    int64_t dst_capacity = 0;

    int* d_sorted_src = nullptr;
    int64_t sorted_src_capacity = 0;

    int* d_sorted_dst = nullptr;
    int64_t sorted_dst_capacity = 0;

    float* d_values = nullptr;
    int64_t values_capacity = 0;

    float* d_hubs0 = nullptr;
    int64_t hubs0_capacity = 0;

    float* d_hubs1 = nullptr;
    int64_t hubs1_capacity = 0;

    float* d_auth = nullptr;
    int64_t auth_capacity = 0;

    float* d_alpha = nullptr;
    float* d_beta = nullptr;

    float* d_partials = nullptr;
    int64_t partials_capacity = 0;

    float* d_partials2 = nullptr;
    int64_t partials2_capacity = 0;

    unsigned int* d_retire1 = nullptr;
    unsigned int* d_retire2 = nullptr;

    float* d_hub_max = nullptr;
    float* d_auth_max = nullptr;
    float* d_diff = nullptr;
    float* d_scalar = nullptr;

    void* d_scan_temp = nullptr;
    int64_t scan_temp_capacity = 0;

    void* d_sort_temp = nullptr;
    int64_t sort_temp_capacity = 0;

    void* d_spmv_buf_AT = nullptr;
    int64_t spmv_buf_AT_capacity = 0;

    void* d_spmv_buf_A = nullptr;
    int64_t spmv_buf_A_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cudaHostAlloc(&h_diff_pinned, sizeof(float), cudaHostAllocDefault);

        
        cudaMalloc(&d_alpha, sizeof(float));
        cudaMalloc(&d_beta, sizeof(float));
        cudaMalloc(&d_retire1, sizeof(unsigned int));
        cudaMalloc(&d_retire2, sizeof(unsigned int));
        cudaMalloc(&d_hub_max, sizeof(float));
        cudaMalloc(&d_auth_max, sizeof(float));
        cudaMalloc(&d_diff, sizeof(float));
        cudaMalloc(&d_scalar, sizeof(float));

        float h_one = 1.0f, h_zero = 0.0f;
        cudaMemcpy(d_alpha, &h_one, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta, &h_zero, sizeof(float), cudaMemcpyHostToDevice);
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (h_diff_pinned) cudaFreeHost(h_diff_pinned);

        if (d_flags) cudaFree(d_flags);
        if (d_positions) cudaFree(d_positions);
        if (d_csc_offsets) cudaFree(d_csc_offsets);
        if (d_csc_indices) cudaFree(d_csc_indices);
        if (d_csr_offsets) cudaFree(d_csr_offsets);
        if (d_dst) cudaFree(d_dst);
        if (d_sorted_src) cudaFree(d_sorted_src);
        if (d_sorted_dst) cudaFree(d_sorted_dst);
        if (d_values) cudaFree(d_values);
        if (d_hubs0) cudaFree(d_hubs0);
        if (d_hubs1) cudaFree(d_hubs1);
        if (d_auth) cudaFree(d_auth);
        if (d_alpha) cudaFree(d_alpha);
        if (d_beta) cudaFree(d_beta);
        if (d_partials) cudaFree(d_partials);
        if (d_partials2) cudaFree(d_partials2);
        if (d_retire1) cudaFree(d_retire1);
        if (d_retire2) cudaFree(d_retire2);
        if (d_hub_max) cudaFree(d_hub_max);
        if (d_auth_max) cudaFree(d_auth_max);
        if (d_diff) cudaFree(d_diff);
        if (d_scalar) cudaFree(d_scalar);
        if (d_scan_temp) cudaFree(d_scan_temp);
        if (d_sort_temp) cudaFree(d_sort_temp);
        if (d_spmv_buf_AT) cudaFree(d_spmv_buf_AT);
        if (d_spmv_buf_A) cudaFree(d_spmv_buf_A);
    }

    void ensure_flags(int64_t n) {
        if (flags_capacity < n) {
            if (d_flags) cudaFree(d_flags);
            cudaMalloc(&d_flags, n * sizeof(int));
            flags_capacity = n;
        }
    }
    void ensure_positions(int64_t n) {
        if (positions_capacity < n) {
            if (d_positions) cudaFree(d_positions);
            cudaMalloc(&d_positions, n * sizeof(int));
            positions_capacity = n;
        }
    }
    void ensure_csc_offsets(int64_t n) {
        if (csc_offsets_capacity < n) {
            if (d_csc_offsets) cudaFree(d_csc_offsets);
            cudaMalloc(&d_csc_offsets, n * sizeof(int));
            csc_offsets_capacity = n;
        }
    }
    void ensure_csc_indices(int64_t n) {
        if (csc_indices_capacity < n) {
            if (d_csc_indices) cudaFree(d_csc_indices);
            cudaMalloc(&d_csc_indices, n * sizeof(int));
            csc_indices_capacity = n;
        }
    }
    void ensure_csr_offsets(int64_t n) {
        if (csr_offsets_capacity < n) {
            if (d_csr_offsets) cudaFree(d_csr_offsets);
            cudaMalloc(&d_csr_offsets, n * sizeof(int));
            csr_offsets_capacity = n;
        }
    }
    void ensure_dst(int64_t n) {
        if (dst_capacity < n) {
            if (d_dst) cudaFree(d_dst);
            cudaMalloc(&d_dst, n * sizeof(int));
            dst_capacity = n;
        }
    }
    void ensure_sorted_src(int64_t n) {
        if (sorted_src_capacity < n) {
            if (d_sorted_src) cudaFree(d_sorted_src);
            cudaMalloc(&d_sorted_src, n * sizeof(int));
            sorted_src_capacity = n;
        }
    }
    void ensure_sorted_dst(int64_t n) {
        if (sorted_dst_capacity < n) {
            if (d_sorted_dst) cudaFree(d_sorted_dst);
            cudaMalloc(&d_sorted_dst, n * sizeof(int));
            sorted_dst_capacity = n;
        }
    }
    void ensure_values(int64_t n) {
        if (values_capacity < n) {
            if (d_values) cudaFree(d_values);
            cudaMalloc(&d_values, n * sizeof(float));
            values_capacity = n;
        }
    }
    void ensure_hubs0(int64_t n) {
        if (hubs0_capacity < n) {
            if (d_hubs0) cudaFree(d_hubs0);
            cudaMalloc(&d_hubs0, n * sizeof(float));
            hubs0_capacity = n;
        }
    }
    void ensure_hubs1(int64_t n) {
        if (hubs1_capacity < n) {
            if (d_hubs1) cudaFree(d_hubs1);
            cudaMalloc(&d_hubs1, n * sizeof(float));
            hubs1_capacity = n;
        }
    }
    void ensure_auth(int64_t n) {
        if (auth_capacity < n) {
            if (d_auth) cudaFree(d_auth);
            cudaMalloc(&d_auth, n * sizeof(float));
            auth_capacity = n;
        }
    }
    void ensure_partials(int64_t n) {
        if (partials_capacity < n) {
            if (d_partials) cudaFree(d_partials);
            cudaMalloc(&d_partials, n * sizeof(float));
            partials_capacity = n;
        }
    }
    void ensure_partials2(int64_t n) {
        if (partials2_capacity < n) {
            if (d_partials2) cudaFree(d_partials2);
            cudaMalloc(&d_partials2, n * sizeof(float));
            partials2_capacity = n;
        }
    }
    void ensure_scan_temp(int64_t n) {
        if (scan_temp_capacity < n) {
            if (d_scan_temp) cudaFree(d_scan_temp);
            cudaMalloc(&d_scan_temp, n);
            scan_temp_capacity = n;
        }
    }
    void ensure_sort_temp(int64_t n) {
        if (sort_temp_capacity < n) {
            if (d_sort_temp) cudaFree(d_sort_temp);
            cudaMalloc(&d_sort_temp, n);
            sort_temp_capacity = n;
        }
    }
    void ensure_spmv_buf_AT(int64_t n) {
        if (spmv_buf_AT_capacity < n) {
            if (d_spmv_buf_AT) cudaFree(d_spmv_buf_AT);
            cudaMalloc(&d_spmv_buf_AT, n);
            spmv_buf_AT_capacity = n;
        }
    }
    void ensure_spmv_buf_A(int64_t n) {
        if (spmv_buf_A_capacity < n) {
            if (d_spmv_buf_A) cudaFree(d_spmv_buf_A);
            cudaMalloc(&d_spmv_buf_A, n);
            spmv_buf_A_capacity = n;
        }
    }
};

}  

HitsResult hits_mask(const graph32_t& graph,
                     float* hubs,
                     float* authorities,
                     float epsilon,
                     std::size_t max_iterations,
                     bool has_initial_hubs_guess,
                     bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;
    cudaStream_t stream = 0;

    if (num_vertices == 0) {
        return HitsResult{max_iterations, false, std::numeric_limits<float>::max()};
    }

    cusparseSetStream(cache.cusparse_handle, stream);

    
    
    
    cache.ensure_flags(num_edges);
    cache.ensure_positions((int64_t)num_edges + 1);

    compute_edge_flags<<<(num_edges + 255) / 256, 256, 0, stream>>>(
        d_edge_mask, cache.d_flags, num_edges);

    size_t scan_temp_bytes = 0;
    {
        int* d_null = nullptr;
        cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes, d_null, d_null, num_edges);
    }
    cache.ensure_scan_temp((int64_t)(scan_temp_bytes + 16));
    cub::DeviceScan::ExclusiveSum(cache.d_scan_temp, scan_temp_bytes,
                                   cache.d_flags, cache.d_positions, num_edges, stream);

    cudaStreamSynchronize(stream);
    int last_pos = 0, last_flag = 0;
    if (num_edges > 0) {
        cudaMemcpy(&last_pos, cache.d_positions + num_edges - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_flag, cache.d_flags + num_edges - 1, sizeof(int), cudaMemcpyDeviceToHost);
    }
    int num_active_edges = last_pos + last_flag;
    cudaMemcpy(cache.d_positions + num_edges, &num_active_edges, sizeof(int), cudaMemcpyHostToDevice);

    int64_t safe_nae = (num_active_edges > 0) ? num_active_edges : 1;
    cache.ensure_csc_offsets((int64_t)num_vertices + 1);
    cache.ensure_csc_indices(safe_nae);

    build_compact_offsets<<<(num_vertices + 1 + 255) / 256, 256, 0, stream>>>(
        d_offsets, cache.d_positions, cache.d_csc_offsets, num_vertices);
    if (num_active_edges > 0) {
        compact_indices<<<(num_edges + 255) / 256, 256, 0, stream>>>(
            d_indices, cache.d_positions, cache.d_flags, cache.d_csc_indices, num_edges);
    }

    
    
    
    cache.ensure_csr_offsets((int64_t)num_vertices + 1);
    int* d_csr_indices = nullptr;

    if (num_active_edges > 0) {
        cache.ensure_dst(num_active_edges);
        expand_offsets_to_dst<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
            cache.d_csc_offsets, cache.d_dst, num_vertices);

        cache.ensure_sorted_src(num_active_edges);
        cache.ensure_sorted_dst(num_active_edges);

        int end_bit = 1;
        { int v = num_vertices - 1; while (v > 0) { end_bit++; v >>= 1; } }
        if (end_bit > 32) end_bit = 32;

        size_t sort_temp_bytes = 0;
        {
            int* d_null = nullptr;
            cub::DeviceRadixSort::SortPairs(nullptr, sort_temp_bytes, d_null, d_null,
                                             d_null, d_null, num_active_edges, 0, end_bit);
        }
        cache.ensure_sort_temp((int64_t)(sort_temp_bytes + 16));
        cub::DeviceRadixSort::SortPairs(cache.d_sort_temp, sort_temp_bytes,
                                         cache.d_csc_indices, cache.d_sorted_src,
                                         cache.d_dst, cache.d_sorted_dst,
                                         num_active_edges, 0, end_bit, stream);

        build_offsets_from_sorted<<<(num_vertices + 1 + 255) / 256, 256, 0, stream>>>(
            cache.d_sorted_src, cache.d_csr_offsets, num_vertices, num_active_edges);

        d_csr_indices = cache.d_sorted_dst;
    } else {
        cudaMemsetAsync(cache.d_csr_offsets, 0, (num_vertices + 1) * sizeof(int), stream);
    }

    
    
    
    cache.ensure_values(safe_nae);
    if (num_active_edges > 0) {
        fill_ones<<<(num_active_edges + 255) / 256, 256, 0, stream>>>(
            cache.d_values, num_active_edges);
    }

    
    
    
    cache.ensure_hubs0(num_vertices);
    cache.ensure_hubs1(num_vertices);
    cache.ensure_auth(num_vertices);

    if (has_initial_hubs_guess) {
        cudaMemcpyAsync(cache.d_hubs0, hubs,
                         num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        init_uniform<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
            cache.d_hubs0, num_vertices);
    }

    
    
    
    cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);

    cusparseSpMatDescr_t mat_AT = nullptr, mat_A = nullptr;
    cusparseDnVecDescr_t vec_hubs = nullptr, vec_auth = nullptr, vec_new_hubs = nullptr;

    if (num_active_edges > 0) {
        cusparseCreateCsr(&mat_AT,
            num_vertices, num_vertices, num_active_edges,
            cache.d_csc_offsets, cache.d_csc_indices, cache.d_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        cusparseCreateCsr(&mat_A,
            num_vertices, num_vertices, num_active_edges,
            cache.d_csr_offsets, d_csr_indices, cache.d_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    }

    cusparseCreateDnVec(&vec_hubs, num_vertices, cache.d_hubs0, CUDA_R_32F);
    cusparseCreateDnVec(&vec_auth, num_vertices, cache.d_auth, CUDA_R_32F);
    cusparseCreateDnVec(&vec_new_hubs, num_vertices, cache.d_hubs1, CUDA_R_32F);

    if (num_active_edges > 0) {
        size_t buf_size_AT = 0, buf_size_A = 0;
        cusparseSpMV_bufferSize(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha, mat_AT, vec_hubs, cache.d_beta, vec_auth,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size_AT);

        cusparseSpMV_bufferSize(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha, mat_A, vec_auth, cache.d_beta, vec_new_hubs,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buf_size_A);

        if (buf_size_AT > 0) {
            cache.ensure_spmv_buf_AT((int64_t)(buf_size_AT + 16));
        }
        if (buf_size_A > 0) {
            cache.ensure_spmv_buf_A((int64_t)(buf_size_A + 16));
        }

        cusparseSpMV_preprocess(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha, mat_AT, vec_hubs, cache.d_beta, vec_auth,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.d_spmv_buf_AT);

        cusparseSpMV_preprocess(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            cache.d_alpha, mat_A, vec_auth, cache.d_beta, vec_new_hubs,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.d_spmv_buf_A);
    }

    
    
    
    int reduce_blocks = (num_vertices + 255) / 256;
    if (reduce_blocks > 512) reduce_blocks = 512;
    if (reduce_blocks < 1) reduce_blocks = 1;

    cache.ensure_partials(reduce_blocks);
    cache.ensure_partials2(reduce_blocks);

    cudaMemsetAsync(cache.d_retire1, 0, sizeof(unsigned int), stream);
    cudaMemsetAsync(cache.d_retire2, 0, sizeof(unsigned int), stream);

    
    
    
    if (has_initial_hubs_guess) {
        l1_sum_reduce<<<reduce_blocks, 256, 0, stream>>>(
            cache.d_hubs0, cache.d_partials, cache.d_retire1, cache.d_scalar,
            num_vertices);
        divide_by_value<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
            cache.d_hubs0, cache.d_scalar, num_vertices);
    }

    
    
    
    float* d_curr_hubs = cache.d_hubs0;
    float* d_new_hubs = cache.d_hubs1;
    float diff = 0.0f;
    std::size_t iterations = 0;
    bool converged = false;
    float threshold = epsilon * (float)num_vertices;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        cusparseDnVecSetValues(vec_hubs, d_curr_hubs);
        cusparseDnVecSetValues(vec_new_hubs, d_new_hubs);

        if (num_active_edges > 0) {
            cusparseSpMV(cache.cusparse_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                cache.d_alpha, mat_AT, vec_hubs, cache.d_beta, vec_auth,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.d_spmv_buf_AT);

            cusparseSpMV(cache.cusparse_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                cache.d_alpha, mat_A, vec_auth, cache.d_beta, vec_new_hubs,
                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.d_spmv_buf_A);
        } else {
            cudaMemsetAsync(cache.d_auth, 0, num_vertices * sizeof(float), stream);
            cudaMemsetAsync(d_new_hubs, 0, num_vertices * sizeof(float), stream);
        }

        compute_max_phase<<<reduce_blocks, 256, 0, stream>>>(
            d_new_hubs, cache.d_auth,
            cache.d_partials, cache.d_partials2,
            cache.d_retire1,
            cache.d_hub_max, cache.d_auth_max,
            num_vertices);

        normalize_diff_phase<<<reduce_blocks, 256, 0, stream>>>(
            d_new_hubs, cache.d_auth, d_curr_hubs,
            cache.d_hub_max, cache.d_auth_max,
            cache.d_partials, cache.d_retire2,
            cache.d_diff,
            num_vertices);

        iterations = iter + 1;

        cudaMemcpyAsync(cache.h_diff_pinned, cache.d_diff, sizeof(float),
                         cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        diff = *cache.h_diff_pinned;

        if (diff < threshold) {
            converged = true;
            std::swap(d_curr_hubs, d_new_hubs);
            break;
        }

        std::swap(d_curr_hubs, d_new_hubs);
    }

    
    
    
    if (normalize) {
        l1_normalize_two<<<reduce_blocks, 256, 0, stream>>>(
            d_curr_hubs, cache.d_auth,
            cache.d_partials, cache.d_partials2,
            cache.d_retire1,
            cache.d_hub_max, cache.d_auth_max,
            num_vertices);
        divide_two_by_values<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
            d_curr_hubs, cache.d_auth,
            cache.d_hub_max, cache.d_auth_max,
            num_vertices);
    }

    
    
    
    cudaMemcpyAsync(hubs, d_curr_hubs, num_vertices * sizeof(float),
                     cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(authorities, cache.d_auth, num_vertices * sizeof(float),
                     cudaMemcpyDeviceToDevice, stream);

    
    if (vec_hubs) cusparseDestroyDnVec(vec_hubs);
    if (vec_auth) cusparseDestroyDnVec(vec_auth);
    if (vec_new_hubs) cusparseDestroyDnVec(vec_new_hubs);
    if (mat_AT) cusparseDestroySpMat(mat_AT);
    if (mat_A) cusparseDestroySpMat(mat_A);

    cudaStreamSynchronize(stream);

    return HitsResult{iterations, converged, diff};
}

}  
