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




__device__ __forceinline__ double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}






__global__ void spmv_gather_high(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ x,
    double* __restrict__ y,
    int seg_start, int seg_end)
{
    typedef cub::BlockReduce<double, 256> BR;
    __shared__ typename BR::TempStorage temp;

    int v = seg_start + blockIdx.x;
    if (v >= seg_end) return;

    int start = offsets[v], end = offsets[v + 1];
    double sum = 0.0;
    for (int j = start + threadIdx.x; j < end; j += 256)
        sum += x[indices[j]];

    double block_sum = BR(temp).Sum(sum);
    if (threadIdx.x == 0) y[v] = block_sum;
}


__global__ void spmv_gather_mid(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ x,
    double* __restrict__ y,
    int seg_start, int seg_end)
{
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v = seg_start + blockIdx.x * (blockDim.x >> 5) + warp_in_block;

    if (v >= seg_end) return;

    int start = offsets[v], end = offsets[v + 1];
    double sum = 0.0;
    for (int j = start + lane; j < end; j += 32)
        sum += x[indices[j]];

    sum = warp_reduce_sum(sum);
    if (lane == 0) y[v] = sum;
}


__global__ void spmv_gather_low(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ x,
    double* __restrict__ y,
    int seg_start, int seg_end)
{
    int v = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= seg_end) return;

    int start = offsets[v], end = offsets[v + 1];
    double sum = 0.0;
    for (int j = start; j < end; j++)
        sum += x[indices[j]];
    y[v] = sum;
}


__global__ void spmv_set_zero(double* __restrict__ y, int seg_start, int seg_end) {
    int v = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (v < seg_end) y[v] = 0.0;
}


__global__ void spmv_gather_generic(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ x,
    double* __restrict__ y,
    int num_vertices)
{
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
    if (v >= num_vertices) return;

    int start = offsets[v], end = offsets[v + 1];
    double sum = 0.0;
    for (int j = start + lane; j < end; j += 32)
        sum += x[indices[j]];
    sum = warp_reduce_sum(sum);
    if (lane == 0) y[v] = sum;
}




__global__ void fill_value(double* arr, int N, double val) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        arr[i] = val;
}

__global__ void normalize_by_dev_max(double* vals, int N, const double* max_val) {
    const double inv = 1.0 / (*max_val);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        vals[i] *= inv;
}

__global__ void l1_normalize(double* vals, int N, const double* sum_val) {
    const double inv = 1.0 / (*sum_val);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x)
        vals[i] *= inv;
}




constexpr int NF_BLOCK = 256;

__global__ void normalize_and_diff(
    double* __restrict__ new_vals,
    const double* __restrict__ old_vals,
    int N,
    const double* __restrict__ max_val,
    double* __restrict__ partials,
    unsigned int* __restrict__ counter,
    double* __restrict__ diff_out)
{
    typedef cub::BlockReduce<double, NF_BLOCK> BR;
    __shared__ typename BR::TempStorage temp;

    const double inv_max = 1.0 / (*max_val);
    double my_diff = 0.0;

    for (int i = blockIdx.x * NF_BLOCK + threadIdx.x; i < N; i += gridDim.x * NF_BLOCK) {
        double normalized = new_vals[i] * inv_max;
        my_diff += fabs(normalized - old_vals[i]);
        new_vals[i] = normalized;
    }

    double block_sum = BR(temp).Sum(my_diff);
    if (threadIdx.x == 0) partials[blockIdx.x] = block_sum;
    __threadfence();

    __shared__ bool am_last;
    if (threadIdx.x == 0) {
        unsigned int ticket = atomicAdd(counter, 1);
        am_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (am_last) {
        double sum = 0.0;
        for (int i = threadIdx.x; i < gridDim.x; i += NF_BLOCK)
            sum += partials[i];
        sum = BR(temp).Sum(sum);
        if (threadIdx.x == 0) {
            *diff_out = sum;
            *counter = 0;
        }
    }
}




__global__ void count_degrees(const int32_t* __restrict__ csc_indices, int num_edges, int32_t* __restrict__ row_counts) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < num_edges; j += gridDim.x * blockDim.x)
        atomicAdd(&row_counts[csc_indices[j]], 1);
}

__global__ void fill_csr_warp(
    const int32_t* __restrict__ csc_offsets, const int32_t* __restrict__ csc_indices,
    int32_t* __restrict__ csr_pos, int32_t* __restrict__ csr_col_indices, int num_vertices)
{
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int v = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
    if (v >= num_vertices) return;

    int start = csc_offsets[v], end = csc_offsets[v + 1];
    for (int j = start + lane; j < end; j += 32) {
        int row = csc_indices[j];
        int pos = atomicAdd(&csr_pos[row], 1);
        csr_col_indices[pos] = v;
    }
}





void launch_spmv_segmented(
    const int32_t* offsets, const int32_t* indices,
    const double* x, double* y,
    int num_vertices, const int32_t* h_seg, cudaStream_t stream)
{
    int s0 = h_seg[0], s1 = h_seg[1], s2 = h_seg[2], s3 = h_seg[3], s4 = h_seg[4];

    if (s1 > s0) spmv_gather_high<<<s1-s0, 256, 0, stream>>>(offsets, indices, x, y, s0, s1);
    int n_mid = s2 - s1;
    if (n_mid > 0) {
        int wpb = 8;
        spmv_gather_mid<<<(n_mid+wpb-1)/wpb, 256, 0, stream>>>(offsets, indices, x, y, s1, s2);
    }
    int n_low = s3 - s2;
    if (n_low > 0) spmv_gather_low<<<(n_low+255)/256, 256, 0, stream>>>(offsets, indices, x, y, s2, s3);
    int n_zero = s4 - s3;
    if (n_zero > 0) spmv_set_zero<<<(n_zero+255)/256, 256, 0, stream>>>(y, s3, s4);
}

void launch_spmv_generic(const int32_t* offsets, const int32_t* indices,
    const double* x, double* y, int N, cudaStream_t stream)
{
    int wpb = 8;
    int nb = (N + wpb - 1) / wpb;
    if (nb > 0) spmv_gather_generic<<<nb, 256, 0, stream>>>(offsets, indices, x, y, N);
}

void launch_fill_value(double* arr, int N, double val, cudaStream_t s) {
    int g = min((N+255)/256, 1024);
    if (g > 0) fill_value<<<g, 256, 0, s>>>(arr, N, val);
}
void launch_normalize_by_dev_max(double* v, int N, const double* m, cudaStream_t s) {
    int g = min((N+255)/256, 1024);
    if (g > 0) normalize_by_dev_max<<<g, 256, 0, s>>>(v, N, m);
}
void launch_l1_normalize(double* v, int N, const double* sum, cudaStream_t s) {
    int g = min((N+255)/256, 1024);
    if (g > 0) l1_normalize<<<g, 256, 0, s>>>(v, N, sum);
}
void launch_normalize_and_diff(double* nv, const double* ov, int N, const double* m,
    double* p, unsigned int* c, double* d, cudaStream_t s)
{
    int g = min((N+NF_BLOCK-1)/NF_BLOCK, 1024);
    if (g > 0) normalize_and_diff<<<g, NF_BLOCK, 0, s>>>(nv, ov, N, m, p, c, d);
}

void launch_count_degrees(const int32_t* idx, int ne, int32_t* rc, cudaStream_t s) {
    int g = min((ne+255)/256, 4096);
    if (g > 0) count_degrees<<<g, 256, 0, s>>>(idx, ne, rc);
}
void launch_fill_csr_warp(const int32_t* co, const int32_t* ci, int32_t* cp, int32_t* cc, int nv, cudaStream_t s) {
    int wpb = 8, nb = (nv+wpb-1)/wpb;
    if (nb > 0) fill_csr_warp<<<nb, 256, 0, s>>>(co, ci, cp, cc, nv);
}





struct Cache : Cacheable {
    
    double* d_scalar = nullptr;
    double* d_partials = nullptr;
    unsigned int* d_counter = nullptr;
    double* d_diff = nullptr;

    
    double* d_temp_hubs = nullptr;
    int64_t temp_hubs_capacity = 0;

    uint8_t* d_cub_temp = nullptr;
    size_t cub_temp_capacity = 0;

    
    int32_t* d_csr_offsets = nullptr;
    int64_t csr_offsets_capacity = 0;

    int32_t* d_csr_col_idx = nullptr;
    int64_t csr_col_idx_capacity = 0;

    int32_t* d_counts = nullptr;
    int64_t counts_capacity = 0;

    int32_t* d_csr_pos = nullptr;
    int64_t csr_pos_capacity = 0;

    Cache() {
        cudaMalloc(&d_scalar, sizeof(double));
        cudaMalloc(&d_partials, 1024 * sizeof(double));
        cudaMalloc(&d_counter, sizeof(unsigned int));
        cudaMalloc(&d_diff, sizeof(double));
    }

    ~Cache() override {
        cudaFree(d_scalar);
        cudaFree(d_partials);
        cudaFree(d_counter);
        cudaFree(d_diff);
        if (d_temp_hubs) cudaFree(d_temp_hubs);
        if (d_cub_temp) cudaFree(d_cub_temp);
        if (d_csr_offsets) cudaFree(d_csr_offsets);
        if (d_csr_col_idx) cudaFree(d_csr_col_idx);
        if (d_counts) cudaFree(d_counts);
        if (d_csr_pos) cudaFree(d_csr_pos);
    }

    void ensure_temp_hubs(int64_t n) {
        if (temp_hubs_capacity < n) {
            if (d_temp_hubs) cudaFree(d_temp_hubs);
            cudaMalloc(&d_temp_hubs, n * sizeof(double));
            temp_hubs_capacity = n;
        }
    }

    void ensure_cub_temp(size_t bytes) {
        if (cub_temp_capacity < bytes) {
            if (d_cub_temp) cudaFree(d_cub_temp);
            cudaMalloc(&d_cub_temp, bytes);
            cub_temp_capacity = bytes;
        }
    }

    void ensure_csr(int64_t nv, int64_t ne) {
        if (csr_offsets_capacity < nv + 1) {
            if (d_csr_offsets) cudaFree(d_csr_offsets);
            cudaMalloc(&d_csr_offsets, (nv + 1) * sizeof(int32_t));
            csr_offsets_capacity = nv + 1;
        }
        if (csr_col_idx_capacity < ne) {
            if (d_csr_col_idx) cudaFree(d_csr_col_idx);
            cudaMalloc(&d_csr_col_idx, ne * sizeof(int32_t));
            csr_col_idx_capacity = ne;
        }
        if (counts_capacity < nv + 1) {
            if (d_counts) cudaFree(d_counts);
            cudaMalloc(&d_counts, (nv + 1) * sizeof(int32_t));
            counts_capacity = nv + 1;
        }
        if (csr_pos_capacity < nv) {
            if (d_csr_pos) cudaFree(d_csr_pos);
            cudaMalloc(&d_csr_pos, nv * sizeof(int32_t));
            csr_pos_capacity = nv;
        }
    }
};

}  

HitsResultDouble hits_seg(const graph32_t& graph,
                          double* hubs,
                          double* authorities,
                          double epsilon,
                          std::size_t max_iterations,
                          bool has_initial_hubs_guess,
                          bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    if (num_vertices == 0) {
        return HitsResultDouble{0, false, 1e30};
    }

    cudaStream_t stream = 0;

    const int32_t* d_csc_offsets = graph.offsets;
    const int32_t* d_csc_indices = graph.indices;
    bool is_symmetric = graph.is_symmetric;

    const auto& seg = graph.segment_offsets.value();
    int32_t h_seg[5] = {seg[0], seg[1], seg[2], seg[3], seg[4]};

    double tolerance = (double)num_vertices * epsilon;

    
    size_t max_temp = 0;
    cub::DeviceReduce::Max(nullptr, max_temp, (double*)nullptr, (double*)nullptr, num_vertices);
    size_t sum_temp = 0;
    cub::DeviceReduce::Sum(nullptr, sum_temp, (double*)nullptr, (double*)nullptr, num_vertices);
    size_t cub_bytes = (max_temp > sum_temp) ? max_temp : sum_temp;

    
    const int32_t* d_hub_offsets = d_csc_offsets;
    const int32_t* d_hub_indices = d_csc_indices;

    if (!is_symmetric) {
        size_t scan_temp = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, scan_temp, (int32_t*)nullptr, (int32_t*)nullptr, num_vertices + 1);
        if (scan_temp > cub_bytes) cub_bytes = scan_temp;

        cache.ensure_csr(num_vertices, num_edges);
        cache.ensure_cub_temp(cub_bytes);

        int32_t* d_csr_off = cache.d_csr_offsets;
        int32_t* d_csr_col = cache.d_csr_col_idx;
        int32_t* d_cnt_buf = cache.d_counts;
        int32_t* d_csr_pos_buf = cache.d_csr_pos;

        cudaMemsetAsync(d_cnt_buf, 0, (num_vertices + 1) * sizeof(int32_t), stream);
        launch_count_degrees(d_csc_indices, num_edges, d_cnt_buf, stream);
        size_t scan_bytes = scan_temp;
        cub::DeviceScan::ExclusiveSum(cache.d_cub_temp, scan_bytes, d_cnt_buf, d_csr_off, num_vertices + 1, stream);
        cudaMemcpyAsync(d_csr_pos_buf, d_csr_off, num_vertices * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        launch_fill_csr_warp(d_csc_offsets, d_csc_indices, d_csr_pos_buf, d_csr_col, num_vertices, stream);

        d_hub_offsets = d_csr_off;
        d_hub_indices = d_csr_col;
    }

    cache.ensure_cub_temp(cub_bytes);
    cache.ensure_temp_hubs(num_vertices);

    double* d_hubs = hubs;
    double* d_auth = authorities;
    double* d_temp = cache.d_temp_hubs;
    double* d_scalar = cache.d_scalar;
    double* d_partials = cache.d_partials;
    unsigned int* d_cnt = cache.d_counter;
    double* d_diff = cache.d_diff;

    cudaMemsetAsync(d_cnt, 0, sizeof(unsigned int), stream);

    
    if (has_initial_hubs_guess) {
        
        size_t st = sum_temp;
        cub::DeviceReduce::Sum(cache.d_cub_temp, st, d_hubs, d_scalar, num_vertices, stream);
        launch_l1_normalize(d_hubs, num_vertices, d_scalar, stream);
    } else {
        launch_fill_value(d_hubs, num_vertices, 1.0 / num_vertices, stream);
    }

    
    double* prev_hubs = d_hubs;
    double* curr_hubs = d_temp;
    double diff_sum = 1e30;
    std::size_t final_iterations = max_iterations;
    bool converged = false;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        launch_spmv_segmented(d_csc_offsets, d_csc_indices,
                              prev_hubs, d_auth, num_vertices, h_seg, stream);

        
        if (is_symmetric) {
            launch_spmv_segmented(d_hub_offsets, d_hub_indices,
                                  d_auth, curr_hubs, num_vertices, h_seg, stream);
        } else {
            launch_spmv_generic(d_hub_offsets, d_hub_indices,
                                d_auth, curr_hubs, num_vertices, stream);
        }

        
        size_t mt = max_temp;
        cub::DeviceReduce::Max(cache.d_cub_temp, mt, curr_hubs, d_scalar, num_vertices, stream);
        launch_normalize_and_diff(curr_hubs, prev_hubs, num_vertices, d_scalar,
                                  d_partials, d_cnt, d_diff, stream);

        
        cudaMemcpyAsync(&diff_sum, d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        
        std::swap(prev_hubs, curr_hubs);

        if (diff_sum < tolerance) {
            final_iterations = iter + 1;
            converged = true;
            break;
        }
    }

    
    if (prev_hubs != d_hubs) {
        cudaMemcpyAsync(d_hubs, prev_hubs, num_vertices * sizeof(double),
                       cudaMemcpyDeviceToDevice, stream);
    }

    
    
    
    if (normalize) {
        size_t st = sum_temp;
        cub::DeviceReduce::Sum(cache.d_cub_temp, st, d_hubs, d_scalar, num_vertices, stream);
        launch_l1_normalize(d_hubs, num_vertices, d_scalar, stream);
        st = sum_temp;
        cub::DeviceReduce::Sum(cache.d_cub_temp, st, d_auth, d_scalar, num_vertices, stream);
        launch_l1_normalize(d_auth, num_vertices, d_scalar, stream);
    } else {
        size_t mt = max_temp;
        cub::DeviceReduce::Max(cache.d_cub_temp, mt, d_auth, d_scalar, num_vertices, stream);
        launch_normalize_by_dev_max(d_auth, num_vertices, d_scalar, stream);
    }

    cudaStreamSynchronize(stream);

    return HitsResultDouble{final_iterations, converged, diff_sum};
}

}  
