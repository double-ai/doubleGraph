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
#include <cmath>
#include <limits>

namespace aai {

namespace {




struct Cache : Cacheable {
    double* temp_hubs = nullptr;
    double* scalars = nullptr;       
    int32_t* csr_offsets = nullptr;
    int32_t* csr_indices = nullptr;
    int32_t* csr_pos = nullptr;
    void* scan_temp = nullptr;

    int64_t temp_hubs_capacity = 0;
    int64_t scalars_capacity = 0;
    int64_t csr_offsets_capacity = 0;
    int64_t csr_indices_capacity = 0;
    int64_t csr_pos_capacity = 0;
    size_t scan_temp_capacity = 0;

    void ensure_temp_hubs(int64_t n) {
        if (temp_hubs_capacity < n) {
            if (temp_hubs) cudaFree(temp_hubs);
            cudaMalloc(&temp_hubs, n * sizeof(double));
            temp_hubs_capacity = n;
        }
    }

    void ensure_scalars(int64_t n) {
        if (scalars_capacity < n) {
            if (scalars) cudaFree(scalars);
            cudaMalloc(&scalars, n * sizeof(double));
            scalars_capacity = n;
        }
    }

    void ensure_csr_offsets(int64_t n) {
        if (csr_offsets_capacity < n) {
            if (csr_offsets) cudaFree(csr_offsets);
            cudaMalloc(&csr_offsets, n * sizeof(int32_t));
            csr_offsets_capacity = n;
        }
    }

    void ensure_csr_indices(int64_t n) {
        if (csr_indices_capacity < n) {
            if (csr_indices) cudaFree(csr_indices);
            cudaMalloc(&csr_indices, n * sizeof(int32_t));
            csr_indices_capacity = n;
        }
    }

    void ensure_csr_pos(int64_t n) {
        if (csr_pos_capacity < n) {
            if (csr_pos) cudaFree(csr_pos);
            cudaMalloc(&csr_pos, n * sizeof(int32_t));
            csr_pos_capacity = n;
        }
    }

    void ensure_scan_temp(size_t bytes) {
        if (scan_temp_capacity < bytes) {
            if (scan_temp) cudaFree(scan_temp);
            cudaMalloc(&scan_temp, bytes);
            scan_temp_capacity = bytes;
        }
    }

    ~Cache() override {
        if (temp_hubs) cudaFree(temp_hubs);
        if (scalars) cudaFree(scalars);
        if (csr_offsets) cudaFree(csr_offsets);
        if (csr_indices) cudaFree(csr_indices);
        if (csr_pos) cudaFree(csr_pos);
        if (scan_temp) cudaFree(scan_temp);
    }
};




__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}




__global__ void spmv_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const double* __restrict__ x,
    double* __restrict__ y,
    int N
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= N) return;

    int start = offsets[warp_id];
    int end = offsets[warp_id + 1];
    int degree = end - start;

    double sum = 0.0;

    if (degree <= 2) {
        if (lane == 0) {
            for (int i = start; i < end; i++) {
                sum += x[indices[i]];
            }
        }
    } else {
        for (int i = start + lane; i < end; i += 32) {
            sum += x[indices[i]];
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    if (lane == 0) {
        y[warp_id] = sum;
    }
}

__global__ void spmv_thread_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const double* __restrict__ x,
    double* __restrict__ y,
    int N
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    int start = offsets[v];
    int end = offsets[v + 1];
    double sum = 0.0;
    for (int i = start; i < end; i++) {
        sum += x[indices[i]];
    }
    y[v] = sum;
}




__global__ void fill_kernel(double* ptr, double val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ptr[i] = val;
}




__global__ void dual_max_kernel(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* max_a,
    double* max_b,
    int n
) {
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_a, temp_b;

    double va = 0.0, vb = 0.0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < n; i += stride) {
        double aa = a[i], bb = b[i];
        if (aa > va) va = aa;
        if (bb > vb) vb = bb;
    }

    struct MaxOp { __device__ double operator()(double x, double y) { return x > y ? x : y; } };

    double bmax_a = BlockReduce(temp_a).Reduce(va, MaxOp{});
    double bmax_b = BlockReduce(temp_b).Reduce(vb, MaxOp{});

    if (threadIdx.x == 0) {
        if (bmax_a > 0.0) atomicMaxDouble(max_a, bmax_a);
        if (bmax_b > 0.0) atomicMaxDouble(max_b, bmax_b);
    }
}

__global__ void fused_scale_diff_kernel(
    double* __restrict__ curr_hubs,
    const double* __restrict__ prev_hubs,
    double* __restrict__ auth,
    const double* max_hubs,
    const double* max_auth,
    double* diff_result,
    int n
) {
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    double local_diff = 0.0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double mh = *max_hubs;
    double ma = *max_auth;
    double inv_mh = (mh > 0.0) ? 1.0 / mh : 0.0;
    double inv_ma = (ma > 0.0) ? 1.0 / ma : 0.0;

    for (int i = tid; i < n; i += stride) {
        double ch = curr_hubs[i] * inv_mh;
        curr_hubs[i] = ch;
        auth[i] *= inv_ma;
        local_diff += fabs(ch - prev_hubs[i]);
    }

    double block_diff = BlockReduce(temp).Sum(local_diff);
    if (threadIdx.x == 0) {
        atomicAdd(diff_result, block_diff);
    }
}

__global__ void sum_reduce_kernel(const double* __restrict__ x, double* result, int n) {
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    double val = 0.0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) val += x[i];
    double block_sum = BlockReduce(temp).Sum(val);
    if (threadIdx.x == 0) atomicAdd(result, block_sum);
}

__global__ void scale_kernel(double* x, const double* divisor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] /= *divisor;
}




__global__ void count_degrees_kernel(
    const int* __restrict__ indices, int* __restrict__ degree, int num_edges
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < num_edges) atomicAdd(&degree[indices[e]], 1);
}

__global__ void scatter_csr_kernel(
    const int* __restrict__ csc_offsets, const int* __restrict__ csc_indices,
    int* __restrict__ position, int* __restrict__ csr_indices, int N
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= N) return;
    int start = csc_offsets[col];
    int end = csc_offsets[col + 1];
    for (int i = start; i < end; i++) {
        int src = csc_indices[i];
        int pos = atomicAdd(&position[src], 1);
        csr_indices[pos] = col;
    }
}




void launch_spmv(const int* offsets, const int* indices,
                 const double* x, double* y, int N, int avg_degree, cudaStream_t stream) {
    if (N == 0) return;
    if (avg_degree <= 4) {
        int block = 256;
        int grid = (N + block - 1) / block;
        spmv_thread_kernel<<<grid, block, 0, stream>>>(offsets, indices, x, y, N);
    } else {
        int block = 256;
        int warps_needed = N;
        int grid = (warps_needed + (block / 32) - 1) / (block / 32);
        spmv_kernel<<<grid, block, 0, stream>>>(offsets, indices, x, y, N);
    }
}

void launch_fill(double* ptr, double val, int n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    fill_kernel<<<grid, block, 0, stream>>>(ptr, val, n);
}

void launch_dual_max(const double* a, const double* b,
                     double* max_a, double* max_b,
                     int n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 2048) grid = 2048;
    dual_max_kernel<<<grid, block, 0, stream>>>(a, b, max_a, max_b, n);
}

void launch_fused_scale_diff(
    double* curr_hubs, const double* prev_hubs, double* auth,
    const double* max_hubs, const double* max_auth, double* diff_result,
    int n, cudaStream_t stream
) {
    if (n == 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 2048) grid = 2048;
    fused_scale_diff_kernel<<<grid, block, 0, stream>>>(
        curr_hubs, prev_hubs, auth, max_hubs, max_auth, diff_result, n);
}

void launch_sum_reduce(const double* x, double* result, int n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 2048) grid = 2048;
    sum_reduce_kernel<<<grid, block, 0, stream>>>(x, result, n);
}

void launch_scale(double* x, const double* divisor, int n, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (n + block - 1) / block;
    scale_kernel<<<grid, block, 0, stream>>>(x, divisor, n);
}

void launch_count_degrees(const int* indices, int* degree, int num_edges, cudaStream_t stream) {
    if (num_edges == 0) return;
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    count_degrees_kernel<<<grid, block, 0, stream>>>(indices, degree, num_edges);
}

void launch_scatter_csr(const int* csc_offsets, const int* csc_indices,
                        int* position, int* csr_indices, int N, cudaStream_t stream) {
    if (N == 0) return;
    int block = 256;
    int grid = (N + block - 1) / block;
    scatter_csr_kernel<<<grid, block, 0, stream>>>(csc_offsets, csc_indices, position, csr_indices, N);
}

}  

HitsResultDouble hits(const graph32_t& graph,
                      double* hubs,
                      double* authorities,
                      double epsilon,
                      std::size_t max_iterations,
                      bool has_initial_hubs_guess,
                      bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_csc_offsets = graph.offsets;
    const int32_t* d_csc_indices = graph.indices;
    int32_t N = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;
    bool is_symmetric = graph.is_symmetric;
    cudaStream_t stream = 0;

    double tolerance = static_cast<double>(N) * epsilon;
    int avg_degree = (N > 0) ? (E / N) : 0;

    
    if (N == 0) {
        return HitsResultDouble{max_iterations, false, std::numeric_limits<double>::max()};
    }

    
    cache.ensure_temp_hubs(N);
    cache.ensure_scalars(3);

    double* d_hubs = hubs;
    double* d_auth = authorities;
    double* d_temp_hubs = cache.temp_hubs;
    double* d_scalars = cache.scalars;
    double* d_max_hubs = &d_scalars[0];
    double* d_max_auth = &d_scalars[1];
    double* d_diff = &d_scalars[2];

    
    const int32_t* d_trans_offsets;
    const int32_t* d_trans_indices;
    int trans_avg_degree = avg_degree;

    if (is_symmetric) {
        d_trans_offsets = d_csc_offsets;
        d_trans_indices = d_csc_indices;
    } else {
        cache.ensure_csr_offsets(N + 1);
        cache.ensure_csr_indices(E);
        cache.ensure_csr_pos(N + 1);

        int32_t* d_csr_offsets = cache.csr_offsets;
        int32_t* d_csr_indices = cache.csr_indices;
        int32_t* d_csr_pos = cache.csr_pos;

        size_t scan_temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            (void*)nullptr, scan_temp_bytes, (int*)nullptr, (int*)nullptr, N + 1);
        if (scan_temp_bytes < 1024) scan_temp_bytes = 1024;
        cache.ensure_scan_temp(scan_temp_bytes);

        cudaMemsetAsync(d_csr_offsets, 0, (N + 1) * sizeof(int32_t), stream);
        launch_count_degrees(d_csc_indices, d_csr_offsets, E, stream);
        cub::DeviceScan::ExclusiveSum(
            cache.scan_temp, scan_temp_bytes, d_csr_offsets, d_csr_offsets, N + 1, stream);
        cudaMemcpyAsync(d_csr_pos, d_csr_offsets, (N + 1) * sizeof(int32_t),
                        cudaMemcpyDeviceToDevice, stream);
        launch_scatter_csr(d_csc_offsets, d_csc_indices, d_csr_pos, d_csr_indices, N, stream);

        d_trans_offsets = d_csr_offsets;
        d_trans_indices = d_csr_indices;
    }

    
    if (has_initial_hubs_guess) {
        
        launch_fill(d_diff, 0.0, 1, stream);
        launch_sum_reduce(d_hubs, d_diff, N, stream);
        launch_scale(d_hubs, d_diff, N, stream);
    } else {
        launch_fill(d_hubs, 1.0 / N, N, stream);
    }

    
    double* prev_hubs = d_hubs;
    double* curr_hubs = d_temp_hubs;
    double diff_sum = std::numeric_limits<double>::max();
    size_t iter = 0;
    bool detected_convergence = false;

    if (max_iterations == 0) goto done;

    while (true) {
        
        launch_spmv(d_csc_offsets, d_csc_indices, prev_hubs, d_auth, N, avg_degree, stream);

        
        launch_spmv(d_trans_offsets, d_trans_indices, d_auth, curr_hubs, N, trans_avg_degree, stream);

        
        launch_fill(d_scalars, 0.0, 3, stream);

        
        launch_dual_max(curr_hubs, d_auth, d_max_hubs, d_max_auth, N, stream);

        
        launch_fused_scale_diff(curr_hubs, prev_hubs, d_auth,
                                d_max_hubs, d_max_auth, d_diff, N, stream);

        
        std::swap(prev_hubs, curr_hubs);
        iter++;

        
        cudaMemcpyAsync(&diff_sum, d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (diff_sum < tolerance) {
            detected_convergence = true;
            break;
        }
        if (iter >= max_iterations) break;
    }

done:
    
    if (normalize) {
        launch_fill(d_diff, 0.0, 1, stream);
        launch_sum_reduce(prev_hubs, d_diff, N, stream);
        launch_scale(prev_hubs, d_diff, N, stream);

        launch_fill(d_diff, 0.0, 1, stream);
        launch_sum_reduce(d_auth, d_diff, N, stream);
        launch_scale(d_auth, d_diff, N, stream);
    }

    
    if (prev_hubs != d_hubs) {
        cudaMemcpyAsync(d_hubs, prev_hubs, N * sizeof(double),
                        cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    return HitsResultDouble{iter, detected_convergence, diff_sum};
}

}  
