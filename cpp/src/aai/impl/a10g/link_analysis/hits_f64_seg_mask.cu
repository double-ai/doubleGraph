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
#include <cmath>
#include <limits>

namespace aai {

namespace {



struct Cache : Cacheable {
    
    int32_t* active_counts = nullptr;
    int32_t* filt_offsets = nullptr;
    int32_t* filt_indices = nullptr;
    int32_t* out_degree = nullptr;
    int32_t* csr_offsets = nullptr;
    int32_t* csr_indices = nullptr;
    int32_t* csr_write_pos = nullptr;

    
    double* hubs_temp = nullptr;
    double* scalars = nullptr;  

    
    uint8_t* cub_temp = nullptr;

    
    int32_t active_counts_cap = 0;
    int32_t filt_offsets_cap = 0;
    int32_t filt_indices_cap = 0;
    int32_t out_degree_cap = 0;
    int32_t csr_offsets_cap = 0;
    int32_t csr_indices_cap = 0;
    int32_t csr_write_pos_cap = 0;
    int32_t hubs_temp_cap = 0;
    bool scalars_allocated = false;
    size_t cub_temp_cap = 0;

    void ensure_vertex_buffers(int32_t nv) {
        if (active_counts_cap < nv) {
            if (active_counts) cudaFree(active_counts);
            cudaMalloc(&active_counts, (size_t)nv * sizeof(int32_t));
            active_counts_cap = nv;
        }
        if (filt_offsets_cap < nv + 1) {
            if (filt_offsets) cudaFree(filt_offsets);
            cudaMalloc(&filt_offsets, (size_t)(nv + 1) * sizeof(int32_t));
            filt_offsets_cap = nv + 1;
        }
        if (out_degree_cap < nv) {
            if (out_degree) cudaFree(out_degree);
            cudaMalloc(&out_degree, (size_t)nv * sizeof(int32_t));
            out_degree_cap = nv;
        }
        if (csr_offsets_cap < nv + 1) {
            if (csr_offsets) cudaFree(csr_offsets);
            cudaMalloc(&csr_offsets, (size_t)(nv + 1) * sizeof(int32_t));
            csr_offsets_cap = nv + 1;
        }
        if (csr_write_pos_cap < nv) {
            if (csr_write_pos) cudaFree(csr_write_pos);
            cudaMalloc(&csr_write_pos, (size_t)nv * sizeof(int32_t));
            csr_write_pos_cap = nv;
        }
        if (hubs_temp_cap < nv) {
            if (hubs_temp) cudaFree(hubs_temp);
            cudaMalloc(&hubs_temp, (size_t)nv * sizeof(double));
            hubs_temp_cap = nv;
        }
    }

    void ensure_edge_buffers(int32_t ne) {
        int32_t alloc_ne = std::max(ne, 1);
        if (filt_indices_cap < alloc_ne) {
            if (filt_indices) cudaFree(filt_indices);
            cudaMalloc(&filt_indices, (size_t)alloc_ne * sizeof(int32_t));
            filt_indices_cap = alloc_ne;
        }
        if (csr_indices_cap < alloc_ne) {
            if (csr_indices) cudaFree(csr_indices);
            cudaMalloc(&csr_indices, (size_t)alloc_ne * sizeof(int32_t));
            csr_indices_cap = alloc_ne;
        }
    }

    void ensure_scalars() {
        if (!scalars_allocated) {
            cudaMalloc(&scalars, 2 * sizeof(double));
            scalars_allocated = true;
        }
    }

    void ensure_cub_temp(size_t bytes) {
        bytes = (bytes + 255) & ~255ULL;
        if (cub_temp_cap < bytes) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, bytes);
            cub_temp_cap = bytes;
        }
    }

    ~Cache() override {
        if (active_counts) cudaFree(active_counts);
        if (filt_offsets) cudaFree(filt_offsets);
        if (filt_indices) cudaFree(filt_indices);
        if (out_degree) cudaFree(out_degree);
        if (csr_offsets) cudaFree(csr_offsets);
        if (csr_indices) cudaFree(csr_indices);
        if (csr_write_pos) cudaFree(csr_write_pos);
        if (hubs_temp) cudaFree(hubs_temp);
        if (scalars) cudaFree(scalars);
        if (cub_temp) cudaFree(cub_temp);
    }
};



__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_counts,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int count = 0;
        for (int e = start; e < end; e++) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1) {
                count++;
            }
        }
        active_counts[v] = count;
    }
}

__global__ void compact_edges_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int old_start = old_offsets[v];
        int old_end = old_offsets[v + 1];
        int new_pos = new_offsets[v];
        for (int e = old_start; e < old_end; e++) {
            if ((edge_mask[e >> 5] >> (e & 31)) & 1) {
                new_indices[new_pos++] = old_indices[e];
            }
        }
    }
}

__global__ void count_out_degree_kernel(
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ out_degree,
    int32_t num_edges)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < num_edges) {
        atomicAdd(&out_degree[indices[e]], 1);
    }
}

__global__ void build_csr_kernel(
    const int32_t* __restrict__ csc_offsets,
    const int32_t* __restrict__ csc_indices,
    int32_t* __restrict__ csr_write_pos,
    int32_t* __restrict__ csr_indices,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int start = csc_offsets[v];
        int end = csc_offsets[v + 1];
        for (int e = start; e < end; e++) {
            int u = csc_indices[e];
            int pos = atomicAdd(&csr_write_pos[u], 1);
            csr_indices[pos] = v;
        }
    }
}



__global__ void spmv_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ in_vec,
    double* __restrict__ out_vec,
    int32_t start_vertex,
    int32_t num_vertices_seg)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_vertices_seg) return;
    int v = start_vertex + warp_id;

    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);

    double sum = 0.0;
    for (int e = start + lane; e < end; e += 32) {
        sum += __ldg(&in_vec[__ldg(&indices[e])]);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        out_vec[v] = sum;
    }
}

__global__ void spmv_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ in_vec,
    double* __restrict__ out_vec,
    int32_t start_vertex,
    int32_t num_vertices_seg)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices_seg) return;
    int v = start_vertex + tid;

    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);
    double sum = 0.0;
    for (int e = start; e < end; e++) {
        sum += __ldg(&in_vec[__ldg(&indices[e])]);
    }
    out_vec[v] = sum;
}



__global__ void normalize_diff_reduce_kernel(
    double* __restrict__ new_hubs,
    const double* __restrict__ old_hubs,
    const double* __restrict__ max_ptr,
    double* __restrict__ global_diff_sum,
    int32_t num_vertices)
{
    double max_val = *max_ptr;
    double inv_max = (max_val > 0.0) ? (1.0 / max_val) : 0.0;

    int v = blockIdx.x * blockDim.x + threadIdx.x;

    double local_diff = 0.0;
    if (v < num_vertices) {
        double val = new_hubs[v] * inv_max;
        new_hubs[v] = val;
        local_diff = fabs(val - old_hubs[v]);
    }

    
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    double block_sum = BlockReduce(temp_storage).Sum(local_diff);

    if (threadIdx.x == 0) {
        atomicAdd(global_diff_sum, block_sum);
    }
}



__global__ void normalize_by_max_kernel(
    double* __restrict__ arr,
    const double* __restrict__ max_ptr,
    int32_t num_vertices)
{
    double max_val = *max_ptr;
    if (max_val <= 0.0) return;
    double inv_max = 1.0 / max_val;
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        arr[v] *= inv_max;
    }
}

__global__ void normalize_by_sum_kernel(
    double* __restrict__ arr,
    const double* __restrict__ sum_ptr,
    int32_t num_vertices)
{
    double sum_val = *sum_ptr;
    if (sum_val <= 0.0) return;
    double inv_sum = 1.0 / sum_val;
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        arr[v] *= inv_sum;
    }
}

__global__ void fill_double_kernel(double* __restrict__ arr, double val, int32_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        arr[i] = val;
    }
}



static void launch_count_active_edges(const int32_t* offsets, const uint32_t* edge_mask,
    int32_t* active_counts, int32_t num_vertices, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    count_active_edges_kernel<<<grid, block, 0, stream>>>(offsets, edge_mask, active_counts, num_vertices);
}

static void launch_compact_edges(const int32_t* old_offsets, const int32_t* old_indices,
    const uint32_t* edge_mask, const int32_t* new_offsets,
    int32_t* new_indices, int32_t num_vertices, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    compact_edges_kernel<<<grid, block, 0, stream>>>(old_offsets, old_indices, edge_mask, new_offsets, new_indices, num_vertices);
}

static void launch_count_out_degree(const int32_t* indices, int32_t* out_degree,
    int32_t num_edges, cudaStream_t stream) {
    int block = 256;
    int grid = (num_edges + block - 1) / block;
    count_out_degree_kernel<<<grid, block, 0, stream>>>(indices, out_degree, num_edges);
}

static void launch_build_csr(const int32_t* csc_offsets, const int32_t* csc_indices,
    int32_t* csr_write_pos, int32_t* csr_indices,
    int32_t num_vertices, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    build_csr_kernel<<<grid, block, 0, stream>>>(csc_offsets, csc_indices, csr_write_pos, csr_indices, num_vertices);
}

static void launch_spmv_segmented(const int32_t* offsets, const int32_t* indices,
    const double* in_vec, double* out_vec,
    int32_t num_vertices, int32_t warp_end, int32_t thread_end,
    cudaStream_t stream)
{
    if (warp_end > 0) {
        int warps_needed = warp_end;
        int tpb = 256;
        int grid = (warps_needed + tpb/32 - 1) / (tpb/32);
        spmv_warp_kernel<<<grid, tpb, 0, stream>>>(offsets, indices, in_vec, out_vec, 0, warp_end);
    }
    int low_count = thread_end - warp_end;
    if (low_count > 0) {
        int block = 256;
        int grid = (low_count + block - 1) / block;
        spmv_thread_kernel<<<grid, block, 0, stream>>>(offsets, indices, in_vec, out_vec, warp_end, low_count);
    }
    int zero_count = num_vertices - thread_end;
    if (zero_count > 0) {
        cudaMemsetAsync(out_vec + thread_end, 0, zero_count * sizeof(double), stream);
    }
}

static void launch_spmv_simple(const int32_t* offsets, const int32_t* indices,
    const double* in_vec, double* out_vec,
    int32_t num_vertices, cudaStream_t stream)
{
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    spmv_thread_kernel<<<grid, block, 0, stream>>>(offsets, indices, in_vec, out_vec, 0, num_vertices);
}

static void launch_spmv_warp(const int32_t* offsets, const int32_t* indices,
    const double* in_vec, double* out_vec,
    int32_t num_vertices, cudaStream_t stream)
{
    int warps_needed = num_vertices;
    int tpb = 256;
    int grid = (warps_needed + tpb/32 - 1) / (tpb/32);
    spmv_warp_kernel<<<grid, tpb, 0, stream>>>(offsets, indices, in_vec, out_vec, 0, num_vertices);
}

static void launch_normalize_diff_reduce(double* new_hubs, const double* old_hubs,
    const double* max_ptr, double* global_diff_sum,
    int32_t num_vertices, cudaStream_t stream)
{
    cudaMemsetAsync(global_diff_sum, 0, sizeof(double), stream);
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    normalize_diff_reduce_kernel<<<grid, block, 0, stream>>>(
        new_hubs, old_hubs, max_ptr, global_diff_sum, num_vertices);
}

static void launch_normalize_by_max(double* arr, const double* max_ptr,
    int32_t num_vertices, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    normalize_by_max_kernel<<<grid, block, 0, stream>>>(arr, max_ptr, num_vertices);
}

static void launch_normalize_by_sum(double* arr, const double* sum_ptr,
    int32_t num_vertices, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    normalize_by_sum_kernel<<<grid, block, 0, stream>>>(arr, sum_ptr, num_vertices);
}

static void launch_fill_double(double* arr, double val, int32_t n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    fill_double_kernel<<<grid, block, 0, stream>>>(arr, val, n);
}

}  

HitsResultDouble hits_seg_mask(const graph32_t& graph,
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
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int32_t csc_warp_end = seg[2];
    int32_t csc_thread_end = seg[3];

    double tolerance = static_cast<double>(num_vertices) * epsilon;
    cudaStream_t stream = 0;

    if (num_vertices == 0) {
        return HitsResultDouble{max_iterations, false, std::numeric_limits<double>::max()};
    }

    
    size_t temp1 = 0, temp2 = 0, temp3 = 0;
    cub::DeviceScan::InclusiveSum((void*)nullptr, temp1, (int32_t*)nullptr, (int32_t*)nullptr, num_vertices);
    cub::DeviceReduce::Max((void*)nullptr, temp2, (double*)nullptr, (double*)nullptr, num_vertices);
    cub::DeviceReduce::Sum((void*)nullptr, temp3, (double*)nullptr, (double*)nullptr, num_vertices);
    size_t max_temp = std::max({temp1, temp2, temp3});

    
    cache.ensure_vertex_buffers(num_vertices);
    cache.ensure_scalars();
    cache.ensure_cub_temp(max_temp);

    int32_t* d_active_counts = cache.active_counts;
    int32_t* d_filt_offsets = cache.filt_offsets;
    int32_t* d_out_degree = cache.out_degree;
    int32_t* d_csr_offsets = cache.csr_offsets;
    int32_t* d_csr_write_pos = cache.csr_write_pos;
    double* d_max_scalar = cache.scalars;
    double* d_diff_sum = cache.scalars + 1;
    void* d_cub_temp = cache.cub_temp;
    size_t cub_temp_bytes = cache.cub_temp_cap;

    
    launch_count_active_edges(d_offsets, d_edge_mask, d_active_counts, num_vertices, stream);

    
    cudaMemsetAsync(d_filt_offsets, 0, sizeof(int32_t), stream);
    cub::DeviceScan::InclusiveSum(d_cub_temp, cub_temp_bytes, d_active_counts, d_filt_offsets + 1, num_vertices, stream);

    int32_t total_filtered_edges;
    cudaMemcpyAsync(&total_filtered_edges, d_filt_offsets + num_vertices, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    
    cache.ensure_edge_buffers(total_filtered_edges);
    int32_t* d_filt_indices = cache.filt_indices;
    int32_t* d_csr_indices = cache.csr_indices;

    
    if (total_filtered_edges > 0)
        launch_compact_edges(d_offsets, d_indices, d_edge_mask, d_filt_offsets, d_filt_indices, num_vertices, stream);

    
    cudaMemsetAsync(d_out_degree, 0, num_vertices * sizeof(int32_t), stream);
    if (total_filtered_edges > 0)
        launch_count_out_degree(d_filt_indices, d_out_degree, total_filtered_edges, stream);

    cudaMemsetAsync(d_csr_offsets, 0, sizeof(int32_t), stream);
    cub::DeviceScan::InclusiveSum(d_cub_temp, cub_temp_bytes, d_out_degree, d_csr_offsets + 1, num_vertices, stream);

    cudaMemcpyAsync(d_csr_write_pos, d_csr_offsets, num_vertices * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
    if (total_filtered_edges > 0)
        launch_build_csr(d_filt_offsets, d_filt_indices, d_csr_write_pos, d_csr_indices, num_vertices, stream);

    double avg_degree = (num_vertices > 0) ? (double)total_filtered_edges / num_vertices : 0;
    bool use_warp_csr = (avg_degree >= 8.0);

    
    double* d_hubs_a = hubs;
    double* d_hubs_b = cache.hubs_temp;

    
    double* d_prev_hubs = d_hubs_a;
    double* d_curr_hubs = d_hubs_b;

    if (has_initial_hubs_guess) {
        
        cub::DeviceReduce::Sum(d_cub_temp, cub_temp_bytes, d_prev_hubs, d_max_scalar, num_vertices, stream);
        launch_normalize_by_sum(d_prev_hubs, d_max_scalar, num_vertices, stream);
    } else {
        double init_val = 1.0 / num_vertices;
        launch_fill_double(d_prev_hubs, init_val, num_vertices, stream);
    }

    
    std::size_t iterations = 0;
    double diff_sum = std::numeric_limits<double>::max();

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        launch_spmv_segmented(d_filt_offsets, d_filt_indices, d_prev_hubs, authorities,
            num_vertices, csc_warp_end, csc_thread_end, stream);

        
        if (use_warp_csr) {
            launch_spmv_warp(d_csr_offsets, d_csr_indices, authorities, d_curr_hubs, num_vertices, stream);
        } else {
            launch_spmv_simple(d_csr_offsets, d_csr_indices, authorities, d_curr_hubs, num_vertices, stream);
        }

        
        cub::DeviceReduce::Max(d_cub_temp, cub_temp_bytes, d_curr_hubs, d_max_scalar, num_vertices, stream);

        
        launch_normalize_diff_reduce(d_curr_hubs, d_prev_hubs, d_max_scalar, d_diff_sum, num_vertices, stream);

        std::swap(d_prev_hubs, d_curr_hubs);
        iterations = iter + 1;

        
        cudaMemcpyAsync(&diff_sum, d_diff_sum, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (diff_sum < tolerance) break;
    }

    
    cub::DeviceReduce::Max(d_cub_temp, cub_temp_bytes, authorities, d_max_scalar, num_vertices, stream);
    launch_normalize_by_max(authorities, d_max_scalar, num_vertices, stream);

    
    if (normalize) {
        cub::DeviceReduce::Sum(d_cub_temp, cub_temp_bytes, d_prev_hubs, d_max_scalar, num_vertices, stream);
        launch_normalize_by_sum(d_prev_hubs, d_max_scalar, num_vertices, stream);
        cub::DeviceReduce::Sum(d_cub_temp, cub_temp_bytes, authorities, d_max_scalar, num_vertices, stream);
        launch_normalize_by_sum(authorities, d_max_scalar, num_vertices, stream);
    }

    
    if (d_prev_hubs != hubs) {
        cudaMemcpyAsync(hubs, d_prev_hubs, num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    }

    bool converged = (iterations < max_iterations);

    cudaStreamSynchronize(stream);

    return HitsResultDouble{iterations, converged, diff_sum};
}

}  
