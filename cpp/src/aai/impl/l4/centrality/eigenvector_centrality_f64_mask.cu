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
#include <cooperative_groups.h>
#include <cub/device/device_scan.cuh>
#include <cstdint>
#include <cmath>
#include <algorithm>

namespace aai {

namespace {

namespace cg = cooperative_groups;

#ifndef MAX_COOP_BLOCKS
#define MAX_COOP_BLOCKS 512
#endif

#define BLOCK_SIZE 256



struct Cache : Cacheable {
    
    double* block_buf = nullptr;     
    double* scalars = nullptr;       
    int32_t* selector = nullptr;     
    int32_t* counts = nullptr;       
    int64_t* d_iterations = nullptr; 
    bool* d_converged = nullptr;     

    
    int32_t* new_offsets = nullptr;
    uint8_t* cat = nullptr;
    float* buf_f = nullptr;
    int32_t* new_indices = nullptr;
    float* new_weights = nullptr;
    void* cub_temp = nullptr;
    int32_t* medium_list = nullptr;
    int32_t* heavy_list = nullptr;

    
    int64_t new_offsets_cap = 0;
    int64_t cat_cap = 0;
    int64_t buf_f_cap = 0;
    int64_t new_indices_cap = 0;
    int64_t new_weights_cap = 0;
    size_t cub_temp_cap = 0;
    int32_t medium_list_cap = 0;
    int32_t heavy_list_cap = 0;

    Cache() {
        cudaMalloc(&block_buf, 1024 * sizeof(double));
        cudaMalloc(&scalars, 4 * sizeof(double));
        cudaMalloc(&selector, sizeof(int32_t));
        cudaMalloc(&counts, 2 * sizeof(int32_t));
        cudaMalloc(&d_iterations, sizeof(int64_t));
        cudaMalloc(&d_converged, sizeof(bool));
    }

    ~Cache() override {
        if (block_buf) cudaFree(block_buf);
        if (scalars) cudaFree(scalars);
        if (selector) cudaFree(selector);
        if (counts) cudaFree(counts);
        if (d_iterations) cudaFree(d_iterations);
        if (d_converged) cudaFree(d_converged);
        if (new_offsets) cudaFree(new_offsets);
        if (cat) cudaFree(cat);
        if (buf_f) cudaFree(buf_f);
        if (new_indices) cudaFree(new_indices);
        if (new_weights) cudaFree(new_weights);
        if (cub_temp) cudaFree(cub_temp);
        if (medium_list) cudaFree(medium_list);
        if (heavy_list) cudaFree(heavy_list);
    }

    void ensure(int32_t num_vertices, int32_t num_edges, size_t cub_bytes,
                int32_t med_cap, int32_t hvy_cap) {
        int64_t nv1 = (int64_t)num_vertices + 1;
        if (new_offsets_cap < nv1) {
            if (new_offsets) cudaFree(new_offsets);
            cudaMalloc(&new_offsets, nv1 * sizeof(int32_t));
            new_offsets_cap = nv1;
        }
        if (cat_cap < num_vertices) {
            if (cat) cudaFree(cat);
            cudaMalloc(&cat, (int64_t)num_vertices * sizeof(uint8_t));
            cat_cap = num_vertices;
        }
        int64_t buf_needed = 2LL * num_vertices;
        if (buf_f_cap < buf_needed) {
            if (buf_f) cudaFree(buf_f);
            cudaMalloc(&buf_f, buf_needed * sizeof(float));
            buf_f_cap = buf_needed;
        }
        if (new_indices_cap < num_edges) {
            if (new_indices) cudaFree(new_indices);
            cudaMalloc(&new_indices, (int64_t)num_edges * sizeof(int32_t));
            new_indices_cap = num_edges;
        }
        if (new_weights_cap < num_edges) {
            if (new_weights) cudaFree(new_weights);
            cudaMalloc(&new_weights, (int64_t)num_edges * sizeof(float));
            new_weights_cap = num_edges;
        }
        if (cub_temp_cap < cub_bytes) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, cub_bytes);
            cub_temp_cap = cub_bytes;
        }
        if (medium_list_cap < med_cap) {
            if (medium_list) cudaFree(medium_list);
            cudaMalloc(&medium_list, (int64_t)med_cap * sizeof(int32_t));
            medium_list_cap = med_cap;
        }
        if (heavy_list_cap < hvy_cap) {
            if (heavy_list) cudaFree(heavy_list);
            cudaMalloc(&heavy_list, (int64_t)hvy_cap * sizeof(int32_t));
            heavy_list_cap = hvy_cap;
        }
    }
};



__device__ __forceinline__ double warp_reduce_sum(double v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ double block_reduce_sum(double v) {
    __shared__ double shared[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    v = warp_reduce_sum(v);
    if (lane == 0) shared[warp_id] = v;
    __syncthreads();

    v = (threadIdx.x < (BLOCK_SIZE >> 5)) ? shared[threadIdx.x] : 0.0;
    if (warp_id == 0) v = warp_reduce_sum(v);
    return v;
}

__device__ __forceinline__ float warp_reduce_sum_f(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_reduce_sum_f(float v) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    v = warp_reduce_sum_f(v);
    if (lane == 0) shared[warp_id] = v;
    __syncthreads();

    v = (threadIdx.x < (BLOCK_SIZE >> 5)) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) v = warp_reduce_sum_f(v);
    return v;
}



__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ counts_plus1, 
    int32_t n)
{
    int v = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (v >= n) return;
    int start = offsets[v];
    int end = offsets[v + 1];
    int cnt = 0;
    if (start < end) {
        int w0 = start >> 5;
        int w1 = (end - 1) >> 5;
        uint32_t lo = 0xffffffffu << (start & 31);
        uint32_t hi;
        int end_bit = end & 31;
        hi = (end_bit == 0) ? 0xffffffffu : ((1u << end_bit) - 1u);
        if (w0 == w1) {
            cnt = __popc(edge_mask[w0] & lo & hi);
        } else {
            cnt += __popc(edge_mask[w0] & lo);
            for (int w = w0 + 1; w < w1; ++w) {
                cnt += __popc(edge_mask[w]);
            }
            cnt += __popc(edge_mask[w1] & hi);
        }
    }
    counts_plus1[v] = cnt;
}

__global__ void compact_active_edges_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const double* __restrict__ old_weights,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    float* __restrict__ new_weights,
    int32_t n)
{
    int v = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (v >= n) return;
    int old_start = old_offsets[v];
    int old_end = old_offsets[v + 1];
    int out = new_offsets[v];
    for (int e = old_start; e < old_end; ++e) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1u) {
            new_indices[out] = old_indices[e];
            new_weights[out] = (float)old_weights[e];
            ++out;
        }
    }
}

void launch_prefilter_csc_masked_to_csc(
    const int32_t* offsets, const int32_t* indices, const double* weights, const uint32_t* edge_mask,
    int32_t* new_offsets, int32_t* new_indices, float* new_weights,
    int32_t num_vertices,
    void* cub_temp, size_t cub_temp_bytes)
{
    int block = BLOCK_SIZE;
    int grid = (num_vertices + block - 1) / block;
    cudaMemsetAsync(new_offsets, 0, sizeof(int32_t));
    count_active_edges_kernel<<<grid, block>>>(offsets, edge_mask, new_offsets + 1, num_vertices);
    cub::DeviceScan::InclusiveSum(cub_temp, cub_temp_bytes, new_offsets + 1, new_offsets + 1, num_vertices);
    compact_active_edges_kernel<<<grid, block>>>(offsets, indices, weights, edge_mask, new_offsets, new_indices, new_weights, num_vertices);
}



__global__ void build_bins_kernel(
    const int32_t* __restrict__ offsets,
    uint8_t* __restrict__ cat,
    int32_t* __restrict__ medium_list,
    int32_t* __restrict__ heavy_list,
    int32_t medium_capacity,
    int32_t heavy_capacity,
    int32_t* __restrict__ counts_out,
    int32_t n)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(gridDim.x * blockDim.x);
    for (int v = tid; v < n; v += stride) {
        int deg = offsets[v + 1] - offsets[v];
        uint8_t c = 0;
        if (deg >= 1024) {
            c = 2;
            int pos = atomicAdd(&counts_out[1], 1);
            if (pos < heavy_capacity) heavy_list[pos] = v;
        } else if (deg >= 32) {
            c = 1;
            int pos = atomicAdd(&counts_out[0], 1);
            if (pos < medium_capacity) medium_list[pos] = v;
        }
        cat[v] = c;
    }
}

void launch_build_degree_bins(
    const int32_t* offsets, int32_t num_vertices, int32_t ,
    uint8_t* cat, int32_t* medium_list, int32_t* heavy_list,
    int32_t medium_capacity, int32_t heavy_capacity,
    int32_t* counts_out)
{
    int block = BLOCK_SIZE;
    
    int grid = (num_vertices + block - 1) / block;
    if (grid > 65535) grid = 65535;
    cudaMemsetAsync(counts_out, 0, 2 * sizeof(int32_t));
    build_bins_kernel<<<grid, block>>>(offsets, cat, medium_list, heavy_list, medium_capacity, heavy_capacity, counts_out, num_vertices);
}



__global__ void select_and_cast_kernel(const float* __restrict__ buf0, const float* __restrict__ buf1,
                                      const int32_t* __restrict__ selector,
                                      double* __restrict__ out, int32_t n)
{
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    int sel = selector[0];
    const float* src = (sel == 0) ? buf0 : buf1;
    out[i] = (double)src[i];
}

void launch_select_and_cast_f32_to_f64(
    const float* buf0, const float* buf1, const int32_t* selector,
    double* out, int32_t n)
{
    int block = BLOCK_SIZE;
    int grid = (n + block - 1) / block;
    select_and_cast_kernel<<<grid, block>>>(buf0, buf1, selector, out, n);
}



__global__ void eigenvector_persistent_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint8_t* __restrict__ cat,
    const int32_t* __restrict__ medium_list,
    const int32_t* __restrict__ heavy_list,
    const int32_t* __restrict__ counts,
    float* __restrict__ buf0,
    float* __restrict__ buf1,
    int32_t n,
    double epsilon,
    uint64_t max_iter,
    const double* __restrict__ init_c,
    bool use_init,
    double* __restrict__ block_sumsq,
    double* __restrict__ block_delta,
    double* __restrict__ scalars, 
    int32_t* __restrict__ final_selector,
    int64_t* __restrict__ out_iterations,
    bool* __restrict__ out_converged)
{
    cg::grid_group grid = cg::this_grid();

    float* x_old = buf0;
    float* x_new = buf1;

    
    for (int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); i < n; i += (int)(gridDim.x * blockDim.x)) {
        if (use_init) x_old[i] = (float)init_c[i];
        else x_old[i] = (float)(1.0 / (double)n);
    }
    grid.sync();

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        final_selector[0] = 0;
        out_iterations[0] = 0;
        out_converged[0] = false;
    }
    grid.sync();

    if (max_iter == 0) {
        
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            final_selector[0] = 0; 
            out_iterations[0] = 0;
            out_converged[0] = false;
        }
        return;
    }

    const double threshold = epsilon * (double)n;

    const int medium_count = counts[0];
    const int heavy_count = counts[1];

    for (uint64_t iter = 0; iter < max_iter; ++iter) {
        
        double local_sumsq = 0.0;

        
        for (int hv = (int)blockIdx.x; hv < heavy_count; hv += (int)gridDim.x) {
            int v = heavy_list[hv];
            int start = offsets[v];
            int end = offsets[v + 1];
            float sum = (threadIdx.x == 0) ? __ldg((const float*)(x_old + v)) : 0.0f;
            for (int e = start + (int)threadIdx.x; e < end; e += (int)blockDim.x) {
                int src = indices[e];
                float xsrc = __ldg((const float*)(x_old + src));
                sum = fmaf(weights[e], xsrc, sum);
            }
            sum = block_reduce_sum_f(sum);
            if (threadIdx.x == 0) {
                x_new[v] = sum;
                double dsum = (double)sum;
                local_sumsq += dsum * dsum;
            }
        }

        
        constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
        int warp_in_block = (int)(threadIdx.x >> 5);
        int lane = (int)(threadIdx.x & 31);
        int global_warp = (int)blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
        int warp_stride = (int)gridDim.x * WARPS_PER_BLOCK;
        for (int mv = global_warp; mv < medium_count; mv += warp_stride) {
            int v = medium_list[mv];
            int start = offsets[v];
            int end = offsets[v + 1];
            float sum = (lane == 0) ? __ldg((const float*)(x_old + v)) : 0.0f;
            for (int e = start + lane; e < end; e += 32) {
                int src = indices[e];
                float xsrc = __ldg((const float*)(x_old + src));
                sum = fmaf(weights[e], xsrc, sum);
            }
            sum = warp_reduce_sum_f(sum);
            if (lane == 0) {
                x_new[v] = sum;
                double dsum = (double)sum;
                local_sumsq += dsum * dsum;
            }
        }

        
        int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        int stride = (int)(gridDim.x * blockDim.x);
        for (int v = tid; v < n; v += stride) {
            if (cat[v] != 0) continue;
            int start = offsets[v];
            int end = offsets[v + 1];
            float sum = __ldg((const float*)(x_old + v));
            for (int e = start; e < end; ++e) {
                int src = indices[e];
                float xsrc = __ldg((const float*)(x_old + src));
                sum = fmaf(weights[e], xsrc, sum);
            }
            x_new[v] = sum;
            double dsum = (double)sum;
            local_sumsq += dsum * dsum;
        }

        double block_sum = block_reduce_sum(local_sumsq);
        if (threadIdx.x == 0) {
            block_sumsq[blockIdx.x] = block_sum;
        }
        grid.sync();

        
        if (blockIdx.x == 0) {
            double s = 0.0;
            for (int i = (int)threadIdx.x; i < (int)gridDim.x; i += (int)blockDim.x) {
                s += block_sumsq[i];
            }
            s = block_reduce_sum(s);
            if (threadIdx.x == 0) {
                scalars[0] = (s > 0.0) ? (1.0 / sqrt(s)) : 0.0; 
            }
        }
        grid.sync();
        double inv_norm = scalars[0];

        
        double local_delta = 0.0;
        for (int v = tid; v < n; v += stride) {
            double newv = (double)x_new[v] * inv_norm;
            double oldv = (double)x_old[v];
            x_new[v] = (float)newv;
            local_delta += fabs(newv - oldv);
        }

        double bdel = block_reduce_sum(local_delta);
        if (threadIdx.x == 0) {
            block_delta[blockIdx.x] = bdel;
        }
        grid.sync();

        bool done = false;
        bool converged = false;
        if (blockIdx.x == 0) {
            double d = 0.0;
            for (int i = (int)threadIdx.x; i < (int)gridDim.x; i += (int)blockDim.x) {
                d += block_delta[i];
            }
            d = block_reduce_sum(d);
            if (threadIdx.x == 0) {
                scalars[1] = d;
                converged = (d < threshold);
                done = converged || (iter + 1 >= max_iter);
                out_iterations[0] = (int64_t)(iter + 1);
                out_converged[0] = converged;
                
                final_selector[0] = (x_new == buf0) ? 0 : 1;
                
                scalars[2] = done ? 1.0 : 0.0;
            }
        }
        grid.sync();
        done = (scalars[2] != 0.0);
        if (done) {
            break;
        }

        
        float* tmp = x_old;
        x_old = x_new;
        x_new = tmp;
        grid.sync();
    }
}

void launch_eigenvector_centrality_persistent(
    const int32_t* offsets, const int32_t* indices, const float* weights,
    const uint8_t* cat,
    const int32_t* medium_list, const int32_t* heavy_list,
    const int32_t* counts,
    float* buf0, float* buf1,
    int32_t num_vertices,
    double epsilon, int64_t max_iterations,
    const double* initial_centralities, bool use_initial,
    double* block_sumsq, double* block_delta,
    double* scalars, int32_t* final_selector,
    int64_t* out_iterations, bool* out_converged)
{
    
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocksPerSM,
        eigenvector_persistent_kernel,
        BLOCK_SIZE,
        0);
    int grid = prop.multiProcessorCount * maxBlocksPerSM;
    if (grid > MAX_COOP_BLOCKS) grid = MAX_COOP_BLOCKS;
    if (grid < 1) grid = 1;

    uint64_t eff_max = (max_iterations < 0) ? 0xffffffffffffffffULL : (uint64_t)max_iterations;

    void* args[] = {
        (void*)&offsets,
        (void*)&indices,
        (void*)&weights,
        (void*)&cat,
        (void*)&medium_list,
        (void*)&heavy_list,
        (void*)&counts,
        (void*)&buf0,
        (void*)&buf1,
        (void*)&num_vertices,
        (void*)&epsilon,
        (void*)&eff_max,
        (void*)&initial_centralities,
        (void*)&use_initial,
        (void*)&block_sumsq,
        (void*)&block_delta,
        (void*)&scalars,
        (void*)&final_selector,
        (void*)&out_iterations,
        (void*)&out_converged
    };

    cudaLaunchCooperativeKernel((void*)eigenvector_persistent_kernel, dim3(grid), dim3(BLOCK_SIZE), args, 0, 0);
}

}  



eigenvector_centrality_result_t eigenvector_centrality_mask(const graph32_t& graph,
                                  const double* edge_weights,
                                  double* centralities,
                                  double epsilon,
                                  std::size_t max_iterations,
                                  const double* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    const uint32_t* edge_mask = graph.edge_mask;

    
    size_t cub_bytes = 0;
    cub::DeviceScan::InclusiveSum(nullptr, cub_bytes, (int32_t*)nullptr, (int32_t*)nullptr, num_vertices);

    int32_t medium_capacity = (int32_t)std::min<int64_t>((int64_t)num_vertices, ((int64_t)num_edges + 31) / 32 + 1);
    int32_t heavy_capacity = (int32_t)std::min<int64_t>((int64_t)num_vertices, ((int64_t)num_edges + 1023) / 1024 + 1);

    cache.ensure(num_vertices, num_edges, cub_bytes, medium_capacity, heavy_capacity);

    
    launch_prefilter_csc_masked_to_csc(
        offsets, indices, edge_weights, edge_mask,
        cache.new_offsets, cache.new_indices, cache.new_weights,
        num_vertices,
        cache.cub_temp, cub_bytes);

    
    launch_build_degree_bins(
        cache.new_offsets, num_vertices, num_edges,
        cache.cat, cache.medium_list, cache.heavy_list,
        medium_capacity, heavy_capacity,
        cache.counts);

    
    float* buf0 = cache.buf_f;
    float* buf1 = cache.buf_f + (int64_t)num_vertices;

    double* block_sumsq = cache.block_buf;
    double* block_delta = cache.block_buf + 512;

    bool use_initial = (initial_centralities != nullptr);

    
    launch_eigenvector_centrality_persistent(
        cache.new_offsets, cache.new_indices, cache.new_weights,
        cache.cat, cache.medium_list, cache.heavy_list,
        cache.counts,
        buf0, buf1,
        num_vertices,
        epsilon, (int64_t)max_iterations,
        initial_centralities, use_initial,
        block_sumsq, block_delta,
        cache.scalars, cache.selector,
        cache.d_iterations, cache.d_converged);

    
    launch_select_and_cast_f32_to_f64(buf0, buf1, cache.selector, centralities, num_vertices);

    
    int64_t h_iterations = 0;
    bool h_converged = false;
    cudaDeviceSynchronize();
    cudaMemcpy(&h_iterations, cache.d_iterations, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_converged, cache.d_converged, sizeof(bool), cudaMemcpyDeviceToHost);

    return {static_cast<std::size_t>(h_iterations), h_converged};
}

}  
