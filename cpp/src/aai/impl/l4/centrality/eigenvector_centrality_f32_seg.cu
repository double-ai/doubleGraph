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
#include <cstdint>
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* x_buf = nullptr;
    float* y_buf = nullptr;
    float* sc_buf = nullptr;
    int64_t x_capacity = 0;
    int64_t y_capacity = 0;
    bool sc_allocated = false;

    void ensure(int32_t nv) {
        if (x_capacity < nv) {
            if (x_buf) cudaFree(x_buf);
            cudaMalloc(&x_buf, nv * sizeof(float));
            x_capacity = nv;
        }
        if (y_capacity < nv) {
            if (y_buf) cudaFree(y_buf);
            cudaMalloc(&y_buf, nv * sizeof(float));
            y_capacity = nv;
        }
        if (!sc_allocated) {
            cudaMalloc(&sc_buf, 2 * sizeof(float));
            sc_allocated = true;
        }
    }

    ~Cache() override {
        if (x_buf) cudaFree(x_buf);
        if (y_buf) cudaFree(y_buf);
        if (sc_buf) cudaFree(sc_buf);
    }
};

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


__global__ void spmv_high_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ norm_sq,
    int seg_start, int seg_end
) {
    __shared__ float smem[8];

    int v = seg_start + blockIdx.x;
    if (v >= seg_end) return;

    int row_start = offsets[v];
    int row_end = offsets[v + 1];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    float sum = 0.0f;
    for (int e = row_start + threadIdx.x; e < row_end; e += blockDim.x) {
        sum += weights[e] * __ldg(&x[indices[e]]);
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) smem[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < 8) ? smem[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) {
            val += x[v];
            y[v] = val;
            atomicAdd(norm_sq, val * val);
        }
    }
}


__global__ void spmv_mid_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ norm_sq,
    int seg_start, int seg_end
) {
    __shared__ float warp_norms[8];

    int global_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int v = seg_start + global_warp;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    float my_sq = 0.0f;

    if (v < seg_end) {
        int row_start = offsets[v];
        int row_end = offsets[v + 1];

        float sum = 0.0f;
        for (int e = row_start + lane; e < row_end; e += 32) {
            sum += weights[e] * __ldg(&x[indices[e]]);
        }

        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            sum += x[v];
            y[v] = sum;
            my_sq = sum * sum;
        }
    }

    if (lane == 0) warp_norms[warp_id] = my_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < 8) ? warp_norms[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0 && val != 0.0f) atomicAdd(norm_sq, val);
    }
}


__global__ void spmv_low_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ x,
    float* __restrict__ y,
    float* __restrict__ norm_sq,
    int seg_start, int seg_end
) {
    __shared__ float warp_norms[8];

    int v = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    float my_sq = 0.0f;

    if (v < seg_end) {
        int row_start = offsets[v];
        int row_end = offsets[v + 1];

        float sum = 0.0f;
        for (int e = row_start; e < row_end; e++) {
            sum += weights[e] * __ldg(&x[indices[e]]);
        }
        sum += x[v];
        y[v] = sum;
        my_sq = sum * sum;
    }

    my_sq = warp_reduce_sum(my_sq);
    if (lane == 0) warp_norms[warp_id] = my_sq;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < 8) ? warp_norms[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0 && val != 0.0f) atomicAdd(norm_sq, val);
    }
}


__global__ void normalize_diff_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ norm_sq,
    float* __restrict__ diff,
    int n
) {
    __shared__ float warp_sums[8];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    float inv_norm = rsqrtf(*norm_sq);

    float local_diff = 0.0f;
    for (int i = tid; i < n; i += stride) {
        float yi = y[i] * inv_norm;
        y[i] = yi;
        local_diff += fabsf(yi - x[i]);
    }

    local_diff = warp_reduce_sum(local_diff);
    if (lane == 0) warp_sums[warp_id] = local_diff;
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < 8) ? warp_sums[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane == 0) atomicAdd(diff, val);
    }
}

__global__ void init_uniform_kernel(float* __restrict__ x, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] = val;
}

}  

eigenvector_centrality_result_t eigenvector_centrality_seg(const graph32_t& graph,
                                const float* edge_weights,
                                float* centralities,
                                float epsilon,
                                std::size_t max_iterations,
                                const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t nv = graph.number_of_vertices;
    const int32_t* d_off = graph.offsets;
    const int32_t* d_ind = graph.indices;
    const float* d_wt = edge_weights;
    cudaStream_t stream = 0;

    const auto& seg = graph.segment_offsets.value();
    int s0 = seg[0], s1 = seg[1], s2 = seg[2], s3 = seg[3], s4 = seg[4];

    cache.ensure(nv);

    float* d_x = cache.x_buf;
    float* d_y = cache.y_buf;
    float* d_sc = cache.sc_buf;

    
    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(d_x, initial_centralities,
                       nv * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        float init_val = 1.0f / nv;
        int block = 256;
        int grid = (nv + block - 1) / block;
        init_uniform_kernel<<<grid, block, 0, stream>>>(d_x, init_val, nv);
    }

    const float threshold = nv * epsilon;
    bool converged = false;
    std::size_t iter = 0;

    for (; iter < max_iterations; ++iter) {
        
        cudaMemsetAsync(d_sc, 0, 2 * sizeof(float), stream);

        
        {
            int nvs = s1 - s0;
            if (nvs > 0) {
                spmv_high_kernel<<<nvs, 256, 0, stream>>>(d_off, d_ind, d_wt, d_x, d_y, &d_sc[0], s0, s1);
            }
        }
        {
            int nvs = s2 - s1;
            if (nvs > 0) {
                int grid = (nvs + 7) / 8;
                spmv_mid_kernel<<<grid, 256, 0, stream>>>(d_off, d_ind, d_wt, d_x, d_y, &d_sc[0], s1, s2);
            }
        }
        {
            int nvs = s4 - s2;
            if (nvs > 0) {
                int grid = (nvs + 255) / 256;
                spmv_low_kernel<<<grid, 256, 0, stream>>>(d_off, d_ind, d_wt, d_x, d_y, &d_sc[0], s2, s4);
            }
        }

        
        {
            int block = 256;
            int grid = (nv + block - 1) / block;
            if (grid > 512) grid = 512;
            normalize_diff_kernel<<<grid, block, 0, stream>>>(d_y, d_x, &d_sc[0], &d_sc[1], nv);
        }

        
        std::swap(d_x, d_y);

        
        float h_diff;
        cudaMemcpy(&h_diff, &d_sc[1], sizeof(float), cudaMemcpyDeviceToHost);

        if (h_diff < threshold) {
            converged = true;
            ++iter;
            break;
        }
    }

    
    cudaMemcpy(centralities, d_x, nv * sizeof(float), cudaMemcpyDeviceToDevice);

    return {iter, converged};
}

}  
