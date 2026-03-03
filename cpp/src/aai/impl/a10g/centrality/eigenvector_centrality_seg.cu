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
#include <cmath>
#include <algorithm>

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)


__device__ __forceinline__ float block_reduce_sum(float val) {
    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    __shared__ float warp_sums[WARPS_PER_BLOCK];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    if (lane == 0) warp_sums[warp] = val;
    __syncthreads();

    
    val = (threadIdx.x < WARPS_PER_BLOCK) ? warp_sums[threadIdx.x] : 0.0f;
    if (warp == 0) {
        #pragma unroll
        for (int offset = WARPS_PER_BLOCK/2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val; 
}

__global__ void init_uniform_kernel(float* __restrict__ x, float val, int32_t n)
{
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i < n) x[i] = val;
}

__global__ __launch_bounds__(BLOCK_SIZE)
void spmv_unified(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ x_old, float* __restrict__ x_new,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg4,
    int32_t high_blocks, int32_t mid_blocks)
{
    if (blockIdx.x < high_blocks) {
        int v = blockIdx.x + seg0; if (v >= seg1) return;
        int rs = offsets[v], re = offsets[v+1];
        float s = 0.0f;
        for (int i = threadIdx.x; i < re-rs; i += BLOCK_SIZE)
            s += __ldg(&x_old[__ldg(&indices[rs+i])]);
        s = block_reduce_sum(s);
        if (threadIdx.x == 0) x_new[v] = s + __ldg(&x_old[v]);
    } else if (blockIdx.x < high_blocks + mid_blocks) {
        int bo = blockIdx.x - high_blocks;
        int wi = threadIdx.x/32, ln = threadIdx.x&31;
        int v = bo*WARPS_PER_BLOCK+wi+seg1; if (v >= seg2) return;
        int rs = __ldg(&offsets[v]), re = __ldg(&offsets[v+1]);
        float s = 0.0f;
        for (int i = ln; i < re-rs; i += 32)
            s += __ldg(&x_old[__ldg(&indices[rs+i])]);
        for (int o=16;o>0;o>>=1) s+=__shfl_down_sync(0xFFFFFFFF,s,o);
        if (ln==0) x_new[v] = s + __ldg(&x_old[v]);
    } else {
        int bo = blockIdx.x-high_blocks-mid_blocks;
        int v = bo*BLOCK_SIZE+threadIdx.x+seg2; if (v >= seg4) return;
        int rs = __ldg(&offsets[v]), re = __ldg(&offsets[v+1]);
        float s = __ldg(&x_old[v]);
        for (int i=rs;i<re;i++) s+=__ldg(&x_old[__ldg(&indices[i])]);
        x_new[v] = s;
    }
}

__global__ __launch_bounds__(BLOCK_SIZE)
void compute_l2_sq_kernel(const float* __restrict__ x, float* __restrict__ result,
    float* __restrict__ partials, unsigned int* __restrict__ counter, int32_t n) {
    float s=0; for(int i=blockIdx.x*BLOCK_SIZE+threadIdx.x;i<n;i+=gridDim.x*BLOCK_SIZE){float v=x[i];s+=v*v;}
    float bs=block_reduce_sum(s); if(threadIdx.x==0)partials[blockIdx.x]=bs;
    __threadfence(); __shared__ bool al;
    if(threadIdx.x==0){al=(atomicAdd(counter,1)==gridDim.x-1);} __syncthreads();
    if(al){float fs=0;for(int i=threadIdx.x;i<gridDim.x;i+=BLOCK_SIZE)fs+=partials[i];
    fs=block_reduce_sum(fs);if(threadIdx.x==0){result[0]=fs;*counter=0;}}
}

__global__ __launch_bounds__(BLOCK_SIZE)
void normalize_diff_kernel(float* __restrict__ xn, const float* __restrict__ xo,
    const float* __restrict__ l2p, float* __restrict__ rd,
    float* __restrict__ partials, unsigned int* __restrict__ counter, int32_t n) {
    float inv=(*l2p>0)?rsqrtf(*l2p):0; float d=0;
    for(int i=blockIdx.x*BLOCK_SIZE+threadIdx.x;i<n;i+=gridDim.x*BLOCK_SIZE){
    float nv=xn[i]*inv;xn[i]=nv;d+=fabsf(nv-xo[i]);}
    float bd=block_reduce_sum(d);if(threadIdx.x==0)partials[blockIdx.x]=bd;
    __threadfence();__shared__ bool al;
    if(threadIdx.x==0){al=(atomicAdd(counter,1)==gridDim.x-1);}__syncthreads();
    if(al){float fd=0;for(int i=threadIdx.x;i<gridDim.x;i+=BLOCK_SIZE)fd+=partials[i];
    fd=block_reduce_sum(fd);if(threadIdx.x==0){rd[0]=fd;*counter=0;}}
}

struct Cache : Cacheable {
    float* x_buf = nullptr;
    int64_t x_buf_capacity = 0;

    float* partials = nullptr;
    int64_t partials_capacity = 0;

    float* l2_sq = nullptr;
    bool l2_sq_allocated = false;

    float* diff = nullptr;
    bool diff_allocated = false;

    unsigned int* counter = nullptr;
    bool counter_allocated = false;

    int reduce_grid = 0;

    void ensure(int32_t n, int reduce_grid_size) {
        if (x_buf_capacity < n) {
            if (x_buf) cudaFree(x_buf);
            cudaMalloc(&x_buf, static_cast<int64_t>(n) * sizeof(float));
            x_buf_capacity = n;
        }
        if (partials_capacity < reduce_grid_size) {
            if (partials) cudaFree(partials);
            cudaMalloc(&partials, static_cast<int64_t>(reduce_grid_size) * sizeof(float));
            partials_capacity = reduce_grid_size;
        }
        if (!l2_sq_allocated) {
            cudaMalloc(&l2_sq, sizeof(float));
            l2_sq_allocated = true;
        }
        if (!diff_allocated) {
            cudaMalloc(&diff, sizeof(float));
            diff_allocated = true;
        }
        if (!counter_allocated) {
            cudaMalloc(&counter, sizeof(unsigned int));
            counter_allocated = true;
        }
        reduce_grid = reduce_grid_size;
    }

    ~Cache() override {
        if (x_buf) cudaFree(x_buf);
        if (partials) cudaFree(partials);
        if (l2_sq) cudaFree(l2_sq);
        if (diff) cudaFree(diff);
        if (counter) cudaFree(counter);
    }
};

}  

eigenvector_centrality_result_t eigenvector_centrality_seg(const graph32_t& graph,
                                float* centralities,
                                float epsilon,
                                std::size_t max_iterations,
                                const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n = graph.number_of_vertices;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg4 = seg[4];

    int32_t high_blocks = seg1 - seg0;
    int32_t mid_blocks = ((seg2 - seg1) + (BLOCK_SIZE / 32) - 1) / (BLOCK_SIZE / 32);
    int32_t low_zero_blocks = (seg4 - seg2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int32_t total_spmv_blocks = high_blocks + mid_blocks + low_zero_blocks;

    int reduce_grid = 320;

    cache.ensure(n, reduce_grid);

    float* x_a = centralities;
    float* x_b = cache.x_buf;
    float* d_partials = cache.partials;
    float* d_l2_sq = cache.l2_sq;
    float* d_diff = cache.diff;
    unsigned int* d_counter = cache.counter;

    
    if (initial_centralities != nullptr) {
        cudaMemcpyAsync(x_a, initial_centralities,
                       static_cast<int64_t>(n) * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        float init_val = 1.0f / sqrtf(static_cast<float>(n));
        init_uniform_kernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(x_a, init_val, n);
    }

    float threshold = static_cast<float>(n) * epsilon;
    bool converged = false;
    std::size_t iter = 0;

    cudaMemsetAsync(d_counter, 0, sizeof(unsigned int));
    float* x_cur = x_a;
    float* x_nxt = x_b;
    const int CHECK_INTERVAL = 10;

    while (iter < max_iterations) {
        std::size_t remaining = max_iterations - iter;
        int batch = static_cast<int>(std::min(remaining, static_cast<std::size_t>(CHECK_INTERVAL)));

        for (int b = 0; b < batch; b++) {
            if (total_spmv_blocks > 0)
                spmv_unified<<<total_spmv_blocks, BLOCK_SIZE>>>(d_offsets, d_indices, x_cur, x_nxt,
                    seg0, seg1, seg2, seg4,
                    high_blocks, mid_blocks);
            compute_l2_sq_kernel<<<reduce_grid, BLOCK_SIZE>>>(x_nxt, d_l2_sq, d_partials, d_counter, n);
            normalize_diff_kernel<<<reduce_grid, BLOCK_SIZE>>>(x_nxt, x_cur, d_l2_sq, d_diff,
                d_partials, d_counter, n);
            std::swap(x_cur, x_nxt);
            iter++;
        }

        float h_diff;
        cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        if (h_diff < threshold) { converged = true; break; }
    }

    if (x_cur != x_a) {
        cudaMemcpy(x_a, x_cur, static_cast<int64_t>(n) * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return {iter, converged};
}

}  
