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
#include <cstring>
#include <vector>

namespace aai {

namespace {

struct Cache : Cacheable {
    float* buf0 = nullptr;
    float* buf1 = nullptr;
    float* scratch = nullptr;   
    float* h_scratch = nullptr; 
    int64_t buf0_capacity = 0;
    int64_t buf1_capacity = 0;

    void ensure(int32_t num_vertices) {
        if (buf0_capacity < num_vertices) {
            if (buf0) cudaFree(buf0);
            cudaMalloc(&buf0, (size_t)num_vertices * sizeof(float));
            buf0_capacity = num_vertices;
        }
        if (buf1_capacity < num_vertices) {
            if (buf1) cudaFree(buf1);
            cudaMalloc(&buf1, (size_t)num_vertices * sizeof(float));
            buf1_capacity = num_vertices;
        }
        if (!scratch) {
            cudaMalloc(&scratch, 2 * sizeof(float));
        }
        if (!h_scratch) {
            cudaMallocHost(&h_scratch, 2 * sizeof(float));
        }
    }

    ~Cache() override {
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (scratch) cudaFree(scratch);
        if (h_scratch) cudaFreeHost(h_scratch);
    }
};





struct DiffNorm {
    float diff;
    float norm_sq;
};

struct DiffNormSum {
    __device__ __forceinline__ DiffNorm operator()(const DiffNorm& a, const DiffNorm& b) const {
        return {a.diff + b.diff, a.norm_sq + b.norm_sq};
    }
};

template<int BLOCK_SIZE, bool USE_BETAS>
__global__ void katz_iteration_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const float alpha,
    const float beta,
    const float* __restrict__ betas,
    float* __restrict__ diff_out,
    float* __restrict__ norm_sq_out,
    const int32_t num_vertices
) {
    typedef cub::BlockReduce<DiffNorm, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float local_diff = 0.0f;
    float local_norm_sq = 0.0f;

    if (v < num_vertices) {
        int start = offsets[v];
        int end = offsets[v + 1];

        float sum = 0.0f;
        for (int j = start; j < end; j++) {
            sum += x_old[indices[j]];
        }

        float new_val = alpha * sum;
        if constexpr (USE_BETAS) {
            new_val += betas[v];
        } else {
            new_val += beta;
        }

        float old_val = x_old[v];
        x_new[v] = new_val;

        local_diff = fabsf(new_val - old_val);
        local_norm_sq = new_val * new_val;
    }

    DiffNorm local_dn = {local_diff, local_norm_sq};
    DiffNorm block_dn = BlockReduce(temp_storage).Reduce(local_dn, DiffNormSum());

    if (threadIdx.x == 0) {
        if (block_dn.diff > 0.0f)
            atomicAdd(diff_out, block_dn.diff);
        if (block_dn.norm_sq > 0.0f)
            atomicAdd(norm_sq_out, block_dn.norm_sq);
    }
}

template<int BLOCK_SIZE>
__global__ void compute_x2_from_degree_kernel(
    const int32_t* __restrict__ offsets,
    float* __restrict__ x_new,
    const float alpha_beta,
    const float beta,
    float* __restrict__ norm_sq_out,
    const int32_t num_vertices
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int v = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float nsq = 0.0f;

    if (v < num_vertices) {
        float degree = (float)(offsets[v + 1] - offsets[v]);
        float val = beta + alpha_beta * degree;
        x_new[v] = val;
        nsq = val * val;
    }

    float block_sum = BlockReduce(temp_storage).Sum(nsq);
    if (threadIdx.x == 0 && block_sum > 0.0f)
        atomicAdd(norm_sq_out, block_sum);
}

template<int BLOCK_SIZE>
__global__ void fill_const_with_norm_kernel(
    float* __restrict__ x,
    const float val,
    float* __restrict__ norm_sq_out,
    const int32_t n
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float nsq = 0.0f;
    if (i < n) { x[i] = val; nsq = val * val; }
    float block_sum = BlockReduce(temp_storage).Sum(nsq);
    if (threadIdx.x == 0 && block_sum > 0.0f) atomicAdd(norm_sq_out, block_sum);
}

template<int BLOCK_SIZE>
__global__ void fill_betas_with_norm_kernel(
    float* __restrict__ x,
    const float* __restrict__ betas,
    float* __restrict__ norm_sq_out,
    const int32_t n
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float nsq = 0.0f;
    if (i < n) { float val = betas[i]; x[i] = val; nsq = val * val; }
    float block_sum = BlockReduce(temp_storage).Sum(nsq);
    if (threadIdx.x == 0 && block_sum > 0.0f) atomicAdd(norm_sq_out, block_sum);
}

__global__ void scale_kernel(float* __restrict__ x, const float scale, const int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= scale;
}

template<int BLOCK_SIZE>
__global__ void l2_norm_sq_kernel(
    const float* __restrict__ x,
    float* __restrict__ norm_sq_out,
    const int32_t n
) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float val = 0.0f;
    if (i < n) { float xv = x[i]; val = xv * xv; }
    float block_sum = BlockReduce(temp_storage).Sum(val);
    if (threadIdx.x == 0 && block_sum > 0.0f) atomicAdd(norm_sq_out, block_sum);
}





void launch_katz_iteration(
    const int32_t* offsets, const int32_t* indices,
    const float* x_old, float* x_new,
    float alpha, float beta, const float* betas,
    float* diff_out, float* norm_sq_out,
    int32_t num_vertices, bool use_betas,
    cudaStream_t stream
) {
    if (num_vertices == 0) return;
    constexpr int BS = 256;
    int grid = (num_vertices + BS - 1) / BS;

    cudaMemsetAsync(diff_out, 0, sizeof(float), stream);
    cudaMemsetAsync(norm_sq_out, 0, sizeof(float), stream);

    if (use_betas) {
        katz_iteration_kernel<BS, true><<<grid, BS, 0, stream>>>(
            offsets, indices, x_old, x_new, alpha, beta, betas,
            diff_out, norm_sq_out, num_vertices);
    } else {
        katz_iteration_kernel<BS, false><<<grid, BS, 0, stream>>>(
            offsets, indices, x_old, x_new, alpha, beta, betas,
            diff_out, norm_sq_out, num_vertices);
    }
}

void launch_compute_x2_from_degree(
    const int32_t* offsets, float* x_new,
    float alpha_beta, float beta, float* norm_sq_out,
    int32_t num_vertices, cudaStream_t stream
) {
    if (num_vertices == 0) return;
    constexpr int BS = 256;
    cudaMemsetAsync(norm_sq_out, 0, sizeof(float), stream);
    compute_x2_from_degree_kernel<BS><<<(num_vertices + BS - 1) / BS, BS, 0, stream>>>(
        offsets, x_new, alpha_beta, beta, norm_sq_out, num_vertices);
}

void launch_fill_const_with_norm(float* x, float val, float* norm_sq_out, int32_t n, cudaStream_t stream) {
    if (n == 0) return;
    constexpr int BS = 256;
    cudaMemsetAsync(norm_sq_out, 0, sizeof(float), stream);
    fill_const_with_norm_kernel<BS><<<(n + BS - 1) / BS, BS, 0, stream>>>(x, val, norm_sq_out, n);
}

void launch_fill_betas_with_norm(float* x, const float* betas, float* norm_sq_out, int32_t n, cudaStream_t stream) {
    if (n == 0) return;
    constexpr int BS = 256;
    cudaMemsetAsync(norm_sq_out, 0, sizeof(float), stream);
    fill_betas_with_norm_kernel<BS><<<(n + BS - 1) / BS, BS, 0, stream>>>(x, betas, norm_sq_out, n);
}

void launch_scale(float* x, float scale, int32_t n, cudaStream_t stream) {
    if (n == 0) return;
    constexpr int BS = 256;
    scale_kernel<<<(n + BS - 1) / BS, BS, 0, stream>>>(x, scale, n);
}

void launch_l2_norm_sq(const float* x, float* norm_sq_out, int32_t n, cudaStream_t stream) {
    if (n == 0) return;
    constexpr int BS = 256;
    cudaMemsetAsync(norm_sq_out, 0, sizeof(float), stream);
    l2_norm_sq_kernel<BS><<<(n + BS - 1) / BS, BS, 0, stream>>>(x, norm_sq_out, n);
}

}  

katz_centrality_result_t katz_centrality(const graph32_t& graph,
                     float* centralities,
                     float alpha,
                     float beta,
                     const float* betas,
                     float epsilon,
                     std::size_t max_iterations,
                     bool has_initial_guess,
                     bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    bool use_betas = (betas != nullptr);

    cache.ensure(num_vertices);

    float* d_buf0 = cache.buf0;
    float* d_buf1 = cache.buf1;
    float* d_diff = cache.scratch;
    float* d_norm_sq = cache.scratch + 1;
    float* h_scratch = cache.h_scratch;

    cudaStream_t stream = 0;

    bool converged = false;
    size_t iterations = 0;
    int current = 0;
    float last_norm_sq = 0.0f;

    
    size_t x_bytes = (size_t)num_vertices * sizeof(float);
    cudaStreamAttrValue stream_attr;
    memset(&stream_attr, 0, sizeof(stream_attr));
    bool l2_set = false;

    if (x_bytes > 0) {
        stream_attr.accessPolicyWindow.base_ptr = (void*)d_buf0;
        stream_attr.accessPolicyWindow.num_bytes = (x_bytes > (48ULL << 20)) ? (48ULL << 20) : x_bytes;
        stream_attr.accessPolicyWindow.hitRatio = 1.0f;
        stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
        l2_set = true;
    }

    if (has_initial_guess && num_vertices > 0) {
        cudaMemcpyAsync(d_buf0, centralities, x_bytes, cudaMemcpyDeviceToDevice, stream);
        current = 0;
    } else if (!use_betas && max_iterations >= 2 && num_vertices > 0) {
        
        launch_compute_x2_from_degree(
            d_offsets, d_buf0, alpha * beta, beta, d_norm_sq,
            num_vertices, stream);
        current = 0;
        iterations = 2;
        
        cudaMemcpyAsync(h_scratch + 1, d_norm_sq, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        last_norm_sq = h_scratch[1];
    } else if (!has_initial_guess && num_vertices > 0) {
        
        if (use_betas) {
            launch_fill_betas_with_norm(d_buf0, betas, d_norm_sq, num_vertices, stream);
        } else {
            launch_fill_const_with_norm(d_buf0, beta, d_norm_sq, num_vertices, stream);
        }
        current = 0;
        iterations = 1;

        if (max_iterations <= 1) {
            float first_diff;
            if (use_betas) {
                std::vector<float> h_betas(num_vertices);
                cudaMemcpyAsync(h_betas.data(), betas,
                                num_vertices * sizeof(float), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                first_diff = 0.0f;
                for (int32_t i = 0; i < num_vertices; i++)
                    first_diff += fabsf(h_betas[i]);
            } else {
                first_diff = (float)num_vertices * fabsf(beta);
            }
            converged = (first_diff < epsilon);
            cudaMemcpyAsync(h_scratch + 1, d_norm_sq, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            last_norm_sq = h_scratch[1];
            goto finalize;
        }
    }

    
    for (size_t iter = iterations; iter < max_iterations; iter++) {
        float* d_src = (current == 0) ? d_buf0 : d_buf1;
        float* d_dst = (current == 0) ? d_buf1 : d_buf0;

        
        if (l2_set) {
            stream_attr.accessPolicyWindow.base_ptr = (void*)d_src;
            cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
        }

        launch_katz_iteration(d_offsets, d_indices, d_src, d_dst,
                             alpha, beta, betas, d_diff, d_norm_sq,
                             num_vertices, use_betas, stream);

        current = 1 - current;
        iterations = iter + 1;

        
        cudaMemcpyAsync(h_scratch, d_diff, 2 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        last_norm_sq = h_scratch[1];

        if (h_scratch[0] < epsilon) {
            converged = true;
            break;
        }
    }

    finalize:
    
    if (l2_set) {
        stream_attr.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attr);
        cudaCtxResetPersistingL2Cache();
    }

    float* d_result = (current == 0) ? d_buf0 : d_buf1;

    if (normalize && num_vertices > 0) {
        if (has_initial_guess && iterations == 0) {
            launch_l2_norm_sq(d_result, d_norm_sq, num_vertices, stream);
            cudaMemcpyAsync(&last_norm_sq, d_norm_sq, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }

        if (last_norm_sq > 0.0f) {
            float inv_norm = 1.0f / sqrtf(last_norm_sq);
            launch_scale(d_result, inv_norm, num_vertices, stream);
        }
    }

    
    if (num_vertices > 0) {
        cudaMemcpyAsync(centralities, d_result, x_bytes, cudaMemcpyDeviceToDevice, stream);
    }

    return katz_centrality_result_t{iterations, converged};
}

}  
