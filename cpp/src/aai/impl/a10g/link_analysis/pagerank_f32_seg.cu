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
#include <cstring>
#include <algorithm>

namespace aai {

namespace {

static constexpr int32_t CUSTOM_SPMV_THRESHOLD = 20000000;

struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;

    float* spmv_buf = nullptr;
    int64_t spmv_buf_capacity = 0;

    float* out_weight_sums = nullptr;
    int64_t out_weight_sums_capacity = 0;

    float* normalized_weights = nullptr;
    int64_t normalized_weights_capacity = 0;

    float* scalars = nullptr;  

    int32_t* dangling_indices = nullptr;
    int64_t dangling_indices_capacity = 0;

    void* cusparse_buffer = nullptr;
    size_t cusparse_buffer_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cudaMalloc(&scalars, 5 * sizeof(float));
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (spmv_buf) cudaFree(spmv_buf);
        if (out_weight_sums) cudaFree(out_weight_sums);
        if (normalized_weights) cudaFree(normalized_weights);
        if (scalars) cudaFree(scalars);
        if (dangling_indices) cudaFree(dangling_indices);
        if (cusparse_buffer) cudaFree(cusparse_buffer);
    }

    void ensure_spmv_buf(int64_t n) {
        if (spmv_buf_capacity < n) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, n * sizeof(float));
            spmv_buf_capacity = n;
        }
    }

    void ensure_out_weight_sums(int64_t n) {
        if (out_weight_sums_capacity < n) {
            if (out_weight_sums) cudaFree(out_weight_sums);
            cudaMalloc(&out_weight_sums, n * sizeof(float));
            out_weight_sums_capacity = n;
        }
    }

    void ensure_normalized_weights(int64_t n) {
        if (normalized_weights_capacity < n) {
            if (normalized_weights) cudaFree(normalized_weights);
            cudaMalloc(&normalized_weights, n * sizeof(float));
            normalized_weights_capacity = n;
        }
    }

    void ensure_dangling_indices(int64_t n) {
        if (dangling_indices_capacity < n) {
            if (dangling_indices) cudaFree(dangling_indices);
            cudaMalloc(&dangling_indices, n * sizeof(int32_t));
            dangling_indices_capacity = n;
        }
    }

    void ensure_cusparse_buffer(size_t n) {
        if (cusparse_buffer_capacity < n) {
            if (cusparse_buffer) cudaFree(cusparse_buffer);
            cudaMalloc(&cusparse_buffer, n);
            cusparse_buffer_capacity = n;
        }
    }
};





__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    float* __restrict__ out_weight_sums,
    int32_t num_edges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_edges) {
        atomicAdd(&out_weight_sums[indices[i]], edge_weights[i]);
    }
}

__global__ void compute_normalized_weights_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ out_weight_sums,
    float* __restrict__ normalized_weights,
    int32_t num_edges)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_edges) {
        float ow = out_weight_sums[indices[i]];
        normalized_weights[i] = (ow > 0.0f) ? __fdividef(edge_weights[i], ow) : 0.0f;
    }
}

__global__ void init_pageranks_kernel(float* pr, int32_t N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        pr[i] = 1.0f / (float)N;
    }
}

__global__ void count_dangling_kernel(
    const float* __restrict__ out_weight_sums,
    int32_t N, int32_t* __restrict__ d_count)
{
    typedef cub::BlockReduce<int32_t, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t val = (idx < N && out_weight_sums[idx] == 0.0f) ? 1 : 0;
    int32_t sum = BlockReduce(temp).Sum(val);
    if (threadIdx.x == 0 && sum > 0) atomicAdd(d_count, sum);
}

__global__ void build_dangling_indices_kernel(
    const float* __restrict__ out_weight_sums,
    int32_t* __restrict__ dangling_indices,
    int32_t N, int32_t* __restrict__ d_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && out_weight_sums[idx] == 0.0f) {
        int pos = atomicAdd(d_count, 1);
        dangling_indices[pos] = idx;
    }
}





__global__ void dangling_sum_compact_kernel(
    const float* __restrict__ pr,
    const int32_t* __restrict__ dangling_indices,
    int32_t num_dangling,
    float* __restrict__ d_dangling_sum)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < num_dangling) ? pr[dangling_indices[idx]] : 0.0f;
    float sum = BlockReduce(temp).Sum(val);
    if (threadIdx.x == 0 && sum != 0.0f) atomicAdd(d_dangling_sum, sum);
}

__global__ void update_and_diff_kernel(
    float* __restrict__ pr,
    const float* __restrict__ spmv_result,
    const float* __restrict__ d_dangling_sum,
    int32_t N,
    float alpha,
    float* __restrict__ d_diff)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    float dangling_sum = *d_dangling_sum;
    float base = (1.0f - alpha) / (float)N;
    float dangling_contrib = alpha * dangling_sum / (float)N;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_diff = 0.0f;
    if (idx < N) {
        float old_pr = pr[idx];
        float new_pr = base + alpha * spmv_result[idx] + dangling_contrib;
        pr[idx] = new_pr;
        local_diff = fabsf(new_pr - old_pr);
    }

    float block_sum = BlockReduce(temp).Sum(local_diff);
    if (threadIdx.x == 0) {
        atomicAdd(d_diff, block_sum);
    }
}

__global__ void update_and_diff_no_dangling_kernel(
    float* __restrict__ pr,
    const float* __restrict__ spmv_result,
    int32_t N,
    float alpha,
    float base_val,
    float* __restrict__ d_diff)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_diff = 0.0f;
    if (idx < N) {
        float old_pr = pr[idx];
        float new_pr = base_val + alpha * spmv_result[idx];
        pr[idx] = new_pr;
        local_diff = fabsf(new_pr - old_pr);
    }

    float block_sum = BlockReduce(temp).Sum(local_diff);
    if (threadIdx.x == 0) {
        atomicAdd(d_diff, block_sum);
    }
}





template<bool HAS_DANGLING>
__global__ void pagerank_high_degree(
    const int32_t* __restrict__ rowptr,
    const int32_t* __restrict__ colidx,
    const float* __restrict__ vals,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    int32_t start_row, int32_t end_row,
    float alpha, float base_no_d, int32_t N,
    const float* __restrict__ d_dangling_sum,
    float* __restrict__ d_diff)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float s_base;

    int row = start_row + blockIdx.x;
    if (row >= end_row) return;

    if (threadIdx.x == 0) {
        float ds = HAS_DANGLING ? (*d_dangling_sum) : 0.0f;
        s_base = base_no_d + alpha * ds / (float)N;
    }
    __syncthreads();

    float sum = 0.0f;
    int start = rowptr[row];
    int end_idx = rowptr[row + 1];
    for (int j = start + threadIdx.x; j < end_idx; j += blockDim.x) {
        sum += __ldg(&pr_old[colidx[j]]) * __ldg(&vals[j]);
    }

    float block_sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) {
        float new_pr = s_base + alpha * block_sum;
        pr_new[row] = new_pr;
        atomicAdd(d_diff, fabsf(new_pr - pr_old[row]));
    }
}

template<bool HAS_DANGLING>
__global__ void pagerank_medium_degree(
    const int32_t* __restrict__ rowptr,
    const int32_t* __restrict__ colidx,
    const float* __restrict__ vals,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    int32_t start_row, int32_t end_row,
    float alpha, float base_no_d, int32_t N,
    const float* __restrict__ d_dangling_sum,
    float* __restrict__ d_diff)
{
    constexpr int WPB = 8;
    __shared__ float warp_diffs[WPB];
    __shared__ float s_base;

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int row = start_row + blockIdx.x * WPB + warp_id;

    if (threadIdx.x == 0) {
        float ds = HAS_DANGLING ? (*d_dangling_sum) : 0.0f;
        s_base = base_no_d + alpha * ds / (float)N;
    }
    __syncthreads();

    float diff = 0.0f;
    if (row < end_row) {
        float sum = 0.0f;
        for (int j = rowptr[row] + lane; j < rowptr[row + 1]; j += 32) {
            sum += __ldg(&pr_old[colidx[j]]) * __ldg(&vals[j]);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) {
            float new_pr = s_base + alpha * sum;
            pr_new[row] = new_pr;
            diff = fabsf(new_pr - pr_old[row]);
        }
    }
    if (lane == 0) warp_diffs[warp_id] = diff;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0.0f;
        #pragma unroll
        for (int w = 0; w < WPB; w++) t += warp_diffs[w];
        if (t > 0.0f) atomicAdd(d_diff, t);
    }
}

template<bool HAS_DANGLING>
__global__ void pagerank_low_degree(
    const int32_t* __restrict__ rowptr,
    const int32_t* __restrict__ colidx,
    const float* __restrict__ vals,
    const float* __restrict__ pr_old,
    float* __restrict__ pr_new,
    int32_t start_row, int32_t end_row,
    float alpha, float base_no_d, int32_t N,
    const float* __restrict__ d_dangling_sum,
    float* __restrict__ d_diff)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float s_base;

    if (threadIdx.x == 0) {
        float ds = HAS_DANGLING ? (*d_dangling_sum) : 0.0f;
        s_base = base_no_d + alpha * ds / (float)N;
    }
    __syncthreads();

    int row = start_row + blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0.0f;
    if (row < end_row) {
        float sum = 0.0f;
        for (int j = rowptr[row]; j < rowptr[row + 1]; j++) {
            sum += __ldg(&pr_old[colidx[j]]) * __ldg(&vals[j]);
        }
        float new_pr = s_base + alpha * sum;
        pr_new[row] = new_pr;
        diff = fabsf(new_pr - pr_old[row]);
    }
    float bs = BlockReduce(temp).Sum(diff);
    if (threadIdx.x == 0 && bs > 0.0f) atomicAdd(d_diff, bs);
}

template<bool HAS_DANGLING>
__global__ void pagerank_zero_degree(
    const float* __restrict__ pr_old, float* __restrict__ pr_new,
    int32_t start_row, int32_t end_row,
    float alpha, float base_no_d, int32_t N,
    const float* __restrict__ d_dangling_sum,
    float* __restrict__ d_diff)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float s_base;
    if (threadIdx.x == 0) {
        float ds = HAS_DANGLING ? (*d_dangling_sum) : 0.0f;
        s_base = base_no_d + alpha * ds / (float)N;
    }
    __syncthreads();
    int row = start_row + blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0.0f;
    if (row < end_row) {
        pr_new[row] = s_base;
        diff = fabsf(s_base - pr_old[row]);
    }
    float bs = BlockReduce(temp).Sum(diff);
    if (threadIdx.x == 0 && bs > 0.0f) atomicAdd(d_diff, bs);
}





void launch_compute_out_weight_sums(const int32_t* indices, const float* edge_weights,
    float* out_weight_sums, int32_t num_edges, cudaStream_t stream)
{
    if (num_edges == 0) return;
    compute_out_weight_sums_kernel<<<(num_edges+255)/256, 256, 0, stream>>>(
        indices, edge_weights, out_weight_sums, num_edges);
}

void launch_compute_normalized_weights(const int32_t* indices, const float* edge_weights,
    const float* out_weight_sums, float* normalized_weights, int32_t num_edges, cudaStream_t stream)
{
    if (num_edges == 0) return;
    compute_normalized_weights_kernel<<<(num_edges+255)/256, 256, 0, stream>>>(
        indices, edge_weights, out_weight_sums, normalized_weights, num_edges);
}

void launch_init_pageranks(float* pr, int32_t N, cudaStream_t stream)
{
    if (N == 0) return;
    init_pageranks_kernel<<<(N+255)/256, 256, 0, stream>>>(pr, N);
}

void launch_count_dangling(const float* out_w, int32_t N, int32_t* d_count, cudaStream_t stream)
{
    if (N == 0) return;
    count_dangling_kernel<<<(N+255)/256, 256, 0, stream>>>(out_w, N, d_count);
}

void launch_build_dangling_indices(const float* out_w, int32_t* d_idx,
    int32_t N, int32_t* d_count, cudaStream_t stream)
{
    if (N == 0) return;
    build_dangling_indices_kernel<<<(N+255)/256, 256, 0, stream>>>(out_w, d_idx, N, d_count);
}

void launch_dangling_sum_compact(const float* pr, const int32_t* d_idx,
    int32_t num_dangling, float* d_sum, cudaStream_t stream)
{
    if (num_dangling == 0) return;
    dangling_sum_compact_kernel<<<(num_dangling+255)/256, 256, 0, stream>>>(
        pr, d_idx, num_dangling, d_sum);
}

void launch_update_and_diff(float* pr, const float* spmv_result, const float* d_dangling_sum,
    int32_t N, float alpha, float* d_diff, cudaStream_t stream)
{
    if (N == 0) return;
    update_and_diff_kernel<<<(N+255)/256, 256, 0, stream>>>(
        pr, spmv_result, d_dangling_sum, N, alpha, d_diff);
}

void launch_update_and_diff_no_dangling(float* pr, const float* spmv_result,
    int32_t N, float alpha, float base_val, float* d_diff, cudaStream_t stream)
{
    if (N == 0) return;
    update_and_diff_no_dangling_kernel<<<(N+255)/256, 256, 0, stream>>>(
        pr, spmv_result, N, alpha, base_val, d_diff);
}

void launch_pagerank_segments_custom(
    const int32_t* rowptr, const int32_t* colidx, const float* vals,
    const float* pr_old, float* pr_new,
    const int32_t* seg, float alpha, float base_no_d, int32_t N,
    const float* d_dangling_sum, bool has_dangling, float* d_diff, cudaStream_t stream)
{
    int32_t s0=seg[0], s1=seg[1], s2=seg[2], s3=seg[3], s4=seg[4];

    #define LAUNCH_SEG(KERNEL, START, END, GRID, ...) \
        if ((END) > (START)) { \
            if (has_dangling) { \
                KERNEL<true><<<GRID, 256, 0, stream>>>( \
                    __VA_ARGS__, alpha, base_no_d, N, d_dangling_sum, d_diff); \
            } else { \
                KERNEL<false><<<GRID, 256, 0, stream>>>( \
                    __VA_ARGS__, alpha, base_no_d, N, nullptr, d_diff); \
            } \
        }

    LAUNCH_SEG(pagerank_high_degree, s0, s1, s1-s0,
        rowptr, colidx, vals, pr_old, pr_new, s0, s1)
    LAUNCH_SEG(pagerank_medium_degree, s1, s2, ((s2-s1)+7)/8,
        rowptr, colidx, vals, pr_old, pr_new, s1, s2)
    LAUNCH_SEG(pagerank_low_degree, s2, s3, ((s3-s2)+255)/256,
        rowptr, colidx, vals, pr_old, pr_new, s2, s3)

    if (s4 > s3) {
        if (has_dangling) {
            pagerank_zero_degree<true><<<((s4-s3)+255)/256, 256, 0, stream>>>(
                pr_old, pr_new, s3, s4, alpha, base_no_d, N, d_dangling_sum, d_diff);
        } else {
            pagerank_zero_degree<false><<<((s4-s3)+255)/256, 256, 0, stream>>>(
                pr_old, pr_new, s3, s4, alpha, base_no_d, N, nullptr, d_diff);
        }
    }

    #undef LAUNCH_SEG
}

}  

PageRankResult pagerank_seg(const graph32_t& graph,
                            const float* edge_weights,
                            float* pageranks,
                            const float* precomputed_vertex_out_weight_sums,
                            float alpha,
                            float epsilon,
                            std::size_t max_iterations,
                            const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t num_vertices = graph.number_of_vertices;
    const int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg_offsets[5] = {seg[0], seg[1], seg[2], seg[3], seg[4]};

    cudaStream_t stream = 0;

    bool use_custom_spmv = (num_vertices >= CUSTOM_SPMV_THRESHOLD);

    
    cache.ensure_spmv_buf(num_vertices);
    int64_t nw_size = (num_edges > 0) ? (int64_t)num_edges : 1;
    cache.ensure_normalized_weights(nw_size);

    float* d_spmv_or_alt = cache.spmv_buf;
    float* d_norm_w = cache.normalized_weights;

    float* d_scalars = cache.scalars;
    float* d_dangling = d_scalars;
    float* d_diff = d_scalars + 1;
    float* d_spmv_alpha = d_scalars + 2;
    float* d_spmv_beta = d_scalars + 3;
    int32_t* d_dangling_count = (int32_t*)(d_scalars + 4);

    float h_scalars[5] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    cudaMemcpyAsync(d_scalars, h_scalars, 5 * sizeof(float), cudaMemcpyHostToDevice, stream);

    
    cache.ensure_out_weight_sums(num_vertices);
    cudaMemsetAsync(cache.out_weight_sums, 0, num_vertices * sizeof(float), stream);
    launch_compute_out_weight_sums(d_indices, edge_weights, cache.out_weight_sums, num_edges, stream);
    const float* d_out_w = cache.out_weight_sums;

    
    launch_compute_normalized_weights(d_indices, edge_weights, d_out_w, d_norm_w, num_edges, stream);

    
    cudaMemsetAsync(d_dangling_count, 0, sizeof(int32_t), stream);
    launch_count_dangling(d_out_w, num_vertices, d_dangling_count, stream);

    int32_t h_dangling_count = 0;
    cudaMemcpy(&h_dangling_count, d_dangling_count, sizeof(int32_t), cudaMemcpyDeviceToHost);

    bool has_dangling = (h_dangling_count > 0);
    int32_t* d_dangling_indices = nullptr;

    if (has_dangling) {
        cache.ensure_dangling_indices(h_dangling_count);
        d_dangling_indices = cache.dangling_indices;
        cudaMemsetAsync(d_dangling_count, 0, sizeof(int32_t), stream);
        launch_build_dangling_indices(d_out_w, d_dangling_indices, num_vertices, d_dangling_count, stream);
    }

    
    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(pageranks, initial_pageranks,
                       num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    } else {
        launch_init_pageranks(pageranks, num_vertices, stream);
    }

    
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;

    if (!use_custom_spmv && num_edges > 0 && num_vertices > 0) {
        cusparseSetStream(cache.cusparse_handle, stream);
        cusparseCreateCsr(&matA,
            num_vertices, num_vertices, num_edges,
            (void*)d_offsets, (void*)d_indices, (void*)d_norm_w,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cusparseCreateDnVec(&vecX, num_vertices, pageranks, CUDA_R_32F);
        cusparseCreateDnVec(&vecY, num_vertices, d_spmv_or_alt, CUDA_R_32F);

        float h_one = 1.0f, h_zero = 0.0f;
        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_HOST);

        size_t cusparse_buffer_size = 0;
        cusparseSpMV_bufferSize(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_one, matA, vecX, &h_zero, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &cusparse_buffer_size);

        if (cusparse_buffer_size > 0) {
            cache.ensure_cusparse_buffer(cusparse_buffer_size);
        }

        cusparseSpMV_preprocess(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &h_one, matA, vecX, &h_zero, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.cusparse_buffer);

        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
    }

    
    bool converged = false;
    size_t iteration = 0;
    float h_diff = 0.0f;
    float base_val = (1.0f - alpha) / (float)num_vertices;
    int batch_size = 1;

    if (use_custom_spmv) {
        
        float* pr_bufs[2] = {pageranks, d_spmv_or_alt};
        int current = 0;

        while (iteration < max_iterations) {
            size_t batch_end = iteration + batch_size;
            if (batch_end > max_iterations) batch_end = max_iterations;

            for (; iteration < batch_end; iteration++) {
                int next = 1 - current;

                if (has_dangling) {
                    cudaMemsetAsync(d_dangling, 0, sizeof(float), stream);
                    launch_dangling_sum_compact(pr_bufs[current], d_dangling_indices,
                        h_dangling_count, d_dangling, stream);
                }

                cudaMemsetAsync(d_diff, 0, sizeof(float), stream);

                launch_pagerank_segments_custom(
                    d_offsets, d_indices, d_norm_w,
                    pr_bufs[current], pr_bufs[next],
                    seg_offsets, alpha, base_val, num_vertices,
                    d_dangling, has_dangling, d_diff, stream);

                current = next;
            }

            cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_diff < epsilon) { converged = true; break; }
        }

        
        if (pr_bufs[current] != pageranks) {
            cudaMemcpyAsync(pageranks, pr_bufs[current],
                num_vertices * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaStreamSynchronize(stream);
        }
    } else {
        
        while (iteration < max_iterations) {
            size_t batch_end = iteration + batch_size;
            if (batch_end > max_iterations) batch_end = max_iterations;

            for (; iteration < batch_end; iteration++) {
                if (has_dangling) {
                    cudaMemsetAsync(d_dangling, 0, sizeof(float), stream);
                    launch_dangling_sum_compact(pageranks, d_dangling_indices,
                        h_dangling_count, d_dangling, stream);
                }

                if (num_edges > 0) {
                    cusparseSpMV(cache.cusparse_handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        d_spmv_alpha, matA, vecX, d_spmv_beta, vecY,
                        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.cusparse_buffer);
                } else {
                    cudaMemsetAsync(d_spmv_or_alt, 0, num_vertices * sizeof(float), stream);
                }

                cudaMemsetAsync(d_diff, 0, sizeof(float), stream);
                if (has_dangling) {
                    launch_update_and_diff(pageranks, d_spmv_or_alt, d_dangling,
                        num_vertices, alpha, d_diff, stream);
                } else {
                    launch_update_and_diff_no_dangling(pageranks, d_spmv_or_alt,
                        num_vertices, alpha, base_val, d_diff, stream);
                }
            }

            cudaMemcpy(&h_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
            if (h_diff < epsilon) { converged = true; break; }
        }
    }

    
    if (matA) cusparseDestroySpMat(matA);
    if (vecX) cusparseDestroyDnVec(vecX);
    if (vecY) cusparseDestroyDnVec(vecY);

    return {iteration, converged};
}

}  
