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

namespace aai {

namespace {

struct Cache : Cacheable {
    
    double* out_ws = nullptr;
    double* inv_ow = nullptr;
    float* pr_norm_f32 = nullptr;
    float* spmv_f32 = nullptr;
    double* pr_a = nullptr;
    double* pr_b = nullptr;
    int64_t out_ws_cap = 0;
    int64_t inv_ow_cap = 0;
    int64_t pr_norm_f32_cap = 0;
    int64_t spmv_f32_cap = 0;
    int64_t pr_a_cap = 0;
    int64_t pr_b_cap = 0;

    
    float* masked_fw = nullptr;
    int64_t masked_fw_cap = 0;

    
    double* dsum_a = nullptr;
    double* dsum_b = nullptr;
    double* diff = nullptr;

    Cache() {
        cudaMalloc(&dsum_a, sizeof(double));
        cudaMalloc(&dsum_b, sizeof(double));
        cudaMalloc(&diff, sizeof(double));
    }

    ~Cache() override {
        cudaFree(out_ws);
        cudaFree(inv_ow);
        cudaFree(pr_norm_f32);
        cudaFree(spmv_f32);
        cudaFree(pr_a);
        cudaFree(pr_b);
        cudaFree(masked_fw);
        cudaFree(dsum_a);
        cudaFree(dsum_b);
        cudaFree(diff);
    }

    void ensure(int32_t nv, int32_t ne) {
        if (out_ws_cap < nv) {
            cudaFree(out_ws);
            cudaMalloc(&out_ws, nv * sizeof(double));
            out_ws_cap = nv;
        }
        if (inv_ow_cap < nv) {
            cudaFree(inv_ow);
            cudaMalloc(&inv_ow, nv * sizeof(double));
            inv_ow_cap = nv;
        }
        if (pr_norm_f32_cap < nv) {
            cudaFree(pr_norm_f32);
            cudaMalloc(&pr_norm_f32, nv * sizeof(float));
            pr_norm_f32_cap = nv;
        }
        if (spmv_f32_cap < nv) {
            cudaFree(spmv_f32);
            cudaMalloc(&spmv_f32, nv * sizeof(float));
            spmv_f32_cap = nv;
        }
        if (pr_a_cap < nv) {
            cudaFree(pr_a);
            cudaMalloc(&pr_a, nv * sizeof(double));
            pr_a_cap = nv;
        }
        if (pr_b_cap < nv) {
            cudaFree(pr_b);
            cudaMalloc(&pr_b, nv * sizeof(double));
            pr_b_cap = nv;
        }
        if (masked_fw_cap < ne) {
            cudaFree(masked_fw);
            cudaMalloc(&masked_fw, ne * sizeof(float));
            masked_fw_cap = ne;
        }
    }
};





__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ out_weight_sums,
    int32_t num_edges)
{
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    uint32_t word = edge_mask[e >> 5];
    if ((word >> (e & 31)) & 1u)
        atomicAdd(&out_weight_sums[indices[e]], edge_weights[e]);
}

__global__ void compute_inv_out_weight_kernel(
    const double* __restrict__ out_weight_sums,
    double* __restrict__ inv_out_weight,
    int32_t num_vertices)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    double w = out_weight_sums[v];
    inv_out_weight[v] = (w != 0.0) ? (1.0 / w) : 0.0;
}


__global__ void prepare_masked_float_weights_kernel(
    const double* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    float* __restrict__ masked_fweights,
    int32_t num_edges)
{
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    uint32_t word = edge_mask[e >> 5];
    masked_fweights[e] = ((word >> (e & 31)) & 1u) ? __double2float_rn(edge_weights[e]) : 0.0f;
}

__global__ void init_pr_kernel(double* __restrict__ pr, double val, int32_t n) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    pr[v] = val;
}





__global__ void pr_norm_f32_dangling_kernel(
    const double* __restrict__ pr,
    const double* __restrict__ inv_out_weight,
    float* __restrict__ pr_norm_f32,
    double* __restrict__ dangling_sum,
    int32_t num_vertices)
{
    __shared__ double warp_sums[32];
    double local_dangling = 0.0;
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;

    for (int32_t v = tid; v < num_vertices; v += stride) {
        double p = pr[v];
        double iw = inv_out_weight[v];
        pr_norm_f32[v] = __double2float_rn(p * iw);
        if (iw == 0.0) local_dangling += p;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_dangling += __shfl_down_sync(0xffffffff, local_dangling, offset);

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = local_dangling;
    __syncthreads();

    int num_warps = blockDim.x >> 5;
    if (warp_id == 0) {
        local_dangling = (lane < num_warps) ? warp_sums[lane] : 0.0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            local_dangling += __shfl_down_sync(0xffffffff, local_dangling, offset);
        if (lane == 0) atomicAdd(dangling_sum, local_dangling);
    }
}







__global__ void spmv_f32_block_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ masked_fweights,
    const float* __restrict__ pr_norm_f32,
    float* __restrict__ spmv_result,
    int32_t seg_start, int32_t seg_end)
{
    __shared__ float warp_sums[32];
    int32_t v = seg_start + blockIdx.x;
    if (v >= seg_end) return;

    int32_t row_start = offsets[v];
    int32_t row_end = offsets[v + 1];

    float sum = 0.0f;
    for (int32_t e = row_start + threadIdx.x; e < row_end; e += blockDim.x) {
        sum += pr_norm_f32[indices[e]] * masked_fweights[e];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp_id] = sum;
    __syncthreads();

    int num_warps = blockDim.x >> 5;
    if (warp_id == 0) {
        sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) spmv_result[v] = sum;
    }
}


__global__ void spmv_f32_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ masked_fweights,
    const float* __restrict__ pr_norm_f32,
    float* __restrict__ spmv_result,
    int32_t seg_start, int32_t seg_end)
{
    int32_t global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t warp_id = global_tid >> 5;
    int32_t lane = global_tid & 31;
    int32_t v = seg_start + warp_id;
    if (v >= seg_end) return;

    int32_t row_start = offsets[v];
    int32_t row_end = offsets[v + 1];

    float sum = 0.0f;
    for (int32_t e = row_start + lane; e < row_end; e += 32) {
        sum += pr_norm_f32[indices[e]] * masked_fweights[e];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (lane == 0) spmv_result[v] = sum;
}


__global__ void spmv_f32_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ masked_fweights,
    const float* __restrict__ pr_norm_f32,
    float* __restrict__ spmv_result,
    int32_t seg_start, int32_t seg_end)
{
    int32_t v = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= seg_end) return;

    int32_t row_start = offsets[v];
    int32_t row_end = offsets[v + 1];

    float sum = 0.0f;
    for (int32_t e = row_start; e < row_end; e++) {
        sum += pr_norm_f32[indices[e]] * masked_fweights[e];
    }
    spmv_result[v] = sum;
}






__global__ void update_diff_mixed_kernel(
    const float* __restrict__ spmv_result_f32,
    const double* __restrict__ pr_old,
    double* __restrict__ pr_new,
    const double* __restrict__ inv_out_weight,
    float* __restrict__ pr_norm_f32_out,
    double alpha,
    double one_minus_alpha_over_n,
    double alpha_over_n,
    const double* __restrict__ dangling_sum_ptr,
    double* __restrict__ diff_out,
    double* __restrict__ next_dangling_sum,
    int32_t num_vertices,
    int32_t zero_deg_start)
{
    __shared__ double warp_diffs[32];
    __shared__ double warp_danglings[32];

    double base_val = one_minus_alpha_over_n + alpha_over_n * (*dangling_sum_ptr);

    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;

    double local_diff = 0.0;
    double local_dangling = 0.0;

    for (int32_t v = tid; v < num_vertices; v += stride) {
        double old_val = pr_old[v];
        double spmv_val = (v < zero_deg_start) ? (double)spmv_result_f32[v] : 0.0;
        double new_val = base_val + alpha * spmv_val;
        pr_new[v] = new_val;
        local_diff += fabs(new_val - old_val);

        double iw = inv_out_weight[v];
        pr_norm_f32_out[v] = __double2float_rn(new_val * iw);
        if (iw == 0.0) local_dangling += new_val;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_diff += __shfl_down_sync(0xffffffff, local_diff, offset);
        local_dangling += __shfl_down_sync(0xffffffff, local_dangling, offset);
    }

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        warp_diffs[warp_id] = local_diff;
        warp_danglings[warp_id] = local_dangling;
    }
    __syncthreads();

    int num_warps = blockDim.x >> 5;
    if (warp_id == 0) {
        local_diff = (lane < num_warps) ? warp_diffs[lane] : 0.0;
        local_dangling = (lane < num_warps) ? warp_danglings[lane] : 0.0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_diff += __shfl_down_sync(0xffffffff, local_diff, offset);
            local_dangling += __shfl_down_sync(0xffffffff, local_dangling, offset);
        }
        if (lane == 0) {
            atomicAdd(diff_out, local_diff);
            atomicAdd(next_dangling_sum, local_dangling);
        }
    }
}





static inline int ceildiv(int a, int b) { return (a + b - 1) / b; }
static inline int mymin(int a, int b) { return a < b ? a : b; }

static void launch_compute_out_weight_sums(
    const int32_t* indices, const double* edge_weights,
    const uint32_t* edge_mask, double* out_weight_sums,
    int32_t num_edges, cudaStream_t stream)
{
    compute_out_weight_sums_kernel<<<ceildiv(num_edges, 256), 256, 0, stream>>>(
        indices, edge_weights, edge_mask, out_weight_sums, num_edges);
}

static void launch_compute_inv_out_weight(
    const double* out_weight_sums, double* inv_out_weight,
    int32_t num_vertices, cudaStream_t stream)
{
    compute_inv_out_weight_kernel<<<ceildiv(num_vertices, 256), 256, 0, stream>>>(
        out_weight_sums, inv_out_weight, num_vertices);
}

static void launch_prepare_masked_float_weights(
    const double* edge_weights, const uint32_t* edge_mask,
    float* masked_fweights, int32_t num_edges, cudaStream_t stream)
{
    prepare_masked_float_weights_kernel<<<ceildiv(num_edges, 256), 256, 0, stream>>>(
        edge_weights, edge_mask, masked_fweights, num_edges);
}

static void launch_init_pr(double* pr, double val, int32_t n, cudaStream_t stream) {
    init_pr_kernel<<<ceildiv(n, 256), 256, 0, stream>>>(pr, val, n);
}

static void launch_pr_norm_f32_dangling(
    const double* pr, const double* inv_out_weight,
    float* pr_norm_f32, double* dangling_sum,
    int32_t num_vertices, cudaStream_t stream)
{
    int grid = mymin(ceildiv(num_vertices, 256), 2048);
    pr_norm_f32_dangling_kernel<<<grid, 256, 0, stream>>>(
        pr, inv_out_weight, pr_norm_f32, dangling_sum, num_vertices);
}

static void launch_spmv_f32_block(
    const int32_t* offsets, const int32_t* indices,
    const float* masked_fweights, const float* pr_norm_f32,
    float* spmv_result, int32_t seg_start, int32_t seg_end, cudaStream_t stream)
{
    int n = seg_end - seg_start;
    if (n <= 0) return;
    spmv_f32_block_kernel<<<n, 256, 0, stream>>>(
        offsets, indices, masked_fweights, pr_norm_f32, spmv_result, seg_start, seg_end);
}

static void launch_spmv_f32_warp(
    const int32_t* offsets, const int32_t* indices,
    const float* masked_fweights, const float* pr_norm_f32,
    float* spmv_result, int32_t seg_start, int32_t seg_end, cudaStream_t stream)
{
    int n = seg_end - seg_start;
    if (n <= 0) return;
    int tpb = 256;
    spmv_f32_warp_kernel<<<ceildiv(n, tpb/32), tpb, 0, stream>>>(
        offsets, indices, masked_fweights, pr_norm_f32, spmv_result, seg_start, seg_end);
}

static void launch_spmv_f32_thread(
    const int32_t* offsets, const int32_t* indices,
    const float* masked_fweights, const float* pr_norm_f32,
    float* spmv_result, int32_t seg_start, int32_t seg_end, cudaStream_t stream)
{
    int n = seg_end - seg_start;
    if (n <= 0) return;
    spmv_f32_thread_kernel<<<ceildiv(n, 256), 256, 0, stream>>>(
        offsets, indices, masked_fweights, pr_norm_f32, spmv_result, seg_start, seg_end);
}

static void launch_update_diff_mixed(
    const float* spmv_result_f32, const double* pr_old,
    double* pr_new, const double* inv_out_weight,
    float* pr_norm_f32_out, double alpha,
    double one_minus_alpha_over_n, double alpha_over_n,
    const double* dangling_sum, double* diff_out,
    double* next_dangling_sum,
    int32_t num_vertices, int32_t zero_deg_start, cudaStream_t stream)
{
    int grid = mymin(ceildiv(num_vertices, 256), 2048);
    update_diff_mixed_kernel<<<grid, 256, 0, stream>>>(
        spmv_result_f32, pr_old, pr_new, inv_out_weight,
        pr_norm_f32_out, alpha,
        one_minus_alpha_over_n, alpha_over_n,
        dangling_sum, diff_out, next_dangling_sum,
        num_vertices, zero_deg_start);
}

}  

PageRankResult pagerank_seg_mask(const graph32_t& graph,
                                 const double* edge_weights,
                                 double* pageranks,
                                 const double* precomputed_vertex_out_weight_sums,
                                 double alpha,
                                 double epsilon,
                                 std::size_t max_iterations,
                                 const double* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg3 = seg[3];

    cache.ensure(num_vertices, num_edges);

    cudaStream_t stream = 0;

    
    double* d_out_ws = cache.out_ws;
    double* d_inv_ow = cache.inv_ow;

    cudaMemsetAsync(d_out_ws, 0, num_vertices * sizeof(double), stream);
    launch_compute_out_weight_sums(d_indices, edge_weights, d_edge_mask, d_out_ws, num_edges, stream);
    launch_compute_inv_out_weight(d_out_ws, d_inv_ow, num_vertices, stream);

    
    float* d_masked_fw = cache.masked_fw;
    launch_prepare_masked_float_weights(edge_weights, d_edge_mask, d_masked_fw, num_edges, stream);

    
    float* d_pr_norm_f32 = cache.pr_norm_f32;
    float* d_spmv_f32 = cache.spmv_f32;
    double* d_pr_a = cache.pr_a;
    double* d_pr_b = cache.pr_b;
    double* d_dsum_a = cache.dsum_a;
    double* d_dsum_b = cache.dsum_b;
    double* d_diff = cache.diff;

    
    double* d_pr_old = d_pr_a;
    double* d_pr_new = d_pr_b;

    if (initial_pageranks != nullptr) {
        cudaMemcpyAsync(d_pr_old, initial_pageranks,
                       num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else {
        launch_init_pr(d_pr_old, 1.0 / num_vertices, num_vertices, stream);
    }

    double one_minus_alpha_over_n = (1.0 - alpha) / num_vertices;
    double alpha_over_n = alpha / num_vertices;

    
    double* d_dsum_cur = d_dsum_a;
    double* d_dsum_next = d_dsum_b;
    cudaMemsetAsync(d_dsum_cur, 0, sizeof(double), stream);
    launch_pr_norm_f32_dangling(d_pr_old, d_inv_ow, d_pr_norm_f32, d_dsum_cur, num_vertices, stream);

    
    bool converged = false;
    std::size_t iterations = 0;
    double h_diff;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        launch_spmv_f32_block(d_offsets, d_indices, d_masked_fw, d_pr_norm_f32,
                            d_spmv_f32, seg0, seg1, stream);
        launch_spmv_f32_warp(d_offsets, d_indices, d_masked_fw, d_pr_norm_f32,
                           d_spmv_f32, seg1, seg2, stream);
        launch_spmv_f32_thread(d_offsets, d_indices, d_masked_fw, d_pr_norm_f32,
                             d_spmv_f32, seg2, seg3, stream);

        cudaMemsetAsync(d_diff, 0, sizeof(double), stream);
        cudaMemsetAsync(d_dsum_next, 0, sizeof(double), stream);

        
        launch_update_diff_mixed(d_spmv_f32, d_pr_old, d_pr_new, d_inv_ow,
                               d_pr_norm_f32, alpha,
                               one_minus_alpha_over_n, alpha_over_n,
                               d_dsum_cur, d_diff, d_dsum_next,
                               num_vertices, seg3, stream);

        cudaMemcpyAsync(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations++;

        if (h_diff < epsilon) {
            converged = true;
            break;
        }

        double* tmp = d_pr_old; d_pr_old = d_pr_new; d_pr_new = tmp;
        tmp = d_dsum_cur; d_dsum_cur = d_dsum_next; d_dsum_next = tmp;
    }

    double* d_result = converged ? d_pr_new : d_pr_old;
    if (iterations == 0) d_result = d_pr_old;

    cudaMemcpyAsync(pageranks, d_result,
                   num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return PageRankResult{iterations, converged};
}

}  
