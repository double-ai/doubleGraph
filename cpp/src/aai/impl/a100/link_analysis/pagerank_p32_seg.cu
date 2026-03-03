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
#include <vector>

namespace aai {

namespace {

struct Cache : Cacheable {
    cudaStream_t stream = 0;
    int32_t* out_degree = nullptr;
    float* inv_out_degree = nullptr;
    float* pers_full = nullptr;
    float* pr_buf = nullptr;
    float* x = nullptr;
    float* dangling_sum = nullptr;
    float* diff_sum = nullptr;

    int64_t out_degree_cap = 0;
    int64_t inv_out_degree_cap = 0;
    int64_t pers_full_cap = 0;
    int64_t pr_buf_cap = 0;
    int64_t x_cap = 0;
    bool dangling_allocated = false;
    bool diff_allocated = false;

    Cache() {
        cudaStreamCreate(&stream);
    }

    void ensure(int32_t N) {
        if (out_degree_cap < N) {
            if (out_degree) cudaFree(out_degree);
            cudaMalloc(&out_degree, N * sizeof(int32_t));
            out_degree_cap = N;
        }
        if (inv_out_degree_cap < N) {
            if (inv_out_degree) cudaFree(inv_out_degree);
            cudaMalloc(&inv_out_degree, N * sizeof(float));
            inv_out_degree_cap = N;
        }
        if (pers_full_cap < N) {
            if (pers_full) cudaFree(pers_full);
            cudaMalloc(&pers_full, N * sizeof(float));
            pers_full_cap = N;
        }
        if (pr_buf_cap < N) {
            if (pr_buf) cudaFree(pr_buf);
            cudaMalloc(&pr_buf, N * sizeof(float));
            pr_buf_cap = N;
        }
        if (x_cap < N) {
            if (x) cudaFree(x);
            cudaMalloc(&x, N * sizeof(float));
            x_cap = N;
        }
        if (!dangling_allocated) {
            cudaMalloc(&dangling_sum, sizeof(float));
            dangling_allocated = true;
        }
        if (!diff_allocated) {
            cudaMalloc(&diff_sum, sizeof(float));
            diff_allocated = true;
        }
    }

    ~Cache() override {
        if (stream) cudaStreamDestroy(stream);
        if (out_degree) cudaFree(out_degree);
        if (inv_out_degree) cudaFree(inv_out_degree);
        if (pers_full) cudaFree(pers_full);
        if (pr_buf) cudaFree(pr_buf);
        if (x) cudaFree(x);
        if (dangling_sum) cudaFree(dangling_sum);
        if (diff_sum) cudaFree(diff_sum);
    }
};



__global__ void compute_out_degree_kernel(
    const int32_t* __restrict__ indices,
    int num_edges,
    int32_t* __restrict__ out_degree)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_edges; i += blockDim.x * gridDim.x) {
        atomicAdd(&out_degree[indices[i]], 1);
    }
}

__global__ void compute_inv_degree_and_init_pr_kernel(
    float* __restrict__ pr,
    float* __restrict__ inv_out_degree,
    const int32_t* __restrict__ out_degree,
    int num_vertices,
    float init_val,
    bool use_initial,
    const float* __restrict__ initial_pr)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vertices; i += blockDim.x * gridDim.x) {
        pr[i] = use_initial ? initial_pr[i] : init_val;
        int deg = out_degree[i];
        inv_out_degree[i] = (deg > 0) ? (1.0f / (float)deg) : 0.0f;
    }
}

__global__ void scatter_pers_kernel(
    const int32_t* __restrict__ pers_verts,
    const float* __restrict__ pers_vals,
    int pers_size,
    float* __restrict__ pers_full,
    float inv_sum)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pers_size) {
        pers_full[pers_verts[i]] = pers_vals[i] * inv_sum;
    }
}



__global__ void zero_scalars_kernel(float* a, float* b) {
    *a = 0.0f;
    *b = 0.0f;
}

__global__ void prepare_x_kernel(
    const float* __restrict__ pr,
    const float* __restrict__ inv_out_degree,
    float* __restrict__ x,
    float* __restrict__ dangling_sum,
    int num_vertices)
{
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage temp;

    float dang = 0.0f;
    for (int i = blockIdx.x * 256 + threadIdx.x; i < num_vertices; i += gridDim.x * 256) {
        float p = pr[i];
        float inv_d = inv_out_degree[i];
        x[i] = p * inv_d;
        if (inv_d == 0.0f) dang += p;
    }

    float bd = BR(temp).Sum(dang);
    if (threadIdx.x == 0 && bd != 0.0f) atomicAdd(dangling_sum, bd);
}

__global__ void spmv_high_kernel(
    const int32_t* __restrict__ col_offsets,
    const int32_t* __restrict__ row_indices,
    const float* __restrict__ x,
    const float* __restrict__ pers,
    const float* __restrict__ old_pr,
    float* __restrict__ new_pr,
    float alpha,
    float one_minus_alpha,
    const float* __restrict__ dangling_sum_ptr,
    float* __restrict__ diff_sum,
    int seg_start,
    int seg_count)
{
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage temp;

    int bid = blockIdx.x;
    if (bid >= seg_count) return;
    int v = bid + seg_start;

    int start = col_offsets[v];
    int end = col_offsets[v + 1];

    float s = 0.0f;
    for (int j = start + threadIdx.x; j < end; j += 256) {
        s += x[row_indices[j]];
    }
    s = BR(temp).Sum(s);

    if (threadIdx.x == 0) {
        float base = alpha * (*dangling_sum_ptr) + one_minus_alpha;
        float nv = alpha * s + base * pers[v];
        new_pr[v] = nv;
        atomicAdd(diff_sum, fabsf(nv - old_pr[v]));
    }
}

__global__ void spmv_mid_kernel(
    const int32_t* __restrict__ col_offsets,
    const int32_t* __restrict__ row_indices,
    const float* __restrict__ x,
    const float* __restrict__ pers,
    const float* __restrict__ old_pr,
    float* __restrict__ new_pr,
    float alpha,
    float one_minus_alpha,
    const float* __restrict__ dangling_sum_ptr,
    float* __restrict__ diff_sum,
    int seg_start,
    int seg_count)
{
    constexpr int WPB = 8;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane = tid % 32;
    int warp_in_block = threadIdx.x / 32;

    __shared__ float warp_diffs[WPB];

    float diff = 0.0f;

    if (warp_id < seg_count) {
        int v = warp_id + seg_start;
        int start = col_offsets[v];
        int end = col_offsets[v + 1];

        float s = 0.0f;
        for (int j = start + lane; j < end; j += 32) {
            s += x[row_indices[j]];
        }

        #pragma unroll
        for (int o = 16; o > 0; o /= 2)
            s += __shfl_down_sync(0xFFFFFFFF, s, o);

        if (lane == 0) {
            float base = alpha * (*dangling_sum_ptr) + one_minus_alpha;
            float nv = alpha * s + base * pers[v];
            new_pr[v] = nv;
            diff = fabsf(nv - old_pr[v]);
        }
    }

    if (lane == 0) {
        warp_diffs[warp_in_block] = diff;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float bd = 0.0f;
        int nw = seg_count - (int)(blockIdx.x * WPB);
        if (nw > WPB) nw = WPB;
        if (nw <= 0) return;
        for (int i = 0; i < nw; i++) bd += warp_diffs[i];
        if (bd != 0.0f) atomicAdd(diff_sum, bd);
    }
}

__global__ void spmv_low_kernel(
    const int32_t* __restrict__ col_offsets,
    const int32_t* __restrict__ row_indices,
    const float* __restrict__ x,
    const float* __restrict__ pers,
    const float* __restrict__ old_pr,
    float* __restrict__ new_pr,
    float alpha,
    float one_minus_alpha,
    const float* __restrict__ dangling_sum_ptr,
    float* __restrict__ diff_sum,
    int seg_start,
    int seg_count)
{
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage temp;

    int tid = blockIdx.x * 256 + threadIdx.x;

    float diff = 0.0f;
    if (tid < seg_count) {
        int v = tid + seg_start;
        int start = col_offsets[v];
        int end = col_offsets[v + 1];

        float s = 0.0f;
        for (int j = start; j < end; j++) {
            s += x[row_indices[j]];
        }

        float base = alpha * (*dangling_sum_ptr) + one_minus_alpha;
        float nv = alpha * s + base * pers[v];
        new_pr[v] = nv;
        diff = fabsf(nv - old_pr[v]);
    }

    float bd = BR(temp).Sum(diff);
    if (threadIdx.x == 0 && bd != 0.0f) atomicAdd(diff_sum, bd);
}



struct IterationParams {
    const int32_t* offsets;
    const int32_t* indices;
    int num_vertices;
    const float* inv_out_degree;
    const float* pers_full;
    float* x;
    float* dangling_sum;
    float* diff_sum;
    float alpha;
    float one_minus_alpha;
    int seg0, seg1, seg2, seg4;
    int grid_prep;
    int n_high;
    int grid_mid;
    int n_mid;
    int grid_low;
    int n_low_zero;
};

static void launch_one_iteration(
    const IterationParams& p,
    float* pr, float* new_pr,
    cudaStream_t stream)
{
    zero_scalars_kernel<<<1, 1, 0, stream>>>(p.dangling_sum, p.diff_sum);

    prepare_x_kernel<<<p.grid_prep, 256, 0, stream>>>(
        pr, p.inv_out_degree, p.x, p.dangling_sum, p.num_vertices);

    if (p.n_high > 0) {
        spmv_high_kernel<<<p.n_high, 256, 0, stream>>>(
            p.offsets, p.indices, p.x, p.pers_full, pr, new_pr,
            p.alpha, p.one_minus_alpha, p.dangling_sum, p.diff_sum,
            p.seg0, p.n_high);
    }

    if (p.n_mid > 0) {
        spmv_mid_kernel<<<p.grid_mid, 256, 0, stream>>>(
            p.offsets, p.indices, p.x, p.pers_full, pr, new_pr,
            p.alpha, p.one_minus_alpha, p.dangling_sum, p.diff_sum,
            p.seg1, p.n_mid);
    }

    if (p.n_low_zero > 0) {
        spmv_low_kernel<<<p.grid_low, 256, 0, stream>>>(
            p.offsets, p.indices, p.x, p.pers_full, pr, new_pr,
            p.alpha, p.one_minus_alpha, p.dangling_sum, p.diff_sum,
            p.seg2, p.n_low_zero);
    }
}

static void do_setup(
    const int32_t* indices, int num_edges,
    int32_t* out_degree, float* inv_out_degree,
    float* pr, int num_vertices,
    float init_val, bool use_initial, const float* initial_pr,
    const int32_t* pers_verts, const float* pers_vals,
    int pers_size, float* pers_full, float inv_pers_sum,
    cudaStream_t stream)
{
    cudaMemsetAsync(out_degree, 0, num_vertices * sizeof(int32_t), stream);
    cudaMemsetAsync(pers_full, 0, num_vertices * sizeof(float), stream);

    int grid1 = (num_edges + 255) / 256;
    if (grid1 > 4096) grid1 = 4096;
    compute_out_degree_kernel<<<grid1, 256, 0, stream>>>(indices, num_edges, out_degree);

    int grid2 = (num_vertices + 255) / 256;
    compute_inv_degree_and_init_pr_kernel<<<grid2, 256, 0, stream>>>(
        pr, inv_out_degree, out_degree, num_vertices, init_val, use_initial, initial_pr);

    if (pers_size > 0) {
        int grid3 = (pers_size + 255) / 256;
        scatter_pers_kernel<<<grid3, 256, 0, stream>>>(
            pers_verts, pers_vals, pers_size, pers_full, inv_pers_sum);
    }
}

static size_t do_pagerank_iterate(
    const int32_t* offsets, const int32_t* indices,
    int num_vertices,
    const float* inv_out_degree,
    const float* pers_full,
    float* pr_a, float* pr_b, float* x,
    float* dangling_sum, float* diff_sum,
    float alpha, float epsilon,
    size_t max_iterations,
    int seg0, int seg1, int seg2, int seg4,
    bool* out_converged,
    cudaStream_t stream)
{
    float one_minus_alpha = 1.0f - alpha;

    int n_high = seg1 - seg0;
    int n_mid = seg2 - seg1;
    int n_low_zero = seg4 - seg2;

    IterationParams p;
    p.offsets = offsets;
    p.indices = indices;
    p.num_vertices = num_vertices;
    p.inv_out_degree = inv_out_degree;
    p.pers_full = pers_full;
    p.x = x;
    p.dangling_sum = dangling_sum;
    p.diff_sum = diff_sum;
    p.alpha = alpha;
    p.one_minus_alpha = one_minus_alpha;
    p.seg0 = seg0;
    p.seg1 = seg1;
    p.seg2 = seg2;
    p.seg4 = seg4;
    p.grid_prep = (num_vertices + 255) / 256;
    if (p.grid_prep > 1024) p.grid_prep = 1024;
    p.n_high = n_high;
    p.grid_mid = n_mid > 0 ? (n_mid + 7) / 8 : 0;
    p.n_mid = n_mid;
    p.grid_low = n_low_zero > 0 ? (n_low_zero + 255) / 256 : 0;
    p.n_low_zero = n_low_zero;

    cudaGraph_t graphA, graphB;
    cudaGraphExec_t graphExecA = nullptr, graphExecB = nullptr;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launch_one_iteration(p, pr_a, pr_b, stream);
    cudaStreamEndCapture(stream, &graphA);
    cudaGraphInstantiate(&graphExecA, graphA, 0);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    launch_one_iteration(p, pr_b, pr_a, stream);
    cudaStreamEndCapture(stream, &graphB);
    cudaGraphInstantiate(&graphExecB, graphB, 0);

    float h_diff;

    size_t check_interval = 10;
    if (max_iterations <= 20) check_interval = 1;
    else if (max_iterations <= 50) check_interval = 5;

    for (size_t iter = 0; iter < max_iterations; iter++) {
        if (iter % 2 == 0) {
            cudaGraphLaunch(graphExecA, stream);
        } else {
            cudaGraphLaunch(graphExecB, stream);
        }

        bool should_check = ((iter + 1) % check_interval == 0) || (iter + 1 >= max_iterations);
        if (should_check) {
            cudaMemcpyAsync(&h_diff, diff_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            if (h_diff < epsilon) {
                *out_converged = true;
                cudaGraphExecDestroy(graphExecA);
                cudaGraphExecDestroy(graphExecB);
                cudaGraphDestroy(graphA);
                cudaGraphDestroy(graphB);
                return iter + 1;
            }
        }
    }

    *out_converged = false;
    cudaGraphExecDestroy(graphExecA);
    cudaGraphExecDestroy(graphExecB);
    cudaGraphDestroy(graphA);
    cudaGraphDestroy(graphB);
    return max_iterations;
}

}  

PageRankResult personalized_pagerank_seg(const graph32_t& graph,
                                         const int32_t* personalization_vertices,
                                         const float* personalization_values,
                                         std::size_t personalization_size,
                                         float* pageranks,
                                         const float* precomputed_vertex_out_weight_sums,
                                         float alpha,
                                         float epsilon,
                                         std::size_t max_iterations,
                                         const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t N = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;

    cache.ensure(N);

    const auto& seg = graph.segment_offsets.value();
    int seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg4 = seg[4];

    int pers_size = static_cast<int>(personalization_size);
    float pers_sum = 0.0f;
    if (pers_size > 0) {
        std::vector<float> h_pers(pers_size);
        cudaMemcpy(h_pers.data(), personalization_values,
                   pers_size * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < pers_size; i++) pers_sum += h_pers[i];
    }
    float inv_pers_sum = (pers_sum > 0.0f) ? (1.0f / pers_sum) : 0.0f;

    bool has_initial = (initial_pageranks != nullptr);
    float init_val = 1.0f / (float)N;

    do_setup(
        graph.indices, E,
        cache.out_degree, cache.inv_out_degree,
        pageranks, N, init_val, has_initial, initial_pageranks,
        personalization_vertices, personalization_values,
        pers_size, cache.pers_full, inv_pers_sum,
        cache.stream);

    cudaStreamSynchronize(cache.stream);

    bool converged = false;
    size_t iterations = do_pagerank_iterate(
        graph.offsets, graph.indices,
        N,
        cache.inv_out_degree,
        cache.pers_full,
        pageranks, cache.pr_buf, cache.x,
        cache.dangling_sum, cache.diff_sum,
        alpha, epsilon, max_iterations,
        seg0, seg1, seg2, seg4,
        &converged, cache.stream);

    if (iterations % 2 == 1) {
        cudaMemcpy(pageranks, cache.pr_buf, N * sizeof(float), cudaMemcpyDeviceToDevice);
    }

    return PageRankResult{iterations, converged};
}

}  
