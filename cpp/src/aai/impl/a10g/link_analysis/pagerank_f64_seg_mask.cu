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

struct __align__(8) PackedEdge {
    float norm_wt;
    int32_t src;
};



__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ out_weight_sums,
    int32_t num_edges)
{
    int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e < num_edges) {
        uint32_t mw = edge_mask[e >> 5];
        if (mw & (1u << (e & 31))) {
            atomicAdd(&out_weight_sums[indices[e]], edge_weights[e]);
        }
    }
}

__global__ void compute_packed_edges_kernel(
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ out_weight_sums,
    PackedEdge* __restrict__ packed_edges,
    int32_t num_edges)
{
    int64_t e = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (e < num_edges) {
        uint32_t mw = edge_mask[e >> 5];
        if (mw & (1u << (e & 31))) {
            int32_t src = indices[e];
            double ow = out_weight_sums[src];
            float nw = (ow > 0.0) ? (float)(edge_weights[e] / ow) : 0.0f;
            packed_edges[e] = {nw, src};
        } else {
            packed_edges[e] = {0.0f, 0};
        }
    }
}

__global__ void find_dangling_kernel(
    const double* __restrict__ out_weight_sums,
    int32_t num_vertices,
    int32_t* __restrict__ dangling_indices,
    int32_t* __restrict__ num_dangling)
{
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices && out_weight_sums[v] == 0.0) {
        int32_t pos = atomicAdd(num_dangling, 1);
        dangling_indices[pos] = v;
    }
}

__global__ void init_pr_kernel(double* __restrict__ pr, int32_t n) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pr[i] = 1.0 / (double)n;
}



__global__ void dangling_sum_kernel(
    const int32_t* __restrict__ dangling_indices,
    const double* __restrict__ pr,
    int32_t num_dangling,
    double* __restrict__ result)
{
    double sum = 0.0;
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_dangling; i += gridDim.x * blockDim.x) {
        sum += pr[dangling_indices[i]];
    }
    typedef cub::BlockReduce<double, 256> BR;
    __shared__ typename BR::TempStorage temp;
    double bs = BR(temp).Sum(sum);
    if (threadIdx.x == 0 && bs != 0.0) atomicAdd(result, bs);
}



__global__ void spmv_unified_kernel(
    const int32_t* __restrict__ offsets,
    const PackedEdge* __restrict__ packed_edges,
    const double* __restrict__ pr_old,
    double* __restrict__ pr_new,
    const double* __restrict__ dangling_sum_ptr,
    double alpha, double one_minus_alpha_over_n, double alpha_over_n,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg4,
    int32_t n_high_blocks, int32_t n_high_mid_blocks,
    double* __restrict__ l1_diff)
{
    double dangling_sum = *dangling_sum_ptr;
    double base_val = one_minus_alpha_over_n + alpha_over_n * dangling_sum;

    typedef cub::BlockReduce<double, 256> BR;
    __shared__ typename BR::TempStorage temp;

    if (blockIdx.x < n_high_blocks) {
        
        int32_t v = seg0 + blockIdx.x;
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        double sum = 0.0;
        for (int32_t e = start + threadIdx.x; e < end; e += 256) {
            PackedEdge pe = packed_edges[e];
            sum += (double)pe.norm_wt * pr_old[pe.src];
        }

        double block_sum = BR(temp).Sum(sum);

        if (threadIdx.x == 0) {
            double new_val = base_val + alpha * block_sum;
            pr_new[v] = new_val;
            double diff = fabs(new_val - pr_old[v]);
            if (diff != 0.0) atomicAdd(l1_diff, diff);
        }
    } else if (blockIdx.x < n_high_mid_blocks) {
        
        int32_t local_block = blockIdx.x - n_high_blocks;
        int32_t warp_in_block = threadIdx.x >> 5;
        int32_t lane = threadIdx.x & 31;
        int32_t v = seg1 + local_block * 8 + warp_in_block;
        if (v >= seg2) return;

        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        double sum = 0.0;
        for (int32_t e = start + lane; e < end; e += 32) {
            PackedEdge pe = packed_edges[e];
            sum += (double)pe.norm_wt * pr_old[pe.src];
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0) {
            double new_val = base_val + alpha * sum;
            pr_new[v] = new_val;
            double diff = fabs(new_val - pr_old[v]);
            if (diff != 0.0) atomicAdd(l1_diff, diff);
        }
    } else {
        
        int32_t local_block = blockIdx.x - n_high_mid_blocks;
        int32_t v = seg2 + local_block * 256 + threadIdx.x;

        double diff = 0.0;
        if (v < seg4) {
            int32_t start = offsets[v];
            int32_t end = offsets[v + 1];

            double sum = 0.0;
            for (int32_t e = start; e < end; e++) {
                PackedEdge pe = packed_edges[e];
                sum += (double)pe.norm_wt * pr_old[pe.src];
            }

            double new_val = base_val + alpha * sum;
            pr_new[v] = new_val;
            diff = fabs(new_val - pr_old[v]);
        }

        double bs = BR(temp).Sum(diff);
        if (threadIdx.x == 0 && bs != 0.0) atomicAdd(l1_diff, bs);
    }
}



struct Cache : Cacheable {
    double* h_pinned = nullptr;
    int32_t* h_pinned_int = nullptr;

    double* out_weight_sums = nullptr;
    int64_t out_weight_sums_cap = 0;

    PackedEdge* packed_edges = nullptr;
    int64_t packed_edges_cap = 0;

    int32_t* dangling_indices = nullptr;
    int64_t dangling_indices_cap = 0;

    int32_t* num_dangling = nullptr;

    double* pr_buf = nullptr;
    int64_t pr_buf_cap = 0;

    double* scalar_buf = nullptr;

    Cache() {
        cudaMallocHost(&h_pinned, sizeof(double));
        cudaMallocHost(&h_pinned_int, sizeof(int32_t));
        cudaMalloc(&num_dangling, sizeof(int32_t));
        cudaMalloc(&scalar_buf, 2 * sizeof(double));
    }

    ~Cache() override {
        if (h_pinned) cudaFreeHost(h_pinned);
        if (h_pinned_int) cudaFreeHost(h_pinned_int);
        if (out_weight_sums) cudaFree(out_weight_sums);
        if (packed_edges) cudaFree(packed_edges);
        if (dangling_indices) cudaFree(dangling_indices);
        if (num_dangling) cudaFree(num_dangling);
        if (pr_buf) cudaFree(pr_buf);
        if (scalar_buf) cudaFree(scalar_buf);
    }

    void ensure(int32_t nv, int32_t ne) {
        if (out_weight_sums_cap < nv) {
            if (out_weight_sums) cudaFree(out_weight_sums);
            cudaMalloc(&out_weight_sums, (int64_t)nv * sizeof(double));
            out_weight_sums_cap = nv;
        }
        if (packed_edges_cap < ne) {
            if (packed_edges) cudaFree(packed_edges);
            cudaMalloc(&packed_edges, (int64_t)ne * sizeof(PackedEdge));
            packed_edges_cap = ne;
        }
        if (dangling_indices_cap < nv) {
            if (dangling_indices) cudaFree(dangling_indices);
            cudaMalloc(&dangling_indices, (int64_t)nv * sizeof(int32_t));
            dangling_indices_cap = nv;
        }
        if (pr_buf_cap < nv) {
            if (pr_buf) cudaFree(pr_buf);
            cudaMalloc(&pr_buf, (int64_t)nv * sizeof(double));
            pr_buf_cap = nv;
        }
    }
};

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
    int32_t seg0 = seg[0];
    int32_t seg1 = seg[1];
    int32_t seg2 = seg[2];
    int32_t seg4 = seg[4];

    cudaStream_t stream = 0;

    cache.ensure(num_vertices, num_edges);

    
    cudaMemsetAsync(cache.out_weight_sums, 0, (int64_t)num_vertices * sizeof(double), stream);
    {
        int block = 256;
        int grid = ((int64_t)num_edges + block - 1) / block;
        compute_out_weight_sums_kernel<<<grid, block, 0, stream>>>(
            d_indices, edge_weights, d_edge_mask, cache.out_weight_sums, num_edges);
    }
    const double* d_out_weight_sums = cache.out_weight_sums;

    {
        int block = 256;
        int grid = ((int64_t)num_edges + block - 1) / block;
        compute_packed_edges_kernel<<<grid, block, 0, stream>>>(
            d_indices, edge_weights, d_edge_mask, d_out_weight_sums,
            cache.packed_edges, num_edges);
    }

    cudaMemsetAsync(cache.num_dangling, 0, sizeof(int32_t), stream);
    {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        find_dangling_kernel<<<grid, block, 0, stream>>>(
            d_out_weight_sums, num_vertices, cache.dangling_indices, cache.num_dangling);
    }

    cudaMemcpyAsync(cache.h_pinned_int, cache.num_dangling, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int32_t h_num_dangling = *cache.h_pinned_int;

    
    if (initial_pageranks) {
        cudaMemcpyAsync(pageranks, initial_pageranks,
                        (int64_t)num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    } else {
        int block = 256;
        int grid = (num_vertices + block - 1) / block;
        init_pr_kernel<<<grid, block, 0, stream>>>(pageranks, num_vertices);
    }

    double one_minus_alpha_over_n = (1.0 - alpha) / (double)num_vertices;
    double alpha_over_n = alpha / (double)num_vertices;

    double* d_dangling_sum = &cache.scalar_buf[0];
    double* d_l1_diff = &cache.scalar_buf[1];

    
    bool converged = false;
    std::size_t iterations = 0;
    double* pr_old = pageranks;
    double* pr_new = cache.pr_buf;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        cudaMemsetAsync(cache.scalar_buf, 0, 2 * sizeof(double), stream);

        if (h_num_dangling > 0) {
            int block = 256;
            int grid = (h_num_dangling + block - 1) / block;
            if (grid > 1024) grid = 1024;
            dangling_sum_kernel<<<grid, block, 0, stream>>>(
                cache.dangling_indices, pr_old, h_num_dangling, d_dangling_sum);
        }

        {
            int32_t n_high = seg1 - seg0;
            int32_t n_mid_count = seg2 - seg1;
            int32_t n_low_count = seg4 - seg2;

            int32_t n_high_blocks = n_high;
            int32_t n_mid_blocks = (n_mid_count + 7) / 8;
            int32_t n_low_blocks = (n_low_count + 255) / 256;
            int32_t total_blocks = n_high_blocks + n_mid_blocks + n_low_blocks;

            if (total_blocks > 0) {
                spmv_unified_kernel<<<total_blocks, 256, 0, stream>>>(
                    d_offsets, cache.packed_edges, pr_old, pr_new, d_dangling_sum,
                    alpha, one_minus_alpha_over_n, alpha_over_n,
                    seg0, seg1, seg2, seg4,
                    n_high_blocks, n_high_blocks + n_mid_blocks, d_l1_diff);
            }
        }

        iterations++;

        cudaMemcpyAsync(cache.h_pinned, d_l1_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (*cache.h_pinned < epsilon) {
            converged = true;
            if (pr_new != pageranks) {
                cudaMemcpyAsync(pageranks, pr_new,
                                (int64_t)num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
            }
            break;
        }

        double* tmp = pr_old; pr_old = pr_new; pr_new = tmp;
    }

    if (!converged && pr_old != pageranks) {
        cudaMemcpyAsync(pageranks, pr_old,
                        (int64_t)num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    }
    cudaStreamSynchronize(stream);

    return PageRankResult{iterations, converged};
}

}  
