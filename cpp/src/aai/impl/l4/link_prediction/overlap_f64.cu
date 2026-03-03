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
#include <cstddef>

namespace aai {

namespace {




struct Cache : Cacheable {
    double* weight_sums = nullptr;
    int64_t weight_sums_capacity = 0;

    void ensure(int64_t num_vertices) {
        if (weight_sums_capacity < num_vertices) {
            if (weight_sums) cudaFree(weight_sums);
            cudaMalloc(&weight_sums, num_vertices * sizeof(double));
            weight_sums_capacity = num_vertices;
        }
    }

    ~Cache() override {
        if (weight_sums) cudaFree(weight_sums);
    }
};




#define WARP_SIZE 32

__global__ void compute_weight_sums_warp_kernel(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ edge_weights,
    double* __restrict__ weight_sums,
    int32_t num_vertices)
{
    const int lane = threadIdx.x & 31;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp_global >= num_vertices) return;

    int32_t start = offsets[warp_global];
    int32_t end = offsets[warp_global + 1];

    double sum = 0.0;
    for (int32_t i = start + lane; i < end; i += WARP_SIZE) {
        sum += edge_weights[i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) weight_sums[warp_global] = sum;
}




__device__ __forceinline__ int32_t lb_smem(
    const int32_t* arr, int32_t lo, int32_t hi, int32_t target)
{
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int32_t lb_gmem(
    const int32_t* __restrict__ arr, int32_t lo, int32_t hi, int32_t target)
{
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (__ldg(arr + mid) < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}




#define SMEM_B_CAP 512
#define WARPS_PER_BLOCK 8
#define BLOCK_SIZE (WARP_SIZE * WARPS_PER_BLOCK)

__global__ void overlap_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const double* __restrict__ weight_sums,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ scores,
    int64_t num_pairs,
    bool is_multigraph)
{
    __shared__ int32_t s_b[WARPS_PER_BLOCK * SMEM_B_CAP];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int64_t pair_idx = (int64_t)blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
    if (pair_idx >= num_pairs) return;

    int32_t* my_s_b = s_b + warp_in_block * SMEM_B_CAP;

    int32_t u = first[pair_idx];
    int32_t v = second[pair_idx];
    int32_t u_start = __ldg(offsets + u);
    int32_t u_end = __ldg(offsets + u + 1);
    int32_t v_start = __ldg(offsets + v);
    int32_t v_end = __ldg(offsets + v + 1);

    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) scores[pair_idx] = 0.0;
        return;
    }

    const int32_t* a_nbrs;
    const int32_t* b_nbrs;
    const double* a_wts;
    const double* b_wts;
    int32_t a_size;
    int32_t b_size;

    if (u_deg <= v_deg) {
        a_nbrs = indices + u_start; a_wts = edge_weights + u_start; a_size = u_deg;
        b_nbrs = indices + v_start; b_wts = edge_weights + v_start; b_size = v_deg;
    } else {
        a_nbrs = indices + v_start; a_wts = edge_weights + v_start; a_size = v_deg;
        b_nbrs = indices + u_start; b_wts = edge_weights + u_start; b_size = u_deg;
    }

    bool b_cached = (b_size <= SMEM_B_CAP);
    if (b_cached) {
        for (int32_t i = lane; i < b_size; i += WARP_SIZE) {
            my_s_b[i] = __ldg(b_nbrs + i);
        }
        __syncwarp();
    }

    int32_t b_min = b_cached ? my_s_b[0] : __ldg(b_nbrs);
    int32_t b_max = b_cached ? my_s_b[b_size - 1] : __ldg(b_nbrs + b_size - 1);

    int32_t a_begin = lb_gmem(a_nbrs, 0, a_size, b_min);
    int32_t a_end_eff = a_size;
    if (a_begin < a_size && __ldg(a_nbrs + a_size - 1) > b_max) {
        a_end_eff = lb_gmem(a_nbrs, a_begin, a_size, b_max + 1);
    }

    if (a_begin >= a_end_eff) {
        if (lane == 0) scores[pair_idx] = 0.0;
        return;
    }

    double local_sum = 0.0;
    int32_t search_lo = 0;

    for (int32_t idx = a_begin + lane; idx < a_end_eff; idx += WARP_SIZE) {
        int32_t a_val = __ldg(a_nbrs + idx);

        int32_t rank = 0;
        if (is_multigraph) {
            rank = idx - lb_gmem(a_nbrs, 0, a_size, a_val);
        }

        int32_t lb;
        if (b_cached) {
            lb = lb_smem(my_s_b, search_lo, b_size, a_val);
        } else {
            lb = lb_gmem(b_nbrs, search_lo, b_size, a_val);
        }
        search_lo = lb;

        int32_t match_pos = lb + rank;

        bool match;
        if (b_cached) {
            match = (match_pos < b_size && my_s_b[match_pos] == a_val);
        } else {
            match = (match_pos < b_size && __ldg(b_nbrs + match_pos) == a_val);
        }

        if (match) {
            local_sum += fmin(__ldg(a_wts + idx), __ldg(b_wts + match_pos));
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    if (lane == 0) {
        double min_ws = fmin(weight_sums[u], weight_sums[v]);
        scores[pair_idx] = (min_ws > 0.0) ? (local_sum / min_ws) : 0.0;
    }
}




__global__ void overlap_thread_inline_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ scores,
    int64_t num_pairs)
{
    int64_t pair_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= num_pairs) return;

    int32_t u = first[pair_idx];
    int32_t v = second[pair_idx];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];

    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        scores[pair_idx] = 0.0;
        return;
    }

    double wu_sum = 0.0;
    for (int32_t i = u_start; i < u_end; i++) wu_sum += edge_weights[i];
    double wv_sum = 0.0;
    for (int32_t i = v_start; i < v_end; i++) wv_sum += edge_weights[i];

    const int32_t* u_nbrs = indices + u_start;
    const int32_t* v_nbrs = indices + v_start;
    const double* u_wts = edge_weights + u_start;
    const double* v_wts = edge_weights + v_start;

    int32_t i = lb_gmem(u_nbrs, 0, u_deg, __ldg(v_nbrs));
    if (i >= u_deg) { scores[pair_idx] = 0.0; return; }
    int32_t j = lb_gmem(v_nbrs, 0, v_deg, __ldg(u_nbrs + i));
    if (j >= v_deg) { scores[pair_idx] = 0.0; return; }

    int32_t u_last = __ldg(u_nbrs + u_deg - 1);
    int32_t v_last = __ldg(v_nbrs + v_deg - 1);
    int32_t max_val = (u_last < v_last) ? u_last : v_last;
    int32_t u_eff = u_deg, v_eff = v_deg;
    if (u_last > max_val) u_eff = lb_gmem(u_nbrs, i, u_deg, max_val + 1);
    if (v_last > max_val) v_eff = lb_gmem(v_nbrs, j, v_deg, max_val + 1);

    double iw = 0.0;
    while (i < u_eff && j < v_eff) {
        int32_t un = __ldg(u_nbrs + i);
        int32_t vn = __ldg(v_nbrs + j);
        if (un == vn) {
            iw += fmin(__ldg(u_wts + i), __ldg(v_wts + j));
            i++; j++;
        } else if (un < vn) {
            i++;
        } else {
            j++;
        }
    }

    double min_ws = fmin(wu_sum, wv_sum);
    scores[pair_idx] = (min_ws > 0.0) ? (iw / min_ws) : 0.0;
}

}  

void overlap_similarity(const graph32_t& graph,
                        const double* edge_weights,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        double* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    bool is_multigraph = graph.is_multigraph;

    int avg_degree = (num_vertices > 0) ? (num_edges / num_vertices) : 0;

    if (avg_degree <= 6 && !is_multigraph) {
        int block = 256;
        int grid = ((int)num_pairs + block - 1) / block;
        overlap_thread_inline_kernel<<<grid, block>>>(
            offsets, indices, edge_weights,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int64_t)num_pairs);
    } else {
        cache.ensure((int64_t)num_vertices);

        int ws_block = 256;
        int warps_per_block = ws_block / WARP_SIZE;
        int ws_grid = (num_vertices + warps_per_block - 1) / warps_per_block;
        compute_weight_sums_warp_kernel<<<ws_grid, ws_block>>>(
            offsets, edge_weights, cache.weight_sums, num_vertices);

        int grid = ((int)num_pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        overlap_warp_kernel<<<grid, BLOCK_SIZE>>>(
            offsets, indices, edge_weights, cache.weight_sums,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int64_t)num_pairs, is_multigraph);
    }
}

}  
