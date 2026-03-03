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

namespace aai {

namespace {

struct Cache : Cacheable {
    float* weight_sums = nullptr;
    int64_t weight_sums_capacity = 0;

    void ensure(int32_t num_vertices) {
        int64_t needed = static_cast<int64_t>(num_vertices);
        if (weight_sums_capacity < needed) {
            if (weight_sums) cudaFree(weight_sums);
            cudaMalloc(&weight_sums, needed * sizeof(float));
            weight_sums_capacity = needed;
        }
    }

    ~Cache() override {
        if (weight_sums) cudaFree(weight_sums);
    }
};

__global__ void compute_weight_sums_kernel(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ edge_weights,
    float* __restrict__ weight_sums,
    int32_t num_vertices
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= num_vertices) return;
    int32_t start = offsets[warp_id];
    int32_t end = offsets[warp_id + 1];
    float sum = 0.0f;
    for (int32_t i = start + lane; i < end; i += 32)
        sum += edge_weights[i];
    for (int s = 16; s > 0; s >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, s);
    if (lane == 0) weight_sums[warp_id] = sum;
}



__device__ __forceinline__ void merge_path_partition(
    const int32_t* __restrict__ A, int na,
    const int32_t* __restrict__ B, int nb,
    int diag, int& ia, int& ib
) {
    int lo = (diag > nb) ? diag - nb : 0;
    int hi = (diag < na) ? diag : na;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int jb = diag - mid - 1;
        if (jb >= 0 && A[mid] > B[jb]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    ia = lo;
    ib = diag - lo;
}

template<bool IS_MULTIGRAPH>
__device__ __forceinline__ void overlap_pair(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ weight_sums,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs,
    int64_t pair_idx,
    int lane
) {
    if (pair_idx >= num_pairs) return;
    const int32_t u = pairs_first[pair_idx];
        const int32_t v = pairs_second[pair_idx];

        if (u == v) {
            if (lane == 0) {
                float ws = weight_sums[u];
                scores[pair_idx] = (ws > 0.0f) ? 1.0f : 0.0f;
            }
            return;
        }

        const float u_wsum = weight_sums[u];
        const float v_wsum = weight_sums[v];
        const float min_wsum = fminf(u_wsum, v_wsum);

        if (min_wsum == 0.0f) {
            if (lane == 0) scores[pair_idx] = 0.0f;
            return;
        }

        const int32_t u_start = offsets[u];
        const int32_t u_end = offsets[u + 1];
        const int32_t v_start = offsets[v];
        const int32_t v_end = offsets[v + 1];
        const int32_t u_deg = u_end - u_start;
        const int32_t v_deg = v_end - v_start;

        
        int32_t a_off, a_deg, b_off, b_deg;
        if (u_deg <= v_deg) {
            a_off = u_start; a_deg = u_deg;
            b_off = v_start; b_deg = v_deg;
        } else {
            a_off = v_start; a_deg = v_deg;
            b_off = u_start; b_deg = u_deg;
        }

        const int32_t* a_idx = indices + a_off;
        const float* a_wt = edge_weights + a_off;
        const int32_t* b_idx = indices + b_off;
        const float* b_wt = edge_weights + b_off;

        float isect_sum = 0.0f;

        
        
        
        const int total = a_deg + b_deg;

        if constexpr (!IS_MULTIGRAPH) {
            if (total > 96 && b_deg < a_deg * 8) {
                
                
                int step = (total + 31) >> 5;  
                int diag_start = lane * step;
                int diag_end = diag_start + step;
                if (diag_end > total) diag_end = total;

                int ia_start, ib_start, ia_end, ib_end;
                merge_path_partition(a_idx, a_deg, b_idx, b_deg, diag_start, ia_start, ib_start);
                merge_path_partition(a_idx, a_deg, b_idx, b_deg, diag_end, ia_end, ib_end);

                
                int ia = ia_start, ib = ib_start;
                while (ia < ia_end && ib < ib_end) {
                    int32_t va = a_idx[ia];
                    int32_t vb = b_idx[ib];
                    if (va == vb) {
                        isect_sum += fminf(a_wt[ia], b_wt[ib]);
                        ia++; ib++;
                    } else if (va < vb) {
                        ia++;
                    } else {
                        ib++;
                    }
                }

                
                
                
                if (lane > 0 && ia_start > 0 && ib_start < b_deg) {
                    if (a_idx[ia_start - 1] == b_idx[ib_start]) {
                        isect_sum += fminf(a_wt[ia_start - 1], b_wt[ib_start]);
                    }
                }
            } else {
                
                int search_lo = 0;
                for (int i = lane; i < a_deg; i += 32) {
                    int32_t target = a_idx[i];
                    float wa = a_wt[i];
                    int lo = search_lo, hi = b_deg;
                    while (lo < hi) {
                        int mid = (lo + hi) >> 1;
                        if (b_idx[mid] < target) lo = mid + 1;
                        else hi = mid;
                    }
                    if (lo < b_deg && b_idx[lo] == target) {
                        isect_sum += fminf(wa, b_wt[lo]);
                    }
                    search_lo = lo;
                }
            }
        } else {
            
            int search_lo = 0;
            for (int i = lane; i < a_deg; i += 32) {
                int32_t target = a_idx[i];
                float wa = a_wt[i];
                int lo = search_lo, hi = b_deg;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (b_idx[mid] < target) lo = mid + 1;
                    else hi = mid;
                }
                if (lo < b_deg && b_idx[lo] == target) {
                    int lb_a = 0, hi_a = i;
                    while (lb_a < hi_a) {
                        int mid = (lb_a + hi_a) >> 1;
                        if (a_idx[mid] < target) lb_a = mid + 1;
                        else hi_a = mid;
                    }
                    int rank = i - lb_a;
                    int count_b = 0;
                    for (int j = lo; j < b_deg && b_idx[j] == target; j++) count_b++;
                    if (rank < count_b)
                        isect_sum += fminf(wa, b_wt[lo + rank]);
                }
                search_lo = lo;
            }
        }

        
        for (int s = 16; s > 0; s >>= 1)
            isect_sum += __shfl_down_sync(0xffffffff, isect_sum, s);

        if (lane == 0) {
            scores[pair_idx] = isect_sum / min_wsum;
        }
    }


template<bool IS_MULTIGRAPH, int UNROLL>
__global__ __launch_bounds__(256)
void overlap_kernel_unroll(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ weight_sums,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    const int lane = threadIdx.x & 31;
    const int warp_group = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int64_t base = (int64_t)warp_group * (int64_t)UNROLL;
    if (base >= num_pairs) return;
    #pragma unroll
    for (int r = 0; r < UNROLL; ++r) {
        overlap_pair<IS_MULTIGRAPH>(offsets, indices, edge_weights, weight_sums,
                                    pairs_first, pairs_second, scores,
                                    num_pairs, base + r, lane);
    }
}

}  

void overlap_similarity_seg(const graph32_t& graph,
                            const float* edge_weights,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    bool is_multigraph = graph.is_multigraph;

    
    cache.ensure(num_vertices);

    if (num_vertices > 0) {
        int threads = 256;
        int blocks = (num_vertices + threads / 32 - 1) / (threads / 32);
        compute_weight_sums_kernel<<<blocks, threads>>>(offsets, edge_weights,
                                                         cache.weight_sums, num_vertices);
    }

    if (num_pairs > 0) {
        const int threads = 256;
        const int warps_per_block = threads / 32;
        constexpr int UNROLL = 2;
        const int pairs_per_block = warps_per_block * UNROLL;
        const int blocks = (int)((num_pairs + pairs_per_block - 1) / pairs_per_block);
        if (is_multigraph)
            overlap_kernel_unroll<true, UNROLL><<<blocks, threads>>>(
                offsets, indices, edge_weights, cache.weight_sums,
                vertex_pairs_first, vertex_pairs_second, similarity_scores,
                static_cast<int64_t>(num_pairs));
        else
            overlap_kernel_unroll<false, UNROLL><<<blocks, threads>>>(
                offsets, indices, edge_weights, cache.weight_sums,
                vertex_pairs_first, vertex_pairs_second, similarity_scores,
                static_cast<int64_t>(num_pairs));
    }
}

}  
