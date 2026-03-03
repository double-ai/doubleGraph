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

#define SMALL_THRESH 32

struct Cache : Cacheable {};






__device__ __forceinline__
int cooperative_merge_intersect(
    const int* __restrict__ a, int size_a,
    const int* __restrict__ b, int size_b,
    int lane_id
) {
    int local_count = 0;
    int a_ptr = 0, b_ptr = 0;
    int b_reg = 0x7fffffff;
    int b_loaded_to = 0;

    while (a_ptr < size_a && b_ptr < size_b) {
        
        int a_n = size_a - a_ptr;
        if (a_n > 32) a_n = 32;
        int a_reg = (lane_id < a_n) ? __ldg(&a[a_ptr + lane_id]) : 0x7fffffff;

        
        if (b_ptr >= b_loaded_to) {
            int b_n = size_b - b_ptr;
            if (b_n > 32) b_n = 32;
            b_reg = (lane_id < b_n) ? __ldg(&b[b_ptr + lane_id]) : 0x7fffffff;
            b_loaded_to = b_ptr + b_n;
        }
        int b_n = b_loaded_to - b_ptr;

        int a_max = __shfl_sync(0xffffffff, a_reg, a_n - 1);
        int b_max = __shfl_sync(0xffffffff, b_reg, b_n - 1);

        
        
        {
            int my_a = a_reg;
            int lo = 0, hi = b_n;

            
            int mid;
            mid = (lo + hi) >> 1; { int bm = __shfl_sync(0xffffffff, b_reg, mid); if (bm < my_a) lo = mid+1; else hi = mid; }
            mid = (lo + hi) >> 1; { int bm = __shfl_sync(0xffffffff, b_reg, mid); if (bm < my_a) lo = mid+1; else hi = mid; }
            mid = (lo + hi) >> 1; { int bm = __shfl_sync(0xffffffff, b_reg, mid); if (bm < my_a) lo = mid+1; else hi = mid; }
            mid = (lo + hi) >> 1; { int bm = __shfl_sync(0xffffffff, b_reg, mid); if (bm < my_a) lo = mid+1; else hi = mid; }
            mid = (lo + hi) >> 1; { int bm = __shfl_sync(0xffffffff, b_reg, mid); if (bm < my_a) lo = mid+1; else hi = mid; }
            mid = (lo + hi) >> 1; { int bm = __shfl_sync(0xffffffff, b_reg, mid); if (bm < my_a) lo = mid+1; else hi = mid; }

            
            int safe_lo = (lo < 32) ? lo : 0;
            int b_found = __shfl_sync(0xffffffff, b_reg, safe_lo);
            if (lane_id < a_n && lo < b_n && b_found == my_a) {
                local_count++;
            }
        }

        
        if (a_max <= b_max) a_ptr += a_n;
        if (b_max <= a_max) { b_ptr = b_loaded_to; }
    }

    return local_count;
}


__device__ __forceinline__
int binary_search_intersect(
    const int* __restrict__ a, int size_a,
    const int* __restrict__ b, int size_b,
    int lane_id
) {
    int local_count = 0;
    for (int i = lane_id; i < size_a; i += 32) {
        int target = __ldg(&a[i]);
        int lo = 0, hi = size_b;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (__ldg(&b[mid]) < target) lo = mid + 1;
            else hi = mid;
        }
        if (lo < size_b && __ldg(&b[lo]) == target) {
            local_count++;
        }
    }
    return local_count;
}


__device__ __forceinline__
int multigraph_intersect(
    const int* __restrict__ a, int size_a,
    const int* __restrict__ b, int size_b,
    int lane_id
) {
    int local_count = 0;
    for (int i = lane_id; i < size_a; i += 32) {
        int target = __ldg(&a[i]);
        int lo = 0, hi = size_a;
        while (lo < hi) { int mid = lo + ((hi-lo)>>1); if (__ldg(&a[mid]) < target) lo = mid+1; else hi = mid; }
        int rank = i - lo;
        lo = 0; hi = size_b;
        while (lo < hi) { int mid = lo + ((hi-lo)>>1); if (__ldg(&b[mid]) < target) lo = mid+1; else hi = mid; }
        int pos_b = lo + rank;
        if (pos_b < size_b && __ldg(&b[pos_b]) == target) local_count++;
    }
    return local_count;
}

template<bool IS_MULTIGRAPH>
__global__ void __launch_bounds__(256)
jaccard_grouped(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const int* __restrict__ pairs_first,
    const int* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    const int lane_id = threadIdx.x & 31;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int64_t base_pair = (int64_t)warp_id * 32;

    if (base_pair >= num_pairs) return;

    int64_t my_pair = base_pair + lane_id;
    int u_start = 0, u_end = 0, v_start = 0, v_end = 0;
    int deg_u = 0, deg_v = 0;
    bool valid = (my_pair < num_pairs);

    if (valid) {
        int u = __ldg(&pairs_first[my_pair]);
        int v = __ldg(&pairs_second[my_pair]);
        u_start = __ldg(&offsets[u]);
        u_end = __ldg(&offsets[u + 1]);
        v_start = __ldg(&offsets[v]);
        v_end = __ldg(&offsets[v + 1]);
        deg_u = u_end - u_start;
        deg_v = v_end - v_start;
    }

    int max_deg = (deg_u > deg_v) ? deg_u : deg_v;

    
    int group_max = max_deg;
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        int other = __shfl_xor_sync(0xffffffff, group_max, s);
        group_max = (other > group_max) ? other : group_max;
    }

    if (group_max < SMALL_THRESH) {
        
        if (valid) {
            int union_size = deg_u + deg_v;
            if (union_size == 0 || deg_u == 0 || deg_v == 0) {
                scores[my_pair] = 0.0f;
            } else {
                const int* a = indices + u_start;
                const int* b = indices + v_start;

                int count = 0, i = 0, j = 0;
                while (i < deg_u && j < deg_v) {
                    int va = __ldg(&a[i]), vb = __ldg(&b[j]);
                    if (va == vb) { count++; i++; j++; }
                    else if (va < vb) { i++; }
                    else { j++; }
                }

                union_size -= count;
                scores[my_pair] = (union_size > 0) ? ((float)count / (float)union_size) : 0.0f;
            }
        }
    } else {
        
        int num_in_group = 32;
        if (base_pair + 32 > num_pairs) num_in_group = (int)(num_pairs - base_pair);

        for (int p = 0; p < num_in_group; p++) {
            int p_u_start = __shfl_sync(0xffffffff, u_start, p);
            int p_u_end = __shfl_sync(0xffffffff, u_end, p);
            int p_v_start = __shfl_sync(0xffffffff, v_start, p);
            int p_v_end = __shfl_sync(0xffffffff, v_end, p);
            int p_deg_u = p_u_end - p_u_start;
            int p_deg_v = p_v_end - p_v_start;
            int64_t pair_idx = base_pair + p;

            int union_size = p_deg_u + p_deg_v;
            if (union_size == 0) {
                if (lane_id == 0) scores[pair_idx] = 0.0f;
                continue;
            }

            const int* a, *b;
            int size_a, size_b;
            if (p_deg_u <= p_deg_v) {
                a = indices + p_u_start; size_a = p_deg_u;
                b = indices + p_v_start; size_b = p_deg_v;
            } else {
                a = indices + p_v_start; size_a = p_deg_v;
                b = indices + p_u_start; size_b = p_deg_u;
            }

            if (size_a == 0) {
                if (lane_id == 0) scores[pair_idx] = 0.0f;
                continue;
            }

            int local_count;

            if constexpr (!IS_MULTIGRAPH) {
                
                if (size_b <= 1024 && size_a * 8 >= size_b) {
                    
                    local_count = cooperative_merge_intersect(a, size_a, b, size_b, lane_id);
                } else {
                    
                    local_count = binary_search_intersect(a, size_a, b, size_b, lane_id);
                }
            } else {
                local_count = multigraph_intersect(a, size_a, b, size_b, lane_id);
            }

            
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                local_count += __shfl_down_sync(0xffffffff, local_count, offset);
            }

            if (lane_id == 0) {
                int intersection = local_count;
                int final_union = union_size - intersection;
                scores[pair_idx] = (final_union > 0) ? ((float)intersection / (float)final_union) : 0.0f;
            }
        }
    }
}

}  

void jaccard_similarity_seg(const graph32_t& graph,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    bool is_multigraph = graph.is_multigraph;

    int threads_per_block = 256;
    int64_t pairs_per_block = threads_per_block;
    int64_t np = static_cast<int64_t>(num_pairs);
    int grid = static_cast<int>((np + pairs_per_block - 1) / pairs_per_block);

    if (is_multigraph) {
        jaccard_grouped<true><<<grid, threads_per_block>>>(
            offsets, indices, vertex_pairs_first, vertex_pairs_second,
            similarity_scores, np);
    } else {
        jaccard_grouped<false><<<grid, threads_per_block>>>(
            offsets, indices, vertex_pairs_first, vertex_pairs_second,
            similarity_scores, np);
    }
}

}  
