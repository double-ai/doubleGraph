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
#include <climits>

namespace aai {

namespace {

struct Cache : Cacheable {};

static constexpr int THREADS_PER_BLOCK = 256;
static constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;


static constexpr int SMEM_PER_WARP = 384; 

__device__ __forceinline__ int32_t ld_g(const int32_t* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ int lower_bound_dev(const int32_t* __restrict__ arr, int len, int32_t val) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int32_t m = ld_g(arr + mid);
        if (m < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int upper_bound_dev(const int32_t* __restrict__ arr, int len, int32_t val) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int32_t m = ld_g(arr + mid);
        if (m <= val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int lower_bound_smem(const int32_t* arr, int len, int32_t val) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int32_t m = arr[mid];
        if (m < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int upper_bound_smem(const int32_t* arr, int len, int32_t val) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int32_t m = arr[mid];
        if (m <= val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int warp_reduce_sum(int v) {
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    return v;
}




__device__ __forceinline__ int intersect_ballot_simple_lane0(
    const int32_t* __restrict__ small_ptr, int small_size,
    const int32_t* __restrict__ large_ptr, int large_size,
    int lane)
{
    int32_t my_large = (lane < large_size) ? ld_g(large_ptr + lane) : INT_MAX;
    int32_t my_small = (lane < small_size) ? ld_g(small_ptr + lane) : INT_MAX;

    int count = 0;
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        if (k >= small_size) break;
        int32_t target = __shfl_sync(0xffffffff, my_small, k);
        uint32_t m = __ballot_sync(0xffffffff, my_large == target);
        if (lane == 0) count += (m != 0);
    }
    return count;
}


__device__ __forceinline__ int intersect_ballot_multiset(
    const int32_t* __restrict__ small_ptr, int small_size,
    const int32_t* __restrict__ large_ptr, int large_size,
    int lane)
{
    
    int32_t my_large = (lane < large_size) ? ld_g(large_ptr + lane) : INT_MAX;
    int32_t my_small = (lane < small_size) ? ld_g(small_ptr + lane) : INT_MAX;

    uint32_t avail = (large_size == 32) ? 0xffffffffu : ((1u << large_size) - 1u);
    int count = 0;

    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        if (k >= small_size) break;
        int32_t target = __shfl_sync(0xffffffff, my_small, k);
        uint32_t m = __ballot_sync(0xffffffff, my_large == target) & avail;
        if (m) {
            
            avail ^= (m & (0u - m));
            ++count;
        }
    }
    return count;
}


__device__ __forceinline__ int binary_search_found(const int32_t* __restrict__ arr, int len, int32_t val) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int32_t m = ld_g(arr + mid);
        if (m < val) lo = mid + 1;
        else if (m > val) hi = mid;
        else return 1;
    }
    return 0;
}

__device__ __forceinline__ int intersect_warp_ilp2(
    const int32_t* __restrict__ small_ptr, int small_size,
    const int32_t* __restrict__ large_ptr, int large_size,
    int lane)
{
    
    
    int32_t large_first = ld_g(large_ptr);
    int32_t large_last = ld_g(large_ptr + (large_size - 1));
    int32_t small_first = ld_g(small_ptr);
    int32_t small_last = ld_g(small_ptr + (small_size - 1));
    if (small_last < large_first || large_last < small_first) return 0;

    int start = 0;
    int end = small_size;
    if (small_first < large_first) {
        start = lower_bound_dev(small_ptr, small_size, large_first);
        if (start >= end) return 0;
    }
    if (small_last > large_last) {
        end = upper_bound_dev(small_ptr, small_size, large_last);
        if (start >= end) return 0;
    }
    small_ptr += start;
    small_size = end - start;

    int local = 0;
    int i = lane;

    for (; i + 32 < small_size; i += 64) {
        int32_t target_a = ld_g(small_ptr + i);
        int32_t target_b = ld_g(small_ptr + (i + 32));

        int lo_a = 0, hi_a = large_size;
        int lo_b = 0, hi_b = large_size;

        while (lo_a < hi_a && lo_b < hi_b) {
            int mid_a = (lo_a + hi_a) >> 1;
            int mid_b = (lo_b + hi_b) >> 1;
            int32_t val_a = ld_g(large_ptr + mid_a);
            int32_t val_b = ld_g(large_ptr + mid_b);
            if (val_a < target_a) lo_a = mid_a + 1; else hi_a = mid_a;
            if (val_b < target_b) lo_b = mid_b + 1; else hi_b = mid_b;
        }
        while (lo_a < hi_a) {
            int mid = (lo_a + hi_a) >> 1;
            if (ld_g(large_ptr + mid) < target_a) lo_a = mid + 1; else hi_a = mid;
        }
        while (lo_b < hi_b) {
            int mid = (lo_b + hi_b) >> 1;
            if (ld_g(large_ptr + mid) < target_b) lo_b = mid + 1; else hi_b = mid;
        }

        if (lo_a < large_size && ld_g(large_ptr + lo_a) == target_a) ++local;
        if (lo_b < large_size && ld_g(large_ptr + lo_b) == target_b) ++local;
    }

    if (i < small_size) {
        int32_t target = ld_g(small_ptr + i);
        local += binary_search_found(large_ptr, large_size, target);
    }

    return local;
}


__device__ __forceinline__ int intersect_bsearch_smem_simple(
    const int32_t* __restrict__ small_ptr, int small_size,
    const int32_t* __restrict__ large_smem, int large_size,
    int lane)
{
    int32_t large_first = large_smem[0];
    int32_t large_last = large_smem[large_size - 1];
    int32_t small_first = ld_g(small_ptr);
    int32_t small_last = ld_g(small_ptr + (small_size - 1));
    if (small_last < large_first || large_last < small_first) return 0;

    int local = 0;
    for (int i = lane; i < small_size; i += 32) {
        int32_t x = ld_g(small_ptr + i);
        if (x < large_first || x > large_last) continue;
        int lb = lower_bound_smem(large_smem, large_size, x);
        local += (lb < large_size && large_smem[lb] == x);
    }
    return warp_reduce_sum(local);
}

__device__ __forceinline__ int intersect_bsearch_smem_multigraph(
    const int32_t* __restrict__ small_ptr, int small_size,
    const int32_t* __restrict__ large_smem, int large_size,
    int lane)
{
    int32_t large_first = large_smem[0];
    int32_t large_last = large_smem[large_size - 1];
    int32_t small_first = ld_g(small_ptr);
    int32_t small_last = ld_g(small_ptr + (small_size - 1));
    if (small_last < large_first || large_last < small_first) return 0;

    int local = 0;
    for (int i = lane; i < small_size; i += 32) {
        int32_t x = ld_g(small_ptr + i);
        if (x < large_first || x > large_last) continue;

        int lb = lower_bound_smem(large_smem, large_size, x);
        if (lb < large_size && large_smem[lb] == x) {
            if (i > 0 && ld_g(small_ptr + (i - 1)) == x) {
                int first_in_small = lower_bound_dev(small_ptr, small_size, x);
                int rank = i - first_in_small;
                int ub = upper_bound_smem(large_smem, large_size, x);
                if (rank < (ub - lb)) ++local;
            } else {
                ++local;
            }
        }
    }
    return warp_reduce_sum(local);
}


__device__ __forceinline__ int intersect_warp_multigraph_rank(
    const int32_t* __restrict__ small_ptr, int small_size,
    const int32_t* __restrict__ large_ptr, int large_size,
    int lane)
{
    int local = 0;
    for (int i = lane; i < small_size; i += 32) {
        int32_t x = ld_g(small_ptr + i);
        int lb_s = lower_bound_dev(small_ptr, small_size, x);
        int rank_s = i - lb_s;

        int lb_l = lower_bound_dev(large_ptr, large_size, x);
        if (lb_l < large_size && ld_g(large_ptr + lb_l) == x) {
            int ub_l = upper_bound_dev(large_ptr, large_size, x);
            if (rank_s < (ub_l - lb_l)) ++local;
        }
    }
    return local;
}


template <bool MULTIGRAPH>
__global__ __launch_bounds__(THREADS_PER_BLOCK, 6)
void sorensen_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int num_pairs)
{
    extern __shared__ int32_t smem[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int total_warps = (gridDim.x * blockDim.x) >> 5;

    int32_t* warp_smem = smem + warp_in_block * SMEM_PER_WARP;

    for (int pid = warp_global; pid < num_pairs; pid += total_warps) {
        int u = ld_g(first + pid);
        int v = ld_g(second + pid);

        int u_start = ld_g(offsets + u);
        int u_end = ld_g(offsets + (u + 1));
        int v_start = ld_g(offsets + v);
        int v_end = ld_g(offsets + (v + 1));

        int deg_u = u_end - u_start;
        int deg_v = v_end - v_start;
        int deg_sum = deg_u + deg_v;

        if (deg_sum == 0 || deg_u == 0 || deg_v == 0) {
            if (lane == 0) scores[pid] = 0.0f;
            continue;
        }

        const int32_t* small_ptr;
        const int32_t* large_ptr;
        int small_size;
        int large_size;
        if (deg_u <= deg_v) {
            small_ptr = indices + u_start;
            small_size = deg_u;
            large_ptr = indices + v_start;
            large_size = deg_v;
        } else {
            small_ptr = indices + v_start;
            small_size = deg_v;
            large_ptr = indices + u_start;
            large_size = deg_u;
        }

        int count = 0;

        if (large_size <= 32) {
            if constexpr (MULTIGRAPH) {
                count = intersect_ballot_multiset(small_ptr, small_size, large_ptr, large_size, lane);
            } else {
                
                count = intersect_ballot_simple_lane0(small_ptr, small_size, large_ptr, large_size, lane);
            }
        } else if (large_size <= SMEM_PER_WARP) {
            
            int iters = (large_size + 31) >> 5;
            #pragma unroll
            for (int t = 0; t < 12; ++t) {  
                if (t >= iters) break;
                int idx = lane + (t << 5);
                if (idx < large_size) warp_smem[idx] = ld_g(large_ptr + idx);
            }
            if constexpr (MULTIGRAPH) {
                count = intersect_bsearch_smem_multigraph(small_ptr, small_size, warp_smem, large_size, lane);
            } else {
                count = intersect_bsearch_smem_simple(small_ptr, small_size, warp_smem, large_size, lane);
            }
        } else {
            int local;
            if constexpr (MULTIGRAPH) {
                local = intersect_warp_multigraph_rank(small_ptr, small_size, large_ptr, large_size, lane);
            } else {
                local = intersect_warp_ilp2(small_ptr, small_size, large_ptr, large_size, lane);
            }
            count = warp_reduce_sum(local);
        }

        if (lane == 0) {
            scores[pid] = (2.0f * (float)count) / (float)deg_sum;
        }
    }
}

}  

void sorensen_similarity(const graph32_t& graph,
                         const int32_t* vertex_pairs_first,
                         const int32_t* vertex_pairs_second,
                         std::size_t num_pairs,
                         float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    if (num_pairs == 0) return;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    bool is_multigraph = graph.is_multigraph;

    int grid = ((int)num_pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    if (grid > 4096) grid = 4096; 
    int smem_bytes = WARPS_PER_BLOCK * SMEM_PER_WARP * (int)sizeof(int32_t);

    if (is_multigraph) {
        sorensen_warp_kernel<true><<<grid, THREADS_PER_BLOCK, smem_bytes>>>(
            offsets, indices, vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int)num_pairs);
    } else {
        sorensen_warp_kernel<false><<<grid, THREADS_PER_BLOCK, smem_bytes>>>(
            offsets, indices, vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int)num_pairs);
    }
}

}  
