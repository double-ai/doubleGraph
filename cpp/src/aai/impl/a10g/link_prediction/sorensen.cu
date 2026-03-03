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

#define SMEM_PER_WARP 384

__device__ __forceinline__ int lower_bound_smem(const int32_t* arr, int size, int32_t target) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int upper_bound_smem(const int32_t* arr, int size, int32_t target) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] <= target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int lower_bound_dev(const int32_t* __restrict__ arr, int size, int32_t target) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int upper_bound_dev(const int32_t* __restrict__ arr, int size, int32_t target) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] <= target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int intersect_ballot(
    const int32_t* __restrict__ small_ptr, int small_size,
    const int32_t* __restrict__ large_ptr, int large_size,
    int lane
) {
    int32_t my_large = (lane < large_size) ? large_ptr[lane] : INT_MAX;
    int32_t my_small = (lane < small_size) ? small_ptr[lane] : INT_MAX;

    uint32_t avail = (large_size >= 32) ? 0xFFFFFFFFu : ((1u << large_size) - 1u);
    int count = 0;

    for (int k = 0; k < small_size; k++) {
        int32_t target = __shfl_sync(0xFFFFFFFF, my_small, k);
        uint32_t ballot = __ballot_sync(0xFFFFFFFF, my_large == target) & avail;
        if (ballot != 0) {
            avail ^= (ballot & (0u - ballot));
            count++;
        }
    }
    return count;
}

__device__ __forceinline__ int intersect_bsearch_smem(
    const int32_t* __restrict__ small_ptr, int small_size,
    const int32_t* large_smem, int large_size,
    int lane
) {
    int32_t large_first = large_smem[0];
    int32_t large_last = large_smem[large_size - 1];
    int32_t small_first = small_ptr[0];
    int32_t small_last = small_ptr[small_size - 1];

    if (small_last < large_first || large_last < small_first) return 0;

    int local_count = 0;
    for (int i = lane; i < small_size; i += 32) {
        int32_t target = small_ptr[i];
        if (target < large_first || target > large_last) continue;

        int lb = lower_bound_smem(large_smem, large_size, target);
        if (lb < large_size && large_smem[lb] == target) {
            if (i > 0 && small_ptr[i - 1] == target) {
                int first_in_small = lower_bound_dev(small_ptr, small_size, target);
                int rank = i - first_in_small;
                int ub = upper_bound_smem(large_smem, large_size, target);
                if (rank < ub - lb) local_count++;
            } else {
                local_count++;
            }
        }
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, s);
    }
    return local_count;
}

__device__ __forceinline__ int intersect_bsearch_global(
    const int32_t* __restrict__ small_ptr, int small_size,
    const int32_t* __restrict__ large_ptr, int large_size,
    int lane
) {
    int32_t large_first = large_ptr[0];
    int32_t large_last = large_ptr[large_size - 1];
    int32_t small_first = small_ptr[0];
    int32_t small_last = small_ptr[small_size - 1];

    if (small_last < large_first || large_last < small_first) return 0;

    int local_count = 0;
    for (int i = lane; i < small_size; i += 32) {
        int32_t target = small_ptr[i];
        if (target < large_first || target > large_last) continue;

        int lb = lower_bound_dev(large_ptr, large_size, target);
        if (lb < large_size && large_ptr[lb] == target) {
            if (i > 0 && small_ptr[i - 1] == target) {
                int first_in_small = lower_bound_dev(small_ptr, small_size, target);
                int rank = i - first_in_small;
                int ub = upper_bound_dev(large_ptr, large_size, target);
                if (rank < ub - lb) local_count++;
            } else {
                local_count++;
            }
        }
    }

    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
        local_count += __shfl_down_sync(0xFFFFFFFF, local_count, s);
    }
    return local_count;
}

__global__ void sorensen_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    extern __shared__ int32_t smem[];

    const int lane = threadIdx.x & 31;
    const int warp_in_block = (threadIdx.x >> 5);
    const int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int64_t total_warps = ((int64_t)gridDim.x * blockDim.x) >> 5;

    int32_t* my_smem = smem + warp_in_block * SMEM_PER_WARP;

    for (int64_t pid = warp_id; pid < num_pairs; pid += total_warps) {
        int32_t u = first[pid];
        int32_t v = second[pid];

        int32_t u_start = offsets[u];
        int32_t u_end = offsets[u + 1];
        int32_t v_start = offsets[v];
        int32_t v_end = offsets[v + 1];

        int32_t deg_u = u_end - u_start;
        int32_t deg_v = v_end - v_start;
        int32_t deg_sum = deg_u + deg_v;

        if (deg_sum == 0 || deg_u == 0 || deg_v == 0) {
            if (lane == 0) scores[pid] = 0.0f;
            continue;
        }

        const int32_t* small_ptr = (deg_u <= deg_v) ? (indices + u_start) : (indices + v_start);
        const int32_t* large_ptr = (deg_u <= deg_v) ? (indices + v_start) : (indices + u_start);
        int32_t small_size = (deg_u <= deg_v) ? deg_u : deg_v;
        int32_t large_size = (deg_u <= deg_v) ? deg_v : deg_u;

        int count;
        if (large_size <= 32) {
            count = intersect_ballot(small_ptr, small_size, large_ptr, large_size, lane);
        } else if (large_size <= SMEM_PER_WARP) {
            for (int i = lane; i < large_size; i += 32) {
                my_smem[i] = large_ptr[i];
            }
            __syncwarp();
            count = intersect_bsearch_smem(small_ptr, small_size, my_smem, large_size, lane);
        } else {
            count = intersect_bsearch_global(small_ptr, small_size, large_ptr, large_size, lane);
        }

        if (lane == 0) {
            scores[pid] = (2.0f * count) / (float)deg_sum;
        }
    }
}

}  

void sorensen_similarity(const graph32_t& graph,
                         const int32_t* vertex_pairs_first,
                         const int32_t* vertex_pairs_second,
                         std::size_t num_pairs,
                         float* similarity_scores) {
    if (num_pairs == 0) return;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    const int block = 128;
    int warps_per_block = block / 32;
    int smem_size = warps_per_block * SMEM_PER_WARP * sizeof(int32_t);

    int grid = 3840;
    int64_t max_grid = ((int64_t)num_pairs * 32LL + block - 1) / block;
    if (grid > (int)max_grid) grid = (int)max_grid;

    sorensen_kernel<<<grid, block, smem_size>>>(
        offsets, indices,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, (int64_t)num_pairs);
}

}  
