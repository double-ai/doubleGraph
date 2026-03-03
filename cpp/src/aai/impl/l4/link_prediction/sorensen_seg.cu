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





__device__ __forceinline__ int32_t ldg32(const int32_t* __restrict__ p)
{
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

__device__ __forceinline__ int lower_bound_dev(const int32_t* __restrict__ arr, int len, int32_t target)
{
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int32_t v = ldg32(arr + mid);
        if (v < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__device__ __forceinline__ int gallop_lower_bound(const int32_t* __restrict__ arr, int pos, int len, int32_t target)
{
    if (pos >= len) return len;
    int32_t v = ldg32(arr + pos);
    if (v >= target) return pos;

    int step = 1;
    int idx = pos;
    while (true) {
        int next = idx + step;
        if (next >= len) {
            next = len;
            break;
        }
        int32_t vv = ldg32(arr + next);
        if (vv >= target) {
            next = next + 1;  
            break;
        }
        idx = next;
        step <<= 1;
    }

    int lo = idx + 1;
    int hi = (idx + step + 1 < len) ? (idx + step + 1) : len;
    
    
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int32_t mv = ldg32(arr + mid);
        if (mv < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int warp_reduce_sum(int v)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}


__device__ __forceinline__ int intersect_merge_nomulti(
    const int32_t* __restrict__ A, int m,
    const int32_t* __restrict__ B, int n,
    int lane)
{
    const int total = m + n;
    const int per_lane = (total + 31) >> 5;
    int diag0 = lane * per_lane;
    int diag1 = diag0 + per_lane;
    if (diag1 > total) diag1 = total;
    if (diag0 >= total) return 0;

    int lo = (diag0 > n) ? (diag0 - n) : 0;
    int hi = (diag0 < m) ? diag0 : m;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        
        if (ldg32(A + (mid - 1)) < ldg32(B + (diag0 - mid))) lo = mid;
        else hi = mid - 1;
    }
    int i = lo;
    int j = diag0 - lo;

    int count = 0;
    for (int step = diag0; step < diag1; ++step) {
        if (i >= m) {
            ++j;
        } else if (j >= n) {
            ++i;
        } else {
            int32_t a = ldg32(A + i);
            int32_t b = ldg32(B + j);
            if (a < b) {
                ++i;
            } else {
                count += (a == b);
                ++j;
            }
        }
    }
    return count;
}





__global__ void __launch_bounds__(128) sorensen_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs,
    bool is_multigraph)
{
    const int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp_id >= num_pairs) return;

    const int32_t u = ldg32(pairs_first + warp_id);
    const int32_t v = ldg32(pairs_second + warp_id);

    const int32_t u0 = ldg32(offsets + u);
    const int32_t u1 = ldg32(offsets + u + 1);
    const int32_t v0 = ldg32(offsets + v);
    const int32_t v1 = ldg32(offsets + v + 1);
    const int du = u1 - u0;
    const int dv = v1 - v0;
    const int denom = du + dv;

    if (denom == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }
    if (u == v) {
        if (lane == 0) scores[warp_id] = (du > 0 ? 1.0f : 0.0f);
        return;
    }

    const int32_t* A;
    const int32_t* B;
    int m, n;
    if (du <= dv) {
        A = indices + u0;
        B = indices + v0;
        m = du;
        n = dv;
    } else {
        A = indices + v0;
        B = indices + u0;
        m = dv;
        n = du;
    }

    if (m == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    int32_t a_min = ldg32(A);
    int32_t a_max = ldg32(A + (m - 1));
    int32_t b_min = ldg32(B);
    int32_t b_max = ldg32(B + (n - 1));
    if (a_max < b_min || b_max < a_min) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    const bool is_multi = is_multigraph;
    int count = 0;

    if (!is_multi) {
        if (m <= 32 && n <= 32) {
            
            int32_t a_val = (lane < m) ? ldg32(A + lane) : 0x7fffffff;
            int32_t b_val = (lane < n) ? ldg32(B + lane) : 0x7ffffffe;
            int found = 0;
            int in_range = (lane < n);
            #pragma unroll
            for (int i = 0; i < 32; i++) {
                if (i >= m) break;
                int32_t a = __shfl_sync(0xffffffff, a_val, i);
                found |= (in_range & (b_val == a));
            }
            unsigned mask = __ballot_sync(0xffffffff, found);
            if (lane == 0) count = __popc(mask);
        } else if (n <= m * 12) {
            
            count = intersect_merge_nomulti(A, m, B, n, lane);
            count = warp_reduce_sum(count);
        } else {
            
            int b_pos = 0;
            
            if (n >= m * 32) {
                int32_t first = (lane < m) ? ldg32(A + lane) : 0x7fffffff;
                b_pos = lower_bound_dev(B, n, first);
            }

            for (int i = lane; i < m; i += 32) {
                int32_t val = ldg32(A + i);
                if (n >= m * 32) {
                    b_pos = gallop_lower_bound(B, b_pos, n, val);
                } else {
                    int lo = b_pos, hi = n;
                    while (lo < hi) {
                        int mid = (lo + hi) >> 1;
                        if (ldg32(B + mid) < val) lo = mid + 1;
                        else hi = mid;
                    }
                    b_pos = lo;
                }
                int hit = (b_pos < n && ldg32(B + b_pos) == val);
                count += hit;
                b_pos += hit;
            }
            count = warp_reduce_sum(count);
        }
    } else {
        
        for (int i = lane; i < m; i += 32) {
            int32_t val = ldg32(A + i);
            int pos_first_a = lower_bound_dev(A, m, val);
            int dup_rank = i - pos_first_a;
            int pos_b = lower_bound_dev(B, n, val);
            int pos = pos_b + dup_rank;
            count += (pos < n && ldg32(B + pos) == val);
        }
        count = warp_reduce_sum(count);
    }

    if (lane == 0) scores[warp_id] = 2.0f * (float)count / (float)denom;
}

}  

void sorensen_similarity_seg(const graph32_t& graph,
                             const int32_t* vertex_pairs_first,
                             const int32_t* vertex_pairs_second,
                             std::size_t num_pairs,
                             float* similarity_scores) {
    if (num_pairs == 0) return;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    bool is_multigraph = graph.is_multigraph;

    constexpr int BS = 128;
    constexpr int WARPS_PER_BLOCK = BS / 32;
    int nb = (int)((num_pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    sorensen_kernel<<<nb, BS>>>(
        offsets, indices, vertex_pairs_first, vertex_pairs_second,
        similarity_scores, (int64_t)num_pairs, is_multigraph);
}

}  
