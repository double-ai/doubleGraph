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

struct Cache : Cacheable {};

__device__ __forceinline__ int32_t dev_lb(
    const int32_t* __restrict__ arr, int32_t lo, int32_t hi, int32_t target
) {
    while (lo < hi) {
        int32_t mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int32_t gallop_lb(
    const int32_t* __restrict__ arr, int32_t start, int32_t end, int32_t target
) {
    if (start >= end || arr[start] >= target) return start;
    int32_t pos = start, step = 1;
    while (pos + step < end && arr[pos + step] < target) {
        pos += step;
        step <<= 1;
    }
    return dev_lb(arr, pos + 1, (pos + step < end) ? pos + step + 1 : end, target);
}

__device__ float overlap_thread_impl(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t u, int32_t v
) {
    int32_t u_s = offsets[u], u_e = offsets[u + 1];
    int32_t v_s = offsets[v], v_e = offsets[v + 1];
    int32_t du = u_e - u_s, dv = v_e - v_s;
    int32_t min_deg = (du < dv) ? du : dv;
    if (min_deg == 0) return 0.0f;
    if (u == v) return 1.0f;

    const int32_t* a = (du <= dv) ? (indices + u_s) : (indices + v_s);
    const int32_t* b = (du <= dv) ? (indices + v_s) : (indices + u_s);
    int32_t sa = (du <= dv) ? du : dv;
    int32_t sb = (du <= dv) ? dv : du;

    int32_t a0 = a[0], al = a[sa-1], b0 = b[0], bl = b[sb-1];
    if (al < b0 || bl < a0) return 0.0f;

    int count = 0;
    if (sb >= sa * 4) {
        int32_t j = dev_lb(b, 0, sb, a0);
        int32_t i = dev_lb(a, 0, sa, b0);
        for (; i < sa && j < sb; i++) {
            int32_t t = a[i];
            if (t > bl) break;
            j = gallop_lb(b, j, sb, t);
            if (j < sb && b[j] == t) { count++; j++; }
        }
    } else {
        int32_t i = 0, j = 0;
        if (a0 < b0) i = dev_lb(a, 0, sa, b0);
        else if (b0 < a0) j = dev_lb(b, 0, sb, a0);
        while (i < sa && j < sb) {
            int32_t va = a[i], vb = b[j];
            count += (va == vb);
            i += (va <= vb);
            j += (va >= vb);
        }
    }
    return (float)count / (float)min_deg;
}


__device__ __noinline__ float overlap_warp_impl(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t u, int32_t v, int lane
) {
    int32_t u_s = offsets[u], u_e = offsets[u + 1];
    int32_t v_s = offsets[v], v_e = offsets[v + 1];
    int32_t du = u_e - u_s, dv = v_e - v_s;
    int32_t min_deg = (du < dv) ? du : dv;
    if (min_deg == 0) return 0.0f;
    if (u == v) return 1.0f;

    const int32_t* sm = (du <= dv) ? (indices + u_s) : (indices + v_s);
    const int32_t* lg = (du <= dv) ? (indices + v_s) : (indices + u_s);
    int32_t ss = (du <= dv) ? du : dv;
    int32_t sl = (du <= dv) ? dv : du;

    if (sm[ss-1] < lg[0] || lg[sl-1] < sm[0]) return 0.0f;

    int total = 0;
    int32_t my_lb = 0;  

    for (int base = 0; base < ss; base += 32) {
        int k = base + lane;
        int lc = 0;
        if (k < ss) {
            int32_t val = sm[k];
            bool first = (k == 0) || (sm[k-1] != val);
            if (first) {
                int cs = 1;
                while (k + cs < ss && sm[k + cs] == val) cs++;
                
                int32_t lo = dev_lb(lg, my_lb, sl, val);
                my_lb = lo;  
                int cl = 0;
                while (lo + cl < sl && lg[lo + cl] == val) cl++;
                lc = (cs < cl) ? cs : cl;
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            lc += __shfl_down_sync(0xffffffff, lc, off);
        if (lane == 0) total += lc;
    }
    total = __shfl_sync(0xffffffff, total, 0);
    return (float)total / (float)min_deg;
}

#define HEAVY_THRESHOLD 40

__global__ __launch_bounds__(256)
void overlap_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int64_t base_pair = (int64_t)warp_global * 32;
    if (base_pair >= num_pairs) return;

    int64_t pair_idx = base_pair + lane;
    int32_t my_u = 0, my_v = 0, my_deg_sum = 0;
    bool valid = pair_idx < num_pairs;

    if (valid) {
        my_u = pairs_first[pair_idx];
        my_v = pairs_second[pair_idx];
        my_deg_sum = (offsets[my_u+1] - offsets[my_u]) + (offsets[my_v+1] - offsets[my_v]);
    }

    int mx = my_deg_sum;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        mx = max(mx, __shfl_xor_sync(0xffffffff, mx, off));

    if (mx <= HEAVY_THRESHOLD) {
        
        if (valid) {
            scores[pair_idx] = overlap_thread_impl(offsets, indices, my_u, my_v);
        }
    } else {
        
        int num_valid = (int)min((int64_t)32, num_pairs - base_pair);
        for (int i = 0; i < num_valid; i++) {
            int32_t cd = __shfl_sync(0xffffffff, my_deg_sum, i);
            if (cd > HEAVY_THRESHOLD) {
                int32_t cu = __shfl_sync(0xffffffff, my_u, i);
                int32_t cv = __shfl_sync(0xffffffff, my_v, i);
                float s = overlap_warp_impl(offsets, indices, cu, cv, lane);
                if (lane == 0) scores[base_pair + i] = s;
            }
        }
        
        if (valid && my_deg_sum <= HEAVY_THRESHOLD) {
            scores[pair_idx] = overlap_thread_impl(offsets, indices, my_u, my_v);
        }
    }
}

}  

void overlap_similarity_seg(const graph32_t& graph,
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

    int warps_needed = (int)((num_pairs + 31) / 32);
    int block = 256;
    int grid = (int)(((int64_t)warps_needed * 32 + block - 1) / block);
    overlap_kernel<<<grid, block>>>(
        offsets, indices, vertex_pairs_first, vertex_pairs_second,
        similarity_scores, (int64_t)num_pairs
    );
}

}  
