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





__device__ __forceinline__ int lower_bound_dev(const int* arr, int size, int target) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__device__ __forceinline__ int merge_intersect_single(
    const int* __restrict__ a, int size_a,
    const int* __restrict__ b, int size_b
) {
    if (size_a == 0 || size_b == 0) return 0;

    
    if (a[size_a - 1] < b[0] || b[size_b - 1] < a[0]) return 0;

    
    int i = 0, j = 0;
    if (b[0] > a[0]) {
        i = lower_bound_dev(a, size_a, b[0]);
        if (i >= size_a) return 0;
    }
    if (a[i] > b[0]) {
        j = lower_bound_dev(b, size_b, a[i]);
        if (j >= size_b) return 0;
    }

    
    int count = 0;
    while (i < size_a && j < size_b) {
        int va = a[i], vb = b[j];
        count += (va == vb);
        i += (va <= vb);
        j += (va >= vb);
    }
    return count;
}


__device__ __forceinline__ int warp_intersect_bsearch(
    const int* __restrict__ a, int size_a,
    const int* __restrict__ b, int size_b,
    int lane
) {
    int count = 0;
    for (int i = lane; i < size_a; i += 32) {
        int target = a[i];
        int lo = 0, hi = size_b;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (b[mid] < target) lo = mid + 1;
            else hi = mid;
        }
        if (lo < size_b && b[lo] == target) count++;
    }
    return count;
}




__device__ __forceinline__ int warp_intersect_merge_path(
    const int* __restrict__ a, int m,
    const int* __restrict__ b, int n,
    int lane
) {
    int total = m + n;

    int diag_start = (int)(((int64_t)lane * total) / 32);
    int diag_end   = (int)(((int64_t)(lane + 1) * total) / 32);
    int steps = diag_end - diag_start;

    
    int lo = (diag_start > n) ? (diag_start - n) : 0;
    int hi = (diag_start < m) ? diag_start : m;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (a[mid] > b[diag_start - 1 - mid])
            hi = mid;
        else
            lo = mid + 1;
    }
    int i = lo;
    int j = diag_start - lo;

    int count = 0;
    for (int s = 0; s < steps; s++) {
        bool take_a = (j >= n) || (i < m && a[i] <= b[j]);
        if (take_a) {
            if (j < n && a[i] == b[j]) count++;
            i++;
        } else {
            j++;
        }
    }

    return count;
}






__global__ void overlap_adaptive_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const int* __restrict__ pairs_first,
    const int* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    const int lane = threadIdx.x & 31;
    
    const int64_t base_pair = ((int64_t)blockIdx.x * blockDim.x + (threadIdx.x & ~31)) ;
    
    const int64_t my_pair = base_pair + lane;

    
    int u = 0, v = 0;
    int u_start = 0, u_end = 0, v_start = 0, v_end = 0;
    int deg_u = 0, deg_v = 0;

    if (my_pair < num_pairs) {
        u = pairs_first[my_pair];
        v = pairs_second[my_pair];
        u_start = offsets[u];
        u_end   = offsets[u + 1];
        v_start = offsets[v];
        v_end   = offsets[v + 1];
        deg_u = u_end - u_start;
        deg_v = v_end - v_start;
    }

    
    int my_max_deg = (deg_u > deg_v) ? deg_u : deg_v;
    int warp_max_deg = my_max_deg;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_xor_sync(0xFFFFFFFF, warp_max_deg, offset);
        warp_max_deg = (warp_max_deg > other) ? warp_max_deg : other;
    }

    if (warp_max_deg <= 32) {
        
        if (my_pair >= num_pairs) return;

        int min_deg = (deg_u < deg_v) ? deg_u : deg_v;
        if (min_deg == 0) {
            scores[my_pair] = 0.0f;
            return;
        }

        
        const int* a = indices + u_start;
        const int* b = indices + v_start;

        
        if (a[deg_u - 1] < b[0] || b[deg_v - 1] < a[0]) {
            scores[my_pair] = 0.0f;
            return;
        }

        int count = 0;
        int i = 0, j = 0;
        while (i < deg_u && j < deg_v) {
            int va = a[i], vb = b[j];
            count += (va == vb);
            i += (va <= vb);
            j += (va >= vb);
        }

        scores[my_pair] = (float)count / (float)min_deg;

    } else {
        
        
        int num_active = 32;
        if (base_pair + 31 >= num_pairs) {
            num_active = (int)(num_pairs - base_pair);
            if (num_active <= 0) return;
        }

        for (int owner = 0; owner < num_active; owner++) {
            
            int o_u_start = __shfl_sync(0xFFFFFFFF, u_start, owner);
            int o_u_end   = __shfl_sync(0xFFFFFFFF, u_end, owner);
            int o_v_start = __shfl_sync(0xFFFFFFFF, v_start, owner);
            int o_v_end   = __shfl_sync(0xFFFFFFFF, v_end, owner);
            int o_deg_u   = o_u_end - o_u_start;
            int o_deg_v   = o_v_end - o_v_start;
            int o_min_deg = (o_deg_u < o_deg_v) ? o_deg_u : o_deg_v;

            if (o_min_deg == 0) {
                if (lane == 0) scores[base_pair + owner] = 0.0f;
                continue;
            }

            
            const int* a_ptr = (o_deg_u <= o_deg_v) ? indices + o_u_start : indices + o_v_start;
            const int* b_ptr = (o_deg_u <= o_deg_v) ? indices + o_v_start : indices + o_u_start;
            int size_a = (o_deg_u <= o_deg_v) ? o_deg_u : o_deg_v;
            int size_b = (o_deg_u <= o_deg_v) ? o_deg_v : o_deg_u;

            
            int a_max = a_ptr[size_a - 1];
            int b_min = b_ptr[0];
            if (a_max < b_min || b_ptr[size_b - 1] < a_ptr[0]) {
                if (lane == 0) scores[base_pair + owner] = 0.0f;
                continue;
            }

            int count;
            if (size_a + size_b <= 128) {
                
                count = warp_intersect_bsearch(a_ptr, size_a, b_ptr, size_b, lane);
            } else {
                
                count = warp_intersect_merge_path(a_ptr, size_a, b_ptr, size_b, lane);
            }

            
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                count += __shfl_down_sync(0xFFFFFFFF, count, offset);
            }

            if (lane == 0) {
                scores[base_pair + owner] = (float)count / (float)o_min_deg;
            }
        }
    }
}




__global__ void overlap_thread_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const int* __restrict__ pairs_first,
    const int* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    int u = pairs_first[idx];
    int v = pairs_second[idx];

    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int v_start = offsets[v];
    int v_end = offsets[v + 1];

    int deg_u = u_end - u_start;
    int deg_v = v_end - v_start;
    int min_deg = (deg_u < deg_v) ? deg_u : deg_v;

    if (min_deg == 0) {
        scores[idx] = 0.0f;
        return;
    }

    int count = merge_intersect_single(
        indices + u_start, deg_u,
        indices + v_start, deg_v
    );

    scores[idx] = (float)count / (float)min_deg;
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
    bool is_multigraph = graph.is_multigraph;

    if (!is_multigraph) {
        const int block_size = 256;
        const int grid_size = (int)((num_pairs + block_size - 1) / block_size);
        overlap_adaptive_kernel<<<grid_size, block_size>>>(
            offsets, indices, vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int64_t)num_pairs);
    } else {
        const int block_size = 256;
        const int grid_size = (int)((num_pairs + block_size - 1) / block_size);
        overlap_thread_kernel<<<grid_size, block_size>>>(
            offsets, indices, vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int64_t)num_pairs);
    }
}

}  
