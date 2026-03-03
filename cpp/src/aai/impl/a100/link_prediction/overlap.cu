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



__global__ void __launch_bounds__(256, 4)
overlap_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    
    int u = first[warp_id];
    int v = second[warp_id];

    
    int u_start, u_end, v_start, v_end;
    if (lane == 0) {
        u_start = offsets[u];
        u_end = offsets[u + 1];
        v_start = offsets[v];
        v_end = offsets[v + 1];
    }
    u_start = __shfl_sync(0xffffffff, u_start, 0);
    u_end = __shfl_sync(0xffffffff, u_end, 0);
    v_start = __shfl_sync(0xffffffff, v_start, 0);
    v_end = __shfl_sync(0xffffffff, v_end, 0);

    int u_deg = u_end - u_start;
    int v_deg = v_end - v_start;
    int min_deg = u_deg < v_deg ? u_deg : v_deg;

    if (min_deg == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    const int32_t* a_ptr;  
    const int32_t* b_ptr;  
    int a_len, b_len;
    if (u_deg <= v_deg) {
        a_ptr = indices + u_start; a_len = u_deg;
        b_ptr = indices + v_start; b_len = v_deg;
    } else {
        a_ptr = indices + v_start; a_len = v_deg;
        b_ptr = indices + u_start; b_len = u_deg;
    }

    
    int a_begin = 0;
    int b_begin = 0;
    int a_end_clip = a_len;
    int b_end_clip = b_len;

    
    int32_t a_first, a_last, b_first, b_last;
    if (lane == 0) {
        a_first = a_ptr[0];
        a_last = a_ptr[a_len - 1];
        b_first = b_ptr[0];
        b_last = b_ptr[b_len - 1];
    }
    a_first = __shfl_sync(0xffffffff, a_first, 0);
    a_last = __shfl_sync(0xffffffff, a_last, 0);
    b_first = __shfl_sync(0xffffffff, b_first, 0);
    b_last = __shfl_sync(0xffffffff, b_last, 0);

    
    if (a_first > b_last || b_first > a_last) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    if (a_first < b_first) {
        
        int lo = 0, hi = a_len;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (a_ptr[mid] < b_first) lo = mid + 1;
            else hi = mid;
        }
        a_begin = lo;
    }
    if (b_first < a_first) {
        int lo = 0, hi = b_len;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (b_ptr[mid] < a_first) lo = mid + 1;
            else hi = mid;
        }
        b_begin = lo;
    }

    
    if (a_last > b_last) {
        int lo = 0, hi = a_len;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (a_ptr[mid] <= b_last) lo = mid + 1;
            else hi = mid;
        }
        a_end_clip = lo;
    }
    if (b_last > a_last) {
        int lo = 0, hi = b_len;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (b_ptr[mid] <= a_last) lo = mid + 1;
            else hi = mid;
        }
        b_end_clip = lo;
    }

    int clipped_b_len = b_end_clip - b_begin;
    const int32_t* b_search = b_ptr + b_begin;

    
    int count = 0;
    for (int i = a_begin + lane; i < a_end_clip; i += 32) {
        int32_t val = a_ptr[i];

        
        int lo = 0, hi = clipped_b_len;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (b_search[mid] < val) lo = mid + 1;
            else hi = mid;
        }
        if (lo < clipped_b_len && b_search[lo] == val) count++;
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        count += __shfl_down_sync(0xffffffff, count, offset);
    }

    if (lane == 0) {
        scores[warp_id] = (float)count / (float)min_deg;
    }
}


__global__ void __launch_bounds__(256, 4)
overlap_warp_kernel_multi(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int u = first[warp_id];
    int v = second[warp_id];

    int u_start, u_end, v_start, v_end;
    if (lane == 0) {
        u_start = offsets[u];
        u_end = offsets[u + 1];
        v_start = offsets[v];
        v_end = offsets[v + 1];
    }
    u_start = __shfl_sync(0xffffffff, u_start, 0);
    u_end = __shfl_sync(0xffffffff, u_end, 0);
    v_start = __shfl_sync(0xffffffff, v_start, 0);
    v_end = __shfl_sync(0xffffffff, v_end, 0);

    int u_deg = u_end - u_start;
    int v_deg = v_end - v_start;
    int min_deg = u_deg < v_deg ? u_deg : v_deg;

    if (min_deg == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    const int32_t* a_ptr;
    const int32_t* b_ptr;
    int a_len, b_len;
    if (u_deg <= v_deg) {
        a_ptr = indices + u_start; a_len = u_deg;
        b_ptr = indices + v_start; b_len = v_deg;
    } else {
        a_ptr = indices + v_start; a_len = v_deg;
        b_ptr = indices + u_start; b_len = u_deg;
    }

    int count = 0;
    for (int i = lane; i < a_len; i += 32) {
        int32_t val = a_ptr[i];

        
        int lo = 0, hi = i;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (a_ptr[mid] < val) lo = mid + 1;
            else hi = mid;
        }
        int rank_in_a = i - lo;

        
        lo = 0; hi = b_len;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (b_ptr[mid] < val) lo = mid + 1;
            else hi = mid;
        }
        int b_lb = lo;

        if (b_lb < b_len && b_ptr[b_lb] == val) {
            
            lo = b_lb; hi = b_len;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (b_ptr[mid] <= val) lo = mid + 1;
                else hi = mid;
            }
            int count_in_b = lo - b_lb;
            if (rank_in_a < count_in_b) count++;
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        count += __shfl_down_sync(0xffffffff, count, offset);
    }

    if (lane == 0) {
        scores[warp_id] = (float)count / (float)min_deg;
    }
}

}  

void overlap_similarity(const graph32_t& graph,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    if (num_pairs == 0) return;

    const int warps_per_block = 8;
    const int threads_per_block = warps_per_block * 32;  
    int num_blocks = (int)((num_pairs + warps_per_block - 1) / warps_per_block);

    if (!graph.is_multigraph) {
        overlap_warp_kernel<<<num_blocks, threads_per_block>>>(
            graph.offsets, graph.indices,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int64_t)num_pairs
        );
    } else {
        overlap_warp_kernel_multi<<<num_blocks, threads_per_block>>>(
            graph.offsets, graph.indices,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int64_t)num_pairs
        );
    }
}

}  
