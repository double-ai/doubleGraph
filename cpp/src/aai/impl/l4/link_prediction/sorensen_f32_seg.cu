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

#define WARPS_PER_BLOCK 4
#define THREADS_PER_BLOCK (WARPS_PER_BLOCK * 32)



#define MAX_SMEM_PER_WARP 256



__global__ __launch_bounds__(THREADS_PER_BLOCK)
void sorensen_fused_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    __shared__ int s_cache[WARPS_PER_BLOCK * MAX_SMEM_PER_WARP];

    const int warp_in_block = threadIdx.x >> 5;
    const int global_warp_id = (int)((blockIdx.x * (int64_t)THREADS_PER_BLOCK + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;

    if (global_warp_id >= num_pairs) return;

    int* my_cache = s_cache + warp_in_block * MAX_SMEM_PER_WARP;

    const int u = pairs_first[global_warp_id];
    const int v = pairs_second[global_warp_id];

    const int u_start = __ldg(&offsets[u]);
    const int u_end = __ldg(&offsets[u + 1]);
    const int v_start = __ldg(&offsets[v]);
    const int v_end = __ldg(&offsets[v + 1]);
    const int u_len = u_end - u_start;
    const int v_len = v_end - v_start;

    
    float deg_u = 0.0f;
    for (int i = lane; i < u_len; i += 32)
        deg_u += __ldg(&edge_weights[u_start + i]);
    for (int off = 16; off > 0; off >>= 1)
        deg_u += __shfl_down_sync(0xffffffff, deg_u, off);
    deg_u = __shfl_sync(0xffffffff, deg_u, 0);

    
    float deg_v = 0.0f;
    for (int i = lane; i < v_len; i += 32)
        deg_v += __ldg(&edge_weights[v_start + i]);
    for (int off = 16; off > 0; off >>= 1)
        deg_v += __shfl_down_sync(0xffffffff, deg_v, off);
    deg_v = __shfl_sync(0xffffffff, deg_v, 0);

    const float denom = deg_u + deg_v;
    if (denom == 0.0f) {
        if (lane == 0) scores[global_warp_id] = 0.0f;
        return;
    }

    
    int a_start, a_len, b_start, b_len;
    if (u_len <= v_len) {
        a_start = u_start; a_len = u_len;
        b_start = v_start; b_len = v_len;
    } else {
        a_start = v_start; a_len = v_len;
        b_start = u_start; b_len = u_len;
    }

    if (a_len == 0) {
        if (lane == 0) scores[global_warp_id] = 0.0f;
        return;
    }

    
    const bool use_cache = (b_len <= MAX_SMEM_PER_WARP);
    if (use_cache) {
        for (int i = lane; i < b_len; i += 32)
            my_cache[i] = indices[b_start + i];
        __syncwarp();
    }

    float local_sum = 0.0f;

    for (int i = lane; i < a_len; i += 32) {
        const int target = __ldg(&indices[a_start + i]);
        const float wa = __ldg(&edge_weights[a_start + i]);

        
        int occ_idx = 0;
        if (i > 0 && __ldg(&indices[a_start + i - 1]) == target) {
            
            int flo = 0, fhi = i;
            while (flo < fhi) {
                int mid = (flo + fhi) >> 1;
                if (__ldg(&indices[a_start + mid]) < target) flo = mid + 1;
                else fhi = mid;
            }
            occ_idx = i - flo;
        }

        
        int lo = 0, hi = b_len;
        if (use_cache) {
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (my_cache[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            int mp = lo + occ_idx;
            if (mp < b_len && my_cache[mp] == target)
                local_sum += fminf(wa, __ldg(&edge_weights[b_start + mp]));
        } else {
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (__ldg(&indices[b_start + mid]) < target) lo = mid + 1;
                else hi = mid;
            }
            int mp = lo + occ_idx;
            if (mp < b_len && __ldg(&indices[b_start + mp]) == target)
                local_sum += fminf(wa, __ldg(&edge_weights[b_start + mp]));
        }
    }

    
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);

    if (lane == 0) {
        scores[global_warp_id] = 2.0f * local_sum / denom;
    }
}

}  

void sorensen_similarity_seg(const graph32_t& graph,
                             const float* edge_weights,
                             const int32_t* vertex_pairs_first,
                             const int32_t* vertex_pairs_second,
                             std::size_t num_pairs,
                             float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    if (num_pairs == 0) return;

    const int grid_size = ((int)num_pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    sorensen_fused_kernel<<<grid_size, THREADS_PER_BLOCK>>>(
        graph.offsets,
        graph.indices,
        edge_weights,
        vertex_pairs_first,
        vertex_pairs_second,
        similarity_scores,
        (int64_t)num_pairs
    );
}

}  
