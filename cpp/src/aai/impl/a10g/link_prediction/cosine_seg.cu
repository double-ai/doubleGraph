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



__global__ __launch_bounds__(256)
void cosine_sim_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int64_t num_pairs
) {
    const int lane = threadIdx.x & 31;
    const int half = lane >> 4;           
    const int sub_lane = lane & 15;       
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int64_t pair_id = (int64_t)warp_id * 2 + half;
    const unsigned hmask = half ? 0xffff0000u : 0x0000ffffu;

    if (pair_id >= num_pairs) return;

    int u = pairs_first[pair_id];
    int v = pairs_second[pair_id];

    int u_start = __ldg(offsets + u);
    int u_end = __ldg(offsets + u + 1);
    int v_start = __ldg(offsets + v);
    int v_end = __ldg(offsets + v + 1);

    int deg_u = u_end - u_start;
    int deg_v = v_end - v_start;

    if (deg_u == 0 || deg_v == 0) {
        if (sub_lane == 0) scores[pair_id] = 0.0f;
        return;
    }
    if (u == v) {
        if (sub_lane == 0) scores[pair_id] = 1.0f;
        return;
    }

    const int32_t* a_ptr, *b_ptr;
    int a_size, b_size;
    if (deg_u <= deg_v) {
        a_ptr = indices + u_start; a_size = deg_u;
        b_ptr = indices + v_start; b_size = deg_v;
    } else {
        a_ptr = indices + v_start; a_size = deg_v;
        b_ptr = indices + u_start; b_size = deg_u;
    }

    int found = 0;
    const int num_iters = (a_size + 15) >> 4;

    
    int b_lo_bound = 0;

    for (int iter = 0; iter < num_iters; iter++) {
        int idx = iter * 16 + sub_lane;
        int my_pos = b_size; 

        if (idx < a_size) {
            int target = a_ptr[idx];
            int lo = b_lo_bound, hi = b_size;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                int val = b_ptr[mid];
                lo = (val < target) ? mid + 1 : lo;
                hi = (val < target) ? hi : mid;
            }
            if (lo < b_size && b_ptr[lo] == target) {
                found = 1;
            }
            my_pos = lo;
        }

        if (__any_sync(hmask, found)) break;

        
        
        int lane0_pos = __shfl_sync(hmask, my_pos, half * 16);
        if (lane0_pos < b_size) b_lo_bound = lane0_pos;
    }

    unsigned result = __ballot_sync(hmask, found);
    if (sub_lane == 0) {
        scores[pair_id] = (result != 0) ? 1.0f : 0.0f;
    }
}

}  

void cosine_similarity_seg(const graph32_t& graph,
                           const int32_t* vertex_pairs_first,
                           const int32_t* vertex_pairs_second,
                           std::size_t num_pairs,
                           float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    if (num_pairs == 0) return;

    const int threads_per_block = 256;
    const int pairs_per_block = (threads_per_block / 32) * 2;
    const int num_blocks = (num_pairs + pairs_per_block - 1) / pairs_per_block;

    cosine_sim_kernel<<<num_blocks, threads_per_block>>>(
        graph.offsets, graph.indices,
        vertex_pairs_first, vertex_pairs_second,
        similarity_scores, static_cast<int64_t>(num_pairs)
    );
}

}  
