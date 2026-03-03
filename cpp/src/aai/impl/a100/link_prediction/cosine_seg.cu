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




__global__ void cosine_similarity_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    int num_pairs)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int u = pairs_first[warp_id];
    int v = pairs_second[warp_id];

    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int v_start = offsets[v];
    int v_end = offsets[v + 1];

    int u_deg = u_end - u_start;
    int v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    
    const int32_t* probe;
    const int32_t* search;
    int probe_len, search_len;

    if (u_deg <= v_deg) {
        probe = indices + u_start;
        probe_len = u_deg;
        search = indices + v_start;
        search_len = v_deg;
    } else {
        probe = indices + v_start;
        probe_len = v_deg;
        search = indices + u_start;
        search_len = u_deg;
    }

    
    
    int found = 0;
    int num_iters = (probe_len + 31) >> 5;  

    for (int iter = 0; iter < num_iters; iter++) {
        int i = iter * 32 + lane;
        if (i < probe_len) {
            int32_t val = probe[i];
            
            int lo = 0, hi = search_len;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (search[mid] < val) lo = mid + 1;
                else hi = mid;
            }
            if (lo < search_len && search[lo] == val) {
                found = 1;
            }
        }
        
        if (__any_sync(0xffffffff, found)) break;
    }

    
    unsigned int ballot = __ballot_sync(0xffffffff, found);
    if (lane == 0) {
        scores[warp_id] = (ballot != 0) ? 1.0f : 0.0f;
    }
}

}  

void cosine_similarity_seg(const graph32_t& graph,
                           const int32_t* vertex_pairs_first,
                           const int32_t* vertex_pairs_second,
                           std::size_t num_pairs,
                           float* similarity_scores) {
    if (num_pairs == 0) return;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    int threads_per_block = 256;  
    int warps_needed = static_cast<int>(num_pairs);
    int blocks = (int)(((int64_t)warps_needed * 32 + threads_per_block - 1) / threads_per_block);

    cosine_similarity_kernel<<<blocks, threads_per_block>>>(
        offsets, indices, vertex_pairs_first, vertex_pairs_second,
        similarity_scores, warps_needed);
}

}  
