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

__global__ void overlap_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int32_t u = first[warp_id];
    int32_t v = second[warp_id];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];

    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) scores[warp_id] = 0.0;
        return;
    }

    
    const int32_t* s_ptr;
    const double* sw_ptr;
    int32_t s_len;
    const int32_t* l_ptr;
    const double* lw_ptr;
    int32_t l_len;

    if (u_deg <= v_deg) {
        s_ptr = indices + u_start; sw_ptr = edge_weights + u_start; s_len = u_deg;
        l_ptr = indices + v_start; lw_ptr = edge_weights + v_start; l_len = v_deg;
    } else {
        s_ptr = indices + v_start; sw_ptr = edge_weights + v_start; s_len = v_deg;
        l_ptr = indices + u_start; lw_ptr = edge_weights + u_start; l_len = u_deg;
    }

    if (l_len <= 32) {
        
        
        int32_t my_s = (lane < s_len) ? s_ptr[lane] : 0x7fffffff;
        int32_t my_l = (lane < l_len) ? l_ptr[lane] : 0x7fffffff;
        double my_sw = (lane < s_len) ? sw_ptr[lane] : 0.0;
        double my_lw = (lane < l_len) ? lw_ptr[lane] : 0.0;

        
        double s_wdeg = my_sw;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            s_wdeg += __shfl_down_sync(0xffffffff, s_wdeg, o);
        s_wdeg = __shfl_sync(0xffffffff, s_wdeg, 0);

        double l_wdeg = my_lw;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            l_wdeg += __shfl_down_sync(0xffffffff, l_wdeg, o);
        l_wdeg = __shfl_sync(0xffffffff, l_wdeg, 0);

        double denom = fmin(s_wdeg, l_wdeg);
        if (denom == 0.0) {
            if (lane == 0) scores[warp_id] = 0.0;
            return;
        }

        
        int32_t x = my_s;
        bool valid = (lane < s_len);

        
        int lo = 0, hi = valid ? lane : 0;
        #pragma unroll
        for (int d = 0; d < 5; d++) {
            int mid = (lo + hi) >> 1;
            int32_t mv = __shfl_sync(0xffffffff, my_s, mid & 31);
            if (lo < hi) {
                if (mv < x) lo = mid + 1;
                else hi = mid;
            }
        }
        int rank = lane - lo;

        
        
        lo = 0; hi = valid ? l_len : 0;
        #pragma unroll
        for (int d = 0; d < 6; d++) {
            int mid = (lo + hi) >> 1;
            int32_t mv = __shfl_sync(0xffffffff, my_l, mid & 31);
            if (lo < hi) {
                if (mv < x) lo = mid + 1;
                else hi = mid;
            }
        }

        
        int match_pos = lo + rank;
        int32_t found = __shfl_sync(0xffffffff, my_l, match_pos & 31);
        double w_l = __shfl_sync(0xffffffff, my_lw, match_pos & 31);

        double isect_sum = 0.0;
        if (valid && match_pos < l_len && found == x) {
            isect_sum = fmin(my_sw, w_l);
        }

        
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            isect_sum += __shfl_down_sync(0xffffffff, isect_sum, o);

        if (lane == 0) scores[warp_id] = isect_sum / denom;

    } else {
        

        
        double u_wdeg = 0.0;
        for (int i = lane; i < u_deg; i += 32)
            u_wdeg += edge_weights[u_start + i];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            u_wdeg += __shfl_down_sync(0xffffffff, u_wdeg, o);
        u_wdeg = __shfl_sync(0xffffffff, u_wdeg, 0);

        double v_wdeg = 0.0;
        for (int i = lane; i < v_deg; i += 32)
            v_wdeg += edge_weights[v_start + i];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            v_wdeg += __shfl_down_sync(0xffffffff, v_wdeg, o);
        v_wdeg = __shfl_sync(0xffffffff, v_wdeg, 0);

        double denom = fmin(u_wdeg, v_wdeg);
        if (denom == 0.0) {
            if (lane == 0) scores[warp_id] = 0.0;
            return;
        }

        
        double isect_sum = 0.0;
        for (int i = lane; i < s_len; i += 32) {
            int32_t x = s_ptr[i];
            double w_s = sw_ptr[i];

            
            int lo = 0, hi = i;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (s_ptr[mid] < x) lo = mid + 1;
                else hi = mid;
            }
            int rank = i - lo;

            
            lo = 0; hi = l_len;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (l_ptr[mid] < x) lo = mid + 1;
                else hi = mid;
            }

            int match_pos = lo + rank;
            if (match_pos < l_len && l_ptr[match_pos] == x) {
                isect_sum += fmin(w_s, lw_ptr[match_pos]);
            }
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            isect_sum += __shfl_down_sync(0xffffffff, isect_sum, o);

        if (lane == 0) scores[warp_id] = isect_sum / denom;
    }
}

}  

void overlap_similarity(const graph32_t& graph,
                        const double* edge_weights,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        double* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    if (num_pairs == 0) return;
    int warps_per_block = 8;
    int threads_per_block = warps_per_block * 32;
    int64_t num_blocks = (num_pairs + warps_per_block - 1) / warps_per_block;

    overlap_kernel<<<(int)num_blocks, threads_per_block>>>(
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
