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
#include <math_constants.h>

namespace aai {

namespace {


__device__ __forceinline__ int read_l_reg(int l_lo, int l_hi, int pos) {
    
    int val_lo = __shfl_sync(0xffffffff, l_lo, pos & 31);
    int val_hi = __shfl_sync(0xffffffff, l_hi, pos & 31);
    return (pos < 32) ? val_lo : val_hi;
}

__global__ void __launch_bounds__(128)
cosine_sim_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    double* __restrict__ scores,
    const int num_pairs,
    const int is_multigraph)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    const int u = __ldg(&first[warp_id]);
    const int v = __ldg(&second[warp_id]);

    const int u_start = __ldg(&offsets[u]);
    const int u_end   = __ldg(&offsets[u + 1]);
    const int v_start = __ldg(&offsets[v]);
    const int v_end   = __ldg(&offsets[v + 1]);

    const int u_len = u_end - u_start;
    const int v_len = v_end - v_start;

    if (u_len == 0 || v_len == 0) {
        if (lane == 0) scores[warp_id] = CUDART_NAN;
        return;
    }

    
    int s_start, s_len, l_start, l_len;
    if (u_len <= v_len) {
        s_start = u_start; s_len = u_len;
        l_start = v_start; l_len = v_len;
    } else {
        s_start = v_start; s_len = v_len;
        l_start = u_start; l_len = u_len;
    }

    double dot = 0.0, ns_acc = 0.0, nl_acc = 0.0;

    if (l_len <= 32 && !is_multigraph) {
        
        const int s_val = (lane < s_len) ? __ldg(&indices[s_start + lane]) : 0x7fffffff;
        const int l_val = (lane < l_len) ? __ldg(&indices[l_start + lane]) : 0x7fffffff;

        
        int s_max_v = __shfl_sync(0xffffffff, s_val, s_len - 1);
        int l_min_v = __shfl_sync(0xffffffff, l_val, 0);
        if (s_max_v < l_min_v) { if (lane == 0) scores[warp_id] = CUDART_NAN; return; }
        int s_min_v = __shfl_sync(0xffffffff, s_val, 0);
        int l_max_v = __shfl_sync(0xffffffff, l_val, l_len - 1);
        if (l_max_v < s_min_v) { if (lane == 0) scores[warp_id] = CUDART_NAN; return; }

        const int target = (lane < s_len) ? s_val : 0x7fffffff;
        int lo = 0, hi = l_len;
        #pragma unroll
        for (int iter = 0; iter < 6; iter++) {
            int mid = (lo + hi) >> 1;
            int mid_val = __shfl_sync(0xffffffff, l_val, mid & 31);
            if (mid_val < target) lo = mid + 1;
            else hi = mid;
        }

        int found = __shfl_sync(0xffffffff, l_val, lo & 31);
        int is_match = (lane < s_len) && (lo < l_len) && (found == target);

        if (is_match) {
            double ws = __ldg(&weights[s_start + lane]);
            double wl = __ldg(&weights[l_start + lo]);
            dot += ws * wl;
            ns_acc += ws * ws;
            nl_acc += wl * wl;
        }

    } else if (l_len <= 64 && s_len <= 32 && !is_multigraph) {
        
        const int s_val = (lane < s_len) ? __ldg(&indices[s_start + lane]) : 0x7fffffff;
        const int l_lo = (lane < l_len) ? __ldg(&indices[l_start + lane]) : 0x7fffffff;
        const int l_hi = (lane + 32 < l_len) ? __ldg(&indices[l_start + lane + 32]) : 0x7fffffff;

        
        int s_max_v = __shfl_sync(0xffffffff, s_val, s_len - 1);
        int l_min_v = __shfl_sync(0xffffffff, l_lo, 0);
        if (s_max_v < l_min_v) { if (lane == 0) scores[warp_id] = CUDART_NAN; return; }
        int s_min_v = __shfl_sync(0xffffffff, s_val, 0);
        
        int l_last_val;
        if (l_len <= 32) {
            l_last_val = __shfl_sync(0xffffffff, l_lo, (l_len - 1) & 31);
        } else {
            l_last_val = __shfl_sync(0xffffffff, l_hi, (l_len - 33) & 31);
        }
        if (l_last_val < s_min_v) { if (lane == 0) scores[warp_id] = CUDART_NAN; return; }

        const int target = (lane < s_len) ? s_val : 0x7fffffff;

        
        int lo = 0, hi = l_len;
        #pragma unroll
        for (int iter = 0; iter < 7; iter++) {
            int mid = (lo + hi) >> 1;
            int mid_val = read_l_reg(l_lo, l_hi, mid);
            if (mid_val < target) lo = mid + 1;
            else hi = mid;
        }

        
        int found = read_l_reg(l_lo, l_hi, lo);
        int is_match = (lane < s_len) && (lo < l_len) && (found == target);

        if (is_match) {
            double ws = __ldg(&weights[s_start + lane]);
            double wl = __ldg(&weights[l_start + lo]);
            dot += ws * wl;
            ns_acc += ws * ws;
            nl_acc += wl * wl;
        }

    } else {
        
        const int s_max = __ldg(&indices[s_start + s_len - 1]);
        const int l_min_v = __ldg(&indices[l_start]);
        if (s_max < l_min_v) { if (lane == 0) scores[warp_id] = CUDART_NAN; return; }
        const int s_min = __ldg(&indices[s_start]);
        const int l_max_v = __ldg(&indices[l_start + l_len - 1]);
        if (l_max_v < s_min) { if (lane == 0) scores[warp_id] = CUDART_NAN; return; }

        int l_hint = 0;
        for (int i = lane; i < s_len; i += 32) {
            int target = __ldg(&indices[s_start + i]);
            if (target > l_max_v) break;
            if (target < l_min_v) continue;

            int rank = 0;
            if (is_multigraph) {
                int lo2 = 0, hi2 = i;
                while (lo2 < hi2) {
                    int mid2 = (lo2 + hi2) >> 1;
                    if (__ldg(&indices[s_start + mid2]) < target)
                        lo2 = mid2 + 1;
                    else
                        hi2 = mid2;
                }
                rank = i - lo2;
            }

            int lo = l_hint, hi = l_len;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (__ldg(&indices[l_start + mid]) < target)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            l_hint = lo;

            int match_pos = lo + rank;
            if (match_pos < l_len && __ldg(&indices[l_start + match_pos]) == target) {
                double ws = __ldg(&weights[s_start + i]);
                double wl = __ldg(&weights[l_start + match_pos]);
                dot += ws * wl;
                ns_acc += ws * ws;
                nl_acc += wl * wl;
            }
        }
    }

    
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        dot    += __shfl_down_sync(0xffffffff, dot, off);
        ns_acc += __shfl_down_sync(0xffffffff, ns_acc, off);
        nl_acc += __shfl_down_sync(0xffffffff, nl_acc, off);
    }

    if (lane == 0) {
        if (ns_acc == 0.0)
            scores[warp_id] = CUDART_NAN;
        else
            scores[warp_id] = dot / (sqrt(ns_acc) * sqrt(nl_acc));
    }
}

void launch_cosine_similarity(
    const int32_t* offsets,
    const int32_t* indices,
    const double* weights,
    const int32_t* first,
    const int32_t* second,
    double* scores,
    int64_t num_pairs,
    int is_multigraph,
    cudaStream_t stream)
{
    if (num_pairs == 0) return;

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    int blocks = ((int)num_pairs + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    cosine_sim_kernel<<<blocks, THREADS, 0, stream>>>(
        offsets, indices, weights, first, second, scores, (int)num_pairs, is_multigraph);
}

}  

void cosine_similarity_seg(const graph32_t& graph,
                           const double* edge_weights,
                           const int32_t* vertex_pairs_first,
                           const int32_t* vertex_pairs_second,
                           std::size_t num_pairs,
                           double* similarity_scores) {
    launch_cosine_similarity(
        graph.offsets,
        graph.indices,
        edge_weights,
        vertex_pairs_first,
        vertex_pairs_second,
        similarity_scores,
        static_cast<int64_t>(num_pairs),
        graph.is_multigraph ? 1 : 0,
        0);
}

}  
