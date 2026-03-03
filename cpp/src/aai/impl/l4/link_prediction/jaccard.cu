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
#include <cuda_pipeline_primitives.h>
#include <cstdint>

namespace aai {

namespace {

struct Cache : Cacheable {};

#define GROUP_SIZE 16
#define SMEM_PER_GROUP 88
#define BLOCK_SIZE 128

__device__ __forceinline__ int lb_smem(const int* arr, int size, int target) {
    const int* base = arr;
    int n = size;
    while (n > 1) {
        int half = n >> 1;
        base += (base[half] < target) ? half : 0;
        n -= half;
    }
    return (int)(base - arr) + (n > 0 && base[0] < target);
}

__device__ __forceinline__ int lb_global(const int* __restrict__ arr, int size, int target) {
    const int* base = arr;
    int n = size;
    while (n > 1) {
        int half = n >> 1;
        base += (__ldg(base + half) < target) ? half : 0;
        n -= half;
    }
    return (int)(base - arr) + (n > 0 && __ldg(base) < target);
}

__global__ __launch_bounds__(BLOCK_SIZE)
void jaccard_simple_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const int* __restrict__ first,
    const int* __restrict__ second,
    float* __restrict__ scores,
    int num_pairs)
{
    const int lane = threadIdx.x % GROUP_SIZE;
    const int group_in_block = threadIdx.x / GROUP_SIZE;
    const int groups_per_block = BLOCK_SIZE / GROUP_SIZE;
    const int pair_id = blockIdx.x * groups_per_block + group_in_block;

    if (pair_id >= num_pairs) return;

    extern __shared__ int smem[];
    int* gs = smem + group_in_block * SMEM_PER_GROUP;

    const int warp_lane = threadIdx.x & 31;
    const int gs_in_warp = (warp_lane / GROUP_SIZE) * GROUP_SIZE;
    const unsigned mask = ((1u << GROUP_SIZE) - 1u) << gs_in_warp;

    int u = first[pair_id];
    int v = second[pair_id];

    int u_start = __ldg(&offsets[u]);
    int u_end = __ldg(&offsets[u + 1]);
    int v_start = __ldg(&offsets[v]);
    int v_end = __ldg(&offsets[v + 1]);

    int u_deg = u_end - u_start;
    int v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) scores[pair_id] = 0.0f;
        return;
    }

    const int* a_base = (u_deg <= v_deg) ? (indices + u_start) : (indices + v_start);
    const int* b_base = (u_deg <= v_deg) ? (indices + v_start) : (indices + u_start);
    int a_size = (u_deg <= v_deg) ? u_deg : v_deg;
    int b_size = (u_deg <= v_deg) ? v_deg : u_deg;

    int count = 0;

    if (b_size <= SMEM_PER_GROUP) {
        
        for (int i = lane; i < b_size; i += GROUP_SIZE)
            __pipeline_memcpy_async(&gs[i], &b_base[i], sizeof(int));
        __pipeline_commit();

        
        int a_iters = (a_size + GROUP_SIZE - 1) / GROUP_SIZE;

        
        int a0 = (lane < a_size) ? __ldg(&a_base[lane]) : 0x7FFFFFFF;

        
        __pipeline_wait_prior(0);
        __syncwarp(mask);

        
        if (lane < a_size) {
            int pos = lb_smem(gs, b_size, a0);
            count += (pos < b_size && gs[pos] == a0);
        }

        
        for (int iter = 1; iter < a_iters; iter++) {
            int i = lane + iter * GROUP_SIZE;
            if (i < a_size) {
                int val = __ldg(&a_base[i]);
                int pos = lb_smem(gs, b_size, val);
                count += (pos < b_size && gs[pos] == val);
            }
        }
    } else {
        
        int search_lo = 0;
        for (int i = lane; i < a_size; i += GROUP_SIZE) {
            int val = __ldg(&a_base[i]);
            int remaining = b_size - search_lo;
            int pos = search_lo + lb_global(b_base + search_lo, remaining, val);
            count += (pos < b_size && __ldg(&b_base[pos]) == val);
            search_lo = pos;
        }
    }

    #pragma unroll
    for (int offset = GROUP_SIZE / 2; offset > 0; offset >>= 1)
        count += __shfl_down_sync(mask, count, offset);

    if (lane == 0) {
        int inter = count;
        int uni = u_deg + v_deg - inter;
        scores[pair_id] = (uni > 0) ? __int2float_rn(inter) / __int2float_rn(uni) : 0.0f;
    }
}

__global__ __launch_bounds__(BLOCK_SIZE)
void jaccard_multi_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const int* __restrict__ first,
    const int* __restrict__ second,
    float* __restrict__ scores,
    int num_pairs)
{
    const int lane = threadIdx.x % GROUP_SIZE;
    const int group_in_block = threadIdx.x / GROUP_SIZE;
    const int groups_per_block = BLOCK_SIZE / GROUP_SIZE;
    const int pair_id = blockIdx.x * groups_per_block + group_in_block;

    if (pair_id >= num_pairs) return;

    extern __shared__ int smem[];
    int* gs = smem + group_in_block * SMEM_PER_GROUP;

    const int warp_lane = threadIdx.x & 31;
    const int gs_in_warp = (warp_lane / GROUP_SIZE) * GROUP_SIZE;
    const unsigned mask = ((1u << GROUP_SIZE) - 1u) << gs_in_warp;

    int u = first[pair_id];
    int v = second[pair_id];

    int u_start = __ldg(&offsets[u]);
    int u_end = __ldg(&offsets[u + 1]);
    int v_start = __ldg(&offsets[v]);
    int v_end = __ldg(&offsets[v + 1]);

    int u_deg = u_end - u_start;
    int v_deg = v_end - v_start;

    if (u_deg == 0 || v_deg == 0) {
        if (lane == 0) scores[pair_id] = 0.0f;
        return;
    }

    const int* a_base = (u_deg <= v_deg) ? (indices + u_start) : (indices + v_start);
    const int* b_base = (u_deg <= v_deg) ? (indices + v_start) : (indices + u_start);
    int a_size = (u_deg <= v_deg) ? u_deg : v_deg;
    int b_size = (u_deg <= v_deg) ? v_deg : u_deg;

    int count = 0;

    if (b_size <= SMEM_PER_GROUP) {
        for (int i = lane; i < b_size; i += GROUP_SIZE)
            __pipeline_memcpy_async(&gs[i], &b_base[i], sizeof(int));
        __pipeline_commit();

        int a0 = (lane < a_size) ? __ldg(&a_base[lane]) : 0x7FFFFFFF;

        __pipeline_wait_prior(0);
        __syncwarp(mask);

        
        for (int i = lane; i < a_size; i += GROUP_SIZE) {
            int val = (i == lane) ? a0 : __ldg(&a_base[i]);
            int pos = lb_smem(gs, b_size, val);
            if (pos < b_size && gs[pos] == val) {
                if (i == 0 || __ldg(&a_base[i - 1]) != val) {
                    count++;
                } else {
                    int lb_a = lb_global(a_base, i, val);
                    int rank = i - lb_a;
                    int ub_pos = pos + 1;
                    while (ub_pos < b_size && gs[ub_pos] == val) ub_pos++;
                    if (rank < ub_pos - pos) count++;
                }
            }
        }
    } else {
        int search_lo = 0;
        for (int i = lane; i < a_size; i += GROUP_SIZE) {
            int val = __ldg(&a_base[i]);
            int remaining = b_size - search_lo;
            int pos = search_lo + lb_global(b_base + search_lo, remaining, val);
            if (pos < b_size && __ldg(&b_base[pos]) == val) {
                if (i == 0 || __ldg(&a_base[i - 1]) != val) {
                    count++;
                } else {
                    int lb_a = lb_global(a_base, i, val);
                    int rank = i - lb_a;
                    int ub_pos = pos + 1;
                    while (ub_pos < b_size && __ldg(&b_base[ub_pos]) == val) ub_pos++;
                    if (rank < ub_pos - pos) count++;
                }
            }
            search_lo = pos;
        }
    }

    #pragma unroll
    for (int offset = GROUP_SIZE / 2; offset > 0; offset >>= 1)
        count += __shfl_down_sync(mask, count, offset);

    if (lane == 0) {
        int inter = count;
        int uni = u_deg + v_deg - inter;
        scores[pair_id] = (uni > 0) ? __int2float_rn(inter) / __int2float_rn(uni) : 0.0f;
    }
}

}  

void jaccard_similarity(const graph32_t& graph,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int groups_per_block = BLOCK_SIZE / GROUP_SIZE;
    int num_blocks = (int)((num_pairs + groups_per_block - 1) / groups_per_block);
    int smem_size = groups_per_block * SMEM_PER_GROUP * sizeof(int);

    if (graph.is_multigraph) {
        jaccard_multi_kernel<<<num_blocks, BLOCK_SIZE, smem_size>>>(
            graph.offsets, graph.indices,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int)num_pairs);
    } else {
        jaccard_simple_kernel<<<num_blocks, BLOCK_SIZE, smem_size>>>(
            graph.offsets, graph.indices,
            vertex_pairs_first, vertex_pairs_second,
            similarity_scores, (int)num_pairs);
    }
}

}  
