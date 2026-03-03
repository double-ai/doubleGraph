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
#include <cstring>
#include <algorithm>
#include <vector>

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)

static inline __device__ int pop_lsb_64(uint64_t &x) {
  int b = __ffsll((long long)x) - 1;
  x &= (x - 1);
  return b;
}



__global__ void init_sources_kernel_128(
    const int32_t* __restrict__ sources,
    int32_t n_sources,
    uint64_t* __restrict__ visited,
    uint64_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_q,
    int32_t* __restrict__ union_q,
    int32_t* __restrict__ union_flag,
    int32_t* __restrict__ last_iter,
    int32_t* __restrict__ frontier_count,
    int32_t* __restrict__ union_count)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid == 0) {
        *frontier_count = n_sources;
        *union_count = n_sources;
    }
    if (tid >= n_sources) return;

    int32_t v = sources[tid];
    uint64_t lo = 0, hi = 0;
    if (tid < 64) lo = 1ull << tid;
    else hi = 1ull << (tid - 64);

    int64_t v2 = (int64_t)v * 2;
    visited[v2] = lo;
    visited[v2 + 1] = hi;
    frontier[v2] = lo;
    frontier[v2 + 1] = hi;

    frontier_q[tid] = v;
    union_q[tid] = v;
    union_flag[v] = 1;
    last_iter[v] = 0;
}

__global__ __launch_bounds__(BLOCK_SIZE, 4)
void bfs_expand_kernel_128(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    uint64_t* __restrict__ visited,
    uint64_t* __restrict__ frontier_in,
    uint64_t* __restrict__ frontier_out,
    const int32_t* __restrict__ frontier_q_in,
    int32_t* __restrict__ frontier_q_out,
    const int32_t* __restrict__ frontier_count_in,
    int32_t* __restrict__ frontier_count_out,
    int32_t* __restrict__ last_iter,
    int32_t* __restrict__ union_flag,
    int32_t* __restrict__ union_q,
    int32_t* __restrict__ union_count,
    int32_t iter,
    int32_t num_vertices)
{
    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int global_warp = (int)(blockIdx.x * WARPS_PER_BLOCK + warp_in_block);
    int total_warps = (int)(gridDim.x * WARPS_PER_BLOCK);

    int32_t fcount = *frontier_count_in;
    for (int32_t fidx = global_warp; fidx < fcount; fidx += total_warps) {
        int32_t u;
        if (lane == 0) u = frontier_q_in[fidx];
        u = __shfl_sync(0xffffffff, u, 0);
        int64_t u2 = (int64_t)u * 2;

        unsigned long long mask_lo, mask_hi;
        int32_t start, end;
        if (lane == 0) {
            mask_lo = frontier_in[u2];
            mask_hi = frontier_in[u2 + 1];
            start = csr_offsets[u];
            end = csr_offsets[u + 1];
        }
        mask_lo = __shfl_sync(0xffffffff, mask_lo, 0);
        mask_hi = __shfl_sync(0xffffffff, mask_hi, 0);
        start = __shfl_sync(0xffffffff, start, 0);
        end = __shfl_sync(0xffffffff, end, 0);

        if (((uint64_t)mask_lo | (uint64_t)mask_hi) == 0ull) continue;

        for (int32_t e = start + lane; e < end; e += 32) {
            int32_t v = csr_indices[e];
            if ((uint32_t)v >= (uint32_t)num_vertices) continue;
            int64_t v2 = (int64_t)v * 2;

            unsigned long long old_lo = atomicOr((unsigned long long*)&visited[v2], mask_lo);
            unsigned long long old_hi = atomicOr((unsigned long long*)&visited[v2 + 1], mask_hi);
            uint64_t add_lo = ((uint64_t)mask_lo) & ~((uint64_t)old_lo);
            uint64_t add_hi = ((uint64_t)mask_hi) & ~((uint64_t)old_hi);
            if ((add_lo | add_hi) == 0ull) continue;

            if (add_lo) atomicOr((unsigned long long*)&frontier_out[v2], (unsigned long long)add_lo);
            if (add_hi) atomicOr((unsigned long long*)&frontier_out[v2 + 1], (unsigned long long)add_hi);

            int32_t prev = atomicExch(&last_iter[v], iter);
            if (prev != iter) {
                int32_t pos = atomicAdd(frontier_count_out, 1);
                frontier_q_out[pos] = v;
            }
            if (atomicCAS((unsigned int*)&union_flag[v], 0u, 1u) == 0u) {
                int32_t upos = atomicAdd(union_count, 1);
                union_q[upos] = v;
            }
        }

        if (lane == 0) {
            frontier_in[u2] = 0ull;
            frontier_in[u2 + 1] = 0ull;
        }
    }
}

__global__ __launch_bounds__(BLOCK_SIZE, 4)
void count_edges_block_kernel_128(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const uint64_t* __restrict__ visited,
    const int32_t* __restrict__ union_q,
    const int32_t* __restrict__ union_count,
    int32_t* __restrict__ block_counts,
    int32_t n_sources,
    int32_t num_vertices)
{
    __shared__ int32_t sh_counts[128];

    for (int i = threadIdx.x; i < n_sources; i += blockDim.x) sh_counts[i] = 0;
    __syncthreads();

    const ulonglong2* __restrict__ vis2 = (const ulonglong2*)visited;
    const unsigned long long* __restrict__ vis64 = (const unsigned long long*)visited;

    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int global_warp = (int)(blockIdx.x * WARPS_PER_BLOCK + warp_in_block);
    int total_warps = (int)(gridDim.x * WARPS_PER_BLOCK);

    int32_t ucount = *union_count;
    uint64_t valid_hi_mask = 0ull;
    if (n_sources > 64) {
        int b = n_sources - 64;
        valid_hi_mask = (b >= 64) ? 0xFFFFFFFFFFFFFFFFull : ((1ull << b) - 1ull);
    }

    for (int32_t idx = global_warp; idx < ucount; idx += total_warps) {
        int32_t u;
        if (lane == 0) u = union_q[idx];
        u = __shfl_sync(0xffffffff, u, 0);
        if ((uint32_t)u >= (uint32_t)num_vertices) continue;

        unsigned long long u_lo, u_hi;
        int32_t start, end;
        if (lane == 0) {
            ulonglong2 uv = vis2[u];
            u_lo = uv.x;
            u_hi = uv.y & valid_hi_mask;
            start = csr_offsets[u];
            end = csr_offsets[u + 1];
        }
        u_lo = __shfl_sync(0xffffffff, u_lo, 0);
        u_hi = __shfl_sync(0xffffffff, u_hi, 0);
        start = __shfl_sync(0xffffffff, start, 0);
        end = __shfl_sync(0xffffffff, end, 0);

        if (((uint64_t)u_lo | (uint64_t)u_hi) == 0ull) continue;

        bool single_lo = (((uint64_t)u_hi) == 0ull) && (((uint64_t)u_lo) != 0ull) && ((((uint64_t)u_lo) & (((uint64_t)u_lo) - 1ull)) == 0ull);
        bool single_hi = (((uint64_t)u_lo) == 0ull) && (((uint64_t)u_hi) != 0ull) && ((((uint64_t)u_hi) & (((uint64_t)u_hi) - 1ull)) == 0ull);
        int sb = -1;
        unsigned long long s_mask = 0ull;
        if (single_lo) {
            sb = __ffsll((long long)u_lo) - 1;
            s_mask = (unsigned long long)u_lo;
        } else if (single_hi) {
            sb = (__ffsll((long long)u_hi) - 1) + 64;
            s_mask = (unsigned long long)u_hi;
        }

        for (int32_t e = start + lane; e < end; e += 32) {
            int32_t v = csr_indices[e];
            if ((uint32_t)v >= (uint32_t)num_vertices) continue;

            if (sb >= 0) {
                if (sb < 64) {
                    unsigned long long v_lo = vis64[((int64_t)v << 1)];
                    if (v_lo & s_mask) { atomicAdd(&sh_counts[sb], 1); }
                } else {
                    unsigned long long v_hi = vis64[((int64_t)v << 1) + 1] & (unsigned long long)valid_hi_mask;
                    if (v_hi & s_mask) { if (sb < n_sources) atomicAdd(&sh_counts[sb], 1); }
                }
            } else {
                ulonglong2 vv = vis2[v];
                uint64_t common_lo = ((uint64_t)u_lo) & (uint64_t)vv.x;
                uint64_t common_hi = ((uint64_t)u_hi) & ((uint64_t)vv.y & valid_hi_mask);
                if ((common_lo | common_hi) == 0ull) continue;

                while (common_lo) {
                    int b = pop_lsb_64(common_lo);
                    atomicAdd(&sh_counts[b], 1);
                }
                while (common_hi) {
                    int b = pop_lsb_64(common_hi) + 64;
                    if (b < n_sources) atomicAdd(&sh_counts[b], 1);
                }
            }
        }
    }

    __syncthreads();
    for (int s = threadIdx.x; s < n_sources; s += blockDim.x) {
        block_counts[(int64_t)blockIdx.x * n_sources + s] = sh_counts[s];
    }
}

__global__ void scan_block_counts_kernel(
    const int32_t* __restrict__ block_counts,
    int64_t* __restrict__ block_offsets,
    int64_t* __restrict__ source_totals,
    int32_t n_blocks,
    int32_t n_sources)
{
    int s = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (s >= n_sources) return;
    int64_t prefix = 0;
    for (int b = 0; b < n_blocks; ++b) {
        int32_t c = block_counts[(int64_t)b * n_sources + s];
        block_offsets[(int64_t)b * n_sources + s] = prefix;
        prefix += (int64_t)c;
    }
    source_totals[s] = prefix;
}

__global__ void scan_source_totals_kernel(
    const int64_t* __restrict__ source_totals,
    int64_t* __restrict__ source_bases,
    int32_t n_sources)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int64_t sum = 0;
    source_bases[0] = 0;
    for (int s = 0; s < n_sources; ++s) {
        sum += source_totals[s];
        source_bases[s + 1] = sum;
    }
}

__global__ __launch_bounds__(BLOCK_SIZE, 4)
void write_edges_block_kernel_128(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const uint64_t* __restrict__ visited,
    const int32_t* __restrict__ union_q,
    const int32_t* __restrict__ union_count,
    const int64_t* __restrict__ block_offsets,
    const int64_t* __restrict__ source_bases,
    int32_t n_sources,
    int32_t num_vertices,
    int32_t* __restrict__ out_srcs,
    int32_t* __restrict__ out_dsts)
{
    __shared__ int32_t sh_pos[128];
    __shared__ int64_t sh_base[128];

    for (int s = threadIdx.x; s < n_sources; s += blockDim.x) {
        sh_pos[s] = 0;
        sh_base[s] = source_bases[s] + block_offsets[(int64_t)blockIdx.x * n_sources + s];
    }
    __syncthreads();

    const ulonglong2* __restrict__ vis2 = (const ulonglong2*)visited;
    const unsigned long long* __restrict__ vis64 = (const unsigned long long*)visited;

    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int global_warp = (int)(blockIdx.x * WARPS_PER_BLOCK + warp_in_block);
    int total_warps = (int)(gridDim.x * WARPS_PER_BLOCK);

    int32_t ucount = *union_count;
    uint64_t valid_hi_mask = 0ull;
    if (n_sources > 64) {
        int b = n_sources - 64;
        valid_hi_mask = (b >= 64) ? 0xFFFFFFFFFFFFFFFFull : ((1ull << b) - 1ull);
    }

    for (int32_t idx = global_warp; idx < ucount; idx += total_warps) {
        int32_t u;
        if (lane == 0) u = union_q[idx];
        u = __shfl_sync(0xffffffff, u, 0);
        if ((uint32_t)u >= (uint32_t)num_vertices) continue;

        unsigned long long u_lo, u_hi;
        int32_t start, end;
        if (lane == 0) {
            ulonglong2 uv = vis2[u];
            u_lo = uv.x;
            u_hi = uv.y & valid_hi_mask;
            start = csr_offsets[u];
            end = csr_offsets[u + 1];
        }
        u_lo = __shfl_sync(0xffffffff, u_lo, 0);
        u_hi = __shfl_sync(0xffffffff, u_hi, 0);
        start = __shfl_sync(0xffffffff, start, 0);
        end = __shfl_sync(0xffffffff, end, 0);

        if (((uint64_t)u_lo | (uint64_t)u_hi) == 0ull) continue;

        bool single_lo = (((uint64_t)u_hi) == 0ull) && (((uint64_t)u_lo) != 0ull) && ((((uint64_t)u_lo) & (((uint64_t)u_lo) - 1ull)) == 0ull);
        bool single_hi = (((uint64_t)u_lo) == 0ull) && (((uint64_t)u_hi) != 0ull) && ((((uint64_t)u_hi) & (((uint64_t)u_hi) - 1ull)) == 0ull);
        int sb = -1;
        unsigned long long s_mask = 0ull;
        if (single_lo) {
            sb = __ffsll((long long)u_lo) - 1;
            s_mask = (unsigned long long)u_lo;
        } else if (single_hi) {
            sb = (__ffsll((long long)u_hi) - 1) + 64;
            s_mask = (unsigned long long)u_hi;
        }

        for (int32_t e = start + lane; e < end; e += 32) {
            int32_t v = csr_indices[e];
            if ((uint32_t)v >= (uint32_t)num_vertices) continue;

            if (sb >= 0) {
                if (sb < 64) {
                    unsigned long long v_lo = vis64[((int64_t)v << 1)];
                    if (v_lo & s_mask) {
                        int64_t pos = sh_base[sb] + (int64_t)atomicAdd(&sh_pos[sb], 1);
                        out_srcs[pos] = u;
                        out_dsts[pos] = v;
                    }
                } else {
                    unsigned long long v_hi = vis64[((int64_t)v << 1) + 1] & (unsigned long long)valid_hi_mask;
                    if (v_hi & s_mask) {
                        if (sb < n_sources) {
                            int64_t pos = sh_base[sb] + (int64_t)atomicAdd(&sh_pos[sb], 1);
                            out_srcs[pos] = u;
                            out_dsts[pos] = v;
                        }
                    }
                }
            } else {
                ulonglong2 vv = vis2[v];
                uint64_t common_lo = ((uint64_t)u_lo) & (uint64_t)vv.x;
                uint64_t common_hi = ((uint64_t)u_hi) & ((uint64_t)vv.y & valid_hi_mask);
                if ((common_lo | common_hi) == 0ull) continue;

                while (common_lo) {
                    int b = pop_lsb_64(common_lo);
                    int64_t pos = sh_base[b] + (int64_t)atomicAdd(&sh_pos[b], 1);
                    out_srcs[pos] = u;
                    out_dsts[pos] = v;
                }
                while (common_hi) {
                    int b = pop_lsb_64(common_hi) + 64;
                    if (b < n_sources) {
                        int64_t pos = sh_base[b] + (int64_t)atomicAdd(&sh_pos[b], 1);
                        out_srcs[pos] = u;
                        out_dsts[pos] = v;
                    }
                }
            }
        }
    }
}



__global__ void init_sources_kernel_n(
    const int32_t* __restrict__ sources,
    int32_t n_sources,
    int32_t word_count,
    uint64_t* __restrict__ visited,
    uint64_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_q,
    int32_t* __restrict__ union_q,
    int32_t* __restrict__ union_flag,
    int32_t* __restrict__ last_iter,
    int32_t* __restrict__ frontier_count,
    int32_t* __restrict__ union_count)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid == 0) {
        *frontier_count = n_sources;
        *union_count = n_sources;
    }
    if (tid >= n_sources) return;

    int32_t v = sources[tid];
    int32_t w = tid >> 6;
    int32_t b = tid & 63;
    uint64_t mask = 1ull << b;
    int64_t base = (int64_t)v * word_count;
    atomicOr((unsigned long long*)&visited[base + w], (unsigned long long)mask);
    atomicOr((unsigned long long*)&frontier[base + w], (unsigned long long)mask);

    frontier_q[tid] = v;
    union_q[tid] = v;
    union_flag[v] = 1;
    last_iter[v] = 0;
}

__global__ __launch_bounds__(BLOCK_SIZE, 4)
void bfs_expand_kernel_n(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    uint64_t* __restrict__ visited,
    uint64_t* __restrict__ frontier_in,
    uint64_t* __restrict__ frontier_out,
    const int32_t* __restrict__ frontier_q_in,
    int32_t* __restrict__ frontier_q_out,
    const int32_t* __restrict__ frontier_count_in,
    int32_t* __restrict__ frontier_count_out,
    int32_t* __restrict__ last_iter,
    int32_t* __restrict__ union_flag,
    int32_t* __restrict__ union_q,
    int32_t* __restrict__ union_count,
    int32_t iter,
    int32_t num_vertices,
    int32_t n_sources,
    int32_t word_count)
{
    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int global_warp = (int)(blockIdx.x * WARPS_PER_BLOCK + warp_in_block);
    int total_warps = (int)(gridDim.x * WARPS_PER_BLOCK);

    int32_t fcount = *frontier_count_in;
    for (int32_t fidx = global_warp; fidx < fcount; fidx += total_warps) {
        int32_t u;
        if (lane == 0) u = frontier_q_in[fidx];
        u = __shfl_sync(0xffffffff, u, 0);
        int64_t ubase = (int64_t)u * word_count;

        bool any_mask = false;
        for (int w = 0; w < word_count; ++w) {
            if (frontier_in[ubase + w] != 0ull) { any_mask = true; break; }
        }
        if (!any_mask) continue;

        int32_t start, end;
        if (lane == 0) {
            start = csr_offsets[u];
            end = csr_offsets[u + 1];
        }
        start = __shfl_sync(0xffffffff, start, 0);
        end = __shfl_sync(0xffffffff, end, 0);

        for (int32_t e = start + lane; e < end; e += 32) {
            int32_t v = csr_indices[e];
            if ((uint32_t)v >= (uint32_t)num_vertices) continue;
            int64_t vbase = (int64_t)v * word_count;

            bool any_added = false;
            for (int w = 0; w < word_count; ++w) {
                uint64_t m = frontier_in[ubase + w];
                if (m == 0ull) continue;
                unsigned long long old = atomicOr((unsigned long long*)&visited[vbase + w], (unsigned long long)m);
                uint64_t add = m & ~((uint64_t)old);
                if (add) {
                    atomicOr((unsigned long long*)&frontier_out[vbase + w], (unsigned long long)add);
                    any_added = true;
                }
            }

            if (!any_added) continue;

            int32_t prev = atomicExch(&last_iter[v], iter);
            if (prev != iter) {
                int32_t pos = atomicAdd(frontier_count_out, 1);
                frontier_q_out[pos] = v;
            }
            if (atomicCAS((unsigned int*)&union_flag[v], 0u, 1u) == 0u) {
                int32_t upos = atomicAdd(union_count, 1);
                union_q[upos] = v;
            }
        }

        if (lane == 0) {
            for (int w = 0; w < word_count; ++w) frontier_in[ubase + w] = 0ull;
        }
    }
}

__global__ __launch_bounds__(BLOCK_SIZE, 4)
void count_edges_kernel_n(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const uint64_t* __restrict__ visited,
    int32_t word_count,
    const int32_t* __restrict__ union_q,
    int32_t union_count,
    unsigned long long* __restrict__ counts,
    int32_t n_sources,
    int32_t num_vertices)
{
    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int global_warp = (int)(blockIdx.x * WARPS_PER_BLOCK + warp_in_block);
    int total_warps = (int)(gridDim.x * WARPS_PER_BLOCK);

    int last_bits = n_sources & 63;
    uint64_t last_mask = (last_bits == 0) ? 0xFFFFFFFFFFFFFFFFull : ((1ull << last_bits) - 1ull);

    for (int32_t idx = global_warp; idx < union_count; idx += total_warps) {
        int32_t u;
        if (lane == 0) u = union_q[idx];
        u = __shfl_sync(0xffffffff, u, 0);
        if ((uint32_t)u >= (uint32_t)num_vertices) continue;
        int64_t ubase = (int64_t)u * word_count;

        int32_t start, end;
        if (lane == 0) {
            start = csr_offsets[u];
            end = csr_offsets[u + 1];
        }
        start = __shfl_sync(0xffffffff, start, 0);
        end = __shfl_sync(0xffffffff, end, 0);

        for (int32_t e = start + lane; e < end; e += 32) {
            int32_t v = csr_indices[e];
            if ((uint32_t)v >= (uint32_t)num_vertices) continue;
            int64_t vbase = (int64_t)v * word_count;

            for (int w = 0; w < word_count; ++w) {
                uint64_t common = visited[ubase + w] & visited[vbase + w];
                if (w == word_count - 1) common &= last_mask;
                while (common) {
                    int b = pop_lsb_64(common);
                    int s = (w << 6) + b;
                    if (s < n_sources) atomicAdd(&counts[s], 1ull);
                }
            }
        }
    }
}

__global__ __launch_bounds__(BLOCK_SIZE, 4)
void write_edges_kernel_n(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const uint64_t* __restrict__ visited,
    int32_t word_count,
    const int32_t* __restrict__ union_q,
    int32_t union_count,
    unsigned long long* __restrict__ write_ptrs,
    int32_t n_sources,
    int32_t num_vertices,
    int32_t* __restrict__ out_srcs,
    int32_t* __restrict__ out_dsts)
{
    int warp_in_block = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int global_warp = (int)(blockIdx.x * WARPS_PER_BLOCK + warp_in_block);
    int total_warps = (int)(gridDim.x * WARPS_PER_BLOCK);

    int last_bits = n_sources & 63;
    uint64_t last_mask = (last_bits == 0) ? 0xFFFFFFFFFFFFFFFFull : ((1ull << last_bits) - 1ull);

    for (int32_t idx = global_warp; idx < union_count; idx += total_warps) {
        int32_t u;
        if (lane == 0) u = union_q[idx];
        u = __shfl_sync(0xffffffff, u, 0);
        if ((uint32_t)u >= (uint32_t)num_vertices) continue;
        int64_t ubase = (int64_t)u * word_count;

        int32_t start, end;
        if (lane == 0) {
            start = csr_offsets[u];
            end = csr_offsets[u + 1];
        }
        start = __shfl_sync(0xffffffff, start, 0);
        end = __shfl_sync(0xffffffff, end, 0);

        for (int32_t e = start + lane; e < end; e += 32) {
            int32_t v = csr_indices[e];
            if ((uint32_t)v >= (uint32_t)num_vertices) continue;
            int64_t vbase = (int64_t)v * word_count;

            for (int w = 0; w < word_count; ++w) {
                uint64_t common = visited[ubase + w] & visited[vbase + w];
                if (w == word_count - 1) common &= last_mask;
                while (common) {
                    int b = pop_lsb_64(common);
                    int s = (w << 6) + b;
                    if (s < n_sources) {
                        unsigned long long pos = atomicAdd(&write_ptrs[s], 1ull);
                        out_srcs[pos] = u;
                        out_dsts[pos] = v;
                    }
                }
            }
        }
    }
}



static inline int grid_blocks() { return 108 * 8; }



struct Cache : Cacheable {
    
    int32_t* q_a = nullptr;        int64_t q_a_cap = 0;
    int32_t* q_b = nullptr;        int64_t q_b_cap = 0;
    int32_t* union_q = nullptr;    int64_t union_q_cap = 0;
    int32_t* last_iter = nullptr;  int64_t last_iter_cap = 0;
    int32_t* union_flag = nullptr; int64_t union_flag_cap = 0;

    
    uint64_t* visited = nullptr;   int64_t visited_cap = 0;
    uint64_t* frontier_a = nullptr; int64_t frontier_a_cap = 0;
    uint64_t* frontier_b = nullptr; int64_t frontier_b_cap = 0;

    
    int32_t* frontier_count_a = nullptr;
    int32_t* frontier_count_b = nullptr;
    int32_t* union_count_buf = nullptr;

    
    int32_t* block_counts = nullptr;    int64_t block_counts_cap = 0;
    int64_t* block_offsets_buf = nullptr; int64_t block_offsets_cap = 0;
    int64_t* source_totals = nullptr;   int64_t source_totals_cap = 0;

    
    int64_t* counts = nullptr;      int64_t counts_cap = 0;
    int64_t* write_ptrs_buf = nullptr; int64_t write_ptrs_cap = 0;

    template <typename T>
    static void ensure_buf(T*& ptr, int64_t& cap, int64_t need) {
        if (cap < need) {
            if (ptr) cudaFree(ptr);
            cudaMalloc(&ptr, (size_t)need * sizeof(T));
            cap = need;
        }
    }

    Cache() {
        cudaMalloc(&frontier_count_a, sizeof(int32_t));
        cudaMalloc(&frontier_count_b, sizeof(int32_t));
        cudaMalloc(&union_count_buf, sizeof(int32_t));
    }

    ~Cache() override {
        if (q_a) cudaFree(q_a);
        if (q_b) cudaFree(q_b);
        if (union_q) cudaFree(union_q);
        if (last_iter) cudaFree(last_iter);
        if (union_flag) cudaFree(union_flag);
        if (visited) cudaFree(visited);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (frontier_count_a) cudaFree(frontier_count_a);
        if (frontier_count_b) cudaFree(frontier_count_b);
        if (union_count_buf) cudaFree(union_count_buf);
        if (block_counts) cudaFree(block_counts);
        if (block_offsets_buf) cudaFree(block_offsets_buf);
        if (source_totals) cudaFree(source_totals);
        if (counts) cudaFree(counts);
        if (write_ptrs_buf) cudaFree(write_ptrs_buf);
    }

    void ensure(int32_t nv, int64_t mask_elems, int32_t ns, bool fast128) {
        ensure_buf(q_a, q_a_cap, (int64_t)nv);
        ensure_buf(q_b, q_b_cap, (int64_t)nv);
        ensure_buf(union_q, union_q_cap, (int64_t)nv);
        ensure_buf(last_iter, last_iter_cap, (int64_t)nv);
        ensure_buf(union_flag, union_flag_cap, (int64_t)nv);

        ensure_buf(visited, visited_cap, mask_elems);
        ensure_buf(frontier_a, frontier_a_cap, mask_elems);
        ensure_buf(frontier_b, frontier_b_cap, mask_elems);

        if (fast128) {
            int64_t kBlocks = (int64_t)grid_blocks();
            ensure_buf(block_counts, block_counts_cap, kBlocks * ns);
            ensure_buf(block_offsets_buf, block_offsets_cap, kBlocks * ns);
            ensure_buf(source_totals, source_totals_cap, (int64_t)ns);
        } else {
            ensure_buf(counts, counts_cap, (int64_t)ns);
            ensure_buf(write_ptrs_buf, write_ptrs_cap, (int64_t)ns);
        }
    }
};

}  

extract_ego_result_t extract_ego_seg(const graph32_t& graph,
                                     const int32_t* source_vertices,
                                     std::size_t n_sources_sz,
                                     int32_t radius) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t ns = (int32_t)n_sources_sz;
    cudaStream_t stream = 0;

    
    if (ns == 0) {
        int64_t* offsets_ptr;
        cudaMalloc(&offsets_ptr, sizeof(int64_t));
        int64_t z = 0;
        cudaMemcpyAsync(offsets_ptr, &z, sizeof(int64_t), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        extract_ego_result_t result;
        result.edge_srcs = nullptr;
        result.edge_dsts = nullptr;
        result.offsets = reinterpret_cast<std::size_t*>(offsets_ptr);
        result.num_edges = 0;
        result.num_offsets = 1;
        return result;
    }

    bool fast128 = (ns <= 128);
    int32_t word_count = fast128 ? 2 : (int32_t)((ns + 63) / 64);
    int64_t mask_elems = (int64_t)num_vertices * (int64_t)word_count;

    cache.ensure(num_vertices, mask_elems, ns, fast128);

    uint64_t* d_visited = cache.visited;
    uint64_t* d_frontier_a = cache.frontier_a;
    uint64_t* d_frontier_b = cache.frontier_b;
    int32_t* d_q_a = cache.q_a;
    int32_t* d_q_b = cache.q_b;
    int32_t* d_union_q = cache.union_q;
    int32_t* d_last_iter = cache.last_iter;
    int32_t* d_union_flag = cache.union_flag;
    int32_t* d_frontier_count_a = cache.frontier_count_a;
    int32_t* d_frontier_count_b = cache.frontier_count_b;
    int32_t* d_union_count = cache.union_count_buf;

    
    cudaMemsetAsync(d_visited, 0, (size_t)mask_elems * sizeof(uint64_t), stream);
    cudaMemsetAsync(d_frontier_a, 0, (size_t)mask_elems * sizeof(uint64_t), stream);
    cudaMemsetAsync(d_frontier_b, 0, (size_t)mask_elems * sizeof(uint64_t), stream);
    cudaMemsetAsync(d_last_iter, 0xFF, (size_t)num_vertices * sizeof(int32_t), stream);
    cudaMemsetAsync(d_union_flag, 0, (size_t)num_vertices * sizeof(int32_t), stream);

    
    if (fast128) {
        int blocks = (ns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (blocks < 1) blocks = 1;
        init_sources_kernel_128<<<blocks, BLOCK_SIZE, 0, stream>>>(
            source_vertices, ns, d_visited, d_frontier_a,
            d_q_a, d_union_q, d_union_flag, d_last_iter,
            d_frontier_count_a, d_union_count);
    } else {
        int blocks = (ns + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (blocks < 1) blocks = 1;
        init_sources_kernel_n<<<blocks, BLOCK_SIZE, 0, stream>>>(
            source_vertices, ns, word_count, d_visited, d_frontier_a,
            d_q_a, d_union_q, d_union_flag, d_last_iter,
            d_frontier_count_a, d_union_count);
    }

    
    uint64_t* frt_in = d_frontier_a;
    uint64_t* frt_out = d_frontier_b;
    int32_t* q_in = d_q_a;
    int32_t* q_out = d_q_b;
    int32_t* cnt_in = d_frontier_count_a;
    int32_t* cnt_out = d_frontier_count_b;

    int gb = grid_blocks();

    for (int iter = 1; iter <= radius; ++iter) {
        cudaMemsetAsync(cnt_out, 0, sizeof(int32_t), stream);
        if (fast128) {
            bfs_expand_kernel_128<<<gb, BLOCK_SIZE, 0, stream>>>(
                d_offsets, d_indices, d_visited,
                frt_in, frt_out,
                q_in, q_out,
                cnt_in, cnt_out,
                d_last_iter, d_union_flag,
                d_union_q, d_union_count,
                iter, num_vertices);
        } else {
            bfs_expand_kernel_n<<<gb, BLOCK_SIZE, 0, stream>>>(
                d_offsets, d_indices, d_visited,
                frt_in, frt_out,
                q_in, q_out,
                cnt_in, cnt_out,
                d_last_iter, d_union_flag,
                d_union_q, d_union_count,
                iter, num_vertices, ns, word_count);
        }
        std::swap(frt_in, frt_out);
        std::swap(q_in, q_out);
        std::swap(cnt_in, cnt_out);
    }

    
    if (fast128) {
        int32_t kBlocks = gb;

        count_edges_block_kernel_128<<<kBlocks, BLOCK_SIZE, 0, stream>>>(
            d_offsets, d_indices, d_visited,
            d_union_q, d_union_count,
            cache.block_counts,
            ns, num_vertices);

        {
            int threads = 128;
            int scan_blocks = (ns + threads - 1) / threads;
            scan_block_counts_kernel<<<scan_blocks, threads, 0, stream>>>(
                cache.block_counts, cache.block_offsets_buf,
                cache.source_totals, kBlocks, ns);
        }

        
        int64_t* out_offsets;
        cudaMalloc(&out_offsets, (size_t)(ns + 1) * sizeof(int64_t));

        scan_source_totals_kernel<<<1, 1, 0, stream>>>(
            cache.source_totals, out_offsets, ns);

        int64_t total_edges = 0;
        cudaMemcpyAsync(&total_edges, out_offsets + ns,
                         sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        
        int32_t* out_srcs = nullptr;
        int32_t* out_dsts = nullptr;
        if (total_edges > 0) {
            cudaMalloc(&out_srcs, (size_t)total_edges * sizeof(int32_t));
            cudaMalloc(&out_dsts, (size_t)total_edges * sizeof(int32_t));

            write_edges_block_kernel_128<<<kBlocks, BLOCK_SIZE, 0, stream>>>(
                d_offsets, d_indices, d_visited,
                d_union_q, d_union_count,
                cache.block_offsets_buf, out_offsets,
                ns, num_vertices,
                out_srcs, out_dsts);
        }

        extract_ego_result_t result;
        result.edge_srcs = out_srcs;
        result.edge_dsts = out_dsts;
        result.offsets = reinterpret_cast<std::size_t*>(out_offsets);
        result.num_edges = (std::size_t)total_edges;
        result.num_offsets = (std::size_t)(ns + 1);
        return result;

    } else {
        
        int32_t h_union_count = 0;
        cudaMemcpyAsync(&h_union_count, d_union_count, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        int64_t* d_counts = cache.counts;
        cudaMemsetAsync(d_counts, 0, (size_t)ns * sizeof(int64_t), stream);
        if (h_union_count > 0) {
            count_edges_kernel_n<<<gb, BLOCK_SIZE, 0, stream>>>(
                d_offsets, d_indices, d_visited, word_count,
                d_union_q, h_union_count,
                (unsigned long long*)d_counts,
                ns, num_vertices);
        }

        std::vector<int64_t> h_counts(ns);
        cudaMemcpyAsync(h_counts.data(), d_counts, (size_t)ns * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::vector<int64_t> h_offsets(ns + 1);
        h_offsets[0] = 0;
        for (int i = 0; i < ns; ++i) h_offsets[i + 1] = h_offsets[i] + h_counts[i];
        int64_t total_edges = h_offsets[ns];

        
        int32_t* out_srcs = nullptr;
        int32_t* out_dsts = nullptr;
        int64_t* out_offsets;
        cudaMalloc(&out_offsets, (size_t)(ns + 1) * sizeof(int64_t));

        cudaMemcpyAsync(out_offsets, h_offsets.data(),
                         (size_t)(ns + 1) * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

        if (total_edges > 0) {
            cudaMalloc(&out_srcs, (size_t)total_edges * sizeof(int32_t));
            cudaMalloc(&out_dsts, (size_t)total_edges * sizeof(int32_t));
        }

        if (total_edges > 0 && h_union_count > 0) {
            int64_t* d_write_ptrs = cache.write_ptrs_buf;
            cudaMemcpyAsync(d_write_ptrs, h_offsets.data(),
                             (size_t)ns * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
            write_edges_kernel_n<<<gb, BLOCK_SIZE, 0, stream>>>(
                d_offsets, d_indices, d_visited, word_count,
                d_union_q, h_union_count,
                (unsigned long long*)d_write_ptrs, ns, num_vertices,
                out_srcs, out_dsts);
        }

        extract_ego_result_t result;
        result.edge_srcs = out_srcs;
        result.edge_dsts = out_dsts;
        result.offsets = reinterpret_cast<std::size_t*>(out_offsets);
        result.num_edges = (std::size_t)total_edges;
        result.num_offsets = (std::size_t)(ns + 1);
        return result;
    }
}

}  
