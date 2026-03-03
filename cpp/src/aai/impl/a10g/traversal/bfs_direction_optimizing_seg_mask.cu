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
#include <algorithm>

namespace aai {

namespace {

#define WARP_SIZE 32

struct Cache : Cacheable {
    int32_t* frontier_a = nullptr;
    int64_t frontier_a_capacity = 0;

    int32_t* frontier_b = nullptr;
    int64_t frontier_b_capacity = 0;

    uint32_t* vis_bm = nullptr;
    int64_t vis_bm_capacity = 0;

    int32_t* cnt_buf = nullptr;
    int64_t cnt_buf_capacity = 0;

    void ensure(int32_t nv) {
        int64_t nv64 = nv;
        int64_t bm_words = (nv64 + 31) / 32;

        if (frontier_a_capacity < nv64) {
            if (frontier_a) cudaFree(frontier_a);
            cudaMalloc(&frontier_a, nv64 * sizeof(int32_t));
            frontier_a_capacity = nv64;
        }
        if (frontier_b_capacity < nv64) {
            if (frontier_b) cudaFree(frontier_b);
            cudaMalloc(&frontier_b, nv64 * sizeof(int32_t));
            frontier_b_capacity = nv64;
        }
        if (vis_bm_capacity < bm_words) {
            if (vis_bm) cudaFree(vis_bm);
            cudaMalloc(&vis_bm, bm_words * sizeof(uint32_t));
            vis_bm_capacity = bm_words;
        }
        if (cnt_buf_capacity < 4) {
            if (cnt_buf) cudaFree(cnt_buf);
            cudaMalloc(&cnt_buf, 4 * sizeof(int32_t));
            cnt_buf_capacity = 4;
        }
    }

    ~Cache() override {
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (vis_bm) cudaFree(vis_bm);
        if (cnt_buf) cudaFree(cnt_buf);
    }
};

__device__ __forceinline__ bool edge_active(const uint32_t* __restrict__ mask, int32_t e) {
    uint32_t val;
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(val) : "l"(mask + ((uint32_t)e >> 5)));
    return (val >> (e & 31)) & 1u;
}

__device__ __forceinline__ bool bit_test(const uint32_t* __restrict__ bm, int32_t v) {
    uint32_t val;
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(val) : "l"(bm + ((uint32_t)v >> 5)));
    return (val >> (v & 31)) & 1u;
}

__global__ void __launch_bounds__(256, 8) k_init_arrays(
    int32_t* __restrict__ dist, int32_t* __restrict__ pred,
    uint32_t* __restrict__ vis, int32_t nv, bool do_pred
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < nv; i += stride) dist[i] = 0x7FFFFFFF;
    if (do_pred) for (int i = tid; i < nv; i += stride) pred[i] = -1;
    int bw = (nv + 31) >> 5;
    for (int i = tid; i < bw; i += stride) vis[i] = 0;
}

__global__ void __launch_bounds__(256, 8) k_init_sources(
    int32_t* __restrict__ dist, uint32_t* __restrict__ vis,
    int32_t* __restrict__ frontier, const int32_t* __restrict__ src, int32_t ns
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < ns) {
        int32_t s = src[tid];
        dist[s] = 0;
        frontier[tid] = s;
        uint32_t wi = (uint32_t)s >> 5;
        uint32_t bt = 1u << (s & 31);
        atomicOr(&vis[wi], bt);
    }
}


__global__ void __launch_bounds__(256, 8)
k_td(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ idx,
    const uint32_t* __restrict__ mask,
    int32_t* __restrict__ dist,
    int32_t* __restrict__ pred,
    uint32_t* __restrict__ vis,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ nf,
    int32_t* __restrict__ cnt,
    int32_t fsize, int32_t depth, bool do_pred
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= fsize) return;

    int32_t v = frontier[warp_id];
    int32_t start = off[v];
    int32_t end = off[v + 1];
    int32_t nd = depth + 1;

    
    for (int32_t e = start + lane; e < end; e += WARP_SIZE) {
        
        uint32_t mask_val;
        asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(mask_val) : "l"(mask + ((uint32_t)e >> 5)));
        if ((mask_val >> (e & 31)) & 1u) {
            int32_t nb;
            asm volatile("ld.global.nc.s32 %0, [%1];" : "=r"(nb) : "l"(idx + e));
            uint32_t wi = (uint32_t)nb >> 5;
            uint32_t bt = 1u << (nb & 31);
            uint32_t old;
            asm volatile("atom.or.b32 %0, [%1], %2;" : "=r"(old) : "l"(vis + wi), "r"(bt));
            if (!(old & bt)) {
                dist[nb] = nd;
                if (do_pred) pred[nb] = v;
                int32_t pos = atomicAdd(cnt, 1);
                nf[pos] = nb;
            }
        }
    }
}


__global__ void __launch_bounds__(256, 8)
k_bu(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ idx,
    const uint32_t* __restrict__ mask,
    int32_t* __restrict__ dist,
    int32_t* __restrict__ pred,
    const uint32_t* __restrict__ vis,
    int32_t* __restrict__ nf,
    int32_t* __restrict__ cnt,
    int32_t nv, int32_t depth, bool do_pred
) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv) return;
    
    uint32_t vis_val;
    asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(vis_val) : "l"(vis + ((uint32_t)v >> 5)));
    if ((vis_val >> (v & 31)) & 1u) return;

    int32_t start = off[v];
    int32_t end = off[v + 1];
    if (start == end) return;
    int32_t nd = depth + 1;

    for (int32_t e = start; e < end; e++) {
        uint32_t mask_val;
        asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(mask_val) : "l"(mask + ((uint32_t)e >> 5)));
        if ((mask_val >> (e & 31)) & 1u) {
            int32_t nb;
            asm volatile("ld.global.nc.s32 %0, [%1];" : "=r"(nb) : "l"(idx + e));
            uint32_t vis_nb;
            asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(vis_nb) : "l"(vis + ((uint32_t)nb >> 5)));
            if ((vis_nb >> (nb & 31)) & 1u) {
                dist[v] = nd;
                if (do_pred) pred[v] = nb;
                int32_t pos = atomicAdd(cnt, 1);
                nf[pos] = v;
                return;
            }
        }
    }
}

__global__ void __launch_bounds__(256, 8) k_upd(
    uint32_t* __restrict__ vis,
    const int32_t* __restrict__ frontier,
    int32_t count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) {
        int32_t v = frontier[tid];
        uint32_t wi = (uint32_t)v >> 5;
        uint32_t bt = 1u << (v & 31);
        atomicOr(&vis[wi], bt);
    }
}


__global__ void __launch_bounds__(256, 8) k_degsum(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ verts,
    int64_t* __restrict__ result,
    int32_t count
) {
    int64_t sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        int32_t v = verts[i];
        sum += (int64_t)(off[v + 1] - off[v]);
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    if ((threadIdx.x & 31) == 0 && sum > 0)
        atomicAdd((unsigned long long*)result, (unsigned long long)sum);
}

}  

void bfs_direction_optimizing_seg_mask(const graph32_t& graph,
                                       int32_t* distances,
                                       int32_t* predecessors,
                                       const int32_t* sources,
                                       std::size_t n_sources,
                                       int32_t depth_limit) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* off = graph.offsets;
    const int32_t* idx = graph.indices;
    const uint32_t* mask = graph.edge_mask;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    int32_t ns = static_cast<int32_t>(n_sources);
    bool do_pred = (predecessors != nullptr);

    if (depth_limit < 0) depth_limit = 0x7FFFFFFF;

    cache.ensure(nv);

    int32_t* fa = cache.frontier_a;
    int32_t* fb = cache.frontier_b;
    uint32_t* vis = cache.vis_bm;
    int32_t* d_scratch = cache.cnt_buf;

    cudaStream_t stream = 0;

    {
        int nb = std::min((nv + 255) / 256, 2048);
        k_init_arrays<<<nb, 256, 0, stream>>>(distances, predecessors, vis, nv, do_pred);
        nb = std::max((ns + 255) / 256, 1);
        k_init_sources<<<nb, 256, 0, stream>>>(distances, vis, fa, sources, ns);
    }

    int32_t frontier_size = ns;
    int32_t* frontier = fa;
    int32_t* next_frontier = fb;
    int32_t depth = 0;

    bool is_td = true;
    double avg_deg = (nv > 0) ? (double)ne / (double)nv : 1.0;
    double alpha_factor = 0.5;
    int32_t beta = 24;
    int64_t visited_count = ns;

    double m_u = (double)ne;

    int32_t* d_cnt = d_scratch;
    int64_t* d_degsum = (int64_t*)(d_scratch + 2);

    while (frontier_size > 0) {
        cudaMemsetAsync(d_cnt, 0, sizeof(int32_t), stream);

        if (is_td) {
            int warps = frontier_size;
            int tpb = 256;
            int wpb = tpb / WARP_SIZE;
            int blocks = (warps + wpb - 1) / wpb;
            if (blocks > 0) {
                k_td<<<blocks, tpb, 0, stream>>>(
                    off, idx, mask, distances, predecessors, vis,
                    frontier, next_frontier, d_cnt,
                    frontier_size, depth, do_pred
                );
            }
        } else {
            int blocks = (nv + 255) / 256;
            k_bu<<<blocks, 256, 0, stream>>>(
                off, idx, mask, distances, predecessors, vis,
                next_frontier, d_cnt,
                nv, depth, do_pred
            );
        }

        int32_t next_size;
        cudaMemcpy(&next_size, d_cnt, sizeof(int32_t), cudaMemcpyDeviceToHost);

        if (!is_td && next_size > 0) {
            int ub = (next_size + 255) / 256;
            k_upd<<<ub, 256, 0, stream>>>(vis, next_frontier, next_size);
        }

        visited_count += next_size;
        int64_t unvisited = (int64_t)nv - visited_count;

        if (is_td) {
            if (next_size >= frontier_size && next_size > 128) {
                cudaMemsetAsync(d_degsum, 0, sizeof(int64_t), stream);
                int db = std::min((next_size + 255) / 256, 2048);
                k_degsum<<<db, 256, 0, stream>>>(off, next_frontier, d_degsum, next_size);
                int64_t h_mf;
                cudaMemcpy(&h_mf, d_degsum, sizeof(int64_t), cudaMemcpyDeviceToHost);

                double m_f = (double)h_mf;
                m_u -= m_f;
                if (m_u < 0.0) m_u = 0.0;

                if (m_f * avg_deg * alpha_factor > m_u) {
                    is_td = false;
                }
            } else {
                m_u -= (double)next_size * avg_deg;
                if (m_u < 0.0) m_u = 0.0;
            }
        } else {
            m_u -= (double)next_size * avg_deg;
            if (m_u < 0.0) m_u = 0.0;

            if ((int64_t)next_size * beta < unvisited && next_size < frontier_size) {
                is_td = true;
            }
        }

        int32_t* tmp = frontier; frontier = next_frontier; next_frontier = tmp;
        frontier_size = next_size;
        depth++;
        if (depth >= depth_limit) break;
    }

    cudaStreamSynchronize(stream);
}

}  
