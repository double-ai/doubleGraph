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
#include <climits>

namespace aai {

namespace {

#define BFS_INF 0x7FFFFFFF
#define BLK 256

struct Cache : Cacheable {
    uint32_t* vis = nullptr;
    uint32_t* fbm = nullptr;
    int32_t* f0 = nullptr;
    int32_t* f1 = nullptr;
    int* d_cnt = nullptr;
    int* h_mapped_cnt = nullptr;
    int* d_mapped_cnt = nullptr;

    int64_t vis_capacity = 0;
    int64_t fbm_capacity = 0;
    int64_t f0_capacity = 0;
    int64_t f1_capacity = 0;
    bool cnt_allocated = false;
    bool pinned_allocated = false;

    void ensure(int32_t nv) {
        int bm_w = (nv + 31) / 32;

        if (vis_capacity < bm_w) {
            if (vis) cudaFree(vis);
            cudaMalloc(&vis, bm_w * sizeof(uint32_t));
            vis_capacity = bm_w;
        }
        if (fbm_capacity < bm_w) {
            if (fbm) cudaFree(fbm);
            cudaMalloc(&fbm, bm_w * sizeof(uint32_t));
            fbm_capacity = bm_w;
        }
        if (f0_capacity < nv) {
            if (f0) cudaFree(f0);
            cudaMalloc(&f0, nv * sizeof(int32_t));
            f0_capacity = nv;
        }
        if (f1_capacity < nv) {
            if (f1) cudaFree(f1);
            cudaMalloc(&f1, nv * sizeof(int32_t));
            f1_capacity = nv;
        }
        if (!cnt_allocated) {
            cudaMalloc(&d_cnt, sizeof(int));
            cnt_allocated = true;
        }
        if (!pinned_allocated) {
            cudaHostAlloc(&h_mapped_cnt, sizeof(int), cudaHostAllocMapped);
            cudaHostGetDevicePointer(&d_mapped_cnt, h_mapped_cnt, 0);
            pinned_allocated = true;
        }
    }

    ~Cache() override {
        if (vis) cudaFree(vis);
        if (fbm) cudaFree(fbm);
        if (f0) cudaFree(f0);
        if (f1) cudaFree(f1);
        if (d_cnt) cudaFree(d_cnt);
        if (h_mapped_cnt) cudaFreeHost(h_mapped_cnt);
    }
};



__global__ void init_kernel(
    int32_t* __restrict__ dist, int32_t* __restrict__ pred,
    uint32_t* __restrict__ vis, int nv, int bm_w, bool do_pred)
{
    int tid = blockIdx.x * BLK + threadIdx.x;
    int stride = gridDim.x * BLK;
    for (int i = tid; i < nv; i += stride) dist[i] = BFS_INF;
    if (do_pred && pred)
        for (int i = tid; i < nv; i += stride) pred[i] = -1;
    for (int i = tid; i < bm_w; i += stride) vis[i] = 0u;
}

__global__ void sources_kernel(
    int32_t* __restrict__ dist, int32_t* __restrict__ pred,
    uint32_t* __restrict__ vis, int32_t* __restrict__ frontier,
    const int32_t* __restrict__ srcs, int nsrc, bool do_pred)
{
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < nsrc) {
        int32_t s = srcs[i];
        dist[s] = 0;
        if (do_pred && pred) pred[s] = -1;
        atomicOr(&vis[s >> 5], 1u << (s & 31));
        frontier[i] = s;
    }
}

template <bool DO_PRED>
__global__ void __launch_bounds__(BLK, 6)
td_kernel(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    int32_t* __restrict__ dist, int32_t* __restrict__ pred,
    uint32_t* __restrict__ vis, const int32_t* __restrict__ frt, int frt_sz,
    int32_t* __restrict__ nxt, int* __restrict__ nxt_cnt, int32_t new_d)
{
    int gtid = blockIdx.x * BLK + threadIdx.x;
    int wid = gtid >> 5, lane = gtid & 31;
    int nwarps = (gridDim.x * BLK) >> 5;
    for (int f = wid; f < frt_sz; f += nwarps) {
        int32_t v = frt[f];
        int32_t s = __ldg(&off[v]), e = __ldg(&off[v + 1]);
        for (int j = s + lane; j < e; j += 32) {
            int32_t d = __ldg(&idx[j]);
            uint32_t w = (uint32_t)d >> 5, b = 1u << (d & 31);
            if (__ldg(&vis[w]) & b) continue;
            uint32_t old = atomicOr(&vis[w], b);
            if (old & b) continue;
            dist[d] = new_d;
            if constexpr (DO_PRED) pred[d] = v;
            int p = atomicAdd(nxt_cnt, 1);
            nxt[p] = d;
        }
    }
}

__global__ void set_fbm_kernel(
    const int32_t* __restrict__ frt, uint32_t* __restrict__ fbm, int frt_sz)
{
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < frt_sz) { int32_t v = frt[i]; atomicOr(&fbm[v >> 5], 1u << (v & 31)); }
}

template <bool DO_PRED>
__global__ void __launch_bounds__(BLK)
bu_kernel(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    int32_t* __restrict__ dist, int32_t* __restrict__ pred,
    uint32_t* __restrict__ vis, const uint32_t* __restrict__ fbm,
    int32_t* __restrict__ nxt, int* __restrict__ nxt_cnt, int nv_active, int32_t new_d)
{
    int stride = gridDim.x * BLK;
    for (int v = blockIdx.x * BLK + threadIdx.x; v < nv_active; v += stride) {
        uint32_t w = (uint32_t)v >> 5, b = 1u << (v & 31);
        if (__ldg(&vis[w]) & b) continue;
        int32_t s = __ldg(&off[v]), e = __ldg(&off[v + 1]);
        for (int j = s; j < e; j++) {
            int32_t nb = __ldg(&idx[j]);
            if (__ldg(&fbm[(uint32_t)nb >> 5]) & (1u << (nb & 31))) {
                dist[v] = new_d;
                if constexpr (DO_PRED) pred[v] = nb;
                atomicOr(&vis[w], b);
                int p = atomicAdd(nxt_cnt, 1);
                nxt[p] = v;
                break;
            }
        }
    }
}

__global__ void write_count_kernel(const int* __restrict__ d_cnt, int* __restrict__ mapped_cnt) {
    *mapped_cnt = *d_cnt;
}

static inline int imin(int a, int b) { return a < b ? a : b; }
static inline int imax(int a, int b) { return a > b ? a : b; }

}  

void bfs_seg(const graph32_t& graph,
             int32_t* distances,
             int32_t* predecessors,
             const int32_t* sources,
             std::size_t n_sources,
             int32_t depth_limit) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* off = graph.offsets;
    const int32_t* idx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    bool is_sym = graph.is_symmetric;

    const auto& seg = graph.segment_offsets.value();
    int seg3 = seg[3];

    int nsrc = static_cast<int>(n_sources);
    bool do_pred = (predecessors != nullptr);
    int depth_lim = (depth_limit < 0) ? INT32_MAX : depth_limit;

    if (nv == 0 || nsrc == 0) return;

    cache.ensure(nv);

    uint32_t* vis = cache.vis;
    uint32_t* fbm = cache.fbm;
    int32_t* f0 = cache.f0;
    int32_t* f1 = cache.f1;
    int* d_cnt = cache.d_cnt;
    int* h_mapped_cnt = cache.h_mapped_cnt;
    int* d_mapped_cnt = cache.d_mapped_cnt;

    int bm_w = (nv + 31) / 32;
    int nv_active = (seg3 > 0 && seg3 <= nv) ? seg3 : nv;

    { int g = imin(imax((nv + BLK-1)/BLK, (bm_w + BLK-1)/BLK), 2048);
      init_kernel<<<g, BLK>>>(distances, predecessors, vis, nv, bm_w, do_pred); }
    { int g = imax((nsrc + BLK-1)/BLK, 1);
      sources_kernel<<<g, BLK>>>(distances, predecessors, vis, f0, sources, nsrc, do_pred); }

    int frt_sz = nsrc, depth = 0, cur = 0;
    int32_t* fs[2] = {f0, f1};
    bool topdown = true;
    double avg_deg = (nv > 0) ? (double)ne / nv : 0.0;
    double alpha = avg_deg * 0.3; if (alpha < 3.0) alpha = 3.0;
    int total_vis = nsrc, prev_frt = 0;

    while (frt_sz > 0 && depth < depth_lim) {
        cudaMemsetAsync(d_cnt, 0, sizeof(int), 0);
        int32_t nd = depth + 1;
        int unvis = nv - total_vis;
        bool growing = (frt_sz >= prev_frt);

        if (is_sym && nv > 5000) {
            if (topdown && growing && (double)frt_sz * alpha > (double)unvis) topdown = false;
            else if (!topdown && !growing && (long long)frt_sz * 24 < (long long)unvis) topdown = true;
        }

        if (topdown) {
            int g = imin((frt_sz * 32 + BLK-1) / BLK, 2048);
            g = imax(g, 1);
            if (do_pred) td_kernel<true><<<g,BLK>>>(off,idx,distances,predecessors,vis,fs[cur],frt_sz,fs[1-cur],d_cnt,nd);
            else td_kernel<false><<<g,BLK>>>(off,idx,distances,predecessors,vis,fs[cur],frt_sz,fs[1-cur],d_cnt,nd);
        } else {
            cudaMemsetAsync(fbm, 0, bm_w * sizeof(uint32_t), 0);
            set_fbm_kernel<<<imax((frt_sz+BLK-1)/BLK,1),BLK>>>(fs[cur],fbm,frt_sz);
            if (do_pred) bu_kernel<true><<<imin((nv_active+BLK-1)/BLK,2048),BLK>>>(off,idx,distances,predecessors,vis,fbm,fs[1-cur],d_cnt,nv_active,nd);
            else bu_kernel<false><<<imin((nv_active+BLK-1)/BLK,2048),BLK>>>(off,idx,distances,predecessors,vis,fbm,fs[1-cur],d_cnt,nv_active,nd);
        }

        write_count_kernel<<<1,1>>>(d_cnt, d_mapped_cnt);
        cudaStreamSynchronize(0);
        frt_sz = *h_mapped_cnt;

        prev_frt = frt_sz;
        total_vis += frt_sz;
        cur = 1 - cur;
        depth++;
    }
}

}  
