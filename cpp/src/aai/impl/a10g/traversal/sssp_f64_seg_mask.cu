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
#include <cfloat>
#include <cstdint>
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    uint32_t* fa = nullptr;
    uint32_t* fb = nullptr;
    int32_t* changed = nullptr;
    int64_t fa_capacity = 0;
    int64_t fb_capacity = 0;
    bool changed_allocated = false;

    void ensure(int32_t bitmap_words) {
        if (fa_capacity < bitmap_words) {
            if (fa) cudaFree(fa);
            cudaMalloc(&fa, bitmap_words * sizeof(uint32_t));
            fa_capacity = bitmap_words;
        }
        if (fb_capacity < bitmap_words) {
            if (fb) cudaFree(fb);
            cudaMalloc(&fb, bitmap_words * sizeof(uint32_t));
            fb_capacity = bitmap_words;
        }
        if (!changed_allocated) {
            cudaMalloc(&changed, sizeof(int32_t));
            changed_allocated = true;
        }
    }

    ~Cache() override {
        if (fa) cudaFree(fa);
        if (fb) cudaFree(fb);
        if (changed) cudaFree(changed);
    }
};


__global__ void init_sssp_kernel(
    double* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    uint32_t* __restrict__ frontier_a,
    uint32_t* __restrict__ frontier_b,
    int32_t num_vertices,
    int32_t bitmap_words,
    int32_t source
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        distances[idx] = (idx == source) ? 0.0 : DBL_MAX;
        predecessors[idx] = -1;
    }
    if (idx < bitmap_words) {
        frontier_a[idx] = (idx == (source >> 5)) ? (1u << (source & 31)) : 0u;
        frontier_b[idx] = 0;
    }
}

__device__ __forceinline__ bool atomicMin_double(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) <= val) return false;
        old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return true;
}


__global__ void relax_high_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ distances,
    uint32_t* __restrict__ frontier_in,
    uint32_t* __restrict__ frontier_out,
    int32_t* __restrict__ changed,
    double cutoff,
    int32_t seg_start, int32_t seg_end
) {
    int32_t vid = seg_start + blockIdx.x;
    if (vid >= seg_end) return;

    __shared__ int s_active;
    __shared__ double s_dist;
    __shared__ int32_t s_start, s_end;

    if (threadIdx.x == 0) {
        uint32_t word = frontier_in[vid >> 5];
        s_active = (word >> (vid & 31)) & 1;
        if (s_active) {
            atomicAnd(&frontier_in[vid >> 5], ~(1u << (vid & 31)));
            s_dist = distances[vid];
            s_start = __ldg(&offsets[vid]);
            s_end = __ldg(&offsets[vid + 1]);
            if (s_dist >= cutoff) s_active = 0;
        }
    }
    __syncthreads();
    if (!s_active) return;

    double dist_u = s_dist;
    for (int32_t e = s_start + threadIdx.x; e < s_end; e += blockDim.x) {
        uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
        if (!((mask_word >> (e & 31)) & 1)) continue;
        int32_t v = __ldg(&indices[e]);
        double w = __ldg(&weights[e]);
        double nd = dist_u + w;
        if (nd >= cutoff) continue;
        if (atomicMin_double(&distances[v], nd)) {
            atomicOr(&frontier_out[v >> 5], 1u << (v & 31));
            *changed = 1;
        }
    }
}


__global__ void relax_mid_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ distances,
    uint32_t* __restrict__ frontier_in,
    uint32_t* __restrict__ frontier_out,
    int32_t* __restrict__ changed,
    double cutoff,
    int32_t seg_start, int32_t seg_end
) {
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;
    int32_t vid = seg_start + warp_id;
    if (vid >= seg_end) return;

    int active = 0;
    double dist_u = 0.0;
    int32_t start = 0, end = 0;

    if (lane == 0) {
        uint32_t word = frontier_in[vid >> 5];
        active = (word >> (vid & 31)) & 1;
        if (active) {
            atomicAnd(&frontier_in[vid >> 5], ~(1u << (vid & 31)));
            dist_u = distances[vid];
            start = __ldg(&offsets[vid]);
            end = __ldg(&offsets[vid + 1]);
            if (dist_u >= cutoff) active = 0;
        }
    }
    active = __shfl_sync(0xffffffff, active, 0);
    if (!active) return;
    dist_u = __shfl_sync(0xffffffff, dist_u, 0);
    start = __shfl_sync(0xffffffff, start, 0);
    end = __shfl_sync(0xffffffff, end, 0);

    for (int32_t e = start + lane; e < end; e += 32) {
        uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
        if (!((mask_word >> (e & 31)) & 1)) continue;
        int32_t v = __ldg(&indices[e]);
        double w = __ldg(&weights[e]);
        double nd = dist_u + w;
        if (nd >= cutoff) continue;
        if (atomicMin_double(&distances[v], nd)) {
            atomicOr(&frontier_out[v >> 5], 1u << (v & 31));
            *changed = 1;
        }
    }
}


__global__ void relax_low_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ distances,
    uint32_t* __restrict__ frontier_in,
    uint32_t* __restrict__ frontier_out,
    int32_t* __restrict__ changed,
    double cutoff,
    int32_t seg_start, int32_t seg_end
) {
    int32_t vid = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= seg_end) return;

    uint32_t word = frontier_in[vid >> 5];
    if (!((word >> (vid & 31)) & 1)) return;

    atomicAnd(&frontier_in[vid >> 5], ~(1u << (vid & 31)));

    double dist_u = distances[vid];
    if (dist_u >= cutoff) return;

    int32_t start = __ldg(&offsets[vid]);
    int32_t end = __ldg(&offsets[vid + 1]);

    for (int32_t e = start; e < end; e++) {
        uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
        if (!((mask_word >> (e & 31)) & 1)) continue;
        int32_t v = __ldg(&indices[e]);
        double w = __ldg(&weights[e]);
        double nd = dist_u + w;
        if (nd >= cutoff) continue;
        if (atomicMin_double(&distances[v], nd)) {
            atomicOr(&frontier_out[v >> 5], 1u << (v & 31));
            *changed = 1;
        }
    }
}


__global__ void pred_high_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t seg_start, int32_t seg_end
) {
    int32_t u = seg_start + blockIdx.x;
    if (u >= seg_end) return;

    __shared__ double s_dist;
    __shared__ int32_t s_start, s_end;
    __shared__ int s_active;

    if (threadIdx.x == 0) {
        s_dist = distances[u];
        s_active = (s_dist < 1e300) ? 1 : 0;
        if (s_active) {
            s_start = offsets[u];
            s_end = offsets[u + 1];
        }
    }
    __syncthreads();
    if (!s_active) return;

    double dist_u = s_dist;
    for (int32_t e = s_start + threadIdx.x; e < s_end; e += blockDim.x) {
        uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
        if (!((mask_word >> (e & 31)) & 1)) continue;
        int32_t v = __ldg(&indices[e]);
        double w = __ldg(&weights[e]);
        double dist_v = distances[v];
        if (dist_v == dist_u + w && dist_u < dist_v) {
            predecessors[v] = u;
        }
    }
}

__global__ void pred_low_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t seg_start, int32_t seg_end
) {
    int u = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= seg_end) return;

    double dist_u = distances[u];
    if (dist_u >= 1e300) return;

    int32_t start = __ldg(&offsets[u]);
    int32_t end = __ldg(&offsets[u + 1]);

    for (int32_t e = start; e < end; e++) {
        uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
        if (!((mask_word >> (e & 31)) & 1)) continue;
        int32_t v = __ldg(&indices[e]);
        double w = __ldg(&weights[e]);
        double dist_v = distances[v];
        if (dist_v == dist_u + w && dist_u < dist_v) {
            predecessors[v] = u;
        }
    }
}


__global__ void pred_zerow_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ distances,
    int32_t* __restrict__ predecessors,
    int32_t* __restrict__ changed,
    int32_t source,
    int32_t num_vertices
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;

    double dist_u = distances[u];
    if (dist_u >= 1e300) return;

    
    if (u != source && predecessors[u] == -1) return;

    int32_t start = __ldg(&offsets[u]);
    int32_t end = __ldg(&offsets[u + 1]);

    for (int32_t e = start; e < end; e++) {
        uint32_t mask_word = __ldg(&edge_mask[e >> 5]);
        if (!((mask_word >> (e & 31)) & 1)) continue;
        int32_t v = __ldg(&indices[e]);
        double w = __ldg(&weights[e]);
        
        if (dist_u + w != distances[v] || dist_u != distances[v]) continue;
        
        if (predecessors[v] == -1 && v != source) {
            predecessors[v] = u;
            *changed = 1;
        }
    }
}

}  

void sssp_seg_mask(const graph32_t& graph,
                   const double* edge_weights,
                   int32_t source,
                   double* distances,
                   int32_t* predecessors,
                   double cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    const uint32_t* d_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int32_t s0 = seg[0], s1 = seg[1], s2 = seg[2], s3 = seg[3];

    int32_t bw = (nv + 31) / 32;
    cache.ensure(bw);

    uint32_t* fa = cache.fa;
    uint32_t* fb = cache.fb;
    int32_t* d_ch = cache.changed;

    init_sssp_kernel<<<(nv + 255) / 256, 256>>>(distances, predecessors, fa, fb, nv, bw, source);

    bool has_high = s1 > s0;
    bool has_mid = s2 > s1;

    uint32_t* cur = fa;
    uint32_t* nxt = fb;
    int32_t h_ch = 1;

    int batch = 1;
    for (int iter = 0; iter < nv && h_ch; ) {
        cudaMemset(d_ch, 0, sizeof(int32_t));

        int be = std::min(iter + batch, (int)nv);
        for (; iter < be; iter++) {
            if (has_high) {
                if (s1 > s0)
                    relax_high_kernel<<<s1 - s0, 256>>>(d_off, d_idx, edge_weights, d_mask, distances, cur, nxt, d_ch, cutoff, s0, s1);
            }
            if (has_mid) {
                int n = s2 - s1;
                if (n > 0)
                    relax_mid_kernel<<<(n * 32 + 255) / 256, 256>>>(d_off, d_idx, edge_weights, d_mask, distances, cur, nxt, d_ch, cutoff, s1, s2);
            }
            {
                int n = s3 - s2;
                if (n > 0)
                    relax_low_kernel<<<(n + 255) / 256, 256>>>(d_off, d_idx, edge_weights, d_mask, distances, cur, nxt, d_ch, cutoff, s2, s3);
            }
            uint32_t* t = cur; cur = nxt; nxt = t;
        }

        cudaMemcpy(&h_ch, d_ch, sizeof(int32_t), cudaMemcpyDeviceToHost);

        if (batch < 128) batch *= 2;
    }

    
    if (s2 > s0)
        pred_high_kernel<<<s2 - s0, 256>>>(d_off, d_idx, edge_weights, d_mask, distances, predecessors, s0, s2);
    {
        int n = s3 - s2;
        if (n > 0)
            pred_low_kernel<<<(n + 255) / 256, 256>>>(d_off, d_idx, edge_weights, d_mask, distances, predecessors, s2, s3);
    }

    
    h_ch = 1;
    for (int ziter = 0; ziter < nv && h_ch; ziter++) {
        cudaMemset(d_ch, 0, sizeof(int32_t));
        int grid = (nv + 255) / 256;
        pred_zerow_kernel<<<grid, 256>>>(
            d_off, d_idx, edge_weights, d_mask, distances, predecessors,
            d_ch, source, nv);
        cudaMemcpy(&h_ch, d_ch, sizeof(int32_t), cudaMemcpyDeviceToHost);
    }
}

}  
