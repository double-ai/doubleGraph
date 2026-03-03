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

namespace aai {

namespace {

struct Cache : Cacheable {};

__device__ __forceinline__ bool edge_active(const uint32_t* __restrict__ mask, int32_t idx) {
    return (__ldg(&mask[idx >> 5]) >> (idx & 31)) & 1u;
}

__device__ __forceinline__ int32_t find(int32_t* __restrict__ parent, int32_t i) {
    int32_t curr = i;
    int32_t next = parent[curr];
    while (next != curr) {
        int32_t nn = parent[next];
        if (nn != next) parent[curr] = nn;
        curr = next;
        next = nn;
    }
    return curr;
}

__device__ __forceinline__ void unite(int32_t* __restrict__ parent, int32_t a, int32_t b) {
    int32_t ra = find(parent, a);
    int32_t rb = find(parent, b);
    while (ra != rb) {
        if (ra > rb) { int32_t t = ra; ra = rb; rb = t; }
        int32_t old = atomicCAS(&parent[rb], rb, ra);
        if (old == rb) return;
        rb = find(parent, rb);
        ra = find(parent, ra);
    }
}

__global__ void kernel_init(int32_t* __restrict__ comp, int32_t n) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t base = idx * 4;
    if (base + 3 < n) {
        reinterpret_cast<int4*>(comp)[idx] = make_int4(base, base+1, base+2, base+3);
    } else {
        for (int32_t i = base; i < n && i < base + 4; i++) comp[i] = i;
    }
}


__global__ __launch_bounds__(1024, 1)
void kernel_hook_unified(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ mask,
    int32_t* __restrict__ comp,
    int32_t s0, int32_t s1, int32_t s2, int32_t s3,
    int32_t blocks_low, int32_t blocks_mid)
{
    int32_t bid = blockIdx.x;

    if (bid < blocks_low) {
        int32_t v = s2 + bid * blockDim.x + threadIdx.x;
        if (v >= s3) return;
        int32_t e_begin = __ldg(&offsets[v]);
        int32_t e_end = __ldg(&offsets[v + 1]);
        int32_t cv = comp[v];
        for (int32_t e = e_begin; e < e_end; ++e) {
            if (edge_active(mask, e)) {
                int32_t u = __ldg(&indices[e]);
                int32_t cu = comp[u];
                if (cv != cu) {
                    unite(comp, v, u);
                    cv = comp[v];
                }
            }
        }
        return;
    }
    bid -= blocks_low;

    if (bid < blocks_mid) {
        int32_t warps_per_block = blockDim.x >> 5;
        int32_t warp_in_block = threadIdx.x >> 5;
        int32_t lane = threadIdx.x & 31;
        int32_t warp_id = bid * warps_per_block + warp_in_block;
        int32_t v = s1 + warp_id;
        if (v >= s2) return;
        int32_t e_begin = __ldg(&offsets[v]);
        int32_t e_end = __ldg(&offsets[v + 1]);
        for (int32_t e = e_begin + lane; e < e_end; e += 32) {
            if (edge_active(mask, e)) {
                int32_t u = __ldg(&indices[e]);
                if (comp[v] != comp[u])
                    unite(comp, v, u);
            }
        }
        return;
    }
    bid -= blocks_mid;

    {
        int32_t v = s0 + bid;
        if (v >= s1) return;
        int32_t e_begin = __ldg(&offsets[v]);
        int32_t e_end = __ldg(&offsets[v + 1]);
        for (int32_t e = e_begin + threadIdx.x; e < e_end; e += blockDim.x) {
            if (edge_active(mask, e)) {
                int32_t u = __ldg(&indices[e]);
                if (comp[v] != comp[u])
                    unite(comp, v, u);
            }
        }
    }
}

__global__ void kernel_compress(int32_t* __restrict__ comp, int32_t n) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int32_t c = comp[v];
    if (c == v) return;
    int32_t cc = comp[c];
    if (cc == c) { comp[v] = c; return; }
    while (cc != comp[cc]) cc = comp[cc];
    comp[v] = cc;
}

}  

void weakly_connected_components_seg_mask(const graph32_t& graph,
                                          int32_t* components) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    const uint32_t* edge_mask = graph.edge_mask;
    int32_t num_vertices = graph.number_of_vertices;

    if (num_vertices == 0) return;

    constexpr int BLK = 1024;

    
    size_t comp_bytes = (size_t)num_vertices * sizeof(int32_t);
    if (comp_bytes > 0) {
        cudaStreamAttrValue attr;
        memset(&attr, 0, sizeof(attr));
        attr.accessPolicyWindow.base_ptr = (void*)components;
        
        size_t pin_bytes = (comp_bytes < 4u * 1024 * 1024) ? comp_bytes : 4u * 1024 * 1024;
        attr.accessPolicyWindow.num_bytes = pin_bytes;
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
    }

    int init_threads = (num_vertices + 3) / 4;
    kernel_init<<<(init_threads + BLK - 1) / BLK, BLK>>>(components, num_vertices);

    const auto& seg = graph.segment_offsets.value();
    int32_t s0 = seg[0], s1 = seg[1], s2 = seg[2], s3 = seg[3];

    int32_t blocks_high = s1 - s0;
    int32_t n_mid = s2 - s1;
    int32_t warps_per_block = BLK >> 5;
    int32_t blocks_mid = (n_mid + warps_per_block - 1) / warps_per_block;
    int32_t n_low = s3 - s2;
    int32_t blocks_low = (n_low + BLK - 1) / BLK;
    int32_t total_blocks = blocks_low + blocks_mid + blocks_high;

    if (total_blocks > 0) {
        kernel_hook_unified<<<total_blocks, BLK>>>(
            offsets, indices, edge_mask, components,
            s0, s1, s2, s3, blocks_low, blocks_mid);
    }

    int gv = (num_vertices + BLK - 1) / BLK;
    kernel_compress<<<gv, BLK>>>(components, num_vertices);

    
    if (comp_bytes > 0) {
        cudaStreamAttrValue attr;
        memset(&attr, 0, sizeof(attr));
        attr.accessPolicyWindow.num_bytes = 0;
        cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &attr);
    }
}

}  
