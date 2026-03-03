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


__device__ __forceinline__ int find_root(int* __restrict__ parent, int v) {
    int curr = v;
    int next = parent[curr];
    while (curr != next) {
        int nn = parent[next];
        parent[curr] = nn; 
        curr = next;
        next = nn;
    }
    return curr;
}


__global__ void __launch_bounds__(256)
kernel_init(int* __restrict__ parent, int n) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += blockDim.x * gridDim.x)
        parent[v] = v;
}






__global__ void __launch_bounds__(256, 8)
kernel_hook_unified(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int* parent,
    int n_non_isolated
) {
    for (int u = blockIdx.x * blockDim.x + threadIdx.x;
         u < n_non_isolated;
         u += blockDim.x * gridDim.x) {

        int start = offsets[u];
        int end = offsets[u + 1];

        for (int e = start; e < end; e++) {
            
            if (!((edge_mask[e >> 5] >> (e & 31)) & 1u)) continue;

            int v = indices[e];

            
            
            if (parent[u] == parent[v]) continue;

            
            int ru = find_root(parent, u);
            int rv = find_root(parent, v);
            while (ru != rv) {
                int big = ru > rv ? ru : rv;
                int small_r = ru > rv ? rv : ru;
                int old = atomicCAS(&parent[big], big, small_r);
                if (old == big) break;
                ru = find_root(parent, u);
                rv = find_root(parent, v);
            }
        }
    }
}






__global__ void __launch_bounds__(512)
kernel_hook_high(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int* parent,
    int v_start, int v_end
) {
    int u = v_start + blockIdx.x;
    if (u >= v_end) return;

    int start = offsets[u];
    int end = offsets[u + 1];

    for (int e = start + threadIdx.x; e < end; e += blockDim.x) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1u) {
            int v = indices[e];
            if (parent[u] == parent[v]) continue;
            int ru = find_root(parent, u);
            int rv = find_root(parent, v);
            while (ru != rv) {
                int big = ru > rv ? ru : rv;
                int small_r = ru > rv ? rv : ru;
                int old = atomicCAS(&parent[big], big, small_r);
                if (old == big) break;
                ru = find_root(parent, u);
                rv = find_root(parent, v);
            }
        }
    }
}


__global__ void __launch_bounds__(256, 8)
kernel_hook_mid(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int* parent,
    int v_start, int v_end
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int u = v_start + warp_id;
    if (u >= v_end) return;

    int start = offsets[u];
    int end = offsets[u + 1];

    for (int e = start + lane; e < end; e += 32) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1u) {
            int v = indices[e];
            if (parent[u] == parent[v]) continue;
            int ru = find_root(parent, u);
            int rv = find_root(parent, v);
            while (ru != rv) {
                int big = ru > rv ? ru : rv;
                int small_r = ru > rv ? rv : ru;
                int old = atomicCAS(&parent[big], big, small_r);
                if (old == big) break;
                ru = find_root(parent, u);
                rv = find_root(parent, v);
            }
        }
    }
}


__global__ void __launch_bounds__(256, 8)
kernel_hook_low(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int* parent,
    int v_start, int v_end
) {
    for (int u = v_start + blockIdx.x * blockDim.x + threadIdx.x;
         u < v_end;
         u += blockDim.x * gridDim.x) {

        int start = offsets[u];
        int end = offsets[u + 1];

        for (int e = start; e < end; e++) {
            if (!((edge_mask[e >> 5] >> (e & 31)) & 1u)) continue;
            int v = indices[e];
            if (parent[u] == parent[v]) continue;
            int ru = find_root(parent, u);
            int rv = find_root(parent, v);
            while (ru != rv) {
                int big = ru > rv ? ru : rv;
                int small_r = ru > rv ? rv : ru;
                int old = atomicCAS(&parent[big], big, small_r);
                if (old == big) break;
                ru = find_root(parent, u);
                rv = find_root(parent, v);
            }
        }
    }
}


__global__ void __launch_bounds__(256, 8)
kernel_flatten(int* __restrict__ parent, int n) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < n; v += blockDim.x * gridDim.x) {
        int curr = v;
        while (parent[curr] != curr) curr = parent[curr];
        parent[v] = curr;
    }
}

}  

void weakly_connected_components_seg_mask(const graph32_t& graph,
                                          int32_t* components) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    const uint32_t* edge_mask = graph.edge_mask;

    const auto& seg = graph.segment_offsets.value();
    int seg0 = seg[0];
    int seg1 = seg[1];
    int seg2 = seg[2];
    int seg3 = seg[3];

    if (num_vertices == 0) return;

    const int THREADS = 256;
    int blocks = (num_vertices + THREADS - 1) / THREADS;

    int n_high = seg1 - seg0;

    if (n_high > 0) {
        
        kernel_init<<<blocks, THREADS>>>(components, num_vertices);

        
        kernel_hook_high<<<n_high, 512>>>(
            offsets, indices, edge_mask, components, seg0, seg1);

        
        int n_mid = seg2 - seg1;
        if (n_mid > 0) {
            int warps_per_block = THREADS / 32;
            int mid_blocks = (n_mid + warps_per_block - 1) / warps_per_block;
            kernel_hook_mid<<<mid_blocks, THREADS>>>(
                offsets, indices, edge_mask, components, seg1, seg2);
        }

        
        int n_low = seg3 - seg2;
        if (n_low > 0) {
            int low_blocks = (n_low + THREADS - 1) / THREADS;
            kernel_hook_low<<<low_blocks, THREADS>>>(
                offsets, indices, edge_mask, components, seg2, seg3);
        }

        
        blocks = (num_vertices + THREADS - 1) / THREADS;
        kernel_flatten<<<blocks, THREADS>>>(components, num_vertices);
    } else {
        
        kernel_init<<<blocks, THREADS>>>(components, num_vertices);

        int n_non_isolated = seg3;
        if (n_non_isolated > 0) {
            int uni_blocks = (n_non_isolated + THREADS - 1) / THREADS;
            kernel_hook_unified<<<uni_blocks, THREADS>>>(
                offsets, indices, edge_mask, components, n_non_isolated);
        }

        blocks = (num_vertices + THREADS - 1) / THREADS;
        kernel_flatten<<<blocks, THREADS>>>(components, num_vertices);
    }
}

}  
