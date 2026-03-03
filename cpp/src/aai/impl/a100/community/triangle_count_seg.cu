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


#define TC_BLOCKS_PER_VERTEX 24




__device__ __forceinline__ int d_lower_bound(const int* __restrict__ arr, int size, int target) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int v = arr[mid];
        if (v < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__device__ __forceinline__ int d_lower_bound_from(const int* __restrict__ arr, int lo, int hi, int target) {
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int v = arr[mid];
        if (v < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}


__device__ __forceinline__ void merge_path_search(
    const int* __restrict__ A, int m,
    const int* __restrict__ B, int n,
    int d, int &i, int &j)
{
    int lo = max(0, d - n);
    int hi = min(d, m);
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        int bj = d - mid - 1;
        
        if (bj >= 0 && mid < m && A[mid] <= B[bj]) lo = mid + 1;
        else hi = mid;
    }
    i = lo;
    j = d - lo;
}







__device__ __forceinline__ int warp_intersect_count(
    const int* __restrict__ a, int size_a,
    const int* __restrict__ b, int size_b,
    int skip_u, int skip_v)
{
    int lane = threadIdx.x & 31;
    if (size_a == 0 || size_b == 0) return 0;

    
    if (size_a > size_b) {
        const int* t = a; a = b; b = t;
        int s = size_a; size_a = size_b; size_b = s;
    }

    
    int a0 = a[0];
    int a1 = a[size_a - 1];
    int b0 = b[0];
    int b1 = b[size_b - 1];
    if (a1 < b0 || a0 > b1) return 0;

    
    
    int start_a = 0, start_b = 0;
    if (lane == 0) {
        start_a = (a0 < b0) ? d_lower_bound(a, size_a, b0) : 0;
        if (start_a < size_a) {
            int needle = a[start_a];
            start_b = (b0 < needle) ? d_lower_bound(b, size_b, needle) : 0;
        } else {
            start_b = size_b;
        }
    }
    start_a = __shfl_sync(0xffffffff, start_a, 0);
    start_b = __shfl_sync(0xffffffff, start_b, 0);
    if (start_a >= size_a || start_b >= size_b) return 0;
    a += start_a;
    b += start_b;
    size_a -= start_a;
    size_b -= start_b;

    
    a0 = a[0];
    a1 = a[size_a - 1];
    b0 = b[0];
    b1 = b[size_b - 1];
    if (a1 < b0 || a0 > b1) return 0;

    
    int ratio = size_b / max(size_a, 1);
    
    bool use_probe = true;

    int total = 0;

    if (use_probe) {
        
        
        int pos = 0;
        for (int base = 0; base < size_a; base += 32) {
            int idx = base + lane;
            bool found = false;
            if (idx < size_a) {
                int val = a[idx];
                if (val != skip_u && val != skip_v) {
                    
                    if (!(val < b0 || val > b1)) {
                        pos = d_lower_bound_from(b, pos, size_b, val);
                        found = (pos < size_b && b[pos] == val);
                        if (found) pos++;
                    }
                }
            }
            total += __popc(__ballot_sync(0xffffffff, found));
        }
        return (lane == 0) ? total : 0;
    } else {
        
        
        
        
        int m = size_a;
        int n = size_b;
        int L = m + n;
        int chunk = (L + 31) >> 5;  
        int d0 = min(lane * chunk, L);
        int d1 = min(d0 + chunk, L);

        int ia0, ib0, ia1, ib1;
        merge_path_search(a, m, b, n, d0, ia0, ib0);
        merge_path_search(a, m, b, n, d1, ia1, ib1);

        
        int prev_val = 0;
        int prev_src = -1;
        if (d0 > 0) {
            if (ia0 == 0) {
                prev_val = b[ib0 - 1];
                prev_src = 1;
            } else if (ib0 == 0) {
                prev_val = a[ia0 - 1];
                prev_src = 0;
            } else {
                int a_last = a[ia0 - 1];
                int b_last = b[ib0 - 1];
                if (a_last <= b_last) {
                    prev_val = a_last;
                    prev_src = 0;
                } else {
                    prev_val = b_last;
                    prev_src = 1;
                }
            }
        }

        int i = ia0;
        int j = ib0;
        int c = 0;

        
        for (int out = d0; out < d1; ++out) {
            bool takeA = (i < ia1) && ((j >= ib1) || (a[i] <= b[j]));
            int val = takeA ? a[i++] : b[j++];
            int src = takeA ? 0 : 1;

            
            if (src == 1 && prev_src == 0 && val == prev_val) {
                if (val != skip_u && val != skip_v) c++;
            }
            prev_val = val;
            prev_src = src;
        }

        unsigned mask = 0xffffffff;
        c += __shfl_down_sync(mask, c, 16);
        c += __shfl_down_sync(mask, c, 8);
        c += __shfl_down_sync(mask, c, 4);
        c += __shfl_down_sync(mask, c, 2);
        c += __shfl_down_sync(mask, c, 1);
        return (lane == 0) ? c : 0;
    }
}





__device__ __forceinline__ int scalar_intersect_count(
    const int* __restrict__ a, int size_a,
    const int* __restrict__ b, int size_b,
    int skip_u, int skip_v)
{
    if (size_a == 0 || size_b == 0) return 0;
    if (size_a > size_b) {
        const int* t = a; a = b; b = t;
        int s = size_a; size_a = size_b; size_b = s;
        int tmp = skip_u; skip_u = skip_v; skip_v = tmp; 
    }

    int a0 = a[0], a1 = a[size_a - 1];
    int b0 = b[0], b1 = b[size_b - 1];
    if (a1 < b0 || a0 > b1) return 0;

    
    int start_a = (a0 < b0) ? d_lower_bound(a, size_a, b0) : 0;
    if (start_a >= size_a) return 0;
    int needle = a[start_a];
    int start_b = (b0 < needle) ? d_lower_bound(b, size_b, needle) : 0;
    if (start_b >= size_b) return 0;
    a += start_a; b += start_b; size_a -= start_a; size_b -= start_b;

    int ratio = size_b / max(size_a, 1);

    int c = 0;
    if (ratio <= 2) {
        
        int i = 0, j = 0;
        while (i < size_a && j < size_b) {
            int va = a[i];
            int vb = b[j];
            if (va == vb) {
                if (va != skip_u && va != skip_v) c++;
                i++; j++;
            } else if (va < vb) {
                i++;
            } else {
                j++;
            }
        }
    } else {
        
        int j = 0;
        for (int i = 0; i < size_a && j < size_b; ++i) {
            int target = a[i];
            if (target == skip_u || target == skip_v) continue;
            if (target < b0 || target > b1) continue;
            j = d_lower_bound_from(b, j, size_b, target);
            if (j < size_b && b[j] == target) {
                c++;
                j++;
            }
        }
    }
    return c;
}




#define TC_HASH_EMPTY 0xFFFFFFFFu
#define TC_HASH_SIZE 8192  // 32KB shared
#define TC_HASH_MASK (TC_HASH_SIZE - 1)
#define TC_HASH_MIN_DEG 256
#define TC_HASH_MAX_DEG (TC_HASH_SIZE / 2)

__device__ __forceinline__ uint32_t tc_hash32(uint32_t x) {
    return x * 2654435761u;
}

__device__ __forceinline__ void tc_hash_insert(int* __restrict__ table, uint32_t key) {
    uint32_t slot = tc_hash32(key) & TC_HASH_MASK;
    #pragma unroll 1
    while (true) {
        uint32_t prev = (uint32_t)atomicCAS((unsigned int*)&table[slot], TC_HASH_EMPTY, key);
        if (prev == TC_HASH_EMPTY || prev == key) return;
        slot = (slot + 1) & TC_HASH_MASK;
    }
}

__device__ __forceinline__ bool tc_hash_find(const int* __restrict__ table, uint32_t key) {
    uint32_t slot = tc_hash32(key) & TC_HASH_MASK;
    #pragma unroll 1
    while (true) {
        uint32_t v = (uint32_t)table[slot];
        if (v == key) return true;
        if (v == TC_HASH_EMPTY) return false;
        slot = (slot + 1) & TC_HASH_MASK;
    }
}






__global__ void tc_block_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ counts,
    int start, int end)
{
    int bid = (int)blockIdx.x;
    int u = start + (bid / TC_BLOCKS_PER_VERTEX);
    if (u >= end) return;
    int block_in_u = bid - (bid / TC_BLOCKS_PER_VERTEX) * TC_BLOCKS_PER_VERTEX;

    int us = __ldg(&offsets[u]);
    int ue = __ldg(&offsets[u + 1]);
    int ud = ue - us;
    const int* __restrict__ u_nbrs = indices + us;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = (int)blockDim.x >> 5;

    int warp_in_u = block_in_u * num_warps + warp_id;
    int warps_total = TC_BLOCKS_PER_VERTEX * num_warps;

    
    __shared__ int s_e0;
    if (tid == 0) {
        s_e0 = d_lower_bound(u_nbrs, ud, u + 1);
    }
    __syncthreads();
    int e0 = s_e0;

    int local = 0;
    for (int e = e0 + warp_in_u; e < ud; e += warps_total) {
        int v = __ldg(&u_nbrs[e]);
        int vs = __ldg(&offsets[v]);
        int vd = __ldg(&offsets[v + 1]) - vs;
        int c = warp_intersect_count(u_nbrs, ud, indices + vs, vd, u, v);
        if (lane == 0 && c) {
            local += c;
            atomicAdd(&counts[v], c);
        }
    }

    __shared__ int warp_sums[16];
    if (lane == 0) warp_sums[warp_id] = local;
    __syncthreads();

    if (tid == 0) {
        int s = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (w < num_warps) s += warp_sums[w];
        }
        for (int w = 8; w < num_warps; ++w) s += warp_sums[w];
        if (s) atomicAdd(&counts[u], s);
    }
}



__global__ void tc_warp_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ counts,
    int start, int end)
{
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int u = start + warp_global;
    if (u >= end) return;

    int us = __ldg(&offsets[u]);
    int ue = __ldg(&offsets[u + 1]);
    int ud = ue - us;
    const int* __restrict__ u_nbrs = indices + us;

    
    int e0;
    if (lane == 0) e0 = d_lower_bound(u_nbrs, ud, u + 1);
    e0 = __shfl_sync(0xffffffff, e0, 0);

    int local = 0;
    for (int e = e0 + lane; e < ud; e += 32) {
        int v = __ldg(&u_nbrs[e]);
        int vs = __ldg(&offsets[v]);
        int vd = __ldg(&offsets[v + 1]) - vs;
        int c = scalar_intersect_count(u_nbrs, ud, indices + vs, vd, u, v);
        if (c) {
            local += c;
            atomicAdd(&counts[v], c);
        }
    }

    
    local += __shfl_down_sync(0xffffffff, local, 16);
    local += __shfl_down_sync(0xffffffff, local, 8);
    local += __shfl_down_sync(0xffffffff, local, 4);
    local += __shfl_down_sync(0xffffffff, local, 2);
    local += __shfl_down_sync(0xffffffff, local, 1);

    if (lane == 0 && local) atomicAdd(&counts[u], local);
}


__global__ void tc_thread_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ counts,
    int start, int end)
{
    int u = start + (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (u >= end) return;

    int us = __ldg(&offsets[u]);
    int ue = __ldg(&offsets[u + 1]);
    int ud = ue - us;
    const int* __restrict__ u_nbrs = indices + us;

    int e0 = d_lower_bound(u_nbrs, ud, u + 1);
    int total = 0;
    for (int e = e0; e < ud; ++e) {
        int v = __ldg(&u_nbrs[e]);
        int vs = __ldg(&offsets[v]);
        int vd = __ldg(&offsets[v + 1]) - vs;
        int c = scalar_intersect_count(u_nbrs, ud, indices + vs, vd, u, v);
        if (c) {
            total += c;
            atomicAdd(&counts[v], c);
        }
    }
    if (total) atomicAdd(&counts[u], total);
}



__global__ void tc_subset_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ counts,
    const int* __restrict__ verts,
    int n)
{
    int oi = (int)blockIdx.x;
    if (oi >= n) return;

    int tid = (int)threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = (int)blockDim.x >> 5;

    int u = verts[oi];
    int us = __ldg(&offsets[u]);
    int ue = __ldg(&offsets[u + 1]);
    int ud = ue - us;
    const int* __restrict__ u_nbrs = indices + us;

    
    int use_hash = (ud >= TC_HASH_MIN_DEG) && (ud <= TC_HASH_MAX_DEG);

    __shared__ int table[TC_HASH_SIZE];
    __shared__ int warp_sums[16];

    if (use_hash) {
        
        for (int i = tid; i < TC_HASH_SIZE; i += (int)blockDim.x) {
            table[i] = (int)TC_HASH_EMPTY;
        }
        __syncthreads();

        
        for (int i = tid; i < ud; i += (int)blockDim.x) {
            uint32_t key = (uint32_t)__ldg(&u_nbrs[i]);
            tc_hash_insert(table, key);
        }
        __syncthreads();

        int local = 0;
        for (int e = warp_id; e < ud; e += num_warps) {
            int v = __ldg(&u_nbrs[e]);
            if (v == u) continue;
            int vs = __ldg(&offsets[v]);
            int ve = __ldg(&offsets[v + 1]);

            int c = 0;
            for (int p = vs + lane; p < ve; p += 32) {
                uint32_t w = (uint32_t)__ldg(&indices[p]);
                if ((int)w == u || (int)w == v) continue;
                c += (int)tc_hash_find(table, w);
            }
            
            c += __shfl_down_sync(0xffffffff, c, 16);
            c += __shfl_down_sync(0xffffffff, c, 8);
            c += __shfl_down_sync(0xffffffff, c, 4);
            c += __shfl_down_sync(0xffffffff, c, 2);
            c += __shfl_down_sync(0xffffffff, c, 1);
            if (lane == 0) local += c;
        }

        if (lane == 0) warp_sums[warp_id] = local;
        __syncthreads();
        if (tid == 0) {
            int s = 0;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                if (w < num_warps) s += warp_sums[w];
            }
            for (int w = 8; w < num_warps; ++w) s += warp_sums[w];
            counts[oi] = s >> 1;
        }
        return;
    }

    
    int local = 0;
    for (int e = warp_id; e < ud; e += num_warps) {
        int v = __ldg(&u_nbrs[e]);
        if (v == u) continue;
        int vs = __ldg(&offsets[v]);
        int vd = __ldg(&offsets[v + 1]) - vs;
        int c = warp_intersect_count(u_nbrs, ud, indices + vs, vd, u, v);
        if (lane == 0) local += c;
    }

    if (lane == 0) warp_sums[warp_id] = local;
    __syncthreads();
    if (tid == 0) {
        int s = 0;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (w < num_warps) s += warp_sums[w];
        }
        for (int w = 8; w < num_warps; ++w) s += warp_sums[w];
        counts[oi] = s >> 1;
    }
}


__global__ void tc_div2_kernel(int* __restrict__ cnt, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) cnt[i] >>= 1;
}

void launch_tc_block(const int* off, const int* ind, int* cnt,
                     int s, int e, cudaStream_t st) {
    int n = e - s;
    if (n <= 0) return;
    int grid = n * TC_BLOCKS_PER_VERTEX;
    tc_block_kernel<<<grid, 256, 0, st>>>(off, ind, cnt, s, e);
}

void launch_tc_warp(const int* off, const int* ind, int* cnt,
                    int s, int e, cudaStream_t st) {
    int n = e - s;
    if (n <= 0) return;
    int tpb = 256;
    int wpb = tpb / 32;
    int blks = (n + wpb - 1) / wpb;
    tc_warp_kernel<<<blks, tpb, 0, st>>>(off, ind, cnt, s, e);
}

void launch_tc_thread(const int* off, const int* ind, int* cnt,
                      int s, int e, cudaStream_t st) {
    int n = e - s;
    if (n <= 0) return;
    int tpb = 256;
    int blks = (n + tpb - 1) / tpb;
    tc_thread_kernel<<<blks, tpb, 0, st>>>(off, ind, cnt, s, e);
}

void launch_tc_subset(const int* off, const int* ind, int* cnt,
                      const int* v, int n, cudaStream_t st) {
    if (n <= 0) return;
    tc_subset_kernel<<<n, 256, 0, st>>>(off, ind, cnt, v, n);
}

void launch_tc_div2(int* cnt, int n, cudaStream_t st) {
    if (n <= 0) return;
    int tpb = 256;
    int blks = (n + tpb - 1) / tpb;
    tc_div2_kernel<<<blks, tpb, 0, st>>>(cnt, n);
}

}  

void triangle_count_seg(const graph32_t& graph,
                        int32_t* counts,
                        const int32_t* vertices,
                        std::size_t n_vertices) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int* d_off = graph.offsets;
    const int* d_ind = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;

    const auto& seg = graph.segment_offsets.value();
    int seg0 = seg[0];
    int seg1 = seg[1];
    int seg2 = seg[2];
    int seg3 = seg[3];

    cudaStream_t stream = 0;

    if (vertices != nullptr) {
        int n_verts = static_cast<int>(n_vertices);
        launch_tc_subset(d_off, d_ind, counts, vertices, n_verts, stream);
    } else {
        
        cudaMemsetAsync(counts, 0, (size_t)num_vertices * sizeof(int), stream);

        
        launch_tc_block(d_off, d_ind, counts, seg0, seg1, stream);

        
        launch_tc_warp(d_off, d_ind, counts, seg1, seg2, stream);

        
        launch_tc_thread(d_off, d_ind, counts, seg2, seg3, stream);

        
        launch_tc_div2(counts, num_vertices, stream);
    }
}

}  
