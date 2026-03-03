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




__device__ __forceinline__ int warp_isect(
    const int* __restrict__ iter_arr, int iter_sz,
    const int* __restrict__ search_arr, int search_sz,
    int lane, int u_id, int v_id
) {
    if (iter_sz == 0 || search_sz == 0) return 0;
    int iter_last = iter_arr[iter_sz - 1];
    int search_first = search_arr[0];
    if (iter_last < search_first) return 0;
    int iter_first = iter_arr[0];
    int search_last = search_arr[search_sz - 1];
    if (iter_first > search_last) return 0;

    int c = 0;
    for (int i = lane; i < iter_sz; i += 32) {
        int t = iter_arr[i];
        if (t == u_id | t == v_id) continue;
        if (t < search_first | t > search_last) continue;
        int lo = 0, hi = search_sz;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (search_arr[mid] < t) lo = mid + 1;
            else hi = mid;
        }
        if (lo < search_sz && search_arr[lo] == t) c++;
    }
    c += __shfl_down_sync(0xffffffff, c, 16);
    c += __shfl_down_sync(0xffffffff, c, 8);
    c += __shfl_down_sync(0xffffffff, c, 4);
    c += __shfl_down_sync(0xffffffff, c, 2);
    c += __shfl_down_sync(0xffffffff, c, 1);
    return c;
}




__global__ void tc_low_half_kernel(
    const int* __restrict__ off,
    const int* __restrict__ idx,
    int* __restrict__ cnt,
    int sv, int ev
) {
    int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int ln = threadIdx.x & 31;
    int u = sv + gw;
    if (u >= ev) return;

    int us = off[u], ue = off[u + 1], ud = ue - us;
    int tot = 0;

    for (int e = 0; e < ud; e++) {
        int v = idx[us + e];
        if (v <= u) continue;  

        int vs = off[v], ve = off[v + 1], vd = ve - vs;

        const int *ia, *sa;
        int isz, ssz;
        if (ud <= vd) {
            ia = idx + us; isz = ud; sa = idx + vs; ssz = vd;
        } else {
            ia = idx + vs; isz = vd; sa = idx + us; ssz = ud;
        }

        int c = warp_isect(ia, isz, sa, ssz, ln, u, v);
        if (ln == 0 && c > 0) {
            tot += c;
            atomicAdd(&cnt[v], c);  
        }
    }

    if (ln == 0 && tot > 0) atomicAdd(&cnt[u], tot);
}




__global__ void tc_block_half_kernel(
    const int* __restrict__ off,
    const int* __restrict__ idx,
    int* __restrict__ cnt,
    int sv, int ev,
    int smem_cap
) {
    extern __shared__ int sm[];
    int u = sv + blockIdx.x;
    if (u >= ev) return;

    int tid = threadIdx.x;
    int nw = blockDim.x >> 5;
    int wi = tid >> 5;
    int ln = tid & 31;

    int us = off[u], ue = off[u + 1], ud = ue - us;

    bool cached = (ud <= smem_cap);
    const int* nu;
    if (cached) {
        for (int i = tid; i < ud; i += blockDim.x) sm[i] = idx[us + i];
        __syncthreads();
        nu = sm;
    } else {
        nu = idx + us;
    }

    int wt = 0;

    for (int e = wi; e < ud; e += nw) {
        int v = nu[e];
        if (v <= u) continue;  

        int vs = off[v], ve = off[v + 1], vd = ve - vs;

        const int *ia, *sa;
        int isz, ssz;
        if (vd <= ud) {
            ia = idx + vs; isz = vd; sa = nu; ssz = ud;
        } else {
            ia = nu; isz = ud; sa = idx + vs; ssz = vd;
        }

        int c = warp_isect(ia, isz, sa, ssz, ln, u, v);
        if (ln == 0 && c > 0) {
            wt += c;
            atomicAdd(&cnt[v], c);  
        }
    }

    
    __syncthreads();
    if (ln == 0) sm[wi] = wt;
    __syncthreads();

    if (tid == 0) {
        int s = 0;
        for (int w = 0; w < nw; w++) s += sm[w];
        if (s > 0) atomicAdd(&cnt[u], s);
    }
}




__global__ void tc_subset_kernel(
    const int* __restrict__ off,
    const int* __restrict__ idx,
    int* __restrict__ cnt,
    const int* __restrict__ vlist,
    int nv,
    int smem_cap
) {
    extern __shared__ int sm[];
    int oi = blockIdx.x;
    if (oi >= nv) return;

    int tid = threadIdx.x;
    int nw = blockDim.x >> 5;
    int wi = tid >> 5;
    int ln = tid & 31;

    int u = vlist[oi];
    int us = off[u], ue = off[u + 1], ud = ue - us;

    bool cached = (ud <= smem_cap);
    const int* nu;
    if (cached) {
        for (int i = tid; i < ud; i += blockDim.x) sm[i] = idx[us + i];
        __syncthreads();
        nu = sm;
    } else {
        nu = idx + us;
    }

    int wt = 0;
    for (int e = wi; e < ud; e += nw) {
        int v = nu[e];
        if (v == u) continue;
        int vs = off[v], ve = off[v + 1], vd = ve - vs;

        const int *ia, *sa;
        int isz, ssz;
        if (vd <= ud) {
            ia = idx + vs; isz = vd; sa = nu; ssz = ud;
        } else {
            ia = nu; isz = ud; sa = idx + vs; ssz = vd;
        }

        int c = warp_isect(ia, isz, sa, ssz, ln, u, v);
        if (ln == 0) wt += c;
    }

    __syncthreads();
    if (ln == 0) sm[wi] = wt;
    __syncthreads();

    if (tid == 0) {
        int s = 0;
        for (int w = 0; w < nw; w++) s += sm[w];
        cnt[oi] = s >> 1;
    }
}




__global__ void divide_by_2_kernel(int* cnt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) cnt[i] >>= 1;
}





void launch_tc_low_half(const int* off, const int* idx, int* cnt, int sv, int ev) {
    int n = ev - sv;
    if (n <= 0) return;
    int tpb = 256;
    int wpb = tpb / 32;
    int blocks = (n + wpb - 1) / wpb;
    tc_low_half_kernel<<<blocks, tpb>>>(off, idx, cnt, sv, ev);
}

void launch_tc_block_half(const int* off, const int* idx, int* cnt,
                          int sv, int ev, int max_deg, int threads) {
    int n = ev - sv;
    if (n <= 0) return;
    int smem_bytes = max_deg * (int)sizeof(int);
    int smem_cap = max_deg;
    if (smem_bytes > 99 * 1024) {
        smem_bytes = 99 * 1024;
        smem_cap = 99 * 1024 / (int)sizeof(int);
    }
    int nw = threads / 32;
    if (smem_bytes < nw * (int)sizeof(int))
        smem_bytes = nw * (int)sizeof(int);
    cudaFuncSetAttribute(tc_block_half_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    tc_block_half_kernel<<<n, threads, smem_bytes>>>(off, idx, cnt, sv, ev, smem_cap);
}

void launch_tc_subset(const int* off, const int* idx, int* cnt,
                      const int* vlist, int nv, int threads) {
    if (nv <= 0) return;
    int smem_bytes = 99 * 1024;
    int smem_cap = smem_bytes / (int)sizeof(int);
    cudaFuncSetAttribute(tc_subset_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    tc_subset_kernel<<<nv, threads, smem_bytes>>>(off, idx, cnt, vlist, nv, smem_cap);
}

void launch_divide_by_2(int* cnt, int n) {
    if (n <= 0) return;
    divide_by_2_kernel<<<(n + 255) / 256, 256>>>(cnt, n);
}

}  

void triangle_count_seg(const graph32_t& graph,
                        int32_t* counts,
                        const int32_t* vertices,
                        std::size_t n_vertices) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;

    const auto& seg = graph.segment_offsets.value();
    int s0 = seg[0], s1 = seg[1], s2 = seg[2], s3 = seg[3];

    if (vertices != nullptr && n_vertices > 0) {
        
        launch_tc_subset(d_off, d_idx, counts,
                         vertices, (int)n_vertices, 512);
    } else {
        
        cudaMemsetAsync(counts, 0, num_vertices * sizeof(int32_t));

        
        if (s1 > s0) {
            launch_tc_block_half(d_off, d_idx, counts, s0, s1, 99 * 1024 / 4, 1024);
        }
        
        if (s2 > s1) {
            launch_tc_block_half(d_off, d_idx, counts, s1, s2, 1024, 512);
        }
        
        if (s3 > s2) {
            launch_tc_low_half(d_off, d_idx, counts, s2, s3);
        }
        

        
        launch_divide_by_2(counts, num_vertices);
    }
}

}  
