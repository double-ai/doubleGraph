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




__device__ __forceinline__ int merge_intersect(
    const int* __restrict__ a, int size_a,
    const int* __restrict__ b, int size_b,
    int skip1, int skip2
) {
    int count = 0;
    int i = 0, j = 0;
    while (i < size_a && j < size_b) {
        int va = __ldg(a + i), vb = __ldg(b + j);
        if (va == vb) {
            if (va != skip1 && va != skip2) count++;
            i++; j++;
        } else if (va < vb) {
            i++;
        } else {
            j++;
        }
    }
    return count;
}

__device__ int bs_intersect(
    const int* __restrict__ small_arr, int small_size,
    const int* __restrict__ large_arr, int large_size,
    int skip1, int skip2
) {
    int count = 0;
    int lb = 0;
    for (int i = 0; i < small_size && lb < large_size; i++) {
        int target = __ldg(small_arr + i);
        if (target == skip1 || target == skip2) continue;
        int lo = lb, hi = large_size;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (__ldg(large_arr + mid) < target) lo = mid + 1;
            else hi = mid;
        }
        if (lo < large_size && __ldg(large_arr + lo) == target) {
            count++;
            lb = lo + 1;
        } else {
            lb = lo;
        }
    }
    return count;
}

__device__ __forceinline__ int intersect_count(
    const int* __restrict__ a, int size_a,
    const int* __restrict__ b, int size_b,
    int skip1, int skip2
) {
    if (size_a == 0 || size_b == 0) return 0;
    if (size_a > size_b) {
        const int* t = a; a = b; b = t;
        int ts = size_a; size_a = size_b; size_b = ts;
    }
    
    int a0 = __ldg(a);
    int a1 = __ldg(a + (size_a - 1));
    int b0 = __ldg(b);
    int b1 = __ldg(b + (size_b - 1));
    if (a1 < b0 || a0 > b1) return 0;

    if (size_b < 2 * size_a)
        return merge_intersect(a, size_a, b, size_b, skip1, skip2);
    return bs_intersect(a, size_a, b, size_b, skip1, skip2);
}




__global__ void tc_thread_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ counts,
    int v_start, int v_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= v_count) return;

    int v = v_start + tid;
    int vb = __ldg(offsets + v), ve = __ldg(offsets + v + 1);
    int vd = ve - vb;

    int local = 0;
    for (int e = vb; e < ve; e++) {
        int u = __ldg(indices + e);
        if (u <= v) continue;  
        int ub = __ldg(offsets + u), ue = __ldg(offsets + u + 1);
        int ud = ue - ub;
        int c = intersect_count(indices + vb, vd, indices + ub, ud, v, u);
        if (c) {
            local += c;
            atomicAdd(&counts[u], c);
        }
    }
    if (local) atomicAdd(&counts[v], local);
}




__global__ void tc_warp_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ counts,
    int v_start, int v_count
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= v_count) return;

    int v = v_start + warp_id;
    int vb = __ldg(offsets + v), ve = __ldg(offsets + v + 1);
    int vd = ve - vb;

    int local = 0;
    for (int e = vb + lane; e < ve; e += 32) {
        int u = __ldg(indices + e);
        if (u <= v) continue;
        int ub = __ldg(offsets + u), ue = __ldg(offsets + u + 1);
        int ud = ue - ub;
        int c = intersect_count(indices + vb, vd, indices + ub, ud, v, u);
        if (c) {
            local += c;
            atomicAdd(&counts[u], c);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local += __shfl_down_sync(0xFFFFFFFF, local, off);
    }

    if (lane == 0 && local) atomicAdd(&counts[v], local);
}




#define BLOCK_SIZE_BM 256

__global__ void tc_block_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ counts,
    int v_start, int v_count
) {
    if (blockIdx.x >= v_count) return;

    int v = v_start + blockIdx.x;
    int vb = __ldg(offsets + v), ve = __ldg(offsets + v + 1);
    int vd = ve - vb;

    int local = 0;
    for (int e = vb + threadIdx.x; e < ve; e += BLOCK_SIZE_BM) {
        int u = __ldg(indices + e);
        if (u <= v) continue;
        int ub = __ldg(offsets + u), ue = __ldg(offsets + u + 1);
        int ud = ue - ub;
        int c = intersect_count(indices + vb, vd, indices + ub, ud, v, u);
        if (c) {
            local += c;
            atomicAdd(&counts[u], c);
        }
    }

    
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local += __shfl_down_sync(0xFFFFFFFF, local, off);
    }
    __shared__ int warp_sums[BLOCK_SIZE_BM / 32];
    if (lane == 0) warp_sums[warp_id] = local;
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        #pragma unroll
        for (int w = 0; w < BLOCK_SIZE_BM / 32; w++) total += warp_sums[w];
        if (total) atomicAdd(&counts[v], total);
    }
}




__global__ void tc_subset_bitmap_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ counts,
    const int* __restrict__ vertex_list,
    int n_vertices,
    int num_vertices
) {
    if (blockIdx.x >= n_vertices) return;

    extern __shared__ uint32_t s_bitmap_sub[];

    int bm_words = (num_vertices + 31) >> 5;

    for (int i = threadIdx.x; i < bm_words; i += BLOCK_SIZE_BM)
        s_bitmap_sub[i] = 0;
    __syncthreads();

    int v = vertex_list[blockIdx.x];
    int vb = __ldg(offsets + v), ve = __ldg(offsets + v + 1);
    int vd = ve - vb;

    for (int i = threadIdx.x; i < vd; i += BLOCK_SIZE_BM) {
        int nbr = indices[vb + i];
        atomicOr(&s_bitmap_sub[nbr >> 5], 1u << (nbr & 31));
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAnd(&s_bitmap_sub[v >> 5], ~(1u << (v & 31)));
    }
    __syncthreads();

    int thread_count = 0;
    for (int e = vb + threadIdx.x; e < ve; e += BLOCK_SIZE_BM) {
        int u = __ldg(indices + e);
        if (u == v) continue;
        int ub = __ldg(offsets + u), ue = __ldg(offsets + u + 1);
        int ud = ue - ub;

        for (int i = 0; i < ud; i++) {
            int w = indices[ub + i];
            if (w == u) continue;
            if (s_bitmap_sub[w >> 5] & (1u << (w & 31)))
                thread_count++;
        }
    }

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        thread_count += __shfl_down_sync(0xFFFFFFFF, thread_count, off);

    __shared__ int warp_sums_sub[BLOCK_SIZE_BM / 32];
    if (lane == 0) warp_sums_sub[warp_id] = thread_count;
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < BLOCK_SIZE_BM / 32; w++) total += warp_sums_sub[w];
        counts[blockIdx.x] = total >> 1;
    }
}

__global__ void tc_subset_fallback_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ counts,
    const int* __restrict__ vertex_list,
    int n_vertices
) {
    if (blockIdx.x >= n_vertices) return;

    int v = vertex_list[blockIdx.x];
    int vb = __ldg(offsets + v), ve = __ldg(offsets + v + 1);
    int vd = ve - vb;

    int thread_total = 0;
    for (int e = vb + threadIdx.x; e < ve; e += BLOCK_SIZE_BM) {
        int u = __ldg(indices + e);
        if (u == v) continue;
        int ub = __ldg(offsets + u), ue = __ldg(offsets + u + 1);
        int ud = ue - ub;
        thread_total += intersect_count(indices + vb, vd, indices + ub, ud, v, u);
    }

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        thread_total += __shfl_down_sync(0xFFFFFFFF, thread_total, off);

    __shared__ int warp_sums[BLOCK_SIZE_BM / 32];
    if (lane == 0) warp_sums[warp_id] = thread_total;
    __syncthreads();

    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < BLOCK_SIZE_BM / 32; w++) total += warp_sums[w];
        counts[blockIdx.x] = total >> 1;
    }
}

__global__ void zero_counts_kernel(int* counts, int start, int count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < count) counts[start + tid] = 0;
}

__global__ void divide_by_2_kernel(int* cnt, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) cnt[i] >>= 1;
}

}  

void triangle_count_seg(const graph32_t& graph,
                        int32_t* counts,
                        const int32_t* vertices,
                        std::size_t n_vertices) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices_g = graph.number_of_vertices;

    if (vertices != nullptr) {
        int nv = static_cast<int>(n_vertices);
        if (nv <= 0) return;

        int bm_words = (num_vertices_g + 31) / 32;
        size_t smem = bm_words * sizeof(uint32_t);

        if (smem <= 99 * 1024) {
            cudaFuncSetAttribute(tc_subset_bitmap_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            tc_subset_bitmap_kernel<<<nv, BLOCK_SIZE_BM, smem>>>(
                d_offsets, d_indices, counts, vertices, nv, num_vertices_g);
        } else {
            tc_subset_fallback_kernel<<<nv, BLOCK_SIZE_BM>>>(
                d_offsets, d_indices, counts, vertices, nv);
        }
    } else {
        const auto& seg = graph.segment_offsets.value();
        int seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg3 = seg[3];

        
        {
            int count = num_vertices_g;
            if (count > 0) {
                int block = 256;
                int grid = (count + block - 1) / block;
                zero_counts_kernel<<<grid, block>>>(counts, 0, count);
            }
        }

        
        {
            int v_count = seg1 - seg0;
            if (v_count > 0) {
                tc_block_kernel<<<v_count, BLOCK_SIZE_BM>>>(
                    d_offsets, d_indices, counts, seg0, v_count);
            }
        }

        
        {
            int v_count = seg2 - seg1;
            if (v_count > 0) {
                int block = 256;
                int warps_per_block = block / 32;
                int grid = (v_count + warps_per_block - 1) / warps_per_block;
                tc_warp_kernel<<<grid, block>>>(
                    d_offsets, d_indices, counts, seg1, v_count);
            }
        }

        
        {
            int v_count = seg3 - seg2;
            if (v_count > 0) {
                int block = 256;
                int grid = (v_count + block - 1) / block;
                tc_thread_kernel<<<grid, block>>>(
                    d_offsets, d_indices, counts, seg2, v_count);
            }
        }

        
        if (num_vertices_g > 0) {
            divide_by_2_kernel<<<(num_vertices_g + 255) / 256, 256>>>(counts, num_vertices_g);
        }
    }
}

}  
