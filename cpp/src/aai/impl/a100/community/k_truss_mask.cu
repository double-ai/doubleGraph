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

struct Cache : Cacheable {
    uint8_t* active = nullptr;
    int32_t active_cap = 0;

    int32_t* support = nullptr;
    int32_t support_cap = 0;

    int32_t* esrc = nullptr;
    int32_t esrc_cap = 0;

    int32_t* rev = nullptr;
    int32_t rev_cap = 0;

    int32_t* changed = nullptr;

    int32_t* osrc = nullptr;
    int32_t osrc_cap = 0;

    int32_t* odst = nullptr;
    int32_t odst_cap = 0;

    int32_t* ocnt = nullptr;

    Cache() {
        cudaMalloc(&changed, sizeof(int32_t));
        cudaMalloc(&ocnt, sizeof(int32_t));
    }

    void ensure(int32_t ne) {
        if (active_cap < ne) {
            if (active) cudaFree(active);
            cudaMalloc(&active, ne * sizeof(uint8_t));
            active_cap = ne;
        }
        if (support_cap < ne) {
            if (support) cudaFree(support);
            cudaMalloc(&support, ne * sizeof(int32_t));
            support_cap = ne;
        }
        if (esrc_cap < ne) {
            if (esrc) cudaFree(esrc);
            cudaMalloc(&esrc, ne * sizeof(int32_t));
            esrc_cap = ne;
        }
        if (rev_cap < ne) {
            if (rev) cudaFree(rev);
            cudaMalloc(&rev, ne * sizeof(int32_t));
            rev_cap = ne;
        }
        if (osrc_cap < ne) {
            if (osrc) cudaFree(osrc);
            cudaMalloc(&osrc, ne * sizeof(int32_t));
            osrc_cap = ne;
        }
        if (odst_cap < ne) {
            if (odst) cudaFree(odst);
            cudaMalloc(&odst, ne * sizeof(int32_t));
            odst_cap = ne;
        }
    }

    ~Cache() override {
        if (active) cudaFree(active);
        if (support) cudaFree(support);
        if (esrc) cudaFree(esrc);
        if (rev) cudaFree(rev);
        if (changed) cudaFree(changed);
        if (osrc) cudaFree(osrc);
        if (odst) cudaFree(odst);
        if (ocnt) cudaFree(ocnt);
    }
};





__device__ __forceinline__ int32_t bs_vertex(
    const int32_t* __restrict__ offsets, int32_t nv, int32_t p)
{
    int32_t lo = 0, hi = nv - 1;
    while (lo < hi) {
        int32_t mid = lo + (hi - lo + 1) / 2;
        if (__ldg(offsets + mid) <= p) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

__device__ __forceinline__ int32_t bs_edge(
    const int32_t* __restrict__ indices, int32_t start, int32_t end, int32_t target)
{
    int32_t lo = start, hi = end;
    while (lo < hi) {
        int32_t mid = lo + (hi - lo) / 2;
        int32_t val = __ldg(indices + mid);
        if (val < target) lo = mid + 1;
        else if (val > target) hi = mid;
        else return mid;
    }
    return -1;
}




__global__ void expand_mask_kernel(
    const uint32_t* __restrict__ mask, uint8_t* __restrict__ active, int32_t ne)
{
    for (int32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < ne;
         i += gridDim.x * blockDim.x) {
        active[i] = (mask[i >> 5] >> (i & 31)) & 1;
    }
}




__global__ void precompute_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    int32_t* __restrict__ edge_src, int32_t* __restrict__ rev_idx,
    int32_t nv, int32_t ne)
{
    for (int32_t p = blockIdx.x * blockDim.x + threadIdx.x; p < ne;
         p += gridDim.x * blockDim.x) {
        int32_t u = bs_vertex(offsets, nv, p);
        int32_t v = __ldg(indices + p);
        edge_src[p] = u;
        int32_t vs = __ldg(offsets + v), ve = __ldg(offsets + v + 1);
        rev_idx[p] = bs_edge(indices, vs, ve, u);
    }
}




__global__ void count_support_warp_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ edge_src, const int32_t* __restrict__ rev_idx,
    const uint8_t* __restrict__ active, int32_t* __restrict__ support,
    int32_t ne)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int total_warps = (gridDim.x * blockDim.x) >> 5;

    for (int32_t p = warp_id; p < ne; p += total_warps) {
        if (!active[p]) continue;

        int32_t u = edge_src[p];
        int32_t v = indices[p];
        if (u >= v) continue;

        int32_t u_start = offsets[u], u_end = offsets[u + 1];
        int32_t v_start = offsets[v], v_end = offsets[v + 1];
        int32_t u_len = u_end - u_start;
        int32_t v_len = v_end - v_start;

        int32_t count = 0;

        if (u_len > 0 && v_len > 0) {
            const int32_t* shorter, *longer;
            const uint8_t* s_act, *l_act;
            int32_t s_len, l_len;

            if (u_len <= v_len) {
                shorter = indices + u_start; longer = indices + v_start;
                s_act = active + u_start; l_act = active + v_start;
                s_len = u_len; l_len = v_len;
            } else {
                shorter = indices + v_start; longer = indices + u_start;
                s_act = active + v_start; l_act = active + u_start;
                s_len = v_len; l_len = u_len;
            }

            int32_t j = 0;
            for (int32_t i = lane; i < s_len; i += 32) {
                int32_t val = shorter[i];

                int32_t lo = j, hi = l_len;
                while (lo < hi) {
                    int32_t mid = lo + (hi - lo) / 2;
                    if (longer[mid] < val) lo = mid + 1;
                    else hi = mid;
                }
                if (lo < l_len && longer[lo] == val) {
                    if (s_act[i] && l_act[lo]) count++;
                    j = lo + 1;
                } else {
                    j = lo;
                }
            }

            for (int offset = 16; offset > 0; offset >>= 1)
                count += __shfl_down_sync(0xffffffff, count, offset);
        }

        if (lane == 0) {
            support[p] = count;
            int32_t rp = rev_idx[p];
            if (rp >= 0) support[rp] = count;
        }
    }
}




__global__ void count_support_thread_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ edge_src, const int32_t* __restrict__ rev_idx,
    const uint8_t* __restrict__ active, int32_t* __restrict__ support,
    int32_t ne)
{
    for (int32_t p = blockIdx.x * blockDim.x + threadIdx.x; p < ne;
         p += gridDim.x * blockDim.x) {
        if (!active[p]) continue;

        int32_t u = edge_src[p];
        int32_t v = indices[p];
        if (u >= v) continue;

        int32_t u_start = offsets[u], u_end = offsets[u + 1];
        int32_t v_start = offsets[v], v_end = offsets[v + 1];
        int32_t u_len = u_end - u_start;
        int32_t v_len = v_end - v_start;

        int32_t count = 0;

        if (u_len > 0 && v_len > 0) {
            const int32_t* A, *B;
            const uint8_t* actA, *actB;
            int32_t la, lb;

            if (u_len <= v_len) {
                A = indices + u_start; B = indices + v_start;
                actA = active + u_start; actB = active + v_start;
                la = u_len; lb = v_len;
            } else {
                A = indices + v_start; B = indices + u_start;
                actA = active + v_start; actB = active + u_start;
                la = v_len; lb = u_len;
            }

            if (lb > la * 8 && la > 0) {
                int32_t j = 0;
                for (int32_t i = 0; i < la; i++) {
                    if (!actA[i]) continue;
                    int32_t val = A[i];
                    int32_t lo = j, hi = lb;
                    while (lo < hi) {
                        int32_t mid = lo + (hi - lo) / 2;
                        if (B[mid] < val) lo = mid + 1;
                        else hi = mid;
                    }
                    if (lo < lb && B[lo] == val) {
                        if (actB[lo]) count++;
                        j = lo + 1;
                    } else {
                        j = lo;
                    }
                }
            } else {
                int32_t ia = 0, ib = 0;
                if (la > 0 && lb > 0) {
                    if (A[0] < B[0]) {
                        int32_t lo2 = 0, hi2 = la;
                        while (lo2 < hi2) { int32_t mid = lo2 + (hi2 - lo2) / 2; if (A[mid] < B[0]) lo2 = mid + 1; else hi2 = mid; }
                        ia = lo2;
                    } else if (B[0] < A[0]) {
                        int32_t lo2 = 0, hi2 = lb;
                        while (lo2 < hi2) { int32_t mid = lo2 + (hi2 - lo2) / 2; if (B[mid] < A[0]) lo2 = mid + 1; else hi2 = mid; }
                        ib = lo2;
                    }
                }
                while (ia < la && ib < lb) {
                    int32_t ai = A[ia], bj = B[ib];
                    if (ai < bj) { ia++; }
                    else if (ai > bj) { ib++; }
                    else {
                        if (actA[ia] && actB[ib]) count++;
                        ia++; ib++;
                    }
                }
            }
        }

        support[p] = count;
        int32_t rp = rev_idx[p];
        if (rp >= 0) support[rp] = count;
    }
}




__global__ void remove_weak_kernel(
    const int32_t* __restrict__ support, uint8_t* __restrict__ active,
    int32_t* __restrict__ changed, int32_t ne, int32_t threshold)
{
    for (int32_t p = blockIdx.x * blockDim.x + threadIdx.x; p < ne;
         p += gridDim.x * blockDim.x) {
        if (active[p] && support[p] < threshold) {
            active[p] = 0;
            *changed = 1;
        }
    }
}




__global__ void compact_kernel(
    const int32_t* __restrict__ edge_src, const int32_t* __restrict__ indices,
    const uint8_t* __restrict__ active,
    int32_t* __restrict__ out_src, int32_t* __restrict__ out_dst,
    int32_t* __restrict__ out_count, int32_t ne)
{
    for (int32_t p = blockIdx.x * blockDim.x + threadIdx.x; p < ne;
         p += gridDim.x * blockDim.x) {
        if (active[p]) {
            int32_t idx = atomicAdd(out_count, 1);
            out_src[idx] = edge_src[p];
            out_dst[idx] = indices[p];
        }
    }
}

}  

k_truss_result_t k_truss_mask(const graph32_t& graph, int32_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    const uint32_t* d_mask = graph.edge_mask;
    int32_t threshold = k - 2;
    cudaStream_t stream = 0;

    
    if (ne == 0) {
        return k_truss_result_t{nullptr, nullptr, 0};
    }

    cache.ensure(ne);

    
    {
        int T = 256, B = (ne + T - 1) / T;
        expand_mask_kernel<<<B, T, 0, stream>>>(d_mask, cache.active, ne);
    }

    
    {
        int T = 256, B = (ne + T - 1) / T;
        precompute_kernel<<<B, T, 0, stream>>>(d_off, d_idx, cache.esrc, cache.rev, nv, ne);
    }

    
    bool use_warp = (ne / nv) > 6;

    
    int32_t h_changed = 1;
    while (h_changed) {
        cudaMemsetAsync(cache.support, 0, ne * sizeof(int32_t), stream);

        if (use_warp) {
            int T = 256;
            int B = (ne + 7) / 8;
            if (B > 65535 * 2) B = 65535 * 2;
            count_support_warp_kernel<<<B, T, 0, stream>>>(
                d_off, d_idx, cache.esrc, cache.rev,
                cache.active, cache.support, ne);
        } else {
            int T = 256, B = (ne + T - 1) / T;
            count_support_thread_kernel<<<B, T, 0, stream>>>(
                d_off, d_idx, cache.esrc, cache.rev,
                cache.active, cache.support, ne);
        }

        cudaMemsetAsync(cache.changed, 0, sizeof(int32_t), stream);

        {
            int T = 256, B = (ne + T - 1) / T;
            remove_weak_kernel<<<B, T, 0, stream>>>(
                cache.support, cache.active, cache.changed,
                ne, threshold);
        }

        cudaMemcpyAsync(&h_changed, cache.changed, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    
    cudaMemsetAsync(cache.ocnt, 0, sizeof(int32_t), stream);
    {
        int T = 256, B = (ne + T - 1) / T;
        compact_kernel<<<B, T, 0, stream>>>(
            cache.esrc, d_idx, cache.active,
            cache.osrc, cache.odst, cache.ocnt, ne);
    }

    int32_t h_count;
    cudaMemcpy(&h_count, cache.ocnt, sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    int32_t* out_srcs = nullptr;
    int32_t* out_dsts = nullptr;
    if (h_count > 0) {
        cudaMalloc(&out_srcs, h_count * sizeof(int32_t));
        cudaMalloc(&out_dsts, h_count * sizeof(int32_t));
        cudaMemcpy(out_srcs, cache.osrc,
                   h_count * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(out_dsts, cache.odst,
                   h_count * sizeof(int32_t), cudaMemcpyDeviceToDevice);
    }

    return k_truss_result_t{out_srcs, out_dsts, static_cast<std::size_t>(h_count)};
}

}  
