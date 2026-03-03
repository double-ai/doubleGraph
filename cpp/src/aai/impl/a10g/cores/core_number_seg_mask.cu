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
#include <cooperative_groups.h>
#include <cstdint>

namespace aai {

namespace {

namespace cg = cooperative_groups;

#define CNT_REMAINING  0
#define CNT_FRONTIER   1
#define CNT_NEW_REM    2
#define CNT_MIN_VAL    3
#define CNT_AFFECTED   4
#define CNT_NEXT_FRONT 5

__global__ void __launch_bounds__(256, 2) core_number_persistent(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t num_vertices,
    const int32_t multiplier,
    const int32_t delta,
    const int32_t k_first,
    const int64_t k_last_i64,
    int32_t* __restrict__ core_numbers,
    int32_t* __restrict__ removed,
    int32_t* __restrict__ decrements,
    int32_t* __restrict__ buf0,  
    int32_t* __restrict__ buf1,  
    int32_t* __restrict__ buf2,  
    int32_t* __restrict__ buf3,  
    int32_t* __restrict__ buf4,  
    int32_t* __restrict__ counters)
{
    cg::grid_group grid = cg::this_grid();
    const int tid = grid.thread_rank();
    const int nthreads = grid.size();
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int nwarps = nthreads >> 5;
    const size_t k_last = static_cast<size_t>(static_cast<uint64_t>(k_last_i64));

    int32_t* remaining = buf0;
    int32_t* remaining_new = buf1;
    int32_t* frontier = buf2;
    int32_t* next_frontier = buf3;
    int32_t* affected = buf4;

    
    
    
    for (int v = tid; v < num_vertices; v += nthreads) {
        int count = 0;
        int s = __ldg(&offsets[v]);
        int e = __ldg(&offsets[v + 1]);

        
        for (int j = s; j < e; j++) {
            
            uint32_t mask_word;
            asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(mask_word) : "l"(&edge_mask[j >> 5]));

            bool edge_active = (mask_word >> (j & 31)) & 1;
            int u = __ldg(&indices[j]);

            if (edge_active && u != v) {
                count++;
            }
        }
        core_numbers[v] = count * multiplier;
        removed[v] = 0;
        decrements[v] = 0;
    }
    if (tid == 0) counters[CNT_REMAINING] = 0;
    grid.sync();

    
    
    
    for (int v = tid; v < num_vertices; v += nthreads) {
        int cn = core_numbers[v];
        if (cn > 0) {
            if (k_first > 1 && cn < k_first) core_numbers[v] = 0;
            int pos = atomicAdd(&counters[CNT_REMAINING], 1);
            remaining[pos] = v;
        }
    }
    grid.sync();

    int rem_count = counters[CNT_REMAINING];
    if (rem_count == 0) return;

    size_t k = (size_t)((k_first < 2) ? 2 : k_first);
    if (delta == 2 && (k & 1)) k++;

    
    
    
    while (k <= k_last && rem_count > 0) {
        const int ik = (int)k;

        
        if (tid == 0) {
            counters[CNT_FRONTIER] = 0;
            counters[CNT_NEW_REM] = 0;
            counters[CNT_MIN_VAL] = 0x7FFFFFFF;
        }
        grid.sync();

        
        
        int local_min = 0x7FFFFFFF;
        for (int i = tid; i < rem_count; i += nthreads) {
            int v = remaining[i];
            if (!removed[v]) {
                int cn = core_numbers[v];
                if (cn < ik) {
                    int pos = atomicAdd(&counters[CNT_FRONTIER], 1);
                    frontier[pos] = v;
                } else {
                    int pos = atomicAdd(&counters[CNT_NEW_REM], 1);
                    remaining_new[pos] = v;
                    if (cn < local_min) local_min = cn;
                }
            }
        }

        
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            int o = __shfl_down_sync(0xFFFFFFFF, local_min, off);
            local_min = (o < local_min) ? o : local_min;
        }
        if (lane == 0 && local_min < 0x7FFFFFFF)
            atomicMin(&counters[CNT_MIN_VAL], local_min);
        grid.sync();

        int front_count = counters[CNT_FRONTIER];
        rem_count = counters[CNT_NEW_REM];
        { int32_t* t = remaining; remaining = remaining_new; remaining_new = t; }

        if (front_count > 0) {
            
            while (front_count > 0) {
                
                for (int i = tid; i < front_count; i += nthreads)
                    removed[frontier[i]] = 1;
                if (tid == 0) {
                    counters[CNT_AFFECTED] = 0;
                    counters[CNT_NEXT_FRONT] = 0;
                }
                grid.sync();

                
                for (int wi = warp_id; wi < front_count; wi += nwarps) {
                    int v = frontier[wi];
                    int s = __ldg(&offsets[v]);
                    int e = __ldg(&offsets[v + 1]);
                    int niters = (e - s + 31) >> 5;

                    
                    for (int it = 0; it < niters; it++) {
                        int j = s + it * 32 + lane;
                        bool push = false;
                        int uv = 0;

                        if (j < e) {
                            
                            uint32_t mask_word;
                            asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(mask_word) : "l"(&edge_mask[j >> 5]));

                            bool edge_active = (mask_word >> (j & 31)) & 1;
                            int u = __ldg(&indices[j]);

                            if (edge_active && u != v && !removed[u]) {
                                int old = atomicAdd(&decrements[u], delta);
                                if (old == 0) {
                                    push = true;
                                    uv = u;
                                }
                            }
                        }

                        
                        unsigned ballot = __ballot_sync(0xFFFFFFFF, push);
                        if (ballot) {
                            int ws;
                            if (lane == 0)
                                ws = atomicAdd(&counters[CNT_AFFECTED], __popc(ballot));
                            ws = __shfl_sync(0xFFFFFFFF, ws, 0);
                            if (push)
                                affected[ws + __popc(ballot & ((1u << lane) - 1))] = uv;
                        }
                    }
                }
                grid.sync();

                
                int aff_count = counters[CNT_AFFECTED];
                if (aff_count == 0) break;
                int kmd = ik - delta;
                if (kmd < 0) kmd = 0;

                for (int i = tid; i < aff_count; i += nthreads) {
                    int v = affected[i];
                    int dec = decrements[v];
                    decrements[v] = 0;
                    int cn = core_numbers[v];

                    
                    int nc = (cn >= dec) ? (cn - dec) : 0;
                    if (nc < kmd) nc = kmd;
                    if (nc < k_first) nc = 0;

                    core_numbers[v] = nc;
                    if (nc < ik && !removed[v]) {
                        int pos = atomicAdd(&counters[CNT_NEXT_FRONT], 1);
                        next_frontier[pos] = v;
                    }
                }
                grid.sync();

                front_count = counters[CNT_NEXT_FRONT];
                { int32_t* t = frontier; frontier = next_frontier; next_frontier = t; }
            }
            k += delta;
        } else {
            
            int mn = counters[CNT_MIN_VAL];
            size_t nk = (size_t)mn + delta;
            k = ((k + delta) > nk) ? (k + delta) : nk;
            if (delta == 2 && (k & 1)) k++;
        }
    }
}

struct Cache : Cacheable {
    int32_t* removed = nullptr;
    int32_t* decrements = nullptr;
    int32_t* buf0 = nullptr;
    int32_t* buf1 = nullptr;
    int32_t* buf2 = nullptr;
    int32_t* buf3 = nullptr;
    int32_t* buf4 = nullptr;
    int32_t* counters = nullptr;
    int64_t vert_capacity = 0;
    bool counters_allocated = false;
    int max_blocks = 0;

    void ensure(int32_t num_vertices) {
        if (max_blocks == 0) {
            int bps;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &bps, core_number_persistent, 256, 0);
            int nsm;
            cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
            
            
            int safe_bps = (bps > 1) ? (bps - 1) : bps;
            max_blocks = safe_bps * nsm;
            if (max_blocks <= 0) max_blocks = nsm;
        }

        if (!counters_allocated) {
            cudaMalloc(&counters, 8 * sizeof(int32_t));
            counters_allocated = true;
        }

        int64_t n = num_vertices;
        if (vert_capacity < n) {
            if (removed) cudaFree(removed);
            if (decrements) cudaFree(decrements);
            if (buf0) cudaFree(buf0);
            if (buf1) cudaFree(buf1);
            if (buf2) cudaFree(buf2);
            if (buf3) cudaFree(buf3);
            if (buf4) cudaFree(buf4);

            cudaMalloc(&removed, n * sizeof(int32_t));
            cudaMalloc(&decrements, n * sizeof(int32_t));
            cudaMalloc(&buf0, n * sizeof(int32_t));
            cudaMalloc(&buf1, n * sizeof(int32_t));
            cudaMalloc(&buf2, n * sizeof(int32_t));
            cudaMalloc(&buf3, n * sizeof(int32_t));
            cudaMalloc(&buf4, n * sizeof(int32_t));

            vert_capacity = n;
        }
    }

    ~Cache() override {
        if (removed) cudaFree(removed);
        if (decrements) cudaFree(decrements);
        if (buf0) cudaFree(buf0);
        if (buf1) cudaFree(buf1);
        if (buf2) cudaFree(buf2);
        if (buf3) cudaFree(buf3);
        if (buf4) cudaFree(buf4);
        if (counters) cudaFree(counters);
    }
};

}  

void core_number_seg_mask(const graph32_t& graph,
                          int32_t* core_numbers,
                          int degree_type,
                          std::size_t k_first,
                          std::size_t k_last) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    if (num_vertices == 0) return;

    cache.ensure(num_vertices);

    int32_t mult = (degree_type == 2) ? 2 : 1;
    int32_t delta = (degree_type == 2) ? 2 : 1;
    int32_t k_first_i32 = static_cast<int32_t>(k_first);
    int64_t k_last_i64 = static_cast<int64_t>(k_last);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    const uint32_t* edge_mask = graph.edge_mask;

    void* args[] = {
        (void*)&offsets,
        (void*)&indices,
        (void*)&edge_mask,
        (void*)&num_vertices,
        (void*)&mult,
        (void*)&delta,
        (void*)&k_first_i32,
        (void*)&k_last_i64,
        (void*)&core_numbers,
        (void*)&cache.removed,
        (void*)&cache.decrements,
        (void*)&cache.buf0,
        (void*)&cache.buf1,
        (void*)&cache.buf2,
        (void*)&cache.buf3,
        (void*)&cache.buf4,
        (void*)&cache.counters
    };

    cudaLaunchCooperativeKernel(
        (void*)core_number_persistent,
        dim3(cache.max_blocks), dim3(256), args, 0, 0);

    cudaStreamSynchronize(0);
}

}  
