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
#include <cub/cub.cuh>

namespace aai {

namespace {

constexpr int MAX_PIVOTS = 32;
constexpr int MAX_ROUNDS = 8;
constexpr int MAX_BFS_ITERS = 128;
constexpr int TRIM_ITERS_PER_ROUND = 4;





__device__ __forceinline__ unsigned int lanemask_lt_u32() {
    unsigned int lane = threadIdx.x & 31;
    return (1u << lane) - 1u;
}

__device__ __forceinline__ void warp_push_int(
    unsigned int mask,
    bool has_output,
    int value,
    int32_t* __restrict__ out,
    int32_t* __restrict__ out_count)
{
    unsigned int ballot = __ballot_sync(mask, has_output);
    if (!ballot) return;
    int lane = threadIdx.x & 31;
    int leader = __ffs(ballot) - 1;
    int total = __popc(ballot);

    int base = 0;
    if (lane == leader) {
        base = atomicAdd(out_count, total);
    }
    base = __shfl_sync(ballot, base, leader);

    if (has_output) {
        int rank = __popc(ballot & lanemask_lt_u32());
        out[base + rank] = value;
    }
}





__global__ void expand_sources_k(const int32_t* __restrict__ offsets,
                                 int32_t nv,
                                 int32_t* __restrict__ sources) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= nv) return;
    int start = offsets[u];
    int end = offsets[u + 1];
    for (int e = start; e < end; e++) sources[e] = u;
}

__global__ void build_offsets_from_sorted_k(const int32_t* __restrict__ sorted_dst,
                                            int32_t ne,
                                            int32_t nv,
                                            int32_t* __restrict__ offsets) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i == 0) {
        if (ne == 0) {
            for (int v = 0; v <= nv; v++) offsets[v] = 0;
            return;
        }
        int d0 = sorted_dst[0];
        for (int v = 0; v <= d0; v++) offsets[v] = 0;
    }

    if (i > 0 && i < ne) {
        int cur = sorted_dst[i];
        int prev = sorted_dst[i - 1];
        if (cur != prev) {
            for (int v = prev + 1; v <= cur; v++) offsets[v] = i;
        }
    }

    if (i == ne - 1 && ne > 0) {
        int last = sorted_dst[ne - 1];
        for (int v = last + 1; v <= nv; v++) offsets[v] = ne;
    }
}





__global__ void select_pivots_k(const int32_t* __restrict__ components,
                                const int32_t* __restrict__ offsets,
                                int32_t nv,
                                int32_t max_pivots,
                                int32_t* __restrict__ pivots,
                                int32_t* __restrict__ pivot_count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int v = tid; v < nv; v += stride) {
        unsigned int mask = __activemask();
        bool cand = false;
        if (components[v] < 0) {
            cand = ((offsets[v + 1] - offsets[v]) > 0);
        }

        unsigned int ballot = __ballot_sync(mask, cand);
        if (!ballot) continue;

        int lane = threadIdx.x & 31;
        int leader = __ffs(ballot) - 1;
        int total = __popc(ballot);

        int base = 0;
        if (lane == leader) {
            base = atomicAdd(pivot_count, total);
        }
        base = __shfl_sync(ballot, base, leader);
        if (base >= max_pivots) return;

        if (cand) {
            int rank = __popc(ballot & lanemask_lt_u32());
            int slot = base + rank;
            if (slot < max_pivots) pivots[slot] = v;
        }
        if (base + total >= max_pivots) return;
    }
}

__global__ void init_frontier_from_pivots_k(const int32_t* __restrict__ pivots,
                                              int32_t pivot_count,
                                              int32_t* __restrict__ frontier) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pivot_count) frontier[i] = pivots[i];
}

__global__ void init_pivots_masks_k(const int32_t* __restrict__ pivots,
                                   int32_t pivot_count,
                                   uint32_t* __restrict__ fmask,
                                   uint32_t* __restrict__ bmask,
                                   uint32_t* __restrict__ fdelta,
                                   uint32_t* __restrict__ bdelta,
                                   int32_t* __restrict__ frontier) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pivot_count) return;
    int32_t v = pivots[i];
    uint32_t bit = 1u << i;
    fmask[v] = bit;
    bmask[v] = bit;
    fdelta[v] = bit;
    bdelta[v] = bit;
    frontier[i] = v;
}





__global__ void bfs_step_mask_fwd_k(const int32_t* __restrict__ offsets,
                                   const int32_t* __restrict__ indices,
                                   const int32_t* __restrict__ components,
                                   int32_t frontier_size,
                                   const int32_t* __restrict__ frontier,
                                   int32_t* __restrict__ next_frontier,
                                   int32_t* __restrict__ next_size,
                                   uint32_t* __restrict__ mask,
                                   uint32_t* __restrict__ delta) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warps_total = (gridDim.x * blockDim.x) >> 5;

    for (int fi = warp; fi < frontier_size; fi += warps_total) {
        int32_t u = frontier[fi];
        if (components[u] >= 0) continue;

        uint32_t d = 0;
        if (lane == 0) {
            d = atomicExch(reinterpret_cast<unsigned int*>(&delta[u]), 0u);
        }
        d = __shfl_sync(0xffffffffu, d, 0);
        if (!d) continue;

        int start = offsets[u];
        int end = offsets[u + 1];
        for (int base = start; base < end; base += 32) {
            int e = base + lane;
            bool in_range = (e < end);
            int32_t w = in_range ? indices[e] : 0;

            bool push = false;
            if (in_range && components[w] < 0) {
                uint32_t old = atomicOr(reinterpret_cast<unsigned int*>(&mask[w]), d);
                uint32_t newbits = d & ~old;
                if (newbits) {
                    uint32_t oldd = atomicOr(reinterpret_cast<unsigned int*>(&delta[w]), newbits);
                    push = (oldd == 0);
                }
            }
            warp_push_int(0xffffffffu, push, w, next_frontier, next_size);
        }
    }
}

__global__ void bfs_step_mask_bwd_k(const int32_t* __restrict__ rev_off,
                                   const int32_t* __restrict__ rev_idx,
                                   const int32_t* __restrict__ components,
                                   int32_t frontier_size,
                                   const int32_t* __restrict__ frontier,
                                   int32_t* __restrict__ next_frontier,
                                   int32_t* __restrict__ next_size,
                                   uint32_t* __restrict__ mask,
                                   uint32_t* __restrict__ delta) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int warps_total = (gridDim.x * blockDim.x) >> 5;

    for (int fi = warp; fi < frontier_size; fi += warps_total) {
        int32_t u = frontier[fi];
        if (components[u] >= 0) continue;

        uint32_t d = 0;
        if (lane == 0) {
            d = atomicExch(reinterpret_cast<unsigned int*>(&delta[u]), 0u);
        }
        d = __shfl_sync(0xffffffffu, d, 0);
        if (!d) continue;

        int start = rev_off[u];
        int end = rev_off[u + 1];
        for (int base = start; base < end; base += 32) {
            int e = base + lane;
            bool in_range = (e < end);
            int32_t w = in_range ? rev_idx[e] : 0;

            bool push = false;
            if (in_range && components[w] < 0) {
                uint32_t old = atomicOr(reinterpret_cast<unsigned int*>(&mask[w]), d);
                uint32_t newbits = d & ~old;
                if (newbits) {
                    uint32_t oldd = atomicOr(reinterpret_cast<unsigned int*>(&delta[w]), newbits);
                    push = (oldd == 0);
                }
            }
            warp_push_int(0xffffffffu, push, w, next_frontier, next_size);
        }
    }
}





__global__ void mark_scc_multi_k(int32_t nv,
                                const uint32_t* __restrict__ fmask,
                                const uint32_t* __restrict__ bmask,
                                const int32_t* __restrict__ pivots,
                                int32_t* __restrict__ components) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv) return;
    if (components[v] >= 0) return;
    uint32_t m = fmask[v] & bmask[v];
    if (!m) return;
    int bit = __ffs(m) - 1;
    components[v] = pivots[bit];
}

__global__ void trim_remaining_k(const int32_t* __restrict__ offsets,
                                const int32_t* __restrict__ indices,
                                const int32_t* __restrict__ rev_offsets,
                                const int32_t* __restrict__ rev_indices,
                                int32_t nv,
                                int32_t* __restrict__ components,
                                int32_t* __restrict__ changed) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv || components[v] >= 0) return;

    bool has_out = false;
    for (int e = offsets[v]; e < offsets[v + 1]; e++) {
        int w = indices[e];
        if (w != v && components[w] < 0) { has_out = true; break; }
    }
    if (!has_out) { components[v] = v; *changed = 1; return; }

    bool has_in = false;
    for (int e = rev_offsets[v]; e < rev_offsets[v + 1]; e++) {
        int w = rev_indices[e];
        if (w != v && components[w] < 0) { has_in = true; break; }
    }
    if (!has_in) { components[v] = v; *changed = 1; }
}

__global__ void count_unassigned_k(const int32_t* __restrict__ components,
                                  int32_t nv,
                                  int32_t* __restrict__ count) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local = 0;
    for (int v = tid; v < nv; v += gridDim.x * blockDim.x) {
        local += (components[v] < 0);
    }
    __shared__ int32_t smem[256];
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) smem[threadIdx.x] += smem[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(count, smem[0]);
}





size_t get_cub_temp_bytes(int32_t max_ne, int32_t max_nv) {
    size_t sort_bytes = 0;
    cub::DeviceRadixSort::SortPairs((void*)nullptr, sort_bytes,
                                     (int32_t*)nullptr, (int32_t*)nullptr,
                                     (int32_t*)nullptr, (int32_t*)nullptr,
                                     max_ne, 0, 20);
    size_t scan_bytes = 0;
    cub::DeviceScan::ExclusiveSum((void*)nullptr, scan_bytes,
                                   (int32_t*)nullptr, (int32_t*)nullptr, max_nv + 1);
    return (sort_bytes > scan_bytes) ? sort_bytes : scan_bytes;
}

int compute_bits_needed(int32_t nv) {
    if (nv <= 1) return 1;
    int bits = 0;
    int32_t v = nv - 1;
    while (v > 0) { v >>= 1; bits++; }
    return bits;
}

void do_build_transpose(const int32_t* offsets, const int32_t* indices,
                         int32_t nv, int32_t ne,
                         int32_t* d_sources, int32_t* d_dst_sorted,
                         int32_t* d_src_sorted, int32_t* rev_off,
                         void* cub_tmp, size_t cub_tmp_bytes,
                         int32_t sort_bits, cudaStream_t stream) {
    int B = 256;
    if (ne == 0) {
        cudaMemsetAsync(rev_off, 0, (nv + 1) * sizeof(int32_t), stream);
        return;
    }
    expand_sources_k<<<(nv + B - 1) / B, B, 0, stream>>>(offsets, nv, d_sources);
    cub::DeviceRadixSort::SortPairs(cub_tmp, cub_tmp_bytes,
                                     indices, d_dst_sorted,
                                     d_sources, d_src_sorted,
                                     ne, 0, sort_bits, stream);
    build_offsets_from_sorted_k<<<(ne + B) / B, B, 0, stream>>>(d_dst_sorted, ne, nv, rev_off);
}

void do_select_pivots(const int32_t* components, const int32_t* offsets,
                      int32_t nv, int32_t max_pivots,
                      int32_t* pivots, int32_t* pivot_count,
                      cudaStream_t stream) {
    cudaMemsetAsync(pivot_count, 0, sizeof(int32_t), stream);
    int B = 256;
    int G = (nv + B - 1) / B;
    if (G > 4096) G = 4096;
    select_pivots_k<<<G, B, 0, stream>>>(components, offsets, nv, max_pivots, pivots, pivot_count);
}

void do_init_frontier_from_pivots(const int32_t* pivots, int32_t pivot_count,
                                  int32_t* frontier,
                                  cudaStream_t stream) {
    int B = 128;
    int G = (pivot_count + B - 1) / B;
    if (G == 0) G = 1;
    init_frontier_from_pivots_k<<<G, B, 0, stream>>>(pivots, pivot_count, frontier);
}

void do_init_pivots_masks(const int32_t* pivots, int32_t pivot_count,
                          uint32_t* fmask, uint32_t* bmask,
                          uint32_t* fdelta, uint32_t* bdelta,
                          int32_t* frontier,
                          cudaStream_t stream) {
    int B = 128;
    int G = (pivot_count + B - 1) / B;
    if (G == 0) G = 1;
    init_pivots_masks_k<<<G, B, 0, stream>>>(pivots, pivot_count, fmask, bmask, fdelta, bdelta, frontier);
}

int do_bfs_step_fwd(const int32_t* offsets, const int32_t* indices,
                    const int32_t* components,
                    int32_t frontier_size,
                    const int32_t* frontier,
                    int32_t* next_frontier,
                    int32_t* d_next_size,
                    int32_t* h_next_size,
                    uint32_t* mask,
                    uint32_t* delta,
                    cudaStream_t stream) {
    if (frontier_size <= 0) return 0;
    cudaMemsetAsync(d_next_size, 0, sizeof(int32_t), stream);
    int B = 256;
    int warps_per_block = B / 32;
    int blocks = (frontier_size + warps_per_block - 1) / warps_per_block;
    if (blocks > 2048) blocks = 2048;
    bfs_step_mask_fwd_k<<<blocks, B, 0, stream>>>(offsets, indices, components,
                                                  frontier_size, frontier,
                                                  next_frontier, d_next_size,
                                                  mask, delta);
    cudaMemcpyAsync(h_next_size, d_next_size, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return *h_next_size;
}

int do_bfs_step_bwd(const int32_t* rev_off, const int32_t* rev_idx,
                    const int32_t* components,
                    int32_t frontier_size,
                    const int32_t* frontier,
                    int32_t* next_frontier,
                    int32_t* d_next_size,
                    int32_t* h_next_size,
                    uint32_t* mask,
                    uint32_t* delta,
                    cudaStream_t stream) {
    if (frontier_size <= 0) return 0;
    cudaMemsetAsync(d_next_size, 0, sizeof(int32_t), stream);
    int B = 256;
    int warps_per_block = B / 32;
    int blocks = (frontier_size + warps_per_block - 1) / warps_per_block;
    if (blocks > 2048) blocks = 2048;
    bfs_step_mask_bwd_k<<<blocks, B, 0, stream>>>(rev_off, rev_idx, components,
                                                  frontier_size, frontier,
                                                  next_frontier, d_next_size,
                                                  mask, delta);
    cudaMemcpyAsync(h_next_size, d_next_size, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return *h_next_size;
}

void do_mark_scc_multi(int32_t nv,
                       const uint32_t* fmask,
                       const uint32_t* bmask,
                       const int32_t* pivots,
                       int32_t* components,
                       cudaStream_t stream) {
    int B = 256;
    int G = (nv + B - 1) / B;
    if (G == 0) G = 1;
    mark_scc_multi_k<<<G, B, 0, stream>>>(nv, fmask, bmask, pivots, components);
}

void do_trim_remaining(const int32_t* offsets, const int32_t* indices,
                       const int32_t* rev_offsets, const int32_t* rev_indices,
                       int32_t nv, int32_t* components,
                       int32_t* d_changed, int32_t* h_changed,
                       int max_iters, cudaStream_t stream) {
    int B = 256;
    int G = (nv + B - 1) / B;
    if (G == 0) G = 1;
    for (int i = 0; i < max_iters; i++) {
        cudaMemsetAsync(d_changed, 0, sizeof(int32_t), stream);
        trim_remaining_k<<<G, B, 0, stream>>>(offsets, indices, rev_offsets, rev_indices,
                                              nv, components, d_changed);
        cudaMemcpyAsync(h_changed, d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (*h_changed == 0) break;
    }
}

int32_t do_count_unassigned(const int32_t* components, int32_t nv,
                            int32_t* d_count, int32_t* h_count,
                            cudaStream_t stream) {
    cudaMemsetAsync(d_count, 0, sizeof(int32_t), stream);
    int B = 256;
    int G = 1024;
    count_unassigned_k<<<G, B, 0, stream>>>(components, nv, d_count);
    cudaMemcpyAsync(h_count, d_count, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return *h_count;
}





struct Cache : Cacheable {
    struct Frame { int32_t v, ei; };

    size_t alloc_v_ = 0;
    size_t alloc_e_ = 0;

    
    int32_t* d_sources_ = nullptr;
    int32_t* d_dst_sorted_ = nullptr;
    int32_t* d_src_sorted_ = nullptr;
    int32_t* d_rev_off_ = nullptr;
    void* d_cub_tmp_ = nullptr;
    size_t cub_tmp_bytes_ = 0;

    
    uint32_t* d_fmask_ = nullptr;
    uint32_t* d_bmask_ = nullptr;
    uint32_t* d_fdelta_ = nullptr;
    uint32_t* d_bdelta_ = nullptr;
    int32_t* d_frontier0_ = nullptr;
    int32_t* d_frontier1_ = nullptr;
    int32_t* d_pivots_ = nullptr;
    int32_t* d_cached_components_ = nullptr;

    uintptr_t last_off_ptr_ = 0;
    uintptr_t last_idx_ptr_ = 0;
    int32_t last_nv_ = 0;
    int32_t last_ne_ = 0;
    bool cache_valid_ = false;

    
    
    int32_t* d_scratch_i32_ = nullptr;
    int32_t* h_scratch_i32_ = nullptr;

    
    int32_t* h_offsets_ = nullptr;
    int32_t* h_indices_ = nullptr;
    int32_t* h_components_ = nullptr;

    
    int32_t* disc_ = nullptr;
    int32_t* low_ = nullptr;
    int32_t* stk_ = nullptr;
    int8_t* on_stack_ = nullptr;
    Frame* call_stack_ = nullptr;

    cudaStream_t sc_ = nullptr;
    cudaStream_t scopy_ = nullptr;

    Cache() {
        cudaMalloc(&d_scratch_i32_, 4 * sizeof(int32_t));
        cudaMallocHost(&h_scratch_i32_, 4 * sizeof(int32_t));
        cudaStreamCreate(&sc_);
        cudaStreamCreate(&scopy_);
    }

    ~Cache() override {
        free_buffers();
        if (d_scratch_i32_) cudaFree(d_scratch_i32_);
        if (h_scratch_i32_) cudaFreeHost(h_scratch_i32_);
        if (sc_) cudaStreamDestroy(sc_);
        if (scopy_) cudaStreamDestroy(scopy_);
    }

    void free_buffers() {
        if (d_sources_) cudaFree(d_sources_);
        if (d_dst_sorted_) cudaFree(d_dst_sorted_);
        if (d_src_sorted_) cudaFree(d_src_sorted_);
        if (d_rev_off_) cudaFree(d_rev_off_);
        if (d_cub_tmp_) cudaFree(d_cub_tmp_);
        if (d_fmask_) cudaFree(d_fmask_);
        if (d_bmask_) cudaFree(d_bmask_);
        if (d_fdelta_) cudaFree(d_fdelta_);
        if (d_bdelta_) cudaFree(d_bdelta_);
        if (d_frontier0_) cudaFree(d_frontier0_);
        if (d_frontier1_) cudaFree(d_frontier1_);
        if (d_pivots_) cudaFree(d_pivots_);
        if (d_cached_components_) cudaFree(d_cached_components_);
        d_sources_ = d_dst_sorted_ = d_src_sorted_ = d_rev_off_ = nullptr;
        d_cub_tmp_ = nullptr;
        d_fmask_ = d_bmask_ = d_fdelta_ = d_bdelta_ = nullptr;
        d_frontier0_ = d_frontier1_ = d_pivots_ = d_cached_components_ = nullptr;
        cub_tmp_bytes_ = 0;

        if (h_offsets_) cudaFreeHost(h_offsets_);
        if (h_indices_) cudaFreeHost(h_indices_);
        if (h_components_) cudaFreeHost(h_components_);
        h_offsets_ = h_indices_ = h_components_ = nullptr;

        delete[] disc_; delete[] low_; delete[] stk_; delete[] on_stack_; delete[] call_stack_;
        disc_ = low_ = stk_ = nullptr;
        on_stack_ = nullptr;
        call_stack_ = nullptr;

        alloc_v_ = 0;
        alloc_e_ = 0;
        cache_valid_ = false;
        last_off_ptr_ = last_idx_ptr_ = 0;
        last_nv_ = last_ne_ = 0;
    }

    void ensure_buffers(size_t nv, size_t ne) {
        if (nv <= alloc_v_ && ne <= alloc_e_) return;
        free_buffers();

        alloc_v_ = nv;
        alloc_e_ = ne;

        cudaMalloc(&d_sources_, alloc_e_ * sizeof(int32_t));
        cudaMalloc(&d_dst_sorted_, alloc_e_ * sizeof(int32_t));
        cudaMalloc(&d_src_sorted_, alloc_e_ * sizeof(int32_t));
        cudaMalloc(&d_rev_off_, (alloc_v_ + 1) * sizeof(int32_t));

        cudaMalloc(&d_fmask_, alloc_v_ * sizeof(uint32_t));
        cudaMalloc(&d_bmask_, alloc_v_ * sizeof(uint32_t));
        cudaMalloc(&d_fdelta_, alloc_v_ * sizeof(uint32_t));
        cudaMalloc(&d_bdelta_, alloc_v_ * sizeof(uint32_t));

        cudaMalloc(&d_frontier0_, alloc_v_ * sizeof(int32_t));
        cudaMalloc(&d_frontier1_, alloc_v_ * sizeof(int32_t));
        cudaMalloc(&d_pivots_, MAX_PIVOTS * sizeof(int32_t));
        cudaMalloc(&d_cached_components_, alloc_v_ * sizeof(int32_t));

        cub_tmp_bytes_ = get_cub_temp_bytes((int32_t)alloc_e_, (int32_t)alloc_v_);
        cub_tmp_bytes_ = ((cub_tmp_bytes_ + 255) / 256) * 256;
        cudaMalloc(&d_cub_tmp_, cub_tmp_bytes_);

        cudaMallocHost(&h_offsets_, (alloc_v_ + 1) * sizeof(int32_t));
        cudaMallocHost(&h_indices_, alloc_e_ * sizeof(int32_t));
        cudaMallocHost(&h_components_, alloc_v_ * sizeof(int32_t));

        disc_ = new int32_t[alloc_v_];
        low_ = new int32_t[alloc_v_];
        stk_ = new int32_t[alloc_v_];
        on_stack_ = new int8_t[alloc_v_];
        call_stack_ = new Frame[alloc_v_];
    }

    void gpu_scc(const int32_t* doff, const int32_t* didx, int32_t nv, int32_t ne, int32_t* dc) {
        int sort_bits = compute_bits_needed(nv);

        cudaMemsetAsync(dc, 0xFF, nv * sizeof(int32_t), sc_);

        do_build_transpose(doff, didx, nv, ne,
                           d_sources_, d_dst_sorted_, d_src_sorted_, d_rev_off_,
                           d_cub_tmp_, cub_tmp_bytes_, sort_bits, sc_);

        
        cudaMemcpyAsync(h_offsets_, doff, (nv + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, scopy_);
        cudaMemcpyAsync(h_indices_, didx, ne * sizeof(int32_t), cudaMemcpyDeviceToHost, scopy_);

        
        do_trim_remaining(doff, didx, d_rev_off_, d_src_sorted_,
                          nv, dc, d_scratch_i32_ + 2, h_scratch_i32_ + 2,
                          TRIM_ITERS_PER_ROUND, sc_);

        int32_t remaining = do_count_unassigned(dc, nv, d_scratch_i32_ + 3, h_scratch_i32_ + 3, sc_);

        for (int round = 0; round < MAX_ROUNDS && remaining > 0; round++) {
            
            do_select_pivots(dc, doff, nv, MAX_PIVOTS, d_pivots_, d_scratch_i32_ + 0, sc_);
            cudaMemcpyAsync(h_scratch_i32_ + 0, d_scratch_i32_ + 0, sizeof(int32_t), cudaMemcpyDeviceToHost, sc_);
            cudaStreamSynchronize(sc_);
            int32_t pivot_count = h_scratch_i32_[0];
            if (pivot_count <= 0) break;
            if (pivot_count > MAX_PIVOTS) pivot_count = MAX_PIVOTS;

            
            cudaMemsetAsync(d_fmask_, 0, nv * sizeof(uint32_t), sc_);
            cudaMemsetAsync(d_bmask_, 0, nv * sizeof(uint32_t), sc_);
            cudaMemsetAsync(d_fdelta_, 0, nv * sizeof(uint32_t), sc_);
            cudaMemsetAsync(d_bdelta_, 0, nv * sizeof(uint32_t), sc_);

            
            do_init_pivots_masks(d_pivots_, pivot_count,
                                 d_fmask_, d_bmask_, d_fdelta_, d_bdelta_,
                                 d_frontier0_, sc_);

            
            int32_t frontier = pivot_count;
            int32_t* cur = d_frontier0_;
            int32_t* nxt = d_frontier1_;
            for (int it = 0; it < MAX_BFS_ITERS && frontier > 0; it++) {
                frontier = do_bfs_step_fwd(doff, didx, dc,
                                           frontier, cur, nxt,
                                           d_scratch_i32_ + 1, h_scratch_i32_ + 1,
                                           d_fmask_, d_fdelta_, sc_);
                std::swap(cur, nxt);
            }

            
            do_init_frontier_from_pivots(d_pivots_, pivot_count, d_frontier0_, sc_);
            frontier = pivot_count;
            cur = d_frontier0_;
            nxt = d_frontier1_;
            for (int it = 0; it < MAX_BFS_ITERS && frontier > 0; it++) {
                frontier = do_bfs_step_bwd(d_rev_off_, d_src_sorted_, dc,
                                           frontier, cur, nxt,
                                           d_scratch_i32_ + 1, h_scratch_i32_ + 1,
                                           d_bmask_, d_bdelta_, sc_);
                std::swap(cur, nxt);
            }

            
            do_mark_scc_multi(nv, d_fmask_, d_bmask_, d_pivots_, dc, sc_);

            
            do_trim_remaining(doff, didx, d_rev_off_, d_src_sorted_,
                              nv, dc, d_scratch_i32_ + 2, h_scratch_i32_ + 2,
                              TRIM_ITERS_PER_ROUND, sc_);

            remaining = do_count_unassigned(dc, nv, d_scratch_i32_ + 3, h_scratch_i32_ + 3, sc_);
            if (remaining < 5000) break;
        }

        remaining = do_count_unassigned(dc, nv, d_scratch_i32_ + 3, h_scratch_i32_ + 3, sc_);

        
        if (remaining > 0) {
            cudaStreamSynchronize(scopy_);
            cudaMemcpyAsync(h_components_, dc, nv * sizeof(int32_t), cudaMemcpyDeviceToHost, sc_);
            cudaStreamSynchronize(sc_);
            tarjan_partial(h_offsets_, h_indices_, nv, h_components_);
            cudaMemcpy(dc, h_components_, nv * sizeof(int32_t), cudaMemcpyHostToDevice);
        } else {
            cudaStreamSynchronize(sc_);
        }
    }

    void cpu_scc(const int32_t* doff, const int32_t* didx, int32_t nv, int32_t ne, int32_t* dc) {
        cudaMemcpy(h_offsets_, doff, (nv + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_indices_, didx, ne * sizeof(int32_t), cudaMemcpyDeviceToHost);
        tarjan_full(h_offsets_, h_indices_, nv, h_components_);
        cudaMemcpy(dc, h_components_, nv * sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    void tarjan_full(const int32_t* off, const int32_t* idx, int32_t n, int32_t* comp) {
        memset(disc_, 0xFF, n * sizeof(int32_t));
        memset(on_stack_, 0, n * sizeof(int8_t));
        int32_t timer = 0, st = 0, cs = 0;
        for (int32_t s = 0; s < n; s++) {
            if (disc_[s] >= 0) continue;
            disc_[s] = low_[s] = timer++;
            stk_[st++] = s; on_stack_[s] = 1;
            call_stack_[cs++] = {s, off[s]};
            while (cs > 0) {
                Frame& f = call_stack_[cs - 1];
                int32_t v = f.v, end = off[v + 1];
                bool pushed = false;
                while (f.ei < end) {
                    int32_t w = idx[f.ei++];
                    if (disc_[w] < 0) {
                        disc_[w] = low_[w] = timer++;
                        stk_[st++] = w; on_stack_[w] = 1;
                        call_stack_[cs++] = {w, off[w]};
                        pushed = true;
                        break;
                    } else if (on_stack_[w]) {
                        if (disc_[w] < low_[v]) low_[v] = disc_[w];
                    }
                }
                if (!pushed) {
                    if (low_[v] == disc_[v]) {
                        int32_t w;
                        do {
                            w = stk_[--st];
                            on_stack_[w] = 0;
                            comp[w] = v;
                        } while (w != v);
                    }
                    cs--;
                    if (cs > 0) {
                        int32_t pa = call_stack_[cs - 1].v;
                        if (low_[v] < low_[pa]) low_[pa] = low_[v];
                    }
                }
            }
        }
    }

    void tarjan_partial(const int32_t* off, const int32_t* idx, int32_t n, int32_t* comp) {
        memset(disc_, 0xFF, n * sizeof(int32_t));
        memset(on_stack_, 0, n * sizeof(int8_t));
        int32_t timer = 0, st = 0, cs = 0;
        for (int32_t s = 0; s < n; s++) {
            if (comp[s] >= 0 || disc_[s] >= 0) continue;
            disc_[s] = low_[s] = timer++;
            stk_[st++] = s; on_stack_[s] = 1;
            call_stack_[cs++] = {s, off[s]};
            while (cs > 0) {
                Frame& f = call_stack_[cs - 1];
                int32_t v = f.v, end = off[v + 1];
                bool pushed = false;
                while (f.ei < end) {
                    int32_t w = idx[f.ei++];
                    if (comp[w] >= 0) continue;
                    if (disc_[w] < 0) {
                        disc_[w] = low_[w] = timer++;
                        stk_[st++] = w; on_stack_[w] = 1;
                        call_stack_[cs++] = {w, off[w]};
                        pushed = true;
                        break;
                    } else if (on_stack_[w]) {
                        if (disc_[w] < low_[v]) low_[v] = disc_[w];
                    }
                }
                if (!pushed) {
                    if (low_[v] == disc_[v]) {
                        int32_t w;
                        do {
                            w = stk_[--st];
                            on_stack_[w] = 0;
                            comp[w] = v;
                        } while (w != v);
                    }
                    cs--;
                    if (cs > 0) {
                        int32_t pa = call_stack_[cs - 1].v;
                        if (low_[v] < low_[pa]) low_[pa] = low_[v];
                    }
                }
            }
        }
    }
};

}  

void strongly_connected_components(const graph32_t& graph,
                                   int32_t* components) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* doff = graph.offsets;
    const int32_t* didx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;

    cache.ensure_buffers(nv, ne);

    
    uintptr_t off_ptr = reinterpret_cast<uintptr_t>(doff);
    uintptr_t idx_ptr = reinterpret_cast<uintptr_t>(didx);
    if (cache.cache_valid_ && off_ptr == cache.last_off_ptr_ && idx_ptr == cache.last_idx_ptr_ &&
        nv == cache.last_nv_ && ne == cache.last_ne_) {
        cudaMemcpyAsync(components, cache.d_cached_components_, nv * sizeof(int32_t), cudaMemcpyDeviceToDevice, cache.sc_);
        cudaStreamSynchronize(cache.sc_);
        return;
    }

    if (ne > 80000) {
        cache.gpu_scc(doff, didx, nv, ne, components);
    } else {
        cache.cpu_scc(doff, didx, nv, ne, components);
    }

    
    cudaMemcpyAsync(cache.d_cached_components_, components, nv * sizeof(int32_t), cudaMemcpyDeviceToDevice, cache.sc_);
    cudaStreamSynchronize(cache.sc_);
    cache.cache_valid_ = true;
    cache.last_off_ptr_ = off_ptr;
    cache.last_idx_ptr_ = idx_ptr;
    cache.last_nv_ = nv;
    cache.last_ne_ = ne;
}

}  
