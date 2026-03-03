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
#include <cub/cub.cuh>
#include <cstdint>
#include <cstring>

namespace aai {

namespace {





__global__ void expand_sources_k(const int32_t* __restrict__ offsets,
                                  int32_t nv,
                                  int32_t* __restrict__ sources) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= nv) return;
    int start = offsets[u];
    int end = offsets[u + 1];
    for (int e = start; e < end; e++) {
        sources[e] = u;
    }
}

__global__ void build_offsets_from_sorted_k(const int32_t* __restrict__ sorted_dst,
                                             int32_t ne, int32_t nv,
                                             int32_t* __restrict__ offsets) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i == 0 && ne > 0) {
        int d = sorted_dst[0];
        for (int v = 0; v <= d; v++) offsets[v] = 0;
    }
    if (i == 0 && ne == 0) {
        for (int v = 0; v <= nv; v++) offsets[v] = 0;
        return;
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





__global__ void bfs_dual_level_k(const int32_t* __restrict__ fwd_off,
                                  const int32_t* __restrict__ fwd_idx,
                                  const int32_t* __restrict__ rev_off,
                                  const int32_t* __restrict__ rev_idx,
                                  int32_t nv,
                                  int32_t* __restrict__ fw_dist,
                                  int32_t* __restrict__ bw_dist,
                                  int32_t level,
                                  int32_t* __restrict__ changed) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv) return;

    if (fw_dist[v] == level) {
        int start = fwd_off[v], end = fwd_off[v + 1];
        for (int e = start; e < end; e++) {
            int w = fwd_idx[e];
            if (fw_dist[w] == -1) {
                fw_dist[w] = level + 1;
                *changed = 1;
            }
        }
    }

    if (bw_dist[v] == level) {
        int start = rev_off[v], end = rev_off[v + 1];
        for (int e = start; e < end; e++) {
            int w = rev_idx[e];
            if (bw_dist[w] == -1) {
                bw_dist[w] = level + 1;
                *changed = 1;
            }
        }
    }
}

__global__ void bfs_dual_level_masked_k(const int32_t* __restrict__ fwd_off,
                                         const int32_t* __restrict__ fwd_idx,
                                         const int32_t* __restrict__ rev_off,
                                         const int32_t* __restrict__ rev_idx,
                                         int32_t nv,
                                         int32_t* __restrict__ fw_dist,
                                         int32_t* __restrict__ bw_dist,
                                         const int32_t* __restrict__ components,
                                         int32_t level,
                                         int32_t* __restrict__ changed) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv || components[v] >= 0) return;

    if (fw_dist[v] == level) {
        int start = fwd_off[v], end = fwd_off[v + 1];
        for (int e = start; e < end; e++) {
            int w = fwd_idx[e];
            if (components[w] < 0 && fw_dist[w] == -1) {
                fw_dist[w] = level + 1;
                *changed = 1;
            }
        }
    }

    if (bw_dist[v] == level) {
        int start = rev_off[v], end = rev_off[v + 1];
        for (int e = start; e < end; e++) {
            int w = rev_idx[e];
            if (components[w] < 0 && bw_dist[w] == -1) {
                bw_dist[w] = level + 1;
                *changed = 1;
            }
        }
    }
}





__global__ void mark_scc_k(int32_t nv, const int32_t* __restrict__ fw_dist,
                            const int32_t* __restrict__ bw_dist,
                            int32_t* __restrict__ components, int32_t pivot) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv) return;
    if (components[v] < 0 && fw_dist[v] >= 0 && bw_dist[v] >= 0) {
        components[v] = pivot;
    }
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





__global__ void count_unassigned_k(const int32_t* __restrict__ components, int32_t nv,
                                    int32_t* __restrict__ count) {
    __shared__ int32_t s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < nv && components[v] < 0) atomicAdd(&s_count, 1);
    __syncthreads();
    if (threadIdx.x == 0 && s_count > 0) atomicAdd(count, s_count);
}

__global__ void find_max_degree_pivot_k(const int32_t* __restrict__ components,
                                         const int32_t* __restrict__ offsets,
                                         int32_t nv,
                                         int32_t* __restrict__ best_deg) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv || components[v] >= 0) return;
    int deg = offsets[v + 1] - offsets[v];
    atomicMax(best_deg, deg);
}

__global__ void find_pivot_with_deg_k(const int32_t* __restrict__ components,
                                       const int32_t* __restrict__ offsets,
                                       int32_t nv,
                                       const int32_t* __restrict__ best_deg,
                                       int32_t* __restrict__ best_vtx) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv || components[v] >= 0) return;
    int deg = offsets[v + 1] - offsets[v];
    if (deg == *best_deg) {
        atomicMin(best_vtx, v);
    }
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





void launch_build_transpose(const int32_t* offsets, const int32_t* indices,
                              int32_t nv, int32_t ne,
                              int32_t* d_sources, int32_t* d_dst_sorted,
                              int32_t* d_src_sorted, int32_t* rev_off,
                              void* cub_tmp, size_t cub_tmp_bytes,
                              int32_t sort_bits,
                              cudaStream_t stream) {
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

    build_offsets_from_sorted_k<<<(ne + B) / B, B, 0, stream>>>(
        d_dst_sorted, ne, nv, rev_off);
}

int launch_bfs_converge(const int32_t* fwd_off, const int32_t* fwd_idx,
                         const int32_t* rev_off, const int32_t* rev_idx,
                         int32_t nv, int32_t* fw_dist, int32_t* bw_dist,
                         int32_t pivot, int32_t max_levels,
                         int32_t* d_changed, int32_t* h_changed,
                         cudaStream_t stream) {
    int B = 256;
    int G = (nv + B - 1) / B;
    if (G == 0) G = 1;

    cudaMemsetAsync(fw_dist, 0xFF, nv * sizeof(int32_t), stream);
    cudaMemsetAsync(bw_dist, 0xFF, nv * sizeof(int32_t), stream);
    cudaMemsetAsync(fw_dist + pivot, 0, sizeof(int32_t), stream);
    cudaMemsetAsync(bw_dist + pivot, 0, sizeof(int32_t), stream);

    int batch_size = 20;

    for (int batch = 0; batch < max_levels; batch += batch_size) {
        cudaMemsetAsync(d_changed, 0, sizeof(int32_t), stream);
        int end = batch + batch_size;
        if (end > max_levels) end = max_levels;
        for (int level = batch; level < end; level++) {
            bfs_dual_level_k<<<G, B, 0, stream>>>(fwd_off, fwd_idx, rev_off, rev_idx,
                                                    nv, fw_dist, bw_dist, level, d_changed);
        }
        cudaMemcpyAsync(h_changed, d_changed, sizeof(int32_t),
                          cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (*h_changed == 0) break;
    }
    return 0;
}

int launch_bfs_converge_masked(const int32_t* fwd_off, const int32_t* fwd_idx,
                                const int32_t* rev_off, const int32_t* rev_idx,
                                int32_t nv, int32_t* fw_dist, int32_t* bw_dist,
                                const int32_t* components,
                                int32_t pivot, int32_t max_levels,
                                int32_t* d_changed, int32_t* h_changed,
                                cudaStream_t stream) {
    int B = 256;
    int G = (nv + B - 1) / B;
    if (G == 0) G = 1;

    cudaMemsetAsync(fw_dist, 0xFF, nv * sizeof(int32_t), stream);
    cudaMemsetAsync(bw_dist, 0xFF, nv * sizeof(int32_t), stream);
    cudaMemsetAsync(fw_dist + pivot, 0, sizeof(int32_t), stream);
    cudaMemsetAsync(bw_dist + pivot, 0, sizeof(int32_t), stream);

    int batch_size = 20;

    for (int batch = 0; batch < max_levels; batch += batch_size) {
        cudaMemsetAsync(d_changed, 0, sizeof(int32_t), stream);
        int end = batch + batch_size;
        if (end > max_levels) end = max_levels;
        for (int level = batch; level < end; level++) {
            bfs_dual_level_masked_k<<<G, B, 0, stream>>>(fwd_off, fwd_idx, rev_off, rev_idx,
                                                           nv, fw_dist, bw_dist, components,
                                                           level, d_changed);
        }
        cudaMemcpyAsync(h_changed, d_changed, sizeof(int32_t),
                          cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (*h_changed == 0) break;
    }
    return 0;
}

void launch_mark_scc(int32_t nv, const int32_t* fw_dist, const int32_t* bw_dist,
                      int32_t* components, int32_t pivot, cudaStream_t stream) {
    int B = 256;
    int G = (nv + B - 1) / B;
    if (G == 0) G = 1;
    mark_scc_k<<<G, B, 0, stream>>>(nv, fw_dist, bw_dist, components, pivot);
}

void launch_trim_remaining(const int32_t* offsets, const int32_t* indices,
                            const int32_t* rev_offsets, const int32_t* rev_indices,
                            int32_t nv, int32_t* components,
                            int32_t* d_changed, int32_t* h_changed,
                            int max_iters, cudaStream_t stream) {
    int B = 256;
    int G = (nv + B - 1) / B;
    if (G == 0) G = 1;
    int batch = 3;
    for (int i = 0; i < max_iters; i += batch) {
        cudaMemsetAsync(d_changed, 0, sizeof(int32_t), stream);
        int end = i + batch;
        if (end > max_iters) end = max_iters;
        for (int j = i; j < end; j++) {
            trim_remaining_k<<<G, B, 0, stream>>>(offsets, indices, rev_offsets, rev_indices,
                                                    nv, components, d_changed);
        }
        cudaMemcpyAsync(h_changed, d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (*h_changed == 0) break;
    }
}

int32_t launch_count_unassigned(const int32_t* components, int32_t nv,
                                  int32_t* d_count, int32_t* h_count,
                                  cudaStream_t stream) {
    int B = 256;
    int G = (nv + B - 1) / B;
    if (G == 0) G = 1;
    cudaMemsetAsync(d_count, 0, sizeof(int32_t), stream);
    count_unassigned_k<<<G, B, 0, stream>>>(components, nv, d_count);
    cudaMemcpyAsync(h_count, d_count, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return *h_count;
}

int32_t launch_find_pivot(const int32_t* components, const int32_t* offsets,
                            int32_t nv, int32_t* d_vals, int32_t* h_vals,
                            cudaStream_t stream) {
    int B = 256;
    int G = (nv + B - 1) / B;
    if (G == 0) G = 1;

    cudaMemsetAsync(d_vals, 0, sizeof(int32_t), stream);
    find_max_degree_pivot_k<<<G, B, 0, stream>>>(components, offsets, nv, d_vals);

    cudaMemsetAsync(d_vals + 1, 0x7F, sizeof(int32_t), stream);
    find_pivot_with_deg_k<<<G, B, 0, stream>>>(components, offsets, nv, d_vals, d_vals + 1);

    cudaMemcpyAsync(h_vals, d_vals + 1, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return *h_vals;
}





struct Frame { int32_t v, ei; };

struct Cache : Cacheable {
    static constexpr int32_t MAX_BFS_LEVELS = 60;
    static constexpr int MAX_FWBW_ROUNDS = 2;
    static constexpr int MAX_TRIM_ITERS = 15;

    size_t alloc_v_ = 0;
    size_t alloc_e_ = 0;

    int32_t* d_sources = nullptr;
    int32_t* d_dst_sorted = nullptr;
    int32_t* d_src_sorted = nullptr;
    int32_t* d_rev_off = nullptr;
    int32_t* d_fw_dist = nullptr;
    int32_t* d_bw_dist = nullptr;
    void* d_cub_tmp = nullptr;
    size_t cub_tmp_bytes = 0;

    int32_t* d_flag = nullptr;

    int32_t* h_offsets = nullptr;
    int32_t* h_indices = nullptr;
    int32_t* h_components = nullptr;
    int32_t* h_flag = nullptr;

    int32_t* disc = nullptr;
    int32_t* low = nullptr;
    int32_t* stk = nullptr;
    int8_t* on_stack = nullptr;
    Frame* call_stack = nullptr;

    cudaStream_t sc = nullptr;
    cudaStream_t scopy = nullptr;

    Cache() {
        cudaMalloc(&d_flag, 4 * sizeof(int32_t));
        cudaMallocHost(&h_flag, 4 * sizeof(int32_t));
        cudaStreamCreate(&sc);
        cudaStreamCreate(&scopy);
    }

    ~Cache() override {
        if (d_sources) cudaFree(d_sources);
        if (d_dst_sorted) cudaFree(d_dst_sorted);
        if (d_src_sorted) cudaFree(d_src_sorted);
        if (d_rev_off) cudaFree(d_rev_off);
        if (d_fw_dist) cudaFree(d_fw_dist);
        if (d_bw_dist) cudaFree(d_bw_dist);
        if (d_cub_tmp) cudaFree(d_cub_tmp);
        if (d_flag) cudaFree(d_flag);
        if (sc) cudaStreamDestroy(sc);
        if (scopy) cudaStreamDestroy(scopy);
        if (h_offsets) cudaFreeHost(h_offsets);
        if (h_indices) cudaFreeHost(h_indices);
        if (h_components) cudaFreeHost(h_components);
        if (h_flag) cudaFreeHost(h_flag);
        delete[] disc;
        delete[] low;
        delete[] stk;
        delete[] on_stack;
        delete[] call_stack;
    }

    void ensure_buffers(size_t nv, size_t ne) {
        if (nv <= alloc_v_ && ne <= alloc_e_) return;

        if (d_sources) { cudaFree(d_sources); d_sources = nullptr; }
        if (d_dst_sorted) { cudaFree(d_dst_sorted); d_dst_sorted = nullptr; }
        if (d_src_sorted) { cudaFree(d_src_sorted); d_src_sorted = nullptr; }
        if (d_rev_off) { cudaFree(d_rev_off); d_rev_off = nullptr; }
        if (d_fw_dist) { cudaFree(d_fw_dist); d_fw_dist = nullptr; }
        if (d_bw_dist) { cudaFree(d_bw_dist); d_bw_dist = nullptr; }
        if (d_cub_tmp) { cudaFree(d_cub_tmp); d_cub_tmp = nullptr; }

        if (h_offsets) { cudaFreeHost(h_offsets); h_offsets = nullptr; }
        if (h_indices) { cudaFreeHost(h_indices); h_indices = nullptr; }
        if (h_components) { cudaFreeHost(h_components); h_components = nullptr; }

        delete[] disc; disc = nullptr;
        delete[] low; low = nullptr;
        delete[] stk; stk = nullptr;
        delete[] on_stack; on_stack = nullptr;
        delete[] call_stack; call_stack = nullptr;

        alloc_v_ = nv;
        alloc_e_ = ne;

        cudaMalloc(&d_sources, alloc_e_ * sizeof(int32_t));
        cudaMalloc(&d_dst_sorted, alloc_e_ * sizeof(int32_t));
        cudaMalloc(&d_src_sorted, alloc_e_ * sizeof(int32_t));
        cudaMalloc(&d_rev_off, (alloc_v_ + 1) * sizeof(int32_t));
        cudaMalloc(&d_fw_dist, alloc_v_ * sizeof(int32_t));
        cudaMalloc(&d_bw_dist, alloc_v_ * sizeof(int32_t));

        cub_tmp_bytes = get_cub_temp_bytes(alloc_e_, alloc_v_);
        cub_tmp_bytes = ((cub_tmp_bytes + 255) / 256) * 256;
        cudaMalloc(&d_cub_tmp, cub_tmp_bytes);

        cudaMallocHost(&h_offsets, (alloc_v_ + 1) * sizeof(int32_t));
        cudaMallocHost(&h_indices, alloc_e_ * sizeof(int32_t));
        cudaMallocHost(&h_components, alloc_v_ * sizeof(int32_t));

        disc = new int32_t[alloc_v_];
        low = new int32_t[alloc_v_];
        stk = new int32_t[alloc_v_];
        on_stack = new int8_t[alloc_v_];
        call_stack = new Frame[alloc_v_];
    }
};





void tarjan_full(Cache& c, const int32_t* off, const int32_t* idx, int32_t n, int32_t* comp) {
    memset(c.disc, 0xFF, n * sizeof(int32_t));
    memset(c.on_stack, 0, n * sizeof(int8_t));
    int32_t timer = 0, st = 0, cs = 0;
    for (int32_t s = 0; s < n; s++) {
        if (c.disc[s] >= 0) continue;
        c.disc[s] = c.low[s] = timer++;
        c.stk[st++] = s; c.on_stack[s] = 1;
        c.call_stack[cs++] = {s, off[s]};
        while (cs > 0) {
            Frame& f = c.call_stack[cs - 1]; int32_t v = f.v, end = off[v + 1]; bool pushed = false;
            while (f.ei < end) { int32_t w = idx[f.ei++];
                if (c.disc[w] < 0) { c.disc[w] = c.low[w] = timer++; c.stk[st++] = w; c.on_stack[w] = 1; c.call_stack[cs++] = {w, off[w]}; pushed = true; break; }
                else if (c.on_stack[w]) { if (c.disc[w] < c.low[v]) c.low[v] = c.disc[w]; }
            }
            if (!pushed) { if (c.low[v] == c.disc[v]) { int32_t w; do { w = c.stk[--st]; c.on_stack[w] = 0; comp[w] = v; } while (w != v); }
                cs--; if (cs > 0) { int32_t pa = c.call_stack[cs - 1].v; if (c.low[v] < c.low[pa]) c.low[pa] = c.low[v]; } }
        }
    }
}

void tarjan_partial(Cache& c, const int32_t* off, const int32_t* idx, int32_t n, int32_t* comp) {
    memset(c.disc, 0xFF, n * sizeof(int32_t));
    memset(c.on_stack, 0, n * sizeof(int8_t));
    int32_t timer = 0, st = 0, cs = 0;
    for (int32_t s = 0; s < n; s++) {
        if (comp[s] >= 0 || c.disc[s] >= 0) continue;
        c.disc[s] = c.low[s] = timer++;
        c.stk[st++] = s; c.on_stack[s] = 1;
        c.call_stack[cs++] = {s, off[s]};
        while (cs > 0) {
            Frame& f = c.call_stack[cs - 1]; int32_t v = f.v, end = off[v + 1]; bool pushed = false;
            while (f.ei < end) { int32_t w = idx[f.ei++];
                if (comp[w] >= 0) continue;
                if (c.disc[w] < 0) { c.disc[w] = c.low[w] = timer++; c.stk[st++] = w; c.on_stack[w] = 1; c.call_stack[cs++] = {w, off[w]}; pushed = true; break; }
                else if (c.on_stack[w]) { if (c.disc[w] < c.low[v]) c.low[v] = c.disc[w]; }
            }
            if (!pushed) { if (c.low[v] == c.disc[v]) { int32_t w; do { w = c.stk[--st]; c.on_stack[w] = 0; comp[w] = v; } while (w != v); }
                cs--; if (cs > 0) { int32_t pa = c.call_stack[cs - 1].v; if (c.low[v] < c.low[pa]) c.low[pa] = c.low[v]; } }
        }
    }
}





void gpu_scc(Cache& c, const int32_t* doff, const int32_t* didx,
             int32_t nv, int32_t ne, int32_t* dc) {
    int sort_bits = compute_bits_needed(nv);

    cudaMemsetAsync(dc, 0xFF, nv * sizeof(int32_t), c.sc);

    launch_build_transpose(doff, didx, nv, ne,
                            c.d_sources, c.d_dst_sorted, c.d_src_sorted, c.d_rev_off,
                            c.d_cub_tmp, c.cub_tmp_bytes, sort_bits, c.sc);

    cudaMemcpyAsync(c.h_offsets, doff, (nv + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, c.scopy);
    cudaMemcpyAsync(c.h_indices, didx, ne * sizeof(int32_t), cudaMemcpyDeviceToHost, c.scopy);

    int32_t pivot = 0;

    launch_bfs_converge(doff, didx, c.d_rev_off, c.d_src_sorted,
                         nv, c.d_fw_dist, c.d_bw_dist, pivot, Cache::MAX_BFS_LEVELS,
                         c.d_flag, c.h_flag, c.sc);
    launch_mark_scc(nv, c.d_fw_dist, c.d_bw_dist, dc, pivot, c.sc);

    launch_trim_remaining(doff, didx, c.d_rev_off, c.d_src_sorted,
                           nv, dc, c.d_flag, c.h_flag, Cache::MAX_TRIM_ITERS, c.sc);

    int32_t remaining = launch_count_unassigned(dc, nv, c.d_flag, c.h_flag, c.sc);

    for (int round = 1; round < Cache::MAX_FWBW_ROUNDS && remaining > 0; round++) {
        if (remaining < 5000) break;

        int32_t new_pivot = launch_find_pivot(dc, doff, nv, c.d_flag, c.h_flag, c.sc);
        if (new_pivot >= nv) break;

        launch_bfs_converge_masked(doff, didx, c.d_rev_off, c.d_src_sorted,
                                    nv, c.d_fw_dist, c.d_bw_dist, dc,
                                    new_pivot, Cache::MAX_BFS_LEVELS,
                                    c.d_flag, c.h_flag, c.sc);
        launch_mark_scc(nv, c.d_fw_dist, c.d_bw_dist, dc, new_pivot, c.sc);

        launch_trim_remaining(doff, didx, c.d_rev_off, c.d_src_sorted,
                               nv, dc, c.d_flag, c.h_flag, Cache::MAX_TRIM_ITERS, c.sc);

        remaining = launch_count_unassigned(dc, nv, c.d_flag, c.h_flag, c.sc);
    }

    if (remaining > 0) {
        cudaStreamSynchronize(c.scopy);
        cudaMemcpyAsync(c.h_components, dc, nv * sizeof(int32_t), cudaMemcpyDeviceToHost, c.sc);
        cudaStreamSynchronize(c.sc);
        tarjan_partial(c, c.h_offsets, c.h_indices, nv, c.h_components);
        cudaMemcpy(dc, c.h_components, nv * sizeof(int32_t), cudaMemcpyHostToDevice);
    } else {
        cudaStreamSynchronize(c.sc);
    }
}





void cpu_scc(Cache& c, const int32_t* doff, const int32_t* didx,
             int32_t nv, int32_t ne, int32_t* dc) {
    cudaMemcpy(c.h_offsets, doff, (nv + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(c.h_indices, didx, ne * sizeof(int32_t), cudaMemcpyDeviceToHost);
    tarjan_full(c, c.h_offsets, c.h_indices, nv, c.h_components);
    cudaMemcpy(dc, c.h_components, nv * sizeof(int32_t), cudaMemcpyHostToDevice);
}

}  

void strongly_connected_components(const graph32_t& graph,
                                   int32_t* components) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    const int32_t* doff = graph.offsets;
    const int32_t* didx = graph.indices;

    cache.ensure_buffers(nv, ne);

    if (ne > 80000) {
        gpu_scc(cache, doff, didx, nv, ne, components);
    } else {
        cpu_scc(cache, doff, didx, nv, ne, components);
    }
}

}  
