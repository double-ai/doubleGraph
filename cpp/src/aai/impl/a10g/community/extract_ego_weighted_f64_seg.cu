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
#include <algorithm>

namespace aai {

namespace {

static_assert(sizeof(std::size_t) == sizeof(int64_t), "size_t must be 64-bit");

struct Cache : Cacheable {
    void* d_temp = nullptr;
    size_t temp_size = 0;

    void ensure_temp(size_t needed) {
        if (needed > temp_size) {
            if (d_temp) cudaFree(d_temp);
            temp_size = needed * 2;
            cudaMalloc(&d_temp, temp_size);
        }
    }

    ~Cache() override {
        if (d_temp) { cudaFree(d_temp); d_temp = nullptr; }
    }
};



__device__ __forceinline__ bool bitmap_test(const uint32_t* bitmap, int32_t v) {
    return (bitmap[v >> 5] >> (v & 31)) & 1;
}


__global__ void bfs_init_kernel(
    const int32_t* __restrict__ sources,
    uint32_t* __restrict__ bitmaps,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_sizes,
    int32_t n_sources, int32_t bitmap_words, int64_t max_frontier
) {
    int sid = blockIdx.x;
    if (sid >= n_sources) return;
    uint32_t* bm = bitmaps + (int64_t)sid * bitmap_words;
    for (int i = threadIdx.x; i < bitmap_words; i += blockDim.x) bm[i] = 0;
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t src = sources[sid];
        bm[src >> 5] = 1u << (src & 31);
        frontier[sid * max_frontier] = src;
        frontier_sizes[sid] = 1;
    }
}


__global__ void bfs_expand_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    uint32_t* __restrict__ bitmaps,
    const int32_t* __restrict__ cur_frontier,
    const int32_t* __restrict__ cur_sizes,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_sizes,
    int32_t n_sources, int32_t bitmap_words, int64_t max_frontier
) {
    int sid = blockIdx.x;
    if (sid >= n_sources) return;
    uint32_t* bm = bitmaps + (int64_t)sid * bitmap_words;
    const int32_t* cur = cur_frontier + (int64_t)sid * max_frontier;
    int32_t* nxt = next_frontier + (int64_t)sid * max_frontier;
    int32_t cur_size = cur_sizes[sid];

    __shared__ int nxt_count;
    if (threadIdx.x == 0) nxt_count = 0;
    __syncthreads();

    if (cur_size <= 64) {
        for (int fi = 0; fi < cur_size; fi++) {
            int32_t v = cur[fi];
            int32_t s = csr_offsets[v], e = csr_offsets[v + 1];
            for (int j = threadIdx.x; j < (e - s); j += blockDim.x) {
                int32_t u = csr_indices[s + j];
                uint32_t old = atomicOr(&bm[u >> 5], 1u << (u & 31));
                if ((old & (1u << (u & 31))) == 0) {
                    int pos = atomicAdd(&nxt_count, 1);
                    if (pos < max_frontier) nxt[pos] = u;
                }
            }
        }
    } else {
        for (int fi = threadIdx.x; fi < cur_size; fi += blockDim.x) {
            int32_t v = cur[fi];
            int32_t s = csr_offsets[v], e = csr_offsets[v + 1];
            for (int j = s; j < e; j++) {
                int32_t u = csr_indices[j];
                uint32_t old = atomicOr(&bm[u >> 5], 1u << (u & 31));
                if ((old & (1u << (u & 31))) == 0) {
                    int pos = atomicAdd(&nxt_count, 1);
                    if (pos < max_frontier) nxt[pos] = u;
                }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)
        next_sizes[sid] = (nxt_count < (int)max_frontier) ? nxt_count : (int)max_frontier;
}


__global__ void popcount_bitmap_kernel(
    const uint32_t* __restrict__ bitmaps,
    int32_t* __restrict__ popcounts,
    int32_t bitmap_words, int32_t n_sources, int32_t num_vertices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_sources * bitmap_words;
    if (idx >= total) return;
    int w = idx % bitmap_words;
    uint32_t word = bitmaps[idx];
    int base = w * 32;
    if (base + 32 > num_vertices) {
        int valid = num_vertices - base;
        if (valid > 0) word &= (1u << valid) - 1;
        else word = 0;
    }
    popcounts[idx] = __popc(word);
}


__global__ void extract_source_offsets_kernel(
    const int32_t* __restrict__ scan,
    const int32_t* __restrict__ counts,
    int64_t* __restrict__ ego_vert_offsets,
    int32_t bitmap_words, int32_t n_sources, int64_t total_words
) {
    int sid = threadIdx.x + blockIdx.x * blockDim.x;
    if (sid > n_sources) return;
    if (sid < n_sources) {
        ego_vert_offsets[sid] = scan[(int64_t)sid * bitmap_words];
    } else {
        
        ego_vert_offsets[n_sources] = scan[total_words - 1] + counts[total_words - 1];
    }
}


__global__ void extract_ego_vertices_kernel(
    const uint32_t* __restrict__ bitmaps,
    const int32_t* __restrict__ popcount_scan,
    int32_t* __restrict__ ego_vertices,
    int32_t bitmap_words, int32_t n_sources, int32_t num_vertices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_sources * bitmap_words;
    if (idx >= total) return;
    int w = idx % bitmap_words;
    uint32_t word = bitmaps[idx];
    int base = w * 32;
    if (base + 32 > num_vertices) {
        int valid = num_vertices - base;
        if (valid > 0) word &= (1u << valid) - 1;
        else word = 0;
    }
    if (word == 0) return;
    int write_pos = popcount_scan[idx];
    while (word) {
        int bit = __ffs(word) - 1;
        ego_vertices[write_pos++] = base + bit;
        word &= word - 1;
    }
}


__global__ void count_ego_edges_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const uint32_t* __restrict__ bitmaps,
    int32_t bitmap_words,
    const int32_t* __restrict__ ego_vertices,
    const int64_t* __restrict__ ego_vert_offsets,
    int32_t n_sources,
    int32_t* __restrict__ edge_counts,
    int64_t total_ego_verts
) {
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_ego_verts) return;

    
    int lo = 0, hi = n_sources;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (ego_vert_offsets[mid + 1] <= gid) lo = mid + 1;
        else hi = mid;
    }

    const uint32_t* bm = bitmaps + (int64_t)lo * bitmap_words;
    int32_t v = ego_vertices[gid];
    int32_t start = csr_offsets[v], end = csr_offsets[v + 1];
    int count = 0;
    for (int j = start; j < end; j++) {
        if (bitmap_test(bm, csr_indices[j])) count++;
    }
    edge_counts[gid] = count;
}


__global__ void extract_edge_offsets_kernel(
    const int64_t* __restrict__ edge_scan,
    const int32_t* __restrict__ edge_counts,
    const int64_t* __restrict__ ego_vert_offsets,
    int64_t* __restrict__ ego_edge_offsets,
    int32_t n_sources,
    int64_t total_ego_verts
) {
    int sid = threadIdx.x + blockIdx.x * blockDim.x;
    if (sid > n_sources) return;
    if (sid < n_sources) {
        ego_edge_offsets[sid] = edge_scan[ego_vert_offsets[sid]];
    } else {
        if (total_ego_verts > 0)
            ego_edge_offsets[n_sources] = edge_scan[total_ego_verts - 1] + edge_counts[total_ego_verts - 1];
        else
            ego_edge_offsets[n_sources] = 0;
    }
}


__global__ void extract_ego_edges_kernel(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const double* __restrict__ csr_weights,
    const uint32_t* __restrict__ bitmaps,
    int32_t bitmap_words,
    const int32_t* __restrict__ ego_vertices,
    const int64_t* __restrict__ ego_vert_offsets,
    const int64_t* __restrict__ edge_scan,
    const int64_t* __restrict__ ego_edge_offsets,
    int32_t n_sources,
    int32_t* __restrict__ out_srcs,
    int32_t* __restrict__ out_dsts,
    double* __restrict__ out_weights,
    int64_t total_ego_verts
) {
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_ego_verts) return;

    int lo = 0, hi = n_sources;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (ego_vert_offsets[mid + 1] <= gid) lo = mid + 1;
        else hi = mid;
    }

    const uint32_t* bm = bitmaps + (int64_t)lo * bitmap_words;
    int32_t v = ego_vertices[gid];
    int32_t start = csr_offsets[v], end = csr_offsets[v + 1];

    int64_t ego_base = ego_edge_offsets[lo];
    int64_t local_offset = edge_scan[gid] - edge_scan[ego_vert_offsets[lo]];
    int64_t write_pos = ego_base + local_offset;

    for (int j = start; j < end; j++) {
        int32_t u = csr_indices[j];
        if (bitmap_test(bm, u)) {
            out_srcs[write_pos] = v;
            out_dsts[write_pos] = u;
            out_weights[write_pos] = csr_weights[j];
            write_pos++;
        }
    }
}


__global__ void sort_weight_ties_kernel(
    const int32_t* __restrict__ srcs,
    const int32_t* __restrict__ dsts,
    double* __restrict__ weights,
    const int64_t* __restrict__ ego_offsets,
    int32_t n_sources,
    int64_t total_edges
) {
    int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_edges) return;

    int32_t my_src = srcs[gid], my_dst = dsts[gid];
    bool is_start = (gid == 0) || (srcs[gid-1] != my_src || dsts[gid-1] != my_dst);

    if (!is_start) {
        
        int lo = 0, hi = n_sources;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (ego_offsets[mid + 1] <= gid) lo = mid + 1;
            else hi = mid;
        }
        if (gid == ego_offsets[lo]) is_start = true;
    }
    if (!is_start) return;

    int lo = 0, hi = n_sources;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (ego_offsets[mid + 1] <= gid) lo = mid + 1;
        else hi = mid;
    }
    int64_t ego_end = ego_offsets[lo + 1];

    int64_t ge = gid + 1;
    while (ge < ego_end && srcs[ge] == my_src && dsts[ge] == my_dst) ge++;
    int gs = (int)(ge - gid);
    if (gs <= 1) return;

    for (int i = 1; i < gs; i++) {
        double key = weights[gid + i];
        int j = i - 1;
        while (j >= 0 && weights[gid + j] > key) {
            weights[gid + j + 1] = weights[gid + j]; j--;
        }
        weights[gid + j + 1] = key;
    }
}


__global__ void int32_to_int64_kernel(const int32_t* __restrict__ in, int64_t* __restrict__ out, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i];
}



static size_t get_scan_temp_int32(int64_t n) {
    size_t s = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, s, (int32_t*)nullptr, (int32_t*)nullptr, n);
    return s;
}

static void do_scan_int32(int32_t* in, int32_t* out, int64_t n, void* t, size_t ts) {
    cub::DeviceScan::ExclusiveSum(t, ts, in, out, n);
}

static size_t get_scan_temp_int64(int64_t n) {
    size_t s = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, s, (int64_t*)nullptr, (int64_t*)nullptr, n);
    return s;
}

static void do_scan_int64(int64_t* in, int64_t* out, int64_t n, void* t, size_t ts) {
    cub::DeviceScan::ExclusiveSum(t, ts, in, out, n);
}

}  

extract_ego_weighted_result_double_t extract_ego_weighted_f64_seg(
    const graph32_t& graph,
    const double* edge_weights,
    const int32_t* source_vertices,
    std::size_t n_sources,
    int32_t radius)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_csr_off = graph.offsets;
    const int32_t* d_csr_idx = graph.indices;
    const double* d_csr_wt = edge_weights;
    const int32_t* d_sources = source_vertices;
    int32_t ns = static_cast<int32_t>(n_sources);

    int32_t bw = (num_vertices + 31) / 32;
    int64_t max_frontier = (int64_t)num_vertices;

    
    uint32_t* d_bm;
    cudaMalloc(&d_bm, (int64_t)ns * bw * sizeof(uint32_t));

    int32_t *fa, *fb, *sa, *sb;
    cudaMalloc(&fa, (int64_t)ns * max_frontier * sizeof(int32_t));
    cudaMalloc(&fb, (int64_t)ns * max_frontier * sizeof(int32_t));
    cudaMalloc(&sa, (int64_t)ns * sizeof(int32_t));
    cudaMalloc(&sb, (int64_t)ns * sizeof(int32_t));

    bfs_init_kernel<<<ns, 256>>>(d_sources, d_bm, fa, sa, ns, bw, max_frontier);

    int32_t* cf = fa, *cs = sa;
    int32_t* nf = fb, *ns_buf = sb;

    for (int h = 0; h < radius; h++) {
        bfs_expand_kernel<<<ns, 256>>>(d_csr_off, d_csr_idx, d_bm, cf, cs, nf, ns_buf,
                                       ns, bw, max_frontier);
        std::swap(cf, nf);
        std::swap(cs, ns_buf);
    }

    
    cudaFree(fa);
    cudaFree(fb);
    cudaFree(sa);
    cudaFree(sb);

    
    int64_t total_words = (int64_t)ns * bw;

    int32_t* pc;
    cudaMalloc(&pc, total_words * sizeof(int32_t));
    popcount_bitmap_kernel<<<(int)((total_words + 255) / 256), 256>>>(d_bm, pc, bw, ns, num_vertices);

    int32_t* pc_scan;
    cudaMalloc(&pc_scan, total_words * sizeof(int32_t));
    cache.ensure_temp(get_scan_temp_int32(total_words));
    do_scan_int32(pc, pc_scan, total_words, cache.d_temp, cache.temp_size);

    int64_t* ego_vo;
    cudaMalloc(&ego_vo, (int64_t)(ns + 1) * sizeof(int64_t));
    {
        int threads = 256;
        int blocks = (ns + 1 + threads - 1) / threads;
        extract_source_offsets_kernel<<<blocks, threads>>>(pc_scan, pc, ego_vo, bw, ns, total_words);
    }

    int64_t total_ego_verts;
    cudaMemcpy(&total_ego_verts, ego_vo + ns, sizeof(int64_t), cudaMemcpyDeviceToHost);

    int32_t* ego_v;
    cudaMalloc(&ego_v, std::max(total_ego_verts, (int64_t)1) * sizeof(int32_t));
    {
        int total = ns * bw;
        extract_ego_vertices_kernel<<<(total + 255) / 256, 256>>>(d_bm, pc_scan, ego_v, bw, ns, num_vertices);
    }

    cudaFree(pc);
    cudaFree(pc_scan);

    
    int32_t* ec;
    cudaMalloc(&ec, std::max(total_ego_verts, (int64_t)1) * sizeof(int32_t));
    if (total_ego_verts > 0) {
        count_ego_edges_kernel<<<(int)((total_ego_verts + 255) / 256), 256>>>(
            d_csr_off, d_csr_idx, d_bm, bw, ego_v, ego_vo, ns, ec, total_ego_verts);
    }

    
    int64_t* ec64;
    cudaMalloc(&ec64, std::max(total_ego_verts, (int64_t)1) * sizeof(int64_t));
    if (total_ego_verts > 0) {
        int32_to_int64_kernel<<<(int)((total_ego_verts + 255) / 256), 256>>>(ec, ec64, total_ego_verts);
    }

    int64_t* escan;
    cudaMalloc(&escan, std::max(total_ego_verts, (int64_t)1) * sizeof(int64_t));
    if (total_ego_verts > 0) {
        cache.ensure_temp(get_scan_temp_int64(total_ego_verts));
        do_scan_int64(ec64, escan, total_ego_verts, cache.d_temp, cache.temp_size);
    }

    int64_t* eeo;
    cudaMalloc(&eeo, (int64_t)(ns + 1) * sizeof(int64_t));
    {
        int threads = 256;
        int blocks = (ns + 1 + threads - 1) / threads;
        extract_edge_offsets_kernel<<<blocks, threads>>>(escan, ec, ego_vo, eeo, ns, total_ego_verts);
    }

    int64_t total_edges;
    cudaMemcpy(&total_edges, eeo + ns, sizeof(int64_t), cudaMemcpyDeviceToHost);

    
    int32_t* out_s = nullptr;
    int32_t* out_d = nullptr;
    double* out_w = nullptr;

    if (total_edges > 0) {
        cudaMalloc(&out_s, total_edges * sizeof(int32_t));
        cudaMalloc(&out_d, total_edges * sizeof(int32_t));
        cudaMalloc(&out_w, total_edges * sizeof(double));

        if (total_ego_verts > 0) {
            extract_ego_edges_kernel<<<(int)((total_ego_verts + 255) / 256), 256>>>(
                d_csr_off, d_csr_idx, d_csr_wt, d_bm, bw, ego_v, ego_vo,
                escan, eeo, ns, out_s, out_d, out_w, total_ego_verts);

            sort_weight_ties_kernel<<<(int)((total_edges + 255) / 256), 256>>>(
                out_s, out_d, out_w, eeo, ns, total_edges);
        }
    }

    
    cudaFree(d_bm);
    cudaFree(ego_vo);
    cudaFree(ego_v);
    cudaFree(ec);
    cudaFree(ec64);
    cudaFree(escan);

    return extract_ego_weighted_result_double_t{
        out_s,
        out_d,
        out_w,
        reinterpret_cast<std::size_t*>(eeo),
        static_cast<std::size_t>(total_edges),
        static_cast<std::size_t>(ns + 1)
    };
}

}  
