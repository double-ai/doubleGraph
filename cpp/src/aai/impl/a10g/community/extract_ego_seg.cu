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
#include <cstddef>

namespace aai {

namespace {

template<typename T>
static void ensure(T*& ptr, int64_t& cap, int64_t needed) {
    if (cap < needed) {
        if (ptr) cudaFree(ptr);
        cudaMalloc(reinterpret_cast<void**>(&ptr), static_cast<std::size_t>(needed) * sizeof(T));
        cap = needed;
    }
}

struct Cache : Cacheable {
    uint32_t* bmp = nullptr;        int64_t bmp_cap = 0;
    uint32_t* old_bmp = nullptr;    int64_t old_bmp_cap = 0;
    uint32_t* frontier = nullptr;   int64_t frontier_cap = 0;
    int32_t* vert_counts = nullptr; int64_t vert_counts_cap = 0;
    int32_t* vert_offsets = nullptr;int64_t vert_offsets_cap = 0;
    void* ps32_temp = nullptr;      std::size_t ps32_temp_cap = 0;
    int32_t* ego_verts = nullptr;   int64_t ego_verts_cap = 0;
    int64_t* edge_counts = nullptr; int64_t edge_counts_cap = 0;
    int64_t* edge_prefix = nullptr; int64_t edge_prefix_cap = 0;
    void* ps64_temp = nullptr;      std::size_t ps64_temp_cap = 0;

    ~Cache() override {
        if (bmp) cudaFree(bmp);
        if (old_bmp) cudaFree(old_bmp);
        if (frontier) cudaFree(frontier);
        if (vert_counts) cudaFree(vert_counts);
        if (vert_offsets) cudaFree(vert_offsets);
        if (ps32_temp) cudaFree(ps32_temp);
        if (ego_verts) cudaFree(ego_verts);
        if (edge_counts) cudaFree(edge_counts);
        if (edge_prefix) cudaFree(edge_prefix);
        if (ps64_temp) cudaFree(ps64_temp);
    }
};



__device__ __forceinline__ void bitmap_set(uint32_t* bmp, int32_t v) {
    atomicOr(&bmp[v >> 5], 1u << (v & 31));
}
__device__ __forceinline__ bool bitmap_test_ldg(const uint32_t* bmp, int32_t v) {
    return (__ldg(&bmp[v >> 5]) >> (v & 31)) & 1;
}



__global__ void mark_hop01_kernel(
    const int32_t* __restrict__ csr_off, const int32_t* __restrict__ csr_idx,
    const int32_t* __restrict__ sources, uint32_t* bitmaps,
    int32_t n_sources, int64_t bmp_words)
{
    int32_t sid = blockIdx.x;
    if (sid >= n_sources) return;
    int32_t src = __ldg(&sources[sid]);
    uint32_t* bmp = bitmaps + (int64_t)sid * bmp_words;
    if (threadIdx.x == 0) bitmap_set(bmp, src);
    int32_t s = __ldg(&csr_off[src]), e = __ldg(&csr_off[src + 1]);
    for (int32_t i = s + threadIdx.x; i < e; i += blockDim.x)
        bitmap_set(bmp, __ldg(&csr_idx[i]));
}

__global__ void mark_hop2_kernel(
    const int32_t* __restrict__ csr_off, const int32_t* __restrict__ csr_idx,
    const int32_t* __restrict__ sources, uint32_t* bitmaps,
    int32_t n_sources, int64_t bmp_words)
{
    int32_t sid = blockIdx.y;
    if (sid >= n_sources) return;
    int32_t src = __ldg(&sources[sid]);
    uint32_t* bmp = bitmaps + (int64_t)sid * bmp_words;
    int32_t start = __ldg(&csr_off[src]), end = __ldg(&csr_off[src + 1]);
    int32_t degree = end - start;
    int32_t gw = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int32_t lane = threadIdx.x & 31;
    int32_t tw = (gridDim.x * blockDim.x) / 32;
    for (int32_t i = gw; i < degree; i += tw) {
        int32_t v = __ldg(&csr_idx[start + i]);
        int32_t vs = __ldg(&csr_off[v]), ve = __ldg(&csr_off[v + 1]);
        for (int32_t j = vs + lane; j < ve; j += 32)
            bitmap_set(bmp, __ldg(&csr_idx[j]));
    }
}

__global__ void save_bitmap_kernel(const uint32_t* __restrict__ s, uint32_t* d, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = s[i];
}
__global__ void compute_frontier_kernel(const uint32_t* __restrict__ e, const uint32_t* __restrict__ o, uint32_t* f, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) f[i] = e[i] & ~o[i];
}
__global__ void expand_frontier_kernel(
    const int32_t* __restrict__ csr_off, const int32_t* __restrict__ csr_idx,
    const uint32_t* __restrict__ fr, uint32_t* eg,
    int32_t n_sources, int32_t nv, int64_t bmp_words) {
    int32_t sid = blockIdx.y;
    if (sid >= n_sources) return;
    const uint32_t* f = fr + (int64_t)sid * bmp_words;
    uint32_t* e = eg + (int64_t)sid * bmp_words;
    int64_t wi = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (wi >= bmp_words) return;
    uint32_t w = f[wi];
    while (w) { int b=__ffs(w)-1; w&=w-1;
        int32_t v=(int32_t)(wi*32+b); if(v>=nv) break;
        int32_t vs=__ldg(&csr_off[v]),ve=__ldg(&csr_off[v+1]);
        for(int j=vs;j<ve;j++) bitmap_set(e, __ldg(&csr_idx[j]));
    }
}



__global__ void count_and_collect_ego_verts_kernel(
    const uint32_t* __restrict__ bitmaps,
    int32_t* vert_counts,
    const int32_t* __restrict__ vert_offsets,
    int32_t* ego_verts,
    int32_t n_sources, int32_t num_vertices, int64_t bmp_words,
    bool collect_mode)
{
    int32_t sid = blockIdx.x;
    if (sid >= n_sources) return;
    const uint32_t* bmp = bitmaps + (int64_t)sid * bmp_words;

    if (!collect_mode) {
        
        int count = 0;
        for (int64_t w = threadIdx.x; w < bmp_words; w += blockDim.x) {
            uint32_t word = bmp[w];
            if (word) count += __popc(word);
        }
        typedef cub::BlockReduce<int, 256> BR;
        __shared__ typename BR::TempStorage tmp;
        int total = BR(tmp).Sum(count);
        if (threadIdx.x == 0) vert_counts[sid] = total;
    } else {
        
        int32_t base = vert_offsets[sid];
        typedef cub::BlockScan<int, 256> BScan;
        __shared__ typename BScan::TempStorage scan_tmp;
        __shared__ int running_offset;
        if (threadIdx.x == 0) running_offset = 0;
        __syncthreads();

        for (int64_t batch = 0; batch < bmp_words; batch += blockDim.x) {
            int64_t word_idx = batch + threadIdx.x;
            uint32_t word = (word_idx < bmp_words) ? bmp[word_idx] : 0;
            int pc = __popc(word);
            int prefix, total;
            BScan(scan_tmp).ExclusiveSum(pc, prefix, total);
            __syncthreads();
            int my_offset = running_offset + prefix;
            while (word) {
                int bit = __ffs(word) - 1;
                word &= word - 1;
                int32_t v = (int32_t)(word_idx * 32 + bit);
                if (v < num_vertices) {
                    ego_verts[base + my_offset] = v;
                    my_offset++;
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) running_offset += total;
            __syncthreads();
        }
    }
}



__global__ __launch_bounds__(256, 6)
void count_edges_kernel(
    const int32_t* __restrict__ csr_off, const int32_t* __restrict__ csr_idx,
    const uint32_t* __restrict__ bitmaps,
    const int32_t* __restrict__ ego_verts,
    const int32_t* __restrict__ vert_offsets,
    int64_t* edge_counts,
    int32_t n_sources, int32_t total_ego_verts, int64_t bmp_words)
{
    extern __shared__ int32_t s_vo[];
    for (int i = threadIdx.x; i <= n_sources; i += blockDim.x)
        s_vo[i] = vert_offsets[i];
    __syncthreads();

    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int32_t lane = threadIdx.x & 31;
    if (warp_id >= total_ego_verts) return;

    int32_t sid = 0;
    { int lo=0,hi=n_sources; while(lo<hi){int m=(lo+hi)/2; if(s_vo[m+1]<=warp_id) lo=m+1; else hi=m;} sid=lo; }

    const uint32_t* bmp = bitmaps + (int64_t)sid * bmp_words;
    int32_t v = __ldg(&ego_verts[warp_id]);
    int32_t vs = __ldg(&csr_off[v]), ve = __ldg(&csr_off[v + 1]);

    int count = 0;
    for (int32_t j = vs + lane; j < ve; j += 32)
        if (bitmap_test_ldg(bmp, __ldg(&csr_idx[j]))) count++;
    for (int o = 16; o > 0; o >>= 1)
        count += __shfl_down_sync(0xffffffff, count, o);
    if (lane == 0) edge_counts[warp_id] = count;
}



__global__ __launch_bounds__(256, 6)
void write_edges_kernel(
    const int32_t* __restrict__ csr_off, const int32_t* __restrict__ csr_idx,
    const uint32_t* __restrict__ bitmaps,
    const int32_t* __restrict__ ego_verts,
    const int32_t* __restrict__ vert_offsets,
    const int64_t* __restrict__ edge_prefix,
    int32_t* __restrict__ edge_srcs, int32_t* __restrict__ edge_dsts,
    int32_t n_sources, int32_t total_ego_verts, int64_t bmp_words)
{
    extern __shared__ int32_t s_vo[];
    for (int i = threadIdx.x; i <= n_sources; i += blockDim.x)
        s_vo[i] = vert_offsets[i];
    __syncthreads();

    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int32_t lane = threadIdx.x & 31;
    if (warp_id >= total_ego_verts) return;

    int32_t sid = 0;
    { int lo=0,hi=n_sources; while(lo<hi){int m=(lo+hi)/2; if(s_vo[m+1]<=warp_id) lo=m+1; else hi=m;} sid=lo; }

    const uint32_t* bmp = bitmaps + (int64_t)sid * bmp_words;
    int32_t v = __ldg(&ego_verts[warp_id]);
    int32_t vs = __ldg(&csr_off[v]), ve = __ldg(&csr_off[v + 1]);
    int32_t deg = ve - vs;
    int64_t pos = __ldg(&edge_prefix[warp_id]);

    if (deg <= 32) {
        
        int32_t j = vs + lane;
        bool active = j < ve;
        int32_t w = active ? __ldg(&csr_idx[j]) : 0;
        bool match = active && bitmap_test_ldg(bmp, w);
        unsigned ballot = __ballot_sync(0xffffffff, match);
        int32_t rank = __popc(ballot & ((1u << lane) - 1));
        if (match) {
            edge_srcs[pos + rank] = v;
            edge_dsts[pos + rank] = w;
        }
    } else {
        
        for (int32_t base = vs; base < ve; base += 32) {
            int32_t j = base + lane;
            bool active = j < ve;
            int32_t w = active ? __ldg(&csr_idx[j]) : 0;
            bool match = active && bitmap_test_ldg(bmp, w);
            unsigned ballot = __ballot_sync(0xffffffff, match);
            int32_t rank = __popc(ballot & ((1u << lane) - 1));
            if (match) {
                edge_srcs[pos + rank] = v;
                edge_dsts[pos + rank] = w;
            }
            pos += __popc(ballot);
        }
    }
}



__global__ void compute_source_offsets_kernel(
    const int64_t* __restrict__ edge_prefix,
    const int64_t* __restrict__ edge_counts,
    const int32_t* __restrict__ vert_offsets,
    int64_t* source_offsets,
    int32_t n_sources, int32_t total_ego_verts)
{
    int32_t sid = threadIdx.x + blockIdx.x * blockDim.x;
    if (sid <= n_sources) {
        if (sid < n_sources) {
            int32_t vi = vert_offsets[sid];
            source_offsets[sid] = (vi < total_ego_verts) ? edge_prefix[vi] :
                (total_ego_verts > 0 ? edge_prefix[total_ego_verts-1] + edge_counts[total_ego_verts-1] : 0);
        } else {
            source_offsets[n_sources] = (total_ego_verts > 0) ?
                edge_prefix[total_ego_verts-1] + edge_counts[total_ego_verts-1] : 0;
        }
    }
}

}  

extract_ego_result_t extract_ego_seg(const graph32_t& graph,
                                     const int32_t* source_vertices,
                                     std::size_t n_sources,
                                     int32_t radius) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t n_src = static_cast<int32_t>(n_sources);
    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;

    if (num_vertices == 0 || n_src == 0) {
        std::size_t* offsets = nullptr;
        cudaMalloc(&offsets, static_cast<std::size_t>(n_src + 1) * sizeof(std::size_t));
        cudaMemset(offsets, 0, static_cast<std::size_t>(n_src + 1) * sizeof(std::size_t));
        return {nullptr, nullptr, offsets, 0, static_cast<std::size_t>(n_src + 1)};
    }

    int64_t bmp_words = ((int64_t)num_vertices + 31) / 32;
    int64_t total_bmp = (int64_t)n_src * bmp_words;

    
    ensure(cache.bmp, cache.bmp_cap, total_bmp);
    cudaMemset(cache.bmp, 0, total_bmp * sizeof(uint32_t));

    
    mark_hop01_kernel<<<n_src, 256>>>(d_off, d_idx, source_vertices, cache.bmp, n_src, bmp_words);
    if (radius >= 2) {
        dim3 grid(128, n_src);
        mark_hop2_kernel<<<grid, 1024>>>(d_off, d_idx, source_vertices, cache.bmp, n_src, bmp_words);
    }

    if (radius > 2) {
        ensure(cache.old_bmp, cache.old_bmp_cap, total_bmp);
        ensure(cache.frontier, cache.frontier_cap, total_bmp);

        
        cudaMemset(cache.bmp, 0, total_bmp * sizeof(uint32_t));
        mark_hop01_kernel<<<n_src, 256>>>(d_off, d_idx, source_vertices, cache.bmp, n_src, bmp_words);
        cudaMemcpy(cache.old_bmp, cache.bmp, total_bmp * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
        {
            dim3 grid(128, n_src);
            mark_hop2_kernel<<<grid, 1024>>>(d_off, d_idx, source_vertices, cache.bmp, n_src, bmp_words);
        }

        cudaMemcpy(cache.frontier, cache.bmp, total_bmp * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

        
        for (int hop = 3; hop <= radius; hop++) {
            int64_t total = total_bmp;
            int blocks = (total + 255) / 256;
            save_bitmap_kernel<<<blocks, 256>>>(cache.bmp, cache.old_bmp, total);
            dim3 grid2((bmp_words + 255) / 256, n_src);
            expand_frontier_kernel<<<grid2, 256>>>(d_off, d_idx, cache.frontier, cache.bmp, n_src, num_vertices, bmp_words);
            compute_frontier_kernel<<<blocks, 256>>>(cache.bmp, cache.old_bmp, cache.frontier, total);
        }
    }

    
    ensure(cache.vert_counts, cache.vert_counts_cap, (int64_t)n_src);
    count_and_collect_ego_verts_kernel<<<n_src, 256>>>(cache.bmp, cache.vert_counts, nullptr, nullptr, n_src, num_vertices, bmp_words, false);

    
    ensure(cache.vert_offsets, cache.vert_offsets_cap, (int64_t)(n_src + 1));
    {
        std::size_t tmp_sz = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_sz, (int32_t*)nullptr, (int32_t*)nullptr, n_src);
        if (cache.ps32_temp_cap < tmp_sz) {
            if (cache.ps32_temp) cudaFree(cache.ps32_temp);
            cudaMalloc(&cache.ps32_temp, tmp_sz);
            cache.ps32_temp_cap = tmp_sz;
        }
        std::size_t ps32_sz = tmp_sz;
        cub::DeviceScan::ExclusiveSum(cache.ps32_temp, ps32_sz, cache.vert_counts, cache.vert_offsets, n_src);
    }

    
    int32_t total_ego_verts = 0;
    {
        int32_t last_offset = 0, last_count = 0;
        cudaMemcpy(&last_offset, cache.vert_offsets + n_src - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_count, cache.vert_counts + n_src - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
        total_ego_verts = last_offset + last_count;
    }

    if (total_ego_verts == 0) {
        std::size_t* offsets = nullptr;
        cudaMalloc(&offsets, static_cast<std::size_t>(n_src + 1) * sizeof(std::size_t));
        cudaMemset(offsets, 0, static_cast<std::size_t>(n_src + 1) * sizeof(std::size_t));
        return {nullptr, nullptr, offsets, 0, static_cast<std::size_t>(n_src + 1)};
    }

    
    cudaMemcpy(cache.vert_offsets + n_src, &total_ego_verts, sizeof(int32_t), cudaMemcpyHostToDevice);

    
    ensure(cache.ego_verts, cache.ego_verts_cap, (int64_t)total_ego_verts);
    count_and_collect_ego_verts_kernel<<<n_src, 256>>>(cache.bmp, nullptr, cache.vert_offsets, cache.ego_verts, n_src, num_vertices, bmp_words, true);

    
    ensure(cache.edge_counts, cache.edge_counts_cap, (int64_t)total_ego_verts);
    {
        int threads = 256;
        int blocks = (int)(((int64_t)total_ego_verts * 32 + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        std::size_t smem = static_cast<std::size_t>(n_src + 1) * sizeof(int32_t);
        count_edges_kernel<<<blocks, threads, smem>>>(d_off, d_idx, cache.bmp, cache.ego_verts, cache.vert_offsets, cache.edge_counts, n_src, total_ego_verts, bmp_words);
    }

    
    ensure(cache.edge_prefix, cache.edge_prefix_cap, (int64_t)total_ego_verts);
    {
        std::size_t tmp_sz = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_sz, (int64_t*)nullptr, (int64_t*)nullptr, total_ego_verts);
        if (cache.ps64_temp_cap < tmp_sz) {
            if (cache.ps64_temp) cudaFree(cache.ps64_temp);
            cudaMalloc(&cache.ps64_temp, tmp_sz);
            cache.ps64_temp_cap = tmp_sz;
        }
        std::size_t ps64_sz = tmp_sz;
        cub::DeviceScan::ExclusiveSum(cache.ps64_temp, ps64_sz, cache.edge_counts, cache.edge_prefix, total_ego_verts);
    }

    
    int64_t total_edges = 0;
    {
        int64_t last_p = 0, last_c = 0;
        cudaMemcpy(&last_p, cache.edge_prefix + total_ego_verts - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&last_c, cache.edge_counts + total_ego_verts - 1, sizeof(int64_t), cudaMemcpyDeviceToHost);
        total_edges = last_p + last_c;
    }

    
    int32_t* out_srcs = nullptr;
    int32_t* out_dsts = nullptr;
    std::size_t* out_offs = nullptr;

    if (total_edges > 0) {
        cudaMalloc(&out_srcs, static_cast<std::size_t>(total_edges) * sizeof(int32_t));
        cudaMalloc(&out_dsts, static_cast<std::size_t>(total_edges) * sizeof(int32_t));
    }
    cudaMalloc(&out_offs, static_cast<std::size_t>(n_src + 1) * sizeof(std::size_t));

    if (total_edges > 0) {
        
        int threads = 256;
        int blocks = (int)(((int64_t)total_ego_verts * 32 + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        std::size_t smem = static_cast<std::size_t>(n_src + 1) * sizeof(int32_t);
        write_edges_kernel<<<blocks, threads, smem>>>(d_off, d_idx, cache.bmp, cache.ego_verts, cache.vert_offsets, cache.edge_prefix, out_srcs, out_dsts, n_src, total_ego_verts, bmp_words);
    }

    
    {
        int threads = 256;
        int blocks = (n_src + 1 + threads - 1) / threads;
        compute_source_offsets_kernel<<<blocks, threads>>>(cache.edge_prefix, cache.edge_counts, cache.vert_offsets, reinterpret_cast<int64_t*>(out_offs), n_src, total_ego_verts);
    }

    return {out_srcs, out_dsts, out_offs, static_cast<std::size_t>(total_edges), static_cast<std::size_t>(n_src + 1)};
}

}  
