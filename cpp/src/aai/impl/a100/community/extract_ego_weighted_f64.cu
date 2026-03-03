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
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

namespace aai {

namespace {


__device__ __forceinline__ int bw(int v) { return v >> 5; }
__device__ __forceinline__ uint32_t bm(int v) { return 1u << (v & 31); }


__global__ void kernel_bfs_mark_source(
    uint32_t* __restrict__ bitmap,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ ego_verts,
    int32_t* __restrict__ ego_size,
    int32_t source
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        bitmap[bw(source)] |= bm(source);
        frontier[0] = source;
        ego_verts[0] = source;
        *ego_size = 1;
    }
}


__global__ void kernel_bfs_expand(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const int32_t* __restrict__ frontier,
    int frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_size,
    uint32_t* __restrict__ bitmap,
    int32_t* __restrict__ ego_verts,
    int32_t* __restrict__ ego_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    if (warp_id >= frontier_size) return;

    int v = frontier[warp_id];
    int start = csr_offsets[v];
    int end = csr_offsets[v + 1];

    for (int i = start + lane; i < end; i += 32) {
        int u = csr_indices[i];
        uint32_t old = atomicOr(&bitmap[bw(u)], bm(u));
        if (!(old & bm(u))) {
            int npos = atomicAdd(next_size, 1);
            next_frontier[npos] = u;
            int epos = atomicAdd(ego_size, 1);
            ego_verts[epos] = u;
        }
    }
}


__global__ void kernel_count_edges(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const int32_t* __restrict__ ego_verts,
    int ego_size,
    const uint32_t* __restrict__ bitmap,
    int32_t* __restrict__ counts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    if (warp_id >= ego_size) return;

    int v = ego_verts[warp_id];
    int start = csr_offsets[v];
    int end = csr_offsets[v + 1];

    int cnt = 0;
    for (int i = start + lane; i < end; i += 32) {
        int u = csr_indices[i];
        if (bitmap[bw(u)] & bm(u)) cnt++;
    }

    
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        cnt += __shfl_down_sync(0xffffffff, cnt, off);

    if (lane == 0) counts[warp_id] = cnt;
}


__global__ void kernel_extract_edges(
    const int32_t* __restrict__ csr_offsets,
    const int32_t* __restrict__ csr_indices,
    const double* __restrict__ csr_weights,
    const int32_t* __restrict__ ego_verts,
    int ego_size,
    const uint32_t* __restrict__ bitmap,
    const int32_t* __restrict__ write_off,
    int64_t global_off,
    int32_t* __restrict__ out_srcs,
    int32_t* __restrict__ out_dsts,
    double* __restrict__ out_wts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    if (warp_id >= ego_size) return;

    int v = ego_verts[warp_id];
    int start = csr_offsets[v];
    int end = csr_offsets[v + 1];
    int64_t wpos = global_off + write_off[warp_id];

    for (int base = start; base < end; base += 32) {
        int i = base + lane;
        bool valid = (i < end);
        int u = valid ? csr_indices[i] : 0;
        bool match = valid && (bitmap[bw(u)] & bm(u));

        uint32_t mask = __ballot_sync(0xffffffff, match);
        int my_off = __popc(mask & ((1u << lane) - 1));

        if (match) {
            int64_t pos = wpos + my_off;
            out_srcs[pos] = v;
            out_dsts[pos] = u;
            out_wts[pos] = csr_weights[i];
        }

        wpos += __popc(mask);
    }
}


__global__ void kernel_mark_bitmap(
    uint32_t* __restrict__ bitmap,
    const int32_t* __restrict__ vertices,
    int count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    int v = vertices[tid];
    atomicOr(&bitmap[bw(v)], bm(v));
}


__global__ void kernel_clear_bitmap(
    uint32_t* __restrict__ bitmap,
    const int32_t* __restrict__ vertices,
    int count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    int v = vertices[tid];
    atomicAnd(&bitmap[bw(v)], ~bm(v));
}

void launch_bfs_mark_source(uint32_t* bitmap, int32_t* frontier,
                             int32_t* ego_verts, int32_t* ego_size, int32_t source) {
    kernel_bfs_mark_source<<<1, 1>>>(bitmap, frontier, ego_verts, ego_size, source);
}

void launch_bfs_expand(const int32_t* offsets, const int32_t* indices,
                        const int32_t* frontier, int frontier_size,
                        int32_t* next_frontier, int32_t* next_size,
                        uint32_t* bitmap, int32_t* ego_verts, int32_t* ego_size) {
    if (frontier_size == 0) return;
    int threads = 256;
    int blocks = ((int64_t)frontier_size * 32 + threads - 1) / threads;
    kernel_bfs_expand<<<blocks, threads>>>(offsets, indices, frontier, frontier_size,
                                            next_frontier, next_size, bitmap, ego_verts, ego_size);
}

void launch_count_edges(const int32_t* offsets, const int32_t* indices,
                         const int32_t* ego_verts, int ego_size,
                         const uint32_t* bitmap, int32_t* counts) {
    if (ego_size == 0) return;
    int threads = 256;
    int blocks = ((int64_t)ego_size * 32 + threads - 1) / threads;
    kernel_count_edges<<<blocks, threads>>>(offsets, indices, ego_verts, ego_size, bitmap, counts);
}

void launch_extract_edges(const int32_t* offsets, const int32_t* indices,
                            const double* weights,
                            const int32_t* ego_verts, int ego_size,
                            const uint32_t* bitmap, const int32_t* write_off,
                            int64_t global_off,
                            int32_t* out_srcs, int32_t* out_dsts, double* out_wts) {
    if (ego_size == 0) return;
    int threads = 256;
    int blocks = ((int64_t)ego_size * 32 + threads - 1) / threads;
    kernel_extract_edges<<<blocks, threads>>>(offsets, indices, weights,
                                               ego_verts, ego_size, bitmap, write_off,
                                               global_off, out_srcs, out_dsts, out_wts);
}

void launch_mark_bitmap(uint32_t* bitmap, const int32_t* vertices, int count) {
    if (count == 0) return;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    kernel_mark_bitmap<<<blocks, threads>>>(bitmap, vertices, count);
}

void launch_clear_bitmap(uint32_t* bitmap, const int32_t* vertices, int count) {
    if (count == 0) return;
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    kernel_clear_bitmap<<<blocks, threads>>>(bitmap, vertices, count);
}

void sort_ego_edges(int32_t* srcs, int32_t* dsts, double* wts, int64_t count) {
    if (count <= 1) return;
    auto begin = thrust::make_zip_iterator(
        thrust::make_tuple(
            thrust::device_pointer_cast(srcs),
            thrust::device_pointer_cast(dsts),
            thrust::device_pointer_cast(wts)
        )
    );
    thrust::sort(thrust::device, begin, begin + count);
}

struct Cache : Cacheable {
    uint32_t* d_bitmap = nullptr;
    int32_t* d_frontier0 = nullptr;
    int32_t* d_frontier1 = nullptr;
    int32_t* d_ego_verts = nullptr;
    int32_t* d_next_frontier_size = nullptr;
    int32_t* d_ego_size_dev = nullptr;
    int32_t* d_edge_counts = nullptr;
    int32_t* d_write_offsets = nullptr;
    int32_t alloc_vertices = 0;

    void free_scratch() {
        if (d_bitmap) { cudaFree(d_bitmap); d_bitmap = nullptr; }
        if (d_frontier0) { cudaFree(d_frontier0); d_frontier0 = nullptr; }
        if (d_frontier1) { cudaFree(d_frontier1); d_frontier1 = nullptr; }
        if (d_ego_verts) { cudaFree(d_ego_verts); d_ego_verts = nullptr; }
        if (d_next_frontier_size) { cudaFree(d_next_frontier_size); d_next_frontier_size = nullptr; }
        if (d_ego_size_dev) { cudaFree(d_ego_size_dev); d_ego_size_dev = nullptr; }
        if (d_edge_counts) { cudaFree(d_edge_counts); d_edge_counts = nullptr; }
        if (d_write_offsets) { cudaFree(d_write_offsets); d_write_offsets = nullptr; }
        alloc_vertices = 0;
    }

    void ensure_scratch(int32_t nv) {
        if (nv <= alloc_vertices) return;
        free_scratch();
        alloc_vertices = nv;
        int32_t bitmap_words = (nv + 31) / 32;
        cudaMalloc(&d_bitmap, (size_t)bitmap_words * sizeof(uint32_t));
        cudaMemset(d_bitmap, 0, (size_t)bitmap_words * sizeof(uint32_t));
        cudaMalloc(&d_frontier0, (size_t)nv * sizeof(int32_t));
        cudaMalloc(&d_frontier1, (size_t)nv * sizeof(int32_t));
        cudaMalloc(&d_ego_verts, (size_t)nv * sizeof(int32_t));
        cudaMalloc(&d_next_frontier_size, sizeof(int32_t));
        cudaMalloc(&d_ego_size_dev, sizeof(int32_t));
        cudaMalloc(&d_edge_counts, (size_t)nv * sizeof(int32_t));
        cudaMalloc(&d_write_offsets, ((size_t)nv + 1) * sizeof(int32_t));
    }

    ~Cache() override {
        free_scratch();
    }
};

}  

extract_ego_weighted_result_double_t extract_ego_weighted_f64(
    const graph32_t& graph,
    const double* edge_weights,
    const int32_t* source_vertices,
    std::size_t n_sources,
    int32_t radius) {

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_csr_offsets = graph.offsets;
    const int32_t* d_csr_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;

    cache.ensure_scratch(num_vertices);

    int32_t* d_frontier[2] = {cache.d_frontier0, cache.d_frontier1};

    
    std::vector<int32_t> h_sources(n_sources);
    cudaMemcpy(h_sources.data(), source_vertices, n_sources * sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    struct EgoInfo {
        std::vector<int32_t> h_ego_verts;
        std::vector<int32_t> h_write_offsets;
        int32_t total_edges;
        int32_t ego_size;
    };
    std::vector<EgoInfo> egos(n_sources);

    for (std::size_t s = 0; s < n_sources; s++) {
        
        launch_bfs_mark_source(cache.d_bitmap, d_frontier[0], cache.d_ego_verts, cache.d_ego_size_dev, h_sources[s]);

        int cur = 0;
        int32_t frontier_size = 1;

        
        for (int hop = 0; hop < radius; hop++) {
            if (frontier_size == 0) break;
            int next = 1 - cur;
            int32_t zero = 0;
            cudaMemcpy(cache.d_next_frontier_size, &zero, sizeof(int32_t), cudaMemcpyHostToDevice);

            launch_bfs_expand(d_csr_offsets, d_csr_indices,
                              d_frontier[cur], frontier_size,
                              d_frontier[next], cache.d_next_frontier_size,
                              cache.d_bitmap, cache.d_ego_verts, cache.d_ego_size_dev);

            cudaMemcpy(&frontier_size, cache.d_next_frontier_size, sizeof(int32_t), cudaMemcpyDeviceToHost);
            cur = next;
        }

        
        int32_t ego_size;
        cudaMemcpy(&ego_size, cache.d_ego_size_dev, sizeof(int32_t), cudaMemcpyDeviceToHost);
        egos[s].ego_size = ego_size;

        if (ego_size > 0) {
            
            launch_count_edges(d_csr_offsets, d_csr_indices,
                               cache.d_ego_verts, ego_size, cache.d_bitmap, cache.d_edge_counts);

            
            egos[s].h_ego_verts.resize(ego_size);
            cudaMemcpy(egos[s].h_ego_verts.data(), cache.d_ego_verts,
                       ego_size * sizeof(int32_t), cudaMemcpyDeviceToHost);

            std::vector<int32_t> h_counts(ego_size);
            cudaMemcpy(h_counts.data(), cache.d_edge_counts,
                       ego_size * sizeof(int32_t), cudaMemcpyDeviceToHost);

            
            egos[s].h_write_offsets.resize(ego_size + 1);
            egos[s].h_write_offsets[0] = 0;
            for (int i = 0; i < ego_size; i++) {
                egos[s].h_write_offsets[i+1] = egos[s].h_write_offsets[i] + h_counts[i];
            }
            egos[s].total_edges = egos[s].h_write_offsets[ego_size];
        } else {
            egos[s].total_edges = 0;
        }

        
        if (ego_size > 0) {
            launch_clear_bitmap(cache.d_bitmap, cache.d_ego_verts, ego_size);
        }
    }

    
    int64_t total_edges = 0;
    std::vector<std::size_t> h_ego_offsets(n_sources + 1);
    h_ego_offsets[0] = 0;
    for (std::size_t s = 0; s < n_sources; s++) {
        total_edges += egos[s].total_edges;
        h_ego_offsets[s+1] = total_edges;
    }

    
    int32_t* out_srcs = nullptr;
    int32_t* out_dsts = nullptr;
    double* out_wts = nullptr;
    std::size_t* out_off = nullptr;

    if (total_edges > 0) {
        cudaMalloc(&out_srcs, total_edges * sizeof(int32_t));
        cudaMalloc(&out_dsts, total_edges * sizeof(int32_t));
        cudaMalloc(&out_wts, total_edges * sizeof(double));
    }
    cudaMalloc(&out_off, (n_sources + 1) * sizeof(std::size_t));
    cudaMemcpy(out_off, h_ego_offsets.data(),
               (n_sources + 1) * sizeof(std::size_t), cudaMemcpyHostToDevice);

    if (total_edges == 0) {
        return extract_ego_weighted_result_double_t{
            out_srcs, out_dsts, out_wts, out_off,
            static_cast<std::size_t>(total_edges),
            n_sources + 1
        };
    }

    
    for (std::size_t s = 0; s < n_sources; s++) {
        auto& info = egos[s];
        if (info.total_edges == 0) continue;
        int ego_size = info.ego_size;

        
        cudaMemcpy(cache.d_ego_verts, info.h_ego_verts.data(),
                   ego_size * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(cache.d_write_offsets, info.h_write_offsets.data(),
                   (ego_size + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);

        
        launch_mark_bitmap(cache.d_bitmap, cache.d_ego_verts, ego_size);

        
        launch_extract_edges(d_csr_offsets, d_csr_indices, edge_weights,
                              cache.d_ego_verts, ego_size, cache.d_bitmap, cache.d_write_offsets,
                              static_cast<int64_t>(h_ego_offsets[s]),
                              out_srcs, out_dsts, out_wts);

        
        sort_ego_edges(out_srcs + h_ego_offsets[s],
                      out_dsts + h_ego_offsets[s],
                      out_wts + h_ego_offsets[s],
                      info.total_edges);

        
        launch_clear_bitmap(cache.d_bitmap, cache.d_ego_verts, ego_size);
    }

    cudaDeviceSynchronize();

    return extract_ego_weighted_result_double_t{
        out_srcs, out_dsts, out_wts, out_off,
        static_cast<std::size_t>(total_edges),
        n_sources + 1
    };
}

}  
