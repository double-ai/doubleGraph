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
#include <cstddef>
#include <vector>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

namespace aai {

namespace {

struct Cache : Cacheable {
    uint32_t* d_bitmap = nullptr;
    int32_t* d_ego_verts = nullptr;
    int32_t* d_ego_size = nullptr;
    unsigned long long* d_counter = nullptr;
    int32_t* h_ego_size = nullptr;
    unsigned long long* h_counter = nullptr;
    int32_t allocated_vertices = 0;

    void ensure(int32_t num_vertices) {
        if (num_vertices > allocated_vertices) {
            if (d_bitmap) cudaFree(d_bitmap);
            if (d_ego_verts) cudaFree(d_ego_verts);
            if (d_ego_size) cudaFree(d_ego_size);
            if (d_counter) cudaFree(d_counter);
            if (h_ego_size) cudaFreeHost(h_ego_size);
            if (h_counter) cudaFreeHost(h_counter);

            int32_t bitmap_words = (num_vertices + 31) / 32;
            cudaMalloc(&d_bitmap, (size_t)bitmap_words * sizeof(uint32_t));
            cudaMalloc(&d_ego_verts, (size_t)num_vertices * sizeof(int32_t));
            cudaMalloc(&d_ego_size, sizeof(int32_t));
            cudaMalloc(&d_counter, sizeof(unsigned long long));
            cudaMemset(d_bitmap, 0, (size_t)bitmap_words * sizeof(uint32_t));
            cudaMallocHost(&h_ego_size, sizeof(int32_t));
            cudaMallocHost(&h_counter, sizeof(unsigned long long));

            allocated_vertices = num_vertices;
        }
    }

    ~Cache() override {
        if (d_bitmap) cudaFree(d_bitmap);
        if (d_ego_verts) cudaFree(d_ego_verts);
        if (d_ego_size) cudaFree(d_ego_size);
        if (d_counter) cudaFree(d_counter);
        if (h_ego_size) cudaFreeHost(h_ego_size);
        if (h_counter) cudaFreeHost(h_counter);
    }
};



__global__ void init_source_kernel(uint32_t* bitmap, int32_t* ego_verts, int32_t src, int32_t* ego_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t wi = (uint32_t)src >> 5;
        uint32_t bit = 1u << ((uint32_t)src & 31u);
        bitmap[wi] |= bit;
        ego_verts[0] = src;
        *ego_size = 1;
    }
}

__global__ void bfs_expand_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    uint32_t* bitmap,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ ego_verts,
    int32_t* __restrict__ ego_size
) {
    for (int fi = blockIdx.x; fi < frontier_size; fi += gridDim.x) {
        int32_t v = frontier[fi];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        for (int j = start + threadIdx.x; j < end; j += blockDim.x) {
            int32_t u = indices[j];
            uint32_t wi = (uint32_t)u >> 5;
            uint32_t bit = 1u << ((uint32_t)u & 31u);
            uint32_t old = atomicOr(&bitmap[wi], bit);
            if (!(old & bit)) {
                int pos = atomicAdd(ego_size, 1);
                ego_verts[pos] = u;
            }
        }
    }
}

__global__ void count_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ bitmap,
    const int32_t* __restrict__ ego_verts,
    int32_t n_ego,
    unsigned long long* __restrict__ total_count
) {
    unsigned long long local = 0;
    for (int vi = blockIdx.x; vi < n_ego; vi += gridDim.x) {
        int32_t v = ego_verts[vi];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        for (int j = start + threadIdx.x; j < end; j += blockDim.x) {
            int32_t u = indices[j];
            if (bitmap[(uint32_t)u >> 5] & (1u << ((uint32_t)u & 31u)))
                local++;
        }
    }
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(0xffffffff, local, off);
    if ((threadIdx.x & 31) == 0)
        atomicAdd(total_count, local);
}

__global__ void extract_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint32_t* __restrict__ bitmap,
    const int32_t* __restrict__ ego_verts,
    int32_t n_ego,
    int32_t* __restrict__ out_s,
    int32_t* __restrict__ out_d,
    float* __restrict__ out_w,
    unsigned long long* __restrict__ wpos
) {
    for (int vi = blockIdx.x; vi < n_ego; vi += gridDim.x) {
        int32_t v = ego_verts[vi];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        for (int j = start + threadIdx.x; j < end; j += blockDim.x) {
            int32_t u = indices[j];
            if (bitmap[(uint32_t)u >> 5] & (1u << ((uint32_t)u & 31u))) {
                unsigned long long p = atomicAdd(wpos, 1ULL);
                out_s[p] = v;
                out_d[p] = u;
                out_w[p] = weights[j];
            }
        }
    }
}

__global__ void clear_verts_kernel(uint32_t* bitmap, const int32_t* verts, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int32_t v = verts[i];
        atomicAnd(&bitmap[(uint32_t)v >> 5], ~(1u << ((uint32_t)v & 31u)));
    }
}

struct EdgeCmp {
    __host__ __device__
    bool operator()(const thrust::tuple<int32_t,int32_t,float>& a,
                    const thrust::tuple<int32_t,int32_t,float>& b) const {
        if (thrust::get<0>(a) != thrust::get<0>(b)) return thrust::get<0>(a) < thrust::get<0>(b);
        if (thrust::get<1>(a) != thrust::get<1>(b)) return thrust::get<1>(a) < thrust::get<1>(b);
        return thrust::get<2>(a) < thrust::get<2>(b);
    }
};



int32_t run_bfs(const int32_t* d_offsets, const int32_t* d_indices,
                int32_t source, int32_t radius, Cache& cache) {
    init_source_kernel<<<1, 1>>>(cache.d_bitmap, cache.d_ego_verts, source, cache.d_ego_size);

    int32_t frontier_start = 0;
    int32_t cur_ego_size = 1;

    for (int hop = 0; hop < radius; hop++) {
        int32_t frontier_size = cur_ego_size - frontier_start;
        if (frontier_size == 0) break;

        int grid = (frontier_size < 4096) ? frontier_size : 4096;
        bfs_expand_kernel<<<grid, 256>>>(d_offsets, d_indices, cache.d_bitmap,
            cache.d_ego_verts + frontier_start, frontier_size,
            cache.d_ego_verts, cache.d_ego_size);

        frontier_start = cur_ego_size;
        cudaMemcpyAsync(cache.h_ego_size, cache.d_ego_size, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, 0);
        cudaStreamSynchronize(0);
        cur_ego_size = *cache.h_ego_size;
    }
    return cur_ego_size;
}

}  

extract_ego_weighted_result_float_t extract_ego_weighted_f32(
    const graph32_t& graph,
    const float* edge_weights,
    const int32_t* source_vertices,
    std::size_t n_sources,
    int32_t radius) {

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;

    cache.ensure(num_vertices);

    
    std::vector<int32_t> h_sources(n_sources);
    if (n_sources > 0)
        cudaMemcpy(h_sources.data(), source_vertices, n_sources * sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    std::vector<int64_t> edge_counts(n_sources, 0);

    for (std::size_t s = 0; s < n_sources; s++) {
        int32_t ego_sz = run_bfs(d_offsets, d_indices, h_sources[s], radius, cache);

        cudaMemsetAsync(cache.d_counter, 0, sizeof(unsigned long long), 0);
        if (ego_sz > 0) {
            int grid = (ego_sz < 4096) ? ego_sz : 4096;
            count_edges_kernel<<<grid, 256>>>(d_offsets, d_indices, cache.d_bitmap,
                                              cache.d_ego_verts, ego_sz, cache.d_counter);
        }
        cudaMemcpyAsync(cache.h_counter, cache.d_counter, sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost, 0);
        cudaStreamSynchronize(0);
        edge_counts[s] = (int64_t)*cache.h_counter;

        
        if (ego_sz > 0)
            clear_verts_kernel<<<(ego_sz + 255) / 256, 256>>>(cache.d_bitmap, cache.d_ego_verts, ego_sz);
    }

    
    std::vector<std::size_t> h_offsets(n_sources + 1);
    h_offsets[0] = 0;
    for (std::size_t s = 0; s < n_sources; s++)
        h_offsets[s + 1] = h_offsets[s] + (std::size_t)edge_counts[s];
    std::size_t total = h_offsets[n_sources];

    extract_ego_weighted_result_float_t result;
    result.num_edges = total;
    result.num_offsets = n_sources + 1;

    if (total > 0) {
        cudaMalloc(&result.edge_srcs, total * sizeof(int32_t));
        cudaMalloc(&result.edge_dsts, total * sizeof(int32_t));
        cudaMalloc(&result.edge_weights, total * sizeof(float));
    } else {
        result.edge_srcs = nullptr;
        result.edge_dsts = nullptr;
        result.edge_weights = nullptr;
    }
    cudaMalloc(&result.offsets, (n_sources + 1) * sizeof(std::size_t));
    cudaMemcpy(result.offsets, h_offsets.data(), (n_sources + 1) * sizeof(std::size_t), cudaMemcpyHostToDevice);

    if (total == 0) return result;

    
    for (std::size_t s = 0; s < n_sources; s++) {
        if (edge_counts[s] == 0) continue;

        int32_t ego_sz = run_bfs(d_offsets, d_indices, h_sources[s], radius, cache);

        cudaMemsetAsync(cache.d_counter, 0, sizeof(unsigned long long), 0);
        int32_t* ps = result.edge_srcs + h_offsets[s];
        int32_t* pd = result.edge_dsts + h_offsets[s];
        float* pw = result.edge_weights + h_offsets[s];

        if (ego_sz > 0) {
            int grid = (ego_sz < 4096) ? ego_sz : 4096;
            extract_edges_kernel<<<grid, 256>>>(d_offsets, d_indices, edge_weights, cache.d_bitmap,
                                                cache.d_ego_verts, ego_sz, ps, pd, pw, cache.d_counter);
        }

        
        if (edge_counts[s] > 1) {
            auto z = thrust::make_zip_iterator(thrust::make_tuple(
                thrust::device_pointer_cast(ps),
                thrust::device_pointer_cast(pd),
                thrust::device_pointer_cast(pw)));
            thrust::sort(thrust::cuda::par, z, z + edge_counts[s], EdgeCmp());
        }

        
        if (ego_sz > 0)
            clear_verts_kernel<<<(ego_sz + 255) / 256, 256>>>(cache.d_bitmap, cache.d_ego_verts, ego_sz);
    }

    cudaDeviceSynchronize();
    return result;
}

}  
