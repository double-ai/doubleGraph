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

namespace aai {

namespace {

static constexpr int32_t BMAX = 32;
static constexpr int32_t SMALL_V_ATOMIC_THRESHOLD = 4096;

struct Cache : Cacheable {
    int32_t* d_dist2d = nullptr;
    float* d_sigma2d = nullptr;
    float* d_delta2d = nullptr;

    int32_t* d_fv0 = nullptr;
    int32_t* d_fv1 = nullptr;
    int16_t* d_fs0 = nullptr;
    int16_t* d_fs1 = nullptr;
    int32_t* d_frontier_size = nullptr;

    int32_t* d_all_fv = nullptr;
    int16_t* d_all_fs = nullptr;

    uint32_t* d_source_bitmap = nullptr;
    int32_t* d_reach_counts = nullptr;

    float* d_centralities = nullptr;

    int32_t* h_frontier_size = nullptr;

    int32_t alloc_V = 0;
    int32_t alloc_bitmap_words = 0;

    Cache() {
        cudaMallocHost(&h_frontier_size, sizeof(int32_t));
    }

    void ensure_alloc(int32_t V) {
        if (V <= alloc_V) return;
        
        auto f = [](auto*& p) {
            if (p) { cudaFree(p); p = nullptr; }
        };
        f(d_dist2d); f(d_sigma2d); f(d_delta2d);
        f(d_fv0); f(d_fv1); f(d_fs0); f(d_fs1);
        f(d_frontier_size);
        f(d_all_fv); f(d_all_fs);
        f(d_source_bitmap);
        f(d_reach_counts);
        f(d_centralities);
        alloc_V = V;
        alloc_bitmap_words = 0;

        int64_t total = (int64_t)BMAX * (int64_t)alloc_V;

        cudaMalloc(&d_dist2d, (size_t)total * sizeof(int32_t));
        cudaMalloc(&d_sigma2d, (size_t)total * sizeof(float));
        cudaMalloc(&d_delta2d, (size_t)total * sizeof(float));

        cudaMalloc(&d_fv0, (size_t)total * sizeof(int32_t));
        cudaMalloc(&d_fv1, (size_t)total * sizeof(int32_t));
        cudaMalloc(&d_fs0, (size_t)total * sizeof(int16_t));
        cudaMalloc(&d_fs1, (size_t)total * sizeof(int16_t));

        cudaMalloc(&d_frontier_size, sizeof(int32_t));

        cudaMalloc(&d_all_fv, (size_t)total * sizeof(int32_t));
        cudaMalloc(&d_all_fs, (size_t)total * sizeof(int16_t));

        cudaMalloc(&d_reach_counts, (size_t)BMAX * sizeof(int32_t));

        cudaMalloc(&d_centralities, (size_t)alloc_V * sizeof(float));

        int32_t bw = (alloc_V + 31) / 32;
        cudaMalloc(&d_source_bitmap, (size_t)bw * sizeof(uint32_t));
        alloc_bitmap_words = bw;
    }

    void ensure_bitmap(int32_t bw) {
        if (bw <= alloc_bitmap_words) return;
        if (d_source_bitmap) cudaFree(d_source_bitmap);
        cudaMalloc(&d_source_bitmap, (size_t)bw * sizeof(uint32_t));
        alloc_bitmap_words = bw;
    }

    ~Cache() override {
        auto f = [](auto*& p) {
            if (p) { cudaFree(p); p = nullptr; }
        };
        f(d_dist2d); f(d_sigma2d); f(d_delta2d);
        f(d_fv0); f(d_fv1); f(d_fs0); f(d_fs1);
        f(d_frontier_size);
        f(d_all_fv); f(d_all_fs);
        f(d_source_bitmap);
        f(d_reach_counts);
        f(d_centralities);
        if (h_frontier_size) { cudaFreeHost(h_frontier_size); h_frontier_size = nullptr; }
        alloc_V = 0;
        alloc_bitmap_words = 0;
    }
};





static __device__ __forceinline__ int lane_id() { return (int)(threadIdx.x & 31); }

template <typename T>
static __device__ __forceinline__ T ldg(const T* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
}

static __device__ __forceinline__ bool edge_active_ldg(const uint32_t* __restrict__ mask, int32_t e) {
    uint32_t w = ldg(mask + ((uint32_t)e >> 5));
    return (w >> (e & 31)) & 1u;
}

__global__ void init_2d_kernel(
    int32_t* __restrict__ dist2d,
    float* __restrict__ sigma2d,
    float* __restrict__ delta2d,
    int64_t total_elems
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = tid; i < total_elems; i += stride) {
        dist2d[i] = -1;
        sigma2d[i] = 0.0f;
        delta2d[i] = 0.0f;
    }
}

__global__ void init_sources_kernel(
    int32_t* __restrict__ dist2d,
    float* __restrict__ sigma2d,
    int32_t* __restrict__ frontier_v,
    int16_t* __restrict__ frontier_s,
    int32_t* __restrict__ reach_counts,
    const int32_t* __restrict__ sample_vertices,
    int32_t Kb,
    int32_t V
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= Kb) return;
    int32_t src = sample_vertices[i];
    int32_t base = i * V;
    dist2d[base + src] = 0;
    sigma2d[base + src] = 1.0f;
    frontier_v[i] = src;
    frontier_s[i] = (int16_t)i;
    reach_counts[i] = 1;
}

__global__ void clear_bitmap_kernel(uint32_t* __restrict__ bitmap, int32_t words) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = blockDim.x * gridDim.x;
    for (int j = i; j < words; j += stride) bitmap[j] = 0u;
}

__global__ void build_bitmap_kernel(const int32_t* __restrict__ samples, int64_t K, uint32_t* __restrict__ bitmap) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t j = i; j < K; j += stride) {
        int32_t v = samples[j];
        atomicOr(&bitmap[(uint32_t)v >> 5], 1u << (v & 31));
    }
}

__global__ void bfs_advance_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ dist2d,
    float* __restrict__ sigma2d,
    const int32_t* __restrict__ frontier_v,
    const int16_t* __restrict__ frontier_s,
    int32_t frontier_size,
    int32_t* __restrict__ next_frontier_v,
    int16_t* __restrict__ next_frontier_s,
    int32_t* __restrict__ next_frontier_size,
    int32_t* __restrict__ reach_counts,
    int32_t V,
    int32_t depth
) {
    int warp = (int)(((blockIdx.x * blockDim.x) + threadIdx.x) >> 5);
    int lane = lane_id();
    int warps_total = (int)((gridDim.x * blockDim.x) >> 5);

    const int32_t nd = depth + 1;

    for (int w = warp; w < frontier_size; w += warps_total) {
        int32_t u = ldg(frontier_v + w);
        int32_t s = (int32_t)ldg(frontier_s + w);
        int32_t base = s * V;
        float sigma_u = sigma2d[base + u];

        int32_t start = ldg(offsets + u);
        int32_t end = ldg(offsets + u + 1);

        for (int32_t e_base = start; e_base < end; e_base += 32) {
            int32_t e = e_base + lane;

            bool push = false;
            int32_t v = 0;

            if (e < end && edge_active_ldg(edge_mask, e)) {
                v = ldg(indices + e);
                int32_t* dist_ptr = &dist2d[base + v];
                int32_t old = *dist_ptr;
                if (old == -1) {
                    old = atomicCAS(dist_ptr, -1, nd);
                }
                if (old == -1 || old == nd) {
                    atomicAdd(&sigma2d[base + v], sigma_u);
                    push = (old == -1);
                }
            }

            unsigned int ballot = __ballot_sync(0xffffffffu, push ? 1 : 0);
            if (ballot) {
                int count = __popc(ballot);
                int warp_start = 0;
                if (lane == 0) {
                    warp_start = atomicAdd(next_frontier_size, count);
                    atomicAdd(&reach_counts[s], count);
                }
                warp_start = __shfl_sync(0xffffffffu, warp_start, 0);

                if (push) {
                    unsigned int lower = (lane == 0) ? 0u : ((1u << lane) - 1u);
                    int off = __popc(ballot & lower);
                    int pos = warp_start + off;
                    next_frontier_v[pos] = v;
                    next_frontier_s[pos] = (int16_t)s;
                }
            }
        }
    }
}

__global__ void backward_delta_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ dist2d,
    const float* __restrict__ sigma2d,
    float* __restrict__ delta2d,
    const int32_t* __restrict__ level_v,
    const int16_t* __restrict__ level_s,
    int32_t level_size,
    int32_t depth,
    int32_t V
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= level_size) return;

    int32_t v = level_v[tid];
    int32_t s = (int32_t)level_s[tid];
    int32_t base = s * V;

    float sigma_v = sigma2d[base + v];
    float accum = 0.0f;
    int32_t succ_depth = depth + 1;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    for (int32_t e = start; e < end; ++e) {
        if (!edge_active_ldg(edge_mask, e)) continue;
        int32_t w = indices[e];
        if (dist2d[base + w] == succ_depth) {
            accum += (1.0f + delta2d[base + w]) / sigma2d[base + w];
        }
    }

    delta2d[base + v] = sigma_v * accum;
}

__global__ void backward_atomic_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ dist2d,
    const float* __restrict__ sigma2d,
    float* __restrict__ delta2d,
    float* __restrict__ centralities,
    const int32_t* __restrict__ level_v,
    const int16_t* __restrict__ level_s,
    int32_t level_size,
    int32_t depth,
    const int32_t* __restrict__ batch_sources,
    int32_t V,
    bool include_endpoints
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= level_size) return;

    int32_t v = level_v[tid];
    int32_t s = (int32_t)level_s[tid];
    int32_t base = s * V;
    int32_t source_vertex = batch_sources[s];

    float sigma_v = sigma2d[base + v];
    float accum = 0.0f;
    int32_t succ_depth = depth + 1;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    for (int32_t e = start; e < end; ++e) {
        if (!edge_active_ldg(edge_mask, e)) continue;
        int32_t w = indices[e];
        if (dist2d[base + w] == succ_depth) {
            accum += (1.0f + delta2d[base + w]) / sigma2d[base + w];
        }
    }

    float delta_v = sigma_v * accum;
    delta2d[base + v] = delta_v;

    if (v != source_vertex) {
        float c = delta_v;
        if (include_endpoints) c += 1.0f;
        atomicAdd(&centralities[v], c);
    }
}

__global__ void accumulate_batch_kernel(
    float* __restrict__ centralities,
    const float* __restrict__ delta2d,
    const int32_t* __restrict__ dist2d,
    const int32_t* __restrict__ batch_sources,
    int32_t V,
    int32_t Kb,
    bool include_endpoints
) {
    __shared__ int32_t sh_src[BMAX];
    if (threadIdx.x < Kb) sh_src[threadIdx.x] = batch_sources[threadIdx.x];
    __syncthreads();

    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = blockDim.x * gridDim.x;

    for (int v = tid; v < V; v += stride) {
        float sum = 0.0f;
        #pragma unroll
        for (int s = 0; s < BMAX; ++s) {
            if (s >= Kb) break;
            if (v == sh_src[s]) continue;
            int32_t idx = s * V + v;
            float d = delta2d[idx];
            if (include_endpoints) {
                if (dist2d[idx] >= 0) sum += d + 1.0f;
            } else {
                sum += d;
            }
        }
        centralities[v] += sum;
    }
}

__global__ void add_endpoints_kernel(
    float* __restrict__ centralities,
    const int32_t* __restrict__ batch_sources,
    const int32_t* __restrict__ reach_counts,
    int32_t Kb
) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= Kb) return;
    int32_t src = batch_sources[i];
    int32_t count = reach_counts[i] - 1;
    if (count > 0) atomicAdd(&centralities[src], (float)count);
}

__global__ void normalize_kernel(
    float* __restrict__ centralities,
    const uint32_t* __restrict__ source_bitmap,
    int32_t V,
    float scale_nonsource,
    float scale_source
) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = blockDim.x * gridDim.x;
    for (int v = tid; v < V; v += stride) {
        bool is_src = (source_bitmap[(uint32_t)v >> 5] >> (v & 31)) & 1u;
        float s = is_src ? scale_source : scale_nonsource;
        centralities[v] /= s;
    }
}





void launch_init_2d(int32_t* dist2d, float* sigma2d, float* delta2d, int64_t total, cudaStream_t s) {
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    if (grid > 65535) grid = 65535;
    init_2d_kernel<<<grid, block, 0, s>>>(dist2d, sigma2d, delta2d, total);
}

void launch_init_sources(
    int32_t* dist2d, float* sigma2d,
    int32_t* fv, int16_t* fs, int32_t* reach_counts,
    const int32_t* batch_sources, int32_t Kb, int32_t V, cudaStream_t s
) {
    int block = 256;
    int grid = (Kb + block - 1) / block;
    init_sources_kernel<<<grid, block, 0, s>>>(dist2d, sigma2d, fv, fs, reach_counts, batch_sources, Kb, V);
}

void launch_clear_bitmap(uint32_t* bitmap, int32_t words, cudaStream_t s) {
    int block = 256;
    int grid = (words + block - 1) / block;
    if (grid > 4096) grid = 4096;
    clear_bitmap_kernel<<<grid, block, 0, s>>>(bitmap, words);
}

void launch_build_bitmap(const int32_t* samples, int64_t K, uint32_t* bitmap, cudaStream_t s) {
    int block = 256;
    int grid = (int)((K + block - 1) / block);
    if (grid > 4096) grid = 4096;
    build_bitmap_kernel<<<grid, block, 0, s>>>(samples, K, bitmap);
}

void launch_bfs_advance(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    int32_t* dist2d, float* sigma2d,
    const int32_t* fv, const int16_t* fs, int32_t fsize,
    int32_t* nfv, int16_t* nfs, int32_t* nfsize,
    int32_t* reach_counts,
    int32_t V, int32_t depth, cudaStream_t s
) {
    if (fsize == 0) return;
    int threads = 512;
    int warps_per_block = threads >> 5;
    int blocks = (fsize + warps_per_block - 1) / warps_per_block;
    if (blocks > 8192) blocks = 8192;
    bfs_advance_warp_kernel<<<blocks, threads, 0, s>>>(
        offsets, indices, edge_mask, dist2d, sigma2d,
        fv, fs, fsize, nfv, nfs, nfsize, reach_counts, V, depth);
}

void launch_backward_delta(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    const int32_t* dist2d, const float* sigma2d, float* delta2d,
    const int32_t* lv, const int16_t* ls, int32_t lsize,
    int32_t depth, int32_t V, cudaStream_t s
) {
    if (lsize == 0) return;
    int block = 256;
    int grid = (lsize + block - 1) / block;
    if (grid > 65535) grid = 65535;
    backward_delta_kernel<<<grid, block, 0, s>>>(
        offsets, indices, edge_mask, dist2d, sigma2d, delta2d,
        lv, ls, lsize, depth, V);
}

void launch_backward_atomic(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    const int32_t* dist2d, const float* sigma2d, float* delta2d, float* cent,
    const int32_t* lv, const int16_t* ls, int32_t lsize,
    int32_t depth, const int32_t* batch_sources, int32_t V, bool include_endpoints, cudaStream_t s
) {
    if (lsize == 0) return;
    int block = 256;
    int grid = (lsize + block - 1) / block;
    if (grid > 65535) grid = 65535;
    backward_atomic_kernel<<<grid, block, 0, s>>>(
        offsets, indices, edge_mask, dist2d, sigma2d, delta2d, cent,
        lv, ls, lsize, depth, batch_sources, V, include_endpoints);
}

void launch_accumulate_batch(float* cent, const float* delta2d, const int32_t* dist2d,
                             const int32_t* batch_sources, int32_t V, int32_t Kb,
                             bool include_endpoints, cudaStream_t s) {
    int block = 256;
    int grid = (V + block - 1) / block;
    if (grid > 65535) grid = 65535;
    accumulate_batch_kernel<<<grid, block, 0, s>>>(cent, delta2d, dist2d, batch_sources, V, Kb, include_endpoints);
}

void launch_add_endpoints(float* cent, const int32_t* batch_sources, const int32_t* reach_counts, int32_t Kb, cudaStream_t s) {
    int block = 256;
    int grid = (Kb + block - 1) / block;
    add_endpoints_kernel<<<grid, block, 0, s>>>(cent, batch_sources, reach_counts, Kb);
}

void launch_normalize(float* cent, const uint32_t* bitmap, int32_t V, float sn, float ss, cudaStream_t s) {
    int block = 256;
    int grid = (V + block - 1) / block;
    if (grid > 65535) grid = 65535;
    normalize_kernel<<<grid, block, 0, s>>>(cent, bitmap, V, sn, ss);
}

}  

void betweenness_centrality_mask(const graph32_t& graph,
                                  float* centralities,
                                  bool normalized,
                                  bool include_endpoints,
                                  const int32_t* sample_vertices,
                                  std::size_t num_samples) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t V = graph.number_of_vertices;
    bool is_symmetric = graph.is_symmetric;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_edge_mask = graph.edge_mask;
    const int32_t* d_samples = sample_vertices;
    int64_t K_total = (int64_t)num_samples;

    cache.ensure_alloc(V);

    cudaStream_t stream = 0;

    cudaMemsetAsync(cache.d_centralities, 0, (size_t)V * sizeof(float), stream);

    
    int32_t bw = (V + 31) / 32;
    cache.ensure_bitmap(bw);
    launch_clear_bitmap(cache.d_source_bitmap, bw, stream);
    launch_build_bitmap(d_samples, K_total, cache.d_source_bitmap, stream);

    const bool use_atomic_bc = (V <= SMALL_V_ATOMIC_THRESHOLD);

    int32_t* d_fv[2] = {cache.d_fv0, cache.d_fv1};
    int16_t* d_fs[2] = {cache.d_fs0, cache.d_fs1};

    for (int64_t b0 = 0; b0 < K_total; b0 += BMAX) {
        int32_t Kb = (int32_t)((K_total - b0) < BMAX ? (K_total - b0) : BMAX);
        const int32_t* batch_sources = d_samples + b0;

        int64_t total = (int64_t)BMAX * (int64_t)V;
        launch_init_2d(cache.d_dist2d, cache.d_sigma2d, cache.d_delta2d, total, stream);
        launch_init_sources(cache.d_dist2d, cache.d_sigma2d, d_fv[0], d_fs[0], cache.d_reach_counts,
                            batch_sources, Kb, V, stream);

        cudaMemcpyAsync(cache.d_all_fv, d_fv[0], (size_t)Kb * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(cache.d_all_fs, d_fs[0], (size_t)Kb * sizeof(int16_t), cudaMemcpyDeviceToDevice, stream);

        int32_t frontier_size = Kb;
        int32_t total_entries = Kb;
        int32_t depth = 0;
        int cur = 0;

        std::vector<int32_t> level_offsets;
        level_offsets.reserve(128);
        level_offsets.push_back(0);

        while (frontier_size > 0) {
            level_offsets.push_back(total_entries);

            cudaMemsetAsync(cache.d_frontier_size, 0, sizeof(int32_t), stream);
            launch_bfs_advance(d_offsets, d_indices, d_edge_mask,
                               cache.d_dist2d, cache.d_sigma2d,
                               d_fv[cur], d_fs[cur], frontier_size,
                               d_fv[1 - cur], d_fs[1 - cur], cache.d_frontier_size,
                               cache.d_reach_counts,
                               V, depth, stream);

            cudaMemcpyAsync(cache.h_frontier_size, cache.d_frontier_size, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            frontier_size = *cache.h_frontier_size;

            if (frontier_size > 0) {
                cudaMemcpyAsync(cache.d_all_fv + total_entries, d_fv[1 - cur],
                                (size_t)frontier_size * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(cache.d_all_fs + total_entries, d_fs[1 - cur],
                                (size_t)frontier_size * sizeof(int16_t), cudaMemcpyDeviceToDevice, stream);
                total_entries += frontier_size;
            }

            cur = 1 - cur;
            depth++;
        }

        if (include_endpoints) {
            launch_add_endpoints(cache.d_centralities, batch_sources, cache.d_reach_counts, Kb, stream);
        }

        int32_t num_levels = (int32_t)level_offsets.size() - 1;
        if (use_atomic_bc) {
            for (int32_t l = num_levels - 1; l >= 0; --l) {
                int32_t lstart = level_offsets[l];
                int32_t lend = level_offsets[l + 1];
                int32_t lsize = lend - lstart;
                launch_backward_atomic(d_offsets, d_indices, d_edge_mask,
                                       cache.d_dist2d, cache.d_sigma2d, cache.d_delta2d, cache.d_centralities,
                                       cache.d_all_fv + lstart, cache.d_all_fs + lstart, lsize,
                                       l, batch_sources, V, include_endpoints, stream);
            }
        } else {
            for (int32_t l = num_levels - 1; l >= 0; --l) {
                int32_t lstart = level_offsets[l];
                int32_t lend = level_offsets[l + 1];
                int32_t lsize = lend - lstart;
                launch_backward_delta(d_offsets, d_indices, d_edge_mask,
                                      cache.d_dist2d, cache.d_sigma2d, cache.d_delta2d,
                                      cache.d_all_fv + lstart, cache.d_all_fs + lstart, lsize,
                                      l, V, stream);
            }
            launch_accumulate_batch(cache.d_centralities, cache.d_delta2d, cache.d_dist2d, batch_sources, V, Kb, include_endpoints, stream);
        }
    }

    float adj = (float)V;
    if (!include_endpoints) adj -= 1.0f;
    float scale_nonsource = 1.0f;
    float scale_source = 1.0f;

    bool all_srcs = (include_endpoints || ((float)K_total == adj));

    if (all_srcs) {
        if (normalized) {
            scale_nonsource = (float)K_total * (adj - 1.0f);
        } else if (is_symmetric) {
            scale_nonsource = (float)K_total * 2.0f / adj;
        } else {
            scale_nonsource = (float)K_total / adj;
        }
        scale_source = scale_nonsource;
    } else if (normalized) {
        scale_nonsource = (float)K_total * (adj - 1.0f);
        scale_source = (float)(K_total - 1) * (adj - 1.0f);
    } else {
        scale_nonsource = (float)K_total / adj;
        scale_source = (float)(K_total - 1) / adj;
        if (is_symmetric) {
            scale_nonsource *= 2.0f;
            scale_source *= 2.0f;
        }
    }

    if (V > 1) {
        launch_normalize(cache.d_centralities, cache.d_source_bitmap, V, scale_nonsource, scale_source, stream);
    }

    
    cudaMemcpyAsync(centralities, cache.d_centralities, (size_t)V * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

}  
