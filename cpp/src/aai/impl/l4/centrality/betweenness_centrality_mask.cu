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

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* d_distances = nullptr;
    unsigned long long* d_sigma = nullptr;
    float* d_delta = nullptr;
    int32_t* d_frontier_0 = nullptr;
    int32_t* d_frontier_1 = nullptr;
    int32_t* d_frontier_stack = nullptr;
    int32_t* d_source_bitmap = nullptr;
    int32_t* d_frontier_size = nullptr;
    int32_t* h_frontier_size = nullptr;
    cudaStream_t stream = 0;
    int32_t alloc_vertices = 0;

    Cache() {
        cudaMalloc(&d_frontier_size, sizeof(int32_t));
        cudaHostAlloc(&h_frontier_size, sizeof(int32_t), cudaHostAllocDefault);
        cudaStreamCreate(&stream);
    }

    ~Cache() override {
        if (d_distances) cudaFree(d_distances);
        if (d_sigma) cudaFree(d_sigma);
        if (d_delta) cudaFree(d_delta);
        if (d_frontier_0) cudaFree(d_frontier_0);
        if (d_frontier_1) cudaFree(d_frontier_1);
        if (d_frontier_stack) cudaFree(d_frontier_stack);
        if (d_source_bitmap) cudaFree(d_source_bitmap);
        if (d_frontier_size) cudaFree(d_frontier_size);
        if (h_frontier_size) cudaFreeHost(h_frontier_size);
        if (stream) cudaStreamDestroy(stream);
    }

    void ensure_buffers(int32_t n) {
        if (n <= alloc_vertices) return;
        if (d_distances) cudaFree(d_distances);
        if (d_sigma) cudaFree(d_sigma);
        if (d_delta) cudaFree(d_delta);
        if (d_frontier_0) cudaFree(d_frontier_0);
        if (d_frontier_1) cudaFree(d_frontier_1);
        if (d_frontier_stack) cudaFree(d_frontier_stack);
        if (d_source_bitmap) cudaFree(d_source_bitmap);

        alloc_vertices = n;
        cudaMalloc(&d_distances, (size_t)n * sizeof(int32_t));
        cudaMalloc(&d_sigma, (size_t)n * sizeof(unsigned long long));
        cudaMalloc(&d_delta, (size_t)n * sizeof(float));
        cudaMalloc(&d_frontier_0, (size_t)n * sizeof(int32_t));
        cudaMalloc(&d_frontier_1, (size_t)n * sizeof(int32_t));
        cudaMalloc(&d_frontier_stack, (size_t)n * sizeof(int32_t));

        int32_t bw = (n + 31) / 32;
        cudaMalloc(&d_source_bitmap, (size_t)bw * sizeof(int32_t));
    }
};



__device__ __forceinline__ bool edge_active(const uint32_t* mask, int32_t idx) {
    return (mask[idx >> 5] >> (idx & 31)) & 1u;
}

__global__ void init_bfs_kernel(
    int32_t* __restrict__ distances,
    unsigned long long* __restrict__ sigma,
    float* __restrict__ delta,
    int32_t* __restrict__ frontier,
    int32_t* __restrict__ frontier_stack,
    int32_t num_vertices,
    int32_t source
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices) return;

    if (tid == source) {
        distances[tid] = 0;
        sigma[tid] = 1ULL;
        frontier[0] = source;
        frontier_stack[0] = source;
    } else {
        distances[tid] = -1;
        sigma[tid] = 0ULL;
    }
    delta[tid] = 0.0f;
}

__global__ void bfs_forward_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    unsigned long long* __restrict__ sigma,
    const int32_t* __restrict__ frontier,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t frontier_size,
    int32_t current_level
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;
    int32_t new_dist = current_level + 1;

    for (int w = warp_id; w < frontier_size; w += total_warps) {
        int32_t src = frontier[w];
        int32_t start = offsets[src];
        int32_t end = offsets[src + 1];
        unsigned long long src_sigma = sigma[src];

        for (int32_t e = start + lane; e < end; e += 32) {
            if (edge_active(edge_mask, e)) {
                int32_t dst = indices[e];
                int32_t old = atomicCAS(&distances[dst], -1, new_dist);
                if (old != -1 && old != new_dist) continue;
                atomicAdd(&sigma[dst], src_sigma);
                if (old == -1) {
                    int32_t pos = atomicAdd(next_frontier_size, 1);
                    next_frontier[pos] = dst;
                }
            }
        }
    }
}

__global__ void backward_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ distances,
    const unsigned long long* __restrict__ sigma,
    float* __restrict__ delta,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t current_level
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;
    int32_t child_level = current_level + 1;

    for (int w = warp_id; w < frontier_size; w += total_warps) {
        int32_t v = frontier[w];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];
        float delta_v = 0.0f;
        float sigma_v = (float)sigma[v];

        for (int32_t e = start + lane; e < end; e += 32) {
            if (edge_active(edge_mask, e)) {
                int32_t w2 = indices[e];
                if (distances[w2] == child_level) {
                    delta_v += (sigma_v / (float)sigma[w2]) * (1.0f + delta[w2]);
                }
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            delta_v += __shfl_down_sync(0xffffffff, delta_v, offset);

        if (lane == 0) delta[v] = delta_v;
    }
}

__global__ void accumulate_kernel(
    float* __restrict__ centrality,
    const float* __restrict__ delta,
    const int32_t* __restrict__ distances,
    int32_t num_vertices,
    int32_t source,
    bool include_endpoints
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertices || tid == source) return;

    if (distances[tid] >= 0) {
        float val = delta[tid];
        if (include_endpoints) val += 1.0f;
        centrality[tid] += val;
    }
}

__global__ void source_endpoint_kernel(
    float* __restrict__ centrality,
    const int32_t* __restrict__ distances,
    int32_t num_vertices,
    int32_t source
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int count = 0;

    for (int i = tid; i < num_vertices; i += stride) {
        if (i != source && distances[i] >= 0) count++;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        count += __shfl_down_sync(0xffffffff, count, offset);

    if ((threadIdx.x & 31) == 0 && count > 0)
        atomicAdd(&centrality[source], (float)count);
}

__global__ void scale_kernel(float* c, int32_t n, float s) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) c[tid] *= s;
}

__global__ void scale_sources_kernel(
    float* c, int32_t n, float ss, float sns,
    const int32_t* bm, int32_t bsz
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    bool is_src = (tid / 32 < bsz) && ((bm[tid >> 5] >> (tid & 31)) & 1);
    c[tid] *= is_src ? ss : sns;
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

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    bool is_symmetric = graph.is_symmetric;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cache.ensure_buffers(num_vertices);
    cudaStream_t stream = cache.stream;

    float* d_centrality = centralities;
    cudaMemsetAsync(d_centrality, 0, num_vertices * sizeof(float), stream);

    std::vector<int32_t> sources;
    bool using_samples = (num_samples > 0 && sample_vertices != nullptr);
    if (using_samples) {
        sources.resize(num_samples);
        cudaMemcpyAsync(sources.data(), sample_vertices,
                   num_samples * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    } else {
        sources.resize(num_vertices);
        for (int32_t i = 0; i < num_vertices; i++) sources[i] = i;
    }

    int32_t* d_frontier[2] = {cache.d_frontier_0, cache.d_frontier_1};

    for (size_t si = 0; si < sources.size(); si++) {
        int32_t source = sources[si];

        init_bfs_kernel<<<(num_vertices+255)/256, 256, 0, stream>>>(
            cache.d_distances, cache.d_sigma, cache.d_delta,
            d_frontier[0], cache.d_frontier_stack, num_vertices, source);

        int32_t frontier_size = 1;
        int cur = 0;
        int32_t current_level = 0;

        std::vector<int32_t> level_sizes;
        std::vector<int32_t> level_offsets;
        int32_t total_stored = 1;

        level_offsets.push_back(0);
        level_sizes.push_back(1);

        while (frontier_size > 0) {
            cudaMemsetAsync(cache.d_frontier_size, 0, sizeof(int32_t), stream);

            {
                int32_t fsize = frontier_size;
                if (fsize > 0) {
                    int warps_needed = fsize;
                    int threads = 256;
                    int warps_per_block = threads / 32;
                    int blocks = (warps_needed + warps_per_block - 1) / warps_per_block;
                    if (blocks > 1024) blocks = 1024;
                    if (blocks < 1) blocks = 1;
                    bfs_forward_kernel<<<blocks, threads, 0, stream>>>(
                        d_offsets, d_indices, d_edge_mask,
                        cache.d_distances, cache.d_sigma,
                        d_frontier[cur], d_frontier[1-cur], cache.d_frontier_size,
                        fsize, current_level);
                }
            }

            cudaMemcpyAsync(cache.h_frontier_size, cache.d_frontier_size, sizeof(int32_t),
                       cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            frontier_size = *cache.h_frontier_size;

            cur = 1 - cur;
            current_level++;

            if (frontier_size > 0 && total_stored + frontier_size <= cache.alloc_vertices) {
                cudaMemcpyAsync(cache.d_frontier_stack + total_stored,
                           d_frontier[cur], frontier_size * sizeof(int32_t),
                           cudaMemcpyDeviceToDevice, stream);
                level_offsets.push_back(total_stored);
                level_sizes.push_back(frontier_size);
                total_stored += frontier_size;
            }
        }

        int max_level = (int)level_sizes.size() - 1;

        for (int level = max_level; level >= 1; level--) {
            int32_t fsize = level_sizes[level];
            if (fsize > 0) {
                int warps_needed = fsize;
                int threads = 256;
                int warps_per_block = threads / 32;
                int blocks = (warps_needed + warps_per_block - 1) / warps_per_block;
                if (blocks > 1024) blocks = 1024;
                if (blocks < 1) blocks = 1;
                backward_kernel<<<blocks, threads, 0, stream>>>(
                    d_offsets, d_indices, d_edge_mask,
                    cache.d_distances, cache.d_sigma, cache.d_delta,
                    cache.d_frontier_stack + level_offsets[level],
                    fsize, level);
            }
        }

        accumulate_kernel<<<(num_vertices+255)/256, 256, 0, stream>>>(
            d_centrality, cache.d_delta, cache.d_distances,
            num_vertices, source, include_endpoints);

        if (include_endpoints) {
            source_endpoint_kernel<<<32, 256, 0, stream>>>(
                d_centrality, cache.d_distances,
                num_vertices, source);
        }
    }

    if (num_vertices >= 2) {
        double n = (double)num_vertices;
        double N = include_endpoints ? n : (n - 1.0);
        int32_t k = (int32_t)sources.size();

        if (N >= 2.0) {
            bool need_src_dist = using_samples && !include_endpoints;

            if (need_src_dist) {
                double corr = is_symmetric ? 2.0 : 1.0;
                double ss, sns;
                if (normalized) {
                    ss = 1.0 / ((double)(k-1) * (N-1.0));
                    sns = 1.0 / ((double)k * (N-1.0));
                } else {
                    ss = N / ((double)(k-1) * corr);
                    sns = N / ((double)k * corr);
                }

                int32_t bw = (num_vertices + 31) / 32;
                std::vector<int32_t> h_bm(bw, 0);
                for (int32_t s : sources)
                    if (s >= 0 && s < num_vertices)
                        h_bm[s >> 5] |= (1 << (s & 31));
                cudaMemcpyAsync(cache.d_source_bitmap, h_bm.data(),
                           bw * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
                scale_sources_kernel<<<(num_vertices+255)/256, 256, 0, stream>>>(
                    d_centrality, num_vertices,
                    (float)ss, (float)sns, cache.d_source_bitmap, bw);
            } else {
                double scale;
                if (normalized) {
                    double K = using_samples ? (double)k : N;
                    scale = 1.0 / (K * (N - 1.0));
                } else {
                    double corr = is_symmetric ? 2.0 : 1.0;
                    double K = using_samples ? (double)k : N;
                    scale = N / (K * corr);
                }
                if (scale != 1.0)
                    scale_kernel<<<(num_vertices+255)/256, 256, 0, stream>>>(
                        d_centrality, num_vertices, (float)scale);
            }
        }
    }

    cudaStreamSynchronize(stream);
}

}  
