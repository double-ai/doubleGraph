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

static const int MAX_CONCURRENT = 10;

struct Cache : Cacheable {
    int32_t* d_distances[MAX_CONCURRENT] = {};
    unsigned long long* d_sigma[MAX_CONCURRENT] = {};
    float* d_delta[MAX_CONCURRENT] = {};
    int32_t* d_frontier_buf[MAX_CONCURRENT] = {};

    int32_t* d_frontier_sizes = nullptr;
    bool* d_is_source = nullptr;
    int32_t* h_frontier_sizes = nullptr;

    cudaStream_t streams[MAX_CONCURRENT] = {};

    int32_t max_vertices = 0;
    int num_streams = 0;

    void ensure(int32_t n, int nc) {
        if (n <= max_vertices && nc <= num_streams) return;

        for (int i = 0; i < num_streams; i++) {
            if (d_distances[i]) cudaFree(d_distances[i]);
            if (d_sigma[i]) cudaFree(d_sigma[i]);
            if (d_delta[i]) cudaFree(d_delta[i]);
            if (d_frontier_buf[i]) cudaFree(d_frontier_buf[i]);
            if (streams[i]) cudaStreamDestroy(streams[i]);
            d_distances[i] = nullptr;
            d_sigma[i] = nullptr;
            d_delta[i] = nullptr;
            d_frontier_buf[i] = nullptr;
            streams[i] = nullptr;
        }
        if (d_frontier_sizes) cudaFree(d_frontier_sizes);
        if (d_is_source) cudaFree(d_is_source);
        if (h_frontier_sizes) cudaFreeHost(h_frontier_sizes);

        max_vertices = n;
        num_streams = nc;

        for (int i = 0; i < nc; i++) {
            cudaMalloc(&d_distances[i], (size_t)n * sizeof(int32_t));
            cudaMalloc(&d_sigma[i], (size_t)n * sizeof(unsigned long long));
            cudaMalloc(&d_delta[i], (size_t)n * sizeof(float));
            cudaMalloc(&d_frontier_buf[i], (size_t)n * sizeof(int32_t));
            cudaStreamCreate(&streams[i]);
        }
        cudaMalloc(&d_frontier_sizes, (size_t)nc * sizeof(int32_t));
        cudaMalloc(&d_is_source, (size_t)n * sizeof(bool));
        cudaMallocHost(&h_frontier_sizes, (size_t)nc * sizeof(int32_t));
    }

    ~Cache() override {
        for (int i = 0; i < num_streams; i++) {
            if (d_distances[i]) cudaFree(d_distances[i]);
            if (d_sigma[i]) cudaFree(d_sigma[i]);
            if (d_delta[i]) cudaFree(d_delta[i]);
            if (d_frontier_buf[i]) cudaFree(d_frontier_buf[i]);
            if (streams[i]) cudaStreamDestroy(streams[i]);
        }
        if (d_frontier_sizes) cudaFree(d_frontier_sizes);
        if (d_is_source) cudaFree(d_is_source);
        if (h_frontier_sizes) cudaFreeHost(h_frontier_sizes);
    }
};





__global__ void init_bfs_kernel(
    int32_t* __restrict__ distances,
    unsigned long long* __restrict__ sigma,
    float* __restrict__ delta,
    int32_t num_vertices,
    int32_t source
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices;
         v += blockDim.x * gridDim.x) {
        distances[v] = (v == source) ? 0 : -1;
        sigma[v] = (v == source) ? 1ULL : 0ULL;
        delta[v] = 0.0f;
    }
}

__global__ void bfs_expand_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ distances,
    unsigned long long* __restrict__ sigma,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t depth
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    const int32_t v = frontier[warp_id];
    const unsigned long long sigma_v = sigma[v];
    const int32_t start = __ldg(&offsets[v]);
    const int32_t end = __ldg(&offsets[v + 1]);
    const int32_t new_depth = depth + 1;

    for (int32_t i = start + lane; i < end; i += 32) {
        const int32_t w = __ldg(&indices[i]);
        const int32_t old_dist = atomicCAS(&distances[w], -1, new_depth);
        if (old_dist != -1 && old_dist != new_depth) continue;
        atomicAdd(&sigma[w], sigma_v);
        if (old_dist == -1) {
            const int32_t pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = w;
        }
    }
}

__global__ void backward_pass_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    const int32_t* __restrict__ distances,
    const unsigned long long* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ centrality,
    int32_t depth,
    int32_t source
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    const int32_t v = frontier[warp_id];
    const float sigma_v = __ull2float_rn(sigma[v]);
    const int32_t start = __ldg(&offsets[v]);
    const int32_t end = __ldg(&offsets[v + 1]);
    const int32_t next_depth = depth + 1;

    float partial = 0.0f;
    for (int32_t i = start + lane; i < end; i += 32) {
        const int32_t w = __ldg(&indices[i]);
        if (__ldg(&distances[w]) == next_depth) {
            partial += (sigma_v / __ull2float_rn(__ldg(&sigma[w]))) * (1.0f + delta[w]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
    }

    if (lane == 0) {
        delta[v] = partial;
        if (v != source) {
            atomicAdd(&centrality[v], partial);
        }
    }
}

__global__ void endpoints_add_kernel(
    float* __restrict__ centrality,
    const int32_t* __restrict__ distances,
    int32_t num_vertices,
    int32_t source
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices;
         v += blockDim.x * gridDim.x) {
        if (distances[v] >= 0 && v != source) {
            centrality[v] += 1.0f;
        }
    }
}

__global__ void add_scalar_kernel(float* __restrict__ arr, int32_t idx, float val) {
    arr[idx] += val;
}

__global__ void mark_sources_kernel(
    bool* __restrict__ is_source,
    const int32_t* __restrict__ samples,
    int32_t num_samples
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_samples) {
        is_source[samples[tid]] = true;
    }
}

__global__ void normalize_uniform_kernel(
    float* __restrict__ centrality,
    int32_t num_vertices,
    float inv_scale
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices;
         v += blockDim.x * gridDim.x) {
        centrality[v] *= inv_scale;
    }
}

__global__ void normalize_split_kernel(
    float* __restrict__ centrality,
    const bool* __restrict__ is_source,
    int32_t num_vertices,
    float inv_scale_nonsource,
    float inv_scale_source
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices;
         v += blockDim.x * gridDim.x) {
        float s = is_source[v] ? inv_scale_source : inv_scale_nonsource;
        centrality[v] *= s;
    }
}





static void launch_init_bfs(int32_t* distances, unsigned long long* sigma, float* delta,
                            int32_t num_vertices, int32_t source, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    if (grid > 512) grid = 512;
    init_bfs_kernel<<<grid, block, 0, stream>>>(distances, sigma, delta, num_vertices, source);
}

static void launch_bfs_expand(const int32_t* offsets, const int32_t* indices,
                               const int32_t* frontier, int32_t frontier_size,
                               int32_t* distances, unsigned long long* sigma,
                               int32_t* next_frontier, int32_t* next_frontier_size,
                               int32_t depth, cudaStream_t stream) {
    if (frontier_size <= 0) return;
    int threads = 256;
    int warps_per_block = threads / 32;
    int grid = (frontier_size + warps_per_block - 1) / warps_per_block;
    bfs_expand_warp_kernel<<<grid, threads, 0, stream>>>(
        offsets, indices, frontier, frontier_size,
        distances, sigma, next_frontier, next_frontier_size, depth);
}

static void launch_backward_pass(const int32_t* offsets, const int32_t* indices,
                                  const int32_t* frontier, int32_t frontier_size,
                                  const int32_t* distances, const unsigned long long* sigma,
                                  float* delta, float* centrality,
                                  int32_t depth, int32_t source, cudaStream_t stream) {
    if (frontier_size <= 0) return;
    int threads = 256;
    int warps_per_block = threads / 32;
    int grid = (frontier_size + warps_per_block - 1) / warps_per_block;
    backward_pass_warp_kernel<<<grid, threads, 0, stream>>>(
        offsets, indices, frontier, frontier_size,
        distances, sigma, delta, centrality, depth, source);
}

static void launch_endpoints_add(float* centrality, const int32_t* distances,
                                  int32_t num_vertices, int32_t source, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    if (grid > 512) grid = 512;
    endpoints_add_kernel<<<grid, block, 0, stream>>>(centrality, distances, num_vertices, source);
}

static void launch_add_scalar(float* arr, int32_t idx, float val, cudaStream_t stream) {
    add_scalar_kernel<<<1, 1, 0, stream>>>(arr, idx, val);
}

static void launch_mark_sources(bool* is_source, const int32_t* samples,
                                 int32_t num_samples, cudaStream_t stream) {
    int block = 256;
    int grid = (num_samples + block - 1) / block;
    mark_sources_kernel<<<grid, block, 0, stream>>>(is_source, samples, num_samples);
}

static void launch_normalize_uniform(float* centrality, int32_t num_vertices,
                                      float inv_scale, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    if (grid > 512) grid = 512;
    normalize_uniform_kernel<<<grid, block, 0, stream>>>(centrality, num_vertices, inv_scale);
}

static void launch_normalize_split(float* centrality, const bool* is_source,
                                    int32_t num_vertices, float inv_scale_nonsource,
                                    float inv_scale_source, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    if (grid > 512) grid = 512;
    normalize_split_kernel<<<grid, block, 0, stream>>>(
        centrality, is_source, num_vertices, inv_scale_nonsource, inv_scale_source);
}





static void process_source_batch(
    Cache& cache,
    const int32_t* d_offsets, const int32_t* d_indices,
    float* d_centrality,
    const int32_t* h_sources, int batch_size, int32_t n,
    bool include_endpoints
) {
    
    for (int i = 0; i < batch_size; i++) {
        launch_init_bfs(cache.d_distances[i], cache.d_sigma[i], cache.d_delta[i],
                        n, h_sources[i], cache.streams[i]);
        cudaMemcpyAsync(cache.d_frontier_buf[i], &h_sources[i], sizeof(int32_t),
                        cudaMemcpyHostToDevice, cache.streams[i]);
    }

    
    struct SourceState {
        std::vector<int32_t> level_starts;
        std::vector<int32_t> level_sizes;
        int32_t current_start;
        int32_t current_size;
        int32_t next_start;
        int32_t depth;
        bool active;
    };

    std::vector<SourceState> states(batch_size);
    for (int i = 0; i < batch_size; i++) {
        states[i].level_starts.push_back(0);
        states[i].level_sizes.push_back(1);
        states[i].current_start = 0;
        states[i].current_size = 1;
        states[i].next_start = 1;
        states[i].depth = 0;
        states[i].active = true;
    }

    int active_count = batch_size;

    
    while (active_count > 0) {
        for (int i = 0; i < batch_size; i++) {
            if (!states[i].active) continue;

            int32_t zero = 0;
            cudaMemcpyAsync(cache.d_frontier_sizes + i, &zero, sizeof(int32_t),
                            cudaMemcpyHostToDevice, cache.streams[i]);

            launch_bfs_expand(d_offsets, d_indices,
                              cache.d_frontier_buf[i] + states[i].current_start,
                              states[i].current_size,
                              cache.d_distances[i], cache.d_sigma[i],
                              cache.d_frontier_buf[i] + states[i].next_start,
                              cache.d_frontier_sizes + i,
                              states[i].depth, cache.streams[i]);

            cudaMemcpyAsync(&cache.h_frontier_sizes[i], cache.d_frontier_sizes + i,
                            sizeof(int32_t), cudaMemcpyDeviceToHost, cache.streams[i]);
        }

        for (int i = 0; i < batch_size; i++) {
            if (!states[i].active) continue;

            cudaStreamSynchronize(cache.streams[i]);
            int32_t next_size = cache.h_frontier_sizes[i];

            if (next_size > 0) {
                states[i].level_starts.push_back(states[i].next_start);
                states[i].level_sizes.push_back(next_size);
                states[i].current_start = states[i].next_start;
                states[i].current_size = next_size;
                states[i].next_start = states[i].current_start + states[i].current_size;
            } else {
                states[i].active = false;
                active_count--;
            }
            states[i].depth++;
        }
    }

    
    for (int i = 0; i < batch_size; i++) {
        int num_levels = (int)states[i].level_sizes.size();
        int32_t source = h_sources[i];

        for (int d = num_levels - 1; d >= 1; d--) {
            launch_backward_pass(d_offsets, d_indices,
                                 cache.d_frontier_buf[i] + states[i].level_starts[d],
                                 states[i].level_sizes[d],
                                 cache.d_distances[i], cache.d_sigma[i],
                                 cache.d_delta[i], d_centrality,
                                 d, source, cache.streams[i]);
        }

        if (include_endpoints) {
            launch_endpoints_add(d_centrality, cache.d_distances[i], n, source, cache.streams[i]);

            int32_t total_reachable = 0;
            for (int d = 0; d < num_levels; d++) {
                total_reachable += states[i].level_sizes[d];
            }
            total_reachable -= 1;

            if (total_reachable > 0) {
                launch_add_scalar(d_centrality, source, (float)total_reachable, cache.streams[i]);
            }
        }
    }

    
    for (int i = 0; i < batch_size; i++) {
        cudaStreamSynchronize(cache.streams[i]);
    }
}

}  

void betweenness_centrality_seg(const graph32_t& graph,
                                float* centralities,
                                bool normalized,
                                bool include_endpoints,
                                const int32_t* sample_vertices,
                                std::size_t num_samples) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n = graph.number_of_vertices;
    bool is_symmetric = graph.is_symmetric;
    int64_t k = (int64_t)num_samples;

    int nc = ((int)k < MAX_CONCURRENT) ? (int)k : MAX_CONCURRENT;
    cache.ensure(n, nc);

    cudaMemset(centralities, 0, (size_t)n * sizeof(float));

    
    std::vector<int32_t> h_samples(k);
    cudaMemcpy(h_samples.data(), sample_vertices, k * sizeof(int32_t), cudaMemcpyDeviceToHost);

    
    for (int64_t batch_start = 0; batch_start < k; batch_start += nc) {
        int batch_size = (int)((k - batch_start < nc) ? (k - batch_start) : nc);
        process_source_batch(cache, d_offsets, d_indices, centralities,
                             h_samples.data() + batch_start, batch_size, n,
                             include_endpoints);
    }

    
    int64_t adj_i = include_endpoints ? (int64_t)n : (int64_t)(n - 1);
    bool all_srcs = (k == adj_i) || include_endpoints;

    cudaStream_t stream = 0;

    if (all_srcs) {
        double scale;
        if (normalized) {
            scale = (double)k * (double)(adj_i - 1);
        } else if (is_symmetric) {
            scale = (double)k * 2.0 / (double)adj_i;
        } else {
            scale = (double)k / (double)adj_i;
        }
        if (scale != 0.0) {
            launch_normalize_uniform(centralities, n, (float)(1.0 / scale), stream);
        }
    } else if (normalized) {
        double scale_ns = (double)k * (double)(adj_i - 1);
        double scale_s = (double)(k - 1) * (double)(adj_i - 1);

        cudaMemset(cache.d_is_source, 0, (size_t)n * sizeof(bool));
        launch_mark_sources(cache.d_is_source, sample_vertices, (int32_t)k, stream);
        launch_normalize_split(centralities, cache.d_is_source, n,
                               (float)(1.0 / scale_ns), (float)(1.0 / scale_s), stream);
    } else {
        double s_ns = (double)k / (double)adj_i;
        double s_s = (double)(k - 1) / (double)adj_i;
        if (is_symmetric) {
            s_ns *= 2.0;
            s_s *= 2.0;
        }

        cudaMemset(cache.d_is_source, 0, (size_t)n * sizeof(bool));
        launch_mark_sources(cache.d_is_source, sample_vertices, (int32_t)k, stream);
        launch_normalize_split(centralities, cache.d_is_source, n,
                               (float)(1.0 / s_ns), (float)(1.0 / s_s), stream);
    }

    cudaStreamSynchronize(stream);
}

}  
