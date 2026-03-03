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
#include <cfloat>
#include <cstdint>

namespace aai {

namespace {

static constexpr int WARP_THRESHOLD = 128;

struct Cache : Cacheable {
    int* pinned_size = nullptr;
    int* frontier_a = nullptr;
    int* frontier_b = nullptr;
    int* frontier_size_dev = nullptr;
    unsigned int* bitmap = nullptr;
    int32_t frontier_a_capacity = 0;
    int32_t frontier_b_capacity = 0;
    int32_t frontier_size_dev_capacity = 0;
    int32_t bitmap_capacity = 0;

    Cache() {
        cudaHostAlloc(&pinned_size, sizeof(int), cudaHostAllocDefault);
    }

    void ensure(int32_t num_vertices) {
        if (frontier_a_capacity < num_vertices) {
            if (frontier_a) cudaFree(frontier_a);
            cudaMalloc(&frontier_a, num_vertices * sizeof(int));
            frontier_a_capacity = num_vertices;
        }
        if (frontier_b_capacity < num_vertices) {
            if (frontier_b) cudaFree(frontier_b);
            cudaMalloc(&frontier_b, num_vertices * sizeof(int));
            frontier_b_capacity = num_vertices;
        }
        if (frontier_size_dev_capacity < 2) {
            if (frontier_size_dev) cudaFree(frontier_size_dev);
            cudaMalloc(&frontier_size_dev, 2 * sizeof(int));
            frontier_size_dev_capacity = 2;
        }
        int bitmap_words = (num_vertices + 31) / 32;
        if (bitmap_capacity < bitmap_words) {
            if (bitmap) cudaFree(bitmap);
            cudaMalloc(&bitmap, bitmap_words * sizeof(unsigned int));
            bitmap_capacity = bitmap_words;
        }
    }

    ~Cache() override {
        if (pinned_size) cudaFreeHost(pinned_size);
        if (frontier_a) cudaFree(frontier_a);
        if (frontier_b) cudaFree(frontier_b);
        if (frontier_size_dev) cudaFree(frontier_size_dev);
        if (bitmap) cudaFree(bitmap);
    }
};

__global__ void sssp_init_kernel(
    double* __restrict__ distances,
    int* __restrict__ predecessors,
    int num_vertices,
    int source
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int v = tid; v < num_vertices; v += stride) {
        distances[v] = (v == source) ? 0.0 : DBL_MAX;
        predecessors[v] = -1;
    }
}


__global__ void sssp_relax_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const double* __restrict__ weights,
    double* __restrict__ distances,
    const double cutoff,
    const int* __restrict__ frontier_in,
    const int frontier_in_size,
    int* __restrict__ frontier_out,
    int* __restrict__ frontier_out_size,
    unsigned int* __restrict__ bitmap
) {
    int lane_id = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int global_warp_id = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    int total_warps = gridDim.x * warps_per_block;

    for (int i = global_warp_id; i < frontier_in_size; i += total_warps) {
        int u = frontier_in[i];
        double dist_u = distances[u];

        if (dist_u >= cutoff) continue;

        int start = offsets[u];
        int end = offsets[u + 1];

        for (int e = start + lane_id; e < end; e += 32) {
            int v = __ldg(&indices[e]);
            double w = __ldg(&weights[e]);
            double new_dist = dist_u + w;

            if (new_dist >= cutoff) continue;

            
            if (new_dist >= distances[v]) continue;

            unsigned long long new_ull = (unsigned long long)__double_as_longlong(new_dist);
            unsigned long long old_ull = atomicMin(
                (unsigned long long*)&distances[v], new_ull);

            if (old_ull > new_ull) {
                int word_idx = v >> 5;
                unsigned int bit_mask = 1u << (v & 31);
                unsigned int old_word = atomicOr(&bitmap[word_idx], bit_mask);
                if (!(old_word & bit_mask)) {
                    int pos = atomicAdd(frontier_out_size, 1);
                    frontier_out[pos] = v;
                }
            }
        }
    }
}


__global__ void sssp_relax_small_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const double* __restrict__ weights,
    double* __restrict__ distances,
    const double cutoff,
    const int* __restrict__ frontier_in,
    const int frontier_in_size,
    int* __restrict__ frontier_out,
    int* __restrict__ frontier_out_size,
    unsigned int* __restrict__ bitmap
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < frontier_in_size; i += stride) {
        int u = frontier_in[i];
        double dist_u = distances[u];

        if (dist_u >= cutoff) continue;

        int start = offsets[u];
        int end = offsets[u + 1];

        for (int e = start; e < end; e++) {
            int v = __ldg(&indices[e]);
            double w = __ldg(&weights[e]);
            double new_dist = dist_u + w;

            if (new_dist >= cutoff) continue;
            if (new_dist >= distances[v]) continue;

            unsigned long long new_ull = (unsigned long long)__double_as_longlong(new_dist);
            unsigned long long old_ull = atomicMin(
                (unsigned long long*)&distances[v], new_ull);

            if (old_ull > new_ull) {
                int word_idx = v >> 5;
                unsigned int bit_mask = 1u << (v & 31);
                unsigned int old_word = atomicOr(&bitmap[word_idx], bit_mask);
                if (!(old_word & bit_mask)) {
                    int pos = atomicAdd(frontier_out_size, 1);
                    frontier_out[pos] = v;
                }
            }
        }
    }
}

__global__ void sssp_predecessors_positive_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ distances,
    int* __restrict__ predecessors,
    int num_vertices,
    int source
) {
    int lane_id = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    int global_warp = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    int total_warps = gridDim.x * warps_per_block;

    for (int u = global_warp; u < num_vertices; u += total_warps) {
        double dist_u = distances[u];
        if (dist_u >= DBL_MAX) continue;

        int start = offsets[u];
        int end = offsets[u + 1];

        for (int e = start + lane_id; e < end; e += 32) {
            int v = __ldg(&indices[e]);
            if (v == source) continue;
            double dist_v = distances[v];
            if (dist_v >= DBL_MAX) continue;

            double w = __ldg(&weights[e]);
            if (dist_u + w == dist_v && dist_u < dist_v) {
                predecessors[v] = u;
            }
        }
    }
}

__global__ void sssp_predecessors_zero_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ distances,
    int* __restrict__ predecessors,
    int num_vertices,
    int source,
    int* __restrict__ changed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int u = tid; u < num_vertices; u += stride) {
        if (u != source && predecessors[u] == -1) continue;
        double dist_u = distances[u];
        if (dist_u >= DBL_MAX) continue;

        int start = offsets[u];
        int end = offsets[u + 1];

        for (int e = start; e < end; e++) {
            int v = __ldg(&indices[e]);
            if (v == source) continue;
            double w = __ldg(&weights[e]);
            if (dist_u + w == distances[v] && distances[v] == dist_u && predecessors[v] == -1) {
                predecessors[v] = u;
                *changed = 1;
            }
        }
    }
}

void launch_sssp_init(double* d, int* p, int nv, int s) {
    int b = 256, g = (nv + b - 1) / b;
    if (g > 65535) g = 65535;
    sssp_init_kernel<<<g, b>>>(d, p, nv, s);
}

void launch_sssp_relax(
    const int* off, const int* idx, const double* wt,
    double* dist, double cutoff,
    const int* fin, int fsz, int* fout, int* fosz,
    unsigned int* bm, int use_warp
) {
    if (use_warp) {
        int b = 256, wpb = b / 32;
        int g = (fsz + wpb - 1) / wpb;
        if (g < 1) g = 1;
        if (g > 8192) g = 8192;
        sssp_relax_kernel<<<g, b>>>(off, idx, wt, dist, cutoff, fin, fsz, fout, fosz, bm);
    } else {
        int b = 256;
        int g = (fsz + b - 1) / b;
        if (g < 1) g = 1;
        if (g > 4096) g = 4096;
        sssp_relax_small_kernel<<<g, b>>>(off, idx, wt, dist, cutoff, fin, fsz, fout, fosz, bm);
    }
}

void launch_sssp_predecessors_positive(
    const int* off, const int* idx, const double* wt,
    const double* dist, int* pred, int nv, int src
) {
    int b = 256, wpb = b / 32;
    int g = (nv + wpb - 1) / wpb;
    if (g > 65535) g = 65535;
    sssp_predecessors_positive_kernel<<<g, b>>>(off, idx, wt, dist, pred, nv, src);
}

void launch_sssp_predecessors_zero(
    const int* off, const int* idx, const double* wt,
    const double* dist, int* pred, int nv, int src, int* changed
) {
    int b = 256;
    int g = (nv + b - 1) / b;
    if (g > 65535) g = 65535;
    sssp_predecessors_zero_kernel<<<g, b>>>(off, idx, wt, dist, pred, nv, src, changed);
}

}  

void sssp_seg(const graph32_t& graph,
              const double* edge_weights,
              int32_t source,
              double* distances,
              int32_t* predecessors,
              double cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    cache.ensure(num_vertices);

    int* fa = cache.frontier_a;
    int* fb = cache.frontier_b;
    int* d_fs = cache.frontier_size_dev;
    unsigned int* d_bm = cache.bitmap;
    int bitmap_words = (num_vertices + 31) / 32;

    launch_sssp_init(distances, predecessors, num_vertices, source);
    cudaMemcpy(fa, &source, sizeof(int), cudaMemcpyHostToDevice);

    int frontier_size = 1;
    int cur = 0;

    while (frontier_size > 0) {
        int next = 1 - cur;
        int* fin = (cur == 0) ? fa : fb;
        int* fout = (cur == 0) ? fb : fa;

        cudaMemsetAsync(&d_fs[next], 0, sizeof(int));
        cudaMemsetAsync(d_bm, 0, bitmap_words * sizeof(unsigned int));

        int use_warp = (frontier_size >= WARP_THRESHOLD) ? 1 : 0;
        launch_sssp_relax(offsets, indices, edge_weights, distances, cutoff,
            fin, frontier_size, fout, &d_fs[next], d_bm, use_warp);

        cudaMemcpy(cache.pinned_size, &d_fs[next], sizeof(int), cudaMemcpyDeviceToHost);
        frontier_size = *cache.pinned_size;
        cur = next;
    }

    
    launch_sssp_predecessors_positive(offsets, indices, edge_weights, distances, predecessors,
        num_vertices, source);

    
    {
        int h_changed = 1;
        for (int iter = 0; iter < num_vertices && h_changed; iter++) {
            cudaMemsetAsync(&d_fs[0], 0, sizeof(int));
            launch_sssp_predecessors_zero(offsets, indices, edge_weights, distances, predecessors,
                num_vertices, source, &d_fs[0]);
            cudaMemcpy(cache.pinned_size, &d_fs[0], sizeof(int), cudaMemcpyDeviceToHost);
            h_changed = *cache.pinned_size;
        }
    }

    cudaDeviceSynchronize();
}

}  
