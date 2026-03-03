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

#define WARP_SIZE 32
#define BS 256
#define WARP_THRESH 64

struct Cache : Cacheable {
    int32_t* dist = nullptr;
    int32_t* sigma = nullptr;
    float* delta = nullptr;
    int32_t* queue = nullptr;
    int32_t* counter = nullptr;
    int32_t vertex_capacity = 0;

    void ensure(int32_t nv) {
        if (vertex_capacity < nv) {
            if (dist) cudaFree(dist);
            if (sigma) cudaFree(sigma);
            if (delta) cudaFree(delta);
            if (queue) cudaFree(queue);
            cudaMalloc(&dist, (size_t)nv * sizeof(int32_t));
            cudaMalloc(&sigma, (size_t)nv * sizeof(int32_t));
            cudaMalloc(&delta, (size_t)nv * sizeof(float));
            cudaMalloc(&queue, (size_t)nv * sizeof(int32_t));
            vertex_capacity = nv;
        }
        if (!counter) {
            cudaMalloc(&counter, sizeof(int32_t));
        }
    }

    ~Cache() override {
        if (dist) cudaFree(dist);
        if (sigma) cudaFree(sigma);
        if (delta) cudaFree(delta);
        if (queue) cudaFree(queue);
        if (counter) cudaFree(counter);
    }
};



__global__ void __launch_bounds__(BS)
bfs_expand_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ dist,
    int32_t* __restrict__ sigma,
    const int32_t* __restrict__ frontier,
    int frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count,
    int depth
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= frontier_size) return;

    int v = frontier[warp_id];
    int sigma_v = sigma[v];
    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);
    int nd = depth + 1;

    for (int j = start + lane; j < end; j += 32) {
        int w = __ldg(&indices[j]);
        int old_d = atomicCAS(&dist[w], -1, nd);
        if (old_d == -1) {
            int pos = atomicAdd(next_count, 1);
            next_frontier[pos] = w;
        }
        if (old_d == -1 || old_d == nd) {
            atomicAdd(&sigma[w], sigma_v);
        }
    }
}

__global__ void __launch_bounds__(BS)
bfs_expand_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ dist,
    int32_t* __restrict__ sigma,
    const int32_t* __restrict__ frontier,
    int frontier_size,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_count,
    int depth
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;

    int v = frontier[tid];
    int sigma_v = sigma[v];
    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);
    int nd = depth + 1;

    for (int j = start; j < end; j++) {
        int w = __ldg(&indices[j]);
        int old_d = atomicCAS(&dist[w], -1, nd);
        if (old_d == -1) {
            int pos = atomicAdd(next_count, 1);
            next_frontier[pos] = w;
        }
        if (old_d == -1 || old_d == nd) {
            atomicAdd(&sigma[w], sigma_v);
        }
    }
}



__global__ void __launch_bounds__(BS)
backward_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ dist,
    const int32_t* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ edge_bc,
    const int32_t* __restrict__ level_verts,
    int level_size,
    int depth
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= level_size) return;

    int v = level_verts[warp_id];
    float sigma_v = __int2float_rn(sigma[v]);
    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);
    int nd = depth + 1;

    float dv = 0.0f;
    for (int j = start + lane; j < end; j += 32) {
        int w = __ldg(&indices[j]);
        if (__ldg(&dist[w]) == nd) {
            float c = sigma_v * delta[w];
            edge_bc[j] += c;
            dv += c;
        }
    }

    
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        dv += __shfl_down_sync(0xffffffff, dv, o);

    if (lane == 0)
        delta[v] = (1.0f + dv) / sigma_v;
}

__global__ void __launch_bounds__(BS)
backward_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ dist,
    const int32_t* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ edge_bc,
    const int32_t* __restrict__ level_verts,
    int level_size,
    int depth
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_size) return;

    int v = level_verts[tid];
    float sigma_v = __int2float_rn(sigma[v]);
    float dv = 0.0f;
    int start = __ldg(&offsets[v]);
    int end = __ldg(&offsets[v + 1]);
    int nd = depth + 1;

    for (int j = start; j < end; j++) {
        int w = __ldg(&indices[j]);
        if (__ldg(&dist[w]) == nd) {
            float c = sigma_v * delta[w];
            edge_bc[j] += c;
            dv += c;
        }
    }
    delta[v] = (1.0f + dv) / sigma_v;
}



__global__ void reset_kernel(
    int32_t* __restrict__ dist,
    int32_t* __restrict__ sigma,
    float* __restrict__ delta,
    const int32_t* __restrict__ verts,
    int count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    int v = verts[tid];
    dist[v] = -1;
    sigma[v] = 0;
    delta[v] = 0.0f;
}

__global__ void scale_kernel(float* __restrict__ data, int n, float factor) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    data[tid] *= factor;
}

__global__ void init_source(int32_t* dist, int32_t* sigma, int32_t* q, int32_t src) {
    dist[src] = 0;
    sigma[src] = 1;
    q[0] = src;
}

}  

void edge_betweenness_centrality(const graph32_t& graph,
                                 float* edge_centralities,
                                 bool normalized,
                                 const int32_t* sample_vertices,
                                 std::size_t num_samples) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    bool is_symmetric = graph.is_symmetric;

    cache.ensure(num_vertices);

    int32_t* dist = cache.dist;
    int32_t* sigma = cache.sigma;
    float* delta = cache.delta;
    int32_t* queue = cache.queue;
    int32_t* counter = cache.counter;

    
    cudaMemsetAsync(edge_centralities, 0, (size_t)num_edges * sizeof(float));
    cudaMemsetAsync(dist, 0xFF, (size_t)num_vertices * sizeof(int32_t));
    cudaMemsetAsync(sigma, 0, (size_t)num_vertices * sizeof(int32_t));
    cudaMemsetAsync(delta, 0, (size_t)num_vertices * sizeof(float));

    
    std::vector<int32_t> h_samples(num_samples);
    cudaMemcpy(h_samples.data(), sample_vertices, num_samples * sizeof(int32_t), cudaMemcpyDeviceToHost);

    std::vector<int> lo; 
    int h_count;

    for (std::size_t s = 0; s < num_samples; s++) {
        init_source<<<1, 1>>>(dist, sigma, queue, h_samples[s]);

        lo.clear();
        lo.push_back(0);
        int cur = 1, depth = 0, total = 1;

        
        while (cur > 0) {
            lo.push_back(lo.back() + cur);
            cudaMemsetAsync(counter, 0, sizeof(int32_t));

            if (cur >= WARP_THRESH) {
                int g = (int)(((int64_t)cur * 32 + BS - 1) / BS);
                bfs_expand_warp<<<g, BS>>>(offsets, indices, dist, sigma,
                    queue + lo[depth], cur, queue + lo[depth+1], counter, depth);
            } else {
                int g = (cur + BS - 1) / BS;
                bfs_expand_thread<<<g, BS>>>(offsets, indices, dist, sigma,
                    queue + lo[depth], cur, queue + lo[depth+1], counter, depth);
            }

            cudaMemcpy(&h_count, counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
            cur = h_count;
            total += cur;
            depth++;
        }

        int nl = depth;

        
        for (int l = nl - 1; l >= 0; l--) {
            int sz = lo[l + 1] - lo[l];
            if (sz <= 0) continue;
            if (sz >= WARP_THRESH) {
                int g = (int)(((int64_t)sz * 32 + BS - 1) / BS);
                backward_warp<<<g, BS>>>(offsets, indices, dist, sigma, delta, edge_centralities,
                    queue + lo[l], sz, l);
            } else {
                int g = (sz + BS - 1) / BS;
                backward_thread<<<g, BS>>>(offsets, indices, dist, sigma, delta, edge_centralities,
                    queue + lo[l], sz, l);
            }
        }

        
        if (total > 0) {
            int g = (total + BS - 1) / BS;
            reset_kernel<<<g, BS>>>(dist, sigma, delta, queue, total);
        }
    }

    
    float n = (float)num_vertices;
    bool do_scale = false;
    float sf = 1.0f;

    if (normalized) {
        sf = n * (n - 1.0f);
        do_scale = true;
    } else if (is_symmetric) {
        sf = 2.0f;
        do_scale = true;
    }

    if (do_scale && num_vertices > 1) {
        if ((int64_t)num_samples < (int64_t)num_vertices)
            sf *= (float)num_samples / n;
        int g = (num_edges + BS - 1) / BS;
        scale_kernel<<<g, BS>>>(edge_centralities, num_edges, 1.0f / sf);
    }

    cudaDeviceSynchronize();
}

}  
