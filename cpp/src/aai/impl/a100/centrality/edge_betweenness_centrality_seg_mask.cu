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
#include <algorithm>
#include <thread>

namespace aai {

namespace {

static const int MAX_SLOTS = 10;

struct SourceSlot {
    int32_t* d_distances = nullptr;
    float* d_sigma = nullptr;
    float* d_delta = nullptr;
    int32_t* d_bfs_stack = nullptr;
    int32_t* d_next_count = nullptr;
    cudaStream_t stream = nullptr;
    int32_t* h_next_count = nullptr;
};

struct Cache : Cacheable {
    SourceSlot slots[MAX_SLOTS];
    int32_t alloc_nv = 0;
    int num_slots = 0;

    void ensure(int32_t nv, int ns) {
        int needed = (ns < MAX_SLOTS) ? ns : MAX_SLOTS;
        if (nv <= alloc_nv && needed <= num_slots) return;
        free_all();
        alloc_nv = nv;
        num_slots = needed;

        for (int i = 0; i < num_slots; i++) {
            cudaMalloc(&slots[i].d_distances, nv * sizeof(int32_t));
            cudaMalloc(&slots[i].d_sigma, nv * sizeof(float));
            cudaMalloc(&slots[i].d_delta, nv * sizeof(float));
            cudaMalloc(&slots[i].d_bfs_stack, nv * sizeof(int32_t));
            cudaMalloc(&slots[i].d_next_count, sizeof(int32_t));
            cudaMallocHost(&slots[i].h_next_count, sizeof(int32_t));
            cudaStreamCreate(&slots[i].stream);
            cudaMemset(slots[i].d_distances, 0xFF, nv * sizeof(int32_t));
            cudaMemset(slots[i].d_sigma, 0, nv * sizeof(float));
            cudaMemset(slots[i].d_delta, 0, nv * sizeof(float));
        }
    }

    void free_all() {
        for (int i = 0; i < num_slots; i++) {
            if (slots[i].d_distances) cudaFree(slots[i].d_distances);
            if (slots[i].d_sigma) cudaFree(slots[i].d_sigma);
            if (slots[i].d_delta) cudaFree(slots[i].d_delta);
            if (slots[i].d_bfs_stack) cudaFree(slots[i].d_bfs_stack);
            if (slots[i].d_next_count) cudaFree(slots[i].d_next_count);
            if (slots[i].h_next_count) cudaFreeHost(slots[i].h_next_count);
            if (slots[i].stream) cudaStreamDestroy(slots[i].stream);
            slots[i] = {};
        }
        alloc_nv = 0;
        num_slots = 0;
    }

    ~Cache() override {
        free_all();
    }
};





__device__ __forceinline__ bool is_edge_active(const uint32_t* __restrict__ edge_mask, int32_t e) {
    return (__ldg(&edge_mask[e >> 5]) >> (e & 31)) & 1;
}

__global__ __launch_bounds__(256, 8)
void bfs_expand_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ distances,
    float* __restrict__ sigma,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    int32_t* __restrict__ bfs_stack,
    int32_t stack_write_offset,
    int32_t* __restrict__ next_count,
    int32_t current_level
) {
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;
    int32_t num_warps = (blockDim.x * gridDim.x) >> 5;
    int32_t next_level = current_level + 1;

    for (int32_t i = warp_id; i < frontier_size; i += num_warps) {
        int32_t v = frontier[i];
        float sv = sigma[v];
        int32_t start = __ldg(&offsets[v]);
        int32_t end = __ldg(&offsets[v + 1]);
        int32_t degree = end - start;

        for (int32_t j = lane; j < degree; j += 32) {
            int32_t e = start + j;
            if (!is_edge_active(edge_mask, e)) continue;

            int32_t w = __ldg(&indices[e]);
            int32_t old_d = atomicCAS(&distances[w], -1, next_level);

            if ((old_d != -1) && (old_d != next_level)) continue;

            atomicAdd(&sigma[w], sv);

            if (old_d == -1) {
                int32_t pos = atomicAdd(next_count, 1);
                bfs_stack[stack_write_offset + pos] = w;
            }
        }
    }
}

__global__ __launch_bounds__(256, 8)
void backward_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ distances,
    const float* __restrict__ sigma,
    float* __restrict__ delta,
    float* __restrict__ edge_bc,
    const int32_t* __restrict__ level_vertices,
    int32_t level_size,
    int32_t current_level
) {
    int32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int32_t lane = threadIdx.x & 31;
    int32_t num_warps = (blockDim.x * gridDim.x) >> 5;
    int32_t next_level = current_level + 1;

    for (int32_t i = warp_id; i < level_size; i += num_warps) {
        int32_t v = level_vertices[i];
        float sv = sigma[v];
        float my_delta = 0.0f;
        int32_t start = __ldg(&offsets[v]);
        int32_t end = __ldg(&offsets[v + 1]);
        int32_t degree = end - start;

        for (int32_t j = lane; j < degree; j += 32) {
            int32_t e = start + j;
            if (!is_edge_active(edge_mask, e)) continue;

            int32_t w = __ldg(&indices[e]);
            if (distances[w] == next_level) {
                float sw = sigma[w];
                float dw = delta[w];
                float c = sv / sw * (1.0f + dw);
                my_delta += c;
                atomicAdd(&edge_bc[e], c);
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            my_delta += __shfl_down_sync(0xffffffff, my_delta, offset);
        }

        if (lane == 0) {
            delta[v] = my_delta;
        }
    }
}

__global__ void scale_kernel(float* __restrict__ data, int32_t n, float factor) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;
    int32_t n4 = n >> 2;
    float4* data4 = reinterpret_cast<float4*>(data);
    for (int32_t i = tid; i < n4; i += stride) {
        float4 val = data4[i];
        val.x *= factor; val.y *= factor; val.z *= factor; val.w *= factor;
        data4[i] = val;
    }
    for (int32_t i = (n4 << 2) + tid; i < n; i += stride) {
        data[i] *= factor;
    }
}

__global__ void reset_arrays_kernel(
    int32_t* __restrict__ distances,
    float* __restrict__ sigma,
    float* __restrict__ delta,
    const int32_t* __restrict__ visited_vertices,
    int32_t num_visited
) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t stride = blockDim.x * gridDim.x;
    for (int32_t i = tid; i < num_visited; i += stride) {
        int32_t v = visited_vertices[i];
        distances[v] = -1;
        sigma[v] = 0.0f;
        delta[v] = 0.0f;
    }
}





void launch_bfs_expand(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    int32_t* distances, float* sigma,
    const int32_t* frontier, int32_t frontier_size,
    int32_t* bfs_stack, int32_t stack_write_offset,
    int32_t* next_count,
    int32_t current_level, cudaStream_t stream
) {
    if (frontier_size == 0) return;
    int threads = 256;
    int warps_per_block = threads / 32;
    int grid = (frontier_size + warps_per_block - 1) / warps_per_block;
    if (grid > 2048) grid = 2048;
    bfs_expand_kernel<<<grid, threads, 0, stream>>>(
        offsets, indices, edge_mask, distances, sigma,
        frontier, frontier_size, bfs_stack, stack_write_offset, next_count,
        current_level
    );
}

void launch_backward(
    const int32_t* offsets, const int32_t* indices, const uint32_t* edge_mask,
    const int32_t* distances, const float* sigma,
    float* delta, float* edge_bc,
    const int32_t* level_vertices, int32_t level_size,
    int32_t current_level, cudaStream_t stream
) {
    if (level_size == 0) return;
    int threads = 256;
    int warps_per_block = threads / 32;
    int grid = (level_size + warps_per_block - 1) / warps_per_block;
    if (grid > 2048) grid = 2048;
    backward_kernel<<<grid, threads, 0, stream>>>(
        offsets, indices, edge_mask, distances, sigma, delta, edge_bc,
        level_vertices, level_size, current_level
    );
}

void launch_scale(float* data, int32_t n, float factor, cudaStream_t stream) {
    if (n == 0) return;
    int block = 256;
    int grid = (n / 4 + block - 1) / block;
    if (grid < 1) grid = 1;
    if (grid > 2048) grid = 2048;
    scale_kernel<<<grid, block, 0, stream>>>(data, n, factor);
}

void launch_reset_arrays(
    int32_t* distances, float* sigma, float* delta,
    const int32_t* visited, int32_t num_visited, cudaStream_t stream
) {
    if (num_visited == 0) return;
    int block = 256;
    int grid = (num_visited + block - 1) / block;
    if (grid > 2048) grid = 2048;
    reset_arrays_kernel<<<grid, block, 0, stream>>>(distances, sigma, delta, visited, num_visited);
}





void thread_process_source(
    SourceSlot* slot,
    int32_t source,
    const int32_t* d_offsets,
    const int32_t* d_indices,
    const uint32_t* d_edge_mask,
    float* d_edge_bc,
    int32_t nv
) {
    cudaStream_t stream = slot->stream;

    int32_t zero = 0;
    float one_f = 1.0f;
    cudaMemcpyAsync(slot->d_distances + source, &zero, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(slot->d_sigma + source, &one_f, sizeof(float), cudaMemcpyHostToDevice, stream);

    cudaMemcpyAsync(slot->d_bfs_stack, &source, sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    std::vector<int32_t> h_level_offsets = {0};
    int num_levels = 0;

    int32_t frontier_start = 0;
    int32_t frontier_sz = 1;
    int32_t bfs_stack_top = 1;
    int32_t level = 0;

    while (frontier_sz > 0) {
        num_levels++;
        h_level_offsets.push_back(bfs_stack_top);

        cudaMemsetAsync(slot->d_next_count, 0, sizeof(int32_t), stream);

        launch_bfs_expand(
            d_offsets, d_indices, d_edge_mask,
            slot->d_distances, slot->d_sigma,
            slot->d_bfs_stack + frontier_start, frontier_sz,
            slot->d_bfs_stack, bfs_stack_top,
            slot->d_next_count,
            level, stream
        );

        cudaMemcpyAsync(slot->h_next_count, slot->d_next_count, sizeof(int32_t),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int32_t next_size = *slot->h_next_count;

        frontier_start = bfs_stack_top;
        bfs_stack_top += next_size;
        frontier_sz = next_size;
        level++;
    }

    int max_level = level - 1;
    for (int L = max_level; L >= 0; L--) {
        int32_t lstart = h_level_offsets[L];
        int32_t lend = h_level_offsets[L + 1];
        int32_t lsize = lend - lstart;
        if (lsize > 0) {
            launch_backward(
                d_offsets, d_indices, d_edge_mask,
                slot->d_distances, slot->d_sigma, slot->d_delta, d_edge_bc,
                slot->d_bfs_stack + lstart, lsize,
                L, stream
            );
        }
    }

    launch_reset_arrays(slot->d_distances, slot->d_sigma, slot->d_delta,
                       slot->d_bfs_stack, bfs_stack_top, stream);
    cudaStreamSynchronize(stream);
}

}  





void edge_betweenness_centrality_seg_mask(const graph32_t& graph,
                                           float* edge_centralities,
                                           bool normalized,
                                           const int32_t* sample_vertices,
                                           std::size_t num_samples) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    bool is_symmetric = graph.is_symmetric;
    const uint32_t* d_edge_mask = graph.edge_mask;

    cache.ensure(nv, (int)num_samples);

    float* d_edge_bc = edge_centralities;
    cudaMemset(d_edge_bc, 0, ne * sizeof(float));

    std::vector<int32_t> h_samples(num_samples);
    cudaMemcpy(h_samples.data(), sample_vertices, num_samples * sizeof(int32_t),
               cudaMemcpyDeviceToHost);

    int num_slots = cache.num_slots;

    for (size_t batch_start = 0; batch_start < num_samples; batch_start += num_slots) {
        size_t batch_end = batch_start + num_slots;
        if (batch_end > num_samples) batch_end = num_samples;
        int batch_size = (int)(batch_end - batch_start);

        if (batch_size == 1) {
            thread_process_source(
                &cache.slots[0], h_samples[batch_start],
                d_offsets, d_indices, d_edge_mask,
                d_edge_bc, nv
            );
        } else {
            std::vector<std::thread> threads;
            threads.reserve(batch_size);
            for (int i = 0; i < batch_size; i++) {
                threads.emplace_back(
                    thread_process_source,
                    &cache.slots[i], h_samples[batch_start + i],
                    d_offsets, d_indices, d_edge_mask,
                    d_edge_bc, nv
                );
            }
            for (auto& t : threads) t.join();
        }
    }

    bool has_scale = false;
    float scale_factor = 1.0f;

    if (normalized) {
        scale_factor = (float)nv * ((float)nv - 1.0f);
        has_scale = true;
    } else if (is_symmetric) {
        scale_factor = 2.0f;
        has_scale = true;
    }

    if (has_scale && nv > 1) {
        if (num_samples < (size_t)nv) {
            scale_factor *= (float)num_samples / (float)nv;
        }
        float factor = 1.0f / scale_factor;
        launch_scale(d_edge_bc, ne, factor, 0);
    }

    cudaDeviceSynchronize();
}

}  
