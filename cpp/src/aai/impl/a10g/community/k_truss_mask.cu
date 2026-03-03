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

namespace aai {

namespace {

struct Cache : Cacheable {
    uint32_t* mask = nullptr;
    int32_t* src = nullptr;
    int32_t* changed = nullptr;
    int32_t* counter = nullptr;
    int64_t mask_capacity = 0;
    int64_t src_capacity = 0;
    bool scalars_allocated = false;

    void ensure(int32_t num_edges) {
        int32_t mask_words = (num_edges + 31) / 32;
        if (mask_capacity < mask_words) {
            if (mask) cudaFree(mask);
            cudaMalloc(&mask, (size_t)mask_words * sizeof(uint32_t));
            mask_capacity = mask_words;
        }
        if (src_capacity < num_edges) {
            if (src) cudaFree(src);
            cudaMalloc(&src, (size_t)num_edges * sizeof(int32_t));
            src_capacity = num_edges;
        }
        if (!scalars_allocated) {
            cudaMalloc(&changed, sizeof(int32_t));
            cudaMalloc(&counter, sizeof(int32_t));
            scalars_allocated = true;
        }
    }

    ~Cache() override {
        if (mask) cudaFree(mask);
        if (src) cudaFree(src);
        if (changed) cudaFree(changed);
        if (counter) cudaFree(counter);
    }
};

__device__ __forceinline__ bool is_active(const uint32_t* mask, int32_t e) {
    return (__ldg(&mask[e >> 5]) >> (e & 31)) & 1u;
}

__device__ __forceinline__ void deactivate(uint32_t* mask, int32_t e) {
    atomicAnd(&mask[e >> 5], ~(1u << (e & 31)));
}

__device__ __forceinline__ int32_t find_src(const int32_t* offsets, int32_t n, int32_t e) {
    int32_t lo = 0, hi = n;
    while (lo < hi) {
        int32_t mid = (lo + hi) >> 1;
        if (__ldg(&offsets[mid + 1]) <= e) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int32_t lower_bound_dev(const int32_t* arr, int32_t start, int32_t end, int32_t target) {
    int32_t lo = start, hi = end;
    while (lo < hi) {
        int32_t mid = (lo + hi) >> 1;
        if (__ldg(&arr[mid]) < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void fill_src_kernel(const int32_t* __restrict__ offsets, int32_t* __restrict__ src,
                                int32_t num_vertices, int32_t num_edges) {
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    src[e] = find_src(offsets, num_vertices, e);
}

__global__ void remove_selfloops_kernel(
    const int32_t* __restrict__ src,
    const int32_t* __restrict__ indices,
    uint32_t* __restrict__ mask,
    int32_t num_edges
) {
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    if (__ldg(&src[e]) == __ldg(&indices[e])) {
        deactivate(mask, e);
    }
}

__global__ void k_truss_peel_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ src,
    uint32_t* __restrict__ mask,
    int32_t num_edges,
    int32_t threshold,
    int32_t* __restrict__ changed
) {
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    if (!is_active(mask, e)) return;

    int32_t u = __ldg(&src[e]);
    int32_t v = __ldg(&indices[e]);
    if (u >= v) return;

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);
    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    int32_t support = 0;

    if (u_deg * 8 < v_deg) {
        for (int32_t i = u_start; i < u_end; i++) {
            int32_t w = __ldg(&indices[i]);
            if (w == u || w == v) continue;
            if (!is_active(mask, i)) continue;

            int32_t j = lower_bound_dev(indices, v_start, v_end, w);
            if (j < v_end && __ldg(&indices[j]) == w && is_active(mask, j)) {
                support++;
                if (support >= threshold) return;
            }
        }
    } else if (v_deg * 8 < u_deg) {
        for (int32_t j = v_start; j < v_end; j++) {
            int32_t w = __ldg(&indices[j]);
            if (w == u || w == v) continue;
            if (!is_active(mask, j)) continue;

            int32_t i = lower_bound_dev(indices, u_start, u_end, w);
            if (i < u_end && __ldg(&indices[i]) == w && is_active(mask, i)) {
                support++;
                if (support >= threshold) return;
            }
        }
    } else {
        int32_t i = u_start, j = v_start;
        while (i < u_end && j < v_end) {
            int32_t ui = __ldg(&indices[i]);
            int32_t vj = __ldg(&indices[j]);
            if (ui == vj) {
                if (ui != u && ui != v && is_active(mask, i) && is_active(mask, j)) {
                    support++;
                    if (support >= threshold) return;
                }
                i++; j++;
            } else if (ui < vj) {
                i++;
            } else {
                j++;
            }
        }
    }

    deactivate(mask, e);
    int32_t rev = lower_bound_dev(indices, v_start, v_end, u);
    if (rev < v_end && __ldg(&indices[rev]) == u)
        deactivate(mask, rev);
    atomicOr(changed, 1);
}

__global__ void count_active_kernel(const uint32_t* __restrict__ mask, int32_t num_edges, int32_t* __restrict__ count) {
    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t num_words = (num_edges + 31) >> 5;
    if (tid >= num_words) return;

    uint32_t word = __ldg(&mask[tid]);
    if (tid == num_words - 1) {
        int32_t valid_bits = num_edges - (tid << 5);
        if (valid_bits < 32) word &= (1u << valid_bits) - 1u;
    }
    int32_t cnt = __popc(word);
    if (cnt > 0) atomicAdd(count, cnt);
}

__global__ void extract_edges_kernel(
    const int32_t* __restrict__ src,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ mask,
    int32_t num_edges,
    int32_t* __restrict__ out_srcs,
    int32_t* __restrict__ out_dsts,
    int32_t* __restrict__ counter
) {
    int32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    if (!is_active(mask, e)) return;
    int32_t pos = atomicAdd(counter, 1);
    out_srcs[pos] = __ldg(&src[e]);
    out_dsts[pos] = __ldg(&indices[e]);
}

}  

k_truss_result_t k_truss_mask(const graph32_t& graph,
                              int32_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;
    int32_t threshold = k - 2;

    cache.ensure(num_edges);

    int32_t mask_words = (num_edges + 31) / 32;
    cudaMemcpyAsync(cache.mask, d_edge_mask, (size_t)mask_words * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    int b = 256, g;

    g = (num_edges + b - 1) / b;
    fill_src_kernel<<<g, b>>>(d_offsets, cache.src, num_vertices, num_edges);

    g = (num_edges + b - 1) / b;
    remove_selfloops_kernel<<<g, b>>>(cache.src, d_indices, cache.mask, num_edges);

    if (threshold > 0) {
        while (true) {
            cudaMemsetAsync(cache.changed, 0, sizeof(int32_t));
            g = (num_edges + b - 1) / b;
            k_truss_peel_kernel<<<g, b>>>(d_offsets, d_indices, cache.src, cache.mask, num_edges, threshold, cache.changed);

            int32_t h_changed = 0;
            cudaMemcpy(&h_changed, cache.changed, sizeof(int32_t), cudaMemcpyDeviceToHost);
            if (!h_changed) break;
        }
    }

    cudaMemsetAsync(cache.counter, 0, sizeof(int32_t));
    int nw = (num_edges + 31) >> 5;
    g = (nw + b - 1) / b;
    count_active_kernel<<<g, b>>>(cache.mask, num_edges, cache.counter);

    int32_t h_count = 0;
    cudaMemcpy(&h_count, cache.counter, sizeof(int32_t), cudaMemcpyDeviceToHost);

    int32_t* out_srcs = nullptr;
    int32_t* out_dsts = nullptr;
    size_t out_size = (h_count > 0) ? (size_t)h_count : 1;
    cudaMalloc(&out_srcs, out_size * sizeof(int32_t));
    cudaMalloc(&out_dsts, out_size * sizeof(int32_t));

    if (h_count > 0) {
        cudaMemsetAsync(cache.counter, 0, sizeof(int32_t));
        g = (num_edges + b - 1) / b;
        extract_edges_kernel<<<g, b>>>(cache.src, d_indices, cache.mask, num_edges,
                                       out_srcs, out_dsts, cache.counter);
    }

    cudaDeviceSynchronize();

    return k_truss_result_t{out_srcs, out_dsts, (std::size_t)h_count};
}

}  
