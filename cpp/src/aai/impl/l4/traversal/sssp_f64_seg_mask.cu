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
#include <cstring>

namespace aai {

namespace {

#define DMAX_BITS 0x7FEFFFFFFFFFFFFFLL

struct Cache : Cacheable {
    uint32_t* d_frontier_a = nullptr;
    uint32_t* d_frontier_b = nullptr;
    int32_t* d_updated_flag = nullptr;
    int32_t* h_updated_pinned = nullptr;
    size_t alloc_words = 0;

    Cache() {
        cudaMalloc(&d_updated_flag, sizeof(int32_t));
        cudaMallocHost(&h_updated_pinned, sizeof(int32_t));
    }

    ~Cache() override {
        if (d_frontier_a) cudaFree(d_frontier_a);
        if (d_frontier_b) cudaFree(d_frontier_b);
        if (d_updated_flag) cudaFree(d_updated_flag);
        if (h_updated_pinned) cudaFreeHost(h_updated_pinned);
    }

    void ensure_frontier(int32_t bitmap_words) {
        if ((size_t)bitmap_words > alloc_words) {
            if (d_frontier_a) cudaFree(d_frontier_a);
            if (d_frontier_b) cudaFree(d_frontier_b);
            cudaMalloc(&d_frontier_a, bitmap_words * sizeof(uint32_t));
            cudaMalloc(&d_frontier_b, bitmap_words * sizeof(uint32_t));
            alloc_words = bitmap_words;
        }
    }
};



__device__ __forceinline__ bool atomicMinDouble(double* addr, double val) {
    long long int* addr_ll = (long long int*)addr;
    long long int val_ll = __double_as_longlong(val);
    long long int old_ll = atomicMin(addr_ll, val_ll);
    return (old_ll > val_ll);
}

__global__ void init_sssp_kernel(
    double* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t n,
    int32_t source
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dist[idx] = (idx == source) ? 0.0 : __longlong_as_double(DMAX_BITS);
        pred[idx] = -1;
    }
}

__launch_bounds__(256)
__global__ void relax_all_frontier_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ dist,
    const uint32_t* __restrict__ frontier_in,
    uint32_t* __restrict__ frontier_out,
    int32_t* __restrict__ updated_flag,
    int32_t n_active,
    long long int cutoff_bits
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n_active) return;

    if (!((frontier_in[u >> 5] >> (u & 31)) & 1)) return;

    long long int du_bits = __double_as_longlong(dist[u]);
    if (du_bits >= cutoff_bits) return;

    double du = __longlong_as_double(du_bits);
    int start = offsets[u];
    int end = offsets[u + 1];

    for (int e = start; e < end; e++) {
        if (!((edge_mask[e >> 5] >> (e & 31)) & 1)) continue;

        int v = indices[e];
        double new_dist = du + weights[e];
        long long int new_bits = __double_as_longlong(new_dist);

        if (new_bits < cutoff_bits) {
            if (atomicMinDouble(&dist[v], new_dist)) {
                atomicOr(&frontier_out[v >> 5], 1U << (v & 31));
                *updated_flag = 1;
            }
        }
    }
}

__launch_bounds__(256)
__global__ void relax_low_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ dist,
    const uint32_t* __restrict__ frontier_in,
    uint32_t* __restrict__ frontier_out,
    int32_t* __restrict__ updated_flag,
    int32_t seg_start,
    int32_t seg_end,
    long long int cutoff_bits
) {
    int u = seg_start + blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= seg_end) return;

    if (!((frontier_in[u >> 5] >> (u & 31)) & 1)) return;

    long long int du_bits = __double_as_longlong(dist[u]);
    if (du_bits >= cutoff_bits) return;

    double du = __longlong_as_double(du_bits);
    int start = offsets[u];
    int end = offsets[u + 1];

    for (int e = start; e < end; e++) {
        if (!((edge_mask[e >> 5] >> (e & 31)) & 1)) continue;
        int v = indices[e];
        double new_dist = du + weights[e];
        long long int new_bits = __double_as_longlong(new_dist);
        if (new_bits < cutoff_bits) {
            if (atomicMinDouble(&dist[v], new_dist)) {
                atomicOr(&frontier_out[v >> 5], 1U << (v & 31));
                *updated_flag = 1;
            }
        }
    }
}

__launch_bounds__(256)
__global__ void relax_mid_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ dist,
    const uint32_t* __restrict__ frontier_in,
    uint32_t* __restrict__ frontier_out,
    int32_t* __restrict__ updated_flag,
    int32_t seg_start,
    int32_t seg_end,
    long long int cutoff_bits
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int u = seg_start + warp_id;
    if (u >= seg_end) return;

    if (!((frontier_in[u >> 5] >> (u & 31)) & 1)) return;

    long long int du_bits = __double_as_longlong(dist[u]);
    if (du_bits >= cutoff_bits) return;

    double du = __longlong_as_double(du_bits);
    int start = offsets[u];
    int end = offsets[u + 1];

    for (int e = start + lane; e < end; e += 32) {
        if (!((edge_mask[e >> 5] >> (e & 31)) & 1)) continue;
        int v = indices[e];
        double new_dist = du + weights[e];
        long long int new_bits = __double_as_longlong(new_dist);
        if (new_bits < cutoff_bits) {
            if (atomicMinDouble(&dist[v], new_dist)) {
                atomicOr(&frontier_out[v >> 5], 1U << (v & 31));
                *updated_flag = 1;
            }
        }
    }
}

__launch_bounds__(512)
__global__ void relax_high_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    double* __restrict__ dist,
    const uint32_t* __restrict__ frontier_in,
    uint32_t* __restrict__ frontier_out,
    int32_t* __restrict__ updated_flag,
    int32_t seg_start,
    int32_t seg_end,
    long long int cutoff_bits
) {
    int u = seg_start + blockIdx.x;
    if (u >= seg_end) return;

    if (!((frontier_in[u >> 5] >> (u & 31)) & 1)) return;

    long long int du_bits = __double_as_longlong(dist[u]);
    if (du_bits >= cutoff_bits) return;

    double du = __longlong_as_double(du_bits);
    int start = offsets[u];
    int end = offsets[u + 1];

    for (int e = start + threadIdx.x; e < end; e += blockDim.x) {
        if (!((edge_mask[e >> 5] >> (e & 31)) & 1)) continue;
        int v = indices[e];
        double new_dist = du + weights[e];
        long long int new_bits = __double_as_longlong(new_dist);
        if (new_bits < cutoff_bits) {
            if (atomicMinDouble(&dist[v], new_dist)) {
                atomicOr(&frontier_out[v >> 5], 1U << (v & 31));
                *updated_flag = 1;
            }
        }
    }
}

__global__ void compute_predecessors_posweight_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t n,
    int32_t source
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;

    long long int du_bits = __double_as_longlong(dist[u]);
    if (du_bits >= DMAX_BITS) return;

    double du = __longlong_as_double(du_bits);
    int start = offsets[u];
    int end = offsets[u + 1];

    for (int e = start; e < end; e++) {
        if (!((edge_mask[e >> 5] >> (e & 31)) & 1)) continue;
        double w = weights[e];
        int v = indices[e];
        if (v == source) continue;
        double dv = dist[v];
        if (du + w == dv && du < dv) {
            pred[v] = u;
        }
    }
}

__global__ void compute_predecessors_zeroweight_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t n,
    int32_t source,
    int32_t* __restrict__ changed
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= n) return;

    if (u != source && pred[u] == -1) return;

    long long int du_bits = __double_as_longlong(dist[u]);
    if (du_bits >= DMAX_BITS) return;

    double du = __longlong_as_double(du_bits);
    int start = offsets[u];
    int end = offsets[u + 1];

    for (int e = start; e < end; e++) {
        if (!((edge_mask[e >> 5] >> (e & 31)) & 1)) continue;
        double w = weights[e];
        int v = indices[e];
        if (v == source) continue;
        if (du + w == dist[v] && dist[v] == du && pred[v] == -1) {
            pred[v] = u;
            *changed = 1;
        }
    }
}



static inline long long int double_to_bits(double val) {
    long long int bits;
    memcpy(&bits, &val, sizeof(double));
    return bits;
}

static void launch_init_sssp(double* dist, int32_t* pred, int32_t n, int32_t source, cudaStream_t stream) {
    init_sssp_kernel<<<(n + 255) / 256, 256, 0, stream>>>(dist, pred, n, source);
}

static void launch_relax_all(
    const int32_t* offsets, const int32_t* indices, const double* weights,
    const uint32_t* edge_mask, double* dist,
    const uint32_t* frontier_in, uint32_t* frontier_out, int32_t* updated_flag,
    int32_t n_active, double cutoff, cudaStream_t stream
) {
    if (n_active <= 0) return;
    relax_all_frontier_kernel<<<(n_active + 255) / 256, 256, 0, stream>>>(
        offsets, indices, weights, edge_mask, dist,
        frontier_in, frontier_out, updated_flag,
        n_active, double_to_bits(cutoff));
}

static void launch_relax_seg(
    const int32_t* offsets, const int32_t* indices, const double* weights,
    const uint32_t* edge_mask, double* dist,
    const uint32_t* frontier_in, uint32_t* frontier_out, int32_t* updated_flag,
    int32_t seg0, int32_t seg1, int32_t seg2, int32_t seg3,
    double cutoff, cudaStream_t stream
) {
    long long int cb = double_to_bits(cutoff);
    
    if (seg1 > seg0) {
        relax_high_kernel<<<seg1 - seg0, 512, 0, stream>>>(
            offsets, indices, weights, edge_mask, dist,
            frontier_in, frontier_out, updated_flag,
            seg0, seg1, cb);
    }
    
    if (seg2 > seg1) {
        int n = seg2 - seg1;
        int64_t total = (int64_t)n * 32;
        relax_mid_kernel<<<(int)((total + 255) / 256), 256, 0, stream>>>(
            offsets, indices, weights, edge_mask, dist,
            frontier_in, frontier_out, updated_flag,
            seg1, seg2, cb);
    }
    
    if (seg3 > seg2) {
        int n = seg3 - seg2;
        relax_low_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
            offsets, indices, weights, edge_mask, dist,
            frontier_in, frontier_out, updated_flag,
            seg2, seg3, cb);
    }
}

static void launch_compute_predecessors_posweight(
    const int32_t* offsets, const int32_t* indices, const double* weights,
    const uint32_t* edge_mask, const double* dist, int32_t* pred, int32_t n,
    int32_t source, cudaStream_t stream
) {
    compute_predecessors_posweight_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        offsets, indices, weights, edge_mask, dist, pred, n, source);
}

static void launch_compute_predecessors_zeroweight(
    const int32_t* offsets, const int32_t* indices, const double* weights,
    const uint32_t* edge_mask, const double* dist, int32_t* pred, int32_t n,
    int32_t source, int32_t* changed, cudaStream_t stream
) {
    compute_predecessors_zeroweight_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        offsets, indices, weights, edge_mask, dist, pred, n, source, changed);
}

}  

void sssp_seg_mask(const graph32_t& graph,
                   const double* edge_weights,
                   int32_t source,
                   double* distances,
                   int32_t* predecessors,
                   double cutoff) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    const uint32_t* d_edge_mask = graph.edge_mask;

    const auto& seg_vec = graph.segment_offsets.value();
    int32_t seg[5];
    for (int i = 0; i < 5; i++) seg[i] = seg_vec[i];

    cudaStream_t stream = 0;
    int32_t bitmap_words = (num_vertices + 31) / 32;
    cache.ensure_frontier(bitmap_words);

    
    launch_init_sssp(distances, predecessors, num_vertices, source, stream);

    
    cudaMemsetAsync(cache.d_frontier_a, 0, bitmap_words * sizeof(uint32_t), stream);
    uint32_t sb = 1U << (source % 32);
    cudaMemcpyAsync(&cache.d_frontier_a[source / 32], &sb, sizeof(uint32_t), cudaMemcpyHostToDevice, stream);

    uint32_t* frontier_in = cache.d_frontier_a;
    uint32_t* frontier_out = cache.d_frontier_b;

    
    bool use_segments = (seg[1] > seg[0]) || (seg[2] > seg[1]);
    int32_t n_active = seg[3]; 

    
    int check_interval = 1;
    if (num_vertices >= 50000) check_interval = 2;
    if (num_vertices >= 500000) check_interval = 4;

    for (int iter = 0; iter < num_vertices; iter++) {
        cudaMemsetAsync(frontier_out, 0, bitmap_words * sizeof(uint32_t), stream);

        if (iter % check_interval == 0) {
            cudaMemsetAsync(cache.d_updated_flag, 0, sizeof(int32_t), stream);
        }

        if (use_segments) {
            launch_relax_seg(d_offsets, d_indices, edge_weights, d_edge_mask,
                distances, frontier_in, frontier_out, cache.d_updated_flag,
                seg[0], seg[1], seg[2], seg[3], cutoff, stream);
        } else {
            launch_relax_all(d_offsets, d_indices, edge_weights, d_edge_mask,
                distances, frontier_in, frontier_out, cache.d_updated_flag,
                n_active, cutoff, stream);
        }

        if ((iter + 1) % check_interval == 0) {
            cudaMemcpyAsync(cache.h_updated_pinned, cache.d_updated_flag, sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (*cache.h_updated_pinned == 0) break;
        }

        uint32_t* tmp = frontier_in;
        frontier_in = frontier_out;
        frontier_out = tmp;
    }

    
    launch_compute_predecessors_posweight(d_offsets, d_indices, edge_weights, d_edge_mask,
        distances, predecessors, num_vertices, source, stream);

    
    for (int bfs_iter = 0; bfs_iter < num_vertices; bfs_iter++) {
        cudaMemsetAsync(cache.d_updated_flag, 0, sizeof(int32_t), stream);
        launch_compute_predecessors_zeroweight(d_offsets, d_indices, edge_weights, d_edge_mask,
            distances, predecessors, num_vertices, source, cache.d_updated_flag, stream);
        cudaMemcpyAsync(cache.h_updated_pinned, cache.d_updated_flag, sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        if (*cache.h_updated_pinned == 0) break;
    }
}

}  
