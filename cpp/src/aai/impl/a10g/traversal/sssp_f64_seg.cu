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

#define UNREACHABLE_DIST DBL_MAX

struct Cache : Cacheable {
    uint32_t* d_frontier_a = nullptr;
    uint32_t* d_frontier_b = nullptr;
    int32_t* d_changed = nullptr;
    int32_t* h_changed = nullptr;
    int32_t* pred_scratch = nullptr;
    size_t frontier_capacity = 0;
    size_t pred_scratch_capacity = 0;

    Cache() {
        cudaMalloc(&d_changed, sizeof(int32_t));
        cudaMallocHost(&h_changed, sizeof(int32_t));
    }

    ~Cache() override {
        if (d_frontier_a) cudaFree(d_frontier_a);
        if (d_frontier_b) cudaFree(d_frontier_b);
        if (d_changed) cudaFree(d_changed);
        if (h_changed) cudaFreeHost(h_changed);
        if (pred_scratch) cudaFree(pred_scratch);
    }

    void ensure_frontier(size_t bitmap_bytes) {
        if (frontier_capacity < bitmap_bytes) {
            if (d_frontier_a) cudaFree(d_frontier_a);
            if (d_frontier_b) cudaFree(d_frontier_b);
            cudaMalloc(&d_frontier_a, bitmap_bytes);
            cudaMalloc(&d_frontier_b, bitmap_bytes);
            frontier_capacity = bitmap_bytes;
        }
    }

    void ensure_pred_scratch(size_t bytes) {
        if (pred_scratch_capacity < bytes) {
            if (pred_scratch) cudaFree(pred_scratch);
            cudaMalloc(&pred_scratch, bytes);
            pred_scratch_capacity = bytes;
        }
    }
};

__global__ void init_sssp(double* __restrict__ dist, int32_t* __restrict__ pred,
                          uint32_t* __restrict__ frontier,
                          int32_t N, int32_t source) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        dist[i] = (i == source) ? 0.0 : UNREACHABLE_DIST;
        pred[i] = -1;
    }
    if (i == 0) {
        frontier[source >> 5] = (1U << (source & 31));
    }
}


__global__ void sssp_single_block(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    double* __restrict__ dist,
    uint32_t* __restrict__ frontier_a,
    uint32_t* __restrict__ frontier_b,
    int32_t N, int32_t bitmap_words, double cutoff, int32_t max_iters)
{
    __shared__ int s_changed;
    uint32_t* curr = frontier_a;
    uint32_t* next = frontier_b;

    for (int iter = 0; iter < max_iters; iter++) {
        if (threadIdx.x == 0) s_changed = 0;
        __syncthreads();

        for (int32_t v = threadIdx.x; v < N; v += blockDim.x) {
            if (!((curr[v >> 5] >> (v & 31)) & 1U)) continue;
            double d_v = dist[v];
            if (d_v >= cutoff) continue;
            int32_t start = offsets[v], end = offsets[v + 1];
            for (int32_t e = start; e < end; e++) {
                int32_t u = indices[e];
                double new_dist = d_v + weights[e];
                if (new_dist < cutoff && new_dist < dist[u]) {
                    unsigned long long nll = __double_as_longlong(new_dist);
                    unsigned long long oll = atomicMin((unsigned long long*)&dist[u], nll);
                    if (nll < oll) {
                        atomicOr(&next[u >> 5], 1U << (u & 31));
                        s_changed = 1;
                    }
                }
            }
        }
        __syncthreads();
        if (s_changed == 0) break;
        for (int32_t i = threadIdx.x; i < bitmap_words; i += blockDim.x) curr[i] = 0;
        __syncthreads();
        uint32_t* tmp = curr; curr = next; next = tmp;
    }
}


__global__ void relax_edges(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    double* __restrict__ dist,
    const uint32_t* __restrict__ frontier_current,
    uint32_t* __restrict__ frontier_next,
    int32_t* __restrict__ d_changed,
    int32_t seg_high_end,    
    int32_t seg_mid_end,     
    int32_t seg_low_end,     
    int32_t num_blocks_high,
    int32_t num_blocks_mid,
    double cutoff)
{
    if (blockIdx.x < num_blocks_high) {
        
        int32_t v = blockIdx.x;
        if (v >= seg_high_end) return;
        if (!((frontier_current[v >> 5] >> (v & 31)) & 1U)) return;
        double d_v = dist[v];
        if (d_v >= cutoff) return;
        int32_t es = offsets[v], ee = offsets[v + 1];
        for (int32_t e = es + threadIdx.x; e < ee; e += blockDim.x) {
            int32_t u = indices[e];
            double new_dist = d_v + weights[e];
            if (new_dist < cutoff && new_dist < dist[u]) {
                unsigned long long nll = __double_as_longlong(new_dist);
                unsigned long long oll = atomicMin((unsigned long long*)&dist[u], nll);
                if (nll < oll) {
                    atomicOr(&frontier_next[u >> 5], 1U << (u & 31));
                    *d_changed = 1;
                }
            }
        }
    } else if (blockIdx.x < num_blocks_high + num_blocks_mid) {
        
        int32_t lb = blockIdx.x - num_blocks_high;
        int32_t wpb = blockDim.x >> 5;
        int32_t wib = threadIdx.x >> 5;
        int32_t lane = threadIdx.x & 31;
        int32_t v = seg_high_end + lb * wpb + wib;
        if (v >= seg_mid_end) return;
        if (!((frontier_current[v >> 5] >> (v & 31)) & 1U)) return;
        double d_v = dist[v];
        if (d_v >= cutoff) return;
        int32_t es = offsets[v], ee = offsets[v + 1];
        for (int32_t e = es + lane; e < ee; e += 32) {
            int32_t u = indices[e];
            double new_dist = d_v + weights[e];
            if (new_dist < cutoff && new_dist < dist[u]) {
                unsigned long long nll = __double_as_longlong(new_dist);
                unsigned long long oll = atomicMin((unsigned long long*)&dist[u], nll);
                if (nll < oll) {
                    atomicOr(&frontier_next[u >> 5], 1U << (u & 31));
                    *d_changed = 1;
                }
            }
        }
    } else {
        
        int32_t lb = blockIdx.x - num_blocks_high - num_blocks_mid;
        int32_t v = seg_mid_end + lb * blockDim.x + threadIdx.x;
        if (v >= seg_low_end) return;
        if (!((frontier_current[v >> 5] >> (v & 31)) & 1U)) return;
        double d_v = dist[v];
        if (d_v >= cutoff) return;
        int32_t es = offsets[v], ee = offsets[v + 1];
        for (int32_t e = es; e < ee; e++) {
            int32_t u = indices[e];
            double new_dist = d_v + weights[e];
            if (new_dist < cutoff && new_dist < dist[u]) {
                unsigned long long nll = __double_as_longlong(new_dist);
                unsigned long long oll = atomicMin((unsigned long long*)&dist[u], nll);
                if (nll < oll) {
                    atomicOr(&frontier_next[u >> 5], 1U << (u & 31));
                    *d_changed = 1;
                }
            }
        }
    }
}



__global__ void fix_predecessors(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t seg_high_end, int32_t seg_mid_end, int32_t seg_low_end,
    int32_t num_blocks_high, int32_t num_blocks_mid)
{
    int32_t u;

    if (blockIdx.x < num_blocks_high) {
        
        u = blockIdx.x;
        if (u >= seg_high_end) return;
        double d_u = dist[u];
        if (d_u >= UNREACHABLE_DIST) return;
        int32_t es = offsets[u], ee = offsets[u + 1];
        for (int32_t e = es + threadIdx.x; e < ee; e += blockDim.x) {
            int32_t v = indices[e];
            double expected = d_u + weights[e];
            if (__double_as_longlong(expected) == __double_as_longlong(dist[v])
                && d_u < dist[v]) {
                pred[v] = u;
            }
        }
        return;
    } else if (blockIdx.x < num_blocks_high + num_blocks_mid) {
        
        int32_t lb = blockIdx.x - num_blocks_high;
        int32_t wpb = blockDim.x >> 5;
        int32_t wib = threadIdx.x >> 5;
        int32_t lane = threadIdx.x & 31;
        u = seg_high_end + lb * wpb + wib;
        if (u >= seg_mid_end) return;
        double d_u = dist[u];
        if (d_u >= UNREACHABLE_DIST) return;
        int32_t es = offsets[u], ee = offsets[u + 1];
        for (int32_t e = es + lane; e < ee; e += 32) {
            int32_t v = indices[e];
            double expected = d_u + weights[e];
            if (__double_as_longlong(expected) == __double_as_longlong(dist[v])
                && d_u < dist[v]) {
                pred[v] = u;
            }
        }
        return;
    } else {
        
        int32_t lb = blockIdx.x - num_blocks_high - num_blocks_mid;
        u = seg_mid_end + lb * blockDim.x + threadIdx.x;
        if (u >= seg_low_end) return;
        double d_u = dist[u];
        if (d_u >= UNREACHABLE_DIST) return;
        int32_t es = offsets[u], ee = offsets[u + 1];
        for (int32_t e = es; e < ee; e++) {
            int32_t v = indices[e];
            double expected = d_u + weights[e];
            if (__double_as_longlong(expected) == __double_as_longlong(dist[v])
                && d_u < dist[v]) {
                pred[v] = u;
            }
        }
    }
}


__global__ void zerow_pred_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ dist,
    int32_t* __restrict__ pred,
    int32_t* __restrict__ changed,
    int32_t source, int32_t N
) {
    int32_t u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= N) return;
    double d_u = dist[u];
    if (d_u >= UNREACHABLE_DIST) return;
    if (u != source && pred[u] == -1) return;  
    int32_t es = offsets[u], ee = offsets[u + 1];
    for (int32_t e = es; e < ee; e++) {
        int32_t v = indices[e];
        if (v == source) continue;
        double w = weights[e];
        if (d_u + w != dist[v]) continue;     
        if (d_u != dist[v]) continue;          
        if (pred[v] == -1) {
            pred[v] = u;
            *changed = 1;
        }
    }
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

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg1 = seg[1];  
    int32_t seg2 = seg[2];  
    int32_t seg3 = seg[3];  

    int32_t bitmap_words = (num_vertices + 31) / 32;
    size_t bitmap_bytes = bitmap_words * sizeof(uint32_t);

    cache.ensure_frontier(bitmap_bytes);

    
    int32_t* pred = predecessors;
    if (!pred) {
        cache.ensure_pred_scratch(num_vertices * sizeof(int32_t));
        pred = cache.pred_scratch;
    }

    cudaMemsetAsync(cache.d_frontier_a, 0, bitmap_bytes);
    cudaMemsetAsync(cache.d_frontier_b, 0, bitmap_bytes);
    init_sssp<<<(num_vertices + 255) / 256, 256>>>(distances, pred, cache.d_frontier_a, num_vertices, source);

    if (num_vertices <= 20000) {
        
        sssp_single_block<<<1, 1024>>>(d_offsets, d_indices, edge_weights, distances,
            cache.d_frontier_a, cache.d_frontier_b, num_vertices, bitmap_words,
            cutoff, num_vertices);
    } else {
        
        uint32_t* fc = cache.d_frontier_a;
        uint32_t* fn = cache.d_frontier_b;

        for (int iter = 0; iter < num_vertices; iter++) {
            cudaMemsetAsync(cache.d_changed, 0, sizeof(int32_t));

            const int BS = 256;
            int32_t nbh = seg1;
            int32_t nbm = (seg2 - seg1 + (BS / 32) - 1) / (BS / 32);
            int32_t nbl = (seg3 - seg2 + BS - 1) / BS;
            int32_t total = nbh + nbm + nbl;
            if (total > 0) {
                relax_edges<<<total, BS>>>(d_offsets, d_indices, edge_weights, distances,
                    fc, fn, cache.d_changed, seg1, seg2, seg3, nbh, nbm, cutoff);
            }

            cudaMemsetAsync(fc, 0, bitmap_bytes);
            uint32_t* t = fc; fc = fn; fn = t;

            cudaMemcpy(cache.h_changed, cache.d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost);
            if (*cache.h_changed == 0) break;
        }
    }

    
    if (predecessors) {
        
        const int BS = 256;
        int32_t nbh = seg1;
        int32_t nbm = (seg2 - seg1 + (BS / 32) - 1) / (BS / 32);
        int32_t nbl = (seg3 - seg2 + BS - 1) / BS;
        int32_t total = nbh + nbm + nbl;
        if (total > 0) {
            fix_predecessors<<<total, BS>>>(d_offsets, d_indices, edge_weights, distances, predecessors,
                seg1, seg2, seg3, nbh, nbm);
        }

        
        int32_t h_ch = 1;
        for (int ziter = 0; ziter < num_vertices && h_ch; ziter++) {
            cudaMemsetAsync(cache.d_changed, 0, sizeof(int32_t));
            int grid = (num_vertices + 255) / 256;
            zerow_pred_kernel<<<grid, 256>>>(
                d_offsets, d_indices, edge_weights, distances, predecessors,
                cache.d_changed, source, num_vertices);
            cudaMemcpy(cache.h_changed, cache.d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost);
            h_ch = *cache.h_changed;
        }
    }
    cudaDeviceSynchronize();
}

}  
