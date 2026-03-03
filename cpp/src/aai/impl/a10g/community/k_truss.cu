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
#include <cub/cub.cuh>

namespace aai {

namespace {





__device__ __forceinline__ int lower_bound_dev(const int32_t* arr, int lo, int hi, int target) {
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}





__global__ void fill_src_kernel(
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ src,
    int32_t num_vertices
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;
    for (int e = offsets[u]; e < offsets[u + 1]; e++) {
        src[e] = u;
    }
}

__global__ void build_reverse_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ src,
    int32_t* __restrict__ reverse_idx,
    int32_t num_edges
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    int u = src[e];
    int v = indices[e];
    int lo = offsets[v], hi = offsets[v + 1];
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (indices[mid] < u) lo = mid + 1;
        else hi = mid;
    }
    reverse_idx[e] = lo;
}

__global__ void init_alive_kernel(int8_t* alive, const int32_t* src, const int32_t* indices, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) alive[i] = (src[i] != indices[i]) ? 1 : 0;
}


__global__ void compute_and_prune_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ src,
    int8_t* __restrict__ alive,
    const int32_t* __restrict__ reverse_idx,
    int32_t num_edges,
    int32_t threshold,
    int32_t* __restrict__ changed
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) return;
    if (!alive[e]) return;

    int u = src[e];
    int v = indices[e];
    if (u >= v) return;

    int start_u = offsets[u], end_u = offsets[u + 1];
    int start_v = offsets[v], end_v = offsets[v + 1];

    if (start_u >= end_u || start_v >= end_v) {
        alive[e] = 0;
        alive[reverse_idx[e]] = 0;
        return;
    }

    
    int i = start_u, j = start_v;
    int first_u = indices[i], first_v = indices[j];
    if (first_u < first_v) {
        i = lower_bound_dev(indices, i, end_u, first_v);
    } else if (first_v < first_u) {
        j = lower_bound_dev(indices, j, end_v, first_u);
    }

    if (i >= end_u || j >= end_v) {
        alive[e] = 0;
        alive[reverse_idx[e]] = 0;
        return;
    }

    
    int last_u = indices[end_u - 1], last_v = indices[end_v - 1];
    int max_val = (last_u < last_v) ? last_u : last_v;
    int eu = (last_u > max_val) ? lower_bound_dev(indices, i, end_u, max_val + 1) : end_u;
    int ev = (last_v > max_val) ? lower_bound_dev(indices, j, end_v, max_val + 1) : end_v;

    
    int su = eu - i, sv = ev - j;

    int sup = 0;

    if (su > 0 && sv > 0 && (su > (sv << 3) || sv > (su << 3))) {
        
        int s_start, s_end;
        int l_start, l_end;

        if (su <= sv) {
            s_start = i; s_end = eu;
            l_start = j; l_end = ev;
        } else {
            s_start = j; s_end = ev;
            l_start = i; l_end = eu;
        }

        int lp = l_start;
        for (int si = s_start; si < s_end && lp < l_end; si++) {
            if (!alive[si]) continue;
            int target = indices[si];
            if (target == u || target == v) continue;

            
            int pos = lp;
            int step = 1;
            while (pos + step < l_end && indices[pos + step] < target) {
                pos += step;
                step <<= 1;
            }
            int lo = pos, hi = (pos + step + 1 < l_end) ? pos + step + 1 : l_end;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (indices[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            lp = lo;

            if (lp < l_end && indices[lp] == target && alive[lp]) {
                sup++;
                if (sup >= threshold) goto done;
                lp++;
            }
        }
    } else {
        
        while (i < eu && j < ev) {
            int a = indices[i], b = indices[j];
            if (a == b) {
                if (a != u && a != v && alive[i] && alive[j]) {
                    sup++;
                    if (sup >= threshold) goto done;
                }
                i++; j++;
            } else if (a < b) {
                i++;
            } else {
                j++;
            }
        }
    }

    if (sup < threshold) {
        alive[e] = 0;
        alive[reverse_idx[e]] = 0;
        *changed = 1;
    }
done:;
}

__global__ void count_alive_kernel(
    const int8_t* __restrict__ alive,
    int32_t num_edges,
    int32_t* __restrict__ count
) {
    typedef cub::BlockReduce<int32_t, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_count = 0;

    for (int e = tid; e < num_edges; e += blockDim.x * gridDim.x) {
        if (alive[e]) local_count++;
    }

    int block_count = BlockReduce(temp_storage).Sum(local_count);
    if (threadIdx.x == 0) atomicAdd(count, block_count);
}

__global__ void compact_kernel(
    const int32_t* __restrict__ src,
    const int32_t* __restrict__ indices,
    const int8_t* __restrict__ alive,
    int32_t num_edges,
    int32_t* __restrict__ out_srcs,
    int32_t* __restrict__ out_dsts,
    int32_t* __restrict__ count
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges || !alive[e]) return;
    int pos = atomicAdd(count, 1);
    out_srcs[pos] = src[e];
    out_dsts[pos] = indices[e];
}





struct Cache : Cacheable {
    int32_t* src = nullptr;
    int32_t* reverse_idx = nullptr;
    int8_t* alive = nullptr;
    int32_t* changed = nullptr;

    int64_t src_capacity = 0;
    int64_t reverse_capacity = 0;
    int64_t alive_capacity = 0;
    bool changed_allocated = false;

    void ensure(int64_t num_edges) {
        if (src_capacity < num_edges) {
            if (src) cudaFree(src);
            cudaMalloc(&src, num_edges * sizeof(int32_t));
            src_capacity = num_edges;
        }
        if (reverse_capacity < num_edges) {
            if (reverse_idx) cudaFree(reverse_idx);
            cudaMalloc(&reverse_idx, num_edges * sizeof(int32_t));
            reverse_capacity = num_edges;
        }
        if (alive_capacity < num_edges) {
            if (alive) cudaFree(alive);
            cudaMalloc(&alive, num_edges * sizeof(int8_t));
            alive_capacity = num_edges;
        }
        if (!changed_allocated) {
            cudaMalloc(&changed, sizeof(int32_t));
            changed_allocated = true;
        }
    }

    ~Cache() override {
        if (src) cudaFree(src);
        if (reverse_idx) cudaFree(reverse_idx);
        if (alive) cudaFree(alive);
        if (changed) cudaFree(changed);
    }
};

}  

k_truss_result_t k_truss(const graph32_t& graph, int32_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    int32_t threshold = k - 2;

    
    if (num_edges == 0 || k <= 2) {
        if (k <= 2 && num_edges > 0) {
            int32_t* edge_srcs = nullptr;
            int32_t* edge_dsts = nullptr;
            cudaMalloc(&edge_srcs, num_edges * sizeof(int32_t));
            cudaMalloc(&edge_dsts, num_edges * sizeof(int32_t));

            int block = 256;
            int grid = (num_vertices + block - 1) / block;
            if (grid > 0) fill_src_kernel<<<grid, block>>>(d_offsets, edge_srcs, num_vertices);
            cudaMemcpy(edge_dsts, d_indices, num_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice);

            return {edge_srcs, edge_dsts, static_cast<std::size_t>(num_edges)};
        }
        return {nullptr, nullptr, 0};
    }

    cache.ensure(num_edges);

    int32_t* d_src = cache.src;
    int32_t* d_reverse = cache.reverse_idx;
    int8_t* d_alive = cache.alive;
    int32_t* d_changed = cache.changed;

    int block = 256;
    int grid;

    grid = (num_vertices + block - 1) / block;
    if (grid > 0) fill_src_kernel<<<grid, block>>>(d_offsets, d_src, num_vertices);

    grid = (num_edges + block - 1) / block;
    if (grid > 0) build_reverse_kernel<<<grid, block>>>(d_offsets, d_indices, d_src, d_reverse, num_edges);

    grid = (num_edges + block - 1) / block;
    if (grid > 0) init_alive_kernel<<<grid, block>>>(d_alive, d_src, d_indices, num_edges);

    cudaDeviceSynchronize();

    
    int32_t h_changed = 1;
    while (h_changed) {
        h_changed = 0;
        cudaMemset(d_changed, 0, sizeof(int32_t));
        grid = (num_edges + block - 1) / block;
        if (grid > 0)
            compute_and_prune_kernel<<<grid, block>>>(d_offsets, d_indices, d_src, d_alive,
                                                       d_reverse, num_edges, threshold, d_changed);
        cudaMemcpy(&h_changed, d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost);
    }

    
    int32_t* out_srcs = nullptr;
    int32_t* out_dsts = nullptr;
    cudaMalloc(&out_srcs, num_edges * sizeof(int32_t));
    cudaMalloc(&out_dsts, num_edges * sizeof(int32_t));
    cudaMemset(d_changed, 0, sizeof(int32_t));

    grid = (num_edges + block - 1) / block;
    if (grid > 0)
        compact_kernel<<<grid, block>>>(d_src, d_indices, d_alive, num_edges, out_srcs, out_dsts, d_changed);

    int32_t h_count;
    cudaMemcpy(&h_count, d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost);

    return {out_srcs, out_dsts, static_cast<std::size_t>(h_count)};
}

}  
