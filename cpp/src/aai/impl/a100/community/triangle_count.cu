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
#include <cub/cub.cuh>
#include <cstdint>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* degrees = nullptr;
    int64_t degrees_cap = 0;

    int32_t* out_counts = nullptr;
    int64_t out_counts_cap = 0;

    int32_t* dag_offsets_buf = nullptr;
    int64_t dag_offsets_cap = 0;

    int32_t* sorted_buf = nullptr;
    int64_t sorted_cap = 0;

    void ensure_vertices(int64_t nv) {
        if (degrees_cap < nv) {
            if (degrees) cudaFree(degrees);
            cudaMalloc(&degrees, nv * sizeof(int32_t));
            degrees_cap = nv;
        }
        if (out_counts_cap < nv) {
            if (out_counts) cudaFree(out_counts);
            cudaMalloc(&out_counts, nv * sizeof(int32_t));
            out_counts_cap = nv;
        }
        if (dag_offsets_cap < nv + 1) {
            if (dag_offsets_buf) cudaFree(dag_offsets_buf);
            cudaMalloc(&dag_offsets_buf, (nv + 1) * sizeof(int32_t));
            dag_offsets_cap = nv + 1;
        }
    }

    void ensure_sorted(int64_t n) {
        if (sorted_cap < n) {
            if (sorted_buf) cudaFree(sorted_buf);
            cudaMalloc(&sorted_buf, n * sizeof(int32_t));
            sorted_cap = n;
        }
    }

    ~Cache() override {
        if (degrees) cudaFree(degrees);
        if (out_counts) cudaFree(out_counts);
        if (dag_offsets_buf) cudaFree(dag_offsets_buf);
        if (sorted_buf) cudaFree(sorted_buf);
    }
};

__device__ __forceinline__ int warp_reduce_sum(int val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void compute_degrees(const int* __restrict__ offsets, int* __restrict__ degrees, int n) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < n) degrees[u] = __ldg(&offsets[u + 1]) - __ldg(&offsets[u]);
}

__global__ void count_out_edges(
    const int* __restrict__ offsets, const int* __restrict__ indices,
    const int* __restrict__ degrees, int* __restrict__ out_counts, int num_vertices
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;
    int start = __ldg(&offsets[u]), end = __ldg(&offsets[u + 1]);
    int du = __ldg(&degrees[u]);
    int count = 0;
    for (int e = start; e < end; e++) {
        int v = __ldg(&indices[e]);
        if (v == u) continue;
        int dv = __ldg(&degrees[v]);
        if (du < dv || (du == dv && u < v)) count++;
    }
    out_counts[u] = count;
}

__global__ void build_dag_indices(
    const int* __restrict__ offsets, const int* __restrict__ indices,
    const int* __restrict__ degrees,
    const int* __restrict__ dag_offsets, int* __restrict__ dag_indices, int num_vertices
) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_vertices) return;
    int start = __ldg(&offsets[u]), end = __ldg(&offsets[u + 1]);
    int du = __ldg(&degrees[u]);
    int pos = __ldg(&dag_offsets[u]);
    for (int e = start; e < end; e++) {
        int v = __ldg(&indices[e]);
        if (v == u) continue;
        int dv = __ldg(&degrees[v]);
        if (du < dv || (du == dv && u < v)) dag_indices[pos++] = v;
    }
}

__global__ __launch_bounds__(256, 8)
void tc_dag_vertex_kernel(
    const int* __restrict__ dag_offsets,
    const int* __restrict__ dag_indices,
    int* __restrict__ counts,
    int num_vertices
) {
    int u = blockIdx.x;
    if (u >= num_vertices) return;

    int u_start = __ldg(&dag_offsets[u]);
    int u_end = __ldg(&dag_offsets[u + 1]);
    int u_deg = u_end - u_start;
    if (u_deg < 1) return;

    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int num_warps = blockDim.x >> 5;

    int u_tc = 0;

    for (int e = warp_id; e < u_deg; e += num_warps) {
        int v = __ldg(&dag_indices[u_start + e]);
        int v_start = __ldg(&dag_offsets[v]);
        int v_end = __ldg(&dag_offsets[v + 1]);
        int v_deg = v_end - v_start;
        if (v_deg == 0) continue;

        const int* a = dag_indices + u_start;
        int alen = u_deg;
        const int* b = dag_indices + v_start;
        int blen = v_deg;
        if (alen > blen) {
            const int* tmp = a; a = b; b = tmp;
            int t = alen; alen = blen; blen = t;
        }

        int a_first = __ldg(&a[0]), a_last = __ldg(&a[alen - 1]);
        int b_first = __ldg(&b[0]), b_last = __ldg(&b[blen - 1]);
        if (a_first > b_last || a_last < b_first) continue;

        int tc = 0;
        int search_lo = 0;

        for (int i = lane; i < alen; i += 32) {
            int val = __ldg(&a[i]);
            if (i > 0 && __ldg(&a[i - 1]) == val) continue;
            if (val > b_last) break;
            if (val < b_first) continue;

            int lo = search_lo, hi = blen;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (__ldg(&b[mid]) < val) lo = mid + 1;
                else hi = mid;
            }
            if (lo < blen && __ldg(&b[lo]) == val) {
                tc++;
                atomicAdd(&counts[val], 1);
                search_lo = lo + 1;
            } else {
                search_lo = lo;
            }
        }

        tc = warp_reduce_sum(tc);
        if (lane == 0 && tc > 0) {
            atomicAdd(&counts[v], tc);
            u_tc += tc;
        }
    }

    if (lane == 0 && u_tc > 0) {
        atomicAdd(&counts[u], u_tc);
    }
}

__global__ __launch_bounds__(512, 2)
void tc_vertex_subset_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    int* __restrict__ counts,
    const int* __restrict__ vertex_list,
    int n_output
) {
    int idx = blockIdx.x;
    if (idx >= n_output) return;
    int u = __ldg(&vertex_list[idx]);
    int u_start = __ldg(&offsets[u]);
    int u_end = __ldg(&offsets[u + 1]);
    int u_deg = u_end - u_start;
    if (u_deg < 2) { if (threadIdx.x == 0) counts[idx] = 0; return; }

    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int num_warps = blockDim.x >> 5;
    int warp_total = 0;

    for (int e = warp_id; e < u_deg; e += num_warps) {
        int v = __ldg(&indices[u_start + e]);
        if (v == u) continue;
        if (e > 0 && __ldg(&indices[u_start + e - 1]) == v) continue;
        int v_start = __ldg(&offsets[v]);
        int v_end = __ldg(&offsets[v + 1]);
        int v_deg = v_end - v_start;
        if (v_deg == 0) continue;

        const int* a = indices + u_start;
        int alen = u_deg;
        const int* b = indices + v_start;
        int blen = v_deg;
        if (alen > blen) {
            const int* tmp = a; a = b; b = tmp;
            int t = alen; alen = blen; blen = t;
        }

        int a_last = __ldg(&a[alen - 1]);
        int b_first = __ldg(&b[0]), b_last = __ldg(&b[blen - 1]);
        if (__ldg(&a[0]) > b_last || a_last < b_first) continue;

        int tc = 0;
        int search_lo = 0;
        for (int i = lane; i < alen; i += 32) {
            int val = __ldg(&a[i]);
            if (val == u || val == v) continue;
            if (i > 0 && __ldg(&a[i - 1]) == val) continue;
            if (val > b_last) break;
            if (val < b_first) continue;
            int lo = search_lo, hi = blen;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (__ldg(&b[mid]) < val) lo = mid + 1;
                else hi = mid;
            }
            if (lo < blen && __ldg(&b[lo]) == val) {
                tc++;
                search_lo = lo + 1;
            } else {
                search_lo = lo;
            }
        }
        tc = warp_reduce_sum(tc);
        if (lane == 0) warp_total += tc;
    }

    __shared__ int smem[32];
    if (lane == 0) smem[warp_id] = warp_total;
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int w = 0; w < num_warps; w++) total += smem[w];
        counts[idx] = total >> 1;
    }
}

}  

void triangle_count(const graph32_t& graph,
                    int32_t* counts,
                    const int32_t* vertices,
                    std::size_t n_vertices) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices_g = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    if (vertices != nullptr && n_vertices > 0) {
        
        cache.ensure_sorted(num_edges);

        size_t sb = 0;
        cub::DeviceSegmentedSort::SortKeys(nullptr, sb,
            (const int*)nullptr, (int*)nullptr,
            num_edges, num_vertices_g, d_offsets, d_offsets + 1);
        void* sort_tmp = nullptr;
        cudaMalloc(&sort_tmp, sb);
        cub::DeviceSegmentedSort::SortKeys(sort_tmp, sb,
            d_indices, cache.sorted_buf,
            num_edges, num_vertices_g, d_offsets, d_offsets + 1);
        cudaFree(sort_tmp);

        int n_out = static_cast<int>(n_vertices);
        if (n_out > 0) {
            tc_vertex_subset_kernel<<<n_out, 512>>>(d_offsets, cache.sorted_buf,
                counts, vertices, n_out);
        }
        return;
    }

    
    cache.ensure_vertices(num_vertices_g);

    compute_degrees<<<(num_vertices_g + 255) / 256, 256>>>(
        d_offsets, cache.degrees, num_vertices_g);

    count_out_edges<<<(num_vertices_g + 255) / 256, 256>>>(
        d_offsets, d_indices, cache.degrees, cache.out_counts, num_vertices_g);

    
    cudaMemsetAsync(cache.dag_offsets_buf, 0, sizeof(int32_t));
    {
        size_t isb = 0;
        cub::DeviceScan::InclusiveSum(nullptr, isb,
            (const int*)nullptr, (int*)nullptr, num_vertices_g);
        void* is_tmp = nullptr;
        cudaMalloc(&is_tmp, isb);
        cub::DeviceScan::InclusiveSum(is_tmp, isb,
            cache.out_counts, cache.dag_offsets_buf + 1, num_vertices_g);
        cudaFree(is_tmp);
    }

    
    int32_t num_dag_edges;
    cudaMemcpy(&num_dag_edges, cache.dag_offsets_buf + num_vertices_g,
        sizeof(int32_t), cudaMemcpyDeviceToHost);

    if (num_dag_edges == 0) {
        cudaMemsetAsync(counts, 0, num_vertices_g * sizeof(int32_t));
        return;
    }

    
    int32_t* dag_indices = nullptr;
    cudaMalloc(&dag_indices, num_dag_edges * sizeof(int32_t));
    build_dag_indices<<<(num_vertices_g + 255) / 256, 256>>>(
        d_offsets, d_indices, cache.degrees,
        cache.dag_offsets_buf, dag_indices, num_vertices_g);

    
    cache.ensure_sorted(num_dag_edges);
    {
        size_t sb = 0;
        cub::DeviceSegmentedSort::SortKeys(nullptr, sb,
            (const int*)nullptr, (int*)nullptr,
            num_dag_edges, num_vertices_g,
            cache.dag_offsets_buf, cache.dag_offsets_buf + 1);
        void* sort_tmp = nullptr;
        cudaMalloc(&sort_tmp, sb);
        cub::DeviceSegmentedSort::SortKeys(sort_tmp, sb,
            dag_indices, cache.sorted_buf,
            num_dag_edges, num_vertices_g,
            cache.dag_offsets_buf, cache.dag_offsets_buf + 1);
        cudaFree(sort_tmp);
    }

    cudaFree(dag_indices);

    
    if (num_vertices_g > 0) {
        cudaMemsetAsync(counts, 0, num_vertices_g * sizeof(int32_t));
        tc_dag_vertex_kernel<<<num_vertices_g, 256>>>(
            cache.dag_offsets_buf, cache.sorted_buf,
            counts, num_vertices_g);
    }
}

}  
