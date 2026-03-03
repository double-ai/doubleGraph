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
#include <cstddef>
#include <limits>

namespace aai {

namespace {




struct Cache : Cacheable {
    
    int32_t* counts = nullptr;
    int32_t* new_offsets = nullptr;
    void* scan_temp = nullptr;
    int32_t* new_indices = nullptr;

    
    float* hub_prev = nullptr;
    float* temp = nullptr;

    
    int64_t counts_capacity = 0;
    int64_t new_offsets_capacity = 0;
    size_t scan_temp_capacity = 0;
    int64_t new_indices_capacity = 0;
    int64_t hub_prev_capacity = 0;
    bool temp_allocated = false;

    void ensure_counts(int64_t n) {
        if (counts_capacity < n) {
            if (counts) cudaFree(counts);
            cudaMalloc(&counts, n * sizeof(int32_t));
            counts_capacity = n;
        }
    }

    void ensure_new_offsets(int64_t n) {
        if (new_offsets_capacity < n) {
            if (new_offsets) cudaFree(new_offsets);
            cudaMalloc(&new_offsets, n * sizeof(int32_t));
            new_offsets_capacity = n;
        }
    }

    void ensure_scan_temp(size_t bytes) {
        if (scan_temp_capacity < bytes) {
            if (scan_temp) cudaFree(scan_temp);
            cudaMalloc(&scan_temp, bytes);
            scan_temp_capacity = bytes;
        }
    }

    void ensure_new_indices(int64_t n) {
        if (new_indices_capacity < n) {
            if (new_indices) cudaFree(new_indices);
            cudaMalloc(&new_indices, n * sizeof(int32_t));
            new_indices_capacity = n;
        }
    }

    void ensure_hub_prev(int64_t n) {
        if (hub_prev_capacity < n) {
            if (hub_prev) cudaFree(hub_prev);
            cudaMalloc(&hub_prev, n * sizeof(float));
            hub_prev_capacity = n;
        }
    }

    void ensure_temp() {
        if (!temp_allocated) {
            cudaMalloc(&temp, 4 * sizeof(float));
            temp_allocated = true;
        }
    }

    ~Cache() override {
        if (counts) cudaFree(counts);
        if (new_offsets) cudaFree(new_offsets);
        if (scan_temp) cudaFree(scan_temp);
        if (new_indices) cudaFree(new_indices);
        if (hub_prev) cudaFree(hub_prev);
        if (temp) cudaFree(temp);
    }
};




__device__ __forceinline__ bool is_edge_active(const uint32_t* edge_mask, int edge_idx) {
    return (edge_mask[edge_idx >> 5] >> (edge_idx & 31)) & 1;
}





__global__ void count_and_compact_low(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_counts,
    int start_v, int end_v
) {
    int v = start_v + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= end_v) return;
    int s = old_offsets[v], e = old_offsets[v + 1];
    int cnt = 0;
    for (int i = s; i < e; i++)
        if (is_edge_active(edge_mask, i)) cnt++;
    active_counts[v] = cnt;
}

__global__ void count_active_high(
    const int32_t* __restrict__ old_offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_counts,
    int start_v, int end_v
) {
    int v = start_v + blockIdx.x;
    if (v >= end_v) return;
    int s = old_offsets[v], e = old_offsets[v + 1];

    int cnt = 0;
    for (int i = s + threadIdx.x; i < e; i += blockDim.x)
        if (is_edge_active(edge_mask, i)) cnt++;

    typedef cub::BlockReduce<int, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    int total = BR(tmp).Sum(cnt);
    if (threadIdx.x == 0) active_counts[v] = total;
}

__global__ void compact_edges(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int start_v, int end_v
) {
    int v = start_v + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= end_v) return;
    int os = old_offsets[v], oe = old_offsets[v + 1];
    int wp = new_offsets[v];
    for (int i = os; i < oe; i++) {
        if (is_edge_active(edge_mask, i))
            new_indices[wp++] = old_indices[i];
    }
}

__global__ void compact_edges_high(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    int start_v, int end_v
) {
    int v = start_v + blockIdx.x;
    if (v >= end_v) return;
    int os = old_offsets[v], oe = old_offsets[v + 1];
    int base_wp = new_offsets[v];

    for (int chunk_start = os; chunk_start < oe; chunk_start += blockDim.x) {
        int i = chunk_start + threadIdx.x;
        bool active = (i < oe) && is_edge_active(edge_mask, i);
        int32_t val = active ? old_indices[i] : 0;

        unsigned mask = __ballot_sync(0xffffffff, active);
        int lane = threadIdx.x & 31;
        int warp_id = threadIdx.x >> 5;
        int warp_offset = __popc(mask & ((1u << lane) - 1));
        int warp_total = __popc(mask);

        __shared__ int warp_bases[8];
        __shared__ int block_total;

        if (lane == 0) warp_bases[warp_id] = warp_total;
        __syncthreads();

        if (threadIdx.x == 0) {
            int sum = 0;
            for (int w = 0; w < (blockDim.x + 31) / 32; w++) {
                int tmp = warp_bases[w];
                warp_bases[w] = sum;
                sum += tmp;
            }
            block_total = sum;
        }
        __syncthreads();

        if (active) {
            int pos = base_wp + warp_bases[warp_id] + warp_offset;
            new_indices[pos] = val;
        }

        base_wp += block_total;
        __syncthreads();
    }
}





__global__ void spmv_auth_high(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ hub, float* __restrict__ auth,
    int start_vertex, int end_vertex
) {
    int v = start_vertex + blockIdx.x;
    if (v >= end_vertex) return;
    int es = offsets[v], ee = offsets[v + 1];

    float sum = 0.0f;
    for (int e = es + threadIdx.x; e < ee; e += blockDim.x)
        sum += __ldg(&hub[indices[e]]);

    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    float bs = BR(tmp).Sum(sum);
    if (threadIdx.x == 0) auth[v] = bs;
}

__global__ void spmv_auth_mid(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ hub, float* __restrict__ auth,
    int start_vertex, int end_vertex
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int v = start_vertex + warp_id;
    if (v >= end_vertex) return;
    int es = offsets[v], ee = offsets[v + 1];

    float sum = 0.0f;
    for (int e = es + lane; e < ee; e += 32)
        sum += __ldg(&hub[indices[e]]);

    #pragma unroll
    for (int m = 16; m > 0; m >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, m);

    if (lane == 0) auth[v] = sum;
}

__global__ void spmv_auth_low(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ hub, float* __restrict__ auth,
    int start_vertex, int end_vertex
) {
    int v = start_vertex + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= end_vertex) return;
    int es = offsets[v], ee = offsets[v + 1];

    float sum = 0.0f;
    for (int e = es; e < ee; e++)
        sum += __ldg(&hub[indices[e]]);
    auth[v] = sum;
}





__global__ void spmv_hub_scatter_high(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ auth, float* __restrict__ hub,
    int start_vertex, int end_vertex
) {
    int v = start_vertex + blockIdx.x;
    if (v >= end_vertex) return;
    float av = auth[v];
    if (av == 0.0f) return;
    int es = offsets[v], ee = offsets[v + 1];
    for (int e = es + threadIdx.x; e < ee; e += blockDim.x)
        atomicAdd(&hub[indices[e]], av);
}

__global__ void spmv_hub_scatter_mid(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ auth, float* __restrict__ hub,
    int start_vertex, int end_vertex
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int v = start_vertex + warp_id;
    if (v >= end_vertex) return;
    float av = auth[v];
    if (av == 0.0f) return;
    int es = offsets[v], ee = offsets[v + 1];
    for (int e = es + lane; e < ee; e += 32)
        atomicAdd(&hub[indices[e]], av);
}

__global__ void spmv_hub_scatter_low(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const float* __restrict__ auth, float* __restrict__ hub,
    int start_vertex, int end_vertex
) {
    int v = start_vertex + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= end_vertex) return;
    float av = auth[v];
    if (av == 0.0f) return;
    int es = offsets[v], ee = offsets[v + 1];
    for (int e = es; e < ee; e++)
        atomicAdd(&hub[indices[e]], av);
}





__global__ void reduce_max_abs(const float* __restrict__ data, float* __restrict__ result, int n) {
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    float mx = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        mx = fmaxf(mx, fabsf(data[i]));
    float bm = BR(tmp).Reduce(mx, ::cuda::maximum<float>{});
    if (threadIdx.x == 0 && bm > 0.0f) {
        unsigned int* addr = (unsigned int*)result;
        unsigned int exp = atomicAdd((unsigned int*)addr, 0u);
        while (__uint_as_float(exp) < bm) {
            unsigned int old = atomicCAS(addr, exp, __float_as_uint(bm));
            if (old == exp) break;
            exp = old;
        }
    }
}

__global__ void normalize_and_diff(float* __restrict__ nd, const float* __restrict__ od,
                                    const float* __restrict__ sv, float* __restrict__ dout, int n) {
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    float s = *sv;
    float is = (s > 0.0f) ? (1.0f / s) : 0.0f;
    float ls = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        float v = nd[i] * is;
        nd[i] = v;
        ls += fabsf(v - od[i]);
    }
    float bs = BR(tmp).Sum(ls);
    if (threadIdx.x == 0) atomicAdd(dout, bs);
}

__global__ void scale_kernel(float* data, int n, const float* sv) {
    float s = *sv;
    if (s <= 0.0f) return;
    float is = 1.0f / s;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) data[tid] *= is;
}

__global__ void reduce_abs_sum(const float* __restrict__ data, float* __restrict__ result, int n) {
    typedef cub::BlockReduce<float, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    float ls = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        ls += fabsf(data[i]);
    float bs = BR(tmp).Sum(ls);
    if (threadIdx.x == 0) atomicAdd(result, bs);
}

__global__ void fill_uniform(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) data[tid] = 1.0f / (float)n;
}





static void do_precompact(
    const int32_t* old_offsets, const int32_t* old_indices, const uint32_t* edge_mask,
    int32_t* counts, int s0, int s1, int s2, int s3, int s4, cudaStream_t stream
) {
    int n;
    n = s1 - s0;
    if (n > 0) count_active_high<<<n, 256, 0, stream>>>(old_offsets, edge_mask, counts, s0, s1);
    n = s3 - s1;
    if (n > 0) count_and_compact_low<<<(n+255)/256, 256, 0, stream>>>(old_offsets, old_indices, edge_mask, counts, s1, s3);
    n = s4 - s3;
    if (n > 0) cudaMemsetAsync(counts + s3, 0, n * sizeof(int32_t), stream);
}

static void do_compact_edges(
    const int32_t* old_offsets, const int32_t* old_indices, const uint32_t* edge_mask,
    const int32_t* new_offsets, int32_t* new_indices,
    int s0, int s1, int s2, int s3, cudaStream_t stream
) {
    int n;
    n = s1 - s0;
    if (n > 0) compact_edges_high<<<n, 256, 0, stream>>>(old_offsets, old_indices, edge_mask, new_offsets, new_indices, s0, s1);
    n = s3 - s1;
    if (n > 0) compact_edges<<<(n+255)/256, 256, 0, stream>>>(old_offsets, old_indices, edge_mask, new_offsets, new_indices, s1, s3);
}

static void do_spmv_auth(const int32_t* off, const int32_t* idx,
                          const float* hub, float* auth,
                          int s0, int s1, int s2, int s3, int s4, cudaStream_t st) {
    int n;
    n = s1 - s0;
    if (n > 0) spmv_auth_high<<<n, 256, 0, st>>>(off, idx, hub, auth, s0, s1);
    n = s2 - s1;
    if (n > 0) spmv_auth_mid<<<(n+7)/8, 256, 0, st>>>(off, idx, hub, auth, s1, s2);
    n = s3 - s2;
    if (n > 0) spmv_auth_low<<<(n+255)/256, 256, 0, st>>>(off, idx, hub, auth, s2, s3);
    n = s4 - s3;
    if (n > 0) cudaMemsetAsync(auth + s3, 0, n * sizeof(float), st);
}

static void do_spmv_hub_scatter(const int32_t* off, const int32_t* idx,
                                 const float* auth, float* hub,
                                 int s0, int s1, int s2, int s3, cudaStream_t st) {
    int n;
    n = s1 - s0;
    if (n > 0) spmv_hub_scatter_high<<<n, 256, 0, st>>>(off, idx, auth, hub, s0, s1);
    n = s2 - s1;
    if (n > 0) spmv_hub_scatter_mid<<<(n+7)/8, 256, 0, st>>>(off, idx, auth, hub, s1, s2);
    n = s3 - s2;
    if (n > 0) spmv_hub_scatter_low<<<(n+255)/256, 256, 0, st>>>(off, idx, auth, hub, s2, s3);
}

static void do_reduce_max_abs(const float* d, float* r, int n, cudaStream_t s) {
    if (n <= 0) return;
    int g = (n + 255) / 256; if (g > 512) g = 512;
    reduce_max_abs<<<g, 256, 0, s>>>(d, r, n);
}

static void do_normalize_and_diff(float* nd, const float* od, const float* sv, float* dout, int n, cudaStream_t s) {
    if (n <= 0) return;
    int g = (n + 255) / 256; if (g > 512) g = 512;
    normalize_and_diff<<<g, 256, 0, s>>>(nd, od, sv, dout, n);
}

static void do_scale(float* d, int n, const float* sv, cudaStream_t s) {
    if (n <= 0) return;
    scale_kernel<<<(n+255)/256, 256, 0, s>>>(d, n, sv);
}

static void do_reduce_abs_sum(const float* d, float* r, int n, cudaStream_t s) {
    if (n <= 0) return;
    int g = (n + 255) / 256; if (g > 512) g = 512;
    reduce_abs_sum<<<g, 256, 0, s>>>(d, r, n);
}

static void do_fill_uniform(float* d, int n, cudaStream_t s) {
    if (n <= 0) return;
    fill_uniform<<<(n+255)/256, 256, 0, s>>>(d, n);
}

}  

HitsResult hits_seg_mask(const graph32_t& graph,
                         float* hubs,
                         float* authorities,
                         float epsilon,
                         std::size_t max_iterations,
                         bool has_initial_hubs_guess,
                         bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;

    const auto& seg_opt = graph.segment_offsets.value();
    int seg[5] = {seg_opt[0], seg_opt[1], seg_opt[2], seg_opt[3], seg_opt[4]};

    cudaStream_t stream = 0;

    if (num_vertices == 0) {
        return HitsResult{max_iterations, false, std::numeric_limits<float>::max()};
    }

    
    
    

    cache.ensure_counts(num_vertices + 1);
    int32_t* d_counts = cache.counts;
    cudaMemsetAsync(d_counts + num_vertices, 0, sizeof(int32_t), stream);

    do_precompact(d_offsets, d_indices, d_edge_mask, d_counts,
                  seg[0], seg[1], seg[2], seg[3], seg[4], stream);

    cache.ensure_new_offsets(num_vertices + 1);
    int32_t* d_new_offsets = cache.new_offsets;

    size_t scan_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, (int32_t*)nullptr, (int32_t*)nullptr, num_vertices + 1);
    cache.ensure_scan_temp(scan_bytes);

    cub::DeviceScan::ExclusiveSum(cache.scan_temp, scan_bytes, d_counts, d_new_offsets,
                                   num_vertices + 1, stream);

    int32_t num_active;
    cudaMemcpyAsync(&num_active, d_new_offsets + num_vertices,
                    sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int64_t indices_size = num_active > 0 ? (int64_t)num_active : 1LL;
    cache.ensure_new_indices(indices_size);
    int32_t* d_new_indices = cache.new_indices;

    if (num_active > 0) {
        do_compact_edges(d_offsets, d_indices, d_edge_mask,
                         d_new_offsets, d_new_indices,
                         seg[0], seg[1], seg[2], seg[3], stream);
    }

    
    
    

    cache.ensure_hub_prev(num_vertices);
    cache.ensure_temp();

    float* d_hub_a = hubs;
    float* d_hub_b = cache.hub_prev;
    float* d_auth = authorities;
    float* d_temp = cache.temp;

    
    if (has_initial_hubs_guess) {
        cudaMemsetAsync(d_temp, 0, sizeof(float), stream);
        do_reduce_abs_sum(d_hub_a, d_temp, num_vertices, stream);
        do_scale(d_hub_a, num_vertices, d_temp, stream);
    } else {
        do_fill_uniform(d_hub_a, num_vertices, stream);
    }

    float* hub_curr = d_hub_a;
    float* hub_next = d_hub_b;

    std::size_t iterations = 0;
    bool converged = false;
    float final_norm = 0.0f;
    float conv_thresh = epsilon * static_cast<float>(num_vertices);

    const int32_t* iter_offsets = d_new_offsets;
    const int32_t* iter_indices = d_new_indices;

    for (std::size_t iter = 0; iter < max_iterations; iter++) {
        
        do_spmv_auth(iter_offsets, iter_indices, hub_curr, d_auth,
                     seg[0], seg[1], seg[2], seg[3], seg[4], stream);

        
        cudaMemsetAsync(hub_next, 0, num_vertices * sizeof(float), stream);
        do_spmv_hub_scatter(iter_offsets, iter_indices, d_auth, hub_next,
                            seg[0], seg[1], seg[2], seg[3], stream);

        
        cudaMemsetAsync(d_temp, 0, sizeof(float), stream);
        do_reduce_max_abs(hub_next, d_temp, num_vertices, stream);
        cudaMemsetAsync(d_temp + 2, 0, sizeof(float), stream);
        do_normalize_and_diff(hub_next, hub_curr, d_temp, d_temp + 2, num_vertices, stream);

        
        cudaMemsetAsync(d_temp + 1, 0, sizeof(float), stream);
        do_reduce_max_abs(d_auth, d_temp + 1, num_vertices, stream);
        do_scale(d_auth, num_vertices, d_temp + 1, stream);

        
        float diff;
        cudaMemcpyAsync(&diff, d_temp + 2, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        iterations = iter + 1;
        final_norm = diff;

        float* tmp = hub_curr; hub_curr = hub_next; hub_next = tmp;

        if (diff < conv_thresh) {
            converged = true;
            break;
        }
    }

    if (hub_curr != d_hub_a)
        cudaMemcpyAsync(d_hub_a, hub_curr, num_vertices * sizeof(float),
                         cudaMemcpyDeviceToDevice, stream);

    if (normalize) {
        cudaMemsetAsync(d_temp, 0, sizeof(float), stream);
        do_reduce_abs_sum(d_hub_a, d_temp, num_vertices, stream);
        do_scale(d_hub_a, num_vertices, d_temp, stream);

        cudaMemsetAsync(d_temp, 0, sizeof(float), stream);
        do_reduce_abs_sum(d_auth, d_temp, num_vertices, stream);
        do_scale(d_auth, num_vertices, d_temp, stream);
    }

    return HitsResult{iterations, converged, final_norm};
}

}  
