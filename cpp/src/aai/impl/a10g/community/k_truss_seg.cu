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
#include <algorithm>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* edge_src = nullptr;
    int32_t* rev = nullptr;
    int32_t* is_fwd = nullptr;
    int32_t* prefix = nullptr;
    uint8_t* alive = nullptr;
    int32_t* fwd_pos_a = nullptr;
    int32_t* fwd_pos_b = nullptr;
    int32_t* removed = nullptr;
    int64_t edge_capacity = 0;

    void* cub_temp = nullptr;
    size_t cub_capacity = 0;

    void ensure_edge_buffers(int64_t ne) {
        if (edge_capacity >= ne) return;
        if (edge_src) {
            cudaFree(edge_src);
            cudaFree(rev);
            cudaFree(is_fwd);
            cudaFree(prefix);
            cudaFree(alive);
            cudaFree(fwd_pos_a);
            cudaFree(fwd_pos_b);
            cudaFree(removed);
        }
        cudaMalloc(&edge_src, ne * sizeof(int32_t));
        cudaMalloc(&rev, ne * sizeof(int32_t));
        cudaMalloc(&is_fwd, ne * sizeof(int32_t));
        cudaMalloc(&prefix, ne * sizeof(int32_t));
        cudaMalloc(&alive, ne * sizeof(uint8_t));
        cudaMalloc(&fwd_pos_a, ne * sizeof(int32_t));
        cudaMalloc(&fwd_pos_b, ne * sizeof(int32_t));
        cudaMalloc(&removed, sizeof(int32_t));
        edge_capacity = ne;
    }

    void ensure_cub_temp(size_t bytes) {
        if (cub_capacity >= bytes) return;
        if (cub_temp) cudaFree(cub_temp);
        cudaMalloc(&cub_temp, bytes);
        cub_capacity = bytes;
    }

    ~Cache() override {
        if (edge_src) cudaFree(edge_src);
        if (rev) cudaFree(rev);
        if (is_fwd) cudaFree(is_fwd);
        if (prefix) cudaFree(prefix);
        if (alive) cudaFree(alive);
        if (fwd_pos_a) cudaFree(fwd_pos_a);
        if (fwd_pos_b) cudaFree(fwd_pos_b);
        if (removed) cudaFree(removed);
        if (cub_temp) cudaFree(cub_temp);
    }
};



__device__ __forceinline__ int d_lower_bound(const int* __restrict__ arr, int lo, int hi, int target) {
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (__ldg(&arr[mid]) < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

__global__ void compute_sources(
    const int* __restrict__ offsets,
    int* __restrict__ edge_src,
    int num_vertices,
    int num_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    int lo = 0, hi = num_vertices;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (__ldg(&offsets[mid]) <= idx) lo = mid;
        else hi = mid - 1;
    }
    edge_src[idx] = lo;
}

__global__ void compute_reverse(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const int* __restrict__ edge_src,
    int* __restrict__ rev,
    int num_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    int u = edge_src[idx];
    int v = __ldg(&indices[idx]);
    int lo = __ldg(&offsets[v]);
    int hi = __ldg(&offsets[v + 1]);
    int pos = d_lower_bound(indices, lo, hi, u);
    rev[idx] = (pos < hi && __ldg(&indices[pos]) == u) ? pos : -1;
}

__global__ void mark_forward(
    const int* __restrict__ edge_src,
    const int* __restrict__ indices,
    int* __restrict__ is_fwd,
    int num_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    is_fwd[idx] = (edge_src[idx] < __ldg(&indices[idx])) ? 1 : 0;
}

__global__ void scatter_forward(
    const int* __restrict__ prefix_sum,
    const int* __restrict__ edge_src,
    const int* __restrict__ indices,
    int* __restrict__ fwd_pos,
    int num_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_edges) return;
    if (edge_src[idx] < __ldg(&indices[idx])) {
        fwd_pos[__ldg(&prefix_sum[idx])] = idx;
    }
}

__global__ void count_and_mark(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const int* __restrict__ edge_src,
    const int* __restrict__ fwd_pos,
    const int* __restrict__ rev,
    uint8_t* __restrict__ alive,
    int* __restrict__ removed_flag,
    int num_fwd_edges,
    int threshold
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_fwd_edges) return;

    int p = fwd_pos[warp_id];
    if (!alive[p]) return;

    int u = edge_src[p];
    int v = __ldg(&indices[p]);

    int u_start = __ldg(&offsets[u]);
    int u_end = __ldg(&offsets[u + 1]);
    int v_start = __ldg(&offsets[v]);
    int v_end = __ldg(&offsets[v + 1]);
    int u_len = u_end - u_start;
    int v_len = v_end - v_start;

    int s_start, s_len, l_start, l_end;
    if (u_len <= v_len) {
        s_start = u_start; s_len = u_len;
        l_start = v_start; l_end = v_end;
    } else {
        s_start = v_start; s_len = v_len;
        l_start = u_start; l_end = u_end;
    }

    int count = 0;
    bool survived = false;

    int num_batches = (s_len + 31) >> 5;

    for (int batch = 0; batch < num_batches; batch++) {
        int i = batch * 32 + lane;

        if (i < s_len) {
            int s_pos = s_start + i;
            if (alive[s_pos]) {
                int w = __ldg(&indices[s_pos]);
                if (w != u && w != v) {
                    int pos = d_lower_bound(indices, l_start, l_end, w);
                    if (pos < l_end && __ldg(&indices[pos]) == w && alive[pos]) {
                        count++;
                    }
                }
            }
        }

        if (__any_sync(0xffffffff, count >= threshold)) {
            survived = true;
            break;
        }
    }

    if (!survived) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            count += __shfl_down_sync(0xffffffff, count, offset);

        if (lane == 0) {
            if (count >= threshold) {
                survived = true;
            }
        }
        survived = __shfl_sync(0xffffffff, survived ? 1 : 0, 0);
    }

    if (!survived && lane == 0) {
        alive[p] = 0;
        int r = rev[p];
        if (r >= 0) alive[r] = 0;
        *removed_flag = 1;
    }
}

__global__ void flag_alive(
    const int* __restrict__ fwd_pos,
    const uint8_t* __restrict__ alive,
    int* __restrict__ flag,
    int num_fwd
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_fwd) return;
    flag[idx] = alive[fwd_pos[idx]] ? 1 : 0;
}

__global__ void compact_alive_edges(
    const int* __restrict__ fwd_pos_old,
    const uint8_t* __restrict__ alive,
    const int* __restrict__ prefix,
    int* __restrict__ fwd_pos_new,
    int num_fwd_old
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_fwd_old) return;
    int p = fwd_pos_old[idx];
    if (alive[p]) {
        fwd_pos_new[prefix[idx]] = p;
    }
}

__global__ void extract_edges(
    const int* __restrict__ fwd_pos,
    const int* __restrict__ edge_src,
    const int* __restrict__ indices,
    int* __restrict__ out_src,
    int* __restrict__ out_dst,
    int num_fwd_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_fwd_edges) return;
    int p = fwd_pos[idx];
    int base = idx * 2;
    int u = edge_src[p];
    int v = __ldg(&indices[p]);
    out_src[base] = u;
    out_dst[base] = v;
    out_src[base + 1] = v;
    out_dst[base + 1] = u;
}



static size_t get_cub_scan_temp_bytes(int n) {
    size_t temp = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp, (int*)nullptr, (int*)nullptr, n);
    return temp;
}

}  

k_truss_result_t k_truss_seg(const graph32_t& graph,
                             int32_t k) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int num_vertices = graph.number_of_vertices;
    int num_edges = graph.number_of_edges;
    int threshold = k - 2;
    cudaStream_t stream = 0;

    if (num_edges == 0) {
        return k_truss_result_t{nullptr, nullptr, 0};
    }

    cache.ensure_edge_buffers(num_edges);

    size_t cub_bytes = get_cub_scan_temp_bytes(num_edges);
    cub_bytes = std::max(cub_bytes, (size_t)256);
    cache.ensure_cub_temp(cub_bytes);

    int32_t* d_edge_src = cache.edge_src;
    int32_t* d_rev = cache.rev;
    int32_t* d_is_fwd = cache.is_fwd;
    int32_t* d_prefix = cache.prefix;
    uint8_t* d_alive = cache.alive;
    int32_t* d_removed = cache.removed;
    void* d_cub_temp = cache.cub_temp;

    
    {
        int b = 256, g = (num_edges + b - 1) / b;
        if (g > 0) compute_sources<<<g, b, 0, stream>>>(d_offsets, d_edge_src, num_vertices, num_edges);
    }

    
    {
        int b = 256, g = (num_edges + b - 1) / b;
        if (g > 0) compute_reverse<<<g, b, 0, stream>>>(d_offsets, d_indices, d_edge_src, d_rev, num_edges);
    }

    
    {
        int b = 256, g = (num_edges + b - 1) / b;
        if (g > 0) mark_forward<<<g, b, 0, stream>>>(d_edge_src, d_indices, d_is_fwd, num_edges);
    }
    cub::DeviceScan::ExclusiveSum(d_cub_temp, cub_bytes, d_is_fwd, d_prefix, num_edges, stream);

    int num_fwd = 0;
    {
        int lp, lf;
        cudaMemcpy(&lp, d_prefix + num_edges - 1, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&lf, d_is_fwd + num_edges - 1, sizeof(int), cudaMemcpyDeviceToHost);
        num_fwd = lp + lf;
    }

    if (num_fwd == 0) {
        return k_truss_result_t{nullptr, nullptr, 0};
    }

    int32_t* d_fwd_pos_a = cache.fwd_pos_a;
    int32_t* d_fwd_pos_b = cache.fwd_pos_b;
    int32_t* d_fwd_cur = d_fwd_pos_a;
    int32_t* d_fwd_next = d_fwd_pos_b;

    int32_t* d_flag = d_is_fwd;
    int32_t* d_fwd_prefix = d_prefix;

    {
        int b = 256, g = (num_edges + b - 1) / b;
        if (g > 0) scatter_forward<<<g, b, 0, stream>>>(d_prefix, d_edge_src, d_indices, d_fwd_cur, num_edges);
    }

    
    cudaMemsetAsync(d_alive, 1, num_edges, stream);

    
    int cur_fwd = num_fwd;

    for (;;) {
        if (cur_fwd == 0) break;

        cudaMemsetAsync(d_removed, 0, sizeof(int), stream);
        {
            int threads = 256;
            int warps_per_block = threads / 32;
            int g = (cur_fwd + warps_per_block - 1) / warps_per_block;
            if (g > 0) count_and_mark<<<g, threads, 0, stream>>>(d_offsets, d_indices, d_edge_src,
                d_fwd_cur, d_rev, d_alive, d_removed,
                cur_fwd, threshold);
        }

        int h_removed;
        cudaMemcpy(&h_removed, d_removed, sizeof(int), cudaMemcpyDeviceToHost);

        if (h_removed == 0) break;

        {
            int b = 256, g = (cur_fwd + b - 1) / b;
            if (g > 0) flag_alive<<<g, b, 0, stream>>>(d_fwd_cur, d_alive, d_flag, cur_fwd);
        }

        size_t cb = get_cub_scan_temp_bytes(cur_fwd);
        if (cb > cache.cub_capacity) {
            cache.ensure_cub_temp(cb);
            d_cub_temp = cache.cub_temp;
        }
        cub::DeviceScan::ExclusiveSum(d_cub_temp, cb, d_flag, d_fwd_prefix, cur_fwd, stream);

        int new_fwd = 0;
        {
            int lp, lf;
            cudaMemcpy(&lp, d_fwd_prefix + cur_fwd - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lf, d_flag + cur_fwd - 1, sizeof(int), cudaMemcpyDeviceToHost);
            new_fwd = lp + lf;
        }

        if (new_fwd == 0) {
            cur_fwd = 0;
            break;
        }

        {
            int b = 256, g = (cur_fwd + b - 1) / b;
            if (g > 0) compact_alive_edges<<<g, b, 0, stream>>>(d_fwd_cur, d_alive, d_fwd_prefix, d_fwd_next, cur_fwd);
        }

        int32_t* tmp = d_fwd_cur;
        d_fwd_cur = d_fwd_next;
        d_fwd_next = tmp;
        cur_fwd = new_fwd;
    }

    
    int64_t total_output = (int64_t)cur_fwd * 2;

    int32_t* out_src = nullptr;
    int32_t* out_dst = nullptr;
    if (total_output > 0) {
        cudaMalloc(&out_src, total_output * sizeof(int32_t));
        cudaMalloc(&out_dst, total_output * sizeof(int32_t));

        int b = 256, g = (cur_fwd + b - 1) / b;
        if (g > 0) extract_edges<<<g, b, 0, stream>>>(d_fwd_cur, d_edge_src, d_indices, out_src, out_dst, cur_fwd);
    }

    cudaStreamSynchronize(stream);

    return k_truss_result_t{out_src, out_dst, (std::size_t)total_output};
}

}  
