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
#include <cuda/std/functional>
#include <cstdint>
#include <algorithm>

namespace aai {

namespace {





#define BLOCK_SIZE 320
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)

#ifndef HT_SIZE_PER_WARP
#define HT_SIZE_PER_WARP 64
#endif
static_assert((HT_SIZE_PER_WARP & (HT_SIZE_PER_WARP - 1)) == 0, "HT_SIZE_PER_WARP must be power-of-two");

#define SEQ_HASH_CAP 128





struct Cache : Cacheable {
    float* orig_k = nullptr;
    int64_t orig_k_cap = 0;

    void* cub_temp = nullptr;
    size_t cub_temp_cap = 0;

    int32_t* changed = nullptr;

    float* tw = nullptr;

    float* sigma_tot = nullptr;
    int64_t sigma_tot_cap = 0;

    double* iw = nullptr;

    double* ssq = nullptr;

    void ensure(int32_t n, size_t cub_bytes) {
        if (orig_k_cap < n) {
            if (orig_k) cudaFree(orig_k);
            cudaMalloc(&orig_k, (size_t)n * sizeof(float));
            orig_k_cap = n;
        }
        if (sigma_tot_cap < n) {
            if (sigma_tot) cudaFree(sigma_tot);
            cudaMalloc(&sigma_tot, (size_t)n * sizeof(float));
            sigma_tot_cap = n;
        }
        if (cub_temp_cap < cub_bytes) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, cub_bytes);
            cub_temp_cap = cub_bytes;
        }
        if (!changed) cudaMalloc(&changed, sizeof(int32_t));
        if (!tw) cudaMalloc(&tw, sizeof(float));
        if (!iw) cudaMalloc(&iw, sizeof(double));
        if (!ssq) cudaMalloc(&ssq, sizeof(double));
    }

    ~Cache() override {
        if (orig_k) cudaFree(orig_k);
        if (cub_temp) cudaFree(cub_temp);
        if (changed) cudaFree(changed);
        if (tw) cudaFree(tw);
        if (sigma_tot) cudaFree(sigma_tot);
        if (iw) cudaFree(iw);
        if (ssq) cudaFree(ssq);
    }
};





__device__ __forceinline__ uint32_t hash32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}





__global__ void init_communities_kernel(int32_t* __restrict__ comm, int32_t n) {
    int32_t v = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (v < n) comm[v] = v;
}

__global__ void compute_vertex_weights_kernel(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ w,
    float* __restrict__ k,
    int32_t n) {
    int32_t v = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (v >= n) return;
    float sum = 0.0f;
    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    for (int32_t e = start; e < end; e++) sum += w[e];
    k[v] = sum;
}

__global__ void reduce_sum_kernel(const float* __restrict__ in, float* __restrict__ out, int32_t n) {
    using BR = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BR::TempStorage tmp;
    float s = 0.0f;
    for (int32_t i = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x); i < n; i += (int32_t)(blockDim.x * gridDim.x)) {
        s += in[i];
    }
    float bs = BR(tmp).Sum(s);
    if (threadIdx.x == 0) atomicAdd(out, bs);
}





__global__ __launch_bounds__(BLOCK_SIZE, 4)
void local_move_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ w,
    int32_t* __restrict__ community,
    const float* __restrict__ k,
    float* __restrict__ sigma_tot,
    float two_inv_m,
    float two_inv_m_sq,
    float resolution,
    float move_threshold,
    int32_t n,
    int32_t* __restrict__ changed,
    uint32_t seed) {

    __shared__ int32_t s_keys[WARPS_PER_BLOCK * HT_SIZE_PER_WARP];
    __shared__ float s_vals[WARPS_PER_BLOCK * HT_SIZE_PER_WARP];

    int warp_id = (int)(threadIdx.x >> 5);
    int lane = (int)(threadIdx.x & 31);
    int32_t v = (int32_t)(blockIdx.x * WARPS_PER_BLOCK + warp_id);
    if (v >= n) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    if (start == end) return;

    int32_t* keys = s_keys + warp_id * HT_SIZE_PER_WARP;
    float* vals = s_vals + warp_id * HT_SIZE_PER_WARP;

    
    for (int i = lane; i < HT_SIZE_PER_WARP; i += 32) {
        keys[i] = -1;
        vals[i] = 0.0f;
    }
    __syncwarp();

    
    int32_t cv = community[v];

    
    float w_to_cv_lane = 0.0f;
    for (int32_t e = start + lane; e < end; e += 32) {
        int32_t u = indices[e];
        if (u == v) continue; 
        float we = w[e];
        int32_t cu = community[u];
        if (cu == cv) {
            w_to_cv_lane += we;
            continue;
        }
        uint32_t slot = ((uint32_t)cu * 2654435761u) & (HT_SIZE_PER_WARP - 1);

        #pragma unroll 1
        for (int probe = 0; probe < HT_SIZE_PER_WARP; probe++) {
            uint32_t pos = (slot + (uint32_t)probe) & (HT_SIZE_PER_WARP - 1);
            int32_t old = atomicCAS(&keys[pos], -1, cu);
            if (old == -1 || old == cu) {
                atomicAdd(&vals[pos], we);
                break;
            }
        }
    }
    __syncwarp();

    
    float w_to_cv = w_to_cv_lane;
    for (int off = 16; off > 0; off >>= 1) w_to_cv += __shfl_down_sync(0xffffffff, w_to_cv, off);
    w_to_cv = __shfl_sync(0xffffffff, w_to_cv, 0);

    
    float kv = k[v];
    float sigma_cv = sigma_tot[cv];

    
    uint32_t dir = hash32(((uint32_t)v) ^ seed) & 1u;

if (lane == 0) {
        float base_gain_bias = -w_to_cv * two_inv_m;
        float best = 0.0f;
        int32_t bc = cv;
        for (int i = 0; i < HT_SIZE_PER_WARP; i++) {
            int32_t c = keys[i];
            if (c == -1 || c == cv) continue;
            if (dir == 0u) { if (c > cv) continue; } else { if (c < cv) continue; }
            float wc = vals[i];
            float sigma_c = sigma_tot[c];
            float gain = wc * two_inv_m + base_gain_bias + resolution * kv * (sigma_cv - kv - sigma_c) * two_inv_m_sq;
            if (gain > best || (gain == best && gain > 0.0f && c < bc)) {
                best = gain;
                bc = c;
            }
        }

        if (bc != cv && best > move_threshold) {
            community[v] = bc;
            atomicAdd(&sigma_tot[cv], -kv);
            atomicAdd(&sigma_tot[bc], kv);
            atomicExch((int*)changed, 1);
        }
    }
}





__device__ bool seq_process_vertex(
    int32_t v,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ w,
    int32_t* __restrict__ community,
    float* __restrict__ sigma_tot,
    const float* __restrict__ k,
    float total_weight,
    float resolution,
    float move_threshold,
    int32_t* hkeys,
    float* hvals) {

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];
    if (start == end) return false;

    int32_t cv = community[v];
    double kv = (double)k[v];

    #pragma unroll
    for (int i = 0; i < SEQ_HASH_CAP; i++) { hkeys[i] = -1; hvals[i] = 0.0f; }

    for (int32_t e = start; e < end; e++) {
        int32_t u = indices[e];
        if (u == v) continue;
        int32_t cu = community[u];
        float we = w[e];
        uint32_t slot = ((uint32_t)cu * 2654435761u) & (SEQ_HASH_CAP - 1);
        for (int p = 0; p < SEQ_HASH_CAP; p++) {
            int pos = (int)((slot + (uint32_t)p) & (SEQ_HASH_CAP - 1));
            if (hkeys[pos] == cu) { hvals[pos] += we; break; }
            if (hkeys[pos] == -1) { hkeys[pos] = cu; hvals[pos] = we; break; }
        }
    }

    double w_to_cv = 0.0;
    for (int i = 0; i < SEQ_HASH_CAP; i++) if (hkeys[i] == cv) { w_to_cv = (double)hvals[i]; break; }

    double m2 = (double)total_weight; 
    double two_inv_m = 2.0 / m2;
    double two_inv_m_sq = 2.0 / (m2 * m2);

    double sigma_cv = (double)sigma_tot[cv];
    double best = 0.0;
    int32_t best_c = cv;

    for (int i = 0; i < SEQ_HASH_CAP; i++) {
        int32_t c = hkeys[i];
        if (c == -1 || c == cv) continue;
        double wc = (double)hvals[i];
        double sigma_c = (double)sigma_tot[c];
        double gain = (wc - w_to_cv) * two_inv_m + (double)resolution * kv * (sigma_cv - kv - sigma_c) * two_inv_m_sq;
        if (gain > best || (gain == best && gain > 0.0 && c < best_c)) {
            best = gain;
            best_c = c;
        }
    }

    if (best_c != cv && best > (double)move_threshold) {
        float kvf = k[v];
        sigma_tot[cv] -= kvf;
        sigma_tot[best_c] += kvf;
        community[v] = best_c;
        return true;
    }
    return false;
}

__global__ void sequential_local_move_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ w,
    int32_t* __restrict__ community,
    float* __restrict__ sigma_tot,
    const float* __restrict__ k,
    int32_t n,
    float total_weight,
    float resolution,
    float move_threshold,
    int32_t* __restrict__ changed,
    int start_offset) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int32_t hkeys[SEQ_HASH_CAP];
    float hvals[SEQ_HASH_CAP];

    bool any = false;
    for (int32_t i = 0; i < n; i++) {
        int32_t v = (i + start_offset) % n;
        if (seq_process_vertex(v, offsets, indices, w, community, sigma_tot, k, total_weight, resolution, move_threshold, hkeys, hvals)) any = true;
    }
    if (any) *changed = 1;
}





__global__ void mark_communities_kernel(const int32_t* __restrict__ community, int* __restrict__ used, int32_t n) {
    int32_t v = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (v < n) used[community[v]] = 1;
}

__global__ void apply_renumber_kernel(int32_t* __restrict__ community, const int* __restrict__ mapping, int32_t n) {
    int32_t v = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (v < n) community[v] = mapping[community[v]];
}

__global__ void map_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ community,
    int64_t* __restrict__ edge_keys,
    int32_t n) {
    int32_t v = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (v >= n) return;
    uint32_t cv = (uint32_t)community[v];
    for (int32_t e = offsets[v]; e < offsets[v + 1]; e++) {
        uint32_t cu = (uint32_t)community[indices[e]];
        edge_keys[e] = (int64_t(cv) << 32) | int64_t(cu);
    }
}

__global__ void count_src_edges_kernel(const int64_t* __restrict__ keys, int* __restrict__ counts, int32_t e) {
    int32_t i = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= e) return;
    int32_t src = (int32_t)(keys[i] >> 32);
    atomicAdd(&counts[src], 1);
}

__global__ void fill_csr_kernel(
    const int64_t* __restrict__ keys,
    const float* __restrict__ vals,
    int32_t* __restrict__ indices,
    float* __restrict__ weights,
    int32_t e) {
    int32_t i = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= e) return;
    indices[i] = (int32_t)(keys[i] & 0xffffffff);
    weights[i] = vals[i];
}

__global__ void compute_new_vw_kernel(
    const float* __restrict__ old_k,
    const int32_t* __restrict__ community,
    float* __restrict__ new_k,
    int32_t old_n) {
    int32_t v = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (v >= old_n) return;
    atomicAdd(&new_k[community[v]], old_k[v]);
}

__global__ void compose_communities_kernel(int32_t* __restrict__ final_comm, const int32_t* __restrict__ level_comm, int32_t orig_n) {
    int32_t v = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (v < orig_n) final_comm[v] = level_comm[final_comm[v]];
}





__global__ void recompute_sigma_tot_kernel(
    const int32_t* __restrict__ comm,
    const float* __restrict__ k,
    float* __restrict__ sigma,
    int32_t n) {
    int32_t v = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x);
    if (v < n) atomicAdd(&sigma[comm[v]], k[v]);
}

__global__ void modularity_iw_kernel(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ idx,
    const float* __restrict__ w,
    const int32_t* __restrict__ c,
    int32_t n,
    double* __restrict__ out) {

    using BR = cub::BlockReduce<double, BLOCK_SIZE>;
    __shared__ typename BR::TempStorage tmp;
    double sum = 0.0;

    for (int32_t v = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x); v < n; v += (int32_t)(blockDim.x * gridDim.x)) {
        int32_t cv = c[v];
        for (int32_t e = off[v]; e < off[v + 1]; e++) {
            int32_t u = idx[e];
            if (c[u] == cv) sum += (double)w[e];
        }
    }

    double block_sum = BR(tmp).Sum(sum);
    if (threadIdx.x == 0) atomicAdd(out, block_sum);
}

__global__ void sigma_sq_kernel(const float* __restrict__ sigma, int32_t n, double* __restrict__ out) {
    using BR = cub::BlockReduce<double, BLOCK_SIZE>;
    __shared__ typename BR::TempStorage tmp;
    double sum = 0.0;
    for (int32_t i = (int32_t)(blockIdx.x * blockDim.x + threadIdx.x); i < n; i += (int32_t)(blockDim.x * gridDim.x)) {
        double v = (double)sigma[i];
        sum += v * v;
    }
    double block_sum = BR(tmp).Sum(sum);
    if (threadIdx.x == 0) atomicAdd(out, block_sum);
}





static size_t cub_temp_bytes_scan_int(int32_t n) {
    size_t temp = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp, (int*)nullptr, (int*)nullptr, n);
    return temp;
}

static size_t cub_temp_bytes_sort_reduce_edges(int32_t e) {
    size_t sort_temp = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sort_temp,
                                   (int64_t*)nullptr, (int64_t*)nullptr,
                                   (float*)nullptr, (float*)nullptr,
                                   e);
    size_t red_temp = 0;
    cub::DeviceReduce::ReduceByKey(nullptr, red_temp,
                                   (int64_t*)nullptr, (int64_t*)nullptr,
                                   (float*)nullptr, (float*)nullptr,
                                   (int*)nullptr,
                                   ::cuda::std::plus<float>{},
                                   e);
    return sort_temp > red_temp ? sort_temp : red_temp;
}





static void launch_compute_vertex_weights(const int32_t* offsets, const float* weights, float* vw, int N) {
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_vertex_weights_kernel<<<grid, BLOCK_SIZE>>>(offsets, weights, vw, N);
}

static void launch_init_communities(int32_t* comm, int N) {
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_communities_kernel<<<grid, BLOCK_SIZE>>>(comm, N);
}

static void launch_reduce_sum(const float* input, float* output, int N) {
    cudaMemset(output, 0, sizeof(float));
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 2048) grid = 2048;
    reduce_sum_kernel<<<grid, BLOCK_SIZE>>>(input, output, N);
}

static void launch_local_move_warp(
    const int32_t* offsets,
    const int32_t* indices,
    const float* weights,
    int32_t* community,
    const float* k,
    float* sigma_tot,
    float two_inv_m,
    float two_inv_m_sq,
    float resolution,
    float move_threshold,
    int32_t n,
    int32_t* changed,
    uint32_t seed) {

    int grid = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    local_move_warp_kernel<<<grid, BLOCK_SIZE>>>(
        offsets, indices, weights, community, k, sigma_tot,
        two_inv_m, two_inv_m_sq, resolution, move_threshold, n, changed, seed);
}

static void launch_sequential_local_move(
    const int32_t* offsets,
    const int32_t* indices,
    const float* weights,
    int32_t* community,
    float* sigma_tot,
    const float* k,
    int32_t n,
    float total_weight,
    float resolution,
    float move_threshold,
    int32_t* changed,
    int start_offset) {
    sequential_local_move_kernel<<<1, 1>>>(offsets, indices, weights, community, sigma_tot, k, n,
                                          total_weight, resolution, move_threshold, changed, start_offset);
}

static int launch_renumber_communities(void* cub_temp, size_t cub_temp_size, int32_t* community, int* used, int* prefix, int32_t n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMemset(used, 0, (size_t)n * sizeof(int));
    mark_communities_kernel<<<grid, BLOCK_SIZE>>>(community, used, n);

    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, used, prefix, n);
    if (temp_bytes > cub_temp_size) return -1;
    cub::DeviceScan::ExclusiveSum(cub_temp, temp_bytes, used, prefix, n);

    int h_last_prefix = 0;
    int h_last_used = 0;
    cudaMemcpy(&h_last_prefix, prefix + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_last_used, used + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
    int new_n = h_last_prefix + h_last_used;
    if (new_n <= 0) new_n = 1;

    apply_renumber_kernel<<<grid, BLOCK_SIZE>>>(community, prefix, n);
    return new_n;
}

static void launch_compose_communities(int32_t* final_comm, const int32_t* level_comm, int32_t orig_n) {
    int grid = (orig_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compose_communities_kernel<<<grid, BLOCK_SIZE>>>(final_comm, level_comm, orig_n);
}

static void launch_map_edges(const int32_t* offsets, const int32_t* indices, const int32_t* community, int64_t* edge_keys, int32_t n) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    map_edges_kernel<<<grid, BLOCK_SIZE>>>(offsets, indices, community, edge_keys, n);
}

static int launch_sort_reduce_edges(
    void* cub_temp,
    size_t cub_temp_size,
    int64_t* keys_in,
    float* vals_in,
    int32_t e,
    int64_t* sort_keys_out,
    float* sort_vals_out,
    int64_t* unique_keys_out,
    float* unique_vals_out,
    int* d_num_unique) {

    size_t sort_temp = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sort_temp, keys_in, sort_keys_out, vals_in, sort_vals_out, e);
    if (sort_temp > cub_temp_size) return -1;
    cub::DeviceRadixSort::SortPairs(cub_temp, sort_temp, keys_in, sort_keys_out, vals_in, sort_vals_out, e);

    size_t red_temp = 0;
    cub::DeviceReduce::ReduceByKey(nullptr, red_temp,
        sort_keys_out, unique_keys_out,
        sort_vals_out, unique_vals_out,
        d_num_unique,
        ::cuda::std::plus<float>{},
        e);
    if (red_temp > cub_temp_size) return -1;
    cub::DeviceReduce::ReduceByKey(cub_temp, red_temp,
        sort_keys_out, unique_keys_out,
        sort_vals_out, unique_vals_out,
        d_num_unique,
        ::cuda::std::plus<float>{},
        e);

    int h_num_unique = 0;
    cudaMemcpy(&h_num_unique, d_num_unique, sizeof(int), cudaMemcpyDeviceToHost);
    return h_num_unique;
}

static void launch_build_csr(
    void* cub_temp,
    size_t cub_temp_size,
    const int64_t* unique_keys,
    const float* unique_vals,
    int32_t new_e,
    int32_t new_n,
    int* counts,
    int32_t* new_offsets,
    int32_t* new_indices,
    float* new_weights) {

    cudaMemset(counts, 0, (size_t)new_n * sizeof(int));
    int grid_e = (new_e + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_src_edges_kernel<<<grid_e, BLOCK_SIZE>>>(unique_keys, counts, new_e);

    size_t scan_temp = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, scan_temp, counts, (int*)new_offsets, new_n);
    if (scan_temp > cub_temp_size) return;
    cub::DeviceScan::ExclusiveSum(cub_temp, scan_temp, counts, (int*)new_offsets, new_n);

    cudaMemcpy(new_offsets + new_n, &new_e, sizeof(int32_t), cudaMemcpyHostToDevice);

    fill_csr_kernel<<<grid_e, BLOCK_SIZE>>>(unique_keys, unique_vals, new_indices, new_weights, new_e);
}

static void launch_compute_new_vw(const float* old_vw, const int32_t* community, float* new_vw, int32_t old_n, int32_t new_n) {
    cudaMemset(new_vw, 0, (size_t)new_n * sizeof(float));
    int grid = (old_n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_new_vw_kernel<<<grid, BLOCK_SIZE>>>(old_vw, community, new_vw, old_n);
}

static void launch_recompute_sigma_tot(const int32_t* community, const float* vw, float* sigma_tot, int32_t n) {
    cudaMemset(sigma_tot, 0, (size_t)n * sizeof(float));
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    recompute_sigma_tot_kernel<<<grid, BLOCK_SIZE>>>(community, vw, sigma_tot, n);
}

static void launch_modularity_accum(
    const int32_t* offsets,
    const int32_t* indices,
    const float* weights,
    const int32_t* clusters,
    int32_t n,
    double* d_iw_out) {

    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 2048) grid = 2048;
    modularity_iw_kernel<<<grid, BLOCK_SIZE>>>(offsets, indices, weights, clusters, n, d_iw_out);
}

static void launch_sigma_sq_accum(const float* sigma_tot, int32_t n, double* d_out) {
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 2048) grid = 2048;
    sigma_sq_kernel<<<grid, BLOCK_SIZE>>>(sigma_tot, n, d_out);
}

}  





louvain_result_float_t louvain(const graph32_t& graph,
                               const float* edge_weights,
                               int32_t* clusters,
                               std::size_t max_level,
                               float threshold,
                               float resolution) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t orig_n = graph.number_of_vertices;
    int32_t orig_e = graph.number_of_edges;

    if (orig_n <= 1 || orig_e == 0) {
        launch_init_communities(clusters, orig_n);
        cudaDeviceSynchronize();
        return {0, 0.0f};
    }

    
    size_t scan_bytes = cub_temp_bytes_scan_int(orig_n);
    size_t sortred_bytes = cub_temp_bytes_sort_reduce_edges(orig_e);
    size_t cub_bytes = std::max(scan_bytes, sortred_bytes);
    cub_bytes = std::min<size_t>(cub_bytes, size_t(512) << 20);
    cub_bytes = std::max<size_t>(cub_bytes, size_t(1) << 20);

    cache.ensure(orig_n, cub_bytes);

    
    launch_compute_vertex_weights(graph.offsets, edge_weights, cache.orig_k, orig_n);

    
    launch_reduce_sum(cache.orig_k, cache.tw, orig_n);
    float total_weight = 0.0f;
    cudaMemcpy(&total_weight, cache.tw, sizeof(float), cudaMemcpyDeviceToHost);

    if (!(total_weight > 0.0f)) {
        launch_init_communities(clusters, orig_n);
        cudaDeviceSynchronize();
        return {0, 0.0f};
    }

    float two_inv_m = 2.0f / total_weight;
    float two_inv_m_sq = 2.0f / (total_weight * total_weight);

    
    const int32_t* cur_off = graph.offsets;
    const int32_t* cur_idx = graph.indices;
    const float* cur_w = edge_weights;
    const float* cur_k = cache.orig_k;
    int32_t cur_n = orig_n;
    int32_t cur_e = orig_e;

    
    int32_t* owned_off = nullptr;
    int32_t* owned_idx = nullptr;
    float* owned_w = nullptr;
    float* owned_k = nullptr;

    
    launch_init_communities(clusters, orig_n);

    int64_t levels_done = 0;
    constexpr int MAX_PASSES = 1000;

    for (int64_t level = 0; level < (int64_t)max_level; level++) {
        
        int32_t* d_comm;
        float* d_sigma;
        cudaMalloc(&d_comm, (size_t)cur_n * sizeof(int32_t));
        cudaMalloc(&d_sigma, (size_t)cur_n * sizeof(float));

        launch_init_communities(d_comm, cur_n);
        cudaMemcpy(d_sigma, cur_k, (size_t)cur_n * sizeof(float), cudaMemcpyDeviceToDevice);

        if (cur_n <= 256) {
            for (int it = 0; it < MAX_PASSES; it++) {
                cudaMemset(cache.changed, 0, sizeof(int32_t));
                launch_sequential_local_move(
                    cur_off, cur_idx, cur_w,
                    d_comm, d_sigma, cur_k,
                    cur_n, total_weight, resolution, threshold,
                    cache.changed, it);
                int32_t h_changed = 0;
                cudaMemcpy(&h_changed, cache.changed, sizeof(int32_t), cudaMemcpyDeviceToHost);
                if (h_changed == 0) break;
            }
        } else {
            for (int it = 0; it < MAX_PASSES; it++) {
                cudaMemset(cache.changed, 0, sizeof(int32_t));
                uint32_t seed = uint32_t((level + 1) * 1315423911u + (it + 1) * 2654435761u);
                launch_local_move_warp(
                    cur_off, cur_idx, cur_w,
                    d_comm, cur_k, d_sigma,
                    two_inv_m, two_inv_m_sq, resolution, threshold,
                    cur_n, cache.changed, seed);
                int32_t h_changed = 0;
                cudaMemcpy(&h_changed, cache.changed, sizeof(int32_t), cudaMemcpyDeviceToHost);
                if (h_changed == 0) break;
            }
        }

        
        int* d_used;
        int* d_prefix;
        cudaMalloc(&d_used, (size_t)cur_n * sizeof(int));
        cudaMalloc(&d_prefix, (size_t)cur_n * sizeof(int));
        int32_t new_n = launch_renumber_communities(
            cache.cub_temp, cache.cub_temp_cap,
            d_comm, d_used, d_prefix, cur_n);

        
        if (levels_done == 0) {
            cudaMemcpy(clusters, d_comm, (size_t)orig_n * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        } else {
            launch_compose_communities(clusters, d_comm, orig_n);
        }

        levels_done++;

        if (new_n >= cur_n || levels_done >= (int64_t)max_level) {
            cudaFree(d_comm);
            cudaFree(d_sigma);
            cudaFree(d_used);
            cudaFree(d_prefix);
            break;
        }

        

        
        int64_t* edge_keys_buf;
        float* edge_vals_buf;
        cudaMalloc(&edge_keys_buf, (size_t)cur_e * sizeof(int64_t));
        cudaMalloc(&edge_vals_buf, (size_t)cur_e * sizeof(float));

        launch_map_edges(cur_off, cur_idx, d_comm, edge_keys_buf, cur_n);
        cudaMemcpy(edge_vals_buf, cur_w, (size_t)cur_e * sizeof(float), cudaMemcpyDeviceToDevice);

        
        int64_t* sort_keys_buf;
        float* sort_vals_buf;
        int64_t* uniq_keys_buf;
        float* uniq_vals_buf;
        int* d_num_uniq;
        cudaMalloc(&sort_keys_buf, (size_t)cur_e * sizeof(int64_t));
        cudaMalloc(&sort_vals_buf, (size_t)cur_e * sizeof(float));
        cudaMalloc(&uniq_keys_buf, (size_t)cur_e * sizeof(int64_t));
        cudaMalloc(&uniq_vals_buf, (size_t)cur_e * sizeof(float));
        cudaMalloc(&d_num_uniq, sizeof(int));

        int32_t new_e = launch_sort_reduce_edges(
            cache.cub_temp, cache.cub_temp_cap,
            edge_keys_buf, edge_vals_buf, cur_e,
            sort_keys_buf, sort_vals_buf,
            uniq_keys_buf, uniq_vals_buf, d_num_uniq);

        
        int32_t* next_off;
        int32_t* next_idx;
        float* next_w;
        int* d_counts;
        cudaMalloc(&next_off, (size_t)(new_n + 1) * sizeof(int32_t));
        cudaMalloc(&next_idx, (size_t)new_e * sizeof(int32_t));
        cudaMalloc(&next_w, (size_t)new_e * sizeof(float));
        cudaMalloc(&d_counts, (size_t)new_n * sizeof(int));

        launch_build_csr(
            cache.cub_temp, cache.cub_temp_cap,
            uniq_keys_buf, uniq_vals_buf,
            new_e, new_n,
            d_counts, next_off, next_idx, next_w);

        
        float* next_k;
        cudaMalloc(&next_k, (size_t)new_n * sizeof(float));
        launch_compute_new_vw(cur_k, d_comm, next_k, cur_n, new_n);

        
        cudaFree(d_comm);
        cudaFree(d_sigma);
        cudaFree(d_used);
        cudaFree(d_prefix);

        
        cudaFree(edge_keys_buf);
        cudaFree(edge_vals_buf);
        cudaFree(sort_keys_buf);
        cudaFree(sort_vals_buf);
        cudaFree(uniq_keys_buf);
        cudaFree(uniq_vals_buf);
        cudaFree(d_num_uniq);
        cudaFree(d_counts);

        
        if (owned_off) {
            cudaFree(owned_off);
            cudaFree(owned_idx);
            cudaFree(owned_w);
            cudaFree(owned_k);
        }

        
        owned_off = next_off;
        owned_idx = next_idx;
        owned_w = next_w;
        owned_k = next_k;

        cur_off = owned_off;
        cur_idx = owned_idx;
        cur_w = owned_w;
        cur_k = owned_k;
        cur_n = new_n;
        cur_e = new_e;
    }

    
    launch_recompute_sigma_tot(clusters, cache.orig_k, cache.sigma_tot, orig_n);

    cudaMemset(cache.iw, 0, sizeof(double));
    cudaMemset(cache.ssq, 0, sizeof(double));

    launch_modularity_accum(graph.offsets, graph.indices, edge_weights,
                            clusters, orig_n, cache.iw);
    launch_sigma_sq_accum(cache.sigma_tot, orig_n, cache.ssq);

    double iw_val = 0.0;
    double ssq_val = 0.0;
    cudaMemcpy(&iw_val, cache.iw, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ssq_val, cache.ssq, sizeof(double), cudaMemcpyDeviceToHost);

    double tw_d = (double)total_weight;
    float modularity = (float)(iw_val / tw_d - (double)resolution * ssq_val / (tw_d * tw_d));

    
    if (owned_off) {
        cudaFree(owned_off);
        cudaFree(owned_idx);
        cudaFree(owned_w);
        cudaFree(owned_k);
    }

    return {(std::size_t)levels_done, modularity};
}

}  
