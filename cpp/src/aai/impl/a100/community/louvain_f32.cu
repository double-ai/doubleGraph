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
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <cstdint>

namespace aai {

namespace {





#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)
#define HT_SIZE_PER_WARP 128  // hash table entries per warp in shared memory

struct Cache : Cacheable {};

__global__ void init_sequence_kernel(int32_t* arr, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = idx;
}

__global__ void compute_weighted_degree_kernel(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ weights,
    float* __restrict__ k,
    int32_t n
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        float sum = 0.0f;
        for (int e = offsets[v]; e < offsets[v + 1]; e++) {
            sum += weights[e];
        }
        k[v] = sum;
    }
}

__global__ void compute_sigma_tot_kernel(
    const int32_t* __restrict__ community,
    const float* __restrict__ k,
    float* __restrict__ sigma_tot,
    int32_t n
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) {
        atomicAdd(&sigma_tot[community[v]], k[v]);
    }
}

__device__ inline uint32_t gpu_hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x45d9f3b;
    x ^= x >> 16;
    x *= 0x45d9f3b;
    x ^= x >> 16;
    return x;
}







__global__ void __launch_bounds__(BLOCK_SIZE, 6)
local_move_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    int32_t* __restrict__ community,
    const float* __restrict__ k,
    float* __restrict__ sigma_tot,
    float two_inv_m,
    float two_inv_m_sq,
    float resolution,
    int32_t n,
    int32_t* __restrict__ changed,
    uint32_t seed
) {
    
    __shared__ int32_t s_ht_keys[WARPS_PER_BLOCK * HT_SIZE_PER_WARP];
    __shared__ float s_ht_vals[WARPS_PER_BLOCK * HT_SIZE_PER_WARP];

    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int global_warp = (blockIdx.x * WARPS_PER_BLOCK) + warp_id;

    int v = global_warp;
    if (v >= n) return;

    int start = offsets[v];
    int end = offsets[v + 1];
    int degree = end - start;

    if (degree == 0) return;

    int32_t c_v = community[v];
    float k_v = k[v];
    float sigma_cv = sigma_tot[c_v];

    
    int32_t* my_keys = s_ht_keys + warp_id * HT_SIZE_PER_WARP;
    float* my_vals = s_ht_vals + warp_id * HT_SIZE_PER_WARP;

    
    for (int i = lane; i < HT_SIZE_PER_WARP; i += 32) {
        my_keys[i] = -1;
        my_vals[i] = 0.0f;
    }
    __syncwarp();

    
    for (int e = start + lane; e < end; e += 32) {
        int32_t u = indices[e];
        if (u == v) continue; 
        float w = weights[e];
        int32_t c_u = community[u];

        
        uint32_t h = (uint32_t)(c_u * 2654435761u) % HT_SIZE_PER_WARP;
        bool inserted = false;
        for (int probe = 0; probe < HT_SIZE_PER_WARP && !inserted; probe++) {
            uint32_t slot = (h + probe) % HT_SIZE_PER_WARP;
            int32_t old_key = atomicCAS(&my_keys[slot], -1, c_u);
            if (old_key == -1 || old_key == c_u) {
                
                atomicAdd(&my_vals[slot], w);
                inserted = true;
            }
        }
    }
    __syncwarp();

    
    if (lane == 0) {
        float w_to_cv = 0.0f;
        for (int i = 0; i < HT_SIZE_PER_WARP; i++) {
            if (my_keys[i] == c_v) { w_to_cv = my_vals[i]; break; }
        }

        
        uint32_t vh = gpu_hash((uint32_t)v ^ seed);
        int direction = vh & 1;

        float best_gain = 0.0f;
        int32_t best_c = c_v;

        for (int i = 0; i < HT_SIZE_PER_WARP; i++) {
            if (my_keys[i] == -1) continue;
            int32_t c = my_keys[i];
            if (c == c_v) continue;

            if (direction == 0 && c > c_v) continue;
            if (direction == 1 && c < c_v) continue;

            float w_to_c = my_vals[i];
            float sigma_c = sigma_tot[c];
            float gain = (w_to_c - w_to_cv) * two_inv_m +
                         resolution * k_v * (sigma_cv - k_v - sigma_c) * two_inv_m_sq;

            if (gain > best_gain) { best_gain = gain; best_c = c; }
            else if (gain == best_gain && gain > 0.0f && c < best_c) { best_c = c; }
        }

        if (best_c != c_v && best_gain > 0.0f) {
            community[v] = best_c;
            atomicAdd(&sigma_tot[c_v], -k_v);
            atomicAdd(&sigma_tot[best_c], k_v);
            *changed = 1;
        }
    }
}


__global__ void local_move_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    int32_t* __restrict__ community,
    const float* __restrict__ k,
    float* __restrict__ sigma_tot,
    float two_inv_m,
    float two_inv_m_sq,
    float resolution,
    int32_t n,
    int32_t* __restrict__ changed,
    uint32_t seed
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;

    int start = offsets[v];
    int end = offsets[v + 1];
    if (start == end) return;

    int32_t c_v = community[v];
    float k_v = k[v];
    float sigma_cv = sigma_tot[c_v];

    const int HT_SIZE = 64;
    int32_t ht_keys[HT_SIZE];
    float ht_vals[HT_SIZE];
    for (int i = 0; i < HT_SIZE; i++) { ht_keys[i] = -1; ht_vals[i] = 0.0f; }

    for (int e = start; e < end; e++) {
        int32_t u = indices[e];
        if (u == v) continue;
        float w = weights[e];
        int32_t c_u = community[u];

        uint32_t h = ((uint32_t)c_u * 2654435761u) % HT_SIZE;
        for (int probe = 0; probe < HT_SIZE; probe++) {
            uint32_t slot = (h + probe) % HT_SIZE;
            if (ht_keys[slot] == c_u) { ht_vals[slot] += w; break; }
            if (ht_keys[slot] == -1) { ht_keys[slot] = c_u; ht_vals[slot] = w; break; }
        }
    }

    float w_to_cv = 0.0f;
    for (int i = 0; i < HT_SIZE; i++) {
        if (ht_keys[i] == c_v) { w_to_cv = ht_vals[i]; break; }
    }

    uint32_t vh = gpu_hash((uint32_t)v ^ seed);
    int direction = vh & 1;

    float best_gain = 0.0f;
    int32_t best_c = c_v;

    for (int i = 0; i < HT_SIZE; i++) {
        if (ht_keys[i] == -1) continue;
        int32_t c = ht_keys[i];
        if (c == c_v) continue;
        if (direction == 0 && c > c_v) continue;
        if (direction == 1 && c < c_v) continue;

        float w_to_c = ht_vals[i];
        float sigma_c = sigma_tot[c];
        float gain = (w_to_c - w_to_cv) * two_inv_m +
                     resolution * k_v * (sigma_cv - k_v - sigma_c) * two_inv_m_sq;

        if (gain > best_gain) { best_gain = gain; best_c = c; }
        else if (gain == best_gain && gain > 0.0f && c < best_c) { best_c = c; }
    }

    if (best_c != c_v && best_gain > 0.0f) {
        community[v] = best_c;
        atomicAdd(&sigma_tot[c_v], -k_v);
        atomicAdd(&sigma_tot[best_c], k_v);
        *changed = 1;
    }
}


__global__ void local_move_serial(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    int32_t* __restrict__ community,
    const float* __restrict__ k,
    float* __restrict__ sigma_tot,
    float two_inv_m,
    float two_inv_m_sq,
    float resolution,
    int32_t n,
    int32_t* __restrict__ changed
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int v = 0; v < n; v++) {
        int start = offsets[v], end = offsets[v + 1];
        if (start == end) continue;
        int32_t c_v = community[v];
        float k_v = k[v], sigma_cv = sigma_tot[c_v];

        const int MAX_C = 512;
        int32_t cids[MAX_C]; float cw[MAX_C]; int nc = 0;
        for (int e = start; e < end; e++) {
            int32_t u = indices[e]; if (u == v) continue;
            float w = weights[e]; int32_t cu = community[u];
            int f = -1;
            for (int j = 0; j < nc; j++) if (cids[j] == cu) { f = j; break; }
            if (f >= 0) cw[f] += w;
            else if (nc < MAX_C) { cids[nc] = cu; cw[nc] = w; nc++; }
        }
        float w_cv = 0.0f;
        for (int j = 0; j < nc; j++) if (cids[j] == c_v) { w_cv = cw[j]; break; }
        float bg = 0.0f; int32_t bc = c_v;
        for (int j = 0; j < nc; j++) {
            int32_t c = cids[j]; if (c == c_v) continue;
            float g = (cw[j] - w_cv) * two_inv_m + resolution * k_v * (sigma_cv - k_v - sigma_tot[c]) * two_inv_m_sq;
            if (g > bg) { bg = g; bc = c; } else if (g == bg && g > 0.0f && c < bc) { bc = c; }
        }
        if (bc != c_v && bg > 0.0f) {
            community[v] = bc; sigma_tot[c_v] -= k_v; sigma_tot[bc] += k_v; *changed = 1;
        }
    }
}





__global__ void map_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ community,
    int32_t* __restrict__ edge_src,
    int32_t* __restrict__ edge_dst,
    float* __restrict__ edge_w,
    int32_t n
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int32_t c_v = community[v];
    for (int e = offsets[v]; e < offsets[v + 1]; e++) {
        edge_src[e] = c_v;
        edge_dst[e] = community[indices[e]];
        edge_w[e] = weights[e];
    }
}

__global__ void count_edges_kernel(const int32_t* __restrict__ src, int32_t* __restrict__ counts, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicAdd(&counts[src[idx] + 1], 1);
}

__global__ void update_mapping_kernel(int32_t* __restrict__ mapping, const int32_t* __restrict__ level_comm, int32_t n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) mapping[v] = level_comm[mapping[v]];
}

__global__ void mark_used_kernel(const int32_t* __restrict__ arr, int32_t* __restrict__ used, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) used[arr[idx]] = 1;
}

__global__ void apply_renumber_kernel(int32_t* __restrict__ arr, const int32_t* __restrict__ new_ids, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = new_ids[arr[idx]];
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
    (void)cache;

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const float* d_weights = edge_weights;
    int32_t* d_clusters = clusters;

    if (num_vertices <= 1 || num_edges == 0) {
        if (num_vertices >= 1) thrust::sequence(thrust::device, d_clusters, d_clusters + num_vertices);
        return {0, 0.0f};
    }

    float total_weight = thrust::reduce(
        thrust::device_pointer_cast(d_weights),
        thrust::device_pointer_cast(d_weights + num_edges), 0.0f);
    float two_inv_m = 2.0f / total_weight;
    float two_inv_m_sq = 2.0f / (total_weight * total_weight);

    int32_t cur_n = num_vertices, cur_edges = num_edges;
    const int32_t* cur_offsets = d_offsets;
    const int32_t* cur_indices = d_indices;
    const float* cur_weights = d_weights;

    int32_t* owned_offsets = nullptr, *owned_indices = nullptr;
    float* owned_weights = nullptr;

    int32_t* community; cudaMalloc(&community, num_vertices * sizeof(int32_t));
    float* kk;          cudaMalloc(&kk, num_vertices * sizeof(float));
    float* sigma_tot;   cudaMalloc(&sigma_tot, num_vertices * sizeof(float));
    int32_t* d_changed; cudaMalloc(&d_changed, sizeof(int32_t));

    init_sequence_kernel<<<(cur_n + 255) / 256, 256>>>(community, cur_n);
    compute_weighted_degree_kernel<<<(cur_n + 255) / 256, 256>>>(cur_offsets, cur_weights, kk, cur_n);
    cudaMemcpy(sigma_tot, kk, cur_n * sizeof(float), cudaMemcpyDeviceToDevice);
    thrust::sequence(thrust::device, d_clusters, d_clusters + num_vertices);

    int32_t* edge_src = nullptr, *edge_dst = nullptr; float* edge_w = nullptr;
    size_t edge_buf_cap = 0;
    int64_t level = 0;
    float prev_modularity = -1e30f;
    float* d_mod_tmp; cudaMalloc(&d_mod_tmp, 2 * sizeof(float));

    int64_t effective_max_level = (int64_t)max_level;
    for (int64_t lvl = 0; lvl < effective_max_level; lvl++) {
        bool level_changed = false;

        
        float cur_avg_deg = (float)cur_edges / (cur_n > 0 ? cur_n : 1);

        if (cur_n <= 200) {
            
            for (int iter = 0; iter < 100; iter++) {
                cudaMemset(d_changed, 0, sizeof(int32_t));
                local_move_serial<<<1, 1>>>(
                    cur_offsets, cur_indices, cur_weights, community, kk, sigma_tot,
                    two_inv_m, two_inv_m_sq, resolution, cur_n, d_changed);
                int32_t h; cudaMemcpy(&h, d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost);
                if (h) level_changed = true;
                if (!h) break;
            }
        } else {
            for (int iter = 0; iter < 100; iter++) {
                if (true) {
                    cudaMemset(sigma_tot, 0, cur_n * sizeof(float));
                    compute_sigma_tot_kernel<<<(cur_n + 255) / 256, 256>>>(
                        community, kk, sigma_tot, cur_n);
                }

                cudaMemset(d_changed, 0, sizeof(int32_t));
                uint32_t seed = (uint32_t)(iter * 0x9e3779b9 + lvl * 0x517cc1b7);

                if (cur_avg_deg >= 8) {
                    
                    int num_warps = cur_n;
                    int grid = (num_warps * 32 + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    local_move_warp<<<grid, BLOCK_SIZE>>>(
                        cur_offsets, cur_indices, cur_weights, community, kk, sigma_tot,
                        two_inv_m, two_inv_m_sq, resolution, cur_n, d_changed, seed);
                } else {
                    
                    local_move_thread<<<(cur_n + 255) / 256, 256>>>(
                        cur_offsets, cur_indices, cur_weights, community, kk, sigma_tot,
                        two_inv_m, two_inv_m_sq, resolution, cur_n, d_changed, seed);
                }

                int32_t h; cudaMemcpy(&h, d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost);
                if (h) level_changed = true;
                else break;
            }
        }

        if (!level_changed) break;

        
        {
            cudaMemset(sigma_tot, 0, cur_n * sizeof(float));
            compute_sigma_tot_kernel<<<(cur_n + 255) / 256, 256>>>(community, kk, sigma_tot, cur_n);
            cudaMemset(d_mod_tmp, 0, 2 * sizeof(float));
            const int32_t* t_off = cur_offsets; const int32_t* t_idx = cur_indices;
            const float* t_wt = cur_weights; const int32_t* t_cm = community;
            const float* t_st = sigma_tot; float* t_acc = d_mod_tmp;
            thrust::for_each(thrust::device,
                thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>((int)cur_n),
                [=] __device__ (int v) {
                    int32_t cv = t_cm[v]; float iw = 0.0f;
                    for (int e = t_off[v]; e < t_off[v + 1]; e++)
                        if (t_cm[t_idx[e]] == cv) iw += t_wt[e];
                    atomicAdd(&t_acc[0], iw);
                    float s = t_st[v];
                    atomicAdd(&t_acc[1], s * s);
                });
            float h_vals[2];
            cudaMemcpy(h_vals, d_mod_tmp, 2 * sizeof(float), cudaMemcpyDeviceToHost);
            float cur_mod = h_vals[0] / total_weight - resolution * h_vals[1] / (total_weight * total_weight);
            if (cur_mod - prev_modularity <= threshold) break;
            prev_modularity = cur_mod;
        }

        level = lvl + 1;

        
        int32_t* used_flags, *new_ids;
        cudaMalloc(&used_flags, cur_n * sizeof(int32_t));
        cudaMalloc(&new_ids, cur_n * sizeof(int32_t));
        cudaMemset(used_flags, 0, cur_n * sizeof(int32_t));
        mark_used_kernel<<<(cur_n + 255) / 256, 256>>>(community, used_flags, cur_n);
        thrust::exclusive_scan(thrust::device_pointer_cast(used_flags),
            thrust::device_pointer_cast(used_flags + cur_n), thrust::device_pointer_cast(new_ids));
        int32_t h_lu, h_ln;
        cudaMemcpy(&h_lu, used_flags + cur_n - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_ln, new_ids + cur_n - 1, sizeof(int32_t), cudaMemcpyDeviceToHost);
        int32_t new_n = h_ln + h_lu;
        apply_renumber_kernel<<<(cur_n + 255) / 256, 256>>>(community, new_ids, cur_n);
        update_mapping_kernel<<<(num_vertices + 255) / 256, 256>>>(d_clusters, community, num_vertices);
        cudaFree(used_flags); cudaFree(new_ids);

        if (new_n >= cur_n || new_n <= 1) break;

        
        if ((size_t)cur_edges > edge_buf_cap) {
            if (edge_src) cudaFree(edge_src);
            if (edge_dst) cudaFree(edge_dst);
            if (edge_w) cudaFree(edge_w);
            edge_buf_cap = cur_edges;
            cudaMalloc(&edge_src, edge_buf_cap * sizeof(int32_t));
            cudaMalloc(&edge_dst, edge_buf_cap * sizeof(int32_t));
            cudaMalloc(&edge_w, edge_buf_cap * sizeof(float));
        }
        map_edges_kernel<<<(cur_n + 255) / 256, 256>>>(
            cur_offsets, cur_indices, cur_weights, community, edge_src, edge_dst, edge_w, cur_n);

        thrust::device_ptr<int32_t> sp(edge_src), dp(edge_dst);
        thrust::device_ptr<float> wp(edge_w);
        auto kb = thrust::make_zip_iterator(thrust::make_tuple(sp, dp));
        auto ke = thrust::make_zip_iterator(thrust::make_tuple(sp + cur_edges, dp + cur_edges));
        thrust::sort_by_key(kb, ke, wp);

        int32_t* rs, *rd; float* rw;
        cudaMalloc(&rs, cur_edges * sizeof(int32_t));
        cudaMalloc(&rd, cur_edges * sizeof(int32_t));
        cudaMalloc(&rw, cur_edges * sizeof(float));
        auto ok = thrust::make_zip_iterator(thrust::make_tuple(
            thrust::device_pointer_cast(rs), thrust::device_pointer_cast(rd)));
        auto res = thrust::reduce_by_key(kb, ke, wp, ok, thrust::device_pointer_cast(rw));
        int32_t new_edges = res.first - ok;

        if (owned_offsets) cudaFree(owned_offsets);
        if (owned_indices) cudaFree(owned_indices);
        if (owned_weights) cudaFree(owned_weights);
        cudaMalloc(&owned_offsets, (new_n + 1) * sizeof(int32_t));
        cudaMalloc(&owned_indices, new_edges * sizeof(int32_t));
        cudaMalloc(&owned_weights, new_edges * sizeof(float));
        cudaMemcpy(owned_indices, rd, new_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(owned_weights, rw, new_edges * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemset(owned_offsets, 0, (new_n + 1) * sizeof(int32_t));
        count_edges_kernel<<<(new_edges + 255) / 256, 256>>>(rs, owned_offsets, new_edges);
        thrust::inclusive_scan(thrust::device_pointer_cast(owned_offsets),
            thrust::device_pointer_cast(owned_offsets + new_n + 1),
            thrust::device_pointer_cast(owned_offsets));
        cudaFree(rs); cudaFree(rd); cudaFree(rw);

        cur_offsets = owned_offsets; cur_indices = owned_indices;
        cur_weights = owned_weights; cur_n = new_n; cur_edges = new_edges;

        init_sequence_kernel<<<(cur_n + 255) / 256, 256>>>(community, cur_n);
        compute_weighted_degree_kernel<<<(cur_n + 255) / 256, 256>>>(cur_offsets, cur_weights, kk, cur_n);
        cudaMemcpy(sigma_tot, kk, cur_n * sizeof(float), cudaMemcpyDeviceToDevice);

        cur_avg_deg = (float)cur_edges / (cur_n > 0 ? cur_n : 1);
    }

    
    float* orig_k, *orig_sigma;
    cudaMalloc(&orig_k, num_vertices * sizeof(float));
    cudaMalloc(&orig_sigma, num_vertices * sizeof(float));
    compute_weighted_degree_kernel<<<(num_vertices + 255) / 256, 256>>>(d_offsets, d_weights, orig_k, num_vertices);
    cudaMemset(orig_sigma, 0, num_vertices * sizeof(float));
    compute_sigma_tot_kernel<<<(num_vertices + 255) / 256, 256>>>(d_clusters, orig_k, orig_sigma, num_vertices);

    int32_t max_comm = thrust::reduce(thrust::device_pointer_cast(d_clusters),
        thrust::device_pointer_cast(d_clusters + num_vertices), (int32_t)0, thrust::maximum<int32_t>());
    int32_t nc = max_comm + 1;
    float* d_internal; cudaMalloc(&d_internal, nc * sizeof(float));
    cudaMemset(d_internal, 0, nc * sizeof(float));

    const int32_t* cl = d_clusters;
    const int32_t* ofs = d_offsets;
    const int32_t* idx2 = d_indices;
    const float* wt = d_weights;
    float* di = d_internal;
    thrust::for_each(thrust::device, thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(num_vertices),
        [=] __device__ (int v) {
            int32_t cv = cl[v]; float sum = 0.0f;
            for (int e = ofs[v]; e < ofs[v + 1]; e++)
                if (cl[idx2[e]] == cv) sum += wt[e];
            atomicAdd(&di[cv], sum);
        });

    std::vector<float> h_internal(nc), h_sigma(nc);
    cudaMemcpy(h_internal.data(), d_internal, nc * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sigma.data(), orig_sigma, nc * sizeof(float), cudaMemcpyDeviceToHost);
    float Q = 0.0f;
    for (int c = 0; c < nc; c++)
        if (h_sigma[c] > 0.0f)
            Q += h_internal[c] / total_weight - resolution * (h_sigma[c] / total_weight) * (h_sigma[c] / total_weight);

    cudaFree(d_mod_tmp);
    cudaFree(community); cudaFree(kk); cudaFree(sigma_tot); cudaFree(d_changed);
    if (edge_src) cudaFree(edge_src); if (edge_dst) cudaFree(edge_dst); if (edge_w) cudaFree(edge_w);
    if (owned_offsets) cudaFree(owned_offsets); if (owned_indices) cudaFree(owned_indices); if (owned_weights) cudaFree(owned_weights);
    cudaFree(orig_k); cudaFree(orig_sigma); cudaFree(d_internal);

    return {(std::size_t)level, Q};
}

}  
