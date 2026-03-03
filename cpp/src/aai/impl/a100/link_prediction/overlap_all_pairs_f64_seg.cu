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
#include <optional>

namespace aai {

namespace {

struct Cache : Cacheable {
    int64_t* total_count = nullptr;
    int32_t* write_counter = nullptr;
    int32_t* num_unique = nullptr;

    double* weighted_deg = nullptr;
    int64_t weighted_deg_cap = 0;

    int32_t* gen_seeds = nullptr;
    int64_t gen_seeds_cap = 0;

    Cache() {
        cudaMalloc(&total_count, sizeof(int64_t));
        cudaMalloc(&write_counter, sizeof(int32_t));
        cudaMalloc(&num_unique, sizeof(int32_t));
    }

    void ensure_weighted_deg(int64_t n) {
        if (weighted_deg_cap < n) {
            if (weighted_deg) cudaFree(weighted_deg);
            cudaMalloc(&weighted_deg, n * sizeof(double));
            weighted_deg_cap = n;
        }
    }

    void ensure_gen_seeds(int64_t n) {
        if (gen_seeds_cap < n) {
            if (gen_seeds) cudaFree(gen_seeds);
            cudaMalloc(&gen_seeds, n * sizeof(int32_t));
            gen_seeds_cap = n;
        }
    }

    ~Cache() override {
        if (total_count) cudaFree(total_count);
        if (write_counter) cudaFree(write_counter);
        if (num_unique) cudaFree(num_unique);
        if (weighted_deg) cudaFree(weighted_deg);
        if (gen_seeds) cudaFree(gen_seeds);
    }
};




__global__ void compute_weighted_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ edge_weights,
    double* __restrict__ weighted_degrees,
    int32_t num_vertices
) {
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < num_vertices;
         v += blockDim.x * gridDim.x) {
        int start = offsets[v];
        int end = offsets[v + 1];
        double sum = 0.0;
        for (int i = start; i < end; i++) sum += edge_weights[i];
        weighted_degrees[v] = sum;
    }
}




__global__ void count_twohop_paths_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ total_count
) {
    int64_t local_count = 0;
    for (int s = blockIdx.x; s < num_seeds; s += gridDim.x) {
        int32_t u = seeds[s];
        int u_start = offsets[u];
        int u_end = offsets[u + 1];
        for (int ci = u_start + threadIdx.x; ci < u_end; ci += blockDim.x) {
            int32_t c = indices[ci];
            local_count += offsets[c + 1] - offsets[c] - 1;
        }
    }
    typedef cub::BlockReduce<int64_t, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    int64_t s = BR(tmp).Sum(local_count);
    if (threadIdx.x == 0 && s > 0)
        atomicAdd((unsigned long long*)total_count, (unsigned long long)s);
}




__global__ void generate_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int32_t num_vertices,
    int64_t* __restrict__ pair_keys,
    int32_t* __restrict__ write_counter,
    int64_t max_pairs
) {
    for (int s = blockIdx.x; s < num_seeds; s += gridDim.x) {
        int32_t u = seeds[s];
        int u_start = offsets[u];
        int u_end = offsets[u + 1];
        for (int ci = u_start + threadIdx.x; ci < u_end; ci += blockDim.x) {
            int32_t c = indices[ci];
            int c_start = offsets[c];
            int c_end = offsets[c + 1];

            
            int count = 0;
            for (int vi = c_start; vi < c_end; vi++) {
                if (indices[vi] != u) count++;
            }
            if (count == 0) continue;

            int base = atomicAdd(write_counter, count);
            if ((int64_t)base + count > max_pairs) continue; 

            int offset = 0;
            for (int vi = c_start; vi < c_end; vi++) {
                int32_t v = indices[vi];
                if (v != u) {
                    pair_keys[base + offset] = (int64_t)s * num_vertices + v;
                    offset++;
                }
            }
        }
    }
}





__device__ __forceinline__ int lower_bound_dev(
    const int32_t* __restrict__ arr, int lo, int hi, int32_t target
) {
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__global__ void compute_intersections_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const double* __restrict__ weighted_degrees,
    const int32_t* __restrict__ seeds,
    const int64_t* __restrict__ unique_keys,
    int32_t num_pairs,
    int32_t num_vertices,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    double* __restrict__ out_scores
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;

    if (warp_id >= num_pairs) return;

    int64_t key = unique_keys[warp_id];
    int32_t seed_idx = (int32_t)(key / num_vertices);
    int32_t v = (int32_t)(key % num_vertices);
    int32_t u = seeds[seed_idx];

    int u_start = offsets[u], u_end = offsets[u + 1];
    int v_start = offsets[v], v_end = offsets[v + 1];
    int u_deg = u_end - u_start;
    int v_deg = v_end - v_start;

    
    const int32_t* small_idx;
    const double* small_wt;
    int small_start, small_end;
    const int32_t* large_idx;
    const double* large_wt;
    int large_start, large_end;

    if (u_deg <= v_deg) {
        small_idx = indices; small_wt = edge_weights;
        small_start = u_start; small_end = u_end;
        large_idx = indices; large_wt = edge_weights;
        large_start = v_start; large_end = v_end;
    } else {
        small_idx = indices; small_wt = edge_weights;
        small_start = v_start; small_end = v_end;
        large_idx = indices; large_wt = edge_weights;
        large_start = u_start; large_end = u_end;
    }

    int small_len = small_end - small_start;
    double my_sum = 0.0;

    
    for (int i = lane; i < small_len; i += 32) {
        int32_t target = small_idx[small_start + i];
        double w_small = small_wt[small_start + i];

        
        int pos = lower_bound_dev(large_idx, large_start, large_end, target);

        if (pos < large_end && large_idx[pos] == target) {
            double w_large = large_wt[pos];
            my_sum += fmin(w_small, w_large);
        }
    }

    
    for (int offset = 16; offset > 0; offset >>= 1)
        my_sum += __shfl_down_sync(0xffffffff, my_sum, offset);

    if (lane == 0) {
        double w_deg_u = weighted_degrees[u];
        double w_deg_v = weighted_degrees[v];
        double score = my_sum / fmin(w_deg_u, w_deg_v);

        out_first[warp_id] = u;
        out_second[warp_id] = v;
        out_scores[warp_id] = score;
    }
}




__global__ void iota_kernel(int32_t* arr, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) arr[i] = i;
}

__global__ void gather_int32_kernel(
    const int32_t* __restrict__ src, const int32_t* __restrict__ perm,
    int32_t* __restrict__ dst, int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[i] = src[perm[i]];
}

__global__ void generate_seq_kernel(int32_t* arr, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = i;
}

}  

similarity_result_double_t overlap_all_pairs_similarity_seg(
    const graph32_t& graph,
    const double* edge_weights,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk) {

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    cudaStream_t stream = 0;
    int32_t n_verts = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    
    int32_t num_seeds;
    const int32_t* d_seeds;

    if (vertices != nullptr && num_vertices > 0) {
        num_seeds = (int32_t)num_vertices;
        d_seeds = vertices;
    } else {
        num_seeds = n_verts;
        cache.ensure_gen_seeds(n_verts);
        generate_seq_kernel<<<(n_verts + 255) / 256, 256, 0, stream>>>(
            cache.gen_seeds, n_verts);
        d_seeds = cache.gen_seeds;
    }

    
    cache.ensure_weighted_deg(n_verts);
    compute_weighted_degrees_kernel<<<(n_verts + 255) / 256, 256, 0, stream>>>(
        d_offsets, edge_weights, cache.weighted_deg, n_verts);

    
    cudaMemsetAsync(cache.total_count, 0, sizeof(int64_t), stream);
    int grid_count = num_seeds > 256 ? 256 : num_seeds;
    if (grid_count < 1) grid_count = 1;
    count_twohop_paths_kernel<<<grid_count, 256, 0, stream>>>(
        d_offsets, d_indices, d_seeds, num_seeds, cache.total_count);

    int64_t h_total_paths = 0;
    cudaMemcpyAsync(&h_total_paths, cache.total_count,
        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (h_total_paths == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t* pair_keys = nullptr;
    cudaMalloc(&pair_keys, h_total_paths * sizeof(int64_t));
    cudaMemsetAsync(cache.write_counter, 0, sizeof(int32_t), stream);

    int grid_gen = num_seeds > 1024 ? 1024 : num_seeds;
    if (grid_gen < 1) grid_gen = 1;
    generate_pairs_kernel<<<grid_gen, 256, 0, stream>>>(
        d_offsets, d_indices, d_seeds, num_seeds, n_verts,
        pair_keys, cache.write_counter, h_total_paths);

    
    int64_t max_key = (int64_t)(num_seeds - 1) * n_verts + (n_verts - 1);
    int end_bit = 1;
    while ((1LL << end_bit) <= max_key) end_bit++;

    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, sort_temp_bytes,
        (const int64_t*)nullptr, (int64_t*)nullptr, (int)h_total_paths);
    size_t unique_temp_bytes = 0;
    cub::DeviceSelect::Unique(
        nullptr, unique_temp_bytes,
        (const int64_t*)nullptr, (int64_t*)nullptr, (int32_t*)nullptr,
        (int)h_total_paths);
    size_t temp_bytes = sort_temp_bytes > unique_temp_bytes
                            ? sort_temp_bytes : unique_temp_bytes;

    uint8_t* temp_buf = nullptr;
    cudaMalloc(&temp_buf, temp_bytes);
    int64_t* sorted_keys = nullptr;
    cudaMalloc(&sorted_keys, h_total_paths * sizeof(int64_t));

    cub::DeviceRadixSort::SortKeys(
        temp_buf, sort_temp_bytes, pair_keys, sorted_keys,
        (int)h_total_paths, 0, end_bit, stream);

    
    int64_t* unique_keys = nullptr;
    cudaMalloc(&unique_keys, h_total_paths * sizeof(int64_t));

    cub::DeviceSelect::Unique(
        temp_buf, unique_temp_bytes, sorted_keys, unique_keys,
        cache.num_unique, (int)h_total_paths, stream);

    int32_t h_num_unique = 0;
    cudaMemcpyAsync(&h_num_unique, cache.num_unique,
        sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    
    cudaFree(pair_keys);
    cudaFree(sorted_keys);
    cudaFree(temp_buf);

    if (h_num_unique == 0) {
        cudaFree(unique_keys);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    double* out_scores = nullptr;
    cudaMalloc(&out_first, h_num_unique * sizeof(int32_t));
    cudaMalloc(&out_second, h_num_unique * sizeof(int32_t));
    cudaMalloc(&out_scores, h_num_unique * sizeof(double));

    int warps_per_block = 8;
    int block = warps_per_block * 32;
    int grid_inter = (h_num_unique + warps_per_block - 1) / warps_per_block;
    if (grid_inter < 1) grid_inter = 1;
    compute_intersections_warp_kernel<<<grid_inter, block, 0, stream>>>(
        d_offsets, d_indices, edge_weights, cache.weighted_deg, d_seeds,
        unique_keys, h_num_unique, n_verts,
        out_first, out_second, out_scores);

    cudaFree(unique_keys);

    std::size_t final_count = h_num_unique;

    
    if (topk.has_value() && (std::size_t)h_num_unique > topk.value()) {
        int topk_val = (int)topk.value();

        int32_t* perm = nullptr;
        cudaMalloc(&perm, h_num_unique * sizeof(int32_t));
        iota_kernel<<<(h_num_unique + 255) / 256, 256, 0, stream>>>(
            perm, h_num_unique);

        size_t topk_sort_temp = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, topk_sort_temp,
            (const double*)nullptr, (double*)nullptr,
            (const int32_t*)nullptr, (int32_t*)nullptr, h_num_unique);

        uint8_t* topk_temp = nullptr;
        cudaMalloc(&topk_temp, topk_sort_temp);
        double* scores_sorted = nullptr;
        cudaMalloc(&scores_sorted, h_num_unique * sizeof(double));
        int32_t* perm_sorted = nullptr;
        cudaMalloc(&perm_sorted, h_num_unique * sizeof(int32_t));

        cub::DeviceRadixSort::SortPairsDescending(
            topk_temp, topk_sort_temp,
            out_scores, scores_sorted, perm, perm_sorted,
            h_num_unique, 0, 64, stream);

        int32_t* final_first = nullptr;
        int32_t* final_second = nullptr;
        double* final_scores = nullptr;
        cudaMalloc(&final_first, topk_val * sizeof(int32_t));
        cudaMalloc(&final_second, topk_val * sizeof(int32_t));
        cudaMalloc(&final_scores, topk_val * sizeof(double));

        gather_int32_kernel<<<(topk_val + 255) / 256, 256, 0, stream>>>(
            out_first, perm_sorted, final_first, topk_val);
        gather_int32_kernel<<<(topk_val + 255) / 256, 256, 0, stream>>>(
            out_second, perm_sorted, final_second, topk_val);
        cudaMemcpyAsync(final_scores, scores_sorted,
            topk_val * sizeof(double), cudaMemcpyDeviceToDevice, stream);

        cudaStreamSynchronize(stream);

        cudaFree(perm);
        cudaFree(topk_temp);
        cudaFree(scores_sorted);
        cudaFree(perm_sorted);
        cudaFree(out_first);
        cudaFree(out_second);
        cudaFree(out_scores);

        out_first = final_first;
        out_second = final_second;
        out_scores = final_scores;
        final_count = topk_val;
    } else {
        cudaStreamSynchronize(stream);
    }

    return {out_first, out_second, out_scores, final_count};
}

}  
