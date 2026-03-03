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
#include <optional>

namespace aai {

namespace {

struct Cache : Cacheable {};




__device__ __forceinline__ int lower_bound_dev(const int32_t* arr, int size, int32_t target) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int intersect_count(
    const int32_t* __restrict__ a, int size_a,
    const int32_t* __restrict__ b, int size_b
) {
    if (size_a == 0 || size_b == 0) return 0;

    
    if (size_a > size_b) {
        const int32_t* tmp = a; a = b; b = tmp;
        int t = size_a; size_a = size_b; size_b = t;
    }

    
    int i = lower_bound_dev(a, size_a, b[0]);
    if (i >= size_a) return 0;
    int j = lower_bound_dev(b, size_b, a[i]);
    if (j >= size_b) return 0;

    
    if (size_b > 8 * size_a) {
        int count = 0;
        for (; i < size_a && j < size_b; i++) {
            int32_t target = a[i];
            
            int pos = j;
            int step = 1;
            while (pos + step < size_b && b[pos + step] < target) {
                pos += step;
                step <<= 1;
            }
            int lo = pos, hi = (pos + step < size_b) ? pos + step + 1 : size_b;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (b[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            j = lo;
            if (j < size_b && b[j] == target) { count++; j++; }
        }
        return count;
    }

    
    int count = 0;
    while (i < size_a && j < size_b) {
        int32_t va = a[i], vb = b[j];
        if (va == vb) { count++; i++; j++; }
        else if (va < vb) { i++; }
        else { j++; }
    }
    return count;
}




__global__ void count_raw_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    int64_t* __restrict__ pair_counts
) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;
    int32_t u = seeds[seed_idx];
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int64_t count = 0;
    for (int32_t i = u_start + threadIdx.x; i < u_end; i += blockDim.x) {
        int32_t w = indices[i];
        count += (int64_t)(offsets[w + 1] - offsets[w]);
    }
    typedef cub::BlockReduce<int64_t, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    int64_t total = BlockReduce(temp).Sum(count);
    if (threadIdx.x == 0) pair_counts[seed_idx] = total;
}




__global__ void enumerate_raw_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    const int64_t* __restrict__ pair_offsets,
    int64_t* __restrict__ out_keys
) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;
    int32_t u = seeds[seed_idx];
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int64_t base = pair_offsets[seed_idx];
    int64_t seed_prefix = (int64_t)seed_idx << 32;
    __shared__ int s_write_pos;
    if (threadIdx.x == 0) s_write_pos = 0;
    __syncthreads();
    for (int32_t i = u_start + threadIdx.x; i < u_end; i += blockDim.x) {
        int32_t w = indices[i];
        int32_t w_start = offsets[w];
        int32_t w_end = offsets[w + 1];
        int32_t deg_w = w_end - w_start;
        int my_pos = atomicAdd(&s_write_pos, deg_w);
        for (int32_t j = 0; j < deg_w; j++) {
            out_keys[base + my_pos + j] = seed_prefix | (uint32_t)indices[w_start + j];
        }
    }
}




__global__ void compute_jaccard_compact_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int64_t* __restrict__ pair_keys,
    const int32_t* __restrict__ seeds,
    int64_t num_pairs,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int* __restrict__ output_counter
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;

    int64_t key = pair_keys[tid];
    int seed_idx = (int)(key >> 32);
    int32_t v = (int32_t)(key & 0xFFFFFFFF);
    int32_t u = seeds[seed_idx];

    if (u == v) return;  

    int32_t u_start = offsets[u], u_end = offsets[u + 1];
    int32_t v_start = offsets[v], v_end = offsets[v + 1];
    int32_t deg_u = u_end - u_start, deg_v = v_end - v_start;

    int intersection = intersect_count(indices + u_start, deg_u, indices + v_start, deg_v);

    float jaccard = 0.0f;
    int union_size = deg_u + deg_v - intersection;
    if (union_size > 0) jaccard = (float)intersection / (float)union_size;

    int pos = atomicAdd(output_counter, 1);
    out_first[pos] = u;
    out_second[pos] = v;
    out_scores[pos] = jaccard;
}

__global__ void iota_kernel(int32_t* out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) out[tid] = tid;
}


__global__ void pack_pairs_kernel(const int32_t* first, const int32_t* second, int64_t* packed, int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    packed[tid] = ((int64_t)first[tid] << 32) | (uint32_t)second[tid];
}


__global__ void unpack_pairs_kernel(const int64_t* packed, int32_t* first, int32_t* second, int64_t n) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    first[tid] = (int32_t)(packed[tid] >> 32);
    second[tid] = (int32_t)(packed[tid] & 0xFFFFFFFF);
}

static int compute_end_bit(int num_seeds) {
    int s_bits = 1;
    int s = num_seeds;
    while (s > 1) { s_bits++; s >>= 1; }
    return 32 + s_bits;
}

}  

similarity_result_float_t jaccard_all_pairs_similarity(const graph32_t& graph,
                                                       const int32_t* vertices,
                                                       std::size_t num_vertices,
                                                       std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_verts = graph.number_of_vertices;
    cudaStream_t stream = 0;

    
    const int32_t* d_seeds;
    int num_seeds;
    int32_t* seeds_alloc = nullptr;
    if (vertices != nullptr) {
        d_seeds = vertices;
        num_seeds = (int)num_vertices;
    } else {
        cudaMalloc(&seeds_alloc, (size_t)num_verts * sizeof(int32_t));
        if (num_verts > 0)
            iota_kernel<<<(num_verts + 255) / 256, 256, 0, stream>>>(seeds_alloc, num_verts);
        d_seeds = seeds_alloc;
        num_seeds = num_verts;
    }

    if (num_seeds == 0) {
        if (seeds_alloc) cudaFree(seeds_alloc);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t* d_counts;
    cudaMalloc(&d_counts, (size_t)num_seeds * sizeof(int64_t));
    count_raw_pairs_kernel<<<num_seeds, 256, 0, stream>>>(d_offsets, d_indices, d_seeds, num_seeds, d_counts);

    
    int64_t* d_offsets_ps;
    cudaMalloc(&d_offsets_ps, (size_t)num_seeds * sizeof(int64_t));
    size_t ps_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, ps_temp_bytes, (int64_t*)nullptr, (int64_t*)nullptr, num_seeds);
    void* d_ps_temp;
    cudaMalloc(&d_ps_temp, ps_temp_bytes);
    cub::DeviceScan::ExclusiveSum(d_ps_temp, ps_temp_bytes, d_counts, d_offsets_ps, num_seeds, stream);
    cudaFree(d_ps_temp);

    
    int64_t last_offset, last_count;
    cudaMemcpyAsync(&last_offset, d_offsets_ps + num_seeds - 1, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&last_count, d_counts + num_seeds - 1, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int64_t total_raw = last_offset + last_count;

    cudaFree(d_counts);

    if (total_raw <= 0) {
        cudaFree(d_offsets_ps);
        if (seeds_alloc) cudaFree(seeds_alloc);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t* d_raw_keys;
    cudaMalloc(&d_raw_keys, (size_t)total_raw * sizeof(int64_t));
    enumerate_raw_pairs_kernel<<<num_seeds, 256, 0, stream>>>(d_offsets, d_indices, d_seeds, num_seeds, d_offsets_ps, d_raw_keys);
    cudaFree(d_offsets_ps);

    
    int end_bit = compute_end_bit(num_seeds);
    if (end_bit > 64) end_bit = 64;

    int64_t* d_sorted_keys;
    cudaMalloc(&d_sorted_keys, (size_t)total_raw * sizeof(int64_t));
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, sort_temp_bytes, (int64_t*)nullptr, (int64_t*)nullptr, (int)total_raw, 0, end_bit);
    void* d_sort_temp;
    cudaMalloc(&d_sort_temp, sort_temp_bytes);
    cub::DeviceRadixSort::SortKeys(d_sort_temp, sort_temp_bytes, d_raw_keys, d_sorted_keys, (int)total_raw, 0, end_bit, stream);
    cudaFree(d_sort_temp);
    cudaFree(d_raw_keys);

    
    int64_t* d_unique_keys;
    cudaMalloc(&d_unique_keys, (size_t)total_raw * sizeof(int64_t));
    int* d_num_selected;
    cudaMalloc(&d_num_selected, sizeof(int));
    size_t unique_temp_bytes = 0;
    cub::DeviceSelect::Unique(nullptr, unique_temp_bytes, (int64_t*)nullptr, (int64_t*)nullptr, (int*)nullptr, (int)total_raw);
    void* d_unique_temp;
    cudaMalloc(&d_unique_temp, unique_temp_bytes);
    cub::DeviceSelect::Unique(d_unique_temp, unique_temp_bytes, d_sorted_keys, d_unique_keys, d_num_selected, (int)total_raw, stream);
    cudaFree(d_unique_temp);
    cudaFree(d_sorted_keys);

    int num_unique;
    cudaMemcpyAsync(&num_unique, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_num_selected);

    
    int32_t* d_first;
    int32_t* d_second;
    float* d_scores;
    cudaMalloc(&d_first, (size_t)num_unique * sizeof(int32_t));
    cudaMalloc(&d_second, (size_t)num_unique * sizeof(int32_t));
    cudaMalloc(&d_scores, (size_t)num_unique * sizeof(float));

    int* d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemsetAsync(d_counter, 0, sizeof(int), stream);

    if (num_unique > 0) {
        int block = 256;
        int grid = (int)(((int64_t)num_unique + block - 1) / block);
        compute_jaccard_compact_kernel<<<grid, block, 0, stream>>>(
            d_offsets, d_indices, d_unique_keys, d_seeds, (int64_t)num_unique,
            d_first, d_second, d_scores, d_counter);
    }

    cudaFree(d_unique_keys);

    int num_valid;
    cudaMemcpyAsync(&num_valid, d_counter, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_counter);

    int64_t final_count = num_valid;

    
    if (topk.has_value() && (int64_t)topk.value() < final_count) {
        int64_t topk_val = (int64_t)topk.value();

        
        int64_t* d_packed;
        cudaMalloc(&d_packed, (size_t)num_valid * sizeof(int64_t));
        if (num_valid > 0)
            pack_pairs_kernel<<<(int)(((int64_t)num_valid + 255) / 256), 256, 0, stream>>>(d_first, d_second, d_packed, (int64_t)num_valid);

        
        float* d_scores_out;
        int64_t* d_packed_out;
        cudaMalloc(&d_scores_out, (size_t)num_valid * sizeof(float));
        cudaMalloc(&d_packed_out, (size_t)num_valid * sizeof(int64_t));
        size_t topk_temp_bytes = 0;
        cub::DeviceRadixSort::SortPairsDescending(nullptr, topk_temp_bytes,
            (float*)nullptr, (float*)nullptr, (int64_t*)nullptr, (int64_t*)nullptr, (int)num_valid, 0, 32);
        void* d_topk_temp;
        cudaMalloc(&d_topk_temp, topk_temp_bytes);
        cub::DeviceRadixSort::SortPairsDescending(d_topk_temp, topk_temp_bytes,
            d_scores, d_scores_out, d_packed, d_packed_out, (int)num_valid, 0, 32, stream);
        cudaFree(d_topk_temp);
        cudaFree(d_packed);

        
        int32_t* out_first;
        int32_t* out_second;
        float* out_scores;
        cudaMalloc(&out_first, (size_t)topk_val * sizeof(int32_t));
        cudaMalloc(&out_second, (size_t)topk_val * sizeof(int32_t));
        cudaMalloc(&out_scores, (size_t)topk_val * sizeof(float));
        if (topk_val > 0)
            unpack_pairs_kernel<<<(int)(((int64_t)topk_val + 255) / 256), 256, 0, stream>>>(d_packed_out, out_first, out_second, topk_val);

        cudaMemcpyAsync(out_scores, d_scores_out, (size_t)topk_val * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        cudaFree(d_scores_out);
        cudaFree(d_packed_out);
        cudaFree(d_first);
        cudaFree(d_second);
        cudaFree(d_scores);
        if (seeds_alloc) cudaFree(seeds_alloc);

        return {out_first, out_second, out_scores, (std::size_t)topk_val};
    }

    
    if (final_count < (int64_t)num_unique) {
        if (final_count > 0) {
            int32_t* out_first;
            int32_t* out_second;
            float* out_scores;
            cudaMalloc(&out_first, (size_t)final_count * sizeof(int32_t));
            cudaMalloc(&out_second, (size_t)final_count * sizeof(int32_t));
            cudaMalloc(&out_scores, (size_t)final_count * sizeof(float));
            cudaMemcpyAsync(out_first, d_first, (size_t)final_count * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_second, d_second, (size_t)final_count * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(out_scores, d_scores, (size_t)final_count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            cudaFree(d_first);
            cudaFree(d_second);
            cudaFree(d_scores);
            if (seeds_alloc) cudaFree(seeds_alloc);
            return {out_first, out_second, out_scores, (std::size_t)final_count};
        }
        cudaFree(d_first);
        cudaFree(d_second);
        cudaFree(d_scores);
        if (seeds_alloc) cudaFree(seeds_alloc);
        return {nullptr, nullptr, nullptr, 0};
    }

    if (seeds_alloc) cudaFree(seeds_alloc);
    return {d_first, d_second, d_scores, (std::size_t)final_count};
}

}  
