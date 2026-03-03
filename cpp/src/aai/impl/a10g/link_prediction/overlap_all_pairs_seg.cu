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
#include <optional>

namespace aai {

namespace {

struct Cache : Cacheable {
    int64_t* h_pinned = nullptr;

    Cache() {
        cudaMallocHost(&h_pinned, 8 * sizeof(int64_t));
    }

    ~Cache() override {
        if (h_pinned) { cudaFreeHost(h_pinned); h_pinned = nullptr; }
    }
};




__global__ void count_2hop_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    int64_t* __restrict__ counts)
{
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int u = seeds[sid];
    int start_u = offsets[u];
    int deg_u = offsets[u + 1] - start_u;
    int64_t local_count = 0;
    for (int j = threadIdx.x; j < deg_u; j += blockDim.x) {
        int w = indices[start_u + j];
        local_count += (int64_t)(offsets[w + 1] - offsets[w]);
    }
    typedef cub::BlockReduce<int64_t, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int64_t total = BlockReduce(temp_storage).Sum(local_count);
    if (threadIdx.x == 0) counts[sid] = total;
}




__global__ void generate_pairs_multiblock_u32(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds, int V,
    const int64_t* __restrict__ seed_offsets,
    uint32_t* __restrict__ keys,
    int32_t* __restrict__ seed_counters,
    int blocks_per_seed)
{
    int sid = blockIdx.x / blocks_per_seed;
    int bid = blockIdx.x % blocks_per_seed;
    if (sid >= num_seeds) return;
    int u = seeds[sid];
    int start_u = offsets[u];
    int deg_u = offsets[u + 1] - start_u;
    int chunk = (deg_u + blocks_per_seed - 1) / blocks_per_seed;
    int j_start = bid * chunk;
    int j_end = j_start + chunk;
    if (j_end > deg_u) j_end = deg_u;
    if (j_start >= deg_u) return;
    int64_t base = seed_offsets[sid];
    uint32_t key_prefix = (uint32_t)sid * V;
    __shared__ int s_offset;
    for (int j = j_start; j < j_end; j++) {
        int w = indices[start_u + j];
        int start_w = offsets[w];
        int deg_w = offsets[w + 1] - start_w;
        if (threadIdx.x == 0) s_offset = atomicAdd(&seed_counters[sid], deg_w);
        __syncthreads();
        int local_off = s_offset;
        for (int k = threadIdx.x; k < deg_w; k += blockDim.x) {
            keys[base + local_off + k] = key_prefix + (uint32_t)indices[start_w + k];
        }
        __syncthreads();
    }
}

__global__ void generate_pairs_multiblock_u64(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds, int64_t V,
    const int64_t* __restrict__ seed_offsets,
    uint64_t* __restrict__ keys,
    int32_t* __restrict__ seed_counters,
    int blocks_per_seed)
{
    int sid = blockIdx.x / blocks_per_seed;
    int bid = blockIdx.x % blocks_per_seed;
    if (sid >= num_seeds) return;
    int u = seeds[sid];
    int start_u = offsets[u];
    int deg_u = offsets[u + 1] - start_u;
    int chunk = (deg_u + blocks_per_seed - 1) / blocks_per_seed;
    int j_start = bid * chunk;
    int j_end = j_start + chunk;
    if (j_end > deg_u) j_end = deg_u;
    if (j_start >= deg_u) return;
    int64_t base = seed_offsets[sid];
    uint64_t key_prefix = (uint64_t)sid * V;
    __shared__ int s_offset;
    for (int j = j_start; j < j_end; j++) {
        int w = indices[start_u + j];
        int start_w = offsets[w];
        int deg_w = offsets[w + 1] - start_w;
        if (threadIdx.x == 0) s_offset = atomicAdd(&seed_counters[sid], deg_w);
        __syncthreads();
        int local_off = s_offset;
        for (int k = threadIdx.x; k < deg_w; k += blockDim.x)
            keys[base + local_off + k] = key_prefix + (uint64_t)indices[start_w + k];
        __syncthreads();
    }
}




__global__ void compute_scores_kernel_u32(
    const uint32_t* __restrict__ unique_keys,
    const int32_t* __restrict__ run_lengths,
    int num_unique, int V,
    const int32_t* __restrict__ seeds,
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ out_first, int32_t* __restrict__ out_second,
    float* __restrict__ out_scores, int32_t* __restrict__ out_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_unique) return;
    uint32_t key = unique_keys[idx];
    int seed_idx = (int)(key / (uint32_t)V);
    int neighbor_id = (int)(key % (uint32_t)V);
    int seed_vertex = seeds[seed_idx];
    if (seed_vertex == neighbor_id) return;
    int deg_u = offsets[seed_vertex + 1] - offsets[seed_vertex];
    int deg_v = offsets[neighbor_id + 1] - offsets[neighbor_id];
    int min_deg = deg_u < deg_v ? deg_u : deg_v;
    if (min_deg == 0) return;
    float score = (float)run_lengths[idx] / (float)min_deg;
    int pos = atomicAdd(out_count, 1);
    out_first[pos] = seed_vertex;
    out_second[pos] = neighbor_id;
    out_scores[pos] = score;
}

__global__ void compute_scores_kernel_u64(
    const uint64_t* __restrict__ unique_keys,
    const int32_t* __restrict__ run_lengths,
    int num_unique, int64_t V,
    const int32_t* __restrict__ seeds,
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ out_first, int32_t* __restrict__ out_second,
    float* __restrict__ out_scores, int32_t* __restrict__ out_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_unique) return;
    uint64_t key = unique_keys[idx];
    int seed_idx = (int)(key / V);
    int neighbor_id = (int)(key % V);
    int seed_vertex = seeds[seed_idx];
    if (seed_vertex == neighbor_id) return;
    int deg_u = offsets[seed_vertex + 1] - offsets[seed_vertex];
    int deg_v = offsets[neighbor_id + 1] - offsets[neighbor_id];
    int min_deg = deg_u < deg_v ? deg_u : deg_v;
    if (min_deg == 0) return;
    float score = (float)run_lengths[idx] / (float)min_deg;
    int pos = atomicAdd(out_count, 1);
    out_first[pos] = seed_vertex;
    out_second[pos] = neighbor_id;
    out_scores[pos] = score;
}




__device__ __forceinline__ int lower_bound_dev(const int32_t* arr, int len, int target) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ int upper_bound_dev(const int32_t* arr, int len, int target) {
    int lo = 0, hi = len;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (arr[mid] <= target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

template<typename KeyT>
__global__ void compute_scores_multigraph_warp(
    const KeyT* __restrict__ unique_keys,
    int num_unique,
    int64_t V,
    const int32_t* __restrict__ seeds,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int32_t* __restrict__ out_count)
{
    int warp_idx = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (warp_idx >= num_unique) return;

    KeyT key = unique_keys[warp_idx];
    int seed_idx = (int)((uint64_t)key / (uint64_t)V);
    int neighbor_id = (int)((uint64_t)key % (uint64_t)V);
    int seed_vertex = seeds[seed_idx];

    if (seed_vertex == neighbor_id) return;

    int deg_u = offsets[seed_vertex + 1] - offsets[seed_vertex];
    int deg_v = offsets[neighbor_id + 1] - offsets[neighbor_id];
    int min_deg = deg_u < deg_v ? deg_u : deg_v;
    if (min_deg == 0) return;

    int su = offsets[seed_vertex], sv = offsets[neighbor_id];

    const int32_t* short_ptr;
    const int32_t* long_ptr;
    int short_len, long_len;
    if (deg_u <= deg_v) {
        short_ptr = indices + su; short_len = deg_u;
        long_ptr = indices + sv; long_len = deg_v;
    } else {
        short_ptr = indices + sv; short_len = deg_v;
        long_ptr = indices + su; long_len = deg_u;
    }

    int local_count = 0;
    for (int i = lane; i < short_len; i += 32) {
        int x = short_ptr[i];
        int lo = lower_bound_dev(long_ptr, long_len, x);
        if (lo < long_len && long_ptr[lo] == x) {
            int hi = upper_bound_dev(long_ptr, long_len, x);
            int count_long = hi - lo;
            int first_in_short = lower_bound_dev(short_ptr, short_len, x);
            int k = i - first_in_short;
            if (k < count_long) local_count++;
        }
    }

    for (int offset = 16; offset > 0; offset /= 2)
        local_count += __shfl_xor_sync(0xffffffff, local_count, offset);

    if (lane == 0 && local_count > 0) {
        float score = (float)local_count / (float)min_deg;
        int pos = atomicAdd(out_count, 1);
        out_first[pos] = seed_vertex;
        out_second[pos] = neighbor_id;
        out_scores[pos] = score;
    }
}




__global__ void gather_topk_kernel(
    const int32_t* __restrict__ si, const int32_t* __restrict__ if_,
    const int32_t* __restrict__ is_, const float* __restrict__ isc,
    int c, int32_t* __restrict__ of_, int32_t* __restrict__ os_, float* __restrict__ osc)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= c) return;
    int src = si[idx];
    of_[idx] = if_[src]; os_[idx] = is_[src]; osc[idx] = isc[src];
}

__global__ void iota_kernel(int32_t* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = idx;
}




static int bits_needed(int64_t max_val) {
    if (max_val <= 0) return 1;
    int bits = 0;
    while (max_val > 0) { bits++; max_val >>= 1; }
    return bits;
}

}  

similarity_result_float_t overlap_all_pairs_similarity_seg(const graph32_t& graph,
                                                           const int32_t* vertices,
                                                           std::size_t num_vertices,
                                                           std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_verts = graph.number_of_vertices;
    bool is_multigraph = graph.is_multigraph;
    cudaStream_t stream = 0;

    
    int num_seeds;
    const int32_t* d_seeds;
    int32_t* d_seeds_alloc = nullptr;

    if (vertices != nullptr && num_vertices > 0) {
        num_seeds = (int)num_vertices;
        d_seeds = vertices;
    } else {
        num_seeds = num_verts;
        cudaMalloc(&d_seeds_alloc, num_seeds * sizeof(int32_t));
        iota_kernel<<<(num_seeds + 255) / 256, 256, 0, stream>>>(d_seeds_alloc, num_seeds);
        d_seeds = d_seeds_alloc;
    }

    auto cleanup_seeds = [&]() {
        if (d_seeds_alloc) { cudaFree(d_seeds_alloc); d_seeds_alloc = nullptr; }
    };

    if (num_seeds == 0) {
        cleanup_seeds();
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t* d_counts;
    cudaMalloc(&d_counts, num_seeds * sizeof(int64_t));
    count_2hop_kernel<<<num_seeds, 256, 0, stream>>>(d_offsets, d_indices, d_seeds, num_seeds, d_counts);

    
    int64_t* d_offsets_out;
    cudaMalloc(&d_offsets_out, (num_seeds + 1) * sizeof(int64_t));
    {
        size_t tb = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tb, (int64_t*)nullptr, (int64_t*)nullptr, num_seeds);
        void* d_tmp;
        cudaMalloc(&d_tmp, tb);
        cub::DeviceScan::ExclusiveSum(d_tmp, tb, d_counts, d_offsets_out, num_seeds, stream);
        cudaFree(d_tmp);
    }

    
    cudaMemcpyAsync(&cache.h_pinned[0], d_offsets_out + num_seeds - 1,
        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&cache.h_pinned[1], d_counts + num_seeds - 1,
        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int64_t total_2hop = cache.h_pinned[0] + cache.h_pinned[1];

    cudaFree(d_counts);

    if (total_2hop == 0) {
        cudaFree(d_offsets_out);
        cleanup_seeds();
        return {nullptr, nullptr, nullptr, 0};
    }

    int64_t V = num_verts;
    int64_t max_key = (int64_t)(num_seeds - 1) * V + (V - 1);
    bool use_u32 = (max_key <= (int64_t)UINT32_MAX);
    int end_bit = bits_needed(max_key);

    
    int blocks_per_seed;
    if (num_seeds <= 100) blocks_per_seed = 64;
    else if (num_seeds <= 1000) blocks_per_seed = 8;
    else blocks_per_seed = 1;

    
    int32_t* d_seed_counters;
    cudaMalloc(&d_seed_counters, num_seeds * sizeof(int32_t));
    cudaMemsetAsync(d_seed_counters, 0, num_seeds * sizeof(int32_t), stream);

    int32_t h_out_count = 0;
    int32_t* d_of = nullptr;
    int32_t* d_os = nullptr;
    float* d_osc = nullptr;

    if (use_u32) {
        uint32_t* d_keys;
        cudaMalloc(&d_keys, total_2hop * sizeof(uint32_t));
        generate_pairs_multiblock_u32<<<num_seeds * blocks_per_seed, 256, 0, stream>>>(
            d_offsets, d_indices, d_seeds, num_seeds, (int)V,
            d_offsets_out, d_keys, d_seed_counters, blocks_per_seed);

        uint32_t* d_keys_sorted;
        cudaMalloc(&d_keys_sorted, total_2hop * sizeof(uint32_t));
        {
            size_t tb = 0;
            cub::DeviceRadixSort::SortKeys(nullptr, tb, (uint32_t*)nullptr, (uint32_t*)nullptr, (int)total_2hop, 0, end_bit);
            void* d_tmp;
            cudaMalloc(&d_tmp, tb);
            cub::DeviceRadixSort::SortKeys(d_tmp, tb, d_keys, d_keys_sorted, (int)total_2hop, 0, end_bit, stream);
            cudaFree(d_tmp);
        }
        cudaFree(d_keys);

        uint32_t* d_unique_keys;
        int32_t* d_run_lengths;
        int32_t* d_num_runs;
        cudaMalloc(&d_unique_keys, total_2hop * sizeof(uint32_t));
        cudaMalloc(&d_run_lengths, total_2hop * sizeof(int32_t));
        cudaMalloc(&d_num_runs, sizeof(int32_t));
        {
            size_t tb = 0;
            cub::DeviceRunLengthEncode::Encode(nullptr, tb, (uint32_t*)nullptr, (uint32_t*)nullptr, (int32_t*)nullptr, (int32_t*)nullptr, (int)total_2hop);
            void* d_tmp;
            cudaMalloc(&d_tmp, tb);
            cub::DeviceRunLengthEncode::Encode(d_tmp, tb, d_keys_sorted, d_unique_keys, d_run_lengths, d_num_runs, (int)total_2hop, stream);
            cudaFree(d_tmp);
        }
        cudaFree(d_keys_sorted);

        int32_t h_num_runs;
        cudaMemcpyAsync(&cache.h_pinned[0], d_num_runs, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        h_num_runs = (int32_t)cache.h_pinned[0];
        cudaFree(d_num_runs);

        if (h_num_runs == 0) {
            cudaFree(d_unique_keys);
            cudaFree(d_run_lengths);
            cudaFree(d_offsets_out);
            cudaFree(d_seed_counters);
            cleanup_seeds();
            return {nullptr, nullptr, nullptr, 0};
        }

        cudaMalloc(&d_of, h_num_runs * sizeof(int32_t));
        cudaMalloc(&d_os, h_num_runs * sizeof(int32_t));
        cudaMalloc(&d_osc, h_num_runs * sizeof(float));
        int32_t* d_oc;
        cudaMalloc(&d_oc, sizeof(int32_t));
        cudaMemsetAsync(d_oc, 0, sizeof(int32_t), stream);

        if (is_multigraph) {
            int warps_per_block = 8;
            int threads = warps_per_block * 32;
            int blocks = (h_num_runs + warps_per_block - 1) / warps_per_block;
            compute_scores_multigraph_warp<uint32_t><<<blocks, threads, 0, stream>>>(
                d_unique_keys, h_num_runs, (int64_t)V, d_seeds, d_offsets, d_indices,
                d_of, d_os, d_osc, d_oc);
        } else {
            compute_scores_kernel_u32<<<(h_num_runs + 255) / 256, 256, 0, stream>>>(
                d_unique_keys, d_run_lengths, h_num_runs, (int)V, d_seeds, d_offsets,
                d_of, d_os, d_osc, d_oc);
        }

        cudaMemcpyAsync(&cache.h_pinned[0], d_oc, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        h_out_count = (int32_t)cache.h_pinned[0];

        cudaFree(d_unique_keys);
        cudaFree(d_run_lengths);
        cudaFree(d_oc);
    } else {
        
        uint64_t* d_keys;
        cudaMalloc(&d_keys, total_2hop * sizeof(uint64_t));
        generate_pairs_multiblock_u64<<<num_seeds * blocks_per_seed, 256, 0, stream>>>(
            d_offsets, d_indices, d_seeds, num_seeds, V,
            d_offsets_out, d_keys, d_seed_counters, blocks_per_seed);

        uint64_t* d_keys_sorted;
        cudaMalloc(&d_keys_sorted, total_2hop * sizeof(uint64_t));
        {
            size_t tb = 0;
            cub::DeviceRadixSort::SortKeys(nullptr, tb, (uint64_t*)nullptr, (uint64_t*)nullptr, (int)total_2hop, 0, end_bit);
            void* d_tmp;
            cudaMalloc(&d_tmp, tb);
            cub::DeviceRadixSort::SortKeys(d_tmp, tb, d_keys, d_keys_sorted, (int)total_2hop, 0, end_bit, stream);
            cudaFree(d_tmp);
        }
        cudaFree(d_keys);

        uint64_t* d_unique_keys;
        int32_t* d_run_lengths;
        int32_t* d_num_runs;
        cudaMalloc(&d_unique_keys, total_2hop * sizeof(uint64_t));
        cudaMalloc(&d_run_lengths, total_2hop * sizeof(int32_t));
        cudaMalloc(&d_num_runs, sizeof(int32_t));
        {
            size_t tb = 0;
            cub::DeviceRunLengthEncode::Encode(nullptr, tb, (uint64_t*)nullptr, (uint64_t*)nullptr, (int32_t*)nullptr, (int32_t*)nullptr, (int)total_2hop);
            void* d_tmp;
            cudaMalloc(&d_tmp, tb);
            cub::DeviceRunLengthEncode::Encode(d_tmp, tb, d_keys_sorted, d_unique_keys, d_run_lengths, d_num_runs, (int)total_2hop, stream);
            cudaFree(d_tmp);
        }
        cudaFree(d_keys_sorted);

        int32_t h_num_runs;
        cudaMemcpyAsync(&cache.h_pinned[0], d_num_runs, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        h_num_runs = (int32_t)cache.h_pinned[0];
        cudaFree(d_num_runs);

        if (h_num_runs == 0) {
            cudaFree(d_unique_keys);
            cudaFree(d_run_lengths);
            cudaFree(d_offsets_out);
            cudaFree(d_seed_counters);
            cleanup_seeds();
            return {nullptr, nullptr, nullptr, 0};
        }

        cudaMalloc(&d_of, h_num_runs * sizeof(int32_t));
        cudaMalloc(&d_os, h_num_runs * sizeof(int32_t));
        cudaMalloc(&d_osc, h_num_runs * sizeof(float));
        int32_t* d_oc;
        cudaMalloc(&d_oc, sizeof(int32_t));
        cudaMemsetAsync(d_oc, 0, sizeof(int32_t), stream);

        if (is_multigraph) {
            int warps_per_block = 8;
            int threads = warps_per_block * 32;
            int blocks = (h_num_runs + warps_per_block - 1) / warps_per_block;
            compute_scores_multigraph_warp<uint64_t><<<blocks, threads, 0, stream>>>(
                d_unique_keys, h_num_runs, V, d_seeds, d_offsets, d_indices,
                d_of, d_os, d_osc, d_oc);
        } else {
            compute_scores_kernel_u64<<<(h_num_runs + 255) / 256, 256, 0, stream>>>(
                d_unique_keys, d_run_lengths, h_num_runs, V, d_seeds, d_offsets,
                d_of, d_os, d_osc, d_oc);
        }

        cudaMemcpyAsync(&cache.h_pinned[0], d_oc, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        h_out_count = (int32_t)cache.h_pinned[0];

        cudaFree(d_unique_keys);
        cudaFree(d_run_lengths);
        cudaFree(d_oc);
    }

    cudaFree(d_offsets_out);
    cudaFree(d_seed_counters);
    cleanup_seeds();

    if (h_out_count == 0) {
        cudaFree(d_of);
        cudaFree(d_os);
        cudaFree(d_osc);
        return {nullptr, nullptr, nullptr, 0};
    }

    bool need_topk = topk.has_value() && (std::size_t)h_out_count > topk.value();
    bool need_sort = topk.has_value();

    if (!need_sort) {
        
        int32_t* r_first;
        int32_t* r_second;
        float* r_scores;
        cudaMalloc(&r_first, h_out_count * sizeof(int32_t));
        cudaMalloc(&r_second, h_out_count * sizeof(int32_t));
        cudaMalloc(&r_scores, h_out_count * sizeof(float));
        cudaMemcpyAsync(r_first, d_of, h_out_count * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(r_second, d_os, h_out_count * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(r_scores, d_osc, h_out_count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaFree(d_of);
        cudaFree(d_os);
        cudaFree(d_osc);
        return {r_first, r_second, r_scores, (std::size_t)h_out_count};
    }

    int output_count = need_topk ? (int)topk.value() : h_out_count;

    
    float* d_scores_sorted;
    int32_t* d_iota;
    int32_t* d_indices_sorted;
    cudaMalloc(&d_scores_sorted, h_out_count * sizeof(float));
    cudaMalloc(&d_iota, h_out_count * sizeof(int32_t));
    cudaMalloc(&d_indices_sorted, h_out_count * sizeof(int32_t));
    iota_kernel<<<(h_out_count + 255) / 256, 256, 0, stream>>>(d_iota, h_out_count);
    {
        size_t tb = 0;
        cub::DeviceRadixSort::SortPairsDescending(nullptr, tb, (float*)nullptr, (float*)nullptr, (int32_t*)nullptr, (int32_t*)nullptr, h_out_count);
        void* d_tmp;
        cudaMalloc(&d_tmp, tb);
        cub::DeviceRadixSort::SortPairsDescending(d_tmp, tb, d_osc, d_scores_sorted, d_iota, d_indices_sorted, h_out_count, 0, 32, stream);
        cudaFree(d_tmp);
    }
    cudaFree(d_scores_sorted);
    cudaFree(d_iota);

    int32_t* r_first;
    int32_t* r_second;
    float* r_scores;
    cudaMalloc(&r_first, output_count * sizeof(int32_t));
    cudaMalloc(&r_second, output_count * sizeof(int32_t));
    cudaMalloc(&r_scores, output_count * sizeof(float));
    gather_topk_kernel<<<(output_count + 255) / 256, 256, 0, stream>>>(
        d_indices_sorted, d_of, d_os, d_osc, output_count, r_first, r_second, r_scores);

    cudaFree(d_indices_sorted);
    cudaFree(d_of);
    cudaFree(d_os);
    cudaFree(d_osc);

    return {r_first, r_second, r_scores, (std::size_t)output_count};
}

}  
