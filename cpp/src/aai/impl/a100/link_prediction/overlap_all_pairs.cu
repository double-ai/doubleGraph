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
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <cstdint>

namespace aai {

namespace {

static bool g_pool_inited = false;
static void init_pool() {
    if (g_pool_inited) return;
    g_pool_inited = true;
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);
    uint64_t threshold = UINT64_MAX;
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
}

#define AMALLOC(ptr, sz, stream) cudaMallocAsync((void**)&(ptr), (sz), (stream))
#define AFREE(ptr, stream) cudaFreeAsync((ptr), (stream))

struct Cache : Cacheable {
    Cache() { init_pool(); }
    ~Cache() override {}
};




__global__ void count_expansion_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    int64_t* __restrict__ counts)
{
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_seeds) return;
    int u = seeds[sid];
    int u_start = offsets[u], u_end = offsets[u + 1];
    int64_t total = 0;
    for (int i = u_start; i < u_end; i++) {
        int w = indices[i];
        total += (int64_t)(offsets[w + 1] - offsets[w]);
    }
    counts[sid] = total;
}




__global__ void expand_two_hop_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    const int64_t* __restrict__ expansion_offsets,
    int64_t* __restrict__ out_keys)
{
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;
    int u = seeds[seed_idx];
    int u_start = offsets[u], u_end = offsets[u + 1];
    int64_t base = expansion_offsets[seed_idx];
    int64_t pos = base;
    for (int i = u_start; i < u_end; i++) {
        int w = indices[i];
        int w_start = offsets[w], w_end = offsets[w + 1];
        int w_deg = w_end - w_start;
        for (int j = threadIdx.x; j < w_deg; j += blockDim.x) {
            int v = indices[w_start + j];
            out_keys[pos + j] = ((int64_t)(uint32_t)u << 32) | (int64_t)(uint32_t)v;
        }
        pos += w_deg;
    }
}

struct IsSelfLoop {
    __host__ __device__ bool operator()(int64_t key) const {
        return ((uint32_t)(key >> 32)) == ((uint32_t)(key & 0xFFFFFFFFLL));
    }
};




template <bool IS_MULTIGRAPH>
__global__ void compute_overlap_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int64_t* __restrict__ keys,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ scores,
    int64_t num_pairs)
{
    int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_global >= num_pairs) return;

    int64_t key = keys[warp_global];
    int u = (int32_t)(key >> 32);
    int v = (int32_t)(key & 0xFFFFFFFFLL);

    if (lane == 0) {
        out_first[warp_global] = u;
        out_second[warp_global] = v;
    }

    int u_start = offsets[u], u_end = offsets[u + 1];
    int v_start = offsets[v], v_end = offsets[v + 1];
    int deg_u = u_end - u_start;
    int deg_v = v_end - v_start;
    int min_deg = (deg_u < deg_v) ? deg_u : deg_v;

    if (min_deg == 0) {
        if (lane == 0) scores[warp_global] = 0.0f;
        return;
    }

    const int32_t* probe_arr;
    const int32_t* search_arr;
    int probe_size, search_size;
    if (deg_u <= deg_v) {
        probe_arr = indices + u_start;
        search_arr = indices + v_start;
        probe_size = deg_u;
        search_size = deg_v;
    } else {
        probe_arr = indices + v_start;
        search_arr = indices + u_start;
        probe_size = deg_v;
        search_size = deg_u;
    }

    int count = 0;

    if constexpr (!IS_MULTIGRAPH) {
        
        for (int i = lane; i < probe_size; i += 32) {
            int32_t target = probe_arr[i];
            int lo = 0, hi = search_size;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (__ldg(&search_arr[mid]) < target) lo = mid + 1;
                else hi = mid;
            }
            if (lo < search_size && __ldg(&search_arr[lo]) == target) count++;
        }
    } else {
        
        for (int i = lane; i < probe_size; i += 32) {
            int32_t target = probe_arr[i];
            
            int plo = 0, phi = i;
            while (plo < phi) {
                int mid = plo + ((phi - plo) >> 1);
                if (probe_arr[mid] < target) plo = mid + 1;
                else phi = mid;
            }
            int rank = i - plo;
            
            int lo = 0, hi = search_size;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (__ldg(&search_arr[mid]) < target) lo = mid + 1;
                else hi = mid;
            }
            if (lo + rank < search_size && __ldg(&search_arr[lo + rank]) == target)
                count++;
        }
    }

    
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        count += __shfl_down_sync(0xFFFFFFFF, count, mask);

    if (lane == 0)
        scores[warp_global] = (float)count / (float)min_deg;
}

}  

similarity_result_float_t overlap_all_pairs_similarity(const graph32_t& graph,
                                                       const int32_t* vertices,
                                                       std::size_t num_vertices,
                                                       std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    bool is_multigraph = graph.is_multigraph;

    cudaStream_t stream = 0;

    
    int32_t* d_all_seeds = nullptr;
    const int32_t* seeds_ptr = vertices;
    int actual_num_seeds = (vertices != nullptr && num_vertices > 0) ? (int)num_vertices : 0;

    if (vertices == nullptr || num_vertices == 0) {
        actual_num_seeds = graph.number_of_vertices;
        AMALLOC(d_all_seeds, (size_t)graph.number_of_vertices * sizeof(int32_t), stream);
        thrust::device_ptr<int32_t> p(d_all_seeds);
        thrust::sequence(thrust::cuda::par.on(stream), p, p + graph.number_of_vertices);
        seeds_ptr = d_all_seeds;
    }

    if (actual_num_seeds == 0) {
        if (d_all_seeds) AFREE(d_all_seeds, stream);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t* d_counts;
    int64_t* d_exp_offsets;
    AMALLOC(d_counts, (size_t)(actual_num_seeds + 1) * sizeof(int64_t), stream);
    AMALLOC(d_exp_offsets, (size_t)(actual_num_seeds + 1) * sizeof(int64_t), stream);

    {
        int block = 256;
        int grid = (actual_num_seeds + block - 1) / block;
        count_expansion_kernel<<<grid, block, 0, stream>>>(
            d_offsets, d_indices, seeds_ptr, actual_num_seeds, d_counts);
    }

    {
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes,
            d_counts, d_exp_offsets, actual_num_seeds + 1, stream);
        AMALLOC(d_temp, temp_bytes, stream);
        cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes,
            d_counts, d_exp_offsets, actual_num_seeds + 1, stream);
        AFREE(d_temp, stream);
    }

    int64_t total_raw;
    cudaMemcpyAsync(&total_raw, d_exp_offsets + actual_num_seeds,
                    sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    AFREE(d_counts, stream);

    if (total_raw <= 0) {
        AFREE(d_exp_offsets, stream);
        if (d_all_seeds) AFREE(d_all_seeds, stream);
        cudaStreamSynchronize(stream);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t* d_keys;
    AMALLOC(d_keys, (size_t)total_raw * sizeof(int64_t), stream);

    expand_two_hop_kernel<<<actual_num_seeds, 512, 0, stream>>>(
        d_offsets, d_indices, seeds_ptr, actual_num_seeds,
        d_exp_offsets, d_keys);

    AFREE(d_exp_offsets, stream);
    if (d_all_seeds) { AFREE(d_all_seeds, stream); d_all_seeds = nullptr; }

    
    int end_bit = 64;
    {
        int bits = 1;
        int v = graph.number_of_vertices - 1;
        while (v > 0) { v >>= 1; bits++; }
        end_bit = bits * 2;
        if (end_bit > 64) end_bit = 64;
        if (end_bit < 32) end_bit = 32;
    }

    {
        int64_t* d_keys_alt;
        AMALLOC(d_keys_alt, (size_t)total_raw * sizeof(int64_t), stream);
        void* d_temp = nullptr;
        size_t temp_bytes = 0;
        cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes,
            d_keys, d_keys_alt, total_raw, 0, end_bit, stream);
        AMALLOC(d_temp, temp_bytes, stream);
        cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes,
            d_keys, d_keys_alt, total_raw, 0, end_bit, stream);
        AFREE(d_temp, stream);
        AFREE(d_keys, stream);
        d_keys = d_keys_alt;
    }

    
    thrust::device_ptr<int64_t> keys_ptr_thrust(d_keys);
    auto new_end = thrust::remove_if(thrust::cuda::par.on(stream),
        keys_ptr_thrust, keys_ptr_thrust + total_raw, IsSelfLoop{});
    int64_t after_remove = new_end - keys_ptr_thrust;

    auto unique_end = thrust::unique(thrust::cuda::par.on(stream),
        keys_ptr_thrust, keys_ptr_thrust + after_remove);
    int64_t num_unique = unique_end - keys_ptr_thrust;

    if (num_unique == 0) {
        AFREE(d_keys, stream);
        cudaStreamSynchronize(stream);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t* d_first;
    int32_t* d_second;
    float* d_scores;
    AMALLOC(d_first, (size_t)num_unique * sizeof(int32_t), stream);
    AMALLOC(d_second, (size_t)num_unique * sizeof(int32_t), stream);
    AMALLOC(d_scores, (size_t)num_unique * sizeof(float), stream);

    {
        int warps_per_block = 8;
        int threads_per_block = warps_per_block * 32;
        int grid = (int)((num_unique + warps_per_block - 1) / warps_per_block);
        if (is_multigraph) {
            compute_overlap_warp_kernel<true><<<grid, threads_per_block, 0, stream>>>(
                d_offsets, d_indices, d_keys, d_first, d_second, d_scores, num_unique);
        } else {
            compute_overlap_warp_kernel<false><<<grid, threads_per_block, 0, stream>>>(
                d_offsets, d_indices, d_keys, d_first, d_second, d_scores, num_unique);
        }
    }

    AFREE(d_keys, stream);

    
    if (topk.has_value() && (int64_t)topk.value() < num_unique) {
        int64_t topk_val = (int64_t)topk.value();

        int32_t* d_perm;
        int32_t* d_perm_out;
        float* d_neg_scores;
        float* d_neg_out;
        AMALLOC(d_perm, (size_t)num_unique * sizeof(int32_t), stream);
        AMALLOC(d_perm_out, (size_t)num_unique * sizeof(int32_t), stream);
        AMALLOC(d_neg_scores, (size_t)num_unique * sizeof(float), stream);
        AMALLOC(d_neg_out, (size_t)num_unique * sizeof(float), stream);

        
        thrust::device_ptr<float> s(d_scores), ns(d_neg_scores);
        thrust::transform(thrust::cuda::par.on(stream), s, s + num_unique, ns,
                         [] __device__ (float x) { return -x; });
        thrust::device_ptr<int32_t> p(d_perm);
        thrust::sequence(thrust::cuda::par.on(stream), p, p + num_unique);

        {
            void* d_temp = nullptr;
            size_t temp_bytes = 0;
            cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
                d_neg_scores, d_neg_out, d_perm, d_perm_out,
                (int)num_unique, 0, 32, stream);
            AMALLOC(d_temp, temp_bytes, stream);
            cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
                d_neg_scores, d_neg_out, d_perm, d_perm_out,
                (int)num_unique, 0, 32, stream);
            AFREE(d_temp, stream);
        }

        AFREE(d_neg_scores, stream);
        AFREE(d_neg_out, stream);
        AFREE(d_perm, stream);

        int32_t* d_first_topk;
        int32_t* d_second_topk;
        float* d_scores_topk;
        AMALLOC(d_first_topk, (size_t)topk_val * sizeof(int32_t), stream);
        AMALLOC(d_second_topk, (size_t)topk_val * sizeof(int32_t), stream);
        AMALLOC(d_scores_topk, (size_t)topk_val * sizeof(float), stream);

        thrust::device_ptr<int32_t> po(d_perm_out);
        thrust::device_ptr<int32_t> f(d_first), se(d_second);
        thrust::gather(thrust::cuda::par.on(stream),
            po, po + topk_val, f, thrust::device_ptr<int32_t>(d_first_topk));
        thrust::gather(thrust::cuda::par.on(stream),
            po, po + topk_val, se, thrust::device_ptr<int32_t>(d_second_topk));
        thrust::gather(thrust::cuda::par.on(stream),
            po, po + topk_val, s, thrust::device_ptr<float>(d_scores_topk));

        AFREE(d_perm_out, stream);
        AFREE(d_first, stream);
        AFREE(d_second, stream);
        AFREE(d_scores, stream);

        cudaStreamSynchronize(stream);
        return {d_first_topk, d_second_topk, d_scores_topk, (std::size_t)topk_val};
    } else {
        cudaStreamSynchronize(stream);
        return {d_first, d_second, d_scores, (std::size_t)num_unique};
    }
}

}  
