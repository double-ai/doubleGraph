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
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <cstdint>
#include <algorithm>
#include <optional>

namespace aai {

namespace {

struct Cache : Cacheable {
    void* cub_temp = nullptr;
    size_t cub_temp_capacity = 0;

    int32_t* src_vertices_buf = nullptr;
    size_t src_vertices_capacity = 0;

    float* degrees = nullptr;
    size_t degrees_capacity = 0;

    int64_t* counts = nullptr;
    size_t counts_capacity = 0;

    int64_t* pair_offsets = nullptr;
    size_t pair_offsets_capacity = 0;

    int64_t* pair_keys = nullptr;
    size_t pair_keys_capacity = 0;

    int64_t* pair_keys_sorted = nullptr;
    size_t pair_keys_sorted_capacity = 0;

    int64_t* num_selected = nullptr;

    uint32_t* sk_in = nullptr;
    size_t sk_in_capacity = 0;

    uint64_t* pv_in = nullptr;
    size_t pv_in_capacity = 0;

    uint32_t* sk_out = nullptr;
    size_t sk_out_capacity = 0;

    uint64_t* pv_out = nullptr;
    size_t pv_out_capacity = 0;

    void ensure_cub_temp(size_t needed) {
        if (needed > cub_temp_capacity) {
            if (cub_temp) cudaFree(cub_temp);
            cub_temp_capacity = std::max(needed * 2, (size_t)(32 * 1024 * 1024));
            cudaMalloc(&cub_temp, cub_temp_capacity);
        }
    }

    void ensure_src_vertices(size_t n) {
        if (n > src_vertices_capacity) {
            if (src_vertices_buf) cudaFree(src_vertices_buf);
            cudaMalloc(&src_vertices_buf, n * sizeof(int32_t));
            src_vertices_capacity = n;
        }
    }

    void ensure_degrees(size_t n) {
        if (n > degrees_capacity) {
            if (degrees) cudaFree(degrees);
            cudaMalloc(&degrees, n * sizeof(float));
            degrees_capacity = n;
        }
    }

    void ensure_counts(size_t n) {
        if (n > counts_capacity) {
            if (counts) cudaFree(counts);
            cudaMalloc(&counts, n * sizeof(int64_t));
            counts_capacity = n;
        }
    }

    void ensure_pair_offsets(size_t n) {
        if (n > pair_offsets_capacity) {
            if (pair_offsets) cudaFree(pair_offsets);
            cudaMalloc(&pair_offsets, n * sizeof(int64_t));
            pair_offsets_capacity = n;
        }
    }

    void ensure_pair_keys(size_t n) {
        if (n > pair_keys_capacity) {
            if (pair_keys) cudaFree(pair_keys);
            cudaMalloc(&pair_keys, n * sizeof(int64_t));
            pair_keys_capacity = n;
        }
    }

    void ensure_pair_keys_sorted(size_t n) {
        if (n > pair_keys_sorted_capacity) {
            if (pair_keys_sorted) cudaFree(pair_keys_sorted);
            cudaMalloc(&pair_keys_sorted, n * sizeof(int64_t));
            pair_keys_sorted_capacity = n;
        }
    }

    void ensure_num_selected() {
        if (!num_selected) {
            cudaMalloc(&num_selected, sizeof(int64_t));
        }
    }

    void ensure_sk_in(size_t n) {
        if (n > sk_in_capacity) {
            if (sk_in) cudaFree(sk_in);
            cudaMalloc(&sk_in, n * sizeof(uint32_t));
            sk_in_capacity = n;
        }
    }

    void ensure_pv_in(size_t n) {
        if (n > pv_in_capacity) {
            if (pv_in) cudaFree(pv_in);
            cudaMalloc(&pv_in, n * sizeof(uint64_t));
            pv_in_capacity = n;
        }
    }

    void ensure_sk_out(size_t n) {
        if (n > sk_out_capacity) {
            if (sk_out) cudaFree(sk_out);
            cudaMalloc(&sk_out, n * sizeof(uint32_t));
            sk_out_capacity = n;
        }
    }

    void ensure_pv_out(size_t n) {
        if (n > pv_out_capacity) {
            if (pv_out) cudaFree(pv_out);
            cudaMalloc(&pv_out, n * sizeof(uint64_t));
            pv_out_capacity = n;
        }
    }

    ~Cache() override {
        if (cub_temp) cudaFree(cub_temp);
        if (src_vertices_buf) cudaFree(src_vertices_buf);
        if (degrees) cudaFree(degrees);
        if (counts) cudaFree(counts);
        if (pair_offsets) cudaFree(pair_offsets);
        if (pair_keys) cudaFree(pair_keys);
        if (pair_keys_sorted) cudaFree(pair_keys_sorted);
        if (num_selected) cudaFree(num_selected);
        if (sk_in) cudaFree(sk_in);
        if (pv_in) cudaFree(pv_in);
        if (sk_out) cudaFree(sk_out);
        if (pv_out) cudaFree(pv_out);
    }
};



__global__ void fill_sequence_kernel(int32_t* arr, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = idx;
}

__global__ void compute_weighted_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ weights,
    float* __restrict__ degrees,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int start = offsets[v];
    int end = offsets[v + 1];
    float sum = 0.0f;
    for (int i = start; i < end; i++) {
        sum += weights[i];
    }
    degrees[v] = sum;
}

__global__ void count_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ src_vertices,
    int32_t num_sources,
    int64_t* __restrict__ counts)
{
    int src_idx = blockIdx.x;
    if (src_idx >= num_sources) return;

    int u = src_vertices[src_idx];
    int u_start = offsets[u];
    int u_end = offsets[u + 1];

    int64_t local_count = 0;
    for (int i = u_start + threadIdx.x; i < u_end; i += blockDim.x) {
        int z = indices[i];
        int z_deg = offsets[z + 1] - offsets[z];
        local_count += (int64_t)(z_deg - 1);
    }

    typedef cub::BlockReduce<int64_t, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int64_t block_count = BlockReduce(temp_storage).Sum(local_count);

    if (threadIdx.x == 0) {
        counts[src_idx] = block_count;
    }
}

__global__ void enumerate_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ src_vertices,
    int32_t num_sources,
    const int64_t* __restrict__ pair_offsets,
    int64_t* __restrict__ pair_keys)
{
    int src_idx = blockIdx.x;
    if (src_idx >= num_sources) return;

    int u = src_vertices[src_idx];
    int u_start = offsets[u];
    int u_end = offsets[u + 1];

    __shared__ unsigned long long s_write_pos;
    if (threadIdx.x == 0) {
        s_write_pos = (unsigned long long)pair_offsets[src_idx];
    }
    __syncthreads();

    for (int i = u_start + threadIdx.x; i < u_end; i += blockDim.x) {
        int z = indices[i];
        int z_start = offsets[z];
        int z_end = offsets[z + 1];

        
        int batch_count = 0;
        for (int j = z_start; j < z_end; j++) {
            if (indices[j] != u) batch_count++;
        }

        
        unsigned long long base_pos = atomicAdd(&s_write_pos, (unsigned long long)batch_count);

        
        int write_idx = 0;
        for (int j = z_start; j < z_end; j++) {
            int w = indices[j];
            if (w != u) {
                pair_keys[base_pos + write_idx] = ((int64_t)(uint32_t)u << 32) | (int64_t)(uint32_t)w;
                write_idx++;
            }
        }
    }
}



__global__ void compute_sorensen_fused_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const float* __restrict__ degrees,
    const int64_t* __restrict__ unique_pairs,
    int64_t num_pairs,
    uint32_t* __restrict__ out_score_keys,   
    uint64_t* __restrict__ out_pair_values)   
{
    int warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int64_t key = unique_pairs[warp_id];
    int32_t u = (int32_t)((uint64_t)key >> 32);
    int32_t w = (int32_t)(key & 0xFFFFFFFFLL);

    int u_start = offsets[u], u_end = offsets[u + 1];
    int w_start = offsets[w], w_end = offsets[w + 1];
    int u_deg = u_end - u_start;
    int w_deg = w_end - w_start;

    
    const int32_t* search_base;
    const float* search_wt_base;
    int search_deg;
    const int32_t* target_base;
    const float* target_wt_base;
    int target_deg;

    if (u_deg <= w_deg) {
        search_base = indices + u_start; search_wt_base = weights + u_start; search_deg = u_deg;
        target_base = indices + w_start; target_wt_base = weights + w_start; target_deg = w_deg;
    } else {
        search_base = indices + w_start; search_wt_base = weights + w_start; search_deg = w_deg;
        target_base = indices + u_start; target_wt_base = weights + u_start; target_deg = u_deg;
    }

    float local_sum = 0.0f;

    
    bool has_overlap = true;
    if (lane == 0 && search_deg > 0 && target_deg > 0) {
        int32_t s_min = search_base[0], s_max = search_base[search_deg - 1];
        int32_t t_min = target_base[0], t_max = target_base[target_deg - 1];
        if (s_max < t_min || t_max < s_min) has_overlap = false;
    }
    has_overlap = __shfl_sync(0xFFFFFFFF, (int)has_overlap, 0);

    if (has_overlap && search_deg > 0 && target_deg > 0) {
        if (search_deg <= 2) {
            
            if (lane == 0) {
                int i = 0, j = 0;
                while (i < search_deg && j < target_deg) {
                    int32_t si = search_base[i];
                    int32_t tj = target_base[j];
                    if (si == tj) {
                        local_sum += fminf(search_wt_base[i], target_wt_base[j]);
                        i++; j++;
                    } else if (si < tj) {
                        i++;
                    } else {
                        j++;
                    }
                }
            }
        } else {
            
            
            int t_lo = 0, t_hi = target_deg;

            for (int i = lane; i < search_deg; i += 32) {
                int32_t val = search_base[i];
                float s_wt = search_wt_base[i];

                int lo = t_lo, hi = t_hi;
                while (lo < hi) {
                    int mid = lo + ((hi - lo) >> 1);
                    if (target_base[mid] < val) lo = mid + 1;
                    else hi = mid;
                }

                if (lo < target_deg && target_base[lo] == val) {
                    local_sum += fminf(s_wt, target_wt_base[lo]);
                }
            }
        }
    }

    
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    if (lane == 0) {
        float deg_sum = degrees[u] + degrees[w];
        float score = (deg_sum > 0.0f) ? (2.0f * local_sum / deg_sum) : 0.0f;

        
        out_score_keys[warp_id] = ~__float_as_uint(score);
        out_pair_values[warp_id] = ((uint64_t)(uint32_t)u << 32) | (uint64_t)(uint32_t)w;
    }
}

__global__ void unpack_sorted_results_kernel(
    const uint32_t* __restrict__ sorted_score_keys,
    const uint64_t* __restrict__ sorted_pair_values,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint64_t pk = sorted_pair_values[idx];
    out_first[idx] = (int32_t)(pk >> 32);
    out_second[idx] = (int32_t)(pk & 0xFFFFFFFF);
    out_scores[idx] = __uint_as_float(~sorted_score_keys[idx]);
}

}  

similarity_result_float_t sorensen_all_pairs_similarity_seg(const graph32_t& graph,
                                                            const float* edge_weights,
                                                            const int32_t* vertices,
                                                            std::size_t num_vertices,
                                                            std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    cudaStream_t stream = 0;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t nv = graph.number_of_vertices;
    const float* d_weights = edge_weights;

    
    const int32_t* d_src_vertices;
    int32_t num_sources;

    if (vertices != nullptr && num_vertices > 0) {
        d_src_vertices = vertices;
        num_sources = (int32_t)num_vertices;
    } else {
        cache.ensure_src_vertices(nv);
        if (nv > 0) {
            fill_sequence_kernel<<<(nv + 255) / 256, 256, 0, stream>>>(cache.src_vertices_buf, nv);
        }
        d_src_vertices = cache.src_vertices_buf;
        num_sources = nv;
    }

    if (num_sources == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int bits_per_vertex = 1;
    int v = nv - 1;
    while (v > 0) { bits_per_vertex++; v >>= 1; }
    int pair_bits = bits_per_vertex * 2;
    pair_bits = ((pair_bits + 7) / 8) * 8;
    pair_bits = std::min(64, 32 + bits_per_vertex);

    
    cache.ensure_degrees(nv);
    if (nv > 0) {
        compute_weighted_degrees_kernel<<<(nv + 255) / 256, 256, 0, stream>>>(
            d_offsets, d_weights, cache.degrees, nv);
    }

    
    cache.ensure_counts(num_sources);
    if (num_sources > 0) {
        count_pairs_kernel<<<num_sources, 256, 0, stream>>>(
            d_offsets, d_indices, d_src_vertices, num_sources, cache.counts);
    }

    
    cache.ensure_pair_offsets((size_t)num_sources + 1);
    cudaMemsetAsync(cache.pair_offsets, 0, sizeof(int64_t), stream);
    if (num_sources > 0) {
        thrust::inclusive_scan(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(cache.counts),
            thrust::device_pointer_cast(cache.counts + num_sources),
            thrust::device_pointer_cast(cache.pair_offsets + 1));
    }

    int64_t total_pairs;
    cudaMemcpyAsync(&total_pairs,
        cache.pair_offsets + num_sources,
        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (total_pairs == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    cache.ensure_pair_keys(total_pairs);
    if (num_sources > 0) {
        enumerate_pairs_kernel<<<num_sources, 256, 0, stream>>>(
            d_offsets, d_indices, d_src_vertices, num_sources,
            cache.pair_offsets, cache.pair_keys);
    }

    
    cache.ensure_pair_keys_sorted(total_pairs);
    cache.ensure_num_selected();

    
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, sort_temp_bytes,
        (int64_t*)nullptr, (int64_t*)nullptr,
        (int)total_pairs, 0, pair_bits, (cudaStream_t)0);

    size_t unique_temp_bytes = 0;
    cub::DeviceSelect::Unique(
        nullptr, unique_temp_bytes,
        (int64_t*)nullptr, (int64_t*)nullptr, (int64_t*)nullptr,
        (int)total_pairs, (cudaStream_t)0);

    size_t score_sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, score_sort_temp_bytes,
        (uint32_t*)nullptr, (uint32_t*)nullptr,
        (uint64_t*)nullptr, (uint64_t*)nullptr,
        (int)total_pairs, 0, 32, (cudaStream_t)0);

    size_t max_temp = std::max({sort_temp_bytes, unique_temp_bytes, score_sort_temp_bytes});
    cache.ensure_cub_temp(max_temp);

    
    cub::DeviceRadixSort::SortKeys(
        cache.cub_temp, sort_temp_bytes,
        cache.pair_keys, cache.pair_keys_sorted,
        (int)total_pairs, 0, pair_bits, stream);

    
    
    cub::DeviceSelect::Unique(
        cache.cub_temp, unique_temp_bytes,
        cache.pair_keys_sorted, cache.pair_keys,
        cache.num_selected,
        (int)total_pairs, stream);

    
    int64_t num_unique;
    cudaMemcpyAsync(&num_unique, cache.num_selected, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (num_unique == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    cache.ensure_sk_in(num_unique);
    cache.ensure_pv_in(num_unique);

    {
        int warps_per_block = 8;
        int block = warps_per_block * 32;
        int grid = (int)((num_unique + warps_per_block - 1) / warps_per_block);
        compute_sorensen_fused_kernel<<<grid, block, 0, stream>>>(
            d_offsets, d_indices, d_weights,
            cache.degrees,
            cache.pair_keys, num_unique,
            cache.sk_in, cache.pv_in);
    }

    
    cache.ensure_sk_out(num_unique);
    cache.ensure_pv_out(num_unique);

    cub::DeviceRadixSort::SortPairs(
        cache.cub_temp, score_sort_temp_bytes,
        cache.sk_in, cache.sk_out,
        cache.pv_in, cache.pv_out,
        (int)num_unique, 0, 32, stream);

    
    int64_t final_count = num_unique;
    if (topk.has_value() && (int64_t)topk.value() < num_unique) {
        final_count = (int64_t)topk.value();
    }

    
    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;
    cudaMalloc(&out_first, final_count * sizeof(int32_t));
    cudaMalloc(&out_second, final_count * sizeof(int32_t));
    cudaMalloc(&out_scores, final_count * sizeof(float));

    
    if (final_count > 0) {
        int block = 256;
        int grid = (int)((final_count + block - 1) / block);
        unpack_sorted_results_kernel<<<grid, block, 0, stream>>>(
            cache.sk_out, cache.pv_out,
            out_first, out_second, out_scores,
            final_count);
    }

    cudaStreamSynchronize(stream);

    return {out_first, out_second, out_scores, (std::size_t)final_count};
}

}  
