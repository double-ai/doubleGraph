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
#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <cstdint>
#include <algorithm>
#include <optional>

namespace aai {

namespace {

struct Cache : Cacheable {
    void* cub_temp = nullptr;
    size_t cub_temp_capacity = 0;

    void ensure_cub_temp(size_t needed) {
        if (needed > cub_temp_capacity) {
            if (cub_temp) cudaFree(cub_temp);
            cub_temp_capacity = std::max(needed * 2, (size_t)(32 * 1024 * 1024));
            cudaMalloc(&cub_temp, cub_temp_capacity);
        }
    }

    ~Cache() override {
        if (cub_temp) {
            cudaFree(cub_temp);
            cub_temp = nullptr;
        }
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



void launch_fill_sequence(int32_t* arr, int32_t n, cudaStream_t stream) {
    if (n == 0) return;
    fill_sequence_kernel<<<(n + 255) / 256, 256, 0, stream>>>(arr, n);
}

void launch_compute_weighted_degrees(
    const int32_t* offsets, const float* weights,
    float* degrees, int32_t num_vertices, cudaStream_t stream)
{
    if (num_vertices == 0) return;
    compute_weighted_degrees_kernel<<<(num_vertices + 255) / 256, 256, 0, stream>>>(
        offsets, weights, degrees, num_vertices);
}

void launch_count_pairs(
    const int32_t* offsets, const int32_t* indices,
    const int32_t* src_vertices, int32_t num_sources,
    int64_t* counts, cudaStream_t stream)
{
    if (num_sources == 0) return;
    count_pairs_kernel<<<num_sources, 256, 0, stream>>>(
        offsets, indices, src_vertices, num_sources, counts);
}

void launch_inclusive_scan(const int64_t* input, int64_t* output, int32_t n, cudaStream_t stream) {
    if (n == 0) return;
    thrust::inclusive_scan(
        thrust::cuda::par.on(stream),
        thrust::device_pointer_cast(input),
        thrust::device_pointer_cast(input + n),
        thrust::device_pointer_cast(output));
}

void launch_enumerate_pairs(
    const int32_t* offsets, const int32_t* indices,
    const int32_t* src_vertices, int32_t num_sources,
    const int64_t* pair_offsets, int64_t* pair_keys, cudaStream_t stream)
{
    if (num_sources == 0) return;
    enumerate_pairs_kernel<<<num_sources, 256, 0, stream>>>(
        offsets, indices, src_vertices, num_sources, pair_offsets, pair_keys);
}


int64_t launch_sort_and_unique_cub(
    int64_t* keys_in, int64_t* keys_out,
    int64_t count,
    int num_bits,
    void* sort_temp, size_t sort_temp_bytes,
    void* unique_temp, size_t unique_temp_bytes,
    int64_t* num_selected_out,
    cudaStream_t stream)
{
    if (count == 0) return 0;

    
    cub::DeviceRadixSort::SortKeys(
        sort_temp, sort_temp_bytes,
        keys_in, keys_out,
        (int)count, 0, num_bits, stream);

    
    cub::DeviceSelect::Unique(
        unique_temp, unique_temp_bytes,
        keys_out, keys_in,  
        num_selected_out,
        (int)count, stream);

    
    int64_t result;
    cudaMemcpyAsync(&result, num_selected_out, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return result;
}


void get_cub_temp_sizes(
    int64_t max_pairs,
    int num_bits,
    size_t* sort_temp_bytes,
    size_t* unique_temp_bytes,
    size_t* score_sort_temp_bytes)
{
    *sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(
        nullptr, *sort_temp_bytes,
        (int64_t*)nullptr, (int64_t*)nullptr,
        (int)max_pairs, 0, num_bits, (cudaStream_t)0);

    *unique_temp_bytes = 0;
    cub::DeviceSelect::Unique(
        nullptr, *unique_temp_bytes,
        (int64_t*)nullptr, (int64_t*)nullptr, (int64_t*)nullptr,
        (int)max_pairs, (cudaStream_t)0);

    *score_sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, *score_sort_temp_bytes,
        (uint32_t*)nullptr, (uint32_t*)nullptr,
        (uint64_t*)nullptr, (uint64_t*)nullptr,
        (int)max_pairs, 0, 32, (cudaStream_t)0);
}

void launch_compute_sorensen_fused(
    const int32_t* offsets, const int32_t* indices, const float* weights,
    const float* degrees,
    const int64_t* unique_pairs, int64_t num_pairs,
    uint32_t* score_keys, uint64_t* pair_values,
    cudaStream_t stream)
{
    if (num_pairs == 0) return;
    int warps_per_block = 8;
    int block = warps_per_block * 32;
    int grid = (int)((num_pairs + warps_per_block - 1) / warps_per_block);
    compute_sorensen_fused_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, weights, degrees, unique_pairs, num_pairs,
        score_keys, pair_values);
}

void launch_score_sort(
    uint32_t* keys_in, uint32_t* keys_out,
    uint64_t* values_in, uint64_t* values_out,
    void* temp, size_t temp_bytes,
    int64_t count, cudaStream_t stream)
{
    if (count == 0) return;
    cub::DeviceRadixSort::SortPairs(
        temp, temp_bytes,
        keys_in, keys_out,
        values_in, values_out,
        (int)count, 0, 32, stream);
}

void launch_unpack(
    const uint32_t* score_keys, const uint64_t* pair_values,
    int32_t* first, int32_t* second, float* scores,
    int64_t count, cudaStream_t stream)
{
    if (count == 0) return;
    int block = 256;
    int grid = (int)((count + block - 1) / block);
    unpack_sorted_results_kernel<<<grid, block, 0, stream>>>(
        score_keys, pair_values, first, second, scores, count);
}

}  

similarity_result_float_t sorensen_all_pairs_similarity_seg(
    const graph32_t& graph,
    const float* edge_weights,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    cudaStream_t stream = 0;
    int32_t n_verts = graph.number_of_vertices;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const float* d_weights = edge_weights;

    
    int32_t* d_src_vertices_buf = nullptr;
    const int32_t* d_src_vertices;
    int32_t num_sources;

    if (vertices != nullptr && num_vertices > 0) {
        d_src_vertices = vertices;
        num_sources = (int32_t)num_vertices;
    } else {
        if (n_verts > 0) {
            cudaMalloc(&d_src_vertices_buf, (size_t)n_verts * sizeof(int32_t));
            launch_fill_sequence(d_src_vertices_buf, n_verts, stream);
        }
        d_src_vertices = d_src_vertices_buf;
        num_sources = n_verts;
    }

    if (num_sources == 0) {
        if (d_src_vertices_buf) cudaFree(d_src_vertices_buf);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int bits_per_vertex = 1;
    int v = n_verts - 1;
    while (v > 0) { bits_per_vertex++; v >>= 1; }
    int pair_bits = bits_per_vertex * 2;
    pair_bits = ((pair_bits + 7) / 8) * 8;
    pair_bits = std::min(64, 32 + bits_per_vertex);

    
    float* degrees = nullptr;
    cudaMalloc(&degrees, (size_t)n_verts * sizeof(float));
    launch_compute_weighted_degrees(d_offsets, d_weights, degrees, n_verts, stream);

    
    int64_t* counts = nullptr;
    cudaMalloc(&counts, (size_t)num_sources * sizeof(int64_t));
    launch_count_pairs(d_offsets, d_indices, d_src_vertices, num_sources, counts, stream);

    
    int64_t* pair_offsets = nullptr;
    cudaMalloc(&pair_offsets, ((size_t)num_sources + 1) * sizeof(int64_t));
    cudaMemsetAsync(pair_offsets, 0, sizeof(int64_t), stream);
    launch_inclusive_scan(counts, pair_offsets + 1, num_sources, stream);

    int64_t total_pairs;
    cudaMemcpyAsync(&total_pairs, pair_offsets + num_sources,
        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cudaFree(counts);

    if (total_pairs == 0) {
        cudaFree(pair_offsets);
        cudaFree(degrees);
        if (d_src_vertices_buf) cudaFree(d_src_vertices_buf);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t* pair_keys = nullptr;
    cudaMalloc(&pair_keys, (size_t)total_pairs * sizeof(int64_t));
    launch_enumerate_pairs(d_offsets, d_indices, d_src_vertices, num_sources,
        pair_offsets, pair_keys, stream);

    cudaFree(pair_offsets);
    if (d_src_vertices_buf) { cudaFree(d_src_vertices_buf); d_src_vertices_buf = nullptr; }

    
    int64_t* pair_keys_sorted = nullptr;
    cudaMalloc(&pair_keys_sorted, (size_t)total_pairs * sizeof(int64_t));
    int64_t* num_selected_out = nullptr;
    cudaMalloc(&num_selected_out, sizeof(int64_t));

    
    size_t sort_temp, unique_temp, score_sort_temp;
    get_cub_temp_sizes(total_pairs, pair_bits, &sort_temp, &unique_temp, &score_sort_temp);
    size_t max_temp = std::max({sort_temp, unique_temp, score_sort_temp});
    cache.ensure_cub_temp(max_temp);

    int64_t num_unique = launch_sort_and_unique_cub(
        pair_keys, pair_keys_sorted,
        total_pairs, pair_bits,
        cache.cub_temp, sort_temp,
        cache.cub_temp, unique_temp,
        num_selected_out, stream);

    cudaFree(pair_keys_sorted);
    cudaFree(num_selected_out);

    if (num_unique == 0) {
        cudaFree(pair_keys);
        cudaFree(degrees);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    
    uint32_t* sk_in = nullptr;
    uint64_t* pv_in = nullptr;
    cudaMalloc(&sk_in, (size_t)num_unique * sizeof(uint32_t));
    cudaMalloc(&pv_in, (size_t)num_unique * sizeof(uint64_t));

    launch_compute_sorensen_fused(d_offsets, d_indices, d_weights,
        degrees, pair_keys, num_unique,
        sk_in, pv_in, stream);

    cudaFree(pair_keys);
    cudaFree(degrees);

    
    uint32_t* sk_out = nullptr;
    uint64_t* pv_out = nullptr;
    cudaMalloc(&sk_out, (size_t)num_unique * sizeof(uint32_t));
    cudaMalloc(&pv_out, (size_t)num_unique * sizeof(uint64_t));

    launch_score_sort(sk_in, sk_out, pv_in, pv_out,
        cache.cub_temp, score_sort_temp,
        num_unique, stream);

    cudaFree(sk_in);
    cudaFree(pv_in);

    
    int64_t final_count = num_unique;
    if (topk.has_value() && (int64_t)topk.value() < num_unique) {
        final_count = (int64_t)topk.value();
    }

    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;
    cudaMalloc(&out_first, (size_t)final_count * sizeof(int32_t));
    cudaMalloc(&out_second, (size_t)final_count * sizeof(int32_t));
    cudaMalloc(&out_scores, (size_t)final_count * sizeof(float));

    launch_unpack(sk_out, pv_out, out_first, out_second, out_scores, final_count, stream);

    cudaStreamSynchronize(stream);

    cudaFree(sk_out);
    cudaFree(pv_out);

    return {out_first, out_second, out_scores, (std::size_t)final_count};
}

}  
