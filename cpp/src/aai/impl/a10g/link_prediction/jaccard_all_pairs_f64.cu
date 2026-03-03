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
#include <cstring>
#include <optional>
#include <vector>

namespace aai {

namespace {



struct Cache : Cacheable {
    
    
};



__global__ void iota_kernel(int32_t* out, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = idx;
}




__global__ void count_candidates_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ subset,
    int32_t num_subset,
    int64_t* __restrict__ counts)
{
    int src_idx = blockIdx.x;
    if (src_idx >= num_subset) return;

    int32_t u = subset[src_idx];
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];

    int64_t my_count = 0;
    for (int i = u_start; i < u_end; i++) {
        int32_t w = indices[i];
        int32_t w_start = offsets[w];
        int32_t w_end = offsets[w + 1];
        for (int j = w_start + threadIdx.x; j < w_end; j += blockDim.x) {
            if (indices[j] != u) my_count++;
        }
    }

    typedef cub::BlockReduce<int64_t, 256> BR;
    __shared__ typename BR::TempStorage temp;
    int64_t total = BR(temp).Sum(my_count);
    if (threadIdx.x == 0) counts[src_idx] = total;
}


__global__ void generate_candidates_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ subset,
    int32_t num_subset,
    const int64_t* __restrict__ write_offsets,
    uint64_t* __restrict__ pairs_out)
{
    int src_idx = blockIdx.x;
    if (src_idx >= num_subset) return;

    int32_t u = subset[src_idx];
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];

    __shared__ unsigned long long s_write_pos;
    if (threadIdx.x == 0) s_write_pos = (unsigned long long)write_offsets[src_idx];
    __syncthreads();

    for (int i = u_start; i < u_end; i++) {
        int32_t w = indices[i];
        int32_t w_start = offsets[w];
        int32_t w_end = offsets[w + 1];
        for (int j = w_start + threadIdx.x; j < w_end; j += blockDim.x) {
            int32_t v = indices[j];
            if (v != u) {
                unsigned long long pos = atomicAdd(&s_write_pos, 1ULL);
                pairs_out[pos] = ((uint64_t)(uint32_t)u << 32) | (uint64_t)(uint32_t)v;
            }
        }
    }
}




__global__ void compute_jaccard_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const uint64_t* __restrict__ pairs,
    int64_t num_pairs,
    int32_t* __restrict__ first_out,
    int32_t* __restrict__ second_out,
    double* __restrict__ scores_out)
{
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    uint64_t packed = pairs[warp_id];
    int32_t u = (int32_t)(packed >> 32);
    int32_t v = (int32_t)(packed & 0xFFFFFFFFULL);

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];
    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    
    double wsum_u = 0.0;
    for (int i = lane; i < u_deg; i += 32) {
        wsum_u += edge_weights[u_start + i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        wsum_u += __shfl_down_sync(0xFFFFFFFF, wsum_u, offset);
    }
    wsum_u = __shfl_sync(0xFFFFFFFF, wsum_u, 0);

    double wsum_v = 0.0;
    for (int i = lane; i < v_deg; i += 32) {
        wsum_v += edge_weights[v_start + i];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        wsum_v += __shfl_down_sync(0xFFFFFFFF, wsum_v, offset);
    }
    wsum_v = __shfl_sync(0xFFFFFFFF, wsum_v, 0);

    
    int32_t iter_start, iter_len, search_start, search_len;
    if (u_deg <= v_deg) {
        iter_start = u_start; iter_len = u_deg;
        search_start = v_start; search_len = v_deg;
    } else {
        iter_start = v_start; iter_len = v_deg;
        search_start = u_start; search_len = u_deg;
    }

    double my_isect = 0.0;
    for (int i = lane; i < iter_len; i += 32) {
        int32_t target = indices[iter_start + i];
        double w_iter = edge_weights[iter_start + i];

        int lo = 0, hi = search_len;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (indices[search_start + mid] < target) lo = mid + 1;
            else hi = mid;
        }

        if (lo < search_len && indices[search_start + lo] == target) {
            double w_search = edge_weights[search_start + lo];
            my_isect += fmin(w_iter, w_search);
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        my_isect += __shfl_down_sync(0xFFFFFFFF, my_isect, offset);
    }

    if (lane == 0) {
        double union_w = wsum_u + wsum_v - my_isect;
        double score = (union_w > 0.0) ? my_isect / union_w : 0.0;
        first_out[warp_id] = u;
        second_out[warp_id] = v;
        scores_out[warp_id] = score;
    }
}



__global__ void create_sort_keys_kernel(
    const double* __restrict__ scores,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    uint64_t* __restrict__ score_keys,
    uint64_t* __restrict__ pair_keys,
    int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double score = scores[idx];
    uint64_t bits;
    memcpy(&bits, &score, sizeof(uint64_t));
    uint64_t mask = (bits >> 63) ? ~0ULL : (1ULL << 63);
    score_keys[idx] = ~(bits ^ mask);

    pair_keys[idx] = ((uint64_t)(uint32_t)first[idx] << 32) | (uint64_t)(uint32_t)second[idx];
}

__global__ void gather_uint64_kernel(
    const uint64_t* __restrict__ src,
    const int32_t* __restrict__ idx,
    uint64_t* __restrict__ dst,
    int64_t n)
{
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
}

__global__ void gather_results_kernel(
    const int32_t* __restrict__ sf, const int32_t* __restrict__ ss,
    const double* __restrict__ ssc,
    const int32_t* __restrict__ perm,
    int32_t* __restrict__ df, int32_t* __restrict__ ds,
    double* __restrict__ dsc,
    int64_t n)
{
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int32_t src = perm[idx];
    df[idx] = sf[src];
    ds[idx] = ss[src];
    dsc[idx] = ssc[src];
}

}  

similarity_result_double_t jaccard_all_pairs_similarity(const graph32_t& graph,
                                                        const double* edge_weights,
                                                        const int32_t* vertices,
                                                        std::size_t num_vertices,
                                                        std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t nv = graph.number_of_vertices;
    cudaStream_t stream = 0;

    
    const int32_t* d_subset = vertices;
    int32_t num_subset = static_cast<int32_t>(num_vertices);
    int32_t* iota_buf = nullptr;

    if (d_subset == nullptr) {
        num_subset = nv;
        cudaMalloc(&iota_buf, (size_t)nv * sizeof(int32_t));
        if (nv > 0) iota_kernel<<<(nv + 255) / 256, 256, 0, stream>>>(iota_buf, nv);
        d_subset = iota_buf;
    }

    
    int64_t* d_counts = nullptr;
    cudaMalloc(&d_counts, (size_t)num_subset * sizeof(int64_t));
    if (num_subset > 0)
        count_candidates_kernel<<<num_subset, 256, 0, stream>>>(d_offsets, d_indices, d_subset, num_subset, d_counts);

    
    std::vector<int64_t> h_counts(num_subset);
    cudaMemcpyAsync(h_counts.data(), d_counts,
                     num_subset * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_counts);

    
    std::vector<int64_t> h_offsets(num_subset);
    int64_t total_candidates = 0;
    for (int64_t i = 0; i < num_subset; i++) {
        h_offsets[i] = total_candidates;
        total_candidates += h_counts[i];
    }

    if (total_candidates <= 0) {
        if (iota_buf) cudaFree(iota_buf);
        int32_t* out_f = nullptr; cudaMalloc(&out_f, sizeof(int32_t));
        int32_t* out_s = nullptr; cudaMalloc(&out_s, sizeof(int32_t));
        double* out_sc = nullptr; cudaMalloc(&out_sc, sizeof(double));
        return {out_f, out_s, out_sc, 0};
    }

    
    int64_t* d_write_offsets = nullptr;
    cudaMalloc(&d_write_offsets, (size_t)num_subset * sizeof(int64_t));
    cudaMemcpyAsync(d_write_offsets, h_offsets.data(),
                     num_subset * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    
    uint64_t* d_candidates = nullptr;
    cudaMalloc(&d_candidates, (size_t)total_candidates * sizeof(uint64_t));
    if (num_subset > 0)
        generate_candidates_kernel<<<num_subset, 256, 0, stream>>>(
            d_offsets, d_indices, d_subset, num_subset,
            d_write_offsets, d_candidates);
    cudaFree(d_write_offsets);

    
    uint64_t* d_sorted_candidates = nullptr;
    cudaMalloc(&d_sorted_candidates, (size_t)total_candidates * sizeof(uint64_t));
    size_t sort_temp_size = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, sort_temp_size, (uint64_t*)nullptr, (uint64_t*)nullptr, total_candidates);
    if (sort_temp_size == 0) sort_temp_size = 1;
    void* d_sort_temp = nullptr;
    cudaMalloc(&d_sort_temp, sort_temp_size);
    cub::DeviceRadixSort::SortKeys(d_sort_temp, sort_temp_size,
        d_candidates, d_sorted_candidates, total_candidates, 0, 64, stream);
    cudaFree(d_sort_temp);
    cudaFree(d_candidates);

    
    uint64_t* d_unique_pairs = nullptr;
    cudaMalloc(&d_unique_pairs, (size_t)total_candidates * sizeof(uint64_t));
    int64_t* d_unique_count = nullptr;
    cudaMalloc(&d_unique_count, sizeof(int64_t));
    size_t unique_temp_size = 0;
    cub::DeviceSelect::Unique(nullptr, unique_temp_size, (uint64_t*)nullptr, (uint64_t*)nullptr, (int64_t*)nullptr, total_candidates);
    if (unique_temp_size == 0) unique_temp_size = 1;
    void* d_unique_temp = nullptr;
    cudaMalloc(&d_unique_temp, unique_temp_size);
    cub::DeviceSelect::Unique(d_unique_temp, unique_temp_size,
        d_sorted_candidates, d_unique_pairs, d_unique_count,
        total_candidates, stream);
    cudaFree(d_unique_temp);
    cudaFree(d_sorted_candidates);

    int64_t h_num_unique;
    cudaMemcpyAsync(&h_num_unique, d_unique_count, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_unique_count);

    if (h_num_unique == 0) {
        cudaFree(d_unique_pairs);
        if (iota_buf) cudaFree(iota_buf);
        int32_t* out_f = nullptr; cudaMalloc(&out_f, sizeof(int32_t));
        int32_t* out_s = nullptr; cudaMalloc(&out_s, sizeof(int32_t));
        double* out_sc = nullptr; cudaMalloc(&out_sc, sizeof(double));
        return {out_f, out_s, out_sc, 0};
    }

    
    int32_t* d_first = nullptr;
    int32_t* d_second = nullptr;
    double* d_scores = nullptr;
    cudaMalloc(&d_first, (size_t)h_num_unique * sizeof(int32_t));
    cudaMalloc(&d_second, (size_t)h_num_unique * sizeof(int32_t));
    cudaMalloc(&d_scores, (size_t)h_num_unique * sizeof(double));
    {
        int warps_per_block = 8;
        int threads = warps_per_block * 32;
        int blocks = ((int)h_num_unique + warps_per_block - 1) / warps_per_block;
        compute_jaccard_kernel<<<blocks, threads, 0, stream>>>(
            d_offsets, d_indices, edge_weights,
            d_unique_pairs, h_num_unique,
            d_first, d_second, d_scores);
    }
    cudaFree(d_unique_pairs);

    
    uint64_t* d_score_keys = nullptr;
    uint64_t* d_pair_keys = nullptr;
    cudaMalloc(&d_score_keys, (size_t)h_num_unique * sizeof(uint64_t));
    cudaMalloc(&d_pair_keys, (size_t)h_num_unique * sizeof(uint64_t));
    create_sort_keys_kernel<<<((int)h_num_unique + 255) / 256, 256, 0, stream>>>(
        d_scores, d_first, d_second, d_score_keys, d_pair_keys, h_num_unique);

    int32_t* d_idx_buf = nullptr;
    cudaMalloc(&d_idx_buf, (size_t)h_num_unique * sizeof(int32_t));
    iota_kernel<<<((int)h_num_unique + 255) / 256, 256, 0, stream>>>(d_idx_buf, (int32_t)h_num_unique);

    size_t sp_temp_size = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sp_temp_size, (uint64_t*)nullptr, (uint64_t*)nullptr, (int32_t*)nullptr, (int32_t*)nullptr, h_num_unique);
    if (sp_temp_size == 0) sp_temp_size = 1;
    void* d_sp_temp = nullptr;
    cudaMalloc(&d_sp_temp, sp_temp_size);

    
    uint64_t* d_pair_keys_sorted = nullptr;
    int32_t* d_idx_sorted1 = nullptr;
    cudaMalloc(&d_pair_keys_sorted, (size_t)h_num_unique * sizeof(uint64_t));
    cudaMalloc(&d_idx_sorted1, (size_t)h_num_unique * sizeof(int32_t));
    cub::DeviceRadixSort::SortPairs(d_sp_temp, sp_temp_size,
        d_pair_keys, d_pair_keys_sorted,
        d_idx_buf, d_idx_sorted1,
        h_num_unique, 0, 64, stream);
    cudaFree(d_pair_keys);
    cudaFree(d_pair_keys_sorted);
    cudaFree(d_idx_buf);

    
    uint64_t* d_score_keys_reordered = nullptr;
    cudaMalloc(&d_score_keys_reordered, (size_t)h_num_unique * sizeof(uint64_t));
    gather_uint64_kernel<<<((int)h_num_unique + 255) / 256, 256, 0, stream>>>(
        d_score_keys, d_idx_sorted1, d_score_keys_reordered, h_num_unique);
    cudaFree(d_score_keys);

    
    uint64_t* d_score_keys_sorted = nullptr;
    int32_t* d_idx_sorted2 = nullptr;
    cudaMalloc(&d_score_keys_sorted, (size_t)h_num_unique * sizeof(uint64_t));
    cudaMalloc(&d_idx_sorted2, (size_t)h_num_unique * sizeof(int32_t));
    cub::DeviceRadixSort::SortPairs(d_sp_temp, sp_temp_size,
        d_score_keys_reordered, d_score_keys_sorted,
        d_idx_sorted1, d_idx_sorted2,
        h_num_unique, 0, 64, stream);
    cudaFree(d_sp_temp);
    cudaFree(d_score_keys_reordered);
    cudaFree(d_score_keys_sorted);
    cudaFree(d_idx_sorted1);

    
    int64_t output_count = h_num_unique;
    if (topk.has_value() && (int64_t)topk.value() < output_count) {
        output_count = (int64_t)topk.value();
    }

    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    double* out_scores = nullptr;
    cudaMalloc(&out_first, (size_t)output_count * sizeof(int32_t));
    cudaMalloc(&out_second, (size_t)output_count * sizeof(int32_t));
    cudaMalloc(&out_scores, (size_t)output_count * sizeof(double));

    gather_results_kernel<<<((int)output_count + 255) / 256, 256, 0, stream>>>(
        d_first, d_second, d_scores,
        d_idx_sorted2,
        out_first, out_second, out_scores, output_count);

    
    cudaFree(d_first);
    cudaFree(d_second);
    cudaFree(d_scores);
    cudaFree(d_idx_sorted2);
    if (iota_buf) cudaFree(iota_buf);

    cudaStreamSynchronize(stream);

    return {out_first, out_second, out_scores, (std::size_t)output_count};
}

}  
