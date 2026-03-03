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
#include <cstddef>
#include <optional>
#include <vector>
#include <cub/cub.cuh>

namespace aai {

namespace {

struct Cache : Cacheable {
    void* scratch = nullptr;
    size_t scratch_size = 0;
    void* staging = nullptr;
    size_t staging_size = 0;
    int32_t* d_all_seeds = nullptr;
    int all_seeds_capacity = 0;

    Cache() {
        scratch_size = 512ULL * 1024 * 1024;
        cudaMalloc(&scratch, scratch_size);
        staging_size = 64ULL * 1024 * 1024;
        cudaMalloc(&staging, staging_size);
    }

    ~Cache() override {
        if (scratch) cudaFree(scratch);
        if (staging) cudaFree(staging);
        if (d_all_seeds) cudaFree(d_all_seeds);
    }

    void ensure_staging(size_t needed) {
        if (needed > staging_size) {
            if (staging) cudaFree(staging);
            staging_size = needed;
            cudaMalloc(&staging, staging_size);
        }
    }

    void ensure_seeds(int n) {
        if (n > all_seeds_capacity) {
            if (d_all_seeds) cudaFree(d_all_seeds);
            all_seeds_capacity = n;
            cudaMalloc(&d_all_seeds, n * sizeof(int32_t));
        }
    }
};



__global__ void count_expanded_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    int64_t* __restrict__ counts
) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_seeds) return;
    int u = seeds[sid];
    int s = offsets[u], e = offsets[u + 1];
    int64_t c = 0;
    for (int i = s; i < e; i++)
        c += (int64_t)(offsets[indices[i] + 1] - offsets[indices[i]]);
    counts[sid] = c;
}


__global__ void generate_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    const int64_t* __restrict__ seed_pair_offsets,
    int64_t* __restrict__ pair_keys,
    int64_t V
) {
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int u = seeds[sid];
    int s_u = offsets[u], e_u = offsets[u + 1];
    int64_t base = seed_pair_offsets[sid];
    int64_t sid_base = (int64_t)sid * V;
    int64_t local_off = 0;
    for (int ni = s_u; ni < e_u; ni++) {
        int w = indices[ni];
        int s_w = offsets[w], e_w = offsets[w + 1];
        int deg_w = e_w - s_w;
        for (int j = threadIdx.x; j < deg_w; j += blockDim.x)
            pair_keys[base + local_off + j] = sid_base + (int64_t)indices[s_w + j];
        local_off += deg_w;
    }
}


__global__ void compute_overlap_kernel(
    const int64_t* __restrict__ unique_keys,
    const int32_t* __restrict__ rle_counts,
    int num_unique,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int64_t V,
    bool is_multigraph,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int32_t* __restrict__ valid_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_unique) return;
    int64_t key = unique_keys[idx];
    int32_t sid = (int32_t)(key / V);
    int32_t v = (int32_t)(key % V);
    int32_t u = seeds[sid];
    if (u == v) return;

    int deg_u = offsets[u + 1] - offsets[u];
    int deg_v = offsets[v + 1] - offsets[v];
    int min_deg = (deg_u < deg_v) ? deg_u : deg_v;
    if (min_deg == 0) return;

    int intersection;
    if (!is_multigraph) {
        intersection = rle_counts[idx];
    } else {
        int s_u = offsets[u], e_u = offsets[u + 1];
        int s_v = offsets[v], e_v = offsets[v + 1];
        int i = s_u, j = s_v;
        intersection = 0;
        while (i < e_u && j < e_v) {
            int a = indices[i], b = indices[j];
            if (a == b) { intersection++; i++; j++; }
            else if (a < b) { i++; }
            else { j++; }
        }
    }
    if (intersection == 0) return;

    float score = __fdividef((float)intersection, (float)min_deg);
    int pos = atomicAdd(valid_count, 1);
    out_first[pos] = u;
    out_second[pos] = v;
    out_scores[pos] = score;
}

__global__ void gather_topk_kernel(
    const int32_t* __restrict__ sorted_indices,
    const int32_t* __restrict__ in_first,
    const int32_t* __restrict__ in_second,
    const float* __restrict__ in_scores,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    int src = sorted_indices[idx];
    out_first[idx] = in_first[src];
    out_second[idx] = in_second[src];
    out_scores[idx] = in_scores[src];
}

__global__ void iota_kernel(int32_t* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = idx;
}


static int compute_num_bits(int64_t max_val) {
    if (max_val <= 0) return 8;
    int bits = 0;
    while (max_val > 0) { bits++; max_val >>= 1; }
    return (bits < 8) ? 8 : bits;
}

static void overlap_compute(
    const int32_t* d_offsets, const int32_t* d_indices,
    int32_t num_vertices, int32_t num_edges, bool is_multigraph,
    const int32_t* d_seeds, int num_seeds, int64_t topk,
    void** pp_scratch, size_t* p_scratch_size,
    int32_t* d_out_first, int32_t* d_out_second, float* d_out_scores,
    int64_t max_output, int64_t* h_out_count
) {
    if (num_seeds == 0) { *h_out_count = 0; return; }

    cudaStream_t stream = 0;
    int64_t V = (int64_t)num_vertices;

    
    int64_t max_key = (int64_t)(num_seeds - 1) * V + (V - 1);
    int sort_bits = compute_num_bits(max_key);

    if (*p_scratch_size < 512ULL*1024*1024) {
        if (*pp_scratch) cudaFree(*pp_scratch);
        *p_scratch_size = 512ULL*1024*1024;
        cudaMalloc(pp_scratch, *p_scratch_size);
    }

    char* scratch = (char*)*pp_scratch;
    size_t soff = 0;
    auto salign = [&]() { soff = (soff + 255) & ~255ULL; };
    auto salloc = [&](size_t bytes) -> void* {
        salign(); void* p = scratch + soff; soff += bytes; return p;
    };

    int64_t* d_counts = (int64_t*)salloc(num_seeds * sizeof(int64_t));
    int64_t* d_offsets_ps = (int64_t*)salloc((num_seeds + 1) * sizeof(int64_t));

    {
        int grid = (num_seeds + 255) / 256;
        count_expanded_kernel<<<grid, 256, 0, stream>>>(
            d_offsets, d_indices, d_seeds, num_seeds, d_counts);
    }
    {
        void* d_temp = nullptr; size_t ts = 0;
        cub::DeviceScan::ExclusiveSum(d_temp, ts, d_counts, d_offsets_ps, num_seeds, stream);
        d_temp = salloc(ts);
        cub::DeviceScan::ExclusiveSum(d_temp, ts, d_counts, d_offsets_ps, num_seeds, stream);
    }

    
    int64_t h_last[2];
    cudaMemcpyAsync(&h_last[0], d_offsets_ps + num_seeds - 1, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_last[1], d_counts + num_seeds - 1, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int64_t total_expanded = h_last[0] + h_last[1];

    if (total_expanded == 0) { *h_out_count = 0; return; }

    
    size_t need = soff;
    need += 256 + total_expanded * 8;
    need += 256 + total_expanded * 8;
    need += 256 + total_expanded * 8 + 64*1024*1024;
    need += 256 + total_expanded * 4;
    need += 256 + 4;
    need += 256 + 16*1024*1024;
    need += 256 + total_expanded * 4 * 2 + total_expanded * 4;
    need += 256 + 4;
    need += 256 + total_expanded * 4 * 2 + total_expanded * 4;
    need += 256 + total_expanded * 8 + 64*1024*1024;

    if (need > *p_scratch_size) {
        if (*pp_scratch) cudaFree(*pp_scratch);
        *p_scratch_size = need + 64*1024*1024;
        cudaMalloc(pp_scratch, *p_scratch_size);
        scratch = (char*)*pp_scratch;
        soff = 0;
        d_counts = (int64_t*)salloc(num_seeds * sizeof(int64_t));
        d_offsets_ps = (int64_t*)salloc((num_seeds + 1) * sizeof(int64_t));
        {
            int grid = (num_seeds + 255) / 256;
            count_expanded_kernel<<<grid, 256, 0, stream>>>(
                d_offsets, d_indices, d_seeds, num_seeds, d_counts);
        }
        {
            void* d_temp = nullptr; size_t ts = 0;
            cub::DeviceScan::ExclusiveSum(d_temp, ts, d_counts, d_offsets_ps, num_seeds, stream);
            d_temp = salloc(ts);
            cub::DeviceScan::ExclusiveSum(d_temp, ts, d_counts, d_offsets_ps, num_seeds, stream);
        }
    }

    
    int64_t* d_pair_keys = (int64_t*)salloc(total_expanded * sizeof(int64_t));
    generate_pairs_kernel<<<num_seeds, 256, 0, stream>>>(
        d_offsets, d_indices, d_seeds, num_seeds, d_offsets_ps, d_pair_keys, V);

    
    int64_t* d_sorted_keys = (int64_t*)salloc(total_expanded * sizeof(int64_t));
    {
        void* d_temp = nullptr; size_t ts = 0;
        cub::DeviceRadixSort::SortKeys(d_temp, ts,
            d_pair_keys, d_sorted_keys, total_expanded, 0, sort_bits, stream);
        d_temp = salloc(ts);
        cub::DeviceRadixSort::SortKeys(d_temp, ts,
            d_pair_keys, d_sorted_keys, total_expanded, 0, sort_bits, stream);
    }

    
    int64_t* d_unique_keys = d_pair_keys;
    int32_t* d_rle_counts = (int32_t*)salloc(total_expanded * sizeof(int32_t));
    int32_t* d_num_runs = (int32_t*)salloc(sizeof(int32_t));
    {
        void* d_temp = nullptr; size_t ts = 0;
        cub::DeviceRunLengthEncode::Encode(d_temp, ts,
            d_sorted_keys, d_unique_keys, d_rle_counts, d_num_runs,
            total_expanded, stream);
        d_temp = salloc(ts);
        cub::DeviceRunLengthEncode::Encode(d_temp, ts,
            d_sorted_keys, d_unique_keys, d_rle_counts, d_num_runs,
            total_expanded, stream);
    }

    
    int32_t num_unique;
    cudaMemcpy(&num_unique, d_num_runs, sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (num_unique == 0) { *h_out_count = 0; return; }

    
    int32_t* d_temp_first = (int32_t*)salloc(num_unique * sizeof(int32_t));
    int32_t* d_temp_second = (int32_t*)salloc(num_unique * sizeof(int32_t));
    float* d_temp_scores = (float*)salloc(num_unique * sizeof(float));
    int32_t* d_valid_count = (int32_t*)salloc(sizeof(int32_t));
    cudaMemsetAsync(d_valid_count, 0, sizeof(int32_t), stream);

    {
        int grid = (num_unique + 255) / 256;
        compute_overlap_kernel<<<grid, 256, 0, stream>>>(
            d_unique_keys, d_rle_counts, num_unique,
            d_offsets, d_indices, d_seeds, V, is_multigraph,
            d_temp_first, d_temp_second, d_temp_scores, d_valid_count);
    }

    
    int32_t valid_count;
    cudaMemcpy(&valid_count, d_valid_count, sizeof(int32_t), cudaMemcpyDeviceToHost);
    if (valid_count == 0) { *h_out_count = 0; return; }

    
    int64_t output_count;
    if (topk >= 0 && (int64_t)valid_count > topk) {
        output_count = topk;
        int32_t* d_idx_in = (int32_t*)salloc(valid_count * sizeof(int32_t));
        int32_t* d_idx_out = (int32_t*)salloc(valid_count * sizeof(int32_t));
        float* d_scores_out = (float*)salloc(valid_count * sizeof(float));

        { int g = (valid_count+255)/256; iota_kernel<<<g,256,0,stream>>>(d_idx_in, valid_count); }
        {
            void* d_temp = nullptr; size_t ts = 0;
            cub::DeviceRadixSort::SortPairsDescending(d_temp, ts,
                d_temp_scores, d_scores_out, d_idx_in, d_idx_out,
                valid_count, 0, 32, stream);
            d_temp = salloc(ts);
            cub::DeviceRadixSort::SortPairsDescending(d_temp, ts,
                d_temp_scores, d_scores_out, d_idx_in, d_idx_out,
                valid_count, 0, 32, stream);
        }
        { int g = ((int)output_count+255)/256;
          gather_topk_kernel<<<g,256,0,stream>>>(
              d_idx_out, d_temp_first, d_temp_second, d_temp_scores,
              d_out_first, d_out_second, d_out_scores, (int)output_count); }
    } else {
        output_count = valid_count;
        if (output_count > max_output) output_count = max_output;
        cudaMemcpyAsync(d_out_first, d_temp_first, output_count * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_out_second, d_temp_second, output_count * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_out_scores, d_temp_scores, output_count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);
    *h_out_count = output_count;
}

}  

similarity_result_float_t overlap_all_pairs_similarity_seg(const graph32_t& graph,
                                                           const int32_t* vertices,
                                                           std::size_t num_vertices,
                                                           std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets_ptr = graph.offsets;
    const int32_t* d_indices_ptr = graph.indices;
    int32_t num_verts = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    bool is_multigraph = graph.is_multigraph;

    const int32_t* d_seeds = nullptr;
    int num_seeds = 0;

    if (vertices != nullptr && num_vertices > 0) {
        d_seeds = vertices;
        num_seeds = (int)num_vertices;
    } else {
        const auto& seg = graph.segment_offsets.value();
        num_seeds = seg[3];
        if (num_seeds > 0) {
            cache.ensure_seeds(num_seeds);
            std::vector<int32_t> h_seeds(num_seeds);
            for (int i = 0; i < num_seeds; i++) h_seeds[i] = i;
            cudaMemcpy(cache.d_all_seeds, h_seeds.data(), num_seeds * sizeof(int32_t), cudaMemcpyHostToDevice);
            d_seeds = cache.d_all_seeds;
        }
    }

    int64_t topk_raw = topk.has_value() ? (int64_t)topk.value() : -1;
    int64_t max_output = topk.has_value() ? (int64_t)topk.value()
        : (int64_t)num_seeds * num_verts;

    size_t staging_needed = max_output * (2 * sizeof(int32_t) + sizeof(float));
    cache.ensure_staging(staging_needed > cache.staging_size ? staging_needed : cache.staging_size);

    char* stage = (char*)cache.staging;
    int32_t* stg_first  = (int32_t*)(stage);
    int32_t* stg_second = stg_first + max_output;
    float*   stg_scores = (float*)(stg_second + max_output);

    int64_t count = 0;
    if (num_seeds > 0) {
        overlap_compute(
            d_offsets_ptr, d_indices_ptr, num_verts, num_edges, is_multigraph,
            d_seeds, num_seeds, topk_raw,
            &cache.scratch, &cache.scratch_size,
            stg_first, stg_second, stg_scores,
            max_output, &count);
    }

    similarity_result_float_t result;
    result.count = (std::size_t)count;
    if (count > 0) {
        cudaMalloc(&result.first, count * sizeof(int32_t));
        cudaMalloc(&result.second, count * sizeof(int32_t));
        cudaMalloc(&result.scores, count * sizeof(float));
        cudaMemcpy(result.first, stg_first, count * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(result.second, stg_second, count * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(result.scores, stg_scores, count * sizeof(float), cudaMemcpyDeviceToDevice);
    } else {
        result.first = nullptr;
        result.second = nullptr;
        result.scores = nullptr;
    }

    return result;
}

}  
