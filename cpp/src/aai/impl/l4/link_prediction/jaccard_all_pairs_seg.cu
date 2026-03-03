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
#include <algorithm>
#include <optional>

namespace aai {

namespace {




struct Cache : Cacheable {
    
    int32_t* deg = nullptr;
    int64_t deg_cap = 0;

    int32_t* prefix = nullptr;
    int64_t prefix_cap = 0;

    void* ps_temp = nullptr;
    size_t ps_temp_cap = 0;

    
    int32_t* pf = nullptr;
    int64_t pf_cap = 0;

    int32_t* ps = nullptr;
    int64_t ps_cap = 0;

    int32_t* pc = nullptr;
    bool pc_allocated = false;

    uint32_t* bitmaps = nullptr;
    int64_t bitmaps_cap = 0;

    
    int32_t* seeds_buf = nullptr;
    int64_t seeds_buf_cap = 0;

    
    float* scores = nullptr;
    int64_t scores_cap = 0;

    
    int32_t* idx_in = nullptr;
    int64_t idx_in_cap = 0;

    int32_t* idx_out = nullptr;
    int64_t idx_out_cap = 0;

    float* sc_sorted = nullptr;
    int64_t sc_sorted_cap = 0;

    void* sort_temp = nullptr;
    size_t sort_temp_cap = 0;

    ~Cache() override {
        if (deg) cudaFree(deg);
        if (prefix) cudaFree(prefix);
        if (ps_temp) cudaFree(ps_temp);
        if (pf) cudaFree(pf);
        if (ps) cudaFree(ps);
        if (pc) cudaFree(pc);
        if (bitmaps) cudaFree(bitmaps);
        if (seeds_buf) cudaFree(seeds_buf);
        if (scores) cudaFree(scores);
        if (idx_in) cudaFree(idx_in);
        if (idx_out) cudaFree(idx_out);
        if (sc_sorted) cudaFree(sc_sorted);
        if (sort_temp) cudaFree(sort_temp);
    }
};




__global__ void compute_seed_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int32_t* __restrict__ degrees
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_seeds) {
        int32_t u = seeds[i];
        degrees[i] = offsets[u + 1] - offsets[u];
    }
}





__global__ void discover_pairs_flat_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int32_t* __restrict__ seed_deg_prefix,
    uint32_t* __restrict__ bitmaps,
    int32_t bitmap_words,
    int32_t* __restrict__ pair_first,
    int32_t* __restrict__ pair_second,
    int32_t* __restrict__ total_pair_count,
    int32_t max_pairs,
    int32_t total_blocks
) {
    int block_id = blockIdx.x;
    if (block_id >= total_blocks) return;

    int lo = 0, hi = num_seeds - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (seed_deg_prefix[mid] <= block_id) lo = mid;
        else hi = mid - 1;
    }
    int seed_idx = lo;

    int32_t u = seeds[seed_idx];
    int32_t u_start = offsets[u];
    int w_local = block_id - seed_deg_prefix[seed_idx];
    int32_t w = indices[u_start + w_local];

    int32_t w_start = offsets[w];
    int32_t w_end = offsets[w + 1];

    uint32_t* seen = bitmaps + (int64_t)seed_idx * bitmap_words;

    for (int32_t j = w_start + threadIdx.x; j < w_end; j += blockDim.x) {
        int32_t v = indices[j];
        uint32_t bit = 1u << (v & 31);
        uint32_t word = seen[v >> 5];
        if (!(word & bit)) {
            uint32_t old = atomicOr(&seen[v >> 5], bit);
            if (!(old & bit)) {
                int32_t idx = atomicAdd(total_pair_count, 1);
                if (idx < max_pairs) {
                    pair_first[idx] = u;
                    pair_second[idx] = v;
                }
            }
        }
    }
}

__global__ void mark_seeds_kernel(
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    uint32_t* __restrict__ bitmaps,
    int32_t bitmap_words
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_seeds) {
        int32_t u = seeds[i];
        uint32_t* seen = bitmaps + (int64_t)i * bitmap_words;
        atomicOr(&seen[u >> 5], 1u << (u & 31));
    }
}




__global__ void compute_jaccard_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pair_first,
    const int32_t* __restrict__ pair_second,
    float* __restrict__ pair_scores,
    int32_t num_pairs
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    int32_t u = pair_first[warp_id];
    int32_t v = pair_second[warp_id];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];
    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;

    const int32_t* shorter_base;
    const int32_t* longer_base;
    int32_t shorter_len, longer_len;

    if (deg_u <= deg_v) {
        shorter_base = indices + u_start;
        shorter_len = deg_u;
        longer_base = indices + v_start;
        longer_len = deg_v;
    } else {
        shorter_base = indices + v_start;
        shorter_len = deg_v;
        longer_base = indices + u_start;
        longer_len = deg_u;
    }

    int local_count = 0;
    for (int i = lane; i < shorter_len; i += 32) {
        int32_t val = shorter_base[i];
        int lo2 = 0, hi2 = longer_len;
        while (lo2 < hi2) {
            int mid = (lo2 + hi2) >> 1;
            if (longer_base[mid] < val) lo2 = mid + 1;
            else hi2 = mid;
        }
        if (lo2 < longer_len && longer_base[lo2] == val) {
            local_count++;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }

    if (lane == 0) {
        int intersection = local_count;
        int union_size = deg_u + deg_v - intersection;
        pair_scores[warp_id] = (union_size > 0) ?
            __fdividef((float)intersection, (float)union_size) : 0.0f;
    }
}

__global__ void compute_jaccard_multigraph_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pair_first,
    const int32_t* __restrict__ pair_second,
    float* __restrict__ pair_scores,
    int32_t num_pairs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;

    int32_t u = pair_first[tid];
    int32_t v = pair_second[tid];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];
    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;

    int intersection = 0;
    int i = 0, j = 0;
    while (i < deg_u && j < deg_v) {
        int32_t a = indices[u_start + i];
        int32_t b = indices[v_start + j];
        if (a == b) { intersection++; i++; j++; }
        else if (a < b) { i++; }
        else { j++; }
    }

    int union_size = deg_u + deg_v - intersection;
    pair_scores[tid] = (union_size > 0) ?
        __fdividef((float)intersection, (float)union_size) : 0.0f;
}




__global__ void iota_kernel(int32_t* out, int32_t n) {
    int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = idx;
}

__global__ void gather_i32_kernel(const int32_t* __restrict__ src,
                                   const int32_t* __restrict__ idx,
                                   int32_t* __restrict__ dst, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
}

__global__ void gather_f32_kernel(const float* __restrict__ src,
                                   const int32_t* __restrict__ idx,
                                   float* __restrict__ dst, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
}

}  




similarity_result_float_t jaccard_all_pairs_similarity_seg(
    const graph32_t& graph,
    const int32_t* vertices,
    std::size_t num_vertices_param,
    std::optional<std::size_t> topk)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t nv = graph.number_of_vertices;
    bool is_multigraph = graph.is_multigraph;
    cudaStream_t stream = 0;

    
    int32_t num_seeds;
    const int32_t* d_seeds;

    if (vertices != nullptr && num_vertices_param > 0) {
        num_seeds = (int32_t)num_vertices_param;
        d_seeds = vertices;
    } else {
        num_seeds = nv;
        if (cache.seeds_buf_cap < nv) {
            if (cache.seeds_buf) cudaFree(cache.seeds_buf);
            cudaMalloc(&cache.seeds_buf, (int64_t)nv * sizeof(int32_t));
            cache.seeds_buf_cap = nv;
        }
        if (nv > 0) {
            iota_kernel<<<(nv + 255) / 256, 256, 0, stream>>>(cache.seeds_buf, nv);
        }
        d_seeds = cache.seeds_buf;
    }

    if (num_seeds == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t deg_size = (int64_t)num_seeds + 1;

    if (cache.deg_cap < deg_size) {
        if (cache.deg) cudaFree(cache.deg);
        cudaMalloc(&cache.deg, deg_size * sizeof(int32_t));
        cache.deg_cap = deg_size;
    }
    if (cache.prefix_cap < deg_size) {
        if (cache.prefix) cudaFree(cache.prefix);
        cudaMalloc(&cache.prefix, deg_size * sizeof(int32_t));
        cache.prefix_cap = deg_size;
    }

    compute_seed_degrees_kernel<<<(num_seeds + 255) / 256, 256, 0, stream>>>(
        d_offsets, d_seeds, num_seeds, cache.deg);
    cudaMemsetAsync(cache.deg + num_seeds, 0, sizeof(int32_t), stream);

    size_t ps_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, ps_temp_bytes,
        (int32_t*)nullptr, (int32_t*)nullptr, (int)deg_size);
    if (cache.ps_temp_cap < ps_temp_bytes) {
        if (cache.ps_temp) cudaFree(cache.ps_temp);
        cudaMalloc(&cache.ps_temp, ps_temp_bytes);
        cache.ps_temp_cap = ps_temp_bytes;
    }

    cub::DeviceScan::ExclusiveSum(cache.ps_temp, ps_temp_bytes,
        cache.deg, cache.prefix, (int)deg_size, stream);

    int32_t total_neighbor_pairs;
    cudaMemcpyAsync(&total_neighbor_pairs, cache.prefix + num_seeds, sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (total_neighbor_pairs == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t bitmap_words = (nv + 31) / 32;
    int64_t bitmap_bytes_per_seed = (int64_t)bitmap_words * sizeof(uint32_t);

    const int64_t MAX_BITMAP_BYTES = 1500000000LL;
    int32_t seeds_per_batch = num_seeds;
    if (bitmap_bytes_per_seed > 0) {
        int64_t max_s = MAX_BITMAP_BYTES / bitmap_bytes_per_seed;
        if (max_s < 1) max_s = 1;
        seeds_per_batch = (int32_t)std::min((int64_t)num_seeds, max_s);
    }

    int64_t max_pairs = 20000000LL;

    if (cache.pf_cap < max_pairs) {
        if (cache.pf) cudaFree(cache.pf);
        cudaMalloc(&cache.pf, max_pairs * sizeof(int32_t));
        cache.pf_cap = max_pairs;
    }
    if (cache.ps_cap < max_pairs) {
        if (cache.ps) cudaFree(cache.ps);
        cudaMalloc(&cache.ps, max_pairs * sizeof(int32_t));
        cache.ps_cap = max_pairs;
    }
    if (!cache.pc_allocated) {
        cudaMalloc(&cache.pc, sizeof(int32_t));
        cache.pc_allocated = true;
    }
    cudaMemsetAsync(cache.pc, 0, sizeof(int32_t), stream);

    int64_t batch_bitmap_words = (int64_t)seeds_per_batch * bitmap_words;
    if (cache.bitmaps_cap < batch_bitmap_words) {
        if (cache.bitmaps) cudaFree(cache.bitmaps);
        cudaMalloc(&cache.bitmaps, batch_bitmap_words * sizeof(uint32_t));
        cache.bitmaps_cap = batch_bitmap_words;
    }

    int32_t discover_block_size = 128;

    for (int32_t batch_start = 0; batch_start < num_seeds; batch_start += seeds_per_batch) {
        int32_t batch_count = std::min(seeds_per_batch, num_seeds - batch_start);
        int64_t clear_words = (int64_t)batch_count * bitmap_words;
        cudaMemsetAsync(cache.bitmaps, 0, clear_words * sizeof(uint32_t), stream);

        if (batch_count > 0) {
            mark_seeds_kernel<<<(batch_count + 255) / 256, 256, 0, stream>>>(
                d_seeds + batch_start, batch_count, cache.bitmaps, bitmap_words);
        }

        int32_t batch_total_blocks;
        if (batch_start == 0 && batch_count == num_seeds) {
            batch_total_blocks = total_neighbor_pairs;
        } else {
            int32_t start_val, end_val;
            cudaMemcpyAsync(&start_val, cache.prefix + batch_start, sizeof(int32_t),
                           cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(&end_val, cache.prefix + batch_start + batch_count, sizeof(int32_t),
                           cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            batch_total_blocks = end_val - start_val;
        }

        if (batch_total_blocks > 0) {
            if (batch_start == 0) {
                discover_pairs_flat_kernel<<<batch_total_blocks, discover_block_size, 0, stream>>>(
                    d_offsets, d_indices,
                    d_seeds, batch_count, cache.prefix,
                    cache.bitmaps, bitmap_words,
                    cache.pf, cache.ps, cache.pc, (int32_t)max_pairs,
                    batch_total_blocks);
            }
        }
    }

    int32_t actual_pairs;
    cudaMemcpyAsync(&actual_pairs, cache.pc, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    if (actual_pairs > (int32_t)max_pairs) actual_pairs = (int32_t)max_pairs;

    if (actual_pairs == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    if (cache.scores_cap < actual_pairs) {
        if (cache.scores) cudaFree(cache.scores);
        cudaMalloc(&cache.scores, (int64_t)actual_pairs * sizeof(float));
        cache.scores_cap = actual_pairs;
    }

    if (is_multigraph) {
        int block = 256;
        compute_jaccard_multigraph_kernel<<<(actual_pairs + block - 1) / block, block, 0, stream>>>(
            d_offsets, d_indices, cache.pf, cache.ps, cache.scores, actual_pairs);
    } else {
        int wpb = 8;
        int tpb = wpb * 32;
        compute_jaccard_kernel<<<(actual_pairs + wpb - 1) / wpb, tpb, 0, stream>>>(
            d_offsets, d_indices, cache.pf, cache.ps, cache.scores, actual_pairs);
    }

    
    int32_t output_count = actual_pairs;

    if (topk.has_value() && (int64_t)actual_pairs > (int64_t)topk.value()) {
        output_count = (int32_t)topk.value();

        if (cache.idx_in_cap < actual_pairs) {
            if (cache.idx_in) cudaFree(cache.idx_in);
            cudaMalloc(&cache.idx_in, (int64_t)actual_pairs * sizeof(int32_t));
            cache.idx_in_cap = actual_pairs;
        }
        if (cache.idx_out_cap < actual_pairs) {
            if (cache.idx_out) cudaFree(cache.idx_out);
            cudaMalloc(&cache.idx_out, (int64_t)actual_pairs * sizeof(int32_t));
            cache.idx_out_cap = actual_pairs;
        }
        if (cache.sc_sorted_cap < actual_pairs) {
            if (cache.sc_sorted) cudaFree(cache.sc_sorted);
            cudaMalloc(&cache.sc_sorted, (int64_t)actual_pairs * sizeof(float));
            cache.sc_sorted_cap = actual_pairs;
        }

        iota_kernel<<<(actual_pairs + 255) / 256, 256, 0, stream>>>(cache.idx_in, actual_pairs);

        size_t sort_tb = 0;
        cub::DeviceRadixSort::SortPairsDescending(nullptr, sort_tb,
            (float*)nullptr, (float*)nullptr,
            (int32_t*)nullptr, (int32_t*)nullptr, actual_pairs);
        if (cache.sort_temp_cap < sort_tb) {
            if (cache.sort_temp) cudaFree(cache.sort_temp);
            cudaMalloc(&cache.sort_temp, sort_tb);
            cache.sort_temp_cap = sort_tb;
        }

        cub::DeviceRadixSort::SortPairsDescending(cache.sort_temp, sort_tb,
            cache.scores, cache.sc_sorted,
            cache.idx_in, cache.idx_out,
            actual_pairs, 0, 32, stream);

        
        int32_t* out_first = nullptr;
        int32_t* out_second = nullptr;
        float* out_scores = nullptr;
        cudaMalloc(&out_first, (int64_t)output_count * sizeof(int32_t));
        cudaMalloc(&out_second, (int64_t)output_count * sizeof(int32_t));
        cudaMalloc(&out_scores, (int64_t)output_count * sizeof(float));

        gather_i32_kernel<<<(output_count + 255) / 256, 256, 0, stream>>>(
            cache.pf, cache.idx_out, out_first, output_count);
        gather_i32_kernel<<<(output_count + 255) / 256, 256, 0, stream>>>(
            cache.ps, cache.idx_out, out_second, output_count);
        gather_f32_kernel<<<(output_count + 255) / 256, 256, 0, stream>>>(
            cache.scores, cache.idx_out, out_scores, output_count);

        return {out_first, out_second, out_scores, (std::size_t)output_count};
    }

    
    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;
    cudaMalloc(&out_first, (int64_t)output_count * sizeof(int32_t));
    cudaMalloc(&out_second, (int64_t)output_count * sizeof(int32_t));
    cudaMalloc(&out_scores, (int64_t)output_count * sizeof(float));

    cudaMemcpyAsync(out_first, cache.pf, output_count * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(out_second, cache.ps, output_count * sizeof(int32_t),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(out_scores, cache.scores, output_count * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    return {out_first, out_second, out_scores, (std::size_t)output_count};
}

}  
