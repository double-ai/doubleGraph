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

namespace aai {

namespace {




struct Cache : Cacheable {
    cudaStream_t stream = nullptr;
    void* scratch = nullptr;
    size_t scratch_size = 0;

    Cache() {
        cudaStreamCreate(&stream);
        cudaMemPool_t pool;
        cudaDeviceGetDefaultMemPool(&pool, 0);
        uint64_t threshold = UINT64_MAX;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    }

    ~Cache() override {
        if (scratch) { cudaFreeAsync(scratch, stream); }
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
    }

    void ensure_scratch(size_t needed) {
        if (needed > scratch_size) {
            if (scratch) cudaFreeAsync(scratch, stream);
            scratch_size = needed * 2;
            cudaMallocAsync(&scratch, scratch_size, stream);
        }
    }
};

template<typename T>
static T* async_alloc(int64_t count, cudaStream_t stream) {
    T* ptr;
    cudaMallocAsync(&ptr, count * sizeof(T), stream);
    return ptr;
}

static void async_free(void* ptr, cudaStream_t stream) {
    cudaFreeAsync(ptr, stream);
}




__global__ void iota_kernel(int32_t* out, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = idx;
}




__global__ void count_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ pair_counts
) {
    int sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_seeds) return;

    int32_t u = seeds[sid];
    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);

    int64_t count = 0;
    for (int32_t i = u_start; i < u_end; i++) {
        int32_t k = __ldg(&indices[i]);
        count += (int64_t)(__ldg(&offsets[k + 1]) - __ldg(&offsets[k]));
    }
    pair_counts[sid] = count;
}




__global__ void expand_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ pair_offsets,
    uint64_t* __restrict__ out_keys,
    uint64_t self_loop_marker
) {
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int32_t u = seeds[sid];
    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);

    int64_t write_pos = pair_offsets[sid];

    for (int32_t i = u_start; i < u_end; i++) {
        int32_t k = __ldg(&indices[i]);
        int32_t k_start = __ldg(&offsets[k]);
        int32_t k_end = __ldg(&offsets[k + 1]);
        int32_t k_deg = k_end - k_start;

        for (int32_t j = threadIdx.x; j < k_deg; j += blockDim.x) {
            int32_t v = __ldg(&indices[k_start + j]);
            uint64_t key = (v == u) ? self_loop_marker :
                (((uint64_t)(uint32_t)u << 32) | (uint32_t)v);
            out_keys[write_pos + j] = key;
        }
        write_pos += k_deg;
    }
}




__global__ void cosine_similarity_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint64_t* __restrict__ pair_keys,  
    float* __restrict__ scores,
    int64_t num_pairs
) {
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    uint64_t key = pair_keys[warp_id];
    int32_t u = (int32_t)(key >> 32);
    int32_t v = (int32_t)(key & 0xFFFFFFFF);

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);

    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    
    const int32_t* a_idx;
    const float* a_wt;
    int32_t a_deg;
    const int32_t* b_idx;
    const float* b_wt;
    int32_t b_deg;
    bool u_is_a;

    if (u_deg <= v_deg) {
        a_idx = indices + u_start; a_wt = weights + u_start; a_deg = u_deg;
        b_idx = indices + v_start; b_wt = weights + v_start; b_deg = v_deg;
        u_is_a = true;
    } else {
        a_idx = indices + v_start; a_wt = weights + v_start; a_deg = v_deg;
        b_idx = indices + u_start; b_wt = weights + u_start; b_deg = u_deg;
        u_is_a = false;
    }

    float dot = 0.0f, norm_u = 0.0f, norm_v = 0.0f;

    if (a_deg > 0 && b_deg > 0) {
        for (int i = lane; i < a_deg; i += 32) {
            int32_t a_val = __ldg(&a_idx[i]);
            float a_w = __ldg(&a_wt[i]);

            
            int lo = 0, hi = b_deg;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (__ldg(&b_idx[mid]) < a_val) lo = mid + 1;
                else hi = mid;
            }

            if (lo < b_deg && __ldg(&b_idx[lo]) == a_val) {
                float b_w = __ldg(&b_wt[lo]);
                float w_u = u_is_a ? a_w : b_w;
                float w_v = u_is_a ? b_w : a_w;
                dot += w_u * w_v;
                norm_u += w_u * w_u;
                norm_v += w_v * w_v;
            }
        }
    }

    
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
        norm_u += __shfl_down_sync(0xffffffff, norm_u, offset);
        norm_v += __shfl_down_sync(0xffffffff, norm_v, offset);
    }

    if (lane == 0) {
        float denom = sqrtf(norm_u) * sqrtf(norm_v);
        scores[warp_id] = (denom > 0.0f) ? (dot / denom) : 0.0f;
    }
}


__global__ void cosine_and_pack_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint64_t* __restrict__ pair_keys,
    float* __restrict__ scores,
    uint64_t* __restrict__ score_keys,  
    int64_t num_pairs
) {
    int64_t warp_id = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    uint64_t key = pair_keys[warp_id];
    int32_t u = (int32_t)(key >> 32);
    int32_t v = (int32_t)(key & 0xFFFFFFFF);

    int32_t u_start = __ldg(&offsets[u]);
    int32_t u_end = __ldg(&offsets[u + 1]);
    int32_t v_start = __ldg(&offsets[v]);
    int32_t v_end = __ldg(&offsets[v + 1]);

    int32_t u_deg = u_end - u_start;
    int32_t v_deg = v_end - v_start;

    const int32_t* a_idx;
    const float* a_wt;
    int32_t a_deg;
    const int32_t* b_idx;
    const float* b_wt;
    int32_t b_deg;
    bool u_is_a;

    if (u_deg <= v_deg) {
        a_idx = indices + u_start; a_wt = weights + u_start; a_deg = u_deg;
        b_idx = indices + v_start; b_wt = weights + v_start; b_deg = v_deg;
        u_is_a = true;
    } else {
        a_idx = indices + v_start; a_wt = weights + v_start; a_deg = v_deg;
        b_idx = indices + u_start; b_wt = weights + u_start; b_deg = u_deg;
        u_is_a = false;
    }

    float dot = 0.0f, norm_u = 0.0f, norm_v = 0.0f;

    if (a_deg > 0 && b_deg > 0) {
        for (int i = lane; i < a_deg; i += 32) {
            int32_t a_val = __ldg(&a_idx[i]);
            float a_w = __ldg(&a_wt[i]);

            int lo = 0, hi = b_deg;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (__ldg(&b_idx[mid]) < a_val) lo = mid + 1;
                else hi = mid;
            }

            if (lo < b_deg && __ldg(&b_idx[lo]) == a_val) {
                float b_w = __ldg(&b_wt[lo]);
                float w_u = u_is_a ? a_w : b_w;
                float w_v = u_is_a ? b_w : a_w;
                dot += w_u * w_v;
                norm_u += w_u * w_u;
                norm_v += w_v * w_v;
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        dot += __shfl_down_sync(0xffffffff, dot, offset);
        norm_u += __shfl_down_sync(0xffffffff, norm_u, offset);
        norm_v += __shfl_down_sync(0xffffffff, norm_v, offset);
    }

    if (lane == 0) {
        float denom = sqrtf(norm_u) * sqrtf(norm_v);
        float score = (denom > 0.0f) ? (dot / denom) : 0.0f;
        scores[warp_id] = score;

        
        uint32_t bits;
        memcpy(&bits, &score, sizeof(uint32_t));
        uint32_t sortable = (bits & 0x80000000) ? ~bits : (bits ^ 0x80000000);
        sortable = ~sortable;
        score_keys[warp_id] = ((uint64_t)sortable << 32) | (uint32_t)warp_id;
    }
}




__global__ void gather_topk_kernel(
    const uint64_t* __restrict__ sorted_keys,
    const uint64_t* __restrict__ pair_keys,
    const float* __restrict__ in_scores,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    int64_t count
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint32_t orig_idx = (uint32_t)(sorted_keys[idx] & 0xFFFFFFFF);
    uint64_t key = pair_keys[orig_idx];
    out_first[idx] = (int32_t)(key >> 32);
    out_second[idx] = (int32_t)(key & 0xFFFFFFFF);
    out_scores[idx] = in_scores[orig_idx];
}


__global__ void unpack_keys_kernel(
    const uint64_t* __restrict__ keys,
    int32_t* __restrict__ first,
    int32_t* __restrict__ second,
    int64_t count
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    uint64_t key = keys[idx];
    first[idx] = (int32_t)(key >> 32);
    second[idx] = (int32_t)(key & 0xFFFFFFFF);
}




__global__ void adjust_unique_count_kernel(
    const uint64_t* __restrict__ unique_keys, int64_t* __restrict__ count,
    uint64_t marker
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int64_t c = *count;
        if (c > 0 && unique_keys[c - 1] == marker) (*count)--;
    }
}




__global__ void get_total_i64_kernel(const int64_t* prefix, const int64_t* counts, int64_t n, int64_t* total) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *total = prefix[n - 1] + counts[n - 1];
    }
}




static void launch_iota(int32_t* out, int32_t n, cudaStream_t stream) {
    if (n <= 0) return;
    iota_kernel<<<(n+255)/256, 256, 0, stream>>>(out, n);
}

static void launch_count_pairs(const int32_t* offsets, const int32_t* indices,
    const int32_t* seeds, int32_t num_seeds, int64_t* pair_counts, cudaStream_t stream) {
    if (num_seeds <= 0) return;
    count_pairs_kernel<<<(num_seeds+255)/256, 256, 0, stream>>>(offsets, indices, seeds, num_seeds, pair_counts);
}

static void launch_expand_pairs(const int32_t* offsets, const int32_t* indices,
    const int32_t* seeds, int32_t num_seeds,
    const int64_t* pair_offsets, uint64_t* out_keys, uint64_t marker, cudaStream_t stream) {
    if (num_seeds <= 0) return;
    expand_pairs_kernel<<<num_seeds, 256, 0, stream>>>(offsets, indices, seeds, num_seeds, pair_offsets, out_keys, marker);
}

static void launch_cosine_similarity(const int32_t* offsets, const int32_t* indices,
    const float* weights, const uint64_t* pair_keys,
    float* scores, int64_t num_pairs, cudaStream_t stream) {
    if (num_pairs <= 0) return;
    int warps_per_block = 4;
    int tpb = warps_per_block * 32;
    int grid = (int)((num_pairs + warps_per_block - 1) / warps_per_block);
    cosine_similarity_kernel<<<grid, tpb, 0, stream>>>(offsets, indices, weights, pair_keys, scores, num_pairs);
}

static void launch_cosine_and_pack(const int32_t* offsets, const int32_t* indices,
    const float* weights, const uint64_t* pair_keys,
    float* scores, uint64_t* score_keys, int64_t num_pairs, cudaStream_t stream) {
    if (num_pairs <= 0) return;
    int warps_per_block = 4;
    int tpb = warps_per_block * 32;
    int grid = (int)((num_pairs + warps_per_block - 1) / warps_per_block);
    cosine_and_pack_kernel<<<grid, tpb, 0, stream>>>(offsets, indices, weights, pair_keys, scores, score_keys, num_pairs);
}

static void launch_gather_topk(const uint64_t* sorted_keys, const uint64_t* pair_keys,
    const float* in_scores,
    int32_t* out_first, int32_t* out_second, float* out_scores,
    int64_t count, cudaStream_t stream) {
    if (count <= 0) return;
    gather_topk_kernel<<<(int)((count+255)/256), 256, 0, stream>>>(sorted_keys, pair_keys, in_scores, out_first, out_second, out_scores, count);
}

static void launch_unpack_keys(const uint64_t* keys, int32_t* first, int32_t* second,
    int64_t count, cudaStream_t stream) {
    if (count <= 0) return;
    unpack_keys_kernel<<<(int)((count+255)/256), 256, 0, stream>>>(keys, first, second, count);
}

static void launch_get_total_i64(const int64_t* prefix, const int64_t* counts,
    int64_t n, int64_t* total, cudaStream_t stream) {
    get_total_i64_kernel<<<1, 1, 0, stream>>>(prefix, counts, n, total);
}

static void launch_adjust_unique(const uint64_t* unique_keys, int64_t* count, uint64_t marker, cudaStream_t stream) {
    adjust_unique_count_kernel<<<1, 1, 0, stream>>>(unique_keys, count, marker);
}




static size_t get_sort_temp_size_u64(int64_t num_items, int begin_bit, int end_bit) {
    size_t temp_size = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, temp_size, (uint64_t*)nullptr, (uint64_t*)nullptr, (int)num_items, begin_bit, end_bit);
    return temp_size;
}

static void launch_sort_u64(void* temp, size_t temp_size,
    const uint64_t* keys_in, uint64_t* keys_out,
    int64_t num_items, int begin_bit, int end_bit, cudaStream_t stream) {
    cub::DeviceRadixSort::SortKeys(temp, temp_size, keys_in, keys_out, (int)num_items, begin_bit, end_bit, stream);
}

static size_t get_unique_temp_size(int64_t num_items) {
    size_t temp_size = 0;
    cub::DeviceSelect::Unique(nullptr, temp_size, (uint64_t*)nullptr, (uint64_t*)nullptr, (int64_t*)nullptr, (int)num_items);
    return temp_size;
}

static void launch_unique(void* temp, size_t temp_size,
    const uint64_t* in, uint64_t* out, int64_t* num_selected,
    int64_t num_items, cudaStream_t stream) {
    cub::DeviceSelect::Unique(temp, temp_size, in, out, num_selected, (int)num_items, stream);
}

static size_t get_scan_temp_size_i64(int64_t num_items) {
    size_t temp_size = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_size, (int64_t*)nullptr, (int64_t*)nullptr, (int)num_items);
    return temp_size;
}

static void launch_exclusive_sum_i64(void* temp, size_t temp_size,
    const int64_t* in, int64_t* out, int64_t num_items, cudaStream_t stream) {
    cub::DeviceScan::ExclusiveSum(temp, temp_size, in, out, (int)num_items, stream);
}

}  




similarity_result_float_t cosine_all_pairs_similarity(const graph32_t& graph,
                                                      const float* edge_weights,
                                                      const int32_t* vertices,
                                                      std::size_t num_vertices,
                                                      std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    cudaStream_t stream = cache.stream;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const float* d_weights = edge_weights;
    int32_t num_verts = graph.number_of_vertices;

    
    int vertex_bits = 1;
    { int32_t v = num_verts; while (v > 1) { v >>= 1; vertex_bits++; } }
    int sort_end_bit = 32 + vertex_bits;
    if (sort_end_bit > 64) sort_end_bit = 64;
    uint64_t self_loop_marker = ((uint64_t)(uint32_t)num_verts << 32) | (uint32_t)num_verts;

    
    int32_t num_seeds;
    int32_t* d_seeds;
    int32_t* d_seeds_alloc = nullptr;

    if (vertices != nullptr && num_vertices > 0) {
        d_seeds = const_cast<int32_t*>(vertices);
        num_seeds = (int32_t)num_vertices;
    } else {
        d_seeds_alloc = async_alloc<int32_t>(num_verts, stream);
        d_seeds = d_seeds_alloc;
        launch_iota(d_seeds, num_verts, stream);
        num_seeds = num_verts;
    }

    if (num_seeds == 0) {
        if (d_seeds_alloc) async_free(d_seeds_alloc, stream);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t* d_pair_counts = async_alloc<int64_t>(num_seeds, stream);
    launch_count_pairs(d_offsets, d_indices, d_seeds, num_seeds, d_pair_counts, stream);

    int64_t* d_pair_offsets = async_alloc<int64_t>(num_seeds, stream);
    size_t scan_temp = get_scan_temp_size_i64(num_seeds);
    cache.ensure_scratch(scan_temp);
    launch_exclusive_sum_i64(cache.scratch, scan_temp, d_pair_counts, d_pair_offsets, num_seeds, stream);

    int64_t* d_total = async_alloc<int64_t>(1, stream);
    launch_get_total_i64(d_pair_offsets, d_pair_counts, num_seeds, d_total, stream);

    int64_t total_expanded;
    cudaMemcpyAsync(&total_expanded, d_total, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    async_free(d_pair_counts, stream);

    if (total_expanded == 0) {
        async_free(d_pair_offsets, stream);
        async_free(d_total, stream);
        if (d_seeds_alloc) async_free(d_seeds_alloc, stream);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    uint64_t* d_keys = async_alloc<uint64_t>(total_expanded, stream);
    launch_expand_pairs(d_offsets, d_indices, d_seeds, num_seeds,
                       d_pair_offsets, d_keys, self_loop_marker, stream);
    async_free(d_pair_offsets, stream);
    if (d_seeds_alloc) { async_free(d_seeds_alloc, stream); d_seeds_alloc = nullptr; }

    
    uint64_t* d_sorted_keys = async_alloc<uint64_t>(total_expanded, stream);
    size_t sort_temp = get_sort_temp_size_u64(total_expanded, 0, sort_end_bit);
    cache.ensure_scratch(sort_temp);
    launch_sort_u64(cache.scratch, sort_temp, d_keys, d_sorted_keys, total_expanded, 0, sort_end_bit, stream);
    async_free(d_keys, stream);

    
    uint64_t* d_unique_keys = async_alloc<uint64_t>(total_expanded, stream);
    int64_t* d_num_unique = d_total;  

    size_t unique_temp = get_unique_temp_size(total_expanded);
    cache.ensure_scratch(unique_temp);
    launch_unique(cache.scratch, unique_temp, d_sorted_keys, d_unique_keys, d_num_unique, total_expanded, stream);
    async_free(d_sorted_keys, stream);

    
    launch_adjust_unique(d_unique_keys, d_num_unique, self_loop_marker, stream);

    int64_t num_unique;
    cudaMemcpyAsync(&num_unique, d_num_unique, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    async_free(d_num_unique, stream);

    if (num_unique == 0) {
        async_free(d_unique_keys, stream);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    bool need_topk = topk.has_value() && topk.value() < (std::size_t)num_unique;

    if (need_topk) {
        int64_t topk_val = (int64_t)topk.value();

        
        float* d_scores = async_alloc<float>(num_unique, stream);
        uint64_t* d_score_keys = async_alloc<uint64_t>(num_unique, stream);

        launch_cosine_and_pack(d_offsets, d_indices, d_weights, d_unique_keys,
                              d_scores, d_score_keys, num_unique, stream);

        
        uint64_t* d_sorted_score_keys = async_alloc<uint64_t>(num_unique, stream);
        size_t st = get_sort_temp_size_u64(num_unique, 0, 64);
        cache.ensure_scratch(st);
        launch_sort_u64(cache.scratch, st, d_score_keys, d_sorted_score_keys, num_unique, 0, 64, stream);
        async_free(d_score_keys, stream);

        
        int32_t* out_first;
        int32_t* out_second;
        float* out_scores;
        cudaMalloc(&out_first, topk_val * sizeof(int32_t));
        cudaMalloc(&out_second, topk_val * sizeof(int32_t));
        cudaMalloc(&out_scores, topk_val * sizeof(float));

        launch_gather_topk(d_sorted_score_keys, d_unique_keys, d_scores,
                          out_first, out_second, out_scores, topk_val, stream);

        async_free(d_sorted_score_keys, stream);
        async_free(d_scores, stream);
        async_free(d_unique_keys, stream);

        cudaStreamSynchronize(stream);
        return {out_first, out_second, out_scores, (std::size_t)topk_val};
    } else {
        
        
        int32_t* out_first;
        int32_t* out_second;
        float* out_scores;
        cudaMalloc(&out_first, num_unique * sizeof(int32_t));
        cudaMalloc(&out_second, num_unique * sizeof(int32_t));
        cudaMalloc(&out_scores, num_unique * sizeof(float));

        launch_cosine_similarity(d_offsets, d_indices, d_weights, d_unique_keys,
                                out_scores, num_unique, stream);

        
        launch_unpack_keys(d_unique_keys, out_first, out_second, num_unique, stream);

        async_free(d_unique_keys, stream);
        cudaStreamSynchronize(stream);
        return {out_first, out_second, out_scores, (std::size_t)num_unique};
    }
}

}  
