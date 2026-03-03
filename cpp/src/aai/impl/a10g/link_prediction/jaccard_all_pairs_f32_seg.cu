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

namespace aai {

namespace {

struct Cache : Cacheable {
    void* cub_scratch = nullptr;
    size_t cub_scratch_size = 0;

    void ensure_cub_scratch(size_t needed) {
        if (cub_scratch_size >= needed) return;
        if (cub_scratch) cudaFree(cub_scratch);
        cub_scratch_size = std::max(needed, (size_t)(16 << 20));
        cudaMalloc(&cub_scratch, cub_scratch_size);
    }

    ~Cache() override {
        if (cub_scratch) cudaFree(cub_scratch);
    }
};

static int compute_bits(int val) {
    if (val <= 1) return 1;
    int bits = 0;
    int tmp = val - 1;
    while (tmp > 0) { bits++; tmp >>= 1; }
    return bits;
}


__global__ void count_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    int64_t* __restrict__ counts
) {
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int u = __ldg(&seeds[sid]);
    int us = __ldg(&offsets[u]);
    int ue = __ldg(&offsets[u + 1]);

    int64_t cnt = 0;
    for (int i = us + threadIdx.x; i < ue; i += blockDim.x) {
        int n = __ldg(&indices[i]);
        cnt += (int64_t)(__ldg(&offsets[n + 1]) - __ldg(&offsets[n]));
    }

    typedef cub::BlockReduce<int64_t, 256> BR;
    __shared__ typename BR::TempStorage tmp;
    int64_t total = BR(tmp).Sum(cnt);
    if (threadIdx.x == 0) counts[sid] = total;
}


__global__ void expand_u32_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    const int64_t* __restrict__ pair_offsets,
    uint32_t* __restrict__ keys,
    int key_bits
) {
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int u = __ldg(&seeds[sid]);
    int us = __ldg(&offsets[u]);
    int ue = __ldg(&offsets[u + 1]);
    int64_t base = pair_offsets[sid];

    uint32_t seed_part = (uint32_t)sid << key_bits;
    int64_t pos = base;

    for (int i = us; i < ue; i++) {
        int n = __ldg(&indices[i]);
        int ns = __ldg(&offsets[n]);
        int ne = __ldg(&offsets[n + 1]);
        int nd = ne - ns;

        for (int j = threadIdx.x; j < nd; j += blockDim.x) {
            int v = __ldg(&indices[ns + j]);
            keys[pos + j] = seed_part | (uint32_t)v;
        }
        pos += nd;
    }
}


__global__ void expand_u64_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    const int64_t* __restrict__ pair_offsets,
    uint64_t* __restrict__ keys,
    int key_bits
) {
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int u = __ldg(&seeds[sid]);
    int us = __ldg(&offsets[u]);
    int ue = __ldg(&offsets[u + 1]);
    int64_t base = pair_offsets[sid];

    uint64_t seed_part = (uint64_t)sid << key_bits;
    int64_t pos = base;

    for (int i = us; i < ue; i++) {
        int n = __ldg(&indices[i]);
        int ns = __ldg(&offsets[n]);
        int ne = __ldg(&offsets[n + 1]);
        int nd = ne - ns;

        for (int j = threadIdx.x; j < nd; j += blockDim.x) {
            int v = __ldg(&indices[ns + j]);
            keys[pos + j] = seed_part | (uint64_t)(uint32_t)v;
        }
        pos += nd;
    }
}


__global__ void fused_dedup_u32_kernel(
    const uint32_t* __restrict__ skeys,
    int n,
    const int32_t* __restrict__ seeds,
    int key_bits,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    int64_t* __restrict__ counter
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    bool valid = false;
    uint32_t key = 0;
    int sid = 0, vid = 0;

    if (i < n) {
        key = skeys[i];
        uint32_t mask = ((uint32_t)1 << key_bits) - 1;
        sid = (int)(key >> key_bits);
        vid = (int)(key & mask);
        int seed_v = __ldg(&seeds[sid]);

        bool is_self = (vid == seed_v);
        bool is_unique = (i == 0) || (skeys[i - 1] != key);
        valid = !is_self && is_unique;
    }

    unsigned int ballot = __ballot_sync(0xffffffff, valid);
    int lane_count = __popc(ballot);
    int lane_offset = __popc(ballot & ((1u << lane) - 1));

    int64_t warp_base = 0;
    if (lane == 0 && lane_count > 0)
        warp_base = (int64_t)atomicAdd((unsigned long long*)counter, (unsigned long long)lane_count);
    int wb_lo = __shfl_sync(0xffffffff, (int)(warp_base & 0xFFFFFFFF), 0);
    int wb_hi = __shfl_sync(0xffffffff, (int)(warp_base >> 32), 0);
    warp_base = ((int64_t)(unsigned int)wb_hi << 32) | (unsigned int)wb_lo;

    if (valid) {
        int64_t pos = warp_base + lane_offset;
        out_first[pos] = __ldg(&seeds[sid]);
        out_second[pos] = vid;
    }
}


__global__ void fused_dedup_u64_kernel(
    const uint64_t* __restrict__ skeys,
    int64_t n,
    const int32_t* __restrict__ seeds,
    int key_bits,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    int64_t* __restrict__ counter
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & 31;

    bool valid = false;
    int sid = 0, vid = 0;

    if (i < n) {
        uint64_t key = skeys[i];
        uint64_t mask = ((uint64_t)1 << key_bits) - 1;
        sid = (int)(key >> key_bits);
        vid = (int)(key & mask);
        int seed_v = __ldg(&seeds[sid]);

        bool is_self = (vid == seed_v);
        bool is_unique = (i == 0) || (skeys[i - 1] != key);
        valid = !is_self && is_unique;
    }

    unsigned int ballot = __ballot_sync(0xffffffff, valid);
    int lane_count = __popc(ballot);
    int lane_offset = __popc(ballot & ((1u << lane) - 1));

    int64_t warp_base = 0;
    if (lane == 0 && lane_count > 0)
        warp_base = (int64_t)atomicAdd((unsigned long long*)counter, (unsigned long long)lane_count);
    int wb_lo = __shfl_sync(0xffffffff, (int)(warp_base & 0xFFFFFFFF), 0);
    int wb_hi = __shfl_sync(0xffffffff, (int)(warp_base >> 32), 0);
    warp_base = ((int64_t)(unsigned int)wb_hi << 32) | (unsigned int)wb_lo;

    if (valid) {
        int64_t pos = warp_base + lane_offset;
        out_first[pos] = __ldg(&seeds[sid]);
        out_second[pos] = vid;
    }
}


__global__ void jaccard_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const int32_t* __restrict__ pfirst,
    const int32_t* __restrict__ psecond,
    int64_t npairs,
    float* __restrict__ scores
) {
    int pid = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    if (pid >= npairs) return;

    int u = __ldg(&pfirst[pid]);
    int v = __ldg(&psecond[pid]);

    int us = __ldg(&offsets[u]);
    int ue = __ldg(&offsets[u + 1]);
    int vs = __ldg(&offsets[v]);
    int ve = __ldg(&offsets[v + 1]);
    int du = ue - us;
    int dv = ve - vs;

    float wu_sum = 0.0f;
    for (int i = us + lane; i < ue; i += 32)
        wu_sum += __ldg(&weights[i]);
    float wv_sum = 0.0f;
    for (int i = vs + lane; i < ve; i += 32)
        wv_sum += __ldg(&weights[i]);

    int ps, pe, ss, se;
    if (du <= dv) {
        ps = us; pe = ue; ss = vs; se = ve;
    } else {
        ps = vs; pe = ve; ss = us; se = ue;
    }

    float sum_min = 0.0f;
    for (int i = ps + lane; i < pe; i += 32) {
        int nbr = __ldg(&indices[i]);
        float w1 = __ldg(&weights[i]);

        int lo = ss, hi = se;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (__ldg(&indices[mid]) < nbr) lo = mid + 1;
            else hi = mid;
        }

        if (lo < se && __ldg(&indices[lo]) == nbr) {
            sum_min += fminf(w1, __ldg(&weights[lo]));
        }
    }

    #pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
        sum_min += __shfl_xor_sync(0xffffffff, sum_min, m);
        wu_sum += __shfl_xor_sync(0xffffffff, wu_sum, m);
        wv_sum += __shfl_xor_sync(0xffffffff, wv_sum, m);
    }

    if (lane == 0) {
        float denom = wu_sum + wv_sum - sum_min;
        scores[pid] = (denom > 0.0f) ? (sum_min / denom) : 0.0f;
    }
}


__global__ void iota_kernel(int32_t* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = i;
}

__global__ void gather_kernel(
    const int32_t* __restrict__ idx,
    const int32_t* __restrict__ sf, const int32_t* __restrict__ ss,
    const float* __restrict__ sc,
    int32_t* __restrict__ df, int32_t* __restrict__ ds,
    float* __restrict__ dsc, int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int j = idx[i];
        df[i] = sf[j];
        ds[i] = ss[j];
        dsc[i] = sc[j];
    }
}

}  

similarity_result_float_t jaccard_all_pairs_similarity_seg(
    const graph32_t& graph,
    const float* edge_weights,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk) {

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    const int32_t* d_off = graph.offsets;
    const int32_t* d_ind = graph.indices;
    const float* d_wt = edge_weights;
    cudaStream_t stream = 0;

    
    int32_t ns;
    int32_t* seeds_buf = nullptr;
    const int32_t* d_seeds;
    if (vertices != nullptr) {
        ns = (int32_t)num_vertices;
        d_seeds = vertices;
    } else {
        ns = nv;
        cudaMalloc(&seeds_buf, (size_t)nv * sizeof(int32_t));
        iota_kernel<<<(nv + 255) / 256, 256, 0, stream>>>(seeds_buf, nv);
        d_seeds = seeds_buf;
    }

    if (ns == 0 || nv == 0) {
        if (seeds_buf) cudaFree(seeds_buf);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int64_t* d_counts = nullptr;
    cudaMalloc(&d_counts, (size_t)ns * sizeof(int64_t));
    count_pairs_kernel<<<ns, 256, 0, stream>>>(d_off, d_ind, d_seeds, ns, d_counts);

    
    int64_t* d_poff = nullptr;
    cudaMalloc(&d_poff, (size_t)ns * sizeof(int64_t));
    {
        size_t tb = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tb, d_counts, d_poff, ns, stream);
        cache.ensure_cub_scratch(tb);
        cub::DeviceScan::ExclusiveSum(cache.cub_scratch, tb, d_counts, d_poff, ns, stream);
    }

    
    int64_t h_total = 0;
    {
        int64_t h_last_off = 0, h_last_cnt = 0;
        cudaMemcpyAsync(&h_last_off, d_poff + ns - 1,
                        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&h_last_cnt, d_counts + ns - 1,
                        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        h_total = h_last_off + h_last_cnt;
    }

    if (h_total == 0) {
        cudaFree(d_counts);
        cudaFree(d_poff);
        if (seeds_buf) cudaFree(seeds_buf);
        return {nullptr, nullptr, nullptr, 0};
    }

    int64_t n_raw = h_total;
    int key_bits = compute_bits(nv);
    int seed_bits = (ns <= 1) ? 0 : compute_bits(ns);
    int total_bits = key_bits + seed_bits;
    bool use_u32 = (total_bits <= 32);

    
    int32_t* pair_first = nullptr;
    int32_t* pair_second = nullptr;
    cudaMalloc(&pair_first, (size_t)n_raw * sizeof(int32_t));
    cudaMalloc(&pair_second, (size_t)n_raw * sizeof(int32_t));

    
    int64_t* d_counter = nullptr;
    cudaMalloc(&d_counter, sizeof(int64_t));
    cudaMemsetAsync(d_counter, 0, sizeof(int64_t), stream);

    int64_t num_unique = 0;

    if (use_u32) {
        uint32_t* keys = nullptr;
        cudaMalloc(&keys, (size_t)n_raw * sizeof(uint32_t));
        expand_u32_kernel<<<ns, 256, 0, stream>>>(d_off, d_ind, d_seeds, ns, d_poff, keys, key_bits);

        uint32_t* sorted_keys = nullptr;
        cudaMalloc(&sorted_keys, (size_t)n_raw * sizeof(uint32_t));
        {
            size_t tb = 0;
            cub::DeviceRadixSort::SortKeys(nullptr, tb, keys, sorted_keys, n_raw, 0, total_bits, stream);
            cache.ensure_cub_scratch(tb);
            cub::DeviceRadixSort::SortKeys(cache.cub_scratch, tb, keys, sorted_keys, n_raw, 0, total_bits, stream);
        }
        cudaFree(keys);

        {
            int t = 256, b = (int)((n_raw + t - 1) / t);
            fused_dedup_u32_kernel<<<b, t, 0, stream>>>(sorted_keys, (int)n_raw, d_seeds, key_bits,
                                                         pair_first, pair_second, d_counter);
        }
        cudaFree(sorted_keys);
    } else {
        uint64_t* keys = nullptr;
        cudaMalloc(&keys, (size_t)n_raw * sizeof(uint64_t));
        expand_u64_kernel<<<ns, 256, 0, stream>>>(d_off, d_ind, d_seeds, ns, d_poff, keys, key_bits);

        uint64_t* sorted_keys = nullptr;
        cudaMalloc(&sorted_keys, (size_t)n_raw * sizeof(uint64_t));
        {
            size_t tb = 0;
            cub::DeviceRadixSort::SortKeys(nullptr, tb, keys, sorted_keys, n_raw, 0, total_bits, stream);
            cache.ensure_cub_scratch(tb);
            cub::DeviceRadixSort::SortKeys(cache.cub_scratch, tb, keys, sorted_keys, n_raw, 0, total_bits, stream);
        }
        cudaFree(keys);

        {
            int t = 256;
            int b = (int)((n_raw + t - 1) / t);
            fused_dedup_u64_kernel<<<b, t, 0, stream>>>(sorted_keys, n_raw, d_seeds, key_bits,
                                                         pair_first, pair_second, d_counter);
        }
        cudaFree(sorted_keys);
    }

    cudaFree(d_counts);
    cudaFree(d_poff);

    
    cudaMemcpyAsync(&num_unique, d_counter, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_counter);

    if (num_unique == 0) {
        cudaFree(pair_first);
        cudaFree(pair_second);
        if (seeds_buf) cudaFree(seeds_buf);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    float* scores = nullptr;
    cudaMalloc(&scores, (size_t)num_unique * sizeof(float));
    {
        int t = 256;
        int wpb = t / 32;
        int b = (int)((num_unique + wpb - 1) / wpb);
        jaccard_kernel<<<b, t, 0, stream>>>(d_off, d_ind, d_wt, pair_first, pair_second, num_unique, scores);
    }

    
    if (topk.has_value() && num_unique > (int64_t)topk.value()) {
        int topk_val = (int)topk.value();

        float* sorted_scores = nullptr;
        int32_t* sort_idx = nullptr;
        int32_t* sorted_idx = nullptr;
        cudaMalloc(&sorted_scores, (size_t)num_unique * sizeof(float));
        cudaMalloc(&sort_idx, (size_t)num_unique * sizeof(int32_t));
        cudaMalloc(&sorted_idx, (size_t)num_unique * sizeof(int32_t));

        iota_kernel<<<((int)num_unique + 255) / 256, 256, 0, stream>>>(sort_idx, (int)num_unique);

        {
            size_t tb = 0;
            cub::DeviceRadixSort::SortPairsDescending(nullptr, tb,
                scores, sorted_scores, sort_idx, sorted_idx, (int)num_unique, 0, 32, stream);
            cache.ensure_cub_scratch(tb);
            cub::DeviceRadixSort::SortPairsDescending(cache.cub_scratch, tb,
                scores, sorted_scores, sort_idx, sorted_idx, (int)num_unique, 0, 32, stream);
        }

        cudaFree(sorted_scores);
        cudaFree(sort_idx);

        int32_t* out_first = nullptr;
        int32_t* out_second = nullptr;
        float* out_scores = nullptr;
        cudaMalloc(&out_first, (size_t)topk_val * sizeof(int32_t));
        cudaMalloc(&out_second, (size_t)topk_val * sizeof(int32_t));
        cudaMalloc(&out_scores, (size_t)topk_val * sizeof(float));

        {
            int t = 256, b = (topk_val + t - 1) / t;
            gather_kernel<<<b, t, 0, stream>>>(sorted_idx, pair_first, pair_second, scores,
                                                out_first, out_second, out_scores, topk_val);
        }

        cudaFree(sorted_idx);
        cudaFree(pair_first);
        cudaFree(pair_second);
        cudaFree(scores);
        if (seeds_buf) cudaFree(seeds_buf);

        return {out_first, out_second, out_scores, (std::size_t)topk_val};
    }

    
    if (num_unique < n_raw) {
        int32_t* f = nullptr;
        int32_t* s = nullptr;
        cudaMalloc(&f, (size_t)num_unique * sizeof(int32_t));
        cudaMalloc(&s, (size_t)num_unique * sizeof(int32_t));
        cudaMemcpyAsync(f, pair_first, num_unique * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(s, pair_second, num_unique * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaFree(pair_first);
        cudaFree(pair_second);
        if (seeds_buf) cudaFree(seeds_buf);
        return {f, s, scores, (std::size_t)num_unique};
    }

    if (seeds_buf) cudaFree(seeds_buf);
    return {pair_first, pair_second, scores, (std::size_t)num_unique};
}

}  
