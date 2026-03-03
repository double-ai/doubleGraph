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


struct DevMem {
    void* ptr = nullptr;
    DevMem() = default;
    ~DevMem() { if (ptr) cudaFree(ptr); }
    DevMem(const DevMem&) = delete;
    DevMem& operator=(const DevMem&) = delete;
    void alloc(size_t bytes) { if (bytes > 0) cudaMalloc(&ptr, bytes); }
    template<typename T> T* as() { return static_cast<T*>(ptr); }
    template<typename T> const T* as() const { return static_cast<const T*>(ptr); }
};

struct Cache : Cacheable {
    int64_t* h_pinned = nullptr;

    Cache() {
        cudaMallocHost(&h_pinned, 8 * sizeof(int64_t));
    }

    ~Cache() override {
        if (h_pinned) { cudaFreeHost(h_pinned); h_pinned = nullptr; }
    }
};

static int bits_needed(int64_t max_val) {
    if (max_val <= 0) return 1;
    int bits = 0;
    while (max_val > 0) { bits++; max_val >>= 1; }
    return bits;
}





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





static void launch_count_2hop(const int32_t* o, const int32_t* i, const int32_t* s, int ns, int64_t* c, cudaStream_t st) {
    if (ns == 0) return;
    count_2hop_kernel<<<ns, 256, 0, st>>>(o, i, s, ns, c);
}

static void launch_generate_multiblock_u32(const int32_t* o, const int32_t* i, const int32_t* s, int ns, int V,
    const int64_t* so, uint32_t* k, int32_t* sc, int bps, cudaStream_t st) {
    if (ns == 0) return;
    generate_pairs_multiblock_u32<<<ns * bps, 256, 0, st>>>(o, i, s, ns, V, so, k, sc, bps);
}

static void launch_generate_multiblock_u64(const int32_t* o, const int32_t* i, const int32_t* s, int ns, int64_t V,
    const int64_t* so, uint64_t* k, int32_t* sc, int bps, cudaStream_t st) {
    if (ns == 0) return;
    generate_pairs_multiblock_u64<<<ns * bps, 256, 0, st>>>(o, i, s, ns, V, so, k, sc, bps);
}

static size_t cub_prefix_sum_temp_bytes(int n) {
    size_t t = 0; cub::DeviceScan::ExclusiveSum(nullptr, t, (int64_t*)nullptr, (int64_t*)nullptr, n); return t;
}

static void launch_cub_prefix_sum(void* dt, size_t tb, int64_t* di, int64_t* d_o, int n, cudaStream_t s) {
    cub::DeviceScan::ExclusiveSum(dt, tb, di, d_o, n, s);
}

static size_t cub_sort_u32_temp_bytes(int64_t n, int bb, int eb) {
    size_t t = 0; cub::DeviceRadixSort::SortKeys(nullptr, t, (uint32_t*)nullptr, (uint32_t*)nullptr, (int)n, bb, eb); return t;
}

static void launch_cub_sort_u32(void* dt, size_t tb, uint32_t* di, uint32_t* d_o, int64_t n, int bb, int eb, cudaStream_t s) {
    cub::DeviceRadixSort::SortKeys(dt, tb, di, d_o, (int)n, bb, eb, s);
}

static size_t cub_sort_u64_temp_bytes(int64_t n, int bb, int eb) {
    size_t t = 0; cub::DeviceRadixSort::SortKeys(nullptr, t, (uint64_t*)nullptr, (uint64_t*)nullptr, (int)n, bb, eb); return t;
}

static void launch_cub_sort_u64(void* dt, size_t tb, uint64_t* di, uint64_t* d_o, int64_t n, int bb, int eb, cudaStream_t s) {
    cub::DeviceRadixSort::SortKeys(dt, tb, di, d_o, (int)n, bb, eb, s);
}

static size_t cub_rle_u32_temp_bytes(int64_t n) {
    size_t t = 0; cub::DeviceRunLengthEncode::Encode(nullptr, t, (uint32_t*)nullptr, (uint32_t*)nullptr, (int32_t*)nullptr, (int32_t*)nullptr, (int)n); return t;
}

static void launch_cub_rle_u32(void* dt, size_t tb, uint32_t* di, uint32_t* du, int32_t* dc, int32_t* dn, int64_t n, cudaStream_t s) {
    cub::DeviceRunLengthEncode::Encode(dt, tb, di, du, dc, dn, (int)n, s);
}

static size_t cub_rle_u64_temp_bytes(int64_t n) {
    size_t t = 0; cub::DeviceRunLengthEncode::Encode(nullptr, t, (uint64_t*)nullptr, (uint64_t*)nullptr, (int32_t*)nullptr, (int32_t*)nullptr, (int)n); return t;
}

static void launch_cub_rle_u64(void* dt, size_t tb, uint64_t* di, uint64_t* du, int32_t* dc, int32_t* dn, int64_t n, cudaStream_t s) {
    cub::DeviceRunLengthEncode::Encode(dt, tb, di, du, dc, dn, (int)n, s);
}

static void launch_compute_scores_u32(const uint32_t* uk, const int32_t* rl, int nu, int V,
    const int32_t* seeds, const int32_t* off, int32_t* of_, int32_t* os_, float* osc, int32_t* oc, cudaStream_t s) {
    if (nu == 0) return;
    compute_scores_kernel_u32<<<(nu+255)/256, 256, 0, s>>>(uk, rl, nu, V, seeds, off, of_, os_, osc, oc);
}

static void launch_compute_scores_u64(const uint64_t* uk, const int32_t* rl, int nu, int64_t V,
    const int32_t* seeds, const int32_t* off, int32_t* of_, int32_t* os_, float* osc, int32_t* oc, cudaStream_t s) {
    if (nu == 0) return;
    compute_scores_kernel_u64<<<(nu+255)/256, 256, 0, s>>>(uk, rl, nu, V, seeds, off, of_, os_, osc, oc);
}

static void launch_compute_scores_multigraph_u32(const uint32_t* uk, int nu, int V,
    const int32_t* seeds, const int32_t* off, const int32_t* ind,
    int32_t* of_, int32_t* os_, float* osc, int32_t* oc, cudaStream_t s) {
    if (nu == 0) return;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (nu + warps_per_block - 1) / warps_per_block;
    compute_scores_multigraph_warp<uint32_t><<<blocks, threads, 0, s>>>(uk, nu, (int64_t)V, seeds, off, ind, of_, os_, osc, oc);
}

static void launch_compute_scores_multigraph_u64(const uint64_t* uk, int nu, int64_t V,
    const int32_t* seeds, const int32_t* off, const int32_t* ind,
    int32_t* of_, int32_t* os_, float* osc, int32_t* oc, cudaStream_t s) {
    if (nu == 0) return;
    int warps_per_block = 8;
    int threads = warps_per_block * 32;
    int blocks = (nu + warps_per_block - 1) / warps_per_block;
    compute_scores_multigraph_warp<uint64_t><<<blocks, threads, 0, s>>>(uk, nu, V, seeds, off, ind, of_, os_, osc, oc);
}

static size_t cub_sort_pairs_desc_temp_bytes(int n) {
    size_t t = 0; cub::DeviceRadixSort::SortPairsDescending(nullptr, t, (float*)nullptr, (float*)nullptr, (int32_t*)nullptr, (int32_t*)nullptr, n); return t;
}

static void launch_cub_sort_pairs_desc(void* dt, size_t tb, float* ki, float* ko, int32_t* vi, int32_t* vo, int n, cudaStream_t s) {
    cub::DeviceRadixSort::SortPairsDescending(dt, tb, ki, ko, vi, vo, n, 0, 32, s);
}

static void launch_gather_topk(const int32_t* si, const int32_t* if_, const int32_t* is_, const float* isc,
    int c, int32_t* of_, int32_t* os_, float* osc, cudaStream_t s) {
    if (c == 0) return;
    gather_topk_kernel<<<(c+255)/256, 256, 0, s>>>(si, if_, is_, isc, c, of_, os_, osc);
}

static void launch_iota(int32_t* d, int n, cudaStream_t s) {
    if (n == 0) return;
    iota_kernel<<<(n+255)/256, 256, 0, s>>>(d, n);
}

}  

similarity_result_float_t overlap_all_pairs_similarity_seg(
    const graph32_t& graph,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t V = graph.number_of_vertices;
    bool is_multigraph = graph.is_multigraph;
    cudaStream_t stream = 0;
    int64_t* h_pinned = cache.h_pinned;

    int64_t topk_raw = topk.has_value() ? (int64_t)topk.value() : -1;

    
    int num_seeds;
    const int32_t* d_seeds;
    DevMem seeds_buf;
    if (vertices != nullptr && num_vertices > 0) {
        num_seeds = (int)num_vertices;
        d_seeds = vertices;
    } else {
        num_seeds = V;
        seeds_buf.alloc(num_seeds * sizeof(int32_t));
        launch_iota(seeds_buf.as<int32_t>(), num_seeds, stream);
        d_seeds = seeds_buf.as<const int32_t>();
    }

    if (num_seeds == 0) return {nullptr, nullptr, nullptr, 0};

    
    DevMem counts_buf;
    counts_buf.alloc(num_seeds * sizeof(int64_t));
    launch_count_2hop(d_offsets, d_indices, d_seeds, num_seeds, counts_buf.as<int64_t>(), stream);

    
    DevMem offsets_out_buf;
    offsets_out_buf.alloc((num_seeds + 1) * sizeof(int64_t));
    {
        size_t tb = cub_prefix_sum_temp_bytes(num_seeds);
        DevMem tmp; tmp.alloc(tb);
        launch_cub_prefix_sum(tmp.ptr, tb,
            counts_buf.as<int64_t>(), offsets_out_buf.as<int64_t>(), num_seeds, stream);
    }

    
    cudaMemcpyAsync(&h_pinned[0], offsets_out_buf.as<int64_t>() + num_seeds - 1,
        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_pinned[1], counts_buf.as<int64_t>() + num_seeds - 1,
        sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int64_t total_2hop = h_pinned[0] + h_pinned[1];

    if (total_2hop == 0) return {nullptr, nullptr, nullptr, 0};

    int64_t max_key = (int64_t)(num_seeds - 1) * V + (V - 1);
    bool use_u32 = (max_key <= (int64_t)UINT32_MAX);
    int end_bit = bits_needed(max_key);

    int blocks_per_seed;
    if (num_seeds <= 100) blocks_per_seed = 64;
    else if (num_seeds <= 1000) blocks_per_seed = 8;
    else blocks_per_seed = 1;

    DevMem seed_counters_buf;
    seed_counters_buf.alloc(num_seeds * sizeof(int32_t));
    cudaMemsetAsync(seed_counters_buf.as<int32_t>(), 0, num_seeds * sizeof(int32_t), stream);

    
    DevMem of_buf, os_buf, osc_buf;

    if (use_u32) {
        DevMem keys_buf; keys_buf.alloc(total_2hop * sizeof(uint32_t));
        launch_generate_multiblock_u32(d_offsets, d_indices, d_seeds, num_seeds, (int)V,
            offsets_out_buf.as<int64_t>(), keys_buf.as<uint32_t>(),
            seed_counters_buf.as<int32_t>(), blocks_per_seed, stream);

        DevMem sorted_keys_buf; sorted_keys_buf.alloc(total_2hop * sizeof(uint32_t));
        {
            size_t tb = cub_sort_u32_temp_bytes(total_2hop, 0, end_bit);
            DevMem tmp; tmp.alloc(tb);
            launch_cub_sort_u32(tmp.ptr, tb,
                keys_buf.as<uint32_t>(), sorted_keys_buf.as<uint32_t>(),
                total_2hop, 0, end_bit, stream);
        }

        DevMem unique_keys_buf; unique_keys_buf.alloc(total_2hop * sizeof(uint32_t));
        DevMem run_lengths_buf; run_lengths_buf.alloc(total_2hop * sizeof(int32_t));
        DevMem num_runs_buf; num_runs_buf.alloc(sizeof(int32_t));
        {
            size_t tb = cub_rle_u32_temp_bytes(total_2hop);
            DevMem tmp; tmp.alloc(tb);
            launch_cub_rle_u32(tmp.ptr, tb,
                sorted_keys_buf.as<uint32_t>(), unique_keys_buf.as<uint32_t>(),
                run_lengths_buf.as<int32_t>(), num_runs_buf.as<int32_t>(), total_2hop, stream);
        }

        cudaMemcpyAsync(&h_pinned[0], num_runs_buf.as<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int32_t h_num_runs = (int32_t)h_pinned[0];
        if (h_num_runs == 0) return {nullptr, nullptr, nullptr, 0};

        of_buf.alloc(h_num_runs * sizeof(int32_t));
        os_buf.alloc(h_num_runs * sizeof(int32_t));
        osc_buf.alloc(h_num_runs * sizeof(float));
        DevMem oc_buf; oc_buf.alloc(sizeof(int32_t));
        cudaMemsetAsync(oc_buf.as<int32_t>(), 0, sizeof(int32_t), stream);

        if (is_multigraph)
            launch_compute_scores_multigraph_u32(unique_keys_buf.as<uint32_t>(), h_num_runs, (int)V,
                d_seeds, d_offsets, d_indices,
                of_buf.as<int32_t>(), os_buf.as<int32_t>(), osc_buf.as<float>(),
                oc_buf.as<int32_t>(), stream);
        else
            launch_compute_scores_u32(unique_keys_buf.as<uint32_t>(), run_lengths_buf.as<int32_t>(),
                h_num_runs, (int)V, d_seeds, d_offsets,
                of_buf.as<int32_t>(), os_buf.as<int32_t>(), osc_buf.as<float>(),
                oc_buf.as<int32_t>(), stream);

        cudaMemcpyAsync(&h_pinned[0], oc_buf.as<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    } else {
        
        DevMem keys_buf; keys_buf.alloc(total_2hop * sizeof(uint64_t));
        launch_generate_multiblock_u64(d_offsets, d_indices, d_seeds, num_seeds, (int64_t)V,
            offsets_out_buf.as<int64_t>(), keys_buf.as<uint64_t>(),
            seed_counters_buf.as<int32_t>(), blocks_per_seed, stream);

        DevMem sorted_keys_buf; sorted_keys_buf.alloc(total_2hop * sizeof(uint64_t));
        {
            size_t tb = cub_sort_u64_temp_bytes(total_2hop, 0, end_bit);
            DevMem tmp; tmp.alloc(tb);
            launch_cub_sort_u64(tmp.ptr, tb,
                keys_buf.as<uint64_t>(), sorted_keys_buf.as<uint64_t>(),
                total_2hop, 0, end_bit, stream);
        }

        DevMem unique_keys_buf; unique_keys_buf.alloc(total_2hop * sizeof(uint64_t));
        DevMem run_lengths_buf; run_lengths_buf.alloc(total_2hop * sizeof(int32_t));
        DevMem num_runs_buf; num_runs_buf.alloc(sizeof(int32_t));
        {
            size_t tb = cub_rle_u64_temp_bytes(total_2hop);
            DevMem tmp; tmp.alloc(tb);
            launch_cub_rle_u64(tmp.ptr, tb,
                sorted_keys_buf.as<uint64_t>(), unique_keys_buf.as<uint64_t>(),
                run_lengths_buf.as<int32_t>(), num_runs_buf.as<int32_t>(), total_2hop, stream);
        }

        cudaMemcpyAsync(&h_pinned[0], num_runs_buf.as<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int32_t h_num_runs = (int32_t)h_pinned[0];
        if (h_num_runs == 0) return {nullptr, nullptr, nullptr, 0};

        of_buf.alloc(h_num_runs * sizeof(int32_t));
        os_buf.alloc(h_num_runs * sizeof(int32_t));
        osc_buf.alloc(h_num_runs * sizeof(float));
        DevMem oc_buf; oc_buf.alloc(sizeof(int32_t));
        cudaMemsetAsync(oc_buf.as<int32_t>(), 0, sizeof(int32_t), stream);

        if (is_multigraph)
            launch_compute_scores_multigraph_u64(unique_keys_buf.as<uint64_t>(), h_num_runs, (int64_t)V,
                d_seeds, d_offsets, d_indices,
                of_buf.as<int32_t>(), os_buf.as<int32_t>(), osc_buf.as<float>(),
                oc_buf.as<int32_t>(), stream);
        else
            launch_compute_scores_u64(unique_keys_buf.as<uint64_t>(), run_lengths_buf.as<int32_t>(),
                h_num_runs, (int64_t)V, d_seeds, d_offsets,
                of_buf.as<int32_t>(), os_buf.as<int32_t>(), osc_buf.as<float>(),
                oc_buf.as<int32_t>(), stream);

        cudaMemcpyAsync(&h_pinned[0], oc_buf.as<int32_t>(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }

    int32_t h_out_count = (int32_t)h_pinned[0];
    if (h_out_count == 0) return {nullptr, nullptr, nullptr, 0};

    
    bool need_topk = (topk_raw >= 0 && topk_raw < h_out_count);
    bool need_sort = (topk_raw >= 0);

    if (!need_sort) {
        similarity_result_float_t result;
        cudaMalloc(&result.first, h_out_count * sizeof(int32_t));
        cudaMalloc(&result.second, h_out_count * sizeof(int32_t));
        cudaMalloc(&result.scores, h_out_count * sizeof(float));
        result.count = h_out_count;
        cudaMemcpyAsync(result.first, of_buf.as<int32_t>(), h_out_count * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(result.second, os_buf.as<int32_t>(), h_out_count * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(result.scores, osc_buf.as<float>(), h_out_count * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        return result;
    }

    int output_count = need_topk ? (int)topk_raw : h_out_count;

    
    DevMem sorted_scores_buf; sorted_scores_buf.alloc(h_out_count * sizeof(float));
    DevMem iota_buf; iota_buf.alloc(h_out_count * sizeof(int32_t));
    DevMem sorted_indices_buf; sorted_indices_buf.alloc(h_out_count * sizeof(int32_t));
    launch_iota(iota_buf.as<int32_t>(), h_out_count, stream);
    {
        size_t tb = cub_sort_pairs_desc_temp_bytes(h_out_count);
        DevMem tmp; tmp.alloc(tb);
        launch_cub_sort_pairs_desc(tmp.ptr, tb,
            osc_buf.as<float>(), sorted_scores_buf.as<float>(),
            iota_buf.as<int32_t>(), sorted_indices_buf.as<int32_t>(),
            h_out_count, stream);
    }

    similarity_result_float_t result;
    cudaMalloc(&result.first, output_count * sizeof(int32_t));
    cudaMalloc(&result.second, output_count * sizeof(int32_t));
    cudaMalloc(&result.scores, output_count * sizeof(float));
    result.count = output_count;
    launch_gather_topk(sorted_indices_buf.as<int32_t>(),
        of_buf.as<int32_t>(), os_buf.as<int32_t>(), osc_buf.as<float>(),
        output_count, result.first, result.second, result.scores, stream);

    return result;
}

}  
