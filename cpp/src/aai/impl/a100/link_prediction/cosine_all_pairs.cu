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
#include <optional>

namespace aai {

namespace {

struct Cache : Cacheable {
    int64_t* base_offsets = nullptr;
    int64_t base_offsets_capacity = 0;

    unsigned long long* counter = nullptr;
    bool counter_allocated = false;

    uint32_t* bitmaps = nullptr;
    int64_t bitmaps_capacity = 0;

    void ensure_base_offsets(int64_t n) {
        if (base_offsets_capacity < n) {
            if (base_offsets) cudaFree(base_offsets);
            cudaMalloc(&base_offsets, n * sizeof(int64_t));
            base_offsets_capacity = n;
        }
    }

    void ensure_counter() {
        if (!counter_allocated) {
            cudaMalloc(&counter, sizeof(unsigned long long));
            counter_allocated = true;
        }
    }

    void ensure_bitmaps(int64_t n) {
        if (bitmaps_capacity < n) {
            if (bitmaps) cudaFree(bitmaps);
            cudaMalloc(&bitmaps, n * sizeof(uint32_t));
            bitmaps_capacity = n;
        }
    }

    ~Cache() override {
        if (base_offsets) cudaFree(base_offsets);
        if (counter) cudaFree(counter);
        if (bitmaps) cudaFree(bitmaps);
    }
};



__global__ void fill_float_kernel(float* __restrict__ data, int64_t n) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    int64_t idx4 = idx * 4;
    if (idx4 + 3 < n) {
        float4 val = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        ((float4*)data)[idx] = val;
    } else {
        
        for (int64_t i = idx4; i < n && i < idx4 + 4; i++)
            data[i] = 1.0f;
    }
}



__device__ __forceinline__ void build_bitmap_warp_coop(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t u, int32_t u_start, int32_t deg_u,
    uint32_t* bitmap
) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int num_warps = blockDim.x >> 5;

    for (int i = warp_id; i < deg_u; i += num_warps) {
        int32_t w = indices[u_start + i];
        int32_t w_start = offsets[w];
        int32_t w_end = offsets[w + 1];
        for (int j = w_start + lane; j < w_end; j += 32) {
            int32_t v = indices[j];
            if (v != u)
                atomicOr(&bitmap[v >> 5], 1u << (v & 31));
        }
    }
}



__device__ __forceinline__ int block_popcount(uint32_t* bitmap, int bitmap_words) {
    int local_count = 0;
    for (int i = threadIdx.x; i < bitmap_words; i += blockDim.x)
        local_count += __popc(bitmap[i]);

    
    for (int offset = 16; offset > 0; offset >>= 1)
        local_count += __shfl_down_sync(0xffffffff, local_count, offset);

    __shared__ int s_warp_counts[16]; 
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int num_warps = (blockDim.x + 31) >> 5;

    if (lane == 0) s_warp_counts[warp_id] = local_count;
    __syncthreads();

    if (warp_id == 0) {
        local_count = (lane < num_warps) ? s_warp_counts[lane] : 0;
        for (int offset = 16; offset > 0; offset >>= 1)
            local_count += __shfl_down_sync(0xffffffff, local_count, offset);
    }
    return local_count; 
}



__global__ void count_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t num_vertices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ base_offsets,
    unsigned long long* __restrict__ global_counter
) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    int32_t u = (seeds != nullptr) ? seeds[seed_idx] : seed_idx;
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t deg_u = u_end - u_start;

    if (deg_u == 0) {
        if (threadIdx.x == 0) base_offsets[seed_idx] = 0;
        return;
    }

    extern __shared__ uint32_t bitmap[];
    int bitmap_words = (num_vertices + 31) / 32;

    for (int i = threadIdx.x; i < bitmap_words; i += blockDim.x)
        bitmap[i] = 0;
    __syncthreads();

    build_bitmap_warp_coop(offsets, indices, u, u_start, deg_u, bitmap);
    __syncthreads();

    int total = block_popcount(bitmap, bitmap_words);

    if (threadIdx.x == 0) {
        if (total > 0)
            base_offsets[seed_idx] = (int64_t)atomicAdd(global_counter, (unsigned long long)total);
        else
            base_offsets[seed_idx] = 0;
    }
}



__global__ void compute_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t num_vertices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ base_offsets,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second
) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    int32_t u = (seeds != nullptr) ? seeds[seed_idx] : seed_idx;
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t deg_u = u_end - u_start;

    if (deg_u == 0) return;

    extern __shared__ char shmem_raw[];
    int bitmap_words = (num_vertices + 31) / 32;
    uint32_t* bitmap = (uint32_t*)shmem_raw;
    int* out_counter = (int*)(shmem_raw + bitmap_words * sizeof(uint32_t));

    for (int i = threadIdx.x; i < bitmap_words; i += blockDim.x)
        bitmap[i] = 0;
    if (threadIdx.x == 0) *out_counter = 0;
    __syncthreads();

    build_bitmap_warp_coop(offsets, indices, u, u_start, deg_u, bitmap);
    __syncthreads();

    int64_t out_base = base_offsets[seed_idx];

    
    for (int w = threadIdx.x; w < bitmap_words; w += blockDim.x) {
        uint32_t word = bitmap[w];
        while (word) {
            int bit = __ffs(word) - 1;
            int32_t v = w * 32 + bit;
            word &= word - 1;
            if (v >= num_vertices) break;

            int pos = atomicAdd(out_counter, 1);
            int64_t gpos = out_base + pos;
            out_first[gpos] = u;
            out_second[gpos] = v;
        }
    }
}



__global__ void compute_topk_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t num_vertices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    unsigned long long* __restrict__ global_counter,
    int64_t topk
) {
    __shared__ unsigned long long s_block_base;
    __shared__ int s_out_counter;

    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    if (*(volatile unsigned long long*)global_counter >= (unsigned long long)topk) return;

    int32_t u = (seeds != nullptr) ? seeds[seed_idx] : seed_idx;
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t deg_u = u_end - u_start;

    if (deg_u == 0) return;

    extern __shared__ uint32_t bitmap[];
    int bitmap_words = (num_vertices + 31) / 32;

    for (int i = threadIdx.x; i < bitmap_words; i += blockDim.x)
        bitmap[i] = 0;
    __syncthreads();

    build_bitmap_warp_coop(offsets, indices, u, u_start, deg_u, bitmap);
    __syncthreads();

    int total = block_popcount(bitmap, bitmap_words);

    if (threadIdx.x == 0)
        s_block_base = atomicAdd(global_counter, (unsigned long long)total);
    __syncthreads();

    if (s_block_base >= (unsigned long long)topk) return;

    if (threadIdx.x == 0) s_out_counter = 0;
    __syncthreads();

    for (int w = threadIdx.x; w < bitmap_words; w += blockDim.x) {
        uint32_t word = bitmap[w];
        while (word) {
            int bit = __ffs(word) - 1;
            int32_t v = w * 32 + bit;
            word &= word - 1;
            if (v >= num_vertices) break;

            int pos = atomicAdd(&s_out_counter, 1);
            unsigned long long gpos = s_block_base + (unsigned long long)pos;
            if (gpos < (unsigned long long)topk) {
                out_first[gpos] = u;
                out_second[gpos] = v;
                out_scores[gpos] = 1.0f;
            }
        }
    }
}



__global__ void count_kernel_gmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t num_vertices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ base_offsets,
    unsigned long long* __restrict__ global_counter,
    uint32_t* __restrict__ global_bitmaps,
    int bitmap_words
) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    int32_t u = (seeds != nullptr) ? seeds[seed_idx] : seed_idx;
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t deg_u = u_end - u_start;

    if (deg_u == 0) {
        if (threadIdx.x == 0) base_offsets[seed_idx] = 0;
        return;
    }

    uint32_t* bitmap = global_bitmaps + (int64_t)seed_idx * bitmap_words;

    for (int i = threadIdx.x; i < bitmap_words; i += blockDim.x)
        bitmap[i] = 0;
    __syncthreads();

    build_bitmap_warp_coop(offsets, indices, u, u_start, deg_u, bitmap);
    __syncthreads();

    int total = block_popcount(bitmap, bitmap_words);

    if (threadIdx.x == 0) {
        if (total > 0)
            base_offsets[seed_idx] = (int64_t)atomicAdd(global_counter, (unsigned long long)total);
        else
            base_offsets[seed_idx] = 0;
    }
}

__global__ void compute_kernel_gmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t num_vertices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ base_offsets,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    uint32_t* __restrict__ global_bitmaps,
    int bitmap_words
) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    int32_t u = (seeds != nullptr) ? seeds[seed_idx] : seed_idx;
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t deg_u = u_end - u_start;

    if (deg_u == 0) return;

    uint32_t* bitmap = global_bitmaps + (int64_t)seed_idx * bitmap_words;
    __shared__ int out_counter;

    for (int i = threadIdx.x; i < bitmap_words; i += blockDim.x)
        bitmap[i] = 0;
    if (threadIdx.x == 0) out_counter = 0;
    __syncthreads();

    build_bitmap_warp_coop(offsets, indices, u, u_start, deg_u, bitmap);
    __syncthreads();

    int64_t out_base = base_offsets[seed_idx];

    for (int w = threadIdx.x; w < bitmap_words; w += blockDim.x) {
        uint32_t word = bitmap[w];
        while (word) {
            int bit = __ffs(word) - 1;
            int32_t v = w * 32 + bit;
            word &= word - 1;
            if (v >= num_vertices) break;

            int pos = atomicAdd(&out_counter, 1);
            int64_t gpos = out_base + pos;
            out_first[gpos] = u;
            out_second[gpos] = v;
        }
    }
}

__global__ void compute_topk_kernel_gmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t num_vertices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    float* __restrict__ out_scores,
    unsigned long long* __restrict__ global_counter,
    int64_t topk,
    uint32_t* __restrict__ global_bitmaps,
    int bitmap_words
) {
    __shared__ unsigned long long s_block_base;
    __shared__ int s_out_counter;

    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    if (*(volatile unsigned long long*)global_counter >= (unsigned long long)topk) return;

    int32_t u = (seeds != nullptr) ? seeds[seed_idx] : seed_idx;
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t deg_u = u_end - u_start;

    if (deg_u == 0) return;

    uint32_t* bitmap = global_bitmaps + (int64_t)seed_idx * bitmap_words;

    for (int i = threadIdx.x; i < bitmap_words; i += blockDim.x)
        bitmap[i] = 0;
    __syncthreads();

    build_bitmap_warp_coop(offsets, indices, u, u_start, deg_u, bitmap);
    __syncthreads();

    int total = block_popcount(bitmap, bitmap_words);

    if (threadIdx.x == 0)
        s_block_base = atomicAdd(global_counter, (unsigned long long)total);
    __syncthreads();

    if (s_block_base >= (unsigned long long)topk) return;

    if (threadIdx.x == 0) s_out_counter = 0;
    __syncthreads();

    for (int w = threadIdx.x; w < bitmap_words; w += blockDim.x) {
        uint32_t word = bitmap[w];
        while (word) {
            int bit = __ffs(word) - 1;
            int32_t v = w * 32 + bit;
            word &= word - 1;
            if (v >= num_vertices) break;

            int pos = atomicAdd(&s_out_counter, 1);
            unsigned long long gpos = s_block_base + (unsigned long long)pos;
            if (gpos < (unsigned long long)topk) {
                out_first[gpos] = u;
                out_second[gpos] = v;
                out_scores[gpos] = 1.0f;
            }
        }
    }
}



void launch_fill_scores(float* data, int64_t n, cudaStream_t stream) {
    if (n == 0) return;
    int64_t n4 = (n + 3) / 4;
    int block = 256;
    int grid = (int)((n4 + block - 1) / block);
    fill_float_kernel<<<grid, block, 0, stream>>>(data, n);
}

void launch_count(
    const int32_t* offsets, const int32_t* indices,
    int32_t num_vertices, const int32_t* seeds, int32_t num_seeds,
    int64_t* base_offsets, unsigned long long* global_counter,
    int block_size, cudaStream_t stream
) {
    if (num_seeds == 0) return;
    int shmem = ((num_vertices + 31) / 32) * (int)sizeof(uint32_t);
    cudaFuncSetAttribute(count_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    count_kernel<<<num_seeds, block_size, shmem, stream>>>(
        offsets, indices, num_vertices, seeds, num_seeds,
        base_offsets, global_counter);
}

void launch_compute(
    const int32_t* offsets, const int32_t* indices,
    int32_t num_vertices, const int32_t* seeds, int32_t num_seeds,
    const int64_t* base_offsets,
    int32_t* out_first, int32_t* out_second,
    int block_size, cudaStream_t stream
) {
    if (num_seeds == 0) return;
    int shmem = ((num_vertices + 31) / 32) * (int)sizeof(uint32_t) + (int)sizeof(int);
    cudaFuncSetAttribute(compute_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    compute_kernel<<<num_seeds, block_size, shmem, stream>>>(
        offsets, indices, num_vertices, seeds, num_seeds,
        base_offsets, out_first, out_second);
}

void launch_compute_topk(
    const int32_t* offsets, const int32_t* indices,
    int32_t num_vertices, const int32_t* seeds, int32_t num_seeds,
    int32_t* out_first, int32_t* out_second, float* out_scores,
    unsigned long long* global_counter, int64_t topk,
    int block_size, cudaStream_t stream
) {
    if (num_seeds == 0) return;
    int shmem = ((num_vertices + 31) / 32) * (int)sizeof(uint32_t);
    cudaFuncSetAttribute(compute_topk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    compute_topk_kernel<<<num_seeds, block_size, shmem, stream>>>(
        offsets, indices, num_vertices, seeds, num_seeds,
        out_first, out_second, out_scores,
        global_counter, topk);
}

void launch_count_gmem(
    const int32_t* offsets, const int32_t* indices,
    int32_t num_vertices, const int32_t* seeds, int32_t num_seeds,
    int64_t* base_offsets, unsigned long long* global_counter,
    uint32_t* global_bitmaps, int bitmap_words,
    int block_size, cudaStream_t stream
) {
    if (num_seeds == 0) return;
    count_kernel_gmem<<<num_seeds, block_size, 0, stream>>>(
        offsets, indices, num_vertices, seeds, num_seeds,
        base_offsets, global_counter, global_bitmaps, bitmap_words);
}

void launch_compute_gmem(
    const int32_t* offsets, const int32_t* indices,
    int32_t num_vertices, const int32_t* seeds, int32_t num_seeds,
    const int64_t* base_offsets,
    int32_t* out_first, int32_t* out_second,
    uint32_t* global_bitmaps, int bitmap_words,
    int block_size, cudaStream_t stream
) {
    if (num_seeds == 0) return;
    compute_kernel_gmem<<<num_seeds, block_size, 0, stream>>>(
        offsets, indices, num_vertices, seeds, num_seeds,
        base_offsets, out_first, out_second, global_bitmaps, bitmap_words);
}

void launch_compute_topk_gmem(
    const int32_t* offsets, const int32_t* indices,
    int32_t num_vertices, const int32_t* seeds, int32_t num_seeds,
    int32_t* out_first, int32_t* out_second, float* out_scores,
    unsigned long long* global_counter, int64_t topk,
    uint32_t* global_bitmaps, int bitmap_words,
    int block_size, cudaStream_t stream
) {
    if (num_seeds == 0) return;
    compute_topk_kernel_gmem<<<num_seeds, block_size, 0, stream>>>(
        offsets, indices, num_vertices, seeds, num_seeds,
        out_first, out_second, out_scores,
        global_counter, topk, global_bitmaps, bitmap_words);
}

}  

similarity_result_float_t cosine_all_pairs_similarity(const graph32_t& graph,
                                                      const int32_t* vertices,
                                                      std::size_t num_vertices,
                                                      std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n_verts = graph.number_of_vertices;
    cudaStream_t stream = 0;

    int32_t num_seeds;
    const int32_t* d_seeds;

    if (vertices != nullptr) {
        d_seeds = vertices;
        num_seeds = (int32_t)num_vertices;
    } else {
        d_seeds = nullptr;
        num_seeds = n_verts;
    }

    if (num_seeds == 0 || n_verts == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int block_size = 512;

    
    int bitmap_words = (n_verts + 31) / 32;
    bool use_gmem = ((int64_t)bitmap_words * (int)sizeof(uint32_t) > 160 * 1024);
    uint32_t* d_bitmaps = nullptr;
    if (use_gmem) {
        cache.ensure_bitmaps((int64_t)num_seeds * bitmap_words);
        d_bitmaps = cache.bitmaps;
    }

    bool use_topk = topk.has_value();

    if (use_topk) {
        int64_t topk_val = (int64_t)topk.value();

        if (topk_val == 0) {
            return {nullptr, nullptr, nullptr, 0};
        }

        int32_t* out_first = nullptr;
        int32_t* out_second = nullptr;
        float* out_scores = nullptr;
        cudaMalloc(&out_first, topk_val * sizeof(int32_t));
        cudaMalloc(&out_second, topk_val * sizeof(int32_t));
        cudaMalloc(&out_scores, topk_val * sizeof(float));

        cache.ensure_counter();
        cudaMemsetAsync(cache.counter, 0, sizeof(unsigned long long), stream);

        if (use_gmem) {
            launch_compute_topk_gmem(
                d_offsets, d_indices, n_verts, d_seeds, num_seeds,
                out_first, out_second, out_scores,
                cache.counter, topk_val,
                d_bitmaps, bitmap_words,
                block_size, stream);
        } else {
            launch_compute_topk(
                d_offsets, d_indices, n_verts, d_seeds, num_seeds,
                out_first, out_second, out_scores,
                cache.counter, topk_val,
                block_size, stream);
        }

        int64_t actual = 0;
        cudaMemcpy(&actual, cache.counter, sizeof(int64_t), cudaMemcpyDeviceToHost);
        int64_t result_count = (actual < topk_val) ? actual : topk_val;

        if (result_count == topk_val) {
            return {out_first, out_second, out_scores, (std::size_t)result_count};
        }

        if (result_count <= 0) {
            cudaFree(out_first);
            cudaFree(out_second);
            cudaFree(out_scores);
            return {nullptr, nullptr, nullptr, 0};
        }

        
        int32_t* f = nullptr;
        int32_t* s = nullptr;
        float* sc = nullptr;
        cudaMalloc(&f, result_count * sizeof(int32_t));
        cudaMalloc(&s, result_count * sizeof(int32_t));
        cudaMalloc(&sc, result_count * sizeof(float));
        cudaMemcpyAsync(f, out_first, result_count * sizeof(int32_t),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(s, out_second, result_count * sizeof(int32_t),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(sc, out_scores, result_count * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
        
        cudaFree(out_first);
        cudaFree(out_second);
        cudaFree(out_scores);
        return {f, s, sc, (std::size_t)result_count};
    }

    
    cache.ensure_base_offsets(num_seeds);
    cache.ensure_counter();
    cudaMemsetAsync(cache.counter, 0, sizeof(unsigned long long), stream);

    
    if (use_gmem) {
        launch_count_gmem(d_offsets, d_indices, n_verts, d_seeds, num_seeds,
                         cache.base_offsets, cache.counter,
                         d_bitmaps, bitmap_words,
                         block_size, stream);
    } else {
        launch_count(d_offsets, d_indices, n_verts, d_seeds, num_seeds,
                     cache.base_offsets, cache.counter,
                     block_size, stream);
    }

    int64_t total = 0;
    cudaMemcpy(&total, cache.counter, sizeof(int64_t), cudaMemcpyDeviceToHost);

    if (total <= 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    int32_t* out_first = nullptr;
    int32_t* out_second = nullptr;
    float* out_scores = nullptr;
    cudaMalloc(&out_first, total * sizeof(int32_t));
    cudaMalloc(&out_second, total * sizeof(int32_t));
    cudaMalloc(&out_scores, total * sizeof(float));

    
    if (use_gmem) {
        launch_compute_gmem(d_offsets, d_indices, n_verts, d_seeds, num_seeds,
                           cache.base_offsets,
                           out_first, out_second,
                           d_bitmaps, bitmap_words,
                           block_size, stream);
    } else {
        launch_compute(d_offsets, d_indices, n_verts, d_seeds, num_seeds,
                       cache.base_offsets,
                       out_first, out_second,
                       block_size, stream);
    }

    
    launch_fill_scores(out_scores, total, stream);

    return {out_first, out_second, out_scores, (std::size_t)total};
}

}  
