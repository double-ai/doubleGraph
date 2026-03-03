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

namespace aai {

namespace {

struct Cache : Cacheable {
    uint8_t* clusters_buf = nullptr;
    float* accum_buf = nullptr;
    double* result_buf = nullptr;
    size_t clusters_buf_cap = 0;
    size_t accum_buf_cap = 0;

    Cache() {
        clusters_buf_cap = 64 * 1024 * 1024;
        accum_buf_cap = 8192 * sizeof(float);
        cudaMalloc(&clusters_buf, clusters_buf_cap);
        cudaMalloc(&accum_buf, accum_buf_cap);
        cudaMalloc(&result_buf, sizeof(double));
    }

    void ensure(size_t clusters_needed, size_t accum_needed) {
        if (clusters_needed > clusters_buf_cap) {
            if (clusters_buf) cudaFree(clusters_buf);
            clusters_buf_cap = clusters_needed * 2;
            cudaMalloc(&clusters_buf, clusters_buf_cap);
        }
        if (accum_needed > accum_buf_cap) {
            if (accum_buf) cudaFree(accum_buf);
            accum_buf_cap = accum_needed * 2;
            cudaMalloc(&accum_buf, accum_buf_cap);
        }
    }

    ~Cache() override {
        if (clusters_buf) cudaFree(clusters_buf);
        if (accum_buf) cudaFree(accum_buf);
        if (result_buf) cudaFree(result_buf);
    }
};


__device__ __forceinline__ float warp_reduce_add(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}



template<int MODE>
__device__ __forceinline__ int get_cluster(const uint8_t* data, int idx) {
    if constexpr (MODE == 0) { 
        return (int)__ldg(&data[idx]);
    } else if constexpr (MODE == 1) { 
        return (int)__ldg(((const int32_t*)data) + idx);
    } else if constexpr (MODE == 2) { 
        return (__ldg(&data[idx >> 3]) >> (idx & 7)) & 1;
    } else { 
        return (__ldg(&data[idx >> 1]) >> ((idx & 1) << 2)) & 0xF;
    }
}


__global__ void convert_to_uint8_kernel(const int32_t* __restrict__ in,
                                         uint8_t* __restrict__ out,
                                         int32_t n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        int4 val = *reinterpret_cast<const int4*>(in + idx);
        uint32_t packed = (uint32_t)(val.x & 0xFF) |
                          ((uint32_t)(val.y & 0xFF) << 8) |
                          ((uint32_t)(val.z & 0xFF) << 16) |
                          ((uint32_t)(val.w & 0xFF) << 24);
        *reinterpret_cast<uint32_t*>(out + idx) = packed;
    } else {
        for (int i = 0; i < 4 && idx + i < n; i++)
            out[idx + i] = (uint8_t)in[idx + i];
    }
}


__global__ void convert_to_bits_kernel(const int32_t* __restrict__ in,
                                        uint8_t* __restrict__ out,
                                        int32_t n) {
    int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = byte_idx << 3;
    if (base >= n) return;

    uint8_t packed = 0;
    if (base + 7 < n) {
        int4 v0 = *reinterpret_cast<const int4*>(in + base);
        int4 v1 = *reinterpret_cast<const int4*>(in + base + 4);
        packed = (uint8_t)((v0.x & 1) | ((v0.y & 1) << 1) | ((v0.z & 1) << 2) | ((v0.w & 1) << 3) |
                 ((v1.x & 1) << 4) | ((v1.y & 1) << 5) | ((v1.z & 1) << 6) | ((v1.w & 1) << 7));
    } else {
        for (int i = 0; i < 8 && base + i < n; i++)
            if (in[base + i] & 1) packed |= (1 << i);
    }
    out[byte_idx] = packed;
}


__global__ void convert_to_nibble_kernel(const int32_t* __restrict__ in,
                                          uint8_t* __restrict__ out,
                                          int32_t n) {
    int byte_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = byte_idx << 1;
    if (base >= n) return;

    uint8_t packed = 0;
    if (base + 1 < n) {
        
        packed = (uint8_t)((in[base] & 0xF) | ((in[base + 1] & 0xF) << 4));
    } else {
        packed = (uint8_t)(in[base] & 0xF);
    }
    out[byte_idx] = packed;
}


template<int MODE>
__global__ void mod_high_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint8_t* __restrict__ clusters,
    float* __restrict__ g_cluster_deg,
    float* __restrict__ g_intra,
    int32_t v_start, int32_t v_end
) {
    int32_t v = v_start + blockIdx.x;
    if (v >= v_end) return;

    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    int32_t e_start = offsets[v], e_end = offsets[v + 1];
    int cv = get_cluster<MODE>(clusters, v);

    float my_intra = 0.0f, my_deg = 0.0f;
    for (int32_t e = e_start + tid; e < e_end; e += 256) {
        float w = __ldg(&weights[e]);
        my_deg += w;
        if (get_cluster<MODE>(clusters, __ldg(&indices[e])) == cv)
            my_intra += w;
    }

    my_intra = warp_reduce_add(my_intra);
    my_deg = warp_reduce_add(my_deg);

    __shared__ float s[16];
    if (lane == 0) { s[wid] = my_intra; s[8 + wid] = my_deg; }
    __syncthreads();

    if (tid < 8) {
        float bi = s[tid], bd = s[8 + tid];
        for (int m = 4; m > 0; m >>= 1) {
            bi += __shfl_xor_sync(0xff, bi, m);
            bd += __shfl_xor_sync(0xff, bd, m);
        }
        if (tid == 0) {
            atomicAdd(g_intra, bi);
            atomicAdd(&g_cluster_deg[cv], bd);
        }
    }
}


template<int MODE>
__global__ void mod_mid_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint8_t* __restrict__ clusters,
    float* __restrict__ g_cluster_deg,
    float* __restrict__ g_intra,
    int32_t v_start, int32_t v_end,
    int32_t num_clusters
) {
    extern __shared__ float smem[];
    float* s_cdeg = smem;
    const int wpb = blockDim.x >> 5;
    float* s_intra = smem + num_clusters;

    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    int gw = (blockIdx.x * blockDim.x + tid) >> 5;
    int32_t v = v_start + gw;

    for (int i = tid; i < num_clusters; i += blockDim.x) s_cdeg[i] = 0.0f;
    __syncthreads();

    float my_intra = 0.0f, my_deg = 0.0f;
    int cv = -1;

    if (v < v_end) {
        int32_t e_start = offsets[v], e_end = offsets[v + 1];
        cv = get_cluster<MODE>(clusters, v);
        for (int32_t e = e_start + lane; e < e_end; e += 32) {
            float w = __ldg(&weights[e]);
            my_deg += w;
            if (get_cluster<MODE>(clusters, __ldg(&indices[e])) == cv)
                my_intra += w;
        }
        my_intra = warp_reduce_add(my_intra);
        my_deg = warp_reduce_add(my_deg);
        if (lane == 0 && cv >= 0) atomicAdd(&s_cdeg[cv], my_deg);
    }

    if (lane == 0) s_intra[wid] = my_intra;
    __syncthreads();
    if (tid == 0) {
        float bi = 0.0f;
        for (int i = 0; i < wpb; i++) bi += s_intra[i];
        if (bi != 0.0f) atomicAdd(g_intra, bi);
    }
    for (int i = tid; i < num_clusters; i += blockDim.x)
        if (s_cdeg[i] != 0.0f) atomicAdd(&g_cluster_deg[i], s_cdeg[i]);
}


template<int MODE>
__global__ void mod_mid_degree_noshmem_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint8_t* __restrict__ clusters,
    float* __restrict__ g_cluster_deg,
    float* __restrict__ g_intra,
    int32_t v_start, int32_t v_end
) {
    const int wpb = blockDim.x >> 5;
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    int gw = (blockIdx.x * blockDim.x + tid) >> 5;
    int32_t v = v_start + gw;

    float my_intra = 0.0f, my_deg = 0.0f;
    int cv = -1;

    if (v < v_end) {
        int32_t e_start = offsets[v], e_end = offsets[v + 1];
        cv = get_cluster<MODE>(clusters, v);
        for (int32_t e = e_start + lane; e < e_end; e += 32) {
            float w = __ldg(&weights[e]);
            my_deg += w;
            if (get_cluster<MODE>(clusters, __ldg(&indices[e])) == cv)
                my_intra += w;
        }
        my_intra = warp_reduce_add(my_intra);
        my_deg = warp_reduce_add(my_deg);
        if (lane == 0 && cv >= 0) atomicAdd(&g_cluster_deg[cv], my_deg);
    }

    __shared__ float s_intra[8];
    if (lane == 0) s_intra[wid] = my_intra;
    __syncthreads();
    if (tid == 0) {
        float bi = 0.0f;
        for (int i = 0; i < wpb; i++) bi += s_intra[i];
        if (bi != 0.0f) atomicAdd(g_intra, bi);
    }
}


template<int MODE>
__global__ void mod_low_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint8_t* __restrict__ clusters,
    float* __restrict__ g_cluster_deg,
    float* __restrict__ g_intra,
    int32_t v_start, int32_t v_end,
    int32_t num_clusters
) {
    extern __shared__ float s_cdeg[];
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;

    for (int i = tid; i < num_clusters; i += blockDim.x) s_cdeg[i] = 0.0f;
    __syncthreads();

    int32_t v = v_start + blockIdx.x * blockDim.x + tid;
    float my_intra = 0.0f, my_deg = 0.0f;
    int cv = -1;

    if (v < v_end) {
        int32_t e_start = __ldg(&offsets[v]);
        int32_t e_end = __ldg(&offsets[v + 1]);
        if (e_start < e_end) {
            cv = get_cluster<MODE>(clusters, v);
            for (int32_t e = e_start; e < e_end; e++) {
                float w = __ldg(&weights[e]);
                my_deg += w;
                if (get_cluster<MODE>(clusters, __ldg(&indices[e])) == cv)
                    my_intra += w;
            }
        }
    }

    float warp_intra = warp_reduce_add(my_intra);
    __shared__ float s_intra[8];
    if (lane == 0) s_intra[wid] = warp_intra;
    __syncthreads();
    if (tid == 0) {
        float bi = 0.0f;
        for (int i = 0; i < (blockDim.x >> 5); i++) bi += s_intra[i];
        if (bi != 0.0f) atomicAdd(g_intra, bi);
    }

    if (cv >= 0 && my_deg != 0.0f) atomicAdd(&s_cdeg[cv], my_deg);
    __syncthreads();
    for (int i = tid; i < num_clusters; i += blockDim.x)
        if (s_cdeg[i] != 0.0f) atomicAdd(&g_cluster_deg[i], s_cdeg[i]);
}


template<int MODE>
__global__ void mod_low_degree_noshmem_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ weights,
    const uint8_t* __restrict__ clusters,
    float* __restrict__ g_cluster_deg,
    float* __restrict__ g_intra,
    int32_t v_start, int32_t v_end
) {
    int tid = threadIdx.x, lane = tid & 31, wid = tid >> 5;
    int32_t v = v_start + blockIdx.x * blockDim.x + tid;
    float my_intra = 0.0f, my_deg = 0.0f;
    int cv = -1;

    if (v < v_end) {
        int32_t e_start = __ldg(&offsets[v]);
        int32_t e_end = __ldg(&offsets[v + 1]);
        if (e_start < e_end) {
            cv = get_cluster<MODE>(clusters, v);
            for (int32_t e = e_start; e < e_end; e++) {
                float w = __ldg(&weights[e]);
                my_deg += w;
                if (get_cluster<MODE>(clusters, __ldg(&indices[e])) == cv)
                    my_intra += w;
            }
        }
    }

    float warp_intra = warp_reduce_add(my_intra);
    __shared__ float s_intra[8];
    if (lane == 0) s_intra[wid] = warp_intra;
    __syncthreads();
    if (tid == 0) {
        float bi = 0.0f;
        for (int i = 0; i < (blockDim.x >> 5); i++) bi += s_intra[i];
        if (bi != 0.0f) atomicAdd(g_intra, bi);
    }
    if (cv >= 0 && my_deg != 0.0f) atomicAdd(&g_cluster_deg[cv], my_deg);
}


__global__ void finalize_kernel(
    const float* __restrict__ cluster_deg,
    const float* __restrict__ intra_sum,
    double* __restrict__ result,
    int32_t num_clusters
) {
    int lane = threadIdx.x;
    double total_w = 0.0;
    for (int i = lane; i < num_clusters; i += 32)
        total_w += (double)cluster_deg[i];
    for (int m = 16; m > 0; m >>= 1)
        total_w += __shfl_xor_sync(0xffffffff, total_w, m);

    if (total_w == 0.0) { if (lane == 0) result[0] = 0.0; return; }

    total_w = __shfl_sync(0xffffffff, total_w, 0);
    double inv_tw = 1.0 / total_w;
    double penalty = 0.0;
    for (int i = lane; i < num_clusters; i += 32) {
        double r = (double)cluster_deg[i] * inv_tw;
        penalty += r * r;
    }
    for (int m = 16; m > 0; m >>= 1)
        penalty += __shfl_xor_sync(0xffffffff, penalty, m);

    if (lane == 0) result[0] = (double)intra_sum[0] * inv_tw - penalty;
}


template<int MODE>
static void launch_segments(
    const int32_t* off, const int32_t* idx, const float* wt,
    const uint8_t* cl, float* cdeg, float* intra,
    int32_t s0, int32_t s1, int32_t s2, int32_t s3,
    int32_t nc, bool use_sm, cudaStream_t st
) {
    if (s1 > s0)
        mod_high_degree_kernel<MODE><<<s1 - s0, 256, 0, st>>>(off, idx, wt, cl, cdeg, intra, s0, s1);

    if (s2 > s1) {
        int nv = s2 - s1, wpb = 8;
        int blk = (nv + wpb - 1) / wpb;
        if (use_sm) {
            size_t sm = (nc + wpb) * sizeof(float);
            mod_mid_degree_kernel<MODE><<<blk, wpb * 32, sm, st>>>(off, idx, wt, cl, cdeg, intra, s1, s2, nc);
        } else {
            mod_mid_degree_noshmem_kernel<MODE><<<blk, wpb * 32, 0, st>>>(off, idx, wt, cl, cdeg, intra, s1, s2);
        }
    }

    if (s3 > s2) {
        int nv = s3 - s2, block = 256, grid = (nv + block - 1) / block;
        if (use_sm) {
            size_t sm = nc * sizeof(float);
            mod_low_degree_kernel<MODE><<<grid, block, sm, st>>>(off, idx, wt, cl, cdeg, intra, s2, s3, nc);
        } else {
            mod_low_degree_noshmem_kernel<MODE><<<grid, block, 0, st>>>(off, idx, wt, cl, cdeg, intra, s2, s3);
        }
    }
}

}  

double analyze_clustering_modularity_seg(const graph32_t& graph,
                                         const float* edge_weights,
                                         std::size_t num_clusters,
                                         const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t nc = (int32_t)num_clusters;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg3 = seg[3];

    cache.ensure((size_t)num_vertices, ((size_t)nc + 1) * sizeof(float));

    float* cluster_degree_buf = cache.accum_buf;
    float* intra_sum_buf = cache.accum_buf + nc;

    cudaMemsetAsync(cache.accum_buf, 0, ((size_t)nc + 1) * sizeof(float), 0);

    cudaStream_t st = 0;
    bool use_sm = (nc <= 4096);

    if (nc == 2 && num_vertices > 0) {
        int nbytes = (num_vertices + 7) / 8;
        int grid = (nbytes + 255) / 256;
        convert_to_bits_kernel<<<grid, 256, 0, st>>>(cluster_assignments, cache.clusters_buf, num_vertices);
        launch_segments<2>(offsets, indices, edge_weights, cache.clusters_buf,
            cluster_degree_buf, intra_sum_buf, seg0, seg1, seg2, seg3, nc, use_sm, st);
    } else if (nc <= 16 && num_vertices > 0) {
        int nbytes = (num_vertices + 1) / 2;
        int grid = (nbytes + 255) / 256;
        convert_to_nibble_kernel<<<grid, 256, 0, st>>>(cluster_assignments, cache.clusters_buf, num_vertices);
        launch_segments<3>(offsets, indices, edge_weights, cache.clusters_buf,
            cluster_degree_buf, intra_sum_buf, seg0, seg1, seg2, seg3, nc, use_sm, st);
    } else if (nc <= 256 && num_vertices > 0) {
        int grid = ((num_vertices + 3) / 4 + 255) / 256;
        convert_to_uint8_kernel<<<grid, 256, 0, st>>>(cluster_assignments, cache.clusters_buf, num_vertices);
        launch_segments<0>(offsets, indices, edge_weights, cache.clusters_buf,
            cluster_degree_buf, intra_sum_buf, seg0, seg1, seg2, seg3, nc, use_sm, st);
    } else {
        launch_segments<1>(offsets, indices, edge_weights, (const uint8_t*)cluster_assignments,
            cluster_degree_buf, intra_sum_buf, seg0, seg1, seg2, seg3, nc, use_sm, st);
    }

    finalize_kernel<<<1, 32, 0, st>>>(cluster_degree_buf, intra_sum_buf, cache.result_buf, nc);

    double result;
    cudaMemcpy(&result, cache.result_buf, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

}  
