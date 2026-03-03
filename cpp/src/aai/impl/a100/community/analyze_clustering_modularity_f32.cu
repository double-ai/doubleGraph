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

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)



#define SMEM_CLUSTER_THRESHOLD 1024


#define MAX_BLOCKS 4096





struct Cache : Cacheable {
    float* d_cluster_degrees = nullptr;
    float* d_intra_sum = nullptr;
    void* d_compressed = nullptr;
    double* d_result = nullptr;

    std::size_t cluster_degrees_capacity = 0;
    std::size_t compressed_capacity_bytes = 0;
    bool intra_sum_allocated = false;
    bool result_allocated = false;

    static constexpr std::size_t kDefaultClusterCapacity = 1u << 16;  
    static constexpr std::size_t kDefaultCompressedCapacityBytes = 64ull * 1024ull * 1024ull;  

    Cache() {
        cudaMalloc(&d_intra_sum, sizeof(float));
        intra_sum_allocated = true;

        cluster_degrees_capacity = kDefaultClusterCapacity;
        cudaMalloc(&d_cluster_degrees, cluster_degrees_capacity * sizeof(float));

        compressed_capacity_bytes = kDefaultCompressedCapacityBytes;
        cudaMalloc(&d_compressed, compressed_capacity_bytes);

        cudaMalloc(&d_result, sizeof(double));
        result_allocated = true;
    }

    ~Cache() override {
        if (d_cluster_degrees) cudaFree(d_cluster_degrees);
        if (d_intra_sum) cudaFree(d_intra_sum);
        if (d_compressed) cudaFree(d_compressed);
        if (d_result) cudaFree(d_result);
    }

    void ensure_cluster_degrees(std::size_t num_clusters) {
        if (cluster_degrees_capacity < num_clusters) {
            if (d_cluster_degrees) cudaFree(d_cluster_degrees);
            cluster_degrees_capacity = num_clusters * 2;
            cudaMalloc(&d_cluster_degrees, cluster_degrees_capacity * sizeof(float));
        }
    }
};





__global__ void compress_to_u8(const int32_t* __restrict__ src, uint8_t* __restrict__ dst, int n)
{
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        dst[i] = static_cast<uint8_t>(src[i]);
    }
}

__global__ void compress_to_i16(const int32_t* __restrict__ src, int16_t* __restrict__ dst, int n)
{
    for (int i = blockIdx.x * BLOCK_SIZE + threadIdx.x; i < n; i += gridDim.x * BLOCK_SIZE) {
        dst[i] = static_cast<int16_t>(src[i]);
    }
}





template <typename CType>
__global__ void modularity_warp_smem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const CType* __restrict__ cluster_assignments,
    float* __restrict__ cluster_degrees,
    float* __restrict__ d_intra_sum,
    int32_t num_clusters,
    int32_t num_vertices)
{
    extern __shared__ float s_cluster_deg[];

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) s_cluster_deg[i] = 0.0f;
    __syncthreads();

    const int warp_id_global = (blockIdx.x * BLOCK_SIZE + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int total_warps = (gridDim.x * BLOCK_SIZE) >> 5;

    float warp_intra_acc = 0.0f;

    for (int v = warp_id_global; v < num_vertices; v += total_warps) {
        int32_t c_v_tmp = 0, start_tmp = 0, end_tmp = 0;
        if (lane == 0) {
            c_v_tmp = (int32_t)__ldg(&cluster_assignments[v]);
            start_tmp = __ldg(&offsets[v]);
            end_tmp = __ldg(&offsets[v + 1]);
        }
        const int32_t c_v = __shfl_sync(0xffffffff, c_v_tmp, 0);
        const int start = __shfl_sync(0xffffffff, start_tmp, 0);
        const int end = __shfl_sync(0xffffffff, end_tmp, 0);

        float deg = 0.0f;
        float intra = 0.0f;

        int e = start + lane;
        
        for (; e + 96 < end; e += 128) {
            float w0 = __ldg(&edge_weights[e]);
            int32_t n0 = __ldg(&indices[e]);
            deg += w0;
            intra += ((int32_t)__ldg(&cluster_assignments[n0]) == c_v) ? w0 : 0.0f;

            int e1 = e + 32;
            float w1 = __ldg(&edge_weights[e1]);
            int32_t n1 = __ldg(&indices[e1]);
            deg += w1;
            intra += ((int32_t)__ldg(&cluster_assignments[n1]) == c_v) ? w1 : 0.0f;

            int e2 = e + 64;
            float w2 = __ldg(&edge_weights[e2]);
            int32_t n2 = __ldg(&indices[e2]);
            deg += w2;
            intra += ((int32_t)__ldg(&cluster_assignments[n2]) == c_v) ? w2 : 0.0f;

            int e3 = e + 96;
            float w3 = __ldg(&edge_weights[e3]);
            int32_t n3 = __ldg(&indices[e3]);
            deg += w3;
            intra += ((int32_t)__ldg(&cluster_assignments[n3]) == c_v) ? w3 : 0.0f;
        }
        for (; e < end; e += 32) {
            float w = __ldg(&edge_weights[e]);
            deg += w;
            int32_t nbr = __ldg(&indices[e]);
            intra += ((int32_t)__ldg(&cluster_assignments[nbr]) == c_v) ? w : 0.0f;
        }

        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            deg += __shfl_down_sync(0xffffffff, deg, offset);
            intra += __shfl_down_sync(0xffffffff, intra, offset);
        }

        if (lane == 0) {
            atomicAdd(&s_cluster_deg[c_v], deg);
            warp_intra_acc += intra;
        }
    }

    
    __shared__ float s_warp_intra[WARPS_PER_BLOCK];
    const int warp_id_local = threadIdx.x >> 5;
    if (lane == 0) s_warp_intra[warp_id_local] = warp_intra_acc;
    __syncthreads();

    if (warp_id_local == 0) {
        float val = (lane < WARPS_PER_BLOCK) ? s_warp_intra[lane] : 0.0f;
        
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (lane == 0) atomicAdd(d_intra_sum, val);
    }

    __syncthreads();

    
    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        float v = s_cluster_deg[i];
        if (v != 0.0f) atomicAdd(&cluster_degrees[i], v);
    }
}

template <typename CType>
__global__ void modularity_warp_gmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const CType* __restrict__ cluster_assignments,
    float* __restrict__ cluster_degrees,
    float* __restrict__ d_intra_sum,
    int32_t num_vertices)
{
    const int warp_id_global = (blockIdx.x * BLOCK_SIZE + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int total_warps = (gridDim.x * BLOCK_SIZE) >> 5;

    float warp_intra_acc = 0.0f;

    for (int v = warp_id_global; v < num_vertices; v += total_warps) {
        int32_t c_v_tmp = 0, start_tmp = 0, end_tmp = 0;
        if (lane == 0) {
            c_v_tmp = (int32_t)__ldg(&cluster_assignments[v]);
            start_tmp = __ldg(&offsets[v]);
            end_tmp = __ldg(&offsets[v + 1]);
        }
        const int32_t c_v = __shfl_sync(0xffffffff, c_v_tmp, 0);
        const int start = __shfl_sync(0xffffffff, start_tmp, 0);
        const int end = __shfl_sync(0xffffffff, end_tmp, 0);

        float deg = 0.0f;
        float intra = 0.0f;

        int e = start + lane;
        
        for (; e + 96 < end; e += 128) {
            float w0 = __ldg(&edge_weights[e]);
            int32_t n0 = __ldg(&indices[e]);
            deg += w0;
            intra += ((int32_t)__ldg(&cluster_assignments[n0]) == c_v) ? w0 : 0.0f;

            int e1 = e + 32;
            float w1 = __ldg(&edge_weights[e1]);
            int32_t n1 = __ldg(&indices[e1]);
            deg += w1;
            intra += ((int32_t)__ldg(&cluster_assignments[n1]) == c_v) ? w1 : 0.0f;

            int e2 = e + 64;
            float w2 = __ldg(&edge_weights[e2]);
            int32_t n2 = __ldg(&indices[e2]);
            deg += w2;
            intra += ((int32_t)__ldg(&cluster_assignments[n2]) == c_v) ? w2 : 0.0f;

            int e3 = e + 96;
            float w3 = __ldg(&edge_weights[e3]);
            int32_t n3 = __ldg(&indices[e3]);
            deg += w3;
            intra += ((int32_t)__ldg(&cluster_assignments[n3]) == c_v) ? w3 : 0.0f;
        }
        for (; e < end; e += 32) {
            float w = __ldg(&edge_weights[e]);
            deg += w;
            int32_t nbr = __ldg(&indices[e]);
            intra += ((int32_t)__ldg(&cluster_assignments[nbr]) == c_v) ? w : 0.0f;
        }

        #pragma unroll
        for (int offset = 16; offset >= 1; offset >>= 1) {
            deg += __shfl_down_sync(0xffffffff, deg, offset);
            intra += __shfl_down_sync(0xffffffff, intra, offset);
        }

        if (lane == 0) {
            atomicAdd(&cluster_degrees[c_v], deg);
            warp_intra_acc += intra;
        }
    }

    __shared__ float s_warp_intra[WARPS_PER_BLOCK];
    const int warp_id_local = threadIdx.x >> 5;
    if (lane == 0) s_warp_intra[warp_id_local] = warp_intra_acc;
    __syncthreads();

    if (warp_id_local == 0) {
        float val = (lane < WARPS_PER_BLOCK) ? s_warp_intra[lane] : 0.0f;
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (lane == 0) atomicAdd(d_intra_sum, val);
    }
}






template <typename CType>
__global__ void modularity_warp16_smem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const CType* __restrict__ cluster_assignments,
    float* __restrict__ cluster_degrees,
    float* __restrict__ d_intra_sum,
    int32_t num_clusters,
    int32_t num_vertices)
{
    extern __shared__ float s_cluster_deg[];

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) s_cluster_deg[i] = 0.0f;
    __syncthreads();

    const int warp_id_global = (blockIdx.x * BLOCK_SIZE + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int group = lane >> 4;      
    const int lane16 = lane & 15;     
    const int total_warps = (gridDim.x * BLOCK_SIZE) >> 5;

    float warp_intra_acc = 0.0f;

    for (int base = warp_id_global * 2; base < num_vertices; base += total_warps * 2) {
        const int v = base + group;

        int32_t c_v_tmp = 0, start_tmp = 0, end_tmp = 0;
        if (lane16 == 0 && v < num_vertices) {
            c_v_tmp = (int32_t)__ldg(&cluster_assignments[v]);
            start_tmp = __ldg(&offsets[v]);
            end_tmp = __ldg(&offsets[v + 1]);
        }

        const int32_t c_v = __shfl_sync(0xffffffff, c_v_tmp, 0, 16);
        const int start = __shfl_sync(0xffffffff, start_tmp, 0, 16);
        const int end = __shfl_sync(0xffffffff, end_tmp, 0, 16);

        float deg = 0.0f;
        float intra = 0.0f;

        if (v < num_vertices) {
            int e = start + lane16;
            
            for (; e + 48 < end; e += 64) {
                float w0 = __ldg(&edge_weights[e]);
                int32_t n0 = __ldg(&indices[e]);
                deg += w0;
                intra += ((int32_t)__ldg(&cluster_assignments[n0]) == c_v) ? w0 : 0.0f;

                int e1 = e + 16;
                float w1 = __ldg(&edge_weights[e1]);
                int32_t n1 = __ldg(&indices[e1]);
                deg += w1;
                intra += ((int32_t)__ldg(&cluster_assignments[n1]) == c_v) ? w1 : 0.0f;

                int e2 = e + 32;
                float w2 = __ldg(&edge_weights[e2]);
                int32_t n2 = __ldg(&indices[e2]);
                deg += w2;
                intra += ((int32_t)__ldg(&cluster_assignments[n2]) == c_v) ? w2 : 0.0f;

                int e3 = e + 48;
                float w3 = __ldg(&edge_weights[e3]);
                int32_t n3 = __ldg(&indices[e3]);
                deg += w3;
                intra += ((int32_t)__ldg(&cluster_assignments[n3]) == c_v) ? w3 : 0.0f;
            }
            for (; e < end; e += 16) {
                float w = __ldg(&edge_weights[e]);
                int32_t nbr = __ldg(&indices[e]);
                deg += w;
                intra += ((int32_t)__ldg(&cluster_assignments[nbr]) == c_v) ? w : 0.0f;
            }
        }

        
        #pragma unroll
        for (int offset = 8; offset >= 1; offset >>= 1) {
            deg += __shfl_down_sync(0xffffffff, deg, offset, 16);
            intra += __shfl_down_sync(0xffffffff, intra, offset, 16);
        }

        float intra_leader = 0.0f;
        if (lane16 == 0 && v < num_vertices) {
            atomicAdd(&s_cluster_deg[c_v], deg);
            intra_leader = intra;
        }

        
        float intra_other = __shfl_sync(0xffffffff, intra_leader, 16);
        if (lane == 0) warp_intra_acc += intra_leader + intra_other;
    }

    
    __shared__ float s_warp_intra[WARPS_PER_BLOCK];
    const int warp_id_local = threadIdx.x >> 5;
    if (lane == 0) s_warp_intra[warp_id_local] = warp_intra_acc;
    __syncthreads();

    if (warp_id_local == 0) {
        float val = (lane < WARPS_PER_BLOCK) ? s_warp_intra[lane] : 0.0f;
        
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (lane == 0) atomicAdd(d_intra_sum, val);
    }

    __syncthreads();

    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) {
        float v = s_cluster_deg[i];
        if (v != 0.0f) atomicAdd(&cluster_degrees[i], v);
    }
}

template <typename CType>
__global__ void modularity_warp16_gmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const CType* __restrict__ cluster_assignments,
    float* __restrict__ cluster_degrees,
    float* __restrict__ d_intra_sum,
    int32_t num_vertices)
{
    const int warp_id_global = (blockIdx.x * BLOCK_SIZE + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int group = lane >> 4;
    const int lane16 = lane & 15;
    const int total_warps = (gridDim.x * BLOCK_SIZE) >> 5;

    float warp_intra_acc = 0.0f;

    for (int base = warp_id_global * 2; base < num_vertices; base += total_warps * 2) {
        const int v = base + group;

        int32_t c_v_tmp = 0, start_tmp = 0, end_tmp = 0;
        if (lane16 == 0 && v < num_vertices) {
            c_v_tmp = (int32_t)__ldg(&cluster_assignments[v]);
            start_tmp = __ldg(&offsets[v]);
            end_tmp = __ldg(&offsets[v + 1]);
        }

        const int32_t c_v = __shfl_sync(0xffffffff, c_v_tmp, 0, 16);
        const int start = __shfl_sync(0xffffffff, start_tmp, 0, 16);
        const int end = __shfl_sync(0xffffffff, end_tmp, 0, 16);

        float deg = 0.0f;
        float intra = 0.0f;

        if (v < num_vertices) {
            int e = start + lane16;
            for (; e + 48 < end; e += 64) {
                float w0 = __ldg(&edge_weights[e]);
                int32_t n0 = __ldg(&indices[e]);
                deg += w0;
                intra += ((int32_t)__ldg(&cluster_assignments[n0]) == c_v) ? w0 : 0.0f;

                int e1 = e + 16;
                float w1 = __ldg(&edge_weights[e1]);
                int32_t n1 = __ldg(&indices[e1]);
                deg += w1;
                intra += ((int32_t)__ldg(&cluster_assignments[n1]) == c_v) ? w1 : 0.0f;

                int e2 = e + 32;
                float w2 = __ldg(&edge_weights[e2]);
                int32_t n2 = __ldg(&indices[e2]);
                deg += w2;
                intra += ((int32_t)__ldg(&cluster_assignments[n2]) == c_v) ? w2 : 0.0f;

                int e3 = e + 48;
                float w3 = __ldg(&edge_weights[e3]);
                int32_t n3 = __ldg(&indices[e3]);
                deg += w3;
                intra += ((int32_t)__ldg(&cluster_assignments[n3]) == c_v) ? w3 : 0.0f;
            }
            for (; e < end; e += 16) {
                float w = __ldg(&edge_weights[e]);
                int32_t nbr = __ldg(&indices[e]);
                deg += w;
                intra += ((int32_t)__ldg(&cluster_assignments[nbr]) == c_v) ? w : 0.0f;
            }
        }

        #pragma unroll
        for (int offset = 8; offset >= 1; offset >>= 1) {
            deg += __shfl_down_sync(0xffffffff, deg, offset, 16);
            intra += __shfl_down_sync(0xffffffff, intra, offset, 16);
        }

        float intra_leader = 0.0f;
        if (lane16 == 0 && v < num_vertices) {
            atomicAdd(&cluster_degrees[c_v], deg);
            intra_leader = intra;
        }

        float intra_other = __shfl_sync(0xffffffff, intra_leader, 16);
        if (lane == 0) warp_intra_acc += intra_leader + intra_other;
    }

    __shared__ float s_warp_intra[WARPS_PER_BLOCK];
    const int warp_id_local = threadIdx.x >> 5;
    if (lane == 0) s_warp_intra[warp_id_local] = warp_intra_acc;
    __syncthreads();

    if (warp_id_local == 0) {
        float val = (lane < WARPS_PER_BLOCK) ? s_warp_intra[lane] : 0.0f;
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        if (lane == 0) atomicAdd(d_intra_sum, val);
    }
}






template <typename CType>
__global__ void modularity_subwarp8_gmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const CType* __restrict__ cluster_assignments,
    float* __restrict__ cluster_degrees,
    float* __restrict__ d_intra_sum,
    int32_t num_vertices)
{
    const int warp_id_global = (blockIdx.x * BLOCK_SIZE + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int group = lane >> 3;   
    const int lane8 = lane & 7;    
    const int total_warps = (gridDim.x * BLOCK_SIZE) >> 5;

    float warp_intra_acc = 0.0f;

    for (int base = warp_id_global * 4; base < num_vertices; base += total_warps * 4) {
        const int v = base + group;

        int32_t c_v_tmp = 0, start_tmp = 0, end_tmp = 0;
        if (lane8 == 0 && v < num_vertices) {
            c_v_tmp = (int32_t)__ldg(&cluster_assignments[v]);
            start_tmp = __ldg(&offsets[v]);
            end_tmp = __ldg(&offsets[v + 1]);
        }
        const int32_t c_v = __shfl_sync(0xffffffff, c_v_tmp, 0, 8);
        const int start = __shfl_sync(0xffffffff, start_tmp, 0, 8);
        const int end = __shfl_sync(0xffffffff, end_tmp, 0, 8);

        float deg = 0.0f;
        float intra = 0.0f;

        if (v < num_vertices) {
            for (int e = start + lane8; e < end; e += 8) {
                float w = __ldg(&edge_weights[e]);
                deg += w;
                int32_t nbr = __ldg(&indices[e]);
                intra += ((int32_t)__ldg(&cluster_assignments[nbr]) == c_v) ? w : 0.0f;
            }
        }

        
        deg += __shfl_down_sync(0xffffffff, deg, 4, 8);
        deg += __shfl_down_sync(0xffffffff, deg, 2, 8);
        deg += __shfl_down_sync(0xffffffff, deg, 1, 8);
        intra += __shfl_down_sync(0xffffffff, intra, 4, 8);
        intra += __shfl_down_sync(0xffffffff, intra, 2, 8);
        intra += __shfl_down_sync(0xffffffff, intra, 1, 8);

        float intra_leader = 0.0f;
        if (lane8 == 0 && v < num_vertices) {
            atomicAdd(&cluster_degrees[c_v], deg);
            intra_leader = intra;
        }

        
        float intra8  = __shfl_sync(0xffffffff, intra_leader, 8);
        float intra16 = __shfl_sync(0xffffffff, intra_leader, 16);
        float intra24 = __shfl_sync(0xffffffff, intra_leader, 24);
        if (lane == 0) {
            warp_intra_acc += intra_leader + intra8 + intra16 + intra24;
        }
    }

    
    if (lane == 0) atomicAdd(d_intra_sum, warp_intra_acc);
}





template <typename CType>
__global__ void modularity_thread_smem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const CType* __restrict__ cluster_assignments,
    float* __restrict__ cluster_degrees,
    float* __restrict__ d_intra_sum,
    int32_t num_clusters,
    int32_t num_vertices)
{
    extern __shared__ float s_cluster_deg[];
    for (int i = threadIdx.x; i < num_clusters; i += BLOCK_SIZE) s_cluster_deg[i] = 0.0f;
    __syncthreads();

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    float local_intra_acc = 0.0f;

    for (int v = blockIdx.x * BLOCK_SIZE + tid; v < num_vertices; v += gridDim.x * BLOCK_SIZE) {
        const int32_t c_v = (int32_t)__ldg(&cluster_assignments[v]);
        const int start = __ldg(&offsets[v]);
        const int end = __ldg(&offsets[v + 1]);

        float deg = 0.0f;
        float intra = 0.0f;
        int e = start;
        for (; e + 3 < end; e += 4) {
            float w0 = __ldg(&edge_weights[e]);
            int32_t n0 = __ldg(&indices[e]);
            deg += w0;
            intra += ((int32_t)__ldg(&cluster_assignments[n0]) == c_v) ? w0 : 0.0f;

            float w1 = __ldg(&edge_weights[e + 1]);
            int32_t n1 = __ldg(&indices[e + 1]);
            deg += w1;
            intra += ((int32_t)__ldg(&cluster_assignments[n1]) == c_v) ? w1 : 0.0f;

            float w2 = __ldg(&edge_weights[e + 2]);
            int32_t n2 = __ldg(&indices[e + 2]);
            deg += w2;
            intra += ((int32_t)__ldg(&cluster_assignments[n2]) == c_v) ? w2 : 0.0f;

            float w3 = __ldg(&edge_weights[e + 3]);
            int32_t n3 = __ldg(&indices[e + 3]);
            deg += w3;
            intra += ((int32_t)__ldg(&cluster_assignments[n3]) == c_v) ? w3 : 0.0f;
        }
        for (; e < end; ++e) {
            float w = __ldg(&edge_weights[e]);
            deg += w;
            int32_t nbr = __ldg(&indices[e]);
            intra += ((int32_t)__ldg(&cluster_assignments[nbr]) == c_v) ? w : 0.0f;
        }
        atomicAdd(&s_cluster_deg[c_v], deg);
        local_intra_acc += intra;
    }

    
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        local_intra_acc += __shfl_down_sync(0xffffffff, local_intra_acc, offset);
    }
    if (lane == 0) atomicAdd(d_intra_sum, local_intra_acc);

    __syncthreads();

    for (int i = tid; i < num_clusters; i += BLOCK_SIZE) {
        float v = s_cluster_deg[i];
        if (v != 0.0f) atomicAdd(&cluster_degrees[i], v);
    }
}

template <typename CType>
__global__ void modularity_thread_gmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const CType* __restrict__ cluster_assignments,
    float* __restrict__ cluster_degrees,
    float* __restrict__ d_intra_sum,
    int32_t num_vertices)
{
    const int tid = threadIdx.x;
    const int lane = tid & 31;

    float local_intra_acc = 0.0f;

    for (int v = blockIdx.x * BLOCK_SIZE + tid; v < num_vertices; v += gridDim.x * BLOCK_SIZE) {
        const int32_t c_v = (int32_t)__ldg(&cluster_assignments[v]);
        const int start = __ldg(&offsets[v]);
        const int end = __ldg(&offsets[v + 1]);

        float deg = 0.0f;
        float intra = 0.0f;
        int e = start;
        for (; e + 3 < end; e += 4) {
            float w0 = __ldg(&edge_weights[e]);
            int32_t n0 = __ldg(&indices[e]);
            deg += w0;
            intra += ((int32_t)__ldg(&cluster_assignments[n0]) == c_v) ? w0 : 0.0f;

            float w1 = __ldg(&edge_weights[e + 1]);
            int32_t n1 = __ldg(&indices[e + 1]);
            deg += w1;
            intra += ((int32_t)__ldg(&cluster_assignments[n1]) == c_v) ? w1 : 0.0f;

            float w2 = __ldg(&edge_weights[e + 2]);
            int32_t n2 = __ldg(&indices[e + 2]);
            deg += w2;
            intra += ((int32_t)__ldg(&cluster_assignments[n2]) == c_v) ? w2 : 0.0f;

            float w3 = __ldg(&edge_weights[e + 3]);
            int32_t n3 = __ldg(&indices[e + 3]);
            deg += w3;
            intra += ((int32_t)__ldg(&cluster_assignments[n3]) == c_v) ? w3 : 0.0f;
        }
        for (; e < end; ++e) {
            float w = __ldg(&edge_weights[e]);
            deg += w;
            int32_t nbr = __ldg(&indices[e]);
            intra += ((int32_t)__ldg(&cluster_assignments[nbr]) == c_v) ? w : 0.0f;
        }

        atomicAdd(&cluster_degrees[c_v], deg);
        local_intra_acc += intra;
    }

    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        local_intra_acc += __shfl_down_sync(0xffffffff, local_intra_acc, offset);
    }
    if (lane == 0) atomicAdd(d_intra_sum, local_intra_acc);
}





__global__ void finalize_kernel(
    const float* __restrict__ cluster_degrees,
    const float* __restrict__ d_intra_sum,
    double* __restrict__ d_result,
    int32_t num_clusters)
{
    __shared__ double s_total[WARPS_PER_BLOCK];
    __shared__ double s_sumsq[WARPS_PER_BLOCK];

    double local_total = 0.0;
    double local_sumsq = 0.0;

    for (int c = threadIdx.x; c < num_clusters; c += BLOCK_SIZE) {
        double d = (double)__ldg(&cluster_degrees[c]);
        local_total += d;
        local_sumsq += d * d;
    }

    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        local_total += __shfl_down_sync(0xffffffff, local_total, offset);
        local_sumsq += __shfl_down_sync(0xffffffff, local_sumsq, offset);
    }

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        s_total[warp_id] = local_total;
        s_sumsq[warp_id] = local_sumsq;
    }
    __syncthreads();

    if (warp_id == 0) {
        double t = (lane < WARPS_PER_BLOCK) ? s_total[lane] : 0.0;
        double ss = (lane < WARPS_PER_BLOCK) ? s_sumsq[lane] : 0.0;
        
        t += __shfl_down_sync(0xffffffff, t, 4);
        t += __shfl_down_sync(0xffffffff, t, 2);
        t += __shfl_down_sync(0xffffffff, t, 1);
        ss += __shfl_down_sync(0xffffffff, ss, 4);
        ss += __shfl_down_sync(0xffffffff, ss, 2);
        ss += __shfl_down_sync(0xffffffff, ss, 1);
        if (lane == 0) {
            double intra = (double)(*d_intra_sum);
            *d_result = intra / t - ss / (t * t);
        }
    }
}





template <typename CType>
static void launch_main(
    const int32_t* offsets,
    const int32_t* indices,
    const float* edge_weights,
    const CType* cluster_assignments,
    float* cluster_degrees,
    float* d_intra_sum,
    int32_t num_vertices,
    int32_t num_edges,
    int32_t num_clusters,
    cudaStream_t stream)
{
    const float avg_degree = (num_vertices > 0) ? (float)num_edges / (float)num_vertices : 0.0f;
    const bool use_smem = (num_clusters <= SMEM_CLUSTER_THRESHOLD);

    if (avg_degree >= 10.0f) {
        const bool use_subwarp8 = (avg_degree < 16.0f);
        const bool use_packed16 = false; (void)use_packed16;
        if (use_packed16) {
            
            int warps_needed = (num_vertices + 1) >> 1;
            int grid = (warps_needed + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
            if (grid > MAX_BLOCKS) grid = MAX_BLOCKS;
            if (grid < 1) grid = 1;
            if (use_smem) {
                const int smem_size = num_clusters * (int)sizeof(float);
                modularity_warp16_smem<CType><<<grid, BLOCK_SIZE, smem_size, stream>>>(
                    offsets, indices, edge_weights, cluster_assignments,
                    cluster_degrees, d_intra_sum, num_clusters, num_vertices);
            } else {
                modularity_warp16_gmem<CType><<<grid, BLOCK_SIZE, 0, stream>>>(
                    offsets, indices, edge_weights, cluster_assignments,
                    cluster_degrees, d_intra_sum, num_vertices);
            }
        } else {
            if (use_subwarp8) {
            int grid = (num_vertices + (4 * WARPS_PER_BLOCK) - 1) / (4 * WARPS_PER_BLOCK);
            if (grid > MAX_BLOCKS) grid = MAX_BLOCKS;
            if (grid < 1) grid = 1;
            modularity_subwarp8_gmem<CType><<<grid, BLOCK_SIZE, 0, stream>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cluster_degrees, d_intra_sum, num_vertices);
        } else {
        
            int grid = (num_vertices + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        if (grid > MAX_BLOCKS) grid = MAX_BLOCKS;
        if (grid < 1) grid = 1;

        if (use_smem) {
            const int smem_size = num_clusters * (int)sizeof(float);
            modularity_warp_smem<CType><<<grid, BLOCK_SIZE, smem_size, stream>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cluster_degrees, d_intra_sum, num_clusters, num_vertices);
        } else {
            modularity_warp_gmem<CType><<<grid, BLOCK_SIZE, 0, stream>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cluster_degrees, d_intra_sum, num_vertices);
        }
        }
        }
    } else {
        
        int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid > MAX_BLOCKS) grid = MAX_BLOCKS;
        if (grid < 1) grid = 1;

        if (use_smem) {
            const int smem_size = num_clusters * (int)sizeof(float);
            modularity_thread_smem<CType><<<grid, BLOCK_SIZE, smem_size, stream>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cluster_degrees, d_intra_sum, num_clusters, num_vertices);
        } else {
            modularity_thread_gmem<CType><<<grid, BLOCK_SIZE, 0, stream>>>(
                offsets, indices, edge_weights, cluster_assignments,
                cluster_degrees, d_intra_sum, num_vertices);
        }
    }
}

}  

double analyze_clustering_modularity(const graph32_t& graph,
                                     const float* edge_weights,
                                     std::size_t num_clusters,
                                     const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t num_vertices = graph.number_of_vertices;
    const int32_t num_edges = graph.number_of_edges;
    const int32_t nc = static_cast<int32_t>(num_clusters);

    cache.ensure_cluster_degrees(num_clusters);

    
    void* compressed_ptr = nullptr;
    if (cache.d_compressed) {
        if (nc <= 255) {
            if (static_cast<std::size_t>(num_vertices) * sizeof(uint8_t) <= cache.compressed_capacity_bytes) {
                compressed_ptr = cache.d_compressed;
            }
        } else if (nc <= 32767) {
            if (static_cast<std::size_t>(num_vertices) * sizeof(int16_t) <= cache.compressed_capacity_bytes) {
                compressed_ptr = cache.d_compressed;
            }
        }
    }

    cudaStream_t stream = 0;

    
    cudaMemsetAsync(cache.d_cluster_degrees, 0, static_cast<std::size_t>(nc) * sizeof(float), stream);
    cudaMemsetAsync(cache.d_intra_sum, 0, sizeof(float), stream);

    if (nc <= 255 && compressed_ptr) {
        int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid > MAX_BLOCKS) grid = MAX_BLOCKS;
        if (grid < 1) grid = 1;
        compress_to_u8<<<grid, BLOCK_SIZE, 0, stream>>>(
            cluster_assignments, static_cast<uint8_t*>(compressed_ptr), num_vertices);
        launch_main<uint8_t>(
            graph.offsets, graph.indices, edge_weights,
            static_cast<const uint8_t*>(compressed_ptr),
            cache.d_cluster_degrees, cache.d_intra_sum,
            num_vertices, num_edges, nc, stream);
    } else if (nc <= 32767 && compressed_ptr) {
        int grid = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (grid > MAX_BLOCKS) grid = MAX_BLOCKS;
        if (grid < 1) grid = 1;
        compress_to_i16<<<grid, BLOCK_SIZE, 0, stream>>>(
            cluster_assignments, static_cast<int16_t*>(compressed_ptr), num_vertices);
        launch_main<int16_t>(
            graph.offsets, graph.indices, edge_weights,
            static_cast<const int16_t*>(compressed_ptr),
            cache.d_cluster_degrees, cache.d_intra_sum,
            num_vertices, num_edges, nc, stream);
    } else {
        launch_main<int32_t>(
            graph.offsets, graph.indices, edge_weights,
            cluster_assignments,
            cache.d_cluster_degrees, cache.d_intra_sum,
            num_vertices, num_edges, nc, stream);
    }

    finalize_kernel<<<1, BLOCK_SIZE, 0, stream>>>(
        cache.d_cluster_degrees, cache.d_intra_sum, cache.d_result, nc);

    double h_result;
    cudaMemcpy(&h_result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);

    return h_result;
}

}  
