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
    double* sigma = nullptr;
    double* intra = nullptr;
    double* d_result = nullptr;
    int64_t sigma_capacity = 0;
    int64_t intra_capacity = 0;

    Cache() {
        cudaMalloc(&d_result, sizeof(double));
    }

    ~Cache() override {
        if (sigma) cudaFree(sigma);
        if (intra) cudaFree(intra);
        if (d_result) cudaFree(d_result);
    }

    void ensure(int32_t num_clusters) {
        if (sigma_capacity < num_clusters) {
            if (sigma) cudaFree(sigma);
            cudaMalloc(&sigma, (size_t)num_clusters * sizeof(double));
            sigma_capacity = num_clusters;
        }
        if (intra_capacity < num_clusters) {
            if (intra) cudaFree(intra);
            cudaMalloc(&intra, (size_t)num_clusters * sizeof(double));
            intra_capacity = num_clusters;
        }
    }
};

static __device__ __forceinline__ double warp_reduce_sum(double v)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

static __device__ __forceinline__ float warp_reduce_sum_f(float v)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}



__global__ void k2_thread_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ w,
    const int32_t* __restrict__ cluster,
    double* __restrict__ sigma,  
    double* __restrict__ intra,  
    int32_t n)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(gridDim.x * blockDim.x);

    double sigma0 = 0.0, sigma1 = 0.0, intra_sum = 0.0;

    for (int v = tid; v < n; v += stride) {
        int32_t cu = cluster[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float deg = 0.f;
        float in = 0.f;
        for (int32_t e = start; e < end; ++e) {
            float we = (float)w[e];
            deg += we;
            int32_t dst = indices[e];
            if (cluster[dst] == cu) in += we;
        }

        if (cu == 0) sigma0 += (double)deg;
        else sigma1 += (double)deg;
        intra_sum += (double)in;
    }

    sigma0 = warp_reduce_sum(sigma0);
    sigma1 = warp_reduce_sum(sigma1);
    double warp_intra = warp_reduce_sum(intra_sum);

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    __shared__ double sbuf[3 * 32];
    if (lane == 0) {
        sbuf[warp] = sigma0;
        sbuf[warps_per_block + warp] = sigma1;
        sbuf[2 * warps_per_block + warp] = warp_intra;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double s0 = 0.0, s1 = 0.0, it = 0.0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i < warps_per_block) {
                s0 += sbuf[i];
                s1 += sbuf[warps_per_block + i];
                it += sbuf[2 * warps_per_block + i];
            }
        }
        if (s0) atomicAdd(&sigma[0], s0);
        if (s1) atomicAdd(&sigma[1], s1);
        if (it) atomicAdd(&intra[0], it);
    }
}

__global__ void k2_warp_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ w,
    const int32_t* __restrict__ cluster,
    double* __restrict__ sigma,
    double* __restrict__ intra,
    int32_t n)
{
    int global_tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int warp_global = global_tid >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;

    double sigma0 = 0.0, sigma1 = 0.0, intra_sum = 0.0;

    for (int v = warp_global; v < n; v += total_warps) {
        int32_t cu = cluster[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float deg = 0.f;
        float in = 0.f;
        for (int32_t e = start + lane; e < end; e += 64) {
            float we0 = (float)w[e];
            deg += we0;
            int32_t dst0 = indices[e];
            if (cluster[dst0] == cu) in += we0;

            int32_t e1 = e + 32;
            if (e1 < end) {
                float we1 = (float)w[e1];
                deg += we1;
                int32_t dst1 = indices[e1];
                if (cluster[dst1] == cu) in += we1;
            }
        }

        deg = warp_reduce_sum_f(deg);
        in = warp_reduce_sum_f(in);

        if (lane == 0) {
            if (cu == 0) sigma0 += (double)deg;
            else sigma1 += (double)deg;
            intra_sum += (double)in;
        }
    }

    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    __shared__ double sbuf[3 * 32];
    if (lane == 0) {
        sbuf[warp_in_block] = sigma0;
        sbuf[warps_per_block + warp_in_block] = sigma1;
        sbuf[2 * warps_per_block + warp_in_block] = intra_sum;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double s0 = 0.0, s1 = 0.0, it = 0.0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i < warps_per_block) {
                s0 += sbuf[i];
                s1 += sbuf[warps_per_block + i];
                it += sbuf[2 * warps_per_block + i];
            }
        }
        if (s0) atomicAdd(&sigma[0], s0);
        if (s1) atomicAdd(&sigma[1], s1);
        if (it) atomicAdd(&intra[0], it);
    }
}



__global__ void cluster_sums_thread_shmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ w,
    const int32_t* __restrict__ cluster,
    double* __restrict__ sigma,
    double* __restrict__ intra,
    int32_t n,
    int32_t K)
{
    extern __shared__ double sh[];
    double* s_sigma = sh;
    double* s_intra = sh + K;

    for (int c = (int)threadIdx.x; c < K; c += (int)blockDim.x) {
        s_sigma[c] = 0.0;
        s_intra[c] = 0.0;
    }
    __syncthreads();

    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(gridDim.x * blockDim.x);

    for (int v = tid; v < n; v += stride) {
        int32_t cu = cluster[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float deg = 0.f;
        float in = 0.f;
        for (int32_t e = start; e < end; ++e) {
            float we = (float)w[e];
            deg += we;
            int32_t dst = indices[e];
            if (cluster[dst] == cu) in += we;
        }

        if (deg) atomicAdd(&s_sigma[cu], (double)deg);
        if (in) atomicAdd(&s_intra[cu], (double)in);
    }

    __syncthreads();

    for (int c = (int)threadIdx.x; c < K; c += (int)blockDim.x) {
        double sv = s_sigma[c];
        double iv = s_intra[c];
        if (sv) atomicAdd(&sigma[c], sv);
        if (iv) atomicAdd(&intra[c], iv);
    }
}


__global__ void cluster_sums_warp_shmem_warp_private(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ w,
    const int32_t* __restrict__ cluster,
    double* __restrict__ sigma,
    double* __restrict__ intra,
    int32_t n,
    int32_t K)
{
    extern __shared__ double sh[];
    int warps_per_block = (int)(blockDim.x >> 5);
    int warp_in_block = (int)(threadIdx.x >> 5);
    int lane = (int)(threadIdx.x & 31);

    double* s_sigma = sh;
    double* s_intra = sh + warps_per_block * K;

    for (int i = (int)threadIdx.x; i < 2 * warps_per_block * K; i += (int)blockDim.x) {
        sh[i] = 0.0;
    }
    __syncthreads();

    double* my_sigma = s_sigma + warp_in_block * K;
    double* my_intra = s_intra + warp_in_block * K;

    int global_tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int warp_global = global_tid >> 5;
    int total_warps = (gridDim.x * blockDim.x) >> 5;

    for (int v = warp_global; v < n; v += total_warps) {
        int32_t cu = cluster[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float deg = 0.f;
        float in = 0.f;
        for (int32_t e = start + lane; e < end; e += 64) {
            float we0 = (float)w[e];
            deg += we0;
            int32_t dst0 = indices[e];
            if (cluster[dst0] == cu) in += we0;

            int32_t e1 = e + 32;
            if (e1 < end) {
                float we1 = (float)w[e1];
                deg += we1;
                int32_t dst1 = indices[e1];
                if (cluster[dst1] == cu) in += we1;
            }
        }

        deg = warp_reduce_sum_f(deg);
        in = warp_reduce_sum_f(in);
        if (lane == 0) {
            my_sigma[cu] += (double)deg;
            my_intra[cu] += (double)in;
        }
    }

    __syncthreads();

    for (int c = (int)threadIdx.x; c < K; c += (int)blockDim.x) {
        double sum_s = 0.0;
        double sum_i = 0.0;
        #pragma unroll
        for (int widx = 0; widx < 8; ++widx) {
            if (widx < warps_per_block) {
                sum_s += s_sigma[widx * K + c];
                sum_i += s_intra[widx * K + c];
            }
        }
        if (sum_s) atomicAdd(&sigma[c], sum_s);
        if (sum_i) atomicAdd(&intra[c], sum_i);
    }
}

__global__ void cluster_sums_warp_shmem(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ w,
    const int32_t* __restrict__ cluster,
    double* __restrict__ sigma,
    double* __restrict__ intra,
    int32_t n,
    int32_t K)
{
    extern __shared__ double sh[];
    double* s_sigma = sh;
    double* s_intra = sh + K;

    for (int c = (int)threadIdx.x; c < K; c += (int)blockDim.x) {
        s_sigma[c] = 0.0;
        s_intra[c] = 0.0;
    }
    __syncthreads();

    int global_tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int warp_global = global_tid >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;

    for (int v = warp_global; v < n; v += total_warps) {
        int32_t cu = cluster[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float deg = 0.f;
        float in = 0.f;
        for (int32_t e = start + lane; e < end; e += 64) {
            float we0 = (float)w[e];
            deg += we0;
            int32_t dst0 = indices[e];
            if (cluster[dst0] == cu) in += we0;

            int32_t e1 = e + 32;
            if (e1 < end) {
                float we1 = (float)w[e1];
                deg += we1;
                int32_t dst1 = indices[e1];
                if (cluster[dst1] == cu) in += we1;
            }
        }

        deg = warp_reduce_sum_f(deg);
        in = warp_reduce_sum_f(in);
        if (lane == 0) {
            if (deg) atomicAdd(&s_sigma[cu], (double)deg);
            if (in) atomicAdd(&s_intra[cu], (double)in);
        }
    }

    __syncthreads();

    for (int c = (int)threadIdx.x; c < K; c += (int)blockDim.x) {
        double sv = s_sigma[c];
        double iv = s_intra[c];
        if (sv) atomicAdd(&sigma[c], sv);
        if (iv) atomicAdd(&intra[c], iv);
    }
}

__global__ void cluster_sums_thread_global(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ w,
    const int32_t* __restrict__ cluster,
    double* __restrict__ sigma,
    double* __restrict__ intra,
    int32_t n)
{
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int stride = (int)(gridDim.x * blockDim.x);

    for (int v = tid; v < n; v += stride) {
        int32_t cu = cluster[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float deg = 0.f;
        float in = 0.f;
        for (int32_t e = start; e < end; ++e) {
            float we = (float)w[e];
            deg += we;
            int32_t dst = indices[e];
            if (cluster[dst] == cu) in += we;
        }

        if (deg) atomicAdd(&sigma[cu], (double)deg);
        if (in) atomicAdd(&intra[cu], (double)in);
    }
}

__global__ void cluster_sums_warp_global(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ w,
    const int32_t* __restrict__ cluster,
    double* __restrict__ sigma,
    double* __restrict__ intra,
    int32_t n)
{
    int global_tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int warp_global = global_tid >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;

    for (int v = warp_global; v < n; v += total_warps) {
        int32_t cu = cluster[v];
        int32_t start = offsets[v];
        int32_t end = offsets[v + 1];

        float deg = 0.f;
        float in = 0.f;
        for (int32_t e = start + lane; e < end; e += 64) {
            float we0 = (float)w[e];
            deg += we0;
            int32_t dst0 = indices[e];
            if (cluster[dst0] == cu) in += we0;

            int32_t e1 = e + 32;
            if (e1 < end) {
                float we1 = (float)w[e1];
                deg += we1;
                int32_t dst1 = indices[e1];
                if (cluster[dst1] == cu) in += we1;
            }
        }

        deg = warp_reduce_sum_f(deg);
        in = warp_reduce_sum_f(in);
        if (lane == 0) {
            if (deg) atomicAdd(&sigma[cu], (double)deg);
            if (in) atomicAdd(&intra[cu], (double)in);
        }
    }
}

__global__ void finalize_modularity_kernel(
    const double* __restrict__ sigma,
    const double* __restrict__ intra,
    double* __restrict__ out,
    int32_t K)
{
    double local_total = 0.0;
    double local_intra = 0.0;
    double local_sigma_sq = 0.0;

    for (int i = (int)threadIdx.x; i < K; i += (int)blockDim.x) {
        double s = sigma[i];
        local_total += s;
        local_intra += intra[i];
        local_sigma_sq += s * s;
    }

    local_total = warp_reduce_sum(local_total);
    local_intra = warp_reduce_sum(local_intra);
    local_sigma_sq = warp_reduce_sum(local_sigma_sq);

    __shared__ double sm_tot[32];
    __shared__ double sm_intra[32];
    __shared__ double sm_sq[32];

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sm_tot[warp] = local_total;
        sm_intra[warp] = local_intra;
        sm_sq[warp] = local_sigma_sq;
    }
    __syncthreads();

    if (warp == 0) {
        double t = (lane < ((blockDim.x + 31) >> 5)) ? sm_tot[lane] : 0.0;
        double i = (lane < ((blockDim.x + 31) >> 5)) ? sm_intra[lane] : 0.0;
        double s2 = (lane < ((blockDim.x + 31) >> 5)) ? sm_sq[lane] : 0.0;

        t = warp_reduce_sum(t);
        i = warp_reduce_sum(i);
        s2 = warp_reduce_sum(s2);

        if (lane == 0) {
            if (t > 0.0) out[0] = (i / t) - (s2 / (t * t));
            else out[0] = 0.0;
        }
    }
}



static inline int sm_count()
{
    int sm = 0;
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    return sm;
}

void launch_cluster_sums_k2_thread(
    const int32_t* offsets,
    const int32_t* indices,
    const double* edge_weights,
    const int32_t* cluster_assignments,
    double* sigma,
    double* intra,
    int32_t num_vertices)
{
    int threads = 256;
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, k2_thread_kernel, threads, 0);
    int blocks = blocks_per_sm * sm_count() * 2;
    if (blocks < 1) blocks = 1;
    k2_thread_kernel<<<blocks, threads>>>(
        offsets, indices, edge_weights, cluster_assignments, sigma, intra, num_vertices);
}

void launch_cluster_sums_k2_warp(
    const int32_t* offsets,
    const int32_t* indices,
    const double* edge_weights,
    const int32_t* cluster_assignments,
    double* sigma,
    double* intra,
    int32_t num_vertices)
{
    int threads = 256;
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, k2_warp_kernel, threads, 0);
    int blocks = blocks_per_sm * sm_count() * 2;
    if (blocks < 1) blocks = 1;
    k2_warp_kernel<<<blocks, threads>>>(
        offsets, indices, edge_weights, cluster_assignments, sigma, intra, num_vertices);
}

void launch_cluster_sums_shmem_thread(
    const int32_t* offsets,
    const int32_t* indices,
    const double* edge_weights,
    const int32_t* cluster_assignments,
    double* sigma,
    double* intra,
    int32_t num_vertices,
    int32_t num_clusters)
{
    int threads = 256;
    size_t shmem = (size_t)num_clusters * 2 * sizeof(double);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, cluster_sums_thread_shmem, threads, (int)shmem);
    int blocks = blocks_per_sm * sm_count() * 2;
    if (blocks < 1) blocks = 1;
    cluster_sums_thread_shmem<<<blocks, threads, shmem>>>(
        offsets, indices, edge_weights, cluster_assignments, sigma, intra, num_vertices, num_clusters);
}

void launch_cluster_sums_shmem_warp(
    const int32_t* offsets,
    const int32_t* indices,
    const double* edge_weights,
    const int32_t* cluster_assignments,
    double* sigma,
    double* intra,
    int32_t num_vertices,
    int32_t num_clusters)
{
    int threads = 256;
    int warps_per_block = threads >> 5;
    int blocks_per_sm = 0;

    if (num_clusters <= 256) {
        size_t shmem = (size_t)warps_per_block * (size_t)num_clusters * 2 * sizeof(double);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, cluster_sums_warp_shmem_warp_private, threads, (int)shmem);
        int blocks = blocks_per_sm * sm_count() * 2;
        if (blocks < 1) blocks = 1;
        cluster_sums_warp_shmem_warp_private<<<blocks, threads, shmem>>>(
            offsets, indices, edge_weights, cluster_assignments, sigma, intra, num_vertices, num_clusters);
    } else {
        size_t shmem = (size_t)num_clusters * 2 * sizeof(double);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, cluster_sums_warp_shmem, threads, (int)shmem);
        int blocks = blocks_per_sm * sm_count() * 2;
        if (blocks < 1) blocks = 1;
        cluster_sums_warp_shmem<<<blocks, threads, shmem>>>(
            offsets, indices, edge_weights, cluster_assignments, sigma, intra, num_vertices, num_clusters);
    }
}

void launch_cluster_sums_global_thread(
    const int32_t* offsets,
    const int32_t* indices,
    const double* edge_weights,
    const int32_t* cluster_assignments,
    double* sigma,
    double* intra,
    int32_t num_vertices)
{
    int threads = 256;
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, cluster_sums_thread_global, threads, 0);
    int blocks = blocks_per_sm * sm_count() * 2;
    if (blocks < 1) blocks = 1;
    cluster_sums_thread_global<<<blocks, threads>>>(
        offsets, indices, edge_weights, cluster_assignments, sigma, intra, num_vertices);
}

void launch_cluster_sums_global_warp(
    const int32_t* offsets,
    const int32_t* indices,
    const double* edge_weights,
    const int32_t* cluster_assignments,
    double* sigma,
    double* intra,
    int32_t num_vertices)
{
    int threads = 256;
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, cluster_sums_warp_global, threads, 0);
    int blocks = blocks_per_sm * sm_count() * 2;
    if (blocks < 1) blocks = 1;
    cluster_sums_warp_global<<<blocks, threads>>>(
        offsets, indices, edge_weights, cluster_assignments, sigma, intra, num_vertices);
}

void launch_finalize_modularity(
    const double* sigma,
    const double* intra,
    double* result,
    int32_t num_clusters)
{
    finalize_modularity_kernel<<<1, 256>>>(sigma, intra, result, num_clusters);
}

}  

double analyze_clustering_modularity(const graph32_t& graph,
                                     const double* edge_weights,
                                     std::size_t num_clusters,
                                     const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    int32_t nc = static_cast<int32_t>(num_clusters);

    cache.ensure(nc);

    cudaMemsetAsync(cache.sigma, 0, sizeof(double) * nc);
    cudaMemsetAsync(cache.intra, 0, sizeof(double) * nc);

    double avg_degree = (num_vertices > 0) ? (double)num_edges / (double)num_vertices : 0.0;

    if (nc == 2) {
        if (avg_degree >= 16.0) {
            launch_cluster_sums_k2_warp(
                offsets, indices, edge_weights, cluster_assignments,
                cache.sigma, cache.intra, num_vertices);
        } else {
            launch_cluster_sums_k2_thread(
                offsets, indices, edge_weights, cluster_assignments,
                cache.sigma, cache.intra, num_vertices);
        }
    } else if (nc <= 2048) {
        if (avg_degree >= 16.0) {
            launch_cluster_sums_shmem_warp(
                offsets, indices, edge_weights, cluster_assignments,
                cache.sigma, cache.intra, num_vertices, nc);
        } else {
            launch_cluster_sums_shmem_thread(
                offsets, indices, edge_weights, cluster_assignments,
                cache.sigma, cache.intra, num_vertices, nc);
        }
    } else {
        if (avg_degree >= 32.0) {
            launch_cluster_sums_global_warp(
                offsets, indices, edge_weights, cluster_assignments,
                cache.sigma, cache.intra, num_vertices);
        } else {
            launch_cluster_sums_global_thread(
                offsets, indices, edge_weights, cluster_assignments,
                cache.sigma, cache.intra, num_vertices);
        }
    }

    launch_finalize_modularity(cache.sigma, cache.intra, cache.d_result, nc);

    double result;
    cudaMemcpy(&result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);

    return result;
}

}  
