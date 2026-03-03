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
#include <algorithm>

namespace aai {

namespace {

static constexpr int INIT_CLUSTERS = 256;

struct Cache : Cacheable {
    double* d_sigma = nullptr;
    double* d_intra_sum = nullptr;
    double* d_result = nullptr;
    int32_t sigma_capacity = 0;

    Cache() {
        cudaMalloc(&d_sigma, INIT_CLUSTERS * sizeof(double));
        sigma_capacity = INIT_CLUSTERS;
        cudaMalloc(&d_intra_sum, sizeof(double));
        cudaMalloc(&d_result, sizeof(double));
    }

    void ensure_sigma_capacity(int32_t num_clusters) {
        if (num_clusters > sigma_capacity) {
            if (d_sigma) cudaFree(d_sigma);
            cudaMalloc(&d_sigma, num_clusters * sizeof(double));
            sigma_capacity = num_clusters;
        }
    }

    ~Cache() override {
        if (d_sigma) cudaFree(d_sigma);
        if (d_intra_sum) cudaFree(d_intra_sum);
        if (d_result) cudaFree(d_result);
    }
};

__device__ __forceinline__ double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}




__global__ void __launch_bounds__(256, 4) modularity_k2_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ d_sigma,
    double* __restrict__ d_intra_sum,
    int32_t num_vertices)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    double sigma0 = 0.0, sigma1 = 0.0, local_intra = 0.0;

    for (int v = tid; v < num_vertices; v += stride) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int my_cluster = cluster_assignments[v];
        double degree_sum = 0.0, intra = 0.0;

        
        int e = start;
        for (; e + 3 < end; e += 4) {
            double w0 = __ldcs(&edge_weights[e]);
            double w1 = __ldcs(&edge_weights[e+1]);
            double w2 = __ldcs(&edge_weights[e+2]);
            double w3 = __ldcs(&edge_weights[e+3]);

            int j0 = __ldcs(&indices[e]);
            int j1 = __ldcs(&indices[e+1]);
            int j2 = __ldcs(&indices[e+2]);
            int j3 = __ldcs(&indices[e+3]);

            degree_sum = fma(1.0, w0, degree_sum);
            degree_sum = fma(1.0, w1, degree_sum);
            degree_sum = fma(1.0, w2, degree_sum);
            degree_sum = fma(1.0, w3, degree_sum);

            if (cluster_assignments[j0] == my_cluster) intra += w0;
            if (cluster_assignments[j1] == my_cluster) intra += w1;
            if (cluster_assignments[j2] == my_cluster) intra += w2;
            if (cluster_assignments[j3] == my_cluster) intra += w3;
        }

        
        for (; e < end; e++) {
            double w = __ldcs(&edge_weights[e]);
            int j = __ldcs(&indices[e]);
            degree_sum += w;
            if (cluster_assignments[j] == my_cluster) intra += w;
        }

        sigma0 += (my_cluster == 0) ? degree_sum : 0.0;
        sigma1 += (my_cluster != 0) ? degree_sum : 0.0;
        local_intra += intra;
    }

    sigma0 = warp_reduce_sum(sigma0);
    sigma1 = warp_reduce_sum(sigma1);
    double warp_intra = warp_reduce_sum(local_intra);

    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int nwarps = blockDim.x >> 5;

    __shared__ double s_buf[3 * 32];
    if (lane == 0) {
        s_buf[warp_id] = sigma0;
        s_buf[nwarps + warp_id] = sigma1;
        s_buf[2 * nwarps + warp_id] = warp_intra;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double s0 = 0, s1 = 0, it = 0;
        for (int i = 0; i < nwarps; i++) {
            s0 += s_buf[i]; s1 += s_buf[nwarps + i]; it += s_buf[2*nwarps + i];
        }
        atomicAdd(&d_sigma[0], s0);
        atomicAdd(&d_sigma[1], s1);
        atomicAdd(d_intra_sum, it);
    }
}


__global__ void __launch_bounds__(256, 4) modularity_k2_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ d_sigma,
    double* __restrict__ d_intra_sum,
    int32_t num_vertices)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    double sigma0 = 0.0, sigma1 = 0.0, local_intra = 0.0;

    for (int v = warp_id; v < num_vertices; v += total_warps) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int my_cluster = cluster_assignments[v];
        double my_degree = 0.0, my_intra = 0.0;

        int e = start + lane;
        for (; e < end; e += 32) {
            double w = __ldcs(&edge_weights[e]);
            int j = __ldcs(&indices[e]);
            my_degree += w;
            if (cluster_assignments[j] == my_cluster) my_intra += w;
        }

        my_degree = warp_reduce_sum(my_degree);
        my_intra = warp_reduce_sum(my_intra);

        if (lane == 0) {
            sigma0 += (my_cluster == 0) ? my_degree : 0.0;
            sigma1 += (my_cluster != 0) ? my_degree : 0.0;
            local_intra += my_intra;
        }
    }

    extern __shared__ double s_buf2[];
    if (lane == 0) {
        s_buf2[warp_in_block] = sigma0;
        s_buf2[warps_per_block + warp_in_block] = sigma1;
        s_buf2[2*warps_per_block + warp_in_block] = local_intra;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double s0 = 0, s1 = 0, it = 0;
        for (int i = 0; i < warps_per_block; i++) {
            s0 += s_buf2[i]; s1 += s_buf2[warps_per_block+i]; it += s_buf2[2*warps_per_block+i];
        }
        atomicAdd(&d_sigma[0], s0);
        atomicAdd(&d_sigma[1], s1);
        atomicAdd(d_intra_sum, it);
    }
}




__global__ void __launch_bounds__(256, 4) modularity_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ d_sigma,
    double* __restrict__ d_intra_sum,
    int32_t num_vertices,
    int32_t num_clusters)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    double local_intra = 0.0;

    for (int v = tid; v < num_vertices; v += stride) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int my_cluster = cluster_assignments[v];
        if ((unsigned)my_cluster >= (unsigned)num_clusters) my_cluster = (my_cluster < 0) ? 0 : (num_clusters - 1);
        double degree_sum = 0.0, intra = 0.0;

        
        int e = start;
        for (; e + 3 < end; e += 4) {
            double w0 = __ldcs(&edge_weights[e]);
            double w1 = __ldcs(&edge_weights[e+1]);
            double w2 = __ldcs(&edge_weights[e+2]);
            double w3 = __ldcs(&edge_weights[e+3]);

            int j0 = __ldcs(&indices[e]);
            int j1 = __ldcs(&indices[e+1]);
            int j2 = __ldcs(&indices[e+2]);
            int j3 = __ldcs(&indices[e+3]);

            degree_sum = fma(1.0, w0, degree_sum);
            degree_sum = fma(1.0, w1, degree_sum);
            degree_sum = fma(1.0, w2, degree_sum);
            degree_sum = fma(1.0, w3, degree_sum);

            if (cluster_assignments[j0] == my_cluster) intra += w0;
            if (cluster_assignments[j1] == my_cluster) intra += w1;
            if (cluster_assignments[j2] == my_cluster) intra += w2;
            if (cluster_assignments[j3] == my_cluster) intra += w3;
        }

        for (; e < end; e++) {
            double w = __ldcs(&edge_weights[e]);
            int j = __ldcs(&indices[e]);
            degree_sum += w;
            if (cluster_assignments[j] == my_cluster) intra += w;
        }

        atomicAdd(&d_sigma[my_cluster], degree_sum);
        local_intra += intra;
    }

    double warp_sum = warp_reduce_sum(local_intra);
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    __shared__ double s_partial[32];
    if (lane == 0) s_partial[warp_id] = warp_sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        double sum = 0;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) sum += s_partial[i];
        atomicAdd(d_intra_sum, sum);
    }
}


__global__ void __launch_bounds__(256, 4) modularity_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int32_t* __restrict__ cluster_assignments,
    double* __restrict__ d_sigma,
    double* __restrict__ d_intra_sum,
    int32_t num_vertices,
    int32_t num_clusters)
{
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_tid >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (gridDim.x * blockDim.x) >> 5;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;

    extern __shared__ double s_intra[];
    double local_intra = 0.0;

    for (int v = warp_id; v < num_vertices; v += total_warps) {
        int start = offsets[v];
        int end = offsets[v + 1];
        int my_cluster = cluster_assignments[v];
        if ((unsigned)my_cluster >= (unsigned)num_clusters) my_cluster = (my_cluster < 0) ? 0 : (num_clusters - 1);
        double my_degree = 0.0, my_intra = 0.0;

        int e = start + lane;
        for (; e < end; e += 32) {
            double w = __ldcs(&edge_weights[e]);
            int j = __ldcs(&indices[e]);
            my_degree += w;
            if (cluster_assignments[j] == my_cluster) my_intra += w;
        }

        my_degree = warp_reduce_sum(my_degree);
        my_intra = warp_reduce_sum(my_intra);

        if (lane == 0) {
            atomicAdd(&d_sigma[my_cluster], my_degree);
            local_intra += my_intra;
        }
    }

    if (lane == 0) s_intra[warp_in_block] = local_intra;
    __syncthreads();
    if (threadIdx.x == 0) {
        double sum = 0;
        for (int i = 0; i < warps_per_block; i++) sum += s_intra[i];
        atomicAdd(d_intra_sum, sum);
    }
}

__global__ void __launch_bounds__(1, 1) finalize_modularity(
    const double* __restrict__ d_sigma,
    const double* __restrict__ d_intra_sum,
    double* __restrict__ d_result,
    int32_t max_clusters)
{
    double intra_sum = __ldg(d_intra_sum);
    double total_weight = 0.0, sum_sigma_sq = 0.0;
    for (int c = 0; c < max_clusters; c++) {
        double s = __ldg(&d_sigma[c]);
        total_weight += s;
        sum_sigma_sq = fma(s, s, sum_sigma_sq);
    }
    *d_result = (total_weight > 0.0) ?
        intra_sum / total_weight - sum_sigma_sq / (total_weight * total_weight) : 0.0;
}

}  

double analyze_clustering_modularity(const graph32_t& graph,
                                     const double* edge_weights,
                                     std::size_t num_clusters,
                                     const int32_t* cluster_assignments) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    int32_t nc = static_cast<int32_t>(num_clusters);

    cache.ensure_sigma_capacity(nc);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;

    cudaStream_t stream = 0;
    cudaMemsetAsync(cache.d_sigma, 0, nc * sizeof(double), stream);
    cudaMemsetAsync(cache.d_intra_sum, 0, sizeof(double), stream);

    double avg_degree = (num_vertices > 0) ?
        static_cast<double>(num_edges) / num_vertices : 0.0;
    bool dense = (avg_degree >= 12.0);

    int tpb = 256;

    if (nc <= 2) {
        if (dense) {
            int wpb = tpb / 32;
            int nb = std::min((num_vertices + wpb - 1) / wpb, 80 * 16);
            modularity_k2_warp<<<nb, tpb, 3*wpb*sizeof(double), stream>>>(
                d_offsets, d_indices, edge_weights, cluster_assignments,
                cache.d_sigma, cache.d_intra_sum, num_vertices);
        } else {
            int nb = std::min((num_vertices + tpb - 1) / tpb, 80 * 16);
            nb = std::max(nb, 1);
            modularity_k2_thread<<<nb, tpb, 0, stream>>>(
                d_offsets, d_indices, edge_weights, cluster_assignments,
                cache.d_sigma, cache.d_intra_sum, num_vertices);
        }
    } else {
        if (dense) {
            int wpb = tpb / 32;
            int nb = std::min((num_vertices + wpb - 1) / wpb, 80 * 16);
            modularity_warp<<<nb, tpb, wpb*sizeof(double), stream>>>(
                d_offsets, d_indices, edge_weights, cluster_assignments,
                cache.d_sigma, cache.d_intra_sum, num_vertices, nc);
        } else {
            int nb = std::min((num_vertices + tpb - 1) / tpb, 80 * 16);
            nb = std::max(nb, 1);
            modularity_thread<<<nb, tpb, 0, stream>>>(
                d_offsets, d_indices, edge_weights, cluster_assignments,
                cache.d_sigma, cache.d_intra_sum, num_vertices, nc);
        }
    }

    finalize_modularity<<<1, 1, 0, stream>>>(
        cache.d_sigma, cache.d_intra_sum, cache.d_result, nc);

    double result;
    cudaMemcpy(&result, cache.d_result, sizeof(double), cudaMemcpyDeviceToHost);
    return result;
}

}  
