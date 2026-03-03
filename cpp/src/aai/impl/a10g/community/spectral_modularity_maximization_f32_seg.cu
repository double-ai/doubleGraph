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
#include <cooperative_groups.h>
#include <math_constants.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace aai {

namespace {

namespace cg = cooperative_groups;

struct Cache : Cacheable {
    cublasHandle_t cublas = nullptr;
    cusparseHandle_t cusparse = nullptr;
    cusolverDnHandle_t cusolver = nullptr;
    float* d_one = nullptr;
    float* d_zero = nullptr;
    float* d_neg_one = nullptr;
    int* d_info = nullptr;
    void* spmv_buffer = nullptr;
    size_t spmv_buffer_size = 0;

    Cache() {
        cublasCreate(&cublas);
        cusparseCreate(&cusparse);
        cusolverDnCreate(&cusolver);

        cudaMalloc(&d_one, sizeof(float));
        cudaMalloc(&d_zero, sizeof(float));
        cudaMalloc(&d_neg_one, sizeof(float));
        float one = 1.0f, zero = 0.0f, neg = -1.0f;
        cudaMemcpy(d_one, &one, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_zero, &zero, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_neg_one, &neg, sizeof(float), cudaMemcpyHostToDevice);

        cudaMalloc(&d_info, sizeof(int));

        spmv_buffer_size = 16 * 1024 * 1024;
        cudaMalloc(&spmv_buffer, spmv_buffer_size);
    }

    void ensure_spmv_buffer(size_t needed) {
        if (needed > spmv_buffer_size) {
            cudaFree(spmv_buffer);
            spmv_buffer_size = needed;
            cudaMalloc(&spmv_buffer, spmv_buffer_size);
        }
    }

    ~Cache() override {
        if (cublas) cublasDestroy(cublas);
        if (cusparse) cusparseDestroy(cusparse);
        if (cusolver) cusolverDnDestroy(cusolver);
        cudaFree(d_one);
        cudaFree(d_zero);
        cudaFree(d_neg_one);
        cudaFree(d_info);
        cudaFree(spmv_buffer);
    }
};





__global__ void compute_degrees_kernel(
    const int32_t* __restrict__ offsets, const float* __restrict__ edge_weights,
    float* __restrict__ degrees, int32_t num_vertices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vertices) {
        int start = offsets[idx], end = offsets[idx + 1];
        float sum = 0.0f;
        for (int j = start; j < end; j++) sum += edge_weights[j];
        degrees[idx] = sum;
    }
}

__global__ void fill_random_kernel(float* data, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned long long state = seed + idx * 6364136223846793005ULL;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        data[idx] = (float)(state >> 33) / (float)(1ULL << 31);
    }
}

__global__ void apply_degree_correction_kernel(
    float* __restrict__ Av, const float* __restrict__ degrees,
    const float* __restrict__ d_dot, float inv_2m, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) Av[idx] -= degrees[idx] * (*d_dot) * inv_2m;
}

__global__ void lanczos_subtract_kernel(
    float* __restrict__ w, const float* __restrict__ qj, const float* __restrict__ qjm1,
    const float* __restrict__ d_alpha, const float* __restrict__ d_beta,
    int n, bool has_prev
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = w[idx] - (*d_alpha) * qj[idx];
        if (has_prev) val -= (*d_beta) * qjm1[idx];
        w[idx] = val;
    }
}

__global__ void apply_degree_correction_dev2_kernel(
    float* __restrict__ Av, const float* __restrict__ degrees,
    const float* __restrict__ d_dot, const float* __restrict__ d_inv_2m, int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) Av[idx] -= degrees[idx] * (*d_dot) * (*d_inv_2m);
}

__global__ void scale_by_inv_device_kernel(float* __restrict__ x, const float* __restrict__ d_norm, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] /= (*d_norm);
}

__global__ void scale_kernel(float* __restrict__ x, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] *= scale;
}

__global__ void row_normalize_kernel(float* __restrict__ embeddings, int n, int k_ev) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float norm = 0.0f;
        for (int d = 0; d < k_ev; d++) {
            float val = embeddings[idx * k_ev + d];
            norm += val * val;
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            float inv_norm = 1.0f / norm;
            for (int d = 0; d < k_ev; d++) embeddings[idx * k_ev + d] *= inv_norm;
        }
    }
}




__global__ void kmeans_fused_cooperative_kernel(
    const float* __restrict__ embeddings,  
    float* __restrict__ centroids,          
    int32_t* __restrict__ assignments,      
    float* __restrict__ new_centroids,      
    int* __restrict__ counts,               
    int* __restrict__ converged_flag,       
    int n, int num_clusters, int k_ev, int max_iter, float kmean_tolerance
) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    for (int iter = 0; iter < max_iter; iter++) {
        
        int acc_size = num_clusters * k_ev;
        for (int i = tid; i < acc_size; i += total_threads) new_centroids[i] = 0.0f;
        for (int i = tid; i < num_clusters; i += total_threads) counts[i] = 0;
        if (tid == 0) converged_flag[0] = 0;  

        grid.sync();

        
        for (int i = tid; i < n; i += total_threads) {
            float best_dist = CUDART_INF_F;
            int best_cluster = 0;
            for (int c = 0; c < num_clusters; c++) {
                float dist = 0.0f;
                for (int d = 0; d < k_ev; d++) {
                    float diff = embeddings[i * k_ev + d] - centroids[c * k_ev + d];
                    dist += diff * diff;
                }
                if (dist < best_dist) { best_dist = dist; best_cluster = c; }
            }

            
            if (assignments[i] != best_cluster) {
                atomicAdd(converged_flag, 1);
            }
            assignments[i] = best_cluster;

            atomicAdd(&counts[best_cluster], 1);
            for (int d = 0; d < k_ev; d++) {
                atomicAdd(&new_centroids[best_cluster * k_ev + d], embeddings[i * k_ev + d]);
            }
        }

        grid.sync();

        
        for (int i = tid; i < acc_size; i += total_threads) {
            int c = i / k_ev;
            if (counts[c] > 0) centroids[i] = new_centroids[i] / (float)counts[c];
        }

        
        grid.sync();
        int num_changed = converged_flag[0];
        if (num_changed == 0 || (float)num_changed / (float)n < kmean_tolerance) return;
    }
}


__global__ void kmeans_assign_kernel(
    const float* __restrict__ embeddings, const float* __restrict__ centroids,
    int32_t* __restrict__ assignments, int n, int num_clusters, int k_ev
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float best_dist = CUDART_INF_F;
        int best_cluster = 0;
        for (int c = 0; c < num_clusters; c++) {
            float dist = 0.0f;
            for (int d = 0; d < k_ev; d++) {
                float diff = embeddings[idx * k_ev + d] - centroids[c * k_ev + d];
                dist += diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; best_cluster = c; }
        }
        assignments[idx] = best_cluster;
    }
}

__global__ void extract_diagonal_kernel(float* dst, const float* src, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = src[idx * stride + idx];
}


__global__ void copy_scalar_kernel(float* dst, const float* src, int src_idx) {
    if (threadIdx.x == 0) dst[0] = src[src_idx];
}

__global__ void fill_constant_kernel(float* data, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val;
}

__global__ void farthest_first_update_kernel(
    const float* __restrict__ embeddings,
    float* __restrict__ centroids,
    float* __restrict__ min_dists,
    int n, int k_ev, int new_centroid_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float dist = 0.0f;
        for (int d = 0; d < k_ev; d++) {
            float diff = embeddings[idx * k_ev + d] - centroids[new_centroid_idx * k_ev + d];
            dist += diff * diff;
        }
        if (dist < min_dists[idx]) min_dists[idx] = dist;
    }
}

__global__ void farthest_first_select_kernel(
    const float* __restrict__ embeddings,
    const float* __restrict__ min_dists,
    float* __restrict__ centroids,
    int n, int k_ev, int c_idx
) {
    __shared__ float s_max_val[256];
    __shared__ int s_max_idx[256];
    int tid = threadIdx.x;

    float best_val = -1.0f;
    int best_idx = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        if (min_dists[i] > best_val) {
            best_val = min_dists[i];
            best_idx = i;
        }
    }
    s_max_val[tid] = best_val;
    s_max_idx[tid] = best_idx;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && s_max_val[tid + s] > s_max_val[tid]) {
            s_max_val[tid] = s_max_val[tid + s];
            s_max_idx[tid] = s_max_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int sel = s_max_idx[0];
        for (int d = 0; d < k_ev; d++)
            centroids[c_idx * k_ev + d] = embeddings[sel * k_ev + d];
    }
}

__global__ void compute_inv_asum_kernel(const float* __restrict__ data, float* __restrict__ result, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) sum += fabsf(data[i]);
    sdata[tid] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) result[0] = 1.0f / sdata[0];
}


__global__ void init_centroids_kernel(
    const float* __restrict__ embeddings,
    float* __restrict__ centroids,
    int n, int num_clusters, int k_ev
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_clusters * k_ev) {
        int c = idx / k_ev;
        int d = idx % k_ev;
        int vertex = (int64_t)c * n / num_clusters;
        centroids[c * k_ev + d] = embeddings[vertex * k_ev + d];
    }
}

}  

void spectral_modularity_maximization_seg(const graph32_t& graph,
                                          const float* edge_weights,
                                          int32_t num_clusters,
                                          int32_t num_eigenvectors,
                                          float evs_tolerance,
                                          int32_t evs_max_iter,
                                          float kmean_tolerance,
                                          int32_t kmean_max_iter,
                                          int32_t* clustering) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t n = graph.number_of_vertices;
    int32_t nnz = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const float* d_weights = edge_weights;

    int blk = 256;
    int grd = (n + blk - 1) / blk;

    
    float* d_degrees;
    cudaMalloc(&d_degrees, n * sizeof(float));
    compute_degrees_kernel<<<grd, blk>>>(d_offsets, d_weights, d_degrees, n);

    float* d_inv_2m;
    cudaMalloc(&d_inv_2m, sizeof(float));
    compute_inv_asum_kernel<<<1, 256>>>(d_degrees, d_inv_2m, n);

    
    int lanczos_dim = evs_max_iter;
    if (lanczos_dim > n - 1) lanczos_dim = n - 1;
    if (lanczos_dim < num_eigenvectors + 2) lanczos_dim = num_eigenvectors + 2;

    float* d_Q;
    cudaMalloc(&d_Q, (int64_t)n * (lanczos_dim + 1) * sizeof(float));
    float* d_alpha;
    float* d_beta;
    cudaMalloc(&d_alpha, lanczos_dim * sizeof(float));
    cudaMalloc(&d_beta, (lanczos_dim + 1) * sizeof(float));
    cudaMemset(d_beta, 0, (lanczos_dim + 1) * sizeof(float));

    float* d_dot;
    float* d_dots_buf;
    cudaMalloc(&d_dot, sizeof(float));
    cudaMalloc(&d_dots_buf, (int64_t)lanczos_dim * lanczos_dim * sizeof(float));

    
    cusparseSpMatDescr_t spmat;
    cusparseCreateCsr(&spmat, n, n, nnz,
                     (void*)d_offsets, (void*)d_indices, (void*)d_weights,
                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseDnVecDescr_t vecIn, vecOut;
    cusparseCreateDnVec(&vecOut, n, (void*)(d_Q + n), CUDA_R_32F);
    cusparseCreateDnVec(&vecIn, n, (void*)d_Q, CUDA_R_32F);

    float h_one = 1.0f, h_zero = 0.0f;
    size_t needed = 0;
    cusparseSpMV_bufferSize(cache.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                           &h_one, spmat, vecIn, &h_zero, vecOut,
                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &needed);
    cache.ensure_spmv_buffer(needed);

    
    fill_random_kernel<<<grd, blk>>>(d_Q, n, 42ULL);
    cublasSetPointerMode(cache.cublas, CUBLAS_POINTER_MODE_DEVICE);
    cublasSnrm2(cache.cublas, n, d_Q, 1, d_dot);
    scale_by_inv_device_kernel<<<grd, blk>>>(d_Q, d_dot, n);

    
    int actual_lanczos_dim = lanczos_dim;
    int min_steps = num_eigenvectors + 2;
    for (int j = 0; j < lanczos_dim; j++) {
        float* d_qj = d_Q + (int64_t)j * n;
        float* d_qj1 = d_Q + (int64_t)(j + 1) * n;

        
        cusparseDnVecSetValues(vecIn, (void*)d_qj);
        cusparseDnVecSetValues(vecOut, (void*)d_qj1);
        cusparseSpMV(cache.cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &h_one, spmat, vecIn, &h_zero, vecOut,
                     CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, cache.spmv_buffer);

        
        cublasSdot(cache.cublas, n, d_degrees, 1, d_qj, 1, d_dot);
        apply_degree_correction_dev2_kernel<<<grd, blk>>>(d_qj1, d_degrees, d_dot, d_inv_2m, n);

        
        int nq = j + 1;
        float* d_dots_j = d_dots_buf + (int64_t)j * lanczos_dim;
        cublasSgemv(cache.cublas, CUBLAS_OP_T, n, nq, cache.d_one, d_Q, n, d_qj1, 1, cache.d_zero, d_dots_j, 1);
        cublasSgemv(cache.cublas, CUBLAS_OP_N, n, nq, cache.d_neg_one, d_Q, n, d_dots_j, 1, cache.d_one, d_qj1, 1);

        
        cublasSnrm2(cache.cublas, n, d_qj1, 1, &d_beta[j+1]);

        
        scale_by_inv_device_kernel<<<grd, blk>>>(d_qj1, &d_beta[j+1], n);

        
        if (j + 1 >= min_steps && (j + 1) % 5 == 0) {
            float h_beta_j1;
            cudaMemcpy(&h_beta_j1, &d_beta[j+1], sizeof(float), cudaMemcpyDeviceToHost);
            if (std::abs(h_beta_j1) < evs_tolerance || std::isnan(h_beta_j1)) {
                actual_lanczos_dim = j + 1;
                break;
            }
        }
    }

    cusparseDestroyDnVec(vecIn);
    cusparseDestroyDnVec(vecOut);

    int m = actual_lanczos_dim;

    
    extract_diagonal_kernel<<<(m + 255) / 256, 256>>>(d_alpha, d_dots_buf, m, lanczos_dim);

    
    std::vector<float> h_alpha(m), h_beta(m + 1);
    cudaMemcpy(h_alpha.data(), d_alpha, m * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_beta.data(), d_beta, (m + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    
    std::vector<float> T(m * m, 0.0f);
    for (int i = 0; i < m; i++) {
        T[i * m + i] = h_alpha[i];
        if (i < m - 1) { T[(i+1)*m + i] = h_beta[i+1]; T[i*m + (i+1)] = h_beta[i+1]; }
    }
    float* d_T;
    float* d_eigenvalues;
    cudaMalloc(&d_T, (int64_t)m * m * sizeof(float));
    cudaMalloc(&d_eigenvalues, m * sizeof(float));
    cudaMemcpy(d_T, T.data(), m * m * sizeof(float), cudaMemcpyHostToDevice);

    int work_size = 0;
    cusolverDnSsyevd_bufferSize(cache.cusolver, CUSOLVER_EIG_MODE_VECTOR,
                                CUBLAS_FILL_MODE_LOWER, m, d_T, m, d_eigenvalues, &work_size);
    float* d_work;
    cudaMalloc(&d_work, (int64_t)work_size * sizeof(float));
    cusolverDnSsyevd(cache.cusolver, CUSOLVER_EIG_MODE_VECTOR,
                    CUBLAS_FILL_MODE_LOWER, m, d_T, m, d_eigenvalues,
                    d_work, work_size, cache.d_info);

    
    cublasSetPointerMode(cache.cublas, CUBLAS_POINTER_MODE_HOST);
    float* d_Z_sel = d_T + (m - num_eigenvectors) * m;
    float* d_V;
    cudaMalloc(&d_V, (int64_t)n * num_eigenvectors * sizeof(float));
    cublasSgemm(cache.cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, num_eigenvectors, m,
               &h_one, d_Q, n, d_Z_sel, m, &h_zero, d_V, n);

    
    int k_ev = num_eigenvectors;
    float* d_emb;
    cudaMalloc(&d_emb, (int64_t)n * k_ev * sizeof(float));
    cublasSgeam(cache.cublas, CUBLAS_OP_T, CUBLAS_OP_N, k_ev, n, &h_one, d_V, n, &h_zero, d_V, k_ev, d_emb, k_ev);
    row_normalize_kernel<<<grd, blk>>>(d_emb, n, k_ev);

    
    float* d_centroids;
    float* d_min_dists;
    cudaMalloc(&d_centroids, (int64_t)num_clusters * k_ev * sizeof(float));
    cudaMalloc(&d_min_dists, n * sizeof(float));

    
    cudaMemcpy(d_centroids, d_emb, k_ev * sizeof(float), cudaMemcpyDeviceToDevice);
    fill_constant_kernel<<<grd, blk>>>(d_min_dists, 1e30f, n);
    farthest_first_update_kernel<<<grd, blk>>>(d_emb, d_centroids, d_min_dists, n, k_ev, 0);

    for (int c = 1; c < num_clusters; c++) {
        farthest_first_select_kernel<<<1, 256>>>(d_emb, d_min_dists, d_centroids, n, k_ev, c);
        farthest_first_update_kernel<<<grd, blk>>>(d_emb, d_centroids, d_min_dists, n, k_ev, c);
    }

    
    cudaMemset(clustering, 0xFF, n * sizeof(int32_t));

    float* d_new_centroids;
    int* d_counts;
    int* d_converged_flag;
    cudaMalloc(&d_new_centroids, (int64_t)num_clusters * k_ev * sizeof(float));
    cudaMalloc(&d_counts, num_clusters * sizeof(int));
    cudaMalloc(&d_converged_flag, sizeof(int));

    
    int block_size = 256;
    int num_blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
        kmeans_fused_cooperative_kernel, block_size, 0);

    int device;
    cudaGetDevice(&device);
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device);

    int max_blocks = num_blocks_per_sm * num_sms;
    int needed_blocks = (n + block_size - 1) / block_size;
    int grid_size = (needed_blocks < max_blocks) ? needed_blocks : max_blocks;

    void* args[] = {
        (void*)&d_emb, (void*)&d_centroids, (void*)&clustering,
        (void*)&d_new_centroids, (void*)&d_counts, (void*)&d_converged_flag,
        (void*)&n, (void*)&num_clusters, (void*)&k_ev, (void*)&kmean_max_iter, (void*)&kmean_tolerance
    };

    cudaLaunchCooperativeKernel(
        (void*)kmeans_fused_cooperative_kernel,
        dim3(grid_size), dim3(block_size), args, 0, 0);

    cusparseDestroySpMat(spmat);

    
    cudaFree(d_degrees);
    cudaFree(d_inv_2m);
    cudaFree(d_Q);
    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_dot);
    cudaFree(d_dots_buf);
    cudaFree(d_T);
    cudaFree(d_eigenvalues);
    cudaFree(d_work);
    cudaFree(d_V);
    cudaFree(d_emb);
    cudaFree(d_centroids);
    cudaFree(d_min_dists);
    cudaFree(d_new_centroids);
    cudaFree(d_counts);
    cudaFree(d_converged_flag);
}

}  
