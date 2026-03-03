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
#include <math_constants.h>
#include <cusparse.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace aai {

namespace {




struct Cache : Cacheable {
    cusparseHandle_t cusparse_h = nullptr;
    cublasHandle_t cublas_h = nullptr;

    
    float* d_V = nullptr;
    int64_t d_V_capacity = 0;

    float* d_w = nullptr;
    int64_t d_w_capacity = 0;

    float* d_gamma = nullptr;
    int64_t d_gamma_capacity = 0;

    float* d_alpha = nullptr;
    int64_t d_alpha_capacity = 0;

    float* d_beta = nullptr;
    int64_t d_beta_capacity = 0;

    float* d_h = nullptr;
    int64_t d_h_capacity = 0;

    float* d_degrees = nullptr;
    int64_t d_degrees_capacity = 0;

    float* d_Y = nullptr;
    int64_t d_Y_capacity = 0;

    float* d_S = nullptr;
    int64_t d_S_capacity = 0;

    float* d_ones = nullptr;
    int64_t d_ones_capacity = 0;

    float* d_centroids = nullptr;
    int64_t d_centroids_capacity = 0;

    float* d_centroid_sums = nullptr;
    int64_t d_centroid_sums_capacity = 0;

    int32_t* d_counts = nullptr;
    int64_t d_counts_capacity = 0;

    int32_t* d_assign_old = nullptr;
    int64_t d_assign_old_capacity = 0;

    int32_t* d_changed = nullptr;
    int64_t d_changed_capacity = 0;

    uint8_t* d_spmv_buffer = nullptr;
    int64_t d_spmv_buffer_capacity = 0;

    Cache() {
        cusparseCreate(&cusparse_h);
        cublasCreate(&cublas_h);
    }

    ~Cache() override {
        if (cusparse_h) cusparseDestroy(cusparse_h);
        if (cublas_h) cublasDestroy(cublas_h);
        if (d_V) cudaFree(d_V);
        if (d_w) cudaFree(d_w);
        if (d_gamma) cudaFree(d_gamma);
        if (d_alpha) cudaFree(d_alpha);
        if (d_beta) cudaFree(d_beta);
        if (d_h) cudaFree(d_h);
        if (d_degrees) cudaFree(d_degrees);
        if (d_Y) cudaFree(d_Y);
        if (d_S) cudaFree(d_S);
        if (d_ones) cudaFree(d_ones);
        if (d_centroids) cudaFree(d_centroids);
        if (d_centroid_sums) cudaFree(d_centroid_sums);
        if (d_counts) cudaFree(d_counts);
        if (d_assign_old) cudaFree(d_assign_old);
        if (d_changed) cudaFree(d_changed);
        if (d_spmv_buffer) cudaFree(d_spmv_buffer);
    }

    void ensure_V(int64_t size) {
        if (d_V_capacity < size) {
            if (d_V) cudaFree(d_V);
            cudaMalloc(&d_V, size * sizeof(float));
            d_V_capacity = size;
        }
    }
    void ensure_w(int64_t size) {
        if (d_w_capacity < size) {
            if (d_w) cudaFree(d_w);
            cudaMalloc(&d_w, size * sizeof(float));
            d_w_capacity = size;
        }
    }
    void ensure_gamma(int64_t size) {
        if (d_gamma_capacity < size) {
            if (d_gamma) cudaFree(d_gamma);
            cudaMalloc(&d_gamma, size * sizeof(float));
            d_gamma_capacity = size;
        }
    }
    void ensure_alpha(int64_t size) {
        if (d_alpha_capacity < size) {
            if (d_alpha) cudaFree(d_alpha);
            cudaMalloc(&d_alpha, size * sizeof(float));
            d_alpha_capacity = size;
        }
    }
    void ensure_beta(int64_t size) {
        if (d_beta_capacity < size) {
            if (d_beta) cudaFree(d_beta);
            cudaMalloc(&d_beta, size * sizeof(float));
            d_beta_capacity = size;
        }
    }
    void ensure_h(int64_t size) {
        if (d_h_capacity < size) {
            if (d_h) cudaFree(d_h);
            cudaMalloc(&d_h, size * sizeof(float));
            d_h_capacity = size;
        }
    }
    void ensure_degrees(int64_t size) {
        if (d_degrees_capacity < size) {
            if (d_degrees) cudaFree(d_degrees);
            cudaMalloc(&d_degrees, size * sizeof(float));
            d_degrees_capacity = size;
        }
    }
    void ensure_Y(int64_t size) {
        if (d_Y_capacity < size) {
            if (d_Y) cudaFree(d_Y);
            cudaMalloc(&d_Y, size * sizeof(float));
            d_Y_capacity = size;
        }
    }
    void ensure_S(int64_t size) {
        if (d_S_capacity < size) {
            if (d_S) cudaFree(d_S);
            cudaMalloc(&d_S, size * sizeof(float));
            d_S_capacity = size;
        }
    }
    void ensure_ones(int64_t size) {
        if (d_ones_capacity < size) {
            if (d_ones) cudaFree(d_ones);
            cudaMalloc(&d_ones, size * sizeof(float));
            d_ones_capacity = size;
        }
    }
    void ensure_centroids(int64_t size) {
        if (d_centroids_capacity < size) {
            if (d_centroids) cudaFree(d_centroids);
            cudaMalloc(&d_centroids, size * sizeof(float));
            d_centroids_capacity = size;
        }
    }
    void ensure_centroid_sums(int64_t size) {
        if (d_centroid_sums_capacity < size) {
            if (d_centroid_sums) cudaFree(d_centroid_sums);
            cudaMalloc(&d_centroid_sums, size * sizeof(float));
            d_centroid_sums_capacity = size;
        }
    }
    void ensure_counts(int64_t size) {
        if (d_counts_capacity < size) {
            if (d_counts) cudaFree(d_counts);
            cudaMalloc(&d_counts, size * sizeof(int32_t));
            d_counts_capacity = size;
        }
    }
    void ensure_assign_old(int64_t size) {
        if (d_assign_old_capacity < size) {
            if (d_assign_old) cudaFree(d_assign_old);
            cudaMalloc(&d_assign_old, size * sizeof(int32_t));
            d_assign_old_capacity = size;
        }
    }
    void ensure_changed(int64_t size) {
        if (d_changed_capacity < size) {
            if (d_changed) cudaFree(d_changed);
            cudaMalloc(&d_changed, size * sizeof(int32_t));
            d_changed_capacity = size;
        }
    }
    void ensure_spmv_buffer(int64_t size) {
        if (d_spmv_buffer_capacity < size) {
            if (d_spmv_buffer) cudaFree(d_spmv_buffer);
            cudaMalloc(&d_spmv_buffer, size);
            d_spmv_buffer_capacity = size;
        }
    }
};





static __device__ __forceinline__ uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__global__ void compute_degrees_kernel(const int* offsets, const float* weights, float* degrees, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int start = offsets[i];
        int end = offsets[i + 1];
        float sum = 0.0f;
        for (int j = start; j < end; j++) sum += weights[j];
        degrees[i] = sum;
    }
}

__global__ void modularity_correction_kernel(float* w, const float* d, const float* gamma, float edge_sum, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] -= (*gamma / edge_sum) * d[i];
}

__global__ void lanczos_recurrence_kernel(float* w, const float* v_j, const float* v_jm1,
                                         const float* alpha_ptr, const float* beta_ptr,
                                         int n, bool has_prev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float a = *alpha_ptr;
        float out = w[i] - a * v_j[i];
        if (has_prev) {
            float b = *beta_ptr;
            out = fmaf(-b, v_jm1[i], out);
        }
        w[i] = out;
    }
}

__global__ void normalize_and_copy_kernel(const float* w, float* v_next, const float* beta_ptr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float b = *beta_ptr;
        v_next[i] = (b > 1e-30f) ? (w[i] / b) : 0.0f;
    }
}

__global__ void fill_kernel(float* v, int n, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = val;
}

__global__ void fill_random_kernel(float* v, int n, uint32_t seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        uint32_t x = hash_u32(((uint32_t)i) ^ seed);
        float u = (float)(x & 0x00FFFFFFu) * (1.0f / 16777216.0f);
        v[i] = u - 0.5f;
    }
}

__global__ void row_normalize_colmajor_kernel(float* data, int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float norm = 0.0f;
        for (int d = 0; d < dim; d++) {
            float v = data[i + (int64_t)d * n];
            norm = fmaf(v, v, norm);
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            float inv = 1.0f / norm;
            for (int d = 0; d < dim; d++) {
                data[i + (int64_t)d * n] *= inv;
            }
        }
    }
}

__global__ void kmeans_assign_count_kernel_global(const float* __restrict__ data,
                                                 const float* __restrict__ centroids,
                                                 const int* __restrict__ old_assign,
                                                 int* __restrict__ new_assign,
                                                 int* __restrict__ changed,
                                                 int n,
                                                 int k,
                                                 int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float best_dist = CUDART_INF_F;
        int best_k = 0;
        for (int c = 0; c < k; c++) {
            float dist = 0.0f;
            const float* cent = centroids + (int64_t)c * dim;
            for (int d = 0; d < dim; d++) {
                float diff = data[i + (int64_t)d * n] - cent[d];
                dist = fmaf(diff, diff, dist);
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_k = c;
            }
        }
        int old = old_assign[i];
        new_assign[i] = best_k;
        bool did_change = (old != best_k);
        unsigned amask = __activemask();
        unsigned mask = __ballot_sync(amask, did_change);
        if (mask) {
            int lane = threadIdx.x & 31;
            int leader = __ffs(mask) - 1;
            if (lane == leader) atomicAdd(changed, __popc(mask));
        }
    }
}

__global__ void kmeans_assign_count_kernel_shared(const float* __restrict__ data,
                                                 const float* __restrict__ centroids,
                                                 const int* __restrict__ old_assign,
                                                 int* __restrict__ new_assign,
                                                 int* __restrict__ changed,
                                                 int n,
                                                 int k,
                                                 int dim) {
    extern __shared__ float s_centroids[];
    for (int idx = threadIdx.x; idx < k * dim; idx += blockDim.x) s_centroids[idx] = centroids[idx];
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float best_dist = CUDART_INF_F;
        int best_k = 0;
        for (int c = 0; c < k; c++) {
            float dist = 0.0f;
            const float* cent = s_centroids + (int64_t)c * dim;
            for (int d = 0; d < dim; d++) {
                float diff = data[i + (int64_t)d * n] - cent[d];
                dist = fmaf(diff, diff, dist);
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_k = c;
            }
        }
        int old = old_assign[i];
        new_assign[i] = best_k;
        bool did_change = (old != best_k);
        unsigned amask = __activemask();
        unsigned mask = __ballot_sync(amask, did_change);
        if (mask) {
            int lane = threadIdx.x & 31;
            int leader = __ffs(mask) - 1;
            if (lane == leader) atomicAdd(changed, __popc(mask));
        }
    }
}

__global__ void kmeans_accumulate_kernel_global(const float* __restrict__ data,
                                               const int* __restrict__ assignments,
                                               float* __restrict__ centroid_sums,
                                               int* __restrict__ counts,
                                               int n,
                                               int k,
                                               int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int c = assignments[i];
        unsigned amask = __activemask();
        unsigned mmask = __match_any_sync(amask, c);
        int lane = threadIdx.x & 31;
        int leader = __ffs(mmask) - 1;
        if (lane == leader) atomicAdd(&counts[c], __popc(mmask));
        float* out = centroid_sums + (int64_t)c * dim;
        for (int d = 0; d < dim; d++) {
            atomicAdd(&out[d], data[i + (int64_t)d * n]);
        }
    }
}

__global__ void kmeans_accumulate_kernel_shared(const float* __restrict__ data,
                                               const int* __restrict__ assignments,
                                               float* __restrict__ centroid_sums,
                                               int* __restrict__ counts,
                                               int n,
                                               int k,
                                               int dim) {
    extern __shared__ unsigned char smem[];
    float* s_sums = (float*)smem;
    int* s_counts = (int*)(smem + (size_t)k * dim * sizeof(float));

    for (int idx = threadIdx.x; idx < k * dim; idx += blockDim.x) s_sums[idx] = 0.0f;
    for (int idx = threadIdx.x; idx < k; idx += blockDim.x) s_counts[idx] = 0;
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        int c = assignments[i];
        atomicAdd(&s_counts[c], 1);
        float* out = s_sums + (int64_t)c * dim;
        for (int d = 0; d < dim; d++) {
            atomicAdd(&out[d], data[i + (int64_t)d * n]);
        }
    }

    __syncthreads();

    for (int idx = threadIdx.x; idx < k * dim; idx += blockDim.x) atomicAdd(&centroid_sums[idx], s_sums[idx]);
    for (int idx = threadIdx.x; idx < k; idx += blockDim.x) atomicAdd(&counts[idx], s_counts[idx]);
}

__global__ void kmeans_divide_kernel(float* centroids, const float* centroid_sums, const int* counts, int k, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k * dim) {
        int c = idx / dim;
        int cnt = counts[c];
        if (cnt > 0) centroids[idx] = centroid_sums[idx] * (1.0f / (float)cnt);
    }
}





static void tridiag_ql(int n, float* d, float* e, float* z)
{
    for (int i = 0; i < n * n; i++) z[i] = 0.0f;
    for (int i = 0; i < n; i++) z[i + i * n] = 1.0f;
    if (n <= 1) return;

    for (int l = 0; l < n; l++) {
        int iter = 0;
        int m;
        do {
            for (m = l; m < n - 1; m++) {
                float dd = fabsf(d[m]) + fabsf(d[m + 1]);
                if (fabsf(e[m]) + dd == dd) break;
            }
            if (m != l) {
                if (++iter > 300) return;
                float g = (d[l + 1] - d[l]) / (2.0f * e[l]);
                float r = sqrtf(g * g + 1.0f);
                g = d[m] - d[l] + e[l] / (g + copysignf(r, g));
                float s = 1.0f, c = 1.0f, p = 0.0f;
                for (int i = m - 1; i >= l; i--) {
                    float f = s * e[i];
                    float b = c * e[i];
                    if (fabsf(f) >= fabsf(g)) {
                        c = g / f;
                        r = sqrtf(c * c + 1.0f);
                        e[i + 1] = f * r;
                        s = 1.0f / r;
                        c *= s;
                    } else {
                        s = f / g;
                        r = sqrtf(s * s + 1.0f);
                        e[i + 1] = g * r;
                        c = 1.0f / r;
                        s *= c;
                    }
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0f * c * b;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - b;
                    for (int k = 0; k < n; k++) {
                        float t = z[k + (i + 1) * n];
                        z[k + (i + 1) * n] = s * z[k + i * n] + c * t;
                        z[k + i * n] = c * z[k + i * n] - s * t;
                    }
                }
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0f;
            }
        } while (m != l);
    }
}





static void launch_compute_degrees(const int* offsets, const float* weights, float* degrees, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    compute_degrees_kernel<<<grid, block, 0, stream>>>(offsets, weights, degrees, n);
}

static void launch_modularity_correction(float* w, const float* d, const float* gamma, float edge_sum, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    modularity_correction_kernel<<<grid, block, 0, stream>>>(w, d, gamma, edge_sum, n);
}

static void launch_lanczos_recurrence(float* w,
                              const float* v_j,
                              const float* v_jm1,
                              const float* alpha_ptr,
                              const float* beta_ptr,
                              int n,
                              bool has_prev,
                              cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    lanczos_recurrence_kernel<<<grid, block, 0, stream>>>(w, v_j, v_jm1, alpha_ptr, beta_ptr, n, has_prev);
}

static void launch_fill(float* v, int n, float val, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    fill_kernel<<<grid, block, 0, stream>>>(v, n, val);
}

static void launch_fill_random(float* v, int n, uint32_t seed, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    fill_random_kernel<<<grid, block, 0, stream>>>(v, n, seed);
}

static void launch_normalize_and_copy(const float* w, float* v_next, const float* beta_ptr, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    normalize_and_copy_kernel<<<grid, block, 0, stream>>>(w, v_next, beta_ptr, n);
}

static void launch_row_normalize_colmajor(float* data, int n, int dim, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    row_normalize_colmajor_kernel<<<grid, block, 0, stream>>>(data, n, dim);
}

static void launch_kmeans_assign_count(const float* data,
                               const float* centroids,
                               const int* old_assign,
                               int* new_assign,
                               int* changed,
                               int n,
                               int k,
                               int dim,
                               cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    size_t shmem = (size_t)k * dim * sizeof(float);
    if (shmem <= 4096) {
        kmeans_assign_count_kernel_shared<<<grid, block, shmem, stream>>>(data, centroids, old_assign, new_assign, changed, n, k, dim);
    } else {
        kmeans_assign_count_kernel_global<<<grid, block, 0, stream>>>(data, centroids, old_assign, new_assign, changed, n, k, dim);
    }
}

static void launch_kmeans_accumulate(const float* data,
                             const int* assignments,
                             float* centroid_sums,
                             int* counts,
                             int n,
                             int k,
                             int dim,
                             cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    size_t shmem = (size_t)k * dim * sizeof(float) + (size_t)k * sizeof(int);
    if (shmem <= 16384) {
        kmeans_accumulate_kernel_shared<<<grid, block, shmem, stream>>>(data, assignments, centroid_sums, counts, n, k, dim);
    } else {
        kmeans_accumulate_kernel_global<<<grid, block, 0, stream>>>(data, assignments, centroid_sums, counts, n, k, dim);
    }
}

static void launch_kmeans_divide(float* centroids,
                          const float* centroid_sums,
                          const int* counts,
                          int k,
                          int dim,
                          cudaStream_t stream) {
    int block = 256;
    int grid = (k * dim + block - 1) / block;
    kmeans_divide_kernel<<<grid, block, 0, stream>>>(centroids, centroid_sums, counts, k, dim);
}

}  




void spectral_modularity_maximization(const graph32_t& graph,
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

    const int32_t n = graph.number_of_vertices;
    const int32_t nnz = graph.number_of_edges;
    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const float* d_weights = edge_weights;
    const int32_t nev = num_eigenvectors;

    cudaStream_t stream = 0;
    cublasSetStream(cache.cublas_h, stream);
    cusparseSetStream(cache.cusparse_h, stream);

    
    cache.ensure_degrees(n);
    float* d_degrees = cache.d_degrees;
    launch_compute_degrees(d_offsets, d_weights, d_degrees, n, stream);

    
    cublasSetPointerMode(cache.cublas_h, CUBLAS_POINTER_MODE_HOST);
    float edge_sum = 0.0f;
    cublasSasum(cache.cublas_h, n, d_degrees, 1, &edge_sum);
    if (edge_sum < 1e-10f) {
        cudaMemsetAsync(clustering, 0, n * sizeof(int32_t), stream);
        cudaStreamSynchronize(stream);
        return;
    }

    
    const int max_m = std::min<int>(std::max<int>(2, evs_max_iter), n);

    cache.ensure_V((int64_t)(max_m + 1) * (int64_t)n);
    float* d_V = cache.d_V;

    cache.ensure_w(n);
    float* d_w = cache.d_w;

    cache.ensure_gamma(1);
    float* d_gamma = cache.d_gamma;

    cache.ensure_alpha(max_m);
    float* d_alpha = cache.d_alpha;

    cache.ensure_beta(max_m + 1);
    float* d_beta = cache.d_beta;
    cudaMemsetAsync(d_beta, 0, (max_m + 1) * sizeof(float), stream);

    cache.ensure_h(max_m);
    float* d_h = cache.d_h;

    
    uint32_t seed = 0x1234u ^ (uint32_t)n ^ ((uint32_t)nnz << 1);
    launch_fill_random(d_V, n, seed, stream);

    cublasSetPointerMode(cache.cublas_h, CUBLAS_POINTER_MODE_DEVICE);
    cublasSnrm2(cache.cublas_h, n, d_V, 1, d_beta + 1);
    launch_normalize_and_copy(d_V, d_V, d_beta + 1, n, stream);
    cudaMemsetAsync(d_beta + 1, 0, sizeof(float), stream);

    
    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA,
                      n,
                      n,
                      nnz,
                      (void*)d_offsets,
                      (void*)d_indices,
                      (void*)d_weights,
                      CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO,
                      CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, n, (void*)d_V, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, n, (void*)d_w, CUDA_R_32F);

    float spmv_alpha = 1.0f, spmv_beta = 0.0f;
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(cache.cusparse_h,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &spmv_alpha,
                            matA,
                            vecX,
                            &spmv_beta,
                            vecY,
                            CUDA_R_32F,
                            CUSPARSE_SPMV_CSR_ALG2,
                            &bufferSize);

    cache.ensure_spmv_buffer(std::max<int64_t>((int64_t)bufferSize, 1));
    void* d_spmv_buffer = cache.d_spmv_buffer;

    
    const int min_check = std::min(max_m, std::max(10, 2 * nev + 4));
    const int check_interval = std::max(10, 4 * nev);

    int actual_dim = 0;

    std::vector<float> h_alpha;
    std::vector<float> h_beta;

    for (int j = 0; j < max_m; j++) {
        float* v_j = d_V + (int64_t)j * n;

        
        cusparseDnVecSetValues(vecX, (void*)v_j);
        cusparseDnVecSetValues(vecY, (void*)d_w);
        cusparseSpMV(cache.cusparse_h,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &spmv_alpha,
                     matA,
                     vecX,
                     &spmv_beta,
                     vecY,
                     CUDA_R_32F,
                     CUSPARSE_SPMV_CSR_ALG2,
                     d_spmv_buffer);

        
        cublasSetPointerMode(cache.cublas_h, CUBLAS_POINTER_MODE_DEVICE);
        cublasSdot(cache.cublas_h, n, d_degrees, 1, v_j, 1, d_gamma);

        
        launch_modularity_correction(d_w, d_degrees, d_gamma, edge_sum, n, stream);

        
        cublasSdot(cache.cublas_h, n, v_j, 1, d_w, 1, d_alpha + j);

        
        float* v_jm1 = (j > 0) ? (d_V + (int64_t)(j - 1) * n) : nullptr;
        launch_lanczos_recurrence(d_w, v_j, v_jm1, d_alpha + j, d_beta + j, n, j > 0, stream);

        
        if (j > 0) {
            cublasSetPointerMode(cache.cublas_h, CUBLAS_POINTER_MODE_HOST);
            float one = 1.0f, zero = 0.0f, neg_one = -1.0f;
            cublasSgemv(cache.cublas_h, CUBLAS_OP_T, n, j + 1, &one, d_V, n, d_w, 1, &zero, d_h, 1);
            cublasSgemv(cache.cublas_h, CUBLAS_OP_N, n, j + 1, &neg_one, d_V, n, d_h, 1, &one, d_w, 1);
        }

        
        cublasSetPointerMode(cache.cublas_h, CUBLAS_POINTER_MODE_DEVICE);
        cublasSnrm2(cache.cublas_h, n, d_w, 1, d_beta + j + 1);

        
        if (j + 1 <= max_m) {
            launch_normalize_and_copy(d_w, d_V + (int64_t)(j + 1) * n, d_beta + j + 1, n, stream);
        }

        int m = j + 1;
        if (m >= min_check && ((m % check_interval) == 0 || j == (max_m - 1))) {
            
            h_alpha.resize(m);
            h_beta.resize(m + 1);
            cudaMemcpy(h_alpha.data(), d_alpha, m * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_beta.data(), d_beta, (m + 1) * sizeof(float), cudaMemcpyDeviceToHost);

            if (fabsf(h_beta[m]) < 1e-12f) {
                actual_dim = m;
                break;
            }

            
            std::vector<float> d_diag(m);
            std::vector<float> e_subdiag(m, 0.0f);
            for (int i = 0; i < m; i++) d_diag[i] = h_alpha[i];
            for (int i = 0; i < m - 1; i++) e_subdiag[i] = h_beta[i + 1];
            std::vector<float> Z(m * m, 0.0f);
            tridiag_ql(m, d_diag.data(), e_subdiag.data(), Z.data());

            std::vector<int> idx(m);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](int a, int b) { return d_diag[a] > d_diag[b]; });

            float beta_m = fabsf(h_beta[m]);
            bool converged = true;
            for (int t = 0; t < nev; t++) {
                int col = idx[t];
                float lambda = d_diag[col];
                float last = fabsf(Z[(m - 1) + col * m]);
                float resid = beta_m * last;
                float thresh = evs_tolerance * fmaxf(1.0f, fabsf(lambda));
                if (resid > thresh) {
                    converged = false;
                    break;
                }
            }

            if (converged) {
                actual_dim = m;
                break;
            }
        }
    }

    if (actual_dim == 0) actual_dim = max_m;

    
    h_alpha.resize(actual_dim);
    h_beta.resize(actual_dim + 1);
    cudaMemcpy(h_alpha.data(), d_alpha, actual_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_beta.data(), d_beta, (actual_dim + 1) * sizeof(float), cudaMemcpyDeviceToHost);

    
    std::vector<float> d_diag(actual_dim);
    std::vector<float> e_subdiag(actual_dim, 0.0f);
    for (int i = 0; i < actual_dim; i++) d_diag[i] = h_alpha[i];
    for (int i = 0; i < actual_dim - 1; i++) e_subdiag[i] = h_beta[i + 1];

    std::vector<float> Z(actual_dim * actual_dim, 0.0f);
    tridiag_ql(actual_dim, d_diag.data(), e_subdiag.data(), Z.data());

    std::vector<int> idx(actual_dim);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return d_diag[a] > d_diag[b]; });

    
    std::vector<float> S_top(actual_dim * nev, 0.0f);
    std::vector<float> sel_eval(nev);
    for (int j = 0; j < nev; j++) {
        int col = idx[j];
        float lambda = d_diag[col];
        sel_eval[j] = lambda;
        if (lambda <= 0.0f) {
            continue;
        }
        for (int i = 0; i < actual_dim; i++) {
            S_top[i + j * actual_dim] = Z[i + col * actual_dim];
        }
    }

    cache.ensure_Y((int64_t)n * nev);
    float* d_Y = cache.d_Y;

    cache.ensure_S((int64_t)actual_dim * nev);
    float* d_S = cache.d_S;
    cudaMemcpyAsync(d_S, S_top.data(), (size_t)actual_dim * nev * sizeof(float), cudaMemcpyHostToDevice, stream);

    cublasSetPointerMode(cache.cublas_h, CUBLAS_POINTER_MODE_HOST);
    float one = 1.0f, zero = 0.0f;
    cublasSgemm(cache.cublas_h,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n,
                nev,
                actual_dim,
                &one,
                d_V,
                n,
                d_S,
                actual_dim,
                &zero,
                d_Y,
                n);

    
    cache.ensure_ones(n);
    float* d_ones = cache.d_ones;
    launch_fill(d_ones, n, 1.0f, stream);

    for (int j = 0; j < nev; j++) {
        float* col = d_Y + (int64_t)j * n;
        float sum_val = 0.0f;
        cublasSdot(cache.cublas_h, n, col, 1, d_ones, 1, &sum_val);
        float mean = sum_val / (float)n;
        float neg_mean = -mean;
        cublasSaxpy(cache.cublas_h, n, &neg_mean, d_ones, 1, col, 1);
        float nrm = 0.0f;
        cublasSnrm2(cache.cublas_h, n, col, 1, &nrm);
        float std_val = nrm / sqrtf((float)n);
        if (std_val > 1e-10f) {
            float inv_std = 1.0f / std_val;
            cublasSscal(cache.cublas_h, n, &inv_std, col, 1);
        }
    }

    
    launch_row_normalize_colmajor(d_Y, n, nev, stream);

    
    cache.ensure_centroids((int64_t)num_clusters * nev);
    float* d_centroids = cache.d_centroids;
    cache.ensure_centroid_sums((int64_t)num_clusters * nev);
    float* d_centroid_sums = cache.d_centroid_sums;
    cache.ensure_counts(num_clusters);
    int32_t* d_counts = cache.d_counts;

    
    std::vector<float> h_Y((size_t)n * nev);
    cudaMemcpy(h_Y.data(), d_Y, (size_t)n * nev * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> h_centroids((size_t)num_clusters * nev);
    std::vector<float> min_dists(n, 1e30f);

    
    for (int d = 0; d < nev; d++) h_centroids[d] = h_Y[0 + (size_t)d * n];

    for (int c = 1; c < num_clusters; c++) {
        for (int i = 0; i < n; i++) {
            float dist = 0.0f;
            for (int d = 0; d < nev; d++) {
                float diff = h_Y[i + (size_t)d * n] - h_centroids[(size_t)(c - 1) * nev + d];
                dist += diff * diff;
            }
            if (dist < min_dists[i]) min_dists[i] = dist;
        }
        int best_idx = 0;
        float best_dist = -1.0f;
        for (int i = 0; i < n; i++) {
            if (min_dists[i] > best_dist) {
                best_dist = min_dists[i];
                best_idx = i;
            }
        }
        for (int d = 0; d < nev; d++) h_centroids[(size_t)c * nev + d] = h_Y[best_idx + (size_t)d * n];
    }

    cudaMemcpyAsync(d_centroids,
                    h_centroids.data(),
                    (size_t)num_clusters * nev * sizeof(float),
                    cudaMemcpyHostToDevice,
                    stream);

    cache.ensure_assign_old(n);
    int32_t* d_assign_old = cache.d_assign_old;
    cudaMemsetAsync(d_assign_old, 0xFF, (size_t)n * sizeof(int32_t), stream);

    cache.ensure_changed(1);
    int32_t* d_changed = cache.d_changed;

    const float change_thresh = kmean_tolerance * (float)n;

    for (int iter = 0; iter < kmean_max_iter; iter++) {
        
        cudaMemsetAsync(d_changed, 0, sizeof(int32_t), stream);
        launch_kmeans_assign_count(d_Y,
                                  d_centroids,
                                  d_assign_old,
                                  clustering,
                                  d_changed,
                                  n,
                                  num_clusters,
                                  nev,
                                  stream);

        if (iter > 0) {
            int32_t h_changed = 0;
            cudaMemcpy(&h_changed, d_changed, sizeof(int32_t), cudaMemcpyDeviceToHost);
            if ((float)h_changed <= change_thresh) break;
        }

        
        cudaMemcpyAsync(d_assign_old, clustering, (size_t)n * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

        
        cudaMemsetAsync(d_centroid_sums, 0, (size_t)num_clusters * nev * sizeof(float), stream);
        cudaMemsetAsync(d_counts, 0, (size_t)num_clusters * sizeof(int32_t), stream);
        launch_kmeans_accumulate(d_Y, clustering, d_centroid_sums, d_counts, n, num_clusters, nev, stream);
        launch_kmeans_divide(d_centroids, d_centroid_sums, d_counts, num_clusters, nev, stream);
    }

    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(matA);

    cudaStreamSynchronize(stream);
}

}  
