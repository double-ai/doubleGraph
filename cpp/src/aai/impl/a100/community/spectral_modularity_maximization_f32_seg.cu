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
#include <cublas_v2.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>
#include <math_constants.h>

namespace aai {

namespace {





struct Cache : Cacheable {
    cublasHandle_t cublas = nullptr;
    float* d_one = nullptr;
    float* d_neg_one = nullptr;
    float* d_zero = nullptr;

    Cache() {
        cublasCreate(&cublas);
        cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);
        cudaMalloc(&d_one, sizeof(float));
        cudaMalloc(&d_neg_one, sizeof(float));
        cudaMalloc(&d_zero, sizeof(float));
        float h = 1.0f; cudaMemcpy(d_one, &h, sizeof(float), cudaMemcpyHostToDevice);
        h = -1.0f; cudaMemcpy(d_neg_one, &h, sizeof(float), cudaMemcpyHostToDevice);
        h = 0.0f; cudaMemcpy(d_zero, &h, sizeof(float), cudaMemcpyHostToDevice);
    }

    ~Cache() override {
        if (cublas) cublasDestroy(cublas);
        if (d_one) cudaFree(d_one);
        if (d_neg_one) cudaFree(d_neg_one);
        if (d_zero) cudaFree(d_zero);
    }
};





__global__ void compute_degrees_kernel(const int32_t* __restrict__ offsets,
                                        const float* __restrict__ weights,
                                        float* __restrict__ degrees, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        int start = offsets[i], end = offsets[i + 1];
        for (int j = start; j < end; j++) sum += weights[j];
        degrees[i] = sum;
    }
}

__global__ void fused_spmv_mod_kernel(const int32_t* __restrict__ offsets,
                                       const int32_t* __restrict__ indices,
                                       const float* __restrict__ values,
                                       const float* __restrict__ degrees,
                                       float inv_2m,
                                       const float* __restrict__ x,
                                       float* __restrict__ y,
                                       int n,
                                       const float* __restrict__ d_dot_val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid & 31;

    if (warp_id < n) {
        int start = offsets[warp_id], end = offsets[warp_id + 1];
        float sum = 0.0f;
        for (int j = start + lane_id; j < end; j += 32)
            sum += values[j] * x[indices[j]];
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane_id == 0)
            y[warp_id] = sum - degrees[warp_id] * (*d_dot_val) * inv_2m;
    }
}

__global__ void neg_axpy_kernel(const float* __restrict__ alpha,
                                 const float* __restrict__ x,
                                 float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] -= (*alpha) * x[i];
}

__global__ void neg_axpy2_kernel(const float* __restrict__ alpha,
                                  const float* __restrict__ x1,
                                  const float* __restrict__ beta,
                                  const float* __restrict__ x2,
                                  float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] -= (*alpha) * x1[i] + (*beta) * x2[i];
}

__global__ void inv_scale_kernel(const float* __restrict__ beta, float* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float b = *beta;
        if (b > 1e-10f) x[i] /= b;
    }
}

__global__ void init_lanczos_kernel(float* __restrict__ v, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned int x = (unsigned int)(i + 1);
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        v[i] = (float)(x & 0x7FFFFF) / (float)(0x7FFFFF) - 0.5f;
    }
}

__global__ void gemv_t_kernel(const float* __restrict__ V, int n, int ncols, int lda,
                               const float* __restrict__ w, float* __restrict__ h) {
    int col = blockIdx.x;
    if (col >= ncols) return;

    const float* v_col = V + (int64_t)col * lda;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += v_col[i] * w[i];

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    __shared__ float sdata[32];
    int warp = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    if (lane == 0) sdata[warp] = sum;
    __syncthreads();

    if (threadIdx.x < 32) {
        int nwarps = (blockDim.x + 31) / 32;
        sum = (threadIdx.x < nwarps) ? sdata[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (threadIdx.x == 0) h[col] = sum;
    }
}

__global__ void gemv_n_sub_kernel(const float* __restrict__ V, int n, int ncols, int lda,
                                   const float* __restrict__ h, float* __restrict__ w) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int j = 0; j < ncols; j++)
            sum += V[i + (int64_t)j * lda] * h[j];
        w[i] -= sum;
    }
}

__global__ void row_normalize_kernel(float* __restrict__ Y, int n, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float norm = 0.0f;
        for (int d = 0; d < dim; d++) {
            float val = Y[i + (size_t)d * n];
            norm += val * val;
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            float inv_norm = 1.0f / norm;
            for (int d = 0; d < dim; d++)
                Y[i + (size_t)d * n] *= inv_norm;
        }
    }
}

__global__ void kmeans_assign_kernel(const float* __restrict__ data,
                                      const float* __restrict__ centroids,
                                      int32_t* __restrict__ assignments,
                                      int n, int k, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float min_dist = CUDART_INF_F;
        int32_t best = 0;
        for (int c = 0; c < k; c++) {
            float dist = 0.0f;
            for (int d = 0; d < dim; d++) {
                float diff = data[i + d * n] - centroids[c * dim + d];
                dist += diff * diff;
            }
            if (dist < min_dist) { min_dist = dist; best = c; }
        }
        assignments[i] = best;
    }
}

__global__ void kmeans_accumulate_kernel(const float* __restrict__ data,
                                          const int32_t* __restrict__ assignments,
                                          float* __restrict__ centroids,
                                          int32_t* __restrict__ counts,
                                          int n, int k, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int32_t c = assignments[i];
        atomicAdd(&counts[c], 1);
        for (int d = 0; d < dim; d++)
            atomicAdd(&centroids[c * dim + d], data[i + d * n]);
    }
}

__global__ void kmeans_divide_kernel(float* __restrict__ centroids,
                                      const int32_t* __restrict__ counts,
                                      int k, int dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k * dim) {
        int c = idx / dim;
        if (counts[c] > 0) centroids[idx] /= (float)counts[c];
    }
}

__global__ void count_changes_kernel(const int32_t* __restrict__ old_assign,
                                      const int32_t* __restrict__ new_assign,
                                      int32_t* __restrict__ count, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && old_assign[i] != new_assign[i]) atomicAdd(count, 1);
}





static void launch_compute_degrees(const int32_t* offsets, const float* weights, float* degrees, int n) {
    compute_degrees_kernel<<<(n+255)/256, 256>>>(offsets, weights, degrees, n);
}

static void launch_fused_spmv_mod(const int32_t* offsets, const int32_t* indices, const float* values,
                                   const float* degrees, float inv_2m, const float* x, float* y, int n,
                                   const float* d_dot_val) {
    int warps_needed = n;
    int threads_per_block = 256;
    int warps_per_block = threads_per_block / 32;
    int grid = (warps_needed + warps_per_block - 1) / warps_per_block;
    fused_spmv_mod_kernel<<<grid, threads_per_block>>>(offsets, indices, values, degrees, inv_2m, x, y, n, d_dot_val);
}

static void launch_neg_axpy(const float* alpha, const float* x, float* y, int n) {
    neg_axpy_kernel<<<(n+255)/256, 256>>>(alpha, x, y, n);
}

static void launch_neg_axpy2(const float* alpha, const float* x1, const float* beta, const float* x2, float* y, int n) {
    neg_axpy2_kernel<<<(n+255)/256, 256>>>(alpha, x1, beta, x2, y, n);
}

static void launch_inv_scale(const float* beta, float* x, int n) {
    inv_scale_kernel<<<(n+255)/256, 256>>>(beta, x, n);
}

static void launch_init_lanczos(float* v, int n) {
    init_lanczos_kernel<<<(n+255)/256, 256>>>(v, n);
}

static void launch_gemv_t(const float* V, int n, int ncols, int lda, const float* w, float* h) {
    gemv_t_kernel<<<ncols, 256>>>(V, n, ncols, lda, w, h);
}

static void launch_gemv_n_sub(const float* V, int n, int ncols, int lda, const float* h, float* w) {
    gemv_n_sub_kernel<<<(n+255)/256, 256>>>(V, n, ncols, lda, h, w);
}

static void launch_row_normalize(float* Y, int n, int dim) {
    row_normalize_kernel<<<(n+255)/256, 256>>>(Y, n, dim);
}

static void launch_kmeans_assign(const float* data, const float* centroids, int32_t* assignments, int n, int k, int dim) {
    kmeans_assign_kernel<<<(n+255)/256, 256>>>(data, centroids, assignments, n, k, dim);
}

static void launch_kmeans_accumulate(const float* data, const int32_t* assignments,
                                      float* centroids, int32_t* counts, int n, int k, int dim) {
    kmeans_accumulate_kernel<<<(n+255)/256, 256>>>(data, assignments, centroids, counts, n, k, dim);
}

static void launch_kmeans_divide(float* centroids, const int32_t* counts, int k, int dim) {
    kmeans_divide_kernel<<<(k*dim+255)/256, 256>>>(centroids, counts, k, dim);
}

static void launch_count_changes(const int32_t* old_assign, const int32_t* new_assign, int32_t* count, int n) {
    count_changes_kernel<<<(n+255)/256, 256>>>(old_assign, new_assign, count, n);
}





static void solve_tridiagonal_eigen(int m, const float* alpha_arr, const float* beta_arr,
                                     std::vector<double>& eigenvalues, std::vector<double>& eigenvectors) {
    eigenvalues.resize(m);
    eigenvectors.resize((size_t)m * m);

    std::vector<double> d(m), e(m, 0.0);
    for (int i = 0; i < m; i++) d[i] = (double)alpha_arr[i];
    for (int i = 0; i < m - 1; i++) e[i] = (double)beta_arr[i];

    std::fill(eigenvectors.begin(), eigenvectors.end(), 0.0);
    for (int i = 0; i < m; i++) eigenvectors[i + (size_t)i * m] = 1.0;

    const double eps = 1e-15;
    for (int l = 0; l < m; l++) {
        int iter = 0; int mm;
        do {
            for (mm = l; mm < m - 1; mm++) {
                double dd = std::abs(d[mm]) + std::abs(d[mm + 1]);
                if (std::abs(e[mm]) <= eps * dd) break;
            }
            if (mm != l) {
                if (++iter > 300) break;
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = std::sqrt(g * g + 1.0);
                g = d[mm] - d[l] + e[l] / (g + (g >= 0.0 ? r : -r));
                double s_val = 1.0, c_val = 1.0, p = 0.0;
                int i;
                for (i = mm - 1; i >= l; i--) {
                    double f = s_val * e[i], b = c_val * e[i];
                    if (std::abs(f) >= std::abs(g)) {
                        c_val = g / f; r = std::sqrt(c_val * c_val + 1.0);
                        e[i + 1] = f * r; s_val = 1.0 / r; c_val *= s_val;
                    } else {
                        s_val = f / g; r = std::sqrt(s_val * s_val + 1.0);
                        e[i + 1] = g * r; c_val = 1.0 / r; s_val *= c_val;
                    }
                    g = d[i + 1] - p; r = (d[i] - g) * s_val + 2.0 * c_val * b;
                    p = s_val * r; d[i + 1] = g + p; g = c_val * r - b;
                    for (int k = 0; k < m; k++) {
                        double fk = eigenvectors[k + (size_t)(i + 1) * m];
                        eigenvectors[k + (size_t)(i + 1) * m] = s_val * eigenvectors[k + (size_t)i * m] + c_val * fk;
                        eigenvectors[k + (size_t)i * m] = c_val * eigenvectors[k + (size_t)i * m] - s_val * fk;
                    }
                }
                if (r < eps && i >= l) continue;
                d[l] -= p; e[l] = g; e[mm] = 0.0;
            }
        } while (mm != l);
    }
    for (int i = 0; i < m; i++) eigenvalues[i] = d[i];

    std::vector<int> perm(m);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int a, int b) { return eigenvalues[a] > eigenvalues[b]; });
    std::vector<double> se(m), sv((size_t)m*m);
    for (int i = 0; i < m; i++) {
        se[i] = eigenvalues[perm[i]];
        for (int k = 0; k < m; k++) sv[k + (size_t)i*m] = eigenvectors[k + (size_t)perm[i]*m];
    }
    eigenvalues = se; eigenvectors = sv;
}

static bool check_lanczos_convergence(int m, const float* h_alpha, const float* h_beta,
                                       int num_eigs, float tol) {
    if (m < num_eigs + 1) return false;
    std::vector<double> ev, evec;
    solve_tridiagonal_eigen(m, h_alpha, h_beta, ev, evec);
    float beta_m = h_beta[m - 1];
    for (int i = 0; i < num_eigs; i++) {
        double last_comp = evec[(m - 1) + (size_t)i * m];
        double residual = std::abs(beta_m * last_comp);
        double eigenval = std::abs(ev[i]);
        if (eigenval < 1e-10) eigenval = 1e-10;
        if (residual > tol * eigenval) return false;
    }
    return true;
}

static void kmeans_pp_init(const std::vector<float>& Y_host, int n, int dim, int k,
                            std::vector<float>& centroids) {
    centroids.resize((size_t)k * dim);
    double max_norm = -1.0; int best_idx = 0;
    for (int i = 0; i < n; i++) {
        double norm = 0.0;
        for (int d = 0; d < dim; d++) { double v = Y_host[i + (size_t)d*n]; norm += v*v; }
        if (norm > max_norm) { max_norm = norm; best_idx = i; }
    }
    for (int d = 0; d < dim; d++) centroids[d] = Y_host[best_idx + (size_t)d*n];

    std::vector<double> min_dists(n, 1e30);
    for (int c = 1; c < k; c++) {
        for (int i = 0; i < n; i++) {
            double dist = 0.0;
            for (int d = 0; d < dim; d++) {
                double diff = Y_host[i + (size_t)d*n] - centroids[(c-1)*dim + d];
                dist += diff * diff;
            }
            if (dist < min_dists[i]) min_dists[i] = dist;
        }
        double mx = -1.0; int fi = 0;
        for (int i = 0; i < n; i++) if (min_dists[i] > mx) { mx = min_dists[i]; fi = i; }
        for (int d = 0; d < dim; d++) centroids[c*dim + d] = Y_host[fi + (size_t)d*n];
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

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n = graph.number_of_vertices;
    const float* d_weights = edge_weights;
    int32_t num_eigs = num_eigenvectors;

    
    float* d_deg = nullptr;
    cudaMalloc(&d_deg, (size_t)n * sizeof(float));
    launch_compute_degrees(d_offsets, d_weights, d_deg, n);

    float* d_sum = nullptr;
    cudaMalloc(&d_sum, sizeof(float));
    cublasSasum(cache.cublas, n, d_deg, 1, d_sum);
    float total_deg;
    cudaMemcpy(&total_deg, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_sum);
    float inv_2m = 1.0f / total_deg;

    
    int maxL = std::min((int)evs_max_iter, n - 1);
    maxL = std::max(maxL, 2 * num_eigs + 1);

    float* d_V = nullptr;
    cudaMalloc(&d_V, (size_t)(maxL + 1) * n * sizeof(float));
    float* d_alpha = nullptr;
    cudaMalloc(&d_alpha, (size_t)maxL * sizeof(float));
    float* d_beta = nullptr;
    cudaMalloc(&d_beta, (size_t)maxL * sizeof(float));
    float* d_dot = nullptr;
    cudaMalloc(&d_dot, sizeof(float));
    float* d_h = nullptr;
    cudaMalloc(&d_h, (size_t)(maxL + 1) * sizeof(float));
    float* d_nrm = nullptr;
    cudaMalloc(&d_nrm, sizeof(float));

    launch_init_lanczos(d_V, n);
    cublasSnrm2(cache.cublas, n, d_V, 1, d_nrm);
    launch_inv_scale(d_nrm, d_V, n);

    int actual_iter = 0;
    bool converged = false;
    std::vector<float> h_alpha(maxL, 0.0f), h_beta(maxL, 0.0f);

    for (int j = 0; j < maxL; j++) {
        float* vj = d_V + (int64_t)j * n;
        float* vj1 = d_V + (int64_t)(j + 1) * n;

        
        cublasSdot(cache.cublas, n, d_deg, 1, vj, 1, d_dot);
        launch_fused_spmv_mod(d_offsets, d_indices, d_weights, d_deg, inv_2m, vj, vj1, n, d_dot);

        
        cublasSdot(cache.cublas, n, vj, 1, vj1, 1, d_alpha + j);

        
        if (j > 0) {
            launch_neg_axpy2(d_alpha + j, vj, d_beta + j - 1, d_V + (int64_t)(j-1)*n, vj1, n);
        } else {
            launch_neg_axpy(d_alpha + j, vj, vj1, n);
        }

        
        int ncols = j + 1;
        for (int pass = 0; pass < 2; pass++) {
            launch_gemv_t(d_V, n, ncols, n, vj1, d_h);
            launch_gemv_n_sub(d_V, n, ncols, n, d_h, vj1);
        }

        
        cublasSnrm2(cache.cublas, n, vj1, 1, d_beta + j);
        actual_iter = j + 1;
        launch_inv_scale(d_beta + j, vj1, n);

        
        if (actual_iter >= 2 * num_eigs && actual_iter % 5 == 0) {
            cudaMemcpy(h_alpha.data(), d_alpha, actual_iter * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_beta.data(), d_beta, actual_iter * sizeof(float), cudaMemcpyDeviceToHost);
            if (h_beta[j] < 1e-7f ||
                check_lanczos_convergence(actual_iter, h_alpha.data(), h_beta.data(), num_eigs, evs_tolerance)) {
                converged = true;
                break;
            }
        }
    }

    
    int m = actual_iter;
    if (!converged) {
        cudaMemcpy(h_alpha.data(), d_alpha, m * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_beta.data(), d_beta, m * sizeof(float), cudaMemcpyDeviceToHost);
    }
    std::vector<double> eigenvalues, eigenvectors_cpu;
    solve_tridiagonal_eigen(m, h_alpha.data(), h_beta.data(), eigenvalues, eigenvectors_cpu);

    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_dot);
    cudaFree(d_h);
    cudaFree(d_nrm);

    
    std::vector<float> S_host((size_t)m * num_eigs);
    for (int c = 0; c < num_eigs; c++)
        for (int r = 0; r < m; r++)
            S_host[r + (size_t)c*m] = (float)eigenvectors_cpu[r + (size_t)c*m];

    float* d_S = nullptr;
    cudaMalloc(&d_S, (size_t)m * num_eigs * sizeof(float));
    cudaMemcpy(d_S, S_host.data(), (size_t)m * num_eigs * sizeof(float), cudaMemcpyHostToDevice);

    float* d_Y = nullptr;
    cudaMalloc(&d_Y, (size_t)n * num_eigs * sizeof(float));
    cublasSgemm(cache.cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, num_eigs, m,
                 cache.d_one, d_V, n, d_S, m, cache.d_zero, d_Y, n);

    cudaFree(d_S);
    cudaFree(d_V);

    launch_row_normalize(d_Y, n, num_eigs);

    
    std::vector<float> Y_host((size_t)n * num_eigs);
    cudaMemcpy(Y_host.data(), d_Y, (size_t)n * num_eigs * sizeof(float), cudaMemcpyDeviceToHost);
    std::vector<float> init_c;
    kmeans_pp_init(Y_host, n, num_eigs, num_clusters, init_c);

    float* d_cent = nullptr;
    cudaMalloc(&d_cent, (size_t)num_clusters * num_eigs * sizeof(float));
    cudaMemcpy(d_cent, init_c.data(), (size_t)num_clusters * num_eigs * sizeof(float), cudaMemcpyHostToDevice);

    int32_t* d_asgn = nullptr;
    cudaMalloc(&d_asgn, (size_t)n * sizeof(int32_t));
    int32_t* d_old = nullptr;
    cudaMalloc(&d_old, (size_t)n * sizeof(int32_t));
    int32_t* d_cnt = nullptr;
    cudaMalloc(&d_cnt, (size_t)num_clusters * sizeof(int32_t));
    int32_t* d_chg = nullptr;
    cudaMalloc(&d_chg, sizeof(int32_t));

    cudaMemset(d_asgn, 0, n * sizeof(int32_t));
    for (int it = 0; it < kmean_max_iter; it++) {
        cudaMemcpy(d_old, d_asgn, n * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        launch_kmeans_assign(d_Y, d_cent, d_asgn, n, num_clusters, num_eigs);
        if (it >= 2 && it % 2 == 0) {
            cudaMemset(d_chg, 0, sizeof(int32_t));
            launch_count_changes(d_old, d_asgn, d_chg, n);
            int32_t changes;
            cudaMemcpy(&changes, d_chg, sizeof(int32_t), cudaMemcpyDeviceToHost);
            if (changes == 0) break;
        }
        cudaMemset(d_cent, 0, (size_t)num_clusters * num_eigs * sizeof(float));
        cudaMemset(d_cnt, 0, (size_t)num_clusters * sizeof(int32_t));
        launch_kmeans_accumulate(d_Y, d_asgn, d_cent, d_cnt, n, num_clusters, num_eigs);
        launch_kmeans_divide(d_cent, d_cnt, num_clusters, num_eigs);
    }

    cudaMemcpy(clustering, d_asgn, n * sizeof(int32_t), cudaMemcpyDeviceToDevice);

    
    cudaFree(d_deg);
    cudaFree(d_Y);
    cudaFree(d_cent);
    cudaFree(d_asgn);
    cudaFree(d_old);
    cudaFree(d_cnt);
    cudaFree(d_chg);
}

}  
