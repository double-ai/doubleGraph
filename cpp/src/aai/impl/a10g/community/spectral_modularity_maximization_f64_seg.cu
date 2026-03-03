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
#include <cusparse.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

namespace aai {

namespace {





static void tqli(std::vector<double>& d, std::vector<double>& e, int n, std::vector<double>& z) {
    
    
    for (int i = 1; i < n; i++) e[i-1] = e[i];
    e[n-1] = 0.0;

    for (int l = 0; l < n; l++) {
        int iter = 0;
        int m;
        do {
            for (m = l; m < n - 1; m++) {
                double dd = std::abs(d[m]) + std::abs(d[m+1]);
                if (std::abs(e[m]) + dd == dd) break;
            }
            if (m != l) {
                if (++iter > 100) break;  
                double g = (d[l+1] - d[l]) / (2.0 * e[l]);
                double r = std::sqrt(g * g + 1.0);
                g = d[m] - d[l] + e[l] / (g + (g >= 0 ? std::abs(r) : -std::abs(r)));
                double s = 1.0, c = 1.0, p = 0.0;
                int i;
                for (i = m - 1; i >= l; i--) {
                    double f = s * e[i];
                    double b = c * e[i];
                    r = std::sqrt(f * f + g * g);
                    e[i+1] = r;
                    if (r == 0.0) {
                        d[i+1] -= p;
                        e[m] = 0.0;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i+1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i+1] = g + p;
                    g = c * r - b;
                    
                    for (int k = 0; k < n; k++) {
                        f = z[k * n + i + 1];
                        z[k * n + i + 1] = s * z[k * n + i] + c * f;
                        z[k * n + i] = c * z[k * n + i] - s * f;
                    }
                }
                if (r == 0.0 && i >= l) continue;
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }
}





__global__ void compute_weighted_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ weights,
    double* __restrict__ degrees,
    int32_t num_vertices)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_vertices) {
        int start = offsets[i];
        int end = offsets[i + 1];
        double sum = 0.0;
        for (int j = start; j < end; j++) sum += weights[j];
        degrees[i] = sum;
    }
}

__global__ void modularity_correction_kernel(
    double* __restrict__ y,
    const double* __restrict__ d,
    const double* d_dot_result,
    double inv_2m, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] -= d[i] * (*d_dot_result) * inv_2m;
}

__global__ void negate_from_array_kernel(double* out, const double* arr, int idx) {
    if (threadIdx.x == 0) *out = -arr[idx];
}

__global__ void reciprocal_scalar_kernel(double* out, const double* in) {
    if (threadIdx.x == 0) *out = 1.0 / (*in);
}

__global__ void kmeans_assign_kernel(
    const double* __restrict__ points,
    const double* __restrict__ centroids,
    int32_t* __restrict__ assignments,
    int n, int d, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double best_dist = 1e30;
        int best_cluster = 0;
        for (int c = 0; c < k; c++) {
            double dist = 0.0;
            for (int j = 0; j < d; j++) {
                double diff = points[i * d + j] - centroids[c * d + j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_cluster = c;
            }
        }
        assignments[i] = best_cluster;
    }
}

__global__ void kmeans_accumulate_kernel(
    const double* __restrict__ points,
    const int32_t* __restrict__ assignments,
    double* __restrict__ centroid_sums,
    int* __restrict__ counts,
    int n, int d, int k)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int c = assignments[i];
        atomicAdd(&counts[c], 1);
        for (int j = 0; j < d; j++)
            atomicAdd(&centroid_sums[c * d + j], points[i * d + j]);
    }
}

__global__ void kmeans_update_centroids_kernel(
    double* __restrict__ centroids,
    const double* __restrict__ centroid_sums,
    const int* __restrict__ counts,
    int k, int d)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = idx / d;
    int j = idx % d;
    if (c < k && j < d) {
        int cnt = counts[c];
        if (cnt > 0) centroids[c * d + j] = centroid_sums[c * d + j] / (double)cnt;
    }
}

__global__ void init_random_vector(double* v, int n, unsigned long long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned long long h = seed ^ ((unsigned long long)i * 6364136223846793005ULL + 1442695040888963407ULL);
        h = h * 6364136223846793005ULL + 1442695040888963407ULL;
        v[i] = (double)(h & 0xFFFFFFFF) / 4294967296.0 - 0.5;
    }
}

__global__ void normalize_rows_kernel(double* data, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double norm = 0.0;
        for (int j = 0; j < d; j++) { double v = data[i * d + j]; norm += v * v; }
        norm = sqrt(norm);
        if (norm > 1e-15) {
            double inv = 1.0 / norm;
            for (int j = 0; j < d; j++) data[i * d + j] *= inv;
        }
    }
}

__global__ void compute_assignment_diff_kernel(
    const int32_t* __restrict__ old_a, const int32_t* __restrict__ new_a,
    int* __restrict__ diff_count, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (old_a[i] != new_a[i]) atomicAdd(diff_count, 1);
    }
}

__global__ void sum_reduce_kernel(const double* __restrict__ data, double* result, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    double sum = 0.0;
    for (int idx = blockIdx.x * blockDim.x + tid; idx < n; idx += blockDim.x * gridDim.x)
        sum += data[idx];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(result, sdata[0]);
}

__global__ void kmeans_pp_min_dist_kernel(
    const double* __restrict__ points,
    const double* __restrict__ latest_centroid,
    double* __restrict__ min_dists,
    int n, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double dist = 0.0;
        for (int j = 0; j < d; j++) {
            double diff = points[i * d + j] - latest_centroid[j];
            dist += diff * diff;
        }
        double cur = min_dists[i];
        if (dist < cur) min_dists[i] = dist;
    }
}

__global__ void argmax_kernel(const double* __restrict__ data, int* __restrict__ result, int n) {
    extern __shared__ char smem[];
    double* sval = (double*)smem;
    int* sidx = (int*)(smem + blockDim.x * sizeof(double));

    int tid = threadIdx.x;
    double best_val = -1e30;
    int best_idx = 0;

    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        if (data[i] > best_val) {
            best_val = data[i];
            best_idx = i;
        }
    }
    sval[tid] = best_val;
    sidx[tid] = best_idx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sval[tid + s] > sval[tid]) {
                sval[tid] = sval[tid + s];
                sidx[tid] = sidx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        result[blockIdx.x] = sidx[0];
    }
}

__global__ void copy_row_kernel(double* dst, const double* src, int idx, int d) {
    int j = threadIdx.x;
    if (j < d) dst[j] = src[idx * d + j];
}





void launch_compute_weighted_degrees(const int32_t* offsets, const double* weights, double* degrees, int32_t nv, cudaStream_t s) {
    int b = 256, g = (nv + b - 1) / b; compute_weighted_degrees_kernel<<<g, b, 0, s>>>(offsets, weights, degrees, nv);
}

void launch_modularity_correction(double* y, const double* d, const double* dot, double inv_2m, int n, cudaStream_t s) {
    int b = 256, g = (n + b - 1) / b; modularity_correction_kernel<<<g, b, 0, s>>>(y, d, dot, inv_2m, n);
}

void launch_negate_from_array(double* out, const double* arr, int idx, cudaStream_t s) {
    negate_from_array_kernel<<<1, 1, 0, s>>>(out, arr, idx);
}

void launch_reciprocal_scalar(double* out, const double* in, cudaStream_t s) {
    reciprocal_scalar_kernel<<<1, 1, 0, s>>>(out, in);
}

void launch_init_random(double* v, int n, unsigned long long seed, cudaStream_t s) {
    int b = 256, g = (n + b - 1) / b; init_random_vector<<<g, b, 0, s>>>(v, n, seed);
}

void launch_kmeans_assign(const double* pts, const double* cen, int32_t* asgn, int n, int d, int k, cudaStream_t s) {
    int b = 256, g = (n + b - 1) / b; kmeans_assign_kernel<<<g, b, 0, s>>>(pts, cen, asgn, n, d, k);
}

void launch_kmeans_accumulate(const double* pts, const int32_t* asgn, double* csums, int* counts, int n, int d, int k, cudaStream_t s) {
    int b = 256, g = (n + b - 1) / b; kmeans_accumulate_kernel<<<g, b, 0, s>>>(pts, asgn, csums, counts, n, d, k);
}

void launch_kmeans_update_centroids(double* cen, const double* csums, const int* counts, int k, int d, cudaStream_t s) {
    int t = k * d, b = 256, g = (t + b - 1) / b; kmeans_update_centroids_kernel<<<g, b, 0, s>>>(cen, csums, counts, k, d);
}

void launch_normalize_rows(double* data, int n, int d, cudaStream_t s) {
    int b = 256, g = (n + b - 1) / b; normalize_rows_kernel<<<g, b, 0, s>>>(data, n, d);
}

void launch_compute_assignment_diff(const int32_t* old_a, const int32_t* new_a, int* diff, int n, cudaStream_t s) {
    int b = 256, g = (n + b - 1) / b; compute_assignment_diff_kernel<<<g, b, 0, s>>>(old_a, new_a, diff, n);
}

void launch_sum_reduce(const double* data, double* result, int n, cudaStream_t s) {
    int b = 256, g = (n + b * 4 - 1) / (b * 4); if (g > 256) g = 256;
    sum_reduce_kernel<<<g, b, b * sizeof(double), s>>>(data, result, n);
}

void launch_kmeans_pp_min_dist(const double* pts, const double* latest_cen, double* min_dists, int n, int d, cudaStream_t s) {
    int b = 256, g = (n + b - 1) / b;
    kmeans_pp_min_dist_kernel<<<g, b, 0, s>>>(pts, latest_cen, min_dists, n, d);
}

void launch_argmax(const double* data, int* result, int n, cudaStream_t s) {
    int b = 256;
    argmax_kernel<<<1, b, b * (sizeof(double) + sizeof(int)), s>>>(data, result, n);
}

void launch_copy_row(double* dst, const double* src, int idx, int d, cudaStream_t s) {
    copy_row_kernel<<<1, d, 0, s>>>(dst, src, idx, d);
}





struct Cache : Cacheable {
    cublasHandle_t cublas = nullptr;
    cusparseHandle_t cusparse = nullptr;

    double* d_dot_result = nullptr;
    double* d_neg_scalar = nullptr;
    double* d_inv_scalar = nullptr;
    double* d_zero = nullptr;
    double* d_one = nullptr;
    double* d_neg_one = nullptr;
    int* d_diff_count = nullptr;
    int* d_argmax_result = nullptr;

    bool initialized = false;

    void init() {
        if (initialized) return;
        cublasCreate(&cublas);
        cusparseCreate(&cusparse);
        cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_DEVICE);

        cudaMalloc(&d_dot_result, sizeof(double));
        cudaMalloc(&d_neg_scalar, sizeof(double));
        cudaMalloc(&d_inv_scalar, sizeof(double));
        cudaMalloc(&d_diff_count, sizeof(int));
        cudaMalloc(&d_argmax_result, sizeof(int));

        cudaMalloc(&d_zero, sizeof(double));
        cudaMalloc(&d_one, sizeof(double));
        cudaMalloc(&d_neg_one, sizeof(double));
        double h0 = 0.0, h1 = 1.0, hn1 = -1.0;
        cudaMemcpy(d_zero, &h0, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_one, &h1, sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_neg_one, &hn1, sizeof(double), cudaMemcpyHostToDevice);

        initialized = true;
    }

    ~Cache() override {
        if (cublas) cublasDestroy(cublas);
        if (cusparse) cusparseDestroy(cusparse);
        cudaFree(d_dot_result);
        cudaFree(d_neg_scalar);
        cudaFree(d_inv_scalar);
        cudaFree(d_zero);
        cudaFree(d_one);
        cudaFree(d_neg_one);
        cudaFree(d_diff_count);
        cudaFree(d_argmax_result);
    }
};





void setup_spmv(cusparseHandle_t cusparse,
                cusparseSpMatDescr_t& matA, cusparseDnVecDescr_t& vecX, cusparseDnVecDescr_t& vecY,
                void*& spmv_buf, size_t& spmv_buf_size,
                const int32_t* d_off, const int32_t* d_idx, const double* d_val,
                int n, int nnz, double* d_x, double* d_y) {
    cusparseCreateCsr(&matA, n, n, nnz,
        (void*)d_off, (void*)d_idx, (void*)d_val,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseCreateDnVec(&vecX, n, d_x, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, n, d_y, CUDA_R_64F);

    double alpha = 1.0, beta = 0.0;
    cusparseSpMV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buf_size);

    spmv_buf = nullptr;
    if (spmv_buf_size > 0) cudaMalloc(&spmv_buf, spmv_buf_size);

    cusparseSpMV_preprocess(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, spmv_buf);
}

void do_spmv(cusparseHandle_t cusparse,
             cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY,
             void* spmv_buf, double* d_x, double* d_y) {
    cusparseDnVecSetValues(vecX, d_x);
    cusparseDnVecSetValues(vecY, d_y);
    double alpha = 1.0, beta = 0.0;
    cusparseSpMV(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, spmv_buf);
}

void modularity_matvec(Cache& cache,
                       cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY,
                       void* spmv_buf,
                       double* d_x, double* d_y, const double* d_deg,
                       double inv_2m, int n, cudaStream_t stream) {
    do_spmv(cache.cusparse, matA, vecX, vecY, spmv_buf, d_x, d_y);
    cublasDdot(cache.cublas, n, d_deg, 1, d_x, 1, cache.d_dot_result);
    launch_modularity_correction(d_y, d_deg, cache.d_dot_result, inv_2m, n, stream);
}

void do_lanczos(Cache& cache,
                cusparseSpMatDescr_t matA, cusparseDnVecDescr_t vecX, cusparseDnVecDescr_t vecY,
                void* spmv_buf,
                const double* d_deg, double inv_2m,
                int n, int num_ev, int max_iter, double evs_tolerance,
                double* d_eigvecs, cudaStream_t stream)
{
    int lanczos_steps = max_iter;
    if (lanczos_steps > n) lanczos_steps = n;
    if (lanczos_steps < num_ev + 2) lanczos_steps = num_ev + 2;

    double* d_V = nullptr;
    double* d_w = nullptr;
    double* d_coeffs = nullptr;
    double* d_alpha = nullptr;
    double* d_beta = nullptr;

    cudaMalloc(&d_V, (size_t)(lanczos_steps + 1) * n * sizeof(double));
    cudaMalloc(&d_w, (size_t)n * sizeof(double));
    cudaMalloc(&d_coeffs, (size_t)lanczos_steps * sizeof(double));
    cudaMalloc(&d_alpha, (size_t)lanczos_steps * sizeof(double));
    cudaMalloc(&d_beta, (size_t)(lanczos_steps + 1) * sizeof(double));

    launch_init_random(d_V, n, 42, stream);
    cublasDnrm2(cache.cublas, n, d_V, 1, cache.d_dot_result);
    launch_reciprocal_scalar(cache.d_inv_scalar, cache.d_dot_result, stream);
    cublasDscal(cache.cublas, n, cache.d_inv_scalar, d_V, 1);
    cudaMemsetAsync(d_beta, 0, sizeof(double), stream);

    int actual_steps = lanczos_steps;
    int min_steps = num_ev + 2;
    for (int j = 0; j < lanczos_steps; j++) {
        double* vj = d_V + (size_t)j * n;
        double* vj1 = d_V + (size_t)(j + 1) * n;

        modularity_matvec(cache, matA, vecX, vecY, spmv_buf, vj, d_w, d_deg, inv_2m, n, stream);

        if (j > 0) {
            launch_negate_from_array(cache.d_neg_scalar, d_beta, j, stream);
            cublasDaxpy(cache.cublas, n, cache.d_neg_scalar, d_V + (size_t)(j-1)*n, 1, d_w, 1);
        }

        cublasDdot(cache.cublas, n, vj, 1, d_w, 1, d_alpha + j);
        launch_negate_from_array(cache.d_neg_scalar, d_alpha, j, stream);
        cublasDaxpy(cache.cublas, n, cache.d_neg_scalar, vj, 1, d_w, 1);

        
        if (j > 0) {
            cublasDgemv(cache.cublas, CUBLAS_OP_T,
                n, j+1, cache.d_one, d_V, n, d_w, 1, cache.d_zero, d_coeffs, 1);
            cublasDgemv(cache.cublas, CUBLAS_OP_N,
                n, j+1, cache.d_neg_one, d_V, n, d_coeffs, 1, cache.d_one, d_w, 1);
        }

        cublasDnrm2(cache.cublas, n, d_w, 1, d_beta + j + 1);
        cudaMemcpyAsync(vj1, d_w, (size_t)n * sizeof(double), cudaMemcpyDeviceToDevice, stream);
        launch_reciprocal_scalar(cache.d_inv_scalar, d_beta + j + 1, stream);
        cublasDscal(cache.cublas, n, cache.d_inv_scalar, vj1, 1);

        
        if (j + 1 >= min_steps && (j + 1) % 5 == 0) {
            double h_beta_j1;
            cudaMemcpy(&h_beta_j1, d_beta + j + 1, sizeof(double), cudaMemcpyDeviceToHost);
            if (std::abs(h_beta_j1) < evs_tolerance || std::isnan(h_beta_j1) || std::isinf(h_beta_j1)) {
                actual_steps = j + 1;
                break;
            }
        }
    }

    
    std::vector<double> h_alpha(actual_steps);
    std::vector<double> h_beta_vec(actual_steps + 1);
    cudaMemcpy(h_alpha.data(), d_alpha, actual_steps * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_beta_vec.data(), d_beta, (actual_steps + 1) * sizeof(double), cudaMemcpyDeviceToHost);

    int m = actual_steps;
    for (int j = 1; j <= actual_steps; j++) {
        if (std::abs(h_beta_vec[j]) < 1e-14 || std::isnan(h_beta_vec[j]) || std::isinf(h_beta_vec[j])) {
            m = j - 1; break;
        }
    }
    int clean_m = m;
    if (m < num_ev + 1) m = num_ev + 1;
    if (m > actual_steps) m = actual_steps;
    
    for (int j = clean_m; j < m; j++) {
        if (std::isnan(h_alpha[j]) || std::isinf(h_alpha[j])) h_alpha[j] = 0.0;
    }
    for (int j = clean_m; j <= m; j++) {
        if (std::isnan(h_beta_vec[j]) || std::isinf(h_beta_vec[j])) h_beta_vec[j] = 0.0;
    }

    
    std::vector<double> diag(m), offdiag(m, 0.0);
    std::vector<double> Z(m * m, 0.0);
    for (int i = 0; i < m; i++) {
        diag[i] = h_alpha[i];
        Z[i * m + i] = 1.0; 
    }
    for (int i = 1; i < m; i++) offdiag[i] = h_beta_vec[i];

    tqli(diag, offdiag, m, Z);

    
    std::vector<int> idx(m);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return diag[a] > diag[b]; });

    
    std::vector<double> h_Z_sel(m * num_ev);
    for (int e = 0; e < num_ev; e++) {
        int col = idx[e];
        for (int r = 0; r < m; r++)
            h_Z_sel[e * m + r] = Z[r * m + col];
    }

    double* d_Z_sel = nullptr;
    cudaMalloc(&d_Z_sel, (size_t)m * num_ev * sizeof(double));
    cudaMemcpy(d_Z_sel, h_Z_sel.data(), (size_t)m * num_ev * sizeof(double), cudaMemcpyHostToDevice);

    cublasSetPointerMode(cache.cublas, CUBLAS_POINTER_MODE_HOST);
    double one = 1.0, zero = 0.0;
    cublasDgemm(cache.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        n, num_ev, m, &one, d_V, n, d_Z_sel, m, &zero, d_eigvecs, n);
    cublasSetPointerMode(cache.cublas, CUBLAS_POINTER_MODE_DEVICE);

    cudaFree(d_Z_sel);
    cudaFree(d_V);
    cudaFree(d_w);
    cudaFree(d_coeffs);
    cudaFree(d_alpha);
    cudaFree(d_beta);
}

void kmeans_pp_init_gpu(Cache& cache, const double* d_pts, int n, int d, int k,
                        double* d_cen, cudaStream_t stream, int seed)
{
    double* d_min_dists = nullptr;
    cudaMalloc(&d_min_dists, (size_t)n * sizeof(double));

    std::vector<double> h_big(n, 1e30);
    cudaMemcpy(d_min_dists, h_big.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    int first = ((unsigned)seed * 2654435761u) % n;
    launch_copy_row(d_cen, d_pts, first, d, stream);

    for (int c = 1; c < k; c++) {
        launch_kmeans_pp_min_dist(d_pts, d_cen + (c-1) * d, d_min_dists, n, d, stream);
        launch_argmax(d_min_dists, cache.d_argmax_result, n, stream);

        int h_best;
        cudaMemcpy(&h_best, cache.d_argmax_result, sizeof(int), cudaMemcpyDeviceToHost);
        launch_copy_row(d_cen + c * d, d_pts, h_best, d, stream);
    }

    cudaFree(d_min_dists);
}

void do_kmeans(Cache& cache, const double* d_pts, int n, int d, int k,
               double tol, int max_iter, int32_t* d_asgn,
               cudaStream_t stream, int seed = 42)
{
    double* d_cen = nullptr;
    double* d_csums = nullptr;
    int* d_cnt = nullptr;
    int32_t* d_old = nullptr;

    cudaMalloc(&d_cen, (size_t)k * d * sizeof(double));
    cudaMalloc(&d_csums, (size_t)k * d * sizeof(double));
    cudaMalloc(&d_cnt, (size_t)k * sizeof(int));
    cudaMalloc(&d_old, (size_t)n * sizeof(int32_t));

    kmeans_pp_init_gpu(cache, d_pts, n, d, k, d_cen, stream, seed);
    launch_kmeans_assign(d_pts, d_cen, d_asgn, n, d, k, stream);

    int h_diff;
    for (int iter = 0; iter < max_iter; iter++) {
        cudaMemcpyAsync(d_old, d_asgn, n * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);
        cudaMemsetAsync(d_csums, 0, (size_t)k * d * sizeof(double), stream);
        cudaMemsetAsync(d_cnt, 0, k * sizeof(int), stream);

        launch_kmeans_accumulate(d_pts, d_asgn, d_csums, d_cnt, n, d, k, stream);
        launch_kmeans_update_centroids(d_cen, d_csums, d_cnt, k, d, stream);
        launch_kmeans_assign(d_pts, d_cen, d_asgn, n, d, k, stream);

        if ((iter + 1) % 5 == 0 || iter == max_iter - 1) {
            cudaMemsetAsync(cache.d_diff_count, 0, sizeof(int), stream);
            launch_compute_assignment_diff(d_old, d_asgn, cache.d_diff_count, n, stream);
            cudaMemcpy(&h_diff, cache.d_diff_count, sizeof(int), cudaMemcpyDeviceToHost);
            if (h_diff == 0 || (double)h_diff / n < tol) break;
        }
    }

    cudaFree(d_cen);
    cudaFree(d_csums);
    cudaFree(d_cnt);
    cudaFree(d_old);
}

}  

void spectral_modularity_maximization_seg(const graph32_t& graph,
                                          const double* edge_weights,
                                          int32_t num_clusters,
                                          int32_t num_eigenvectors,
                                          double evs_tolerance,
                                          int32_t evs_max_iter,
                                          double kmean_tolerance,
                                          int32_t kmean_max_iter,
                                          int32_t* clustering)
{
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    cache.init();

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t n = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const double* d_wt = edge_weights;

    cudaStream_t stream = 0;

    
    double* d_deg = nullptr;
    cudaMalloc(&d_deg, (size_t)n * sizeof(double));
    launch_compute_weighted_degrees(d_off, d_wt, d_deg, n, stream);

    
    double* d_total = nullptr;
    cudaMalloc(&d_total, sizeof(double));
    cudaMemsetAsync(d_total, 0, sizeof(double), stream);
    launch_sum_reduce(d_deg, d_total, n, stream);
    double total_weight;
    cudaMemcpy(&total_weight, d_total, sizeof(double), cudaMemcpyDeviceToHost);
    double inv_2m = 1.0 / total_weight;
    cudaFree(d_total);

    
    double* d_tmpx = nullptr;
    double* d_tmpy = nullptr;
    cudaMalloc(&d_tmpx, (size_t)n * sizeof(double));
    cudaMalloc(&d_tmpy, (size_t)n * sizeof(double));

    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr;
    cusparseDnVecDescr_t vecY = nullptr;
    void* spmv_buf = nullptr;
    size_t spmv_buf_size = 0;

    setup_spmv(cache.cusparse, matA, vecX, vecY, spmv_buf, spmv_buf_size,
               d_off, d_idx, d_wt, n, num_edges, d_tmpx, d_tmpy);

    
    double* d_eigvecs = nullptr;
    cudaMalloc(&d_eigvecs, (size_t)n * num_eigenvectors * sizeof(double));

    do_lanczos(cache, matA, vecX, vecY, spmv_buf, d_deg, inv_2m,
               n, num_eigenvectors, evs_max_iter, evs_tolerance, d_eigvecs, stream);

    
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    if (spmv_buf) cudaFree(spmv_buf);
    cudaFree(d_tmpx);
    cudaFree(d_tmpy);

    
    double* d_emb = nullptr;
    cudaMalloc(&d_emb, (size_t)n * num_eigenvectors * sizeof(double));

    cublasSetPointerMode(cache.cublas, CUBLAS_POINTER_MODE_HOST);
    double one = 1.0, zero = 0.0;
    cublasDgeam(cache.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        num_eigenvectors, n, &one, d_eigvecs, n, &zero, d_eigvecs, num_eigenvectors, d_emb, num_eigenvectors);
    cublasSetPointerMode(cache.cublas, CUBLAS_POINTER_MODE_DEVICE);

    cudaFree(d_eigvecs);

    
    launch_normalize_rows(d_emb, n, num_eigenvectors, stream);
    do_kmeans(cache, d_emb, n, num_eigenvectors, num_clusters,
              kmean_tolerance, kmean_max_iter, clustering, stream, 42);

    cudaFree(d_emb);
    cudaFree(d_deg);

    cudaDeviceSynchronize();
}

}  
