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
#include <cusparse.h>
#include <cusolverDn.h>
#include <curand.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

namespace aai {

namespace {





static void tqli_eigenvalues_only(std::vector<double>& d, std::vector<double>& e, int n) {
    for (int i = 1; i < n; i++) e[i-1] = e[i];
    e[n-1] = 0.0;
    for (int l = 0; l < n; l++) {
        int iter = 0, m;
        do {
            for (m = l; m < n-1; m++) {
                double dd = fabs(d[m]) + fabs(d[m+1]);
                if (fabs(e[m]) + dd == dd) break;
            }
            if (m != l) {
                if (++iter > 30) return;
                double g = (d[l+1] - d[l]) / (2.0 * e[l]);
                double r = hypot(g, 1.0);
                g = d[m] - d[l] + e[l] / (g + copysign(r, g));
                double s = 1.0, c = 1.0, p = 0.0;
                int i;
                for (i = m-1; i >= l; i--) {
                    double f = s * e[i], b = c * e[i];
                    r = hypot(f, g);
                    e[i+1] = r;
                    if (r == 0.0) { d[i+1] -= p; e[m] = 0.0; break; }
                    s = f / r; c = g / r;
                    g = d[i+1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    d[i+1] = g + (p = s * r);
                    g = c * r - b;
                }
                if (r == 0.0 && i >= l) continue;
                d[l] -= p; e[l] = g; e[m] = 0.0;
            }
        } while (m != l);
    }
    std::sort(d.begin(), d.begin() + n);
}




__global__ void compute_degrees_kernel(
    const int32_t* __restrict__ offsets, const double* __restrict__ weights,
    double* __restrict__ degrees, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { double d=0; for(int j=offsets[i];j<offsets[i+1];j++) d+=weights[j]; degrees[i]=d; }
}

__global__ void fill_tridiag_kernel(
    double* __restrict__ T, const double* __restrict__ alpha,
    const double* __restrict__ beta, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m*m) {
        int row=idx%m, col=idx/m; double val=0;
        if(row==col) val=alpha[row];
        else if(row==col+1 && col<m-1) val=beta[col];
        else if(col==row+1 && row<m-1) val=beta[row];
        T[idx]=val;
    }
}

__global__ void axpy_device_scalar_kernel(
    double* __restrict__ w, const double* __restrict__ vec,
    const double* __restrict__ d_coeff, double divisor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] += (-(*d_coeff) / divisor) * vec[i];
}

__global__ void lanczos_update_kernel(
    double* __restrict__ w, const double* __restrict__ qj,
    const double* __restrict__ qjm1, const double* d_alpha,
    const double* d_beta, int n, bool has_prev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double wi = w[i] - (*d_alpha) * qj[i];
        if (has_prev) wi -= (*d_beta) * qjm1[i];
        w[i] = wi;
    }
}

__global__ void copy_and_scale_kernel(
    double* __restrict__ dst, const double* __restrict__ src,
    const double* __restrict__ d_norm, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] / (*d_norm);
}

__global__ void init_constants_kernel(double* buf) {
    buf[0] = 1.0; buf[1] = 0.0; buf[2] = -1.0;
}

__global__ void kmeans_assign_kernel(
    const double* __restrict__ V, const double* __restrict__ C,
    int32_t* __restrict__ asgn, int n, int d, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double bd=1e300; int bc=0;
        for(int c=0;c<k;c++){double dist=0;for(int j=0;j<d;j++){double df=V[j*n+i]-C[c*d+j];dist+=df*df;}
        if(dist<bd){bd=dist;bc=c;}} asgn[i]=bc;
    }
}

__global__ void kmeans_min_dist_kernel(
    const double* __restrict__ V, const double* __restrict__ C,
    double* __restrict__ md, int n, int d, int ns) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double m=1e300;
        for(int c=0;c<ns;c++){double dist=0;for(int j=0;j<d;j++){double df=V[j*n+i]-C[c*d+j];dist+=df*df;}
        if(dist<m) m=dist;} md[i]=m;
    }
}

__global__ void extract_centroid_kernel(
    const double* __restrict__ V, double* __restrict__ C,
    int ci, int pi, int n, int d) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < d) C[ci*d+j] = V[j*n+pi];
}

__global__ void compute_centroid_sums_kernel(
    const double* __restrict__ V, const int32_t* __restrict__ asgn,
    double* __restrict__ sums, int* __restrict__ counts, int n, int d, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { int c=asgn[i]; atomicAdd(&counts[c],1);
        for(int j=0;j<d;j++) atomicAdd(&sums[c*d+j],V[j*n+i]); }
}

__global__ void normalize_centroids_kernel(
    double* __restrict__ C, const double* __restrict__ sums,
    const int* __restrict__ counts, int k, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k*d) { int c=idx/d; if(counts[c]>0) C[idx]=sums[idx]/(double)counts[c]; }
}

__global__ void compute_objective_kernel(
    const double* __restrict__ V, const double* __restrict__ C,
    const int32_t* __restrict__ asgn, double* __restrict__ costs, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { int c=asgn[i]; double dist=0;
        for(int j=0;j<d;j++){double df=V[j*n+i]-C[c*d+j];dist+=df*df;} costs[i]=dist; }
}

__global__ void count_changed_kernel(
    const int32_t* __restrict__ o, const int32_t* __restrict__ ne,
    int* __restrict__ changed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && o[i] != ne[i]) atomicAdd(changed, 1);
}


struct BumpAlloc {
    double* base; size_t offset;
    BumpAlloc(double* b, size_t) : base(b), offset(0) {}
    double* alloc_doubles(size_t c) { double* p=base+offset; offset+=c; return p; }
    int32_t* alloc_int32(size_t c) { return (int32_t*)alloc_doubles((c*4+7)/8); }
    int* alloc_int(size_t c) { return (int*)alloc_doubles((c*4+7)/8); }
    void* alloc_bytes(size_t b) { return (void*)alloc_doubles((b+7)/8); }
};


struct Cache : Cacheable {
    cublasHandle_t cublas_h = nullptr;
    cusparseHandle_t cusparse_h = nullptr;
    cusolverDnHandle_t cusolver_h = nullptr;
    double* scratch = nullptr;
    size_t scratch_capacity = 0;

    Cache() {
        cublasCreate(&cublas_h);
        cusparseCreate(&cusparse_h);
        cusolverDnCreate(&cusolver_h);
    }

    ~Cache() override {
        if (cublas_h) cublasDestroy(cublas_h);
        if (cusparse_h) cusparseDestroy(cusparse_h);
        if (cusolver_h) cusolverDnDestroy(cusolver_h);
        if (scratch) cudaFree(scratch);
    }

    void ensure_scratch(size_t needed) {
        if (scratch_capacity < needed) {
            if (scratch) cudaFree(scratch);
            cudaMalloc(&scratch, needed * sizeof(double));
            scratch_capacity = needed;
        }
    }
};


static void spectral_mod_max_impl(
    cublasHandle_t cublas_h, cusparseHandle_t cusparse_h, cusolverDnHandle_t cusolver_h,
    const int32_t* d_offsets, const int32_t* d_indices, const double* d_weights,
    int32_t num_vertices, int32_t num_edges,
    int32_t num_clusters, int32_t num_eigenvectors,
    double evs_tolerance, int32_t evs_max_iter,
    double kmean_tolerance, int32_t kmean_max_iter,
    int32_t* d_clustering, double* d_scratch, size_t scratch_size
) {
    int n = num_vertices, nev = num_eigenvectors, k = num_clusters;
    int blk = 256, grid = (n+blk-1)/blk;
    int max_lanczos = evs_max_iter;
    if (max_lanczos > n-1) max_lanczos = n-1;

    BumpAlloc ba(d_scratch, scratch_size);

    double* d_consts = ba.alloc_doubles(3);
    init_constants_kernel<<<1,1>>>(d_consts);
    double *d_one=d_consts, *d_zero=d_consts+1, *d_neg_one=d_consts+2;

    double* d_deg = ba.alloc_doubles(n);
    double* d_Q = ba.alloc_doubles((size_t)n * max_lanczos);
    double* d_w = ba.alloc_doubles(n);
    double* d_h = ba.alloc_doubles(max_lanczos);
    double* d_alpha_arr = ba.alloc_doubles(max_lanczos);
    double* d_beta_arr = ba.alloc_doubles(max_lanczos);
    double* d_dot_kq = ba.alloc_doubles(1);

    
    compute_degrees_kernel<<<grid, blk>>>(d_offsets, d_weights, d_deg, n);
    double two_m;
    cublasSetPointerMode(cublas_h, CUBLAS_POINTER_MODE_HOST);
    cublasDasum(cublas_h, n, d_deg, 1, &two_m);

    
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rng, 42);
    curandGenerateUniformDouble(rng, d_Q, n);
    curandDestroyGenerator(rng);

    double nrm;
    cublasDnrm2(cublas_h, n, d_Q, 1, &nrm);
    double inv_nrm = 1.0/nrm;
    cublasDscal(cublas_h, n, &inv_nrm, d_Q, 1);

    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, n, n, num_edges,
        (void*)d_offsets, (void*)d_indices, (void*)d_weights,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, n, (void*)d_Q, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, n, (void*)d_w, CUDA_R_64F);

    double alpha_sp=1.0, beta_sp=0.0;
    size_t spmv_buf_size=0;
    cusparseSpMV_bufferSize(cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_sp, matA, vecX, &beta_sp, vecY, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buf_size);
    void* d_spmv_buf = ba.alloc_bytes(spmv_buf_size > 0 ? spmv_buf_size : 8);

    cublasSetPointerMode(cublas_h, CUBLAS_POINTER_MODE_DEVICE);

    int m = 0;
    
    int check_interval = 5;
    int min_steps = nev + 10;  
    if (min_steps < 15) min_steps = 15;
    std::vector<double> prev_ritz(nev, -1e300);

    for (int j = 0; j < max_lanczos; j++) {
        double* qj = d_Q + (size_t)j * n;

        cusparseDnVecSetValues(vecX, (void*)qj);
        cusparseDnVecSetValues(vecY, (void*)d_w);
        cusparseSpMV(cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_sp, matA, vecX, &beta_sp, vecY,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buf);

        cublasDdot(cublas_h, n, d_deg, 1, qj, 1, d_dot_kq);
        axpy_device_scalar_kernel<<<grid, blk>>>(d_w, d_deg, d_dot_kq, two_m, n);

        cublasDdot(cublas_h, n, qj, 1, d_w, 1, d_alpha_arr + j);
        lanczos_update_kernel<<<grid, blk>>>(d_w, qj,
            j>0 ? d_Q+(size_t)(j-1)*n : nullptr,
            d_alpha_arr+j, j>0 ? d_beta_arr+(j-1) : nullptr, n, j>0);

        int cols = j+1;
        cublasDgemv(cublas_h, CUBLAS_OP_T, n, cols, d_one, d_Q, n, d_w, 1, d_zero, d_h, 1);
        cublasDgemv(cublas_h, CUBLAS_OP_N, n, cols, d_neg_one, d_Q, n, d_h, 1, d_one, d_w, 1);

        cublasDnrm2(cublas_h, n, d_w, 1, d_beta_arr + j);
        m = j+1;

        if (j+1 < max_lanczos)
            copy_and_scale_kernel<<<grid, blk>>>(d_Q+(size_t)(j+1)*n, d_w, d_beta_arr+j, n);

        
        if (m >= min_steps && (m % check_interval == 0 || j+1 == max_lanczos)) {
            
            std::vector<double> h_alpha(m), h_beta(m);
            cudaMemcpy(h_alpha.data(), d_alpha_arr, m*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_beta.data(), d_beta_arr, m*sizeof(double), cudaMemcpyDeviceToHost);

            double beta_m = h_beta[m-1];
            
            if (beta_m < 1e-14) break;

            
            std::vector<double> d_eig = h_alpha;
            std::vector<double> e_eig = h_beta;
            tqli_eigenvalues_only(d_eig, e_eig, m);

            
            
            bool conv_by_eigenvalue = true;
            for (int i = 0; i < nev; i++) {
                double cur = d_eig[m - 1 - i];
                double prev = prev_ritz[i];
                double scale = fabs(cur) > 1.0 ? fabs(cur) : 1.0;
                if (fabs(cur - prev) > evs_tolerance * scale) {
                    conv_by_eigenvalue = false;
                }
                prev_ritz[i] = cur;
            }

            
            bool conv_by_residual = true;
            for (int i = 0; i < nev; i++) {
                double lambda_i = d_eig[m - 1 - i];
                double scale_r = fabs(lambda_i) > 1.0 ? fabs(lambda_i) : 1.0;
                if (fabs(beta_m) > evs_tolerance * scale_r) {
                    conv_by_residual = false;
                    break;
                }
            }

            if (conv_by_eigenvalue || conv_by_residual) break;
        }
    }

    cublasSetPointerMode(cublas_h, CUBLAS_POINTER_MODE_HOST);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    
    double* d_T = ba.alloc_doubles((size_t)m*m);
    double* d_eigenvalues = ba.alloc_doubles(m);
    fill_tridiag_kernel<<<(m*m+blk-1)/blk, blk>>>(d_T, d_alpha_arr, d_beta_arr, m);

    int lwork;
    cusolverDnDsyevd_bufferSize(cusolver_h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        m, d_T, m, d_eigenvalues, &lwork);
    double* d_work = ba.alloc_doubles(lwork);
    int* d_info = ba.alloc_int(1);
    cusolverDnDsyevd(cusolver_h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        m, d_T, m, d_eigenvalues, d_work, lwork, d_info);
    cudaDeviceSynchronize();

    
    int actual_nev = (nev<=m) ? nev : m;
    double* d_V = ba.alloc_doubles((size_t)n * actual_nev);
    double one=1.0, zero=0.0;
    cublasDgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N,
        n, actual_nev, m, &one, d_Q, n, d_T+(size_t)(m-actual_nev)*m, m, &zero, d_V, n);

    
    int d_dim = actual_nev;
    double* d_centroids = ba.alloc_doubles(k*d_dim);
    double* d_min_dists = ba.alloc_doubles(n);
    double* d_centroid_sums = ba.alloc_doubles(k*d_dim);
    double* d_point_costs = ba.alloc_doubles(n);
    int* d_counts = ba.alloc_int(k);
    int32_t* d_old_asgn = ba.alloc_int32(n);
    int* d_changed = ba.alloc_int(1);
    int32_t* d_best_clustering = ba.alloc_int32(n);

    std::vector<double> h_min_dists(n);
    double best_cost = 1e300;

    for (int restart = 0; restart < 2; restart++) {
        srand(42 + restart*1000);
        int first_idx = rand() % n;
        extract_centroid_kernel<<<1,32>>>(d_V, d_centroids, 0, first_idx, n, d_dim);
        cudaDeviceSynchronize();

        for (int ci = 1; ci < k; ci++) {
            kmeans_min_dist_kernel<<<grid,blk>>>(d_V, d_centroids, d_min_dists, n, d_dim, ci);
            cudaMemcpy(h_min_dists.data(), d_min_dists, n*sizeof(double), cudaMemcpyDeviceToHost);
            double total=0; for(int i=0;i<n;i++) total+=h_min_dists[i];
            double r=((double)rand()/RAND_MAX)*total, cumsum=0; int sel=n-1;
            for(int i=0;i<n;i++){cumsum+=h_min_dists[i];if(cumsum>=r){sel=i;break;}}
            extract_centroid_kernel<<<1,32>>>(d_V, d_centroids, ci, sel, n, d_dim);
            cudaDeviceSynchronize();
        }

        kmeans_assign_kernel<<<grid,blk>>>(d_V, d_centroids, d_clustering, n, d_dim, k);
        cudaDeviceSynchronize();

        for (int iter = 0; iter < kmean_max_iter; iter++) {
            cudaMemcpy(d_old_asgn, d_clustering, n*sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemset(d_centroid_sums, 0, k*d_dim*sizeof(double));
            cudaMemset(d_counts, 0, k*sizeof(int));
            compute_centroid_sums_kernel<<<grid,blk>>>(d_V, d_clustering, d_centroid_sums, d_counts, n, d_dim, k);
            cudaDeviceSynchronize();
            normalize_centroids_kernel<<<(k*d_dim+blk-1)/blk,blk>>>(d_centroids, d_centroid_sums, d_counts, k, d_dim);
            kmeans_assign_kernel<<<grid,blk>>>(d_V, d_centroids, d_clustering, n, d_dim, k);
            cudaMemset(d_changed, 0, sizeof(int));
            count_changed_kernel<<<grid,blk>>>(d_old_asgn, d_clustering, d_changed, n);
            int h_changed;
            cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
            if (h_changed==0 || (double)h_changed/n < kmean_tolerance) break;
        }

        compute_objective_kernel<<<grid,blk>>>(d_V, d_centroids, d_clustering, d_point_costs, n, d_dim);
        double cost;
        cublasDasum(cublas_h, n, d_point_costs, 1, &cost);
        if (cost < best_cost) {
            best_cost = cost;
            cudaMemcpy(d_best_clustering, d_clustering, n*sizeof(int32_t), cudaMemcpyDeviceToDevice);
        }
    }
    cudaMemcpy(d_clustering, d_best_clustering, n*sizeof(int32_t), cudaMemcpyDeviceToDevice);
}

}  

void spectral_modularity_maximization(const graph32_t& graph,
                                      const double* edge_weights,
                                      int32_t num_clusters,
                                      int32_t num_eigenvectors,
                                      double evs_tolerance,
                                      int32_t evs_max_iter,
                                      double kmean_tolerance,
                                      int32_t kmean_max_iter,
                                      int32_t* clustering) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t n = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;

    
    int max_lanczos = evs_max_iter;
    if (max_lanczos > n - 1) max_lanczos = n - 1;
    int k = num_clusters;
    int nev = num_eigenvectors;

    size_t total = (size_t)n * max_lanczos + n + max_lanczos + n
        + (size_t)max_lanczos * max_lanczos + 2 * max_lanczos
        + max_lanczos + (size_t)max_lanczos * max_lanczos * 4 + 1
        + (size_t)n * nev
        + (size_t)k * nev + n + (size_t)k * nev + n
        + 524288
        + (n + k + 2 + n + 15) / 2 + n;

    cache.ensure_scratch(total);

    spectral_mod_max_impl(
        cache.cublas_h, cache.cusparse_h, cache.cusolver_h,
        graph.offsets, graph.indices, edge_weights,
        n, num_edges,
        num_clusters, num_eigenvectors,
        evs_tolerance, evs_max_iter,
        kmean_tolerance, kmean_max_iter,
        clustering, cache.scratch, total
    );
}

}  
