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
    const int32_t* __restrict__ offsets, const float* __restrict__ weights,
    float* __restrict__ degrees, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float d=0; for(int j=offsets[i];j<offsets[i+1];j++) d+=weights[j]; degrees[i]=d; }
}

__global__ void fill_tridiag_kernel(
    float* __restrict__ T, const float* __restrict__ alpha,
    const float* __restrict__ beta, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m*m) {
        int row=idx%m, col=idx/m; float val=0;
        if(row==col) val=alpha[row];
        else if(row==col+1 && col<m-1) val=beta[col];
        else if(col==row+1 && row<m-1) val=beta[row];
        T[idx]=val;
    }
}

__global__ void axpy_device_scalar_kernel(
    float* __restrict__ w, const float* __restrict__ vec,
    const float* __restrict__ d_coeff, float divisor, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) w[i] += (-(*d_coeff) / divisor) * vec[i];
}

__global__ void lanczos_update_kernel(
    float* __restrict__ w, const float* __restrict__ qj,
    const float* __restrict__ qjm1, const float* d_alpha,
    const float* d_beta, int n, bool has_prev) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float wi = w[i] - (*d_alpha) * qj[i];
        if (has_prev) wi -= (*d_beta) * qjm1[i];
        w[i] = wi;
    }
}

__global__ void copy_and_scale_kernel(
    float* __restrict__ dst, const float* __restrict__ src,
    const float* __restrict__ d_norm, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] / (*d_norm);
}

__global__ void init_constants_kernel(float* buf) {
    buf[0] = 1.0f; buf[1] = 0.0f; buf[2] = -1.0f;
}

__global__ void kmeans_assign_kernel(
    const float* __restrict__ V, const float* __restrict__ C,
    int32_t* __restrict__ asgn, int n, int d, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float bd=1e30f; int bc=0;
        for(int c=0;c<k;c++){float dist=0;for(int j=0;j<d;j++){float df=V[j*n+i]-C[c*d+j];dist+=df*df;}
        if(dist<bd){bd=dist;bc=c;}} asgn[i]=bc;
    }
}

__global__ void kmeans_min_dist_kernel(
    const float* __restrict__ V, const float* __restrict__ C,
    float* __restrict__ md, int n, int d, int ns) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float m=1e30f;
        for(int c=0;c<ns;c++){float dist=0;for(int j=0;j<d;j++){float df=V[j*n+i]-C[c*d+j];dist+=df*df;}
        if(dist<m) m=dist;} md[i]=m;
    }
}

__global__ void extract_centroid_kernel(
    const float* __restrict__ V, float* __restrict__ C,
    int ci, int pi, int n, int d) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < d) C[ci*d+j] = V[j*n+pi];
}

__global__ void compute_centroid_sums_kernel(
    const float* __restrict__ V, const int32_t* __restrict__ asgn,
    float* __restrict__ sums, int* __restrict__ counts, int n, int d, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { int c=asgn[i]; atomicAdd(&counts[c],1);
        for(int j=0;j<d;j++) atomicAdd(&sums[c*d+j],V[j*n+i]); }
}

__global__ void normalize_centroids_kernel(
    float* __restrict__ C, const float* __restrict__ sums,
    const int* __restrict__ counts, int k, int d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k*d) { int c=idx/d; if(counts[c]>0) C[idx]=sums[idx]/(float)counts[c]; }
}

__global__ void compute_objective_kernel(
    const float* __restrict__ V, const float* __restrict__ C,
    const int32_t* __restrict__ asgn, float* __restrict__ costs, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { int c=asgn[i]; float dist=0;
        for(int j=0;j<d;j++){float df=V[j*n+i]-C[c*d+j];dist+=df*df;} costs[i]=dist; }
}

__global__ void count_changed_kernel(
    const int32_t* __restrict__ o, const int32_t* __restrict__ ne,
    int* __restrict__ changed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && o[i] != ne[i]) atomicAdd(changed, 1);
}


struct BumpAlloc {
    float* base; size_t offset;
    BumpAlloc(float* b, size_t) : base(b), offset(0) {}
    float* alloc_floats(size_t c) { float* p=base+offset; offset+=c; return p; }
    int32_t* alloc_int32(size_t c) { return (int32_t*)alloc_floats(c); }
    int* alloc_int(size_t c) { return (int*)alloc_floats(c); }
    void* alloc_bytes(size_t b) { return (void*)alloc_floats((b+3)/4); }
};


struct Cache : Cacheable {
    cublasHandle_t cublas_h = nullptr;
    cusparseHandle_t cusparse_h = nullptr;
    cusolverDnHandle_t cusolver_h = nullptr;

    float* scratch = nullptr;
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

    void ensure_scratch(size_t total) {
        if (scratch_capacity < total) {
            if (scratch) cudaFree(scratch);
            cudaMalloc(&scratch, total * sizeof(float));
            scratch_capacity = total;
        }
    }
};

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

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const float* d_weights = edge_weights;

    int n = num_vertices, nev = num_eigenvectors, k = num_clusters;
    int blk = 256, grid = (n+blk-1)/blk;
    int max_lanczos = evs_max_iter;
    if (max_lanczos > n-1) max_lanczos = n-1;

    
    size_t total = 3                                           
        + (size_t)n                                            
        + (size_t)n * max_lanczos                              
        + (size_t)n                                            
        + max_lanczos                                          
        + 2 * max_lanczos                                      
        + 1                                                    
        + (size_t)max_lanczos * max_lanczos                    
        + max_lanczos                                          
        + (size_t)max_lanczos * max_lanczos * 4 + 1            
        + (size_t)n * nev                                      
        + (size_t)k * nev                                      
        + (size_t)n                                            
        + (size_t)k * nev                                      
        + (size_t)n                                            
        + k                                                    
        + (size_t)n                                            
        + 1                                                    
        + (size_t)n                                            
        + 524288;                                              

    cache.ensure_scratch(total);

    BumpAlloc ba(cache.scratch, total);

    float* d_consts = ba.alloc_floats(3);
    init_constants_kernel<<<1,1>>>(d_consts);
    float *d_one=d_consts, *d_zero=d_consts+1, *d_neg_one=d_consts+2;

    float* d_deg = ba.alloc_floats(n);
    float* d_Q = ba.alloc_floats((size_t)n * max_lanczos);
    float* d_w = ba.alloc_floats(n);
    float* d_h = ba.alloc_floats(max_lanczos);
    float* d_alpha_arr = ba.alloc_floats(max_lanczos);
    float* d_beta_arr = ba.alloc_floats(max_lanczos);
    float* d_dot_kq = ba.alloc_floats(1);

    
    compute_degrees_kernel<<<grid, blk>>>(d_offsets, d_weights, d_deg, n);
    float two_m;
    cublasSetPointerMode(cache.cublas_h, CUBLAS_POINTER_MODE_HOST);
    cublasSasum(cache.cublas_h, n, d_deg, 1, &two_m);

    
    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rng, 42);
    curandGenerateUniform(rng, d_Q, n);
    curandDestroyGenerator(rng);

    float nrm;
    cublasSnrm2(cache.cublas_h, n, d_Q, 1, &nrm);
    float inv_nrm = 1.0f/nrm;
    cublasSscal(cache.cublas_h, n, &inv_nrm, d_Q, 1);

    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, n, n, num_edges,
        (void*)d_offsets, (void*)d_indices, (void*)d_weights,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, n, (void*)d_Q, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, n, (void*)d_w, CUDA_R_32F);

    float alpha_sp=1.0f, beta_sp=0.0f;
    size_t spmv_buf_size=0;
    cusparseSpMV_bufferSize(cache.cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_sp, matA, vecX, &beta_sp, vecY, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &spmv_buf_size);
    void* d_spmv_buf = ba.alloc_bytes(spmv_buf_size > 0 ? spmv_buf_size : 8);

    cublasSetPointerMode(cache.cublas_h, CUBLAS_POINTER_MODE_DEVICE);

    int m = 0;
    int check_interval = 5;
    int min_steps = nev + 10;
    if (min_steps < 15) min_steps = 15;
    std::vector<double> prev_ritz(nev, -1e300);

    for (int j = 0; j < max_lanczos; j++) {
        float* qj = d_Q + (size_t)j * n;

        
        cusparseDnVecSetValues(vecX, (void*)qj);
        cusparseDnVecSetValues(vecY, (void*)d_w);
        cusparseSpMV(cache.cusparse_h, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_sp, matA, vecX, &beta_sp, vecY,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_buf);

        
        cublasSdot(cache.cublas_h, n, d_deg, 1, qj, 1, d_dot_kq);
        axpy_device_scalar_kernel<<<grid, blk>>>(d_w, d_deg, d_dot_kq, two_m, n);

        
        cublasSdot(cache.cublas_h, n, qj, 1, d_w, 1, d_alpha_arr + j);
        
        lanczos_update_kernel<<<grid, blk>>>(d_w, qj,
            j>0 ? d_Q+(size_t)(j-1)*n : nullptr,
            d_alpha_arr+j, j>0 ? d_beta_arr+(j-1) : nullptr, n, j>0);

        
        int cols = j+1;
        cublasSgemv(cache.cublas_h, CUBLAS_OP_T, n, cols, d_one, d_Q, n, d_w, 1, d_zero, d_h, 1);
        cublasSgemv(cache.cublas_h, CUBLAS_OP_N, n, cols, d_neg_one, d_Q, n, d_h, 1, d_one, d_w, 1);

        
        cublasSnrm2(cache.cublas_h, n, d_w, 1, d_beta_arr + j);
        m = j+1;

        
        if (j+1 < max_lanczos)
            copy_and_scale_kernel<<<grid, blk>>>(d_Q+(size_t)(j+1)*n, d_w, d_beta_arr+j, n);

        
        if (m >= min_steps && (m % check_interval == 0 || j+1 == max_lanczos)) {
            std::vector<float> h_alpha(m), h_beta(m);
            cudaMemcpy(h_alpha.data(), d_alpha_arr, m*sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_beta.data(), d_beta_arr, m*sizeof(float), cudaMemcpyDeviceToHost);

            float beta_m = h_beta[m-1];
            if (beta_m < 1e-10f) break;

            
            std::vector<double> d_eig(m), e_eig(m);
            for (int i = 0; i < m; i++) d_eig[i] = h_alpha[i];
            for (int i = 0; i < m; i++) e_eig[i] = h_beta[i];
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
                if (fabs((double)beta_m) > evs_tolerance * scale_r) {
                    conv_by_residual = false;
                    break;
                }
            }

            if (conv_by_eigenvalue || conv_by_residual) break;
        }
    }

    cublasSetPointerMode(cache.cublas_h, CUBLAS_POINTER_MODE_HOST);
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);

    
    float* d_T = ba.alloc_floats((size_t)m*m);
    float* d_eigenvalues = ba.alloc_floats(m);
    fill_tridiag_kernel<<<(m*m+blk-1)/blk, blk>>>(d_T, d_alpha_arr, d_beta_arr, m);

    int lwork;
    cusolverDnSsyevd_bufferSize(cache.cusolver_h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        m, d_T, m, d_eigenvalues, &lwork);
    float* d_work = ba.alloc_floats(lwork);
    int* d_info = ba.alloc_int(1);
    cusolverDnSsyevd(cache.cusolver_h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        m, d_T, m, d_eigenvalues, d_work, lwork, d_info);
    cudaDeviceSynchronize();

    
    int actual_nev = (nev<=m) ? nev : m;
    float* d_V = ba.alloc_floats((size_t)n * actual_nev);
    float one_h=1.0f, zero_h=0.0f;
    cublasSgemm(cache.cublas_h, CUBLAS_OP_N, CUBLAS_OP_N,
        n, actual_nev, m, &one_h, d_Q, n, d_T+(size_t)(m-actual_nev)*m, m, &zero_h, d_V, n);

    
    int d_dim = actual_nev;
    float* d_centroids = ba.alloc_floats(k*d_dim);
    float* d_min_dists = ba.alloc_floats(n);
    float* d_centroid_sums = ba.alloc_floats(k*d_dim);
    float* d_point_costs = ba.alloc_floats(n);
    int* d_counts = ba.alloc_int(k);
    int32_t* d_old_asgn = ba.alloc_int32(n);
    int* d_changed = ba.alloc_int(1);
    int32_t* d_best_clustering = ba.alloc_int32(n);

    std::vector<float> h_min_dists(n);
    float best_cost = 1e30f;

    for (int restart = 0; restart < 2; restart++) {
        srand(42 + restart*1000);
        int first_idx = rand() % n;
        extract_centroid_kernel<<<1,32>>>(d_V, d_centroids, 0, first_idx, n, d_dim);
        cudaDeviceSynchronize();

        for (int ci = 1; ci < k; ci++) {
            kmeans_min_dist_kernel<<<grid,blk>>>(d_V, d_centroids, d_min_dists, n, d_dim, ci);
            cudaMemcpy(h_min_dists.data(), d_min_dists, n*sizeof(float), cudaMemcpyDeviceToHost);
            double total_dist=0; for(int i=0;i<n;i++) total_dist+=h_min_dists[i];
            double r=((double)rand()/RAND_MAX)*total_dist, cumsum=0; int sel=n-1;
            for(int i=0;i<n;i++){cumsum+=h_min_dists[i];if(cumsum>=r){sel=i;break;}}
            extract_centroid_kernel<<<1,32>>>(d_V, d_centroids, ci, sel, n, d_dim);
            cudaDeviceSynchronize();
        }

        kmeans_assign_kernel<<<grid,blk>>>(d_V, d_centroids, clustering, n, d_dim, k);
        cudaDeviceSynchronize();

        for (int iter = 0; iter < kmean_max_iter; iter++) {
            cudaMemcpy(d_old_asgn, clustering, n*sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemset(d_centroid_sums, 0, k*d_dim*sizeof(float));
            cudaMemset(d_counts, 0, k*sizeof(int));
            compute_centroid_sums_kernel<<<grid,blk>>>(d_V, clustering, d_centroid_sums, d_counts, n, d_dim, k);
            cudaDeviceSynchronize();
            normalize_centroids_kernel<<<(k*d_dim+blk-1)/blk,blk>>>(d_centroids, d_centroid_sums, d_counts, k, d_dim);
            kmeans_assign_kernel<<<grid,blk>>>(d_V, d_centroids, clustering, n, d_dim, k);
            cudaMemset(d_changed, 0, sizeof(int));
            count_changed_kernel<<<grid,blk>>>(d_old_asgn, clustering, d_changed, n);
            int h_changed;
            cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
            if (h_changed==0 || (float)h_changed/n < kmean_tolerance) break;
        }

        compute_objective_kernel<<<grid,blk>>>(d_V, d_centroids, clustering, d_point_costs, n, d_dim);
        float cost;
        cublasSasum(cache.cublas_h, n, d_point_costs, 1, &cost);
        if (cost < best_cost) {
            best_cost = cost;
            cudaMemcpy(d_best_clustering, clustering, n*sizeof(int32_t), cudaMemcpyDeviceToDevice);
        }
    }
    cudaMemcpy(clustering, d_best_clustering, n*sizeof(int32_t), cudaMemcpyDeviceToDevice);
}

}  
