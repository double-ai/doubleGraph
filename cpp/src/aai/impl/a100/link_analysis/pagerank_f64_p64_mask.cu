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
#include <cub/cub.cuh>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace aai {

namespace {



__device__ __forceinline__ bool is_edge_active(const uint32_t* mask, int32_t e) {
    return (mask[e >> 5] >> (e & 31)) & 1;
}

__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_counts,
    int32_t num_vertices
) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v <= num_vertices) {
        if (v < num_vertices) {
            int32_t count = 0;
            int32_t start = offsets[v];
            int32_t end = offsets[v + 1];
            for (int32_t e = start; e < end; e++) {
                if (is_edge_active(edge_mask, e)) count++;
            }
            active_counts[v] = count;
        } else {
            active_counts[num_vertices] = 0;
        }
    }
}

__global__ void compact_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    double* __restrict__ new_weights,
    int32_t num_vertices
) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int32_t old_start = offsets[v];
        int32_t old_end = offsets[v + 1];
        int32_t new_pos = new_offsets[v];
        for (int32_t e = old_start; e < old_end; e++) {
            if (is_edge_active(edge_mask, e)) {
                new_indices[new_pos] = indices[e];
                new_weights[new_pos] = weights[e];
                new_pos++;
            }
        }
    }
}

__global__ void compute_out_weights_kernel(
    const int32_t* __restrict__ new_offsets,
    const int32_t* __restrict__ new_indices,
    const double* __restrict__ new_weights,
    double* __restrict__ out_weight,
    int32_t num_vertices
) {
    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int32_t start = new_offsets[v];
        int32_t end = new_offsets[v + 1];
        for (int32_t e = start; e < end; e++) {
            atomicAdd(&out_weight[new_indices[e]], new_weights[e]);
        }
    }
}

__global__ void build_pers_norm_kernel(
    const int32_t* __restrict__ pers_vertices,
    const double* __restrict__ pers_values,
    double* __restrict__ pers_norm,
    int32_t pers_size,
    double pers_sum_inv
) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pers_size) {
        pers_norm[pers_vertices[i]] = pers_values[i] * pers_sum_inv;
    }
}

__global__ void init_pr_kernel(double* pr, int32_t num_vertices, double init_val) {
    int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_vertices) {
        pr[i] = init_val;
    }
}

__global__ void compute_x_and_dangling_kernel(
    const double* __restrict__ pr,
    const double* __restrict__ out_weight,
    double* __restrict__ x,
    double* __restrict__ d_dangling_sum,
    int32_t num_vertices
) {
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    double my_dangling = 0.0;

    if (tid < num_vertices) {
        double ow = out_weight[tid];
        double p = pr[tid];
        if (ow > 0.0) {
            x[tid] = p / ow;
        } else {
            x[tid] = 0.0;
            my_dangling = p;
        }
    }

    double block_sum = BlockReduce(temp).Sum(my_dangling);
    if (threadIdx.x == 0 && block_sum != 0.0) {
        atomicAdd(d_dangling_sum, block_sum);
    }
}

__global__ void add_base_and_diff_kernel(
    double* __restrict__ pr_new,
    const double* __restrict__ pr_old,
    const double* __restrict__ pers_norm,
    const double* __restrict__ d_dangling_sum,
    double alpha,
    double one_minus_alpha,
    double* __restrict__ d_diff,
    int32_t num_vertices
) {
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int32_t v = blockIdx.x * blockDim.x + threadIdx.x;
    double local_diff = 0.0;

    if (v < num_vertices) {
        double base = alpha * (*d_dangling_sum) + one_minus_alpha;
        double new_val = pr_new[v] + base * pers_norm[v];
        pr_new[v] = new_val;
        local_diff = fabs(new_val - pr_old[v]);
    }

    double block_diff = BlockReduce(temp).Sum(local_diff);
    if (threadIdx.x == 0 && block_diff != 0.0) {
        atomicAdd(d_diff, block_diff);
    }
}



void launch_count_active_edges(const int32_t* offsets, const uint32_t* edge_mask,
    int32_t* active_counts, int32_t num_vertices, cudaStream_t stream) {
    int block = 256;
    int grid = ((num_vertices + 1) + block - 1) / block;
    count_active_edges_kernel<<<grid, block, 0, stream>>>(
        offsets, edge_mask, active_counts, num_vertices);
}

void launch_compact_edges(const int32_t* offsets, const int32_t* indices,
    const double* weights, const uint32_t* edge_mask,
    const int32_t* new_offsets, int32_t* new_indices, double* new_weights,
    int32_t num_vertices, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    compact_edges_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, weights, edge_mask, new_offsets, new_indices,
        new_weights, num_vertices);
}

void launch_compute_out_weights(const int32_t* new_offsets, const int32_t* new_indices,
    const double* new_weights, double* out_weight, int32_t num_vertices, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    compute_out_weights_kernel<<<grid, block, 0, stream>>>(
        new_offsets, new_indices, new_weights, out_weight, num_vertices);
}

void launch_build_pers_norm(const int32_t* pers_vertices, const double* pers_values,
    double* pers_norm, int32_t pers_size, double pers_sum_inv, cudaStream_t stream) {
    if (pers_size == 0) return;
    int block = 256;
    int grid = (pers_size + block - 1) / block;
    build_pers_norm_kernel<<<grid, block, 0, stream>>>(
        pers_vertices, pers_values, pers_norm, pers_size, pers_sum_inv);
}

void launch_init_pr(double* pr, int32_t num_vertices, double init_val, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    init_pr_kernel<<<grid, block, 0, stream>>>(pr, num_vertices, init_val);
}

void launch_compute_x_and_dangling(const double* pr, const double* out_weight,
    double* x, double* d_dangling_sum, int32_t num_vertices, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    compute_x_and_dangling_kernel<<<grid, block, 0, stream>>>(
        pr, out_weight, x, d_dangling_sum, num_vertices);
}

void launch_add_base_and_diff(double* pr_new, const double* pr_old,
    const double* pers_norm, const double* d_dangling_sum,
    double alpha, double one_minus_alpha, double* d_diff,
    int32_t num_vertices, cudaStream_t stream) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    add_base_and_diff_kernel<<<grid, block, 0, stream>>>(
        pr_new, pr_old, pers_norm, d_dangling_sum, alpha, one_minus_alpha,
        d_diff, num_vertices);
}



struct Cache : Cacheable {
    cusparseHandle_t cusparse_handle = nullptr;

    
    int32_t* active_counts = nullptr;
    int64_t active_counts_cap = 0;

    int32_t* new_offsets = nullptr;
    int64_t new_offsets_cap = 0;

    double* out_weight = nullptr;
    int64_t out_weight_cap = 0;

    double* pers_norm = nullptr;
    int64_t pers_norm_cap = 0;

    double* pr_alt = nullptr;
    int64_t pr_alt_cap = 0;

    double* x_buf = nullptr;
    int64_t x_buf_cap = 0;

    
    double* scalars = nullptr;

    
    int32_t* compact_indices = nullptr;
    int64_t compact_indices_cap = 0;

    double* compact_weights = nullptr;
    int64_t compact_weights_cap = 0;

    void* cub_temp = nullptr;
    size_t cub_temp_cap = 0;

    void* spmv_buf = nullptr;
    size_t spmv_buf_cap = 0;

    Cache() {
        cusparseCreate(&cusparse_handle);
        cudaMalloc(&scalars, 4 * sizeof(double));
    }

    ~Cache() override {
        if (cusparse_handle) cusparseDestroy(cusparse_handle);
        if (active_counts) cudaFree(active_counts);
        if (new_offsets) cudaFree(new_offsets);
        if (out_weight) cudaFree(out_weight);
        if (pers_norm) cudaFree(pers_norm);
        if (pr_alt) cudaFree(pr_alt);
        if (x_buf) cudaFree(x_buf);
        if (scalars) cudaFree(scalars);
        if (compact_indices) cudaFree(compact_indices);
        if (compact_weights) cudaFree(compact_weights);
        if (cub_temp) cudaFree(cub_temp);
        if (spmv_buf) cudaFree(spmv_buf);
    }

    void ensure_vertex_buffers(int32_t num_vertices) {
        int64_t nv = num_vertices;
        int64_t nv1 = num_vertices + 1;
        if (active_counts_cap < nv1) {
            if (active_counts) cudaFree(active_counts);
            cudaMalloc(&active_counts, nv1 * sizeof(int32_t));
            active_counts_cap = nv1;
        }
        if (new_offsets_cap < nv1) {
            if (new_offsets) cudaFree(new_offsets);
            cudaMalloc(&new_offsets, nv1 * sizeof(int32_t));
            new_offsets_cap = nv1;
        }
        if (out_weight_cap < nv) {
            if (out_weight) cudaFree(out_weight);
            cudaMalloc(&out_weight, nv * sizeof(double));
            out_weight_cap = nv;
        }
        if (pers_norm_cap < nv) {
            if (pers_norm) cudaFree(pers_norm);
            cudaMalloc(&pers_norm, nv * sizeof(double));
            pers_norm_cap = nv;
        }
        if (pr_alt_cap < nv) {
            if (pr_alt) cudaFree(pr_alt);
            cudaMalloc(&pr_alt, nv * sizeof(double));
            pr_alt_cap = nv;
        }
        if (x_buf_cap < nv) {
            if (x_buf) cudaFree(x_buf);
            cudaMalloc(&x_buf, nv * sizeof(double));
            x_buf_cap = nv;
        }
    }

    void ensure_compact_buffers(int32_t num_edges) {
        int64_t ne = num_edges;
        if (compact_indices_cap < ne) {
            if (compact_indices) cudaFree(compact_indices);
            cudaMalloc(&compact_indices, ne * sizeof(int32_t));
            compact_indices_cap = ne;
        }
        if (compact_weights_cap < ne) {
            if (compact_weights) cudaFree(compact_weights);
            cudaMalloc(&compact_weights, ne * sizeof(double));
            compact_weights_cap = ne;
        }
    }

    void ensure_cub_temp(size_t bytes) {
        if (cub_temp_cap < bytes) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, bytes);
            cub_temp_cap = bytes;
        }
    }

    void ensure_spmv_buf(size_t bytes) {
        if (spmv_buf_cap < bytes) {
            if (spmv_buf) cudaFree(spmv_buf);
            cudaMalloc(&spmv_buf, bytes);
            spmv_buf_cap = bytes;
        }
    }
};

}  

PageRankResult personalized_pagerank_mask(const graph32_t& graph,
                                          const double* edge_weights,
                                          const int32_t* personalization_vertices,
                                          const double* personalization_values,
                                          std::size_t personalization_size,
                                          double* pageranks,
                                          const double* precomputed_vertex_out_weight_sums,
                                          double alpha,
                                          double epsilon,
                                          std::size_t max_iterations,
                                          const double* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    const uint32_t* d_edge_mask = graph.edge_mask;
    int32_t pers_size = static_cast<int32_t>(personalization_size);

    cudaStream_t stream = 0;
    cusparseSetStream(cache.cusparse_handle, stream);

    cache.ensure_vertex_buffers(num_vertices);

    
    launch_count_active_edges(d_offsets, d_edge_mask, cache.active_counts, num_vertices, stream);

    size_t cub_temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, cub_temp_bytes,
        (int32_t*)nullptr, (int32_t*)nullptr, num_vertices + 1);
    cache.ensure_cub_temp(cub_temp_bytes + 16);
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, cub_temp_bytes,
        cache.active_counts, cache.new_offsets, num_vertices + 1, stream);

    int32_t num_active_edges;
    cudaMemcpyAsync(&num_active_edges, &cache.new_offsets[num_vertices], sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int32_t alloc_edges = (num_active_edges > 0) ? num_active_edges : 1;
    cache.ensure_compact_buffers(alloc_edges);

    launch_compact_edges(d_offsets, d_indices, edge_weights, d_edge_mask,
        cache.new_offsets, cache.compact_indices, cache.compact_weights, num_vertices, stream);

    
    cudaMemsetAsync(cache.out_weight, 0, num_vertices * sizeof(double), stream);
    launch_compute_out_weights(cache.new_offsets, cache.compact_indices, cache.compact_weights,
        cache.out_weight, num_vertices, stream);

    
    cudaMemsetAsync(cache.pers_norm, 0, num_vertices * sizeof(double), stream);

    double h_pers_sum = 0.0;
    if (pers_size > 0) {
        std::vector<double> h_pers(pers_size);
        cudaMemcpyAsync(h_pers.data(), personalization_values, pers_size * sizeof(double),
            cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (int i = 0; i < pers_size; i++) h_pers_sum += h_pers[i];
    }

    double pers_sum_inv = (h_pers_sum > 0.0) ? (1.0 / h_pers_sum) : 0.0;
    launch_build_pers_norm(personalization_vertices, personalization_values, cache.pers_norm,
        pers_size, pers_sum_inv, stream);

    
    cusparseSpMatDescr_t mat_descr = nullptr;
    cusparseDnVecDescr_t x_descr = nullptr, y_descr = nullptr;

    
    double* d_pr[2] = {pageranks, cache.pr_alt};
    double* d_x = cache.x_buf;
    double* d_dangling_sum = cache.scalars;
    double* d_diff = cache.scalars + 1;
    double* d_alpha_scalar = cache.scalars + 2;
    double* d_zero_scalar = cache.scalars + 3;

    
    double h_alpha_scalar = alpha;
    double h_zero = 0.0;
    cudaMemcpyAsync(d_alpha_scalar, &h_alpha_scalar, sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_zero_scalar, &h_zero, sizeof(double), cudaMemcpyHostToDevice, stream);

    if (num_active_edges > 0) {
        cusparseCreateCsr(&mat_descr,
            (int64_t)num_vertices, (int64_t)num_vertices, (int64_t)num_active_edges,
            cache.new_offsets, cache.compact_indices, cache.compact_weights,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        cusparseCreateDnVec(&x_descr, (int64_t)num_vertices, d_x, CUDA_R_64F);
        cusparseCreateDnVec(&y_descr, (int64_t)num_vertices, d_pr[1], CUDA_R_64F);

        cusparseSetPointerMode(cache.cusparse_handle, CUSPARSE_POINTER_MODE_DEVICE);
    }

    
    size_t spmv_buffer_size = 0;
    if (num_active_edges > 0) {
        cusparseSpMV_bufferSize(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_alpha_scalar, mat_descr, x_descr,
            d_zero_scalar, y_descr,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            &spmv_buffer_size);

        cache.ensure_spmv_buf(spmv_buffer_size + 16);

        cusparseSpMV_preprocess(cache.cusparse_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            d_alpha_scalar, mat_descr, x_descr,
            d_zero_scalar, y_descr,
            CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
            cache.spmv_buf);

        
        int cur = 0;
        if (initial_pageranks != nullptr) {
            cudaMemcpyAsync(d_pr[cur], initial_pageranks,
                num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
        } else {
            launch_init_pr(d_pr[cur], num_vertices, 1.0 / num_vertices, stream);
        }

        
        double one_minus_alpha = 1.0 - alpha;
        size_t iterations = 0;
        bool converged = false;
        double h_diff;
        const int CHECK_INTERVAL = 10;

        for (size_t iter = 0; iter < max_iterations; iter++) {
            int next = 1 - cur;

            cudaMemsetAsync(d_dangling_sum, 0, 2 * sizeof(double), stream);

            launch_compute_x_and_dangling(d_pr[cur], cache.out_weight, d_x, d_dangling_sum,
                num_vertices, stream);

            
            cusparseDnVecSetValues(y_descr, d_pr[next]);
            cusparseSpMV(cache.cusparse_handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                d_alpha_scalar, mat_descr, x_descr,
                d_zero_scalar, y_descr,
                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                cache.spmv_buf);

            
            launch_add_base_and_diff(d_pr[next], d_pr[cur], cache.pers_norm, d_dangling_sum,
                alpha, one_minus_alpha, d_diff, num_vertices, stream);

            cur = next;
            iterations = iter + 1;

            
            bool should_check = (iterations % CHECK_INTERVAL == 0) || (iterations == max_iterations);
            if (should_check) {
                cudaMemcpyAsync(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                if (h_diff < epsilon) {
                    converged = true;
                    break;
                }
            }
        }

        
        if (cur != 0) {
            cudaMemcpyAsync(pageranks, d_pr[cur],
                num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
            cudaStreamSynchronize(stream);
        }

        
        cusparseDestroyDnVec(x_descr);
        cusparseDestroyDnVec(y_descr);
        cusparseDestroySpMat(mat_descr);

        return PageRankResult{iterations, converged};
    } else {
        
        int cur = 0;
        if (initial_pageranks != nullptr) {
            cudaMemcpyAsync(d_pr[cur], initial_pageranks,
                num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
        } else {
            launch_init_pr(d_pr[cur], num_vertices, 1.0 / num_vertices, stream);
        }

        double one_minus_alpha = 1.0 - alpha;
        size_t iterations = 0;
        bool converged = false;
        double h_diff;

        for (size_t iter = 0; iter < max_iterations; iter++) {
            int next = 1 - cur;
            cudaMemsetAsync(d_dangling_sum, 0, 2 * sizeof(double), stream);
            launch_compute_x_and_dangling(d_pr[cur], cache.out_weight, d_x, d_dangling_sum,
                num_vertices, stream);
            
            cudaMemsetAsync(d_pr[next], 0, num_vertices * sizeof(double), stream);
            launch_add_base_and_diff(d_pr[next], d_pr[cur], cache.pers_norm, d_dangling_sum,
                alpha, one_minus_alpha, d_diff, num_vertices, stream);
            cur = next;
            iterations = iter + 1;
            cudaMemcpyAsync(&h_diff, d_diff, sizeof(double), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (h_diff < epsilon) { converged = true; break; }
        }
        if (cur != 0) {
            cudaMemcpyAsync(pageranks, d_pr[cur],
                num_vertices * sizeof(double), cudaMemcpyDeviceToDevice, stream);
            cudaStreamSynchronize(stream);
        }
        return PageRankResult{iterations, converged};
    }
}

}  
