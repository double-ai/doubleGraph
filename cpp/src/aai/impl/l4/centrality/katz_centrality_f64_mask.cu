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
#include <cub/block/block_reduce.cuh>
#include <cstdint>

namespace aai {

namespace {

namespace cg = cooperative_groups;

static constexpr int BLOCK = 256;

__device__ __forceinline__ bool edge_active(const uint32_t* __restrict__ edge_mask, int e) {
    uint32_t w = __ldg(edge_mask + (e >> 5));
    return (w >> (e & 31)) & 1u;
}

template <bool USE_BETAS, bool WARP_MODE>
__global__ __launch_bounds__(BLOCK)
void katz_coop_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights_d,
    float* __restrict__ weights_f,
    const uint32_t* __restrict__ edge_mask,
    const double* __restrict__ betas_d,
    float* __restrict__ betas_f,
    const double* __restrict__ alpha_d,
    const double* __restrict__ beta_scalar_d,
    const double* __restrict__ epsilon_d,
    const int64_t* __restrict__ max_iters_i64,
    const bool* __restrict__ normalize_b,
    const bool* __restrict__ has_init_b,
    const double* __restrict__ init_centralities_d,
    float* __restrict__ x0,
    float* __restrict__ x1,
    float* __restrict__ scratch_f, 
    int* __restrict__ scratch_i,   
    double* __restrict__ out_centralities,
    int64_t* __restrict__ out_iters,
    bool* __restrict__ out_converged,
    int32_t nv,
    int32_t ne)
{
    cg::grid_group grid = cg::this_grid();

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_in_block = tid >> 5;
    const int warps_per_block = BLOCK / 32;

    
    const float alpha = (float)(*alpha_d);
    const float beta_scalar = (float)(*beta_scalar_d);
    const double eps_in = *epsilon_d;
    float eps = (float)eps_in;
    if (eps <= 0.0f) {
        eps = -1.0f; 
    } else {
        float eff = (float)nv * 1.0e-6f;
        if (eff > eps) eps = eff;
    }

    uint64_t max_iter;
    int64_t mi64 = *max_iters_i64;
    max_iter = (mi64 < 0) ? 0xFFFFFFFFFFFFFFFFull : (uint64_t)mi64;

    const bool do_norm = *normalize_b;
    const bool has_init = *has_init_b;

    
    const int global_tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total_threads = (int)(gridDim.x * blockDim.x);

    
    if constexpr (USE_BETAS) {
        for (int i = global_tid; i < nv; i += total_threads) {
            betas_f[i] = (float)betas_d[i];
        }
    }

    
    if (has_init) {
        for (int i = global_tid; i < nv; i += total_threads) {
            x0[i] = (float)init_centralities_d[i];
        }
    } else {
        for (int i = global_tid; i < nv; i += total_threads) {
            x0[i] = 0.0f;
        }
    }

    
    if (global_tid == 0) {
        scratch_f[0] = 0.0f;
        scratch_f[1] = 0.0f;
        scratch_i[0] = 0;
    }

    grid.sync();

    float* x_old = x0;
    float* x_new = x1;

    uint64_t iter_start = 0;

    
    if (!has_init && max_iter > 0) {
        if constexpr (USE_BETAS) {
            for (int i = global_tid; i < nv; i += total_threads) {
                x_old[i] = betas_f[i];
            }
        } else {
            
            if (beta_scalar == 0.0f) {
                
                if (global_tid == 0) scratch_i[0] = 1; 
            } else {
                for (int i = global_tid; i < nv; i += total_threads) {
                    x_old[i] = beta_scalar;
                }
                
                if (global_tid == 0) {
                    float first_diff = (float)nv * fabsf(beta_scalar);
                    if (first_diff < eps) scratch_i[0] = 1;
                }
            }
        }
        iter_start = 1;
    }

    grid.sync();

    bool converged = (scratch_i[0] != 0);
    
    uint64_t iterations = iter_start;

    
    if (!converged && max_iter > 0) {
        for (uint64_t iter = iter_start; iter < max_iter; ++iter) {
            
            float thread_diff = 0.0f;

            if constexpr (WARP_MODE) {
                const int warp_global = (int)(blockIdx.x * warps_per_block + warp_in_block);
                const int total_warps = (int)(gridDim.x * warps_per_block);

                for (int v = warp_global; v < nv; v += total_warps) {
                    int start = __ldg(offsets + v);
                    int end = __ldg(offsets + v + 1);

                    float sum = 0.0f;
                    const bool convert_weights = (iter == iter_start);

                    for (int e = start + lane; e < end; e += 32) {
                        if (edge_active(edge_mask, e)) {
                            int u = __ldg(indices + e);
                            float xv = __ldg(x_old + u);
                            float w;
                            if (convert_weights) {
                                w = (float)weights_d[e];
                                weights_f[e] = w;
                            } else {
                                w = __ldg(weights_f + e);
                            }
                            sum = fmaf(w, xv, sum);
                        }
                    }

                    
                    #pragma unroll
                    for (int off = 16; off > 0; off >>= 1) {
                        sum += __shfl_down_sync(0xffffffff, sum, off);
                    }

                    if (lane == 0) {
                        float bv;
                        if constexpr (USE_BETAS) {
                            bv = __ldg(betas_f + v);
                        } else {
                            bv = beta_scalar;
                        }
                        float nval = fmaf(alpha, sum, bv);
                        x_new[v] = nval;
                        thread_diff += fabsf(nval - __ldg(x_old + v));
                    }
                }
            } else {
                
                for (int v = global_tid; v < nv; v += total_threads) {
                    int start = __ldg(offsets + v);
                    int end = __ldg(offsets + v + 1);
                    float sum = 0.0f;
                    const bool convert_weights = (iter == iter_start);

                    for (int e = start; e < end; ++e) {
                        if (edge_active(edge_mask, e)) {
                            int u = __ldg(indices + e);
                            float xv = __ldg(x_old + u);
                            float w;
                            if (convert_weights) {
                                w = (float)weights_d[e];
                                weights_f[e] = w;
                            } else {
                                w = __ldg(weights_f + e);
                            }
                            sum = fmaf(w, xv, sum);
                        }
                    }

                    float bv;
                    if constexpr (USE_BETAS) {
                        bv = __ldg(betas_f + v);
                    } else {
                        bv = beta_scalar;
                    }
                    float nval = fmaf(alpha, sum, bv);
                    x_new[v] = nval;
                    thread_diff += fabsf(nval - __ldg(x_old + v));
                }
            }

            using BlockReduce = cub::BlockReduce<float, BLOCK>;
            __shared__ typename BlockReduce::TempStorage temp;
            float block_sum = BlockReduce(temp).Sum(thread_diff);
            if (tid == 0 && block_sum > 0.0f) {
                atomicAdd(scratch_f + 0, block_sum);
            }

            grid.sync();

            if (global_tid == 0) {
                float diff = scratch_f[0];
                scratch_f[0] = 0.0f;
                scratch_i[0] = (diff < eps) ? 1 : 0;
            }

            grid.sync();

            
            float* tmp = x_old;
            x_old = x_new;
            x_new = tmp;
            iterations = iter + 1;

            if (scratch_i[0] != 0) {
                converged = true;
                break;
            }
        }
    }

    
    float inv_norm = 1.0f;
    if (do_norm) {
        
        float local = 0.0f;
        for (int i = global_tid; i < nv; i += total_threads) {
            float v = x_old[i];
            local = fmaf(v, v, local);
        }
        using BlockReduce = cub::BlockReduce<float, BLOCK>;
        __shared__ typename BlockReduce::TempStorage temp2;
        float bs = BlockReduce(temp2).Sum(local);
        if (tid == 0 && bs > 0.0f) atomicAdd(scratch_f + 1, bs);

        grid.sync();
        if (global_tid == 0) {
            
            scratch_f[1] = rsqrtf(scratch_f[1]);
        }
        grid.sync();
        inv_norm = scratch_f[1];
    }

    
    for (int i = global_tid; i < nv; i += total_threads) {
        out_centralities[i] = (double)(x_old[i] * inv_norm);
    }

    if (global_tid == 0) {
        out_iters[0] = (int64_t)iterations;
        out_converged[0] = converged;
    }
}



static inline int get_sms() {
    int dev = 0;
    cudaGetDevice(&dev);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return sms;
}

template <typename Kernel>
static inline int coop_grid_for(Kernel k, int block, int items_per_unit, int32_t n_items) {
    static int blocks_per_sm = -1;
    if (blocks_per_sm < 0) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, k, block, 0);
        if (blocks_per_sm < 1) blocks_per_sm = 1;
    }
    int grid_limit = blocks_per_sm * get_sms();
    int need = (n_items + items_per_unit - 1) / items_per_unit;
    int grid = (need < grid_limit) ? need : grid_limit;
    if (grid < 1) grid = 1;
    return grid;
}

static void launch_katz_coop_warp(
    const int32_t* offsets,
    const int32_t* indices,
    const double* weights_d,
    float* weights_f,
    const uint32_t* edge_mask,
    const double* betas_d,
    float* betas_f,
    const double* alpha_d,
    const double* beta_scalar_d,
    const double* epsilon_d,
    const int64_t* max_iters_i64,
    const bool* normalize_b,
    const bool* has_init_b,
    const double* init_centralities_d,
    float* x0,
    float* x1,
    float* scratch,
    int* scratch_i,
    double* out_centralities,
    int64_t* out_iters,
    bool* out_converged,
    int32_t nv,
    int32_t ne,
    cudaStream_t stream)
{
    dim3 block(BLOCK);

    if (betas_d) {
        auto k = katz_coop_kernel<true, true>;
        int grid = coop_grid_for(k, BLOCK, (BLOCK / 32), nv);
        void* args[] = {(void*)&offsets, (void*)&indices, (void*)&weights_d, (void*)&weights_f,
                        (void*)&edge_mask, (void*)&betas_d, (void*)&betas_f,
                        (void*)&alpha_d, (void*)&beta_scalar_d, (void*)&epsilon_d, (void*)&max_iters_i64,
                        (void*)&normalize_b, (void*)&has_init_b, (void*)&init_centralities_d,
                        (void*)&x0, (void*)&x1, (void*)&scratch, (void*)&scratch_i,
                        (void*)&out_centralities, (void*)&out_iters, (void*)&out_converged,
                        (void*)&nv, (void*)&ne};
        cudaLaunchCooperativeKernel((void*)k, dim3(grid), block, args, 0, stream);
    } else {
        auto k = katz_coop_kernel<false, true>;
        int grid = coop_grid_for(k, BLOCK, (BLOCK / 32), nv);
        void* args[] = {(void*)&offsets, (void*)&indices, (void*)&weights_d, (void*)&weights_f,
                        (void*)&edge_mask, (void*)&betas_d, (void*)&betas_f,
                        (void*)&alpha_d, (void*)&beta_scalar_d, (void*)&epsilon_d, (void*)&max_iters_i64,
                        (void*)&normalize_b, (void*)&has_init_b, (void*)&init_centralities_d,
                        (void*)&x0, (void*)&x1, (void*)&scratch, (void*)&scratch_i,
                        (void*)&out_centralities, (void*)&out_iters, (void*)&out_converged,
                        (void*)&nv, (void*)&ne};
        cudaLaunchCooperativeKernel((void*)k, dim3(grid), block, args, 0, stream);
    }
}

static void launch_katz_coop_thread(
    const int32_t* offsets,
    const int32_t* indices,
    const double* weights_d,
    float* weights_f,
    const uint32_t* edge_mask,
    const double* betas_d,
    float* betas_f,
    const double* alpha_d,
    const double* beta_scalar_d,
    const double* epsilon_d,
    const int64_t* max_iters_i64,
    const bool* normalize_b,
    const bool* has_init_b,
    const double* init_centralities_d,
    float* x0,
    float* x1,
    float* scratch,
    int* scratch_i,
    double* out_centralities,
    int64_t* out_iters,
    bool* out_converged,
    int32_t nv,
    int32_t ne,
    cudaStream_t stream)
{
    dim3 block(BLOCK);

    if (betas_d) {
        auto k = katz_coop_kernel<true, false>;
        int grid = coop_grid_for(k, BLOCK, BLOCK, nv);
        void* args[] = {(void*)&offsets, (void*)&indices, (void*)&weights_d, (void*)&weights_f,
                        (void*)&edge_mask, (void*)&betas_d, (void*)&betas_f,
                        (void*)&alpha_d, (void*)&beta_scalar_d, (void*)&epsilon_d, (void*)&max_iters_i64,
                        (void*)&normalize_b, (void*)&has_init_b, (void*)&init_centralities_d,
                        (void*)&x0, (void*)&x1, (void*)&scratch, (void*)&scratch_i,
                        (void*)&out_centralities, (void*)&out_iters, (void*)&out_converged,
                        (void*)&nv, (void*)&ne};
        cudaLaunchCooperativeKernel((void*)k, dim3(grid), block, args, 0, stream);
    } else {
        auto k = katz_coop_kernel<false, false>;
        int grid = coop_grid_for(k, BLOCK, BLOCK, nv);
        void* args[] = {(void*)&offsets, (void*)&indices, (void*)&weights_d, (void*)&weights_f,
                        (void*)&edge_mask, (void*)&betas_d, (void*)&betas_f,
                        (void*)&alpha_d, (void*)&beta_scalar_d, (void*)&epsilon_d, (void*)&max_iters_i64,
                        (void*)&normalize_b, (void*)&has_init_b, (void*)&init_centralities_d,
                        (void*)&x0, (void*)&x1, (void*)&scratch, (void*)&scratch_i,
                        (void*)&out_centralities, (void*)&out_iters, (void*)&out_converged,
                        (void*)&nv, (void*)&ne};
        cudaLaunchCooperativeKernel((void*)k, dim3(grid), block, args, 0, stream);
    }
}

struct Cache : Cacheable {
    
    float* scratch_f = nullptr;   
    int* scratch_i = nullptr;     

    
    double* d_double_scalars = nullptr;  
    int64_t* d_max_iters = nullptr;
    bool* d_bool_flags = nullptr;        

    
    int64_t* d_out_iters = nullptr;
    bool* d_out_converged = nullptr;

    
    float* weights_f = nullptr;
    int64_t weights_f_capacity = 0;

    float* x0 = nullptr;
    int64_t x0_capacity = 0;

    float* x1 = nullptr;
    int64_t x1_capacity = 0;

    float* betas_f = nullptr;
    int64_t betas_f_capacity = 0;

    Cache() {
        cudaMalloc(&scratch_f, 2 * sizeof(float));
        cudaMalloc(&scratch_i, sizeof(int));
        cudaMalloc(&d_double_scalars, 3 * sizeof(double));
        cudaMalloc(&d_max_iters, sizeof(int64_t));
        cudaMalloc(&d_bool_flags, 2 * sizeof(bool));
        cudaMalloc(&d_out_iters, sizeof(int64_t));
        cudaMalloc(&d_out_converged, sizeof(bool));
    }

    ~Cache() override {
        if (scratch_f) cudaFree(scratch_f);
        if (scratch_i) cudaFree(scratch_i);
        if (d_double_scalars) cudaFree(d_double_scalars);
        if (d_max_iters) cudaFree(d_max_iters);
        if (d_bool_flags) cudaFree(d_bool_flags);
        if (d_out_iters) cudaFree(d_out_iters);
        if (d_out_converged) cudaFree(d_out_converged);
        if (weights_f) cudaFree(weights_f);
        if (x0) cudaFree(x0);
        if (x1) cudaFree(x1);
        if (betas_f) cudaFree(betas_f);
    }

    void ensure_buffers(int32_t nv, int32_t ne) {
        if (weights_f_capacity < ne) {
            if (weights_f) cudaFree(weights_f);
            cudaMalloc(&weights_f, (size_t)ne * sizeof(float));
            weights_f_capacity = ne;
        }
        if (x0_capacity < nv) {
            if (x0) cudaFree(x0);
            cudaMalloc(&x0, (size_t)nv * sizeof(float));
            x0_capacity = nv;
        }
        if (x1_capacity < nv) {
            if (x1) cudaFree(x1);
            cudaMalloc(&x1, (size_t)nv * sizeof(float));
            x1_capacity = nv;
        }
        if (betas_f_capacity < nv) {
            if (betas_f) cudaFree(betas_f);
            cudaMalloc(&betas_f, (size_t)nv * sizeof(float));
            betas_f_capacity = nv;
        }
    }
};

}  

katz_centrality_result_t katz_centrality_mask(const graph32_t& graph,
                           const double* edge_weights,
                           double* centralities,
                           double alpha,
                           double beta,
                           const double* betas,
                           double epsilon,
                           std::size_t max_iterations,
                           bool has_initial_guess,
                           bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    const uint32_t* edge_mask = graph.edge_mask;

    cache.ensure_buffers(nv, ne);

    
    double h_scalars[3] = {alpha, beta, epsilon};
    cudaMemcpy(cache.d_double_scalars, h_scalars, 3 * sizeof(double), cudaMemcpyHostToDevice);

    int64_t h_max_iters = (int64_t)max_iterations;
    cudaMemcpy(cache.d_max_iters, &h_max_iters, sizeof(int64_t), cudaMemcpyHostToDevice);

    bool h_flags[2] = {normalize, has_initial_guess};
    cudaMemcpy(cache.d_bool_flags, h_flags, 2 * sizeof(bool), cudaMemcpyHostToDevice);

    
    const double* alpha_d = &cache.d_double_scalars[0];
    const double* beta_scalar_d = &cache.d_double_scalars[1];
    const double* epsilon_d = &cache.d_double_scalars[2];
    const int64_t* max_iters_i64 = cache.d_max_iters;
    const bool* normalize_b = &cache.d_bool_flags[0];
    const bool* has_init_b = &cache.d_bool_flags[1];

    
    const double* init_centralities_d = has_initial_guess ? centralities : nullptr;

    
    float* betas_f = (betas != nullptr) ? cache.betas_f : nullptr;

    cudaStream_t stream = 0;

    
    int32_t avg_deg = (nv > 0) ? (ne / nv) : 0;
    bool use_warp = (avg_deg >= 4);

    if (use_warp) {
        launch_katz_coop_warp(
            offsets, indices, edge_weights, cache.weights_f, edge_mask,
            betas, betas_f,
            alpha_d, beta_scalar_d, epsilon_d, max_iters_i64,
            normalize_b, has_init_b, init_centralities_d,
            cache.x0, cache.x1,
            cache.scratch_f, cache.scratch_i,
            centralities, cache.d_out_iters, cache.d_out_converged,
            nv, ne, stream);
    } else {
        launch_katz_coop_thread(
            offsets, indices, edge_weights, cache.weights_f, edge_mask,
            betas, betas_f,
            alpha_d, beta_scalar_d, epsilon_d, max_iters_i64,
            normalize_b, has_init_b, init_centralities_d,
            cache.x0, cache.x1,
            cache.scratch_f, cache.scratch_i,
            centralities, cache.d_out_iters, cache.d_out_converged,
            nv, ne, stream);
    }

    
    int64_t h_iters;
    bool h_converged;
    cudaMemcpy(&h_iters, cache.d_out_iters, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_converged, cache.d_out_converged, sizeof(bool), cudaMemcpyDeviceToHost);

    return {(std::size_t)h_iters, h_converged};
}

}  
