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
#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define LM_BLOCK 128
#define HASH_CAP 31

struct Cache : Cacheable {
    void* cub_temp = nullptr;
    size_t cub_temp_size = 0;

    
    float* edge_weights_f = nullptr;
    size_t edge_weights_f_cap = 0;

    float* orig_vw = nullptr;
    size_t orig_vw_cap = 0;

    float* tw_buf = nullptr;  
    bool tw_buf_allocated = false;

    int* count_buf = nullptr;  
    bool count_buf_allocated = false;

    void ensure_cub_temp(size_t needed) {
        if (needed > cub_temp_size) {
            if (cub_temp) cudaFree(cub_temp);
            cub_temp_size = needed * 2;
            cudaMalloc(&cub_temp, cub_temp_size);
        }
    }

    void ensure_edge_weights_f(size_t n) {
        if (edge_weights_f_cap < n) {
            if (edge_weights_f) cudaFree(edge_weights_f);
            cudaMalloc(&edge_weights_f, n * sizeof(float));
            edge_weights_f_cap = n;
        }
    }

    void ensure_orig_vw(size_t n) {
        if (orig_vw_cap < n) {
            if (orig_vw) cudaFree(orig_vw);
            cudaMalloc(&orig_vw, n * sizeof(float));
            orig_vw_cap = n;
        }
    }

    void ensure_tw_buf() {
        if (!tw_buf_allocated) {
            cudaMalloc(&tw_buf, sizeof(float));
            tw_buf_allocated = true;
        }
    }

    void ensure_count_buf() {
        if (!count_buf_allocated) {
            cudaMalloc(&count_buf, sizeof(int));
            count_buf_allocated = true;
        }
    }

    ~Cache() override {
        if (cub_temp) cudaFree(cub_temp);
        if (edge_weights_f) cudaFree(edge_weights_f);
        if (orig_vw) cudaFree(orig_vw);
        if (tw_buf) cudaFree(tw_buf);
        if (count_buf) cudaFree(count_buf);
    }
};



__global__ void compute_vertex_weights_kernel(
    const int32_t* __restrict__ offsets,
    const float* __restrict__ edge_weights,
    float* __restrict__ vertex_weight,
    int N
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    float sum = 0.0f;
    for (int e = offsets[v]; e < offsets[v + 1]; e++)
        sum += edge_weights[e];
    vertex_weight[v] = sum;
}

__global__ void init_communities_kernel(int32_t* __restrict__ community, int N) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    community[v] = v;
}

__global__ void reduce_sum_kernel(const float* __restrict__ input, float* __restrict__ output, int N) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += gridDim.x * blockDim.x)
        sum += input[i];
    sdata[tid] = sum;
    __syncthreads();
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(output, sdata[0]);
}

__global__ void recompute_sigma_tot_kernel(
    const int32_t* __restrict__ community,
    const float* __restrict__ vertex_weight,
    float* __restrict__ sigma_tot,
    int N
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    atomicAdd(&sigma_tot[community[v]], vertex_weight[v]);
}



__global__ void local_move_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    int32_t* __restrict__ community,
    const float* __restrict__ sigma_tot,
    const float* __restrict__ vertex_weight,
    int N, float total_weight, float resolution,
    int* __restrict__ changed,
    int color, int num_colors
) {
    extern __shared__ char smem[];
    int* s_keys = (int*)smem;
    float* s_vals = (float*)(smem + blockDim.x * HASH_CAP * sizeof(int));

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int v = idx * num_colors + color;

    int base = tid * HASH_CAP;
    for (int i = 0; i < HASH_CAP; i++) {
        s_keys[base + i] = -1;
        s_vals[base + i] = 0.0f;
    }

    if (v >= N) return;

    int my_comm = community[v];
    float my_k = vertex_weight[v];
    int start = offsets[v];
    int end = offsets[v + 1];
    if (start == end) return;

    for (int e = start; e < end; e++) {
        if (indices[e] == v) continue;
        int c = community[indices[e]];
        float w = edge_weights[e];
        unsigned int slot = ((unsigned int)c * 2654435761u) % (unsigned int)HASH_CAP;
        for (int p = 0; p < HASH_CAP; p++) {
            int pos = base + ((int)(slot + p) % HASH_CAP);
            if (s_keys[pos] == c) { s_vals[pos] += w; break; }
            else if (s_keys[pos] == -1) { s_keys[pos] = c; s_vals[pos] = w; break; }
        }
    }

    float k_own = 0.0f;
    for (int i = 0; i < HASH_CAP; i++) {
        if (s_keys[base + i] == my_comm) { k_own = s_vals[base + i]; break; }
    }

    float sigma_own = sigma_tot[my_comm];
    float base_score = k_own - resolution * my_k * (sigma_own - my_k) / total_weight;
    float best_score = base_score;
    int best_comm = my_comm;

    for (int i = 0; i < HASH_CAP; i++) {
        int c = s_keys[base + i];
        if (c == -1 || c == my_comm) continue;
        float score = s_vals[base + i] - resolution * my_k * sigma_tot[c] / total_weight;
        if (score > best_score) { best_score = score; best_comm = c; }
    }

    if (best_comm != my_comm) {
        community[v] = best_comm;
        *changed = 1;
    }
}



__global__ void compute_best_move_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const int32_t* __restrict__ community,
    const float* __restrict__ sigma_tot,
    const float* __restrict__ vertex_weight,
    int N, float total_weight, float resolution,
    int32_t* __restrict__ best_comm_out,
    float* __restrict__ best_gain_out
) {
    extern __shared__ char smem[];
    int* s_keys = (int*)smem;
    float* s_vals = (float*)(smem + blockDim.x * HASH_CAP * sizeof(int));

    int tid = threadIdx.x;
    int v = blockIdx.x * blockDim.x + tid;

    int base = tid * HASH_CAP;
    for (int i = 0; i < HASH_CAP; i++) {
        s_keys[base + i] = -1;
        s_vals[base + i] = 0.0f;
    }

    if (v >= N) return;

    int my_comm = community[v];
    float my_k = vertex_weight[v];
    int start = offsets[v];
    int end = offsets[v + 1];

    if (start == end) {
        best_comm_out[v] = my_comm;
        best_gain_out[v] = 0.0f;
        return;
    }

    for (int e = start; e < end; e++) {
        if (indices[e] == v) continue;
        int c = community[indices[e]];
        float w = edge_weights[e];
        unsigned int slot = ((unsigned int)c * 2654435761u) % (unsigned int)HASH_CAP;
        for (int p = 0; p < HASH_CAP; p++) {
            int pos = base + ((int)(slot + p) % HASH_CAP);
            if (s_keys[pos] == c) { s_vals[pos] += w; break; }
            else if (s_keys[pos] == -1) { s_keys[pos] = c; s_vals[pos] = w; break; }
        }
    }

    float k_own = 0.0f;
    for (int i = 0; i < HASH_CAP; i++) {
        if (s_keys[base + i] == my_comm) { k_own = s_vals[base + i]; break; }
    }

    float sigma_own = sigma_tot[my_comm];
    float base_score = k_own - resolution * my_k * (sigma_own - my_k) / total_weight;
    float best_score = 0.0f;
    int best_c = my_comm;

    for (int i = 0; i < HASH_CAP; i++) {
        int c = s_keys[base + i];
        if (c == -1 || c == my_comm) continue;
        float score = s_vals[base + i] - resolution * my_k * sigma_tot[c] / total_weight;
        float delta = score - base_score;
        if (delta > best_score || (delta == best_score && delta > 0.0f && c < best_c)) {
            best_score = delta;
            best_c = c;
        }
    }

    best_comm_out[v] = best_c;
    best_gain_out[v] = best_score;
}

__global__ void count_updown_kernel(
    const int32_t* __restrict__ community,
    const int32_t* __restrict__ best_comm,
    const float* __restrict__ best_gain,
    int N, bool up_down, int* __restrict__ count
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    if (best_gain[v] > 0.0f) {
        bool is_up = best_comm[v] > community[v];
        if (is_up == up_down) atomicAdd(count, 1);
    }
}

__global__ void apply_updown_kernel(
    int32_t* __restrict__ community,
    const int32_t* __restrict__ best_comm,
    const float* __restrict__ best_gain,
    int N, bool up_down
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    if (best_gain[v] > 0.0f) {
        bool is_up = best_comm[v] > community[v];
        if (is_up == up_down) {
            community[v] = best_comm[v];
        }
    }
}


#define SEQ_HASH_CAP 64

__device__ bool seq_process_vertex(
    int v,
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    int32_t* __restrict__ community,
    float* __restrict__ sigma_tot,
    const float* __restrict__ vertex_weight,
    float total_weight, float resolution,
    int* hash_keys, float* hash_vals
) {
    int start = offsets[v];
    int end = offsets[v + 1];
    if (start == end) return false;

    int my_comm = community[v];
    double my_k = (double)vertex_weight[v];

    for (int i = 0; i < SEQ_HASH_CAP; i++) {
        hash_keys[i] = -1;
        hash_vals[i] = 0.0f;
    }

    for (int e = start; e < end; e++) {
        if (indices[e] == v) continue;
        int c = community[indices[e]];
        float w = edge_weights[e];
        unsigned int slot = ((unsigned int)c * 2654435761u) % (unsigned int)SEQ_HASH_CAP;
        for (int p = 0; p < SEQ_HASH_CAP; p++) {
            int pos = ((int)slot + p) % SEQ_HASH_CAP;
            if (hash_keys[pos] == c) { hash_vals[pos] += w; break; }
            else if (hash_keys[pos] == -1) { hash_keys[pos] = c; hash_vals[pos] = w; break; }
        }
    }

    double k_own = 0.0;
    for (int i = 0; i < SEQ_HASH_CAP; i++) {
        if (hash_keys[i] == my_comm) { k_own = (double)hash_vals[i]; break; }
    }

    double sigma_own = (double)sigma_tot[my_comm];
    double tw_d = (double)total_weight;
    double res_d = (double)resolution;
    double base_score = k_own - res_d * my_k * (sigma_own - my_k) / tw_d;
    double best_score = base_score;
    int best_comm = my_comm;

    for (int i = 0; i < SEQ_HASH_CAP; i++) {
        int c = hash_keys[i];
        if (c == -1 || c == my_comm) continue;
        double score = (double)hash_vals[i] - res_d * my_k * (double)sigma_tot[c] / tw_d;
        if (score > best_score || (score == best_score && c < best_comm && score > base_score)) {
            best_score = score;
            best_comm = c;
        }
    }

    if (best_comm != my_comm) {
        sigma_tot[my_comm] -= vertex_weight[v];
        sigma_tot[best_comm] += vertex_weight[v];
        community[v] = best_comm;
        return true;
    }
    return false;
}

__global__ void sequential_local_move_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    int32_t* __restrict__ community,
    float* __restrict__ sigma_tot,
    const float* __restrict__ vertex_weight,
    int N, float total_weight, float resolution,
    int* __restrict__ changed,
    int start_offset
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    int hash_keys[SEQ_HASH_CAP];
    float hash_vals[SEQ_HASH_CAP];

    for (int i = 0; i < N; i++) {
        int v = (i + start_offset) % N;
        if (seq_process_vertex(v, offsets, indices, edge_weights, community,
                               sigma_tot, vertex_weight, total_weight, resolution,
                               hash_keys, hash_vals))
            *changed = 1;
    }
}


#define BATCH_MAX_N 256

__global__ void batch_sequential_louvain_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const float* __restrict__ edge_weights,
    const float* __restrict__ vertex_weight,
    int N, float total_weight, float resolution,
    int32_t* __restrict__ all_communities,
    float* __restrict__ all_sigma,
    float* __restrict__ all_modularity,
    int num_restarts
) {
    int restart = blockIdx.x;
    if (restart >= num_restarts || threadIdx.x != 0) return;

    int32_t* comm = all_communities + restart * N;
    float* sigma = all_sigma + restart * N;

    int perm[BATCH_MAX_N];
    for (int i = 0; i < N; i++) perm[i] = i;

    if (restart == 1) {
        for (int i = 0; i < N / 2; i++) {
            int tmp = perm[i]; perm[i] = perm[N-1-i]; perm[N-1-i] = tmp;
        }
    } else if (restart >= 2) {
        unsigned long long seed = (unsigned long long)restart * 6364136223846793005ULL + 1442695040888963407ULL;
        for (int i = N - 1; i > 0; i--) {
            seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
            int j = (int)((seed >> 16) % (unsigned long long)(i + 1));
            int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
        }
    }

    for (int i = 0; i < N; i++) comm[i] = i;

    int hash_keys[SEQ_HASH_CAP];
    float hash_vals[SEQ_HASH_CAP];

    for (int iter = 0; iter < 200; iter++) {
        for (int i = 0; i < N; i++) sigma[i] = 0.0f;
        for (int i = 0; i < N; i++) sigma[comm[i]] += vertex_weight[i];

        bool changed = false;
        for (int i = 0; i < N; i++) {
            int v = perm[i];
            if (seq_process_vertex(v, offsets, indices, edge_weights,
                                   comm, sigma, vertex_weight,
                                   total_weight, resolution,
                                   hash_keys, hash_vals))
                changed = true;
        }
        for (int i = N - 1; i >= 0; i--) {
            int v = perm[i];
            if (seq_process_vertex(v, offsets, indices, edge_weights,
                                   comm, sigma, vertex_weight,
                                   total_weight, resolution,
                                   hash_keys, hash_vals))
                changed = true;
        }
        if (!changed) break;
    }

    for (int i = 0; i < N; i++) sigma[i] = 0.0f;
    for (int i = 0; i < N; i++) sigma[comm[i]] += vertex_weight[i];

    double iw = 0.0;
    for (int v = 0; v < N; v++) {
        int cv = comm[v];
        for (int e = offsets[v]; e < offsets[v + 1]; e++) {
            if (comm[indices[e]] == cv) iw += (double)edge_weights[e];
        }
    }
    double ssq = 0.0;
    for (int i = 0; i < N; i++) {
        double s = (double)sigma[i];
        ssq += s * s;
    }
    double m = (double)total_weight / 2.0;
    all_modularity[restart] = (float)(iw / (2.0 * m) - (double)resolution * ssq / (4.0 * m * m));
}



__global__ void mark_communities_kernel(const int32_t* __restrict__ community, int* __restrict__ used, int N) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N) used[community[v]] = 1;
}

__global__ void apply_renumber_kernel(int32_t* __restrict__ community, const int* __restrict__ mapping, int N) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < N) community[v] = mapping[community[v]];
}

__global__ void map_edges_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ community,
    int64_t* __restrict__ edge_keys,
    int N
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= N) return;
    int cv = community[v];
    for (int e = offsets[v]; e < offsets[v + 1]; e++) {
        int cu = community[indices[e]];
        edge_keys[e] = ((int64_t)(unsigned int)cv << 32) | (int64_t)(unsigned int)cu;
    }
}

__global__ void count_src_edges_kernel(const int64_t* __restrict__ unique_keys, int* __restrict__ counts, int num_edges) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;
    int src = (int)(unique_keys[i] >> 32);
    atomicAdd(&counts[src], 1);
}

__global__ void fill_csr_kernel(
    const int64_t* __restrict__ unique_keys,
    const float* __restrict__ unique_vals,
    const int* __restrict__ offsets,
    int32_t* __restrict__ indices,
    float* __restrict__ weights,
    int num_edges
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;
    int dst = (int)(unique_keys[i] & 0xFFFFFFFF);
    indices[i] = dst;
    weights[i] = unique_vals[i];
}

__global__ void compute_new_vw_kernel(
    const float* __restrict__ old_vw,
    const int32_t* __restrict__ community,
    float* __restrict__ new_vw,
    int old_N
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= old_N) return;
    atomicAdd(&new_vw[community[v]], old_vw[v]);
}

__global__ void compose_communities_kernel(
    int32_t* __restrict__ final_comm,
    const int32_t* __restrict__ level_comm,
    int orig_N
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= orig_N) return;
    final_comm[v] = level_comm[final_comm[v]];
}



__global__ void compute_iw_kernel(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    const float* __restrict__ w, const int32_t* __restrict__ c,
    int N, double* __restrict__ result) {
    typedef cub::BlockReduce<double, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage tmp;
    double s = 0.0;
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < N; v += blockDim.x * gridDim.x) {
        int cv = c[v];
        for (int e = off[v]; e < off[v + 1]; e++)
            if (c[idx[e]] == cv) s += (double)w[e];
    }
    double bs = BR(tmp).Sum(s);
    if (threadIdx.x == 0) atomicAdd(result, bs);
}

__global__ void compute_ssq_kernel(
    const float* __restrict__ st, int N, double* __restrict__ result) {
    typedef cub::BlockReduce<double, BLOCK_SIZE> BR;
    __shared__ typename BR::TempStorage tmp;
    double s = 0.0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        double v = (double)st[i]; s += v * v;
    }
    double bs = BR(tmp).Sum(s);
    if (threadIdx.x == 0) atomicAdd(result, bs);
}


__global__ void convert_d2f_kernel(const double* __restrict__ in, float* __restrict__ out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (float)in[i];
}



static void do_compute_vertex_weights(const int32_t* offsets, const float* weights, float* vw, int N) {
    compute_vertex_weights_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(offsets, weights, vw, N);
}

static void do_init_communities(int32_t* comm, int N) {
    init_communities_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(comm, N);
}

static void do_reduce_sum(const float* input, float* output, int N) {
    cudaMemset(output, 0, sizeof(float));
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 1024) grid = 1024;
    reduce_sum_kernel<<<grid, BLOCK_SIZE>>>(input, output, N);
}

static void do_recompute_sigma_tot(const int32_t* community, const float* vw, float* sigma_tot, int N) {
    cudaMemset(sigma_tot, 0, N * sizeof(float));
    recompute_sigma_tot_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(community, vw, sigma_tot, N);
}

static void do_local_move(
    const int32_t* offsets, const int32_t* indices, const float* weights,
    int32_t* community, const float* sigma_tot, const float* vw,
    int N, float tw, float res, int* changed, int color, int num_colors
) {
    int num_verts = (N + num_colors - 1 - color) / num_colors;
    if (num_verts <= 0) return;
    int grid = (num_verts + LM_BLOCK - 1) / LM_BLOCK;
    size_t smem_size = LM_BLOCK * HASH_CAP * (sizeof(int) + sizeof(float));
    local_move_kernel<<<grid, LM_BLOCK, smem_size>>>(
        offsets, indices, weights, community, sigma_tot, vw,
        N, tw, res, changed, color, num_colors);
}

static void do_compute_best_move(
    const int32_t* offsets, const int32_t* indices, const float* weights,
    const int32_t* community, const float* sigma_tot, const float* vw,
    int N, float tw, float res,
    int32_t* best_comm, float* best_gain
) {
    if (N <= 0) return;
    int grid = (N + LM_BLOCK - 1) / LM_BLOCK;
    size_t smem_size = LM_BLOCK * HASH_CAP * (sizeof(int) + sizeof(float));
    compute_best_move_kernel<<<grid, LM_BLOCK, smem_size>>>(
        offsets, indices, weights, community, sigma_tot, vw,
        N, tw, res, best_comm, best_gain);
}

static void do_count_updown(
    const int32_t* community, const int32_t* best_comm, const float* best_gain,
    int N, bool up_down, int* count
) {
    if (N <= 0) return;
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_updown_kernel<<<grid, BLOCK_SIZE>>>(community, best_comm, best_gain, N, up_down, count);
}

static void do_apply_updown(
    int32_t* community, const int32_t* best_comm, const float* best_gain,
    int N, bool up_down
) {
    if (N <= 0) return;
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_updown_kernel<<<grid, BLOCK_SIZE>>>(community, best_comm, best_gain, N, up_down);
}

static void do_sequential_local_move(
    const int32_t* offsets, const int32_t* indices, const float* weights,
    int32_t* community, float* sigma_tot, const float* vw,
    int N, float tw, float res, int* changed, int start_offset
) {
    if (N <= 0) return;
    sequential_local_move_kernel<<<1, 1>>>(
        offsets, indices, weights, community, sigma_tot, vw,
        N, tw, res, changed, start_offset);
}

static void do_batch_sequential_louvain(
    const int32_t* offsets, const int32_t* indices, const float* weights,
    const float* vw, int N, float tw, float res,
    int32_t* all_comm, float* all_sigma, float* all_mod, int num_restarts
) {
    if (num_restarts <= 0 || N <= 0) return;
    batch_sequential_louvain_kernel<<<num_restarts, 1>>>(
        offsets, indices, weights, vw, N, tw, res,
        all_comm, all_sigma, all_mod, num_restarts);
}

static int do_renumber_communities(Cache& cache, int32_t* community, int* used, int* prefix, int N) {
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMemset(used, 0, N * sizeof(int));
    mark_communities_kernel<<<grid, BLOCK_SIZE>>>(community, used, N);

    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, used, prefix, N);
    cache.ensure_cub_temp(temp_bytes);
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, temp_bytes, used, prefix, N);

    int h_last_prefix, h_last_used;
    cudaMemcpy(&h_last_prefix, prefix + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_last_used, used + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int new_N = h_last_prefix + h_last_used;

    apply_renumber_kernel<<<grid, BLOCK_SIZE>>>(community, prefix, N);

    return new_N;
}

static int do_sort_reduce_edges(
    Cache& cache,
    int64_t* keys_in, float* vals_in, int E,
    int64_t* sort_keys_out, float* sort_vals_out,
    int64_t* unique_keys_out, float* unique_vals_out,
    int* d_num_unique
) {
    size_t sort_temp = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sort_temp, keys_in, sort_keys_out, vals_in, sort_vals_out, E);
    cache.ensure_cub_temp(sort_temp);
    cub::DeviceRadixSort::SortPairs(cache.cub_temp, sort_temp, keys_in, sort_keys_out, vals_in, sort_vals_out, E);

    size_t reduce_temp = 0;
    cub::DeviceReduce::ReduceByKey(nullptr, reduce_temp, sort_keys_out, unique_keys_out,
        sort_vals_out, unique_vals_out, d_num_unique, ::cuda::std::plus<float>{}, E);
    cache.ensure_cub_temp(reduce_temp);
    cub::DeviceReduce::ReduceByKey(cache.cub_temp, reduce_temp, sort_keys_out, unique_keys_out,
        sort_vals_out, unique_vals_out, d_num_unique, ::cuda::std::plus<float>{}, E);

    int h_num_unique;
    cudaMemcpy(&h_num_unique, d_num_unique, sizeof(int), cudaMemcpyDeviceToHost);
    return h_num_unique;
}

static void do_build_csr(
    Cache& cache,
    const int64_t* unique_keys, const float* unique_vals,
    int new_E, int new_N,
    int* counts, int32_t* new_offsets, int32_t* new_indices, float* new_weights
) {
    int grid_e = (new_E + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMemset(counts, 0, new_N * sizeof(int));
    count_src_edges_kernel<<<grid_e, BLOCK_SIZE>>>(unique_keys, counts, new_E);

    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, counts, (int*)new_offsets, new_N);
    cache.ensure_cub_temp(temp_bytes);
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, temp_bytes, counts, (int*)new_offsets, new_N);
    int h_new_E = new_E;
    cudaMemcpy(new_offsets + new_N, &h_new_E, sizeof(int32_t), cudaMemcpyHostToDevice);

    fill_csr_kernel<<<grid_e, BLOCK_SIZE>>>(unique_keys, unique_vals, (int*)new_offsets, new_indices, new_weights, new_E);
}

static void do_map_edges(
    const int32_t* offsets, const int32_t* indices, const int32_t* community,
    int64_t* edge_keys, int N
) {
    map_edges_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(offsets, indices, community, edge_keys, N);
}

static void do_compute_new_vw(const float* old_vw, const int32_t* community, float* new_vw, int old_N, int new_N) {
    cudaMemset(new_vw, 0, new_N * sizeof(float));
    compute_new_vw_kernel<<<(old_N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(old_vw, community, new_vw, old_N);
}

static void do_compose_communities(int32_t* final_comm, const int32_t* level_comm, int orig_N) {
    compose_communities_kernel<<<(orig_N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(final_comm, level_comm, orig_N);
}

static float do_compute_modularity(
    const int32_t* offsets, const int32_t* indices, const float* weights,
    const int32_t* clusters, const float* vw, int N, float total_weight, float resolution
) {
    float* sigma;
    cudaMalloc(&sigma, N * sizeof(float));
    cudaMemset(sigma, 0, N * sizeof(float));
    recompute_sigma_tot_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(clusters, vw, sigma, N);

    double* d_accum;
    cudaMalloc(&d_accum, sizeof(double));

    cudaMemset(d_accum, 0, sizeof(double));
    int g = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (g > 1024) g = 1024;
    compute_iw_kernel<<<g, BLOCK_SIZE>>>(offsets, indices, weights, clusters, N, d_accum);
    double iw;
    cudaMemcpy(&iw, d_accum, sizeof(double), cudaMemcpyDeviceToHost);

    cudaMemset(d_accum, 0, sizeof(double));
    compute_ssq_kernel<<<g, BLOCK_SIZE>>>(sigma, N, d_accum);
    double ssq;
    cudaMemcpy(&ssq, d_accum, sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(sigma);
    cudaFree(d_accum);

    double m = (double)total_weight / 2.0;
    return (float)(iw / (2.0 * m) - (double)resolution * ssq / (4.0 * m * m));
}

static void do_convert_double_to_float(const double* in, float* out, int N) {
    if (N <= 0) return;
    convert_d2f_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(in, out, N);
}

}  

louvain_result_double_t louvain_seg(const graph32_t& graph,
                                    const double* edge_weights,
                                    int32_t* clusters,
                                    std::size_t max_level,
                                    double threshold,
                                    double resolution) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t orig_N = graph.number_of_vertices;
    int32_t orig_E = graph.number_of_edges;
    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;

    float threshold_f = (float)threshold;
    float resolution_f = (float)resolution;

    
    cache.ensure_edge_weights_f(orig_E);
    do_convert_double_to_float(edge_weights, cache.edge_weights_f, orig_E);
    float* ew = cache.edge_weights_f;

    
    cache.ensure_orig_vw(orig_N);
    do_compute_vertex_weights(offsets, ew, cache.orig_vw, orig_N);
    float* orig_vw = cache.orig_vw;

    
    cache.ensure_tw_buf();
    do_reduce_sum(orig_vw, cache.tw_buf, orig_N);
    float total_weight;
    cudaMemcpy(&total_weight, cache.tw_buf, sizeof(float), cudaMemcpyDeviceToHost);

    
    do_init_communities(clusters, orig_N);

    if (total_weight == 0.0f || orig_N <= 1) {
        return {0, 0.0};
    }

    cache.ensure_count_buf();
    int* d_count = cache.count_buf;
    float overall_best_Q = -1.0f;
    int64_t actual_level_count = 1;

    
    if (orig_N <= 256) {
        int num_restarts = 2048;
        int32_t* all_comm; cudaMalloc(&all_comm, (int64_t)num_restarts * orig_N * sizeof(int32_t));
        float* all_sigma; cudaMalloc(&all_sigma, (int64_t)num_restarts * orig_N * sizeof(float));
        float* all_mod; cudaMalloc(&all_mod, num_restarts * sizeof(float));

        do_batch_sequential_louvain(offsets, indices, ew, orig_vw, orig_N, total_weight, resolution_f,
            all_comm, all_sigma, all_mod, num_restarts);
        cudaDeviceSynchronize();

        std::vector<float> h_mod(num_restarts);
        cudaMemcpy(h_mod.data(), all_mod, num_restarts * sizeof(float), cudaMemcpyDeviceToHost);
        std::vector<int> sorted_idx(num_restarts);
        for (int i = 0; i < num_restarts; i++) sorted_idx[i] = i;
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) { return h_mod[a] > h_mod[b]; });

        int topK = std::min(50, num_restarts);
        int32_t* trial_comm; cudaMalloc(&trial_comm, orig_N * sizeof(int32_t));
        int* d_lc; cudaMalloc(&d_lc, sizeof(int));

        for (int ki = 0; ki < topK; ki++) {
            int idx_r = sorted_idx[ki];
            cudaMemcpy(trial_comm, all_comm + (int64_t)idx_r * orig_N,
                       orig_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);

            int32_t* ml_comm; cudaMalloc(&ml_comm, orig_N * sizeof(int32_t));
            cudaMemcpy(ml_comm, trial_comm, orig_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);

            int cur_N = orig_N, cur_E = orig_E;
            const int32_t* cur_off = offsets;
            const int32_t* cur_idx2 = indices;
            const float* cur_wt = ew;

            
            int32_t* c_off = nullptr;
            int32_t* c_idx = nullptr;
            float* c_wt = nullptr;
            float* c_vw = nullptr;

            float* cur_vw = orig_vw;
            float level_best = h_mod[idx_r];
            bool first_level = true;

            for (int level = 0; level < (int)max_level; level++) {
                int* used; cudaMalloc(&used, cur_N * sizeof(int));
                int* pref; cudaMalloc(&pref, cur_N * sizeof(int));
                int new_N = do_renumber_communities(cache, ml_comm, used, pref, cur_N);
                cudaFree(used);
                cudaFree(pref);
                if (new_N >= cur_N) break;

                if (first_level) {
                    cudaMemcpy(trial_comm, ml_comm, orig_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);
                    first_level = false;
                } else {
                    do_compose_communities(trial_comm, ml_comm, orig_N);
                }

                int64_t* ek; cudaMalloc(&ek, cur_E * sizeof(int64_t));
                float* ev; cudaMalloc(&ev, cur_E * sizeof(float));
                do_map_edges(cur_off, cur_idx2, ml_comm, ek, cur_N);
                cudaMemcpy(ev, cur_wt, cur_E * sizeof(float), cudaMemcpyDeviceToDevice);

                int64_t* sk; cudaMalloc(&sk, cur_E * sizeof(int64_t));
                float* sv; cudaMalloc(&sv, cur_E * sizeof(float));
                int64_t* uk; cudaMalloc(&uk, cur_E * sizeof(int64_t));
                float* uv; cudaMalloc(&uv, cur_E * sizeof(float));
                int* nu; cudaMalloc(&nu, sizeof(int));
                int new_E = do_sort_reduce_edges(cache, ek, ev, cur_E, sk, sv, uk, uv, nu);
                cudaFree(ek); cudaFree(ev); cudaFree(sk); cudaFree(sv); cudaFree(nu);

                
                int32_t* old_c_off = c_off;
                int32_t* old_c_idx = c_idx;
                float* old_c_wt = c_wt;
                float* old_c_vw = c_vw;

                cudaMalloc(&c_off, (new_N + 1) * sizeof(int32_t));
                cudaMalloc(&c_idx, new_E * sizeof(int32_t));
                cudaMalloc(&c_wt, new_E * sizeof(float));
                int* cnt; cudaMalloc(&cnt, new_N * sizeof(int));
                do_build_csr(cache, uk, uv, new_E, new_N, cnt, c_off, c_idx, c_wt);
                cudaFree(uk); cudaFree(uv); cudaFree(cnt);

                cudaMalloc(&c_vw, new_N * sizeof(float));
                do_compute_new_vw(cur_vw, ml_comm, c_vw, cur_N, new_N);

                
                if (old_c_off) { cudaFree(old_c_off); cudaFree(old_c_idx); cudaFree(old_c_wt); cudaFree(old_c_vw); }

                cur_N = new_N; cur_E = new_E;
                cur_off = c_off;
                cur_idx2 = c_idx;
                cur_wt = c_wt;
                cur_vw = c_vw;

                cudaFree(ml_comm);
                cudaMalloc(&ml_comm, cur_N * sizeof(int32_t));
                float* ls; cudaMalloc(&ls, cur_N * sizeof(float));
                do_init_communities(ml_comm, cur_N);

                for (int iter = 0; iter < 200; iter++) {
                    do_recompute_sigma_tot(ml_comm, cur_vw, ls, cur_N);
                    cudaMemset(d_lc, 0, sizeof(int));
                    do_sequential_local_move(cur_off, cur_idx2, cur_wt,
                        ml_comm, ls, cur_vw, cur_N, total_weight, resolution_f, d_lc, 0);
                    int h_changed = 0;
                    cudaMemcpy(&h_changed, d_lc, sizeof(int), cudaMemcpyDeviceToHost);
                    if (!h_changed) break;
                }
                cudaFree(ls);

                do_compose_communities(trial_comm, ml_comm, orig_N);
                float new_Q = do_compute_modularity(offsets, indices, ew,
                    trial_comm, orig_vw, orig_N, total_weight, resolution_f);
                if (new_Q <= level_best) break;
                level_best = new_Q;
            }

            float trial_Q = do_compute_modularity(offsets, indices, ew,
                trial_comm, orig_vw, orig_N, total_weight, resolution_f);
            if (trial_Q > overall_best_Q) {
                overall_best_Q = trial_Q;
                cudaMemcpy(clusters, trial_comm, orig_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            }

            cudaFree(ml_comm);
            if (c_off) { cudaFree(c_off); cudaFree(c_idx); cudaFree(c_wt); cudaFree(c_vw); }
        }

        cudaFree(trial_comm);
        cudaFree(d_lc);
        cudaFree(all_comm);
        cudaFree(all_sigma);
        cudaFree(all_mod);
    }

    
    {
        int32_t* d_par; cudaMalloc(&d_par, orig_N * sizeof(int32_t));
        do_init_communities(d_par, orig_N);

        int cur_N = orig_N, cur_E = orig_E;
        const int32_t* cur_off = offsets;
        const int32_t* cur_idx2 = indices;
        const float* cur_wt = ew;

        
        int32_t* c_off = nullptr;
        int32_t* c_idx = nullptr;
        float* c_wt = nullptr;
        float* c_vw = nullptr;

        float* cur_vw = orig_vw;
        float best_modularity = -1.0f;
        int par_levels = 0;

        for (int level = 0; level < (int)max_level; level++) {
            int32_t* d_comm; cudaMalloc(&d_comm, cur_N * sizeof(int32_t));
            float* d_sigma; cudaMalloc(&d_sigma, cur_N * sizeof(float));
            int32_t* bm_comm; cudaMalloc(&bm_comm, cur_N * sizeof(int32_t));
            float* bm_gain; cudaMalloc(&bm_gain, cur_N * sizeof(float));
            int32_t* best_saved; cudaMalloc(&best_saved, cur_N * sizeof(int32_t));

            do_init_communities(d_comm, cur_N);
            float new_Q = do_compute_modularity(cur_off, cur_idx2, cur_wt, d_comm,
                cur_vw, cur_N, total_weight, resolution_f);
            float cur_Q = new_Q - 1;
            cudaMemcpy(best_saved, d_comm, cur_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);

            bool up_down = true;
            while (new_Q > cur_Q + threshold_f) {
                cur_Q = new_Q;
                do_recompute_sigma_tot(d_comm, cur_vw, d_sigma, cur_N);
                do_compute_best_move(cur_off, cur_idx2, cur_wt, d_comm, d_sigma,
                    cur_vw, cur_N, total_weight, resolution_f, bm_comm, bm_gain);
                cudaMemset(d_count, 0, sizeof(int));
                do_count_updown(d_comm, bm_comm, bm_gain, cur_N, up_down, d_count);
                int h_count;
                cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
                bool eff = up_down; if (h_count == 0) eff = !up_down;
                do_apply_updown(d_comm, bm_comm, bm_gain, cur_N, eff);
                up_down = !up_down;
                new_Q = do_compute_modularity(cur_off, cur_idx2, cur_wt, d_comm,
                    cur_vw, cur_N, total_weight, resolution_f);
                if (new_Q > cur_Q) cudaMemcpy(best_saved, d_comm, cur_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            }
            cudaMemcpy(d_comm, best_saved, cur_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);

            cudaFree(d_sigma);
            cudaFree(bm_comm);
            cudaFree(bm_gain);
            cudaFree(best_saved);

            if (cur_Q <= best_modularity) { cudaFree(d_comm); break; }
            best_modularity = cur_Q;

            int* used; cudaMalloc(&used, cur_N * sizeof(int));
            int* pref; cudaMalloc(&pref, cur_N * sizeof(int));
            int new_N = do_renumber_communities(cache, d_comm, used, pref, cur_N);
            cudaFree(used);
            cudaFree(pref);

            if (new_N >= cur_N) { cudaFree(d_comm); break; }
            if (level == 0) cudaMemcpy(d_par, d_comm, orig_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            else do_compose_communities(d_par, d_comm, orig_N);
            par_levels = level + 1;

            if (level + 1 >= (int)max_level) { cudaFree(d_comm); break; }

            int64_t* ek; cudaMalloc(&ek, cur_E * sizeof(int64_t));
            float* ev; cudaMalloc(&ev, cur_E * sizeof(float));
            do_map_edges(cur_off, cur_idx2, d_comm, ek, cur_N);
            cudaMemcpy(ev, cur_wt, cur_E * sizeof(float), cudaMemcpyDeviceToDevice);

            int64_t* sk; cudaMalloc(&sk, cur_E * sizeof(int64_t));
            float* sv; cudaMalloc(&sv, cur_E * sizeof(float));
            int64_t* uk; cudaMalloc(&uk, cur_E * sizeof(int64_t));
            float* uv; cudaMalloc(&uv, cur_E * sizeof(float));
            int* nu; cudaMalloc(&nu, sizeof(int));
            int new_E = do_sort_reduce_edges(cache, ek, ev, cur_E, sk, sv, uk, uv, nu);
            cudaFree(ek); cudaFree(ev); cudaFree(sk); cudaFree(sv); cudaFree(nu);

            
            int32_t* old_c_off = c_off;
            int32_t* old_c_idx = c_idx;
            float* old_c_wt = c_wt;
            float* old_c_vw = c_vw;

            cudaMalloc(&c_off, (new_N + 1) * sizeof(int32_t));
            cudaMalloc(&c_idx, new_E * sizeof(int32_t));
            cudaMalloc(&c_wt, new_E * sizeof(float));
            int* cnt; cudaMalloc(&cnt, new_N * sizeof(int));
            do_build_csr(cache, uk, uv, new_E, new_N, cnt, c_off, c_idx, c_wt);
            cudaFree(uk); cudaFree(uv); cudaFree(cnt);

            cudaMalloc(&c_vw, new_N * sizeof(float));
            do_compute_new_vw(cur_vw, d_comm, c_vw, cur_N, new_N);

            
            if (old_c_off) { cudaFree(old_c_off); cudaFree(old_c_idx); cudaFree(old_c_wt); cudaFree(old_c_vw); }

            cudaFree(d_comm);

            cur_N = new_N; cur_E = new_E;
            cur_off = c_off;
            cur_idx2 = c_idx;
            cur_wt = c_wt;
            cur_vw = c_vw;
        }

        
        {
            float* ref_sigma; cudaMalloc(&ref_sigma, orig_N * sizeof(float));
            int* d_ref_changed; cudaMalloc(&d_ref_changed, sizeof(int));
            int refine_max = (orig_N > 1000000) ? 10 : 50;
            for (int iter = 0; iter < refine_max; iter++) {
                cudaMemset(d_ref_changed, 0, sizeof(int));
                for (int c = 0; c < 2; c++) {
                    do_recompute_sigma_tot(d_par, orig_vw, ref_sigma, orig_N);
                    do_local_move(offsets, indices, ew,
                        d_par, ref_sigma, orig_vw,
                        orig_N, total_weight, resolution_f, d_ref_changed, c, 2);
                }
                int h_changed = 0;
                cudaMemcpy(&h_changed, d_ref_changed, sizeof(int), cudaMemcpyDeviceToHost);
                if (!h_changed) break;
            }
            cudaFree(ref_sigma);
            cudaFree(d_ref_changed);
        }

        float par_Q = do_compute_modularity(offsets, indices, ew,
            d_par, orig_vw, orig_N, total_weight, resolution_f);
        if (par_Q > overall_best_Q) {
            overall_best_Q = par_Q;
            cudaMemcpy(clusters, d_par, orig_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            actual_level_count = (par_levels > 0) ? (int64_t)par_levels : 1;
        }

        cudaFree(d_par);
        if (c_off) { cudaFree(c_off); cudaFree(c_idx); cudaFree(c_wt); cudaFree(c_vw); }
    }

    int* ru; cudaMalloc(&ru, (size_t)orig_N * sizeof(int));
    int* rp; cudaMalloc(&rp, (size_t)orig_N * sizeof(int));
    do_renumber_communities(cache, clusters, ru, rp, orig_N);
    cudaFree(ru); cudaFree(rp);

    float modularity = do_compute_modularity(offsets, indices, ew,
        clusters, orig_vw, orig_N, total_weight, resolution_f);

    return {(std::size_t)actual_level_count, (double)modularity};
}

}  
