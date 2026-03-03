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
#include <algorithm>
#include <vector>

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define LM_BLOCK 64
#define HASH_CAP 31
#define SEQ_HASH_CAP 64
#define BATCH_MAX_N 256

struct Cache : Cacheable {
    void* cub_temp = nullptr;
    size_t cub_temp_size = 0;

    void ensure_cub(size_t needed) {
        if (needed > cub_temp_size) {
            if (cub_temp) cudaFree(cub_temp);
            cub_temp_size = needed * 2;
            cudaMalloc(&cub_temp, cub_temp_size);
        }
    }

    ~Cache() override {
        if (cub_temp) cudaFree(cub_temp);
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



void launch_compute_vertex_weights(const int32_t* offsets, const float* weights, float* vw, int N) {
    compute_vertex_weights_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(offsets, weights, vw, N);
}

void launch_init_communities(int32_t* comm, int N) {
    init_communities_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(comm, N);
}

void launch_reduce_sum(const float* input, float* output, int N) {
    cudaMemset(output, 0, sizeof(float));
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (grid > 1024) grid = 1024;
    reduce_sum_kernel<<<grid, BLOCK_SIZE>>>(input, output, N);
}

void launch_recompute_sigma_tot(const int32_t* community, const float* vw, float* sigma_tot, int N) {
    cudaMemset(sigma_tot, 0, N * sizeof(float));
    recompute_sigma_tot_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(community, vw, sigma_tot, N);
}

void launch_local_move(
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

void launch_compute_best_move(
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

void launch_count_updown(
    const int32_t* community, const int32_t* best_comm, const float* best_gain,
    int N, bool up_down, int* count
) {
    if (N <= 0) return;
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    count_updown_kernel<<<grid, BLOCK_SIZE>>>(community, best_comm, best_gain, N, up_down, count);
}

void launch_apply_updown(
    int32_t* community, const int32_t* best_comm, const float* best_gain,
    int N, bool up_down
) {
    if (N <= 0) return;
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    apply_updown_kernel<<<grid, BLOCK_SIZE>>>(community, best_comm, best_gain, N, up_down);
}

void launch_sequential_local_move(
    const int32_t* offsets, const int32_t* indices, const float* weights,
    int32_t* community, float* sigma_tot, const float* vw,
    int N, float tw, float res, int* changed, int start_offset
) {
    if (N <= 0) return;
    sequential_local_move_kernel<<<1, 1>>>(
        offsets, indices, weights, community, sigma_tot, vw,
        N, tw, res, changed, start_offset);
}

void launch_batch_sequential_louvain(
    const int32_t* offsets, const int32_t* indices, const float* weights,
    const float* vw, int N, float tw, float res,
    int32_t* all_comm, float* all_sigma, float* all_mod, int num_restarts
) {
    if (num_restarts <= 0 || N <= 0) return;
    batch_sequential_louvain_kernel<<<num_restarts, 1>>>(
        offsets, indices, weights, vw, N, tw, res,
        all_comm, all_sigma, all_mod, num_restarts);
}

int launch_renumber_communities(int32_t* community, int* used, int* prefix, int N, Cache& cache) {
    int grid = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMemset(used, 0, N * sizeof(int));
    mark_communities_kernel<<<grid, BLOCK_SIZE>>>(community, used, N);

    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, used, prefix, N);
    cache.ensure_cub(temp_bytes);
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, temp_bytes, used, prefix, N);

    int h_last_prefix, h_last_used;
    cudaMemcpy(&h_last_prefix, prefix + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_last_used, used + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int new_N = h_last_prefix + h_last_used;

    apply_renumber_kernel<<<grid, BLOCK_SIZE>>>(community, prefix, N);

    return new_N;
}

int launch_sort_reduce_edges(
    int64_t* keys_in, float* vals_in, int E,
    int64_t* sort_keys_out, float* sort_vals_out,
    int64_t* unique_keys_out, float* unique_vals_out,
    int* d_num_unique, Cache& cache
) {
    size_t sort_temp = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sort_temp, keys_in, sort_keys_out, vals_in, sort_vals_out, E);
    cache.ensure_cub(sort_temp);
    cub::DeviceRadixSort::SortPairs(cache.cub_temp, sort_temp, keys_in, sort_keys_out, vals_in, sort_vals_out, E);

    size_t reduce_temp = 0;
    cub::DeviceReduce::ReduceByKey(nullptr, reduce_temp, sort_keys_out, unique_keys_out,
        sort_vals_out, unique_vals_out, d_num_unique, ::cuda::std::plus<float>{}, E);
    cache.ensure_cub(reduce_temp);
    cub::DeviceReduce::ReduceByKey(cache.cub_temp, reduce_temp, sort_keys_out, unique_keys_out,
        sort_vals_out, unique_vals_out, d_num_unique, ::cuda::std::plus<float>{}, E);

    int h_num_unique;
    cudaMemcpy(&h_num_unique, d_num_unique, sizeof(int), cudaMemcpyDeviceToHost);
    return h_num_unique;
}

void launch_build_csr(
    const int64_t* unique_keys, const float* unique_vals,
    int new_E, int new_N,
    int* counts, int32_t* new_offsets, int32_t* new_indices, float* new_weights,
    Cache& cache
) {
    int grid_e = (new_E + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaMemset(counts, 0, new_N * sizeof(int));
    count_src_edges_kernel<<<grid_e, BLOCK_SIZE>>>(unique_keys, counts, new_E);

    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, counts, (int*)new_offsets, new_N + 1);
    cache.ensure_cub(temp_bytes);
    cub::DeviceScan::ExclusiveSum(cache.cub_temp, temp_bytes, counts, (int*)new_offsets, new_N);

    int h_new_E = new_E;
    cudaMemcpy(new_offsets + new_N, &h_new_E, sizeof(int32_t), cudaMemcpyHostToDevice);

    fill_csr_kernel<<<grid_e, BLOCK_SIZE>>>(unique_keys, unique_vals, (int*)new_offsets, new_indices, new_weights, new_E);
}

void launch_map_edges(
    const int32_t* offsets, const int32_t* indices, const int32_t* community,
    int64_t* edge_keys, int N
) {
    map_edges_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(offsets, indices, community, edge_keys, N);
}

void launch_compute_new_vw(const float* old_vw, const int32_t* community, float* new_vw, int old_N, int new_N) {
    cudaMemset(new_vw, 0, new_N * sizeof(float));
    compute_new_vw_kernel<<<(old_N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(old_vw, community, new_vw, old_N);
}

void launch_compose_communities(int32_t* final_comm, const int32_t* level_comm, int orig_N) {
    compose_communities_kernel<<<(orig_N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(final_comm, level_comm, orig_N);
}

float launch_compute_modularity(
    const int32_t* offsets, const int32_t* indices, const float* weights,
    const int32_t* clusters, const float* vw, int N, float total_weight, float resolution) {
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

void launch_convert_double_to_float(const double* in, float* out, int N) {
    if (N <= 0) return;
    convert_d2f_kernel<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(in, out, N);
}

}  

louvain_result_double_t louvain(const graph32_t& graph,
                                const double* edge_weights,
                                int32_t* clusters,
                                std::size_t max_level,
                                double threshold,
                                double resolution) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t orig_N = graph.number_of_vertices;
    int32_t orig_E = graph.number_of_edges;

    float threshold_f = (float)threshold;
    float resolution_f = (float)resolution;

    
    float* edge_weights_f;
    cudaMalloc(&edge_weights_f, (size_t)orig_E * sizeof(float));
    launch_convert_double_to_float(edge_weights, edge_weights_f, orig_E);

    
    float* orig_vw;
    cudaMalloc(&orig_vw, (size_t)orig_N * sizeof(float));
    launch_compute_vertex_weights(offsets, edge_weights_f, orig_vw, orig_N);

    
    float* tw_buf;
    cudaMalloc(&tw_buf, sizeof(float));
    launch_reduce_sum(orig_vw, tw_buf, orig_N);
    float total_weight;
    cudaMemcpy(&total_weight, tw_buf, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(tw_buf);

    
    launch_init_communities(clusters, orig_N);

    if (total_weight == 0.0f || orig_N <= 1) {
        cudaFree(edge_weights_f);
        cudaFree(orig_vw);
        return {0, 0.0};
    }

    int* d_count;
    cudaMalloc(&d_count, sizeof(int));
    float overall_best_Q = -1.0f;
    int64_t actual_level_count = 1;

    
    if (orig_N <= 256) {
        int num_restarts = 16384;
        int32_t* all_comm;
        float* all_sigma;
        float* all_mod;
        cudaMalloc(&all_comm, (size_t)num_restarts * orig_N * sizeof(int32_t));
        cudaMalloc(&all_sigma, (size_t)num_restarts * orig_N * sizeof(float));
        cudaMalloc(&all_mod, (size_t)num_restarts * sizeof(float));

        launch_batch_sequential_louvain(offsets, indices, edge_weights_f,
            orig_vw, orig_N, total_weight, resolution_f,
            all_comm, all_sigma, all_mod, num_restarts);
        cudaDeviceSynchronize();

        std::vector<float> h_mod(num_restarts);
        cudaMemcpy(h_mod.data(), all_mod, num_restarts * sizeof(float), cudaMemcpyDeviceToHost);
        std::vector<int> sorted_idx(num_restarts);
        for (int i = 0; i < num_restarts; i++) sorted_idx[i] = i;
        std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) { return h_mod[a] > h_mod[b]; });

        int topK = std::min(200, num_restarts);
        int32_t* trial_comm;
        int* d_lc;
        cudaMalloc(&trial_comm, (size_t)orig_N * sizeof(int32_t));
        cudaMalloc(&d_lc, sizeof(int));

        for (int ki = 0; ki < topK; ki++) {
            int idx_k = sorted_idx[ki];
            cudaMemcpy(trial_comm, all_comm + (int64_t)idx_k * orig_N,
                       orig_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);

            int32_t* ml_comm;
            cudaMalloc(&ml_comm, (size_t)orig_N * sizeof(int32_t));
            cudaMemcpy(ml_comm, trial_comm, orig_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);

            int cur_N = orig_N, cur_E = orig_E;
            const int32_t* cur_off = offsets;
            const int32_t* cur_idx = indices;
            const float* cur_wt = edge_weights_f;
            const float* cur_vw = orig_vw;
            int32_t* c_off2 = nullptr;
            int32_t* c_idx2 = nullptr;
            float* c_wt2 = nullptr;
            float* c_vw2 = nullptr;
            float level_best = h_mod[idx_k];
            bool first_level = true;

            for (int level = 0; level < (int)max_level; level++) {
                
                int32_t* used_buf;
                int32_t* pref_buf;
                cudaMalloc(&used_buf, (size_t)cur_N * sizeof(int32_t));
                cudaMalloc(&pref_buf, (size_t)cur_N * sizeof(int32_t));
                int32_t* ctr = ml_comm;
                int new_N = launch_renumber_communities(ctr, (int*)used_buf, (int*)pref_buf, cur_N, cache);
                cudaFree(used_buf);
                cudaFree(pref_buf);

                if (new_N >= cur_N) break;

                if (first_level) {
                    cudaMemcpy(trial_comm, ctr, orig_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);
                    first_level = false;
                } else {
                    launch_compose_communities(trial_comm, ctr, orig_N);
                }

                
                int64_t* ek;
                float* ev;
                cudaMalloc(&ek, (size_t)cur_E * sizeof(int64_t));
                cudaMalloc(&ev, (size_t)cur_E * sizeof(float));
                launch_map_edges(cur_off, cur_idx, ctr, ek, cur_N);
                cudaMemcpy(ev, cur_wt, cur_E * sizeof(float), cudaMemcpyDeviceToDevice);

                
                int64_t* sk;
                float* sv;
                int64_t* uk;
                float* uv;
                int* nu;
                cudaMalloc(&sk, (size_t)cur_E * sizeof(int64_t));
                cudaMalloc(&sv, (size_t)cur_E * sizeof(float));
                cudaMalloc(&uk, (size_t)cur_E * sizeof(int64_t));
                cudaMalloc(&uv, (size_t)cur_E * sizeof(float));
                cudaMalloc(&nu, sizeof(int));
                int new_E = launch_sort_reduce_edges(ek, ev, cur_E, sk, sv, uk, uv, nu, cache);

                
                int32_t* old_c_off2 = c_off2;
                int32_t* old_c_idx2 = c_idx2;
                float* old_c_wt2 = c_wt2;
                float* old_c_vw2 = c_vw2;

                
                cudaMalloc(&c_off2, (size_t)(new_N + 1) * sizeof(int32_t));
                cudaMalloc(&c_idx2, (size_t)new_E * sizeof(int32_t));
                cudaMalloc(&c_wt2, (size_t)new_E * sizeof(float));
                int32_t* cnt;
                cudaMalloc(&cnt, (size_t)new_N * sizeof(int32_t));
                launch_build_csr(uk, uv, new_E, new_N, (int*)cnt, c_off2, c_idx2, c_wt2, cache);
                cudaFree(cnt);

                
                cudaMalloc(&c_vw2, (size_t)new_N * sizeof(float));
                launch_compute_new_vw(cur_vw, ctr, c_vw2, cur_N, new_N);

                
                if (old_c_off2) { cudaFree(old_c_off2); cudaFree(old_c_idx2); cudaFree(old_c_wt2); }
                if (old_c_vw2) cudaFree(old_c_vw2);

                
                cudaFree(ek);
                cudaFree(ev);
                cudaFree(sk);
                cudaFree(sv);
                cudaFree(uk);
                cudaFree(uv);
                cudaFree(nu);

                
                cur_N = new_N; cur_E = new_E;
                cur_off = c_off2;
                cur_idx = c_idx2;
                cur_wt = c_wt2;
                cur_vw = c_vw2;

                
                cudaFree(ml_comm);
                cudaMalloc(&ml_comm, (size_t)cur_N * sizeof(int32_t));
                float* ls;
                cudaMalloc(&ls, (size_t)cur_N * sizeof(float));
                launch_init_communities(ml_comm, cur_N);

                for (int iter = 0; iter < 200; iter++) {
                    launch_recompute_sigma_tot(ml_comm, cur_vw, ls, cur_N);
                    cudaMemset(d_lc, 0, sizeof(int));
                    launch_sequential_local_move(cur_off, cur_idx, cur_wt,
                        ml_comm, ls, cur_vw, cur_N, total_weight, resolution_f, d_lc, 0);
                    int h_changed = 0;
                    cudaMemcpy(&h_changed, d_lc, sizeof(int), cudaMemcpyDeviceToHost);
                    if (!h_changed) break;
                }
                cudaFree(ls);

                launch_compose_communities(trial_comm, ml_comm, orig_N);
                float new_Q = launch_compute_modularity(offsets, indices,
                    edge_weights_f, trial_comm, orig_vw, orig_N, total_weight, resolution_f);
                if (new_Q <= level_best) break;
                level_best = new_Q;
            }

            
            float trial_Q = launch_compute_modularity(offsets, indices,
                edge_weights_f, trial_comm, orig_vw, orig_N, total_weight, resolution_f);
            if (trial_Q > overall_best_Q) {
                overall_best_Q = trial_Q;
                cudaMemcpy(clusters, trial_comm, orig_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            }

            
            cudaFree(ml_comm);
            if (c_off2) { cudaFree(c_off2); cudaFree(c_idx2); cudaFree(c_wt2); cudaFree(c_vw2); }
        }

        cudaFree(trial_comm);
        cudaFree(d_lc);
        cudaFree(all_comm);
        cudaFree(all_sigma);
        cudaFree(all_mod);
    }

    
    {
        int32_t* par_comm;
        cudaMalloc(&par_comm, (size_t)orig_N * sizeof(int32_t));
        launch_init_communities(par_comm, orig_N);

        int cur_N = orig_N, cur_E = orig_E;
        const int32_t* cur_off = offsets;
        const int32_t* cur_idx = indices;
        const float* cur_wt = edge_weights_f;
        const float* cur_vw = orig_vw;
        int32_t* c_off_p = nullptr;
        int32_t* c_idx_p = nullptr;
        float* c_wt_p = nullptr;
        float* c_vw_p = nullptr;
        float best_modularity = -1.0f;
        int par_levels = 0;

        for (int level = 0; level < (int)max_level; level++) {
            
            int32_t* comm_p;
            float* sigma_p;
            int32_t* bm_comm_p;
            float* bm_gain_p;
            int32_t* best_saved_p;
            cudaMalloc(&comm_p, (size_t)cur_N * sizeof(int32_t));
            cudaMalloc(&sigma_p, (size_t)cur_N * sizeof(float));
            cudaMalloc(&bm_comm_p, (size_t)cur_N * sizeof(int32_t));
            cudaMalloc(&bm_gain_p, (size_t)cur_N * sizeof(float));
            cudaMalloc(&best_saved_p, (size_t)cur_N * sizeof(int32_t));

            launch_init_communities(comm_p, cur_N);
            float new_Q = launch_compute_modularity(cur_off, cur_idx, cur_wt, comm_p,
                cur_vw, cur_N, total_weight, resolution_f);
            float cur_Q = new_Q - 1;
            cudaMemcpy(best_saved_p, comm_p, cur_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);

            bool up_down = true;
            while (new_Q > cur_Q + threshold_f) {
                cur_Q = new_Q;
                launch_recompute_sigma_tot(comm_p, cur_vw, sigma_p, cur_N);
                launch_compute_best_move(cur_off, cur_idx, cur_wt, comm_p, sigma_p,
                    cur_vw, cur_N, total_weight, resolution_f, bm_comm_p, bm_gain_p);
                cudaMemset(d_count, 0, sizeof(int));
                launch_count_updown(comm_p, bm_comm_p, bm_gain_p, cur_N, up_down, d_count);
                int h_count;
                cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
                bool eff = up_down; if (h_count == 0) eff = !up_down;
                launch_apply_updown(comm_p, bm_comm_p, bm_gain_p, cur_N, eff);
                up_down = !up_down;
                new_Q = launch_compute_modularity(cur_off, cur_idx, cur_wt, comm_p,
                    cur_vw, cur_N, total_weight, resolution_f);
                if (new_Q > cur_Q) cudaMemcpy(best_saved_p, comm_p, cur_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            }
            cudaMemcpy(comm_p, best_saved_p, cur_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);

            
            cudaFree(sigma_p);
            cudaFree(bm_comm_p);
            cudaFree(bm_gain_p);
            cudaFree(best_saved_p);

            if (cur_Q <= best_modularity) {
                cudaFree(comm_p);
                break;
            }
            best_modularity = cur_Q;

            int32_t* used_p;
            int32_t* pref_p;
            cudaMalloc(&used_p, (size_t)cur_N * sizeof(int32_t));
            cudaMalloc(&pref_p, (size_t)cur_N * sizeof(int32_t));
            int new_N = launch_renumber_communities(comm_p, (int*)used_p, (int*)pref_p, cur_N, cache);
            cudaFree(used_p);
            cudaFree(pref_p);

            if (new_N >= cur_N) {
                cudaFree(comm_p);
                break;
            }

            if (level == 0) cudaMemcpy(par_comm, comm_p, orig_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            else launch_compose_communities(par_comm, comm_p, orig_N);
            par_levels = level + 1;

            if (level + 1 >= (int)max_level) {
                cudaFree(comm_p);
                break;
            }

            
            int64_t* ek;
            float* ev;
            cudaMalloc(&ek, (size_t)cur_E * sizeof(int64_t));
            cudaMalloc(&ev, (size_t)cur_E * sizeof(float));
            launch_map_edges(cur_off, cur_idx, comm_p, ek, cur_N);
            cudaMemcpy(ev, cur_wt, cur_E * sizeof(float), cudaMemcpyDeviceToDevice);

            int64_t* sk;
            float* sv;
            int64_t* uk;
            float* uv;
            int* nu;
            cudaMalloc(&sk, (size_t)cur_E * sizeof(int64_t));
            cudaMalloc(&sv, (size_t)cur_E * sizeof(float));
            cudaMalloc(&uk, (size_t)cur_E * sizeof(int64_t));
            cudaMalloc(&uv, (size_t)cur_E * sizeof(float));
            cudaMalloc(&nu, sizeof(int));
            int new_E = launch_sort_reduce_edges(ek, ev, cur_E, sk, sv, uk, uv, nu, cache);

            
            int32_t* old_c_off_p = c_off_p;
            int32_t* old_c_idx_p = c_idx_p;
            float* old_c_wt_p = c_wt_p;
            float* old_c_vw_p = c_vw_p;

            cudaMalloc(&c_off_p, (size_t)(new_N + 1) * sizeof(int32_t));
            cudaMalloc(&c_idx_p, (size_t)new_E * sizeof(int32_t));
            cudaMalloc(&c_wt_p, (size_t)new_E * sizeof(float));
            int32_t* cnt;
            cudaMalloc(&cnt, (size_t)new_N * sizeof(int32_t));
            launch_build_csr(uk, uv, new_E, new_N, (int*)cnt, c_off_p, c_idx_p, c_wt_p, cache);
            cudaFree(cnt);

            cudaMalloc(&c_vw_p, (size_t)new_N * sizeof(float));
            launch_compute_new_vw(cur_vw, comm_p, c_vw_p, cur_N, new_N);

            
            if (old_c_off_p) { cudaFree(old_c_off_p); cudaFree(old_c_idx_p); cudaFree(old_c_wt_p); }
            if (old_c_vw_p) cudaFree(old_c_vw_p);

            
            cudaFree(ek);
            cudaFree(ev);
            cudaFree(sk);
            cudaFree(sv);
            cudaFree(uk);
            cudaFree(uv);
            cudaFree(nu);

            cur_N = new_N; cur_E = new_E;
            cur_off = c_off_p;
            cur_idx = c_idx_p;
            cur_wt = c_wt_p;
            cur_vw = c_vw_p;

            cudaFree(comm_p);
        }

        
        {
            float* ref_sigma;
            cudaMalloc(&ref_sigma, (size_t)orig_N * sizeof(float));
            int* d_ref_changed;
            cudaMalloc(&d_ref_changed, sizeof(int));
            int refine_max = (orig_N > 1000000) ? 10 : 50;
            for (int iter = 0; iter < refine_max; iter++) {
                cudaMemset(d_ref_changed, 0, sizeof(int));
                for (int c = 0; c < 2; c++) {
                    launch_recompute_sigma_tot(par_comm, orig_vw, ref_sigma, orig_N);
                    launch_local_move(offsets, indices, edge_weights_f,
                        par_comm, ref_sigma, orig_vw,
                        orig_N, total_weight, resolution_f, d_ref_changed, c, 2);
                }
                int h_changed = 0;
                cudaMemcpy(&h_changed, d_ref_changed, sizeof(int), cudaMemcpyDeviceToHost);
                if (!h_changed) break;
            }
            cudaFree(ref_sigma);
            cudaFree(d_ref_changed);
        }

        float par_Q = launch_compute_modularity(offsets, indices,
            edge_weights_f, par_comm, orig_vw, orig_N, total_weight, resolution_f);
        if (par_Q > overall_best_Q) {
            overall_best_Q = par_Q;
            cudaMemcpy(clusters, par_comm, orig_N * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            actual_level_count = (par_levels > 0) ? (int64_t)par_levels : 1;
        }

        cudaFree(par_comm);
        if (c_off_p) { cudaFree(c_off_p); cudaFree(c_idx_p); cudaFree(c_wt_p); cudaFree(c_vw_p); }
    }

    int* ru; cudaMalloc(&ru, (size_t)orig_N * sizeof(int));
    int* rp; cudaMalloc(&rp, (size_t)orig_N * sizeof(int));
    launch_renumber_communities(clusters, ru, rp, orig_N, cache);
    cudaFree(ru); cudaFree(rp);

    
    float modularity = launch_compute_modularity(offsets, indices,
        edge_weights_f, clusters, orig_vw, orig_N, total_weight, resolution_f);

    
    cudaFree(d_count);
    cudaFree(edge_weights_f);
    cudaFree(orig_vw);

    return {(std::size_t)actual_level_count, (double)modularity};
}

}  
