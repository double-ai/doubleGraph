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
#include <cstdint>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>

namespace aai {

namespace {

struct Cache : Cacheable {
    int32_t* d_counter = nullptr;

    Cache() {
        cudaMalloc(&d_counter, sizeof(int32_t));
    }

    ~Cache() override {
        if (d_counter) cudaFree(d_counter);
    }
};



__device__ inline int gallop_lower_bound(
    const int32_t* __restrict__ arr, int lo, int hi, int32_t key
) {
    if (lo >= hi) return hi;
    
    int step = 1;
    while (lo + step < hi && arr[lo + step] < key) {
        step <<= 1;
    }
    
    int bs_lo = lo + (step >> 1);
    int bs_hi = (lo + step < hi) ? lo + step + 1 : hi;
    while (bs_lo < bs_hi) {
        int mid = (bs_lo + bs_hi) >> 1;
        if (arr[mid] < key) bs_lo = mid + 1;
        else bs_hi = mid;
    }
    return bs_lo;
}




__global__ void eager_cosine_topk_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    double* __restrict__ out_scores,
    int32_t* __restrict__ global_counter,
    int topk
) {
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;

    int u = seeds[sid];
    int u_start = offsets[u];
    int u_end = offsets[u + 1];
    int u_deg = u_end - u_start;

    for (int i = u_start; i < u_end; i++) {
        if (*global_counter >= topk) return;

        int k = indices[i];
        int k_start = offsets[k];
        int k_end = offsets[k + 1];

        for (int j = k_start + threadIdx.x; j < k_end; j += blockDim.x) {
            if (*global_counter >= topk) break;

            int v = indices[j];
            if (v == u) continue;

            int v_start = offsets[v];
            int v_end = offsets[v + 1];
            int v_deg = v_end - v_start;

            int common = 0;

            
            if (u_deg <= v_deg) {
                
                int large_pos = v_start;
                for (int p = u_start; p < u_end && common <= 1; p++) {
                    int w = indices[p];
                    large_pos = gallop_lower_bound(indices, large_pos, v_end, w);
                    if (large_pos < v_end && indices[large_pos] == w) {
                        common++;
                        large_pos++;
                    }
                }
            } else {
                
                int large_pos = u_start;
                for (int p = v_start; p < v_end && common <= 1; p++) {
                    int w = indices[p];
                    large_pos = gallop_lower_bound(indices, large_pos, u_end, w);
                    if (large_pos < u_end && indices[large_pos] == w) {
                        common++;
                        large_pos++;
                    }
                }
            }

            if (common == 1) {
                int idx = atomicAdd(global_counter, 1);
                if (idx < topk) {
                    out_first[idx] = u;
                    out_second[idx] = v;
                    out_scores[idx] = 1.0;
                }
            }
        }
    }
}


__device__ inline void ht_insert_or_increment(
    int32_t* __restrict__ keys, int32_t* __restrict__ counts,
    int cap, int32_t key
) {
    uint32_t h = ((uint32_t)key * 2654435761u) & (cap - 1);
    for (int probe = 0; probe < 256; probe++) {
        uint32_t idx = (h + probe) & (cap - 1);
        int32_t old = atomicCAS(&keys[idx], -1, key);
        if (old == -1 || old == key) {
            atomicAdd(&counts[idx], 1);
            return;
        }
    }
}

__global__ void compute_ht_sizes_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int num_seeds, int num_vertices,
    int64_t* __restrict__ ht_sizes,
    int64_t* __restrict__ raw_totals
) {
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int u = seeds[sid];
    int u_start = offsets[u], u_end = offsets[u + 1];
    int64_t local_sum = 0;
    for (int i = u_start + threadIdx.x; i < u_end; i += blockDim.x) {
        int k = indices[i];
        local_sum += offsets[k + 1] - offsets[k];
    }
    __shared__ int64_t shared[256];
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int64_t total = shared[0];
        int64_t estimated = total < (int64_t)num_vertices ? total : (int64_t)num_vertices;
        int64_t ht_size = 64;
        while (ht_size < 2 * estimated + 32) ht_size *= 2;
        if (ht_size > 4194304) {
            ht_sizes[sid] = 0;        
            raw_totals[sid] = total;   
        } else {
            ht_sizes[sid] = ht_size;
            raw_totals[sid] = 0;
        }
    }
}

__global__ void init_ht_kernel(int32_t* keys, int64_t total_size) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) keys[idx] = -1;
}

__global__ void count_2hop_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds, int num_seeds,
    int32_t* __restrict__ ht_keys, int32_t* __restrict__ ht_counts,
    const int64_t* __restrict__ ht_offsets, const int64_t* __restrict__ ht_sizes
) {
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int u = seeds[sid];
    int u_start = offsets[u], u_end = offsets[u + 1];
    int32_t* my_keys = ht_keys + ht_offsets[sid];
    int32_t* my_counts = ht_counts + ht_offsets[sid];
    int cap = (int)ht_sizes[sid];
    if (cap <= 0) return;  
    for (int i = u_start; i < u_end; i++) {
        int k = indices[i];
        int k_start = offsets[k], k_end = offsets[k + 1];
        for (int j = k_start + threadIdx.x; j < k_end; j += blockDim.x) {
            int v = indices[j];
            if (v != u) ht_insert_or_increment(my_keys, my_counts, cap, v);
        }
    }
}

__global__ void count_pairs_kernel(
    const int32_t* __restrict__ ht_keys, const int32_t* __restrict__ ht_counts,
    const int64_t* __restrict__ ht_offsets, const int64_t* __restrict__ ht_sizes,
    int num_seeds, int64_t* __restrict__ pair_counts, int64_t* __restrict__ count1_counts
) {
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int cap = (int)ht_sizes[sid];
    if (cap <= 0) { pair_counts[sid] = 0; count1_counts[sid] = 0; return; }
    const int32_t* my_keys = ht_keys + ht_offsets[sid];
    const int32_t* my_counts = ht_counts + ht_offsets[sid];
    int64_t lp = 0, lc1 = 0;
    for (int i = threadIdx.x; i < cap; i += blockDim.x) {
        if (my_keys[i] != -1) { lp++; if (my_counts[i] == 1) lc1++; }
    }
    __shared__ int64_t sp[256], sc[256];
    sp[threadIdx.x] = lp; sc[threadIdx.x] = lc1;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) { sp[threadIdx.x] += sp[threadIdx.x + s]; sc[threadIdx.x] += sc[threadIdx.x + s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { pair_counts[sid] = sp[0]; count1_counts[sid] = sc[0]; }
}

__global__ void extract_and_score_kernel(
    const int32_t* __restrict__ offsets, const int32_t* __restrict__ indices,
    const double* __restrict__ weights, const int32_t* __restrict__ seeds, int num_seeds,
    const int32_t* __restrict__ ht_keys, const int32_t* __restrict__ ht_counts,
    const int64_t* __restrict__ ht_offsets, const int64_t* __restrict__ ht_sizes,
    const int64_t* __restrict__ pair_offsets,
    int32_t* __restrict__ out_first, int32_t* __restrict__ out_second,
    double* __restrict__ out_scores, int32_t* __restrict__ atomic_counters
) {
    int sid = blockIdx.x;
    if (sid >= num_seeds) return;
    int u = seeds[sid];
    int u_start = offsets[u], u_end = offsets[u + 1];
    int cap = (int)ht_sizes[sid];
    if (cap <= 0) return;  
    const int32_t* my_keys = ht_keys + ht_offsets[sid];
    const int32_t* my_counts = ht_counts + ht_offsets[sid];
    int64_t out_base = pair_offsets[sid];

    for (int i = threadIdx.x; i < cap; i += blockDim.x) {
        int v = my_keys[i];
        if (v == -1) continue;
        int cnt = my_counts[i];
        int local_idx = atomicAdd(&atomic_counters[sid], 1);
        int64_t out_idx = out_base + local_idx;
        out_first[out_idx] = u;
        out_second[out_idx] = v;
        if (cnt == 1) {
            out_scores[out_idx] = 1.0;
        } else {
            int v_start = offsets[v], v_end = offsets[v + 1];
            double dot = 0.0, norm_u_sq = 0.0, norm_v_sq = 0.0;
            int pi = u_start, pj = v_start;
            while (pi < u_end && pj < v_end) {
                int a = indices[pi], b = indices[pj];
                if (a == b) {
                    double wu = weights[pi], wv = weights[pj];
                    dot += wu * wv; norm_u_sq += wu * wu; norm_v_sq += wv * wv;
                    pi++; pj++;
                } else if (a < b) pi++; else pj++;
            }
            double denom = sqrt(norm_u_sq) * sqrt(norm_v_sq);
            out_scores[out_idx] = (denom > 0.0) ? (dot / denom) : 0.0;
        }
    }
}

__global__ void fill_sequence_kernel(int32_t* arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = idx;
}

__global__ void gather_kernel(
    const int32_t* __restrict__ sf, const int32_t* __restrict__ ss,
    const double* __restrict__ sc, const int32_t* __restrict__ perm,
    int32_t* __restrict__ df, int32_t* __restrict__ ds, double* __restrict__ dc, int64_t n
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { int32_t j = perm[i]; df[i] = sf[j]; ds[i] = ss[j]; dc[i] = sc[j]; }
}

struct ScoreDescComp {
    const double* scores;
    __host__ __device__ bool operator()(int32_t a, int32_t b) const { return scores[a] > scores[b]; }
};



__global__ void compute_nbr_degrees_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t u_start, int32_t u_end,
    int64_t* __restrict__ degrees
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int u_deg = u_end - u_start;
    if (i < u_deg) {
        int k = indices[u_start + i];
        degrees[i] = (int64_t)(offsets[k + 1] - offsets[k]);
    }
}

__global__ void gather_2hop_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    int32_t u_start, int32_t u_end,
    int32_t* __restrict__ output,
    const int64_t* __restrict__ write_offsets
) {
    int nbr_idx = blockIdx.x;
    int u_deg = u_end - u_start;
    if (nbr_idx >= u_deg) return;
    int k = indices[u_start + nbr_idx];
    int k_start = offsets[k], k_end = offsets[k + 1];
    int k_deg = k_end - k_start;
    int64_t base = write_offsets[nbr_idx];
    for (int j = threadIdx.x; j < k_deg; j += blockDim.x) {
        output[base + j] = indices[k_start + j];
    }
}

__global__ void score_unique_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    int32_t seed, int32_t u_start, int32_t u_end,
    const int32_t* __restrict__ unique_vertices,
    const int32_t* __restrict__ unique_counts,
    int64_t n_unique,
    int32_t* __restrict__ out_first,
    int32_t* __restrict__ out_second,
    double* __restrict__ out_scores,
    int32_t* __restrict__ out_counter
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_unique) return;
    int v = unique_vertices[i];
    if (v == seed) return;  
    int cnt = unique_counts[i];

    int pos = atomicAdd(out_counter, 1);
    out_first[pos] = seed;
    out_second[pos] = v;

    if (cnt == 1) {
        
        out_scores[pos] = 1.0;
    } else {
        int v_start = offsets[v], v_end = offsets[v + 1];
        double dot = 0.0, norm_u_sq = 0.0, norm_v_sq = 0.0;
        int pi = u_start, pj = v_start;
        while (pi < u_end && pj < v_end) {
            int a = indices[pi], b = indices[pj];
            if (a == b) {
                double wu = weights[pi], wv = weights[pj];
                dot += wu * wv; norm_u_sq += wu * wu; norm_v_sq += wv * wv;
                pi++; pj++;
            } else if (a < b) pi++; else pj++;
        }
        double denom = sqrt(norm_u_sq) * sqrt(norm_v_sq);
        out_scores[pos] = (denom > 0.0) ? (dot / denom) : 0.0;
    }
}

}  

similarity_result_double_t cosine_all_pairs_similarity_seg(
    const graph32_t& graph,
    const double* edge_weights,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk_opt) {

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n_verts = graph.number_of_vertices;
    int32_t n_edges = graph.number_of_edges;
    const double* d_weights = edge_weights;

    bool has_topk = topk_opt.has_value();
    int64_t topk = has_topk ? (int64_t)topk_opt.value() : 0;

    
    int32_t num_seeds;
    const int32_t* d_seeds;
    int32_t* d_seeds_alloc = nullptr;

    if (vertices != nullptr && num_vertices > 0) {
        d_seeds = vertices;
        num_seeds = (int32_t)num_vertices;
    } else {
        cudaMalloc(&d_seeds_alloc, (size_t)n_verts * sizeof(int32_t));
        fill_sequence_kernel<<<(n_verts + 255) / 256, 256>>>(d_seeds_alloc, n_verts);
        d_seeds = d_seeds_alloc;
        num_seeds = n_verts;
    }

    if (num_seeds == 0 || n_edges == 0) {
        if (d_seeds_alloc) cudaFree(d_seeds_alloc);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    if (has_topk && topk > 0) {
        int32_t* eager_first;
        int32_t* eager_second;
        double* eager_scores;
        cudaMalloc(&eager_first, (size_t)topk * sizeof(int32_t));
        cudaMalloc(&eager_second, (size_t)topk * sizeof(int32_t));
        cudaMalloc(&eager_scores, (size_t)topk * sizeof(double));

        cudaMemsetAsync(cache.d_counter, 0, sizeof(int32_t));

        eager_cosine_topk_kernel<<<num_seeds, 256>>>(
            d_offsets, d_indices, d_seeds, num_seeds,
            eager_first, eager_second, eager_scores,
            cache.d_counter, (int)topk);

        int32_t found;
        cudaMemcpy(&found, cache.d_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);

        if (found >= (int32_t)topk) {
            if (d_seeds_alloc) cudaFree(d_seeds_alloc);
            return {eager_first, eager_second, eager_scores, (std::size_t)topk};
        }
        
        cudaFree(eager_first);
        cudaFree(eager_second);
        cudaFree(eager_scores);
    }

    

    
    int64_t* d_ht_sizes;
    int64_t* d_raw_totals;
    cudaMalloc(&d_ht_sizes, (size_t)num_seeds * sizeof(int64_t));
    cudaMalloc(&d_raw_totals, (size_t)num_seeds * sizeof(int64_t));

    compute_ht_sizes_kernel<<<num_seeds, 256>>>(
        d_offsets, d_indices, d_seeds, num_seeds, n_verts,
        d_ht_sizes, d_raw_totals);

    
    int64_t* d_ht_offsets;
    cudaMalloc(&d_ht_offsets, ((size_t)num_seeds + 1) * sizeof(int64_t));
    {
        thrust::device_ptr<int64_t> in_ptr(d_ht_sizes);
        thrust::device_ptr<int64_t> out_ptr(d_ht_offsets);
        thrust::inclusive_scan(in_ptr, in_ptr + num_seeds, out_ptr + 1);
        cudaMemsetAsync(d_ht_offsets, 0, sizeof(int64_t));
    }

    int64_t total_ht_size;
    cudaMemcpy(&total_ht_size, d_ht_offsets + num_seeds,
               sizeof(int64_t), cudaMemcpyDeviceToHost);

    
    int32_t* d_ht_keys = nullptr;
    int32_t* d_ht_counts = nullptr;
    if (total_ht_size > 0) {
        cudaMalloc(&d_ht_keys, (size_t)total_ht_size * sizeof(int32_t));
        cudaMalloc(&d_ht_counts, (size_t)total_ht_size * sizeof(int32_t));
        init_ht_kernel<<<(int)((total_ht_size + 255) / 256), 256>>>(d_ht_keys, total_ht_size);
        cudaMemsetAsync(d_ht_counts, 0, (size_t)total_ht_size * sizeof(int32_t));

        count_2hop_kernel<<<num_seeds, 256>>>(
            d_offsets, d_indices, d_seeds, num_seeds,
            d_ht_keys, d_ht_counts, d_ht_offsets, d_ht_sizes);
    }

    
    int64_t* d_pair_counts;
    int64_t* d_count1_counts;
    cudaMalloc(&d_pair_counts, (size_t)num_seeds * sizeof(int64_t));
    cudaMalloc(&d_count1_counts, (size_t)num_seeds * sizeof(int64_t));

    if (total_ht_size > 0) {
        count_pairs_kernel<<<num_seeds, 256>>>(
            d_ht_keys, d_ht_counts, d_ht_offsets, d_ht_sizes,
            num_seeds, d_pair_counts, d_count1_counts);
    } else {
        cudaMemsetAsync(d_pair_counts, 0, (size_t)num_seeds * sizeof(int64_t));
        cudaMemsetAsync(d_count1_counts, 0, (size_t)num_seeds * sizeof(int64_t));
    }

    cudaDeviceSynchronize();
    int64_t hash_total_pairs;
    {
        thrust::device_ptr<int64_t> ptr(d_pair_counts);
        hash_total_pairs = thrust::reduce(ptr, ptr + num_seeds, (int64_t)0);
    }

    
    int32_t* d_hash_first = nullptr;
    int32_t* d_hash_second = nullptr;
    double* d_hash_scores = nullptr;

    if (hash_total_pairs > 0) {
        int64_t* d_pair_offsets;
        cudaMalloc(&d_pair_offsets, ((size_t)num_seeds + 1) * sizeof(int64_t));
        {
            thrust::device_ptr<int64_t> in_ptr(d_pair_counts);
            thrust::device_ptr<int64_t> out_ptr(d_pair_offsets);
            thrust::inclusive_scan(in_ptr, in_ptr + num_seeds, out_ptr + 1);
            cudaMemsetAsync(d_pair_offsets, 0, sizeof(int64_t));
        }

        cudaMalloc(&d_hash_first, (size_t)hash_total_pairs * sizeof(int32_t));
        cudaMalloc(&d_hash_second, (size_t)hash_total_pairs * sizeof(int32_t));
        cudaMalloc(&d_hash_scores, (size_t)hash_total_pairs * sizeof(double));

        int32_t* d_counters;
        cudaMalloc(&d_counters, (size_t)num_seeds * sizeof(int32_t));
        cudaMemsetAsync(d_counters, 0, (size_t)num_seeds * sizeof(int32_t));

        extract_and_score_kernel<<<num_seeds, 256>>>(
            d_offsets, d_indices, d_weights, d_seeds, num_seeds,
            d_ht_keys, d_ht_counts, d_ht_offsets, d_ht_sizes,
            d_pair_offsets,
            d_hash_first, d_hash_second, d_hash_scores, d_counters);

        cudaFree(d_counters);
        cudaFree(d_pair_offsets);
    }

    
    std::vector<int64_t> h_ht_sizes(num_seeds), h_raw_totals(num_seeds);
    cudaMemcpy(h_ht_sizes.data(), d_ht_sizes,
               (size_t)num_seeds * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_raw_totals.data(), d_raw_totals,
               (size_t)num_seeds * sizeof(int64_t), cudaMemcpyDeviceToHost);

    
    cudaFree(d_ht_sizes);
    cudaFree(d_raw_totals);
    cudaFree(d_ht_offsets);
    if (d_ht_keys) cudaFree(d_ht_keys);
    if (d_ht_counts) cudaFree(d_ht_counts);
    cudaFree(d_pair_counts);
    cudaFree(d_count1_counts);

    bool has_fallback = false;
    for (int i = 0; i < num_seeds; i++) {
        if (h_ht_sizes[i] == 0 && h_raw_totals[i] > 0) { has_fallback = true; break; }
    }

    if (!has_fallback) {
        
        if (hash_total_pairs == 0) {
            if (d_seeds_alloc) cudaFree(d_seeds_alloc);
            return {nullptr, nullptr, nullptr, 0};
        }

        if (has_topk && hash_total_pairs > topk) {
            int32_t* final_first;
            int32_t* final_second;
            double* final_scores;
            cudaMalloc(&final_first, (size_t)topk * sizeof(int32_t));
            cudaMalloc(&final_second, (size_t)topk * sizeof(int32_t));
            cudaMalloc(&final_scores, (size_t)topk * sizeof(double));

            int32_t* d_perm;
            cudaMalloc(&d_perm, (size_t)hash_total_pairs * sizeof(int32_t));

            {
                thrust::device_ptr<int32_t> pp(d_perm);
                thrust::sequence(pp, pp + hash_total_pairs);
                ScoreDescComp cmp; cmp.scores = d_hash_scores;
                thrust::sort(pp, pp + hash_total_pairs, cmp);
                int64_t actual = (topk < hash_total_pairs) ? topk : hash_total_pairs;
                if (actual > 0) {
                    gather_kernel<<<(int)((actual + 255) / 256), 256>>>(
                        d_hash_first, d_hash_second, d_hash_scores, d_perm,
                        final_first, final_second, final_scores, actual);
                }
            }

            cudaFree(d_perm);
            cudaFree(d_hash_first);
            cudaFree(d_hash_second);
            cudaFree(d_hash_scores);
            if (d_seeds_alloc) cudaFree(d_seeds_alloc);
            return {final_first, final_second, final_scores, (std::size_t)topk};
        }

        if (d_seeds_alloc) cudaFree(d_seeds_alloc);
        return {d_hash_first, d_hash_second, d_hash_scores, (std::size_t)hash_total_pairs};
    }

    
    std::vector<int32_t*> fb_firsts, fb_seconds;
    std::vector<double*> fb_scores_vec;
    std::vector<int32_t> fb_counts;
    int64_t fb_total = 0;

    for (int i = 0; i < num_seeds; i++) {
        if (h_ht_sizes[i] != 0 || h_raw_totals[i] <= 0) continue;
        int64_t raw_total = h_raw_totals[i];

        
        int32_t h_seed;
        cudaMemcpy(&h_seed, d_seeds + i, sizeof(int32_t), cudaMemcpyDeviceToHost);
        int32_t h_off[2];
        cudaMemcpy(h_off, d_offsets + h_seed, 2 * sizeof(int32_t), cudaMemcpyDeviceToHost);
        int32_t u_start = h_off[0], u_end = h_off[1];
        int u_deg = u_end - u_start;
        if (u_deg == 0) continue;

        
        int64_t* d_nbr_degs;
        cudaMalloc(&d_nbr_degs, (size_t)u_deg * sizeof(int64_t));
        compute_nbr_degrees_kernel<<<(u_deg + 255) / 256, 256>>>(
            d_offsets, d_indices, u_start, u_end, d_nbr_degs);

        int64_t* d_write_offs;
        cudaMalloc(&d_write_offs, ((size_t)u_deg + 1) * sizeof(int64_t));
        {
            thrust::device_ptr<int64_t> in_ptr(d_nbr_degs);
            thrust::device_ptr<int64_t> out_ptr(d_write_offs);
            thrust::inclusive_scan(in_ptr, in_ptr + u_deg, out_ptr + 1);
            cudaMemsetAsync(d_write_offs, 0, sizeof(int64_t));
        }

        
        int32_t* d_gather_buf;
        cudaMalloc(&d_gather_buf, (size_t)raw_total * sizeof(int32_t));
        gather_2hop_kernel<<<u_deg, 256>>>(
            d_offsets, d_indices, u_start, u_end,
            d_gather_buf, d_write_offs);

        cudaFree(d_nbr_degs);
        cudaFree(d_write_offs);

        
        if (raw_total > 1) {
            thrust::device_ptr<int32_t> ptr(d_gather_buf);
            thrust::sort(ptr, ptr + raw_total);
        }

        
        int32_t* d_unique_v;
        int32_t* d_unique_c;
        cudaMalloc(&d_unique_v, (size_t)raw_total * sizeof(int32_t));
        cudaMalloc(&d_unique_c, (size_t)raw_total * sizeof(int32_t));
        cudaDeviceSynchronize();

        int64_t n_unique;
        if (raw_total == 0) {
            n_unique = 0;
        } else {
            thrust::device_ptr<const int32_t> in_ptr(d_gather_buf);
            thrust::device_ptr<int32_t> keys_ptr(d_unique_v);
            thrust::device_ptr<int32_t> counts_ptr(d_unique_c);
            auto end = thrust::reduce_by_key(
                in_ptr, in_ptr + raw_total,
                thrust::constant_iterator<int32_t>(1),
                keys_ptr, counts_ptr);
            n_unique = end.first - keys_ptr;
        }

        cudaFree(d_gather_buf);

        if (n_unique == 0) {
            cudaFree(d_unique_v);
            cudaFree(d_unique_c);
            continue;
        }

        
        int32_t* d_fb_first;
        int32_t* d_fb_second;
        double* d_fb_score;
        cudaMalloc(&d_fb_first, (size_t)n_unique * sizeof(int32_t));
        cudaMalloc(&d_fb_second, (size_t)n_unique * sizeof(int32_t));
        cudaMalloc(&d_fb_score, (size_t)n_unique * sizeof(double));

        int32_t* d_fb_counter;
        cudaMalloc(&d_fb_counter, sizeof(int32_t));
        cudaMemsetAsync(d_fb_counter, 0, sizeof(int32_t));

        {
            int blocks = (int)((n_unique + 255) / 256);
            score_unique_pairs_kernel<<<blocks, 256>>>(
                d_offsets, d_indices, d_weights,
                h_seed, u_start, u_end,
                d_unique_v, d_unique_c, n_unique,
                d_fb_first, d_fb_second, d_fb_score, d_fb_counter);
        }

        cudaFree(d_unique_v);
        cudaFree(d_unique_c);

        int32_t fb_count;
        cudaMemcpy(&fb_count, d_fb_counter, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaFree(d_fb_counter);

        if (fb_count > 0) {
            fb_firsts.push_back(d_fb_first);
            fb_seconds.push_back(d_fb_second);
            fb_scores_vec.push_back(d_fb_score);
            fb_counts.push_back(fb_count);
            fb_total += fb_count;
        } else {
            cudaFree(d_fb_first);
            cudaFree(d_fb_second);
            cudaFree(d_fb_score);
        }
    }

    
    int64_t grand_total = hash_total_pairs + fb_total;
    if (grand_total == 0) {
        if (d_hash_first) cudaFree(d_hash_first);
        if (d_hash_second) cudaFree(d_hash_second);
        if (d_hash_scores) cudaFree(d_hash_scores);
        if (d_seeds_alloc) cudaFree(d_seeds_alloc);
        return {nullptr, nullptr, nullptr, 0};
    }

    int32_t* d_out_first;
    int32_t* d_out_second;
    double* d_out_scores;
    cudaMalloc(&d_out_first, (size_t)grand_total * sizeof(int32_t));
    cudaMalloc(&d_out_second, (size_t)grand_total * sizeof(int32_t));
    cudaMalloc(&d_out_scores, (size_t)grand_total * sizeof(double));

    int64_t offset = 0;
    if (hash_total_pairs > 0) {
        cudaMemcpyAsync(d_out_first, d_hash_first,
                        (size_t)hash_total_pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(d_out_second, d_hash_second,
                        (size_t)hash_total_pairs * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(d_out_scores, d_hash_scores,
                        (size_t)hash_total_pairs * sizeof(double), cudaMemcpyDeviceToDevice);
        offset = hash_total_pairs;
    }

    if (d_hash_first) cudaFree(d_hash_first);
    if (d_hash_second) cudaFree(d_hash_second);
    if (d_hash_scores) cudaFree(d_hash_scores);

    for (size_t fi = 0; fi < fb_firsts.size(); fi++) {
        int32_t cnt = fb_counts[fi];
        cudaMemcpyAsync(d_out_first + offset, fb_firsts[fi],
                        (size_t)cnt * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(d_out_second + offset, fb_seconds[fi],
                        (size_t)cnt * sizeof(int32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpyAsync(d_out_scores + offset, fb_scores_vec[fi],
                        (size_t)cnt * sizeof(double), cudaMemcpyDeviceToDevice);
        offset += cnt;
    }

    for (size_t fi = 0; fi < fb_firsts.size(); fi++) {
        cudaFree(fb_firsts[fi]);
        cudaFree(fb_seconds[fi]);
        cudaFree(fb_scores_vec[fi]);
    }

    if (has_topk && grand_total > topk) {
        int32_t* final_first;
        int32_t* final_second;
        double* final_scores;
        cudaMalloc(&final_first, (size_t)topk * sizeof(int32_t));
        cudaMalloc(&final_second, (size_t)topk * sizeof(int32_t));
        cudaMalloc(&final_scores, (size_t)topk * sizeof(double));

        int32_t* d_perm;
        cudaMalloc(&d_perm, (size_t)grand_total * sizeof(int32_t));

        {
            thrust::device_ptr<int32_t> pp(d_perm);
            thrust::sequence(pp, pp + grand_total);
            ScoreDescComp cmp; cmp.scores = d_out_scores;
            thrust::sort(pp, pp + grand_total, cmp);
            int64_t actual = (topk < grand_total) ? topk : grand_total;
            if (actual > 0) {
                gather_kernel<<<(int)((actual + 255) / 256), 256>>>(
                    d_out_first, d_out_second, d_out_scores, d_perm,
                    final_first, final_second, final_scores, actual);
            }
        }

        cudaFree(d_perm);
        cudaFree(d_out_first);
        cudaFree(d_out_second);
        cudaFree(d_out_scores);
        if (d_seeds_alloc) cudaFree(d_seeds_alloc);
        return {final_first, final_second, final_scores, (std::size_t)topk};
    }

    if (d_seeds_alloc) cudaFree(d_seeds_alloc);
    return {d_out_first, d_out_second, d_out_scores, (std::size_t)grand_total};
}

}  
