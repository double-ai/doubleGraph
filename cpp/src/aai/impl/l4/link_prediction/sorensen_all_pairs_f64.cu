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
#include <cstdint>
#include <optional>

namespace aai {

namespace {

struct Cache : Cacheable {};





__device__ __forceinline__ int lower_bound_dev(const int32_t* arr, int size, int32_t target) {
    int lo = 0, hi = size;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (arr[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}




__global__ void compute_weight_sums_kernel(
    const int32_t* __restrict__ offsets,
    const double* __restrict__ weights,
    double* __restrict__ weight_sums,
    int32_t num_vertices
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int32_t start = offsets[v];
    int32_t end = offsets[v + 1];

    double sum = 0.0;
    for (int i = start; i < end; i++) {
        sum += weights[i];
    }
    weight_sums[v] = sum;
}




__global__ void count_raw_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int64_t* __restrict__ counts
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_seeds) return;

    int32_t u = seeds[idx];
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];

    int64_t count = 0;
    for (int i = u_start; i < u_end; i++) {
        int32_t n = indices[i];
        int32_t deg_n = offsets[n + 1] - offsets[n];
        count += deg_n - 1;
    }
    counts[idx] = count;
}




__global__ void expand_raw_pairs_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    const int64_t* __restrict__ pair_offsets,
    int32_t* __restrict__ out_v
) {
    int seed_idx = blockIdx.x;
    if (seed_idx >= num_seeds) return;

    int32_t u = seeds[seed_idx];
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int64_t base = pair_offsets[seed_idx];

    __shared__ int32_t write_count;
    if (threadIdx.x == 0) write_count = 0;
    __syncthreads();

    for (int i = 0; i < u_end - u_start; i++) {
        int32_t n = indices[u_start + i];
        int32_t n_start = offsets[n];
        int32_t n_end = offsets[n + 1];
        int32_t deg_n = n_end - n_start;

        for (int j = threadIdx.x; j < deg_n; j += blockDim.x) {
            int32_t v = indices[n_start + j];
            if (v != u) {
                int32_t pos = atomicAdd(&write_count, 1);
                out_v[base + pos] = v;
            }
        }
    }
}




__global__ void mark_unique_kernel(
    const int32_t* __restrict__ sorted_v,
    const int64_t* __restrict__ seg_offsets,
    int32_t num_seeds,
    int64_t total_raw,
    int32_t* __restrict__ is_first
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_raw) return;

    int lo = 0, hi = num_seeds;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (seg_offsets[mid + 1] <= idx) lo = mid + 1;
        else hi = mid;
    }
    int seg = lo;

    int64_t seg_start = seg_offsets[seg];

    if (idx == seg_start) {
        is_first[idx] = 1;
    } else if (sorted_v[idx] != sorted_v[idx - 1]) {
        is_first[idx] = 1;
    } else {
        is_first[idx] = 0;
    }
}




__global__ void compact_unique_kernel(
    const int32_t* __restrict__ sorted_v,
    const int32_t* __restrict__ prefix_sum,
    const int32_t* __restrict__ is_first,
    int64_t total_raw,
    const int32_t* __restrict__ seeds,
    const int64_t* __restrict__ seg_offsets,
    int32_t num_seeds,
    int32_t* __restrict__ unique_first,
    int32_t* __restrict__ unique_second,
    int32_t* __restrict__ unique_group_id
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_raw) return;

    if (is_first[idx]) {
        int32_t pos = prefix_sum[idx];

        int lo = 0, hi = num_seeds;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (seg_offsets[mid + 1] <= idx) lo = mid + 1;
            else hi = mid;
        }

        unique_first[pos] = seeds[lo];
        unique_second[pos] = sorted_v[idx];
        unique_group_id[pos] = lo;
    }
}




__global__ void find_group_boundaries_kernel(
    const int32_t* __restrict__ unique_group_id,
    int32_t num_unique,
    int32_t num_seeds,
    int32_t* __restrict__ group_offsets  
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx == 0) {
        
        group_offsets[unique_group_id[0]] = 0;
        
        group_offsets[num_seeds] = num_unique;
    }

    if (idx < num_unique - 1) {
        if (unique_group_id[idx] != unique_group_id[idx + 1]) {
            group_offsets[unique_group_id[idx + 1]] = idx + 1;
        }
    }
}


__global__ void init_group_offsets_kernel(
    int32_t* __restrict__ group_offsets,
    int32_t num_seeds
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= num_seeds) {
        group_offsets[idx] = 0;
    }
}


__global__ void fix_group_offsets_kernel(
    int32_t* __restrict__ group_offsets,
    int32_t num_seeds
) {
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = num_seeds - 1; i >= 0; i--) {
            if (group_offsets[i] == 0 && group_offsets[i + 1] > 0) {
                
                
                
            }
        }
        
        
        
        for (int i = num_seeds - 1; i >= 0; i--) {
            if (group_offsets[i] > group_offsets[i + 1]) {
                group_offsets[i] = group_offsets[i + 1];
            }
        }
    }
}




__global__ void compute_scores_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ weight_sums,
    const int32_t* __restrict__ pair_first,
    const int32_t* __restrict__ pair_second,
    double* __restrict__ scores,
    const int32_t* __restrict__ group_starts,
    int32_t num_groups
) {
    int group = blockIdx.x;
    if (group >= num_groups) return;

    int32_t g_start = group_starts[group];
    int32_t g_end = group_starts[group + 1];
    int32_t group_size = g_end - g_start;

    if (group_size == 0) return;

    int32_t u = pair_first[g_start];
    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t deg_u = u_end - u_start;

    extern __shared__ char smem[];
    int32_t* s_neighbors = (int32_t*)smem;
    size_t neighbors_bytes = (size_t)deg_u * sizeof(int32_t);
    size_t aligned_offset = (neighbors_bytes + 7) & ~(size_t)7;
    double* s_weights = (double*)(smem + aligned_offset);

    for (int i = threadIdx.x; i < deg_u; i += blockDim.x) {
        s_neighbors[i] = indices[u_start + i];
        s_weights[i] = weights[u_start + i];
    }

    __shared__ double s_ws_u;
    if (threadIdx.x == 0) {
        s_ws_u = weight_sums[u];
    }
    __syncthreads();

    double ws_u = s_ws_u;

    for (int t = threadIdx.x; t < group_size; t += blockDim.x) {
        int32_t pair_idx = g_start + t;
        int32_t v = pair_second[pair_idx];

        int32_t v_start = offsets[v];
        int32_t v_end = offsets[v + 1];
        int32_t deg_v = v_end - v_start;

        double intersection_sum = 0.0;

        if (deg_u > 0 && deg_v > 0) {
            int32_t u_first = s_neighbors[0];
            int32_t u_last = s_neighbors[deg_u - 1];
            int32_t v_first = indices[v_start];
            int32_t v_last = indices[v_end - 1];

            if (u_first <= v_last && v_first <= u_last) {
                int i = 0;
                if (v_first > u_first) {
                    i = lower_bound_dev(s_neighbors, deg_u, v_first);
                }

                int j = v_start;
                if (i < deg_u && s_neighbors[i] > v_first) {
                    j = v_start + lower_bound_dev(indices + v_start, deg_v, s_neighbors[i]);
                }

                while (i < deg_u && j < v_end) {
                    int32_t nu = s_neighbors[i];
                    int32_t nv = indices[j];
                    if (nu < nv) {
                        i++;
                    } else if (nu > nv) {
                        j++;
                    } else {
                        double wu = s_weights[i];
                        double wv = weights[j];
                        intersection_sum += (wu < wv) ? wu : wv;
                        i++;
                        j++;
                    }
                }
            }
        }

        double denom = ws_u + weight_sums[v];
        scores[pair_idx] = (denom > 0.0) ? (2.0 * intersection_sum / denom) : 0.0;
    }
}




__global__ void compute_scores_simple_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ weights,
    const double* __restrict__ weight_sums,
    const int32_t* __restrict__ pair_first,
    const int32_t* __restrict__ pair_second,
    double* __restrict__ scores,
    int32_t num_pairs
) {
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= num_pairs) return;

    int32_t u = pair_first[pair_idx];
    int32_t v = pair_second[pair_idx];

    int32_t u_start = offsets[u];
    int32_t u_end = offsets[u + 1];
    int32_t v_start = offsets[v];
    int32_t v_end = offsets[v + 1];
    int32_t deg_u = u_end - u_start;
    int32_t deg_v = v_end - v_start;

    double intersection_sum = 0.0;

    if (deg_u > 0 && deg_v > 0) {
        int i = u_start, j = v_start;
        while (i < u_end && j < v_end) {
            int32_t nu = indices[i];
            int32_t nv = indices[j];
            if (nu < nv) i++;
            else if (nu > nv) j++;
            else {
                double wu = weights[i];
                double wv = weights[j];
                intersection_sum += (wu < wv) ? wu : wv;
                i++; j++;
            }
        }
    }

    double denom = weight_sums[u] + weight_sums[v];
    scores[pair_idx] = (denom > 0.0) ? (2.0 * intersection_sum / denom) : 0.0;
}




__global__ void negate_scores_kernel(const double* __restrict__ in, double* __restrict__ out, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = -in[idx];
}

__global__ void generate_sequence_kernel(int32_t* __restrict__ arr, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = idx;
}

__global__ void gather_results_kernel(
    const int32_t* __restrict__ sorted_indices,
    const int32_t* __restrict__ first_in,
    const int32_t* __restrict__ second_in,
    const double* __restrict__ scores_in,
    int32_t* __restrict__ first_out,
    int32_t* __restrict__ second_out,
    double* __restrict__ scores_out,
    int32_t count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int32_t src = sorted_indices[idx];
        first_out[idx] = first_in[src];
        second_out[idx] = second_in[src];
        scores_out[idx] = scores_in[src];
    }
}




__global__ void generate_all_vertices_kernel(int32_t* __restrict__ arr, int32_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = idx;
}




__global__ void compute_max_seed_degree_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ seeds,
    int32_t num_seeds,
    int32_t* __restrict__ max_deg
) {
    __shared__ int32_t s_max;
    if (threadIdx.x == 0) s_max = 0;
    __syncthreads();

    for (int i = threadIdx.x; i < num_seeds; i += blockDim.x) {
        int32_t u = seeds[i];
        int32_t deg = offsets[u + 1] - offsets[u];
        atomicMax(&s_max, deg);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicMax(max_deg, s_max);
    }
}




__global__ void build_group_offsets_kernel(
    const int32_t* __restrict__ unique_group_id,
    int32_t num_unique,
    int32_t num_seeds,
    int32_t* __restrict__ group_offsets
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx <= num_seeds) {
        group_offsets[idx] = num_unique;
    }
}

__global__ void build_group_offsets_kernel2(
    const int32_t* __restrict__ unique_group_id,
    int32_t num_unique,
    int32_t num_seeds,
    int32_t* __restrict__ group_offsets
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_unique) {
        if (idx == 0 || unique_group_id[idx] != unique_group_id[idx - 1]) {
            group_offsets[unique_group_id[idx]] = idx;
        }
    }
}

__global__ void fix_group_offsets_scan_kernel(
    int32_t* __restrict__ group_offsets,
    int32_t num_seeds
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int32_t next_start = group_offsets[num_seeds];
        for (int i = num_seeds - 1; i >= 0; i--) {
            if (group_offsets[i] > next_start) {
                group_offsets[i] = next_start;
            } else {
                next_start = group_offsets[i];
            }
        }
    }
}





void launch_compute_weight_sums(
    const int32_t* offsets, const double* weights,
    double* weight_sums, int32_t num_vertices, cudaStream_t stream
) {
    int block = 256;
    int grid = (num_vertices + block - 1) / block;
    compute_weight_sums_kernel<<<grid, block, 0, stream>>>(offsets, weights, weight_sums, num_vertices);
}

void launch_count_raw_pairs(
    const int32_t* offsets, const int32_t* indices,
    const int32_t* seeds, int32_t num_seeds,
    int64_t* counts, cudaStream_t stream
) {
    int block = 256;
    int grid = (num_seeds + block - 1) / block;
    count_raw_pairs_kernel<<<grid, block, 0, stream>>>(offsets, indices, seeds, num_seeds, counts);
}

void launch_expand_raw_pairs(
    const int32_t* offsets, const int32_t* indices,
    const int32_t* seeds, int32_t num_seeds,
    const int64_t* pair_offsets, int32_t* out_v, cudaStream_t stream
) {
    int block = 256;
    expand_raw_pairs_kernel<<<num_seeds, block, 0, stream>>>(
        offsets, indices, seeds, num_seeds, pair_offsets, out_v);
}

size_t cub_scan_temp_bytes(int32_t num_items) {
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, (int64_t*)nullptr, (int64_t*)nullptr, num_items);
    return temp_bytes;
}

void cub_scan_exclusive_int64(void* temp, size_t temp_bytes,
    const int64_t* in, int64_t* out, int32_t num_items, cudaStream_t stream
) {
    cub::DeviceScan::ExclusiveSum(temp, temp_bytes, in, out, num_items, stream);
}

size_t cub_scan_temp_bytes_int32(int64_t num_items) {
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, (int32_t*)nullptr, (int32_t*)nullptr, (int)num_items);
    return temp_bytes;
}

void cub_scan_exclusive_int32(void* temp, size_t temp_bytes,
    const int32_t* in, int32_t* out, int64_t num_items, cudaStream_t stream
) {
    cub::DeviceScan::ExclusiveSum(temp, temp_bytes, in, out, (int)num_items, stream);
}

size_t cub_seg_sort_temp_bytes(int64_t num_items, int32_t num_segments) {
    size_t temp_bytes = 0;
    cub::DeviceSegmentedSort::SortKeys(
        nullptr, temp_bytes,
        (int32_t*)nullptr, (int32_t*)nullptr,
        (int)num_items, (int)num_segments,
        (int64_t*)nullptr, (int64_t*)nullptr + 1);
    return temp_bytes;
}

void cub_seg_sort_int32(void* temp, size_t temp_bytes,
    const int32_t* keys_in, int32_t* keys_out,
    int64_t num_items, int32_t num_segments,
    const int64_t* seg_begin_offsets, const int64_t* seg_end_offsets,
    cudaStream_t stream
) {
    cub::DeviceSegmentedSort::SortKeys(
        temp, temp_bytes,
        keys_in, keys_out,
        (int)num_items, (int)num_segments,
        seg_begin_offsets, seg_end_offsets, stream);
}

void launch_mark_unique(
    const int32_t* sorted_v, const int64_t* seg_offsets,
    int32_t num_seeds, int64_t total_raw, int32_t* is_first,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (int)((total_raw + block - 1) / block);
    mark_unique_kernel<<<grid, block, 0, stream>>>(sorted_v, seg_offsets, num_seeds, total_raw, is_first);
}

void launch_compact_unique(
    const int32_t* sorted_v, const int32_t* prefix_sum, const int32_t* is_first,
    int64_t total_raw, const int32_t* seeds, const int64_t* seg_offsets,
    int32_t num_seeds,
    int32_t* unique_first, int32_t* unique_second, int32_t* unique_group_id,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (int)((total_raw + block - 1) / block);
    compact_unique_kernel<<<grid, block, 0, stream>>>(
        sorted_v, prefix_sum, is_first, total_raw,
        seeds, seg_offsets, num_seeds,
        unique_first, unique_second, unique_group_id);
}

void launch_compute_scores_simple(
    const int32_t* offsets, const int32_t* indices,
    const double* weights, const double* weight_sums,
    const int32_t* pair_first, const int32_t* pair_second,
    double* scores, int32_t num_pairs, cudaStream_t stream
) {
    int block = 256;
    int grid = (num_pairs + block - 1) / block;
    compute_scores_simple_kernel<<<grid, block, 0, stream>>>(
        offsets, indices, weights, weight_sums,
        pair_first, pair_second, scores, num_pairs);
}

size_t cub_sort_pairs_temp_bytes(int32_t num_items) {
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes,
        (double*)nullptr, (double*)nullptr,
        (int32_t*)nullptr, (int32_t*)nullptr,
        num_items);
    return temp_bytes;
}

void cub_sort_pairs_double_int32(void* temp, size_t temp_bytes,
    const double* keys_in, double* keys_out,
    const int32_t* vals_in, int32_t* vals_out,
    int32_t num_items, cudaStream_t stream
) {
    cub::DeviceRadixSort::SortPairs(
        temp, temp_bytes,
        keys_in, keys_out,
        vals_in, vals_out,
        num_items, 0, 64, stream);
}

void launch_build_group_offsets(
    const int32_t* unique_group_id, int32_t num_unique,
    int32_t num_seeds, int32_t* group_offsets, cudaStream_t stream
) {
    int block = 256;
    int grid1 = (num_seeds + 1 + block - 1) / block;
    build_group_offsets_kernel<<<grid1, block, 0, stream>>>(
        unique_group_id, num_unique, num_seeds, group_offsets);

    int grid2 = (num_unique + block - 1) / block;
    build_group_offsets_kernel2<<<grid2, block, 0, stream>>>(
        unique_group_id, num_unique, num_seeds, group_offsets);

    fix_group_offsets_scan_kernel<<<1, 1, 0, stream>>>(group_offsets, num_seeds);
}

}  

similarity_result_double_t sorensen_all_pairs_similarity(const graph32_t& graph,
                                                         const double* edge_weights,
                                                         const int32_t* vertices,
                                                         std::size_t num_vertices,
                                                         std::optional<std::size_t> topk) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);
    (void)cache;

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    int32_t n_verts = graph.number_of_vertices;
    const double* d_weights = edge_weights;
    cudaStream_t stream = 0;

    
    int32_t num_seeds;
    int32_t* d_seeds_alloc = nullptr;
    const int32_t* d_seeds;

    if (vertices == nullptr) {
        num_seeds = n_verts;
        cudaMalloc(&d_seeds_alloc, (size_t)n_verts * sizeof(int32_t));
        int block = 256;
        int grid = (n_verts + block - 1) / block;
        generate_all_vertices_kernel<<<grid, block, 0, stream>>>(d_seeds_alloc, n_verts);
        d_seeds = d_seeds_alloc;
    } else {
        num_seeds = (int32_t)num_vertices;
        d_seeds = vertices;
    }

    if (num_seeds == 0) {
        if (d_seeds_alloc) cudaFree(d_seeds_alloc);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    double* d_weight_sums = nullptr;
    cudaMalloc(&d_weight_sums, (size_t)n_verts * sizeof(double));
    launch_compute_weight_sums(d_offsets, d_weights, d_weight_sums, n_verts, stream);

    
    int64_t* d_counts = nullptr;
    cudaMalloc(&d_counts, (size_t)num_seeds * sizeof(int64_t));
    launch_count_raw_pairs(d_offsets, d_indices, d_seeds, num_seeds, d_counts, stream);

    
    int64_t* d_pair_offsets = nullptr;
    cudaMalloc(&d_pair_offsets, (size_t)(num_seeds + 1) * sizeof(int64_t));

    size_t scan_temp_bytes = cub_scan_temp_bytes(num_seeds);
    void* d_scan_temp = nullptr;
    cudaMalloc(&d_scan_temp, scan_temp_bytes);
    cub_scan_exclusive_int64(d_scan_temp, scan_temp_bytes,
        d_counts, d_pair_offsets, num_seeds, stream);
    cudaFree(d_scan_temp);

    
    int64_t h_last_offset, h_last_count;
    cudaMemcpyAsync(&h_last_offset, d_pair_offsets + num_seeds - 1, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_last_count, d_counts + num_seeds - 1, sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int64_t total_raw = h_last_offset + h_last_count;

    
    cudaFree(d_counts);

    
    cudaMemcpyAsync(d_pair_offsets + num_seeds, &total_raw, sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    if (total_raw == 0) {
        cudaFree(d_pair_offsets);
        cudaFree(d_weight_sums);
        if (d_seeds_alloc) cudaFree(d_seeds_alloc);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t* d_raw_v = nullptr;
    cudaMalloc(&d_raw_v, (size_t)total_raw * sizeof(int32_t));
    launch_expand_raw_pairs(d_offsets, d_indices, d_seeds, num_seeds,
        d_pair_offsets, d_raw_v, stream);

    
    int32_t* d_sorted_v = nullptr;
    cudaMalloc(&d_sorted_v, (size_t)total_raw * sizeof(int32_t));

    size_t seg_sort_temp_bytes = cub_seg_sort_temp_bytes(total_raw, num_seeds);
    void* d_seg_sort_temp = nullptr;
    cudaMalloc(&d_seg_sort_temp, seg_sort_temp_bytes);
    cub_seg_sort_int32(d_seg_sort_temp, seg_sort_temp_bytes,
        d_raw_v, d_sorted_v, total_raw, num_seeds,
        d_pair_offsets, d_pair_offsets + 1, stream);

    
    cudaFree(d_raw_v);
    cudaFree(d_seg_sort_temp);

    
    int32_t* d_is_first = nullptr;
    cudaMalloc(&d_is_first, (size_t)total_raw * sizeof(int32_t));
    launch_mark_unique(d_sorted_v, d_pair_offsets, num_seeds, total_raw, d_is_first, stream);

    
    int32_t* d_prefix = nullptr;
    cudaMalloc(&d_prefix, (size_t)total_raw * sizeof(int32_t));

    size_t scan32_temp_bytes = cub_scan_temp_bytes_int32(total_raw);
    void* d_scan32_temp = nullptr;
    cudaMalloc(&d_scan32_temp, scan32_temp_bytes);
    cub_scan_exclusive_int32(d_scan32_temp, scan32_temp_bytes,
        d_is_first, d_prefix, total_raw, stream);
    cudaFree(d_scan32_temp);

    
    int32_t h_last_prefix, h_last_is_first;
    cudaMemcpyAsync(&h_last_prefix, d_prefix + total_raw - 1, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&h_last_is_first, d_is_first + total_raw - 1, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int32_t num_unique = h_last_prefix + h_last_is_first;

    if (num_unique == 0) {
        cudaFree(d_sorted_v);
        cudaFree(d_is_first);
        cudaFree(d_prefix);
        cudaFree(d_pair_offsets);
        cudaFree(d_weight_sums);
        if (d_seeds_alloc) cudaFree(d_seeds_alloc);
        return {nullptr, nullptr, nullptr, 0};
    }

    
    int32_t* d_unique_first = nullptr;
    int32_t* d_unique_second = nullptr;
    int32_t* d_unique_group = nullptr;
    cudaMalloc(&d_unique_first, (size_t)num_unique * sizeof(int32_t));
    cudaMalloc(&d_unique_second, (size_t)num_unique * sizeof(int32_t));
    cudaMalloc(&d_unique_group, (size_t)num_unique * sizeof(int32_t));

    launch_compact_unique(d_sorted_v, d_prefix, d_is_first, total_raw,
        d_seeds, d_pair_offsets, num_seeds,
        d_unique_first, d_unique_second, d_unique_group, stream);

    
    cudaFree(d_sorted_v);
    cudaFree(d_is_first);
    cudaFree(d_prefix);
    cudaFree(d_pair_offsets);

    
    int32_t* d_group_offsets = nullptr;
    cudaMalloc(&d_group_offsets, (size_t)(num_seeds + 1) * sizeof(int32_t));
    launch_build_group_offsets(d_unique_group, num_unique,
        num_seeds, d_group_offsets, stream);
    cudaFree(d_unique_group);

    
    int32_t* d_max_deg = nullptr;
    cudaMalloc(&d_max_deg, sizeof(int32_t));
    cudaMemsetAsync(d_max_deg, 0, sizeof(int32_t), stream);
    {
        int block = 256;
        compute_max_seed_degree_kernel<<<1, block, 0, stream>>>(d_offsets, d_seeds, num_seeds, d_max_deg);
    }

    int32_t h_max_deg;
    cudaMemcpyAsync(&h_max_deg, d_max_deg, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_max_deg);
    if (d_seeds_alloc) { cudaFree(d_seeds_alloc); d_seeds_alloc = nullptr; }

    
    double* d_scores = nullptr;
    cudaMalloc(&d_scores, (size_t)num_unique * sizeof(double));

    launch_compute_scores_simple(
        d_offsets, d_indices, d_weights, d_weight_sums,
        d_unique_first, d_unique_second,
        d_scores, num_unique, stream);

    cudaFree(d_group_offsets);
    cudaFree(d_weight_sums);

    
    if (topk.has_value() && (std::size_t)num_unique > topk.value()) {
        int32_t topk_count = (int32_t)topk.value();

        
        double* d_neg_scores = nullptr;
        cudaMalloc(&d_neg_scores, (size_t)num_unique * sizeof(double));
        {
            int block = 256;
            int grid = (num_unique + block - 1) / block;
            negate_scores_kernel<<<grid, block, 0, stream>>>(d_scores, d_neg_scores, num_unique);
        }

        int32_t* d_idx = nullptr;
        cudaMalloc(&d_idx, (size_t)num_unique * sizeof(int32_t));
        {
            int block = 256;
            int grid = (num_unique + block - 1) / block;
            generate_sequence_kernel<<<grid, block, 0, stream>>>(d_idx, num_unique);
        }

        double* d_sorted_neg_scores = nullptr;
        int32_t* d_sorted_idx = nullptr;
        cudaMalloc(&d_sorted_neg_scores, (size_t)num_unique * sizeof(double));
        cudaMalloc(&d_sorted_idx, (size_t)num_unique * sizeof(int32_t));

        size_t sort_temp_bytes = cub_sort_pairs_temp_bytes(num_unique);
        void* d_sort_temp = nullptr;
        cudaMalloc(&d_sort_temp, sort_temp_bytes);

        cub_sort_pairs_double_int32(
            d_sort_temp, sort_temp_bytes,
            d_neg_scores, d_sorted_neg_scores,
            d_idx, d_sorted_idx,
            num_unique, stream);

        cudaFree(d_sort_temp);
        cudaFree(d_neg_scores);
        cudaFree(d_idx);
        cudaFree(d_sorted_neg_scores);

        
        int32_t* d_out_first = nullptr;
        int32_t* d_out_second = nullptr;
        double* d_out_scores = nullptr;
        cudaMalloc(&d_out_first, (size_t)topk_count * sizeof(int32_t));
        cudaMalloc(&d_out_second, (size_t)topk_count * sizeof(int32_t));
        cudaMalloc(&d_out_scores, (size_t)topk_count * sizeof(double));

        {
            int block = 256;
            int grid = (topk_count + block - 1) / block;
            gather_results_kernel<<<grid, block, 0, stream>>>(
                d_sorted_idx,
                d_unique_first, d_unique_second, d_scores,
                d_out_first, d_out_second, d_out_scores,
                topk_count);
        }

        cudaFree(d_sorted_idx);
        cudaFree(d_unique_first);
        cudaFree(d_unique_second);
        cudaFree(d_scores);

        return {d_out_first, d_out_second, d_out_scores, (std::size_t)topk_count};
    }

    return {d_unique_first, d_unique_second, d_scores, (std::size_t)num_unique};
}

}  
