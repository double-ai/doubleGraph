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
#include <cooperative_groups.h>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace cg = cooperative_groups;

namespace aai {

namespace {

static constexpr int32_t MAX_BFS_LEVELS = 60;
static constexpr int MAX_FWBW_ROUNDS = 6;
static constexpr int MAX_TRIM_ITERS = 12;
static constexpr int MAX_CC_ITERS = 50;

struct Frame { int32_t v, ei; };

static int compute_bits_needed(int32_t nv) {
    if (nv <= 1) return 1;
    int bits = 0;
    int32_t v = nv - 1;
    while (v > 0) { bits++; v >>= 1; }
    return bits;
}





__global__ void init_labels_k(int32_t* __restrict__ labels, int32_t nv) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < nv) labels[v] = v;
}

__global__ void label_propagate_k(const int32_t* __restrict__ offsets,
                                   const int32_t* __restrict__ indices,
                                   int32_t* __restrict__ labels,
                                   int32_t nv,
                                   int32_t* __restrict__ changed) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv) return;

    int32_t my_label = labels[v];
    int32_t min_label = my_label;

    int start = offsets[v];
    int end = offsets[v + 1];

    for (int e = start; e < end; e++) {
        int w = indices[e];
        int32_t w_label = labels[w];
        if (w_label < min_label) min_label = w_label;
    }

    if (min_label < my_label) {
        int32_t jumped = labels[min_label];
        if (jumped < min_label) min_label = jumped;

        atomicMin(&labels[v], min_label);
        atomicMin(&labels[my_label], min_label);
        *changed = 1;
    }
}

__global__ void compress_labels_k(int32_t* __restrict__ labels, int32_t nv,
                                   int32_t* __restrict__ changed) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv) return;

    int32_t l = labels[v];
    int32_t ll = labels[l];
    if (ll != l) {
        labels[v] = ll;
        *changed = 1;
    }
}





__global__ void expand_edges_k(const int32_t* __restrict__ offsets,
                                const int32_t* __restrict__ indices,
                                int32_t nv,
                                int32_t* __restrict__ dst_arr,
                                int32_t* __restrict__ src_arr) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= nv) return;
    int start = offsets[u];
    int end = offsets[u + 1];
    for (int e = start; e < end; e++) {
        dst_arr[e] = indices[e];
        src_arr[e] = u;
    }
}

__global__ void build_offsets_from_sorted_k(const int32_t* __restrict__ sorted_dst,
                                             int32_t ne, int32_t nv,
                                             int32_t* __restrict__ rev_off) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid > ne) return;

    int curr = (tid < ne) ? sorted_dst[tid] : nv;
    int prev = (tid > 0) ? sorted_dst[tid - 1] : -1;

    if (curr != prev) {
        for (int v = prev + 1; v <= curr; v++) {
            rev_off[v] = tid;
        }
    }
    if (tid == ne) {
        for (int v = curr; v <= nv; v++) {
            rev_off[v] = ne;
        }
    }
}

__global__ void count_indegrees_k(const int32_t* __restrict__ indices, int32_t ne,
                                   int32_t* __restrict__ indeg) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < ne) atomicAdd(&indeg[indices[tid]], 1);
}

__global__ void scatter_transpose_k(const int32_t* __restrict__ offsets,
                                     const int32_t* __restrict__ indices,
                                     int32_t nv,
                                     int32_t* __restrict__ rev_off_tmp,
                                     int32_t* __restrict__ rev_indices) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= nv) return;
    for (int e = offsets[u]; e < offsets[u + 1]; e++) {
        int v = indices[e];
        int pos = atomicAdd(&rev_off_tmp[v], 1);
        rev_indices[pos] = u;
    }
}





__global__ void bfs_dual_highdeg_k(
    const int32_t* __restrict__ fwd_off,
    const int32_t* __restrict__ fwd_idx,
    const int32_t* __restrict__ rev_off,
    const int32_t* __restrict__ rev_idx,
    int32_t seg_begin, int32_t seg_end,
    int32_t* __restrict__ fw_dist,
    int32_t* __restrict__ bw_dist,
    int32_t level,
    int32_t* __restrict__ changed) {

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int v = seg_begin + warp_id;
    if (v >= seg_end) return;

    if (fw_dist[v] == level) {
        int start = fwd_off[v];
        int end = fwd_off[v + 1];
        for (int e = start + lane; e < end; e += 32) {
            int w = fwd_idx[e];
            if (fw_dist[w] == -1) {
                fw_dist[w] = level + 1;
                *changed = 1;
            }
        }
    }

    if (bw_dist[v] == level) {
        int start = rev_off[v];
        int end = rev_off[v + 1];
        for (int e = start + lane; e < end; e += 32) {
            int w = rev_idx[e];
            if (bw_dist[w] == -1) {
                bw_dist[w] = level + 1;
                *changed = 1;
            }
        }
    }
}

__global__ void bfs_dual_lowdeg_k(
    const int32_t* __restrict__ fwd_off,
    const int32_t* __restrict__ fwd_idx,
    const int32_t* __restrict__ rev_off,
    const int32_t* __restrict__ rev_idx,
    int32_t seg_begin, int32_t seg_end,
    int32_t* __restrict__ fw_dist,
    int32_t* __restrict__ bw_dist,
    int32_t level,
    int32_t* __restrict__ changed) {

    int v = seg_begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= seg_end) return;

    if (fw_dist[v] == level) {
        for (int e = fwd_off[v]; e < fwd_off[v + 1]; e++) {
            int w = fwd_idx[e];
            if (fw_dist[w] == -1) {
                fw_dist[w] = level + 1;
                *changed = 1;
            }
        }
    }

    if (bw_dist[v] == level) {
        for (int e = rev_off[v]; e < rev_off[v + 1]; e++) {
            int w = rev_idx[e];
            if (bw_dist[w] == -1) {
                bw_dist[w] = level + 1;
                *changed = 1;
            }
        }
    }
}

__global__ void bfs_dual_highdeg_restricted_k(
    const int32_t* __restrict__ fwd_off,
    const int32_t* __restrict__ fwd_idx,
    const int32_t* __restrict__ rev_off,
    const int32_t* __restrict__ rev_idx,
    int32_t seg_begin, int32_t seg_end,
    int32_t* __restrict__ fw_dist,
    int32_t* __restrict__ bw_dist,
    const int32_t* __restrict__ components,
    int32_t level,
    int32_t* __restrict__ changed) {

    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x & 31;
    int v = seg_begin + warp_id;
    if (v >= seg_end) return;

    if (fw_dist[v] == level) {
        int start = fwd_off[v];
        int end = fwd_off[v + 1];
        for (int e = start + lane; e < end; e += 32) {
            int w = fwd_idx[e];
            if (fw_dist[w] == -1 && components[w] < 0) {
                fw_dist[w] = level + 1;
                *changed = 1;
            }
        }
    }

    if (bw_dist[v] == level) {
        int start = rev_off[v];
        int end = rev_off[v + 1];
        for (int e = start + lane; e < end; e += 32) {
            int w = rev_idx[e];
            if (bw_dist[w] == -1 && components[w] < 0) {
                bw_dist[w] = level + 1;
                *changed = 1;
            }
        }
    }
}

__global__ void bfs_dual_lowdeg_restricted_k(
    const int32_t* __restrict__ fwd_off,
    const int32_t* __restrict__ fwd_idx,
    const int32_t* __restrict__ rev_off,
    const int32_t* __restrict__ rev_idx,
    int32_t seg_begin, int32_t seg_end,
    int32_t* __restrict__ fw_dist,
    int32_t* __restrict__ bw_dist,
    const int32_t* __restrict__ components,
    int32_t level,
    int32_t* __restrict__ changed) {

    int v = seg_begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= seg_end) return;

    if (fw_dist[v] == level) {
        for (int e = fwd_off[v]; e < fwd_off[v + 1]; e++) {
            int w = fwd_idx[e];
            if (fw_dist[w] == -1 && components[w] < 0) {
                fw_dist[w] = level + 1;
                *changed = 1;
            }
        }
    }

    if (bw_dist[v] == level) {
        for (int e = rev_off[v]; e < rev_off[v + 1]; e++) {
            int w = rev_idx[e];
            if (bw_dist[w] == -1 && components[w] < 0) {
                bw_dist[w] = level + 1;
                *changed = 1;
            }
        }
    }
}





__global__ void bfs_converge_segmented_coop_k(
    const int32_t* __restrict__ fwd_off,
    const int32_t* __restrict__ fwd_idx,
    const int32_t* __restrict__ rev_off,
    const int32_t* __restrict__ rev_idx,
    int32_t nv,
    int32_t* __restrict__ fw_dist,
    int32_t* __restrict__ bw_dist,
    int32_t pivot,
    int32_t max_levels,
    int32_t* __restrict__ flags,
    int32_t seg_mid) {

    cg::grid_group grid = cg::this_grid();

    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int nthreads = (int)(gridDim.x * blockDim.x);

    for (int v = tid; v < nv; v += nthreads) {
        fw_dist[v] = -1;
        bw_dist[v] = -1;
    }
    if (tid == 0) {
        fw_dist[pivot] = 0;
        bw_dist[pivot] = 0;
        flags[0] = 0;
        flags[1] = 1;
    }
    grid.sync();

    int level = 0;
    while (level < max_levels) {
        if (tid == 0) flags[0] = 0;
        grid.sync();

        int local_changed = 0;

        int lane = tid & 31;
        int warp_id = tid >> 5;
        int nwarps = nthreads >> 5;

        for (int v = warp_id; v < nv; v += nwarps) {
            if (fw_dist[v] == level) {
                int start = fwd_off[v];
                int end = fwd_off[v + 1];
                for (int e = start + lane; e < end; e += 32) {
                    int w = fwd_idx[e];
                    if (fw_dist[w] == -1) {
                        fw_dist[w] = level + 1;
                        local_changed = 1;
                    }
                }
            }
            if (bw_dist[v] == level) {
                int start = rev_off[v];
                int end = rev_off[v + 1];
                for (int e = start + lane; e < end; e += 32) {
                    int w = rev_idx[e];
                    if (bw_dist[w] == -1) {
                        bw_dist[w] = level + 1;
                        local_changed = 1;
                    }
                }
            }
        }

        int warp_any = __any_sync(0xffffffff, local_changed);
        if (warp_any && lane == 0) atomicExch(flags, 1);

        grid.sync();
        if (tid == 0) flags[1] = flags[0];
        grid.sync();

        if (flags[1] == 0) break;
        level++;
    }
}

__global__ void bfs_converge_segmented_restricted_coop_k(
    const int32_t* __restrict__ fwd_off,
    const int32_t* __restrict__ fwd_idx,
    const int32_t* __restrict__ rev_off,
    const int32_t* __restrict__ rev_idx,
    int32_t nv,
    int32_t* __restrict__ fw_dist,
    int32_t* __restrict__ bw_dist,
    const int32_t* __restrict__ components,
    int32_t pivot,
    int32_t max_levels,
    int32_t* __restrict__ flags,
    int32_t seg_mid) {

    cg::grid_group grid = cg::this_grid();

    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int nthreads = (int)(gridDim.x * blockDim.x);

    for (int v = tid; v < nv; v += nthreads) {
        fw_dist[v] = -1;
        bw_dist[v] = -1;
    }
    if (tid == 0) {
        fw_dist[pivot] = 0;
        bw_dist[pivot] = 0;
        flags[0] = 0;
        flags[1] = 1;
    }
    grid.sync();

    int level = 0;
    while (level < max_levels) {
        if (tid == 0) flags[0] = 0;
        grid.sync();

        int local_changed = 0;

        int lane = tid & 31;
        int warp_id = tid >> 5;
        int nwarps = nthreads >> 5;

        for (int v = warp_id; v < nv; v += nwarps) {
            if (fw_dist[v] == level) {
                int start = fwd_off[v];
                int end = fwd_off[v + 1];
                for (int e = start + lane; e < end; e += 32) {
                    int w = fwd_idx[e];
                    if (fw_dist[w] == -1 && components[w] < 0) {
                        fw_dist[w] = level + 1;
                        local_changed = 1;
                    }
                }
            }
            if (bw_dist[v] == level) {
                int start = rev_off[v];
                int end = rev_off[v + 1];
                for (int e = start + lane; e < end; e += 32) {
                    int w = rev_idx[e];
                    if (bw_dist[w] == -1 && components[w] < 0) {
                        bw_dist[w] = level + 1;
                        local_changed = 1;
                    }
                }
            }
        }

        int warp_any = __any_sync(0xffffffff, local_changed);
        if (warp_any && lane == 0) atomicExch(flags, 1);

        grid.sync();
        if (tid == 0) flags[1] = flags[0];
        grid.sync();

        if (flags[1] == 0) break;
        level++;
    }
}





__global__ void mark_scc_k(int32_t nv, const int32_t* __restrict__ fw_dist,
                            const int32_t* __restrict__ bw_dist,
                            int32_t* __restrict__ components, int32_t pivot) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv) return;
    if (fw_dist[v] >= 0 && bw_dist[v] >= 0) {
        components[v] = pivot;
    }
}

__global__ void mark_scc_restricted_k(int32_t nv, const int32_t* __restrict__ fw_dist,
                                       const int32_t* __restrict__ bw_dist,
                                       int32_t* __restrict__ components, int32_t pivot) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv || components[v] >= 0) return;
    if (fw_dist[v] >= 0 && bw_dist[v] >= 0) {
        components[v] = pivot;
    }
}

__global__ void trim_and_count_k(const int32_t* __restrict__ offsets,
                                  const int32_t* __restrict__ indices,
                                  const int32_t* __restrict__ rev_offsets,
                                  const int32_t* __restrict__ rev_indices,
                                  int32_t nv,
                                  int32_t* __restrict__ components,
                                  int32_t* __restrict__ changed,
                                  int32_t* ) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv || components[v] >= 0) return;

    bool has_out = false;
    for (int e = offsets[v]; e < offsets[v + 1]; e++) {
        int w = indices[e];
        if (w != v && components[w] < 0) { has_out = true; break; }
    }
    if (!has_out) {
        components[v] = v;
        if (changed) atomicExch(changed, 1);
        return;
    }

    bool has_in = false;
    for (int e = rev_offsets[v]; e < rev_offsets[v + 1]; e++) {
        int w = rev_indices[e];
        if (w != v && components[w] < 0) { has_in = true; break; }
    }
    if (!has_in) {
        components[v] = v;
        if (changed) atomicExch(changed, 1);
    }
}

__global__ void count_unassigned_k(const int32_t* __restrict__ components, int32_t nv,
                                    int32_t* __restrict__ count) {
    __shared__ int32_t s_count;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < nv && components[v] < 0) atomicAdd(&s_count, 1);
    __syncthreads();

    if (threadIdx.x == 0 && s_count > 0) atomicAdd(count, s_count);
}

__global__ void find_unassigned_pivot_k(const int32_t* __restrict__ components,
                                         const int32_t* __restrict__ offsets,
                                         int32_t nv, int32_t* __restrict__ pivot) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv || components[v] >= 0) return;
    int deg = offsets[v + 1] - offsets[v];
    if (deg > 0) atomicMin(pivot, v);
}





static int do_cc_label_prop(const int32_t* offsets, const int32_t* indices,
                            int32_t nv, int32_t* labels,
                            int32_t* d_flag, int32_t* h_flag,
                            int max_iters, cudaStream_t stream) {
    int B = 256;
    int G = (nv + B - 1) / B;
    if (G == 0) G = 1;

    init_labels_k<<<G, B, 0, stream>>>(labels, nv);

    for (int i = 0; i < max_iters; i++) {
        cudaMemsetAsync(d_flag, 0, sizeof(int32_t), stream);
        label_propagate_k<<<G, B, 0, stream>>>(offsets, indices, labels, nv, d_flag);

        if ((i + 1) % 3 == 0 || i == max_iters - 1) {
            cudaMemcpyAsync(h_flag, d_flag, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (*h_flag == 0) break;
        }
    }

    for (int i = 0; i < 20; i++) {
        cudaMemsetAsync(d_flag, 0, sizeof(int32_t), stream);
        compress_labels_k<<<G, B, 0, stream>>>(labels, nv, d_flag);
        if ((i + 1) % 3 == 0) {
            cudaMemcpyAsync(h_flag, d_flag, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            if (*h_flag == 0) break;
        }
    }

    return 0;
}

static void do_build_transpose_sort(const int32_t* offsets, const int32_t* indices,
                                    int32_t nv, int32_t ne,
                                    int32_t* d_dst_keys, int32_t* d_src_vals,
                                    int32_t* d_dst_keys_out, int32_t* d_src_vals_out,
                                    int32_t* rev_off,
                                    void* cub_tmp, size_t cub_tmp_bytes,
                                    int bits_needed,
                                    cudaStream_t stream) {
    int B = 256;
    if (nv > 0)
        expand_edges_k<<<(nv + B - 1) / B, B, 0, stream>>>(offsets, indices, nv, d_dst_keys, d_src_vals);
    cub::DeviceRadixSort::SortPairs(cub_tmp, cub_tmp_bytes,
                                     d_dst_keys, d_dst_keys_out,
                                     d_src_vals, d_src_vals_out,
                                     ne, 0, bits_needed, stream);
    cudaMemsetAsync(rev_off, 0, (nv + 1) * sizeof(int32_t), stream);
    if (ne > 0)
        build_offsets_from_sorted_k<<<(ne + 2 + B - 1) / B, B, 0, stream>>>(d_dst_keys_out, ne, nv, rev_off);
}

static void do_build_transpose_atomic(const int32_t* offsets, const int32_t* indices,
                                      int32_t nv, int32_t ne,
                                      int32_t* indeg, int32_t* rev_off,
                                      int32_t* rev_off_tmp, int32_t* rev_indices,
                                      void* cub_tmp, size_t cub_tmp_bytes,
                                      cudaStream_t stream) {
    int B = 256;
    cudaMemsetAsync(indeg, 0, (nv + 1) * sizeof(int32_t), stream);
    if (ne > 0)
        count_indegrees_k<<<(ne + B - 1) / B, B, 0, stream>>>(indices, ne, indeg);
    cub::DeviceScan::ExclusiveSum(cub_tmp, cub_tmp_bytes, indeg, rev_off, nv + 1, stream);
    cudaMemcpyAsync(rev_off_tmp, rev_off, (nv + 1) * sizeof(int32_t),
                     cudaMemcpyDeviceToDevice, stream);
    if (nv > 0)
        scatter_transpose_k<<<(nv + B - 1) / B, B, 0, stream>>>(
            offsets, indices, nv, rev_off_tmp, rev_indices);
}

static int do_bfs_converge_segmented(const int32_t* fwd_off, const int32_t* fwd_idx,
                                     const int32_t* rev_off, const int32_t* rev_idx,
                                     int32_t nv, int32_t* fw_dist, int32_t* bw_dist,
                                     int32_t pivot, int32_t max_levels,
                                     int32_t* d_flags,
                                     int32_t seg_mid,
                                     cudaStream_t stream) {
    int B = 256;

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM,
                                                  bfs_converge_segmented_coop_k,
                                                  B, 0);
    int grid = prop.multiProcessorCount * maxBlocksPerSM;
    if (grid < 1) grid = 1;

    void* args[] = {
        (void*)&fwd_off, (void*)&fwd_idx, (void*)&rev_off, (void*)&rev_idx,
        (void*)&nv, (void*)&fw_dist, (void*)&bw_dist,
        (void*)&pivot, (void*)&max_levels,
        (void*)&d_flags, (void*)&seg_mid};

    cudaLaunchCooperativeKernel((void*)bfs_converge_segmented_coop_k,
                                dim3(grid), dim3(B), args, 0, stream);
    return 0;
}

static int do_bfs_converge_segmented_restricted(const int32_t* fwd_off, const int32_t* fwd_idx,
                                                const int32_t* rev_off, const int32_t* rev_idx,
                                                int32_t nv, int32_t* fw_dist, int32_t* bw_dist,
                                                const int32_t* components,
                                                int32_t pivot, int32_t max_levels,
                                                int32_t* d_flags,
                                                int32_t seg_mid,
                                                cudaStream_t stream) {
    int B = 256;

    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    int maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM,
                                                  bfs_converge_segmented_restricted_coop_k,
                                                  B, 0);
    int grid = prop.multiProcessorCount * maxBlocksPerSM;
    if (grid < 1) grid = 1;

    void* args[] = {
        (void*)&fwd_off, (void*)&fwd_idx, (void*)&rev_off, (void*)&rev_idx,
        (void*)&nv, (void*)&fw_dist, (void*)&bw_dist,
        (void*)&components,
        (void*)&pivot, (void*)&max_levels,
        (void*)&d_flags, (void*)&seg_mid};

    cudaLaunchCooperativeKernel((void*)bfs_converge_segmented_restricted_coop_k,
                                dim3(grid), dim3(B), args, 0, stream);
    return 0;
}

static void do_mark_scc(int32_t nv, const int32_t* fw_dist, const int32_t* bw_dist,
                         int32_t* components, int32_t pivot, cudaStream_t stream) {
    int B = 256, G = (nv + B - 1) / B;
    if (G == 0) G = 1;
    mark_scc_k<<<G, B, 0, stream>>>(nv, fw_dist, bw_dist, components, pivot);
}

static void do_mark_scc_restricted(int32_t nv, const int32_t* fw_dist, const int32_t* bw_dist,
                                    int32_t* components, int32_t pivot, cudaStream_t stream) {
    int B = 256, G = (nv + B - 1) / B;
    if (G == 0) G = 1;
    mark_scc_restricted_k<<<G, B, 0, stream>>>(nv, fw_dist, bw_dist, components, pivot);
}

static int32_t do_trim_and_count(const int32_t* offsets, const int32_t* indices,
                                 const int32_t* rev_offsets, const int32_t* rev_indices,
                                 int32_t nv, int32_t* components,
                                 int32_t* d_flags, int32_t* h_flags,
                                 int max_iters, cudaStream_t stream) {
    int B = 256;
    int G = (nv + B - 1) / B;
    if (G == 0) G = 1;

    for (int i = 0; i < max_iters; i++) {
        trim_and_count_k<<<G, B, 0, stream>>>(offsets, indices, rev_offsets, rev_indices,
                                              nv, components, d_flags, d_flags + 1);
    }

    cudaMemsetAsync(d_flags, 0, sizeof(int32_t), stream);
    count_unassigned_k<<<G, B, 0, stream>>>(components, nv, d_flags);
    cudaMemcpyAsync(h_flags, d_flags, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h_flags[0];
}

static int32_t do_find_pivot(const int32_t* components, const int32_t* offsets,
                             int32_t nv, int32_t* d_pivot, int32_t* h_pivot,
                             cudaStream_t stream) {
    int B = 256, G = (nv + B - 1) / B;
    if (G == 0) G = 1;
    int32_t big_val = 0x7FFFFFFF;
    cudaMemcpyAsync(d_pivot, &big_val, sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    find_unassigned_pivot_k<<<G, B, 0, stream>>>(components, offsets, nv, d_pivot);
    cudaMemcpyAsync(h_pivot, d_pivot, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return *h_pivot;
}

static size_t get_cub_sort_temp_bytes(int32_t ne) {
    size_t bytes = 0;
    cub::DeviceRadixSort::SortPairs((void*)nullptr, bytes,
                                     (int32_t*)nullptr, (int32_t*)nullptr,
                                     (int32_t*)nullptr, (int32_t*)nullptr,
                                     ne, 0, 20);
    return bytes;
}

static size_t get_cub_scan_temp_bytes(int32_t nv) {
    size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum((void*)nullptr, bytes,
                                   (int32_t*)nullptr, (int32_t*)nullptr, nv + 1);
    return bytes;
}





struct Cache : Cacheable {
    size_t alloc_v_ = 0;
    size_t alloc_e_ = 0;

    int32_t* d_dst_keys = nullptr;
    int32_t* d_src_vals = nullptr;
    int32_t* d_dst_keys_out = nullptr;
    int32_t* d_src_vals_out = nullptr;

    int32_t* d_indeg = nullptr;
    int32_t* d_rev_off_tmp = nullptr;

    int32_t* d_rev_off = nullptr;
    int32_t* d_rev_idx = nullptr;
    int32_t* d_fw_dist = nullptr;
    int32_t* d_bw_dist = nullptr;
    void* d_cub_tmp = nullptr;
    size_t cub_tmp_bytes = 0;

    int32_t* d_flag = nullptr;

    int32_t* h_offsets = nullptr;
    int32_t* h_indices = nullptr;
    int32_t* h_components = nullptr;
    int32_t* h_flag = nullptr;

    int32_t* disc = nullptr;
    int32_t* low = nullptr;
    int32_t* stk = nullptr;
    int8_t* on_stack = nullptr;
    Frame* call_stack = nullptr;

    cudaStream_t sc = nullptr;
    cudaStream_t scopy = nullptr;

    Cache() {
        cudaMalloc(&d_flag, 4 * sizeof(int32_t));
        cudaMallocHost(&h_flag, 4 * sizeof(int32_t));
        cudaStreamCreate(&sc);
        cudaStreamCreate(&scopy);
    }

    ~Cache() override {
        if (d_dst_keys) cudaFree(d_dst_keys);
        if (d_src_vals) cudaFree(d_src_vals);
        if (d_dst_keys_out) cudaFree(d_dst_keys_out);
        if (d_src_vals_out) cudaFree(d_src_vals_out);
        if (d_indeg) cudaFree(d_indeg);
        if (d_rev_off) cudaFree(d_rev_off);
        if (d_rev_off_tmp) cudaFree(d_rev_off_tmp);
        if (d_rev_idx) cudaFree(d_rev_idx);
        if (d_fw_dist) cudaFree(d_fw_dist);
        if (d_bw_dist) cudaFree(d_bw_dist);
        if (d_cub_tmp) cudaFree(d_cub_tmp);
        if (d_flag) cudaFree(d_flag);
        if (sc) cudaStreamDestroy(sc);
        if (scopy) cudaStreamDestroy(scopy);
        if (h_offsets) cudaFreeHost(h_offsets);
        if (h_indices) cudaFreeHost(h_indices);
        if (h_components) cudaFreeHost(h_components);
        if (h_flag) cudaFreeHost(h_flag);
        delete[] disc;
        delete[] low;
        delete[] stk;
        delete[] on_stack;
        delete[] call_stack;
    }

    void ensure_buffers(size_t nv, size_t ne) {
        if (nv <= alloc_v_ && ne <= alloc_e_) return;

        if (d_dst_keys) { cudaFree(d_dst_keys); d_dst_keys = nullptr; }
        if (d_src_vals) { cudaFree(d_src_vals); d_src_vals = nullptr; }
        if (d_dst_keys_out) { cudaFree(d_dst_keys_out); d_dst_keys_out = nullptr; }
        if (d_src_vals_out) { cudaFree(d_src_vals_out); d_src_vals_out = nullptr; }
        if (d_indeg) { cudaFree(d_indeg); d_indeg = nullptr; }
        if (d_rev_off) { cudaFree(d_rev_off); d_rev_off = nullptr; }
        if (d_rev_off_tmp) { cudaFree(d_rev_off_tmp); d_rev_off_tmp = nullptr; }
        if (d_rev_idx) { cudaFree(d_rev_idx); d_rev_idx = nullptr; }
        if (d_fw_dist) { cudaFree(d_fw_dist); d_fw_dist = nullptr; }
        if (d_bw_dist) { cudaFree(d_bw_dist); d_bw_dist = nullptr; }
        if (d_cub_tmp) { cudaFree(d_cub_tmp); d_cub_tmp = nullptr; }

        if (h_offsets) { cudaFreeHost(h_offsets); h_offsets = nullptr; }
        if (h_indices) { cudaFreeHost(h_indices); h_indices = nullptr; }
        if (h_components) { cudaFreeHost(h_components); h_components = nullptr; }

        delete[] disc; disc = nullptr;
        delete[] low; low = nullptr;
        delete[] stk; stk = nullptr;
        delete[] on_stack; on_stack = nullptr;
        delete[] call_stack; call_stack = nullptr;

        alloc_v_ = nv;
        alloc_e_ = ne;

        cudaMalloc(&d_dst_keys, alloc_e_ * sizeof(int32_t));
        cudaMalloc(&d_src_vals, alloc_e_ * sizeof(int32_t));
        cudaMalloc(&d_dst_keys_out, alloc_e_ * sizeof(int32_t));
        cudaMalloc(&d_src_vals_out, alloc_e_ * sizeof(int32_t));

        cudaMalloc(&d_indeg, (alloc_v_ + 1) * sizeof(int32_t));
        cudaMalloc(&d_rev_off, (alloc_v_ + 1) * sizeof(int32_t));
        cudaMalloc(&d_rev_off_tmp, (alloc_v_ + 1) * sizeof(int32_t));
        cudaMalloc(&d_rev_idx, alloc_e_ * sizeof(int32_t));

        cudaMalloc(&d_fw_dist, alloc_v_ * sizeof(int32_t));
        cudaMalloc(&d_bw_dist, alloc_v_ * sizeof(int32_t));

        size_t sort_bytes = get_cub_sort_temp_bytes(alloc_e_);
        size_t scan_bytes = get_cub_scan_temp_bytes(alloc_v_);
        cub_tmp_bytes = std::max(sort_bytes, scan_bytes);
        cudaMalloc(&d_cub_tmp, cub_tmp_bytes);

        cudaMallocHost(&h_offsets, (alloc_v_ + 1) * sizeof(int32_t));
        cudaMallocHost(&h_indices, alloc_e_ * sizeof(int32_t));
        cudaMallocHost(&h_components, alloc_v_ * sizeof(int32_t));

        disc = new int32_t[alloc_v_];
        low = new int32_t[alloc_v_];
        stk = new int32_t[alloc_v_];
        on_stack = new int8_t[alloc_v_];
        call_stack = new Frame[alloc_v_];
    }

    void tarjan_full(const int32_t* off, const int32_t* idx, int32_t n, int32_t* comp) {
        memset(disc, 0xFF, n * sizeof(int32_t));
        memset(on_stack, 0, n * sizeof(int8_t));
        int32_t timer = 0, st = 0, cs = 0;
        for (int32_t s = 0; s < n; s++) {
            if (disc[s] >= 0) continue;
            disc[s] = low[s] = timer++;
            stk[st++] = s; on_stack[s] = 1;
            call_stack[cs++] = {s, off[s]};
            while (cs > 0) {
                Frame& f = call_stack[cs - 1];
                int32_t v = f.v, end = off[v + 1]; bool p = false;
                while (f.ei < end) {
                    int32_t w = idx[f.ei++];
                    if (disc[w] < 0) {
                        disc[w] = low[w] = timer++;
                        stk[st++] = w; on_stack[w] = 1;
                        call_stack[cs++] = {w, off[w]};
                        p = true; break;
                    } else if (on_stack[w]) {
                        if (disc[w] < low[v]) low[v] = disc[w];
                    }
                }
                if (!p) {
                    if (low[v] == disc[v]) {
                        int32_t w; do { w = stk[--st]; on_stack[w] = 0; comp[w] = v; } while (w != v);
                    }
                    cs--;
                    if (cs > 0) { int32_t pa = call_stack[cs - 1].v; if (low[v] < low[pa]) low[pa] = low[v]; }
                }
            }
        }
    }

    void tarjan_partial(const int32_t* off, const int32_t* idx, int32_t n, int32_t* comp) {
        memset(disc, 0xFF, n * sizeof(int32_t));
        memset(on_stack, 0, n * sizeof(int8_t));
        int32_t timer = 0, st = 0, cs = 0;
        for (int32_t s = 0; s < n; s++) {
            if (comp[s] >= 0 || disc[s] >= 0) continue;
            disc[s] = low[s] = timer++;
            stk[st++] = s; on_stack[s] = 1;
            call_stack[cs++] = {s, off[s]};
            while (cs > 0) {
                Frame& f = call_stack[cs - 1];
                int32_t v = f.v, end = off[v + 1]; bool p = false;
                while (f.ei < end) {
                    int32_t w = idx[f.ei++];
                    if (comp[w] >= 0) continue;
                    if (disc[w] < 0) {
                        disc[w] = low[w] = timer++;
                        stk[st++] = w; on_stack[w] = 1;
                        call_stack[cs++] = {w, off[w]};
                        p = true; break;
                    } else if (on_stack[w]) {
                        if (disc[w] < low[v]) low[v] = disc[w];
                    }
                }
                if (!p) {
                    if (low[v] == disc[v]) {
                        int32_t w; do { w = stk[--st]; on_stack[w] = 0; comp[w] = v; } while (w != v);
                    }
                    cs--;
                    if (cs > 0) { int32_t pa = call_stack[cs - 1].v; if (low[v] < low[pa]) low[pa] = low[v]; }
                }
            }
        }
    }

    void gpu_cc(const int32_t* doff, const int32_t* didx, int32_t nv, int32_t ne, int32_t* dc) {
        do_cc_label_prop(doff, didx, nv, dc, d_flag, h_flag, MAX_CC_ITERS, sc);
        cudaStreamSynchronize(sc);
    }

    void cpu_scc(const int32_t* doff, const int32_t* didx, int32_t nv, int32_t ne, int32_t* dc) {
        cudaMemcpy(h_offsets, doff, (nv + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_indices, didx, ne * sizeof(int32_t), cudaMemcpyDeviceToHost);
        tarjan_full(h_offsets, h_indices, nv, h_components);
        cudaMemcpy(dc, h_components, nv * sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    void gpu_scc(const int32_t* doff, const int32_t* didx, int32_t nv, int32_t ne,
                 int32_t* dc, int32_t seg_mid) {
        cudaMemcpyAsync(h_offsets, doff, (nv + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost, scopy);
        cudaMemcpyAsync(h_indices, didx, ne * sizeof(int32_t), cudaMemcpyDeviceToHost, scopy);

        cudaMemsetAsync(dc, 0xFF, nv * sizeof(int32_t), sc);

        int bits = compute_bits_needed(nv);

        const int32_t* rev_off_ptr;
        const int32_t* rev_idx_ptr;

        if (ne > 500000) {
            do_build_transpose_sort(doff, didx, nv, ne,
                                    d_dst_keys, d_src_vals,
                                    d_dst_keys_out, d_src_vals_out,
                                    d_rev_off,
                                    d_cub_tmp, cub_tmp_bytes, bits, sc);
            rev_off_ptr = d_rev_off;
            rev_idx_ptr = d_src_vals_out;
        } else {
            do_build_transpose_atomic(doff, didx, nv, ne,
                                      d_indeg, d_rev_off, d_rev_off_tmp, d_rev_idx,
                                      d_cub_tmp, cub_tmp_bytes, sc);
            rev_off_ptr = d_rev_off;
            rev_idx_ptr = d_rev_idx;
        }

        int32_t pivot = 0;
        cudaStreamSynchronize(sc);

        do_bfs_converge_segmented(doff, didx, rev_off_ptr, rev_idx_ptr,
                                  nv, d_fw_dist, d_bw_dist, pivot, MAX_BFS_LEVELS,
                                  d_flag, seg_mid, sc);
        do_mark_scc(nv, d_fw_dist, d_bw_dist, dc, pivot, sc);

        int32_t remaining = do_trim_and_count(doff, didx, rev_off_ptr, rev_idx_ptr,
                                              nv, dc, d_flag, h_flag, MAX_TRIM_ITERS, sc);

        if (remaining > 0 && remaining < nv * 3 / 4) {
            for (int round = 1; round < MAX_FWBW_ROUNDS && remaining > 0; round++) {
                if (remaining < nv / 8) break;

                int32_t new_pivot = do_find_pivot(dc, doff, nv, d_flag, h_flag, sc);
                if (new_pivot >= nv) break;

                do_bfs_converge_segmented_restricted(doff, didx, rev_off_ptr, rev_idx_ptr,
                                                     nv, d_fw_dist, d_bw_dist, dc,
                                                     new_pivot, MAX_BFS_LEVELS,
                                                     d_flag, seg_mid, sc);
                do_mark_scc_restricted(nv, d_fw_dist, d_bw_dist, dc, new_pivot, sc);

                remaining = do_trim_and_count(doff, didx, rev_off_ptr, rev_idx_ptr,
                                              nv, dc, d_flag, h_flag, MAX_TRIM_ITERS, sc);
            }
        }

        if (remaining > 0) {
            cudaStreamSynchronize(scopy);
            cudaMemcpyAsync(h_components, dc, nv * sizeof(int32_t), cudaMemcpyDeviceToHost, sc);
            cudaStreamSynchronize(sc);
            tarjan_partial(h_offsets, h_indices, nv, h_components);
            cudaMemcpy(dc, h_components, nv * sizeof(int32_t), cudaMemcpyHostToDevice);
        } else {
            cudaStreamSynchronize(sc);
        }
    }
};

}  

void strongly_connected_components_seg(const graph32_t& graph,
                                       int32_t* components) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* doff = graph.offsets;
    const int32_t* didx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;

    cache.ensure_buffers(nv, ne);

    if (ne <= 50000) {
        cache.cpu_scc(doff, didx, nv, ne, components);
        return;
    }

    bool is_symmetric = graph.is_symmetric;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg_mid = seg[2];

    if (is_symmetric) {
        cache.gpu_cc(doff, didx, nv, ne, components);
    } else {
        cache.gpu_scc(doff, didx, nv, ne, components, seg_mid);
    }
}

}  
