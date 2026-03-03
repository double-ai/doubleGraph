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
#include <algorithm>
#include <vector>

namespace aai {

namespace {





__global__ void init_expand_kernel(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ idx,
    uint32_t* __restrict__ bitmap,
    int32_t source,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ ego_verts,
    int32_t* __restrict__ counters  
) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    if (threadIdx.x == 0) {
        bitmap[source >> 5] |= (1u << (source & 31));
        ego_verts[0] = source;
        counters[0] = 0;
        counters[1] = 1;
    }
    __syncthreads();

    
    int32_t s = off[source];
    int32_t e = off[source + 1];
    int32_t total_lanes = blockDim.x;

    for (int32_t j = s + threadIdx.x; j < e; j += total_lanes) {
        int32_t u = idx[j];
        uint32_t mask = 1u << (u & 31);
        uint32_t old = atomicOr(&bitmap[u >> 5], mask);
        if (!(old & mask)) {
            int pos = atomicAdd(&counters[0], 1);
            next_frontier[pos] = u;
            int ep = atomicAdd(&counters[1], 1);
            ego_verts[ep] = u;
        }
    }
}

__global__ void bfs_expand_warp_kernel(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ idx,
    const int32_t* __restrict__ frontier,
    int32_t frontier_size,
    uint32_t* __restrict__ bitmap,
    int32_t* __restrict__ next_frontier,
    int32_t* __restrict__ next_frontier_size,
    int32_t* __restrict__ ego_verts,
    int32_t* __restrict__ ego_count
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= frontier_size) return;

    int32_t v = frontier[warp_id];
    int32_t s = off[v];
    int32_t e = off[v + 1];

    for (int32_t j = s + lane; j < e; j += 32) {
        int32_t u = idx[j];
        uint32_t mask = 1u << (u & 31);
        uint32_t old = atomicOr(&bitmap[u >> 5], mask);
        if (!(old & mask)) {
            int pos = atomicAdd(next_frontier_size, 1);
            next_frontier[pos] = u;
            int ep = atomicAdd(ego_count, 1);
            ego_verts[ep] = u;
        }
    }
}

__global__ void count_edges_warp_kernel(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ idx,
    const int32_t* __restrict__ ego_verts,
    int32_t ego_count,
    const uint32_t* __restrict__ bitmap,
    int32_t* __restrict__ edge_counts
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= ego_count) return;

    int32_t v = ego_verts[warp_id];
    int32_t s = off[v];
    int32_t e = off[v + 1];
    int32_t count = 0;

    for (int32_t j = s + lane; j < e; j += 32) {
        int32_t u = idx[j];
        if (bitmap[u >> 5] & (1u << (u & 31)))
            count++;
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        count += __shfl_down_sync(0xFFFFFFFF, count, o);

    if (lane == 0)
        edge_counts[warp_id] = count;
}

__global__ void write_edges_warp_kernel(
    const int32_t* __restrict__ off,
    const int32_t* __restrict__ idx,
    const int32_t* __restrict__ ego_verts,
    int32_t ego_count,
    const uint32_t* __restrict__ bitmap,
    const int32_t* __restrict__ edge_offsets,
    int32_t* __restrict__ out_srcs,
    int32_t* __restrict__ out_dsts,
    int64_t base_offset
) {
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;

    if (warp_id >= ego_count) return;

    int32_t v = ego_verts[warp_id];
    int32_t s = off[v];
    int32_t e = off[v + 1];
    int32_t deg = e - s;
    int32_t chunks = (deg + 31) >> 5;
    int64_t wb = base_offset + edge_offsets[warp_id];

    for (int32_t c = 0; c < chunks; c++) {
        int32_t j = s + (c << 5) + lane;
        int match = 0;
        int32_t u = 0;
        if (j < e) {
            u = idx[j];
            match = (bitmap[u >> 5] & (1u << (u & 31))) ? 1 : 0;
        }

        unsigned int ballot = __ballot_sync(0xFFFFFFFF, match);
        int prefix = __popc(ballot & ((1u << lane) - 1));

        if (match) {
            out_srcs[wb + prefix] = v;
            out_dsts[wb + prefix] = u;
        }
        wb += __popc(ballot);
    }
}

__global__ void clear_bitmap_kernel(
    uint32_t* __restrict__ bitmap,
    const int32_t* __restrict__ verts,
    int32_t count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    int32_t v = verts[tid];
    atomicAnd(&bitmap[v >> 5], ~(1u << (v & 31)));
}

__global__ void check_sorted_kernel(
    const int32_t* __restrict__ sr, const int32_t* __restrict__ ds,
    int64_t n, int32_t* __restrict__ flag
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n - 1) return;
    if (sr[tid] > sr[tid + 1] || (sr[tid] == sr[tid + 1] && ds[tid] > ds[tid + 1]))
        *flag = 0;
}

__global__ void pack_keys_kernel(
    const int32_t* __restrict__ sr, const int32_t* __restrict__ ds,
    int64_t* __restrict__ k, int64_t n, int64_t base
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    k[tid] = ((int64_t)sr[base + tid] << 32) | (uint32_t)ds[base + tid];
}

__global__ void unpack_keys_kernel(
    const int64_t* __restrict__ k, int32_t* __restrict__ sr, int32_t* __restrict__ ds,
    int64_t n, int64_t base
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    sr[base + tid] = (int32_t)(k[tid] >> 32);
    ds[base + tid] = (int32_t)(k[tid] & 0xFFFFFFFF);
}





struct Cache : Cacheable {
    
    uint32_t* bm = nullptr;
    int32_t* fa = nullptr;
    int32_t* fb = nullptr;
    int32_t* ego = nullptr;
    int32_t* sego = nullptr;
    int32_t* ec = nullptr;
    int32_t* eo = nullptr;
    int32_t* cnt = nullptr;
    int32_t nv_cap = 0;

    
    void* cub_ws = nullptr;
    size_t cub_ws_cap = 0;

    
    int64_t* keys = nullptr;
    int64_t* kout = nullptr;
    int64_t sort_cap = 0;

    ~Cache() override {
        if (bm) cudaFree(bm);
        if (fa) cudaFree(fa);
        if (fb) cudaFree(fb);
        if (ego) cudaFree(ego);
        if (sego) cudaFree(sego);
        if (ec) cudaFree(ec);
        if (eo) cudaFree(eo);
        if (cnt) cudaFree(cnt);
        if (cub_ws) cudaFree(cub_ws);
        if (keys) cudaFree(keys);
        if (kout) cudaFree(kout);
    }

    void ensure_cub_ws(size_t needed) {
        if (cub_ws_cap < needed) {
            if (cub_ws) cudaFree(cub_ws);
            cudaMalloc(&cub_ws, needed);
            cub_ws_cap = needed;
        }
    }

    void ensure_nv(int32_t nv) {
        if (nv_cap >= nv) return;

        if (bm) cudaFree(bm);
        if (fa) cudaFree(fa);
        if (fb) cudaFree(fb);
        if (ego) cudaFree(ego);
        if (sego) cudaFree(sego);
        if (ec) cudaFree(ec);
        if (eo) cudaFree(eo);
        if (cnt) cudaFree(cnt);

        int32_t bw = (nv + 31) / 32;
        cudaMalloc(&bm, (size_t)bw * sizeof(uint32_t));
        cudaMalloc(&fa, (size_t)nv * sizeof(int32_t));
        cudaMalloc(&fb, (size_t)nv * sizeof(int32_t));
        cudaMalloc(&ego, (size_t)nv * sizeof(int32_t));
        cudaMalloc(&sego, (size_t)nv * sizeof(int32_t));
        cudaMalloc(&ec, ((size_t)nv + 1) * sizeof(int32_t));
        cudaMalloc(&eo, ((size_t)nv + 1) * sizeof(int32_t));
        cudaMalloc(&cnt, 4 * sizeof(int32_t));

        nv_cap = nv;
    }

    void ensure_sort(int64_t n) {
        if (sort_cap >= n) return;

        if (keys) cudaFree(keys);
        if (kout) cudaFree(kout);
        cudaMalloc(&keys, (size_t)n * sizeof(int64_t));
        cudaMalloc(&kout, (size_t)n * sizeof(int64_t));
        sort_cap = n;
    }
};

}  

extract_ego_result_t extract_ego(const graph32_t& graph,
                                 const int32_t* source_vertices,
                                 std::size_t n_sources,
                                 int32_t radius) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    int32_t ne = graph.number_of_edges;
    int64_t ns = (int64_t)n_sources;
    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    cudaStream_t stream = 0;

    
    cache.ensure_nv(nv);

    
    size_t st32 = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, st32, (int32_t*)nullptr, (int32_t*)nullptr, nv);
    size_t stscan = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, stscan, (int32_t*)nullptr, (int32_t*)nullptr, nv + 1);
    size_t max_t = std::max(st32, stscan);
    cache.ensure_cub_ws(max_t);

    
    int64_t max_sort = std::max((int64_t)ne, (int64_t)1);
    size_t st64 = 0;
    cub::DeviceRadixSort::SortKeys(nullptr, st64, (int64_t*)nullptr, (int64_t*)nullptr,
                                   ne > 0 ? (int32_t)std::min((int64_t)ne, (int64_t)INT32_MAX) : 1);
    cache.ensure_sort(max_sort);
    cache.ensure_cub_ws(st64);

    
    std::vector<int32_t> h_src(ns);
    if (ns > 0) {
        cudaMemcpy(h_src.data(), source_vertices, ns * sizeof(int32_t), cudaMemcpyDeviceToHost);
    }

    
    int32_t bw = (nv + 31) / 32;
    cudaMemsetAsync(cache.bm, 0, (size_t)bw * sizeof(uint32_t), stream);

    
    int64_t buf_cap = std::max((int64_t)ne, (int64_t)1);
    int32_t* buf_s = nullptr;
    int32_t* buf_d = nullptr;
    cudaMalloc(&buf_s, (size_t)buf_cap * sizeof(int32_t));
    cudaMalloc(&buf_d, (size_t)buf_cap * sizeof(int32_t));

    int64_t cur_off = 0;
    std::vector<std::size_t> h_ego_off(ns + 1);
    h_ego_off[0] = 0;

    for (int64_t s = 0; s < ns; s++) {
        
        init_expand_kernel<<<1, 256, 0, stream>>>(d_off, d_idx, cache.bm, h_src[s],
                                                   cache.fa, cache.ego, cache.cnt);

        int32_t h_batch[2]; 

        if (radius == 1) {
            cudaMemcpyAsync(&h_batch[1], &cache.cnt[1], sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            h_batch[0] = 0;
        } else {
            cudaMemcpyAsync(h_batch, cache.cnt, 2 * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            int32_t h_fs = h_batch[0];
            int32_t* cur = cache.fa;
            int32_t* nxt = cache.fb;

            
            for (int32_t r = 1; r < radius && h_fs > 0; r++) {
                cudaMemsetAsync(&cache.cnt[0], 0, sizeof(int32_t), stream);

                int warps = h_fs;
                int64_t threads = (int64_t)warps * 32;
                int b = 256, g = (int)((threads + b - 1) / b);
                bfs_expand_warp_kernel<<<g, b, 0, stream>>>(d_off, d_idx, cur, h_fs,
                    cache.bm, nxt, &cache.cnt[0], cache.ego, &cache.cnt[1]);

                if (r == radius - 1) {
                    cudaMemcpyAsync(h_batch, cache.cnt, 2 * sizeof(int32_t),
                                    cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    h_fs = h_batch[0];
                } else {
                    cudaMemcpyAsync(&h_fs, &cache.cnt[0], sizeof(int32_t),
                                    cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                }

                int32_t* tmp = cur; cur = nxt; nxt = tmp;
            }
        }

        int32_t h_ec = h_batch[1];

        if (h_ec == 0) {
            h_ego_off[s + 1] = cur_off;
            continue;
        }

        
        cub::DeviceRadixSort::SortKeys(cache.cub_ws, st32, cache.ego, cache.sego, h_ec, 0, 32, stream);

        
        {
            int64_t threads = (int64_t)h_ec * 32;
            int b = 256, g = (int)((threads + b - 1) / b);
            count_edges_warp_kernel<<<g, b, 0, stream>>>(d_off, d_idx, cache.sego, h_ec,
                                                          cache.bm, cache.ec);
        }

        
        cudaMemsetAsync(&cache.ec[h_ec], 0, sizeof(int32_t), stream);
        cub::DeviceScan::ExclusiveSum(cache.cub_ws, stscan, cache.ec, cache.eo, h_ec + 1, stream);

        
        int32_t h_te;
        cudaMemcpyAsync(&h_te, &cache.eo[h_ec], sizeof(int32_t),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        int64_t h_total = (int64_t)h_te;

        
        if (cur_off + h_total > buf_cap) {
            int64_t new_cap = std::max(cur_off + h_total, buf_cap * 2);
            int32_t* new_s = nullptr;
            int32_t* new_d = nullptr;
            cudaMalloc(&new_s, (size_t)new_cap * sizeof(int32_t));
            cudaMalloc(&new_d, (size_t)new_cap * sizeof(int32_t));
            if (cur_off > 0) {
                cudaMemcpyAsync(new_s, buf_s, cur_off * sizeof(int32_t),
                                cudaMemcpyDeviceToDevice, stream);
                cudaMemcpyAsync(new_d, buf_d, cur_off * sizeof(int32_t),
                                cudaMemcpyDeviceToDevice, stream);
            }
            cudaStreamSynchronize(stream);
            cudaFree(buf_s);
            cudaFree(buf_d);
            buf_s = new_s;
            buf_d = new_d;
            buf_cap = new_cap;
        }

        if (h_total > 0) {
            
            {
                int64_t threads = (int64_t)h_ec * 32;
                int b = 256, g = (int)((threads + b - 1) / b);
                write_edges_warp_kernel<<<g, b, 0, stream>>>(d_off, d_idx, cache.sego, h_ec,
                    cache.bm, cache.eo, buf_s, buf_d, cur_off);
            }

            
            int32_t h_sorted = 1;
            cudaMemcpyAsync(&cache.cnt[2], &h_sorted, sizeof(int32_t),
                            cudaMemcpyHostToDevice, stream);
            if (h_total > 1) {
                int b = 256, g = (int)((h_total - 1 + b - 1) / b);
                check_sorted_kernel<<<g, b, 0, stream>>>(buf_s + cur_off, buf_d + cur_off,
                                                          h_total, &cache.cnt[2]);
            }
            cudaMemcpyAsync(&h_sorted, &cache.cnt[2], sizeof(int32_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);

            if (!h_sorted) {
                if (h_total > max_sort) {
                    max_sort = h_total;
                    st64 = 0;
                    cub::DeviceRadixSort::SortKeys(nullptr, st64, (int64_t*)nullptr, (int64_t*)nullptr,
                        (int32_t)std::min(h_total, (int64_t)INT32_MAX));
                    cache.ensure_sort(h_total);
                    cache.ensure_cub_ws(st64);
                }
                {
                    int b = 256, g = (int)((h_total + b - 1) / b);
                    pack_keys_kernel<<<g, b, 0, stream>>>(buf_s, buf_d, cache.keys,
                                                           h_total, cur_off);
                }
                cub::DeviceRadixSort::SortKeys(cache.cub_ws, st64, cache.keys, cache.kout,
                                               (int32_t)h_total, 0, 64, stream);
                {
                    int b = 256, g = (int)((h_total + b - 1) / b);
                    unpack_keys_kernel<<<g, b, 0, stream>>>(cache.kout, buf_s, buf_d,
                                                              h_total, cur_off);
                }
            }
        }

        cur_off += h_total;
        h_ego_off[s + 1] = cur_off;

        
        {
            int b = 256, g = (h_ec + b - 1) / b;
            clear_bitmap_kernel<<<g, b, 0, stream>>>(cache.bm, cache.sego, h_ec);
        }
    }

    
    int64_t total = cur_off;
    extract_ego_result_t result;
    result.num_edges = (std::size_t)total;
    result.num_offsets = (std::size_t)(ns + 1);

    if (total > 0) {
        if (total == buf_cap) {
            result.edge_srcs = buf_s;
            result.edge_dsts = buf_d;
        } else {
            cudaMalloc(&result.edge_srcs, (size_t)total * sizeof(int32_t));
            cudaMalloc(&result.edge_dsts, (size_t)total * sizeof(int32_t));
            cudaMemcpyAsync(result.edge_srcs, buf_s, total * sizeof(int32_t),
                            cudaMemcpyDeviceToDevice, stream);
            cudaMemcpyAsync(result.edge_dsts, buf_d, total * sizeof(int32_t),
                            cudaMemcpyDeviceToDevice, stream);
            cudaStreamSynchronize(stream);
            cudaFree(buf_s);
            cudaFree(buf_d);
        }
    } else {
        result.edge_srcs = nullptr;
        result.edge_dsts = nullptr;
        cudaFree(buf_s);
        cudaFree(buf_d);
    }

    
    cudaMalloc(&result.offsets, (size_t)(ns + 1) * sizeof(std::size_t));
    cudaMemcpyAsync(result.offsets, h_ego_off.data(), (ns + 1) * sizeof(std::size_t),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    return result;
}

}  
