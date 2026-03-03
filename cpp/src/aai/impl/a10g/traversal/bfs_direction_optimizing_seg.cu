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

namespace aai {

namespace {

#define BLOCK_SIZE 256
#define WARP_SIZE 32



struct Cache : Cacheable {
    uint32_t* vis_bmp = nullptr;
    uint32_t* f_bmp = nullptr;
    uint32_t* nf_bmp = nullptr;
    int32_t* fq1 = nullptr;
    int32_t* fq2 = nullptr;
    int32_t* scratch = nullptr;  
    int32_t* h_counter = nullptr;
    int32_t capacity = 0;

    Cache() {
        cudaMallocHost(&h_counter, sizeof(int32_t));
    }

    ~Cache() override {
        if (vis_bmp) cudaFree(vis_bmp);
        if (f_bmp) cudaFree(f_bmp);
        if (nf_bmp) cudaFree(nf_bmp);
        if (fq1) cudaFree(fq1);
        if (fq2) cudaFree(fq2);
        if (scratch) cudaFree(scratch);
        if (h_counter) cudaFreeHost(h_counter);
    }

    void ensure(int32_t num_vertices) {
        if (capacity >= num_vertices) return;
        if (vis_bmp) cudaFree(vis_bmp);
        if (f_bmp) cudaFree(f_bmp);
        if (nf_bmp) cudaFree(nf_bmp);
        if (fq1) cudaFree(fq1);
        if (fq2) cudaFree(fq2);
        if (scratch) cudaFree(scratch);
        int32_t bitmap_words = (num_vertices + 31) / 32;
        cudaMalloc(&vis_bmp, bitmap_words * sizeof(uint32_t));
        cudaMalloc(&f_bmp, bitmap_words * sizeof(uint32_t));
        cudaMalloc(&nf_bmp, bitmap_words * sizeof(uint32_t));
        cudaMalloc(&fq1, num_vertices * sizeof(int32_t));
        cudaMalloc(&fq2, num_vertices * sizeof(int32_t));
        cudaMalloc(&scratch, 3 * sizeof(int32_t));
        capacity = num_vertices;
    }
};




__global__ void bfs_init(int32_t* __restrict__ dist, int32_t* __restrict__ pred,
                         int32_t n, bool has_pred) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dist[i] = 0x7FFFFFFF;
        if (has_pred) pred[i] = -1;
    }
}

__global__ void bfs_set_sources(const int32_t* __restrict__ srcs, int32_t ns,
                                int32_t* __restrict__ dist,
                                uint32_t* __restrict__ vis,
                                int32_t* __restrict__ f_queue,
                                int32_t* __restrict__ f_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ns) {
        int32_t s = srcs[i];
        dist[s] = 0;
        atomicOr(&vis[s >> 5], 1u << (s & 31));
        f_queue[atomicAdd(f_count, 1)] = s;
    }
}


__global__ void set_depth_kernel(int32_t* d, int32_t val) { *d = val; }
__global__ void inc_depth_kernel(int32_t* d) { *d += 1; }


__global__ void bfs_td_warp(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    int32_t* __restrict__ dist, int32_t* __restrict__ pred,
    const int32_t* __restrict__ cur_q, int32_t fsize,
    int32_t* __restrict__ nxt_q, int32_t* __restrict__ nxt_cnt,
    uint32_t* __restrict__ vis, uint32_t* __restrict__ nxt_bmp,
    int32_t d, bool has_pred)
{
    int warp_gid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (blockDim.x * gridDim.x) >> 5;
    for (int i = warp_gid; i < fsize; i += total_warps) {
        int32_t v = cur_q[i];
        int32_t s = off[v], e = off[v + 1];
        for (int32_t j = s + lane; j < e; j += WARP_SIZE) {
            int32_t u = idx[j];
            uint32_t w = u >> 5, b = 1u << (u & 31);
            if (vis[w] & b) continue;
            uint32_t old = atomicOr(&vis[w], b);
            if (!(old & b)) {
                dist[u] = d;
                if (has_pred) pred[u] = v;
                nxt_q[atomicAdd(nxt_cnt, 1)] = u;
                if (nxt_bmp) atomicOr(&nxt_bmp[w], b);
            }
        }
    }
}


__global__ void bfs_td_graph(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    int32_t* __restrict__ dist, int32_t* __restrict__ pred,
    const int32_t* __restrict__ cur_q, const int32_t* __restrict__ cur_cnt,
    int32_t* __restrict__ nxt_q, int32_t* __restrict__ nxt_cnt,
    uint32_t* __restrict__ vis,
    const int32_t* __restrict__ d_depth, bool has_pred)
{
    int32_t fsize = *cur_cnt;
    if (fsize == 0) return;
    int32_t d = *d_depth;
    int warp_gid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    int total_warps = (blockDim.x * gridDim.x) >> 5;
    for (int i = warp_gid; i < fsize; i += total_warps) {
        int32_t v = cur_q[i];
        int32_t s = off[v], e = off[v + 1];
        for (int32_t j = s + lane; j < e; j += WARP_SIZE) {
            int32_t u = idx[j];
            uint32_t w = u >> 5, b = 1u << (u & 31);
            if (vis[w] & b) continue;
            uint32_t old = atomicOr(&vis[w], b);
            if (!(old & b)) {
                dist[u] = d;
                if (has_pred) pred[u] = v;
                nxt_q[atomicAdd(nxt_cnt, 1)] = u;
            }
        }
    }
}


__global__ void bfs_bu(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    int32_t* __restrict__ dist, int32_t* __restrict__ pred,
    const uint32_t* __restrict__ f_bmp, uint32_t* __restrict__ vis,
    uint32_t* __restrict__ next_bmp, int32_t* __restrict__ next_count,
    int32_t nzd_vertices, int32_t d, bool has_pred)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nzd_vertices) return;
    uint32_t vis_word = vis[v >> 5];
    int lane = v & 31;
    bool is_visited = (vis_word >> lane) & 1;
    if (__all_sync(0xffffffff, is_visited)) return;
    if (is_visited) return;
    int32_t s = off[v], e = off[v + 1];
    for (int32_t i = s; i < e; i++) {
        int32_t u = idx[i];
        if (f_bmp[u >> 5] & (1u << (u & 31))) {
            dist[v] = d;
            if (has_pred) pred[v] = u;
            atomicOr(&vis[v >> 5], 1u << lane);
            atomicOr(&next_bmp[v >> 5], 1u << lane);
            atomicAdd(next_count, 1);
            break;
        }
    }
}


__global__ void queue_to_bitmap_sized(const int32_t* __restrict__ queue,
                                       int32_t qsize, uint32_t* __restrict__ bitmap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < qsize) {
        int32_t v = queue[i];
        atomicOr(&bitmap[v >> 5], 1u << (v & 31));
    }
}

__global__ void bitmap_to_queue(const uint32_t* __restrict__ bitmap,
                                int32_t num_vertices, int32_t* __restrict__ queue,
                                int32_t* __restrict__ qsize) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    if (bitmap[v >> 5] & (1u << (v & 31))) {
        queue[atomicAdd(qsize, 1)] = v;
    }
}

}  

void bfs_direction_optimizing_seg(const graph32_t& graph,
                                  int32_t* distances,
                                  int32_t* predecessors,
                                  const int32_t* sources,
                                  std::size_t n_sources,
                                  int32_t depth_limit) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* offsets = graph.offsets;
    const int32_t* indices = graph.indices;
    int32_t num_vertices = graph.number_of_vertices;
    int32_t num_edges = graph.number_of_edges;
    bool compute_pred = (predecessors != nullptr);

    const auto& seg = graph.segment_offsets.value();
    int32_t nzd_vertices = seg[3];

    cache.ensure(num_vertices);

    uint32_t* vis_bmp = cache.vis_bmp;
    uint32_t* f_bmp = cache.f_bmp;
    uint32_t* nf_bmp = cache.nf_bmp;
    int32_t* fq1 = cache.fq1;
    int32_t* fq2 = cache.fq2;
    int32_t* cnt0 = cache.scratch;
    int32_t* cnt1 = cache.scratch + 1;
    int32_t* d_depth = cache.scratch + 2;
    int32_t* h_counter = cache.h_counter;

    int32_t bitmap_words = (num_vertices + 31) / 32;
    int bv = (num_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int bnzd = (nzd_vertices + BLOCK_SIZE - 1) / BLOCK_SIZE;

    
    int numBlocksPerSm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, bfs_td_graph, BLOCK_SIZE, 0);
    if (numBlocksPerSm <= 0) numBlocksPerSm = 4;
    int max_grid = numBlocksPerSm * 80;

    
    bfs_init<<<bv, BLOCK_SIZE>>>(distances, predecessors, num_vertices, compute_pred);
    cudaMemsetAsync(vis_bmp, 0, bitmap_words * sizeof(uint32_t));
    cudaMemsetAsync(cnt0, 0, sizeof(int32_t));

    int32_t ns = static_cast<int32_t>(n_sources);
    int bs = (ns + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (bs < 1) bs = 1;
    bfs_set_sources<<<bs, BLOCK_SIZE>>>(sources, ns, distances, vis_bmp, fq1, cnt0);
    cudaMemcpy(h_counter, cnt0, sizeof(int32_t), cudaMemcpyDeviceToHost);
    int32_t frontier_size = *h_counter;

    int32_t depth = 0;
    int32_t max_depth = (depth_limit < 0) ? 0x7FFFFFFF : depth_limit;

    double avg_degree = (num_vertices > 0) ? (double)num_edges / num_vertices : 0.0;
    double alpha = avg_degree * 0.5;
    if (alpha < 2.0) alpha = 2.0;
    const int32_t beta = 24;
    bool enable_do = (avg_degree >= 4.0) && (num_vertices > 10000);

    if (enable_do) {
        
        int32_t* cur = fq1; int32_t* nxt = fq2;
        bool is_top_down = true;
        int64_t total_visited = ns;
        int32_t prev_frontier_size = 0;

        while (frontier_size > 0) {
            int32_t new_dist = depth + 1;
            if (is_top_down) {
                int64_t remaining = (int64_t)nzd_vertices - total_visited;
                if (remaining < 0) remaining = 0;
                bool growing = (frontier_size >= prev_frontier_size);
                if ((double)frontier_size * alpha > (double)remaining && growing && remaining > 0) {
                    is_top_down = false;
                    cudaMemsetAsync(f_bmp, 0, bitmap_words * sizeof(uint32_t));
                    int qb = (frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    queue_to_bitmap_sized<<<qb, BLOCK_SIZE>>>(cur, frontier_size, f_bmp);
                }
            }
            if (is_top_down) {
                cudaMemsetAsync(cnt0, 0, sizeof(int32_t));
                cudaMemsetAsync(nf_bmp, 0, bitmap_words * sizeof(uint32_t));
                int grid = (int)(((int64_t)frontier_size * WARP_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
                if (grid < 1) grid = 1;
                if (grid > max_grid) grid = max_grid;
                bfs_td_warp<<<grid, BLOCK_SIZE>>>(offsets, indices, distances, predecessors,
                    cur, frontier_size, nxt, cnt0, vis_bmp, nf_bmp, new_dist, compute_pred);
                cudaMemcpy(h_counter, cnt0, sizeof(int32_t), cudaMemcpyDeviceToHost);
                prev_frontier_size = frontier_size;
                frontier_size = *h_counter;
                total_visited += frontier_size;
                int32_t* tmp = cur; cur = nxt; nxt = tmp;
                uint32_t* tmp2 = f_bmp; f_bmp = nf_bmp; nf_bmp = tmp2;
            } else {
                cudaMemsetAsync(nf_bmp, 0, bitmap_words * sizeof(uint32_t));
                cudaMemsetAsync(cnt0, 0, sizeof(int32_t));
                bfs_bu<<<bnzd, BLOCK_SIZE>>>(offsets, indices, distances, predecessors,
                    f_bmp, vis_bmp, nf_bmp, cnt0, nzd_vertices, new_dist, compute_pred);
                cudaMemcpy(h_counter, cnt0, sizeof(int32_t), cudaMemcpyDeviceToHost);
                int32_t old_fs = frontier_size;
                frontier_size = *h_counter;
                total_visited += frontier_size;
                prev_frontier_size = old_fs;
                uint32_t* tmp = f_bmp; f_bmp = nf_bmp; nf_bmp = tmp;
                int64_t remaining = (int64_t)nzd_vertices - total_visited;
                if (remaining < 0) remaining = 0;
                if ((frontier_size < old_fs) && (int64_t)frontier_size * beta < remaining) {
                    is_top_down = true;
                    cudaMemsetAsync(cnt0, 0, sizeof(int32_t));
                    bitmap_to_queue<<<bnzd, BLOCK_SIZE>>>(f_bmp, nzd_vertices, cur, cnt0);
                }
            }
            depth++;
            if (depth >= max_depth) break;
        }
    } else {
        
        int32_t* q[2] = {fq1, fq2};
        int32_t* cnt[2] = {cnt0, cnt1};
        cudaMemsetAsync(cnt1, 0, sizeof(int32_t));

        
        set_depth_kernel<<<1, 1>>>(d_depth, 1);
        cudaDeviceSynchronize();

        
        cudaStream_t cap_stream;
        cudaStreamCreate(&cap_stream);
        cudaGraph_t graph_obj;
        cudaGraphExec_t graph_exec;

        cudaStreamBeginCapture(cap_stream, cudaStreamCaptureModeGlobal);

        
        cudaMemsetAsync(cnt[1], 0, sizeof(int32_t), cap_stream);
        bfs_td_graph<<<max_grid, BLOCK_SIZE, 0, cap_stream>>>(
            offsets, indices, distances, predecessors,
            q[0], cnt[0], q[1], cnt[1], vis_bmp, d_depth, compute_pred);
        inc_depth_kernel<<<1, 1, 0, cap_stream>>>(d_depth);

        
        cudaMemsetAsync(cnt[0], 0, sizeof(int32_t), cap_stream);
        bfs_td_graph<<<max_grid, BLOCK_SIZE, 0, cap_stream>>>(
            offsets, indices, distances, predecessors,
            q[1], cnt[1], q[0], cnt[0], vis_bmp, d_depth, compute_pred);
        inc_depth_kernel<<<1, 1, 0, cap_stream>>>(d_depth);

        cudaStreamEndCapture(cap_stream, &graph_obj);
        cudaGraphInstantiate(&graph_exec, graph_obj, NULL, NULL, 0);
        cudaGraphDestroy(graph_obj);
        cudaStreamDestroy(cap_stream);

        
        const int REPLAYS_PER_CHECK = 32;

        while (frontier_size > 0) {
            int remaining = max_depth - depth;
            int graph_replays = remaining / 2;

            if (graph_replays > 0) {
                int replays = graph_replays < REPLAYS_PER_CHECK ? graph_replays : REPLAYS_PER_CHECK;

                for (int r = 0; r < replays; r++) {
                    cudaGraphLaunch(graph_exec, 0);
                }

                
                cudaMemcpy(h_counter, cnt[0], sizeof(int32_t), cudaMemcpyDeviceToHost);
                depth += replays * 2;
                frontier_size = *h_counter;
                if (depth >= max_depth) break;
            } else {
                
                cudaMemsetAsync(cnt[1], 0, sizeof(int32_t));
                bfs_td_graph<<<max_grid, BLOCK_SIZE>>>(
                    offsets, indices, distances, predecessors,
                    q[0], cnt[0], q[1], cnt[1], vis_bmp, d_depth, compute_pred);
                inc_depth_kernel<<<1, 1>>>(d_depth);
                cudaMemcpy(h_counter, cnt[1], sizeof(int32_t), cudaMemcpyDeviceToHost);
                frontier_size = *h_counter;
                depth++;
                break;
            }
        }

        cudaGraphExecDestroy(graph_exec);
    }
}

}  
