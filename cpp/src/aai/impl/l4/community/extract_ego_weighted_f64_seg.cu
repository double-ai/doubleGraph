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
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <vector>

namespace aai {

namespace {

struct Cache : Cacheable {
    uint32_t* d_vis = nullptr;
    uint32_t* d_fr = nullptr;
    uint32_t* d_nf = nullptr;
    int64_t* d_ec = nullptr;
    int64_t* d_eo = nullptr;
    void* d_pt = nullptr;
    int* d_wc = nullptr;
    size_t bas = 0;
    int mns = 0;
    size_t ptb = 0;

    void ensure(int ns, int bw) {
        size_t need = (size_t)ns * bw * sizeof(uint32_t);
        if (need > bas || ns > mns) {
            auto sf = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
            sf(d_vis); sf(d_fr); sf(d_nf); sf(d_ec); sf(d_eo); sf(d_pt); sf(d_wc);
            cudaMalloc(&d_vis, need);
            cudaMalloc(&d_fr, need);
            cudaMalloc(&d_nf, need);
            cudaMalloc(&d_ec, (size_t)(ns + 1) * sizeof(int64_t));
            cudaMalloc(&d_eo, (size_t)(ns + 1) * sizeof(int64_t));
            cudaMalloc(&d_wc, (size_t)ns * sizeof(int));
            ptb = 0;
            cub::DeviceScan::ExclusiveSum(nullptr, ptb, (int64_t*)nullptr, (int64_t*)nullptr, ns + 1);
            cudaMalloc(&d_pt, ptb);
            bas = need;
            mns = ns;
        }
    }

    ~Cache() override {
        auto sf = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        sf(d_vis); sf(d_fr); sf(d_nf); sf(d_ec); sf(d_eo); sf(d_pt); sf(d_wc);
    }
};




__global__ void bfs_mark_sources_and_hop1(
    const int32_t* __restrict__ csr_offsets, const int32_t* __restrict__ csr_indices,
    const int32_t* __restrict__ sources, uint32_t* __restrict__ visited,
    int n_sources, int bitmap_words) {
    int src_id = blockIdx.x;
    if (src_id >= n_sources) return;
    uint32_t* v = visited + (int64_t)src_id * bitmap_words;
    int s = sources[src_id];
    if (threadIdx.x == 0) atomicOr(&v[s>>5], 1u<<(s&31));
    int rs = csr_offsets[s], re = csr_offsets[s+1];
    for (int e = rs + (int)threadIdx.x; e < re; e += (int)blockDim.x) {
        int n = csr_indices[e];
        atomicOr(&v[n>>5], 1u<<(n&31));
    }
}

__global__ void bfs_expand_hop2(
    const int32_t* __restrict__ csr_offsets, const int32_t* __restrict__ csr_indices,
    const int32_t* __restrict__ sources, uint32_t* __restrict__ visited,
    int n_sources, int bitmap_words, int bps) {
    int sid = blockIdx.x / bps, bw = blockIdx.x % bps;
    if (sid >= n_sources) return;
    uint32_t* v = visited + (int64_t)sid * bitmap_words;
    int s = sources[sid];
    int ss = csr_offsets[s], se = csr_offsets[s+1], deg = se-ss;
    int ipb = (deg+bps-1)/bps;
    int ms = ss+bw*ipb, me = ms+ipb; if (me>se) me=se;
    int wid = threadIdx.x/32, lid = threadIdx.x&31, nw = blockDim.x/32;
    for (int i = ms+wid; i < me; i += nw) {
        int n = csr_indices[i];
        int ns = csr_offsets[n], ne = csr_offsets[n+1];
        for (int e = ns+lid; e < ne; e += 32) {
            int m = csr_indices[e];
            atomicOr(&v[m>>5], 1u<<(m&31));
        }
    }
}

__global__ void bfs_expand_general(
    const int32_t* __restrict__ csr_offsets, const int32_t* __restrict__ csr_indices,
    uint32_t* __restrict__ visited, const uint32_t* __restrict__ frontier,
    uint32_t* __restrict__ nf, int ns, int bw, int nv) {
    int sid = blockIdx.x; if (sid >= ns) return;
    uint32_t* v = visited+(int64_t)sid*bw;
    const uint32_t* f = frontier+(int64_t)sid*bw;
    uint32_t* n = nf+(int64_t)sid*bw;
    for (int w = (int)threadIdx.x; w < bw; w += (int)blockDim.x) {
        uint32_t word = f[w];
        while (word) { int bit = __ffs(word)-1; word &= word-1;
            int vv = w*32+bit; if (vv>=nv) break;
            int rs = csr_offsets[vv], re = csr_offsets[vv+1];
            for (int e = rs; e < re; e++) {
                int m = csr_indices[e]; uint32_t mask = 1u<<(m&31);
                uint32_t old = atomicOr(&v[m>>5], mask);
                if (!(old&mask)) atomicOr(&n[m>>5], mask);
            }
        }
    }
}




__global__ void count_ego_edges(
    const int32_t* __restrict__ co, const int32_t* __restrict__ ci,
    const uint32_t* __restrict__ vis, int64_t* __restrict__ ec,
    int ns, int bw, int nv, int bps) {
    int sid = blockIdx.x/bps, blk = blockIdx.x%bps;
    if (sid >= ns) return;
    const uint32_t* v = vis+(int64_t)sid*bw;
    int wpb = (bw+bps-1)/bps, ws = blk*wpb, we = ws+wpb; if(we>bw)we=bw;
    int64_t cnt = 0;
    for (int w = ws+(int)threadIdx.x; w < we; w += (int)blockDim.x) {
        uint32_t word = v[w];
        while (word) { int bit = __ffs(word)-1; word &= word-1;
            int vv = w*32+bit; if (vv>=nv) break;
            int rs = co[vv], re = co[vv+1];
            for (int e = rs; e < re; e++) {
                int d = ci[e]; if (v[d>>5]&(1u<<(d&31))) cnt++;
            }
        }
    }
    typedef cub::BlockReduce<int64_t,256> BR;
    __shared__ typename BR::TempStorage ts;
    int64_t bc = BR(ts).Sum(cnt);
    if (threadIdx.x==0 && bc>0) atomicAdd((unsigned long long*)&ec[sid],(unsigned long long)bc);
}




__global__ void write_ego_edges(
    const int32_t* __restrict__ co, const int32_t* __restrict__ ci,
    const double* __restrict__ cw, const uint32_t* __restrict__ vis,
    int32_t* __restrict__ os, int32_t* __restrict__ od, double* __restrict__ ow,
    const int64_t* __restrict__ eo, int* __restrict__ wc,
    int ns, int bw, int nv, int bps) {
    int sid = blockIdx.x/bps, blk = blockIdx.x%bps;
    if (sid >= ns) return;
    const uint32_t* v = vis+(int64_t)sid*bw;
    int64_t base = eo[sid];
    int wpb = (bw+bps-1)/bps, ws = blk*wpb, we = ws+wpb; if(we>bw)we=bw;

    int my_count = 0;
    for (int w = ws+(int)threadIdx.x; w < we; w += (int)blockDim.x) {
        uint32_t word = v[w];
        while (word) { int bit = __ffs(word)-1; word &= word-1;
            int vv = w*32+bit; if (vv>=nv) break;
            int rs = co[vv], re = co[vv+1];
            for (int e = rs; e < re; e++) {
                int d = ci[e]; if (v[d>>5]&(1u<<(d&31))) my_count++;
            }
        }
    }

    typedef cub::BlockScan<int,256> BS;
    __shared__ typename BS::TempStorage scan_ts;
    int my_offset, block_total;
    BS(scan_ts).ExclusiveSum(my_count, my_offset, block_total);

    __shared__ int block_base;
    if (threadIdx.x == 0 && block_total > 0)
        block_base = atomicAdd(&wc[sid], block_total);
    __syncthreads();

    if (block_total == 0) return;

    int write_pos = block_base + my_offset;
    for (int w = ws+(int)threadIdx.x; w < we; w += (int)blockDim.x) {
        uint32_t word = v[w];
        while (word) { int bit = __ffs(word)-1; word &= word-1;
            int vv = w*32+bit; if (vv>=nv) break;
            int rs = co[vv], re = co[vv+1];
            for (int e = rs; e < re; e++) {
                int d = ci[e];
                if (v[d>>5]&(1u<<(d&31))) {
                    int64_t gp = base + write_pos++;
                    os[gp] = vv; od[gp] = d; ow[gp] = cw[e];
                }
            }
        }
    }
}




__global__ void create_packed_keys_and_iota(
    const int32_t* __restrict__ s, const int32_t* __restrict__ d,
    uint64_t* __restrict__ k, int32_t* __restrict__ v, int64_t n, int vbits) {
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = (int64_t)blockIdx.x*blockDim.x+threadIdx.x; i < n; i += stride) {
        k[i] = ((uint64_t)(uint32_t)s[i]<<vbits)|(uint64_t)(uint32_t)d[i];
        v[i] = (int32_t)i;
    }
}

__global__ void gather_edges(
    const int32_t* __restrict__ is, const int32_t* __restrict__ id,
    const double* __restrict__ iw, const int32_t* __restrict__ perm,
    int32_t* __restrict__ os, int32_t* __restrict__ od,
    double* __restrict__ ow, int64_t n) {
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = (int64_t)blockIdx.x*blockDim.x+threadIdx.x; i < n; i += stride) {
        int32_t p = perm[i];
        os[i] = is[p]; od[i] = id[p]; ow[i] = iw[p];
    }
}

__global__ void convert_offsets(const int64_t* __restrict__ eo,
    int* __restrict__ bo, int* __restrict__ eoo, int ns) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < ns) { bo[i]=(int)eo[i]; eoo[i]=(int)eo[i+1]; }
}

struct EdgeCmp {
    __host__ __device__ bool operator()(
        const thrust::tuple<int32_t,int32_t,double>& a,
        const thrust::tuple<int32_t,int32_t,double>& b) const {
        if (thrust::get<0>(a)!=thrust::get<0>(b)) return thrust::get<0>(a)<thrust::get<0>(b);
        if (thrust::get<1>(a)!=thrust::get<1>(b)) return thrust::get<1>(a)<thrust::get<1>(b);
        return thrust::get<2>(a)<thrust::get<2>(b);
    }
};

}  

extract_ego_weighted_result_double_t extract_ego_weighted_f64_seg(
    const graph32_t& graph,
    const double* edge_weights,
    const int32_t* source_vertices,
    std::size_t n_sources,
    int32_t radius) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t nv = graph.number_of_vertices;
    bool mg = graph.is_multigraph;
    int32_t ns = (int32_t)n_sources;
    int32_t rad = radius;
    const int32_t* co = graph.offsets;
    const int32_t* ci = graph.indices;
    const double* cw = edge_weights;
    const int32_t* src = source_vertices;

    int bw = (nv + 31) / 32;
    cache.ensure(ns, bw);

    size_t bsz = (size_t)ns * bw * sizeof(uint32_t);
    cudaMemsetAsync(cache.d_vis, 0, bsz);

    int bps = ns > 0 ? (1024 + ns - 1) / ns : 1;
    if (bps < 1) bps = 1;
    if (bps > 32) bps = 32;

    if (rad == 1) {
        if (ns > 0) bfs_mark_sources_and_hop1<<<ns, 256>>>(co, ci, src, cache.d_vis, ns, bw);
    } else if (rad == 2) {
        if (ns > 0) bfs_mark_sources_and_hop1<<<ns, 256>>>(co, ci, src, cache.d_vis, ns, bw);
        if (ns > 0) bfs_expand_hop2<<<ns * bps, 256>>>(co, ci, src, cache.d_vis, ns, bw, bps);
    } else {
        cudaMemsetAsync(cache.d_fr, 0, bsz);
        if (ns > 0) bfs_mark_sources_and_hop1<<<ns, 256>>>(co, ci, src, cache.d_vis, ns, bw);
        cudaMemcpyAsync(cache.d_fr, cache.d_vis, bsz, cudaMemcpyDeviceToDevice);
        for (int h = 2; h <= rad; h++) {
            cudaMemsetAsync(cache.d_nf, 0, bsz);
            if (ns > 0) bfs_expand_general<<<ns, 256>>>(co, ci, cache.d_vis, cache.d_fr, cache.d_nf, ns, bw, nv);
            uint32_t* t = cache.d_fr;
            cache.d_fr = cache.d_nf;
            cache.d_nf = t;
        }
    }

    cudaMemsetAsync(cache.d_ec, 0, (ns + 1) * sizeof(int64_t));
    if (ns > 0) count_ego_edges<<<ns * bps, 256>>>(co, ci, cache.d_vis, cache.d_ec, ns, bw, nv, bps);

    size_t pt = cache.ptb;
    cub::DeviceScan::ExclusiveSum(cache.d_pt, pt, cache.d_ec, cache.d_eo, ns + 1);

    std::vector<int64_t> ho(ns + 1);
    cudaMemcpy(ho.data(), cache.d_eo, (ns + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost);
    int64_t te = ho[ns];

    int32_t* out_srcs = nullptr;
    int32_t* out_dsts = nullptr;
    double* out_wts = nullptr;
    std::size_t* out_offsets = nullptr;

    cudaMalloc(&out_srcs, te * sizeof(int32_t));
    cudaMalloc(&out_dsts, te * sizeof(int32_t));
    cudaMalloc(&out_wts, te * sizeof(double));
    cudaMalloc(&out_offsets, (size_t)(ns + 1) * sizeof(std::size_t));

    cudaMemcpyAsync(out_offsets, cache.d_eo, (size_t)(ns + 1) * sizeof(int64_t), cudaMemcpyDeviceToDevice);

    if (te > 0) {
        cudaMemsetAsync(cache.d_wc, 0, ns * sizeof(int));
        if (ns > 0) write_ego_edges<<<ns * bps, 256>>>(co, ci, cw, cache.d_vis,
            out_srcs, out_dsts, out_wts, cache.d_eo, cache.d_wc, ns, bw, nv, bps);

        if (mg) {
            for (int i = 0; i < ns; i++) {
                int64_t a = ho[i], b = ho[i + 1], c = b - a;
                if (c <= 1) continue;
                auto bg = thrust::make_zip_iterator(thrust::make_tuple(
                    thrust::device_pointer_cast(out_srcs + a),
                    thrust::device_pointer_cast(out_dsts + a),
                    thrust::device_pointer_cast(out_wts + a)));
                thrust::sort(thrust::cuda::par, bg, bg + c, EdgeCmp{});
            }
        } else {
            uint64_t* ki = nullptr;
            uint64_t* ko = nullptr;
            int32_t* vi = nullptr;
            int32_t* vo = nullptr;
            int32_t* ts = nullptr;
            int32_t* td = nullptr;
            double* tw = nullptr;
            int* bo = nullptr;
            int* eoo = nullptr;

            cudaMalloc(&ki, te * sizeof(uint64_t));
            cudaMalloc(&ko, te * sizeof(uint64_t));
            cudaMalloc(&vi, te * sizeof(int32_t));
            cudaMalloc(&vo, te * sizeof(int32_t));
            cudaMalloc(&ts, te * sizeof(int32_t));
            cudaMalloc(&td, te * sizeof(int32_t));
            cudaMalloc(&tw, te * sizeof(double));
            cudaMalloc(&bo, ns * sizeof(int));
            cudaMalloc(&eoo, ns * sizeof(int));

            size_t ssz = 0;
            cub::DeviceSegmentedRadixSort::SortPairs(nullptr, ssz,
                (uint64_t*)nullptr, (uint64_t*)nullptr,
                (int32_t*)nullptr, (int32_t*)nullptr,
                (int)te, ns, (int*)nullptr, (int*)nullptr, 0, 64);
            void* srt = nullptr;
            cudaMalloc(&srt, ssz);

            int vbits = 32 - __builtin_clz((unsigned int)nv | 1u);
            int total_bits = 2 * vbits;
            if (total_bits > 64) total_bits = 64;

            int ob = (ns + 255) / 256;
            convert_offsets<<<ob, 256>>>(cache.d_eo, bo, eoo, ns);

            int blocks = (int)(((int64_t)te + 255) / 256);
            if (blocks > 65535) blocks = 65535;
            create_packed_keys_and_iota<<<blocks, 256>>>(out_srcs, out_dsts, ki, vi, te, vbits);

            size_t tb = ssz;
            cub::DeviceSegmentedRadixSort::SortPairs(
                srt, tb, ki, ko, vi, vo, (int)te, ns, bo, eoo, 0, total_bits);

            gather_edges<<<blocks, 256>>>(out_srcs, out_dsts, out_wts, vo, ts, td, tw, te);

            cudaMemcpyAsync(out_srcs, ts, te * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(out_dsts, td, te * sizeof(int32_t), cudaMemcpyDeviceToDevice);
            cudaMemcpyAsync(out_wts, tw, te * sizeof(double), cudaMemcpyDeviceToDevice);

            cudaDeviceSynchronize();

            cudaFree(ki);
            cudaFree(ko);
            cudaFree(vi);
            cudaFree(vo);
            cudaFree(ts);
            cudaFree(td);
            cudaFree(tw);
            cudaFree(bo);
            cudaFree(eoo);
            cudaFree(srt);
        }
    }

    cudaDeviceSynchronize();

    return extract_ego_weighted_result_double_t{
        out_srcs,
        out_dsts,
        out_wts,
        out_offsets,
        (std::size_t)te,
        (std::size_t)(ns + 1)
    };
}

}  
