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

namespace aai {

namespace {

struct Cache : Cacheable {
    
    int32_t* adeg = nullptr;
    int32_t* noff = nullptr;
    float* ows = nullptr;
    uint32_t* dmask = nullptr;
    float* pra = nullptr;
    float* prb = nullptr;
    int32_t vert_cap = 0;

    
    int32_t* cidx = nullptr;
    float* cwt = nullptr;
    int64_t edge_cap = 0;

    
    void* cub_temp = nullptr;
    size_t cub_temp_cap = 0;

    
    float* ds = nullptr;
    float* l1 = nullptr;
    float* bv = nullptr;
    bool scalars_init = false;

    
    float* h_l1 = nullptr;
    bool host_init = false;

    void ensure_vert(int32_t nv) {
        if (vert_cap < nv) {
            if (adeg) cudaFree(adeg);
            if (noff) cudaFree(noff);
            if (ows) cudaFree(ows);
            if (dmask) cudaFree(dmask);
            if (pra) cudaFree(pra);
            if (prb) cudaFree(prb);
            cudaMalloc(&adeg, (size_t)nv * sizeof(int32_t));
            cudaMalloc(&noff, ((size_t)nv + 1) * sizeof(int32_t));
            cudaMalloc(&ows, (size_t)nv * sizeof(float));
            int32_t mw = (nv + 31) / 32;
            cudaMalloc(&dmask, (size_t)mw * sizeof(uint32_t));
            cudaMalloc(&pra, (size_t)nv * sizeof(float));
            cudaMalloc(&prb, (size_t)nv * sizeof(float));
            vert_cap = nv;
        }
    }

    void ensure_edge(int64_t ce) {
        if (edge_cap < ce) {
            if (cidx) cudaFree(cidx);
            if (cwt) cudaFree(cwt);
            cudaMalloc(&cidx, (size_t)ce * sizeof(int32_t));
            cudaMalloc(&cwt, (size_t)ce * sizeof(float));
            edge_cap = ce;
        }
    }

    void ensure_cub(size_t ts) {
        size_t need = ts > 0 ? ts : 1;
        if (cub_temp_cap < need) {
            if (cub_temp) cudaFree(cub_temp);
            cudaMalloc(&cub_temp, need);
            cub_temp_cap = need;
        }
    }

    void ensure_scalars() {
        if (!scalars_init) {
            cudaMalloc(&ds, sizeof(float));
            cudaMalloc(&l1, sizeof(float));
            cudaMalloc(&bv, sizeof(float));
            scalars_init = true;
        }
    }

    void ensure_host() {
        if (!host_init) {
            cudaMallocHost(&h_l1, sizeof(float));
            host_init = true;
        }
    }

    ~Cache() override {
        if (adeg) cudaFree(adeg);
        if (noff) cudaFree(noff);
        if (ows) cudaFree(ows);
        if (dmask) cudaFree(dmask);
        if (pra) cudaFree(pra);
        if (prb) cudaFree(prb);
        if (cidx) cudaFree(cidx);
        if (cwt) cudaFree(cwt);
        if (cub_temp) cudaFree(cub_temp);
        if (ds) cudaFree(ds);
        if (l1) cudaFree(l1);
        if (bv) cudaFree(bv);
        if (h_l1) cudaFreeHost(h_l1);
    }
};



__global__ void count_active_edges_kernel(
    const int32_t* __restrict__ offsets,
    const uint32_t* __restrict__ edge_mask,
    int32_t* __restrict__ active_degree,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int start = offsets[v];
    int end = offsets[v + 1];
    int count = 0;
    for (int e = start; e < end; e++)
        count += (edge_mask[e >> 5] >> (e & 31)) & 1u;
    active_degree[v] = count;
}

__global__ void compact_edges_kernel(
    const int32_t* __restrict__ old_offsets,
    const int32_t* __restrict__ old_indices,
    const float* __restrict__ old_weights,
    const uint32_t* __restrict__ edge_mask,
    const int32_t* __restrict__ new_offsets,
    int32_t* __restrict__ new_indices,
    float* __restrict__ new_weights,
    int32_t num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int old_start = old_offsets[v];
    int old_end = old_offsets[v + 1];
    int new_pos = new_offsets[v];
    for (int e = old_start; e < old_end; e++) {
        if ((edge_mask[e >> 5] >> (e & 31)) & 1u) {
            new_indices[new_pos] = old_indices[e];
            new_weights[new_pos] = old_weights[e];
            new_pos++;
        }
    }
}



__global__ void compute_out_weight_sums_kernel(
    const int32_t* __restrict__ ci, const float* __restrict__ cw,
    float* __restrict__ ows, int32_t nce)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nce) return;
    atomicAdd(&ows[ci[e]], cw[e]);
}

__global__ void prescale_weights_kernel(
    const int32_t* __restrict__ ci, float* __restrict__ cw,
    const float* __restrict__ ows, int32_t nce)
{
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nce) return;
    float o = ows[ci[e]];
    cw[e] = (o > 0.0f) ? (cw[e] / o) : 0.0f;
}



__global__ void compute_dangling_bitmask_kernel(
    const float* __restrict__ ows, uint32_t* __restrict__ mask, int32_t nv)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    bool is_d = (v < nv) && (ows[v] == 0.0f);
    unsigned int ballot = __ballot_sync(0xffffffff, is_d);
    if ((threadIdx.x & 31) == 0) {
        int word = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        mask[word] = ballot;
    }
}



__global__ void init_pr_kernel(float* __restrict__ pr, float val, int32_t n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pr[i] = val;
}



__global__ void dangling_sum_kernel(
    const uint32_t* __restrict__ dmask, const float* __restrict__ pr,
    float* __restrict__ ds, int32_t nv)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;
    if (v < nv && ((dmask[v >> 5] >> (v & 31)) & 1u))
        val = pr[v];
    float s = BlockReduce(temp).Sum(val);
    if (threadIdx.x == 0 && s != 0.0f)
        atomicAdd(ds, s);
}



__global__ void compute_base_val_kernel(
    const float* __restrict__ ds, float* __restrict__ bv, float alpha, float inv_n)
{
    *bv = (1.0f - alpha) * inv_n + alpha * (*ds) * inv_n;
}



__global__ void spmv_warp_kernel(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    const float* __restrict__ wt, const float* __restrict__ pr_old,
    float* __restrict__ pr_new, float* __restrict__ l1,
    const float* __restrict__ bv_ptr, float alpha,
    int32_t vs, int32_t ve)
{
    constexpr int WPB = 8;
    int lw = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int v = vs + blockIdx.x * WPB + lw;

    float diff = 0.0f;
    if (v < ve) {
        int s = off[v], e = off[v + 1];
        float sum = 0.0f;
        for (int i = s + lane; i < e; i += 32)
            sum += wt[i] * pr_old[idx[i]];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, o);
        if (lane == 0) {
            float nv = *bv_ptr + alpha * sum;
            pr_new[v] = nv;
            diff = fabsf(nv - pr_old[v]);
        }
    }

    
    __shared__ float sl[WPB];
    if (lane == 0) sl[lw] = diff;
    __syncthreads();

    
    float rval = (threadIdx.x < WPB) ? sl[threadIdx.x] : 0.0f;
    if (threadIdx.x < 32) {
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1)
            rval += __shfl_down_sync(0xffffffff, rval, o);
        if (threadIdx.x == 0 && rval != 0.0f)
            atomicAdd(l1, rval);
    }
}



__global__ void spmv_thread_kernel(
    const int32_t* __restrict__ off, const int32_t* __restrict__ idx,
    const float* __restrict__ wt, const float* __restrict__ pr_old,
    float* __restrict__ pr_new, float* __restrict__ l1,
    const float* __restrict__ bv_ptr, float alpha,
    int32_t vs, int32_t ve)
{
    typedef cub::BlockReduce<float, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    int v = vs + blockIdx.x * blockDim.x + threadIdx.x;
    float diff = 0.0f;
    if (v < ve) {
        int s = off[v], e = off[v + 1];
        float sum = 0.0f;
        for (int i = s; i < e; i++)
            sum += wt[i] * pr_old[idx[i]];
        float nv = *bv_ptr + alpha * sum;
        pr_new[v] = nv;
        diff = fabsf(nv - pr_old[v]);
    }
    float bs = BlockReduce(temp).Sum(diff);
    if (threadIdx.x == 0 && bs != 0.0f)
        atomicAdd(l1, bs);
}

}  

PageRankResult pagerank_seg_mask(const graph32_t& graph,
                                 const float* edge_weights,
                                 float* pageranks,
                                 const float* precomputed_vertex_out_weight_sums,
                                 float alpha,
                                 float epsilon,
                                 std::size_t max_iterations,
                                 const float* initial_pageranks) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int32_t nv = graph.number_of_vertices;
    const uint32_t* d_emask = graph.edge_mask;
    const auto& seg = graph.segment_offsets.value();
    cudaStream_t stream = 0;

    cache.ensure_vert(nv);
    cache.ensure_scalars();
    cache.ensure_host();

    
    {
        int b = 256, g = (nv + b - 1) / b;
        count_active_edges_kernel<<<g, b, 0, stream>>>(d_off, d_emask, cache.adeg, nv);
    }

    cudaMemsetAsync(cache.noff, 0, sizeof(int32_t), stream);
    size_t ts = 0;
    cub::DeviceScan::InclusiveSum(nullptr, ts, (int32_t*)nullptr, (int32_t*)nullptr, nv);
    cache.ensure_cub(ts);
    cub::DeviceScan::InclusiveSum(cache.cub_temp, ts, cache.adeg, cache.noff + 1, nv, stream);

    int32_t nce;
    cudaMemcpyAsync(&nce, cache.noff + nv, sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    int64_t ce = nce > 0 ? (int64_t)nce : 1;
    cache.ensure_edge(ce);

    {
        int b = 256, g = (nv + b - 1) / b;
        compact_edges_kernel<<<g, b, 0, stream>>>(d_off, d_idx, edge_weights, d_emask,
            cache.noff, cache.cidx, cache.cwt, nv);
    }

    
    cudaMemsetAsync(cache.ows, 0, (size_t)nv * sizeof(float), stream);
    if (nce > 0) {
        int b = 256, g = (nce + b - 1) / b;
        compute_out_weight_sums_kernel<<<g, b, 0, stream>>>(cache.cidx, cache.cwt, cache.ows, nce);
    }
    if (nce > 0) {
        int b = 256, g = (nce + b - 1) / b;
        prescale_weights_kernel<<<g, b, 0, stream>>>(cache.cidx, cache.cwt, cache.ows, nce);
    }

    
    {
        int b = 256, g = (nv + b - 1) / b;
        compute_dangling_bitmask_kernel<<<g, b, 0, stream>>>(cache.ows, cache.dmask, nv);
    }

    
    float inv_n = 1.0f / nv;
    if (initial_pageranks) {
        cudaMemcpyAsync(cache.pra, initial_pageranks, (size_t)nv * sizeof(float),
            cudaMemcpyDeviceToDevice, stream);
    } else {
        int b = 256, g = (nv + b - 1) / b;
        init_pr_kernel<<<g, b, 0, stream>>>(cache.pra, inv_n, nv);
    }

    
    float* pr_old = cache.pra;
    float* pr_new = cache.prb;
    std::size_t iteration = 0;
    bool converged = false;

    for (iteration = 0; iteration < max_iterations; iteration++) {
        cudaMemsetAsync(cache.ds, 0, sizeof(float), stream);
        cudaMemsetAsync(cache.l1, 0, sizeof(float), stream);

        {
            int b = 256, g = (nv + b - 1) / b;
            dangling_sum_kernel<<<g, b, 0, stream>>>(cache.dmask, pr_old, cache.ds, nv);
        }

        compute_base_val_kernel<<<1, 1, 0, stream>>>(cache.ds, cache.bv, alpha, inv_n);

        {
            int n = seg[2] - seg[0];
            if (n > 0) {
                int wpb = 8, g = (n + wpb - 1) / wpb;
                spmv_warp_kernel<<<g, 256, 0, stream>>>(cache.noff, cache.cidx, cache.cwt,
                    pr_old, pr_new, cache.l1, cache.bv, alpha, seg[0], seg[2]);
            }
        }

        {
            int n = seg[4] - seg[2];
            if (n > 0) {
                int b = 256, g = (n + b - 1) / b;
                spmv_thread_kernel<<<g, b, 0, stream>>>(cache.noff, cache.cidx, cache.cwt,
                    pr_old, pr_new, cache.l1, cache.bv, alpha, seg[2], seg[4]);
            }
        }

        cudaMemcpyAsync(cache.h_l1, cache.l1, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        float* tmp = pr_old; pr_old = pr_new; pr_new = tmp;
        if (*cache.h_l1 < epsilon) { converged = true; iteration++; break; }
    }

    
    cudaMemcpyAsync(pageranks, pr_old, (size_t)nv * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    return {iteration, converged};
}

}  
