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
#include <cstddef>
#include <vector>
#include <cub/block/block_reduce.cuh>

namespace aai {

namespace {

static constexpr int BLK = 256;
static constexpr int WARP_SZ = 32;
static constexpr int EDGES_PER_BLOCK = 4096;

struct Cache : Cacheable {
    float* h_pinned = nullptr;
    float* buf = nullptr;
    float* scratch = nullptr;
    int32_t* work_v = nullptr;
    int32_t* work_s = nullptr;
    int32_t* work_e = nullptr;

    bool pinned_allocated = false;
    bool scratch_allocated = false;
    int64_t buf_capacity = 0;
    int64_t work_capacity = 0;

    void ensure_pinned() {
        if (!pinned_allocated) {
            cudaMallocHost(&h_pinned, sizeof(float));
            pinned_allocated = true;
        }
    }

    void ensure_buf(int64_t n) {
        if (buf_capacity < n) {
            if (buf) cudaFree(buf);
            cudaMalloc(&buf, n * sizeof(float));
            buf_capacity = n;
        }
    }

    void ensure_scratch() {
        if (!scratch_allocated) {
            cudaMalloc(&scratch, 2 * sizeof(float));
            scratch_allocated = true;
        }
    }

    void ensure_work(int64_t n) {
        if (work_capacity < n) {
            if (work_v) cudaFree(work_v);
            if (work_s) cudaFree(work_s);
            if (work_e) cudaFree(work_e);
            cudaMalloc(&work_v, n * sizeof(int32_t));
            cudaMalloc(&work_s, n * sizeof(int32_t));
            cudaMalloc(&work_e, n * sizeof(int32_t));
            work_capacity = n;
        }
    }

    ~Cache() override {
        if (h_pinned) cudaFreeHost(h_pinned);
        if (buf) cudaFree(buf);
        if (scratch) cudaFree(scratch);
        if (work_v) cudaFree(work_v);
        if (work_s) cudaFree(work_s);
        if (work_e) cudaFree(work_e);
    }
};





__global__ void spmv_high_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    const int32_t* __restrict__ work_vertex,
    const int32_t* __restrict__ work_edge_start,
    const int32_t* __restrict__ work_edge_end,
    int32_t num_work_items)
{
    int wid = blockIdx.x;
    if (wid >= num_work_items) return;

    int v = work_vertex[wid];
    int es = work_edge_start[wid];
    int ee = work_edge_end[wid];

    float sum = 0.0f;
    for (int i = es + threadIdx.x; i < ee; i += BLK) {
        uint32_t w = __ldg(&edge_mask[i >> 5]);
        if ((w >> (i & 31)) & 1u)
            sum += __ldg(&x_old[__ldg(&indices[i])]);
    }

    typedef cub::BlockReduce<float, BLK> BR;
    __shared__ typename BR::TempStorage temp;
    float block_sum = BR(temp).Sum(sum);

    if (threadIdx.x == 0 && block_sum != 0.0f) {
        float id_term = (es == offsets[v]) ? __ldg(&x_old[v]) : 0.0f;
        atomicAdd(&x_new[v], block_sum + id_term);
    }
}


__global__ void high_norm_kernel(
    const float* __restrict__ x_new,
    float* __restrict__ g_norm_sq,
    int32_t v_start, int32_t v_end)
{
    typedef cub::BlockReduce<float, BLK> BR;
    __shared__ typename BR::TempStorage temp;

    int v = v_start + blockIdx.x * BLK + threadIdx.x;
    float val = 0.0f;
    if (v < v_end) {
        val = x_new[v];
    }
    float bns = BR(temp).Sum(val * val);
    if (threadIdx.x == 0 && bns > 0.0f)
        atomicAdd(g_norm_sq, bns);
}





__global__ void spmv_mid_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float* __restrict__ g_norm_sq,
    int32_t v_start, int32_t v_end)
{
    const int warps_per_block = BLK / WARP_SZ;
    int warp_id = blockIdx.x * warps_per_block + (threadIdx.x / WARP_SZ);
    int lane = threadIdx.x & (WARP_SZ - 1);
    int v = v_start + warp_id;

    typedef cub::BlockReduce<float, BLK> BR;
    __shared__ typename BR::TempStorage temp;

    float val = 0.0f;
    if (v < v_end) {
        int s = offsets[v];
        int e = offsets[v + 1];
        float sum = 0.0f;

        for (int i = s + lane; i < e; i += WARP_SZ) {
            uint32_t w = __ldg(&edge_mask[i >> 5]);
            if ((w >> (i & 31)) & 1u)
                sum += __ldg(&x_old[__ldg(&indices[i])]);
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);

        if (lane == 0) {
            float total = sum + __ldg(&x_old[v]);
            x_new[v] = total;
            val = total;
        }
    }

    float bns = BR(temp).Sum(val * val);
    if (threadIdx.x == 0 && bns > 0.0f)
        atomicAdd(g_norm_sq, bns);
}





__global__ void spmv_low_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const uint32_t* __restrict__ edge_mask,
    const float* __restrict__ x_old,
    float* __restrict__ x_new,
    float* __restrict__ g_norm_sq,
    int32_t v_start, int32_t v_end)
{
    typedef cub::BlockReduce<float, BLK> BR;
    __shared__ typename BR::TempStorage temp;

    int v = v_start + blockIdx.x * BLK + threadIdx.x;
    float val = 0.0f;

    if (v < v_end) {
        int s = offsets[v];
        int e = offsets[v + 1];
        float sum = __ldg(&x_old[v]);

        for (int i = s; i < e; ++i) {
            uint32_t w = __ldg(&edge_mask[i >> 5]);
            if ((w >> (i & 31)) & 1u)
                sum += __ldg(&x_old[__ldg(&indices[i])]);
        }
        x_new[v] = sum;
        val = sum;
    }

    float bns = BR(temp).Sum(val * val);
    if (threadIdx.x == 0 && bns > 0.0f)
        atomicAdd(g_norm_sq, bns);
}




__global__ void normalize_diff_kernel(
    float* __restrict__ x_new,
    const float* __restrict__ x_old,
    const float* __restrict__ g_norm_sq,
    float* __restrict__ g_diff,
    int32_t num_vertices)
{
    typedef cub::BlockReduce<float, BLK> BR;
    __shared__ typename BR::TempStorage temp;

    float inv_norm = (*g_norm_sq > 0.0f) ? rsqrtf(*g_norm_sq) : 0.0f;
    float d = 0.0f;

    int num_vec4 = num_vertices >> 2;
    for (int i = blockIdx.x * BLK + threadIdx.x; i < num_vec4; i += gridDim.x * BLK) {
        float4 xn4 = reinterpret_cast<float4*>(x_new)[i];
        float4 xo4 = reinterpret_cast<const float4*>(x_old)[i];
        xn4.x *= inv_norm; xn4.y *= inv_norm; xn4.z *= inv_norm; xn4.w *= inv_norm;
        d += fabsf(xn4.x - xo4.x) + fabsf(xn4.y - xo4.y) +
             fabsf(xn4.z - xo4.z) + fabsf(xn4.w - xo4.w);
        reinterpret_cast<float4*>(x_new)[i] = xn4;
    }

    if (blockIdx.x == 0) {
        for (int i = num_vec4 * 4 + threadIdx.x; i < num_vertices; i += BLK) {
            float nv = x_new[i] * inv_norm;
            x_new[i] = nv;
            d += fabsf(nv - x_old[i]);
        }
    }

    float bd = BR(temp).Sum(d);
    if (threadIdx.x == 0 && bd > 0.0f)
        atomicAdd(g_diff, bd);
}




__global__ void init_kernel(float* x, int32_t n, float val) {
    int i = blockIdx.x * BLK + threadIdx.x;
    if (i < n) x[i] = val;
}





void launch_spmv_high(
    const int32_t* off, const int32_t* idx, const uint32_t* mask,
    const float* xo, float* xn,
    const int32_t* wv, const int32_t* wes, const int32_t* wee,
    int32_t n_work, cudaStream_t s)
{
    if (n_work > 0) {
        spmv_high_kernel<<<n_work, BLK, 0, s>>>(off, idx, mask, xo, xn, wv, wes, wee, n_work);
    }
}

void launch_high_norm(const float* xn, float* nsq, int32_t v_start, int32_t v_end, cudaStream_t s)
{
    int n = v_end - v_start;
    if (n > 0) {
        int grid = (n + BLK - 1) / BLK;
        high_norm_kernel<<<grid, BLK, 0, s>>>(xn, nsq, v_start, v_end);
    }
}

void launch_spmv_mid(
    const int32_t* off, const int32_t* idx, const uint32_t* mask,
    const float* xo, float* xn, float* nsq,
    int32_t v_start, int32_t v_end, cudaStream_t s)
{
    int n = v_end - v_start;
    if (n > 0) {
        int warps_per_block = BLK / WARP_SZ;
        int grid = (n + warps_per_block - 1) / warps_per_block;
        spmv_mid_kernel<<<grid, BLK, 0, s>>>(off, idx, mask, xo, xn, nsq, v_start, v_end);
    }
}

void launch_spmv_low(
    const int32_t* off, const int32_t* idx, const uint32_t* mask,
    const float* xo, float* xn, float* nsq,
    int32_t v_start, int32_t v_end, cudaStream_t s)
{
    int n = v_end - v_start;
    if (n > 0) {
        int grid = (n + BLK - 1) / BLK;
        spmv_low_kernel<<<grid, BLK, 0, s>>>(off, idx, mask, xo, xn, nsq, v_start, v_end);
    }
}

void launch_normalize_diff(
    float* xn, const float* xo, const float* nsq, float* diff,
    int32_t nv, cudaStream_t s)
{
    cudaMemsetAsync(diff, 0, sizeof(float), s);
    int grid = ((nv >> 2) + BLK - 1) / BLK;
    if (grid < 1) grid = 1;
    if (grid > 2048) grid = 2048;
    normalize_diff_kernel<<<grid, BLK, 0, s>>>(xn, xo, nsq, diff, nv);
}

void launch_init(float* x, int32_t n, float val, cudaStream_t s) {
    int grid = (n + BLK - 1) / BLK;
    init_kernel<<<grid, BLK, 0, s>>>(x, n, val);
}

}  

eigenvector_centrality_result_t eigenvector_centrality_seg_mask(const graph32_t& graph,
                                      float* centralities,
                                      float epsilon,
                                      std::size_t max_iterations,
                                      const float* initial_centralities) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_offsets = graph.offsets;
    const int32_t* d_indices = graph.indices;
    const uint32_t* d_mask = graph.edge_mask;
    int32_t N = graph.number_of_vertices;
    cudaStream_t stream = 0;

    const auto& seg = graph.segment_offsets.value();
    int32_t seg0 = seg[0], seg1 = seg[1], seg2 = seg[2], seg3 = seg[3], seg4 = seg[4];

    cache.ensure_pinned();
    cache.ensure_buf(N);
    cache.ensure_scratch();

    float* x1 = centralities;
    float* x2 = cache.buf;
    float* d_norm = cache.scratch;
    float* d_diff = d_norm + 1;

    if (initial_centralities != nullptr)
        cudaMemcpyAsync(x1, initial_centralities, N * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    else
        launch_init(x1, N, 1.0f / N, stream);

    
    int n_high = seg1 - seg0;
    int32_t n_work = 0;

    if (n_high > 0) {
        std::vector<int32_t> h_off(n_high + 1);
        cudaMemcpy(h_off.data(), d_offsets + seg0, (n_high + 1) * sizeof(int32_t), cudaMemcpyDeviceToHost);

        std::vector<int32_t> wv, ws, we;
        wv.reserve(n_high * 4);
        ws.reserve(n_high * 4);
        we.reserve(n_high * 4);

        for (int i = 0; i < n_high; i++) {
            int s = h_off[i], e = h_off[i + 1];
            if (s == e) {
                
                wv.push_back(seg0 + i); ws.push_back(s); we.push_back(e);
            } else {
                for (int b = s; b < e; b += EDGES_PER_BLOCK) {
                    wv.push_back(seg0 + i);
                    ws.push_back(b);
                    we.push_back(b + EDGES_PER_BLOCK < e ? b + EDGES_PER_BLOCK : e);
                }
            }
        }
        n_work = (int32_t)wv.size();

        cache.ensure_work(n_work);
        cudaMemcpyAsync(cache.work_v, wv.data(), n_work * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(cache.work_s, ws.data(), n_work * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(cache.work_e, we.data(), n_work * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    }

    float threshold = (float)N * epsilon;
    std::size_t iter = 0;
    bool converged = false;
    float *xo = x1, *xn = x2;

    while (iter < max_iterations) {
        
        cudaMemsetAsync(d_norm, 0, sizeof(float), stream);

        
        if (n_high > 0) {
            cudaMemsetAsync(xn + seg0, 0, n_high * sizeof(float), stream);
        }

        
        if (n_work > 0) {
            launch_spmv_high(d_offsets, d_indices, d_mask, xo, xn,
                            cache.work_v, cache.work_s, cache.work_e,
                            n_work, stream);
            
            launch_high_norm(xn, d_norm, seg0, seg1, stream);
        }

        
        launch_spmv_mid(d_offsets, d_indices, d_mask, xo, xn, d_norm, seg1, seg2, stream);
        
        launch_spmv_low(d_offsets, d_indices, d_mask, xo, xn, d_norm, seg2, seg4, stream);

        
        launch_normalize_diff(xn, xo, d_norm, d_diff, N, stream);

        ++iter;

        cudaMemcpyAsync(cache.h_pinned, d_diff, sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        if (*cache.h_pinned < threshold) { converged = true; break; }

        float* tmp = xo; xo = xn; xn = tmp;
    }

    float* result = converged ? xn : xo;
    if (result != x1)
        cudaMemcpyAsync(x1, result, N * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    return {iter, converged};
}

}  
