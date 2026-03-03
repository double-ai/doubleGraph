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
#include <cstddef>
#include <limits>

namespace aai {

namespace {




__device__ __forceinline__ void atomicMaxDouble(double* addr, double val) {
    unsigned long long* addr_as_ull = (unsigned long long*)addr;
    unsigned long long old = *addr_as_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        if (__longlong_as_double(assumed) >= val) return;
        old = atomicCAS(addr_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
}

struct MaxOp {
    __device__ __forceinline__ double operator()(const double &a, const double &b) const {
        return (a > b) ? a : b;
    }
};





__global__ void expand_col_ids_edge(
    const int32_t* __restrict__ offsets,
    int32_t* __restrict__ col_ids,
    int32_t N, int32_t E)
{
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= E) return;
    int lo = 0, hi = N;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (offsets[mid + 1] <= edge) lo = mid + 1;
        else hi = mid;
    }
    col_ids[edge] = lo;
}

__global__ void compute_csr_offsets(const int32_t* __restrict__ sorted_src,
    int32_t* __restrict__ csr_off, int32_t E, int32_t N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > E) return;

    if (idx == 0) {
        int fs = (E > 0) ? sorted_src[0] : N;
        for (int i = 0; i <= fs; i++) csr_off[i] = 0;
    }
    if (idx == E) {
        int ls = (E > 0) ? sorted_src[E - 1] : -1;
        for (int i = ls + 1; i <= N; i++) csr_off[i] = E;
    }
    if (idx > 0 && idx < E) {
        int ps = sorted_src[idx - 1], cs = sorted_src[idx];
        if (cs != ps) {
            for (int s = ps + 1; s <= cs; s++) csr_off[s] = idx;
        }
    }
}





__global__ void spmv_thread(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t num_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    int s = offsets[row], e = offsets[row + 1];
    double sum = 0.0;
    for (int k = s; k < e; k++)
        sum += x[indices[k]];
    y[row] = sum;
}

__global__ void spmv_warp(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t num_rows)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp_id >= num_rows) return;
    int start = offsets[warp_id], end = offsets[warp_id + 1];
    double sum = 0.0;
    for (int k = start + lane; k < end; k += 32)
        sum += x[indices[k]];
    for (int s = 16; s > 0; s >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, s);
    if (lane == 0) y[warp_id] = sum;
}

__global__ void spmv_block(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const double* __restrict__ x,
    double* __restrict__ y,
    int32_t row_start, int32_t row_end)
{
    int row = row_start + blockIdx.x;
    if (row >= row_end) return;
    int s = offsets[row], e = offsets[row + 1];
    typedef cub::BlockReduce<double, 256> BR;
    __shared__ typename BR::TempStorage temp;
    double sum = 0.0;
    for (int k = s + threadIdx.x; k < e; k += 256)
        sum += x[indices[k]];
    sum = BR(temp).Sum(sum);
    if (threadIdx.x == 0) y[row] = sum;
}




__global__ void compute_max_norms(const double* __restrict__ a,
    const double* __restrict__ b, int32_t n, double* __restrict__ buf) {
    constexpr int BS = 256;
    typedef cub::BlockReduce<double, BS> BR;
    __shared__ typename BR::TempStorage ta, tb;
    int idx = blockIdx.x * BS + threadIdx.x;
    double ma = 0.0, mb = 0.0;
    if (idx < n) { ma = fabs(a[idx]); mb = fabs(b[idx]); }
    ma = BR(ta).Reduce(ma, MaxOp());
    mb = BR(tb).Reduce(mb, MaxOp());
    if (threadIdx.x == 0) { atomicMaxDouble(buf, ma); atomicMaxDouble(buf+1, mb); }
}

__global__ void normalize_and_diff(double* __restrict__ a, double* __restrict__ b,
    const double* __restrict__ c, int32_t n, double* __restrict__ buf) {
    constexpr int BS = 256;
    typedef cub::BlockReduce<double, BS> BR;
    __shared__ typename BR::TempStorage td;
    int idx = blockIdx.x * BS + threadIdx.x;
    double am = buf[0], bm = buf[1];
    double diff = 0.0;
    if (idx < n) {
        double av = a[idx], bv = b[idx];
        if (am > 0.0) av /= am;
        if (bm > 0.0) bv /= bm;
        a[idx] = av; b[idx] = bv;
        diff = fabs(av - c[idx]);
    }
    diff = BR(td).Sum(diff);
    if (threadIdx.x == 0) atomicAdd(buf+2, diff);
}




__global__ void init_k(double* arr, int32_t n, double val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) arr[idx] = val;
}

__global__ void compute_sum_k(const double* __restrict__ arr, int32_t n, double* __restrict__ out) {
    constexpr int BS = 256;
    typedef cub::BlockReduce<double, BS> BR;
    __shared__ typename BR::TempStorage t;
    int idx = blockIdx.x * BS + threadIdx.x;
    double v = (idx < n) ? arr[idx] : 0.0;
    v = BR(t).Sum(v);
    if (threadIdx.x == 0) atomicAdd(out, v);
}

__global__ void div_scalar_k(double* arr, int32_t n, const double* s) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    double sv = *s;
    if (sv > 0.0) arr[idx] /= sv;
}





size_t get_sort_temp_size(int32_t E) {
    size_t sz = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, sz,
        (int32_t*)nullptr, (int32_t*)nullptr,
        (int32_t*)nullptr, (int32_t*)nullptr, E, 0, 32);
    return sz;
}

void launch_build_csr(
    const int32_t* csc_off, const int32_t* csc_idx,
    int32_t* csr_off, int32_t* csr_idx,
    int32_t* tk_a, int32_t* tk_b, int32_t* tv_a, int32_t* tv_b,
    void* sort_tmp, size_t sort_tmp_sz,
    int32_t N, int32_t E, cudaStream_t stream)
{
    if (E == 0) {
        cudaMemsetAsync(csr_off, 0, (N + 1) * sizeof(int32_t), stream);
        return;
    }

    cudaMemcpyAsync(tk_a, csc_idx, E * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

    expand_col_ids_edge<<<(E+255)/256, 256, 0, stream>>>(csc_off, tv_a, N, E);

    cub::DeviceRadixSort::SortPairs(sort_tmp, sort_tmp_sz,
        tk_a, tk_b, tv_a, tv_b, E, 0, 32, stream);

    cudaMemcpyAsync(csr_idx, tv_b, E * sizeof(int32_t), cudaMemcpyDeviceToDevice, stream);

    compute_csr_offsets<<<(E + 256) / 256, 256, 0, stream>>>(tk_b, csr_off, E, N);
}

void launch_spmv_seg(const int32_t* off, const int32_t* idx,
    const double* x, double* y, int32_t N,
    int32_t s0, int32_t s1, int32_t s2, int32_t s3,
    cudaStream_t stream)
{
    if (N == 0) return;

    if (s0 < s1) {
        spmv_block<<<s1 - s0, 256, 0, stream>>>(off, idx, x, y, s0, s1);
    }
    if (s1 < s2) {
        int n = s2 - s1;
        spmv_warp<<<(n+7)/8, 256, 0, stream>>>(off + s1, idx, x, y + s1, n);
    }
    if (s2 < s3) {
        int n = s3 - s2;
        spmv_thread<<<(n+255)/256, 256, 0, stream>>>(off + s2, idx, x, y + s2, n);
    }
    if (s3 < N) {
        int n = N - s3;
        cudaMemsetAsync(y + s3, 0, n * sizeof(double), stream);
    }
}

void launch_spmv_adaptive(const int32_t* off, const int32_t* idx,
    const double* x, double* y, int32_t n, int32_t e, cudaStream_t s) {
    if (n == 0) return;
    if (n > 0 && e / n < 16) {
        spmv_thread<<<(n+255)/256, 256, 0, s>>>(off, idx, x, y, n);
    } else {
        spmv_warp<<<(n+7)/8, 256, 0, s>>>(off, idx, x, y, n);
    }
}

void launch_max_norms(const double* a, const double* b, int32_t n, double* buf, cudaStream_t s) {
    if (n == 0) return;
    compute_max_norms<<<(n+255)/256, 256, 0, s>>>(a, b, n, buf);
}

void launch_norm_diff(double* a, double* b, const double* c, int32_t n, double* buf, cudaStream_t s) {
    if (n == 0) return;
    normalize_and_diff<<<(n+255)/256, 256, 0, s>>>(a, b, c, n, buf);
}

void launch_init(double* arr, int32_t n, double val, cudaStream_t s) {
    if (n == 0) return;
    init_k<<<(n+255)/256, 256, 0, s>>>(arr, n, val);
}

void launch_l1_norm(double* arr, int32_t n, double* tmp, cudaStream_t s) {
    if (n == 0) return;
    int b = (n+255)/256;
    cudaMemsetAsync(tmp, 0, sizeof(double), s);
    compute_sum_k<<<b, 256, 0, s>>>(arr, n, tmp);
    div_scalar_k<<<b, 256, 0, s>>>(arr, n, tmp);
}





struct Cache : Cacheable {
    
    int32_t* csr_off = nullptr;
    int32_t* csr_idx = nullptr;
    int32_t* tk_a = nullptr;
    int32_t* tk_b = nullptr;
    int32_t* tv_a = nullptr;
    int32_t* tv_b = nullptr;
    void* sort_tmp = nullptr;

    
    double* hubs_buf = nullptr;
    double* auth_buf = nullptr;
    double* nhubs_buf = nullptr;
    double* rbuf = nullptr;  

    
    int64_t csr_off_cap = 0;
    int64_t csr_idx_cap = 0;
    int64_t tk_a_cap = 0;
    int64_t tk_b_cap = 0;
    int64_t tv_a_cap = 0;
    int64_t tv_b_cap = 0;
    size_t sort_tmp_cap = 0;
    int64_t hubs_cap = 0;
    int64_t auth_cap = 0;
    int64_t nhubs_cap = 0;
    int64_t rbuf_cap = 0;

    void ensure(int32_t N, int32_t E, size_t sort_sz) {
        int64_t Np1 = (int64_t)N + 1;
        int64_t Ei = (E > 0) ? (int64_t)E : 1;

        if (csr_off_cap < Np1) {
            if (csr_off) cudaFree(csr_off);
            cudaMalloc(&csr_off, Np1 * sizeof(int32_t));
            csr_off_cap = Np1;
        }
        if (csr_idx_cap < Ei) {
            if (csr_idx) cudaFree(csr_idx);
            cudaMalloc(&csr_idx, Ei * sizeof(int32_t));
            csr_idx_cap = Ei;
        }
        if (tk_a_cap < Ei) {
            if (tk_a) cudaFree(tk_a);
            cudaMalloc(&tk_a, Ei * sizeof(int32_t));
            tk_a_cap = Ei;
        }
        if (tk_b_cap < Ei) {
            if (tk_b) cudaFree(tk_b);
            cudaMalloc(&tk_b, Ei * sizeof(int32_t));
            tk_b_cap = Ei;
        }
        if (tv_a_cap < Ei) {
            if (tv_a) cudaFree(tv_a);
            cudaMalloc(&tv_a, Ei * sizeof(int32_t));
            tv_a_cap = Ei;
        }
        if (tv_b_cap < Ei) {
            if (tv_b) cudaFree(tv_b);
            cudaMalloc(&tv_b, Ei * sizeof(int32_t));
            tv_b_cap = Ei;
        }
        if (sort_tmp_cap < sort_sz) {
            if (sort_tmp) cudaFree(sort_tmp);
            cudaMalloc(&sort_tmp, sort_sz);
            sort_tmp_cap = sort_sz;
        }
        if (hubs_cap < (int64_t)N) {
            if (hubs_buf) cudaFree(hubs_buf);
            cudaMalloc(&hubs_buf, N * sizeof(double));
            hubs_cap = N;
        }
        if (auth_cap < (int64_t)N) {
            if (auth_buf) cudaFree(auth_buf);
            cudaMalloc(&auth_buf, N * sizeof(double));
            auth_cap = N;
        }
        if (nhubs_cap < (int64_t)N) {
            if (nhubs_buf) cudaFree(nhubs_buf);
            cudaMalloc(&nhubs_buf, N * sizeof(double));
            nhubs_cap = N;
        }
        if (rbuf_cap < 4) {
            if (rbuf) cudaFree(rbuf);
            cudaMalloc(&rbuf, 4 * sizeof(double));
            rbuf_cap = 4;
        }
    }

    ~Cache() override {
        if (csr_off) cudaFree(csr_off);
        if (csr_idx) cudaFree(csr_idx);
        if (tk_a) cudaFree(tk_a);
        if (tk_b) cudaFree(tk_b);
        if (tv_a) cudaFree(tv_a);
        if (tv_b) cudaFree(tv_b);
        if (sort_tmp) cudaFree(sort_tmp);
        if (hubs_buf) cudaFree(hubs_buf);
        if (auth_buf) cudaFree(auth_buf);
        if (nhubs_buf) cudaFree(nhubs_buf);
        if (rbuf) cudaFree(rbuf);
    }
};

}  

HitsResultDouble hits_seg(const graph32_t& graph,
                          double* hubs,
                          double* authorities,
                          double epsilon,
                          std::size_t max_iterations,
                          bool has_initial_hubs_guess,
                          bool normalize) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int32_t N = graph.number_of_vertices;
    int32_t E = graph.number_of_edges;

    if (N == 0) {
        return HitsResultDouble{max_iterations, false, std::numeric_limits<double>::max()};
    }

    const int32_t* d_csc_off = graph.offsets;
    const int32_t* d_csc_idx = graph.indices;

    const auto& seg = graph.segment_offsets.value();
    int32_t s0 = seg[0], s1 = seg[1], s2 = seg[2], s3 = seg[3];

    cudaStream_t stream = 0;

    
    size_t sort_sz = get_sort_temp_size(E);
    cache.ensure(N, E, sort_sz);

    
    launch_build_csr(d_csc_off, d_csc_idx,
        cache.csr_off, cache.csr_idx,
        cache.tk_a, cache.tk_b, cache.tv_a, cache.tv_b,
        cache.sort_tmp, sort_sz, N, E, stream);

    const int32_t* d_csr_off = cache.csr_off;
    const int32_t* d_csr_idx = cache.csr_idx;

    double* d_hubs = cache.hubs_buf;
    double* d_auth = cache.auth_buf;
    double* d_nhubs = cache.nhubs_buf;
    double* d_rbuf = cache.rbuf;

    double tol = (double)N * epsilon;

    if (has_initial_hubs_guess) {
        cudaMemcpyAsync(d_hubs, hubs,
            N * sizeof(double), cudaMemcpyDeviceToDevice, stream);
        launch_l1_norm(d_hubs, N, d_rbuf + 3, stream);
    } else {
        launch_init(d_hubs, N, 1.0 / N, stream);
    }

    double* prev = d_hubs;
    double* curr = d_nhubs;
    double diff = 0.0;
    std::size_t iter = 0;
    bool converged = false;

    while (iter < max_iterations) {
        
        launch_spmv_seg(d_csc_off, d_csc_idx, prev, d_auth, N, s0, s1, s2, s3, stream);

        
        launch_spmv_adaptive(d_csr_off, d_csr_idx, d_auth, curr, N, E, stream);

        
        cudaMemsetAsync(d_rbuf, 0, 3 * sizeof(double), stream);
        launch_max_norms(curr, d_auth, N, d_rbuf, stream);
        launch_norm_diff(curr, d_auth, prev, N, d_rbuf, stream);

        cudaMemcpyAsync(&diff, d_rbuf + 2, sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::swap(prev, curr);
        iter++;
        if (diff < tol) { converged = true; break; }
    }

    if (normalize) {
        launch_l1_norm(prev, N, d_rbuf + 3, stream);
        launch_l1_norm(d_auth, N, d_rbuf + 3, stream);
    }

    
    cudaMemcpyAsync(hubs, prev, N * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(authorities, d_auth, N * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    return HitsResultDouble{iter, converged, diff};
}

}  
