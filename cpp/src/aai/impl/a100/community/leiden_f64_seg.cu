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
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/functional.h>

namespace aai {

namespace {

#define BLOCK_SIZE 128
#define HM_SIZE 32
#define HM_MASK (HM_SIZE - 1)



__global__ void iota_kernel(int* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = i;
}

__global__ void zero_double_kernel(double* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = 0.0;
}

__global__ void compare_arrays_kernel(const int* __restrict__ a, const int* __restrict__ b, int n, int* diff_flag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && a[i] != b[i]) {
        *diff_flag = 1;
    }
}

__global__ void reduce_sum_kernel(const double* input, double* output, int n) {
    typedef cub::BlockReduce<double, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    double sum = 0.0;
    for (int i = threadIdx.x + blockIdx.x * 256; i < n; i += 256 * gridDim.x)
        sum += input[i];
    double block_sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) atomicAdd(output, block_sum);
}

__global__ void compute_vertex_weights_kernel(
    const int* __restrict__ offsets, const double* __restrict__ edge_weights,
    double* __restrict__ vertex_weights, int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    double w = 0.0;
    for (int e = offsets[v]; e < offsets[v + 1]; e++) w += edge_weights[e];
    vertex_weights[v] = w;
}



__global__ void local_moving_kernel(
    const int* __restrict__ offsets, const int* __restrict__ indices,
    const double* __restrict__ edge_weights,
    int* __restrict__ community, double* __restrict__ community_weight,
    const double* __restrict__ vertex_weights,
    double total_edge_weight, double resolution, int num_vertices,
    int* __restrict__ moved_flag, int color, int num_colors)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices || v % num_colors != color) return;

    int my_comm = community[v];
    double k_v = vertex_weights[v];
    int start = offsets[v], end = offsets[v + 1];
    if (start == end || k_v == 0.0) return;

    int hm_keys[HM_SIZE];
    double hm_vals[HM_SIZE];
    #pragma unroll
    for (int i = 0; i < HM_SIZE; i++) { hm_keys[i] = -1; hm_vals[i] = 0.0; }

    double w_to_own = 0.0;
    for (int e = start; e < end; e++) {
        int u = indices[e];
        if (u == v) continue;
        int c = community[u];
        double w = edge_weights[e];
        if (c == my_comm) { w_to_own += w; }
        else {
            unsigned int h = ((unsigned int)c * 2654435761u) & HM_MASK;
            for (int probe = 0; probe < HM_SIZE; probe++) {
                int slot = (h + probe) & HM_MASK;
                if (hm_keys[slot] == c) { hm_vals[slot] += w; break; }
                if (hm_keys[slot] == -1) { hm_keys[slot] = c; hm_vals[slot] = w; break; }
            }
        }
    }

    double inv_tw = 1.0 / total_edge_weight;
    double coeff1 = 2.0 * inv_tw;
    double coeff2 = 2.0 * resolution * k_v * inv_tw * inv_tw;
    double sigma_own = community_weight[my_comm];
    double best_gain = 0.0;
    int best_comm = my_comm;

    for (int i = 0; i < HM_SIZE; i++) {
        if (hm_keys[i] != -1) {
            double gain = coeff1 * (hm_vals[i] - w_to_own) + coeff2 * (sigma_own - k_v - community_weight[hm_keys[i]]);
            if (gain > best_gain) { best_gain = gain; best_comm = hm_keys[i]; }
        }
    }

    if (best_comm != my_comm) {
        community[v] = best_comm;
        atomicAdd(&community_weight[my_comm], -k_v);
        atomicAdd(&community_weight[best_comm], k_v);
        *moved_flag = 1;
    }
}



__global__ void leiden_refine_kernel(
    const int* __restrict__ offsets, const int* __restrict__ indices,
    const double* __restrict__ edge_weights,
    const int* __restrict__ flat_community,
    int* __restrict__ refined,
    double* __restrict__ refined_weight,
    const double* __restrict__ vertex_weights,
    double total_edge_weight, double resolution, double theta,
    int num_vertices, int* __restrict__ moved_flag,
    int color, int num_colors)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices || v % num_colors != color) return;

    int my_flat = flat_community[v];
    int my_ref = refined[v];
    double k_v = vertex_weights[v];
    int start = offsets[v], end = offsets[v + 1];
    if (start == end || k_v == 0.0) return;

    int hm_keys[HM_SIZE];
    double hm_vals[HM_SIZE];
    #pragma unroll
    for (int i = 0; i < HM_SIZE; i++) { hm_keys[i] = -1; hm_vals[i] = 0.0; }

    double w_to_own = 0.0;
    for (int e = start; e < end; e++) {
        int u = indices[e];
        if (u == v) continue;
        if (flat_community[u] != my_flat) continue;
        int r = refined[u];
        double w = edge_weights[e];
        if (r == my_ref) { w_to_own += w; }
        else {
            unsigned int h = ((unsigned int)r * 2654435761u) & HM_MASK;
            for (int probe = 0; probe < HM_SIZE; probe++) {
                int slot = (h + probe) & HM_MASK;
                if (hm_keys[slot] == r) { hm_vals[slot] += w; break; }
                if (hm_keys[slot] == -1) { hm_keys[slot] = r; hm_vals[slot] = w; break; }
            }
        }
    }

    double inv_tw = 1.0 / total_edge_weight;
    double coeff1 = 2.0 * inv_tw;
    double coeff2 = 2.0 * resolution * k_v * inv_tw * inv_tw;
    double sigma_own = refined_weight[my_ref];
    double best_gain = 0.0;
    int best_ref = my_ref;

    for (int i = 0; i < HM_SIZE; i++) {
        if (hm_keys[i] != -1) {
            double gain = coeff1 * (hm_vals[i] - w_to_own) + coeff2 * (sigma_own - k_v - refined_weight[hm_keys[i]]);
            if (gain > best_gain && gain * theta > 1e-6) {
                best_gain = gain;
                best_ref = hm_keys[i];
            }
        }
    }

    if (best_ref != my_ref) {
        refined[v] = best_ref;
        atomicAdd(&refined_weight[my_ref], -k_v);
        atomicAdd(&refined_weight[best_ref], k_v);
        *moved_flag = 1;
    }
}



__global__ void cc_hook_kernel(
    const int* __restrict__ offsets, const int* __restrict__ indices,
    const int* __restrict__ community, int* __restrict__ parent,
    int num_vertices, int* __restrict__ changed) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int my_comm = community[v];
    for (int e = offsets[v]; e < offsets[v + 1]; e++) {
        int u = indices[e];
        if (u == v || community[u] != my_comm) continue;
        int rv = v; while (parent[rv] != rv) rv = parent[rv];
        int ru = u; while (parent[ru] != ru) ru = parent[ru];
        if (rv != ru) {
            int rmin = (rv < ru) ? rv : ru, rmax = (rv < ru) ? ru : rv;
            if (atomicMin(&parent[rmax], rmin) != rmin) *changed = 1;
        }
    }
}

__global__ void cc_compress_kernel(int* parent, int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int p = parent[v]; while (parent[p] != p) p = parent[p]; parent[v] = p;
}

__global__ void apply_cc_labels_kernel(int* __restrict__ dst, const int* __restrict__ parent, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) dst[v] = parent[v];
}

__global__ void recompute_cw_kernel(const int* __restrict__ community,
    const double* __restrict__ vw, double* __restrict__ cw, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) atomicAdd(&cw[community[v]], vw[v]);
}

__global__ void compute_internal_weight_kernel(
    const int* __restrict__ offsets, const int* __restrict__ indices,
    const double* __restrict__ edge_weights, const int* __restrict__ partition,
    double* __restrict__ vertex_internal, int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int cv = partition[v];
    double sum = 0.0;
    for (int e = offsets[v]; e < offsets[v + 1]; e++) {
        if (partition[indices[e]] == cv) sum += edge_weights[e];
    }
    vertex_internal[v] = sum;
}

__global__ void square_array_kernel(const double* __restrict__ in, double* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * in[i];
}



__global__ void map_edges_to_communities(const int* __restrict__ offsets, const int* __restrict__ indices,
    const int* __restrict__ community, long long* __restrict__ edge_keys, int nv, int nc) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv) return;
    int cv = community[v];
    for (int e = offsets[v]; e < offsets[v + 1]; e++)
        edge_keys[e] = (long long)cv * nc + community[indices[e]];
}

__global__ void compose_partitions_kernel(int* __restrict__ p, const int* __restrict__ cp, int n) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < n) p[v] = cp[p[v]];
}

__global__ void count_edges_per_vertex(const long long* __restrict__ keys, int* __restrict__ counts, int ne, int nc) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < ne) atomicAdd(&counts[(int)(keys[e] / nc)], 1);
}

__global__ void unpack_keys(const long long* __restrict__ keys, int* __restrict__ out, int ne, int nc) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < ne) out[e] = (int)(keys[e] % nc);
}

__global__ void build_coarse_init_kernel(
    const int* __restrict__ flat_comm,
    const int* __restrict__ refined_comm,
    int* __restrict__ coarse_init,
    int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        coarse_init[refined_comm[v]] = flat_comm[v];
    }
}



void launch_iota(int* a, int n) { iota_kernel<<<(n+255)/256, 256>>>(a, n); }
void launch_zero_double(double* a, int n) { zero_double_kernel<<<(n+255)/256, 256>>>(a, n); }
void launch_compare(const int* a, const int* b, int n, int* d) { compare_arrays_kernel<<<(n+255)/256, 256>>>(a, b, n, d); }
void launch_reduce_sum(const double* in, double* out, int n) { if(n<=0)return; int b = (n+255)/256; if(b>256)b=256; reduce_sum_kernel<<<b,256>>>(in,out,n); }
void launch_compute_vw(const int* off, const double* w, double* vw, int n) { compute_vertex_weights_kernel<<<(n+255)/256, 256>>>(off, w, vw, n); }

void launch_local_moving(const int* off, const int* idx, const double* w,
    int* comm, double* cw, const double* vw, double tw, double res, int n, int* moved, int c, int nc) {
    local_moving_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(off, idx, w, comm, cw, vw, tw, res, n, moved, c, nc);
}

void launch_leiden_refine(const int* off, const int* idx, const double* w,
    const int* flat, int* refined, double* rw, const double* vw,
    double tw, double res, double theta, int n, int* moved, int c, int nc) {
    leiden_refine_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(off, idx, w, flat, refined, rw, vw, tw, res, theta, n, moved, c, nc);
}

void launch_cc_hook(const int* off, const int* idx, const int* comm, int* par, int n, int* ch) {
    cc_hook_kernel<<<(n+BLOCK_SIZE-1)/BLOCK_SIZE, BLOCK_SIZE>>>(off, idx, comm, par, n, ch);
}
void launch_cc_compress(int* par, int n) { cc_compress_kernel<<<(n+255)/256, 256>>>(par, n); }
void launch_apply_cc(int* dst, const int* par, int n) { apply_cc_labels_kernel<<<(n+255)/256, 256>>>(dst, par, n); }
void launch_recompute_cw(const int* comm, const double* vw, double* cw, int n) {
    recompute_cw_kernel<<<(n+255)/256, 256>>>(comm, vw, cw, n);
}

void launch_compute_internal_weight(const int* off, const int* idx, const double* w,
    const int* part, double* out, int n) {
    compute_internal_weight_kernel<<<(n+255)/256, 256>>>(off, idx, w, part, out, n);
}

void launch_square_array(const double* in, double* out, int n) {
    square_array_kernel<<<(n+255)/256, 256>>>(in, out, n);
}

int launch_relabel(int* community, int* temp_buf, int num_vertices) {
    thrust::device_ptr<int> comm_ptr(community);
    thrust::device_ptr<int> temp_ptr(temp_buf);
    thrust::copy(thrust::device, comm_ptr, comm_ptr + num_vertices, temp_ptr);
    thrust::sort(thrust::device, temp_ptr, temp_ptr + num_vertices);
    int K = (int)(thrust::unique(thrust::device, temp_ptr, temp_ptr + num_vertices) - temp_ptr);
    thrust::device_ptr<int> new_ptr(temp_buf + num_vertices);
    thrust::lower_bound(thrust::device, temp_ptr, temp_ptr + K, comm_ptr, comm_ptr + num_vertices, new_ptr);
    thrust::copy(thrust::device, new_ptr, new_ptr + num_vertices, comm_ptr);
    return K;
}

int launch_coarsen_graph(const int* offsets, const int* indices, const double* weights,
    const int* community, int nv, int ne, int nc,
    int* new_off, int* new_idx, double* new_w,
    long long* kb, double* vb, int* cb) {
    map_edges_to_communities<<<(nv+255)/256, 256>>>(offsets, indices, community, kb, nv, nc);
    cudaMemcpy(vb, weights, ne * sizeof(double), cudaMemcpyDeviceToDevice);
    thrust::device_ptr<long long> kp(kb); thrust::device_ptr<double> vp(vb);
    thrust::sort_by_key(thrust::device, kp, kp + ne, vp);
    thrust::device_ptr<long long> okp(kb + ne); thrust::device_ptr<double> ovp(vb + ne);
    auto pair = thrust::reduce_by_key(thrust::device, kp, kp + ne, vp, okp, ovp);
    int new_ne = (int)(pair.first - okp);
    cudaMemset(cb, 0, (nc + 1) * sizeof(int));
    if (new_ne > 0) {
        count_edges_per_vertex<<<(new_ne+255)/256, 256>>>(kb + ne, cb, new_ne, nc);
        thrust::device_ptr<int> cp(cb);
        thrust::exclusive_scan(thrust::device, cp, cp + nc + 1, thrust::device_ptr<int>(new_off));
        unpack_keys<<<(new_ne+255)/256, 256>>>(kb + ne, new_idx, new_ne, nc);
    } else {
        cudaMemset(new_off, 0, (nc + 1) * sizeof(int));
    }
    cudaMemcpy(new_w, vb + ne, new_ne * sizeof(double), cudaMemcpyDeviceToDevice);
    return new_ne;
}

void launch_compose(int* p, const int* cp, int n) { compose_partitions_kernel<<<(n+255)/256, 256>>>(p, cp, n); }
void launch_build_coarse_init(const int* flat, const int* ref, int* out, int n) {
    build_coarse_init_kernel<<<(n+255)/256, 256>>>(flat, ref, out, n);
}



struct Cache : Cacheable {
    int* flag = nullptr;
    double* tw = nullptr;

    double* vw = nullptr;    int64_t vw_cap = 0;
    double* cw = nullptr;    int64_t cw_cap = 0;
    int* par = nullptr;      int64_t par_cap = 0;
    int* comm = nullptr;     int64_t comm_cap = 0;
    int* flat = nullptr;     int64_t flat_cap = 0;
    int* ref = nullptr;      int64_t ref_cap = 0;
    double* rw = nullptr;    int64_t rw_cap = 0;
    int* tmp = nullptr;      int64_t tmp_cap = 0;
    long long* kb = nullptr; int64_t kb_cap = 0;
    double* vb = nullptr;    int64_t vb_cap = 0;
    int* cnt = nullptr;      int64_t cnt_cap = 0;
    int* c_init = nullptr;   int64_t c_init_cap = 0;

    void ensure_flag() {
        if (!flag) cudaMalloc(&flag, sizeof(int));
    }
    void ensure_tw() {
        if (!tw) cudaMalloc(&tw, sizeof(double));
    }
    void ensure_vw(int64_t n) {
        if (vw_cap < n) { if (vw) cudaFree(vw); cudaMalloc(&vw, n * sizeof(double)); vw_cap = n; }
    }
    void ensure_cw(int64_t n) {
        if (cw_cap < n) { if (cw) cudaFree(cw); cudaMalloc(&cw, n * sizeof(double)); cw_cap = n; }
    }
    void ensure_par(int64_t n) {
        if (par_cap < n) { if (par) cudaFree(par); cudaMalloc(&par, n * sizeof(int)); par_cap = n; }
    }
    void ensure_comm(int64_t n) {
        if (comm_cap < n) { if (comm) cudaFree(comm); cudaMalloc(&comm, n * sizeof(int)); comm_cap = n; }
    }
    void ensure_flat(int64_t n) {
        if (flat_cap < n) { if (flat) cudaFree(flat); cudaMalloc(&flat, n * sizeof(int)); flat_cap = n; }
    }
    void ensure_ref(int64_t n) {
        if (ref_cap < n) { if (ref) cudaFree(ref); cudaMalloc(&ref, n * sizeof(int)); ref_cap = n; }
    }
    void ensure_rw(int64_t n) {
        if (rw_cap < n) { if (rw) cudaFree(rw); cudaMalloc(&rw, n * sizeof(double)); rw_cap = n; }
    }
    void ensure_tmp(int64_t n) {
        if (tmp_cap < n) { if (tmp) cudaFree(tmp); cudaMalloc(&tmp, n * sizeof(int)); tmp_cap = n; }
    }
    void ensure_kb(int64_t n) {
        if (kb_cap < n) { if (kb) cudaFree(kb); cudaMalloc(&kb, n * sizeof(long long)); kb_cap = n; }
    }
    void ensure_vb(int64_t n) {
        if (vb_cap < n) { if (vb) cudaFree(vb); cudaMalloc(&vb, n * sizeof(double)); vb_cap = n; }
    }
    void ensure_cnt(int64_t n) {
        if (cnt_cap < n) { if (cnt) cudaFree(cnt); cudaMalloc(&cnt, n * sizeof(int)); cnt_cap = n; }
    }
    void ensure_c_init(int64_t n) {
        if (c_init_cap < n) { if (c_init) cudaFree(c_init); cudaMalloc(&c_init, n * sizeof(int)); c_init_cap = n; }
    }

    ~Cache() override {
        if (flag) cudaFree(flag);
        if (tw) cudaFree(tw);
        if (vw) cudaFree(vw);
        if (cw) cudaFree(cw);
        if (par) cudaFree(par);
        if (comm) cudaFree(comm);
        if (flat) cudaFree(flat);
        if (ref) cudaFree(ref);
        if (rw) cudaFree(rw);
        if (tmp) cudaFree(tmp);
        if (kb) cudaFree(kb);
        if (vb) cudaFree(vb);
        if (cnt) cudaFree(cnt);
        if (c_init) cudaFree(c_init);
    }
};



int pick_nc(int nv) { return (nv < 256) ? 4 : 2; }

void do_local_moving(const int* off, const int* idx, const double* w,
    int* comm, double* cw, const double* vw,
    double tw, double res, int nv, int* flag, int mi) {
    int nc = pick_nc(nv);
    for (int iter = 0; iter < mi; ) {
        int batch = (mi - iter < 5) ? (mi - iter) : 5;
        cudaMemset(flag, 0, 4);
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < nc; c++)
                launch_local_moving(off, idx, w, comm, cw, vw, tw, res, nv, flag, c, nc);
        int m; cudaMemcpy(&m, flag, 4, cudaMemcpyDeviceToHost);
        iter += batch;
        if (!m) break;
    }
}

void do_leiden_refine(const int* off, const int* idx, const double* w,
    const int* flat, int* refined, double* rw, const double* vw,
    double tw, double res, double theta, int nv, int* flag, int mi) {
    int nc = pick_nc(nv);
    for (int iter = 0; iter < mi; ) {
        int batch = (mi - iter < 3) ? (mi - iter) : 3;
        cudaMemset(flag, 0, 4);
        for (int b = 0; b < batch; b++)
            for (int c = 0; c < nc; c++)
                launch_leiden_refine(off, idx, w, flat, refined, rw, vw, tw, res, theta, nv, flag, c, nc);
        int m; cudaMemcpy(&m, flag, 4, cudaMemcpyDeviceToHost);
        iter += batch;
        if (!m) break;
    }
}

void do_cc(const int* off, const int* idx, const int* comm,
           int* par, int nv, int* flag) {
    launch_iota(par, nv);
    for (int i = 0; i < 20; ) {
        int batch = (20 - i < 4) ? (20 - i) : 4;
        cudaMemset(flag, 0, 4);
        for (int b = 0; b < batch; b++) {
            launch_cc_hook(off, idx, comm, par, nv, flag);
            launch_cc_compress(par, nv);
        }
        int ch; cudaMemcpy(&ch, flag, 4, cudaMemcpyDeviceToHost);
        i += batch;
        if (!ch) break;
    }
}

}  

leiden_result_double_t leiden_seg(const graph32_t& graph,
                                  const double* edge_weights,
                                  int32_t* clusters,
                                  std::size_t max_level,
                                  double resolution,
                                  double theta) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    int nv = graph.number_of_vertices;
    int ne = graph.number_of_edges;
    const int* d_off = graph.offsets;
    const int* d_idx = graph.indices;
    const double* d_w = edge_weights;

    int* d_part = clusters;
    launch_iota(d_part, nv);

    
    cache.ensure_flag();
    cache.ensure_tw();
    cache.ensure_vw(nv);
    cache.ensure_cw(nv);
    cache.ensure_par(nv);
    cache.ensure_comm(nv);
    cache.ensure_flat(nv);
    cache.ensure_ref(nv);
    cache.ensure_rw(nv);
    cache.ensure_tmp((int64_t)nv * 2);

    double* d_vw = cache.vw;
    double* d_cw = cache.cw;
    int* d_flag = cache.flag;
    int* d_par = cache.par;
    int* d_tmp = cache.tmp;
    int* d_comm = cache.comm;
    int* d_flat = cache.flat;
    int* d_ref = cache.ref;
    double* d_rw = cache.rw;

    
    cudaMemset(cache.tw, 0, sizeof(double));
    launch_reduce_sum(d_w, cache.tw, ne);
    double total_weight;
    cudaMemcpy(&total_weight, cache.tw, sizeof(double), cudaMemcpyDeviceToHost);
    if (total_weight <= 0.0 || nv <= 1) return {0, 0.0};

    
    cache.ensure_kb((int64_t)ne * 2);
    cache.ensure_vb((int64_t)ne * 2);

    const int MI = 15;

    const int* cur_off = d_off;
    const int* cur_idx = d_idx;
    const double* cur_w = d_w;
    int cur_V = nv, cur_E = ne;

    
    int* c_off = nullptr;
    int* c_idx = nullptr;
    double* c_w = nullptr;

    std::size_t level_count = 0;

    for (size_t level = 0; level < max_level; level++) {
        level_count++;

        
        launch_iota(d_comm, cur_V);

        
        if (cur_V <= nv) {
            launch_compute_vw(cur_off, cur_w, d_vw, cur_V);
            cudaMemcpy(d_cw, d_vw, cur_V * sizeof(double), cudaMemcpyDeviceToDevice);
        } else {
            cache.ensure_vw(cur_V);
            d_vw = cache.vw;
            launch_compute_vw(cur_off, cur_w, d_vw, cur_V);
            cache.ensure_cw(cur_V);
            d_cw = cache.cw;
            cudaMemcpy(d_cw, d_vw, cur_V * sizeof(double), cudaMemcpyDeviceToDevice);
        }

        
        do_local_moving(cur_off, cur_idx, cur_w, d_comm, d_cw, d_vw,
            total_weight, resolution, cur_V, d_flag, MI);

        
        if (cur_V <= nv) {
            cudaMemcpy(d_flat, d_comm, cur_V * sizeof(int), cudaMemcpyDeviceToDevice);
        } else {
            cache.ensure_flat(cur_V);
            d_flat = cache.flat;
            cudaMemcpy(d_flat, d_comm, cur_V * sizeof(int), cudaMemcpyDeviceToDevice);
        }

        
        if (cur_V > nv) {
            cache.ensure_par(cur_V);
            d_par = cache.par;
            cache.ensure_ref(cur_V);
            d_ref = cache.ref;
            cache.ensure_rw(cur_V);
            d_rw = cache.rw;
        }

        do_cc(cur_off, cur_idx, d_flat, d_par, cur_V, d_flag);
        launch_apply_cc(d_ref, d_par, cur_V);

        
        if (cur_V <= nv) {
            cudaMemset(d_rw, 0, cur_V * sizeof(double));
        } else {
            launch_zero_double(d_rw, cur_V);
        }
        launch_recompute_cw(d_ref, d_vw, d_rw, cur_V);

        
        do_leiden_refine(cur_off, cur_idx, cur_w, d_flat, d_ref, d_rw, d_vw,
            total_weight, resolution, theta, cur_V, d_flag, 5);

        
        if ((int64_t)cur_V * 2 > cache.tmp_cap) {
            cache.ensure_tmp((int64_t)cur_V * 2);
            d_tmp = cache.tmp;
        }
        int K_flat = launch_relabel(d_flat, d_tmp, cur_V);
        int K_ref = launch_relabel(d_ref, d_tmp, cur_V);

        if (K_ref >= cur_V) break; 

        
        launch_compose(d_part, d_ref, nv);

        
        cache.ensure_c_init(K_ref);
        int* d_cinit = cache.c_init;
        launch_build_coarse_init(d_flat, d_ref, d_cinit, cur_V);
        int K_cinit = launch_relabel(d_cinit, d_tmp, K_ref);
        (void)K_cinit;

        
        
        int* old_c_off = c_off;
        int* old_c_idx = c_idx;
        double* old_c_w = c_w;

        cudaMalloc(&c_off, ((int64_t)K_ref + 1) * sizeof(int));
        cudaMalloc(&c_idx, (int64_t)cur_E * sizeof(int));
        cudaMalloc(&c_w, (int64_t)cur_E * sizeof(double));

        cache.ensure_cnt((int64_t)K_ref + 1);

        
        if ((int64_t)cur_E * 2 > cache.kb_cap) {
            cache.ensure_kb((int64_t)cur_E * 2);
            cache.ensure_vb((int64_t)cur_E * 2);
        }

        int new_E = launch_coarsen_graph(cur_off, cur_idx, cur_w, d_ref,
            cur_V, cur_E, K_ref,
            c_off, c_idx, c_w,
            cache.kb, cache.vb, cache.cnt);

        
        if (old_c_off) cudaFree(old_c_off);
        if (old_c_idx) cudaFree(old_c_idx);
        if (old_c_w) cudaFree(old_c_w);

        cur_off = c_off;
        cur_idx = c_idx;
        cur_w = c_w;
        cur_V = K_ref;
        cur_E = new_E;

        
        if ((int64_t)cur_V > cache.comm_cap) {
            cache.ensure_comm(cur_V);
        }
        d_comm = cache.comm;
        cudaMemcpy(d_comm, d_cinit, cur_V * sizeof(int), cudaMemcpyDeviceToDevice);

        if (cur_V <= 1) break;
    }

    
    if (c_off) cudaFree(c_off);
    if (c_idx) cudaFree(c_idx);
    if (c_w) cudaFree(c_w);

    
    if ((int64_t)nv > cache.par_cap) {
        cache.ensure_par(nv);
    }
    d_par = cache.par;
    do_cc(d_off, d_idx, d_part, d_par, nv, d_flag);
    launch_apply_cc(d_part, d_par, nv);

    
    cache.ensure_vw(nv);
    cache.ensure_cw(nv);
    d_vw = cache.vw;
    d_cw = cache.cw;
    launch_compute_vw(d_off, d_w, d_vw, nv);
    cudaMemset(d_cw, 0, nv * sizeof(double));
    launch_recompute_cw(d_part, d_vw, d_cw, nv);

    do_local_moving(d_off, d_idx, d_w, d_part, d_cw, d_vw,
        total_weight, resolution, nv, d_flag, MI);

    
    do_cc(d_off, d_idx, d_part, d_par, nv, d_flag);
    launch_apply_cc(d_part, d_par, nv);

    
    d_tmp = cache.tmp;
    launch_relabel(d_part, d_tmp, nv);

    
    
    {
        
        cudaMemset(d_cw, 0, nv * sizeof(double));
        launch_recompute_cw(d_part, d_vw, d_cw, nv);

        
        launch_square_array(d_cw, d_vw, nv);

        
        cudaMemset(cache.tw, 0, sizeof(double));
        launch_reduce_sum(d_vw, cache.tw, nv);
        double sum_deg_sq;
        cudaMemcpy(&sum_deg_sq, cache.tw, sizeof(double), cudaMemcpyDeviceToHost);

        
        launch_compute_internal_weight(d_off, d_idx, d_w, d_part, d_vw, nv);

        
        cudaMemset(cache.tw, 0, sizeof(double));
        launch_reduce_sum(d_vw, cache.tw, nv);
        double sum_internal;
        cudaMemcpy(&sum_internal, cache.tw, sizeof(double), cudaMemcpyDeviceToHost);

        double W = total_weight;
        double modularity = sum_internal / W - resolution * sum_deg_sq / (W * W);
        return {level_count, modularity};
    }
}

}  
