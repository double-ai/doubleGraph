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

struct Cache : Cacheable {
    bool* d_is_multigraph = nullptr;
    bool cached_value = false;
    bool initialized = false;

    void ensure(bool is_multigraph) {
        if (!d_is_multigraph) {
            cudaMalloc(&d_is_multigraph, sizeof(bool));
        }
        if (!initialized || cached_value != is_multigraph) {
            cudaMemcpy(d_is_multigraph, &is_multigraph, sizeof(bool), cudaMemcpyHostToDevice);
            cached_value = is_multigraph;
            initialized = true;
        }
    }

    ~Cache() override {
        if (d_is_multigraph) cudaFree(d_is_multigraph);
    }
};

__global__ void __launch_bounds__(128)
sorensen_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ pairs_first,
    const int32_t* __restrict__ pairs_second,
    float* __restrict__ scores,
    const int num_pairs,
    const bool* __restrict__ d_is_multigraph
) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp_id >= num_pairs) return;

    const bool is_multigraph = *d_is_multigraph;

    const int u = pairs_first[warp_id];
    const int v = pairs_second[warp_id];

    const int u_start = offsets[u];
    const int u_end = offsets[u + 1];
    const int v_start = offsets[v];
    const int v_end = offsets[v + 1];

    const int u_deg = u_end - u_start;
    const int v_deg = v_end - v_start;
    const int denom = u_deg + v_deg;

    if (denom == 0) {
        if (lane == 0) scores[warp_id] = 0.0f;
        return;
    }

    const int32_t* A, *B;
    int m, n;
    if (u_deg <= v_deg) {
        A = indices + u_start; m = u_deg;
        B = indices + v_start; n = v_deg;
    } else {
        A = indices + v_start; m = v_deg;
        B = indices + u_start; n = u_deg;
    }

    int count = 0;

    if (m == 0) {
        
    } else if (!is_multigraph && m <= 32 && n <= 32) {
        
        int a_val = (lane < m) ? A[lane] : 0x7fffffff;
        int b_val = (lane < n) ? B[lane] : 0x7ffffffe;

        int found = 0;
        int in_range = (lane < n);
        for (int i = 0; i < m; i++) {
            int a = __shfl_sync(0xffffffff, a_val, i);
            found |= (in_range & (b_val == a));
        }
        unsigned match_mask = __ballot_sync(0xffffffff, found);
        if (lane == 0) count = __popc(match_mask);

    } else if (!is_multigraph && n <= m * 12) {
        
        
        int total = m + n;
        int per_thread = (total + 31) >> 5;
        int diag_start = lane * per_thread;
        int diag_end = diag_start + per_thread;
        if (diag_end > total) diag_end = total;

        if (diag_start < total) {
            int lo = (diag_start > n) ? (diag_start - n) : 0;
            int hi = (diag_start < m) ? diag_start : m;

            while (lo < hi) {
                int mid = (lo + hi + 1) >> 1;
                if (A[mid - 1] < B[diag_start - mid]) lo = mid;
                else hi = mid - 1;
            }

            int i = lo, j = diag_start - lo;
            for (int k = diag_start; k < diag_end; k++) {
                if (i >= m) { j++; }
                else if (j >= n) { i++; }
                else if (A[i] < B[j]) { i++; }
                else { if (A[i] == B[j]) count++; j++; }
            }
        }
    } else if (is_multigraph) {
        
        int b_hint = 0;
        for (int i = lane; i < m; i += 32) {
            int val = A[i];
            int lo = b_hint, hi = n;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (B[mid] < val) lo = mid + 1;
                else hi = mid;
            }
            int b_lb = lo;
            b_hint = lo;

            if (b_lb < n && B[b_lb] == val) {
                int lo2 = b_lb + 1, hi2 = n;
                while (lo2 < hi2) {
                    int mid = (lo2 + hi2) >> 1;
                    if (B[mid] <= val) lo2 = mid + 1;
                    else hi2 = mid;
                }
                int b_count = lo2 - b_lb;
                int lo3 = 0, hi3 = i;
                while (lo3 < hi3) {
                    int mid = (lo3 + hi3) >> 1;
                    if (A[mid] < val) lo3 = mid + 1;
                    else hi3 = mid;
                }
                int rank = i - lo3;
                if (rank < b_count) count++;
            }
        }
    } else {
        
        int b_lo = 0;
        for (int i = lane; i < m; i += 32) {
            int val = A[i];
            int lo = b_lo, hi = n;
            while (lo < hi) {
                int mid = (lo + hi) >> 1;
                if (B[mid] < val) lo = mid + 1;
                else hi = mid;
            }
            if (lo < n && B[lo] == val) count++;
            b_lo = lo;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        count += __shfl_down_sync(0xffffffff, count, offset);
    }

    if (lane == 0) {
        scores[warp_id] = 2.0f * (float)count / (float)denom;
    }
}

}  

void sorensen_similarity_seg(const graph32_t& graph,
                             const int32_t* vertex_pairs_first,
                             const int32_t* vertex_pairs_second,
                             std::size_t num_pairs,
                             float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    if (num_pairs == 0) return;

    cache.ensure(graph.is_multigraph);

    const int grid = (static_cast<int>(num_pairs) + 3) / 4;
    sorensen_kernel<<<grid, 128>>>(
        graph.offsets,
        graph.indices,
        vertex_pairs_first,
        vertex_pairs_second,
        similarity_scores,
        static_cast<int>(num_pairs),
        cache.d_is_multigraph
    );
}

}  
