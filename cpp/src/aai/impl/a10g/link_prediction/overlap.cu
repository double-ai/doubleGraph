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

    Cache() {
        cudaMalloc(&d_is_multigraph, sizeof(bool));
    }

    ~Cache() override {
        if (d_is_multigraph) cudaFree(d_is_multigraph);
    }
};

__global__ void __launch_bounds__(128, 12)
overlap_kernel(
    const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ indices,
    const int32_t* __restrict__ first,
    const int32_t* __restrict__ second,
    float* __restrict__ scores,
    int64_t num_pairs,
    const bool* __restrict__ d_is_multigraph
) {
    const int lane_id = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int warp_id_in_block = threadIdx.x >> 5;
    const int64_t pair_idx = (int64_t)blockIdx.x * warps_per_block + warp_id_in_block;

    if (pair_idx >= num_pairs) return;

    
    bool is_multigraph = *d_is_multigraph;

    int u = first[pair_idx];
    int v = second[pair_idx];

    int u_start = __ldg(&offsets[u]);
    int u_end = __ldg(&offsets[u + 1]);
    int v_start = __ldg(&offsets[v]);
    int v_end = __ldg(&offsets[v + 1]);

    int deg_u = u_end - u_start;
    int deg_v = v_end - v_start;
    int min_deg = min(deg_u, deg_v);

    if (min_deg == 0) {
        if (lane_id == 0) scores[pair_idx] = 0.0f;
        return;
    }

    const int32_t* a = (deg_u <= deg_v) ? (indices + u_start) : (indices + v_start);
    const int32_t* b = (deg_u <= deg_v) ? (indices + v_start) : (indices + u_start);
    int size_a = min(deg_u, deg_v);
    int size_b = max(deg_u, deg_v);

    int count = 0;

    if (!is_multigraph) {
        if (size_b <= 32) {
            
            int my_a = (lane_id < size_a) ? __ldg(&a[lane_id]) : 0x7fffffff;
            int my_b = (lane_id < size_b) ? __ldg(&b[lane_id]) : 0x7fffffff;
            unsigned valid_b_mask = (size_b >= 32) ? 0xffffffffu : ((1u << size_b) - 1u);

            for (int i = 0; i < size_a; i++) {
                int val = __shfl_sync(0xffffffff, my_a, i);
                unsigned mask = __ballot_sync(0xffffffff, my_b == val) & valid_b_mask;
                if (mask) count++;
            }

            if (lane_id == 0) scores[pair_idx] = (float)count / (float)min_deg;
            return;
        }

        if (size_a <= 32 && size_b <= 64) {
            
            int my_a = (lane_id < size_a) ? __ldg(&a[lane_id]) : 0x7fffffff;
            int my_b0 = __ldg(&b[lane_id]);  
            int my_b1 = (32 + lane_id < size_b) ? __ldg(&b[32 + lane_id]) : 0x7fffffff;

            int remaining = size_b - 32;
            unsigned valid_b1 = (remaining >= 32) ? 0xffffffffu : ((1u << remaining) - 1u);

            for (int i = 0; i < size_a; i++) {
                int val = __shfl_sync(0xffffffff, my_a, i);
                unsigned m0 = __ballot_sync(0xffffffff, my_b0 == val);
                unsigned m1 = __ballot_sync(0xffffffff, my_b1 == val) & valid_b1;
                if (m0 | m1) count++;
            }

            if (lane_id == 0) scores[pair_idx] = (float)count / (float)min_deg;
            return;
        }

        
        int search_lo = 0;
        for (int i = lane_id; i < size_a; i += 32) {
            int val = __ldg(&a[i]);
            int lo = search_lo, hi = size_b;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (__ldg(&b[mid]) < val) lo = mid + 1;
                else hi = mid;
            }
            if (lo < size_b && __ldg(&b[lo]) == val) count++;
            search_lo = lo;
        }
    } else {
        
        int search_lo = 0;
        for (int i = lane_id; i < size_a; i += 32) {
            int val = a[i];
            int lo = search_lo, hi = size_b;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (b[mid] < val) lo = mid + 1;
                else hi = mid;
            }
            search_lo = lo;
            if (i > 0 && a[i - 1] == val) continue;
            if (lo < size_b && b[lo] == val) {
                int cnt_a = 1;
                while (i + cnt_a < size_a && a[i + cnt_a] == val) cnt_a++;
                int cnt_b = 1;
                while (lo + cnt_b < size_b && b[lo + cnt_b] == val) cnt_b++;
                count += min(cnt_a, cnt_b);
            }
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        count += __shfl_down_sync(0xffffffff, count, offset);
    }

    if (lane_id == 0) scores[pair_idx] = (float)count / (float)min_deg;
}

void launch_overlap_nosync(
    const int32_t* offsets,
    const int32_t* indices,
    const int32_t* first,
    const int32_t* second,
    float* scores,
    int64_t num_pairs,
    const bool* d_is_multigraph,
    cudaStream_t stream
) {
    if (num_pairs == 0) return;
    const int block_size = 128;
    const int warps_per_block = block_size / 32;
    const int grid_size = (int)((num_pairs + warps_per_block - 1) / warps_per_block);
    overlap_kernel<<<grid_size, block_size, 0, stream>>>(
        offsets, indices, first, second, scores, num_pairs, d_is_multigraph
    );
}

}  

void overlap_similarity(const graph32_t& graph,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        float* similarity_scores) {
    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    bool is_mg = graph.is_multigraph;
    cudaMemcpy(cache.d_is_multigraph, &is_mg, sizeof(bool), cudaMemcpyHostToDevice);

    launch_overlap_nosync(
        graph.offsets,
        graph.indices,
        vertex_pairs_first,
        vertex_pairs_second,
        similarity_scores,
        static_cast<int64_t>(num_pairs),
        cache.d_is_multigraph,
        0
    );
}

}  
