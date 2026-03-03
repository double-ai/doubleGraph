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

struct Slot { int key; int val; };

struct Cache : Cacheable {
    
    int32_t* seeds = nullptr;
    int64_t seeds_cap = 0;

    
    int32_t* pc = nullptr;
    int64_t pc_cap = 0;

    int32_t* sd = nullptr;
    int64_t sd_cap = 0;

    int32_t* caps_buf = nullptr;
    int64_t caps_buf_cap = 0;

    int64_t* hmo = nullptr;
    int64_t hmo_cap = 0;

    int32_t* wo = nullptr;
    int64_t wo_cap = 0;

    int32_t* uc = nullptr;
    int64_t uc_cap = 0;

    int64_t* oo = nullptr;
    int64_t oo_cap = 0;

    
    Slot* hm = nullptr;
    int64_t hm_cap = 0;

    int32_t* ul = nullptr;
    int64_t ul_cap = 0;

    
    int32_t* iota_buf = nullptr;
    int64_t iota_cap = 0;

    float* ssc = nullptr;
    int64_t ssc_cap = 0;

    int32_t* si = nullptr;
    int64_t si_cap = 0;

    uint8_t* sort_tmp = nullptr;
    size_t sort_tmp_cap = 0;

    template <typename T>
    static void ensure(T*& ptr, int64_t& cap, int64_t needed) {
        if (cap < needed) {
            if (ptr) cudaFree(ptr);
            cudaMalloc(&ptr, needed * sizeof(T));
            cap = needed;
        }
    }

    ~Cache() override {
        if (seeds) cudaFree(seeds);
        if (pc) cudaFree(pc);
        if (sd) cudaFree(sd);
        if (caps_buf) cudaFree(caps_buf);
        if (hmo) cudaFree(hmo);
        if (wo) cudaFree(wo);
        if (uc) cudaFree(uc);
        if (oo) cudaFree(oo);
        if (hm) cudaFree(hm);
        if (ul) cudaFree(ul);
        if (iota_buf) cudaFree(iota_buf);
        if (ssc) cudaFree(ssc);
        if (si) cudaFree(si);
        if (sort_tmp) cudaFree(sort_tmp);
    }
};





__device__ __forceinline__ uint32_t hash_fn(int key, int cap_mask) {
    return ((uint32_t)key * 2654435761u) & (uint32_t)cap_mask;
}

__device__ __forceinline__ void hash_insert_tracked(
    Slot* table, int cap_mask, int key,
    int* unique_list, int* unique_count
) {
    uint32_t slot = hash_fn(key, cap_mask);
    while (true) {
        int prev = atomicCAS(&table[slot].key, -1, key);
        if (prev == -1) {
            atomicAdd(&table[slot].val, 1);
            int pos = atomicAdd(unique_count, 1);
            unique_list[pos] = slot;
            return;
        }
        if (prev == key) {
            atomicAdd(&table[slot].val, 1);
            return;
        }
        slot = (slot + 1) & (uint32_t)cap_mask;
    }
}

__device__ __forceinline__ void hash_insert_unique_tracked(
    Slot* table, int cap_mask, int key,
    int* unique_list, int* unique_count
) {
    uint32_t slot = hash_fn(key, cap_mask);
    while (true) {
        int prev = atomicCAS(&table[slot].key, -1, key);
        if (prev == -1) {
            int pos = atomicAdd(unique_count, 1);
            unique_list[pos] = slot;
            return;
        }
        if (prev == key) return;
        slot = (slot + 1) & (uint32_t)cap_mask;
    }
}





__global__ void iota_kernel(int* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = idx;
}

__global__ void count_seed_info_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const int* __restrict__ seeds,
    int ns, int* __restrict__ path_counts,
    int* __restrict__ seed_degrees,
    bool is_multigraph
) {
    int sid = blockIdx.x;
    if (sid >= ns) return;
    int u = seeds[sid];
    int us = offsets[u], ue = offsets[u + 1];
    int du = ue - us;
    if (threadIdx.x == 0) seed_degrees[sid] = du;

    typedef cub::BlockReduce<int, 256> BR;
    __shared__ typename BR::TempStorage tmp;

    int my_cnt = 0;
    if (is_multigraph) {
        for (int i = us + threadIdx.x; i < ue; i += blockDim.x) {
            if (i == us || indices[i] != indices[i - 1]) {
                int w = indices[i];
                int ws = offsets[w], we = offsets[w + 1];
                int ud = 0;
                for (int j = ws; j < we; j++)
                    if (j == ws || indices[j] != indices[j - 1]) ud++;
                my_cnt += ud;
            }
        }
    } else {
        for (int i = us + threadIdx.x; i < ue; i += blockDim.x)
            my_cnt += offsets[indices[i] + 1] - offsets[indices[i]];
    }
    int total = BR(tmp).Sum(my_cnt);
    if (threadIdx.x == 0)
        path_counts[sid] = (total > 2097152) ? 2097152 : total;
}

__global__ void init_hashmaps_kernel(int2* __restrict__ hm, int64_t total) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += stride)
        hm[idx] = make_int2(-1, 0);
}

__global__ void populate_workitem_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const int* __restrict__ seeds,
    int ns,
    Slot* __restrict__ hm_buf,
    const int64_t* __restrict__ hm_offsets,
    const int* __restrict__ capacities,
    int* __restrict__ ul_buf,
    int* __restrict__ ucnts,
    const int* __restrict__ work_offsets,
    int total_work
) {
    int wid = blockIdx.x;
    if (wid >= total_work) return;
    int lo = 0, hi = ns;
    while (lo < hi) {
        int mid = (lo + hi + 1) / 2;
        if (work_offsets[mid] <= wid) lo = mid; else hi = mid - 1;
    }
    int sid = lo;
    int w_idx = wid - work_offsets[sid];
    int u = seeds[sid];
    int w = indices[offsets[u] + w_idx];
    int cap_mask = capacities[sid] - 1;
    Slot* table = hm_buf + hm_offsets[sid];
    int* ulist = ul_buf + hm_offsets[sid];
    int* ucnt = &ucnts[sid];
    int ws = offsets[w], we = offsets[w + 1];
    for (int j = ws + threadIdx.x; j < we; j += blockDim.x) {
        int v = indices[j];
        if (v != u) hash_insert_tracked(table, cap_mask, v, ulist, ucnt);
    }
}

__global__ void populate_multi_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const int* __restrict__ seeds,
    int ns,
    Slot* __restrict__ hm_buf,
    const int64_t* __restrict__ hm_offsets,
    const int* __restrict__ capacities,
    int* __restrict__ ul_buf,
    int* __restrict__ ucnts
) {
    int sid = blockIdx.x;
    if (sid >= ns) return;
    int u = seeds[sid];
    int cap_mask = capacities[sid] - 1;
    Slot* table = hm_buf + hm_offsets[sid];
    int* ulist = ul_buf + hm_offsets[sid];
    int* ucnt = &ucnts[sid];
    int us = offsets[u], ue = offsets[u + 1];
    for (int i = us; i < ue; i++) {
        if (i > us && indices[i] == indices[i - 1]) continue;
        int w = indices[i];
        int ws = offsets[w], we = offsets[w + 1];
        for (int j = ws + threadIdx.x; j < we; j += blockDim.x) {
            if (j > ws && indices[j] == indices[j - 1]) continue;
            int v = indices[j];
            if (v != u) hash_insert_unique_tracked(table, cap_mask, v, ulist, ucnt);
        }
    }
}

__global__ void extract_simple_kernel(
    const int* __restrict__ offsets,
    const Slot* __restrict__ hm_buf,
    const int64_t* __restrict__ hm_offsets,
    const int* __restrict__ ul_buf,
    const int* __restrict__ ucnts,
    const int* __restrict__ seeds,
    int ns,
    const int64_t* __restrict__ out_off,
    int* __restrict__ out_f,
    int* __restrict__ out_s,
    float* __restrict__ out_sc
) {
    int sid = blockIdx.x;
    if (sid >= ns) return;
    int u = seeds[sid];
    int du = offsets[u + 1] - offsets[u];
    const Slot* table = hm_buf + hm_offsets[sid];
    const int* ulist = ul_buf + hm_offsets[sid];
    int ucnt = ucnts[sid];
    int64_t base = out_off[sid];
    for (int i = threadIdx.x; i < ucnt; i += blockDim.x) {
        int slot = ulist[i];
        int v = table[slot].key;
        int cnt = table[slot].val;
        int dv = offsets[v + 1] - offsets[v];
        float jac = __fdividef((float)cnt, (float)(du + dv - cnt));
        out_f[base + i] = u;
        out_s[base + i] = v;
        out_sc[base + i] = jac;
    }
}

__global__ void extract_pairs_kernel(
    const Slot* __restrict__ hm_buf,
    const int64_t* __restrict__ hm_offsets,
    const int* __restrict__ ul_buf,
    const int* __restrict__ ucnts,
    const int* __restrict__ seeds,
    int ns,
    const int64_t* __restrict__ out_off,
    int* __restrict__ out_f,
    int* __restrict__ out_s
) {
    int sid = blockIdx.x;
    if (sid >= ns) return;
    int u = seeds[sid];
    const Slot* table = hm_buf + hm_offsets[sid];
    const int* ulist = ul_buf + hm_offsets[sid];
    int ucnt = ucnts[sid];
    int64_t base = out_off[sid];
    for (int i = threadIdx.x; i < ucnt; i += blockDim.x)
    { out_f[base + i] = u; out_s[base + i] = table[ulist[i]].key; }
}

__global__ void merge_jaccard_kernel(
    const int* __restrict__ offsets,
    const int* __restrict__ indices,
    const int* __restrict__ pf,
    const int* __restrict__ ps,
    float* __restrict__ scores,
    int64_t np
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < np; idx += stride) {
    int u = pf[idx], v = ps[idx];
    int us = offsets[u], ue = offsets[u + 1];
    int vs = offsets[v], ve = offsets[v + 1];
    int du = ue - us, dv = ve - vs;
    int i = us, j = vs, cnt = 0;
    while (i < ue && j < ve) {
        int a = indices[i], b = indices[j];
        if (a == b) { cnt++; i++; j++; }
        else if (a < b) i++; else j++;
    }
    scores[idx] = (du + dv - cnt > 0) ? __fdividef((float)cnt, (float)(du + dv - cnt)) : 0.0f;
    }
}

__global__ void gather_kernel(
    const int* __restrict__ sf, const int* __restrict__ ss,
    const float* __restrict__ ssc, const int* __restrict__ perm,
    int* __restrict__ df, int* __restrict__ ds, float* __restrict__ dsc,
    int64_t n
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;
    for (; idx < n; idx += stride) {
        int p = perm[idx];
        df[idx] = sf[p]; ds[idx] = ss[p]; dsc[idx] = ssc[p];
    }
}





void launch_iota(int* o, int n, cudaStream_t s) {
    if (n > 0) iota_kernel<<<(n+255)/256, 256, 0, s>>>(o, n);
}

void launch_count_seed_info(
    const int* off, const int* idx, const int* seeds, int ns,
    int* pc, int* sd, bool multi, cudaStream_t s
) {
    if (ns > 0) count_seed_info_kernel<<<ns, 256, 0, s>>>(off, idx, seeds, ns, pc, sd, multi);
}

void launch_init_hashmaps(int2* hm, int64_t total, cudaStream_t s) {
    if (total > 0) {
        int grid = (int)std::min((total+255)/256, (int64_t)2147483647);
        init_hashmaps_kernel<<<grid, 256, 0, s>>>(hm, total);
    }
}

void launch_populate_workitem(
    const int* off, const int* idx, const int* seeds, int ns,
    Slot* hm, const int64_t* hmo, const int* caps,
    int* ul, int* uc, const int* wo, int tw, int tpb, cudaStream_t s
) {
    if (tw > 0) populate_workitem_kernel<<<tw, tpb, 0, s>>>(off, idx, seeds, ns, hm, hmo, caps, ul, uc, wo, tw);
}

void launch_populate_multi(
    const int* off, const int* idx, const int* seeds, int ns,
    Slot* hm, const int64_t* hmo, const int* caps,
    int* ul, int* uc, int tpb, cudaStream_t s
) {
    if (ns > 0) populate_multi_kernel<<<ns, tpb, 0, s>>>(off, idx, seeds, ns, hm, hmo, caps, ul, uc);
}

void launch_extract_simple(
    const int* off, const Slot* hm, const int64_t* hmo,
    const int* ul, const int* uc, const int* seeds, int ns,
    const int64_t* oo, int* f, int* sc, float* scores, cudaStream_t s
) {
    if (ns > 0) extract_simple_kernel<<<ns, 256, 0, s>>>(off, hm, hmo, ul, uc, seeds, ns, oo, f, sc, scores);
}

void launch_extract_pairs(
    const Slot* hm, const int64_t* hmo,
    const int* ul, const int* uc, const int* seeds, int ns,
    const int64_t* oo, int* f, int* sc, cudaStream_t s
) {
    if (ns > 0) extract_pairs_kernel<<<ns, 256, 0, s>>>(hm, hmo, ul, uc, seeds, ns, oo, f, sc);
}

void launch_merge_jaccard(
    const int* off, const int* idx,
    const int* pf, const int* ps, float* sc, int64_t np, cudaStream_t s
) {
    if (np > 0) {
        int grid = (int)std::min((np+255)/256, (int64_t)2147483647);
        merge_jaccard_kernel<<<grid, 256, 0, s>>>(off, idx, pf, ps, sc, np);
    }
}

void launch_gather(
    const int* sf, const int* ss, const float* ssc, const int* p,
    int* df, int* ds, float* dsc, int64_t n, cudaStream_t s
) {
    if (n > 0) {
        int grid = (int)std::min((n+255)/256, (int64_t)2147483647);
        gather_kernel<<<grid, 256, 0, s>>>(sf, ss, ssc, p, df, ds, dsc, n);
    }
}

static inline int next_pow2(int v) {
    if (v <= 256) return 256;
    v--; v |= v>>1; v |= v>>2; v |= v>>4; v |= v>>8; v |= v>>16;
    return v + 1;
}

}  

similarity_result_float_t jaccard_all_pairs_similarity_seg(
    const graph32_t& graph,
    const int32_t* vertices,
    std::size_t num_vertices,
    std::optional<std::size_t> topk) {

    static int tag;
    auto& cache = cache_pool().acquire<Cache>(&tag);

    const int32_t* d_off = graph.offsets;
    const int32_t* d_idx = graph.indices;
    int nv = graph.number_of_vertices;
    bool is_multi = graph.is_multigraph;
    cudaStream_t s = 0;

    
    int ns;
    const int32_t* d_seeds;

    if (vertices != nullptr && num_vertices > 0) {
        ns = (int)num_vertices;
        d_seeds = vertices;
    } else {
        ns = nv;
        Cache::ensure(cache.seeds, cache.seeds_cap, (int64_t)nv);
        launch_iota(cache.seeds, nv, s);
        d_seeds = cache.seeds;
    }

    if (ns == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    Cache::ensure(cache.pc, cache.pc_cap, (int64_t)ns);
    Cache::ensure(cache.sd, cache.sd_cap, (int64_t)ns);

    
    launch_count_seed_info(d_off, d_idx, d_seeds, ns,
        cache.pc, cache.sd, is_multi, s);

    
    std::vector<int> h_pc(ns), h_sd(ns);
    cudaMemcpyAsync(h_pc.data(), cache.pc, ns * 4, cudaMemcpyDeviceToHost, s);
    cudaMemcpyAsync(h_sd.data(), cache.sd, ns * 4, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    
    std::vector<int> h_caps(ns);
    std::vector<int64_t> h_hmo(ns + 1);
    std::vector<int> h_wo(ns + 1);
    int64_t hm_sum = 0;
    int work_sum = 0;
    for (int i = 0; i < ns; i++) {
        int cap = next_pow2(std::max(256, std::min(h_pc[i] * 2, 2097152)));
        h_caps[i] = cap;
        h_hmo[i] = hm_sum;
        hm_sum += cap;
        h_wo[i] = work_sum;
        work_sum += h_sd[i];
    }
    h_hmo[ns] = hm_sum;
    h_wo[ns] = work_sum;

    int64_t total_slots = hm_sum;
    int total_work = work_sum;

    if (total_slots == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    Cache::ensure(cache.caps_buf, cache.caps_buf_cap, (int64_t)ns);
    Cache::ensure(cache.hmo, cache.hmo_cap, (int64_t)(ns + 1));
    Cache::ensure(cache.wo, cache.wo_cap, (int64_t)(ns + 1));
    cudaMemcpyAsync(cache.caps_buf, h_caps.data(), ns * 4, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(cache.hmo, h_hmo.data(), (ns + 1) * 8, cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(cache.wo, h_wo.data(), (ns + 1) * 4, cudaMemcpyHostToDevice, s);

    
    Cache::ensure(cache.hm, cache.hm_cap, total_slots);
    Cache::ensure(cache.ul, cache.ul_cap, total_slots);
    Cache::ensure(cache.uc, cache.uc_cap, (int64_t)ns);
    cudaMemsetAsync(cache.uc, 0, ns * 4, s);

    
    launch_init_hashmaps((int2*)cache.hm, total_slots, s);

    
    if (is_multi) {
        launch_populate_multi(d_off, d_idx, d_seeds, ns,
            cache.hm, cache.hmo, cache.caps_buf,
            cache.ul, cache.uc, 256, s);
    } else {
        launch_populate_workitem(d_off, d_idx, d_seeds, ns,
            cache.hm, cache.hmo, cache.caps_buf,
            cache.ul, cache.uc,
            cache.wo, total_work, 256, s);
    }

    
    std::vector<int> h_uc(ns);
    cudaMemcpyAsync(h_uc.data(), cache.uc, ns * 4, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    std::vector<int64_t> h_oo(ns + 1);
    int64_t total_pairs = 0;
    for (int i = 0; i < ns; i++) {
        h_oo[i] = total_pairs;
        total_pairs += h_uc[i];
    }
    h_oo[ns] = total_pairs;

    if (total_pairs == 0) {
        return {nullptr, nullptr, nullptr, 0};
    }

    
    Cache::ensure(cache.oo, cache.oo_cap, (int64_t)(ns + 1));
    cudaMemcpyAsync(cache.oo, h_oo.data(), (ns + 1) * 8, cudaMemcpyHostToDevice, s);

    
    int32_t* out_first;
    int32_t* out_second;
    float* out_scores;
    cudaMalloc(&out_first, total_pairs * sizeof(int32_t));
    cudaMalloc(&out_second, total_pairs * sizeof(int32_t));
    cudaMalloc(&out_scores, total_pairs * sizeof(float));

    if (is_multi) {
        launch_extract_pairs(cache.hm, cache.hmo,
            cache.ul, cache.uc, d_seeds, ns,
            cache.oo, out_first, out_second, s);
        launch_merge_jaccard(d_off, d_idx,
            out_first, out_second,
            out_scores, total_pairs, s);
    } else {
        launch_extract_simple(d_off, cache.hm, cache.hmo,
            cache.ul, cache.uc, d_seeds, ns,
            cache.oo, out_first, out_second,
            out_scores, s);
    }

    
    if (topk.has_value() && (std::size_t)total_pairs > topk.value()) {
        std::size_t k = topk.value();
        int n = (int)total_pairs;

        Cache::ensure(cache.iota_buf, cache.iota_cap, (int64_t)n);
        launch_iota(cache.iota_buf, n, s);

        Cache::ensure(cache.ssc, cache.ssc_cap, (int64_t)n);
        Cache::ensure(cache.si, cache.si_cap, (int64_t)n);

        size_t tb = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, tb, out_scores, cache.ssc,
            cache.iota_buf, cache.si, n, 0, 32, s);

        if (cache.sort_tmp_cap < tb) {
            if (cache.sort_tmp) cudaFree(cache.sort_tmp);
            cudaMalloc(&cache.sort_tmp, tb);
            cache.sort_tmp_cap = tb;
        }

        cub::DeviceRadixSort::SortPairsDescending(
            cache.sort_tmp, tb, out_scores, cache.ssc,
            cache.iota_buf, cache.si, n, 0, 32, s);

        
        int32_t* topk_first;
        int32_t* topk_second;
        float* topk_scores;
        cudaMalloc(&topk_first, k * sizeof(int32_t));
        cudaMalloc(&topk_second, k * sizeof(int32_t));
        cudaMalloc(&topk_scores, k * sizeof(float));

        launch_gather(out_first, out_second, out_scores,
            cache.si, topk_first, topk_second,
            topk_scores, (int64_t)k, s);

        
        cudaFree(out_first);
        cudaFree(out_second);
        cudaFree(out_scores);

        return {topk_first, topk_second, topk_scores, k};
    }

    return {out_first, out_second, out_scores, (std::size_t)total_pairs};
}

}  
