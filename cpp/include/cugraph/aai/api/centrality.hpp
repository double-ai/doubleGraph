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
 *
 * AAI Centrality Algorithms: Betweenness, Edge Betweenness, Eigenvector, Katz
 */
#pragma once

#include <cugraph/aai/compact_graph.hpp>
#include <cugraph/aai/types.hpp>

#include <cstddef>
#include <cstdint>

namespace aai {

// =============================================================================
// Betweenness Centrality
// =============================================================================
//
// NOTE: Only unweighted variants are provided. cuGraph's betweenness centrality
// implementation uses Brandes' algorithm with BFS, which computes shortest paths
// by hop count only. cuGraph does NOT support weighted betweenness centrality -
// edge weights would be silently ignored if provided.
//
// A proper weighted betweenness centrality would require Dijkstra-based shortest
// path computation, which cuGraph does not currently implement.
// =============================================================================

/**
 * Betweenness Centrality - Unweighted.
 *
 * Computes betweenness centrality for all vertices using Brandes' algorithm.
 *
 * Preconditions (from cpp/src/centrality/betweenness_centrality_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 1141]
 *   - If sample_vertices is provided, vertices must be valid
 *       [CUGRAPH_EXPECTS in do_expensive_check block, lines 1167-1168]
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * @param graph              Input graph. MUST be CSR (is_csc=false).
 * @param centralities       [out] Pre-allocated array of size num_vertices.
 * @param normalized         If true, normalize results (see normalization details below).
 * @param include_endpoints  If true, include endpoint contributions (see below).
 * @param sample_vertices    [in] Vertex IDs for k-sampling. Always provided (non-null).
 *                           When all vertices are used, this is a shuffled permutation
 *                           of all vertex IDs (never sorted).
 * @param num_samples        Number of sample vertices (always > 0).
 *
 * Output specification:
 *
 * The output is the standard Brandes betweenness centrality, defined as:
 *   BC(v) = sum over source vertices s of delta_s(v)
 * where delta_s(v) is the dependency of s on v:
 *   delta_s(v) = sum over vertices w where v is a predecessor of w in the
 *                BFS from s of: (sigma_sv / sigma_sw) * (1 + delta_s(w))
 * sigma_sv = number of shortest paths from s to v.
 * The source vertex s does not contribute to its own centrality (s is
 * excluded from the dependency accumulation for source s).
 *
 * Only the vertices listed in sample_vertices are used as sources.
 *
 * include_endpoints:
 *   When true, the following is added to the raw centrality BEFORE
 *   normalization, for each source s:
 *     - Source vertex s:             centrality[s] += (reachable count, excluding s)
 *     - Each reachable non-source v: centrality[v] += 1
 *     - Unreachable vertices:        unchanged
 *
 * Normalization (applied AFTER all sources have been processed):
 *   Let n = num_vertices, k = num_samples.
 *   Let adj = (include_endpoints ? n : n - 1).
 *   Let all_srcs = (k == adj) || include_endpoints.
 *
 *   If all_srcs:
 *     - normalized:              scale = k * (adj - 1)
 *     - !normalized, symmetric:  scale = k * 2 / adj
 *     - !normalized, !symmetric: scale = k / adj
 *     All vertices: centrality[v] /= scale
 *
 *   Else if normalized:
 *     Non-source vertices: centrality[v] /= k * (adj - 1)
 *     Source vertices:     centrality[v] /= (k - 1) * (adj - 1)
 *
 *   Else (!normalized, !all_srcs):
 *     s_ns = k / adj;           s_s = (k - 1) / adj
 *     If symmetric: s_ns *= 2;  s_s *= 2
 *     Non-source vertices: centrality[v] /= s_ns
 *     Source vertices:     centrality[v] /= s_s
 *
 *   "Source vertex" means v appears in sample_vertices.
 *   Note: when all n vertices are used as sources and !include_endpoints,
 *   adj = n-1 so all_srcs = (n == n-1) = false. In this case every vertex
 *   IS a source, so every vertex uses scale_source.
 */
void betweenness_centrality(const graph32_t& graph,
                            float* centralities,
                            bool normalized,
                            bool include_endpoints,
                            const int32_t* sample_vertices,
                            std::size_t num_samples);

/**
 * Betweenness Centrality - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes betweenness centrality for all vertices using Brandes' algorithm.
 *
 * Preconditions (from cpp/src/centrality/betweenness_centrality_impl.cuh):
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 1141]
 *   - If sample_vertices is provided, vertices must be valid
 *       [CUGRAPH_EXPECTS in do_expensive_check block, lines 1167-1168]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * @param graph              Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets.
 * @param centralities       [out] Pre-allocated array of size num_vertices.
 * @param normalized         If true, normalize results (see normalization details below).
 * @param include_endpoints  If true, include endpoint contributions (see below).
 * @param sample_vertices    [in] Vertex IDs for k-sampling. Always provided (non-null).
 *                           When all vertices are used, this is a shuffled permutation
 *                           of all vertex IDs (never sorted).
 * @param num_samples        Number of sample vertices (always > 0).
 *
 * Output specification:
 *
 * The output is the standard Brandes betweenness centrality, defined as:
 *   BC(v) = sum over source vertices s of delta_s(v)
 * where delta_s(v) is the dependency of s on v:
 *   delta_s(v) = sum over vertices w where v is a predecessor of w in the
 *                BFS from s of: (sigma_sv / sigma_sw) * (1 + delta_s(w))
 * sigma_sv = number of shortest paths from s to v.
 * The source vertex s does not contribute to its own centrality (s is
 * excluded from the dependency accumulation for source s).
 *
 * Only the vertices listed in sample_vertices are used as sources.
 *
 * include_endpoints:
 *   When true, the following is added to the raw centrality BEFORE
 *   normalization, for each source s:
 *     - Source vertex s:             centrality[s] += (reachable count, excluding s)
 *     - Each reachable non-source v: centrality[v] += 1
 *     - Unreachable vertices:        unchanged
 *
 * Normalization (applied AFTER all sources have been processed):
 *   Let n = num_vertices, k = num_samples.
 *   Let adj = (include_endpoints ? n : n - 1).
 *   Let all_srcs = (k == adj) || include_endpoints.
 *
 *   If all_srcs:
 *     - normalized:              scale = k * (adj - 1)
 *     - !normalized, symmetric:  scale = k * 2 / adj
 *     - !normalized, !symmetric: scale = k / adj
 *     All vertices: centrality[v] /= scale
 *
 *   Else if normalized:
 *     Non-source vertices: centrality[v] /= k * (adj - 1)
 *     Source vertices:     centrality[v] /= (k - 1) * (adj - 1)
 *
 *   Else (!normalized, !all_srcs):
 *     s_ns = k / adj;           s_s = (k - 1) / adj
 *     If symmetric: s_ns *= 2;  s_s *= 2
 *     Non-source vertices: centrality[v] /= s_ns
 *     Source vertices:     centrality[v] /= s_s
 *
 *   "Source vertex" means v appears in sample_vertices.
 *   Note: when all n vertices are used as sources and !include_endpoints,
 *   adj = n-1 so all_srcs = (n == n-1) = false. In this case every vertex
 *   IS a source, so every vertex uses scale_source.
 */
void betweenness_centrality_seg(const graph32_t& graph,
                                float* centralities,
                                bool normalized,
                                bool include_endpoints,
                                const int32_t* sample_vertices,
                                std::size_t num_samples);

/**
 * Betweenness Centrality - Unweighted - Edge mask variant (no precomputed segments).
 *
 * Same as betweenness_centrality but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSR format (is_csc=false)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * @param graph              Input graph. MUST be CSR (is_csc=false).
 * @param centralities       [out] Pre-allocated array of size num_vertices.
 * @param normalized         If true, normalize results (see normalization details below).
 * @param include_endpoints  If true, include endpoint contributions (see below).
 * @param sample_vertices    [in] Vertex IDs for k-sampling. Always provided (non-null).
 *                           When all vertices are used, this is a shuffled permutation
 *                           of all vertex IDs (never sorted).
 * @param num_samples        Number of sample vertices (always > 0).
 *
 * Output specification:
 *
 * The output is the standard Brandes betweenness centrality, defined as:
 *   BC(v) = sum over source vertices s of delta_s(v)
 * where delta_s(v) is the dependency of s on v:
 *   delta_s(v) = sum over vertices w where v is a predecessor of w in the
 *                BFS from s of: (sigma_sv / sigma_sw) * (1 + delta_s(w))
 * sigma_sv = number of shortest paths from s to v.
 * The source vertex s does not contribute to its own centrality (s is
 * excluded from the dependency accumulation for source s).
 *
 * Only the vertices listed in sample_vertices are used as sources.
 *
 * include_endpoints:
 *   When true, the following is added to the raw centrality BEFORE
 *   normalization, for each source s:
 *     - Source vertex s:             centrality[s] += (reachable count, excluding s)
 *     - Each reachable non-source v: centrality[v] += 1
 *     - Unreachable vertices:        unchanged
 *
 * Normalization (applied AFTER all sources have been processed):
 *   Let n = num_vertices, k = num_samples.
 *   Let adj = (include_endpoints ? n : n - 1).
 *   Let all_srcs = (k == adj) || include_endpoints.
 *
 *   If all_srcs:
 *     - normalized:              scale = k * (adj - 1)
 *     - !normalized, symmetric:  scale = k * 2 / adj
 *     - !normalized, !symmetric: scale = k / adj
 *     All vertices: centrality[v] /= scale
 *
 *   Else if normalized:
 *     Non-source vertices: centrality[v] /= k * (adj - 1)
 *     Source vertices:     centrality[v] /= (k - 1) * (adj - 1)
 *
 *   Else (!normalized, !all_srcs):
 *     s_ns = k / adj;           s_s = (k - 1) / adj
 *     If symmetric: s_ns *= 2;  s_s *= 2
 *     Non-source vertices: centrality[v] /= s_ns
 *     Source vertices:     centrality[v] /= s_s
 *
 *   "Source vertex" means v appears in sample_vertices.
 *   Note: when all n vertices are used as sources and !include_endpoints,
 *   adj = n-1 so all_srcs = (n == n-1) = false. In this case every vertex
 *   IS a source, so every vertex uses scale_source.
 */
void betweenness_centrality_mask(const graph32_t& graph,
                                  float* centralities,
                                  bool normalized,
                                  bool include_endpoints,
                                  const int32_t* sample_vertices,
                                  std::size_t num_samples);

/**
 * Betweenness Centrality - Unweighted - Precomputed segments + edge mask variant.
 *
 * Same as betweenness_centrality_seg but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSR format (is_csc=false)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * @param graph              Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets.
 * @param centralities       [out] Pre-allocated array of size num_vertices.
 * @param normalized         If true, normalize results (see normalization details below).
 * @param include_endpoints  If true, include endpoint contributions (see below).
 * @param sample_vertices    [in] Vertex IDs for k-sampling. Always provided (non-null).
 *                           When all vertices are used, this is a shuffled permutation
 *                           of all vertex IDs (never sorted).
 * @param num_samples        Number of sample vertices (always > 0).
 *
 * Output specification:
 *
 * The output is the standard Brandes betweenness centrality, defined as:
 *   BC(v) = sum over source vertices s of delta_s(v)
 * where delta_s(v) is the dependency of s on v:
 *   delta_s(v) = sum over vertices w where v is a predecessor of w in the
 *                BFS from s of: (sigma_sv / sigma_sw) * (1 + delta_s(w))
 * sigma_sv = number of shortest paths from s to v.
 * The source vertex s does not contribute to its own centrality (s is
 * excluded from the dependency accumulation for source s).
 *
 * Only the vertices listed in sample_vertices are used as sources.
 *
 * include_endpoints:
 *   When true, the following is added to the raw centrality BEFORE
 *   normalization, for each source s:
 *     - Source vertex s:             centrality[s] += (reachable count, excluding s)
 *     - Each reachable non-source v: centrality[v] += 1
 *     - Unreachable vertices:        unchanged
 *
 * Normalization (applied AFTER all sources have been processed):
 *   Let n = num_vertices, k = num_samples.
 *   Let adj = (include_endpoints ? n : n - 1).
 *   Let all_srcs = (k == adj) || include_endpoints.
 *
 *   If all_srcs:
 *     - normalized:              scale = k * (adj - 1)
 *     - !normalized, symmetric:  scale = k * 2 / adj
 *     - !normalized, !symmetric: scale = k / adj
 *     All vertices: centrality[v] /= scale
 *
 *   Else if normalized:
 *     Non-source vertices: centrality[v] /= k * (adj - 1)
 *     Source vertices:     centrality[v] /= (k - 1) * (adj - 1)
 *
 *   Else (!normalized, !all_srcs):
 *     s_ns = k / adj;           s_s = (k - 1) / adj
 *     If symmetric: s_ns *= 2;  s_s *= 2
 *     Non-source vertices: centrality[v] /= s_ns
 *     Source vertices:     centrality[v] /= s_s
 *
 *   "Source vertex" means v appears in sample_vertices.
 *   Note: when all n vertices are used as sources and !include_endpoints,
 *   adj = n-1 so all_srcs = (n == n-1) = false. In this case every vertex
 *   IS a source, so every vertex uses scale_source.
 */
void betweenness_centrality_seg_mask(const graph32_t& graph,
                                      float* centralities,
                                      bool normalized,
                                      bool include_endpoints,
                                      const int32_t* sample_vertices,
                                      std::size_t num_samples);

// =============================================================================
// Edge Betweenness Centrality
// =============================================================================
//
// NOTE: Only unweighted variants are provided. cuGraph's edge betweenness centrality
// implementation uses Brandes' algorithm with BFS, which computes shortest paths
// by hop count only. cuGraph does NOT support weighted edge betweenness centrality -
// edge weights would be silently ignored if provided.
//
// A proper weighted edge betweenness centrality would require Dijkstra-based shortest
// path computation, which cuGraph does not currently implement.
// =============================================================================

/**
 * Edge Betweenness Centrality - Unweighted.
 *
 * Computes betweenness centrality for edges using Brandes' algorithm.
 *
 * Preconditions (from cpp/src/centrality/betweenness_centrality_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 1316]
 *   - If sample_vertices is provided, vertices must be valid
 *       [CUGRAPH_EXPECTS in do_expensive_check block, lines 1343-1344]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal lower_bound assumes sorted neighbors)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * @param graph             Input graph. MUST be CSR (is_csc=false).
 * @param edge_centralities [out] Pre-allocated array of size num_edges.
 * @param normalized        If true, normalize results.
 * @param sample_vertices   [in] Vertex IDs for k-sampling, or nullptr for all vertices.
 * @param num_samples       Number of sample vertices. Ignored if sample_vertices is nullptr.
 */
void edge_betweenness_centrality(const graph32_t& graph,
                                 float* edge_centralities,
                                 bool normalized,
                                 const int32_t* sample_vertices,
                                 std::size_t num_samples);

/**
 * Edge Betweenness Centrality - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes betweenness centrality for edges using Brandes' algorithm.
 *
 * Preconditions (from cpp/src/centrality/betweenness_centrality_impl.cuh):
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 1316]
 *   - If sample_vertices is provided, vertices must be valid
 *       [CUGRAPH_EXPECTS in do_expensive_check block, lines 1343-1344]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * @param graph             Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets.
 * @param edge_centralities [out] Pre-allocated array of size num_edges.
 * @param normalized        If true, normalize results.
 * @param sample_vertices   [in] Vertex IDs for k-sampling, or nullptr for all vertices.
 * @param num_samples       Number of sample vertices. Ignored if sample_vertices is nullptr.
 */
void edge_betweenness_centrality_seg(const graph32_t& graph,
                                     float* edge_centralities,
                                     bool normalized,
                                     const int32_t* sample_vertices,
                                     std::size_t num_samples);

/**
 * Edge Betweenness Centrality - Unweighted - Edge mask variant (no precomputed segments).
 *
 * Same as edge_betweenness_centrality but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSR format (is_csc=false)
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal lower_bound assumes sorted neighbors)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * @param graph             Input graph. MUST be CSR (is_csc=false).
 * @param edge_centralities [out] Pre-allocated array of size num_edges.
 * @param normalized        If true, normalize results.
 * @param sample_vertices   [in] Vertex IDs for k-sampling, or nullptr for all vertices.
 * @param num_samples       Number of sample vertices. Ignored if sample_vertices is nullptr.
 */
void edge_betweenness_centrality_mask(const graph32_t& graph,
                                       float* edge_centralities,
                                       bool normalized,
                                       const int32_t* sample_vertices,
                                       std::size_t num_samples);

/**
 * Edge Betweenness Centrality - Unweighted - Precomputed segments + edge mask variant.
 *
 * Same as edge_betweenness_centrality_seg but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSR format (is_csc=false)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * @param graph             Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets.
 * @param edge_centralities [out] Pre-allocated array of size num_edges.
 * @param normalized        If true, normalize results.
 * @param sample_vertices   [in] Vertex IDs for k-sampling, or nullptr for all vertices.
 * @param num_samples       Number of sample vertices. Ignored if sample_vertices is nullptr.
 */
void edge_betweenness_centrality_seg_mask(const graph32_t& graph,
                                           float* edge_centralities,
                                           bool normalized,
                                           const int32_t* sample_vertices,
                                           std::size_t num_samples);

// =============================================================================
// Eigenvector Centrality
// =============================================================================

/**
 * Eigenvector Centrality - Unweighted.
 *
 * Computes eigenvector centrality using power iteration with the formula
 * x_new = normalize((A + I) * x_old), where A is the adjacency matrix and
 * I is the identity matrix. The identity addition improves numerical stability
 * without changing the final eigenvector.
 *
 * Preconditions (from cpp/src/centrality/eigenvector_centrality_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [implicit via template parameter is_storage_transposed=true, line 163]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 175]
 *   - If initial_centralities is provided, it must match vertex partition range size
 *       [CUGRAPH_EXPECTS, lines 176-179]
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * @param graph                 Input graph. MUST be CSC (is_csc=true).
 * @param centralities          [out] Pre-allocated array of size num_vertices.
 * @param epsilon               Convergence tolerance. Converges when
 *                              sum(|x_new - x_old|) < num_vertices * epsilon.
 *                              Must be >= 0.0.
 * @param max_iterations        Maximum power iterations.
 * @param initial_centralities  [in] Initial values (size = num_vertices), or nullptr.
 *
 * eigenvector_centrality_result_t struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @return                                 Result metadata (convergence status).
 */
eigenvector_centrality_result_t eigenvector_centrality(const graph32_t& graph,
                            float* centralities,
                            float epsilon,
                            std::size_t max_iterations,
                            const float* initial_centralities);

/**
 * Eigenvector Centrality - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes eigenvector centrality using power iteration with the formula
 * x_new = normalize((A + I) * x_old), where A is the adjacency matrix and
 * I is the identity matrix. The identity addition improves numerical stability
 * without changing the final eigenvector.
 *
 * Preconditions (from cpp/src/centrality/eigenvector_centrality_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [implicit via template parameter is_storage_transposed=true, line 163]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 175]
 *   - If initial_centralities is provided, it must match vertex partition range size
 *       [CUGRAPH_EXPECTS, lines 176-179]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * @param graph                 Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param centralities          [out] Pre-allocated array of size num_vertices.
 * @param epsilon               Convergence tolerance. Converges when
 *                              sum(|x_new - x_old|) < num_vertices * epsilon.
 *                              Must be >= 0.0.
 * @param max_iterations        Maximum power iterations.
 * @param initial_centralities  [in] Initial values (size = num_vertices), or nullptr.
 *
 * eigenvector_centrality_result_t struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @return                                 Result metadata (convergence status).
 */
eigenvector_centrality_result_t eigenvector_centrality_seg(const graph32_t& graph,
                                float* centralities,
                                float epsilon,
                                std::size_t max_iterations,
                                const float* initial_centralities);

/**
 * Eigenvector Centrality - Float weights.
 *
 * Computes eigenvector centrality using power iteration with the formula
 * x_new = normalize((A + I) * x_old), where A is the adjacency matrix and
 * I is the identity matrix. The identity addition improves numerical stability
 * without changing the final eigenvector.
 *
 * Preconditions (from cpp/src/centrality/eigenvector_centrality_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [implicit via template parameter is_storage_transposed=true, line 163]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 175]
 *   - If initial_centralities is provided, it must match vertex partition range size
 *       [CUGRAPH_EXPECTS, lines 176-179]
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * @param graph                 Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param centralities          [out] Pre-allocated array of size num_vertices.
 * @param epsilon               Convergence tolerance. Converges when
 *                              sum(|x_new - x_old|) < num_vertices * epsilon.
 *                              Must be >= 0.0.
 * @param max_iterations        Maximum power iterations.
 * @param initial_centralities  [in] Initial values (size = num_vertices), or nullptr.
 *
 * eigenvector_centrality_result_t struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @return                                 Result metadata (convergence status).
 */
eigenvector_centrality_result_t eigenvector_centrality(const graph32_t& graph,
                            const float* edge_weights,
                            float* centralities,
                            float epsilon,
                            std::size_t max_iterations,
                            const float* initial_centralities);

/**
 * Eigenvector Centrality - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes eigenvector centrality using power iteration with the formula
 * x_new = normalize((A + I) * x_old), where A is the adjacency matrix and
 * I is the identity matrix. The identity addition improves numerical stability
 * without changing the final eigenvector.
 *
 * Preconditions (from cpp/src/centrality/eigenvector_centrality_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [implicit via template parameter is_storage_transposed=true, line 163]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 175]
 *   - If initial_centralities is provided, it must match vertex partition range size
 *       [CUGRAPH_EXPECTS, lines 176-179]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * @param graph                 Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param centralities          [out] Pre-allocated array of size num_vertices.
 * @param epsilon               Convergence tolerance. Converges when
 *                              sum(|x_new - x_old|) < num_vertices * epsilon.
 *                              Must be >= 0.0.
 * @param max_iterations        Maximum power iterations.
 * @param initial_centralities  [in] Initial values (size = num_vertices), or nullptr.
 *
 * eigenvector_centrality_result_t struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @return                                 Result metadata (convergence status).
 */
eigenvector_centrality_result_t eigenvector_centrality_seg(const graph32_t& graph,
                                const float* edge_weights,
                                float* centralities,
                                float epsilon,
                                std::size_t max_iterations,
                                const float* initial_centralities);

/**
 * Eigenvector Centrality - Double weights.
 *
 * Computes eigenvector centrality using power iteration with the formula
 * x_new = normalize((A + I) * x_old), where A is the adjacency matrix and
 * I is the identity matrix. The identity addition improves numerical stability
 * without changing the final eigenvector.
 *
 * Preconditions (from cpp/src/centrality/eigenvector_centrality_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [implicit via template parameter is_storage_transposed=true, line 163]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 175]
 *   - If initial_centralities is provided, it must match vertex partition range size
 *       [CUGRAPH_EXPECTS, lines 176-179]
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * @param graph                 Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param centralities          [out] Pre-allocated array of size num_vertices.
 * @param epsilon               Convergence tolerance. Converges when
 *                              sum(|x_new - x_old|) < num_vertices * epsilon.
 *                              Must be >= 0.0.
 * @param max_iterations        Maximum power iterations.
 * @param initial_centralities  [in] Initial values (size = num_vertices), or nullptr.
 *
 * eigenvector_centrality_result_t struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @return                                 Result metadata (convergence status).
 */
eigenvector_centrality_result_t eigenvector_centrality(const graph32_t& graph,
                            const double* edge_weights,
                            double* centralities,
                            double epsilon,
                            std::size_t max_iterations,
                            const double* initial_centralities);

/**
 * Eigenvector Centrality - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes eigenvector centrality using power iteration with the formula
 * x_new = normalize((A + I) * x_old), where A is the adjacency matrix and
 * I is the identity matrix. The identity addition improves numerical stability
 * without changing the final eigenvector.
 *
 * Preconditions (from cpp/src/centrality/eigenvector_centrality_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [implicit via template parameter is_storage_transposed=true, line 163]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 175]
 *   - If initial_centralities is provided, it must match vertex partition range size
 *       [CUGRAPH_EXPECTS, lines 176-179]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * @param graph                 Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param centralities          [out] Pre-allocated array of size num_vertices.
 * @param epsilon               Convergence tolerance. Converges when
 *                              sum(|x_new - x_old|) < num_vertices * epsilon.
 *                              Must be >= 0.0.
 * @param max_iterations        Maximum power iterations.
 * @param initial_centralities  [in] Initial values (size = num_vertices), or nullptr.
 *
 * eigenvector_centrality_result_t struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @return                                 Result metadata (convergence status).
 */
eigenvector_centrality_result_t eigenvector_centrality_seg(const graph32_t& graph,
                                const double* edge_weights,
                                double* centralities,
                                double epsilon,
                                std::size_t max_iterations,
                                const double* initial_centralities);

// -----------------------------------------------------------------------------
// Eigenvector Centrality - Edge mask variants
// -----------------------------------------------------------------------------

/**
 * Eigenvector Centrality - Unweighted - Edge mask variant (no precomputed segments).
 *
 * Same as eigenvector_centrality (unweighted) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Computes eigenvector centrality using power iteration with the formula
 * x_new = normalize((A + I) * x_old), where A is the adjacency matrix and
 * I is the identity matrix. The identity addition improves numerical stability
 * without changing the final eigenvector.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * @param graph                 Input graph. MUST be CSC (is_csc=true).
 * @param centralities          [out] Pre-allocated array of size num_vertices.
 * @param epsilon               Convergence tolerance. Converges when
 *                              sum(|x_new - x_old|) < num_vertices * epsilon.
 *                              Must be >= 0.0.
 * @param max_iterations        Maximum power iterations.
 * @param initial_centralities  [in] Initial values (size = num_vertices), or nullptr.
 *
 * eigenvector_centrality_result_t struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @return                                 Result metadata (convergence status).
 */
eigenvector_centrality_result_t eigenvector_centrality_mask(const graph32_t& graph,
                                  float* centralities,
                                  float epsilon,
                                  std::size_t max_iterations,
                                  const float* initial_centralities);

/**
 * Eigenvector Centrality - Unweighted - Precomputed segments + edge mask variant.
 *
 * Same as eigenvector_centrality_seg (unweighted) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Computes eigenvector centrality using power iteration with the formula
 * x_new = normalize((A + I) * x_old), where A is the adjacency matrix and
 * I is the identity matrix. The identity addition improves numerical stability
 * without changing the final eigenvector.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * @param graph                 Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param centralities          [out] Pre-allocated array of size num_vertices.
 * @param epsilon               Convergence tolerance. Converges when
 *                              sum(|x_new - x_old|) < num_vertices * epsilon.
 *                              Must be >= 0.0.
 * @param max_iterations        Maximum power iterations.
 * @param initial_centralities  [in] Initial values (size = num_vertices), or nullptr.
 *
 * eigenvector_centrality_result_t struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @return                                 Result metadata (convergence status).
 */
eigenvector_centrality_result_t eigenvector_centrality_seg_mask(const graph32_t& graph,
                                      float* centralities,
                                      float epsilon,
                                      std::size_t max_iterations,
                                      const float* initial_centralities);

/**
 * Eigenvector Centrality - Float weights - Edge mask variant (no precomputed segments).
 *
 * Same as eigenvector_centrality (float weights) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Computes eigenvector centrality using power iteration with the formula
 * x_new = normalize((A + I) * x_old), where A is the adjacency matrix and
 * I is the identity matrix. The identity addition improves numerical stability
 * without changing the final eigenvector.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * @param graph                 Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param centralities          [out] Pre-allocated array of size num_vertices.
 * @param epsilon               Convergence tolerance. Converges when
 *                              sum(|x_new - x_old|) < num_vertices * epsilon.
 *                              Must be >= 0.0.
 * @param max_iterations        Maximum power iterations.
 * @param initial_centralities  [in] Initial values (size = num_vertices), or nullptr.
 *
 * eigenvector_centrality_result_t struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @return                                 Result metadata (convergence status).
 */
eigenvector_centrality_result_t eigenvector_centrality_mask(const graph32_t& graph,
                                  const float* edge_weights,
                                  float* centralities,
                                  float epsilon,
                                  std::size_t max_iterations,
                                  const float* initial_centralities);

/**
 * Eigenvector Centrality - Float weights - Precomputed segments + edge mask variant.
 *
 * Same as eigenvector_centrality_seg (float weights) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Computes eigenvector centrality using power iteration with the formula
 * x_new = normalize((A + I) * x_old), where A is the adjacency matrix and
 * I is the identity matrix. The identity addition improves numerical stability
 * without changing the final eigenvector.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * @param graph                 Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param centralities          [out] Pre-allocated array of size num_vertices.
 * @param epsilon               Convergence tolerance. Converges when
 *                              sum(|x_new - x_old|) < num_vertices * epsilon.
 *                              Must be >= 0.0.
 * @param max_iterations        Maximum power iterations.
 * @param initial_centralities  [in] Initial values (size = num_vertices), or nullptr.
 *
 * eigenvector_centrality_result_t struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @return                                 Result metadata (convergence status).
 */
eigenvector_centrality_result_t eigenvector_centrality_seg_mask(const graph32_t& graph,
                                      const float* edge_weights,
                                      float* centralities,
                                      float epsilon,
                                      std::size_t max_iterations,
                                      const float* initial_centralities);

/**
 * Eigenvector Centrality - Double weights - Edge mask variant (no precomputed segments).
 *
 * Same as eigenvector_centrality (double weights) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Computes eigenvector centrality using power iteration with the formula
 * x_new = normalize((A + I) * x_old), where A is the adjacency matrix and
 * I is the identity matrix. The identity addition improves numerical stability
 * without changing the final eigenvector.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * @param graph                 Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param centralities          [out] Pre-allocated array of size num_vertices.
 * @param epsilon               Convergence tolerance. Converges when
 *                              sum(|x_new - x_old|) < num_vertices * epsilon.
 *                              Must be >= 0.0.
 * @param max_iterations        Maximum power iterations.
 * @param initial_centralities  [in] Initial values (size = num_vertices), or nullptr.
 *
 * eigenvector_centrality_result_t struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @return                                 Result metadata (convergence status).
 */
eigenvector_centrality_result_t eigenvector_centrality_mask(const graph32_t& graph,
                                  const double* edge_weights,
                                  double* centralities,
                                  double epsilon,
                                  std::size_t max_iterations,
                                  const double* initial_centralities);

/**
 * Eigenvector Centrality - Double weights - Precomputed segments + edge mask variant.
 *
 * Same as eigenvector_centrality_seg (double weights) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Computes eigenvector centrality using power iteration with the formula
 * x_new = normalize((A + I) * x_old), where A is the adjacency matrix and
 * I is the identity matrix. The identity addition improves numerical stability
 * without changing the final eigenvector.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * @param graph                 Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param centralities          [out] Pre-allocated array of size num_vertices.
 * @param epsilon               Convergence tolerance. Converges when
 *                              sum(|x_new - x_old|) < num_vertices * epsilon.
 *                              Must be >= 0.0.
 * @param max_iterations        Maximum power iterations.
 * @param initial_centralities  [in] Initial values (size = num_vertices), or nullptr.
 *
 * eigenvector_centrality_result_t struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @return                                 Result metadata (convergence status).
 */
eigenvector_centrality_result_t eigenvector_centrality_seg_mask(const graph32_t& graph,
                                      const double* edge_weights,
                                      double* centralities,
                                      double epsilon,
                                      std::size_t max_iterations,
                                      const double* initial_centralities);

// =============================================================================
// Katz Centrality
// =============================================================================

/**
 * Katz Centrality - Unweighted.
 *
 * Computes Katz centrality with attenuation factor alpha.
 *
 * Preconditions (from cpp/src/centrality/katz_centrality_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, lines 53-54]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 61-62]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 63]
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * KatzResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Katz iteration (unweighted):
 *   x_new[v] = alpha * sum_{u in in_neighbors(v)} x_old[u] + beta_v
 *   where beta_v = betas[v] if betas != nullptr, else scalar beta.
 *
 * Convergence: converges when sum(|x_new - x_old|) < epsilon (global L1 norm).
 *
 * @param graph             Input graph. MUST be CSC (is_csc=true).
 * @param centralities      [out] Pre-allocated array of size num_vertices.
 * @param alpha             Attenuation factor. Must be in [0.0, 1.0].
 * @param beta              Constant added per vertex per iteration (used if betas is nullptr).
 * @param betas             [in] Per-vertex beta values (size = num_vertices), or nullptr for scalar beta.
 * @param epsilon           Convergence tolerance. Converges when
 *                          sum(|x_new - x_old|) < epsilon (global L1, not scaled by
 *                          num_vertices). Must be >= 0.0.
 * @param max_iterations    Maximum iterations.
 * @param has_initial_guess If true, use values in centralities as initial guess.
 * @param normalize         If true, normalize output by L2 norm.
 * @return                  Result struct with iterations count and convergence status.
 */
katz_centrality_result_t katz_centrality(const graph32_t& graph,
                     float* centralities,
                     float alpha,
                     float beta,
                     const float* betas,
                     float epsilon,
                     std::size_t max_iterations,
                     bool has_initial_guess,
                     bool normalize);

/**
 * Katz Centrality - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes Katz centrality with attenuation factor alpha.
 *
 * Preconditions (from cpp/src/centrality/katz_centrality_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, lines 53-54]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 61-62]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 63]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * KatzResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Katz iteration (unweighted):
 *   x_new[v] = alpha * sum_{u in in_neighbors(v)} x_old[u] + beta_v
 *   where beta_v = betas[v] if betas != nullptr, else scalar beta.
 *
 * Convergence: converges when sum(|x_new - x_old|) < epsilon (global L1 norm).
 *
 * @param graph             Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param centralities      [out] Pre-allocated array of size num_vertices.
 * @param alpha             Attenuation factor. Must be in [0.0, 1.0].
 * @param beta              Constant added per vertex per iteration (used if betas is nullptr).
 * @param betas             [in] Per-vertex beta values (size = num_vertices), or nullptr for scalar beta.
 * @param epsilon           Convergence tolerance. Converges when
 *                          sum(|x_new - x_old|) < epsilon (global L1, not scaled by
 *                          num_vertices). Must be >= 0.0.
 * @param max_iterations    Maximum iterations.
 * @param has_initial_guess If true, use values in centralities as initial guess.
 * @param normalize         If true, normalize output by L2 norm.
 * @return                  Result struct with iterations count and convergence status.
 */
katz_centrality_result_t katz_centrality_seg(const graph32_t& graph,
                         float* centralities,
                         float alpha,
                         float beta,
                         const float* betas,
                         float epsilon,
                         std::size_t max_iterations,
                         bool has_initial_guess,
                         bool normalize);

/**
 * Katz Centrality - Unweighted.
 *
 * Computes Katz centrality with attenuation factor alpha.
 *
 * Preconditions (from cpp/src/centrality/katz_centrality_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, lines 53-54]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 61-62]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 63]
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * KatzResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Katz iteration (unweighted):
 *   x_new[v] = alpha * sum_{u in in_neighbors(v)} w(u,v) * x_old[u] + beta_v
 *   where beta_v = betas[v] if betas != nullptr, else scalar beta.
 *
 * Convergence: converges when sum(|x_new - x_old|) < epsilon (global L1 norm).
 *
 * @param graph             Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights      [in] Edge weights (size = num_edges), float.
 * @param centralities      [out] Pre-allocated array of size num_vertices.
 * @param alpha             Attenuation factor. Must be in [0.0, 1.0].
 * @param beta              Constant added per vertex per iteration (used if betas is nullptr).
 * @param betas             [in] Per-vertex beta values (size = num_vertices), or nullptr for scalar beta.
 * @param epsilon           Convergence tolerance. Converges when
 *                          sum(|x_new - x_old|) < epsilon (global L1, not scaled by
 *                          num_vertices). Must be >= 0.0.
 * @param max_iterations    Maximum iterations.
 * @param has_initial_guess If true, use values in centralities as initial guess.
 * @param normalize         If true, normalize output by L2 norm.
 * @return                  Result struct with iterations count and convergence status.
 */
katz_centrality_result_t katz_centrality(const graph32_t& graph,
                     const float* edge_weights,
                     float* centralities,
                     float alpha,
                     float beta,
                     const float* betas,
                     float epsilon,
                     std::size_t max_iterations,
                     bool has_initial_guess,
                     bool normalize);

/**
 * Katz Centrality - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes Katz centrality with attenuation factor alpha.
 *
 * Preconditions (from cpp/src/centrality/katz_centrality_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, lines 53-54]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 61-62]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 63]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * KatzResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Katz iteration (unweighted):
 *   x_new[v] = alpha * sum_{u in in_neighbors(v)} w(u,v) * x_old[u] + beta_v
 *   where beta_v = betas[v] if betas != nullptr, else scalar beta.
 *
 * Convergence: converges when sum(|x_new - x_old|) < epsilon (global L1 norm).
 *
 * @param graph             Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param edge_weights      [in] Edge weights (size = num_edges), float.
 * @param centralities      [out] Pre-allocated array of size num_vertices.
 * @param alpha             Attenuation factor. Must be in [0.0, 1.0].
 * @param beta              Constant added per vertex per iteration (used if betas is nullptr).
 * @param betas             [in] Per-vertex beta values (size = num_vertices), or nullptr for scalar beta.
 * @param epsilon           Convergence tolerance. Converges when
 *                          sum(|x_new - x_old|) < epsilon (global L1, not scaled by
 *                          num_vertices). Must be >= 0.0.
 * @param max_iterations    Maximum iterations.
 * @param has_initial_guess If true, use values in centralities as initial guess.
 * @param normalize         If true, normalize output by L2 norm.
 * @return                  Result struct with iterations count and convergence status.
 */
katz_centrality_result_t katz_centrality_seg(const graph32_t& graph,
                         const float* edge_weights,
                         float* centralities,
                         float alpha,
                         float beta,
                         const float* betas,
                         float epsilon,
                         std::size_t max_iterations,
                         bool has_initial_guess,
                         bool normalize);

/**
 * Katz Centrality - Unweighted.
 *
 * Computes Katz centrality with attenuation factor alpha.
 *
 * Preconditions (from cpp/src/centrality/katz_centrality_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, lines 53-54]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 61-62]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 63]
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * KatzResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Katz iteration (unweighted):
 *   x_new[v] = alpha * sum_{u in in_neighbors(v)} w(u,v) * x_old[u] + beta_v
 *   where beta_v = betas[v] if betas != nullptr, else scalar beta.
 *
 * Convergence: converges when sum(|x_new - x_old|) < epsilon (global L1 norm).
 *
 * @param graph             Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights      [in] Edge weights (size = num_edges), double.
 * @param centralities      [out] Pre-allocated array of size num_vertices.
 * @param alpha             Attenuation factor. Must be in [0.0, 1.0].
 * @param beta              Constant added per vertex per iteration (used if betas is nullptr).
 * @param betas             [in] Per-vertex beta values (size = num_vertices), or nullptr for scalar beta.
 * @param epsilon           Convergence tolerance. Converges when
 *                          sum(|x_new - x_old|) < epsilon (global L1, not scaled by
 *                          num_vertices). Must be >= 0.0.
 * @param max_iterations    Maximum iterations.
 * @param has_initial_guess If true, use values in centralities as initial guess.
 * @param normalize         If true, normalize output by L2 norm.
 * @return                  Result struct with iterations count and convergence status.
 */
katz_centrality_result_t katz_centrality(const graph32_t& graph,
                     const double* edge_weights,
                     double* centralities,
                     double alpha,
                     double beta,
                     const double* betas,
                     double epsilon,
                     std::size_t max_iterations,
                     bool has_initial_guess,
                     bool normalize);

/**
 * Katz Centrality - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes Katz centrality with attenuation factor alpha.
 *
 * Preconditions (from cpp/src/centrality/katz_centrality_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, lines 53-54]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 61-62]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 63]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * KatzResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Katz iteration (unweighted):
 *   x_new[v] = alpha * sum_{u in in_neighbors(v)} w(u,v) * x_old[u] + beta_v
 *   where beta_v = betas[v] if betas != nullptr, else scalar beta.
 *
 * Convergence: converges when sum(|x_new - x_old|) < epsilon (global L1 norm).
 *
 * @param graph             Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param edge_weights      [in] Edge weights (size = num_edges), double.
 * @param centralities      [out] Pre-allocated array of size num_vertices.
 * @param alpha             Attenuation factor. Must be in [0.0, 1.0].
 * @param beta              Constant added per vertex per iteration (used if betas is nullptr).
 * @param betas             [in] Per-vertex beta values (size = num_vertices), or nullptr for scalar beta.
 * @param epsilon           Convergence tolerance. Converges when
 *                          sum(|x_new - x_old|) < epsilon (global L1, not scaled by
 *                          num_vertices). Must be >= 0.0.
 * @param max_iterations    Maximum iterations.
 * @param has_initial_guess If true, use values in centralities as initial guess.
 * @param normalize         If true, normalize output by L2 norm.
 * @return                  Result struct with iterations count and convergence status.
 */
katz_centrality_result_t katz_centrality_seg(const graph32_t& graph,
                         const double* edge_weights,
                         double* centralities,
                         double alpha,
                         double beta,
                         const double* betas,
                         double epsilon,
                         std::size_t max_iterations,
                         bool has_initial_guess,
                         bool normalize);

// -----------------------------------------------------------------------------
// Katz Centrality - Edge mask variants
// -----------------------------------------------------------------------------

/**
 * Katz Centrality - Unweighted - Edge mask variant.
 *
 * Same as katz_centrality but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Computes Katz centrality with attenuation factor alpha.
 *
 * Preconditions (from cpp/src/centrality/katz_centrality_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, lines 53-54]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 61-62]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 63]
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * KatzResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Katz iteration (unweighted):
 *   x_new[v] = alpha * sum_{u in in_neighbors(v)} x_old[u] + beta_v
 *   where beta_v = betas[v] if betas != nullptr, else scalar beta.
 *
 * Convergence: converges when sum(|x_new - x_old|) < epsilon (global L1 norm).
 *
 * @param graph             Input graph. MUST be CSC (is_csc=true).
 * @param centralities      [out] Pre-allocated array of size num_vertices.
 * @param alpha             Attenuation factor. Must be in [0.0, 1.0].
 * @param beta              Constant added per vertex per iteration (used if betas is nullptr).
 * @param betas             [in] Per-vertex beta values (size = num_vertices), or nullptr for scalar beta.
 * @param epsilon           Convergence tolerance. Converges when
 *                          sum(|x_new - x_old|) < epsilon (global L1, not scaled by
 *                          num_vertices). Must be >= 0.0.
 * @param max_iterations    Maximum iterations.
 * @param has_initial_guess If true, use values in centralities as initial guess.
 * @param normalize         If true, normalize output by L2 norm.
 * @return                  Result struct with iterations count and convergence status.
 */
katz_centrality_result_t katz_centrality_mask(const graph32_t& graph,
                           float* centralities,
                           float alpha,
                           float beta,
                           const float* betas,
                           float epsilon,
                           std::size_t max_iterations,
                           bool has_initial_guess,
                           bool normalize);

/**
 * Katz Centrality - Unweighted - Precomputed segments + edge mask variant.
 *
 * Same as katz_centrality_seg but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Computes Katz centrality with attenuation factor alpha.
 *
 * Preconditions (from cpp/src/centrality/katz_centrality_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, lines 53-54]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 61-62]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 63]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * KatzResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Katz iteration (unweighted):
 *   x_new[v] = alpha * sum_{u in in_neighbors(v)} x_old[u] + beta_v
 *   where beta_v = betas[v] if betas != nullptr, else scalar beta.
 *
 * Convergence: converges when sum(|x_new - x_old|) < epsilon (global L1 norm).
 *
 * @param graph             Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param centralities      [out] Pre-allocated array of size num_vertices.
 * @param alpha             Attenuation factor. Must be in [0.0, 1.0].
 * @param beta              Constant added per vertex per iteration (used if betas is nullptr).
 * @param betas             [in] Per-vertex beta values (size = num_vertices), or nullptr for scalar beta.
 * @param epsilon           Convergence tolerance. Converges when
 *                          sum(|x_new - x_old|) < epsilon (global L1, not scaled by
 *                          num_vertices). Must be >= 0.0.
 * @param max_iterations    Maximum iterations.
 * @param has_initial_guess If true, use values in centralities as initial guess.
 * @param normalize         If true, normalize output by L2 norm.
 * @return                  Result struct with iterations count and convergence status.
 */
katz_centrality_result_t katz_centrality_seg_mask(const graph32_t& graph,
                               float* centralities,
                               float alpha,
                               float beta,
                               const float* betas,
                               float epsilon,
                               std::size_t max_iterations,
                               bool has_initial_guess,
                               bool normalize);

/**
 * Katz Centrality - Unweighted - Edge mask variant.
 *
 * Same as katz_centrality but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Computes Katz centrality with attenuation factor alpha.
 *
 * Preconditions (from cpp/src/centrality/katz_centrality_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, lines 53-54]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 61-62]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 63]
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * KatzResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Katz iteration (unweighted):
 *   x_new[v] = alpha * sum_{u in in_neighbors(v)} w(u,v) * x_old[u] + beta_v
 *   where beta_v = betas[v] if betas != nullptr, else scalar beta.
 *
 * Convergence: converges when sum(|x_new - x_old|) < epsilon (global L1 norm).
 *
 * @param graph             Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights      [in] Edge weights (size = num_edges), float.
 * @param centralities      [out] Pre-allocated array of size num_vertices.
 * @param alpha             Attenuation factor. Must be in [0.0, 1.0].
 * @param beta              Constant added per vertex per iteration (used if betas is nullptr).
 * @param betas             [in] Per-vertex beta values (size = num_vertices), or nullptr for scalar beta.
 * @param epsilon           Convergence tolerance. Converges when
 *                          sum(|x_new - x_old|) < epsilon (global L1, not scaled by
 *                          num_vertices). Must be >= 0.0.
 * @param max_iterations    Maximum iterations.
 * @param has_initial_guess If true, use values in centralities as initial guess.
 * @param normalize         If true, normalize output by L2 norm.
 * @return                  Result struct with iterations count and convergence status.
 */
katz_centrality_result_t katz_centrality_mask(const graph32_t& graph,
                           const float* edge_weights,
                           float* centralities,
                           float alpha,
                           float beta,
                           const float* betas,
                           float epsilon,
                           std::size_t max_iterations,
                           bool has_initial_guess,
                           bool normalize);

/**
 * Katz Centrality - Unweighted - Precomputed segments + edge mask variant.
 *
 * Same as katz_centrality_seg but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Computes Katz centrality with attenuation factor alpha.
 *
 * Preconditions (from cpp/src/centrality/katz_centrality_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, lines 53-54]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 61-62]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 63]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * KatzResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Katz iteration (unweighted):
 *   x_new[v] = alpha * sum_{u in in_neighbors(v)} w(u,v) * x_old[u] + beta_v
 *   where beta_v = betas[v] if betas != nullptr, else scalar beta.
 *
 * Convergence: converges when sum(|x_new - x_old|) < epsilon (global L1 norm).
 *
 * @param graph             Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param edge_weights      [in] Edge weights (size = num_edges), float.
 * @param centralities      [out] Pre-allocated array of size num_vertices.
 * @param alpha             Attenuation factor. Must be in [0.0, 1.0].
 * @param beta              Constant added per vertex per iteration (used if betas is nullptr).
 * @param betas             [in] Per-vertex beta values (size = num_vertices), or nullptr for scalar beta.
 * @param epsilon           Convergence tolerance. Converges when
 *                          sum(|x_new - x_old|) < epsilon (global L1, not scaled by
 *                          num_vertices). Must be >= 0.0.
 * @param max_iterations    Maximum iterations.
 * @param has_initial_guess If true, use values in centralities as initial guess.
 * @param normalize         If true, normalize output by L2 norm.
 * @return                  Result struct with iterations count and convergence status.
 */
katz_centrality_result_t katz_centrality_seg_mask(const graph32_t& graph,
                               const float* edge_weights,
                               float* centralities,
                               float alpha,
                               float beta,
                               const float* betas,
                               float epsilon,
                               std::size_t max_iterations,
                               bool has_initial_guess,
                               bool normalize);

/**
 * Katz Centrality - Unweighted - Edge mask variant.
 *
 * Same as katz_centrality but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Computes Katz centrality with attenuation factor alpha.
 *
 * Preconditions (from cpp/src/centrality/katz_centrality_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, lines 53-54]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 61-62]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 63]
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * KatzResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Katz iteration (unweighted):
 *   x_new[v] = alpha * sum_{u in in_neighbors(v)} w(u,v) * x_old[u] + beta_v
 *   where beta_v = betas[v] if betas != nullptr, else scalar beta.
 *
 * Convergence: converges when sum(|x_new - x_old|) < epsilon (global L1 norm).
 *
 * @param graph             Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights      [in] Edge weights (size = num_edges), double.
 * @param centralities      [out] Pre-allocated array of size num_vertices.
 * @param alpha             Attenuation factor. Must be in [0.0, 1.0].
 * @param beta              Constant added per vertex per iteration (used if betas is nullptr).
 * @param betas             [in] Per-vertex beta values (size = num_vertices), or nullptr for scalar beta.
 * @param epsilon           Convergence tolerance. Converges when
 *                          sum(|x_new - x_old|) < epsilon (global L1, not scaled by
 *                          num_vertices). Must be >= 0.0.
 * @param max_iterations    Maximum iterations.
 * @param has_initial_guess If true, use values in centralities as initial guess.
 * @param normalize         If true, normalize output by L2 norm.
 * @return                  Result struct with iterations count and convergence status.
 */
katz_centrality_result_t katz_centrality_mask(const graph32_t& graph,
                           const double* edge_weights,
                           double* centralities,
                           double alpha,
                           double beta,
                           const double* betas,
                           double epsilon,
                           std::size_t max_iterations,
                           bool has_initial_guess,
                           bool normalize);

/**
 * Katz Centrality - Unweighted - Precomputed segments + edge mask variant.
 *
 * Same as katz_centrality_seg but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Computes Katz centrality with attenuation factor alpha.
 *
 * Preconditions (from cpp/src/centrality/katz_centrality_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, lines 53-54]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 61-62]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 63]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices are sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree vertices (degree >= 1024)
 *                               [1] to [2]: mid-degree vertices (32 <= degree < 1024)
 *                               [2] to [3]: low-degree vertices (1 <= degree < 32)
 *                               [3] to [4]: zero-degree vertices (isolated)
 *   const uint32_t* edge_mask     - Device pointer. Packed bitmask over edges.
 *                             Bit j of word i (edge_mask[j/32] >> (j%32) & 1) indicates
 *                             whether edge j is active (1) or masked out (0).
 *                             Size: ceil(number_of_edges / 32) uint32_t words.
 *                             MUST be non-null for this function variant.
 *
 * KatzResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Katz iteration (unweighted):
 *   x_new[v] = alpha * sum_{u in in_neighbors(v)} w(u,v) * x_old[u] + beta_v
 *   where beta_v = betas[v] if betas != nullptr, else scalar beta.
 *
 * Convergence: converges when sum(|x_new - x_old|) < epsilon (global L1 norm).
 *
 * @param graph             Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param edge_weights      [in] Edge weights (size = num_edges), double.
 * @param centralities      [out] Pre-allocated array of size num_vertices.
 * @param alpha             Attenuation factor. Must be in [0.0, 1.0].
 * @param beta              Constant added per vertex per iteration (used if betas is nullptr).
 * @param betas             [in] Per-vertex beta values (size = num_vertices), or nullptr for scalar beta.
 * @param epsilon           Convergence tolerance. Converges when
 *                          sum(|x_new - x_old|) < epsilon (global L1, not scaled by
 *                          num_vertices). Must be >= 0.0.
 * @param max_iterations    Maximum iterations.
 * @param has_initial_guess If true, use values in centralities as initial guess.
 * @param normalize         If true, normalize output by L2 norm.
 * @return                  Result struct with iterations count and convergence status.
 */
katz_centrality_result_t katz_centrality_seg_mask(const graph32_t& graph,
                               const double* edge_weights,
                               double* centralities,
                               double alpha,
                               double beta,
                               const double* betas,
                               double epsilon,
                               std::size_t max_iterations,
                               bool has_initial_guess,
                               bool normalize);

}  // namespace aai
