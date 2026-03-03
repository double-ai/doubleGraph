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
 * AAI Core Algorithms: Core Number, K-Core
 */
#pragma once

#include <cugraph/aai/compact_graph.hpp>

#include <cstddef>
#include <cstdint>

namespace aai {

// =============================================================================
// Core Number
// =============================================================================

/**
 * Core Number.
 *
 * Computes the core number (k-core decomposition) for each vertex.
 * The core number of a vertex is the largest k such that the vertex belongs to the k-core
 * (maximal subgraph where all vertices have degree at least k).
 *
 * Preconditions (from cpp/src/cores/core_number_impl.cuh:70-78):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:70-71]
 *   - Graph MUST NOT be a multigraph
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:72-73]
 *   - degree_type MUST be IN (0), OUT (1), or INOUT (2)
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:74-77]
 *   - k_first MUST be <= k_last
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:78]
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
 * @param graph        Input graph. MUST be CSR (is_csc=false).
 * @param core_numbers [out] Pre-allocated device array of size num_vertices.
 *                     core_numbers[v] = core number of vertex v.
 * @param degree_type  Degree type: 0=in, 1=out, 2=inout.
 * @param k_first      Only compute k-cores for k >= k_first. Vertices with core number < k_first
 *                     will have their core numbers set to 0.
 * @param k_last       Only compute k-cores for k <= k_last. Algorithm stops after reaching k_last.
 */
void core_number(const graph32_t& graph,
                 int32_t* core_numbers,
                 int degree_type,
                 std::size_t k_first,
                 std::size_t k_last);

/**
 * Core Number - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes the core number (k-core decomposition) for each vertex.
 * The core number of a vertex is the largest k such that the vertex belongs to the k-core
 * (maximal subgraph where all vertices have degree at least k).
 *
 * Preconditions (from cpp/src/cores/core_number_impl.cuh:70-78):
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:70-71]
 *   - Graph MUST NOT be a multigraph
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:72-73]
 *   - degree_type MUST be IN (0), OUT (1), or INOUT (2)
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:74-77]
 *   - k_first MUST be <= k_last
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:78]
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
 * @param graph        Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets.
 * @param core_numbers [out] Pre-allocated device array of size num_vertices.
 *                     core_numbers[v] = core number of vertex v.
 * @param degree_type  Degree type: 0=in, 1=out, 2=inout.
 * @param k_first      Only compute k-cores for k >= k_first. Vertices with core number < k_first
 *                     will have their core numbers set to 0.
 * @param k_last       Only compute k-cores for k <= k_last. Algorithm stops after reaching k_last.
 */
void core_number_seg(const graph32_t& graph,
                     int32_t* core_numbers,
                     int degree_type,
                     std::size_t k_first,
                     std::size_t k_last);

/**
 * Core Number - Edge mask variant.
 *
 * Same as core_number but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Computes the core number (k-core decomposition) for each vertex.
 * The core number of a vertex is the largest k such that the vertex belongs to the k-core
 * (maximal subgraph where all vertices have degree at least k).
 *
 * Preconditions (from cpp/src/cores/core_number_impl.cuh:70-78):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:70-71]
 *   - Graph MUST NOT be a multigraph
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:72-73]
 *   - degree_type MUST be IN (0), OUT (1), or INOUT (2)
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:74-77]
 *   - k_first MUST be <= k_last
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:78]
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
 * @param graph        Input graph. MUST be CSR (is_csc=false). MUST have edge_mask.
 * @param core_numbers [out] Pre-allocated device array of size num_vertices.
 *                     core_numbers[v] = core number of vertex v.
 * @param degree_type  Degree type: 0=in, 1=out, 2=inout.
 * @param k_first      Only compute k-cores for k >= k_first. Vertices with core number < k_first
 *                     will have their core numbers set to 0.
 * @param k_last       Only compute k-cores for k <= k_last. Algorithm stops after reaching k_last.
 */
void core_number_mask(const graph32_t& graph,
                      int32_t* core_numbers,
                      int degree_type,
                      std::size_t k_first,
                      std::size_t k_last);

/**
 * Core Number - Precomputed segments + edge mask variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Same as core_number_seg but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Computes the core number (k-core decomposition) for each vertex.
 * The core number of a vertex is the largest k such that the vertex belongs to the k-core
 * (maximal subgraph where all vertices have degree at least k).
 *
 * Preconditions (from cpp/src/cores/core_number_impl.cuh:70-78):
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:70-71]
 *   - Graph MUST NOT be a multigraph
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:72-73]
 *   - degree_type MUST be IN (0), OUT (1), or INOUT (2)
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:74-77]
 *   - k_first MUST be <= k_last
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:78]
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
 * @param graph        Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets
 *                     and edge_mask.
 * @param core_numbers [out] Pre-allocated device array of size num_vertices.
 *                     core_numbers[v] = core number of vertex v.
 * @param degree_type  Degree type: 0=in, 1=out, 2=inout.
 * @param k_first      Only compute k-cores for k >= k_first. Vertices with core number < k_first
 *                     will have their core numbers set to 0.
 * @param k_last       Only compute k-cores for k <= k_last. Algorithm stops after reaching k_last.
 */
void core_number_seg_mask(const graph32_t& graph,
                          int32_t* core_numbers,
                          int degree_type,
                          std::size_t k_first,
                          std::size_t k_last);

// =============================================================================
// K-Core
// =============================================================================

/**
 * K-Core - Unweighted.
 *
 * Finds the k-core subgraph: the maximal subgraph where all vertices have
 * degree >= k. Returns the edge list of the k-core subgraph.
 *
 * Self-loops: Input graphs may contain self-loops (edges where src == dst).
 * Self-loops MUST be excluded when computing vertex degrees for the
 * k-core peeling process (i.e., an edge (v, v) does not contribute to
 * the degree of v). However, self-loops ARE included in the output edge list
 * (the extracted K-Core subgraph retains self-loops from the original graph).
 *
 * Preconditions (from k_core_impl.cuh, core_number_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:70-71]
 *   - Graph MUST NOT be a multigraph
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:72-73]
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
 * @param graph         Input graph. MUST be CSR (is_csc=false).
 * @param k             Order of the core. Must be >= 0.
 * @param degree_type   Degree type: -1=use core_numbers, 0=in, 1=out, 2=inout.
 *                      If core_numbers is provided, this is ignored.
 *                      For degree_type=2 (INOUT), the degree of a vertex v is
 *                      computed as in_degree(v) + out_degree(v), excluding
 *                      self-loops. On a symmetric graph without edge masks,
 *                      in_degree == out_degree so INOUT == 2 * out_degree.
 * @param core_numbers  [in] Optional pre-computed core numbers (from core_number()).
 *                      Can be nullptr, in which case they are computed internally.
 * @param edge_srcs     [out] Pre-allocated array for source vertices (size = max_edges).
 * @param edge_dsts     [out] Pre-allocated array for destination vertices (size = max_edges).
 * @param max_edges     Maximum number of edges that can be written to output arrays.
 * @return              Actual number of edges in the k-core subgraph.
 */
std::size_t k_core(const graph32_t& graph,
                   std::size_t k,
                   int degree_type,
                   const int32_t* core_numbers,
                   int32_t* edge_srcs,
                   int32_t* edge_dsts,
                   std::size_t max_edges);

/**
 * K-Core - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Finds the k-core subgraph: the maximal subgraph where all vertices have
 * degree >= k. Returns the edge list of the k-core subgraph.
 *
 * Self-loops: Input graphs may contain self-loops (edges where src == dst).
 * Self-loops MUST be excluded when computing vertex degrees for the
 * k-core peeling process (i.e., an edge (v, v) does not contribute to
 * the degree of v). However, self-loops ARE included in the output edge list
 * (the extracted K-Core subgraph retains self-loops from the original graph).
 *
 * Preconditions (from k_core_impl.cuh, core_number_impl.cuh):
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:70-71]
 *   - Graph MUST NOT be a multigraph
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:72-73]
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
 * @param graph         Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets.
 * @param k             Order of the core. Must be >= 0.
 * @param degree_type   Degree type: -1=use core_numbers, 0=in, 1=out, 2=inout.
 *                      If core_numbers is provided, this is ignored.
 *                      For degree_type=2 (INOUT), the degree of a vertex v is
 *                      computed as in_degree(v) + out_degree(v), excluding
 *                      self-loops. On a symmetric graph without edge masks,
 *                      in_degree == out_degree so INOUT == 2 * out_degree.
 * @param core_numbers  [in] Optional pre-computed core numbers (from core_number()).
 *                      Can be nullptr, in which case they are computed internally.
 * @param edge_srcs     [out] Pre-allocated array for source vertices (size = max_edges).
 * @param edge_dsts     [out] Pre-allocated array for destination vertices (size = max_edges).
 * @param max_edges     Maximum number of edges that can be written to output arrays.
 * @return              Actual number of edges in the k-core subgraph.
 */
std::size_t k_core_seg(const graph32_t& graph,
                       std::size_t k,
                       int degree_type,
                       const int32_t* core_numbers,
                       int32_t* edge_srcs,
                       int32_t* edge_dsts,
                       std::size_t max_edges);

/**
 * K-Core - Unweighted - Edge mask variant.
 *
 * Same as k_core but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Finds the k-core subgraph: the maximal subgraph where all vertices have
 * degree >= k. Returns the edge list of the k-core subgraph.
 *
 * Self-loops: Input graphs may contain self-loops (edges where src == dst).
 * Self-loops MUST be excluded when computing vertex degrees for the
 * k-core peeling process (i.e., an edge (v, v) does not contribute to
 * the degree of v). However, self-loops ARE included in the output edge list
 * (the extracted K-Core subgraph retains self-loops from the original graph).
 *
 * Preconditions (from k_core_impl.cuh, core_number_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:70-71]
 *   - Graph MUST NOT be a multigraph
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:72-73]
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
 * @param graph         Input graph. MUST be CSR (is_csc=false). MUST have edge_mask.
 * @param k             Order of the core. Must be >= 0.
 * @param degree_type   Degree type: -1=use core_numbers, 0=in, 1=out, 2=inout.
 *                      If core_numbers is provided, this is ignored.
 *                      For degree_type=2 (INOUT) on a symmetric graph, the
 *                      degree is always computed as 2 * out_degree (excluding
 *                      self-loops), even when an asymmetric edge mask is
 *                      applied. The peeling delta is also 2 (each removed
 *                      neighbor decreases the core number estimate by 2).
 *                      Core numbers for INOUT on symmetric graphs are always
 *                      even.
 * @param core_numbers  [in] Optional pre-computed core numbers (from core_number()).
 *                      Can be nullptr, in which case they are computed internally.
 * @param edge_srcs     [out] Pre-allocated array for source vertices
 *                      (size = max_edges).
 * @param edge_dsts     [out] Pre-allocated array for destination vertices
 *                      (size = max_edges).
 * @param max_edges     Maximum number of edges that can be written to output arrays.
 * @return              Actual number of edges in the k-core subgraph.
 */
std::size_t k_core_mask(const graph32_t& graph,
                        std::size_t k,
                        int degree_type,
                        const int32_t* core_numbers,
                        int32_t* edge_srcs,
                        int32_t* edge_dsts,
                        std::size_t max_edges);

/**
 * K-Core - Unweighted - Precomputed segments + edge mask variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Same as k_core_seg but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Finds the k-core subgraph: the maximal subgraph where all vertices have
 * degree >= k. Returns the edge list of the k-core subgraph.
 *
 * Self-loops: Input graphs may contain self-loops (edges where src == dst).
 * Self-loops MUST be excluded when computing vertex degrees for the
 * k-core peeling process (i.e., an edge (v, v) does not contribute to
 * the degree of v). However, self-loops ARE included in the output edge list
 * (the extracted K-Core subgraph retains self-loops from the original graph).
 *
 * Preconditions (from k_core_impl.cuh, core_number_impl.cuh):
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:70-71]
 *   - Graph MUST NOT be a multigraph
 *       [CUGRAPH_EXPECTS, core_number_impl.cuh:72-73]
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
 * @param graph         Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets
 *                      and edge_mask.
 * @param k             Order of the core. Must be >= 0.
 * @param degree_type   Degree type: -1=use core_numbers, 0=in, 1=out, 2=inout.
 *                      If core_numbers is provided, this is ignored.
 *                      For degree_type=2 (INOUT) on a symmetric graph, the
 *                      degree is always computed as 2 * out_degree (excluding
 *                      self-loops), even when an asymmetric edge mask is
 *                      applied. The peeling delta is also 2 (each removed
 *                      neighbor decreases the core number estimate by 2).
 *                      Core numbers for INOUT on symmetric graphs are always
 *                      even.
 * @param core_numbers  [in] Optional pre-computed core numbers (from core_number()).
 *                      Can be nullptr, in which case they are computed internally.
 * @param edge_srcs     [out] Pre-allocated array for source vertices (size = max_edges).
 * @param edge_dsts     [out] Pre-allocated array for destination vertices (size = max_edges).
 * @param max_edges     Maximum number of edges that can be written to output arrays.
 * @return              Actual number of edges in the k-core subgraph.
 */
std::size_t k_core_seg_mask(const graph32_t& graph,
                            std::size_t k,
                            int degree_type,
                            const int32_t* core_numbers,
                            int32_t* edge_srcs,
                            int32_t* edge_dsts,
                            std::size_t max_edges);

}  // namespace aai
