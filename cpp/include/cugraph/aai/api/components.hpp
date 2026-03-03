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
 * AAI Component Algorithms: Weakly Connected Components, Strongly Connected Components
 */
#pragma once

#include <cugraph/aai/compact_graph.hpp>

#include <cstddef>
#include <cstdint>

namespace aai {

// =============================================================================
// Weakly Connected Components
// =============================================================================

/**
 * Weakly Connected Components.
 *
 * Finds weakly connected components in an undirected or directed graph.
 * Treats directed edges as undirected for connectivity.
 *
 * Preconditions (from cpp/src/components/weakly_connected_components_impl.cuh:286-288):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, weakly_connected_components_impl.cuh:286-288]
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
 * @param graph      Input graph. MUST be CSR (is_csc=false) and symmetric.
 * @param components [out] Pre-allocated device array of size num_vertices.
 *                   components[v] = component ID. Vertices in the same WCC share the same ID.
 *                   The specific ID value is implementation-defined (not necessarily
 *                   the smallest or largest vertex in the component).
 */
void weakly_connected_components(const graph32_t& graph,
                                 int32_t* components);

/**
 * Weakly Connected Components - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Finds weakly connected components in an undirected or directed graph.
 * Treats directed edges as undirected for connectivity.
 *
 * Preconditions (from cpp/src/components/weakly_connected_components_impl.cuh:286-288):
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, weakly_connected_components_impl.cuh:286-288]
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
 * @param graph      Input graph. MUST be CSR (is_csc=false) and symmetric. MUST have segment_offsets.
 * @param components [out] Pre-allocated device array of size num_vertices.
 *                   components[v] = component ID. Vertices in the same WCC share the same ID.
 *                   The specific ID value is implementation-defined (not necessarily
 *                   the smallest or largest vertex in the component).
 */
void weakly_connected_components_seg(const graph32_t& graph,
                                     int32_t* components);

/**
 * Weakly Connected Components - Edge mask variant (no precomputed segments).
 *
 * Same as weakly_connected_components but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be symmetric (undirected)
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
 * @param graph      Input graph. MUST be CSR (is_csc=false) and symmetric.
 * @param components [out] Pre-allocated device array of size num_vertices.
 *                   components[v] = component ID. Vertices in the same WCC share the same ID.
 *                   The specific ID value is implementation-defined (not necessarily
 *                   the smallest or largest vertex in the component).
 */
void weakly_connected_components_mask(const graph32_t& graph,
                                      int32_t* components);

/**
 * Weakly Connected Components - Precomputed segments + edge mask variant.
 *
 * Same as weakly_connected_components_seg but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be symmetric (undirected)
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
 * @param graph      Input graph. MUST be CSR (is_csc=false) and symmetric. MUST have segment_offsets.
 * @param components [out] Pre-allocated device array of size num_vertices.
 *                   components[v] = component ID. Vertices in the same WCC share the same ID.
 *                   The specific ID value is implementation-defined (not necessarily
 *                   the smallest or largest vertex in the component).
 */
void weakly_connected_components_seg_mask(const graph32_t& graph,
                                          int32_t* components);

// =============================================================================
// Strongly Connected Components
// =============================================================================

/**
 * Strongly Connected Components.
 *
 * Finds strongly connected components in a directed graph.
 * Vertices are strongly connected if there is a path in both directions between them.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
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
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * @param graph      Input graph. MUST be CSR (is_csc=false).
 * @param components [out] Pre-allocated device array of size num_vertices.
 *                   components[v] = component ID. Vertices in the same SCC share the same ID.
 *                   The specific ID value is implementation-defined (not necessarily
 *                   the smallest or largest vertex in the component).
 */
void strongly_connected_components(const graph32_t& graph,
                                   int32_t* components);

/**
 * Strongly Connected Components - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Finds strongly connected components in a directed graph.
 * Vertices are strongly connected if there is a path in both directions between them.
 *
 * Preconditions:
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
 * @param graph      Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets.
 * @param components [out] Pre-allocated device array of size num_vertices.
 *                   components[v] = component ID. Vertices in the same SCC share the same ID.
 *                   The specific ID value is implementation-defined (not necessarily
 *                   the smallest or largest vertex in the component).
 */
void strongly_connected_components_seg(const graph32_t& graph,
                                       int32_t* components);

}  // namespace aai
