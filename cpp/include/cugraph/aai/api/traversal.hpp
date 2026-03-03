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
 * AAI Traversal Algorithms: BFS, SSSP, K-Hop Neighbors
 */
#pragma once

#include <cugraph/aai/compact_graph.hpp>
#include <cugraph/aai/types.hpp>

#include <cstddef>
#include <cstdint>

namespace aai {

// =============================================================================
// BFS - Breadth-First Search
// =============================================================================

/**
 * Breadth-First Search (32-bit vertex/edge types).
 *
 * Computes shortest hop distances from source vertices to all reachable vertices.
 * Works on directed or undirected graphs.
 *
 * Preconditions (from cpp/src/traversal/bfs_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [static_assert !is_storage_transposed, line 144-145]
 *   - sources != nullptr when n_sources > 0
 *       [CUGRAPH_EXPECTS, line 155-156]
 *   - n_sources > 0
 *       [CUGRAPH_EXPECTS, line 168-169]
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
 * @param distances     [out] Pre-allocated device array of size num_vertices.
 *                      distances[v] = hop count from nearest source, INT32_MAX if unreachable.
 * @param predecessors  [out] Pre-allocated device array of size num_vertices, or nullptr to skip.
 *                      predecessors[v] = parent in BFS tree, -1 if source or unreachable.
 * @param sources       [in] Device array of source vertex IDs.
 * @param n_sources     Number of source vertices. Must be > 0.
 * @param depth_limit   Max BFS depth. INT32_MAX = unlimited.
 */
void bfs(const graph32_t& graph,
         int32_t* distances,
         int32_t* predecessors,
         const int32_t* sources,
         std::size_t n_sources,
         int32_t depth_limit);

/**
 * Breadth-First Search (32-bit vertex/edge types) - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes shortest hop distances from source vertices to all reachable vertices.
 * Works on directed or undirected graphs.
 *
 * Preconditions (from cpp/src/traversal/bfs_impl.cuh):
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [static_assert !is_storage_transposed, line 144-145]
 *   - sources != nullptr when n_sources > 0
 *       [CUGRAPH_EXPECTS, line 155-156]
 *   - n_sources > 0
 *       [CUGRAPH_EXPECTS, line 168-169]
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
 * @param distances     [out] Pre-allocated device array of size num_vertices.
 *                      distances[v] = hop count from nearest source, INT32_MAX if unreachable.
 * @param predecessors  [out] Pre-allocated device array of size num_vertices, or nullptr to skip.
 *                      predecessors[v] = parent in BFS tree, -1 if source or unreachable.
 * @param sources       [in] Device array of source vertex IDs.
 * @param n_sources     Number of source vertices. Must be > 0.
 * @param depth_limit   Max BFS depth. INT32_MAX = unlimited.
 */
void bfs_seg(const graph32_t& graph,
             int32_t* distances,
             int32_t* predecessors,
             const int32_t* sources,
             std::size_t n_sources,
             int32_t depth_limit);

/**
 * Direction-Optimizing Breadth-First Search (32-bit vertex/edge types).
 *
 * Computes shortest hop distances from source vertices to all reachable vertices.
 * Optimized for large-diameter graphs with high average degree.
 * Requires symmetric (undirected) graph.
 *
 * Preconditions (from cpp/src/traversal/bfs_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [static_assert !is_storage_transposed, line 144-145]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS is_symmetric, line 172-174]
 *   - sources != nullptr when n_sources > 0
 *       [CUGRAPH_EXPECTS, line 155-156]
 *   - n_sources > 0
 *       [CUGRAPH_EXPECTS, line 168-169]
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
 * @param graph         Input graph. MUST be CSR (is_csc=false) AND symmetric.
 * @param distances     [out] Pre-allocated device array of size num_vertices.
 *                      distances[v] = hop count from nearest source, INT32_MAX if unreachable.
 * @param predecessors  [out] Pre-allocated device array of size num_vertices, or nullptr to skip.
 *                      predecessors[v] = parent in BFS tree, -1 if source or unreachable.
 * @param sources       [in] Device array of source vertex IDs.
 * @param n_sources     Number of source vertices. Must be > 0.
 * @param depth_limit   Max BFS depth. INT32_MAX = unlimited.
 */
void bfs_direction_optimizing(const graph32_t& graph,
                              int32_t* distances,
                              int32_t* predecessors,
                              const int32_t* sources,
                              std::size_t n_sources,
                              int32_t depth_limit);

/**
 * Direction-Optimizing Breadth-First Search (32-bit vertex/edge types) - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes shortest hop distances from source vertices to all reachable vertices.
 * Optimized for large-diameter graphs with high average degree.
 * Requires symmetric (undirected) graph.
 *
 * Preconditions (from cpp/src/traversal/bfs_impl.cuh):
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [static_assert !is_storage_transposed, line 144-145]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS is_symmetric, line 172-174]
 *   - sources != nullptr when n_sources > 0
 *       [CUGRAPH_EXPECTS, line 155-156]
 *   - n_sources > 0
 *       [CUGRAPH_EXPECTS, line 168-169]
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
 * @param graph         Input graph. MUST be CSR (is_csc=false) AND symmetric. MUST have segment_offsets.
 * @param distances     [out] Pre-allocated device array of size num_vertices.
 *                      distances[v] = hop count from nearest source, INT32_MAX if unreachable.
 * @param predecessors  [out] Pre-allocated device array of size num_vertices, or nullptr to skip.
 *                      predecessors[v] = parent in BFS tree, -1 if source or unreachable.
 * @param sources       [in] Device array of source vertex IDs.
 * @param n_sources     Number of source vertices. Must be > 0.
 * @param depth_limit   Max BFS depth. INT32_MAX = unlimited.
 */
void bfs_direction_optimizing_seg(const graph32_t& graph,
                                  int32_t* distances,
                                  int32_t* predecessors,
                                  const int32_t* sources,
                                  std::size_t n_sources,
                                  int32_t depth_limit);

// =============================================================================
// BFS - Edge mask variants
// =============================================================================

/**
 * BFS - Edge mask variant (no precomputed segments).
 *
 * Same as bfs but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from traversal.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSR format (is_csc=false)
 *   - sources != nullptr when n_sources > 0
 *   - n_sources > 0
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
 * @param graph         Input graph. MUST be CSR (is_csc=false).
 * @param distances     [out] Pre-allocated device array of size num_vertices.
 *                      distances[v] = hop count from nearest source, INT32_MAX if unreachable.
 * @param predecessors  [out] Pre-allocated device array of size num_vertices, or nullptr to skip.
 *                      predecessors[v] = parent in BFS tree, -1 if source or unreachable.
 * @param sources       [in] Device array of source vertex IDs.
 * @param n_sources     Number of source vertices. Must be > 0.
 * @param depth_limit   Max BFS depth. INT32_MAX = unlimited.
 */
void bfs_mask(const graph32_t& graph,
              int32_t* distances,
              int32_t* predecessors,
              const int32_t* sources,
              std::size_t n_sources,
              int32_t depth_limit);

/**
 * BFS - Precomputed segments + edge mask variant.
 *
 * Same as bfs_seg but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from traversal.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSR format (is_csc=false)
 *   - sources != nullptr when n_sources > 0
 *   - n_sources > 0
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
 * @param graph         Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets.
 * @param distances     [out] Pre-allocated device array of size num_vertices.
 *                      distances[v] = hop count from nearest source, INT32_MAX if unreachable.
 * @param predecessors  [out] Pre-allocated device array of size num_vertices, or nullptr to skip.
 *                      predecessors[v] = parent in BFS tree, -1 if source or unreachable.
 * @param sources       [in] Device array of source vertex IDs.
 * @param n_sources     Number of source vertices. Must be > 0.
 * @param depth_limit   Max BFS depth. INT32_MAX = unlimited.
 */
void bfs_seg_mask(const graph32_t& graph,
                  int32_t* distances,
                  int32_t* predecessors,
                  const int32_t* sources,
                  std::size_t n_sources,
                  int32_t depth_limit);

/**
 * Direction-Optimizing BFS - Edge mask variant (no precomputed segments).
 *
 * Same as bfs_direction_optimizing but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from traversal.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSR format (is_csc=false)
 *   - Graph MUST be symmetric (undirected)
 *   - sources != nullptr when n_sources > 0
 *   - n_sources > 0
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
 * @param graph         Input graph. MUST be CSR (is_csc=false) AND symmetric.
 * @param distances     [out] Pre-allocated device array of size num_vertices.
 *                      distances[v] = hop count from nearest source, INT32_MAX if unreachable.
 * @param predecessors  [out] Pre-allocated device array of size num_vertices, or nullptr to skip.
 *                      predecessors[v] = parent in BFS tree, -1 if source or unreachable.
 * @param sources       [in] Device array of source vertex IDs.
 * @param n_sources     Number of source vertices. Must be > 0.
 * @param depth_limit   Max BFS depth. INT32_MAX = unlimited.
 */
void bfs_direction_optimizing_mask(const graph32_t& graph,
                                   int32_t* distances,
                                   int32_t* predecessors,
                                   const int32_t* sources,
                                   std::size_t n_sources,
                                   int32_t depth_limit);

/**
 * Direction-Optimizing BFS - Precomputed segments + edge mask variant.
 *
 * Same as bfs_direction_optimizing_seg but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from traversal.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSR format (is_csc=false)
 *   - Graph MUST be symmetric (undirected)
 *   - sources != nullptr when n_sources > 0
 *   - n_sources > 0
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph
 *                               (= offsets[number_of_vertices] - offsets[0]).
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
 * @param graph         Input graph. MUST be CSR (is_csc=false) AND symmetric. MUST have segment_offsets.
 * @param distances     [out] Pre-allocated device array of size num_vertices.
 *                      distances[v] = hop count from nearest source, INT32_MAX if unreachable.
 * @param predecessors  [out] Pre-allocated device array of size num_vertices, or nullptr to skip.
 *                      predecessors[v] = parent in BFS tree, -1 if source or unreachable.
 * @param sources       [in] Device array of source vertex IDs.
 * @param n_sources     Number of source vertices. Must be > 0.
 * @param depth_limit   Max BFS depth. INT32_MAX = unlimited.
 */
void bfs_direction_optimizing_seg_mask(const graph32_t& graph,
                                       int32_t* distances,
                                       int32_t* predecessors,
                                       const int32_t* sources,
                                       std::size_t n_sources,
                                       int32_t depth_limit);

// =============================================================================
// SSSP - Single-Source Shortest Path
// =============================================================================

/**
 * Single-Source Shortest Path - Float weights.
 *
 * Computes shortest path distances from a single source vertex to all
 * reachable vertices using edge weights.
 *
 * Preconditions (from cpp/src/traversal/sssp_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [static_assert !is_storage_transposed, line 178-179]
 *   - source vertex must be valid (in range [0, num_vertices))
 *       [CUGRAPH_EXPECTS is_valid_vertex, line 194-195]
 *   - Edge weights should be non-negative (not enforced, but required for correctness)
 *       [CUGRAPH_EXPECTS in do_expensive_check, line 197-206]
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
 * @param edge_weights  [in] Edge weights array (size = num_edges). Must be non-negative.
 * @param source        Source vertex ID. Must be in [0, num_vertices).
 * @param distances     [out] Pre-allocated array of size num_vertices.
 * @param predecessors  [out] Pre-allocated array of size num_vertices, or nullptr.
 * @param cutoff        Maximum distance to search (exclusive: vertices with shortest-path
 *                      distance >= cutoff are unreachable). Use std::numeric_limits<float>::max() for unlimited.
 */
void sssp(const graph32_t& graph,
          const float* edge_weights,
          int32_t source,
          float* distances,
          int32_t* predecessors,
          float cutoff);

/**
 * Single-Source Shortest Path - Double weights.
 *
 * Computes shortest path distances from a single source vertex to all
 * reachable vertices using edge weights.
 *
 * Preconditions (from cpp/src/traversal/sssp_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [static_assert !is_storage_transposed, line 178-179]
 *   - source vertex must be valid (in range [0, num_vertices))
 *       [CUGRAPH_EXPECTS is_valid_vertex, line 194-195]
 *   - Edge weights should be non-negative (not enforced, but required for correctness)
 *       [CUGRAPH_EXPECTS in do_expensive_check, line 197-206]
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
 * @param edge_weights  [in] Edge weights array (size = num_edges). Must be non-negative.
 * @param source        Source vertex ID. Must be in [0, num_vertices).
 * @param distances     [out] Pre-allocated array of size num_vertices.
 * @param predecessors  [out] Pre-allocated array of size num_vertices, or nullptr.
 * @param cutoff        Maximum distance to search (exclusive: vertices with shortest-path
 *                      distance >= cutoff are unreachable). Use std::numeric_limits<double>::max() for unlimited.
 */
void sssp(const graph32_t& graph,
          const double* edge_weights,
          int32_t source,
          double* distances,
          int32_t* predecessors,
          double cutoff);

/**
 * Single-Source Shortest Path - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes shortest path distances from a single source vertex to all
 * reachable vertices using edge weights.
 *
 * Preconditions (from cpp/src/traversal/sssp_impl.cuh):
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [static_assert !is_storage_transposed, line 178-179]
 *   - source vertex must be valid (in range [0, num_vertices))
 *       [CUGRAPH_EXPECTS is_valid_vertex, line 194-195]
 *   - Edge weights should be non-negative (not enforced, but required for correctness)
 *       [CUGRAPH_EXPECTS in do_expensive_check, line 197-206]
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
 * @param edge_weights  [in] Edge weights array (size = num_edges). Must be non-negative.
 * @param source        Source vertex ID. Must be in [0, num_vertices).
 * @param distances     [out] Pre-allocated array of size num_vertices.
 * @param predecessors  [out] Pre-allocated array of size num_vertices, or nullptr.
 * @param cutoff        Maximum distance to search (exclusive: vertices with shortest-path
 *                      distance >= cutoff are unreachable). Use std::numeric_limits<float>::max() for unlimited.
 */
void sssp_seg(const graph32_t& graph,
              const float* edge_weights,
              int32_t source,
              float* distances,
              int32_t* predecessors,
              float cutoff);

/**
 * Single-Source Shortest Path - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes shortest path distances from a single source vertex to all
 * reachable vertices using edge weights.
 *
 * Preconditions (from cpp/src/traversal/sssp_impl.cuh):
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [static_assert !is_storage_transposed, line 178-179]
 *   - source vertex must be valid (in range [0, num_vertices))
 *       [CUGRAPH_EXPECTS is_valid_vertex, line 194-195]
 *   - Edge weights should be non-negative (not enforced, but required for correctness)
 *       [CUGRAPH_EXPECTS in do_expensive_check, line 197-206]
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
 * @param edge_weights  [in] Edge weights array (size = num_edges). Must be non-negative.
 * @param source        Source vertex ID. Must be in [0, num_vertices).
 * @param distances     [out] Pre-allocated array of size num_vertices.
 * @param predecessors  [out] Pre-allocated array of size num_vertices, or nullptr.
 * @param cutoff        Maximum distance to search (exclusive: vertices with shortest-path
 *                      distance >= cutoff are unreachable). Use std::numeric_limits<double>::max() for unlimited.
 */
void sssp_seg(const graph32_t& graph,
              const double* edge_weights,
              int32_t source,
              double* distances,
              int32_t* predecessors,
              double cutoff);

// =============================================================================
// SSSP - Edge mask variants
// =============================================================================

/**
 * SSSP - Float weights - Edge mask variant (no precomputed segments).
 *
 * Same as sssp (float) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSR format (is_csc=false)
 *   - source vertex must be valid (in range [0, num_vertices))
 *   - Edge weights should be non-negative
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
 * @param graph         Input graph. MUST be CSR (is_csc=false).
 * @param edge_weights  [in] Edge weights array (size = num_edges). Must be non-negative.
 * @param source        Source vertex ID. Must be in [0, num_vertices).
 * @param distances     [out] Pre-allocated array of size num_vertices.
 * @param predecessors  [out] Pre-allocated array of size num_vertices, or nullptr.
 * @param cutoff        Maximum distance to search (exclusive: vertices with shortest-path
 *                      distance >= cutoff are unreachable). Use std::numeric_limits<float>::max() for unlimited.
 */
void sssp_mask(const graph32_t& graph,
               const float* edge_weights,
               int32_t source,
               float* distances,
               int32_t* predecessors,
               float cutoff);

/**
 * SSSP - Float weights - Precomputed segments + edge mask variant.
 *
 * Same as sssp_seg (float) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSR format (is_csc=false)
 *   - source vertex must be valid (in range [0, num_vertices))
 *   - Edge weights should be non-negative
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
 * @param graph         Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets.
 * @param edge_weights  [in] Edge weights array (size = num_edges). Must be non-negative.
 * @param source        Source vertex ID. Must be in [0, num_vertices).
 * @param distances     [out] Pre-allocated array of size num_vertices.
 * @param predecessors  [out] Pre-allocated array of size num_vertices, or nullptr.
 * @param cutoff        Maximum distance to search (exclusive: vertices with shortest-path
 *                      distance >= cutoff are unreachable). Use std::numeric_limits<float>::max() for unlimited.
 */
void sssp_seg_mask(const graph32_t& graph,
                   const float* edge_weights,
                   int32_t source,
                   float* distances,
                   int32_t* predecessors,
                   float cutoff);

/**
 * SSSP - Double weights - Edge mask variant (no precomputed segments).
 *
 * Same as sssp (double) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSR format (is_csc=false)
 *   - source vertex must be valid (in range [0, num_vertices))
 *   - Edge weights should be non-negative
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
 * @param graph         Input graph. MUST be CSR (is_csc=false).
 * @param edge_weights  [in] Edge weights array (size = num_edges). Must be non-negative.
 * @param source        Source vertex ID. Must be in [0, num_vertices).
 * @param distances     [out] Pre-allocated array of size num_vertices.
 * @param predecessors  [out] Pre-allocated array of size num_vertices, or nullptr.
 * @param cutoff        Maximum distance to search (exclusive: vertices with shortest-path
 *                      distance >= cutoff are unreachable). Use std::numeric_limits<double>::max() for unlimited.
 */
void sssp_mask(const graph32_t& graph,
               const double* edge_weights,
               int32_t source,
               double* distances,
               int32_t* predecessors,
               double cutoff);

/**
 * SSSP - Double weights - Precomputed segments + edge mask variant.
 *
 * Same as sssp_seg (double) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSR format (is_csc=false)
 *   - source vertex must be valid (in range [0, num_vertices))
 *   - Edge weights should be non-negative
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
 * @param graph         Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets.
 * @param edge_weights  [in] Edge weights array (size = num_edges). Must be non-negative.
 * @param source        Source vertex ID. Must be in [0, num_vertices).
 * @param distances     [out] Pre-allocated array of size num_vertices.
 * @param predecessors  [out] Pre-allocated array of size num_vertices, or nullptr.
 * @param cutoff        Maximum distance to search (exclusive: vertices with shortest-path
 *                      distance >= cutoff are unreachable). Use std::numeric_limits<double>::max() for unlimited.
 */
void sssp_seg_mask(const graph32_t& graph,
                   const double* edge_weights,
                   int32_t source,
                   double* distances,
                   int32_t* predecessors,
                   double cutoff);

// =============================================================================
// K-Hop Neighbors
// =============================================================================

/**
 * K-Hop Neighbors.
 *
 * Finds all vertices reachable in exactly k hops from each start vertex.
 * Note: Returns vertices at distance exactly k, not vertices at distance <= k.
 * Used by two_hop_neighbors (with k=2).
 *
 * Preconditions (from cpp/src/traversal/k_hop_nbrs_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [static_assert !is_storage_transposed, line 75-76]
 *   - k > 0
 *       [CUGRAPH_EXPECTS, line 99]
 *   - num_start_vertices > 0
 *       [CUGRAPH_EXPECTS, line 96-97]
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
 * k_hop_nbrs_result_t struct (return type):
 *   std::size_t* offsets    - Device pointer. Size: num_start_vertices + 1.
 *                             offsets[i] = start index in neighbors for start_vertex i.
 *   int32_t* neighbors      - Device pointer. Neighbor vertex IDs at exactly k hops.
 *   std::size_t num_offsets - Size of offsets array (num_start_vertices + 1).
 *   std::size_t num_neighbors - Total number of neighbors found.
 *   Caller MUST free offsets and neighbors with cudaFree after use.
 *
 * @param graph              Input graph. MUST be CSR (is_csc=false).
 * @param start_vertices     [in] Device array of starting vertex IDs.
 * @param num_start_vertices Number of starting vertices. Must be > 0.
 * @param k                  Number of hops (e.g., 2 for two-hop neighbors). Must be > 0.
 * @return                   Result struct with offsets and neighbors arrays.
 *                           Caller MUST free result.offsets and result.neighbors with cudaFree.
 */
k_hop_nbrs_result_t k_hop_nbrs(const graph32_t& graph,
                               const int32_t* start_vertices,
                               std::size_t num_start_vertices,
                               std::size_t k);

/**
 * K-Hop Neighbors - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Finds all vertices reachable in exactly k hops from each start vertex.
 * Note: Returns vertices at distance exactly k, not vertices at distance <= k.
 * Used by two_hop_neighbors (with k=2).
 *
 * Preconditions (from cpp/src/traversal/k_hop_nbrs_impl.cuh):
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [static_assert !is_storage_transposed, line 75-76]
 *   - k > 0
 *       [CUGRAPH_EXPECTS, line 99]
 *   - num_start_vertices > 0
 *       [CUGRAPH_EXPECTS, line 96-97]
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
 * k_hop_nbrs_result_t struct (return type):
 *   std::size_t* offsets    - Device pointer. Size: num_start_vertices + 1.
 *                             offsets[i] = start index in neighbors for start_vertex i.
 *   int32_t* neighbors      - Device pointer. Neighbor vertex IDs at exactly k hops.
 *   std::size_t num_offsets - Size of offsets array (num_start_vertices + 1).
 *   std::size_t num_neighbors - Total number of neighbors found.
 *   Caller MUST free offsets and neighbors with cudaFree after use.
 *
 * @param graph              Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets.
 * @param start_vertices     [in] Device array of starting vertex IDs.
 * @param num_start_vertices Number of starting vertices. Must be > 0.
 * @param k                  Number of hops (e.g., 2 for two-hop neighbors). Must be > 0.
 * @return                   Result struct with offsets and neighbors arrays.
 *                           Caller MUST free result.offsets and result.neighbors with cudaFree.
 */
k_hop_nbrs_result_t k_hop_nbrs_seg(const graph32_t& graph,
                                   const int32_t* start_vertices,
                                   std::size_t num_start_vertices,
                                   std::size_t k);

// =============================================================================
// K-Hop Neighbors - Edge mask variants
// =============================================================================

/**
 * K-Hop Neighbors - Edge mask variant (no precomputed segments).
 *
 * Same as k_hop_nbrs but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from traversal.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSR format (is_csc=false)
 *   - k > 0
 *   - num_start_vertices > 0
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
 * k_hop_nbrs_result_t struct (return type):
 *   std::size_t* offsets    - Device pointer. Size: num_start_vertices + 1.
 *                             offsets[i] = start index in neighbors for start_vertex i.
 *   int32_t* neighbors      - Device pointer. Neighbor vertex IDs at exactly k hops.
 *   std::size_t num_offsets - Size of offsets array (num_start_vertices + 1).
 *   std::size_t num_neighbors - Total number of neighbors found.
 *   Caller MUST free offsets and neighbors with cudaFree after use.
 *
 * @param graph              Input graph. MUST be CSR (is_csc=false).
 * @param start_vertices     [in] Device array of starting vertex IDs.
 * @param num_start_vertices Number of starting vertices. Must be > 0.
 * @param k                  Number of hops (e.g., 2 for two-hop neighbors). Must be > 0.
 * @return                   Result struct with offsets and neighbors arrays.
 *                           Caller MUST free result.offsets and result.neighbors with cudaFree.
 */
k_hop_nbrs_result_t k_hop_nbrs_mask(const graph32_t& graph,
                                    const int32_t* start_vertices,
                                    std::size_t num_start_vertices,
                                    std::size_t k);

/**
 * K-Hop Neighbors - Precomputed segments + edge mask variant.
 *
 * Same as k_hop_nbrs_seg but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from traversal.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSR format (is_csc=false)
 *   - k > 0
 *   - num_start_vertices > 0
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
 * k_hop_nbrs_result_t struct (return type):
 *   std::size_t* offsets    - Device pointer. Size: num_start_vertices + 1.
 *                             offsets[i] = start index in neighbors for start_vertex i.
 *   int32_t* neighbors      - Device pointer. Neighbor vertex IDs at exactly k hops.
 *   std::size_t num_offsets - Size of offsets array (num_start_vertices + 1).
 *   std::size_t num_neighbors - Total number of neighbors found.
 *   Caller MUST free offsets and neighbors with cudaFree after use.
 *
 * @param graph              Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets.
 * @param start_vertices     [in] Device array of starting vertex IDs.
 * @param num_start_vertices Number of starting vertices. Must be > 0.
 * @param k                  Number of hops (e.g., 2 for two-hop neighbors). Must be > 0.
 * @return                   Result struct with offsets and neighbors arrays.
 *                           Caller MUST free result.offsets and result.neighbors with cudaFree.
 */
k_hop_nbrs_result_t k_hop_nbrs_seg_mask(const graph32_t& graph,
                                        const int32_t* start_vertices,
                                        std::size_t num_start_vertices,
                                        std::size_t k);

}  // namespace aai
