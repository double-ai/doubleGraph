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
 * AAI Tree Algorithms: Minimum Spanning Tree
 */
#pragma once

#include <cugraph/aai/compact_graph.hpp>

#include <cstddef>
#include <cstdint>

namespace aai {

// =============================================================================
// Minimum Spanning Tree
// =============================================================================

/**
 * Minimum Spanning Tree - Float weights.
 *
 * Computes a minimum spanning tree (or forest if graph is disconnected).
 *
 * Note: MST inherently requires edge weights to define "minimum".
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST NOT be a multigraph
 *       [RAFT Boruvka MST algorithm not designed for multigraphs]
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
 * @param graph         Input graph. MUST be symmetric (undirected). MUST be CSR (is_csc=false).
 *                      MUST NOT be a multigraph.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param mst_srcs      [out] Pre-allocated device array for MST edge sources.
 *                      Size must be >= 2 * (num_vertices - 1). Output is
 *                      symmetrized: each undirected MST edge appears in both
 *                      directions (u→v and v→u).
 * @param mst_dsts      [out] Pre-allocated device array for MST edge destinations.
 *                      Size must be >= 2 * (num_vertices - 1).
 * @param mst_weights   [out] Pre-allocated device array for MST edge weights.
 *                      Size must be >= 2 * (num_vertices - 1).
 * @param max_edges     Maximum number of edges that can be written to output arrays.
 * @return              Number of directed edges written (= 2 * number of
 *                      undirected MST edges). For a connected graph, this
 *                      is 2 * (num_vertices - 1).
 */
std::size_t minimum_spanning_tree(const graph32_t& graph,
                                  const float* edge_weights,
                                  int32_t* mst_srcs,
                                  int32_t* mst_dsts,
                                  float* mst_weights,
                                  std::size_t max_edges);

/**
 * Minimum Spanning Tree - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes a minimum spanning tree (or forest if graph is disconnected).
 *
 * Note: MST inherently requires edge weights to define "minimum".
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST NOT be a multigraph
 *       [RAFT Boruvka MST algorithm not designed for multigraphs]
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
 * @param graph         Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 *                      MUST NOT be a multigraph.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param mst_srcs      [out] Pre-allocated device array for MST edge sources.
 *                      Size must be >= 2 * (num_vertices - 1). Output is
 *                      symmetrized: each undirected MST edge appears in both
 *                      directions (u→v and v→u).
 * @param mst_dsts      [out] Pre-allocated device array for MST edge destinations.
 *                      Size must be >= 2 * (num_vertices - 1).
 * @param mst_weights   [out] Pre-allocated device array for MST edge weights.
 *                      Size must be >= 2 * (num_vertices - 1).
 * @param max_edges     Maximum number of edges that can be written to output arrays.
 * @return              Number of directed edges written (= 2 * number of
 *                      undirected MST edges). For a connected graph, this
 *                      is 2 * (num_vertices - 1).
 */
std::size_t minimum_spanning_tree_seg(const graph32_t& graph,
                                      const float* edge_weights,
                                      int32_t* mst_srcs,
                                      int32_t* mst_dsts,
                                      float* mst_weights,
                                      std::size_t max_edges);

/**
 * Minimum Spanning Tree - Double weights.
 *
 * Computes a minimum spanning tree (or forest if graph is disconnected).
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST NOT be a multigraph
 *       [RAFT Boruvka MST algorithm not designed for multigraphs]
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
 * @param graph         Input graph. MUST be symmetric (undirected). MUST be CSR (is_csc=false).
 *                      MUST NOT be a multigraph.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param mst_srcs      [out] Pre-allocated device array for MST edge sources.
 *                      Size must be >= 2 * (num_vertices - 1). Output is
 *                      symmetrized: each undirected MST edge appears in both
 *                      directions (u→v and v→u).
 * @param mst_dsts      [out] Pre-allocated device array for MST edge destinations.
 *                      Size must be >= 2 * (num_vertices - 1).
 * @param mst_weights   [out] Pre-allocated device array for MST edge weights.
 *                      Size must be >= 2 * (num_vertices - 1).
 * @param max_edges     Maximum number of edges that can be written to output arrays.
 * @return              Number of directed edges written (= 2 * number of
 *                      undirected MST edges). For a connected graph, this
 *                      is 2 * (num_vertices - 1).
 */
std::size_t minimum_spanning_tree(const graph32_t& graph,
                                  const double* edge_weights,
                                  int32_t* mst_srcs,
                                  int32_t* mst_dsts,
                                  double* mst_weights,
                                  std::size_t max_edges);

/**
 * Minimum Spanning Tree - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes a minimum spanning tree (or forest if graph is disconnected).
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST NOT be a multigraph
 *       [RAFT Boruvka MST algorithm not designed for multigraphs]
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
 * @param graph         Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 *                      MUST NOT be a multigraph.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param mst_srcs      [out] Pre-allocated device array for MST edge sources.
 *                      Size must be >= 2 * (num_vertices - 1). Output is
 *                      symmetrized: each undirected MST edge appears in both
 *                      directions (u→v and v→u).
 * @param mst_dsts      [out] Pre-allocated device array for MST edge destinations.
 *                      Size must be >= 2 * (num_vertices - 1).
 * @param mst_weights   [out] Pre-allocated device array for MST edge weights.
 *                      Size must be >= 2 * (num_vertices - 1).
 * @param max_edges     Maximum number of edges that can be written to output arrays.
 * @return              Number of directed edges written (= 2 * number of
 *                      undirected MST edges). For a connected graph, this
 *                      is 2 * (num_vertices - 1).
 */
std::size_t minimum_spanning_tree_seg(const graph32_t& graph,
                                      const double* edge_weights,
                                      int32_t* mst_srcs,
                                      int32_t* mst_dsts,
                                      double* mst_weights,
                                      std::size_t max_edges);

}  // namespace aai
