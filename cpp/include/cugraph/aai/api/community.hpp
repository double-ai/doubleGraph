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
 * AAI Community Detection Algorithms: Louvain, Leiden, ECG, Triangle Count,
 * K-Truss, Egonet, Clustering Analysis, Spectral Modularity
 */
#pragma once

#include <cugraph/aai/compact_graph.hpp>
#include <cugraph/aai/types.hpp>

#include <cstddef>
#include <cstdint>

namespace aai {

// =============================================================================
// Louvain Community Detection
// =============================================================================

/**
 * Louvain Community Detection - Float weights.
 *
 * Hierarchical community detection by maximizing modularity.
 *
 * Preconditions (from cpp/src/community/louvain_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 42]
 *   - Graph MUST be weighted
 *       [CUGRAPH_EXPECTS, line 318]
 *   - clustering != nullptr (if graph has vertices)
 *       [detail::check_clustering, lines 31-36]
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
 * louvain_result_float_t struct:
 *   std::size_t level_count - Number of hierarchical levels
 *   float modularity        - Final modularity score
 *
 * @param graph         Input graph. MUST be CSR (is_csc=false) and weighted.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param clusters      [out] Pre-allocated array of size num_vertices for cluster IDs.
 * @param max_level     Maximum hierarchy levels.
 * @param threshold     Modularity gain threshold for convergence.
 * @param resolution    Resolution parameter (gamma). Higher = more smaller communities.
 * @return              Result struct with level_count and modularity.
 */
louvain_result_float_t louvain(const graph32_t& graph,
                               const float* edge_weights,
                               int32_t* clusters,
                               std::size_t max_level,
                               float threshold,
                               float resolution);

/**
 * Louvain Community Detection - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Hierarchical community detection by maximizing modularity.
 *
 * Preconditions (from cpp/src/community/louvain_impl.cuh):
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 42]
 *   - Graph MUST be weighted
 *       [CUGRAPH_EXPECTS, line 318]
 *   - clustering != nullptr (if graph has vertices)
 *       [detail::check_clustering, lines 31-36]
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
 * louvain_result_float_t struct:
 *   std::size_t level_count - Number of hierarchical levels
 *   float modularity        - Final modularity score
 *
 * @param graph         Input graph. MUST be CSR (is_csc=false) and weighted. MUST have segment_offsets.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param clusters      [out] Pre-allocated array of size num_vertices for cluster IDs.
 * @param max_level     Maximum hierarchy levels.
 * @param threshold     Modularity gain threshold for convergence.
 * @param resolution    Resolution parameter (gamma). Higher = more smaller communities.
 * @return              Result struct with level_count and modularity.
 */
louvain_result_float_t louvain_seg(const graph32_t& graph,
                                   const float* edge_weights,
                                   int32_t* clusters,
                                   std::size_t max_level,
                                   float threshold,
                                   float resolution);

/**
 * Louvain Community Detection - Double weights.
 *
 * Hierarchical community detection by maximizing modularity.
 *
 * Preconditions (from cpp/src/community/louvain_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 42]
 *   - Graph MUST be weighted
 *       [CUGRAPH_EXPECTS, line 318]
 *   - clustering != nullptr (if graph has vertices)
 *       [detail::check_clustering, lines 31-36]
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
 * louvain_result_double_t struct:
 *   std::size_t level_count - Number of hierarchical levels
 *   double modularity       - Final modularity score
 *
 * @param graph         Input graph. MUST be CSR (is_csc=false) and weighted.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param clusters      [out] Pre-allocated array of size num_vertices for cluster IDs.
 * @param max_level     Maximum hierarchy levels.
 * @param threshold     Modularity gain threshold for convergence.
 * @param resolution    Resolution parameter (gamma). Higher = more smaller communities.
 * @return              Result struct with level_count and modularity.
 */
louvain_result_double_t louvain(const graph32_t& graph,
                                const double* edge_weights,
                                int32_t* clusters,
                                std::size_t max_level,
                                double threshold,
                                double resolution);

/**
 * Louvain Community Detection - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Hierarchical community detection by maximizing modularity.
 *
 * Preconditions (from cpp/src/community/louvain_impl.cuh):
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 42]
 *   - Graph MUST be weighted
 *       [CUGRAPH_EXPECTS, line 318]
 *   - clustering != nullptr (if graph has vertices)
 *       [detail::check_clustering, lines 31-36]
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
 * louvain_result_double_t struct:
 *   std::size_t level_count - Number of hierarchical levels
 *   double modularity       - Final modularity score
 *
 * @param graph         Input graph. MUST be CSR (is_csc=false) and weighted. MUST have segment_offsets.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param clusters      [out] Pre-allocated array of size num_vertices for cluster IDs.
 * @param max_level     Maximum hierarchy levels.
 * @param threshold     Modularity gain threshold for convergence.
 * @param resolution    Resolution parameter (gamma). Higher = more smaller communities.
 * @return              Result struct with level_count and modularity.
 */
louvain_result_double_t louvain_seg(const graph32_t& graph,
                                    const double* edge_weights,
                                    int32_t* clusters,
                                    std::size_t max_level,
                                    double threshold,
                                    double resolution);

// =============================================================================
// Leiden Community Detection
// =============================================================================

/**
 * Leiden Community Detection - Float weights.
 *
 * Improved community detection guaranteeing well-connected communities.
 *
 * Preconditions (from cpp/src/community/leiden_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 73]
 *   - Graph MUST be weighted
 *       [CUGRAPH_EXPECTS, line 701]
 *   - clustering != nullptr (if graph has vertices)
 *       [detail::check_clustering, lines 30-35]
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
 * leiden_result_float_t struct:
 *   std::size_t level_count - Number of hierarchical levels
 *   float modularity        - Final modularity score
 *
 * @param graph         Input graph. MUST be CSR (is_csc=false) and weighted.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param clusters      [out] Pre-allocated array of size num_vertices for cluster IDs.
 * @param max_level     Maximum hierarchy levels.
 * @param resolution    Resolution parameter (gamma).
 * @param theta         Refinement probability scaling.
 * @return              Result struct with level_count and modularity.
 *
 * Note: This implementation is deterministic.
 */
leiden_result_float_t leiden(const graph32_t& graph,
                             const float* edge_weights,
                             int32_t* clusters,
                             std::size_t max_level,
                             float resolution,
                             float theta);

/**
 * Leiden Community Detection - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Improved community detection guaranteeing well-connected communities.
 *
 * Preconditions (from cpp/src/community/leiden_impl.cuh):
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 73]
 *   - Graph MUST be weighted
 *       [CUGRAPH_EXPECTS, line 701]
 *   - clustering != nullptr (if graph has vertices)
 *       [detail::check_clustering, lines 30-35]
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
 * leiden_result_float_t struct:
 *   std::size_t level_count - Number of hierarchical levels
 *   float modularity        - Final modularity score
 *
 * @param graph         Input graph. MUST be CSR (is_csc=false) and weighted. MUST have segment_offsets.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param clusters      [out] Pre-allocated array of size num_vertices for cluster IDs.
 * @param max_level     Maximum hierarchy levels.
 * @param resolution    Resolution parameter (gamma).
 * @param theta         Refinement probability scaling.
 * @return              Result struct with level_count and modularity.
 *
 * Note: This implementation is deterministic.
 */
leiden_result_float_t leiden_seg(const graph32_t& graph,
                                 const float* edge_weights,
                                 int32_t* clusters,
                                 std::size_t max_level,
                                 float resolution,
                                 float theta);

/**
 * Leiden Community Detection - Double weights.
 *
 * Improved community detection guaranteeing well-connected communities.
 *
 * Preconditions (from cpp/src/community/leiden_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 73]
 *   - Graph MUST be weighted
 *       [CUGRAPH_EXPECTS, line 701]
 *   - clustering != nullptr (if graph has vertices)
 *       [detail::check_clustering, lines 30-35]
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
 * leiden_result_double_t struct:
 *   std::size_t level_count - Number of hierarchical levels
 *   double modularity       - Final modularity score
 *
 * @param graph         Input graph. MUST be CSR (is_csc=false) and weighted.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param clusters      [out] Pre-allocated array of size num_vertices for cluster IDs.
 * @param max_level     Maximum hierarchy levels.
 * @param resolution    Resolution parameter (gamma).
 * @param theta         Refinement probability scaling.
 * @return              Result struct with level_count and modularity.
 *
 * Note: This implementation is deterministic.
 */
leiden_result_double_t leiden(const graph32_t& graph,
                              const double* edge_weights,
                              int32_t* clusters,
                              std::size_t max_level,
                              double resolution,
                              double theta);

/**
 * Leiden Community Detection - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Improved community detection guaranteeing well-connected communities.
 *
 * Preconditions (from cpp/src/community/leiden_impl.cuh):
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 73]
 *   - Graph MUST be weighted
 *       [CUGRAPH_EXPECTS, line 701]
 *   - clustering != nullptr (if graph has vertices)
 *       [detail::check_clustering, lines 30-35]
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
 * leiden_result_double_t struct:
 *   std::size_t level_count - Number of hierarchical levels
 *   double modularity       - Final modularity score
 *
 * @param graph         Input graph. MUST be CSR (is_csc=false) and weighted. MUST have segment_offsets.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param clusters      [out] Pre-allocated array of size num_vertices for cluster IDs.
 * @param max_level     Maximum hierarchy levels.
 * @param resolution    Resolution parameter (gamma).
 * @param theta         Refinement probability scaling.
 * @return              Result struct with level_count and modularity.
 *
 * Note: This implementation is deterministic.
 */
leiden_result_double_t leiden_seg(const graph32_t& graph,
                                  const double* edge_weights,
                                  int32_t* clusters,
                                  std::size_t max_level,
                                  double resolution,
                                  double theta);

// =============================================================================
// ECG (Ensemble Clustering for Graphs)
// =============================================================================

/**
 * ECG - Float weights.
 *
 * Ensemble Clustering for Graphs. Runs truncated Louvain on an ensemble of
 * permutations of the input graph, then uses the ensemble partitions to
 * determine weights for the input graph. The final result is found by running
 * full Louvain on the input graph using the determined weights.
 *
 * Preconditions (from cpp/src/community/ecg_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 34]
 *   - Graph MUST be weighted
 *       [implicit: dereferences edge_weight_view at lines 99, 120]
 *   - min_weight >= 0.0
 *       [CUGRAPH_EXPECTS, lines 44-45]
 *   - ensemble_size >= 1
 *       [CUGRAPH_EXPECTS, lines 46-47]
 *   - threshold in (0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 48-50]
 *   - resolution in (0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 51-53]
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph
 *                             (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * ecg_result_float_t struct:
 *   std::size_t level_count - Number of hierarchical levels
 *   float modularity        - Final modularity score
 *
 * @param graph         Input graph. MUST be CSR (is_csc=false) and weighted.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param clusters      [out] Pre-allocated array of size num_vertices for cluster IDs.
 * @param min_weight    Minimum edge weight for ECG. Must be >= 0.0.
 * @param ensemble_size Number of graph permutations for ensemble. Must be >= 1.
 * @param max_level     Maximum hierarchy levels.
 * @param threshold     Modularity gain threshold. Must be in (0.0, 1.0].
 * @param resolution    Resolution parameter (gamma). Must be in (0.0, 1.0].
 * @return              Result struct with level_count and modularity.
 *
 * Note: This implementation is deterministic.
 */
ecg_result_float_t ecg(const graph32_t& graph,
                       const float* edge_weights,
                       int32_t* clusters,
                       float min_weight,
                       std::size_t ensemble_size,
                       std::size_t max_level,
                       float threshold,
                       float resolution);

/**
 * ECG - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Ensemble Clustering for Graphs. Runs truncated Louvain on an ensemble of
 * permutations of the input graph, then uses the ensemble partitions to
 * determine weights for the input graph. The final result is found by running
 * full Louvain on the input graph using the determined weights.
 *
 * Preconditions (from cpp/src/community/ecg_impl.cuh):
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 34]
 *   - Graph MUST be weighted
 *       [implicit: dereferences edge_weight_view at lines 99, 120]
 *   - min_weight >= 0.0
 *       [CUGRAPH_EXPECTS, lines 44-45]
 *   - ensemble_size >= 1
 *       [CUGRAPH_EXPECTS, lines 46-47]
 *   - threshold in (0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 48-50]
 *   - resolution in (0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 51-53]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph
 *                             (= offsets[number_of_vertices] - offsets[0]).
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
 * ecg_result_float_t struct:
 *   std::size_t level_count - Number of hierarchical levels
 *   float modularity        - Final modularity score
 *
 * @param graph         Input graph. MUST be CSR (is_csc=false) and weighted.
 *                      MUST have segment_offsets.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param clusters      [out] Pre-allocated array of size num_vertices for cluster IDs.
 * @param min_weight    Minimum edge weight for ECG. Must be >= 0.0.
 * @param ensemble_size Number of graph permutations for ensemble. Must be >= 1.
 * @param max_level     Maximum hierarchy levels.
 * @param threshold     Modularity gain threshold. Must be in (0.0, 1.0].
 * @param resolution    Resolution parameter (gamma). Must be in (0.0, 1.0].
 * @return              Result struct with level_count and modularity.
 *
 * Note: This implementation is deterministic.
 */
ecg_result_float_t ecg_seg(const graph32_t& graph,
                           const float* edge_weights,
                           int32_t* clusters,
                           float min_weight,
                           std::size_t ensemble_size,
                           std::size_t max_level,
                           float threshold,
                           float resolution);

/**
 * ECG - Double weights.
 *
 * Ensemble Clustering for Graphs. Runs truncated Louvain on an ensemble of
 * permutations of the input graph, then uses the ensemble partitions to
 * determine weights for the input graph. The final result is found by running
 * full Louvain on the input graph using the determined weights.
 *
 * Preconditions (from cpp/src/community/ecg_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 34]
 *   - Graph MUST be weighted
 *       [implicit: dereferences edge_weight_view at lines 99, 120]
 *   - min_weight >= 0.0
 *       [CUGRAPH_EXPECTS, lines 44-45]
 *   - ensemble_size >= 1
 *       [CUGRAPH_EXPECTS, lines 46-47]
 *   - threshold in (0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 48-50]
 *   - resolution in (0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 51-53]
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *                             offsets[i] = start index in indices for vertex i.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *                             Neighbor vertex IDs (dst if CSR, src if CSC).
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph
 *                             (= offsets[number_of_vertices] - offsets[0]).
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR (row = src), true = CSC (row = dst).
 *   std::optional<std::vector<int32_t>> segment_offsets
 *                           - NOT PROVIDED (std::nullopt) for this function variant.
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * ecg_result_double_t struct:
 *   std::size_t level_count - Number of hierarchical levels
 *   double modularity       - Final modularity score
 *
 * @param graph         Input graph. MUST be CSR (is_csc=false) and weighted.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param clusters      [out] Pre-allocated array of size num_vertices for cluster IDs.
 * @param min_weight    Minimum edge weight for ECG. Must be >= 0.0.
 * @param ensemble_size Number of graph permutations for ensemble. Must be >= 1.
 * @param max_level     Maximum hierarchy levels.
 * @param threshold     Modularity gain threshold. Must be in (0.0, 1.0].
 * @param resolution    Resolution parameter (gamma). Must be in (0.0, 1.0].
 * @return              Result struct with level_count and modularity.
 *
 * Note: This implementation is deterministic.
 */
ecg_result_double_t ecg(const graph32_t& graph,
                        const double* edge_weights,
                        int32_t* clusters,
                        double min_weight,
                        std::size_t ensemble_size,
                        std::size_t max_level,
                        double threshold,
                        double resolution);

/**
 * ECG - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Ensemble Clustering for Graphs. Runs truncated Louvain on an ensemble of
 * permutations of the input graph, then uses the ensemble partitions to
 * determine weights for the input graph. The final result is found by running
 * full Louvain on the input graph using the determined weights.
 *
 * Preconditions (from cpp/src/community/ecg_impl.cuh):
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 34]
 *   - Graph MUST be weighted
 *       [implicit: dereferences edge_weight_view at lines 99, 120]
 *   - min_weight >= 0.0
 *       [CUGRAPH_EXPECTS, lines 44-45]
 *   - ensemble_size >= 1
 *       [CUGRAPH_EXPECTS, lines 46-47]
 *   - threshold in (0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 48-50]
 *   - resolution in (0.0, 1.0]
 *       [CUGRAPH_EXPECTS, lines 51-53]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
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
 *   const uint32_t* edge_mask     - NOT PROVIDED (nullptr) for this function variant.
 *
 * ecg_result_double_t struct:
 *   std::size_t level_count - Number of hierarchical levels
 *   double modularity       - Final modularity score
 *
 * @param graph         Input graph. MUST be CSR (is_csc=false) and weighted.
 *                      MUST have segment_offsets.
 * @param edge_weights  [in] Edge weights array (size = num_edges).
 * @param clusters      [out] Pre-allocated array of size num_vertices for cluster IDs.
 * @param min_weight    Minimum edge weight for ECG. Must be >= 0.0.
 * @param ensemble_size Number of graph permutations for ensemble. Must be >= 1.
 * @param max_level     Maximum hierarchy levels.
 * @param threshold     Modularity gain threshold. Must be in (0.0, 1.0].
 * @param resolution    Resolution parameter (gamma). Must be in (0.0, 1.0].
 * @return              Result struct with level_count and modularity.
 *
 * Note: This implementation is deterministic.
 */
ecg_result_double_t ecg_seg(const graph32_t& graph,
                            const double* edge_weights,
                            int32_t* clusters,
                            double min_weight,
                            std::size_t ensemble_size,
                            std::size_t max_level,
                            double threshold,
                            double resolution);

// =============================================================================
// Clustering Analysis
// =============================================================================

/**
 * Analyze Clustering - Edge Cut (Float weights).
 *
 * Computes the edge cut score of a clustering assignment.
 * Edge cut is the total weight of edges crossing cluster boundaries.
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
 * @param graph               Input graph. MUST be symmetric (undirected).
 * @param edge_weights        [in] Edge weights array (size = num_edges).
 * @param num_clusters        Number of clusters in the assignment.
 * @param cluster_assignments [in] Cluster ID for each vertex (size = num_vertices).
 *                            Cluster IDs should be in range [0, num_clusters).
 * @return                    The edge cut score (sum of inter-cluster edge weights).
 */
double analyze_clustering_edge_cut(const graph32_t& graph,
                                   const float* edge_weights,
                                   std::size_t num_clusters,
                                   const int32_t* cluster_assignments);

/**
 * Analyze Clustering - Edge Cut (Float weights) - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes the edge cut score of a clustering assignment.
 * Edge cut is the total weight of edges crossing cluster boundaries.
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
 * @param graph               Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights        [in] Edge weights array (size = num_edges).
 * @param num_clusters        Number of clusters in the assignment.
 * @param cluster_assignments [in] Cluster ID for each vertex (size = num_vertices).
 *                            Cluster IDs should be in range [0, num_clusters).
 * @return                    The edge cut score (sum of inter-cluster edge weights).
 */
double analyze_clustering_edge_cut_seg(const graph32_t& graph,
                                       const float* edge_weights,
                                       std::size_t num_clusters,
                                       const int32_t* cluster_assignments);

/**
 * Analyze Clustering - Edge Cut (Double weights).
 *
 * Computes the edge cut score of a clustering assignment.
 * Edge cut is the total weight of edges crossing cluster boundaries.
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
 * @param graph               Input graph. MUST be symmetric (undirected).
 * @param edge_weights        [in] Edge weights array (size = num_edges).
 * @param num_clusters        Number of clusters in the assignment.
 * @param cluster_assignments [in] Cluster ID for each vertex (size = num_vertices).
 *                            Cluster IDs should be in range [0, num_clusters).
 * @return                    The edge cut score (sum of inter-cluster edge weights).
 */
double analyze_clustering_edge_cut(const graph32_t& graph,
                                   const double* edge_weights,
                                   std::size_t num_clusters,
                                   const int32_t* cluster_assignments);

/**
 * Analyze Clustering - Edge Cut (Double weights) - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes the edge cut score of a clustering assignment.
 * Edge cut is the total weight of edges crossing cluster boundaries.
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
 * @param graph               Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights        [in] Edge weights array (size = num_edges).
 * @param num_clusters        Number of clusters in the assignment.
 * @param cluster_assignments [in] Cluster ID for each vertex (size = num_vertices).
 *                            Cluster IDs should be in range [0, num_clusters).
 * @return                    The edge cut score (sum of inter-cluster edge weights).
 */
double analyze_clustering_edge_cut_seg(const graph32_t& graph,
                                       const double* edge_weights,
                                       std::size_t num_clusters,
                                       const int32_t* cluster_assignments);

/**
 * Analyze Clustering - Ratio Cut (Float weights).
 *
 * Computes the ratio cut score of a clustering assignment.
 * Ratio cut normalizes edge cut by cluster sizes.
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
 * @param graph               Input graph. MUST be symmetric (undirected).
 * @param edge_weights        [in] Edge weights array (size = num_edges).
 * @param num_clusters        Number of clusters in the assignment.
 * @param cluster_assignments [in] Cluster ID for each vertex (size = num_vertices).
 * @return                    The ratio cut score.
 */
double analyze_clustering_ratio_cut(const graph32_t& graph,
                                    const float* edge_weights,
                                    std::size_t num_clusters,
                                    const int32_t* cluster_assignments);

/**
 * Analyze Clustering - Ratio Cut (Float weights) - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes the ratio cut score of a clustering assignment.
 * Ratio cut normalizes edge cut by cluster sizes.
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
 * @param graph               Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights        [in] Edge weights array (size = num_edges).
 * @param num_clusters        Number of clusters in the assignment.
 * @param cluster_assignments [in] Cluster ID for each vertex (size = num_vertices).
 * @return                    The ratio cut score.
 */
double analyze_clustering_ratio_cut_seg(const graph32_t& graph,
                                        const float* edge_weights,
                                        std::size_t num_clusters,
                                        const int32_t* cluster_assignments);

/**
 * Analyze Clustering - Ratio Cut (Double weights).
 *
 * Computes the ratio cut score of a clustering assignment.
 * Ratio cut normalizes edge cut by cluster sizes.
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
 * @param graph               Input graph. MUST be symmetric (undirected).
 * @param edge_weights        [in] Edge weights array (size = num_edges).
 * @param num_clusters        Number of clusters in the assignment.
 * @param cluster_assignments [in] Cluster ID for each vertex (size = num_vertices).
 * @return                    The ratio cut score.
 */
double analyze_clustering_ratio_cut(const graph32_t& graph,
                                    const double* edge_weights,
                                    std::size_t num_clusters,
                                    const int32_t* cluster_assignments);

/**
 * Analyze Clustering - Ratio Cut (Double weights) - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes the ratio cut score of a clustering assignment.
 * Ratio cut normalizes edge cut by cluster sizes.
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
 * @param graph               Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights        [in] Edge weights array (size = num_edges).
 * @param num_clusters        Number of clusters in the assignment.
 * @param cluster_assignments [in] Cluster ID for each vertex (size = num_vertices).
 * @return                    The ratio cut score.
 */
double analyze_clustering_ratio_cut_seg(const graph32_t& graph,
                                        const double* edge_weights,
                                        std::size_t num_clusters,
                                        const int32_t* cluster_assignments);

/**
 * Analyze Clustering - Modularity (Float weights).
 *
 * Computes the modularity score of a clustering assignment.
 * Modularity measures the density of edges within clusters compared to
 * a random graph with the same degree sequence.
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
 * @param graph               Input graph. MUST be symmetric (undirected).
 * @param edge_weights        [in] Edge weights array (size = num_edges).
 * @param num_clusters        Number of clusters in the assignment.
 * @param cluster_assignments [in] Cluster ID for each vertex (size = num_vertices).
 * @return                    The modularity score (range: -0.5 to 1.0).
 */
double analyze_clustering_modularity(const graph32_t& graph,
                                     const float* edge_weights,
                                     std::size_t num_clusters,
                                     const int32_t* cluster_assignments);

/**
 * Analyze Clustering - Modularity (Float weights) - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes the modularity score of a clustering assignment.
 * Modularity measures the density of edges within clusters compared to
 * a random graph with the same degree sequence.
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
 * @param graph               Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights        [in] Edge weights array (size = num_edges).
 * @param num_clusters        Number of clusters in the assignment.
 * @param cluster_assignments [in] Cluster ID for each vertex (size = num_vertices).
 * @return                    The modularity score (range: -0.5 to 1.0).
 */
double analyze_clustering_modularity_seg(const graph32_t& graph,
                                         const float* edge_weights,
                                         std::size_t num_clusters,
                                         const int32_t* cluster_assignments);

/**
 * Analyze Clustering - Modularity (Double weights).
 *
 * Computes the modularity score of a clustering assignment.
 * Modularity measures the density of edges within clusters compared to
 * a random graph with the same degree sequence.
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
 * @param graph               Input graph. MUST be symmetric (undirected).
 * @param edge_weights        [in] Edge weights array (size = num_edges).
 * @param num_clusters        Number of clusters in the assignment.
 * @param cluster_assignments [in] Cluster ID for each vertex (size = num_vertices).
 * @return                    The modularity score (range: -0.5 to 1.0).
 */
double analyze_clustering_modularity(const graph32_t& graph,
                                     const double* edge_weights,
                                     std::size_t num_clusters,
                                     const int32_t* cluster_assignments);

/**
 * Analyze Clustering - Modularity (Double weights) - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes the modularity score of a clustering assignment.
 * Modularity measures the density of edges within clusters compared to
 * a random graph with the same degree sequence.
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
 * @param graph               Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights        [in] Edge weights array (size = num_edges).
 * @param num_clusters        Number of clusters in the assignment.
 * @param cluster_assignments [in] Cluster ID for each vertex (size = num_vertices).
 * @return                    The modularity score (range: -0.5 to 1.0).
 */
double analyze_clustering_modularity_seg(const graph32_t& graph,
                                         const double* edge_weights,
                                         std::size_t num_clusters,
                                         const int32_t* cluster_assignments);

// =============================================================================
// Triangle Count
// =============================================================================

/**
 * Triangle Count - Per-vertex counts.
 *
 * Counts triangles for each vertex (or specified vertices).
 *
 * Preconditions (from cpp/src/community/triangle_count_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 130]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, lines 139-141]
 *   - Graph MUST NOT be a multigraph
 *       [CUGRAPH_EXPECTS, lines 143-144]
 *   - counts.size() MUST match vertices.size() if vertices provided
 *       [CUGRAPH_EXPECTS, lines 145-147]
 *   - counts.size() MUST match local_vertex_partition_range_size() if vertices not provided
 *       [CUGRAPH_EXPECTS, lines 148-152]
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
 * @param counts     [out] Pre-allocated array. Size = n_vertices if vertices != nullptr,
 *                   else num_vertices.
 * @param vertices   [in] Vertex IDs to count triangles for, or nullptr for all vertices.
 * @param n_vertices Number of vertices in array. Ignored if vertices is nullptr.
 */
void triangle_count(const graph32_t& graph,
                    int32_t* counts,
                    const int32_t* vertices,
                    std::size_t n_vertices);

/**
 * Triangle Count - Per-vertex counts - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Counts triangles for each vertex (or specified vertices).
 *
 * Preconditions (from cpp/src/community/triangle_count_impl.cuh):
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 130]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, lines 139-141]
 *   - Graph MUST NOT be a multigraph
 *       [CUGRAPH_EXPECTS, lines 143-144]
 *   - counts.size() MUST match vertices.size() if vertices provided
 *       [CUGRAPH_EXPECTS, lines 145-147]
 *   - counts.size() MUST match local_vertex_partition_range_size() if vertices not provided
 *       [CUGRAPH_EXPECTS, lines 148-152]
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
 * @param counts     [out] Pre-allocated array. Size = n_vertices if vertices != nullptr,
 *                   else num_vertices.
 * @param vertices   [in] Vertex IDs to count triangles for, or nullptr for all vertices.
 * @param n_vertices Number of vertices in array. Ignored if vertices is nullptr.
 */
void triangle_count_seg(const graph32_t& graph,
                        int32_t* counts,
                        const int32_t* vertices,
                        std::size_t n_vertices);

/**
 * Triangle Count - Edge mask variant (no precomputed segments).
 *
 * Same as triangle_count but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be symmetric, not a multigraph, CSR format
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
 * @param graph      Input graph. MUST be CSR (is_csc=false).
 * @param counts     [out] Pre-allocated array. Size = n_vertices if vertices != nullptr,
 *                   else num_vertices.
 * @param vertices   [in] Vertex IDs to count triangles for, or nullptr for all vertices.
 * @param n_vertices Number of vertices in array. Ignored if vertices is nullptr.
 */
void triangle_count_mask(const graph32_t& graph,
                         int32_t* counts,
                         const int32_t* vertices,
                         std::size_t n_vertices);

/**
 * Triangle Count - Precomputed segments + edge mask variant.
 *
 * Same as triangle_count_seg but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be symmetric, not a multigraph, CSR format
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
 * @param graph      Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets.
 * @param counts     [out] Pre-allocated array. Size = n_vertices if vertices != nullptr,
 *                   else num_vertices.
 * @param vertices   [in] Vertex IDs to count triangles for, or nullptr for all vertices.
 * @param n_vertices Number of vertices in array. Ignored if vertices is nullptr.
 */
void triangle_count_seg_mask(const graph32_t& graph,
                             int32_t* counts,
                             const int32_t* vertices,
                             std::size_t n_vertices);

// =============================================================================
// K-Truss
// =============================================================================

/**
 * K-Truss - Unweighted.
 *
 * Finds the k-truss subgraph, which is the maximal subgraph where each edge
 * is part of at least (k-2) triangles within the subgraph.
 *
 * Preconditions (from cpp/src/community/k_truss_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 165]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, lines 172-173]
 *   - Graph MUST NOT be a multigraph
 *       [CUGRAPH_EXPECTS, lines 174-175]
 *   - Adjacency lists MUST be sorted in ascending order by neighbor vertex ID
 *       (indices[offsets[i]..offsets[i+1]] is sorted for every vertex i)
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
 * k_truss_result_t struct (all pointers are device memory, caller MUST free with cudaFree):
 *   int32_t* edge_srcs      - Device pointer: source vertices of k-truss edges
 *   int32_t* edge_dsts      - Device pointer: destination vertices of k-truss edges
 *   std::size_t num_edges   - Number of edges in the k-truss subgraph
 *
 * @param graph      Input graph. MUST be symmetric (undirected) and not a multigraph.
 * @param k          The truss number.
 * @return           Result struct with edge arrays.
 *                   Caller MUST free all pointers with cudaFree.
 */
k_truss_result_t k_truss(const graph32_t& graph,
                         int32_t k);

/**
 * K-Truss - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Finds the k-truss subgraph, which is the maximal subgraph where each edge
 * is part of at least (k-2) triangles within the subgraph.
 *
 * Preconditions (from cpp/src/community/k_truss_impl.cuh):
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 165]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, lines 172-173]
 *   - Graph MUST NOT be a multigraph
 *       [CUGRAPH_EXPECTS, lines 174-175]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Adjacency lists MUST be sorted in ascending order by neighbor vertex ID
 *       (indices[offsets[i]..offsets[i+1]] is sorted for every vertex i)
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
 * k_truss_result_t struct (all pointers are device memory, caller MUST free with cudaFree):
 *   int32_t* edge_srcs      - Device pointer: source vertices of k-truss edges
 *   int32_t* edge_dsts      - Device pointer: destination vertices of k-truss edges
 *   std::size_t num_edges   - Number of edges in the k-truss subgraph
 *
 * @param graph      Input graph. MUST be symmetric (undirected) and not a multigraph. MUST have segment_offsets.
 * @param k          The truss number.
 * @return           Result struct with edge arrays.
 *                   Caller MUST free all pointers with cudaFree.
 */
k_truss_result_t k_truss_seg(const graph32_t& graph,
                             int32_t k);

/**
 * K-Truss - Unweighted - Edge mask variant (no precomputed segments).
 *
 * Same as k_truss but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be symmetric, not a multigraph, CSR format
 *   - Adjacency lists MUST be sorted in ascending order by neighbor vertex ID
 *       (indices[offsets[i]..offsets[i+1]] is sorted for every vertex i)
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
 * k_truss_result_t struct (all pointers are device memory, caller MUST free with cudaFree):
 *   int32_t* edge_srcs      - Device pointer: source vertices of k-truss edges
 *   int32_t* edge_dsts      - Device pointer: destination vertices of k-truss edges
 *   std::size_t num_edges   - Number of edges in the k-truss subgraph
 *
 * @param graph      Input graph. MUST be symmetric (undirected) and not a multigraph.
 * @param k          The truss number.
 * @return           Result struct with edge arrays.
 *                   Caller MUST free all pointers with cudaFree.
 */
k_truss_result_t k_truss_mask(const graph32_t& graph,
                              int32_t k);

/**
 * K-Truss - Unweighted - Precomputed segments + edge mask variant.
 *
 * Same as k_truss_seg but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be symmetric, not a multigraph, CSR format
 *   - Adjacency lists MUST be sorted in ascending order by neighbor vertex ID
 *       (indices[offsets[i]..offsets[i+1]] is sorted for every vertex i)
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
 * @param graph      Input graph. MUST be symmetric (undirected) and not a multigraph. MUST have segment_offsets.
 * @param k          The truss number.
 * @return           Result struct with edge arrays.
 *                   Caller MUST free all pointers with cudaFree.
 */
k_truss_result_t k_truss_seg_mask(const graph32_t& graph,
                                  int32_t k);

// =============================================================================
// Extract Ego (Egonet)
// =============================================================================

/**
 * Extract Ego - Unweighted.
 *
 * Extracts ego networks (induced subgraphs) centered at specified source vertices,
 * within a given radius (number of hops).
 *
 * Preconditions (from cpp/src/community/egonet_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 58]
 *   - radius > 0
 *       [CUGRAPH_EXPECTS, line 249]
 *   - radius < number_of_vertices
 *       [CUGRAPH_EXPECTS, line 250]
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
 * extract_ego_result_t struct (all pointers are device memory, caller MUST free with cudaFree):
 *   int32_t* edge_srcs      - Device pointer: source vertices of edges
 *   int32_t* edge_dsts      - Device pointer: destination vertices of edges
 *   std::size_t* offsets    - Device pointer: offsets array (size = n_sources + 1)
 *   std::size_t num_edges   - Total number of edges across all ego networks
 *   std::size_t num_offsets - Size of offsets array (n_sources + 1)
 *
 * @param graph             Input graph. MUST be CSR (is_csc=false).
 * @param source_vertices   [in] Array of center vertices (size = n_sources).
 * @param n_sources         Number of source vertices (ego centers).
 * @param radius            Number of hops from each source vertex. Must be > 0 and < num_vertices.
 * @return                  Result struct with edge arrays and offsets.
 *                          Caller MUST free all pointers with cudaFree.
 */
extract_ego_result_t extract_ego(const graph32_t& graph,
                                 const int32_t* source_vertices,
                                 std::size_t n_sources,
                                 int32_t radius);

/**
 * Extract Ego - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Extracts ego networks (induced subgraphs) centered at specified source vertices,
 * within a given radius (number of hops).
 *
 * Preconditions (from cpp/src/community/egonet_impl.cuh):
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 58]
 *   - radius > 0
 *       [CUGRAPH_EXPECTS, line 249]
 *   - radius < number_of_vertices
 *       [CUGRAPH_EXPECTS, line 250]
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
 * extract_ego_result_t struct (all pointers are device memory, caller MUST free with cudaFree):
 *   int32_t* edge_srcs      - Device pointer: source vertices of edges
 *   int32_t* edge_dsts      - Device pointer: destination vertices of edges
 *   std::size_t* offsets    - Device pointer: offsets array (size = n_sources + 1)
 *   std::size_t num_edges   - Total number of edges across all ego networks
 *   std::size_t num_offsets - Size of offsets array (n_sources + 1)
 *
 * @param graph             Input graph. MUST be CSR (is_csc=false). MUST have segment_offsets.
 * @param source_vertices   [in] Array of center vertices (size = n_sources).
 * @param n_sources         Number of source vertices (ego centers).
 * @param radius            Number of hops from each source vertex. Must be > 0 and < num_vertices.
 * @return                  Result struct with edge arrays and offsets.
 *                          Caller MUST free all pointers with cudaFree.
 */
extract_ego_result_t extract_ego_seg(const graph32_t& graph,
                                     const int32_t* source_vertices,
                                     std::size_t n_sources,
                                     int32_t radius);

// =============================================================================
// Extract Ego (Egonet) - Weighted
// =============================================================================

/**
 * Extract Ego - Float weights.
 *
 * Extracts ego networks (induced subgraphs) centered at specified source vertices,
 * within a given radius (number of hops).
 *
 * Preconditions (from cpp/src/community/egonet_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 58]
 *   - radius > 0
 *       [CUGRAPH_EXPECTS, line 249]
 *   - radius < number_of_vertices
 *       [CUGRAPH_EXPECTS, line 250]
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
 * extract_ego_weighted_result_float_t struct (all pointers are device memory, caller MUST free with cudaFree):
 *   int32_t* edge_srcs      - Device pointer: source vertices of edges
 *   int32_t* edge_dsts      - Device pointer: destination vertices of edges
 *   float* edge_weights     - Device pointer: weights of edges
 *   std::size_t* offsets    - Device pointer: offsets array (size = n_sources + 1)
 *   std::size_t num_edges   - Total number of edges across all ego networks
 *   std::size_t num_offsets - Size of offsets array (n_sources + 1)
 *
 * @param graph             Input graph. MUST be CSR (is_csc=false) and weighted.
 * @param edge_weights      [in] Edge weights array (size = num_edges).
 * @param source_vertices   [in] Array of center vertices (size = n_sources).
 * @param n_sources         Number of source vertices (ego centers).
 * @param radius            Number of hops from each source vertex. Must be > 0 and < num_vertices.
 * @return                  Result struct with edge arrays, weights, and offsets.
 *                          Caller MUST free all pointers with cudaFree.
 */
extract_ego_weighted_result_float_t extract_ego_weighted_f32(
    const graph32_t& graph,
    const float* edge_weights,
    const int32_t* source_vertices,
    std::size_t n_sources,
    int32_t radius);

/**
 * Extract Ego - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Extracts ego networks (induced subgraphs) centered at specified source vertices,
 * within a given radius (number of hops).
 *
 * Preconditions (from cpp/src/community/egonet_impl.cuh):
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 58]
 *   - radius > 0
 *       [CUGRAPH_EXPECTS, line 249]
 *   - radius < number_of_vertices
 *       [CUGRAPH_EXPECTS, line 250]
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
 * extract_ego_weighted_result_float_t struct (all pointers are device memory, caller MUST free with cudaFree):
 *   int32_t* edge_srcs      - Device pointer: source vertices of edges
 *   int32_t* edge_dsts      - Device pointer: destination vertices of edges
 *   float* edge_weights     - Device pointer: weights of edges
 *   std::size_t* offsets    - Device pointer: offsets array (size = n_sources + 1)
 *   std::size_t num_edges   - Total number of edges across all ego networks
 *   std::size_t num_offsets - Size of offsets array (n_sources + 1)
 *
 * @param graph             Input graph. MUST be CSR (is_csc=false) and weighted. MUST have segment_offsets.
 * @param edge_weights      [in] Edge weights array (size = num_edges).
 * @param source_vertices   [in] Array of center vertices (size = n_sources).
 * @param n_sources         Number of source vertices (ego centers).
 * @param radius            Number of hops from each source vertex. Must be > 0 and < num_vertices.
 * @return                  Result struct with edge arrays, weights, and offsets.
 *                          Caller MUST free all pointers with cudaFree.
 */
extract_ego_weighted_result_float_t extract_ego_weighted_f32_seg(
    const graph32_t& graph,
    const float* edge_weights,
    const int32_t* source_vertices,
    std::size_t n_sources,
    int32_t radius);

// =============================================================================
// Extract Ego (Egonet) - Weighted (Double)
// =============================================================================

/**
 * Extract Ego - Double weights.
 *
 * Extracts ego networks (induced subgraphs) centered at specified source vertices,
 * within a given radius (number of hops).
 *
 * Preconditions (from cpp/src/community/egonet_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 58]
 *   - radius > 0
 *       [CUGRAPH_EXPECTS, line 249]
 *   - radius < number_of_vertices
 *       [CUGRAPH_EXPECTS, line 250]
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
 * extract_ego_weighted_result_double_t struct (all pointers are device memory, caller MUST free with cudaFree):
 *   int32_t* edge_srcs      - Device pointer: source vertices of edges
 *   int32_t* edge_dsts      - Device pointer: destination vertices of edges
 *   double* edge_weights    - Device pointer: weights of edges
 *   std::size_t* offsets    - Device pointer: offsets array (size = n_sources + 1)
 *   std::size_t num_edges   - Total number of edges across all ego networks
 *   std::size_t num_offsets - Size of offsets array (n_sources + 1)
 *
 * @param graph             Input graph. MUST be CSR (is_csc=false) and weighted.
 * @param edge_weights      [in] Edge weights array (size = num_edges).
 * @param source_vertices   [in] Array of center vertices (size = n_sources).
 * @param n_sources         Number of source vertices (ego centers).
 * @param radius            Number of hops from each source vertex. Must be > 0 and < num_vertices.
 * @return                  Result struct with edge arrays, weights, and offsets.
 *                          Caller MUST free all pointers with cudaFree.
 */
extract_ego_weighted_result_double_t extract_ego_weighted_f64(
    const graph32_t& graph,
    const double* edge_weights,
    const int32_t* source_vertices,
    std::size_t n_sources,
    int32_t radius);

/**
 * Extract Ego - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Extracts ego networks (induced subgraphs) centered at specified source vertices,
 * within a given radius (number of hops).
 *
 * Preconditions (from cpp/src/community/egonet_impl.cuh):
 *   - Graph MUST be CSR format (is_csc=false)
 *       [implicit via template parameter is_storage_transposed=false, line 58]
 *   - radius > 0
 *       [CUGRAPH_EXPECTS, line 249]
 *   - radius < number_of_vertices
 *       [CUGRAPH_EXPECTS, line 250]
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
 * extract_ego_weighted_result_double_t struct (all pointers are device memory, caller MUST free with cudaFree):
 *   int32_t* edge_srcs      - Device pointer: source vertices of edges
 *   int32_t* edge_dsts      - Device pointer: destination vertices of edges
 *   double* edge_weights    - Device pointer: weights of edges
 *   std::size_t* offsets    - Device pointer: offsets array (size = n_sources + 1)
 *   std::size_t num_edges   - Total number of edges across all ego networks
 *   std::size_t num_offsets - Size of offsets array (n_sources + 1)
 *
 * @param graph             Input graph. MUST be CSR (is_csc=false) and weighted. MUST have segment_offsets.
 * @param edge_weights      [in] Edge weights array (size = num_edges).
 * @param source_vertices   [in] Array of center vertices (size = n_sources).
 * @param n_sources         Number of source vertices (ego centers).
 * @param radius            Number of hops from each source vertex. Must be > 0 and < num_vertices.
 * @return                  Result struct with edge arrays, weights, and offsets.
 *                          Caller MUST free all pointers with cudaFree.
 */
extract_ego_weighted_result_double_t extract_ego_weighted_f64_seg(
    const graph32_t& graph,
    const double* edge_weights,
    const int32_t* source_vertices,
    std::size_t n_sources,
    int32_t radius);

// =============================================================================
// Spectral Modularity Maximization
// =============================================================================

/**
 * Spectral Modularity Maximization - Float weights.
 *
 * Computes clustering using spectral modularity maximization.
 * Uses eigendecomposition of the modularity matrix followed by k-means clustering.
 *
 * Preconditions (from cpp/src/aai/integration/community/spectral.cu,
 *                detail::spectralModularityMaximization_impl lines 123-139):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST have edge weights (edge_data != nullptr)
 *       [RAFT_EXPECTS, line 123]
 *   - evs_tolerance MUST be in range [0.0, 1.0)
 *       [RAFT_EXPECTS, lines 124-127]
 *   - kmean_tolerance MUST be in range [0.0, 1.0)
 *       [RAFT_EXPECTS, lines 128-131]
 *   - num_clusters MUST be > 1
 *       [RAFT_EXPECTS, line 132]
 *   - num_clusters MUST be < number_of_vertices
 *       [RAFT_EXPECTS, lines 133-134]
 *   - num_eigenvectors MUST be <= num_clusters
 *       [RAFT_EXPECTS, lines 135-136]
 *   - clustering MUST NOT be null
 *       [RAFT_EXPECTS, line 137]
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
 * @param graph              Input graph. MUST be symmetric (undirected) and CSR format.
 * @param edge_weights       [in] Edge weights array (size = num_edges).
 * @param num_clusters       Number of clusters to find. Must be > 1.
 * @param num_eigenvectors   Number of eigenvectors to use. Must be <= num_clusters.
 * @param evs_tolerance      Eigensolver convergence tolerance (0.0 to 1.0).
 * @param evs_max_iter       Maximum eigensolver iterations.
 * @param kmean_tolerance    K-means convergence tolerance (0.0 to 1.0).
 * @param kmean_max_iter     Maximum k-means iterations.
 * @param clustering         [out] Pre-allocated array of size num_vertices for cluster assignments.
 *
 * Note: This implementation is deterministic.
 */
void spectral_modularity_maximization(const graph32_t& graph,
                                      const float* edge_weights,
                                      int32_t num_clusters,
                                      int32_t num_eigenvectors,
                                      float evs_tolerance,
                                      int32_t evs_max_iter,
                                      float kmean_tolerance,
                                      int32_t kmean_max_iter,
                                      int32_t* clustering);

/**
 * Spectral Modularity Maximization - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes clustering using spectral modularity maximization.
 * Uses eigendecomposition of the modularity matrix followed by k-means clustering.
 *
 * Preconditions (from cpp/src/aai/integration/community/spectral.cu,
 *                detail::spectralModularityMaximization_impl lines 123-139):
 *   - Graph MUST have edge weights (edge_data != nullptr)
 *       [RAFT_EXPECTS, line 123]
 *   - evs_tolerance MUST be in range [0.0, 1.0)
 *       [RAFT_EXPECTS, lines 124-127]
 *   - kmean_tolerance MUST be in range [0.0, 1.0)
 *       [RAFT_EXPECTS, lines 128-131]
 *   - num_clusters MUST be > 1
 *       [RAFT_EXPECTS, line 132]
 *   - num_clusters MUST be < number_of_vertices
 *       [RAFT_EXPECTS, lines 133-134]
 *   - num_eigenvectors MUST be <= num_clusters
 *       [RAFT_EXPECTS, lines 135-136]
 *   - clustering MUST NOT be null
 *       [RAFT_EXPECTS, line 137]
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
 * @param graph              Input graph. MUST be symmetric (undirected) and CSR format. MUST have segment_offsets.
 * @param edge_weights       [in] Edge weights array (size = num_edges).
 * @param num_clusters       Number of clusters to find. Must be > 1.
 * @param num_eigenvectors   Number of eigenvectors to use. Must be <= num_clusters.
 * @param evs_tolerance      Eigensolver convergence tolerance (0.0 to 1.0).
 * @param evs_max_iter       Maximum eigensolver iterations.
 * @param kmean_tolerance    K-means convergence tolerance (0.0 to 1.0).
 * @param kmean_max_iter     Maximum k-means iterations.
 * @param clustering         [out] Pre-allocated array of size num_vertices for cluster assignments.
 *
 * Note: This implementation is deterministic.
 */
void spectral_modularity_maximization_seg(const graph32_t& graph,
                                          const float* edge_weights,
                                          int32_t num_clusters,
                                          int32_t num_eigenvectors,
                                          float evs_tolerance,
                                          int32_t evs_max_iter,
                                          float kmean_tolerance,
                                          int32_t kmean_max_iter,
                                          int32_t* clustering);

/**
 * Spectral Modularity Maximization - Double weights.
 *
 * Computes clustering using spectral modularity maximization.
 * Uses eigendecomposition of the modularity matrix followed by k-means clustering.
 *
 * Preconditions (from cpp/src/aai/integration/community/spectral.cu,
 *                detail::spectralModularityMaximization_impl lines 123-139):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST have edge weights (edge_data != nullptr)
 *       [RAFT_EXPECTS, line 123]
 *   - evs_tolerance MUST be in range [0.0, 1.0)
 *       [RAFT_EXPECTS, lines 124-127]
 *   - kmean_tolerance MUST be in range [0.0, 1.0)
 *       [RAFT_EXPECTS, lines 128-131]
 *   - num_clusters MUST be > 1
 *       [RAFT_EXPECTS, line 132]
 *   - num_clusters MUST be < number_of_vertices
 *       [RAFT_EXPECTS, lines 133-134]
 *   - num_eigenvectors MUST be <= num_clusters
 *       [RAFT_EXPECTS, lines 135-136]
 *   - clustering MUST NOT be null
 *       [RAFT_EXPECTS, line 137]
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
 * @param graph              Input graph. MUST be symmetric (undirected) and CSR format.
 * @param edge_weights       [in] Edge weights array (size = num_edges).
 * @param num_clusters       Number of clusters to find. Must be > 1.
 * @param num_eigenvectors   Number of eigenvectors to use. Must be <= num_clusters.
 * @param evs_tolerance      Eigensolver convergence tolerance (0.0 to 1.0).
 * @param evs_max_iter       Maximum eigensolver iterations.
 * @param kmean_tolerance    K-means convergence tolerance (0.0 to 1.0).
 * @param kmean_max_iter     Maximum k-means iterations.
 * @param clustering         [out] Pre-allocated array of size num_vertices for cluster assignments.
 *
 * Note: This implementation is deterministic.
 */
void spectral_modularity_maximization(const graph32_t& graph,
                                      const double* edge_weights,
                                      int32_t num_clusters,
                                      int32_t num_eigenvectors,
                                      double evs_tolerance,
                                      int32_t evs_max_iter,
                                      double kmean_tolerance,
                                      int32_t kmean_max_iter,
                                      int32_t* clustering);

/**
 * Spectral Modularity Maximization - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes clustering using spectral modularity maximization.
 * Uses eigendecomposition of the modularity matrix followed by k-means clustering.
 *
 * Preconditions (from cpp/src/aai/integration/community/spectral.cu,
 *                detail::spectralModularityMaximization_impl lines 123-139):
 *   - Graph MUST have edge weights (edge_data != nullptr)
 *       [RAFT_EXPECTS, line 123]
 *   - evs_tolerance MUST be in range [0.0, 1.0)
 *       [RAFT_EXPECTS, lines 124-127]
 *   - kmean_tolerance MUST be in range [0.0, 1.0)
 *       [RAFT_EXPECTS, lines 128-131]
 *   - num_clusters MUST be > 1
 *       [RAFT_EXPECTS, line 132]
 *   - num_clusters MUST be < number_of_vertices
 *       [RAFT_EXPECTS, lines 133-134]
 *   - num_eigenvectors MUST be <= num_clusters
 *       [RAFT_EXPECTS, lines 135-136]
 *   - clustering MUST NOT be null
 *       [RAFT_EXPECTS, line 137]
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
 * @param graph              Input graph. MUST be symmetric (undirected) and CSR format. MUST have segment_offsets.
 * @param edge_weights       [in] Edge weights array (size = num_edges).
 * @param num_clusters       Number of clusters to find. Must be > 1.
 * @param num_eigenvectors   Number of eigenvectors to use. Must be <= num_clusters.
 * @param evs_tolerance      Eigensolver convergence tolerance (0.0 to 1.0).
 * @param evs_max_iter       Maximum eigensolver iterations.
 * @param kmean_tolerance    K-means convergence tolerance (0.0 to 1.0).
 * @param kmean_max_iter     Maximum k-means iterations.
 * @param clustering         [out] Pre-allocated array of size num_vertices for cluster assignments.
 *
 * Note: This implementation is deterministic.
 */
void spectral_modularity_maximization_seg(const graph32_t& graph,
                                          const double* edge_weights,
                                          int32_t num_clusters,
                                          int32_t num_eigenvectors,
                                          double evs_tolerance,
                                          int32_t evs_max_iter,
                                          double kmean_tolerance,
                                          int32_t kmean_max_iter,
                                          int32_t* clustering);

}  // namespace aai
