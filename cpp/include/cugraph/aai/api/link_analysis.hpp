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
 * AAI Link Analysis Algorithms: PageRank, HITS
 */
#pragma once

#include <cugraph/aai/compact_graph.hpp>

#include <cstddef>
#include <cstdint>

namespace aai {

// =============================================================================
// PageRank
// =============================================================================

/**
 * Result metadata for PageRank algorithms.
 */
struct PageRankResult {
  std::size_t iterations;  // Number of iterations performed
  bool converged;          // True if converged within epsilon tolerance
};

/**
 * PageRank - Unweighted.
 *
 * Computes PageRank scores using power iteration without edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true).
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param precomputed_vertex_out_weight_sums [in] Precomputed out-degree sums (size = num_vertices), or nullptr.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param initial_pageranks                [in] Initial values (size = num_vertices), or nullptr.
 * @return                                 Result metadata (convergence status).
 */
PageRankResult pagerank(const graph32_t& graph,
                        float* pageranks,
                        const float* precomputed_vertex_out_weight_sums,
                        float alpha,
                        float epsilon,
                        std::size_t max_iterations,
                        const float* initial_pageranks);

/**
 * PageRank - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes PageRank scores using power iteration without edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param precomputed_vertex_out_weight_sums [in] Precomputed out-degree sums (size = num_vertices), or nullptr.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param initial_pageranks                [in] Initial values (size = num_vertices), or nullptr.
 * @return                                 Result metadata (convergence status).
 */
PageRankResult pagerank_seg(const graph32_t& graph,
                            float* pageranks,
                            const float* precomputed_vertex_out_weight_sums,
                            float alpha,
                            float epsilon,
                            std::size_t max_iterations,
                            const float* initial_pageranks);

/**
 * PageRank - Float weights.
 *
 * Computes PageRank scores using power iteration with float edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights                     [in] Edge weights array (size = num_edges).
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param precomputed_vertex_out_weight_sums [in] Precomputed out-weight sums (size = num_vertices), or nullptr.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param initial_pageranks                [in] Initial values (size = num_vertices), or nullptr.
 * @return                                 Result metadata (convergence status).
 */
PageRankResult pagerank(const graph32_t& graph,
                        const float* edge_weights,
                        float* pageranks,
                        const float* precomputed_vertex_out_weight_sums,
                        float alpha,
                        float epsilon,
                        std::size_t max_iterations,
                        const float* initial_pageranks);

/**
 * PageRank - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes PageRank scores using power iteration with float edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param edge_weights                     [in] Edge weights array (size = num_edges).
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param precomputed_vertex_out_weight_sums [in] Precomputed out-weight sums (size = num_vertices), or nullptr.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param initial_pageranks                [in] Initial values (size = num_vertices), or nullptr.
 * @return                                 Result metadata (convergence status).
 */
PageRankResult pagerank_seg(const graph32_t& graph,
                            const float* edge_weights,
                            float* pageranks,
                            const float* precomputed_vertex_out_weight_sums,
                            float alpha,
                            float epsilon,
                            std::size_t max_iterations,
                            const float* initial_pageranks);

/**
 * PageRank - Double weights.
 *
 * Computes PageRank scores using power iteration with double edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights                     [in] Edge weights array (size = num_edges).
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param precomputed_vertex_out_weight_sums [in] Precomputed out-weight sums (size = num_vertices), or nullptr.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param initial_pageranks                [in] Initial values (size = num_vertices), or nullptr.
 * @return                                 Result metadata (convergence status).
 */
PageRankResult pagerank(const graph32_t& graph,
                        const double* edge_weights,
                        double* pageranks,
                        const double* precomputed_vertex_out_weight_sums,
                        double alpha,
                        double epsilon,
                        std::size_t max_iterations,
                        const double* initial_pageranks);

/**
 * PageRank - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes PageRank scores using power iteration with double edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param edge_weights                     [in] Edge weights array (size = num_edges).
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param precomputed_vertex_out_weight_sums [in] Precomputed out-weight sums (size = num_vertices), or nullptr.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param initial_pageranks                [in] Initial values (size = num_vertices), or nullptr.
 * @return                                 Result metadata (convergence status).
 */
PageRankResult pagerank_seg(const graph32_t& graph,
                            const double* edge_weights,
                            double* pageranks,
                            const double* precomputed_vertex_out_weight_sums,
                            double alpha,
                            double epsilon,
                            std::size_t max_iterations,
                            const double* initial_pageranks);

/**
 * Personalized PageRank - Unweighted.
 *
 * Computes personalized PageRank scores using power iteration without edge
 * weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
 *   - personalization_size > 0
 *       [CUGRAPH_EXPECTS, line 82-85]
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Convergence criterion (cuGraph):
 *   sum(|PR_new - PR_old|) < epsilon
 *
 * Personalized PageRank iteration:
 *   out_weight[u]  = number of outgoing edges from u (out-degree for unweighted)
 *   dangling_sum   = sum of PR[u] for all u where out_weight[u] == 0
 *   PR_new[v]      = alpha * sum_{u->v} PR[u] / out_weight[u]
 *                   + (alpha * dangling_sum + (1 - alpha)) * pers_norm[v]
 *   where pers_norm[v] = personalization_values[i] / sum(personalization_values)
 *                        for personalization vertex v, and 0 otherwise.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true).
 * @param personalization_vertices         [in] Vertex IDs for personalization.
 * @param personalization_values           [in] Personalization weights (normalized internally).
 * @param personalization_size             Number of personalization vertices. Must be > 0.
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param has_initial_guess                If true, use values in pageranks as initial guess.
 *                                         Initial values must sum to 1.0 (used as-is, not
 *                                         renormalized by the solver).
 * @return                                 Result metadata (iterations count, convergence status).
 */
PageRankResult personalized_pagerank(const graph32_t& graph,
                                     const int32_t* personalization_vertices,
                                     const float* personalization_values,
                                     std::size_t personalization_size,
                                     float* pageranks,
                                     const float* precomputed_vertex_out_weight_sums,
                                     float alpha,
                                     float epsilon,
                                     std::size_t max_iterations,
                                     const float* initial_pageranks);

/**
 * Personalized PageRank - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes personalized PageRank scores without edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
 *   - personalization_size > 0
 *       [CUGRAPH_EXPECTS, line 82-85]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph.
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR, true = CSC.
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree (>= 1024)
 *                               [1] to [2]: mid-degree (32..1023)
 *                               [2] to [3]: low-degree (1..31)
 *                               [3] to [4]: zero-degree (isolated)
 *   const uint32_t* edge_mask - NOT PROVIDED (nullptr).
 *
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Convergence criterion (cuGraph):
 *   sum(|PR_new - PR_old|) < epsilon
 *
 * @param graph                            Input graph. MUST be CSC. MUST have segment_offsets.
 * @param personalization_vertices         [in] Vertex IDs for personalization.
 * @param personalization_values           [in] Personalization weights (normalized internally).
 * @param personalization_size             Number of personalization vertices. Must be > 0.
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0].
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param has_initial_guess                If true, use values in pageranks as initial guess.
 * @return                                 Result metadata (iterations count, convergence status).
 */
PageRankResult personalized_pagerank_seg(const graph32_t& graph,
                                         const int32_t* personalization_vertices,
                                         const float* personalization_values,
                                         std::size_t personalization_size,
                                         float* pageranks,
                                         const float* precomputed_vertex_out_weight_sums,
                                         float alpha,
                                         float epsilon,
                                         std::size_t max_iterations,
                                         const float* initial_pageranks);

/**
 * Personalized PageRank - Float32 weights.
 *
 * Computes personalized PageRank scores with float32 edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
 *   - personalization_size > 0
 *       [CUGRAPH_EXPECTS, line 82-85]
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Convergence criterion (cuGraph):
 *   sum(|PR_new - PR_old|) < epsilon
 *
 * Weighted Personalized PageRank iteration:
 *   out_weight[u]  = sum of weights of outgoing edges from u
 *   dangling_sum   = sum of PR[u] for all u where out_weight[u] == 0
 *   PR_new[v]      = alpha * sum_{u->v} w(u,v) * PR[u] / out_weight[u]
 *                   + (alpha * dangling_sum + (1 - alpha)) * pers_norm[v]
 *   where pers_norm[v] = personalization_values[i] / sum(personalization_values)
 *                        for personalization vertex v, and 0 otherwise.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights                     [in] Edge weights array (size = num_edges), float32.
 * @param personalization_vertices         [in] Vertex IDs for personalization.
 * @param personalization_values           [in] Personalization weights (normalized internally).
 * @param personalization_size             Number of personalization vertices. Must be > 0.
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param has_initial_guess                If true, use values in pageranks as initial guess.
 *                                         Initial values must sum to 1.0 (used as-is, not
 *                                         renormalized by the solver).
 * @return                                 Result metadata (iterations count, convergence status).
 */
PageRankResult personalized_pagerank(const graph32_t& graph,
                                     const float* edge_weights,
                                     const int32_t* personalization_vertices,
                                     const float* personalization_values,
                                     std::size_t personalization_size,
                                     float* pageranks,
                                     const float* precomputed_vertex_out_weight_sums,
                                     float alpha,
                                     float epsilon,
                                     std::size_t max_iterations,
                                     const float* initial_pageranks);

/**
 * Personalized PageRank - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes personalized PageRank scores with float edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
 *   - personalization_size > 0
 *       [CUGRAPH_EXPECTS, line 82-85]
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param edge_weights                     [in] Edge weights array (size = num_edges).
 * @param personalization_vertices         [in] Vertex IDs for personalization.
 * @param personalization_values           [in] Personalization weights (normalized internally).
 * @param personalization_size             Number of personalization vertices. Must be > 0.
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param precomputed_vertex_out_weight_sums [in] Precomputed out-weight sums (size = num_vertices), or nullptr.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param initial_pageranks                [in] Initial values (size = num_vertices), or nullptr.
 * @return                                 Result metadata (convergence status).
 */
PageRankResult personalized_pagerank_seg(const graph32_t& graph,
                                         const float* edge_weights,
                                         const int32_t* personalization_vertices,
                                         const float* personalization_values,
                                         std::size_t personalization_size,
                                         float* pageranks,
                                         const float* precomputed_vertex_out_weight_sums,
                                         float alpha,
                                         float epsilon,
                                         std::size_t max_iterations,
                                         const float* initial_pageranks);

/**
 * Personalized PageRank - Float64 weights.
 *
 * Computes personalized PageRank scores with float64 edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
 *   - personalization_size > 0
 *       [CUGRAPH_EXPECTS, line 82-85]
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Convergence criterion (cuGraph):
 *   sum(|PR_new - PR_old|) < epsilon
 *
 * Weighted Personalized PageRank iteration:
 *   out_weight[u]  = sum of weights of outgoing edges from u
 *   dangling_sum   = sum of PR[u] for all u where out_weight[u] == 0
 *   PR_new[v]      = alpha * sum_{u->v} w(u,v) * PR[u] / out_weight[u]
 *                   + (alpha * dangling_sum + (1 - alpha)) * pers_norm[v]
 *   where pers_norm[v] = personalization_values[i] / sum(personalization_values)
 *                        for personalization vertex v, and 0 otherwise.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights                     [in] Edge weights array (size = num_edges), float64.
 * @param personalization_vertices         [in] Vertex IDs for personalization.
 * @param personalization_values           [in] Personalization weights (normalized internally).
 * @param personalization_size             Number of personalization vertices. Must be > 0.
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param has_initial_guess                If true, use values in pageranks as initial guess.
 *                                         Initial values must sum to 1.0 (used as-is, not
 *                                         renormalized by the solver).
 * @return                                 Result metadata (iterations count, convergence status).
 */
PageRankResult personalized_pagerank(const graph32_t& graph,
                                     const double* edge_weights,
                                     const int32_t* personalization_vertices,
                                     const double* personalization_values,
                                     std::size_t personalization_size,
                                     double* pageranks,
                                     const double* precomputed_vertex_out_weight_sums,
                                     double alpha,
                                     double epsilon,
                                     std::size_t max_iterations,
                                     const double* initial_pageranks);

/**
 * Personalized PageRank - Float64 weights.
 *
 * Computes personalized PageRank scores with float64 edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
 *   - personalization_size > 0
 *       [CUGRAPH_EXPECTS, line 82-85]
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Convergence criterion (cuGraph):
 *   sum(|PR_new - PR_old|) < epsilon
 *
 * Weighted Personalized PageRank iteration:
 *   out_weight[u]  = sum of weights of outgoing edges from u
 *   dangling_sum   = sum of PR[u] for all u where out_weight[u] == 0
 *   PR_new[v]      = alpha * sum_{u->v} w(u,v) * PR[u] / out_weight[u]
 *                   + (alpha * dangling_sum + (1 - alpha)) * pers_norm[v]
 *   where pers_norm[v] = personalization_values[i] / sum(personalization_values)
 *                        for personalization vertex v, and 0 otherwise.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights                     [in] Edge weights array (size = num_edges), float64.
 * @param personalization_vertices         [in] Vertex IDs for personalization.
 * @param personalization_values           [in] Personalization weights (normalized internally).
 * @param personalization_size             Number of personalization vertices. Must be > 0.
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param has_initial_guess                If true, use values in pageranks as initial guess.
 *                                         Initial values must sum to 1.0 (used as-is, not
 *                                         renormalized by the solver).
 * @return                                 Result metadata (iterations count, convergence status).
 */
PageRankResult personalized_pagerank_seg(const graph32_t& graph,
                                         const double* edge_weights,
                                         const int32_t* personalization_vertices,
                                         const double* personalization_values,
                                         std::size_t personalization_size,
                                         double* pageranks,
                                         const double* precomputed_vertex_out_weight_sums,
                                         double alpha,
                                         double epsilon,
                                         std::size_t max_iterations,
                                         const double* initial_pageranks);

// =============================================================================
// PageRank - Edge mask variants
// =============================================================================

/**
 * PageRank - Unweighted - Edge mask variant (no precomputed segments).
 *
 * Same as pagerank (unweighted) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *   - alpha in [0.0, 1.0]
 *   - epsilon >= 0.0
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true).
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param precomputed_vertex_out_weight_sums [in] Precomputed out-degree sums (size = num_vertices), or nullptr.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param initial_pageranks                [in] Initial values (size = num_vertices), or nullptr.
 * @return                                 Result metadata (convergence status).
 */
PageRankResult pagerank_mask(const graph32_t& graph,
                             float* pageranks,
                             const float* precomputed_vertex_out_weight_sums,
                             float alpha,
                             float epsilon,
                             std::size_t max_iterations,
                             const float* initial_pageranks);

/**
 * PageRank - Unweighted - Precomputed segments + edge mask variant.
 *
 * Same as pagerank_seg (unweighted) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *   - alpha in [0.0, 1.0]
 *   - epsilon >= 0.0
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param precomputed_vertex_out_weight_sums [in] Precomputed out-degree sums (size = num_vertices), or nullptr.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param initial_pageranks                [in] Initial values (size = num_vertices), or nullptr.
 * @return                                 Result metadata (convergence status).
 */
PageRankResult pagerank_seg_mask(const graph32_t& graph,
                                 float* pageranks,
                                 const float* precomputed_vertex_out_weight_sums,
                                 float alpha,
                                 float epsilon,
                                 std::size_t max_iterations,
                                 const float* initial_pageranks);

/**
 * PageRank - Float weights - Edge mask variant (no precomputed segments).
 *
 * Same as pagerank (float weights) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *   - alpha in [0.0, 1.0]
 *   - epsilon >= 0.0
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights                     [in] Edge weights array (size = num_edges).
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param precomputed_vertex_out_weight_sums [in] Precomputed out-weight sums (size = num_vertices), or nullptr.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param initial_pageranks                [in] Initial values (size = num_vertices), or nullptr.
 * @return                                 Result metadata (convergence status).
 */
PageRankResult pagerank_mask(const graph32_t& graph,
                             const float* edge_weights,
                             float* pageranks,
                             const float* precomputed_vertex_out_weight_sums,
                             float alpha,
                             float epsilon,
                             std::size_t max_iterations,
                             const float* initial_pageranks);

/**
 * PageRank - Float weights - Precomputed segments + edge mask variant.
 *
 * Same as pagerank_seg (float weights) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *   - alpha in [0.0, 1.0]
 *   - epsilon >= 0.0
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param edge_weights                     [in] Edge weights array (size = num_edges).
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param precomputed_vertex_out_weight_sums [in] Precomputed out-weight sums (size = num_vertices), or nullptr.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param initial_pageranks                [in] Initial values (size = num_vertices), or nullptr.
 * @return                                 Result metadata (convergence status).
 */
PageRankResult pagerank_seg_mask(const graph32_t& graph,
                                 const float* edge_weights,
                                 float* pageranks,
                                 const float* precomputed_vertex_out_weight_sums,
                                 float alpha,
                                 float epsilon,
                                 std::size_t max_iterations,
                                 const float* initial_pageranks);

/**
 * PageRank - Double weights - Edge mask variant (no precomputed segments).
 *
 * Same as pagerank (double weights) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *   - alpha in [0.0, 1.0]
 *   - epsilon >= 0.0
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights                     [in] Edge weights array (size = num_edges).
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param precomputed_vertex_out_weight_sums [in] Precomputed out-weight sums (size = num_vertices), or nullptr.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param initial_pageranks                [in] Initial values (size = num_vertices), or nullptr.
 * @return                                 Result metadata (convergence status).
 */
PageRankResult pagerank_mask(const graph32_t& graph,
                             const double* edge_weights,
                             double* pageranks,
                             const double* precomputed_vertex_out_weight_sums,
                             double alpha,
                             double epsilon,
                             std::size_t max_iterations,
                             const double* initial_pageranks);

/**
 * PageRank - Double weights - Precomputed segments + edge mask variant.
 *
 * Same as pagerank_seg (double weights) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *   - alpha in [0.0, 1.0]
 *   - epsilon >= 0.0
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param edge_weights                     [in] Edge weights array (size = num_edges).
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param precomputed_vertex_out_weight_sums [in] Precomputed out-weight sums (size = num_vertices), or nullptr.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param initial_pageranks                [in] Initial values (size = num_vertices), or nullptr.
 * @return                                 Result metadata (convergence status).
 */
PageRankResult pagerank_seg_mask(const graph32_t& graph,
                                 const double* edge_weights,
                                 double* pageranks,
                                 const double* precomputed_vertex_out_weight_sums,
                                 double alpha,
                                 double epsilon,
                                 std::size_t max_iterations,
                                 const double* initial_pageranks);

/**
 * Personalized PageRank - Unweighted - Edge mask variant (no precomputed segments).
 *
 * Same as personalized_pagerank (unweighted) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *   - alpha in [0.0, 1.0]
 *   - epsilon >= 0.0
 *   - personalization_size > 0
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true).
 * @param personalization_vertices         [in] Vertex IDs for personalization.
 * @param personalization_values           [in] Personalization weights (normalized internally).
 * @param personalization_size             Number of personalization vertices. Must be > 0.
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param precomputed_vertex_out_weight_sums [in] Precomputed out-degree sums (size = num_vertices), or nullptr.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param initial_pageranks                [in] Initial values (size = num_vertices), or nullptr.
 * @return                                 Result metadata (convergence status).
 */
PageRankResult personalized_pagerank_mask(const graph32_t& graph,
                                          const int32_t* personalization_vertices,
                                          const float* personalization_values,
                                          std::size_t personalization_size,
                                          float* pageranks,
                                          const float* precomputed_vertex_out_weight_sums,
                                          float alpha,
                                          float epsilon,
                                          std::size_t max_iterations,
                                          const float* initial_pageranks);

/**
 * Personalized PageRank - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes personalized PageRank scores without edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
 *   - personalization_size > 0
 *       [CUGRAPH_EXPECTS, line 82-85]
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *
 * graph32_t struct:
 *   int32_t* offsets        - Device pointer. Size: number_of_vertices + 1.
 *   int32_t* indices        - Device pointer. Size: number_of_edges.
 *   int32_t number_of_vertices - Total vertices in graph.
 *   int32_t number_of_edges    - Total edges in graph.
 *   bool is_symmetric       - True if edge (u,v) implies edge (v,u).
 *   bool is_multigraph      - True if duplicate edges may exist.
 *   bool is_csc             - false = CSR, true = CSC.
 *   std::vector<int32_t> segment_offsets
 *                           - HOST memory. Vertices sorted by degree (descending).
 *                             5 elements defining 4 segments:
 *                               [0] to [1]: high-degree (>= 1024)
 *                               [1] to [2]: mid-degree (32..1023)
 *                               [2] to [3]: low-degree (1..31)
 *                               [3] to [4]: zero-degree (isolated)
 *   const uint32_t* edge_mask - NOT PROVIDED (nullptr).
 *
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Convergence criterion (cuGraph):
 *   sum(|PR_new - PR_old|) < epsilon
 *
 * @param graph                            Input graph. MUST be CSC. MUST have segment_offsets.
 * @param personalization_vertices         [in] Vertex IDs for personalization.
 * @param personalization_values           [in] Personalization weights (normalized internally).
 * @param personalization_size             Number of personalization vertices. Must be > 0.
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0].
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param has_initial_guess                If true, use values in pageranks as initial guess.
 * @return                                 Result metadata (iterations count, convergence status).
 */
PageRankResult personalized_pagerank_seg_mask(const graph32_t& graph,
                                              const int32_t* personalization_vertices,
                                              const float* personalization_values,
                                              std::size_t personalization_size,
                                              float* pageranks,
                                              const float* precomputed_vertex_out_weight_sums,
                                              float alpha,
                                              float epsilon,
                                              std::size_t max_iterations,
                                              const float* initial_pageranks);

/**
 * Personalized PageRank - Float32 weights.
 *
 * Computes personalized PageRank scores with float32 edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
 *   - personalization_size > 0
 *       [CUGRAPH_EXPECTS, line 82-85]
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Convergence criterion (cuGraph):
 *   sum(|PR_new - PR_old|) < epsilon
 *
 * Weighted Personalized PageRank iteration:
 *   out_weight[u]  = sum of weights of outgoing edges from u
 *   dangling_sum   = sum of PR[u] for all u where out_weight[u] == 0
 *   PR_new[v]      = alpha * sum_{u->v} w(u,v) * PR[u] / out_weight[u]
 *                   + (alpha * dangling_sum + (1 - alpha)) * pers_norm[v]
 *   where pers_norm[v] = personalization_values[i] / sum(personalization_values)
 *                        for personalization vertex v, and 0 otherwise.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights                     [in] Edge weights array (size = num_edges), float32.
 * @param personalization_vertices         [in] Vertex IDs for personalization.
 * @param personalization_values           [in] Personalization weights (normalized internally).
 * @param personalization_size             Number of personalization vertices. Must be > 0.
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param has_initial_guess                If true, use values in pageranks as initial guess.
 *                                         Initial values must sum to 1.0 (used as-is, not
 *                                         renormalized by the solver).
 * @return                                 Result metadata (iterations count, convergence status).
 */
PageRankResult personalized_pagerank_mask(const graph32_t& graph,
                                          const float* edge_weights,
                                          const int32_t* personalization_vertices,
                                          const float* personalization_values,
                                          std::size_t personalization_size,
                                          float* pageranks,
                                          const float* precomputed_vertex_out_weight_sums,
                                          float alpha,
                                          float epsilon,
                                          std::size_t max_iterations,
                                          const float* initial_pageranks);

/**
 * Personalized PageRank - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes personalized PageRank scores with float edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
 *   - personalization_size > 0
 *       [CUGRAPH_EXPECTS, line 82-85]
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param edge_weights                     [in] Edge weights array (size = num_edges).
 * @param personalization_vertices         [in] Vertex IDs for personalization.
 * @param personalization_values           [in] Personalization weights (normalized internally).
 * @param personalization_size             Number of personalization vertices. Must be > 0.
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param precomputed_vertex_out_weight_sums [in] Precomputed out-weight sums (size = num_vertices), or nullptr.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param initial_pageranks                [in] Initial values (size = num_vertices), or nullptr.
 * @return                                 Result metadata (convergence status).
 */
PageRankResult personalized_pagerank_seg_mask(const graph32_t& graph,
                                              const float* edge_weights,
                                              const int32_t* personalization_vertices,
                                              const float* personalization_values,
                                              std::size_t personalization_size,
                                              float* pageranks,
                                              const float* precomputed_vertex_out_weight_sums,
                                              float alpha,
                                              float epsilon,
                                              std::size_t max_iterations,
                                              const float* initial_pageranks);

/**
 * Personalized PageRank - Float64 weights.
 *
 * Computes personalized PageRank scores with float64 edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
 *   - personalization_size > 0
 *       [CUGRAPH_EXPECTS, line 82-85]
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Convergence criterion (cuGraph):
 *   sum(|PR_new - PR_old|) < epsilon
 *
 * Weighted Personalized PageRank iteration:
 *   out_weight[u]  = sum of weights of outgoing edges from u
 *   dangling_sum   = sum of PR[u] for all u where out_weight[u] == 0
 *   PR_new[v]      = alpha * sum_{u->v} w(u,v) * PR[u] / out_weight[u]
 *                   + (alpha * dangling_sum + (1 - alpha)) * pers_norm[v]
 *   where pers_norm[v] = personalization_values[i] / sum(personalization_values)
 *                        for personalization vertex v, and 0 otherwise.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights                     [in] Edge weights array (size = num_edges), float64.
 * @param personalization_vertices         [in] Vertex IDs for personalization.
 * @param personalization_values           [in] Personalization weights (normalized internally).
 * @param personalization_size             Number of personalization vertices. Must be > 0.
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param has_initial_guess                If true, use values in pageranks as initial guess.
 *                                         Initial values must sum to 1.0 (used as-is, not
 *                                         renormalized by the solver).
 * @return                                 Result metadata (iterations count, convergence status).
 */
PageRankResult personalized_pagerank_mask(const graph32_t& graph,
                                          const double* edge_weights,
                                          const int32_t* personalization_vertices,
                                          const double* personalization_values,
                                          std::size_t personalization_size,
                                          double* pageranks,
                                          const double* precomputed_vertex_out_weight_sums,
                                          double alpha,
                                          double epsilon,
                                          std::size_t max_iterations,
                                          const double* initial_pageranks);

/**
 * Personalized PageRank - Float64 weights.
 *
 * Computes personalized PageRank scores with float64 edge weights.
 *
 * Preconditions (from cpp/src/link_analysis/pagerank_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 61-62]
 *   - alpha in [0.0, 1.0]
 *       [CUGRAPH_EXPECTS, line 86-87]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 88]
 *   - personalization_size > 0
 *       [CUGRAPH_EXPECTS, line 82-85]
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
 * PageRankResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *
 * Convergence criterion (cuGraph):
 *   sum(|PR_new - PR_old|) < epsilon
 *
 * Weighted Personalized PageRank iteration:
 *   out_weight[u]  = sum of weights of outgoing edges from u
 *   dangling_sum   = sum of PR[u] for all u where out_weight[u] == 0
 *   PR_new[v]      = alpha * sum_{u->v} w(u,v) * PR[u] / out_weight[u]
 *                   + (alpha * dangling_sum + (1 - alpha)) * pers_norm[v]
 *   where pers_norm[v] = personalization_values[i] / sum(personalization_values)
 *                        for personalization vertex v, and 0 otherwise.
 *
 * @param graph                            Input graph. MUST be CSC (is_csc=true).
 * @param edge_weights                     [in] Edge weights array (size = num_edges), float64.
 * @param personalization_vertices         [in] Vertex IDs for personalization.
 * @param personalization_values           [in] Personalization weights (normalized internally).
 * @param personalization_size             Number of personalization vertices. Must be > 0.
 * @param pageranks                        [out] Pre-allocated device array of size num_vertices.
 * @param alpha                            Damping factor. Must be in [0.0, 1.0] (typically 0.85).
 * @param epsilon                          Convergence tolerance. Must be >= 0.0.
 * @param max_iterations                   Maximum iterations.
 * @param has_initial_guess                If true, use values in pageranks as initial guess.
 *                                         Initial values must sum to 1.0 (used as-is, not
 *                                         renormalized by the solver).
 * @return                                 Result metadata (iterations count, convergence status).
 */
PageRankResult personalized_pagerank_seg_mask(const graph32_t& graph,
                                              const double* edge_weights,
                                              const int32_t* personalization_vertices,
                                              const double* personalization_values,
                                              std::size_t personalization_size,
                                              double* pageranks,
                                              const double* precomputed_vertex_out_weight_sums,
                                              double alpha,
                                              double epsilon,
                                              std::size_t max_iterations,
                                              const double* initial_pageranks);

// =============================================================================
// HITS - Hyperlink-Induced Topic Search
// =============================================================================

struct HitsResult {
  std::size_t iterations;  // Number of iterations performed
  bool converged;          // True if converged within epsilon tolerance
  float final_norm;        // Final L1 norm (sum of absolute differences between iterations)
};

struct HitsResultDouble {
  std::size_t iterations;  // Number of iterations performed
  bool converged;          // True if converged within epsilon tolerance
  double final_norm;       // Final L1 norm (sum of absolute differences between iterations)
};

/**
 * HITS - Hyperlink-Induced Topic Search.
 *
 * Computes hub and authority scores using power iteration.
 *
 * Preconditions (from cpp/src/link_analysis/hits_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 62-63]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 71]
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
 * HitsResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *   float final_norm        - Final L1 norm (sum of absolute differences between iterations).
 *
 * @param graph                  Input graph. MUST be CSC (is_csc=true).
 * @param hubs                   [out] Pre-allocated array of size num_vertices for hub scores.
 * @param authorities            [out] Pre-allocated array of size num_vertices for authority scores.
 * @param epsilon                Convergence tolerance. Must be >= 0.0.
 * @param max_iterations         Maximum iterations.
 * @param has_initial_hubs_guess If true, use values in hubs as initial guess.
 *                               Initial hubs are L1-normalized (divided by their sum)
 *                               before iteration begins.
 * @param normalize              If true, normalize output scores.
 * @return                       Result struct with iterations count, convergence status, and final L1 norm.
 */
HitsResult hits(const graph32_t& graph,
                float* hubs,
                float* authorities,
                float epsilon,
                std::size_t max_iterations,
                bool has_initial_hubs_guess,
                bool normalize);

/**
 * HITS - Hyperlink-Induced Topic Search - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes hub and authority scores using power iteration.
 *
 * Preconditions (from cpp/src/link_analysis/hits_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 62-63]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 71]
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
 * HitsResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *   float final_norm        - Final L1 norm (sum of absolute differences between iterations).
 *
 * @param graph                  Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param hubs                   [out] Pre-allocated array of size num_vertices for hub scores.
 * @param authorities            [out] Pre-allocated array of size num_vertices for authority scores.
 * @param epsilon                Convergence tolerance. Must be >= 0.0.
 * @param max_iterations         Maximum iterations.
 * @param has_initial_hubs_guess If true, use values in hubs as initial guess.
 *                               Initial hubs are L1-normalized (divided by their sum)
 *                               before iteration begins.
 * @param normalize              If true, normalize output scores.
 * @return                       Result struct with iterations count, convergence status, and final L1 norm.
 */
HitsResult hits_seg(const graph32_t& graph,
                    float* hubs,
                    float* authorities,
                    float epsilon,
                    std::size_t max_iterations,
                    bool has_initial_hubs_guess,
                    bool normalize);

/**
 * HITS - Hyperlink-Induced Topic Search (double precision).
 *
 * Computes hub and authority scores using power iteration.
 *
 * Preconditions (from cpp/src/link_analysis/hits_impl.cuh):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 62-63]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 71]
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
 * HitsResultDouble struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *   double final_norm       - Final L1 norm
 *                             (sum of absolute differences between iterations).
 *
 * @param graph                  Input graph. MUST be CSC (is_csc=true).
 * @param hubs                   [out] Pre-allocated array of size num_vertices
 *                               for hub scores.
 * @param authorities            [out] Pre-allocated array of size num_vertices
 *                               for authority scores.
 * @param epsilon                Convergence tolerance. Must be >= 0.0.
 * @param max_iterations         Maximum iterations.
 * @param has_initial_hubs_guess If true, use values in hubs as initial guess.
 *                               Initial hubs are L1-normalized (divided by their sum)
 *                               before iteration begins.
 * @param normalize              If true, normalize output scores.
 * @return                       Result struct with iterations count,
 *                               convergence status, and final L1 norm.
 */
HitsResultDouble hits(const graph32_t& graph,
                      double* hubs,
                      double* authorities,
                      double epsilon,
                      std::size_t max_iterations,
                      bool has_initial_hubs_guess,
                      bool normalize);

/**
 * HITS - Hyperlink-Induced Topic Search (double precision) - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes hub and authority scores using power iteration.
 *
 * Preconditions (from cpp/src/link_analysis/hits_impl.cuh):
 *   - Graph MUST be CSC format (is_csc=true)
 *       [static_assert is_storage_transposed, line 62-63]
 *   - epsilon >= 0.0
 *       [CUGRAPH_EXPECTS, line 71]
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
 * HitsResultDouble struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *   double final_norm       - Final L1 norm (sum of absolute differences
 *                             between iterations).
 *
 * @param graph                  Input graph. MUST be CSC (is_csc=true).
 *                               MUST have segment_offsets.
 * @param hubs                   [out] Pre-allocated array of size num_vertices
 *                               for hub scores.
 * @param authorities            [out] Pre-allocated array of size num_vertices
 *                               for authority scores.
 * @param epsilon                Convergence tolerance. Must be >= 0.0.
 * @param max_iterations         Maximum iterations.
 * @param has_initial_hubs_guess If true, use values in hubs as initial guess.
 *                               Initial hubs are L1-normalized (divided by their sum)
 *                               before iteration begins.
 * @param normalize              If true, normalize output scores.
 * @return                       Result struct with iterations count,
 *                               convergence status, and final L1 norm.
 */
HitsResultDouble hits_seg(const graph32_t& graph,
                          double* hubs,
                          double* authorities,
                          double epsilon,
                          std::size_t max_iterations,
                          bool has_initial_hubs_guess,
                          bool normalize);

// =============================================================================
// HITS - Edge mask variants
// =============================================================================

/**
 * HITS - Float - Edge mask variant (no precomputed segments).
 *
 * Same as hits (float) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *   - epsilon >= 0.0
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
 * HitsResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *   float final_norm        - Final L1 norm (sum of absolute differences between iterations).
 *
 * @param graph                  Input graph. MUST be CSC (is_csc=true).
 * @param hubs                   [out] Pre-allocated array of size num_vertices for hub scores.
 * @param authorities            [out] Pre-allocated array of size num_vertices for authority scores.
 * @param epsilon                Convergence tolerance. Must be >= 0.0.
 * @param max_iterations         Maximum iterations.
 * @param has_initial_hubs_guess If true, use values in hubs as initial guess.
 *                               Initial hubs are L1-normalized (divided by their sum)
 *                               before iteration begins.
 * @param normalize              If true, normalize output scores.
 * @return                       Result struct with iterations count, convergence status, and final L1 norm.
 */
HitsResult hits_mask(const graph32_t& graph,
                     float* hubs,
                     float* authorities,
                     float epsilon,
                     std::size_t max_iterations,
                     bool has_initial_hubs_guess,
                     bool normalize);

/**
 * HITS - Float - Precomputed segments + edge mask variant.
 *
 * Same as hits_seg (float) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *   - epsilon >= 0.0
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
 * HitsResult struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *   float final_norm        - Final L1 norm (sum of absolute differences between iterations).
 *
 * @param graph                  Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param hubs                   [out] Pre-allocated array of size num_vertices for hub scores.
 * @param authorities            [out] Pre-allocated array of size num_vertices for authority scores.
 * @param epsilon                Convergence tolerance. Must be >= 0.0.
 * @param max_iterations         Maximum iterations.
 * @param has_initial_hubs_guess If true, use values in hubs as initial guess.
 *                               Initial hubs are L1-normalized (divided by their sum)
 *                               before iteration begins.
 * @param normalize              If true, normalize output scores.
 * @return                       Result struct with iterations count, convergence status, and final L1 norm.
 */
HitsResult hits_seg_mask(const graph32_t& graph,
                         float* hubs,
                         float* authorities,
                         float epsilon,
                         std::size_t max_iterations,
                         bool has_initial_hubs_guess,
                         bool normalize);

/**
 * HITS - Double - Edge mask variant (no precomputed segments).
 *
 * Same as hits (double) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 *
 * Preconditions:
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *   - epsilon >= 0.0
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
 * HitsResultDouble struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *   double final_norm       - Final L1 norm (sum of absolute differences between iterations).
 *
 * @param graph                  Input graph. MUST be CSC (is_csc=true).
 * @param hubs                   [out] Pre-allocated array of size num_vertices for hub scores.
 * @param authorities            [out] Pre-allocated array of size num_vertices for authority scores.
 * @param epsilon                Convergence tolerance. Must be >= 0.0.
 * @param max_iterations         Maximum iterations.
 * @param has_initial_hubs_guess If true, use values in hubs as initial guess.
 *                               Initial hubs are L1-normalized (divided by their sum)
 *                               before iteration begins.
 * @param normalize              If true, normalize output scores.
 * @return                       Result struct with iterations count, convergence status, and final L1 norm.
 */
HitsResultDouble hits_mask(const graph32_t& graph,
                           double* hubs,
                           double* authorities,
                           double epsilon,
                           std::size_t max_iterations,
                           bool has_initial_hubs_guess,
                           bool normalize);

/**
 * HITS - Double - Precomputed segments + edge mask variant.
 *
 * Same as hits_seg (double) but the graph has an edge mask applied.
 * Masked edges (bit == 0) must be excluded from all computation.
 * Note: segment_offsets reflect the ORIGINAL (unmasked) degrees.
 *
 * Preconditions:
 *   - graph.segment_offsets MUST have a value
 *   - graph.edge_mask MUST be non-null
 *   - Graph MUST be CSC format (is_csc=true)
 *   - epsilon >= 0.0
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
 * HitsResultDouble struct (return type):
 *   std::size_t iterations  - Number of iterations performed.
 *   bool converged          - True if converged within epsilon tolerance.
 *   double final_norm       - Final L1 norm (sum of absolute differences between iterations).
 *
 * @param graph                  Input graph. MUST be CSC (is_csc=true). MUST have segment_offsets.
 * @param hubs                   [out] Pre-allocated array of size num_vertices for hub scores.
 * @param authorities            [out] Pre-allocated array of size num_vertices for authority scores.
 * @param epsilon                Convergence tolerance. Must be >= 0.0.
 * @param max_iterations         Maximum iterations.
 * @param has_initial_hubs_guess If true, use values in hubs as initial guess.
 *                               Initial hubs are L1-normalized (divided by their sum)
 *                               before iteration begins.
 * @param normalize              If true, normalize output scores.
 * @return                       Result struct with iterations count, convergence status, and final L1 norm.
 */
HitsResultDouble hits_seg_mask(const graph32_t& graph,
                               double* hubs,
                               double* authorities,
                               double epsilon,
                               std::size_t max_iterations,
                               bool has_initial_hubs_guess,
                               bool normalize);

}  // namespace aai
