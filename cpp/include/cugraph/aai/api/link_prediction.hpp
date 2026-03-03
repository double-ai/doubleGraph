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
 * AAI Link Prediction Algorithms: Jaccard, Cosine, Overlap, Sorensen Similarity
 */
#pragma once

#include <cugraph/aai/compact_graph.hpp>
#include <cugraph/aai/types.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace aai {

// =============================================================================
// Jaccard Similarity
// =============================================================================

/**
 * Jaccard Similarity - Unweighted.
 *
 * Computes Jaccard similarity coefficient for specified vertex pairs.
 * Jaccard(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
 *
 * Preconditions (from cpp/src/link_prediction/jaccard_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, jaccard_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void jaccard_similarity(const graph32_t& graph,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        float* similarity_scores);

/**
 * Jaccard Similarity - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes Jaccard similarity coefficient for specified vertex pairs.
 * Jaccard(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
 *
 * Preconditions (from cpp/src/link_prediction/jaccard_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, jaccard_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - graph.segment_offsets MUST have a value
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected). MUST have segment_offsets.
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void jaccard_similarity_seg(const graph32_t& graph,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            float* similarity_scores);

/**
 * Jaccard Similarity - Float weights.
 *
 * Computes Jaccard similarity coefficient for specified vertex pairs.
 * Jaccard(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
 *
 * Preconditions (from cpp/src/link_prediction/jaccard_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, jaccard_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected).
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void jaccard_similarity(const graph32_t& graph,
                        const float* edge_weights,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        float* similarity_scores);

/**
 * Jaccard Similarity - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes Jaccard similarity coefficient for specified vertex pairs.
 * Jaccard(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
 *
 * Preconditions (from cpp/src/link_prediction/jaccard_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, jaccard_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - graph.segment_offsets MUST have a value
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void jaccard_similarity_seg(const graph32_t& graph,
                            const float* edge_weights,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            float* similarity_scores);

/**
 * Jaccard Similarity - Double weights.
 *
 * Computes Jaccard similarity coefficient for specified vertex pairs.
 * Jaccard(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
 *
 * Preconditions (from cpp/src/link_prediction/jaccard_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, jaccard_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected).
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void jaccard_similarity(const graph32_t& graph,
                        const double* edge_weights,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        double* similarity_scores);

/**
 * Jaccard Similarity - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes Jaccard similarity coefficient for specified vertex pairs.
 * Jaccard(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
 *
 * Preconditions (from cpp/src/link_prediction/jaccard_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, jaccard_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - graph.segment_offsets MUST have a value
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void jaccard_similarity_seg(const graph32_t& graph,
                            const double* edge_weights,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            double* similarity_scores);

// =============================================================================
// Jaccard All-Pairs Similarity
// =============================================================================

/**
 * Jaccard All-Pairs Similarity - Unweighted.
 *
 * Computes Jaccard similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Jaccard(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/jaccard_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-214):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, jaccard_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t jaccard_all_pairs_similarity(const graph32_t& graph,
                                                       const int32_t* vertices,
                                                       std::size_t num_vertices,
                                                       std::optional<std::size_t> topk);

/**
 * Jaccard All-Pairs Similarity - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes Jaccard similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Jaccard(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/jaccard_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-214):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, jaccard_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - graph.segment_offsets MUST have a value
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t jaccard_all_pairs_similarity_seg(const graph32_t& graph,
                                                           const int32_t* vertices,
                                                           std::size_t num_vertices,
                                                           std::optional<std::size_t> topk);

/**
 * Jaccard All-Pairs Similarity - Float weights.
 *
 * Computes Jaccard similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Jaccard(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/jaccard_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, jaccard_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected).
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t jaccard_all_pairs_similarity(const graph32_t& graph,
                                                       const float* edge_weights,
                                                       const int32_t* vertices,
                                                       std::size_t num_vertices,
                                                       std::optional<std::size_t> topk);

/**
 * Jaccard All-Pairs Similarity - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes Jaccard similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Jaccard(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/jaccard_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, jaccard_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
 *   - graph.segment_offsets MUST have a value
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t jaccard_all_pairs_similarity_seg(const graph32_t& graph,
                                                           const float* edge_weights,
                                                           const int32_t* vertices,
                                                           std::size_t num_vertices,
                                                           std::optional<std::size_t> topk);

/**
 * Jaccard All-Pairs Similarity - Double weights.
 *
 * Computes Jaccard similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Jaccard(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/jaccard_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, jaccard_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
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
 * similarity_result_double_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   double* scores      - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected).
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_double_t jaccard_all_pairs_similarity(const graph32_t& graph,
                                                        const double* edge_weights,
                                                        const int32_t* vertices,
                                                        std::size_t num_vertices,
                                                        std::optional<std::size_t> topk);

/**
 * Jaccard All-Pairs Similarity - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes Jaccard similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Jaccard(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/jaccard_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, jaccard_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
 *   - graph.segment_offsets MUST have a value
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
 * similarity_result_double_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   double* scores      - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_double_t jaccard_all_pairs_similarity_seg(const graph32_t& graph,
                                                            const double* edge_weights,
                                                            const int32_t* vertices,
                                                            std::size_t num_vertices,
                                                            std::optional<std::size_t> topk);

// =============================================================================
// Cosine Similarity
// =============================================================================

/**
 * Cosine Similarity - Unweighted.
 *
 * Computes cosine similarity coefficient for specified vertex pairs.
 * For this (unweighted) variant, the score is a binary indicator:
 * score(u,v) = 1.0 if N(u) ∩ N(v) is non-empty, 0.0 otherwise.
 *
 * Preconditions (from cpp/src/link_prediction/cosine_similarity_impl.cuh:37
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, cosine_similarity_impl.cuh:37]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric
 *                              (undirected).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void cosine_similarity(const graph32_t& graph,
                       const int32_t* vertex_pairs_first,
                       const int32_t* vertex_pairs_second,
                       std::size_t num_pairs,
                       float* similarity_scores);

/**
 * Cosine Similarity - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes cosine similarity coefficient for specified vertex pairs.
 * For this (unweighted) variant, the score is a binary indicator:
 * score(u,v) = 1.0 if N(u) ∩ N(v) is non-empty, 0.0 otherwise.
 *
 * Preconditions (from cpp/src/link_prediction/cosine_similarity_impl.cuh:37
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, cosine_similarity_impl.cuh:37]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - graph.segment_offsets MUST have a value
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric
 *                              (undirected). MUST have segment_offsets.
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void cosine_similarity_seg(const graph32_t& graph,
                           const int32_t* vertex_pairs_first,
                           const int32_t* vertex_pairs_second,
                           std::size_t num_pairs,
                           float* similarity_scores);

/**
 * Cosine Similarity - Float weights.
 *
 * Computes cosine similarity coefficient for specified vertex pairs.
 * Let I = N(u) ∩ N(v) and w(v,k) = weight of edge (v,k). Then:
 * score(u,v) = (Σ_{k∈I} w(u,k)·w(v,k)) / (sqrt(Σ_{k∈I} w(u,k)²) · sqrt(Σ_{k∈I} w(v,k)²))
 * Note: norms are computed over the intersection only, not over full neighborhoods.
 *
 * Preconditions (from cpp/src/link_prediction/cosine_similarity_impl.cuh:37
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, cosine_similarity_impl.cuh:37]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected).
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void cosine_similarity(const graph32_t& graph,
                       const float* edge_weights,
                       const int32_t* vertex_pairs_first,
                       const int32_t* vertex_pairs_second,
                       std::size_t num_pairs,
                       float* similarity_scores);

/**
 * Cosine Similarity - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes cosine similarity coefficient for specified vertex pairs.
 * Let I = N(u) ∩ N(v) and w(v,k) = weight of edge (v,k). Then:
 * score(u,v) = (Σ_{k∈I} w(u,k)·w(v,k)) / (sqrt(Σ_{k∈I} w(u,k)²) · sqrt(Σ_{k∈I} w(v,k)²))
 * Note: norms are computed over the intersection only, not over full neighborhoods.
 *
 * Preconditions (from cpp/src/link_prediction/cosine_similarity_impl.cuh:37
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, cosine_similarity_impl.cuh:37]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - graph.segment_offsets MUST have a value
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void cosine_similarity_seg(const graph32_t& graph,
                           const float* edge_weights,
                           const int32_t* vertex_pairs_first,
                           const int32_t* vertex_pairs_second,
                           std::size_t num_pairs,
                           float* similarity_scores);

/**
 * Cosine Similarity - Double weights.
 *
 * Computes cosine similarity coefficient for specified vertex pairs.
 * Let I = N(u) ∩ N(v) and w(v,k) = weight of edge (v,k). Then:
 * score(u,v) = (Σ_{k∈I} w(u,k)·w(v,k)) / (sqrt(Σ_{k∈I} w(u,k)²) · sqrt(Σ_{k∈I} w(v,k)²))
 * Note: norms are computed over the intersection only, not over full neighborhoods.
 *
 * Preconditions (from cpp/src/link_prediction/cosine_similarity_impl.cuh:37
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, cosine_similarity_impl.cuh:37]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected).
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void cosine_similarity(const graph32_t& graph,
                       const double* edge_weights,
                       const int32_t* vertex_pairs_first,
                       const int32_t* vertex_pairs_second,
                       std::size_t num_pairs,
                       double* similarity_scores);

/**
 * Cosine Similarity - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes cosine similarity coefficient for specified vertex pairs.
 * Let I = N(u) ∩ N(v) and w(v,k) = weight of edge (v,k). Then:
 * score(u,v) = (Σ_{k∈I} w(u,k)·w(v,k)) / (sqrt(Σ_{k∈I} w(u,k)²) · sqrt(Σ_{k∈I} w(v,k)²))
 * Note: norms are computed over the intersection only, not over full neighborhoods.
 *
 * Preconditions (from cpp/src/link_prediction/cosine_similarity_impl.cuh:37
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, cosine_similarity_impl.cuh:37]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - graph.segment_offsets MUST have a value
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void cosine_similarity_seg(const graph32_t& graph,
                           const double* edge_weights,
                           const int32_t* vertex_pairs_first,
                           const int32_t* vertex_pairs_second,
                           std::size_t num_pairs,
                           double* similarity_scores);

// =============================================================================
// Cosine All-Pairs Similarity
// =============================================================================

/**
 * Cosine All-Pairs Similarity - Unweighted.
 *
 * Computes cosine similarity for all 2-hop vertex pairs (or a subset of vertices).
 * For this (unweighted) variant, the score is a binary indicator:
 * score(u,v) = 1.0 if N(u) ∩ N(v) is non-empty, 0.0 otherwise.
 * N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/cosine_similarity_impl.cuh:59
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-214):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, cosine_similarity_impl.cuh:59]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t cosine_all_pairs_similarity(const graph32_t& graph,
                                                      const int32_t* vertices,
                                                      std::size_t num_vertices,
                                                      std::optional<std::size_t> topk);

/**
 * Cosine All-Pairs Similarity - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes cosine similarity for all 2-hop vertex pairs (or a subset of vertices).
 * For this (unweighted) variant, the score is a binary indicator:
 * score(u,v) = 1.0 if N(u) ∩ N(v) is non-empty, 0.0 otherwise.
 * N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/cosine_similarity_impl.cuh:59
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-214):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, cosine_similarity_impl.cuh:59]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - graph.segment_offsets MUST have a value
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t cosine_all_pairs_similarity_seg(const graph32_t& graph,
                                                          const int32_t* vertices,
                                                          std::size_t num_vertices,
                                                          std::optional<std::size_t> topk);

/**
 * Cosine All-Pairs Similarity - Float weights.
 *
 * Computes cosine similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Let I = N(u) ∩ N(v) and w(v,k) = weight of edge (v,k). Then:
 * score(u,v) = (Σ_{k∈I} w(u,k)·w(v,k)) / (sqrt(Σ_{k∈I} w(u,k)²) · sqrt(Σ_{k∈I} w(v,k)²))
 * Norms are computed over the intersection only, not over full neighborhoods.
 * N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/cosine_similarity_impl.cuh:59
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, cosine_similarity_impl.cuh:59]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected).
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t cosine_all_pairs_similarity(const graph32_t& graph,
                                                      const float* edge_weights,
                                                      const int32_t* vertices,
                                                      std::size_t num_vertices,
                                                      std::optional<std::size_t> topk);

/**
 * Cosine All-Pairs Similarity - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes cosine similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Let I = N(u) ∩ N(v) and w(v,k) = weight of edge (v,k). Then:
 * score(u,v) = (Σ_{k∈I} w(u,k)·w(v,k)) / (sqrt(Σ_{k∈I} w(u,k)²) · sqrt(Σ_{k∈I} w(v,k)²))
 * Norms are computed over the intersection only, not over full neighborhoods.
 * N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/cosine_similarity_impl.cuh:59
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, cosine_similarity_impl.cuh:59]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
 *   - graph.segment_offsets MUST have a value
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t cosine_all_pairs_similarity_seg(const graph32_t& graph,
                                                          const float* edge_weights,
                                                          const int32_t* vertices,
                                                          std::size_t num_vertices,
                                                          std::optional<std::size_t> topk);

/**
 * Cosine All-Pairs Similarity - Double weights.
 *
 * Computes cosine similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Let I = N(u) ∩ N(v) and w(v,k) = weight of edge (v,k). Then:
 * score(u,v) = (Σ_{k∈I} w(u,k)·w(v,k)) / (sqrt(Σ_{k∈I} w(u,k)²) · sqrt(Σ_{k∈I} w(v,k)²))
 * Norms are computed over the intersection only, not over full neighborhoods.
 * N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/cosine_similarity_impl.cuh:59
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, cosine_similarity_impl.cuh:59]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
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
 * similarity_result_double_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   double* scores      - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected).
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_double_t cosine_all_pairs_similarity(const graph32_t& graph,
                                                       const double* edge_weights,
                                                       const int32_t* vertices,
                                                       std::size_t num_vertices,
                                                       std::optional<std::size_t> topk);

/**
 * Cosine All-Pairs Similarity - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes cosine similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Let I = N(u) ∩ N(v) and w(v,k) = weight of edge (v,k). Then:
 * score(u,v) = (Σ_{k∈I} w(u,k)·w(v,k)) / (sqrt(Σ_{k∈I} w(u,k)²) · sqrt(Σ_{k∈I} w(v,k)²))
 * Norms are computed over the intersection only, not over full neighborhoods.
 * N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/cosine_similarity_impl.cuh:59
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, cosine_similarity_impl.cuh:59]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
 *   - graph.segment_offsets MUST have a value
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
 * similarity_result_double_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   double* scores      - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_double_t cosine_all_pairs_similarity_seg(const graph32_t& graph,
                                                           const double* edge_weights,
                                                           const int32_t* vertices,
                                                           std::size_t num_vertices,
                                                           std::optional<std::size_t> topk);

// =============================================================================
// Overlap Similarity
// =============================================================================

/**
 * Overlap Similarity - Unweighted.
 *
 * Computes overlap similarity coefficient for specified vertex pairs.
 * Overlap(u,v) = |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)
 *
 * Preconditions (from cpp/src/link_prediction/overlap_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, overlap_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void overlap_similarity(const graph32_t& graph,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        float* similarity_scores);

/**
 * Overlap Similarity - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes overlap similarity coefficient for specified vertex pairs.
 * Overlap(u,v) = |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)
 *
 * Preconditions (from cpp/src/link_prediction/overlap_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, overlap_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - graph.segment_offsets MUST have a value
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected). MUST have segment_offsets.
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void overlap_similarity_seg(const graph32_t& graph,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            float* similarity_scores);

/**
 * Overlap Similarity - Float weights.
 *
 * Computes overlap similarity coefficient for specified vertex pairs.
 * Overlap(u,v) = |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)
 *
 * Preconditions (from cpp/src/link_prediction/overlap_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, overlap_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected).
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void overlap_similarity(const graph32_t& graph,
                        const float* edge_weights,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        float* similarity_scores);

/**
 * Overlap Similarity - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes overlap similarity coefficient for specified vertex pairs.
 * Overlap(u,v) = |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)
 *
 * Preconditions (from cpp/src/link_prediction/overlap_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, overlap_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - graph.segment_offsets MUST have a value
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void overlap_similarity_seg(const graph32_t& graph,
                            const float* edge_weights,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            float* similarity_scores);

/**
 * Overlap Similarity - Double weights.
 *
 * Computes overlap similarity coefficient for specified vertex pairs.
 * Overlap(u,v) = |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)
 *
 * Preconditions (from cpp/src/link_prediction/overlap_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, overlap_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected).
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void overlap_similarity(const graph32_t& graph,
                        const double* edge_weights,
                        const int32_t* vertex_pairs_first,
                        const int32_t* vertex_pairs_second,
                        std::size_t num_pairs,
                        double* similarity_scores);

/**
 * Overlap Similarity - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes overlap similarity coefficient for specified vertex pairs.
 * Overlap(u,v) = |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)
 *
 * Preconditions (from cpp/src/link_prediction/overlap_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, overlap_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - graph.segment_offsets MUST have a value
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void overlap_similarity_seg(const graph32_t& graph,
                            const double* edge_weights,
                            const int32_t* vertex_pairs_first,
                            const int32_t* vertex_pairs_second,
                            std::size_t num_pairs,
                            double* similarity_scores);

// =============================================================================
// Overlap All-Pairs Similarity
// =============================================================================

/**
 * Overlap All-Pairs Similarity - Unweighted.
 *
 * Computes overlap similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Overlap(u,v) = |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/overlap_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-214):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, overlap_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t overlap_all_pairs_similarity(const graph32_t& graph,
                                                       const int32_t* vertices,
                                                       std::size_t num_vertices,
                                                       std::optional<std::size_t> topk);

/**
 * Overlap All-Pairs Similarity - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes overlap similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Overlap(u,v) = |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/overlap_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-214):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, overlap_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - graph.segment_offsets MUST have a value
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t overlap_all_pairs_similarity_seg(const graph32_t& graph,
                                                           const int32_t* vertices,
                                                           std::size_t num_vertices,
                                                           std::optional<std::size_t> topk);

/**
 * Overlap All-Pairs Similarity - Float weights.
 *
 * Computes overlap similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Overlap(u,v) = |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/overlap_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, overlap_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected).
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t overlap_all_pairs_similarity(const graph32_t& graph,
                                                       const float* edge_weights,
                                                       const int32_t* vertices,
                                                       std::size_t num_vertices,
                                                       std::optional<std::size_t> topk);

/**
 * Overlap All-Pairs Similarity - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes overlap similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Overlap(u,v) = |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/overlap_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, overlap_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
 *   - graph.segment_offsets MUST have a value
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t overlap_all_pairs_similarity_seg(const graph32_t& graph,
                                                           const float* edge_weights,
                                                           const int32_t* vertices,
                                                           std::size_t num_vertices,
                                                           std::optional<std::size_t> topk);

/**
 * Overlap All-Pairs Similarity - Double weights.
 *
 * Computes overlap similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Overlap(u,v) = |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/overlap_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, overlap_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
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
 * similarity_result_double_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   double* scores      - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected).
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_double_t overlap_all_pairs_similarity(const graph32_t& graph,
                                                        const double* edge_weights,
                                                        const int32_t* vertices,
                                                        std::size_t num_vertices,
                                                        std::optional<std::size_t> topk);

/**
 * Overlap All-Pairs Similarity - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes overlap similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Overlap(u,v) = |N(u) ∩ N(v)| / min(|N(u)|, |N(v)|)
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/overlap_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, overlap_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
 *   - graph.segment_offsets MUST have a value
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
 * similarity_result_double_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   double* scores      - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_double_t overlap_all_pairs_similarity_seg(const graph32_t& graph,
                                                            const double* edge_weights,
                                                            const int32_t* vertices,
                                                            std::size_t num_vertices,
                                                            std::optional<std::size_t> topk);

// =============================================================================
// Sorensen Similarity
// =============================================================================

/**
 * Sorensen Similarity - Unweighted.
 *
 * Computes Sorensen (Dice) similarity coefficient for specified vertex pairs.
 * Sorensen(u,v) = 2 * |N(u) ∩ N(v)| / (|N(u)| + |N(v)|)
 *
 * Preconditions (from cpp/src/link_prediction/sorensen_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, sorensen_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void sorensen_similarity(const graph32_t& graph,
                         const int32_t* vertex_pairs_first,
                         const int32_t* vertex_pairs_second,
                         std::size_t num_pairs,
                         float* similarity_scores);

/**
 * Sorensen Similarity - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes Sorensen (Dice) similarity coefficient for specified vertex pairs.
 * Sorensen(u,v) = 2 * |N(u) ∩ N(v)| / (|N(u)| + |N(v)|)
 *
 * Preconditions (from cpp/src/link_prediction/sorensen_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, sorensen_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - graph.segment_offsets MUST have a value
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected). MUST have segment_offsets.
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void sorensen_similarity_seg(const graph32_t& graph,
                             const int32_t* vertex_pairs_first,
                             const int32_t* vertex_pairs_second,
                             std::size_t num_pairs,
                             float* similarity_scores);

/**
 * Sorensen Similarity - Float weights.
 *
 * Computes Sorensen (Dice) similarity coefficient for specified vertex pairs.
 * Sorensen(u,v) = 2 * |N(u) ∩ N(v)| / (|N(u)| + |N(v)|)
 *
 * Preconditions (from cpp/src/link_prediction/sorensen_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, sorensen_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected).
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void sorensen_similarity(const graph32_t& graph,
                         const float* edge_weights,
                         const int32_t* vertex_pairs_first,
                         const int32_t* vertex_pairs_second,
                         std::size_t num_pairs,
                         float* similarity_scores);

/**
 * Sorensen Similarity - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes Sorensen (Dice) similarity coefficient for specified vertex pairs.
 * Sorensen(u,v) = 2 * |N(u) ∩ N(v)| / (|N(u)| + |N(v)|)
 *
 * Preconditions (from cpp/src/link_prediction/sorensen_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, sorensen_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - graph.segment_offsets MUST have a value
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void sorensen_similarity_seg(const graph32_t& graph,
                             const float* edge_weights,
                             const int32_t* vertex_pairs_first,
                             const int32_t* vertex_pairs_second,
                             std::size_t num_pairs,
                             float* similarity_scores);

/**
 * Sorensen Similarity - Double weights.
 *
 * Computes Sorensen (Dice) similarity coefficient for specified vertex pairs.
 * Sorensen(u,v) = 2 * |N(u) ∩ N(v)| / (|N(u)| + |N(v)|)
 *
 * Preconditions (from cpp/src/link_prediction/sorensen_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, sorensen_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected).
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void sorensen_similarity(const graph32_t& graph,
                         const double* edge_weights,
                         const int32_t* vertex_pairs_first,
                         const int32_t* vertex_pairs_second,
                         std::size_t num_pairs,
                         double* similarity_scores);

/**
 * Sorensen Similarity - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes Sorensen (Dice) similarity coefficient for specified vertex pairs.
 * Sorensen(u,v) = 2 * |N(u) ∩ N(v)| / (|N(u)| + |N(v)|)
 *
 * Preconditions (from cpp/src/link_prediction/sorensen_impl.cuh:39
 *                + cpp/src/link_prediction/similarity_impl.cuh:45-48):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, sorensen_impl.cuh:39]
 *   - vertex_pairs_first.size() MUST equal vertex_pairs_second.size()
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:45-46]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:47-48]
 *   - graph.segment_offsets MUST have a value
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
 * @param graph                 Input graph. MUST be CSR (is_csc=false) and symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights          [in] Edge weights array (size = num_edges).
 * @param vertex_pairs_first    [in] First vertices of pairs (size = num_pairs).
 * @param vertex_pairs_second   [in] Second vertices of pairs (size = num_pairs).
 * @param num_pairs             Number of vertex pairs to compute.
 * @param similarity_scores     [out] Pre-allocated array of size num_pairs for results.
 */
void sorensen_similarity_seg(const graph32_t& graph,
                             const double* edge_weights,
                             const int32_t* vertex_pairs_first,
                             const int32_t* vertex_pairs_second,
                             std::size_t num_pairs,
                             double* similarity_scores);

// =============================================================================
// Sorensen All-Pairs Similarity
// =============================================================================

/**
 * Sorensen All-Pairs Similarity - Unweighted.
 *
 * Computes Sorensen similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Sorensen(u,v) = 2 * |N(u) ∩ N(v)| / (|N(u)| + |N(v)|)
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/sorensen_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-214):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, sorensen_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t sorensen_all_pairs_similarity(const graph32_t& graph,
                                                        const int32_t* vertices,
                                                        std::size_t num_vertices,
                                                        std::optional<std::size_t> topk);

/**
 * Sorensen All-Pairs Similarity - Unweighted - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes Sorensen similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Sorensen(u,v) = 2 * |N(u) ∩ N(v)| / (|N(u)| + |N(v)|)
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/sorensen_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-214):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, sorensen_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - graph.segment_offsets MUST have a value
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t sorensen_all_pairs_similarity_seg(const graph32_t& graph,
                                                            const int32_t* vertices,
                                                            std::size_t num_vertices,
                                                            std::optional<std::size_t> topk);

/**
 * Sorensen All-Pairs Similarity - Float weights.
 *
 * Computes Sorensen similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Sorensen(u,v) = 2 * |N(u) ∩ N(v)| / (|N(u)| + |N(v)|)
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/sorensen_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, sorensen_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected).
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t sorensen_all_pairs_similarity(const graph32_t& graph,
                                                        const float* edge_weights,
                                                        const int32_t* vertices,
                                                        std::size_t num_vertices,
                                                        std::optional<std::size_t> topk);

/**
 * Sorensen All-Pairs Similarity - Float weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes Sorensen similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Sorensen(u,v) = 2 * |N(u) ∩ N(v)| / (|N(u)| + |N(v)|)
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/sorensen_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, sorensen_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
 *   - graph.segment_offsets MUST have a value
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
 * similarity_result_float_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   float* scores       - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_float_t sorensen_all_pairs_similarity_seg(const graph32_t& graph,
                                                            const float* edge_weights,
                                                            const int32_t* vertices,
                                                            std::size_t num_vertices,
                                                            std::optional<std::size_t> topk);

/**
 * Sorensen All-Pairs Similarity - Double weights.
 *
 * Computes Sorensen similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Sorensen(u,v) = 2 * |N(u) ∩ N(v)| / (|N(u)| + |N(v)|)
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/sorensen_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.segment_offsets does NOT have a value (is std::nullopt)  [integration layer calls _seg variant when segment_offsets is present]
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, sorensen_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Adjacency lists MUST be sorted (cuGraph bug: internal set_intersection assumes sorted neighbors)
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
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
 * similarity_result_double_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   double* scores      - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected).
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_double_t sorensen_all_pairs_similarity(const graph32_t& graph,
                                                         const double* edge_weights,
                                                         const int32_t* vertices,
                                                         std::size_t num_vertices,
                                                         std::optional<std::size_t> topk);

/**
 * Sorensen All-Pairs Similarity - Double weights - Precomputed segments variant.
 *
 * Note that segment_offsets have been precomputed, and are provided within the
 * input, and the vertices of the graph are sorted in descending order by degrees.
 *
 * Computes Sorensen similarity for all 2-hop vertex pairs (or a subset of vertices).
 * Sorensen(u,v) = 2 * |N(u) ∩ N(v)| / (|N(u)| + |N(v)|)
 * where N(v) is the set of neighbors of vertex v.
 *
 * A "2-hop vertex pair" (u,v) is a pair of distinct vertices that share at least
 * one common neighbor, i.e., N(u) ∩ N(v) is non-empty. Self-pairs are excluded.
 * Pairs are directed: for each seed vertex u, all 2-hop neighbors v (v != u)
 * produce a pair (u, v). On a symmetric graph with all vertices as seeds,
 * both (u, v) and (v, u) will appear.
 *
 * Output ordering: Without topk, pairs are returned in unspecified order.
 * When topk has a value and topk < total pairs, the output is sorted by score
 * descending (ties broken arbitrarily) and only the top-k pairs are returned.
 *
 * Preconditions (from cpp/src/link_prediction/sorensen_impl.cuh:61
 *                + cpp/src/link_prediction/similarity_impl.cuh:213-219):
 *   - graph.edge_mask MUST be nullptr (no edge mask applied)
 *       [CUGRAPH_EXPECTS, sorensen_impl.cuh:61]
 *   - Graph MUST be symmetric (undirected)
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:213-214]
 *   - Weighted graphs MUST NOT be multigraphs
 *       [CUGRAPH_EXPECTS, similarity_impl.cuh:218-219]
 *   - graph.segment_offsets MUST have a value
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
 * similarity_result_double_t struct (return type):
 *   int32_t* first      - Device pointer: first vertex of each pair.
 *   int32_t* second     - Device pointer: second vertex of each pair.
 *   double* scores      - Device pointer: similarity scores.
 *   std::size_t count   - Number of pairs returned.
 *   Caller MUST free first, second, scores with cudaFree after use.
 *
 * @param graph           Input graph. MUST be symmetric (undirected). MUST have segment_offsets.
 * @param edge_weights    [in] Edge weights array (size = num_edges).
 * @param vertices        [in] Subset of vertices to compute for, or nullptr for all vertices.
 * @param num_vertices    Number of vertices in subset (ignored if vertices is nullptr).
 * @param topk            Maximum pairs to return, or std::nullopt for unlimited.
 * @return                Result struct containing device arrays and count.
 *                        Caller MUST free the device pointers with cudaFree.
 */
similarity_result_double_t sorensen_all_pairs_similarity_seg(const graph32_t& graph,
                                                             const double* edge_weights,
                                                             const int32_t* vertices,
                                                             std::size_t num_vertices,
                                                             std::optional<std::size_t> topk);

}  // namespace aai
