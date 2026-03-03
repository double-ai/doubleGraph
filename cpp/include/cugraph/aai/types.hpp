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
 * AAI Result Types and Common Definitions
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace aai {

// =============================================================================
// AAI Implementation Notes
// =============================================================================
//
// CUDA Error Handling: AAI glue code intentionally omits CUDA_CHECK macros for
// cudaDeviceSynchronize(), cudaMemcpyAsync(), and cudaFree() in performance-
// critical paths. CUDA errors will be caught by the next synchronizing operation
// or reported via CUDA error state queries. For debugging, enable CUDA_LAUNCH_BLOCKING=1.
//
// Stream Usage: AAI functions use the default CUDA stream. Callers must sync their
// streams (via handle.sync_stream()) before calling AAI to ensure data is ready.

// =============================================================================
// Result Types for Variable-Sized Outputs
// =============================================================================

/**
 * Result type for all-pairs similarity functions returning float scores.
 *
 * All pointers are device memory allocated by AAI (via cudaMalloc).
 * Caller MUST free these pointers with cudaFree after use.
 */
struct similarity_result_float_t {
  int32_t* first;      // Device pointer: first vertex of each pair
  int32_t* second;     // Device pointer: second vertex of each pair
  float* scores;       // Device pointer: similarity scores
  std::size_t count;   // Number of pairs
};

/**
 * Result type for all-pairs similarity functions returning double scores.
 *
 * All pointers are device memory allocated by AAI (via cudaMalloc).
 * Caller MUST free these pointers with cudaFree after use.
 */
struct similarity_result_double_t {
  int32_t* first;      // Device pointer: first vertex of each pair
  int32_t* second;     // Device pointer: second vertex of each pair
  double* scores;      // Device pointer: similarity scores
  std::size_t count;   // Number of pairs
};

/**
 * Result type for ECG returning float modularity.
 */
struct ecg_result_float_t {
  std::size_t level_count;  // Number of hierarchical levels
  float modularity;         // Final modularity score
};

/**
 * Result type for ECG returning double modularity.
 */
struct ecg_result_double_t {
  std::size_t level_count;  // Number of hierarchical levels
  double modularity;        // Final modularity score
};

/**
 * Result type for Louvain returning float modularity.
 */
struct louvain_result_float_t {
  std::size_t level_count;  // Number of hierarchical levels
  float modularity;         // Final modularity score
};

/**
 * Result type for Louvain returning double modularity.
 */
struct louvain_result_double_t {
  std::size_t level_count;  // Number of hierarchical levels
  double modularity;        // Final modularity score
};

/**
 * Result type for Leiden returning float modularity.
 */
struct leiden_result_float_t {
  std::size_t level_count;  // Number of hierarchical levels
  float modularity;         // Final modularity score
};

/**
 * Result type for Leiden returning double modularity.
 */
struct leiden_result_double_t {
  std::size_t level_count;  // Number of hierarchical levels
  double modularity;        // Final modularity score
};

/**
 * Result type for k_hop_nbrs.
 *
 * All pointers are device memory allocated by AAI (via cudaMalloc).
 * Caller MUST free these pointers with cudaFree after use.
 */
struct k_hop_nbrs_result_t {
  std::size_t* offsets;      // Device pointer: offsets array (size = num_start_vertices + 1)
  int32_t* neighbors;        // Device pointer: neighbor vertex IDs
  std::size_t num_offsets;   // Size of offsets array (num_start_vertices + 1)
  std::size_t num_neighbors; // Number of neighbors found
};

/**
 * Result type for extract_ego (unweighted).
 *
 * All pointers are device memory allocated by AAI (via cudaMalloc).
 * Caller MUST free these pointers with cudaFree after use.
 */
struct extract_ego_result_t {
  int32_t* edge_srcs;        // Device pointer: source vertices of edges
  int32_t* edge_dsts;        // Device pointer: destination vertices of edges
  std::size_t* offsets;      // Device pointer: offsets array (size = n_sources + 1)
  std::size_t num_edges;     // Total number of edges across all ego networks
  std::size_t num_offsets;   // Size of offsets array (n_sources + 1)
};

/**
 * Result type for extract_ego (weighted, float).
 *
 * All pointers are device memory allocated by AAI (via cudaMalloc).
 * Caller MUST free these pointers with cudaFree after use.
 */
struct extract_ego_weighted_result_float_t {
  int32_t* edge_srcs;        // Device pointer: source vertices of edges
  int32_t* edge_dsts;        // Device pointer: destination vertices of edges
  float* edge_weights;       // Device pointer: weights of edges
  std::size_t* offsets;      // Device pointer: offsets array (size = n_sources + 1)
  std::size_t num_edges;     // Total number of edges across all ego networks
  std::size_t num_offsets;   // Size of offsets array (n_sources + 1)
};

/**
 * Result type for extract_ego (weighted, double).
 *
 * All pointers are device memory allocated by AAI (via cudaMalloc).
 * Caller MUST free these pointers with cudaFree after use.
 */
struct extract_ego_weighted_result_double_t {
  int32_t* edge_srcs;        // Device pointer: source vertices of edges
  int32_t* edge_dsts;        // Device pointer: destination vertices of edges
  double* edge_weights;      // Device pointer: weights of edges
  std::size_t* offsets;      // Device pointer: offsets array (size = n_sources + 1)
  std::size_t num_edges;     // Total number of edges across all ego networks
  std::size_t num_offsets;   // Size of offsets array (n_sources + 1)
};

/**
 * Result type for k_truss (unweighted).
 *
 * All pointers are device memory allocated by AAI (via cudaMalloc).
 * Caller MUST free these pointers with cudaFree after use.
 */
struct k_truss_result_t {
  int32_t* edge_srcs;        // Device pointer: source vertices of k-truss edges
  int32_t* edge_dsts;        // Device pointer: destination vertices of k-truss edges
  std::size_t num_edges;     // Number of edges in the k-truss subgraph
};

/**
 * Result type for iterative eigenvector centrality algorithms.
 */
struct eigenvector_centrality_result_t {
  std::size_t iterations;  // Number of iterations performed
  bool converged;          // True if converged within epsilon tolerance
};

/**
 * Result type for iterative Katz centrality algorithms.
 */
struct katz_centrality_result_t {
  std::size_t iterations;  // Number of iterations performed
  bool converged;          // True if converged within epsilon tolerance
};

}  // namespace aai
