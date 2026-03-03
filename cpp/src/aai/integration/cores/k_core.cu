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
// Include original implementation for non-specialized cases
#include "cores/k_core_impl.cuh"

#include <cugraph/utilities/error.hpp>

#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <rmm/exec_policy.hpp>

#include <stdexcept>

namespace cugraph {

namespace {

// Look up edge weights from CSR structure given (src, dst) pairs.
// For each output edge, finds the edge index in the CSR using binary search
// and copies the corresponding weight.
//
// WARNING: This could be SLOW. Check during benchmarking if this is ever an issue.
//
template <typename weight_t>
void lookup_edge_weights(const int32_t* offsets,
                         const int32_t* indices,
                         const weight_t* input_weights,
                         const int32_t* edge_srcs,
                         const int32_t* edge_dsts,
                         weight_t* output_weights,
                         std::size_t num_edges,
                         rmm::cuda_stream_view stream)
{
  if (num_edges == 0) return;

  thrust::for_each(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<std::size_t>(0),
    thrust::make_counting_iterator<std::size_t>(num_edges),
    [offsets, indices, input_weights, edge_srcs, edge_dsts, output_weights] __device__(
      std::size_t i) {
      int32_t src = edge_srcs[i];
      int32_t dst = edge_dsts[i];

      // Binary search for dst in the neighbor list of src
      const int32_t* nbr_start = indices + offsets[src];
      const int32_t* nbr_end   = indices + offsets[src + 1];
      const int32_t* pos       = thrust::lower_bound(thrust::seq, nbr_start, nbr_end, dst);

      // Copy the weight at this edge index
      output_weights[i] = input_weights[pos - indices];
    });
}

}  // namespace

}  // namespace cugraph

#ifdef AAI_ROUTE_K_CORE

#include <cugraph/aai/algorithms.hpp>

namespace cugraph {

// =============================================================================
// Specialized k_core to route to AAI
// =============================================================================
//
// NOTE ON WEIGHTS: K-core is a purely topological algorithm - it finds the maximal
// subgraph where all vertices have degree >= k. The "degree" is a count of edges,
// not a sum of weights. Edge weights have no effect on which edges are selected
// for the k-core subgraph.
//
// Therefore, the AAI layer only implements an unweighted k_core variant. When the
// caller provides edge weights, we:
//   1. Call the unweighted AAI k_core to get the (src, dst) edge list
//   2. Look up the corresponding weights from the original graph's CSR structure
//
// This avoids duplicating algorithm code for float/double weight types when the
// weights don't affect the algorithm's behavior.
// =============================================================================

// Float specialization
template <>
std::tuple<rmm::device_uvector<int32_t>,
           rmm::device_uvector<int32_t>,
           std::optional<rmm::device_uvector<float>>>
k_core<int32_t, int32_t, float, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
  size_t k,
  std::optional<k_core_degree_type_t> degree_type,
  std::optional<raft::device_span<int32_t const>> core_numbers,
  bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from k_core_impl.cuh:32-34, core_number_impl.cuh:70-73)
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "k_core requires CSR format (is_csc=false)");
  if (!core_numbers) {
    CUGRAPH_EXPECTS(degree_type.has_value(),
                    "If core_numbers is not specified then degree_type must be specified");
  }
  // The following checks are from core_number which k_core calls internally
  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "Invalid input argument: core_number currently supports only undirected graphs.");
  CUGRAPH_EXPECTS(!graph_view.is_multigraph(),
                  "Invalid input argument: core_number currently does not support multi-graphs.");

  // Determine degree_type int value
  // -1 means "use core_numbers", 0=in, 1=out, 2=inout
  int degree_type_int = -1;
  if (degree_type.has_value()) {
    degree_type_int = static_cast<int>(degree_type.value());
  }

  // Get core_numbers pointer (can be nullptr)
  const int32_t* core_numbers_ptr = nullptr;
  if (core_numbers.has_value()) {
    core_numbers_ptr = core_numbers->data();
  }

  // Allocate output arrays - use num_edges as max since k-core can't have more edges
  std::size_t max_edges = compact_graph.number_of_edges;
  rmm::device_uvector<int32_t> edge_srcs(max_edges, handle.get_stream());
  rmm::device_uvector<int32_t> edge_dsts(max_edges, handle.get_stream());

  std::size_t num_k_core_edges;

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  // =========================================================================
  // ROUTE TO AAI IMPLEMENTATION (always unweighted)
  // =========================================================================

  // 4-way dispatch: mask x segment
  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      num_k_core_edges = aai::k_core_seg_mask(compact_graph, k, degree_type_int, core_numbers_ptr,
                                              edge_srcs.data(), edge_dsts.data(), max_edges);
    } else {
      num_k_core_edges = aai::k_core_mask(compact_graph, k, degree_type_int, core_numbers_ptr,
                                          edge_srcs.data(), edge_dsts.data(), max_edges);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      num_k_core_edges = aai::k_core_seg(compact_graph, k, degree_type_int, core_numbers_ptr,
                                         edge_srcs.data(), edge_dsts.data(), max_edges);
    } else {
      num_k_core_edges = aai::k_core(compact_graph, k, degree_type_int, core_numbers_ptr,
                                     edge_srcs.data(), edge_dsts.data(), max_edges);
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  // Resize to actual size
  edge_srcs.resize(num_k_core_edges, handle.get_stream());
  edge_dsts.resize(num_k_core_edges, handle.get_stream());

  // =========================================================================
  // Handle weights in integration layer (if provided)
  // =========================================================================

  if (edge_weight_view.has_value()) {
    const float* input_weights = edge_weight_view->value_firsts()[0];
    rmm::device_uvector<float> out_weights(num_k_core_edges, handle.get_stream());

    // Look up weights for each returned edge using CSR structure
    lookup_edge_weights(compact_graph.offsets, compact_graph.indices, input_weights,
                        edge_srcs.data(), edge_dsts.data(), out_weights.data(),
                        num_k_core_edges, handle.get_stream());

    return std::make_tuple(std::move(edge_srcs), std::move(edge_dsts),
                           std::make_optional(std::move(out_weights)));
  } else {
    return std::make_tuple(std::move(edge_srcs), std::move(edge_dsts), std::nullopt);
  }
}

// Double specialization
template <>
std::tuple<rmm::device_uvector<int32_t>,
           rmm::device_uvector<int32_t>,
           std::optional<rmm::device_uvector<double>>>
k_core<int32_t, int32_t, double, false>(
  raft::handle_t const& handle,
  graph_view_t<int32_t, int32_t, false, false> const& graph_view,
  std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
  size_t k,
  std::optional<k_core_degree_type_t> degree_type,
  std::optional<raft::device_span<int32_t const>> core_numbers,
  bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from k_core_impl.cuh:32-34, core_number_impl.cuh:70-73)
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "k_core requires CSR format (is_csc=false)");
  if (!core_numbers) {
    CUGRAPH_EXPECTS(degree_type.has_value(),
                    "If core_numbers is not specified then degree_type must be specified");
  }
  // The following checks are from core_number which k_core calls internally
  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "Invalid input argument: core_number currently supports only undirected graphs.");
  CUGRAPH_EXPECTS(!graph_view.is_multigraph(),
                  "Invalid input argument: core_number currently does not support multi-graphs.");

  // Determine degree_type int value
  int degree_type_int = -1;
  if (degree_type.has_value()) {
    degree_type_int = static_cast<int>(degree_type.value());
  }

  // Get core_numbers pointer (can be nullptr)
  const int32_t* core_numbers_ptr = nullptr;
  if (core_numbers.has_value()) {
    core_numbers_ptr = core_numbers->data();
  }

  // Allocate output arrays - use num_edges as max since k-core can't have more edges
  std::size_t max_edges = compact_graph.number_of_edges;
  rmm::device_uvector<int32_t> edge_srcs(max_edges, handle.get_stream());
  rmm::device_uvector<int32_t> edge_dsts(max_edges, handle.get_stream());

  std::size_t num_k_core_edges;

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  // =========================================================================
  // ROUTE TO AAI IMPLEMENTATION (always unweighted)
  // =========================================================================

  // 4-way dispatch: mask x segment
  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      num_k_core_edges = aai::k_core_seg_mask(compact_graph, k, degree_type_int, core_numbers_ptr,
                                              edge_srcs.data(), edge_dsts.data(), max_edges);
    } else {
      num_k_core_edges = aai::k_core_mask(compact_graph, k, degree_type_int, core_numbers_ptr,
                                          edge_srcs.data(), edge_dsts.data(), max_edges);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      num_k_core_edges = aai::k_core_seg(compact_graph, k, degree_type_int, core_numbers_ptr,
                                         edge_srcs.data(), edge_dsts.data(), max_edges);
    } else {
      num_k_core_edges = aai::k_core(compact_graph, k, degree_type_int, core_numbers_ptr,
                                     edge_srcs.data(), edge_dsts.data(), max_edges);
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  // Resize to actual size
  edge_srcs.resize(num_k_core_edges, handle.get_stream());
  edge_dsts.resize(num_k_core_edges, handle.get_stream());

  // =========================================================================
  // Handle weights in integration layer (if provided)
  // =========================================================================

  if (edge_weight_view.has_value()) {
    const double* input_weights = edge_weight_view->value_firsts()[0];
    rmm::device_uvector<double> out_weights(num_k_core_edges, handle.get_stream());

    // Look up weights for each returned edge using CSR structure
    lookup_edge_weights(compact_graph.offsets, compact_graph.indices, input_weights,
                        edge_srcs.data(), edge_dsts.data(), out_weights.data(),
                        num_k_core_edges, handle.get_stream());

    return std::make_tuple(std::move(edge_srcs), std::move(edge_dsts),
                           std::make_optional(std::move(out_weights)));
  } else {
    return std::make_tuple(std::move(edge_srcs), std::move(edge_dsts), std::nullopt);
  }
}

}  // namespace cugraph

#else

namespace cugraph {
template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<float>>> k_core<int32_t, int32_t, float, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, std::optional<edge_property_view_t<int32_t, float const*>>, size_t, std::optional<k_core_degree_type_t>, std::optional<raft::device_span<int32_t const>>, bool);
template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<double>>> k_core<int32_t, int32_t, double, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, std::optional<edge_property_view_t<int32_t, double const*>>, size_t, std::optional<k_core_degree_type_t>, std::optional<raft::device_span<int32_t const>>, bool);
}  // namespace cugraph

#endif
