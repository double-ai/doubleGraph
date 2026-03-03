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
#include "community/k_truss_impl.cuh"

#include <cugraph/utilities/error.hpp>

#include <cuda_runtime.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <rmm/exec_policy.hpp>

#include <limits>
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

// =============================================================================
// Template specializations that route to AAI implementation
// =============================================================================
//
// NOTE ON WEIGHTS: K-truss is a purely topological algorithm - it finds the maximal
// subgraph where each edge participates in at least (k-2) triangles. Triangle
// membership is determined by graph structure, not edge weights. Edge weights have
// no effect on which edges are selected for the k-truss subgraph.
//
// Therefore, the AAI layer only implements an unweighted k_truss variant. When the
// caller provides edge weights, we:
//   1. Call the unweighted AAI k_truss to get the (src, dst) edge list
//   2. Look up the corresponding weights from the original graph's CSR structure
//
// This avoids duplicating algorithm code for float/double weight types when the
// weights don't affect the algorithm's behavior.
// =============================================================================

}  // namespace cugraph

#ifdef AAI_ROUTE_K_TRUSS

#include <cugraph/aai/algorithms.hpp>

namespace cugraph {

// Float specialization (weighted and unweighted)
template <>
std::tuple<rmm::device_uvector<int32_t>,
           rmm::device_uvector<int32_t>,
           std::optional<rmm::device_uvector<float>>>
k_truss<int32_t, int32_t, float, false>(
        raft::handle_t const& handle,
        graph_view_t<int32_t, int32_t, false, false> const& graph_view,
        std::optional<edge_property_view_t<int32_t, float const*>> edge_weight_view,
        int32_t k,
        bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from k_truss_impl.cuh, lines 172-175)
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "K-truss requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "Invalid input arguments: K-truss currently supports undirected graphs only.");
  CUGRAPH_EXPECTS(!graph_view.is_multigraph(),
                  "Invalid input arguments: K-truss currently does not support multi-graphs.");

  // Sync stream before AAI call (AAI uses default stream)
  handle.sync_stream();

  // =========================================================================
  // ROUTE TO AAI IMPLEMENTATION (always unweighted)
  // =========================================================================

  aai::k_truss_result_t result;

  // 4-way dispatch: mask x segment
  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::k_truss_seg_mask(compact_graph, k);
    } else {
      result = aai::k_truss_mask(compact_graph, k);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::k_truss_seg(compact_graph, k);
    } else {
      result = aai::k_truss(compact_graph, k);
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  // Wrap AAI result in RMM vectors
  rmm::device_uvector<int32_t> srcs(result.num_edges, handle.get_stream());
  rmm::device_uvector<int32_t> dsts(result.num_edges, handle.get_stream());

  cudaMemcpyAsync(srcs.data(), result.edge_srcs,
                  result.num_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                  handle.get_stream());
  cudaMemcpyAsync(dsts.data(), result.edge_dsts,
                  result.num_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                  handle.get_stream());

  // Sync stream before freeing AAI-allocated memory to ensure copies complete
  handle.sync_stream();

  // Free AAI-allocated memory
  cudaFree(result.edge_srcs);
  cudaFree(result.edge_dsts);

  // =========================================================================
  // Handle weights in integration layer (if provided)
  // =========================================================================

  if (edge_weight_view.has_value()) {
    const float* input_weights = edge_weight_view->value_firsts()[0];
    rmm::device_uvector<float> out_weights(result.num_edges, handle.get_stream());

    // Look up weights for each returned edge using CSR structure
    lookup_edge_weights(compact_graph.offsets, compact_graph.indices, input_weights,
                        srcs.data(), dsts.data(), out_weights.data(),
                        result.num_edges, handle.get_stream());

    return std::make_tuple(std::move(srcs), std::move(dsts),
                           std::make_optional(std::move(out_weights)));
  } else {
    return std::make_tuple(std::move(srcs), std::move(dsts), std::nullopt);
  }
}

// Double specialization (weighted and unweighted)
template <>
std::tuple<rmm::device_uvector<int32_t>,
           rmm::device_uvector<int32_t>,
           std::optional<rmm::device_uvector<double>>>
k_truss<int32_t, int32_t, double, false>(
        raft::handle_t const& handle,
        graph_view_t<int32_t, int32_t, false, false> const& graph_view,
        std::optional<edge_property_view_t<int32_t, double const*>> edge_weight_view,
        int32_t k,
        bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (from k_truss_impl.cuh, lines 172-175)
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "K-truss requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(graph_view.is_symmetric(),
                  "Invalid input arguments: K-truss currently supports undirected graphs only.");
  CUGRAPH_EXPECTS(!graph_view.is_multigraph(),
                  "Invalid input arguments: K-truss currently does not support multi-graphs.");

  // Sync stream before AAI call (AAI uses default stream)
  handle.sync_stream();

  // =========================================================================
  // ROUTE TO AAI IMPLEMENTATION (always unweighted)
  // =========================================================================

  aai::k_truss_result_t result;

  // 4-way dispatch: mask x segment
  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::k_truss_seg_mask(compact_graph, k);
    } else {
      result = aai::k_truss_mask(compact_graph, k);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      result = aai::k_truss_seg(compact_graph, k);
    } else {
      result = aai::k_truss(compact_graph, k);
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }

  // Wrap AAI result in RMM vectors
  rmm::device_uvector<int32_t> srcs(result.num_edges, handle.get_stream());
  rmm::device_uvector<int32_t> dsts(result.num_edges, handle.get_stream());

  cudaMemcpyAsync(srcs.data(), result.edge_srcs,
                  result.num_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                  handle.get_stream());
  cudaMemcpyAsync(dsts.data(), result.edge_dsts,
                  result.num_edges * sizeof(int32_t), cudaMemcpyDeviceToDevice,
                  handle.get_stream());

  // Sync stream before freeing AAI-allocated memory to ensure copies complete
  handle.sync_stream();

  // Free AAI-allocated memory
  cudaFree(result.edge_srcs);
  cudaFree(result.edge_dsts);

  // =========================================================================
  // Handle weights in integration layer (if provided)
  // =========================================================================

  if (edge_weight_view.has_value()) {
    const double* input_weights = edge_weight_view->value_firsts()[0];
    rmm::device_uvector<double> out_weights(result.num_edges, handle.get_stream());

    // Look up weights for each returned edge using CSR structure
    lookup_edge_weights(compact_graph.offsets, compact_graph.indices, input_weights,
                        srcs.data(), dsts.data(), out_weights.data(),
                        result.num_edges, handle.get_stream());

    return std::make_tuple(std::move(srcs), std::move(dsts),
                           std::make_optional(std::move(out_weights)));
  } else {
    return std::make_tuple(std::move(srcs), std::move(dsts), std::nullopt);
  }
}

}  // namespace cugraph

#else  // !AAI_ROUTE_K_TRUSS

namespace cugraph {
template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<float>>> k_truss<int32_t, int32_t, float, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, std::optional<edge_property_view_t<int32_t, float const*>>, int32_t, bool);
template std::tuple<rmm::device_uvector<int32_t>, rmm::device_uvector<int32_t>, std::optional<rmm::device_uvector<double>>> k_truss<int32_t, int32_t, double, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, std::optional<edge_property_view_t<int32_t, double const*>>, int32_t, bool);
}  // namespace cugraph

#endif  // AAI_ROUTE_K_TRUSS
