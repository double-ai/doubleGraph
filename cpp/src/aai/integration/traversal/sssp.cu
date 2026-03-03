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
// Include original implementation for other instantiations
#include "traversal/sssp_impl.cuh"

#ifdef AAI_ROUTE_SSSP

#include <cugraph/aai/algorithms.hpp>
#include <cugraph/utilities/error.hpp>

#include <cuda_runtime.h>

namespace cugraph {

// =============================================================================
// Template specializations - route to AAI
// =============================================================================

// Float specialization
template <>
void sssp<int32_t, int32_t, float, false>(raft::handle_t const& handle,
                                          graph_view_t<int32_t, int32_t, false, false> const& graph_view,
                                          edge_property_view_t<int32_t, float const*> edge_weight_view,
                                          float* distances,
                                          int32_t* predecessors,
                                          int32_t source_vertex,
                                          float cutoff,
                                          bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (match original sssp_impl.cuh)
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "SSSP requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(graph_view.is_valid_vertex(source_vertex),
                  "Invalid input argument: source vertex out-of-range.");

  const float* weights = edge_weight_view.value_firsts()[0];

  // =========================================================================
  // ROUTE TO AAI IMPLEMENTATION
  // =========================================================================

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      aai::sssp_seg_mask(compact_graph, weights, source_vertex, distances, predecessors, cutoff);
    } else {
      aai::sssp_mask(compact_graph, weights, source_vertex, distances, predecessors, cutoff);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      aai::sssp_seg(compact_graph, weights, source_vertex, distances, predecessors, cutoff);
    } else {
      aai::sssp(compact_graph, weights, source_vertex, distances, predecessors, cutoff);
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }
}

// Double specialization
template <>
void sssp<int32_t, int32_t, double, false>(raft::handle_t const& handle,
                                           graph_view_t<int32_t, int32_t, false, false> const& graph_view,
                                           edge_property_view_t<int32_t, double const*> edge_weight_view,
                                           double* distances,
                                           int32_t* predecessors,
                                           int32_t source_vertex,
                                           double cutoff,
                                           bool do_expensive_check)
{
  auto compact_graph = aai::graph32_t::from_graph_view(graph_view);

  // Preconditions (match original sssp_impl.cuh)
  CUGRAPH_EXPECTS(!compact_graph.is_csc, "SSSP requires CSR format (is_csc=false)");
  CUGRAPH_EXPECTS(graph_view.is_valid_vertex(source_vertex),
                  "Invalid input argument: source vertex out-of-range.");

  const double* weights = edge_weight_view.value_firsts()[0];

  // =========================================================================
  // ROUTE TO AAI IMPLEMENTATION
  // =========================================================================

  // Sync stream before calling AAI (AAI uses default stream)
  handle.sync_stream();

  if (compact_graph.edge_mask != nullptr) {
    if (compact_graph.segment_offsets.has_value()) {
      aai::sssp_seg_mask(compact_graph, weights, source_vertex, distances, predecessors, cutoff);
    } else {
      aai::sssp_mask(compact_graph, weights, source_vertex, distances, predecessors, cutoff);
    }
  } else {
    if (compact_graph.segment_offsets.has_value()) {
      aai::sssp_seg(compact_graph, weights, source_vertex, distances, predecessors, cutoff);
    } else {
      aai::sssp(compact_graph, weights, source_vertex, distances, predecessors, cutoff);
    }
  }

  cudaDeviceSynchronize();
  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      CUGRAPH_FAIL((std::string("AAI CUDA error after cudaDeviceSynchronize: ") + cudaGetErrorString(err)).c_str());
    }
  }
}

}  // namespace cugraph

#else

namespace cugraph {
template void sssp<int32_t, int32_t, float, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, edge_property_view_t<int32_t, float const*>, float*, int32_t*, int32_t, float, bool);
template void sssp<int32_t, int32_t, double, false>(raft::handle_t const&, graph_view_t<int32_t, int32_t, false, false> const&, edge_property_view_t<int32_t, double const*>, double*, int32_t*, int32_t, double, bool);
}  // namespace cugraph

#endif
